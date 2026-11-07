"""Compare the production adapter against pinned WRF module outputs.

The default fixtures are synthetic regression diagnostics. A verified
``--capture-manifest`` selects literal inputs from a real native forecast.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tests"),str(ROOT),str(ROOT/"tools/thompson_real_column_parity")]
from woof.core.fp32_ulp import monotone_fp32_key
SCENARIOS=("gh-convective","gh-stratiform","gh-hail","gh-density-sweep","gh-warm-melt")
def metric(got,want):
    got=np.asarray(got,np.float32); want=np.asarray(want,np.float32)
    ulp=np.abs(monotone_fp32_key(got)-monotone_fp32_key(want))
    differ=got.view(np.uint32)!=want.view(np.uint32)
    scale=np.maximum(np.abs(want).astype(np.float64),1.e-30)
    differing = np.flatnonzero(differ)
    return {"words": int(got.size),
            "differing_words": [{"index": int(i), "got": float(got.flat[i]),
                "wrf": float(want.flat[i]), "got_bits": f"{got.view(np.uint32).flat[i]:08x}",
                "wrf_bits": f"{want.view(np.uint32).flat[i]:08x}"}
                for i in differing],
            "differing_zero_sign_words": int(np.count_nonzero((got == 0) & (want == 0) & differ)),
            "nonfinite": int(np.count_nonzero(~np.isfinite(got))),
            "max_ulp":int(ulp.max(initial=0)),"unequal":int(np.count_nonzero(differ)),
            "max_abs":float(np.max(np.abs(got.astype(np.float64)-want),initial=0)),
            "max_relative":float(np.max(np.abs(got.astype(np.float64)-want)/scale,initial=0)),
            "got_sha256":hashlib.sha256(got.tobytes()).hexdigest(),
            "wrf_sha256":hashlib.sha256(want.tobytes()).hexdigest()}
def run(args):
    requested_numerics = "strict" if os.environ.get("GPUWM_WRF_EXACT") == "1" else os.environ.get("GPUWM_NUMERICS", "fast")
    scenarios = SCENARIOS
    capture = None
    if args.capture_manifest is not None:
        capture = json.loads(args.capture_manifest.read_text())
        assert capture["schema"] == "gpuwm-native-hail-capture-v1"
        assert capture["public_source_sha256"] == "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802"
        assert len(capture["native_executable_sha256"]) == 64
        assert capture["native_run_command"]
        scenarios = tuple(capture["scenarios"])
        assert scenarios and all(Path(name).name == name for name in scenarios)
        for name, digest in capture["files"].items():
            assert Path(name).name == name
            assert hashlib.sha256((args.fixtures / name).read_bytes()).hexdigest() == digest, name
        for scenario in scenarios:
            for suffix in ("column", "surface", "tendencies", "metadata"):
                assert scenario + "-" + suffix + ".csv" in capture["files"]
            with (args.fixtures / (scenario + "-metadata.csv")).open() as stream:
                metadata = next(csv.DictReader(stream))
            assert int(metadata["aer_init_opt"]) in (0, 1)
            assert int(metadata["wif_input_opt"]) in (0, 1)
            assert int(metadata["itimestep"]) >= 1
            assert all(int(metadata[key]) in (0, 1)
                       for key in ("has_reqc", "has_reqi", "has_reqs"))
            assert int(metadata["diagflag"]) == 1 and int(metadata["do_radar_ref"]) == 1
    if args.backend=="host":
        strict = os.environ.get("GPUWM_WRF_EXACT") == "1"
        os.environ["GPUWM_WRF_EXACT"] = "0"
        import host_backend
        host_backend.install()
        os.environ["GPUWM_WRF_EXACT"] = "1" if strict else "0"
        if strict:
            from woof import wrf_exact
            wrf_exact.ENABLED = True
            for module in ("thompson", "thompson_aerosol_state", "thompson_aerosol_sat",
                           "thompson_aerosol_cold", "thompson_aerosol_warm",
                           "thompson_aerosol_sed", "refl", "microphysics"):
                host_backend.set_transform(module, lambda source: "#define GPUWM_WRF_EXACT 1\n" + source)
    import cupy as xp
    if args.mutation:
        from woof.core import thompson_aerosol_launch as launch
        launch.HAIL_DEFINES += (("GPUWM_HAIL_ORACLE_MUTATION", 1),)
    from test_thompson_aerosol_adapter import (
        _ColumnState,_column,_reconstruct_entry_state,_adapter_outputs,_oracle_expectation,
        _END_TO_END_SURFACE_FIELDS)
    import test_thompson_aerosol_adapter as common
    # The test-state helper imports conftest, which sets the test battery's
    # numerics to strict. Preserve this standalone replay's launch policy.
    from woof import numerics
    numerics.select(requested_numerics)
    from woof.core import constants
    def pii_of(xp, pressure):
        from woof.core.thompson_aerosol_launch import hail_aware_scope
        from woof.core.thompson_aerosol_state import launch_hail_exner
        pressure = xp.asarray(pressure)
        theta = xp.ones(pressure.shape, dtype=xp.float32)
        pii = xp.zeros(pressure.shape, dtype=xp.float32)
        temperature = xp.zeros(pressure.shape, dtype=xp.float32)
        with hail_aware_scope(True):
            launch_hail_exner(pressure, theta, pii, temperature)
        return xp.asnumpy(pii)
    common._pii_of = pii_of
    from types import SimpleNamespace
    from woof.core.microphysics_aerosol import _apply_thompson_aerosol
    results={}
    captured={}
    for scenario in scenarios:
        with (args.fixtures/(scenario+"-column.csv")).open() as f: rows=list(csv.DictReader(f))
        with (args.fixtures/(scenario+"-surface.csv")).open() as f: surface=next(csv.DictReader(f))
        before,after=rows[:len(rows)//2],rows[len(rows)//2:]
        if capture is None:
            pressure,theta,phi,reconstruction=_reconstruct_entry_state(xp,before,max_pressure_ulps=0)
        else:
            # A native capture must retain the actual pressure AND theta.
            # The synthetic fixtures instead invert their module-entry T.
            pressure = _column(before, "p_pa")
            theta = _column(before, "theta_k")
            # Native dz is an independent module input. Do not perturb it
            # to fit the forecast adapter's geopotential representation.
            native_dz = _column(before, "dz_m")
            phi = (np.concatenate(([0.], np.cumsum(native_dz, dtype=np.float64)))
                   * np.float64(constants.G)).astype(np.float32)
            pii = pii_of(xp, pressure)
            gpu_phi = xp.asarray(phi)
            gpu_z = gpu_phi / xp.float32(constants.G)
            reconstruction = {
                "pressure_perturbed_levels": 0,
                "theta_perturbed_levels": 0,
                "mode": "literal-native-module-entry",
                "forecast_preparation_metrics": {
                    "pii": metric(pii, _column(before, "pii")),
                    "temperature": metric(xp.asnumpy(xp.asarray(theta) * xp.asarray(pii)),
                                          _column(before, "temp_k")),
                    "dz": metric(xp.asnumpy(gpu_z[1:] - gpu_z[:-1]), _column(before, "dz_m")),
                },
            }
        state=_ColumnState(xp,before,surface,pressure,theta,phi)
        if capture is not None:
            # Native re_* arrays are metres. Preserve even inactive caller
            # diagnostics in the production state's micron convention.
            for field in ("effc", "effi", "effs"):
                getattr(state, field)[...] *= xp.float32(1.e6)
        state.ng=xp.asarray(_column(before,"ng_per_kg").reshape(-1,1,1))
        state.qvolg=xp.asarray(_column(before,"qb_m3_per_kg").reshape(-1,1,1))
        cfg=SimpleNamespace(mp_physics=38,no_mp_heating=0,mp_tend_lim=10.,
                            thompson_version="wrf_461",thompson_fork_snow_fall="blend")
        effective_radii_due = True  # Synthetic fixture caller sets all three.
        module_entry = None
        if capture is not None:
            with (args.fixtures / (scenario + "-metadata.csv")).open() as stream:
                metadata = next(csv.DictReader(stream))
            effective_radii_due = all(int(metadata[key]) != 0
                                     for key in ("has_reqc", "has_reqi", "has_reqs"))
            module_entry = {key: xp.asarray(_column(before, field).reshape(-1, 1, 1))
                            for key, field in (("pii", "pii"), ("temperature", "temp_k"), ("dz", "dz_m"))}
            reconstruction["input_metrics"] = {
                key: metric(xp.asnumpy(value).ravel(), _column(before, field))
                for (key, value), field in zip(module_entry.items(), ("pii", "temp_k", "dz_m"))}
        from woof.core.microphysics import microphysics_init
        microphysics_init(state, cfg)
        initial_surface_path = args.fixtures / (scenario + "-initial-surface.csv")
        if initial_surface_path.exists():
            assert capture is not None and initial_surface_path.name in capture["files"]
            with initial_surface_path.open() as stream:
                initial_surface = next(csv.DictReader(stream))
            for field, slot in _END_TO_END_SURFACE_FIELDS.items():
                state.scratch((1, 1), slot)[...] = xp.float32(initial_surface[field])
        from woof.core import thompson_aerosol_state as state_launch
        original_terminal = state_launch.launch_hail_graupel_terminal
        snapshot = {}
        def capture_tendencies(*values, **keywords):
            for field, slot in (
                ("qvten", "mp_thompson_aero_qvten"),
                ("tten", "mp_thompson_aero_tten"),
                ("qsten", "mp_thompson_aero_qsten"),
                ("qcten", "mp_thompson_aero_qcten"),
                ("qrten", "mp_thompson_aero_qrten"),
                ("nrten", "mp_thompson_aero_nrten"),
                ("qiten", "mp_thompson_aero_qiten"),
                ("niten", "mp_thompson_aero_niten"),
                ("qgten", "mp_thompson_aero_qgten"),
                ("ngten", "mp_thompson_aero_ngten"),
                ("qbten", "mp_thompson_aero_qbten"),
                ("ncten", "mp_thompson_aero_ncten"),
                ("nwfaten", "mp_thompson_aero_nwfaten"),
                ("nifaten", "mp_thompson_aero_nifaten")):
                snapshot[field] = xp.asnumpy(state.existing_scratch(slot)).ravel().copy()
            original_terminal(*values, **keywords)
        state_launch.launch_hail_graupel_terminal = capture_tendencies
        try:
            _apply_thompson_aerosol(state,cfg,float(surface["dt_s"]),refl_10cm_due=True,
                                   effective_radii_due=effective_radii_due,
                                   module_entry=module_entry)
        finally:
            state_launch.launch_hail_graupel_terminal = original_terminal
        tendency_path = args.fixtures / (scenario + "-tendencies.csv")
        if tendency_path.exists():
            with tendency_path.open() as stream:
                tendencies = list(csv.DictReader(stream))
            captured[scenario] = {key: metric(value, _column(tendencies, key))
                                  for key, value in snapshot.items()}

        from woof.core.refl import compute_refl_10cm
        same_call_refl = xp.asnumpy(state.physics.refl_10cm).copy()
        between_step_refl = compute_refl_10cm(state, cfg,
            temperature=state.existing_scratch("mp_thompson_temperature"), pressure=state.p)
        assert np.array_equal(same_call_refl, xp.asnumpy(between_step_refl)), "between-step hail reflectivity changed"
        assert state.existing_scratch("mp_thompson_graupel_mvd_max") is not None
        output=_adapter_outputs(xp,state,before)
        output.update(ng_per_kg=xp.asnumpy(state.ng).ravel(),
                      qb_m3_per_kg=xp.asnumpy(state.qvolg).ravel(),
                      theta_k=xp.asnumpy(state.thp).ravel(),
                      refl_dbz=xp.asnumpy(state.physics.refl_10cm).ravel())
        def expectation(key):
            if not effective_radii_due and key in ("effc_m", "effi_m", "effs_m"):
                return _column(after, key) * np.float32(1.e6)
            return _oracle_expectation(after, key)
        fields={key:metric(value,expectation(key)) for key,value in output.items()}
        for key,slot in _END_TO_END_SURFACE_FIELDS.items():
            fields[key]=metric(xp.asnumpy(state.existing_scratch(slot)).ravel(),
                               np.array([float(surface[key])],np.float32))
        results[scenario]={"reconstruction":reconstruction,"fields":fields}
    receipt={"backend":args.backend,"strict":os.environ.get("GPUWM_WRF_EXACT")=="1",
             "numerics": numerics.receipt(),
             "input_origin": "real-native-wrf-capture" if capture else "synthetic-column-fixtures",
             "real_native_capture": capture is not None,
             "capture_manifest": capture,
             "scenarios":results,"tendencies":captured,"command":sys.argv,
             "unmeasured_literal_tendencies": [scenario for scenario in scenarios if scenario not in captured],
             "effective_increment_tendencies": [],
             "literal_tendency_coverage": 14 if len(captured) == len(scenarios) else 0,
             "all_bitwise": False,
             "captured_tendencies_bitwise": all(v["max_ulp"] == 0 and v["nonfinite"] == 0
                 for row in captured.values() for v in row.values()),
             "state_bitwise": all(v["max_ulp"] == 0 and v["nonfinite"] == 0
                 for row in results.values() for v in row["fields"].values())}
    receipt["entry_inputs_bitwise"] = all(
        v["max_ulp"] == 0 and v["nonfinite"] == 0
        for row in results.values()
        for v in row["reconstruction"].get("input_metrics", {}).values())
    receipt["all_bitwise"] = (receipt["entry_inputs_bitwise"]
        and not receipt["unmeasured_literal_tendencies"]
        and receipt["state_bitwise"] and receipt["captured_tendencies_bitwise"])
    receipt["word_counts"] = {
        group: {"compared": sum(v["words"] for v in metrics),
                "differing": sum(v["unequal"] for v in metrics)}
        for group, metrics in (
            ("state", [v for row in results.values() for v in row["fields"].values()]),
            ("tendencies", [v for row in captured.values() for v in row.values()]))}
    args.out.write_text(json.dumps(receipt,indent=2)+"\n")
    for key,row in results.items():
        worst=sorted(row["fields"].items(),key=lambda kv:kv[1]["max_ulp"],reverse=True)[:6]
        print(key,[(name,m["max_ulp"],m["max_relative"]) for name,m in worst],flush=True)
    if args.compare_port:
        baseline = json.loads(args.compare_port.read_text())
        unequal = [(case, field) for case, row in results.items()
                   for field, values in row["fields"].items()
                   if values["got_sha256"] !=
                   baseline["scenarios"][case]["fields"][field]["got_sha256"]]
        receipt["changed_from_port_baseline"] = unequal
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
        print("CHANGED_FROM_PORT_BASELINE", unequal, flush=True)
        return 1 if unequal else 0
    if args.tolerance is not None:
        passed = all(v["nonfinite"] == 0 for row in results.values() for v in row["fields"].values()) and all(v["max_abs"] <= 2.e-4 if field == "refl_dbz"
                     else v["max_relative"] <= args.tolerance
                     for row in results.values() for field, v in row["fields"].items())
        tendency_pass = (args.tendency_tolerance is not None and all(
            v["nonfinite"] == 0 and v["max_relative"] <= args.tendency_tolerance
            for row in captured.values() for v in row.values()))
        if args.tendency_tolerance is not None:
            passed = passed and tendency_pass
        receipt["default_tendency_tolerance"] = args.tendency_tolerance
        receipt["default_tendency_tolerance_pass"] = tendency_pass
        receipt["default_tolerance"] = args.tolerance
        receipt["default_tolerance_pass"] = passed
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
        return 0 if passed else 1
    return 0 if receipt["all_bitwise"] else 1
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--backend",choices=("host","gpu"),required=True)
    p.add_argument("--fixtures",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--capture-manifest", type=Path,
                   help="Hash-bound manifest from a native forecast capture")
    p.add_argument("--tolerance",type=float)
    p.add_argument("--tendency-tolerance",type=float)
    p.add_argument("--mutation",action="store_true")
    p.add_argument("--compare-port",type=Path)
    raise SystemExit(run(p.parse_args()))
