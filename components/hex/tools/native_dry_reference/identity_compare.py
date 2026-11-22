#!/usr/bin/env python3
"""Bitwise hex-versus-native comparison at every native output time.

Starts hex from the native time-zero internal state (the dycore-match recipe),
steps the CUDA v8.4.1-lineage driver to each later native output time and
compares all 12 raw fields word for word (rho_zz, rho_theta, ru, rw, scalars,
theta_m, exner, rho_p, rtheta_p, pressure_p, u, w), each scalar species
separately, plus the 500 hPa height / 250 hPa wind RMS of forecast_divergence.
Nothing is tolerance-gated: a field is identical only if every 32-bit word is.
"""
from __future__ import annotations
import os as _hex_os
_hex_os.environ["WOOF_HEX_NUMERICS"] = "strict"  # a bitwise proof against native MPAS runs strict numerics (fast is the default, strict the opt-in)

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def time_of(path: Path) -> datetime:
    return datetime.strptime(path.name.removeprefix("tracer.").removesuffix(".nc"), "%Y-%m-%d_%H.%M.%S")


STATE_FIELDS = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
SAVED_FIELDS = ("theta_m", "exner", "density_perturbation", "rho_theta_perturbation",
                "pressure_perturbation", "normal_velocity", "vertical_velocity")


class SingleRunner:
    def __init__(self, driver, cp):
        self.driver, self.cp = driver, cp

    def advance(self, steps):
        for _ in range(steps):
            self.driver.atmosphere = self.driver.step_device().atmosphere
        self.cp.cuda.Device().synchronize()

    def fetch(self):
        from types import SimpleNamespace
        atm = self.driver.atmosphere
        state = SimpleNamespace(**{n: self.cp.asnumpy(getattr(atm.state, n)) for n in STATE_FIELDS})
        saved = SimpleNamespace(**{n: self.cp.asnumpy(getattr(atm.saved, n)) for n in SAVED_FIELDS})
        return state, saved

    def partition_record(self):
        return None

    def close(self):
        pass


class MultiRankRunner:
    """N resident ranks over native's own METIS partition, byte-only halos.

    Uses the existing same-host executor (partition_samehost_v841) exactly as
    tools/check_samehost_dycore_v841.py does; owned blocks are gathered back
    to global order for the word-for-word comparison with native.
    """

    def __init__(self, a, devices, mesh, state, grid, reference, saved, profiles, config):
        import cupy as cp
        tools = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(tools))
        from check_samehost_dycore_v841 import _write_partition_inputs, _reconstruct, _download, _assert_device
        from woof.hex.cuda_dualrun import PreparedCudaInputs
        from woof.hex.cuda_backend import KernelCache, require_cuda
        from woof.hex.cuda_driver import CudaDryDycoreDriver
        from woof.hex.integration import RKSchedule
        from woof.hex.partition_assets_v841 import build_partition_layouts
        from woof.hex.partition_executor_v841 import expected_round_sequence
        from woof.hex.partition_local_mesh_v841 import slice_prepared_host
        from woof.hex.partition_samehost_tables_v841 import build_all
        from woof.hex.partition_samehost_transport_v841 import SameHostTransport
        from woof.hex.partition_samehost_v841 import SameHostHaloExchanger, SameHostRanks
        from types import SimpleNamespace
        self.cp, self.devices, self.config = cp, devices, config
        self._reconstruct, self._download = _reconstruct, _download
        prepared = PreparedCudaInputs.validated(
            config=config, profile="identity-extend-nrank", target="native MPAS v8.4.2 identity on N ranks",
            preparation_method="native time-zero state, existing loaders", mesh=mesh, state=state,
            vertical=grid.vertical_grid, reference=reference, saved_diagnostics=saved,
            terrain_metrics=grid.terrain_metrics, input_bytes={}, reference_wind_profiles=profiles)
        n = len(devices)
        part_source = next(a.native.glob(f"*.graph.info.part.{n}"), None)
        if part_source is None:
            mesh_dir = Path(str(a.native.resolve()).split("/jw/")[0]) / "mesh"
            part_source = next(mesh_dir.glob(f"*.graph.info.part.{n}"))
        work = a.out.with_suffix(".partition")
        work.mkdir(parents=True, exist_ok=False)
        graph, part = _write_partition_inputs(prepared.mesh, n, work, part_source)
        self.part_source = part_source
        self.layouts = build_partition_layouts(
            graph, part, halo_rings=3, cells_on_edge=np.asarray(prepared.mesh.cellsOnEdge),
            cells_on_vertex=np.asarray(prepared.mesh.cellsOnVertex))
        tables = build_all(self.layouts)
        schedule = RKSchedule.from_mpas(config.config_dt, order=config.config_time_integration_order,
                                        acoustic_substeps=config.config_number_of_sub_steps,
                                        dynamics_splits=config.config_dynamics_split_steps)
        expected = expected_round_sequence(
            dynamics_splits=config.config_dynamics_split_steps,
            acoustic_steps=tuple(stage.acoustic_steps for stage in schedule.stages),
            monotonic=config.config_monotonic or config.config_positive_definite)
        source = SimpleNamespace(**{name: getattr(prepared, name) for name in (
            "mesh", "state", "vertical", "reference", "saved_diagnostics", "terrain_metrics",
            "reference_wind_profiles", "advection_coefficients")})
        local = [slice_prepared_host(source, layout) for layout in self.layouts]
        self.transport = SameHostTransport(tables, devices, xp=cp, copy_policy="auto", timeout=600.0)
        self.ranks = SameHostRanks(devices, self.transport, cp=cp)
        self.ranks.__enter__()

        def construct(rank):
            cache_dir = a.cache_root / f"rank-{rank}"
            capability = require_cuda(device_id=devices[rank], cache_dir=cache_dir)
            cache = KernelCache(capability=capability, cache_dir=cache_dir)
            item = local[rank]
            driver = CudaDryDycoreDriver.from_host(
                item.mesh, item.state.copy(), item.vertical, item.reference, config,
                saved_diagnostics=item.saved_diagnostics, terrain_metrics=item.terrain_metrics,
                advection_coefficients=item.advection_coefficients,
                reference_wind_profiles=item.reference_wind_profiles, kernel_cache=cache)
            _assert_device(driver, devices[rank])
            driver.halo_exchanger_v841 = SameHostHaloExchanger(tables[rank], self.transport,
                                                               expected_sequence=expected)
            return driver

        self.drivers = self.ranks.call(construct)
        self.step_index = 0

    def advance(self, steps):
        for _ in range(steps):
            self.step_index += 1
            step = self.step_index

            def one(rank):
                driver = self.drivers[rank]
                exchanger = driver.halo_exchanger_v841
                exchanger.begin_step(step)
                result = driver.step_device()
                driver.atmosphere = result.atmosphere
                exchanger.round_step_boundary(driver.atmosphere.state, driver.atmosphere.saved)
                exchanger.end_step()
                return None

            self.ranks.step(one, lambda rank: self.drivers[rank].atmosphere.state.time_seconds,
                            dt=self.config.config_dt)

    def fetch(self):
        blocks = self.ranks.call(lambda rank: self._download(self.drivers[rank].atmosphere, self.cp))
        whole = self._reconstruct(blocks, self.layouts)
        return whole.state, whole.saved

    def partition_record(self):
        return {"ranks": len(self.devices), "devices": list(self.devices), "metis_file": str(self.part_source),
                "metis_sha256": digest(self.part_source), "halo_rings": 3,
                "exchangers": [d.halo_exchanger_v841.receipt() for d in self.drivers]}

    def close(self):
        self.ranks.__exit__(None, None, None)


def digest_row(native, actual, actual_saved, names, out, seconds, steps):
    """Compare hex arrays with native by SHA-256; save differing hex arrays."""
    fields = {}
    for name in STATE_FIELDS:
        fields[name] = np.ascontiguousarray(getattr(actual, name), dtype=np.float32)
    for name in SAVED_FIELDS:
        fields[name] = np.ascontiguousarray(getattr(actual_saved, name), dtype=np.float32)
    same = {n: hashlib.sha256(v.tobytes()).hexdigest() == native["fields"][n] for n, v in fields.items()}
    scal = np.asarray(fields["scalars"])
    species = {n: hashlib.sha256(np.ascontiguousarray(scal[i]).tobytes()).hexdigest() == native["species"][n]
               for i, n in enumerate(names)}
    differing = [n for n, ok in same.items() if not ok]
    saved_path = None
    if differing:
        saved_path = out.with_name(out.stem + f"-hex-{native['name'][7:-3]}.npz")
        np.savez(saved_path, **{n: fields[n] for n in differing})
    return {"hours": seconds / 3600, "steps": steps, "native": native["name"],
            "native_output_sha256": native["file_sha256"], "compare": "sha256",
            "fields_identical": sum(same.values()), "fields": {n: {"identical": ok} for n, ok in same.items()},
            "differing_fields": differing, "species": {n: {"identical": ok} for n, ok in species.items()},
            "species_differing": [n for n, ok in species.items() if not ok],
            "saved_hex_arrays": str(saved_path) if saved_path else None}


def hex_identity(tree: Path) -> dict:
    """Commit of the hexcore that ran: git, or the exported archive's stamp."""
    import os
    try:
        commit = subprocess.check_output(["git", "-C", str(tree), "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(tree), "status", "--porcelain", "--", "src"],
                                             text=True).strip())
        return {"hex_commit": commit, "hex_dirty": dirty, "hex_source": "git"}
    except (subprocess.CalledProcessError, FileNotFoundError):
        sources = sorted((tree / "src" / "hexcore").glob("*.py"))
        h = hashlib.sha256()
        for f in sources:
            h.update(f.name.encode()); h.update(f.read_bytes())
        return {"hex_commit": os.environ.get("IDENTITY_HEX_COMMIT"), "hex_dirty": None,
                "hex_source": "git archive export", "hexcore_py_sha256": h.hexdigest()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True, help="native run directory with tracer.*.nc")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--scalars", default="qv", help="comma-separated scalar names in native order")
    p.add_argument("--dt", type=float, default=360.0)
    p.add_argument("--order", type=int, choices=(2, 3), default=2)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--mixing", choices=("off", "smag"), default="smag")
    p.add_argument("--smdiv", type=float, default=0.1)
    p.add_argument("--xnutr", type=float, default=0.2)
    p.add_argument("--max-outputs", type=int, default=0, help="stop after this many later outputs (0 = all)")
    p.add_argument("--devices", default="",
                   help="comma-separated card ordinals for N resident ranks (repeats = ranks sharing a card); "
                        "the partition is native's own METIS file MESH.graph.info.part.N")
    p.add_argument("--speed-steps", type=int, default=0,
                   help="after the comparison, time this many extra steps (synchronized)")
    p.add_argument("--native-digests", type=Path, default=None,
                   help="native_digests.py JSON: compare by per-field SHA-256 on a box that holds only the "
                        "native time-zero file; differing fields' hex arrays are saved for compare_saved.py")
    p.add_argument("--allow-unfinished", action="store_true",
                   help="compare the complete outputs of a native run that is still integrating "
                        "(the receipt records native_finished=false)")
    p.add_argument("--hex-tree", type=Path, default=None,
                   help="run another checkout's hexcore (e.g. the dycore-match tip) for paired speed/bits")
    a = p.parse_args()
    tree = (a.hex_tree or Path(__file__).resolve().parents[2]).resolve()
    sys.path[:0] = [str(tree / "src"), str(Path(__file__).resolve().parent)]
    from compare_dry_identity import compare, field_report
    from forecast_divergence import metrics, native_metrics, rms_pair
    from woof.hex.driver import load_mpas_vertical_grid, load_mpas_initial_state
    from woof.hex.dynamics_v841 import load_v841_reference_wind_profiles
    from woof.hex.config_v841 import V841DryDycoreConfig
    from woof.hex.mesh import Mesh
    from woof.hex.vector import initialize_reconstruction_coefficients
    from woof.hex.cuda_backend import require_cuda
    from woof.hex.cuda_backend.runtime import KernelCache
    from woof.hex.cuda_driver import CudaDryDycoreDriver
    import cupy as cp

    names = tuple(n.strip() for n in a.scalars.split(",") if n.strip())
    files = sorted(a.native.glob("tracer.*.nc"))
    digests = None
    if a.native_digests:
        digests = json.loads(a.native_digests.read_text())
        if tuple(digests["scalars"]) != names:
            raise ValueError("digest scalar list differs from --scalars")
        if files[0].name != digests["rows"][0]["name"] or digest(files[0]) != digests["rows"][0]["file_sha256"]:
            raise ValueError("native time-zero file does not match the digests")
        files = [a.native / r["name"] for r in digests["rows"]]
    if len(files) < 2:
        raise ValueError("need the time-zero native output and at least one later output")
    seconds = ([r["seconds"] for r in digests["rows"]] if digests else
               [(time_of(f) - time_of(files[0])).total_seconds() for f in files])
    steps_at = [s / a.dt for s in seconds]
    if any(not float(s).is_integer() for s in steps_at):
        raise ValueError("native output times are not whole hex steps")
    if a.max_outputs:
        files = files[: a.max_outputs + 1]
        steps_at = steps_at[: a.max_outputs + 1]
    native_init = next(a.native.glob("*.init.nc"))
    if digests:
        # The remote box carries copies of the deck and log; the digests bind
        # every later native output by content.
        pass
    if digest(native_init) != digest(a.init):
        raise ValueError("native initial file differs from --init")
    finished = "Finished running the atmosphere core" in (a.native / "log.atmosphere.0000.out").read_text()
    if not finished:
        if not a.allow_unfinished:
            raise ValueError("native run did not complete")
        # The newest file may still be being written; compare only older ones.
        files = files[:-1]
        steps_at = steps_at[:-1]
        if len(files) < 2:
            raise ValueError("unfinished native run has no complete later output yet")

    mesh = Mesh.from_netcdf(a.init)
    coefficients = initialize_reconstruction_coefficients(mesh, include_halos=True)
    mesh.arrays["coeffs_reconstruct"] = coefficients
    grid = load_mpas_vertical_grid(a.init, mesh, config_coef_3rd_order=.25)
    profiles = load_v841_reference_wind_profiles(a.init, n_vert_levels=grid.vertical_grid.n_vert_levels)

    def load(path):
        return load_mpas_initial_state(path, mesh, grid.vertical_grid, scalar_names=names,
                                       reference_path=path, terrain_metrics=grid.terrain_metrics,
                                       return_saved_diagnostics=True)

    state, reference, saved = load(files[0])
    for name in ("defc_a", "defc_b"):
        mesh.arrays[name] = np.zeros((len(mesh.arrays["nEdgesOnCell"]), mesh.arrays["edgesOnCell"].shape[1]), np.float32)
    config = V841DryDycoreConfig(
        config_dt=a.dt, config_time_integration_order=a.order, config_number_of_sub_steps=a.substeps,
        config_dynamics_split_steps=3, config_split_dynamics_transport=True, config_monotonic=True,
        config_positive_definite=False, config_scalar_adv_order=3, config_scalar_vadv_order=3,
        config_coef_3rd_order=.25, config_apvm_upwinding=.5, config_smdiv=a.smdiv,
        config_divergence_damping=a.smdiv > 0, config_xnutr=a.xnutr, config_zd=22000.,
        config_visc4_2dsmag=.05 if a.mixing == "smag" else 0.,
        config_smagorinsky_coef=.125 if a.mixing == "smag" else 0.,
        config_del4u_div_factor=10., config_h_ScaleWithMesh=True,
        config_horiz_mixing="2d_smagorinsky" if a.mixing == "smag" else "off")
    config.validate()
    t_build = time.perf_counter()
    devices = tuple(int(v) for v in a.devices.split(",")) if a.devices else ()
    if len(devices) >= 2:
        runner = MultiRankRunner(a, devices, mesh, state, grid, reference, saved, profiles, config)
    else:
        driver = CudaDryDycoreDriver.from_host(
            mesh, state, grid.vertical_grid, reference, config, saved_diagnostics=saved,
            terrain_metrics=grid.terrain_metrics, reference_wind_profiles=profiles,
            kernel_cache=KernelCache(capability=require_cuda(min_compute=(7, 5)), cache_dir=str(a.cache_root)))
        runner = SingleRunner(driver, cp)
    build_seconds = time.perf_counter() - t_build
    with __import__("netCDF4").Dataset(a.init) as d:
        z = np.asarray(d["zgrid"][:], np.float64)
    height = ((z[:, 1:] + z[:, :-1]) / 2).T
    area = np.asarray(mesh.arrays["areaCell"], np.float64)

    rows = []
    done = 0
    step_seconds = 0.0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for index in range(1, len(files)):
        target_steps = int(steps_at[index])
        t0 = time.perf_counter()
        runner.advance(target_steps - done)
        done = target_steps
        step_seconds += time.perf_counter() - t0
        actual, actual_saved = runner.fetch()
        if digests:
            row = digest_row(digests["rows"][index], actual, actual_saved, names, a.out, seconds[index], done)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("hours", "steps", "fields_identical", "differing_fields",
                                                  "species_differing")}), flush=True)
            a.out.write_text(json.dumps({"partial": True, "rows": rows}, indent=1) + "\n")
            continue
        target, _, target_saved = load(files[index])
        fields = compare(actual, actual_saved, target, target_saved)
        species = {n: field_report(np.asarray(actual.scalars)[i], np.asarray(target.scalars)[i])
                   for i, n in enumerate(names)}
        hexm = metrics(np.asarray(reference.pressure_base, np.float64)
                       + np.asarray(actual_saved.pressure_perturbation, np.float64),
                       np.asarray(actual_saved.normal_velocity), height, mesh)
        natm = native_metrics(files[index], height, mesh)
        common_z = np.isfinite(hexm["z500"]) & np.isfinite(natm["z500"])
        common_w = np.isfinite(hexm["u250"]) & np.isfinite(natm["u250"])
        for arm in (hexm, natm):
            arm["z500"][~common_z] = np.nan
            for n in ("u250", "v250"):
                arm[n][~common_w] = np.nan
        row = {"hours": seconds[index] / 3600, "steps": done,
               "differing_words_total": int(sum(0 if f.get("identical") else f.get("different_words", f.get("words", 1))
                                                for f in fields.values())),
               "fields_identical": sum(1 for f in fields.values() if f.get("identical")),
               "fields": fields, "species": species, "rms_vs_native": rms_pair(hexm, natm, area),
               "native_output_sha256": digest(files[index])}
        rows.append(row)
        print(json.dumps({"hours": row["hours"], "steps": done, "fields_identical": row["fields_identical"],
                          "differing_words_total": row["differing_words_total"],
                          "rms": row["rms_vs_native"],
                          "differing": {k: v.get("different_words") for k, v in fields.items() if not v.get("identical")},
                          "species_differing": {k: v.get("different_words") for k, v in species.items()
                                                if not v.get("identical")}}), flush=True)
        a.out.write_text(json.dumps({"partial": True, "rows": rows}, indent=1) + "\n")
    speed = None
    if a.speed_steps:
        runner.advance(1)
        t0 = time.perf_counter()
        runner.advance(a.speed_steps)
        per_step = (time.perf_counter() - t0) / a.speed_steps
        speed = {"steps": a.speed_steps, "seconds_per_step": per_step,
                 "seconds_per_forecast_hour": per_step * 3600 / a.dt}
    receipt = {
        "schema": "gpuwm-hex.identity-extend/v1",
        "init": str(a.init), "init_sha256": digest(a.init), "native": str(a.native),
        "native_digests": str(a.native_digests) if a.native_digests else None,
        "native_digests_sha256": digest(a.native_digests) if a.native_digests else None,
        "native_namelist_sha256": digest(a.native / "namelist.atmosphere"),
        "native_streams_sha256": digest(a.native / "streams.atmosphere"),
        "scalars": names, "dt": a.dt, "order": a.order, "substeps": a.substeps, "mixing": a.mixing,
        "smdiv": a.smdiv, "xnutr": a.xnutr, "cells": int(len(area)), "levels": int(height.shape[0]),
        **hex_identity(tree),
        "tool_sha256": digest(Path(__file__)),
        "host": platform.node(), "card": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
        "driver_build_seconds": build_seconds, "step_wall_seconds_including_sync": step_seconds,
        "native_finished": finished, "devices": list(devices) or None, "partition": runner.partition_record(),
        "speed": speed, "all_identical": all(r["fields_identical"] == 12 for r in rows),
        "rows": rows,
    }
    runner.close()
    a.out.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({"all_identical": receipt["all_identical"], "outputs": len(rows), "speed": speed}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
