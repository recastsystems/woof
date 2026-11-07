"""Public HRRR wrapper source-window orchestration regressions.

The cold-start receipt this wrapper validates is **not** hand-built here.
It used to be, and that is exactly how a one-line schema desync shipped
through 800+ green tests: ``tools/hrrr_single_domain_benchmark.py`` bumped
the receipt to ``...-initialization-v2`` while this consumer still demanded
``v1``, so the documented HRRR preparation front door raised ``RuntimeError:
HRRR preparation omitted deterministic cold-start evidence`` *after* a fully
successful preparation, for every one of the seven profiles -- and the suite
could not see it, because the fixture below constructed the receipt it
wanted rather than the one the producer emits.

So the fixture now RUNS the producer: real decoded native fields through
``woof.ingest.real.initialize_real`` and then through
``_initial_hrrr_microphysics_receipt``, the same function the benchmark
binds into ``report.json``.  A future schema bump, a renamed field, or a
dropped correspondence breaks this consumer's tests the moment the producer
changes, which is the only property that would have caught the regression.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timedelta
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from woof.physics_compat import (
    KESSLER_PROFILE_ID,
    MORRISON_PROFILE_ID,
    NSSL2_LEGACY_RRTMG_PROFILE_ID,
    NSSL2_PROFILE_ID,
    THOMPSON_LEGACY_RRTMG_PROFILE_ID,
    THOMPSON_RTE_RRTMGP_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID,
    WSM6_PROFILE_ID,
)
from woof.ingest.prepared_cache import (
    _reader_boundaries, PreparedCacheReader,
)
from woof.ingest.cpu_backend import resolve_cpu_bridge
from woof.core.grid import BaseState, VerticalCoord
from woof.hrrr_native_static import _array_sha256, sha256_file
from woof.hrrr_hierarchy_direct import _expected_root_cache_identity
from woof.native_wrf_contract import native_geometry_contract
from woof.native_domain_artifacts import write_native_hierarchy_artifacts
from woof.static.lambert import grids_from_projection_config
from woof.stream import _materialize_input_namelist
from woof.vertical_contract import explicit_vertical_from_wrf_namelist
from woof.experiment import load_experiment
from woof.hrrr_configuration import resolved_run_settings
from woof.io import restart as restart_io
from tools import prepare_hrrr_wrf as prepare
from tools import prepared_domain_tree_forecast as tree_runner
from tools.hrrr_single_domain_benchmark import (
    _experiment, _initial_hrrr_microphysics_receipt)

# The real decoded-source -> initialized-state driver.  It lives beside the
# producer's own tests; importing it is what makes this a cross-lane seam
# test instead of a second hand-built fixture.
from test_hrrr_single_domain_benchmark import (
    _decoded_native_hrrr_initialization, _write_native_physics_namelist)
from test_hrrr_native_static import _fixture as _native_static_fixture
from test_native_domain_artifacts import _inputs as _domain_artifact_inputs
from test_restart import (
    _sealed_tree_fixture,
)


#: The microphysics option each fixed profile pins, so the producer runs the
#: scheme whose cold start the consumer is about to validate.
_PROFILE_MICROPHYSICS = {
    KESSLER_PROFILE_ID: 1,
    WSM6_PROFILE_ID: 6,
    THOMPSON_PROFILE_ID: 8,
    # The legacy-RRTMG twins select the SAME microphysics as the row
    # above; they differ only in radiation, and one of them is the
    # route's default since 1.8.  Which species HRRR supplies and what an
    # absent one cold-starts to is a microphysics property, which is why
    # the shipped cold-start contract aliases them the same way.
    THOMPSON_LEGACY_RRTMG_PROFILE_ID: 8,
    THOMPSON_RTE_RRTMGP_PROFILE_ID: 8,
    THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID: 8,
    MORRISON_PROFILE_ID: 10,
    NSSL2_PROFILE_ID: 18,
    NSSL2_LEGACY_RRTMG_PROFILE_ID: 18,
}


@functools.lru_cache(maxsize=None)
def _producer_receipt(profile: str) -> dict[str, object]:
    """Whatever the real producer emits for ``profile``, verbatim."""

    from woof.physics_compat import single_domain_runtime_switches

    # Resolve the actual named profile, including newly added source forms.
    # The receipt still comes from the unchanged real cold-start producer.
    scheme = single_domain_runtime_switches(profile)["mp_physics"]
    fixture_inputs = {} if scheme != 28 else {
        "landmask": np.ones((2, 3), dtype=np.float64),
        "mp28_aerosol_source": "synthetic"}
    result = _decoded_native_hrrr_initialization(scheme, **fixture_inputs)
    return _initial_hrrr_microphysics_receipt(
        result.state, profile, result.hydrometeor_initialization)


def _cold_start_receipt(profile: str) -> dict[str, object]:
    return copy.deepcopy(_producer_receipt(profile))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_decoder(tmp_path: Path) -> Path:
    """Executable decoder fixture; all orchestration around it is production.

    The fixture replaces only GRIB interpretation with small deterministic
    meteorological arrays.  The public wrapper still launches a real process,
    consumes the production READY protocol, maps/initializes through the native
    CPU backend, seals the bridge, writes the cache, and later extends both.
    """

    script = tmp_path / "minimal_hrrr_decoder.py"
    script.write_text(textwrap.dedent(r'''
        from pathlib import Path
        import os
        import shutil
        import sys
        import time

        import numpy as np

        def put(path, value):
            np.ascontiguousarray(value, dtype="<f4").tofile(path)

        def ready(path, rows):
            path.write_text("".join(f"{key}\t{value}\n" for key, value in rows),
                            encoding="ascii")

        def main(argv):
            if len(argv) != 11 or argv[1] != "--series-workers-ready":
                raise SystemExit(f"unexpected decoder arguments: {argv!r}")
            workers, series, output, signals, cycle = (
                argv[2], Path(argv[3]), Path(argv[4]), Path(argv[5]), argv[6])
            i0, i1, j0, j1 = map(int, argv[7:11])
            rows = [line.split("\t") for line in series.read_text().splitlines()
                    if line.strip()]
            hours = [int(row[0]) for row in rows]
            ny, nx = j1 - j0 + 1, i1 - i0 + 1
            started = time.perf_counter()
            # On Windows the executable path is a .cmd shim; the producer owns
            # that waiting cmd.exe PID, while this Python worker is its child.
            if os.name == "nt":
                import psutil
                marker = Path(__file__).with_suffix(".cmd").name.lower()
                parents = psutil.Process().parents()
                launchers = []
                for candidate in parents:
                    try:
                        command = " ".join(candidate.cmdline()).lower()
                        name = candidate.name().lower()
                    except (psutil.AccessDenied, psutil.NoSuchProcess):
                        continue
                    if name in {"cmd", "cmd.exe"} and marker in command:
                        launchers.append(candidate)
                # The nearest marker-bearing cmd.exe is the process returned
                # by Popen.  No fixed ancestry depth: redirectors and shell
                # policy can add ancestors, but cannot change this marker.
                owner_pid = (launchers[0] if launchers
                             else parents[0]).pid
            else:
                owner_pid = os.getpid()
            staging = output.parent / f".{output.name}.partial-{owner_pid}"
            publish = output.parent / f".{output.name}.publish-{owner_pid}"
            staging.mkdir(parents=True)
            signals.mkdir(parents=True)
            gate = (
                "status\tPASS\n"
                f"cycle\t{cycle}\n"
                "atmosphere_selected_per_time\t561\n"
                "hybrid_levels\t50\n"
                "soil_selected_per_time\t18\n"
                "optional_soil_surface_fields\tVEGFRA\n"
                "optional_soil_surface_units\tVEGFRA=percent\n"
                f"window_zero_based_inclusive\ti={i0}..{i1} j={j0}..{j1}\n"
                f"window_shape\t{ny}x{nx}\n"
                "qice_mapping\tPASS discipline=0 category=1 parameter=82 "
                "level_type=105; deterministic minimal fixture\n"
                "cross_time_inventory\tPASS deterministic minimal fixture\n"
                f"forecast_hours\t{','.join(map(str, hours))}\n"
                f"series_count\t{len(hours)}\n"
            )
            (staging / "gate.txt").write_text(gate, encoding="ascii")
            ready(signals / "preflight.ready", (
                ("status", "PASS"), ("series_count", len(hours)),
                ("staging_retention", "until_consumer_finish"),
                ("canonical_output", output.resolve()), ("workers", workers),
                ("producer_elapsed_seconds", time.perf_counter() - started),
                ("staging_root", staging.resolve()),
            ))
            pressure_levels = np.linspace(5000.0, 100000.0, 50,
                                          dtype=np.float32)
            pressure = np.broadcast_to(
                pressure_levels[:, None, None], (50, ny, nx)).copy()
            height = np.broadcast_to(
                -7900.0 * np.log(pressure_levels / 100000.0)[:, None, None],
                (50, ny, nx)).copy()
            level = np.arange(50, dtype=np.float32)[:, None, None]
            row = np.arange(ny, dtype=np.float32)[None, :, None]
            column = np.arange(nx, dtype=np.float32)[None, None, :]
            for hour in hours:
                atmosphere = staging / f"atmosphere-f{hour:02d}"
                soil = staging / f"soil-f{hour:02d}"
                atmosphere.mkdir()
                soil.mkdir()
                temperature = (215.0 + 75.0 * (pressure / 100000.0) ** 0.22
                               + np.float32(hour) * np.float32(0.05))
                fields3 = {
                    "PRES": pressure, "HGT": height, "TT": temperature,
                    "SPFH": np.full_like(pressure, 0.004),
                    "U_MASS": np.full_like(pressure, 8.0 + 0.1 * hour),
                    "V_MASS": np.full_like(pressure, -2.0 + 0.05 * hour),
                }
                for index, name in enumerate(("QC", "QI", "QR", "QS", "QG"), 1):
                    fields3[name] = np.asarray(
                        index * 1.0e-7 * (1.0 + level + 0.01 * row
                                          + 0.001 * column), dtype=np.float32)
                for name, value in fields3.items():
                    put(atmosphere / f"{name}.f32le", value)
                surface = np.ones((ny, nx), dtype=np.float32)
                fields2 = {
                    "PSFC": surface * 100000.0,
                    "SOILHGT": surface * 100.0,
                    "SKINTEMP": surface * (289.0 + 0.05 * hour),
                    "SNOW": surface * 0.0, "SNOWH": surface * 0.0,
                    "T2": surface * (289.0 + 0.05 * hour),
                    "Q2": surface * 0.004,
                    "U10_MASS": surface * (7.0 + 0.1 * hour),
                    "V10_MASS": surface * (-1.0 + 0.05 * hour),
                    "LANDSEA": surface, "XICE": surface * 0.0,
                }
                for name, value in fields2.items():
                    put(atmosphere / f"{name}.f32le", value)
                put(soil / "SOILT.f32le", np.full((9, ny, nx),
                                                   283.0 + 0.05 * hour))
                put(soil / "SOILW.f32le", np.full((9, ny, nx), 0.25))
                put(soil / "VEGFRA.f32le", surface * 37.5)
                ready(signals / f"f{hour:02d}.ready", (
                    ("status", "PASS"), ("forecast_hour", hour),
                    ("payload_files", 25),
                    ("producer_elapsed_seconds", time.perf_counter() - started),
                ))
            shutil.copytree(staging, publish)
            os.replace(publish, output)
            ready(signals / "complete.ready", (
                ("status", "PASS"), ("series_count", len(hours)),
                ("staging_retained", "true"),
                ("canonical_output", output.resolve()),
                ("producer_elapsed_seconds", time.perf_counter() - started),
            ))

        if __name__ == "__main__":
            main(sys.argv)
    ''').lstrip(), encoding="utf-8")
    # The resolver reads the binary for this release's contract marker
    # before it hands the path to anything, so a fixture decoder has to
    # carry it too -- as a comment, since this launcher is a script.  A
    # fixture that did not carry it would be exactly the stale bridge
    # the gate exists to stop.
    from woof import bridges
    contract = bridges.BRIDGE_ABI_MARKERS["hrrr_grib2_bridge"].decode("ascii")
    if os.name == "nt":
        launcher = tmp_path / "minimal_hrrr_decoder.cmd"
        launcher.write_text(
            f'@echo off\nrem {contract}\n"{sys.executable}" "{script}" %*\n',
            encoding="ascii")
    else:
        launcher = tmp_path / "minimal_hrrr_decoder"
        launcher.write_text(
            f"#!{sys.executable}\n"
            f"# {contract}\n"
            f"exec(compile(open({str(script)!r}, 'rb').read(), "
            f"{str(script)!r}, 'exec'))\n",
            encoding="utf-8")
        launcher.chmod(0o755)
    return launcher.resolve()


def _contract_carrying_decoder(path: Path) -> Path:
    """A stand-in bridge binary that speaks this release's contract.

    ``b"decoder"`` used to be enough because the resolver only asked
    whether the file existed.  It asks the contract question itself now,
    so a placeholder has to answer it -- which is the point: a build
    that predates the contract no longer reaches a production door.
    """
    from woof import bridges

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"decoder\x00" + bridges.BRIDGE_ABI_MARKERS["hrrr_grib2_bridge"])
    return path


def _configured_wrapper_input(tmp_path, namelist, profile=WSM6_PROFILE_ID,
                              *, target=None, write_namelist=True, vertical=None, changes=None, history_interval=3600.):
    from tools.hrrr_single_domain_benchmark import _experiment_tables
    from woof.experiment import VerticalConfig, build_experiment
    from woof.experiment_document import render_experiment_document
    from woof.hrrr_route_inputs import render_namelist_input
    from woof.ingest.hrrr_target import HrrrTargetDomain
    import copy
    target = target or HrrrTargetDomain.legacy_500x500()
    vertical = vertical or VerticalConfig(
        eta_levels=tuple(float(v) for v in np.linspace(1., 0., target.nz + 1)),
        p_top=5000., hybrid_opt=2, etac=.2)
    raw, _ = _experiment_tables(vertical, run_seconds=7200., target=target,
        start_time=datetime(2026, 7, 18, 5), physics_profile=profile,
        history_interval_seconds=history_interval)
    if changes:
        raw["shared"].update(changes)
    config = tmp_path / "native-authority.toml"
    config.write_text(render_experiment_document(raw), encoding="utf-8")
    if write_namelist:
        namelist.write_text(render_namelist_input(build_experiment(
            copy.deepcopy(raw), source="wrapper test authority")), encoding="utf-8")
    return config


def _real_wrapper_inputs(tmp_path: Path):
    """Small input data for the unmodified public preparation tool path."""

    static_root = tmp_path / "static-fixture"
    static_root.mkdir()
    target, static_cache, static_receipt = _native_static_fixture(static_root)
    with np.load(static_cache, allow_pickle=False) as archive:
        fields = {name: np.asarray(archive[name]).copy()
                  for name in archive.files}
    fields["MAPFAC_M"].fill(1.0)
    fields["LANDMASK"].fill(1.0)
    fields["HGT_M"].fill(100.0)
    fields["F"].fill(8.0e-5)
    fields["SOILTEMP"].fill(285.0)
    fields["TMN"].fill(285.0)
    fields["GREENFRAC"].fill(0.5)
    fields["LAI12M"].fill(2.0)
    fields["ALBEDO12M"].fill(0.2)
    for name in ("LANDUSEF", "SOILCTOP", "SOILCBOT"):
        fields[name].fill(0.0)
        fields[name][0].fill(1.0)
    np.savez(static_cache, **fields)
    receipt = json.loads(static_receipt.read_text(encoding="utf-8"))
    receipt["cache"].update({
        "bytes": static_cache.stat().st_size,
        "sha256": sha256_file(static_cache),
    })
    receipt["array_sha256"] = {
        name: _array_sha256(value) for name, value in sorted(fields.items())}
    receipt["geometry"] = native_geometry_contract(
        target.grid(), target.contract_cfg())
    static_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    domain = tmp_path / "domain.json"
    domain.write_text(
        json.dumps(target.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    namelist = tmp_path / "namelist.input"
    _write_native_physics_namelist(namelist)
    eta = ", ".join(map(str, np.linspace(1.0, 0.0, target.nz + 1)))
    text = namelist.read_text(encoding="ascii").replace(
        "&dynamics\n", "&dynamics\n hybrid_opt = 2,\n etac = 0.2,\n")
    namelist.write_text(
        "&time_control\n/\n" + text + ("&domains\n"
                " max_dom = 2,\n"
                f" e_vert = {target.nz + 1},\n"
                " p_top_requested = 10000.0,\n"
                f" eta_levels = {eta},\n"
                "/\n"),
        encoding="ascii")
    return target, static_cache, static_receipt, domain, namelist


def _fake_sealed_bridge(root: Path, hours, *, cycle="2026-07-18 05:00:00"):
    root.mkdir()
    for hour in hours:
        for role, count in (("atmosphere", 22), ("soil", 2)):
            directory = root / f"{role}-f{hour:02d}"
            directory.mkdir()
            for index in range(count):
                (directory / f"field-{index:02d}.f32le").write_bytes(
                    f"{role}:{hour}:{index}".encode("ascii"))
    gate = root / "gate.txt"
    gate.write_text(
        "status\tPASS\n"
        f"forecast_hours\t{','.join(map(str, hours))}\n"
        f"series_count\t{len(hours)}\n"
        f"cycle\t{cycle}\n"
        "window_shape\t207x207\n",
        encoding="utf-8")
    payloads = sorted(
        path for path in root.rglob("*") if path.is_file())
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{_file_sha256(path)}  ./{path.relative_to(root).as_posix()}\n"
            for path in payloads),
        encoding="utf-8")
    return root


@pytest.mark.parametrize("profile", sorted(_PROFILE_MICROPHYSICS))
def test_the_consumer_accepts_exactly_what_the_producer_emits(profile: str):
    """The binding the schema desync escaped through.

    Nothing here is written down twice: the receipt comes from the producer
    and the constant comes from the consumer, so they cannot drift apart
    silently again.
    """

    receipt = _producer_receipt(profile)
    assert receipt["schema"] == prepare.HRRR_INITIALIZATION_SCHEMA
    assert prepare._validated_physics_receipt(
        {"physics": {
            "schema": "gpuwm-prepared-physics-profile-v1",
            "profile": profile,
            "hrrr_initialization": receipt,
        }},
        requested_profile=profile)["profile"] == profile


def test_the_consumer_accepts_the_correct_mode_morrison_receipt(monkeypatch):
    """THE BREAKAGE (IFS-9KM-DEFAULT-2026-11-04): ``woof go`` on an HRRR
    start with the Morrison profile fetched 11.8 GB, then refused its own
    complete receipt with "omitted deterministic cold-start evidence".
    In correct mode (HN1 on, the default outside a strict process) the
    producer states Morrison's nc as the NDCNST droplet number, a shape the
    consumer did not know.  The receipt here is the real producer's."""

    from woof import correct_mode
    from woof.ingest.microphysics_cold_start import hrrr_cold_start_preflight

    monkeypatch.setattr(correct_mode, "STRICT", False)
    monkeypatch.setattr(correct_mode, "CORRECT_KEYS_DEFAULT", True)
    _producer_receipt.cache_clear()
    try:
        receipt = _cold_start_receipt(MORRISON_PROFILE_ID)
    finally:
        _producer_receipt.cache_clear()
    assert receipt["state_source_absent_fields"]["nc"] == {
        "ndcnst_droplet_number": True, "all_finite_above_zero": True}
    report = {"physics": {
        "schema": "gpuwm-prepared-physics-profile-v1",
        "profile": MORRISON_PROFILE_ID,
        "hrrr_initialization": receipt}}
    assert prepare._validated_physics_receipt(
        report, requested_profile=MORRISON_PROFILE_ID)["profile"] \
        == MORRISON_PROFILE_ID
    # The preflight the run plan calls before the fetch admits it too...
    from woof.physics_compat import single_domain_runtime_switches
    assert "QNCLOUD" not in hrrr_cold_start_preflight(
        single_domain_runtime_switches(MORRISON_PROFILE_ID))
    # ...and a forged nc is still refused.
    forged = copy.deepcopy(report)
    forged["physics"]["hrrr_initialization"]["state_source_absent_fields"][
        "nc"] = {"ndcnst_droplet_number": True}
    with pytest.raises(RuntimeError, match="cold-start evidence"):
        prepare._validated_physics_receipt(
            forged, requested_profile=MORRISON_PROFILE_ID)


def test_an_hrrr_start_without_a_cold_start_contract_is_refused_before_the_fetch():
    from woof.ingest.microphysics_cold_start import hrrr_cold_start_preflight

    with pytest.raises(ValueError, match="refused before the download"):
        hrrr_cold_start_preflight({"mp_physics": 987654})


def test_the_consumer_refuses_the_previous_producer_schema():
    """v1/v2 acceptance is deliberately not retained.

    ``main`` runs the producer itself and reads back the ``report.json``
    that invocation just wrote, so an older-schema receipt cannot reach this
    consumer; accepting one would widen the gate for a case that does not
    exist.  Dropping the version is therefore a refusal, not a widening.
    """

    for stale in ("gpuwm-hrrr-microphysics-initialization-v1",
                  "gpuwm-hrrr-microphysics-initialization-v2",
                  "gpuwm-hrrr-microphysics-initialization-v3"):
        receipt = _cold_start_receipt(NSSL2_PROFILE_ID)
        receipt["schema"] = stale
        with pytest.raises(RuntimeError, match="cold-start evidence"):
            prepare._validated_physics_receipt(
                {"physics": {
                    "schema": "gpuwm-prepared-physics-profile-v1",
                    "profile": NSSL2_PROFILE_ID,
                    "hrrr_initialization": receipt,
                }},
                requested_profile=NSSL2_PROFILE_ID)


def test_the_consumer_validates_the_correspondence_fields_not_the_string():
    """Every v2/v3/v4 field the producer added is checked, one at a time."""

    def _refuse(mutate, match="correspondence evidence"):
        receipt = _cold_start_receipt(NSSL2_PROFILE_ID)
        mutate(receipt)
        with pytest.raises(RuntimeError, match=match):
            prepare._validated_physics_receipt(
                {"physics": {
                    "schema": "gpuwm-prepared-physics-profile-v1",
                    "profile": NSSL2_PROFILE_ID,
                    "hrrr_initialization": receipt,
                }},
                requested_profile=NSSL2_PROFILE_ID)

    _refuse(lambda r: r.pop("source_to_state_correspondence"))
    _refuse(lambda r: r.pop("retention_evidence"))
    _refuse(lambda r: r.pop("state_mass_fields"))
    _refuse(lambda r: r.pop("vertical_disposition"))
    _refuse(lambda r: r["source_to_state_correspondence"].pop("QG"))
    _refuse(lambda r: r["state_mass_fields"]["qg"].pop("nonzero_mask_sha256"))
    _refuse(lambda r: r["state_mass_fields"]["qg"].pop("decoded_source"))
    # A species claimed PROVEN on a source that carried no mass.
    _refuse(
        lambda r: r["retention_evidence"]["QG"].update(
            {"source_nonzero_count": 0}),
        match="disagrees with its own fingerprints")
    # Evidence that disagrees with the fingerprints beside it.
    _refuse(
        lambda r: r["retention_evidence"]["QG"].update(
            {"state_nonzero_count": 1}),
        match="disagrees with its own fingerprints")
    # The headline B-H1 failure: every nonzero analyzed cell lost.
    def _lose_all_graupel(receipt):
        receipt["state_mass_fields"]["qg"]["nonzero_count"] = 0
        receipt["retention_evidence"]["QG"]["state_nonzero_count"] = 0
    _refuse(_lose_all_graupel, match="lost every nonzero analyzed QG")


def test_the_consumer_reports_a_vacuous_retention_claim_without_refusing_it():
    """A cloud-free analysis is a legitimate preparation, and says so.

    The wrapper must not pass a vacuous receipt off as proof, and must not
    refuse it either -- a genuinely cloud-free domain prepares correctly.
    """

    empty = _decoded_native_hrrr_initialization(18, analyzed_species=())
    receipt = _initial_hrrr_microphysics_receipt(
        empty.state, NSSL2_PROFILE_ID, empty.hydrometeor_initialization)
    summary = prepare._validated_retention_evidence(receipt)
    assert summary["strength"] == "VACUOUS"
    assert summary["vacuous_species"] == ["QC", "QG", "QI", "QR", "QS"]

    proven = prepare._validated_retention_evidence(
        _producer_receipt(NSSL2_PROFILE_ID))
    assert proven["strength"] == "PARTIALLY_WRF_EXCLUDED"
    assert proven["vacuous_species"] == []
    assert proven["partially_excluded_species"] == [
        "QC", "QG", "QI", "QR", "QS"]


def test_the_consumer_accepts_an_exhaustive_wrf_exclusion_partition():
    excluded = _decoded_native_hrrr_initialization(
        18, analyzed_source_levels={"QC": 5})
    receipt = _initial_hrrr_microphysics_receipt(
        excluded.state, NSSL2_PROFILE_ID,
        excluded.hydrometeor_initialization)

    summary = prepare._validated_retention_evidence(receipt)

    assert summary["strength"] == "WRF_EXCLUDED"
    assert summary["excluded_species"] == ["QC"]
    assert summary["vacuous_species"] == ["QG", "QI", "QR", "QS"]


def _cuda_preparation_receipt(requested=None, host_workers=8):
    """The CUDA backend's receipt and budget as the benchmark writes them.

    The masked surface fields run on the host under CUDA, on
    ``requested`` threads (``auto``: every CPU, ``host_workers`` here).
    """
    return {
        "preprocess_backend": {
            "backend": "cuda",
            "masked_surface_chain": {
                "workers": "auto" if requested is None else requested},
        },
        "preprocess_worker_budget": {
            "schema": "gpuwm-preprocess-worker-budget-v1",
            "backend": "cuda",
            "applicable": False,
            "requested_total_native_workers": requested,
            "host_step_native_workers": (
                host_workers if requested is None else requested),
            "pipeline_decoder_workers_included": False,
            "peak_active_native_workers": (
                host_workers if requested is None else requested),
        },
    }


def test_cuda_worker_receipt_names_the_host_step_request():
    """Under CUDA --preprocess-workers sets the host steps' threads.

    The receipt check used to demand zero native workers under CUDA,
    which stopped being true when the masked surface fields moved to the
    Rust library on the host; it now holds the request against the
    masked chain's receipt and the budget.
    """
    def report(preparation):
        return {"status": "PASS", "preparation": preparation,
                "pipeline": {"workers": {"requested": "4", "selected": 4}}}

    for requested in (None, 1, 6):
        prepare._validated_worker_receipts(
            report(_cuda_preparation_receipt(requested)),
            selected_backend="cuda", requested_preprocess_workers=requested,
            requested_pipeline_workers="4", final_hour=1)
    # A request the preparation did not honour, either way round.
    for ran, asked in ((None, 6), (6, None), (5, 6)):
        with pytest.raises(RuntimeError, match="differs from the request"):
            prepare._validated_worker_receipts(
                report(_cuda_preparation_receipt(ran)),
                selected_backend="cuda", requested_preprocess_workers=asked,
                requested_pipeline_workers="4", final_hour=1)
    # The receipt the old budget wrote: no host-step count, peak zero.
    stale = _cuda_preparation_receipt()
    del stale["preprocess_worker_budget"]["host_step_native_workers"]
    stale["preprocess_worker_budget"]["peak_active_native_workers"] = 0
    with pytest.raises(RuntimeError, match="host-step worker budget"):
        prepare._validated_worker_receipts(
            report(stale), selected_backend="cuda",
            requested_preprocess_workers=None,
            requested_pipeline_workers="4", final_hour=1)


def test_decoder_workers_are_independent_of_the_native_preprocess_budget():
    """The wrapper's own point: two worker budgets, separately accounted.

    Written at the decode pipeline's real ceiling.  It used to be
    written at 64 -- a number no run could reach, because the sealed
    decoder refuses anything over 13 and the pipeline refused it in
    turn, after the static build had already been paid for.
    """
    workers = prepare.MAX_PIPELINE_WORKERS
    report = {
        "status": "PASS",
        "preparation": _cuda_preparation_receipt(),
        "pipeline": {"workers": {"requested": str(workers),
                                 "selected": workers}},
    }

    _preprocess, native_budget, decoder = prepare._validated_worker_receipts(
        report, selected_backend="cuda", requested_preprocess_workers=None,
        requested_pipeline_workers=str(workers), final_hour=12)

    assert decoder["selected"] == workers
    assert native_budget["pipeline_decoder_workers_included"] is False


def test_pipeline_workers_refuses_over_the_decoders_ceiling_at_parse_time():
    """The late refusal this replaces cost a whole static build first.

    The wrapper advertised 1..64 while the decoder it spawns accepts
    1..13, so `--pipeline-workers 32` was accepted, the geometry receipt
    and native static cache were built, and only then did the pipeline
    refuse.  argparse now answers with a usage message instead.
    """
    from tools.hrrr_pipeline import MAX_PIPELINE_WORKERS

    assert prepare.MAX_PIPELINE_WORKERS == MAX_PIPELINE_WORKERS

    parser = prepare._parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--source-root", "src", "--source-manifest", "m",
            "--source-manifest-sha256", "0" * 64, "--namelist-input", "n",
            "--valid-time", "2026-08-01_00:00:00", "--output-root", "out",
            "--static-cache", "c", "--static-receipt", "r",
            "--pipeline-workers", str(MAX_PIPELINE_WORKERS + 1)])

    # Watched firing: the ceiling itself, and 'auto', both parse.
    for accepted in (str(MAX_PIPELINE_WORKERS), "auto"):
        args = parser.parse_args([
            "--source-root", "src", "--source-manifest", "m",
            "--source-manifest-sha256", "0" * 64, "--namelist-input", "n",
            "--valid-time", "2026-08-01_00:00:00", "--output-root", "out",
            "--static-cache", "c", "--static-receipt", "r",
            "--pipeline-workers", accepted])
        assert args.pipeline_workers == accepted
def test_bridge_extension_hardlinks_prefix_and_one_absolute_new_hour(tmp_path):
    prior = _fake_sealed_bridge(tmp_path / "prior", [0, 1])
    suffix = _fake_sealed_bridge(tmp_path / "suffix", [1, 2])
    output = tmp_path / "merged"

    receipt = prepare._bridge_manifest_extension(
        predecessor=prior, suffix=suffix, output=output,
        old_hours=[0, 1], new_hours=[0, 1, 2])

    assert receipt["old_source_forecast_hours"] == [0, 1]
    assert receipt["suffix_source_forecast_hours"] == [1, 2]
    assert receipt["new_source_forecast_hours"] == [0, 1, 2]
    assert receipt["extended_sha256"] == _file_sha256(
        output / "SHA256SUMS")
    assert (output / "atmosphere-f00" / "field-00.f32le").samefile(
        prior / "atmosphere-f00" / "field-00.f32le")
    assert (output / "atmosphere-f01" / "field-00.f32le").samefile(
        prior / "atmosphere-f01" / "field-00.f32le")
    assert (output / "atmosphere-f02" / "field-00.f32le").samefile(
        suffix / "atmosphere-f02" / "field-00.f32le")
    gate = dict(
        row.split("\t", 1)
        for row in (output / "gate.txt").read_text().splitlines())
    assert gate["forecast_hours"] == "0,1,2"
    assert gate["series_count"] == "3"
    entries = prepare._manifest_entries(output / "SHA256SUMS")
    prepare._verify_manifest_payloads(output, entries)


def test_bridge_extension_copies_its_prefix_on_a_drive_without_hard_links(
        tmp_path, monkeypatch):
    """exFAT has no hard links: the prefix is copied and proven identical,
    and the merged manifest verifies.  This refused, citing only the cost."""
    import errno

    def link(source, destination, *args, **kwargs):
        if os.name == "nt":
            raise OSError(errno.EINVAL, "Incorrect function", str(source), 1,
                          str(destination))
        raise OSError(errno.EPERM, "Operation not permitted", str(source),
                      None, str(destination))

    prior = _fake_sealed_bridge(tmp_path / "prior", [0, 1])
    suffix = _fake_sealed_bridge(tmp_path / "suffix", [1, 2])
    output = tmp_path / "merged"
    monkeypatch.setattr(os, "link", link)

    receipt = prepare._bridge_manifest_extension(
        predecessor=prior, suffix=suffix, output=output,
        old_hours=[0, 1], new_hours=[0, 1, 2])

    assert receipt["new_source_forecast_hours"] == [0, 1, 2]
    for name, origin in (("atmosphere-f00/field-00.f32le", prior),
                         ("atmosphere-f01/field-00.f32le", prior),
                         ("atmosphere-f02/field-00.f32le", suffix)):
        assert (output / name).read_bytes() == (origin / name).read_bytes()
        assert not (output / name).samefile(origin / name)
    entries = prepare._manifest_entries(output / "SHA256SUMS")
    prepare._verify_manifest_payloads(output, entries)


def test_bridge_extension_refuses_changed_suffix_authority(tmp_path):
    prior = _fake_sealed_bridge(tmp_path / "prior", [0, 1])
    suffix = _fake_sealed_bridge(
        tmp_path / "suffix", [1, 2], cycle="2026-07-18 06:00:00")

    with pytest.raises(ValueError, match="immutable gate field cycle"):
        prepare._bridge_manifest_extension(
            predecessor=prior, suffix=suffix, output=tmp_path / "merged",
            old_hours=[0, 1], new_hours=[0, 1, 2])


def test_bridge_extension_refuses_dropped_optional_soil_surface_field(tmp_path):
    prior = _fake_sealed_bridge(tmp_path / "prior", [0, 1])
    suffix = _fake_sealed_bridge(tmp_path / "suffix", [1, 2])
    gate = prior / "gate.txt"
    gate.write_text(gate.read_text() +
        "optional_soil_surface_fields\tVEGFRA\n"
        "optional_soil_surface_units\tVEGFRA=percent\n")
    for hour in (0, 1):
        (prior / f"soil-f{hour:02d}" / "VEGFRA.f32le").write_bytes(b"vegetation")
    (prior / "SHA256SUMS").write_text("".join(
        f"{_file_sha256(path)}  ./{path.relative_to(prior).as_posix()}\n"
        for path in sorted(prior.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"))
    with pytest.raises(ValueError, match="optional soil surface inventory changes"):
        prepare._bridge_manifest_extension(
            predecessor=prior, suffix=suffix, output=tmp_path / "merged",
            old_hours=[0, 1], new_hours=[0, 1, 2])


def test_source_manifest_extension_requires_exact_unchanged_prefix(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    cycle = datetime(2026, 7, 18, 5)
    old_names = []
    for hour in (0, 1, 2):
        for product in ("wrfnat", "soil"):
            name = f"hrrr.t05z.{product}f{hour:02d}.grib2"
            (root / name).write_bytes(f"{product}:{hour}".encode("ascii"))
            if hour < 2:
                old_names.append(name)
    old = tmp_path / "prior-SHA256SUMS"
    old.write_text("".join(
        f"{_file_sha256(root / name)}  {name}\n"
        for name in sorted(old_names)), encoding="utf-8")
    current = root / "SHA256SUMS"
    current.write_text("".join(
        f"{_file_sha256(path)}  {path.name}\n"
        for path in sorted(root.glob("*.grib2"))), encoding="utf-8")

    receipt = prepare._source_manifest_extension(
        predecessor=old, extended=current, source_root=root,
        old_hours=[0, 1], new_hours=[0, 1, 2], cycle=cycle)

    assert receipt["retained_entries"] == 4
    assert [row["path"] for row in receipt["added_entries"]] == [
        "hrrr.t05z.soilf02.grib2", "hrrr.t05z.wrfnatf02.grib2"]
    target = root / old_names[0]
    old_digest = _file_sha256(target)
    target.write_bytes(b"mutated-prefix")
    changed = current.read_text().replace(
        old_digest, _file_sha256(target), 1)
    current.write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="changes its sealed prefix"):
        prepare._source_manifest_extension(
            predecessor=old, extended=current, source_root=root,
            old_hours=[0, 1], new_hours=[0, 1, 2], cycle=cycle)


def test_extension_refuses_legacy_hydrometeor_cache_before_suffix_work(
        tmp_path, monkeypatch):
    prior = tmp_path / "prior"
    native = prior / "native"
    cache = native / "prepared-cache"
    report = native / "preparation-report"
    bridge = native / "native-bridge"
    for directory in (cache, report, bridge):
        directory.mkdir(parents=True, exist_ok=True)
    snapshot = native / "source-manifest.snapshot"
    snapshot.write_text("sealed\n", encoding="utf-8")
    for path in (
            report / "report.json",
            prior / "native-static.npz",
            prior / "native-static-receipt.json",
            prior / "native-geometry-receipt.json"):
        path.write_bytes(b"fixture")
    (prior / "public-wrapper-result.json").write_text(json.dumps({
        "status": "PASS",
        "source_cycle": "2026-07-18T05:00:00",
        "source_forecast_hours": [0, 1],
        "history_interval_seconds": 3600.0,
        "physics": {"profile": WSM6_PROFILE_ID},
        "prepared_cache_contract": {"mode": "sealed-prefix-v1"},
    }), encoding="utf-8")
    (cache / "header.json").write_text(json.dumps({
        "identity": {},
        "metadata": {
            "forcing_extension_mode": "sealed-prefix-v1",
            "hydrometeor_initialization": {
                "schema": "gpuwm-real-hydrometeor-correspondence-v1",
            },
        },
    }), encoding="utf-8")
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_manifest = source_root / "SHA256SUMS"
    source_manifest.write_text("extended\n", encoding="utf-8")
    suffix_runs = []
    monkeypatch.setattr(
        prepare, "_source_manifest_extension", lambda **_kwargs: {})
    monkeypatch.setattr(
        prepare, "_run", lambda *args, **kwargs:
        suffix_runs.append((args, kwargs)))
    args = SimpleNamespace(
        extend_root_preparation=prior,
        forecast_start_hour=0,
        run_seconds=7200.0,
        physics_profile=WSM6_PROFILE_ID,
        history_interval_seconds=3600.0,
        source_manifest=source_manifest,
        source_manifest_sha256=_file_sha256(source_manifest),
        source_root=source_root,
    )
    output = tmp_path / "extended"

    with pytest.raises(ValueError, match="rebuild it with the current GPUWM"):
        prepare._sealed_extension(
            args, valid_time=datetime(2026, 7, 18, 5),
            source_forecast_hours=(0, 1, 2), output=output,
            env={}, decoder=tmp_path / "decoder", started=0.0,
            namelist_invariant={
                "schema": "gpuwm-namelist-extension-invariant-v1",
                "sha256": "a" * 64,
            })

    assert suffix_runs == []
    assert not output.exists()


def _write_minimal_stream_tree_config(
        path: Path, *, cycle: datetime, run_seconds: float, root) -> Path:
    # The tree's d01 is the wrapper's own domain (``root``, its
    # HrrrTargetDomain) on the wrapper namelist's eta levels: the tree joins
    # that domain's prepared boundaries, and boundaries only fit the grid
    # they were prepared on.
    eta = ", ".join(repr(float(value))
                    for value in np.linspace(1.0, 0.0, root.nz + 1))
    path.write_text(textwrap.dedent(f"""
        [experiment]
        name = "stream-extension-tree"
        start_time = {cycle.isoformat()}
        run_seconds = {run_seconds}
        restart_interval_s = 3600.0
        # This fixture runs Noah with no radiation selector, which the
        # 2026-08-09 radiation-off land-surface guard refuses undeclared
        # (woof/physics_compat.py radiation_off_land_surface_refusal).
        # The test is about stream-tree extension mechanics, not the sky;
        # declaring the experiment keeps the physics this fixture always
        # exercised while satisfying the guard the product now enforces.
        # Two claims, two tokens, exactly as the refusal spells them.
        acknowledgements = [
            "radiation-off-land-surface-v1",
            "constant-downward-longwave-v1",
        ]
        # The wrapper's domain is small, so its 3:1 nest clears the
        # parent's Davies zone and no terrain-blend band past it; the
        # ground is flat, so there is no terrain to blend.
        blend_width = 0

        [projection]
        map_proj = "{root.map_proj}"
        ref_lat = {float(root.ref_lat)}
        ref_lon = {float(root.ref_lon)}
        truelat1 = {float(root.truelat1)}
        truelat2 = {float(root.truelat2)}
        stand_lon = {float(root.stand_lon)}

        [shared]
        nz = {int(root.nz)}
        ztop = 16000.0
        p_top = 10000.0
        eta_levels = [{eta}]
        hybrid_opt = 2
        etac = 0.2
        map_proj = 1
        moist = true
        mp_physics = 6
        bl_pbl_physics = 1
        sf_sfclay_physics = 91
        sf_surface_physics = 2
        ra_lw_physics = 1
        ra_sw_physics = 1
        radt = 12.0

        [[domain]]
        grid_id = 1
        parent_id = 0
        i_parent_start = 1
        j_parent_start = 1
        parent_grid_ratio = 1
        parent_time_step_ratio = 1
        nx = {int(root.nx)}
        ny = {int(root.ny)}
        dx = {float(root.dx_m)}
        time_step = {int(root.time_step_seconds)}
        history_interval_s = 3600.0

        [[domain]]
        grid_id = 2
        parent_id = 1
        i_parent_start = 6
        j_parent_start = 6
        parent_grid_ratio = 3
        parent_time_step_ratio = 3
        nx = 6
        ny = 6
        history_interval_s = 1800.0
    """).lstrip(), encoding="utf-8")
    return path


def _stream_tree_static(ny: int, nx: int) -> dict[str, np.ndarray]:
    plane = np.ones((ny, nx), dtype=np.float32)
    landuse = np.zeros((21, ny, nx), dtype=np.float32)
    soil_category = np.zeros((16, ny, nx), dtype=np.float32)
    landuse[0] = 1.0
    soil_category[0] = 1.0
    return {
        "HGT_M": 100.0 * plane,
        "LANDMASK": plane,
        "LU_INDEX": plane,
        "SCT_DOM": plane,
        "SCB_DOM": plane,
        "LANDUSEF": landuse,
        "SOILCTOP": soil_category,
        "SOILCBOT": soil_category.copy(),
        "GREENFRAC": np.ones((12, ny, nx), dtype=np.float32),
        "LAI12M": np.ones((12, ny, nx), dtype=np.float32),
        "ALBEDO12M": np.ones((12, ny, nx), dtype=np.float32),
        "SNOALB": plane,
        "SOILTEMP": 278.0 * plane,
        "TMN": 278.0 * plane,
    }


def _prepared_coord_and_base(reader):
    """The vertical coordinate and base state a prepared cache restores."""

    metadata = reader.header["metadata"]
    coord = VerticalCoord(**dict(metadata["coord_scalars"]), **{
        name: reader.read_array(f"coord/{name}")
        for name in metadata["coord_arrays"]})
    base = BaseState(**dict(metadata["base_scalars"]), **{
        name: reader.read_array(f"base/{name}")
        for name in metadata["base_arrays"]})
    return coord, base


def _build_and_preflight_stream_tree(
        root: Path, *, config: Path, wrapper: Path,
        cycle: datetime, forcing_hours: tuple[int, ...], root_cache_domain):
    experiment = load_experiment(config)
    grids = tuple(grids_from_projection_config(experiment))
    root_initial, root_met, root_soil, _static, _boundaries, _grid = \
        _domain_artifact_inputs()
    child_initial, child_met, child_soil, _static, _boundaries, _grid = \
        _domain_artifact_inputs()
    wrapper_header = json.loads((
        wrapper / "native" / "prepared-cache" / "header.json"
    ).read_text(encoding="utf-8"))
    identity = wrapper_header["identity"]
    expected_root_identity = _expected_root_cache_identity(
        identity, root_domain=root_cache_domain,
        bridge_manifest_sha256=identity["bridge_manifest_sha256"],
        source_manifest_sha256=identity["source_manifest_sha256"],
        static_cache_sha256=identity["static_cache_sha256"],
        forcing_hours=forcing_hours)
    wrapper_reader = PreparedCacheReader(
        wrapper / "native" / "prepared-cache",
        expected_identity=expected_root_identity)
    wrapper_reader.verify_all()
    boundaries = _reader_boundaries(wrapper_reader)
    # The root cache carries the wrapper's boundaries, which fit only the
    # grid and base state they were prepared with, so the tree's root takes
    # the wrapper's vertical coordinate and base state, and its nest a base
    # state on its own grid under the same coordinate.
    coord, root_initial.base = _prepared_coord_and_base(wrapper_reader)
    nest = experiment.domains[1].run
    nz, ny, nx = nest.nz, nest.ny, nest.nx
    child_initial.base = BaseState(
        mub=np.full((ny, nx), 90_000.0), p_top=10_000.0,
        pb=np.full((nz, ny, nx), 50_000.0),
        alb=np.full((nz, ny, nx), 0.8),
        thb=np.full((nz, ny, nx), 290.0),
        phb=np.zeros((nz + 1, ny, nx)), terrain_z=np.zeros((ny, nx)))
    root_initial.coord = child_initial.coord = coord
    root_initial.state.lateral_boundaries = boundaries
    child_initial.state.lateral_boundaries = None
    child = SimpleNamespace(
        domain=experiment.domains[1], real=child_initial, grid=grids[1],
        horizontal=child_met, soil=child_soil,
        static_fields=_stream_tree_static(
            experiment.domains[1].run.ny, experiment.domains[1].run.nx),
        preprocess_receipt={"backend": "cpu", "workers": 1},
        input_preparation_seconds=0.0)
    root.mkdir()
    artifacts = write_native_hierarchy_artifacts(
        root / "hierarchy-artifacts", exp=experiment,
        root_grid=grids[0], root_initial_result=root_initial,
        root_met=root_met, root_soil=root_soil,
        root_static_fields=_stream_tree_static(
            experiment.root.run.ny, experiment.root.run.nx),
        root_boundaries=boundaries, child_results=(child,),
        bridge_manifest_sha256=identity["bridge_manifest_sha256"],
        source_manifest_sha256=identity["source_manifest_sha256"],
        namelist_sha256=identity["namelist_sha256"],
        forcing_hours=forcing_hours,
        source_identity=identity["source_identity"], valid_time=cycle,
        root_metadata={
            "source_prepared_content_sha256":
                wrapper_header["content_sha256"],
        })
    receipt = {
        "schema": tree_runner.HIERARCHY_SCHEMA,
        "status": "PASS",
        "valid_time": cycle.isoformat(),
        "domain_count": len(experiment.domains),
        "forcing_hours": list(forcing_hours),
        "source_manifest_sha256": identity["source_manifest_sha256"],
        "provenance": {
            "bridge_manifest_sha256": identity["bridge_manifest_sha256"],
            "source_manifest_sha256": identity["source_manifest_sha256"],
            "native_namelist_input_sha256": identity["namelist_sha256"],
        },
        "artifact_receipt": dict(artifacts.receipt),
    }
    receipt_path = root / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    inputs = tree_runner.preflight_prepared_tree(
        prepared_root=root,
        preparation_receipt_sha256=_file_sha256(receipt_path),
        experiment_config=config,
        experiment_config_sha256=_file_sha256(config))
    return inputs, wrapper_reader, boundaries


def test_public_wrapper_extension_passes_production_tree_contracts(
        tmp_path, monkeypatch):
    """Run f001 -> f002 through the public root-preparation tool.

    Only the source bytes and decoded meteorological arrays are fixture data.
    ``prepare.main``, its subprocess benchmark, producer protocol, bridge
    sealer/merge, CPU preprocessing, prepared-cache writer/extension, and all
    integrity readers are the production components shipped to operators.
    The downstream assertions exercise production artifact/preflight/restart
    contracts; actual forecast advancement remains the live controller proof.
    """

    cycle = datetime(2026, 7, 18, 5)
    source = tmp_path / "source"
    source.mkdir()
    names_by_hour = {}
    for hour in (0, 1, 2):
        names_by_hour[hour] = []
        for product in ("wrfnat", "soil"):
            name = f"hrrr.t05z.{product}f{hour:02d}.grib2"
            (source / name).write_bytes(
                f"minimal-fixture:{product}:{hour}".encode("ascii"))
            names_by_hour[hour].append(name)

    source_manifest = source / "SHA256SUMS"

    def publish_source_manifest(hours):
        source_manifest.write_text("".join(
            f"{_file_sha256(source / name)}  {name}\n"
            for hour in hours for name in sorted(names_by_hour[hour])
        ), encoding="utf-8")
        return _file_sha256(source_manifest)

    (target, static_cache, static_receipt,
     domain, namelist_template) = _real_wrapper_inputs(tmp_path)
    f001_namelist = tmp_path / "namelist-f001.input"
    f002_namelist = tmp_path / "namelist-f002.input"
    _materialize_input_namelist(
        namelist_template, f001_namelist, cycle=cycle, lead=1,
        domain_starts=[cycle, cycle])
    _materialize_input_namelist(
        namelist_template, f002_namelist, cycle=cycle, lead=2,
        domain_starts=[cycle, cycle])
    assert _file_sha256(f001_namelist) != _file_sha256(f002_namelist)
    vertical = explicit_vertical_from_wrf_namelist(
        f001_namelist, expected_nz=target.nz,
        context="wrapper/hierarchy composition test")
    prior_root_domain = _experiment(
        vertical, run_seconds=3600.0, start_time=cycle, target=target,
        physics_profile=WSM6_PROFILE_ID,
        history_interval_seconds=3600.0).root
    extended_root_domain = _experiment(
        vertical, run_seconds=7200.0, start_time=cycle, target=target,
        physics_profile=WSM6_PROFILE_ID,
        history_interval_seconds=3600.0).root
    decoder = _fixture_decoder(tmp_path)
    monkeypatch.setenv("WOOF_HRRR_DECODER", str(decoder))
    try:
        cpu_bridge = resolve_cpu_bridge()
    except FileNotFoundError:
        pytest.skip("native CPU preprocessing bridge is not installed")

    authority = _configured_wrapper_input(tmp_path, f001_namelist,
        target=target, write_namelist=False, vertical=vertical)
    common = [
        "--experiment-config", str(authority),
        "--source-root", str(source),
        "--source-manifest", str(source_manifest),
        "--static-cache", str(static_cache),
        "--static-receipt", str(static_receipt),
        "--domain-spec", str(domain),
        "--physics-profile", WSM6_PROFILE_ID,
        "--valid-time", cycle.strftime("%Y-%m-%d_%H:%M:%S"),
        "--forecast-start-hour", "0",
        "--history-interval-seconds", "3600",
        "--pipeline-workers", "1",
        "--prepare-workers", "1",
        "--preprocess-backend", "cpu",
        "--preprocess-workers", "1",
        "--cpu-preprocess-bridge", str(cpu_bridge),
        "--sealed-prepared-cache",
        "--skip-stock-wrf-export",
    ]
    prior = tmp_path / "prepared-f001"
    initial_manifest_sha = publish_source_manifest((0, 1))
    assert prepare.main(common + [
        "--source-manifest-sha256", initial_manifest_sha,
        "--namelist-input", str(f001_namelist),
        "--forecast-end-hour", "1", "--run-seconds", "3600",
        "--output-root", str(prior),
    ]) == 0

    output = tmp_path / "prepared-f002"
    extended_manifest_sha = publish_source_manifest((0, 1, 2))
    assert prepare.main(common + [
        "--source-manifest-sha256", extended_manifest_sha,
        "--namelist-input", str(f002_namelist),
        "--forecast-end-hour", "2", "--run-seconds", "7200",
        "--output-root", str(output),
        "--extend-root-preparation", str(prior),
    ]) == 0

    result = json.loads(
        (output / "public-wrapper-result.json").read_text(encoding="utf-8"))
    assert result["source_forecast_hours"] == [0, 1, 2]
    assert result["prepared_cache_contract"]["operation"] == "extend-one-hour"
    prior_bridge = prior / "native" / "native-bridge"
    extended_bridge = output / "native" / "native-bridge"
    assert (extended_bridge / "atmosphere-f00" / "TT.f32le").samefile(
        prior_bridge / "atmosphere-f00" / "TT.f32le")
    assert (extended_bridge / "atmosphere-f01" / "TT.f32le").samefile(
        prior_bridge / "atmosphere-f01" / "TT.f32le")
    assert (extended_bridge / "soil-f00" / "VEGFRA.f32le").samefile(
        prior_bridge / "soil-f00" / "VEGFRA.f32le")
    for hour in (0, 1, 2):
        vegetation = np.fromfile(
            extended_bridge / f"soil-f{hour:02d}" / "VEGFRA.f32le", dtype="<f4")
        assert np.all(vegetation == np.float32(37.5))
    assert (output / "native-static.npz").samefile(
        prior / "native-static.npz")
    prior_header = json.loads((
        prior / "native" / "prepared-cache" / "header.json"
    ).read_text(encoding="utf-8"))
    header = json.loads((
        output / "native" / "prepared-cache" / "header.json"
    ).read_text(encoding="utf-8"))
    assert prior_header["identity"]["namelist_sha256"] == _file_sha256(
        f001_namelist)
    assert header["identity"]["namelist_sha256"] == _file_sha256(
        f002_namelist)
    assert prior_header["identity"]["namelist_extension_invariant"] == \
        header["identity"]["namelist_extension_invariant"]
    assert header["identity"]["forcing_hours"] == [0, 1, 2]
    assert len(header["metadata"]["lbc"]["intervals"]) == 2
    cache_reader = PreparedCacheReader(
        output / "native" / "prepared-cache",
        expected_identity=header["identity"])
    cache_reader.verify_all()
    np.testing.assert_allclose(cache_reader.read_array("met/VEGFRA"), 37.5,
                               rtol=0.0, atol=1e-5)
    old_name, old_entry = next(iter(prior_header["arrays"].items()))
    new_entry = header["arrays"][old_name]
    assert (output / "native" / "prepared-cache" /
            new_entry["file"]).samefile(
        prior / "native" / "prepared-cache" / old_entry["file"])
    assert header["metadata"]["source_manifest_extension"][
        "suffix_source_forecast_hours"] \
        == [1, 2]
    assert header["metadata"]["bridge_manifest_extension"][
        "suffix_source_forecast_hours"] \
        == [1, 2]
    assert not (output / "native" / "extension-work").exists()

    prior_config = _write_minimal_stream_tree_config(
        tmp_path / "tree-f001.toml", cycle=cycle, run_seconds=3600.0,
        root=target)
    extended_config = _write_minimal_stream_tree_config(
        tmp_path / "tree-f002.toml", cycle=cycle, run_seconds=7200.0,
        root=target)
    prior_tree, prior_tree_reader, prior_boundaries = \
        _build_and_preflight_stream_tree(
            tmp_path / "tree-f001", config=prior_config, wrapper=prior,
            cycle=cycle, forcing_hours=(0, 1),
            root_cache_domain=prior_root_domain)
    extended_tree, extended_tree_reader, extended_boundaries = \
        _build_and_preflight_stream_tree(
            tmp_path / "tree-f002", config=extended_config,
            wrapper=output, cycle=cycle, forcing_hours=(0, 1, 2),
            root_cache_domain=extended_root_domain)
    assert prior_tree.forcing_hours == (0, 1)
    assert extended_tree.forcing_hours == (0, 1, 2)
    assert [bundle.grid_id for bundle in prior_tree.domains] == [1, 2]
    assert [bundle.grid_id for bundle in extended_tree.domains] == [1, 2]
    # The preflight's time-step reading read the wrapper's boundaries
    # against the tree's own base state, to the end of each window; the
    # ground is flat, so every domain runs as configured.
    for tree, last in ((prior_tree, "+1 h"), (extended_tree, "+2 h")):
        rows = {row["grid_id"]: row for row in tree.terrain_clock["domains"]}
        assert sorted(rows) == [1, 2]
        for row in rows.values():
            assert row["status"] == "AS_CONFIGURED", row
            assert row["source"] == "d01", row
            assert row["when"] == f"boundary at {last}", row
    assert prior_tree.domains[0].cache_identity[
        "source_manifest_sha256"] == prior_header["identity"][
            "source_manifest_sha256"]
    assert extended_tree.domains[0].cache_identity[
        "source_manifest_sha256"] == header["identity"][
            "source_manifest_sha256"]
    hierarchy_lbc_keys = sorted(
        key for key in extended_tree.domains[0].cache_reader.arrays
        if key.startswith("lbc/"))
    wrapper_lbc_keys = sorted(
        key for key in extended_tree_reader.arrays
        if key.startswith("lbc/"))
    assert hierarchy_lbc_keys == wrapper_lbc_keys
    for key in hierarchy_lbc_keys:
        np.testing.assert_array_equal(
            extended_tree.domains[0].cache_reader.read_array(key),
            extended_tree_reader.read_array(key))

    source_model, _model_start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0,
        payload_seed=131)
    source_model.schedule.clock.start_time = cycle
    source_model.root.cfg.run = replace(
        source_model.root.cfg.run,
        spec_bdy_width=prior_boundaries.spec_bdy_width,
        spec_zone=prior_boundaries.spec_zone,
        relax_zone=prior_boundaries.relax_zone)
    source_model.root.state.lateral_boundaries = prior_boundaries
    resumed_model, _model_start = _sealed_tree_fixture(
        monkeypatch, forcing_count=3, run_seconds=7200.0,
        payload_seed=231)
    resumed_model.schedule.clock.start_time = cycle
    resumed_model.root.cfg.run = replace(
        resumed_model.root.cfg.run,
        spec_bdy_width=extended_boundaries.spec_bdy_width,
        spec_zone=extended_boundaries.spec_zone,
        relax_zone=extended_boundaries.relax_zone)
    resumed_model.root.state.lateral_boundaries = extended_boundaries
    checkpoint_root = tmp_path / "checkpoint-generations"
    f001_root = restart_io.write_tree_restart(
        checkpoint_root, source_model, cycle + timedelta(hours=1),
        sealed_forcing_extension=True)
    f001 = restart_io.restore_tree_restart(
        f001_root, resumed_model, sealed_forcing_extension=True)
    assert f001.elapsed_ticks / f001.tick_den == 3600.0
    assert sorted(f001.headers_by_grid_id) == [1, 2]
    f001_ids = {
        row["checkpoint_set_id"]
        for row in f001.headers_by_grid_id.values()}
    assert len(f001_ids) == 1
    assert all(row["elapsed_seconds"] == 3600.0
               for row in f001.headers_by_grid_id.values())

    for node in resumed_model.walk_parent_first():
        node.clock.ticks = 7200
        node.clock.step_count = 7200
        node.state.elapsed_seconds = 7200.0
    f002_root = restart_io.write_tree_restart(
        checkpoint_root, resumed_model, cycle + timedelta(hours=2),
        sealed_forcing_extension=True)
    f002_paths = restart_io._tree_restart_paths(f002_root, {1, 2})
    f002_headers = {
        grid_id: restart_io.read_restart_header(path)
        for grid_id, path in f002_paths.items()}
    f002_ids = {
        row["checkpoint_set_id"] for row in f002_headers.values()}
    assert len(f002_ids) == 1
    assert f002_ids != f001_ids
    assert sorted(f002_headers) == [1, 2]
    assert all(row["domain_ids"] == [1, 2]
               for row in f002_headers.values())
    assert all(row["elapsed_seconds"] == 7200.0
               for row in f002_headers.values())


@pytest.mark.parametrize(
    "physics_profile",
    (
        KESSLER_PROFILE_ID,
        WSM6_PROFILE_ID,
        THOMPSON_PROFILE_ID,
        MORRISON_PROFILE_ID,
        NSSL2_PROFILE_ID,
        NSSL2_LEGACY_RRTMG_PROFILE_ID,
    ),
)
def test_public_wrapper_preserves_absolute_source_leads_and_rebases_model_time(
        tmp_path: Path, monkeypatch, pinned_thompson_tables,
        physics_profile: str) -> None:
    commands, output = _run_public_wrapper_on_fake_children(
        tmp_path, monkeypatch, physics_profile)
    _assert_source_leads_and_model_time(commands, output, physics_profile)


def test_public_wrapper_forwards_cuda_host_workers(
        tmp_path: Path, monkeypatch) -> None:
    """--preprocess-workers under CUDA reaches the benchmark and is held.

    It used to be refused here, after the front door had accepted and
    forwarded it, although the CUDA backend runs the masked surface
    fields on the host on exactly that many threads.
    """
    commands, output = _run_public_wrapper_on_fake_children(
        tmp_path, monkeypatch, WSM6_PROFILE_ID,
        extra_arguments=("--preprocess-backend", "cuda",
                         "--preprocess-workers", "1"),
        preparation=_cuda_preparation_receipt(1))
    _assert_source_leads_and_model_time(commands, output, WSM6_PROFILE_ID)
    benchmark = next(
        command for command in commands
        if any(value.endswith("hrrr_single_domain_benchmark.py")
               for value in command))
    assert benchmark[benchmark.index("--preprocess-backend") + 1] == "cuda"
    assert benchmark[benchmark.index("--preprocess-workers") + 1] == "1"


def _run_public_wrapper_on_fake_children(
        tmp_path, monkeypatch, physics_profile, *, extra_arguments=(),
        preparation=None, experiment_config=True, static_receipt_bytes=None,
        authority_suffix=""):
    if preparation is None:
        preparation = _cuda_preparation_receipt()
    source = tmp_path / "source"
    source.mkdir()
    for hour in (12, 13):
        (source / f"hrrr.t05z.wrfnatf{hour:02d}.grib2").write_bytes(
            f"atmos-{hour}".encode("ascii"))
        (source / f"hrrr.t05z.soilf{hour:02d}.grib2").write_bytes(
            f"soil-{hour}".encode("ascii"))
    source_manifest = source / "SHA256SUMS"
    source_manifest.write_text("fixture\n", encoding="ascii")
    static_cache = tmp_path / "static.npz"
    static_receipt = tmp_path / "static.json"
    namelist = tmp_path / "namelist.input"
    for path in (static_cache, static_receipt, namelist):
        path.write_bytes(b"fixture")
    if static_receipt_bytes is not None:
        static_receipt.write_bytes(static_receipt_bytes)
    authority = _configured_wrapper_input(tmp_path, namelist, physics_profile)
    if authority_suffix:
        authority.write_text(
            authority.read_text(encoding="utf-8") + authority_suffix,
            encoding="utf-8")
    decoder = tmp_path / "hrrr_grib2_bridge"
    decoder.write_bytes(b"decoder")
    monkeypatch.setattr(prepare, "_decoder", lambda _env: decoder)

    commands: list[list[str]] = []
    if experiment_config:
        selection = load_experiment(authority).root.run
    else:
        # What the namelist route resolves, which is what the preparer
        # checks the report's physics against on that route.
        from woof.hrrr_configuration import native_configuration_defaults
        selection = native_configuration_defaults(
            namelist_input=namelist, physics_profile=physics_profile).root.run

    def fake_run(command: list[str], _env: dict[str, str],
                 cwd=None) -> None:
        commands.append(command)
        if any(value.endswith("hrrr_single_domain_benchmark.py")
               for value in command):
            outdir = Path(command[command.index("--outdir") + 1])
            outdir.mkdir(parents=True)
            (outdir / "report.json").write_text(json.dumps({
                "status": "PASS",
                "history_interval_seconds": 3600.0,
                "physics": {
                    "schema": "gpuwm-prepared-physics-profile-v1",
                    "resolved": resolved_run_settings(selection),
                    "profile": physics_profile,
                    "hrrr_initialization": _cold_start_receipt(
                        physics_profile),
                },
                "preparation": preparation,
                "pipeline": {"workers": {"requested": "8", "selected": 8}},
            }), encoding="utf-8")
        elif "woof.wrf_direct" in command:
            # Behave like the real atomic exporter on success: the
            # fail-closed step verifies the declared outputs at PASS.
            export_output = Path(command[command.index("--output") + 1])
            export_output.mkdir(parents=True)
            for name in ("wrfinput_d01", "wrfbdy_d01"):
                (export_output / name).write_bytes(b"exported")
            (export_output / "manifest.json").write_text(json.dumps({
                "files": {"wrfinput_d01": {"sha256": "0" * 64},
                          "wrfbdy_d01": {"sha256": "0" * 64}},
            }), encoding="utf-8")

    monkeypatch.setattr(prepare, "_run", fake_run)
    monkeypatch.setattr(prepare, "_run_keeping_refusal",
                        _keeping_refusal(fake_run))
    output = tmp_path / "output"
    assert prepare.main([
        *(("--experiment-config", str(authority)) if experiment_config else ()),
        "--source-root", str(source),
        "--source-manifest", str(source_manifest),
        "--source-manifest-sha256", "0" * 64,
        "--static-cache", str(static_cache),
        "--static-receipt", str(static_receipt),
        "--namelist-input", str(namelist),
        "--physics-profile", physics_profile,
        "--valid-time", "2026-07-18_05:00:00",
        "--forecast-start-hour", "12",
        "--forecast-end-hour", "13",
        "--run-seconds", "3600",
        "--history-interval-seconds", "3600",
        "--output-root", str(output),
        *extra_arguments,
    ]) == 0
    return commands, output


@pytest.mark.parametrize("attested", [False, True])
def test_a_prebuilt_static_attesting_d01_smoothing_goes_through_the_builder(
        tmp_path: Path, monkeypatch, attested: bool) -> None:
    """The breakage: a default run linked a smoothed prebuilt static.

    On the --static-cache route a preparation with no carrier and default
    d01 smoothing linked the prebuilt static unchecked.  The builder's own
    refusal of a static built under another d01 smoothing ("was built with
    d01 terrain smoothing") never ran, and the root seam passes a default
    request, so a static whose receipt attests 1-2-1 x3 integrated its
    smoothed terrain under a default configuration.  Such a static now goes
    through the builder, which holds it to the request; one that attests no
    smoothing is still linked byte for byte.
    """
    from woof.static.terrain_smoothing import TerrainSmoothing

    receipt = (json.dumps({"terrain_smoothing": {
        "d01": TerrainSmoothing("1-2-1", 3).echo()}}).encode("utf-8")
               if attested else None)
    # The wrapper's sub-kilometre root takes the grid-spacing default
    # carrier, which goes through the builder whatever its receipt says;
    # the opt-out is a run with no carrier that replaces statics.
    commands, output = _run_public_wrapper_on_fake_children(
        tmp_path, monkeypatch, WSM6_PROFILE_ID,
        static_receipt_bytes=receipt,
        authority_suffix=('\n[static.highres]\nenabled = false\n'
                          'cache_root = "cache"\n'))
    builders = [command for command in commands
                if any(value.endswith("hrrr_build_native_static.py")
                       for value in command)]
    if attested:
        (builder,) = builders
        assert builder[builder.index("--static-receipt") + 1] == str(
            (tmp_path / "static.json").resolve())
        assert builder[builder.index("--output") + 1] == str(
            output / "native-static.npz")
        assert not (output / "native-static-receipt.json").exists()
    else:
        assert builders == []
        linked = output / "native-static-receipt.json"
        assert linked.read_bytes() == b"fixture"


def test_a_namelist_only_preparation_builds_the_static_the_benchmark_checks(
        tmp_path: Path, monkeypatch) -> None:
    """The breakage: a namelist-only preparation of a 1 km root was refused.

    Without --experiment-config the static builder was handed no
    configuration and built the 30-arc-second baseline, while the
    benchmark resolved the grid-spacing default from the same namelist and
    refused the static receipt for not binding it -- "rebuild the static
    preparation with the declared experiment configuration", a
    configuration the user never declared.  The builder now receives the
    block the root configuration resolved, and a receipt applied under it
    is the one the benchmark accepts.
    """
    from woof.hrrr_configuration import resolve_root_experiment
    from woof.ingest.hrrr_target import HrrrTargetDomain
    from woof.static import highres_production as production

    # A suite with both radiation streams: the namelist alone has no
    # [experiment] table to carry a shortwave-only suite's night-window
    # acknowledgement, and this preparation's window starts at night.
    commands, _output = _run_public_wrapper_on_fake_children(
        tmp_path, monkeypatch, MORRISON_PROFILE_ID, experiment_config=False)
    builder = next(
        command for command in commands
        if any(value.endswith("hrrr_build_native_static.py")
               for value in command))
    assert "--experiment-config" not in builder
    table = json.loads(builder[builder.index("--static-highres") + 1])
    case_date = builder[builder.index("--case-date") + 1]
    assert case_date == "2026-07-18"
    # The carrier itself, as the identity a seal records (A169: the
    # [static] table alone dropped d01's smoothing and the urban legend).
    built = production.parse_sealed_static_highres(
        table, source="--static-highres", base_dir=tmp_path)
    assert production.default_row_of(built) is not None

    # What the benchmark resolves for this same namelist-only preparation.
    benchmark = next(
        command for command in commands
        if any(value.endswith("hrrr_single_domain_benchmark.py")
               for value in command))
    assert "--experiment-config" not in benchmark
    namelist = Path(benchmark[benchmark.index("--namelist-input") + 1])
    target = HrrrTargetDomain.legacy_500x500()
    assert target.dx_m < 1000.0
    vertical = explicit_vertical_from_wrf_namelist(
        namelist, expected_nz=target.nz, context="namelist-only witness")
    _exp, tables = resolve_root_experiment(
        target=target, vertical=vertical, namelist_input=namelist,
        start_time=datetime(2026, 7, 18, 17), run_seconds=3600.0,
        physics_profile=MORRISON_PROFILE_ID)
    checked = production.resolve_static_highres(
        tables, source=str(namelist), base_dir=namelist.parent)
    assert checked is not None and built.echo() == checked.echo()
    grid = target.grid()
    receipt = {"highres": {
        "status": "APPLIED", "config": built.echo(), "case_date": case_date,
        "grid": production._grid_identity(grid, 1)}}
    production.require_prepared_highres(
        receipt, grid, config=checked, domain_id=1,
        case_date=datetime(2026, 7, 18, 17).date())
    # The baseline receipt the old route built is what the benchmark refused.
    with pytest.raises(ValueError, match="do not bind"):
        production.require_prepared_highres(
            {}, grid, config=checked, domain_id=1,
            case_date=datetime(2026, 7, 18, 17).date())


def _assert_source_leads_and_model_time(commands, output, physics_profile):
    series = output / "native" / "hrrr-f12-f13-series.tsv"
    assert [line.split("\t", 1)[0]
            for line in series.read_text(encoding="utf-8").splitlines()] \
        == ["12", "13"]
    benchmark = next(
        command for command in commands
        if any(value.endswith("hrrr_single_domain_benchmark.py")
               for value in command))
    assert benchmark[benchmark.index("--forecast-start-hour") + 1] == "12"
    assert benchmark[benchmark.index("--forecast-end-hour") + 1] == "13"
    assert benchmark[benchmark.index("--physics-profile") + 1] \
        == physics_profile
    assert benchmark[benchmark.index("--history-interval-seconds") + 1] \
        == "3600.0"
    assert "--io-mode" not in benchmark
    export = next(command for command in commands if "woof.wrf_direct" in command)
    assert export[export.index("--valid-time") + 1] == "2026-07-18_17:00:00"

    receipt = json.loads(
        (output / "public-wrapper-result.json").read_text(encoding="utf-8"))
    assert receipt["source_cycle"] == "2026-07-18T05:00:00"
    assert receipt["model_start_time"] == "2026-07-18T17:00:00"
    assert receipt["source_forecast_hours"] == [12, 13]
    assert receipt["model_forcing_hours"] == [0, 1]
    assert receipt["forcing_hours"] == [0, 1]
    assert receipt["history_interval_seconds"] == 3600.0
    assert receipt["physics"]["profile"] == physics_profile


def test_public_wrapper_rejects_mismatched_or_incomplete_physics_receipt():
    complete = {
        "physics": {
            "schema": "gpuwm-prepared-physics-profile-v1",
            "profile": MORRISON_PROFILE_ID,
            "hrrr_initialization": _cold_start_receipt(
                MORRISON_PROFILE_ID),
        },
    }
    assert prepare._validated_physics_receipt(
        complete, requested_profile=MORRISON_PROFILE_ID) \
        == complete["physics"]

    with pytest.raises(RuntimeError, match="differs from the request"):
        prepare._validated_physics_receipt(
            complete, requested_profile=NSSL2_PROFILE_ID)
    incomplete = json.loads(json.dumps(complete))
    del incomplete["physics"]["hrrr_initialization"][
        "state_source_absent_fields"]
    with pytest.raises(RuntimeError, match="cold-start evidence"):
        prepare._validated_physics_receipt(
            incomplete, requested_profile=MORRISON_PROFILE_ID)

    wrong_fields = json.loads(json.dumps(complete))
    wrong_fields["physics"]["hrrr_initialization"][
        "source_absent_wrf_fields"] = []
    with pytest.raises(RuntimeError, match="cold-start evidence"):
        prepare._validated_physics_receipt(
            wrong_fields, requested_profile=MORRISON_PROFILE_ID)

    false_exact = json.loads(json.dumps(complete))
    false_exact["physics"]["hrrr_initialization"][
        "state_source_absent_fields"]["ng"]["all_exact_expected"] = False
    with pytest.raises(RuntimeError, match="cold-start evidence"):
        prepare._validated_physics_receipt(
            false_exact, requested_profile=MORRISON_PROFILE_ID)

    wrong_bits = json.loads(json.dumps(complete))
    wrong_bits["physics"]["hrrr_initialization"][
        "state_source_absent_fields"]["ng"]["expected_uint32_bits"] = 1
    with pytest.raises(RuntimeError, match="cold-start evidence"):
        prepare._validated_physics_receipt(
            wrong_bits, requested_profile=MORRISON_PROFILE_ID)


def test_public_wrapper_holds_thompson_numbers_to_the_seed_rule():
    """nr and ni carry real.exe's seed evidence (A99, A110), every leg."""

    complete = {
        "physics": {
            "schema": "gpuwm-prepared-physics-profile-v1",
            "profile": THOMPSON_PROFILE_ID,
            "hrrr_initialization": _cold_start_receipt(THOMPSON_PROFILE_ID),
        },
    }
    fields = complete["physics"]["hrrr_initialization"][
        "state_source_absent_fields"]
    assert fields["nr"]["seeded_cells"] > 0
    assert fields["ni"]["seeded_cells"] > 0
    assert prepare._validated_physics_receipt(
        complete, requested_profile=THOMPSON_PROFILE_ID) \
        == complete["physics"]

    for name, key, value in (
            ("nr", "all_exact_expected_where_mass_is_zero", False),
            ("ni", "seeded_all_finite_above_zero", False),
            ("nr", "paired_mass_field", "qi"),
            ("ni", "seed_receipt", "rain_number_seed"),
            ("nr", "seeded_cells", -1),
            ("nr", "seeded_cells", True),
            ("ni", "expected_uint32_bits", 1),
            ("nr", "all_exact_expected", True)):
        broken = json.loads(json.dumps(complete))
        broken["physics"]["hrrr_initialization"][
            "state_source_absent_fields"][name][key] = value
        with pytest.raises(RuntimeError, match="cold-start evidence"):
            prepare._validated_physics_receipt(
                broken, requested_profile=THOMPSON_PROFILE_ID)


# ---------------------------------------------------------------------------
# The decoder this wrapper launches, and the disposition of the optional
# stock-WRF export.  Both are field failures: the first refused on a healthy
# wheel install, and the second threw away a completed preparation.
# ---------------------------------------------------------------------------

def _keeping_refusal(fake_run, error_text: str = ""):
    """A faked launcher for the seam that KEEPS what the child said.

    The WRF-arm export is launched through it rather than through the
    plain one, because its refusal has to be repeatable by the step that
    launched it, so a case that fakes ``_run`` fakes this beside it and
    hands back the same pair the real one does: the exit code and the
    child's own error text.
    """

    import subprocess

    def run(command, env, cwd=None):
        try:
            fake_run(command, env, cwd)
        except subprocess.CalledProcessError as error:
            return error.returncode, error_text
        return 0, ""

    return run


def _wrapper_case(tmp_path: Path, monkeypatch, *, export_returncode: int = 0,
                  export_error_text: str = ""):
    """One runnable wrapper invocation, with every subprocess faked.

    Returns ``(argv, commands, output)``: the argument vector, the list
    every faked launcher appends to, and the output root -- so a test can
    assert what was launched and what was written.

    ``export_error_text`` is what the faked converter wrote to its error
    stream, which the step that launched it has to be able to repeat.
    """

    import subprocess as subprocess_module

    source = tmp_path / "source"
    source.mkdir()
    for hour in (0, 1):
        (source / f"hrrr.t05z.wrfnatf{hour:02d}.grib2").write_bytes(b"atmos")
        (source / f"hrrr.t05z.soilf{hour:02d}.grib2").write_bytes(b"soil")
    source_manifest = source / "SHA256SUMS"
    source_manifest.write_text("fixture\n", encoding="ascii")
    static_cache = tmp_path / "static.npz"
    static_receipt = tmp_path / "static.json"
    namelist = tmp_path / "namelist.input"
    for path in (static_cache, static_receipt, namelist):
        path.write_bytes(b"fixture")
    authority = _configured_wrapper_input(tmp_path, namelist)
    decoder = tmp_path / "hrrr_grib2_bridge"
    decoder.write_bytes(b"decoder")
    monkeypatch.setattr(prepare, "_decoder", lambda _env: decoder)

    commands: list[list[str]] = []

    def fake_run(command, _env, cwd=None):
        commands.append(command)
        if any(str(value).endswith("hrrr_single_domain_benchmark.py")
               for value in command):
            outdir = Path(command[command.index("--outdir") + 1])
            outdir.mkdir(parents=True)
            (outdir / "report.json").write_text(json.dumps({
                "status": "PASS",
                "history_interval_seconds": 3600.0,
                "physics": {
                    "schema": "gpuwm-prepared-physics-profile-v1",
                    "resolved": resolved_run_settings(load_experiment(authority).root.run),
                    "profile": WSM6_PROFILE_ID,
                    "hrrr_initialization": _cold_start_receipt(
                        WSM6_PROFILE_ID),
                },
                "preparation": _cuda_preparation_receipt(),
                "pipeline": {"workers": {"requested": "8", "selected": 8}},
            }), encoding="utf-8")
        elif "woof.wrf_direct" in command and export_returncode:
            raise subprocess_module.CalledProcessError(
                export_returncode, command)
        elif "woof.wrf_direct" in command:
            # The fake converter behaves like the real atomic exporter:
            # exit 0 means the declared outputs and their manifest exist
            # (the fail-closed step verifies exactly that at PASS).
            export_output = Path(command[command.index("--output") + 1])
            export_output.mkdir(parents=True)
            for name in ("wrfinput_d01", "wrfbdy_d01"):
                (export_output / name).write_bytes(b"exported")
            (export_output / "manifest.json").write_text(json.dumps({
                "files": {"wrfinput_d01": {"sha256": "0" * 64},
                          "wrfbdy_d01": {"sha256": "0" * 64}},
            }), encoding="utf-8")

    monkeypatch.setattr(prepare, "_run", fake_run)
    monkeypatch.setattr(prepare, "_run_keeping_refusal",
                        _keeping_refusal(fake_run, export_error_text))
    output = tmp_path / "output"
    argv = [
        "--experiment-config", str(authority),
        "--source-root", str(source),
        "--source-manifest", str(source_manifest),
        "--source-manifest-sha256", "0" * 64,
        "--static-cache", str(static_cache),
        "--static-receipt", str(static_receipt),
        "--namelist-input", str(namelist),
        "--physics-profile", WSM6_PROFILE_ID,
        "--valid-time", "2026-07-18_05:00:00",
        "--run-seconds", "3600",
        "--history-interval-seconds", "3600",
        "--output-root", str(output),
    ]
    return argv, commands, output


def test_a_refused_stock_wrf_export_fails_the_command_keeping_the_work(
        tmp_path: Path, monkeypatch) -> None:
    """The 2026-08-04 ruling: a requested export that fails is fatal.

    History in two swings: the original ``check=True`` destroyed a
    completed preparation on any converter refusal; the warn-and-continue
    fix for that then PASS-logged the battery case run whose converter
    died on EMFILE -- no WRF-arm inputs, no manifest, one warning line
    (out/node19-shakedown/b04/B04-EXPORT-REFUSAL.txt).  Now the command
    fails LOUDLY, and the preparation's own artifacts stay on disk --
    the refusal costs a re-run of the export, never the preparation.
    """

    argv, commands, output = _wrapper_case(
        tmp_path, monkeypatch, export_returncode=1)
    with pytest.raises(RuntimeError, match="requested and failed"):
        prepare.main(argv)

    assert any("woof.wrf_direct" in command for command in commands)
    # The preparation completed before the export and stands on disk.
    report = (output / "native" / "preparation-report" / "report.json")
    assert report.is_file()
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "PASS"
    # No wrapper PASS receipt is written over the failure.
    assert not (output / "public-wrapper-result.json").exists()


def test_a_refused_export_repeats_what_the_converter_said(
        tmp_path: Path, monkeypatch) -> None:
    """The refusal carries the sentence, not a pointer to a stream.

    Measured on the shipped nowcast front door: the step said "fix the
    converter's refusal (quoted on stderr)", and the caller, which keeps
    the last lines of a failed stage, kept exactly this message and the
    frames under it while the converter's own sentence sat further up,
    outside that window.  A refusal that names its breakage has to carry
    the words with it.
    """

    said = ("stock-WRF wrfinput export for mp_physics=1 is not available "
            "on this route: no evidenced package contract is packaged for "
            "that selector.")
    frames = f'''Traceback (most recent call last):
  File 'x', line 1, in <module>
ValueError: {said}
'''
    argv, _commands, _output = _wrapper_case(
        tmp_path, monkeypatch, export_returncode=1,
        export_error_text=frames)
    with pytest.raises(RuntimeError) as refusal:
        prepare.main(argv)
    message = str(refusal.value)
    assert said in message
    assert "--skip-stock-wrf-export" in message
    # The frames are the child's, and they are not the reader's problem.
    assert "Traceback" not in message


def test_a_refused_export_says_so_when_the_converter_said_nothing(
        tmp_path: Path, monkeypatch) -> None:
    """A silent child is itself the thing to report, and is reported."""

    argv, _commands, _output = _wrapper_case(
        tmp_path, monkeypatch, export_returncode=1)
    with pytest.raises(RuntimeError) as refusal:
        prepare.main(argv)
    assert "wrote nothing to its error stream" in str(refusal.value)


def test_a_required_stock_wrf_export_still_fails_the_command(
        tmp_path: Path, monkeypatch) -> None:
    """The legacy flag is honored: failure is fatal with it too."""

    argv, _commands, _output = _wrapper_case(
        tmp_path, monkeypatch, export_returncode=1)
    with pytest.raises(RuntimeError, match="requested and failed"):
        prepare.main(argv + ["--require-stock-wrf-export"])


def test_skipping_the_stock_wrf_export_never_launches_it(
        tmp_path: Path, monkeypatch) -> None:
    argv, commands, output = _wrapper_case(tmp_path, monkeypatch)
    assert prepare.main(argv + ["--skip-stock-wrf-export"]) == 0
    assert not any("woof.wrf_direct" in command for command in commands)
    receipt = json.loads(
        (output / "public-wrapper-result.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["stock_wrf_export"]["status"] == "SKIPPED"
    assert receipt["wrf_input"] is None


def test_a_successful_stock_wrf_export_still_names_its_output(
        tmp_path: Path, monkeypatch) -> None:
    argv, commands, output = _wrapper_case(tmp_path, monkeypatch)
    assert prepare.main(argv) == 0
    assert any("woof.wrf_direct" in command for command in commands)
    receipt = json.loads(
        (output / "public-wrapper-result.json").read_text(encoding="utf-8"))
    assert receipt["stock_wrf_export"]["status"] == "PASS"
    assert receipt["wrf_input"] == str(output / "wrf-native-input")
    # PASS is earned: the receipt binds the verified outputs by name and
    # by the manifest's own digest.
    assert receipt["stock_wrf_export"]["files"] == [
        "wrfbdy_d01", "wrfinput_d01"]
    assert len(receipt["stock_wrf_export"]["manifest_sha256"]) == 64


def test_the_two_export_flags_are_mutually_exclusive(tmp_path: Path,
                                                     monkeypatch) -> None:
    argv, _commands, _output = _wrapper_case(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        prepare.main(argv + ["--skip-stock-wrf-export",
                             "--require-stock-wrf-export"])


def test_the_wrapper_resolves_the_decoder_the_shared_ladder_resolves(
        monkeypatch, tmp_path: Path) -> None:
    """Issue two, at the wrapper: a wheel has no cargo workspace.

    ``REPO`` is site-packages on a pip install, so the old
    ``REPO/tools/grib1_bridge/target/release/hrrr_grib2_bridge`` named a
    path that cannot exist there, and the cargo fallback ran in a
    directory that does not exist either.  Now it is the same ladder
    ``woof doctor`` reports, and the staged bridge wins.
    """

    from woof import bridges

    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    staged = tmp_path / "userdir"
    staged.mkdir()
    for variable in bridges.BRIDGE_ENV.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(bridges, "crate_dir", lambda: tmp_path / "no-crate")
    monkeypatch.setattr(bridges, "_package_parent", lambda: site_packages)
    monkeypatch.setattr(bridges, "default_bridge_dir", lambda: staged)
    monkeypatch.setattr(prepare, "BRIDGE_CRATE", tmp_path / "no-crate")

    # Negative control, watched firing: nothing staged, named refusal --
    # and it names ~/.woof/bridges rather than a cargo target directory.
    with pytest.raises(FileNotFoundError) as error:
        prepare._decoder({})
    assert str(staged) in str(error.value)

    decoder = _contract_carrying_decoder(
        staged / bridges.executable_name("hrrr_grib2_bridge"))
    assert prepare._decoder({}) == decoder.resolve()

    # And the ladder is not a name match: a staged binary that predates
    # this release's contract refuses here, at the production door, the
    # way it always did in the report.
    decoder.write_bytes(b"not a bridge")
    with pytest.raises(bridges.DecoderContractError, match="predates"):
        prepare._decoder({})


def test_an_explicit_decoder_override_is_honored_and_fails_loud(
        monkeypatch, tmp_path: Path) -> None:
    override = _contract_carrying_decoder(tmp_path / "my_bridge")
    monkeypatch.setenv("WOOF_HRRR_DECODER", str(override))
    assert prepare._decoder({}) == override.resolve()

    monkeypatch.setenv("WOOF_HRRR_DECODER", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="WOOF_HRRR_DECODER"):
        prepare._decoder({})


def test_requested_stock_wrf_export_failure_is_fatal(tmp_path):
    """The fail-closed contract, driven by an injected converter death.

    The 2026-08-04 battery case run's converter subprocess died on EMFILE
    and the old warn-and-continue posture PASS-logged a preparation with
    no WRF-arm inputs and no manifest.  A requested export that fails now
    refuses loudly, whatever the converter's exit reason -- the subprocess
    boundary cannot tell an environmental death from a typed refusal.
    """

    import sys

    with pytest.raises(RuntimeError, match="requested and failed"):
        prepare._stock_wrf_export(
            [sys.executable, "-c", "import sys; sys.exit(24)"],
            dict(os.environ), skip=False, required=False,
            output=tmp_path / "wrf-native-input")

    # --skip-stock-wrf-export remains the deliberate opt-out, recorded.
    receipt = prepare._stock_wrf_export(
        [sys.executable, "-c", "import sys; sys.exit(24)"],
        dict(os.environ), skip=True, required=False,
        output=tmp_path / "wrf-native-input")
    assert receipt == {"status": "SKIPPED",
                       "reason": "--skip-stock-wrf-export",
                       "required": False}


def test_stock_wrf_export_pass_is_earned_by_manifested_outputs(tmp_path):
    """PASS requires the declared outputs to exist and hash-manifest."""

    import sys

    output = tmp_path / "wrf-native-input"
    quiet = [sys.executable, "-c", "pass"]

    # Exit 0 with no manifest at all: refused by name.
    with pytest.raises(RuntimeError, match="left no manifest"):
        prepare._stock_wrf_export(quiet, dict(os.environ), skip=False,
                                  required=False, output=output)

    # A manifest whose declared file is missing on disk: refused by name.
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps({
        "files": {"wrfinput_d01": {"sha256": "0" * 64},
                  "wrfbdy_d01": {"sha256": "0" * 64}},
    }), encoding="utf-8")
    (output / "wrfinput_d01").write_bytes(b"present")
    with pytest.raises(RuntimeError, match="missing on disk.*wrfbdy_d01"):
        prepare._stock_wrf_export(quiet, dict(os.environ), skip=False,
                                  required=False, output=output)

    # Every declared file present: PASS, with the manifest's own digest.
    (output / "wrfbdy_d01").write_bytes(b"present")
    receipt = prepare._stock_wrf_export(quiet, dict(os.environ), skip=False,
                                        required=False, output=output)
    assert receipt["status"] == "PASS"
    assert receipt["files"] == ["wrfbdy_d01", "wrfinput_d01"]
    assert receipt["manifest_sha256"] == hashlib.sha256(
        (output / "manifest.json").read_bytes()).hexdigest()
    assert "REFUSED" not in prepare.STOCK_WRF_EXPORT_STATES


def _source_and_decoder_fixture(tmp_path, monkeypatch, hours=(0, 1, 2)):
    """Deterministic source files and decoder for native preparation."""
    source = tmp_path / "source"
    source.mkdir()
    for hour in hours:
        for product in ("wrfnat", "soil"):
            (source / f"hrrr.t05z.{product}f{hour:02d}.grib2").write_bytes(
                f"fixture:{product}:{hour}".encode())
    manifest = source / "SHA256SUMS"
    manifest.write_text("".join(f"{_file_sha256(path)}  {path.name}\n"
                        for path in sorted(source.iterdir())), encoding="utf-8")
    monkeypatch.setenv("WOOF_HRRR_DECODER", str(_fixture_decoder(tmp_path)))
    return source, manifest


def test_cuda_native_preparation_runs_its_host_steps_on_the_requested_workers(
        tmp_path, monkeypatch):
    """Real CUDA native preparation with --preprocess-workers, end to end.

    The wrapper used to refuse the pair after the front door forwarded
    it.  Two forcing hours past the start on two preparation slots send
    the boundary hours through the spawned worker pool, whose backends
    must carry the same host-step count as the controller's for their
    receipts to agree.
    """
    import cupy

    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA device")
    target, static_cache, static_receipt, domain, namelist = (
        _real_wrapper_inputs(tmp_path))
    vertical = explicit_vertical_from_wrf_namelist(
        namelist, expected_nz=target.nz, context="cuda host workers test")
    authority = _configured_wrapper_input(
        tmp_path, namelist, target=target, write_namelist=False,
        vertical=vertical)
    source, manifest = _source_and_decoder_fixture(tmp_path, monkeypatch)
    try:
        resolve_cpu_bridge()
    except FileNotFoundError:
        pytest.skip("native CPU preprocessing bridge is not installed")
    output = tmp_path / "prepared"
    assert prepare.main([
        "--source-root", str(source), "--source-manifest", str(manifest),
        "--source-manifest-sha256", _file_sha256(manifest),
        "--static-cache", str(static_cache), "--static-receipt", str(static_receipt),
        "--domain-spec", str(domain), "--namelist-input", str(namelist),
        "--experiment-config", str(authority), "--cycle", "2026-07-18_05:00:00",
        "--run-seconds", "7200", "--pipeline-workers", "1", "--prepare-workers", "2",
        "--preprocess-backend", "cuda", "--preprocess-workers", "2",
        "--skip-stock-wrf-export", "--output-root", str(output)]) == 0
    report = json.loads(
        (output / "native/preparation-report/report.json").read_text())
    preparation = report["preparation"]
    assert preparation["preprocess_backend"]["backend"] == "cuda"
    assert preparation["preprocess_backend"]["masked_surface_chain"][
        "workers"] == 2
    budget = preparation["preprocess_worker_budget"]
    assert budget["host_step_native_workers"] == 2
    assert budget["peak_active_native_workers"] == 2
    assert budget["requested_total_native_workers"] == 2


@pytest.mark.parametrize("selector", ("cpu", None))
def test_cpu_native_preparation_checks_every_slot_against_the_controller(
        tmp_path, monkeypatch, selector):
    """Real CPU native preparation on two slots of the worker budget.

    Two forcing hours past the start on two preparation slots split the
    CPU budget, so every boundary hour runs on its slot's share: mapped
    in the controller on that share and initialized in the spawned
    pool.  Each slot's receipt was refused against the controller's at
    the first boundary hour: the masked surface chain names the slot's
    share where the controller's names the whole budget, the slot began
    its own record of vertical routes, and a CPU that auto chose came
    back from every slot as one "named by the caller".  ``None`` is the
    bare default, auto, on a machine with no usable card.
    """

    target, static_cache, static_receipt, domain, namelist = (
        _real_wrapper_inputs(tmp_path))
    vertical = explicit_vertical_from_wrf_namelist(
        namelist, expected_nz=target.nz, context="cpu slot receipt test")
    authority = _configured_wrapper_input(
        tmp_path, namelist, target=target, write_namelist=False,
        vertical=vertical)
    source, manifest = _source_and_decoder_fixture(tmp_path, monkeypatch)
    # Auto must find no card, whether or not this box has one.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    try:
        bridge = resolve_cpu_bridge()
    except FileNotFoundError:
        pytest.skip("native CPU preprocessing bridge is not installed")
    selection = (() if selector is None else
                 ("--preprocess-backend", selector,
                  "--cpu-preprocess-bridge", str(bridge)))
    output = tmp_path / "prepared"
    assert prepare.main([
        "--source-root", str(source), "--source-manifest", str(manifest),
        "--source-manifest-sha256", _file_sha256(manifest),
        "--static-cache", str(static_cache), "--static-receipt", str(static_receipt),
        "--domain-spec", str(domain), "--namelist-input", str(namelist),
        "--experiment-config", str(authority), "--cycle", "2026-07-18_05:00:00",
        "--run-seconds", "7200", "--pipeline-workers", "1", "--prepare-workers", "2",
        "--preprocess-workers", "4", *selection,
        "--skip-stock-wrf-export", "--output-root", str(output)]) == 0
    report = json.loads(
        (output / "native/preparation-report/report.json").read_text())
    preparation = report["preparation"]
    receipt = preparation["preprocess_backend"]
    assert receipt["backend"] == "cpu"
    assert receipt["workers"] == 4
    assert receipt["masked_surface_chain"]["workers"] == 4
    assert receipt["selection"]["requested"] == (selector or "auto")
    budget = preparation["preprocess_worker_budget"]
    shares = sorted(
        (job["forecast_hour"], job["phase"], job["slot"],
         job["effective_native_workers"])
        for job in budget["effective_allocation_per_job"])
    assert shares == [
        (0, "full_domain_initialization", None, 4),
        (1, "boundary_initialization", 0, 2),
        (1, "boundary_mapping", 0, 2),
        (2, "boundary_initialization", 1, 2),
        (2, "boundary_mapping", 1, 2),
    ]


@pytest.mark.parametrize("surface,layers,pbl,sfclay", [(2, 4, 1, 1), (3, 6, 5, 91), (3, 9, 5, 91)])
def test_configured_native_preparation_publishes_selected_soil_and_physics(
        tmp_path, monkeypatch, surface, layers, pbl, sfclay):
    """Real CPU native preparation with fixture decoded source data; no forecast claim."""
    target, static_cache, static_receipt, domain, namelist = _real_wrapper_inputs(tmp_path)
    vertical = explicit_vertical_from_wrf_namelist(
        namelist, expected_nz=target.nz, context="configured soil test")
    changes = dict(sf_surface_physics=surface, num_soil_layers=layers,
                   bl_pbl_physics=pbl, sf_sfclay_physics=sfclay)
    authority = _configured_wrapper_input(tmp_path, namelist, target=target,
        write_namelist=False, vertical=vertical, changes=changes)
    source, manifest = _source_and_decoder_fixture(
        tmp_path, monkeypatch, hours=(0, 1))
    try:
        bridge = resolve_cpu_bridge()
    except FileNotFoundError:
        pytest.skip("native CPU preprocessing bridge is not installed")
    output = tmp_path / "prepared"
    assert prepare.main([
        "--source-root", str(source), "--source-manifest", str(manifest),
        "--source-manifest-sha256", _file_sha256(manifest),
        "--static-cache", str(static_cache), "--static-receipt", str(static_receipt),
        "--domain-spec", str(domain), "--namelist-input", str(namelist),
        "--experiment-config", str(authority), "--cycle", "2026-07-18_05:00:00",
        "--run-seconds", "3600", "--pipeline-workers", "1", "--prepare-workers", "1",
        "--preprocess-backend", "cpu", "--preprocess-workers", "1",
        "--cpu-preprocess-bridge", str(bridge), "--skip-stock-wrf-export",
        "--output-root", str(output)]) == 0
    wrapper = json.loads((output / "public-wrapper-result.json").read_text())
    assert wrapper["portable_bundle"] is not None, wrapper.get("portable_bundle_refusal")
    report = json.loads((output / "native/preparation-report/report.json").read_text())
    assert report["physics"]["profile"] is None
    poisoned = copy.deepcopy(report)
    poisoned["physics"]["resolved"]["sf_sfclay_physics"] = 999
    with pytest.raises(RuntimeError, match="resolved physics differs"):
        prepare._validated_physics_receipt(poisoned, requested_profile=None,
            expected_selection=load_experiment(authority).root.run)
    for name, value in changes.items():
        assert report["physics"]["resolved"][name] == value
    cache = output / "native/prepared-cache"
    header = json.loads((cache / "header.json").read_text())
    reader = PreparedCacheReader(cache, expected_identity=header["identity"])
    reader.verify_all()
    for field in ("TSLB", "SMOIS", "SH2O"):
        data = reader.read_array("surface/" + field)
        assert data.shape == (layers, target.ny, target.nx)
        assert np.isfinite(data).all()


def test_the_root_preparer_says_each_of_its_steps(
        tmp_path: Path, monkeypatch, capsys) -> None:
    """The breakage: a single-domain HRRR preparation wrote no step record.

    `woof go` reads this program's output live and puts each
    ``GPUWM_PREP_EVENT`` line on the run's stream; with none written,
    a plain single-domain HRRR run page showed only the stage for the
    whole preparation.
    """

    from woof.prep_progress import step_record

    capsys.readouterr()
    _run_public_wrapper_on_fake_children(tmp_path, monkeypatch, WSM6_PROFILE_ID)
    records = [record for record in map(step_record, capsys.readouterr().err.splitlines())
               if record is not None]

    assert [(record["stage"], record["event"]) for record in records] == [
        ("root_static", "started"), ("root_static", "finished"),
        ("root_prepare", "started"), ("root_prepare", "finished"),
        ("wrf_export", "started"), ("wrf_export", "finished"),
    ]
    assert records[-1]["outcome"] == "produced"
