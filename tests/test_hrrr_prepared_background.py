"""HRRR as a prepared background for the cycling radar-DA driver.

The bundle tests here are deliberately WRITER-TO-READER: the fixture
builds a prepared cache with the shipped cache writer, publishes the
portable authorities with the shipped bundle writer
(:mod:`woof.hrrr_prepared_bundle`), and then admits the result through
the front door the driver actually calls
(``preflight_prepared_forecast``).  Nothing here re-spells a schema, so
the two sides cannot drift into agreeing with each other's fixtures
while disagreeing with each other.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import woof.prepared_single_domain_forecast as runner  # noqa: E402
from woof.experiment import VerticalConfig, load_experiment  # noqa: E402
from woof.hrrr_route_inputs import (  # noqa: E402
    ROUTE_DEFAULT_PHYSICS_PROFILE,
)
from woof.experiment_document import (  # noqa: E402
    ExperimentDocumentError, publish_experiment_document,
    render_experiment_document)
from woof.hrrr_prepared_bundle import (  # noqa: E402
    HrrrBundleError, publish_hrrr_prepared_bundle)
from woof.ingest.hrrr_target import HrrrTargetDomain  # noqa: E402
from woof.ingest.prepared_cache import (  # noqa: E402
    PREPARED_CACHE_SCHEMA, _array_sha256, prepared_cache_identity,
    prepared_domain_config_identity)

from hrrr_single_domain_benchmark import (  # noqa: E402
    _experiment, _experiment_tables, _initial_hrrr_microphysics_receipt)
from woof.physics_compat import THOMPSON_RTE_RRTMGP_PROFILE_ID  # noqa: E402


#: The suite these bundles are built with.  It is the ROUTE's own
#: default rather than a literal, because the fixtures below build
#: their experiment through ``_experiment_tables`` -- which binds
#: that default -- and every downstream stage is handed the same
#: name.  Pinning a literal here is how the two halves drifted when
#: 1.8 flipped the default.
PROFILE = ROUTE_DEFAULT_PHYSICS_PROFILE

#: That default is a Thompson suite, so every bundle binds the Thompson
#: table authority; on a development tree the tables resolve to the
#: checkout's pinned set whatever the installed recast-woof-data carries, and
#: elsewhere to the machine's staged set (tests/conftest.py says why).
pytestmark = pytest.mark.usefixtures("pinned_thompson_tables")

CYCLE = datetime(2026, 8, 5, 4)
SOURCE_HOURS = (0, 1, 2)
NZ = 8


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _on_the_baseline_terrain(tables: dict) -> dict:
    """Declare the 30-arc-second terrain these fixtures stand in for.

    The synthetic preparations below never build terrain: their static
    cache is a few placeholder bytes.  Their 1 km target would otherwise
    take the grid-spacing default
    (:data:`woof.static.highres_production.HIGHRES_DEFAULT_BY_DX`), which
    a real preparation builds and records and these stand-ins do not, so
    the configuration says what they model.
    """
    tables["static"] = {"highres": {"enabled": False,
                                    "cache_root": "highres-cache"}}
    return tables


def _hrrr_experiment(*, run_seconds: float = 7200.0,
                     history_interval_seconds: float = 900.0,
                     configuration_profile: str = PROFILE,
                     shared: dict | None = None):
    vertical = VerticalConfig(
        eta_levels=tuple(1.0 - index / NZ for index in range(NZ + 1)),
        p_top=5000.0, hybrid_opt=2, etac=0.2)
    target = dataclasses.replace(
        HrrrTargetDomain.legacy_500x500(), nz=NZ, nx=48, ny=48)
    kwargs = dict(run_seconds=run_seconds, target=target,
                  start_time=CYCLE + timedelta(hours=SOURCE_HOURS[0]),
                  history_interval_seconds=history_interval_seconds,
                  physics_profile=configuration_profile)
    tables, _resolved = _experiment_tables(vertical, **kwargs)
    if shared:
        # A config-driven preparation (``physics_profile=None``): the
        # tables the namelist import would hand the preparer, with the
        # named selectors changed.
        from woof.experiment import build_experiment

        tables["shared"].update(shared)
        return tables, build_experiment(tables, "test:config-suite")
    return tables, _experiment(vertical, **kwargs)


def _write_prepared_cache(cache: Path, identity, forcing_hours, exp,
                          *, user_receipts=None) -> dict:
    """A minimal cache with the header shape the reader verifies.

    Written by hand rather than by ``write_prepared_cache`` because the
    real writer needs decoded HRRR state; the HEADER contract -- schema,
    identity, metadata, arrays, content digest -- is what preflight
    reads, and it is reproduced here exactly.
    """

    cache.mkdir(parents=True)
    array = np.asarray([1.0], dtype=np.float32)
    with (cache / "a00000.npy").open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    arrays = {"test": {"file": "a00000.npy", "shape": [1],
                       "dtype": "float32", "nbytes": 4,
                       "sha256": _array_sha256(array)}}
    metadata = {
        "user": {
            "initial_valid_time": exp.start_time.isoformat(),
            "last_valid_time": (
                exp.start_time
                + timedelta(hours=forcing_hours[-1])).isoformat(),
            "source_cycle": CYCLE.isoformat(),
            "source_forecast_hours": list(SOURCE_HOURS),
            "model_forcing_hours": list(forcing_hours),
            "forcing_hours": list(forcing_hours),
            "mapping_reports": {"hrrr": "synthetic-decode-report"},
        },
        "state_names": [], "coord_arrays": [],
        # A real cache records the hybrid pair beside the coefficient
        # arrays it restores (ingest/prepared_cache._coord_metadata dumps
        # every non-array field of the VerticalCoord), and the runner
        # holds the coordinate it adopts against exactly these.
        "coord_scalars": {
            "hybrid_opt": exp.vertical.hybrid_opt,
            "etac": exp.vertical.etac,
            "p_top": exp.vertical.p_top,
        },
        "base_arrays": [], "base_scalars": {},
        "met_fields": sorted(runner._REQUIRED_MET_FIELDS),
        "surface_fields": sorted(runner._CANONICAL_SURFACE_FIELDS),
        "lbc": {
            "spec_bdy_width": exp.root.run.spec_bdy_width,
            "spec_zone": exp.root.run.spec_zone,
            "relax_zone": exp.root.run.relax_zone,
            "intervals": [
                {"start_seconds": float(first * 3600),
                 "end_seconds": float(second * 3600),
                 "fields": list(runner._LBC_FIELDS)}
                for first, second in zip(forcing_hours, forcing_hours[1:])],
        },
        "setup_fingerprint": "test",
    }
    metadata["user"].update(user_receipts or {})
    basis = {"schema": PREPARED_CACHE_SCHEMA, "identity": identity,
             "metadata": metadata, "arrays": arrays, "payload_bytes": 4}
    header = {**basis, "status": "READY",
              "created_utc": "2026-08-05T00:00:00+00:00",
              "content_sha256": hashlib.sha256(
                  _canonical(basis).encode()).hexdigest()}
    (cache / "header.json").write_text(
        json.dumps(header, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    return header


@dataclasses.dataclass
class _Bundle:
    root: Path
    handoff: dict
    experiment: object
    run_seconds: float
    history_interval_seconds: float


def _hrrr_bundle(tmp_path: Path, *, run_seconds: float = 7200.0,
                 history_interval_seconds: float = 900.0,
                 physics_profile: str | None = PROFILE,
                 forcing_hours=(0, 1, 2),
                 rendered_wps: bool = False,
                 publish_cycle: datetime | None = None,
                 scientific_identity: dict | None = None,
                 user_receipts: dict | None = None,
                 configuration_profile: str = PROFILE,
                 shared: dict | None = None) -> _Bundle:
    tables, exp = _hrrr_experiment(
        run_seconds=run_seconds,
        history_interval_seconds=history_interval_seconds,
        configuration_profile=configuration_profile, shared=shared)
    _on_the_baseline_terrain(tables)
    root = tmp_path / "prepared"
    native = root / "native"
    bridge_dir = native / "native-bridge"
    bridge_dir.mkdir(parents=True)
    bridge = bridge_dir / "SHA256SUMS"
    bridge.write_bytes(b"decoded-bridge-manifest\n")

    static = root / "native-static.npz"
    static.write_bytes(b"hash-bound-hrrr-static")
    static_receipt = root / "native-static-receipt.json"
    static_receipt.write_text('{"schema": "test"}\n', encoding="utf-8")
    geometry = root / "native-geometry-receipt.json"
    geometry.write_text(json.dumps({
        "schema": "gpuwm-native-static-direct-v1", "status": "PASS",
        "geometry": {"test": "hrrr"},
        "cache": {"path": static.name, "bytes": static.stat().st_size,
                  "sha256": _sha256(static)},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source_manifest = tmp_path / "SHA256SUMS"
    source_manifest.write_bytes(b"raw-hrrr-grib2-manifest\n")
    namelist_input = tmp_path / "namelist.input"
    namelist_input.write_text("&domains\n/\n", encoding="utf-8")
    wps_namelist = tmp_path / "namelist.wps"
    wps_namelist.write_text("&share\n/\n", encoding="utf-8")

    source_identity = {
        "installed_wheel": None,
        "source_sha256": {
            name: hashlib.sha256(name.encode()).hexdigest()
            for name in runner._HRRR_DECODE_SOURCES},
        "source_cycle": CYCLE.isoformat(),
        "model_start_time": exp.start_time.isoformat(),
        "source_forecast_hours": list(SOURCE_HOURS),
        "model_forcing_hours": list(forcing_hours),
    }
    source_identity.update(scientific_identity or {})
    identity = prepared_cache_identity(
        bridge_manifest_sha256=_sha256(bridge),
        source_manifest_sha256=_sha256(source_manifest),
        static_cache_sha256=_sha256(static),
        namelist_sha256=_sha256(namelist_input),
        domain_config=exp.root, forcing_hours=forcing_hours,
        source_identity=source_identity)
    _write_prepared_cache(
        native / "prepared-cache", identity, forcing_hours, exp,
        user_receipts=user_receipts)

    # Published by the benchmark's own renderer, exactly as the
    # preparation does it, then bound by the bundle writer.
    config = publish_experiment_document(
        root / "experiment.toml", tables, exp)

    handoff = publish_hrrr_prepared_bundle(
        output_root=root,
        prepared_cache=native / "prepared-cache",
        static_cache=static, static_receipt=static_receipt,
        geometry_receipt=geometry, bridge_manifest=bridge,
        namelist_input=namelist_input,
        # ``None`` is the ROUTE's own case: the native preparation is
        # driven by a target domain and has no namelist.wps to copy, so
        # the writer renders one from this experiment.
        wps_namelist=(None if rendered_wps else wps_namelist),
        source_manifest=source_manifest,
        experiment_config=config,
        source_cycle=publish_cycle or CYCLE,
        source_forecast_hours=SOURCE_HOURS,
        model_forcing_hours=forcing_hours,
        preprocessing={"schema": "gpuwm-preprocess-implementation-v2",
                       "backend": "cpu",
                       "implementation": "test-native-preprocess",
                       "workers": 2},
        source_identity=source_identity,
        physics_profile=physics_profile)
    return _Bundle(root=root, handoff=handoff, experiment=exp,
                   run_seconds=run_seconds,
                   history_interval_seconds=history_interval_seconds)


def _bind_synthetic_geometry(monkeypatch):
    """Stand in for the Lambert/static verification the fixture cannot do."""

    monkeypatch.setattr(
        runner, "validate_native_lambert_contract",
        lambda exp, wps, source_name=None: object())
    monkeypatch.setattr(
        runner, "verify_native_static_receipt",
        lambda receipt, static, grid, cfg: {"status": "PASS"})
    monkeypatch.setattr(
        runner, "load_native_static_cache",
        lambda path, grid, ny, nx: {
            "LANDMASK": np.zeros((ny, nx)), "HGT_M": np.zeros((ny, nx))})


def _preflight(bundle: _Bundle, **overrides):
    kwargs = dict(
        source="hrrr", prepared_root=bundle.root,
        proof_sha256=bundle.handoff["proof_sha256"],
        source_manifest_sha256=bundle.handoff["source_manifest_sha256"],
        prepared_content_sha256=bundle.handoff["prepared_content_sha256"],
        experiment_config=Path(bundle.handoff["experiment_config"]),
        wps_namelist=Path(bundle.handoff["wps_namelist"]),
        physics_profile=bundle.handoff["physics_profile"],
        run_seconds=bundle.run_seconds,
        history_interval_seconds=bundle.history_interval_seconds)
    kwargs.update(overrides)
    return runner.preflight_prepared_forecast(**kwargs)


# ---------------------------------------------------------------------
# the experiment document the whole handoff rests on
# ---------------------------------------------------------------------

def test_published_experiment_document_reloads_to_the_prepared_identity(
        tmp_path):
    """The bridge between a programmatic preparation and a config-driven run.

    The native HRRR route builds its experiment in code.  The prepared
    cache stores ``domain_config`` and every restore compares it by
    strict equality, so the document this publishes has to reload to
    exactly that -- not approximately, and not to a hand-written config
    that happens to agree today.
    """

    tables, exp = _hrrr_experiment()
    path = publish_experiment_document(
        tmp_path / "experiment.toml", tables, exp)
    reloaded = load_experiment(path)
    assert (prepared_domain_config_identity(reloaded.root)
            == prepared_domain_config_identity(exp.root))
    assert reloaded.start_time == exp.start_time


def test_experiment_document_refuses_an_unrenderable_value():
    tables, _exp = _hrrr_experiment()
    tables["experiment"]["name"] = object()
    with pytest.raises(ExperimentDocumentError, match="cannot render"):
        render_experiment_document(tables)


def test_experiment_document_leaves_no_file_when_the_reload_disagrees(
        tmp_path, monkeypatch):
    """A document that does not reload must not be left for the next stage."""

    tables, exp = _hrrr_experiment()
    tables["shared"]["p_top"] = 4999.0        # not what `exp` was built with
    target = tmp_path / "experiment.toml"
    with pytest.raises(ExperimentDocumentError, match="does not reload"):
        publish_experiment_document(target, tables, exp)
    assert not target.exists()


# ---------------------------------------------------------------------
# the bundle: published by the shipped writer, admitted by the front door
# ---------------------------------------------------------------------

def test_the_route_renders_the_namelist_it_has_no_file_for(tmp_path):
    """The native route's missing authority, produced from what it holds.

    A native HRRR preparation is driven by a strict target domain, not by
    a ``namelist.wps`` -- so there was no such file to publish, and the
    portable authorities were therefore opt-in behind a flag that made
    the caller supply one.  A finished tree could then exist that no
    forecast front door would accept, which is exactly what the Linux
    shakeout produced on 2026-08-18.

    The numbers were never missing, only the file.  This asserts the
    rendered namelist reads back as the experiment's own geometry
    through the FORECAST STAGE'S parser, not through a fixture of its
    own: same grids, same order, same values.
    """

    from woof.static.projection import (
        grids_from_projection_config, grids_from_wps_namelist)

    bundle = _hrrr_bundle(tmp_path, rendered_wps=True)
    rendered = bundle.root / "namelist.wps"
    assert rendered.is_file()
    assert Path(bundle.handoff["wps_namelist"]) == rendered

    observed = grids_from_wps_namelist(rendered)
    expected = grids_from_projection_config(bundle.experiment)
    assert len(observed) == len(expected) == 1
    for name in ("ref_lat", "ref_lon", "truelat1", "truelat2", "stand_lon",
                 "dx", "dy", "e_we", "e_sn", "known_x", "known_y"):
        assert getattr(observed[0], name) == getattr(expected[0], name), name


def test_preflight_admits_a_bundle_whose_namelist_was_rendered(
        tmp_path, monkeypatch):
    """Rendered or supplied, the front door cannot tell and must not.

    The bundle writer binds ``namelist.wps`` by digest either way, so
    this is the assertion that a default `woof prep --source hrrr` tree
    -- one where nobody hand-wrote a namelist -- is admitted by the same
    preflight a hand-supplied one is.
    """

    _bind_synthetic_geometry(monkeypatch)
    bundle = _hrrr_bundle(tmp_path, rendered_wps=True)
    inputs = _preflight(bundle)
    assert Path(inputs.wps_namelist) == bundle.root / "namelist.wps"


def test_sim_accepts_a_published_hrrr_tree_and_composes_its_runner_line(
        tmp_path):
    """The other direction of the shakeout's refusal: this tree RUNS.

    ``woof sim`` reads the bundle's own document for source and layout
    and relays the three digests the single-domain runner binds.  Every
    path it needs is inside the tree the preparation wrote, so the two
    commands a reader types name no file they have to produce
    themselves.
    """

    from woof import stage_cli

    bundle = _hrrr_bundle(tmp_path, rendered_wps=True)
    resolved = stage_cli.resolve_bundle(bundle.root)
    assert resolved["source"] == "hrrr"
    assert resolved["layout"] == "single"

    command = stage_cli.sim_command(
        resolved,
        experiment_config=bundle.root / "experiment.toml",
        wps_namelist=bundle.root / "namelist.wps",
        outdir=tmp_path / "out")
    # The digests are the ones the publisher handed back, relayed rather
    # than recomputed by this seam.
    assert (command[command.index("--proof-sha256") + 1]
            == bundle.handoff["proof_sha256"])
    assert (command[command.index("--source-manifest-sha256") + 1]
            == bundle.handoff["source_manifest_sha256"])
    assert (command[command.index("--prepared-content-sha256") + 1]
            == bundle.handoff["prepared_content_sha256"])
    assert command[command.index("--source") + 1] == "hrrr"


def test_preflight_admits_a_published_hrrr_bundle(tmp_path, monkeypatch):
    _bind_synthetic_geometry(monkeypatch)
    bundle = _hrrr_bundle(tmp_path)
    inputs = _preflight(bundle)
    assert inputs.source == "hrrr"
    assert inputs.layout == runner.HRRR_DIRECT_LAYOUT
    assert inputs.forcing_hours == (0, 1, 2)
    assert inputs.experiment.root.run.nx == 48


def test_a_scheme_with_no_stock_wrf_contract_publishes_and_runs(
        tmp_path, monkeypatch):
    """WDM6 has no stock-WRF package contract, and that is about exporting.

    ``stock_wrf_physics_inventory`` refuses mp_physics=16 because a
    wrfinput written for an unchanged WRF v4.6.1 would under-declare
    WDM6's own package; it says nothing about running WDM6 here.  The
    bundle writer hashed that stock contract into the proof's export slot
    anyway, so publication raised, the preparation published no portable
    bundle, and ``woof run-plan`` stopped on the shipped Grell-Freitas
    suite (WDM6) at prepare.  The slot now records the stock export as
    refused for that scheme, and the forecast reader admits the bundle.
    """

    from woof.wrf_physics_inventory import supported_stock_wrf_mp_physics

    assert 16 not in supported_stock_wrf_mp_physics()
    _bind_synthetic_geometry(monkeypatch)
    # 4193eb0da, development branch, selects the coupled MP28 suite at
    # the native default. This configured WDM6 export control starts from
    # the admitted uncoupled baseline, without MP28-only scalar or forks.
    bundle = _hrrr_bundle(tmp_path, physics_profile=None,
                          configuration_profile=THOMPSON_RTE_RRTMGP_PROFILE_ID,
                          # 188ffdf41, development branch, keeps the configured
                          # WDM6 export control explicit about MP28-only
                          # scalar mixing and neutral Thompson selectors.
                          shared={"mp_physics": 16, "scalar_pblmix": 0,
                                  "thompson_version": "wrf_461",
                                  "thompson_fork_snow_fall": "blend"})
    proof = json.loads(Path(bundle.handoff["proof"]).read_text(
        encoding="utf-8"))
    assert proof["stock_wrf_export"] == "optional"
    assert proof["export"]["status"] == "REFUSED"
    assert proof["export"]["unsupported"] == {"mp_physics": 16}
    assert "mp_physics=16" in proof["export"]["reason"]
    inputs = _preflight(bundle)
    assert inputs.experiment.root.run.mp_physics == 16
    assert inputs.experiment.root.run.scalar_pblmix == 0
    assert inputs.experiment.root.run.thompson_version == "wrf_461"
    assert inputs.export_source_receipt["status"] == "REFUSED"

    # A scheme WITH a contract keeps the READY slot it always had.
    control = _hrrr_bundle(tmp_path / "control")
    proof = json.loads(Path(control.handoff["proof"]).read_text(
        encoding="utf-8"))
    assert "stock_wrf_export" not in proof
    assert proof["export"]["status"] == "READY"


@pytest.mark.parametrize("shared", [None, {
    "mp_physics": 16, "scalar_pblmix": 0,
    "thompson_version": "wrf_461", "thompson_fork_snow_fall": "blend"}],
    ids=["ready-export", "optional-export"])
def test_preflight_checks_the_preprocessing_digest_with_or_without_export(
        tmp_path, monkeypatch, shared):
    """The proof's digest of its preprocessing receipt binds the
    preparation, not the stock-WRF export.  The reader checked it only
    inside the READY export branch, so a bundle for a scheme with no
    stock-WRF contract (optional export) ran with a digest that did not
    match its receipt."""

    _bind_synthetic_geometry(monkeypatch)
    # 4193eb0da, development branch: the default now carries MP28-only
    # scalar mixing. Use the same configured WDM6 control as above so this
    # test reaches the preprocessing-digest refusal it is intended to test.
    kwargs = {} if shared is None else dict(
        physics_profile=None, shared=shared,
        configuration_profile=THOMPSON_RTE_RRTMGP_PROFILE_ID)
    bundle = _hrrr_bundle(tmp_path, **kwargs)
    proof_path = bundle.root / "proof.json"
    proof = json.loads(proof_path.read_text())
    assert ("stock_wrf_export" in proof) == (shared is not None)
    proof["preprocessing_receipt_sha256"] = "0" * 64
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8", newline="\n")
    with pytest.raises(ValueError,
                       match="preprocessing receipt hash differs"):
        _preflight(bundle, proof_sha256=_sha256(proof_path))


def test_preflight_binds_the_three_hrrr_digests_separately(
        tmp_path, monkeypatch):
    """Bridge, source manifest and namelist are three values, not one.

    On the portable route the bridge and source manifest digests happen
    to be the same file, and the namelist digest is the experiment
    TOML's.  On this route all three differ, and reading any of them
    from the wrong place would produce an identity that cannot match a
    real HRRR cache.
    """

    _bind_synthetic_geometry(monkeypatch)
    bundle = _hrrr_bundle(tmp_path)
    manifest = json.loads(
        (bundle.root / "source-input-manifest.json").read_text())
    files = manifest["files"]
    digests = {files["bridge"]["sha256"], files["source_manifest"]["sha256"],
               files["namelist_input"]["sha256"],
               files["experiment_config"]["sha256"]}
    assert len(digests) == 4
    header = json.loads(
        (bundle.root / "native" / "prepared-cache"
         / "header.json").read_text())
    identity = header["identity"]
    assert identity["bridge_manifest_sha256"] == files["bridge"]["sha256"]
    assert identity["source_manifest_sha256"] \
        == files["source_manifest"]["sha256"]
    assert identity["namelist_sha256"] == files["namelist_input"]["sha256"]
    _preflight(bundle)                       # and the front door agrees


def test_preflight_refuses_a_cache_decoded_by_another_ingest(
        tmp_path, monkeypatch):
    _bind_synthetic_geometry(monkeypatch)
    bundle = _hrrr_bundle(tmp_path)
    proof_path = bundle.root / "proof.json"
    proof = json.loads(proof_path.read_text())
    name = runner._HRRR_DECODE_SOURCES[0]
    proof["source_sha256"][name] = "0" * 64
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="decode identity differs"):
        _preflight(bundle, proof_sha256=_sha256(proof_path))


def test_preflight_refuses_a_swapped_bridge_manifest(tmp_path, monkeypatch):
    _bind_synthetic_geometry(monkeypatch)
    bundle = _hrrr_bundle(tmp_path)
    bridge = bundle.root / "native" / "native-bridge" / "SHA256SUMS"
    bridge.write_bytes(b"a-different-decode\n")
    with pytest.raises(ValueError, match="native bridge manifest differs"):
        _preflight(bundle)


def test_preflight_refuses_a_swapped_namelist(tmp_path, monkeypatch):
    _bind_synthetic_geometry(monkeypatch)
    bundle = _hrrr_bundle(tmp_path)
    (bundle.root / "namelist.input").write_text(
        "&domains\ne_vert = 9\n/\n", encoding="utf-8")
    with pytest.raises(ValueError, match="namelist.input differs"):
        _preflight(bundle)


def test_preflight_refuses_a_non_hourly_hrrr_cadence(tmp_path, monkeypatch):
    _bind_synthetic_geometry(monkeypatch)
    bundle = _hrrr_bundle(tmp_path)
    proof_path = bundle.root / "proof.json"
    proof = json.loads(proof_path.read_text())
    proof["forcing_hours"] = [0, 2, 4]
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="not hourly"):
        _preflight(bundle, proof_sha256=_sha256(proof_path))


def test_preflight_refuses_an_hrrr_hierarchy_proof(tmp_path, monkeypatch):
    """The tree route is a division of labour, and the refusal says so."""

    _bind_synthetic_geometry(monkeypatch)
    bundle = _hrrr_bundle(tmp_path)
    proof_path = bundle.root / "proof.json"
    proof = json.loads(proof_path.read_text())
    proof["schema"] = "gpuwm-hrrr-native-hierarchy-proof-v1"
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="prepared_domain_tree_forecast"):
        _preflight(bundle, proof_sha256=_sha256(proof_path))


# ---------------------------------------------------------------------
# path separators: the identity is a set of repository paths, not a set
# of strings some machine happened to spell
# ---------------------------------------------------------------------
#
# Live defect, released 1.8.2, Windows: the prepared HRRR chain ALWAYS
# died at the forecast handoff with
#
#     ValueError: HRRR prepared cache decode identity omits
#     ['woof/hrrr_forecast.py', ...all ten...]
#
# and the artifact said why -- proof.json was keyed
# ``woof\hrrr_forecast.py``.  Both producers built their digest dict
# with ``str(path.relative_to(REPO))``, which is backslashed on Windows,
# while the reader looks names up by the forward-slash constants in
# ``_HRRR_DECODE_SOURCES``.  On Linux the two spellings are the same
# string, which is why every end-to-end HRRR proof -- all of them run on
# rentals or WSL -- passed while the Windows route had never worked.


def _identity_pair(*, separator: str = "/", digest: str = "a"):
    """An identity and the proof beside it, keyed with one separator."""

    digests = {
        name.replace("/", separator): (digest * 64)[:64]
        for name in runner._HRRR_DECODE_SOURCES
    }
    body = {
        "source_cycle": CYCLE.isoformat(),
        "model_start_time": CYCLE.isoformat(),
        "source_forecast_hours": list(SOURCE_HOURS),
        "model_forcing_hours": list(SOURCE_HOURS),
    }
    return ({"source_sha256": dict(digests), **body},
            {"source_sha256": dict(digests), **body})


def test_a_windows_sealed_cache_identity_validates():
    """The shape 1.8.2 sealed on Windows must still restore.

    Those caches are keyed with backslashes and cannot be rewritten,
    but their digests bind the same file BYTES as a POSIX-keyed cache
    of the same tree.  Refusing them would orphan real work over a
    property of the machine that wrote the JSON.
    """

    identity, proof = _identity_pair(separator="\\")
    assert any("\\" in key for key in identity["source_sha256"])
    assert runner._validate_hrrr_source_identity(identity, proof) is identity


def test_a_posix_keyed_identity_validates():
    identity, proof = _identity_pair(separator="/")
    assert runner._validate_hrrr_source_identity(identity, proof) is identity


def test_a_windows_cache_validates_against_a_posix_proof():
    """The mixed estate: a cache sealed on one machine, read on another."""

    identity, _ = _identity_pair(separator="\\")
    _, proof = _identity_pair(separator="/")
    assert runner._validate_hrrr_source_identity(identity, proof) is identity


def test_an_absent_decode_source_is_still_refused():
    """Normalizing keys must not turn a missing file into a present one."""

    identity, proof = _identity_pair(separator="\\")
    dropped = runner._HRRR_DECODE_SOURCES[0].replace("/", "\\")
    del identity["source_sha256"][dropped]
    del proof["source_sha256"][dropped]
    with pytest.raises(ValueError, match="decode identity omits"):
        runner._validate_hrrr_source_identity(identity, proof)


def test_a_different_ingest_is_still_refused_across_separators():
    """The bar the leniency must not cross.

    Reading both sides POSIX-normalized is only safe if a genuine
    digest difference still lands.  Same file, two spellings, two
    digests: refused.
    """

    identity, _ = _identity_pair(separator="\\", digest="a")
    _, proof = _identity_pair(separator="/", digest="b")
    with pytest.raises(ValueError, match="decode identity differs"):
        runner._validate_hrrr_source_identity(identity, proof)


def test_one_file_under_two_spellings_is_refused():
    """A merge would silently drop one of two digests for one file."""

    identity, proof = _identity_pair(separator="/")
    name = runner._HRRR_DECODE_SOURCES[0]
    identity["source_sha256"][name.replace("/", "\\")] = "c" * 64
    with pytest.raises(ValueError, match="two path spellings"):
        runner._validate_hrrr_source_identity(identity, proof)


def test_the_identity_key_expression_is_posix_on_a_windows_path():
    """Pins the expression itself, so a Linux runner catches a regression.

    ``str()`` of a relative ``PureWindowsPath`` is the shipped bug,
    evaluated identically on every platform.  This is the assertion
    that would have failed in CI before a Windows user ever ran it.
    """

    from pathlib import PureWindowsPath

    repo = PureWindowsPath(r"C:\gpuwm")
    path = repo / "woof" / "hrrr_forecast.py"
    assert path.relative_to(repo).as_posix() == "woof/hrrr_forecast.py"
    assert str(path.relative_to(repo)) == "woof\\hrrr_forecast.py"


@pytest.mark.parametrize("module_name", [
    "hrrr_single_domain_benchmark", "hrrr_state_proof"])
def test_both_producers_emit_posix_identity_keys(module_name):
    """The real producers, on whatever platform this runs.

    Not a re-spelling of the expression above: this calls the shipped
    ``_source_identity`` and looks at the dict a proof gets written
    from, which is where the released defect actually lived.
    """

    import importlib

    module = importlib.import_module(module_name)
    digests = module._source_identity()["source_sha256"]
    offenders = [key for key in digests if "\\" in key]
    assert not offenders, (
        f"{module_name} keyed its serialized identity with backslashes, "
        f"which the forecast reader cannot look up: {offenders}")


def test_the_benchmark_producer_covers_every_name_the_reader_demands():
    """Producer and reader, checked against each other rather than a list.

    ``_HRRR_DECODE_SOURCES`` is the reader's demand; the benchmark's
    ``_source_identity`` is what a real cache is keyed with.  The
    released defect was these two agreeing on Linux and not on Windows,
    so the agreement itself is worth a test rather than only the
    spelling.
    """

    from hrrr_single_domain_benchmark import _source_identity

    keys = set(_source_identity()["source_sha256"])
    assert set(runner._HRRR_DECODE_SOURCES) <= keys


def test_bundle_writer_refuses_a_start_time_that_is_not_the_lead(tmp_path):
    """A cycle/lead pair that does not date the experiment is refused.

    The two vocabularies are the thing this route gets wrong most
    easily: an absolute NOAA lead of one cycle, and the model's own
    start.  A bundle whose cycle plus first lead is not the experiment
    start would produce forcing frames dated at times the run never
    reaches, so it is refused at publication rather than at restore.
    """

    with pytest.raises(HrrrBundleError, match="is not cycle"):
        _hrrr_bundle(tmp_path, publish_cycle=CYCLE + timedelta(hours=1))


# ---------------------------------------------------------------------
# the prepare front door: opting into the portable bundle
# ---------------------------------------------------------------------

def test_the_prepare_front_door_forwards_the_portable_opt_in(tmp_path):
    """``--wps-namelist`` on the single-domain HRRR route means one thing.

    It used to be refused outright ("not used by --source hrrr").  It
    now selects the PORTABLE publication, and omitting it has to leave
    the certified native command exactly as it was -- that second half
    is the reproducibility half and is asserted here beside the first.
    """

    import argparse

    from woof.source_cli import _hrrr_command, _required_hrrr_args

    def _args(**extra):
        base = dict(
            root_preparation=None, source_root=tmp_path,
            source_sha256s=tmp_path / "SHA256SUMS",
            source_sha256s_sha256="a" * 64,
            namelist_input=tmp_path / "namelist.input",
            valid_time="2026-08-05_04:00:00", output_root=tmp_path / "out",
            physics_profile=None, experiment_config=None, run_seconds=7200,
            forecast_start_hour=0, forecast_end_hour=2,
            history_interval_seconds=900.0, geog_root=None,
            static_cache=tmp_path / "s.npz",
            static_receipt=tmp_path / "s.json",
            domain_spec=tmp_path / "domain.json", prepare_workers=None,
            preprocess_backend=None, preprocess_workers=None,
            cpu_preprocess_bridge=None, sealed_prepared_cache=False,
            extend_root_preparation=None, ack=(), pipeline_workers=None,
            wps_namelist=None, child_workers=None, no_stock_wrf_export=False,
        )
        base.update(extra)
        return argparse.Namespace(**base)

    without = _hrrr_command(_args())
    assert "--wps-namelist" not in without

    with_wps = _hrrr_command(
        _args(wps_namelist=tmp_path / "namelist.wps"))
    assert "--wps-namelist" in with_wps
    # and the only difference is that pair
    index = with_wps.index("--wps-namelist")
    assert with_wps[:index] + with_wps[index + 2:] == without


def _run_wrapper(tmp_path, monkeypatch, *, publish: bool):
    """Drive ``tools/prepare_hrrr_wrf.py`` with a stand-in benchmark.

    The stand-in produces what the real benchmark produces -- the
    preparation report, the sealed bridge manifest, the prepared cache,
    and, when asked, the experiment authority rendered by the SHIPPED
    renderer.  Everything downstream of it is production code: the
    wrapper's own publication step and the bundle writer.
    """

    import tools.prepare_hrrr_wrf as prepare
    from test_hrrr_single_domain_benchmark import _decoded_native_hrrr_initialization

    run_seconds, history_seconds = 7200.0, 900.0
    tables, exp = _hrrr_experiment(
        run_seconds=run_seconds,
        history_interval_seconds=history_seconds)
    # 4193eb0da, development branch, changes the route default to MP28/GSD.
    # Generate the selected MP28 scheme's receipt with an explicitly authored
    # analyzed fixture through the real producer. This publication fixture
    # does not numerically qualify the full route's runtime configuration.
    from woof.ingest import real as real_initialization
    initialize_real = real_initialization.initialize_real

    def initialize_fixture_aerosol(snapshot, cfg, *args, **kwargs):
        # These explicitly authored analyzed aerosol carriers belong only
        # to the small decoded-source fixture. They pass through the real
        # vertical operator and cloudy-cell number closure, with no WIF
        # donor or missing-aerosol fallback supplying the receipt.
        fields = dict(snapshot.fields)
        for name, number in (("QNWFA", 1e8), ("QNIFA", 1e6)):
            fields[name] = np.full_like(fields["TT"], number)
            fields[name + "_SFC"] = fields[name][0].copy()
        snapshot = dataclasses.replace(snapshot, fields=fields)
        cfg = dataclasses.replace(cfg, mp28_aerosol_source="analysis")
        return initialize_real(snapshot, cfg, *args, **kwargs)

    with monkeypatch.context() as initialization_fixture:
        initialization_fixture.setattr(
            real_initialization, "initialize_real", initialize_fixture_aerosol)
        initialized = _decoded_native_hrrr_initialization(exp.root.run.mp_physics)
    cloudy = initialized.state.qc > 0
    assert np.count_nonzero(cloudy) == 48
    assert np.all(initialized.state.nwfa[cloudy] > 0)
    assert np.all(initialized.state.nifa[cloudy] > 0)
    assert np.all(initialized.state.nc[cloudy] > 0)
    cold_start_receipt = _initial_hrrr_microphysics_receipt(
        initialized.state, exp.root.run, initialized.hydrometeor_initialization)
    assert cold_start_receipt["state_source_absent_fields"]["nc"]["seeded_cells"] == 48
    _on_the_baseline_terrain(tables)
    forcing_hours = (0, 1, 2)

    source = tmp_path / "source"
    source.mkdir()
    for hour in SOURCE_HOURS:
        (source / f"hrrr.t04z.wrfnatf{hour:02d}.grib2").write_bytes(
            f"atmos-{hour}".encode())
        (source / f"hrrr.t04z.soilf{hour:02d}.grib2").write_bytes(
            f"soil-{hour}".encode())
    source_manifest = source / "SHA256SUMS"
    source_manifest.write_text("fixture\n", encoding="ascii")
    static_cache = tmp_path / "static.npz"
    static_receipt = tmp_path / "static.json"
    namelist = tmp_path / "namelist.input"
    wps = tmp_path / "namelist.wps"
    for path in (static_cache, static_receipt, namelist, wps):
        path.write_bytes(b"fixture")
    from woof.hrrr_route_inputs import render_namelist_input, target_domain
    from woof.hrrr_configuration import resolved_run_settings
    authority = tmp_path / "configuration.toml"
    authority.write_text(render_experiment_document(tables), encoding="utf-8")
    namelist.write_text(render_namelist_input(exp), encoding="utf-8")
    domain_spec = tmp_path / "target.json"
    domain_spec.write_text(json.dumps(target_domain(exp).to_payload()), encoding="utf-8")
    decoder = tmp_path / "hrrr_grib2_bridge"
    decoder.write_bytes(b"decoder")
    monkeypatch.setattr(prepare, "_decoder", lambda _env: decoder)
    output = tmp_path / "output"

    def fake_run(command, _env, cwd=None):
        if "woof.wrf_direct" in command:
            # What the stock-WRF export leaves behind.  It is simulated
            # because the step is fail-closed by ruling (2026-08-04):
            # a REQUESTED export must land a manifest and every file the
            # manifest declares, or prepare_hrrr_wrf refuses rather than
            # reporting PASS over outputs that do not exist.  A fixture
            # that runs the step and writes nothing is asking for that
            # refusal.
            export_out = Path(command[command.index("--output") + 1])
            export_out.mkdir(parents=True, exist_ok=True)
            names = ("wrfinput_d01", "wrfbdy_d01")
            for name in names:
                (export_out / name).write_bytes(name.encode() + b"\n")
            (export_out / "manifest.json").write_text(json.dumps({
                "schema": "gpuwm-wrf-direct-export-v1",
                "files": {
                    name: {"bytes": (export_out / name).stat().st_size,
                           "sha256": _sha256(export_out / name)}
                    for name in names
                },
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return
        if any(str(value).endswith("write_hrrr_native_geometry_receipt.py")
               for value in command):
            receipt = Path(command[command.index("--output") + 1])
            static = Path(command[command.index("--static-cache") + 1])
            receipt.write_text(json.dumps({
                "schema": "gpuwm-native-static-direct-v1", "status": "PASS",
                "geometry": {"test": "hrrr"},
                "cache": {"path": static.name,
                          "bytes": static.stat().st_size,
                          "sha256": _sha256(static)},
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return
        if not any(str(value).endswith("hrrr_single_domain_benchmark.py")
                   for value in command):
            return
        # what the real benchmark leaves behind
        if "--publish-experiment-config" in command:
            config = Path(
                command[command.index("--publish-experiment-config") + 1])
            publish_experiment_document(config, tables, exp)
        bridge = Path(command[command.index("--bridge") + 1]) / "SHA256SUMS"
        bridge.parent.mkdir(parents=True, exist_ok=True)
        bridge.write_bytes(b"sealed-bridge\n")
        cache = Path(command[command.index("--prepared-cache") + 1])
        identity = prepared_cache_identity(
            bridge_manifest_sha256=_sha256(bridge),
            source_manifest_sha256=_sha256(source_manifest),
            static_cache_sha256=_sha256(static_cache),
            namelist_sha256=_sha256(namelist),
            domain_config=exp.root, forcing_hours=forcing_hours,
            source_identity={
                "source_sha256": {
                    name: hashlib.sha256(name.encode()).hexdigest()
                    for name in runner._HRRR_DECODE_SOURCES},
                "source_cycle": CYCLE.isoformat(),
                "model_start_time": exp.start_time.isoformat(),
                "source_forecast_hours": list(SOURCE_HOURS),
                "model_forcing_hours": list(forcing_hours),
            })
        _write_prepared_cache(cache, identity, forcing_hours, exp)
        outdir = Path(command[command.index("--outdir") + 1])
        outdir.mkdir(parents=True)
        (outdir / "report.json").write_text(json.dumps({
            "status": "PASS",
            "history_interval_seconds": history_seconds,
            "physics": {
                "schema": "gpuwm-prepared-physics-profile-v1",
                "profile": PROFILE,
                "resolved": resolved_run_settings(exp.root.run),
                "hrrr_initialization": cold_start_receipt,
            },
            "preparation": {
                "preprocess_backend": {
                    "backend": "cuda",
                    "masked_surface_chain": {"workers": "auto"}},
                "preprocess_worker_budget": {
                    "schema": "gpuwm-preprocess-worker-budget-v1",
                    "backend": "cuda", "applicable": False,
                    "requested_total_native_workers": None,
                    "host_step_native_workers": 8,
                    "pipeline_decoder_workers_included": False,
                    "peak_active_native_workers": 8,
                },
            },
            "pipeline": {"workers": {"requested": "8", "selected": 8}},
        }), encoding="utf-8")

    def fake_run_keeping_refusal(command, env, cwd=None):
        # The WRF-arm export is launched through the seam that keeps
        # what the child said; this case's child says nothing and
        # succeeds, so the pair is the exit code and an empty text.
        fake_run(command, env, cwd)
        return 0, ""

    monkeypatch.setattr(prepare, "_run", fake_run)
    monkeypatch.setattr(prepare, "_run_keeping_refusal",
                        fake_run_keeping_refusal)
    argv = [
        "--source-root", str(source),
        "--source-manifest", str(source_manifest),
        "--source-manifest-sha256", "0" * 64,
        "--static-cache", str(static_cache),
        "--static-receipt", str(static_receipt),
        "--namelist-input", str(namelist),
        "--experiment-config", str(authority), "--domain-spec", str(domain_spec),
        "--physics-profile", PROFILE,
        "--valid-time", "2026-08-05_04:00:00",
        "--forecast-start-hour", str(SOURCE_HOURS[0]),
        "--forecast-end-hour", str(SOURCE_HOURS[-1]),
        "--run-seconds", str(int(run_seconds)),
        "--history-interval-seconds", str(history_seconds),
        "--output-root", str(output),
    ]
    if publish:
        argv[argv.index("--namelist-input") + 2:
             argv.index("--namelist-input") + 2] = [
                 "--wps-namelist", str(wps)]
    assert prepare.main(argv) == 0
    receipt = json.loads(
        (output / "public-wrapper-result.json").read_text(encoding="utf-8"))
    return output, receipt, run_seconds, history_seconds, forcing_hours


def test_the_wrapper_publishes_a_bundle_the_front_door_admits(
        tmp_path, monkeypatch):
    """The whole chain, from the wrapper's argv to the driver's binding.

    What this pins is that the digests the wrapper prints are exactly the
    digests ``preflight_prepared_forecast`` wants -- a caller runs the
    case it just prepared without hashing a file by hand.
    """

    output, receipt, run_seconds, history_seconds, forcing_hours = (
        _run_wrapper(tmp_path, monkeypatch, publish=True))
    handoff = receipt["portable_bundle"]
    assert handoff is not None

    _bind_synthetic_geometry(monkeypatch)
    inputs = runner.preflight_prepared_forecast(
        source="hrrr", prepared_root=output,
        proof_sha256=handoff["proof_sha256"],
        source_manifest_sha256=handoff["source_manifest_sha256"],
        prepared_content_sha256=handoff["prepared_content_sha256"],
        experiment_config=Path(handoff["experiment_config"]),
        wps_namelist=Path(handoff["wps_namelist"]),
        physics_profile=PROFILE,
        run_seconds=run_seconds,
        history_interval_seconds=history_seconds)
    assert inputs.layout == runner.HRRR_DIRECT_LAYOUT
    assert inputs.forcing_hours == forcing_hours
    assert inputs.proof["source_forecast_hours"] == list(SOURCE_HOURS)


def test_a_bare_preparation_publishes_the_bundle_and_sim_accepts_it(
        tmp_path, monkeypatch):
    """The default run, with no --wps-namelist, is a runnable tree.

    This is the shakeout's exact argv shape.  Before, it produced a
    finished native bundle carrying none of the portable authorities,
    and ``woof sim`` refused it -- calling it partial, which it was
    not.  A prepared tree with no front door is not a prepared tree, so
    publication is the default now and the flag only chooses WHICH
    namelist gets bound.

    Every artifact is asserted where a reader would look for it, and the
    tree is then handed to ``woof sim``'s own resolver rather than to a
    restatement of what that resolver wants.
    """

    from woof import stage_cli
    from woof.hrrr_prepared_bundle import (
        EXPERIMENT_CONFIG_NAME, PROOF_NAME, SOURCE_MANIFEST_NAME,
        WPS_NAMELIST_NAME, WRF_NAMELIST_NAME)

    output, receipt, *_rest = _run_wrapper(
        tmp_path, monkeypatch, publish=False)
    handoff = receipt["portable_bundle"]
    assert handoff is not None
    assert receipt["portable_bundle_refusal"] is None
    for name in (PROOF_NAME, SOURCE_MANIFEST_NAME, EXPERIMENT_CONFIG_NAME,
                 WPS_NAMELIST_NAME, WRF_NAMELIST_NAME):
        assert (output / name).is_file(), name
    # the artifacts the native route has always written are untouched
    assert (output / "native" / "prepared-cache" / "header.json").is_file()
    assert (output / "public-wrapper-result.json").is_file()

    resolved = stage_cli.resolve_bundle(output)
    assert resolved["source"] == "hrrr"
    assert resolved["layout"] == "single"
    command = stage_cli.sim_command(
        resolved, experiment_config=output / EXPERIMENT_CONFIG_NAME,
        wps_namelist=output / WPS_NAMELIST_NAME, outdir=tmp_path / "out")
    assert (command[command.index("--proof-sha256") + 1]
            == handoff["proof_sha256"])


def test_a_publication_that_cannot_finish_leaves_the_native_tree_and_says_why(
        tmp_path, monkeypatch):
    """The native preparation is not held hostage by the portable half.

    A tree whose portable authorities could not be published is still a
    complete native tree that the native benchmark runs, so the
    preparation succeeds -- and the reason is recorded in the receipt
    where ``woof sim`` reads it back, rather than being lost to a
    stderr line nobody kept.
    """

    import tools.prepare_hrrr_wrf as prepare

    from woof import stage_cli
    from woof.hrrr_prepared_bundle import HrrrBundleError, PROOF_NAME

    def refuse(**_kwargs):
        raise HrrrBundleError("a prepared case needs at least two frames")

    monkeypatch.setattr(
        prepare, "publish_hrrr_prepared_bundle", refuse)
    output, receipt, *_rest = _run_wrapper(
        tmp_path, monkeypatch, publish=False)

    assert receipt["status"] == "PASS"
    assert receipt["portable_bundle"] is None
    assert ("a prepared case needs at least two frames"
            in receipt["portable_bundle_refusal"])
    assert not (output / PROOF_NAME).exists()
    assert (output / "native" / "prepared-cache" / "header.json").is_file()

    with pytest.raises(stage_cli.StageRefusal) as refusal:
        stage_cli.resolve_bundle(output)
    sentence = str(refusal.value)
    assert "a prepared case needs at least two frames" in sentence
    assert "complete, not partial" in sentence


@pytest.mark.parametrize("enabled", [True, False])
def test_native_ingest_declaration_is_bound_to_preparation_proof(enabled):
    identity, proof = _identity_pair()
    identity["ingest"] = {"soil_texture_downscale": enabled}
    proof["ingest"] = dict(identity["ingest"])
    assert runner._validate_hrrr_source_identity(identity, proof) is identity
    proof["ingest"]["soil_texture_downscale"] = not enabled
    with pytest.raises(ValueError, match="ingest settings differ"):
        runner._validate_hrrr_source_identity(identity, proof)


@pytest.mark.parametrize("bad", [None, {}, {"soil_texture_downscale": 1},
                                {"soil_texture_downscale": "false"}])
def test_native_ingest_declaration_cannot_be_missing_or_untyped(bad):
    identity, proof = _identity_pair()
    identity["ingest"] = bad
    proof["ingest"] = {"soil_texture_downscale": False}
    with pytest.raises(ValueError, match="ingest settings differ"):
        runner._validate_hrrr_source_identity(identity, proof)


@pytest.mark.parametrize("settings", [
    {"ingest": {"soil_texture_downscale": True}},
    {"ingest": {"soil_texture_downscale": False}},
    {"static_highres": {"enabled": False, "cache_root": "highres"}},
    {"trace_gas_overrides": {"co2": 0.000731}},
    {"trace_gas_overrides": None},
])
def test_native_publisher_relay_reaches_unchanged_strict_consumer(tmp_path, settings):
    bundle=_hrrr_bundle(tmp_path,scientific_identity=settings)
    proof=json.loads((bundle.root/"proof.json").read_text(encoding="utf-8"))
    header=json.loads((bundle.root/"native/prepared-cache/header.json").read_text(encoding="utf-8"))
    identity=header["identity"]["source_identity"]
    for key,value in settings.items():
        assert key in proof and proof[key]==value
    assert runner._validate_hrrr_source_identity(identity,proof) is identity
    # Removing an active field reproduces the original public sim refusal;
    # changing a declared value remains refused by the existing consumer.
    key=next(iter(settings))
    altered=dict(proof)
    if settings[key] is not None:
        altered.pop(key)
        with pytest.raises(ValueError,match="settings differ"):
            runner._validate_hrrr_source_identity(identity,altered)
    altered=dict(proof)
    altered[key]=({"soil_texture_downscale":not settings[key]["soil_texture_downscale"]}
                  if key=="ingest" else {"changed":True})
    with pytest.raises(ValueError,match="settings differ"):
        runner._validate_hrrr_source_identity(identity,altered)


def test_native_publisher_preserves_absent_optional_identity(tmp_path):
    bundle=_hrrr_bundle(tmp_path)
    proof=json.loads((bundle.root/"proof.json").read_text(encoding="utf-8"))
    assert not {"ingest","static_highres","trace_gas_overrides"}&proof.keys()


@pytest.mark.parametrize("receipt_key", [
    "soil_moisture_floor", "deep_soil_repair", "soil_texture_downscale",
    "soil_temperature_repair",
])
def test_native_soil_receipt_roundtrip_is_exact(tmp_path, monkeypatch, receipt_key):
    from copy import deepcopy
    from woof.ingest.prepared_cache import PreparedCacheReader
    receipt = {"schema": "actual-operation-receipt", "applied": False,
               "reason": "target not finer than source", "count": 3}
    bundle = _hrrr_bundle(tmp_path, user_receipts={receipt_key: receipt})
    _bind_synthetic_geometry(monkeypatch)
    _preflight(bundle)
    proof = json.loads((bundle.root / "proof.json").read_text(encoding="utf-8"))
    assert proof[receipt_key] == receipt
    cache = bundle.root / "native/prepared-cache"
    header = json.loads((cache / "header.json").read_text(encoding="utf-8"))
    reader = PreparedCacheReader(cache, expected_identity=header["identity"])
    def validate(candidate):
        runner._validate_cache_metadata(
            reader, source="hrrr", exp=bundle.experiment,
            forcing_hours=(0, 1, 2), boundary_interval_seconds=3600,
            proof=candidate, layout=runner.HRRR_DIRECT_LAYOUT)
    validate(proof)
    missing = deepcopy(proof)
    missing.pop(receipt_key)
    with pytest.raises(ValueError, match="user metadata differs"):
        validate(missing)
    changed = deepcopy(proof)
    changed[receipt_key]["count"] += 1
    with pytest.raises(ValueError, match="user metadata differs"):
        validate(changed)


def test_native_publisher_does_not_invent_or_approve_user_metadata(tmp_path, monkeypatch):
    bundle = _hrrr_bundle(tmp_path, user_receipts={"unregistered_operation": {"applied": True}})
    _bind_synthetic_geometry(monkeypatch)
    proof = json.loads((bundle.root / "proof.json").read_text(encoding="utf-8"))
    assert not {"soil_moisture_floor", "deep_soil_repair", "soil_texture_downscale",
                "soil_temperature_repair", "unregistered_operation"} & proof.keys()
    with pytest.raises(ValueError, match="user metadata differs"):
        _preflight(bundle)



@pytest.mark.parametrize("roles", [("experiment_config",), ("wps_namelist",),
                                   ("experiment_config", "wps_namelist")])
def test_native_exact_authorities_can_be_located_under_new_names(tmp_path, monkeypatch, roles):
    bundle = _hrrr_bundle(tmp_path)
    _bind_synthetic_geometry(monkeypatch)
    canonical = _preflight(bundle)
    original_proof = (bundle.root / "proof.json").read_bytes()
    original_manifest = (bundle.root / "source-input-manifest.json").read_bytes()
    caller = tmp_path / "caller copies"
    caller.mkdir()
    overrides = {}
    for role in roles:
        original = Path(bundle.handoff[role])
        copied = caller / ("chosen.toml" if role == "experiment_config" else "chosen.wps")
        copied.write_bytes(original.read_bytes())
        overrides[role] = copied
    admitted = _preflight(bundle, **overrides)
    assert admitted.cache_identity == canonical.cache_identity
    assert dict(admitted.file_sha256) == dict(canonical.file_sha256)
    assert (bundle.root / "proof.json").read_bytes() == original_proof
    assert (bundle.root / "source-input-manifest.json").read_bytes() == original_manifest
    for role, copied in overrides.items():
        assert admitted.authority_paths[role] == copied.resolve()
        assert admitted.source_manifest["files"][role]["name"] == canonical.source_manifest["files"][role]["name"]
    runner._verify_inputs_unchanged(admitted)
    copied = overrides[roles[-1]]
    copied.write_bytes(copied.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="inputs changed during execution"):
        runner._verify_inputs_unchanged(admitted)


@pytest.mark.parametrize("role", ["experiment_config", "wps_namelist"])
def test_native_renamed_authority_does_not_accept_semantically_equal_changed_bytes(
        tmp_path, monkeypatch, role):
    bundle = _hrrr_bundle(tmp_path)
    _bind_synthetic_geometry(monkeypatch)
    copied = tmp_path / ("changed.toml" if role == "experiment_config" else "changed.wps")
    copied.write_bytes(Path(bundle.handoff[role]).read_bytes() + b"\n")
    with pytest.raises(ValueError, match=f"supplied {role} differs from the portable source manifest"):
        _preflight(bundle, **{role: copied})
