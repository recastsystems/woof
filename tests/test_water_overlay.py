"""Water-temperature overlay (task #71): byte-identity first.

CPU-only, synthetic, no downloads.  The absent-configuration path is
tested before the feature: no overlay must mean the exact same objects,
fingerprints, manifests, and catalogs as a tree without the feature.
"""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from woof.ingest.grib import Era5Snapshot
from woof.ingest.water_overlay import (
    WaterOverlayError,
    WaterTemperatureOverlay,
    apply_water_temperature_overlay,
    load_water_temperature_overlay,
    masked_bilinear_sample,
    overlay_snapshots,
    overlay_snapshots_by_time,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _analytic(lat, lon):
    """Linear in lat/lon: bilinear interpolation reproduces it exactly."""
    return 290.0 + 1.25 * (lat - 40.0) + 0.75 * (lon + 84.0)


def make_snapshot(valid_time=datetime(1985, 5, 31, 12)):
    """Coarse source crop with an inland water block (a synthetic lake)."""
    latitude = np.arange(40.0, 44.01, 0.5)
    longitude = np.arange(-84.0, -77.99, 0.5)
    ny, nx = latitude.size, longitude.size
    landsea = np.ones((ny, nx))
    landsea[2:7, 3:10] = 0.0
    rows = np.arange(ny)[:, None] * np.ones((1, nx))
    cols = np.ones((ny, 1)) * np.arange(nx)[None, :]
    skintemp = 285.0 + 0.7 * rows + 0.3 * cols
    sst = np.where(landsea < 0.5, 284.0 + 0.5 * rows + 0.2 * cols, np.nan)
    psfc = 101300.0 + 10.0 * rows
    tt = np.stack([280.0 + rows + cols, 250.0 + rows - cols])
    return Era5Snapshot(
        valid_time=valid_time,
        levels_hpa=np.array([1000.0, 500.0]),
        latitude=latitude, longitude=longitude,
        fields={"LANDSEA": landsea, "SKINTEMP": skintemp, "SST": sst,
                "PSFC": psfc, "TT": tt})


def write_overlay(
        path, latitude, longitude, values, *, units="kelvin",
        variable="analysed_sst", standard_name="sea_water_temperature",
        mask=None, time_size=1):
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", time_size)
        dataset.createDimension("lat", latitude.size)
        dataset.createDimension("lon", longitude.size)
        dataset.createVariable("lat", "f8", ("lat",))[:] = latitude
        dataset.createVariable("lon", "f8", ("lon",))[:] = longitude
        var = dataset.createVariable(
            variable, "f8", ("time", "lat", "lon"), fill_value=-1.0e30)
        if units is not None:
            var.units = units
        if standard_name is not None:
            var.standard_name = standard_name
        data = np.broadcast_to(values, (time_size, *values.shape)).copy()
        if mask is not None:
            data = np.ma.array(
                data, mask=np.broadcast_to(mask, data.shape))
        var[:] = data
    return Path(path)


def write_fine_overlay(path, **kwargs):
    latitude = np.arange(39.9, 44.11, 0.05)
    longitude = np.arange(-84.1, -77.89, 0.05)
    values = _analytic(latitude[:, None], longitude[None, :])
    return write_overlay(path, latitude, longitude, values, **kwargs)


# ---------------------------------------------------------------------------
# 1. Absent configuration is the identity
# ---------------------------------------------------------------------------

def test_absent_overlay_is_the_identity():
    snapshots = (make_snapshot(),)
    result, receipt = overlay_snapshots(snapshots, None)
    assert result is snapshots
    assert receipt is None
    by_time = {snapshots[0].valid_time: snapshots[0]}
    mapped, receipt = overlay_snapshots_by_time(by_time, None)
    assert mapped is by_time
    assert receipt is None


#: The stability anchor this file's overlay work was measured against,
#: computed at lane base 4211b9e8 before the overlay existed, from the
#: scaffold and catalog stub below.
_PRE_SASE_FLIP_FINGERPRINT = (
    "5ca17be315a646bf4d45ab39450efad46b8d04497c2eaad54166db51b7d1d2d8")

#: The same anchor after ``sase_additive_dissipation`` flipped default-on
#: in 1a0e8a7f8 ("feat(sase): the additive dissipation channel is the
#: default, on real-data evidence", merged d996b9dbe).  That flip is the
#: ONLY reason this hash moved: the test below re-derives the pre-flip
#: anchor by setting the flag back, so a future drift that is NOT the
#: SASE flip fails here instead of being rebound on faith.  A default
#: that changes the integration belongs in the run provenance, so the
#: hash moving was the fingerprint working.
_PRE_TIEDTKE_KEY_FINGERPRINT = (
    "577998f0fc3e1c17f6a082346d7b803f404f3bbe7ed66d50b0449d9e023c6c7a")

#: The anchor after d0c23dad0 (2.6.4, the New Tiedtke port) appended
#: ``ntiedtke_tiedtke_closure`` to RunConfig UNSCOPED: the restart identity
#: scopes fields by mp_physics (woof.core.model.SCHEME_SCOPED_RUN_FIELDS)
#: and has no cumulus row, so a cu_physics = 16 knob binds every
#: experiment's fingerprint at its False default, this scaffold's included
#: (cu_physics = 0 on both domains).  That key is the ONLY reason this
#: hash moved, and it is measured rather than asserted: the test below
#: drops the key from the identity payload and recovers the pre-2.6.4
#: anchor byte for byte, then flips SASE back and recovers the original.
#: Re-pinned on development branch; the port re-pinned nothing.
_PRE_CQ_DEFAULT_FINGERPRINT = (
    "b288ea01ca75ea49b0c3d0d8c4779d2997fb550b5af8b32bcb6f7236315a3b76")

#: e13fa45c0 / 59f7e280f made moist_cq default-on.  The bound run identity
#: now carries True; setting only that value back recovers the old anchor.
#: The test below proves this move independently of all older ledger rows.
_PRE_CAPABILITY_KEYS_FINGERPRINT = (
    "8724f01da9aa498927c560d9d2a83084b61f3ec5ad6fbd4c014c63f4ca845287")

#: The anchor while the 2.9.4 capabilities campaign's RunConfig keys
#: (CAPABILITY_CAMPAIGN_RUN_KEYS) were bound unscoped at their defaults, on
#: the development branch tip before the lanes' document rules merged.  Kept as
#: the record of that move; nothing produces it any more.
_CAMPAIGN_KEYS_BOUND_FINGERPRINT = (
    "415c6ad0e3e28cdc5247f031c4b65818e44415c873dfd2e6cae8084529cc0d10")

#: The anchor on the merged campaign tree.  The public experiment document
#: (woof.experiment, which restart_identity_payload reads) now drops every
#: campaign key that reaches nothing: hailcast_opt and haildt at their
#: defaults, ocean_config and the three lake_state_* keys at their defaults
#: (woof.checkpoint_identity.drop_capability_campaign_keys), the four
#: seaice_* keys where no land surface runs sea ice, ra_sw_eclipse at its
#: default-on 1, and development branch's PWP controls (SLAB_OCEAN_RUN_KEYS
#: below) at their defaults while the ocean is off.  So a default run keeps
#: the 2.9.3 bytes and the anchor is the pre-campaign anchor again; the
#: test below proves the keys are absent and that an active value binds.
_ANCHOR_FINGERPRINT = _PRE_CAPABILITY_KEYS_FINGERPRINT

#: development branch's PWP controls, absent while sf_ocean_physics = 0.
SLAB_OCEAN_RUN_KEYS = ("ocean_levels", "ocean_analysis",
                       "ocean_analysis_source", "ocean_mld_climatology",
                       "omdt", "ocean_z", "ocean_t", "ocean_s")

#: RunConfig keys the capabilities campaign added to the run identity, by
#: the lane commit that added each.  None changes this scaffold's numbers at
#: its default (no hail diagnostic output, no lake/ocean analysis file, no
#: eclipse in the window, no sea ice on the ideal case).
CAPABILITY_CAMPAIGN_RUN_KEYS = {
    "hailcast_opt": "97e9a3ca061 lane/cap-hail",
    "haildt": "97e9a3ca061 lane/cap-hail",
    "lake_state_source": "f9eb12244a9 lane/cap-lake",
    "lake_state_path": "f9eb12244a9 lane/cap-lake",
    "lake_state_accept_licence": "f9eb12244a9 lane/cap-lake",
    "ocean_config": "8cd59d74c15 lane/cap-waves-couple",
    "ra_sw_eclipse": "d4ce97c6bbc lane/cap-eclipse",
    "seaice_prognostic": "7047f0c9bd7 lane/cap-seaice",
    "seaice_initial_thickness": "7047f0c9bd7 lane/cap-seaice",
    "seaice_new_ice_thickness": "7047f0c9bd7 lane/cap-seaice",
    "seaice_ocean_heat_flux": "7047f0c9bd7 lane/cap-seaice",
}


def without_the_capability_campaign_keys(monkeypatch, exp):
    """Drop the campaign's keys from the identity payload, at their defaults.

    The same mechanism as ``_without_the_tiedtke_key``: widen
    RESTART_TOLERATED_RUN_FIELDS for the test, so the writer itself produces
    the payload every fingerprint carried before the keys existed.  Each key
    must be at its RunConfig default here, so a value move is not hidden.
    """
    import woof.core.model as model
    from woof.config import RunConfig

    fields = RunConfig.__dataclass_fields__
    for domain in exp.domains:
        for key in CAPABILITY_CAMPAIGN_RUN_KEYS:
            assert getattr(domain.run, key) == fields[key].default, key
    assert not set(CAPABILITY_CAMPAIGN_RUN_KEYS) & set(
        model.RESTART_TOLERATED_RUN_FIELDS)
    monkeypatch.setattr(model, "RESTART_TOLERATED_RUN_FIELDS", (
        *model.RESTART_TOLERATED_RUN_FIELDS, *CAPABILITY_CAMPAIGN_RUN_KEYS))


def _stability_anchor_catalog():
    return SimpleNamespace(run_provenance={
        "product_id": "stability-anchor",
        "files": ({"role": "forcing", "path": "a.grib", "sha256": "00",
                   "size": 1, "product_id": "era5", "provenance": ""},)})


def _under_the_anchor_clock(exp):
    """Inverse 48847c9eb's declared default without moving any old pin."""
    from dataclasses import replace
    from woof.core.model import experiment_fingerprint, restart_identity_payload

    assert all(domain.run.terrain_clock == "local_face" for domain in exp.domains)
    # ... and 2.9.0's WRF 4.8 solar default (golden/wrf-fixes), likewise.
    assert all(domain.run.solar_generation == "wrf_480" for domain in exp.domains)
    # ... and the two 2.9.7 keys a moist nest echoes on: correct-mode HN1
    # (hydrometeor_handover_numbers, read under this scaffold's Thompson,
    # F22) and tc/nest's fixed-frame nest_accumulation_archive.  Off, each
    # leaves the payload, which is the payload every recorded anchor carried.
    historical = replace(exp, domains=tuple(replace(
        domain, run=replace(domain.run, terrain_clock="measured",
                            solar_generation="wrf_461",
                            hydrometeor_handover_numbers=False,
                            nest_accumulation_archive=False))
        for domain in exp.domains))
    current_payload = restart_identity_payload(exp)
    historical_payload = restart_identity_payload(historical)
    for current_domain, historical_domain in zip(
            current_payload["domains"], historical_payload["domains"]):
        assert current_domain["run"].pop("terrain_clock") == "local_face"
        assert "terrain_clock" not in historical_domain["run"]
        assert current_domain["run"].pop("solar_generation") == "wrf_480"
        assert "solar_generation" not in historical_domain["run"]
        for key in ("hydrometeor_handover_numbers", "nest_accumulation_archive"):
            assert current_domain["run"].pop(key, None) in (None, True), key
            assert key not in historical_domain["run"], key
    assert current_payload == historical_payload
    assert experiment_fingerprint(exp, _stability_anchor_catalog()) != (
        experiment_fingerprint(historical, _stability_anchor_catalog()))
    return historical


def test_no_overlay_keeps_the_experiment_fingerprint(monkeypatch):
    """The option must not move any existing fingerprint when off.

    The overlay lives on [case_data]/prepare kwargs, never
    ExperimentConfig, so the restart identity payload does not see it.
    The hash below therefore moves only when something else that IS bound
    to run identity changes: the SASE flip, the unscoped 2.6.4 cumulus
    key and the moist_cq default flip, each attributed by reconstruction
    below; the capabilities campaign keys named above leave it at their
    defaults.
    """
    from woof.core.model import experiment_fingerprint
    from woof.verify.cases.nest_ideal_r1_moist import load_scaffold

    exp = _under_the_anchor_clock(load_scaffold())
    _without_the_later_eta_key(monkeypatch, exp)
    assert experiment_fingerprint(
        exp, _stability_anchor_catalog()) == _ANCHOR_FINGERPRINT


def _without_the_later_eta_key(monkeypatch, exp):
    """The pre-80a3009c2 payload is what the production rule now produces.

    ``RunConfig.eta_levels`` landed defaulting to None and, unpopped, moved
    every recorded fingerprint without changing one number any of them
    integrates.  ``restart_identity_payload`` now drops it at its default
    (the absent-stays-absent convention the blocks beside it use), so the
    three recorded hashes are reached by the writer itself; this helper no
    longer widens ``RESTART_TOLERATED_RUN_FIELDS`` to get there, it asserts
    the rule that replaced that monkeypatch.  ``monkeypatch`` is kept in the
    signature so the three call sites read unchanged.
    """
    import woof.core.model as model
    assert all(domain.run.eta_levels is None for domain in exp.domains)
    assert "eta_levels" not in model.RESTART_TOLERATED_RUN_FIELDS
    payload = model.restart_identity_payload(exp)
    assert all("eta_levels" not in domain["run"] for domain in payload["domains"])


def _before_the_moist_cq_default_flip(exp):
    """Restore the sole value moved by e13fa45c0 / 59f7e280f."""
    from dataclasses import replace
    from woof.core.model import restart_identity_payload

    assert all(domain.run.moist_cq is True for domain in exp.domains)
    historical = replace(exp, domains=tuple(
        replace(domain, run=replace(domain.run, moist_cq=False))
        for domain in exp.domains))
    current_payload = restart_identity_payload(exp)
    historical_payload = restart_identity_payload(historical)
    for current_domain, historical_domain in zip(
            current_payload["domains"], historical_payload["domains"]):
        assert current_domain["run"]["moist_cq"] is True
        assert historical_domain["run"]["moist_cq"] is False
        current_domain["run"]["moist_cq"] = False
    assert current_payload == historical_payload
    return historical


def test_the_capability_campaign_keys_leave_the_anchor_at_their_defaults(
        monkeypatch):
    from dataclasses import replace
    from woof.core.model import (experiment_fingerprint,
                                  restart_identity_payload)
    from woof.verify.cases.nest_ideal_r1_moist import load_scaffold

    exp = _under_the_anchor_clock(load_scaffold())
    _without_the_later_eta_key(monkeypatch, exp)
    assert experiment_fingerprint(
        exp, _stability_anchor_catalog()) == _ANCHOR_FINGERPRINT
    assert _ANCHOR_FINGERPRINT != _CAMPAIGN_KEYS_BOUND_FINGERPRINT
    # The named keys are absent from the identity the writer produces.
    named = set(CAPABILITY_CAMPAIGN_RUN_KEYS) | set(SLAB_OCEAN_RUN_KEYS)
    for domain in restart_identity_payload(exp)["domains"]:
        assert not named & set(domain["run"]), named & set(domain["run"])
    # Tolerating them as well changes nothing: they were the only move.
    without_the_capability_campaign_keys(monkeypatch, exp)
    assert experiment_fingerprint(
        exp, _stability_anchor_catalog()) == _PRE_CAPABILITY_KEYS_FINGERPRINT
    monkeypatch.undo()
    # An active value still binds the identity.
    for key, value in (("ra_sw_eclipse", 0), ("ocean_config", "ocean.toml"),
                       ("hailcast_opt", 1)):
        active = replace(exp, domains=tuple(
            replace(domain, run=replace(domain.run, **{key: value}))
            for domain in exp.domains))
        assert experiment_fingerprint(
            active, _stability_anchor_catalog()) != _ANCHOR_FINGERPRINT, key


def test_the_anchor_moved_for_the_moist_cq_default_flip_and_nothing_else(
        monkeypatch):
    from woof.core.model import experiment_fingerprint
    from woof.verify.cases.nest_ideal_r1_moist import load_scaffold

    exp = _under_the_anchor_clock(load_scaffold())
    _without_the_later_eta_key(monkeypatch, exp)
    without_the_capability_campaign_keys(monkeypatch, exp)
    historical = _before_the_moist_cq_default_flip(exp)
    assert experiment_fingerprint(
        exp, _stability_anchor_catalog()) == _PRE_CAPABILITY_KEYS_FINGERPRINT
    assert _PRE_CAPABILITY_KEYS_FINGERPRINT != _PRE_CQ_DEFAULT_FINGERPRINT
    assert experiment_fingerprint(
        historical, _stability_anchor_catalog()) == _PRE_CQ_DEFAULT_FINGERPRINT


def test_current_fingerprint_still_binds_the_later_eta_ladder():
    from dataclasses import replace
    from woof.core.model import experiment_fingerprint
    from woof.verify.cases.nest_ideal_r1_moist import load_scaffold
    exp = load_scaffold()
    changed = replace(exp, domains=tuple(replace(domain, run=replace(
        domain.run, eta_levels=tuple(np.linspace(1., 0., domain.run.nz + 1))))
        for domain in exp.domains))
    assert experiment_fingerprint(exp, _stability_anchor_catalog()) != (
        experiment_fingerprint(changed, _stability_anchor_catalog()))


def _without_the_tiedtke_key(monkeypatch):
    """Drop ``ntiedtke_tiedtke_closure`` from the identity payload.

    ``restart_identity_payload`` pops RESTART_TOLERATED_RUN_FIELDS from
    every domain's run table at call time, so widening that tuple for the
    duration of a test reproduces the payload every fingerprint carried
    before the key existed -- the same hashing, the same catalog, one key
    fewer.  A reconstruction that edited the hash input by hand would
    prove nothing about the writer.
    """
    import woof.core.model as model

    assert "ntiedtke_tiedtke_closure" not in model.RESTART_TOLERATED_RUN_FIELDS
    monkeypatch.setattr(model, "RESTART_TOLERATED_RUN_FIELDS", (
        *model.RESTART_TOLERATED_RUN_FIELDS, "ntiedtke_tiedtke_closure"))


def test_the_anchor_moved_for_the_unscoped_cumulus_key_and_nothing_else(
        monkeypatch):
    """Name the provenance of the 2.6.4 rebind, mechanically.

    With the one appended key removed from the payload, the anchor is the
    pre-2.6.4 value byte for byte.  Had a second identity-bound default
    moved in that release -- or had the field been placed somewhere the
    payload orders by -- this reconstruction would miss and the rebind
    above would be exposed as unexplained rather than attributed.
    """
    from woof.core.model import experiment_fingerprint
    from woof.verify.cases.nest_ideal_r1_moist import load_scaffold

    exp = _before_the_moist_cq_default_flip(_under_the_anchor_clock(load_scaffold()))
    assert all(domain.run.cu_physics == 0 for domain in exp.domains), (
        "the anchor scaffold selects no cumulus scheme, which is what makes "
        "an unscoped cumulus knob moving it identity churn rather than a "
        "trajectory change")
    assert all(
        domain.run.ntiedtke_tiedtke_closure is False for domain in exp.domains)
    _without_the_later_eta_key(monkeypatch, exp)
    without_the_capability_campaign_keys(monkeypatch, exp)
    _without_the_tiedtke_key(monkeypatch)
    assert experiment_fingerprint(
        exp, _stability_anchor_catalog()) == _PRE_TIEDTKE_KEY_FINGERPRINT


def test_the_anchor_moved_for_the_sase_default_flip_and_nothing_else(
        monkeypatch):
    """Name the provenance of the rebind before that one, mechanically.

    Putting the pre-flip default back reproduces the pre-flip anchor
    byte-for-byte.  That is the whole diff: had a second identity-bound
    default moved as well, this reconstruction would miss and the rebind
    above would be exposed as unexplained rather than attributed.  The
    2.6.4 key is dropped first, so the chain of attributions runs all the
    way back to the pre-overlay anchor rather than stopping one rebind
    short.
    """
    from dataclasses import replace

    from woof.core.model import experiment_fingerprint
    from woof.verify.cases.nest_ideal_r1_moist import load_scaffold

    exp = _before_the_moist_cq_default_flip(_under_the_anchor_clock(load_scaffold()))
    assert all(
        domain.run.sase_additive_dissipation for domain in exp.domains), (
        "the additive dissipation channel is default-on since 1a0e8a7f8")
    pre_flip = replace(exp, domains=tuple(
        replace(domain, run=replace(
            domain.run, sase_additive_dissipation=False))
        for domain in exp.domains))
    _without_the_later_eta_key(monkeypatch, exp)
    without_the_capability_campaign_keys(monkeypatch, exp)
    _without_the_tiedtke_key(monkeypatch)
    assert experiment_fingerprint(
        pre_flip, _stability_anchor_catalog()) == _PRE_SASE_FLIP_FINGERPRINT


def test_catalog_identity_moves_exactly_when_the_overlay_does():
    from woof.ingest.preflight import CatalogFile, InputCatalog

    def catalog(files):
        return InputCatalog(
            files=files, product_id="era5", provenance={},
            raw_valid_times=(), valid_times=(), excluded_valid_times=(),
            levels_hpa=(), inventory=(), units={}, spatial_coverage=None,
            masks={}, snapshots=(), field_sources={})

    base_files = (CatalogFile(
        role="forcing", path=Path("a.grib"), sha256="00", size=1),)
    overlay_file = CatalogFile(
        role="water_temperature_overlay", path=Path("sst.nc"),
        sha256="11", size=2)
    without = catalog(base_files)
    unchanged = catalog(base_files)
    with_overlay = catalog((*base_files, overlay_file))
    assert without.fingerprint == unchanged.fingerprint
    assert without.fingerprint != with_overlay.fingerprint
    roles = [item["role"] for item in with_overlay.run_provenance["files"]]
    assert "water_temperature_overlay" in roles
    assert "water_temperature_overlay" not in [
        item["role"] for item in without.run_provenance["files"]]


def test_case_data_key_is_optional_and_off_by_default(tmp_path):
    from woof.case_data import build_case_data

    for name in ("era5.grib", "Vtable.ERA5", "namelist.wps"):
        (tmp_path / name).write_text("x")
    (tmp_path / "geog").mkdir()
    raw = {
        "forcing": ["era5.grib"], "vtable": "Vtable.ERA5",
        "wps_namelist": "namelist.wps", "geog_root": "geog",
        "sfcp_to_sfcp": True, "output_title": "t",
    }
    data = build_case_data(dict(raw), source="test", base_dir=tmp_path)
    assert data.water_temperature_overlay is None
    assert data.water_temperature_overlay_identity() is None
    assert not [record for record in data.resolved_inputs()
                if record.role == "water_temperature_overlay"]

    with pytest.raises(ValueError, match="does not exist"):
        build_case_data(
            {**raw, "water_temperature_overlay": "absent.nc"},
            source="test", base_dir=tmp_path)

    (tmp_path / "sst.nc").write_text("x")
    declared = build_case_data(
        {**raw, "water_temperature_overlay": "sst.nc"},
        source="test", base_dir=tmp_path)
    assert declared.water_temperature_overlay == tmp_path / "sst.nc"
    overlay_records = [record for record in declared.resolved_inputs()
                       if record.role == "water_temperature_overlay"]
    assert [record.path for record in overlay_records] == [
        tmp_path / "sst.nc"]


def test_supervisor_remap_carries_the_overlay(tmp_path):
    from woof.case_data import build_case_data, remap_case_data_files

    for name in ("era5.grib", "Vtable.ERA5", "namelist.wps", "sst.nc"):
        (tmp_path / name).write_text("x")
    (tmp_path / "geog").mkdir()
    data = build_case_data({
        "forcing": ["era5.grib"], "vtable": "Vtable.ERA5",
        "wps_namelist": "namelist.wps", "geog_root": "geog",
        "sfcp_to_sfcp": True, "output_title": "t",
        "water_temperature_overlay": "sst.nc",
    }, source="test", base_dir=tmp_path)
    snapshot_dir = tmp_path / "cas"
    snapshot_dir.mkdir()
    replacements = {}
    for record in data.resolved_inputs():
        if record.role == "geog_root":
            continue
        target = snapshot_dir / record.path.name
        target.write_text("y")
        replacements[record.path] = target
    remapped = remap_case_data_files(data, replacements)
    assert remapped.water_temperature_overlay == snapshot_dir / "sst.nc"
    assert remapped.water_temperature_overlay_identity() == (
        tmp_path / "sst.nc")


# ---------------------------------------------------------------------------
# 2/3. Replacement semantics
# ---------------------------------------------------------------------------

def test_covered_water_replaced_everything_else_bit_identical(tmp_path):
    snapshot = make_snapshot()
    overlay = load_water_temperature_overlay(
        write_fine_overlay(tmp_path / "sst.nc"))
    (result,), receipt = overlay_snapshots((snapshot,), overlay)

    water = np.asarray(snapshot.fields["LANDSEA"]) < 0.5
    lat2d = snapshot.latitude[:, None] * np.ones(snapshot.longitude.size)
    lon2d = np.ones((snapshot.latitude.size, 1)) * snapshot.longitude
    expected = _analytic(lat2d, lon2d)
    for name in ("SKINTEMP", "SST"):
        np.testing.assert_allclose(
            result.fields[name][water], expected[water],
            rtol=0.0, atol=1.0e-9)
        assert np.array_equal(
            result.fields[name][~water], snapshot.fields[name][~water],
            equal_nan=True)
    for name in ("LANDSEA", "PSFC", "TT"):
        assert np.array_equal(
            result.fields[name], snapshot.fields[name], equal_nan=True)
    assert np.array_equal(result.latitude, snapshot.latitude)
    assert result.valid_time == snapshot.valid_time
    assert receipt["replaced_cells"] == int(water.sum())
    assert receipt["fallback_cells"] == 0
    assert receipt["fields"] == ["SST", "SKINTEMP"]
    assert receipt["snapshots"] == 1


def test_uncovered_water_keeps_era5(tmp_path):
    snapshot = make_snapshot()
    latitude = np.arange(39.9, 44.11, 0.05)
    longitude = np.arange(-84.1, -81.31, 0.05)   # stops short of the lake's east
    values = _analytic(latitude[:, None], longitude[None, :])
    overlay = load_water_temperature_overlay(
        write_overlay(tmp_path / "west.nc", latitude, longitude, values))
    result, receipt = apply_water_temperature_overlay(snapshot, overlay)
    water = np.asarray(snapshot.fields["LANDSEA"]) < 0.5
    covered = water & (snapshot.longitude[None, :] <= longitude[-1])
    fallback = water & ~covered
    assert fallback.any() and covered.any()
    assert np.array_equal(
        result.fields["SKINTEMP"][fallback],
        snapshot.fields["SKINTEMP"][fallback])
    assert not np.array_equal(
        result.fields["SKINTEMP"][covered],
        snapshot.fields["SKINTEMP"][covered])
    assert receipt["replaced_cells"] == int(covered.sum())
    assert receipt["fallback_cells"] == int(fallback.sum())


def test_zero_360_longitude_overlay_is_unwrapped(tmp_path):
    snapshot = make_snapshot()
    latitude = np.arange(39.9, 44.11, 0.05)
    longitude = np.arange(275.9, 282.11, 0.05)   # -84.1..-77.89 in 0..360
    values = _analytic(latitude[:, None], longitude[None, :] - 360.0)
    overlay = load_water_temperature_overlay(
        write_overlay(tmp_path / "wrapped.nc", latitude, longitude, values))
    result, receipt = apply_water_temperature_overlay(snapshot, overlay)
    water = np.asarray(snapshot.fields["LANDSEA"]) < 0.5
    assert receipt["replaced_cells"] == int(water.sum())
    lat2d = snapshot.latitude[:, None] * np.ones(snapshot.longitude.size)
    lon2d = np.ones((snapshot.latitude.size, 1)) * snapshot.longitude
    np.testing.assert_allclose(
        result.fields["SKINTEMP"][water], _analytic(lat2d, lon2d)[water],
        rtol=0.0, atol=1.0e-9)


# ---------------------------------------------------------------------------
# 4. Masked bilinear corners
# ---------------------------------------------------------------------------

def _tiny_overlay(valid):
    return WaterTemperatureOverlay(
        path=Path("synthetic"), source_format="netcdf", variable="sst",
        declared_units="K",
        latitude=np.array([0.0, 1.0]), longitude=np.array([0.0, 1.0]),
        temperature_k=np.array([[280.0, 290.0], [300.0, 310.0]]),
        valid=np.asarray(valid))


def test_masked_bilinear_renormalizes_over_valid_corners():
    overlay = _tiny_overlay([[True, True], [True, False]])
    values, covered = masked_bilinear_sample(
        overlay, np.array([0.5]), np.array([0.5]))
    assert covered.all()
    np.testing.assert_allclose(values, [(280.0 + 290.0 + 300.0) / 3.0])


def test_masked_bilinear_all_invalid_is_not_covered():
    overlay = _tiny_overlay([[False, False], [False, False]])
    values, covered = masked_bilinear_sample(
        overlay, np.array([0.5]), np.array([0.5]))
    assert not covered.any()
    assert np.isnan(values).all()


def test_masked_bilinear_outside_bbox_is_not_covered():
    overlay = _tiny_overlay([[True, True], [True, True]])
    values, covered = masked_bilinear_sample(
        overlay, np.array([2.0, 0.5]), np.array([0.5, 0.5]))
    assert list(covered) == [False, True]


# ---------------------------------------------------------------------------
# 5. Refusal catalog
# ---------------------------------------------------------------------------

def test_refuses_unknown_container(tmp_path):
    path = tmp_path / "junk.bin"
    path.write_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07")
    with pytest.raises(WaterOverlayError, match="neither netCDF"):
        load_water_temperature_overlay(path)


def test_refuses_grib_edition_1(tmp_path):
    path = tmp_path / "one.grib"
    path.write_bytes(b"GRIB\x00\x00\x00\x01" + b"\x00" * 16)
    with pytest.raises(WaterOverlayError, match="edition 1"):
        load_water_temperature_overlay(path)


def test_refuses_zero_and_ambiguous_candidates(tmp_path):
    latitude = np.arange(0.0, 1.01, 0.5)
    longitude = np.arange(0.0, 1.01, 0.5)
    values = np.full((3, 3), 290.0)
    nothing = write_overlay(
        tmp_path / "none.nc", latitude, longitude, values,
        variable="mystery", standard_name=None)
    with pytest.raises(WaterOverlayError, match="exactly one"):
        load_water_temperature_overlay(nothing)
    ambiguous = tmp_path / "two.nc"
    write_overlay(ambiguous, latitude, longitude, values, variable="sst",
                  standard_name=None)
    with netCDF4.Dataset(ambiguous, "a") as dataset:
        second = dataset.createVariable("wtmp", "f8", ("time", "lat", "lon"))
        second.units = "kelvin"
        second[:] = 290.0
    with pytest.raises(WaterOverlayError, match="exactly one"):
        load_water_temperature_overlay(ambiguous)
    # An explicit variable resolves the ambiguity.
    picked = load_water_temperature_overlay(ambiguous, variable="wtmp")
    assert picked.variable == "wtmp"


def test_refuses_missing_and_unknown_units(tmp_path):
    latitude = np.arange(0.0, 1.01, 0.5)
    longitude = np.arange(0.0, 1.01, 0.5)
    values = np.full((3, 3), 290.0)
    unitless = write_overlay(
        tmp_path / "unitless.nc", latitude, longitude, values, units=None)
    with pytest.raises(WaterOverlayError, match="no units"):
        load_water_temperature_overlay(unitless)
    strange = write_overlay(
        tmp_path / "strange.nc", latitude, longitude, values,
        units="furlongs")
    with pytest.raises(WaterOverlayError, match="kelvin and Celsius"):
        load_water_temperature_overlay(strange)


def test_refuses_multi_time_and_scrambled_axis(tmp_path):
    latitude = np.arange(0.0, 1.01, 0.5)
    longitude = np.arange(0.0, 1.01, 0.5)
    values = np.full((3, 3), 290.0)
    stacked = write_overlay(
        tmp_path / "stack.nc", latitude, longitude, values, time_size=2)
    with pytest.raises(WaterOverlayError, match="leading dimension"):
        load_water_temperature_overlay(stacked)
    scrambled = write_overlay(
        tmp_path / "scrambled.nc", np.array([0.0, 1.0, 0.5]), longitude,
        values)
    with pytest.raises(WaterOverlayError, match="not strictly monotonic"):
        load_water_temperature_overlay(scrambled)


def test_refuses_undeclared_sentinel(tmp_path):
    latitude = np.arange(0.0, 2.01, 0.5)
    longitude = np.arange(0.0, 2.01, 0.5)
    values = np.full((5, 5), 290.0)
    values[0, 0] = 9999.0
    path = write_overlay(tmp_path / "sentinel.nc", latitude, longitude,
                         values)
    with pytest.raises(WaterOverlayError, match=r"1 valid cell"):
        load_water_temperature_overlay(path)


def test_refuses_disjoint_overlay_and_missing_fields(tmp_path):
    snapshot = make_snapshot()
    latitude = np.arange(-10.0, -8.99, 0.05)
    longitude = np.arange(10.0, 11.01, 0.05)
    faraway = load_water_temperature_overlay(write_overlay(
        tmp_path / "faraway.nc", latitude, longitude,
        np.full((latitude.size, longitude.size), 290.0)))
    with pytest.raises(WaterOverlayError, match="does not intersect"):
        overlay_snapshots((snapshot,), faraway)

    naked = Era5Snapshot(
        valid_time=snapshot.valid_time, levels_hpa=snapshot.levels_hpa,
        latitude=snapshot.latitude, longitude=snapshot.longitude,
        fields={"SKINTEMP": np.asarray(snapshot.fields["SKINTEMP"])})
    with pytest.raises(WaterOverlayError, match="requires LANDSEA"):
        apply_water_temperature_overlay(naked, faraway)
    landsea_only = Era5Snapshot(
        valid_time=snapshot.valid_time, levels_hpa=snapshot.levels_hpa,
        latitude=snapshot.latitude, longitude=snapshot.longitude,
        fields={"LANDSEA": np.asarray(snapshot.fields["LANDSEA"])})
    with pytest.raises(WaterOverlayError, match="neither SST nor SKINTEMP"):
        apply_water_temperature_overlay(landsea_only, faraway)


# ---------------------------------------------------------------------------
# 6. Units guard, both directions
# ---------------------------------------------------------------------------

def test_celsius_shaped_values_under_kelvin_declaration(tmp_path):
    latitude = np.arange(0.0, 1.01, 0.5)
    longitude = np.arange(0.0, 1.01, 0.5)
    path = write_overlay(tmp_path / "c_as_k.nc", latitude, longitude,
                         np.full((3, 3), 15.0), units="K")
    with pytest.raises(WaterOverlayError, match="Celsius-shaped"):
        load_water_temperature_overlay(path)


def test_kelvin_shaped_values_under_celsius_declaration(tmp_path):
    latitude = np.arange(0.0, 1.01, 0.5)
    longitude = np.arange(0.0, 1.01, 0.5)
    path = write_overlay(tmp_path / "k_as_c.nc", latitude, longitude,
                         np.full((3, 3), 288.0), units="degC")
    with pytest.raises(WaterOverlayError, match="kelvin-shaped"):
        load_water_temperature_overlay(path)


def test_celsius_declaration_converts(tmp_path):
    latitude = np.arange(0.0, 1.01, 0.5)
    longitude = np.arange(0.0, 1.01, 0.5)
    path = write_overlay(tmp_path / "celsius.nc", latitude, longitude,
                         np.full((3, 3), 16.5), units="degrees_Celsius")
    overlay = load_water_temperature_overlay(path)
    np.testing.assert_allclose(overlay.temperature_k, 289.65)
    assert overlay.declared_units == "degrees_Celsius"


def test_masked_cells_fall_back_not_replace(tmp_path):
    """Declared-missing overlay cells never paint water cells."""
    snapshot = make_snapshot()
    latitude = np.arange(39.9, 44.11, 0.05)
    longitude = np.arange(-84.1, -77.89, 0.05)
    values = _analytic(latitude[:, None], longitude[None, :])
    mask = np.zeros(values.shape, dtype=bool)
    mask[:, longitude > -81.0] = True
    overlay = load_water_temperature_overlay(write_overlay(
        tmp_path / "masked.nc", latitude, longitude, values, mask=mask))
    result, receipt = apply_water_temperature_overlay(snapshot, overlay)
    water = np.asarray(snapshot.fields["LANDSEA"]) < 0.5
    east = water & (snapshot.longitude[None, :] > -81.0 + 0.05)
    assert east.any()
    assert np.array_equal(
        result.fields["SKINTEMP"][east], snapshot.fields["SKINTEMP"][east])
    assert receipt["fallback_cells"] > 0


# ---------------------------------------------------------------------------
# 7. Route wiring
# ---------------------------------------------------------------------------

def test_runtime_forcing_snapshots_applies_the_declared_overlay(
        tmp_path, monkeypatch, capsys):
    from woof import runtime

    snapshot = make_snapshot()
    decoded = SimpleNamespace(snapshots=(snapshot,))
    monkeypatch.setattr(
        runtime, "cached_era5_forcing", lambda *args, **kwargs: decoded)
    catalog = SimpleNamespace(
        files=(), valid_times=(snapshot.valid_time,),
        excluded_valid_times=())
    overlay_path = write_fine_overlay(tmp_path / "sst.nc")

    plain = SimpleNamespace(
        forcing=(tmp_path / "era5.grib",), vtable=tmp_path / "vt",
        water_temperature_overlay=None)
    untouched = runtime.forcing_snapshots(plain, catalog)
    assert untouched[snapshot.valid_time] is snapshot

    declared = SimpleNamespace(
        forcing=(tmp_path / "era5.grib",), vtable=tmp_path / "vt",
        water_temperature_overlay=overlay_path)
    overlaid = runtime.forcing_snapshots(declared, catalog)
    water = np.asarray(snapshot.fields["LANDSEA"]) < 0.5
    assert not np.array_equal(
        overlaid[snapshot.valid_time].fields["SKINTEMP"][water],
        snapshot.fields["SKINTEMP"][water])
    assert "water-temperature overlay: replaced" in capsys.readouterr().out


def test_era5_direct_binds_the_overlay_into_the_manifest(
        tmp_path, monkeypatch):
    import hashlib
    import json

    from woof import era5_direct

    roles = {}
    for name in ("era5.grib", "Vtable.ERA5", "bridge.exe", "namelist.wps",
                 "exp.toml", "sst.nc"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        path.chmod(0o755)
        roles[name] = path

    def manifest_for(role_paths):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({
            "schema": era5_direct.INPUT_MANIFEST_SCHEMA,
            "files": {
                role: {"name": path.name,
                       "sha256": hashlib.sha256(
                           path.read_bytes()).hexdigest()}
                for role, path in role_paths.items()
            },
        }), encoding="utf-8")
        return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()

    base_roles = {
        "grib": roles["era5.grib"], "vtable": roles["Vtable.ERA5"],
        "bridge": roles["bridge.exe"],
        "wps_namelist": roles["namelist.wps"],
        "experiment_config": roles["exp.toml"],
    }
    kwargs = dict(
        grib=roles["era5.grib"], vtable=roles["Vtable.ERA5"],
        bridge=roles["bridge.exe"], wps_namelist=roles["namelist.wps"],
        static_input=None, static_receipt=None, source_orography=None,
        source_orography_variable="SOILHGT",
        experiment_config=roles["exp.toml"],
        output_root=tmp_path / "out", geog_root=tmp_path)

    # Configured overlay against a manifest without the role: refused.
    manifest, digest = manifest_for(base_roles)
    with pytest.raises(ValueError, match="role inventory differs"):
        era5_direct.prepare_era5_wrf(
            **kwargs, input_manifest=manifest,
            input_manifest_sha256=digest,
            water_temperature_overlay=roles["sst.nc"])

    # The overlay role present and hash-bound: the manifest stage passes
    # (a sentinel replaces the heavy preparation that follows it).
    class Sentinel(Exception):
        pass

    def stop(*args, **kwargs):
        raise Sentinel()

    monkeypatch.setattr(era5_direct, "resolve_preprocess_backend", stop)
    manifest, digest = manifest_for(
        {**base_roles, "water_temperature_overlay": roles["sst.nc"]})
    with pytest.raises(Sentinel):
        era5_direct.prepare_era5_wrf(
            **kwargs, input_manifest=manifest,
            input_manifest_sha256=digest,
            water_temperature_overlay=roles["sst.nc"])

    # Absent kwarg: the historical five-role manifest still verifies.
    manifest, digest = manifest_for(base_roles)
    with pytest.raises(Sentinel):
        era5_direct.prepare_era5_wrf(
            **kwargs, input_manifest=manifest,
            input_manifest_sha256=digest)


# ---------------------------------------------------------------------------
# 8. GRIB2 container (skips where the Rust tools are absent)
# ---------------------------------------------------------------------------

def _grib2_tools():
    import os
    root = Path(__file__).resolve().parents[1]
    release = root / "tools" / "grib1_bridge" / "target" / "release"
    tools = []
    for name in ("grib2_inventory", "grib2_dump"):
        explicit = os.environ.get(f"GPUWM_{name.upper()}")
        for candidate in ([Path(explicit)] if explicit else [
                release / name, release / f"{name}.exe"]):
            if candidate.is_file():
                tools.append(candidate)
                break
    return tools if len(tools) == 2 else None


def test_grib2_water_temperature_round_trip(tmp_path):
    eccodes = pytest.importorskip("eccodes")
    tools = _grib2_tools()
    if tools is None:
        pytest.skip("Rust grib2_inventory/grib2_dump are not built")
    sample = eccodes.codes_grib_new_from_samples("GRIB2")
    latitude = np.arange(40.0, 44.01, 0.5)
    longitude = np.arange(-84.0, -77.99, 0.5)
    values = _analytic(latitude[:, None],
                       longitude[None, :]).ravel()
    eccodes.codes_set(sample, "discipline", 10)
    eccodes.codes_set(sample, "parameterCategory", 3)
    eccodes.codes_set(sample, "parameterNumber", 0)
    eccodes.codes_set(sample, "Ni", longitude.size)
    eccodes.codes_set(sample, "Nj", latitude.size)
    eccodes.codes_set(sample, "latitudeOfFirstGridPointInDegrees",
                      float(latitude[0]))
    eccodes.codes_set(sample, "latitudeOfLastGridPointInDegrees",
                      float(latitude[-1]))
    eccodes.codes_set(sample, "longitudeOfFirstGridPointInDegrees",
                      float(longitude[0] % 360.0))
    eccodes.codes_set(sample, "longitudeOfLastGridPointInDegrees",
                      float(longitude[-1] % 360.0))
    eccodes.codes_set(sample, "iDirectionIncrementInDegrees", 0.5)
    eccodes.codes_set(sample, "jDirectionIncrementInDegrees", 0.5)
    eccodes.codes_set(sample, "jScansPositively", 1)
    eccodes.codes_set_values(sample, values)
    path = tmp_path / "water.grib2"
    with path.open("wb") as stream:
        eccodes.codes_write(sample, stream)
    eccodes.codes_release(sample)

    overlay = load_water_temperature_overlay(
        path, grib2_inventory=tools[0], grib2_dump=tools[1])
    assert overlay.source_format == "grib2"
    assert overlay.declared_units == "K"
    np.testing.assert_allclose(overlay.latitude, latitude)
    np.testing.assert_allclose(
        overlay.longitude, longitude, atol=1.0e-9)
    np.testing.assert_allclose(
        overlay.temperature_k,
        _analytic(latitude[:, None], longitude[None, :]), atol=2.0e-2)


def test_the_mp28_aerosol_pair_is_scheme_scoped_and_refused_off_scheme():
    """The two selectors bind mp=28 identity and cannot exist elsewhere.

    Same property, same proof shape as the WDM6 pair
    (tests/test_wdm6.py): the keys are ABSENT from a non-mp=28 domain's
    identity payload, so both frozen anchors above hash exactly as they
    did before the pair existed, and they are PRESENT AND BOUND for an
    mp=28 domain, so an aerosol-aware checkpoint cannot resume under a
    different aerosol initial state.  The scoping is lossless only
    because validate_run_config refuses an off-scheme non-default, so
    that refusal is asserted here beside the scoping it justifies.
    """
    from dataclasses import replace

    import pytest

    from woof.config import validate_run_config
    from woof.core.model import (SCHEME_SCOPED_RUN_FIELDS,
                                  restart_identity_payload)
    from woof.verify.cases.nest_ideal_r1_moist import load_scaffold

    assert SCHEME_SCOPED_RUN_FIELDS[28] == (
        "wif_climatology_path", "mp28_aerosol_source")

    scaffold = load_scaffold()
    run = scaffold.domains[0].run
    assert run.mp_physics != 28

    def payload(new_run):
        exp = replace(scaffold, domains=(
            replace(scaffold.domains[0], run=new_run),))
        return restart_identity_payload(exp)["domains"][0]["run"]

    off = payload(run)
    for name in SCHEME_SCOPED_RUN_FIELDS[28]:
        assert name not in off, name

    on = payload(replace(run, mp_physics=28,
                         mp28_aerosol_source="synthetic"))
    assert on["mp28_aerosol_source"] == "synthetic"
    assert on["wif_climatology_path"] == ""

    for field, stray in (("mp28_aerosol_source", "synthetic"),
                         ("wif_climatology_path", "somewhere.dat")):
        with pytest.raises(ValueError, match="requires mp_physics=28"):
            validate_run_config(replace(run, **{field: stray}))
