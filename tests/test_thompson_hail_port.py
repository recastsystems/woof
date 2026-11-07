"""Admission, table identity and WRF graupel-volume contract."""
from pathlib import Path
import hashlib
from woof.config import RunConfig, validate_run_config

def test_bare_default_hail_config():
    cfg = RunConfig(nx=4, ny=4, nz=24, dx=1000., dy=1000., ztop=12000.,
                    dt=10., run_seconds=10., moist=True, mp_physics=38)
    assert validate_run_config(cfg) is cfg

def test_hail_has_a_table_root():
    from woof.physics_compat import thompson_hail_table_root
    assert thompson_hail_table_root()


def test_hail_native_radius_flags_are_off():
    from woof.core.rrtmg_legacy import legacy_scheme_has_req
    assert legacy_scheme_has_req(38, 1) == (0, 0, 0)
    assert legacy_scheme_has_req(28, 1) == (1, 1, 1)

def test_table_pins_are_complete():
    from woof.core.thompson_contract import HAIL_TABLE_ASSETS, HAIL_HARNESS_SHA256
    for asset in HAIL_TABLE_ASSETS:
        assert len(asset.sha256) == 64, asset
    root = Path(__file__).parents[1] / "woof/data/thompson/hail-build"
    for name, expected in HAIL_HARNESS_SHA256.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected

def test_checkpoint_names_prognostic_graupel():
    from woof.checkpoint_identity import MICROPHYSICS_ALGORITHM_IDENTITIES
    assert "prognostic" in MICROPHYSICS_ALGORITHM_IDENTITIES[38]


def test_hail_transports_number_and_volume():
    from types import SimpleNamespace
    from woof.core.moist import extra_moist_species
    state = SimpleNamespace(qi=1, qir=None, nh=None, qh=None, nwfa=1,
                            ng=1, qvolg=1)
    fields = extra_moist_species(state)
    assert "ng" in fields and "qvolg" in fields
    from woof.core.nest_fields import nest_field_kinds
    cfg = RunConfig(nx=4, ny=4, nz=24, dx=1000., dy=1000., ztop=12000.,
                    dt=10., run_seconds=10., moist=True, mp_physics=38)
    assert {"ng", "qvolg"} <= set(nest_field_kinds(cfg))


def test_single_bit_change_is_visible_to_metric():
    import importlib.util
    import numpy as np
    path = Path(__file__).parents[1] / "tools/thompson_wrf461_oracle/validate_hail.py"
    spec = importlib.util.spec_from_file_location("hail_metric", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = np.array([1.], np.float32)
    changed = (reference.view(np.uint32) ^ np.uint32(1)).view(np.float32)
    result = module.metric(changed, reference)
    assert result["unequal"] == 1 and result["max_ulp"] == 1


def test_hail_preserves_wrfs_cold_start_package():
    from woof.boundary_fields import COLD_START_SEEDED_NUMBERS, scalar_array_boundary_fields
    from woof.ingest.microphysics_cold_start import cold_start_seeded_numbers
    from types import SimpleNamespace
    assert 38 not in COLD_START_SEEDED_NUMBERS
    # WRF v4.7.1 module_initialize_real.F:4801 admits 8, 28 and 29,
    # not THOMPSON_GH (38), to the real.exe number-seeding block.
    assert cold_start_seeded_numbers(SimpleNamespace(mp_physics=38)) == {}
    assert {"ng", "qvolg"} <= set(scalar_array_boundary_fields(SimpleNamespace(mp_physics=38)))
    assert scalar_array_boundary_fields(SimpleNamespace(mp_physics=28)) == ("nwfa", "nifa", "nc", "nr", "ni", "qke_adv")


def test_hail_rejects_the_fork_without_prognostic_volume():
    from dataclasses import replace
    import pytest
    cfg=RunConfig(nx=4, ny=4, nz=24, dx=1000., dy=1000., ztop=12000.,
                  dt=10., run_seconds=10., moist=True, mp_physics=38)
    with pytest.raises(ValueError, match="no prognostic ng/qvolg"):
        validate_run_config(replace(cfg, thompson_version="wrf_39_noaa"))


def test_hail_physics_inventory_carries_ice_radar_and_precipitation():
    from woof.core.physics_inventory import (
        PBL_RQI_MICROPHYSICS, REFL_10CM_MICROPHYSICS, microphysics_scratch_slots)
    assert 38 in PBL_RQI_MICROPHYSICS and 38 in REFL_10CM_MICROPHYSICS
    assert microphysics_scratch_slots(38) == microphysics_scratch_slots(28)


def test_hail_declaration_survives_registry_regeneration(monkeypatch):
    import copy
    import json
    from tools import build_registry
    from woof.physics_registry import REGISTRY_REBUILD_ENV
    monkeypatch.setenv(REGISTRY_REBUILD_ENV, "1")
    seed = json.loads((Path(__file__).parents[1] / "woof/physics_registry_v2.json").read_text())
    seed["components"]["microphysics"]["options"].pop("thompson-graupel-hail-mp38", None)
    rebuilt = build_registry.build(copy.deepcopy(seed))
    row = rebuilt["components"]["microphysics"]["options"]["thompson-graupel-hail-mp38"]
    assert row["implemented"] is True
    assert row["selectors"] == {"mp_physics": 38}
    assert row["constraints"]["admitted_setting_values"]["thompson_version"] == ["wrf_461"]
    assert row["consumers"]["cloud_optics"]["legacy_declares_radii"] is False
    table_sets = {asset["id"]: asset for asset in row["asset_requirements"]}
    assert len(table_sets["wrf-v4.6.1-hail-aware-thompson-mp38-gfortran13-v1"]["assets"]) == 4
    assert "qvolg" in row["consumers"]["moments"]["unpaired"]
    assert {"qg", "ng"} <= {v for pair in row["consumers"]["moments"]["pairs"] for k,v in pair.items() if k in ("mass", "number")}
    assert row["consumers"]["reflectivity_input_species"]
    assert build_registry.render(build_registry.build(copy.deepcopy(rebuilt))) == build_registry.render(rebuilt)


def test_hail_preflight_covers_every_scratch_slot_without_dead_heat_buffers():
    from woof.core.preflight import scratch_slot_registry, scratch_slot_lifetime
    cfg = RunConfig(nx=4, ny=4, nz=24, dx=1000., dy=1000., ztop=12000.,
                    dt=10., run_seconds=10., moist=True, mp_physics=38)
    shapes = scratch_slot_registry(cfg)
    assert "mp_thompson_aero_heat_ocp" not in shapes
    assert "mp_thompson_aero_heat_lvap" not in shapes
    port_slots = [name for name in shapes if name.startswith("mp_thompson")]
    assert port_slots
    for name in port_slots:
        assert scratch_slot_lifetime(name) is not None, name


def test_older_aerosol_generation_does_not_price_unused_frozen_buffers():
    from dataclasses import replace
    from woof.core.preflight import scratch_slot_registry
    cfg = RunConfig(nx=4, ny=4, nz=24, dx=1000., dy=1000., ztop=12000.,
                    dt=10., run_seconds=10., moist=True, mp_physics=28)
    unused = {"mp_thompson_aero_" + n for n in ("qvten", "tten", "qsten", "qgten", "ngten")}
    assert not unused.intersection(scratch_slot_registry(replace(cfg, thompson_version="wrf_39_noaa")))
    assert unused <= scratch_slot_registry(cfg).keys()


from types import SimpleNamespace
def test_hail_offline_consumer_names_missing_transport_mapping():
    from woof.physics_registry import physics_registry
    from woof.offline_child import offline_child_refusal
    row = physics_registry()['components']['microphysics']['options']['thompson-graupel-hail-mp38']['consumers']['offline_child']
    assert row['same_scheme'] is False
    reason = offline_child_refusal(38)
    assert reason == row['refusal']
    assert 'number and volume mapping' in reason
def test_hail_forcing_precondition_requires_requested_monthly_input(tmp_path):
    from woof.config import mp28_aerosol_lateral_forcing_precondition
    cfg = SimpleNamespace(mp_physics=38, specified=True, use_rap_aero_icbc=False,
        mp28_aerosol_source='climatology', wif_climatology_path=str(tmp_path/'missing-monthly.nc'))
    reason = mp28_aerosol_lateral_forcing_precondition(cfg)
    assert isinstance(reason, str)
    assert 'missing-monthly.nc' in reason

import pytest
@pytest.mark.parametrize('selector', [28, 38])
def test_shared_aerosol_fields_remain_bound_for_each_owner(selector):
    from dataclasses import replace
    from woof.core.model import restart_identity_payload
    from woof.verify.cases.nest_ideal_r1_moist import load_scaffold
    scaffold=load_scaffold()
    root=scaffold.domains[0]
    cfg=replace(root.run,mp_physics=selector,mp28_aerosol_source='synthetic')
    exp=replace(scaffold,domains=(replace(root,run=cfg),))
    run=restart_identity_payload(exp)['domains'][0]['run']
    assert run['mp28_aerosol_source']=='synthetic'
    assert run['wif_climatology_path']==''


def test_fortran_table_reference_logarithm_retains_its_word(tmp_path):
    import shutil
    import subprocess
    import sys

    from woof import thompson_density_tables as tables

    if (sys.platform != "linux" or not shutil.which("gfortran")
            or not shutil.which("dpkg-deb")):
        pytest.skip("requires the Linux Fortran table generator tools")
    lib = tables.prepare_reference_runtime(tmp_path)
    source = tmp_path / "probe.f90"
    source.write_text(
        "program probe\n"
        " use iso_fortran_env, only: int32,real32\n"
        " implicit none\n"
        " integer(int32)::arg,answer\n"
        " real(real32)::x,y\n"
        " read(*,'(Z8)')arg\n"
        " x=transfer(arg,x)\n"
        " y=log10(x)\n"
        " answer=transfer(y,answer)\n"
        " write(*,'(Z8.8)')answer\n"
        "end program probe\n")
    binary = tmp_path / "probe"
    subprocess.run(
        ["gfortran", "-O2", "-fno-tree-vectorize", str(source),
         "-L" + str(lib), "-Wl,--disable-new-dtags,-rpath=" + str(lib),
         "-o", str(binary)], check=True, capture_output=True, text=True)
    result = subprocess.run(
        [str(binary)], input=b"3F000106\n", check=True,
        capture_output=True)
    assert result.stdout.strip() == b"BE9A1ED4"


def test_hail_local_frame_pricing_uses_its_own_variant_and_domain_levels():
    from datetime import datetime
    from woof.core import preflight as pf
    from woof.experiment import experiment_from_run_config

    def frames(selector, levels):
        cfg = RunConfig(nx=4, ny=4, nz=levels, dx=1000., dy=1000.,
                        ztop=12000., dt=10., run_seconds=10.,
                        moist=True, mp_physics=selector)
        return pf.kernel_local_frame_bytes(
            experiment_from_run_config(cfg, datetime(2000, 1, 1)))

    for levels in (24, 64):
        assert frames(38, levels)["thompson_aerosol_sed"] == 3072
    for levels in (65, 120):
        assert frames(38, levels)["thompson_aerosol_sed"] == 12288
    assert frames(38, 24)["thompson_aerosol_state"] >= 48
    assert frames(28, 24)["thompson_aerosol_sed"] == 2304
    assert frames(28, 120)["thompson_aerosol_sed"] == 9216


def test_hail_table_memory_is_priced_separately_from_the_aerosol_owner():
    from dataclasses import replace
    from types import SimpleNamespace
    from woof.core.preflight import (
        thompson_generation_coefficient_bytes, experiment_thompson_coefficient_bytes)
    cfg = RunConfig(nx=4, ny=4, nz=24, dx=1000., dy=1000., ztop=12000.,
                    dt=10., run_seconds=10., moist=True, mp_physics=38)
    aerosol = replace(cfg, mp_physics=28)
    hail_bytes = thompson_generation_coefficient_bytes(cfg)
    aerosol_bytes = thompson_generation_coefficient_bytes(aerosol)
    assert hail_bytes > aerosol_bytes > 0
    exp = SimpleNamespace(domains=[SimpleNamespace(run=c) for c in (cfg, aerosol, cfg)])
    assert experiment_thompson_coefficient_bytes(exp) == hail_bytes + aerosol_bytes


def test_stock_namelist_import_preserves_hail_on_both_domains(tmp_path):
    from woof.namelist_import import import_namelists
    from woof.experiment import load_experiment
    from test_namelist_import import INPUT_TEXT, _pair
    native = INPUT_TEXT.replace(" mp_physics = 10, 10,", " mp_physics = 38, 38,")
    assert native != INPUT_TEXT
    emitted, _ = import_namelists(*_pair(tmp_path, inp=native), name="hail-import")
    config = tmp_path / "hail.toml"
    config.write_text(emitted)
    exp = load_experiment(config)
    assert len(exp.domains) == 2
    assert all(domain.run.mp_physics == 38 for domain in exp.domains)
    assert all(domain.run.thompson_version == "wrf_461" for domain in exp.domains)


def test_stock_hail_cache_controls_reach_both_domains(tmp_path):
    from woof.namelist_import import import_namelists
    from woof.experiment import load_experiment
    from test_namelist_import import INPUT_TEXT, _pair
    native = INPUT_TEXT.replace(" mp_physics = 10, 10,",
        " mp_physics = 38, 38,\n force_read_thompson = .true.,\n"
        " write_thompson_tables = .false.,\n write_thompson_mp38table = .true.,")
    emitted, _ = import_namelists(*_pair(tmp_path, inp=native), name="hail-cache")
    config = tmp_path / "hail.toml"
    config.write_text(emitted)
    for domain in load_experiment(config).domains:
        assert domain.run.force_read_thompson is True
        assert domain.run.write_thompson_tables is False
        assert domain.run.write_thompson_mp38table is True


def test_hail_cache_policy_keeps_existing_bytes_and_honors_missing_policy(tmp_path, monkeypatch):
    from woof import thompson_density_tables as tables
    from woof.core.thompson_contract import TableAsset
    from woof.table_assets import TableAssetError
    payloads = {tables.HAIL_MP38_TABLE_FILE: b"native hail", "shared.dat": b"native shared"}
    assets = tuple(TableAsset(name, len(data), hashlib.sha256(data).hexdigest())
                   for name, data in payloads.items())
    monkeypatch.setattr(tables, "HAIL_TABLE_ASSETS", assets)
    calls = []
    def build(path):
        calls.append(Path(path))
        Path(path).mkdir(parents=True, exist_ok=True)
        for name, data in payloads.items():
            (Path(path) / name).write_bytes(data)
        return Path(path)
    monkeypatch.setattr(tables, "ensure_thompson_hail_tables", build)
    tables.prepare_thompson_hail_tables.cache_clear()
    missing = tmp_path / "missing"
    with pytest.raises(TableAssetError, match="force_read_thompson"):
        tables.prepare_thompson_hail_tables(str(missing), force_read=True)
    with pytest.raises(TableAssetError, match="write_thompson_mp38table"):
        tables.prepare_thompson_hail_tables(str(missing))
    assert not missing.exists() and not calls
    transient = tables.prepare_thompson_hail_tables(str(missing), write_tables=False,
                                                   write_mp38table=True)
    assert transient != missing and not missing.exists()
    assert all((transient / name).read_bytes() == data for name, data in payloads.items())
    persistent = tables.prepare_thompson_hail_tables(str(missing), write_mp38table=True)
    assert persistent == missing
    calls.clear()
    assert tables.prepare_thompson_hail_tables(str(missing), force_read=True,
                                              write_tables=False) == missing
    assert not calls
    # The cached startup decision must not re-stat/hash a gigabyte per step.
    monkeypatch.setattr(tables, "classify_assets", lambda *a: pytest.fail("rechecked cache"))
    assert tables.prepare_thompson_hail_tables(str(missing), force_read=True,
                                              write_tables=False) == missing
    tables.prepare_thompson_hail_tables.cache_clear()


def test_bare_install_stages_the_verified_hail_table_before_wrf_policy(tmp_path, monkeypatch):
    """2.9.12 pass 17 R11: on a bare install the mp38 profile refused at its
    first step ("write_thompson_mp38table is false and the native hail table is
    missing").  The door now stages the verified table first (the fetch-tables
    route), then WRF's policy reads a present table; a failed staging refuses by
    name; force_read keeps WRF's read-only refusal."""
    from woof import thompson_density_tables as tables
    from woof.core.thompson_contract import TableAsset
    from woof.table_assets import TableAssetError
    payloads = {tables.HAIL_MP38_TABLE_FILE: b"native hail", "shared.dat": b"native shared"}
    assets = tuple(TableAsset(name, len(data), hashlib.sha256(data).hexdigest())
                   for name, data in payloads.items())
    monkeypatch.setattr(tables, "HAIL_TABLE_ASSETS", assets)
    calls = []

    def stage(path):
        calls.append(Path(path))
        Path(path).mkdir(parents=True, exist_ok=True)
        for name, data in payloads.items():
            (Path(path) / name).write_bytes(data)
        return Path(path)

    monkeypatch.setattr(tables, "ensure_thompson_hail_tables", stage)
    tables.prepare_thompson_hail_tables.cache_clear()
    tables.stage_hail_table_for_run.cache_clear()
    bare = tmp_path / "bare-home-hail"
    # force_read: nothing staged, WRF's refusal by name.
    assert tables.stage_hail_table_for_run(str(bare), force_read=True) == bare and not calls
    with pytest.raises(TableAssetError, match="force_read_thompson"):
        tables.prepare_thompson_hail_tables(str(bare), force_read=True)
    # Default controls (write_thompson_mp38table = .false.): staged, then read.
    root = tables.stage_hail_table_for_run(str(bare))
    assert calls == [bare]
    assert tables.prepare_thompson_hail_tables(str(root)) == bare
    # Present: one stat, no second staging.
    tables.stage_hail_table_for_run.cache_clear()
    assert tables.stage_hail_table_for_run(str(bare)) == bare and calls == [bare]

    # A staging failure is a refusal naming the table and the commands.
    def fail(path):
        raise FileNotFoundError("no gfortran")

    monkeypatch.setattr(tables, "ensure_thompson_hail_tables", fail)
    tables.stage_hail_table_for_run.cache_clear()
    with pytest.raises(TableAssetError) as refused:
        tables.stage_hail_table_for_run(str(tmp_path / "offline"))
    text = str(refused.value)
    assert tables.HAIL_MP38_TABLE_FILE in text and "no gfortran" in text
    assert "woof fetch-tables --schemes thompson38" in text
    tables.prepare_thompson_hail_tables.cache_clear()
    tables.stage_hail_table_for_run.cache_clear()


def test_fetch_tables_names_the_hail_family(tmp_path, monkeypatch):
    from woof import tables
    from woof.core.thompson_contract import HAIL_TABLE_ASSETS
    monkeypatch.setenv("WOOF_THOMPSON_HAIL_TABLE_ROOT", str(tmp_path / "hail"))
    files = tables._files("thompson38", {})
    assert [f.path.name for f in files] == [a.filename for a in HAIL_TABLE_ASSETS]
    assert all(f.path.parent == tmp_path / "hail" for f in files)
    report = tables.status(["thompson38"])
    assert {item.path.name for item in report.missing} == {a.filename for a in HAIL_TABLE_ASSETS}
