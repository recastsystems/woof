"""Physics parameter sets: off by default, refused when they cannot act.

What these tests prevent, one breakage each:

* a default run whose kernel source, compile-cache key or parameter tables
  move because the mechanism exists (``test_default_*``);
* a registry row that no longer reaches its literal after a kernel edit, so
  a set is stamped on a run whose constant never changed
  (``test_every_registered_anchor_occurs_as_declared``);
* a misspelled or out-of-range constant running the default, or a value the
  scheme was not written for (``test_make_set_refuses_*``);
* a set stamped on a run that has no domain using its scheme
  (``test_a_set_for_an_absent_scheme_is_refused``);
* one process compiling kernels under one set and then running another
  (``test_one_process_runs_one_set``);
* a resume under other constants (``test_the_restart_identity_binds_*``);
* a roughness multiplier that takes a seasonal-crop class under the RUC
  kernel's seasonal decrement, which gave a NaN surface drag on step 4 of a
  January run (``test_no_registered_z0_multiplier_*``,
  ``test_a_z0_multiplier_reaching_a_seasonal_crop_is_refused``).
"""

from __future__ import annotations

import tomllib
from dataclasses import replace

import numpy as np
import pytest

from woof import physics_params as pp
from woof.core import kernels as kernel_loader

_EXPERIMENT = """
[experiment]
name = "physics-params-test"
start_time = 2026-09-19T19:00:00
run_seconds = 3600.0
feedback = 1
smooth_option = 0
blend_width = 5
spec_bdy_width = 5
restart_interval_s = 0.0

[projection]
map_proj = "lambert"
ref_lat = 37.9
ref_lon = -122.6
truelat1 = 30.0
truelat2 = 60.0
stand_lon = -122.6

[shared]
nz = 59
ztop = 20000.0
p_top = 5000.0
eta_levels = [
    1, 0.993814707, 0.985950649, 0.976014256, 0.963557541,
    0.948093116, 0.929123759, 0.90619123, 0.87894237, 0.847207963,
    0.811077714, 0.770949006, 0.727525413, 0.684030771, 0.642961025,
    0.604180932, 0.567562938, 0.532986403, 0.500337601, 0.469508916,
    0.440399021, 0.412912011, 0.386957437, 0.362449884, 0.339308649,
    0.317457527, 0.296824664, 0.277342081, 0.258945674, 0.241574913,
    0.225172549, 0.2096847, 0.195060253, 0.181251153, 0.168211967,
    0.155899644, 0.144273847, 0.133296132, 0.122930467, 0.113142714,
    0.103900604, 0.095173724, 0.0869334266, 0.0791524947, 0.0718053728,
    0.0648678541, 0.0583171472, 0.0521316081, 0.0462909527, 0.0407758839,
    0.0355683193, 0.030651059, 0.026007941, 0.0216237046, 0.0174838807,
    0.0135748768, 0.00988376327, 0.00639845803, 0.00310745789, 0,
]
hybrid_opt = 2
etac = 0.2
base_temp = 290.0
time_step_sound = 4
epssm = 0.1
emdiv = 0.01
hypsometric_opt = 2
h_sca_adv_order = 5
smdiv = 0.1
top_lid = false
moist = true
moist_cq = true
mp_physics = 8
moist_adv_opt = 1
no_mp_heating = 0
mp_tend_lim = 10.0
ra_physics = 0
ra_lw_physics = 4
ra_sw_physics = 4
wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-legacy-v1"
ra_rrtmg_variant = "rrtmg_legacy"
icloud = 1
swrad_scat = 1.0
o3input = 2
use_mp_re = 1
surface_radiation_policy = "required"
sf_sfclay_physics = 5
sf_surface_physics = 3
bl_pbl_physics = 5
num_soil_layers = 6
isfflx = 1
mosaic_lu = 0
mosaic_soil = 0
flag_sm_adj = 0
spp_lsm = 0
bl_mynn_closure = 2.6
bl_mynn_cloudpdf = 2
bl_mynn_mixlength = 1
bl_mynn_edmf = 1
bl_mynn_edmf_mom = 1
bl_mynn_edmf_tke = 0
bl_mynn_mixscalars = 0
bl_mynn_cloudmix = 1
bl_mynn_mixqt = 0
bl_mynn_output = 0
bl_mynn_tkeadvect = false
icloud_bl = 1
w_damping = 1
damp_opt = 3
zdamp = 5000.0
dampcoef = 0.2
khdif = 0.0
kvdif = 0.0
km_opt = 4
diff_6th_opt = 0
diff_6th_slopeopt = 0
spec_zone = 1
relax_zone = 4
bldt = 0.0
nwp_diagnostics = 0
terrain_opt = 1
map_proj = 1

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 64
ny = 64
time_step = 10
time_step_fract_num = 0
time_step_fract_den = 1
dx = 3000.0
specified = true
nested = false
history_interval_s = 3600.0
mp_physics = 8
radt = 10.0
cu_physics = 0
cudt_minutes = 0.0
diff_6th_factor = 0.12
"""

_SET = """
[physics_params]
name = "test-set"
values = { "mynn.prandtl" = 0.8, "mynn.czil" = 0.1, "ruc.z0.short" = 0.5 }
"""


@pytest.fixture(autouse=True)
def _fresh_process_binding(monkeypatch):
    monkeypatch.delenv(pp.ENV_VAR, raising=False)
    pp.reset_for_tests()
    yield
    pp.reset_for_tests()


def _scale(values, factors, crop_flags):
    # Host oracle for metadata-selection tests; production scales on the GPU.
    scaled = tuple(float(np.float32(value * factor))
                   for value, factor in zip(values, factors))
    if any(value <= 0 or (crop and value <= 0.125)
           for value, crop in zip(scaled, crop_flags)):
        raise ValueError("seasonal-crop roughness would be non-positive")
    return scaled


def _load_ruc():
    from woof.core.ruc import load_ruc_parameters
    return load_ruc_parameters()


def _cu(module: str) -> str:
    source = "mynn_pbl" if module == "mynn_dmp_sibling" else module
    return (kernel_loader._KDIR / f"{source}.cu").read_text(encoding="utf-8")


def _build(text: str):
    from woof.experiment import build_experiment
    return build_experiment(tomllib.loads(text), source="test_physics_params")


# ----------------------------------------------------------------- registry
def test_every_registered_anchor_occurs_as_declared():
    assert pp.validate_kernel_sites(_cu) == []


def test_registry_defaults_are_the_literals_in_the_kernels():
    rows = pp.registry()
    assert {n for n in rows if not n.startswith("sas.")} == {
        "mynn.czil", "mynn.prandtl", "mynn.cns", "mynn.alp1",
        "mynn.edmf_entrainment", "ruc.z0.tall", "ruc.z0.short", "ruc.rs",
        "boulac.ceps_b", "boulac.ck_b", "boulac.temin", "boulac.th0",
        "boulac.third", "boulac.ep", "boulac.dthv_min", "boulac.exch_min"}
    for row in rows.values():
        for site in row.sites:
            assert pp._literal_value(site.literal) == pp._float32(row.default)


def test_c_float_literal_parses_back_to_the_float32_value():
    for value in (0.74, 0.8, 0.085, 3.5, 1.0, 1e-5, 0.3333333):
        literal = pp.c_float_literal(value)
        assert literal.endswith("f")
        assert np.float32(float(literal[:-1])) == np.float32(value)


def test_canonical_range_endpoints_roundtrip_without_broadening_the_range():
    for name, row in pp.registry().items():
        for value in (row.lower, row.upper):
            pset = pp.make_set("endpoint", {name: value})
            assert pp.make_set(pset.name, dict(pset.values)) == pset
        for value in (row.lower - 0.001, row.upper + 0.001):
            with pytest.raises(pp.PhysicsParamsError, match="range"):
                pp.make_set("outside", {name: value})


def test_referenced_set_files_validate_inner_keys_types_and_cycles(tmp_path):
    path = tmp_path / "set.toml"
    path.write_text('[physics_params]\nname="inner"\nvalues={"mynn.czil"=0.1}\nignored=1\n', encoding="utf-8")
    with pytest.raises(pp.PhysicsParamsError, match="unknown key"):
        pp.parse_table({"set": str(path)}, source="test")
    path.write_text('physics_params=0\n', encoding="utf-8")
    with pytest.raises(pp.PhysicsParamsError, match="must be a table"):
        pp.parse_table({"set": str(path)}, source="test")
    path.write_text('[physics_params]\nset="set.toml"\n', encoding="utf-8")
    with pytest.raises(pp.PhysicsParamsError, match="cyclic"):
        pp.parse_table({"set": str(path)}, source="test")


# ----------------------------------------------------------------- default path
def test_default_kernel_source_is_the_file_itself():
    for path in sorted(kernel_loader._KDIR.glob("*.cu")):
        name = path.stem
        text = path.read_text(encoding="utf-8")
        assert pp.edit_kernel_source(name, text) is text
    expected = (kernel_loader._preamble()
                + kernel_loader._extra_header_text("mynn_pbl")
                + _cu("mynn_pbl"))
    assert kernel_loader.module_source("mynn_pbl") == expected


def test_default_process_has_no_tables_attributes_or_receipt():
    assert pp.ruc_bundle_for_forecast(_load_ruc, _scale) is None
    assert pp.wrfout_global_attrs() == {}
    assert pp.receipt() is None


def test_default_experiment_carries_no_set_and_no_identity_key():
    from woof.core.model import restart_identity_payload
    exp = _build(_EXPERIMENT)
    assert exp.physics_params is None
    assert "physics_params" not in restart_identity_payload(exp)
    assert pp.active() is None


# ----------------------------------------------------------------- sets
def test_make_set_refuses_an_unknown_constant():
    with pytest.raises(pp.PhysicsParamsError, match="not a registered"):
        pp.make_set("x", {"mynn.pranndtl": 0.8})


@pytest.mark.parametrize("value", [0.0, -0.74, 1.5, float("nan"),
                                   float("inf")])
def test_make_set_refuses_a_value_outside_the_registered_range(value):
    with pytest.raises(pp.PhysicsParamsError, match="outside"):
        pp.make_set("x", {"mynn.prandtl": value})


def test_make_set_refuses_empty_unnamed_and_non_numeric_sets():
    with pytest.raises(pp.PhysicsParamsError, match="names no constant"):
        pp.make_set("x", {})
    with pytest.raises(pp.PhysicsParamsError, match="name"):
        pp.make_set("", {"mynn.prandtl": 0.8})
    with pytest.raises(pp.PhysicsParamsError, match="not a number"):
        pp.make_set("x", {"mynn.prandtl": "0.8"})


def test_the_set_is_canonical_float32_and_its_hash_follows_its_values():
    a = pp.make_set("a", {"mynn.prandtl": 0.8, "mynn.cns": 3.0})
    b = pp.make_set("a", {"mynn.cns": 3.0, "mynn.prandtl": 0.8000000001})
    assert a.values == b.values
    assert a.sha256() == b.sha256()
    c = pp.make_set("a", {"mynn.prandtl": 0.81, "mynn.cns": 3.0})
    assert c.sha256() != a.sha256()


def test_set_file_and_unknown_table_keys(tmp_path):
    set_file = tmp_path / "wc.toml"
    set_file.write_text(_SET, encoding="utf-8")
    pset = pp.parse_table({"set": "wc.toml"}, source="t", base_dir=tmp_path)
    assert pset.name == "test-set"
    assert pset.value("mynn.prandtl") == pp._float32(0.8)
    with pytest.raises(pp.PhysicsParamsError, match="unknown key"):
        pp.parse_table({"name": "x", "valuess": {}}, source="t")
    with pytest.raises(pp.PhysicsParamsError, match="both"):
        pp.parse_table({"set": "wc.toml", "values": {"mynn.cns": 3.0}},
                       source="t", base_dir=tmp_path)


# ----------------------------------------------------------------- kernels
def test_an_active_set_rewrites_exactly_its_literals():
    pp.declare(pp.make_set("k", {"mynn.prandtl": 0.8, "mynn.czil": 0.1,
                                  "mynn.cns": 2.5}), source="t")
    pbl = pp.edit_kernel_source("mynn_pbl", _cu("mynn_pbl"))
    assert pbl.count("const real pr = 0.8f,") == 3
    assert "pr = 0.74f" not in pbl
    assert pbl.count("MYNN_MUL(2.5f, mynn_min2(MYNN_MUL(zwk, rmo), 1.0f))));") == 1
    # alp1 and entrainment were not in the set: untouched.
    assert "MYNN_MUL(0.23f, elt)" in pbl
    assert "MYNN_DIV(0.33f, MYNN_MUL(" in pbl
    surface = pp.edit_kernel_source("mynn_surface", _cu("mynn_surface"))
    assert "SL_EXP(-0.4f * 0.1f * sqrtf(restar))" in surface
    # Everything else in the unit is byte-for-byte the file.
    assert len(surface) == len(_cu("mynn_surface")) - len("0.085f") + len("0.1f")
    other = _cu("ysu_default")
    assert pp.edit_kernel_source("ysu", other) is other


def test_a_moved_anchor_refuses_rather_than_stamping_a_dead_constant():
    pp.declare(pp.make_set("k", {"mynn.prandtl": 0.8}), source="t")
    edited = _cu("mynn_pbl").replace(
        "const real pr = 0.74f, g1 = 0.235f, b1 = 24.0f, b2 = 15.0f;",
        "const real pr = 0.74f, g1 = 0.235f,  b1 = 24.0f, b2 = 15.0f;", 1)
    with pytest.raises(pp.PhysicsParamsError, match="anchor"):
        pp.edit_kernel_source("mynn_pbl", edited)


# ----------------------------------------------------------------- tables
def test_ruc_multipliers_scale_only_their_categories_and_are_recorded():
    from woof.core.ruc import load_ruc_parameters
    base = load_ruc_parameters()
    pset = pp.make_set("t", {"ruc.z0.short": 0.5, "ruc.rs": 2.0})
    edited = pp.apply_ruc_edits(base, pset, _scale)
    modis_before = base.vegetation_for("MODIFIED_IGBP_MODIS_NOAH").rows
    modis_after = edited.vegetation_for("MODIFIED_IGBP_MODIS_NOAH").rows
    short = set(pp.registry()["ruc.z0.short"].categories["MODI-RUC"])
    vegetated = set(pp.registry()["ruc.rs"].categories["MODI-RUC"])
    for before, after in zip(modis_before, modis_after):
        expect_z0 = (np.float32(before.z0 * 0.5) if before.category in short
                     else before.z0)
        assert after.z0 == pytest.approx(float(expect_z0), abs=0)
        expect_rs = (np.float32(before.rs * 2.0)
                     if before.category in vegetated else before.rs)
        assert after.rs == pytest.approx(float(expect_rs), abs=0)
        assert after.albedo == before.albedo and after.lai == before.lai
    # The pinned bytes are still what was verified; the edit is beside them.
    assert edited.receipt["assets"] == base.receipt["assets"]
    assert edited.receipt["physics_params"]["sha256"] == pset.sha256()
    # And the default bundle object is not modified in place.
    assert load_ruc_parameters().vegetation_for(
        "MODIFIED_IGBP_MODIS_NOAH").rows == modis_before


_CROP_BRANCH = """        roughness = z0tbl[vegetation_index];
        if (forest_class == 7) {
            roughness = __fsub_rn(
                roughness, __fmul_rn(0.125f, factor));
        }"""


def test_no_registered_z0_multiplier_can_make_a_roughness_non_positive():
    # 2026-10-01: ruc.z0.short = 0.6 scaled the MODIS cropland Z0 (0.2 m,
    # IFOR 7) under the kernel's seasonal-crop decrement and the run stopped
    # on step 4 with a NaN surface drag coefficient.  Every z0 row, at both
    # ends of its range, must leave every category it scales positive at the
    # worst seasonal factor (1).
    from woof.core.ruc import load_ruc_parameters
    assert _cu("ruc").count(_CROP_BRANCH) == 1
    assert "0.125f" in _CROP_BRANCH
    assert pp.SEASONAL_CROP_Z0_DECREMENT == 0.125
    base = load_ruc_parameters()
    for row in pp.registry().values():
        if row.column != "z0":
            continue
        for factor in (row.lower, row.upper):
            edited = pp.apply_ruc_edits(base, pp.make_set("t", {row.name: factor}), _scale)
            for section, table in edited.vegetation.items():
                for category in row.categories.get(section, ()):
                    veg = table.rows[category - 1]
                    floor = (pp.SEASONAL_CROP_Z0_DECREMENT
                             if int(veg.ifor) == 7 else 0.0)
                    assert veg.z0 > floor, (row.name, factor, section, category)


def test_a_z0_multiplier_reaching_a_seasonal_crop_is_refused(monkeypatch):
    from woof.core.ruc import load_ruc_parameters
    rows = dict(pp.registry())
    short = rows["ruc.z0.short"]
    rows["ruc.z0.short"] = replace(short, categories={"MODI-RUC": (12,)})
    monkeypatch.setattr(pp, "registry", lambda: rows)
    pset = pp.make_set("t", {"ruc.z0.short": 0.6})
    with pytest.raises(pp.PhysicsParamsError, match="seasonal-crop"):
        pp.apply_ruc_edits(load_ruc_parameters(), pset, _scale)
    # A multiplier that keeps the crop above the decrement still applies.
    edited = pp.apply_ruc_edits(load_ruc_parameters(), pp.make_set("t", {"ruc.z0.short": 1.5}), _scale)
    assert edited.vegetation["MODI-RUC"].rows[11].z0 == pytest.approx(0.3, rel=1e-6)


def test_the_forecast_gets_edited_tables_only_under_a_table_set():
    pp.declare(pp.make_set("k", {"mynn.prandtl": 0.8}), source="t")
    assert pp.ruc_bundle_for_forecast(_load_ruc, _scale) is None
    pp.reset_for_tests()
    pp.declare(pp.make_set("k", {"ruc.z0.tall": 1.5}), source="t")
    bundle = pp.ruc_bundle_for_forecast(_load_ruc, _scale)
    assert bundle is not None
    assert bundle.receipt["physics_params"]["set"] == "k"


# ----------------------------------------------------------------- experiment
def test_an_experiment_set_binds_the_process_and_the_restart_identity():
    from woof.core.model import restart_identity_payload
    exp = _build(_EXPERIMENT + _SET)
    assert exp.physics_params.name == "test-set"
    assert pp.active() == exp.physics_params
    identity = restart_identity_payload(exp)
    assert identity["physics_params"]["name"] == "test-set"
    other = replace(exp, physics_params=pp.make_set(
        "test-set", {"mynn.prandtl": 0.81}))
    assert restart_identity_payload(other) != identity
    assert pp.wrfout_global_attrs() == {
        pp.WRFOUT_NAME_ATTR: "test-set",
        pp.WRFOUT_SHA_ATTR: exp.physics_params.sha256()}


def test_a_set_for_an_absent_scheme_is_refused():
    text = _EXPERIMENT.replace("bl_pbl_physics = 5", "bl_pbl_physics = 1") \
        .replace("sf_sfclay_physics = 5", "sf_sfclay_physics = 1")
    text = "\n".join(line for line in text.splitlines()
                     if not line.startswith("bl_mynn_") and
                     not line.startswith("icloud_bl"))
    try:
        _build(text)
    except ValueError:
        pytest.skip("the YSU variant of the fixture is not a valid config")
    pp.reset_for_tests()
    with pytest.raises(ValueError, match="no domain runs that scheme"):
        _build(text + _SET.replace(
            '"ruc.z0.short" = 0.5', '"ruc.z0.short" = 1.0'))


def test_one_process_runs_one_set():
    _build(_EXPERIMENT + _SET)
    _build(_EXPERIMENT + _SET)  # the same set again is fine
    with pytest.raises(ValueError, match="one set"):
        _build(_EXPERIMENT + _SET.replace("0.8", "0.9"))
    with pytest.raises(ValueError, match="one set"):
        _build(_EXPERIMENT)


def test_a_set_after_default_kernels_compiled_is_refused():
    _build(_EXPERIMENT)
    pp.note_compiled("mynn_pbl")
    with pytest.raises(ValueError, match="default constants"):
        _build(_EXPERIMENT + _SET)


def test_the_environment_set_rides_a_shared_preparation(tmp_path, monkeypatch):
    """Members share one prepared root bound to one experiment file's bytes;
    the set arrives through WOOF_PHYSICS_PARAMS and is attached to the
    experiment exactly as a table would be."""
    from woof.core.model import restart_identity_payload
    set_file = tmp_path / "member.toml"
    set_file.write_text(_SET, encoding="utf-8")
    monkeypatch.setenv(pp.ENV_VAR, str(set_file))
    exp = _build(_EXPERIMENT)
    assert exp.physics_params.name == "test-set"
    assert pp.active() == exp.physics_params
    assert restart_identity_payload(exp)["physics_params"]["name"] == "test-set"
    _build(_EXPERIMENT + _SET)  # the same set in the file too: fine
    pp.reset_for_tests()
    with pytest.raises(ValueError, match=r"two\s+sources"):
        _build(_EXPERIMENT + _SET.replace("0.8", "0.9"))
    monkeypatch.setenv(pp.ENV_VAR, str(tmp_path / "missing.toml"))
    pp.reset_for_tests()
    with pytest.raises(ValueError, match="cannot read"):
        _build(_EXPERIMENT)


def test_unknown_table_is_known_to_branch():
    from woof import branch
    assert "physics_params" in branch._KNOWN_TABLES


def test_branch_rebases_only_the_parameter_set_file(tmp_path):
    from woof.branch import _rebase_declared_paths

    path = tmp_path / "set.toml"
    path.write_text('[physics_params]\nname="member"\nvalues={"mynn.czil"=0.1}\n', encoding="utf-8")
    (tmp_path / "member").mkdir()
    raw = {"physics_params": {"name": "member", "set": "set.toml"}}
    edits = _rebase_declared_paths(raw, tmp_path)
    assert raw["physics_params"] == {"name": "member", "set": str(path.resolve())}
    assert edits == [{"setting": "physics_params.set", "from": "set.toml",
                      "to": str(path.resolve())}]
    inline = {"physics_params": {"name": "member", "values": {"mynn.czil": 0.1}}}
    assert _rebase_declared_paths(inline, tmp_path) == []
    assert inline["physics_params"]["name"] == "member"


@pytest.mark.gpu
def test_an_edited_mynn_unit_compiles():
    cp = pytest.importorskip("cupy")
    pp.declare(pp.make_set("k", {
        "mynn.prandtl": 0.8, "mynn.cns": 2.5, "mynn.alp1": 0.3,
        "mynn.edmf_entrainment": 0.4, "mynn.czil": 0.1}), source="t")
    for module in ("mynn_pbl", "mynn_surface"):
        src = kernel_loader.module_source(module)
        mod = cp.RawModule(code=src, options=("-std=c++17",))
        mod.compile()
