"""The K-epsilon PBL (bl_pbl_physics=17): its doors, refusals and pins.

CPU only.  The card checks are tests/test_keps_wrf471_parity.py (every
output word of every oracle call) and tests/test_keps_driver_seam.py (the
driver seam and the bit-identical restart).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from woof.config import (KEPS_PBL_SCHEME, PBL_SCHEMES, RunConfig,
                          validate_run_config)

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "woof" / "core" / "kernels" / "keps.cu"


def _cfg(**overrides) -> RunConfig:
    base = dict(nx=8, ny=6, nz=30, dx=3000.0, dy=3000.0, dt=15.0,
                ztop=15000.0, run_seconds=60.0, moist=True, mp_physics=8,
                sf_sfclay_physics=1, bl_pbl_physics=KEPS_PBL_SCHEME,
                sf_surface_physics=2, num_soil_layers=4, bldt=0.0,
                cu_physics=0)
    base.update(overrides)
    return RunConfig(**base)


def test_the_selector_is_admitted_with_every_surface_layer_that_feeds_it():
    assert KEPS_PBL_SCHEME == 17 and 17 in PBL_SCHEMES
    for sfclay in (1, 5, 91):
        validate_run_config(_cfg(sf_sfclay_physics=sfclay))


@pytest.mark.parametrize("sfclay", (0, 2))
def test_a_surface_layer_without_ust_psim_psih_is_refused_by_name(sfclay):
    with pytest.raises(ValueError, match="PSIH|Eta"):
        validate_run_config(_cfg(sf_sfclay_physics=sfclay))


def test_kpbl_readers_are_refused_because_keps_never_writes_kpbl():
    with pytest.raises(ValueError, match="KPBL"):
        validate_run_config(_cfg(gwd_opt=1))
    with pytest.raises(ValueError, match="KPBL"):
        validate_run_config(_cfg(cu_physics=3, cudt_minutes=0.0))


def test_bep_admits_keps_as_wrf_does():
    from woof.config import URBAN_BEP_PBL_SCHEMES
    assert KEPS_PBL_SCHEME in URBAN_BEP_PBL_SCHEMES


def test_dispatch_restart_identity_and_namelist_rows():
    from woof.checkpoint_identity import PBL_ALGORITHM_IDENTITIES
    from woof.core.physics_inventory import PHYSICS_SLOT_DISPATCH
    from woof.namelist_import import _BL_MAP

    assert PHYSICS_SLOT_DISPATCH["bl_pbl_physics"][17] == "_run_keps"
    assert PBL_ALGORITHM_IDENTITIES[17].startswith("keps-pbl-wrf-v4.7.1")
    assert _BL_MAP[17][0] == 17


def test_native_carrier_geometry_and_nest_coupling_are_complete():
    from woof.ingest.wrfinput import PHYSICS_FIELD_ALIASES, WRFINPUT_DIMENSIONS
    from woof.core.physics_inventory import KEPS_FIELDS
    from woof.core.nest_fields import nest_field_kinds
    from woof.ingest.lateral_bc import COUPLED_SCALAR_STATE_FIELDS
    for field in KEPS_FIELDS:
        native = PHYSICS_FIELD_ALIASES[field][0]
        zdim = "bottom_top_stag" if field == "keps_tke" else "bottom_top"
        assert WRFINPUT_DIMENSIONS[native] == (zdim, "south_north", "west_east")
    carried = {"keps_tke_adv", "keps_diss_adv", "keps_tpe_adv"}
    assert carried <= set(nest_field_kinds(_cfg()))
    assert carried <= COUPLED_SCALAR_STATE_FIELDS


def test_registry_publishes_the_option_with_its_warnings():
    from woof.physics_registry import physics_registry

    option = physics_registry()["components"]["pbl"]["options"]["keps"]
    assert option["selectors"] == {"bl_pbl_physics": 17}
    assert option["implemented"] is True
    text = " ".join(option["warnings"])
    for needle in ("ttke", "radsum", "WV_TUR", "0/0", "bldt", "KPBL"):
        assert needle in text, needle


def test_kernel_slot_count_and_geometry_match_the_inventory():
    from woof.core.physics_inventory import KEPS_SLOTS, keps_workspace_floats

    src = KERNEL.read_text(encoding="utf-8")
    enum = re.search(r"enum \{(.*?)KE_SLOTS", src, re.S).group(1)
    names = [n.strip().split("=")[0].strip() for n in enum.split(",")
             if n.strip()]
    assert len(names) == KEPS_SLOTS
    assert keps_workspace_floats(40, 64) == KEPS_SLOTS * 42 * 64


def _hexconst(src: str, name: str) -> np.float32:
    m = re.search(rf"#define {name}\s+\(?(-?0x[0-9a-fp.+-]+)f\)?", src)
    return np.float32(float.fromhex(m.group(1)))


def test_folded_constants_are_the_words_gfortran_folds():
    """Every folded constant in keps.cu recomputed in float32 from the
    module's PARAMETER values (module_bl_keps.F:3-6, :1036-1037): one
    IEEE rounding per operation, which is how gfortran folds them."""
    f = np.float32
    from woof.core.kernels import module_source
    src = module_source("keps")
    c1, c2, c3 = f(1.44), f(1.92), f(1.44)
    want = {
        "KE_C2M1": c2 - f(1), "KE_SQ_C2M1": np.sqrt(c2 - f(1)),
        "KE_C1M1": c1 - f(1), "KE_C3M1": c3 - f(1),
        "KE_BETA": f(-1) / c2, "KE_INV_AB": f(1) / (f(1) + f(-1) / c2),
        "KE_1O1P4": f(1) / f(1.4), "KE_SQRT_CMU": np.sqrt(f(0.09)),
        "KE_RVRD_M1": f(461) / f(287) - f(1),
        "KE_CONPR": (f(6.8) * f(0.4)) * f(0.1), "KE_4VK": f(4) * f(0.4),
        "KE_TKEEPS_LO": f(0.025) / f(2),
    }
    for name, value in want.items():
        assert _hexconst(src, name).view(np.uint32) == \
            np.float32(value).view(np.uint32), name


def test_the_oracle_fixtures_are_the_pinned_words():
    """The packaged fixtures decompress to the bytes the oracle wrote
    (oracle-sha256sums.txt) and every set replays eight calls."""
    import lzma

    from woof.verify.keps_oracle import FIXTURE_DIR, SETS, load_set

    pins = {}
    for line in (FIXTURE_DIR / "oracle-sha256sums.txt").read_text().splitlines():
        digest, path = line.split()
        if path.startswith("declared/"):
            pins[Path(path).name] = digest
    for name in SETS:
        raw = lzma.decompress((FIXTURE_DIR / f"keps-{name}.bin.xz").read_bytes())
        assert hashlib.sha256(raw).hexdigest() == pins[f"keps-{name}.bin"]
        fx = load_set(name)
        assert fx.ncol == 72 and fx.nstep == 8
    assert "TOTAL differing words: 0" in (
        FIXTURE_DIR / "bep-identity.txt").read_text()


def test_nothing_is_allocated_for_keps_under_another_scheme():
    from woof.core.preflight import (keps_column_workspace_bytes,
                                      keps_output_transient_shapes)

    assert keps_output_transient_shapes(_cfg(bl_pbl_physics=1)) == {}
    assert keps_output_transient_shapes(_cfg())

    class _Exp:
        def __init__(self, cfg):
            self.domains = [type("D", (), {"run": cfg})()]

    assert keps_column_workspace_bytes(_Exp(_cfg(bl_pbl_physics=1))) == 0
    assert keps_column_workspace_bytes(_Exp(_cfg())) > 0


def test_parameters_are_pinned_and_not_inline():
    from woof.table_assets import kernel_parameter_source
    from woof.core.kernels import module_source
    assert "#define KE_CMU" not in KERNEL.read_text()
    assert "#define KE_CMU" in kernel_parameter_source("keps")
    assert module_source("keps").count("#define KE_CMU") == 1


def test_shared_exact_loader_options(monkeypatch):
    from woof import wrf_exact
    from woof.core.kernels import module_options
    monkeypatch.setattr(wrf_exact, "ENABLED", True)
    assert module_options("keps") == ("-std=c++17",)
    monkeypatch.setattr(wrf_exact, "ENABLED", False)
    assert module_options("keps") == ("-std=c++17", "--maxrregcount=112")


def test_scalar_package_allocates_with_warm_rain_and_restarts():
    from woof.core.state import DomainState
    from woof.core.physics_inventory import pbl_advected_scalars
    from woof.state_serialization_contract import STATE_SERIALIZED_ATTRS
    cfg = _cfg(mp_physics=1)
    state = DomainState(cfg, array_module=np)
    for name in pbl_advected_scalars(cfg):
        assert getattr(state, name).shape == (cfg.nz, cfg.ny, cfg.nx)
        assert getattr(state, name + "0").shape == (cfg.nz, cfg.ny, cfg.nx)
        assert name in STATE_SERIALIZED_ATTRS


def test_positive_bldt_keeps_wrfs_model_step_binding():
    source = (ROOT / "woof/core/physics.py").read_text()
    arm = source.split("    def _run_keps", 1)[1].split("    def _run_sase", 1)[0]
    assert "dt=float(cfg.dt)" in arm


def test_scalar_state_pricing_matches_the_allocated_arrays():
    from woof.core.state import DomainState
    from woof.core.device_inventory import state_array_shapes
    from woof.core.physics_inventory import pbl_advected_scalars
    cfg = _cfg(mp_physics=1)
    state = DomainState(cfg, array_module=np)
    priced = state_array_shapes(cfg)
    for name in pbl_advected_scalars(cfg):
        assert priced[name] == getattr(state, name).shape
        assert priced[name + "0"] == getattr(state, name + "0").shape


def test_a_one_bit_parameter_table_substitution_is_refused(tmp_path, monkeypatch):
    from woof import data_assets
    from woof.table_assets import kernel_parameter_source
    original = data_assets.data_path("keps/tables/constants.cuh").read_bytes()
    changed = bytearray(original)
    changed[original.index(b"0.09f")] ^= 1
    (tmp_path / "constants.cuh").write_bytes(changed)
    monkeypatch.setattr(data_assets, "data_path", lambda path: tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        kernel_parameter_source("keps")


def test_runtime_and_manifest_options_share_one_exact_route():
    from woof.core.kernels import module_options
    from woof.core.keps import MODULE_OPTIONS
    assert module_options("keps") == MODULE_OPTIONS


def test_transported_carriers_are_priced_once_at_their_state_owner():
    from woof.core.state import DomainState
    from woof.core.physics_inventory import pbl_advected_scalars
    from woof.core.preflight import physics_array_shapes, state_array_shapes
    cfg = _cfg(mp_physics=1)
    state = DomainState(cfg, array_module=np)
    state_shapes = state_array_shapes(cfg)
    driver_shapes = physics_array_shapes(cfg)
    for name in pbl_advected_scalars(cfg):
        assert getattr(state, name).shape == state_shapes[name]
        assert getattr(state, name + "0").shape == state_shapes[name + "0"]
        assert f"fields/{name}" not in driver_shapes


def test_adaptive_keps_reset_uses_the_clock_step_count(monkeypatch):
    from types import SimpleNamespace
    import woof.core.adaptive_clock as adaptive
    cfg = _cfg(dt=1.)
    physics = SimpleNamespace(stepbl=1)
    node = SimpleNamespace(cfg=SimpleNamespace(run=cfg),
                           state=SimpleNamespace(physics=physics),
                           clock=SimpleNamespace(step_count=2))
    monkeypatch.setattr(adaptive, "_physics_deadline_due", lambda *args: None)
    adaptive.AdaptiveClockDriver._drive_surface_pbl_on_time(None, node)
    assert physics.keps_itimestep_override == 3


def test_adaptive_keps_binding_does_not_reconstruct_elapsed_over_dt(monkeypatch):
    from types import SimpleNamespace
    import woof.core.physics as physics
    import woof.core.keps as keps
    from woof.core.physics_inventory import KEPS_FIELDS
    cfg = _cfg(dt=1.)
    a = np.ones((cfg.nz, cfg.ny, cfg.nx), np.float32)
    surface = np.ones((cfg.ny, cfg.nx), np.float32)
    driver = object.__new__(physics.PhysicsDriver)
    driver.state = SimpleNamespace(alt=a, elapsed_seconds=600., qi=None)
    driver.fields = {n: a.copy() for n in (*KEPS_FIELDS, 'exch_h', 'exch_m',
                                           'wu_tur', 'wv_tur', 'wt_tur')}
    driver.fields.update({n: surface for n in ('pblh','ust','hfx','qfx','br',
                                              'znt','psim','psih')})
    driver.rthratenlw = a
    driver.rthratensw = a
    driver.keps_itimestep_override = 3
    slot = SimpleNamespace(materialize=lambda _: None)
    driver._couple_pbl_slot = lambda *args, **kwargs: slot
    calls = []
    def call(inputs, carried, **kwargs):
        calls.append(kwargs)
        return {n: a for n in ('du','dv','dtheta','dqv','dqc','exch_h','exch_m',
                               'wu_tur','wv_tur','wt_tur')}
    monkeypatch.setattr(physics, 'cp', np)
    monkeypatch.setattr(keps, 'keps_step', call)
    atmosphere = {n: a for n in ('u','v','theta','qv','qc','dz','exner','p_interface')}
    driver._run_keps(atmosphere, cfg)
    assert calls == [dict(dt=1., itimestep=3)]
