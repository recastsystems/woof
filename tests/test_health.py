"""Phase-5 Task-15 full-state corruption gates (CPU + GPU twins)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from woof.wrf_exact import ENABLED as WRF_EXACT
from woof.core.health import (
    GPU_INTEGER_EXCLUSIONS, MAX_HEALTH_FIELDS, HealthCheckError,
    StateHealthValidator,
    collect_state_fields, cuda_source, validate_fields_cpu,
    validate_state_cpu,
)


_SCALAR_CORRUPTION = np.nan if WRF_EXACT else None

# Every uncovered class named by scout-robust-report.md.  Each fixture has
# exactly one failing value so field attribution is unambiguous.
CORRUPTIONS = (
    ("v", 501.0),
    ("php", np.nan),
    ("mup", 0.0),
    ("p", 0.0),
    ("al", np.inf),
    ("alt", 0.0),
    # Strict mode (GPUWM_WRF_EXACT=1) keeps WRF's own below-zero scalar
    # residuals (woof.core.health.rule_for_field), so there the transported
    # scalars are corrupted through finiteness instead of their sign.
    *((name, _SCALAR_CORRUPTION or -1.0e-6) for name in ("qv", "qc", "qr", "qi", "qs", "qg")),
    *((name, _SCALAR_CORRUPTION or -1.0) for name in ("nc", "nr", "ni", "ns", "ng")),
    *((name, _SCALAR_CORRUPTION or -1.0e-12) for name in ("qvolg", "qvolh")),
    *((name, -1.0) for name in ("effc", "effr", "effi", "effs")),
    ("held.pbl.ru", np.nan),
    ("lbc.d01.u.west.value", np.nan),
    ("nest.d02.theta.west.value", np.nan),
    ("surface.tsk", 401.0),
    ("surface.smois", 1.01),
    # The soil-moisture floor was retired with the RUC refusal (2026-10-07
    # combo sweep round 2: real.exe writes small negative deep-layer words
    # that WRF integrates), so sh2o is corrupted through its ceiling.
    ("surface.sh2o", 1.01),
    ("surface.tslb", 149.0),
)


@pytest.mark.parametrize(("field", "corruption"), CORRUPTIONS)
def test_cpu_mirror_attributes_every_uncovered_corruption(field, corruption):
    baseline = 300.0 if field in ("surface.tsk", "surface.tslb") else 1.0
    values = np.full((2, 3), baseline, dtype=np.float32)
    values.flat[4] = corruption
    report = validate_fields_cpu({field: values}, phase="injected")
    assert not report.ok
    assert report.first_bad_field == field
    assert report.first_bad_index == (1, 1)
    assert report.first_bad_flat_index == 4
    assert report.status_bits != 0
    assert report.reason


def test_cpu_mirror_ors_classes_but_keeps_deterministic_first_field():
    report = validate_fields_cpu({
        "v": np.array([0.0, 900.0], dtype=np.float32),
        "p": np.array([100_000.0, np.nan], dtype=np.float32),
        "surface.smois": np.array([1.2], dtype=np.float32),
    })
    assert not report.ok
    assert report.first_bad_field == "v"
    assert set(report.failing_classes) == {
        "wind", "pressure", "soil_moisture"}


def test_cpu_derived_theta_and_coupled_mass_denominators():
    from woof.core.health import HealthField, rule_for_field

    thp = np.array([[[0.0]], [[-250.0]]], dtype=np.float32)
    thb = np.array([300.0, 300.0], dtype=np.float32)
    mup = np.array([[-101.0]], dtype=np.float32)
    mub = np.array([[100.0]], dtype=np.float32)
    fields = (
        HealthField("thp", thp, rule_for_field("thp"), thb, "level", 1),
        HealthField("mup", mup, rule_for_field("mup"), mub, "direct", 0),
    )
    report = validate_fields_cpu(fields)
    assert report.first_bad_field == "thp"
    assert report.first_bad_index == (1, 0, 0)
    assert set(report.failing_classes) == {"theta", "coupled_mass"}


def test_theta_bound_admits_legal_wk82_class_upper_state():
    legal = validate_fields_cpu({
        "thp": np.array([505.0], dtype=np.float32),
    })
    runaway = validate_fields_cpu({
        "thp": np.array([601.0], dtype=np.float32),
    })
    assert legal.ok
    assert not runaway.ok


def test_aerosol_moment_bound_admits_an_emitting_surface_and_refuses_a_runaway():
    """The moment ceiling is the instrument, not the thing that was wrong.

    ``nwfa``'s lowest level sits ABOVE WRF's own terminal clamp by design:
    the surface emission at module_mp_thompson.F:1310-1327 runs after that
    clamp and is deliberately unclamped, so a legal value decades above
    9999.E6 must pass.  The value a specified domain reached at its
    outermost boundary corner while its spec zone was not being forced back
    onto the boundary table -- five decades above the clamp -- must not, and
    a negative number concentration must not either.
    """
    emitting = validate_fields_cpu({
        "nwfa": np.array([1.0e11], dtype=np.float32),
    })
    runaway = validate_fields_cpu({
        "nwfa": np.array([1011411111116800.0], dtype=np.float32),
    })
    negative = validate_fields_cpu({
        "nifa": np.array([-1.0], dtype=np.float32),
    })
    assert emitting.ok
    assert not runaway.ok
    assert runaway.first_bad_field == "nwfa"
    assert set(runaway.failing_classes) == {"moment"}
    if not WRF_EXACT:   # strict mode keeps WRF's below-zero scalar residuals
        assert not negative.ok
        assert negative.first_bad_field == "nifa"


def test_health_error_is_terminal_and_carries_report():
    report = validate_fields_cpu({"p": np.array([0.0], dtype=np.float32)},
                                 phase="post-sync")
    error = HealthCheckError(report)
    assert error.report is report
    assert "post-sync" in str(error)
    assert "p" in str(error)


def test_cpu_state_gate_includes_restart_persistent_scratch():
    state = SimpleNamespace(
        physics=None, lateral_boundaries=None,
        _scratch={"mp_rainnc": np.array([np.nan], dtype=np.float32)})
    report = validate_state_cpu(state)
    assert not report.ok
    assert report.first_bad_field == "surface.microphysics.scratch.mp_rainnc"


def test_nssl2_post_step_descriptor_high_watermark_fits_cpu_inventory():
    """Pin the 527-field high-water mark observed after a real NSSL step."""
    one = np.ones((1,), dtype=np.float32)
    state = SimpleNamespace(
        qvolg=one, physics=None, lateral_boundaries=None, _scratch={})
    extra = {f"post_step_{index:03d}": one for index in range(526)}
    fields = collect_state_fields(state, backend="cpu", extra_tables=extra)
    assert len(fields) == 527
    assert MAX_HEALTH_FIELDS == 1024


def test_health_cuda_source_compiles_offline_with_nvrtc():
    """Host-only NVRTC gate: compilation, no module load or device launch."""
    from cupy.cuda.compiler import compile_using_nvrtc

    ptx, _ = compile_using_nvrtc(
        cuda_source(), options=("-std=c++17",), arch="80",
        filename="health.cu")
    assert ptx


_REAL_DRIVER_INVENTORY = tuple("""
u v w thp php mup
p al alt qv qc qr
qi qs qg nc nr ni
ns ng effc effr effi effs
h_diabatic held.pbl.ru held.pbl.rv held.pbl.rtheta held.pbl.rqv held.pbl.rqc
held.pbl.rqr held.pbl.rqi held.pbl.rqs held.radiation.ru held.radiation.rv held.radiation.rtheta
held.radiation.rqv held.radiation.rqc held.cumulus.ru held.cumulus.rv held.cumulus.rtheta held.cumulus.rqv
held.cumulus.rqc held.cumulus.rqr held.cumulus.rqi held.cumulus.rqs held.rthratenlw held.rthratensw
held.cu_nca held.cu_pratec held.cu_raincv held.cu_rates.rqccuten held.cu_rates.rqicuten held.cu_rates.rqrcuten
held.cu_rates.rqscuten held.cu_rates.rqvcuten held.cu_rates.rthcuten held._pending_rainbl surface.acgrdflx surface.achfx
surface.aclhf surface.aclwdnb surface.aclwdnbc surface.aclwdnt surface.aclwdntc surface.aclwupb
surface.aclwupbc surface.aclwupt surface.aclwuptc surface.acsnom surface.acsnow surface.acswdnb
surface.acswdnbc surface.acswdnt surface.acswdntc surface.acswupb surface.acswupbc surface.acswupt
surface.acswuptc surface.albbck surface.albedo surface.br surface.canwat surface.cd
surface.cda surface.chklowq surface.chs surface.chs2 surface.ck surface.cka
surface.coszen surface.cpm surface.cqs2 surface.dz8w1 surface.ebal surface.embck
surface.emiss surface.exch_h surface.exch_m surface.fh surface.flhc surface.flqc
surface.fm surface.glw surface.grdflx surface.gz1oz0 surface.hfx surface.isltyp
surface.ivgtyp surface.kpbl surface.lai surface.lakemask surface.landmask surface.lh
surface.lwdnb surface.lwdnbc surface.lwdnt surface.lwdntc surface.lwupb surface.lwupbc
surface.lwupt surface.lwuptc surface.mavail surface.mol surface.noahres surface.p_hyd
surface.pblh surface.potevp surface.psfc surface.psih surface.psim surface.q2
surface.qfx surface.qgh surface.qsfc surface.qv1 surface.rainbl surface.refd_max
surface.regime surface.reslin surface.rib surface.rmol surface.sfcprs surface.sfcrunoff
surface.sfctmp surface.sh2o surface.shdmax surface.shdmin surface.smcrel surface.smois
surface.smstav surface.smstot surface.snoalb surface.snopcx surface.snotime surface.snow
surface.snowc surface.snowh surface.sr surface.sst surface.sstsk surface.swddif
surface.swddir surface.swddni surface.swdnb surface.swdnbc surface.swdnt surface.swdntc
surface.swdown surface.swnorm surface.swupb surface.swupbc surface.swupt surface.swuptc
surface.t2 surface.th2 surface.tmn surface.tsk surface.tslb surface.u10
surface.udrunoff surface.ust surface.ustm surface.v10 surface.vegfra surface.wspd
surface.xice surface.xicem surface.xland surface.z0 surface.znt surface.zol
surface.microphysics.rainnc surface.microphysics.rainncv surface.microphysics.sr surface.microphysics.snownc surface.microphysics.snowncv surface.microphysics.graupelnc
surface.microphysics.graupelncv held.scratch.cu_nca held.scratch.cu_pratec held.scratch.cu_rainc held.scratch.cu_raincv held.scratch.cu_rqccuten
held.scratch.cu_rqicuten held.scratch.cu_rqrcuten held.scratch.cu_rqscuten held.scratch.cu_rqvcuten held.scratch.cu_rthcuten surface.microphysics.scratch.mp_graupelnc
surface.microphysics.scratch.mp_graupelncv surface.microphysics.scratch.mp_rainnc surface.microphysics.scratch.mp_rainncv surface.microphysics.scratch.mp_snownc surface.microphysics.scratch.mp_snowncv surface.microphysics.scratch.mp_sr
lbc.forcing_tables lbc.lbc_weights_0
""".split())


def _inventory_config():
    from woof.config import RunConfig

    return RunConfig(
        nx=3, ny=2, nz=8, dx=2000.0, dy=2000.0, ztop=8000.0,
        dt=10.0, run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=10, ra_physics=4, cu_physics=1,
        # This CPU inventory fixture has no ocean ice. Avoid entering the
        # independent CuPy sea-ice initializer with NumPy-owned fields.
        seaice_prognostic=False,
        sf_sfclay_physics=1, sf_surface_physics=2, bl_pbl_physics=1)


def _attach_real_driver(state, cfg, physics_mod, xp):
    physics_mod.initialize_physics(
        state, cfg, noah_params=object(), radiation=lambda **kwargs: None,
        cumulus=lambda **kwargs: None)
    state._scratch["lbc_forcing_tables"] = xp.zeros((5,), xp.float32)
    state._scratch["lbc_weights_0"] = xp.zeros((2,), xp.float32)


def test_real_domainstate_physics_inventory_names_and_dtypes_are_pinned(
        monkeypatch):
    """Production constructors under a CPU CuPy stub; no synthetic state."""
    import woof.core.physics as physics_mod
    import woof.core.state as state_mod

    monkeypatch.setattr(state_mod, "cp", np)
    monkeypatch.setattr(state_mod, "DTYPE", np.float32)
    monkeypatch.setattr(physics_mod, "cp", np)
    monkeypatch.setattr(physics_mod, "DTYPE", np.float32)
    cfg = _inventory_config()
    state = state_mod.DomainState(cfg)
    _attach_real_driver(state, cfg, physics_mod, np)
    fields = collect_state_fields(state, backend="gpu")

    assert tuple(field.name for field in fields) == _REAL_DRIVER_INVENTORY
    integer_fields = {
        field.name for field in fields
        if np.dtype(field.values.dtype) == np.dtype(np.int32)
    }
    assert integer_fields == {
        "surface.ebal", "surface.isltyp", "surface.ivgtyp", "surface.kpbl"}
    # surface.utype_urb2d is the urban category map (sf_urban_physics 1-3),
    # absent from this default-off inventory and excluded like IVGTYP; the
    # urban inventories are classified by tests/
    # test_urban_default_off_identity.py.
    # fa2e5efd8, development branch: Noah mosaic added the
    # immutable int32 tile-category map. Mosaic is off in this constructor,
    # so the map is absent here but still belongs in the global exclusions.
    from woof.core.noah_mosaic import MOSAIC_CATEGORY_FIELDS

    assert MOSAIC_CATEGORY_FIELDS["mosaic_cat_index"] == "int32"
    assert GPU_INTEGER_EXCLUSIONS == {
        "surface.isltyp", "surface.ivgtyp", "surface.utype_urb2d",
        "surface.mosaic_cat_index",
        # Native CAM/SSiB Registry integer workspace indices.
        "surface.ideep2d", "surface.jt2d",
        "surface.maxg2d", "surface.lengath2d"} | {
        f"nest.scratch.nest_sint_{name}_{stag}"
        for name in ("ci", "ip", "cj", "jp") for stag in ("m", "x", "y")}
    assert all(np.dtype(field.values.dtype) in {
        np.dtype(np.float32), np.dtype(np.int32)} for field in fields)


@pytest.mark.gpu
def test_real_gpu_inventory_descriptor_build_handles_integer_fields():
    """Controller twin: the real constructor inventory must reach _refresh."""
    import cupy as cp
    import woof.core.physics as physics_mod
    from woof.core.state import DomainState

    cfg = _inventory_config()
    state = DomainState(cfg)
    _attach_real_driver(state, cfg, physics_mod, cp)
    validator = StateHealthValidator(state)
    validator._refresh()
    assert {field.name for field in validator.excluded_integer_fields} == {
        "surface.ivgtyp", "surface.isltyp"}
    assert {field.name for field in validator.fields
            if field.values.dtype == cp.int32} == {
                "surface.ebal", "surface.kpbl"}


class _GPUState:
    """Minimal state/scratch owner for controller-run GPU corruption twins."""

    def __init__(self, cp, field, corruption):
        self._cp = cp
        self._scratch = {}
        self.physics = None
        self.lateral_boundaries = None
        self._lateral_boundary_device = None
        self._extra = None
        baseline = 300.0 if field in ("surface.tsk", "surface.tslb") else 1.0
        value = cp.full((2, 3), cp.float32(baseline), dtype=cp.float32)
        value.reshape(-1)[4] = cp.float32(corruption)
        if field.startswith("held."):
            component = field.rsplit(".", 1)[-1]
            tendency = SimpleNamespace(**{component: value})
            self.physics = SimpleNamespace(
                pbl_tendencies=tendency, radiation_tendencies=None,
                cumulus_tendencies=None, fields={}, microphysics=None,
                rthratenlw=None, rthratensw=None, cu_nca=None,
                cu_pratec=None, cu_raincv=None, cu_rates=None,
                _pending_rainbl=None)
        elif field.startswith("lbc."):
            # Exercise the production packed-table branch.  Prepared real
            # states always own this resident allocation.
            self._scratch["lbc_forcing_tables"] = value
        elif field.startswith("nest."):
            self._extra = {"bad": value}
        elif field.startswith("surface."):
            leaf = field.rsplit(".", 1)[-1]
            self.physics = SimpleNamespace(
                pbl_tendencies=None, radiation_tendencies=None,
                cumulus_tendencies=None, fields={leaf: value},
                microphysics=None, rthratenlw=None, rthratensw=None,
                cu_nca=None, cu_pratec=None, cu_raincv=None, cu_rates=None,
                _pending_rainbl=None)
        else:
            setattr(self, field, value)

    def scratch(self, shape, slot):
        if slot not in self._scratch:
            self._scratch[slot] = self._cp.zeros(shape, dtype=self._cp.float32)
        return self._scratch[slot]


@pytest.mark.gpu
def test_gpu_validator_handles_nssl2_post_step_descriptor_high_watermark():
    import cupy as cp

    state = _GPUState(cp, "qvolg", 0.0)
    # Distinct allocations: the device build keeps one descriptor per
    # allocation, so aliasing every table onto one buffer would exercise the
    # de-duplication rather than the 527-descriptor high-water mark.
    extra = {f"post_step_{index:03d}": cp.ones((1,), dtype=cp.float32)
             for index in range(526)}
    validator = StateHealthValidator(state, extra_tables=extra)
    report = validator.validate(phase="nssl2-post-step-high-watermark")
    assert len(validator.fields) == 527
    assert report.ok


@pytest.mark.gpu
@pytest.mark.parametrize(("field", "corruption"), CORRUPTIONS)
def test_gpu_fused_gate_twins_attribute_every_uncovered_corruption(
        field, corruption):
    import cupy as cp

    state = _GPUState(cp, field, corruption)
    validator = StateHealthValidator(state, extra_tables=state._extra)
    report = validator.validate(phase="gpu-injected")
    expected = {
        "held.pbl.ru": "held.pbl.ru",
        "lbc.d01.u.west.value": "lbc.forcing_tables",
        "nest.d02.theta.west.value": "nest.bad",
    }.get(field, field)
    assert not report.ok
    assert report.first_bad_field == expected
    assert report.first_bad_flat_index == 4


class _ChunkedGPUState:
    """Device state whose scanned fields span several gate chunks each.

    The fused gate spreads one descriptor over ``ceil(size / chunk)`` blocks,
    so a fault that only ever lands in the first chunk would not exercise the
    geometry at all.  16,384 elements is four chunks at this inventory size.
    """

    def __init__(self, cp):
        self._cp = cp
        self._scratch = {}
        self.physics = None
        self.lateral_boundaries = None
        self._lateral_boundary_device = None
        self.u = cp.zeros((4, 64, 64), dtype=cp.float32)
        self.w = cp.zeros((4, 64, 64), dtype=cp.float32)

    def scratch(self, shape, slot):
        if slot not in self._scratch:
            self._scratch[slot] = self._cp.zeros(shape, dtype=self._cp.float32)
        return self._scratch[slot]


@pytest.mark.gpu
@pytest.mark.parametrize(("corruption", "flat"), (
    (np.nan, 9000),
    (np.inf, 13000),
    (-np.inf, 4096),        # the first element of a chunk
    (-300.0, 12287),        # finite, but outside the |w| <= 200 bound
    (250.0, 16383),         # the very last element of the last chunk
))
def test_gpu_chunked_gate_catches_a_fault_outside_the_first_chunk(
        corruption, flat):
    import cupy as cp

    state = _ChunkedGPUState(cp)
    state.w.reshape(-1)[flat] = cp.float32(corruption)
    validator = StateHealthValidator(state)
    report = validator.validate(phase="gpu-chunked")
    assert validator._chunk_count > 1
    assert flat >= validator._chunk
    assert not report.ok
    assert report.first_bad_field == "w"
    assert report.first_bad_flat_index == flat
    assert report.status_bits != 0
    assert report.reason


@pytest.mark.gpu
def test_gpu_chunked_gate_keeps_index_first_then_field_first_attribution():
    """Chunking must not disturb which fault is reported.

    The lowest flat index within a field wins even when a later chunk finds
    its own fault first, and the lowest field id still wins overall -- both
    come from the same atomicMin over (field << 48) | index.
    """
    import cupy as cp

    state = _ChunkedGPUState(cp)
    for flat in (15000, 9000, 4100, 12000):
        state.w.reshape(-1)[flat] = cp.float32(np.nan)
    validator = StateHealthValidator(state)
    report = validator.validate(phase="gpu-chunked-many")
    mirror = validate_fields_cpu({"u": cp.asnumpy(state.u),
                                  "w": cp.asnumpy(state.w)})
    assert not report.ok
    assert report.first_bad_field == "w"
    assert report.first_bad_flat_index == 4100
    assert (report.first_bad_field, report.first_bad_flat_index) == (
        mirror.first_bad_field, mirror.first_bad_flat_index)

    state.u.reshape(-1)[15000] = cp.float32(np.inf)
    report = validator.validate(phase="gpu-chunked-field-first")
    assert report.first_bad_field == "u"
    assert report.first_bad_flat_index == 15000


class _AliasedGPUState:
    """Device state registering a cumulus tendency under both spellings.

    ``collect_state_fields`` reaches ``driver.cu_rates`` and the restart-
    persistent ``cu_rthcuten`` scratch slot separately; on a real prepared
    state they are one allocation.
    """

    def __init__(self, cp, held, scratch):
        self._cp = cp
        self._scratch = {"cu_rthcuten": scratch}
        self.lateral_boundaries = None
        self._lateral_boundary_device = None
        self.physics = SimpleNamespace(
            pbl_tendencies=None, radiation_tendencies=None,
            cumulus_tendencies=None, fields={}, microphysics=None,
            rthratenlw=None, rthratensw=None, cu_nca=None, cu_pratec=None,
            cu_raincv=None, cu_rates=SimpleNamespace(rthcuten=held),
            _pending_rainbl=None)

    def scratch(self, shape, slot):
        if slot not in self._scratch:
            self._scratch[slot] = self._cp.zeros(shape, dtype=self._cp.float32)
        return self._scratch[slot]


@pytest.mark.gpu
@pytest.mark.parametrize("aliased", (True, False))
def test_gpu_gate_scans_an_aliased_buffer_once_and_still_attributes_it(
        aliased):
    import cupy as cp

    held = cp.zeros((2, 3), dtype=cp.float32)
    scratch = held if aliased else cp.zeros((2, 3), dtype=cp.float32)
    held.reshape(-1)[4] = cp.float32(np.nan)
    validator = StateHealthValidator(_AliasedGPUState(cp, held, scratch))
    report = validator.validate(phase="gpu-aliased")
    assert [field.name for field in validator.fields] == (
        ["held.cu_rates.rthcuten"] if aliased
        else ["held.cu_rates.rthcuten", "held.scratch.cu_rthcuten"])
    assert not report.ok
    assert report.first_bad_field == "held.cu_rates.rthcuten"
    assert report.first_bad_flat_index == 4


def test_a_fast_sbm_domain_takes_the_fsbm_ceiling():
    """A nested FAST_SBM (mp_physics=30) tree measured 2,889 descriptors
    and stopped at its first validation against 1024.  A state carrying the
    bins gets FSBM_HEALTH_FIELDS, the config rule preflight prices by
    agrees, and every other state and config is unchanged."""
    from woof.core.health import (FSBM_HEALTH_FIELDS,
                                   health_field_ceiling,
                                   health_field_ceiling_for_config)

    one = np.ones((1,), dtype=np.float32)
    fsbm = SimpleNamespace(fast_sbm_bins=one, physics=None,
                           lateral_boundaries=None, _scratch={})
    plain = SimpleNamespace(qvolg=one, physics=None, lateral_boundaries=None,
                            _scratch={})
    assert FSBM_HEALTH_FIELDS == 4096
    assert health_field_ceiling(fsbm) == FSBM_HEALTH_FIELDS
    assert health_field_ceiling(plain) == MAX_HEALTH_FIELDS
    for mp, want in ((30, FSBM_HEALTH_FIELDS), (28, MAX_HEALTH_FIELDS),
                     (8, MAX_HEALTH_FIELDS), (0, MAX_HEALTH_FIELDS)):
        cfg = SimpleNamespace(mp_physics=mp)
        assert health_field_ceiling_for_config(cfg) == want


def test_successful_required_health_releases_replaced_array_generations(monkeypatch):
    """A completed check must not keep old cadence carriers alive next step."""
    import weakref
    from woof.core.health import HealthField, ValidationReport, rule_for_field

    state = SimpleNamespace(values=np.ones(8, dtype=np.float32),
                            auxiliary=np.ones(8, dtype=np.float32),
                            integers=np.ones(8, dtype=np.int32))
    validator = StateHealthValidator.__new__(StateHealthValidator)
    validator.state = state
    reports = []

    def scan(*, phase):
        validator.fields = (HealthField('u', state.values, rule_for_field('u'),
                                        auxiliary=state.auxiliary),)
        validator.excluded_integer_fields = (
            HealthField('surface.ivgtyp', state.integers, rule_for_field('u')),)
        report = ValidationReport(not np.isnan(state.values).any(), 0, phase=phase)
        reports.append(report)
        return report

    monkeypatch.setattr(validator, 'validate', scan)
    old = tuple(weakref.ref(a) for a in (state.values, state.auxiliary, state.integers))
    assert validator.require_healthy(phase='old') is reports[-1]
    for name in ('values', 'auxiliary', 'integers'):
        setattr(state, name, getattr(state, name).copy())
    assert all(ref() is None for ref in old)
    # Recollection still sees newly allocated carriers and preserves a failed
    # scan's descriptors for inspection instead of weakening the gate.
    state.values[3] = np.nan
    with pytest.raises(HealthCheckError):
        validator.require_healthy(phase='replacement')
    assert validator.fields[0].values is state.values
    assert validator.excluded_integer_fields[0].values is state.integers
    assert reports[-1].phase == 'replacement'


@pytest.mark.gpu
def test_gpu_required_health_recollects_replaced_arrays_and_catches_corruption():
    import weakref
    import cupy as cp

    state = _GPUState(cp, 'u', 1.0)
    validator = StateHealthValidator(state)
    old = weakref.ref(state.u)
    assert validator.require_healthy(phase='original').ok
    assert validator.fields == ()
    state.u = state.u.copy()
    assert old() is None
    state.u.reshape(-1)[4] = cp.float32(cp.nan)
    with pytest.raises(HealthCheckError) as failure:
        validator.require_healthy(phase='replacement')
    report = failure.value.report
    assert report.first_bad_field == 'u'
    assert report.first_bad_flat_index == 4
    assert report.phase == 'replacement'
    assert validator.fields[0].values is state.u
