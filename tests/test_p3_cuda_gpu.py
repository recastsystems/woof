"""Device gates for the P3 CUDA port (``mp_physics=50``).

Split from tests/test_p3_cuda.py because tests/conftest.py marks a whole
module ``gpu`` as soon as cupy appears in a helper, and the CPU-side gates
in that file must still run on a CPU-only invocation.

Every gate here names the breakage it prevents.  The per-field agreement
against running Fortran is a campaign, not a test: it lives in
``evidence/p3-cuda-20260829/`` with the p3-fortref lane's fixtures and its
arm-A dumps.
"""
from __future__ import annotations

import numpy as np


def _skip_control_under_strict_build(what):
    """GPUWM_WRF_EXACT=1 rewrites every compile's arithmetic options to the
    strict set, so a control built with the opposite flag cannot exist in
    that process; the positive half of the gate has already run."""
    from woof import wrf_exact
    if wrf_exact.ENABLED:
        import pytest
        pytest.skip(f"{what}: the control build is rewritten to the strict "
                    "options under GPUWM_WRF_EXACT=1")


def _module(options):
    import cupy as cp                                    # noqa: F401
    from woof.core import p3_device as PD
    return PD.p3_module(options)


def test_contraction_is_off_and_the_probe_can_tell():
    """``-fmad=false`` is a CORRECTNESS flag for this port, not a tuning
    knob: the Fortran reference arm is built ``-ffp-contract=off``, so a
    contracted ``a*b+c`` on the device is a different number.  Measured on
    the twelve p3-fortref fixtures, turning contraction on moves 11 of 12
    cases -- up to 110 ULP on a single step and to complete trajectory
    divergence on the rest.

    This gate proves the flag still bites: under ``false`` the infix form
    this file is written in must equal the explicit _rn intrinsics, and
    under ``true`` it must not.  A test that only checked the equality
    would pass forever with contraction silently on."""
    import cupy as cp

    from woof.core import p3_device as PD

    rng = np.random.default_rng(20260829)
    n = 1 << 16
    a = cp.asarray(rng.standard_normal(n, dtype=np.float32))
    b = cp.asarray(rng.standard_normal(n, dtype=np.float32))
    c = cp.asarray(rng.standard_normal(n, dtype=np.float32))

    def run(options):
        mod = PD.p3_module(options)
        out = []
        for name in ("p3k_probe_infix", "p3k_probe_rn"):
            o = cp.zeros(n, dtype=cp.float32)
            mod.get_function(name)(((n + 255) // 256,), (256,),
                                   (a, b, c, o, np.int32(n)))
            out.append(cp.asnumpy(o))
        return out

    off_infix, off_rn = run(PD.DEFAULT_OPTIONS)
    assert "-fmad=false" in PD.DEFAULT_OPTIONS
    assert off_infix.tobytes() == off_rn.tobytes()
    _skip_control_under_strict_build("the -fmad=true control")
    on_infix, on_rn = run(("-std=c++17", "-fmad=true"))
    assert on_infix.tobytes() != on_rn.tobytes(), (
        "the contraction probe cannot distinguish the two builds, so the "
        "equality above proves nothing")


def _tiny_column_case(nk=24, ncol=8):
    """A saturated mixed-phase slab that actually reaches every kernel.

    Built from a real profile rather than round numbers: an unsaturated
    column clips its condensate away in k_loop_1 and every later kernel
    returns immediately, which makes a byte comparison between two arms
    trivially true.  The positive-evidence assertions in the byte gate are
    what caught exactly that.
    """
    rng = np.random.default_rng(4)
    k = np.arange(nk, dtype=np.float64)
    z = 250.0 * k
    pres = 100000.0 * np.exp(-z / 8500.0)
    t = 295.0 - 6.5e-3 * z
    th = t * (100000.0 / pres) ** 0.2857
    # Tetens over liquid, only to place qv near saturation; the scheme uses
    # its own polysvp1.
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    qsat = 0.622 * es / (pres - es)
    qv = 1.02 * qsat                      # slightly supersaturated
    f = {}
    for name in ("qc", "qr", "nr", "qi", "qir", "ni", "qib"):
        f[name] = np.zeros((nk, ncol), dtype=np.float32)
    def col(a):
        return np.repeat(np.asarray(a, dtype=np.float32)[:, None], ncol,
                         axis=1)
    f["th"] = col(th)
    f["pres"] = col(pres)
    f["dz"] = np.full((nk, ncol), 250.0, dtype=np.float32)
    f["qv"] = col(qv)
    warm = t > 273.15
    cold = ~warm
    f["qc"][warm] = 8.0e-4
    f["qr"][warm] = 3.0e-4
    f["nr"][warm] = 2.0e4
    f["qi"][cold] = 4.0e-4
    f["ni"][cold] = 5.0e4
    f["qir"][cold] = 1.0e-4
    f["qib"][cold] = 2.0e-7
    f["th_old"] = f["th"].copy()
    f["qv_old"] = f["qv"].copy()
    f["nc"] = np.zeros((nk, ncol), dtype=np.float32)
    f["ssat"] = np.zeros((nk, ncol), dtype=np.float32)
    # column-to-column spread so the gate is not one column repeated
    f["qv"] = (f["qv"] * (1.0 + 0.01 * rng.standard_normal((nk, ncol)))
               ).astype(np.float32)
    return f, nk, ncol


def _run_arm(arm, steps=3, options=None):
    import cupy as cp

    from woof.core import p3_device as PD

    host, nk, ncol = _tiny_column_case()
    fields = {k: cp.asarray(v) for k, v in host.items()}
    diag = {d: cp.zeros((nk, ncol), dtype=cp.float32) for d in PD.DIAG_SLOTS}
    surf = {s: cp.zeros(ncol, dtype=cp.float32) for s in PD.SURF_SLOTS}
    ws = PD.make_workspace(ncol, nk)
    for it in range(1, steps + 1):
        PD.run_p3_device(fields, diag, surf, workspace=ws, dt=20.0, it=it,
                         arm=arm,
                         options=options or PD.DEFAULT_OPTIONS)
    out = {("f", k): cp.asnumpy(v) for k, v in fields.items()}
    out.update({("d", k): cp.asnumpy(v) for k, v in diag.items()})
    out.update({("s", k): cp.asnumpy(v) for k, v in surf.items()})
    return out


def test_subnormal_results_survive_and_the_probe_can_tell():
    """``--ftz=false`` is a correctness flag too.  gfortran on x86 keeps a
    float32 result that is subnormal; through ``cupy.RawModule`` the unit
    was compiled with CuPy's appended ``-ftz=true`` and flushed every such
    result to zero, which the column oracle against WRF 4.6.1 measured as
    up to 19 ULP in a snow column's ice fields after 28 steps
    (tools/p3_wrf461_oracle/README.md).  Under the shipped options a product
    of two normal floats with a subnormal result must come back subnormal,
    and under an explicit ``--ftz=true`` it must not, so the probe is shown
    to bite."""
    import cupy as cp

    from woof.core import p3_device as PD

    a = cp.asarray(np.full(256, 1.5e-20, dtype=np.float32))
    b = cp.asarray(np.full(256, 3.0e-20, dtype=np.float32))
    c = cp.zeros(256, dtype=cp.float32)
    want = np.float32(np.float32(1.5e-20) * np.float32(3.0e-20))
    assert want != 0 and abs(want) < np.finfo(np.float32).tiny

    def run(options):
        o = cp.zeros(256, dtype=cp.float32)
        PD.p3_module(options).get_function("p3k_probe_rn")(
            (1,), (256,), (a, b, c, o, np.int32(256)))
        return cp.asnumpy(o)

    assert "--ftz=false" in PD.DEFAULT_OPTIONS
    kept = run(PD.DEFAULT_OPTIONS)
    assert kept.view(np.uint32).tolist() == [want.view(np.uint32)] * 256
    _skip_control_under_strict_build("the --ftz=true control")
    flushed = run(("-std=c++17", "-fmad=false", "--ftz=true"))
    assert not flushed.any(), (
        "the subnormal probe cannot distinguish the two builds, so the "
        "survival assertion above proves nothing")


def test_the_fused_arm_reproduces_the_unfused_one_byte_for_byte():
    """Fusing reorders floating point and CAN change the numbers, so the
    unfused arm is the reference and the fused one must reproduce it
    exactly.  This program shipped a 9x vectorisation that was wrong in 39
    elements out of 37.8 million and only a byte gate caught it -- and that
    gate existed only because an unfused original did.

    Compared with ``tobytes()``, not ``allclose``: the failure this is
    guarding against is a last-bit one."""
    import cupy as cp                                    # noqa: F401

    unfused = _run_arm("unfused")
    fused = _run_arm("fused")
    assert set(unfused) == set(fused)
    moved = [k for k in unfused
             if unfused[k].tobytes() != fused[k].tobytes()]
    assert not moved, moved
    # Positive evidence both arms actually integrated.  An exact-0.0 delta
    # between two arms that never ran is also zero, so the byte gate above
    # is necessary and not sufficient: these assert the run MOVED state.
    host, _, _ = _tiny_column_case()
    moved_state = [k for k in ("qc", "qr", "qi", "qv", "th", "ni")
                   if unfused[("f", k)].tobytes() != host[k].tobytes()]
    assert len(moved_state) >= 5, moved_state
    assert float(np.abs(unfused[("d", "zdbz")]).max()) > 0.0
    assert float(np.abs(unfused[("f", "nc")]).max()) > 0.0


def test_the_block_size_is_an_occupancy_knob_and_not_a_number():
    """One thread is one column, so a different block size must change
    nothing.  If it ever does, the port has acquired a cross-thread
    dependence it is not supposed to have."""
    import cupy as cp

    from woof.core import p3_device as PD

    host, nk, ncol = _tiny_column_case()

    def go(block):
        fields = {k: cp.asarray(v) for k, v in host.items()}
        diag = {d: cp.zeros((nk, ncol), dtype=cp.float32)
                for d in PD.DIAG_SLOTS}
        surf = {s: cp.zeros(ncol, dtype=cp.float32) for s in PD.SURF_SLOTS}
        ws = PD.make_workspace(ncol, nk)
        for it in (1, 2):
            PD.run_p3_device(fields, diag, surf, workspace=ws, dt=20.0,
                             it=it, block=block)
        return {k: cp.asnumpy(v) for k, v in fields.items()}

    a, b = go(32), go(256)
    moved = [k for k in a if a[k].tobytes() != b[k].tobytes()]
    assert not moved, moved


def test_the_prognostic_nc_path_runs_and_changes_nc():
    """SEAM 3, exercised rather than asserted.  The specified-Nc path
    rewrites nc from nccnst/rho every call; the prognostic path integrates
    it.  If the two produced the same nc the branch would be decorative."""
    import cupy as cp                                    # noqa: F401

    spec = _run_arm("unfused", steps=2)
    import cupy as cp2

    from woof.core import p3_device as PD

    host, nk, ncol = _tiny_column_case()
    fields = {k: cp2.asarray(v) for k, v in host.items()}
    fields["nc"][...] = cp2.float32(1.0e8)
    diag = {d: cp2.zeros((nk, ncol), dtype=cp2.float32)
            for d in PD.DIAG_SLOTS}
    surf = {s: cp2.zeros(ncol, dtype=cp2.float32) for s in PD.SURF_SLOTS}
    ws = PD.make_workspace(ncol, nk)
    for it in (1, 2):
        PD.run_p3_device(fields, diag, surf, workspace=ws, dt=20.0, it=it,
                         log_predictNc=True, arm="unfused")
    prog_nc = cp2.asnumpy(fields["nc"])
    assert np.isfinite(prog_nc).all()
    # the specified path rewrites nc to nccnst/rho every call; the
    # prognostic path integrates it from the activation step, so a cell
    # with cloud must hold a different number under the two flags
    assert prog_nc.tobytes() != spec[("f", "nc")].tobytes()
    assert float(prog_nc.max()) > 0.0


def test_the_tables_stay_resident_and_the_state_never_leaves_the_device():
    """A per-step host transfer of a prognostic field is not completion.

    The gate is structural: after a step every prognostic field is still a
    cupy array on the same device, the table upload is cached so a second
    call adds no allocation, and the module's own accounting reports the
    resident table bytes."""
    import cupy as cp

    from woof.core import p3_device as PD

    t1 = PD.device_tables()
    t2 = PD.device_tables()
    assert t1 is t2, "p3_init's tables were uploaded twice"
    assert t1.nbytes == (14000 + 60000 + 3 * 3000) * 4
    for arr in t1.arrays:
        assert isinstance(arr, cp.ndarray)

    host, nk, ncol = _tiny_column_case()
    fields = {k: cp.asarray(v) for k, v in host.items()}
    diag = {d: cp.zeros((nk, ncol), dtype=cp.float32) for d in PD.DIAG_SLOTS}
    surf = {s: cp.zeros(ncol, dtype=cp.float32) for s in PD.SURF_SLOTS}
    ws = PD.make_workspace(ncol, nk)
    PD.run_p3_device(fields, diag, surf, workspace=ws, dt=20.0, it=1)
    for name, arr in fields.items():
        assert isinstance(arr, cp.ndarray), name
    assert PD.scratch_bytes_per_cell() == 72


def test_the_registry_prices_exactly_the_slots_p3_asks_for():
    """The dynamic half of the completeness gate, and the reason
    ``("woof/core/p3.py", "apply")`` is on the variable-slot allowlist in
    tests/test_preflight.py.

    A real mp=50 step runs with ``DomainState.scratch`` instrumented, and
    the requested slot set must equal the registry's P3 rows in BOTH
    directions -- so neither an unpriced allocation nor a stale registry row
    survives.  The MYNN lane's gate has the same shape for the same
    reason."""
    import cupy as cp                                    # noqa: F401

    from woof.config import RunConfig, validate_run_config
    from woof.core import preflight as pf
    from woof.core.microphysics import apply as apply_microphysics
    from woof.core.state import DomainState

    cfg = RunConfig(nx=8, ny=8, nz=12, dx=1000.0, dy=1000.0, dt=10.0,
                    ztop=12000.0, run_seconds=60.0, moist=True,
                    mp_physics=50)
    validate_run_config(cfg)
    state = DomainState(cfg)
    asked: set[str] = set()
    original = type(state).scratch

    def spy(self, shape, slot, dtype=None):
        asked.add(slot)
        return original(self, shape, slot, dtype)

    type(state).scratch = spy
    try:
        apply_microphysics(state, cfg, float(cfg.dt))
    finally:
        type(state).scratch = original

    registry = set(pf.scratch_slot_registry(cfg, n_lbc_intervals=2))
    p3_rows = {n for n in registry
               if n.startswith("p3_") or n.startswith("mp_")}
    p3_asked = {n for n in asked if n.startswith("p3_") or n.startswith("mp_")}
    assert p3_asked - p3_rows == set(), sorted(p3_asked - p3_rows)
    assert p3_rows - p3_asked == set(), sorted(p3_rows - p3_asked)


def test_the_composed_unit_compiles_to_the_frame_it_is_priced_at():
    """The one platform check the ``.cu`` enumeration cannot reach.

    ``preflight.KERNEL_MAX_LOCAL_SIZE_BYTES`` is a CEILING over named
    compile platforms, re-read per platform in
    ``woof/core/kernel_frame_recordings.py`` and compared against the
    driver by ``tests/test_preflight.py::
    test_the_recorded_local_frames_match_the_driver``.  Both of those walk
    ``woof/core/kernels/*.cu`` and drop whatever NVRTC refuses standalone,
    and ``p3.cu`` is refused: it borrows the tree's single audited glibc
    r_pow/r_exp/r_log from ``noahmp_leaves.cu`` rather than carrying a
    second copy that could drift.  So an ``mp_physics = 50`` domain is
    priced from ``CHAINED_TRANSLATION_UNIT_FRAMES["p3_composed"]``, one
    reading taken on sm_120 / cupy 14.2.0, and
    ``under_priced_kernel_frames`` is structurally unable to notice it
    drifting -- the composed unit is not in the ``observed`` dict it is
    handed.

    THE BREAKAGE THIS PREVENTS, in the unit that breaks: the priced row is
    0 B, and 0 B is structural rather than lucky -- ``p3.cu`` declares no
    per-thread column array at all, because P3 carries ONE ice category
    with a rime pair (qir mass, qib volume) and no qs and no qg, and all
    eighteen ``(nk, ncol)`` companions live in the global workspace
    ``p3_device.make_workspace()`` allocates.  What is NOT structural is
    SPILLING: ``p3k_kloopmain`` sits at 244 registers and the fused
    ``p3k_fused_process`` at 250, against a 255-register ceiling, so
    another architecture or NVRTC build can spill where the recorded one
    did not.  A spilled frame would make the local-memory reservation
    under-charge by the frame times the whole resident-thread capacity,
    and a run this card cannot hold would be admitted by ``woof check``
    and OOM later with nothing pointing back here.  This gate is what
    turns that into a red test on any box with a device.
    """
    import re
    from pathlib import Path

    import cupy as cp

    import woof.core.kernels as K
    from woof.core import p3_device as PD
    from woof.core import preflight as pf

    row = pf.CHAINED_TRANSLATION_UNIT_FRAMES["p3_composed"]
    # The kernels the UNIT launches are p3.cu's own; noahmp_leaves.cu is in
    # the same translation unit but P3 launches none of it, and it already
    # has a standalone row in every recording.
    p3_cu = Path(K.__file__).resolve().parent / "p3.cu"
    symbol = re.compile(
        r'extern\s+"C"\s+__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)')
    text = p3_cu.read_text(encoding="utf-8")
    instantiated = re.findall(r"^P3_BOTH_CATEGORY_COUNTS\(([^)]+)\)", text, re.M)
    names = sorted({PD.kernel_name(name, cat) for name in instantiated for cat in (1,2)}
                   | {name for name in symbol.findall(text) if name != "name"})
    assert len(names) >= 12, names

    module = PD.p3_module(PD.DEFAULT_OPTIONS)
    frames = {name: int(cp.cuda.driver.funcGetAttribute(
                  cp.cuda.driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                  module.get_function(name).ptr))
              for name in names}
    widest = max(frames.values())
    profile = pf.local_memory_profile_from_device(cp)
    unpriced = ((widest - row.max_local_size_bytes)
                * profile.resident_thread_capacity)
    assert widest <= row.max_local_size_bytes, (
        f"the P3 composed unit compiles to {widest} B per thread on this "
        f"platform ({profile.name}) against the {row.max_local_size_bytes} "
        "B woof/core/preflight.py CHAINED_TRANSLATION_UNIT_FRAMES prices "
        f"it at, so every mp_physics=50 run under-charges the local-memory "
        f"reservation by {unpriced / 1024 ** 3:.3f} GiB and can be admitted "
        "on a card that cannot hold it.  Widest kernels: "
        + ", ".join(f"{n} {b} B" for n, b in
                    sorted(frames.items(), key=lambda kv: -kv[1])[:3])
        + ".  Remedy: move the row to this reading and record the platform "
        "in CHAINED_UNITS_WITHOUT_A_PER_PLATFORM_ROW "
        "(woof/core/kernel_frame_recordings.py)")
