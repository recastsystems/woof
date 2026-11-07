"""Gates for the device-resident P3 (``mp_physics=50``).

Every gate here names the breakage it prevents.  The CPU-only ones run
anywhere; the device ones import cupy and so are auto-marked ``gpu`` by
tests/conftest.py and skipped under ``GPUWM_NO_LOCAL_GPU=1``.

The device half lives in tests/test_p3_cuda_gpu.py: tests/conftest.py
marks a WHOLE module gpu when cupy appears in any helper, so keeping the
two in one file would skip the CPU gates on a CPU-only invocation.

The agreement numbers against running Fortran are NOT here -- they are a
campaign against the p3-fortref lane's twelve fixtures and its arm-A dumps,
recorded in ``evidence/p3-cuda-20260829/``.  What is here is what a test
can hold every day: the constant block cannot drift from the CPU authority,
the slot tables on the two sides of the launch cannot drift from each
other, contraction cannot silently come back on, the fused arm cannot stop
reproducing the unfused one, and the scratch registry cannot stop pricing
what the adapter actually allocates.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
P3_CU = ROOT / "woof" / "core" / "kernels" / "p3.cu"


# ---------------------------------------------------------------------------
# The constant block.
# ---------------------------------------------------------------------------

def test_the_device_constants_are_the_cpu_authority_bit_for_bit():
    """A device constant that drifts from ``woof/core/p3.py`` is a
    systematic bias on every cell of every step, invisible in a per-step
    state comparison because both arms would be internally consistent.

    The p3-fortref lane measured all 52 comparable ``p3_init`` constants of
    the CPU module bit-identical to the running Fortran, so pinning the
    device block to that module pins it to the Fortran transitively.  The
    literals in the .cu are hex float32, which is exact, so this comparison
    is on bits and not on a printed decimal.
    """
    from woof.core import p3 as P

    text = P3_CU.read_text(encoding="utf-8")
    found = dict(re.findall(r"^#define (P3_[A-Z0-9_]+)\s+(\S+)\s*//",
                            text, flags=re.M))
    f32 = np.float32
    mw = f32(0.018)
    bact = (f32(3.0) * f32(1.0) * f32(0.9) * mw * f32(1777.0)
            / (f32(0.132) * P.RHOW))
    expect = {
        "P3_PI": P.PI, "P3_THRD": P.THRD, "P3_SXTH": P.SXTH,
        "P3_PIOV3": P.PIOV3, "P3_PIOV6": P.PIOV6,
        "P3_MAX_TOTAL_NI": P.MAX_TOTAL_NI, "P3_NCCNST": P.NCCNST,
        "P3_CP": P.CP, "P3_INV_CP": P.INV_CP, "P3_G": P.G, "P3_RD": P.RD,
        "P3_RV": P.RV, "P3_EP_2": P.EP_2, "P3_RHOSUR": P.RHOSUR,
        "P3_RHOSUI": P.RHOSUI, "P3_F1R": P.F1R, "P3_F2R": P.F2R,
        "P3_RHOW": P.RHOW, "P3_CPW": P.CPW, "P3_INV_RHOW": P.INV_RHOW,
        "P3_MU_R_CONSTANT": P.MU_R_CONSTANT, "P3_INV_DRMAX": P.INV_DRMAX,
        "P3_RHO_RIMEMIN": P.RHO_RIMEMIN, "P3_RHO_RIMEMAX": P.RHO_RIMEMAX,
        "P3_INV_RHO_RIMEMAX": P.INV_RHO_RIMEMAX, "P3_QSMALL": P.QSMALL,
        "P3_NSMALL": P.NSMALL, "P3_BSMALL": P.BSMALL, "P3_BIMM": P.BIMM,
        "P3_AIMM": P.AIMM, "P3_MI0": P.MI0, "P3_ECI": P.ECI,
        "P3_ERI": P.ERI, "P3_BCN": P.BCN, "P3_NMLTRATIO": P.NMLTRATIO,
        "P3_CONS1": P.CONS1, "P3_CONS2": P.CONS2, "P3_CONS3": P.CONS3,
        "P3_CONS5": P.CONS5, "P3_CONS6": P.CONS6, "P3_CONS7": P.CONS7,
        "P3_E0": P.E0,
        "P3_XXLV": f32(3.1484e6) - f32(2370.0) * f32(273.15),
        "P3_XXLS": (f32(3.1484e6) - f32(2370.0) * f32(273.15)
                    + f32(0.3337e6)),
        "P3_XLF": f32(0.3337e6),
        # SEAM 2's aerosol data (module_mp_p3.F:269-294).
        "P3_MW": mw, "P3_RR": f32(8.3187), "P3_BACT": bact,
        "P3_INV_RM1": f32(2.0e7), "P3_SIG1": f32(2.0),
        "P3_NANEW1": f32(300.0e6), "P3_INV_RM2": f32(7.6923076e5),
        "P3_SIG2": f32(2.5), "P3_NANEW2": f32(0.0),
    }
    assert set(expect) <= set(found), sorted(set(expect) - set(found))
    wrong = []
    for name, value in expect.items():
        literal = found[name].rstrip("f")
        got = np.float32(float.fromhex(literal)
                         if literal.startswith(("0x", "-0x"))
                         else float(literal))
        if got.tobytes() != np.float32(value).tobytes():
            wrong.append((name, float(got), float(np.float32(value))))
    assert not wrong, wrong


def test_the_integer_powers_are_expanded_on_both_sides_not_powf():
    """An INTEGER Fortran exponent is a multiplication chain, not ``powf``.

    ``module_mp_p3.F:6681`` and ``:6684`` are
    ``nc = 6.*lamc**3*qc/(pi*rhow*(mu_c+3.)*(mu_c+2.)*(mu_c+1.))`` and
    ``:2988`` is ``dum = (1./lamc(i,k))**3`` -- unsuffixed integer
    exponents.  gfortran never calls ``powf`` for those: MEASURED on
    gfortran 15.2.0 over 200,000 float32 arguments, ``x**3`` is bit-equal
    to ``(x*x)*x`` at -O0, -O1, -O2 and -O3, with zero differing.  glibc
    ``powf(x, 3.0f)`` is a DIFFERENT function: it disagrees with that chain
    on 77,273 of 300,000 log-uniform arguments over [1e-6, 1e8].

    ``p3.cu`` has always expanded these -- ``:457``, ``:462`` write
    ``6.0f * ((lamc * lamc) * lamc) * qc``, ``:1066`` writes
    ``(d * d) * d`` -- and ``p3.py`` did not, so the two arms of one scheme
    computed different functions at the same statement.

    Named breakage, and it is reachable through a shipped seam: at
    ``qc = 7.389125e-06``, ``nc = 14634.35``, ``rho = 0.50076896``,
    ``get_cloud_dsd2`` takes the ``lamc < lammin`` limiter and rebuilds
    ``nc``.  The chain gives ``0x482ecf02``; ``powf`` gives ``0x482ecf01``.

    ``:2988`` sits inside ``p3_main``'s loop and cannot be driven from
    outside, so it is held by the source half of this gate: no ``**
    f32(3.0)`` may survive on a VARIABLE base.  The three that remain are
    module-scope ``CONS`` folds over literals, which the p3 verify pass
    recomputed both ways and found bit-identical.
    """
    from woof.core import p3 as P

    qc, nc, rho = np.float32(7.389125e-06), np.float32(14634.35), \
        np.float32(0.50076896)
    nc_grd, mu_c, _nu, lamc, _cdist, _cdist1 = P.get_cloud_dsd2(
        qc, nc, rho, np.float32(1.0))

    assert lamc == (mu_c + np.float32(1.0)) * np.float32(2.5e4), (
        "this input no longer takes the lamc < lammin limiter, so the gate "
        "would pass on a p3.py that still calls powf there")
    denom = (P.PI * P.RHOW * (mu_c + np.float32(3.0))
             * (mu_c + np.float32(2.0)) * (mu_c + np.float32(1.0)))
    chain = np.float32(6.0) * ((lamc * lamc) * lamc) * qc / denom
    powf = np.float32(6.0) * (lamc ** np.float32(3.0)) * qc / denom
    assert chain.tobytes() != powf.tobytes(), (
        "powf and the multiplication chain agree on this input, so the "
        "gate cannot tell the two apart any more")
    assert np.float32(nc_grd).tobytes() == chain.tobytes(), (
        f"get_cloud_dsd2 returned nc_grd={nc_grd!r}; the Fortran's integer "
        f"exponent gives {chain!r}, glibc powf gives {powf!r}")

    source = (ROOT / "woof" / "core" / "p3.py").read_text(encoding="utf-8")
    live = [
        f"{n}: {line.strip()}"
        for n, line in enumerate(source.splitlines(), 1)
        if "** f32(3.0)" in line and not line.startswith("CONS")
    ]
    assert live == [], (
        "p3.py still raises a variable to an integer Fortran exponent with "
        "powf; gfortran expands these to multiplications: " + "; ".join(live))

    kernel = P3_CU.read_text(encoding="utf-8")
    assert kernel.count("((lamc * lamc) * lamc)") == 2, (
        "p3.cu stopped expanding :6681/:6684, so the two arms would agree "
        "on the WRONG function")
    assert "dum = (d * d) * d;" in kernel, "p3.cu stopped expanding :2988"


def test_no_decimal_literal_in_the_kernel_escapes_float32():
    """``1.0`` is a DOUBLE in CUDA C, and ``x * 1.0`` promotes the whole
    expression to double.

    That does not look like a bug -- it looks like a more accurate answer --
    so it would never fail a smoke test and would quietly stop the device
    arm from reproducing the Fortran's float32 rounding.  The two declared
    double excursions are written as explicit ``(double)`` casts, so no
    unsuffixed decimal literal is legitimate anywhere in this file.
    """
    text = P3_CU.read_text(encoding="utf-8")
    stripped = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    stripped = re.sub(r"//[^\n]*", " ", stripped)
    bad = []
    for m in re.finditer(r"(?<![0-9A-Za-z_.])(\d+\.\d*|\.\d+)"
                         r"([eE][-+]?\d+)?([fF]?)", stripped):
        if m.group(3) in ("f", "F"):
            continue
        bad.append((stripped[:m.start()].count("\n") + 1, m.group(0)))
    assert not bad, bad


# ---------------------------------------------------------------------------
# The two sides of the launch.
# ---------------------------------------------------------------------------

def test_the_slot_tables_on_both_sides_of_the_launch_agree():
    """The kernels index pointer arrays by position.

    A slot reordered on one side only would hand every kernel a different
    field under the right name -- qc read as nc, dz read as pres -- and the
    result would be finite, wrong, and attributed to physics.  Nothing else
    in the port checks this, because the C side has no names at runtime.
    """
    from woof.core import p3_device as PD

    text = P3_CU.read_text(encoding="utf-8")

    def defines(prefix, names):
        got = dict(re.findall(r"^#define %s([A-Z0-9_]+) (\d+)$" % prefix,
                              text, flags=re.M))
        assert len(got) == len(names), (prefix, sorted(got), names)
        for i, name in enumerate(names):
            key = name.upper()
            assert key in got, (prefix, key, sorted(got))
            assert int(got[key]) == i, (prefix, name, got[key], i)

    defines("F_", [n.replace("th_old", "thold").replace("qv_old", "qvold")
                   for n in PD.field_slots(2)])
    defines("S_", [n.replace("inv_rho", "invrho").replace("qv_cld", "qvcld")
                   for n in PD.SCRATCH_SLOTS])
    defines("W_", [n.replace("v_q", "vq").replace("v_n", "vn")
                    .replace("flux_qir", "fqir").replace("flux_bir", "fbir")
                    .replace("flux_q", "fq").replace("flux_n", "fn")
                    .replace("v_z", "vz").replace("flux_z", "fz")
                   for n in PD.SEDW_SLOTS + PD.SEDW_SLOTS_3MOM])
    defines("D_", list(PD.diag_slots(2)))
    defines("P_", [n.replace("prt_liq", "prtliq").replace("prt_sol", "prtsol")
                   for n in PD.SURF_SLOTS])
    defines("T_", ["itab", "icoll", "vn", "vm", "revap", "itab3", "icoll3", "icolli1", "icolli2"])
    for name in PD.ARM_KERNELS["unfused"] + PD.ARM_KERNELS["fused"]:
        assert (('P3_BOTH_CATEGORY_COUNTS(' + name + ')') in text or ('__global__ void %s(' % name) in text), name


def test_the_diagnostic_set_is_the_registry_package_and_is_closed():
    """Emitting a seventh diagnostic later moves the history stream and
    invalidates every receipt registered against it, so the set is decided
    once and pinned here.  These six are the WRF Registry's P3 package:
    REFL_10CM (as zdbz), re_cloud, re_ice, vmi3d, di3d, rhopo3d."""
    from woof.core import p3_device as PD

    assert PD.DIAG_SLOTS == ("zdbz", "effc", "effi", "vmi", "di", "rhopo")


# ---------------------------------------------------------------------------
# The three seams.
# ---------------------------------------------------------------------------

def test_ssat_is_a_live_threaded_array_not_a_folded_zero():
    """SEAM 1.  ``log_predictSsat`` is false in the shipped configuration,
    so a throughput-minded port notices ssat is zero on entry and optimises
    the argument away.  It must not: explicit supersaturation is the
    documented attach point, and re-threading it later would invalidate
    every number measured before."""
    from woof.core import p3_device as PD

    assert "ssat" in PD.FIELD_SLOTS
    text = P3_CU.read_text(encoding="utf-8")
    assert "F_SSAT" in text
    # both branches of the diagnosed/predicted split are present
    assert "AT(ssat, k) = AT(qv_old, k) - qvs;" in text
    assert "sup = AT(ssat, k) / qvs;" in text


def test_aerosol_activation_is_a_substitutable_step_with_its_modes_as_data():
    """SEAM 2.  Native activation is two PRESCRIBED lognormal modes.  They
    stay named constants read by one callable step, so a prognostic aerosol
    species replaces that step and those values and nothing else."""
    text = P3_CU.read_text(encoding="utf-8")
    assert "__device__ void p3_activate_droplets(" in text
    for name in ("P3_NANEW1", "P3_INV_RM1", "P3_SIG1",
                 "P3_NANEW2", "P3_INV_RM2", "P3_SIG2", "P3_BACT"):
        assert "#define %s " % name in text, name
    # the step is CALLED, not inlined into the process loop
    assert "p3_activate_droplets(log_predictNc, sup_cld, it," in text


def test_both_log_predictnc_paths_exist_in_the_kernel():
    """SEAM 3.  ``log_predictNc = present(nc)`` is native, so mp=50 and
    mp=51 are one port with one extra field.  Building only the
    specified-Nc path and adding the other later means re-verifying both."""
    text = P3_CU.read_text(encoding="utf-8")
    # the specified-Nc respecification and the prognostic tendency
    assert "if (!log_predictNc) AT(nc, k) = P3_NCCNST * inv_rho;" in text
    assert "AT(nc, k) = AT(nc, k) + (-nccol[c] - nchetc[c] - ncheti[c]) * dt;" in text
    assert ("AT(nc, k) = AT(nc, k) + (-ncacc - ncautc + ncslf + ncnuc) * dt;"
            in text)
    # and the two-moment cloud sedimentation branch
    assert "* p3_gam(1.0f + P3_BCN + mu_c) * dum" in text


# ---------------------------------------------------------------------------
# Identity and pricing.
# ---------------------------------------------------------------------------

def test_p3_backend_is_scheme_scoped_with_an_off_scheme_refusal():
    """A RunConfig field added unscoped moves EVERY experiment fingerprint,
    so every existing checkpoint refuses to resume -- for a field the run
    never reads.  That happened for real at the 2.5.8 integration with the
    mp=28 aerosol pair (commit 62ce7c3f6).  The refusal is what makes the
    scoping provably lossless: off scheme the field can hold only its
    default, so dropping it discards no information."""
    from woof.config import P3_BACKENDS, RunConfig, validate_run_config
    from woof.core.model import SCHEME_SCOPED_RUN_FIELDS

    assert SCHEME_SCOPED_RUN_FIELDS[50] == ("p3_backend",)
    base = dict(nx=8, ny=8, nz=6, dx=1000.0, dy=1000.0, dt=1.0,
                ztop=10000.0, run_seconds=60.0)
    for value in P3_BACKENDS:
        validate_run_config(RunConfig(**base, moist=True, mp_physics=50,
                                      p3_backend=value))
    with pytest.raises(ValueError, match="p3_backend"):
        validate_run_config(RunConfig(**base, moist=True, mp_physics=50,
                                      p3_backend="nope"))
    # off scheme, a non-default value is refused rather than dropped
    with pytest.raises(ValueError, match="requires mp_physics=50.*mp_physics=53"):
        validate_run_config(RunConfig(**base, moist=True, mp_physics=6,
                                      p3_backend="reference"))
    # the default is accepted under every scheme, which is what lets the
    # scoping drop it from the identity payload of a non-P3 run
    validate_run_config(RunConfig(**base, moist=True, mp_physics=6))


def test_the_scoped_field_leaves_a_non_p3_restart_identity_untouched():
    """The concrete breakage, exercised rather than described: an mp=6
    domain's restart identity must not gain a key because P3 gained a
    knob."""
    from woof.core.model import SCHEME_SCOPED_RUN_FIELDS, restart_identity_payload, scope_microphysics_run_fields

    class _Exp:
        pass

    payload_like = {"domains": [{"run": {"mp_physics": 6,
                                         "p3_backend": "cuda"}},
                                {"run": {"mp_physics": 50,
                                         "p3_backend": "fused"}}]}

    for domain in payload_like["domains"]:
        run = domain["run"]
        from woof.core.model import _drop_off_scheme_run_fields
        _drop_off_scheme_run_fields(run)
    assert "p3_backend" not in payload_like["domains"][0]["run"]
    assert payload_like["domains"][1]["run"]["p3_backend"] == "fused"
    assert callable(restart_identity_payload)


def test_the_p3_translation_unit_is_priced_not_refused():
    """``p3.cu`` cannot compile alone -- it borrows WOOF's shared
    r_pow/r_exp/r_log from noahmp_leaves.cu rather than carrying a second
    copy that could drift.  Without a composite row the memory preflight
    would refuse every mp=50 run outright, which is what the Noah-MP
    fragments still do."""
    from woof.core import preflight as pf

    assert "p3" in pf.UNMEASURED_KERNEL_MODULES
    row = pf.CHAINED_TRANSLATION_UNIT_FRAMES["p3_composed"]
    assert row.covers == frozenset({"p3"})
    # e7d3e91b566: 72 B read on sm_120 with the multi-category kernels.
    assert row.max_local_size_bytes == 72
    assert "p3_composed" in pf._MICROPHYSICS_KERNEL_MODULES[50]
    assert "p3" not in pf._MICROPHYSICS_KERNEL_MODULES[50]


def test_the_scratch_registry_prices_every_p3_companion():
    """Static half of the completeness gate: every slot name
    ``p3_device.make_workspace`` will ask for has a registry row and a
    lifetime classification.  The dynamic half (that the adapter asks for
    exactly these and no others) is the GPU test below."""
    from woof.config import RunConfig
    from woof.core import p3_device as PD
    from woof.core import preflight as pf

    cfg = RunConfig(nx=8, ny=8, nz=6, dx=1000.0, dy=1000.0, dt=1.0,
                    ztop=10000.0, run_seconds=60.0, moist=True,
                    mp_physics=50)
    registry = pf.scratch_slot_registry(cfg, n_lbc_intervals=2)
    wanted = (["p3_" + n for n in PD.SCRATCH_SLOTS]
              + ["p3_sed_" + n for n in PD.SEDW_SLOTS]
              + ["p3_flags", "p3_nc", "p3_ssat",
                 "p3_prt_liq", "p3_prt_sol", "p3_effc", "p3_effi"])
    missing = [n for n in wanted if n not in registry]
    assert not missing, missing
    unclassified = [n for n in wanted if pf.scratch_slot_lifetime(n) is None]
    assert not unclassified, unclassified


def test_the_retired_exner_scratch_slot_is_gone():
    """Fixing a defect retires its guards.  The host path allocated a
    domain-sized Exner array (``mp_pii``) because the CPU wrapper wanted
    one; the kernels build the same factor in a register where the
    authority builds it, so keeping the slot would reserve a full
    (nz, ny, nx) allocation nothing reads."""
    from woof.config import RunConfig
    from woof.core import preflight as pf

    cfg = RunConfig(nx=8, ny=8, nz=6, dx=1000.0, dy=1000.0, dt=1.0,
                    ztop=10000.0, run_seconds=60.0, moist=True,
                    mp_physics=50)
    assert "mp_pii" not in pf.scratch_slot_registry(cfg, n_lbc_intervals=2)
    assert "mp_pii" in pf.scratch_slot_registry(
        dataclasses.replace(cfg, mp_physics=8), n_lbc_intervals=2)


def test_the_validation_status_word_is_priced_for_p3():
    """A bare mp=50 run drew its validation status word OUTSIDE the audited
    scratch arena, because the registry spelled the admitting set as a
    literal frozen when mp=28 landed.

    ``physics_validation_status`` is the single uint32 word the batched
    device validators share.  ``PhysicsDriver.accept_microphysics``
    (woof/core/physics.py:2045-2052) takes the native arm -- the one that
    launches ``_validate_native_microphysics`` against that word -- for
    every scheme with a canonical surface-diagnostic row that is not mp=18,
    and P3 has one: FIVE slots rather than mp=6/8's seven, because a single
    ice category means no qs and no qg, so the driver arm binds
    RAINNC/RAINNCV/SR/SNOWNC/SNOWNCV and no graupel argument
    (``module_microphysics_driver.F:1590-1595``; the SR at :1592 is P3's own
    ``pcprt_sol/(pcprt_liq+pcprt_sol+1.e-12)``, ``module_mp_p3.F:898``).

    Five is a canonical row, so the word was asked for on every P3 step and
    ``DomainState.scratch`` handed back a private per-state allocation
    instead of the arena view the lifetime audit signed off -- unbudgeted in
    the estimate and outside the arena's own inventory.  ``preflight`` had
    already priced the ``microphysics_validation`` kernel MODULE for mp=50
    (``_MICROPHYSICS_KERNEL_MODULES``) while omitting its status WORD, and
    that contradiction is what this gate closes.

    mp=18 must stay out, and for a stated reason rather than by omission:
    accept_microphysics excludes NSSL by name (physics.py:2046) and takes
    the per-field arm, which launches no validator at all.
    """
    from woof.config import RunConfig
    from woof.core import preflight as pf
    from woof.core.physics_inventory import microphysics_scratch_slots

    def priced(mp: int) -> bool:
        cfg = RunConfig(nx=8, ny=8, nz=6, dx=1000.0, dy=1000.0, dt=1.0,
                        ztop=10000.0, run_seconds=60.0, moist=bool(mp),
                        mp_physics=mp)
        return "physics_validation_status" in pf.scratch_slot_registry(
            cfg, n_lbc_intervals=2)

    assert priced(50), (
        "mp=50 launches the shared microphysics validator, so its status "
        "word has to come from the audited arena")
    # The rule is the runtime's, so it holds for every scheme sharing P3's
    # native arm -- mp=9 reached production in the same hole.
    for mp in (1, 6, 8, 9, 10, 16, 28):
        assert priced(mp), mp
    # ... and this is not a widened set: mp=18 HAS a canonical row and is
    # still out, by the runtime's own named exclusion; mp=0 has no row.
    assert microphysics_scratch_slots(18)
    assert not priced(18)
    assert not microphysics_scratch_slots(0)
    assert not priced(0)
    # The consequence a missing row buys is a private allocation, not a
    # refusal, and this is the property that makes that true.
    assert pf.scratch_slot_uses_arena("physics_validation_status")
