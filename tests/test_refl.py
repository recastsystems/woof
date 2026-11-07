"""Radar reflectivity diagnostics (WRF ``do_radar_ref=1``, REFL_10CM).

The authority is the local WRF v4.6.1 ``refl10cm_hm``
(phys/module_mp_morr_two_moment.F:4502-4675, wrapper floor :913-917) with
its ``radar_init`` module state (phys/module_mp_radar.F); the Kessler
fallback is the Smith et al. (1975) fixed-intercept rain form derived in
woof/core/refl.py.  Hand fixtures below are computed from the Fortran
directly -- closed forms evaluated in-test where tractable, and frozen
constants from an independent offline transcription pass for the 50-bin
melting integrals -- never via the npref mirror under test.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu

# Consumer expectations come from the independent generated declarations.
from woof.physics_registry import implemented_selector_values as _intake_selectors
from woof.physics_registry import consumer_rows_by_selector as _intake_rows
_INTAKE_MP = tuple(sorted(_intake_selectors("microphysics")))
_INTAKE_REFL = tuple(mp for mp in _INTAKE_MP if mp)
_INTAKE_PRODUCERS = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "reflectivity_input_species").items()
    if row is not None))
_INTAKE_NATIVE = tuple(sorted(set(_INTAKE_REFL) - set(_INTAKE_PRODUCERS)))
_INTAKE_SNOW = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "ring_guard").items()
    if "qs" in (row or {}).get("state_fields", ())))
_INTAKE_SIX = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "ring_guard").items()
    if "qg" in (row or {}).get("state_fields", ())))

#: Morrison module PI (module_mp_morr_two_moment.F:99).
PI = 3.1415926535897932384626434


def _hand_rho(qv, t, p):
    """refl10cm_hm's own air density (F:4540-4542): the qv floor and the
    literal 0.622/R = 287 pair."""
    return 0.622 * p / (287.0 * t * (max(qv, 1.0e-10) + 0.622))


def _hand_rain_ze(qr, nr, rho):
    """Closed-form rain Ze (m6 m-3) for the two-moment exponential PSD.

    From F:4547-4551/:4609: lamr = (xam_r*Gamma(4)*nr_v/rr)**(1/3) with
    xam_r = PI*997/6, so lamr**3 = PI*997*nr_v/rr; N0_r = nr_v*lamr; and
    ze_rain = N0_r*Gamma(7)/lamr**7 = 720*nr_v/lamr**6.
    """
    rr = qr * rho
    nrv = nr * rho
    lamr = (PI * 997.0 * nrv / rr) ** (1.0 / 3.0)
    return 720.0 * nrv / lamr ** 6


def _hand_frozen_ze(q, n, rho, am):
    """Closed-form dry snow/graupel Ze (m6 m-3), F:4610-4615: the rain
    form times (0.176/0.93)*(6/PI)^2*(am/900)^2 with the species'
    mass prefactor am (snow 100*PI/6, graupel 400*PI/6)."""
    rq = q * rho
    nv = n * rho
    lam = (am * 6.0 * nv / rq) ** (1.0 / 3.0)
    return ((0.176 / 0.93) * (6.0 / PI) ** 2 * (am / 900.0) ** 2
            * 720.0 * nv / lam ** 6)


def test_kessler_cell_launch_uses_full_elementwise_block():
    import woof.core.refl as refl

    source = (Path(__file__).resolve().parents[1] / "woof" / "core" /
              "kernels" / "refl.cu").read_text(encoding="utf-8")
    body = source[source.index(
        'extern "C" __global__ void refl10cm_kessler_cell'):]
    assert "__shared__" not in body
    assert "__syncthreads" not in body
    assert "atomic" not in body
    assert "blockIdx.x * blockDim.x + threadIdx.x" in body
    assert refl._CELL_TPB == 256


def test_radar_init_pins_wrf_constants():
    """radar_init products match the Fortran's construction (offline
    hand evaluation of module_mp_radar.F:72-186)."""
    from woof.core.refl import radar_init

    # This legacy offline fixture covers WRF's explicit graupel branch;
    # test_morr_rimed_ice.py separately pins the Registry-default hail path.
    rc = radar_init(0)
    # Ray (1972) water and Maetzler (1998) ice refractive indices at
    # 0 degC / 10 cm, and the |K_w|^2 dielectric factor (:78-80).
    np.testing.assert_allclose(
        [rc.m_w_0.real, rc.m_w_0.imag],
        [9.038708836358222, -1.3879002993934035], rtol=1.0e-9)
    np.testing.assert_allclose(
        [rc.m_i_0.real, rc.m_i_0.imag],
        [1.7856091448532874, -1.3484690495450238e-4], rtol=1.0e-9)
    np.testing.assert_allclose(rc.k_w, 0.9341686541102098, rtol=1.0e-9)
    # PI5 uses WRF's TRUNCATED pi (:76), not the module PI.
    np.testing.assert_allclose(rc.pi5, 3.14159 ** 5, rtol=0.0)
    assert rc.lamda4 == 0.10 ** 4
    # Log-spaced bins, 100 um to 2 cm (snow) / 5 cm (graupel) (:120-143):
    # first center sqrt(edge0*edge1), widths telescope to dmax - 100 um.
    np.testing.assert_allclose(rc.xxds[0], 1.0544119030826875e-4,
                               rtol=1.0e-12)
    np.testing.assert_allclose(rc.xxds[-1], 1.8967919407517922e-2,
                               rtol=1.0e-12)
    np.testing.assert_allclose(rc.xdts.sum(), 0.02 - 1.0e-4, rtol=1.0e-12)
    np.testing.assert_allclose(rc.xdtg.sum(), 0.05 - 1.0e-4, rtol=1.0e-12)
    # Composite-Simpson weights (:82-90): 1/3, 4/3, 2/3, ... on the
    # shared interior points.
    np.testing.assert_allclose(rc.simpson[:4],
                               [1 / 3, 4 / 3, 2 / 3, 4 / 3], rtol=1.0e-15)
    # Gamma moments for the exponential PSDs (bm = 3, mu = 0).
    assert rc.xcre == (4.0, 1.0, 4.0, 7.0)
    assert rc.xcrg[2] == 6.0 and rc.xcrg[3] == 720.0
    assert rc.xorg2 == 1.0
    # Scheme m(D) prefactors: RHOW = 997 rain, CS snow, and the explicit
    # IHAIL = 0 graupel branch (RHOG = 400).
    np.testing.assert_allclose(
        [rc.xam_r, rc.xam_s, rc.xam_g],
        [PI * 997.0 / 6.0, 100.0 * PI / 6.0, 400.0 * PI / 6.0], rtol=0.0)


def test_refl_morrison_mirror_rain_only_matches_hand():
    """Warm rain-only point vs the in-test closed form and the frozen
    offline evaluation of the full Fortran path."""
    from woof.verify.npref import np_refl10cm_morrison_column

    qv, qr, nr, t, p = 5.0e-3, 1.0e-3, 2.0e5, 285.0, 85000.0
    out = np_refl10cm_morrison_column([qv], [qr], [nr], [0.0], [0.0],
                                      [0.0], [0.0], [t], [p])
    rho = _hand_rho(qv, t, p)                    # = 1.0308963758055036
    ze = _hand_rain_ze(qr, nr, rho) + 2.0e-22    # empty snow/graupel floors
    np.testing.assert_allclose(out[0], 10.0 * math.log10(ze * 1.0e18),
                               atol=1.0e-5, rtol=0.0)
    np.testing.assert_allclose(out[0], 25.77827681195457,
                               atol=1.0e-6, rtol=0.0)


def test_refl_morrison_mirror_dry_snow_graupel_matches_hand():
    """Sub-freezing snow+graupel point: dry Rayleigh forms only."""
    from woof.verify.npref import np_refl10cm_morrison_column

    qv, qs, ns, qg, ng = 1.0e-3, 2.0e-3, 8.0e4, 1.0e-3, 4.0e3
    t, p = 258.0, 50000.0
    out = np_refl10cm_morrison_column([qv], [0.0], [0.0], [qs], [ns],
                                      [qg], [ng], [t], [p])
    rho = _hand_rho(qv, t, p)                    # = 0.6741720441100146
    ze = (_hand_frozen_ze(qs, ns, rho, 100.0 * PI / 6.0)
          + _hand_frozen_ze(qg, ng, rho, 900.0 * PI / 6.0) + 1.0e-22)
    np.testing.assert_allclose(out[0], 10.0 * math.log10(ze * 1.0e18),
                               atol=1.0e-5, rtol=0.0)
    np.testing.assert_allclose(out[0], 35.374694584130594,
                               atol=1.0e-6, rtol=0.0)


#: Three-level melting column: warm rain+snow+graupel at k = 0-1 under a
#: cold snow/graupel level, so k_0 = 2 and both soak paths run at two
#: melt fractions.  (qv, qr, nr, qs, ns, qg, ng, t, p) bottom-to-top.
_MELTING_COLUMN = (
    np.array([6.0e-3, 5.0e-3, 3.0e-3]),
    np.array([1.5e-3, 1.0e-3, 0.0]),
    np.array([3.0e5, 2.0e5, 0.0]),
    np.array([1.0e-3, 1.2e-3, 2.0e-3]),
    np.array([5.0e4, 6.0e4, 1.0e5]),
    np.array([5.0e-4, 6.0e-4, 8.0e-4]),
    np.array([3.0e3, 3.5e3, 5.0e3]),
    np.array([279.0, 275.5, 268.0]),
    np.array([92000.0, 88000.0, 70000.0]),
)


def test_refl_morrison_mirror_melting_column_matches_hand():
    """Blahak water-coated melting levels against frozen hand values.

    The 50-bin Maxwell-Garnett integrals are not tractable in closed
    form; the expected dBZ triple is frozen from an independent offline
    transcription pass of module_mp_radar.F (NOT the mirror), and the
    bright-band physics is asserted against an all-cold rerun of the
    same water loadings.
    """
    from woof.verify.npref import np_refl10cm_morrison_column

    args = tuple(c.copy() for c in _MELTING_COLUMN)
    # Frozen values were produced for explicit IHAIL=0 graupel.
    out = np_refl10cm_morrison_column(*args, morr_rimed_ice=0)
    np.testing.assert_allclose(
        out, [43.892513802446, 42.805937813983, 34.138784664944],
        atol=1.0e-6, rtol=0.0)

    # Same hydrometeors, all-cold column: no melting level, dry forms
    # everywhere.  The melting treatment must ENHANCE the warm levels
    # (bright band) and leave the k_0 level itself untouched.
    cold = list(c.copy() for c in _MELTING_COLUMN)
    cold[7] = np.array([268.0, 268.0, 268.0])
    dry = np_refl10cm_morrison_column(*cold)
    assert out[0] > dry[0] + 5.0 and out[1] > dry[1] + 5.0
    np.testing.assert_allclose(out[2], dry[2], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "theta0,qv0,theta1,qv1",
    [
        (300.0, 0.012, 302.5, 0.011),
        (300.0, 0.012, 305.0, 0.010),
        (300.0, 0.012, 297.5, 0.013),
    ],
)
def test_prepared_pressure_seam_has_predicted_dry_dbz_offset(
        theta0, qv0, theta1, qv1):
    """The retired output-time EOS seam has the shadow's exact dry offset.

    At unchanged specific volume woof's post-microphysics EOS pressure
    changes reflectivity density by the virtual-theta ratio below.  Rain's
    dry Rayleigh Ze is linear in density, so prepared-p and re-diagnosed-p
    inputs differ by exactly 10*log10(ratio), before the output floor.
    """
    from woof.core import constants as c
    from woof.verify.npref import np_refl10cm_morrison_column

    rho_ratio = (theta1 * (1.0 + c.RVOVRD * qv1)
                 / (theta0 * (1.0 + c.RVOVRD * qv0)))
    p_prepared = 85000.0
    # rho_refl = p/(R*T_v), T = theta1*(p/P0)^kappa, so this
    # pressure change realizes the EOS density ratio above exactly.
    p_rediagnosed = p_prepared * rho_ratio ** (1.0 / (1.0 - c.RCP))
    t_prepared = theta1 * (p_prepared / c.P0) ** c.RCP
    t_rediagnosed = theta1 * (p_rediagnosed / c.P0) ** c.RCP
    zeros = np.zeros(1)

    def diagnose(t, p):
        return np_refl10cm_morrison_column(
            [qv1], [1.0e-3], [2.0e5], zeros, zeros, zeros, zeros,
            [t], [p])[0]

    seam = diagnose(t_rediagnosed, p_rediagnosed) - diagnose(
        t_prepared, p_prepared)
    np.testing.assert_allclose(
        seam, 10.0 * math.log10(rho_ratio), rtol=0.0, atol=5.0e-8)


def test_rediagnosed_pressure_can_flip_strict_melting_scan():
    """A 0.002 K EOS shift flips WRF's strict T > 273.15 bright band."""
    from woof.core import constants as c
    from woof.verify.npref import np_refl10cm_morrison_column

    hydrometeors = [column.copy() for column in _MELTING_COLUMN[:7]]
    p_prepared = _MELTING_COLUMN[8].copy()
    t_prepared = np.array([270.0, 273.149, 268.0])
    theta_post_scheme = t_prepared / (p_prepared / c.P0) ** c.RCP
    t_rediagnosed = t_prepared.copy()
    t_rediagnosed[1] = 273.151
    p_rediagnosed = c.P0 * (
        t_rediagnosed / theta_post_scheme) ** (1.0 / c.RCP)

    dry = np_refl10cm_morrison_column(
        *hydrometeors, t_prepared, p_prepared)
    bright = np_refl10cm_morrison_column(
        *hydrometeors, t_rediagnosed, p_rediagnosed)

    assert t_prepared[1] <= 273.15 < t_rediagnosed[1]
    assert bright[0] > dry[0] + 5.0
    assert bright[1] > dry[1] + 5.0
    # The cold k_0 reference level itself is not replaced.
    np.testing.assert_allclose(bright[2], dry[2], rtol=0.0, atol=0.0)


def test_invalid_active_number_moment_surfaces_nonfinite_output():
    """F2: active mass/zero number is outside WRF's Morrison contract.

    refl10cm_hm guards only q (:4544-4584); Morrison normally repairs its
    number moments through nonnegative clipping and bounded slopes
    (:1528-1635).  The mirror/kernel leave corrupt active pairs non-finite
    rather than turning them into the -35 dBZ clear-air floor.
    """
    from woof.verify.npref import np_refl10cm_morrison_column

    zeros = np.zeros(1)
    with np.errstate(all="ignore"):
        out = np_refl10cm_morrison_column(
            [0.005], [1.0e-3], [0.0], zeros, zeros, zeros, zeros,
            [280.0], [85000.0])
    assert np.isnan(out[0])


def test_refl_stash_is_consumed_once_and_missing_is_loud():
    """The output handoff cannot be overwritten or consumed twice."""
    from woof.core.refl import consume_refl_10cm, stash_refl_10cm

    state = SimpleNamespace(physics=SimpleNamespace(refl_10cm=None))
    field = np.zeros((2, 3, 4), dtype=np.float32)
    stash_refl_10cm(state, field)
    with pytest.raises(RuntimeError, match="not consumed"):
        stash_refl_10cm(state, field.copy())
    assert consume_refl_10cm(state) is field
    with pytest.raises(RuntimeError, match="no microphysics-time field"):
        consume_refl_10cm(state)


def test_refl_stash_has_no_trajectory_or_restart_reader():
    """The handoff feeds history/lightning diagnostics, never prognostics."""
    repo = Path(__file__).resolve().parents[1]
    woof = repo / "woof"

    # Grep-style pin for the sole reader API: definition plus production and
    # verification output assembly are the only repository hits.  The ideal
    # nest cases consume the one-frame handoff at their history callback but
    # deliberately discard the field because they emit metric JSON, not
    # wrfout.  The N5S shadow case is the other verification wrfout producer.
    # The offline downscale child (woof downscale) consumes at its own
    # emit_output history seam -- added by the STEP14 downscale lane, which
    # missed this pin; recorded and admitted by the STEP17 lane.
    #
    # The prepared single-domain forecast runner joined this set in
    # v1.3.1 without changing one line of its own: it has always consumed
    # the handoff at its wrfout assembly, and it used to live at
    # tools/prepared_single_domain_forecast.py, outside the gpuwm/ tree
    # this grep walks.  Moving its substance into the package (so a pip
    # install can run a forecast) made an existing reader visible to the
    # pin rather than adding a new one.  It is the third wrfout producer
    # and belongs here on exactly the same grounds as woof/runtime.py.
    #
    # The N5S shadow case LEFT this set the same way it entered: without
    # changing what it does.  Its history callback had hand-rolled a copy
    # of runtime._submit_tree_history_frame -- the same lines, including
    # this consume, minus the reset of the nwp_diagnostics running maxima
    # that the shared helper also performs.  It calls the helper now, so
    # the consume happens one frame deeper and this file no longer names
    # it, exactly as woof/prepared_domain_tree_forecast.py (the other
    # caller of that helper) has never named it.
    consume_hits = {
        path.relative_to(repo).as_posix()
        for path in woof.rglob("*.py")
        if "consume_refl_10cm" in path.read_text(encoding="utf-8")
    }
    assert consume_hits == {
        "woof/core/refl.py", "woof/runtime.py",
        "woof/offline_child_run.py",
        "woof/prepared_single_domain_forecast.py",
        # Ensemble output consumes the same handoff, once per due frame.
        "woof/ensemble/native_forecast.py",
        "woof/ensemble/batch_forecast.py",
        "woof/verify/cases/nest_ideal_common.py",
        # The STREAMED path, and it consumes for a different reason from
        # every other entry here: it is not publishing a frame, it is
        # CLEARING a per-tile handoff.  ``stash_refl_10cm`` refuses to
        # overwrite an unconsumed stash, which in a resident run is the
        # right refusal (a second unconsumed field IS a cadence bug) and in
        # a sweep is wrong -- the handoff is per TILE and the frame is per
        # DOMAIN, so the second tile a buffer serves would raise "REFL_10CM
        # stash was not consumed before reuse", naming a defect that is not
        # present.  ``streaming.refl_handoff_hook`` consumes each tile's
        # reference after its step; the VALUES are in the scratch slot,
        # which the scatter carries into the domain field like any other
        # carrier.
        "woof/core/streaming.py",
        # The VTSM native retention (4dcca7d16, woof/da/spread_vts.py)
        # also consumes only to CLEAR: its history drain has no output
        # writer, and an unconsumed stash would refuse the next due frame.
        # The field is discarded, never read back into the model.
        "woof/da/spread_vts.py",
        # The hex column-batch seam (6e333822e, phase two of the mpas
        # seam: WRF's history-step diagflag).  On a due step it consumes
        # the handoff inside the same transaction that builds the step
        # RECEIPT the hex caller assembles its history from -- output
        # assembly on exactly the grounds woof/runtime.py is here, and
        # an unconsumed stash would be the same cadence refusal one call
        # later.
        "woof/core/mpas_column_batch.py",
        # The configurable column suite builds the same history receipt.
        "woof/core/mpas_column_suite.py",
        # Lightning consumes only to clear a diagnostic-only handoff when
        # history is not due; its flash diagnostics read the native field.
        "woof/core/lightning.py",
        # NWP diagnostics clear a held handoff after a non-history step;
        # it is not fed back into the model state.
        "woof/core/dycore.py",
    }

    # Prevent a direct attribute read or constant-name getattr from bypassing
    # that API anywhere in the package.  The defining module is the one
    # legitimate exception; assignment elsewhere is a Store context.
    reader_paths = [
        path for path in woof.rglob("*.py")
        if path not in (woof / "core" / "refl.py", woof / "simulated_radar.py")
    ]
    # The radar history snapshot reads the held diagnostic without advancing physics.
    radar = ast.parse((woof / "simulated_radar.py").read_text())
    direct = [node for node in ast.walk(radar) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name) and node.func.id == "getattr"
              and len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
              and node.args[1].value == "refl_10cm"]
    snapshot = next(n for n in radar.body if isinstance(n, ast.FunctionDef) and n.name == "_snapshot_history")
    assert len(direct) == 1 and direct[0] in list(ast.walk(snapshot))
    for path in reader_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        readers = [
            node for node in ast.walk(tree)
            if ((isinstance(node, ast.Attribute)
                 and node.attr == "refl_10cm"
                 and isinstance(node.ctx, ast.Load))
                or (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "refl_10cm"))
        ]
        if path == woof / "core" / "lightning.py":
            assert len(readers) == 1
            assert ast.unparse(readers[0]) == "state.physics.refl_10cm"
            continue
        if path == woof / "core" / "dycore.py":
            # Only the two step epilogues may inspect the held reference.
            # Their branch discards it through the public consume API.
            assert len(readers) == 2
            for reader in readers:
                guards = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                          and reader in list(ast.walk(n.test))]
                assert len(guards) == 1
                guard = guards[0]
                assert ast.unparse(guard.test) == (
                    "not refl_10cm_due and getattr(state.physics, 'refl_10cm', None) is not None")
                assert ast.unparse(guard.body[-1]) == "consume_refl_10cm(state)"
                assert len(guard.body) == 2 and not guard.orelse
            continue
        assert not readers, f"package code reads REFL_10CM: {path}"


def test_refl_morrison_mirror_soak_requires_species_at_k0():
    """The soak gates on the species being present AT the k_0 reference
    level (F:4630/:4649 ``L_qs(k) .and. L_qs(k_0)``), not merely below."""
    from woof.verify.npref import np_refl10cm_morrison_column

    # k=0: warm rain+snow; k=1: warm rain; k=2: cold graupel only.
    # k_0 = 2 with L_qs(k_0) false: snow at k=0 stays DRY.
    qv = np.array([6.0e-3, 5.0e-3, 3.0e-3])
    qr = np.array([1.5e-3, 1.0e-3, 0.0])
    nr = np.array([3.0e5, 2.0e5, 0.0])
    qs = np.array([1.0e-3, 0.0, 0.0])
    ns = np.array([5.0e4, 0.0, 0.0])
    qg = np.array([0.0, 0.0, 8.0e-4])
    ng = np.array([0.0, 0.0, 5.0e3])
    t = np.array([279.0, 275.5, 268.0])
    p = np.array([92000.0, 88000.0, 70000.0])
    gated = np_refl10cm_morrison_column(qv, qr, nr, qs, ns, qg, ng, t, p)
    # With the soak gated off, k=0 must equal the DRY closed forms
    # (rain + dry snow) evaluated by hand at the warm-level density.
    rho0 = _hand_rho(qv[0], t[0], p[0])
    ze0 = (_hand_rain_ze(qr[0], nr[0], rho0)
           + _hand_frozen_ze(qs[0], ns[0], rho0, 100.0 * PI / 6.0)
           + 1.0e-22)
    np.testing.assert_allclose(gated[0], 10.0 * math.log10(ze0 * 1.0e18),
                               atol=1.0e-5, rtol=0.0)

    # Adding snow at the k_0 level opens the gate and enhances k=0.
    qs2 = qs.copy()
    ns2 = ns.copy()
    qs2[2] = 2.0e-3
    ns2[2] = 1.0e5
    soaked = np_refl10cm_morrison_column(qv, qr, nr, qs2, ns2, qg, ng, t, p)
    assert soaked[0] > gated[0] + 1.0


def test_refl_mirror_empty_column_floors_at_minus_35():
    """Zero hydrometeors: all three ze floors are 1e-22 (F:4602-4604),
    10*log10(3e-22*1e18) = -35.23 dBZ, and the wrapper clamp (F:916)
    emits exactly -35.0.  The activation threshold is strictly > 1e-9
    (F:4544), so a column AT the threshold also floors."""
    from woof.verify.npref import (np_refl10cm_kessler_column,
                                    np_refl10cm_morrison_column)

    zeros = np.zeros(4)
    t = np.full(4, 280.0)
    p = np.full(4, 90000.0)
    out = np_refl10cm_morrison_column(np.full(4, 2.0e-3), zeros, zeros,
                                      zeros, zeros, zeros, zeros, t, p)
    np.testing.assert_array_equal(out, np.full(4, -35.0))
    at_threshold = np.full(4, 1.0e-9)
    out = np_refl10cm_morrison_column(np.full(4, 2.0e-3), at_threshold,
                                      np.full(4, 1.0e5), zeros, zeros,
                                      zeros, zeros, t, p)
    np.testing.assert_array_equal(out, np.full(4, -35.0))
    out = np_refl10cm_kessler_column(np.full(4, 2.0e-3), zeros, t, p)
    np.testing.assert_array_equal(out, np.full(4, -35.0))


def test_refl_kessler_mirror_matches_hand():
    """Smith-1975 fixed-intercept rain form vs the in-test closed form
    and its frozen offline evaluation; monotone in qr."""
    from woof.verify.npref import np_refl10cm_kessler_column

    qv, qr, t, p = 5.0e-3, 2.0e-3, 285.0, 85000.0
    out = np_refl10cm_kessler_column([qv], [qr], [t], [p])
    rho = _hand_rho(qv, t, p)
    lam = (PI * 1000.0 * 8.0e6 / (rho * qr)) ** 0.25
    ze = 720.0 * 8.0e6 / lam ** 7
    np.testing.assert_allclose(out[0], 10.0 * math.log10(ze * 1.0e18),
                               atol=1.0e-6, rtol=0.0)
    np.testing.assert_allclose(out[0], 48.59931493840356,
                               atol=1.0e-6, rtol=0.0)

    ladder = np_refl10cm_kessler_column(
        np.full(3, qv), np.array([5.0e-4, 2.0e-3, 8.0e-3]),
        np.full(3, t), np.full(3, p))
    assert ladder[0] < ladder[1] < ladder[2]


def _stress_columns(ncol=8, nz=30, seed=7):
    """Deterministic mixed-regime FP32 column batch: warm rain low,
    snow/graupel aloft, melting layers where the profiles overlap the
    freezing level, and some empty columns."""
    rng = np.random.default_rng(seed)
    z = np.linspace(100.0, 12000.0, nz)
    cols = []
    for n in range(ncol):
        t = 293.0 - 0.0062 * z + rng.normal(0.0, 1.0, nz)
        p = 96000.0 * np.exp(-z / 8200.0)
        qv = 0.012 * np.exp(-z / 3000.0)

        def bump(scale, zc, width):
            prof = (scale * np.exp(-((z - zc) / width) ** 2)
                    * rng.uniform(0.2, 1.5))
            prof[prof < 2.0e-9] = 0.0
            return prof

        empty = n == ncol - 1
        qr = np.zeros(nz) if empty else bump(3.0e-3, 2000.0, 2500.0)
        qs = np.zeros(nz) if empty else bump(2.5e-3, 5500.0, 3000.0)
        qg = np.zeros(nz) if empty else bump(1.5e-3, 4500.0, 2800.0)
        nr = np.where(qr > 0.0, rng.uniform(1.0e4, 1.0e6, nz), 0.0)
        ns = np.where(qs > 0.0, rng.uniform(1.0e4, 5.0e5, nz), 0.0)
        ng = np.where(qg > 0.0, rng.uniform(1.0e3, 5.0e4, nz), 0.0)
        cols.append({"qv": qv, "qr": qr, "nr": nr, "qs": qs, "ns": ns,
                     "qg": qg, "ng": ng, "t": t, "p": p})
    batch = {}
    for name in cols[0]:
        stack = np.stack([c[name] for c in cols], axis=1)    # (nz, ncol)
        batch[name] = np.ascontiguousarray(
            stack.reshape(nz, 2, ncol // 2), dtype=np.float32)
    return batch


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("morr_rimed_ice", (0, 1))
def test_refl_kernel_matches_mirror_morrison(morr_rimed_ice):
    """One CUDA thread per column matches the float64 mirror.

    Tolerance: the FP32 species-density/ze storage bounds the kernel at
    ~50 output ULPs; the kernel source host-compiled against the mirror
    measured <= 2.3e-6 dBZ over this exact batch, so 1e-3 dBZ carries
    ~400x headroom while any formula slip is O(1) dB.
    """
    import cupy as cp

    from woof.core.refl import launch_refl10cm_morrison
    from woof.verify.npref import np_refl10cm_morrison_column

    batch = _stress_columns()
    nz, ny, nx = batch["t"].shape
    dev = {name: cp.asarray(value) for name, value in batch.items()}
    refl = cp.full((nz, ny, nx), cp.nan, dtype=cp.float32)
    launch_refl10cm_morrison(dev["qv"], dev["qr"], dev["nr"], dev["qs"],
                             dev["ns"], dev["qg"], dev["ng"], dev["t"],
                             dev["p"], refl,
                             morr_rimed_ice=morr_rimed_ice)
    got = cp.asnumpy(refl)
    assert np.isfinite(got).all()
    expected = np.empty((nz, ny, nx))
    for j in range(ny):
        for i in range(nx):
            expected[:, j, i] = np_refl10cm_morrison_column(
                *(batch[name][:, j, i].astype(np.float64)
                  for name in ("qv", "qr", "nr", "qs", "ns",
                               "qg", "ng", "t", "p")),
                morr_rimed_ice=morr_rimed_ice)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1.0e-3)
    # The batch must actually exercise the paths it claims to cover.
    assert expected.max() > 30.0 and (expected == -35.0).any()


@pytest.mark.gpu
@requires_gpu
def test_refl_kernel_matches_mirror_kessler():
    """Per-cell Kessler fallback kernel matches its float64 mirror."""
    import cupy as cp

    from woof.core.refl import launch_refl10cm_kessler
    from woof.verify.npref import np_refl10cm_kessler_column

    batch = _stress_columns(seed=11)
    nz, ny, nx = batch["t"].shape
    dev = {name: cp.asarray(batch[name]) for name in ("qv", "qr", "t", "p")}
    refl = cp.full((nz, ny, nx), cp.nan, dtype=cp.float32)
    launch_refl10cm_kessler(dev["qv"], dev["qr"], dev["t"], dev["p"], refl)
    got = cp.asnumpy(refl)
    assert np.isfinite(got).all()
    expected = np.empty((nz, ny, nx))
    for j in range(ny):
        for i in range(nx):
            expected[:, j, i] = np_refl10cm_kessler_column(
                *(batch[name][:, j, i].astype(np.float64)
                  for name in ("qv", "qr", "t", "p")))
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1.0e-3)
    assert expected.max() > 20.0 and (expected == -35.0).any()


def _small_moist_state(mp_physics):
    from woof.config import RunConfig
    from woof.core.diagnostics import update_diagnostics
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced

    cfg = RunConfig(nx=3, ny=2, nz=24, dx=1000.0, dy=1000.0,
                    ztop=12000.0, dt=30.0, run_seconds=0.0,
                    moist=True, mp_physics=mp_physics)
    vc = make_vertical_coord(cfg.nz)
    base = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(
        cfg, vc, base, lambda z: 0.012 * np.exp(-np.asarray(z) / 3000.0))
    state.qc[...] = 1.0e-3
    update_diagnostics(state)
    return cfg, state


@pytest.mark.gpu
@requires_gpu
def test_compute_refl_10cm_dispatches_on_active_scheme():
    """The output-time seam consumes the post-microphysics state fields
    and dispatches per scheme like microphysics.apply."""
    import cupy as cp

    from woof.core import microphysics
    from woof.core.refl import compute_refl_10cm

    for mp in (1, 10):
        cfg, state = _small_moist_state(mp)
        microphysics.apply(state, cfg, cfg.dt)
        refl = compute_refl_10cm(state, cfg)
        assert refl.shape == state.p.shape and refl.dtype == cp.float32
        got = cp.asnumpy(refl)
        assert np.isfinite(got).all()
        assert got.min() >= -35.0
        assert refl is state.scratch(state.p.shape, "refl_10cm")

    cfg, state = _small_moist_state(0)
    with pytest.raises(ValueError, match="mp_physics"):
        compute_refl_10cm(state, cfg)


def test_p3_refuses_by_name_rather_than_being_absent_from_the_gate():
    """mp=50 is CORRECTLY out of the producer gate, and now says so.

    P3 has no standalone reflectivity routine to dispatch to: WRF computes
    dBZ inside ``p3_main`` (ze_rain from the scheme's own rain PSD,
    module_mp_p3.F:4755; ze_ice from the lookup-table normalised moment
    f1pr13, :4789/:4868; diag_ze at :4887) and the driver binds
    ``diag_zdbz_3d`` straight to ``refl_10cm``.  It also predicts ONE ice
    category with a rime pair and allocates no qs and no qg, so the only
    branch here it could reach is Kessler's rain-only form.

    The omission was therefore right and the WORDING was wrong: the refusal
    told a P3 caller that mp=50 is not "an active microphysics scheme",
    which is the mp=28 failure mode -- a reachability gap dressed as a
    not-implemented message.  The radar observation operator
    (woof/da/obsop.py) sends every scheme except 18 through this function,
    so that sentence was the whole answer a radar-DA caller got.
    """
    from woof.core.refl import (NativeReflectivityScheme,
                                 SCHEME_NATIVE_REFL_10CM, compute_refl_10cm)

    class _S:
        qv = np.zeros((4, 3, 2), dtype=np.float32)
        p = np.zeros((4, 3, 2), dtype=np.float32)

        @staticmethod
        def scratch(shape, _name):
            return np.zeros(shape, dtype=np.float32)

    with pytest.raises(NativeReflectivityScheme) as caught:
        compute_refl_10cm(_S(), SimpleNamespace(mp_physics=50))
    message = str(caught.value)

    # Still a ValueError, so no caller loses its handler.
    assert isinstance(caught.value, ValueError)
    # It names the scheme, why it is out, and where its dBZ actually is.
    assert "mp_physics=50" in message
    assert "p3_main" in message and "module_mp_p3.F:4755" in message
    assert "no qs" in message.lower() and "no qg" in message.lower()
    assert "refl_10cm" in message
    # The clear-air floor differs from this dispatcher's family, and a DA
    # caller differencing H(x) against a clear sky has to know it.
    assert "-99 dBZ" in message and "-35 dBZ" in message
    # The sentence the mp=28 audit taught us not to write.
    assert "active microphysics scheme" not in message

    # And the gate itself is unchanged: naming the refusal is not a licence
    # to widen it.  50 would fall through to the Kessler arm.
    assert 50 in SCHEME_NATIVE_REFL_10CM
    # The gate reads the PUBLISHED input-species table now (one row per
    # dispatched scheme, the same row the physics registry carries), so the
    # pin is on the table's keys rather than on a literal tuple in the
    # source, and the source is asserted to read the table.
    from woof.core.refl import REFL_10CM_INPUT_SPECIES
    assert set(REFL_10CM_INPUT_SPECIES) == {1, 2, 3, 4, 6, 8, 10, 13, 14, 16, 26, 28, 38, 40, 106}
    import woof.core.refl as refl_module
    source = Path(refl_module.__file__).read_text(encoding="utf-8")
    assert "cfg.mp_physics not in REFL_10CM_INPUT_SPECIES" in source


def test_native_reflectivity_table_is_the_consumer_minus_producer_set():
    """Every scheme that stages REFL_10CM without dispatching here has a
    RECORDED reason, and the table cannot drift from the consumer census.

    ``tests/test_mp28_runtime_reachability.py`` asserts the arithmetic
    ``REFL_ADMISSION - {1, 6, 8, 10, 16, 28, 38} == {9, 18, 50}``.  That is the
    same statement, bound to the constant a reader of woof/core/refl.py
    actually reaches, so a scheme joining the consumer set without a
    recorded reason fails here instead of reaching a user as an omission.
    """
    from woof import runtime
    from woof.core.refl import SCHEME_NATIVE_REFL_10CM

    producer = frozenset({1, 2, 3, 4, 6, 8, 10, 13, 14, 16, 26, 28, 38, 40, 106})
    consumer = frozenset(runtime.REFL_10CM_MICROPHYSICS)
    # FAST_SBM (30) is recorded here without being a consumer: it writes no
    # REFL_10CM at all (woof.core.fast_sbm_runtime.apply refuses a due
    # frame, so the history lane never asks it), and its row is the reason
    # the generic bulk operator must never be dispatched for it either.
    refused = frozenset({30})
    assert not (refused & consumer)
    assert frozenset(SCHEME_NATIVE_REFL_10CM) == (consumer - producer) | refused
    assert not (frozenset(SCHEME_NATIVE_REFL_10CM) & producer)
    for scheme, reason in SCHEME_NATIVE_REFL_10CM.items():
        assert len(reason) > 80, f"mp={scheme} has a reason in name only"


def test_wrfout_refl_10cm_variable_is_registry_faithful(tmp_path):
    """Opt-in REFL_10CM rides a frame dict into wrfout with the
    Registry.EM_COMMON:1596 attribute strings, unstaggered 3-D layout,
    and georeferencing alongside XLAT/XLONG."""
    import netCDF4

    from woof.io.wrfout import _VAR_META, WrfoutWriter, wrf_time_str

    assert _VAR_META["REFL_10CM"] == (
        "Radar reflectivity (lamda = 10 cm)", "dBZ")

    nz, ny, nx = 4, 3, 5
    rng = np.random.default_rng(3)
    refl = np.maximum(rng.uniform(-40.0, 60.0, (nz, ny, nx)),
                      -35.0).astype(np.float32)
    lat = np.linspace(30.0, 40.0, ny * nx).reshape(ny, nx).astype(np.float32)
    lon = np.linspace(-100.0, -80.0, ny * nx).reshape(ny, nx).astype(
        np.float32)
    path = tmp_path / "wrfout_refl.nc"
    with WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=1000.0,
                      dy=1000.0) as writer:
        writer.write_frame(wrf_time_str(0.0),
                           {"REFL_10CM": refl, "XLAT": lat, "XLONG": lon})

    with netCDF4.Dataset(path) as ds:
        var = ds.variables["REFL_10CM"]
        assert var.dimensions == ("Time", "bottom_top", "south_north",
                                  "west_east")
        assert var.description == "Radar reflectivity (lamda = 10 cm)"
        assert var.units == "dBZ"
        assert var.stagger == ""
        assert var.MemoryOrder == "XYZ"
        assert int(var.FieldType) == 104
        got = np.asarray(var[0])
        assert np.isfinite(got).all() and got.min() >= -35.0
        np.testing.assert_allclose(got, refl, rtol=0.0, atol=0.0)
        assert ds.variables["XLAT"].shape == (1, ny, nx)
        assert ds.variables["XLONG"].shape == (1, ny, nx)


def test_make_reflectivity_map_is_optional_and_renders(tmp_path):
    """The Task 9 map hook writes the composite PNG from a wrfout that
    carries REFL_10CM and refuses one that does not (the current T7
    product path never calls it, so nothing there changes).

    Drawn by ``rw_wrfbatch`` since the render-law rewiring (bceb46073),
    whose raw-wrfout road probes ``T`` for the grid dimensions and
    requires a run origin (``START_DATE``) -- both of which every real
    product tape carries -- so the tape here carries the real shape
    rather than the matplotlib-era three-variable minimum.  The refusal
    moved with the same commit: the renderer skips the panel and the
    hook raises RuntimeError naming the missing field, where the old
    in-process reader raised ValueError.
    """
    from woof.io.wrfout import WrfoutWriter, wrf_time_str
    from woof.verify.cases.real74_d01 import make_reflectivity_map

    nz, ny, nx = 4, 6, 8
    lat = np.linspace(30.0, 40.0, ny)[:, None].repeat(nx, 1).astype(
        np.float32)
    lon = np.linspace(-100.0, -88.0, nx)[None, :].repeat(ny, 0).astype(
        np.float32)
    refl = np.full((nz, ny, nx), -35.0, dtype=np.float32)
    refl[:2, 2:4, 3:6] = 55.0                      # a convective core
    theta_pert = np.zeros((nz, ny, nx), dtype=np.float32)
    stamp = wrf_time_str(0.0)
    origin = {"START_DATE": stamp, "SIMULATION_START_DATE": stamp}
    path = tmp_path / "wrfout_refl.nc"
    with WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=1000.0,
                      dy=1000.0, global_attrs=origin) as writer:
        writer.write_frame(stamp, {"REFL_10CM": refl, "XLAT": lat,
                                   "XLONG": lon, "T": theta_pert})
    out = make_reflectivity_map(path, tmp_path / "maps")
    assert out.exists() and out.stat().st_size > 0

    bare = tmp_path / "wrfout_bare.nc"
    with WrfoutWriter(bare, nx=nx, ny=ny, nz=nz, dx=1000.0,
                      dy=1000.0, global_attrs=origin) as writer:
        writer.write_frame(stamp, {"XLAT": lat, "XLONG": lon,
                                   "T": theta_pert})
    with pytest.raises(RuntimeError, match="REFL_10CM"):
        make_reflectivity_map(bare, tmp_path / "maps")


def test_wk82_reflectivity_gate_admits_complete_no_pbl_operator():
    from dataclasses import replace

    import woof.verify.cases.wk82 as wk82
    from woof.config import validate_run_config

    cfg = replace(wk82.default_config(), mp_physics=10,
                  run_seconds=5400.0)
    admitted = validate_run_config(cfg)
    assert admitted.km_opt == 4 and admitted.bl_pbl_physics == 0
