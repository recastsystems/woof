"""Downward longwave has a source, or the run refuses to start.

THE DEFECT THIS FILE CLOSES.

``woof.core.physics.initialize_physics`` defaulted ``glw=300.0`` through
1.8.7, and ``woof/core/dudhia.py`` -- shortwave only -- returns
``glw=fields["glw"]``, the array it was handed, unchanged, on every
radiation call.  No production call site ever passed ``glw=``.  So every
run with ``ra_lw_physics = 0`` integrated a downward longwave of exactly
300.0 W m-2, everywhere, forever: a number with no relationship to
temperature, humidity or cloud, and one that looks entirely plausible in
a wrfout GLW row.

Radiative equilibrium at 300 W m-2 is 269.7 K.  A Gulf-coast October
night runs near 410 W m-2, or 291.6 K.  That ~105 W m-2 deficit is what
produced the shipped user report of 2 m dewpoints collapsing tens of
degrees below the airmass overnight.

WHAT WRF v4.6.1 DOES, read out of the source rather than remembered
(``phys/module_radiation_driver.F`` in the pinned reference bundle):

* ``radiation_driver`` returns at ``:1068`` only when BOTH streams are
  off, so ``ra_lw_physics = 0`` with ``ra_sw_physics > 0`` reaches
  ``lwrad_select`` at ``:1839``.  That select has cases for RRTM,
  Goddard, GFDL, CAM, RRTMG, RRTMG-fast, RRTMK, Held-Suarez and
  Fu-Liou-Gu -- and NO ``CASE (0)``.  It falls to ``CASE DEFAULT`` at
  ``:2245`` and calls ``wrf_error_fatal('The longwave option does not
  exist: lw_physics = 0')``.  Longwave-only is fatal too:
  ``swrad_select``'s ``CASE (0)`` at ``:2827`` calls ``wrf_error_fatal``
  for every ``lw_physics`` except Held-Suarez (``:2831-2835``), the one
  idealised case it excepts.
* With both streams off the driver returns early and ``GLW`` keeps its
  Registry-allocated 0.0 W m-2, which the land surface then consumes.

So woof refusing the shortwave-only pairing is WRF-conformant, and
woof's declared constant is a documented divergence from WRF's 0.0 in
the both-off case -- because zero downward longwave is not a physical
atmosphere either.

Every test here runs the real ``initialize_physics`` on a real card.
"""
# Replaced in 2.9.1 (the WRF exact chain): test_a_longwave_scheme_owns_the_buffer_and_gets_the_historical_fill -> test_a_longwave_scheme_owns_the_buffer_and_preserves_its_mode_fill.
from __future__ import annotations

import numpy as np
import pytest

from woof.core.physics import DECLARED_CONSTANT_GLW_WM2, _resolve_initial_glw

from conftest import requires_gpu  # noqa: E402


# ---------------------------------------------------------------------------
# The resolver, in isolation.  No card needed: it decides on selectors.
# ---------------------------------------------------------------------------

def test_the_constant_is_the_value_1_8_7_allocated():
    """Unchanged on purpose, so declaring it moves no existing column."""
    assert DECLARED_CONSTANT_GLW_WM2 == 300.0


@pytest.mark.parametrize("exact", [False, True])
def test_a_longwave_scheme_owns_the_buffer_and_preserves_its_mode_fill(monkeypatch, exact):
    """The breakage this prevents: the t=0 history frame is written before
    the first radiation call, so the unwritten buffer's fill IS what wrfout
    publishes at t=0.  With the old 300 fill every LW-active run published
    "GLW = 300 W m-2" in every column at t=0 (WRF publishes 0.0), and the
    matched woof-default-suite run and all 279 integrated combo-sweep runs
    differed from WRF in that word.  The fill is WRF's Registry 0.0; the
    carrier contract keeps any surface call from integrating it."""
    from woof.core import physics
    from woof.core.physics import SCHEME_GLW_UNWRITTEN_FILL_WM2
    monkeypatch.setattr(physics, "WRF_EXACT", exact)
    for lsm in (0, 2, 3, 4):
        value, provenance = _resolve_initial_glw(
            None, ra_lw_physics=4, radiation_active=True,
            sf_surface_physics=lsm)
        assert provenance == "scheme"
        expected = SCHEME_GLW_UNWRITTEN_FILL_WM2 if exact else DECLARED_CONSTANT_GLW_WM2
        assert value == expected
        assert value == (0.0 if exact else 300.0)


@pytest.mark.parametrize("exact", [False, True])
def test_selector31_initial_surface_flux_matches_its_native_zero(monkeypatch, exact):
    import woof.core.physics as physics
    monkeypatch.setattr(physics, "WRF_EXACT", exact)
    value, provenance = _resolve_initial_glw(
        None, ra_lw_physics=31, radiation_active=True, sf_surface_physics=2)
    assert np.float32(value).tobytes() == np.float32(0).tobytes()
    assert provenance == "scheme"


def test_a_typed_value_is_a_declaration_and_is_used_verbatim():
    """The idealised route: the caller says the number out loud."""
    value, provenance = _resolve_initial_glw(
        275.0, ra_lw_physics=0, radiation_active=True, sf_surface_physics=2)
    assert value == 275.0
    assert provenance == "declared"

    field = np.full((3, 4), 402.5, np.float32)
    value, provenance = _resolve_initial_glw(
        field, ra_lw_physics=0, radiation_active=False,
        sf_surface_physics=2)
    assert value is field
    assert provenance == "declared"


def test_a_buffer_nothing_reads_and_nothing_publishes_is_not_a_refusal():
    """No land surface and no radiation: GLW reaches no consumer or file."""
    value, provenance = _resolve_initial_glw(
        None, ra_lw_physics=0, radiation_active=False,
        sf_surface_physics=0)
    assert value == DECLARED_CONSTANT_GLW_WM2
    assert provenance == "unused"


@pytest.mark.parametrize("scheme,name", [(2, "Noah LSM"), (3, "RUC LSM"),
                                         (4, "Noah-MP")])
def test_a_land_surface_with_no_longwave_scheme_refuses(scheme, name):
    """The consumer case: something reads GLW and nothing computes it."""
    with pytest.raises(ValueError) as caught:
        _resolve_initial_glw(None, ra_lw_physics=0, radiation_active=True,
                             sf_surface_physics=scheme)
    message = str(caught.value)
    assert "ra_lw_physics=0" in message              # the offending key
    assert f"sf_surface_physics={scheme}" in message  # who reads it
    assert name in message
    assert "will not invent one" in message          # no silent substitution
    assert "ra_lw_physics=4" in message              # remedy 1
    assert "glw=<W m-2>" in message                  # remedy 2
    assert "glw=<array>" in message                  # remedy 3
    assert "269.7 K" in message and "410 W m-2" in message   # the physics
    assert "module_radiation_driver.F:2245" in message       # WRF's answer


def test_a_published_glw_row_with_no_longwave_scheme_also_refuses():
    """The publisher case: shortwave on, so wrfout carries a GLW row.

    Even with no land-surface scheme to consume it, an active radiation
    slot writes GLW into every history frame.  A fabricated field in a
    wrfout file is indistinguishable from a measured one downstream, so
    it is refused rather than written.
    """
    with pytest.raises(ValueError) as caught:
        _resolve_initial_glw(None, ra_lw_physics=0, radiation_active=True,
                             sf_surface_physics=0)
    assert "written to every wrfout frame" in str(caught.value)


# ---------------------------------------------------------------------------
# The real driver, on the card.
# ---------------------------------------------------------------------------

def _state(**overrides):
    from test_physics_driver import _base_config
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced

    cfg = _base_config(**overrides)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 298.0 + 0.004 * np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(
        cfg, coord, base,
        lambda z: 0.010 * np.exp(-np.asarray(z, np.float64) / 2400.0))
    return state, cfg


@pytest.mark.gpu
@requires_gpu
def test_initialize_physics_refuses_an_undeclared_constant_on_the_card():
    """End to end through the real entry point, not the helper."""
    from woof.core.physics import initialize_physics

    state, cfg = _state(ra_physics=0, ra_lw_physics=0, ra_sw_physics=1,
                        sf_surface_physics=2, sf_sfclay_physics=91,
                        bl_pbl_physics=1)
    with pytest.raises(ValueError, match="downward longwave"):
        initialize_physics(
            state, cfg, radiation_start_time=__import__(
                "datetime").datetime(2011, 4, 26, 8),
            radiation_latitude=np.full((cfg.ny, cfg.nx), 33.8),
            radiation_longitude=np.full((cfg.ny, cfg.nx), -87.29))


@pytest.mark.gpu
@requires_gpu
def test_a_declared_constant_still_runs_and_says_so_on_the_driver():
    """An idealised column that types the number keeps working."""
    import cupy as cp

    from woof.core.physics import initialize_physics

    state, cfg = _state(ra_physics=0, ra_lw_physics=0, ra_sw_physics=1,
                        sf_surface_physics=2, sf_sfclay_physics=91,
                        bl_pbl_physics=1)
    driver = initialize_physics(
        state, cfg, glw=DECLARED_CONSTANT_GLW_WM2,
        radiation_start_time=__import__("datetime").datetime(2011, 4, 26, 8),
        radiation_latitude=np.full((cfg.ny, cfg.nx), 33.8),
        radiation_longitude=np.full((cfg.ny, cfg.nx), -87.29))
    assert driver.glw_provenance == "declared"
    assert float(cp.asnumpy(driver.fields["glw"]).max()) == 300.0


# ---------------------------------------------------------------------------
# Door/engine parity: the load guard refuses EXACTLY what the resolver
# refuses, so a config either fails at load or runs.
# ---------------------------------------------------------------------------

def test_the_load_guard_and_the_resolver_refuse_the_same_selector_set():
    """Finding of record: a config must never pass the door and die deep.

    Sweeps every (lw, sw, sf_surface_physics) combination and asserts
    the config-load guard (physics_compat.constant_longwave_refusal,
    no token) refuses precisely when ``_resolve_initial_glw`` (no typed
    ``glw``) raises.  Both read
    ``physics_compat.downward_longwave_disposition``; this test is what
    keeps that true when someone edits one of them.
    """
    from types import SimpleNamespace

    from woof.physics_compat import constant_longwave_refusal

    for lw in (0, 4):
        for sw in (0, 1, 4):
            for surface in (0, 2, 3, 4):
                run = SimpleNamespace(
                    grid_id=1, ra_lw_physics=lw, ra_sw_physics=sw,
                    ra_physics=0, sf_surface_physics=surface)
                door = constant_longwave_refusal([run]) is not None
                try:
                    _resolve_initial_glw(
                        None, ra_lw_physics=lw, radiation_active=bool(lw or sw),
                        sf_surface_physics=surface)
                except ValueError:
                    engine = True
                else:
                    engine = False
                assert door == engine, (
                    f"lw={lw} sw={sw} sfc={surface}: load guard "
                    f"{'refuses' if door else 'admits'} but the engine "
                    f"{'refuses' if engine else 'admits'}")


def test_the_load_guard_names_the_consumer_the_classifier_returned(
        monkeypatch):
    """The door must not re-derive what the classifier already told it.

    ``downward_longwave_disposition`` returns ``(kind, consumer)``, and
    the engine and the receipt both use the returned name.  The door used
    to throw it away and subscript ``_GLW_CONSUMING_SURFACE_SCHEMES``
    again, which is a second, independent definition of "consuming": the
    first surface scheme the classifier calls consuming before that table
    lists it turns this refusal into ``KeyError``, and a guard that
    crashes is a guard that does not refuse.

    Stubbing the classifier is the whole point -- it is what lets the
    table and the classifier disagree today, on purpose, instead of
    waiting for a future scheme to do it by accident.
    """
    from types import SimpleNamespace

    from woof import physics_compat
    from woof.physics_compat import CONSTANT_DOWNWARD_LONGWAVE_ACK

    def run(surface):
        return SimpleNamespace(grid_id=1, ra_lw_physics=0, ra_sw_physics=1,
                               ra_physics=0, sf_surface_physics=surface)

    # Byte-for-byte, the reachable schemes are untouched by all of this.
    assert physics_compat.constant_longwave_refusal([run(2)]).startswith(
        "domain(s) 1 run Noah LSM (sf_surface_physics 2) with "
        "ra_lw_physics 0 -- no longwave scheme, with shortwave still ON "
        "(ra_sw_physics 1) -- so their downward longwave would be a fixed "
        "300 W m-2 for the whole forecast rather than a computed flux.")

    # A scheme the classifier consumes and the table has never heard of.
    for consumer, expected in (("Bogus LSM", "Bogus LSM"),
                               (None, "a GLW-consuming land-surface scheme")):
        monkeypatch.setattr(
            physics_compat, "downward_longwave_disposition",
            lambda **kw: (("consumed", consumer)
                          if int(kw["sf_surface_physics"]) == 5
                          else ("scheme", None)))
        assert 5 not in physics_compat._GLW_CONSUMING_SURFACE_SCHEMES
        refusal = physics_compat.constant_longwave_refusal([run(5)])
        assert refusal is not None, "consumed must refuse, not admit"
        assert expected in refusal
        assert "sf_surface_physics 5" in refusal
        assert CONSTANT_DOWNWARD_LONGWAVE_ACK in refusal   # the remedy


def test_the_receipt_line_tells_the_truth_for_every_disposition():
    """One source of truth: the receipt says what the engine enacts.

    The lw=0/sw=1/sfc=0 pairing used to get 'nothing reads or publishes
    GLW in this suite' with no token (false: the row is published) and
    'the land surface integrates this one number' with the token (false:
    there is no land surface).  Every (kind, declared) cell is asserted.
    """
    from types import SimpleNamespace

    from woof.physics_compat import CONSTANT_DOWNWARD_LONGWAVE_ACK
    from woof.runtime import downward_longwave_source

    def line(lw, sw, surface, *, token):
        exp = SimpleNamespace(acknowledgements=(
            (CONSTANT_DOWNWARD_LONGWAVE_ACK,) if token else ()))
        cfg = SimpleNamespace(ra_lw_physics=lw, ra_sw_physics=sw,
                              ra_physics=0, sf_surface_physics=surface)
        return downward_longwave_source(exp, cfg)

    assert "computed every radiation call" in line(4, 4, 2, token=False)
    consumed = line(0, 1, 2, token=True)
    assert "DECLARED CONSTANT 300" in consumed
    assert "Noah LSM" in consumed and "integrates" in consumed
    published = line(0, 1, 0, token=True)
    assert "DECLARED CONSTANT 300" in published
    assert "wrfout frame" in published
    assert "integrates" not in published        # nothing does
    unused_declared = line(0, 0, 0, token=True)
    assert "UNUSED" in unused_declared
    assert "no scheme and no" in unused_declared
    unused = line(0, 0, 0, token=False)
    assert "nothing reads or publishes GLW" in unused
    # Guarded configurations without the token cannot arrive through
    # build_experiment; a hand-assembled experiment gets the accurate
    # refusal sentence, never the unused one.
    for lw, sw, surface in ((0, 1, 2), (0, 1, 0), (0, 0, 2)):
        no_source = line(lw, sw, surface, token=False)
        assert "NO SOURCE" in no_source, (lw, sw, surface, no_source)


def test_the_tree_report_names_every_domain_not_just_the_root():
    """The tree receipt's stated job is to name where EACH domain's
    longwave comes from; it used to print one line describing the root.
    Asserted on a real two-domain experiment built from the shipped MYNN
    no-radiation descriptor, geometry shrunk.
    """
    import tomllib
    from pathlib import Path
    from types import SimpleNamespace

    from woof.experiment import build_experiment
    from woof.runtime import resolved_tree_config_report

    raw = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "configs"
         / "real74_4dom_mynn_norad.toml").read_text(encoding="utf-8"))
    raw.pop("case_data", None)
    raw["domain"] = raw["domain"][:2]
    raw["domain"][0]["nx"] = 60
    raw["domain"][0]["ny"] = 50
    raw["domain"][1].update({"nx": 48, "ny": 40,
                             "i_parent_start": 12, "j_parent_start": 12})
    exp = build_experiment(raw, source="<tree-report-probe>")
    report = resolved_tree_config_report(
        exp, SimpleNamespace(resolved_inputs=lambda: []),
        SimpleNamespace(fingerprint="0" * 8))
    for grid_id in (1, 2):
        line = [l for l in report.splitlines() if l.startswith(
            f"  domain.d{grid_id:02d}.radiation.downward_longwave = ")]
        assert len(line) == 1, (grid_id, report)
        assert "DECLARED CONSTANT 300" in line[0]
    # The root-only spelling is gone; a reader can no longer mistake one
    # root line for a statement about the children.
    assert "\n  radiation.downward_longwave" not in report


def test_the_hrrr_state_proof_tool_declares_its_frozen_constant():
    """Finding of record: the TEST was patched while the TOOL kept the
    refused call.  The tool's frozen namelist is lw=0/sw=1/Noah -- the
    consumed disposition -- so its ``initialize_physics`` call must type
    ``glw=``.  Checked in the tool's AST, so reverting the tool edit
    while its green test still passes goes red HERE.
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1]
              / "tools" / "hrrr_state_proof.py").read_text(encoding="utf-8")
    calls = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call)
             and getattr(node.func, "id", getattr(node.func, "attr", None))
             == "initialize_physics"]
    assert calls, "tools/hrrr_state_proof.py no longer calls initialize_physics"
    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        assert "glw" in keywords, (
            f"initialize_physics call at line {call.lineno} of "
            "tools/hrrr_state_proof.py passes no glw=; its frozen d01 "
            "namelist runs Noah with ra_lw_physics 0, so the call refuses "
            "at runtime without the declaration")


# ---------------------------------------------------------------------------
# The preflight and instrument paths, on the card.
# ---------------------------------------------------------------------------

def _small_declared_experiment():
    """configs/real74_4dom_mynn_norad.toml shrunk to one 24x20 domain.

    The real file is the point: it is one of the two configs that exist
    ONLY to measure device footprint with ``woof check --alloc``, and
    the one the false refusal was reproduced on.  Only the geometry is
    shrunk so the alloc proof fits beside a test suite.
    """
    import tomllib
    from pathlib import Path

    from woof.experiment import build_experiment

    raw = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "configs"
         / "real74_4dom_mynn_norad.toml").read_text(encoding="utf-8"))
    raw.pop("case_data", None)
    raw["domain"] = raw["domain"][:1]
    raw["domain"][0]["nx"] = 24
    raw["domain"][0]["ny"] = 20
    return build_experiment(raw, source="<alloc-declared>")


def test_a_declared_experiment_can_name_its_own_constant():
    """The acknowledgement names the CLAIM; the field names the NUMBER.

    Until ``[experiment] constant_glw_wm2`` existed, an experiment whose
    case radiates near 410 W m-2 had to run at the shipped 300 and call
    the difference declared -- the engine accepts any float, and only the
    config had no way to say one.  The field is readable exactly where
    the fabrication is declared, because a number that looks like it took
    effect and did not is the defect this whole guard is about.
    """

    import tomllib
    from pathlib import Path

    import pytest as _pytest

    from woof import runtime
    from woof.core.physics import DECLARED_CONSTANT_GLW_WM2
    from woof.experiment import build_experiment
    from woof.physics_compat import CONSTANT_DOWNWARD_LONGWAVE_ACK

    raw = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "configs"
         / "real74_4dom_mynn_norad.toml").read_text(encoding="utf-8"))
    raw.pop("case_data", None)
    raw["domain"] = raw["domain"][:1]
    raw["domain"][0]["nx"] = 24
    raw["domain"][0]["ny"] = 20

    exp = build_experiment(dict(raw), source="<declared-default>")
    assert CONSTANT_DOWNWARD_LONGWAVE_ACK in exp.acknowledgements
    assert runtime.declared_constant_glw(exp) == DECLARED_CONSTANT_GLW_WM2

    named = dict(raw)
    named["experiment"] = dict(raw["experiment"]) | {"constant_glw_wm2": 410.0}
    exp = build_experiment(named, source="<declared-410>")
    assert runtime.declared_constant_glw(exp) == 410.0

    undeclared = dict(raw)
    undeclared["experiment"] = dict(raw["experiment"]) | {
        "constant_glw_wm2": 410.0,
        "acknowledgements": [
            token for token in raw["experiment"].get("acknowledgements", [])
            if token != CONSTANT_DOWNWARD_LONGWAVE_ACK],
    }
    with _pytest.raises(ValueError) as caught:
        build_experiment(undeclared, source="<undeclared-410>")
    assert "constant_glw_wm2" in str(caught.value)
    assert CONSTANT_DOWNWARD_LONGWAVE_ACK in str(caught.value)

    negative = dict(raw)
    negative["experiment"] = dict(raw["experiment"]) | {
        "constant_glw_wm2": -1.0}
    with _pytest.raises(ValueError, match="finite positive"):
        build_experiment(negative, source="<negative>")


def test_an_absent_constant_leaves_the_restart_identity_where_it_was():
    """ABSENT stays absent, the convention every added field follows.

    A new experiment field that serialized as ``null`` would move the
    fingerprint of every experiment written before it existed and refuse
    every checkpoint on disk, so the identity payload drops it when it is
    None and binds it when it is declared -- a resume across 300 and 410
    is not the same trajectory.
    """

    from woof.core.model import restart_identity_payload

    exp = _small_declared_experiment()
    assert "constant_glw_wm2" not in restart_identity_payload(exp)
    from dataclasses import replace
    assert restart_identity_payload(
        replace(exp, constant_glw_wm2=410.0))["constant_glw_wm2"] == 410.0


@pytest.mark.gpu
@requires_gpu
def test_alloc_preflight_reaches_the_device_for_a_declared_config():
    """Finding of record: ``woof check --alloc`` false-refused every
    config that correctly declared the constant, because preflight's
    driver materialization never typed the declaration the preparers
    type.  The experiment here carries the token (asserted), and the
    alloc proof must therefore construct the full driver set and report.
    """
    from woof.core.preflight import run_alloc_preflight
    from woof.physics_compat import CONSTANT_DOWNWARD_LONGWAVE_ACK

    exp = _small_declared_experiment()
    assert CONSTANT_DOWNWARD_LONGWAVE_ACK in exp.acknowledgements
    report = run_alloc_preflight(exp)
    assert report.free_before_bytes > 0
    assert report.pool_used_peak_bytes > 0


@pytest.mark.gpu
@requires_gpu
def test_the_health_census_instrument_survives_the_lw0_combinations():
    """Finding of record (at risk): the census sweeps sf_surface_physics
    with radiation inherited from its base descriptor, so aimed at a
    no-radiation descriptor its ``initialize_physics`` call refused.
    ``_attach_driver`` is the census's real call path; run it on the
    exact exposed combination (lw=0, Noah) and on the scheme path.
    """
    from datetime import datetime

    from tools.health_field_census import _attach_driver

    state, cfg = _state(ra_physics=0, ra_lw_physics=0, ra_sw_physics=1,
                        sf_surface_physics=2, sf_sfclay_physics=91,
                        bl_pbl_physics=1)
    driver = _attach_driver(state, cfg, datetime(2011, 4, 26, 8),
                            33.8, -87.29)
    assert driver.glw_provenance == "declared"


@pytest.mark.gpu
@requires_gpu
def test_the_committed_noop_digest_receipt_reproduces():
    """Finding of record: the full-state byte-identity proof was run by
    hand and gated by nothing.  This test runs the committed probe
    (evidence/glw-source/glw_noop_probe.py -- the artifact itself, as a
    subprocess) against THIS tree and compares every digest line with
    the committed receipt, so any future change that moves a healthy
    full-radiation run goes red here rather than at certification.
    After an intentional physics or output-inventory change, regenerate
    the receipt with that same probe on the named card.

    The receipt's digests were produced on the RTX 5090 this project
    certifies on; on any other device the byte comparison is void, so it
    is skipped there with the weaker non-vacuity assertions still made.
    """
    import subprocess
    import sys
    from pathlib import Path

    import cupy as cp

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "evidence" / "glw-source"
                             / "glw_noop_probe.py"), str(root)],
        capture_output=True, text=True, timeout=600)
    if result.returncode != 0 and "cudaErrorNoDevice" in result.stderr:
        # HARNESS CONDITION, not a probe verdict: THIS process holds the
        # device (requires_gpu passed and the cupy import above worked),
        # yet the child's very first enumeration saw none.  The child
        # inherits this environment unchanged and runs no woof code
        # before that first cp.zeros, so nothing the probe measures has
        # executed -- the environment withheld the device from child
        # processes.  Seen on a containerized rental mid-session (2.0
        # battery, node 7: parent shard green for 3127 GPU tests around
        # this row, probe + test bytes identical at the passing 1.9.1
        # base).  A digest mismatch, a probe crash past enumeration, or
        # any other nonzero exit still FAILS below.
        pytest.skip(
            "the probe subprocess could not see a CUDA device although "
            "this process holds one; the environment withheld device "
            "visibility from child processes, so the byte-identity "
            "measurement is void here rather than failed: "
            + result.stderr[-500:])
    assert result.returncode == 0, result.stderr[-2000:]
    lines = result.stdout.splitlines()
    by_key = {key.strip(): value.strip() for key, _, value in
              (line.partition("=") for line in lines) if key.strip()}
    # Non-vacuity on any card: the scheme wrote GLW over the allocated
    # fill, and the provenance says a scheme owns it.
    assert by_key["glw_provenance"] == "scheme"
    post_glw = [line for line in lines if line.split()[:2] == ["FIELD", "glw"]]
    assert post_glw and post_glw[0].split()[-1] != by_key["ALLOC glw"]

    receipt = (root / "evidence" / "glw-source"
               / "full-radiation.lane-glw.digests.txt").read_text(
        encoding="utf-8").splitlines()
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    if "RTX 5090" not in device:
        pytest.skip(f"receipt digests are RTX 5090-produced; byte "
                    f"comparison void on {device}")
    # Drop the two tree-path header lines from both sides; every
    # remaining line -- fill value, step count, every digest -- must
    # match byte for byte.
    assert lines[2:] == receipt[2:]


@pytest.mark.gpu
@requires_gpu
def test_a_full_radiation_run_is_untouched_by_any_of_this():
    """THE no-op proof, on the configuration this lane must not move.

    Both streams on: the GLW buffer is scratch that the longwave scheme
    overwrites at its first radiation call, and its allocated contents
    are WRF's own unwritten word, 0.0 (they were 300, which the t=0
    history frame then published as a downward longwave nobody computed;
    see _resolve_initial_glw).  Compared as raw bytes, not as floats, so
    a negative zero cannot pass.
    """
    import cupy as cp

    from woof.core.physics import initialize_physics

    state, cfg = _state(ra_physics=0, ra_lw_physics=4, ra_sw_physics=4,
                        sf_surface_physics=2, sf_sfclay_physics=91,
                        bl_pbl_physics=1)
    driver = initialize_physics(
        state, cfg,
        radiation_start_time=__import__("datetime").datetime(2011, 4, 26, 8),
        radiation_latitude=np.full((cfg.ny, cfg.nx), 33.8),
        radiation_longitude=np.full((cfg.ny, cfg.nx), -87.29))
    assert driver.glw_provenance == "scheme"
    allocated = cp.asnumpy(driver.fields["glw"])
    from woof.wrf_exact import ENABLED
    expected = np.full(allocated.shape, 0.0 if ENABLED else 300.0, allocated.dtype)
    assert allocated.tobytes() == expected.tobytes()
