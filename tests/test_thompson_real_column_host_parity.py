"""mp_physics=28 on real model columns: the port's own kernels, compiled for
the host, against WRF v4.6.1's own Fortran, process rate by process rate.

The breakage each gate prevents, named:

1. ``test_rate_readback_anchors_match_the_kernels`` -- the harness reads
   every process rate out of the production kernels by inserting readback
   calls at one anchor line per kernel, next to local variables it names
   (tools/thompson_real_column_parity/port_rates.py): the three aerosol
   units for mp=28 and the classic kernels of thompson.cu for mp=8.  A
   kernel edit that moves an anchor or renames a rate variable breaks the
   harness, and with it the only process-by-process check of either port
   against WRF on real columns.  Pure text; runs everywhere.

2. ``test_the_fixture_covers_every_repaired_regime`` -- the committed
   fixture was cut per regime from WRF's own checkpoint streams
   (tools/thompson_real_column_parity/make_fixture.py).  A re-cut that
   loses a regime would leave the repair that lives there unguarded while
   gate 3 stays green.  Pure data; runs everywhere.

3. ``test_real_columns_against_wrf461`` -- the stock fixture
   ``tests/data/thompson_real_columns_wrf461.npz`` retains 42 saved real-data
   model columns and stock answers. The numeric gate uses a separate
   independently compiled WRF reference with exactly eight invalid
   rain-graupel table reads changed to MIN(idx_bg(k),dimNRHG), on the same
   input words. No threshold or exception allowance changes. The port's production adapter runs on the
   same float32 inputs through the host backend, and every process rate,
   the final state, reflectivity and surface precipitation are compared.

   What it pins: the cells where a quantity differs from WRF by more than
   1e-2 relative and neither rounding rule explains the gap (the port's own
   response to a one-unit nudge of every input, or four float32 units of
   the largest value the cell held; under mp=8 also the two named rules
   for the rates rounding decides there, see fixture_check.py -- mp=28 has
   none since lane/mp28fix-warm-network repaired the transcription
   differences they covered).  Every process rate and
   every final-state quantity must have none.  (Until the no-micro column
   exit and the terminal vapour floor were repaired the final vapour was
   pinned at four such cells: G, WRF floors vapour at 1.E-10 at every level
   of a column with microphysics (:3974) and the port left a level at
   exactly zero at zero.)  Undoing any of the twelve repairs
   of 2026-09-23 fails this gate: measured for the entry rewrite (the echo
   then differs by 9.4 dB, and ice nucleation in four cells) and the cloud
   fallout gate (cloud water and droplet number in two cells), and the
   port as it stood before the repairs fails it on 20 rates and 10
   final-state quantities, its echo 9.4 dB from WRF.  Reflectivity is
   held to 0.01 dB.

   Where it runs: a POSIX box with a C++ compiler and the staged Thompson
   table set, or the Windows desktop through WSL with the same.  It skips,
   naming the missing piece, where either is absent -- the hosted CI runners
   carry no ``freezeH2O.dat`` (254 MB, not packaged).

4. ``test_real_columns_against_wrf461_mp8`` -- the same 42 columns with the
   answers WRF's same module gives them as classic Thompson
   (``tests/data/thompson_real_columns_wrf461_mp8.npz``, written by
   ``make_fixture.py --mp8``), graded through ``woof.core.microphysics.
   _apply_thompson``.  Like gate 3, the numeric grade reads the independently
   compiled reference with only the eight rain-graupel table reads changed
   to MIN(idx_bg(k),dimNRHG)
   (``thompson_real_columns_wrf461_mp8_corrected_racg.npz``,
   ``corrected_real_reference.py fixture --mp 8``) on the same input words:
   classic Thompson reads the one slab too (thompson_racg_index), a declared
   divergence.  Against the stock answers the port now differs in exactly
   the rain-graupel rates and what follows them (74 prr_rcg, 80 pnr_rcg,
   74 prg_rcg and 55 png_rcg cells, 7 prw_vcd, one final qg).  mp=8 runs the shared ``thompson.cu`` kernels, and
   before this gate nothing compared its process rates with WRF's.  It pins,
   per rate and per final-state quantity, the cells beyond 1e-2 that no
   rounding rule explains, EXACTLY: a count that grows is a new difference
   from WRF, and a count that shrinks is a repair that must retire its pin
   here.  Every count it recorded when it was committed (f382c8825) was
   retired by the repair that closed it, so the pins are empty and the
   echo is held to 0.05 dB (see ``PINNED_RATE_CELLS_MP8``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TOOL = _ROOT / "tools" / "thompson_real_column_parity"
_FIXTURE = _ROOT / "tests" / "data" / "thompson_real_columns_wrf461.npz"
_FIXTURE_CORRECTED_RACG = (_ROOT / "tests" / "data"
                           / "thompson_real_columns_wrf461_corrected_racg.npz")
_FIXTURE_MP8 = (_ROOT / "tests" / "data"
                / "thompson_real_columns_wrf461_mp8.npz")
_FIXTURE_MP8_CORRECTED_RACG = (
    _ROOT / "tests" / "data"
    / "thompson_real_columns_wrf461_mp8_corrected_racg.npz")

#: Final-state cells beyond 1e-2 relative that no rounding rule explains, per
#: quantity, on the committed fixture.  Quantities absent here must have none.
#: Empty: the one entry it held, ``"qv": 4`` for the unrepaired vapour floor
#: (G, module_mp_thompson.F:2020 and :3974), was retired with its repair.
PINNED_FINAL_CELLS: dict[str, int] = {}
REFL_MAX_DB = 1.0e-2

# Retained base diagnostics: no allowances added by this intake.
PINNED_RATE_CELLS_MP8: dict[str, int] = {}
PINNED_FINAL_CELLS_MP8: dict[str, int] = {}
PINNED_REFL_MP8 = {"max_abs_db": 0.05, "n_gt_0p1_db": 0}
PINNED_T_EXIT_REL_MP8 = 1.0e-5

#: Every regime a repair acts in, by the label make_fixture.py gives it.
REQUIRED_REGIMES = (
    "A entry ice without number",
    "B cold snow riming",
    "C rain emptied at the source stage",
    "C/F graupel emptied at the source stage",
    "D/H mass above R1 below it as a concentration",
    "E ice collection",
    "F melting graupel",
    "G vapour at zero",
    "K orphan ice number where WRF nucleates",
    "K orphan rain number",
    "L cloud formed where the gate is shut",
    "O graupel at or below R1 at entry",
    "rain evaporation at saturation",
    "rain break-up diameter",
)


def test_collision_pair_rounding_and_sign_survive_source_assembly():
    """Compile the real probe and check conserved transfer words, including zero."""
    from tools.thompson_real_column_parity import host_backend

    try:
        host_backend.compiler()
    except RuntimeError:
        pytest.skip("the Thompson host probe requires a C++ compiler")
    kernel = host_backend.host_get_kernel(
        "thompson_aerosol_probe", "thompson_aa_probe_reenforce_pair")
    a = np.array([1.0, -2.0, 1.0 + 2.0**-25, -2.0**-160], dtype=np.float64)
    b = np.array([-3.0, 0.25, -4.0, 1.0], dtype=np.float64)
    actual_a = np.empty_like(a)
    actual_b = np.empty_like(b)
    kernel((1,), (4,), (a, b, actual_a, actual_b, np.int32(len(a))))
    expected_a = np.array([1.0, -0.25, 1.0, -0.0], dtype=np.float64)
    np.testing.assert_array_equal(actual_a.view(np.uint64), expected_a.view(np.uint64))
    np.testing.assert_array_equal(actual_b.view(np.uint64), (-expected_a).view(np.uint64))


def _is_level_rate(text, expr):
    """``L.<name>`` where ``<name>`` is a DOUBLE PRECISION field of the
    ThompsonAaLevel record the v4.6.1 source networks keep WRF's rates in."""
    if not expr.startswith("L."):
        return False
    start = text.find("struct ThompsonAaLevel {")
    if start < 0:
        return False
    body = text[start:text.find("};", start)]
    name = expr[2:]
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("double ") and name in [
                f.strip(" ;") for f in line[len("double "):].split(",")]:
            return True
    # fields continued on the next line of a multi-line declaration
    decl = " ".join(body.split())
    for chunk in decl.split(";"):
        chunk = chunk.strip()
        if chunk.startswith("double ") and name in [
                f.strip() for f in chunk[len("double "):].split(",")]:
            return True
    return False



def test_rate_readback_anchors_match_the_kernels():
    sys.path.insert(0, str(_TOOL))
    try:
        import port_rates
        from instrument_wrf_rates import RATES
    finally:
        sys.path.remove(str(_TOOL))
    from woof.core.kernels import module_source

    for mp, table in port_rates.ANCHORS_BY_MP.items():
        for module, anchors in table.items():
            text = module_source(module)
            instrumented = port_rates.instrument(module, text, mp)
            for _anchor, rate_map in anchors:
                for wrf_name, expr in rate_map.items():
                    assert (wrf_name in RATES
                            or wrf_name in port_rates.DIAGNOSTICS)
                    assert wrf_name not in port_rates.NOT_CARRIED[mp]
                    assert f"/* {wrf_name} */" in instrumented, (
                        mp, module, wrf_name)
                    # every rate expression is a local the kernel really
                    # declares
                    assert (f" {expr} = " in text or f" {expr};" in text
                            or f"double {expr}" in text
                            or _is_level_rate(text, expr)), (mp, module, expr)
            # nothing but the guarded readback was inserted: every added
            # line is a HOST_RATE call or its #ifdef/#endif guard
            added = (len(instrumented.splitlines())
                     - len(text.splitlines()))
            expected = sum(len(rate_map) + 2 for _a, rate_map in anchors)
            assert added == expected, (mp, module, added, expected)
            calls = (sum("HOST_RATE(" in line
                         for line in instrumented.splitlines())
                     - sum("HOST_RATE(" in line for line in text.splitlines()))
            assert calls == sum(len(m) for _a, m in anchors), (
                mp, module)
    # mp=8: the classic unit publishes every WRF rate from its own source,
    # in RATES order, the not-carried aerosol terms as zero.
    import re
    assert set(port_rates.ANCHORS_MP8) == {"thompson_classic"}
    text = module_source("thompson_classic")
    block = re.search(r"const double r_\[64\] = \{([^}]*)\};", text)
    assert block is not None
    names = [n.strip() for n in block.group(1).split(",")]
    want = ["0.0" if n in port_rates.NOT_CARRIED_MP8 else n for n in RATES]
    assert names == want
    # and the cp2 rates again after the rain evaporation
    for name in ("prw_vcd", "pnc_wcd", "prv_rev", "pnr_rev"):
        assert f"HOST_RATE({port_rates.SLOT[name]}, idx, {name});" in text


def test_the_fixture_covers_every_repaired_regime():
    assert _FIXTURE.exists(), _FIXTURE
    z = np.load(_FIXTURE)
    labels = set(str(label) for label in z["labels"])
    missing = [r for r in REQUIRED_REGIMES if r not in labels]
    assert not missing, missing
    receipt = str(z["wrf_build_receipt"])
    # The WRF source the answers came from, by content.
    assert ("fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802"
            in receipt), receipt


def _wsl_python():
    wsl = shutil.which("wsl.exe")
    if wsl is None:
        return None
    probe = subprocess.run(
        [wsl, "--exec", "bash", "-lc",
         "command -v g++ >/dev/null && python3 -c 'import numpy'"],
        capture_output=True, check=False)
    if probe.returncode != 0:
        return None

    def unix(path):
        done = subprocess.run([wsl, "--exec", "wslpath", "-a", str(path)],
                              capture_output=True, text=True, check=True)
        return done.stdout.strip()

    return wsl, unix


def _run_tool(tmp_path, script_name, *args, exact=False):
    """Run one of the harness scripts on the host backend and return its
    JSON: in process's own Python on a POSIX box with a C++ compiler, and
    through WSL on Windows."""
    script = _TOOL / script_name
    build = tmp_path / "host-kernels"
    if os.name == "nt":
        found = _wsl_python()
        if found is None:
            pytest.skip("no native C++ compiler on Windows and no WSL with "
                        "g++ and numpy to build the host kernels in")
        wsl, unix = found
        cmd = [wsl, "--exec", "env",
               f"WOOF_HOST_KERNEL_BUILD_DIR={unix(build)}",
               f"GPUWM_WRF_EXACT={int(exact)}",
               "python3", unix(script),
               *[unix(a) if isinstance(a, Path) else a for a in args]]
        env = None
    else:
        if shutil.which(os.environ.get("CXX", "g++")) is None:
            pytest.skip("no C++ compiler to build the host kernels")
        cmd = [sys.executable, str(script), *[str(a) for a in args]]
        env = dict(os.environ, WOOF_HOST_KERNEL_BUILD_DIR=str(build),
                   GPUWM_WRF_EXACT=str(int(exact)))
    done = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          check=False)
    if done.returncode != 0:
        tail = (done.stdout + done.stderr)[-3000:]
        lowered = tail.lower()
        if "freezeh2o" in lowered or ("table" in lowered and (
                "missing" in lowered or "not found" in lowered)):
            pytest.skip("the classic Thompson table set is not staged "
                        "(freezeH2O.dat is not packaged):\n" + tail)
        raise AssertionError(f"{script_name} failed:\n" + tail)
    return json.loads(done.stdout)


def _run_check(tmp_path, fixture=_FIXTURE):
    return _run_tool(tmp_path, "fixture_check.py", fixture)


def test_real_columns_against_wrf461(tmp_path):
    """Grade unchanged inputs against WRF with its eight invalid slab reads repaired."""
    assert _FIXTURE_CORRECTED_RACG.exists(), (
        "generate the independent corrected RACG reference with "
        "tools/thompson_real_column_parity/corrected_real_reference.py; "
        "the stock fixture and all numerical limits remain unchanged")
    result = _run_check(tmp_path, _FIXTURE_CORRECTED_RACG)

    lines = []
    for name, entry in result["rates"].items():
        if entry["n_far_unexplained"]:
            lines.append(f"rates.{name}: {entry['n_far_unexplained']} cells "
                         "beyond 1e-2 that rounding does not explain")
    for name, entry in result["final"].items():
        if "n_far_unexplained" not in entry:
            continue
        pinned = PINNED_FINAL_CELLS.get(name, 0)
        if entry["n_far_unexplained"] != pinned:
            lines.append(f"final.{name}: {entry['n_far_unexplained']} cells "
                         f"beyond 1e-2 unexplained, pinned {pinned}")
    for name in ("T_exit", "rainnc", "snownc", "graupelnc"):
        if result["final"][name]["rel_max"] > 1.0e-5:
            lines.append(f"final.{name}: relative "
                         f"{result['final'][name]['rel_max']:.3e}")
    if result["refl"]["max_abs_db"] > REFL_MAX_DB:
        lines.append(f"refl: {result['refl']['max_abs_db']:.3e} dB from WRF")
    assert not lines, "\n".join(lines)


@pytest.mark.parametrize("stock_path,corrected_path,mp", (
    (_FIXTURE, _FIXTURE_CORRECTED_RACG, 28),
    (_FIXTURE_MP8, _FIXTURE_MP8_CORRECTED_RACG, 8),
))
def test_corrected_racg_reference_retains_every_stock_input_word(
        stock_path, corrected_path, mp):
    """The oracle correction must not select easier columns or alter input bytes."""
    import hashlib
    from tools.thompson_real_column_parity.corrected_real_reference import (
        VARIANT, PINNED, OLD, NEW, input_words)

    assert corrected_path.exists(), corrected_path
    with np.load(stock_path, allow_pickle=False) as stock, np.load(
            corrected_path, allow_pickle=False) as corrected:
        assert int(corrected["mp_physics"]) == mp
        assert corrected["col_p"].shape[0] == 42
        assert str(corrected["reference_variant"]) == VARIANT
        assert str(corrected["stock_fixture_sha256"]) == hashlib.sha256(stock_path.read_bytes()).hexdigest()
        source = json.loads(str(corrected["source_repair_receipt"]))
        assert source["stock_source_sha256"] == PINNED
        assert source["replacement_count"] == 8
        assert source["only_replacements"] is True
        assert (source["old"], source["new"]) == (OLD, NEW)
        actual_inputs = input_words(corrected)
        assert actual_inputs == input_words(stock)
        assert actual_inputs == json.loads(str(corrected["input_words_receipt"]))
        assert "oracle_source_variant = corrected-racg" in str(corrected["wrf_build_receipt"])
        assert source["corrected_source_sha256"] in str(corrected["wrf_build_receipt"])


# ---------------------------------------------------------------------------
# mp_physics=8 on the same columns.
# ---------------------------------------------------------------------------

def _bitwise_report(tmp_path, fixture, *extra):
    return _run_tool(tmp_path, "mp8_wrf_ulp.py", fixture, "--backend",
                     "host", "--json-only", *extra, exact=True)


def _differing(report):
    return {name: (entry["mismatches"], entry["max_ulp"])
            for name, entry in report["fields"].items()
            if entry["mismatches"]}


#: WRF v4.6.1's module with its two classic-Thompson defects repaired
#: (tools/thompson_mp8_column_oracle/repair_wrf461.py: the rain-collecting-
#: graupel table slot and the graupel TAU+1 flag), the control the classic
#: unit is graded against word for word.  Same columns as _FIXTURE_MP8,
#: written by ``REPAIR=1 build_wrf.sh`` + ``make_fixture.py --mp8``.
_FIXTURE_MP8_REPAIRED = (_ROOT / "tests" / "data"
                         / "thompson_real_columns_wrf461_mp8_repaired.npz")
_WRF461_SHA = ("fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802")
_WRF461_REPAIRED_SHA = (
    "ccd74ab8a7296b1f58bf08b4c1c12ca44d7ea495860e210ee5ef9f901e8ab62d")


def test_real_columns_against_wrf461_mp8(tmp_path):
    """Every output word and every carried rate equals UNMODIFIED WRF
    v4.6.1's.  The classic unit is the WRF transcription (GPUWM_WRF_EXACT=1,
    exact mode) and keeps WRF's two v4.6.1 classic-Thompson defects, the
    rain-collecting-graupel read at idx_bg1 = 5 and the graupel TAU+1 flag
    (thompson_classic.cu); the forecast's default adapter keeps the repairs.
    Breakage prevented: an exact-mode Thompson that answers for a repaired
    WRF nobody runs, which moved rain wherever rain met graupel."""
    assert _FIXTURE_MP8_BOUND239.exists(), _FIXTURE_MP8_BOUND239
    z = np.load(_FIXTURE_MP8_BOUND239)
    z28 = np.load(_FIXTURE)
    # the same columns as the mp=28 fixture, and WRF's same source, unmodified
    assert int(z["mp_physics"]) == 8
    for key in ("col_p", "col_th", "col_qv", "col_qc", "col_qi"):
        assert np.array_equal(z[key], z28[key]), key
    receipt = str(z["wrf_build_receipt"])
    assert _WRF461_SHA in receipt
    report = _bitwise_report(tmp_path, _FIXTURE_MP8_BOUND239, "--rates")
    rates = [name for name in report["fields"] if name.startswith("rate_")]
    assert len(rates) == 58, len(rates)
    assert report["overall"]["values"] > 27048
    assert not _differing(report), _differing(report)
    assert report["overall"]["bitwise"]


_BOUND239 = _ROOT / "tests" / "data" / "oracles" / "thompson_bound239"
_FIXTURE_MP8_BOUND239 = _BOUND239 / "stock.npz"
_FIXTURE_MP8_REPAIRED_BOUND239 = _BOUND239 / "repaired.npz"


def test_mp8_differs_from_repaired_wrf461_only_where_the_repairs_act(
        tmp_path):
    """Against REPAIRED WRF v4.6.1 the unit differs in exactly the words the
    two repairs move, cell for cell.

    Breakage prevented: the two v4.6.1 defects the unit keeps acting on
    words they do not reach.  The two fixtures hold the same 42 columns
    through unmodified and repaired WRF; the words where the unit's answer
    differs from repaired WRF must be exactly the words where repaired WRF
    differs from unmodified WRF (585 on the input-bound fixture: the rain-collecting-
    graupel slot; the TAU+1 flag acts on none of these columns).
    """
    pristine = np.load(_FIXTURE_MP8_BOUND239)
    repaired = np.load(_FIXTURE_MP8_REPAIRED_BOUND239)
    assert _WRF461_SHA in str(pristine["wrf_build_receipt"])
    assert _WRF461_REPAIRED_SHA in str(repaired["wrf_build_receipt"])
    saved = tmp_path / "unit-words.npz"
    _bitwise_report(tmp_path, _FIXTURE_MP8_BOUND239, "--save", saved)
    got = np.load(saved)
    total = 0
    for name in got.files:
        want = np.asarray(pristine[f"wrf_{name}"], np.float32)
        fixed = np.asarray(repaired[f"wrf_{name}"], np.float32)
        if name.startswith("re_"):
            want = (want * np.float32(1.0e6)).astype(np.float32)
            fixed = (fixed * np.float32(1.0e6)).astype(np.float32)
        have = np.asarray(got[name], np.float32).reshape(want.shape)
        unit_moves = have.view(np.uint32) != fixed.view(np.uint32)
        repair_moves = fixed.view(np.uint32) != want.view(np.uint32)
        assert np.array_equal(unit_moves, repair_moves), name
        total += int(repair_moves.sum())
    assert total == 585, total


def test_mp8_rule_columns_are_wrfs_own_answers(tmp_path):
    """The named rule columns use captured scalar-239 wrapper inputs too.

    The older relative control called the default adapter while spying only
    on strict launchers, then indexed a nonexistent rain_evaporation stage.
    Grade this strict path against the pristine native answers word for word.
    The old corrected references remain untouched for the default controls.
    """
    report = _bitwise_report(tmp_path, _BOUND239 / "rules.npz", "--rates")
    assert report["overall"]["values"] == 6176
    assert not _differing(report), _differing(report)


def test_mp8_legacy_reference_has_a_different_wrapper_temperature():
    """A saved raw column is insufficient to bind a derived wrapper input."""
    old = np.load(_FIXTURE_MP8)
    bound = np.load(_FIXTURE_MP8_BOUND239)
    temperature = bound["wrapper_th"] * bound["wrapper_pii"]
    native_old = old["cp1_t1d"]
    valid = np.isfinite(native_old)
    different = valid & (temperature.view(np.uint32) != native_old.view(np.uint32))
    assert int(different.sum()) == 252
    for name in ("th", "p", "geop", "w", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr"):
        np.testing.assert_array_equal(bound["col_" + name], old["col_" + name])


# ---------------------------------------------------------------------------
# Named WRF rules on built columns (tools/thompson_real_column_parity/
# rule_checks.py), mp=28 through its production adapter.  mp=8's rule
# columns are graded against WRF's own answers above
# (test_mp8_rule_columns_are_wrfs_own_answers).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rules(tmp_path_factory):
    return _run_tool(tmp_path_factory.mktemp("rules"), "rule_checks.py")


@pytest.mark.parametrize("mp", (28,))
def test_a_column_wrf_leaves_at_its_no_micro_exit_is_returned_untouched(
        rules, mp):
    """module_mp_thompson.F:2020 and :3974.

    Breakage prevented: a column with no condensate above R1 and no ice
    supersaturation leaves mp_thompson before the source loop, so WRF
    neither floors its vapour at 1.E-10 nor clamps its aerosol; every other
    column leaves with every level's vapour at 1.E-10 or above.  The port
    applied neither rule: it clamped the aerosol of every column (up to 1.5
    percent of nwfa at one saved analysis cell) and left zero vapour at zero
    in columns with microphysics (1 to 71 levels per saved forecast frame,
    all 2,794 analysis levels of that kind) against WRF v4.6.1's own
    Fortran.  Column 0 is the no-micro column, column 1 the same with cloud.
    """
    got = rules[f"G_no_micro_column_mp{mp}"]
    assert got["qv_top"] == [0.0, pytest.approx(1.0e-10, rel=1e-7)]
    assert got["qc_orphan"] == [0.0, 0.0]
    assert got["qi_orphan"] == [0.0, 0.0]
    assert got["ni_orphan"] == [0.0, 0.0]
    if mp == 28:
        # untouched above the 9999.E6 ceiling, and clamped to it
        assert got["nwfa_max"][0] == pytest.approx(2.0e10, rel=1e-7)
        assert got["nwfa_max"][1] == pytest.approx(9999.0e6, rel=1e-7)


@pytest.mark.parametrize("mp", (28,))
def test_ice_above_0c_keeps_its_number_inside_the_size_bounds(rules, mp):
    """module_mp_thompson.F:3033-3055.

    Breakage prevented: WRF's cloud ice mass/number balance runs at every
    level of a column with microphysics, so ice at a level above 0 C (which
    advection and analyses both leave) reaches the fallout with a number
    inside the 5 to 300 micron bounds.  The port applied the balance only in
    its cold source network, which returns above 0 C, so ice there fell with
    no number (the fallout then gave it 5 micron crystals) or with whatever
    out-of-bounds number it arrived with: three ice-number cells of a saved
    analysis state differed from WRF v4.6.1's own Fortran beyond 1e-2, and
    the cloud the melt makes from that ice moved with them.  Level 5 holds
    ice with no number, level 4 ice with a number far above the 5 micron
    bound; both sit near 282 to 285 K.
    """
    got = rules[f"Awarm_ice_number_mp{mp}"]
    assert all(t > 273.15 for t in got["temperature"])
    assert got["ni_after_sources"] == pytest.approx(got["ni_balance"],
                                                    rel=1.0e-6)


@pytest.mark.parametrize("mp", (28,))
def test_cloud_and_ice_at_or_below_r1_melt_before_they_are_removed(
        rules, mp):
    """module_mp_thompson.F:3943-3966, then :4007-4009 and :4023-4027.

    Breakage prevented: WRF melts any positive ice at a warm level into
    cloud water, and only the terminal apply after that removes cloud and
    ice at or below R1, so a level whose cloud and melted ice are each at or
    below R1 but together above it keeps that cloud.  The port's cloud and
    ice fallout removed both before the melt (mp=28's ice fallout is the
    shared thompson.cu kernel): 0 to 2 cloud cells of 1.6e-12 to 2.0e-12
    kg/kg per saved forecast frame, and the droplet number on them, came
    back as zero against WRF v4.6.1's own Fortran.  Level 2 holds 0.8e-12
    kg/kg each of cloud and ice, level 3 0.4e-12 each, near 280 K.
    """
    got = rules[f"J_melt_before_r1_mp{mp}"]
    both = float(np.float32(0.8e-12) + np.float32(0.8e-12))
    assert got["qc"][2] == pytest.approx(both, rel=1.0e-6)
    assert got["qc"][3] == 0.0
    assert got["qi"] == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("mp", (28,))
def test_melting_snow_blends_with_the_rain_pass_s_own_fall_speed(rules, mp):
    """module_mp_thompson.F:3612-3634 and :3722-3724.

    Breakage prevented: melting snow falls at vts*SR + (1-SR)*vtrk(k) with
    SR = rs/(rs+rr), whatever the level's own rain: a level without rain has
    rr = R1 and the rain speed of the level above, and a column without rain
    has vtrk = 0.  The port blended only where the level itself held rain, at
    a rain speed and rr formed from the snow's density, so melting snow
    without rain fell unblended and the snow a melting layer leaves near R1
    differed from WRF v4.6.1's own Fortran (1 to 15 cells per saved forecast
    frame beyond 1e-2, up to 2.3e-9 kg/kg on the analysis states through the
    snow substep count).  Before the repair the two melting arms removed
    exactly what the non-melting one did; after it they remove what the rule
    says: SR = 0.84 of it with no rain, and 34 times it under rain aloft.
    """
    got = rules[f"I_melting_snow_blend_mp{mp}"]
    assert 0.5 < got["solid_fraction"] < 0.95
    assert got["vtrk"] > 10.0 * got["vts"]
    assert got["removed"]["dry"] == pytest.approx(got["rule"]["dry"],
                                                  rel=1.0e-3)
    assert got["removed"]["rain_above"] == pytest.approx(
        got["rule"]["rain_above"], rel=1.0e-3)


@pytest.mark.parametrize("mp", (28,))
def test_a_species_is_present_where_its_mixing_ratio_passes_r1(rules, mp):
    """module_mp_thompson.F:1827-1949 with :2783-2825.

    Breakage prevented: WRF's L_qc, L_qs, L_qg and L_qr are the MIXING
    RATIO tests of the entry block, and every process gated on them runs
    wherever they hold, with rs(k), rg(k) and rc(k) whatever q*rho is.  The
    classic source networks gated on the concentration instead, so in thin
    air, where a mixing ratio just above R1 is a concentration at or below
    it, snow and graupel did not melt or sublimate, rain did not collect
    cloud and the conservation limits skipped them: 454 snow-melt, 547
    graupel-melt and 209 rain-collects-cloud cells per five saved forecast
    frames differed from WRF v4.6.1's own Fortran beyond 1e-2, and 691 snow
    cells of the final state.  Snow of 1.1e-12 kg/kg at a level near 276 K
    where rho is about 0.8 melts in WRF and in both ports now; the classic
    port kept it.
    """
    got = rules[f"P_presence_mp{mp}"]
    assert got["concentration"][0] <= 1.0e-12 < 1.1e-12
    assert all(t > 273.15 for t in got["temperature"])
    assert got["qs"] == 0.0
    assert got["qg"] == 0.0


def test_the_classic_rain_evaporation_hands_the_fallout_wrfs_l_qr(rules):
    """module_mp_thompson.F:3236, :3252-3253 and :3501-3572.

    Breakage prevented: WRF's rain fallout forms rr(k) only where L_qr holds
    (:3236, the post-source mixing ratio), from the TAU+1 density there, and
    from the post-condensation density, floored at R1, where the rain
    evaporation rewrote it (:3568).  The classic rain fallout guessed L_qr
    from the post-evaporation mixing ratio and could not tell the floored
    rewrite from the held pair, so the rain number of saved real-data frames
    differed from WRF v4.6.1's own Fortran at thousands of levels beyond
    2e-6 and the echo by up to 13.8 dB.  The classic rain evaporation now
    marks the three cases in the fallout's density: zero, the held density,
    the negative of its own.
    """
    got = rules["R_classic_rain_presence"]
    assert got["density"][0] == 0.0
    assert got["density"][1] == pytest.approx(-got["own"][1], rel=1.0e-7)
    assert got["qr"][1] < 1.0e-4
    assert got["density"][2] == got["held"][2]
    assert got["qr"][2] == pytest.approx(1.0e-4, rel=1.0e-7)


def test_graupel_at_or_below_r1_leaves_the_call_as_zero(rules):
    """module_mp_thompson.F:4058-4063.

    Breakage prevented: WRF's terminal apply writes graupel at or below R1
    as zero, mass and number, in every column.  The port zeroed it only in
    the columns whose graupel fallout ran, so a residue the sources left in
    any other column stayed in the state, the history and every between-step
    reader (188,969 cells of one saved 19,600-column history frame carried
    0 < qg <= 1e-12).  mp=8 and mp=28 share the kernel.
    """
    got = rules["T_terminal_graupel_zero"]
    above = float(np.nextafter(np.float32(1.0e-12), np.float32(1.0)))
    assert got["qg"][:5] == [0.0, 0.0, above, pytest.approx(2.0e-12), 0.0]
    assert got["qg"][5] == pytest.approx(1.0e-4)
    assert got["ng_zero"] == [True, True, False, False, True, False]
