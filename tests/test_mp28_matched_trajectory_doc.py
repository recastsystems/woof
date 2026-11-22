"""Every number ``mp28-matched-trajectory.md`` publishes is read back.

WHY THIS FILE EXISTS.  The matched-trajectory document was written by hand
from comparison receipts that lived only on the run node.  Four of its
published numbers did not match those receipts:

* it said the t = 0 field difference was "exactly 0.0 for every field" when
  the receipt records ``T`` at 2.5e-09 and ``QNWFA``/``QNIFA`` at 2e-08;
* it counted V3 failures for WOOF-ArWen over the 111 scalar rows ("7 of 111")
  and for the control over all 195 -- two denominators in one table row;
* it published the domain aerosol budget agreement as 1.6e-04 when the
  series give 1.530e-04;
* it claimed a dual-run byte screen on the short window that the
  short-window tool did not perform.

None of them changed a conclusion, which is exactly why none of them would
have been noticed.  A document that republishes a measurement drifts from it
unless something compares the two, so this compares the two.  The receipts
are committed beside the document (``docs/public/receipts/``) precisely so
that the comparison is possible in a checkout, on any machine, with no GPU
and no run directory.

This gate is one-directional in the same way the registry/PHYSICS.md gate
is: the page may say MORE than the receipts, but every number it does
publish must be the receipt's.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "public" / "wrf-comparison" / "mp28-matched-trajectory.md"
PHYSICS_MD = ROOT / "docs" / "public" / "PHYSICS.md"
RECEIPTS = ROOT / "docs" / "public" / "receipts" / "mp28-matched-trajectory"

ARWEN = RECEIPTS / "comparison-arwen-vs-wrf.json"
CONTROL = RECEIPTS / "comparison-control-wrf-vs-wrf.json"
SHORT = RECEIPTS / "shortwindow.json"


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _scalar_rows(report: dict) -> list[dict]:
    """The M1-M4 rows.  M8 rows are named ``M8:<FIELD>``."""
    return [r for r in report["V3"]["rows"]
            if not str(r["metric"]).startswith("M8:")]


def _m8_rows(report: dict) -> list[dict]:
    return [r for r in report["V3"]["rows"]
            if str(r["metric"]).startswith("M8:")]


# --- the receipts are real, and are the ones the runs wrote ----------------

def test_the_receipts_exist_and_describe_the_case_this_document_describes():
    """Grounding.  Without this the rest could pass against empty files."""
    for path in (ARWEN, CONTROL, SHORT):
        assert path.is_file(), path

    arwen, control = _load(ARWEN), _load(CONTROL)
    assert arwen["tested_model"] == "arwen"
    assert control["tested_model"] == "novec", (
        "the control must put a SECOND WRF BUILD in the tested slot; a "
        "control that tests WOOF-ArWen again measures nothing")
    assert arwen["wrf_build"] == control["wrf_build"] == "vec", (
        "both comparisons must be read against the same reference build")

    # 13 history frames over 7200 s, the case as designed.
    assert len(arwen["times"]) == 13, arwen["times"]
    assert arwen["times"][0] == 0.0 and arwen["times"][-1] == 7200.0

    # And the runs really were ArWen on the device the document names.
    run = _load(RECEIPTS / "arwen-mp28-a-run.json")
    assert run["model"] == "woof" and run["mp_physics"] == 28
    assert (run["nx"], run["ny"], run["nz"]) == (120, 120, 40)
    assert run["dx"] == 2000.0 and run["steps"] == 600
    assert "5090" in run["device"], run["device"]
    assert run["wrfinput_attrs"]["TITLE"].strip().startswith(
        "OUTPUT FROM IDEAL"), (
        "the initial condition must come from WRF's own ideal.exe, which is "
        "the whole basis of section 2.1")


def test_the_declared_verdict_letters_are_the_receipts_verdict_letters():
    """§9's table, against the receipts' own pass flags."""
    arwen, control = _load(ARWEN), _load(CONTROL)
    assert arwen["verdict"] == {"V1": True, "V2": True, "V3": False,
                                "V4": True, "ship": False}
    assert control["verdict"] == {"V1": True, "V2": True, "V3": False,
                                  "V4": True, "ship": False}

    page = _doc()
    assert "**By the rule declared in §6, before any run: HOLD. V3 failed.**" \
        in page, (
            "V3 is false in the receipt, so the document must still publish "
            "the declared verdict as a HOLD; a passing receipt is the only "
            "thing that may change this sentence")


# --- the published numbers -------------------------------------------------

def test_the_v1_and_v3_counts_are_the_receipts_counts():
    arwen, control = _load(ARWEN), _load(CONTROL)
    page = _doc()

    v1a, v1c = arwen["V1"], control["V1"]
    assert f"{v1a['sign_agree']}/{v1a['resolvable_pairs']}" in page
    assert f"{v1a['fraction'] * 100:.1f}%" in page
    assert f"{v1c['sign_agree']}/{v1c['resolvable_pairs']}" in page
    assert f"{v1c['fraction'] * 100:.1f}%" in page

    # V3 as DECLARED covers M1-M4 and M8, so the denominator is every row.
    # The two columns of one table must be counted the same way -- that was
    # the defect: 7 of 111 (scalar only) beside 17 of 195 (all rows).
    for report in (arwen, control):
        rows = report["V3"]["rows"]
        failing = [r for r in rows if not r["pass"]]
        assert f"{len(failing)} of {len(rows)} rows over 3×" in page, (
            f"V3 must be published as {len(failing)} of {len(rows)}")

    assert f"{arwen['V3']['worst']:.2e}".replace("e+0", "e+0") in page \
        or "4.95e+04" in page
    assert "8.08e+02" in page


def test_the_v3_distribution_statistics_are_recomputed_from_the_receipts():
    """The medians and the <=3 counts, on both denominators, both sides."""
    page = _doc()
    for report in (_load(ARWEN), _load(CONTROL)):
        for rows in (report["V3"]["rows"], _scalar_rows(report)):
            ratios = [r["ratio"] for r in rows]
            median = statistics.median(ratios)
            below = sum(1 for x in ratios if x <= 3.0)
            assert f"{median:.3f}" in page, (
                f"median {median:.3f} over {len(rows)} rows is not published")
            assert f"{below} of {len(rows)}" in page, (
                f"'{below} of {len(rows)}' (ratio <= 3) is not published")

    # And the claim that no M8 row fails, which is what makes the scalar
    # subset the interesting one.
    assert all(r["pass"] for r in _m8_rows(_load(ARWEN))), (
        "an M8 row now fails V3; the document says none does")


def test_the_t0_field_differences_are_published_as_measured():
    """The claim that was wrong, pinned in both directions."""
    page = _doc()
    t0 = _load(ARWEN)["t0_field_rms"]

    exact = [f for f, v in t0["mp28"].items() if v == 0.0]
    nonzero = {f: v for f, v in t0["mp28"].items() if v != 0.0}
    assert len(exact) == 10 and len(nonzero) == 3, (exact, nonzero)
    assert set(nonzero) == {"T", "QNWFA", "QNIFA"}, nonzero

    for field, value in nonzero.items():
        assert f"{value:.4e}" in page, (
            f"{field}'s t=0 residual {value:.4e} is not published")

    # The superseded claim must not come back.  Its exact words, because it
    # is a claim about exactness.
    for banned in ("exactly 0.0 for every field",
                   "**exactly 0.0 for every field in both configurations**"):
        assert banned not in page, banned
    assert "RMS difference **exactly 0** in every field" not in page

    # mp=8 has no aerosol fields at all, and T is the same rounding.
    assert t0["mp08"]["T"] == t0["mp28"]["T"]
    assert "QNWFA" not in t0["mp08"] and "QNIFA" not in t0["mp08"]


def test_the_m8_table_publishes_every_history_time_not_a_selection():
    """Ten of thirteen rows were printed, and the three missing ones were
    the three least favourable.  Uncurated means all of them."""
    page = _doc()
    start = page.index("M8, the normalised RMS field difference")
    end = page.index("Read this as a perturbation-growth experiment")
    table = page[start:end]

    times = _load(ARWEN)["times"]
    printed = {int(m) for m in re.findall(r"^\| (\d+) \|", table, re.M)}
    assert printed == {int(t) for t in times}, (
        f"the M8 table prints {sorted(printed)} but the receipt has "
        f"{[int(t) for t in times]}")

    # Spot the values themselves, at the frame the text calls out and at the
    # worst one, so a right-shaped table of wrong numbers still fails.
    m8 = _load(ARWEN)["m8_divergence"]
    for time_s in ("600.0", "6000.0", "7200.0"):
        for config in ("mp08", "mp28"):
            assert f"{m8[config][time_s]['W']:.4f}" in table, (
                f"{config} W at t={time_s} is not in the M8 table")


def test_the_aerosol_budget_agreement_is_the_series_own_number():
    """1.6e-04 was published; the two series give 1.530e-04.

    Recomputed from BOTH models' committed series rather than from the
    comparison receipt, because the comparison receipt only carries the
    tested model's aerosol budget -- WRF's end value had no artifact in the
    tree at all, which is how a typed number survived.
    """
    page = _doc()
    wrf = _load(RECEIPTS / "series-wrf-vec-mp28-x.json")
    arwen = _load(RECEIPTS / "series-arwen-mp28-a.json")

    assert wrf[0]["time_s"] == arwen[0]["time_s"] == 0.0
    assert wrf[-1]["time_s"] == arwen[-1]["time_s"] == 7200.0

    for field, digits in (("nwfa_mean", 2), ("nifa_mean", 2)):
        relative = abs(arwen[-1][field] - wrf[-1][field]) / wrf[-1][field]
        assert f"{relative:.3e}" in page, (
            f"{field} agreement must be published as {relative:.3e}")

    # The endpoints themselves, so the ratio can be re-derived by hand.
    for row, model in ((wrf, "WRF"), (arwen, "WOOF-ArWen")):
        for frame in (row[0], row[-1]):
            assert f"{frame['nwfa_mean']:.2f}" in page, (
                f"{model} nwfa_mean {frame['nwfa_mean']:.2f} is not published")

    assert "1.6e-04" not in page, "the superseded 1.6e-04 figure is back"
    # PHYSICS.md republishes it for a reader who never opens this page.
    relative = (abs(arwen[-1]["nwfa_mean"] - wrf[-1]["nwfa_mean"])
                / wrf[-1]["nwfa_mean"])
    physics = PHYSICS_MD.read_text(encoding="utf-8")
    assert f"{relative:.3e}" in physics and "1.6e-04" not in physics

    # Both models trend UP, which is the claim that makes this diagnostic:
    # with periodic boundaries a downward trend would be a conservation
    # defect, not a boundary artifact.
    assert arwen[-1]["nwfa_mean"] > arwen[0]["nwfa_mean"]
    assert wrf[-1]["nwfa_mean"] > wrf[0]["nwfa_mean"]

    # V4's substance, not just its letter.
    v4 = _load(ARWEN)["V4"]
    assert v4["nonfinite"] == [] and v4["bound_violations"] == []
    assert not v4["nwfa_monotone_depletion"]
    assert not v4["nifa_monotone_depletion"]
    assert v4["nwfa_mean"][-1] == pytest.approx(arwen[-1]["nwfa_mean"])


def test_the_short_window_numbers_and_its_ordering_are_the_receipts():
    page = _doc()
    short = _load(SHORT)
    rows = sorted(short["V3_short_window"]["rows"],
                  key=lambda r: -r["ratio"])

    assert short["V3_short_window"]["pass"] is True
    assert f"111 rows, worst ratio **{rows[0]['ratio']:.3f}**" in page
    assert f"(`{rows[0]['field']}` at step {rows[0]['step']})" in page or (
        f"`{rows[0]['field']}`\nat step {rows[0]['step']}" in page)

    # The runner-up, because "every other field is at or below 1.3" was the
    # claim and 1.501 was the number.
    assert f"**{rows[1]['ratio']:.3f}**" in page, (
        f"the second-worst short-window ratio {rows[1]['ratio']:.3f} is not "
        "published, and the claim about the rest of the distribution cannot "
        "be checked without it")
    assert "at or below 1.3" not in page, (
        f"contradicted by the runner-up at {rows[1]['ratio']:.3f}")

    # The window is post-hoc and must keep saying so.
    assert short["start_s"] == 1800.0
    assert "post-hoc" in page


def test_the_dual_run_byte_screen_was_performed_on_both_windows():
    """No ECC on this card: a single run is not a result.

    The long run's screen lives in the comparison receipt and the short
    window's in the short-window receipt.  The document claimed both; only
    one was ever computed.
    """
    page = _doc()
    long_run = _load(ARWEN)["dual_run"]
    short_run = _load(SHORT)["dual_run"]

    for config in ("mp08", "mp28"):
        assert long_run[config]["identical"] is True
        assert long_run[config]["differing"] == []
        assert long_run[config]["frames_compared"] == 13

        assert short_run[config]["status"] == "ok", (
            "the short-window dual-run screen did not run; the document "
            "must not claim a screen the tool skipped")
        assert short_run[config]["identical"] is True
        assert short_run[config]["differing"] == []
        assert short_run[config]["frames_compared"] == 11

    assert "13/13 frames" in page and "11/11 frames" in page


def test_the_document_does_not_upgrade_the_scheme_on_this_evidence():
    """The receipt says ship=false.  The page may recommend; it may not
    relabel, and no maturity tier may be invented to hold it."""
    page = _doc()
    assert _load(ARWEN)["verdict"]["ship"] is False

    assert "implemented-unverified" in page
    for overclaim in ("model-validated", "validation-candidate",
                      "wrf-matched-run` on `mp_physics = 28",
                      "wrf-matched-run-candidate` on `mp_physics = 28",
                      "matched-forecast", "forecast-validated"):
        if overclaim in ("model-validated", "validation-candidate"):
            assert f"mp=28 {overclaim}" not in page, overclaim
        else:
            assert overclaim not in page, overclaim

    registry = json.loads(
        (ROOT / "woof" / "physics_registry_v2.json").read_text(
            encoding="utf-8"))
    option = registry["components"]["microphysics"]["options"][
        "thompson-aerosol-mp28"]
    assert option["maturity"] == "implemented-unverified", option["maturity"]


@pytest.mark.parametrize("name", [
    "comparison-arwen-vs-wrf.json",
    "comparison-control-wrf-vs-wrf.json",
    "shortwindow.json",
    "arwen-mp08-a-run.json",
    "arwen-mp08-b-run.json",
    "arwen-mp28-a-run.json",
    "arwen-mp28-b-run.json",
    "series-arwen-mp08-a.json",
    "series-arwen-mp28-a.json",
    "series-wrf-vec-mp08-x.json",
    "series-wrf-vec-mp28-x.json",
])
def test_every_committed_receipt_is_parseable_and_non_empty(name):
    """A receipt that fails to parse would make every gate above vacuous."""
    payload = _load(RECEIPTS / name)
    assert isinstance(payload, (dict, list)) and payload


#: Everything committed beside the document, and what reads it.  A stray
#: file here would look like evidence and be checked by nothing.
_RECEIPT_INVENTORY = {
    # measurements, read by the gates above
    "comparison-arwen-vs-wrf.json", "comparison-control-wrf-vs-wrf.json",
    "shortwindow.json", "SHA256SUMS.txt",
    "arwen-mp08-a-run.json", "arwen-mp08-b-run.json",
    "arwen-mp28-a-run.json", "arwen-mp28-b-run.json",
    "series-arwen-mp08-a.json", "series-arwen-mp28-a.json",
    "series-wrf-vec-mp08-x.json", "series-wrf-vec-mp28-x.json",
    # the case itself, read by the gate below
    "ic-mp08-namelist.input", "ic-mp28-namelist.input",
    "wrf-vec-mp08-namelist.input", "wrf-vec-mp28-namelist.input",
    "wrf-novec-mp08-namelist.input", "wrf-novec-mp28-namelist.input",
    "sw-wrf-mp28-namelist.input", "input_sounding",
    "configure-vec.txt", "configure-novec.txt",
    # the recipe, so the runs are re-derivable without the run node
    "build-wrf-vec.sh", "build-wrf-novec.sh", "make-namelists.sh",
    "stage-runs.sh", "extract-wrfout.sh", "run-arwen.sh", "run-analyze.sh",
    "run-shortwindow.sh",
}


def test_the_receipt_directory_holds_nothing_the_gates_do_not_read():
    present = {p.name for p in RECEIPTS.iterdir() if p.is_file()}
    assert present == _RECEIPT_INVENTORY, (
        f"unread: {sorted(present - _RECEIPT_INVENTORY)}; "
        f"missing: {sorted(_RECEIPT_INVENTORY - present)}")


def test_the_committed_namelists_are_the_case_the_document_describes():
    """§2's table, against the namelists the runs actually consumed.

    The runs are on a rented node that will not outlive this lane.  What
    makes the case reproducible afterwards is these files, so they are
    checked against the prose rather than merely stored next to it.
    """
    page = _doc()
    for name in ("ic-mp08-namelist.input", "ic-mp28-namelist.input",
                 "wrf-vec-mp08-namelist.input", "wrf-vec-mp28-namelist.input",
                 "wrf-novec-mp08-namelist.input",
                 "wrf-novec-mp28-namelist.input"):
        text = (RECEIPTS / name).read_text(encoding="utf-8")
        expected_mp = 28 if "mp28" in name else 8
        checks = {
            "max_dom": "1", "e_we": "121", "e_sn": "121", "e_vert": "41",
            "dx": "2000", "dy": "2000", "time_step": "12",
            "time_step_sound": "6", "mp_physics": str(expected_mp),
            "periodic_x": ".true.", "periodic_y": ".true.",
        }
        for key, want in checks.items():
            found = re.search(rf"^\s*{key}\s*=\s*([^,\n]+)", text, re.M)
            assert found, f"{name} has no {key}"
            assert found.group(1).strip() == want, (
                f"{name}: {key} = {found.group(1).strip()}, not {want}")
        # Every other physics option off -- microphysics is the only one.
        for off in ("ra_lw_physics", "ra_sw_physics", "sf_sfclay_physics",
                    "sf_surface_physics", "bl_pbl_physics", "cu_physics"):
            found = re.search(rf"^\s*{off}\s*=\s*([^,\n]+)", text, re.M)
            assert found and found.group(1).strip() == "0", (
                f"{name}: {off} is not 0; the document says microphysics is "
                "the only physics")

    # 120 x 120 x 40 mass points is 121 x 121 x 41 staggered, which is what
    # the namelists say and what the document's table must say.
    assert "| grid | 120 × 120 × 40 |" in page
    assert "| dx = dy | 2000 m |" in page

    # The two builds differ in exactly one flag and nothing else.
    vec = (RECEIPTS / "configure-vec.txt").read_text(encoding="utf-8")
    novec = (RECEIPTS / "configure-novec.txt").read_text(encoding="utf-8")
    assert "-ftree-vectorize" in vec and "-fno-tree-vectorize" not in vec
    assert "-fno-tree-vectorize" in novec
    assert vec != novec

    # The sounding is unsheared, which is the design's essential choice.
    rows = [line.split() for line in
            (RECEIPTS / "input_sounding").read_text(
                encoding="utf-8").splitlines()[1:] if line.strip()]
    winds = {(float(row[3]), float(row[4])) for row in rows if len(row) >= 5}
    assert winds == {(0.0, 0.0)}, (
        f"the sounding carries wind {sorted(winds)[:4]}; §2 says u = v = 0 "
        "and the whole no-self-interaction argument rests on it")
