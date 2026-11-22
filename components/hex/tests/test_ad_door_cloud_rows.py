"""Which row runs which cloud derivative files (processors only).

THE BREAKAGE THIS PREVENTS: the README names, row by row, the cloud derivative files a gradient runs, and says why the
global rows keep their own. For two builds the table and the README said the repaired gated images "have not yet run a
window" on a globe and that "a second storm day" was not measured, after both had been measured (a 96 km globe over 8
and 16 steps, and the example's own day over 12 steps). A reader choosing a row was told less than was known, and a
row that gained or lost the option would have left the README's table wrong with nothing failing.
"""
import json
from pathlib import Path

from woof.hex import ad_doors as d

TREE = Path(__file__).resolve().parents[1]
STORM = "hrrrv5-q3.96.38278.n40.60w91.30"
GATED = ("c2chain-0xc6-rd3600n100.cubin", "tlfd-0xc6-rd3600n100.cubin", "split2-0xc6-rd3600n100.cubin")


def _cloud(row):
    return (row.get("gradient_line") or {}).get("cloud_derivative") or {}


def test_only_the_3_km_storm_row_offers_the_gated_cloud_files():
    rows = d.load_table()["rows"]
    offering = sorted(name for name, row in rows.items() if _cloud(row))
    assert offering == [STORM]
    gated = _cloud(rows[STORM])["routes"]["gated"]
    assert _cloud(rows[STORM])["default"] == "number"      # 0.6.0: rain and droplet number follow; gated a word
    assert tuple(Path(a["path"]).name for a in gated["assets"].values()) == (GATED[0], GATED[1], GATED[2])
    for name, row in rows.items():
        if name != STORM:                      # every other row runs the kernel set's own cloud images
            assert "cloud-gated" not in json.dumps(row.get("assets") or {}), name


def test_the_table_states_the_second_day_and_the_globe_as_measured():
    cloud = _cloud(d.load_table()["rows"][STORM])
    measured = cloud["routes"]["gated"]["measured"]
    before = cloud["routes"]["gated"]["default_when_before"]     # gated's own reasons, kept under its route
    assert "have not yet run a window there" not in before
    for figure in ("5.2e-05", "1.5e-08", "9.8e-09", "1.6e-07", "2.9e-09", "2.7e-08"):   # the 96 km readings
        assert figure in before, figure
    day2 = measured["second_day"]
    assert "2023-03-31" in day2["what"]
    near = day2["storm_rows_drawn_on_this_day"] + day2["rows_inside_or_within_20_km_of_the_target"]
    assert len(near) == 14 and all(abs(r - 1.0) <= 0.025 for r in near)
    assert "-0.105" in day2["rows_over_170_km_from_the_centre"]          # the two far rows that miss stay stated
    assert "healthy directions" in day2["shipped"] and "first day only" in day2["shipped"]
    assert "a second storm day;" not in measured["not_yet"]
    assert "cloud or rain target on a second storm day" in measured["not_yet"]


def test_the_readme_table_of_rows_is_the_door_table():
    text = (TREE / "README.md").read_text(encoding="utf-8")
    assert "**Which rows run which cloud files.**" in text
    lines = [ln for ln in text.splitlines() if ln.startswith("| the ") and "cloud" in ln]
    by_row = {ln.split("|")[1].strip(): ln for ln in lines}
    assert set(by_row) == {"the 3 km storm box", "the 12 km cut of the storm box",
                           "the 120 km global row and any row made with `--mesh`"}
    assert "`number` (default)" in by_row["the 3 km storm box"] and "--cloud-derivative shipped" in by_row["the 3 km storm box"]
    assert "--cloud-derivative gated" in by_row["the 3 km storm box"]
    assert _cloud(d.load_table()["rows"][STORM])["routes"]["number"]["assets_pinned_in"] in by_row["the 3 km storm box"]
    gated = _cloud(d.load_table()["rows"][STORM])["routes"]["gated"]
    assert gated["assets_pinned_in"] in by_row["the 3 km storm box"]      # the pack the table pins, by name
    for other in ("the 12 km cut of the storm box", "the 120 km global row and any row made with `--mesh`"):
        assert "no `--cloud-derivative` option" in by_row[other]
    assert "a second storm day, and the storm direction with the repaired" not in text
    for figure in ("0.995 to 1.004", "0.986 to 1.025", "-0.105 and 1.255", "1.6e-07", "2.7e-08"):
        assert figure in text, figure
