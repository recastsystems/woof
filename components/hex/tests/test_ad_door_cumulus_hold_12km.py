# SPDX-License-Identifier: Apache-2.0
"""The 12 km row holds the cumulus derivative by name (0.6.0): its kinds join the row's refuse list, and only there."""
import json
from pathlib import Path

from woof.hex import ad_doors

ROW12 = "hrrrv5-q12.96.2890.n40.60w91.30"


def _table():
    return json.loads((Path(ad_doors.__file__).parent / "data" / "ad-doors.v1.json").read_text(encoding="utf-8"))


def test_the_12km_row_holds_the_26_cumulus_kinds_and_no_other_row_does():
    table = _table()
    hold = table["rows"][ROW12]["derivative_holds"]["cumulus"]
    assert len(hold["kinds"]) == 26 and len(set(hold["kinds"])) == 26
    assert "ntiedtke_cuascn" in hold["kinds"] and "rrfs_convection_forcing" in hold["kinds"]
    assert "not in this release" in hold["what"] and "forecast itself is unchanged" in hold["what"]
    others = [n for n, r in table["rows"].items() if n != ROW12 and r.get("derivative_holds")]
    assert others == []


def test_the_held_kinds_join_the_rows_refuse_file(tmp_path):
    table = _table()
    row = dict(table["rows"][ROW12])
    refuse = tmp_path / "refuse.txt"
    refuse.write_text("rrtmg_a,rrtmg_b\n", encoding="utf-8")
    assets = {k: tmp_path for k in set(row["asset_args"].values()) | set(row["asset_env"].values())}
    assets["refuse"] = refuse
    door = "adjoint-test" if "adjoint-test" in table["doors"] else next(iter(table["doors"]))
    argv, _ = ad_doors.window_command(table, row, assets, door, tmp_path / "ad-binder", 2, tmp_path / "out", [])
    assert argv.count("--refuse") == 1
    names = argv[argv.index("--refuse") + 1].split(",")
    assert names[:2] == ["rrtmg_a", "rrtmg_b"] and len(names) == 28
    assert set(row["derivative_holds"]["cumulus"]["kinds"]) <= set(names)
