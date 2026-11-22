# SPDX-License-Identifier: Apache-2.0
"""The door binds a packed plan's library folder to this install and refuses, before any download, an install that does
not hold the libraries the example plan replays byte for byte. No card."""
import hashlib
import os
from pathlib import Path

import pytest

from woof.hex import ad_doors as d


def _plan(tmp_path: Path, text: str) -> Path:
    plan = tmp_path / "plan"
    plan.mkdir()
    (plan / "plan.json").write_text(text)
    return plan


def test_the_plan_library_folder_is_linked_to_this_install(tmp_path):
    plan = _plan(tmp_path, '{"library_path": "<site-packages>/nvidia/cu13/lib/libcublas.so.13"}')
    site = tmp_path / "site"
    site.mkdir()
    d.bind_plan_libraries(plan, site)
    link = plan / d.PLAN_SITE_PACKAGES
    assert link.is_symlink() and Path(os.readlink(link)) == site
    other = tmp_path / "other"
    other.mkdir()
    d.bind_plan_libraries(plan, other)   # another install: the link follows it
    assert Path(os.readlink(link)) == other
    d.bind_plan_libraries(plan, other)   # and a second call changes nothing
    assert Path(os.readlink(link)) == other


def test_a_missing_or_different_library_refuses_before_any_download(tmp_path):
    site = tmp_path / "site"
    (site / "nvidia").mkdir(parents=True)
    lib = site / "nvidia" / "libx.so"
    lib.write_bytes(b"the recorded library")
    row = {"example_plan": "plans/p"}
    table = {"example_plan_libraries": {"plans/p": [{"path": "nvidia/libx.so", "from": "library X 1.0",
                                                    "sha256": hashlib.sha256(b"the recorded library").hexdigest()}]},
             "example_plan_install": {"plans/p": "Install X 1.0."}}
    d.refuse_plan_libraries_missing(table, "r", row, site)   # equal bytes: passes
    lib.write_bytes(b"another build")
    with pytest.raises(d.AdDoorError, match="is another build here.*Install X 1.0. Nothing was downloaded"):
        d.refuse_plan_libraries_missing(table, "r", row, site)
    lib.unlink()
    with pytest.raises(d.AdDoorError, match="is not in this install"):
        d.refuse_plan_libraries_missing(table, "r", row, site)
    d.refuse_plan_libraries_missing(table, "r", {"example_plan": "plans/other"}, site)   # a plan with no list: not judged


def test_the_shipped_table_names_the_example_plans_libraries_and_the_default_row_holds_the_drag_closure():
    table = d.load_table()
    row = table["rows"][table["default_row"]]
    libraries = d.plan_libraries(table, row)
    assert [x["path"].split("/")[0] for x in libraries] == ["nvidia", "cupy", "cupy"]
    assert all(len(x["sha256"]) == 64 for x in libraries)
    assert "3.14" in table["example_plan_install"][row["example_plan"]]
    assert row["assets"]["bundle"] == "latest/bundle/genB9-gfrz" and "mountain-drag closure" in row["what"]
