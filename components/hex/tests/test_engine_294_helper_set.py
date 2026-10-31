"""woof 2.9.4 radiation helpers are an admitted set (0.4.2 floor), without relaxing mixed-set refusal."""
import json
from pathlib import Path

from woof.hex import cuda_arwen_helper_scope_v2 as scope
from woof.hex import engine_pin


def test_294_source_set_is_admitted_on_the_ordinary_route():
    # THE BREAKAGE: the 2.9.4 slimming rewrote five of the six helpers, so with the floor at 2.9.4
    # every default-physics forecast refused with "unknown or mixed woof execution source set".
    mode = "engine-2.9.4-source"
    assert set(scope.ADMITTED_SOURCE_SETS[mode]) == set(scope.PUBLISHED_SOURCES)
    assert mode in scope.PUBLISHED_MODES
    rows = json.loads(Path(scope.__file__).with_name("data").joinpath(
        "profiles", "engine-helper-sets.v1.json").read_text())["sets"]
    row = next(row for row in rows if row["id"] == mode)
    assert row["engine_commit"] == "2147c15faf63511b673d22d12c5656e3270c5176"
    assert all(len(digest) == 64 for digest in row["sources"].values())
    changed = {name for name, digest in row["sources"].items()
               if digest != scope.ADMITTED_SOURCE_SETS["engine-2.9.3-high-top-buffer-source"][name]}
    assert changed == {"woof/core/rrtmg_lw.py", "woof/core/rrtmg_sw.py", "woof/core/rrtmg_mcica.py",
                       "woof/core/rrtmg_legacy_prep.py", "woof/ingest/wrf_ozone.py"}


#: The admitted helper set each installation floor runs on. woof 2.9.10's six helper sources
#: (rrtmg_sw.cu, rrtmg_legacy_prep.py, rrtmg_lw.py, rrtmg_mcica.py, rrtmg_sw.py, wrf_ozone.py) hash
#: to the engine-2.9.7-source row byte for byte (measured on the 2.9.10 candidate, 2026-10-29), so
#: the 2.9.10 floor needs no new row. woof 2.9.11 moved rrtmg_sw.py and wrf_ozone.py (the
#: downstream table checks and the wind-farm row), so the 0.4.11 floor runs on engine-2.9.11-source,
#: measured at the 2.9.11 public commit 82e16d580a.
FLOOR_HELPER_SET = {"2.9.10": "engine-2.9.7-source", "2.9.11": "engine-2.9.11-source"}


def test_the_floor_engine_has_an_admitted_helper_set():
    floor = engine_pin.gpuwm_floor()
    assert floor == "2.9.11"
    assert FLOOR_HELPER_SET[floor] in scope.ADMITTED_SOURCE_SETS
    assert FLOOR_HELPER_SET[floor] in scope.PUBLISHED_MODES
