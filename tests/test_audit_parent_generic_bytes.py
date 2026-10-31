"""Candidate byte controls generated only by original audit-parent emitters.

Copy this test into tests/ and the matrix plus generated historical output
folders into tests/fixtures/generic_byte_controls/. Generation receipts must
name the old Git objects; candidate runs only consume those existing pins.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from woof.domain_wizard import render_config
from woof.namelist_import import import_namelists
from ishmael_retirement import requested_by_own_number, without_retired_line
from wrf_clock_lines import without_wrf_clock_lines
from woof_engine_spelling import engine_spelling


FIXTURES = Path(__file__).parent / "fixtures/generic_byte_controls"
PARENTS = (
    "425e3e068012eafae8d9e6f4ccfb1249151ae91e",  # sfclay audit parent
    "1077c6e8abe502aab59de103d7e6d70dbb8588a1",  # fixed and vadv audit parent
    "32261e42e85fadb9fa4b934f65aba0a22586d5f1",  # dycore audit parent
)
NAMELISTS = (
    "omitted_legacy_defaults", "explicit_vertical_order3",
    "ordinary_fixed60_triplet", "ruc_monthly_omitted", "ruc_monthly_false0",
)
RECIPES = ("gfs", "era5", "rap", "rrfs")


def controls(parent):
    payload = (FIXTURES / "generic-matrix.json").read_bytes()
    pins = json.loads((FIXTURES / parent / "historical-pins.json").read_bytes())
    assert pins["mode"] == "baseline"
    assert pins["audit_parent"] == parent
    assert hashlib.sha256(payload).hexdigest() == pins["matrix_sha256"]
    assert set(pins["cases"]) == {
        *("namelist/" + name for name in NAMELISTS),
        *("recipe/" + source for source in RECIPES),
    }
    matrix = json.loads(payload)
    # The pinned namelists asked for ISHMAEL (55), refused by name since
    # 2026-10-12; every test that imports them asks for Morrison by its own
    # number instead (tests/ishmael_retirement.py).
    for documents in matrix["namelists"].values():
        documents["input"] = requested_by_own_number(documents["input"])
    return matrix, pins


#: Lines an imported namelist gained after these parents were pinned, each
#: from a capability lane that added a default-on key the importer always
#: states: d4ce97c6bbc (GPU eclipse shortwave, RunConfig ra_sw_eclipse = 1)
#: and 97e9a3ca061 (GPU HAILCAST diagnostic, hailcast_opt = -1 automatic,
#: haildt = 60 s).  Each must appear exactly once; nothing else may move.
IMPORT_CAPABILITY_LINES = (
    "ra_sw_eclipse = 1\n", "hailcast_opt = -1\n", "haildt = 60.0\n",
)


def without_import_capability_lines(candidate):
    lines = candidate.splitlines(keepends=True)
    for line in IMPORT_CAPABILITY_LINES:
        assert lines.count(line) == 1, line
        lines.remove(line)
    return "".join(lines)


def assert_old_bytes(parent, pins, key, candidate):
    candidate = engine_spelling(candidate)
    expected = pins["cases"][key]
    path = Path(expected["file"])
    assert len(path.parts) == 1 and path.suffix == ".toml"
    old = (FIXTURES / parent / path).read_bytes()
    assert len(old) == expected["bytes"]
    assert hashlib.sha256(old).hexdigest() == expected["sha256"]
    # A fixed-step WRF namelist gained the D8 pinned-clock block in 2.8.9;
    # everything else is the parent's bytes.
    if key.startswith("namelist/"):
        # ... and, for an imported namelist, exactly the two lines the
        # 2026-10-12 door rule changed: the retired ISHMAEL header row and
        # the omitted lid, now WRF's open top.
        old = without_retired_line(old)
        candidate = without_import_capability_lines(candidate)
    assert without_wrf_clock_lines(candidate).encode("utf-8") == old


#: Lines a recipe emission gained after these parents were pinned, by
#: source.  e19441c1e gave RAP the operational fork's request row (RAP runs
#: the same WRF 3.9 fork as HRRR), so an authored rap recipe states the
#: fork's surface-layer generation in [shared].  The matrix recipe is not
#: mp_physics = 28, so the fork Thompson keys stay scoped out of it.
FORK_GENERATION_LINES = {"rap": ('mynn_sfclay_variant = "gsl_wrf39"\n',)}

#: The line a recipe whose land is RUC (sf_surface_physics = 3) gained after
#: these parents were pinned, from any source: fix/ruc-qvg-authoring-air
#: authors the RUC air cold start (physics_source_defaults.
#: AUTHORED_RUC_DEFAULTS) into [shared].  A non-RUC recipe must not carry it.
RUC_AUTHORING_LINE = 'ruc_qvg_cold_start = "air"\n'


def assert_old_recipe_bytes(parent, pins, source, candidate):
    """The parent's recipe bytes, plus exactly the fork-generation lines and,
    on RUC land, the RUC air cold-start line."""
    lines = candidate.splitlines(keepends=True)
    shared = tomllib.loads(candidate)["shared"]
    for line in FORK_GENERATION_LINES.get(source, ()):
        assert lines.count(line) == 1, line
        key = line.split(" = ")[0]
        assert shared[key] == tomllib.loads(line)[key], line
        lines.remove(line)
    if shared.get("sf_surface_physics") == 3:
        assert lines.count(RUC_AUTHORING_LINE) == 1
        assert shared["ruc_qvg_cold_start"] == "air"
        lines.remove(RUC_AUTHORING_LINE)
    else:
        assert RUC_AUTHORING_LINE not in lines
    assert_old_bytes(parent, pins, "recipe/" + source, "".join(lines))


@pytest.mark.parametrize("parent", PARENTS)
@pytest.mark.parametrize("case", NAMELISTS)
def test_non_hrrr_imported_config_bytes_match_original_audit_parent(parent, case, tmp_path):
    matrix, pins = controls(parent)
    documents = matrix["namelists"][case]
    wps, inp = tmp_path / "namelist.wps", tmp_path / "namelist.input"
    wps.write_text(documents["wps"], encoding="utf-8")
    inp.write_text(documents["input"], encoding="utf-8")
    emitted, _ = import_namelists(wps, inp)
    assert_old_bytes(parent, pins, "namelist/" + case, emitted)


@pytest.mark.parametrize("parent", PARENTS)
@pytest.mark.parametrize("source", RECIPES)
def test_non_hrrr_recipe_bytes_match_original_audit_parent(parent, source):
    matrix, pins = controls(parent)
    recipe = matrix["recipes"]
    emitted = render_config(
        name=recipe["name"], start_time=datetime.fromisoformat(recipe["start_time"]),
        hours=recipe["hours"], projection=recipe["projection"],
        dims=[tuple(row) for row in recipe["dims"]], ratios=tuple(recipe["ratios"]),
        root_dx_m=recipe["root_dx_m"], fetch_hints={"source": source},
        case_data=None, profile=recipe["profile"],
    )
    assert_old_recipe_bytes(parent, pins, source, emitted)
