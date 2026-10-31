"""The pair door: derivation, the identity proof, collection, the summary.

Everything here runs on a CPU-only box and never starts a forecast.  The
door's one expensive contact -- running each leg through the forecast door as
a subprocess -- is the injected ``runner`` of :func:`woof.hex.pair_door.run_pair`,
so a fake runner writes tiny frames and every decision this door makes is
exercised at numbers the test states outright.

WHY THE NUMBERS ARE ASSERTED RATHER THAN SPOT-CHECKED.  The summary is the
instrument: it is what a reader will quote.  A test that only checked the
summary was written would pass on an area weighting silently dropped, on a
difference taken the wrong way round, and on a treatment-only field counted
per value instead of per cell.  So the fixture uses four cells with unequal
areas and two levels, and each expected mean is a different number under each
of those mistakes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import pair_door as door  # noqa: E402
from woof.hex.cli import build_parser, main  # noqa: E402

CELLS = 4
LEVELS = 2

#: Deliberately unequal, and deliberately not the areas the grid file below
#: carries.  Every mean in this file is a different number under each of the
#: three weighting routes, so the summary cannot be right by accident.
FRAME_AREAS = (1.0, 2.0, 3.0, 4.0)
GRID_AREAS = (4.0, 3.0, 2.0, 1.0)

#: control theta, and the per-cell/per-level amount the treatment leg adds.
CONTROL_THETA = 300.0
THETA_DELTA = ((0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0))

#: A three-dimensional field the treatment leg carries and the control does
#: not, and a two-dimensional accumulator beside it.
EXTRA_SCALAR = ((0.0, 0.0), (0.0, 2.0), (4.0, 4.0), (0.0, 0.0))
EXTRA_ACCUMULATOR = (0.0, 5.0, 0.0, 10.0)

FRAME_NAME = "cuda-history.0001-01-01_00.00.00.nc"


# ---------------------------------------------------------------------------
# the fixture frames and the fake runner
# ---------------------------------------------------------------------------
def _write_frame(path: Path, *, treatment: bool, areas: tuple[float, ...] | None):
    """One history frame in the shape the real writer emits.

    Written with netCDF4 directly rather than through the port's history
    writer because this door reads plain netCDF and nothing here is testing
    the writer; what matters is the dimension names the door keys on --
    ``Time``, ``nCells``, ``nVertLevels`` -- and those are asserted against
    the real writer's conventions by that writer's own tests.
    """

    import numpy
    from netCDF4 import Dataset

    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(str(path), "w", format="NETCDF4") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("nCells", CELLS)
        dataset.createDimension("nVertLevels", LEVELS)
        if areas is not None:
            variable = dataset.createVariable("areaCell", "f8", ("nCells",))
            variable[:] = numpy.asarray(areas, dtype=float)
        theta = dataset.createVariable(
            "theta", "f8", ("Time", "nCells", "nVertLevels")
        )
        values = numpy.full((CELLS, LEVELS), CONTROL_THETA, dtype=float)
        if treatment:
            values = values + numpy.asarray(THETA_DELTA, dtype=float)
        theta[0, :, :] = values
        if treatment:
            extra = dataset.createVariable(
                "declared_extra_scalar", "f8", ("Time", "nCells", "nVertLevels")
            )
            extra[0, :, :] = numpy.asarray(EXTRA_SCALAR, dtype=float)
            accumulated = dataset.createVariable(
                "declared_extra_accumulator", "f8", ("Time", "nCells")
            )
            accumulated[0, :] = numpy.asarray(EXTRA_ACCUMULATOR, dtype=float)
    return path


def _write_grid(path: Path) -> Path:
    import numpy
    from netCDF4 import Dataset

    with Dataset(str(path), "w", format="NETCDF4") as dataset:
        dataset.createDimension("nCells", CELLS)
        variable = dataset.createVariable("areaCell", "f8", ("nCells",))
        variable[:] = numpy.asarray(GRID_AREAS, dtype=float)
    return path


def _fake_runner(*, areas: tuple[float, ...] | None, rc: dict[str, int] | None = None):
    """A runner that writes what a finished leg leaves behind, and nothing else.

    It records the legs it was asked to run, so a test can prove the pair
    STOPPED rather than merely that it reported a failure.
    """

    rc = rc or {}
    calls: list[str] = []

    def run(plan: door.LegPlan) -> int:
        calls.append(plan.name)
        plan.out.mkdir(parents=True, exist_ok=True)
        plan.log.parent.mkdir(parents=True, exist_ok=True)
        plan.log.write_text(
            " ".join(plan.command) + "\n", encoding="utf-8", newline="\n"
        )
        code = int(rc.get(plan.name, 0))
        if code != 0:
            return code
        (plan.out / "forecast-receipt.json").write_text(
            json.dumps({"schema": "gpuwm-hex.forecast-run/v1", "leg": plan.name}),
            encoding="utf-8",
            newline="\n",
        )
        _write_frame(
            plan.out / FRAME_NAME, treatment=plan.name == "treatment", areas=areas
        )
        if plan.name == "treatment":
            (plan.out / "budget.json").write_text("{}\n", encoding="utf-8")
            (plan.out / "ledger.csv").write_text("row\n", encoding="utf-8")
        return 0

    run.calls = calls  # type: ignore[attr-defined]
    return run


# ---------------------------------------------------------------------------
# the authored request
# ---------------------------------------------------------------------------
def _table(tmp_path: Path) -> Path:
    path = tmp_path / "sources.csv"
    path.write_text(
        "latitude,longitude,height_m,rate\n40.0,-100.0,1500.0,1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _arguments(tmp_path: Path, *extra: str) -> argparse.Namespace:
    """The pair door's own parser, reached the way a user reaches it."""

    return build_parser().parse_args(
        [
            "pair",
            "--mesh", "x1.2562",
            "--init-source", "reanalysis 2026-01-01 00Z",
            "--hours", "6.0",
            "--history-every-minutes", "60",
            "--source-table", str(_table(tmp_path)),
            "--pair-out", str(tmp_path / "pair"),
            *extra,
        ]
    )


# ---------------------------------------------------------------------------
# the derivation
# ---------------------------------------------------------------------------
def test_the_control_leg_is_the_treatment_leg_with_the_table_cleared(
    tmp_path: Path,
) -> None:
    """One authored request; the control differs by the table and nothing else."""

    request = door.resolve_pair_request(_arguments(tmp_path))

    control = list(request.control_argv)
    treatment = list(request.treatment_argv)

    assert door.SOURCE_TABLE_FLAG not in control
    assert door.SOURCE_TABLE_FLAG in treatment
    assert treatment[treatment.index(door.SOURCE_TABLE_FLAG) + 1] == str(
        request.source_table
    )

    # Strip the one flag and the two derived destinations and the two vectors
    # are the SAME LIST, in the same order.  Compared as a sequence rather
    # than as a set, because a re-ordered vector is a vector somebody edited.
    def stripped(argv: list[str]) -> list[str]:
        table = door.forecast_option_table()
        out: list[str] = []
        index = 0
        while index < len(argv):
            flag = argv[index]
            step = 2 if table[flag] else 1
            if flag not in (door.SOURCE_TABLE_FLAG, *door.PER_LEG_FLAGS):
                out += argv[index:index + step]
            index += step
        return out

    assert stripped(control) == stripped(treatment)
    assert stripped(control) == list(request.authored_argv)

    # The derived destinations, and the kernel cache derived the way the
    # forecast door derives its own default.
    from woof.hex.forecast_door import default_scratch

    assert request.control_out == tmp_path / "pair" / "control" / "out"
    assert request.treatment_out == tmp_path / "pair" / "treatment" / "out"
    assert request.control_scratch == default_scratch(request.control_out)
    assert request.treatment_scratch == default_scratch(request.treatment_out)
    assert request.control_scratch != request.treatment_scratch


def test_an_explicit_scratch_is_still_split_per_leg(tmp_path: Path) -> None:
    shared = tmp_path / "cache"
    request = door.resolve_pair_request(
        _arguments(tmp_path, "--scratch", str(shared))
    )
    assert request.control_scratch == shared / "control"
    assert request.treatment_scratch == shared / "treatment"


def test_the_authored_vector_carries_the_forecast_doors_own_flags(
    tmp_path: Path,
) -> None:
    """Reused, not restated: the pair parser IS the forecast argument surface."""

    request = door.resolve_pair_request(
        _arguments(tmp_path, "--convection", "gf", "--local-timestep")
    )
    authored = list(request.authored_argv)
    assert authored[authored.index("--convection") + 1] == "gf"
    assert "--local-timestep" in authored
    # Defaults resolve to explicit tokens, so the manifest records what ran.
    assert authored[authored.index("--horiz-mixing") + 1] == "2d_smagorinsky"


# ---------------------------------------------------------------------------
# the identity proof
# ---------------------------------------------------------------------------
BASE = ["--mesh", "x1.2562", "--hours", "6.0", "--history-every-minutes", "60"]
LEGS = {
    "control": ["--out", "/c/out", "--scratch", "/c/cache"],
    "treatment": ["--out", "/t/out", "--scratch", "/t/cache", "--source-table", "/t.csv"],
}


def test_the_identity_proof_passes_a_correctly_derived_pair() -> None:
    proof = door.assert_pair_identity(BASE + LEGS["control"], BASE + LEGS["treatment"])
    assert proof["proved"] is True
    assert proof["token_diff"]["treatment_only"] == ["--source-table", "/t.csv"]
    assert proof["token_diff"]["control_only"] == []
    assert proof["token_diff"]["value_differs"] == {
        "--out": ["/c/out", "/t/out"],
        "--scratch": ["/c/cache", "/t/cache"],
    }
    assert proof["identical_flags"]["--hours"] == "6.0"


def test_the_identity_proof_names_a_flag_only_one_leg_carries() -> None:
    with pytest.raises(door.PairRefusal) as refusal:
        door.assert_pair_identity(
            BASE + LEGS["control"],
            BASE + ["--convection", "gf"] + LEGS["treatment"],
        )
    message = str(refusal.value)
    assert "--convection" in message
    assert "'gf'" in message
    assert "treatment leg and not for the control leg" in message


def test_the_identity_proof_names_a_flag_whose_value_drifted() -> None:
    with pytest.raises(door.PairRefusal) as refusal:
        door.assert_pair_identity(
            ["--mesh", "x1.2562", "--hours", "6.0"] + LEGS["control"],
            ["--mesh", "x1.2562", "--hours", "12.0"] + LEGS["treatment"],
        )
    message = str(refusal.value)
    assert "--hours" in message
    assert "'6.0'" in message and "'12.0'" in message
    assert "attributed to the table alone" in message


def test_the_identity_proof_names_every_differing_token_at_once() -> None:
    """Not the first one.  A pair that drifted twice is fixed once."""

    with pytest.raises(door.PairRefusal) as refusal:
        door.assert_pair_identity(
            ["--mesh", "x1.2562", "--hours", "6.0", "--convection", "off"]
            + LEGS["control"],
            ["--mesh", "x4.163842", "--hours", "12.0", "--convection", "gf"]
            + LEGS["treatment"],
        )
    message = str(refusal.value)
    for flag in ("--mesh", "--hours", "--convection"):
        assert flag in message
    assert "3 problem(s)" in message


def test_a_control_leg_carrying_the_table_is_refused() -> None:
    with pytest.raises(door.PairRefusal) as refusal:
        door.assert_pair_identity(
            BASE + LEGS["control"] + ["--source-table", "/t.csv"],
            BASE + LEGS["treatment"],
        )
    assert "appears in the CONTROL leg" in str(refusal.value)


def test_two_legs_sharing_one_destination_are_refused() -> None:
    with pytest.raises(door.PairRefusal) as refusal:
        door.assert_pair_identity(
            BASE + ["--out", "/same", "--scratch", "/c/cache"],
            BASE + ["--out", "/same", "--scratch", "/t/cache", "--source-table", "/t.csv"],
        )
    assert "--out is '/same' for BOTH legs" in str(refusal.value)


def test_an_unreadable_token_is_named_rather_than_skipped() -> None:
    with pytest.raises(door.PairRefusal) as refusal:
        door.assert_pair_identity(
            BASE + ["--not-a-flag", "x"] + LEGS["control"], BASE + LEGS["treatment"]
        )
    message = str(refusal.value)
    assert "'--not-a-flag'" in message
    assert "would go unreported" in message


# ---------------------------------------------------------------------------
# the destination
# ---------------------------------------------------------------------------
def test_a_non_fresh_pair_out_is_refused_and_names_what_is_in_it(
    tmp_path: Path,
) -> None:
    occupied = tmp_path / "pair"
    occupied.mkdir()
    (occupied / "control").mkdir()
    (occupied / "pair_manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(door.PairRefusal) as refusal:
        door.resolve_pair_request(_arguments(tmp_path))
    message = str(refusal.value)
    assert "is not empty" in message
    assert "control" in message
    assert "pair_manifest.json" in message


def test_an_empty_pair_out_directory_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "pair").mkdir()
    assert door.resolve_pair_request(_arguments(tmp_path)).pair_out.is_dir()


def test_a_missing_source_table_is_refused_by_name(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments.source_table = tmp_path / "absent.csv"
    with pytest.raises(door.PairRefusal) as refusal:
        door.resolve_pair_request(arguments)
    assert "absent.csv" in str(refusal.value)


def test_the_forecast_doors_own_out_flag_is_refused(tmp_path: Path) -> None:
    with pytest.raises(door.PairRefusal) as refusal:
        door.resolve_pair_request(
            _arguments(tmp_path, "--out", str(tmp_path / "elsewhere"))
        )
    message = str(refusal.value)
    assert "--out names ONE destination and a pair has two" in message
    assert "--pair-out" in message


def test_the_forecast_doors_own_receipt_flag_is_refused(tmp_path: Path) -> None:
    with pytest.raises(door.PairRefusal) as refusal:
        door.resolve_pair_request(
            _arguments(tmp_path, "--receipt", str(tmp_path / "r.json"))
        )
    assert "a pair writes two receipts" in str(refusal.value)


# ---------------------------------------------------------------------------
# end to end, on the injected runner
# ---------------------------------------------------------------------------
def _rows(summary: str) -> dict[str, list[str]]:
    """The summary's field rows, keyed by field name, whitespace normalised."""

    rows: dict[str, list[str]] = {}
    for line in summary.splitlines():
        pieces = line.split()
        if len(pieces) >= 4 and pieces[1] == "mean" and line.startswith("    "):
            rows[pieces[0]] = pieces
    return rows


def test_the_pair_writes_a_manifest_and_a_summary(tmp_path: Path) -> None:
    request = door.resolve_pair_request(_arguments(tmp_path))
    runner = _fake_runner(areas=FRAME_AREAS)
    manifest = door.run_pair(request, runner=runner)

    assert runner.calls == ["control", "treatment"]
    assert manifest["schema"] == "gpuwm-hex.pair/v1"
    assert manifest["leg_order"] == ["control", "treatment"]
    assert manifest["identity"]["proved"] is True
    assert manifest["identity"]["token_diff"]["treatment_only"] == [
        "--source-table", str(request.source_table)
    ]
    assert manifest["source_table"]["sha256"] == door.sha256_file(
        request.source_table
    )

    # Both legs: rc, wall clock, receipt, one history frame with a digest.
    for name in ("control", "treatment"):
        leg = manifest["legs"][name]
        assert leg["rc"] == 0
        assert leg["wall_seconds"] >= 0.0
        assert Path(leg["receipt"]).is_file()
        assert leg["receipt_sha256"] == door.sha256_file(Path(leg["receipt"]))
        assert [Path(row["path"]).name for row in leg["history"]] == [FRAME_NAME]
        for row in leg["history"]:
            assert row["sha256"] == door.sha256_file(Path(row["path"]))
        assert Path(leg["log"]).is_file()

    # The opaque accounting files, carried by path and digest and not read.
    assert [Path(row["path"]).name for row in manifest["legs"]["control"]["extra_files"]] == []
    assert sorted(
        Path(row["path"]).name for row in manifest["legs"]["treatment"]["extra_files"]
    ) == ["budget.json", "ledger.csv"]

    # The manifest on disk is the manifest returned.
    written = json.loads(
        (request.pair_out / "pair_manifest.json").read_text(encoding="utf-8")
    )
    assert written == manifest

    summary = (request.pair_out / "summary.txt").read_text(encoding="utf-8")
    assert summary.splitlines()[0] == door.RESULT_SENTENCE
    assert "MODEL RESULT" in summary.splitlines()[0]
    assert "areaCell carried in the treatment leg's own frames" in summary
    assert manifest["area_weighting"] in summary

    rows = _rows(summary)
    # theta: area-weighted mean of the difference is 40/20 = 2.0, not the
    # unweighted 1.5, and the peak difference is the deepest cell's 3.0.
    assert rows["theta"][2] == "+2.000000e+00"
    assert rows["theta"][4] == "+3.000000e+00"
    # The declared extra scalar, treatment leg only: 28/20 = 1.4, peak 4.0,
    # and two of the four cells carry anything above zero at all.
    assert rows["declared_extra_scalar"][2] == "+1.400000e+00"
    assert rows["declared_extra_scalar"][4] == "+4.000000e+00"
    assert rows["declared_extra_scalar"][-3:] == ["2", "of", "4"]
    # The two-dimensional accumulator: 50/10 = 5.0, peak 10.0, two cells.
    assert rows["declared_extra_accumulator"][2] == "+5.000000e+00"
    assert rows["declared_extra_accumulator"][4] == "+1.000000e+01"
    assert rows["declared_extra_accumulator"][-3:] == ["2", "of", "4"]
    # areaCell is mesh geometry and is not reported as a field.
    assert "areaCell" not in rows


def test_frames_with_no_area_say_the_means_are_unweighted(tmp_path: Path) -> None:
    request = door.resolve_pair_request(_arguments(tmp_path))
    door.run_pair(request, runner=_fake_runner(areas=None))

    summary = (request.pair_out / "summary.txt").read_text(encoding="utf-8")
    assert "UNWEIGHTED" in summary
    assert summary.splitlines()[0] == door.RESULT_SENTENCE
    assert "over-weights the refined region" in summary
    # 12/8 = 1.5, the plain average over cells and levels.
    assert _rows(summary)["theta"][2] == "+1.500000e+00"


def test_the_grid_supplies_the_area_when_the_frames_carry_none(
    tmp_path: Path,
) -> None:
    grid = _write_grid(tmp_path / "grid.nc")
    request = door.resolve_pair_request(_arguments(tmp_path, "--grid", str(grid)))
    door.run_pair(request, runner=_fake_runner(areas=None))

    summary = (request.pair_out / "summary.txt").read_text(encoding="utf-8")
    assert "areaCell from --grid" in summary
    # 20/20 = 1.0 under the grid's areas, which are neither the frame's
    # areas (2.0) nor unweighted (1.5).
    assert _rows(summary)["theta"][2] == "+1.000000e+00"


def test_a_failing_leg_stops_the_pair_naming_the_leg_and_the_rc(
    tmp_path: Path,
) -> None:
    request = door.resolve_pair_request(_arguments(tmp_path))
    runner = _fake_runner(areas=FRAME_AREAS, rc={"treatment": 3})

    with pytest.raises(door.PairRefusal) as refusal:
        door.run_pair(request, runner=runner)

    message = str(refusal.value)
    assert "the treatment leg exited 3" in message
    assert str(request.pair_out / "treatment" / "forecast.log") in message
    assert not (request.pair_out / "pair_manifest.json").exists()
    assert not (request.pair_out / "summary.txt").exists()


def test_the_mpas_suite_routes_stream_frames_are_the_legs_frames(tmp_path: Path) -> None:
    # hex parity run group 7, #27: the MPAS suite route writes MPAS's own history.* stream files, and the
    # pair counted "frames control 0, treatment 0" beside legs that had written theirs.
    request = door.resolve_pair_request(_arguments(tmp_path))
    base = _fake_runner(areas=FRAME_AREAS)

    def suite_route(plan: door.LegPlan) -> int:
        code = base(plan)
        (plan.out / FRAME_NAME).rename(plan.out / "history.2025-05-19_13.00.00.nc")
        (plan.out / "native-physics-request.json").write_text("{}\n", encoding="utf-8")
        return code

    manifest = door.run_pair(request, runner=suite_route)
    for name in ("control", "treatment"):
        assert [Path(row["path"]).name for row in manifest["legs"][name]["history"]] == [
            "history.2025-05-19_13.00.00.nc"]
    assert _rows((request.pair_out / "summary.txt").read_text(encoding="utf-8"))["theta"][2] == "+2.000000e+00"


def test_a_pair_whose_legs_left_no_frames_refuses_after_its_manifest(tmp_path: Path) -> None:
    # hex parity run group 7, #27: "NO COMPARISON" was printed and the door still exited 0.
    request = door.resolve_pair_request(_arguments(tmp_path))
    base = _fake_runner(areas=FRAME_AREAS)

    def no_frames(plan: door.LegPlan) -> int:
        code = base(plan)
        (plan.out / FRAME_NAME).unlink()
        return code

    with pytest.raises(door.PairRefusal, match="NO COMPARISON: the control and treatment leg"):
        door.run_pair(request, runner=no_frames)
    assert (request.pair_out / "pair_manifest.json").is_file()
    assert "NO COMPARISON" in (request.pair_out / "summary.txt").read_text(encoding="utf-8")


def test_a_failing_control_leg_never_spends_the_treatment_leg(
    tmp_path: Path,
) -> None:
    request = door.resolve_pair_request(_arguments(tmp_path))
    runner = _fake_runner(areas=FRAME_AREAS, rc={"control": 2})

    with pytest.raises(door.PairRefusal) as refusal:
        door.run_pair(request, runner=runner)

    assert "the control leg exited 2" in str(refusal.value)
    assert runner.calls == ["control"]


# ---------------------------------------------------------------------------
# the door itself
# ---------------------------------------------------------------------------
def test_the_help_path_reaches_the_pair_door() -> None:
    with pytest.raises(SystemExit) as exit_code:
        main(["pair", "--help"])
    assert exit_code.value.code == 0


def test_the_command_is_registered_beside_forecast() -> None:
    parser = build_parser()
    choices: dict[str, argparse.ArgumentParser] = {}
    for action in parser._subparsers._group_actions:  # type: ignore[union-attr]
        choices.update(getattr(action, "choices", {}))
    assert "pair" in choices
    assert choices["pair"].get_default("handler") is door.run_pair_door


def test_the_pair_parser_carries_every_forecast_flag() -> None:
    """The two doors share one argument surface and cannot drift apart."""

    forecast = argparse.ArgumentParser(add_help=False)
    from woof.hex.forecast_door import add_forecast_arguments

    add_forecast_arguments(forecast)
    expected = {
        option for action in forecast._actions for option in action.option_strings
    }

    pair = argparse.ArgumentParser(add_help=False)
    door.add_pair_arguments(pair)
    present = {option for action in pair._actions for option in action.option_strings}

    assert expected <= present
    assert {"--source-table", "--pair-out"} <= present


def test_each_leg_scratch_parent_exists_and_the_leaf_does_not(tmp_path: Path) -> None:
    """The forecast door refuses an existing scratch and its driver makes
    the leaf with no parents, so this door owes exactly the parent."""

    seen: list[tuple[bool, bool]] = []

    def runner(plan) -> int:
        seen.append((plan.scratch.parent.is_dir(), plan.scratch.exists()))
        return 2

    request = door.resolve_pair_request(
        _arguments(tmp_path, "--scratch", str(tmp_path / "fresh-scratch"))
    )
    with pytest.raises(door.PairRefusal):
        door.run_pair(request, runner=runner)
    assert seen == [(True, False)]


def test_pair_stores_have_independent_leg_prefixes(tmp_path):
    args = _arguments(tmp_path)
    args.output_format = 'icechunk'
    args.icechunk_store = 's3://bucket/base'
    request = door.resolve_pair_request(args)
    for leg, argv in [('control', request.control_argv), ('treatment', request.treatment_argv)]:
        assert argv[argv.index('--icechunk-store')+1] == 's3://bucket/base/'+leg
    assert '--icechunk-store' not in request.authored_argv
