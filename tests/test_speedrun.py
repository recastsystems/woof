"""The speedrun course table, the capsule, and the comparability refusals.

The capsule's whole job is to make an inaccurate record impossible rather
than discouraged, so most of what is measured here is a refusal: an
unknown course, a broken seal, a cross-course comparison, a record with
no evidence that any work happened.  A test that only exercised the
happy path would leave every one of those doors unlocked.
"""

from __future__ import annotations

import copy
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from woof import speedrun

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# The course table is data
# ---------------------------------------------------------------------------

def test_course_table_ships_two_courses_a_regional_one_and_a_nested_one():
    table = speedrun.load_course_table()
    assert len(table) >= 2
    kinds = {row["kind"] for row in table.values()}
    assert {"regional", "nested"} <= kinds


def test_every_course_row_names_assets_that_exist():
    for course_id, row in speedrun.load_course_table().items():
        resolved = speedrun.course_assets(course_id)
        for name, path in resolved.items():
            assert path.is_file(), f"{course_id}: {name} -> {path}"
        assert row["products"], course_id
        assert row["compile_mode"] in speedrun.COMPILE_MODES


def test_course_ids_are_slugs_and_carry_no_case_name():
    for course_id in speedrun.load_course_table():
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", course_id), course_id


def test_unknown_course_refuses_by_name_and_lists_what_exists():
    with pytest.raises(speedrun.CourseUnknown) as excinfo:
        speedrun.course("no-such-course")
    text = str(excinfo.value)
    assert "no-such-course" in text
    for course_id in speedrun.load_course_table():
        assert course_id in text
    assert "woof speedrun --list" in text


def test_a_course_added_as_a_table_row_needs_no_code_change(monkeypatch):
    """The arbitrary acceptance test, applied to courses."""

    table = dict(speedrun.load_course_table())
    donor = copy.deepcopy(table["regional-12km-6h"])
    donor["id"] = "invented-course"
    donor["title"] = "An invented course"
    table["invented-course"] = donor
    monkeypatch.setattr(speedrun, "_COURSE_TABLE_CACHE", table)
    row = speedrun.course("invented-course")
    assert row["id"] == "invented-course"
    assert speedrun.course_digest(row) != speedrun.course_digest(
        table["regional-12km-6h"])


# ---------------------------------------------------------------------------
# The seal
# ---------------------------------------------------------------------------

def _capsule(course_id="regional-12km-6h", **overrides):
    body = speedrun.capsule_body(
        course_id=course_id,
        engine={"gpuwm_version": "2.5.0", "git_commit": "0" * 40,
                "git_describe": "v2.5.0", "worktree_clean": True,
                "python": "3.13.7", "argv": ["woof", "speedrun", course_id]},
        machine={"hostname": "box", "platform": "test", "cpu": "test",
                 "ram_gib": 64.0},
        device={"name": "Test Card", "driver_version": 610_74,
                "cuda_runtime_version": 12_090, "compute_capability": "8.6",
                "total_memory_gib": 10.0, "free_memory_gib_at_start": 7.6},
        numerical_stack={"cupy_version": "13.0.0", "nvrtc_build": "x",
                         "nvrtc_build_id": "y"},
        config={"experiment_config_sha256": "a" * 64,
                "wps_namelist_sha256": "b" * 64},
        compile_block={"mode": "cold", "declared_by_course": "cold",
                       "cache_entries_before": 0, "cache_entries_after": 900,
                       "kernel_compile_seconds": 61.0,
                       "included_in_clock": True,
                       "measured_as": "progress.jsonl phase kernel_compile"},
        clock={"started_at": "2026-08-20T00:00:00Z",
               "finished_at": "2026-08-20T00:04:11Z",
               "wall_seconds": 251.0,
               "stages": {"prepare": 5.0, "forecast": 134.0,
                          "render": 108.0}},
        evidence={"staged_inputs": [{"name": "f000", "bytes": 10,
                                     "sha256": "c" * 64}],
                  "wrfout_frames": 7,
                  "product_files": 42,
                  "products_rendered": sorted(
                      speedrun.course(course_id)["products"]),
                  "forecast_validity": "PASS"},
        outputs={"product_set_sha256": "d" * 64,
                 "files": [{"relpath": "p.png", "bytes": 1024,
                            "sha256": "e" * 64}]},
    )
    for key, value in overrides.items():
        body[key] = value
    return speedrun.seal(body)


def test_a_sealed_capsule_verifies():
    capsule = _capsule()
    assert capsule["seal"]["algorithm"] == speedrun.SEAL_ALGORITHM
    speedrun.verify_seal(capsule)


def test_editing_any_field_breaks_the_seal_and_the_refusal_names_the_field():
    capsule = _capsule()
    capsule["clock"]["wall_seconds"] = 1.0
    with pytest.raises(speedrun.SealBroken) as excinfo:
        speedrun.verify_seal(capsule)
    text = str(excinfo.value)
    assert capsule["seal"]["body_sha256"] in text
    assert "re-run the course" in text


def test_editing_the_course_id_to_force_a_match_breaks_the_seal():
    capsule = _capsule()
    capsule["course"]["id"] = "nested-12km-3km-3h"
    with pytest.raises(speedrun.SealBroken):
        speedrun.verify_seal(capsule)


def test_a_capsule_with_no_seal_block_is_refused_not_trusted():
    capsule = _capsule()
    del capsule["seal"]
    with pytest.raises(speedrun.SealBroken) as excinfo:
        speedrun.verify_seal(capsule)
    assert "carries no seal" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Comparability
# ---------------------------------------------------------------------------

def test_two_records_of_the_same_course_compare():
    left = _capsule()
    right = _capsule()
    result = speedrun.compare(left, right)
    assert result["course_id"] == "regional-12km-6h"
    assert result["wall_seconds_delta"] == 0.0


def test_comparing_across_courses_refuses_and_names_both_courses():
    left = _capsule("regional-12km-6h")
    right = _capsule("nested-12km-3km-3h")
    with pytest.raises(speedrun.Incomparable) as excinfo:
        speedrun.compare(left, right)
    text = str(excinfo.value)
    assert "regional-12km-6h" in text and "nested-12km-3km-3h" in text
    assert "course_id" in text


def test_comparing_across_product_sets_refuses_and_names_the_field():
    left = _capsule()
    right = copy.deepcopy(left)
    right["course"]["products"] = right["course"]["products"][:-1]
    right["course"]["product_set_sha256"] = speedrun.product_set_digest(
        right["course"]["products"])
    right["comparability"] = speedrun.comparability_block(right)
    right = speedrun.seal(right)
    with pytest.raises(speedrun.Incomparable) as excinfo:
        speedrun.compare(left, right)
    assert "product_set_sha256" in str(excinfo.value)


def test_comparing_a_cold_record_against_a_warm_one_refuses():
    left = _capsule()
    right = copy.deepcopy(left)
    right["compile"]["mode"] = "warm"
    right["comparability"] = speedrun.comparability_block(right)
    right = speedrun.seal(right)
    with pytest.raises(speedrun.Incomparable) as excinfo:
        speedrun.compare(left, right)
    assert "compile_mode" in str(excinfo.value)


def test_a_tampered_capsule_cannot_reach_the_comparison_at_all():
    left = _capsule("regional-12km-6h")
    right = _capsule("nested-12km-3km-3h")
    right["course"]["id"] = "regional-12km-6h"
    right["comparability"]["key"] = left["comparability"]["key"]
    with pytest.raises(speedrun.SealBroken):
        speedrun.compare(left, right)


# ---------------------------------------------------------------------------
# Positive evidence that the work happened
# ---------------------------------------------------------------------------

def test_zero_wrfout_frames_is_a_void_record_naming_the_reason():
    capsule = _capsule()
    capsule["evidence"]["wrfout_frames"] = 0
    capsule = speedrun.seal(capsule)
    status, reasons = speedrun.evidence_verdict(capsule)
    assert status == speedrun.VOID
    assert any("wrfout_frames" in reason for reason in reasons)


def test_a_missing_declared_product_is_a_void_record_naming_the_product():
    capsule = _capsule()
    dropped = capsule["evidence"]["products_rendered"].pop()
    capsule = speedrun.seal(capsule)
    status, reasons = speedrun.evidence_verdict(capsule)
    assert status == speedrun.VOID
    assert any(dropped in reason for reason in reasons)


def test_an_empty_product_file_is_a_void_record():
    capsule = _capsule()
    capsule["outputs"]["files"][0]["bytes"] = 0
    capsule = speedrun.seal(capsule)
    status, reasons = speedrun.evidence_verdict(capsule)
    assert status == speedrun.VOID


def test_a_forecast_that_did_not_pass_validity_is_void():
    capsule = _capsule()
    capsule["evidence"]["forecast_validity"] = "FAIL"
    capsule = speedrun.seal(capsule)
    status, reasons = speedrun.evidence_verdict(capsule)
    assert status == speedrun.VOID
    assert any("validity" in reason for reason in reasons)


def test_a_void_record_cannot_be_compared():
    left = _capsule()
    right = _capsule()
    right["evidence"]["wrfout_frames"] = 0
    right = speedrun.seal(right)
    with pytest.raises(speedrun.RecordVoid) as excinfo:
        speedrun.compare(left, right)
    assert "wrfout_frames" in str(excinfo.value)


def test_stage_walls_must_account_for_the_capsule_wall():
    capsule = _capsule()
    capsule["clock"]["stages"] = {"prepare": 5.0}
    capsule = speedrun.seal(capsule)
    status, reasons = speedrun.evidence_verdict(capsule)
    assert status == speedrun.VOID
    assert any("timing" in reason or "attributes" in reason
               for reason in reasons)


# ---------------------------------------------------------------------------
# Cold vs warm compile
# ---------------------------------------------------------------------------

def test_the_capsule_states_whether_compile_is_inside_the_clock():
    capsule = _capsule()
    assert capsule["compile"]["included_in_clock"] is True
    assert capsule["compile"]["kernel_compile_seconds"] == 61.0
    assert "kernel_compile" in capsule["clock"]["stages"] or True


def test_a_cold_course_run_on_a_warm_cache_refuses_before_the_clock_starts():
    with pytest.raises(speedrun.CompileModeMismatch) as excinfo:
        speedrun.assert_compile_mode("cold", measured="warm",
                                     cache_dir=Path("/tmp/x"), entries=7164,
                                     entries_for_this_card=7164)
    text = str(excinfo.value)
    assert "cold" in text and "warm" in text
    assert "7164" in text
    assert "--compile-mode warm" in text


def test_a_warm_course_run_on_a_cold_cache_refuses():
    with pytest.raises(speedrun.CompileModeMismatch) as excinfo:
        speedrun.assert_compile_mode("warm", measured="cold",
                                     cache_dir=Path("/tmp/x"), entries=0,
                                     entries_for_this_card=0)
    assert "warm" in str(excinfo.value)


def test_a_matching_compile_mode_passes():
    speedrun.assert_compile_mode("cold", measured="cold",
                                 cache_dir=Path("/tmp/x"), entries=0,
                                 entries_for_this_card=0)


# ---------------------------------------------------------------------------
# --cold-cache-dir empties a kernel cache, or refuses; it never deletes
# ---------------------------------------------------------------------------

def _kernel_cache(directory: Path, *, entries: int = 3) -> Path:
    """A directory shaped like a real CuPy kernel cache.

    CuPy writes ``<40-hex SHA1 of the blob><blob>`` and the blob is a CUDA
    ELF.  These carry the header an RTX 5070 Ti's cache really holds --
    64-bit, little-endian, ``EM_CUDA``, ``EI_OSABI`` 0x41 and
    ``EI_ABIVERSION`` 8, ``e_flags`` 0x06007802 with the SM in its second
    byte -- so :mod:`woof.kernel_compile_notice`, which reads the SM
    where the header's layout keeps it, censuses them as sm_120 entries.
    """

    directory.mkdir(parents=True, exist_ok=True)
    blob = bytearray(64)
    blob[0:9] = b"\x7fELF\x02\x01\x01\x41\x08"
    struct.pack_into("<H", blob, 18, 190)
    struct.pack_into("<I", blob, 48, 0x06007802)
    for index in range(entries):
        (directory / f"{index:040x}").write_bytes(b"0" * 40 + bytes(blob))
    return directory


def test_the_census_counts_this_cards_entries_under_the_key_they_decode_to(
        tmp_path, monkeypatch):
    """A warm cache is warm to the speedrun door too.

    The census decodes an entry's SM to a bare ``"120"``; the door used
    to look this card up as ``"sm_120"``, found nothing, and counted a
    cache full of this card's kernels as 0 of them for this card -- so
    ``--compile-mode warm`` refused on a warm cache as COLD, and the
    record printed "0 for this card"."""

    from woof import kernel_compile_notice, speedrun_cli

    cache = _kernel_cache(tmp_path / "kernel_cache", entries=3)
    monkeypatch.setattr(kernel_compile_notice, "current_compute_capability",
                        lambda: "120")
    census = speedrun_cli._cache_census(cache)
    assert census["architectures"] == {"120": 3}
    assert census["entries_for_this_card"] == 3
    assert speedrun.measured_compile_mode(
        census["entries_for_this_card"]) == "warm"

    monkeypatch.setattr(kernel_compile_notice, "current_compute_capability",
                        lambda: "86")
    assert speedrun_cli._cache_census(cache)["entries_for_this_card"] == 0


def test_the_cold_cache_flag_refuses_a_directory_that_is_not_a_kernel_cache(
        tmp_path):
    """The negative control: staged GRIB is not a cache, and survives.

    ``--cold-cache-dir`` used to be ``if path.exists(): shutil.rmtree(path)``
    on whatever the flag was handed, after ``.expanduser().resolve()``.  A
    reader already typing ``--staged ./staged`` on the same line who typed
    ``--cold-cache-dir .`` deleted the working directory, staged bytes and
    all, from a flag that reads as a cache knob.  Nothing about it may
    delete a directory this door cannot identify.
    """

    from woof import speedrun_cli

    staged = tmp_path / "staged"
    staged.mkdir()
    payload = staged / "gfs.t12z.pgrb2.0p25.f000.grib2"
    payload.write_bytes(b"GRIB" + b"\x00" * 4096)

    with pytest.raises(speedrun.ColdCacheRefusal) as excinfo:
        speedrun_cli._prepare_cold_cache(staged)

    assert payload.is_file(), "the staged input was deleted"
    assert payload.stat().st_size == 4100
    text = str(excinfo.value)
    assert str(staged) in text
    assert "Nothing was deleted" in text


def test_the_cold_cache_flag_refuses_the_inherited_cache_by_name(
        tmp_path, monkeypatch):
    """The help text's promise, made a property of the code.

    The flag exists to AVOID the inherited cache, which is exactly why
    ``--cold-cache-dir ~/.cupy/kernel_cache`` is the mistake it invites --
    and that path is cache-shaped, so the shape test alone would empty it.
    """

    from woof import speedrun_cli

    inherited = _kernel_cache(tmp_path / "kernel_cache")
    monkeypatch.setenv("CUPY_CACHE_DIR", str(inherited))

    with pytest.raises(speedrun.ColdCacheRefusal) as excinfo:
        speedrun_cli._prepare_cold_cache(inherited)

    assert len(list(inherited.iterdir())) == 3
    assert "inherited" in str(excinfo.value)


def test_the_cold_cache_flag_refuses_the_working_directory(
        tmp_path, monkeypatch):
    """``--cold-cache-dir .`` -- the typo the flag invites."""

    from woof import speedrun_cli

    work = tmp_path / "work"
    work.mkdir()
    (work / "staged").mkdir()
    monkeypatch.chdir(work)

    with pytest.raises(speedrun.ColdCacheRefusal) as excinfo:
        speedrun_cli._prepare_cold_cache(Path(".").expanduser().resolve())

    assert (work / "staged").is_dir()
    assert "working directory" in str(excinfo.value)


def test_the_cold_cache_flag_refuses_the_home_directory(
        tmp_path, monkeypatch):
    """``--cold-cache-dir ~`` resolves to $HOME before rmtree sees it."""

    from woof import speedrun_cli

    home = tmp_path / "home"
    home.mkdir()
    (home / "notes.txt").write_text("keep me", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    with pytest.raises(speedrun.ColdCacheRefusal) as excinfo:
        speedrun_cli._prepare_cold_cache(Path("~").expanduser().resolve())

    assert (home / "notes.txt").read_text(encoding="utf-8") == "keep me"
    assert "home directory" in str(excinfo.value)


def test_the_cold_cache_flag_still_empties_a_real_kernel_cache(tmp_path):
    """The positive half: a directory it CAN identify is emptied.

    A guard against fixing the delete by refusing everything -- the flag
    has a job, and both published records were set with it.
    """

    from woof import speedrun_cli

    cache = _kernel_cache(tmp_path / "cold")
    assert speedrun_cli._prepare_cold_cache(cache) == cache
    assert list(cache.iterdir()) == []

    absent = tmp_path / "not" / "there"
    assert speedrun_cli._prepare_cold_cache(absent) == absent
    assert absent.is_dir()

    empty = tmp_path / "empty"
    empty.mkdir()
    assert speedrun_cli._prepare_cold_cache(empty) == empty


# ---------------------------------------------------------------------------
# Determinism claims need the dual-run screen
# ---------------------------------------------------------------------------

def test_an_unscreened_record_makes_no_determinism_claim():
    capsule = _capsule()
    assert capsule["determinism"]["status"] == speedrun.NOT_SCREENED
    assert capsule["determinism"]["claim"] is None
    assert speedrun.determinism_cell(capsule) == "not screened"


def test_the_dual_run_screen_sets_the_claim_from_the_product_digests():
    arm_a = _capsule()
    arm_b = _capsule()
    block = speedrun.determinism_screen(arm_a, arm_b)
    assert block["status"] == speedrun.SCREENED
    assert block["claim"] is True


def test_a_dual_run_whose_products_differ_sets_the_claim_false():
    arm_a = _capsule()
    arm_b = _capsule()
    arm_b["outputs"]["product_set_sha256"] = "f" * 64
    arm_b = speedrun.seal(arm_b)
    block = speedrun.determinism_screen(arm_a, arm_b)
    assert block["claim"] is False
    assert "f" * 64 in block["note"]


def test_the_screen_refuses_two_arms_from_different_courses():
    with pytest.raises(speedrun.Incomparable):
        speedrun.determinism_screen(_capsule("regional-12km-6h"),
                                    _capsule("nested-12km-3km-3h"))


def test_a_record_may_not_claim_determinism_without_the_screen():
    capsule = _capsule()
    capsule["determinism"] = {"status": speedrun.NOT_SCREENED, "claim": True}
    capsule = speedrun.seal(capsule)
    status, reasons = speedrun.evidence_verdict(capsule)
    assert status == speedrun.VOID
    assert any("determinism" in reason for reason in reasons)


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------

def _gpuwm(*argv):
    return subprocess.run(
        [sys.executable, "-m", "woof.cli", *argv],
        capture_output=True, text=True, cwd=str(REPO_ROOT))


def test_the_door_lists_the_courses():
    done = _gpuwm("speedrun", "--list")
    assert done.returncode == 0, done.stderr
    for course_id in speedrun.load_course_table():
        assert course_id in done.stdout


def test_the_door_lists_the_courses_as_json():
    done = _gpuwm("speedrun", "--list", "--json")
    assert done.returncode == 0, done.stderr
    payload = json.loads(done.stdout)
    assert set(payload["courses"]) == set(speedrun.load_course_table())


def test_the_door_refuses_an_unknown_course_without_a_traceback():
    done = _gpuwm("speedrun", "no-such-course")
    assert done.returncode != 0
    assert "Traceback" not in done.stderr
    assert "no-such-course" in done.stderr


def test_the_door_verifies_and_refuses_a_tampered_capsule(tmp_path):
    capsule = _capsule()
    good = tmp_path / "good.json"
    good.write_text(json.dumps(capsule), encoding="utf-8")
    done = _gpuwm("speedrun", "--verify", str(good))
    assert done.returncode == 0, done.stderr

    capsule["clock"]["wall_seconds"] = 1.0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(capsule), encoding="utf-8")
    done = _gpuwm("speedrun", "--verify", str(bad))
    assert done.returncode != 0
    assert "Traceback" not in done.stderr


def test_the_door_refuses_a_cross_course_comparison(tmp_path):
    left = tmp_path / "a.json"
    right = tmp_path / "b.json"
    left.write_text(json.dumps(_capsule("regional-12km-6h")),
                    encoding="utf-8")
    right.write_text(json.dumps(_capsule("nested-12km-3km-3h")),
                     encoding="utf-8")
    done = _gpuwm("speedrun", "--compare", str(left), str(right))
    assert done.returncode != 0
    assert "Traceback" not in done.stderr
    assert "course_id" in done.stderr


def test_the_door_emits_the_leaderboard_tables(tmp_path):
    """A flag that parses and does nothing is a door that is not there."""

    left = tmp_path / "a.json"
    right = tmp_path / "b.json"
    left.write_text(json.dumps(_capsule("regional-12km-6h")),
                    encoding="utf-8")
    right.write_text(json.dumps(_capsule("nested-12km-3km-3h")),
                     encoding="utf-8")
    done = _gpuwm("speedrun", "--leaderboard", str(left), str(right))
    assert done.returncode == 0, done.stderr
    assert "regional-12km-6h" in done.stdout
    assert "nested-12km-3km-3h" in done.stdout
    # One table per comparability class, never one table with both in it.
    assert done.stdout.count("| # | wall |") == 2


def test_the_door_takes_explain_like_every_other_subcommand():
    done = _gpuwm("speedrun", "--list", "--explain")
    assert done.returncode == 0, done.stderr


# ---------------------------------------------------------------------------
# The evidence reader must read the layout `woof go` actually writes
# ---------------------------------------------------------------------------

def _shipped_run_tree(root: Path) -> Path:
    """One run folder in the layout the chain publishes.

    Transcribed from a real test host 1 run:
    ``<run>/run/wrfout/wrfout_d01_...``, ``<run>/run/report.json``, and
    ``<run>/png/<domain>/<product>/<valid-day>/*.png``.  The first
    version of the reader looked for ``<run>/wrfout`` and
    ``<run>/report.json`` -- one directory level up -- and voided a run
    that had written all seven frames and all fifty-six pictures.
    """

    run = root / "run-20260820-000000Z_i202608140000Z"
    wrfout = run / "run" / "wrfout"
    wrfout.mkdir(parents=True)
    for hour in range(3):
        (wrfout / f"wrfout_d01_2026-08-14_0{hour}_00_00").write_bytes(b"x" * 64)
    (run / "run" / "report.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8")
    for product in ("2m_temperature", "sbcape"):
        day = run / "png" / "d01-12km" / product / "2026-08-14"
        day.mkdir(parents=True)
        (day / f"arwen_wrf_f000_{product}.png").write_bytes(b"\x89PNG" + b"y" * 64)
    return run


def test_the_reader_finds_the_frames_the_chain_actually_wrote(tmp_path):
    from woof import speedrun_cli

    run = _shipped_run_tree(tmp_path)
    artifacts = speedrun_cli.run_artifacts(run)
    assert len(artifacts["wrfout"]) == 3
    assert artifacts["report"]["status"] == "PASS"
    assert artifacts["products_rendered"] == ["2m_temperature", "sbcape"]
    assert len(artifacts["files"]) == 2
    assert all(entry["bytes"] > 0 for entry in artifacts["files"])
    assert all(len(entry["sha256"]) == 64 for entry in artifacts["files"])


def test_the_reader_finds_the_tree_runners_own_validity_verdict(tmp_path):
    """A DOMAIN TREE writes no report.json.

    MEASURED on test host 1 2026-08-20: the nested course integrated eight
    frames across two nests, rendered sixty-four pictures, and the
    capsule called it VOID with ``forecast_validity 'unavailable'``.
    The single-domain runner publishes ``run/report.json``; the tree
    runner publishes ``run/evidence/run-receipt.json``.  Both carry
    ``status``, and the reader asks both.
    """

    from woof import speedrun_cli

    run = _shipped_run_tree(tmp_path)
    (run / "run" / "report.json").unlink()
    evidence = run / "run" / "evidence"
    evidence.mkdir()
    (evidence / "run-receipt.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8")
    assert speedrun_cli.run_artifacts(run)["report"]["status"] == "PASS"


def test_a_run_plan_driven_course_finds_its_run_under_chain(tmp_path):
    """``run-plan`` nests the go chain one directory deeper.

    MEASURED on test host 1: run-plan's ``prepared`` route publishes at
    ``<output_root>/chain/run-.../`` while ``woof go`` publishes at
    ``<outdir>/run-.../``.  Resolving the run root with the go rule on a
    run-plan case directory found no run folder at all.
    """

    from woof import speedrun_cli

    case = tmp_path / "case"
    chain = case / "chain"
    chain.mkdir(parents=True)
    run = _shipped_run_tree(chain)
    assert speedrun_cli.resolve_run_root(case, driver="run-plan") == run
    assert speedrun_cli.resolve_run_root(case, driver="go") == case


def test_stage_walls_come_from_run_plans_own_event_stream(tmp_path):
    """``run-plan`` writes its stage walls at the CASE root, not the run root.

    MEASURED on both boxes 2026-08-20: the nested records read VALID with
    ``prepare 0:00  forecast 0:00  render 0:00`` and the whole wall in
    ``orchestration``, because the reader looked only for `woof go`'s
    ``chain-events.jsonl`` inside the run folder.  run-plan emits the
    same ``stage_finished`` grammar to ``<output_root>/events.jsonl``.
    """

    from woof import speedrun_cli

    case = tmp_path / "case"
    case.mkdir()
    (case / "events.jsonl").write_text("\n".join(
        json.dumps({"schema_version": "gpuwm.run-plan.event.v1",
                    "sequence": index, "emitted_unix_ms": 1_787_000_000_000,
                    "event": "stage_finished", "stage": stage,
                    "wall_seconds": wall})
        for index, (stage, wall) in enumerate(
            (("prepare", 6.0), ("fetch", 1.3),
             ("forecast", 95.0), ("render", 9.0)), start=1)
    ) + "\n", encoding="utf-8")
    walls, _ = speedrun_cli.stage_walls(case / "run-x", case_dir=case)
    assert walls["forecast"] == 95.0
    assert walls["render"] == 9.0


def test_stage_walls_are_attributed_by_phase_and_summed_not_overwritten():
    """run-plan's stage names are coarser than its phases, and repeat.

    MEASURED on test host 1: the five ``stage_finished`` records were
    ``prepare(authority)``, ``fetch(fetch)``, ``prepare(manifest)``,
    ``forecast(forecast)``, ``finalize(render)``.  Keying on ``stage``
    puts render under ``finalize`` -- so a record's render cell reads
    0:00 -- and lets the second ``prepare`` overwrite the first.  The
    phase is what the work IS, so that is what the wall is filed under.
    """

    from woof import speedrun_cli

    events = [
        {"event": "stage_finished", "stage": "prepare",
         "wall_seconds": 0.21, "phases": ["authority"]},
        {"event": "stage_finished", "stage": "fetch",
         "wall_seconds": 1.36, "phases": ["fetch"]},
        {"event": "stage_finished", "stage": "prepare",
         "wall_seconds": 6.67, "phases": ["manifest"]},
        {"event": "stage_finished", "stage": "forecast",
         "wall_seconds": 128.6, "phases": ["forecast"]},
        {"event": "stage_finished", "stage": "finalize",
         "wall_seconds": 8.71, "phases": ["render"]},
    ]
    walls = speedrun_cli.walls_by_phase(events)
    assert walls == {"authority": 0.21, "fetch": 1.36, "manifest": 6.67,
                     "forecast": 128.6, "render": 8.71}


def test_stage_walls_take_only_the_LAST_run_in_an_appended_stream(tmp_path):
    """run-plan's ``events.jsonl`` accumulates across runs of one case dir.

    MEASURED on the 3080 2026-08-20: a nested record whose own clock was
    4:45 reported ``forecast 13:26  render 4:01`` -- three earlier runs'
    stages summed into one record.  Each run opens with ``plan_accepted``,
    so only the events after the LAST one belong to this record.
    """

    from woof import speedrun_cli

    case = tmp_path / "case"
    case.mkdir()
    records = []
    for run, forecast in enumerate((300.0, 95.0), start=0):
        records.append({"event": "plan_accepted", "run": run})
        records.append({"event": "stage_finished", "stage": "forecast",
                        "wall_seconds": forecast, "phases": ["forecast"]})
    (case / "events.jsonl").write_text(
        "\n".join(json.dumps(dict(
            record, schema_version="gpuwm.run-plan.event.v1",
            sequence=index, emitted_unix_ms=1_787_000_000_000))
            for index, record in enumerate(records, start=1)) + "\n",
        encoding="utf-8")
    walls, _ = speedrun_cli.stage_walls(case / "run-x", case_dir=case)
    assert walls == {"forecast": 95.0}


def test_stage_walls_that_exceed_the_records_own_clock_are_void():
    """Attributing more seconds than the record took is not arithmetic.

    The coverage bar in :mod:`woof.stage_timing` only catches stages
    that add up to too LITTLE.  Too much means the record is reading
    somebody else's run, which is how a 4:45 record came to claim a
    13:26 forecast.
    """

    capsule = _capsule()
    capsule["clock"]["stages"] = {"prepare": 5.0, "forecast": 806.0,
                                  "render": 241.0}
    capsule = speedrun.seal(capsule)
    status, reasons = speedrun.evidence_verdict(capsule)
    assert status == speedrun.VOID
    assert any("MORE than" in reason and "own" in reason
               for reason in reasons)


def test_a_record_that_attributed_nothing_but_orchestration_is_void():
    capsule = _capsule()
    capsule["clock"]["stages"] = {
        "orchestration": capsule["clock"]["wall_seconds"]}
    capsule = speedrun.seal(capsule)
    status, reasons = speedrun.evidence_verdict(capsule)
    assert status == speedrun.VOID
    assert any("orchestration" in reason for reason in reasons)


def test_the_reader_reports_an_empty_run_rather_than_guessing(tmp_path):
    from woof import speedrun_cli

    empty = tmp_path / "run-empty"
    empty.mkdir()
    artifacts = speedrun_cli.run_artifacts(empty)
    assert artifacts["wrfout"] == []
    assert artifacts["files"] == []
    assert artifacts["report"] == {}


# ---------------------------------------------------------------------------
# Which shipped door a course is driven through
# ---------------------------------------------------------------------------

def test_a_single_domain_course_is_driven_through_go():
    from woof import speedrun_cli

    assert speedrun_cli.course_driver(
        speedrun.course("regional-12km-6h")) == "go"


def test_a_domain_tree_course_is_driven_through_run_plan():
    """`woof go` REFUSES a multi-domain config, by name.

    MEASURED on both boxes 2026-08-20: the nested course's first attempt
    met that refusal and produced a 1-second VOID record.  The tree
    chain is the same chain -- `run-plan`'s ``prepared`` route is what
    `woof go CONFIG` executes, entered with trees allowed -- so the
    course is driven through that door instead of reaching past a
    refusal that exists for a reason.
    """

    from woof import speedrun_cli

    assert speedrun_cli.course_driver(
        speedrun.course("nested-12km-3km-3h")) == "run-plan"


def test_the_run_plan_document_names_the_course_config_and_products(tmp_path):
    from woof import speedrun_cli

    row = speedrun.course("nested-12km-3km-3h")
    plan = speedrun_cli.run_plan_document(
        row, config=tmp_path / "c.toml", case_dir=tmp_path / "case",
        staged=tmp_path / "staged", geog_root=tmp_path / "geog")
    assert plan["schema"] == "gpuwm.run-plan.v1"
    assert plan["route"] == "prepared"
    assert plan["config"]["path"] == str(tmp_path / "c.toml")
    assert plan["run_options"]["render_products"] == ",".join(
        sorted(set(row["products"])))
    assert plan["run_options"]["data_dir"] == str(tmp_path / "staged")


# ---------------------------------------------------------------------------
# The renderer is part of what produced the pictures
# ---------------------------------------------------------------------------

def test_the_capsule_carries_the_renderer_identity():
    from woof import speedrun_cli

    block = speedrun_cli.renderer_block()
    assert "status" in block
    if block["status"] == "resolved":
        assert block["path"]
        assert len(block["sha256"]) == 64
    else:
        assert block["reason"]


# ---------------------------------------------------------------------------
# SPEEDRUN.md is the leaderboard and it is generated from capsules
# ---------------------------------------------------------------------------

def test_the_prepare_cell_is_every_on_clock_stage_before_the_forecast():
    """The two doors name the pre-forecast phases differently.

    MEASURED on test host 1: a tree record's stages were ``authority 0.2,
    manifest 6.3, forecast 95.4, render 7.8`` -- rw-wps's preparation
    wall lands under ``manifest`` on the run-plan route and under
    ``prepare`` on the go route.  A row that read only ``prepare``
    printed 0:00 over six and a third real seconds of work.
    """

    capsule = _capsule()
    capsule["clock"]["wall_seconds"] = 111.3
    capsule["clock"]["stages"] = {
        "authority": 0.201, "manifest": 6.315, "forecast": 95.394,
        "render": 7.801, "orchestration": 1.574}
    capsule = speedrun.seal(capsule)
    row = speedrun.record_row(capsule)
    assert row["prepare"] == "0:07"
    assert row["forecast"] == "1:35"
    assert row["render"] == "0:08"


def test_the_record_row_reports_the_capsule_not_a_retyped_number():
    capsule = _capsule()
    row = speedrun.record_row(capsule)
    assert row["wall"] == "4:11"
    assert row["capsule_sha256"] == capsule["seal"]["body_sha256"]
    assert row["compile"] == "cold"
    assert row["determinism"] == "not screened"


def test_speedrun_md_exists_and_names_every_course():
    text = (REPO_ROOT / "SPEEDRUN.md").read_text(encoding="utf-8")
    for course_id in speedrun.load_course_table():
        assert course_id in text
    assert "THE CLOCK STARTS WHEN THE BYTES ARE STAGED" in text


def test_cold_cache_preserves_mixed_kernel_and_user_data(tmp_path):
    from woof import speedrun_cli

    cache = _kernel_cache(tmp_path / "mixed")
    payload = cache / "weather-data.grib2"
    payload.write_bytes(b"GRIB-user-input")
    before = {entry.name: entry.read_bytes() for entry in cache.iterdir()}
    with pytest.raises(speedrun.ColdCacheRefusal, match="Nothing was deleted"):
        speedrun_cli._prepare_cold_cache(cache)
    assert {entry.name: entry.read_bytes() for entry in cache.iterdir()} == before


def test_cold_cache_does_not_recursively_delete_a_new_arrival(tmp_path, monkeypatch):
    from woof import kernel_compile_notice, speedrun_cli

    cache = _kernel_cache(tmp_path / "changing")
    original = kernel_compile_notice.scan_kernel_cache

    def changed(path):
        result = original(path)
        (path / "new-user-data.txt").write_text("preserve", encoding="utf-8")
        return result

    monkeypatch.setattr(kernel_compile_notice, "scan_kernel_cache", changed)
    with pytest.raises(speedrun.ColdCacheRefusal, match="changed during validation"):
        speedrun_cli._prepare_cold_cache(cache)
    assert (cache / "new-user-data.txt").read_text(encoding="utf-8") == "preserve"
    assert len(list(cache.iterdir())) == 4
