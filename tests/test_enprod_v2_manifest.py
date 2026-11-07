"""``woof enprod`` reads the ensemble ``woof go --members`` writes.

The breakage: ``go --members`` (2.8.7 and the 2.8.8 candidate) writes one
``gpuwm-ensemble-output.v2`` manifest per domain at
``<run>/<domain>/ensemble-manifest.json``, with the roster in
``member_order`` and the retained histories in ``member_files``.  enprod
read only ``gpuwm-ensemble-manifest.v1``: on the run folder it found no
manifest, and on the domain folder it warned about the schema, handed the
manifest to ``rw_ensbatch``, which refused it ("'members' must be a list")
and printed its usage text, and enprod reported the usage line as the
reason.  Measured on test host 1 with a real 2-member time-lagged run kept with
--keep-member-files: exit 1, "rw_ensbatch --list-fields | --help | --abi".
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from woof import cli, rustwx_lanes
from woof.da import enprod

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _v2_run(tmp_path, *, members=(0, 1), frames=2, keep=True, domain="d01",
            sizes=None):
    """A ``go --members`` run folder: manifest plus member histories."""
    run = tmp_path / "run"
    files = []
    for member in members:
        for frame in range(frames):
            relative = (f"members/member-{member:04d}/wrfout/"
                        f"wrfout_{domain}_2026-10-08_0{6 + frame}_00_00")
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * (100 + frame))
            files.append({"bytes": (sizes or {}).get((member, frame),
                                                     100 + frame),
                          "domain": domain, "member_id": member,
                          "path": relative})
    manifest = run / domain / enprod.MANIFEST_FILENAME
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "schema": enprod.V2_MANIFEST_SCHEMA,
        "members_requested": len(members), "member_order": list(members),
        "member_metadata": [], "keep_member_files": keep,
        "member_files": files if keep else [], "frames": [],
        "products": [], "events": {}, "unavailable_products": [],
        "deleted_scratch": [], "probability_denominator": len(members),
        "spread_ddof": 1, "nonfinite_policy": "mask aggregate; retain "
        "finite_count"}), encoding="utf-8")
    return run, manifest


def test_the_v2_schema_is_the_one_the_ensemble_writer_publishes():
    from woof.ensemble.batch_product_output import NativeDiagnosticSpool

    assert enprod.V2_MANIFEST_SCHEMA == NativeDiagnosticSpool.CONTRACT


def test_the_run_folder_and_the_domain_folder_both_name_the_manifest(tmp_path):
    run, manifest = _v2_run(tmp_path)
    assert enprod.manifest_path(run) == manifest
    assert enprod.manifest_path(run / "d01") == manifest
    assert enprod.manifest_path(run, "d01") == manifest


def test_two_domains_are_refused_until_one_is_named(tmp_path):
    run, d01 = _v2_run(tmp_path)
    _, d02 = _v2_run(tmp_path, domain="d02")
    with pytest.raises(enprod.EnsembleRefusal, match="d01, d02"):
        enprod.manifest_path(run)
    assert enprod.manifest_path(run, "d02") == d02
    with pytest.raises(enprod.EnsembleRefusal, match="d01, d02 only"):
        enprod.manifest_path(run, "d03")


def test_the_roster_is_member_order_read_from_the_kept_histories(tmp_path):
    run, _ = _v2_run(tmp_path, members=(0, 1, 2), frames=3)
    roster = enprod.load_manifest(run)
    assert roster.schema == enprod.V2_MANIFEST_SCHEMA
    assert roster.root == run
    assert [m.number for m in roster.members] == [0, 1, 2]
    for member in roster.members:
        assert member.directory == run / "members" / \
            f"member-{member.number:04d}" / "wrfout"
        assert member.status == "DONE" and not member.overridden
        assert member.declared_wrfout_count == 3


def test_an_aggregate_only_run_is_refused_naming_its_own_maps(tmp_path):
    run, manifest = _v2_run(tmp_path, keep=False)
    with pytest.raises(enprod.EnsembleRefusal) as caught:
        enprod.load_manifest(run)
    message = str(caught.value)
    assert "kept no member histories" in message
    assert str(run / "maps" / "d01") in message
    assert "--keep-member-files" in message


@pytest.mark.parametrize("damage, words", [
    ("missing", "member 1: its history"),
    ("size", "member 1: "),
    ("short", "member(s) 1 recorded fewer d01 histories"),
])
def test_a_damaged_member_is_refused_by_name(tmp_path, damage, words):
    run, manifest = _v2_run(tmp_path)
    if damage == "missing":
        (run / "members/member-0001/wrfout/wrfout_d01_2026-10-08_07_00_00"
         ).unlink()
    elif damage == "size":
        (run / "members/member-0001/wrfout/wrfout_d01_2026-10-08_07_00_00"
         ).write_bytes(b"stale")
    else:
        document = json.loads(manifest.read_text(encoding="utf-8"))
        document["member_files"] = [
            row for row in document["member_files"]
            if not (row["member_id"] == 1 and row["path"].endswith("07_00_00"))]
        manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(enprod.EnsembleRefusal) as caught:
        enprod.load_manifest(run)
    assert words in str(caught.value)
    assert "2-member ensemble" in str(caught.value)


def test_removed_member_histories_name_the_retention_and_restore_remedies(tmp_path):
    run, _ = _v2_run(tmp_path, members=(0, 1, 2, 3), frames=4)
    for history in (run / "members").rglob("wrfout_d01_*"):
        history.unlink()
    with pytest.raises(enprod.EnsembleRefusal) as caught:
        enprod.load_manifest(run)
    message = str(caught.value)
    assert "4-member ensemble" in message
    assert "member 0: its history" in message
    assert "member 3: no d01 history recorded" in message
    assert "restore" in message.lower()
    assert "--keep-member-files" in message


def test_size_damage_does_not_claim_member_histories_were_removed(tmp_path):
    run, _ = _v2_run(tmp_path)
    history = run / "members/member-0001/wrfout/wrfout_d01_2026-10-08_07_00_00"
    history.write_bytes(b"stale")
    with pytest.raises(enprod.EnsembleRefusal) as caught:
        enprod.load_manifest(run)
    assert "where the run recorded" in str(caught.value)
    assert "--keep-member-files" not in str(caught.value)


def test_the_engine_gets_a_v1_roster_of_the_validated_members(tmp_path):
    run, _ = _v2_run(tmp_path)
    roster = enprod.load_manifest(run)
    path = enprod.engine_roster(roster, tmp_path / "store")
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema"] == enprod.MANIFEST_SCHEMA
    assert [(row["member"], row["status"]) for row in document["members"]] \
        == [(0, "DONE"), (1, "DONE")]
    for row in document["members"]:
        assert Path(row["member_dir"]).is_absolute()
        assert Path(row["member_dir"]).is_dir()


def test_the_rust_route_hands_the_engine_the_roster(tmp_path, monkeypatch):
    run, _ = _v2_run(tmp_path)
    seen = {}

    def fake_engine(engine, manifest, *, store_root, out_dir, **kwargs):
        seen["manifest"] = json.loads(Path(manifest).read_text(
            encoding="utf-8"))
        panel = Path(out_dir) / "ens00" / "ens_mean_refl" / "x.png"
        panel.parent.mkdir(parents=True, exist_ok=True)
        panel.write_bytes(_PNG_MAGIC + b"panel")
        return [panel], [], [], {"rendered": [("mean", panel)]}

    monkeypatch.setattr(rustwx_lanes, "find_ensemble_bin",
                        lambda: Path("rw_ensbatch"))
    monkeypatch.setattr(rustwx_lanes, "probe_ensemble_bin",
                        lambda path: (True, "--abi matches the contract"))
    monkeypatch.setattr(rustwx_lanes, "run_ensemble_renderer", fake_engine)
    rc = cli.main(["enprod", str(run), "--field", "refl", "--products",
                   "mean", "--out", str(tmp_path / "png"),
                   "--engine", "rust"])
    assert rc == 0
    assert seen["manifest"]["schema"] == enprod.MANIFEST_SCHEMA
    assert [row["member"] for row in seen["manifest"]["members"]] == [0, 1]


def test_an_aggregate_only_run_is_refused_at_the_door(tmp_path, monkeypatch,
                                                      capsys):
    run, _ = _v2_run(tmp_path, keep=False)
    monkeypatch.setattr(rustwx_lanes, "find_ensemble_bin",
                        lambda: Path("rw_ensbatch"))
    monkeypatch.setattr(rustwx_lanes, "probe_ensemble_bin",
                        lambda path: (True, "--abi matches the contract"))
    monkeypatch.setattr(rustwx_lanes, "run_ensemble_renderer",
                        lambda *a, **k: pytest.fail("engine reached"))
    rc = cli.main(["enprod", str(run), "--out", str(tmp_path / "png"),
                   "--engine", "rust"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "kept no member histories" in err
    assert "rw_ensbatch --list-fields" not in err


def test_a_failing_engine_call_reports_its_reason_not_its_usage():
    stderr = ("/run/d01/ensemble-manifest.json: 'members' must be a list "
              "of member records\n"
              "usage: rw_ensbatch --store-root DIR --out-dir DIR (--manifest "
              "FILE.json | --member N=WRFOUT_OR_MEMBER_DIR ...)\n"
              "       rw_ensbatch --list-fields | --help | --abi\n")
    assert rustwx_lanes.engine_failure_reason(stderr, 2) == (
        "/run/d01/ensemble-manifest.json: 'members' must be a list of "
        "member records")
    assert rustwx_lanes.engine_failure_reason("plain failure\n", 1) == \
        "plain failure"
    assert rustwx_lanes.engine_failure_reason("", 3) == \
        "exit 3 with no message"


def _v1_cycle_legs(tmp_path, legs):
    """A DA cycle root: ``cycle_NNN/`` each a whole v1 ensemble root."""
    root = tmp_path / "cycled"
    for leg in range(legs):
        leg_root = root / f"cycle_{leg:03d}"
        for member in (0, 1):
            (leg_root / f"member_{member:03d}").mkdir(parents=True)
        (leg_root / enprod.MANIFEST_FILENAME).write_text(json.dumps({
            "schema": enprod.MANIFEST_SCHEMA, "n_members": 2,
            "members": [{"member": member, "status": "DONE",
                         "member_dir": f"member_{member:03d}"}
                        for member in (0, 1)]}), encoding="utf-8")
    return root


def test_a_v1_cycle_leg_one_folder_down_is_not_a_domain(tmp_path):
    """Review finding: a v1 leg was read as a v2 domain of its parent.

    Its member_dir entries are relative to the leg's own folder, so read
    from the parent the matplotlib route named directories that do not
    exist and the rust route filed the leg as an undated native grid.
    The parent stays refused exactly as before this release.
    """
    root = _v1_cycle_legs(tmp_path, 1)
    assert enprod.manifest_path(root) == root / enprod.MANIFEST_FILENAME
    with pytest.raises(enprod.EnsembleRefusal, match="no ensemble manifest"):
        enprod.load_manifest(root)


def test_two_v1_cycle_legs_are_not_offered_as_domains(tmp_path):
    root = _v1_cycle_legs(tmp_path, 2)
    assert enprod.manifest_path(root) == root / enprod.MANIFEST_FILENAME
    # --domain filters wrfout names; a cycle folder is never taken for one.
    assert (enprod.manifest_path(root, "cycle_001")
            == root / enprod.MANIFEST_FILENAME)


def test_a_v1_cycle_leg_named_directly_reads_its_own_members(tmp_path):
    root = _v1_cycle_legs(tmp_path, 2)
    leg = root / "cycle_001"
    assert enprod.manifest_path(leg) == leg / enprod.MANIFEST_FILENAME
    manifest = enprod.load_manifest(leg)
    assert [Path(m.directory) for m in manifest.members] == [
        leg / "member_000", leg / "member_001"]


def test_a_v2_domain_beside_a_v1_folder_is_the_only_one_found(tmp_path):
    run, manifest = _v2_run(tmp_path)
    stray = run / "cycle_000"
    stray.mkdir()
    (stray / enprod.MANIFEST_FILENAME).write_text(json.dumps({
        "schema": enprod.MANIFEST_SCHEMA, "n_members": 0, "members": []}),
        encoding="utf-8")
    assert enprod.manifest_path(run) == manifest
