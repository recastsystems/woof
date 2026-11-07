"""A native posted tree keeps its recovery binding and propagates a stop."""

from __future__ import annotations

import hashlib
import json
import time
from types import SimpleNamespace

import pytest

from woof import go_cli, runplan, stage_reuse
from woof.ingest import boundary_stream


def test_a_posted_hierarchy_records_the_sealed_digest_for_its_next_launch(
        tmp_path, monkeypatch):
    prep_root = tmp_path / "root"
    prep_root.mkdir()
    manifest = prep_root / "native" / "posted-source" / "SHA256SUMS"
    run_dir = tmp_path / "run"
    tree_root = run_dir / "chain" / "hrrr-hierarchy"
    placeholder = boundary_stream.as_posted_placeholder("a" * 64)
    head = {"basis": {
        "as_posted": {"documents": {"source_manifest": {
            "path": "native/posted-source/SHA256SUMS"}}},
        "cache": {"identity": {"source_manifest_sha256": placeholder}}}}
    monkeypatch.setattr(boundary_stream, "live_chained_head", lambda root: head)
    monkeypatch.setattr(stage_reuse, "engine_source_identity", lambda: {})
    inputs = {}
    for role in ("target_domain", "wps_namelist", "namelist_input",
                 "stock_namelist_input"):
        path = tmp_path / role
        path.write_text(role)
        inputs[role] = path
    geog = tmp_path / "geog"
    geog.mkdir()
    executed = []

    def stage(label, command, **kwargs):
        executed.append(list(command))
        assert command[command.index("--source-manifest-sha256") + 1] \
            == placeholder
        assert not manifest.exists()
        manifest.parent.mkdir(parents=True)
        manifest.write_text("b" * 64 + "  input.grib2\n")
        tree_root.mkdir(parents=True)

    monkeypatch.setattr(go_cli, "run_stage", stage)
    observer = SimpleNamespace(finish_stage=lambda **kwargs: None)
    arguments = dict(
        prep_root=prep_root, inputs=inputs, hints={}, geog_root=geog,
        manifest=None, cycle="2026-10-01_00:00:00", run_dir=run_dir,
        observer=observer, observe_stage=False)
    assert runplan._hrrr_hierarchy_stage(**arguments) == tree_root
    binding = json.loads((tree_root / stage_reuse.BINDING_NAME).read_text())
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert binding["stated"]["source_manifest_sha256"] == digest
    assert binding["arguments"]["--source-manifest-sha256"] == digest

    # A retry sees the completed root. Its instructions must agree with
    # the binding the first launch wrote after its seal, so preparation
    # reuse is decided from the same authority instead of the old plan.
    monkeypatch.setattr(boundary_stream, "live_chained_head", lambda root: None)

    def reuse(root, *, stated, arguments):
        assert root == tree_root
        assert stated == binding["stated"]
        assert stage_reuse.argument_binding(arguments) == binding["arguments"]
        return {"decision": stage_reuse.REUSE}

    monkeypatch.setattr(stage_reuse, "decide", reuse)
    assert runplan._hrrr_hierarchy_stage(**arguments) == tree_root
    assert len(executed) == 1


def test_a_hierarchy_interrupt_before_its_head_stops_the_waiting_root(
        tmp_path, monkeypatch):
    from test_boundary_stream import _snapshots
    from test_native_hrrr_chain import _tree_writer

    prep_root = tmp_path / "root"
    tree_root = tmp_path / "tree"
    stopped = []

    def preparation():
        # The writer checks for a stop before each segment it writes, and
        # the chain starts the hierarchy as soon as the root's head is out.
        # When the chain sees the head before the root reaches the wait
        # below (a loaded host scheduled it so on test host 2, 2026-10-02: the
        # test failed `assert stopped` in 0.03 s), the stop lands between
        # the head and segment 0 and the root stops inside its writer.
        # That is the root being stopped, so it is recorded here too.
        try:
            writer = _tree_writer(
                tmp_path, _snapshots(3), proof_name="proof.json", name="root",
                stop_after=0)
        except boundary_stream.BoundaryStreamStopped as error:
            stopped.append(str(error))
            raise
        # A real root waits for its stop or its seal with no limit; this
        # deadline only keeps a regression from hanging the suite.  At 5 s
        # it failed a loaded focused run (test host 4 swapping) with this test's
        # own signature, the interrupt reaching the root after its wait had
        # given up, which a hierarchy delayed by 6 s reproduces every time.
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            try:
                writer.check_stop()
            except boundary_stream.BoundaryStreamStopped as error:
                stopped.append(str(error))
                writer.fail(error)
                raise
            time.sleep(0.02)
        writer.fail(RuntimeError("root was not stopped"))
        raise AssertionError("the interrupted hierarchy left its root waiting")

    def hierarchy():
        assert not (boundary_stream.stream_dir(tree_root)
                    / boundary_stream.HEAD_NAME).exists()
        raise runplan.StageExitError("hierarchy", runplan.INTERRUPT_EXIT_CODE)

    observer = SimpleNamespace(finish_stage=lambda **kwargs: None, events=None)
    with pytest.raises(runplan.StageExitError) as raised:
        runplan._hrrr_tree_chain(
            prep_root=prep_root, tree_root=tree_root, preparation=preparation,
            hierarchy=hierarchy, config_path=tmp_path / "config.toml",
            forecast_dir=tmp_path / "forecast", observer=observer)
    assert raised.value.exit_code == runplan.INTERRUPT_EXIT_CODE
    assert stopped
    assert (boundary_stream.stream_dir(prep_root)
            / boundary_stream.STOP_NAME).is_file()
