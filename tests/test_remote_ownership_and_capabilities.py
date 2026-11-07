"""One ownership provider and one capability statement, read by every door.

CPU-only: no process is started and no card is opened.
"""
import pytest

from woof import remote_input_transfer as transfer, remote_worker as rw


def test_every_door_refuses_an_unsupported_platform_with_one_sentence(monkeypatch, capsysbinary):
    """Five doors in this package ask the platform question; one sentence answers it."""
    import json
    from woof import (remote_artifacts as artifacts, remote_processed as store,
                       remote_processed_v2 as viewer)
    monkeypatch.setattr(rw.sys, "platform", "win32")
    with pytest.raises(ValueError) as failure:
        rw.dispatch({"schema": "gpuwm.remote.request.v1", "action": "status",
                     "workspace": "/work", "job": "x"})
    worker = str(failure.value)
    # Each stream door refuses before it reads a byte of the request.
    assert transfer.receive_main() == 2
    reply = json.loads(capsysbinary.readouterr().out.decode())
    assert reply["ok"] is False
    assert reply["error"]["message"] == worker, "both doors refuse one configuration with one sentence"
    for module, prefix in ((artifacts, "remote artifact: "),
                           (store, "remote native store: "),
                           (viewer, "remote native viewer: ")):
        assert module.stream_main() == 2, module.__name__
        printed = capsysbinary.readouterr().err.decode().strip()
        assert printed == prefix + worker, module.__name__ + " states that one sentence too"
    assert "win32" in worker
    assert "/proc" in worker
    assert "register an ownership provider for this platform" in worker
    assert "linux" in worker


def test_linux_passes_both_doors_unchanged(monkeypatch):
    assert rw.sys.platform == "linux"
    assert rw._ownership_provider()["signal"] == "Linux pidfd"


def test_the_ownership_provider_is_a_table_row_not_a_branch():
    assert set(rw.OWNERSHIP_PROVIDERS) == {"linux"}
    assert set(rw.OWNERSHIP_PROVIDERS["linux"]) == {"identity", "membership", "descendants", "signal"}


def test_the_node_states_its_capabilities_on_every_reply_a_client_keeps(tmp_path, monkeypatch):
    from woof import remote_plan as rp
    declared = rw.capabilities()
    assert declared["artifact_index_v1"] is True and declared["artifact_sequence_v1"] is True
    assert declared["process_handles"] == rw.OWNERSHIP_PROVIDERS["linux"]["signal"]
    monkeypatch.setattr(rp, "hardware_probe", lambda *_a, **_k: {"devices": []})
    reply = rw.dispatch({"schema": "gpuwm.remote.request.v1", "action": "probe",
                         "workspace": str(tmp_path)})
    assert reply["capabilities"] == declared


def test_a_status_reply_carries_the_artifact_capabilities_too(tmp_path):
    store = tmp_path / ".arwen-jobs"
    store.mkdir(mode=0o700)
    directory = store / "20260907T180000-0123456789abcdef"
    directory.mkdir(mode=0o700)
    saved = directory / "case.toml"
    saved.write_text("[experiment]\n")
    rw._write(directory / "job.json", {
        "schema": "gpuwm.remote.job.v1", "id": directory.name, "token": "a" * 64,
        "created_at": "2026-09-07T17:59:59+00:00", "action": "start", "config": str(saved),
        "outdir": str(tmp_path / "run"), "runtime": {"version": "2.7.4"}, "snapshot_plan": None,
        "snapshot_config": str(saved), "snapshot_sha256": rw._file_sha(saved),
        "config_sha256": rw._file_sha(saved)})
    job = rw._status(directory)
    declared = rw.capabilities()
    for key in ("processed_frame_v2", "processed_member_stream_v2",
                "artifact_sync_v1", "artifact_index_v1", "artifact_sequence_v1"):
        assert job["viewer_capabilities"][key] == declared[key]


def test_an_old_node_refusal_names_its_version_and_the_missing_capability(tmp_path, monkeypatch):
    """The client produces the sentence once, from facts both sides have."""
    import woof
    from argparse import ArgumentParser
    from woof import remote_cli as rc, remote_plan as rp

    config = tmp_path / "saved map.toml"
    from datetime import datetime
    from woof import domain_wizard as dw
    config.write_text(dw.render_config(name="remote-map", start_time=datetime(2026, 9, 7, 18),
        hours=24, projection=dw._projection_entries(18, -162.1, "auto"),
        dims=dw._dims_for_scale(1, ()), ratios=(),
        fetch_hints=dict(source="gfs", cycle="2026-09-07T12", forecast_start_hour=6, hours=24,
                         out=str(tmp_path / "forcing"), cadence=3), case_data=None), encoding="utf-8")
    import hashlib
    import json as _json
    plan = tmp_path / "plan.json"
    plan.write_text(_json.dumps({"schema": "gpuwm.run-plan.v1", "name": "reviewed-map",
        "route": "prepared", "config": {"path": str(config)},
        "output_root": str(tmp_path / "local-output"), "run_options": {"render_products": "none"}}),
        encoding="utf-8")

    parser = ArgumentParser()
    rc.register_cli(parser.add_subparsers())
    args = parser.parse_args(["remote", "review-plan", "--host", "test-host-1", "--python", "/node/python",
        "--workspace", "/node/work", "--plan", str(plan), "--outdir", "/node/work/new",
        "--expected-plan-sha256", hashlib.sha256(plan.read_bytes()).hexdigest(),
        "--expected-config-sha256", hashlib.sha256(config.read_bytes()).hexdigest(), "--json"])
    monkeypatch.setattr(rc, "ssh_command", lambda *_a, **_k: ["never-executed"])
    probes = []

    def transport(_command, request, **_kwargs):
        if request["action"] == "stage-plan":
            return rc.result("stage-plan", ok=False,
                             error={"type": "ValueError", "message": "unsupported remote action"})
        probes.append(request)
        return rc.result("probe", runtime={"version": "2.7.1"}, capabilities={"durable_jobs": True})

    monkeypatch.setattr(rc, "_transport", transport)
    import io
    import contextlib
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert rc.remote_main(args) != 0
    message = _json.loads(out.getvalue())["error"]["message"]
    assert "2.7.1" in message
    assert woof.__version__ in message
    assert "review_plan_v1" in message
    assert "unsupported remote action" in message
    assert [request["action"] for request in probes] == ["probe"]


@pytest.mark.parametrize("action, refusal, capability", [
    ("start", "unsupported remote request fields: keepalive, request_id", "keepalive_v1"),
    ("resume", "unsupported remote request fields: request_id", "launch_attempt_v1"),
    ("review-plan", "unsupported staged-plan request fields: keepalive", "keepalive_v1"),
])
def test_a_node_too_old_for_the_keepalive_or_the_named_attempt_is_named_with_the_way_out(
        tmp_path, monkeypatch, action, refusal, capability):
    """The fields this client sends at every slow door are capabilities a node declares.

    A node that does not pop the keepalive or record the attempt refuses the
    request with a bare field list; the reader gets the node's version, the
    capability that serves the field, the client version to update to and that
    the same request can be retried after the update.
    """
    import contextlib
    import hashlib
    import io
    import json as _json
    import woof
    from argparse import ArgumentParser
    from woof import remote_cli as rc

    parser = ArgumentParser()
    rc.register_cli(parser.add_subparsers())
    common = ["--host", "test-host-1", "--python", "/node/python", "--workspace", "/node/work", "--json"]
    if action == "review-plan":
        from datetime import datetime
        from woof import domain_wizard as dw
        config = tmp_path / "saved map.toml"
        config.write_text(dw.render_config(name="remote-map", start_time=datetime(2026, 9, 7, 18),
            hours=24, projection=dw._projection_entries(18, -162.1, "auto"),
            dims=dw._dims_for_scale(1, ()), ratios=(),
            fetch_hints=dict(source="gfs", cycle="2026-09-07T12", forecast_start_hour=6, hours=24,
                             out=str(tmp_path / "forcing"), cadence=3), case_data=None), encoding="utf-8")
        plan = tmp_path / "plan.json"
        plan.write_text(_json.dumps({"schema": "gpuwm.run-plan.v1", "name": "reviewed-map",
            "route": "prepared", "config": {"path": str(config)},
            "output_root": str(tmp_path / "local-output"), "run_options": {"render_products": "none"}}),
            encoding="utf-8")
        argv = ["remote", action, *common, "--plan", str(plan), "--outdir", "/node/work/new",
                "--expected-plan-sha256", hashlib.sha256(plan.read_bytes()).hexdigest(),
                "--expected-config-sha256", hashlib.sha256(config.read_bytes()).hexdigest()]
    elif action == "resume":
        argv = ["remote", action, *common, "--job", "valid_123", "--outdir", "/node/work/new"]
    else:
        argv = ["remote", action, *common, "--config", "/node/work/case.toml", "--outdir", "/node/work/new"]
    args = parser.parse_args(argv)
    monkeypatch.setattr(rc, "ssh_command", lambda *_a, **_k: ["never-executed"])
    seen = []

    def transport(_command, request, **_kwargs):
        seen.append(request["action"])
        if request["action"] == "probe":
            # A 2.7.4 node: it serves staged plans but declares neither protocol fact.
            return rc.result("probe", runtime={"version": "2.7.4"},
                             capabilities={"durable_jobs": True, "review_plan_v1": True, "start_plan_v1": True})
        return rc.result(request["action"], ok=False, error={"type": "ValueError", "message": refusal})

    monkeypatch.setattr(rc, "_transport", transport)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert rc.remote_main(args) == 2
    message = _json.loads(out.getvalue())["error"]["message"]
    assert "2.7.4" in message
    assert f"capability {capability}" in message
    assert f"Update it to {woof.__version__} and retry this request as it was" in message
    assert refusal in message
    assert seen.count("probe") == 1


def test_a_current_node_declares_the_protocol_facts_the_client_depends_on():
    """The verdict above is keyed on capabilities the node states on every reply it keeps."""
    from woof import remote_cli as rc
    declared = rw.capabilities()
    assert declared["keepalive_v1"] is True and declared["launch_attempt_v1"] is True
    for capability in ("keepalive_v1", "launch_attempt_v1", "review_plan_v1", "start_plan_v1"):
        assert capability in rc._CAPABILITY_WORDS
