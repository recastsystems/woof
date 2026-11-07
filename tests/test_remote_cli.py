"""Remote control never interpolates request data into a shell or trusts prose."""
from argparse import Namespace
import json
import os
from pathlib import Path
import shlex
import sys
import time

import pytest

from woof import remote_cli as rc


def args(**overrides):
    return Namespace(**{**dict(host="test-host", python="/opt/the model's runtime/bin/python",
        workspace="/srv/weather space", port=None, identity=None, ssh_config=None,
        remote_action="probe", json=True), **overrides})


@pytest.mark.parametrize("host", ["-oProxyCommand=x", "node;echo", "node\nnext", "$(secret)", "node x", "", "node`x`"])
def test_host_cannot_add_ssh_options_or_shell_words(host):
    with pytest.raises(ValueError, match="--host"):
        rc.ssh_command(args(host=host))


def test_only_quoted_python_and_fixed_protocol_words_enter_remote_shell(monkeypatch, tmp_path):
    config = tmp_path / "a config"
    config.write_text("Host test-host\n")
    key = tmp_path / "identity"
    key.write_bytes(b"fixture placeholder, not read by this test")
    monkeypatch.setattr(rc.shutil, "which", lambda *names, **options: "ssh-fixture")
    options = args(ssh_config=str(config), identity=str(key), port=2222)
    command = rc.ssh_command(options)
    assert command[-3:-1] == ["--", "test-host"]
    assert shlex.split(command[-1]) == [options.python, "-I", "-m", "woof.remote_worker", "--rpc"]
    assert options.workspace not in command[-1]
    assert "BatchMode=yes" in command and "StrictHostKeyChecking=yes" in command
    assert command[command.index("-F") + 1] == str(config.resolve())
    assert command[command.index("-i") + 1] == str(key.resolve())
    assert command[command.index("-p") + 1] == "2222"


def test_windows_prefers_the_system_openssh_client_and_the_record_names_it(monkeypatch, tmp_path):
    system = tmp_path / "Windows"
    openssh = system / "System32" / "OpenSSH" / "ssh.exe"
    openssh.parent.mkdir(parents=True)
    openssh.write_bytes(b"")
    git_bin = tmp_path / "Git" / "usr" / "bin"
    git_bin.mkdir(parents=True)
    (git_bin / "ssh.EXE").write_bytes(b"")
    environ = {"SystemRoot": str(system), "PATH": str(git_bin), "PATHEXT": ".EXE;.BAT"}
    assert rc.ssh_executable(environ=environ, windows=True) == str(openssh)
    # Without the system client, PATH order decides, as before.
    openssh.unlink()
    assert Path(rc.ssh_executable(environ=environ, windows=True)) == git_bin / "ssh.EXE"
    openssh.write_bytes(b"")
    # The resolved client rides on every record, including a refusal.
    monkeypatch.setattr(rc, "ssh_executable", lambda: str(openssh))
    command = rc.ssh_command(args())
    assert command[0] == str(openssh)


def test_records_and_ssh_level_failures_name_the_client_that_ran(monkeypatch, capsys, tmp_path):
    transport = rc._transport
    monkeypatch.setattr(rc, "ssh_executable", lambda: "C:/Windows/System32/OpenSSH/ssh.exe")
    monkeypatch.setattr(rc, "_transport", lambda command, request, **kwargs: rc.result("probe", runtime={}, capabilities={}))
    assert rc.remote_main(args()) == 0
    reply = json.loads(capsys.readouterr().out.splitlines()[0])
    assert reply["ok"] is True and reply["ssh_client"] == "C:/Windows/System32/OpenSSH/ssh.exe"
    # An authentication failure is an SSH-level failure: the node never
    # answered, and the message says which client asked.
    monkeypatch.setattr(rc, "_transport", transport)
    program = _program(tmp_path, "sys.stderr.write('Permission denied (publickey).\\n')\nraise SystemExit(255)\n")
    with pytest.raises(ValueError, match=r"Permission denied \(publickey\).*\(ssh client: ") as failure:
        rc._transport(program, {"action": "probe"})
    assert str(failure.value).endswith(f"(ssh client: {program[0]})")
    monkeypatch.setattr(rc, "ssh_command", lambda options, **kwargs: program)
    assert rc.remote_main(args()) == 2
    reply = json.loads(capsys.readouterr().out.splitlines()[0])
    assert reply["ok"] is False and reply["ssh_client"] == program[0]
    assert "Permission denied (publickey)" in reply["error"]["message"] and program[0] in reply["error"]["message"]


@pytest.mark.parametrize("field", ["ssh_config", "identity"])
def test_relative_ssh_config_and_identity_paths_are_refused(monkeypatch, tmp_path, field):
    (tmp_path / "sshconf").write_text("Host *\n  ProxyCommand planted\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rc.shutil, "which", lambda *names, **options: "ssh-fixture")
    with pytest.raises(ValueError, match="absolute local path"):
        rc.ssh_command(args(**{field: "sshconf"}))
    command = rc.ssh_command(args(**{field: str(tmp_path / "sshconf")}))
    assert command[command.index("-F" if field == "ssh_config" else "-i") + 1] == str((tmp_path / "sshconf").resolve())


@pytest.mark.parametrize("python,workspace,port", [("python", "/srv/x", None),
    ("/bin/python\nnext", "/srv/x", None), ("/bin/python", "relative", None),
    ("/bin/python", "/srv/x", 0), ("/bin/python", "/srv/x", 65536)])
def test_transport_profile_validation(monkeypatch, python, workspace, port):
    monkeypatch.setattr(rc.shutil, "which", lambda *names, **options: "ssh")
    with pytest.raises(ValueError):
        rc.ssh_command(args(python=python, workspace=workspace, port=port))


def test_windows_client_search_is_independent_of_host_platform(tmp_path):
    """Inspect Windows PATH/PATHEXT on either test host.

    shutil.which reads the path separator and PATHEXT from the interpreter's
    own platform rather than from its arguments, so ssh_executable(windows=True)
    used to answer for Windows in its first half and for the running host in
    its second. A Windows PATH is separated by ';' and its entries carry an
    extension from PATHEXT; neither survives a POSIX reading.
    """
    first = tmp_path / "Git" / "usr" / "bin"
    first.mkdir(parents=True)
    (first / "ssh.EXE").write_bytes(b"")
    second = tmp_path / "OpenSSH"
    second.mkdir()
    (second / "ssh.exe").write_bytes(b"")
    windows = {"PATH": ";".join([str(first), str(second)]), "PATHEXT": ".COM;.EXE"}
    assert rc.ssh_executable(environ=windows, windows=True) == str(first / "ssh.EXE")
    # PATHEXT decides which extensions count, and PATH order decides between them.
    assert rc.ssh_executable(environ={**windows, "PATHEXT": ".COM"}, windows=True) is None
    assert rc.ssh_executable(environ={**windows, "PATH": str(second)},
                             windows=True) == str(second / "ssh.exe")
    # An environment carrying no PATH has no PATH to search, on either platform.
    assert rc.ssh_executable(environ={}, windows=True) is None
    assert rc.ssh_executable(environ={}, windows=False) is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable lookup runs on POSIX hosts")
def test_native_posix_client_search_ignores_windows_root_and_extensions(tmp_path, monkeypatch):
    system = tmp_path / "Windows"
    system_client = system / "System32" / "OpenSSH" / "ssh.exe"
    system_client.parent.mkdir(parents=True)
    system_client.write_bytes(b"")
    posix_bin = tmp_path / "bin"
    posix_bin.mkdir()
    (posix_bin / "ssh").write_bytes(b"")
    (posix_bin / "ssh").chmod(0o755)
    windows_bin = tmp_path / "Git"
    windows_bin.mkdir()
    (windows_bin / "ssh.EXE").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    environ = {"PATH": "missing:bin", "SystemRoot": str(system), "PATHEXT": ".EXE"}
    assert Path(rc.ssh_executable(environ=environ, windows=False)).resolve() == posix_bin / "ssh"
    assert rc.ssh_executable(environ={**environ, "PATH": "Git"}, windows=False) is None


def test_a_typed_option_is_refused_for_itself_not_for_a_missing_client(monkeypatch, tmp_path):
    """Configuration is checked before this desktop is.

    ssh_command used to resolve the OpenSSH client between the workspace check
    and the port check, so on a machine without ssh a port of 65536 came back
    as "OpenSSH client 'ssh' is unavailable", which names neither the mistake
    nor the way out. The missing client answers only once the configuration is
    whole.
    """
    monkeypatch.setattr(rc, "ssh_executable", lambda: None)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sshconf").write_text("Host *\n")
    for options, message in ((args(port=65536), "--port must be between 1 and 65535"),
                             (args(port=0), "--port must be between 1 and 65535"),
                             (args(workspace="relative"), "--workspace"),
                             (args(python="python"), "--python"),
                             (args(host="node;echo"), "--host"),
                             (args(ssh_config="sshconf"), "absolute local path"),
                             (args(identity="sshconf"), "absolute local path")):
        with pytest.raises(ValueError, match=message):
            rc.ssh_command(options)
    # With nothing left to say about the configuration, the client refuses, and
    # that refusal names the breakage and the way out.
    with pytest.raises(ValueError, match="OpenSSH client 'ssh' is unavailable; install it and retry"):
        rc.ssh_command(args())
    # Both doors read one configuration through one function.
    profile = rc.transport_profile(args(port=2222))
    assert profile["host"] == "test-host" and profile["port"] == 2222


def _program(tmp_path, body):
    script = tmp_path / "ssh_fixture.py"
    script.write_text("import sys,json,time\nrequest=json.loads(sys.stdin.buffer.read())\n" + body, encoding="utf-8")
    return [sys.executable, "-u", str(script)]


def test_transport_preserves_request_and_parses_only_one_envelope(tmp_path):
    request = {"action": "start", "config": "/tmp/' $(unchanged) 日本語.toml"}
    command = _program(tmp_path, "print(json.dumps({'schema':'gpuwm.remote.result.v1','ok':True,'action':request['action'],'request':request}))\n")
    assert rc._transport(command, request)["request"] == request


@pytest.mark.parametrize("body,match", [
    ("print('plain prose')", "Expecting value"),
    ("print('{}\\n{}')", "one WOOF response"),
    ("print(json.dumps({'schema':'wrong','ok':True,'action':'probe'}))", "incompatible"),
    ("print(json.dumps({'schema':'gpuwm.remote.result.v1','ok':True,'action':'start'}))", "incompatible"),
    ("print(json.dumps({'schema':'gpuwm.remote.result.v1','ok':True,'action':'probe'}));sys.exit(7)", "disagrees"),
    ("sys.stderr.write('Permission denied (publickey).');sys.exit(255)", "Permission denied"),
    ("sys.stdout.write('x'*200000);sys.stdout.flush();time.sleep(10)", "bounded protocol"),
    ("sys.stderr.write('x'*20000)", "bounded protocol"),
])
def test_transport_refuses_protocol_and_ssh_errors(tmp_path, monkeypatch, body, match):
    if body == "sys.stderr.write('x'*20000)":
        # Check the post-exit bound separately from the live stdout overflow.
        popen = rc.subprocess.Popen
        def exited_transport(*args, **kwargs):
            process = popen(*args, **kwargs)
            process.poll = lambda: process.wait(timeout=10)
            return process
        monkeypatch.setattr(rc.subprocess, "Popen", exited_transport)
    with pytest.raises(ValueError, match=match):
        rc._transport(_program(tmp_path, body), {"action": "probe"})


def test_transport_timeout_does_not_claim_start_did_not_happen(tmp_path):
    with pytest.raises(ValueError) as failure:
        rc._transport(_program(tmp_path, "time.sleep(10)"), {"action": "start"}, timeout=.1)
    message = str(failure.value)
    assert "may exist" in message and "List this workspace's jobs" in message


def test_transport_timeout_names_the_attempt_and_the_flag_that_retries_it(tmp_path):
    """C-287: the reader is told the identity a retry must carry, not sent to a job list."""
    with pytest.raises(ValueError) as failure:
        rc._transport(_program(tmp_path, "time.sleep(10)"), {"action": "start", "request_id": "a1" * 16},
                      timeout=.1)
    message = str(failure.value)
    assert "a1" * 16 in message and "--request-id" in message and "already created" in message


def test_a_named_attempt_is_forwarded_verbatim_and_a_fresh_one_is_minted(monkeypatch, capsys):
    from woof.cli import build_parser
    options = build_parser().parse_args(["remote", "start", "--host", "node", "--python", "/opt/python",
        "--workspace", "/work", "--config", "/work/case.toml", "--outdir", "/work/out",
        "--request-id", "a1" * 16, "--json"])
    assert options.request_id == "a1" * 16
    monkeypatch.setattr(rc, "ssh_command", lambda options: ["ssh-fixture"])
    seen = []
    monkeypatch.setattr(rc, "_transport", lambda command, request, **kwargs:
                        seen.append(request) or rc.result(request["action"], dry_run=True))
    assert rc.remote_main(options) == 0
    assert seen[0]["request_id"] == "a1" * 16
    reply = json.loads(capsys.readouterr().out)
    assert reply["request_id"] == "a1" * 16, "the attempt rides on the reply so a retry can name it"
    fresh = build_parser().parse_args(["remote", "start", "--host", "node", "--python", "/opt/python",
        "--workspace", "/work", "--config", "/work/case.toml", "--outdir", "/work/out", "--json"])
    assert rc.remote_main(fresh) == 0
    assert len(seen[1]["request_id"]) == 32 and seen[1]["request_id"] != seen[0]["request_id"]
    capsys.readouterr()
    with pytest.raises(ValueError, match="32 hexadecimal characters"):
        rc.attempt_identity("../escape")


def test_a_node_that_keeps_saying_it_works_is_not_cut_off_at_the_deadline(tmp_path):
    """C-287: the deadline measures silence, never how long the work takes."""
    reply = json.dumps({"schema": rc.SCHEMA, "ok": True, "action": "start"})
    keepalive = json.dumps({"schema": rc.KEEPALIVE_SCHEMA, "action": "start"})
    program = _program(tmp_path, "import sys\n"
        "for _ in range(15):\n"
        f"    sys.stdout.write({keepalive!r} + chr(10)); sys.stdout.flush(); time.sleep(.3)\n"
        f"sys.stdout.write({reply!r} + chr(10)); sys.stdout.flush()")
    # The whole call outlasts the deadline; no single silence inside it does.
    # The interpreter's start is silence too, and on a loaded Windows host it
    # took over a second (the Windows CPU job's command failed here with a
    # 1 s deadline), so the deadline is 3 s and the program talks every 0.3 s
    # for 4.5 s.
    started = time.monotonic()
    value = rc._transport(program, {"action": "start"}, timeout=3)
    assert time.monotonic() - started > 3, "the call never outlasted the deadline"
    assert value["ok"] is True and value["action"] == "start"


def test_the_slow_actions_ask_the_node_to_keep_saying_it_is_alive(monkeypatch):
    seen = []
    monkeypatch.setattr(rc, "ssh_command", lambda options: ["ssh-fixture"])
    monkeypatch.setattr(rc, "_transport", lambda command, request, **kwargs:
                        seen.append((request, kwargs)) or rc.result(request["action"], dry_run=True))
    options = args(remote_action="resume", job="valid_123", outdir="/srv/new output", json=False)
    assert rc.remote_main(options) == 0
    request, kwargs = seen[0]
    assert request["keepalive"] is True and kwargs["timeout"] == rc.REVIEW_SECONDS
    assert rc.REVIEW_SECONDS > 120


def test_refusal_is_one_json_line_with_exit_two(monkeypatch, capsys):
    assert rc.remote_main(args(host="-oops")) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    reply = json.loads(lines[0])
    assert reply["schema"] == rc.SCHEMA and reply["action"] == "probe" and reply["ok"] is False


def test_cli_forwards_binding_and_paths_only_as_request_data(monkeypatch, capsys):
    monkeypatch.setattr(rc, "ssh_command", lambda options: ["ssh-fixture"])
    seen = []
    monkeypatch.setattr(rc, "_transport", lambda command, request, **kwargs: seen.append((command, request)) or rc.result("start", dry_run=True))
    options = args(remote_action="start", config="/srv/a path.toml", outdir="/srv/new output",
                   geog_root="/srv/geography", products="t2,wind10", dry_run=True,
                   expected_input_sha256="1" * 64)
    assert rc.remote_main(options) == 0
    assert seen[0][0] == ["ssh-fixture"]
    assert seen[0][1]["expected_input_sha256"] == "1" * 64
    assert seen[0][1]["config"] == options.config
    assert seen[0][1]["products"] == options.products
    # The client names its own launch attempt, so a retry can reconcile.
    assert len(seen[0][1]["request_id"]) == 32
    assert len(capsys.readouterr().out.splitlines()) == 1


@pytest.mark.parametrize("action,section", [
    ("start", "40,-100,41,-99"), ("resume", "-40,100,-41,99"),
])
def test_section_line_reaches_the_remote_request(action, section, monkeypatch, capsys):
    from woof.cli import build_parser
    identity = ["--config", "/srv/config.toml"] if action == "start" else ["--job", "old-job"]
    options = build_parser().parse_args([
        "remote", action, "--host", "node", "--python", "/opt/python",
        "--workspace", "/work", "--outdir", "/new-output", *identity,
        "--products", "xsec:wa", "--section=" + section, "--json"])
    seen = []
    monkeypatch.setattr(rc, "ssh_command", lambda options: ["ssh-fixture"])
    monkeypatch.setattr(rc, "_transport", lambda command, request, **kwargs:
                        seen.append(request) or rc.result(action))
    assert rc.remote_main(options) == 0
    assert seen[0]["section"] == section
    assert seen[0]["products"] == "xsec:wa"
    capsys.readouterr()


def test_public_parser_has_review_and_reconnect_options():
    from woof.cli import build_parser
    parser = build_parser()
    options = parser.parse_args(["remote", "resume", "--host", "node", "--python", "/opt/python",
        "--workspace", "/work", "--job", "valid_123", "--outdir", "/new-output", "--from", "latest",
        "--dry-run", "--expected-input-sha256", "a" * 64, "--json"])
    assert options.func is rc.remote_main
    assert options.from_checkpoint == "latest" and options.dry_run and options.json


def test_resume_takes_the_same_inputs_a_start_takes():
    """C-279: a resume that names an input reaches the node instead of argparse."""
    from woof.cli import build_parser
    options = build_parser().parse_args(["remote", "resume", "--host", "node", "--python", "/opt/python",
        "--workspace", "/work", "--job", "valid_123", "--outdir", "/new-output",
        "--geog-root", "/srv/geography", "--prepared-root", "/srv/prepared",
        "--wps-namelist", "/srv/prepared/namelist.wps", "--products", "t2,wind10"])
    assert options.geog_root == "/srv/geography" and options.prepared_root == "/srv/prepared"
    assert options.wps_namelist == "/srv/prepared/namelist.wps" and options.products == "t2,wind10"
    # Every one of them is a key the node's own request door already allows, so
    # the door and this parser agree about what a resume may carry.
    from woof import remote_worker as rw
    with pytest.raises(ValueError) as failure:
        rw.dispatch({"schema": "gpuwm.remote.request.v1", "action": "resume", "workspace": "/work",
                     "job": "valid_123", "outdir": "/new-output", "geog_root": "/srv/geography",
                     "prepared_root": "/srv/prepared", "wps_namelist": "/srv/p/namelist.wps",
                     "products": "t2,wind10"})
    assert "unsupported remote request fields" not in str(failure.value)


def test_a_named_resume_input_reaches_the_node_request(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(rc.shutil, "which", lambda _, path=None: "ssh-fixture")
    monkeypatch.setattr(rc, "_transport", lambda command, request, **kwargs:
                        seen.append(request) or {"schema": rc.SCHEMA, "ok": True, "action": request["action"]})
    options = args(remote_action="resume", job="valid_123", outdir="/srv/new output",
                   geog_root="/srv/geography", prepared_root="/srv/prepared",
                   wps_namelist="/srv/prepared/namelist.wps", products="t2,wind10")
    assert rc.remote_main(options) == 0
    assert seen[0]["geog_root"] == "/srv/geography" and seen[0]["prepared_root"] == "/srv/prepared"
    assert seen[0]["wps_namelist"] == "/srv/prepared/namelist.wps" and seen[0]["products"] == "t2,wind10"
    capsys.readouterr()


def test_artifact_parser_and_fixed_binary_stream_keep_selectors_off_the_shell(monkeypatch):
    from woof.cli import build_parser
    options = build_parser().parse_args(["remote", "sync-artifacts", "--host", "node", "--python", "/opt/python",
        "--workspace", "/owned/work", "--job", "job-fixture", "--domain", "2", "--cache-root", "C:/owned/cache", "--json"])
    assert options.domain == 2 and options.cache_root == "C:/owned/cache"
    monkeypatch.setattr(rc.shutil, "which", lambda *names, **options: "ssh-fixture")
    command = rc.ssh_command(options, artifact_stream=True)
    assert shlex.split(command[-1]) == [options.python, "-I", "-m", "woof.remote_worker", "--artifact-stream"]
    assert options.job not in command[-1] and options.cache_root not in command[-1]
