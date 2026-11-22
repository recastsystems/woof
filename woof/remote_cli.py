"""OpenSSH client for durable WOOF-ArWen jobs on an existing Linux installation."""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time

SCHEMA = "gpuwm.remote.result.v1"
KEEPALIVE_SCHEMA = "gpuwm.remote.keepalive.v1"
MAX_REPLY = 128 * 1024
MAX_REQUEST = 128 * 1024
#: What a reviewing or launching call is allowed to take. A memory review, a
#: staged transfer and a launch are minutes of real work on the node, and the
#: node says it is still working every few seconds, so this deadline measures
#: silence rather than duration.
REVIEW_SECONDS = 900
#: The actions that review, stage or launch, and therefore take that deadline
#: and ask the node to keep saying it is alive.
SLOW_ACTIONS = ("start", "resume", "start-plan", "review-plan", "stage-plan")
#: The actions that create a job, and therefore name their attempt.
LAUNCH_ACTIONS = ("start", "resume", "start-plan")
REQUEST_ID = re.compile(r"[0-9a-f]{32}\Z")


def attempt_identity(named=None):
    """This launch attempt's own name: the reader's, or a fresh one."""
    if named is None:
        return os.urandom(16).hex()
    if not isinstance(named, str) or not REQUEST_ID.fullmatch(named):
        raise ValueError("--request-id names a launch attempt as 32 hexadecimal characters, the "
                         "identity a previous reply or timeout printed; a fresh invocation may omit it.")
    return named
ACTIONS = ("probe", "start", "list", "status", "logs", "stop", "resume", "review-plan", "start-plan", "sync-artifacts", "artifact-index", "sync-processed-frame", "sync-processed-frame-v2", "sync-native-plots", "sync-outputs", "list-products")


def result(action: str, *, ok: bool = True, **data) -> dict:
    return {"schema": SCHEMA, "ok": ok, "action": action, **data}


def remote_path(value, name: str) -> str:
    if (not isinstance(value, str) or not value or "\x00" in value
            or any(ord(char) < 32 for char in value)
            or not PurePosixPath(value).is_absolute()):
        raise ValueError(f"{name} must be an absolute Linux path")
    return value


WINDOWS_EXTENSIONS = ".COM;.EXE;.BAT;.CMD"


def _path_client(name: str, *, environ, windows: bool) -> str | None:
    """Find `name` on the PATH in `environ` under the rules of `windows`.

    Windows PATH and PATHEXT can be inspected without changing the process
    environment. POSIX lookup uses the host's native executable permissions
    through shutil.which; a live launch always selects its current platform.
    """
    search = environ.get("PATH")
    if not search:
        return None
    if not windows:
        return shutil.which(name, path=search)
    extensions = [suffix for suffix in
                  (environ.get("PATHEXT") or WINDOWS_EXTENSIONS).split(";") if suffix]
    wanted = [name] if any(name.lower().endswith(suffix.lower()) for suffix in extensions) else []
    wanted += [name + suffix for suffix in extensions]
    for entry in search.split(";"):
        if not entry:
            continue
        try:
            # Windows matches a filename without regard to case.
            present = {item.name.lower(): item for item in Path(entry).iterdir() if item.is_file()}
        except OSError:
            continue
        for candidate in wanted:
            found = present.get(candidate.lower())
            if found is not None:
                return str(found)
    return None


def ssh_executable(*, environ=None, windows=None) -> str | None:
    """Prefer Windows' own OpenSSH client, then PATH.

    PATH order on a developer desktop puts Git's MSYS ssh.EXE first; it cannot
    reach the Windows OpenSSH agent service, so a passphrase-protected key
    fails under BatchMode=yes with an opaque "Permission denied (publickey)".
    PATHEXT also means an ssh.bat earlier on PATH would win.

    The environment and the platform are arguments, so both halves of the
    choice, the system client and the PATH search, answer for the same
    platform. An environment carrying no PATH has no PATH to search.
    """
    environ = os.environ if environ is None else environ
    windows = (os.name == "nt") if windows is None else windows
    if windows:
        system_root = environ.get("SystemRoot") or environ.get("SYSTEMROOT") or environ.get("WINDIR")
        if system_root:
            candidate = Path(system_root) / "System32" / "OpenSSH" / "ssh.exe"
            if candidate.is_file():
                return str(candidate)
    return _path_client("ssh", environ=environ, windows=windows)


def transport_profile(args) -> dict:
    """Check every typed transport option, before anything about this desktop.

    One function, so review-plan and a live run cannot disagree about one
    configuration, and so a typed option is refused for what is wrong with it.
    Resolving the client first made a missing OpenSSH install answer for a port
    of 65536, which names neither the breakage nor the way out.
    """
    host = args.host
    if (not isinstance(host, str) or len(host) > 255
            or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:\[\]-]*", host)):
        raise ValueError("--host must be an SSH alias or user@host without shell syntax")
    python = remote_path(args.python, "--python")
    remote_path(args.workspace, "--workspace")
    if args.port is not None and not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    files = []
    for flag, value in (("-F", args.ssh_config), ("-i", args.identity)):
        if value is not None:
            path = Path(value).expanduser()
            if not path.is_absolute():
                # ssh runs a ProxyCommand from -F before any host-key check; a
                # relative path would let the current folder supply that file.
                raise ValueError(f"{flag} must be an absolute local path, not one relative to the current folder")
            path = path.resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"{flag} must name an existing local file")
            files.append((flag, str(path)))
    return {"host": host, "python": python, "port": args.port, "files": files}


def ssh_command(args, *, artifact_stream=False, input_stream=False, processed_stream=False, processed_member_stream_v2=False) -> list[str]:
    profile = transport_profile(args)
    host, python = profile["host"], profile["python"]
    ssh = ssh_executable()
    if ssh is None:
        raise ValueError("OpenSSH client 'ssh' is unavailable; install it and retry")
    command = [ssh, "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10",
               "-o", "ServerAliveCountMax=2"]
    if profile["port"] is not None:
        command += ["-p", str(profile["port"])]
    for flag, value in profile["files"]:
        command += [flag, value]
    # ssh passes its command through the remote shell. Only these fixed words
    # and the quoted interpreter enter that shell; request data travels on stdin.
    command += ["--", host, shlex.join([python, "-I", "-m", "woof.remote_worker",
                                     "--input-stream" if input_stream else "--processed-member-stream-v2" if processed_member_stream_v2 else "--processed-stream" if processed_stream else "--artifact-stream" if artifact_stream else "--rpc"])]
    return command


def _transport(command: list[str], request: dict, *, timeout: float = 40) -> dict:
    payload = (json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if len(payload) > MAX_REQUEST:
        raise ValueError("remote request exceeds 128 KiB")
    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    chunks: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    overflow = threading.Event()

    def read(name, stream, maximum):
        # read1 returns what has arrived rather than waiting for a full buffer,
        # which is what makes a node's keepalive visible while it is still
        # working instead of only when its pipe closes.
        read_available = getattr(stream, "read1", stream.read)
        try:
            while block := read_available(4096):
                room = maximum + 1 - len(chunks[name])
                chunks[name].extend(block[:max(0, room)])
                if len(chunks[name]) > maximum:
                    overflow.set()
        finally:
            stream.close()

    readers = [threading.Thread(target=read, args=(name, getattr(process, name), maximum),
                                daemon=True)
               for name, maximum in (("stdout", MAX_REPLY), ("stderr", 16384))]
    for reader in readers:
        reader.start()
    def write_request():
        try:
            process.stdin.write(payload)
            process.stdin.flush()
        except (OSError, ValueError):
            pass  # the SSH exit and protocol response own the diagnostic
        finally:
            process.stdin.close()

    writer = threading.Thread(target=write_request, daemon=True)
    writer.start()
    try:
        deadline, seen = time.monotonic() + timeout, 0
        while process.poll() is None:
            if overflow.is_set():
                raise ValueError("remote response exceeds its bounded protocol limit")
            if len(chunks["stdout"]) != seen:
                # The node said something, so it is working: the deadline
                # measures silence from this node, never how long its work takes.
                seen, deadline = len(chunks["stdout"]), time.monotonic() + timeout
            if time.monotonic() >= deadline:
                attempt = request.get("request_id")
                named = (f" This attempt is {attempt}: retry the same request with --request-id "
                         f"{attempt} and the node answers with the job it already created for it, "
                         "instead of starting a second one." if attempt else
                         " List this workspace's jobs before retrying.")
                raise ValueError(f"This node said nothing for {timeout:g} seconds. A start it had "
                                 "already accepted may exist." + named)
            time.sleep(.02)
        for reader in readers:
            reader.join(timeout=2)
        if overflow.is_set() or any(reader.is_alive() for reader in readers):
            raise ValueError("remote response exceeds its bounded protocol limit or did not close")
        text = bytes(chunks["stdout"]).decode("utf-8", errors="strict")
        lines = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except ValueError:
                parsed = None
            # A keepalive is not an answer; it is the node saying it is working.
            if isinstance(parsed, dict) and parsed.get("schema") == KEEPALIVE_SCHEMA:
                continue
            lines.append(line)
        if len(lines) != 1:
            detail = bytes(chunks["stderr"]).decode("utf-8", errors="replace").strip()
            # The node never answered, so the client that ran is the fact the
            # user needs: which ssh spoke, and therefore which agent and keys.
            raise ValueError("SSH did not return one WOOF-ArWen response" + (f": {detail[:2000]}" if detail else "")
                             + f" (ssh client: {command[0]})")
        reply = json.loads(lines[0])
        if (not isinstance(reply, dict) or reply.get("schema") != SCHEMA
                or type(reply.get("ok")) is not bool
                or reply.get("action") != request["action"]):
            raise ValueError("SSH returned an incompatible WOOF-ArWen response")
        if process.returncode != (0 if reply["ok"] else 2):
            raise ValueError(f"SSH exit {process.returncode} disagrees with the WOOF-ArWen response")
        return reply
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        writer.join(timeout=2)


#: What a node says when it does not serve a capability at all, rather than
#: when it refused a request it understood.
_MISSING_CAPABILITY_WORDS = ("unsupported remote action", "unsupported staged-plan request fields",
                             "unsupported remote request fields", "unsupported remote request schema")

#: What each capability a client depends on serves, in the words the verdict
#: uses, and the way forward beside updating the node when one exists.
_CAPABILITY_WORDS = {
    "review_plan_v1": ("staged map plans",
                       ", or run this setup from a configuration that already exists on the node"),
    "start_plan_v1": ("launching a staged map plan",
                      ", or run this setup from a configuration that already exists on the node"),
    "keepalive_v1": ("the keepalive this client asks for on a slow request, so its deadline "
                     "measures the node's silence rather than how long its work takes", ""),
    "launch_attempt_v1": ("a launch attempt named by the client (--request-id), which is what "
                          "lets a retry reconcile with the job already created", ""),
}

#: The request field a node refused, mapped to the capability that serves it.
_FIELD_CAPABILITIES = {"keepalive": "keepalive_v1", "request_id": "launch_attempt_v1"}

#: The capability a slow action depends on when the node names no field at all.
_ACTION_CAPABILITIES = {"review-plan": "review_plan_v1", "stage-plan": "review_plan_v1",
                        "start-plan": "start_plan_v1"}


def _protocol_capability(action, message):
    """Which capability a bare protocol refusal is about.

    A refusal that lists the fields it did not understand names the capability
    that serves the first such field this client depends on; one that names no
    field is about the action itself.
    """
    if "request fields:" in message:
        listed = message.split("request fields:", 1)[1]
        for field, capability in _FIELD_CAPABILITIES.items():
            if field in {word.strip() for word in listed.split(",")}:
                return capability
    return _ACTION_CAPABILITIES.get(action, "durable_jobs")


def _capability_verdict(command, workspace, error, capability):
    """Turn a node's bare refusal into one sentence that names what is missing.

    The node is asked for its own runtime version rather than having one
    guessed for it, so the reader is told which node this is, which capability
    it does not serve, and the way forward: update the node to this client's
    version and retry the same request. The node's own words are kept.
    """
    if not isinstance(error, dict):
        return error
    message = str(error.get("message", ""))
    if not any(word in message for word in _MISSING_CAPABILITY_WORDS):
        return error
    from woof import __version__ as client_version
    version = None
    try:
        probe = _transport(command, {"schema": "gpuwm.remote.request.v1", "action": "probe",
                                     "workspace": workspace})
        if probe.get("ok") and (probe.get("capabilities") or {}).get(capability) is True:
            # The node does serve it; this refusal is about something else.
            return error
        version = (probe.get("runtime") or {}).get("version")
    except (OSError, ValueError, KeyError):
        version = None
    named = f"WOOF-ArWen {version}" if version else "a WOOF-ArWen this client could not identify"
    serves, alternative = _CAPABILITY_WORDS.get(capability, (f"this request (capability {capability})", ""))
    return {**error, "message": (
        f"This node runs {named}, which does not serve {serves} (capability {capability}). "
        f"Update it to {client_version} and retry this request as it was{alternative}. "
        f"The node said: {message[:1000]}")}


def remote_main(args) -> int:
    action = args.remote_action
    ssh_client = None
    request = None
    command = None
    try:
        command = ssh_command(args)
        ssh_client = command[0]
        request = {"schema": "gpuwm.remote.request.v1", "action": action,
                   "workspace": args.workspace}
        if action in LAUNCH_ACTIONS:
            # This client names its own launch attempt. A fresh invocation is a
            # fresh attempt, so running one configuration again is a new run;
            # a retry that names the attempt reconciles with the job it created.
            request["request_id"] = attempt_identity(getattr(args, "request_id", None))
        for name in ("config", "outdir", "geog_root", "prepared_root", "wps_namelist", "products", "section", "device", "job", "cursor",
                     "limit", "from_checkpoint", "dry_run", "expected_config_sha256",
                     "expected_wps_sha256", "expected_input_sha256", "expected_checkpoint_sha256",
                     "expected_checkpoint_set_sha256", "expected_prepared_sha256", "bundle_id",
                     "expected_bundle_sha256", "expected_plan_sha256"):
            value = getattr(args, name, None)
            if value is not None:
                request[name] = value
        if action == "sync-artifacts":
            from woof.remote_artifacts import sync
            reply = result(action, **sync(args, command, ssh_command(args, artifact_stream=True)))
        elif action == "sync-processed-frame":
            from woof.remote_processed import sync
            reply = result(action, **sync(args, command, ssh_command(args, processed_stream=True)))
        elif action == "sync-processed-frame-v2":
            from woof.remote_processed_cache_v2 import sync
            reply = result(action, **sync(args, command, ssh_command(args, processed_member_stream_v2=True)))
        elif action == "artifact-index":
            from woof.remote_artifacts import index
            reply = result(action, **index(args, command))
        elif action == "sync-native-plots":
            from woof.remote_native_plots import sync
            reply = result(action, **sync(args, command, ssh_command(args, artifact_stream=True)))
        elif action == "sync-outputs":
            from woof.remote_artifacts import sync_outputs
            reply = result(action, **sync_outputs(args, command, ssh_command(args, artifact_stream=True)))
        elif action == "review-plan":
            from woof.remote_plan import build_bundle
            from woof.remote_input_transfer import transfer_bundle_inputs, source_blobs, verify_sources
            bundle = build_bundle(args.plan, workspace=args.workspace, outdir=args.outdir,
                                  geog_root=args.geog_root,
                                  prepared_root=getattr(args, "prepared_root", None),
                                  wps_namelist=getattr(args, "wps_namelist", None),
                                  restart=getattr(args, "restart", None),
                                  device=getattr(args, "device", None),
                                  expected_plan_sha256=args.expected_plan_sha256,
                                  expected_config_sha256=args.expected_config_sha256)
            if bundle.get("blobs"):
                transfer_bundle_inputs(bundle, command, ssh_command(args, input_stream=True))
                verify_sources(source_blobs(bundle))
            staged = _transport(command, {"schema": request["schema"], "action": "stage-plan",
                                          "workspace": args.workspace, "bundle": bundle,
                                          "keepalive": True}, timeout=REVIEW_SECONDS)
            if not staged["ok"]:
                reply = result(action, ok=False, error=staged["error"])
            else:
                if staged.get("bundle_id") != bundle["id"] or staged.get("bundle_sha256") != bundle["sha256"]:
                    raise ValueError("node staged a different plan bundle")
                reply = _transport(command, {"schema": request["schema"], "action": action,
                    "workspace": args.workspace, "bundle_id": bundle["id"],
                    "expected_bundle_sha256": bundle["sha256"], "keepalive": True},
                    timeout=REVIEW_SECONDS)
                if reply["ok"] and bundle.get("blobs"):
                    verify_sources(source_blobs(bundle))
        else:
            if action == "start-plan" and getattr(args, "source_inputs_file", None):
                from woof.remote_input_transfer import verify_review_file
                request["expected_source_blobs_sha256"] = verify_review_file(args.source_inputs_file)
            reply = (_transport(command, {**request, "keepalive": True}, timeout=REVIEW_SECONDS)
                     if action in SLOW_ACTIONS else _transport(command, request))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        reply = result(action, ok=False, error={"type": type(error).__name__, "message": str(error)})
    if action in SLOW_ACTIONS and not reply["ok"] and command is not None and isinstance(reply.get("error"), dict):
        # A node too old for what this client sent at a slow door (the
        # keepalive, the named attempt, a staged plan) refuses with a bare
        # protocol sentence; the reader gets the node's version, the missing
        # capability and the way out instead.
        reply["error"] = _capability_verdict(command, args.workspace, reply["error"],
                                             _protocol_capability(action, str(reply["error"].get("message", ""))))
    # Every record (status, plan review, refusal) names the ssh client that ran.
    reply["ssh_client"] = ssh_client
    if action in LAUNCH_ACTIONS and request is not None and request.get("request_id"):
        # The attempt rides on the reply, refusals included, so a reader who
        # lost the connection can retry it by name.
        reply["request_id"] = request["request_id"]
    if args.json:
        output = json.dumps(reply, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        if hasattr(sys.stdout, "buffer"):
            sys.stdout.buffer.write(output.encode("utf-8"))
            sys.stdout.buffer.flush()
        else:
            sys.stdout.write(output)
    elif not reply["ok"]:
        print("remote: " + reply["error"]["message"], file=sys.stderr)
    elif action == "logs":
        print(reply.get("text", ""), end="")
    else:
        print(json.dumps(reply, ensure_ascii=False, indent=2))
    return 0 if reply["ok"] else 2


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser("remote", help="control durable WOOF-ArWen jobs on an existing Linux SSH node")
    actions = parser.add_subparsers(dest="remote_action", required=True)
    for action in ACTIONS:
        command = actions.add_parser(action, help={
            "probe": "show the remote interpreter and installed WOOF-ArWen identity",
            "start": "review or start a forecast using existing remote inputs",
            "list": "list recorded jobs in this remote workspace",
            "status": "read a job's durable state", "logs": "read a bounded log chunk",
            "stop": "terminate only the recorded job's owned processes",
            "resume": "review or restart a job's checkpoint into new output",
            "review-plan": "stage selected saved-plan inputs and review actual node memory without a forecast",
            "start-plan": "start the exact reviewed staged plan with the durable job owner",
            "sync-artifacts": "retrieve one exact or latest committed raw WRF frame for a domain",
            "sync-processed-frame": "retrieve a native processed store for one committed domain and forecast time",
            "sync-processed-frame-v2": "derive compact fields for selected loop frames and retrieve their native store members",
            "sync-native-plots": "retrieve the native PNG gallery for one exact committed forecast frame",
            "artifact-index": "read a bounded page of native committed forecast times without opening WRF files",
            "sync-outputs": "retrieve this run's whole committed output set, verified file by file",
            "list-products": "print the product vocabulary the selected node's own renderer serves"}[action])
        command.add_argument("--host", required=True, help="existing SSH alias or user@host")
        command.add_argument("--python", required=True, help="absolute remote Python path with WOOF-ArWen installed")
        command.add_argument("--workspace", required=True, help="existing absolute remote workspace directory")
        command.add_argument("--port", type=int, help="SSH port (otherwise SSH configuration applies)")
        command.add_argument("--identity", help="existing local SSH identity path; contents are never copied")
        command.add_argument("--ssh-config", help="existing local OpenSSH configuration path")
        command.add_argument("--json", action="store_true", help="one versioned JSON result line; exit 0 or 2")
        if action in ("status", "logs", "stop", "resume", "sync-artifacts", "artifact-index", "sync-processed-frame", "sync-processed-frame-v2", "sync-native-plots", "sync-outputs"):
            command.add_argument("--job", required=True, help="job ID returned by start or list")
        if action in ("sync-artifacts", "artifact-index", "sync-processed-frame", "sync-processed-frame-v2", "sync-native-plots", "sync-outputs"):
            command.add_argument("--domain", type=int, default=1, help="selected committed domain, 1..999")
        if action == "sync-artifacts":
            command.add_argument("--cache-root", required=True, help="owned local cache for this job's raw frame objects")
            command.add_argument("--sequence", type=int, help="exact native output commit sequence; omit for latest")
            command.add_argument("--reader-leases", action="store_true", help="the visual reader retains shared OS leases for every frame clone")
        if action == "sync-native-plots":
            command.add_argument("--cache-root", required=True, help="local folder for selected native PNG galleries")
            command.add_argument("--sequence", type=int, required=True, help="exact native output commit sequence")
            command.add_argument("--profile", choices=("viewer-2d-v1", "full-science-v1"), default=None,
                                 help="native processing profile the gallery draws from; omit for the compact viewer profile")
            command.add_argument("--products", help="render catalog selectors separated by commas; omit for this run's own selection, empty for the node's default set")
            command.add_argument("--width", type=int, help="panel width in pixels, 256..4096; default 1200")
            command.add_argument("--height", type=int, help="panel height in pixels, 256..4096; default 900")
            command.add_argument("--theme", help="built-in theme or theme JSON path on the node; files may extend woof-light or woof-dark")
            command.add_argument("--layout", choices=("auto", "fixed"), help="auto sizes each canvas from its domain; fixed keeps the requested size, default 1200x900")
        if action == "sync-processed-frame":
            command.add_argument("--cache-root", required=True, help="owned local directory for immutable native processed stores")
            command.add_argument("--sequence", type=int, help="exact native output commit sequence; omit for latest")
        if action == "sync-processed-frame-v2":
            command.add_argument("--cache-root", required=True, help="owned bounded local cache for compact native fields")
            command.add_argument("--sequence", type=int, help="exact committed sequence; omit for latest")
            command.add_argument("--profile", choices=("viewer-2d-v1", "full-science-v1"), default="viewer-2d-v1")
            command.add_argument("--products", help="render catalog selectors separated by commas; an empty value takes the node's own default set")
            command.add_argument("--prefetch-sequences", help="up to eight committed loop sequences separated by commas")
            command.add_argument("--expected-run-id", help="require this exact native producer run identity")
            command.add_argument("--reader-leases", action="store_true", help="viewer retains shared native-store object leases")
            command.add_argument("--cache-bytes", type=int, help="local viewer cache budget in bytes; default 2 GiB")
        if action in ("artifact-index", "sync-outputs"):
            command.add_argument("--after-sequence", type=int, default=0, help="last native sequence from the previous timeline page")
        if action == "sync-outputs":
            command.add_argument("--cache-root", required=True, help="owned local directory for this run's committed output set")
        if action in LAUNCH_ACTIONS:
            command.add_argument("--request-id", help="name this launch attempt with the 32-character identity a previous reply or timeout printed, so the node answers a retry with the job that attempt already created; omit for a fresh attempt")
        if action in ("start", "resume"):
            command.add_argument("--outdir", required=True, help="absolute remote output directory; the run claims its own stamped run folder inside it, so a directory that already collects runs takes another beside them, and a run folder that already exists is refused")
            command.add_argument("--dry-run", action="store_true", help="review inputs and command; create and launch nothing")
            for name in ("config", "wps", "input"):
                command.add_argument(f"--expected-{name}-sha256", help="refuse inputs changed since the reviewed SHA-256")
            command.add_argument("--expected-prepared-sha256", help="refuse a prepared receipt changed since review")
        if action == "start":
            command.add_argument("--config", required=True, help="existing absolute remote experiment TOML")
        if action in ("start", "resume"):
            # A resume takes the same inputs a start takes. Omitting one keeps
            # the source job's own recorded value; naming one changes it here.
            command.add_argument("--geog-root", help="existing absolute remote geography directory")
            command.add_argument("--prepared-root", help="existing absolute remote prepared bundle; reuse it without fetch or preparation")
            command.add_argument("--wps-namelist", help="with --prepared-root: exact absolute remote WPS authority required by a single-domain bundle")
            command.add_argument("--products", help="render catalog selectors, all, or none")
            command.add_argument("--section", help="line for xsec: products: --section=LAT,LON,LAT,LON or a JSON file on the node; relative paths use the configuration directory (the source job's working directory on resume)")
            command.add_argument("--device", help="card index or full GPU UUID on the node; omit to take the node's own default")
        if action == "resume":
            command.add_argument("--from", dest="from_checkpoint", default="latest", help="latest valid checkpoint, or its absolute remote path")
            command.add_argument("--expected-checkpoint-sha256", help="refuse a selected checkpoint changed since review")
            command.add_argument("--expected-checkpoint-set-sha256", help="refuse any checkpoint set member changed since review")
        if action == "logs":
            command.add_argument("--cursor", type=int, default=0, help="byte cursor from the previous log result")
            command.add_argument("--limit", type=int, default=16384, help="maximum bytes requested, 1..131072; each response may return less")
        if action == "list":
            command.add_argument("--limit", type=int, default=20, help="newest 1..100 jobs")
        if action == "review-plan":
            command.add_argument("--plan", required=True, help="saved local run-plan JSON to stage")
            command.add_argument("--outdir", required=True, help="new absolute remote output directory")
            command.add_argument("--geog-root", help="existing remote geography directory")
            command.add_argument("--prepared-root", help="existing remote prepared bundle this plan's run option is relocated onto")
            command.add_argument("--wps-namelist", help="existing remote WPS authority this plan's run option is relocated onto")
            command.add_argument("--restart", help="existing remote checkpoint this plan's run option is relocated onto")
            command.add_argument("--device", help="card index or full GPU UUID on the node; omit to take the node's own default")
            command.add_argument("--expected-plan-sha256", required=True)
            command.add_argument("--expected-config-sha256", required=True)
        if action == "start-plan":
            command.add_argument("--bundle-id", required=True)
            command.add_argument("--source-inputs-file", help="completed local review whose selected raw inputs must still match")
            for name in ("bundle", "plan", "config", "input"):
                command.add_argument(f"--expected-{name}-sha256", required=True)
        command.set_defaults(func=remote_main)
