"""Local CDS credential status and explicit, private credential updates.

Status never returns a key. Updates accept secrets on stdin, never argv, and
do not contact CDS or start an acquisition.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

SCHEMA = "arwen.cds-credentials.v1"
DEFAULT_URL = "https://cds.climate.copernicus.eu/api"
_LIMIT = 16384
#: The endpoint of the CDS API that was switched off in September 2024. A
#: file written for it carries ``url: .../api/v2`` and a ``UID:KEY`` token;
#: cdsapi 0.7 still routes that shape to its retired client, so every
#: request it makes fails at the server instead of at the file.
_RETIRED_ENDPOINT_SUFFIX = "/api/v2"
_LEGACY_KEY = re.compile(r"^\d+:\S+$")
#: Token shapes worth masking even when the value itself is not known: a
#: personal access token is a UUID, a retired key is ``UID:hex-uuid``.
_TOKEN_SHAPES = re.compile(
    r"\b\d+:[0-9a-fA-F-]{8,}\b"
    r"|\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    r"|(?i:\bbearer\s+\S+|\b(?:private-token|key)\s*[:=]\s*\S+)")


def _path() -> Path:
    from woof.fetch import cds_credentials_path
    return cds_credentials_path().absolute()


def _endpoint(value) -> str:
    if not isinstance(value, str):
        raise ValueError("Enter an HTTPS CDS API endpoint.")
    value = value.strip().rstrip("/")
    try:
        parts = urlsplit(value)
        valid = (parts.scheme == "https" and parts.hostname
                 and not parts.username and not parts.password
                 and not parts.query and not parts.fragment
                 and not any(character.isspace() for character in value))
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Enter an HTTPS CDS API endpoint without a key or password in its URL.")
    if _retired(value):
        raise ValueError("The CDS API v2 endpoint was switched off in 2024, so every request to it fails. "
                         f"Use {DEFAULT_URL} with a personal access token.")
    return value


def _retired(url) -> bool:
    return isinstance(url, str) and url.strip().rstrip("/").lower().endswith(_RETIRED_ENDPOINT_SUFFIX)


def _profile(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as stream:
            content = stream.read(_LIMIT + 1)
        if len(content) > _LIMIT:
            return {}
        # Match cdsapi.read_config: it reads plain colon-delimited lines,
        # not YAML scalars, so quotes would become part of the token/URL.
        profile = {}
        for line in content.splitlines():
            key, separator, value = line.strip().partition(":")
            if separator and key in ("url", "key", "verify"):
                profile[key] = value.strip()
        return profile
    except Exception:
        # Parser exceptions can contain the credential line itself.
        return {}


def _encoding(head: bytes) -> str:
    """How cdsapi will read these bytes: it opens the file as text with no
    BOM handling, so a byte-order mark becomes part of the first key and a
    UTF-16 file is two bytes per character that never spell ``url:``."""
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-bom"
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return "undecodable"
    return "utf-8"


def inspect() -> dict:
    """Where the CDS client will look, what it will find there, and why that
    is or is not usable.  Never a key value: every field is a path, a kind,
    a count or a sentence.

    THE BREAKAGE THIS PREVENTS: the fetch used to refuse with one sentence
    whether the file was missing, sat in a different home than the one the
    fetch process resolves, was saved by Notepad as ``.cdsapirc.txt``, was
    written by a PowerShell redirect as UTF-16, carried a byte-order mark,
    named only a key, or held the retired ``UID:KEY`` format.  Nobody could
    tell those apart from the sentence, so a user who had entered a key was
    told to enter a key.
    """
    override = os.environ.get("CDSAPI_RC")
    try:
        home = str(Path.home())
    except (RuntimeError, OSError):
        home = None
    try:
        path = _path()
    except (RuntimeError, OSError):
        path = Path(override) if override is not None else Path(".cdsapirc")
    report = {"path": str(path), "selected_by": "CDSAPI_RC" if override is not None else "home",
              "home": home, "exists": False, "bytes": None, "encoding": None,
              "url_line": False, "key_line": False, "key_shape": None, "url_shape": None,
              "environment_overrides": [name for name in ("CDSAPI_KEY", "CDSAPI_URL")
                                        if os.environ.get(name) is not None],
              "sibling": None, "problem": None, "remedy": None}
    profile = {}
    try:
        if path.is_file():
            report["exists"] = True
            report["bytes"] = path.stat().st_size
            with path.open("rb") as stream:
                head = stream.read(_LIMIT + 1)
            report["encoding"] = _encoding(head[:_LIMIT])
            if report["encoding"] == "utf-8":
                profile = _profile(path)
        else:
            for name in (path.name + ".txt", path.name.lstrip(".") + ".txt", path.name.lstrip(".")):
                if name != path.name and path.with_name(name).is_file():
                    report["sibling"] = str(path.with_name(name))
                    break
    except OSError as error:
        report["problem"] = "unreadable"
        report["remedy"] = (f"The credential file {path} cannot be read ({type(error).__name__}). "
                            "Make it readable by the account that runs WOOF-ArWen, or save the key again "
                            "from the CDS key panel.")
    env_key = os.environ.get("CDSAPI_KEY")
    env_url = os.environ.get("CDSAPI_URL")
    key = env_key if env_key is not None else profile.get("key")
    url = env_url if env_url is not None else profile.get("url")
    report["url_line"] = "url" in profile
    report["key_line"] = "key" in profile
    if isinstance(key, str) and key.strip():
        report["key_shape"] = "legacy-uid-key" if _LEGACY_KEY.match(key.strip()) else "personal-access-token"
    if isinstance(url, str) and url.strip():
        try:
            _endpoint(url)
            report["url_shape"] = "current"
        except ValueError:
            report["url_shape"] = "retired-v2" if _retired(url) else "invalid"
    where = (f"{path} (chosen by CDSAPI_RC)" if override is not None
             else f"{path} (the .cdsapirc in the home folder {home} of the process that fetches)")
    panel = "Save the key from the CDS key panel in WOOF-ArWen's terminal (Settings, CDS key), which writes this file correctly"
    then_panel = "save it from the CDS key panel in WOOF-ArWen's terminal (Settings, CDS key), which writes this file correctly"
    if report["problem"] is None and not report["exists"] and (env_key is None or env_url is None):
        if env_key is not None:
            report["problem"] = "no-url"
            report["remedy"] = (f"CDSAPI_KEY is set but CDSAPI_URL is not, and no {where} exists to supply the "
                                f"endpoint. Set CDSAPI_URL={DEFAULT_URL} beside it, or clear CDSAPI_KEY and {then_panel}.")
        else:
            report["problem"] = "missing"
            sibling = (f" A file named {report['sibling']} is beside it; the client reads only the exact name "
                       f"{path.name}, so rename it." if report["sibling"] else "")
            report["remedy"] = (f"No credential file exists at {where}.{sibling} {panel}, or write the file "
                                f"there yourself as plain UTF-8 with two lines, 'url: {DEFAULT_URL}' and "
                                "'key: <your personal access token>'.")
    elif report["problem"] is None and report["exists"] and report["encoding"] != "utf-8" and (env_key is None or env_url is None):
        report["problem"] = "encoding"
        how = {"utf-16": "UTF-16, which is what a PowerShell redirect writes",
               "utf-8-bom": "UTF-8 with a byte-order mark, which hides the first line's name",
               "undecodable": "bytes that are not UTF-8 text"}[report["encoding"]]
        report["remedy"] = (f"The credential file {where} is saved as {how}; the CDS client reads it as plain "
                            f"UTF-8 and finds no 'url:' or 'key:' line. {panel}, or re-save the file as UTF-8 "
                            "without a byte-order mark.")
    elif report["problem"] is None and not (isinstance(key, str) and key.strip()):
        report["problem"] = "no-key"
        report["remedy"] = (f"The credential file {where} exists but has no 'key:' line"
                            + (" and no 'url:' line" if not report["url_line"] and env_url is None else "")
                            + f". {panel}, or add 'key: <your personal access token>' to it.")
    elif report["problem"] is None and not (isinstance(url, str) and url.strip()):
        report["problem"] = "no-url"
        report["remedy"] = (f"The credential file {where} has a key but no 'url:' line, and CDSAPI_URL is not set. "
                            f"{panel}, or add 'url: {DEFAULT_URL}' to it.")
    elif report["problem"] is None and report["key_shape"] == "legacy-uid-key":
        report["problem"] = "legacy-key"
        report["remedy"] = ("The key has the retired 'UID:KEY' shape of the CDS API that was switched off in 2024; "
                            "the client routes it to that retired service and every request fails. Create a "
                            "personal access token on your CDS profile page at https://cds.climate.copernicus.eu "
                            f"and {then_panel}.")
    elif report["problem"] is None and report["url_shape"] != "current":
        report["problem"] = "retired-url" if report["url_shape"] == "retired-v2" else "invalid-url"
        report["remedy"] = (("The endpoint names the CDS API v2 service that was switched off in 2024, so every request "
                             "to it fails." if report["url_shape"] == "retired-v2" else
                             "The endpoint is not an HTTPS URL the CDS client can use.")
                            + f" Use 'url: {DEFAULT_URL}'" + (" in CDSAPI_URL" if env_url is not None else
                                                              f" in {where}") + f", or {then_panel}.")
    return report


def evidence(report: dict | None = None) -> str:
    """One sentence a refusal can carry: which file, whether it is there,
    which source the client will use.  Never a key value."""
    report = inspect() if report is None else report
    if report["selected_by"] == "CDSAPI_RC":
        where = f"Credential file {report['path']} (chosen by CDSAPI_RC)"
    else:
        where = f"Credential file {report['path']} (the .cdsapirc of home folder {report['home']})"
    if report["problem"] == "unreadable":
        state = "cannot be read"
    elif report["exists"]:
        state = f"exists ({report['bytes']} bytes, {report['encoding']}" + (
            f", url line {'present' if report['url_line'] else 'absent'}, key line "
            f"{'present' if report['key_line'] else 'absent'}" if report["encoding"] == "utf-8" else "") + ")"
    else:
        state = "is missing" + (f"; {report['sibling']} is beside it" if report["sibling"] else "")
    overrides = report["environment_overrides"]
    if len(overrides) == 2:
        source = "the client uses CDSAPI_URL and CDSAPI_KEY from the environment, not the file"
    elif overrides:
        source = f"the client takes {overrides[0]} from the environment and the rest from the file"
    else:
        source = "the client reads the file"
    return f"{where} {state}; {source}."


def redact(text, *secrets) -> str:
    """``text`` with every known secret and every token-shaped run masked."""
    text = str(text)
    for secret in secrets:
        if isinstance(secret, str) and secret.strip():
            text = text.replace(secret.strip(), "[redacted]")
    return _TOKEN_SHAPES.sub("[redacted]", text)


def _secrets() -> tuple:
    secrets = [os.environ.get("CDSAPI_KEY")]
    try:
        secrets.append(_profile(_path()).get("key"))
    except Exception:
        pass
    return tuple(secret for secret in secrets if isinstance(secret, str))


def client_refusal(error: BaseException, *, credential_context: str | None = None,
                   secrets: tuple = ()) -> str:
    """The sentence for a ``cdsapi.Client()`` that raised: the client's own
    error class and (redacted) message, the file it looked for and whether
    that file exists, the source it would use, and the way out."""
    if credential_context is not None:
        message = redact(error, *secrets)
        return (f"Cannot initialize data-store client ({credential_context}): "
                f"{message}. No key value is logged.")
    report = inspect()
    message = redact(error, *_secrets()).strip().splitlines()
    raised = f"{type(error).__name__}: {message[0][:300]}" if message else type(error).__name__
    remedy = report["remedy"] or (
        "The file and its lines look usable, so the client itself failed to start: reinstall "
        "cdsapi>=0.7.7 and ecmwf-datastores-client in this WOOF-ArWen Python environment.")
    return (f"Cannot initialize the CDS client: cdsapi raised {raised}. {evidence(report)} "
            f"Next: {remedy} No key value is logged.")


def retrieval_refusal(error: BaseException) -> str:
    """The sentence for a CDS request that failed after the client started."""
    report = inspect()
    message = redact(error, *_secrets()).strip().splitlines()
    raised = f"{type(error).__name__}: {message[0][:300]}" if message else type(error).__name__
    status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(status, int):
        raised = f"HTTP {status}, {raised}"
    if report["problem"] in ("legacy-key", "retired-url"):
        remedy = report["remedy"]
    elif isinstance(status, int) and status in (401, 403):
        remedy = ("CDS rejected the token. Check that the key line holds your current personal access "
                  "token from https://cds.climate.copernicus.eu, and that both ERA5 dataset licences are "
                  "accepted on that site.")
    else:
        remedy = ("Check the configured token, accept both ERA5 dataset licences in the CDS website, "
                  "and check network and CDS service availability.")
    return (f"ERA5 retrieval failed at CDS: {raised}. {evidence(report)} Next: {remedy} "
            "No input file was published and no key value is logged.")


def status() -> dict:
    report = inspect()
    path = Path(report["path"])
    overrides = report["environment_overrides"]
    configured = report["problem"] is None
    url = os.environ.get("CDSAPI_URL")
    if url is None:
        try:
            url = _profile(path).get("url") if report["encoding"] == "utf-8" else None
        except Exception:
            url = None
    try:
        url = _endpoint(url)
    except ValueError:
        url = DEFAULT_URL
    return {"schema": SCHEMA, "configured": configured, "path": str(path),
            "source": "environment" if overrides else "file" if configured else "missing",
            "url": url, "editable": not overrides,
            "environment_overrides": overrides, "problem": report["problem"],
            "message": ("Configured locally; authentication is checked when downloading."
                        if configured else report["remedy"] or "No usable CDS credentials are configured.")}


def acquisition_readiness() -> dict:
    """Safe local readiness for a requested CDS acquisition, without contact.

    Call on the computer that will fetch the data. The returned contract
    contains neither credential values nor paths, endpoints, or raw errors;
    file/environment status remains available through the explicit settings
    command. Authentication and dataset access are checked only by CDS during
    a requested download. A verified cached acquisition needs no such gate.
    """
    from importlib.util import find_spec

    try:
        client_available = find_spec("cdsapi") is not None
    except Exception:
        client_available = False
    try:
        configured_status = status()
        configured = bool(configured_status["configured"])
        source = configured_status["source"]
        if source not in ("environment", "file", "missing"):
            source = "missing"
    except Exception:
        configured = False
        source = "missing"
    problems = []
    if not client_available:
        problems.append("Install cdsapi>=0.7.7 in the selected computer's WOOF-ArWen Python environment.")
    if not configured:
        problems.append("Configure CDS credentials on the selected computer in ~/.cdsapirc, "
                        "or choose its private credential file with CDSAPI_RC. "
                        "Credentials saved only on the local computer do not configure an SSH node.")
    return {"schema": "arwen.cds-acquisition-readiness.v1",
            "ready": configured and client_available, "configured": configured,
            "client_available": client_available, "authentication_checked": False,
            "credential_source": source,
            "message": " ".join(problems) if problems else
                "CDS credentials and the client are configured on this computer; "
                "authentication and dataset access are checked when downloading."}


def _hidden_process_options() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _wsl_location(path: Path):
    text = str(path).replace("/", "\\")
    if text.lower().startswith("\\\\?\\unc\\"):
        text = "\\\\" + text[8:]
    parts = text.split("\\")
    if (len(parts) >= 5 and parts[:2] == ["", ""]
            and parts[2].lower() in ("wsl.localhost", "wsl$")):
        return parts[3], "/" + "/".join(parts[4:])
    return None


_WSL_SAVE = r'''
import json, os, pathlib, sys, tempfile
request = json.load(sys.stdin)
path = pathlib.Path(request["path"]).expanduser().resolve()
path.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary = tempfile.mkstemp(prefix=".cdsapirc-", dir=path.parent)
try:
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(request["content"])
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
'''


def _private_windows_file(path: Path) -> None:
    result = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"],
        capture_output=True, text=True, check=False, **_hidden_process_options())
    try:
        row = next(csv.reader(io.StringIO(result.stdout.strip())))
        sid = row[1]
        if result.returncode or not sid.startswith("S-1-"):
            raise ValueError
    except (ValueError, IndexError, StopIteration):
        raise ValueError("Cannot determine the Windows account for private credential storage.") from None
    result = subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"*{sid}:(F)"],
        capture_output=True, text=True, check=False, **_hidden_process_options())
    if result.returncode:
        raise ValueError("Cannot restrict the credential file to the current Windows account.")


def save(request: dict) -> dict:
    if not isinstance(request, dict):
        raise ValueError("Credential input must be an object.")
    current = status()
    if not current["editable"]:
        raise ValueError("CDS credentials are overridden by environment variables. Change those variables and reopen WOOF-ArWen.")
    path = _path()
    prior = _profile(path)
    key = request.get("key", "")
    if not isinstance(key, str):
        raise ValueError("Enter a CDS personal access token.")
    key = key.strip() or prior.get("key", "")
    if (not isinstance(key, str) or not key or len(key) > 8192
            or any(character in key for character in ("\r", "\n", "\0"))):
        raise ValueError("Enter a CDS personal access token on one line.")
    if _LEGACY_KEY.match(key):
        raise ValueError("That is a 'UID:KEY' pair for the CDS API switched off in 2024; the client would send "
                         "it to that retired service and every request would fail. Enter the personal access "
                         "token from your profile page at https://cds.climate.copernicus.eu instead.")
    # A file written for the retired endpoint keeps its dead URL through a
    # key change unless the caller names one; a new token is for the
    # current service, so the current endpoint replaces the retired one.
    requested_url = request.get("url")
    if not requested_url and _retired(prior.get("url")):
        requested_url = DEFAULT_URL
    url = _endpoint(requested_url or prior.get("url") or DEFAULT_URL)
    body = f"url: {url}\nkey: {key}\n"
    if "verify" in prior:
        body += f"verify: {prior['verify']}\n"
    location = _wsl_location(path) if os.name == "nt" else None
    if location is not None:
        distribution, linux_path = location
        result = subprocess.run(
            ["wsl.exe", "--distribution", distribution, "--exec", "python3", "-c", _WSL_SAVE],
            input=json.dumps({"path": linux_path, "content": body}),
            capture_output=True, text=True, check=False, **_hidden_process_options())
        if result.returncode:
            raise ValueError("Cannot save the CDS credential file in WSL. Check that the displayed distribution and home folder are available.")
    else:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".cdsapirc-", dir=path.parent)
        try:
            if os.name == "nt":
                _private_windows_file(Path(temporary))
            else:
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                descriptor = None
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if os.path.exists(temporary):
                os.unlink(temporary)
    result = status()
    result["message"] = "CDS credentials saved. Authentication will be checked when downloading."
    return result


def credentials_main(args) -> int:
    try:
        if args.save:
            raw = sys.stdin.read(_LIMIT + 1)
            if len(raw) > _LIMIT:
                raise ValueError("Credential input is too long.")
            try:
                request = json.loads(raw)
            except (ValueError, TypeError):
                raise ValueError("Credential input is not valid JSON.") from None
            result = save(request)
        else:
            result = status()
    except ValueError:
        raise
    except Exception:
        raise ValueError("Could not access the displayed CDS credential file. No credential values were logged.") from None
    if args.json:
        print(json.dumps(result))
    else:
        print(result["message"])
        print(f"Source: {result['source']}\nFile: {result['path']}\nEndpoint: {result['url']}")
    return 0


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser("cds-credentials", help="show or update local CDS credentials without revealing the key")
    parser.add_argument("--json", action="store_true", help="return safe credential status as JSON")
    parser.add_argument("--save", action="store_true", help="read endpoint and key as JSON from stdin and save privately")
    parser.set_defaults(func=credentials_main)
