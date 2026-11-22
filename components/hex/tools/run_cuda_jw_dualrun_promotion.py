#!/usr/bin/env python3
"""Literal trust anchor for the v8.2.3 CUDA JW dual-run promotion.

Run this file with ``python -I -S -B``.  It validates literal SHA-256 pins for
the payload, measured tool, isolated child bootstrap, and complete authority
document before compiling the already-validated payload bytes in memory.  No
output, cache, capsule, or completion target is inspected or created before
all four subordinate anchors have passed.

The outer launcher itself is the small, reviewable trust root; it deliberately
does not contain a self-referential hash.  Its bytes are recorded after launch
and revalidated by the admitted payload for the completion receipt.
"""

from __future__ import annotations
import os as _hex_os
_hex_os.environ["WOOF_HEX_NUMERICS"] = "strict"  # a bitwise proof against native MPAS runs strict numerics (fast is the default, strict the opt-in)

import hashlib
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "tools" / "run_cuda_jw_dualrun_trust.py"
FROZEN_TOOL = ROOT / "tools" / "run_cuda_jw_dualrun.py"
ISOLATED_BOOTSTRAP = ROOT / "tools" / "cuda_jw_dualrun_isolated_bootstrap.py"
AUTHORITY_PINS = ROOT / "tools" / "cuda_jw_dualrun_authority_pins.json"

UNFROZEN_SENTINEL = "SRC_FREEZE_REQUIRED"
PAYLOAD_SHA256 = UNFROZEN_SENTINEL
FROZEN_TOOL_SHA256 = UNFROZEN_SENTINEL
ISOLATED_BOOTSTRAP_SHA256 = UNFROZEN_SENTINEL
AUTHORITY_PINS_SHA256 = UNFROZEN_SENTINEL

OUTER_AUTHORITY_SCHEMA = "mpas-port.cuda-jw-dualrun-outer-authority/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_MODULE_NAME = "_mpas_cuda_jw_dualrun_trust_payload"


class PromotionRefusal(RuntimeError):
    """The literal outer trust boundary was not satisfied."""


def _read_stable_file(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.expanduser()
    if raw.is_symlink():
        raise PromotionRefusal(f"trust-anchor path must not be a symlink: {raw}")
    selected = raw.resolve()
    if not selected.is_file():
        raise PromotionRefusal(f"trust-anchor path is not a regular file: {selected}")
    before = selected.stat(follow_symlinks=False)
    payload = selected.read_bytes()
    after = selected.stat(follow_symlinks=False)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != after.st_size:
        raise PromotionRefusal(f"trust-anchor bytes changed while reading: {selected}")
    return {
        "path": str(selected),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }, payload


def _admit_literal_pin(
    *, label: str, path: Path, expected_sha256: str
) -> tuple[dict[str, Any], bytes]:
    if expected_sha256 == UNFROZEN_SENTINEL:
        raise PromotionRefusal(
            f"SRC FREEZE has not installed the outer {label} SHA-256"
        )
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise PromotionRefusal(f"outer {label} SHA-256 literal is invalid")
    record, payload = _read_stable_file(path)
    if record["sha256"] != expected_sha256:
        raise PromotionRefusal(f"outer {label} bytes differ from the literal SHA-256")
    return record, payload


def _assert_isolated_startup() -> None:
    required = {
        "isolated": 1,
        "ignore_environment": 1,
        "no_site": 1,
        "no_user_site": 1,
        "dont_write_bytecode": 1,
        "safe_path": True,
    }
    mismatches = {
        name: (getattr(sys.flags, name, None), expected)
        for name, expected in required.items()
        if getattr(sys.flags, name, None) != expected
    }
    if mismatches:
        raise PromotionRefusal(
            f"outer launcher requires python -I -S -B: {mismatches}"
        )
    forbidden = sorted(
        name
        for name in sys.modules
        if name in {"site", "sitecustomize", "usercustomize"}
        or name == "hexcore"
        or name.startswith("hexcore.")
        or name == "woof"
        or name.startswith("woof.")
        or name == "cupy"
        or name.startswith("cupy.")
        or name == "numpy"
        or name.startswith("numpy.")
        or name == "netCDF4"
        or name.startswith("netCDF4.")
    )
    if forbidden:
        raise PromotionRefusal(
            f"target/runtime modules loaded before literal admission: {forbidden}"
        )


def _execute_admitted_payload(
    payload_bytes: bytes,
    *,
    authority: dict[str, Any],
    argv: Sequence[str] | None,
) -> int:
    if _PAYLOAD_MODULE_NAME in sys.modules:
        raise PromotionRefusal("trust payload module name was already occupied")
    module = ModuleType(_PAYLOAD_MODULE_NAME)
    module.__file__ = str(PAYLOAD.resolve())
    module.__package__ = ""
    module.__loader__ = None
    sys.modules[_PAYLOAD_MODULE_NAME] = module
    try:
        code = compile(payload_bytes, str(PAYLOAD.resolve()), "exec", dont_inherit=True)
        exec(code, module.__dict__)
        module.OUTER_LAUNCHER_AUTHORITY = authority
        result = module.main(None if argv is None else list(argv))
    except BaseException:
        sys.modules.pop(_PAYLOAD_MODULE_NAME, None)
        raise
    if type(result) is not int or result != 0:
        raise PromotionRefusal(f"admitted trust payload returned {result!r}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    _assert_isolated_startup()
    payload_record, payload_bytes = _admit_literal_pin(
        label="payload", path=PAYLOAD, expected_sha256=PAYLOAD_SHA256
    )
    tool_record, _ = _admit_literal_pin(
        label="frozen tool", path=FROZEN_TOOL, expected_sha256=FROZEN_TOOL_SHA256
    )
    bootstrap_record, _ = _admit_literal_pin(
        label="isolated bootstrap",
        path=ISOLATED_BOOTSTRAP,
        expected_sha256=ISOLATED_BOOTSTRAP_SHA256,
    )
    pins_record, _ = _admit_literal_pin(
        label="authority pins",
        path=AUTHORITY_PINS,
        expected_sha256=AUTHORITY_PINS_SHA256,
    )
    outer_record, _ = _read_stable_file(Path(__file__).resolve())
    authority = {
        "schema": OUTER_AUTHORITY_SCHEMA,
        "outer_launcher": outer_record,
        "payload": payload_record,
        "frozen_tool": tool_record,
        "isolated_bootstrap": bootstrap_record,
        "authority_pins": pins_record,
    }
    return _execute_admitted_payload(
        payload_bytes,
        authority=authority,
        argv=sys.argv[1:] if argv is None else argv,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PromotionRefusal as error:
        raise SystemExit(f"promotion refusal: {error}") from error
