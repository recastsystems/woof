#!/usr/bin/env python3
"""Recapture an advection oracle GPU receipt on this card, compare it word
for word with the committed one, and move it only when production reproduces.

    python tools/advect_wrf471_oracle/recapture_receipt.py \\
        --fixture tests/data/wrf471_advect --scratch DIR [--install] [--as-card]
    python tools/advect_wrf471_oracle/recapture_receipt.py \\
        --fixture tests/data/wrf_legacy_advect --scratch DIR [--install] [--as-card]

THE BREAKAGE THIS PREVENTS
--------------------------
Every receipt pins the assembled kernel source it was measured at
(``kernels.advection``, ``kernels.pd_advection``, ``kernels.openbc``), so a
kernel edit leaves each receipt red until a capture on its card reproduces
its words.  The vertical-order lane (286-vadv5) left the H100 receipt of the
WRF 4.7.1 fixture behind that way (b70a94a48): no lane can reach an sm_90
card.  Byte identity is certified on Blackwell and newer only (ruling
2026-10-04), so an older card's receipt that trails the tree is reported,
not failed, and the index's ``uncertified_stale`` record names the commit
that staled it; installing that receipt again drops the record.  The
HRRR-fork fixture has no sm_100 receipt at all, so its device tests assert a
coverage gap there.  Moving a receipt by hand on a rented box is
exactly where a receipt gets words that were never measured, or a
production change on that architecture gets recorded as the new truth.
This tool is the one procedure for that move:

1. it copies the fixture's inputs and references into ``--scratch`` (the
   validator writes word archives inside the directory it measures, and the
   checkout must not change before the comparison);
2. it runs ``validate_advect_oracle.py --gpu --controls --mutation`` there,
   with the WRF-exact switches removed so the default compile is measured;
3. it compares the capture with the receipt this card selects through the
   fixture's ``gpu-receipts.json``: every case's measurement rows and level
   rows (each carries the SHA-256 of the words it measured) and, where the
   committed folder stores the archive, every float32 word;
4. it writes a record of all of that, and with ``--install`` moves the
   receipt, the stored archives, the fixture's checksum lines and the index
   ONLY if every production word reproduced (or the card has no receipt).

A production word that moved is a finding, not a recapture: the tool exits
1, installs nothing, and the record names every differing case and count.

Modes, by what ``gpu-receipts.json`` says about this card:

``card``          the card has its own receipt: production must reproduce;
                  controls may move (they are text-patched compiles whose
                  multiply-add fusion follows unrelated source edits) and
                  are re-recorded from the capture.
``architecture``  another card's receipt of the same compute capability:
                  installing moves that receipt's pins only if production
                  AND every control reproduced (the RTX 5070 Ti precedent in
                  README.md); otherwise ``--as-card`` records this card's own.
``first``         nothing recorded for the card or its architecture: the
                  capture becomes ``gpu-receipt-sm<MM>.json`` with every
                  variant's archives in ``gpu-words-sm<MM>``, and the index
                  maps the card and, if unmapped, its architecture.

Exit status: 0 reproduced (or a first capture), 1 production differs,
2 install refused, 3 capture failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
VALIDATOR = REPO / "tools" / "advect_wrf471_oracle" / "validate_advect_oracle.py"
INDEX_NAME = "gpu-receipts.json"
#: Where each fixture's capture records live, beside its oracle's build.
FIXTURE_ORACLES = {
    "wrf471_advect": "advect_wrf471_oracle",
    "wrf_legacy_advect": "advect_wrf_legacy_oracle",
}
#: The checksum file that pins each fixture's receipts, when it has one.
FIXTURE_SUMS = {
    "wrf471_advect": "oracle-sha256sums.txt",
}
CAPTURED_WORDS = "captured-words"
#: woof.verify.advect_oracle.UNCERTIFIED_STALE_KEY: the index's record of
#: which commit staled an uncertified receipt.  A record that outlived the
#: capture that brought its receipt current would fail the parity tests.
STALE_RECORD_KEY = "uncertified_stale"
CAPTURED_RECEIPT = "captured-receipt.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_receipt(path: Path, value) -> None:
    # The validator's own spelling, so a reproduced receipt is byte-identical.
    # Bytes, so a Windows checkout writes the same LF file a node does.
    Path(path).write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii"))


def load_index(fixture: Path) -> dict:
    """The fixture's receipt index: card name -> receipt, "M.m" -> receipt."""

    index = _read_json(fixture / INDEX_NAME)
    assert index.get("schema_version") == 1, index.get("schema_version")
    return index


def write_index(fixture: Path, index: dict) -> None:
    (fixture / INDEX_NAME).write_bytes(
        (json.dumps(index, indent=1, sort_keys=True) + "\n").encode("ascii"))


def select_receipt(index: dict, fixture: Path, gpu: str,
                   capability, nvrtc: str | None = None) -> tuple[str, str | None]:
    """``(mode, receipt name)`` for a card, as the device tests select it.

    A receipt recorded for this NVRTC build and architecture (the index's
    ``compilers``) comes first: the production compile allows multiply-add
    contraction, so its words follow the compiler as well as the card."""

    if nvrtc is not None:
        name = index.get("compilers", {}).get(nvrtc, {}).get(
            f"{int(capability[0])}.{int(capability[1])}")
        if name is not None:
            return "compiler", name
    name = index["cards"].get(gpu)
    if name is not None:
        return "card", name
    key = f"{int(capability[0])}.{int(capability[1])}"
    name = index["architectures"].get(key)
    if name is not None:
        return "architecture", name
    for name in sorted(set(index["cards"].values())):
        path = fixture / name
        if path.is_file() and list(_read_json(path)["compute_capability"]) == [
                int(capability[0]), int(capability[1])]:
            return "architecture", name
    return "first", None


def _words(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: np.ascontiguousarray(data[key]) for key in data.files}


def _compare_archives(a: dict, b: dict) -> tuple[int, int] | None:
    """``(words compared, words different)`` by raw bits, or None on a layout change."""

    if set(a) != set(b):
        return None
    total = different = 0
    for key in sorted(a):
        x, y = a[key], b[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            return None
        if x.dtype.itemsize == 4:
            xv, yv = x.view(np.uint32), y.view(np.uint32)
        else:
            xv, yv = x.view(np.uint8), y.view(np.uint8)
        total += int(xv.size)
        different += int(np.count_nonzero(xv != yv))
    return total, different


def _variants(receipt: dict) -> dict:
    out = {"production": receipt.get("cases", {})}
    out.update(receipt.get("controls", {}))
    return out


def compare(committed: dict, committed_words: Path, captured: dict,
            captured_words: Path) -> dict:
    """Every variant, case by case: rows, level rows and stored words."""

    old, new = _variants(committed), _variants(captured)
    report = {}
    for variant in sorted(set(old) | set(new), key=lambda v: (v != "production", v)):
        rows_old, rows_new = old.get(variant), new.get(variant)
        entry = {"identical": True, "cases": {}, "words_compared": 0,
                 "words_different": 0}
        if rows_old is None or rows_new is None:
            entry.update(identical=False, missing="committed" if rows_old is None else "captured")
            report[variant] = entry
            continue
        for case in sorted(set(rows_old) | set(rows_new)):
            a, b = rows_old.get(case), rows_new.get(case)
            if a is None or b is None:
                entry["cases"][case] = {"missing": "committed" if a is None else "captured"}
                entry["identical"] = False
                continue
            row = {"measurements_identical": a["measurements"] == b["measurements"],
                   "levels_identical": a.get("levels") == b.get("levels"),
                   "words_stored": False}
            stored = _words(committed_words / a["words_file"])
            if stored is not None:
                result = _compare_archives(stored, _words(captured_words / b["words_file"]) or {})
                row["words_stored"] = True
                if result is None:
                    row["words_layout_changed"] = True
                    row["identical"] = False
                else:
                    row["words_compared"], row["words_different"] = result
                    entry["words_compared"] += result[0]
                    entry["words_different"] += result[1]
            row["identical"] = (row["measurements_identical"] and row["levels_identical"]
                                and not row.get("words_layout_changed")
                                and not row.get("words_different"))
            if not row["identical"]:
                entry["identical"] = False
                row["differing_measurements"] = sorted(
                    key for key in set(a["measurements"]) | set(b["measurements"])
                    if a["measurements"].get(key) != b["measurements"].get(key))
            entry["cases"][case] = row
        report[variant] = entry
    return report


def _sums_update(sums: Path, written: list[Path], add_missing: bool) -> list[str]:
    """Rewrite each written file's line; add lines only for a first capture."""

    if not sums.is_file():
        return []
    lines = sums.read_text(encoding="ascii").splitlines()
    by_path = {}
    for i, line in enumerate(lines):
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            by_path[parts[1]] = i
    changed = []
    for path in written:
        rel = path.resolve().relative_to(REPO).as_posix()
        line = f"{_sha256(path)}  {rel}"
        if rel in by_path:
            if lines[by_path[rel]] != line:
                lines[by_path[rel]] = line
                changed.append(rel)
        elif add_missing:
            lines.append(line)
            by_path[rel] = len(lines) - 1
            changed.append(rel)
    if changed:
        sums.write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    return changed


def settle(fixture: Path, captured_receipt: Path, captured_words: Path, *,
           install: bool, as_card: bool = False,
           native_fix_verified: bool = False,
           nvrtc: str | None = None, as_compiler: bool = False) -> tuple[int, dict]:
    """Compare a finished capture with the fixture and install it if allowed.

    Separated from the capture so the comparison and the install rules are
    tested on the CPU (tests/test_advect_recapture_receipt.py).
    """

    fixture = Path(fixture).resolve()
    captured = _read_json(captured_receipt)
    gpu, capability = captured["gpu"], tuple(captured["compute_capability"])
    index = load_index(fixture)
    mode, name = select_receipt(index, fixture, gpu, capability, nvrtc)
    if as_card and mode == "architecture":
        mode, name = "first", None
    if as_compiler:
        if nvrtc is None:
            return 2, {"refused": "--as-compiler needs the capture's NVRTC version",
                       "installed": False, "mode": mode, "receipt": name, "gpu": gpu}
        if mode != "compiler":
            # This compiler's words are recorded beside the card receipt the
            # other compilers keep; the comparison still reads that receipt.
            mode = "compiler-first"
    record = {"tool": "tools/advect_wrf471_oracle/recapture_receipt.py",
              "fixture": fixture.relative_to(REPO).as_posix() if fixture.is_relative_to(REPO) else str(fixture),
              "gpu": gpu, "compute_capability": list(capability),
              "cupy": captured.get("cupy"), "cuda_runtime": captured.get("cuda_runtime"),
              "kernels_captured": captured["kernels"], "mode": mode,
              "receipt": name, "mutation_rejected": captured.get("mutation_rejected"),
              "installed": False, "written": []}
    record["native_fix_verified"] = native_fix_verified
    if captured.get("mutation_rejected") is not True:
        record["refused"] = "the known flux5 mutation was not rejected by the capture"
        return 2, record
    committed = None
    if name is not None:
        committed = _read_json(fixture / name)
        record["kernels_committed"] = committed["kernels"]
        record["committed_gpu"] = committed["gpu"]
        report = compare(committed, fixture / committed["words_directory"],
                         captured, Path(captured_words))
        record["variants"] = report
        production = report.get("production", {})
        record["production_reproduced"] = bool(production.get("identical"))
        record["every_variant_reproduced"] = all(v.get("identical") for v in report.values())
        record["production_words_compared"] = production.get("words_compared", 0)
        record["production_words_different"] = production.get("words_different", 0)
        if not record["production_reproduced"]:
            record["finding"] = (
                "production output on this card differs from the committed receipt: "
                "the kernels' default compile changed words here; nothing was installed")
            if mode == "compiler-first":
                record["finding"] = (
                    f"production words under NVRTC {nvrtc} differ from {name}; "
                    "recorded as this compiler's own receipt")
            elif not native_fix_verified:
                return 1, record
            elif fixture.name != "wrf_legacy_advect" or mode != "card":
                record["refused"] = "a native fix can requalify only this card's fork receipt"
                return 2, record
            else:
                record["finding"] = "production words changed with a bitwise native fork fix; requalified on this card"
    else:
        record["production_reproduced"] = None
    if mode == "compiler-first" and record.get("production_reproduced"):
        record["refused"] = (f"production reproduces {name}; NVRTC {nvrtc} needs no "
                             "receipt of its own")
        return 2, record
    if not install:
        return 0, record
    if mode == "architecture" and not record["every_variant_reproduced"]:
        record["refused"] = (
            f"{name} names {committed['gpu']}; production reproduced but a control "
            "moved, and a control row measured here would be filed under that card. "
            "Rerun with --as-card to record this card's own receipt.")
        return 2, record
    written: list[Path] = []
    captured_words = Path(captured_words)
    if mode == "compiler-first":
        arch = f"{capability[0]}.{capability[1]}"
        tag = f"sm{capability[0]}{capability[1]}-nvrtc{nvrtc}"
        name, folder = f"gpu-receipt-{tag}.json", f"gpu-words-{tag}"
        if (fixture / name).exists() or (fixture / folder).exists():
            record["refused"] = f"{name} or {folder} already exists; the index does not map this compiler to it"
            return 2, record
        (fixture / folder).mkdir()
        receipt = dict(captured, words_directory=folder,
                       nvrtc=[int(part) for part in nvrtc.split(".")])
        # Store what the receipt it sits beside stores (production only for
        # the HRRR-fork sm_120 folder), so a compiler adds no control words.
        stored = (None if committed is None
                  else fixture / committed["words_directory"])
        for rows in _variants(receipt).values():
            for row in rows.values():
                if stored is not None and not (stored / row["words_file"]).is_file():
                    continue
                target = fixture / folder / row["words_file"]
                shutil.copyfile(captured_words / row["words_file"], target)
                written.append(target)
        _write_receipt(fixture / name, receipt)
        written.append(fixture / name)
        index.setdefault("compilers", {}).setdefault(nvrtc, {})[arch] = name
        write_index(fixture, index)
        written.append(fixture / INDEX_NAME)
        record["receipt"] = name
    elif mode == "first":
        tag = f"sm{capability[0]}{capability[1]}"
        if (fixture / f"gpu-receipt-{tag}.json").exists() or (fixture / f"gpu-words-{tag}").exists():
            # The architecture's name is taken by another card's receipt
            # (--as-card on an RTX 5070 Ti beside the RTX 5090's sm120
            # receipt): this card's own files carry its name.
            tag = re.sub(r"[^a-z0-9]+", "-", gpu.lower()).strip("-")
        name = f"gpu-receipt-{tag}.json"
        folder = f"gpu-words-{tag}"
        if (fixture / name).exists() or (fixture / folder).exists():
            record["refused"] = f"{name} or {folder} already exists; the index does not map this card to it"
            return 2, record
        (fixture / folder).mkdir()
        receipt = dict(captured, words_directory=folder)
        for rows in _variants(receipt).values():
            for row in rows.values():
                target = fixture / folder / row["words_file"]
                shutil.copyfile(captured_words / row["words_file"], target)
                written.append(target)
        _write_receipt(fixture / name, receipt)
        written.append(fixture / name)
        index["cards"][gpu] = name
        index["architectures"].setdefault(f"{capability[0]}.{capability[1]}", name)
        write_index(fixture, index)
        written.append(fixture / INDEX_NAME)
        record["receipt"] = name
    elif mode in ("card", "compiler"):
        folder = committed["words_directory"]
        receipt = dict(captured, words_directory=folder)
        for rows in _variants(receipt).values():
            for row in rows.values():
                target = fixture / folder / row["words_file"]
                if target.is_file():
                    # Keep the folder's own schema: it stores what it stored.
                    if _sha256(target) != _sha256(captured_words / row["words_file"]):
                        shutil.copyfile(captured_words / row["words_file"], target)
                    written.append(target)
        _write_receipt(fixture / name, receipt)
        written.append(fixture / name)
    else:  # architecture, every variant reproduced: the pins move, the card stays
        receipt = dict(committed, kernels=captured["kernels"])
        _write_receipt(fixture / name, receipt)
        written.append(fixture / name)
        # This card now replays that receipt through its architecture (the
        # RTX 5070 Ti precedent); without the mapping its device tests would
        # still report a coverage gap.
        key = f"{capability[0]}.{capability[1]}"
        if index["architectures"].get(key) != name:
            index["architectures"][key] = name
            write_index(fixture, index)
            written.append(fixture / INDEX_NAME)
    stale = index.get(STALE_RECORD_KEY, {})
    if name in stale:
        # This capture is that receipt's new truth; the staling it recorded is over.
        del stale[name]
        if not stale:
            del index[STALE_RECORD_KEY]
        write_index(fixture, index)
        written.append(fixture / INDEX_NAME)
        record["stale_record_dropped"] = True
    sums_name = FIXTURE_SUMS.get(fixture.name)
    if sums_name:
        record["sums_changed"] = _sums_update(fixture / sums_name, written,
                                              add_missing=(mode == "first"))
    record["installed"] = True
    record["written"] = sorted({p.relative_to(fixture).as_posix() for p in written})
    return 0, record


def _stage_inputs(fixture: Path, scratch: Path) -> Path:
    """Copy the fixture's top-level files and the inputs its cases name."""

    staged = scratch / "data" / fixture.name
    staged.mkdir(parents=True, exist_ok=True)
    for path in sorted(fixture.iterdir()):
        if path.is_file():
            shutil.copyfile(path, staged / path.name)
    manifest = _read_json(fixture / "cases.json")
    for case in manifest["cases"]:
        source = (fixture / case["file"]).resolve()
        target = (staged / case["file"]).resolve()
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return staged


def _environment() -> dict:
    out = {}
    try:
        import cupy as cp
        out["cupy"] = cp.__version__
        out["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
        out["cuda_driver"] = cp.cuda.runtime.driverGetVersion()
        out["nvrtc"] = list(cp.cuda.nvrtc.getVersion())
    except Exception as error:  # noqa: BLE001 - recorded, not fatal
        out["error"] = repr(error)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--as-card", action="store_true")
    parser.add_argument("--as-compiler", action="store_true",
                        help="record this NVRTC build's own receipt for the card's architecture "
                             "(the production compile's words follow the compiler)")
    parser.add_argument("--native-fix", action="store_true",
                        help="fork only: require the bitwise native total-tendency gate before recording an intentional fix")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args(argv)
    fixture = args.fixture.resolve()
    if fixture.name not in FIXTURE_ORACLES:
        parser.error(f"--fixture must be one of {sorted(FIXTURE_ORACLES)} under tests/data")
    scratch = args.scratch.resolve()
    if scratch.exists() and any(scratch.iterdir()):
        parser.error(f"--scratch {scratch} is not empty")
    scratch.mkdir(parents=True, exist_ok=True)
    if args.native_fix:
        if fixture.name != "wrf_legacy_advect":
            parser.error("--native-fix applies only to the fork order-5 fixture")
        gate_env = dict(os.environ, GPUWM_WRF_EXACT="1", WOOF_WRF_EXACT_ADVECTION="1")
        gate = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                "--basetemp=" + str(scratch / "native-gate-tmp"),
                "tests/test_advect_wrf_legacy_parity.py::test_exact_build_holds_the_fork_reference_bitwise_on_the_explicit_routines"]
        if subprocess.run(gate, env=gate_env, cwd=REPO).returncode:
            print("native fix was not bitwise against the fork", file=sys.stderr)
            return 4
    if scratch.exists() and any(p.name != "native-gate-tmp" for p in scratch.iterdir()):
        parser.error(f"--scratch {scratch} is not empty")
    staged = _stage_inputs(fixture, scratch)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GPUWM_WRF_EXACT")}
    command = [sys.executable, str(VALIDATOR), "--directory", str(staged), "--gpu",
               "--controls", "--mutation", "--receipt", str(staged / CAPTURED_RECEIPT),
               "--words-directory", str(staged / CAPTURED_WORDS)]
    print("capture:", " ".join(command), flush=True)
    if subprocess.run(command, env=env, cwd=REPO).returncode != 0:
        print("capture failed", file=sys.stderr)
        return 3
    environment = _environment()
    nvrtc = (f"{environment['nvrtc'][0]}.{environment['nvrtc'][1]}"
             if "nvrtc" in environment else None)
    status, record = settle(fixture, staged / CAPTURED_RECEIPT, staged / CAPTURED_WORDS,
                            install=args.install, as_card=args.as_card,
                            native_fix_verified=args.native_fix,
                            nvrtc=nvrtc, as_compiler=args.as_compiler)
    record["environment"] = environment
    text = json.dumps(record, indent=1, sort_keys=True) + "\n"
    (scratch / "recapture.json").write_text(text, encoding="utf-8")
    if args.record:
        args.record.write_text(text, encoding="utf-8")
    if record["installed"]:
        slug = re.sub(r"[^a-z0-9]+", "-", record["gpu"].lower()).strip("-")
        pin = record["kernels_captured"]["advection"][:8]
        if record["mode"] == "compiler-first":
            slug += f"-nvrtc{nvrtc}"
        out = REPO / "tools" / FIXTURE_ORACLES[fixture.name] / "receipts" / f"{slug}-{fixture.name}-{pin}-recapture.json"
        out.write_text(text, encoding="utf-8")
        print("record:", out.relative_to(REPO).as_posix())
    for variant, entry in record.get("variants", {}).items():
        print(f"{variant:16s} identical={entry['identical']} words compared="
              f"{entry.get('words_compared')} different={entry.get('words_different')}")
    print(f"mode={record['mode']} receipt={record['receipt']} "
          f"production_reproduced={record['production_reproduced']} installed={record['installed']}"
          + (f" REFUSED: {record['refused']}" if record.get("refused") else ""))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
