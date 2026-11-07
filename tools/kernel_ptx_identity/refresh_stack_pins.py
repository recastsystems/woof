"""Regenerate the stack's source pins from the live loader, on a sprint box.

Records every changed pin and actual NVRTC PTX hashes for requested modules.
Numerical qualification is supplied separately by the column gates. A source
pin records bytes, not forecast skill or equality to the previous arithmetic.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
KDIR = ROOT / "woof/core/kernels"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def assignment(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        return getattr(node.targets[0], "id", None), node.value
    if isinstance(node, ast.AnnAssign):
        return getattr(node.target, "id", None), node.value
    return None, None


def patch_file(path, dictionaries, constants, readings, write):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []

    def replace(node, value, label):
        # Some pins refer to the frozen table's own constant. Preserve that
        # single authority instead of replacing the reference with a copy.
        if isinstance(node, (ast.Attribute, ast.Subscript, ast.Name)):
            return
        old = ast.literal_eval(node)
        if old == value:
            return
        start = offsets[node.lineno - 1] + node.col_offset
        stop = offsets[node.end_lineno - 1] + node.end_col_offset
        edits.append((start, stop, repr(value)))
        readings.append({"file": str(path.relative_to(ROOT)), "line": node.lineno,
                         "pin": label, "before": old, "after": value})

    for node in ast.walk(ast.parse(text)):
        name, value = assignment(node)
        if name in dictionaries and isinstance(value, ast.Dict):
            for key, old in zip(value.keys, value.values):
                label = ast.literal_eval(key)
                replacement = dictionaries[name](label)
                if replacement is None:
                    # Retire pins for headers no active loader composes.
                    start = offsets[key.lineno - 1]
                    stop = offsets[old.end_lineno]
                    edits.append((start, stop, ""))
                    readings.append({"file": str(path.relative_to(ROOT)),
                                     "line": key.lineno, "pin": label,
                                     "after": "unused header removed"})
                elif isinstance(old, ast.Tuple):
                    for index, new_value in enumerate(replacement):
                        replace(old.elts[index], new_value, f"{name}.{label}[{index}]")
                else:
                    replace(old, replacement, f"{name}.{label}")
        elif name in constants:
            replace(value, constants[name], name)
    if write:
        for start, stop, replacement in sorted(edits, reverse=True):
            text = text[:start] + replacement + text[stop:]
        path.write_text(text, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--compile", action="append", default=[])
    args = parser.parse_args()
    from woof.core import kernels, terrain_drag
    from tools.mp8_freeze_receipt import kernel_source_path
    from tools.kernel_ptx_identity.compare import compile_ptx, ptx_functions

    sources = {}
    def source(name):
        if name not in sources:
            sources[name] = kernels.module_source(name)
        return sources[name]

    def physical_module_digest(name):
        if name == "rrtmg_lw_chain_coalesced":
            # The production loader generates this retired duplicate from
            # rrtmg_lw_chain.cu; use the same authority as the per-module gate.
            from woof.core.rrtmg_lw import _LW_COALESCED_HEADER, _lw_coalesced_source
            chain = (KDIR / "rrtmg_lw_chain.cu").read_text(encoding="utf-8")
            return digest((_LW_COALESCED_HEADER + _lw_coalesced_source(chain)).encode())
        return digest((KDIR / f"{name}.cu").read_bytes())

    readings = []
    pending = []
    def record(path, dictionaries, constants):
        pending.append((path, dictionaries, constants))
        patch_file(path, dictionaries, constants, readings, False)

    record(ROOT / "tests/test_mp8_frozen.py", {
        "FROZEN_MODULE_DIGESTS": lambda name: (
            digest(kernel_source_path(name).read_bytes()), digest(source(name).encode()))}, {
        "THOMPSON_CU_SHA256": digest((KDIR / "thompson.cu").read_bytes()),
        "THOMPSON_COMPILED_SOURCE_SHA256": digest(source("thompson").encode()),
        "THOMPSON_COMPILED_SOURCE_LEN": len(source("thompson")),
        "COMMON_CUH_SHA256": digest((KDIR / "common.cuh").read_bytes()),
        "PREAMBLE_SHA256": digest(kernels._preamble().encode()),
        "PREAMBLE_LEN": len(kernels._preamble()),
        "THOMPSON_PY_SHA256": digest((ROOT / "woof/core/thompson.py").read_bytes()),
    })
    from woof.core.ruc_tier import RUC_FUSED_SOURCES
    live_headers = {"common.cuh", *RUC_FUSED_SOURCES,
                    *(header for headers in kernels.EXTRA_HEADERS.values()
                      for header in headers)}
    record(ROOT / "tests/test_kernel_source_freeze_per_module.py", {
        "BASELINE_PINNED": physical_module_digest,
        "PINNED_HEADERS": lambda name: (digest((KDIR / name).read_bytes())
                                        if name in live_headers else None),
    }, {})
    record(ROOT / "tests/test_spp_consumers.py", {
        "pins": lambda name: digest(source(name).encode()),
    }, {})
    record(ROOT / "woof/core/kernel_frame_recordings.py", {}, {
        "TERRAIN_DRAG_COMPOSED_SOURCE_SHA256": digest(terrain_drag.module_source().encode()),
    })
    compiled = {}
    for name in args.compile:
        options = (*kernels.module_options(name), "-ftz=true")
        ptx = compile_ptx(source(name), options, "compute_120", name)
        functions = ptx_functions(ptx)
        compiled[name] = {"source_sha256": digest(source(name).encode()),
                          "options": options, "arch": "compute_120",
                          "ptx_sha256": digest(ptx.encode()),
                          "entry_count": len(functions["entry"]),
                          "function_count": len(functions["func"])}
    if args.write:
        for path, dictionaries, constants in pending:
            patch_file(path, dictionaries, constants, [], True)
    # Complete the pin closure when an intake adds translation units or
    # headers. Existing values still flow through patch_file and its ledger.
    freeze_path = ROOT / "tests/test_kernel_source_freeze_per_module.py"
    def dictionary_keys(path, name):
        for node in ast.parse(path.read_text()).body:
            label, value = assignment(node)
            if label == name:
                return {ast.literal_eval(key) for key in value.keys}
        raise ValueError(f"missing dictionary {name} in {path}")
    frozen = {"FROZEN_MODULE_DIGESTS": dictionary_keys(
        ROOT / "tests/test_mp8_frozen.py", "FROZEN_MODULE_DIGESTS")}
    closure = {name: dictionary_keys(freeze_path, name)
               for name in ("BASELINE_PINNED", "PINNED_HEADERS", "COMPOSED_HEADERS")}
    extension_path = ROOT / "tests/data/kernel_source_pins.json"
    extension_pins = json.loads(extension_path.read_text())
    additions = {
        "COMPOSED_HEADERS": {name: digest((KDIR / name).read_bytes())
            for name in ("rrtmgp_planck_common_default.cuh", "ysu_topo_default.cuh")
            if name not in closure["COMPOSED_HEADERS"]},
        "BASELINE_PINNED": {p.stem: digest(p.read_bytes()) for p in KDIR.glob("*.cu")
            if p.stem not in frozen["FROZEN_MODULE_DIGESTS"]
            and p.stem not in closure["BASELINE_PINNED"]
            and p.stem not in extension_pins["modules"]},
        "PINNED_HEADERS": {name: digest((KDIR / name).read_bytes())
            for name in live_headers if name not in closure["PINNED_HEADERS"]},
    }
    closure_text = freeze_path.read_text()
    closure_lines = closure_text.splitlines(keepends=True)
    insertions = []
    for node in ast.parse(closure_text).body:
        name, value = assignment(node)
        if name not in additions or not additions[name]: continue
        payload = "    # r290 exact-chain intake: source identity, numerical gates separate.\n"
        for key, sha in sorted(additions[name].items()):
            payload += f"    {key!r}: {sha!r},\n"
            readings.append({"file": str(freeze_path.relative_to(ROOT)),
                "pin": name + "." + key, "before": "UNPINNED", "after": sha})
        offset = sum(map(len, closure_lines[:value.end_lineno - 1]))
        insertions.append((offset, payload))
    if args.write:
        for offset, payload in sorted(insertions, reverse=True):
            closure_text = closure_text[:offset] + payload + closure_text[offset:]
        freeze_path.write_text(closure_text)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    args.out.write_text(json.dumps({"revision": revision, "written": args.write,
                                   "pin_changes": readings, "compiled": compiled},
                                  indent=2) + "\n", encoding="utf-8")
    print(f"{'wrote' if args.write else 'read'} {len(readings)} changed pins; "
          f"compiled {len(compiled)} PTX units")


if __name__ == "__main__":
    main()
