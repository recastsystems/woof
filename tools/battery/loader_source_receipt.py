"""Seal a slimming loader change only after maintained measured word replays.

First run tools/assembled_legacy_replay.py for coordinate, drivers and acoustic
on the reference and integrated trees with GPUWM_NUMERICS=strict. Supply the
six output folders as REPLAY/{base,tip}-{coordinate,drivers,acoustic}. Historical
WRF and GPU receipts stay immutable. Any changed output refuses all writes.

--moved-kernel MODULE=PATH admits a CUDA kernel source whose bytes moved (a path the
replayed configuration does not take, e.g. a GPUWM_WRF_EXACT-only branch): the module's
assembled-source digest may differ between the two arms, every other capture identity
key and every output word must not, and the transition is recorded in the extension.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {"bytes": path.stat().st_size, "sha256": sha(path)}


def dictionary(path, name):
    from tools.kernel_ptx_identity.refresh_stack_pins import assignment
    return next(ast.literal_eval(value) for node in ast.parse(path.read_text()).body
                for label, value in [assignment(node)] if label == name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--moved", action="append", default=[],
                        help="another source-extension source whose bytes moved since the last seal; "
                             "its after-pin is re-earned by these same zero-difference replays")
    parser.add_argument("--moved-kernel", action="append", default=[], metavar="MODULE=PATH",
                        help="a kernel source whose bytes moved; its module digest may differ between the arms "
                             "only while every output word is unchanged")
    args = parser.parse_args()
    moved_kernels = dict(item.split("=", 1) for item in args.moved_kernel)
    from tools.assembled_legacy_replay import word_hashes
    from tools.kernel_ptx_identity.refresh_stack_pins import patch_file, assignment
    from tools.mp8_freeze_receipt import capture_compiled_source, kernel_source_path
    from woof.core import kernels

    measurements, artifacts, identities = {}, {}, {}
    for family, count, words in (("coordinate", 272, 178128),
                                  ("drivers", 2618, 31686592),
                                  ("acoustic", 132, 305632)):
        rows = []
        for arm, tree in (("base", args.base), ("tip", ROOT)):
            folder = args.replay / f"{arm}-{family}"
            audit = read(folder / "source-audit.json")
            identity_keys = ("device", "compute_capability", "nvrtc_version", "cupy_version",
                             "module_source_sha256", "module_options", "strict", "default_diffusion",
                             "acoustic_function_options", "default_acoustic_diffusion_route")
            identities[f"{arm}-{family}"] = {key: audit[key] for key in identity_keys}
            assert audit["compute_capability"][0] >= 10
            assert audit["strict"] is False and audit["source_fingerprint_unchanged"]
            assert audit["engine_commit"] == subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=tree, text=True).strip()
            files = [folder / "source-audit.json"]
            if family == "coordinate":
                proof = read(folder / "direct-native-proof.json")
                for name, recorded in proof["source_inputs"].items():
                    assert pin(tree / name) == recorded, (arm, name)
                assert proof["native_different_words"] == 0 and proof["native_exact"]
                assert proof["changed_previously_native_equal_words"] == 0
                actual = {key: {"sha256": row["current_sha256"], "words": row["current_to_native"]["words"]}
                          for key, row in proof["fields"].items()}
                files.append(folder / "direct-native-proof.json")
            elif family == "drivers":
                assert audit["status"] == "complete"
                actual = {}
                for name in ("horizontal-driver", "vertical-driver", "km-mutations"):
                    path = folder / f"{name}.json"
                    files.append(path)
                    actual.update({name+key: {"sha256": row["gpu_sha256"], "words": row["words"]}
                                   for key, row in word_hashes(read(path)["cases"]).items()})
            else:
                proof = read(folder / "acoustic-published-default-proof.json")
                for name, recorded in proof["source_inputs"].items():
                    assert pin(tree / name) == recorded, (arm, name)
                assert proof["changed_published_words"] == 0 and proof["published_exact"]
                actual = {key: {"sha256": row["actual_sha256"], "words": row["words"]}
                          for key, row in proof["fields"].items()}
                files.append(folder / "acoustic-published-default-proof.json")
            assert (len(actual), sum(row["words"] for row in actual.values())) == (count, words)
            for path in files:
                artifacts[str(path.relative_to(args.replay))] = sha(path)
            rows.append(actual)
        def held(identity):
            held = dict(identity, module_source_sha256={
                module: digest for module, digest in identity["module_source_sha256"].items()
                if module not in moved_kernels})
            return held
        assert held(identities[f"base-{family}"]) == held(identities[f"tip-{family}"]), family
        assert rows[0] == rows[1], f"STOP: {family} output words changed"
        measurements[family] = dict(arrays=count, words=words, changed_arrays=0,
                                    different_words=0, changes={})

    loader = "woof/core/kernels/__init__.py"
    # Read both raw and assembled source through their actual source trees.
    code = 'import json; from woof.core.kernels import module_source; print(json.dumps(module_source("ysu")))'
    base_source = json.loads(subprocess.check_output([sys.executable, "-c", code],
        cwd=args.base, env=os.environ | {"PYTHONPATH": str(args.base)}, text=True))
    current_source, method = capture_compiled_source("ysu")
    assert method == "loader-capture" and current_source == base_source, "YSU compiler source moved"
    assert kernel_source_path("ysu").read_bytes() == (args.base / "woof/core/kernels/ysu_default.cu").read_bytes()

    folder = ROOT / "tests/data/assembled_legacy_proofs"
    extension = read(folder / "source-extension.json")
    moved = sorted(set(args.moved) - {loader})
    for name in moved:
        assert name in extension["sources"], name
        assert pin(ROOT / name) != extension["sources"][name]["after"], f"{name} did not move"
    kernel_moves = {}
    for module, name in sorted(moved_kernels.items()):
        before_digest = identities["base-acoustic"]["module_source_sha256"][module]
        after_digest = identities["tip-acoustic"]["module_source_sha256"][module]
        assert before_digest != after_digest and pin(args.base / name) != pin(ROOT / name), f"{name} did not move"
        for family in ("coordinate", "drivers", "acoustic"):
            assert identities[f"base-{family}"]["module_source_sha256"][module] == before_digest, (family, module)
            assert identities[f"tip-{family}"]["module_source_sha256"][module] == after_digest, (family, module)
        kernel_moves[name] = dict(module=module, before=pin(args.base / name), after=pin(ROOT / name),
                                  module_source_sha256=dict(before=before_digest, after=after_digest))
    for name, transition in extension["sources"].items():
        if name != loader and name not in moved and name not in kernel_moves:
            assert pin(ROOT / name) == transition["after"], name
    result = dict(schema="measured-slimming-loader-source-replay-v1",
                  base=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.base, text=True).strip(),
                  tip=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  source=loader, before=pin(args.base / loader), after=pin(ROOT / loader),
                  generator="tools/assembled_legacy_replay.py",
                  generator_sha256=sha(ROOT / "tools/assembled_legacy_replay.py"),
                  recorder="tools/battery/loader_source_receipt.py", recorder_sha256=sha(Path(__file__)),
                  numerics="strict", measurements=measurements, artifacts=artifacts,
                  capture_identity=identities["tip-coordinate"],
                  ysu_compiled_source_sha256=hashlib.sha256(current_source.encode()).hexdigest())
    if moved:
        result["moved_sources"] = {name: {"sealed": extension["sources"][name]["after"],
                                          "after": pin(ROOT / name)} for name in moved}
    if kernel_moves:
        result["moved_kernels"] = kernel_moves

    readings = []
    mp8 = ROOT / "tests/test_mp8_frozen.py"
    frozen = dictionary(mp8, "FROZEN_MODULE_DIGESTS")
    # Only YSU's physical authority moved; its assembled compile string did not.
    frozen["ysu"] = (sha(kernel_source_path("ysu")), frozen["ysu"][1])
    assert frozen["ysu"][1] == result["ysu_compiled_source_sha256"]
    patch_file(mp8, {"FROZEN_MODULE_DIGESTS": frozen.get}, {}, readings, False)
    spp = ROOT / "tests/test_spp_consumers.py"
    spp_tree = ast.parse(spp.read_text())
    spp_pins = next(ast.literal_eval(node.value) for node in ast.walk(spp_tree)
                    if isinstance(node, ast.Assign) and any(getattr(t,"id",None)=="pins" for t in node.targets))
    spp_pins["mynn_pbl"] = frozen["mynn_pbl"][1]
    assert hashlib.sha256(kernels.module_source("mynn_pbl").encode()).hexdigest() == spp_pins["mynn_pbl"]
    patch_file(spp, {"pins": spp_pins.get}, {}, readings, False)
    closure = ROOT / "tests/test_kernel_source_freeze_per_module.py"
    header = "ysu_topo_default.cuh"
    assert sha(kernels._KDIR / header) == dictionary(closure, "COMPOSED_HEADERS")[header]
    assert header in kernels.EXTRA_HEADERS["ysu"]
    header_node = next(value for item in ast.parse(closure.read_text()).body
                       for label, value in [assignment(item)] if label == "PINNED_HEADERS")
    needs_header = header not in {ast.literal_eval(key) for key in header_node.keys}
    if args.write:
        patch_file(mp8, {"FROZEN_MODULE_DIGESTS": frozen.get}, {}, [], True)
        patch_file(spp, {"pins": spp_pins.get}, {}, [], True)
    if args.write and needs_header:
        text = closure.read_text()
        node = next(value for item in ast.parse(text).body for label,value in [assignment(item)] if label=="PINNED_HEADERS")
        lines = text.splitlines(keepends=True)
        offset = sum(map(len, lines[:node.end_lineno-1]))
        text = text[:offset] + f'    # Slimming promotes the retained YSU header to the loader closure.\n    {header!r}: {sha(kernels._KDIR / header)!r},\n' + text[offset:]
        closure.write_text(text)
    result["pin_changes"] = readings
    payload = json.dumps(result, indent=2) + "\n"
    args.out.write_text(payload)
    if args.write:
        receipt = folder / "slimming-loader-source-replay.json"
        receipt.write_text(payload)
        extension["sources"][loader]["after"] = pin(ROOT / loader)
        for name in moved:
            extension["sources"][name]["after"] = pin(ROOT / name)
        for name, move in kernel_moves.items():
            transition = extension["sources"].setdefault(name, {"before": move["before"]})
            transition["after"] = move["after"]
            extension.setdefault("module_sources", {})[move["module"]] = {
                "before": move["module_source_sha256"]["before"], "after": move["module_source_sha256"]["after"],
                "source": name}
        if "woof/core/dycore.py" in moved:
            # This replay re-earns dycore.py's whole move from the base, so the per-row seals of earlier,
            # intermediate dycore.py bytes no longer describe the current file.
            extension.pop("moving_frame_source_replay", None)
            extension.pop("forecast_entry_replay", None)
        extension["measured_tip"] = result["tip"]
        extension["measurements"] = measurements
        extension["loader_slimming_replay"] = {"receipt": receipt.name, "sha256": sha(receipt)}
        extension_path = folder / "source-extension.json"
        extension_path.write_text(json.dumps(extension, indent=2)+"\n")
        patch_file(ROOT / "tests/assembled_legacy_proofs.py", {},
                   {"SOURCE_EXTENSION_SHA": sha(extension_path)}, [], True)
    print(json.dumps({"measurements": measurements, "pin_changes": readings, "written": args.write}), flush=True)


if __name__ == "__main__":
    main()
