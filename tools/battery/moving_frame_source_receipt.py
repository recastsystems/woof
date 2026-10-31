"""Re-earn the dycore source seal with unchanged maintained word captures.

Run the coordinate, driver and acoustic captures once on each revision first.
No native fixture or historical receipt is replaced; any changed word refuses
the extension.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from tools.battery import fusion_source_receipt as captures

ROOT = Path(__file__).resolve().parents[2]
BASE = "afe2905d568d41880ee04a1a26d3efeb80791034"
SOURCE = "woof/core/dycore.py"
DEPENDENCIES = ("woof/moving_frame.py", "woof/config.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    # Reuse the maintained measurements with this extension's reference tree.
    captures.BASE = BASE
    tip = captures.git("rev-parse", "HEAD").decode().strip()
    measurements, artifacts, identities, inputs = captures.measure(args.replay.resolve(), tip)
    folder = ROOT / "tests/data/assembled_legacy_proofs"
    extension = captures.read(folder / "source-extension.json")
    before, after = (captures.source_pin(revision, SOURCE) for revision in (BASE, tip))
    assert extension["sources"][SOURCE]["after"] == before and before != after
    for name, transition in extension["sources"].items():
        if name != SOURCE:
            assert transition["after"] == captures.source_pin(tip, name), name
    result = dict(
        schema="measured-moving-frame-source-replay-v1", base=BASE, tip=tip,
        sources={SOURCE: dict(before=before, after=after)},
        dependencies={name: captures.source_pin(tip, name) for name in DEPENDENCIES},
        measurements=measurements, artifacts=artifacts,
        capture_identity=identities["tip-coordinate"], source_inputs=inputs,
        generator=captures.GENERATOR, generator_sha256=captures.sha(ROOT / captures.GENERATOR),
        recorder="tools/battery/moving_frame_source_receipt.py", recorder_sha256=captures.sha(__file__),
        measurement_tool="tools/battery/fusion_source_receipt.py",
        measurement_tool_sha256=captures.sha(ROOT / "tools/battery/fusion_source_receipt.py"),
        numerics="strict", scope="default frame; unchanged base/tip words; coordinate native exact and acoustic published exact",
    )
    payload = (json.dumps(result, indent=2) + "\n").encode()
    args.out.write_bytes(payload)
    if args.write:
        receipt = folder / "moving-frame-source-replay.json"
        receipt.write_bytes(payload)
        extension["sources"][SOURCE]["after"] = after
        extension["moving_frame_source_replay"] = dict(receipt=receipt.name, sha256=captures.sha(receipt))
        extension["measured_tip"] = tip
        path = folder / "source-extension.json"
        path.write_bytes((json.dumps(extension, indent=2) + "\n").encode())
        helper = ROOT / "tests/assembled_legacy_proofs.py"
        lines = helper.read_text().splitlines(keepends=True)
        node = next(node for node in ast.parse("".join(lines)).body if isinstance(node, ast.Assign)
                    and any(getattr(target, "id", None) == "SOURCE_EXTENSION_SHA" for target in node.targets))
        lines[node.lineno-1:node.end_lineno] = [f"SOURCE_EXTENSION_SHA = {captures.sha(path)!r}\n"]
        helper.write_bytes("".join(lines).encode())
    print(json.dumps(dict(measurements=measurements, written=args.write), indent=2))


if __name__ == "__main__":
    main()
