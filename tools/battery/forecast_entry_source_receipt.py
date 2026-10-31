"""Re-earn the dycore source seal after a forecast entry-admission change.

Uses the maintained three-family base/tip replay recorder. No recorded words,
native fixtures, numerical limits or historical receipts are replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'woof/core/dycore.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path):
    return dict(bytes=path.stat().st_size, sha256=sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    folder = ROOT / 'tests/data/assembled_legacy_proofs'
    extension_path = folder / 'source-extension.json'
    extension = json.loads(extension_path.read_text())
    before = extension['sources'][SOURCE]['after']
    assert pin(args.base / SOURCE) == before, 'base must be the previously sealed source'
    subprocess.run([
        sys.executable, str(ROOT / 'tools/battery/loader_source_receipt.py'),
        '--base', str(args.base), '--replay', str(args.replay),
        '--out', str(args.out), '--moved', SOURCE,
    ], cwd=ROOT, check=True)
    replay = json.loads(args.out.read_text())
    assert replay['base'] == 'afe2905d568d41880ee04a1a26d3efeb80791034'
    assert replay['before'] == replay['after'], 'the CUDA loader must not change'
    assert not replay['pin_changes'], 'this admission change moves no kernel-source pins'
    assert set(replay['moved_sources']) == {SOURCE}
    assert replay['moved_sources'][SOURCE]['sealed'] == before
    after = pin(ROOT / SOURCE)
    assert replay['moved_sources'][SOURCE]['after'] == after
    replay.update(purpose='forecast entry admission',
                  admission_recorder='tools/battery/forecast_entry_source_receipt.py',
                  admission_recorder_sha256=sha(Path(__file__)))
    payload = json.dumps(replay, indent=2) + '\n'
    args.out.write_text(payload)
    if args.write:
        receipt = folder / 'forecast-entry-source-replay.json'
        receipt.write_text(payload)
        extension['sources'][SOURCE]['after'] = after
        extension['forecast_entry_replay'] = dict(receipt=receipt.name,
                                                  sha256=sha(receipt), before=before)
        extension_path.write_text(json.dumps(extension, indent=2) + '\n')
        from tools.kernel_ptx_identity.refresh_stack_pins import patch_file

        patch_file(ROOT / 'tests/assembled_legacy_proofs.py', {},
                   {'SOURCE_EXTENSION_SHA': sha(extension_path)}, [], True)
    print(json.dumps(dict(measurements=replay['measurements'], written=args.write)), flush=True)


if __name__ == '__main__':
    main()
