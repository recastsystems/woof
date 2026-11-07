"""Grade a real-run SLAB1D capture through the production CUDA launcher."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from woof.numerics import mode
from woof.verify import slab_oracle
from woof.wrf_exact import ENABLED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    oracle = slab_oracle.load_native_capture(args.input, args.output)
    actual = slab_oracle.run_cuda_columns(oracle, constants=oracle.constants)
    report = slab_oracle.compare(actual, oracle.output)
    receipt = {
        'source': 'real native WRF SLAB1D call, capture_native.py',
        'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'numerics': mode(), 'wrf_exact': ENABLED,
        'columns': len(oracle.cases), 'steps': oracle.nstep,
        'words': int(actual.size),
        'differing_words': sum(row['differing'] for row in report.values()),
        'max_absolute': float(np.max(np.abs(actual - oracle.output))),
        'report': report,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    if mode() != 'strict' or not ENABLED or receipt['differing_words']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
