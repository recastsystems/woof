#!/usr/bin/env python3
"""Compose the two observation writers without changing native arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path

from instrument_nested import instrument, PIN
from instrument_reset import insert_calls, writer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    path=a.build/'share/dfi.F'
    original=path.read_bytes()
    text=insert_calls(instrument(original))+writer()
    path.write_text(text)
    a.receipt.write_text(json.dumps(dict(original_sha256=PIN,
        instrumented_sha256=hashlib.sha256(text.encode()).hexdigest(),
        native_arithmetic_changed=False,
        scope='all-domain accumulation; root full-column save/clear/reset/rebalance'),indent=2)+'\n')


if __name__=='__main__':main()
