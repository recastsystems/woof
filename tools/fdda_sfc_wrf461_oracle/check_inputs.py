#!/usr/bin/env python3
"""Exercise the Rust-backed reader on an actual native auxinput9 file."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("--report",required=True,type=Path)
    args = parser.parse_args()
    from woof.netcdf_bridge import resolve_netcdf_bin
    from woof.spectral_nudging.surface_inputs import read_native_surface

    record = read_native_surface(args.input,surface=True,px=True)
    assert len(record.fields) == 14
    fields = {}
    for name,value in record.fields.items():
        assert value.dtype.kind == "f" and value.dtype.itemsize == 4
        assert not value.flags.writeable
        fields[name] = {"shape":list(value.shape),"dtype":str(value.dtype),
                        "sha256":hashlib.sha256(value.tobytes()).hexdigest()}
    refused = []
    for kwargs in ({"record":-1},{"record":True},{"record":10**9},
                   {"surface":False,"px":False}):
        try:
            read_native_surface(args.input,**kwargs)
        except ValueError as error:
            refused.append({"request":kwargs,"error":str(error)})
        else:
            raise AssertionError(f"invalid input request was accepted: {kwargs}")
    binary = resolve_netcdf_bin()
    receipt = {"input":os.fspath(args.input),"timestamp":record.timestamp,
               "rust_decoder":os.fspath(binary),
               "rust_decoder_sha256":hashlib.sha256(binary.read_bytes()).hexdigest(),
               "fields":fields,"refused_requests":refused}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()
