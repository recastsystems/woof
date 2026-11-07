"""Capture framing must preserve native words and reject broken records."""
import gzip
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TOOL=ROOT/"tools/sas_wrf471_capture/canonicalize.py"
FIXTURE=ROOT/"tests/data/oracles/sas"


def test_capture_canonicalization_preserves_all_words_and_rejects_partial_record(tmp_path):
    capture=tmp_path/"capture";capture.mkdir()
    reference={key:gzip.decompress((FIXTURE/(key+".bin.gz")).read_bytes())
               for key in ("cases","stage3")}
    for order in ("little","big"):
        for key,data in reference.items():
            raw=(b"".join(data[i:i+4][::-1] for i in range(0,len(data),4))
                 if order=="big" else data)
            (capture/("sas-0-"+key+".bin")).write_bytes(raw)
        out=tmp_path/order
        subprocess.run([sys.executable,str(TOOL),str(capture),str(out),"--byte-order",order],
                       check=True,capture_output=True)
        for key,data in reference.items():
            assert (out/(key+".bin")).read_bytes()==data
    path=capture/"sas-0-stage3.bin"
    path.write_bytes(path.read_bytes()[:-4])
    run=subprocess.run([sys.executable,str(TOOL),str(capture),str(tmp_path/"broken"),
                        "--byte-order","big"],capture_output=True,text=True)
    assert run.returncode != 0
    assert "expected" in run.stderr or "buffer" in run.stderr
