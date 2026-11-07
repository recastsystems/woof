import json
from woof.wrf_exact import ENABLED as EXACT
from pathlib import Path
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from woof.core.kernels import get_kernel, module_source, compile_noftz_module
from woof.core.fp32_ulp import monotone_fp32_key

FIXTURE = Path(__file__).parent / "fixtures/sstskin/columns.json"

def capture(mutate=False):
    fixture = json.loads(FIXTURE.read_text())
    kernel = get_kernel("sstskin", "sstskin_column")
    if mutate:
        source = module_source("sstskin").replace("// f1 absorption", "f1 = __uint_as_float(__float_as_uint(f1) ^ 1u);")
        assert source != module_source("sstskin")
        kernel = compile_noftz_module("sstskin", source, "woof:sstskin:absorption-mutation").get_function("sstskin_column")
    results=[]
    for arm in fixture["arms"]:
        fields = {key:cp.asarray(np.asarray(words,dtype=np.uint32).view(np.float32).reshape(1,-1)) for key,words in fixture["initial"].items()}
        snapshots=[]
        for step in fixture["flux"]:
            fields.update({key:cp.asarray(np.asarray(words,dtype=np.uint32).view(np.float32).reshape(1,-1)) for key,words in step.items()})
            args = tuple(fields[key] for key in ("xland","glw","gsw","hfx","qfx","tsk","ust","emiss","sst","dtw","sstsk","tslb"))
            kernel((1,),(128,),args+(np.float32(fixture["dt"]),np.float32(fixture["stbolt"]),np.int32(arm["sst_update"]),np.int32(fixture["columns"])))
            snapshots.append(np.stack([cp.asnumpy(fields[key]).ravel() for key in ("sstsk","dtw","tsk","tslb")]))
        got = np.asarray(snapshots,dtype=np.float32)
        expected = np.asarray(arm["expected_words"],dtype=np.uint32).view(np.float32)
        results.append((arm["sst_update"],got,expected))
    return results

def metrics(got,expected):
    a,b=monotone_fp32_key(got),monotone_fp32_key(expected)
    return {"misses":int(np.count_nonzero(got.view(np.uint32)!=expected.view(np.uint32))),"max_ulp":int(np.max(np.abs(a-b))),"max_abs":float(np.max(np.abs(got-expected)))}

def test_column_words():
    receipt={"arms":[{"sst_update":update,"fields":{key:metrics(got[:,i],expected[:,i]) for i,key in enumerate(("SSTSK","DTW","TSK","TSLB"))}} for update,got,expected in capture()]}
    print(json.dumps(receipt,sort_keys=True))
    if EXACT:
        assert all(row["misses"]==0 for arm in receipt["arms"] for row in arm["fields"].values())
    else:
        for arm in receipt["arms"]:
            for key, row in arm["fields"].items():
                # RTX 5090 / NVRTC 13.4.59: the old FTZ image reaches
                # 2.384185791015625e-6 DTW error; IEEE division lowers it
                # to 2.1457672119140625e-6. Keep default FMA and native
                # strict word equality; the measured fast bound is 2.5e-6.
                assert row["max_abs"] <= {"SSTSK":6.2e-5,"DTW":2.5e-6,"TSK":6.2e-5,"TSLB":0.}[key]

def test_one_bit_absorption_is_caught():
    normal = capture()
    changed = capture(mutate=True)
    differences = [metrics(m[1],n[1]) for n,m in zip(normal,changed)]
    print(json.dumps({"one_bit_absorption": differences}, sort_keys=True))
    assert any(row["misses"] > 0 for row in differences)
