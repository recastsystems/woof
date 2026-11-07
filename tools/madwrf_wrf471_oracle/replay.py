"""Replay entry/exit captures from whole strict WRF 4.7.1 real-data runs.

Native stream files use the build's big-endian REAL32/INTEGER32 encoding.
Only physical mass levels are passed to the port; WRF's extra storage
level is retained in the captures and is not a physical output level.
"""
from pathlib import Path
import argparse
import json
import os
import numpy as np


class Reader:
    def __init__(self, path):
        self.file = Path(path).open("rb")

    def read(self, count, kind="f4"):
        result = np.fromfile(self.file, dtype=">" + kind, count=count)
        if len(result) != count:
            raise ValueError("Truncated native capture")
        return result.astype(kind)

    def f3(self, ni, nj, nk):
        return self.read(ni*nj*nk).reshape((ni,nk,nj), order="F").transpose(1,2,0).copy()

    def f2(self, ni, nj):
        return self.read(ni*nj).reshape((ni,nj), order="F").T.copy()

    def finish(self):
        assert not self.file.read(1), "Trailing capture bytes"
        self.file.close()


def compare(actual, expected):
    a = np.asarray(actual, dtype=np.float32).view(np.uint32)
    b = np.asarray(expected, dtype=np.float32).view(np.uint32)
    assert a.shape == b.shape
    bad = a != b
    first = np.argwhere(bad)[:5]
    return {"words":int(a.size), "differing_words":int(bad.sum()),
            "examples":[{"index":p.tolist(),"actual":int(a[tuple(p)]),
                         "native":int(b[tuple(p)])} for p in first]}


def replay(root, output):
    if os.environ.get("GPUWM_WRF_EXACT") != "1":
        raise RuntimeError("Native word replay requires GPUWM_WRF_EXACT=1")
    import cupy as cp
    from woof.core.madwrf import init_madwrf_clouds_device, cloud_tracer_nudge_device
    root = Path(root)
    ci = Reader(root/"mad-cloud.in")
    ni,nj,nk,fm,ft,fb,fbr,width,hold = map(int,ci.read(9,"i4"))
    dx,dy,p00 = ci.read(3)
    fields = {name:ci.f3(ni,nj,nk) for name in ("t","p","ph_2","phb","qv","qc","qi","qs")}
    fields.update({name:ci.f2(ni,nj) for name in ("xland","ht","cldmask","cldtopz","cldbasez","brtemp")})
    ci.finish()
    co = Reader(root/"mad-cloud.out")
    expected = {name:co.f3(ni,nj,nk)[:-1].copy() for name in ("cldfra","qv","qc","qi","qs")}
    expected.update({name:co.f2(ni,nj) for name in ("cldmask","cldtopz")})
    co.finish()
    device = {name:cp.asarray(value if name in ("ph_2","phb") or value.ndim==2 else value[:-1].copy())
              for name,value in fields.items()}
    cf = init_madwrf_clouds_device(**device,dx=dx,dy=dy,p00=p00,
        flag_cldmask=fm,flag_cldtopz=ft,flag_cldbasez=fb,flag_brtemp=fbr,
        em_width=width,hold_ups=bool(hold))
    actual = {name:cp.asnumpy(device[name]) for name in expected if name!="cldfra"}
    actual["cldfra"] = cp.asnumpy(cf[:-1])
    result = {"cloud_dimensions":[ni,nj,nk-1],"cloud_flags":[fm,ft,fb,fbr],
              "cloud":{name:compare(actual[name],value) for name,value in expected.items()}}
    ni_file = root/"mad-nudge.in"
    if ni_file.exists():
        reader = Reader(ni_file)
        ni,nj,nk = map(int,reader.read(3,"i4"))
        dt,relax,nudge,xtime = reader.read(4)
        values = {name:cp.asarray(reader.f3(ni,nj,nk)) for name in ("qc","qi","qs","tr_qc","tr_qi","tr_qs")}
        reader.finish()
        cloud_tracer_nudge_device(*values.values(),dt,relax,nudge,xtime)
        reader = Reader(root/"mad-nudge.out")
        result["nudge"] = {name:compare(cp.asnumpy(values[name]),reader.f3(ni,nj,nk))
                           for name in ("qc","qi","qs")}
        reader.finish()
    # real.exe's actual output is the reference for the tracer-copy hook.
    # Rust reads the NetCDF bytes; Python only selects and compares words.
    if (root/"wrfinput_d01").exists():
        from types import SimpleNamespace
        from woof import netcdf_bridge
        from woof.core.madwrf_runtime import initialize
        pairs = (("QCLOUD","tr_qcloud","qc","tr_qc"),
                 ("QICE","tr_qice","qi","tr_qi"),
                 ("QSNOW","tr_qsnow","qs","tr_qs"))
        with netcdf_bridge.open_dataset(root/"wrfinput_d01") as dataset:
            if all(tr in dataset.variables for _,tr,_,_ in pairs):
                state = SimpleNamespace()
                references = {}
                for water,tr,wn,tn in pairs:
                    native = np.asarray(dataset.variables[water][:],np.float32)[0].copy()
                    setattr(state,wn,cp.asarray(native))
                    setattr(state,tn,cp.zeros_like(getattr(state,wn)))
                    references[tn] = np.asarray(dataset.variables[tr][:],np.float32)[0].copy()
                initialize(state,SimpleNamespace(madwrf_opt=2,madwrf_cldinit=0))
                result["tracer_initialization"] = {
                    name:compare(cp.asnumpy(getattr(state,name)),value)
                    for name,value in references.items()}
    rows = (list(result["cloud"].values()) + list(result.get("nudge",{}).values())
            + list(result.get("tracer_initialization",{}).values()))
    result["words"] = sum(row["words"] for row in rows)
    result["differing_words"] = sum(row["differing_words"] for row in rows)
    Path(output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    return result["differing_words"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("output")
    args = parser.parse_args()
    raise SystemExit(bool(replay(args.capture,args.output)))
