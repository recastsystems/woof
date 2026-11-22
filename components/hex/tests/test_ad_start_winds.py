# SPDX-License-Identifier: Apache-2.0
"""The physics' start cell winds folded into dJ/du (woof.hex.ad.startwinds), on a made-up plan of 4 cells.

The fold must (1) find the arrays the physics copies the rebuild's zonal and meridional outputs into, also when they sit
at an offset inside a shared allocation, (2) apply exactly the transpose of the rebuild the kernel does, (3) count a
step-1 reader wherever the command names the pointer (kernel argument or copy-plan row), and (4) say no fold is needed
for a plan without the rebuild. The door adds those allocations to its dump list and folds the term into dJ/du."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import ad_doors as D
from woof.hex.ad import startwinds as W

NL, NC, NE, ME = 2, 4, 7, 3          # kernel cells include one padding cell (index 3); the physics arrays keep 3
EOC = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 6, 6]], np.int32)
NEO = np.array([3, 3, 3, 0], np.int32)
POOL_OFF = 64                        # the physics arrays sit 64 bytes into a shared allocation, rows of NC - 1 words


def _i32(v):
    return {"kind": "bytes", "bytes_hex": int(v).to_bytes(4, "little", signed=True).hex()}


def _dev(a, off=0):
    return {"kind": "device", "target": {"allocation": a, "offset": off}}


def _arr(a, off, width, row):
    words = [NL * width, NL, width, 4 * row, 4]
    hx = (0).to_bytes(8, "little") + b"".join(w.to_bytes(8, "little") for w in words)
    return {"kind": "patched", "bytes_hex": hx.hex(), "pointers": [{"byte_offset": 0, "target": {"allocation": a, "offset": off}}]}


def _plan(tmp_path, reader_first=True, with_kernel=True):
    rng = np.random.default_rng(5)
    co = rng.standard_normal((NC, ME, 3)).astype(np.float32)
    lat = rng.uniform(-1, 1, NC).astype(np.float32)
    lon = rng.uniform(0, 6, NC).astype(np.float32)
    blobs = [EOC.tobytes(), NEO.tobytes(), co.tobytes(), lat.tobytes(), lon.tobytes()]
    offs = np.cumsum([0] + [len(b) for b in blobs])[:-1]
    (tmp_path / "mesh.bin").write_bytes(b"".join(blobs))
    u = rng.standard_normal((NL, NE)).astype(np.float32)
    R = np.zeros((2 * NC, NE))
    for c in range(NC):
        for j in range(NEO[c]):
            e = EOC[c, j]
            cx, cy, cz = (float(x) for x in co[c, j])
            sl, cl, so, cs = math.sin(lat[c]), math.cos(lat[c]), math.sin(lon[c]), math.cos(lon[c])
            R[c, e] += -so * cx + cs * cy
            R[NC + c, e] += -sl * (cs * cx + so * cy) + cl * cz
    shared = np.zeros(16 + 2 * NL * (NC - 1), np.float32)   # zonal then meridional, both inside allocation 20
    z = np.stack([(R @ u[k])[:NC - 1] for k in range(NL)]).astype(np.float32)
    m = np.stack([(R @ u[k])[NC:2 * NC - 1] for k in range(NL)]).astype(np.float32)
    shared[16:16 + z.size] = z.ravel()
    shared[16 + z.size:] = m.ravel()
    (tmp_path / "shared.bin").write_bytes(shared.tobytes())
    al = [{"id": 2, "bytes": 4096, "initial": {"path": "mesh.bin"}},
          {"id": 20, "bytes": shared.nbytes, "initial": {"path": "shared.bin"}}]
    al += [{"id": i, "bytes": 512} for i in (1, 10, 11, 12, 13, 14, 15)]
    rec = {"kind": "kernel", "function": "reconstruct_winds_v841_f32",
           "arguments": [_i32(NL), _i32(NC), _i32(NE), _i32(ME), _dev(1)] + [_dev(2, int(o)) for o in offs]
                        + [_dev(2, int(o)) for o in offs] + [_dev(10), _dev(11), _dev(12), _dev(13), _dev(14), _dev(1)]}
    m_off = POOL_OFF + 4 * NL * (NC - 1)
    copies = [  # a one-column copy of the output first (the padding column): not the physics array
              {"kind": "kernel", "function": "cupy_copy__float32_float32",
               "arguments": [_arr(13, 8, 1, NC), _arr(15, 0, 1, 1)]},
              {"kind": "kernel", "function": "cupy_copy__float32_float32",
               "arguments": [_arr(13, 0, NC - 1, NC), _arr(20, POOL_OFF, NC - 1, NC - 1)]},
              {"kind": "kernel", "function": "cupy_copy__float32_float32",
               "arguments": [_arr(14, 0, NC - 1, NC), _arr(20, m_off, NC - 1, NC - 1)]}]
    reader = {"kind": "rrfs_copy_plan_upload", "rows": [{"src": {"allocation": 20, "offset": POOL_OFF},
                                                         "dst": {"allocation": 15, "offset": 0}}]}
    if not with_kernel:
        s1 = [reader]
    else:
        s1 = ([reader, rec] + copies) if reader_first else ([rec] + copies + [reader])
    (tmp_path / "plan.json").write_text(json.dumps({"allocations": al, "blocks": {"step_01": s1, "step_02": s1}}))
    return R, u, z, m


def test_fold_is_the_rebuild_transpose(tmp_path):
    R, u, z, m = _plan(tmp_path)
    info = W.plan_info(tmp_path)
    assert info["needed"] and W.dump_allocations(info) == [20]
    assert [(p["alloc"], p["offset"], p["width"]) for p in info["pools"]] == [(20, POOL_OFF, NC - 1),
                                                                              (20, POOL_OFF + 4 * NL * (NC - 1), NC - 1)]
    assert info["step1_reads_before_rebuild"] == 1
    assert max(W.start_check(info, tmp_path, u.astype(np.float64)).values()) < 1e-6
    rng = np.random.default_rng(1)
    gz, gm = rng.standard_normal((NL, NC - 1)), rng.standard_normal((NL, NC - 1))
    flat = np.zeros(16 + 2 * NL * (NC - 1), np.float32)
    flat[16:16 + gz.size] = gz.ravel()
    flat[16 + gz.size:] = gm.ravel()
    (tmp_path / "grad-20.f32").write_bytes(flat.tobytes())
    t = W.term(info, tmp_path, tmp_path, NE - 1)
    gzf, gmf = flat[16:16 + gz.size].reshape(NL, NC - 1), flat[16 + gz.size:].reshape(NL, NC - 1)
    want = np.stack([R.T @ np.concatenate([gzf[k], [0.0], gmf[k], [0.0]]) for k in range(NL)])[:, :NE - 1]
    assert np.allclose(t, want, rtol=1e-6, atol=1e-6)


def test_rebuild_first_has_no_reader(tmp_path):
    _plan(tmp_path, reader_first=False)
    info = W.plan_info(tmp_path)
    assert info["step1_reads_before_rebuild"] == 0   # a rank plan: the start words are overwritten unread


def test_no_rebuild_no_fold(tmp_path):
    _plan(tmp_path, with_kernel=False)
    info = W.plan_info(tmp_path)
    assert not info["needed"] and W.dump_allocations(info) == [] and W.term(info, tmp_path, tmp_path, NE) is None


def test_missing_dump_refuses(tmp_path):
    _plan(tmp_path)
    with pytest.raises(FileNotFoundError):
        W.term(W.plan_info(tmp_path), tmp_path, tmp_path, NE)


def test_door_dumps_the_start_winds(tmp_path, monkeypatch):
    _plan(tmp_path)
    monkeypatch.setenv(D.OBJECTIVES_ENV, str(tmp_path / "kept" / "objectives"))
    u = {"name": "u", "copies": [(1, 0)], "field": "u"}
    fmap = SimpleNamespace(fields={"u": {}}, doc={"plan_sha256": "ab" * 32}, resolve=lambda table, n: u)
    table = {"sensitivity_inputs": ["u"], "fields": {"u": {"field": "u"}}}
    ctx = SimpleNamespace(fmap=fmap, table=table, plan=tmp_path, split=None)
    inputs, allocs = D._sensitivity_inputs(ctx)
    assert allocs == [1, 20]
    kept = tmp_path / "kept" / "start-cell-winds" / ("ab" * 32 + ".json")
    assert json.loads(kept.read_text())["needed"]
    ctx2 = SimpleNamespace(fmap=fmap, table=table, plan=tmp_path, split={"ranks": 2})
    assert D._sensitivity_inputs(ctx2)[1] == [1]   # a split window carries the term itself
