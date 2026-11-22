# SPDX-License-Identifier: Apache-2.0
"""woof.hex.ad.target_loss on a small synthetic mesh (host arrays): the transposes and the gradient. The real-case
check is ``python -m woof.hex.ad.target_loss check`` on the 30 km cull (the torch bridge page has its numbers)."""
import numpy as np

from woof.hex.ad import target_loss as TL


def _mesh(nc=40, maxe=6, seed=3):
    rs = np.random.default_rng(seed)
    ne = nc                                                  # a ring: edge e joins cell e and cell e+1
    coe = np.stack([np.arange(nc), (np.arange(nc) + 1) % nc], 1) + 1
    coe[-1, 1] = nc + 1                                      # one edge leaves the mesh (a limited-area edge)
    eoc = np.zeros((nc, maxe), int)
    eoc[:, 0], eoc[:, 1] = np.arange(nc) + 1, (np.arange(nc) - 1) % nc + 1
    return {"nCells": nc, "nEdges": ne, "areaCell": rs.uniform(0.8, 1.2, nc) * 9e8, "cellsOnEdge": coe,
            "dvEdge": rs.uniform(0.9, 1.1, ne) * 1.7e4, "dcEdge": rs.uniform(0.9, 1.1, ne) * 3e4,
            "edgesOnCell": eoc, "nEdgesOnCell": np.full(nc, 2), "coeffs_reconstruct": rs.standard_normal((nc, maxe, 3)),
            "latCell": rs.uniform(0.5, 0.8, nc), "lonCell": rs.uniform(-1.8, -1.5, nc), "bdyMaskCell": np.zeros(nc, int)}


def _target(st, nz=5, seed=4):
    rs = np.random.default_rng(seed)
    nc = st["nCells"]
    y = {"theta": 300 + rs.standard_normal((nz, nc)), "qv": 0.01 * rs.uniform(0.2, 1, (nz, nc)),
         "uzonal": rs.standard_normal((nz, nc)), "umerid": rs.standard_normal((nz, nc)),
         "plow": 9e4 + 100 * rs.standard_normal((1, nc))}
    w3 = rs.uniform(0, 1, (nz, nc))
    w3 /= w3.sum()
    w = {f: (np.full((1, nc), 1.0 / nc) if f == "plow" else w3) for f in TL.FIELDS}
    sig = {"theta": np.ones(nz), "qv": np.full(nz, 1e-3), "uzonal": np.full(nz, 2.0), "umerid": np.full(nz, 2.0),
           "plow": np.full(1, 100.0)}
    return TL.Target(y, w, sig, {})


def test_transposes_and_gradient():
    st = _mesh()
    ops = TL.MeshOps(st, np)
    tg = _target(st)
    loss = TL.StateLoss(ops, tg, scales=((0.0, 1.0), (60e3, 0.5)))
    rs = np.random.default_rng(5)
    x = {f: tg.y[f] + tg.sigma[f][:, None] * rs.standard_normal(tg.y[f].shape) for f in TL.FIELDS}
    assert loss.value_and_grad(dict(tg.y), want_grad=False)[0] == 0.0
    nz = 5
    prog = {"theta_m": x["theta"] * (1 + TL.RVORD * x["qv"]), "qv": x["qv"],
            "u": rs.standard_normal((nz, st["nEdges"])), "pressure_p": 100 * rs.standard_normal((nz, st["nCells"])),
            "pressure_base_low": np.full(st["nCells"], 9e4)}
    ent = {"theta": x["theta"], "qv": x["qv"], "rho": rs.uniform(0.9, 1.1, (nz, st["nCells"])), "u": prog["u"]}
    res = TL.gradient_check(loss, x, prognostic=prog, entry=ent, hs=(1e-2, 1e-3))
    assert all(r["rel"] < 1e-9 for r in res["state"]), res["state"]
    assert res["prognostic"][-1]["rel"] < 1e-6, res["prognostic"]
    assert res["entry"][-1]["rel"] < 1e-6, res["entry"]
    assert all(v["rel"] < 1e-12 for v in res["transpose"].values()), res["transpose"]


def test_station_loss_gradient():
    st = _mesh(nc=60)
    rs = np.random.default_rng(7)
    nc = st["nCells"]
    lat, lon = np.degrees(st["latCell"]), np.degrees(st["lonCell"])
    obs = [{"id": str(i), "lat": lat[i] + 0.01, "lon": lon[i] - 0.01, "elev": 200.0, "t2m": 290 + rs.standard_normal(),
            "td2m": 280 + rs.standard_normal(), "u10": rs.standard_normal(), "v10": rs.standard_normal()}
           for i in range(0, nc, 3)]
    sl = TL.StationLoss(st, np.full(nc, 230.0), np.full(nc, 200.0), np.full(nc, 95000.0), obs)
    assert len(sl.obs) > 0
    x = [295 + rs.standard_normal(nc), 0.008 + 0.001 * rs.standard_normal(nc), rs.standard_normal(nc), rs.standard_normal(nc)]
    _, _, g = sl.value_and_grad(*x)
    d = [rs.standard_normal(nc), 1e-4 * rs.standard_normal(nc), rs.standard_normal(nc), rs.standard_normal(nc)]
    gd = sum(float((g[k] * dk).sum()) for k, dk in zip(("theta", "qv", "uzonal", "umerid"), d))
    h = 1e-4
    jp = sl.value_and_grad(*[a + h * b for a, b in zip(x, d)])[0]
    jm = sl.value_and_grad(*[a - h * b for a, b in zip(x, d)])[0]
    assert abs((jp - jm) / (2 * h) - gd) < 1e-6 * abs(gd)
