"""Kernel time: WRF 4.8.0 MYNN-SFC (new) vs v4.6.1 MYNN surface layer (today), same card,
same real captured state tiled to a CONUS-like surface (ny x nx), default numerics."""
import sys, json, numpy as np, cupy as cp
from types import SimpleNamespace
sys.path.insert(0, sys.argv[2])
import replay as R
from woof.core.physics import PhysicsDriver
from woof.core.mynn_sfclay import launch_mynn_surface_layer, MynnSurfaceResult
from woof.core.physics_inventory import MYNN_SURFACE_OUTPUTS
cap = dict(np.load(sys.argv[1]))
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 25
ty, tx = 26, 45            # 40x40 -> 1040 x 1800 (1.87 M columns, ~CONUS 3 km)
def tile(a):
    a = np.asarray(a)
    if a.ndim == 2: return np.tile(a, (ty, tx))
    if a.ndim == 3: return np.tile(a, (1, ty, tx))
    return a
big = {k: tile(v) for k, v in cap.items()}
big["ny"], big["nx"] = big["xland"].shape
def ev():
    s, e = cp.cuda.Event(), cp.cuda.Event(); return s, e
out = {}
for mode in (0, 1):
    big["mode"] = mode
    # new path, via the forecast method on a stand-in owner (as replay.run_gpu)
    f = {k: cp.asarray(big[k], dtype=cp.float32) for k in R.N2 if k not in ("psfc", "sst", "dx")}
    f["tsk_sea"] = cp.asarray(big["sst"], dtype=cp.float32)
    from woof.core.mynnsfc_b66187f import SEAICE_OWN
    for n in SEAICE_OWN: f.setdefault(n + "_sea", cp.zeros_like(f["xland"]))
    atm = {k: cp.ascontiguousarray(cp.asarray(big[k], dtype=cp.float32)) for k in R.N3 if k != "pattern_spp_pbl"}
    atm["p_interface"] = cp.asarray(big["psfc"], dtype=cp.float32)[None]
    cfg = SimpleNamespace(dx=3000.0, fractional_seaice=mode, sf_surface_physics=3, isfflx=1, spp_pbl=0,
                          sf_mynn_sfcflux_land=0, sf_mynn_sfcflux_water=1, shalwater_z0=0)
    owner = SimpleNamespace(fields=f, spp_patterns={})
    ice = cp.zeros(f["xland"].shape, dtype=cp.bool_)
    call = lambda: PhysicsDriver._run_mynnsfc_b66187f(owner, atm, cfg, itimestep=50,
        ice_params=SimpleNamespace(xice_threshold=0.02 if mode else 0.5), ice_component=ice, carriers=True)
    call(); cp.cuda.Device().synchronize()
    s, e = ev(); s.record()
    for _ in range(reps): call()
    e.record(); e.synchronize()
    out[f"new_mode{mode}_ms"] = cp.cuda.get_elapsed_time(s, e) / reps
# today's kernel: one v4.6.1 launch per call (twice under RUC's wrapper, as physics.py does today)
inputs = {"u1": atm["u"][0], "v1": atm["v"][0], "t1": atm["temperature"][0], "qv1": atm["qv"][0],
          "p1": atm["pressure"][0], "rho1": atm["rho"][0], "dz1": atm["dz"][0], "u2": atm["u"][1],
          "v2": atm["v"][1], "dz2": atm["dz"][1], "psfc": atm["p_interface"][0]}
for k in ("tsk", "pblh", "mavail", "hfx", "qfx", "znt", "qsfc", "ust", "xland", "snowh"):
    inputs[k] = cp.ascontiguousarray(cp.asarray(big[k], dtype=cp.float32))
res = MynnSurfaceResult(**{n: cp.zeros(f["xland"].shape, cp.float32) for n in MYNN_SURFACE_OUTPUTS})
mol = cp.asarray(big["mol"], dtype=cp.float32); ustm = cp.asarray(big["ustm"], dtype=cp.float32)
one = lambda: launch_mynn_surface_layer(inputs, mol, ustm, res, dx=3000.0, itimestep=50)
one(); cp.cuda.Device().synchronize()
s, e = ev(); s.record()
for _ in range(reps): one()
e.record(); e.synchronize()
out["today_single_launch_ms"] = cp.cuda.get_elapsed_time(s, e) / reps
out["columns"] = int(np.prod(f["xland"].shape))
out["device"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
print(json.dumps(out))
