"""Capture hook for the MYNN-SFC module replay: dumps the actual arguments of WRF
4.8.0's MYNN-SFC call site just before WOOF-ArWen's call, at chosen steps."""
import os
if os.environ.get("L04_CAPTURE_DIR"):
    import numpy as np
    import woof.core.physics as P
    _orig = P.PhysicsDriver._run_mynnsfc_b66187f
    _steps = {int(s) for s in os.environ.get("L04_CAPTURE_STEPS", "1,2,50,100,240").split(",")}
    _out = os.environ["L04_CAPTURE_DIR"]

    def _hooked(self, atmosphere, cfg, *, itimestep, ice_params, ice_component, carriers):
        if itimestep in _steps:
            import cupy as cp
            f = self.fields
            g = lambda a: cp.asnumpy(a).astype(np.float32)
            ny, nx = f["xland"].shape
            thr = (float(ice_params.xice_threshold) if ice_params is not None
                   else (0.02 if cfg.fractional_seaice else 0.5))
            cap = dict(nx=nx, ny=ny, mode=int(cfg.fractional_seaice), itimestep=itimestep,
                       isfflx=int(cfg.isfflx), spp_pbl=int(cfg.spp_pbl),
                       land=int(cfg.sf_mynn_sfcflux_land), water=int(cfg.sf_mynn_sfcflux_water),
                       shal=int(cfg.shalwater_z0), flag_lsm=int(cfg.sf_surface_physics),
                       restart=0, tice2tsk_if2cold=0, xice_threshold=np.float32(thr))
            for k, src in (("u", "u"), ("v", "v"), ("temperature", "temperature"), ("qv", "qv"),
                           ("pressure", "pressure"), ("dz", "dz"), ("theta", "theta"), ("rho", "rho")):
                cap[k] = g(atmosphere[src][0:2])
            cap["pattern_spp_pbl"] = np.zeros((2, ny, nx), np.float32)
            cap["psfc"] = g(atmosphere["p_interface"][0])
            cap["dx"] = np.full((ny, nx), np.float32(cfg.dx), np.float32)
            cap["water_depth"] = np.zeros((ny, nx), np.float32)
            cap["sst"] = g(f["tsk_sea"]) if "tsk_sea" in f else g(f["tsk"])
            runner = getattr(self, "_mynnsfc_b66187f", None)
            cap["cqs"] = g(runner.cqs) if runner is not None else np.zeros((ny, nx), np.float32)
            for k in ("mavail", "xland", "tsk", "snowh", "xice", "lakemask", "chs", "chs2", "cqs2",
                      "cpm", "znt", "ust", "pblh", "zol", "mol", "psim", "psih", "qgh", "hfx", "qfx",
                      "lh", "flhc", "flqc", "qsfc", "rmol", "gz1oz0", "wspd", "br", "ch", "ustm", "ck",
                      "cka", "cd", "cda", "u10", "v10", "th2", "t2", "q2"):
                cap[k] = g(f[k])
            for k in ("qgh", "chs2", "chs", "cpm", "cqs2", "flhc", "flqc", "hfx", "lh", "qfx", "qsfc", "znt"):
                cap[k + "_sea"] = g(f[k + "_sea"]) if k + "_sea" in f else np.zeros((ny, nx), np.float32)
            cap["tsk_sea"] = cap["sst"].copy()
            os.makedirs(_out, exist_ok=True)
            np.savez(os.path.join(_out, f"a195-mynnsfc-step{itimestep:04d}.npz"), **cap)
        return _orig(self, atmosphere, cfg, itimestep=itimestep, ice_params=ice_params,
                     ice_component=ice_component, carriers=carriers)

    P.PhysicsDriver._run_mynnsfc_b66187f = _hooked

if os.environ.get("L04_VARIANT"):
    import dataclasses
    import woof.core.physics as P2
    _orig_sfclay = P2.PhysicsDriver._run_sfclay
    _swapped = {}

    def _sfclay(self, atmosphere, cfg):
        want = os.environ["L04_VARIANT"]
        if int(cfg.sf_sfclay_physics) == 5 and cfg.mynn_sfclay_variant != want:
            key = id(cfg)
            if key not in _swapped:
                _swapped[key] = (cfg, dataclasses.replace(cfg, mynn_sfclay_variant=want))
            cfg = _swapped[key][1]
        return _orig_sfclay(self, atmosphere, cfg)

    P2.PhysicsDriver._run_sfclay = _sfclay
