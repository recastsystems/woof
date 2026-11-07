"""Persistent TC device state; step methods only enqueue kernels."""
import numpy as np


class SurfaceBundle:
    def __init__(self, cfg, fields):
        import cupy as cp
        from woof.core.kernels import get_kernel
        from .curves import make_table
        self.cfg = cfg
        from woof.hydro.cost import HydroCostMeter
        self.cost = HydroCostMeter()
        self.shape = fields["ust"].shape
        self.n = fields["ust"].size
        self.zero = cp.zeros(self.shape, dtype=cp.float32)
        self.lake_mask_override = None
        # MYNN's QSFC is specific humidity, MM5's is mixing ratio. Keep
        # the stock carrier convention and convert inside the shared kernel.
        self.mynn = cfg.sf_sfclay_physics == 5
        self.table = None
        if cfg.tc_flux_curve != "off":
            self.table = make_table(cfg.tc_flux_curve, xp=cp)
            self.exchange_kernel = get_kernel("tc_exchange", "tc_exchange")
        self.diag = {k: cp.zeros(self.shape, cp.float32) for k in
                     ("cdn", "chn", "cqn", "zt", "zq")}
        self.waves = None
        if cfg.tc_wave_roughness or cfg.tc_spray == "barr2023":
            from .waves import WaveState
            self.waves = WaveState(cfg, self.shape)
        self.spray_diag = {k: cp.zeros(self.shape, cp.float32) for k in
                           ("sensible_increment", "latent_increment", "flags")}
        if cfg.tc_spray != "off":
            from .spray import PARAMETERS
            self.spray_params = cp.asarray(list(PARAMETERS[cfg.tc_spray].values()), cp.float32)
            self.spray_kernel = get_kernel("tc_spray", "tc_bao" if cfg.tc_spray == "bao2000" else "tc_barr")

    def exchange(self, atmosphere, fields, result):
        self.cost.before_lsm_start()
        if self.waves is not None:
            self.waves.update(fields)
        if self.table is None:
            self.cost.before_lsm_end()
            return
        f, a, r = fields, atmosphere, result
        args = [a[k][0] for k in ("u", "v", "temperature", "qv", "pressure", "dz")]
        args += [f[k] for k in ("psfc", "tsk", "xland")]
        args += [self.lake_mask_override if self.lake_mask_override is not None else f["lakemask"]]
        args += [f.get("xice", self.zero), self.table,
                 self.waves.roughness if self.waves is not None else self.zero,
                 np.int32(self.cfg.tc_wave_roughness)]
        args += [getattr(r, k) for k in (
            "znt", "ust", "ustm", "mol", "hfx", "qfx", "qsfc", "zol",
            "psim", "psih", "fm", "fh", "lh", "u10", "v10", "th2", "t2", "q2",
            "chs", "chs2", "cqs2", "flhc", "flqc", "rmol", "cd", "cda", "ck", "cka")]
        args += list(self.diag.values())
        args += [f[k] if self.mynn else self.zero
                 for k in ("ch", "qstar", "cpm", "gz1oz0")]
        args += [np.int32(self.mynn), np.int32(self.n)]
        self.exchange_kernel(((self.n+127)//128,), (128,), tuple(args))
        self.cost.before_lsm_end()

    def spray(self, atmosphere, fields):
        self.cost.advance_start()
        if self.cfg.tc_spray == "off":
            self.cost.advance_end()
            return
        f, a = fields, atmosphere
        args = [a[k][0] for k in ("temperature", "qv", "pressure")]
        args += [f[k] for k in ("tsk", "u10", "v10", "xland")]
        args += [self.lake_mask_override if self.lake_mask_override is not None else f["lakemask"]]
        args += [f.get("xice", self.zero), self.spray_params]
        if self.cfg.tc_spray == "bao2000":
            args += [np.float32(self.cfg.tc_spray_partition)]
        else:
            args += [a["dz"][0], f["ust"], f["znt"], f["rmol"],
                     f["flhc"], f["flqc"], f["qsfc"], self.waves.hs,
                     self.waves.phase, self.waves.dissipation, self.waves.slope]
        args += [f[k] for k in ("hfx", "qfx", "lh")]
        args += list(self.spray_diag.values()) + [np.int32(self.n)]
        self.spray_kernel(((self.n+127)//128,), (128,), tuple(args))
        self.cost.advance_end()


def create_surface(cfg, fields):
    if cfg.tc_flux_curve == "off" and cfg.tc_spray == "off" and not cfg.tc_wave_roughness:
        return None
    return SurfaceBundle(cfg, fields)
