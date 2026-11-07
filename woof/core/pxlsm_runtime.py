"""Persistent PX land state and WRF surface-driver bindings.

The native soil column must already be on init_soil_depth_7's two layers.
No Noah column is truncated or relabelled here. All step arithmetic is CUDA.
"""
from __future__ import annotations

import numpy as np

from woof.core.pxlsm import (PXLSM_CARRIED, PXLSM_INPUTS, PXLSM_PUBLISHED,
                             PXLSM_TABLES_SHA256, px_dataset, prepare_pxlsm,
                             launch_pxlsm)

PX_STATE_2D = tuple(dict.fromkeys((*PXLSM_INPUTS, *PXLSM_CARRIED,
                                  *PXLSM_PUBLISHED, "sfcevp", "chklowq")))
PX_FRACTIONS = ("landusef", "soilctop", "soilcbot")
PX_ICE_FLUXES = ("flhc", "flqc", "cpm", "cqs2", "chs2", "chs",
                 "qsfc", "qgh", "hfx", "qfx", "lh")
PX_ICE_FIELDS = ("tsk_save", "tsk_sea", "znt_sea", "ust_sea", "mol_sea", "zol_sea",
                 *(name+"_sea" for name in PX_ICE_FLUXES))


class PxParameters:
    def __init__(self, nlcat, *, fractional_seaice=0):
        self.nlcat = int(nlcat)
        self.dataset = px_dataset(nlcat)
        self.fractional_seaice = int(fractional_seaice)
        self.xice_threshold = .02 if fractional_seaice else .5

    @property
    def restart_identity(self):
        return {"land_categories": self.nlcat, "soil_categories": 16,
                "table_sha256": PXLSM_TABLES_SHA256,
                "fractional_seaice": self.fractional_seaice,
                "thickness_m": [0.01, 0.99], "centres_m": [0.005, 0.505]}


def initialize(fields, cfg, *, landusef, soilctop, soilcbot, imperv=0., canfra=0.):
    import cupy as cp
    shape = fields["tsk"].shape
    if landusef is None or soilcbot is None:
        raise ValueError("PX land requires LANDUSEF and SOILCBOT category fractions: "
                         "WRF VEGELAND and SOILPROP weight these fractions; "
                         "a dominant-category replacement changes the physics")
    for name, value in (("landusef", landusef), ("soilctop", soilctop),
                        ("soilcbot", soilcbot)):
        if value is None:
            # SOILCTOP is in PX's signature but no statement reads it.
            value = soilcbot
        a = cp.ascontiguousarray(cp.asarray(value, dtype=cp.float32))
        if a.ndim != 3 or a.shape[1:] != shape:
            raise ValueError(f"PX {name.upper()} needs (category,ny,nx), got {a.shape}")
        if name != "landusef" and a.shape[0] != 16:
            raise ValueError("PX SOILPROP requires the native 16 STATSGO categories")
        fields[name] = a.copy()
    params = PxParameters(fields["landusef"].shape[0], fractional_seaice=cfg.fractional_seaice)
    for name in PX_STATE_2D:
        if name not in fields:
            fields[name] = cp.zeros(shape, dtype=cp.float32)
    for name, value in (("imperv", imperv), ("canfra", canfra)):
        fields[name][...] = cp.asarray(value, dtype=cp.float32)
    # PX's shared LSMINIT runs after the input carriers have been restored.
    # The first PXLSM call then derives its vegetation and soil parameters.
    prepare_pxlsm(params.nlcat)
    from woof.core.kernels import get_kernel
    get_kernel("pxlsm_driver", "pxlsm_post")
    return params


def cold_start(fields, params):
    """WRF's shared LSMINIT with PX's native two-layer column."""
    import cupy as cp
    from woof.core.noah import _device_tables
    from woof.core.kernels import get_kernel
    _, soil_table, _, _ = _device_tables(params, np.array([.01, .99], np.float32))
    n = fields["isltyp"].size
    get_kernel("pxlsm_init", "pxlsm_initialize_sh2o")(
        ((n+127)//128,), (128,),
        (fields["isltyp"], fields["smois"], fields["tslb"], fields["sh2o"],
         soil_table, np.int32(params.slcats), np.int32(n)))
    fields["canwat"].fill(0.)


def step(driver, atmosphere, cfg, itimestep):
    import cupy as cp
    f = driver.fields
    f["psfc"][...] = atmosphere["p_interface"][0]
    f["rainbl"] += driver._pending_rainbl
    a = {"dz8w1": atmosphere["dz"][0], "dz8w2": atmosphere["dz"][1],
         "qv3d1": atmosphere["qv"][0], "qv3d2": atmosphere["qv"][1],
         "t3d1": atmosphere["temperature"][0],
         "th3d1": atmosphere["theta"][0], "rho1": atmosphere["rho"][0]}
    from woof.core.kernels import get_kernel
    n = f["tsk"].size
    ice = bool(driver.pxlsm_params.fractional_seaice)
    surface_called = np.int32(int(cfg.sf_sfclay_physics) != 0)
    if ice:
        get_kernel("pxlsm_driver", "pxlsm_ice_prepare")(
            ((n+127)//128,), (128,),
            (np.int32(n), surface_called, f["xice"], f["sst"], f["tsk_sea"], f["tsk"]))
    launch_pxlsm(f, a, landusef=f["landusef"], soilcbot=f["soilcbot"],
                 dt=float(cfg.dt), dtbl=driver.bldt_seconds,
                 itimestep=int(itimestep), curr_secs=float(driver.state.elapsed_seconds),
                 smois_init=int(cfg.pxlsm_smois_init),
                 soil_nudge=int(cfg.pxlsm_soil_nudge),
                 modis_veg=int(cfg.pxlsm_modis_veg),
                 anal_interval=int(cfg.sgfdda_interval_m)*60,
                 domain_shape=f["tsk"].shape,
                 device_clock=(getattr(driver.state, "_px_inputs_owner", None).current
                     if getattr(driver.state, "_px_inputs_owner", None) is not None else None))
    if ice:
        args = [np.int32(n), surface_called, f["xice"], f["tsk_sea"], f["tsk"], f["tsk_save"]]
        args.extend(f[name] for name in PX_ICE_FLUXES)
        args.extend(f[name+"_sea"] for name in PX_ICE_FLUXES)
        get_kernel("pxlsm_driver", "pxlsm_ice_blend")(((n+127)//128,), (128,), tuple(args))
    get_kernel("pxlsm_driver", "pxlsm_post")(
        ((n+127)//128,), (128,),
        (np.int32(n), np.float32(driver.bldt_seconds), f["psfc"],
         f["t2"], f["qfx"], f["th2"], f["chklowq"], f["sfcevp"],
         f["rainbl"], driver._pending_rainbl))


def wrfinput_inputs(raw, cfg):
    if int(cfg.sf_surface_physics) != 7:
        return {}
    missing = [name for name in ("LANDUSEF", "SOILCBOT") if name not in raw]
    if missing:
        raise ValueError(f"PX native input lacks {missing}; real.exe must retain "
                         "the category fractions consumed by PXLSM")
    return {"landusef": raw["LANDUSEF"], "soilctop": raw.get("SOILCTOP"),
            "soilcbot": raw["SOILCBOT"], "imperv": raw.get("IMPERV", 0.),
            "canfra": raw.get("CANFRA", 0.)}


def feedback_surface(parent, child, reg, *, spec_zone, skip_names=()):
    """WRF Registry surface u flags, including bounded streamed stores.

    T2/Q2 use copy_fcn's area average (Registry:380-381). The remaining
    fields use copy_fcnm's one-point restriction (Registry:790,839-842,
    868-872,1417,1946,1980). PX resistances and soil parameters have no u
    flag and remain local to the parent.
    """
    import cupy as cp
    from woof.core.nest_operands import (NestWindowSource, rectangle_windows,
                                         streamed_chunk_shape)
    from woof.core.nest_interp import (copy_fcn, feedback_parent_bounds,
                                        feedback_child_window)
    donor, target = NestWindowSource(child.state), NestWindowSource(parent.state)
    ilo, ihi, jlo, jhi = feedback_parent_bounds(reg, spec_zone=spec_zone)
    chunk = next((streamed_chunk_shape(node.state) for node in (parent, child)
                  if getattr(node.state, "_streamed_domain", None) is not None),
                 (64, 64))
    names = ("t2", "q2", "tslb", "smois", "sh2o", "xice", "snow",
             "snowh", "canwat", "tsk", "xland", "snowc", "imperv", "canfra",
             "vegfra", "shdmin", "shdmax", "isltyp", "ivgtyp", *PX_FRACTIONS)
    for name in names:
        key = f"fields/{name}"
        if donor.array(key) is None or target.array(key) is None:
            raise ValueError(f"PX surface feedback requires child and parent {name}")
        if name in skip_names:
            continue
        for window in rectangle_windows((slice(jlo, jhi+1), slice(ilo, ihi+1)), chunk):
            y, x = window
            if name in ("t2", "q2", *PX_FRACTIONS):
                source_window = feedback_child_window(reg, window, spec_zone=spec_zone)
                out = target.device(key, window)
                copy_fcn(out, donor.device(key, source_window), reg,
                         spec_zone=spec_zone, parent_window=window,
                         child_window=source_window)
            else:
                # copy_fcnm: odd-ratio centre, even-ratio southwest point.
                # The parent cell coordinate is zero-based; registration
                # starts are WRF one-based. These are the native donors.
                ri, rj = reg.nri, reg.nrj
                oi = 1 if rj % 2 else ri//2-1
                oj = 1 if rj % 2 else rj//2-1
                ci = (x.start+1-reg.i_parent_start)*ri+oi
                cj = (y.start+1-reg.j_parent_start)*rj+oj
                cy = slice(cj, cj+(y.stop-y.start)*rj, rj)
                cx = slice(ci, ci+(x.stop-x.start)*ri, ri)
                source = donor.array(key)[(..., cy, cx)]
                out = donor.device_array(source)
            target.write(key, window, out)
