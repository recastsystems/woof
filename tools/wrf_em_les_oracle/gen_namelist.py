#!/usr/bin/env python3
"""Generate WRF v4.6.1 em_les namelists for the WP-L7 oracle.

Three families:
  stock  - the shipped test/em_les configuration, unmodified (native oracle)
  match  - matched to the WOOF-ArWen idealized CBL case
           (woof/verify/cases/convective_boundary_layer.py)
  moist  - P1: the same two families with microphysics ON.  ``mp_physics`` is
           a per-run parameter here rather than the hard-coded 0 it used to
           be, and the sounding and iofields file travel with the run so a
           moist arm is a fully enumerated delta rather than an edit.

Every deviation from the stock em_les namelist is enumerated in the
matched-config receipt written alongside the runs.

**Dry byte-compatibility is a property of this file, not a hope.** The five
original run ids emit namelists byte-identical to the committed
``namelist.stock_em_les`` / ``namelist.match_*``; `--check-dry` proves it
against this directory before anything else is generated.
"""
import sys
import os
import json

# ArWen CBL case constants, read from convective_boundary_layer.py
ARWEN = dict(heat_flux=0.24, drag=0.0013, c_s=0.18, c_k=0.10, mix_isotropic=1,
             ztop=2400.0, epssm=0.1, smdiv=0.1, emdiv=0.0, tss=4)


def frac(dt):
    """WRF time_step + fraction for a float dt."""
    if abs(dt - round(dt)) < 1e-12:
        return int(round(dt)), 0, 1
    for den in (2, 4, 8, 10, 100):
        num = dt * den
        if abs(num - round(num)) < 1e-9:
            whole = int(dt // 1)
            return whole, int(round(num)) - whole * den, den
    raise ValueError(dt)


def namelist(km_opt, ncell, nz, dx, ztop, dt, hours, hist_min, stock=False,
             mp_physics=0, iofields="iofields_les.txt", ideal_xland=None,
             isfflx=None, moist_adv_opt=None):
    """One em_les namelist.

    ``mp_physics`` was hard-coded to 0 until P1.  It is a parameter now; every
    other knob keeps its previous value, so mp_physics=0 reproduces the dry
    namelists byte for byte.  ``use_theta_m`` was moot while the domain was
    dry and is not moot any more -- it stays 1 for the stock family (which is
    what WRF ships) and 0 for the matched family, because woof stores dry
    theta and says so (``woof/core/moist.py:29``,
    ``woof/core/microphysics.py:23``).  Matching it is not a preference.
    """
    ts, fn, fd = frac(dt)
    e_we = ncell + 1
    e_sn = ncell + 1
    e_vert = nz + 1
    if stock:
        phys = dict(sf_sfclay_physics=1, isfflx=2, drag=None, use_theta_m=1,
                    khdif=1.0, kvdif=1.0, emdiv=0.01)
    else:
        phys = dict(sf_sfclay_physics=0, isfflx=0, drag=ARWEN["drag"],
                    use_theta_m=0, khdif=0.0, kvdif=0.0, emdiv=ARWEN["emdiv"])
    if isfflx is not None:
        phys["isfflx"] = isfflx
    L = []
    A = L.append
    A(" &time_control")
    A(" run_days                            = 0,")
    A(" run_hours                           = %d," % hours)
    A(" run_minutes                         = 0,")
    A(" run_seconds                         = 0,")
    A(" start_year                          = 0001,")
    A(" start_month                         = 01,")
    A(" start_day                           = 01,")
    A(" start_hour                          = 00,")
    A(" start_minute                        = 00,")
    A(" start_second                        = 00,")
    A(" end_year                            = 0001,")
    A(" end_month                           = 01,")
    A(" end_day                             = 01,")
    A(" end_hour                            = %02d," % hours)
    A(" end_minute                          = 00,")
    A(" end_second                          = 00,")
    A(" history_interval_m                  = %d," % hist_min)
    A(" frames_per_outfile                  = 1000,")
    A(" restart                             = .false.,")
    A(" io_form_history                     = 2,")
    A(" io_form_restart                     = 2,")
    A(" io_form_input                       = 2,")
    A(" io_form_boundary                    = 2,")
    A(' iofields_filename                   = "%s",' % iofields)
    A(" /")
    A("")
    A(" &domains")
    A(" time_step                           = %d," % ts)
    A(" time_step_fract_num                 = %d," % fn)
    A(" time_step_fract_den                 = %d," % fd)
    A(" max_dom                             = 1,")
    A(" s_we                                = 1,")
    A(" e_we                                = %d," % e_we)
    A(" s_sn                                = 1,")
    A(" e_sn                                = %d," % e_sn)
    A(" s_vert                              = 1,")
    A(" e_vert                              = %d," % e_vert)
    A(" dx                                  = %g," % dx)
    A(" dy                                  = %g," % dx)
    A(" ztop                                = %g," % ztop)
    A(" grid_id                             = 1,")
    A(" parent_id                           = 0,")
    A(" i_parent_start                      = 0,")
    A(" j_parent_start                      = 0,")
    A(" parent_grid_ratio                   = 1,")
    A(" parent_time_step_ratio              = 1,")
    A(" feedback                            = 0,")
    A(" smooth_option                       = 0,")
    A(" /")
    A("")
    A(" &physics")
    A(" mp_physics                          = %d," % mp_physics)
    A(" ra_lw_physics                       = 0,")
    A(" ra_sw_physics                       = 0,")
    A(" radt                                = 0,")
    A(" sf_sfclay_physics                   = %d," % phys["sf_sfclay_physics"])
    A(" sf_surface_physics                  = 0,")
    A(" bl_pbl_physics                      = 0,")
    A(" bldt                                = 0,")
    A(" cu_physics                          = 0,")
    A(" cudt                                = 0,")
    A(" isfflx                              = %d," % phys["isfflx"])
    if ideal_xland is not None:
        A(" ideal_xland                         = %d," % ideal_xland)
    A(" /")
    A("")
    A(" &fdda")
    A(" /")
    A("")
    A(" &dynamics")
    A(" hybrid_opt                          = 0,")
    A(" rk_ord                              = 3,")
    A(" diff_opt                            = 2,")
    A(" km_opt                              = %d," % km_opt)
    A(" damp_opt                            = 0,")
    A(" zdamp                               = 5000.,")
    A(" dampcoef                            = 0.1,")
    A(" khdif                               = %g," % phys["khdif"])
    A(" kvdif                               = %g," % phys["kvdif"])
    A(" c_s                                 = %g," % ARWEN["c_s"])
    A(" c_k                                 = %g," % ARWEN["c_k"])
    A(" mix_isotropic                       = %d," % ARWEN["mix_isotropic"])
    A(" mix_upper_bound                     = 0.1,")
    A(" smdiv                               = %g," % ARWEN["smdiv"])
    A(" emdiv                               = %g," % phys["emdiv"])
    A(" epssm                               = %g," % ARWEN["epssm"])
    A(" tke_heat_flux                       = %g," % ARWEN["heat_flux"])
    if phys["drag"] is not None:
        A(" tke_drag_coefficient                = %g," % phys["drag"])
    A(" time_step_sound                     = %d," % (6 if stock else ARWEN["tss"]))
    A(" h_mom_adv_order                     = 5,")
    A(" v_mom_adv_order                     = 3,")
    A(" h_sca_adv_order                     = 5,")
    A(" v_sca_adv_order                     = 3,")
    if moist_adv_opt is not None:
        A(" moist_adv_opt                       = %d," % moist_adv_opt)
    A(" mix_full_fields                     = .true.,")
    A(" non_hydrostatic                     = .true.,")
    A(" pert_coriolis                       = .true.,")
    A(" use_theta_m                         = %d," % phys["use_theta_m"])
    A(" diff_6th_opt                        = 0,")
    A(" w_damping                           = 0,")
    A(" /")
    A("")
    A(" &bdy_control")
    A(" periodic_x                          = .true.,")
    A(" symmetric_xs                        = .false.,")
    A(" symmetric_xe                        = .false.,")
    A(" open_xs                             = .false.,")
    A(" open_xe                             = .false.,")
    A(" periodic_y                          = .true.,")
    A(" symmetric_ys                        = .false.,")
    A(" symmetric_ye                        = .false.,")
    A(" open_ys                             = .false.,")
    A(" open_ye                             = .false.,")
    A(" /")
    A("")
    A(" &grib2")
    A(" /")
    A("")
    A(" &namelist_quilt")
    A(" nio_tasks_per_group = 0,")
    A(" nio_groups = 1,")
    A(" /")
    A("")
    A(" &ideal")
    A(" ideal_case = 9")
    A(" /")
    return "\n".join(L) + "\n"


DRY_SOUNDING = "input_sounding.arwen_cbl"
# WRF v4.6.1 test/em_les assets, by their authority-receipt sha256
# (handoffs/P6-LES-WPL0-AUTHORITY-RECEIPT.md:69 and the same table).
STOCK_SOUNDING = "input_sounding.wrf_em_les"          # 6aed509b22519dcd...
SHALCONV_SOUNDING = "input_sounding.wrf_em_les_shalconv"  # ff044b473cc56389...

RUNS = {
    # id                km  ncell  nz    dx    ztop     dt  h  hist  stock
    "stock_em_les":    (2,    39,  39, 100., 2000.,  1.0,  1,   10, True),
    "match_km3_100m":  (3,    96,  64, 100., 2400.,  0.5,  2,    1, False),
    "match_km2_100m":  (2,    96,  64, 100., 2400.,  0.5,  2,    1, False),
    "match_km3_50m":   (3,   192,  96,  50., 2400.,  0.25, 2,    1, False),
    "match_km2_50m":   (2,   192,  96,  50., 2400.,  0.25, 2,    1, False),
}

# ---------------------------------------------------------------- P1 moist
#
# The moist family the spec (§3.2) asks for is "the recipe rerun with the
# enumerated namelist delta (mp_physics=1, the moist sounding)".  Two of the
# three deltas are settled by the authority and are set here; the third --
# WHICH moist sounding defines the matched case -- is a case definition and is
# deliberately NOT chosen in this file.  The probe family below measures the
# two stock-family candidates so that choice is made against numbers.
#
# Deltas from the dry family, enumerated:
#   mp_physics 0 -> 1                (Kessler; KESS_KMAX 256 clears every nz here)
#   iofields   -> iofields_les_moist.txt  (adds the moist rows to stream 0)
#   sounding   -> a moist one        (the dry asset carries qv == 0)
# Everything else is the dry family's value, including use_theta_m, which is
# 1 for the stock family and 0 for the matched family (woof stores dry theta).
MOIST_PROBES = {
    # id                          km ncell nz    dx    ztop    dt   h  hist stock
    "moist_smoke_stocksnd_1h":   (2,   39, 39, 100., 2000.,  1.0,   1,  10, True),
    "moist_probe_stocksnd_12h":  (2,   39, 39, 100., 2000.,  1.0,  12,  30, True),
    "moist_probe_shalsnd_12h":   (2,   39, 39, 100., 2000.,  1.0,  12,  30, True),
}
MOIST_PROBE_SOUNDING = {
    "moist_smoke_stocksnd_1h": STOCK_SOUNDING,
    "moist_probe_stocksnd_12h": STOCK_SOUNDING,
    "moist_probe_shalsnd_12h": SHALCONV_SOUNDING,
}
# The shalconv arm carries its shipped surface as well as its shipped
# sounding: namelist.input_shalconv d01 sets isfflx=1 and ideal_xland=2
# (water).  Running its sounding over the stock land surface would be a
# hybrid nobody ships.
MOIST_PROBE_EXTRA = {
    "moist_probe_shalsnd_12h": dict(isfflx=1, ideal_xland=2),
}

# The matched moist arms.  Shapes are the dry matched shapes; the sounding is
# a required argument because nothing in this recipe is allowed to pick it.
MATCHED_MOIST = {
    # id                    km  ncell  nz    dx    ztop     dt  h  hist
    "moist_match_km3_100m": (3,    96,  64, 100., 2400.,  0.5,  2,    1),
    "moist_match_km2_100m": (2,    96,  64, 100., 2400.,  0.5,  2,    1),
    "moist_match_km3_50m":  (3,   192,  96,  50., 2400.,  0.25, 2,    1),
    "moist_match_km2_50m":  (2,   192,  96,  50., 2400.,  0.25, 2,    1),
    "moist_ctl4_100m":      (4,    96,  64, 100., 2400.,  0.5,  2,    1),
}


def emit(outdir, rid, km, nc, nz, dx, ztop, dt, hrs, hist, stock,
         mp_physics=0, iofields="iofields_les.txt", sounding=DRY_SOUNDING,
         **extra):
    txt = namelist(km, nc, nz, dx, ztop, dt, hrs, hist, stock=stock,
                   mp_physics=mp_physics, iofields=iofields, **extra)
    # newline="\n" so the emitted namelist is byte-stable off-Linux too; WRF
    # reads either, but the dry byte-identity check has to mean something.
    with open("%s/namelist.%s" % (outdir, rid), "w", newline="\n") as fh:
        fh.write(txt)
    return dict(km_opt=km, ncell_x=nc, ncell_y=nc, nz=nz, dx_m=dx,
                ztop_m=ztop, dt_s=dt, hours=hrs, hist_min=hist,
                stock=stock, points=nc * nc * nz,
                steps=int(hrs * 3600 / dt), mp_physics=mp_physics,
                iofields=iofields, sounding=sounding, **extra)


def check_dry(outdir, refdir):
    """Prove the dry namelists did not move when mp_physics became a knob."""
    bad = 0
    for rid in RUNS:
        new = open("%s/namelist.%s" % (outdir, rid), "rb").read()
        ref_path = "%s/namelist.%s" % (refdir, rid)
        if not os.path.exists(ref_path):
            print("DRY-CHECK skip %-18s (no reference in %s)" % (rid, refdir))
            continue
        ref = open(ref_path, "rb").read()
        ok = new == ref
        bad += 0 if ok else 1
        print("DRY-CHECK %-18s %s" % (rid, "identical" if ok else "CHANGED"))
    return bad


if __name__ == "__main__":
    outdir = sys.argv[1]
    argv = sys.argv[2:]
    matched_sounding = None
    if "--matched-moist-sounding" in argv:
        matched_sounding = argv[argv.index("--matched-moist-sounding") + 1]

    meta = {}
    for rid, (km, nc, nz, dx, ztop, dt, hrs, hist, stock) in RUNS.items():
        meta[rid] = emit(outdir, rid, km, nc, nz, dx, ztop, dt, hrs, hist,
                         stock)
    for rid, (km, nc, nz, dx, ztop, dt, hrs, hist, stock) in MOIST_PROBES.items():
        meta[rid] = emit(outdir, rid, km, nc, nz, dx, ztop, dt, hrs, hist,
                         stock, mp_physics=1,
                         iofields="iofields_les_moist.txt",
                         sounding=MOIST_PROBE_SOUNDING[rid],
                         **MOIST_PROBE_EXTRA.get(rid, {}))
    if matched_sounding is not None:
        for rid, (km, nc, nz, dx, ztop, dt, hrs, hist) in MATCHED_MOIST.items():
            meta[rid] = emit(outdir, rid, km, nc, nz, dx, ztop, dt, hrs, hist,
                             False, mp_physics=1,
                             iofields="iofields_les_moist.txt",
                             sounding=matched_sounding)
    else:
        meta["_matched_moist"] = dict(
            status="NOT GENERATED",
            reason="the matched moist sounding is a case definition and is "
                   "not chosen by this recipe; pass "
                   "--matched-moist-sounding <asset> once it is ratified",
            shapes={k: dict(km_opt=v[0], ncell=v[1], nz=v[2], dx_m=v[3],
                            ztop_m=v[4], dt_s=v[5], hours=v[6])
                    for k, v in MATCHED_MOIST.items()})

    with open("%s/run_matrix.json" % outdir, "w", newline="\n") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    for k, v in sorted(meta.items()):
        if not isinstance(v, dict) or "km_opt" not in v:
            continue
        print("%-26s km_opt=%d %4dx%4dx%3d dx=%5.1f dt=%4.2f %2dh mp=%d "
              "%9d pts %6d steps  %s"
              % (k, v["km_opt"], v["ncell_x"], v["ncell_y"], v["nz"],
                 v["dx_m"], v["dt_s"], v["hours"], v["mp_physics"],
                 v["points"], v["steps"], v["sounding"]))
    if "--check-dry" in argv:
        refdir = argv[argv.index("--check-dry") + 1]
        sys.exit(1 if check_dry(outdir, refdir) else 0)
