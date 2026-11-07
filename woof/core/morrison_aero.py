"""Morrison two-moment with CESM-NCSU prescribed aerosol (WRF ``mp_physics=40``).

Launcher for ``woof/core/kernels/morrison_aero.cu``, a WRF-order
transcription of ``phys/module_mp_morr_two_moment_aero.F`` (WRF v4.6.1,
commit d66e442).  Three launches carry one WRF call: the per-level process
loop (one thread per cell), the split-step sedimentation (ten category
lanes per column) and the final loop with the wrapper's write-back (one
thread per cell).  Fields are contiguous FP32 ``(nz, ny, nx)``; surface
fields ``(ny, nx)``.

The kernel's constants are the words WRF's own ``MORR_TWO_MOMENT_INIT_AERO``
holds, read from ``woof/data/morrison/constants.toml`` (``[morrison_aero.*]``)
and uploaded once per device and ``morr_rimed_ice``.

``aercu_opt`` follows WRF: 0 runs fixed 250 cm-3 droplets; 1 also writes
the RRTMG effective radii EFCG/EFIG/EFSG and WACT; 2 predicts droplet
number from the prescribed aerosol (Abdul-Razzak-Ghan activation, Liu et
al. ice nucleation) and writes CCN1..7.  The column launcher runs all three;
the positive forecast route is an explicit opt-in using public NASA forcing,
written from papers, not bitwise to WRF. Its measured cost is published in
``docs/public-cumulus-aerosol.md``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tomllib

import numpy as np

from woof.core.device_cache import cuda_cache

CONSTANTS_PATH = (Path(__file__).resolve().parents[1] / "data" / "morrison"
                  / "constants.toml")

_SCALARS = (
    "NDCNST", "AI", "AC", "AS", "AR", "AG", "BI", "BC", "BS", "BR", "BG",
    "RHOSU", "RHOW", "RHOI", "RHOSN", "RHOG", "AIMM", "BIMM", "ECR", "DCS",
    "MI0", "MG0", "F1S", "F2S", "F1R", "F2R", "QSMALL", "CI", "DI", "CS",
    "DS", "CG", "DG", "EII", "ECI", "RIN", "CPW", "MW", "MA", "RR", "MMULT",
    "LAMMAXI", "LAMMINI", "LAMMAXR", "LAMMINR", "LAMMAXS", "LAMMINS",
    "LAMMAXG", "LAMMING")
_AEROSOL = ("ATEN", "ALOGATEN", "ALOG2", "ALOG3", "THIRD", "SQ2")
_PER_SPECIES = ("ALOGSIG", "EXP45LOGSIG", "SMCRIT", "F1AER", "F2AER",
                "HYGRO", "DENSITY", "NUM_TO_MASS")

#: The order of the constant array the kernel indexes with its MA_* macros
#: (kernels/morrison_aero.cu); tests/test_morrison_aero.py binds the two.
MA_CONSTANT_NAMES: tuple[str, ...] = (
    _SCALARS
    + tuple(f"CONS{n}" for n in range(1, 42))
    + _AEROSOL
    + tuple(f"{name}_{m}" for name in _PER_SPECIES for m in range(1, 11))
    + tuple(f"CCNFACT_{s}_{m}" for m in range(1, 11) for s in range(1, 8)))

#: Local sedimentation profiles are sized by a compile-time tier.
from woof.state_serialization_contract import MORRISON_AEROSOL_RESTART_FIELDS as AERO_DIAGNOSTIC_FIELDS
KMAX_TIERS = (64, 256)
VERTICAL_LEVEL_BOUNDS = (2, KMAX_TIERS[-1])

_CELL_TPB = 128
_SEDIMENT_TPB = 128           # four warps, three columns per warp
_COLUMNS_PER_WARP = 3
_SCRATCH_PLANES = 22          # S_NPLANES in the kernel


def _table_name(morr_rimed_ice: int) -> str:
    if morr_rimed_ice == 1:
        return "hail"
    if morr_rimed_ice == 0:
        return "graupel"
    raise ValueError("morr_rimed_ice must be 0 (graupel) or 1 (hail, WRF "
                     f"default), got {morr_rimed_ice!r}")


@lru_cache(maxsize=4)
def constant_words(morr_rimed_ice: int = 1, activation_temperature: str = "wrf") -> np.ndarray:
    """The kernel's constant array as uint32 words, in MA_CONSTANT_NAMES order."""
    with CONSTANTS_PATH.open("rb") as stream:
        table = tomllib.load(stream)["morrison_aero"][_table_name(morr_rimed_ice)]
    if int(table["morr_rimed_ice"]) != morr_rimed_ice:
        raise ValueError("constants.toml morrison_aero table is mislabelled")
    if activation_temperature not in ("wrf", "melting-point"):
        raise ValueError("activation_temperature must be wrf or melting-point")
    words = np.array([int(table.get(name + "_PINNED", table[name]) if activation_temperature == "wrf" else table[name], 16) for name in MA_CONSTANT_NAMES],
                     dtype=np.uint32)
    words.setflags(write=False)
    return words


def constant_value(name: str, morr_rimed_ice: int = 1) -> float:
    """One constant as a Python float (tests and diagnostics)."""
    word = constant_words(morr_rimed_ice)[MA_CONSTANT_NAMES.index(name)]
    return float(np.uint32(word).view(np.float32))


@cuda_cache(maxsize=None, ready=True)
def _device_constants(morr_rimed_ice: int, activation_temperature: str):
    import cupy as cp
    return cp.asarray(constant_words(morr_rimed_ice, activation_temperature).view(np.float32))


def _kernel(func: str, nz: int, activation_temperature: str, positive_sedimentation=False):
    from woof.core.kernels.no_ftz import get_kernel_no_ftz
    tier = next(t for t in KMAX_TIERS if nz <= t)
    return get_kernel_no_ftz("morrison_aero", func, (("MA_KMAX", tier), ("MA_T0_MILLIKELVIN", 300000 if activation_temperature == "wrf" else 273150))
                             + ((("MA_POSITIVE_SEDIMENTATION", 1),) if positive_sedimentation else ()))


def refuse_aercu_opt(aercu_opt: int, *, cu_physics: int = 0, mp_physics: int = 40) -> None:
    if aercu_opt not in (0, 1, 2):
        raise ValueError("aercu_opt must be 0, 1 or 2")
    if aercu_opt > 0 and (cu_physics != 11 or mp_physics != 40):
        raise ValueError(f"aercu_opt > 0 requires cu_physics=11 and mp_physics=40 (got cu_physics={cu_physics}, mp_physics={mp_physics}); "
                         "aercu_opt requires mp_physics=40 with multiscale Kain-Fritsch: without this pair the effective-radius handoff "
                         "has no aerosol activation (WRF module_check_a_mundo.F:1391-1404)")


def _require(name, value, shape, dtype=np.float32):
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must be float32, got {value.dtype}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def launch_morrison_aero(th, qv, qc, qr, qi, qs, qg, ni, ns, nr, ng, nc,
                         pii, pres, dz,
                         rainnc, rainncv, snownc, snowncv,
                         graupelnc, graupelncv, sr, dt: float, *,
                         morr_rimed_ice: int = 1, aercu_opt: int = 0,
                         aercu_fct: float = 1.0, activation_temperature: str = "wrf", w=None, kzh_up=None,
                         aerocu=None, qrcuten=None, qscuten=None,
                         qicuten=None, effc=None, effi=None, effs=None,
                         effr=None, ccn=None, efcg=None, efig=None,
                         efsg=None, wact=None, scratch=None,
                         positive_sedimentation=False):
    """One WRF ``MP_MORR_TWO_MOMENT_AERO`` call over a batch of columns.

    Default ``activation_temperature="wrf"`` retains the public T0=300 K.
    ``"melting-point"`` is an explicit diagnostic arm with SVPT0=273.15 K,
    graded against t0fix.patch and never selected by a forecast default.

    ``th`` through ``nc`` and the seven surface fields are updated in place
    exactly as the WRF wrapper updates them.  ``kzh_up`` is the eddy
    diffusivity one level above each mass level (WRF's ``KZH(I,K+1,J)``);
    ``w``/``kzh_up``/``aerocu`` are read only under ``aercu_opt = 2`` (and
    ``w``/``kzh_up`` for WACT under ``aercu_opt > 0``).  ``aerocu`` is
    ``(10, nz, ny, nx)`` in the Registry's species order (dust1..4,
    seasalt, sulfate, phobcar, phibcar, phoocar, phiocar), ug m-3.
    ``ccn`` is ``(7, nz, ny, nx)``.  Returns the scratch block, whose plane
    0 is the post-call air temperature WRF's reflectivity reads.
    """
    import cupy as cp

    shape = th.shape
    if len(shape) != 3:
        raise ValueError(f"mp_physics=40 fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    lo, hi = VERTICAL_LEVEL_BOUNDS
    if nz > hi:
        raise ValueError(
            f"nz={nz} exceeds {hi}: the sedimentation lanes hold each "
            "column's profiles in fixed local arrays of that size")
    if nz < lo:
        raise ValueError(f"mp_physics=40 needs nz >= {lo}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if aercu_opt not in (0, 1, 2):
        raise ValueError(f"aercu_opt must be 0, 1 or 2, got {aercu_opt!r}")
    for name, value in (("th", th), ("qv", qv), ("qc", qc), ("qr", qr),
                        ("qi", qi), ("qs", qs), ("qg", qg), ("ni", ni),
                        ("ns", ns), ("nr", nr), ("ng", ng), ("nc", nc),
                        ("pii", pii), ("pres", pres), ("dz", dz)):
        _require(name, value, shape)
    for name, value in (("rainnc", rainnc), ("rainncv", rainncv),
                        ("snownc", snownc), ("snowncv", snowncv),
                        ("graupelnc", graupelnc), ("graupelncv", graupelncv),
                        ("sr", sr)):
        _require(name, value, (ny, nx))
    cu = (qrcuten, qscuten, qicuten)
    has_cu = any(v is not None for v in cu)
    if has_cu and not all(v is not None for v in cu):
        raise ValueError("qrcuten, qscuten and qicuten must be supplied together")
    if has_cu:
        for name, value in zip(("qrcuten", "qscuten", "qicuten"), cu):
            _require(name, value, shape)
    if aercu_opt > 0:
        if w is None or kzh_up is None:
            raise ValueError("aercu_opt > 0 needs w and kzh_up")
        _require("w", w, shape)
        _require("kzh_up", kzh_up, shape)
    if aercu_opt == 2:
        if aerocu is None:
            raise ValueError("aercu_opt = 2 needs the prescribed aerosol aerocu")
        _require("aerocu", aerocu, (10,) + shape)

    ncell = nz * ny * nx
    ncol = ny * nx
    if scratch is None:
        scratch = cp.empty((_SCRATCH_PLANES,) + shape, dtype=np.float32)
    else:
        _require("scratch", scratch, (_SCRATCH_PLANES,) + shape)
    outputs = {"effc": effc, "effi": effi, "effs": effs, "effr": effr}
    for name, value in outputs.items():
        if value is None:
            outputs[name] = cp.empty(shape, dtype=np.float32)
        else:
            _require(name, value, shape)
    radiation = {"efcg": efcg, "efig": efig, "efsg": efsg, "wact": wact}
    for name, value in radiation.items():
        if aercu_opt > 0:
            if value is None:
                raise ValueError(f"aercu_opt > 0 writes {name}; supply it")
            _require(name, value, shape)
        elif value is None:
            radiation[name] = scratch[0]       # never written when aercu_opt = 0
    if aercu_opt > 0:
        if ccn is None:
            raise ValueError("aercu_opt > 0 writes CCN1..7; supply ccn")
        _require("ccn", ccn, (7,) + shape)
    dummy = scratch[0]
    kc = _device_constants(morr_rimed_ice, activation_temperature)
    f32 = np.float32

    process = _kernel("morrison_aero_process", nz, activation_temperature)
    sediment = _kernel("morrison_aero_sediment", nz, activation_temperature, positive_sedimentation)
    finalize = _kernel("morrison_aero_finalize", nz, activation_temperature)
    cell_blocks = (ncell + _CELL_TPB - 1) // _CELL_TPB
    warps = (ncol + _COLUMNS_PER_WARP - 1) // _COLUMNS_PER_WARP
    sediment_blocks = (warps * 32 + _SEDIMENT_TPB - 1) // _SEDIMENT_TPB

    process((cell_blocks,), (_CELL_TPB,),
            (th, qv, qc, qr, qi, qs, qg, ni, ns, nr, ng, nc, pii, pres,
             w if w is not None else dummy,
             kzh_up if kzh_up is not None else dummy,
             qrcuten if has_cu else dummy, qscuten if has_cu else dummy,
             qicuten if has_cu else dummy,
             aerocu if aerocu is not None else dummy,
             ccn if ccn is not None else dummy, scratch, kc,
             f32(dt), f32(aercu_fct), np.int32(aercu_opt),
             np.int32(has_cu), np.int32(ncell)))
    if aercu_opt == 2:
        _check_activation_status(scratch[21])
    sediment((sediment_blocks,), (_SEDIMENT_TPB,),
             (qc, qr, qi, qs, qg, nc, nr, ni, ns, ng, pres, dz, scratch, kc,
              rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr,
              f32(dt), np.int32(aercu_opt), np.int32(nz), np.int32(ncol)))
    finalize((cell_blocks,), (_CELL_TPB,),
             (th, qv, qc, qr, qi, qs, qg, ni, ns, nr, ng, nc, pii, pres,
              w if w is not None else dummy,
              kzh_up if kzh_up is not None else dummy,
              outputs["effc"], outputs["effi"], outputs["effs"],
              outputs["effr"], radiation["efcg"], radiation["efig"],
              radiation["efsg"], radiation["wact"], scratch, kc,
              f32(dt), np.int32(aercu_opt), np.int32(ncell), np.int32(ncol)))
    return scratch


def _raise_activation_status(status):
    if status:
        from woof.core.health_ledger import deferred_note
        raise ValueError("mp40 prescribed ice nucleation left WRF's [0,9999] cm-3 range; WRF stops at module_mp_morr_two_moment_aero.F:6009-6014 because the predicted crystal count is invalid. The column cannot continue." + deferred_note())


def _check_activation_status(flags):
    # Keep the native per-cell flags and reduction. Only their readback moves
    # to the resident executor's mandatory health drain before publication.
    import cupy as cp
    from woof.core.health_ledger import read_status
    status = cp.any(flags).astype(cp.int32).reshape(1)
    failure = read_status(status, site="mp40-aerosol-ice",
                          describe=_raise_activation_status)
    if failure:
        _raise_activation_status(failure)


__all__ = ["CONSTANTS_PATH", "MA_CONSTANT_NAMES", "KMAX_TIERS",
           "VERTICAL_LEVEL_BOUNDS", "constant_words", "constant_value",
           "launch_morrison_aero", "refuse_aercu_opt"]


@lru_cache(maxsize=2)
def radar_products(morr_rimed_ice=1):
    """WRF radar_init products from the existing constants table path."""
    with CONSTANTS_PATH.open("rb") as stream:
        table=tomllib.load(stream)["morrison_aero"][_table_name(morr_rimed_ice)]["radar"]
    def value(name):
        word=table[name][2:]
        import struct
        return struct.unpack(">f" if len(word)==8 else ">d",bytes.fromhex(word))[0]
    names="xam_r xam_s xam_g xorg2 xosg2 xogg2 xobmr xobms xobmg xocms xocmg xcrg_3 xcrg_4 pi5 lamda4".split()
    rc=np.array([value(name) for name in names],dtype=np.float64)
    bins=np.array([value(f"{name}_{n}") for name in ("xxds","xdts","xxdg","xdtg","simpson") for n in range(1,51)],dtype=np.float64)
    materials=tuple(value(name) for name in ("k_w","m_w_0_re","m_w_0_im","m_i_0_re","m_i_0_im"))
    rc.setflags(write=False); bins.setflags(write=False)
    return rc,bins,materials

@cuda_cache(maxsize=None, ready=True)
def _radar_device(morr_rimed_ice):
    import cupy as cp
    rc,bins,_=radar_products(morr_rimed_ice)
    return cp.asarray(rc),cp.asarray(bins)

def launch_refl_morrison_aero(qv,qr,nr,qs,ns,qg,ng,t,p,refl,*,morr_rimed_ice=1):
    from woof.core.kernels.no_ftz import get_kernel_no_ftz
    nz,ny,nx=qv.shape
    for name,value in (("qv",qv),("qr",qr),("nr",nr),("qs",qs),("ns",ns),("qg",qg),("ng",ng),("t",t),("p",p),("refl",refl)):
        _require(name,value,qv.shape)
    rc,bins=_radar_device(morr_rimed_ice)
    _,_,materials=radar_products(morr_rimed_ice)
    kernel=get_kernel_no_ftz("morrison_aero_refl","refl10cm_morrison_aero_column",(("REFL_KMAX",nz),))
    kernel(((ny*nx+31)//32,), (32,), (qv,qr,nr,qs,ns,qg,ng,t,p,bins,*map(np.float64,materials),rc,refl,np.int32(nz),np.int32(ny),np.int32(nx)))


def apply(state, cfg, dt, *, refl_10cm_due=False):
    import cupy as cp
    from woof.core.morrison import _prepare_fields, launch_exner
    from woof.core import constants as c
    from woof.core.microphysics import MicrophysicsDiagnostics, moist_physics_finish, save_pre_mp_theta
    refuse_aercu_opt(cfg.aercu_opt, cu_physics=cfg.cu_physics, mp_physics=cfg.mp_physics)
    nz,ny,nx=state.p.shape
    theta=state.scratch((nz,ny,nx),"morr_theta")
    pii=state.scratch((nz,ny,nx),"morr_pii")
    dz=state.scratch((nz,ny,nx),"morr_dz")
    thb=state.thb if state.thb.ndim==3 else state.thb[:,None,None]
    phb=state.phb if state.phb.ndim==3 else state.phb[:,None,None]
    _prepare_fields(thb, phb, state.thp, state.php, np.int32(ny*nx),
                    np.int32(state.thb.ndim == 3),
                    np.int32(state.phb.ndim == 3), np.float32(c.G), theta, dz)
    launch_exner(state.p, pii)
    rainnc=state.scratch((ny,nx),"mp_rainnc")
    rainncv=state.scratch((ny,nx),"mp_rainncv")
    snownc=state.scratch((ny,nx),"mp_snownc")
    snowncv=state.scratch((ny,nx),"mp_snowncv")
    graupelnc=state.scratch((ny,nx),"mp_graupelnc")
    graupelncv=state.scratch((ny,nx),"mp_graupelncv")
    sr=state.scratch((ny,nx),"mp_sr")
    work=state.scratch((22,nz,ny,nx),"morr_aero_work")
    save_pre_mp_theta(state)
    rates=getattr(getattr(state,"physics",None),"cu_rates",None) if cfg.cu_physics else None
    ccn=state.scratch((7,nz,ny,nx),"morr_aero_ccn") if cfg.aercu_opt>0 else None
    kzh_up=None
    if cfg.aercu_opt>0:
        # WRF wrapper reads KZH(k+1), module_mp_morr_two_moment_aero.F:931.
        # The shared PBL interface retains k=1..nz; the top interface is zero.
        kzh_up=state.scratch((nz,ny,nx),"morr_aero_kzh_up")
        kzh_up[:-1]=state.physics.fields["exch_h"][1:]
        kzh_up[-1]=0


    launch_morrison_aero(theta,state.qv,state.qc,state.qr,state.qi,state.qs,state.qg,state.ni,state.ns,state.nr,state.ng,state.nc,pii,state.p,dz,rainnc,rainncv,snownc,snowncv,graupelnc,graupelncv,sr,dt,morr_rimed_ice=cfg.morr_rimed_ice,aercu_opt=cfg.aercu_opt,aercu_fct=cfg.aercu_fct,w=cp.ascontiguousarray(state.w[:nz]),kzh_up=kzh_up,aerocu=getattr(state,"aerocu",None),ccn=ccn,efcg=getattr(state,"efcg",None),efig=getattr(state,"efig",None),efsg=getattr(state,"efsg",None),wact=getattr(state,"wact",None),effc=state.effc,effi=state.effi,effs=state.effs,effr=state.effr,scratch=work,positive_sedimentation=True,qrcuten=None if rates is None else rates["rqrcuten"],qscuten=None if rates is None else rates["rqscuten"],qicuten=None if rates is None else rates["rqicuten"])
    if ccn is not None:
        for index in range(7):
            getattr(state,f"ccn{index+1}_gs")[...]=ccn[index]
    # The native inactive-column shortcut leaves its raw diagnostic radii
    # at zero. They are undefined sizes for absent particles, not micron
    # radii. Publish the same no-particle backgrounds as the active arm,
    # while leaving launch_morrison_aero's native replay outputs intact.
    inactive = work[20, 0][None, ...] == 0
    for name, background in (("effc", 2.49), ("effi", 4.99),
                             ("effs", 25.0), ("effr", 25.0)):
        cp.copyto(getattr(state, name), cp.float32(background), where=inactive)
    if refl_10cm_due:
        from woof.core.refl import compute_and_stash_refl_10cm
        compute_and_stash_refl_10cm(state,cfg,cp.ascontiguousarray(work[0]),state.p)
    moist_physics_finish(state,cfg,theta,dt)
    from woof.core.morrison import bound_snow_ratio
    bound_snow_ratio(sr)
    return MicrophysicsDiagnostics(rainnc=rainnc,rainncv=rainncv,sr=sr,snownc=snownc,snowncv=snowncv,graupelnc=graupelnc,graupelncv=graupelncv)
