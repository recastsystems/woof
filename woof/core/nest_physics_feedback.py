"""WRF 4.6.1 live non-prognostic nest feedback.

Registry.EM_COMMON's ``u`` rows select the operator. Only fields actually
carried on both domains participate. Physics fields use their native Registry
symbols; atmospheric prognostics and base state remain in nest.py. Every
runtime two-way feedback invokes this inventory, independently of the thermal
variable and numerics mode.
"""
from __future__ import annotations

# Public-domain WRF Registry.EM_COMMON, non-dyn_em ij/ilj/ikj copy u rows.
# NEST_POS/NEST_MASK use mark_domain, a geometry marker, not a field copy.
FEEDBACK_OPERATORS = {
    "lu_index": "pick",
    "q2": "avg",
    "t2": "avg",
    "th2": "avg",
    "psfc": "avg",
    "u10": "avg",
    "v10": "avg",
    "lpi": "avg",
    "qvold": "avg",
    "qnwfa2d": "avg",
    "qnifa2d": "avg",
    "qnbca2d": "avg",
    "qnocbb2d": "avg",
    "qnbcbb2d": "avg",
    "toposlpx": "avg",
    "toposlpy": "avg",
    "slope": "avg",
    "slp_azi": "avg",
    "shdmax": "pick",
    "shdmin": "pick",
    "shdavg": "pick",
    "irrigation": "avg",
    "irr_rand_field": "avg",
    "tslb": "pick",
    "smois": "pick",
    "sh2o": "pick",
    "smcrel": "pick",
    "xice": "pick",
    "icedepth": "pick",
    "xicem": "pick",
    "albsi": "pick",
    "snowsi": "pick",
    "ivgtyp": "ipick",
    "isltyp": "ipick",
    "vegfra": "pick",
    "acgrdflx": "avg",
    "acsnow": "pick",
    "acrunoff": "pick",
    "acsnom": "pick",
    "snow": "pick",
    "snowh": "pick",
    "canwat": "pick",
    "sstsk": "pick",
    "tsk_rural": "pick",
    "tr_urb2d": "pick",
    "tgr_urb2d": "pick",
    "tb_urb2d": "pick",
    "tg_urb2d": "pick",
    "tc_urb2d": "pick",
    "qc_urb2d": "pick",
    "uc_urb2d": "pick",
    "xxxr_urb2d": "pick",
    "xxxb_urb2d": "pick",
    "xxxg_urb2d": "pick",
    "xxxc_urb2d": "pick",
    "cmcr_urb2d": "pick",
    "drelr_urb2d": "pick",
    "drelb_urb2d": "pick",
    "drelg_urb2d": "pick",
    "flxhumr_urb2d": "pick",
    "flxhumb_urb2d": "pick",
    "flxhumg_urb2d": "pick",
    "tgrl_urb3d": "pick",
    "smr_urb3d": "pick",
    "trl_urb3d": "pick",
    "tbl_urb3d": "pick",
    "tgl_urb3d": "pick",
    "sh_urb2d": "pick",
    "lh_urb2d": "pick",
    "g_urb2d": "pick",
    "rn_urb2d": "pick",
    "ts_urb2d": "pick",
    "frc_urb2d": "pick",
    "utype_urb2d": "ipick",
    "imperv": "pick",
    "canfra": "pick",
    "u10e": "avg",
    "v10e": "avg",
    "ctopo": "avg",
    "ctopo2": "avg",
    "f_ice_phy": "avg",
    "f_rain_phy": "avg",
    "f_rimef_phy": "avg",
    "om_ml": "pick",
    "h_diabatic": "avg",
    "qv_diabatic": "avg",
    "qc_diabatic": "avg",
    "refl_10cm": "avg",
    "msft": "pick",
    "msfu": "pick",
    "msfv": "pick",
    "msftx": "pick",
    "msfty": "pick",
    "msfux": "pick",
    "msfuy": "pick",
    "msfvx": "pick",
    "msfvx_inv": "pick",
    "msfvy": "pick",
    "f": "pick",
    "e": "pick",
    "sina": "pick",
    "cosa": "pick",
    "ht": "avg",
    "tsk": "pick",
    "rainc": "avg",
    "rainsh": "avg",
    "rainnc": "avg",
    "i_rainc": "ipick",
    "i_rainnc": "ipick",
    "snownc": "avg",
    "graupelnc": "avg",
    "hailnc": "avg",
    "th_old": "avg",
    "qv_old": "avg",
    "vmi3d": "avg",
    "di3d": "avg",
    "rhopo3d": "avg",
    "phii3d": "avg",
    "vmi3d_2": "avg",
    "di3d_2": "avg",
    "rhopo3d_2": "avg",
    "phii3d_2": "avg",
    "vmi3d_3": "avg",
    "di3d_3": "avg",
    "rhopo3d_3": "avg",
    "phii3d_3": "avg",
    "itype": "avg",
    "itype_2": "avg",
    "itype_3": "avg",
    "acswupt": "avg",
    "acswuptc": "avg",
    "acswdnt": "avg",
    "acswdntc": "avg",
    "acswupb": "avg",
    "acswupbc": "avg",
    "acswdnb": "avg",
    "acswdnbc": "avg",
    "aclwupt": "avg",
    "aclwuptc": "avg",
    "aclwdnt": "avg",
    "aclwdntc": "avg",
    "aclwupb": "avg",
    "aclwupbc": "avg",
    "aclwdnb": "avg",
    "aclwdnbc": "avg",
    "i_acswupt": "ipick",
    "i_acswuptc": "ipick",
    "i_acswdnt": "ipick",
    "i_acswdntc": "ipick",
    "i_acswupb": "ipick",
    "i_acswupbc": "ipick",
    "i_acswdnb": "ipick",
    "i_acswdnbc": "ipick",
    "i_aclwupt": "ipick",
    "i_aclwuptc": "ipick",
    "i_aclwdnt": "ipick",
    "i_aclwdntc": "ipick",
    "i_aclwupb": "ipick",
    "i_aclwupbc": "ipick",
    "i_aclwdnb": "ipick",
    "i_aclwdnbc": "ipick",
    "swupt": "avg",
    "swuptc": "avg",
    "swuptcln": "avg",
    "swdnt": "avg",
    "swdntc": "avg",
    "swdntcln": "avg",
    "swupb": "avg",
    "swupbc": "avg",
    "swupbcln": "avg",
    "swdnb": "avg",
    "swdnbc": "avg",
    "swdnbcln": "avg",
    "lwupt": "avg",
    "lwuptc": "avg",
    "lwuptcln": "avg",
    "lwdnt": "avg",
    "lwdntc": "avg",
    "lwdntcln": "avg",
    "lwupb": "avg",
    "lwupbc": "avg",
    "lwupbcln": "avg",
    "lwdnb": "avg",
    "lwdnbc": "avg",
    "lwdnbcln": "avg",
    "clat": "pick",
    "tmn": "pick",
    "tyr": "pick",
    "tyra": "pick",
    "tdly": "pick",
    "xland": "pick",
    "achfx": "avg",
    "aclhf": "avg",
    "snowc": "pick",
    "prec_acc_c": "avg",
    "prec_acc_nc": "avg",
    "snow_acc_nc": "avg",
    "tml": "pick",
    "t0ml": "pick",
    "hml": "pick",
    "h0ml": "pick",
    "huml": "pick",
    "hvml": "pick",
    "tmoml": "pick",
}


def _carriers(state):
    """Registry symbol -> live array, without allocating device storage."""
    owner = getattr(state, "_streamed_domain", None)
    if owner is not None:
        store = owner.store
        fields = {key[7:]: value for key, value in store.items()
                  if key.startswith("fields/")}
        get_state = lambda name: store.get(f"state/{name}")
        get_scratch = lambda name: store.get(f"scratch/{name}")
        rainc = get_scratch("cu_rainc")
    else:
        physics = getattr(state, "physics", None)
        fields = dict(getattr(physics, "fields", {}))
        get_state = lambda name: getattr(state, name, None)
        get_scratch = lambda name: getattr(state, "_scratch", {}).get(name)
        rainc = getattr(physics, "rainc", None)
        urban = getattr(physics, "urban", None)
        if urban is not None:
            fields.update(getattr(urban, "fields", {}))
    if rainc is not None:
        fields["rainc"] = rainc
    # Registry's hdu radar field uses copy_fcn too. The history handoff
    # aliases this carrying scratch, so feedback must update it before the
    # parent frame consumes it (nest_feedbackup_interp.inc:2159).
    refl = get_scratch("refl_10cm")
    if refl is not None:
        fields["refl_10cm"] = refl
    for name in ("rainnc", "snownc", "graupelnc", "hailnc"):
        value = get_scratch("mp_" + name)
        if value is not None:
            fields[name] = value
    for native, attr in (
            ("h_diabatic", "h_diabatic"), ("qv_diabatic", "qv_diabatic"),
            ("qc_diabatic", "qc_diabatic"), ("qnwfa2d", "nwfa2d"),
            ("qnifa2d", "nifa2d"), ("f_ice_phy", "f_ice"),
            ("f_rain_phy", "f_rain"), ("f_rimef_phy", "f_rimef"),
            ("th_old", "th_old"), ("qv_old", "qv_old"),
            ("vmi3d", "vmi3d"), ("di3d", "di3d"),
            ("rhopo3d", "rhopo3d")):
        value = get_state(attr)
        if value is not None:
            fields[native] = value
    return fields


def feedback_physics(coupler, node, *, skip_names=()):
    """Restrict each live surface, soil and carried-physics Registry field.

    Resident arrays use the certified native copy kernels. Canonical stores
    are read and written in bounded rectangles; masked copy is a single
    strided child pick, with WRF's different odd/even offsets.
    """
    import cupy as cp
    import numpy as np
    from woof.core.nest_interp import (
        copy_fcn, copy_fcnm, copy_fcni, feedback_child_window,
        feedback_parent_bounds)
    from woof.core.nest_operands import NestWindowSource, rectangle_windows

    parent = node.parent
    coarse, fine = _carriers(parent.state), _carriers(node.state)
    reg = coupler.registrations["m"]
    zone = node.cfg.run.spec_zone
    bounded = (getattr(parent.state, "_streamed_domain", None) is not None
               or getattr(node.state, "_streamed_domain", None) is not None)
    source, target = NestWindowSource(node.state), NestWindowSource(parent.state)
    operators = {"avg": copy_fcn, "pick": copy_fcnm, "ipick": copy_fcni}
    committed = []
    for name, operator in FEEDBACK_OPERATORS.items():
        if name in skip_names:
            continue
        c, n = coarse.get(name), fine.get(name)
        if c is None or n is None:
            continue
        if (getattr(c, "ndim", 0) not in (2, 3)
                or getattr(n, "ndim", 0) != c.ndim):
            continue
        if c.shape[:-2] != n.shape[:-2]:
            # Different soil or snow grids cannot exchange layer words.
            raise ValueError(f"nest physics feedback {name}: parent and child "
                             "layer counts differ")
        if c.shape[-2:] != (reg.nyp, reg.nxp) or n.shape[-2:] != (reg.nyc, reg.nxc):
            raise ValueError(f"nest physics feedback {name}: carrier grid "
                             "does not match the domain")
        committed.append(name)
        if not bounded:
            operators[operator](c, n, reg, spec_zone=zone)
            continue
        ilo, ihi, jlo, jhi = feedback_parent_bounds(reg, spec_zone=zone)
        sy, sx = coupler._transaction_chunk_shape()
        chunks = (max(1, sy // reg.nrj), max(1, sx // reg.nri))
        for window in rectangle_windows(
                (slice(jlo, jhi + 1), slice(ilo, ihi + 1)), chunks):
            if operator == "avg":
                donor = feedback_child_window(reg, window, spec_zone=zone)
                child = source.device_array(n, donor)
                result = cp.empty(c[(...,) + window].shape, dtype=c.dtype)
                copy_fcn(result, child, reg, spec_zone=zone,
                         parent_window=window, child_window=donor)
            else:
                y, x = window
                ox = 1 if reg.nri % 2 else reg.nri // 2 - 1
                oy = 1 if reg.nrj % 2 else reg.nrj // 2 - 1
                donor = (slice((y.start + 1 - reg.j_parent_start) * reg.nrj + oy,
                               (y.stop + 1 - reg.j_parent_start) * reg.nrj + oy,
                               reg.nrj),
                         slice((x.start + 1 - reg.i_parent_start) * reg.nri + ox,
                               (x.stop + 1 - reg.i_parent_start) * reg.nri + ox,
                               reg.nri))
                result = source.device_array(n, donor)
            dest = c[(...,) + window]
            if isinstance(dest, np.ndarray):
                dest[...] = cp.asnumpy(result)
                target.device_to_host_bytes += int(result.nbytes)
            else:
                dest[...] = result
    coupler.feedback_sync_bytes += (source.host_to_device_bytes
                                    + target.device_to_host_bytes)
    return tuple(committed)
