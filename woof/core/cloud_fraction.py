"""WRF 4.7.1 cal_cldfra3 and temporary radiation moisture.

Native radiation_driver.F:3131-3154 restores qc/qi after radiation. The
condensate added here must never enter prognostic state. Source pins and
licence are recorded in tools/cloud_aerosol_wrf471_oracle.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _bad_profile(_flags):
    raise ValueError(
        "cal_cldfra3: the -12 C level is within the top three model levels; "
        "WRF find_cloudLayers refuses this profile "
        "(module_radiation_driver.F:4193)")


def cal_cldfra3(qv, qc, qi, qs, dz, pressure, temperature, xland, gridkm,
                *, modify_qvapor=False, max_relh=1.5):
    """Return fraction and adjusted qv/qc/qi without modifying any input.

    Arrays are bottom-up (ncol,nlev) float32 columns. Model steps defer the
    native profile stop through the existing health ledger. Standalone
    module oracles check it immediately.
    """
    import cupy as cp
    from woof.core.health_ledger import read_status
    from woof.core.kernels import get_kernel
    shape = qv.shape
    if len(shape) != 2 or shape[1] < 6:
        raise ValueError("cal_cldfra3 needs (ncol,nlev) with at least six levels")
    arrays = []
    names = ("qv", "qc", "qi", "qs", "dz", "pressure", "temperature")
    for index, (name, value) in enumerate(zip(
            names, (qv, qc, qi, qs, dz, pressure, temperature))):
        array = cp.asarray(value, dtype=cp.float32)
        if array.shape != shape:
            raise ValueError(f"cal_cldfra3 {name}: {array.shape} != {shape}")
        arrays.append(cp.array(array, order="C", copy=True) if index < 3
                      else cp.ascontiguousarray(array))
    nc, nz = shape
    land = cp.ascontiguousarray(cp.broadcast_to(
        cp.asarray(xland, dtype=cp.float32), (nc,)))
    spacing = cp.ascontiguousarray(cp.broadcast_to(
        cp.asarray(gridkm, dtype=cp.float32), (nc,)))
    fraction = cp.empty(shape, dtype=cp.float32)
    work = cp.empty((4,nc,nz), dtype=cp.float32)
    status = cp.zeros(1, dtype=cp.uint32)
    get_kernel("cldfra3", "cal_cldfra3_column")(
        ((nc + 127) // 128,), (128,),
        (np.int32(nc), np.int32(nz), *arrays, land, spacing,
         np.int32(modify_qvapor), np.float32(max_relh), fraction, work, status))
    if read_status(status, site="radiation.cal_cldfra3", describe=_bad_profile):
        _bad_profile(1)
    return fraction, *arrays[:3]


class _RadiationMoistureView:
    """Read-through state with only radiation qc/qi replaced."""
    def __init__(self, state, qc, qi):
        self._state, self.qc, self.qi = state, qc, qi

    def __getattr__(self, name):
        return getattr(self._state, name)


@dataclass(frozen=True)
class RadiationCloudView:
    atmosphere: dict
    fields: dict
    state: object
    fraction: object


def icloud_cu(cfg):
    """Derived physics_init selector; ICLOUD_CU is not a namelist knob."""
    scheme = int(cfg.cu_physics)
    if scheme == 11:
        return 2
    if not cfg.cu_rad_feedback:
        return 0
    return 2 if scheme == 1 else 1 if scheme in (3, 5, 93) else 0


def merge_cumulus_clouds(qc, qi, fraction, fields, *, mode, fqc=True, fqi=True):
    """Native ordered merge into private device buffers, without host reads."""
    import cupy as cp
    from woof.core.kernels import get_kernel
    shape = qc.shape
    qc, qi = qc.copy(order="C"), qi.copy(order="C")
    fraction = (cp.zeros(shape, cp.float32) if fraction is None
                else fraction.copy(order="C"))
    cu_fraction = fields.get("cldfra_cu")
    if cu_fraction is None:
        cu_fraction = cp.empty(shape, cp.float32)
    required = ("qc_cu", "qi_cu") + (("cldfra_dp", "cldfra_sh") if mode == 2 else ())
    for key in required:
        if key not in fields or fields[key].shape != shape:
            raise ValueError(f"cumulus radiation feedback needs carried {key} {shape}")
    dummy = qc
    get_kernel("cu_radiation", "cu_radiation_merge")(
        ((qc.size + 127)//128,), (128,),
        (np.int64(qc.size),np.int32(mode),np.int32(fqc),np.int32(fqi),
         fields["qc_cu"],fields["qi_cu"],fields.get("cldfra_dp",dummy),
         fields.get("cldfra_sh",dummy),qc,qi,fraction,cu_fraction))
    return qc, qi, fraction


def prepare_radiation_clouds(atmosphere, fields, state, cfg):
    """WRF temporary condensate, then the selected cloud diagnosis/merge.

    MYNN's later merge remains with each radiation adapter. FARMS applies
    that same merge after this view, on its every-Solar_step call.
    """
    import cupy as cp
    from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags
    mode = icloud_cu(cfg)
    if atmosphere.get("cumulus_feedback") is not None:
        # MSKF's radiation adapters merge its carried condensate, snow,
        # fraction and aerosol radii together. Do not merge qc/qi here too.
        mode = 0
    fqi, fqs = legacy_cloud_fraction_flags(int(cfg.mp_physics))
    original_fields = fields
    if mode == 1:
        qc, qi, _ = merge_cumulus_clouds(atmosphere["qc"],atmosphere["qi"],
            None,fields,mode=1,fqc=int(cfg.mp_physics)!=0,fqi=fqi)
        atmosphere = dict(atmosphere,qc=qc,qi=qi)
        state = _RadiationMoistureView(state,qc,qi)
    if int(cfg.icloud) == 3:
        return prepare_cloud_fraction3(atmosphere,fields,state,cfg)
    if mode == 2 and int(cfg.icloud) == 1:
        from woof.core.rrtmgp import cal_cldfra1
        nz,ny,nx = shape = atmosphere["qc"].shape
        cols = lambda a: cp.ascontiguousarray(a.reshape(nz,ny*nx).T)
        fraction = cal_cldfra1(*(cols(atmosphere[key]) for key in
            ("qv","qc","qi","qs","temperature","pressure")),
            f_qc=int(cfg.mp_physics)!=0,f_qi=fqi,f_qs=fqs,mp_physics=int(cfg.mp_physics),
            f_ice_phy=cols(state.f_ice_phy) if int(cfg.mp_physics)==95 else None)
        fraction = cp.ascontiguousarray(fraction.T.reshape(shape))
        qc,qi,fraction = merge_cumulus_clouds(atmosphere["qc"],atmosphere["qi"],
            fraction,fields,mode=2)
        return RadiationCloudView(dict(atmosphere,qc=qc,qi=qi),
            dict(fields,radiation_cldfra3=fraction),
            _RadiationMoistureView(state,qc,qi),fraction)
    return RadiationCloudView(dict(atmosphere),original_fields,state,None)


def prepare_cloud_fraction3(atmosphere, fields, state, cfg):
    """Prepare both solvers' temporary moisture with WRF's diagonal dx.

    Original atmosphere, fields, and state arrays are never written. The
    temporary view can feed FARMS before being discarded after this step.
    """
    import cupy as cp
    nz, ny, nx = shape = atmosphere["temperature"].shape

    def columns(value):
        return cp.ascontiguousarray(value.reshape(nz, ny * nx).T)

    def grid(value):
        return cp.ascontiguousarray(value.T.reshape(shape))

    from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags
    if not legacy_cloud_fraction_flags(int(cfg.mp_physics))[0]:
        # Native ICLOUD=3 runs only when both F_QC and F_QI are present.
        raise ValueError("Can not use icloud = 3 option, missing QC or QI field.")
    # radiation_driver.F:1056, left-associated REAL multiplications.
    f = np.float32
    dx, dy = f(cfg.dx), f(cfg.dy)
    gridkm = f(np.sqrt(f(f(f(f(dx*f(.001))*dx)*f(.001))
                        + f(f(f(dy*f(.001))*dy)*f(.001)))))
    cf, _qv, qc, qi = cal_cldfra3(
        *(columns(atmosphere[key]) for key in
          ("qv", "qc", "qi", "qs", "dz", "pressure", "temperature")),
        cp.asarray(fields["xland"]).reshape(-1), gridkm)
    qc, qi, fraction = grid(qc), grid(qi), grid(cf)
    return RadiationCloudView(
        dict(atmosphere, qc=qc, qi=qi),
        dict(fields, radiation_cldfra3=fraction),
        _RadiationMoistureView(state, qc, qi), fraction)


def cloud_fraction_column_max(fraction):
    """Maximum over bottom-up (ncol,nlev) fraction, on the device."""
    import cupy as cp
    from woof.core.kernels import get_kernel
    fraction = cp.ascontiguousarray(fraction, dtype=cp.float32)
    nc, nz = fraction.shape
    out = cp.empty(nc, dtype=cp.float32)
    get_kernel("cldfra3", "cldfra3_column_max")(
        ((nc + 127) // 128,), (128,), (np.int32(nc), np.int32(nz), fraction, out))
    return out
