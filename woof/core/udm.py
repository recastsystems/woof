"""WRF 4.7.1 UDM (MP27), with its retained initialized table words."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data/udm"
VOLUME = "th q qc qr qi qs qg qh nn nc nr den pii p delz re_cloud re_ice re_snow".split()
SURFACE = "xland xice rain rainncv sr".split()
OPTIONAL = "refl_10cm snow snowncv graupel graupelncv hail hailncv".split()
RADIUS_REASON = ("WRF 4.7.1 module_mp_udm.F:411 passes den1d to its radius "
                 "calculation without assigning it (declaration :289). "
                 "UDM scheme-radius radiation coupling has no defined native reference.")


@lru_cache(None)
def tables(device):
    import cupy as cp
    schema = json.loads((DATA / "table-schema.json").read_text())
    identity = json.loads((DATA / "identity.json").read_text())
    from woof import packed_data
    # (WOOF 1.1.6) a wheel carries udm-tables.bin.xz (woof.packed_data, PyPI's
    # per-file limit); the receipt is checked on the decompressed bytes.
    raw = packed_data.read_bytes(DATA / "udm-tables.bin")
    if len(raw) != identity["bytes"] or hashlib.sha256(raw).hexdigest() != identity["sha256"]:
        raise ValueError("UDM initialized table bytes do not match their WRF 4.7.1 receipt")
    offset = 0
    owner = {}
    for name, spec in sorted(schema.items()):
        a = np.frombuffer(raw, "<f8" if spec["typ"] == "double" else "<f4",
                          int(np.prod(spec["shape"])), offset)
        owner[name] = cp.asarray(a)
        offset += a.nbytes
    if offset != len(raw):
        raise ValueError("UDM table schema does not consume its full stream")
    pointers = cp.asarray(np.array([owner[n].data.ptr for n in sorted(owner)], dtype=np.uint64))
    return owner, pointers


def launch_driver(fields, scalars, *, first=False, radar=False):
    """The complete retained native boundary; C-order float32 (k,j,i)."""
    import cupy as cp
    from woof.core.kernels import load_module, load_module_int_defines
    from woof.core.wsm_family import check_volume, check_surface
    shape = fields["q"].shape
    nz, ny, nx = shape
    if not 1 <= nz <= 256:
        raise ValueError("UDM private column arrays support 1..256 levels")
    check_volume({n: fields[n] for n in VOLUME + ["refl_10cm"]}, shape)
    check_surface({n: fields[n] for n in SURFACE + OPTIONAL[1:]}, (ny, nx))
    if len(scalars) != 19:
        raise ValueError("UDM driver requires its 19 native scalar arguments")
    capacity = next(n for n in (64, 128, 256) if nz <= n)
    module = (load_module("udm") if capacity == 64 else
              load_module_int_defines("udm", (("UDM_COLUMN_KMAX", capacity),)))
    owner, tp = tables(cp.cuda.Device().id)
    fp = cp.asarray(np.array([fields[n].data.ptr for n in VOLUME + SURFACE + OPTIONAL], dtype=np.uint64))
    s = cp.asarray(np.asarray(scalars, dtype=np.float32))
    module.get_function("udm_step")(((ny * nx + 31) // 32,), (32,),
        (fp, s, np.int32(nz), np.int32(ny * nx), np.int32(first), np.int32(radar), tp))


def scalar_arguments(dt, ccn):
    # module_model_constants.F; the driver's REAL32 argument list.
    from woof.core import constants as c
    f = np.float32
    return (dt, c.G, c.CP, 1846.4, ccn, c.RD, c.RV, 273.15,
            f(f(c.RV) / f(c.RD)) - f(1), f(c.RD) / f(c.RV),
            1.e-15, 2.85e6, 2.5e6, 3.5e5, 1.28, 1000., 4190., 2106., 610.78)


def apply(state, cfg, dt, *, refl_10cm_due=False):
    """Prepare WRF driver fields and publish native precipitation and heating."""
    from woof.core.wsm6 import _wsm6_prepare
    from woof.core.microphysics import save_pre_mp_theta, moist_physics_finish, MicrophysicsDiagnostics
    from woof.core.refl import stash_refl_10cm
    theta, rho, pii, dz = _wsm6_prepare(state, "udm", rounded_exner=True)
    shape = state.p.shape
    horizontal = shape[1:]
    fields = {n: getattr(state, n) for n in ("qc", "qr", "qi", "qs", "qg", "qh", "nn", "nc", "nr")}
    fields.update(th=theta, q=state.qv, den=rho, pii=pii, p=state.p, delz=dz)
    surface = {}
    for cls in ("rain", "snow", "graupel", "hail"):
        for suffix in ("nc", "ncv"):
            key = cls + suffix
            surface[key] = state.scratch(horizontal, "mp_" + key)
    surface["sr"] = state.scratch(horizontal, "mp_sr")
    fields.update(rain=surface["rainnc"], rainncv=surface["rainncv"], sr=surface["sr"],
                  snow=surface["snownc"], snowncv=surface["snowncv"],
                  graupel=surface["graupelnc"], graupelncv=surface["graupelncv"],
                  hail=surface["hailnc"], hailncv=surface["hailncv"])
    driver = getattr(state, "physics", None)
    physical = getattr(driver, "fields", {})
    for name in ("xland", "xice"):
        if name not in physical:
            raise ValueError(f"UDM needs the physics driver's {name} mask (module_mp_udm.F:320-326)")
        fields[name] = physical[name]
    for name in ("re_cloud", "re_ice", "re_snow", "refl_10cm"):
        fields[name] = state.scratch(shape, "udm_" + name)
    save_pre_mp_theta(state)
    launch_driver(fields, scalar_arguments(dt, cfg.wdm6_ccn_conc), radar=refl_10cm_due)
    if refl_10cm_due:
        stash_refl_10cm(state, fields["refl_10cm"])
    moist_physics_finish(state, cfg, theta, dt)
    return MicrophysicsDiagnostics(**surface)
