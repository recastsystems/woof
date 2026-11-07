"""MPAS GPU sea-ice energy balance in the WOOF Noah-MP PBL cadence.

Noah-MP skips sea ice. Its sea-ice energy owner is the MPAS Noah sea-ice
solver, called after the land solver and before PBL and its energy ledger.
Persistent component surface-layer inputs live in driver.fields, which the
engine's restart manifest and the hex device rollback already serialize.
"""
from __future__ import annotations

import numpy as np

_PREFIX = "hex_seaice_"
_SEA = ("cpm", "chs", "chs2", "cqs2", "hfx", "lh", "qfx", "qgh", "qsfc")
_INOUT = ("znt", "ust", "ustm", "mol", "hfx", "qfx", "qsfc", "zol")


def install_noahmp_seaice(backend, seam):
    """Bind only this seam, without changing the shared engine class/module."""
    import cupy as cp

    driver = seam._driver
    if driver.noahmp_params is None:
        return
    retire_engine_seaice(seam)
    if int(seam._cfg.sf_sfclay_physics) not in (1, 91):
        raise ValueError("Noah-MP sea ice requires the routed MM5 surface layer (1 or 91)")
    from .ocean_pwp import select_ice_owner
    select_ice_owner(driver)
    f = driver.fields
    for name in dict.fromkeys((*_SEA, *_INOUT)):
        f[_PREFIX + name] = f[name].copy()
    f[_PREFIX + "sst"] = cp.maximum(f["tsk"], np.float32(271.4))
    f[_PREFIX + "tice"] = f["tsk"].copy()
    # The diagnostic accumulators are absent from the Noah-MP land-only block.
    for name in ("acsnow", "acsnom", "sfcrunoff"):
        if name not in f:
            f[name] = cp.zeros_like(f["tsk"])
    base = type(driver)

    class NoahmpSeaiceDriver(base):
        # No extra instance attributes: all carried component fields are in the
        # existing serialized field dictionary; closures contain setup only.
        # The class marker lets a coupled ocean prove it will be stepped.
        hex_seaice_hook = True

        def _run_sfclay(self, atmosphere, cfg):
            _surface_layer(self, atmosphere, cfg, backend, base._run_sfclay)

        def _run_noahmp(self, atmosphere, cfg, itimestep):
            # Noah-MP clears rainbl and pending rain on return. Preserve the
            # interval precipitation for the sea-ice solver before that clear.
            rain = self.fields["rainbl"] + self._pending_rainbl
            base._run_noahmp(self, atmosphere, cfg, itimestep)
            service = getattr(backend, "_surface_updates", None)
            if service is not None:
                # Snapshot the layout actually handed to Noah-MP, before ice
                # evolution or a later analysis record changes its categories.
                # These are existing host counts: no device read or sync.
                census = self.noahmp_params._surface_layout_cache[2]["census"]
                service.physics_classification = dict(service.classification,
                    sflx_land_columns=census["land"], open_water_columns=census["water"],
                    sea_ice_columns=census["sea_ice"], glacier_columns=census["glacier"])
            if service is not None and getattr(service, "ocean", None) is not None:
                service.ocean.step(self, atmosphere, self.bldt_seconds, rain)
            else:
                _energy_balance(self, atmosphere, cfg, rain)

    driver.__class__ = NoahmpSeaiceDriver


def retire_engine_seaice(seam):
    """Leave this seam's ocean sea ice to the hex owner installed here.

    THE BREAKAGE THIS PREVENTS: gpuwm 2.9.4 turned on its own prognostic
    zero-layer sea ice by default (RunConfig.seaice_prognostic, seeded by
    woof.ocean.seaice.initialize inside the seam's physics initialization),
    and its ice_step rewrites XICE after the land call. On the WOOF route the
    MPAS Noah sea-ice solver below (and, in subseasonal runs, the hex mixed
    layer ocean) already owns those columns, so two owners changed the ice
    cover and the next step's Noah-MP census disagreed with the classification
    the surface owner sealed: every 46-day s2s run on hex 0.4.2 + gpuwm 2.9.4
    stopped at its first composite step ("Arwen NoahMP census/provenance
    differs from sealed constructor authority", sea_ice 2078 vs 2157). The
    engine model is active exactly when its state is in the field dictionary
    (woof.ocean.seaice.step and the Noah sea-ice call test "ice_volume"), so
    removing that state, and the freezing-point floor its seed put on SST over
    its mask (SST was the skin-temperature copy), restores the single owner
    gpuwm 2.9.3 had. Runs before the first step, so no rollback or restart
    snapshot carries the engine state.
    """
    import dataclasses

    import cupy as cp

    f = seam._driver.fields
    if "ice_volume" not in f:
        return
    from woof.ocean.seaice import COLD_SEEDED

    mask = f["ice_ocean_mask"] > 0
    f["sst"][...] = cp.where(mask, f["tsk"], f["sst"])
    for name in COLD_SEEDED:
        f.pop(name, None)
    cfg = getattr(seam, "_cfg", None)
    if dataclasses.is_dataclass(cfg) and getattr(cfg, "seaice_prognostic", False):
        seam._cfg = dataclasses.replace(cfg, seaice_prognostic=False)


def _ice_mask(f, threshold):
    # This is fractional sea ice, not the permanent land-ice category.
    return (f["xice"] >= np.float32(threshold)) & (f["xice"] <= np.float32(1))


def _surface_layer(driver, a, cfg, backend, original):
    import cupy as cp
    from woof.core.sfclay import sfclay

    f = driver.fields
    ice = _ice_mask(f, driver.noahmp_params.xice_threshold)
    service = getattr(backend, "_surface_updates", None)
    if service is not None:
        f[_PREFIX + "sst"][...] = service.owned["sstsk"].reshape(f["tsk"].shape)
    sst = cp.maximum(f[_PREFIX + "sst"], np.float32(271.4))
    prognostic = service is not None and getattr(service, "ocean", None) is not None
    if prognostic:
        sst = service.owned["sstsk"].reshape(f["tsk"].shape)
    blended = f["tsk"].copy()
    fraction = cp.where(ice, f["xice"], np.float32(1))
    tice = cp.maximum((blended - (np.float32(1)-fraction)*sst) / fraction,
                      np.float32(221.4))
    tice = cp.where((fraction < np.float32(.2)) & (blended < np.float32(253.15)),
                   np.float32(253.15), tice)
    tice = cp.where((fraction < np.float32(.1)) & (blended < np.float32(263.15)),
                   np.float32(263.15), tice)
    if prognostic:
        tice = service.ocean.array("ice_temp").reshape(f["tsk"].shape)
    f[_PREFIX + "tice"][...] = cp.where(ice, tice, blended)
    f["tsk"][...] = f[_PREFIX + "tice"]
    original(driver, a, cfg)
    f["tsk"][...] = blended
    sea = sfclay(a["u"][0], a["v"][0], a["temperature"][0], a["qv"][0],
        a["pressure"][0], a["dz"][0], a["p_interface"][0], sst,
        f[_PREFIX+"znt"], f["pblh"], cp.ones_like(f["mavail"]),
        cp.full_like(f["xland"], 2),
        **{name: f[_PREFIX+name] for name in _INOUT if name != "znt"},
        lakemask=cp.zeros_like(f["lakemask"]), option=int(cfg.sf_sfclay_physics),
        dx=cfg.dx, isfflx=bool(cfg.isfflx), isftcflx=cfg.isftcflx,
        iz0tlnd=cfg.iz0tlnd)
    for name in dict.fromkeys((*_SEA, *_INOUT)):
        f[_PREFIX+name][...] = getattr(sea, name)
    # WRF fractional wrapper diagnostics are grid-cell values immediately;
    # the nine flux/exchange fields wait for the post-LSM sea-ice blend.
    for name in ("br", "gz1oz0", "mol", "psih", "psim", "rmol", "ust",
                 "wspd", "zol", "q2", "t2", "th2", "u10", "ustm", "v10",
                 "ck", "cka", "cd", "cda"):
        f[name][...] = cp.where(ice, fraction*f[name] +
            (np.float32(1)-fraction)*getattr(sea, name), f[name])


def _energy_balance(driver, a, cfg, rain):
    import cupy as cp
    from .mpasphys import seaice_v842 as si
    from woof.core.noahmp_runtime import _noahmp_post_lsm_diagnostics

    f = driver.fields
    ice = _ice_mask(f, driver.noahmp_params.xice_threshold)
    flat = lambda x: x.reshape(-1)
    n = f["tsk"].size
    f["tsk"][...] = cp.where(ice, f[_PREFIX+"tice"], f["tsk"])
    names = {"sfc_albedo": "albedo", "sfc_emiss": "emiss"}
    diag = {key: flat(f[names.get(key, key)]) for key in si._DIAG_FIELDS if key != "gsw"}
    diag["gsw"] = flat(f["swdown"] * (np.float32(1)-f["albedo"]))
    diag["sr"] = flat(f["sr"])
    diag["rainncv"] = flat(rain)
    sfc = {key: flat(f[{"skintemp": "tsk"}.get(key,key)])
           for key in si._SFC_FIELDS if key != "tslb"}
    sfc["tslb"] = f["tslb"].reshape(f["tslb"].shape[0], n)
    shared = {key: a[src].reshape(a[src].shape[0], n)
              for key,src in (("t3d","temperature"),("qv3d","qv"),
                              ("p8w3d","p_interface"),("dz8w","dz"))}
    shared.update({name+"_sea": flat(f[_PREFIX+name]) for name in _SEA})
    from dataclasses import replace
    options = replace(si.options_from_MPAS(driver.bldt_seconds, config_frac_seaice=True),
                      xice_threshold=float(driver.noahmp_params.xice_threshold))
    si.driver_seaice(diag, sfc, shared, options,
        lsm_scheme="sf_noahmp", microp_scheme="on", convection_scheme="off",
        sea_temperature=flat(f[_PREFIX+"sst"]))
    # Noah-MP's earlier diagnostic used the pre-energy endpoint. Republish
    # ice T2/Q2/TH2 from the new blended fluxes; retain every non-ice word.
    saved = {name: f[name].copy() for name in ("t2", "th2", "q2")}
    _noahmp_post_lsm_diagnostics(f, params=driver.noahmp_params)
    for name, old in saved.items():
        f[name][...] = cp.where(ice, f[name], old)
