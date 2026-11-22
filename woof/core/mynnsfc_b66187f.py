"""WRF v4.8.0 MYNN-SFC (b66187f5): ``sf_sfclay_physics=5`` under
``mynn_sfclay_variant = "mynnsfc_b66187f"``.

WRF v4.8.0 replaced the in-tree ``module_sf_mynn.F`` (the v4.6.1/v4.7.1
surface layer ``mynn_surface.cu`` ports) with the MYNN-SFC submodule at
b66187f5: separate land, water and ice modules behind ``mynnsfc_driver``.
That is the same commit woof hex already runs for RRFS-MPAS, so the column
code is the hex transcription with WRF's gravity (see the unit's header,
``woof/core/kernels/mynnsfc_b66187f.cu``).  This module is the WRF side:

* the two WRF call sites, ``module_surface_driver.F:2462-2486`` (one
  ``mynnsfc_driver`` call, ``fractional_seaice = 0``) and
  ``mynn_seaice_wrapper`` ``:5378-5711`` (land/frozen-water call,
  open-water call, ice-fraction blend, ``fractional_seaice = 1``), with each
  call's optional-argument presence;
* the WOOF-ArWen field mapping: the driver's 68 actual arguments are read in place
  through a device pointer table, so nothing is packed or copied per step.

WRF 4.8.0 behaviour this generation carries over the v4.6.1 one:

* the cold start lives in the driver (``:367-382``): UST reset where it is
  outside (1e-4, 4], PBLH = max(10, 700 UST), HFX = QFX = MOL = 0 at
  ``itimestep == 1``.  The v4.6.1 ``SFCLAY_mynn`` first-step seed
  (:func:`woof.core.mynn_sfclay.seed_mynn_surface_first_step`) is not run;
* REGIME is no longer an argument, so it is not written;
* the open-water call shares CH, CQS and PBLH with the first call
  (``:5622-5640``);
* ``sf_mynn_sfcflux_land`` / ``sf_mynn_sfcflux_water`` (Registry defaults
  0 and 1) choose the roughness forms; ``isftcflx``/``iz0tlnd`` are not read.

Refused option values, each with the breakage it prevents, are in
:func:`validate_mynnsfc_b66187f`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Actual arguments of ``m_drv_mynnsfc_driver``, in its order (the hex
#: ``sf_mynnsfc_v600.FIELDS``).  ``mynnsfc_wrf_columns`` reads ``p[k]`` as
#: the address of ``FIELDS[k]``.
FIELDS = (
    'u3d2', 'v3d2', 'dz8w2', 'u3d', 'v3d', 't3d', 'qv3d', 'p3d', 'dz8w',
    'th3d', 'rho3d', 'sigmaf', 'shdmax', 'z0pert', 'ztpert', 'psfcpa', 'chs',
    'chs2', 'cqs', 'cqs2', 'cpm', 'znt', 'ust', 'ustm', 'pblh', 'mavail',
    'zol', 'mol', 'rmol', 'psim', 'psih', 'xland', 'qgh', 'hfx', 'qfx', 'lh',
    'tsk', 'flhc', 'flqc', 'qsfc', 'u10', 'v10', 'th2', 't2', 'lakemask',
    'q2', 'snowh', 'gz1oz0', 'wspd', 'br', 'dx', 'ch', 'ck', 'cka', 'cd',
    'cda', 'stress', 'hflx', 'qflx', 'cm', 'fm', 'fh', 'fm10', 'fh2', 'tsurf',
    'water_depth', 'xice', 'pattern_spp_pbl')

#: The driver's optional-argument presence bits, in its has_* order.
PRESENCE = (
    'restart', 'cycling', 'initflag', 'sf_mynn_sfcflux_water',
    'sf_mynn_sfcflux_land', 'shalwater_z0', 'flag_lsm', 'spp_pbl', 'ivegsrc',
    'sfc_z0_type', 'redrag', 'flag_iter', 'vegtype', 'sigmaf', 'shdmax',
    'z0pert', 'ztpert', 'pattern_spp_pbl', 'tsurf', 'lakemask', 'water_depth',
    'cm', 'ck', 'cka', 'cd', 'cda', 'ustm', 'fm', 'fh', 'fm10', 'fh2', 'hflx',
    'qflx', 'stress')


def _mask(names) -> int:
    return sum(1 << PRESENCE.index(n) for n in names)


#: ``module_surface_driver.F:2462-2486`` and the wrapper's first call
#: ``:5553-5576`` pass these optionals.
WRF_PRESENT = _mask((
    'restart', 'sf_mynn_sfcflux_water', 'sf_mynn_sfcflux_land',
    'shalwater_z0', 'flag_lsm', 'spp_pbl', 'pattern_spp_pbl', 'lakemask',
    'water_depth', 'ck', 'cka', 'cd', 'cda', 'ustm'))
#: The wrapper's open-water call ``:5622-5640`` omits shalwater_z0,
#: water_depth and lakemask, so the driver takes lakemask_1 = 0 and
#: wat_depth_1 = 1000 there (``:420-429``).
WRF_SEA_PRESENT = WRF_PRESENT & ~_mask(('shalwater_z0', 'lakemask', 'water_depth'))

#: Optional arrays WRF never passes; their slots point at scratch.
ABSENT = ('sigmaf', 'shdmax', 'z0pert', 'ztpert', 'stress', 'hflx', 'qflx',
          'cm', 'fm', 'fh', 'fm10', 'fh2', 'tsurf')

#: ``mynn_seaice_wrapper`` ``*_HOLD`` set (``:5527-5545``): the open-water call
#: starts from these values as they were before the first call.
SEAICE_HOLD = ('br', 'cpm', 'flhc', 'flqc', 'gz1oz0', 'hfx', 'lh', 'mol',
               'psih', 'psim', 'qfx', 'qgh', 'rmol', 'ust', 'ustm', 'wspd',
               'znt', 'zol')

#: Fields blended at sea-ice cells after the open-water call (``:5644-5708``).
SEAICE_BLEND = ('br', 'gz1oz0', 'mol', 'psih', 'psim', 'rmol', 'ust', 'wspd',
                'zol', 'ch', 'cd', 'cda', 'ck', 'cka', 'q2', 't2', 'th2', 'u10',
                'ustm', 'v10')

#: The open-water call's own outputs (``*_sea`` actuals of ``:5622-5640``).
SEAICE_OWN = ('chs', 'chs2', 'cqs2', 'cpm', 'znt', 'ust', 'ustm', 'zol',
              'mol', 'psim', 'psih', 'qgh', 'hfx', 'qfx', 'lh', 'flhc', 'flqc',
              'qsfc', 'rmol', 'u10', 'v10', 'th2', 't2', 'q2', 'gz1oz0',
              'wspd', 'br', 'ck', 'cka', 'cd', 'cda')

#: WRF state carriers the open-water call leaves for the post-LSM blend
#: (the wrapper's ``INTENT(OUT)`` ``*_SEA`` dummies, ``:5470-5482``).
SEAICE_STATE = ('chs2', 'chs', 'cpm', 'cqs2', 'flhc', 'flqc', 'hfx', 'lh',
                'qfx', 'qgh', 'qsfc', 'znt')

#: The ``mynn_sfclay_variant`` each ``wrf_reference`` pin resolves to:
#: "4.7.1" is today's engine (WRF v4.6.1 module_sf_mynn.F, unchanged in
#: 4.7.1), "4.8.0" this module.
MYNN_SFCLAY_BY_WRF_REFERENCE = {"4.7.1": "wrf_461", "4.8.0": "mynnsfc_b66187f"}

LAND_OPTIONS = (0, 1, 3)
WATER_OPTIONS = (0, 1, 2, 3, 4, 5)

_TPB = 128
_KDIR = Path(__file__).resolve().parent / "kernels"
_MODULES: dict = {}


def validate_mynnsfc_b66187f(cfg) -> None:
    """Refuse option values with no defined WRF 4.8.0 answer.

    * ``sf_mynn_sfcflux_land = 2`` (Yang 2008) reads the driver's ``qstar``,
      a local automatic array (``WRF/module_sf_mynnsfc_driver.F90:410``) that
      WRF zeroes only at ``itimestep == 1``: every later step reads
      undefined memory.
    * ``sf_mynn_sfcflux_land = 4`` (GFS) reads ``ivegsrc`` and ``vegtype``
      (``module_sf_mynnsfc_land.F90:1060-1065``), which WRF's call
      (``module_surface_driver.F:2462``) never passes.
    * ``shalwater_z0 = 1`` reads the bathymetry ``water_depth``, which our
      init does not produce.
    """
    from woof.config import ConfigurationRefused
    land = getattr(cfg, "sf_mynn_sfcflux_land", 0)
    water = getattr(cfg, "sf_mynn_sfcflux_water", 1)
    shallow = getattr(cfg, "shalwater_z0", 0)
    if type(land) is not int or land not in LAND_OPTIONS:
        raise ConfigurationRefused(
            f"sf_mynn_sfcflux_land={land!r}: MYNN-SFC b66187f admits "
            f"{LAND_OPTIONS}. 2 (Yang 2008) reads WRF's driver-local qstar, "
            "undefined after the first step "
            "(WRF/module_sf_mynnsfc_driver.F90:410); 4 (GFS) reads ivegsrc "
            "and vegtype, which module_surface_driver.F:2462 never passes")
    if type(water) is not int or water not in WATER_OPTIONS:
        raise ConfigurationRefused(
            f"sf_mynn_sfcflux_water={water!r}: MYNN-SFC b66187f defines "
            f"{WATER_OPTIONS} (module_sf_mynnsfc_water.F90:429-441)")
    if shallow != 0:
        raise ConfigurationRefused(
            f"shalwater_z0={shallow!r}: the shallow-water roughness blend "
            "reads WRF's water_depth bathymetry static, which our init does "
            "not produce")


def module_source() -> str:
    """The exact text handed to NVRTC (headers, then the unit)."""
    k = _KDIR
    # ATAN is glibc 2.39's atanf (mynn_glibc_atanf), the word the referee's
    # libm returns.  woof hex compiles this text with glibc_atanf, the
    # CORE-MATH atanf of glibc 2.41+ its GNU 15.2 referee links; against
    # the GNU 13.3/glibc 2.39 referee that moves 76 of psi_init's 1001
    # unstable-momentum and 22 unstable-heat table words by an ULP.
    return ("typedef float real;\n#define SL_LOG mynn_glibc_logf\n"
            "#define SL_EXP mynn_glibc_expf\n#define SL_POW mynn_glibc_powf\n"
            "#define SL_ATAN mynn_glibc_atanf\n#define SL_SQRT __fsqrt_rn\n"
            + (k / "mynn_libm.cuh").read_text(encoding="utf-8")
            + (k / "mynnsfc_b66187f.cu").read_text(encoding="utf-8"))


#: Every operation in the unit is an explicitly rounded intrinsic and every
#: transcendental a glibc float32 word, so one image serves fast and strict.
OPTIONS = ("-std=c++17", "--ftz=false", "--prec-div=true", "--prec-sqrt=true",
           "--fmad=false")


def module():
    """The compiled unit for the current device, psi tables filled."""
    import cupy as cp
    dev = cp.cuda.runtime.getDevice()
    mod = _MODULES.get(dev)
    if mod is not None:
        return mod
    from woof import nvrtc_ptx_cache as compiler
    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile
    from woof.numerics import effective_options
    source = module_source()
    options = effective_options(OPTIONS)
    key = "woof.core.kernels:mynnsfc_b66187f"
    with observe_module_compile(key):
        binary, _ = compiler.compile_using_nvrtc(
            source, options, None, "mynnsfc_b66187f.cu")
        mod = cp.cuda.function.Module()
        mod.load(binary.encode() if isinstance(binary, str) else binary)
    record_module(key, source=source, options=options, module=None)
    error = cp.zeros(1, dtype=cp.int32)
    mod.get_function("mynnsfc_wrf_tables")((1,), (1,), (error,))
    if int(error.get()[0]) != 0:
        raise RuntimeError("MYNN-SFC psi_init reported an error")
    _MODULES[dev] = mod
    return mod


class MynnSfcB66187f:
    """Per-grid scratch and pointer tables for the WRF MYNN-SFC calls."""

    def __init__(self, shape, dx: float):
        import cupy as cp
        self.shape = tuple(shape)
        self.n = int(np.prod(self.shape))
        f32 = cp.float32
        self.scratch = cp.zeros(self.shape, dtype=f32)   # absent optionals
        self.dx = cp.full(self.shape, np.float32(dx), dtype=f32)  # dx2d = dx
        self.cqs = cp.zeros(self.shape, dtype=f32)       # surface_driver local
        self.water_depth = cp.zeros(self.shape, dtype=f32)
        self.ch_land = cp.empty(self.shape, dtype=f32)
        self.xland_sea = cp.empty(self.shape, dtype=f32)
        self.mavail_sea = cp.empty(self.shape, dtype=f32)
        self.tsk_sea_in = cp.empty(self.shape, dtype=f32)
        self.errors = cp.zeros(self.n, dtype=cp.int32)
        self._tables: dict = {}
        self._carrier: dict = {}

    def _table(self, arrays):
        import cupy as cp
        ptrs = []
        for a in arrays:
            if (a.shape != self.shape or a.dtype != cp.float32
                    or not a.flags.c_contiguous):
                raise ValueError(
                    "MYNN-SFC fields must be contiguous float32 on the "
                    f"surface grid {self.shape}; got {a.shape}/{a.dtype}")
            ptrs.append(int(a.data.ptr))
        key = tuple(ptrs)
        table = self._tables.get(key)
        if table is None:
            table = cp.asarray(np.asarray(ptrs, dtype=np.uint64))
            self._tables[key] = table
        return table

    def call(self, fields, *, present, xice_threshold, cfg, itimestep,
             flag_lsm):
        """One WRF ``mynnsfc_driver`` call; ``fields`` maps FIELDS names."""
        arrays = [fields[name] if name in fields else self.scratch
                  for name in FIELDS]
        table = self._table(arrays)
        spp = int(getattr(cfg, "spp_pbl", 0))
        module().get_function("mynnsfc_wrf_columns")(
            ((self.n + _TPB - 1) // _TPB,), (_TPB,),
            (table, self.errors, np.float32(xice_threshold),
             np.uint64(present), np.int32(self.n), np.int32(spp),
             np.int32(getattr(cfg, "sf_mynn_sfcflux_water", 1)),
             np.int32(getattr(cfg, "sf_mynn_sfcflux_land", 0)),
             np.int32(getattr(cfg, "shalwater_z0", 0)),
             np.int32(int(cfg.isfflx)), np.int32(0), np.int32(flag_lsm),
             np.int32(itimestep)))

    def seaice_setup(self, f, *, xice_threshold, tsk_local, sea):
        module().get_function("mynnsfc_wrf_seaice_setup")(
            ((self.n + _TPB - 1) // _TPB,), (_TPB,),
            (f["xice"], np.float32(xice_threshold), f["xland"], f["mavail"],
             tsk_local, f["qsfc"], f["tsk_sea"], self.xland_sea,
             self.mavail_sea, sea["znt"], self.tsk_sea_in, sea["qsfc"],
             np.int32(self.n)))

    def seaice_blend(self, grid, sea, *, xice, xice_threshold):
        import cupy as cp
        arrays = []
        for name in SEAICE_BLEND:
            arrays += [grid[name], sea[name]]
        key = tuple(int(a.data.ptr) for a in arrays)
        table = self._tables.get(key)
        if table is None:
            table = cp.asarray(np.asarray(key, dtype=np.uint64))
            self._tables[key] = table
        module().get_function("mynnsfc_wrf_seaice_blend")(
            ((self.n + _TPB - 1) // _TPB,), (_TPB,),
            (table, np.int32(len(SEAICE_BLEND)), xice,
             np.float32(xice_threshold), np.int32(self.n)))


def atmosphere_fields(atmosphere, f, *, tsk, pattern_spp_pbl=None,
                      runner: MynnSfcB66187f):
    """FIELDS names -> device arrays for one WRF call on the grid fields."""
    out = {
        'u3d': atmosphere["u"][0], 'v3d': atmosphere["v"][0],
        'u3d2': atmosphere["u"][1], 'v3d2': atmosphere["v"][1],
        't3d': atmosphere["temperature"][0], 'qv3d': atmosphere["qv"][0],
        'p3d': atmosphere["pressure"][0], 'dz8w': atmosphere["dz"][0],
        'dz8w2': atmosphere["dz"][1], 'th3d': atmosphere["theta"][0],
        'rho3d': atmosphere["rho"][0], 'psfcpa': atmosphere["p_interface"][0],
        'tsk': tsk, 'dx': runner.dx, 'cqs': runner.cqs,
        'water_depth': f.get("water_depth", runner.water_depth),
    }
    for name in ('chs', 'chs2', 'cqs2', 'cpm', 'znt', 'ust', 'ustm', 'pblh',
                 'mavail', 'zol', 'mol', 'rmol', 'psim', 'psih', 'xland',
                 'qgh', 'hfx', 'qfx', 'lh', 'flhc', 'flqc', 'qsfc', 'u10',
                 'v10', 'th2', 't2', 'lakemask', 'q2', 'snowh', 'gz1oz0',
                 'wspd', 'br', 'ch', 'ck', 'cka', 'cd', 'cda', 'xice'):
        out[name] = f[name]
    if pattern_spp_pbl is not None:
        out['pattern_spp_pbl'] = pattern_spp_pbl
    return out


def run_wrf_surface_layer(f, atmosphere, cfg, *, itimestep: int,
                          xice_threshold: float, tsk_local, runner,
                          fractional: bool, seam_carriers: bool,
                          pattern_spp_pbl=None) -> None:
    """WRF v4.8.0's MYNN-SFC call sites on WOOF-ArWen's grid fields, in place.

    ``fractional``: ``fractional_seaice == 1``, so WRF runs
    ``mynn_seaice_wrapper`` (``:5378-5711``) with ``tsk_local`` from
    ``get_local_ice_tsk``.  Otherwise WRF makes one ``mynnsfc_driver`` call on
    the grid TSK (``:2462-2486``).  ``seam_carriers``: a later seam (RUC's
    post-LSM ice blend) reads the ``*_sea`` carriers although WRF made no
    open-water call; they are filled by an open-water call on private copies
    of the shared CH/CQS/PBLH, so the grid fields stay WRF's single call.
    """
    import cupy as cp
    flag_lsm = int(cfg.sf_surface_physics)
    grid = atmosphere_fields(atmosphere, f,
                             tsk=(tsk_local if fractional else f["tsk"]),
                             pattern_spp_pbl=pattern_spp_pbl, runner=runner)
    if not (fractional or seam_carriers):
        runner.call(grid, present=WRF_PRESENT, xice_threshold=xice_threshold,
                    cfg=cfg, itimestep=itimestep, flag_lsm=flag_lsm)
        return
    sea = {name: f[f"{name}_sea"] for name in SEAICE_OWN}
    # *_HOLD (:5527-5545), restored into the open-water actuals (:5600-5616).
    for name in SEAICE_HOLD:
        sea[name][...] = f[name]
    if fractional:
        runner.call(grid, present=WRF_PRESENT,
                    xice_threshold=xice_threshold, cfg=cfg,
                    itimestep=itimestep, flag_lsm=flag_lsm)
        shared = {'ch': f["ch"], 'cqs': runner.cqs, 'pblh': f["pblh"]}
        runner.ch_land[...] = f["ch"]
    else:
        # WRF's grid answer: one call, from the same pre-call state.
        runner.call(grid, present=WRF_PRESENT,
                    xice_threshold=xice_threshold, cfg=cfg,
                    itimestep=itimestep, flag_lsm=flag_lsm)
        key = ('carrier',)
        private = runner._carrier.get(key)
        if private is None:
            private = {name: cp.empty(runner.shape, dtype=cp.float32)
                       for name in ('ch', 'cqs', 'pblh')}
            runner._carrier[key] = private
        private['ch'][...] = f["ch"]
        private['cqs'][...] = runner.cqs
        private['pblh'][...] = f["pblh"]
        shared = private
    runner.seaice_setup(f, xice_threshold=xice_threshold,
                        tsk_local=(tsk_local if fractional else f["tsk"]),
                        sea=sea)
    sea_fields = dict(grid)
    sea_fields.update(sea)
    sea_fields.update(shared)
    sea_fields.update({'xland': runner.xland_sea, 'mavail': runner.mavail_sea,
                       'tsk': runner.tsk_sea_in})
    runner.call(sea_fields, present=WRF_SEA_PRESENT,
                xice_threshold=xice_threshold, cfg=cfg, itimestep=itimestep,
                flag_lsm=flag_lsm)
    if not fractional:
        return
    blend_grid = {name: f[name] for name in SEAICE_BLEND}
    blend_grid['ch'] = runner.ch_land
    blend_sea = {name: sea[name] for name in SEAICE_BLEND if name != 'ch'}
    blend_sea['ch'] = f["ch"]
    runner.seaice_blend(blend_grid, blend_sea, xice=f["xice"],
                        xice_threshold=xice_threshold)
    # The blended CH lives in ch_land (the first term's buffer); WRF's CH
    # at non-ice cells is the open-water call's (ch=ch at :5631).
    ice = (f["xice"] >= np.float32(xice_threshold)) & (f["xice"] <= np.float32(1.0))
    f["ch"][...] = cp.where(ice, runner.ch_land, f["ch"])


__all__ = [
    "ABSENT", "FIELDS", "LAND_OPTIONS", "MYNN_SFCLAY_BY_WRF_REFERENCE", "PRESENCE", "SEAICE_BLEND",
    "SEAICE_HOLD", "SEAICE_OWN", "SEAICE_STATE", "WATER_OPTIONS",
    "WRF_PRESENT", "WRF_SEA_PRESENT", "MynnSfcB66187f", "atmosphere_fields",
    "module", "module_source", "run_wrf_surface_layer",
    "validate_mynnsfc_b66187f",
]
