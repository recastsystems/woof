"""The whole Noah-MP column, assembled over the column axis.

Every piece of NOAHMP_SFLX except this file already exists in slab form and is
gated bitwise on its own: the eight leaf batches
(:mod:`woof.core.noahmp_sflx_pre_slab`, :mod:`woof.core.noahmp_thermal_slab`,
:mod:`woof.core.noahmp_flux_slab`, :mod:`woof.core.noahmp_water_slab`),
ENERGY's four composition segments (:mod:`woof.core.noahmp_energy_slab`) and
``sflx_post``'s two (:mod:`woof.core.noahmp_sflx_post_slab`).  This module is
the orchestration those conversions were built for: the straight-line launch
sequence that runs one whole land-surface call for ``n`` columns with **no
Python executed per column** -- the same statements
:func:`woof.core.noahmp_sflx.sflx_steps` drives one frame at a time, in the
same order, with the array spelling of each hand-off.

Nothing here computes physics.  Every value is produced by a slab module that
is itself gated against the scalar authority; what this file owns is the
*hand-offs* -- which array reaches which leaf, and when.  That is exactly the
defect class the device-column report has warned about for five sections (a
per-column value reaching the wrong slot survives every leaf's own oracle), so
``tests/test_noahmp_column_slab.py`` gates this file the only way that can
catch it: the assembled slab against :func:`sflx_steps` driven per column on
the host, every output field, bitwise.

The hand-offs that are easy to get wrong, written down
------------------------------------------------------
* **VEGE_FLUX runs on the vegetated subset only**, exactly as the scalar
  column only pauses there when ``tile_veg``. A device mask guards each
  full-width column before the unchanged physics leaf. Inactive outputs are
  positive zero, matching the former scatter. The
  five INOUT scalars (``tv tg->tgv cm->cmv ch->chv eah``, plus ``tah`` and
  ``qsfc``) fall back to their entry values on the bare lanes -- which is what
  the scalar path holds there.
* **BARE_FLUX's ``qsfc`` argument is the post-VEGE value** on vegetated
  columns: the scalar path assigns ``qsfc = vf.QSFC`` before BARE_FLUX reads
  it.  Handing BARE_FLUX the entry ``qsfc`` everywhere would be bitwise wrong
  on every vegetated column and invisible to both leaves' own oracles.
* **TSNOSOI reads the entry ``stc``; PHASECHANGE reads TSNOSOI's.**  The
  scalar generator rebinds ``stc`` between the two calls.
* **WATER reads ENERGY's exit column through** :func:`sflx_post_seg1`, which
  lands PHASECHANGE's arrays and copies them -- WRF's own aliasing, where
  NOAHMP_SFLX hands one set of arrays to both ENERGY and WATER.
* **ERROR reads the post-WATER column**: ``k_sflx_error``'s DZSNSO and the
  snow clamp's SNOWH/SNEQV are WATER's outputs, not ENERGY's.

Parameters arrive as per-column slabs
-------------------------------------
The scalar path reads parameters off one memoised ``SflxParameters`` handle
per (vegetation, soil) identity.  A slab cannot chase Python attributes per
column, so :func:`parameter_rows` flattens one handle into one row of named
values and the caller gathers rows into per-column arrays with one indexed
read (:func:`parameter_fields`).  The row layout is asserted against the
consuming slab modules' own slot tables where those exist, and the whole-column
gate is what holds the rest.

What this module refuses
------------------------
``n == 0`` answers an empty, well-formed bundle.  Everything the scalar path
refuses -- ``ICE != 0``, ``IST != 1``, a non-zero crop or irrigation state,
the three balance aborts, emitted longwave <= 0 -- is refused by the slab
modules themselves, for the slab, naming the first offending column.
"""

from __future__ import annotations

import numpy as np

from woof.core import noahmp_vegeflux_gpu as _vege
from woof.core.noahmp_energy import PINNED_OPTIONS
from woof.core.noahmp_energy_slab import (energy_seg1, energy_seg2,
                                           energy_seg3, energy_seg4)
from woof.core.noahmp_flux_slab import (evaluate_bare_flux_slab,
                                         evaluate_vege_flux_masked_slab)
from woof.core.noahmp_radiation_gpu import (BATS_PARAMETER_NAMES,
                                           PARAMETER_SLOTS as _RAD_PARAMETERS)
from woof.core.noahmp_sflx import NSNOW_DEFAULT, NSOIL_DEFAULT
from woof.core.noahmp_sflx_post_slab import sflx_post_seg1, sflx_post_seg2
from woof.core.noahmp_sflx_pre_slab import evaluate_sflx_pre_slab
from woof.core.noahmp_thermal_slab import (RADIATION_PARAMETER_WIDTH,
                                            evaluate_phasechange_slab,
                                            evaluate_radiation_slab,
                                            evaluate_thermoprop_slab,
                                            evaluate_tsnosoi_slab)
from woof.core.noahmp_water_slab import evaluate_water_slab

NSNOW = NSNOW_DEFAULT
NSOIL = NSOIL_DEFAULT
_NFULL = NSNOW + NSOIL
_ZERO = np.float32(0.0)

# ---------------------------------------------------------------------------
# the parameter rows: one SflxParameters handle, flattened by name
# ---------------------------------------------------------------------------

#: Integer per-identity values, and where each is read from the handle.
PARAMETER_INTS = ("nroot", "iswater", "isbarren", "isice", "urban_flag")

#: Float per-identity values that are read by name somewhere in the slab
#: composition, mapped to their per-column width.  The RADIATION widths are
#: imported from the slab module that consumes them rather than restated;
#: everything else is a scalar, a soil-layer vector or a monthly vector.
PARAMETER_WIDTHS: dict[str, int] = {
    # the SFLX prefix (PHENOLOGY, PRECIP_HEAT)
    "hvt": 1, "hvb": 1, "tmin": 1, "ch2op": 1, "laim": 12, "saim": 12,
    # ENERGY's own composition (seg1, seg2)
    "mfsno": 1, "scffac": 1, "z0sno": 1, "z0mvt": 1, "cwpvt": 1,
    "snow_emis": 1, "rsurf_exp": 1, "eg": 1,
    "smcmax": NSOIL, "smcref": NSOIL, "smcwlt": NSOIL,
    "psisat": NSOIL, "bexp": NSOIL,
    # THERMOPROP
    "quartz": NSOIL, "csoil": 1,
    # TSNOSOI
    "zbot": 1,
    # RADIATION
    **dict(RADIATION_PARAMETER_WIDTH),
    # VEGE_FLUX (kept in WRF's upper-case spelling, as the batch reads them)
    **{name: 1 for name in _vege.PARAMETER_NAMES},
    # WATER
    "dksat": NSOIL, "dwsat": NSOIL,
    "kdt": 1, "frzx": 1, "slope": 1, "ssi": 1, "snow_ret_fac": 1,
}

if tuple(name for name in _RAD_PARAMETERS if name not in PARAMETER_WIDTHS):
    raise AssertionError("a RADIATION parameter slot has no row entry")


def _selected_options(opt_alb, opt_stc, opt_rsf, opt_btr=1, opt_snf=1, dveg=4,
                      opt_inf=1, opt_run=3, opt_rad=3, opt_frz=1, opt_tbot=2, opt_crs=1, opt_sfc=1, opt_soil=1, opt_infdv=0, opt_tdrn=0, opt_irr=0, opt_irrm=0):
    """Validate domain-wide branch metadata without reading device fields.

    These are the implemented slab branches, not full forecast admission.
    Config/scalar/runtime guards remain until their native replay is complete.
    """
    for name, value, allowed in (("opt_alb", opt_alb, (1, 2, 3)),
                                 ("opt_stc", opt_stc, (1, 2, 3)),
                                 ("opt_rsf", opt_rsf, (1, 2, 3, 4)),
                                 ("opt_btr", opt_btr, (1, 2, 3)),
                                 ("opt_snf", opt_snf, (1, 2, 3, 4, 5)),
                                 ("dveg", dveg, (1, 2, 3, 4, 5, 6, 7, 8, 9)),
                                 ("opt_inf", opt_inf, (1, 2)),
                                 ("opt_infdv", opt_infdv, (0, 1, 2, 3)),
                                 ("opt_tdrn",opt_tdrn,(0,1,2)),
                                 ("opt_irr",opt_irr,(0,1,2,3)),
                                 ("opt_irrm",opt_irrm,(0,1,2,3)),
                                 ("opt_run", opt_run, (1, 2, 3, 4, 5, 6, 7, 8)),
                                 ("opt_rad", opt_rad, (1, 2, 3)),
                                 ("opt_frz", opt_frz, (1, 2)),
                                 ("opt_tbot", opt_tbot, (1, 2)),
                                 ("opt_crs", opt_crs, (1, 2)),
                                 ("opt_sfc", opt_sfc, (1, 2)),
                                 ("opt_soil", opt_soil, (1, 2, 3, 4))):
        if type(value) is not int or value not in allowed:
            raise ValueError(f"land slab {name} must be scalar int in {allowed}")


def slab_parameter_widths(*, opt_alb=2, opt_rsf=1, opt_run=3, opt_rad=3, opt_crs=1, opt_sfc=1, dveg=4, opt_tdrn=0, opt_irr=0, opt_crop=0) -> dict[str, int]:
    """Required parameter slabs for the explicitly selected land branches.

    CLASS/RSF1 retain the original resident inventory. BATS and RSF4 require
    their actual transferred values; no placeholder/default is inserted.
    """
    _selected_options(opt_alb, 1, opt_rsf, opt_run=opt_run, opt_rad=opt_rad, opt_crs=opt_crs, opt_sfc=opt_sfc, dveg=dveg,opt_tdrn=opt_tdrn,opt_irr=opt_irr)
    widths = dict(PARAMETER_WIDTHS)
    if opt_irr:
        from woof.core.noahmp_irrigation_slab import PARAMETERS
        widths.update(dict.fromkeys(PARAMETERS,1))
    if opt_tdrn:
        from woof.core.noahmp_tile_drain import PARAMETERS
        widths.update(dict.fromkeys(PARAMETERS,1))
    if opt_alb == 1:
        widths.update({name: 1 for name in BATS_PARAMETER_NAMES})
    if opt_rsf == 4:
        widths["rsurf_snow"] = 1
    if opt_run in (1,2): widths.update(fsatmx=1,timean=1)
    if opt_run==5:widths.update(fsatmx=1)
    if opt_run == 8: widths.update({name:1 for name in ("bbvic","bdvic","gdvic")})
    if opt_run == 6: widths["bvic"] = 1
    if opt_run == 7: widths.update({name: 1 for name in ("axaj", "bxaj", "xxaj")})
    if opt_rad == 1: widths["rc"] = 1
    if opt_crs == 2: widths.update({name:1 for name in ("RGL","RSMIN","RSMAX","TOPT","HS")})
    if opt_sfc == 2: widths["CZIL"] = 1
    if dveg in (2,5,6) or opt_crop:
        from woof.core.noahmp_carbon_slab import CARBON_PARAMETERS
        widths.update({name:1 for name in CARBON_PARAMETERS})
    return widths


def parameter_row(parameters) -> dict:
    """One ``SflxParameters`` handle as ``{name: numpy value}``.

    ``eg`` is ``EG(IST)`` with ``IST == 1`` -- the only surface type this port
    admits, refused everywhere else -- so the row carries the scalar the
    column would read, not the two-entry table.
    """
    p = parameters
    e = p.energy
    if e is None:
        raise TypeError("parameter_row needs a handle with .energy bound")
    row = {
        "nroot": int(p.nroot), "iswater": int(p.iswater),
        "isbarren": int(p.isbarren), "isice": int(p.isice),
        "urban_flag": int(bool(p.urban_flag)),
        "hvt": e.hvt, "hvb": p.hvb, "tmin": p.tmin, "ch2op": p.ch2op,
        "laim": np.asarray(p.laim, np.float32),
        "saim": np.asarray(p.saim, np.float32),
        "mfsno": e.mfsno, "scffac": e.scffac, "z0sno": e.z0sno,
        "z0mvt": e.z0mvt, "cwpvt": e.cwpvt, "snow_emis": e.snow_emis,
        "rsurf_exp": e.rsurf_exp, "rsurf_snow": e.rsurf_snow,
        "eg": np.float32(e.eg[0]),
        "smcmax": np.asarray(e.smcmax, np.float32),
        "smcref": np.asarray(e.smcref, np.float32),
        "smcwlt": np.asarray(e.smcwlt, np.float32),
        "psisat": np.asarray(e.psisat, np.float32),
        "bexp": np.asarray(e.bexp, np.float32),
        "quartz": np.asarray(e.quartz, np.float32), "csoil": e.csoil,
        "zbot": e.zbot,
        "tau0": e.tau0, "grain_growth": e.grain_growth,
        "extra_growth": e.extra_growth, "dirt_soot": e.dirt_soot,
        "swemx": e.swemx,
        "albsat": np.asarray(e.albsat, np.float32),
        "albdry": np.asarray(e.albdry, np.float32),
        "alblak": np.asarray(e.alblak, np.float32),
        "rhol": np.asarray(e.rhol, np.float32),
        "rhos": np.asarray(e.rhos, np.float32),
        "taul": np.asarray(e.taul, np.float32),
        "taus": np.asarray(e.taus, np.float32),
        "xl": e.xl, "omegas": np.asarray(e.omegas, np.float32),
        "betads": e.betads, "betais": e.betais,
        "dksat": np.asarray(p.dksat, np.float32),
        "dwsat": np.asarray(p.dwsat, np.float32),
        "kdt": p.kdt, "frzx": p.frzx, "slope": p.slope, "ssi": p.ssi,
        "snow_ret_fac": p.snow_ret_fac,
    }
    row.update({name: np.float32(getattr(e.vege, name))
                for name in _vege.PARAMETER_NAMES})
    if e.bats is not None:
        if len(e.bats) != len(BATS_PARAMETER_NAMES):
            raise ValueError("BATS needs all seven actual transferred parameters")
        row.update(zip(BATS_PARAMETER_NAMES, e.bats))
    return row


def parameter_fields(rows, index, *, opt_alb=2, opt_rsf=1, opt_run=3, opt_rad=3, opt_crs=1, opt_sfc=1, dveg=4, opt_tdrn=0, opt_irr=0, opt_crop=0) -> dict:
    """Per-column parameter slabs from per-identity rows and an index.

    ``rows`` is a sequence of :func:`parameter_row` results, one per distinct
    (vegetation, soil) identity; ``index`` maps each column to its row.  The
    gather is one indexed read per field over the whole slab, so the Python
    here is proportional to the number of parameter names and never to the
    number of columns.
    """
    if not rows:
        raise ValueError("parameter_fields needs at least one identity row")
    widths = slab_parameter_widths(opt_alb=opt_alb, opt_rsf=opt_rsf, opt_run=opt_run, opt_rad=opt_rad, opt_crs=opt_crs, opt_sfc=opt_sfc, dveg=dveg,opt_tdrn=opt_tdrn,opt_irr=opt_irr,opt_crop=opt_crop)
    for row in rows:
        missing = set(widths).union(PARAMETER_INTS).difference(row)
        if missing:
            raise ValueError("parameter_fields lacks actual transferred values: "
                             + ", ".join(sorted(missing)))
    import cupy as cp

    idx = cp.asarray(np.ascontiguousarray(index, dtype=np.int32))
    out = {}
    for name, width in widths.items():
        table = np.array([np.broadcast_to(
            np.asarray(row[name], np.float32), (width,)) for row in rows],
            dtype=np.float32)
        gathered = cp.asarray(table)[idx]
        out[name] = cp.ascontiguousarray(
            gathered[:, 0] if width == 1 else gathered)
    for name in PARAMETER_INTS:
        table = np.array([int(row[name]) for row in rows], dtype=np.int32)
        out[name] = cp.ascontiguousarray(cp.asarray(table)[idx])
    return out


# ---------------------------------------------------------------------------
# the composition
# ---------------------------------------------------------------------------

#: VEGE_FLUX's outputs under the names ENERGY's composition reads them by,
#: exactly the table ``tests/test_noahmp_energy_slab.py`` gates seg3 with.
#: The five INOUT names fall back to the column's entry value on bare lanes;
#: everything else falls back to the zeros :2033-2053 writes.
_VEGE_TO_ENERGY = {
    "tauxv": "TAUXV", "tauyv": "TAUYV", "irg": "IRG", "irc": "IRC",
    "shg": "SHG", "shc": "SHC", "evg": "EVG", "evc": "EVC", "tr": "TR",
    "ghv": "GH", "t2mv": "T2MV", "psnsun": "PSNSUN", "psnsha": "PSNSHA",
    "rssun": "RSSUN", "rssha": "RSSHA", "q2v": "Q2V",
    "tgv": "TG", "cmv": "CM", "chv": "CH", "eah": "EAH", "tv": "TV",
}
_VEGE_ENTRY_FALLBACK = {"tgv": "tg", "cmv": "cm", "chv": "ch",
                        "eah": "eah", "tv": "tv"}

_VEGE_SPREAD_NAMES = {**_VEGE_TO_ENERGY, "tah": "TAH", "canhs": "CANHS",
                      "qsfc": "QSFC"}
_VEGE_SPREAD_FALLBACK = {**_VEGE_ENTRY_FALLBACK, "tah": "tah", "qsfc": "qsfc"}
_VEGE_SPREAD_KERNELS = {}


def _vege_spread_kernels():
    """Two word-copy kernels, cached in the device context that owns them."""
    import cupy as cp
    from woof.core.noahmp_kernel_sources import compile_generated_slab_kernel

    device = cp.cuda.runtime.getDevice()
    if device not in _VEGE_SPREAD_KERNELS:
        defaults = tuple(_VEGE_SPREAD_FALLBACK)
        init_args = ", ".join(f"const unsigned int* d{k}"
                              for k in range(len(defaults)))
        init_lines, scatter_lines = [], []
        for k, name in enumerate(_VEGE_SPREAD_NAMES):
            value = f"d{defaults.index(name)}[c]" if name in defaults else "0u"
            init_lines.append(f"out[{k}LL*n+c] = {value};")
            scatter_lines.append(f"out[{k}LL*n+index[c]] = v{k}[c];")
        init_source = ('extern "C" __global__ void noahmp_vege_defaults('
                       f'unsigned int* out, int n, {init_args}) {{ '
                       'int c=blockDim.x*blockIdx.x+threadIdx.x; if(c>=n)return; '
                       + "\n".join(init_lines) + '}')
        scatter_args = ", ".join(f"const unsigned int* v{k}"
                                 for k in range(len(_VEGE_SPREAD_NAMES)))
        scatter_source = ('extern "C" __global__ void noahmp_vege_spread('
                          'unsigned int* out, const long long* index, '
                          f'int n, int count, {scatter_args}) {{ '
                          'int c=blockDim.x*blockIdx.x+threadIdx.x; '
                          'if(c>=count)return; '
                          + "\n".join(scatter_lines) + '}')
        _VEGE_SPREAD_KERNELS[device] = (
            compile_generated_slab_kernel(init_source, "noahmp_vege_defaults"),
            compile_generated_slab_kernel(scatter_source, "noahmp_vege_spread"))
    return _VEGE_SPREAD_KERNELS[device]


def _spread_vege_flux(vf, index, entry, n):
    """Expand the unique vegetated subset in two launches, without arithmetic.

    ``index`` comes from nonzero(tile_veg), so destinations are unique and
    in range. Initialize the bare-lane defaults first, then overwrite only
    vegetated lanes on the same stream. Owned contiguous rows remain safe to
    pass directly to raw kernels, just as the previous per-field copies were.
    """
    import cupy as cp

    out = cp.empty((len(_VEGE_SPREAD_NAMES), n), dtype=cp.float32)
    if n:
        initialize, scatter = _vege_spread_kernels()
        defaults = tuple(cp.ascontiguousarray(entry(source), dtype=cp.float32)
                         for source in _VEGE_SPREAD_FALLBACK.values())
        initialize(((n + 127) // 128,), (128,), (out, np.int32(n), *defaults))
        count = int(index.size)
        if count:
            index = cp.ascontiguousarray(index, dtype=cp.int64)
            values = tuple(cp.ascontiguousarray(vf[source], dtype=cp.float32)
                           for source in _VEGE_SPREAD_NAMES.values())
            scatter(((count + 127) // 128,), (128,),
                    (out, index, np.int32(n), np.int32(count), *values))
    return {name: out[k] for k, name in enumerate(_VEGE_SPREAD_NAMES)}

#: The names VEGE_FLUX reads straight off the ENERGY bundle -- the same value
#: under the same name on both sides of the seam.
_VEGE_CALL_THROUGH = (
    "dt", "lwdn", "uu", "vv", "sfctmp", "qair", "eair", "rhoair", "snowh",
    "fwet", "canliq", "canice", "igs", "foln", "co2air", "o2air", "sfcprs",
    "pahv", "pahg", "eah", "tah", "tv", "tg", "cm", "ch", "psfc", "fveg",
)

#: Everything :func:`evaluate_sflx_slab` returns, so a consumer that stops
#: receiving a field fails by name instead of silently writing stale state.
OUTPUT_NAMES = (
    # the SflxResult scalars the driver's write-back reads, same spelling
    "trad", "fsh", "ecan", "edir", "etran", "fcev", "fgev", "fctr", "ssoil",
    "runsrf", "runsub", "albedo", "fsno", "canliq", "canice", "fpice",
    "qmelt", "ponding", "ponding1", "ponding2", "emissi", "qsfc", "tv", "tg",
    "eah", "tah", "cm", "ch", "fwet", "sneqvo", "albold", "qsnow", "qrain",
    "wslake", "lai", "sai", "tauss", "z0wrf", "t2mv", "t2mb", "q2v", "q2b",
    "fveg", "fsa", "fira", "rssun", "rssha", "chv", "chb", "canhs", "qsnbot",
    "eflxb", "laisun", "laisha", "rb",
    # the mutated column state
    "isnow", "snowh", "sneqv", "snice", "snliq", "stc", "zsnso", "sh2o",
    "smc", "hcpct",
    # ERROR's residuals, which no WRF output carries and every gate wants
    "errwat", "end_wb", "errsw", "erreng", "beg_wb",
)


def _shaped(cp, fields, name, n, width):
    """One extra input (beyond the pre slab's own contract), shape-checked."""
    try:
        value = fields[name]
    except KeyError:
        raise KeyError(
            f"evaluate_sflx_slab: no {name!r} in fields; the slab carries the "
            "sflx_pre keyword set, the SnowColumn attributes, the per-column "
            "parameter values, wslake and ficeold") from None
    value = cp.asarray(value)
    want = (n,) if width == 1 else (n, width)
    if value.shape != want:
        raise ValueError(
            f"evaluate_sflx_slab: {name!r} must have shape {want}, got "
            f"{tuple(value.shape)}")
    return cp.ascontiguousarray(value, dtype=cp.float32)


def evaluate_sflx_slab(fields: dict, n: int, *, noahmp_snow_age="wrf_470",
                       opt_alb=2, opt_stc=1, opt_rsf=1, opt_btr=1, opt_snf=1, dveg=4,
                       opt_inf=1, opt_run=3, opt_rad=3, opt_frz=1, opt_tbot=2, opt_crs=1, opt_sfc=1, opt_soil=1, opt_infdv=0, opt_tdrn=0, opt_irr=0, opt_irrm=0, opt_crop=0, opt_gla=1,
                       soil_update_steps=1, calculate_soil=True, carry_soil_accumulators=False,
                       snicar_options=None, module_trace=None, tanh_profile="glibc_2_39", log10_profile="glibc_2_39",
                       atan_profile="glibc_2_39") -> dict:
    """One NOAHMP_SFLX land-surface call for ``n`` columns, on the device.

    ``fields`` carries, each as one CuPy (or uploadable) array over the column
    axis with the layer axis last:

    * everything :func:`woof.core.noahmp_sflx_pre_slab.evaluate_sflx_pre_slab`
      documents -- the ``sflx_pre`` keyword set, the ``SnowColumn`` attributes
      and the per-column parameter values;
    * the rest of :data:`PARAMETER_WIDTHS` and :data:`PARAMETER_INTS`, which
      :func:`parameter_fields` produces;
    * ``wslake`` ``(n,)`` and ``ficeold`` ``(n, NSNOW)``.

    Selected land branches ALB1/2, STC1/2/3, RSF1/2/3/4 and BTR1/2/3 are
    explicit host scalar metadata. ALB1 needs all seven BATS parameters, RSF4 needs
    ``rsurf_snow``; :func:`parameter_fields` prepares only the selected slabs.
    Glacier remains a separate entry point. Full forecast option admission
    and real captured replay of the new combination are separate work.

    Returns :data:`OUTPUT_NAMES`, each ``(n,)`` float32 unless it is column
    state (layer axis last) or ``isnow`` (int32).  Raises exactly where the
    scalar column raises: the glacier/lake refusals, the emitted-longwave
    check, and the three balance aborts, each naming the first offending
    column.
    """
    _selected_options(opt_alb, opt_stc, opt_rsf, opt_btr, opt_snf, dveg, opt_inf, opt_run, opt_rad, opt_frz, opt_tbot, opt_crs, opt_sfc, opt_soil, opt_infdv,opt_tdrn,opt_irr,opt_irrm)
    if type(soil_update_steps) is not int or soil_update_steps<1 or type(calculate_soil) is not bool:
        raise ValueError("soil cadence requires a positive native step count and logical calculate_soil")
    cadence_on=carry_soil_accumulators or soil_update_steps!=1 or not calculate_soil
    widths = slab_parameter_widths(opt_alb=opt_alb, opt_rsf=opt_rsf, opt_run=opt_run, opt_rad=opt_rad, opt_crs=opt_crs, opt_sfc=opt_sfc, dveg=dveg,opt_tdrn=opt_tdrn,opt_irr=opt_irr,opt_crop=opt_crop)
    import cupy as cp

    n = int(n)
    if n < 0:
        raise ValueError(f"evaluate_sflx_slab: n must not be negative: {n}")
    if n == 0:
        empty = cp.empty(0, dtype=cp.float32)
        out = {name: empty for name in OUTPUT_NAMES}
        out["isnow"] = cp.empty(0, dtype=cp.int32)
        for name, width in (("snice", NSNOW), ("snliq", NSNOW),
                            ("stc", _NFULL), ("zsnso", _NFULL),
                            ("sh2o", NSOIL), ("smc", NSOIL),
                            ("hcpct", _NFULL)):
            out[name] = cp.empty((0, width), dtype=cp.float32)
        return out

    def par(name):
        return _shaped(cp, fields, name, n, widths[name])

    # Reject an incomplete selected parameter inventory before launching any
    # of the column prefix. Do not substitute a CLASS table or snow constant.
    missing = set(widths).difference(fields)
    if missing:
        raise ValueError("evaluate_sflx_slab lacks selected parameter fields: "
                         + ", ".join(sorted(missing)))

    urban = cp.ascontiguousarray(cp.asarray(fields["urban_flag"]) != 0)

    # ---- NOAHMP_SFLX's first half, 806-946 --------------------------------
    pre_options = {}
    if opt_snf != 1: pre_options["opt_snf"] = opt_snf
    if dveg != 4: pre_options["dveg"] = dveg
    if opt_irr: pre_options.update(opt_irr=opt_irr,opt_irrm=opt_irrm,calculate_soil=calculate_soil)
    if opt_crop:pre_options["opt_crop"]=opt_crop
    pre = evaluate_sflx_pre_slab(fields, n, **pre_options)

    def c(name):
        return pre["call." + name]

    # ---- ENERGY's entry state, spelled the way the segments read it -------
    s = {
        "ice": c("ice"), "ist": c("ist"), "isnow": c("isnow"),
        "uu": c("uu"), "vv": c("vv"), "elai": c("elai"), "esai": c("esai"),
        "snowh": c("snowh"), "sneqv": c("sneqv"), "zref": c("zref"),
        "fveg": c("fveg"), "sfcprs": c("sfcprs"), "tv": c("tv"),
        "tg": c("tg"), "lwdn": c("lwdn"), "cosz": c("cosz"), "dt": c("dt"),
        "acc_ssoil": c("acc_ssoil"),
        "pahv": c("pahv"), "pahg": c("pahg"), "pahb": c("pahb"),
        "sh2o": c("sh2o"), "zsoil": c("zsoil"), "dzsnso": c("dzsnso"),
        "mfsno": par("mfsno"), "scffac": par("scffac"),
        "z0sno": par("z0sno"), "z0mvt": par("z0mvt"), "hvt": par("hvt"),
        "cwpvt": par("cwpvt"), "snow_emis": par("snow_emis"),
        "rsurf_exp": par("rsurf_exp"),
        "nroot": cp.ascontiguousarray(cp.asarray(fields["nroot"],
                                                 dtype=cp.int32)),
        "urban_flag": urban,
        "smcmax": par("smcmax"), "smcref": par("smcref"),
        "smcwlt": par("smcwlt"), "psisat": par("psisat"),
        "bexp": par("bexp"), "eg": par("eg"),
        "soil_update_steps": soil_update_steps,
    }
    if opt_rsf == 4:
        s["rsurf_snow"] = par("rsurf_snow")

    # ---- ENERGY seg1, then the two thermal-side leaves --------------------
    g = energy_seg1(s, tanh_profile=tanh_profile)

    thermo = evaluate_thermoprop_slab({
        "dzsnso": c("dzsnso"), "snice": c("snice"), "snliq": c("snliq"),
        "smc": c("smc"), "sh2o": c("sh2o"), "stc": c("stc"),
        "snowh": c("snowh"), "dt": c("dt"),
        "smcmax": s["smcmax"], "csoil": par("csoil"), "quartz": par("quartz"),
        "isnow": c("isnow"), "ist": c("ist"), "urban_flag": urban}, n,log10_profile=log10_profile)

    rad_fields = {name: par(name) for name in _RAD_PARAMETERS}
    if opt_rad == 1:
        rad_fields.update({name: par(name) for name in ("rc", "hvt", "hvb")})
    if opt_alb == 1:
        rad_fields.update({name: par(name) for name in BATS_PARAMETER_NAMES})
    rad_fields.update({
        "dt": c("dt"), "cosz": c("cosz"), "elai": c("elai"),
        "esai": c("esai"), "tg": c("tg"), "tv": c("tv"),
        "snowh": c("snowh"), "fsno": g["fsno"], "fwet": c("fwet"),
        "sneqvo": c("sneqvo"), "sneqv": c("sneqv"), "qsnow": c("qsnow"),
        "fveg": c("fveg"), "smc": c("smc"), "albold": c("albold"),
        "tauss": c("tauss"), "solad": c("solad"), "solai": c("solai"),
        "ist": c("ist")})
    if opt_alb == 3:
        rad_fields["snicar_layers"] = cp.stack((c("snice"), c("snliq"),
            c("dzsnso")[:, :NSNOW], c("stc")[:, :NSNOW]), axis=2)
        rad_fields["snicar_state"] = fields.get("snicar_state")
        rad_fields["snicar_options"] = snicar_options
        rad_fields["snicar_air_temperature"] = c("sfctmp")
    rad = evaluate_radiation_slab(rad_fields, n,
                                  noahmp_snow_age=noahmp_snow_age,
                                  opt_alb=opt_alb, opt_rad=opt_rad)

    # ---- ENERGY seg2, then the two flux leaves -----------------------------
    e = energy_seg2(s, g, opt_rsf=opt_rsf, opt_btr=opt_btr)
    tile_veg = e["tile_veg"]

    # VEGE_FLUX answers the vegetated subset, exactly as the scalar column
    # only pauses there under tile_veg. The device mask keeps shape fixed.

    vf_sources = {name: c(name) for name in _VEGE_CALL_THROUGH}
    vf_sources["qsfc"] = c("qsfc")
    vf_sources.update({name: rad[name]
                      for name in ("sav", "sag", "laisun", "laisha",
                                   "parsun", "parsha", "fsr")})
    vf_sources.update({name: g[name]
                      for name in ("ur", "vai", "cwp", "zlvl", "zpd",
                                   "z0m", "z0mg")})
    vf_sources.update({name: e[name]
                      for name in ("gammav", "gammag", "emv", "emg",
                                   "rsurf", "latheav", "btran", "rhsur")})
    vf_sources["latheag"] = e["latheag"]
    vf_sources.update({name: par(name)
                      for name in _vege.PARAMETER_NAMES})
    vf_sources.update({"isnow": c("isnow"),
                      "dzsnso": c("dzsnso"),
                      "stc": c("stc"),
                      "df": thermo["df"]})
    if opt_stc == 3 or opt_crs == 2 or opt_sfc == 2:
        vf_sources["fsno"] = g["fsno"]
    vf_sources["opt_stc"] = opt_stc
    if opt_crs == 2:
        vf_sources["opt_crs"] = opt_crs
        vf_sources.update({name:par(name) for name in ("RGL","RSMIN","RSMAX","TOPT","HS")})
    if opt_sfc == 2:
        vf_sources.update(opt_sfc=2, thair=c("thair"), CZIL=par("CZIL"))
    from woof.core.noahmp_surface_device import active_probe, column_mask
    gecros_state={}
    if opt_crop==2:
        from woof.core.noahmp_gecros_slab import STATE,evaluate
        from woof.core.noahmp_slab_libm import copy_slab_slots,split_slab_slots
        gs=cp.empty((n,60),cp.float32)
        copy_slab_slots(gs,[(k,fields[name])for k,name in enumerate(STATE)])
        vf=evaluate(vf_sources,n,active_probe(tile_veg),gs,{"julian":fields["julian"],"swdown":pre["swdown"],"prcp":pre["prcp"],"fb":pre["fb"],"croptype":fields["croptype"],"sh2o":c("sh2o"),"smcwlt":par("smcwlt")},atan_profile=atan_profile)
        gecros_state=split_slab_slots(gs,{name:(k,1)for k,name in enumerate(STATE)})
        # Native ENERGY receives GECROS radiation by reference before BARE_FLUX.
        for name,upper in (("sav","SAV"),("sag","SAG"),("fsr","FSR")):
            rad[name]=cp.where(tile_veg,vf[upper],rad[name])
        rad["fsa"]=rad["sav"]+rad["sag"]
    else:
        vf = evaluate_vege_flux_masked_slab(vf_sources, n, active_probe(tile_veg),
                                          atan_profile=atan_profile)

    v = {}
    for name, upper in _VEGE_TO_ENERGY.items():
        scattered = vf[upper]
        fallback = _VEGE_ENTRY_FALLBACK.get(name)
        if fallback is not None:
            scattered = cp.where(tile_veg, scattered, c(fallback))
        v[name] = scattered
    tah = cp.where(tile_veg, vf["TAH"], c("tah"))
    canhs = cp.where(tile_veg, vf["CANHS"], _ZERO)
    # :2262 -- the scalar path assigns qsfc = vf.QSFC before BARE_FLUX reads
    # it, so on a vegetated column BARE_FLUX's qsfc argument is VEGE_FLUX's.
    qsfc_after_vege = cp.where(tile_veg, vf["QSFC"], c("qsfc"))

    bf = evaluate_bare_flux_slab({
        "dt": c("dt"), "sag": rad["sag"], "lwdn": c("lwdn"), "ur": g["ur"],
        "uu": c("uu"), "vv": c("vv"), "sfctmp": c("sfctmp"),
        "thair": c("thair"), "qair": c("qair"), "eair": c("eair"),
        "rhoair": c("rhoair"), "snowh": c("snowh"), "zlvl": g["zlvl"],
        "zpd": g["zpdg"], "z0m": g["z0mg"], "fsno": g["fsno"],
        "emg": e["emg"], "rsurf": e["rsurf"], "lathea": e["latheag"],
        "gamma": e["gammag"], "rhsur": e["rhsur"], "q2": c("q2"),
        "pahb": c("pahb"), "dx": c("dx"), "dz8w": c("dz8w"), "qc": c("qc"),
        "psfc": c("psfc"), "sfcprs": c("sfcprs"),
        "tgb": c("tg"), "cm": c("cm"), "ch": c("ch"),
        "qsfc": qsfc_after_vege,
        "dzsnso": c("dzsnso"), "stc": c("stc"), "df": thermo["df"],
        "isnow": c("isnow"), "ivgtyp": c("vegtyp"),
        "iloc": np.int32(1), "jloc": np.int32(1),
        "urban_flag": urban, "opt_stc": opt_stc,
        "active": column_mask(),
        **({"opt_sfc": 2, "CZIL": par("CZIL")} if opt_sfc == 2 else {})}, n)
    b = dict(bf)
    b["cmb"] = b.pop("cm")
    b["chb"] = b.pop("ch")

    # ---- ENERGY seg3, the soil column, seg4 --------------------------------
    s3 = dict(s)
    s3.update({name: rad[name]
               for name in ("laisun", "laisha", "parsun", "parsha")})
    seg3 = energy_seg3(s3, g, e, v, b)

    thermal_fields = {
        "zsnso": c("zsnso"), "stc": c("stc"), "df": thermo["df"],
        "hcpct": thermo["hcpct"], "tbot": c("tbot"),
        "ssoil": seg3["ssoil_avg"], "dt": seg3["dt_soil"],
        "snowh": c("snowh"), "zbot": par("zbot"), "isnow": c("isnow")}
    if opt_alb == 3:
        thermal_fields["snicar_heating"] = rad["snicar_heating"]
    if calculate_soil:
        tsn = evaluate_tsnosoi_slab(thermal_fields, n,
            opt_stc=opt_stc, opt_tbot=opt_tbot)
    else:
        tsn={"stc":c("stc"),"eflxb":cp.zeros(n,dtype=cp.float32)}
    if opt_stc == 2 and n:
        from woof.core.noahmp_kernel_sources import selected_kernel
        selected_kernel("noahmp_thermal", "noahmp_energy_stc2_snow")(
            ((n+127)//128,), (128,),
            (cp.ascontiguousarray(c("snowh")), cp.ascontiguousarray(s["fveg"]),
             cp.ascontiguousarray(v["tv"]), cp.ascontiguousarray(tile_veg),
             seg3["tg"], seg3["ts"], seg3["tgv"], b["tgb"], np.int32(n)))

    pc = evaluate_phasechange_slab({
        "fact": thermo["fact"], "dzsnso": c("dzsnso"), "stc": tsn["stc"],
        "snice": c("snice"), "snliq": c("snliq"), "smc": c("smc"),
        "sh2o": c("sh2o"), "sneqv": c("sneqv"), "snowh": c("snowh"),
        "dt": c("dt"), "smcmax": s["smcmax"], "psisat": s["psisat"],
        "bexp": s["bexp"], "isnow": c("isnow"), "ist": c("ist")}, n, opt_frz=opt_frz)
    if opt_alb == 3:
        from woof.core.snow_papers import refreeze
        refreeze(rad["snicar_state"], c("snice"), pc["snice"], pc["snliq"])

    seg4 = energy_seg4(s, g, {**e, **seg3}, {
        "eflxb": tsn["eflxb"], "hcpct": thermo["hcpct"], "imelt": pc["imelt"],
        "snicev": thermo["snicev"], "snliqv": thermo["snliqv"],
        "epore": thermo["epore"], "fsrv": rad["fsrv"], "fsrg": rad["fsrg"],
        "bgap": rad["bgap"], "wgap": rad["wgap"]})

    # ---- NOAHMP_SFLX's second half, 979-1076 -------------------------------
    p1 = sflx_post_seg1({
        "stc": pc["stc"], "dzsnso": pre["dzsnso"], "snice": pc["snice"],
        "snliq": pc["snliq"], "sh2o": pc["sh2o"], "smc": pc["smc"],
        "snowh": pc["snowh"], "sneqv": pc["sneqv"],
        "fgev": seg3["fgev"], "latheag": e["latheag"]})

    # The native driver clears these accumulators every call at soiltstep=0.
    # Positive cadence carries their resident values through skipped calls.
    acc_zero = cp.zeros(n, dtype=cp.float32)
    acc_zero_soil = cp.zeros((n, NSOIL), dtype=cp.float32)
    def acc(name,width=1):
        return _shaped(cp,fields,name,n,width) if cadence_on else (acc_zero if width==1 else acc_zero_soil)
    water_options = {"soil_update_steps":soil_update_steps,"calculate_soil":calculate_soil} if cadence_on else {}
    if opt_inf != 1: water_options["opt_inf"] = opt_inf
    if opt_run != 3: water_options["opt_run"] = opt_run
    if opt_infdv: water_options["opt_infdv"] = opt_infdv
    if opt_tdrn: water_options["opt_tdrn"]=opt_tdrn
    if opt_irr: water_options["opt_irr"]=opt_irr
    water_parameters = {}
    if opt_irr:
        from woof.core.noahmp_irrigation_slab import PARAMETERS
        water_parameters.update({name:par(name) for name in PARAMETERS})
        water_parameters.update(smcref=par("smcref"),**{name:fields[name] for name in ("pltday","hsday","croplu")},**{name:pre[name] for name in ("mifac","fifac","iramtmi","iramtfi")})
    if opt_tdrn:
        from woof.core.noahmp_tile_drain import PARAMETERS
        water_parameters.update({name:par(name) for name in PARAMETERS})
        water_parameters.update(smcref=par("smcref"),**{name:_shaped(cp,fields,name,n,1) for name in ("zwt","tdfracmp","dx")})
    if opt_run in (1,2):
        water_parameters.update({name:par(name) for name in ("fsatmx","timean","psisat")})
        water_parameters.update({name:_shaped(cp,fields,name,n,1) for name in ("zwt","wa","wt")})
    if opt_run==5:
        water_parameters.update(fsatmx=par("fsatmx"),psisat=par("psisat"),smceq=_shaped(cp,fields,"smceq",n,NSOIL),**{name:_shaped(cp,fields,name,n,1)for name in ("zwt","wa","smcwtd","deeprech","rech")})
    if opt_run == 8: water_parameters.update({name:par(name) for name in ("bbvic","bdvic","gdvic")})
    if opt_run == 6: water_parameters["bvic"] = par("bvic")
    if opt_run == 7:
        water_parameters.update({name: par(name) for name in ("smcref", "axaj", "bxaj", "xxaj")})
    wat = evaluate_water_slab({
        **water_parameters,
        "smcmax": s["smcmax"], "smcwlt": s["smcwlt"], "bexp": s["bexp"],
        "dksat": par("dksat"), "dwsat": par("dwsat"),
        "kdt": par("kdt"), "frzx": par("frzx"), "slope": par("slope"),
        "ch2op": par("ch2op"), "ssi": par("ssi"),
        "snow_ret_fac": par("snow_ret_fac"),
        "snowh": p1["snowh"], "sneqv": p1["sneqv"], "snice": p1["snice"],
        "snliq": p1["snliq"], "stc": p1["stc"], "zsnso": c("zsnso"),
        "dzsnso": p1["dzsnso"], "sh2o": p1["sh2o"], "sice": p1["sice"],
        "dt": c("dt"), "fcev": seg3["fcev"], "fctr": seg3["fctr"],
        "elai": c("elai"), "esai": c("esai"), "fveg": c("fveg"),
        "bdfall": pre["bdfall"], "sfctmp": c("sfctmp"),
        "qvap": p1["qvap"], "qdew": p1["qdew"],
        "qsnow": c("qsnow"), "qrain": pre["qrain"],
        "snowhin": pre["snowhin"], "ponding": pc["ponding"],
        "canliq": c("canliq"), "canice": c("canice"), "tv": v["tv"],
        "wslake": _shaped(cp, fields, "wslake", n, 1),
        "acc_qinsur": acc("acc_qinsur"), "acc_qseva": acc("acc_qseva"),
        "zsoil": c("zsoil"), "btrani": seg4["btrani"],
        "acc_etrani": acc("acc_etrani",NSOIL), "smc": p1["smc"],
        "ficeold": _shaped(cp, fields, "ficeold", n, NSNOW),
        "isnow": c("isnow"), "urban_flag": urban, "nroot": s["nroot"],
        "ist": c("ist"), "frozen_canopy": e["frozen_canopy"],
        "frozen_ground": e["frozen_ground"],
        "imelt": seg4["imelt"][:, :NSNOW]}, n, **water_options)

    carbon = {}
    if dveg in (2,5,6):
        from woof.core.noahmp_carbon_slab import evaluate_carbon_slab,CARBON_PARAMETERS,CARBON_POOLS
        carbon=evaluate_carbon_slab({
            **{name:par(name) for name in (*CARBON_PARAMETERS,"tmin","smcmax")},
            **{name:fields[name] for name in ("vegtyp","nroot","iswater","isbarren","isice","urban_flag","foln",*CARBON_POOLS)},
            "dt":c("dt"),"tv":wat["tv"],"psn":seg3["psn"],"btran":e["btran"],
            "igs":pre["igs"],"lai":pre["lai"],"sai":pre["sai"],
            "smc":wat["smc"],"dzsnso":wat["dzsnso"],"stc":wat["stc"],"zsoil":c("zsoil")})

    if opt_crop==1:
        from woof.core.noahmp_crop_slab import evaluate_crop
        from woof.core.noahmp_carbon_slab import CARBON_POOLS
        carbon=evaluate_crop({**fields,"dt":c("dt"),"tv":wat["tv"],"psn":seg3["psn"],"btran":e["btran"],"lai":pre["lai"],"sai":pre["sai"],"smc":wat["smc"],"dzsnso":wat["dzsnso"],"stc":wat["stc"],"zsoil":c("zsoil"),"smcmax":s["smcmax"],"t2m":cp.where((pre["elai"]+pre["esai"]>np.float32(0))&(pre["fveg"]>np.float32(0)),pre["fveg"]*v["t2mv"]+(np.float32(1)-pre["fveg"])*b["t2mb"],b["t2mb"])},carbon)

    post_prcp=pre["prcp"];post_fsh=seg3["fsh"];irrigation_post={}
    if opt_irr:
        applied=(fields["croplu"]!=0)&(fields["irrfra"]>=par("irr_frac"))
        post_prcp=cp.where(applied,pre["prcp"]+(pre["irsirate"]*np.float32(1000.0)/c("dt")),pre["prcp"])
        post_fsh=cp.where(applied,seg3["fsh"]-pre["firr"],seg3["fsh"])
        irrigation_post={"firr":pre["firr"],"irmirate":wat["irmirate"],"irfirate":wat["irfirate"]}
    p2 = sflx_post_seg2({
        **irrigation_post,
        "edir": p1["edir"],
        "fsa": rad["fsa"], "fsr": rad["fsr"], "fira": seg3["fira"],
        "fsh": post_fsh, "fcev": seg3["fcev"], "fgev": seg3["fgev"],
        "fctr": seg3["fctr"], "ssoil": seg3["ssoil"], "sav": rad["sav"],
        "sag": rad["sag"], "pah": seg3["pah"], "canhs": canhs,
        "qsfc": b["qsfc"], "q2b": b["q2b"], "ch": seg3["ch"],
        "swdown": pre["swdown"], "beg_wb": pre["beg_wb"],
        "prcp": post_prcp, "rhoair": pre["rhoair"], "qair": pre["qair"],
        "canliq": wat["canliq"], "canice": wat["canice"],
        "ecan": wat["ecan"], "etran": wat["etran"], "runsrf": wat["runsrf"],
        "runsub": wat["runsub"], "qtldrn": wat["qtldrn"], "smc": wat["smc"],
        "sneqv": wat["sneqv"], "snowh": wat["snowh"],
        "dzsnso": wat["dzsnso"],
        "wa": wat["wa"] if opt_run in (1,2,5) else _shaped(cp, fields, "wa", n, 1), "dt": c("dt"),
        "acc_dwater": acc("acc_dwater"), "acc_prcp": acc("acc_prcp"),
        "acc_ecan": acc("acc_ecan"), "acc_etran": acc("acc_etran"), "acc_edir": acc("acc_edir"),
        **({"calculate_soil":cp.full(n,calculate_soil,dtype=cp.int32)} if cadence_on else {}),
        "ist": c("ist"), "urban_flag": urban}, carbon=carbon)

    # The native module oracle also observes diagnostics which the WRF
    # driver does not carry. Keep their existing device results available to
    # that opt-in observer without adding any forecast arithmetic or copies.
    if module_trace is not None:
        module_trace.update(pre=pre, geometry=g, energy=e, radiation=rad,
                            vegetation_raw=vf,
                            vegetation=v, bare=b, energy3=seg3, energy4=seg4,
                            phasechange=pc, water=wat, post1=p1, post2=p2, carbon=carbon)

    output_terms={"sag":rad["sag"],"sav":rad["sav"],"irg":v["irg"],"shg":v["shg"],"evg":v["evg"],"ghv":v["ghv"],"irc":v["irc"],"shc":v["shc"],"evc":v["evc"],"tr":v["tr"],"irb":b["irb"],"shb":b["shb"],"evb":b["evb"],"ghb":b["ghb"],"pahg":pre["pahg"],"pahb":pre["pahb"],"pahv":pre["pahv"],"qints":pre["qints"],"qintr":pre["qintr"],"qdrips":pre["qdrips"],"qdripr":pre["qdripr"],"qthros":pre["qthros"],"qthror":pre["qthror"],"qsnsub":wat["qsnsub"],"qsnfro":wat["qsnfro"],"qsubc":wat["qsubc"],"qfroc":wat["qfroc"],"qevac":wat["qevac"],"qdewc":wat["qdewc"],"qfrzc":wat["qfrzc"],"qmeltc":wat["qmeltc"],"rainin":pre["rain"],"snowin":pre["snow"]}
    # ---- the answer, under the SflxResult names ----------------------------
    return {
        **output_terms,
        **gecros_state,
        **({**{name:pre[name] for name in ("ircntsi","ircntmi","ircntfi","iramtsi","irsirate","firr","eirr")},**{name:wat[name] for name in ("iramtmi","iramtfi","irmirate","irfirate")}} if opt_irr else {}),
        **({"acc_ssoil":seg3["acc_ssoil"],**{name:wat[name] for name in ("acc_qinsur","acc_qseva","acc_etrani")},**{name:p2[name] for name in ("acc_dwater","acc_prcp","acc_ecan","acc_etran","acc_edir")}} if cadence_on else {}),
        **carbon,
        **({"zwt":wat["zwt"],"qtldrn":wat["qtldrn"]} if opt_tdrn else {}),
        **({name:wat[name] for name in ("wa","wt","zwt")} if opt_run in (1,2) else {}),
        **({name:wat[name]for name in ("zwt","wa","smcwtd","deeprech","rech")}if opt_run==5 else {}),
        "trad": seg3["trad"], "fsh": p2["fsh"], "ecan": wat["ecan"],
        "edir": p1["edir"], "etran": wat["etran"], "fcev": seg3["fcev"],
        "fgev": seg3["fgev"], "fctr": seg3["fctr"], "ssoil": seg3["ssoil"],
        "runsrf": wat["runsrf"], "runsub": wat["runsub"],
        "soldrain": wat["soldrain"],
        "albedo": p2["albedo"], "fsno": g["fsno"], "canliq": wat["canliq"],
        "canice": wat["canice"], "fpice": pre["fpice"], "qmelt": pc["qmelt"],
        "ponding": pc["ponding"], "ponding1": wat["ponding1"],
        "ponding2": wat["ponding2"], "emissi": seg3["emissi"],
        "qsfc": p2["qsfc"], "tv": wat["tv"], "tg": seg3["tg"],
        "eah": v["eah"], "tah": tah, "cm": seg3["cm"], "ch": seg3["ch"],
        "fwet": wat["fwet"], "sneqvo": p1["sneqvo"], "albold": rad["albold"],
        "qsnow": c("qsnow"), "qrain": pre["qrain"], "wslake": wat["wslake"],
        "lai": carbon.get("lai",pre["lai"]), "sai": carbon.get("sai",pre["sai"]), "tauss": rad["tauss"],
        "z0wrf": seg3["z0wrf"], "t2mv": v["t2mv"], "t2mb": b["t2mb"],
        "q2v": v["q2v"], "q2b": p2["q2b"], "fveg": pre["fveg"],
        "fsa": rad["fsa"], "fira": seg3["fira"], "rssun": seg3["rssun"],
        "rssha": seg3["rssha"], "chv": seg3["chv"], "chb": b["chb"],
        "canhs": canhs, "qsnbot": wat["qsnbot"], "eflxb": seg4["eflxb"],
        "laisun": rad["laisun"], "laisha": rad["laisha"],
        "rb": cp.zeros(n, dtype=cp.float32),
        "isnow": wat["isnow"], "snowh": p2["snowh"], "sneqv": p2["sneqv"],
        "snice": wat["snice"], "snliq": wat["snliq"], "stc": wat["stc"],
        "zsnso": wat["zsnso"], "sh2o": wat["sh2o"], "smc": wat["smc"],
        "hcpct": seg4["hcpct"],
        "errwat": p2["errwat"], "end_wb": p2["end_wb"],
        "errsw": p2["errsw"], "erreng": p2["erreng"], "beg_wb": pre["beg_wb"],
        **({"snicar_state": rad["snicar_state"]} if opt_alb == 3 else {}),
    }


__all__ = [
    "OUTPUT_NAMES",
    "PARAMETER_INTS",
    "PARAMETER_WIDTHS",
    "evaluate_sflx_slab",
    "parameter_fields",
    "parameter_row",
    "slab_parameter_widths",
]
