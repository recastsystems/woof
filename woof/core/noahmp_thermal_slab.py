"""Whole-slab CuPy packing for four Noah-MP leaves: the thermal three and
RADIATION.

:mod:`woof.core.noahmp_thermal_gpu` and
:mod:`woof.core.noahmp_radiation_gpu` already answer every land column in one
launch per leaf, but they get there through a Python loop: one bound argument
dict per column, one row of slice assignments per column, and one tuple or
dict rebuilt per column on the way back.  At d04 that loop *is* the cost --
the launch is not.  This module is the same packing with the column axis
vectorised: the caller hands over whole-slab CuPy arrays and the Python that
runs is proportional to the number of *fields*, not to the number of columns.

Nothing here computes physics.  The four kernels are reached through exactly
the same module and symbol the per-column wrappers use --
``noahmp_leaf_thermoprop``, ``noahmp_thermal_tsnosoi``,
``noahmp_thermal_phasechange`` from
:func:`woof.core.noahmp_thermal_gpu.thermal_module`, and
``noahmp_radiation_batch`` through :func:`woof.core.kernels.get_kernel` --
with the same block geometry.  What this module adds is assignment, and that
is exactly what ``tests/test_noahmp_thermal_slab.py`` compares bit for bit
against the per-column packers.

Why the layouts are restated here
---------------------------------
:mod:`woof.core.noahmp_water_slab` can *import* its slot dictionaries because
``noahmp_water_gpu`` publishes them.  The two modules this one shadows do not:
their slot order lives in a running cursor inside ``pack_thermoprop_calls``,
``pack_tsnosoi_calls``, ``pack_phasechange_calls`` and
``pack_radiation_calls``.  So the layouts below are a second statement of the
same order, written with the same cursor arithmetic and the same
``NLAY``/``NSNOW``/``NSOIL``/``PARAMETER_SLOTS``/``STATE_SLOTS`` the originals
use, and the byte-exact packing gate is what holds the two statements
together.  A restated layout that is *not* gated would be the worst of both
worlds; this one is gated on every slot of every leaf.

Gaps
----
Three of the four layouts contain slots the per-column packer deliberately
leaves at zero, and they are spelled ``(None, width)`` here:

* THERMOPROP's five inert slots (TG, UR, LAT, Z0M, ZLVL), which the kernel
  does not read;
* TSNOSOI's DZSNSO block and its SAG and TG slots.  Note that
  ``pack_tsnosoi_calls``'s docstring says "DZSNSO is packed because the caller
  has it" -- the code does not pack it, and this module reproduces the *code*,
  because the code is what the pinned answers were measured from;
* PHASECHANGE's HCPCT block, which the routine never references.

The ``fields`` mapping
----------------------
Keys are the keyword names the per-column packers read.  A per-column scalar
has shape ``(n,)``; a per-column vector has shape ``(n, width)`` with the
**layer axis last**, in the same Fortran order the per-column packer writes
into its row.  Widths are checked, because a ``(n, NSOIL)`` array silently
broadcast into an ``(n, NLAY)`` destination is a wrong forecast, not an
exception.  The width check also subsumes the per-column packers' ``nsnow``
and ``nsoil`` refusals: a slab of three-snow-layer columns cannot be handed to
a four-snow-layer destination without the shape check firing.

For RADIATION every entry of ``PARAMETER_SLOTS`` becomes its own per-column
array rather than a memoised row shared by every column with the same
parameter handle.  That memoisation (``_PARAMETER_CACHE``) exists only because
the per-column path pays a Python frame per column; a slab pays one assignment
per parameter for the whole nest, so there is nothing left to memoise.

Contiguity
----------
A CuPy raw-kernel argument carries a device pointer and nothing else -- no
shape, no strides.  Handing a kernel a strided view therefore does not fail;
it reads the wrong words.  Two rules follow, and both are tested:

* every array passed to a launch is ``cupy.ascontiguousarray``;
* every array *returned* is ``cupy.ascontiguousarray`` too, because the
  natural spelling of an output column, ``out[:, slot]``, is a strided view
  and the caller's next step is usually another raw kernel.
"""

from __future__ import annotations

from woof.core.noahmp_slab_libm import copy_slab_slots, split_slab_slots

import numpy as np

from woof.core.noahmp_radiation_gpu import (
    BATS_PARAMETER_NAMES,
    N_INPUT as RADIATION_N_IN,
    N_OUTPUT as RADIATION_N_OUT,
    OUTPUT_NAMES as RADIATION_OUTPUT_NAMES,
    PARAMETER_SLOTS,
    STATE_SLOTS,
    THREADS as RADIATION_THREADS,
    _UNDEFINED_ENTRY,
)
from woof.core.noahmp_thermal_gpu import (
    NLAY,
    NSNOW,
    NSOIL,
    PHASECHANGE_N_IN,
    PHASECHANGE_N_OUT,
    THERMOPROP_N_IN,
    THERMOPROP_N_OUT,
    TSNOSOI_N_IN,
    TSNOSOI_N_OUT,
    _THREADS as THERMAL_THREADS,
    _thermoprop_kernel,
    thermal_module,
)


def _layout(entries):
    """``({name: (start, width)}, stride)`` from ``(name, width)`` in order.

    A ``None`` name is a gap: slots the per-column packer allocates and leaves
    at zero.  Naming them costs one line and makes a future reader ask what is
    in them, which is the question that matters at a slot boundary.
    """
    spec: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, width in entries:
        if name is not None:
            if name in spec:
                raise ValueError(f"duplicate slab field {name!r}")
            spec[name] = (cursor, width)
        cursor += width
    return spec, cursor


def _checked(entries, stride: int, label: str):
    """A layout, refused unless it fills exactly the row the kernel reads."""
    spec, total = _layout(entries)
    if total != stride:
        raise AssertionError(
            f"{label} slab layout covers {total} slots, but the per-column "
            f"packer's row is {stride} wide")
    return spec


# --------------------------------------------------------------------------
# THERMOPROP -- pack_thermoprop_calls' 44 float slots and 4 integer slots
# --------------------------------------------------------------------------
THERMOPROP_IN = _checked((
    ("dzsnso", NLAY), ("snice", NSNOW), ("snliq", NSNOW),
    ("smc", NSOIL), ("sh2o", NSOIL), ("stc", NLAY),
    ("snowh", 1), ("dt", 1),
    (None, 5),                       # TG, UR, LAT, Z0M, ZLVL: declared, unread
    ("smcmax", NSOIL), ("csoil", 1), ("quartz", NSOIL),
), THERMOPROP_N_IN, "THERMOPROP")

THERMOPROP_OUT = _checked((
    ("df", NLAY), ("hcpct", NLAY), ("snicev", NSNOW), ("snliqv", NSNOW),
    ("epore", NSNOW), ("fact", NLAY),
), THERMOPROP_N_OUT, "THERMOPROP output")

#: ``ix`` slot 2 is VEGTYP, which ``pack_thermoprop_calls`` writes as zero.
THERMOPROP_INT = {"isnow": 0, "ist": 1}
THERMOPROP_FLAG = {"urban_flag": 3}
THERMOPROP_INT_STRIDE = 4


# --------------------------------------------------------------------------
# TSNOSOI -- pack_tsnosoi_calls' 42 float slots and 5 integer slots
# --------------------------------------------------------------------------
TSNOSOI_IN = _checked((
    ("zsnso", NLAY), ("stc", NLAY), ("df", NLAY), ("hcpct", NLAY),
    (None, NLAY),                    # DZSNSO: read only after :5346's RETURN
    ("tbot", 1), ("ssoil", 1),
    (None, 1),                       # SAG: not passed to this leaf by ENERGY
    ("dt", 1), ("snowh", 1),
    (None, 1),                       # TG: likewise
    ("zbot", 1),
), TSNOSOI_N_IN, "TSNOSOI")

TSNOSOI_OUT = _checked((
    ("stc", NLAY), ("eflxb", 1),
), TSNOSOI_N_OUT, "TSNOSOI output")

TSNOSOI_INT = {"isnow": 0}
TSNOSOI_INT_STRIDE = 5


# --------------------------------------------------------------------------
# PHASECHANGE -- pack_phasechange_calls' 57 float slots, 4 integer slots
# --------------------------------------------------------------------------
PHASECHANGE_IN = _checked((
    ("fact", NLAY), ("dzsnso", NLAY),
    (None, NLAY),                    # HCPCT: never referenced (:5617)
    ("stc", NLAY), ("snice", NSNOW), ("snliq", NSNOW),
    ("smc", NSOIL), ("sh2o", NSOIL),
    ("sneqv", 1), ("snowh", 1), ("dt", 1),
    ("smcmax", NSOIL), ("psisat", NSOIL), ("bexp", NSOIL),
), PHASECHANGE_N_IN, "PHASECHANGE")

#: ``imelt`` is the one integer output; it lives in the float row and the
#: per-column unpacker casts it, so the slab casts the same block.
PHASECHANGE_OUT = _checked((
    ("stc", NLAY), ("snice", NSNOW), ("snliq", NSNOW),
    ("smc", NSOIL), ("sh2o", NSOIL),
    ("sneqv", 1), ("snowh", 1), ("qmelt", 1), ("ponding", 1),
    ("imelt", NLAY),
), PHASECHANGE_N_OUT, "PHASECHANGE output")

PHASECHANGE_INT_OUT = ("imelt",)
PHASECHANGE_INT = {"isnow": 0, "ist": 1}
PHASECHANGE_INT_STRIDE = 4


# --------------------------------------------------------------------------
# RADIATION -- pack_radiation_calls' 56 float slots and the IST vector
# --------------------------------------------------------------------------
#: Per-column width of every ``PARAMETER_SLOTS`` entry, as ``_parameter_row``
#: writes it.  The *order* is imported; only the widths are stated here.
RADIATION_PARAMETER_WIDTH = {
    "tau0": 1, "grain_growth": 1, "extra_growth": 1, "dirt_soot": 1,
    "swemx": 1, "albsat": 2, "albdry": 2, "alblak": 2, "rhol": 2, "rhos": 2,
    "taul": 2, "taus": 2, "xl": 1, "omegas": 2, "betads": 1, "betais": 1,
}
if tuple(RADIATION_PARAMETER_WIDTH) != tuple(PARAMETER_SLOTS):
    raise AssertionError(
        "the RADIATION parameter block moved: "
        f"{tuple(RADIATION_PARAMETER_WIDTH)} vs {tuple(PARAMETER_SLOTS)}")

#: FAGE, FREVD/I and FREGD/I -- nine slots WRF leaves uninitialised at
#: ``COSZ <= 0`` and both transcriptions pass as a defined constant.
RADIATION_UNDEFINED = 9

RADIATION_IN = _checked(
    tuple((name, RADIATION_PARAMETER_WIDTH[name]) for name in PARAMETER_SLOTS)
    + tuple((name, 1) for name in STATE_SLOTS)
    + (("smc", NSOIL), ("albold", 1), ("tauss", 1),
       (None, RADIATION_UNDEFINED),
       ("solad", 2), ("solai", 2)),
    RADIATION_N_IN, "RADIATION")

#: The gap is the only one of the four that is filled with a named constant
#: rather than left at the allocation's zero, so its start is derived from the
#: layout either side of it rather than written down twice.
RADIATION_UNDEFINED_START = RADIATION_IN["tauss"][0] + 1
if RADIATION_UNDEFINED_START + RADIATION_UNDEFINED != RADIATION_IN["solad"][0]:
    raise AssertionError(
        "the RADIATION undefined-entry block no longer sits between TAUSS and "
        "SOLAD")

RADIATION_OUT = _checked(
    tuple((name, 1) for name in RADIATION_OUTPUT_NAMES)
    + (("albsnd", 2), ("albsni", 2),
       ("bgap", 1), ("wgap", 1), ("albold", 1), ("tauss", 1)),
    RADIATION_N_OUT, "RADIATION output")


# --------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------

def _take(fields, name: str, width: int, n: int, leaf: str, cp):
    """The slab for one field, shape-checked and on the device."""
    try:
        value = fields[name]
    except KeyError:
        raise KeyError(f"{leaf} slab is missing {name!r}") from None
    array = cp.asarray(value)
    want = (n,) if width == 1 else (n, width)
    if array.shape != want:
        raise ValueError(
            f"{leaf} slab field {name!r} must have shape {want}, got "
            f"{tuple(array.shape)}")
    return array


def _fill(flat, fields, spec, n: int, leaf: str, cp) -> None:
    """Copy validated fields together to reduce driver submissions."""
    _slots_flat = []
    for name, (start, width) in spec.items():
        block = _take(fields, name, width, n, leaf, cp)
        _slots_flat.append((start, block))
    copy_slab_slots(flat, _slots_flat)


def _integers(fields, spec, stride: int, n: int, leaf: str, cp):
    """The integer row.

    ``astype`` rather than a bare assignment so that a float slab lands as the
    truncated-toward-zero integer the per-column packers' ``int()`` produces.
    """
    row = cp.zeros((n, stride), dtype=cp.int32)
    blocks = [(slot, _take(fields, name, 1, n, leaf, cp).astype(
        cp.int32, copy=False)) for name, slot in spec.items()]
    copy_slab_slots(row, blocks)
    return row


def _width(n: int, leaf: str) -> None:
    if n < 0:
        raise ValueError(f"{leaf} slab needs a non-negative width, got {n}")


def _blocks(n: int, threads: int) -> int:
    return (n + threads - 1) // threads


def _split(out, spec, cp, integers=()) -> dict:
    """Split a device output row into one owned, contiguous slab per name."""
    values = split_slab_slots(out, spec)
    for name in integers:
        values[name] = values[name].astype(cp.int32)
    return values


def pack_thermoprop_slab(fields, n: int):
    """``(float32[n, 44], int32[n, 4])`` for ``n`` land columns held as slabs.

    The rows are the ones ``noahmp_leaf_thermoprop`` reads, in the order
    :func:`woof.core.noahmp_thermal_gpu.pack_thermoprop_calls` writes them.
    """
    _slots_ix = []
    import cupy as cp

    _width(n, "THERMOPROP")
    x = cp.zeros((n, THERMOPROP_N_IN), dtype=cp.float32)
    _fill(x, fields, THERMOPROP_IN, n, "THERMOPROP", cp)
    ix = _integers(fields, THERMOPROP_INT, THERMOPROP_INT_STRIDE, n,
                   "THERMOPROP", cp)
    # ``1 if kw["urban_flag"] else 0``, vectorised: truth, not truncation, so a
    # bool slab and a float slab both land as 0/1 the way the loop does.
    for name, slot in THERMOPROP_FLAG.items():
        flag = _take(fields, name, 1, n, "THERMOPROP", cp)
        _slots_ix.append((slot, (flag != 0).astype(cp.int32)))
    copy_slab_slots(ix, _slots_ix)
    return x, ix


def evaluate_thermoprop_slab(fields, n: int, *, log10_profile="glibc_2_39") -> dict:
    """One launch of THERMOPROP over ``n`` columns; WRF's six arrays as slabs.

    Keys are ``df``, ``hcpct``, ``snicev``, ``snliqv``, ``epore`` and
    ``fact``, in the order :func:`evaluate_thermoprop_calls` returns them,
    with the layer axis last.
    """
    import cupy as cp

    x, ix = pack_thermoprop_slab(fields, n)
    y = cp.zeros((n, THERMOPROP_N_OUT), dtype=cp.float32)
    if n:
        assert y.flags.c_contiguous
        _thermoprop_kernel(log10_profile)(
            (_blocks(n, THERMAL_THREADS),), (THERMAL_THREADS,),
            (cp.ascontiguousarray(x), cp.ascontiguousarray(ix), y,
             np.int32(n)))
    return _split(y, THERMOPROP_OUT, cp)


def pack_tsnosoi_slab(fields, n: int):
    """``(float32[n, 42], int32[n, 5])`` for ``n`` land columns."""
    import cupy as cp

    _width(n, "TSNOSOI")
    x = cp.zeros((n, TSNOSOI_N_IN), dtype=cp.float32)
    _fill(x, fields, TSNOSOI_IN, n, "TSNOSOI", cp)
    ix = _integers(fields, TSNOSOI_INT, TSNOSOI_INT_STRIDE, n, "TSNOSOI", cp)
    return x, ix


def evaluate_tsnosoi_slab(fields, n: int, *, opt_stc=1, opt_tbot=2) -> dict:
    """One launch of TSNOSOI over ``n`` columns; ``{"stc", "eflxb"}``."""
    import cupy as cp

    if type(opt_stc) is not int or opt_stc not in (1,2,3):
        raise ValueError("TSNOSOI opt_stc must be 1, 2 or 3")
    if type(opt_tbot) is not int or opt_tbot not in (1,2):
        raise ValueError("TSNOSOI opt_tbot must be 1 or 2")
    x, ix = pack_tsnosoi_slab(fields, n)
    y = cp.zeros((n, TSNOSOI_N_OUT), dtype=cp.float32)
    if n:
        assert y.flags.c_contiguous
        if "snicar_heating" in fields:
            from woof.core.noahmp_kernel_sources import selected_kernel
            kernel = selected_kernel("noahmp_thermal", "noahmp_tsnosoi_snow_papers")
            args = (cp.ascontiguousarray(x), cp.ascontiguousarray(ix),
                    cp.ascontiguousarray(fields["snicar_heating"]), y, np.int32(n), np.int32(opt_stc), np.int32(opt_tbot))
        else:
            selected = opt_stc == 2 or opt_tbot == 1
            if selected:
                from woof.core.noahmp_kernel_sources import selected_kernel
                kernel = selected_kernel("noahmp_thermal", "noahmp_thermal_tsnosoi_selected")
            else:
                kernel = thermal_module().get_function("noahmp_thermal_tsnosoi")
            kernel(
                (_blocks(n, THERMAL_THREADS),), (THERMAL_THREADS,),
                (cp.ascontiguousarray(x), cp.ascontiguousarray(ix), y,
                 np.int32(n), *((np.int32(opt_stc), np.int32(opt_tbot)) if selected else ())))
        if "snicar_heating" in fields:
            kernel((_blocks(n, THERMAL_THREADS),), (THERMAL_THREADS,), args)
    return _split(y, TSNOSOI_OUT, cp)


def pack_phasechange_slab(fields, n: int):
    """``(float32[n, 57], int32[n, 4])`` for ``n`` land columns."""
    import cupy as cp

    _width(n, "PHASECHANGE")
    x = cp.zeros((n, PHASECHANGE_N_IN), dtype=cp.float32)
    _fill(x, fields, PHASECHANGE_IN, n, "PHASECHANGE", cp)
    ix = _integers(fields, PHASECHANGE_INT, PHASECHANGE_INT_STRIDE, n,
                   "PHASECHANGE", cp)
    return x, ix


def evaluate_phasechange_slab(fields, n: int, *, opt_frz=1) -> dict:
    """One launch of PHASECHANGE over ``n`` columns; WRF's ten return values.

    ``imelt`` comes back ``int32``, as the per-column unpacker's
    ``.astype(np.int32)`` gives it; everything else is ``float32``.
    """
    import cupy as cp

    if type(opt_frz) is not int or opt_frz not in (1, 2):
        raise ValueError("PHASECHANGE opt_frz must be 1 or 2")
    x, ix = pack_phasechange_slab(fields, n)
    y = cp.zeros((n, PHASECHANGE_N_OUT), dtype=cp.float32)
    if n:
        assert y.flags.c_contiguous
        if opt_frz == 2:
            from woof.core.noahmp_kernel_sources import selected_kernel
            kernel = selected_kernel("noahmp_thermal", "noahmp_thermal_phasechange_frz2")
        else:
            kernel = thermal_module().get_function("noahmp_thermal_phasechange")
        kernel(
            (_blocks(n, THERMAL_THREADS),), (THERMAL_THREADS,),
            (cp.ascontiguousarray(x), cp.ascontiguousarray(ix), y,
             np.int32(n)))
    return _split(y, PHASECHANGE_OUT, cp, integers=PHASECHANGE_INT_OUT)


def pack_radiation_slab(fields, n: int):
    """``(float32[n, 56], int32[n])`` for ``n`` land columns held as slabs.

    Every ``PARAMETER_SLOTS`` and ``STATE_SLOTS`` entry is its own key; see the
    module docstring for why the per-parameter-handle row cache does not
    survive the conversion.
    """
    import cupy as cp

    _width(n, "RADIATION")
    inputs = cp.zeros((n, RADIATION_N_IN), dtype=cp.float32)
    _fill(inputs, fields, RADIATION_IN, n, "RADIATION", cp)
    inputs[:, RADIATION_UNDEFINED_START:
           RADIATION_UNDEFINED_START + RADIATION_UNDEFINED] = _UNDEFINED_ENTRY
    ist = _take(fields, "ist", 1, n, "RADIATION", cp).astype(cp.int32,
                                                             copy=False)
    return inputs, cp.ascontiguousarray(ist)


def evaluate_radiation_slab(fields, n: int, *, noahmp_snow_age="wrf_470",
                            opt_alb=2, opt_rad=3) -> dict:
    """One launch of RADIATION over ``n`` columns.

    Keys are the ones ``noahmp_radiation_gpu.unpack_radiation_row``
    produces: ``OUTPUT_NAMES``, then ``albsnd``, ``albsni``, ``bgap``,
    ``wgap``, ``albold`` and ``tauss``.  ``albsnd`` and ``albsni`` are
    ``(n, 2)`` slabs rather than per-column pairs. ``opt_alb=1`` requires
    all seven ``BATS_PARAMETER_NAMES`` fields, each shape ``(n,)``. It uses
    the native BATS branch inside this same radiation launch. Full-column
    option admission is separate from this leaf's selected branch.
    """
    import cupy as cp

    from woof.core.kernels import get_kernel

    from woof.noahmp_generation import radiation_kernel
    if type(opt_alb) is not int or opt_alb not in (1, 2, 3):
        raise ValueError("RADIATION opt_alb must be 1 (BATS) or 2 (CLASS) or 3 (snow from papers)")
    if type(opt_rad) is not int or opt_rad not in (1, 2, 3):
        raise ValueError("RADIATION opt_rad must be 1 (crown), 2 (dense) or 3 (fractional)")
    kernel_name = radiation_kernel(noahmp_snow_age)
    inputs, ist = pack_radiation_slab(fields, n)
    bats = None
    if opt_alb == 1:
        bats = cp.empty((n, len(BATS_PARAMETER_NAMES)), dtype=cp.float32)
        _fill(bats, fields, {name: (i, 1) for i, name in
                            enumerate(BATS_PARAMETER_NAMES)}, n, "BATS", cp)
    out = cp.zeros((n, RADIATION_N_OUT), dtype=cp.float32)
    if n:
        assert out.flags.c_contiguous
        args = (cp.ascontiguousarray(inputs), cp.ascontiguousarray(ist))
        if opt_alb == 3:
            from woof.core.snow_papers import evaluate
            snow, carried, absorption = evaluate(inputs, ist, carbon=fields.get("snicar_carbon"),
                layers=fields.get("snicar_layers"), state=fields.get("snicar_state"),
                options=fields.get("snicar_options"),
                air_temperature=fields.get("snicar_air_temperature"), return_fluxes=True)
            heating = cp.zeros((n,7),dtype=cp.float32)
            crown = None
            if opt_rad == 1:
                crown = cp.empty((n, 3), dtype=cp.float32)
                _fill(crown, fields, {name: (i, 1) for i, name in
                                     enumerate(("rc", "hvt", "hvb"))}, n, "RAD1 crown", cp)
            args += (snow, absorption, heating, crown if crown is not None else np.uint64(0), out,
                     np.int32(n), np.int32(kernel_name.endswith("_night_age")), np.int32(opt_rad))
            kernel_name = "noahmp_radiation_batch_snow_papers"
        elif opt_rad == 1:
            crown = cp.empty((n, 3), dtype=cp.float32)
            _fill(crown, fields, {name: (i, 1) for i, name in
                                 enumerate(("rc", "hvt", "hvb"))}, n, "RAD1 crown", cp)
            args += (bats if bats is not None else np.uint64(0), crown, out, np.int32(n),
                     np.int32(kernel_name.endswith("_night_age")))
            kernel_name = "noahmp_radiation_batch_crown"
        elif opt_rad == 2:
            args += (bats if bats is not None else np.uint64(0), out, np.int32(n),
                     np.int32(kernel_name.endswith("_night_age")))
            kernel_name = "noahmp_radiation_batch_dense"
        elif bats is None:
            args += (out, np.int32(n))
        else:
            args += (bats, out, np.int32(n),
                     np.int32(kernel_name.endswith("_night_age")))
            kernel_name = "noahmp_radiation_batch_bats"
        if opt_alb == 3 or bats is not None or opt_rad != 3:
            from woof.core.noahmp_kernel_sources import selected_kernel
            kernel = selected_kernel("noahmp_radiation", kernel_name)
        else:
            kernel = get_kernel("noahmp_radiation", kernel_name)
        kernel(
            (_blocks(n, RADIATION_THREADS),), (RADIATION_THREADS,),
            args)
    result = _split(out, RADIATION_OUT, cp)
    if n and opt_alb == 3:
        result["snicar_state"] = carried
        result["snicar_heating"] = heating
    return result


__all__ = [
    "PHASECHANGE_IN",
    "PHASECHANGE_INT",
    "PHASECHANGE_OUT",
    "RADIATION_IN",
    "RADIATION_OUT",
    "RADIATION_PARAMETER_WIDTH",
    "THERMOPROP_IN",
    "THERMOPROP_INT",
    "THERMOPROP_OUT",
    "TSNOSOI_IN",
    "TSNOSOI_INT",
    "TSNOSOI_OUT",
    "evaluate_phasechange_slab",
    "evaluate_radiation_slab",
    "evaluate_thermoprop_slab",
    "evaluate_tsnosoi_slab",
    "pack_phasechange_slab",
    "pack_radiation_slab",
    "pack_thermoprop_slab",
    "pack_tsnosoi_slab",
]
