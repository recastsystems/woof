"""Whole-slab CuPy execution of Noah-MP's WATER subtree.

:mod:`woof.core.noahmp_water_gpu` already evaluates every land column in one
``k_water`` launch, but it gets there through a Python loop: one bound
argument dict per column, one row of slice assignments per column, and on the
way back one :class:`~woof.core.noahmp_water.WaterFluxes` and one mutated
:class:`~woof.core.noahmp_snow.SnowColumn` per column.  At d04 that loop is
the cost -- the launch is not.  This module is the same packing with the
column axis vectorised: the caller hands over whole-slab CuPy arrays and the
Python that runs is proportional to the number of *fields* -- fifty-two, see
:data:`FIELD_NAMES` -- and not to the number of columns.

Nothing here computes physics.  ``k_water`` is used unchanged, and every
layout constant is *imported* from :mod:`woof.core.noahmp_water_gpu` rather
than restated, so the two packers cannot drift apart in slot order: they are
reading the same dictionaries.  What this module adds is the assignment, and
that is exactly what ``tests/test_noahmp_water_slab.py`` compares bit for bit
against the per-column packer.

The ``fields`` mapping
----------------------
Keys are the names :func:`woof.core.noahmp_water_gpu._bind_call` produces --
that is, ``CALL_NAMES`` -- with three flattenings:

* the ``SnowColumn`` arrives as its own attributes (``isnow``, ``snowh``,
  ``sneqv``, ``snice``, ``snliq``, ``stc``, ``zsnso``, ``dzsnso``, ``sh2o``,
  ``sice``) rather than as an object;
* the ``parameters`` handle arrives as the values named in ``P_LAYER`` and
  ``P_SCALAR``, plus ``urban_flag`` and ``nroot``, which are the only two the
  packer reads into the integer row;
* ``parameters`` and ``col`` themselves are therefore not keys.

A per-column scalar has shape ``(n,)``; a per-column vector has shape
``(n, width)`` with the **layer axis last** -- ``(n, NSOIL)``, ``(n, NSNOW)``
or ``(n, NFULL)`` as :data:`FLOAT_FIELDS` and :data:`INT_FIELDS` say.  Widths
are checked, because a ``(n, NSOIL)`` array silently broadcast into an
``(n, NFULL)`` destination is a wrong forecast, not an exception.

Contiguity
----------
A CuPy raw-kernel argument carries a device pointer and nothing else -- no
shape, no strides.  Handing ``k_water`` a strided view therefore does not
fail; it reads the wrong words.  Two rules follow, and both are tested:

* every array passed to the launch is ``cupy.ascontiguousarray``;
* every array *returned* is ``cupy.ascontiguousarray`` too, because the
  natural spelling of an output column, ``out[:, slot]``, is a strided view
  and the caller's next step is usually another raw kernel.

Return order
------------
:func:`pack_water_slab` returns ``(par, fin, iin)`` -- deliberately the same
triple, in the same order, as
:func:`~woof.core.noahmp_water_gpu.pack_water_calls`.  Two functions that
pack the identical layout must not disagree about which one comes first; that
is precisely the transposition this lane keeps getting bitten by.

``etrani``
----------
Left out, as in the per-column path.  WRF's WATER computes it as a local,
``NOAHMP_SFLX`` never reads it, and the device row does not carry it.  It is
named in :data:`ABSENT_OUTPUTS` so a caller that expects it gets a missing
key rather than a recomputed second implementation of something nothing
consumes.
"""

from __future__ import annotations

from woof.core.noahmp_slab_libm import copy_slab_slots, split_slab_slots

import numpy as np

from woof.core.noahmp_water_gpu import (
    IN_SCALAR,
    IN_STRIDE,
    IN_VECTOR,
    INT_STRIDE,
    NSNOW,
    NSOIL,
    NSTATE,
    OUT_SCALAR,
    OUT_DIAGNOSTIC,
    OUT_STRIDE,
    OUT_VECTOR,
    P_LAYER,
    P_SCALAR,
    P_STRIDE,
    STATE_SLOT,
    STATE_WIDTH,
    THREADS,
)

#: The integer row, ``noahmp_water.cu``'s ``ip[]``.  Slots 0-5 are per-column
#: scalars and 6-8 are IMELT's three snow layers; this is the same order
#: ``pack_water_calls`` writes as a tuple.
INT_SCALAR = {"isnow": 0, "urban_flag": 1, "nroot": 2, "ist": 3,
              "frozen_canopy": 4, "frozen_ground": 5}
INT_VECTOR = {"imelt": (6, NSNOW)}

#: Every float field the slab consumes, mapped to its per-column width.
FLOAT_FIELDS: dict[str, int] = {}
FLOAT_FIELDS.update({name: NSOIL for name in P_LAYER})
FLOAT_FIELDS.update({name: 1 for name in P_SCALAR})
FLOAT_FIELDS.update({name: STATE_WIDTH[name] for name in STATE_SLOT})
FLOAT_FIELDS.update({name: 1 for name in IN_SCALAR})
FLOAT_FIELDS.update({name: width for name, (_o, width) in IN_VECTOR.items()})

#: Every integer field the slab consumes, mapped to its per-column width.
INT_FIELDS: dict[str, int] = {name: 1 for name in INT_SCALAR}
INT_FIELDS.update({name: width for name, (_o, width) in INT_VECTOR.items()})

FIELD_NAMES = tuple(FLOAT_FIELDS) + tuple(INT_FIELDS)

#: Everything :func:`evaluate_water_slab` returns: the mutated column state
#: first, then the flux record.  ``isnow`` comes from the kernel's integer
#: output rather than from the float row.
OUTPUT_NAMES = (("isnow",) + tuple(STATE_SLOT) + tuple(OUT_VECTOR)
                + tuple(OUT_SCALAR) + tuple(OUT_DIAGNOSTIC))

#: ``WaterFluxes`` members the device row does not carry.  See the module
#: docstring; the per-column path returns ``None`` for this one.
ABSENT_OUTPUTS = ("etrani",)


def _take(fields, name: str, width: int, n: int, cp):
    """The slab for one field, shape-checked and on the device."""
    try:
        value = fields[name]
    except KeyError:
        raise KeyError(f"WATER slab is missing {name!r}") from None
    array = cp.asarray(value)
    want = (n,) if width == 1 else (n, width)
    if array.shape != want:
        raise ValueError(
            f"WATER slab field {name!r} must have shape {want}, got "
            f"{tuple(array.shape)}")
    return array


def pack_water_slab(fields: dict, n: int):
    """Return ``(par, fin, iin)`` for ``n`` land columns held as slabs.

    The rows are the ones ``noahmp_water.cu`` reads, in the order
    :func:`woof.core.noahmp_water_gpu.pack_water_calls` writes them; the only
    difference is that each slot is filled by one vectorised assignment across
    all columns instead of by one assignment per column.
    """
    _slots_par = []
    _slots_fin = []
    _slots_iin = []
    import cupy as cp

    if n < 0:
        raise ValueError(f"WATER slab needs a non-negative width, got {n}")

    par = cp.zeros((n, P_STRIDE), dtype=cp.float32)
    fin = cp.zeros((n, IN_STRIDE), dtype=cp.float32)
    iin = cp.zeros((n, INT_STRIDE), dtype=cp.int32)

    # --- the parameter block ------------------------------------------------
    for name, base in P_LAYER.items():
        _slots_par.append((base, _take(fields, name, NSOIL, n, cp)))
    for name, slot in P_SCALAR.items():
        _slots_par.append((slot, _take(fields, name, 1, n, cp)))

    # --- the packed snow/soil column ---------------------------------------
    for name, base in STATE_SLOT.items():
        width = STATE_WIDTH[name]
        block = _take(fields, name, width, n, cp)
        _slots_fin.append((base, block))

    # --- the forcing ---------------------------------------------------------
    for name, slot in IN_SCALAR.items():
        _slots_fin.append((NSTATE + slot, _take(fields, name, 1, n, cp)))
    for name, (offset, width) in IN_VECTOR.items():
        start = NSTATE + offset
        _slots_fin.append((start, _take(fields, name, width, n, cp)))

    # --- the integer row -----------------------------------------------------
    # ``astype`` rather than a bare assignment so a bool slab (FROZEN_CANOPY,
    # URBAN_FLAG) and a float slab both land as the 0/1 and truncated-toward-
    # zero integers the per-column packer's ``int()`` produces.
    #
    # One documented difference: the per-column packer spells the three flags
    # ``1 if value else 0``, so it collapses any truthy value to 1, while this
    # carries the value through.  For a bool or 0/1 slab -- which is what the
    # runtime holds and what the kernel's ``if (ip[k])`` reads -- the two are
    # identical; a slab carrying, say, 2 would not be, and is not a thing a
    # flag should carry.
    for name, slot in INT_SCALAR.items():
        _slots_iin.append((slot, _take(fields, name, 1, n, cp).astype(cp.int32,
                                                            copy=False)))
    for name, (offset, width) in INT_VECTOR.items():
        _slots_iin.append((offset, _take(
            fields, name, width, n, cp).astype(cp.int32, copy=False)))

    copy_slab_slots(iin, _slots_iin)
    copy_slab_slots(fin, _slots_fin)
    copy_slab_slots(par, _slots_par)
    return par, fin, iin


def _unpack_slab(fout, iout, cp) -> dict:
    """Split the device rows into one owned, contiguous slab per output."""
    specs = {name: (base, STATE_WIDTH[name]) for name, base in STATE_SLOT.items()}
    specs.update({name: (NSTATE + offset, width)
                  for name, (offset, width) in OUT_VECTOR.items()})
    specs.update({name: (NSTATE + slot, 1) for name, slot in OUT_SCALAR.items()})
    specs.update({name: (NSTATE + slot, 1) for name, slot in OUT_DIAGNOSTIC.items()})
    return {"isnow": cp.ascontiguousarray(iout),
            **split_slab_slots(fout, specs)}


_SELECTED_MODULES = {}


def evaluate_water_slab(fields: dict, n: int, *, opt_inf=1, opt_run=3, opt_infdv=0, soil_update_steps=1, calculate_soil=True, opt_tdrn=0, opt_irr=0) -> dict:
    """Evaluate ``n`` land columns of WATER in one launch, slab in slab out.

    Returns every name in :data:`OUTPUT_NAMES`: the column state WRF mutates
    in place (``isnow``, ``snowh``, ``sneqv``, ``snice``, ``snliq``, ``stc``,
    ``zsnso``, ``dzsnso``, ``sh2o``, ``sice``) and every ``WaterFluxes``
    member the device row carries.  Layer axis last, as on the way in.
    """
    import cupy as cp

    from woof.core.kernels import get_kernel

    if type(opt_inf) is not int or opt_inf not in (1, 2):
        raise ValueError("WATER opt_inf must be 1 or 2")
    if type(opt_run) is not int or opt_run not in (1, 2, 3, 4, 5, 6, 7, 8):
        raise ValueError("WATER opt_run must be 1, 2, 3, 4, 5, 6, 7 or 8")

    if type(opt_infdv) is not int or opt_infdv not in (0,1,2,3):
        raise ValueError("WATER opt_infdv must be 0,1,2,3")
    if opt_run==8 and opt_infdv==0:
        raise ValueError("RUN8 requires opt_infdv=1,2,3: native FSUR is unassigned at zero")
    if type(opt_tdrn) is not int or opt_tdrn not in (0,1,2):
        raise ValueError("WATER opt_tdrn must be 0,1,2")
    par, fin, iin = pack_water_slab(fields, n)
    if opt_run == 8:
        selected_par=cp.empty((n,P_STRIDE+3),dtype=cp.float32)
        copy_slab_slots(selected_par,[(0,par),*[(P_STRIDE+k,_take(fields,name,1,n,cp))
                        for k,name in enumerate(("bbvic","bdvic","gdvic"))]])
        par=selected_par
    if opt_run in (6, 7):
        selected_par = cp.zeros((n, P_STRIDE + 8), dtype=cp.float32)
        slots = [(0, par)]
        if opt_run == 6:
            slots.append((P_STRIDE + 4, _take(fields, "bvic", 1, n, cp)))
        else:
            slots.append((P_STRIDE, _take(fields, "smcref", NSOIL, n, cp)))
            for offset, name in enumerate(("axaj", "bxaj", "xxaj"), P_STRIDE + 5):
                slots.append((offset, _take(fields, name, 1, n, cp)))
        copy_slab_slots(selected_par, slots)
        par = selected_par
    groundwater_fields=("zwt","wa","wt") if opt_run in (1,2) else ()
    if groundwater_fields:
        selected_par=cp.zeros((n,P_STRIDE+6),dtype=cp.float32)
        copy_slab_slots(selected_par,[(0,par),(P_STRIDE,_take(fields,"fsatmx",1,n,cp)),
            (P_STRIDE+1,_take(fields,"timean",1,n,cp)),(P_STRIDE+2,_take(fields,"psisat",NSOIL,n,cp))])
        par=selected_par
        selected_fin=cp.zeros((n,IN_STRIDE+3),dtype=cp.float32)
        copy_slab_slots(selected_fin,[(0,fin),*[(IN_STRIDE+k,_take(fields,name,1,n,cp)) for k,name in enumerate(groundwater_fields)]])
        fin=selected_fin
    if opt_run==5:
        groundwater_fields=("zwt","wa","smcwtd","deeprech","rech")
        selected_par=cp.empty((n,P_STRIDE+5),dtype=cp.float32)
        copy_slab_slots(selected_par,[(0,par),(P_STRIDE,_take(fields,"fsatmx",1,n,cp)),(P_STRIDE+1,_take(fields,"psisat",NSOIL,n,cp))]);par=selected_par
        selected_fin=cp.empty((n,IN_STRIDE+9),dtype=cp.float32)
        copy_slab_slots(selected_fin,[(0,fin),*[(IN_STRIDE+k,_take(fields,name,1,n,cp))for k,name in enumerate(groundwater_fields)],(IN_STRIDE+5,_take(fields,"smceq",NSOIL,n,cp))]);fin=selected_fin
    if opt_tdrn:
        from woof.core.noahmp_tile_drain import PARAMETERS
        base=par.shape[1]
        extended=cp.empty((n,base+15),dtype=cp.float32)
        copy_slab_slots(extended,[(0,par),(base,_take(fields,"smcref",NSOIL,n,cp)),*[(base+4+k,_take(fields,name,1,n,cp)) for k,name in enumerate(PARAMETERS)]])
        par=extended
        base=fin.shape[1]
        extended=cp.empty((n,base+3),dtype=cp.float32)
        copy_slab_slots(extended,[(0,fin),*[(base+k,_take(fields,name,1,n,cp)) for k,name in enumerate(("tdfracmp","dx","zwt"))]])
        fin=extended
    if opt_irr:
        from woof.core.noahmp_irrigation_slab import parameter_slab
        base=par.shape[1];extended=cp.empty((n,base+35),dtype=cp.float32)
        copy_slab_slots(extended,[(0,par),(base,parameter_slab(fields,n))]);par=extended
        base=fin.shape[1];extended=cp.empty((n,base+5),dtype=cp.float32)
        copy_slab_slots(extended,[(0,fin),*[(base+k,_take(fields,name,1,n,cp))for k,name in enumerate(('croplu','mifac','fifac','iramtmi','iramtfi'))]]);fin=extended
    fout = cp.zeros((n, OUT_STRIDE+len(groundwater_fields)+(1 if opt_tdrn else 0)+(4 if opt_irr else 0)), dtype=cp.float32)
    iout = cp.zeros(n, dtype=cp.int32)
    if n:
        extra = ()
        if opt_inf == 1 and opt_run == 3 and soil_update_steps==1 and calculate_soil and opt_tdrn==0 and opt_irr==0:
            kernel = get_kernel("noahmp_water", "k_water")
        else:
            from woof.core.noahmp_kernel_sources import compile_runtime_unit
            card = cp.cuda.runtime.getDevice()
            if card not in _SELECTED_MODULES:
                _SELECTED_MODULES[card] = compile_runtime_unit(
                    "noahmp_water", module_key="noahmp_water_selected",
                    selected_options=True)
            kernel = _SELECTED_MODULES[card].get_function("k_water_selected")
            extra = (np.int32(opt_inf), np.int32(opt_run), np.int32(opt_infdv),np.int32(soil_update_steps),np.int32(calculate_soil),np.int32(opt_tdrn),np.int32(bool(opt_irr)))
        blocks = (n + THREADS - 1) // THREADS
        # Every *input* pointer is made contiguous first.  All three are
        # allocations here, so nothing is copied; the call is a guard that
        # stays correct if a future caller ever hands the launch a slice.
        # The two outputs are deliberately NOT wrapped: ``ascontiguousarray``
        # of a strided array returns a copy, and a kernel writing into a copy
        # would lose every answer silently.  They must be allocations, and
        # they are asserted to be.
        assert fout.flags.c_contiguous and iout.flags.c_contiguous
        kernel(
            (blocks,), (THREADS,),
            (cp.ascontiguousarray(par), cp.ascontiguousarray(fin),
             cp.ascontiguousarray(iin), fout, iout, np.int32(n), *extra),
        )
    result=_unpack_slab(fout,iout,cp)
    if groundwater_fields:
        result.update(split_slab_slots(fout,{name:(OUT_STRIDE+k,1) for k,name in enumerate(groundwater_fields)}))
    if opt_tdrn:
        result["zwt"]=split_slab_slots(fout,{"zwt":(OUT_STRIDE+len(groundwater_fields),1)})["zwt"]
    if opt_irr:
        base=OUT_STRIDE+len(groundwater_fields)+(1 if opt_tdrn else 0)
        result.update(split_slab_slots(fout,{name:(base+k,1)for k,name in enumerate(('iramtmi','iramtfi','irmirate','irfirate'))}))
    return result


__all__ = [
    "ABSENT_OUTPUTS",
    "FIELD_NAMES",
    "FLOAT_FIELDS",
    "INT_FIELDS",
    "INT_SCALAR",
    "INT_VECTOR",
    "OUTPUT_NAMES",
    "evaluate_water_slab",
    "pack_water_slab",
]
