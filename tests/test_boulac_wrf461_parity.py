"""BouLac (bl_pbl_physics=8) through its PRODUCT launcher, against WRF v4.6.1.

``woof.core.boulac.launch_boulac`` is what ``PhysicsDriver._run_boulac``
calls.  This module drives it over every step of every column of the
oracle's fixtures of record (tools/boulac_wrf461_oracle: the driver linked
against the scalar WRF build's own ``phys/module_bl_boulac.o``, the object
inside the 0 ULP referee ``wrf.exe``) and compares every float32 output word
for EQUALITY: tendencies, TKE, mixing length, both exchange coefficients,
the four turbulent fluxes, the PBL height and, under BEP, the written-back
ground distance.

Multi-step: the scheme carries its own TKE (and DLG_BEP under BEP) from
step to step, exactly as WRF carries them; only the resolved state the
oracle advanced by its own tendencies is taken from the record.  So a word
wrong at step 1 is caught again at every later step it feeds.

The one declared divergence is graded where it lives: WRF's 0/0 rural drag
on a calm, zero-stress lowest level (``calm-ust0`` in ``calmprobe40``) makes
WRF's U and V columns NaN; the port's are finite and every other word of
that column, and every word of every other column, is WRF's.

Strict replay requires GPUWM_WRF_EXACT=1. Forecast arithmetic permits
contraction and CUDA libm; its accuracy is checked separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.verify.boulac_oracle import BEP_LEVEL_INPUTS, grids, load_grid
from woof.wrf_exact import ENABLED

pytestmark = pytest.mark.skipif(not ENABLED, reason="bitwise replay requires strict arithmetic")

GRIDS = tuple(g["grid"] for g in grids())

#: The outputs compared word for word, in record names.
COMPARED = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten", "tke",
            "el_pbl", "exch_h", "exch_m", "wu", "wv", "wt", "wq")
#: The U/V words WRF's 0/0 drag makes NaN on the declared column.
DIVERGENT_UV = ("rublten", "rvblten", "wu", "wv")
DECLARED_COLUMN = "calm-ust0"


def _bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def run_grid(name, *, kernel=None):
    """Run every step of one grid; return ``(index, record, results)``.

    ``results[s]`` maps each compared output (and ``pblh``/``dlg_bep``) to
    the port's ``(nz, ncol)`` / ``(ncol,)`` words at step ``s``.  ``kernel``
    replaces the compiled kernel (the planted-constant check).
    """
    import cupy as cp

    from woof.core import boulac

    inp, rec = load_grid(name)
    nz, nc = inp["nz"], inp["ncol"]

    def dev(a):
        return cp.asarray(np.ascontiguousarray(
            np.asarray(a, np.float32).reshape(a.shape[0], 1, nc)))

    def dev2(a):
        return cp.asarray(np.ascontiguousarray(
            np.asarray(a, np.float32).reshape(1, nc)))

    tke = dev(inp["tke"])
    bep = None
    frc = None
    if inp["flag_bep"]:
        bep = {key: dev(inp[key]) for key in BEP_LEVEL_INPUTS}
        bep["sf_bep"] = dev(inp["sf_bep"])
        frc = dev2(inp["frc_urb2d"])
    surface = {"hfx": dev2(inp["hfx"]), "qfx": dev2(inp["qfx"]),
               "ust": dev2(inp["ust"])}
    flux = {k: cp.zeros((nz, 1, nc), dtype=cp.float32)
            for k in ("wu", "wv", "wt", "wq")}
    results = []
    if kernel is not None:
        original = boulac.get_kernel
        boulac.get_kernel = lambda *_args: kernel
    try:
        for step in rec["steps"]:
            si = step["in"]
            columns = {"dz": dev(inp["dz8w"]), "u": dev(si["u"]),
                       "v": dev(si["v"]), "th": dev(si["th"]),
                       "rho": dev(inp["rho"]), "qv": dev(si["qv"]),
                       "qc": dev(si["qc"])}
            outputs = dict(flux)
            # WRF's driver zeroes the tendency family before every call
            # (module_pbl_driver.F:1036-1042); idiff = 1 leaves it so.
            for key in ("rublten", "rvblten", "rthblten", "rqvblten",
                        "rqcblten", "el_pbl", "exch_h", "exch_m"):
                outputs[key] = cp.zeros((nz, 1, nc), dtype=cp.float32)
            out = boulac.launch_boulac(
                columns, surface, tke, dt=float(inp["dt"]),
                cp_air=float(inp["cp"]), g=float(inp["g"]),
                outputs=outputs, bep=bep, frc_urb2d=frc,
                idiff=int(inp["idiff"]))
            got = {key: cp.asnumpy(out[key])[:, 0, :]
                   for key in COMPARED if key != "tke"}
            got["tke"] = cp.asnumpy(tke)[:, 0, :]
            got["pblh"] = cp.asnumpy(out["pblh"])[0]
            got["dlg_bep"] = (cp.asnumpy(bep["dlg_bep"])[:, 0, :]
                              if bep is not None else step["out"]["dlg_bep"])
            results.append(got)
    finally:
        if kernel is not None:
            boulac.get_kernel = original
    return inp, rec, results


def mismatches(name, rec, results, labels):
    """``[(step, field, column label, level)]`` of every unequal word."""
    bad = []
    declared = labels.index(DECLARED_COLUMN) if DECLARED_COLUMN in labels \
        else None
    for s, (step, got) in enumerate(zip(rec["steps"], results), start=1):
        want = step["out"]
        for key in COMPARED + ("dlg_bep",):
            w, g = want[key], got[key]
            neq = _bits(w) != _bits(g)
            if declared is not None and key in DIVERGENT_UV:
                col_w = w[:, declared]
                col_g = g[:, declared]
                # Where WRF wrote its 0/0 NaN the port is finite; every
                # word WRF wrote finite (the zero surface flux WU(kts),
                # WV(kts)) the port wrote identically.
                nan_w = np.isnan(col_w)
                if not (nan_w.any() and np.all(np.isfinite(col_g[nan_w]))):
                    bad.append((s, key, DECLARED_COLUMN, -1))
                neq[nan_w, declared] = False
            for k, c in zip(*np.nonzero(neq)):
                bad.append((s, key, labels[c], int(k)))
        neq = _bits(want["pblh"]) != _bits(got["pblh"])
        for c in np.nonzero(neq)[0]:
            bad.append((s, "pblh", labels[c], 0))
    return bad


@pytest.mark.gpu
@pytest.mark.parametrize("grid", GRIDS)
def test_every_step_of_every_column_is_wrfs_words(grid):
    import cupy  # noqa: F401  (marks this module for -m "not gpu")

    labels = next(g["labels"] for g in grids() if g["grid"] == grid)
    _inp, rec, results = run_grid(grid)
    bad = mismatches(grid, rec, results, labels)
    assert not bad, (f"{grid}: {len(bad)} words differ from WRF's, first "
                     f"{bad[:8]}")


@pytest.mark.gpu
def test_the_declared_column_is_finite_where_wrf_divides_zero_by_zero():
    """The 0/0 is real in WRF's record, and the port's column is finite."""
    import cupy  # noqa: F401

    labels = next(g["labels"] for g in grids()
                  if g["grid"] == "calmprobe40")
    _inp, rec, results = run_grid("calmprobe40")
    c = labels.index(DECLARED_COLUMN)
    want = rec["steps"][0]["out"]
    got = results[0]
    assert np.all(np.isnan(want["rublten"][:, c]))
    for key in DIVERGENT_UV:
        assert np.all(np.isfinite(got[key][:, c])), key
    # With no stress there is no drag: the lowest wind diffuses only
    # upward, never through the surface.
    assert got["wu"][0, c] == 0.0


@pytest.mark.gpu
def test_a_planted_one_bit_change_in_a_kernel_constant_is_caught():
    """ck_b one float32 step up (0x3ECCCCCD -> 0x3ECCCCCE) must fail.

    The check would be vacuous if a perturbation this small could pass it;
    it is planted in the assembled source the loader hands NVRTC, so it
    is the product kernel in every other respect.
    """
    import cupy as cp

    from woof.core.kernels import module_source, compile_noftz_module

    src = module_source("boulac")
    line = "#define BL_CK_B 0.40000000596046448f"
    assert line in src
    planted = src.replace(line, "#define BL_CK_B 0.40000003576278687f")
    assert np.float32("0.40000003576278687").view(np.uint32) == \
        np.float32(0.4).view(np.uint32) + 1
    module = compile_noftz_module(
        "boulac", planted, "boulac-planted")
    kernel = module.get_function("boulac_column")
    labels = next(g["labels"] for g in grids() if g["grid"] == "rural40")
    _inp, rec, results = run_grid("rural40", kernel=kernel)
    bad = mismatches("rural40", rec, results, labels)
    fields = {b[1] for b in bad}
    assert len(bad) > 1000, len(bad)
    assert {"exch_h", "tke", "rthblten"} <= fields, fields
