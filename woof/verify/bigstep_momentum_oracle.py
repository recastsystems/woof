"""Compiled WRF momentum fixtures, native layout adapters, and exact measurements.

There is no numerical tolerance in this module. WRF reads the same binary32
words as the production launches, with its real staggering and halo bounds.
The source WRF perturbation pressure is retained separately: the pressure
launch stores total pressure and loses low bits during its total/base roundtrip.
"""
from __future__ import annotations

from woof.backend import compilation_target as _compilation_target

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from woof.core.fp32_ulp import fp32_ulp_distance
from woof.verify.wrf471_fixtures import require_fixture_dir

# Test data in a source checkout, not package data (woof.verify.wrf471_fixtures).
MOMENTUM_ORACLE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "wrf471_bigstep"
MOMENTUM_CASES = ("real", "southern", "vertical-motion", "steep-terrain", "map-extremes", "periodic", "near-zero", "zero")
MOMENTUM_ROUTINES = ("horizontal_pressure_gradient", "coriolis", "curvature", "combined")


def load_momentum_measurement_pins():
    """Select measured output words for the current compile platform.

    The 4090/CUDA 12.9 receipt has distinct V curvature contraction words.
    Its WRF-order controls still match every compiled reference output.
    Other platforms run the original exact gate and report their identity
    if their measured words drift; this never skips a device comparison.
    """
    import cupy as cp
    from woof.certify.compile_platform import nvrtc_build

    platform = (str(_compilation_target(xp=cp)), nvrtc_build())
    filename = ("momentum-measurements-sm89-nvrtc12.9.86.json"
                if platform == ("89", "12.9.86") else "momentum-measurements.json")
    directory = require_fixture_dir(MOMENTUM_ORACLE_DIR, "big-step momentum")
    return json.loads((directory / filename).read_text()), platform


def make_momentum_inputs(real, case):
    f32 = np.float32
    names = dict(u="U", v="V", w="W", mup="MU", mub2d="MUB", pb="PB", php="PH", phb="PHB",
                 al="AL", msft="MAPFAC_M", msfu="MAPFAC_U", msfv="MAPFAC_V", f="F", e="E",
                 sina="SINALPHA", cosa="COSALPHA", xlat="XLAT", c1h="C1H", c2h="C2H",
                 c1f="C1F", c2f="C2F", fnm="FNM", fnp="FNP", rdnw="RDNW")
    inputs = {key: np.ascontiguousarray(real[value], dtype=f32).copy() for key, value in names.items()}
    inputs["p_raw"] = np.ascontiguousarray(real["P"], dtype=f32).copy()
    inputs["p"] = np.asarray(real["P"] + real["PB"], dtype=f32)
    inputs["alt"] = np.asarray(real["AL"] + real["ALB"], dtype=f32)
    for name in ("cf1", "cf2", "cf3", "cfn", "cfn1"):
        inputs[name] = np.asarray(real[name.upper()], dtype=f32).reshape(())
    nz, ny, nx = inputs["p"].shape
    metadata = {"case": case, "boundary_x": case != "periodic", "boundary_y": case != "periodic",
                "top_lid": case == "steep-terrain", "dx": 3000.0, "dy": 3000.0,
                "nx": nx, "ny": ny, "nz": nz}
    if "DX" in real:
        metadata["dx"] = float(np.asarray(real["DX"]).reshape(()))
        metadata["dy"] = float(np.asarray(real["DY"]).reshape(()))
    if case == "southern":
        inputs["f"] *= f32(-1)
        inputs["xlat"] *= f32(-1)
    if case in ("vertical-motion", "steep-terrain", "map-extremes"):
        levels = np.linspace(0, np.pi, nz + 1, dtype=f32)[:, None, None]
        pattern = np.sin(levels) * f32(0.3)
        inputs["w"][:] = pattern
    if case == "steep-terrain":
        slope = np.arange(nx, dtype=f32)[None, :] * f32(300.0 * 9.81)
        inputs["phb"] += slope[None]
    if case == "map-extremes":
        for name in ("msft", "msfu", "msfv"):
            j, i = np.indices(inputs[name].shape)
            inputs[name][:] = f32(0.35) + f32(2.5) * np.asarray((i + 2 * j) / (nx + 2 * ny), dtype=f32)
    if case in ("near-zero", "zero"):
        scale = f32(1.e-12 if case == "near-zero" else 0)
        for name in ("u", "v", "w", "mup", "php", "al"):
            inputs[name] *= scale
        inputs["p_raw"] *= scale
        inputs["p"] = np.asarray(inputs["p_raw"] + inputs["pb"], dtype=f32)
    if case == "periodic":
        # Periodic staggered grids own a duplicate face at nx/ny. WRF's
        # ghost face and the compact launch's duplicate face must be the
        # same word before applying a stencil across the seam.
        inputs["u"][:, :, -1] = inputs["u"][:, :, 0]
        inputs["v"][:, -1] = inputs["v"][:, 0]
        inputs["msfu"][:, -1] = inputs["msfu"][:, 0]
        inputs["msfv"][-1] = inputs["msfv"][0]
    mut = np.asarray(inputs["mub2d"] + inputs["mup"], dtype=f32)
    mux = np.empty((ny, nx + 1), dtype=f32)
    muy = np.empty((ny + 1, nx), dtype=f32)
    mux[:, 1:nx] = f32(.5) * (mut[:, :-1] + mut[:, 1:])
    muy[1:ny, :] = f32(.5) * (mut[:-1] + mut[1:])
    if metadata["boundary_x"]:
        mux[:, 0], mux[:, -1] = mut[:, 0], mut[:, -1]
    else:
        mux[:, 0] = mux[:, -1] = f32(.5) * (mut[:, 0] + mut[:, -1])
    if metadata["boundary_y"]:
        muy[0], muy[-1] = mut[0], mut[-1]
    else:
        muy[0] = muy[-1] = f32(.5) * (mut[0] + mut[-1])
    inputs["mut"] = mut
    inputs["muu"], inputs["muv"] = mux, muy
    inputs["ru"] = np.asarray((inputs["c1h"][:, None, None] * mux + inputs["c2h"][:, None, None]) * inputs["u"] / inputs["msfu"], dtype=f32)
    inputs["rv"] = np.asarray((inputs["c1h"][:, None, None] * muy + inputs["c2h"][:, None, None]) * inputs["v"] / inputs["msfv"], dtype=f32)
    inputs["rw"] = np.asarray((inputs["c1f"][:, None, None] * mut + inputs["c2f"][:, None, None]) * inputs["w"] / inputs["msft"], dtype=f32)
    # The moist coupling factor is supplied explicitly to both implementations.
    # It is derived at native faces by WRF's calc_cq argument convention.
    q = np.asarray(sum(real[key] for key in ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")), dtype=f32)
    qu = np.empty_like(inputs["u"]); qv = np.empty_like(inputs["v"])
    qu[:, :, 1:nx] = f32(.5) * (q[:, :, :-1] + q[:, :, 1:])
    qv[:, 1:ny] = f32(.5) * (q[:, :-1] + q[:, 1:])
    qu[:, :, 0] = qu[:, :, -1] = f32(.5) * (q[:, :, 0] + q[:, :, -1])
    qv[:, 0] = qv[:, -1] = f32(.5) * (q[:, 0] + q[:, -1])
    inputs["cqu"] = np.asarray(f32(1) / (f32(1) + qu), dtype=f32)
    inputs["cqv"] = np.asarray(f32(1) / (f32(1) + qv), dtype=f32)
    for name, shape in (("ru_t", inputs["ru"].shape), ("rv_t", inputs["rv"].shape), ("rw_t", inputs["rw"].shape)):
        inputs[name] = np.zeros(shape, dtype=f32)
    canonical = np.asarray(inputs["p"] - inputs["pb"], dtype=f32)
    metadata["pressure_roundtrip_differing"] = int(np.count_nonzero(canonical.view(np.uint32) != inputs["p_raw"].view(np.uint32)))
    metadata["pressure_roundtrip_max_abs_Pa"] = float(np.max(np.abs(canonical.astype(np.float64) - inputs["p_raw"])))
    return inputs, metadata


def _pad(value, shape, bx, by):
    """Native k,j,i -> WRF i,k,j, with two horizontal halo rows."""
    nzp, nyp, nxp = shape
    if value.ndim == 3:
        nk, nj, ni = value.shape
        kk = np.minimum(np.arange(nzp), nk - 1)
        ii = np.arange(-2, nxp - 2)
        jj = np.arange(-2, nyp - 2)
        nx, ny = nxp - 4, nyp - 4
        xi = np.clip(ii, 0, ni - 1) if bx else np.where((ii >= 0) & (ii < ni), ii, ii % nx)
        yj = np.clip(jj, 0, nj - 1) if by else np.where((jj >= 0) & (jj < nj), jj, jj % ny)
        return value[np.ix_(kk, yj, xi)].transpose(2, 0, 1)
    nj, ni = value.shape
    ii = np.arange(-2, nxp - 2); jj = np.arange(-2, nyp - 2)
    nx, ny = nxp - 4, nyp - 4
    xi = np.clip(ii, 0, ni - 1) if bx else np.where((ii >= 0) & (ii < ni), ii, ii % nx)
    yj = np.clip(jj, 0, nj - 1) if by else np.where((jj >= 0) & (jj < nj), jj, jj % ny)
    return value[np.ix_(yj, xi)].T


def pack_wrf_inputs(inputs, metadata, mode):
    nz, ny, nx = inputs["p"].shape
    shape = (nz + 1, ny + 4, nx + 4)
    bx, by = metadata["boundary_x"], metadata["boundary_y"]
    a = np.zeros((nx + 4, nz + 1, ny + 4, 19), dtype=np.float32, order="F")
    b = np.zeros((nx + 4, ny + 4, 12), dtype=np.float32, order="F")
    z = np.zeros((nz + 1, 7), dtype=np.float32, order="F")
    canonical = inputs["p_raw"] if mode == 5 else np.asarray(inputs["p"] - inputs["pb"], dtype=np.float32)
    php_half = np.float32(.5) * ((inputs["php"][:-1] + inputs["php"][1:]) + (inputs["phb"][:-1] + inputs["phb"][1:]))
    fields = (inputs["php"], inputs["alt"], canonical, inputs["pb"], inputs["al"], php_half,
              inputs["cqu"], inputs["cqv"], inputs["ru"], inputs["rv"], inputs["rw"],
              inputs["u"], inputs["v"], inputs["w"])
    for index, value in enumerate(fields):
        a[:, :, :, index] = _pad(value, shape, bx, by)
    for index, name in enumerate(("muu", "muv", "mup", "msfu", "msfv", "msft", "f", "e", "sina", "cosa", "xlat")):
        b[:, :, index] = _pad(inputs[name], shape, bx, by)
    for index, name in enumerate(("c1h", "c2h", "fnm", "fnp", "rdnw", "fnm", "fnp")):
        value = inputs[name]
        z[:value.size, index] = value
    for index, name in enumerate(("ru_t", "rv_t", "rw_t"), 16):
        a[:, :, :, index] = _pad(inputs[name], shape, bx, by)
    scalars = np.asarray([inputs[key] for key in ("cf1", "cf2", "cf3", "cfn", "cfn1")] + [1 / metadata["dx"], 1 / metadata["dy"]], dtype=np.float32)
    header = np.asarray([nx, ny, nz, mode, bx, by, metadata["top_lid"]], dtype=np.int32)
    return header.tobytes() + a.tobytes(order="F") + b.tobytes(order="F") + z.tobytes(order="F") + scalars.tobytes(), (a[:, :, :, 16:19].copy(), nz, ny, nx)


def unpack_wrf_outputs(raw, layout):
    initial, nz, ny, nx = layout
    full = np.frombuffer(raw, dtype=np.float32).reshape((nx + 4, nz + 1, ny + 4, 3), order="F")
    outputs = {}
    halo_equal = True
    for index, (name, shape) in enumerate((("ru_t", (nz, ny, nx + 1)), ("rv_t", (nz, ny + 1, nx)), ("rw_t", (nz + 1, ny, nx)))):
        nk, nj, ni = shape
        selection = (slice(2, ni + 2), slice(0, nk), slice(2, nj + 2))
        outputs[name] = np.ascontiguousarray(full[selection + (index,)].transpose(1, 2, 0))
        mask = np.ones(full.shape[:3], dtype=bool)
        mask[selection] = False
        halo_equal &= np.array_equal(full[:, :, :, index][mask].view(np.uint32), initial[:, :, :, index][mask].view(np.uint32))
    return outputs, halo_equal


@dataclass(frozen=True)
class MomentumFixture:
    case: str
    inputs: dict
    metadata: dict
    reference: dict


def load_momentum_fixture(case, directory=None):
    directory = require_fixture_dir(
        MOMENTUM_ORACLE_DIR if directory is None else directory, "big-step momentum")
    with np.load(directory / f"momentum-{case}.npz", allow_pickle=False) as archive:
        inputs = {key[3:]: archive[key].copy() for key in archive.files if key.startswith("in_")}
        metadata = json.loads(str(archive["metadata"]))
        reference = {routine: {field: archive[f"{routine}_{field}"].copy() for field in ("ru_t", "rv_t", "rw_t")} for routine in (*MOMENTUM_ROUTINES, "original_pressure")}
    return MomentumFixture(case, inputs, metadata, reference)


def momentum_port_outputs(fixture, routine):
    import cupy as cp
    from woof.core.dycore import _launch_slow_pgf, launch_coriolis_curvature
    device = {name: cp.asarray(value) for name, value in fixture.inputs.items()}
    metadata = fixture.metadata
    if routine == "horizontal_pressure_gradient":
        state = SimpleNamespace(**device)
        # RawKernel scalar arguments must be host scalar words. A zero-rank
        # device array passes an address, even when the signature wants REAL.
        for name in ("cf1", "cf2", "cf3", "cfn", "cfn1"):
            setattr(state, name, np.float32(fixture.inputs[name]))
        cfg = SimpleNamespace(dx=metadata["dx"], dy=metadata["dy"], top_lid=metadata["top_lid"],
                              open_x=metadata["boundary_x"], open_y=metadata["boundary_y"])
        _launch_slow_pgf(state, cfg, cq=(device["cqu"], device["cqv"], device["cqu"], True))
    else:
        zeros = cp.zeros_like(device["msft"])
        u = cp.zeros_like(device["u"]) if routine == "coriolis" else device["u"]
        v = cp.zeros_like(device["v"]) if routine == "coriolis" else device["v"]
        f = zeros if routine == "curvature" else device["f"]
        e = zeros if routine == "curvature" else device["e"]
        launch_coriolis_curvature(device["ru"], device["rv"], u, v, device["w"], device["mut"], device["msft"], device["msfu"], device["msfv"], f, e,
                                 device["c1f"], device["c2f"], device["fnm"], device["fnp"], metadata["dx"], metadata["dy"],
                                 device["ru_t"], device["rv_t"], device["rw_t"], sina=device["sina"], cosa=device["cosa"],
                                 boundary_x=metadata["boundary_x"], boundary_y=metadata["boundary_y"])
    return {field: cp.asnumpy(device[field]) for field in ("ru_t", "rv_t", "rw_t")}


def measure_momentum_parity(reference, outputs):
    result = {}
    for field, expected in reference.items():
        actual = outputs[field]
        distance = fp32_ulp_distance(actual, expected)
        result[field] = {"max_ulp": int(distance.max()), "differing_words": int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))),
                         "words": actual.size, "max_abs": float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))),
                         "actual_sha256": hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest(),
                         "reference_sha256": hashlib.sha256(np.ascontiguousarray(expected).tobytes()).hexdigest()}
    return result


def pressure_launch_arithmetic_outputs(fixture):
    """Diagnostic: reproduce slow_pgf's explicit FP32 operation boundaries.

    This is an attribution control, not a WRF reference or an acceptance
    tolerance. Its purpose is to distinguish the launch's reassociation from
    a changed term, staggering, halo, or coefficient after the compiled WRF
    comparison has measured the disagreement.
    """
    a = fixture.inputs
    metadata = fixture.metadata
    f32 = np.float32
    nz, ny, nx = a["p"].shape
    pp = np.asarray(a["p"] - a["pb"], dtype=f32)
    dpn = np.empty((nz + 1, ny, nx), dtype=f32)
    dpn[0] = (a["cf1"] * pp[0] + a["cf2"] * pp[1]) + a["cf3"] * pp[2]
    dpn[1:nz] = a["fnm"][1:, None, None] * pp[1:] + a["fnp"][1:, None, None] * pp[:-1]
    dpn[nz] = (a["cfn"] * pp[-1] + a["cfn1"] * pp[-2]) if metadata["top_lid"] else f32(0)
    phh = f32(.5) * ((a["php"][:-1] + a["php"][1:]) + (a["phb"][:-1] + a["phb"][1:]))
    outputs = {name: a[name].copy() for name in ("ru_t", "rv_t", "rw_t")}
    for axis, length, field, cq in ((2, nx, "ru_t", "cqu"), (1, ny, "rv_t", "cqv")):
        ix = np.arange(length + 1) % length
        im = (np.arange(length + 1) - 1) % length
        mass_axis = axis - 1
        take = lambda value: (np.take(value, ix, axis=axis), np.take(value, im, axis=axis))
        ma, mb = np.take(a["mut"], ix, axis=mass_axis), np.take(a["mut"], im, axis=mass_axis)
        mua, mub = np.take(a["mup"], ix, axis=mass_axis), np.take(a["mup"], im, axis=mass_axis)
        muf = f32(.5) * (ma + mb)
        dmu = f32(.5) * (mua + mub)
        layer = a["c1h"][:, None, None] * muf + a["c2h"][:, None, None]
        pha, phb = take(a["php"])
        bracket = (pha[1:] - phb[1:]) + (pha[:-1] - phb[:-1])
        alta, altb = take(a["alt"]); ppa, ppb = take(pp)
        bracket = bracket + (alta + altb) * (ppa - ppb)
        ala, alb = take(a["al"]); pba, pbb = take(a["pb"])
        bracket = bracket + (ala + alb) * (pba - pbb)
        rd = f32(1 / metadata["dx" if axis == 2 else "dy"])
        left = (f32(.5) * rd * layer) * bracket
        phha, phhb = take(phh)
        dpna, dpnb = take(dpn)
        dp_hi = f32(.5) * (dpna[1:] + dpnb[1:])
        dp_lo = f32(.5) * (dpna[:-1] + dpnb[:-1])
        vertical = a["rdnw"][:, None, None] * (dp_hi - dp_lo)
        vertical = vertical - a["c1h"][:, None, None] * dmu
        right = (rd * (phha - phhb)) * vertical
        term = (left + right) * a[cq]
        result = outputs[field] - term
        if metadata["boundary_x" if axis == 2 else "boundary_y"]:
            boundary = [slice(None)] * 3
            boundary[axis] = (0, -1)
            result[tuple(boundary)] = outputs[field][tuple(boundary)]
        outputs[field] = np.asarray(result, dtype=f32)
    return outputs
