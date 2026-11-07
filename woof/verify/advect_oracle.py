"""Compiled WRF v4.7.1 advection fixtures and word-level measurements.

The reference is produced by the native Fortran calls in
``tools/advect_wrf471_oracle/run_advect.F90``.  No NumPy transcription is
used here.  Launches go through the production advection and moisture
launchers.  Measurements have no acceptance tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import re
from types import SimpleNamespace

import numpy as np

from woof.core.fp32_ulp import fp32_ulp_distance
from woof.verify.default_kernel_source import default_source
from woof.verify.wrf471_fixtures import require_fixture_dir


# Test data in a source checkout, not package data (woof.verify.wrf471_fixtures).
ADVECT_ORACLE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "wrf471_advect"
ROUTINES = {
    "advect_scalar": (1, "scalar", "tend_scalar"),
    "advect_u": (2, "u", "tend_u"),
    "advect_v": (3, "v", "tend_v"),
    "advect_w": (4, "w", "tend_w"),
    "advect_scalar_pd": (5, "scalar_pd", "tend_pd"),
    "advect_scalar_mono": (6, "scalar_pd", "tend_pd"),
}
SUPPORTED_ROUTINES = tuple(name for name in ROUTINES if name != "advect_scalar_mono")
SUPPORTED_OUTPUTS = SUPPORTED_ROUTINES + ("advect_scalar_pd.h_tendency", "advect_scalar_pd.z_tendency")
SENTINEL = np.float32(-999999.0)
PROTOCOL_VERSION = 1


def advect_gpu_identity() -> tuple[str, tuple[int, int]]:
    """Query the active card only when replaying its CUDA receipt."""
    import cupy as cp
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = device["name"]
    return (name.decode() if isinstance(name, bytes) else name,
            (device["major"], device["minor"]))


# Byte identity is CERTIFIED on Blackwell and newer only: compute capability
# 10.x (sm_100) and 12.x (sm_120) and later.  Ada, Ampere and Hopper run but
# are not certified (ruling 2026-10-04).  So a receipt measured on an
# older architecture is informational: it stays committed and is checked for
# presence and format, but a kernel or fixture move that leaves it behind is
# reported (UncertifiedReceiptStale, naming the commit that staled it) and
# fails nothing.  A Blackwell receipt gates exactly as before.
#
# The breakage this prevents: b70a94a48 (the vert_order 5 ladder) moved every
# advection entry point; 71be2bf27 recaptured the RTX 4090 and RTX 5090
# receipts, and the H100 receipt, which no lane can reach, held the 2.8.6
# receipt-pin test red on an architecture nobody certifies.
CERTIFIED_MIN_COMPUTE_MAJOR = 10
#: The receipt-index key that records which commit staled an uncertified
#: receipt: {receipt file: {"staled_by": 40-hex commit, "pins": {pin: digest}}}.
#: ``pins`` are the receipt's own digests that commit left behind ("fixture"
#: for ``fixture_manifest_sha256``).  tools/advect_wrf471_oracle/
#: recapture_receipt.py drops the record when it installs that receipt again.
UNCERTIFIED_STALE_KEY = "uncertified_stale"
_HEX64 = re.compile(r"[0-9a-f]{64}")


class UncertifiedReceiptStale(UserWarning):
    """A receipt of an uncertified architecture trails the source or fixture."""


def architecture_is_certified(compute_capability) -> bool:
    """Blackwell and newer (ruling 2026-10-04); ``None`` (no card) is not."""
    return (compute_capability is not None
            and int(tuple(compute_capability)[0]) >= CERTIFIED_MIN_COMPUTE_MAJOR)


def check_receipt_format(receipt: dict, fixture_dir: Path, modules, cases) -> None:
    """Presence and format, for every receipt, certified or not."""
    assert receipt.get("schema_version") == 1, receipt.get("schema_version")
    assert isinstance(receipt.get("gpu"), str) and receipt["gpu"], receipt.get("gpu")
    capability = receipt.get("compute_capability")
    assert (isinstance(capability, list) and len(capability) == 2
            and all(isinstance(part, int) for part in capability)), capability
    for module in modules:
        assert _HEX64.fullmatch(str(receipt["kernels"].get(module))), (receipt["gpu"], module)
    assert _HEX64.fullmatch(str(receipt.get("fixture_manifest_sha256"))), receipt["gpu"]
    assert (Path(fixture_dir) / receipt.get("words_directory", "gpu-words")).is_dir(), receipt["gpu"]
    assert receipt.get("mutation_rejected") is True, receipt["gpu"]
    for case in cases:
        row = receipt["cases"][case.name]
        assert _HEX64.fullmatch(str(row["words_sha256"])), (receipt["gpu"], case.name)
        assert row["words_file"] and set(row["measurements"]) >= set(SUPPORTED_OUTPUTS), (
            receipt["gpu"], case.name)


def receipt_staleness(receipt: dict, fixture_dir: Path, modules) -> dict[str, tuple[str, str]]:
    """``{pin: (recorded, current)}`` for each pin the receipt trails.

    Kernel pins are the SHA-256 of exactly the source the default loader
    compiles; ``"fixture"`` is the fixture's ``cases.json``.
    """
    from woof.core.kernels import module_source
    moved = {}
    for module in modules:
        current = hashlib.sha256(module_source(module).encode("utf-8")).hexdigest()
        if receipt["kernels"][module] != current:
            moved[module] = (receipt["kernels"][module], current)
    manifest = sha256(Path(fixture_dir) / "cases.json")
    if receipt["fixture_manifest_sha256"] != manifest:
        moved["fixture"] = (receipt["fixture_manifest_sha256"], manifest)
    return moved


def _recorded_pin(receipt: dict, pin: str) -> str:
    return receipt["fixture_manifest_sha256"] if pin == "fixture" else receipt["kernels"][pin]


def check_uncertified_stale_records(index: dict, fixture_dir: Path, modules) -> dict:
    """Every staleness record must be true; returns them by receipt file.

    A record names an uncertified receipt the index selects, a full commit,
    and digests the receipt still carries and the tree has left.  A record on
    a Blackwell receipt would excuse a certified pin, and one that outlives a
    recapture would name a staling that no longer holds: both fail here.
    """
    records = index.get(UNCERTIFIED_STALE_KEY, {})
    selectable = set(index["cards"].values()) | set(index["architectures"].values())
    for name, record in records.items():
        assert name in selectable, f"{UNCERTIFIED_STALE_KEY} names {name}, which the index never selects"
        receipt = json.loads((Path(fixture_dir) / name).read_text(encoding="ascii"))
        assert not architecture_is_certified(receipt["compute_capability"]), (
            f"{name} is a certified (Blackwell or newer) receipt; it gates and has no staleness record")
        assert re.fullmatch(r"[0-9a-f]{40}", str(record.get("staled_by"))), (name, record.get("staled_by"))
        moved = receipt_staleness(receipt, fixture_dir, modules)
        assert record.get("pins"), name
        for pin, digest in record["pins"].items():
            assert digest == _recorded_pin(receipt, pin), (
                f"{name}'s staleness record pins {pin} {str(digest)[:8]}, but the receipt now "
                f"pins {_recorded_pin(receipt, pin)[:8]}: drop or rewrite the record")
            assert pin in moved, (
                f"{name}'s {pin} pin is current again; its staleness record must go")
    return records


def uncertified_stale_report(name: str, receipt: dict, moved: dict, record: dict | None) -> str:
    """The informational line for an uncertified receipt that trails the tree."""
    major, minor = receipt["compute_capability"]
    recorded = (record or {}).get("pins", {})
    parts = []
    for pin, (was, now) in sorted(moved.items()):
        cause = (f"staled by {record['staled_by'][:9]}" if pin in recorded
                 else "staling commit not recorded under "
                      f"{UNCERTIFIED_STALE_KEY!r} in gpu-receipts.json")
        parts.append(f"{pin} {was[:8]} -> {now[:8]} ({cause})")
    return (f"{name} ({receipt['gpu']}, sm_{major}{minor}) is informational, not a gate: "
            "byte identity is certified on Blackwell and newer only (ruling 2026-10-04). "
            f"It trails the tree: {'; '.join(parts)}. A capture on that card with "
            "tools/advect_wrf471_oracle/recapture_receipt.py --install brings it current.")


@dataclass(frozen=True)
class AdvectOracleCase:
    name: str
    metadata: dict
    inputs: dict[str, np.ndarray]
    reference: dict[str, np.ndarray]

    @property
    def shape(self):
        return self.inputs["scalar"].shape


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_advect_cases(directory: Path | None = None) -> tuple[AdvectOracleCase, ...]:
    directory = require_fixture_dir(
        directory if directory is not None else ADVECT_ORACLE_DIR, "advection")
    manifest = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    rows = manifest["cases"] if isinstance(manifest, dict) else manifest
    cases = []
    for row in rows:
        with np.load(directory / row["file"], allow_pickle=False) as data:
            inputs = {key: np.ascontiguousarray(data[key]) for key in data.files}
        reference = {}
        ref_path = directory / row.get("reference", f"{row['name']}-wrf.npz")
        if ref_path.exists():
            with np.load(ref_path, allow_pickle=False) as data:
                reference = {key: np.ascontiguousarray(data[key]) for key in data.files}
        cases.append(AdvectOracleCase(row["name"], row["metadata"], inputs, reference))
    return tuple(cases)


def validate_case(case: AdvectOracleCase) -> None:
    """Reject a staggering or map convention the launchers cannot express."""
    a = case.inputs
    nz, ny, nx = case.shape
    shapes = {
        "scalar": (nz, ny, nx), "scalar_pd": (nz, ny, nx), "q0": (nz, ny, nx),
        "u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx),
        "w": (nz + 1, ny, nx), "ru": (nz, ny, nx + 1),
        "rv": (nz, ny + 1, nx), "rw": (nz + 1, ny, nx),
        "mu_old": (ny, nx), "muts": (ny, nx),
        "msfux": (ny, nx + 1), "msfuy": (ny, nx + 1),
        "msfvx": (ny + 1, nx), "msfvy": (ny + 1, nx),
        "msftx": (ny, nx), "msfty": (ny, nx),
    }
    for key, shape in shapes.items():
        if a[key].shape != shape or a[key].dtype != np.float32:
            raise ValueError(f"{case.name} {key}: expected float32 {shape}, got {a[key].dtype} {a[key].shape}")
    for x, y in (("msfux", "msfuy"), ("msfvx", "msfvy"), ("msftx", "msfty")):
        if not np.array_equal(a[x].view(np.uint32), a[y].view(np.uint32)):
            raise ValueError(f"{case.name}: anisotropic {x}/{y} maps have no engine launch counterpart")
    for key in ("rdnw", "rdn", "fnm", "fnp", "c1h", "c2h"):
        if a[key].dtype != np.float32 or a[key].ndim != 1 or a[key].size < nz:
            raise ValueError(f"{case.name}: invalid {key}")
    for key in ("dx", "dy", "dt", "open_x", "open_y", "specified"):
        if key not in case.metadata:
            raise ValueError(f"{case.name}: missing {key}")


def w_lid_control_cases(base: AdvectOracleCase) -> tuple[AdvectOracleCase, ...]:
    """Two representable edge cases isolate the native W lid terms.

    The atmospheric state and dry-mass coordinate come from the real case.
    Controlled fluxes separate horizontal transport from the vertical lid
    contribution, so neither term can conceal an omitted term in the other.
    """
    cases = []
    nz, ny, nx = base.shape
    for name in ("w_lid_horizontal", "w_lid_vertical"):
        a = {key: value.copy() for key, value in base.inputs.items()}
        m = dict(base.metadata, dx=1.0, dy=1.0, open_x=False, open_y=False, specified=False)
        for key in ("ru", "rv", "rw"):
            a[key].fill(0.0)
        a["w"][-2:] = np.float32(0.25)
        a["w_old"] = a["w"].copy()
        a["tend_w"].fill(0.0)
        for key in ("msftx", "msfty", "msfux", "msfuy", "msfvx", "msfvy"):
            a[key].fill(1.0)
        if name == "w_lid_horizontal":
            a["fnm"].fill(0.5)
            a["fnp"].fill(0.5)
            values = np.arange(nx + 1, dtype=np.float32) * np.float32(64.0)
            values[-1] = values[0]
            a["ru"][:] = values[None, None, :]
        else:
            a["rw"][-2].fill(1.0)
        cases.append(replace(base, name=name, metadata=m, inputs=a, reference={}))
    return tuple(cases)


def mapped_radiation_control_cases(base: AdvectOracleCase) -> tuple[AdvectOracleCase, ...]:
    """Mapped normal winds put the radiation clamp in a different branch."""
    cases = []
    for name, field, flux in (("mapped_open_u", "u", "ru"), ("mapped_open_v", "v", "rv")):
        a = {key: value.copy() for key, value in base.inputs.items()}
        m = dict(base.metadata, dx=1.0, dy=1.0, open_x=True, open_y=True, specified=False)
        for key in ("u", "v", "ru", "rv", "rw", "mu_perturbation", "c2h", "tend_u", "tend_v"):
            a[key].fill(0.0)
        for key in ("mu_old", "muts", "mub"):
            a[key].fill(65536.0)
        a["c1h"].fill(1.0)
        for key in ("msftx", "msfty", "msfux", "msfuy", "msfvx", "msfvy"):
            a[key].fill(2.0)
        a[field].fill(40.0)
        if field == "u":
            a[field][..., 1] = 41.0
            a[field][..., -2] = 39.0
        else:
            a[field][:, 1, :] = 41.0
            a[field][:, -2, :] = 39.0
        a[flux][:] = np.float32(32768.0) * a[field]
        a[field + "_old"] = a[field].copy()
        other = "v" if field == "u" else "u"
        a[other + "_old"] = a[other].copy()
        cases.append(replace(base, name=name, metadata=m, inputs=a, reference={}))
    return tuple(cases)


def _initial(case, key, shape):
    value = case.inputs.get(key)
    return np.zeros(shape, dtype=np.float32) if value is None else np.array(value, dtype=np.float32, copy=True)


def _memory_shape(case):
    nz, ny, nx = case.shape
    return nz + 1, ny + 8, nx + 8


def _pad3(case, values, *, sentinel=False):
    """WRF i=-3..nx+4, j=-3..ny+4, k=1..nz+1.

    Four halo cells surround the physical mass domain.  Periodic axes
    wrap the independent mass cells; specified and open halos use a
    zero-gradient copy.  A redundant staggered end face is retained.
    Tendency halos are sentinels rather than extrapolated values.
    """
    values = np.asarray(values, dtype=np.float32)
    nz, ny, nx = case.shape
    if sentinel:
        padded = np.full(_memory_shape(case), SENTINEL, dtype=np.float32)
        padded[:values.shape[0], 4:4 + values.shape[1], 4:4 + values.shape[2]] = values
        return padded
    kz = np.clip(np.arange(nz + 1), 0, values.shape[0] - 1)
    yi = np.arange(-4, ny + 4)
    xi = np.arange(-4, nx + 4)
    if case.metadata["open_y"]:
        yi = np.clip(yi, 0, values.shape[1] - 1)
    else:
        yi = yi % ny
        if values.shape[1] == ny + 1:
            yi[np.arange(-4, ny + 4) == ny] = ny
    if case.metadata["open_x"]:
        xi = np.clip(xi, 0, values.shape[2] - 1)
    else:
        xi = xi % nx
        if values.shape[2] == nx + 1:
            xi[np.arange(-4, nx + 4) == nx] = nx
    return np.ascontiguousarray(values[np.ix_(kz, yi, xi)])


def _pad2(case, values):
    return _pad3(case, np.asarray(values)[None])[0]


def _vertical(case, name):
    values = case.inputs[name]
    nz = case.shape[0]
    return np.pad(values[:nz + 1], (0, max(0, nz + 1 - len(values)))).astype(np.float32)


def advection_orders(metadata) -> dict[str, int]:
    """The four WRF advection orders a case runs, WRF's Registry defaults
    (h 5, v 3) where its metadata names none; the native header and the
    CUDA launchers both read them so one case means one configuration."""
    return {"h_mom_adv_order": int(metadata.get("h_mom_adv_order", 5)),
            "v_mom_adv_order": int(metadata.get("v_mom_adv_order", 3)),
            "h_sca_adv_order": int(metadata.get("h_sca_adv_order", 5)),
            "v_sca_adv_order": int(metadata.get("v_sca_adv_order", 3))}


def write_fortran_input(case: AdvectOracleCase, routine: str, path: Path) -> np.ndarray:
    """Serialize the real argument arrays using the fixed Fortran stream ABI."""
    validate_case(case)
    mode, field, tend_name = ROUTINES[routine]
    nz, ny, nx = case.shape
    m = case.metadata
    flags = (int(not m["open_x"]) | (int(not m["open_y"]) << 1)
             | (int(m["specified"]) << 2))
    if m["open_x"] and not m["specified"]:
        flags |= (1 << 4) | (1 << 5)
    if m["open_y"] and not m["specified"]:
        flags |= (1 << 6) | (1 << 7)
    orders = advection_orders(m)
    header = np.asarray([
        PROTOCOL_VERSION, mode, 1, nx + 1, 1, ny + 1, 1, nz + 1,
        -3, nx + 4, -3, ny + 4, 1, nz + 1,
        1, nx + 1, 1, ny + 1, 1, nz + 1,
        1, orders["h_mom_adv_order"], orders["v_mom_adv_order"],
        orders["h_sca_adv_order"], orders["v_sca_adv_order"],
        flags, int(mode >= 5),
    ], dtype="<i4")
    scalars = np.asarray([1.0 / m["dx"], 1.0 / m["dy"], m["dt"]], dtype="<f4")
    a = case.inputs
    old = a["q0"] if mode >= 5 else a.get(f"{field}_old", a[field])
    initial = _pad3(case, _initial(case, tend_name, a[field].shape), sentinel=True)
    arrays3 = [_pad3(case, a[field]), _pad3(case, old), initial,
               _pad3(case, a["ru"]), _pad3(case, a["rv"]),
               _pad3(case, a["rw"]), _pad3(case, a.get("rw_i", np.zeros_like(a["rw"])))]
    mub = a.get("mub", a["mu_old"])
    mu_old_perturbation = a.get("mu_perturbation", a.get("mu", np.asarray(a["mu_old"] - mub, dtype=np.float32)))
    arrays2 = [a["muts"], mub, mu_old_perturbation,
               *[a[key] for key in ("msfux", "msfuy", "msfvx", "msfvy", "msftx", "msfty")]]
    # mut is total stage mass; mu_old is the old perturbation mass.
    # WRF ph_low retains the separate base/perturbation expression tree.
    with Path(path).open("wb") as out:
        out.write(header.tobytes())
        out.write(scalars.tobytes())
        for value in arrays3:
            out.write(value.transpose(1, 0, 2).astype("<f4").tobytes(order="C"))
        for value in arrays2:
            out.write(_pad2(case, value).astype("<f4").tobytes())
        vertical_names = ("c1f", "c2f") if mode == 4 else ("c1h", "c2h")
        for name in (*vertical_names, "fnm", "fnp", "rdnw", "rdn"):
            out.write(_vertical(case, name).astype("<f4").tobytes())
    return initial


def read_fortran_output(case: AdvectOracleCase, path: Path, *, diagnostic=False) -> dict[str, np.ndarray]:
    nzm, nym, nxm = _memory_shape(case)
    words = np.fromfile(path, dtype="<f4")
    if words.size != 3 * nzm * nym * nxm:
        raise ValueError(f"{path}: expected {3 * nzm * nym * nxm} words, got {words.size}")
    names = ("tendency", "h_tendency", "z_tendency")
    arrays = words.reshape(3, nym, nzm, nxm)
    result = {key: np.ascontiguousarray(value.transpose(1, 0, 2)) for key, value in zip(names, arrays)}
    if not diagnostic:
        for key in ("h_tendency", "z_tendency"):
            if not np.all(result[key].view(np.uint32) == SENTINEL.view(np.uint32)):
                raise ValueError(f"{path}: inactive optional output {key} was written")
    return result


def generate_reference(case: AdvectOracleCase, executable: Path, scratch: Path) -> dict[str, np.ndarray]:
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    output = {}
    for routine in ROUTINES:
        input_path = scratch / f"{case.name}-{routine}.in.bin"
        output_path = scratch / f"{case.name}-{routine}.out.bin"
        write_fortran_input(case, routine, input_path)
        subprocess.run([str(executable), str(input_path), str(output_path)], check=True)
        arrays = read_fortran_output(case, output_path, diagnostic=ROUTINES[routine][0] >= 5)
        output[routine] = arrays["tendency"]
        if ROUTINES[routine][0] >= 5:
            for key in ("h_tendency", "z_tendency"):
                output[f"{routine}.{key}"] = arrays[key]
    return output


def advect_port_outputs(case: AdvectOracleCase, *, variant="production") -> dict[str, np.ndarray]:
    """Run five supported routines through their ordinary engine launchers."""
    import cupy as cp
    from woof.core.advection import (
        launch_flux_div_scalar, launch_flux_div_u, launch_flux_div_v, launch_flux_div_w)
    from woof.core.moist import launch_pd_fluxes, launch_pd_renorm_apply

    validate_case(case)
    a = {key: cp.asarray(value) for key, value in case.inputs.items() if value.dtype == np.float32}
    coord = SimpleNamespace(**{key: a[key] for key in ("rdnw", "rdn", "fnm", "fnp", "c1h", "c2h")})
    m = case.metadata
    common = dict(open_x=bool(m["open_x"]), open_y=bool(m["open_y"]), has_msf=True)
    orders = advection_orders(m)
    if orders["h_mom_adv_order"] != 5 or orders["h_sca_adv_order"] != 5:
        raise ValueError(f"{case.name}: the launchers carry WRF's 5th-order horizontal stencils only")
    output = {}
    tendencies = {}
    # WRF advect_w keys its vertical order on v_sca_adv_order; u and v on
    # v_mom_adv_order (module_advect_em.F).
    launchers = {
        "advect_scalar": (launch_flux_div_scalar, "msftx", orders["v_sca_adv_order"]),
        "advect_u": (launch_flux_div_u, "msfux", orders["v_mom_adv_order"]),
        "advect_v": (launch_flux_div_v, "msfvx", orders["v_mom_adv_order"]),
        "advect_w": (launch_flux_div_w, "msftx", orders["v_sca_adv_order"]),
    }
    for routine, (launch, map_name, vorder) in launchers.items():
        _, field, tend_name = ROUTINES[routine]
        tend = cp.asarray(_initial(case, tend_name, case.inputs[field].shape))
        launch(a[field], a["ru"], a["rv"], a["rw"], tend, coord,
               m["dx"], m["dy"], msf=a[map_name], spec=bool(m["specified"]),
               vorder=vorder, **common)
        tendencies[routine] = tend
        output[routine] = _pad3(case, cp.asnumpy(tend), sentinel=True)
    nz, ny, nx = case.shape
    if (m["open_x"] or m["open_y"]) and not m["specified"]:
        # Native advect_u/v include the radiative cb blocks.  Production
        # separates them into its ordinary slow-tendency boundary launch.
        from woof.core.dycore import apply_open_radiative_bc
        state = SimpleNamespace(
            u=a["u"], v=a["v"], mup=a["mu_perturbation"], mub2d=a["mub"],
            c1h=a["c1h"], c2h=a["c2h"],
            msfu=a["msfux"], msfv=a["msfvx"], has_msf=True,
            ru_t=tendencies["advect_u"], rv_t=tendencies["advect_v"])
        cfg = SimpleNamespace(nz=nz, ny=ny, nx=nx, dx=m["dx"], dy=m["dy"],
                              open_x=bool(m["open_x"]), open_y=bool(m["open_y"]))
        apply_open_radiative_bc(state, cfg)
        for routine in ("advect_u", "advect_v"):
            output[routine] = _pad3(case, cp.asnumpy(tendencies[routine]), sentinel=True)
    shapes = ((nz, ny, nx + 1),) * 2 + ((nz, ny + 1, nx),) * 2 + ((nz + 1, ny, nx),) * 2
    fluxes = [cp.full(shape, np.nan, dtype=cp.float32) for shape in shapes]
    tend = cp.asarray(_initial(case, "tend_pd", case.shape))
    launch_pd_fluxes(a["scalar_pd"], a["q0"], a["ru"], a["rv"], a["rw"], a["muts"],
                     coord, m["dx"], m["dy"], m["dt"], *fluxes, msft=a["msftx"],
                     vorder=orders["v_sca_adv_order"], **common)
    launch_pd_renorm_apply(a["q0"], a["mu_old"], *fluxes, tend=tend, coord=coord,
                           dx=m["dx"], dy=m["dy"], dt=m["dt"], msft=a["msftx"], **common)
    output["advect_scalar_pd"] = _pad3(case, cp.asnumpy(tend), sentinel=True)
    h, z = _pd_diagnostics(case, a, coord, fluxes, variant)
    output["advect_scalar_pd.h_tendency"] = _pad3(case, cp.asnumpy(h), sentinel=True)
    output["advect_scalar_pd.z_tendency"] = _pad3(case, cp.asnumpy(z), sentinel=True)
    # Every field declared as input must remain unmodified by these calls.
    for key, value in a.items():
        if not np.array_equal(cp.asnumpy(value).view(np.uint32), case.inputs[key].view(np.uint32)):
            raise AssertionError(f"{case.name}: launcher modified input {key}")
    return output


def measure_words(got, reference, *, defined=None) -> dict:
    """Measure all words, including signed zeros, nonfinite values and halos."""
    got = np.ascontiguousarray(got, dtype=np.float32)
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    if got.shape != reference.shape:
        raise ValueError(f"shape mismatch {got.shape} != {reference.shape}")
    undefined_words = 0
    if defined is not None:
        defined = np.asarray(defined, dtype=bool)
        if defined.shape != got.shape:
            raise ValueError("output defined mask shape differs")
        undefined_words = int(got.size - np.count_nonzero(defined))
        got = got.copy()
        reference = reference.copy()
        got[~defined] = reference[~defined] = SENTINEL
    gb, rb = got.view(np.uint32), reference.view(np.uint32)
    delta = fp32_ulp_distance(got, reference)
    absolute = np.abs(got.astype(np.float64) - reference.astype(np.float64))
    unequal = gb != rb
    count = int(np.count_nonzero(unequal))
    row = {
        "words": int(got.size), "different_words": count,
        "max_ulp": int(delta.max()) if delta.size else 0,
        "max_abs_difference": float(absolute.max()) if absolute.size else 0.0,
        "signed_zero_words": int(np.count_nonzero(unequal & (got == 0) & (reference == 0))),
        "nonfinite_different_words": int(np.count_nonzero(unequal & (~np.isfinite(got) | ~np.isfinite(reference)))),
        "got_sha256": hashlib.sha256(got.tobytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "undefined_words": undefined_words,
    }
    if count:
        first = int(np.flatnonzero(unequal.ravel())[0])
        worst = int(np.argmax(delta.ravel()))
        row.update(first_index=[int(i) for i in np.unravel_index(first, got.shape)],
                   worst_index=[int(i) for i in np.unravel_index(worst, got.shape)],
                   worst_got=float(got.ravel()[worst]),
                   worst_reference=float(reference.ravel()[worst]),
                   worst_got_bits=f"{int(gb.ravel()[worst]):08x}",
                   worst_reference_bits=f"{int(rb.ravel()[worst]):08x}")
    return row


def measure_advect_parity(case, outputs) -> dict:
    if set(outputs) != set(SUPPORTED_OUTPUTS):
        raise ValueError("comparison must include every supported routine")
    return {routine: measure_words(outputs[routine], case.reference[routine],
                                   defined=defined_output_mask(case, routine))
            for routine in SUPPORTED_OUTPUTS}


def defined_output_mask(case, output):
    """Exclude only WRF's unassigned h channel subsequently read by +=.

    Native PD/mono h_tendency assigns x contributions only on the x
    interior, then adds y contributions on every x column.  At an open or
    specified x edge this addition reads no defined native h value.  These
    words are recorded in raw dumps but cannot measure a physical output.
    """
    mask = np.ones(_memory_shape(case), dtype=bool)
    if output.endswith(".h_tendency") and case.metadata["open_x"]:
        nz, ny, nx = case.shape
        y0, y1 = (5, ny + 3) if case.metadata["open_y"] else (4, ny + 4)
        mask[:nz, y0:y1, 4] = False
        mask[:nz, y0:y1, nx + 3] = False
    return mask


def _replace_function(source, name, body):
    pattern = rf"(real {name}\([^{{]+\{{).*?\n\}}"
    replaced, count = re.subn(pattern, lambda match: match.group(1) + "\n" + body + "\n}", source, count=1, flags=re.S)
    if count != 1:
        raise ValueError(f"diagnostic source cannot locate {name}")
    return replaced


def control_source(module: str, variant: str) -> str:
    """Test-only controls preserve the caller and vary named arithmetic choices."""
    from woof.core.kernels import module_source
    source = module_source(module)
    if variant in ("production", "no_fma"):
        return source
    if module == "openbc":
        return source
    # Patch the code the default compile runs: the first flux5 coefficient
    # in the module text otherwise sits in an opt-in WRF-exact branch.
    source = default_source(_default_order_terms(source))
    if variant == "mutation":  # default order-5 flux5; a bare 37.0f now lands first in the opt-in order-6 branch
        if "__fdiv_rn((vel * (37.0f" not in source:
            raise ValueError("flux5 mutation site disappeared")
        return source.replace("__fdiv_rn((vel * (37.0f", "__fdiv_rn((vel * (38.0f", 1)
    if variant not in ("wrf_flux", "wrf_flux_no_fma"):
        raise ValueError(variant)
    prefix = "pd_" if module in ("pd_advection", "pd_vertical_sl") else ""
    if prefix:
        center5 = "0.6166666746139526f * (q0 + qm1) - 0.13333334028720856f * (qp1 + qm2) + 0.01666666753590107f * (qp2 + qm3)"
        dissip5 = "copysignf(1.0f, vel) * 0.01666666753590107f * ((qp2 - qm3) - 5.0f * (qp1 - qm2) + 10.0f * (q0 - qm1))"
        center3 = "0.5833333134651184f * (q0 + qm1) - 0.0833333358168602f * (qp1 + qm2)"
        dissip3 = "copysignf(1.0f, vel) * 0.0833333358168602f * ((qp1 - qm2) - 3.0f * (q0 - qm1))"
    else:
        center5 = "__fdiv_rn(37.0f * (q0 + qm1) - 8.0f * (qp1 + qm2) + (qp2 + qm3), 60.0f)"
        dissip5 = "__fdiv_rn(copysignf(1.0f, vel) * ((qp2 - qm3) - 5.0f * (qp1 - qm2) + 10.0f * (q0 - qm1)), 60.0f)"
        center3 = "__fdiv_rn(7.0f * (q0 + qm1) - (qp1 + qm2), 12.0f)"
        dissip3 = "__fdiv_rn(copysignf(1.0f, vel) * ((qp1 - qm2) - 3.0f * (q0 - qm1)), 12.0f)"
    source = _replace_function(source, prefix + "flux5", f"    return vel * (({center5}) - ({dissip5}));")
    source = _replace_function(source, prefix + "flux3", f"    return vel * (({center3}) - ({dissip3}));")
    source = _replace_function(source, prefix + "flux3h", f"    return vel * (({center3}) + ({dissip3}));")
    return source


@contextmanager
def arithmetic_control(variant="production"):
    """Use the production launchers with a test-only compiled source variant."""
    if variant == "production":
        yield
        return
    import cupy as cp
    import woof.core.advection as advect
    import woof.core.moist as moist
    import woof.core.dycore as dycore
    options = ("-std=c++17", "--fmad=false") if variant.endswith("no_fma") else ("-std=c++17",)
    modules = {name: cp.RawModule(code=control_source(name, variant), options=options)
               for name in ("advection", "pd_advection", "pd_vertical_sl", "openbc")}
    originals = advect.get_kernel, moist.get_kernel, dycore.get_kernel
    def get_control(name, func, shape=()):
        if name in modules:
            return modules[name].get_function(func)
        return originals[0](name, func, shape)
    advect.get_kernel = moist.get_kernel = dycore.get_kernel = get_control
    try:
        yield
    finally:
        advect.get_kernel, moist.get_kernel, dycore.get_kernel = originals


def _pd_diagnostics(case, a, coord, fluxes, variant):
    """Expose native h/z optional outputs in a test-only kernel clone.

    The total tendency is independently obtained from the unmodified
    production launch.  This clone preserves its limiter computation and
    adds stores of horizontal and vertical divergence.  WRF's h diagnostic
    begins with the wrapper sentinel on rows skipped by the x loop and
    subsequently receives the y loop's additive write.
    """
    import cupy as cp
    # The clone is a default compile, so it extends the default parameter
    # list; the opt-in WRF-exact parameters are resolved away first.
    source = default_source(control_source("pd_advection", variant))
    signature = re.compile(r"int open_x, int open_y\s*\)\n\{\n    int i = blockIdx\.x")
    replacement = "int open_x, int open_y, real* diag_h, real* diag_z)\n{\n    int i = blockIdx.x"
    split = source.index("void pd_renorm_apply(")
    preamble, apply_source = source[:split], source[split:]
    if len(signature.findall(apply_source)) != 1:
        raise ValueError("PD diagnostic signature changed")
    source = preamble + signature.sub(replacement, apply_source, count=1)
    anchor = "#undef PD_SCALE"
    body = r'''
    real m_diag = has_msf ? msft[(size_t)j * nx + i] : 1.0f;
    real x_diag = -m_diag * dx_inv *
       ((sx_r * fxc_r + fxl[I3(k,j,i+1,ny,nx+1)])
        - (sx_l * fxc_l + fxl[I3(k,j,i,ny,nx+1)]));
    real y_diag = -m_diag * dy_inv *
       ((sy_r * fyc_r + fyl[I3(k,j+1,i,ny+1,nx)])
        - (sy_l * fyc_l + fyl[I3(k,j,i,ny+1,nx)]));
    real z_diag = -rdnw[k] *
       ((sz_t * fzc_t + fzl[IDX3(k+1,j,i)])
        - (sz_b * fzc_b + fzl[IDX3(k,j,i)]));
    diag_z[IDX3(k,j,i)] = 0.0f + z_diag;
    if (!open_x || (i >= 1 && i <= nx-2))
        diag_h[IDX3(k,j,i)] = 0.0f + x_diag;
    if (!open_y || (j >= 1 && j <= ny-2))
        diag_h[IDX3(k,j,i)] += y_diag;
'''
    source = source.replace(anchor, anchor + "\n" + body)
    options = ("-std=c++17", "--fmad=false") if variant.endswith("no_fma") else ("-std=c++17",)
    module = cp.RawModule(code=source, options=options)
    kernel = module.get_function("pd_renorm_apply")
    nz, ny, nx = case.shape
    h = cp.full(case.shape, SENTINEL, dtype=cp.float32)
    z = cp.full(case.shape, SENTINEL, dtype=cp.float32)
    unused_tend = cp.zeros(case.shape, dtype=cp.float32)
    m = case.metadata
    args = (a["q0"], a["mu_old"], *fluxes, a["c1h"], a["c2h"], a["rdnw"], a["msftx"],
            np.float32(1.0 / m["dx"]), np.float32(1.0 / m["dy"]), np.float32(m["dt"]), unused_tend,
            np.int32(nz), np.int32(ny), np.int32(nx), np.int32(1),
            np.int32(m["open_x"]), np.int32(m["open_y"]), h, z)
    kernel(((nx + 127) // 128, ny, nz), (128, 1, 1), args)
    return h, z


#: The transport-order kernels (dyn-options fold 6a79a9fb05) open with
#: ``#if GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_NO_HADV || GPUWM_HORDER == 2 ...``.
_EXACT_OR_ORDER_TERMS = re.compile(
    r"^([ \t]*#[ \t]*if[ \t]+)GPUWM_WRF_EXACT_C_ADVECTION[ \t]*\|\|[ \t]*"
    r"((?:GPUWM_NO_HADV|GPUWM_HORDER[ \t]*==[ \t]*[0-9]+)"
    r"(?:[ \t]*\|\|[ \t]*(?:GPUWM_NO_HADV|GPUWM_HORDER[ \t]*==[ \t]*[0-9]+))*)[ \t]*$",
    re.M)


def _default_order_terms(source: str) -> str:
    """The advection exact selector is undefined (0) in the default compile,
    so such a directive is its order terms alone, still decided by the
    compiler.  Only this enumerated form is rewritten; default_source keeps
    refusing every other compound expression with an exact selector."""
    return _EXACT_OR_ORDER_TERMS.sub(lambda match: match.group(1) + match.group(2), source)


def advect_nvrtc_key() -> str | None:
    """``"major.minor"`` of the NVRTC this process compiles with, or None.

    The production advection compile allows multiply-add contraction, so a
    receipt's words follow the NVRTC build as well as the card.  The query
    lives in this module, as :func:`advect_gpu_identity` does, so the parity test module
    holds no device import of its own: a module-level or helper import would
    mark its CPU fixture rows GPU (tests/test_gpu_pin_set.py).
    """
    try:
        from cupy.cuda import nvrtc
        major, minor = nvrtc.getVersion()
    except Exception:  # noqa: BLE001 - no CuPy NVRTC: no compiler receipt applies
        return None
    return f"{int(major)}.{int(minor)}"
