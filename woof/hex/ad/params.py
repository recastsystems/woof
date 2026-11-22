# SPDX-License-Identifier: Apache-2.0
"""Physics constants as differentiable inputs: the table, the runner's parameter file, objectives and results.

hex's adjoint can differentiate a forecast metric with respect to selected physics constants, not only the initial
state. Each constant below is a literal in one scheme's generated CUDA source. The parameter carrier
(:mod:`woof.hex.ad.carrier`) rewrites that one occurrence so that

* the forward reads ``on[k] ? value[k] : literal`` (module globals, off by default, so the default build does the
  same float operation on the same value: gradients off is bitwise the shipped forward);
* the tangent-linear copy carries a tangent for constant ``k``;
* the reverse copy makes the constant a tape leaf and adds its cotangent per column; the host sums the columns in
  index order in float64, so the gradient is deterministic run to run and card to card.

The window runner reads a parameter file (:func:`carrier_document`) with ``--params`` and reports, per constant,
``dJ/dp`` and ``dJ/d ln p`` (:func:`gradients_from_receipt`). ``--param-tl c`` runs the parameter tangent along
``d ln p = c`` so the tangent and the adjoint can be checked against each other (:func:`param_dot`).

The forward can also be run at chosen values of the same constants (:mod:`woof.hex.mpasphys.ad_params_override`,
environment ``WOOF_HEX_AD_PARAMS``; unset, nothing is touched).

Upstream references are to the scheme sources hex ports (NOAA GSL RRFS physics: MYNN surface layer v6.0.0,
MYNN-EDMF, GSL drag), cited by routine.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCHEMA = "woof.hex.ad.params/v1"

#: The carrier's slots per family (``P4_NPAR`` / ``MW_NPAR`` in the generated sources).
SLOTS_PER_FAMILY = 8
#: The carrier's per-column cotangent rows (``P4_PAR_COLS`` / ``MW_PAR_COLS``); a mesh with more cells sets the
#: overflow flag and the runner refuses the gradient.
PARAM_COLUMNS = 4096


@dataclass(frozen=True)
class Constant:
    """One physics constant the carrier can differentiate."""

    name: str
    scheme: str            #: hex physics slot and scheme the constant belongs to
    family: str            #: carrier family: "mynnsfc" | "ugwp" (P4 modules) | "mynnedmf" (P3 MYNN-EDMF)
    slot: int              #: carrier slot inside the family, 0..7
    default: float         #: the shipped value (exactly the float32 literal in the source)
    literal: str           #: the literal as it appears in the generated CUDA source
    units: str
    description: str
    upstream: str          #: upstream routine and published method the literal comes from
    sites: tuple[str, ...] = field(default=())   #: verbatim forward patterns (each must occur exactly once)
    derivative_prefix: str = ""   #: prefix of each site's derivative copy in the P4 sources ("" or "adx_")

    @property
    def bits(self) -> str:
        return float32_bits(self.default)

    def row(self) -> dict:
        return {"name": self.name, "scheme": self.scheme, "family": self.family, "slot": self.slot,
                "default": self.default, "p0_bits": self.bits, "units": self.units,
                "description": self.description, "upstream": self.upstream}


def float32_bits(value: float) -> str:
    """The float32 bit pattern of ``value`` as the runner's ``p0_bits`` string (``0x3c343958``)."""
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", float(value)))[0]


def float32_from_bits(bits: str | int) -> float:
    word = int(str(bits), 16) if isinstance(bits, str) else int(bits)
    return struct.unpack("<f", struct.pack("<I", word))[0]


def float32_literal(lit: str) -> float:
    """The float32 value of a CUDA literal (``0x1.6872b00000000p-7f``, ``8.84e-6f``, ``24.0f``)."""
    text = lit[:-1] if lit.endswith("f") else lit
    value = float.fromhex(text) if "0x" in text else float(text)
    return struct.unpack("<f", struct.pack("<f", value))[0]


#: The constants, in carrier order. Defaults are the shipped float32 values bit for bit.
CONSTANTS: tuple[Constant, ...] = (
    Constant(
        name="coare30_charnock", scheme="surface_layer: MYNN surface layer (RRFS v6.0.0)", family="mynnsfc", slot=0,
        default=float32_literal("0x1.6872b00000000p-7f"), literal="0x1.6872b00000000p-7f", units="1 (dimensionless)",
        description="COARE 3.0 Charnock base coefficient over water: czc = 0.011 + 0.007 clamp((U10 - 10) / 8)",
        upstream="module_sf_mynnsfc_water, water_coare30_znt (COARE 3.0, Fairall et al. 2003); "
                 "used with sf_mynn_sfcflux_water = 0",
        sites=("czc = __fadd_rn(0x1.6872b00000000p-7f,__fmul_rn(0x1.cac0840000000p-8f,",)),
    Constant(
        name="mynnsfc_zil_kczil", scheme="surface_layer: MYNN surface layer (RRFS v6.0.0)", family="mynnsfc", slot=1,
        default=float32_literal("0x1.99999a0000000p-2f"), literal="0x1.99999a0000000p-2f", units="1 (dimensionless)",
        description="Zilitinkevich land thermal roughness: the factor k multiplying czil in "
                    "z0t = z0 exp(-k czil sqrt(Re*)), i.e. the z0t/z0 exponent scale (both land call sites)",
        upstream="module_sf_mynnsfc_land, land_zilitinkevich_1995 (Zilitinkevich 1995; Chen et al. 1997)",
        sites=("m_land_zilitinkevich_1995(zntstoch,zt,zq,restar,ust,0x1.99999a0000000p-2f,sf_mynn_sfcflux_land,",
               "m_land_zilitinkevich_1995(zntstoch,zt,zq,restar,ust,0x1.99999a0000000p-2f,0,lsm,lsm_ruc,true);"),
        derivative_prefix="adx_"),
    Constant(
        name="tofd_beljaars_a1", scheme="gwdo: GSL orographic drag (small-scale)", family="ugwp", slot=0,
        default=float32_literal("0.00026615161f"), literal="0.00026615161f",
        units="m^-1 per m^2 of subgrid orographic variance (a1 = coefficient * var^2)",
        description="Turbulent orographic form drag prefactor: a1 = 0.00026615161 * var^2",
        upstream="bl_ugwp, TOFD block (Beljaars, Brown and Wood 2004)",
        sites=("a1=0.00026615161f*(var_temp*var_temp)",)),
    Constant(
        name="mesoscale_gwd_xlinv", scheme="gwdo: GSL orographic drag (mesoscale)", family="ugwp", slot=1,
        default=float32_literal("8.84e-6f"), literal="8.84e-6f", units="m^-1",
        description="Mesoscale orographic gravity-wave drag base-stress coefficient: xlinv = coefm * 8.84e-6",
        upstream="bl_ugwp, mesoscale GWD block (Kim and Arakawa 1995; Choi and Hong 2015)",
        sites=("xlinv=coefm*8.84e-6f;",)),
    Constant(
        name="flow_blocking_cd", scheme="gwdo: GSL orographic drag (flow blocking)", family="ugwp", slot=2,
        default=float32_literal("1.92e-4f"), literal="1.92e-4f", units="m^-1",
        description="Flow-blocking stress coefficient: taufb = 0.5 * 1.92e-4 * rhobar * coefm * cd * olp * zblk * ulow^2",
        upstream="bl_ugwp, flow-blocking block (Lott and Miller 1997; Kim and Doyle 2005)",
        sites=("taufb[1]=0.5f*1.92e-4f*rhobar",)),
    Constant(
        name="mynn_qke_dissipation_b1", scheme="pbl: MYNN-EDMF (RRFS)", family="mynnedmf", slot=0,
        default=24.0, literal="24.0f", units="1 (dimensionless)",
        description="MYNN level-2.5 B1 in the TKE dissipation of the qke prediction only: "
                    "b1l = B1 * 0.25 * (el(k+1) + el(k)); the closure's B1 (stability functions) stays the constant",
        upstream="module_bl_mynn, mym_predict (Nakanishi and Niino 2009: B1 = 24)",
        sites=("b1l = r6_mul(r6_mul(b1, 0.25f), r6_add(el(((k) + (1))), el(k)));",)),
)

BY_NAME: dict[str, Constant] = {c.name: c for c in CONSTANTS}
FAMILIES: tuple[str, ...] = ("mynnsfc", "ugwp", "mynnedmf")


def _check_table() -> None:
    seen = set()
    for c in CONSTANTS:
        if c.family not in FAMILIES:
            raise AssertionError(f"{c.name}: family {c.family}")
        if not 0 <= c.slot < SLOTS_PER_FAMILY or (c.family, c.slot) in seen:
            raise AssertionError(f"{c.name}: slot {c.slot}")
        seen.add((c.family, c.slot))
        if float32_from_bits(c.bits) != float32_literal(c.literal):
            raise AssertionError(f"{c.name}: default {c.default!r} is not the source literal {c.literal}")
        for s in c.sites:
            if c.family != "mynnedmf" and s.count(c.literal) != 1:
                raise AssertionError(f"{c.name}: literal not unique in site {s!r}")


_check_table()


def table() -> list[dict]:
    """The constants table as rows (name, scheme, family, slot, default, p0_bits, units, description, upstream)."""
    return [c.row() for c in CONSTANTS]


def select(names: Iterable[str] | None = None) -> list[Constant]:
    """The named constants in table order (all of them when ``names`` is None); an unknown name is refused."""
    if names is None:
        return list(CONSTANTS)
    wanted = [n.strip() for n in names if n and n.strip()]
    unknown = sorted(set(wanted) - set(BY_NAME))
    if unknown:
        raise ValueError(f"unknown physics constants {unknown}; known: {', '.join(BY_NAME)}")
    if not wanted:
        raise ValueError("no physics constants named")
    return [c for c in CONSTANTS if c.name in wanted]


def carrier_document(names: Iterable[str] | None = None, values: Mapping[str, float] | None = None,
                     title: str = "physics constants") -> dict:
    """The runner's ``--params`` file: one row per constant with its family, slot and float32 value bits.

    ``values`` moves a constant away from its default. The forward then runs at that value and no longer matches the
    recording's boundary hashes, so the runner needs ``--params-unchecked 1`` (``runner_arguments(unchecked=True)``).
    """
    values = dict(values or {})
    chosen = select(names if names is not None else (list(values) or None))
    unknown = sorted(set(values) - {c.name for c in chosen})
    if unknown:
        raise ValueError(f"values for constants not selected: {unknown}")
    rows = []
    for c in chosen:
        v = float(values.get(c.name, c.default))
        if not math.isfinite(v):
            raise ValueError(f"{c.name}: value {v!r} is not finite")
        rows.append({"name": c.name, "family": c.family, "slot": c.slot, "p0_bits": float32_bits(v),
                     "default_bits": c.bits, "units": c.units})
    return {"schema": SCHEMA, "name": title, "params": rows}


def write_carrier_document(path: str | Path, names: Iterable[str] | None = None,
                           values: Mapping[str, float] | None = None) -> dict:
    doc = carrier_document(names, values)
    Path(path).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def moved(doc: Mapping) -> bool:
    """True when any row of a parameter file is away from its default (the forward is then not the recorded one)."""
    return any(r["p0_bits"] != BY_NAME[r["name"]].bits for r in doc["params"])


def runner_arguments(params_file: str | Path, objectives: Sequence[str | Path], *, tangent_scale: float | None = 1e-12,
                     unchecked: bool | None = None) -> list[str]:
    """The window runner's parameter-gradient arguments (append to a window command that has ``--native-schemes 1``).

    One reverse per objective (``--grad-all 1``). With ``tangent_scale``, the parameter tangent also runs along
    ``d ln p = tangent_scale`` for every constant and the receipt carries the tangent-vs-adjoint check. The scale is
    small because the tangent is a float32 state: ``d ln p = 1`` overflowed float32 at step 8 of 18 on the x1.2562
    hrrrv5 window; the check is linear in the scale, so it cancels. ``unchecked`` (default: whether the file moves a
    constant) lets the forward run at moved values.
    """
    if not objectives:
        raise ValueError("at least one objective file is needed")
    doc = json.loads(Path(params_file).read_text(encoding="utf-8"))
    args = ["--params", str(params_file), "--objective", ",".join(str(o) for o in objectives), "--grad-all", "1",
            "--zero", "0", "--tangent-only", "1"]
    if tangent_scale is not None:
        args += ["--param-tl", ",".join(repr(float(tangent_scale)) for _ in doc["params"])]
    if unchecked if unchecked is not None else moved(doc):
        args += ["--params-unchecked", "1"]
    return args


# ---------------------------------------------------------------- objectives

def field_view(alloc_map: Mapping, name: str) -> tuple[int, int, list[int]]:
    """(allocation, word offset, shape) of a named field in a recording's allocation map (both map layouts)."""
    fields = alloc_map.get("fields", {})
    if name not in fields:
        raise KeyError(f"field {name!r} is not in this recording's allocation map (has: {', '.join(sorted(fields))})")
    f = fields[name]
    if "canonical" in f:
        c = f["canonical"]
        alloc, offset, shape = int(c["alloc"]), int(c["offset"]), list(c["shape"])
    else:
        alloc, offset, shape = int(f["allocation"]), int(f.get("offset", 0)), list(f["shape"])
    if alloc is None or offset % 4:
        raise ValueError(f"{name}: allocation {alloc}, byte offset {offset}: not a whole float32 view")
    return alloc, offset // 4, shape


def mean_objective(alloc_map: Mapping, field_name: str, levels: Sequence[int], cells: Sequence[int] | None,
                   name: str | None = None, units: str = "") -> dict:
    """A linear objective: the mean of a level x cell field over 1-based model ``levels`` and ``cells`` at the window end.

    ``cells`` None means every cell. The terms are ``[allocation, word, weight]`` with
    word = offset + (level - 1) * ncell + cell, the runner's objective form.
    """
    alloc, base, shape = field_view(alloc_map, field_name)
    if len(shape) != 2:
        raise ValueError(f"{field_name}: a level x cell field is needed, shape {shape}")
    nlev, ncell = shape
    cells = list(range(ncell)) if cells is None else [int(c) for c in cells]
    levels = [int(k) for k in levels]
    if not cells or not levels:
        raise ValueError("empty objective")
    if min(levels) < 1 or max(levels) > nlev or min(cells) < 0 or max(cells) >= ncell:
        raise ValueError(f"{field_name}: levels or cells outside {nlev} x {ncell}")
    w = 1.0 / (len(levels) * len(cells))
    terms = [[alloc, base + (k - 1) * ncell + c, w] for k in levels for c in cells]
    label = name or (f"mean {field_name}, model levels {levels[0]}-{levels[-1]}, {len(cells)} cells, window end"
                     + (f"; {units}" if units else ""))
    return {"name": label, "terms": terms, "field": field_name, "levels_1based": levels, "cells": cells}


def cells_within(lat_deg: Sequence[float], lon_deg: Sequence[float], lat0: float, lon0: float,
                 radius_km: float) -> list[int]:
    """Indices of cells within ``radius_km`` (great circle, R = 6371 km) of (lat0, lon0) in degrees."""
    a, b = math.radians(lat0), math.radians(lon0)
    out = []
    for i, (la, lo) in enumerate(zip(lat_deg, lon_deg)):
        p, q = math.radians(la), math.radians(lo)
        c = math.sin(a) * math.sin(p) + math.cos(a) * math.cos(p) * math.cos(q - b)
        if 6371.0 * math.acos(max(-1.0, min(1.0, c))) <= radius_km:
            out.append(i)
    return out


def parse_objective_spec(spec: str) -> dict:
    """``FIELD:K0-K1[:LAT,LON,RADIUS_KM]`` -> {"field", "levels", "region"}: the door's objective option.

    Example: ``theta_m:1-8`` (global mean of theta_m over model levels 1-8 at the window end) or
    ``theta_m:3-12:20.4,-26.2,1500`` (the same over the cells within 1,500 km of 20.4N 26.2W).
    """
    parts = spec.split(":")
    if len(parts) not in (2, 3) or not parts[0]:
        raise ValueError(f"objective {spec!r}: want FIELD:K0-K1[:LAT,LON,RADIUS_KM]")
    k = parts[1].split("-")
    levels = list(range(int(k[0]), int(k[-1]) + 1))
    region = None
    if len(parts) == 3:
        lat, lon, radius = (float(x) for x in parts[2].split(","))
        region = (lat, lon, radius)
    return {"field": parts[0], "levels": levels, "region": region}


def prepare(out_dir: str | Path, alloc_map: Mapping, objective_specs: Sequence[str], *,
            names: Iterable[str] | None = None, values: Mapping[str, float] | None = None,
            lat_deg: Sequence[float] | None = None, lon_deg: Sequence[float] | None = None,
            tangent_scale: float | None = 1e-12) -> list[str]:
    """Write the parameter file and the objectives into ``out_dir``; return the runner's parameter-gradient arguments.

    ``alloc_map`` is the recording's allocation map (``alloc-map.json`` beside its plan). A regional objective needs the
    mesh's cell latitudes and longitudes in degrees.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pfile = out / "params.json"
    write_carrier_document(pfile, names, values)
    files = []
    for i, spec in enumerate(objective_specs):
        s = parse_objective_spec(spec)
        cells = None
        if s["region"] is not None:
            if lat_deg is None or lon_deg is None:
                raise ValueError(f"objective {spec!r}: a region needs the mesh cell latitudes and longitudes")
            cells = cells_within(lat_deg, lon_deg, *s["region"])
            if not cells:
                raise ValueError(f"objective {spec!r}: no cell inside the region")
        o = mean_objective(alloc_map, s["field"], s["levels"], cells)
        o["spec"] = spec
        f = out / f"objective-{i}.json"
        f.write_text(json.dumps(o), encoding="utf-8")
        files.append(f)
    return runner_arguments(pfile, files, tangent_scale=tangent_scale)


def column_gradients(cols_dir: str | Path, ncell: int) -> dict[tuple[int, str], list[float]]:
    """Per-column cotangents dJ/dp written by the runner when ``AD_PARAM_COLS_DIR`` is set: where each constant matters.

    Keys are (objective index, constant name); values are the first ``ncell`` float64 columns (cell order).
    """
    out = {}
    for f in sorted(Path(cols_dir).glob("pcol-o*-*.f64")):
        head, name = f.stem.split("-", 2)[1:]
        data = f.read_bytes()
        n = len(data) // 8
        if ncell > n:
            raise ValueError(f"{f.name}: {n} columns, mesh has {ncell}")
        out[(int(head[1:]), name)] = list(struct.unpack(f"<{ncell}d", data[: 8 * ncell]))
    return out


# ---------------------------------------------------------------- results

def gradients_from_receipt(receipt: Mapping) -> list[dict]:
    """Per objective, per constant: dJ/dp, dJ/d ln p, units and the tangent-vs-adjoint check, from a window receipt."""
    out = []
    first = receipt.get("param_gradients")
    if first is not None:
        objs = (receipt.get("param_tl") or {}).get("all") or []
        out.append({"objective": (objs[0]["name"] if objs else None), "rows": _rows(first),
                    "dot": receipt.get("param_dot")})
    for more in receipt.get("param_gradients_more") or []:
        out.append({"objective": more.get("name"), "file": more.get("file"), "rows": _rows(more.get("param_gradients", [])),
                    "dot": more.get("param_dot"), "reverse_wall_s": more.get("reverse_wall_s")})
    return out


def _rows(rows: Iterable[Mapping]) -> list[dict]:
    res = []
    for r in rows:
        c = BY_NAME.get(r.get("name"))
        res.append({"name": r.get("name"), "p0": r.get("p0"), "dJ_dp": r.get("dJdp"), "dJ_dlnp": r.get("dJdlnp"),
                    "units": c.units if c else None, "nonzero_columns": r.get("nonzero_columns"),
                    "columns_sha256": r.get("columns_sha256")})
    return res


def param_dot(receipt: Mapping) -> list[dict]:
    """The tangent-vs-adjoint check per objective: tl_dJ against sum_k c_k p_k dJ/dp_k, relative difference, pass."""
    return [{"objective": g["objective"], **(g["dot"] or {})} for g in gradients_from_receipt(receipt)]


def read_receipt(out_dir: str | Path) -> dict:
    return json.loads((Path(out_dir) / "window.json").read_text(encoding="utf-8"))


def markdown_table(receipt: Mapping) -> str:
    """A plain table of the gradients and the check, for a run's report."""
    lines = []
    for g in gradients_from_receipt(receipt):
        lines += [f"objective: {g['objective']}", "", "| constant | p0 | dJ/dp | dJ/d ln p |", "|---|---|---|---|"]
        for r in g["rows"]:
            lines.append(f"| {r['name']} | {r['p0']:.6g} | {r['dJ_dp']:.6e} | {r['dJ_dlnp']:.6e} |")
        d = g["dot"] or {}
        if d:
            lines.append("")
            lines.append(f"tangent {d.get('tl_dJ'):.9e}, adjoint {d.get('adjoint_sum_c_dJdlnp'):.9e}, "
                         f"rel {d.get('rel'):.2e}, pass {d.get('pass')}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m woof.hex.ad.params``: print the constants table (``--json`` for JSON)."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--json" in argv:
        print(json.dumps(table(), indent=1))
        return 0
    for c in CONSTANTS:
        print(f"{c.name:26s} {c.family:9s} slot {c.slot}  default {c.default:.8g}  [{c.units}]  {c.scheme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
