"""Physics parameter sets: tunable constants as table rows, off by default.

A ``[physics_params]`` table in an experiment names a set of values for
constants registered in ``physics_params_registry_v1.json``.  Nothing else
can be set, and with no table kernel source strings and parameter tables
retain their default bytes. No parameter identity or output attribute is
added to a default forecast.

Two kinds of constant exist, each a registry row:

``kernel-literal``
    A float literal in a CUDA kernel.  The row names the exact source text
    that holds it (the *anchor*) and how many times that text occurs.  An
    active set rewrites the literal inside each anchor before the source
    reaches NVRTC; CuPy's compile cache is keyed by the source, so a set
    compiles its own kernels and never shares a cache entry with the
    default.  If an anchor does not occur exactly ``count`` times the set
    is refused: the kernel was edited and the constant no longer reaches
    the literal it names.
``table-multiplier``
    One column of one parameter table scaled for the listed categories,
    applied to the bundle AFTER its pinned-hash check (the pinned bytes are
    still verified; the edit is recorded beside them).

One process runs one set.  The set is fixed by the first experiment the
process parses; parsing an experiment with a different set later is
refused, because a kernel compiled under the first would run under the
second's name.

Every run that carries a set records its name and SHA-256 in the wrfout
globals (``WOOF_PHYSICS_PARAMS``, ``GPUWM_PHYSICS_PARAMS_SHA256``) and in
the experiment identity a checkpoint binds, so a resume under another set
refuses and a child built from the run's history can read which set made it.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

SCHEMA = "gpuwm.physics-params/v1"
REGISTRY_SCHEMA = "gpuwm.physics-params-registry/v1"
REGISTRY_PATH = Path(__file__).with_name("physics_params_registry_v1.json")

#: Keys a [physics_params] table may carry.
TABLE_KEYS = ("name", "values", "set", "knobs")

WRFOUT_NAME_ATTR = "WOOF_PHYSICS_PARAMS"
WRFOUT_SHA_ATTR = "GPUWM_PHYSICS_PARAMS_SHA256"


class PhysicsParamsError(ValueError):
    """A [physics_params] declaration that cannot run as declared."""


# --------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class KernelSite:
    module: str
    anchor: str
    literal: str
    count: int


@dataclass(frozen=True)
class TunableConstant:
    name: str
    scheme: str
    requires: Mapping[str, tuple[int, ...]]
    kind: str
    default: float
    lower: float
    upper: float
    units: str
    description: str
    sites: tuple[KernelSite, ...] = ()
    table: str | None = None
    column: str | None = None
    categories: Mapping[str, tuple[int, ...]] = field(
        default_factory=lambda: MappingProxyType({}))


def registry_bytes() -> bytes:
    return REGISTRY_PATH.read_bytes()


@lru_cache(maxsize=1)
def registry_sha256() -> str:
    return hashlib.sha256(registry_bytes().replace(b"\r\n", b"\n")).hexdigest()


@lru_cache(maxsize=1)
def registry() -> Mapping[str, TunableConstant]:
    """The registered constants by name, validated on first read."""
    document = json.loads(registry_bytes().decode("utf-8"))
    if document.get("schema") != REGISTRY_SCHEMA:
        raise PhysicsParamsError(
            f"{REGISTRY_PATH.name} schema {document.get('schema')!r}; "
            f"expected {REGISTRY_SCHEMA!r}")
    rows: dict[str, TunableConstant] = {}
    for row in document["constants"]:
        name = str(row["name"])
        if name in rows:
            raise PhysicsParamsError(f"duplicate registry row {name!r}")
        kind = str(row["kind"])
        default, lower, upper = (float(row["default"]), float(row["lower"]),
                                 float(row["upper"]))
        if not (lower <= default <= upper) or not lower < upper:
            raise PhysicsParamsError(
                f"registry row {name!r}: default {default} outside "
                f"[{lower}, {upper}]")
        requires = MappingProxyType({
            str(key): tuple(int(item) for item in values)
            for key, values in dict(row.get("requires", {})).items()})
        if kind == "kernel-literal":
            sites = tuple(
                KernelSite(module=str(site["module"]),
                           anchor=str(site["anchor"]),
                           literal=str(site["literal"]),
                           count=int(site["count"]))
                for site in row["sites"])
            for site in sites:
                if site.anchor.count(site.literal) != 1:
                    raise PhysicsParamsError(
                        f"registry row {name!r}: literal {site.literal!r} "
                        f"must occur once in its anchor {site.anchor!r}")
                if _literal_value(site.literal) != _float32(default):
                    raise PhysicsParamsError(
                        f"registry row {name!r}: literal {site.literal!r} is "
                        f"not the registered default {default}")
            rows[name] = TunableConstant(
                name=name, scheme=str(row["scheme"]), requires=requires,
                kind=kind, default=default, lower=lower, upper=upper,
                units=str(row["units"]),
                description=str(row["description"]), sites=sites)
        elif kind == "table-multiplier":
            if default != 1.0:
                raise PhysicsParamsError(
                    f"registry row {name!r}: a multiplier's default is 1")
            rows[name] = TunableConstant(
                name=name, scheme=str(row["scheme"]), requires=requires,
                kind=kind, default=default, lower=lower, upper=upper,
                units=str(row["units"]),
                description=str(row["description"]),
                table=str(row["table"]), column=str(row["column"]),
                categories=MappingProxyType({
                    str(section): tuple(int(c) for c in cats)
                    for section, cats in dict(row["categories"]).items()}))
        else:
            raise PhysicsParamsError(
                f"registry row {name!r}: unknown kind {kind!r}")
    return MappingProxyType(rows)


def _float32(value: float) -> float:
    return float(np.float32(value))


def _literal_value(literal: str) -> float:
    text = literal[:-1] if literal.endswith("f") else literal
    return _float32(float(text))


def c_float_literal(value: float) -> str:
    """The shortest C spelling that parses to exactly ``float32(value)``."""
    text = str(np.float32(value))
    if not any(ch in text for ch in ".eE"):
        text += ".0"
    return text + "f"


# --------------------------------------------------------------------------- sets
@dataclass(frozen=True)
class PhysicsParamSet:
    """A named set of registered constants.  ``values`` is sorted by name
    and every value is already rounded to float32, which is what the kernel
    and the device tables read."""

    name: str
    values: tuple[tuple[str, float], ...]

    def value(self, key: str) -> float:
        for name, value in self.values:
            if name == key:
                return value
        return _float32(registry()[key].default)

    def identity(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "name": self.name,
            "values": {name: value for name, value in self.values},
            "registry_sha256": registry_sha256(),
        }

    def sha256(self) -> str:
        encoded = json.dumps(self.identity(), sort_keys=True,
                             separators=(",", ":")).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def changed(self) -> tuple[str, ...]:
        """Names whose value differs from the registered default."""
        rows = registry()
        return tuple(name for name, value in self.values
                     if value != _float32(rows[name].default))


def make_set(name: str, values: Mapping[str, object], *,
             source: str = "physics_params") -> PhysicsParamSet:
    """Validate ``values`` against the registry and build the set."""
    if not isinstance(name, str) or not name.strip():
        raise PhysicsParamsError(
            f"[physics_params] of {source} must carry name = \"...\": the "
            "set's name is what every output of the run is stamped with")
    if not isinstance(values, Mapping) or not values:
        raise PhysicsParamsError(
            f"[physics_params] of {source} names no constant; an empty set "
            "would stamp a run as tuned while running every default. Remove "
            "the table for a default run.")
    rows = registry()
    resolved: dict[str, float] = {}
    for key, raw in values.items():
        key = str(key)
        if key not in rows:
            from woof.experiment import did_you_mean
            raise PhysicsParamsError(
                f"[physics_params] of {source}: {key!r} is not a registered "
                f"constant{did_you_mean(key, tuple(rows))}. A misspelled "
                "constant would run its default under the set's name. "
                f"Registered: {sorted(rows)}.")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise PhysicsParamsError(
                f"[physics_params] of {source}: {key} = {raw!r} is not a "
                "number")
        value = float(raw)
        row = rows[key]
        in_range = (row.lower <= value <= row.upper
                    or value in (_float32(row.lower), _float32(row.upper)))
        if not math.isfinite(value) or not in_range:
            raise PhysicsParamsError(
                f"[physics_params] of {source}: {key} = {value!r} is outside "
                f"its registered range [{row.lower}, {row.upper}] "
                f"({row.units}). The scheme was not written for values past "
                "that range (a Prandtl number at or below zero makes the "
                "heat diffusivity negative, a roughness multiplier of zero "
                "takes the log of zero); widen the registry row with "
                "evidence rather than the set.")
        resolved[key] = _float32(value)
    return PhysicsParamSet(name=name.strip(),
                           values=tuple(sorted(resolved.items())))


def parse_table(table, *, source: str,
                base_dir: Path | None = None,
                _set_paths: tuple[Path, ...] = ()) -> PhysicsParamSet | None:
    """``[physics_params]`` of an experiment: ``name`` and ``values``, or
    ``set = "file.toml"`` naming a set file that carries both."""
    if not isinstance(table, Mapping):
        raise PhysicsParamsError(
            f"[physics_params] of {source} must be a table, got {table!r}")
    unknown = sorted(set(table) - set(TABLE_KEYS))
    if unknown:
        raise PhysicsParamsError(
            f"[physics_params] of {source}: unknown key(s) {unknown}; known "
            f"{list(TABLE_KEYS)}. No key is ignored, because an ignored key "
            "runs the default under the set's name.")
    if "knobs" in table:
        from woof.knobs.physics import parse_parameter_table
        return parse_parameter_table(table, source=source)
    if "set" in table:
        if "values" in table:
            raise PhysicsParamsError(
                f"[physics_params] of {source} carries both set = and "
                "values; one source of values, so the two cannot disagree")
        path = Path(str(table["set"]))
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        path = path.resolve()
        if path in _set_paths:
            raise PhysicsParamsError(
                f"[physics_params] of {source}: cyclic set-file reference "
                f"at {path}; no constants can be resolved")
        try:
            document = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise PhysicsParamsError(
                f"[physics_params] of {source}: cannot read set file "
                f"{path}: {exc}") from None
        inner = document.get("physics_params", document)
        pset = parse_table(inner, source=f"{source} (set file {path})",
                           base_dir=path.parent, _set_paths=(*_set_paths, path))
        resolved_name = None if pset is None else pset.name
        if "name" in table and table["name"] != resolved_name:
            raise PhysicsParamsError(
                f"[physics_params] of {source} names the set "
                f"{table['name']!r} but {path} is {resolved_name!r}")
        return pset
    return make_set(table.get("name"), table.get("values", {}), source=source)


#: A set file a forecast process runs under without the set being written
#: into its experiment file.  This is the door for many members sharing ONE
#: preparation: a prepared root is bound to the exact bytes of the
#: experiment file it was prepared from, and a set is forecast-time only (no
#: preparation step reads a registered constant), so the members run the
#: same file and differ only in this variable.  An experiment that carries
#: its own [physics_params] table must carry the same set.
ENV_VAR = "WOOF_PHYSICS_PARAMS"


def environment_set() -> PhysicsParamSet | None:
    """The set named by :data:`ENV_VAR`, or ``None`` when it is unset."""
    import os
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        return None
    path = Path(raw)
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PhysicsParamsError(
            f"{ENV_VAR}={raw}: cannot read the set file ({exc}); a run that "
            "asked for a set must not start on the defaults") from None
    inner = document.get("physics_params", document)
    return parse_table(inner, source=f"{ENV_VAR}={raw}", base_dir=path.parent)


def document(pset: PhysicsParamSet | None) -> dict[str, object] | None:
    """The set as the experiment document and the receipts carry it."""
    if pset is None:
        return None
    return dict(pset.identity(), sha256=pset.sha256())


def check_schemes(pset: PhysicsParamSet, run_configs, *, source: str) -> None:
    """Refuse a set whose constants act on no domain of the experiment."""
    runs = list(run_configs)
    rows = registry()
    for name in pset.changed():
        row = rows[name]
        if not any(all(int(getattr(run, key, -1)) in allowed
                       for key, allowed in row.requires.items())
                   for run in runs):
            wanted = ", ".join(f"{key} in {list(allowed)}"
                               for key, allowed in row.requires.items())
            raise PhysicsParamsError(
                f"[physics_params] of {source} sets {name} ({row.scheme}), "
                f"but no domain runs that scheme ({wanted}). The run would "
                "be stamped with a set whose constant never acted.")


# --------------------------------------------------------------------------- process binding
_STATE: dict[str, object] = {
    "declared": False,       # has any experiment been parsed in this process
    "active": None,          # the process's PhysicsParamSet, or None
    "compiled": {},          # kernel module -> set sha256 ("" for default)
    "tables_issued": False,  # a forecast built parameter tables
}


def _affected_modules() -> frozenset[str]:
    return frozenset(site.module for row in registry().values()
                     for site in row.sites)


def active() -> PhysicsParamSet | None:
    return _STATE["active"]  # type: ignore[return-value]


def _active_sha() -> str:
    pset = active()
    return "" if pset is None else pset.sha256()


def declare(pset: PhysicsParamSet | None, *, source: str) -> None:
    """Bind the set an experiment declares to this process.

    Called for every experiment the process parses.  The first binds; each
    later one must carry the same set.  A switch is refused, and so is a
    first set arriving after the affected kernels or tables were already
    built with the defaults.
    """
    current = active()
    same = ((current is None and pset is None)
            or (current is not None and pset is not None
                and current.sha256() == pset.sha256()))
    if _STATE["declared"] and same:
        return
    if _STATE["declared"] and current is not None:
        raise PhysicsParamsError(
            f"{source} declares physics parameter set "
            f"{'none' if pset is None else repr(pset.name)} but this process "
            f"already runs set {current.name!r}. One process runs one set: "
            "a kernel compiled under the first would run under the second's "
            "name. Run the second experiment in its own process.")
    if pset is not None:
        built = sorted(module for module, sha in dict(
            _STATE["compiled"]).items()
            if module in _affected_modules() and sha != pset.sha256())
        if built or _STATE["tables_issued"]:
            raise PhysicsParamsError(
                f"{source} declares physics parameter set {pset.name!r} after "
                "this process built "
                + (f"the kernels {built}" if built else "its parameter tables")
                + " with the default constants; they would run under the "
                "set's name. Declare the set before the forecast starts.")
    _STATE["declared"] = True
    _STATE["active"] = pset


def reset_for_tests() -> None:
    """Forget the process binding (tests only: a run never unbinds)."""
    _STATE["declared"] = False
    _STATE["active"] = None
    _STATE["compiled"] = {}
    _STATE["tables_issued"] = False


# --------------------------------------------------------------------------- kernel source
def edit_kernel_source(module: str, text: str, *, value_expressions=None) -> str:
    """``text`` with the active set's literals written in.

    With no active set, or a set that edits nothing in ``module``, the
    input string is returned unchanged (the same object).
    """
    pset = active()
    if pset is None:
        return text
    edits = []
    for name in pset.changed():
        row = registry()[name]
        # Family-owned source transformations retain sites in the registry
        # for validation and process binding, and apply their own edits.
        if row.kind == "family-source":
            continue
        for site in row.sites:
            if site.module == module:
                edits.append((row, site))
    if not edits:
        return text
    for row, site in edits:
        found = text.count(site.anchor)
        if found != site.count:
            raise PhysicsParamsError(
                f"physics parameter {row.name}: kernel {module}.cu holds "
                f"{found} copies of its anchor {site.anchor!r}, the registry "
                f"expects {site.count}. The kernel changed and the constant "
                "no longer reaches the literal it names; update the registry "
                "row against the new source.")
        expression = ((value_expressions or {}).get(row.name)
                      or c_float_literal(pset.value(row.name)))
        replacement = site.anchor.replace(site.literal, expression)
        text = text.replace(site.anchor, replacement)
    return text


def note_compiled(module: str, compiled_module=None) -> None:
    """Record which set a kernel module was compiled under."""
    from woof.kernel_plan import refuse_while_capturing
    refuse_while_capturing()  # a static-plan capture compiled nothing for the run
    binder = getattr(edit_kernel_source, "bind_module", None)
    if binder is not None and compiled_module is not None:
        binder(module, compiled_module)
    compiled = dict(_STATE["compiled"])
    compiled[module] = _active_sha()
    _STATE["compiled"] = compiled


def validate_kernel_sites(read_source) -> list[str]:
    """Every anchor's occurrence count in today's sources; the problems.

    ``read_source(module)`` returns the ``.cu`` text.  Used by the tests
    and by ``woof check`` style tooling to prove the registry still
    reaches every literal it names.
    """
    problems = []
    for row in registry().values():
        for site in row.sites:
            found = read_source(site.module).count(site.anchor)
            if found != site.count:
                problems.append(
                    f"{row.name}: {site.module}.cu anchor found {found}x, "
                    f"expected {site.count}x: {site.anchor!r}")
    return problems


# --------------------------------------------------------------------------- tables
#: ``ruc.cu`` ``ruc_surface_parameters``: a seasonal-crop class (IFOR 7) gets
#: roughness ``z0tbl - 0.125 * factor`` with ``factor`` in [0, 1].  A Z0
#: multiplier that takes such a class to this value or below gives it a
#: non-positive roughness and a NaN surface drag.
SEASONAL_CROP_Z0_DECREMENT = 0.125


def ruc_bundle_for_forecast(load_default, scale_values=None):
    """The RUC parameter bundle a forecast should run, or ``None``.

    ``None`` (no set, or a set that scales no RUC column) tells the caller
    to load the default bundle exactly as it does without this module.
    ``load_default`` is :func:`woof.core.ruc.load_ruc_parameters`, passed
    by the caller rather than imported here: this module is reachable from
    the experiment parser, which the RW-WPS wheel stages without the RUC
    tables, and its staging refuses an internal import it does not carry.
    """
    _STATE["tables_issued"] = True
    pset = active()
    if pset is None:
        return None
    rows = [registry()[name] for name in pset.changed()
            if registry()[name].table == "ruc-vegparm"]
    if not rows:
        return None
    return apply_ruc_edits(load_default(), pset, scale_values)


def apply_ruc_edits(bundle, pset: PhysicsParamSet, scale_values=None):
    """Select registered table cells; the supplied GPU driver scales them.

    The parser stages without the numerical runtime. The forecast passes
    its driver here, and no driver or table is read when no row changes.
    """
    rows = [registry()[name] for name in pset.changed()
            if registry()[name].table == "ruc-vegparm"]
    if not rows:
        return bundle
    if scale_values is None:
        raise PhysicsParamsError(
            "RUC parameter edits need the forecast's GPU scaling driver; "
            "otherwise the named constants would remain at their defaults")
    sections = dict(bundle.vegetation)
    selected = []
    cells = set()
    for section_name, table in sections.items():
        for row in rows:
            for category in row.categories.get(section_name, ()):
                cell = (section_name, category, row.column)
                if cell in cells:
                    raise PhysicsParamsError(
                        f"physics parameter rows overlap at {cell}; one table "
                        "cell cannot receive two independently named edits")
                cells.add(cell)
                old = table.rows[category - 1]
                selected.append((section_name, category, row, old))
    values = [float(getattr(old, row.column)) for _, _, row, old in selected]
    factors = [pset.value(row.name) for _, _, row, _ in selected]
    crop_flags = [row.column == "z0" and int(old.ifor) == 7
                  for _, _, row, old in selected]
    try:
        scaled = scale_values(values, factors, crop_flags)
    except ValueError as exc:
        raise PhysicsParamsError(
            f"physics parameter set {pset.name!r}: {exc}; non-positive "
            "roughness or a seasonal-crop roughness at or below 0.125 m "
            "causes a NaN surface drag after the RUC crop decrement. "
            "Leave seasonal-crop categories out of roughness rows.") from None
    if len(scaled) != len(selected):
        raise PhysicsParamsError("GPU parameter driver returned an incomplete table edit")
    edits = []
    new_rows = {section: list(table.rows) for section, table in sections.items()}
    for (section, category, row, old), before, factor, after in zip(
            selected, values, factors, scaled):
        current = new_rows[section][category - 1]
        new_rows[section][category - 1] = replace(current, **{row.column: after})
        edits.append({"section": section, "category": category,
                      "column": row.column, "constant": row.name,
                      "multiplier": factor, "before": before, "after": after})
    for section, table in sections.items():
        sections[section] = replace(table, rows=tuple(new_rows[section]))
    receipt = dict(bundle.receipt)
    receipt["physics_params"] = MappingProxyType({
        "set": pset.name, "sha256": pset.sha256(),
        "edits": tuple(MappingProxyType(item) for item in edits)})
    return replace(bundle, vegetation=MappingProxyType(sections),
                   receipt=MappingProxyType(receipt))


# --------------------------------------------------------------------------- records
def wrfout_global_attrs() -> dict[str, str]:
    """The wrfout globals of the active set; ``{}`` without one."""
    pset = active()
    if pset is None:
        return {}
    return {WRFOUT_NAME_ATTR: pset.name, WRFOUT_SHA_ATTR: pset.sha256()}


def receipt() -> dict[str, object] | None:
    return document(active())
