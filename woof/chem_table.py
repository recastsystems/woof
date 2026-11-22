"""The chem species table: rows, sets, sources and table diagnostics.

Every chem species WOOF-ArWen can carry is a ROW in a JSON file under
``woof/data/chem/species/``; every named species set is a file under
``sets/``; every external emission, boundary, oxidant or static source is a
file under ``sources/``; every output diagnostic built by a term sum is a file
under ``diagnostics/``.  All of them validate against
``woof/data/chem/schema.v1.json``.  Adding a species, a set, a source or a
table diagnostic is therefore data, never a code path -- the arbitrary
acceptance test -- and nothing in the engine tests a species name.

A PROCESS (dust emission, plume rise, deposition, vertical mixing, ...) is
code, written once, keyed by a process key in :data:`CHEM_PROCESS_MODULES`.
A row names the processes that act on it in its ``processes`` column; a
process reads its parameters from the rows that name it.  A key whose module
is absent from this build is refused at the configuration door by
:func:`woof.config.validate_chem_config` ("process X is not in this
build"), and that refusal retires itself the day the module lands.

This module is CuPy-free: configuration validation, the offline child, the
product inventory and the GUI read it without a device.

Set membership lives on the ROW (its ``sets`` column).  A set FILE carries
the set's own facts (description, the WRF namelist values that map onto it)
so that namelist import is table-driven too.  A row belongs to every set it
names; a set with no rows is refused, as is a row naming a set with no file.

Row order.  The active rows of a run are ordered by first appearance while
walking ``cfg.chem_sets`` in the order given and, inside each set, the
species files in sorted file name order and the rows in file order.  The
order fixes the arena slot of each species and nothing else: transport and
every per-row process are independent per species.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

__all__ = [
    "CHEM_DATA_ROOT", "CHEM_PROCESS_MODULES", "CHEM_STATE_PREFIX",
    "CHEM_TIME_PREFIX", "ChemTableError", "SpeciesRow", "SetRow", "SourceRow",
    "DiagnosticRow", "ChemCatalog", "ChemTable", "catalog", "chem_names",
    "load",
    "load_sets", "process_in_build", "state_attr", "time_attr",
]

#: Where the table lives inside the package.
CHEM_DATA_ROOT = Path(__file__).resolve().parent / "data" / "chem"

#: DomainState attribute prefixes: ``chem_<name>`` is a species' current
#: field (restart class ``serialize``), ``chem0_<name>`` its RK time-t copy
#: (``rebuild``, like ``qv0``).  Defined with the serialization contract;
#: see its comment for why they are prefixes.
from woof.state_serialization_contract import (  # noqa: E402
    CHEM_STATE_PREFIX, CHEM_TIME_PREFIX)

#: Process key -> the module that implements it, imported lazily.  A process
#: is code; which ROWS it acts on is data.  Every key the v1 design names is
#: listed here from the start so that no lane edits this table: a key whose
#: module is absent is refused as "not in this build" and retires itself when
#: the module lands.  WRF order of the chem step is in woof.core.chem_driver.
CHEM_PROCESS_MODULES: Mapping[str, str] = MappingProxyType({
    "emission.sfire": "woof.core.chem_sfire",
    "emission.fire": "woof.core.chem_fire",
    "plumerise.freitas": "woof.core.chem_plumerise",
    "wetdep.ls": "woof.core.chem_wetdep",
    "emission.dust": "woof.core.chem_dust",
    "emission.seasalt": "woof.core.chem_seasalt",
    "emission.inventory": "woof.core.chem_inventory",
    "drydep.gocart": "woof.core.chem_drydep_aerosol",
    "settling.gocart": "woof.core.chem_settling",
    "chem.sulfur": "woof.core.chem_sulfur",
    "aging.gocart": "woof.core.chem_ageing",
    "optics.gocart": "woof.core.chem_optics",
    "coupling.thompson": "woof.core.chem_mp_coupling",
    "drydep.wesely": "woof.core.chem_drydep_gas",
    "mixing.vertmx": "woof.core.chem_vertmx",
})


class ChemTableError(ValueError):
    """A chem table file is malformed, or a selection cannot be satisfied."""


def state_attr(name: str) -> str:
    """DomainState attribute of species ``name``'s current field."""
    return CHEM_STATE_PREFIX + name


def time_attr(name: str) -> str:
    """DomainState attribute of species ``name``'s RK time-t copy."""
    return CHEM_TIME_PREFIX + name


def process_in_build(key: str) -> bool:
    """True when the module implementing process ``key`` ships in this build."""
    module = CHEM_PROCESS_MODULES.get(key)
    if module is None:
        return False
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _freeze(value):
    """Deep-freeze parsed JSON: dicts become read-only mappings, lists tuples."""
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class SpeciesRow:
    """One chem species (schema ``species_row``)."""

    name: str
    output_name: str | None
    long_name: str
    units: str
    phase: str
    family: str
    sets: tuple[str, ...]
    default_inflow: float
    processes: tuple[str, ...]
    provenance: str
    source_file: str
    transported: bool = True
    floor: float = 1e-16
    molar_mass_g_mol: float | None = None
    emissions: tuple = ()
    boundary: tuple = ()
    drydep: Mapping | None = None
    settling: Mapping | None = None
    wetdep_ls_alpha: float = 0.0
    aging: Mapping | None = None
    optics: Mapping | None = None
    mp_coupling: Mapping | None = None
    exclusive_group: str | None = None
    #: GOCART process parameters (aq-gocart): the emission.dust bin constants,
    #: the emission.seasalt bin edges, and the chem.sulfur role of a row.
    dust_emission: Mapping | None = None
    seasalt_emission: Mapping | None = None
    sulfur_role: str | None = None
    #: Per-set gating of ``processes`` (schema ``process_sets``): a key listed
    #: here acts on this row only in a run selecting one of its sets.
    process_sets: Mapping | None = None
    #: Logical WRF Registry array for operators that distinguish chemistry
    #: from tracers. The arena remains shared; groups keep their own indices.
    wrf_array: str = "chem"
    #: How the row reaches the radiation (schema ``radiation``), or None:
    #: ``{"shortwave": "hrrr_smoke_aod", ...}`` makes its mass a layer AOD in
    #: the legacy RRTMG aer_opt = 3 shortwave when the run sets
    #: ``simple_dir_fdb = 1`` (the HRRR v4 fork's smoke feedback,
    #: woof.core.rrtmg_aerosol_optics smoke_feedback).
    radiation: Mapping | None = None

    @property
    def feeds_shortwave_aod(self) -> bool:
        """True when the row declares a shortwave layer-AOD coupling.

        The coupling's spelling is table data (schema.v1.json's
        ``radiation.shortwave`` enum, which admits exactly one value, the
        transported smoke layer AOD; species/smoke.json), so generic code
        asks this instead of naming it.  A second coupling value in the
        schema would add a parameter here.
        """
        return bool((self.radiation or {}).get("shortwave"))

    def acts_in(self, process: str, sets) -> bool:
        """True when ``process`` acts on this row in a run selecting ``sets``.

        The row must name the process; a process the row gates by set
        (``process_sets``) acts only when one of its sets is selected.
        """
        if process not in self.processes:
            return False
        gate = self.process_sets.get(process) if self.process_sets else None
        return gate is None or any(name in gate for name in sets)

    @property
    def state_attr(self) -> str:
        return state_attr(self.name)

    @property
    def time_attr(self) -> str:
        return time_attr(self.name)

    def to_json(self) -> dict:
        out = {}
        for key in _ROW_FIELDS:
            out[key] = _thaw(getattr(self, key))
        if self.radiation is not None:
            # Only a row that declares it carries the key, so every other
            # row's serialization (and the table identity) is unchanged.
            out["radiation"] = _thaw(self.radiation)
        return out


_ROW_FIELDS = ("name", "output_name", "long_name", "units", "phase", "family", "wrf_array",
               "sets", "transported", "floor", "default_inflow",
               "molar_mass_g_mol", "processes", "emissions", "boundary",
               "drydep", "settling", "wetdep_ls_alpha", "aging", "optics",
               "mp_coupling", "exclusive_group", "dust_emission",
               "seasalt_emission", "sulfur_role", "process_sets",
               "provenance")


@dataclass(frozen=True)
class SetRow:
    """One named species set (schema ``set_file``)."""

    name: str
    description: str
    wrf_chem_opt: int | None
    wrf_tracer_opt: int | None
    provenance: str
    source_file: str
    test_only: bool = False
    #: RunConfig chem keys this set's reference scheme turns on by default
    #: (``namelist_defaults`` in the set file): applied by the loaders only
    #: to keys the configuration leaves unset
    #: (:func:`woof.config.apply_chem_set_defaults`).
    namelist_defaults: Mapping = field(
        default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class SourceRow:
    """One external source (schema ``source_file``)."""

    name: str
    kind: str
    route: str | None
    credential: str | None
    format: str
    grid: Mapping
    variables: Mapping
    time: Mapping
    remap: str
    licence: str
    attribution: str
    provenance: str
    source_file: str
    raises_mixing_floor: bool = False
    #: What the source is, in one or two sentences (optional).
    description: str = ""
    #: The source's own vertical coordinate and how it is carried onto the
    #: model's levels (e.g. hybrid sigma-pressure with its coefficients in the
    #: GRIB records), or ``None`` for a 2-D source.
    vertical: Mapping | None = None
    #: How the bytes are acquired when no fetch-route row describes it: a
    #: data-store API's request grammar (dataset, request keys, cycles,
    #: cadence, horizon).  ``None`` for a route-table source.
    acquisition: Mapping | None = None
    #: The ``[fetch]`` sources (the run's meteorological start, e.g. ``hrrr``,
    #: ``hrrr-prs``) under which this boundary source is enabled without
    #: being named in ``chem_sources``: HRRR-Smoke's own smoke starts and
    #: bounds a run that starts from HRRR
    #: (:func:`woof.config.apply_route_chem_sources`).
    default_for_fetch_sources: tuple = ()


@dataclass(frozen=True)
class DiagnosticRow:
    """One output diagnostic built by a term sum (schema ``diagnostic_file``).

    ``terms`` is ordered: each term's species are summed left to right in
    float32, then its ``ops`` are applied in order, then the term is added to
    the accumulator -- which reproduces a Fortran expression such as
    ``pm = pm + chem(p_dust_2)*d_2_5 + chem(p_seas_1)`` word for word.  A term
    whose species are not all active contributes nothing (exactly as adding
    WRF's zero-initialized absent species would, the accumulator being
    non-negative).
    """

    output_name: str
    units: str
    description: str
    kind: str
    terms: tuple
    provenance: str
    source_file: str
    divide_by_alt: bool = False
    column_scale: float = 1.0


@lru_cache(maxsize=1)
def _schema() -> dict:
    return json.loads((CHEM_DATA_ROOT / "schema.v1.json").read_text(
        encoding="utf-8"))


_SCHEMA_BY_TAG = {
    "gpuwm.chem.species.v1": ("species_file", "species"),
    "gpuwm.chem.set.v1": ("set_file", "sets"),
    "gpuwm.chem.source.v1": ("source_file", "sources"),
    "gpuwm.chem.diagnostic.v1": ("diagnostic_file", "diagnostics"),
}


def _validate(document: dict, definition: str, where: str) -> None:
    import jsonschema

    schema = _schema()
    ref = {"$schema": schema["$schema"],
           "$ref": f"#/definitions/{definition}",
           "definitions": schema["definitions"]}
    try:
        jsonschema.Draft7Validator(ref).validate(document)
    except jsonschema.ValidationError as error:
        path = "/".join(str(p) for p in error.absolute_path)
        raise ChemTableError(
            f"{where}: does not validate against the chem schema "
            f"(woof/data/chem/schema.v1.json, {definition}) at "
            f"'{path or '<root>'}': {error.message}") from error


def _read(path: Path, subdir: str) -> tuple[dict, str]:
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ChemTableError(f"{subdir}/{path.name}: not UTF-8 JSON: "
                             f"{error}") from error
    tag = document.get("schema") if isinstance(document, dict) else None
    expected = _SCHEMA_BY_TAG.get(tag)
    if expected is None or expected[1] != subdir:
        raise ChemTableError(
            f"{subdir}/{path.name}: 'schema' is {tag!r}; a file in {subdir}/ "
            f"must say "
            f"{[t for t, (_d, s) in _SCHEMA_BY_TAG.items() if s == subdir]}")
    _validate(document, expected[0], f"{subdir}/{path.name}")
    return document, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ChemCatalog:
    """Every row, set, source and diagnostic the package carries."""

    species: Mapping[str, SpeciesRow]
    sets: Mapping[str, SetRow]
    sources: Mapping[str, SourceRow]
    diagnostics: Mapping[str, DiagnosticRow]
    file_hashes: Mapping[str, str]
    species_order: tuple[str, ...]

    @property
    def digest(self) -> str:
        """SHA-256 over the schema and every table file's bytes."""
        h = hashlib.sha256()
        for rel in sorted(self.file_hashes):
            h.update(f"{rel}\0{self.file_hashes[rel]}\n".encode())
        return h.hexdigest()

    def set_members(self, set_name: str) -> tuple[str, ...]:
        return tuple(n for n in self.species_order
                     if set_name in self.species[n].sets)


def _build_catalog(root: Path) -> ChemCatalog:
    file_hashes: dict[str, str] = {}
    schema_path = root / "schema.v1.json"
    file_hashes["schema.v1.json"] = hashlib.sha256(
        schema_path.read_bytes()).hexdigest()

    species: dict[str, SpeciesRow] = {}
    order: list[str] = []
    for path in sorted((root / "species").glob("*.json")):
        document, digest = _read(path, "species")
        rel = f"species/{path.name}"
        file_hashes[rel] = digest
        for raw in document["rows"]:
            name = raw["name"]
            if name in species:
                raise ChemTableError(
                    f"{rel}: species {name!r} is already defined in "
                    f"{species[name].source_file}; a species is ONE row "
                    "(add the set to that row's 'sets' instead)")
            kwargs = {k: _freeze(v) for k, v in raw.items()}
            species[name] = SpeciesRow(source_file=rel, **kwargs)
            order.append(name)

    sets: dict[str, SetRow] = {}
    for path in sorted((root / "sets").glob("*.json")):
        document, digest = _read(path, "sets")
        rel = f"sets/{path.name}"
        file_hashes[rel] = digest
        if path.stem != document["name"]:
            raise ChemTableError(
                f"{rel}: the file is named {path.stem!r} but defines set "
                f"{document['name']!r}; one set per file, named after it")
        sets[document["name"]] = SetRow(
            name=document["name"], description=document["description"],
            wrf_chem_opt=document["wrf_chem_opt"],
            wrf_tracer_opt=document["wrf_tracer_opt"],
            provenance=document["provenance"], source_file=rel,
            test_only=bool(document.get("test_only", False)),
            namelist_defaults=MappingProxyType(
                dict(document.get("namelist_defaults", {}))))

    sources: dict[str, SourceRow] = {}
    for path in sorted((root / "sources").glob("*.json")):
        document, digest = _read(path, "sources")
        rel = f"sources/{path.name}"
        file_hashes[rel] = digest
        if path.stem != document["name"]:
            raise ChemTableError(
                f"{rel}: the file is named {path.stem!r} but defines source "
                f"{document['name']!r}; one source per file, named after it")
        kwargs = {k: _freeze(v) for k, v in document.items() if k != "schema"}
        sources[document["name"]] = SourceRow(source_file=rel, **kwargs)

    diagnostics: dict[str, DiagnosticRow] = {}
    for path in sorted((root / "diagnostics").glob("*.json")):
        document, digest = _read(path, "diagnostics")
        rel = f"diagnostics/{path.name}"
        file_hashes[rel] = digest
        name = document["output_name"]
        if name in diagnostics:
            raise ChemTableError(
                f"{rel}: diagnostic {name!r} is already defined in "
                f"{diagnostics[name].source_file}")
        kwargs = {k: _freeze(v) for k, v in document.items() if k != "schema"}
        diagnostics[name] = DiagnosticRow(source_file=rel, **kwargs)

    _cross_check(species, sets, sources, diagnostics)
    return ChemCatalog(
        species=MappingProxyType(species), sets=MappingProxyType(sets),
        sources=MappingProxyType(sources),
        diagnostics=MappingProxyType(diagnostics),
        file_hashes=MappingProxyType(file_hashes),
        species_order=tuple(order))


def _cross_check(species, sets, sources, diagnostics) -> None:
    """Every reference between rows resolves; no two rows claim one name."""
    outputs: dict[str, str] = {}
    for row in species.values():
        where = f"{row.source_file} row {row.name!r}"
        for set_name in row.sets:
            if set_name not in sets:
                raise ChemTableError(
                    f"{where}: names set {set_name!r}, which has no file "
                    f"under woof/data/chem/sets/")
        for key in row.processes:
            if key not in CHEM_PROCESS_MODULES:
                raise ChemTableError(
                    f"{where}: names process {key!r}, which is not a process "
                    f"key (known: {sorted(CHEM_PROCESS_MODULES)})")
        for key, gate in (row.process_sets or {}).items():
            if key not in row.processes:
                raise ChemTableError(
                    f"{where}: process_sets gates {key!r}, which the row's "
                    "processes do not name, so the gate would act on nothing")
            stray = [name for name in gate if name not in row.sets]
            if stray:
                raise ChemTableError(
                    f"{where}: process_sets gates {key!r} on {stray}, which "
                    f"are not the row's sets {list(row.sets)}; the process "
                    "could never act through them")
        for ref in (*row.emissions, *row.boundary):
            if ref["source"] not in sources:
                raise ChemTableError(
                    f"{where}: names source {ref['source']!r}, which has no "
                    f"file under woof/data/chem/sources/")
        for ref in row.boundary:
            if len(ref["fields"]) != len(ref["weights"]):
                raise ChemTableError(
                    f"{where}: boundary source {ref['source']!r} has "
                    f"{len(ref['fields'])} fields and {len(ref['weights'])} "
                    "weights; each field needs its weight")
        if row.aging is not None and row.aging["to"] not in species:
            raise ChemTableError(
                f"{where}: ages into {row.aging['to']!r}, which is no row")
        if row.phase == "gas" and row.molar_mass_g_mol is None:
            raise ChemTableError(
                f"{where}: a gas row needs molar_mass_g_mol, because the mass "
                "ledger converts its ppmv to kilograms with M_i/M_air")
        if not row.transported and row.output_name is None and not row.processes:
            raise ChemTableError(
                f"{where}: a prescribed row with no output and no process is "
                "read by nothing")
        if row.output_name is not None:
            other = outputs.get(row.output_name)
            if other is not None:
                raise ChemTableError(
                    f"{where}: output_name {row.output_name!r} is already "
                    f"written by {other}")
            outputs[row.output_name] = where
    for name, row in diagnostics.items():
        other = outputs.get(name)
        if other is not None:
            raise ChemTableError(
                f"{row.source_file}: diagnostic {name!r} collides with the "
                f"output of {other}")
        outputs[name] = row.source_file
        for term in row.terms:
            for sp in term["species"]:
                if sp not in species:
                    raise ChemTableError(
                        f"{row.source_file}: term names species {sp!r}, "
                        "which is no row")
    for set_name, set_row in sets.items():
        if not any(set_name in row.sets for row in species.values()):
            raise ChemTableError(
                f"{set_row.source_file}: set {set_name!r} has no member row")


@lru_cache(maxsize=4)
def _catalog_at(root: str) -> ChemCatalog:
    return _build_catalog(Path(root))


def catalog(root: str | Path | None = None) -> ChemCatalog:
    """The packaged chem catalog (cached), or one read from ``root``."""
    return _catalog_at(str(Path(root) if root is not None else CHEM_DATA_ROOT))


@dataclass(frozen=True)
class ChemTable:
    """The rows one run carries, in arena order, plus what they reference."""

    sets: tuple[str, ...]
    rows: tuple[SpeciesRow, ...]
    sources: Mapping[str, SourceRow]
    diagnostics: tuple[DiagnosticRow, ...]
    identity: str
    catalog_digest: str
    enabled_sources: tuple[str, ...] = ()
    _by_name: Mapping[str, SpeciesRow] = field(default=None, repr=False,
                                               compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_by_name", MappingProxyType(
            {row.name: row for row in self.rows}))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(row.name for row in self.rows)

    @property
    def transported(self) -> tuple[SpeciesRow, ...]:
        return tuple(row for row in self.rows if row.transported)

    @property
    def prescribed(self) -> tuple[SpeciesRow, ...]:
        return tuple(row for row in self.rows if not row.transported)

    def row(self, name: str) -> SpeciesRow:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"chem species {name!r} is not active in this run "
                           f"(active: {list(self.names)})") from None

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def rows_for(self, process: str) -> tuple[SpeciesRow, ...]:
        """Active rows that name ``process`` in their ``processes`` column and
        whose ``process_sets`` gate (if any) admits this run's sets."""
        if process not in CHEM_PROCESS_MODULES:
            raise KeyError(f"{process!r} is not a chem process key")
        return tuple(row for row in self.rows
                     if row.acts_in(process, self.sets))

    @property
    def processes(self) -> tuple[str, ...]:
        """Every process key that acts on some active row, in registry order
        (a row's ``process_sets`` gate is applied against this run's sets)."""
        named = {key for row in self.rows for key in row.processes
                 if row.acts_in(key, self.sets)}
        return tuple(key for key in CHEM_PROCESS_MODULES if key in named)


def load_sets(sets: Sequence[str], sources: Iterable[str] = (), *,
              root: str | Path | None = None) -> ChemTable:
    """The table for an explicit set selection (see :func:`load`)."""
    cat = catalog(root)
    sets = tuple(sets)
    if not sets:
        raise ChemTableError("load_sets needs at least one set")
    if len(set(sets)) != len(sets):
        raise ChemTableError(f"chem_sets names a set twice: {list(sets)}")
    unknown = [s for s in sets if s not in cat.sets]
    if unknown:
        raise ChemTableError(
            f"chem_sets names unknown set(s) {unknown}; the table has "
            f"{sorted(cat.sets)} (woof/data/chem/sets/)")
    chosen: list[str] = []
    seen: set[str] = set()
    for set_name in sets:
        for name in cat.set_members(set_name):
            if name not in seen:
                seen.add(name)
                chosen.append(name)
    rows = tuple(cat.species[n] for n in chosen)
    for row in rows:
        if row.aging is not None and row.aging["to"] not in seen:
            raise ChemTableError(
                f"species {row.name!r} ages into {row.aging['to']!r}, which "
                f"none of the selected sets {list(sets)} carries, so the aged "
                "mass would leave the run unaccounted")
    enabled = tuple(sources)
    unknown_sources = [s for s in enabled if s not in cat.sources]
    if unknown_sources:
        raise ChemTableError(
            f"chem_sources names unknown source(s) {unknown_sources}; the "
            f"table has {sorted(cat.sources)} (woof/data/chem/sources/)")
    referenced = sorted({ref["source"] for row in rows
                         for ref in (*row.emissions, *row.boundary)})
    used_sources = {name: cat.sources[name] for name in referenced}
    diagnostics = tuple(
        d for d in cat.diagnostics.values()
        if any(all(sp in seen for sp in term["species"]) for term in d.terms))
    h = hashlib.sha256()
    h.update(cat.digest.encode())
    h.update(json.dumps({"sets": list(sets), "rows": list(chosen),
                         "sources": list(enabled)},
                        sort_keys=True).encode())
    return ChemTable(sets=sets, rows=rows,
                     sources=MappingProxyType(used_sources),
                     diagnostics=diagnostics, identity=h.hexdigest(),
                     catalog_digest=cat.digest, enabled_sources=enabled)


_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-")


def chem_names(value, key: str = "chem_sets") -> tuple[str, ...]:
    """RunConfig's comma-separated chem name list as a tuple of names.

    Accepts a tuple/list of names too (callers building configs by hand).
    Blank entries and surrounding spaces are dropped; an entry that could
    name no table file is refused here, so a typo cannot pass as an empty
    selection.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = [str(part).strip() for part in value]
    names = tuple(part for part in parts if part)
    for name in names:
        if not set(name) <= _NAME_CHARS:
            raise ChemTableError(
                f"{key} entry {name!r} is not a chem table name (lowercase "
                "letters, digits, '_' and '-'), so it could name no file")
    return names


def load(cfg) -> ChemTable | None:
    """The chem table ``cfg`` selects, or ``None`` when chem is off.

    ``cfg.chem_sets`` is the door: empty (the default) means no table read,
    no allocation, no launch and no identity entry anywhere.
    """
    sets = chem_names(getattr(cfg, "chem_sets", ""), "chem_sets")
    if not sets:
        return None
    return load_sets(sets, chem_names(getattr(cfg, "chem_sources", ""),
                                      "chem_sources"))
