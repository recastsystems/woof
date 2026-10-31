"""NOAA fork driver delta, with pristine native scheme and interface providers.

Authority: pinned MPAS-Model-rrfs at 1d6e370e, src/core_atmosphere/physics/
MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver.F:127-419, MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_control.F:87-391,
MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_manager.F:137-832. No scheme implementation is selected by a
suite-name branch. Table rows carry the module, pin set and dycore path.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics

from dataclasses import dataclass, field, replace as _replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ...config import parse_fortran_namelist
from ...physics_suites import load_table
from .. import driver_v842 as shared
from ...cuda_backend.runtime import KernelCache
from ...native_powf import CUDA_NATIVE_POWF
from ..math_v842 import bind_common_carrier_math
from ..diagnostics_v842 import native_diagnostic_source

Backend = shared.Backend
SlotCall = shared.SlotCall
StepFlags = shared.StepFlags
NativeCalendarState = shared.NativeCalendarState
SCHEMA = "mpas-port.physics-driver-rrfs/v1"
TABLE_KEY = "mpasphys_rrfs"


def specification(config=None, *, table_key=TABLE_KEY) -> Mapping[str, Any]:
    """Fork table contract; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_control.F:133-183."""
    return load_table()[getattr(config, "table_key", table_key)]


def native_table_inputs(roots, *, override_roots=()):
    """Collect the fork's native initialization inputs without decoding them.

    MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_init.F:396-470.
    Extra native filenames are table metadata. Identical published aliases
    denote one input; conflicting base bytes must not silently replace it.
    Explicit override roots follow the native forecast runner's ordered table
    staging. They replace inputs before installation, without decoding them.
    Absent files remain subject to the pristine scheme's initialization logic.
    """
    import hashlib
    from pathlib import Path
    names = set(specification()['native_table_names'])
    selected, digests = {}, {}
    for root, override in (*((root, False) for root in roots),
                           *((root, True) for root in override_roots)):
        for path in sorted(Path(root).iterdir()):
            if not path.is_file() or not (path.name in names
                    or path.suffix in ('.TBL', '.DBL')
                    or path.name.startswith(('RRTMG_', 'CCN_'))):
                continue
            if path.name in selected:
                previous = selected[path.name]
                if previous.resolve() == path.resolve():
                    continue
                if override:
                    selected[path.name] = path
                    digests.pop(path.name, None)
                    continue
                def digest(item):
                    """Hash opaque native inputs; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_init.F:396-470."""
                    with item.open('rb') as stream:
                        return hashlib.file_digest(stream, 'sha256').hexdigest()
                digests.setdefault(path.name, digest(previous))
                if digest(path) != digests[path.name]:
                    raise ValueError('conflicting native table bytes for ' + path.name)
            else:
                selected[path.name] = path
    return tuple(selected.values())


def native_table_stream_inputs(config, roots, *, override_roots=()):
    """Bind active climatology streams before Rust inspects their dimensions.

    MPAS-Model-rrfs/src/core_atmosphere/Registry.xml:55-58,1299-1310;
    MPAS-Model-rrfs/src/core_atmosphere/Registry.xml:4366-4392.
    Selection is table metadata; NetCDF dimensions and words remain Rust's.
    Missing inputs remain subject to the real stream/input machinery.
    """
    tables = {p.name: p for p in native_table_inputs(roots, override_roots=override_roots)}
    return {row['stream']: tables[row['filename']]
            for row in specification(config)['native_table_streams']
            if config.execution_options[row['option']] == row['value']
            and row['filename'] in tables}


def stage_native_initialization_namelist(source, directory):
    """Retain internal scheme groups in the pristine initializer's working dir.

    MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_mp_nssl_2mom.F:1227-1245.
    Copy the original opaque namelist, including non-Registry groups. Refuse an
    existing different file because it would initialize from different options.
    """
    import hashlib
    import shutil
    from pathlib import Path
    source = Path(source)
    target = Path(directory) / 'namelist.atmosphere'
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if target.exists():
        with target.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise ValueError('native initialization namelist already differs: ' + str(target))
    else:
        shutil.copyfile(source, target)
    return digest


def prepare_native_registry(root) -> None:
    """Normalize included metadata before allocation, as the native generator.

    MPAS-Model-rrfs/src/tools/registry/parse.c:54-57;
    MPAS-Model-rrfs/src/tools/registry/gen_inc.c:2227-2521.
    First declarations retain their attributes. Later declarations contribute
    children in native order; chemistry constituents extend the same carrier.
    """
    def inherit(parent):
        """Push inherited attributes; MPAS-Model-rrfs/src/tools/registry/gen_inc.c:2227-2305."""
        for array in parent.findall('var_array'):
            for key in ('time_levs', 'packages'):
                if key not in array.attrib and key in parent.attrib:
                    array.set(key, parent.get(key))
            for field in array.findall('var'):
                if 'packages' not in field.attrib and 'packages' in array.attrib:
                    field.set('packages', array.get('packages'))
        for tag in ('var', 'var_struct'):
            for child in parent.findall(tag):
                for key in ('packages', 'time_levs'):
                    if key not in child.attrib and key in parent.attrib:
                        child.set(key, parent.get(key))
                if tag == 'var_struct':
                    inherit(child)

    def merge(parent, tag, child_tags):
        """Keep first declarations; MPAS-Model-rrfs/src/tools/registry/gen_inc.c:2308-2521."""
        first = {}
        for item in list(parent.findall(tag)):
            name = item.get('name')
            if name not in first:
                first[name] = item
                continue
            target = first[name]
            for child_tag in child_tags:
                for child in list(item.findall(child_tag)):
                    item.remove(child)
                    target.append(child)
            parent.remove(item)

    def merge_structs(parent):
        """Recursively join pools and carriers; MPAS-Model-rrfs/src/tools/registry/gen_inc.c:2308-2404."""
        merge(parent, 'var_struct', ('var', 'var_array', 'var_struct'))
        merge(parent, 'var_array', ('var',))
        for child in parent.findall('var_struct'):
            merge_structs(child)

    inherit(root)
    merge_structs(root)
    blocks = root.findall('streams')
    if blocks:
        for block in blocks[1:]:
            for stream in list(block.findall('stream')):
                block.remove(stream)
                blocks[0].append(stream)
            root.remove(block)
        merge(blocks[0], 'stream', ('var', 'var_array', 'var_struct', 'stream'))


def prepare_native_layout(layout) -> None:
    """Inactive array metadata still exists; MPAS-Model-rrfs/src/tools/registry/gen_inc.c:1321-1489; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_manager.F:461-485,707-726."""
    # Native bootstrapping retains configured layers even on a rank with no
    # remote cells: MPAS-Model-rrfs/src/framework/mpas_bootstrapping.F:144-156;
    # MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_halos.F:181-186.
    # Distributed layouts must supply their actual cumulative native ends.
    halo_ends = dict(getattr(layout, 'halo_ends', {}))
    for entity in ('nCells', 'nEdges', 'nVertices'):
        # Partial Registry metadata views do not allocate absent entities.
        if entity not in layout.dimensions:
            continue
        if entity in halo_ends:
            continue
        count = layout.dimensions[entity]
        if layout.dimensions[entity + 'Solve'] != count:
            raise ValueError('RRFS distributed halo ends missing for ' + entity +
                             '; layer2 exchange cannot use a guessed partition')
        halo_ends[entity] = (count,) * layout.configs['config_num_halos']
    layout.halo_ends = halo_ends
    # Generated allocation creates constituent dimensions even when every
    # package in an array is inactive. Do not allocate inactive weather fields
    # or invent input-file dimensions such as nlcat and nscat.
    for struct in layout.root.findall('./var_struct'):
        if not layout.enabled(struct):
            continue
        pool = struct.get('name_in_code', struct.get('name'))
        for array in struct.findall('var_array'):
            if layout.enabled(array):
                continue
            layout.indices[pool, 'num_' + array.get('name')] = 0
            for component in array.findall('var'):
                name = component.get('name_in_code', component.get('name'))
                layout.indices[pool, 'index_' + name] = -1
                group = component.get('array_group')
                if group:
                    layout.indices[pool, group + '_start'] = -1
                    layout.indices[pool, group + '_end'] = -1


def _condition(options: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    """Package/chemistry predicates; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_packages.F:109-293."""
    return all(options[k] != v["not"] if isinstance(v, Mapping) else options[k] == v
               for k, v in row.items())


@dataclass(frozen=True)
class CorrectionInventoryGap:
    """Fork policy metadata; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_control.F:87-391."""
    finding: str
    slot: str
    scheme: str | None
    backend: str | None
    reason: str
    owner: str
    breakage: str

    def as_dict(self) -> dict[str, Any]:
        """Record an unqualified fork inventory; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_control.F:87-391."""
        return {**vars(self), "runs": "the selected pinned fork behavior"}


@dataclass(frozen=True)
class DriverConfig(shared.DriverConfig):
    _table_key: str = field(default=TABLE_KEY, kw_only=True)

    @property
    def native_registry_prepare(self) -> str:
        """Pre-allocation cleanup; MPAS-Model-rrfs/src/tools/registry/parse.c:54-57."""
        return specification(self)['native_registry_prepare']

    @property
    def native_layout_prepare(self) -> str:
        """Metadata provider; MPAS-Model-rrfs/src/tools/registry/gen_inc.c:1321-1489."""
        return specification(self)['native_layout_prepare']

    @property
    def table_key(self) -> str:
        """Provider row identity; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_control.F:133-183."""
        return self._table_key

    @property
    def driver_module(self) -> str:
        """Table-selected driver; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_control.F:133-183."""
        return specification(self)["driver_module"]

    @property
    def native_suite(self) -> str:
        """Resolved fork spelling; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_control.F:133-183."""
        return self.options["config_physics_suite"]

    @property
    def pin_set(self) -> str:
        """Native fork authority; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_control.F:133-183."""
        return specification(self)["pin_set"]

    @property
    def dycore_path(self) -> str:
        """Fork dynamics selection; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_control.F:133-183."""
        return specification(self)["dycore_path"]

    def active(self, slot: str) -> bool:
        """Respect the embedded UGWP gate; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_gwdo.F:762-823."""
        row = specification(self)["slots"][slot]
        parent = row.get("native_parent")
        derived = row.get("derived_from")
        return (self.selectors[slot] != "off" and self.backends[slot] != Backend.OFF
                and (not derived or self.active(derived))
                and (not parent or (self.active(parent)
                     and self.selectors[parent] in row["parent_schemes"])))

    # Mask disabled logical/string selectors without changing native option types.
    # MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_control.F:364-376.
    @property
    def execution_options(self) -> Mapping[str, Any]:
        """Backend masks retain native types; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_control.F:364-376."""
        result = dict(self.options)
        for slot, row in specification(self)["slots"].items():
            if self.backends[slot] == Backend.OFF and row["selector"] is not None:
                result[row["selector"]] = False if row.get("selector_type") == "logical" else "off"
                for clause in row.get("activation", ()):
                    for key in clause:
                        result[key] = "off"
        return MappingProxyType(result)

    def native_embedded(self, slot: str) -> bool:
        """NGW belongs to UGWP's native call; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_gwdo.F:692-834."""
        parent = specification(self)["slots"][slot].get("native_parent")
        return bool(parent and self.active(slot) and self.active(parent)
                    and self.backends[slot] == self.backends[parent] == Backend.NATIVE_CPU)

    def gap_warning(self) -> str | None:
        """Keep fork authority explicit; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_control.F:87-391."""
        if self.mode == "exact" or not self.gaps:
            return None
        lines = [f"CORRECT MODE: named gaps ({len(self.gaps)}): selected fork behavior retained"]
        for gap in self.gaps:
            lines.append(f"CORRECT MODE:   {gap.finding} on {gap.slot}: {gap.reason}; "
                         f"owner {gap.owner}. {gap.breakage}")
        return "\n".join(lines)

    def receipt(self) -> dict[str, Any]:
        """Resolved fork identity; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_control.F:133-183."""
        spec = specification(self)
        result = {"schema": SCHEMA, "suite": self.suite,
                "native_suite": self.options["config_physics_suite"],
                "backend": spec["backend"], "driver_module": spec["driver_module"],
                "pin_set": spec["pin_set"], "model_commit": spec["model_commit"],
                "submodule_pins": dict(spec["submodule_pins"]), "dycore_path": spec["dycore_path"],
                "required_extensions": dict(spec["matched_extensions"]),
                "selectors": dict(self.selectors), "options": dict(self.options),
                "execution_options": dict(self.execution_options),
                "backends": {s: str(b) for s, b in self.backends.items()}}
        if self.cpu_slots:
            result["cpu_slots"] = [item.as_dict() for item in self.cpu_slots]
        if self.mode != "exact":
            result.update(physics_mode=self.mode, corrections=sorted(self.corrections),
                          correct_skip=sorted(self.correct_skip), correct_mode=self.correct_mode_record())
        return result


def resolve_namelist(text: str, *, backends: Mapping[str, str] | None = None,
                     default_backend: str | None = None, physics_mode: str | None = None,
                     correct_skip: Iterable[str] | None = None, table_key: str = TABLE_KEY) -> DriverConfig:
    """Exact ordered control checks; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_control.F:87-391.

    Native suite spellings can be resolved by an explicitly selected fork
    provider. Public routing uses unique table aliases for suites shared with
    v8.4.2; it never changes the existing unqualified MPAS routes.
    ``default_backend`` as in the v8.4.2 resolver (driver_v842.resolve_namelist): None is the library
    default for replays (native-cpu, no ``cpu_slots``); the route passes its table default.
    """
    spec = specification(table_key=table_key)
    options = {k: r["value"] for k, r in spec["registry_defaults"].items()}
    groups = {group: dict(values) for group, values in parse_fortran_namelist(text).items()}
    mode, skip = shared._physics_mode(groups, physics_mode, correct_skip)
    # Hex mode controls are orchestration, absent from the pristine fork Registry.
    for key in ("config_physics_mode", "config_physics_correct_skip"):
        groups.get("nhyd_model", {}).pop(key, None)
    for group, keys in spec["registry_groups"].items():
        for key, value in groups.get(group, {}).items():
            if key not in keys:
                raise ValueError(f"fork Registry has no {group} option {key!r}")
            kind = spec["registry_defaults"][key]["type"]
            valid = ((kind == "character" and isinstance(value, str))
                     or (kind == "logical" and type(value) is bool)
                     or (kind == "integer" and type(value) is int)
                     or (kind == "real" and type(value) in (int, float)))
            if not valid:
                raise ValueError(f"{key} requires fork Registry type {kind}")
            options[key] = value.rstrip() if kind == "character" else value
    suite = options["config_physics_suite"].rstrip()
    native_suite = spec["suite_routes"].get(suite, suite)
    if native_suite not in spec["suites"]:
        raise ValueError(f"{suite!r} has no fork driver suite row")
    options["config_physics_suite"] = native_suite
    selectors = {}
    for slot, row in spec["slots"].items():
        value = selectors[row["derived_from"]] if row["selector"] is None else options[row["selector"]]
        if row.get("selector_type") == "logical":
            selected = "on" if value else "off"
        elif "activation" in row:
            selected = "on" if any(_condition(options, c) for c in row["activation"]) else "off"
        else:
            selected = spec["suites"][native_suite][row["selector"]] if value == "suite" else value
        if selected not in row["schemes"]:
            raise ValueError(f"illegal fork {row['selector']}={selected!r}")
        selectors[slot] = selected
    # The source first validates input cloud spelling, then fills an unset
    # radiation cloud, then checks MYNN pairing, then sets internal cld_tempo.
    if (selectors["radiation_lw"] != "off" or selectors["radiation_sw"] != "off") \
            and selectors["cloudiness"] == "off":
        selectors["cloudiness"] = "cld_incidence"
    if selectors["cloudiness"] == "cld_fraction_mynn" \
            and selectors["pbl"] not in ("bl_mynn", "bl_mynnedmf"):
        raise ValueError("fork cld_fraction_mynn requires a MYNN PBL")
    if selectors["microphysics"] == "mp_tempo" and options["config_tempo_cldfra"]:
        selectors["cloudiness"] = "cld_tempo"
        # This correction names bl_mynn only, not bl_mynnedmf, in the pin.
        if selectors["pbl"] == "bl_mynn" and options["config_mynn_cloudpdf"] >= 0:
            options["config_mynn_cloudpdf"] = -2
    forced = {"bl_mynn": "sf_mynn", "bl_mynnedmf": "sf_mynnsfclay", "bl_myj": "sf_myj"}
    if selectors["pbl"] in forced:
        selectors["surface_layer"] = forced[selectors["pbl"]]
    if selectors["pbl"] == "bl_ysu" and selectors["surface_layer"] not in (
            "sf_monin_obukhov", "sf_monin_obukhov_rev"):
        raise ValueError("YSU PBL requires an MPAS Monin-Obukhov surface layer")
    if selectors["land_surface"] != "off" and selectors["surface_layer"] == "off":
        raise ValueError("MPAS land surface requires a surface layer")
    for slot, row in spec["slots"].items():
        if row["selector"] is not None and "activation" not in row and row.get("selector_type") != "logical":
            options[row["selector"]] = selectors[slot]
    library = default_backend is None
    chosen, cpu_reasons = shared.default_slot_backends(
        table_key, selectors, Backend.NATIVE_CPU if library else default_backend)
    explicit = set()
    backend_names = {r["backend_selector"]: s for s, r in spec["slots"].items()}
    supplied = groups.get(spec["backend_namelist_group"], {})
    # Driver-boundary services (the fork surface update), as the v8.4.2 row: a GPU route that updates
    # SST from a surface series takes the service on the GPU; module replays (library) keep native-cpu.
    services = spec.get("services", {})
    # A route takes the fork surface service on its default backend: the GPU on a hex-cuda route, native-cpu
    # for a run that names the native referee.  THE BREAKAGE THIS PREVENTS: on the native library the fork's
    # physics_update_surface stopped at the first greenness alarm with "Fortran runtime error: Bad value during
    # integer read" in mpas_atmphys_date_time (hrrrv5 x1.2562, 2026-10-28), so a GPU route left on the
    # native-cpu service never reached step 2.  Module replays (library) keep the row's library default.
    chosen_services = {
        name: Backend(row.get("library_default_backend", row["default_backend"]) if library
                      else str(default_backend) if str(default_backend) in row["backends"]
                      else row["default_backend"])
        for name, row in services.items()}
    service_names = {row["backend_selector"]: name for name, row in services.items()}
    for key, value in supplied.items():
        if key in service_names:
            name = service_names[key]
            if value not in services[name]["backends"]:
                raise ValueError(f"unsupported fork {name} service backend {value!r}")
            chosen_services[name] = Backend(value)
            continue
        if key not in backend_names:
            raise ValueError(f"unknown fork slot backend option {key!r}")
        chosen[backend_names[key]] = Backend(value)
        explicit.add(backend_names[key])
    for slot, backend in (backends or {}).items():
        if slot not in selectors:
            raise ValueError(f"unknown fork slot {slot!r}")
        key = spec["slots"][slot]["backend_selector"]
        if key in supplied and chosen[slot] != Backend(backend):
            raise ValueError(f"conflicting fork backend selections for {slot}")
        chosen[slot] = Backend(backend)
        explicit.add(slot)
    # v8.4.2 findings must not silently change v8.3.1/GSL science. Until the
    # fork owners qualify its inventory, correct mode reports that named gap.
    gaps = () if mode == "exact" else (CorrectionInventoryGap(
        **spec["correct_mode_inventory"]),)
    config = DriverConfig(suite, MappingProxyType(selectors), MappingProxyType(options),
                          MappingProxyType(chosen), mode=mode, correct_skip=skip, gaps=gaps,
                          service_backends=MappingProxyType(chosen_services), _table_key=table_key)
    if library:
        return config
    return _replace(config, cpu_slots=shared.cpu_slot_record(config, table_key, cpu_reasons, explicit))


def route_namelist(text: str, *, physics_mode: str | None = None,
                   correct_skip: Iterable[str] | None = None,
                   default_backend: str | None = None, table_key: str = TABLE_KEY) -> DriverConfig | None:
    """Table-only route; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_control.F:133-183."""
    suite = parse_fortran_namelist(text).get("physics", {}).get("config_physics_suite")
    if isinstance(suite, str) and suite.strip() in specification(table_key=table_key)["suite_routes"]:
        return resolve_namelist(text, default_backend=default_backend or specification(table_key=table_key)["default_backend"],
                                physics_mode=physics_mode, correct_skip=correct_skip, table_key=table_key)
    return None


def package_selection(config: DriverConfig) -> Mapping[str, bool]:
    """All fork physics packages; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_packages.F:27-295."""
    return MappingProxyType({name: any(_condition(config.execution_options, c) for c in clauses)
                             for name, clauses in specification(config)["package_rules"].items()})


def scalar_registry(config: DriverConfig) -> tuple[Mapping[str, Any], ...]:
    """Registry order, never a scheme's invented row; MPAS-Model-rrfs/src/core_atmosphere/Registry.xml:1992-2083.

    The pin has ng/volg for hail-aware TEMPO and no qb scalar. Extra scalar
    groups from chemistry are allocated by r14's Registry provider.
    """
    packages = package_selection(config)
    return tuple(row for row in specification(config)["scalar_registry"]
                 if not row.get("packages") or any(packages.get(p, False)
                                                   for p in row["packages"].split(";")))


class PhysicsManager(shared.PhysicsManager):
    """Reuse identical alarm code; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_manager.F:137-832."""

    def __init__(self, config: DriverConfig, dt: float, *, streams=None):
        """Slot intervals remain independent; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_manager.F:480-705."""
        super().__init__(config, dt, streams=streams)
        self.scheme_dt["gwdo"] = self.scheme_dt["pbl"]
        self.scheme_dt["ngw"] = self.scheme_dt["pbl"]

    def native_timestep_state(self) -> Mapping[str, int | float]:
        """Clock metadata; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_manager.F:737-762.

        These are RKIND schedule scalars, not weather-field processing. Keep
        the native rounding at each assignment, including positive NINT.
        Pristine native initialization still owns its module variables.
        """
        from ctypes import c_float
        rkind = lambda value: c_float(float(value)).value
        dt = rkind(self.dt)
        micro_dt, substeps = dt, 1
        micro = self.config.selectors["microphysics"]
        if micro in ("mp_thompson", "mp_thompson_aerosols", "mp_tempo"):
            substeps = max(int(rkind(dt / rkind(90)) + 0.5), 1)
            micro_dt = rkind(dt / rkind(substeps))
            if dt <= micro_dt:
                micro_dt = dt
        elif micro == "mp_nssl2m" and dt > rkind(rkind(1.25001) * rkind(60)):
            substeps = max(int(rkind(dt / rkind(60)) + 0.5), 2)
            micro_dt = rkind(dt / rkind(substeps))
        return MappingProxyType({
            "dt_dyn": dt, "dt_microp": micro_dt, "n_microp": substeps,
            "dt_cu": rkind(self.scheme_dt["convection"]),
            "dt_pbl": rkind(self.scheme_dt["pbl"]),
            "dt_radtlw": rkind(self.scheme_dt["radiation_lw"]),
            "dt_radtsw": rkind(self.scheme_dt["radiation_sw"]),
            "n_cu": max(int(rkind(rkind(self.scheme_dt["convection"]) / dt) + 0.5), 1),
        })


class SlotRegistry(shared.SlotRegistry):
    def register(self, slot: str, scheme: str, backend: str, callback, *,
                 corrections: Iterable[str] = ()) -> None:
        """Fork scheme registration; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver.F:234-416."""
        rows = specification()["slots"]
        key = slot, scheme, Backend(backend)
        if slot not in rows or scheme not in (*rows[slot]["schemes"], *rows[slot].get("internal_schemes", ())) \
                or scheme == "off":
            raise ValueError(f"unknown fork slot/scheme {slot}/{scheme}")
        if key[2] == Backend.OFF or not callable(callback):
            raise ValueError("register an executable native-cpu or hex-cuda slot")
        if key in self._callbacks:
            raise ValueError(f"physics slot already registered: {key}")
        declared = frozenset(corrections)
        if declared:
            raise ValueError("RRFS correction declarations require fork source and replay qualification; "
                             "the v8.4.2 inventory cannot certify a fork callback")
        self._callbacks[key] = callback
        self._corrections[key] = declared

    def require(self, config: DriverConfig) -> None:
        """Require actual executed slots; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver.F:196-416."""
        if not config.moist_physics:
            return
        missing = [(s, config.selectors[s], str(config.backends[s])) for s in config.selectors
                   if config.active(s) and not config.native_embedded(s)
                   and (s == "microphysics" or config.driver_physics)
                   and (s != "pbl" or config.active("surface_layer"))
                   and (s, config.selectors[s], config.backends[s]) not in self._callbacks]
        if missing:
            raise RuntimeError(f"fork physics slots not registered: {missing}")


class NativePoolArguments(shared.NativePoolArguments):
    """Fork ABI uses state, mesh, emissions and halo callbacks explicitly."""

    def __init__(self, native, *, pools, its, ite, synchronize, exchange_halo_group=None):
        """Bind fork halo contract; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver.F:41-52,127-416."""
        super().__init__(native, pools=pools, its=its, ite=ite, synchronize=synchronize)
        self.exchange_halo_group = exchange_halo_group

    def __call__(self, entry: str, call: SlotCall) -> Mapping[str, Any]:
        """Exact names/time levels; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver.F:234-416; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_manager.F:382-832; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_update_surface.F:54-143."""
        micro = entry in ("driver_microphysics", "microphysics_from_mpas", "microphysics_to_mpas")
        values = dict(self.pools)
        values.update(its=self.its, ite=self.ite, itimestep=call.itimestep,
                      xtime_s=call.time_seconds, dt=call.dt, time_lev=2 if micro else 1,
                      exchange_halo_group=self.exchange_halo_group)
        values.update(self.calendar)
        values.update({k: call.options[k] for k in ("config_frac_seaice", "config_sfc_albedo",
                                                   "config_gvf_update", "config_dust_scheme")})
        if entry == "physics_get_tend":
            values.update({k: call.pools[k] for k in ("rk_step", "dynamics_substep")})
        result = {}
        for name in self.native.abi["entries"][entry]["arguments"]:
            if name in values:
                if name == "exchange_halo_group" and values[name] is None:
                    raise RuntimeError("fork convection requires the real MPAS halo callback; the ABI exposes an unbound callback")
                result[name] = values[name]
            elif name in ("dminfo", "clock", "stream_manager", "domain", "block", "streaminfo", "iocontext"):
                result[name] = self.native.framework_object(name)
            else:
                raise RuntimeError(f"native fork {entry}.{name} has no bound pool or calendar value")
        self.synchronize(entry, call)
        return result


class NativeDriverHooks(shared.NativeDriverHooks):
    def configure(self, config: DriverConfig) -> None:
        """Reject mixed authorities before mutation; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_init.F:84-477.

        THE BREAKAGE THIS PREVENTS: the fork driver's glue (microphysics writeback, fork interface and
        todynamics) bound to a v8.4.2 framework or library would run v8.4.2 glue on fork pools.  The pin
        is read from the native library's build receipt, or from the ported fork framework's own
        ``source_pin`` (route_table framework row), never assumed.
        """
        build = getattr(self.native, "build", None) or {}
        pin = build.get("source_pin") or getattr(self.native, "source_pin", None)
        if pin != specification(config)["model_commit"]:
            raise ValueError("fork driver requires the fork framework (native library or ported "
                             "mpasphys_rrfs framework); v8.4.2 changes microphysics writeback")
        super().configure(config)

    def missing_operations(self, config: DriverConfig) -> tuple[str, ...]:
        """Complete fork closure; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver.F:196-419."""
        # Reuse shared operation selection but omit the v8.4 OpenACC-only
        # pre/post microphysics entries, which do not exist in this fork.
        missing = set(super().missing_operations(config)) - {"pre_microphysics", "post_microphysics"}
        if config.backends["oml"] == Backend.HEX_CUDA:
            missing.discard("driver_oml1d")
        extra = set()
        if config.driver_physics and config.active("land_surface") \
                and config.selectors["land_surface"] == "sf_ruc":
            extra.update(("allocate_lsm", "deallocate_lsm"))
        if config.driver_physics and config.active("smoke"):
            extra.update(("allocate_smoke", "deallocate_smoke"))
        for op in extra:
            if op not in self.external and op not in self.native.abi["entries"]:
                missing.add(op)
        return tuple(sorted(missing))

    def register_slots(self, registry: SlotRegistry, config: DriverConfig) -> None:
        """Pristine fork drivers only; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver.F:234-416."""
        for slot, scheme in config.selectors.items():
            if not config.active(slot) or config.native_embedded(slot) \
                    or config.backends[slot] != Backend.NATIVE_CPU:
                continue
            entry = specification(config)["slots"][slot]["native_entries"][scheme]
            if entry not in self.native.abi["entries"] and entry not in self.external:
                raise RuntimeError(f"native fork ABI has no slot entry: {entry}")
            registry.register(slot, scheme, Backend.NATIVE_CPU,
                              lambda call, entry=entry: self.invoke(entry, call))


#: Marks this step's physics_get_tend result as handed to the runtime and dropped (release_held_tendencies).
_RELEASED_TENDENCIES = object()


class PhysicsDriver(shared.PhysicsDriver):
    """Fork orchestration with shared lifecycle/restart transaction handling."""

    def restart_state(self):
        """Fork identity protects persistent pools; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_init.F:174-396."""
        result = super().restart_state()
        result["schema"] = SCHEMA
        return result

    def restore_restart_state(self, payload):
        """Keep shared transactional restore; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_init.F:174-396."""
        if payload.get("schema") != SCHEMA:
            raise ValueError("fork restart schema differs from live driver")
        super().restore_restart_state({**payload, "schema": shared.SCHEMA})

    def begin_step(self, *, itimestep: int, time_seconds: float, pools: Any) -> StepFlags:
        """Full fork pre-dynamics order; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver.F:127-419; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_manager.F:137-379."""
        if not self._initialized or self._pending is not None or itimestep < 1:
            raise RuntimeError("physics step requires initialization and no pending step")
        c = self.config
        flags = self.manager.advance(time_seconds) if c.moist_physics else StepFlags()
        self.calls = []
        self._pending = itimestep, time_seconds, flags, pools
        self._tendencies_ready, self._tendencies = False, None
        if not c.moist_physics:
            return flags
        def glue(op):
            """Native glue; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver.F:196-410."""
            self._call(op, itimestep, time_seconds, flags, pools)
        def run(slot):
            """Selected entry; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver.F:234-416."""
            self._call(slot, itimestep, time_seconds, flags, pools, slot=slot)
        glue("physics_timetracker_state")
        if flags.greeness:
            glue("physics_update_surface")
        if flags.surface:
            glue("physics_update_sst")
        for key, operation in (("config_sstdiurn_update", "physics_update_sstskin"),
                               ("config_deepsoiltemp_update", "physics_update_deepsoiltemp")):
            if c.options[key]:
                glue(operation)
        if c.options["config_o3climatology"] and (
                flags.radiation_lw and c.selectors["radiation_lw"] == "rrtmg_lw"
                or flags.radiation_sw and c.selectors["radiation_sw"] == "rrtmg_sw"):
            glue("o3climatology_from_MPAS")
        if not c.driver_physics:
            return flags
        glue("allocate_forall_physics")
        glue("MPAS_to_physics")
        if flags.radiation_lw or flags.radiation_sw:
            if c.active("cloudiness"):
                glue("allocate_cloudiness")
                run("cloudiness")
        for slot, active in (("radiation_sw", flags.radiation_sw), ("radiation_lw", flags.radiation_lw)):
            if active:
                glue("allocate_" + slot)
                run(slot)
        if c.options["config_bucket_update"] != "none" and c.options["config_bucket_radt"] > 0:
            glue("update_radiation_diagnostics")
        if c.active("radiation_sw") or c.active("radiation_lw"):
            glue("deallocate_cloudiness")
        for slot in ("radiation_sw", "radiation_lw"):
            if c.active(slot):
                glue("deallocate_" + slot)
        if c.active("surface_layer"):
            glue("allocate_sfclayer")
            run("surface_layer")
            glue("deallocate_sfclayer")
        if c.active("oml"):
            run("oml")
        if c.active("land_surface"):
            allocate = c.selectors["land_surface"] in ("sf_noah", "sf_ruc")
            if allocate:
                glue("allocate_lsm")
            run("land_surface")
            if allocate:
                glue("deallocate_lsm")
            if c.active("seaice"):
                glue("allocate_seaice")
                run("seaice")
                glue("deallocate_seaice")
            glue("atmphys_sfc_diagnostics")
        if c.active("smoke"):
            glue("allocate_smoke")
            run("smoke")
            glue("deallocate_smoke")
        if c.active("pbl") and c.active("surface_layer"):
            glue("allocate_pbl")
            run("pbl")
            glue("deallocate_pbl")
        if c.active("gwdo"):
            glue("allocate_gwdo")
            separate_ngw = c.active("ngw") and not c.native_embedded("ngw")
            if separate_ngw:
                glue("snapshot_gwdo_tendencies")
            run("gwdo")
            if separate_ngw:
                run("ngw")
            glue("deallocate_gwdo")
        glue("update_convection_step1")
        if flags.convection:
            glue("allocate_convection")
            run("convection")
            glue("deallocate_convection")
        if c.active("convection"):
            glue("update_convection_step2")
        glue("deallocate_forall_physics")
        return flags

    def _call(self, operation, step, time, flags, pools, *, slot=None):
        """Preserve independent backend options; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_gwdo.F:692-834."""
        c = self.config
        options = dict(c.execution_options)
        if slot == "ngw":
            options["config_ngw_scheme"] = c.options["config_ngw_scheme"]
        if slot == "gwdo" and c.active("ngw") and not c.native_embedded("ngw"):
            options["config_ngw_scheme"] = False
        # Glue backend: the service row's backend, or hex-cuda for an operation the resident glue owns.
        service = next((name for name, row in specification(c).get("services", {}).items()
                        if slot is None and operation in row["operations"]), None)
        glue_backend = c.service_backends.get(service, Backend.NATIVE_CPU)
        if slot is None and (operation in getattr(self.hooks, "resident_operations", ())
                or (operation == "atm_reset_diagnostics"
                    and getattr(getattr(pools, "device_pools", None), "resident", False))):
            glue_backend = Backend.HEX_CUDA
        call = SlotCall(operation if slot is None else slot,
                        "driver" if slot is None else c.selectors[slot],
                        glue_backend if slot is None else c.backends[slot],
                        step, time, float(self.manager.scheme_dt.get(slot, self.manager.dt)),
                        flags, pools, MappingProxyType(options))
        self.calls.append({"operation": operation, "slot": call.slot,
                           "backend": str(call.backend), "itimestep": step, "time_seconds": time})
        return self.hooks.invoke(operation, call) if slot is None else self.registry.run(call)

    def physics_get_tend(self, *, rk_step, dynamics_substep, pools):
        """Hold once before RK; MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1076-1084; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_todynamics.F:255-493."""
        if self._pending is None or rk_step not in (1, 2, 3) or dynamics_substep < 1:
            raise RuntimeError("physics tendency requested outside an active RK stage")
        if not self.config.moist_physics:
            return None
        if self._tendencies_ready:
            if self._tendencies is _RELEASED_TENDENCIES:
                raise RuntimeError("physics sources were already handed to the runtime and released this step "
                                   "(release_held_tendencies); prevents a second consumer reading freed sources")
            return self._tendencies
        if (rk_step, dynamics_substep) != (1, 1):
            raise RuntimeError("fork physics sources must be assembled before the first RK stage")
        step, time, flags, _ = self._pending
        self._tendencies = self._call("physics_get_tend", step, time, flags,
                                     {"pools": pools, "rk_step": rk_step, "dynamics_substep": dynamics_substep})
        self._tendencies_ready = True
        return self._tendencies

    def release_held_tendencies(self):
        """Drop this step's held physics_get_tend result once the runtime holds its own stage sources.

        runtime_v842 ResidentNativeCoupling.stage is the one caller: it runs once a step (coupling.sources is
        None only at the step start) and keeps its own sources, padded copies on a limited-area mesh. The held
        result is read by nothing after that. THE BREAKAGE THIS PREVENTS (vram census 2026-10-28, mv3km 1 card):
        the held unpadded mass, momentum, theta and scalar sources (4,250 B per column) stayed live beside the
        padded copies through every acoustic substep, the binding phase, and release_dynamics_sources could free
        nothing because this cache still held the arrays it dropped. A second request in the same step refuses.
        """
        if self._tendencies_ready:
            self._tendencies = _RELEASED_TENDENCIES

    def finish_step(self, *, pools):
        """Fork has no ACC pre/post hooks; MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1824-1878; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:643-1286."""
        if self._pending is None:
            raise RuntimeError("no physics step awaits post-transport microphysics")
        step, time, flags, _ = self._pending
        result = None
        if self.config.moist_physics:
            self._call("post_transport_prepare", step, time, flags, pools)
            if self.config.active("microphysics"):
                result = self._call("microphysics", step, time, flags, pools, slot="microphysics")
        self._pending = None
        return result

    def after_output(self, *, pools, diagnostics_alarm: bool, itimestep: int, time_seconds: float):
        """Reset only after diagnostics writes; MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:883-894,971-1002.

        The caller supplies the stream manager's diagnostics output alarm,
        not the earlier l_diags lookahead flag and not any output stream.
        """
        if self._pending is not None:
            raise RuntimeError("diagnostics output reset requires a completed timestep")
        if diagnostics_alarm:
            self._call("atm_reset_diagnostics", itimestep, time_seconds, StepFlags(), pools)


def coupling_selectors(options: Mapping[str, Any]) -> Mapping[str, str]:
    """Select identical arithmetic windows only; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_todynamics.F:330-405.

    Both fork MYNN cases contain the same qs/ni statements as the shared
    v8.4.2 MYNN case. TEMPO's aerosol triplet repeats exactly the Thompson
    aerosol statements. This mapping never selects a scheme driver.
    """
    result = dict(options)
    if result["config_pbl_scheme"] == "bl_mynnedmf":
        result["config_pbl_scheme"] = "bl_mynn"
    if result["config_microp_scheme"] == "mp_tempo":
        result["config_microp_scheme"] = ("mp_thompson_aerosols" if options["config_tempo_aerosolaware"] else "mp_wsm6")
    return MappingProxyType(result)


class StrictPhysicsKernelCache(KernelCache):
    """Fork RKIND compilation; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_todynamics.F:255-493."""

    def raw_kernel(self, name, source, *, module_key, options=()):
        """Preserve subnormal words; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:284-505."""
        import hashlib
        from cupy.cuda import compiler, function
        from ...cuda_backend.compile_contract import NvrtcCompileCapture, validate_module_key
        # CuPy RawModule appends --ftz=true (p5's measured FTZ defect).
        # Reuse f4's strict-cache pattern locally until its module lands.
        module_key = validate_module_key(module_key)
        if module_key == 'hexcore.cuda_physics_prep_v841':
            # The shared carriers (wind reconstruction, interface pressure) call sinf/cosf/logf/expf. The fork
            # build takes them from GNU libm, as v8.4.2 does (mpas_vector_reconstruction.F:303-323;
            # mpas_atmphys_interface.F:504-519), so they bind to the same GNU single-precision words the
            # v8.4.2 carrier cache uses. With CUDA's own sinf/cosf, uReconstructZonal/Meridional differed from
            # native in the last places at 732 of 2562 x1.2562 cells from the first step.
            source = bind_common_carrier_math(source)
        extra = (*options, '--fmad=false', '--ftz=false', '--prec-div=true', '--prec-sqrt=true')
        requested = _hex_numerics.options((*self.base_options, *extra))
        compiled_key = self._module_key(source, extra)
        key = f'{compiled_key}:{name}'
        manifest = self._compile_manifest
        manifest.declare_module(module_key, source=source, requested_options=requested,
                                module_cache_key=compiled_key)
        kernel = self._kernels.get(key)
        if kernel is None:
            module = self._modules.get(compiled_key)
            if module is None:
                with NvrtcCompileCapture() as capture:
                    image, _ = compiler.compile_using_nvrtc(source, options=requested, cache_in_memory=True)
                manifest.record_effective_compile(module_key, capture.evidence(
                    hashlib.sha256(source.encode()).hexdigest()))
                module = function.Module()
                module.load(image)
                self._modules[compiled_key] = module
            kernel = module.get_function(name)
            self._kernels[key] = kernel
        manifest.record_kernel(module_key, name)
        return self._observe(key, kernel, name, module_key)


def couple_physics_tendencies(*, options, kernel_cache, n_edges=None, **arguments):
    """Reuse word-identical coupling maths; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_todynamics.F:255-493."""
    from ...cuda_physics_v841 import couple_mpas_physics_tendencies_v842
    if not isinstance(kernel_cache, StrictPhysicsKernelCache):
        raise TypeError('fork coupling requires strict direct NVRTC; RawModule flushes native subnormals')
    # Native writes this persistent diagnostic through nEdges, while its
    # allocated descriptor includes the dummy edge. The shared work kernel
    # receives a resident extent; preserve the native descriptor's tail here.
    # MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_todynamics.F:327-333,434-442.
    tail = None
    if n_edges is not None:
        if not arguments['n_edges_solve'] <= n_edges <= arguments['tend_u_phys'].shape[1]:
            raise ValueError('native nEdges must contain solve edges and fit the allocated diagnostic')
        tail = arguments['tend_u_phys'][:, n_edges:].copy()
    result = couple_mpas_physics_tendencies_v842(selectors=coupling_selectors(options),
                                                kernel_cache=kernel_cache, **arguments)
    if tail is not None:
        arguments['tend_u_phys'][:, n_edges:] = tail
    return result


def prepare_number_carriers(prepared, *, options, kernel_cache, **arguments):
    """Identical MYNN/TEMPO scratch statements; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:357-505."""
    from ...cuda_physics_prep_v841 import prepare_mpas_number_carriers_v842
    if not isinstance(kernel_cache, StrictPhysicsKernelCache):
        raise TypeError('fork preparation requires strict direct NVRTC; RawModule flushes native subnormals')
    micro = options["config_microp_scheme"]
    if micro == "mp_tempo":
        micro = "mp_thompson_aerosols" if options["config_tempo_aerosolaware"] else "off"
    if micro == "mp_nssl2m":
        micro = "off"
    pbl = "bl_mynn" if options["config_pbl_scheme"] == "bl_mynnedmf" else options["config_pbl_scheme"]
    if pbl == "bl_myj":
        pbl = "off"
    return prepare_mpas_number_carriers_v842(prepared, microphysics=micro, pbl=pbl,
                                            kernel_cache=kernel_cache, **arguments)


def initialize_native_core(session, mesh):
    """Bind the complete Rust second ring after scheme initialization.

    MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:598,1481-1584.
    The r3 bridge owns traversal; this adapter copies native pool storage.
    """
    from ...core_rrfs import initialize_second_ring_rrfs
    solved = session.layout.dimensions['nCellsSolve']
    counts, neighbors = initialize_second_ring_rrfs(mesh, cell_solve_count=solved)
    # Native writes only owned counts and active neighbor entries. Preserve
    # halo, unused padding and dummy storage from the hydrated mesh pools.
    native_counts = session.native.get_field('mesh', 'nCellsOnCellsOnCell')
    native_neighbors = session.native.get_field('mesh', 'cellsOnCellsOnCell')
    native_counts[:solved] = counts[:solved]
    # The Rust bridge returns zero-based neighbors; native uses one-based.  Each owned cell's first ``count`` slots,
    # written as one masked assignment (the same words the per-cell loop wrote).  THE COST THIS REMOVES: the loop ran
    # once per owned cell on one host thread, 2.34 M Python iterations on NOAA's conus3km before the first step.
    import numpy as np
    width = native_neighbors.shape[0]
    active = np.arange(width)[:, None] < np.asarray(counts[:solved])[None, :]
    block = native_neighbors[:, :solved]
    block[active] = (np.asarray(neighbors[:solved, :width]).T + 1)[active]
    session.put('mesh', 'nCellsOnCellsOnCell', native_counts)
    session.put('mesh', 'cellsOnCellsOnCell', native_neighbors)


def bind_native_output_diagnostics(session, streams):
    """Set the hourly maximum reflectivity switch from the configured output streams.

    MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:126-129 at 1d6e370e:
    convective_diagnostics_setup sets do_diag_dbz to 1 when any output stream carries refl10cm_1km_max,
    and driver_microphysics (mpas_atmphys_driver_microphysics.F:557-560,801-806) then calls
    compute_hourly_max_radar_reflectivity every step. The native diagnostics framework that runs that
    setup is not part of the physics library, so the switch is set here from the same stream list;
    left at its Registry value of 0, refl10cm_1km_max stayed zero in every diagnostics file.
    """
    import numpy as np
    metadata = session.native._fields.get(('diag_physics', 'do_diag_dbz'))
    if metadata is None:
        return
    dtype, shape, _ = metadata
    included = any('output' in stream.directions and 'refl10cm_1km_max' in stream.fields
                   for stream in streams)
    session.put('diag_physics', 'do_diag_dbz', np.full(shape, 1 if included else 0, dtype=dtype))


def reset_native_diagnostics(native, session=None):
    """Reset whole associated allocations after successful diagnostics output.

    MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:883-894,971-1002.
    Zero allocation is storage orchestration; no weather payload is decoded.
    On a resident run the GPU pools are the authority for these three fields (TEMPO's radar and hail updates
    write them there), so the device words are zeroed too. THE BREAKAGE THIS PREVENTS: zeroing only the host
    storage left the device running maxima untouched, so from the second diagnostics write on every
    refl10cm_1km_max and max_hail_diameter_*_acc word kept the maximum since the start instead of since the last
    write (native-free hrrrv5 x1.2562, step 2: 542 diagnostics words differed from native).
    """
    import numpy as np
    views = getattr(session, 'device_pools', None)
    resident = views is not None and getattr(views, 'resident', False)
    config = getattr(session, 'config', None)
    row = specification(config)
    values = row.get('diagnostic_reset_values', {})
    names = tuple(row['diagnostic_resets'])
    if resident:
        # One CopyPlan for every reset word of the pinned row (44c0849 resets eight fields, compref to -35 dBZ).
        from .glue_plan import CopyPlan
        targets = []
        facade = views.native
        for name in names:
            metadata = native._fields.get(('diag_physics', name))
            if metadata is None:
                continue
            fill = values.get(name, 0)
            dtype, shape, levels = metadata
            value = views.field('diag_physics', name)
            targets.append((value, fill))
            # Bind current tails: a whole-allocation scheme write can replace
            # the facade's tail allocation between output events.
            if value.shape != shape:
                key = ('diag_physics', name, 1)
                tail = facade._field_rest.get(key)
                if tail is None:
                    tail = facade._field_rest[key] = views.cp.empty(
                        (*shape[:-1], shape[-1] - value.shape[-1]), dtype=dtype)
                targets.append((tail, fill))
        signature = tuple((int(a.data.ptr), a.shape, a.strides, fill) for a, fill in targets)
        plan = getattr(session, '_rrfs_reset_plan', None)
        if plan is None or getattr(session, '_rrfs_reset_signature', None) != signature:
            plan = CopyPlan('rrfs.atm_reset_diagnostics')
            for value, fill in targets:
                plan.fill(value, fill)
            session._rrfs_reset_plan = plan.build()
            session._rrfs_reset_signature = signature
        plan()
        return
    for name in names:
        value = values.get(name, 0)
        metadata = native._fields.get(('diag_physics', name))
        if metadata is not None:
            dtype, shape, levels = metadata
            native.put_field('diag_physics', name,
                             np.full(shape, value, dtype=dtype, order='F'), nlevels=levels)


def interface_cuda_source():
    """Complete shared-source delta; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:341-652."""
    from ...cuda_physics_prep_v841 import _CUDA_SOURCE
    source = _CUDA_SOURCE.replace('float *plrad, int *invalid)',
        'float *plrad, int *invalid, const float *pressure_perturbation, const float *pressure_base, float *weights_minus, float *weights_plus)')
    source = source.replace('psfc_p[cell] = mpas_add(hydro, pres_p[cell]);',
        'psfc_p[cell] = mpas_add(mpas_add(hydro, pressure_perturbation[cell]), pressure_base[cell]);')
    source = source.replace('t2_p[i] = mpas_add(\n            mpas_mul(fzm, t_p[i]),',
        'weights_minus[i] = fzm; weights_plus[i] = fzp;\n        t2_p[i] = mpas_add(\n            mpas_mul(fzm, t_p[i]),')
    return bind_common_carrier_math(source)


def prepare_mpas_to_physics(*, fields, species, options, scalar_flags, kernel_cache):
    """Fork preparation; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:284-652.

    Reuse identical v842 mass/interpolation/hydrostatic statements. Bind shared
    GNU device math and retain the fork's left-associated surface-pressure sum.
    Reconstructed winds are native diagnostic inputs, not reconstructed again.
    Persistent allocation tails belong to the caller, just as in native pools.
    """
    import cupy as cp
    import numpy as np
    from types import SimpleNamespace
    from ...cuda_physics_prep_v841 import _device_pointer_table, RV_OVER_RD_F32, GRAVITY_F32
    from ..math_v842 import bind_common_carrier_math
    if not isinstance(kernel_cache, StrictPhysicsKernelCache):
        raise TypeError('fork preparation requires strict direct NVRTC')
    from ...cuda_backend.containers import require_resident_array
    nlev, ncells = fields['rho_zz'].shape
    for name in ('rho_zz','theta_m','zz','exner','pressure_base','pressure_p',
                 'uReconstructZonal','uReconstructMeridional'):
        require_resident_array(name,fields[name],dtype=np.float32,shape=(nlev,ncells))
    for name in ('zgrid','w','fzm_p','fzp_p'):
        require_resident_array(name,fields[name],dtype=np.float32,shape=(nlev+1,ncells))
    names = ('qv','qc','qr','qi','qs','qg')
    for name,value in species.items():
        require_resident_array(name,value,dtype=np.float32,shape=(nlev,ncells))
    mass = ('zz_p','rho_dry','rho_p','th_p','t_p','pi_p','pres_p','zmid_p','dz_p',
            'pres_hyd_p','pres_hydd_p','znu_p','znu_hyd_p')
    interfaces = ('w_p','z_p','t2_p','pres2_p','pres2_hyd_p','pres2_hydd_p')
    surfaces = ('psfc_p','psfc_hyd_p','psfc_hydd_p','plrad')
    out = {name:cp.empty((nlev,ncells),dtype=cp.float32) for name in mass}
    out.update({name:cp.empty((nlev+1,ncells),dtype=cp.float32) for name in interfaces})
    out.update({name:cp.empty(ncells,dtype=cp.float32) for name in surfaces})
    scratch = {name:cp.empty((nlev,ncells),dtype=cp.float32) for name in names}
    scalars = cp.stack([species[name] for name in names])
    invalid = cp.zeros(1,dtype=cp.int32)
    pointer_table = _device_pointer_table(cp,[scratch[name] for name in names])
    def launch(name, args):
        """Launch native slice; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:341-652."""
        kernel = kernel_cache.raw_kernel(name,interface_cuda_source(),module_key='hexcore.mpasphys.rrfs.interface')
        kernel(((ncells+127)//128,), (128,), args)
    launch('prep_mass_v841_f32',(np.int32(nlev),np.int32(ncells),np.int32(len(names)),RV_OVER_RD_F32,
        fields['rho_zz'],fields['theta_m'],scalars,fields['exner'],fields['pressure_base'],
        fields['pressure_p'],fields['zgrid'],fields['zz'],pointer_table,
        *(out[name] for name in mass[:9]),invalid))
    out.update({name+'_p':value for name,value in scratch.items()})
    out['u_p'],out['v_p'] = fields['uReconstructZonal'].copy(),fields['uReconstructMeridional'].copy()
    # Native writes only interior interpolation weights. Carry PRE boundaries.
    out['fzm_p'],out['fzp_p'] = fields['fzm_p'].copy(),fields['fzp_p'].copy()
    launch('prep_interfaces_v841_f32',(np.int32(nlev),np.int32(ncells),GRAVITY_F32,
        fields['rho_zz'],fields['zz'],out['qv_p'],out['pres_p'],out['t_p'],out['rho_p'],
        out['dz_p'],fields['zgrid'],fields['w'],out['w_p'],out['z_p'],out['t2_p'],
        out['pres2_p'],out['pres_hyd_p'],out['pres2_hyd_p'],out['pres_hydd_p'],
        out['pres2_hydd_p'],out['psfc_p'],out['psfc_hyd_p'],out['psfc_hydd_p'],
        out['znu_p'],out['znu_hyd_p'],out['plrad'],invalid,fields['pressure_p'],
        fields['pressure_base'],out['fzm_p'],out['fzp_p']))
    # Number clamping is shared with the existing four-carrier delta.
    carrier_species = dict(species)
    for name in ('nc','ni','nifa','nwfa'):
        if name in carrier_species:
            carrier_species[name] = cp.maximum(cp.float32(0),carrier_species[name])
    out.update({name+'_p':value for name,value in prepare_number_carriers(
        SimpleNamespace(species_p=carrier_species), options=options,
        scalar_flags=scalar_flags,n_cells_solve=ncells,kernel_cache=kernel_cache).items()})
    del out['rho_dry']  # intermediate only; native exposes moist rho_p.
    return out


def _microphysics_carriers(options):
    """Full carrier selection; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:757-1026,1272-1505 at 1d6e370e."""
    scheme = options['config_microp_scheme']
    names = ['qv','qc','qr']
    if scheme in ('mp_wsm6','mp_thompson','mp_thompson_aerosols','mp_tempo','mp_nssl2m'):
        names += ['qi','qs','qg']
    if scheme in ('mp_thompson','mp_thompson_aerosols','mp_tempo','mp_nssl2m'):
        names += ['ni','nr']
    if scheme == 'mp_thompson_aerosols' or (scheme == 'mp_tempo' and options['config_tempo_aerosolaware']):
        names += ['nc','nifa','nwfa']
    if scheme == 'mp_tempo' and options['config_tempo_hailaware']:
        names += ['ng','volg']
    if scheme == 'mp_nssl2m':
        names += ['nc','qh','ns','ng','nh','nccn','volg','volh']
        if options['config_nssl_moments'] == 'nssl3m': names += ['zrw','zgw','zhw']
    return tuple(names)


def _microphysics_diagnostics(options, *, entering):
    """Diagnostic copies; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:767-936,1282-1474 at 1d6e370e."""
    scheme = options['config_microp_scheme'];pairs = {}
    if scheme in ('mp_wsm6','mp_thompson','mp_thompson_aerosols','mp_tempo','mp_nssl2m'):
        pairs.update(rainprod='rainprod_p',evapprod='evapprod_p',re_cloud='recloud_p',re_ice='reice_p',re_snow='resnow_p')
        # Native retrieves WSM/Thompson reflectivity at exit without writing it.
        if entering or scheme in ('mp_tempo','mp_nssl2m'):pairs['refl10cm']='refl10cm_p'
    if entering and scheme in ('mp_thompson','mp_thompson_aerosols'):pairs.update(nt_c='ntc_p',mu_c='muc_p')
    if scheme == 'mp_thompson_aerosols' or (scheme == 'mp_tempo' and options['config_tempo_aerosolaware']):
        pairs.update(nifa2d='nifa2d_p',nwfa2d='nwfa2d_p')
    if scheme == 'mp_tempo':
        pairs.update(max_hail_diameter_sfc='max_hail_diameter_sfc_p',max_hail_diameter_column='max_hail_diameter_column_p')
        if entering and options['config_tempo_ml_for_bl_nc']:pairs.update(qc_bl='qcbl_p',cldfrac_bl='cldfrac_p')
    return pairs


_MICROPHYSICS_HANDOFF_CUDA = r'''
// MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:1029-1127,1508-1623 at 1d6e370e.
extern "C" __global__ void r4_microp_pressure(int count,const float *base,const float *pert,float *out){
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i<count)out[i]=__fadd_rn(base[i],pert[i]);
}
extern "C" __global__ void r4_microp_rate(int count,int heat,float rvord,float dt,
 const float *current,const float *qv,const float *previous,float *out){
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=count)return;
 float v=current[i];if(heat)v=__fdiv_rn(v,__fadd_rn(1.f,__fmul_rn(rvord,fmaxf(0.f,qv[i]))));
 out[i]=__fdiv_rn(__fsub_rn(v,previous[i]),dt);
}
'''


def microphysics_handoff_source():
    """Identical source assembly; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:643-1286."""
    from ..kessler_v842 import CUDA_SOURCE, shared_math_header
    return shared_math_header()+CUDA_SOURCE+_MICROPHYSICS_HANDOFF_CUDA


def microphysics_from_mpas(*, fields, species, diagnostics, options, kernel_cache):
    """Complete entry handoff; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:643-1129 at 1d6e370e."""
    if not isinstance(kernel_cache, StrictPhysicsKernelCache):
        raise TypeError('fork handoff requires strict direct NVRTC; RawModule flushes native subnormals')
    import cupy as cp
    import numpy as np
    from ...cuda_physics_prep_v841 import RV_OVER_RD_F32
    from ...cuda_backend.containers import require_resident_array
    nz,nc = fields['theta_m'].shape
    for name in ('theta_m','rho_zz','zz','exner','pressure_base','pressure_p'):
        require_resident_array(name,fields[name],dtype=np.float32,shape=(nz,nc))
    require_resident_array('zgrid',fields['zgrid'],dtype=np.float32,shape=(nz+1,nc))
    names = _microphysics_carriers(options)
    for name in names:require_resident_array(name,species[name],dtype=np.float32,shape=(nz,nc))
    out = {name+'_p':species[name].copy() for name in names}
    out.update({name:cp.empty((nz,nc),dtype=cp.float32) for name in ('th_p','rho_p','pi_p','z_p','dz_p','pres_p')})
    require_resident_array('w',fields['w'],dtype=np.float32,shape=(nz+1,nc))
    out['w_p'] = fields['w'][:nz].copy()
    # Only common thermodynamic statements are reused; no Kessler scheme call.
    kernel = kernel_cache.raw_kernel('kessler_from_mpas',microphysics_handoff_source(),module_key='hexcore.mpasphys.rrfs.microphysics_interface')
    kernel(((nz*nc+127)//128,), (128,), (*(fields[k] if k=='theta_m' else species[k] for k in ('theta_m','qv','qc','qr')),
        *(fields[k] for k in ('rho_zz','zz','exner','zgrid')),
        *(out[k] for k in ('th_p','qv_p','qc_p','qr_p','rho_p','pi_p','z_p','dz_p')),RV_OVER_RD_F32,np.int32(nz),np.int32(nc)))
    kernel = kernel_cache.raw_kernel('r4_microp_pressure',microphysics_handoff_source(),module_key='hexcore.mpasphys.rrfs.microphysics_interface')
    kernel(((nz*nc+127)//128,), (128,), (np.int32(nz*nc),fields['pressure_base'],fields['pressure_p'],out['pres_p']))
    for name,target in _microphysics_diagnostics(options,entering=True).items():out[target]=diagnostics[name].copy()
    # The native tendency seed set excludes hail/NSSL extra moments.
    tendencies = {}
    if len(names)>3:
        tendencies['rthmpten']=out['th_p'].copy()
        for name in ('qv','qc','qr','qi','qs','qg','ni','nr','nc','nifa','nwfa'):
            if name in names:tendencies['r'+name+'mpten']=species[name].copy()
    return out,tendencies


def microphysics_to_mpas(*, fields, species, prepared, previous_tendencies, options, dt_dyn, kernel_cache,
                         mp_top_level=None, previous_diagnostics=None):
    """Complete exit handoff; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:1132-1625 at 1d6e370e."""
    if not isinstance(kernel_cache, StrictPhysicsKernelCache):
        raise TypeError('fork handoff requires strict direct NVRTC; RawModule flushes native subnormals')
    import cupy as cp
    import numpy as np
    from ...cuda_physics_prep_v841 import RV_OVER_RD_F32,RD_F32,GRAVITY_F32
    from ...cuda_backend.containers import require_resident_array
    nz,nc = fields['theta_m'].shape
    top = nz if mp_top_level is None else int(mp_top_level)
    volume = ('theta_m','rho_zz','zz','exner','exner_base','rtheta_base','rtheta_p','pressure_p','pressure_base')
    for name in volume:require_resident_array(name,fields[name],dtype=np.float32,shape=(nz,nc))
    for name in ('surface_pressure',):require_resident_array(name,fields[name],dtype=np.float32,shape=(nc,))
    require_resident_array('zgrid',fields['zgrid'],dtype=np.float32,shape=(nz+1,nc))
    names = _microphysics_carriers(options)
    for name in names:require_resident_array(name,species[name],dtype=np.float32,shape=(nz,nc))
    for name in ('th_p',*(n+'_p' for n in names)):require_resident_array(name,prepared[name],dtype=np.float32,shape=(nz,nc))
    result = {name:fields[name].copy() for name in volume}
    result['surface_pressure'] = fields['surface_pressure'].copy()
    result.update({name:cp.empty((nz,nc),dtype=cp.float32) for name in ('rt_diabatic_tend','dtheta_dt_mp')})
    if top < nz:
        # 44c0849 interface:1272-1318 leaves dry heating above the top untouched.
        result['dtheta_dt_mp'][top:] = fields['dtheta_dt_mp'][top:]
    result['tend_sfc_pressure'] = cp.empty(nc,dtype=cp.float32)
    new_species = {name:prepared[name+'_p'].copy() for name in names}
    # The common kernel updates qv before evaluating old dry heating; provide
    # independent old species to preserve native PRE values at that statement.
    old_water = {name:species[name].copy() for name in ('qv','qc','qr')}
    kernel = kernel_cache.raw_kernel('kessler_to_mpas',microphysics_handoff_source(),module_key='hexcore.mpasphys.rrfs.microphysics_interface')
    rcv = np.float32(RD_F32 / np.float32(np.float32(np.float32(7)*RD_F32/np.float32(2))-RD_F32))
    kernel(((nc+127)//128,), (128,), (result['theta_m'],*(old_water[k] for k in ('qv','qc','qr')),
        fields['rho_zz'],fields['zz'],result['exner'],fields['exner_base'],fields['rtheta_base'],result['rtheta_p'],
        result['pressure_p'],fields['pressure_base'],fields['zgrid'],result['surface_pressure'],result['tend_sfc_pressure'],
        result['rt_diabatic_tend'],result['dtheta_dt_mp'],prepared['th_p'],*(prepared[k+'_p'] for k in ('qv','qc','qr')),
        RV_OVER_RD_F32,RD_F32,np.float32(100000),rcv,GRAVITY_F32,np.float32(dt_dyn),np.int32(top),np.int32(nz),np.int32(nc)))
    new_species.update(old_water)
    diagnostics = {name:prepared[target].copy() for name,target in _microphysics_diagnostics(options,entering=False).items()}
    if top < nz and options['config_microp_scheme'] in ('mp_wsm6','mp_thompson','mp_thompson_aerosols'):
        # 44c0849 interface:1368-1385; TEMPO and NSSL retain their full-height copies.
        for name in ('qi','qs','qg'):
            new_species[name][top:] = 0
        for name in ('rainprod','evapprod'):
            diagnostics[name][top:] = 0
        for name, background in (('re_cloud',2.49e-6),('re_ice',4.99e-6),('re_snow',9.99e-6)):
            diagnostics[name][top:] = np.float32(background)
    if mp_top_level is not None and options['config_microp_scheme'] in ('mp_thompson','mp_thompson_aerosols','mp_tempo'):
        # 44c0849 interface:1405-1410,1474-1481. These are pool diagnostics,
        # not scheme arguments, so preserve the existing maximum explicitly.
        diagnostics['refl10cm_compref_max'] = cp.maximum(
            previous_diagnostics['refl10cm_compref_max'], previous_diagnostics['refl10cm_max'])
    tendencies = {}
    if len(names)>3:
        rate_names = ['qv','qc','qr','qi','qs','qg']
        # NSSL seeds ni/nr/nc on entry but native does not finish those rates.
        if options['config_microp_scheme'] != 'mp_nssl2m':rate_names += [n for n in ('ni','nr','nc','nifa','nwfa') if n in names]
        for name in ('th',*rate_names):
            key='r'+name+'mpten';old=previous_tendencies[key]
            require_resident_array(key,old,dtype=np.float32,shape=(nz,nc))
            value=result['theta_m'] if name=='th' else new_species[name]
            output=cp.empty_like(old)
            kernel=kernel_cache.raw_kernel('r4_microp_rate',microphysics_handoff_source(),module_key='hexcore.mpasphys.rrfs.microphysics_interface')
            kernel(((nz*nc+127)//128,), (128,), (np.int32(nz*nc),np.int32(name=='th'),RV_OVER_RD_F32,np.float32(dt_dyn),value,new_species['qv'],old,output))
            if name in ('th','qv','qc','qr','qi','qs','qg'):
                output[top:] = 0
            tendencies[key]=output
    return result,new_species,diagnostics,tendencies


def compute_native_output_diagnostics(*, fields, qv, kernel_cache):
    """Complete physics output fields; MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:919-968 at 1d6e370e."""
    if not isinstance(kernel_cache, StrictPhysicsKernelCache):
        raise TypeError('fork output requires strict direct NVRTC; RawModule flushes native subnormals')
    import cupy as cp
    import numpy as np
    from ...cuda_backend.containers import require_resident_array
    from ...cuda_physics_prep_v841 import RV_OVER_RD_F32
    nz, nc = fields['theta_m'].shape
    shape = (nz, nc)
    for name in ('theta_m', 'rho_zz', 'zz', 'pressure_base', 'pressure_p'):
        require_resident_array(name, fields[name], dtype=np.float32, shape=shape)
    require_resident_array('qv', qv, dtype=np.float32, shape=shape)
    out = {name: cp.empty(shape, dtype=cp.float32)
           for name in ('theta', 'rho', 'pressure')}
    count = nz * nc
    if count == 0:
        return out
    from ..diagnostics_v842 import native_diagnostic_source
    kernel = kernel_cache.raw_kernel('mpas_output_diagnostics_v842', native_diagnostic_source(),
                                    module_key='hexcore.mpasphys.rrfs.output')
    kernel(((count + 127) // 128,), (128,),
           (np.int32(count), RV_OVER_RD_F32, fields['theta_m'], qv,
            fields['rho_zz'], fields['zz'], fields['pressure_base'],
            fields['pressure_p'], out['theta'], out['rho'], out['pressure']))
    return out


_SURFACE_INOUT = (
    'br', 'cpm', 'chs', 'chs2', 'cqs2', 'flhc', 'flqc', 'gz1oz0', 'hfx',
    'qfx', 'qgh', 'qsfc', 'lh', 'mol', 'psim', 'psih', 'regime', 'rmol',
    'ust', 'ustm', 'wspd', 'znt', 'zol',
)
_SURFACE_OUTPUT = ('q2', 't2m', 'th2m', 'u10', 'v10', 'cd', 'cda', 'ck', 'cka')
_SURFACE_BLEND = (
    'br', 'gz1oz0', 'mol', 'psih', 'psim', 'rmol', 'ust', 'wspd', 'zol',
    'q2', 't2m', 'th2m', 'u10', 'v10', 'cd', 'cda', 'ck', 'cka', 'ustm',
)
_MYJ_SURFACE_INOUT = ('thz0', 'qz0', 'uz0', 'vz0', 'akms', 'akhs')
_MYJ_SURFACE_OUTPUT = ('ct', 'tshltr', 'th10', 'qshltr', 'q10', 'pshltr')

_SURFACE_CUDA = r"""
extern "C" __global__ void rrfs_pbl_radiation_sum(
    int n, const float *lw, const float *sw, float *out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) out[i] = __fadd_rn(lw[i], sw[i]);
}
extern "C" __global__ void rrfs_surface_spacing(
    int n, const float *density, float length, float *dx) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) dx[i] = __fdiv_rn(length, mpas_native_powf_positive(density[i], 0.25f));
}
extern "C" __global__ void rrfs_surface_blend(
    int n, const float *main, const float *sea, const float *ice,
    float threshold, float *out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float x = ice[i];
    out[i] = x >= threshold && x < 1.0f
        ? __fadd_rn(__fmul_rn(main[i], x),
                    __fmul_rn(__fsub_rn(1.0f, x), sea[i])) : main[i];
}
"""


def surface_layer_from_mpas(*, pools, prepared, options, its, ite,
                            jts, jte, kts, kte, xice_threshold, kernel_cache):
    """Complete entry packing; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:434-769 at 1d6e370e.

    Pools retain MPAS (level, cell) layouts; prepared retains native (cell,
    level, tile) layouts and complete allocations. Only the active tile is
    changed. The caller owns native allocation, pool identity and bounds.
    """
    import cupy as cp
    import numpy as np
    from ...native_powf import CUDA_NATIVE_POWF
    from ...cuda_backend.containers import require_resident_array
    for name, value in prepared.items():
        require_resident_array(name, value, dtype=value.dtype, shape=value.shape)
    for pool, fields in pools.items():
        for name, value in fields.items():
            require_resident_array(pool + '.' + name, value, dtype=value.dtype, shape=value.shape)
    out = {name: value.copy() for name, value in prepared.items()}
    cells, tiles = slice(its - 1, ite), slice(jts - 1, jte)
    nc = ite - its + 1
    diag, sfc = pools['diag_physics'], pools['sfc_input']
    frac = bool(options['config_frac_seaice'])
    scheme = options['config_sfclayer_scheme']

    def put(name, value):
        # Native writes all active j tiles from the same cell pool.
        out[name][cells, tiles] = value[:, None]

    def copy(name, field, pool=diag):
        put(name, pool[field][cells])

    density = cp.ascontiguousarray(pools['mesh']['meshDensity'][cells])
    dx = cp.empty(nc, dtype=cp.float32)
    kernel_cache.raw_kernel('rrfs_surface_spacing', CUDA_NATIVE_POWF + _SURFACE_CUDA,
                           module_key='rrfs.driver.surface.spacing')(
        ((nc + 127) // 128,), (128,),
        (np.int32(nc), density, np.float32(options['config_len_disp']), dx))
    put('dx_p', dx)
    for field in ('hpbl', 'mavail'):
        copy(field + '_p', field)
    for name, field in (('tsk_p', 'skintemp'), ('xland_p', 'xland'), ('xice_p', 'xice')):
        copy(name, field, sfc)
    for field in _SURFACE_INOUT:
        copy(field + '_p', field)
    for field in _SURFACE_OUTPUT:
        out[field + '_p'][cells, tiles] = cp.float32(0)

    if frac:
        copy('sst_p', 'sst', sfc)
        for field in ('mavail', *_SURFACE_INOUT):
            if field != 'qsfc':
                copy(field + '_sea', field)
        copy('tsk_sea', 'skintemp', sfc)
        copy('xland_sea', 'xland', sfc)
        copy('regime_hold', 'regime')
        for field in (*_SURFACE_OUTPUT, 'qsfc'):
            out[field + '_sea'][cells, tiles] = cp.float32(0)
        ice = sfc['xice'][cells]
        mixed = (ice >= np.float32(xice_threshold)) & (ice < cp.float32(1))
        put('xland_sea', cp.where(mixed, cp.float32(2), sfc['xland'][cells]))
        put('xland_p', cp.where(mixed, cp.float32(1), sfc['xland'][cells]))
        put('mavail_sea', cp.where(mixed, cp.float32(1), diag['mavail'][cells]))
        put('znt_sea', cp.where(mixed, cp.float32(0.0001), diag['znt'][cells]))
        put('tsk_sea', cp.where(mixed, cp.maximum(sfc['sst'][cells], cp.float32(271.4)),
                                 sfc['skintemp'][cells]))

    if scheme in ('sf_monin_obukhov', 'sf_monin_obukhov_rev'):
        for field in ('fh', 'fm'):
            copy(field + '_p', field)
            if frac:
                copy(field + '_sea', field)
        if scheme == 'sf_monin_obukhov_rev':
            for field in ('waterdepth_p', 'lakedepth_p', 'lakemask_p'):
                out[field][cells, tiles] = cp.float32(0)
    elif scheme in ('sf_mynn', 'sf_mynnsfclay'):
        copy('snowh_p', 'snowh', sfc)
        if scheme == 'sf_mynn':
            copy('qcg_p', 'qcg')
        for field in (('ch',) if scheme == 'sf_mynn' else ('ch', 'cqs')):
            copy(field + '_p', field)
            if frac:
                copy(field + '_sea', field)
    elif scheme == 'sf_myj':
        copy('ht_p', 'zgrid', {'zgrid': pools['mesh']['zgrid'][0]})
        for field in ('z0', 'lowlyr', *_MYJ_SURFACE_INOUT):
            copy(field + '_p', field)
        for field in _MYJ_SURFACE_OUTPUT:
            out[field + '_p'][cells, tiles] = cp.float32(0)
        out['tkepbl_p'][cells, kts - 1:kte, tiles] = (
            diag['tke_pbl'][kts - 1:kte, cells].T[:, :, None])
        if frac:
            for field in _MYJ_SURFACE_INOUT:
                copy(field + '_sea', field)
            for field in _MYJ_SURFACE_OUTPUT:
                out[field + '_sea'][cells, tiles] = cp.float32(0)
    return out


def surface_layer_to_mpas(*, pools, prepared, options, its, ite,
                          jts, jte, xice_threshold, kernel_cache):
    """Complete writeback/blend; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:772-1052 at 1d6e370e.

    Fluxes/coefficient fields that native leaves on the main branch remain on
    that branch. In particular the fork blends ch, but does not blend cqs.
    """
    import cupy as cp
    import numpy as np
    from ...native_powf import CUDA_NATIVE_POWF
    from ...cuda_backend.containers import require_resident_array
    for name, value in prepared.items():
        require_resident_array(name, value, dtype=value.dtype, shape=value.shape)
    for pool, fields in pools.items():
        for name, value in fields.items():
            require_resident_array(pool + '.' + name, value, dtype=value.dtype, shape=value.shape)
    diag = {name: value.copy() for name, value in pools['diag_physics'].items()}
    cells = slice(its - 1, ite)
    nc = ite - its + 1
    frac = bool(options['config_frac_seaice'])
    scheme = options['config_sfclayer_scheme']
    blend = kernel_cache.raw_kernel('rrfs_surface_blend', CUDA_NATIVE_POWF + _SURFACE_CUDA,
                                    module_key='rrfs.driver.surface.blend')
    ice = cp.ascontiguousarray(pools['sfc_input']['xice'][cells]) if frac else None

    def write(field, *, mixing=False):
        # Native loop assigns the cell repeatedly for jts:jte; last tile wins.
        main = cp.ascontiguousarray(prepared[field + '_p'][cells, jte - 1])
        if mixing and frac:
            sea = cp.ascontiguousarray(prepared[field + '_sea'][cells, jte - 1])
            value = cp.empty_like(main)
            blend(((nc + 127) // 128,), (128,),
                  (np.int32(nc), main, sea, ice, np.float32(xice_threshold), value))
        else:
            value = main
        diag[field][cells] = value

    for field in (*_SURFACE_INOUT, *_SURFACE_OUTPUT):
        write(field, mixing=field in _SURFACE_BLEND)
    if frac:
        mixed = (ice >= np.float32(xice_threshold)) & (ice < cp.float32(1))
        diag['regime'][cells] = cp.where(
            mixed & (ice >= cp.float32(0.5)),
            prepared['regime_hold'][cells, jte - 1], diag['regime'][cells])
    if scheme in ('sf_monin_obukhov', 'sf_monin_obukhov_rev'):
        for field in ('fh', 'fm'):
            write(field, mixing=True)
    elif scheme in ('sf_mynn', 'sf_mynnsfclay'):
        write('ch', mixing=True)
        if scheme == 'sf_mynnsfclay':
            write('cqs')
    elif scheme == 'sf_myj':
        for field in (*_MYJ_SURFACE_INOUT, *_MYJ_SURFACE_OUTPUT):
            write(field, mixing=True)
        write('hpbl')
    return diag


_PBL_VOLUME = {
    'elpbl_p': 'el_pbl', 'cov_p': 'cov', 'qke_p': 'qke',
    'qsq_p': 'qsq', 'tsq_p': 'tsq', 'qkeadv_p': 'qke_adv',
    'sh3d_p': 'sh3d', 'sm3d_p': 'sm3d', 'cldfrabl_p': 'cldfrac_bl',
    'qcbl_p': 'qc_bl', 'qibl_p': 'qi_bl', 'edmfa_p': 'edmf_a',
    'edmfent_p': 'edmf_ent', 'edmfqc_p': 'edmf_qc', 'edmfqt_p': 'edmf_qt',
    'edmfthl_p': 'edmf_thl', 'edmfw_p': 'edmf_w',
    'subthl_p': 'sub_thl', 'subqv_p': 'sub_qv',
    'detthl_p': 'det_thl', 'detqv_p': 'det_qv',
}
_PBL_COMMON_TEND = ('rublten', 'rvblten', 'rthblten', 'rqvblten', 'rqcblten', 'rqiblten')
_PBL_DIAG_TEND = ('dqke', 'qbuoy', 'qdiss', 'qshear', 'qwt')
_PBL_NUMBER_TEND = ('rqsblten', 'rncblten', 'rniblten', 'rnifablten', 'rnwfablten')


def pbl_from_mpas(*, pools, prepared, options, flags, dimensions,
                  its, ite, jts, jte, kts, kte, kernel_cache, extra_surface_fields=()):
    """Complete PBL entry; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:461-915 at 1d6e370e.

    Retain native pool and full scratch layouts, including inactive allocation
    words. Allocation and scheme calls remain separate native/scheme hooks.
    Flags and dimensions are native scalar metadata, never field computations.
    """
    import cupy as cp
    import numpy as np
    from ...native_powf import CUDA_NATIVE_POWF
    from ...cuda_backend.containers import require_resident_array
    for name, value in prepared.items():
        require_resident_array(name, value, dtype=value.dtype, shape=value.shape)
    for pool, fields in pools.items():
        for name, value in fields.items():
            require_resident_array(pool + '.' + name, value, dtype=value.dtype, shape=value.shape)
    out = {name: value.copy() for name, value in prepared.items()}
    scalars = {}
    cells, levels, tiles = slice(its - 1, ite), slice(kts - 1, kte), slice(jts - 1, jte)
    nc, nz = ite - its + 1, kte - kts + 1
    diag, sfc, tend = pools['diag_physics'], pools['sfc_input'], pools['tend_physics']

    def surface(name, value):
        out[name][cells, tiles] = value[:, None] if getattr(value, 'ndim', 0) else value

    def volume(name, value):
        out[name][cells, levels, tiles] = value[:, :, None] if getattr(value, 'ndim', 0) else value

    for field in ('hfx', 'hpbl', 'qfx', 'ust', 'wspd', 'znt'):
        surface(field + '_p', diag[field][cells])
    surface('xland_p', sfc['xland'][cells])
    surface('kpbl_p', cp.int32(1))
    for name in ('uoce_p', 'voce_p'):
        surface(name, cp.float32(0))
    lw = cp.ascontiguousarray(tend['rthratenlw'][levels, cells].T)
    sw = cp.ascontiguousarray(tend['rthratensw'][levels, cells].T)
    radiation = cp.empty((nc, nz), dtype=cp.float32)
    kernel_cache.raw_kernel('rrfs_pbl_radiation_sum', CUDA_NATIVE_POWF + _SURFACE_CUDA,
                           module_key='rrfs.driver.pbl.radiation')(
        ((nc * nz + 127) // 128,), (128,), (np.int32(nc * nz), lw, sw, radiation))
    volume('rthraten_p', radiation)

    if flags['do_chemistry']:
        nchem = int(flags['num_chem'])
        start = int(dimensions['chemistry_start']) - 1
        chemistry = pools['state']['scalars'][start:start + nchem, levels, cells]
        out['chem_p'][cells, levels, tiles, :nchem] = chemistry.transpose(2, 1, 0)[:, :, None, :]
        deposition = int(options['drydep_opt']) != 0
        if options['config_mynn_enh_mix']:
            surface('frp_out_p', diag['frp_out'][cells]
                    if options['config_smoke_scheme'].strip() == 'on' else cp.float32(0))
            surface('aero_emis_for_enhmix_p', diag['aero_emis_for_enhmix'][cells]
                    if options['config_anthro_scheme'].strip() != 'off'
                    or options['config_rwc_scheme'].strip() != 'off' else cp.float32(0))
        if deposition:
            out['ddvel_p'][cells, tiles, :nchem] = diag['ddvel'][:nchem, cells].T[:, None, :]
            out['tend_chem_settle_p'][cells, levels, tiles, :nchem] = (
                tend['tend_chem_settle'][:nchem, levels, cells].transpose(2, 1, 0)[:, :, None, :])
        else:
            # Native ddvel assignment spans the complete species allocation.
            if nchem:
                out['ddvel_p'][cells, tiles, :] = cp.float32(0)
            out['tend_chem_settle_p'][cells, levels, tiles, :nchem] = cp.float32(0)

    scheme = options['config_pbl_scheme'].strip()
    if scheme == 'bl_ysu':
        scalars['ysu_pblmix'] = int(bool(options['config_ysu_pblmix']))
        for name, field in (('br_p', 'br'), ('psim_p', 'fm'), ('psih_p', 'fh'),
                            ('u10_p', 'u10'), ('v10_p', 'v10'),
                            ('delta_p', 'delta'), ('wstar_p', 'wstar')):
            surface(name, diag[field][cells])
        for name in ('ctopo_p', 'ctopo2_p'):
            surface(name, cp.float32(1))
        volume('exch_p', cp.float32(0))
    elif scheme in ('bl_mynn', 'bl_mynnedmf'):
        density = cp.ascontiguousarray(pools['mesh']['meshDensity'][cells])
        dx = cp.empty(nc, dtype=cp.float32)
        kernel_cache.raw_kernel('rrfs_surface_spacing', CUDA_NATIVE_POWF + _SURFACE_CUDA,
                               module_key='rrfs.driver.surface.spacing')(
            ((nc + 127) // 128,), (128,),
            (np.int32(nc), density, np.float32(options['config_len_disp']), dx))
        surface('dx_p', dx)
        for field in ('ch', 'qsfc'):
            surface(field + '_p', diag[field][cells])
        surface('tsk_p', sfc['skintemp'][cells])
        for name, field in _PBL_VOLUME.items():
            volume(name, diag[field][levels, cells].T)
        if scheme == 'bl_mynn':
            surface('rmol_p', diag['rmol'][cells])
            volume('tkepbl_p', diag['tke_pbl'][levels, cells].T)
            surface('kbl_plume_p', cp.int32(0))
        else:
            for name in ('excessh_p', 'excessq_p', *(n + '_p' for n in extra_surface_fields)):
                surface(name, cp.float32(0))
        for field in (*_PBL_DIAG_TEND, *_PBL_NUMBER_TEND):
            volume(field + '_p', cp.float32(0))
        volume('pattern_spp_pbl', cp.float32(0))
        for name in ('maxwidthbl_p', 'maxmfbl_p', 'zbl_plume_p'):
            surface(name, cp.float32(0))
    elif scheme == 'bl_myj':
        surface('ht_p', pools['mesh']['zgrid'][0, cells])
        for field in ('lowlyr', 'chlowq', 'lh', 'thz0', 'qz0', 'uz0', 'vz0', 'akms', 'akhs', 'qsfc'):
            surface(field + '_p', diag[field][cells])
        surface('tsk_p', sfc['skintemp'][cells])
        for name in ('ct_p', 'mixht_p'):
            surface(name, cp.float32(0))
        volume('tkepbl_p', diag['tke_pbl'][levels, cells].T)
    for field in (*_PBL_COMMON_TEND, 'kzh', 'kzm', 'kzq'):
        volume(field + '_p', cp.float32(0))
    return out, scalars


def pbl_to_mpas(*, pools, prepared, options, flags, dimensions,
                its, ite, jts, jte, kts, kte, extra_surface_fields=()):
    """Complete PBL writeback; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F:918-1283 at 1d6e370e.

    Native kzq is assigned kzh_p. Preserve that pinned behavior and all pool
    allocation tails; every tile writes in native increasing-j order.
    """
    from ...cuda_backend.containers import require_resident_array
    for name, value in prepared.items():
        require_resident_array(name, value, dtype=value.dtype, shape=value.shape)
    for pool, fields in pools.items():
        for name, value in fields.items():
            require_resident_array(pool + '.' + name, value, dtype=value.dtype, shape=value.shape)
    out = {pool: {name: value.copy() for name, value in fields.items()}
           for pool, fields in pools.items()}
    cells, levels = slice(its - 1, ite), slice(kts - 1, kte)
    scheme = options['config_pbl_scheme'].strip()
    for j in range(jts - 1, jte):
        def surface(field, name=None):
            out['diag_physics'][field][cells] = prepared[name or field + '_p'][cells, j]

        def volume(pool, field, name=None):
            out[pool][field][levels, cells] = prepared[name or field + '_p'][cells, levels, j].T

        for field in ('hpbl', 'kpbl'):
            surface(field)
        for field in _PBL_COMMON_TEND:
            volume('tend_physics', field)
        for field, name in (('kzh', 'kzh_p'), ('kzm', 'kzm_p'), ('kzq', 'kzh_p')):
            volume('diag_physics', field, name)
        if flags['do_chemistry']:
            nchem = int(flags['num_chem'])
            start = int(dimensions['chemistry_start']) - 1
            out['state']['scalars'][start:start + nchem, levels, cells] = (
                prepared['chem_p'][cells, levels, j, :nchem].transpose(2, 1, 0))
        if scheme == 'bl_ysu':
            for field in ('delta', 'wstar'):
                surface(field)
            volume('diag_physics', 'exch_h', 'exch_p')
        elif scheme in ('bl_mynn', 'bl_mynnedmf'):
            for name, field in _PBL_VOLUME.items():
                volume('diag_physics', field, name)
            for field in _PBL_DIAG_TEND:
                volume('diag_physics', field)
            volume('tend_physics', 'rqsblten')
            if flags['f_ni']:
                volume('tend_physics', 'rniblten')
            if flags['f_nc'] and flags['f_nifa'] and flags['f_nwfa']:
                for field in ('rncblten', 'rnifablten', 'rnwfablten'):
                    volume('tend_physics', field)
            if scheme == 'bl_mynn':
                volume('diag_physics', 'tke_pbl', 'tkepbl_p')
            else:
                for field, name in (('maxmf', 'maxmfbl_p'), ('maxwidth', 'maxwidthbl_p'),
                                    ('ztop_plume', 'zbl_plume_p'),
                                    ('excess_h', 'excessh_p'), ('excess_q', 'excessq_p'),
                                    *((n, n + '_p') for n in extra_surface_fields)):
                    surface(field, name)
        elif scheme == 'bl_myj':
            for field in ('thz0', 'qz0', 'uz0', 'vz0', 'akms', 'akhs', 'qsfc', 'ust', 'znt', 'ct', 'mixht'):
                surface(field)
            volume('diag_physics', 'tke_pbl', 'tkepbl_p')
            volume('diag_physics', 'el_pbl', 'elpbl_p')
    return out


_CONVECTION_COMMON_TEND = ('rthcuten', 'rqvcuten', 'rqccuten', 'rqicuten')
_CONVECTION_GF_SURFACE = ('cubot', 'cutop', 'xmb_total', 'xmb_shallow',
                          'k22_shallow', 'kbcon_shallow', 'ktop_shallow', 'ktop_deep')
_CONVECTION_FORCING_CUDA = r"""
extern "C" __global__ void rrfs_convection_forcing(
    int n, const float *qdy, const float *qbl, const float *tdy,
    const float *tbl, const float *lw, const float *sw, const float *pi,
    float *qout, float *tout) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    qout[i] = __fadd_rn(qdy[i], qbl[i]);
    tout[i] = __fmul_rn(__fadd_rn(__fadd_rn(__fadd_rn(tdy[i], tbl[i]), lw[i]), sw[i]), pi[i]);
}
"""


def convection_from_mpas(*, pools, prepared, options, flags, dimensions,
                         its, ite, jts, jte, kts, kte, kernel_cache):
    """Complete entry; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:820-1167 at 1d6e370e.

    Full native pool/scratch indices, allocations and unassigned tails remain
    intact. Scheme numerical kernels remain their owners' callbacks. The GF
    source's rvcuten(k,k) diagonal read is deliberately preserved.
    """
    import cupy as cp
    import numpy as np
    from ...native_powf import CUDA_NATIVE_POWF
    from ...cuda_backend.containers import require_resident_array
    for name, value in prepared.items():
        require_resident_array(name, value, dtype=value.dtype, shape=value.shape)
    for pool, fields in pools.items():
        for name, value in fields.items():
            require_resident_array(pool + '.' + name, value, dtype=value.dtype, shape=value.shape)
    out = {name: value.copy() for name, value in prepared.items()}
    cells, levels, tiles = slice(its - 1, ite), slice(kts - 1, kte), slice(jts - 1, jte)
    nc, nz = ite - its + 1, kte - kts + 1
    diag, sfc, tend = pools['diag_physics'], pools['sfc_input'], pools['tend_physics']

    def surface(name, value):
        out[name][cells, tiles] = value[:, None]

    def volume(name, value):
        out[name][cells, levels, tiles] = value[:, :, None]

    for name, field in (('raincv_p', 'raincv'), ('rainncv_p', 'rainncv'), ('pratec_p', 'cuprec')):
        surface(name, diag[field][cells])
    for field in _CONVECTION_COMMON_TEND:
        volume(field + '_p', tend[field][levels, cells].T)
    if flags['do_chemistry']:
        start, nchem = int(dimensions['chemistry_start']) - 1, int(flags['num_chem'])
        out['chem_p'][cells, levels, tiles, :nchem] = (
            pools['state']['scalars'][start:start + nchem, levels, cells].transpose(2, 1, 0)[:, :, None, :])
    scheme = options['config_convection_scheme'].strip()
    if scheme in ('cu_grell_freitas', 'cu_grell_freitas_li', 'cu_kain_fritsch', 'cu_ntiedtke'):
        density = cp.ascontiguousarray(pools['mesh']['meshDensity'][cells])
        dx = cp.empty(nc, dtype=cp.float32)
        kernel_cache.raw_kernel('rrfs_surface_spacing', CUDA_NATIVE_POWF + _SURFACE_CUDA,
                               module_key='rrfs.driver.surface.spacing')(
            ((nc + 127) // 128,), (128,),
            (np.int32(nc), density, np.float32(options['config_len_disp']), dx))
        surface('dx_p', dx)
    if scheme in ('cu_grell_freitas', 'cu_grell_freitas_li'):
        surface('area_p', pools['mesh']['areaCell'][cells])
        surface('ht_p', pools['mesh']['zgrid'][0, cells])
        surface('xland_p', sfc['xland'][cells])
        for field in ('gsw', 'hfx', 'qfx', 'kpbl', *_CONVECTION_GF_SURFACE):
            surface(field + '_p', diag[field][cells])
        for name, field in (('qccu_p', 'qc_cu'), ('qicu_p', 'qi_cu')):
            volume(name, diag[field][levels, cells].T)
        for field in ('rthblten', 'rthdynten', 'rqvblten', 'rqvdynten', 'rucuten'):
            volume(field + '_p', tend[field][levels, cells].T)
        if scheme == 'cu_grell_freitas':
            volume('rvcuten_p', cp.broadcast_to(cp.diagonal(tend['rvcuten'])[levels], (nc, nz)))
        else:
            volume('rvcuten_p', tend['rvcuten'][levels, cells].T)
            surface('bilbc_p', pools['mesh']['bdyMaskCell'][cells])
            for name, field in (('vcpool_p', 'vcpool'), ('sigma_deep_p', 'sigma_deep'),
                                ('sigma_deep_f_p', 'sigma_deep_far')):
                surface(name, diag[field][cells])
            for field in ('cldfrac_cu', 'refl10cm_cu'):
                volume(field + '_p', diag[field][levels, cells].T)
        lw, sw = (cp.ascontiguousarray(tend[field][levels, cells].T)
                  for field in ('rthratenlw', 'rthratensw'))
        radiation = cp.empty((nc, nz), dtype=cp.float32)
        kernel_cache.raw_kernel('rrfs_pbl_radiation_sum', CUDA_NATIVE_POWF + _SURFACE_CUDA,
                               module_key='rrfs.driver.pbl.radiation')(
            ((nc * nz + 127) // 128,), (128,), (np.int32(nc * nz), lw, sw, radiation))
        volume('rthraten_p', radiation)
    elif scheme == 'cu_kain_fritsch':
        surface('area_p', pools['mesh']['areaCell'][cells])
        for field in ('cubot', 'cutop', 'nca'):
            surface(field + '_p', diag[field][cells])
        volume('w0avg_p', diag['w0avg'][levels, cells].T)
        for field in ('rqrcuten', 'rqscuten'):
            volume(field + '_p', tend[field][levels, cells].T)
    elif scheme in ('cu_tiedtke', 'cu_ntiedtke'):
        surface('xland_p', sfc['xland'][cells])
        for field in ('hfx', 'qfx'):
            surface(field + '_p', diag[field][cells])
        for field in ('rucuten', 'rvcuten'):
            volume(field + '_p', tend[field][levels, cells].T)
        if scheme == 'cu_tiedtke':
            volume('rqvdynblten_p', tend['rqvblten'][levels, cells].T)
            volume('rqvdynten_p', tend['rqvdynten'][levels, cells].T)
        else:
            values = tuple(cp.ascontiguousarray(tend[field][levels, cells].T) for field in
                           ('rqvdynten', 'rqvblten', 'rthdynten', 'rthblten', 'rthratenlw', 'rthratensw'))
            qout, tout = cp.empty((nc, nz), dtype=cp.float32), cp.empty((nc, nz), dtype=cp.float32)
            forcing = kernel_cache.raw_kernel('rrfs_convection_forcing', _CONVECTION_FORCING_CUDA,
                                             module_key='rrfs.driver.convection.forcing')
            for j in range(jts - 1, jte):
                pi = cp.ascontiguousarray(prepared['pi_p'][cells, levels, j])
                forcing(((nc * nz + 127) // 128,), (128,),
                        (np.int32(nc * nz), *values, pi, qout, tout))
                out['rqvften_p'][cells, levels, j] = qout
                out['rthften_p'][cells, levels, j] = tout
    return out


def convection_to_mpas(*, pools, prepared, options, flags, dimensions,
                       its, ite, jts, jte, kts, kte,
                       exchange_halo_group=None, subsidence_spreader=None):
    """Complete exit; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:1170-1474 at 1d6e370e.

    Increasing native tile order; rainncv is deliberately not written. Reuse
    r13's full source-identical spreading callback after native pool writeback.
    The real callback exchanges the newly written pools before staging neighbor
    reads for the reused numerical spreading routine.
    """
    import cupy as cp
    from ...cuda_backend.containers import require_resident_array
    for name, value in prepared.items():
        require_resident_array(name, value, dtype=value.dtype, shape=value.shape)
    for pool, fields in pools.items():
        for name, value in fields.items():
            require_resident_array(pool + '.' + name, value, dtype=value.dtype, shape=value.shape)
    out = {pool: {name: value.copy() for name, value in fields.items()}
           for pool, fields in pools.items()}
    cells, levels = slice(its - 1, ite), slice(kts - 1, kte)
    scheme = options['config_convection_scheme'].strip()
    for j in range(jts - 1, jte):
        def surface(field, name=None):
            out['diag_physics'][field][cells] = prepared[name or field + '_p'][cells, j]

        def volume(pool, field, name=None):
            out[pool][field][levels, cells] = prepared[name or field + '_p'][cells, levels, j].T

        surface('raincv')
        surface('cuprec', 'pratec_p')
        for field in _CONVECTION_COMMON_TEND:
            volume('tend_physics', field)
        if flags['do_chemistry']:
            start, nchem = int(dimensions['chemistry_start']) - 1, int(flags['num_chem'])
            out['state']['scalars'][start:start + nchem, levels, cells] = (
                prepared['chem_p'][cells, levels, j, :nchem].transpose(2, 1, 0))
        if scheme in ('cu_grell_freitas', 'cu_grell_freitas_li'):
            for field in _CONVECTION_GF_SURFACE:
                surface(field)
            for field, name in (('qc_cu', 'qccu_p'), ('qi_cu', 'qicu_p')):
                volume('diag_physics', field, name)
            volume('tend_physics', 'rucuten')
            if scheme == 'cu_grell_freitas':
                # Native loops overwrite only rvcuten(k,k), ending with i=ite.
                diagonal = cp.arange(kts - 1, kte, dtype=cp.int32)
                out['tend_physics']['rvcuten'][diagonal, diagonal] = prepared['rvcuten_p'][ite - 1, levels, j]
            else:
                volume('tend_physics', 'rvcuten')
                for field, name in (('vcpool', 'vcpool_p'), ('sigma_deep', 'sigma_deep_p'),
                                    ('sigma_deep_far', 'sigma_deep_f_p')):
                    surface(field, name)
                for field in ('cldfrac_cu', 'refl10cm_cu'):
                    volume('diag_physics', field)
                for field in ('rthcuten', 'rqvcuten', 'rucuten', 'rvcuten'):
                    volume('tend_physics', 'sub3d_' + field)
        elif scheme == 'cu_kain_fritsch':
            for field in ('cubot', 'cutop', 'nca'):
                surface(field)
            volume('diag_physics', 'w0avg')
            for field in ('rqrcuten', 'rqscuten'):
                volume('tend_physics', field)
        elif scheme in ('cu_tiedtke', 'cu_ntiedtke'):
            for field in ('rucuten', 'rvcuten'):
                volume('tend_physics', field)
    if scheme == 'cu_grell_freitas_li':
        if subsidence_spreader is None:
            from .cu_gfl_bf6a4b0 import _spread_after_halo
            subsidence_spreader = _spread_after_halo
        if int(options['config_gfl_sub3d']) > 0:
            if exchange_halo_group is None:
                raise ValueError('GFL neighbor reads require the actual physics:sub3d_cuten pool halo exchange')
            exchange_halo_group('physics:sub3d_cuten', pools=out)
        fields = ('rthcuten', 'rqvcuten', 'rucuten', 'rvcuten')
        nz, nsolve = int(dimensions['nVertLevels']), int(dimensions['nCellsSolve'])
        tendencies = cp.ascontiguousarray(cp.stack([out['tend_physics'][field][:nz].T for field in fields]))
        subsidence = cp.ascontiguousarray(cp.stack([out['tend_physics']['sub3d_' + field][:nz].T for field in fields]))
        mesh = dict(out['mesh'])
        for field in ('cellsOnCell', 'cellsOnCellsOnCell'):
            mesh[field] = cp.ascontiguousarray(mesh[field].T)
        subsidence_spreader(tendencies, subsidence, mesh=mesh,
                            sigma_deep=out['diag_physics']['sigma_deep'],
                            sigma_deep_far=out['diag_physics']['sigma_deep_far'],
                            config_gfl_sub3d=int(options['config_gfl_sub3d']), nCellsSolve=nsolve)
        for i, field in enumerate(fields):
            out['tend_physics'][field][:nz, :nsolve] = tendencies[i, :nsolve].T
    return out


_LSM_DIAG = (
    'acsnom', 'acsnow', 'canwat', 'chs', 'chs2', 'chklowq', 'cpm', 'cqs2',
    'glw', 'grdflx', 'gsw', 'hfx', 'lai', 'lh', 'noahres', 'potevp', 'qfx',
    'qgh', 'qsfc', 'br', 'sfc_albedo', 'sfc_emibck', 'sfc_emiss', 'sfcrunoff',
    'smstav', 'smstot', 'snotime', 'snopcx', 'udrunoff', 'z0', 'znt',
)
_LSM_SFC = ('snoalb', 'sfc_albbck', 'snow', 'snowc', 'snowh', 'tmn',
            'vegfra', 'xice', 'xland')
_LSM_RUC = ('rhosnf', 'precipfr', 'qsg', 'qvg', 'dew', 'soilt1', 'tsnav',
            'acrunoff', 'snowfallac', 'graupelncv', 'snowncv', 'rainncv',
            'qcg', 'flqc', 'flhc', 'sfcexc', 'sfcevp', 'mavail')

# MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_lsm.F:591-639 at 1d6e370e.
_LSM_CUDA = r"""
extern "C" __global__ void rrfs_lsm_surface_forcing(
    int n, int micro, int convection, const float* sr, const float* rainncv,
    const float* raincv, const float* gsw, const float* albedo,
    float* sr_out, float* rainbl, float* swdown) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;
    float rain = 0.0f;
    sr_out[q] = 0.0f;
    if (micro) { sr_out[q] = sr[q]; rain = __fadd_rn(rain, rainncv[q]); }
    if (convection) rain = __fadd_rn(rain, raincv[q]);
    rainbl[q] = rain;
    swdown[q] = __fdiv_rn(gsw[q], __fsub_rn(1.0f, albedo[q]));
}
"""


def lsm_from_mpas(*, pools, prepared, options, dimensions,
                  its, ite, jts, jte, kernel_cache):
    """Complete LSM entry; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_lsm.F:350-642 at 1d6e370e.

    Preserve the pinned qsg pointer rebound to qcg and all allocation tails.
    Species physics and initialization stay in their native or sibling hooks.
    """
    import cupy as cp
    import numpy as np
    from ...cuda_backend.containers import require_resident_array
    for name, value in prepared.items():
        require_resident_array(name, value, dtype=value.dtype, shape=value.shape)
    for pool, fields in pools.items():
        for name, value in fields.items():
            require_resident_array(pool + '.' + name, value, dtype=value.dtype, shape=value.shape)
    out = {name: value.copy() for name, value in prepared.items()}
    cells, tiles = slice(its - 1, ite), slice(jts - 1, jte)
    soils = slice(0, int(dimensions['num_soils']))
    diag, sfc = pools['diag_physics'], pools['sfc_input']
    ruc = options['config_lsm_scheme'].strip() == 'sf_ruc'

    def surface(name, value):
        out[name][cells, tiles] = value[:, None] if getattr(value, 'ndim', 0) else value

    def soil(name, value, depth=soils):
        out[name][cells, depth, tiles] = value[:, :, None] if getattr(value, 'ndim', 0) else value

    out['dzs_p'][soils] = sfc['dzs'][soils, its - 1]
    if ruc:
        out['zs_p'][soils] = sfc['zs'][soils, its - 1]
    for field in ('sh2o', 'smcrel', 'smois', 'tslb'):
        soil(field + '_p', sfc[field][soils, cells].T)
    if ruc:
        for field in ('smfr3d', 'keepfr3d'):
            soil(field + '_p', diag[field][soils, cells].T)
        for name in ('pattern_spp_lsm', 'field_sf'):
            soil(name, cp.float32(0))
        for field, extent in (('landusef', 'num_landcat'), ('soilf', 'num_soilcat')):
            depth = slice(0, int(dimensions[extent]))
            soil(field + '_p', sfc[field][depth, cells].T, depth)
    for field in (*_LSM_DIAG, 'swddif', 'swddir'):
        surface(field + '_p', diag[field][cells])
    for field in (*_LSM_SFC, 'isltyp', 'ivgtyp', 'shdmin', 'shdmax'):
        surface(field + '_p', sfc[field][cells])
    surface('tsk_p', sfc['skintemp'][cells])
    for name in ('qz0_p', 'flxsnow_p', 'fvbsnow_p', 'fbursnow_p', 'fgsnsnow_p',
                 'frc_urb_p', 'ust_urb_p'):
        surface(name, cp.float32(0))
    surface('utype_urb_p', cp.int32(0))
    if ruc:
        for field in _LSM_RUC:
            # The second getter at :450 binds qsg to qcg storage.
            surface(field + '_p', diag['qcg' if field == 'qsg' else field][cells])
        surface('lakemask_p', cp.float32(0))
    nc = ite - its + 1
    microp = options['config_microp_scheme'].strip() != 'off'
    convection = options['config_convection_scheme'].strip() != 'off'
    zero = cp.zeros(nc, dtype=cp.float32)
    arrays = [cp.ascontiguousarray(diag[field][cells]) if enabled else zero
              for field, enabled in (('sr', microp), ('rainncv', microp), ('raincv', convection))]
    gsw = cp.ascontiguousarray(diag['gsw'][cells])
    albedo = cp.ascontiguousarray(diag['sfc_albedo'][cells])
    sr, rain, swdown = (cp.empty(nc, dtype=cp.float32) for _ in range(3))
    kernel_cache.raw_kernel('rrfs_lsm_surface_forcing', _LSM_CUDA,
                           module_key='rrfs.driver.lsm.surface_forcing')(
        ((nc + 127) // 128,), (128,),
        (np.int32(nc), np.int32(microp), np.int32(convection), *arrays,
         gsw, albedo, sr, rain, swdown))
    for name, value in (('sr_p', sr), ('rainbl_p', rain), ('swdown_p', swdown)):
        surface(name, value)
    return out


def lsm_to_mpas(*, pools, prepared, options, dimensions,
                its, ite, jts, jte):
    """Complete LSM exit; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_lsm.F:645-853 at 1d6e370e.

    Write only the fields assigned by the pinned source, in native tile order.
    Soil, land-category and untouched diagnostic allocation words remain intact.
    """
    from ...cuda_backend.containers import require_resident_array
    for name, value in prepared.items():
        require_resident_array(name, value, dtype=value.dtype, shape=value.shape)
    for pool, fields in pools.items():
        for name, value in fields.items():
            require_resident_array(pool + '.' + name, value, dtype=value.dtype, shape=value.shape)
    out = {pool: {name: value.copy() for name, value in fields.items()}
           for pool, fields in pools.items()}
    cells, soils = slice(its - 1, ite), slice(0, int(dimensions['num_soils']))
    ruc = options['config_lsm_scheme'].strip() == 'sf_ruc'
    for j in range(jts - 1, jte):
        for field in ('sh2o', 'smcrel', 'smois', 'tslb'):
            out['sfc_input'][field][soils, cells] = prepared[field + '_p'][cells, soils, j].T
        for field in _LSM_DIAG:
            out['diag_physics'][field][cells] = prepared[field + '_p'][cells, j]
        for field in _LSM_SFC:
            out['sfc_input'][field][cells] = prepared[field + '_p'][cells, j]
        out['sfc_input']['skintemp'][cells] = prepared['tsk_p'][cells, j]
        if ruc:
            for field in ('smfr3d', 'keepfr3d'):
                out['diag_physics'][field][soils, cells] = prepared[field + '_p'][cells, soils, j].T
            for field in _LSM_RUC:
                out['diag_physics'][field][cells] = prepared[field + '_p'][cells, j]
        if options['config_microp_scheme'].strip() != 'off':
            out['diag_physics']['sr'][cells] = prepared['sr_p'][cells, j]
    return out


_LW_SURFACE = ('glw', 'lwcf', 'lwdnb', 'lwdnbc', 'lwdnt', 'lwdntc',
               'lwupb', 'lwupbc', 'lwupt', 'lwuptc', 'olrtoa')
_RAD_RADIUS_SCHEMES = frozenset(('mp_thompson', 'mp_thompson_aerosols',
                                'mp_wsm6', 'mp_nssl2m', 'mp_tempo'))
# MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_lw.F:561-562 at 1d6e370e.
_LW_ANGLES_CUDA = r"""
extern "C" __global__ void rrfs_lw_angles(int n, const float* radians,
                                         float degrad, float* degrees) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) degrees[q] = __fdiv_rn(radians[q], degrad);
}
"""


def radiation_lw_from_mpas(*, pools, prepared, options, dimensions,
                           its, ite, kts, kte, jts, jte, xtime_s,
                           kernel_cache, constants=None):
    """Complete entry; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_lw.F:300-653 at 1d6e370e.

    Reuse identical p22 ozone interpolation; retain native flat sequence
    association at o3clim_p(1,1,j), allocation tails and CAM restart rules.
    Allocation, initialization and radiation schemes remain existing hooks.
    """
    import cupy as cp
    import numpy as np
    from ...cuda_backend.containers import require_resident_array
    from ..rrtmg_lw_v842 import _gpu_kernel, vinterp_ozn_device
    for name, value in prepared.items():
        require_resident_array(name, value, dtype=value.dtype, shape=value.shape)
    for pool, fields in pools.items():
        for name, value in fields.items():
            require_resident_array(pool+'.'+name, value, dtype=value.dtype, shape=value.shape)
    out = {name: value.copy() for name, value in prepared.items()}
    result = {pool: {name: value.copy() for name, value in fields.items()}
              for pool, fields in pools.items()}
    cells, levels, tiles = slice(its-1, ite), slice(kts-1, kte), slice(jts-1, jte)
    diag, sfc = pools['diag_physics'], pools['sfc_input']

    def surface(name, value):
        out[name][cells, tiles] = value[:, None] if getattr(value, 'ndim', 0) else value

    def volume(name, value, depth=levels):
        out[name][cells, depth, tiles] = value[:, :, None] if getattr(value, 'ndim', 0) else value

    for name, value in (('sfc_emiss_p', diag['sfc_emiss'][cells]),
                        ('tsk_p', sfc['skintemp'][cells])):
        surface(name, value)
    for field in ('snow', 'xice', 'xland'):
        surface(field+'_p', sfc[field][cells])
    volume('cldfrac_p', diag['cldfrac'][levels, cells].T)
    for name in ('f_ice', 'f_rain', 'rthratenlw_p'):
        volume(name, cp.float32(0))
    for field in _LW_SURFACE:
        surface(field+'_p', cp.float32(0))
    scheme = options['config_radt_lw_scheme'].strip()
    if scheme == 'rrtmg_lw':
        if options['config_microp_scheme'].strip() in _RAD_RADIUS_SCHEMES:
            for field, name in (('cloud', 'cloud'), ('ice', 'ice'), ('snow', 'snow')):
                value = diag['re_'+field][levels, cells].T if options['config_microp_re'] else cp.float32(0)
                volume('re'+name+'_p', value)
                volume('rre'+name+'_p', cp.float32(0))
        for name in ('lwdnflx_p', 'lwdnflxc_p', 'lwupflx_p', 'lwupflxc_p'):
            volume(name, cp.float32(0), slice(kts-1, kte+2))
        nz, nc = int(dimensions['num_oznlevels']), ite-its+1
        ozone_levels = slice(0, nz)
        if options['config_o3climatology']:
            out['pin_p'][ozone_levels] = pools['atm_input']['pin'][ozone_levels]
            volume('o3clim_p', diag['o3clim'][ozone_levels, cells].T, ozone_levels)
            for j in range(jts-1, jte):
                pressure = cp.ascontiguousarray(out['pres_hyd_p'][cells, levels, j])
                pmid = cp.empty_like(pressure)
                _gpu_kernel('mpas_pressure_mb')(((pressure.size+127)//128,), (128,),
                    (np.int32(pressure.size), pressure, pmid))
                # The source passes an array element to an explicit-shape
                # dummy. Its native stride is ncols, not the scratch extent.
                ozone = out['o3clim_p'][:, :, j].ravel(order='F')[:nc*nz].reshape((nc, nz), order='F')
                result['diag_physics']['o3vmr'][levels, cells] = vinterp_ozn_device(
                    pmid, out['pin_p'][ozone_levels], ozone).T
        else:
            out['pin_p'][ozone_levels] = cp.float32(0)
            volume('o3clim_p', cp.float32(0), ozone_levels)
    elif scheme == 'cam_lw':
        for field, target in (('latCell', 'xlat_p'), ('lonCell', 'xlon_p')):
            radians = cp.ascontiguousarray(pools['mesh'][field][cells])
            degrees = cp.empty_like(radians)
            kernel_cache.raw_kernel('rrfs_lw_angles', _LW_ANGLES_CUDA,
                                    module_key='rrfs.driver.lw.angles')(
                ((radians.size+127)//128,), (128,),
                (np.int32(radians.size), radians, np.float32(constants['degrad']), degrees))
            surface(target, degrees)
        surface('sfc_albedo_p', diag['sfc_albedo'][cells])
        for field in ('coszr', 'gsw', 'swcf', 'swdnb', 'swdnbc', 'swdnt', 'swdntc',
                      'swupb', 'swupbc', 'swupt', 'swuptc'):
            surface(field+'_p', cp.float32(0))
        for name in ('rthratensw_p', 'cemiss_p', 'taucldc_p', 'taucldi_p'):
            volume(name, cp.float32(0))
        if xtime_s < np.float32(1.e-12):
            out['absnxt_p'][cells, levels, :int(dimensions['cam_abs_dim1']), tiles] = cp.float32(0)
            out['abstot_p'][cells, kts-1:kte+1, :int(dimensions['cam_abs_dim2']), tiles] = cp.float32(0)
            volume('emstot_p', cp.float32(0), slice(kts-1, kte+1))
        noz = int(dimensions['num_oznlevels'])
        naer, nspecies = int(dimensions['num_aerlevels']), int(dimensions['num_aerosols'])
        nmonth = int(dimensions['num_months'])
        out['pin_p'][:noz] = pools['atm_input']['pin'][:noz]
        ozone = pools['atm_input']['ozmixm'][:nmonth, :noz, cells].transpose(2, 1, 0)
        out['ozmixm_p'][cells, :noz, tiles, :nmonth] = ozone[:, :, None, :]
        out['m_hybi_p'][:naer] = diag['m_hybi'][:naer, 0]
        for name in ('m_psp_p', 'm_psn_p'):
            surface(name, diag['m_ps'][cells])
        aerosol = pools['state']['aerosols'][:nspecies, :naer, cells].transpose(2, 1, 0)
        for name in ('aerosolcp_p', 'aerosolcn_p'):
            out[name][cells, :naer, tiles, :nspecies] = aerosol[:, :, None, :]
    return out, result


def radiation_lw_to_mpas(*, pools, prepared, options, its, ite, kts, kte, jts, jte):
    """Complete exit; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_lw.F:656-787 at 1d6e370e."""
    from ...cuda_backend.containers import require_resident_array
    for name, value in prepared.items():
        require_resident_array(name, value, dtype=value.dtype, shape=value.shape)
    for pool, fields in pools.items():
        for name, value in fields.items():
            require_resident_array(pool+'.'+name, value, dtype=value.dtype, shape=value.shape)
    out = {pool: {name: value.copy() for name, value in fields.items()}
           for pool, fields in pools.items()}
    cells, levels = slice(its-1, ite), slice(kts-1, kte)
    for j in range(jts-1, jte):
        for field in _LW_SURFACE:
            out['diag_physics'][field][cells] = prepared[field+'_p'][cells, j]
        out['tend_physics']['rthratenlw'][levels, cells] = prepared['rthratenlw_p'][cells, levels, j].T
        if (options['config_radt_lw_scheme'].strip() == 'rrtmg_lw'
                and options['config_microp_scheme'].strip() in _RAD_RADIUS_SCHEMES):
            for field in ('cloud', 'ice', 'snow'):
                out['diag_physics']['rre_'+field][levels, cells] = (
                    prepared['rre'+field+'_p'][cells, levels, j].T if options['config_microp_re'] else 0)
    return out


def radiation_camlw_to_mpas(*, pools, prepared, dimensions, its, ite, kts, kte, jts, jte):
    """Complete restart exit; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_lw.F:790-834 at 1d6e370e."""
    from ...cuda_backend.containers import require_resident_array
    for name, value in prepared.items():
        require_resident_array(name, value, dtype=value.dtype, shape=value.shape)
    for pool, fields in pools.items():
        for name, value in fields.items():
            require_resident_array(pool+'.'+name, value, dtype=value.dtype, shape=value.shape)
    out = {pool: {name: value.copy() for name, value in fields.items()}
           for pool, fields in pools.items()}
    cells, levels = slice(its-1, ite), slice(kts-1, kte)
    for j in range(jts-1, jte):
        out['diag_physics']['absnxt'][levels, :int(dimensions['cam_abs_dim1']), cells] = (
            prepared['absnxt_p'][cells, levels, :int(dimensions['cam_abs_dim1']), j].transpose(1, 2, 0))
        out['diag_physics']['abstot'][kts-1:kte+1, :int(dimensions['cam_abs_dim2']), cells] = (
            prepared['abstot_p'][cells, kts-1:kte+1, :int(dimensions['cam_abs_dim2']), j].transpose(1, 2, 0))
        out['diag_physics']['emstot'][kts-1:kte+1, cells] = prepared['emstot_p'][cells, kts-1:kte+1, j].T
    return out


_SW_SURFACE = ('coszr', 'gsw', 'swcf', 'swdnb', 'swdnbc', 'swdnt', 'swdntc',
               'swupb', 'swupbc', 'swupt', 'swuptc', 'swddir', 'swddni', 'swddif')


def shortwave_aerosol_source():
    """Reuse whole identical aerosol routines, without loading radiation tables.

    MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_mp_thompson_aerosols.F:23-210;
    MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_ra_rrtmg_sw_aerosols.F:70-921;
    MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_functions.F:156-186 at 1d6e370e.
    Source-identical p23 kernels and the shared strict device maths stay shared.
    """
    from ..rrtmg_sw_v842 import CUDA_SOURCE
    from ..kessler_v842 import shared_math_header
    tables = CUDA_SOURCE[CUDA_SOURCE.index('__device__ const float MPAS_RAOD'):
                         CUDA_SOURCE.index('// MPAS module_ra_rrtmg_sw.F:10026-10664, wrapper and outputs.')]
    kernels = CUDA_SOURCE[CUDA_SOURCE.index('// MPAS module_ra_rrtmg_sw_aerosols.F:890-921'):
                          CUDA_SOURCE.index('// MPAS module_ra_rrtmg_sw.F:3113-3167,6192-8015 constant tables.')]
    return shared_math_header() + tables + kernels


def radiation_sw_from_mpas(*, pools, prepared, options, dimensions,
                           its, ite, kts, kte, jts, jte, xtime_s,
                           kernel_cache, constants, rrtmg_only=False):
    """Complete entry; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_sw.F:370-808 at 1d6e370e.

    Retain native radii, TEMPO/Thompson aerosol preparation, untouched allocation
    tails and CAM persistent arrays. Radiation schemes remain existing hooks.
    """
    import cupy as cp
    import numpy as np
    from ...cuda_backend.containers import require_resident_array
    for name, value in prepared.items():
        require_resident_array(name, value, dtype=value.dtype, shape=value.shape)
    for pool, fields in pools.items():
        for name, value in fields.items():
            require_resident_array(pool+'.'+name, value, dtype=value.dtype, shape=value.shape)
    out = {name: value.copy() for name, value in prepared.items()}
    result = {pool: {name: value.copy() for name, value in fields.items()}
              for pool, fields in pools.items()}
    cells, levels, tiles = slice(its-1, ite), slice(kts-1, kte), slice(jts-1, jte)
    diag, sfc = pools['diag_physics'], pools['sfc_input']

    def surface(name, value):
        out[name][cells, tiles] = value[:, None] if getattr(value, 'ndim', 0) else value

    def volume(name, value, depth=levels):
        out[name][cells, depth, tiles] = value[:, :, None] if getattr(value, 'ndim', 0) else value

    for field, target in (('latCell', 'xlat_p'), ('lonCell', 'xlon_p')):
        radians = cp.ascontiguousarray(pools['mesh'][field][cells])
        degrees = cp.empty_like(radians)
        kernel_cache.raw_kernel('rrfs_lw_angles', _LW_ANGLES_CUDA,
                                module_key='rrfs.driver.sw.angles')(
            ((radians.size+127)//128,), (128,),
            (np.int32(radians.size), radians, np.float32(constants['degrad']), degrees))
        surface(target, degrees)
    surface('sfc_albedo_p', diag['sfc_albedo'][cells])
    surface('tsk_p', sfc['skintemp'][cells])
    for field in ('snow', 'xice', 'xland'):
        surface(field+'_p', sfc[field][cells])
    volume('cldfrac_p', diag['cldfrac'][levels, cells].T)
    for name in ('f_ice', 'f_rain', 'rthratensw_p'):
        volume(name, cp.float32(0))
    scheme = options['config_radt_sw_scheme'].strip()
    # 44c0849 driver_sw.F:476-510 initializes these only for RRTMG.
    for field in _SW_SURFACE:
        if not rrtmg_only or scheme == 'rrtmg_sw' or field not in ('swddir', 'swddni', 'swddif'):
            surface(field+'_p', cp.float32(0))
    aer_opt = 0
    if not rrtmg_only or scheme == 'rrtmg_sw':
        nbands = out['tauaer_p'].shape[3]
        for name, value in (('tauaer_p', 0), ('ssaaer_p', 1), ('asyaer_p', 0)):
            out[name][cells, levels, tiles, :nbands] = cp.float32(value)
    micro = options['config_microp_scheme'].strip()
    if scheme == 'rrtmg_sw':
        if micro in _RAD_RADIUS_SCHEMES:
            for field in ('cloud', 'ice', 'snow'):
                value = diag['re_'+field][levels, cells].T if options['config_microp_re'] else cp.float32(0)
                volume('re'+field+'_p', value)
        if micro == 'mp_thompson_aerosols' or (micro == 'mp_tempo' and options['config_tempo_aerosolaware']):
            aer_opt = 3
            surface('ht_p', pools['mesh']['zgrid'][0, cells])
            nc, nz = ite-its+1, kte-kts+1
            gt = kernel_cache.raw_kernel('mpas_sw_gt_aod', shortwave_aerosol_source(),
                                         module_key='rrfs.driver.sw.aerosols')
            optical = kernel_cache.raw_kernel('mpas_sw_aerosol', shortwave_aerosol_source(),
                                              module_key='rrfs.driver.sw.aerosols')
            for j in range(jts-1, jte):
                inputs = [cp.ascontiguousarray(out[name][cells, levels, j])
                          for name in ('pres_hyd_p', 't_p', 'qv_p', 'dz_p', 'nwfa_p', 'nifa_p')]
                xland = cp.ascontiguousarray(out['xland_p'][cells, j])
                types = cp.ascontiguousarray(out['taer_type_p'][cells, j])
                aod3, aod2 = cp.empty((nc, nz), cp.float32), cp.empty(nc, cp.float32)
                gt(((nc+127)//128,), (128,),
                   (np.int32(nc), np.int32(nz), *inputs, xland, aod3, aod2, types))
                out['taer_type_p'][cells, j] = types
                out['taod5503d_p'][cells, levels, j] = aod3
                out['taod5502d_p'][cells, j] = aod2
                result['diag_physics']['taod5503d'][levels, cells] = aod3.T
                result['diag_physics']['taod5502d'][cells] = aod2
                tau, ssa, asymmetry = (cp.empty((nc, nbands, nz), cp.float32) for _ in range(3))
                unused = [cp.empty(nc, cp.float32) for _ in range(3)]
                height = cp.ascontiguousarray(out['ht_p'][cells, j])
                # Native vars.F:840-854 parameters; optional diagnostics absent.
                optical(((nc+127)//128,), (128,),
                    (np.int32(nc), np.int32(nz), np.int32(nz), np.int32(1),
                     np.int32(2), np.int32(3), np.int32(3), np.int32(3),
                     np.float32(.12), np.float32(1.3), np.float32(.85), np.float32(.9), np.int32(0),
                     *inputs[:4], height, types, aod2, aod3, *unused, tau, ssa, asymmetry))
                for name, value in (('tauaer_p', tau), ('ssaaer_p', ssa), ('asyaer_p', asymmetry)):
                    out[name][cells, levels, j, :nbands] = value.transpose(0, 2, 1)
        for name in ('swdnflx_p', 'swdnflxc_p', 'swupflx_p', 'swupflxc_p'):
            volume(name, cp.float32(0), slice(kts-1, kte+2))
        noz = int(dimensions['num_oznlevels'])
        if options['config_o3climatology']:
            out['pin_p'][:noz] = pools['atm_input']['pin'][:noz]
            volume('o3clim_p', diag['o3clim'][:noz, cells].T, slice(0, noz))
        else:
            out['pin_p'][:noz] = cp.float32(0)
            volume('o3clim_p', cp.float32(0), slice(0, noz))
    elif scheme == 'cam_sw':
        surface('sfc_emiss_p', diag['sfc_emiss'][cells])
        for field in _LW_SURFACE:
            surface(field+'_p', cp.float32(0))
        for name in ('rthratenlw_p', 'cemiss_p', 'taucldc_p', 'taucldi_p'):
            volume(name, cp.float32(0))
        if xtime_s < np.float32(1.e-12):
            out['absnxt_p'][cells, levels, :int(dimensions['cam_abs_dim1']), tiles] = cp.float32(0)
            out['abstot_p'][cells, kts-1:kte+1, :int(dimensions['cam_abs_dim2']), tiles] = cp.float32(0)
            volume('emstot_p', cp.float32(0), slice(kts-1, kte+1))
        noz, naer = int(dimensions['num_oznlevels']), int(dimensions['num_aerlevels'])
        nspecies, nmonth = int(dimensions['num_aerosols']), int(dimensions['num_months'])
        out['pin_p'][:noz] = pools['atm_input']['pin'][:noz]
        ozone = pools['atm_input']['ozmixm'][:nmonth, :noz, cells].transpose(2, 1, 0)
        out['ozmixm_p'][cells, :noz, tiles, :nmonth] = ozone[:, :, None, :]
        out['m_hybi_p'][:naer] = diag['m_hybi'][:naer, 0]
        for name in ('m_psp_p', 'm_psn_p'):
            surface(name, diag['m_ps'][cells])
        aerosol = pools['state']['aerosols'][:nspecies, :naer, cells].transpose(2, 1, 0)
        for name in ('aerosolcp_p', 'aerosolcn_p'):
            out[name][cells, :naer, tiles, :nspecies] = aerosol[:, :, None, :]
    return out, result, aer_opt


def radiation_sw_to_mpas(*, pools, prepared, its, ite, kts, kte, jts, jte, options=None, rrtmg_only=False):
    """Complete exit; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_sw.F:811-879 at 1d6e370e."""
    from ...cuda_backend.containers import require_resident_array
    for name, value in prepared.items():
        require_resident_array(name, value, dtype=value.dtype, shape=value.shape)
    for pool, fields in pools.items():
        for name, value in fields.items():
            require_resident_array(pool+'.'+name, value, dtype=value.dtype, shape=value.shape)
    out = {pool: {name: value.copy() for name, value in fields.items()}
           for pool, fields in pools.items()}
    cells, levels = slice(its-1, ite), slice(kts-1, kte)
    for j in range(jts-1, jte):
        for field in _SW_SURFACE:
            if (rrtmg_only and options['config_radt_sw_scheme'].strip() != 'rrtmg_sw'
                    and field in ('swddir', 'swddni', 'swddif')):
                continue
            out['diag_physics'][field][cells] = prepared[field+'_p'][cells, j]
        out['tend_physics']['rthratensw'][levels, cells] = prepared['rthratensw_p'][cells, levels, j].T
    return out


# MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_init.F:549-605.
_INIT_DIRECTIONS_CUDA = r"""
extern "C" __global__ void rrfs_init_directions(
    const int cells, const int extent, const float *lat, const float *lon,
    float *east, float *north) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=cells) return;
    float slon=sinf(lon[c]),clon=cosf(lon[c]);
    float slat=sinf(lat[c]),clat=cosf(lat[c]);
    float ax=-slon,ay=clon,az=0.f;
    float mi=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(
        __fadd_rn(__fmul_rn(ax,ax),__fmul_rn(ay,ay)),__fmul_rn(az,az))));
    east[c]=__fmul_rn(ax,mi);
    east[extent+c]=__fmul_rn(ay,mi);
    east[2*extent+c]=__fmul_rn(az,mi);
    ax=__fmul_rn(-clon,slat);ay=__fmul_rn(-slon,slat);az=clat;
    mi=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(
        __fadd_rn(__fmul_rn(ax,ax),__fmul_rn(ay,ay)),__fmul_rn(az,az))));
    north[c]=__fmul_rn(ax,mi);
    north[extent+c]=__fmul_rn(ay,mi);
    north[2*extent+c]=__fmul_rn(az,mi);
}
// RRFS 44c0849 mpas_atmphys_init.F:284-292. Sequential vertical sum,
// matching native binary32 addition order; one shared model-column scalar.
extern "C" __global__ void rrfs_init_microphysics_top(
    int levels, float top, const float *rdzw, const float *dzu, int *level) {
    if(blockIdx.x || threadIdx.x) return;
    int result=0;
    float height=__fdiv_rn(0.5f,rdzw[0]);
    if(height<=top) result=1;
    for(int k=1;k<levels;++k){
        height=__fadd_rn(height,dzu[k]);
        if(height<=top) result=k+1;
    }
    *level=result;
}
extern "C" __global__ void rrfs_init_ocean_temperature(
    const int cells,const float *sst,float *deep) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<cells) deep[c]=__fsub_rn(sst[c],5.f);
}
"""


_INIT_FLUXES = ("acswdnb", "acswdnbc", "acswdnt", "acswdntc",
                "acswupb", "acswupbc", "acswupt", "acswuptc",
                "aclwdnb", "aclwdnbc", "aclwdnt", "aclwdntc",
                "aclwupb", "aclwupbc", "aclwupt", "aclwuptc")
_INIT_SPECIES = ("qc", "qr", "qi", "qs", "qg", "smoke_fine", "dust_fine",
                 "dust_coarse", "unspc_fine", "nc", "ni", "nifa", "nwfa")


def initialize_physics_pools(*, pools, options, dimensions, scalar_indices, kernel_cache):
    """Complete driver-owned numerical preparation before delegated initializers.

    MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_init.F:84-396,480-605.
    The native hooks still own ozone, landuse, scheme tables and their private
    state at :397-470. This phase preserves every inactive/halo/dummy word and
    returns the actual scalar flags alongside its resident pool allocations.
    """
    import numpy as np
    from ..math_v842 import bind_common_carrier_math
    result = {name: {field: value.copy() for field, value in fields.items()}
              for name, fields in pools.items()}
    mesh, diag, sfc = result['mesh'], result['diag_physics'], result['sfc_input']
    cells, solved, lags = (int(dimensions[name]) for name in ('nCells','nCellsSolve','nLags'))
    directions = kernel_cache.raw_kernel('rrfs_init_directions',
        bind_common_carrier_math(_INIT_DIRECTIONS_CUDA), module_key='rrfs-physics-initialization')
    if cells:
        directions(((cells+127)//128,), (128,), (np.int32(cells),
            np.int32(mesh['east'].shape[1]), mesh['latCell'], mesh['lonCell'],
            mesh['east'], mesh['north']))
    if 'config_microphysics_top' in options:
        kernel_cache.raw_kernel('rrfs_init_microphysics_top',
            bind_common_carrier_math(_INIT_DIRECTIONS_CUDA), module_key='rrfs-physics-initialization')(
            (1,), (1,), (np.int32(dimensions['nVertLevels']),
                        np.float32(options['config_microphysics_top']), mesh['rdzw'], mesh['dzu'],
                        result['tend_physics']['mp_top_level']))
    flags = {'f_'+name: int(int(scalar_indices['index_'+name]) > -1) for name in _INIT_SPECIES}
    flags.update(f_qoz=0, f_nbca=0)
    restart = bool(options['config_do_restart'])
    if not restart:
        if options['config_convection_scheme'].strip() != 'off':
            diag['i_rainc'][:solved] = 0
        if options['config_microp_scheme'].strip() != 'off':
            diag['i_rainnc'][:solved] = 0
        for name in _INIT_FLUXES:
            diag['i_'+name][:solved] = 0
            diag[name][:solved] = 0
        if not bool(options['config_do_DAcycling']):
            diag['xicem'][:solved] = sfc['xice'][:solved]
            diag['sstsk'][:solved] = sfc['sst'][:solved]
        if bool(options['config_oml1d']):
            diag['t_oml'][:solved] = sfc['sst'][:solved]
            diag['t_oml_initial'][:solved] = sfc['sst'][:solved]
            hml0 = np.float32(options['config_oml_hml0'])
            if hml0 > 0:
                diag['h_oml'][:solved] = hml0
                diag['h_oml_initial'][:solved] = hml0
            else:
                diag['h_oml'][:solved] = diag['h_oml_initial'][:solved]
            if hml0 >= 0:
                diag['hu_oml'][:solved] = 0
                diag['hv_oml'][:solved] = 0
                if solved:
                    ocean = kernel_cache.raw_kernel('rrfs_init_ocean_temperature',
                        bind_common_carrier_math(_INIT_DIRECTIONS_CUDA), module_key='rrfs-physics-initialization')
                    ocean(((solved+127)//128,), (128,), (np.int32(solved),
                        sfc['sst'], diag['t_oml_200m_initial']))
            # Native negative-depth missing-value predicate is contradictory.
            # Preserve its actual no-write behavior, including NaN words.
        for name in ('nsteps_accum', 'ndays_accum', 'tday_accum', 'tyear_accum'):
            diag[name][:solved] = 0
        diag['tyear_mean'][:solved] = sfc['tmn'][:solved]
        diag['tlag'][:lags, :solved] = sfc['tmn'][None, :solved]
    return result, flags


class ResidentDriverPreparation:
    """Persistent destinations for interface.F:341-652, with unchanged arithmetic.

    The old resident adapter copied 38 results and issued 17 kernels per step.
    Bind the mass/interface kernels straight to their resident destinations;
    one word-copy plan publishes the two winds and duplicate surface pressure.
    Number-carrier selection is static, and each selected max(+0,q) still uses
    CuPy maximum, including its signed-zero and NaN behavior.
    """

    def __init__(self, glue, call):
        import numpy as np
        from .glue_plan import CopyPlan
        from ...cuda_physics_prep_v841 import _device_pointer_table, RV_OVER_RD_F32, GRAVITY_F32
        self.glue = glue
        p, cp, nz, nc = glue.provider, glue.cp, glue.nlev, glue.ncells
        p.level = 1
        fields = {name: p.field('state', name) for name in ('rho_zz', 'theta_m', 'w')}
        fields.update({name: p.field('mesh', name) for name in ('zgrid', 'zz')})
        fields.update({name: p.field('diag', name) for name in
                       ('exner', 'pressure_base', 'pressure_p', 'uReconstructZonal', 'uReconstructMeridional')})
        scalars = p.field('state', 'scalars')
        names = ('qv', 'qc', 'qr', 'qi', 'qs', 'qg')
        self.pack = None
        if tuple(glue.names[:6]) == names:
            mass_scalars = scalars[:6]
        else:
            mass_scalars = cp.empty((6, nz, nc), dtype=cp.float32)
            self.pack = CopyPlan('rrfs.prepare.scalar_order')
            for i, name in enumerate(names):
                self.pack.copy(mass_scalars[i], scalars[glue.names.index(name)])
            self.pack.build()
        mass = ('zz_p', 'rho_dry', 'rho_p', 'th_p', 't_p', 'pi_p', 'pres_p', 'zmid_p', 'dz_p',
                'pres_hyd_p', 'pres_hydd_p', 'znu_p', 'znu_hyd_p')
        interfaces = ('w_p', 'z_p', 't2_p', 'pres2_p', 'pres2_hyd_p', 'pres2_hydd_p', 'fzm_p', 'fzp_p')
        surfaces = ('psfc_p', 'psfc_hyd_p', 'psfc_hydd_p')
        out = {name: p.prepared(name)[:nz] for name in mass if name != 'rho_dry'}
        out['rho_dry'] = cp.empty((nz, nc), dtype=cp.float32)
        out.update({name: p.prepared(name)[:nz+1] for name in interfaces})
        out.update({name: p.prepared(name) for name in surfaces})
        out['plrad'] = p.field('diag_physics', 'plrad')
        out.update({name+'_p': p.prepared(name+'_p')[:nz] for name in names})
        pointers = _device_pointer_table(cp, [out[name+'_p'] for name in names])
        self.invalid = cp.zeros(1, dtype=cp.int32)
        plan = self.plan = CopyPlan('rrfs.MPAS_to_physics')
        plan.copy(p.prepared('u_p')[:nz], fields['uReconstructZonal'])
        plan.copy(p.prepared('v_p')[:nz], fields['uReconstructMeridional'])
        plan.copy(p.field('diag', 'surface_pressure'), out['psfc_p'])
        # prep_mass/prep_interfaces only set invalid. Reset after their call,
        # as the old adapter discarded its fresh flag without consuming it.
        plan.fill(self.invalid, 0)
        options = coupling_selectors(call.options)
        aerosol = options['config_microp_scheme'] == 'mp_thompson_aerosols'
        mynn = options['config_pbl_scheme'] == 'bl_mynn'
        flags = {name: bool(p['scalar_state']['f_'+name]) for name in ('nc', 'ni', 'nifa', 'nwfa')}
        allocated = ('nc', 'ni', 'nifa', 'nwfa') if mynn else ('nifa', 'nwfa') if aerosol else ()
        selected = set()
        if mynn:
            if flags['ni']:
                selected.add('ni')
            if all(flags[n] for n in ('nc', 'nifa', 'nwfa')):
                selected.update(('nc', 'nifa', 'nwfa'))
        elif aerosol:
            selected.update(('nifa', 'nwfa'))
        self.clamps = []
        for name in allocated:
            target = p.prepared(name+'_p')[:nz]
            if name in selected:
                self.clamps.append((scalars[glue.names.index(name)], target))
            else:
                plan.fill(target, 0)
        self.mass_args = (np.int32(nz), np.int32(nc), np.int32(6), RV_OVER_RD_F32,
            fields['rho_zz'], fields['theta_m'], mass_scalars, fields['exner'], fields['pressure_base'],
            fields['pressure_p'], fields['zgrid'], fields['zz'], pointers,
            *(out[name] for name in mass[:9]), self.invalid)
        self.interface_args = (np.int32(nz), np.int32(nc), GRAVITY_F32,
            fields['rho_zz'], fields['zz'], out['qv_p'], out['pres_p'], out['t_p'], out['rho_p'],
            out['dz_p'], fields['zgrid'], fields['w'], out['w_p'], out['z_p'], out['t2_p'],
            out['pres2_p'], out['pres_hyd_p'], out['pres2_hyd_p'], out['pres_hydd_p'],
            out['pres2_hydd_p'], out['psfc_p'], out['psfc_hyd_p'], out['psfc_hydd_p'],
            out['znu_p'], out['znu_hyd_p'], out['plrad'], self.invalid, fields['pressure_p'],
            fields['pressure_base'], out['fzm_p'], out['fzp_p'])

        self.kernels = [glue.cache.raw_kernel(name, interface_cuda_source(), module_key='hexcore.mpasphys.rrfs.interface')
                        for name in ('prep_mass_v841_f32', 'prep_interfaces_v841_f32')]
        self.grid = ((nc+127)//128,)
        plan.build()
        p.commit()

    def run(self):
        if self.pack is not None:
            self.pack()
        for kernel, args in zip(self.kernels, (self.mass_args, self.interface_args)):
            kernel(self.grid, (128,), args)
        for source, target in self.clamps:
            self.glue.cp.maximum(self.glue.cp.float32(0), source, out=target)
        self.plan()


def resident_prepare(glue, call):
    prepared = getattr(glue, '_driver_preparation', None)
    if prepared is None:
        if not getattr(glue, '_driver_preparation_hooked', False):
            # The preparation binds prepared arrays into kernel arguments, a pointer table and a word-copy plan once;
            # after release_prepared it is rebuilt on the new arrays (NativeDevicePoolViews.on_prepared_release says
            # why: bound once, it kept writing the released arrays and RUC read zeros, fatal at step 2).
            register = getattr(glue.provider, 'on_prepared_release', None)
            if register is not None:
                register(lambda: setattr(glue, '_driver_preparation', None))
            glue._driver_preparation_hooked = True
        prepared = glue._driver_preparation = ResidentDriverPreparation(glue, call)
    glue.provider.level = 1
    prepared.run()


class ResidentDriverTendencies:
    """Persistent coupling tables and one fill/pack plan, todynamics.F:98-552.

    Keep the existing projection and RK kernels, including their operation
    order and solve masks. Only storage allocation and word moves change.
    """

    def __init__(self, glue):
        import numpy as np
        from .glue_plan import CopyPlan
        from ...cuda_physics_v841 import CudaMpasStageSourcesV842
        self.glue = glue
        p, cp, a = glue.provider, glue.cp, glue.assembler
        self.plan = plan = CopyPlan('rrfs.physics_get_tend')
        p.level = 2
        mass = p.field('state', 'rho_zz')
        p.level = 1
        mass_edge = p.field('diag', 'rho_edge')
        theta = p.field('state', 'theta_m')
        scalars = p.field('state', 'scalars')
        rates = {row['pool_name']: p.field('tend_physics', row['pool_name'])
                 for row in glue.session.layout.fields if row['pool'] == 'tend_physics'
                 and (row['pool_name'].endswith('ten') or row['pool_name'] in ('rthratenlw', 'rthratensw'))}
        rates.update(glue.local_rates)
        cfg, names = a.selectors, a.scalar_names
        pbl = cfg['config_pbl_scheme'] != 'off'
        cu_scheme = cfg['config_convection_scheme']
        cu = cu_scheme != 'off'
        pbl_species = ['qv', 'qc', 'qi'] if pbl else []
        if pbl and cfg['config_pbl_scheme'] == 'bl_mynn':
            pbl_species += ['qs', 'ni']
            if cfg['config_microp_scheme'] == 'mp_thompson_aerosols':
                pbl_species += ['nc', 'nifa', 'nwfa']
        cu_species = ['qv', 'qc', 'qi'] if cu else []
        if cu_scheme == 'cu_kain_fritsch':
            cu_species += ['qr', 'qs']
        needed = set(pbl_species + cu_species + ['qv'])
        if not needed.issubset(names):
            raise ValueError(f'MPAS coupling missing species {sorted(needed-set(names))}')
        pbl_order = ('qv', 'qc', 'qi', 'qs', 'ni', 'nc', 'nifa', 'nwfa')
        cu_order = ('qv', 'qc', 'qi', 'qr', 'qs')
        def indices(order, used):
            return cp.asarray(np.array([names.index(n) if n in used or n == 'qv' else -1
                                        for n in order], dtype=np.int32))
        pbl_index, cu_index = indices(pbl_order, pbl_species), indices(cu_order, cu_species)
        rate_names = ('rthblten', 'rqvblten', 'rqcblten', 'rqiblten', 'rqsblten',
                      'rniblten', 'rncblten', 'rnifablten', 'rnwfablten',
                      'rthcuten', 'rqvcuten', 'rqccuten', 'rqicuten', 'rqrcuten',
                      'rqscuten', 'rthratenlw', 'rthratensw')
        required = set()
        if pbl:
            required.add('rthblten')
            required.update(rate_names[1+pbl_order.index(n)] for n in pbl_species)
        if cu:
            required.add('rthcuten')
            required.update(rate_names[10+cu_order.index(n)] for n in cu_species)
        lw, sw = cfg['config_radt_lw_scheme'] != 'off', cfg['config_radt_sw_scheme'] != 'off'
        if lw:
            required.add('rthratenlw')
        if sw:
            required.add('rthratensw')
        missing = required-set(rates)
        if missing:
            raise ValueError(f'MPAS coupling missing tendencies {sorted(missing)}')
        zero = cp.zeros_like(mass)
        table_arrays = [rates[name] if name in required else zero for name in rate_names]
        from ...cuda_backend.containers import require_resident_array
        for name, array in zip(rate_names, table_arrays):
            require_resident_array(name, array, dtype=np.float32, shape=mass.shape)
        self.table_arrays = table_arrays
        table = cp.asarray(np.array([array.data.ptr for array in table_arrays], dtype=np.uint64))
        self.result = CudaMpasStageSourcesV842(cp.empty_like(mass), cp.empty_like(mass_edge),
                                             cp.empty_like(mass), p.field('tend', 'scalars_tend'))
        out = self.result
        for value in (out.rho, out.rho_u, out.rho_theta, out.scalars):
            plan.fill(value, 0)
        self.projections = []
        self.groups = []
        g = a.geometry
        for active, group, zonal, meridional, target in (
                (pbl, 'physics:blten', 'rublten', 'rvblten', a.pbl_edge),
                (cu_scheme in ('cu_tiedtke', 'cu_ntiedtke'),
                 'physics:cuten', 'rucuten', 'rvcuten', a.convection_edge)):
            if not active:
                continue
            self.groups.append(group)
            inputs = [rates[zonal], rates[meridional]]
            if a.regional_garbage:
                padded = []
                for value in inputs:
                    buffer = cp.zeros((value.shape[0], value.shape[1]+1), dtype=value.dtype)
                    plan.copy(buffer[:, :value.shape[1]], value)
                    padded.append(buffer)
                inputs = padded
            for name, value in zip((zonal, meridional), inputs):
                require_resident_array(name, value, dtype=np.float32, shape=(mass.shape[0], g.n_cells))
            require_resident_array(group+' edge cache', target, dtype=np.float32,
                                   shape=(mass.shape[0], g.n_edges))
            unit = cp.ones_like(target)
            self.projections.append((np.int32(mass.shape[0]), np.int32(g.n_cells), np.int32(g.n_edges),
                g.cells_on_edge, g.east_cell, g.north_cell, g.edge_normal_vectors, *inputs,
                unit, target, glue.invalid))
        nz, nc = mass.shape
        ne = mass_edge.shape[1]
        from ...cuda_physics_prep_v841 import RV_OVER_RD_F32
        self.cell_args = (np.int32(nz), np.int32(nc), np.int32(glue.ncells), np.int32(len(names)),
            np.int32(pbl), np.int32(cu), np.int32(lw), np.int32(sw), pbl_index, cu_index,
            RV_OVER_RD_F32, mass, theta, scalars[names.index('qv')], table,
            out.rho_theta, out.scalars, out.rho)
        self.edge_args = (np.int32(nz), np.int32(ne), np.int32(glue.dimensions['nEdgesSolve']),
            np.int32(pbl), np.int32(cu_scheme in ('cu_tiedtke', 'cu_ntiedtke')), np.int32(1),
            mass_edge, a.pbl_edge, a.convection_edge, out.rho_u, a.tend_u_phys)
        self.counts = nz*nc, nz*ne
        plan.build()
        p.commit()

    def run(self, call):
        from ...cuda_physics_v841 import _launch, _mpas_v842_launch
        if (call.pools['rk_step'], call.pools['dynamics_substep']) != (1, 1):
            raise RuntimeError('resident RRFS sources must be assembled before the first RK stage')
        g = self.glue
        for group in self.groups:
            g.assembler.exchange_halo_group(group)
        self.plan()
        for args in self.projections:
            _launch(g.cache, 'project_edges_v841_f32', g.assembler.geometry.n_edges, args)
        _mpas_v842_launch(g.cache, 'mpas_rk_cells_v842_f32', self.counts[0], self.cell_args)
        _mpas_v842_launch(g.cache, 'mpas_rk_edges_v842_f32', self.counts[1], self.edge_args)
        return self.result


def resident_get_tend(glue, call):
    tendencies = getattr(glue, '_driver_tendencies', None)
    if tendencies is None:
        tendencies = glue._driver_tendencies = ResidentDriverTendencies(glue)
    glue.provider.level = 1
    return tendencies.run(call)


def resident_transport_copy_source():
    """Fixed argument roster: rotating dycore pointers never need a device table."""
    arguments, cases = [], []
    for op in range(12):
        arguments += [f'unsigned int* dst_arg{op}', f'const unsigned int* src_arg{op}']
        arguments += [f'long long {name}{op}' for name in
                      ('count', 'n1_', 'n2_', 'n3_', 'd0_', 'd1_', 'd2_', 'd3_',
                       's0_', 's1_', 's2_', 's3_')]
        cases.append(f'case {op}: dst=dst_arg{op}; src=src_arg{op}; count=count{op}; ' +
            ' '.join(f'{name}={name}_{op};' for name in
                     ('n1', 'n2', 'n3', 'd0', 'd1', 'd2', 'd3', 's0', 's1', 's2', 's3')) + ' break;')
    return r'''extern "C" __global__ void rrfs_post_transport_copy(''' + ', '.join(arguments) + r''') {
        unsigned int* dst; const unsigned int* src;
        long long count, n1, n2, n3, d0, d1, d2, d3, s0, s1, s2, s3;
        switch (blockIdx.y) {''' + '\n'.join(cases) + r''' default: return; }
        for (long long e=(long long)blockIdx.x*blockDim.x+threadIdx.x; e<count;
             e+=(long long)gridDim.x*blockDim.x) {
            long long r=e;
            const long long i3=r%n3; r/=n3;
            const long long i2=r%n2; r/=n2;
            const long long i1=r%n1; r/=n1;
            dst[r*d0+i1*d1+i2*d2+i3*d3] = src[r*s0+i1*s1+i2*s2+i3*s3];
        }
    }'''


def resident_transport_copy_plan(coupling):
    """CPU-only overlap proof for the current argument rows; never build/upload it."""
    from .glue_plan import CopyPlan
    plan = CopyPlan('rrfs.post_transport_prepare.arguments')
    for dst, src in coupling._rrfs_transport_copies:
        plan.copy(dst, src)
    return plan


def resident_prepare_transport(coupling, atmosphere):
    """Publish twelve carriers in one argument-bound launch, interface.F:284-305.

    Called instead of synchronize(candidate, level=2). Reconstruction and
    output thermodynamics retain their original kernels. Pointers rotate as
    kernel arguments, with no CopyPlan table upload. Same-host wind fallback
    keeps its existing complete synchronization path.
    """
    import numpy as np
    from ..diagnostics_v842 import native_diagnostic_source
    from ...cuda_physics_prep_v841 import RV_OVER_RD_F32
    chemistry = getattr(coupling, 'chemistry', None)
    if chemistry is not None:
        atmosphere = chemistry.registry_view(atmosphere)
    columns = coupling._resident_winds(atmosphere)
    if columns is None:
        return coupling.synchronize(atmosphere, level=2)
    state, saved, session = atmosphere.state, atmosphere.saved, coupling.session
    p = session.device_pools
    assignments = [('state', name, value) for name, value in (
        ('rho_zz', state.rho), ('theta_m', saved.theta_m), ('u', saved.normal_velocity),
        ('w', saved.vertical_velocity), ('scalars', state.scalars))]
    assignments += [('diag', name, value) for name, value in (
        ('rho_p', saved.density_perturbation), ('rtheta_p', saved.rho_theta_perturbation),
        ('pressure_p', saved.pressure_perturbation), ('exner', saved.exner))]
    held = getattr(coupling, '_rrfs_transport_diagnostics', None)
    shape = state.rho.shape
    if held is None or held[0] != shape:
        outputs = tuple(coupling.cp.empty_like(state.rho) for _ in range(3))
        kernel = coupling.physics_cache.raw_kernel('mpas_output_diagnostics_v842', native_diagnostic_source(),
                                                   module_key='hexcore.mpasphys.output_diagnostics_v842')
        held = coupling._rrfs_transport_diagnostics = (shape, outputs, kernel)
    _, outputs, kernel = held
    count = state.rho.size
    kernel(((count+127)//128,), (128,), (
        np.int32(count), RV_OVER_RD_F32, saved.theta_m, state.scalars[tuple(coupling.names).index('qv')],
        state.rho, atmosphere.vertical.zz, atmosphere.reference.pressure_base,
        saved.pressure_perturbation, *outputs))
    assignments += [('diag', name, value) for name, value in zip(('theta', 'rho', 'pressure'), outputs)]
    # write_field's owned-extent contract, including a regional dummy column
    # or a same-host rank's halo columns. No device scalar is read here.
    copies = []
    old, p.level = p.level, 2
    try:
        for pool, name, source in assignments:
            target = p.field(pool, name)
            if source.shape != target.shape:
                if (source.shape[:-1] != target.shape[:-1]
                        or not (source.shape[-1] == target.shape[-1]+1
                                or (getattr(session.layout, 'samehost_partition', False)
                                    and source.shape[-1] > target.shape[-1]))):
                    raise ValueError(f'resident {pool}.{name} extent differs: {source.shape} != {target.shape}')
                source = source[..., :target.shape[-1]]
            copies.append((target, source))
    finally:
        p.level = old
    # Layout metadata persists independently of source pointer rotation.
    key = tuple((dst.shape, dst.strides, src.strides, dst.dtype, src.dtype)
                for dst, src in copies)
    held_copy = getattr(coupling, '_rrfs_transport_copy', None)
    if held_copy is None or held_copy[0] != key:
        metadata = []
        largest = 0
        for dst, src in copies:
            if dst.dtype != src.dtype or dst.dtype.itemsize != 4 or dst.ndim > 4:
                raise TypeError('RRFS transport copies require matching 32-bit words, rank <=4')
            if any(stride % 4 for stride in (*dst.strides, *src.strides)):
                raise ValueError('RRFS transport strides must be whole words')
            shape = (1,) * (4-dst.ndim) + dst.shape
            ds = (0,) * (4-dst.ndim) + tuple(s//4 for s in dst.strides)
            ss = (0,) * (4-src.ndim) + tuple(s//4 for s in src.strides)
            # Same coalesced destination traversal as CopyPlan.
            order = sorted(range(4), key=lambda k: (shape[k] > 1, -abs(ds[k])))
            sizes = tuple(shape[k] for k in order)
            metadata.append(tuple(np.int64(v) for v in
                (dst.size, *sizes[1:], *(ds[k] for k in order), *(ss[k] for k in order))))
            largest = max(largest, dst.size)
        copy_kernel = coupling.physics_cache.raw_kernel('rrfs_post_transport_copy',
            resident_transport_copy_source(), module_key='hexcore.mpasphys.rrfs.transport_copy')
        grid = (max(1, min(1024, (largest+255)//256)), 12)
        held_copy = coupling._rrfs_transport_copy = (key, metadata, copy_kernel, grid)
    _, metadata, copy_kernel, grid = held_copy
    # Keep only the current arrays alive; the proof harness reads these rows.
    coupling._rrfs_transport_copies = copies
    args = tuple(value for (dst, src), row in zip(copies, metadata)
                 for value in (dst, src, *row))
    copy_kernel(grid, (256,), args)
    if getattr(coupling.dycore, 'les_dissipation', None) is not None:
        coupling.dycore.les_dissipation.winds = columns
    p.commit()
    return columns
