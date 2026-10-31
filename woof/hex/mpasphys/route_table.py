"""Table-selected framework, resident glue and services of an MPAS suite route.

Each MPAS suite row of ``data/profiles/mpas-physics-suites.v1.json`` (``mpasphys_v842``, ``mpasphys_rrfs``) names
the pieces that let its route run with no native MPAS library:

* ``framework``: the ported framework backend class, its scheme initialization, the shipped Registry and (for a
  row whose table streams read physics tables) where those tables live;
* ``resident``: the GPU-resident interface glue, the selections it admits and the receipt scope;
* ``services.<name>.service``: the CUDA service class bound at the driver boundary;
* ``stream_contract``: the Registry stream module, the module that hands streams to ``rw_mpas_stream``, the
  per-file global attributes and the pinned build's ``git_version`` the route's history, diagnostics and restart
  files are written with (:func:`stream_contract`);
* ``output_diagnostics``: the class that computes the diagnostic stream fields on the device.

Admission is by table row, never by a hard-coded row key, so a route with no native-free piece in this
distribution says which one (:func:`framework_absence`) and a new route is a table row, not a code path.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from importlib import import_module
import importlib.util
from typing import Any, Callable, Mapping

DEFAULT_ROW = 'mpasphys_v842'

# A row with no ``stream_contract`` block writes MPAS-A v8.4.2's (its git_version is stream_metadata_v842's pin).
_DEFAULT_STREAM_CONTRACT = {'registry': 'woof.hex.streams_v842', 'writer': 'woof.hex.restart_v842'}
_DEFAULT_OUTPUT_DIAGNOSTICS = 'woof.hex.mpasphys.resident_output_diagnostics_v842:ResidentOutputDiagnostics'


def route_row(config: Any) -> Mapping[str, Any]:
    """The suite table row of ``config`` (v8.4.2 when the configuration names none)."""
    from ..physics_suites import load_table
    return load_table()[getattr(config, 'table_key', DEFAULT_ROW)]


def resolve(reference: str) -> Any:
    """``package.module:attribute`` from a table row."""
    module, attribute = reference.split(':', 1)
    return getattr(import_module(module), attribute)


def _present(reference: str) -> bool:
    module = reference.split(':', 1)[0]
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def framework_absence(config: Any) -> str | None:
    """None when this distribution carries the row's ported framework, else what is missing."""
    row = route_row(config)
    framework = row.get('framework')
    if not framework:
        return f'suite row {getattr(config, "table_key", DEFAULT_ROW)} declares no ported framework'
    missing = sorted({reference.split(':', 1)[0] for reference in framework.values() if not _present(reference)})
    if missing:
        return 'its ported framework modules are not in this distribution: ' + ', '.join(missing)
    return None


def framework_entry(config: Any, name: str) -> Any:
    """``backend``, ``initialization``, ``registry_source`` or ``table_source`` of the row's ported framework."""
    framework = route_row(config).get('framework') or {}
    if name not in framework:
        raise RuntimeError(f'suite row {getattr(config, "table_key", DEFAULT_ROW)} declares no framework {name}')
    entry = resolve(framework[name])
    arguments = route_row(config).get('framework_arguments', {}).get(name, {})
    if arguments:
        from functools import partial
        return partial(entry, **arguments)
    return entry


def resident_row(config: Any) -> Mapping[str, Any] | None:
    """The row's resident glue declaration, or None when the row has none."""
    return route_row(config).get('resident')


def resident_targets(config: Any) -> tuple[Mapping[str, str], ...]:
    """The slot selections the row's resident glue admits (empty when it declares none)."""
    row = resident_row(config)
    return tuple(row.get('targets', ())) if row else ()


def resident_glue_present(config: Any) -> bool:
    row = resident_row(config)
    return bool(row) and _present(row['glue'])


def resident_glue_type(config: Any) -> Any:
    row = resident_row(config)
    if not row:
        raise RuntimeError(f'suite row {getattr(config, "table_key", DEFAULT_ROW)} declares no resident glue')
    return resolve(row['glue'])


def forecast_arg_overrides(forecast_args: Any) -> dict[str, str]:
    """The ``--physics KEY=VALUE`` overrides a door's forecast flags carry (``--forecast-arg=`` unwrapped; last wins)."""
    items = [str(item)[len('--forecast-arg='):] if str(item).startswith('--forecast-arg=') else str(item)
             for item in forecast_args or ()]
    found: dict[str, str] = {}
    index = 0
    while index < len(items):
        flag, equals, value = items[index].partition('=')
        if flag == '--physics':
            if not equals:
                index += 1
                value = items[index] if index < len(items) else ''
            key, _, setting = value.partition('=')
            if key:
                found[key.strip()] = setting.strip()
        index += 1
    return found


def named_variant(suite: str | None, overrides: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]] | None:
    """The named variant (a route row's ``variants`` entry) a suite plus selector overrides is, or None.

    A variant is a suite with the selector changes its row declares (``hrrrv5 + storm scheme``: hrrrv5 with
    GF-Li on the coarse cells); its name and its differences from the suite go on the run's receipt, so a run
    that is not the suite as its authors run it never carries the suite's name alone.
    """
    if not suite:
        return None
    from ..physics_suites import load_table
    given = {str(k): str(v).strip() for k, v in (overrides or {}).items()}
    for row in load_table().values():
        if not isinstance(row, Mapping):
            continue
        for name, variant in (row.get('variants') or {}).items():
            if variant.get('suite') == suite and all(given.get(k) == v for k, v in variant['selectors'].items()):
                return name, variant
    return None


def resident_scope(config: Any) -> str:
    row = resident_row(config) or {}
    return str(row.get('scope', ''))


def service_type(config: Any, service: str, default: str) -> Any:
    """The CUDA service class a row names for ``service`` (``default`` for rows that name none)."""
    row = route_row(config).get('services', {}).get(service, {})
    return resolve(row.get('service', default))


@dataclass(frozen=True)
class StreamContract:
    """The stream contract one route writes its files with, bound per writer, never per process.

    ``registry`` is the Registry stream module (``Stream``, ``StreamAlarms``, ``field_definition``,
    ``registry_stream``, ``parse_streams`` and the interval helpers), ``writer`` the module that hands a stream to
    ``rw_mpas_stream`` (``write_stream``, ``write_restart``, ``read_restart``, ``RestartControl``),
    ``file_attributes`` the global attributes each new file adds after its ``file_id`` and ``git_version`` the
    pinned build's (None: stream_metadata_v842's).  THE BREAKAGE THIS PREVENTS: the writer bound MPAS-A v8.4.2's
    Registry at import, so the RRFS route refused its own history fields (ng, ns, volg, nh, qh, nbca and the fork's
    defc_a [maxEdges, nCells], zs [nSoilLevels, nCells]) and ran with history off (hrrrv5 x1.2562, 2026-10-28).
    """
    registry: Any
    writer: Any
    git_version: str | None = None
    file_attributes: Callable[[Any], Mapping[str, Any]] | None = None

    def stream_file_attributes(self, stream: Any) -> dict:
        return {} if self.file_attributes is None else dict(self.file_attributes(stream))

    def alarms(self, streams, start, initial, final, control, active_packages):
        """The contract's stream alarms with every restart/DA control flag its alarm class declares."""
        alarms = self.registry.StreamAlarms
        declared = {item.name for item in fields(alarms)}
        flags = {name: getattr(control, 'config_' + name)
                 for name in ('do_restart', 'do_DAcycling') if name in declared}
        return alarms(streams, start, initial, final, active_packages=frozenset(active_packages), **flags)


def stream_contract(config: Any = None) -> StreamContract:
    """The row's stream contract (MPAS-A v8.4.2's for a row, or a configuration, that names none)."""
    block = dict(route_row(config).get('stream_contract') or _DEFAULT_STREAM_CONTRACT)
    attributes = block.get('file_attributes')
    registry = import_module(block['registry'])
    if 'registry_data' in block:
        from pathlib import Path
        registry = registry.RegistryContract(Path(__file__).resolve().parents[1] / block['registry_data'],
                                             **block.get('registry_options', {}))
    return StreamContract(registry=registry, writer=import_module(block['writer']),
                          git_version=block.get('git_version'),
                          file_attributes=None if attributes is None else resolve(attributes))


def output_diagnostics_type(config: Any = None) -> Any:
    """The class the row computes its diagnostic stream fields with (v8.4.2's for a row that names none)."""
    return resolve(route_row(config).get('output_diagnostics') or _DEFAULT_OUTPUT_DIAGNOSTICS)
