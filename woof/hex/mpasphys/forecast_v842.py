# Modified 2026-10-28: experimental HRRR-like profile and aerosol route support.
"""MPAS suite entry at the real forecast door; mpas_atm_core.F:589-596,1036-1040."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from datetime import datetime, timedelta
from fractions import Fraction
import json
import os
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from ..config import parse_fortran_namelist
from ..physics_suites import load_table, route_default_for
from importlib import import_module
from .driver_v842 import interval_seconds
from .pool_v842 import decode_registry_pools, pool_transport_threads


def integral_duration(duration, dt):
    """Resolve CLI float hours to whole steps with the forecast door's tolerance.

    Five minutes arrives through argparse as 0.08333333333333333 hours.
    Its decimal Fraction is just below 300 seconds. Snap only within 1e-9
    of a step, as the regular forecast door does; reject real partial steps.
    """
    if duration is None or dt <= 0 or duration <= 0:
        raise ValueError('native run duration must be a positive integral number of config_dt steps')
    steps = duration / dt
    nearest = round(steps)
    if nearest <= 0 or abs(steps - nearest) > Fraction(1, 10**9):
        raise ValueError('native run duration must be an integral number of config_dt steps')
    return nearest * dt


@dataclass(frozen=True)
class Stream:
    name: str
    directions: tuple[str, ...]
    filename: str
    input_interval: str
    output_interval: str
    fields: tuple[str, ...]
    filename_interval: str = "none"
    input_file: str | None = None
    # Retain precision, clobber and file-format controls for the Rust writer.
    options: dict | None = None


def stream_input_bindings(values):
    """Bind native input files explicitly; mpas_atmphys_init.F:451-475."""
    result = {}
    for value in values or ():
        name, separator, filename = value.partition('=')
        if not separator or not name or not filename:
            raise ValueError('stream input requires NAME=FILE')
        if name in result:
            raise ValueError(f'duplicate native stream input {name}')
        path = Path(filename).resolve()
        if not path.is_file():
            raise ValueError(f'native stream input {name} does not exist: {path}')
        result[name] = path
    return result


def deck_stream_inputs(streams_path, init, *, bound=(), explicit=(), fallback=(), missing=None):
    """Fixed-name start-time input files a deck's streams.atmosphere names, found where native would read them.

    Native runs in the deck's directory and reads every input stream with input_interval="initial_only" from its
    filename_template there at startup (the RRFS fork's ``invariant`` stream, ``ugwp_oro_data_in``), not only the
    ``input`` stream (MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F, the startup stream reads). The door
    binds ``input`` from --init and the orographic file from --orographic-statistics; every other such stream is
    bound here from the deck's directory, then the init's directory, when the file is there. A name bound
    explicitly (--stream-input, or one of the door's own options) is never replaced. Templates with time tokens
    (restart, IAU, boundaries) and streams in a package are left to their own machinery.
    THE BREAKAGE THIS PREVENTS: an RRFS deck's invariant file (land use, soil category and other static fields)
    was read only with --stream-input invariant=..., so a default run kept the init's land and soil category at
    129 sea-ice cells on x1.2562 and differed from native at f000.
    ``missing``, when a list, receives ``(name, template, searched directories)`` for each such file not found, and
    nothing is printed: the run refuses the ones its streams leave active (``unread_deck_inputs_refusal``).
    """
    import xml.etree.ElementTree as ET
    if streams_path is None:
        return {}
    # The directories as named, not their link targets: an init that is a link into a case store still sits in
    # the run's own directory, where native reads (resolving it searched the store and found nothing).
    roots = [Path(streams_path).absolute().parent]
    if init is not None:
        roots.append(Path(init).absolute().parent)
    found, absent = {}, []
    for element in ET.parse(streams_path).getroot():
        name = element.get('name')
        kind = (element.get('type') or '').split(';')
        template = element.get('filename_template') or ''
        if (name is None or name in ('input', 'restart') or name in bound or name in explicit
                or 'input' not in kind or element.get('input_interval') != 'initial_only'
                or element.get('packages') or not template or '$' in template):
            continue
        path = next((root / template for root in dict.fromkeys(roots) if (root / template).is_file()), None)
        if path is not None:
            found[name] = path.resolve()
        elif name not in fallback:
            absent.append((name, template, tuple(dict.fromkeys(roots))))
    if missing is not None:
        missing.extend(absent)
    elif absent:
        import sys
        print('woof hex: deck start-time input not found beside the deck or the init, its fields keep the '
              "init's values: " + ', '.join(f'{name} ({template})' for name, template, _ in absent)
              + '; bind one with --stream-input NAME=FILE', file=sys.stderr)
    return found


def unread_deck_inputs_refusal(streams_path, missing, active):
    """The refusal for deck start-time input files the run would read and cannot find, else ``None``.

    THE BREAKAGE THIS PREVENTS: a deck whose invariant stream (input_interval="initial_only") names a file that is
    not beside the deck or the init ran on with a stderr line, keeping the init's land use and soil category
    (x1.2562 hrrrv5 2023-03-31 12Z: 12,800,256 of 29,644,818 history and diagnostics words different by f006
    against the same run reading the file). Native MPAS ignores the read error; a run that cannot be told apart
    from a correct one at a glance is refused here instead, naming the file. Streams the Registry's packages leave
    inactive are never read, so they are not refused.
    """
    unread = [(name, template, roots) for name, template, roots in missing if name in active]
    if not unread:
        return None
    named = '; '.join(f"stream {name} reads {template}, not found in {', '.join(str(root) for root in roots)}"
                      for name, template, roots in unread)
    return (f'the streams deck {streams_path} reads start-time input files the run cannot find ({named}). '
            "Without them their fields keep the init's values (for the RRFS invariant stream that is land use and "
            'soil category: 12,800,256 of 29,644,818 words different by f006 on x1.2562 hrrrv5). Put each file '
            'beside the deck or beside --init as named (a link is fine), bind it with --stream-input NAME=FILE, or '
            "set that stream's input_interval to \"none\" in the deck to run on the init's values")


def require_table_stream_inputs(config, bindings):
    """An active table stream must have real bytes before native pool hydration.

    RRFS Registry.xml:1299-1310,4366-4392 defines NGW climatology independently
    of the initial weather state. Missing it must not retain init/default tau.
    """
    row = load_table()[getattr(config, 'table_key', 'mpasphys_v842')]
    for stream in row.get('native_table_streams', ()):
        if config.execution_options[stream['option']] != stream['value']:
            continue
        name = stream['stream']
        if name not in bindings or not Path(bindings[name]).is_file():
            raise FileNotFoundError(f"active climatology stream {name} requires {stream['filename']}; "
                                    f"install the fork tables or bind --stream-input {name}=FILE")


def initial_state_input(arguments, config, text, streams_path, bindings):
    """Resolve the row's initial stream once for pool hydration and dycore input.

    RRFS mpas_atm_core.F:120-166 reads invariant, then exactly one of input,
    da_state or restart. Explicit stream bindings take precedence over a deck
    filename. Without a streams deck, --init names the selected state file.
    """
    row = load_table()[getattr(config, 'table_key', 'mpasphys_v842')]
    selector = row.get('initial_state_selector')
    if selector is None:
        return 'input', Path(arguments.init).resolve()
    configs = {key: value for group in parse_fortran_namelist(text).values()
               for key, value in group.items()}
    configs.update(config.execution_options)
    module, operation = selector.split(':', 1)
    name = getattr(import_module(module), operation)(configs)
    if name in bindings:
        return name, bindings[name]
    if name != 'input' and streams_path is not None:
        from .runtime_v842 import expand_input_filename
        parser_module, parser_call = row['streams_parser'].split(':', 1)
        tree = getattr(import_module(parser_module), parser_call)(streams_path)
        element = next((e for e in tree if e.get('name') == name), None)
        if element is None or 'input' not in element.get('type', '').split(';'):
            raise ValueError(f'initial state stream {name} is not an input stream in {streams_path}')
        start = getattr(arguments, 'start_time', None) or configs.get('config_start_time')
        template = element.get('filename_template', '')
        if start and template:
            filename = expand_input_filename(template, start)
            for directory in (Path(streams_path).absolute().parent, Path(arguments.init).absolute().parent):
                candidate = directory / filename
                if candidate.is_file():
                    return name, candidate.resolve()
        raise FileNotFoundError(f'initial state stream {name} has no readable file for {template!r}; '
                                f'bind it with --stream-input {name}=FILE')
    return name, Path(arguments.init).resolve()


def bind_stream_inputs(streams, bindings):
    """Keep native input and output paths distinct; mpas_atmphys_init.F:451-475."""
    known = {stream.name:stream for stream in streams}
    for name in bindings:
        if name not in known or 'input' not in known[name].directions:
            raise ValueError(f'native stream input {name} is not an active input stream')
    return tuple(replace(stream,input_file=str(bindings[stream.name]))
                 if stream.name in bindings else stream for stream in streams)


_MODE_KEYS = ('config_physics_mode', 'config_physics_correct_skip', 'config_numerics')


def split_physics_mode(text: str) -> tuple[str, str | None, tuple[str, ...]]:
    """Take the correct-mode keys out of &nhyd_model (MODE-KEY.md).

    The pristine MPAS v8.4.2 Registry has no such options, so the native pool decoder and the native
    library must never see them. A namelist without them is returned unchanged, character for character.
    """
    if not text.strip():
        return text, None, ()
    groups = {name: dict(values) for name, values in parse_fortran_namelist(text).items()}
    nhyd = {k.casefold(): k for k in groups.get('nhyd_model', {})}
    has_numerics = any(k.casefold() == 'config_numerics' for values in groups.values() for k in values)
    if not has_numerics and not any(key in nhyd for key in _MODE_KEYS):
        return text, None, ()
    mode = groups['nhyd_model'].pop(nhyd['config_physics_mode'], None) if 'config_physics_mode' in nhyd else None
    skip = groups['nhyd_model'].pop(nhyd['config_physics_correct_skip'], '') \
        if 'config_physics_correct_skip' in nhyd else ''
    for values in groups.values():
        for key in list(values):
            if key.casefold() == 'config_numerics':
                values.pop(key)
    if not isinstance(mode, (str, type(None))) or not isinstance(skip, str):
        raise ValueError('config_physics_mode and config_physics_correct_skip are character options')
    text = '\n'.join('&'+name+'\n'+'\n'.join(f'{k}={_literal(v)}' for k,v in values.items())+'\n/'
                     for name,values in groups.items())
    return text, None if mode is None else mode.strip(), tuple(s.strip() for s in skip.split(',') if s.strip())


def requested_physics_mode(arguments: Any, namelist_mode: str | None,
                           namelist_skip: tuple[str, ...]) -> tuple[str | None, tuple[str, ...]]:
    """The mode a run names (``--physics-mode`` or the namelist's config_physics_mode; None when neither
    does) and its skip list (``--physics-correct-skip``, repeatable or comma-separated, plus the namelist's)."""
    cli_mode = getattr(arguments, 'physics_mode', None)
    if cli_mode is not None and namelist_mode is not None and cli_mode != namelist_mode:
        raise ValueError(f'--physics-mode {cli_mode} and the namelist config_physics_mode={namelist_mode!r} '
                         'name two modes for one run')
    cli_skip = tuple(s.strip() for value in getattr(arguments, 'physics_correct_skip', None) or ()
                     for s in str(value).split(',') if s.strip())
    return (cli_mode if cli_mode is not None else namelist_mode,
            tuple(dict.fromkeys((*namelist_skip, *cli_skip))))


INIT_PROVENANCE_RRFS = 'gpuwm-hex.init-provenance/rrfs-v1'


def init_pin_refusal(init: Any, route_key: str, route: Mapping[str, Any], suite: str | None) -> None:
    """Refuse an init the RRFS producer made for another fork pin than the selected suite route's.

    THE BREAKAGE THIS PREVENTS (hrrrv5da, V8X6000 2026-10-29): the HRRR start deck and 15 lagged-HRRR
    member inits were built by the rrfs-mpas-1d6e370e producer (an older venv), then queued under the
    44c0849 LAM-DA forecast, and nothing said so. Measured on that case (ne3km hrrrv5, 2023-07-10 09Z,
    same met and recipe, only the pin changed): the 44c0849 init writes tke and the 1d6e370e init does
    not, w differs in 93,554 of 9,763,380 words, and the start deck's q2/u10/v10 differ in every cell
    (zgrid, theta, qv and u are identical: the L60 hybrid transition sits above the model top). So the
    run starts from another pin's state and is not the RRFS v2 model it names. Only an init whose
    ``<init>.provenance.json`` (cold_start_rrfs, schema rrfs-v1) names a producer is checked; an
    analysis, a restart or a native init carries no such receipt and is not judged here. The LBC set is
    not judged: case 9 of the two pins wrote the same 230,673,900 words on that case.
    """

    if init is None:
        return
    real = Path(init).expanduser().resolve()
    receipt = real.with_name(real.name + '.provenance.json')
    if not receipt.is_file():
        return
    try:
        document = json.loads(receipt.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return
    if not isinstance(document, Mapping) or document.get('schema') != INIT_PROVENANCE_RRFS:
        return
    producer = document.get('producer')
    made_by = producer.get('key') if isinstance(producer, Mapping) else None
    if not made_by:
        return
    from .rrfs.cold_start_rrfs import load_producers
    producers = load_producers()['producers']
    wanted = {key for key, row in producers.items() if row.get('physics_table_key') == route_key}
    if route.get('init_producer'):
        wanted.add(route['init_producer'])
    if not wanted or made_by in wanted:
        return
    made_pin = str(producers.get(made_by, {}).get('source_pin') or producer.get('source_pin') or '?')[:8]
    run_pin = route.get('pin_set') or route_key
    raise ValueError(
        f'{init} was made by the init producer {made_by} (fork {made_pin}, {receipt.name}), but this '
        f'forecast runs suite {suite!r} on the {run_pin} route, whose producer is {" or ".join(sorted(wanted))}. '
        'The two producers write different initial states (tke, w, the 2 m/10 m diagnostics), so the run '
        'would start from the other pin\'s state, neither pin. Rebuild the init with woof hex init '
        f'--physics-pin-set '
        f'{run_pin}, or run the forecast with the --physics-pin-set of the route that made the init')


def native_route(arguments: Any, *, default_mode: str = 'exact'):
    """Table-selected explicit suite; mpas_atmphys_control.F:92-332.

    ``default_mode`` is the mode a resolved MPAS suite route takes when the run names none.  The library
    default is exact (replay and reproduction callers are unchanged); the forecast door passes correct
    (owner ruling 2026-10-13: correct mode is the standard; coordinator 2026-10-14: the default is correct).
    """
    path = getattr(arguments, 'physics_namelist', None)
    text = path.read_text() if path else ''
    text, namelist_mode, namelist_skip = split_physics_mode(text)
    named_mode, correct_skip = requested_physics_mode(arguments, namelist_mode, namelist_skip)
    physics_mode = named_mode if named_mode is not None else default_mode
    groups = parse_fortran_namelist(text) if text.strip() else {}
    supplied = getattr(arguments, 'physics_suite', None)
    from .profiles_v842 import resolve_profile, finish_profile
    profile_name = getattr(arguments, 'physics_profile', None)
    profile = resolve_profile(profile_name, physics_mode)
    if profile:
        if supplied not in (None, profile['base_suite']):
            raise ValueError('physics profile and command-line suite differ')
        supplied = profile['base_suite']
    overrides = list(getattr(arguments, 'physics', None) or ())
    if profile:
        explicit = {k for options in groups.values() for k in options}
        overrides = [f'{k}={_literal(v)}' for k,v in profile['options'].items()
                     if k not in explicit] + overrides
    if getattr(arguments, 'physics_backend', None) == 'mpasphys':
        # The public backend selector names the MPAS suite route, rather than a
        # column-provider row. Without a suite, use MPAS's Registry default
        # (bundled core_atmosphere/Registry.xml:2299). Keep an explicit suite.
        supplied = supplied or groups.get('physics', {}).get('config_physics_suite') or 'mesoscale_reference'
    table = load_table()
    providers = [(key,row) for key,row in table.items()
                 if isinstance(row,Mapping) and (row.get('backend')=='mpasphys' or row.get('backend_row')==key)
                 and 'suite_routes' in row]
    routes = {alias for _,row in providers for alias in row['suite_routes']}
    selected = supplied or groups.get('physics',{}).get('config_physics_suite')
    pin_set = getattr(arguments, 'physics_pin_set', None)
    matches = [(key,row) for key,row in providers if selected in row['suite_routes']
               and (row.get('pin_set') == pin_set if pin_set else route_default_for(row, selected))]
    if pin_set and not matches:
        raise ValueError(f'physics pin set {pin_set!r} has no route for {selected!r}')
    if len(matches)>1:
        raise ValueError('native physics route is ambiguous in the suites table')
    if matches:
        init_pin_refusal(getattr(arguments, 'init', None), *matches[0], selected)
    provider = import_module(matches[0][1].get('driver_module',
                'woof.hex.mpasphys.driver_v842')) if matches else None
    # --physics-default-backend replaces the route's table default (hex-cuda on the v8.4.2 suites) for every
    # slot that names no backend of its own; native-cpu there is the native referee.
    default_backend = getattr(arguments, 'physics_default_backend', None)
    resolver_arguments = matches[0][1].get('resolver_arguments', {}) if matches else {}
    route_namelist = (partial(provider.route_namelist, default_backend=default_backend, **resolver_arguments)
                      if provider is not None else lambda text, **_: None)
    if supplied in routes and provider is not None:
        # An explicit provider alias may select an unchanged native namelist.
        # Compare table-resolved suite identities before retaining its bytes.
        # It takes the route's table default like the namelist route (it took the library's native-cpu
        # default before, so --physics-suite NAME ran every slot on the CPU).
        route_namelist = partial(provider.resolve_namelist,
                                 default_backend=default_backend or matches[0][1]['default_backend'],
                                 **resolver_arguments)
        existing = groups.get('physics', {}).get('config_physics_suite')
        def native_suite(name):
            resolved=provider.resolve_namelist(f'&physics\nconfig_physics_suite={name!r}\n/\n', **resolver_arguments)
            return getattr(resolved,'native_suite',resolved.suite)
        if existing and native_suite(existing) != native_suite(supplied):
            raise ValueError('command-line and namelist physics suites differ')
        if existing is None:
            text += f'\n&physics\nconfig_physics_suite={supplied!r}\n/\n'
    mode = dict(physics_mode=physics_mode, correct_skip=correct_skip)
    config = route_namelist(text, **mode) if text.strip() else None
    # A run with no MPAS suite route goes to the ArWen door, which reads the same --physics-mode and
    # --physics-correct-skip (woof.hex.correct_mode, lane H1): its corrections are the engine's (WB4, WB5),
    # and one the engine cannot apply is a named gap in that run's receipt, never a refusal.  Only a mode
    # written in the namelist's &nhyd_model is refused there, because the ArWen door never reads it.
    if config is None and namelist_mode not in (None, 'exact'):
        raise ValueError('config_physics_mode in the namelist applies to an explicit MPAS suite route '
                         '(--physics-suite or config_physics_suite); this run has none.  Name the mode with '
                         '--physics-mode, which the WOOF route reads')
    if config is None and namelist_skip:
        raise ValueError('config_physics_correct_skip in the namelist applies to an explicit MPAS suite '
                         'route; this run has none.  Name the skips with --physics-correct-skip')
    if config is None and default_backend is not None:
        # THE BREAKAGE THIS PREVENTS: a run naming the MPAS slot backend (the native referee, say) with no
        # MPAS suite route would run ArWen's physics while its author believes it ran MPAS's.
        raise ValueError('--physics-default-backend applies to an explicit MPAS suite route (--physics-suite '
                         'or config_physics_suite); this run has none')
    if config is not None and overrides:
        # Reconstruct one native physics group so parse_fortran_namelist never
        # discards a repeated group. WOOF-only selectors have no Registry row.
        groups = {name:dict(options) for name,options in parse_fortran_namelist(text).items()}
        for override in overrides:
            key, separator, value = override.partition('=')
            if not separator:
                raise ValueError(f'physics override requires KEY=VALUE: {override}')
            spec = matches[0][1]
            backend_options = {row['backend_selector'] for row in spec['slots'].values()}
            backend_options.update(row['backend_selector'] for row in spec.get('services', {}).values())
            option = key.strip().casefold()
            registry_groups = spec.get('registry_groups',{})
            owners = [name for name,keys in registry_groups.items() if option in keys]
            group = (spec['backend_namelist_group'] if option in backend_options
                     else owners[0] if owners else 'physics')
            value = override_literal(key.strip(), value.strip(), spec)
            parsed = parse_fortran_namelist(f'&{group}\n{key}={value}\n/')[group]
            groups.setdefault(group, {}).update(parsed)
        text = '\n'.join('&'+name+'\n'+'\n'.join(f'{k}={_literal(v)}' for k,v in values.items())+'\n/'
                         for name,values in groups.items())
        config = route_namelist(text, **mode)
    if config is not None and matches and path is None:
        # Only a run that brings no namelist of its own: a native namelist keeps pristine Registry defaults.
        config, text = _workflow_namelist(config, text, matches[0][1], route_namelist, mode)
    if profile:
        config = finish_profile(profile_name, profile, config, getattr(arguments, 'init', None))
    return config, text


def _workflow_namelist(config, text, row, route_namelist, mode):
    """The operational workflow's namelist pairings a suite row declares (``workflow_namelist``).

    A rule sets an option the run's own namelist leaves unset when the resolved selection matches, the way the
    operational workflow writes its namelist (RRFS: rrfs-workflow 08eee4ce parm/hrrrv5/namelist.atmosphere:58-59
    and scripts/exrrfs_fcst.sh:52-53 pair config_lsm_scheme='sf_ruc' with num_soil_layers=9).  THE BREAKAGE THIS
    PREVENTS: ``--physics-suite hrrrv5`` resolved the fork Registry's num_soil_layers=4 under RUC, whose nine
    soil nodes (and the nine-level RRFS init) then disagree with nSoilLevels at hydration.  An option the run
    names (``--physics KEY=VALUE``) is never changed, and a run that brings its own namelist is not touched.
    """
    rules = row.get('workflow_namelist') or ()
    if not rules:
        return config, text
    groups = {name: dict(values) for name, values in parse_fortran_namelist(text).items()}
    changed = False
    for rule in rules:
        if not all(str(config.options.get(key, '')).strip() == str(value)
                   for key, value in rule['when'].items()):
            continue
        for group, values in rule['set'].items():
            present = {k.casefold() for k in groups.get(group, {})}
            for key, value in values.items():
                if key.casefold() not in present:
                    groups.setdefault(group, {})[key] = value
                    changed = True
    if not changed:
        return config, text
    text = '\n'.join('&'+name+'\n'+'\n'.join(f'{k}={_literal(v)}' for k,v in values.items())+'\n/'
                     for name,values in groups.items())
    return route_namelist(text, **mode), text


def _registry_option_kinds() -> dict[str, str]:
    """``config_* -> Registry type`` for every nml_option of the bundled MPAS-A Registry (casefolded keys)."""
    global _OPTION_KINDS
    if _OPTION_KINDS is None:
        from .pool_v842 import registry_text
        root = Path(__file__).resolve().parents[1] / 'data' / 'mpasphys-framework'
        kinds: dict[str, str] = {}
        try:
            tree = ET.fromstring(registry_text(root / 'src/core_atmosphere/Registry.xml'))
            for option in tree.findall('.//nml_option'):
                if option.get('name') and option.get('type'):
                    kinds[option.get('name').casefold()] = option.get('type')
        except (OSError, ET.ParseError):
            kinds = {}
        _OPTION_KINDS = kinds
    return _OPTION_KINDS


_OPTION_KINDS: dict[str, str] | None = None


def override_literal(key: str, value: str, spec: Mapping[str, Any] | None = None) -> str:
    """One ``--physics KEY=VALUE`` value as a namelist literal, the way the WOOF route reads it.

    The WOOF route's ``physics_suites.parse_override`` takes ``KEY=VALUE`` with the value quoted or bare
    (``config_microp_scheme=mp_thompson``); this route used to paste the bare text into a namelist record,
    where an unquoted character value is not a scalar, so every bare spelling was refused
    (HEX-PHYSICS-PARITY run group 2 #6: the packaged moving-ensemble roster, Thompson members included).
    A value already quoted, or one that reads as a logical, integer or real, is kept as given; a bare word
    is quoted when the option is a character option in the Registry, a backend selector, or not a Registry
    option at all (WOOF-only selectors).  A bare word for a logical/integer/real option stays bare, so the
    namelist parser still refuses it by name.
    """
    text = value.strip()
    if not text or text[0] in '\'"':
        return text
    try:
        parse_fortran_namelist(f'&probe\n{key}={text}\n/')
        return text
    except Exception:
        pass
    option = key.strip().casefold()
    kind = _registry_option_kinds().get(option)
    defaults = (spec or {}).get('registry_defaults', {}) if isinstance(spec, Mapping) else {}
    for name, row in defaults.items():
        if name.casefold() == option and isinstance(row, Mapping):
            kind = row.get('type', kind)
    backend = option.endswith('_backend')
    if kind in (None, 'character') or backend:
        return repr(text)
    return text


def bind_door_iau(arguments: Any, text: str) -> str:
    """The door's IAU flags on the explicit suite route, as MPAS's own inputs.

    ``--iau-increment FILE`` (or ``--iau-analysis`` with the run's --init as the background, or an analysis
    bundle with --iau on) and ``--iau-window-seconds`` become the namelist's &iau group
    (config_IAU_option='on', config_IAU_window_length_s) and the iau stream's input file, so the run reads
    and applies the increment exactly as an MPAS namelist that asks for IAU does (runtime_v842.SuiteIau).
    Returns the namelist text with that &iau group.
    """
    window = float(arguments.iau_window_seconds)
    if not window > 0:
        raise ValueError('--iau-window-seconds must be positive; a zero window is direct insertion')
    if getattr(arguments, 'iau_route', None) == 'hex':
        # Breakage: woof.hex.iau_forcing's coupled increment is an ArWen-route executor hook; on this route the
        # flag would be accepted and never used.
        raise ValueError("--iau-route hex adds a coupled increment on the WOOF route's executors; the explicit "
                         "MPAS suite route applies MPAS's own IAU (atm_add_tend_anal_incr).  Drop --iau-route or "
                         "name native")
    groups = {name: dict(values) for name, values in parse_fortran_namelist(text).items()} if text.strip() else {}
    existing = {key.casefold(): value for key, value in groups.get('iau', {}).items()}
    if str(existing.get('config_iau_option', 'off')).strip().lower() != 'off':
        raise ValueError("the namelist's &iau and --iau-window-seconds both name an IAU window; give it once")
    increment = getattr(arguments, 'iau_increment', None)
    if increment is None:
        analysis = getattr(arguments, 'iau_analysis', None)
        if analysis is None:
            raise ValueError('--iau-window-seconds needs --iau-increment, or --iau-analysis to form one')
        if not Path(analysis).expanduser().is_file():
            raise ValueError(f'--iau-analysis {analysis} is not a file')
        from ..analysis_bundle import write_increment
        work = Path(str(Path(arguments.out).expanduser().resolve()) + '.iau-work')
        work.mkdir(parents=True, exist_ok=False)
        increment = work / 'iau-increment.nc'
        write_increment(Path(analysis).expanduser().resolve(), Path(arguments.init).expanduser().resolve(), increment)
    if not Path(increment).expanduser().is_file():
        raise ValueError(f'--iau-increment {increment} is not a file')
    bindings = list(getattr(arguments, 'stream_input', None) or ())
    if any(str(binding).partition('=')[0] == 'iau' for binding in bindings):
        raise ValueError('--stream-input iau= and the IAU flags both name the increment file; give it once')
    arguments.stream_input = [*bindings, f'iau={Path(increment).expanduser().resolve()}']
    groups['iau'] = {'config_IAU_option': 'on', 'config_IAU_window_length_s': window}
    return '\n'.join('&'+name+'\n'+'\n'.join(f'{k}={_literal(v)}' for k,v in values.items())+'\n/'
                     for name,values in groups.items())


def _literal(value: Any) -> str:
    """Namelist values preserve Registry kinds; mpas_atmphys_control.F:92-332."""
    if isinstance(value, bool):
        return '.true.' if value else '.false.'
    return repr(value)


def set_namelist_options(text: str, options: Mapping[str, Mapping[str, Any]]) -> str:
    """Set namelist options, keeping every other record (rebuilt as native_route rebuilds overrides)."""
    groups = {name: dict(values) for name, values in parse_fortran_namelist(text).items()} if text.strip() else {}
    for group, values in options.items():
        target = groups.setdefault(group, {})
        for key, value in values.items():
            for existing in [k for k in target if k.casefold() == key.casefold()]:
                del target[existing]
            target[key] = value
    return '\n'.join('&'+name+'\n'+'\n'.join(f'{k}={_literal(v)}' for k,v in values.items())+'\n/'
                     for name,values in groups.items())+'\n'


def namelist_option(text: str, group: str, key: str):
    """One namelist option (case-insensitive), or None."""
    if not text.strip():
        return None
    values = {k.casefold(): v for k, v in dict(parse_fortran_namelist(text).get(group, {})).items()}
    return values.get(key.casefold())


def apply_start_time(text: str, start_time: str | None) -> str:
    """``--start-time`` on the MPAS route names the run's config_start_time.

    THE BREAKAGE THIS PREVENTS: the MPAS route read its start only from the namelist, so a run (or a moving
    case's ``physics_args``) naming a suite without a namelist start ran silently from the Registry default,
    2014-09-10_00:00:00: wrong sun, wrong ozone month, wrong output times.  A namelist naming a different
    start than ``--start-time`` is refused.  A run without ``--start-time`` is unchanged.
    """
    if not start_time:
        return text
    named = namelist_option(text, 'nhyd_model', 'config_start_time')
    if named is not None:
        if str(named).strip() != start_time:
            raise ValueError(f'--start-time {start_time} and the namelist config_start_time {named} name two '
                             'starts for one run')
        return text
    return set_namelist_options(text, {'nhyd_model': {'config_start_time': start_time}})


def bind_init_start(text: str, init: Any) -> str:
    """The run's config_start_time from the init's own ``xtime`` when nothing else names one.

    THE BREAKAGE THIS PREVENTS: a suite run with no namelist start and no ``--start-time``
    (``woof hex forecast --physics-suite convection_permitting`` from an install) ran from MPAS's Registry
    default start, 2010-10-23_00:00:00: output named ``output.2010-10-23_00.00.00.nc``, the sun and the ozone
    and aerosol months of that date (seen 2026-10-16, hexmx proof, x1.2562 2023-03-31 12Z init).  The init is
    the authority for the run's start, as the door's help says; a namelist start or ``--start-time`` still
    decides, and a restart run (config_do_restart) keeps MPAS's own restart timestamp rule.
    """
    if namelist_option(text, 'nhyd_model', 'config_start_time') is not None:
        return text
    restart = namelist_option(text, 'restart', 'config_do_restart')
    if restart is True or str(restart).strip().lower() in ('.true.', 'true', 't'):
        return text
    if init is None:
        return text
    from ..lbc import _open_lbc, _read_xtime
    path = Path(init)
    try:
        valid = _read_xtime(_open_lbc(path), path)
    except Exception as error:
        raise ValueError(f'the init {path} carries no readable xtime, so the run has no start time; name '
                         f'--start-time or a namelist config_start_time ({error})') from error
    start = valid.strftime('%Y-%m-%d_%H:%M:%S')
    import sys
    print(f'woof hex: MPAS suite start {start} (the init\'s xtime)', file=sys.stderr, flush=True)
    return set_namelist_options(text, {'nhyd_model': {'config_start_time': start}})


def bind_limited_area(arguments: Any, text: str) -> str:
    """``--lbc-dir`` on the MPAS suite route is MPAS's own limited-area run: ``config_apply_lbcs = .true.``.

    THE BREAKAGE THIS FIXES: the suite route never read ``--lbc-dir``.  A region, point, country, swath or cycle fine
    forecast with ``--physics-suite convection_permitting`` reached the route's global dycore build with a culled mesh
    and stopped in its terrain coupling ("cells_on_edge contains a missing or out-of-range cell", 2026-10-24 run group
    1 #3, after 13 min of prep), so no limited-area run could take MPAS physics.  With the flag, the pools decode the
    limited_area package and runtime_v842 opens the regional dycore (the WOOF route's boundary driver, which already
    carries the MPAS-A v8.4.1 specified and relaxation zones) under the same physics glue.

    Combinations whose state the regional dycore does not carry yet are refused by name below.  A namelist that says
    ``config_apply_lbcs = .false.`` beside ``--lbc-dir`` names two different runs and is refused.
    """
    lbc_dir = getattr(arguments, 'lbc_dir', None)
    if lbc_dir is None:
        named = namelist_option(text, 'limited_area', 'config_apply_lbcs')
        if named is True or str(named).strip().lower() in ('.true.', 'true', 't'):
            # THE BREAKAGE THIS PREVENTS: a namelist limited-area run with no boundary series would run its relaxation
            # zone against nothing.
            raise ValueError('the namelist sets config_apply_lbcs = .true. but the run names no --lbc-dir: a limited-'
                             'area MPAS suite run needs its lateral-boundary series')
        return text
    directory = Path(lbc_dir).expanduser()
    if not directory.is_dir():
        raise ValueError(f'--lbc-dir {directory} is not a directory')
    named = namelist_option(text, 'limited_area', 'config_apply_lbcs')
    if named is not None and not (named is True or str(named).strip().lower() in ('.true.', 'true', 't')):
        raise ValueError('--lbc-dir names a limited-area run and the namelist sets config_apply_lbcs = .false.; '
                         'give one')
    gaps = []
    # More than one card (--samehost-devices) is admitted: each rank builds its regional dycore on its own cut of the
    # region (the partition layout rides the local mesh; PartitionLbcPoolV841 slices the boundary series to it), and
    # the scalar halo round native issues before atm_bdy_adjust_scalars wraps the rank's boundary adjust
    # (samehost_v842.SameHostRankExecution.bind_dycore), as the ArWen route's regional ranks do.  IAU is admitted
    # on one card: its increment is held on the padded extent and the stage-density wrapper is carried onto the
    # regional dycore (guide/l11-regional-iau; the RRFS fork's regional constructor takes it too, fix/hex-lam-da-v842);
    # a same-host rank's regional dycore does not take that wrapper yet.
    # The JEDI hand-off is admitted (fix/hex-lam-da-v842): the background is the native pools on the solve cells
    # (cycling_v842.background_fields strips the garbage element), the analysis is recoupled on the culled mesh
    # (allow_regional_sentinels) and the regional dycore opens on it at the analysis time, so its boundary series,
    # relaxation zone and boundary tendencies are the straight run's (state_at is a function of the clock and the
    # two bracketing lbc files only).  THE BREAKAGE THE OLD REFUSAL NAMED ("carries the global extent only") was the
    # checkpoint carrier holding the padded regional state; the carrier now holds the solve extent.
    # IAU is the door's flags or the namelist's own &iau group (a flag run and a namelist run execute the same code,
    # forecast_door's bind_door_iau).  THE BREAKAGE THIS PREVENTS: reading only the flags admitted a namelist IAU run
    # on several cards here, and every rank then refused it in runtime_v842 after the whole multi-minute launch.
    named_iau = str(namelist_option(text, 'iau', 'config_IAU_option') or 'off').strip().strip('\'"').lower()
    iau = (getattr(arguments, 'iau_window_seconds', None) is not None or getattr(arguments, 'iau_increment', None)
           or named_iau != 'off')
    if iau and getattr(arguments, 'samehost_devices', None):
        gaps.append("IAU on more than one card (the regional IAU wrapper runs on one card; a same-host "
                    "rank's regional dycore does not carry it yet)")
    if getattr(arguments, 'stochastic', None):
        gaps.append('stochastic physics (its pattern is built on the unpadded mesh)')
    if getattr(arguments, 'chemistry_config', None):
        gaps.append('chemistry transport (its tracer rows are not driven by the lateral-boundary series)')
    if gaps:
        raise ValueError('a limited-area MPAS suite run (--lbc-dir) cannot take ' + '; '.join(gaps) +
                         ' yet.  Drop that option, or run the region on the default WOOF physics')
    return set_namelist_options(text, {'limited_area': {'config_apply_lbcs': True}})


def _mpas_interval(seconds: float) -> str:
    """Seconds as an MPAS interval, D_HH:MM:SS."""
    whole = int(round(seconds))
    if whole <= 0 or abs(seconds - whole) > 1e-6:
        raise ValueError(f'an MPAS stream interval must be a whole positive number of seconds, not {seconds}')
    days, rest = divmod(whole, 86400)
    return f'{days}_{rest // 3600:02d}:{rest % 3600 // 60:02d}:{rest % 60:02d}'


_HEADER_KEYS = ('type', 'filename_template', 'filename_interval', 'reference_time', 'record_interval',
                'clobber_mode', 'precision', 'io_type', 'packages', 'input_interval', 'output_interval')


def _in_defaults(element: Any) -> int:
    """streams_gen's is_structure_writable for the build's one key, in_defaults=true: 1, 0, or -1 (absent)."""
    value = element.get('in_defaults')
    if value is None:
        return -1
    return 1 if 'true' in value.split(';') else 0


def write_default_streams(layout: Any, path: Path, *, history_minutes: float | None = None,
                          checkpoint_hours: Any = None, surface_interval: str | None = None,
                          omit: Mapping[str, Any] | None = None, bound_inputs=()) -> Path:
    """MPAS's own default streams.atmosphere and stream_list.atmosphere.* files, from the Registry this run decoded.

    THE BREAKAGE THIS PREVENTS: the MPAS suite route refused every run that named no streams.atmosphere
    ("requires the actual streams.atmosphere (--streams)") and every run whose streams file named
    stream_list.atmosphere.{output,diagnostics,surface,diag_ugwp} files the install did not ship, so
    ``woof hex forecast --physics-suite convection_permitting`` and ``woof hex global`` with an MPAS suite
    could not run from an install at all: only hand-made decks carrying a native run directory's files could
    (seen 2026-10-16, image proofs).  MPAS's build writes both from the preprocessed Registry
    (src/core_atmosphere/Makefile:38, ``streams_gen Registry_processed.xml streams.atmosphere
    stream_list.atmosphere. listed in_defaults=true``; src/tools/input_gen/streams_gen.c); this does the same,
    in the same order: every stream ``in_defaults`` admits, a separate_file stream naming its
    ``stream_list.atmosphere.<name>`` (var_arrays then vars, one per line) written beside the streams file,
    a single_file stream carrying its members inline.  The run's own choices are then applied: output and
    diagnostics every ``history_minutes`` (default: the Registry's own intervals), each time to its own file;
    restart only at ``checkpoint_hours`` (a single interval), else never; the surface stream at
    ``surface_interval`` when the run updates SST from a bound series (:func:`surface_series_interval`).  A run
    that names its streams file is unchanged.

    ``omit`` is the route row's ``default_streams_omit`` (``{stream: {"fields": [...], "source": ...}}``): fields a
    fork Registry lists in a default stream that the route computes no word for and the operational deck does not
    write.  THE BREAKAGE IT PREVENTS (hrrrv5 hex region, 2026-10-29): the RRFS fork's Registry lists the PV
    diagnostics (ertel_pv ... iLev_DT) in its default output stream, the RRFS route ports no mpas_pv_diagnostics, so
    every suite-named hrrrv5 run with no deck refused at its first write ("MPAS output fields lack computed
    authority"), while RRFS's own deck writes none of them.  A run that names its streams file is still held to
    every field it lists.

    A member whose Registry dimensions include one of length zero in this run is left out of every default stream:
    no NetCDF file holds a zero-length dimension other than the unlimited one.  THE BREAKAGE IT PREVENTS (same run):
    the fork's chemistry Registry adds ddvel, drydep_flux and wetdep_resolved (nChem x nCells, no package) to the
    default output stream, nChem is num_chem = 0 without smoke, and the first write refused ("dimension 'nChem' has
    length 0").
    """
    root = ET.Element('streams')
    bound = set(bound_inputs)
    dropped = {name: set((spec or {}).get('fields') or ()) for name, spec in (omit or {}).items()}
    sizes = getattr(layout, 'dimensions', None) or {}
    empty = {f['name'] for f in layout.fields if any(sizes.get(d) == 0 for d in f.get('dimensions') or ())}
    history = _mpas_interval(float(history_minutes) * 60.0) if history_minutes else None
    restart = 'none'
    if checkpoint_hours:
        values = [v for v in str(checkpoint_hours).split(',') if v.strip()]
        if len(values) == 1:
            restart = _mpas_interval(float(values[0]) * 3600.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    for declared in layout.root.findall('./streams/stream'):
        name = declared.get('name')
        immutable = declared.get('immutable') == 'true'
        bound_input = name in bound and 'input' in declared.get('type', '').split(';')
        writable = _in_defaults(declared) != 0 or bound_input
        if immutable:
            if not writable:
                continue
            members = []
        else:
            members = [m for tag in ('stream', 'var_struct', 'var_array', 'var') for m in declared.findall(tag)
                       if (_in_defaults(m) == 1 or (_in_defaults(m) == -1 and writable))
                       and m.get('name') not in dropped.get(name, ()) and m.get('name') not in empty]
            if not writable and not members:
                continue
        element = ET.SubElement(root, 'immutable_stream' if immutable else 'stream', {'name': name})
        for key in _HEADER_KEYS:
            if declared.get(key) is not None:
                element.set(key, declared.get(key))
        types = declared.get('type', '').split(';')
        if bound_input and element.get('input_interval', 'none') == 'none':
            element.set('input_interval', 'initial_only')
        if name in ('output', 'diagnostics') and history:
            element.set('output_interval', history)
        if name == 'restart':
            element.set('output_interval', restart)
        if name == 'surface' and surface_interval:
            element.set('input_interval', surface_interval)
        if 'output' in types and element.get('output_interval', 'none') != 'none' \
                and declared.get('filename_interval') is None:
            element.set('filename_interval', 'output_interval')
        if immutable:
            continue
        separate = declared.get('runtime_format') == 'separate_file'
        listed = []
        if separate:
            list_name = f'stream_list.atmosphere.{name}'
            ET.SubElement(element, 'file', {'name': list_name})
        for member in members:
            if separate and member.tag in ('var_array', 'var'):
                listed.append(member.get('name'))
                continue
            attributes = {'name': member.get('name')}
            if member.get('packages') is not None:
                attributes['packages'] = member.get('packages')
            ET.SubElement(element, member.tag, attributes)
        if name == 'output':
            from woof.hex.cycle.land_state import LAND_FIELDS
            active = {f['name'] for f in layout.fields}
            for field in LAND_FIELDS:
                if field.name not in active:
                    continue
                if separate:
                    if field.name not in listed:
                        listed.append(field.name)
                elif not any(e.get('name') == field.name for e in element):
                    ET.SubElement(element, 'var', {'name': field.name})
        if separate:
            (path.parent / list_name).write_text(''.join(f'{field}\n' for field in listed))
    ET.indent(root)
    path.write_text(ET.tostring(root, encoding='unicode') + '\n')
    return path


def surface_series_interval(path: Path, n_cells: int) -> str:
    """The record interval of a bound surface series, as the default surface stream's ``input_interval``.

    THE BREAKAGE THIS PREVENTS (U-H1, best-180 smoke j1, 2026-10-27): MPAS's Registry declares the surface stream
    with ``input_interval="none"``, so a run on the default streams that turned on ``config_sst_update`` and bound
    ``--stream-input surface=FILE`` either refused at the surface alarm or never read a record, and SST and sea
    ice stayed bit-identical from day 0 to day 7 (the s2s door's ``--surface-series``).  Native MPAS needs the
    interval written into streams.atmosphere by hand; this writes the series' own uniform record spacing.
    """
    from ..surface_series_reader import reader_library
    from .surface_update_v842 import RustSurfaceRecordReader
    times = RustSurfaceRecordReader(reader_library(), path, n_cells=n_cells).times
    if len(times) < 2:
        raise ValueError(f'surface series {path} needs at least two record times to update SST')
    spacing = {b - a for a, b in zip(times, times[1:])}
    if len(spacing) != 1:
        raise ValueError(f'surface series {path} record interval must be uniform for the surface stream alarm')
    return _mpas_interval(spacing.pop().total_seconds())


def read_streams(path: Path, layout: Any, list_directory: Path | None = None, *, tree=None) -> tuple[Stream, ...]:
    """Resolve Registry/file lists and packages; Registry.xml:498-1351."""
    elements = list(ET.parse(path).getroot() if tree is None else tree)
    # mpas_atm_core_interface.F:193-202: an initial-only invariant stream
    # disables its embedding in input/restart, independently of direction.
    # Allocation already includes those static fields; this selects I/O only.
    invariant = next((element for element in elements if element.get('name') == 'invariant'), None)
    layout.packages.discard('no_invariant_stream')
    if invariant is None or invariant.get('input_interval') != 'initial_only':
        layout.packages.add('no_invariant_stream')
    active = {r['name']:r for r in layout.fields}
    declared = {s.get('name'):s for s in layout.root.findall('./streams/stream')}
    # MPAS recursive stream expansion: Registry.xml:498-1351.
    def members(element, seen=(), order=None, copy_fields=False):
        result = []
        # Immutable Registry streams are compiled by member kind, rather than
        # XML declaration order: MPAS gen_inc.c:2095-2188.
        children = list(element) if order is None else [child for tag in order for child in element.findall(tag)]
        for child in children:
            if not copy_fields and not layout.enabled(child):
                continue
            name = child.get('name')
            if child.tag in ('var','var_array'):
                if name in active:
                    result.append(name)
            elif child.tag=='var_struct':
                # Native allocates variable arrays before ordinary fields, even
                # when their XML declarations are interleaved: gen_inc.c:1944-1952.
                rows = [r for r in layout.fields if r['pool'] == name]
                result.extend(r['name'] for arrays in (True, False) for r in rows
                              if bool(r['components']) == arrays)
            elif child.tag=='stream':
                if name in seen:
                    raise ValueError(f'cyclic native stream {name}')
                # add_stream_fields copies allocated members without their old package gates
                # (mpas_stream_manager.F:1004-1059). An XML include supplies its own gate.
                copied = copy_fields or order is None
                result.extend(members(declared[name], (*seen,name),
                    ('stream', 'var_struct', 'var_array', 'var'), copy_fields=copied))
            elif child.tag=='file':
                member_path=(list_directory or path.parent)/name
                for line in member_path.read_text().splitlines():
                    field = line.split('#',1)[0].strip()
                    if field:
                        result.append(field)
            else:
                raise ValueError(f'unknown MPAS stream member {child.tag}')
        return result
    streams=[]
    for element in elements:
        if not layout.enabled(element):
            continue
        name=element.get('name')
        fields=(members(declared[name], (name,), ('stream', 'var_struct', 'var_array', 'var'))
                if element.tag == 'immutable_stream' else members(element))
        streams.append(Stream(name,tuple(element.get('type').split(';')),
                              element.get('filename_template'),element.get('input_interval','none'),
                              element.get('output_interval','none'),tuple(dict.fromkeys(fields)),
                              {'input_interval':element.get('input_interval','none'),
                               'output_interval':element.get('output_interval','none')}.get(
                                   element.get('filename_interval','none'),
                                   element.get('filename_interval','none')),
                              options=dict(element.attrib)))
    return tuple(streams)


def _marked_durable(frame: Path, marker: Path) -> bool:
    """Whether ``frame`` already carries a durable sidecar that names its current size and mtime."""
    try:
        receipt = json.loads(marker.read_text(encoding='utf-8'))
        stat = frame.stat()
    except (OSError, ValueError):
        return False
    return (isinstance(receipt, dict) and receipt.get('bytes') == stat.st_size
            and receipt.get('mtime_ns') == stat.st_mtime_ns)


def named_library_route(arguments: Any, config: Any) -> bool:
    """Whether a run that names a native MPAS library takes it because the ported framework cannot run the selection.

    THE BREAKAGE THIS PREVENTS: with every active slot on hex-cuda, ``runtime_v842.framework_only`` picks the
    ported framework, which runs only what its resident interface glue admits.  The RRFS fork's hrrrv5 put every
    slot on the GPU once the UGWP ports shipped again (owner ruling 2026-10-14), so a run naming a site-built RRFS
    library (``--native-physics-library`` or MPAS_NATIVE_LIBRARY), the documented RRFS route, was refused by name
    ("its resident interface glue is not in this distribution") instead of running on that library with the GPU
    slots, as it did while gwdo was native-cpu.  When the ported framework can run the selection it is still taken.
    """
    if config is None:
        return False
    library = getattr(arguments, 'native_physics_library', None) or os.environ.get('MPAS_NATIVE_LIBRARY')
    if not library:
        return False
    if not any(config.active(slot) for slot in (getattr(config, 'selectors', None) or {})):
        # No active physics slot: the ported framework runs the selection itself, library named or not.
        return False
    from .resident_v842 import ported_route_refusal
    return ported_route_refusal(config) is not None


def native_table_source(config: Any, named_source: Any, source: Any, ported_framework: bool) -> Any:
    """The MPAS tree whose ``physics_wrf/files`` holds the tables a suite row's table streams read.

    A named ``--mpas-source`` (or MPAS_SOURCE) is that tree, and so is the library route's provider tree.  On the
    ported framework with none named, the Registry comes from the wheel (``registry_source``), which carries no
    tables; the row's ``framework.table_source`` names where its tables live.  THE BREAKAGE THIS PREVENTS: the
    ported route took the wheel's Registry-only tree as its table root, so hrrrv5 on a plain install with no
    --mpas-source refused at start ("No such file or directory: .../hexcore/data/mpasphys-framework-rrfs/src/
    core_atmosphere/physics/physics_wrf/files"; V5090 RRFS env, woof hex c2efafcf, 2026-10-28).
    """
    if named_source or not ported_framework:
        return source
    from .route_table import framework_entry
    return framework_entry(config, 'table_source')()


def run_forecast(arguments: Any, config: Any, text: str) -> int:
    """Hydrate Registry pools for resident physics; core.F:589-596."""
    pin_set = getattr(config, 'pin_set', None) or 'v8.4.2'
    reader = getattr(arguments, 'native_pool_reader', None) or os.environ.get('MPAS_NATIVE_POOL_READER')
    source = getattr(arguments, 'mpas_source', None) or os.environ.get('MPAS_SOURCE')
    named_source = source
    from .runtime_v842 import framework_only
    ported_framework = config is None or (framework_only(config) and not named_library_route(arguments, config))
    if ported_framework:
        # Framework pools, clock and initialization are Python-owned. The
        # suite row's expanded pinned Registry (v8.4.2 or the RRFS fork)
        # ships with the wheel. No native library or MPAS source tree is
        # requested, including on a clean install; a row whose table streams
        # read physics tables takes them from its framework.table_source
        # (native_table_source).
        from .route_table import framework_entry
        library = None
        source = source or framework_entry(config, 'registry_source')()
        if not reader:
            from .resources_build import ensure_route_resources
            reader = ensure_route_resources(('pool-reader',), pin_set=pin_set)['pool-reader']
    else:
        # The separate RRFS fork retains its own provider and pin set.
        library = getattr(arguments, 'native_physics_library', None) or os.environ.get('MPAS_NATIVE_LIBRARY')
        from .resources_build import ROUTE_KEYS, ensure_route_resources
        keys = ROUTE_KEYS[pin_set]
        unnamed = tuple(key for key, value in ((keys['native-library'], library), ('pool-reader', reader),
                                               (keys['mpas-source'], source)) if not value)
        if unnamed:
            installed = ensure_route_resources(unnamed, pin_set=pin_set)
            library = library or installed.get(keys['native-library'])
            reader = reader or installed.get('pool-reader')
            source = source or installed.get(keys['mpas-source'])
    required = {'native-pool-reader': reader, 'mpas-source': source,
                'init': getattr(arguments, 'init', None)}
    if not ported_framework:
        required['native-physics-library'] = library
    for option, value in required.items():
        if not value:
            raise ValueError(f'MPAS physics route requires --{option}: its provider is not bound')
        if not Path(value).exists():
            raise ValueError(f'MPAS physics route --{option} does not exist: {value}')
    if config is not None and library is not None:
        from ..licence_gate import native_library_refusal
        refusal = native_library_refusal(config, Path(library))
        if refusal is not None:
            raise refusal
    namelist = getattr(arguments,'physics_namelist',None)
    # A run that names no streams file takes the namelist's sibling when there is one, else MPAS's own default
    # written from the Registry once the pools are decoded (write_default_streams).
    streams_path = getattr(arguments,'streams',None)
    if streams_path is None and namelist is not None and namelist.with_name('streams.atmosphere').is_file():
        streams_path = namelist.with_name('streams.atmosphere')
    if streams_path is not None and not Path(streams_path).is_file():
        raise ValueError(f'MPAS native route: the named streams.atmosphere does not exist: {streams_path}')
    out = getattr(arguments,'out',None)
    if out is None:
        raise ValueError('MPAS native route requires --out for stream products and the receipt')
    out = Path(out).resolve()
    scratch = getattr(arguments,'scratch',None) or out/'scratch'
    scratch = Path(scratch).resolve()
    sources = [Path(v).resolve() for k in ('grid','static','init','orographic_statistics')
               if (v:=getattr(arguments,k,None)) is not None]
    bindings = stream_input_bindings(getattr(arguments,'stream_input',None))
    row = load_table()[getattr(config,'table_key','mpasphys_v842')]
    table_bindings = {}
    selector = row.get('native_table_stream_selector')
    if selector is not None:
        module, operation = selector.split(':', 1)
        roots = [Path(native_table_source(config, named_source, source, ported_framework)).resolve()
                 / 'src/core_atmosphere/physics/physics_wrf/files']
        extra = getattr(arguments,'native_physics_tables',None) or []
        keyword = row.get('native_table_override_keyword')
        callback = getattr(import_module(module), operation)
        table_bindings = (callback(config, roots, **{keyword: extra}) if keyword
                          else callback(config, [*roots, *extra]))
    explicit = {'ugwp_oro_data_in'} if getattr(arguments,'orographic_statistics',None) else set()
    deck_missing = []
    deck_inputs = deck_stream_inputs(streams_path, getattr(arguments,'init',None), bound=bindings,
                                     explicit=explicit, fallback=table_bindings, missing=deck_missing)
    bindings = {**deck_inputs, **bindings}
    require_table_stream_inputs(config, {**table_bindings, **bindings})
    initial_stream, initial_file = initial_state_input(arguments, config, text, streams_path, bindings)
    if initial_stream != 'input':
        # Do not hydrate a warm start from the cold input or the unselected restart/DA file.
        from copy import copy
        arguments = copy(arguments)
        arguments.init = initial_file
        sources = [Path(v).resolve() for k in ('grid', 'static', 'orographic_statistics')
                   if (v := getattr(arguments, k, None)) is not None]
        bindings = {name: path for name, path in bindings.items()
                    if name not in ('input', 'restart', 'da_state') or name == initial_stream}
        bindings[initial_stream] = initial_file
    # Timed surface input is first read in the forecast loop, after startup
    # output; mpas_atm_core.F:198-210,655-724,779-806. Preserve init SST/xice
    # during pool hydration and install timed records only on their alarms.
    startup_bindings = (path for name, path in bindings.items()
                        if name != 'surface' or not config.options['config_sst_update'])
    sources = list(dict.fromkeys((*sources,*table_bindings.values(),*startup_bindings)))
    if initial_stream != 'input':
        sources = [path for path in sources if path != initial_file] + [initial_file]
    text = apply_start_time(text, getattr(arguments, 'start_time', None))
    text = bind_limited_area(arguments, text)
    from .checkpoint_v842 import resolve_route
    moving, text = resolve_route(arguments, text)
    if (moving is not None and not getattr(moving, 'fixed_mesh', False)
            and getattr(arguments, 'lbc_dir', None) is not None):
        # THE BREAKAGE THIS PREVENTS: a moving region carries its own boundary (its parent's frames through the move
        # hooks); a second, fixed --lbc-dir series would drive a zone that no longer sits where the files were cut.
        # A checkpoint, restart or DA analysis on the run's own fixed mesh (--checkpoint-hours, --restart-from,
        # --da-analysis with no WOOF_HEX_MOVING_* switch) is not a move: its zone sits where the files were cut, and
        # the regional dycore re-reads the bracketing lbc files at the resume time (fix/hex-lam-da-v842).
        raise ValueError('a moving region on the MPAS suite route takes its boundary from the move hooks; drop --lbc-dir')
    text = bind_init_start(text, getattr(arguments, 'init', None))
    from ..stage_timing import stage
    timings = []
    out.mkdir(parents=True, exist_ok=True)
    def save_timings():
        (out/'stage-timings.json').write_text(json.dumps({'stages':timings},indent=2)+'\n')
    beside_transport = None
    if (library is None and getattr(arguments, 'samehost_devices', None) and moving is None
            and not getattr(arguments, 'preflight', False)):
        from .lts_v842 import lts_requested
        if not lts_requested(arguments):
            # Same-host ranks on a limited area: the mesh, geometry, vertical grid and initial state read only the
            # grid and init files, so they are read while the pool transport runs (runtime_v842.start_regional_inputs).
            def beside_transport(layout):
                if not layout.configs.get('config_apply_lbcs', False):
                    return lambda: None
                from .runtime_v842 import start_regional_inputs
                return start_regional_inputs(arguments, layout)
    try:
        pool_threads = pool_transport_threads()  # what decode_registry_pools hands the reader (U-H3)
        decode_parallelism = {'device':'CPU','threads':pool_threads}
        if beside_transport is not None:
            decode_parallelism['beside'] = ('limited-area mesh, geometry, vertical grid and initial state on a side '
                                            'thread (waited for at the end of this stage)')
        with stage(timings, 'native-pool-decode', parallelism=decode_parallelism,
                   timeline='prep.native.pool_decode'):
            call = dict(reader=Path(reader).resolve(), source=Path(source).resolve(), config=config, text=text,
                        sources=sources, output=scratch/'native-pools',
                        initial_state=(initial_stream, streams_path,
                                       getattr(arguments, "stream_list_directory", None))
                        if initial_stream != 'input' else None)
            decode = lambda: decode_registry_pools(call['reader'], call['source'], config, text, sources,
                                                   call['output'], initial_state=call['initial_state'],
                                                   beside_transport=beside_transport)
            # A resident moving process (moving_resident_v842) decoded this member's pools beside the previous
            # segment; the prepared layout is taken only when its call is this one, word for word.
            from . import moving_resident_v842 as resident
            layout = resident.decode(call, arguments, decode) if beside_transport is None else decode()
    finally:
        save_timings()
    if moving is not None:
        moving.mark_restart(layout)
    if streams_path is None:
        timed = ({'surface_interval': surface_series_interval(bindings['surface'], layout.dimensions['nCells'])}
                 if config.options['config_sst_update'] and 'surface' in bindings else {})
        streams_path = write_default_streams(layout, scratch/'streams.atmosphere',
                                             history_minutes=getattr(arguments,'history_every_minutes',None),
                                             checkpoint_hours=None, omit=row.get('default_streams_omit'),
                                             bound_inputs=bindings, **timed)
    parser=load_table()[getattr(config,'table_key','mpasphys_v842')].get('streams_parser')
    tree=None
    if parser is not None:
        module,callable_name=parser.split(':')
        tree=getattr(import_module(module),callable_name)(streams_path,library)
    streams=read_streams(streams_path,layout,getattr(arguments,'stream_list_directory',None),tree=tree)
    from ..forecast_start import radar_streams
    streams = radar_streams(streams, layout, out, Stream)
    defaults = {initial_stream: initial_file}
    if getattr(arguments,'orographic_statistics',None):
        defaults['ugwp_oro_data_in'] = Path(arguments.orographic_statistics).resolve()
    active = {s.name for s in streams if 'input' in s.directions}
    refusal = unread_deck_inputs_refusal(streams_path, deck_missing, active)
    if refusal is not None:
        raise ValueError(refusal)
    defaults = {name:path for name,path in defaults.items() if name in active}
    # A deck file found by default binds only a stream the Registry/packages left active (never a refusal).
    bindings = {name: path for name, path in bindings.items() if name not in deck_inputs or name in active}
    streams = bind_stream_inputs(streams,{**defaults,**table_bindings,**bindings})
    start=layout.configs['config_start_time']
    dt=Fraction(str(layout.configs['config_dt']))
    selected_dt=getattr(arguments,'dt',None)
    if selected_dt not in (None,'auto','row'):
        dt=Fraction(str(selected_dt))
    if dt<=0:
        raise ValueError('native physics needs a positive config_dt')
    hours=getattr(arguments,'hours',None)
    duration=Fraction(str(hours))*3600 if hours is not None else interval_seconds(layout.configs['config_run_duration'])
    duration = integral_duration(duration, dt)
    if callable(getattr(moving, 'resolve_schedule', None)):
        duration = moving.resolve_schedule(duration, dt, streams)
    stop=(datetime.strptime(start,'%Y-%m-%d_%H:%M:%S')+timedelta(seconds=float(duration))).strftime('%Y-%m-%d_%H:%M:%S')
    out.mkdir(parents=True,exist_ok=True)
    from .. import numerics as _numerics
    plan=dict(schema='mpas-native-door-v1',physics=config.receipt(),numerics=_numerics.receipt_block(),
              start=start,stop=stop,dt=str(dt),
              grid=None if getattr(arguments,'grid',None) is None else str(Path(arguments.grid).resolve()),
              init=str(Path(arguments.init).resolve()),
              steps=int(duration/dt),streams=[s.__dict__ for s in streams],
              native_library=None if library is None else str(Path(library).resolve()),
              framework='python' if library is None else 'native-referee',
              rust_pool_reader=str(Path(reader).resolve()),
              pool_manifest=str(scratch/'native-pools/manifest.json'),identity_checked=False)
    (out/'native-physics-request.json').write_text(json.dumps(plan,indent=2)+'\n')
    if getattr(arguments,'preflight',False):
        print(json.dumps(dict(status='native-pool-preflight',identity_checked=False,request=plan)))
        return 0
    # The start record (woof.hex.forecast_start), before the first step, as the ArWen route writes it: a door that
    # joins this live run (woof hex add-nest, a site drawing hours as they land) reads mesh, init, start, hours
    # and cadence here instead of a receipt complete only at the end.
    history = next((s for s in streams if s.name == 'output' and 'output' in s.directions), None)
    try:
        history_seconds = None if history is None else interval_seconds(history.output_interval)
    except ValueError:  # an interval with no fixed length (a calendar month, initial_only): no cadence to record
        history_seconds = None
    history_minutes = getattr(arguments, 'history_every_minutes', None)
    if history_minutes is None and history_seconds:
        history_minutes = float(history_seconds) / 60.0
    from ..forecast_start import write_start_record
    write_start_record(out, paths={'grid': getattr(arguments, 'grid', None) or arguments.init,
                                   'static': getattr(arguments, 'static', None), 'init': arguments.init},
                       schedule={'start_time': start, 'forecast_hours': float(duration) / 3600.0,
                                 'history_every_minutes': history_minutes,
                                 'history_from_minutes': getattr(arguments, 'history_from_minutes', None)})
    if getattr(arguments, 'physics_namelist', None) is None and row.get('native_initialization_namelist'):
        # A run that names its suite instead of a namelist file (`--physics-suite hrrrv5`, the regional door)
        # stages the namelist it resolved where the route reads one. THE BREAKAGE IT PREVENTS: the RRFS route
        # copies the run's namelist into the native working directory (driver_rrfs.stage_native_initialization_
        # namelist) and died on None (TypeError) for every hrrrv5 run with no --physics-namelist.
        resolved = scratch / 'namelist.atmosphere.resolved'
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(text)
        arguments.resolved_physics_namelist = resolved
    from .features_v842 import run_native_dynamics
    parallelism = {'device':'GPU plus Python framework' if library is None else 'GPU plus native CPU','cards':1,
                   'requested_omp_threads':int(os.environ.get('OMP_NUM_THREADS','1'))}
    if getattr(arguments, 'samehost_devices', None):
        from ..drivers.samehost_forecast_v841 import device_list
        parallelism.update(device=('same-host GPU rank threads plus Python framework' if library is None
                                   else 'MPI GPU ranks plus native CPU'),
                           cards=len(device_list(arguments.samehost_devices)))
    try:
        with stage(timings, 'native-forecast-runtime', parallelism=parallelism):
            try:
                rc = run_native_dynamics(arguments, config, layout, scratch/'native-pools', streams,
                                         None if library is None else Path(library).resolve(),
                                         start, stop, float(dt), int(duration/dt), out,
                                         moving=moving)
            finally:
                from .runtime_v842 import discard_regional_inputs
                discard_regional_inputs(layout)
        if rc == 0:
            from ..history_writer import durable_marker, mark_closed_frame_durable
            from ..products_door import native_output_frames
            # Frames in a file of their own were published by the writer as each closed
            # (streams_output_v842._publish_closed); only multi-record files are left here.
            with stage(timings, 'native-history-durability', parallelism={'device':'CPU','threads':1},
                       timeline='post.native.history_durability'):
                for frame in native_output_frames(out):
                    if not _marked_durable(frame, durable_marker(frame)):
                        mark_closed_frame_durable(frame)
        return rc
    finally:
        save_timings()
