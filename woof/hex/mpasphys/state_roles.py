"""Native scalar allocation authority, distinct from a producer's input slots.

Roles come from the selected driver/Registry and producer declaration tables.
An inactive input slot is retained source data, never an evolved model field.
"""
from __future__ import annotations
import hashlib
import inspect
import json
from importlib import import_module, resources
from pathlib import Path
from collections.abc import Mapping

SCHEMA = 'gpuwm-hex.native-state-roles/v1'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def _literal(value):
    if type(value) is bool:
        return '.true.' if value else '.false.'
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if type(value) in (int, float):
        return repr(value)
    raise ValueError('native role resolver requires scalar namelist values')


def resolve_receipt(physics):
    """Re-run the actual table driver, checking its complete recorded selection."""
    from ..physics_suites import load_table, TABLE_PATH
    table = load_table()
    matches = [(k, r) for k, r in table.items() if isinstance(r, Mapping)
               and r.get('driver_module') == physics.get('driver_module')
               and r.get('pin_set') == physics.get('pin_set')
               and r.get('model_commit') == physics.get('model_commit')]
    if len(matches) != 1:
        raise ValueError('native roles need a unique recorded driver/model/pin table authority')
    key, row = matches[0]
    module = import_module(row['driver_module'])
    spec = module.specification(table_key=key)
    options = physics['options']
    if set(options) != set(spec['registry_defaults']):
        raise ValueError('native roles require the complete recorded Registry options')
    text = '\n'.join('&' + group + '\n' + '\n'.join(
        f"{name} = {_literal(physics['suite'] if name == 'config_physics_suite' else options[name])}" for name in names if name in options) + '\n/'
        for group, names in spec['registry_groups'].items())
    # Version selectors live outside Registry options. Replaying without them
    # would resolve the library default instead of the recorded scheme version.
    if versions := spec.get('mynn_versions'):
        text += ('\n&' + spec['backend_namelist_group'] + '\n'
                 + versions['selector'] + ' = '
                 + _literal(physics.get('mynn_version', 'pinned')) + '\n/')
    keywords = dict(backends=physics['backends'], physics_mode=physics.get('physics_mode', 'exact'),
                    correct_skip=physics.get('correct_skip', ()))
    if 'table_key' in inspect.signature(module.resolve_namelist).parameters:
        keywords['table_key'] = key
    config = module.resolve_namelist(text, **keywords)
    actual = config.receipt()
    if physics != actual:
        changed = sorted(k for k in physics.keys() | actual.keys() if physics.get(k) != actual.get(k))
        raise ValueError(f'native roles resolver/recorded physics disagree: {changed}')
    if not hasattr(module, 'scalar_registry'):
        raise ValueError('native roles driver needs an authoritative scalar_registry adapter')
    active = [r['name'] for r in module.scalar_registry(config)]
    known = [r['name'] for r in spec['scalar_registry']]
    return config, row, dict(table_key=key, active=active, known=known,
        selector_table_sha256=digest(resources.files('woof.hex').joinpath(TABLE_PATH).read_bytes()),
        resolver_source_sha256=digest(Path(inspect.getsourcefile(module.resolve_namelist)).read_bytes()))


def allocation_contract(physics, manifest, *, native_init, start):
    """An explicit allocation/producer layout; no values or clocks are fabricated.

    The manifest's origins, not a radar scalar list, prove ordered native slots.
    Only a table-declared init producer can authorize inactive retained slots.
    """
    config, row, authority = resolve_receipt(physics)
    from .rrfs.cold_start_rrfs import load_producers, declarations
    producer_key = row.get('init_producer')
    if producer_key is None:
        producers = load_producers()['producers']
        candidates = [(k, v) for k, v in producers.items()
                      if v['physics_table_key'] == authority['table_key']]
        if len(candidates) != 1:
            raise ValueError('native roles require an explicit unique producer declaration')
        producer_key, producer = candidates[0]
    else:
        producer = load_producers()['producers'][producer_key]
    if producer['source_pin'] != row['model_commit']:
        raise ValueError('native role producer/model source pins disagree')
    slots = [r for r in manifest['fields'] if r['pool'] == 'state' and r['name'] == 'scalars']
    if len(slots) != 1:
        raise ValueError('native roles require exactly one allocated state.scalars manifest row')
    slot = slots[0]
    active = authority['active']
    if (slot['kind'] != 'real' or slot['dimensions'] != ['num_scalars', 'nVertLevels', 'nCells']
            or len(slot['shape']) != 3 or slot['shape'][0] != len(active)
            or any(type(n) is not int or n < 1 for n in slot['shape'])):
        raise ValueError('native allocated scalar shape disagrees with resolved Registry')
    expected = [f'{Path(native_init)}:{n}' for n in active]
    if slot.get('origins') != expected:
        raise ValueError('native scalar origins/order do not bind the exact selected init and Registry')
    declared = {v['name'] for v in declarations(producer, start, 'prepare')['variables']}
    retained = sorted(set(authority['known']).intersection(declared) - set(active))
    return dict(schema=SCHEMA, physics=physics, authority=authority,
                producer_key=producer_key, producer_source_pin=producer['source_pin'],
                producer_declarations_sha256=digest(resources.files('woof.hex').joinpath(
                    'data/profiles/' + producer['declarations']).read_bytes()),
                producer_table_sha256=digest(resources.files('woof.hex').joinpath(
                    'data/profiles/init-producers.v1.json').read_bytes()),
                native_active=active, producer_only=retained,
                semantics={'native_active':'actual allocated transported native scalar slots',
                           'producer_only':'retained input source data; not evolved or transported prognostic state'},
                source_init=str(Path(native_init)), source_clock=start,
                native_scalar_payload_sha256=slot['sha256'], native_scalar_manifest=slot)
