"""Bind regional initialization to the existing preparation and input owners."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

SCHEMA = 'gpuwm.regional-background.v1'


class MissingPreparationManifest(ValueError):
    """The acquisition stage completed without its required manifest."""


def validate_request(request):
    for name in ('source', 'source_cycle', 'source_product', 'source_provider',
                 'source_root', 'prepared_root', 'prepared_config', 'prepared_namelist'):
        value = getattr(request, name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f'{name} must be nonempty text; supply the value or omit it.')
    cadence = request.forcing_cadence_hours
    if cadence is not None and (type(cadence) is not int or cadence <= 0):
        raise ValueError('Forcing cadence needs a positive integer number of hours; correct the source cadence.')
    if request.source_inputs and not request.prepared_root:
        raise ValueError('Original source role bindings accompany a prepared bundle; supply prepared_root or remove those role bindings.')
    if request.source_root and request.prepared_root:
        raise ValueError('Raw and prepared input roots are alternative initialization paths; select one.')
    if (request.prepared_config or request.prepared_namelist) and not request.prepared_root:
        raise ValueError('Prepared authorities need their prepared root; supply it or remove the authority paths.')
    if not isinstance(request.supplements, (list, tuple)):
        raise ValueError('Supplements must be an array of ROLE=PATH bindings.')
    for binding in request.supplements:
        if not isinstance(binding, str) or not all(binding.partition('=')[::2]):
            raise ValueError('A supplement needs ROLE=PATH; correct its binding.')
    if request.source_inputs is not None and (not isinstance(request.source_inputs, dict)
            or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
                   for k, v in request.source_inputs.items())):
        raise ValueError('Source inputs must map role names to existing file paths.')


class PlanError(ValueError):
    """A regional background request this preparation cannot honour."""

    def __init__(self, message: str, *, code: str = "INVALID_PLAN", details=None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


def _stamp(text):
    """An ISO timestamp with a UTC offset, on a whole second, as UTC."""
    try:
        stamp = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PlanError(f"epoch {text!r} is not an ISO timestamp; include a UTC offset.") from exc
    if stamp.tzinfo is None:
        raise PlanError("epoch has no timezone; append Z or an explicit UTC offset.")
    stamp = stamp.astimezone(timezone.utc)
    if stamp.microsecond:
        raise PlanError("epoch has fractional seconds; choose a whole-second boundary.")
    return stamp


def _files(paths):
    from woof.output_identity import file_record
    return [file_record(Path(path).expanduser().resolve()) for path in sorted(set(map(str, paths)))]


def verify_files(records):
    from woof.output_identity import file_record
    for expected in records:
        actual = file_record(Path(expected['path']))
        if actual != expected:
            raise ValueError(f"Background input {expected['path']} changed after review; restore its bound bytes or review the new inputs.")


def read_prepared(root, config, namelist, *, source, run_seconds, history_interval_seconds, physics_profile=None):
    """Use the ordinary portable reader, including its source/cache checks."""
    from woof import stage_cli
    from woof.prepared_single_domain_forecast import preflight_prepared_forecast
    root = Path(root).expanduser().resolve()
    bundle = stage_cli.resolve_bundle(root)
    if bundle['source'] != source:
        raise ValueError(f"Prepared source {bundle['source']} differs from selected {source}; select the actual bundle source.")
    if bundle['layout'] != 'single' or bundle['domains'] != 1:
        raise ValueError('Regional cycling needs one prepared domain; supply a single-domain bundle for the reviewed region.')
    digests = stage_cli.single_domain_digests(bundle)
    return preflight_prepared_forecast(
        source=source, prepared_root=root, proof_sha256=digests['proof'],
        source_manifest_sha256=digests['source_manifest'], prepared_content_sha256=digests['prepared_content'],
        experiment_config=Path(config).expanduser().resolve(), wps_namelist=Path(namelist).expanduser().resolve(),
        run_seconds=run_seconds, history_interval_seconds=history_interval_seconds, physics_profile=physics_profile)


def _prepared_request(request, source, duration, cadence):
    root = Path(request.prepared_root).expanduser().resolve()
    config = Path(request.prepared_config) if request.prepared_config else root / 'experiment.toml'
    namelist = Path(request.prepared_namelist) if request.prepared_namelist else root / 'namelist.wps'
    inputs = read_prepared(root, config, namelist, source=source,
                          run_seconds=duration, history_interval_seconds=cadence, physics_profile=request.profile)
    paths = [inputs.proof_path, inputs.source_manifest_path, inputs.experiment_config,
             inputs.wps_namelist, inputs.static_path, inputs.geometry_receipt_path]
    paths.extend(inputs.authority_paths.values())
    if (root / 'forcing-member.json').is_file():
        paths.append(root / 'forcing-member.json')
    return inputs, dict(kind='prepared', root=str(root), config=str(inputs.experiment_config),
                        namelist=str(inputs.wps_namelist), files=_files(paths))


def _prepared_cycle(inputs):
    """Read cycle evidence from the actual bundle's bound authorities."""
    from woof.prepared_single_domain_forecast import proof_initial_forecast_lead
    from woof.fetch import parse_cycle
    import tomllib
    initial = inputs.experiment.start_time.replace(tzinfo=timezone.utc)
    def cycle_value(text):
        value = datetime.fromisoformat(text.replace('Z', '+00:00'))
        value = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
        if value.minute or value.second or value.microsecond:
            raise ValueError('The prepared source cycle is not on a whole UTC hour; restore its original cycle authority.')
        return parse_cycle(value.strftime('%Y-%m-%dT%H'), inputs.source).replace(tzinfo=timezone.utc)
    if inputs.proof.get('source_cycle'):
        return cycle_value(inputs.proof['source_cycle'])
    if 'initial_condition' in inputs.proof:
        return initial - timedelta(hours=proof_initial_forecast_lead(inputs.proof))
    source = inputs.source_manifest.get('source')
    if isinstance(source, dict) and source.get('cycle'):
        return cycle_value(source['cycle'])
    declared = tomllib.loads(inputs.experiment_config.read_text(encoding='utf-8')).get('fetch', {})
    cycle = declared.get('cycle')
    if cycle and cycle != 'latest':
        from woof.source_adapters import get_source_adapter
        if get_source_adapter(declared.get('source', '')).source_id != inputs.source:
            raise ValueError('The prepared fetch authority names another source; restore the original source authority.')
        return cycle_value(cycle)
    raise ValueError('The supplied forecast bundle has no bound source cycle; supply its original configuration authority or prepare a bundle that records the cycle.')


def _prepared_member(request, inputs, adapter):
    """A selected member needs the actual native payload identity."""
    member = inputs.source_member
    if adapter.member_set and member is None:
        from woof.forcing_member import member_contract, verify_unchanged
        from woof.member_prep import verify_member_file
        from woof.source_authorities import packaged_member_grammar_sha256
        path = inputs.prepared_root / 'forcing-member.json'
        if not path.is_file():
            raise ValueError('This prepared bundle lacks its selected-member binding; supply the original verified member inputs or a bundle that retains member identity.')
        receipt = json.loads(path.read_text(encoding='utf-8'))
        selection = request.source_member if request.source_member is not None else receipt.get('member')
        _, grammar, selection = member_contract(adapter.source_id, selection)
        if (receipt.get('schema') != 'arwen.forcing-member.v1'
                or receipt.get('source') != adapter.source_id or receipt.get('member') != selection
                or receipt.get('grammar_sha256') != packaged_member_grammar_sha256(adapter.member_set)):
            raise ValueError('The preparation member receipt contradicts the current member contract; restore its verified member inputs.')
        primary = inputs.source_manifest.get('primary_files', [])
        files = receipt.get('files', [])
        if not files or [row.get('sha256') for row in files] != [row.get('sha256') for row in primary]:
            raise ValueError('The member receipt does not describe the prepared primary input order; restore the matching preparation receipt.')
        verify_unchanged(receipt)
        for row in files:
            verify_member_file(grammar, selection, Path(row['path']))
        verify_unchanged(receipt)
        member = selection
    if adapter.selection_owner:
        # This owner's preparation manifest binds the original GRIB role.
        # The optional original bytes let its native checker establish the
        # selected ensemble member when the portable cache omits that label.
        from importlib import import_module
        owner = import_module(adapter.selection_owner)
        selected = owner.validate_selection(product_type=request.source_product or 'reanalysis',
            member=request.source_member, cadence=request.forcing_cadence_hours or
            int(inputs.boundary_interval_seconds / 3600), provider=request.source_provider or 'cds',
            cycle=inputs.experiment.start_time)
        if selected is not None:
            supplied = request.source_inputs or {}
            path = supplied.get('grib')
            if path is None:
                raise ValueError('This portable bundle has no source-member label; supply the original grib role to verify the selected member.')
            record = _files([path])[0]
            bound = inputs.source_manifest.get('files', {}).get('grib', {})
            if record['sha256'] != bound.get('sha256'):
                raise ValueError('The supplied member bytes differ from the prepared source manifest; restore its original grib role.')
            owner.check_member(Path(path), selected)
            verify_files([record])
            member = selected
    if request.source_member is not None and str(member) != str(request.source_member):
        raise ValueError('The prepared source member differs from the requested member; select the actual member or supply matching inputs.')
    return member


def review_background(request, rung, *, now=None, probe=None):
    from woof.background_contract import capability, plan
    from woof.source_adapters import get_source_adapter
    from woof.da.background import DEFAULT_BACKGROUND_SOURCE
    from woof.background_contract import BackgroundWindowError
    validate_request(request)
    adapter = get_source_adapter(request.source or DEFAULT_BACKGROUND_SOURCE)
    source, facts = adapter.source_id, capability(adapter.source_id)
    duration = rung['cycles'] * rung['cadence_seconds'] + rung['forecast_seconds']
    initial = _stamp(request.epoch)
    kwargs = dict(init=initial, now=now or datetime.now(timezone.utc), run_seconds=duration,
                  product=request.source_product, member=request.source_member,
                  provider=request.source_provider, cadence_hours=request.forcing_cadence_hours,
                  cycle=None if request.source_cycle is None else _stamp(request.source_cycle))
    binding, inputs = dict(kind='automatic', files=[]), None
    if request.prepared_root:
        inputs, binding = _prepared_request(request, source, duration, rung['cadence_seconds'])
        member = _prepared_member(request, inputs, adapter)
        kwargs['supplied_member'] = member
        if member is not None:
            kwargs['member'] = member
        if inputs.experiment.start_time.replace(tzinfo=timezone.utc) != initial:
            raise ValueError('The prepared initial state has another valid time; select that epoch or prepare the requested initial state.')
        kwargs['inventory_times'] = tuple(inputs.experiment.start_time.replace(tzinfo=timezone.utc) +
                                          timedelta(hours=h) for h in inputs.forcing_hours)
        kwargs['cadence_hours'] = kwargs['cadence_hours'] or inputs.boundary_interval_seconds // 3600
        if facts['time_axis'] == 'forecast_leads':
            actual_cycle = _prepared_cycle(inputs)
            if kwargs['cycle'] is not None and kwargs['cycle'] != actual_cycle:
                raise ValueError('The prepared source cycle differs from the selected cycle; use matching prepared inputs.')
            kwargs['cycle'] = actual_cycle
        binding['files'] += _files((request.source_inputs or {}).values())
        binding['experiment'] = _experiment_identity(inputs.experiment)
    elif request.source_root:
        # A staged root stays usable when its source gains acquisition
        # (f17bdb8c3, woof.source_drivability.local_input_requested): the
        # fetch, go and run-plan doors took that rule and this one kept
        # asking whether the source REQUIRES a root, so a local 20CRv3 CF
        # archive was refused here once the source could also be fetched.
        from woof.source_drivability import local_input_requested
        if not local_input_requested({'source': source, 'source_root': request.source_root}):
            raise ValueError('This source uses its automatic chain or a prepared bundle; supply prepared_root for an existing preparation.')
        from woof.local_preparation import inspect_local_inputs
        interval = request.forcing_cadence_hours or int(adapter.forcing_interval_seconds / 3600)
        anchor = kwargs['cycle'] or initial
        start = (initial - anchor).total_seconds() / 3600
        if not start.is_integer() or start < 0:
            raise ValueError('Local initialization must be on the supplied cycle hour lattice; correct its cycle or epoch.')
        snapshot = inspect_local_inputs(source, Path(request.source_root),
            cycle=anchor.replace(tzinfo=None), hours=math.ceil(duration / (3600 * interval)) * interval,
            cadence=interval, start_hour=int(start), supplements=request.supplements)
        kwargs['inventory_times'] = tuple(_stamp(t + 'Z' if '+' not in t and not t.endswith('Z') else t)
                                          for t in snapshot['valid_times'])
        kwargs['supplied_member'] = snapshot.get('manifest', {}).get('member')
        binding = dict(kind='local', root=snapshot['source_root'], snapshot=snapshot,
                       files=_files([row['path'] for row in snapshot['files']]))
    elif facts['local_preparation_operation'] not in ('prepared:go', 'prepared:hrrr', 'prepared:staged') or facts['requires_source_root']:
        raise ValueError('This preparation route needs supplied input authorities; supply source_root with its local handoff or prepared_root with a verified bundle.')
    try:
        selection = plan(source, **kwargs, probe=probe)
    except BackgroundWindowError as exc:
        raise PlanError(str(exc), code='FORCING_HORIZON') from exc
    hints = selection.fetch_hints()
    if binding['kind'] == 'local':
        hints = dict(source=source, cycle=(selection.cycle or selection.init)[:13],
                     hours=math.ceil(duration / (3600 * interval)) * interval,
                     cadence=interval, source_root=binding['root'])
        if selection.cycle and selection.forecast_leads[0]:
            hints['forecast_start_hour'] = selection.forecast_leads[0]
    binding['files'] += _files(binding.partition('=')[2] for binding in request.supplements)
    return dict(schema=SCHEMA, selection=selection.record(), inputs=binding,
                fetch_hints=hints, supplements=list(request.supplements),
                input_time_basis=('required local window; native preparation verifies payload coverage'
                                  if binding['kind'] == 'local' else selection.publication_basis))


def _experiment_identity(exp):
    from woof.core import streaming
    from woof.io.history_selection import resolve
    from woof.runtime import declared_constant_glw
    output = []
    for domain in exp.domains:
        selection = asdict(resolve(exp.output, domain.output))
        selection.pop('source', None)
        output.append(dict(history_interval_s=domain.history_interval_s, selection=selection,
                           tiles=(domain.tiles or exp.tiles or streaming.OFF).to_mapping()))
    return dict(start_time=exp.start_time.isoformat(), run_seconds=exp.run_seconds,
                restart_interval_s=exp.restart_interval_s, output=output,
                constant_glw_wm2=declared_constant_glw(exp),
                domains=[asdict(domain.run) for domain in exp.domains],
                projection=None if exp.projection is None else asdict(exp.projection),
                vertical=asdict(exp.vertical))


def validate_authored_background(background, exp):
    binding = background['inputs']
    if binding['kind'] == 'prepared' and binding['experiment'] != _experiment_identity(exp):
        raise ValueError('The prepared geometry, physics, output or clock differs from the reviewed regional experiment; prepare this region with its reviewed configuration.')


def member_preparation_chain(source):
    """A recipe can also seal a config-driven source through its row's owner."""
    from woof import runplan
    from woof.source_adapters import get_source_adapter
    owner = get_source_adapter(source).ensemble_preparation_owner
    return "prepared:case-data" if owner else runplan.prepared_chain_for_source(source, source_root=None)


def _case_preparation_owner(raw):
    from importlib import import_module
    from woof.source_adapters import get_source_adapter
    return import_module(get_source_adapter(raw["fetch"]["source"]).ensemble_preparation_owner)


def _review_case(config_path, exp, *, raw, scratch, posting):
    return _case_preparation_owner(raw).review(config_path, exp, raw=raw,
                                               scratch=scratch, posting=posting)


def _prepare_case(plan, *, config_path, exp, observer, run_dir, prepare_only):
    import tomllib
    raw = tomllib.loads(Path(config_path).read_text())
    return _case_preparation_owner(raw).prepare(plan, config_path=config_path, exp=exp,
        observer=observer, run_dir=run_dir, prepare_only=prepare_only)


def preparation_chains():
    """Executable preparation-only mechanisms, indexed by shared chain IDs."""
    from woof import runplan
    return {'prepared:go': _prepare_native, 'prepared:hrrr': _prepare_hourly,
            'prepared:staged': runplan._staged_chain, 'prepared:case-data': _prepare_case}


def preparation_chain_reviews():
    """Each chain's own review of a config written for it, before any fetch.

    One row per ID of :func:`preparation_chains`; ``None`` is a chain whose
    plan has nothing to answer before its fetch stage runs.  A door that
    writes a config for a source's chain calls the row as
    ``review(config_path, exp, raw=..., scratch=..., posting=...)``:
    ``raw`` is the config's parsed TOML, ``scratch`` a folder the caller
    discards, and ``posting`` the host pin and posting rule the door
    carries into the chain's fetch (``transport``, ``as_posted``,
    ``late_after_minutes``).  A review raises ``ValueError`` with the
    sentence the chain itself would refuse with.

    The door reads the row instead of testing the chain ID, so a chain is
    added here and in :func:`preparation_chains`, and no door gains a
    branch on a chain or source name.
    """
    return {'prepared:go': _review_native, 'prepared:hrrr': _review_hourly,
            'prepared:staged': None, 'prepared:case-data': _review_case}


def _review_native(config_path, exp, *, raw, scratch, posting):
    """The native chain's own planner: the keys its stages need, the
    preparation preconditions and the profile, planned into ``scratch``."""
    from woof import go_cli
    scratch = Path(scratch)
    go_cli.plan_from_config(Path(config_path), outdir=scratch / 'plan', run_stamp=False,
                            data_dir=scratch / 'data', **posting)


def _review_hourly(config_path, exp, *, raw, scratch, posting):
    """The namelists the hourly chain runs from, asked of the function the
    chain itself calls; the set is rendered into a discarded folder.  The
    host the fetch is pinned to does not change them."""
    from woof.hrrr_route_inputs import run_route_inputs
    run_route_inputs(Path(config_path), exp, raw=raw)


def _prepare_hourly(plan, *, config_path, exp, observer, run_dir, prepare_only):
    from woof.hrrr_route_inputs import write_hrrr_route_inputs
    from woof import runplan
    from woof.prepared_documents import write_document
    namelist = config_path.with_name(config_path.stem + '.namelist.wps')
    write_hrrr_route_inputs(config_path, exp, wps_text=namelist.read_text(encoding='utf-8'),
        writer=lambda path, text: write_document(path, text.encode('utf-8'), reused=path.exists()))
    return runplan._hrrr_chain(plan, config_path=config_path, exp=exp,
                              observer=observer, run_dir=run_dir, prepare_only=prepare_only)


def _prepare_native(plan, *, config_path, exp, observer, run_dir, prepare_only):
    from woof import go_cli, runplan
    # The host pin and posting rule a door carries reach this chain's fetch
    # stage the way run_options reach the other two chains' (their
    # _pinned_fetch_hints); absent, the config's own [fetch] table decides.
    go = go_cli.plan_from_config(config_path, outdir=run_dir, run_stamp=False,
                                  data_dir=Path(plan.run_options['data_dir']),
                                  transport=plan.run_options.get('transport'),
                                  as_posted=plan.run_options.get('as_posted'),
                                  late_after_minutes=plan.run_options.get('late_after_minutes'))
    bridge = go_cli.resolve_bridge()
    go_cli.claim_run_root(go)
    # Regions prepared from one shared download each bind their own manifest.
    manifest = go_cli.front_door_manifest(go)
    for name, command in (('authority', go_cli.authority_command(go)),
                          ('fetch', go_cli.fetch_command(go)),
                          ('manifest', go_cli.manifest_command(go, bridge))):
        if name == 'fetch' and observer is not None:
            # The fetch is a stage on the stream: an ensemble's members that
            # share this download wait for its end (recipe_door._SharedDownloads).
            observer.enter_stage('fetch', phase='fetch')
        go_cli._run_stage(name, command, explain=False)
        if name == 'fetch' and observer is not None:
            # Everything after the download is preparation: an ensemble
            # member takes its host-RAM slot here, once its inputs are on
            # disk (recipe_door._MemberStages).
            observer.enter_stage('prepare', phase='prepare')
    if not manifest.is_file():
        raise MissingPreparationManifest('The preparation route did not publish its input manifest; repair the source fetch before forecasting.')
    command = go_cli.prepare_command(go, bridge, manifest=manifest,
        manifest_sha256=_files([manifest])[0]['sha256'], cycle_stamp=go_cli._cycle_stamp(go['cycle']),
        geog_root=plan.run_options['geog_root'])
    go_cli.announce_policy_backend(command)
    runplan._prepare_stage(go['prepared'], arguments=command, stated={},
                          run=lambda: go_cli._run_stage('prepare', command, explain=False))
    return runplan._prepared_chain_result(go['prepared'], go['authority'] / 'experiment.toml',
                                         go['authority'] / 'namelist.wps')


def validate_saved_background(plan, config_path):
    """Check the saved selection without consulting a later publication clock."""
    from woof.background_contract import from_record, capability_fingerprint
    from woof.source_adapters import get_source_adapter
    from woof.da.background import DEFAULT_BACKGROUND_SOURCE
    import tomllib
    background = plan['background']
    if background.get('schema') != SCHEMA:
        raise ValueError('The saved background contract has another schema; review the requested inputs again.')
    selected = from_record(background['selection'])
    request = plan['request']
    expected_source = get_source_adapter(request.get('source') or DEFAULT_BACKGROUND_SOURCE).source_id
    expected_end = _stamp(request['epoch']) + timedelta(seconds=plan['selected']['cycles'] *
                    plan['selected']['cadence_seconds'] + plan['selected']['forecast_seconds'])
    if selected.source != expected_source or selected.init != _stamp(request['epoch']).isoformat() or selected.end != expected_end.isoformat():
        raise ValueError('The saved background source or window contradicts the reviewed request; restore the original plan.')
    for field, request_field in (('product', 'source_product'), ('member', 'source_member'),
                                 ('provider', 'source_provider')):
        value = request.get(request_field)
        if value is not None and str(getattr(selected, field)) != str(value):
            raise ValueError(f'The saved background {field} contradicts the request; restore its reviewed selection.')
    if request.get('source_cycle') and selected.cycle != _stamp(request['source_cycle']).isoformat():
        raise ValueError('The saved source cycle contradicts the explicitly requested cycle; restore its reviewed selection.')
    if selected.source_contract_sha256 != capability_fingerprint(selected.source):
        raise ValueError('The source preparation or acquisition contract changed after review; review again against the current declared inputs.')
    raw = tomllib.loads(Path(config_path).read_text(encoding='utf-8'))
    if raw.get('fetch') != background['fetch_hints']:
        raise ValueError('The configuration fetch intent contradicts the saved background selection; restore the reviewed configuration.')
    expected = selected.fetch_hints()
    if expected is not None:
        actual = background['fetch_hints']
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError('The authored fetch cycle, member or duration differs from the saved selection; restore its exact intent.')
    verify_files(background['inputs']['files'])
    return selected


def _owned_result(result, root):
    for key in ('prepared_root', 'experiment_config', 'wps_namelist'):
        path = Path(result[key]).resolve()
        if path == root.resolve() or not path.is_relative_to(root.resolve()):
            raise ValueError('The cached preparation receipt names a path outside this case; restore its original owned preparation receipt.')


def _new_generation(root):
    generation = root / 'background'
    index = 0
    while True:
        try:
            generation.mkdir()
            return generation
        except FileExistsError:
            index += 1
            generation = root / f'background-attempt-{index:03d}'


def _refuse_replacing_committed_state(root, error):
    if any((root / 'cycles').rglob('*.npz')):
        raise ValueError('The retained preparation no longer reads under the current contract, and committed regional state depends on it. Preserve this case and publish a new review for a fresh run.') from error


def _prepare_new_background(plan, root, exp, *, geog, selected):
    from woof import runplan
    from woof.prepared_documents import write_document
    background, binding = plan['background'], plan['background']['inputs']
    generation = _new_generation(root)
    chain = runplan.prepared_chain_for_source(selected.source, source_root=binding.get('root'))
    owner = preparation_chains()[chain]
    # Native preparation may author additional authority documents. Work in
    # this case's preparation namespace and preserve the reviewed originals.
    authority = root / 'background-authority'
    authority.mkdir(parents=True, exist_ok=True)
    config = authority / 'experiment.toml'
    namelist = authority / 'experiment.namelist.wps'
    write_document(config, (root / 'experiment.toml').read_bytes(), reused=config.exists())
    write_document(namelist, (root / 'experiment.namelist.wps').read_bytes(), reused=namelist.exists())
    from woof.go_cli import fetch_request, managed_download_dir
    # Keyed on the request the chain's fetch stage makes, which carries the
    # model top this configuration's ladder needs.
    options = dict(geog_root=str(geog), supplement=background['supplements'],
                   data_dir=(binding['root'] if binding['kind'] == 'local' else
                             str(managed_download_dir(root, fetch_request(
                                 background['fetch_hints'], p_top=exp.vertical.p_top)))))
    prepared_plan = SimpleNamespace(run_options=options, config_intent={}, sha256=plan['review_sha256'])
    events = runplan.EventStream(root / 'background-events.jsonl', mirror=None)
    try:
        result = owner(prepared_plan, config_path=config, exp=exp,
                       observer=runplan.RunObserver(events), run_dir=generation, prepare_only=True,
                       **({'reviewed_inputs': binding['snapshot']} if binding['kind'] == 'local' else {}))
    finally:
        events.close()
    result = {key: result[key] for key in ('prepared_root', 'experiment_config', 'wps_namelist')}
    return result


def prepare_background(plan, root, exp, *, geog):
    """Produce or restore one verified bundle before any regional member starts."""
    from woof import runplan
    from woof.ensemble.manifest import write_json_atomically
    from woof.prepared_documents import write_document
    background = plan['background']
    selected = validate_saved_background(plan, root / 'experiment.toml')
    binding = background['inputs']
    receipt_path = root / 'background-preparation.json'
    if binding['kind'] == 'prepared':
        result = dict(prepared_root=binding['root'], experiment_config=binding['config'],
                      wps_namelist=binding['namelist'])
    elif receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('review_sha256') != plan['review_sha256'] or receipt.get('background') != background:
            raise ValueError('The committed preparation belongs to another reviewed background; restore its plan or use a new case directory.')
        result = receipt['result']
    else:
        result = _prepare_new_background(plan, root, exp, geog=geog, selected=selected)
    if binding['kind'] != 'prepared':
        _owned_result(result, root)
    recovery = None
    try:
        if binding['kind'] != 'prepared' and receipt_path.is_file():
            verify_files(receipt['files'])
        inputs = read_prepared(result['prepared_root'], result['experiment_config'], result['wps_namelist'],
            source=selected.source, run_seconds=exp.run_seconds,
            history_interval_seconds=plan['selected']['cadence_seconds'],
            physics_profile=plan['request'].get('profile') if binding['kind'] == 'prepared' else None)
    except (ValueError, OSError) as error:
        if binding['kind'] == 'prepared' or not receipt_path.is_file():
            raise
        # Existing numerical state belongs to the original preparation. A
        # different cache cannot silently replace that analysis context.
        _refuse_replacing_committed_state(root, error)
        old_receipt = receipt_path.read_bytes()
        retained = root / ('background-preparation-' + hashlib.sha256(old_receipt).hexdigest() + '.json')
        write_document(retained, old_receipt, reused=retained.exists())
        result = _prepare_new_background(plan, root, exp, geog=geog, selected=selected)
        _owned_result(result, root)
        inputs = read_prepared(result['prepared_root'], result['experiment_config'], result['wps_namelist'],
            source=selected.source, run_seconds=exp.run_seconds,
            history_interval_seconds=plan['selected']['cadence_seconds'],
            physics_profile=plan['request'].get('profile') if binding['kind'] == 'prepared' else None)
        recovery = dict(previous_receipt=str(retained), reason=str(error), previous_files_preserved=True)
    if selected.cycle is not None and _prepared_cycle(inputs).isoformat() != selected.cycle:
        raise ValueError('The prepared source cycle differs from the reviewed selection; restore the selected cycle inputs.')
    actual_times = tuple((inputs.experiment.start_time.replace(tzinfo=timezone.utc) +
                          timedelta(hours=h)).isoformat() for h in inputs.forcing_hours)
    if any(stamp not in actual_times for stamp in selected.valid_times):
        raise ValueError('Preparation is missing a selected forcing time or final boundary bracket; supply the complete selected window.')
    if _experiment_identity(exp) != _experiment_identity(inputs.experiment):
        raise ValueError('Preparation changed the reviewed geometry, physics, output or clock; restore matching authorities before forecasting.')
    member = _prepared_member(SimpleNamespace(**{key: plan['request'].get(key) for key in
        ('source_product', 'source_member', 'forcing_cadence_hours', 'source_provider', 'source_inputs')}),
        inputs, __import__('woof.source_adapters', fromlist=['get_source_adapter']).get_source_adapter(selected.source))
    if selected.member is not None and str(member) != str(selected.member):
        raise ValueError('Preparation did not retain the selected source member; supply matching member-bound inputs.')
    verify_files(binding['files'])
    if binding['kind'] != 'prepared' and (not receipt_path.exists() or recovery is not None):
        receipt = dict(schema='gpuwm.regional-preparation.v1', review_sha256=plan['review_sha256'],
                       background=background, result=result, recovery=recovery,
                       files=_files([inputs.proof_path, inputs.source_manifest_path,
                                     inputs.experiment_config, inputs.wps_namelist]))
        write_json_atomically(receipt_path, receipt)
    return inputs


def prepare_legacy_background(go, root, exp, *, geog, cadence, review_sha256):
    """Retain an older review's literal fetch intent and recover stale inputs."""
    from woof import go_cli
    from woof.ensemble.manifest import write_json_atomically
    from woof.prepared_documents import write_document
    source = go['source']
    record_path = root / 'background-legacy-preparation.json'
    recipe = _files([root / 'experiment.toml', root / 'experiment.namelist.wps'])
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if (record.get('review_sha256') != review_sha256 or record.get('recipe') != recipe):
            raise ValueError('The retained preparation pointer differs from the original saved review; restore its matching recipe.')
        saved = record['result']
        _owned_result(saved, root)
        go = dict(go, prepared=Path(saved['prepared_root']),
                  authority=Path(saved['experiment_config']).parent)
    def read(result):
        return read_prepared(result['prepared'], result['authority'] / 'experiment.toml',
                             result['authority'] / 'namelist.wps', source=source,
                             run_seconds=exp.run_seconds, history_interval_seconds=cadence)
    if (go['prepared'] / 'proof.json').is_file():
        try:
            return read(go), go
        except (ValueError, OSError) as error:
            _refuse_replacing_committed_state(root, error)
    generation = _new_generation(root)
    resolved = go_cli.plan_from_config(root / 'experiment.toml', outdir=generation,
                                      data_dir=go['data'], run_stamp=False)
    options = SimpleNamespace(run_options=dict(geog_root=str(geog), data_dir=str(go['data'])))
    result = _prepare_native(options, config_path=root / 'experiment.toml', exp=exp,
                             observer=None, run_dir=generation, prepare_only=True)
    result = {key: result[key] for key in ('prepared_root', 'experiment_config', 'wps_namelist')}
    _owned_result(result, root)
    resolved['prepared'] = Path(result['prepared_root'])
    resolved['authority'] = Path(result['experiment_config']).parent
    inputs = read(resolved)
    if record_path.exists():
        previous = record_path.read_bytes()
        retained = root / ('background-legacy-preparation-' + hashlib.sha256(previous).hexdigest() + '.json')
        write_document(retained, previous, reused=retained.exists())
    write_json_atomically(record_path, dict(schema='gpuwm.regional-legacy-preparation.v1',
        review_sha256=review_sha256, recipe=recipe, result=result))
    return inputs, resolved


RENEWAL_SCHEMA = 'arwen.local-da-forcing-renewal.v1'


class ForcingRenewal:
    """One renewed preparation: the same initial state under longer forcing."""
    def __init__(self, inputs, receipt_path, receipt):
        self.inputs = inputs
        self.receipt_path = Path(receipt_path)
        self.receipt = receipt
        self.generation = receipt['generation']
        start = inputs.experiment.start_time.replace(tzinfo=timezone.utc)
        self.forcing_times = tuple(start + timedelta(hours=h) for h in inputs.forcing_hours)

    @property
    def prepared_cache_path(self):
        return self.inputs.prepared_cache_path

    @property
    def cache_identity(self):
        return self.inputs.cache_identity

    @property
    def forcing_hours(self):
        return self.inputs.forcing_hours


def _renewal_identity(exp):
    """The reviewed experiment with its window length taken out.

    The length is stated twice, on the experiment and again on every
    domain's run configuration, and a renewal changes exactly those two.
    """
    value = _experiment_identity(exp)
    value.pop('run_seconds')
    for domain in value['domains']:
        domain.pop('run_seconds', None)
    return value


def _check_renewal(original, kept_hours, inputs):
    """A renewal keeps the initial state and every frame it has run under."""
    if _renewal_identity(original.experiment) != _renewal_identity(inputs.experiment):
        raise ValueError('The renewed preparation changed the reviewed geometry, physics, output or clock; the analysed state cannot continue under it. Restore the reviewed configuration before renewing.')
    if original.proof.get('initial_condition') != inputs.proof.get('initial_condition'):
        raise ValueError('The renewed preparation initialised from another source cycle or lead; the analysed state belongs to the original initial condition. Renew from the same cycle.')
    kept = tuple(kept_hours)
    renewed = tuple(inputs.forcing_hours)
    if renewed[:len(kept)] != kept or len(renewed) <= len(kept):
        raise ValueError(f'The renewed forcing must keep the frames already run under ({list(kept)} h) and append later ones; it carries {list(renewed)} h. Renew over a longer window from the same source cycle.')


def renew_background(plan, root, original, *, previous, end_time, directory, geog, now=None, probe=None):
    """Prepare the reviewed source cycle over a window reaching ``end_time``.

    Nothing already run under is replaced: the source, its cycle, the
    initial condition and every forcing frame of ``previous`` (or of the
    original preparation) are kept, and later frames of the same cycle are
    appended. Frames the source has not published raise
    :class:`woof.background_contract.BackgroundWindowError`, which a caller
    may wait on; a reviewed input kind that cannot be extended is refused
    by name.
    """
    import tomllib
    from woof import runplan
    from woof.background_contract import from_record, plan as select
    from woof.ensemble.manifest import write_json_atomically
    from woof.experiment import load_experiment
    from woof.go_cli import fetch_request, managed_download_dir
    from woof.prepared_documents import write_document
    from woof.starter_template import render_tables
    background = plan.get('background')
    if background is None:
        raise ValueError('This saved review predates the background contract, so its forcing cannot be renewed; publish a new review to cycle continuously.')
    binding = background['inputs']
    if binding['kind'] != 'automatic':
        raise ValueError(f"Forcing renewal runs the automatic preparation chain, and this review binds {binding['kind']} inputs; supply inputs that already cover the whole continuous window, or review on an automatically prepared source.")
    selected = from_record(background['selection'])
    init = _stamp(plan['request']['epoch'])
    end_time = _stamp(end_time) if isinstance(end_time, str) else end_time.astimezone(timezone.utc)
    duration = (end_time - init).total_seconds()
    base = previous.inputs if previous is not None else original
    request = plan['request']
    renewed = select(selected.source, init=init, now=now or datetime.now(timezone.utc), run_seconds=duration,
        product=request.get('source_product'), member=selected.member, provider=request.get('source_provider'),
        cadence_hours=selected.forcing_interval_seconds // 3600,
        cycle=None if selected.cycle is None else _stamp(selected.cycle), probe=probe)
    if renewed.cycle != selected.cycle or renewed.init != selected.init or renewed.source != selected.source:
        raise ValueError('The renewal selected another source cycle or initial time than the review; the analysed state belongs to the reviewed cycle.')
    hints = dict(renewed.fetch_hints())
    for key in ('out', 'area'):
        if key in background['fetch_hints']:
            hints[key] = background['fetch_hints'][key]
    directory = Path(directory)
    generation = 1 if previous is None else previous.generation + 1
    authority = directory / 'authority'
    authority.mkdir(parents=True, exist_ok=True)
    raw = tomllib.loads((root / 'experiment.toml').read_text(encoding='utf-8'))
    raw['experiment']['run_seconds'] = duration
    raw['fetch'] = hints
    config = authority / 'experiment.toml'
    namelist = authority / 'experiment.namelist.wps'
    write_document(config, ('# Renewed forcing window.\n' + render_tables(raw)).encode('utf-8'), reused=config.exists())
    write_document(namelist, (root / 'experiment.namelist.wps').read_bytes(), reused=namelist.exists())
    exp = load_experiment(config)
    chain = runplan.prepared_chain_for_source(selected.source)
    owner = preparation_chains()[chain]
    options = dict(geog_root=str(geog), supplement=background['supplements'],
                   data_dir=str(managed_download_dir(root, fetch_request(
                       hints, p_top=exp.vertical.p_top))))
    prepared_plan = SimpleNamespace(run_options=options, config_intent={}, sha256=plan['review_sha256'])
    attempt = 0
    while True:
        run_dir = directory / ('generation' if attempt == 0 else f'generation-attempt-{attempt:03d}')
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            attempt += 1
    events = runplan.EventStream(directory / 'events.jsonl', mirror=None)
    try:
        result = owner(prepared_plan, config_path=config, exp=exp, observer=runplan.RunObserver(events),
                       run_dir=run_dir, prepare_only=True)
    finally:
        events.close()
    result = {key: result[key] for key in ('prepared_root', 'experiment_config', 'wps_namelist')}
    _owned_result(result, root)
    inputs = read_prepared(result['prepared_root'], result['experiment_config'], result['wps_namelist'],
        source=selected.source, run_seconds=duration, history_interval_seconds=plan['selected']['cadence_seconds'])
    _check_renewal(original, base.forcing_hours, inputs)
    receipt = dict(schema=RENEWAL_SCHEMA, review_sha256=plan['review_sha256'], generation=generation,
        selection=renewed.record(), fetch_hints=hints, end_time=end_time.isoformat(), run_seconds=duration,
        previous_receipt=None if previous is None else str(previous.receipt_path),
        kept_forcing_hours=list(base.forcing_hours), forcing_hours=list(inputs.forcing_hours),
        result=result, files=_files([inputs.proof_path, inputs.source_manifest_path,
                                     inputs.experiment_config, inputs.wps_namelist]))
    receipt_path = directory / 'renewal.json'
    write_json_atomically(receipt_path, receipt)
    return ForcingRenewal(inputs, receipt_path, receipt)


def read_background_renewal(receipt_path, *, plan, root, original):
    """Reopen a committed renewal, verifying its bytes before anything reads them."""
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt.get('schema') != RENEWAL_SCHEMA or receipt.get('review_sha256') != plan['review_sha256']:
        raise ValueError(f'{receipt_path} is not a forcing renewal of this saved review; restore the original renewal receipt.')
    verify_files(receipt['files'])
    result = receipt['result']
    _owned_result(result, Path(root))
    inputs = read_prepared(result['prepared_root'], result['experiment_config'], result['wps_namelist'],
        source=receipt['selection']['source'], run_seconds=receipt['run_seconds'],
        history_interval_seconds=plan['selected']['cadence_seconds'])
    if list(inputs.forcing_hours) != receipt['forcing_hours']:
        raise ValueError(f'{receipt_path}: the renewed preparation no longer carries the forcing hours it was committed with; restore the original preparation.')
    _check_renewal(original, receipt['kept_forcing_hours'], inputs)
    return ForcingRenewal(inputs, receipt_path, receipt)
