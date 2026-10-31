"""Per-run nudging selection for the ordinary forecast command."""

def preparation_data_directory(args, payload, case):
    from pathlib import Path
    from woof.go_cli import managed_download_dir, config_fetch_request
    return (Path(args.data_dir) if args.data_dir is not None else
            managed_download_dir(case, config_fetch_request(payload)))

def at_flag(args, config, payload):
    mode = getattr(args, 'nudging', None)
    if mode is None:
        return config, payload
    from copy import deepcopy
    from woof.go_cli import GoRefusal, _prepared_launch
    from woof.companion_setups import _resolve_declared_paths
    from woof.case_data import config_fetch_out
    from woof.core.preflight import config_forcing_source
    from woof.smoke_execution import _restate
    if _prepared_launch(args):
        from woof.experiment import load_experiment
        prepared = load_experiment(config)
        options = prepared.nudging
        root_control = None if options is None else options.for_domain(prepared.root.grid_id)
        if (mode == 'off' and options is None) or (options is not None and
                root_control is not None and root_control.mode == mode):
            return config,payload
        raise GoRefusal('NUDGING_PREPARED_IDENTITY: --nudging must prepare its analysis store; run it with the source configuration, then resume with the resulting unchanged configuration')
    source = config_forcing_source(config, priced_only=False)
    raw = _resolve_declared_paths(deepcopy(payload), base_dir=config.parent, source=source)
    if raw.get('fetch'):
        raw['fetch']['out'] = str(config_fetch_out(payload, base_dir=config.parent))
    row = raw.pop('spectral_nudging', raw.get('nudging', {}))
    if mode == 'off':
        raw.pop('nudging', None)
    else:
        row.update(enabled=True, mode=mode)
        row.setdefault('boundary_cells', int(raw.get('experiment',{}).get('spec_bdy_width',5)))
        if mode == 'grid':
            row['convention'] = 'wrf'
        raw['nudging'] = row
    out, loaded = _restate(args,config,payload,raw,'nudging-'+mode,source=source)
    print(f'go: nudging {mode}: {out}', flush=True)
    return out,loaded


def source_preparation(args, config, payload):
    """Use a source row's existing native preparation owner on ordinary go.

    This is the same owner used by the ensemble door. Sources without this
    hook keep their ordinary prepared chain; adding a source is table work.
    """
    if not (payload.get('nudging') or payload.get('spectral_nudging') or
            getattr(args,'nudging',None) == 'off'):
        return None
    from woof.go_cli import _prepared_launch
    if _prepared_launch(args):
        return None
    from woof.source_adapters import get_source_adapter
    source = payload.get('fetch',{}).get('source')
    if source is None:
        return None
    adapter = get_source_adapter(source)
    owner = getattr(adapter,'ensemble_preparation_owner',None)
    if not owner:
        return None
    from copy import deepcopy
    from importlib import import_module
    from pathlib import Path
    from types import SimpleNamespace
    import hashlib
    import json
    import tomllib
    from woof import runplan
    from woof.case_data import config_fetch_out
    from woof.experiment import load_experiment
    from woof.geog_assets import config_geog_root
    from woof.go_cli import _registered_launch, geography_refusal, GoRefusal
    from woof.toml_document import emit_experiment_toml
    module = import_module(owner)
    geog = Path(args.geog_root) if args.geog_root is not None else config_geog_root(config)
    refusal = geography_refusal(geog,config=config)
    if refusal:
        raise GoRefusal(refusal)
    identity = hashlib.sha256(config.read_bytes()).hexdigest()
    case = Path(args.outdir) if args.outdir is not None else config.parent/(config.stem+'-go')
    root = case/'preparation'/identity[:16]
    root.mkdir(parents=True,exist_ok=True)
    target = root/'source.toml'
    raw = deepcopy(payload)
    module.configure_member(raw,target=target,original=config)
    raw['case_data']['geog_root'] = str(geog)
    target.write_text(emit_experiment_toml(raw))
    from woof.companion_domains import _wps_text
    exp = load_experiment(target)
    target.with_suffix('.namelist.wps').write_text(_wps_text(exp,None,target.with_suffix('.namelist.wps'),raw,len(exp.domains),original_domain_ids=[d.grid_id for d in exp.domains]))
    # Match ordinary go's request-owned cache. A changed cycle or window must
    # not collide with another request's flat [fetch].out directory.
    data = preparation_data_directory(args, payload, case)
    plan = SimpleNamespace(run_options={'data_dir':str(data),'geog_root':str(geog),'experiment_config_suite':True},config_intent={},sha256=identity)
    events = runplan.EventStream(root/'preparation-events.jsonl',mirror=None)
    observer = runplan.RunObserver(events)
    print('go: preparing source analyses',flush=True)
    try:
        result = module.prepare(plan,config_path=target,exp=exp,observer=observer,run_dir=root,prepare_only=True)
    finally:
        observer.close()
        events.close()
    (root/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    if getattr(args,'prepare_only',False):
        print(json.dumps({'status':'PREPARED',**result},default=str),flush=True)
        return 0
    args.prepared_root = Path(result['prepared_root'])
    args.config = Path(result['experiment_config'])
    args.nudging = None
    return _registered_launch(args,config=args.config,payload=tomllib.loads(args.config.read_text()))
