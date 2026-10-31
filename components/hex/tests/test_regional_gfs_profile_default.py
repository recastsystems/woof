"""Scoped promotion and the native request boundary, captured before promotion."""
from datetime import datetime, timedelta
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from woof.hex import physics_default as defaults, init_physics
from woof.hex.cli import build_parser
from woof.hex.errors import MpasPortError
from woof.hex.mpasphys.forecast_v842 import native_route

GEOMETRIES = ('zoom-globe', 'limited-area', 'storm-following')

def context(geometry='zoom-globe', source='gfs', start='cold'):
    return dict(geometry=geometry, source=source, start=start)

def resolve(flags):
    args = build_parser().parse_args(['forecast', '--mesh', 'x1.40962', '--hours', '1', *flags])
    return native_route(args, default_mode='correct')

@pytest.mark.parametrize('geometry', GEOMETRIES)
@pytest.mark.parametrize('source', ['gfs', 'gfs-gdex'])
def test_regional_cold_rows_select_aerosols_with_native_mynn(geometry, source):
    ctx = context(geometry, source)
    flags = defaults.ensure([], context=ctx)
    config, _ = resolve(flags)
    base, _ = resolve(['--physics-suite=convection_permitting', '--physics-mode=correct'])
    assert config.selectors['microphysics'] == 'mp_thompson_aerosols'
    assert config.options['config_hex_icloud_bl'] is True
    assert config.receipt()['options']['hex_physics_profile'] == 'hrrr-like-aerosol'
    assert {k:v for k,v in config.options.items() if 'mynn' in k} == {k:v for k,v in base.options.items() if 'mynn' in k}
    assert defaults.record(context=ctx)['profile'] == 'hrrr-like-aerosol'
    assert defaults.init_flags(flags) == ['--physics-profile', 'hrrr-like-aerosol']

@pytest.mark.parametrize('ctx', [None, context('global'), context(source='ifs'), context(start='warm'),
                               {'geometry':'zoom-globe', 'source':'gfs'}])
def test_unproven_contexts_retain_native_default(ctx):
    assert defaults.ensure([], context=ctx) == ['--physics-suite=convection_permitting']

@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_explicit_exact_and_native_requests_match_pre_promotion_bytes(geometry):
    fixture = json.loads((Path(__file__).parent/'fixtures/regional-default-native.json').read_text())
    assert fixture['source_commit'] == 'f308a001016a479d16c4a881220d2bece2e558e5'
    for case in fixture['cases']:
        flags = defaults.ensure(case['request'], context=context(geometry))
        assert flags == case['argv']
        config, text = resolve(flags)
        assert text.encode() == case['namelist'].encode()
        assert json.dumps(config.receipt(), sort_keys=True).encode() == json.dumps(case['receipt'], sort_keys=True).encode()
        assert defaults.init_flags(flags) == []
        assert 'hex_physics_profile' not in config.options


def test_arbitrary_recipe_row_needs_no_source_code(monkeypatch):
    table = copy.deepcopy(defaults.load_table())
    row = copy.deepcopy(table['recipe_rules'][0])
    row.update(id='another-recipe', profile='hrrr-like')
    row['match'] = dict(source='other-provider', geometry='other-shape', start='other-state')
    table['recipe_rules'].insert(0, row)
    monkeypatch.setattr(defaults, 'load_table', lambda:table)
    assert defaults.ensure([], context=row['match'])[-1] == '--physics-profile=hrrr-like'


def test_profile_init_uses_declared_carriers_and_requires_monthly_wif():
    selected = init_physics.init_selection(None, None, profile='hrrr-like-aerosol')
    assert selected['scalar_names'] == ('nc', 'nifa', 'nwfa')
    assert selected['requires_climatology'] and selected['force_climatology']
    assert selected['land_surface'] == 'noah'
    assert init_physics.init_selection('convection_permitting', None)['scalar_names'] == ()
    with pytest.raises(MpasPortError, match='differ'):
        init_physics.init_selection('mesoscale_reference', None, profile='hrrr-like-aerosol')


def test_monthly_requirement_cannot_accept_analysis_or_synthetic_fallback(tmp_path, monkeypatch):
    import woof.ingest.wif_climatology as wif
    import woof.table_assets as assets
    selected = init_physics.init_selection(None, None, profile='hrrr-like-aerosol')
    monkeypatch.setattr(init_physics, 'aerosol_status', lambda *a:{'analysis_pair':True})
    monkeypatch.setattr(wif, 'resolve_wif_climatology', lambda *a:SimpleNamespace(resolved=False))
    monkeypatch.setattr(wif, 'describe_wif_source', lambda *a:{'resolved':False})
    monkeypatch.setattr(assets, 'stage_wif_dataset', lambda:0)
    with pytest.raises(ValueError, match='monthly WIF'):
        init_physics.initialize_species('unused', tmp_path/'init.nc', selected,
                                       SimpleNamespace(start_time='2023-07-08_12:00:00'))


def test_regional_plan_and_init_share_profile(tmp_path):
    from woof.hex import regional_door
    args = build_parser().parse_args(['regional', '--source', 'gfs', '--point', '36.5,-97.5', '--cycle', '2024-05-06T18',
                                      '--hours', '6', '--work-dir', str(tmp_path)])
    plan = regional_door.plan(args)
    assert plan['physics']['profile'] == 'hrrr-like-aerosol'
    assert regional_door._init_suite_argv(plan) == ['--physics-profile', 'hrrr-like-aerosol']
    args.forecast_arg = ['--physics-mode=exact']
    plan = regional_door.plan(args)
    assert plan['forecast_args'] == ['--physics-mode=exact', '--physics-suite=convection_permitting']
    assert regional_door._init_suite_argv(plan) == []


def test_moving_late_exact_override_discards_only_implicit_profile(tmp_path):
    from woof.hex.moving_case import load_case, resolve_physics
    from woof.hex.moving_run import segment_command
    case = dict(start_time='2023-07-08_12:00:00', dt_seconds=18, n_levels=55, vertical_spec='tc55-v1.json',
                storm={'initial_center_deg':[44,-73]}, mesh=dict(background_km=48, fine_km=3,
                fine_radius_km=300, fine_transition_km=54, rings=[]))
    path=tmp_path/'case.json';path.write_text(json.dumps(case))
    loaded = load_case(path)
    assert '--physics-profile=hrrr-like-aerosol' in loaded['physics_args']
    assert resolve_physics(loaded, ['--physics-mode=exact']) == ['--physics-mode=exact','--physics-suite=convection_permitting']
    assert resolve_physics(loaded, source='ifs') == ['--physics-suite=convection_permitting']
    member=dict(mesh='x', grid='grid', static='static', init='init', dt_seconds=18)
    argv=segment_command(loaded, member, tmp_path/'seg', 1, 60, init_source='gfs', extra=['--physics-mode=exact'])
    assert not any('--physics-profile' in s for s in argv)
    assert '--physics-mode=exact' in argv


def test_regional_boundary_hours_do_not_request_analysis_soil_donor(tmp_path, monkeypatch):
    from woof.hex import regional_door as door
    args = build_parser().parse_args(['regional','--source','gfs','--point','36.5,-97.5',
        '--cycle','2024-05-06T18','--hours','6','--work-dir',str(tmp_path)])
    plan=door.plan(args)
    captured=[]
    def run(argv, **kwargs):
        lead=int(argv[argv.index('--lead-hours')+1]);captured.append(argv)
        out=Path(argv[argv.index('--out-dir')+1])
        # Each lead writes its own valid time (the door now takes every lead GFS posts, hourly, so leads 1-6 must not
        # share one file name: they are linked into one met directory concurrently).
        valid=datetime(2024,5,6,18)+timedelta(hours=lead)
        (out/f'MET:{valid:%Y-%m-%d_%H}').write_bytes(b'met')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(door.subprocess,'run',run)
    assert door._stage_intermediate_from_table(args,plan,SimpleNamespace(note=lambda *a:None))['returncode']==0
    for argv in captured:
        lead=int(argv[argv.index('--lead-hours')+1])
        assert ('--soil-remap' in argv)==(lead!=0)
        if lead:assert argv[argv.index('--soil-remap')+1]=='off'


def test_regional_init_binds_profile_and_verified_native_donor(tmp_path, monkeypatch):
    from woof.hex import regional_door as door, init_sources
    args=build_parser().parse_args(['regional','--source','gfs','--point','36.5,-97.5',
        '--cycle','2024-05-06T18','--hours','6','--work-dir',str(tmp_path)])
    value=door.plan(args)
    donor=tmp_path/'native.nc';donor.write_bytes(b'donor')
    decoded=Path(value['paths']['met'])/'leads/f000/init-source-receipt.json'
    decoded.parent.mkdir(parents=True)
    policy=init_sources.resolve_source(init_sources.load_table(),'gfs')['cold_soil']
    decoded.write_text(json.dumps({'products':[dict(role=policy['product_role'],file=str(donor),sha256=init_sources._sha256_file(donor))]}))
    monkeypatch.setattr(door,'_generated',lambda value:{'vertical':{'cull_artifact':'vertical'},'cull':{'static':'static'}})
    monkeypatch.setattr(door,'_met_file',lambda value:tmp_path/'MET')
    seen=[];monkeypatch.setattr(door,'_cli',lambda argv,status:seen.append(argv) or 0)
    assert door._stage_init(args,value,SimpleNamespace())['returncode']==0
    argv=seen[0]
    assert argv[argv.index('--physics-profile')+1]=='hrrr-like-aerosol'
    assert argv[argv.index('--soil-donor')+1]==str(donor)
    # A donor whose bytes changed after acquisition is never used: the init starts from the source's own soil
    # (--soil-remap off, as before the remap) instead of being refused (WOOF 0.4.10 switch candidate, 2026-10-31).
    donor.write_bytes(b'changed')
    seen.clear()
    assert door._stage_init(args,value,SimpleNamespace())['returncode']==0
    argv=seen[0]
    assert '--soil-donor' not in argv and argv[argv.index('--soil-remap')+1]=='off'
    with pytest.raises(MpasPortError, match='differs'):
        init_sources.cold_soil_init_flags(json.loads(decoded.read_text())['products'],policy,'on')


def test_cached_cold_init_needs_verified_profile_receipt(tmp_path):
    init=tmp_path/'init.nc'
    flags=defaults.ensure([],context=context())
    assert init_physics.profile_init_matches(init,['--physics-suite=convection_permitting'])
    assert not init_physics.profile_init_matches(init,flags)
    receipt=init.with_name(init.name+'.provenance.json')
    for verified in (False,True):
        receipt.write_text(json.dumps({'physics_initialization':dict(profile='hrrr-like-aerosol',
            resolution=dict(matches_wrf_471_reference=verified))}))
        assert init_physics.profile_init_matches(init,flags)==verified
    assert not init_physics.profile_init_matches(init,['--physics-profile=hrrr-like'])


def test_moving_staging_keeps_soil_donor_out_of_meteorology_decode(tmp_path, monkeypatch):
    from woof.hex import moving_members, init_sources as src
    monkeypatch.setattr(moving_members,'load_case',lambda _:dict(start_time='2023-07-08_12:00:00',_path=str(tmp_path/'case.json')))
    monkeypatch.setattr(src,'resolve_writer',lambda *a:tmp_path/'writer')
    monkeypatch.setattr(src,'resolve_vtable',lambda *a:tmp_path/'vtable')
    def acquire(plans,*a,**kw):
        products=[]
        for plan in plans:
            p=tmp_path/plan.filename;p.write_bytes(plan.role.encode())
            products.append(dict(role=plan.role,file=str(p),sha256=src._sha256_file(p)))
        return products
    captured=[]
    def decode(row,inputs,*args):
        assert inputs
        assert all(Path(p).read_bytes()!=b'cold-soil-donor' for p in inputs)
        captured.extend(inputs)
        return dict(met=str(tmp_path/'MET'))
    monkeypatch.setattr(src,'acquire',acquire)
    monkeypatch.setattr(src,'decode',decode)
    stage=moving_members.stage_case(tmp_path/'case.json',tmp_path/'stage',log=lambda *a:None)
    row=src.resolve_source(src.load_table(),stage['source'])
    flags=src.cold_soil_init_flags(stage['products'],row['cold_soil'])
    donor=flags[flags.index('--soil-donor')+1]
    assert Path(donor).read_bytes()==b'cold-soil-donor' and donor not in captured


@pytest.mark.parametrize('source_id, promoted', [('gfs', True), ('ifs', False)])
def test_moving_source_identity_is_separate_from_human_label(tmp_path, source_id, promoted):
    from woof.hex.moving_run import segment_command
    case=dict(start_time='2023-07-08_12:00:00', mesh={'fine_km':3}, physics_args=[])
    member=dict(mesh='x',grid='grid',static='static',init='init',dt_seconds=18)
    argv=segment_command(case,member,tmp_path/'seg',1,60,init_source='Human-readable label',source_id=source_id)
    assert ('--physics-profile=hrrr-like-aerosol' in argv)==promoted
