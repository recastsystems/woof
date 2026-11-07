"""CPU terrain policy, WPS arithmetic, route and front-door contracts."""
from dataclasses import replace
from datetime import datetime, date
from pathlib import Path
from types import SimpleNamespace
import json
import tomllib
import numpy as np
import pytest
from woof.static.terrain_smoothing import (
    TerrainSmoothing, WPS_DEFAULT, WPS_EXACT_DEFAULT, SMOOTH_OPTIONS,
    SMOOTH_PRECISIONS, parse_domain_static, static_inline, with_precision,
    parse_smoothing_spec, geogrid_tbl_smoothing, smooth_terrain_reference,
    smooth_terrain, resolve_domain_smoothing, smoothing_for, emit_smoothing,
    catalog_with_smoothing, require_root_smoothing)
from woof.static import build, rust_bridge
from woof.static.highres_production import HighresStaticConfig, resolve_static_highres, apply_prepared_highres

SETTINGS = [TerrainSmoothing('none'), TerrainSmoothing('1-2-1', 1),
            TerrainSmoothing('1-2-1', 3), TerrainSmoothing('1-2-1', 5),
            TerrainSmoothing('smth-desmth', 1), TerrainSmoothing('smth-desmth', 4),
            TerrainSmoothing('smth-desmth_special', 2), TerrainSmoothing('smth-desmth_special', 5)]

def plane():
    a = np.random.default_rng(27).uniform(-100, 1500, (11, 13))
    a[4:7, 5:8] = 0
    return a

def bits(a, b):
    assert a.dtype == b.dtype == np.float64
    assert a.shape == b.shape
    assert a.tobytes() == b.tobytes()

def fortran_loop(extended, setting):
    if setting.option == 'none':
        return extended.copy()
    a = extended.astype(np.float32)
    original = a.copy()
    ny, nx = a.shape
    for _ in range(setting.passes):
        for desmooth in range(1 if setting.option == '1-2-1' else 2):
            c1 = np.float32(1.52 if desmooth else .5)
            c2 = np.float32(.26 if desmooth else .25)
            s = a.copy()
            for j in range(ny):
                for i in range(1, nx-1):
                    t = np.float32(c1*a[j,i])
                    u = np.float32(a[j,i-1]+a[j,i+1])
                    v = np.float32(c2*u)
                    s[j,i] = np.float32(t-v if desmooth else t+v)
            for j in range(1, ny-1):
                for i in range(1, nx-1):
                    t = np.float32(c1*s[j,i])
                    u = np.float32(s[j-1,i]+s[j+1,i])
                    v = np.float32(c2*u)
                    a[j,i] = np.float32(t-v if desmooth else t+v)
    if setting.option == 'smth-desmth_special':
        for j in range(ny):
            for i in range(nx):
                if a[j,i] < 0 and original[j,i] >= 0:
                    a[j,i] = original[j,i]
    return a.astype(np.float64)

@pytest.mark.parametrize('setting', SETTINGS)
def test_reference_matches_fortran_and_freezes_border(setting):
    a = plane()
    got = smooth_terrain_reference(a, setting)
    bits(got, fortran_loop(a, setting))
    orig = a if setting.option == 'none' else a.astype('f4').astype('f8')
    bits(got[[0,-1],:], orig[[0,-1],:])
    bits(got[:,[0,-1]], orig[:,[0,-1]])

def test_default_keeps_legacy_arithmetic(monkeypatch):
    a = plane()
    expected = build.smth_desmth_special(a, passes=1)
    bits(smooth_terrain_reference(a, WPS_DEFAULT), expected)
    monkeypatch.setattr(rust_bridge, 'route', lambda op: pytest.fail('default overlay must keep legacy call'))
    bits(smooth_terrain(a, TerrainSmoothing('smth-desmth_special',1)), expected)
    assert not np.array_equal(expected, fortran_loop(a, WPS_DEFAULT))

def test_special_restores_after_all_passes():
    a = np.zeros((11,13))
    a[5,6] = 1000
    plain = smooth_terrain_reference(a, TerrainSmoothing('smth-desmth',5))
    special = smooth_terrain_reference(a, TerrainSmoothing('smth-desmth_special',5))
    restore = (plain < 0) & (a >= 0)
    assert restore.any()
    bits(special[restore], a.astype('f4').astype('f8')[restore])
    bits(special[~restore], plain[~restore])

@pytest.mark.parametrize('setting', SETTINGS + [WPS_DEFAULT, WPS_EXACT_DEFAULT])
def test_rust_entry_matches_reference(setting):
    reason = rust_bridge.unavailable_reason()
    if reason:
        pytest.skip(reason)
    bits(rust_bridge.terrain_smooth(plane(),setting),smooth_terrain_reference(plane(),setting))

def test_diagnostic_route(monkeypatch, capsys):
    monkeypatch.setenv('WOOF_STATIC_PYTHON','1')
    rust_bridge._REPORTED_FALLBACKS.discard('terrain_smooth')
    setting=TerrainSmoothing('1-2-1',3)
    bits(smooth_terrain(plane(),setting),fortran_loop(plane(),setting))
    assert 'WORKAROUND' in capsys.readouterr().out

def test_old_library_missing_symbol_refuses_only_new_option(monkeypatch):
    monkeypatch.setattr(rust_bridge,'load',lambda:SimpleNamespace())
    with pytest.raises(rust_bridge.StaticBridgeError,match='predates terrain-smoothing options.*rebuild'):
        rust_bridge.terrain_smooth(plane(),TerrainSmoothing('none'))
    bits(smooth_terrain(plane(),WPS_DEFAULT),build.smth_desmth_special(plane(),1))

@pytest.mark.parametrize('text,want',[
 ('name=HGT_M\nsmooth_option=smth-desmth_special',WPS_DEFAULT),
 ('name=HGT_M',TerrainSmoothing('none')),
 ('name=HGT_M\n#smooth_option=smth-desmth_special',TerrainSmoothing('none')),
 ('name=HGT_M;smooth_option=1-2-1;smooth_passes=0',TerrainSmoothing('none')),
 ('name=HGT_M;smooth_option=1-2-1;smooth_passes=-2',TerrainSmoothing('none')),
 ('name = HGT_M ; smooth_option = 1-2-1 ; smooth_passes = 3 # comment',TerrainSmoothing('1-2-1',3)),
 ('name=HGT_M;smooth_option=unknown',TerrainSmoothing('none')),
 ('name=HGT_M;smooth_option=unknown\n=====\nname=HGT_M;smooth_option=1-2-1',TerrainSmoothing('1-2-1')),
 ('name=HGT_M;smooth_option=smth-desmth_special\n=====\nname=HGT_M;smooth_option=1-2-1',WPS_DEFAULT),
 ('name=HGT_M;smooth=1-2-1;passes=3',TerrainSmoothing('1-2-1',3)),
 ('name=HGT_M;option=1-2-1',TerrainSmoothing('none')),
 ('name=HGT_M;smooth_option=1-2-1;smooth_option=unknown',TerrainSmoothing('1-2-1')),
 ('name=HGT_M;smooth_option=1-2-1\n#=====\nsmooth_passes=3',TerrainSmoothing('1-2-1',3)),
 ('name=HGT_M;smooth_option=smth-desmth_special;smooth_precision=wps-float32',WPS_EXACT_DEFAULT),
 ('name = HGT_M\n smooth_option = smth-desmth_special; smooth_passes=1\n smooth_precision = float64',WPS_DEFAULT),
 ('name=HGT_M;smooth_option=1-2-1;smooth_passes=2;smooth_precision=wps-float32',TerrainSmoothing('1-2-1',2)),
 ('name=HGT_M;smooth_option=unknown;smooth_precision=float64\n=====\nname=HGT_M;smooth_option=smth-desmth_special;smooth_precision=wps-float32',WPS_EXACT_DEFAULT),
])
def test_geogrid_parser(tmp_path,text,want):
    p=tmp_path/'GEOGRID.TBL';p.write_text(text)
    assert geogrid_tbl_smoothing(p)==want
    assert geogrid_tbl_smoothing(tmp_path)==want

@pytest.mark.parametrize('table,match',[
 ({},'requires smooth_option'), ({'smooth_option':'none','smooth_passes':1},'ignored'),
 ({'smooth_option':'1-2-1','smooth_passes':0},'write smooth_option'),
 ({'smooth_option':'1-2-1','smooth_passes':True},'integer'),
 ({'smooth_option':'1-2-1','smooth_passes':1.0},'integer'),
 ({'smooth_option':'bad'},'unknown terrain'),
 ({'smooth_option':'none','smooth_pases':2},'unknown keys'),
 ([], 'must be a table')])
def test_domain_static_refusals(table,match):
    with pytest.raises(ValueError,match=match):
        parse_domain_static(table,source='synthetic',grid_id=3)

@pytest.mark.parametrize('option',SMOOTH_OPTIONS)
def test_domain_static_valid(option):
    assert parse_domain_static({'smooth_option':option},source='synthetic',grid_id=1)==TerrainSmoothing(option)

def carrier(tmp_path):
    return HighresStaticConfig(False,tmp_path,terrain_smoothing=((3,'none',0),(7,'1-2-1',3)))

def test_resolution_preserves_default_identity(tmp_path):
    args=dict(source='synthetic',base_dir=tmp_path,spacings_m=[12000])
    original=resolve_static_highres({},**args)
    assert original is None
    assert resolve_static_highres({'domain':[{'grid_id':3,'static':WPS_DEFAULT.echo()}]},**args) is None
    raw={'domain':[{'grid_id':7,'static':{'smooth_option':'1-2-1','smooth_passes':3}},
                   {'grid_id':3,'static':{'smooth_option':'none'}}]}
    cfg=resolve_static_highres(raw,**args)
    assert not cfg.enabled
    assert cfg.terrain_smoothing==carrier(tmp_path).terrain_smoothing
    assert cfg.smoothing_for(7)==TerrainSmoothing('1-2-1',3)
    assert cfg.smoothing_for(1)==WPS_DEFAULT
    old=HighresStaticConfig(False,tmp_path)
    assert 'terrain_smoothing' not in old.echo()
    assert cfg.echo()['terrain_smoothing']==[[3,'none',0],[7,'1-2-1',3]]
    assert resolve_domain_smoothing({},old,source='synthetic') is old

def test_selections_and_inner_catalog_view(tmp_path):
    from woof.ingest.nest_init import _static_catalog
    wps=tmp_path/'namelist.wps'
    wps.write_text('&share\nmax_dom=1,\n/\n&geogrid\ngeog_data_res="default",\n/\n')
    cfg=HighresStaticConfig(False,tmp_path,terrain_smoothing=((1,'none',0),))
    data=SimpleNamespace(wps_namelist=wps,geog_root=tmp_path,static_highres=cfg)
    assert build.GeogSelection.from_case_data(data,1).terrain_smoothing==TerrainSmoothing('none')
    inner=SimpleNamespace(files=(SimpleNamespace(role='wps_namelist',path=wps),SimpleNamespace(role='geog_index',path=tmp_path/'topo'/'index')))
    outer=SimpleNamespace(static_catalog=inner,static_highres=cfg)
    view=_static_catalog(outer)
    assert view is not inner and view.static_highres is cfg
    assert build.geog_selection_from_catalog(view,1).terrain_smoothing==TerrainSmoothing('none')
    assert _static_catalog(SimpleNamespace(static_catalog=inner)) is inner
    assert catalog_with_smoothing(view,cfg) is view

@pytest.mark.parametrize('receipt',[None,{}, {'terrain_smoothing':{'d03':WPS_DEFAULT.echo()}}])
def test_root_guard_refuses_before_disabled_overlay(tmp_path,receipt):
    with pytest.raises(ValueError,match='terrain-smoothing root seam.*configuration did not ask for'):
        apply_prepared_highres({},SimpleNamespace(),config=carrier(tmp_path),domain_id=3,
            case_date=date(2000,1,1),landuse_attrs=None,baseline_receipt=receipt)

@pytest.mark.parametrize('nested',[False,True])
def test_root_guard_accepts_attestation(tmp_path,nested):
    receipt={'terrain_smoothing':{'d03':TerrainSmoothing('none').echo()}}
    if nested: receipt={'baseline':receipt}
    base={"HGT_M": np.zeros((2, 2))}
    got=apply_prepared_highres(base,SimpleNamespace(),config=carrier(tmp_path),domain_id=3,
            case_date=date(2000,1,1),landuse_attrs=None,baseline_receipt=receipt)
    assert got[0] is base and got[1] is receipt

def test_spec_and_repeat_last():
    text='# [[domain]] in a comment is not a domain\n[[domain]]\ngrid_id=1\n[[domain]]\ngrid_id=3\n[[domain]]\ngrid_id=7\n'
    emitted=tomllib.loads(emit_smoothing(text,parse_smoothing_spec('none,1-2-1:3')))
    assert [d['static']['smooth_option'] for d in emitted['domain']]==['none','1-2-1','1-2-1']
    assert emit_smoothing(text,parse_smoothing_spec('smth-desmth_special:1'))==text
    with pytest.raises(ValueError,match='longer'):
        emit_smoothing(text,parse_smoothing_spec('none,none,none,none'))
    for invalid in ('none:2','bad','1-2-1:0','1-2-1:1.5','1-2-1:2:3',''):
        with pytest.raises(ValueError): parse_smoothing_spec(invalid)

def test_wizard_flag(tmp_path):
    from test_domain_wizard import _run_wizard
    rc,out=_run_wizard(tmp_path,'--terrain-smoothing','none,1-2-1:3')
    assert rc==0
    rows=tomllib.loads(out.read_text())['domain']
    assert rows[0]['static']=={'smooth_option':'none'}
    assert rows[1]['static']=={'smooth_option':'1-2-1','smooth_passes':3}

@pytest.mark.parametrize('lookup',['explicit','opt','default'])
def test_import_table_lookup(tmp_path,lookup):
    from test_namelist_import import _pair,WPS_TEXT
    from woof.namelist_import import import_namelists
    dirname='chosen' if lookup!='default' else 'geogrid'
    directory=tmp_path/dirname;directory.mkdir()
    table=directory/'GEOGRID.TBL';table.write_text('name=HGT_M;smooth_option=1-2-1;smooth_passes=3')
    wps=WPS_TEXT
    if lookup=='opt': wps=wps.replace('&geogrid','&geogrid\nopt_geogrid_tbl_path="chosen",',1)
    pair=_pair(tmp_path,wps=wps)
    text,report=import_namelists(*pair,geogrid_tbl=table if lookup=='explicit' else None)
    assert all(d['static']=={'smooth_option':'1-2-1','smooth_passes':3} for d in tomllib.loads(text)['domain'])
    assert str(table) in report.format()

def test_import_default_byte_identity_and_missing_notice(tmp_path):
    from test_namelist_import import _pair
    from woof.namelist_import import import_namelists
    pair=_pair(tmp_path)
    before,report=import_namelists(*pair)
    table=tmp_path/'GEOGRID.TBL';table.write_text('name=HGT_M;smooth_option=smth-desmth_special')
    after,report2=import_namelists(*pair,geogrid_tbl=table)
    assert before==after and report==report2
    missing,report3=import_namelists(*pair,geogrid_tbl=tmp_path/'absent')
    assert missing==before
    assert len(report3.notices)==len(report.notices)+1
    assert any('was not imported' in notice for notice in report3.notices)

@pytest.mark.parametrize('partial',[False,True])
@pytest.mark.parametrize('setting',[WPS_DEFAULT,TerrainSmoothing('none'),TerrainSmoothing('1-2-1',3),WPS_EXACT_DEFAULT])
def test_overlay_path(tmp_path,monkeypatch,partial,setting):
    from woof.static.highres import _terrain_on_coverage
    monkeypatch.setenv('WOOF_STATIC_PYTHON','1')
    a=plane();grid=SimpleNamespace(e_sn=6,e_we=8)
    base=np.full((5,7),80.0)
    if partial: a[0:4,0:4]=np.nan
    got,audit=_terrain_on_coverage(a,grid,halo=3,baseline={'HGT_M':base},source_id='synthetic',latlon=None,terrain_smoothing=setting)
    if not partial:
        bits(got,smooth_terrain_reference(a,setting)[3:8,3:10])
    else:
        from woof.static.highres import _coverage_weight
        covered=np.isfinite(a);weight=_coverage_weight(covered)[3:8,3:10]
        filled=np.where(covered,a,np.pad(base,3,mode='edge'))
        high=smooth_terrain_reference(filled,setting)[3:8,3:10]
        bits(got,weight*high+(1-weight)*base)

@pytest.fixture
def synthetic_geog(tmp_path):
    from test_static_build import _write_index, _write_tiles
    from woof.static.projection import MercatorGrid
    root=tmp_path/'GEOG';root.mkdir()
    selection=build.GeogSelection.fallback(root)
    for field in build._DEFAULT_GEOG_DIRS:
        directory=selection.path(field);directory.mkdir(parents=True,exist_ok=True)
        categorical=field in ('landuse','soil_top','soil_bottom')
        nz=12 if field in ('greenfrac','lai','albedo') else 1
        kv=_write_index(directory,dx=10,dy=10,known_lat=-85,known_lon=-175,tile_x=6,tile_y=6,
                        tile_z=nz,type='categorical' if categorical else 'continuous',
                        category_min=1 if categorical else None,category_max=21 if categorical else None,
                        iswater=17 if field=='landuse' else None,islake=21 if field=='landuse' else None)
        a=np.full((nz,18,36),2 if categorical else 80,dtype=np.int16)
        if field=='terrain':
            a[0]=np.arange(648).reshape(18,36)*3
        _write_tiles(directory,a,kv)
    grid=MercatorGrid(0,0,0,0,0,100000,100000,8,6)
    return root,selection,grid

@pytest.mark.parametrize('route',['python','rust'])
@pytest.mark.parametrize('setting',[WPS_DEFAULT,TerrainSmoothing('none'),TerrainSmoothing('1-2-1',3),TerrainSmoothing('smth-desmth',2),TerrainSmoothing('smth-desmth_special',4),WPS_EXACT_DEFAULT])
def test_both_builders_use_policy(synthetic_geog,monkeypatch,route,setting):
    root,selection,grid=synthetic_geog
    if route=='rust' and rust_bridge.unavailable_reason():
        pytest.skip(rust_bridge.unavailable_reason())
    monkeypatch.setenv('WOOF_STATIC_PYTHON','1' if route=='python' else '0')
    selected=replace(selection,terrain_smoothing=setting)
    # Compute the expected sample through this route's unsmoothed builder.
    # Portable Rust ('wps-sampling-portable-v1') and the Python fallback
    # ('python-platform-sampling') have separate sampling contracts
    # (woof/static/sampling_contract.py): the Python float32 WPS twin takes
    # np.arctan, which NumPy runs through its AVX-512 loop on such hosts and
    # which lands one float32 ULP from the Rust twin's vendored libm atanf on
    # 5 of this grid's 11 extended rows. Holding the Rust route to Python
    # samples failed the smth-desmth_special x4 case on an AVX-512 runner and
    # passed on AVX2 ones. Each route is therefore held to its own samples.
    # A builder returns only the cropped domain, so the extended samples are
    # the route's own unsmoothed build of this grid widened by the halo with
    # the known point moved with it: every float32 twin offset (x - knowni)
    # and the float64 longitude offset (x - known_x) are then exactly those
    # of the halo cell, so each widened cell IS that extended cell.
    halo=build.HALO
    none=replace(selection,terrain_smoothing=TerrainSmoothing('none'))
    wide=type(grid)(grid.ref_lat,grid.ref_lon,grid.truelat1,grid.truelat2,grid.stand_lon,
                    grid.dx,grid.dy,grid.e_we+2*halo,grid.e_sn+2*halo,
                    known_x=grid.known_x+halo,known_y=grid.known_y+halo)
    extended=build.build_terrain(wide,root,selection=none)
    sampler=build._DomainSampler(grid,halo)
    if route=='python':
        # Validate the widening on the route whose extended plane is
        # visible: it must be the sampler's own halo-extended samples.
        from woof.static.geog import GeogDataset
        ds=GeogDataset(selection.path('terrain'))
        bits(extended,sampler.continuous(ds,sampler.window(ds),0))
    expected=smooth_terrain_reference(extended,setting)[sampler.crop]
    fields=build.build_static(grid,root,selection=selected)
    terrain=build.build_terrain(grid,root,selection=selected)
    bits(terrain,expected)
    bits(fields['HGT_M'],terrain)
    if setting.is_default:
        bits(fields['HGT_M'],build.smth_desmth_special(extended,1)[sampler.crop])
        default=build.build_static(grid,root)
        assert set(default)==set(fields)
        for name in fields: bits(default[name],fields[name])

ORACLE_PATH=Path(__file__).parent/'fixtures'/'wps_smooth_v460'/'wps_smooth_v460.npz'

@pytest.mark.parametrize('name',['ridges_noise','coast_valley','spikes','near_subnormal'])
@pytest.mark.parametrize('code,passes',[(c,p) for c,ps in [(1,(0,1,2,3,5)),(2,(0,1,2,3)),(3,(0,1,2,3))] for p in ps])
def test_wps_v460_oracle(name,code,passes):
    if not ORACLE_PATH.is_file():
        pytest.skip('separately supplied WPS v4.6 oracle fixture is absent')
    from woof.static.terrain_smoothing import _wps_smooth_f32
    with np.load(ORACLE_PATH) as oracle:
        index=list(map(tuple,oracle['cases'])).index((code,passes))
        a=oracle['in_'+name].astype('f8')
        expected=oracle['out_'+name][index].astype('f8')
        setting=TerrainSmoothing('none') if passes==0 else TerrainSmoothing({1:'1-2-1',2:'smth-desmth',3:'smth-desmth_special'}[code],passes)
        # WPS special x1 intentionally differs from the preserved production
        # default; smooth_precision = "wps-float32" is the option that
        # reproduces it, through both production entries.
        got=a.copy() if passes==0 else _wps_smooth_f32(a,setting)
        bits(got,expected)
        production=WPS_EXACT_DEFAULT if setting.is_default else setting
        bits(smooth_terrain_reference(a,production),expected)
        if rust_bridge.unavailable_reason() is None:
            bits(rust_bridge.terrain_smooth(a,production),expected)

def test_case_loader_keeps_policy_out_of_experiment_identity(tmp_path):
    from test_case_data import make_case_toml, _EXPERIMENT_TOML
    from woof.case_data import load_experiment_case
    plain=make_case_toml(tmp_path)
    old,data=load_experiment_case(plain)
    request=_EXPERIMENT_TOML.replace('[[domain]]','[[domain]]\nstatic = { smooth_option = "none" }')
    changed=make_case_toml(tmp_path,experiment=request)
    new,data2=load_experiment_case(changed)
    assert new==old
    assert data.static_highres is None
    assert data2.static_highres.smoothing_for(1)==TerrainSmoothing('none')
    assert not hasattr(new.root,'static')

def test_catalog_receipt_attests_only_nondefault_domains(synthetic_geog,tmp_path):
    from woof.hrrr_native_static import verified_static_catalog
    root,_,_=synthetic_geog
    wps=tmp_path/'namelist.wps';wps.write_text('&share\nmax_dom=1,\n/\n&geogrid\ngeog_data_res="default",\n/\n')
    cfg=HighresStaticConfig(False,tmp_path,terrain_smoothing=((1,'none',0),))
    catalog,receipt=verified_static_catalog(wps,root,(1,),static_highres=cfg)
    assert receipt['terrain_smoothing']=={'d01':TerrainSmoothing('none').echo()}
    assert build.geog_selection_from_catalog(catalog,1).terrain_smoothing==TerrainSmoothing('none')
    default,old=verified_static_catalog(wps,root,(1,))
    assert 'terrain_smoothing' not in old
    assert not hasattr(default,'static_highres')

def test_prepared_settings_bind_smoothing(tmp_path):
    from woof.static.highres_production import prepared_highres_settings_match
    cfg=carrier(tmp_path)
    assert prepared_highres_settings_match(cfg.echo(),cfg)
    default=replace(cfg,terrain_smoothing=())
    assert not prepared_highres_settings_match(default.echo(),cfg)
    assert not prepared_highres_settings_match(cfg.echo(),default)

def test_vertical_survey_uses_same_policy(tmp_path,monkeypatch):
    from woof.vertical_adaptation import run_terrain_fields
    from woof.static import corridor
    wps=tmp_path/'namelist.wps';wps.write_text('&share\nmax_dom=2,\n/\n&geogrid\ngeog_data_res="default","default",\n/\n')
    inner=SimpleNamespace(files=(SimpleNamespace(role='wps_namelist',path=wps),SimpleNamespace(role='geog_index',path=tmp_path/'topo'/'index')))
    cfg=HighresStaticConfig(False,tmp_path,terrain_smoothing=((2,'none',0),))
    seen=[]
    def terrain(grid,root,*,selection):
        seen.append(selection.terrain_smoothing)
        return np.zeros((2,2))
    monkeypatch.setattr(build,'build_terrain',terrain)
    monkeypatch.setattr(corridor,'moving_grid_ids',lambda exp:())
    root=SimpleNamespace(grid_id=1,parent_id=0,run=SimpleNamespace(base_temp=300))
    child=SimpleNamespace(grid_id=2,parent_id=1,run=SimpleNamespace(base_temp=300))
    exp=SimpleNamespace(domains=(root,child))
    result=run_terrain_fields(exp,(object(),object()),root_terrain=np.zeros((2,2)),static_catalog=inner,static_highres=cfg,corridors=False)
    assert len(result)==2 and seen==[TerrainSmoothing('none')]

def test_wizard_default_emission_is_identical():
    from woof.domain_wizard import render_config
    from test_domain_wizard import _RADT_PROJECTION
    kwargs=dict(name='synthetic',start_time=datetime(2000,1,1,12),hours=6,
                projection=dict(_RADT_PROJECTION),dims=[(100,80),(61,61)],ratios=(3,),
                fetch_hints={'source':'era5'},case_data=None)
    old=render_config(**kwargs)
    assert emit_smoothing(old,parse_smoothing_spec('smth-desmth_special:1'))==old
    new=emit_smoothing(old,parse_smoothing_spec('none'))
    assert all(d['static']=={'smooth_option':'none'} for d in tomllib.loads(new)['domain'])

def test_import_missing_namelist_table_notice(tmp_path):
    from test_namelist_import import _pair,WPS_TEXT
    from woof.namelist_import import import_namelists
    text=WPS_TEXT.replace('&geogrid','&geogrid\nopt_geogrid_tbl_path="missing",',1)
    pair=_pair(tmp_path,wps=text)
    toml,report=import_namelists(*pair)
    assert all('static' not in d for d in tomllib.loads(toml)['domain'])
    assert any('was not imported' in notice for notice in report.notices)

@pytest.mark.parametrize('setting',[TerrainSmoothing('none'),TerrainSmoothing('1-2-1',3),WPS_EXACT_DEFAULT])
def test_production_overlay_threads_domain_policy(tmp_path,monkeypatch,setting):
    reason=rust_bridge.unavailable_reason()
    if reason: pytest.skip(reason)
    from test_prepared_highres import _grid
    from test_static_highres_warp_routing import HIGHRES_FIXTURES,_bound
    from test_static_highres_international import _baseline
    from woof.static import highres_production as owner
    from woof.static.highres import build_terrain_override,merge_terrain_override
    grid=_grid()
    baseline=_baseline(grid.e_sn-1,grid.e_we-1)
    source=_bound(HIGHRES_FIXTURES/'terrain_clip.tif',role='terrain')
    cfg=HighresStaticConfig(True,tmp_path,fields='terrain',terrain_smoothing=(setting.row(3),))
    monkeypatch.setattr(owner,'_fetch_terrain',lambda *a,**k:(source,{'terrain_bytes_fetched':0}))
    overrides,audit=build_terrain_override(grid,terrain=source,baseline=baseline,terrain_smoothing=setting)
    assert f'WPS terrain smoothing {setting.label()}' in audit['method']
    expected,_=merge_terrain_override(baseline,overrides)
    fields,receipt=owner.apply_highres_statics(baseline,grid,config=cfg,domain_id=3,
        case_date=date(2000,1,1),landuse_attrs=None)
    bits(fields['HGT_M'],expected['HGT_M'])
    assert receipt['config']['terrain_smoothing']==[list(setting.row(3))]


# ---------------------------------------------------------------------------
# WPS geogrid.exe end to end on a real domain.
# ---------------------------------------------------------------------------

GEOGRID_PATH = (Path(__file__).parent / 'fixtures' / 'wps_smooth_v460'
                / 'geogrid_alps_small.npz')
#: (run, smooth_option, smooth_passes) exactly as geogrid_fixture.sh ran them.
GEOGRID_RUNS = [
    ('none', 'none', 0), ('passes0', 'smth-desmth_special', 0),
    ('special1', 'smth-desmth_special', 1),
    ('special2', 'smth-desmth_special', 2),
    ('special3', 'smth-desmth_special', 3),
    ('smdes1', 'smth-desmth', 1), ('smdes2', 'smth-desmth', 2),
    ('smdes3', 'smth-desmth', 3), ('121x1', '1-2-1', 1),
    ('121x2', '1-2-1', 2), ('121x3', '1-2-1', 3), ('121x5', '1-2-1', 5)]


def _bits32(got, want):
    got = np.asarray(got).astype(np.float32)
    assert got.shape == want.shape
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))


@pytest.mark.parametrize('run,option,passes', GEOGRID_RUNS)
def test_geogrid_exe_identity_on_a_real_domain(run, option, passes):
    """Each GEOGRID.TBL setting against WPS v4.6.0 geogrid.exe's own HGT_M.

    The input is geogrid's own unsmoothed HGT_M on the domain widened by
    the 3-cell halo (tools/wps_smooth_v460_oracle/geogrid_fixture.sh), so
    the comparison isolates the smoother: every non-default setting is
    bit for bit, in Python and through the Rust entry point.  The default
    keeps the engine's historical float64 smoother, 1.2 mm at most from
    WPS on this ground; WPS's own float32 arithmetic is exact there too.
    """
    with np.load(GEOGRID_PATH) as fixture:
        assert [tuple(r) for r in fixture['settings'].tolist()] == [
            (n, o, str(p)) for n, o, p in GEOGRID_RUNS]
        raw = fixture['raw_extended'].astype(np.float64)
        want = fixture[f'hgt_{run}']
    setting = (TerrainSmoothing('none') if option == 'none' or passes <= 0
               else TerrainSmoothing(option, passes))
    crop = (slice(3, -3), slice(3, -3))
    got = smooth_terrain_reference(raw, setting)[crop]
    if setting.is_default:
        from woof.static.terrain_smoothing import _wps_smooth_f32
        _bits32(_wps_smooth_f32(raw, setting)[crop], want)
        drift = np.abs(got - want.astype(np.float64)).max()
        assert 0.0 < drift <= 1.3e-3
        # The option: the same setting in WPS's arithmetic is geogrid.exe's
        # HGT_M bit for bit, through both production entries.
        setting = WPS_EXACT_DEFAULT
        got = smooth_terrain_reference(raw, setting)[crop]
    _bits32(got, want)
    if rust_bridge.unavailable_reason() is None:
        _bits32(rust_bridge.terrain_smooth(raw, setting)[crop], want)


# ---------------------------------------------------------------------------
# Moving domains: the reach rule, measured, and its refusal.
# ---------------------------------------------------------------------------

REACH_SETTINGS = ([TerrainSmoothing('1-2-1', n) for n in range(1, 7)]
                  + [TerrainSmoothing(o, n) for o in ('smth-desmth',
                                                      'smth-desmth_special')
                     for n in (1, 2, 3)] + [WPS_EXACT_DEFAULT])


@pytest.mark.parametrize('setting', REACH_SETTINGS)
def test_footprint_independence_is_the_reach_rule(setting):
    """A setting keeps shared ground identical across two footprints of
    one lattice exactly when its reach stays inside the 3-cell halo."""
    from woof.static.terrain_smoothing import STATIC_HALO
    world = np.random.default_rng(5).uniform(0.0, 3000.0, (60, 70))
    ny, nx, h = 24, 30, STATIC_HALO
    assert h == build.HALO

    def footprint(j0, i0):
        ext = world[j0 - h:j0 + ny + h, i0 - h:i0 + nx + h]
        return smooth_terrain_reference(ext, setting)[h:-h, h:-h]

    old, new = footprint(15, 15), footprint(18, 20)
    differing = np.count_nonzero(old[3:, 5:] != new[:ny - 3, :nx - 5])
    assert (differing == 0) == (setting.reach <= STATIC_HALO)


def _movers(follow_grid=None, relocation_grid=None):
    domains = tuple(SimpleNamespace(grid_id=g, parent_id=p,
                                    follow=object() if g == follow_grid
                                    else None)
                    for g, p in ((1, 0), (2, 1), (3, 2)))
    relocation = None
    if relocation_grid is not None:
        relocation = SimpleNamespace(enabled=True, follow=object(), moves=(),
                                     grid_id=relocation_grid,
                                     containment=None)
    return SimpleNamespace(domains=domains, relocation=relocation)


@pytest.mark.parametrize('movers', [dict(follow_grid=2),
                                    dict(relocation_grid=2)])
def test_moving_domain_refuses_reach_past_the_halo(movers):
    from woof.static.terrain_smoothing import refuse_moving_reach
    exp = _movers(**movers)
    fine = [{'grid_id': 1, 'static': {'smooth_option': '1-2-1',
                                      'smooth_passes': 5}},
            {'grid_id': 2, 'static': {'smooth_option': '1-2-1',
                                      'smooth_passes': 3}},
            {'grid_id': 3, 'static': {'smooth_option': 'smth-desmth'}}]
    refuse_moving_reach(fine, exp, source='synthetic')
    # d03 moves with its parent d02, so it is refused as well.
    for grid_id, table in ((2, {'smooth_option': '1-2-1',
                                'smooth_passes': 4}),
                           (3, {'smooth_option': 'smth-desmth_special',
                                'smooth_passes': 2})):
        tables = [{'grid_id': grid_id, 'static': table}]
        with pytest.raises(ValueError, match=(
                f'grid_id = {grid_id} of synthetic moves.*past the 3-cell '
                'halo.*first relocation is refused')):
            refuse_moving_reach(tables, exp, source='synthetic')
    # Nothing moves: any setting stands.
    refuse_moving_reach([{'grid_id': 2, 'static': {
        'smooth_option': '1-2-1', 'smooth_passes': 6}}], _movers(),
        source='synthetic')


def test_loader_refuses_reach_past_the_halo_on_a_following_nest(tmp_path):
    from test_corridor_reach import _follow_config
    from woof.experiment import load_experiment
    path = _follow_config(tmp_path, '')
    text = path.read_text(encoding='utf-8')
    head, sep, tail = text.rpartition('\nfollow = {')
    assert sep
    for option, passes, refused in (('smth-desmth', 2, True),
                                    ('1-2-1', 3, False)):
        out = path.with_name(f'{option}{passes}.toml')
        line = (f'\nstatic = {{ smooth_option = "{option}", '
                f'smooth_passes = {passes} }}')
        out.write_text(head + line + sep + tail, encoding='utf-8')
        out.with_suffix('.namelist.wps').write_bytes(
            path.with_suffix('.namelist.wps').read_bytes())
        if refused:
            with pytest.raises(ValueError,
                               match='first relocation is refused'):
                load_experiment(out)
        else:
            assert load_experiment(out).domains[1].follow is not None


def test_table_reader_keeps_quoted_spaces_like_wps(tmp_path):
    from woof.static.terrain_smoothing import _despace
    assert _despace(' name = HGT_M ;\tsmooth_option = 1-2-1') == (
        'name=HGT_M;smooth_option=1-2-1')
    assert _despace('descr = "Topography height"') == (
        'descr="Topography height"')
    table = tmp_path / 'GEOGRID.TBL'
    table.write_text('name = HGT_M\n  descr = "a b"; smooth_option = 1-2-1\n')
    assert geogrid_tbl_smoothing(table) == TerrainSmoothing('1-2-1')


# ---------------------------------------------------------------------------
# smooth_precision: WPS's arithmetic for the default smoother is an option;
# the historical float64 smoother stays the default.
# ---------------------------------------------------------------------------

def test_precision_model_keeps_every_prior_setting_and_names_only_the_option():
    assert SMOOTH_PRECISIONS == ('float64', 'wps-float32')
    assert WPS_DEFAULT.precision == 'float64' and WPS_DEFAULT.is_default
    assert TerrainSmoothing(precision='float64') == WPS_DEFAULT
    exact = WPS_EXACT_DEFAULT
    assert (exact.option, exact.passes, exact.precision) == (
        'smth-desmth_special', 1, 'wps-float32')
    assert not exact.is_default and exact.names_precision
    assert exact.echo() == {'smooth_option': 'smth-desmth_special',
                            'smooth_passes': 1,
                            'smooth_precision': 'wps-float32'}
    assert exact.label() == 'smth-desmth_special x1 wps-float32'
    assert exact.reach == WPS_DEFAULT.reach == 2
    # Every setting that existed before the option keeps its echo, row and
    # label byte for byte, and naming its own arithmetic changes nothing.
    for setting in SETTINGS + [WPS_DEFAULT]:
        assert not setting.names_precision
        assert set(setting.echo()) == {'smooth_option', 'smooth_passes'}
        assert len(setting.row(4)) == 3
        assert TerrainSmoothing(setting.option, setting.passes,
                                setting.precision) == setting
    assert TerrainSmoothing('1-2-1', 3, 'wps-float32') == TerrainSmoothing('1-2-1', 3)
    assert TerrainSmoothing('none', 0, 'float64') == TerrainSmoothing('none')


@pytest.mark.parametrize('option,passes,precision,match', [
    ('1-2-1', 3, 'float64', 'only the default'),
    ('smth-desmth', 1, 'float64', 'only the default'),
    ('smth-desmth_special', 2, 'float64', 'only the default'),
    ('none', 0, 'wps-float32', 'runs no smoother'),
    ('smth-desmth_special', 1, 'float32', 'unknown terrain smoothing precision'),
    ('smth-desmth_special', 1, 'WPS-FLOAT32', 'unknown terrain smoothing precision'),
])
def test_precision_refusals_name_the_ignored_arithmetic(option, passes,
                                                        precision, match):
    with pytest.raises(ValueError, match=match):
        TerrainSmoothing(option, passes, precision)


def test_precision_rows_round_trip_and_refuse_a_guess():
    for setting in SETTINGS + [WPS_DEFAULT, WPS_EXACT_DEFAULT]:
        assert TerrainSmoothing.from_row(setting.row(5)) == (5, setting)
        assert TerrainSmoothing.from_row(list(setting.row(5))) == (5, setting)
    assert WPS_EXACT_DEFAULT.row(2) == (2, 'smth-desmth_special', 1, 'wps-float32')
    with pytest.raises(ValueError, match='canonical row'):
        TerrainSmoothing.from_row((2, '1-2-1', 3, 'wps-float32'))
    for bad in ((2, 'none'), (2, 'smth-desmth_special', 1, 'wps-float32', 0)):
        with pytest.raises(ValueError, match='would be guessed'):
            TerrainSmoothing.from_row(bad)


def test_precision_arithmetic_is_wps_and_the_default_is_untouched(monkeypatch):
    a = plane()
    legacy = build.smth_desmth_special(a, passes=1)
    exact = smooth_terrain_reference(a, WPS_EXACT_DEFAULT)
    bits(exact, fortran_loop(a, WPS_DEFAULT))
    assert not np.array_equal(exact, legacy)
    bits(smooth_terrain_reference(a, WPS_DEFAULT), legacy)
    # The option is a non-default setting: the production entry routes it
    # to the Rust bridge (here the Python reference, reported).
    monkeypatch.setenv('WOOF_STATIC_PYTHON', '1')
    bits(smooth_terrain(a, WPS_EXACT_DEFAULT), exact)


def test_old_library_names_the_precision_option_stale(monkeypatch):
    class Library:
        gpuwm_static_terrain_smooth = SimpleNamespace()
    monkeypatch.setattr(rust_bridge, 'load', lambda: Library())
    with pytest.raises(rust_bridge.StaticBridgeError,
                       match='predates terrain-smoothing options for '
                             'smooth_precision.*rebuild'):
        rust_bridge.terrain_smooth(plane(), WPS_EXACT_DEFAULT)
    bits(smooth_terrain(plane(), WPS_DEFAULT),
         build.smth_desmth_special(plane(), 1))


def test_build_terrain_keeps_the_stale_library_refusal_whole(monkeypatch, tmp_path):
    message = ('the staged static-fields library predates terrain-smoothing '
               'options for smooth_precision (x); restage the rebuilt bridge '
               'or rebuild from this checkout: cd tools/rustwx')

    def stale(*args):
        raise rust_bridge.StaticBridgeError(message)
    bridge = SimpleNamespace(build_terrain_smoothed=stale)
    monkeypatch.setattr(rust_bridge, 'route', lambda op: bridge)
    grid = SimpleNamespace(_rust_sampling_handle=lambda b: 1)
    selection = replace(build.GeogSelection.fallback(tmp_path),
                        terrain_smoothing=WPS_EXACT_DEFAULT)
    with pytest.raises(rust_bridge.StaticBridgeError) as caught:
        build.build_terrain(grid, tmp_path, selection=selection)
    assert str(caught.value) == message


@pytest.mark.parametrize('table,want', [
    ({'smooth_precision': 'wps-float32'}, WPS_EXACT_DEFAULT),
    ({'smooth_precision': 'float64'}, WPS_DEFAULT),
    ({'smooth_option': 'smth-desmth_special', 'smooth_passes': 1,
      'smooth_precision': 'wps-float32'}, WPS_EXACT_DEFAULT),
    ({'smooth_option': 'smth-desmth_special',
      'smooth_precision': 'wps-float32'}, WPS_EXACT_DEFAULT),
    ({'smooth_option': '1-2-1', 'smooth_passes': 3,
      'smooth_precision': 'wps-float32'}, TerrainSmoothing('1-2-1', 3)),
])
def test_domain_static_precision(table, want):
    assert parse_domain_static(table, source='synthetic', grid_id=2) == want


@pytest.mark.parametrize('table,match', [
    ({'smooth_passes': 2, 'smooth_precision': 'wps-float32'}, 'requires smooth_option'),
    ({'smooth_option': 'none', 'smooth_precision': 'float64'}, 'precision for no smoother'),
    ({'smooth_option': '1-2-1', 'smooth_precision': 'float64'}, 'only the default'),
    ({'smooth_precision': 'float32'}, 'unknown terrain smoothing precision'),
    ({'smooth_precision': True}, 'unknown terrain smoothing precision'),
    ({'smooth_precison': 'wps-float32'}, "unknown keys 'smooth_precison'.*smooth_precision"),
])
def test_domain_static_precision_refusals(table, match):
    with pytest.raises(ValueError, match=match):
        parse_domain_static(table, source='synthetic', grid_id=2)


@pytest.mark.parametrize('setting', SETTINGS + [WPS_EXACT_DEFAULT])
def test_static_line_round_trips_through_toml(setting):
    line = static_inline(setting)
    assert ('smooth_precision' in line) == (setting is WPS_EXACT_DEFAULT)
    table = tomllib.loads(line)['static']
    assert parse_domain_static(table, source='synthetic', grid_id=1) == setting
    text = '[[domain]]\ngrid_id = 1\n[[domain]]\ngrid_id = 2\n'
    emitted = tomllib.loads(emit_smoothing(text, (setting,)))
    assert [parse_domain_static(d['static'], source='s', grid_id=d['grid_id'])
            for d in emitted['domain']] == [setting, setting]


def test_precision_rides_the_carrier_receipts_and_root_seam(tmp_path):
    args = dict(source='synthetic', base_dir=tmp_path, spacings_m=[12000, 4000])
    raw = {'domain': [{'grid_id': 1, 'static': {'smooth_precision': 'wps-float32'}},
                      {'grid_id': 2, 'static': {'smooth_option': '1-2-1', 'smooth_passes': 3}}]}
    cfg = resolve_static_highres(raw, **args)
    assert cfg.terrain_smoothing == ((1, 'smth-desmth_special', 1, 'wps-float32'),
                                     (2, '1-2-1', 3))
    assert cfg.echo()['terrain_smoothing'] == [[1, 'smth-desmth_special', 1, 'wps-float32'],
                                               [2, '1-2-1', 3]]
    assert cfg.smoothing_for(1) == WPS_EXACT_DEFAULT
    assert smoothing_for(cfg, 2) == TerrainSmoothing('1-2-1', 3)
    from woof.static.terrain_smoothing import smoothing_receipt
    assert smoothing_receipt(cfg)['d01'] == WPS_EXACT_DEFAULT.echo()
    # A precision named as float64 is the default: no row, no carrier.
    assert resolve_static_highres(
        {'domain': [{'grid_id': 1, 'static': {'smooth_precision': 'float64'}}]},
        **args) is None
    # The root seam tells the two arithmetics apart.
    with pytest.raises(ValueError, match='root seam.*wps-float32'):
        require_root_smoothing(cfg, 1, {'terrain_smoothing': {'d01': WPS_DEFAULT.echo()}})
    require_root_smoothing(cfg, 1, {'terrain_smoothing': smoothing_receipt(cfg)})
    from woof.static.highres_production import prepared_highres_settings_match
    assert prepared_highres_settings_match(cfg.echo(), cfg)
    float64 = replace(cfg, terrain_smoothing=cfg.terrain_smoothing[1:])
    assert not prepared_highres_settings_match(float64.echo(), cfg)


def test_case_loader_carries_precision_outside_experiment_identity(tmp_path):
    from test_case_data import make_case_toml, _EXPERIMENT_TOML
    from woof.case_data import load_experiment_case
    old, data = load_experiment_case(make_case_toml(tmp_path))
    request = _EXPERIMENT_TOML.replace(
        '[[domain]]', '[[domain]]\nstatic = { smooth_precision = "wps-float32" }')
    new, data2 = load_experiment_case(make_case_toml(tmp_path, experiment=request))
    assert new == old
    assert data2.static_highres.smoothing_for(1) == WPS_EXACT_DEFAULT
    request = _EXPERIMENT_TOML.replace(
        '[[domain]]', '[[domain]]\nstatic = { smooth_precision = "float64" }')
    same, data3 = load_experiment_case(make_case_toml(tmp_path, experiment=request))
    assert same == old and data3.static_highres is None


def test_catalog_receipt_attests_the_precision(synthetic_geog, tmp_path):
    from woof.hrrr_native_static import verified_static_catalog
    root, _, _ = synthetic_geog
    wps = tmp_path / 'namelist.wps'
    wps.write_text('&share\nmax_dom=1,\n/\n&geogrid\ngeog_data_res="default",\n/\n')
    cfg = HighresStaticConfig(False, tmp_path, terrain_smoothing=(WPS_EXACT_DEFAULT.row(1),))
    catalog, receipt = verified_static_catalog(wps, root, (1,), static_highres=cfg)
    assert receipt['terrain_smoothing'] == {'d01': WPS_EXACT_DEFAULT.echo()}
    assert build.geog_selection_from_catalog(catalog, 1).terrain_smoothing == WPS_EXACT_DEFAULT


def test_with_precision_touches_only_the_default_smoother():
    for setting in SETTINGS:
        assert with_precision(setting, 'wps-float32') == setting
        assert with_precision(setting, 'float64') == setting
    assert with_precision(WPS_DEFAULT, 'wps-float32') == WPS_EXACT_DEFAULT
    assert with_precision(WPS_EXACT_DEFAULT, 'float64') == WPS_DEFAULT
    assert with_precision(WPS_EXACT_DEFAULT, None) is WPS_EXACT_DEFAULT
    with pytest.raises(ValueError, match='unknown terrain smoothing precision'):
        with_precision(TerrainSmoothing('none'), 'float32')


def test_wizard_precision_flag(tmp_path):
    from test_domain_wizard import _run_wizard
    rc, out = _run_wizard(tmp_path, '--terrain-smoothing-precision', 'wps-float32')
    assert rc == 0
    rows = tomllib.loads(out.read_text())['domain']
    assert len(rows) >= 2
    assert all(row['static'] == WPS_EXACT_DEFAULT.echo() for row in rows)
    rc, out = _run_wizard(tmp_path, '--terrain-smoothing', 'none,smth-desmth_special',
                          '--terrain-smoothing-precision', 'wps-float32')
    assert rc == 0
    rows = tomllib.loads(out.read_text())['domain']
    assert rows[0]['static'] == {'smooth_option': 'none'}
    assert all(row['static'] == WPS_EXACT_DEFAULT.echo() for row in rows[1:])


def test_wizard_float64_precision_is_the_default_emission():
    from woof.domain_wizard import render_config
    from test_domain_wizard import _RADT_PROJECTION
    kwargs = dict(name='synthetic', start_time=datetime(2000, 1, 1, 12), hours=6,
                  projection=dict(_RADT_PROJECTION), dims=[(100, 80), (61, 61)], ratios=(3,),
                  fetch_hints={'source': 'era5'}, case_data=None)
    old = render_config(**kwargs)
    assert emit_smoothing(old, (with_precision(WPS_DEFAULT, 'float64'),)) == old


def _import(tmp_path, table_text=None, **kwargs):
    from test_namelist_import import _pair
    from woof.namelist_import import import_namelists
    table = None
    if table_text is not None:
        table = tmp_path / 'GEOGRID.TBL'
        table.write_text(table_text)
    return import_namelists(*_pair(tmp_path), geogrid_tbl=table, **kwargs)


def _statics(text):
    return [d.get('static') for d in tomllib.loads(text)['domain']]


STOCK_HGT_M = ('name = HGT_M\n        priority = 1\n        dest_type = continuous\n'
               '        interp_option = default:average_gcell(4.0)+four_pt+average_4pt\n'
               '        smooth_option = smth-desmth_special; smooth_passes=1\n')


def test_import_selects_the_option_by_flag_or_table(tmp_path):
    before, _ = _import(tmp_path)
    stock, _ = _import(tmp_path, STOCK_HGT_M)
    kept, _ = _import(tmp_path, STOCK_HGT_M, terrain_smoothing_precision='float64')
    assert before == stock == kept
    flagged, report = _import(tmp_path, STOCK_HGT_M, terrain_smoothing_precision='wps-float32')
    assert all(s == WPS_EXACT_DEFAULT.echo() for s in _statics(flagged))
    assert any('smth-desmth_special x1 wps-float32' in n and '--terrain-smoothing-precision' in n
               for n in report.notices)
    no_table, _ = _import(tmp_path, terrain_smoothing_precision='wps-float32')
    assert no_table == flagged
    tabled, report = _import(tmp_path, STOCK_HGT_M + '        smooth_precision = wps-float32\n')
    assert tabled == flagged
    assert any('imported from' in n and 'wps-float32' in n for n in report.notices)
    # The flag wins over the table.
    overridden, _ = _import(tmp_path, STOCK_HGT_M + '        smooth_precision = wps-float32\n',
                            terrain_smoothing_precision='float64')
    assert overridden == before


def test_import_precision_leaves_a_one_arithmetic_smoother(tmp_path):
    table = 'name=HGT_M;smooth_option=1-2-1;smooth_passes=3'
    plain, _ = _import(tmp_path, table)
    flagged, report = _import(tmp_path, table, terrain_smoothing_precision='wps-float32')
    assert plain == flagged
    assert all(s == {'smooth_option': '1-2-1', 'smooth_passes': 3} for s in _statics(flagged))
    assert any('leaves terrain smoothing 1-2-1 x3 as it is' in notice for notice in report.notices)


@pytest.mark.parametrize('table,match', [
    ('name=HGT_M;smooth_option=1-2-1;smooth_passes=0;smooth_precision=wps-float32',
     'runs no smoother'),
    ('name=HGT_M;smooth_option=smth-desmth_special;smooth_precision=float32',
     'HGT_M entry: unknown terrain smoothing precision'),
    ('name=HGT_M;smooth_option=1-2-1;smooth_precision=float64', 'HGT_M entry: .*only the default'),
])
def test_geogrid_table_precision_refusals(tmp_path, table, match):
    path = tmp_path / 'GEOGRID.TBL'
    path.write_text(table)
    with pytest.raises(ValueError, match=match):
        geogrid_tbl_smoothing(path)


def test_import_cli_flag_round_trips(tmp_path, capsys):
    from test_namelist_import import _pair
    from woof.cli import main as cli_main
    wps, inp = _pair(tmp_path)
    out = tmp_path / 'imported.toml'
    rc = cli_main(['import-namelist', str(wps), str(inp), '--output', str(out),
                   '--terrain-smoothing-precision', 'wps-float32'])
    assert rc == 0
    text = out.read_text(encoding='utf-8')
    assert all(s == WPS_EXACT_DEFAULT.echo() for s in _statics(text))
    from woof.domain_wizard import experiment_from_text
    experiment_from_text(text, source=str(out))
    cfg = resolve_static_highres(tomllib.loads(text), source=str(out),
                                 base_dir=tmp_path)
    ids = [d['grid_id'] for d in tomllib.loads(text)['domain']]
    assert [cfg.smoothing_for(i) for i in ids] == [WPS_EXACT_DEFAULT] * len(ids)


def test_a_carrier_rebuilt_from_its_echo_keeps_the_precision(tmp_path):
    """A sealed preparation keeps the carrier's echo (JSON lists) and a
    rebuild reads it back: the precision column survives and the rebuilt
    carrier equals the sealed one."""
    import json
    from woof.static.terrain_smoothing import smoothing_rows_from_echo
    raw = {'domain': [{'grid_id': 2, 'static': {'smooth_precision': 'wps-float32'}},
                      {'grid_id': 1, 'static': {'smooth_option': 'none'}}]}
    cfg = resolve_static_highres(raw, source='synthetic', base_dir=tmp_path,
                                 spacings_m=[3000, 1000])
    sealed = json.loads(json.dumps(cfg.echo()))
    assert sealed['terrain_smoothing'] == [[1, 'none', 0],
                                           [2, 'smth-desmth_special', 1, 'wps-float32']]
    rows = smoothing_rows_from_echo(sealed['terrain_smoothing'], source='sealed')
    assert rows == cfg.terrain_smoothing
    rebuilt = replace(cfg, terrain_smoothing=rows)
    assert rebuilt == cfg and rebuilt.smoothing_for(2) == WPS_EXACT_DEFAULT
    # The sealed readers (the native hierarchy and the prepared restore)
    # read the whole identity through parse_sealed_static_highres, which
    # hands the rows to the reader above: the precision column survives.
    from woof.static.highres_production import (
        parse_sealed_static_highres, static_highres_identity)
    identity = json.loads(json.dumps(static_highres_identity(cfg)))
    back = parse_sealed_static_highres(identity, source='sealed root preparation',
                                       base_dir=tmp_path)
    assert back == cfg and back.smoothing_for(2) == WPS_EXACT_DEFAULT


# ---------------------------------------------------------------------------
# A157: GFS and mapped-source roots carry the setting; sealed echoes read it.
# ---------------------------------------------------------------------------

_DOOR_STATICS = ('LANDMASK', 'LU_INDEX', 'HGT_M', 'SCT_DOM', 'TMN', 'MAPFAC_M',
                 'MAPFAC_U', 'MAPFAC_V', 'F', 'E', 'SINALPHA', 'COSALPHA')


def _gfs_door(tmp_path, monkeypatch, setting, name):
    """The real GFS door over the CPU fixture; returns (carriers seen, proof).

    Only the expensive source and array work is substituted
    (tests/test_gfs_initial_perturbation.py).  The root static build is
    replaced by what era5_direct._static_from_geog returns for the carrier
    it is handed: the catalog receipt attests exactly the non-default rows
    (test_catalog_receipt_attests_only_nondefault_domains).
    """
    from test_gfs_initial_perturbation import _config, _cpu_preparation, _inputs
    from woof import gfs_direct
    from woof.experiment import load_experiment
    from woof.static.terrain_smoothing import smoothing_receipt
    config = _config(tmp_path, domains=2)
    if setting is not None:
        config.write_text(emit_smoothing(config.read_text(encoding='utf-8'),
                                         (setting,)), encoding='utf-8')
    monkeypatch.setenv('WOOF_CHAINED_PREP', '0')
    _cpu_preparation(monkeypatch, load_experiment(config))
    seen = []
    statics = {key: np.ones((3, 3)) for key in _DOOR_STATICS}

    def static_from_geog(*_args, static_highres=None):
        seen.append(static_highres)
        receipt = smoothing_receipt(static_highres)
        return statics, ({'terrain_smoothing': receipt} if receipt else {}), None

    monkeypatch.setattr(gfs_direct, '_static_from_geog', static_from_geog)
    return seen, gfs_direct.prepare_gfs_wrf(**_inputs(tmp_path, config, name))


@pytest.mark.parametrize('setting', [TerrainSmoothing('none'), TerrainSmoothing('1-2-1', 3),
                                     WPS_EXACT_DEFAULT])
def test_gfs_door_builds_the_root_with_the_setting_and_passes_the_seam(
        tmp_path, monkeypatch, setting):
    # Before A157 the door built its root without the carrier, so a
    # non-default root setting built default terrain and the root seam
    # refused the preparation.
    seen, proof = _gfs_door(tmp_path, monkeypatch, setting, 'smoothed')
    (handed,) = seen
    assert handed.smoothing_for(1) == setting
    assert handed.smoothing_for(2) == setting
    assert proof['schema']


def test_gfs_door_default_root_build_gets_no_carrier_rows(tmp_path, monkeypatch):
    seen, _ = _gfs_door(tmp_path, monkeypatch, None, 'default')
    (handed,) = seen
    assert not getattr(handed, 'terrain_smoothing', ())


def _mapped_door(tmp_path, monkeypatch, setting):
    from test_mapped_direct import _install_prepare_fakes
    from woof import mapped_direct
    from woof.static.terrain_smoothing import static_inline
    args, calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend='cpu')
    if setting is not None:
        args['experiment_config'].write_text(
            '[static.highres]\nenabled = false\ncache_root = "hr-cache"\n\n'
            '[[domain]]\ngrid_id = 1\n' + static_inline(setting) + '\n',
            encoding='utf-8')
    selections = []
    fake = mapped_direct.GeogSelection

    class Capturing:
        @staticmethod
        def from_case_data(case, domain_id):
            selections.append((case, domain_id))
            return fake.from_case_data(case, domain_id)

    monkeypatch.setattr(mapped_direct, 'GeogSelection', Capturing)
    return selections, calls, mapped_direct.prepare_mapped_wrf(**args)


@pytest.mark.parametrize('setting', [TerrainSmoothing('none'), TerrainSmoothing('smth-desmth', 1),
                                     WPS_EXACT_DEFAULT])
def test_mapped_door_builds_the_root_with_the_setting_and_attests_it(
        tmp_path, monkeypatch, setting):
    selections, calls, proof = _mapped_door(tmp_path, monkeypatch, setting)
    ((case, domain_id),) = selections
    assert domain_id == 1
    assert smoothing_for(case.static_highres, 1) == setting
    assert calls['build_static'] == 1
    assert proof['execution_inputs']['root_static_receipt'] == {
        'terrain_smoothing': {'d01': setting.echo()}}


def test_mapped_door_default_root_receipt_is_unchanged(tmp_path, monkeypatch):
    selections, _calls, proof = _mapped_door(tmp_path, monkeypatch, None)
    ((case, _),) = selections
    assert smoothing_for(case.static_highres, 1) == WPS_DEFAULT
    assert proof['execution_inputs']['root_static_receipt'] is None


def test_sealed_echo_round_trips_the_carrier(tmp_path):
    from woof.static.highres_production import (
        parse_sealed_static_highres, parse_static_table, static_highres_identity)
    cfg = carrier(tmp_path)
    echo = json.loads(json.dumps(static_highres_identity(cfg)))
    # parse_static_table alone still refuses the rows (they are no
    # [static.highres] key), which is why the sealed readers use this.
    with pytest.raises(ValueError, match="does not have a key 'terrain_smoothing'"):
        parse_static_table({'highres': echo}, source='sealed', base_dir=tmp_path)
    back = parse_sealed_static_highres(echo, source='sealed', base_dir=tmp_path)
    assert back == cfg
    assert static_highres_identity(back) == static_highres_identity(cfg)
    plain = HighresStaticConfig(True, tmp_path / 'cache', fields='terrain')
    assert parse_sealed_static_highres(
        static_highres_identity(plain), source='sealed', base_dir=tmp_path) == plain
    assert parse_sealed_static_highres(None, source='sealed', base_dir=tmp_path) is None


@pytest.mark.parametrize('rows,match', [
    ([], 'non-empty list'),
    ([[3, 'none']], r'not a \[grid_id'),
    ([[3, 'smth-desmth_special', 1, 'wps-float32', 0]], r'not a \[grid_id'),
    ([[3, 'smth-desmth_special', 1]], 'never records'),
    ([[3, 'smth-desmth_special', 1, 'float64']], 'never records'),
    ([[3, '1-2-1', 2, 'wps-float32']], 'never records'),
    ([[3, 'smth-desmth_special', 1, 'float32']], 'unknown terrain smoothing precision'),
    ([[3, '1-2-1', 2, 'float64']], 'only the default'),
    ([[3, 'none', 0, 'wps-float32']], 'runs no smoother'),
    ([[7, 'none', 0], [3, 'smth-desmth_special', 1, 'wps-float32']], 'out of order'),
    ([[3, 'none', 0], [3, 'smth-desmth_special', 1, 'wps-float32']], 'twice'),
])
def test_sealed_rows_refuse_what_the_carrier_never_writes(rows, match):
    from woof.static.terrain_smoothing import smoothing_rows_from_echo
    with pytest.raises(ValueError, match=match):
        smoothing_rows_from_echo(rows, source='sealed')


@pytest.mark.parametrize('rows,match', [
    ([], 'non-empty list'),
    ('none', 'non-empty list'),
    ([[3, 'none']], 'not a \\[grid_id'),
    ([[0, 'none', 0]], 'not a \\[grid_id'),
    ([[True, 'none', 0]], 'not a \\[grid_id'),
    ([[3, 'spline', 1]], 'unknown terrain smoother'),
    ([[3, '1-2-1', 0]], 'passes must be an integer'),
    ([[3, 'smth-desmth_special', 1]], 'never records'),
    ([[3, 'none', 2]], 'never records'),
    ([[7, 'none', 0], [3, 'none', 0]], 'out of order'),
    ([[3, 'none', 0], [3, '1-2-1', 2]], 'twice'),
])
def test_sealed_echo_refuses_rows_the_carrier_never_writes(tmp_path, rows, match):
    from woof.static.highres_production import parse_sealed_static_highres
    echo = {**HighresStaticConfig(False, tmp_path).echo(), 'enabled': False,
            'terrain_smoothing': rows}
    with pytest.raises(ValueError, match=match):
        parse_sealed_static_highres(echo, source='sealed', base_dir=tmp_path)


def test_sealed_hierarchy_and_prepared_restore_read_the_rows(tmp_path):
    # The two readers that re-parse a sealed carrier echo.  Before A157 both
    # went through parse_static_table and refused a tree whose nests asked
    # for their own smoothing.
    import inspect
    from woof import hrrr_hierarchy_direct, prepared_single_domain_forecast
    from woof.static.highres_production import static_highres_identity
    assert 'parse_sealed_static_highres(' in inspect.getsource(
        hrrr_hierarchy_direct.prepare_hrrr_hierarchy)
    validate = inspect.getsource(
        prepared_single_domain_forecast._validate_hrrr_source_identity)
    assert 'parse_static_table' not in validate
    assert 'parse_sealed_static_highres(' in validate
    # The echo a sealed tree with its nests' own smoothing carries reads
    # back to the carrier's rows, which both readers pass on.
    from woof.static.highres_production import parse_sealed_static_highres
    echo = json.loads(json.dumps(static_highres_identity(carrier(tmp_path))))
    assert parse_sealed_static_highres(
        echo, source='sealed root preparation',
        base_dir=tmp_path).terrain_smoothing == carrier(tmp_path).terrain_smoothing


# --- A169: the native HRRR root builds and attests d01's smoothing --------

def _native_builder(tmp_path, monkeypatch, carrier, *, prior=False):
    """Run tools/hrrr_build_native_static.py with ``carrier`` as the
    namelist-only door hands it over, the field build replaced by the
    fixture's arrays and the selection it was asked for recorded."""
    import sys
    from test_hrrr_native_static import _fixture
    from tools import hrrr_build_native_static as producer
    from woof.static.highres_production import static_highres_identity
    tmp_path.mkdir(parents=True, exist_ok=True)
    target, cache, receipt = _fixture(tmp_path)
    if prior:
        # The geography a sealed static records (as a real receipt does).
        stored = json.loads(receipt.read_text(encoding='utf-8'))
        stored.setdefault('geog_root', str(tmp_path))
        stored.setdefault('geog_selection', {
            key: value['dataset']
            for key, value in stored['geog_source_coverage'].items()})
        receipt.write_text(json.dumps(stored), encoding='utf-8')
    domain = tmp_path / 'domain.json'
    domain.write_text(json.dumps(target.to_payload()), encoding='utf-8')
    with np.load(cache) as stored:
        arrays = {name: stored[name] for name in stored.files}
    geog = tmp_path / 'geog'
    for directory in build._DEFAULT_GEOG_DIRS.values():
        (geog / directory).mkdir(parents=True, exist_ok=True)
        (geog / directory / 'index').write_text('type=continuous\n')
    selections = []
    def fake_build(grid, root, *, selection, source_coverage_report):
        selections.append(selection)
        return dict(arrays)
    monkeypatch.setattr(producer, 'build_static', fake_build)
    out, sealed = tmp_path / 'out.npz', tmp_path / 'out.json'
    source = (['--static-cache', str(cache), '--static-receipt', str(receipt)]
              if prior else ['--geog-root', str(geog)])
    monkeypatch.setattr(sys, 'argv', [
        'native-static', *source, '--domain-spec', str(domain),
        '--static-highres', json.dumps(static_highres_identity(carrier)),
        '--output', str(out), '--receipt', str(sealed)])
    producer.main()
    return selections, json.loads(sealed.read_text(encoding='utf-8'))


def test_native_hrrr_root_builds_and_attests_the_d01_smoothing(tmp_path, monkeypatch):
    """The native HRRR static builder built the default terrain whatever d01
    asked for, so a non-default d01 smoothing was refused at the root seam
    (the ERA5, GFS and mapped roots already carried it)."""
    setting = TerrainSmoothing('1-2-1', 3)
    cfg = HighresStaticConfig(False, tmp_path / 'cache',
                              terrain_smoothing=((1, '1-2-1', 3), (2, 'none', 0)))
    (selection,), receipt = _native_builder(tmp_path, monkeypatch, cfg)
    assert selection == replace(build.GeogSelection.fallback(tmp_path / 'geog'),
                                terrain_smoothing=setting)
    # The root attests its own domain's setting, the one the seam reads.
    assert receipt['terrain_smoothing'] == {'d01': setting.echo()}
    require_root_smoothing(cfg, 1, receipt)


def test_native_hrrr_root_default_build_is_unchanged(tmp_path, monkeypatch):
    for rows in ((), ((2, 'none', 0),)):
        cfg = HighresStaticConfig(False, tmp_path / 'cache', terrain_smoothing=rows)
        work = tmp_path / str(len(rows))
        work.mkdir()
        (selection,), receipt = _native_builder(work, monkeypatch, cfg)
        assert selection == build.GeogSelection.fallback(work / 'geog')
        assert 'terrain_smoothing' not in receipt


def test_native_hrrr_root_from_a_sealed_static_keeps_its_seam(tmp_path, monkeypatch):
    # A prebuilt static that records no smoothing is still refused for a
    # non-default d01 ...
    cfg = HighresStaticConfig(False, tmp_path / 'cache',
                              terrain_smoothing=((1, 'smth-desmth', 2),))
    with pytest.raises(ValueError, match='terrain-smoothing root seam'):
        _native_builder(tmp_path / 'a', monkeypatch, cfg, prior=True)
    # ... and one built under a d01 smoothing is refused for any other
    # request, the default included, which the seam alone passes.
    (tmp_path / 'b').mkdir()
    from test_hrrr_native_static import _fixture
    _target, _cache, receipt_path = _fixture(tmp_path / 'b')
    stored = json.loads(receipt_path.read_text(encoding='utf-8'))
    stored['terrain_smoothing'] = {'d01': TerrainSmoothing('1-2-1', 3).echo()}
    receipt_path.write_text(json.dumps(stored), encoding='utf-8')
    monkeypatch.setattr('test_hrrr_native_static._fixture',
                        lambda path: (_target, _cache, receipt_path))
    default = HighresStaticConfig(False, tmp_path / 'cache')
    with pytest.raises(ValueError, match='was built with d01 terrain smoothing'):
        _native_builder(tmp_path / 'b', monkeypatch, default, prior=True)
    same = HighresStaticConfig(False, tmp_path / 'cache',
                               terrain_smoothing=((1, '1-2-1', 3),))
    _, receipt = _native_builder(tmp_path / 'b', monkeypatch, same, prior=True)
    assert receipt['terrain_smoothing'] == stored['terrain_smoothing']


def test_namelist_only_native_root_hands_the_builder_its_resolved_carrier(tmp_path):
    """A namelist-only preparation handed the builder the [static] table
    alone, which carries neither d01's smoothing nor the urban legend, and
    a carrier holding only smoothing has no [static] table (KeyError)."""
    import inspect
    from tools import prepare_hrrr_wrf as prepare
    # The preparation body moved into _run_configured when the argv door
    # gained its preprocessing-math scope; both halves are read.
    source = (inspect.getsource(prepare._prepare_from_argv)
              + inspect.getsource(prepare._run_configured))
    assert 'json.dumps(static_highres_identity(highres), sort_keys=True)' in source
    assert 'experiment_tables["static"], sort_keys' not in source
