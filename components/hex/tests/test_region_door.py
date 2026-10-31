import json
from pathlib import Path

import pytest

from woof.hex.cli import build_parser
from woof.hex.country_door import plan, REGIONS
from woof.hex.errors import MpasPortError
from woof.hex.lbc_producer import source_endpoint_commands, produce_source_endpoints


def args(tmp_path, *extra):
    return build_parser().parse_args(['country', '--cycle', '2023-03-31T12', '--work-dir', str(tmp_path/'run'), *extra])


def test_existing_country_source_override_binds_both_initialization_and_provenance(tmp_path):
    result = plan(args(tmp_path, '--row', 'philippines', '--source', 'era5'))
    assert result['commands']['init'][2] == 'era5'
    assert 'era5 2023-03-31T12' in result['commands']['forecast'][result['commands']['forecast'].index('--init-source')+1]


@pytest.mark.parametrize('source,nlevels,humidity,endpoints', [('gfs',34,'no',7),('era5',38,'yes',7),('era5-cds',38,'no',7),('aifs',15,'yes',2)])
def test_named_region_connects_cull_init_source_boundary_and_products(tmp_path, source, nlevels, humidity, endpoints):
    # AIFS boundaries come from its own 6-hourly forecast leads since region-any-init
    # (3dc8b074): two endpoints over the 6 h window. GFS posts hourly to 120 h, so it has seven.
    result = plan(args(tmp_path, '--region','east-africa','--source',source))
    commands = result['commands']
    assert commands['init'][2] == source
    assert '/parent/' in commands['init'][commands['init'].index('--grid')+1]
    assert commands['lbc'][commands['lbc'].index('--nfglevels')+1] == str(nlevels)
    assert commands['lbc'][commands['lbc'].index('--use-spechumd')+1] == humidity
    assert '--lbc-dir' in commands['forecast']
    assert commands['forecast'][commands['forecast'].index('--init')+1].endswith('mesh.init.nc')
    assert len(result['boundary_endpoints']) == endpoints
    for command in [*commands.values(), *(e['argv'] for e in result['boundary_endpoints'])]:
        build_parser().parse_args(command)
    assert not (tmp_path/'run').exists()


def test_unseen_named_geometry_is_table_work(tmp_path):
    table = json.loads(REGIONS.read_text())
    row = table['countries'][0]
    row['id'] = 'unseen-area'
    row['geometry'] = {'kind':'polygon','vertices_deg':[[10,20],[10,30],[20,30],[20,20]]}
    path = tmp_path/'table.json'; path.write_text(json.dumps(table))
    result = plan(args(tmp_path,'--region','unseen-area','--table',str(path)))
    assert result['row']['geometry']['kind'] == 'polygon'
    assert result['commands']['cull'][0] == 'cull'


def test_arbitrary_cap_identity_and_products_follow_geometry(tmp_path):
    a = plan(args(tmp_path,'--point=-8,149','--radius-km','1200'))
    b = plan(args(tmp_path,'--point=-8,150','--radius-km','1200'))
    assert a['row']['id'] != b['row']['id']
    assert a['row']['geometry']['center_deg'] == [-8,149]
    assert 'region=region:-8.0,149.0,2400.0,2400.0' in a['commands']['products']


@pytest.mark.parametrize('point,radius',[('nan,10','100'),('0,0','nan'),('91,0','100'),('1,2','-1')])
def test_invalid_cap_refuses_before_transfer(tmp_path,point,radius):
    with pytest.raises(MpasPortError):
        plan(args(tmp_path,'--point',point,'--radius-km',radius))


def test_arbitrary_polygon(tmp_path):
    p = tmp_path/'polygon.json'; p.write_text(json.dumps({'kind':'polygon','vertices_deg':[[2,20],[2,30],[8,30]]}))
    result = plan(args(tmp_path,'--polygon',str(p)))
    assert result['row']['geometry']['vertices_deg'][0] == [2,20]
    assert '--region' in result['commands']['products']


def test_date_line_polygon_views_and_scorecard_follow_the_short_arc(tmp_path):
    path = tmp_path/'polygon.json'
    path.write_text(json.dumps({'kind':'polygon', 'vertices_deg':[[8,175],[8,-175],[16,-175],[16,175]]}))
    result = plan(args(tmp_path, '--polygon', str(path)))
    assert result['row']['mesh']['point_deg'] == [12, -180]
    box = result['scorecard']['region']
    assert box['east']-box['west'] == 10
    assert box['west'] == -185
    assert 'region=region:12.0,-180.0,' in ' '.join(result['commands']['products'])


def test_polar_cap_scorecard_never_spans_more_than_the_globe(tmp_path):
    result = plan(args(tmp_path, '--point', '89,170', '--radius-km', '1200'))
    box = result['scorecard']['region']
    assert box['east']-box['west'] == 360
    assert box['north'] == 90


def test_forecast_endpoints_cover_fractional_run_and_analysis_crosses_date(tmp_path):
    commands = source_endpoint_commands(source='gfs',cycle='2023-03-31T18',hours=6.25,out_dir=tmp_path)
    # GFS posts hourly to 120 h: one endpoint per hour through the first one past the run's end.
    assert [c['valid_time'] for c in commands] == ['2023-03-31_%02d:00:00' % h for h in range(18, 24)] + [
        '2023-04-01_00:00:00', '2023-04-01_01:00:00']
    commands = source_endpoint_commands(source='era5',cycle='2023-03-31T23',hours=1,out_dir=tmp_path)
    assert commands[-1]['argv'][commands[-1]['argv'].index('--cycle')+1] == '2023-04-01T00'
    assert commands[-1]['argv'][commands[-1]['argv'].index('--lead-hours')+1] == '0'


def test_endpoint_clock_and_digest_checked_before_boundary_publication(tmp_path):
    import hashlib
    commands = source_endpoint_commands(source='gfs',cycle='2023-03-31T12',hours=6,out_dir=tmp_path/'inputs')
    def runner(argv):
        root = Path(argv[argv.index('--out-dir')+1]); root.mkdir(parents=True)
        moment = next(c['valid_time'] for c in commands if c['argv']==argv)
        met = root/'met'; met.write_bytes(b'decoder stand-in')
        (root/'init-source-receipt.json').write_text(json.dumps({'valid_time':moment,'decode':{'met':str(met),'met_sha256':hashlib.sha256(met.read_bytes()).hexdigest()}}))
        return 0
    results = produce_source_endpoints(commands,tmp_path/'met',runner=runner)
    assert len(results)==7
    assert all(Path(r['met']).is_symlink() for r in results)
    assert json.loads((tmp_path/'met/source-endpoints.json').read_text())['workers']==2


def test_every_shipped_region_has_a_generator_admitted_resolution(tmp_path):
    from woof.hex.country_door import region_specification
    for row in json.loads(REGIONS.read_text())['countries']:
        value = plan(args(tmp_path, '--region', row['id']))
        specification = region_specification(value['row'])
        assert specification['regions'][-1]['spacing_km'] == row['mesh']['fine_dx_m']/1000


def test_a_drawn_polygon_takes_the_grown_ladder_at_the_asked_spacing(tmp_path):
    # The CONUS outline the 3 km region door was refused on: one region with the defaults' 540 km ramp never reached
    # 3 km inside it.  Now the fine rung is the outline grown by one 54 km ramp, and each coarser rung nests outside.
    from woof.hex.country_door import region_specification
    conus = {"kind": "polygon", "vertices_deg": [[49.5, -125.0], [49.5, -95.0], [49.0, -83.0], [47.5, -67.0],
             [43.5, -66.5], [35.0, -74.5], [30.5, -80.0], [24.5, -80.0], [24.5, -82.5], [29.0, -89.0], [25.5, -97.0],
             [29.0, -104.5], [31.3, -111.0], [32.5, -117.5], [40.0, -125.0]]}
    path = tmp_path / "conus.json"
    path.write_text(json.dumps(conus))
    value = plan(args(tmp_path, "--polygon", str(path), "--fine-dx-m", "3000", "--background-km", "96"))
    spec = region_specification(value["row"])
    assert [r["spacing_km"] for r in spec["regions"]] == [48.0, 24.0, 12.0, 6.0, 3.0]
    assert [r["transition_km"] for r in spec["regions"]] == [864.0, 432.0, 216.0, 108.0, 54.0]
    assert spec["background_km"] == 96.0 and all(r["shape"]["kind"] == "polygon" for r in spec["regions"])
    fine = spec["regions"][-1]["shape"]
    # grown outward: the fine outline's southern tip (Florida, drawn at 24.5N) moves about 54 km south
    assert min(v[0] for v in fine["vertices_deg"]) == pytest.approx(24.5 - 54.0 / 111.2, abs=0.08)


def test_region_rejects_disconnected_input_override_before_transfer(tmp_path):
    with pytest.raises(MpasPortError, match='overrides cannot identify'):
        plan(args(tmp_path, '--region', 'east-africa', '--init', '/another/init.nc'))


def test_existing_parent_is_revalidated_for_a_different_region(tmp_path, monkeypatch):
    from woof.hex import country_door, engines, prep_store, cli
    arguments = args(tmp_path, '--point=-8,149', '--radius-km', '1200')
    value = plan(arguments)
    parent = arguments.work_dir/'parent'
    parent.mkdir(parents=True)
    grid, static = parent/'mesh.grid.nc', parent/'mesh.static.nc'
    grid.write_bytes(b'grid for an earlier region')
    static.write_bytes(b'static for an earlier region')
    seen = []

    def prepare(argv):
        assert '--clobber' in argv
        seen.append(argv)
        output = Path(argv[argv.index('--out')+1])
        if output == grid:
            spec = json.loads(Path(argv[argv.index('--spec')+1]).read_text())
            assert spec['regions'][-1]['shape']['center_deg'] == [-8, 149]
            output.write_bytes(b'current grid')
        else:
            assert grid.read_bytes() == b'current grid'
            output.write_bytes(b'current static')

    monkeypatch.setattr(engines, 'resolve', lambda spec: Path('/rust')/spec.name)
    monkeypatch.setattr(prep_store, 'run_or_fetch', prepare)
    monkeypatch.setattr(cli, 'main', lambda command: 1)  # End after preparation, before field initialization.
    status = type('Status', (), {'note': lambda *a: None})()
    assert country_door._prepare_region(arguments, value, status) == 1
    assert len(seen) == 2
    assert static.read_bytes() == b'current static'


def test_source_endpoint_reuse_requires_current_table_digest(tmp_path):
    import hashlib
    from woof.hex.init_sources import load_table
    commands = source_endpoint_commands(source='gfs', cycle='2023-03-31T12', hours=6, out_dir=tmp_path/'inputs')
    invoked = []
    def runner(argv):
        invoked.append(argv)
        root = Path(argv[argv.index('--out-dir')+1]); root.mkdir(parents=True, exist_ok=True)
        item = next(c for c in commands if c['argv']==argv)
        met = root/'met'; met.write_bytes(b'decoder stand-in')
        (root/'init-source-receipt.json').write_text(json.dumps({
            'source':'gfs', 'source_cycle':'2023-03-31_12:00:00', 'member':None,
            'lead_hours':int(argv[argv.index('--lead-hours')+1]),
            'table':{'sha256':load_table()['_sha256']}, 'valid_time':item['valid_time'],
            'decode':{'met':str(met), 'met_sha256':hashlib.sha256(met.read_bytes()).hexdigest()}}))
        return 0
    produce_source_endpoints(commands, tmp_path/'met', runner=runner)
    assert len(invoked) == 7
    produce_source_endpoints(commands, tmp_path/'met', runner=runner)
    assert len(invoked) == 7
    receipt = tmp_path/'inputs/endpoint-0000/init-source-receipt.json'
    data = json.loads(receipt.read_text()); data['table']['sha256'] = 'old table'
    receipt.write_text(json.dumps(data))
    produce_source_endpoints(commands, tmp_path/'met', runner=runner)
    assert len(invoked) == 8


def test_cull_static_is_bound_to_the_final_child_grid(tmp_path, monkeypatch):
    from woof.hex import cull_door
    for name in ('parent.grid.nc','parent.static.nc','parent.init.nc','shape.json'):
        (tmp_path/name).write_text('{}')
    monkeypatch.setattr(cull_door,'resolve_mesh_engine',lambda _:Path('/rust/mesh'))
    seen = []
    def cull(engine,parent,region,out,**options):
        seen.append((out,options['provenance_grid']))
        return {'out':str(out),'seconds':0}
    monkeypatch.setattr(cull_door,'cull_one',cull)
    monkeypatch.setattr(cull_door,'carry_lineage',lambda *a,**k:{})
    command = ['cull','--parent-grid',str(tmp_path/'parent.grid.nc'),
        '--parent-static',str(tmp_path/'parent.static.nc'),'--parent-init',str(tmp_path/'parent.init.nc'),
        '--region',str(tmp_path/'shape.json'),'--out-dir',str(tmp_path),'--name','child']
    assert cull_door.run_cull(build_parser().parse_args(command)) == 0
    assert seen == [(tmp_path/'child.grid.nc',None),
                    (tmp_path/'child.static.nc',tmp_path/'child.grid.nc'),
                    (tmp_path/'child.init.nc',None)]


def test_point_products_follow_registered_region_instead_of_global_cities(tmp_path):
    result = plan(args(tmp_path, '--region', 'east-africa'))
    products = result['commands']['products']
    assert products[products.index('--points')+1] == 'east-africa@0,37'
    point = plan(args(tmp_path, '--point=-8,149', '--radius-km', '1200'))
    products = point['commands']['products']
    assert products[products.index('--points')+1].endswith('@-8.0,149.0')
    table = json.loads(REGIONS.read_text())
    table['countries'][0]['product_points'] = [{'name': 'regional station', 'point_deg': [1,38]}]
    path = tmp_path/'regions.json'; path.write_text(json.dumps(table))
    products = plan(args(tmp_path, '--region', 'east-africa', '--table', str(path)))['commands']['products']
    assert products[products.index('--points')+1] == 'regional station@1,38'


def test_only_the_initial_endpoint_takes_the_cold_land_donor(tmp_path):
    """WOOF 0.4.10 bake, 2026-10-30: GFS's cold-soil donor row is analysis-only, and the 1 h boundary endpoint of
    every GFS limited-area region asked for it and was refused ('boundary source endpoint ... exited 2')."""
    commands = source_endpoint_commands(source='gfs', cycle='2023-03-31T12', hours=3, out_dir=tmp_path)
    flags = [c['argv'][c['argv'].index('--soil-remap') + 1] if '--soil-remap' in c['argv'] else 'auto'
             for c in commands]
    assert flags == ['auto', 'off', 'off', 'off']
