from argparse import Namespace
from pathlib import Path
import pytest


def test_gfs_cold_door_keeps_cold_soil_without_donor_and_on_refuses(tmp_path):
    from woof.hex.init_door import prepare_soil_remap, InitDoorRefusal
    a=Namespace(land_surface='noah',soil_remap='auto',soil_donor=None,soil_contract=None)
    assert prepare_soil_remap(a,('NCEP GFS Analysis',)) is None
    assert 'soil-donor' in a.soil_remap_reason
    a.soil_remap='on'
    with pytest.raises(InitDoorRefusal,match='soil-donor'):
        prepare_soil_remap(a,('NCEP GFS Analysis',))
    a.soil_remap='auto'
    a.soil_donor=tmp_path/'donor.nc';a.soil_donor.touch()
    assert prepare_soil_remap(a,('NCEP GFS Analysis',))['id']=='gfs16-noah-to-mpas842-stas'
    a.soil_remap='off'
    assert prepare_soil_remap(a,('NCEP GFS Analysis',)) is None


@pytest.mark.parametrize('land', ['noahmp', 'ruc'])
def test_named_contract_on_another_land_model_keeps_cold_soil(tmp_path, land):
    # init-source names the table's contract and donor on every GFS init, so a named contract is no request to
    # refuse: the ArWen route's Noah-MP and hrrrv5's RUC start from their cold soil (WOOF 0.4.10 bake, 2026-10-30).
    from woof.hex.init_door import prepare_soil_remap, InitDoorRefusal
    donor=tmp_path/'donor.nc';donor.touch()
    a=Namespace(land_surface=land,soil_remap='auto',soil_donor=donor,soil_contract='gfs16-noah-to-mpas842-stas')
    assert prepare_soil_remap(a,('NCEP GFS Analysis',)) is None
    assert land in a.soil_remap_reason
    a.soil_remap='on'
    with pytest.raises(InitDoorRefusal,match="contract is Noah's"):
        prepare_soil_remap(a,('NCEP GFS Analysis',))


def test_other_source_policy_is_selected_only_by_table():
    from woof.hex.init_door import prepare_soil_remap
    a=Namespace(land_surface='noah')
    assert prepare_soil_remap(a,('unregistered-source',)) is None


def test_source_row_has_default_native_soil_product():
    from woof.hex.init_sources import load_table
    table=load_table()
    # Do not let a new source-specific branch supply this product.
    rows=[r for r in table['sources'] if r['id'] in ('gfs','gfs-gdex')]
    for r in rows:
        assert r['cold_soil']['contract']=='gfs16-noah-to-mpas842-stas'
        p=[p for p in r['products'] if p['role']==r['cold_soil']['product_role']]
        assert len(p)==1 and p[0]['transport']=='https'


@pytest.mark.parametrize('profile', [None, 'hrrr-like-aerosol'])
@pytest.mark.parametrize('resume', [False, True])
@pytest.mark.parametrize('mode', ['auto', 'off'])
def test_native_donor_is_separate_from_decode_and_survives_resume(tmp_path, monkeypatch, resume, mode, profile):
    import argparse
    import json
    from woof.hex import init_sources as src
    parser = argparse.ArgumentParser()
    src.add_init_source_parser(parser.add_subparsers())
    args = parser.parse_args(['init-source', '--source', 'gfs', '--cycle', '2023-07-08T12',
        '--out-dir', str(tmp_path), '--grid', 'grid', '--static', 'static',
        '--vertical-spec', 'vertical', '--stop-after', 'decode', '--soil-remap', mode])
    args.physics_profile = profile
    monkeypatch.setattr(src, 'resolve_writer', lambda path: tmp_path / 'writer')
    monkeypatch.setattr(src, 'resolve_vtable', lambda row: tmp_path / 'vtable')
    calls = []
    def acquire(plans, *a, **kw):
        calls.append('acquire')
        result = []
        for plan in plans:
            path = tmp_path / plan.filename
            path.write_bytes(plan.role.encode())
            result.append(dict(role=plan.role, file=str(path), sha256=src._sha256_file(path)))
        return result
    def decode(row, inputs, *a):
        calls.append('decode')
        assert all('sfcanl' not in path for path in inputs)
        met = tmp_path / 'MET'
        met.write_bytes(b'met')
        return dict(met=str(met), met_sha256=src._sha256_file(met))
    captured = []
    monkeypatch.setattr(src, 'acquire', acquire)
    monkeypatch.setattr(src, 'decode', decode)
    monkeypatch.setattr(src, 'run_init', lambda argv: captured.append(argv) or 0)
    if resume:
        assert src.run_init_source_command(args) == 0
        args.from_decoded = True
    args.stop_after = 'init'
    assert src.run_init_source_command(args) == 0
    # The donor is acquired on its own after the source's products (a donor the archive lacks keeps the cold soil).
    assert calls == ['acquire'] * (2 if mode == 'auto' else 1) + ['decode']
    argv = captured[0]
    if profile:
        assert argv[argv.index('--physics-profile') + 1] == profile
    else:
        assert '--physics-profile' not in argv
    assert argv[argv.index('--soil-remap') + 1] == mode
    assert ('--soil-donor' in argv) == (mode == 'auto')
    if mode == 'auto':
        assert Path(argv[argv.index('--soil-donor') + 1]).read_bytes() == b'cold-soil-donor'


@pytest.mark.parametrize('case', ['lead', 'early', 'missing'])
@pytest.mark.parametrize('mode', ['auto', 'on'])
def test_init_source_keeps_cold_soil_when_the_donor_cannot_serve(tmp_path, monkeypatch, case, mode):
    # A forecast-hour start, a cycle before GFS v16 wrote the NetCDF analysis, and a donor the archive lacks each
    # start from the source's own soil under auto (receipt and log say why) and refuse under on.
    import argparse
    import json
    from woof.hex import init_sources as src
    parser = argparse.ArgumentParser()
    src.add_init_source_parser(parser.add_subparsers())
    cycle = '2020-07-08T12' if case == 'early' else '2026-10-30T12'
    argv = ['init-source', '--source', 'gfs', '--cycle', cycle, '--out-dir', str(tmp_path), '--grid', 'grid',
            '--static', 'static', '--vertical-spec', 'vertical', '--soil-remap', mode]
    if case == 'lead':
        argv += ['--lead-hours', '3']
    args = parser.parse_args(argv)
    monkeypatch.setattr(src, 'resolve_writer', lambda path: tmp_path / 'writer')
    monkeypatch.setattr(src, 'resolve_vtable', lambda row: tmp_path / 'vtable')
    def acquire(plans, *a, **kw):
        result = []
        for plan in plans:
            if case == 'missing' and plan.role == 'cold-soil-donor':
                src._refuse(f'{plan.filename}: no address has it (noaa-gfs-bdp-pds.s3.amazonaws.com: HTTP 404).')
            path = tmp_path / plan.filename
            path.write_bytes(plan.role.encode())
            result.append(dict(role=plan.role, file=str(path), sha256=src._sha256_file(path)))
        return result
    def decode(row, inputs, *a):
        met = tmp_path / 'MET'
        met.write_bytes(b'met')
        return dict(met=str(met), met_sha256=src._sha256_file(met))
    captured = []
    monkeypatch.setattr(src, 'acquire', acquire)
    monkeypatch.setattr(src, 'decode', decode)
    monkeypatch.setattr(src, 'run_init', lambda argv: captured.append(argv) or 0)
    if mode == 'on':
        with pytest.raises(src.InitSourceRefusal, match='--soil-remap on'):
            src.run_init_source_command(args)
        return
    assert src.run_init_source_command(args) == 0
    init_argv = captured[0]
    assert init_argv[init_argv.index('--soil-remap') + 1] == 'off'
    assert '--soil-donor' not in init_argv
    receipt = json.loads((tmp_path / 'init-source-receipt.json').read_text())
    assert receipt['cold_soil_remap_off']
