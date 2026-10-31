import json
from pathlib import Path
import numpy as np
import pytest
from netCDF4 import Dataset
from woof.hex.cycle.soil_wetness import source_contract,remap_cold_init,SoilRemapNotApplied
from woof.hex.cycle.land_state import context,read,validate_land,LAND_ATTRIBUTE


def pair(tmp_path,valid='2023-07-08 12:00:00',ocean=False):
    tmp_path.mkdir(parents=True,exist_ok=True)
    target=tmp_path/'init.nc'; donor=tmp_path/'donor.nc'
    with Dataset(target,'w',format='NETCDF3_64BIT_DATA') as d:
        for n,size in [('Time',1),('nCells',3),('nSoilLevels',4),('StrLen',64)]:d.createDimension(n,size)
        land={'isltyp':[14,14,14], 'ivgtyp':[17,17,17], 'xland':[2,2,2]} if ocean else {'isltyp':[1,6,14], 'ivgtyp':[2,13,17], 'xland':[1,1,2]}
        for n,v in {**land, 'xice':[0,0,0], 'latCell':[0,0,0], 'lonCell':[0,0,0]}.items():
            d.createVariable(n,'f4',('nCells',))[:]=v
        d.createVariable('isice_lu','i4').assignValue(15)
        d.createVariable('mminlu','S1',('StrLen',))[:]=np.array(list('MODIFIED_IGBP_MODIS_NOAH'.ljust(64)),dtype='S1')
        d.createVariable('dzs','f4',('nSoilLevels',))[:]=[.1,.3,.6,1]
        for n,vals in {'smois':[.42,.3,1], 'sh2o':[.2,.1,1], 'tslb':[280,281,282]}.items():
            d.createVariable(n,'f4',('Time','nCells','nSoilLevels'))[0]=np.array(vals)[:,None]
    with Dataset(donor,'w') as d:
        for n,size in [('time',1),('y',2),('x',2)]:d.createDimension(n,size)
        # GFS v16's own header facts (gfs.tHHz.sfcanl.nc, 2026-10-30 18Z): what the contract is keyed on.
        d.setncattr('nsoil',np.int32(4));d.source='FV3GFS';d.grid='gaussian'
        t=d.createVariable('time','f8',('time',));t.units=f'hours since {valid}';t[:]=0
        for n,dim in [('grid_yt','y'),('grid_xt','x')]:d.createVariable(n,'f8',(dim,))[:]=[0,.1]
        fields={'land':1,'vtype':2,'sotyp':6,'veg':50}
        for n in range(1,5):fields.update({f'soilw{n}':.3,f'soill{n}':.15,f'soilt{n}':285})
        layers=['0-10cm','10-40cm','40-100cm','100-200cm']
        for n,value in fields.items():
            v=d.createVariable(n,'f4',('time','y','x'));v[:]=value
            if n[:5] in ('soilw','soilt'):
                v.long_name=('volumetric soil moisture ' if n[:5]=='soilw' else 'soil temperature ')+layers[int(n[5])-1]
    return target,donor


def test_cold_state_remaps_hydraulics_and_liquid_fraction_with_receipt(tmp_path):
    target,donor=pair(tmp_path)
    receipt=remap_cold_init(target,donor,source_contract(['NCEP GFS Analysis']),'2023-07-08_12:00:00')
    assert receipt['cold_before_bound_violations']==4
    assert receipt['cold_after_bound_violations']==0 and receipt['clipped_values']==0
    assert receipt['source_rejected_columns']==0
    with Dataset(target) as d:
        fields={n:read(d[n]) for n in ('smois','sh2o','tslb')}
        validate_land(fields,context(d))
        np.testing.assert_allclose(fields['sh2o'][:2],fields['smois'][:2]*.5,rtol=0,atol=0)
        np.testing.assert_array_equal(fields['smois'][2],1)
        np.testing.assert_array_equal(fields['sh2o'][2],1)
        np.testing.assert_array_equal(fields['tslb'],np.broadcast_to([[280],[281],[282]],(3,4)))
        assert not d.ncattrs()  # do not rewrite a large classic header to add provenance
    assert json.loads(Path(str(target)+'.soil-remap.json').read_text())==receipt


def test_contract_has_no_calendar_end(tmp_path):
    # THE DEFECT (2026-10-31): valid_through 2026-10-28 refused every later GFS init on Noah. An unchanged GFS
    # analysis from after that date is remapped like any other.
    target,donor=pair(tmp_path,valid='2026-10-30 12:00:00')
    contract=source_contract(['NCEP GFS Analysis'])
    assert 'valid_through' not in contract and 'valid_from' not in contract
    receipt=remap_cold_init(target,donor,contract,'2026-10-30_12:00:00')
    assert receipt['applied'] is True and receipt['cold_after_bound_violations']==0


@pytest.mark.parametrize('bad',['time','classes','nsoil','source','noahmp','ruc_layers','layer_depth','vtype','missing'])
def test_donor_that_is_not_the_contract_is_not_applied_and_writes_nothing(tmp_path,bad):
    target,donor=pair(tmp_path);contract=source_contract(['NCEP GFS Analysis'])
    with Dataset(donor,'r+') as d:
        if bad=='time':d['time'][:]=1
        elif bad=='classes':d['sotyp'][:]=6.2
        elif bad=='nsoil':d.setncattr('nsoil',np.int32(9))
        elif bad=='source':d.source='GDAS-other'
        elif bad=='noahmp':d.createVariable('zwtxy','f4',('time','y','x'))[:]=2.5
        elif bad=='ruc_layers':d.createVariable('soilw5','f4',('time','y','x'))[:]=.3
        elif bad=='layer_depth':d['soilw2'].long_name='volumetric soil moisture 10-30cm'
        elif bad=='vtype':d['vtype'][:]=21
        elif bad=='missing':d.renameVariable('soill4','soill4_old')
    before=target.read_bytes()
    with pytest.raises(SoilRemapNotApplied):remap_cold_init(target,donor,contract,'2023-07-08_12:00:00')
    assert before==target.read_bytes()
    assert not Path(str(target)+'.soil-remap.json').exists()


@pytest.mark.parametrize('case',['distance','warm','ocean'])
def test_nothing_to_remap_passes_cleanly_and_writes_nothing(tmp_path,case):
    target,donor=pair(tmp_path,ocean=case=='ocean');contract=source_contract(['NCEP GFS Analysis'])
    if case=='warm':
        with Dataset(target,'r+') as d:d.setncattr(LAND_ATTRIBUTE,'["smois"]')
    elif case=='distance':contract['max_donor_distance_km']=-1
    before=target.read_bytes()
    receipt=remap_cold_init(target,donor,contract,'2023-07-08_12:00:00')
    assert receipt['applied'] is False and receipt['reason']
    assert before==target.read_bytes()
    assert not Path(str(target)+'.soil-remap.json').exists()


def door_run(tmp_path,monkeypatch,mode='auto',valid='2023-07-08 12:00:00',start='2023-07-08_12:00:00'):
    import argparse
    from woof.hex import init_door as door
    target,donor=pair(tmp_path,valid=valid);seed=tmp_path/'seed.nc';target.rename(seed)
    engine=tmp_path/'engine'
    engine.write_text('#!/usr/bin/env python3\nimport sys,shutil\n'
                      f'shutil.copyfile({str(seed)!r},sys.argv[sys.argv.index("--out")+1])\n'
                      'print("{\\"n_cells\\":3,\\"n_vert_levels\\":2}")\n')
    engine.chmod(0o700)
    parser=argparse.ArgumentParser();door.add_init_parser(parser.add_subparsers())
    args=parser.parse_args(['init','--met',str(donor),'--static',str(seed),
        '--capsule',str(seed),'--reference',str(seed),'--out',str(target),
        '--start-time',start,'--nfglevels','34','--nfgsoillevels','4',
        '--extrap-airtemp','constant','--use-spechumd','no','--theta-adv-order','3',
        '--coef-3rd-order','0.25','--virtual-factor','reproduce-fortran',
        '--deep-soil-moisture','reproduce-fortran','--landuse-table','MODIFIED_IGBP_MODIS_NOAH',
        '--frac-seaice','yes','--tsk-seaice-threshold','100','--oned-underflow','preserve',
        '--engine',str(engine),'--soil-donor',str(donor),'--soil-remap',mode])
    monkeypatch.setattr(door,'scan_met_file',lambda *a,**kw:door.MetInventory(
        ('SM000010',),(start,),34,True,('NCEP GFS Analysis',)))
    monkeypatch.setattr(door,'prepare_vertical_source',lambda *a:door.VerticalSource(
        mode='compatibility-capsule',capsule=seed,reference=seed,summary={}))
    return door,args,target


def test_init_door_runs_default_remap_before_output_receipt(tmp_path,monkeypatch):
    door,args,target=door_run(tmp_path,monkeypatch)
    assert door.run_init_command(args)==0
    receipt=json.loads(Path(str(target)+'.provenance.json').read_text())
    assert receipt['cold_soil_remap']['applied'] is True
    assert receipt['cold_soil_remap']['cold_before_bound_violations']==4
    with Dataset(target) as d:
        validate_land({n:read(d[n]) for n in ('smois','sh2o','tslb')},context(d))


def test_init_door_keeps_cold_soil_when_the_donor_does_not_fit_and_on_refuses(tmp_path,monkeypatch):
    # The donor's clock differs from the init's: the remap does not apply. auto keeps the complete cold init and
    # names why; on removes it and refuses.
    door,args,target=door_run(tmp_path/'auto',monkeypatch,valid='2026-10-30 06:00:00',start='2026-10-30_12:00:00')
    assert door.run_init_command(args)==0
    receipt=json.loads(Path(str(target)+'.provenance.json').read_text())['cold_soil_remap']
    assert receipt['applied'] is False and 'differs' in receipt['reason']
    assert target.is_file() and not Path(str(target)+'.soil-remap.json').exists()
    door,args,target=door_run(tmp_path/'on',monkeypatch,mode='on',valid='2026-10-30 06:00:00',start='2026-10-30_12:00:00')
    with pytest.raises(door.InitDoorRefusal,match='--soil-remap on'):
        door.run_init_command(args)
    assert not target.exists()
