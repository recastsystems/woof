"""Loaded init balance: native force, mass, public switch and actual file completion."""
import argparse
import hashlib

import numpy as np
import pytest
from netCDF4 import Dataset

from woof.hex.init_balance import (CONDENSATE, G, RD, RV, CP, balance_columns,
                                  cold_pressure, complete_init)
from woof.hex.init_door import add_init_parser


def column_inputs():
    z = np.array([50., 250., 750., 1800.])
    pb = 1.e5 * np.exp(-G*z/(RD*250.))
    rho = np.broadcast_to(pb/(RD*250.), (3,4)).copy()
    theta = np.broadcast_to(250.*(1.e5/pb)**(RD/CP), (3,4)).copy()
    q = np.zeros_like(rho)
    cond = np.zeros_like(rho);cond[0,1]=.003;cond[1,2]=.004
    return dict(rho=rho,theta=theta,qv=q,condensate=cond,zz=np.ones_like(rho),
                rho_base=rho.copy(),theta_base=theta.copy(),
                fzm=np.full(4,.5),fzp=np.full(4,.5),rdzu=np.r_[0.,1./np.diff(z)],rdzw0=.01)


def test_exact_column_force_and_fixed_mass():
    d=column_inputs();p,t,s=balance_columns(**d)
    qt=d['qv']+d['condensate'];r=d['rho']/d['zz'];rb=d['rho_base']
    pb=cold_pressure(rb*d['zz'],d['theta_base'],0.)
    b=-G*(rb*qt+(r-rb)*(1+qt))
    force=np.diff((p-pb)[:2],axis=1)*d['rdzu'][1:]
    force-=d['fzm'][1:]*b[:2,1:]+d['fzp'][1:]*b[:2,:-1]
    assert np.max(abs(force))<1.e-10
    np.testing.assert_allclose(cold_pressure(d['rho'],t,d['qv']),p,rtol=2.e-15)
    assert np.array_equal(t[2],d['theta'][2])
    assert np.array_equal(p[:,-1],cold_pressure(d['rho'],d['theta'],d['qv'])[:,-1])
    assert np.all(t[:2,:-1]>=d['theta'][:2,:-1]-1.e-12)


def make_file(path,species,dtype='f8'):
    d=column_inputs()
    with Dataset(path,'w') as ds:
        ds.createDimension('Time',1);ds.createDimension('nCells',3);ds.createDimension('nVertLevels',4)
        shape=('Time','nCells','nVertLevels')
        for n in ('rho','theta','qv','zz','rho_base','theta_base'):
            ds.createVariable(n,dtype,shape)[0]=d[n]
        for n in CONDENSATE:
            ds.createVariable(n,dtype,shape)[0]=d['condensate'] if n==species else 0.
        for n in ('fzm','fzp','rdzu'):
            ds.createVariable(n,dtype,('nVertLevels',))[:]=d[n]
        ds.createVariable('rdzw',dtype,('nVertLevels',))[:]=d['rdzw0']
        pb=cold_pressure(d['rho_base']*d['zz'],d['theta_base'],0.)
        ds.createVariable('pressure_base',dtype,shape)[0]=pb
        ds.createVariable('pressure_p',dtype,shape)[0]=0.
        ds.createVariable('pressure',dtype,shape)[0]=pb
        ds.createVariable('surface_pressure',dtype,('Time','nCells'))[0]=1.e5
        # Number must not be added to moist mass.
        ds.createVariable('ng',dtype,shape)[0]=1.e8
    return d


@pytest.mark.parametrize('species',CONDENSATE)
def test_writer_carries_each_mass_species_but_not_numbers(tmp_path,species):
    path=tmp_path/'init.nc';d=make_file(path,species)
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    assert complete_init(path,enabled=False)['enabled'] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    receipt=complete_init(path,block_size=1)
    assert receipt['columns']==2
    p,t,s=balance_columns(**d)
    with Dataset(path) as ds:
        np.testing.assert_array_equal(ds['theta'][0],t)
        np.testing.assert_array_equal(ds['rho'][0],d['rho'])
        np.testing.assert_array_equal(ds[species][0],d['condensate'])
        np.testing.assert_array_equal(ds['pressure'][0,:2],p[:2])
        np.testing.assert_array_equal(ds['surface_pressure'][0,:2],s[:2])
        assert ds['surface_pressure'][0,2]==1.e5  # clear column never written
        assert np.all(ds['ng'][0]==1.e8)


def test_zero_condensate_is_byte_identical(tmp_path):
    path=tmp_path/'clear.nc';make_file(path,'none')
    before=path.read_bytes()
    assert complete_init(path)['columns']==0
    assert path.read_bytes()==before


def test_disabled_does_not_even_open_file(tmp_path):
    assert complete_init(tmp_path/'absent.nc',enabled=False)['columns']==0


def test_default_and_comparison_switch():
    parser=argparse.ArgumentParser();add_init_parser(parser.add_subparsers())
    assert parser.parse_args(['init']).condensate_balance is None
    assert parser.parse_args(['init','--condensate-balance','no']).condensate_balance=='no'


def test_invalid_mass_cannot_enter_balance():
    d=column_inputs();d['condensate'][0,0]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):
        balance_columns(**d)


def test_repeated_completion_is_stable(tmp_path):
    path=tmp_path/'loaded.nc';make_file(path,'qg')
    complete_init(path)
    with Dataset(path) as ds: first=np.asarray(ds['theta'][0])
    complete_init(path)
    with Dataset(path) as ds: second=np.asarray(ds['theta'][0])
    np.testing.assert_allclose(second,first,atol=3.e-13,rtol=0)


@pytest.mark.parametrize('enabled',(True,False))
def test_standard_init_door_uses_balance_after_producer(tmp_path,monkeypatch,enabled):
    """Mock only interpolation; the door, completion, NetCDF writes and digest are real."""
    import json
    from types import SimpleNamespace
    from woof.hex import init_door, init_physics, hrrr_intermediate, numerics
    # The battery runs strict (conftest); the default yes is the fast-numerics door's.
    monkeypatch.setattr(numerics,'is_strict',lambda:False)
    from test_init_door import COMPLETE_SWITCHES

    parser=argparse.ArgumentParser();add_init_parser(parser.add_subparsers())
    paths={n:tmp_path/n for n in ('met','static','capsule','reference','engine')}
    for path in paths.values():path.write_bytes(b'fixture')
    out=tmp_path/'init.nc'
    argv=['init','--out',str(out),'--soil-remap','off',*COMPLETE_SWITCHES]
    for n,path in paths.items():argv.extend(['--'+n,str(path)])
    if not enabled:argv.extend(['--condensate-balance','no'])
    args=parser.parse_args(argv)
    monkeypatch.setattr(init_physics,'init_selection',lambda *a,**kw:
                        dict(producer=None,land_surface='noah',scalar_names=()))
    monkeypatch.setattr(hrrr_intermediate,'uncovered_cells_refusal',lambda *a:None)
    monkeypatch.setattr(init_door,'resolve_met_source',lambda p:p)
    monkeypatch.setattr(init_door,'scan_met_file',lambda *a,**kw:
                        SimpleNamespace(fields=[],map_sources=[],valid_times=[],upper_air_levels=4))
    monkeypatch.setattr(init_door,'resolve_engine',lambda p:p)
    monkeypatch.setattr(init_door,'first_guess_hydrometeors',lambda *a:{'argv':[]})
    monkeypatch.setattr(init_door,'prepare_vertical_source',lambda a:
                        SimpleNamespace(mode='compatibility',lock=None,summary={},
                                        capsule=paths['capsule'],reference=paths['reference']))
    monkeypatch.setattr(init_door,'host_record',lambda:{'test':'local'})
    original_run=init_door.subprocess.run

    def producer(command,**kw):
        if command[0]!=str(paths['engine']):return original_run(command,**kw)
        make_file(out,'qg')
        with Dataset(out,'r+') as ds:ds.createDimension('nSoilLevels',4)
        return SimpleNamespace(returncode=0,stdout='{}',stderr='')

    monkeypatch.setattr(init_door.subprocess,'run',producer)
    assert init_door.run_init_command(args)==0
    receipt=json.loads(out.with_name(out.name+'.provenance.json').read_text())
    assert receipt['condensate_balance']['enabled'] is enabled
    assert receipt['output']['sha256']==hashlib.sha256(out.read_bytes()).hexdigest()
    d=column_inputs()
    expected=balance_columns(**d)[1] if enabled else d['theta']
    with Dataset(out) as ds:np.testing.assert_array_equal(ds['theta'][0],expected)


def test_balance_defaults_yes_fast_and_no_strict(monkeypatch):
    """Strict numerics is the NOAA/MPAS-A match: the producer's own vapour-only balance stays."""
    from woof.hex import numerics
    from woof.hex.init_balance import balance_choice
    monkeypatch.setattr(numerics,'is_strict',lambda:False)
    assert balance_choice(None)==(True,'default')
    monkeypatch.setattr(numerics,'is_strict',lambda:True)
    assert balance_choice(None)[0] is False
    assert 'strict' in balance_choice(None)[1]
    assert balance_choice('yes')==(True,'--condensate-balance')
    monkeypatch.setattr(numerics,'is_strict',lambda:False)
    assert balance_choice('no')==(False,'--condensate-balance')


def test_producer_negative_vapour_is_balanced_not_refused():
    """GFS 2024-05-21 12Z: the producer left qv down to -6.9e-6; the balance takes it as the forecast will."""
    d=column_inputs();d['qv'][0,2]=-6.9e-6
    p,t,s=balance_columns(**d)
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(t))
    assert np.array_equal(t[2],d['theta'][2])


def test_vapour_breaking_the_equation_of_state_is_refused():
    d=column_inputs();d['qv'][0,2]=-1.0
    with pytest.raises(ValueError,match='equation of state'):
        balance_columns(**d)


def test_refused_balance_removes_the_half_written_init(tmp_path,monkeypatch):
    """A refusal after some blocks were written must not leave a half-balanced init behind."""
    from types import SimpleNamespace
    from woof.hex import init_balance
    from woof.hex.mpasphys.rrfs import cold_start_rrfs as cs
    out=tmp_path/'init.nc';make_file(out,'qg')
    def refuse(path,*,enabled):
        raise ValueError('loaded hydrostatic balance produced invalid pressure')
    monkeypatch.setattr(init_balance,'complete_init',refuse)
    monkeypatch.setattr(cs,'producer_for_suite',lambda s:('rrfs',{'soil':{}}))
    monkeypatch.setattr(cs,'static_missing_fields',lambda *a:[])
    monkeypatch.setattr(cs,'resolve_worker',lambda r:tmp_path/'worker')
    monkeypatch.setattr(cs,'resolve_aerosol_table',lambda a:tmp_path/'aero')
    monkeypatch.setattr(cs,'case7_jobs',lambda *a,**k:[None])
    monkeypatch.setattr(cs,'run_jobs',lambda *a:[])
    monkeypatch.setattr(cs,'_refuse_cold_land_soil',lambda *a:None)
    import woof.hex.soil_handover as sh
    monkeypatch.setattr(sh,'handover_or_fork',lambda met,dest,**k:(met,4,{'applied':False}))
    if not hasattr(cs,'init_on_template'):
        pytest.skip('this line has no init_on_template')
    with pytest.raises(cs.RrfsInitError,match='incomplete file was removed'):
        cs.init_on_template(suite='hrrrv5',template=tmp_path/'t.nc',met=tmp_path/'met',out=out,
                            start_time='2024-05-21_12:00:00',nfglevels=34,nfgsoillevels=4,condensate_balance=True)
    assert not out.exists()
