"""Met_em contracts through independent NetCDF files and the Rust decoder."""
from datetime import datetime, timedelta
from dataclasses import replace
from pathlib import Path
import os

import numpy as np
import pytest

from woof.ingest.metem import (
    MetgridRefusal, read_met_em, parse_met_em_name, met_em_series, check_met_em_series,
)
from woof.static.lambert import LambertGrid
from conftest import requires_netcdf_bridge

nc = pytest.importorskip("netCDF4")
# The gate is the CAPABILITY, not the file: every case here reads a WRF
# met_em through the bridge, and a bridge that is present and too old to
# read a character array failed all 34 of them with a decode error rather
# than skipping.  `find_netcdf_bin() is None` could not see that.
pytestmark = requires_netcdf_bridge


def write_case(path, *, time="2021-06-01_00:00:00", rh_units="%", rh=55.0,
               global_override=None, mutate=None, specific=False):
    ny, nx, nz = 5, 5, 4
    grid = LambertGrid(ref_lat=38, ref_lon=-97, truelat1=30, truelat2=60,
                       stand_lon=-97, dx=12000, dy=12000, e_we=nx+1, e_sn=ny+1)
    with nc.Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        for name, size in (("Time",1),("DateStrLen",19),("west_east",nx),("south_north",ny),
                           ("west_east_stag",nx+1),("south_north_stag",ny+1),("num_metgrid_levels",nz)):
            ds.createDimension(name,size)
        ds.createVariable("Times","S1",("Time","DateStrLen"))[0] = np.asarray(list(time),dtype="S1")
        attrs = dict(DX=12000., DY=12000., MAP_PROJ=1, TRUELAT1=30., TRUELAT2=60.,
                     STAND_LON=-97., CEN_LAT=38., CEN_LON=-97., MOAD_CEN_LAT=38.,
                     MMINLU="MODIFIED_IGBP_MODIS_NOAH", NUM_LAND_CAT=21, ISWATER=17,
                     ISLAKE=21, ISICE=15, ISURBAN=13, ISOILWATER=14, grid_id=1,
                     **{"WEST-EAST_GRID_DIMENSION":nx+1,"SOUTH-NORTH_GRID_DIMENSION":ny+1})
        attrs.update(global_override or {})
        ds.setncatts(attrs)
        def add(name, value, units="", stagger="M", vertical=False):
            y = "south_north_stag" if stagger=="V" else "south_north"
            x = "west_east_stag" if stagger=="U" else "west_east"
            dims = ("Time",)+(("num_metgrid_levels",) if vertical else ())+ (y,x)
            v=ds.createVariable(name,"f4",dims)
            v[0]=value
            v.units=units
        pres=np.broadcast_to(np.array([98000.,95000.,70000.,20000.])[:,None,None],(nz,ny,nx))
        add("PRES",pres,vertical=True)
        add("TT",np.broadcast_to(np.array([290.,285.,270.,215.])[:,None,None],pres.shape),"K",vertical=True)
        add("RH",np.full(pres.shape,rh),rh_units,vertical=True)
        add("GHT",np.broadcast_to(np.array([0.,500.,3000.,12000.])[:,None,None],pres.shape),"m",vertical=True)
        add("UU",np.full((nz,ny,nx+1),5.),"m s-1","U",True)
        add("VV",np.full((nz,ny+1,nx),-3.),"m s-1","V",True)
        add("PSFC",pres[0],"Pa")
        for name in ("SOILHGT","HGT_M"): add(name,np.zeros((ny,nx)),"m")
        add("LANDMASK",np.ones((ny,nx)))
        if specific:
            ds.FLAG_SH=1
            add("SPECHUMD",np.full(pres.shape,.005),"kg kg-1",vertical=True)
        for stagger, coord, mf in (("M",grid.latlon_mass,grid.mapfac_m),("U",grid.latlon_u,grid.mapfac_u),("V",grid.latlon_v,grid.mapfac_v)):
            lat,lon=coord()
            add("XLAT_"+stagger,lat,"degrees_north",stagger)
            add("XLONG_"+stagger,lon,"degrees_east",stagger)
            add("MAPFAC_"+stagger,mf(),"",stagger)
        if mutate: mutate(ds)
    return path


def case(tmp_path, **kw):
    return write_case(tmp_path/"met_em.d01.2021-06-01_00_00_00.nc",**kw)


def test_pressure_humidity_field_and_surface_contract(tmp_path):
    c=read_met_em(case(tmp_path))
    assert c.lane=="rh" and c.shape==(5,5) and c.surface_level==0
    np.testing.assert_array_equal(c.snapshot.levels_hpa,[950,700,200])
    assert c.snapshot.fields["UU"].shape==(3,5,6)
    assert c.snapshot.fields["V10"].shape==(6,5)
    assert c.snapshot.fields["RH2"][0,0]==55


@pytest.mark.parametrize("name", ["QNI", "QNC", "QNR", "QNS", "QNG", "QNH"])
def test_flagged_number_stack_retains_original_surface_and_upper_levels(tmp_path, name):
    values = np.arange(100, dtype=np.float32).reshape(4,5,5) + 3
    def add(ds):
        variable = ds.createVariable(name, "f4", ("Time","num_metgrid_levels","south_north","west_east"))
        variable.units = "# kg-1"
        variable[0] = values
        ds.setncattr("FLAG_"+name, 1)
    c = read_met_em(case(tmp_path, mutate=add, global_override={"FLAG_PSFC":1,"FLAG_SOILHGT":1}))
    np.testing.assert_array_equal(c.snapshot.fields[name], values[1:])
    np.testing.assert_array_equal(c.snapshot.fields[name+"_SFC"], values[0])
    assert c.field_map[name+"[0]"] == name+"_SFC"
    from woof.metem_door import metgrid_initialization_controls
    from types import SimpleNamespace
    controls = metgrid_initialization_controls(c, SimpleNamespace(controls={"domains":{"sfcp_to_sfcp":[True]}}))
    assert controls["analyzed_number_fields"] == (name,)


@pytest.mark.parametrize("units", ["", "kg kg-1", "m-3"])
def test_number_quantity_units_cannot_be_guessed(tmp_path, units):
    def add(ds):
        variable = ds.createVariable("QNI", "f4", ("Time","num_metgrid_levels","south_north","west_east"))
        variable.units = units
        variable[0] = 1
        ds.FLAG_QNI = 1
    with pytest.raises(MetgridRefusal, match="number per kilogram"):
        read_met_em(case(tmp_path, mutate=add))


@pytest.mark.parametrize("name", ["QC", "QR", "QI", "QS", "QG", "QH"])
def test_flagged_mass_stack_retains_supplied_surface(tmp_path, name):
    values = np.arange(100, dtype=np.float32).reshape(4,5,5) * np.float32(1e-6)
    def add(ds):
        variable = ds.createVariable(name, "f4", ("Time","num_metgrid_levels","south_north","west_east"))
        variable.units = "kg kg-1"
        variable[0] = values
        ds.setncattr("FLAG_"+name, 1)
    c = read_met_em(case(tmp_path, mutate=add, global_override={"FLAG_PSFC":1,"FLAG_SOILHGT":1}))
    np.testing.assert_array_equal(c.snapshot.fields[name], values[1:])
    np.testing.assert_array_equal(c.snapshot.fields[name+"_SFC"], values[0])
    from woof.metem_door import metgrid_initialization_controls
    from types import SimpleNamespace
    controls = metgrid_initialization_controls(c, SimpleNamespace(controls={"domains":{"sfcp_to_sfcp":[True]}}))
    assert controls["analyzed_species"] == controls["analyzed_surface_fields"] == (name,)


def test_declared_number_stack_must_exist_and_have_valid_surface(tmp_path):
    with pytest.raises(MetgridRefusal, match="FLAG_QNI=1 but QNI is absent"):
        read_met_em(case(tmp_path, global_override={"FLAG_QNI":1}))
    def add(ds):
        variable = ds.createVariable("QNI", "f4", ("Time","num_metgrid_levels","south_north","west_east"))
        variable.units = "# kg-1"
        variable[0] = 1
        variable[0,0,0,0] = -1
        ds.FLAG_QNI = 1
    with pytest.raises(MetgridRefusal, match="negative number concentration"):
        read_met_em(case(tmp_path, mutate=add))


def test_fractional_rh_is_converted_in_rust_with_independent_oracle(tmp_path):
    path=case(tmp_path,rh_units="1",rh=.55)
    c=read_met_em(path)
    with nc.Dataset(path) as ds:
        wanted=(ds["RH"][0].astype(np.float64)*100).astype(np.float32)
    np.testing.assert_array_equal(c.snapshot.fields["RH"],wanted[1:])
    np.testing.assert_array_equal(c.snapshot.fields["RH2"],wanted[0])
    assert "Rust" in c.notes[0]


@pytest.mark.parametrize("units",["", "unknown", "kg kg-1"])
def test_rh_with_unknown_quantity_units_refused(tmp_path,units):
    with pytest.raises(MetgridRefusal,match="RH: units"):
        read_met_em(case(tmp_path,rh_units=units))


def test_physically_dry_percent_is_not_guessed_to_be_fraction(tmp_path):
    c=read_met_em(case(tmp_path,rh_units="%",rh=.5))
    assert c.snapshot.fields["RH"].max()==.5


@pytest.mark.parametrize("attrs,match",[({"CEN_LAT":39.},"projection attributes"),
                                        ({"DY":24000.},"dx == dy"),
                                        ({"WEST-EAST_GRID_DIMENSION":8},"geometry contradicts")])
def test_geometry_poison_is_refused(tmp_path,attrs,match):
    with pytest.raises(MetgridRefusal,match=match):
        read_met_em(case(tmp_path,global_override=attrs))


def test_map_factor_poison_is_refused(tmp_path):
    def mutate(ds): ds["MAPFAC_U"][:]=1.2
    with pytest.raises(MetgridRefusal,match="MAPFAC_U"):
        read_met_em(case(tmp_path,mutate=mutate))


def test_filename_and_internal_time_must_agree(tmp_path):
    with pytest.raises(MetgridRefusal,match="Times.*disagrees"):
        read_met_em(case(tmp_path,time="2021-06-01_01:00:00"))


def test_nocolons_times_record_is_read_like_real_exe(tmp_path):
    # metgrid with nocolons = .true. in namelist.wps writes the Times
    # record as YYYY-MM-DD_HH_MM_SS; real.exe reads it, so the door does.
    c = read_met_em(case(tmp_path, time="2021-06-01_00_00_00"))
    assert c.valid_time == datetime(2021, 6, 1)
    with pytest.raises(MetgridRefusal, match="Times.*disagrees"):
        read_met_em(write_case(tmp_path/"met_em.d01.2021-06-01_01_00_00.nc",
                               time="2021-06-01_00_00_00"))
    # The real.exe differential oracle reads the same files, both spellings.
    from woof.verify.metem_differential import read_metem
    assert read_metem(case(tmp_path, time="2021-06-01_00_00_00")).valid_time == c.valid_time
    assert read_metem(case(tmp_path)).valid_time == c.valid_time


@pytest.mark.parametrize("record", ["2021-06-01T00:00:00", "2021-06-01_00-00-00",
                                    "2021-06-01 00:00:00", "2021-13-01_00:00:00"])
def test_times_record_outside_both_wrf_spellings_is_refused(tmp_path, record):
    with pytest.raises(MetgridRefusal, match="invalid ASCII Times record"):
        read_met_em(case(tmp_path, time=record))


def test_series_checks_cadence_and_static_identity(tmp_path):
    a=read_met_em(case(tmp_path))
    b=replace(a,path=tmp_path/"later.nc",valid_time=a.valid_time+timedelta(hours=1))
    kwargs=dict(interval_seconds=3600,start_time=a.valid_time,end_time=b.valid_time)
    check_met_em_series((a,b),**kwargs)
    with pytest.raises(MetgridRefusal,match="forcing gap"):
        check_met_em_series((a,replace(b,valid_time=b.valid_time+timedelta(hours=1))),**kwargs)
    statics=dict(b.statics,LANDMASK=np.zeros_like(b.statics["LANDMASK"]))
    with pytest.raises(MetgridRefusal,match="LANDMASK changed"):
        check_met_em_series((a,replace(b,statics=statics)),**kwargs)


def test_model_level_pressure_may_evolve_but_isobaric_identity_may_not(tmp_path):
    a=read_met_em(case(tmp_path,specific=True))
    snap=replace(a.snapshot,levels_hpa=a.snapshot.levels_hpa+1)
    b=replace(a,path=tmp_path/"later.nc",valid_time=a.valid_time+timedelta(hours=1),snapshot=snap)
    kwargs=dict(interval_seconds=3600,start_time=a.valid_time,end_time=b.valid_time)
    check_met_em_series((a,b),**kwargs)
    with pytest.raises(MetgridRefusal,match="isobaric forcing levels"):
        check_met_em_series((replace(a,lane="rh"),replace(b,lane="rh")),**kwargs)


def test_real_retained_metgrid_series_against_independent_reader():
    directory=os.environ.get("WOOF_TEST_METEM_DIR")
    if not directory: pytest.skip("set WOOF_TEST_METEM_DIR for retained WPS fixture")
    paths=met_em_series(directory)
    cases=[read_met_em(path) for path in paths]
    cadence=(cases[1].valid_time-cases[0].valid_time).total_seconds()
    check_met_em_series(cases,interval_seconds=cadence,start_time=cases[0].valid_time,end_time=cases[-1].valid_time)
    for c in cases:
        with nc.Dataset(c.path) as ds:
            np.testing.assert_array_equal(c.snapshot.fields["TT"],ds["TT"][0,1:])
            np.testing.assert_array_equal(c.snapshot.fields["U10"],ds["UU"][0,0])
            if c.lane=="specific_humidity":
                np.testing.assert_array_equal(c.snapshot.fields["SPFH"],ds["SPECHUMD"][0,1:])
                np.testing.assert_array_equal(c.snapshot.fields["PRES"],ds["PRES"][0,1:])


def test_specific_humidity_does_not_need_or_decode_unused_rh(tmp_path):
    def mutate(ds): ds.renameVariable("RH","UNUSED_RH")
    c=read_met_em(case(tmp_path,specific=True,mutate=mutate))
    assert c.lane=="specific_humidity"
    assert "RH" not in c.snapshot.fields and "RH2" not in c.snapshot.fields


def test_model_level_rh_without_spechumd_is_read_like_real_exe(tmp_path):
    """RH, no SPECHUMD, levels that are not isobaric: refused before.

    real.exe forms qv from RH, TT and the 3-D PRES (rh_to_mxrat1,
    module_initialize_real.F:1113-1142).  The reader forms the matching
    specific humidity and hands it to the FLAG_SH lane with PRES.
    """
    from woof.ingest.metem import RH_DERIVED_LANE
    def tilt(ds):
        # Terrain-following levels: each level above the surface spans
        # 20 hPa across the domain, as on a model-level tape.
        pres=np.asarray(ds["PRES"][0],np.float64)
        pres[1:]+=np.linspace(-1000.,1000.,pres.shape[2])[None,None,:]
        ds["PRES"][0]=pres
        rh=np.asarray(ds["RH"][0],np.float64)
        rh[:]=np.linspace(8.,101.,rh.shape[0])[:,None,None]
        ds["RH"][0]=rh
    path=case(tmp_path,mutate=tilt)
    c=read_met_em(path)
    assert c.lane==RH_DERIVED_LANE
    fields=c.snapshot.fields
    assert {"SPFH","Q2","PRES"}<=set(fields)
    assert "RH" not in fields and "RH2" not in fields
    with nc.Dataset(path) as ds:
        tt=np.asarray(ds["TT"][0],np.float64)
        pres=np.asarray(ds["PRES"][0],np.float64)
        rh=np.clip(np.asarray(ds["RH"][0],np.float64),0.,100.)
    # WRF rh_to_mxrat1 (liquid, rh2qv_method 1), transcribed independently.
    es=.01*rh*6.112*np.exp(17.67*(tt-273.15)/(tt-29.65))
    qv=np.maximum(.622*es/(pres/100.-es),1e-6)
    expected=(qv/(1.+qv)).astype(np.float32)
    np.testing.assert_allclose(fields["SPFH"],expected[1:],rtol=2e-7,atol=0)
    np.testing.assert_allclose(fields["Q2"],expected[0],rtol=2e-7,atol=0)
    np.testing.assert_array_equal(fields["PRES"],pres[1:].astype(np.float32))
    assert float(np.ptp(fields["PRES"][0]))>1000.
    assert any("not isobaric" in note and "rh_to_mxrat1" in note for note in c.notes)
    assert c.field_map["rh_to_mxrat1(RH, TT, PRES)[1:]"]=="SPFH"
    # A series of such files keeps evolving model-level pressure, like
    # the SPECHUMD lane, instead of demanding one isobaric ladder.
    snap=replace(c.snapshot,levels_hpa=c.snapshot.levels_hpa+1)
    later=replace(c,path=tmp_path/"later.nc",valid_time=c.valid_time+timedelta(hours=1),snapshot=snap)
    check_met_em_series((c,later),interval_seconds=3600,start_time=c.valid_time,end_time=later.valid_time)


def test_isobaric_rh_stays_on_the_rh_lane(tmp_path):
    c=read_met_em(case(tmp_path))
    assert c.lane=="rh" and "RH" in c.snapshot.fields and "SPFH" not in c.snapshot.fields


def test_analyzed_flags_keep_selective_mass_inventory(tmp_path):
    def mutate(ds):
        dims=("Time","num_metgrid_levels","south_north","west_east")
        for name,value,flag in (("QC",1e-5,1),("QR",2e-5,0)):
            ds.createVariable(name,"f4",dims)[:]=value
            ds.setncattr("FLAG_"+name,flag)
    c=read_met_em(case(tmp_path,specific=True,mutate=mutate))
    assert "QC" in c.snapshot.fields and "QR" not in c.snapshot.fields
    assert "QI" not in c.snapshot.fields


@pytest.mark.parametrize("flag", ["SH","QC"])
def test_analyzed_flag_requires_its_named_field(tmp_path,flag):
    def mutate(ds): ds.setncattr("FLAG_"+flag,1)
    with pytest.raises(MetgridRefusal,match="FLAG_"+flag+"=1.*absent"):
        read_met_em(case(tmp_path,mutate=mutate))


def test_finite_input_that_overflows_float32_is_refused(tmp_path):
    def mutate(ds):
        ds.renameVariable("TT","ORIGINAL_TT")
        variable=ds.createVariable("TT","f8",ds["ORIGINAL_TT"].dimensions)
        variable[:]=1e300
        variable.units="K"
    with pytest.raises(MetgridRefusal,match="float32 representation"):
        read_met_em(case(tmp_path,mutate=mutate))


def test_the_source_top_pressure_is_read_from_the_real_file(tmp_path):
    """The number the vertical contract bounds the model top with.

    Read through the real decoder, not a stub: ``resolve_metem_run``
    calls this for every met_em resolve
    (woof/metem_door.py:116) and hands the answer to
    ``validate_explicit_eta_grid``.
    """
    from woof.ingest.metem import met_em_source_top_pressure_pa
    assert met_em_source_top_pressure_pa(case(tmp_path)) == 20000.0


def test_an_empty_pressure_stack_names_the_breakage_and_the_way_out(tmp_path):
    """A refusal states what breaks AND how to get out of it."""
    from woof.ingest.metem import met_em_source_top_pressure_pa

    def mutate(ds):
        ds["PRES"][:] = np.nan

    with pytest.raises(MetgridRefusal) as refused:
        met_em_source_top_pressure_pa(case(tmp_path, mutate=mutate))
    said = str(refused.value)
    assert "no finite positive pressure" in said
    assert "Re-run metgrid" in said and "point --met-em at" in said
