"""Source-to-LSM/CLM defects: units, missing coasts, age, ice volume and restart."""
from datetime import datetime, timezone
from types import SimpleNamespace
import numpy as np
import pytest
from woof.ingest.lake_state import source_rows, load_state, sample_field
from conftest import requires_gpu


def test_source_licence_and_missing_file_are_explicit(tmp_path):
    row = source_rows()['era5-land']
    # CC-BY-4.0 allows commercial use with attribution, so no named acceptance.
    assert row['licence'] == 'CC-BY-4.0' and not row['research_only'] and row['citation']
    with pytest.raises(ValueError, match='lake_state_path'):
        load_state('era5-land', datetime(2022,11,17), np.array([42.]), np.array([-80.]), cache=tmp_path)


def test_native_coastal_sampling_units_and_no_extrapolation():
    result=sample_field([43.,42.],[-81.,-80.],[[10.,np.nan],[12.,14.]],
                        [42.5,44.,43.],[-80.5,-80.5,-80.],offset=273.15)
    np.testing.assert_allclose(result[0],285.15)
    assert np.isnan(result[1:]).all()
    # CF permits descending longitude axes crossing zero. Equivalent longitudes
    # must select the same donor cell, while out-of-extent points stay missing.
    descending = sample_field([43.,42.],[1.,0.,-1.],[[10.,20.,30.],[10.,20.,30.]],
                              [42.5]*4,[-.5,359.5,2.,362.])
    np.testing.assert_allclose(descending[:2],[25.,25.])
    assert np.isnan(descending[2:]).all()
    with pytest.raises(ValueError, match='incompatible dimensions'):
        sample_field([42.,43.],[-81.,-80.],np.ones((2,3)),[42.5],[-80.5])
    with pytest.raises(ValueError, match='malformed coordinates'):
        sample_field([42.,43.,45.],[-81.,-80.],np.ones((3,2)),[42.5],[-80.5])


def test_native_source_zero_roundoff_preserves_material_depths_and_missing():
    values=sample_field([43.,42.],[-81.,-80.],
        [[-2.16840486e-19,2.16840486e-19],[-.001,.001]],
        [43.,43.,42.,42.,44.],[-81.,-80.,-81.,-80.,-80.],zero_roundoff_tolerance=1e-12)
    np.testing.assert_array_equal(values[:4],[0.,0.,-.001,.001])
    assert np.isnan(values[4])


@pytest.mark.gpu
@requires_gpu
def test_observed_surface_and_era_profile_ice_volume_and_restart(monkeypatch,tmp_path):
    import cupy as cp
    import woof.ingest.lake_state as obs
    from woof.core.lake import initialize_lake
    monkeypatch.setenv('WOOF_LAKE_RECEIPT_DIR',str(tmp_path))
    shape=(2,3)
    def plane(v): return cp.full(shape,v,cp.float32)
    fields=dict(tsk=plane(290),sst=plane(290),xice=plane(0),xicem=plane(0),
        lakemask=cp.asarray([[1,1,0],[0,0,0]],cp.float32),isltyp=plane(3),snow=plane(0),
        ivgtyp=plane(17),xland=plane(2))
    supplied=dict(water_temperature=np.array([280.,273.15]),bottom_temperature=np.array([277.,277.]),
        mixed_layer_depth=np.array([2.,1.]),temperature_shape_factor=np.array([.5,.5]),
        ice_thickness=np.array([0.,.6]),ice_fraction=np.array([0.,.5]),ice_temperature=np.array([273.15,269.]))
    monkeypatch.setattr(obs,'load_state',lambda *a,**kw:(supplied,dict(schema='gpuwm.lake-init.v1',status='bound')))
    initial,receipt=obs.initialize_surface(fields,SimpleNamespace(lake_state_source='auto'),
        valid_time=datetime(2022,11,17,tzinfo=timezone.utc),latitude=plane(42),longitude=plane(-80))
    assert float(fields['tsk'][0,0]) == 280.
    assert float(fields['tsk'][0,1]) == 269.
    np.testing.assert_equal(cp.asnumpy(fields['tsk'])[:,2],290.)
    owner=initialize_lake(fields,latitude=plane(42),lake_depth=plane(20),initial_state=initial)
    static=cp.asnumpy(owner.static);columns=cp.asnumpy(owner.columns)
    np.testing.assert_allclose((columns[15:25]*static[11:21]).sum(axis=0),[0.,.3],atol=1e-6)
    expected=280.+(277.-280.)*np.clip((static[1:11,0]-2.)/18.,0.,1.)
    np.testing.assert_allclose(columns[5:15,0],expected,atol=3e-5)
    before=fields['lake_columns'].copy()
    initialize_lake(fields,latitude=plane(42),lake_depth=plane(20),initial_state=initial)
    cp.testing.assert_array_equal(fields['lake_columns'],before)
    fresh = {k: v for k, v in fields.items() if not k.startswith('lake_')}
    with pytest.raises(ValueError, match='exceeds model bathymetry'):
        initialize_lake(fresh, latitude=plane(42), lake_depth=plane(.1), initial_state=initial)


def test_buoy_water_temperature_and_nearby_water_only():
    from woof.ingest.lake_state import decode_buoy_temperature, nearest_lake_value
    text=b'#YY MM DD hh mm ATMP WTMP\n2022 11 17 12 00 -5.0 99.0\n2022 11 17 12 10 -6.0 10.5\n'
    record=decode_buoy_temperature(text,datetime(2022,11,17,12),tolerance_seconds=900)
    assert record['water_temperature_k']==283.65
    with pytest.raises(ValueError,match='unavailable'):
        decode_buoy_temperature(text,datetime(2022,11,17,12),tolerance_seconds=300)
    match=nearest_lake_value([42.,42.01],[280.,280.01],[270.,280.],[0.,1.],42.,-80.,max_distance_km=3.)
    assert match['flat_index']==1 and match['value']==280.
    with pytest.raises(ValueError,match='no finite lake value'):
        nearest_lake_value([42.],[280.],[280.],[1.],42.,-79.,max_distance_km=3.)


def test_diagnostic_coasts_and_missing_verification_pairs():
    from woof.ingest.lake_state import diagnostic_values, verification_scores
    water = diagnostic_values([273.15, 280., np.nan], [1., 0., 1.], scale=1.8, offset=-459.67)
    np.testing.assert_allclose(water[0], 32.)
    assert np.isnan(water[1:]).all()
    score = verification_scores([280., 282., np.nan, 290.], [281., 280., 285., np.nan])
    assert score['count'] == 2 and score['bias'] == .5 and score['mae'] == 1.5
    assert score['rmse'] == pytest.approx(np.sqrt(2.5))
    empty = verification_scores([np.nan], [280.])
    assert empty == dict(count=0., bias=None, mae=None, rmse=None, max_absolute_error=None)
    __import__('json').dumps(empty, allow_nan=False)


def test_licensed_cf_file_fields_time_and_age(tmp_path):
    from woof.io.classic_product import ClassicProduct
    from datetime import timedelta
    valid = datetime(2022, 11, 17, 12, tzinfo=timezone.utc)
    path = tmp_path/'lake-fields.nc'
    row = source_rows()['era5-land']
    samples = {'lmlt': 280., 'lblt': 277., 'lmld': 2., 'lshf': .5, 'licd': -2.16840486e-19, 'lict': 273.15}
    with ClassicProduct(path) as ds:
        for name, size in (('valid_time', 1), ('latitude', 2), ('longitude', 2)):
            ds.createDimension(name, size)
        for name, values, units in (('valid_time', [valid.timestamp()], 'seconds since 1970-01-01 00:00:00'),
                                     ('latitude', [43., 42.], 'degrees_north'),
                                     ('longitude', [-81., -80.], 'degrees_east')):
            v = ds.createVariable(name, 'f8', (name,)); v.units = units
            v[:] = np.asarray(values, np.float64)
        for f in row['field']:
            v = ds.createVariable(f['variable'], 'f8', ('valid_time', 'latitude', 'longitude'))
            v.units = 'dimensionless' if f['variable'] == 'lshf' else f['units']; v[:] = np.full((1, 2, 2), samples[f['variable']], np.float64)
    kwargs = dict(path=path, accept_licence='CC-BY-4.0', cache=tmp_path)
    values, receipt = load_state('era5-land', valid, np.array([42.2]), np.array([-80.6]), **kwargs)
    assert receipt['status'] == 'bound' and len(receipt['records']) == 6
    _, partial = load_state('era5-land', valid, np.array([42.2, 45.]), np.array([-80.6, -80.6]), **kwargs)
    assert partial['status'] == 'partial_forcing_fallback'
    assert partial['uncovered_targets'] == {f['name']: 1 for f in row['field']}
    for f in row['field']:
        assert values[f['name']][0] == pytest.approx(0. if f['variable']=='licd' else samples[f['variable']])
    assert values['ice_thickness'][0] == 0.
    assert next(v for v in receipt['records'] if v['field']=='ice_thickness')['zero_roundoff_tolerance'] == 1e-12
    with pytest.raises(ValueError, match='old, row limit'):
        load_state('era5-land', valid+timedelta(hours=4), np.array([42.2]), np.array([-80.6]), **kwargs)
    with pytest.raises(ValueError, match='no lake analysis at or before'):
        load_state('era5-land', valid-timedelta(hours=1), np.array([42.2]), np.array([-80.6]), **kwargs)


def test_native_request_coverage_wraps_longitude_and_ignores_missing():
    from woof.ingest.lake_state import coordinate_coverage
    row = source_rows()['glsea']
    result = coordinate_coverage([42., 44., np.nan], [280., -83., -179.], row)
    assert result == dict(south=42., north=44., west=-83., east=-80., covered=2.)
    assert coordinate_coverage([0.], [0.], row)['covered'] == 0
    with pytest.raises(ValueError, match='no finite geographic coordinates'):
        coordinate_coverage([np.nan], [-80.], row)


def test_a_source_with_an_extent_is_requested_whole_whatever_the_targets(monkeypatch, tmp_path):
    """X12: per-slab crops sampled the same lake cell off different windows."""
    import woof.ingest.lake_state as obs
    asked = []
    monkeypatch.setattr(obs, 'acquire_field', lambda row, field, valid_time, bounds, cache, attempts=None:
                        (asked.append(dict(bounds)) or (tmp_path / 'x.nc', 'url')))
    monkeypatch.setattr(obs, 'decode_field', lambda path, row, field, valid_time, lat, lon:
                        (np.full(lat.shape, 1.), dict(covered_targets=int(lat.size))))
    row = source_rows()['glsea']
    valid = datetime(2025, 3, 14, 12, tzinfo=timezone.utc)
    load_state('glsea', valid, np.array([42.2]), np.array([-80.6]), cache=tmp_path)
    load_state('glsea', valid, np.array([42.2, 46.5]), np.array([-80.6, -86.]), cache=tmp_path)
    whole = {key: row[key] for key in ('south', 'north', 'west', 'east')}
    assert asked and all(bounds == whole for bounds in asked)


@pytest.mark.gpu
@requires_gpu
def test_an_uncovered_column_keeps_its_lakeini_state(monkeypatch, tmp_path):
    """X12: binding an all-missing column rewrote its T_GRND from the top layer."""
    import cupy as cp
    from woof.core.lake import initialize_lake
    from woof.ingest.lake_state import FIELDS
    shape = (1, 2)
    def plane(v): return cp.full(shape, v, cp.float32)
    def surface():
        return dict(tsk=cp.asarray([[290., 276.]], cp.float32), sst=plane(285), xice=plane(0),
                    xicem=plane(0), lakemask=plane(1), isltyp=plane(3), snow=plane(0),
                    ivgtyp=plane(17), xland=plane(2))
    plain = surface()
    initialize_lake(plain, latitude=plane(42), lake_depth=plane(20))
    bound = surface()
    supplied = {name: cp.full(shape, cp.nan, cp.float32) for name in FIELDS}
    supplied['water_temperature'][0, 0] = 280.
    initialize_lake(bound, latitude=plane(42), lake_depth=plane(20), initial_state=supplied)
    cp.testing.assert_array_equal(bound['lake_columns'][:, 0, 1], plain['lake_columns'][:, 0, 1])
    assert float(bound['lake_columns'][4, 0, 0]) == 280.
