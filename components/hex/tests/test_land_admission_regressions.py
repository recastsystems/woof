"""Live T5 gate regressions: native diagnostic q2 and soil-class conversion."""
import numpy as np
import pytest
from woof.hex.cycle.land_state import validate_land, LAND_BY_NAME


def test_negative_native_q2_is_a_diagnostic_not_a_land_water_store():
    # T5 segment 0, 2023-07-08 12:15, cell 1459. Native post-LSM
    # qsfc-qfx/(rho*cqs2) is negative with these physical flux inputs.
    validate_land({'q2': np.array([-.002457842230796814], dtype='f4')}, {})
    assert LAND_BY_NAME['q2'].role == 'diagnostic'
    with pytest.raises(ValueError, match='canwat'):
        validate_land({'canwat': np.array([-.002457842230796814])}, {})
    with pytest.raises(ValueError, match='q2'):
        validate_land({'q2': np.array([np.nan])}, {})


def test_native_effective_surface_humidity_is_not_a_water_reservoir():
    # Noah reports an effective flux-derived humidity, including signed dew.
    validate_land({'qsfc': np.array([-.00031870874227024615], dtype='f4')}, {})
    assert LAND_BY_NAME['qsfc'].role == 'diagnostic'
    with pytest.raises(ValueError, match='qsfc'):
        validate_land({'qsfc': np.array([np.inf])}, {})


def test_ruc_soil_column_is_bounded_by_ruc_porosity_not_noah():
    # hrrrv5 (RUC, 9 levels) histories, V8X6000 2026-10-29: smois 0.3398 on sand (Noah .339, RUC .395),
    # 0.4387 on sandy loam (RUC mixes soil fractions) and 0.451 on urban (Noah .45) were refused by the Noah bound.
    ctx = dict(isltyp=np.array([1, 3, 6, 14, 1], 'f4'), ivgtyp=np.array([10, 10, 13, 10, 17], 'f4'),
               xland=np.array([1, 1, 1, 1, 2], 'f4'), xice=np.zeros(5, 'f4'), isice=np.array(15),
               mminlu=np.array(list('MODIFIED_IGBP_MODIS_NOAH'), 'S1'))
    sm = np.tile(np.array([.3398, .4387, .451, .9, 1.], 'f4')[:, None], (1, 9))
    fields = dict(smois=sm, sh2o=sm * np.float32(.5), tslb=np.full(sm.shape, 290, 'f4'))
    validate_land(fields, ctx)
    with pytest.raises(ValueError, match='soil bounds cell=0'):
        validate_land({n: a[:, :4] for n, a in fields.items()}, ctx)
    with pytest.raises(ValueError, match='soil bounds cell=1'):
        validate_land(dict(fields, smois=np.where(np.arange(5)[:, None] == 1, np.float32(.6), sm)), ctx)
