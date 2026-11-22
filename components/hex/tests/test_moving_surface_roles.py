"""A moved member's sea points take the role their carried ice implies.

Breakage it guards: Milton hex storm-following (2026-10-27). The moved
member's xland/vegetation/soil words came from its cold geography (the initial
ice extent on the new mesh) while its ice concentration was carried, so polar
sea points were left as glacier or land; the history census refused, Noah-MP's
glacier water budget failed and the next move's sea-ice remap refused.
"""
import numpy as np

from woof.hex.moving_physics import settle_ocean_roles

ISICE, ISWATER, THRESHOLD = 15, 17, .02


def classify(f):
    from woof.core.noahmp_runtime import classify_noahmp_surface
    masks = classify_noahmp_surface(f['xland'], f['xice'], f['ivgtyp'],
                                    xice_threshold=THRESHOLD, isice=ISICE)
    return {k: int(np.count_nonzero(v)) for k, v in masks.items()}


def columns():
    # 0: ice-covered at init, ice gone after the move (the glacier flip)
    # 1: open water at init, ice arrived after the move
    # 2: ice at init, still ice      3: open water, still water
    # 4: a real land glacier         5: ordinary land       6: lake
    return dict(
        xice=np.array([0.0, .6, .9, 0.0, 0.0, 0.0, 0.0], np.float32),
        xland=np.array([1, 2, 1, 2, 1, 1, 2], np.float32),
        ivgtyp=np.array([15, 17, 15, 17, 15, 7, 17], np.int32),
        isltyp=np.array([16, 14, 16, 14, 16, 3, 14], np.int32),
        vegfra=np.array([0, 0, 0, 0, 0, 40, 0], np.float32),
        albbck=np.array([.8, .08, .8, .08, .8, .2, .08], np.float32),
        embck=np.full(7, .98, np.float32))


def test_unsettled_geography_classifies_a_sea_point_as_glacier():
    assert classify(columns())['glacier'] == 2      # the real one plus the defect


def test_wet_columns_follow_their_ice_and_land_is_untouched():
    f = columns()
    land = {k: v[4:6].copy() for k, v in f.items()}
    wet = np.array([1, 1, 1, 1, 0, 0, 0], bool)    # the lake is not ocean
    arrays = {k: v for k, v in f.items()}
    changed = settle_ocean_roles(arrays, wet, THRESHOLD, ISICE, ISWATER)
    assert changed == 2
    census = classify(f)
    assert census['glacier'] == 1 and census['sea_ice'] == 2 and census['open_water'] == 3
    np.testing.assert_array_equal(f['xland'][:4], [2, 1, 1, 2])
    np.testing.assert_array_equal(f['ivgtyp'][:4], [ISWATER, ISICE, ISICE, ISWATER])
    np.testing.assert_array_equal(f['isltyp'][:4], [14, 16, 16, 14])
    for k, v in land.items():
        np.testing.assert_array_equal(f[k][4:6], v)
    assert f['ivgtyp'].dtype == np.int32 and f['xland'].dtype == np.float32


def test_settled_roles_survive_mpas_conversion_without_a_false_crossing():
    # With xicem equal to the settled xice, MPAS's threshold-crossing rule
    # (sst_update) changes nothing: no sea point is reconverted or stranded.
    f = columns()
    wet = np.array([1, 1, 1, 1, 0, 0, 0], bool)
    settle_ocean_roles(f, wet, THRESHOLD, ISICE, ISWATER)
    xicem = np.where(f['xice'] >= THRESHOLD, f['xice'], 0)
    x, o = f['xice'], xicem
    to_ice = (f['xland'] >= 1.5) & (x >= THRESHOLD) & (o < THRESHOLD)
    to_water = (f['xland'] < 1.5) & (x < THRESHOLD) & (o >= THRESHOLD)
    assert not to_ice.any() and not to_water.any()
