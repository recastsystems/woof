import numpy as np

from woof.hex.wave_prepared import coordinate_difference


def test_native_radians_and_prepared_degrees_accept_only_encoding_roundoff():
    exact = np.array([0., 275.047048, 293.047048, 347.047048, 359.99999, -179.99999])
    mesh = np.deg2rad(exact).astype('f4')
    prepared = exact.astype('f4')
    difference, allowance = coordinate_difference(mesh, prepared, longitude=True, xp=np)
    assert np.all(difference <= allowance)
    assert allowance.max() < 3.e-5
    # A shifted point and reversed cell order must still fail admission.
    shifted = prepared.copy()
    shifted[2] += .001
    difference, allowance = coordinate_difference(mesh, shifted, longitude=True, xp=np)
    assert difference[2] > allowance[2]
    difference, allowance = coordinate_difference(mesh, prepared[::-1].copy(), longitude=True, xp=np)
    assert np.any(difference > allowance)


def test_longitude_wrap_and_latitude_are_distinct():
    mesh = np.deg2rad(np.array([270., 90.])).astype('f4')
    prepared = np.array([-90., 90.], dtype='f4')
    difference, allowance = coordinate_difference(mesh, prepared, longitude=True, xp=np)
    assert np.all(difference <= allowance)
    difference, allowance = coordinate_difference(mesh, prepared, xp=np)
    assert difference[0] > allowance[0]
