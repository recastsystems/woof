"""Legacy sigma input admission must prove its representation, not guess it."""
import numpy as np
import pytest
from woof.ingest.wrfinput_identity import effective_hybrid_opt

@pytest.mark.parametrize('title,etac,expected', [
    ('OUTPUT FROM REAL_EM V3.9 PREPROCESSOR', 0., 0),
    ('OUTPUT FROM REAL_EM V4.6.1 PREPROCESSOR', 0., -1),
    ('unknown', 0., -1),
    ('OUTPUT FROM REAL_EM V3.9 PREPROCESSOR', .2, -1),
])
def test_sigma_sentinel_requires_the_native_writer_and_zero_etac(title, etac, expected):
    assert effective_hybrid_opt(dict(HYBRID_OPT=-1, TITLE=title, ETAC=etac)) == expected


def test_sigma_input_coefficients_are_checked_before_admission(tmp_path, monkeypatch):
    from woof import netcdf_bridge
    from woof.ingest.wrfinput_identity import read_wrfinput_identity
    from test_wrf_warm_start import _Variable, _restart_dataset
    ds = _restart_dataset()
    ds.attributes.update(HYBRID_OPT=-1, ETAC=0., DX=12000., DY=12000., TITLE='OUTPUT FROM REAL_EM V3.9 PREPROCESSOR')
    # Identity reads vertical coefficients before the remaining identity fields.
    eta = np.array([1., .5, 0.], np.float32)
    ds.variables['ZNW'] = _Variable('ZNW', ('bottom_top_stag',), eta)
    ds.variables['ZNU'] = _Variable('ZNU', ('bottom_top',), (eta[:-1]+eta[1:])*np.float32(.5))
    for suffix, grid in [('H', ds.variables['ZNU'][0]), ('F', eta)]:
        for prefix, value in [('C1', np.ones_like(grid)), ('C2', np.zeros_like(grid)),
                              ('C3', grid), ('C4', np.zeros_like(grid))]:
            ds.variables[prefix+suffix] = _Variable(prefix+suffix, ('bottom_top' if suffix == 'H' else 'bottom_top_stag',), value)
    ds.variables['C3H'] = _Variable('C3H', ('bottom_top',), np.zeros(2, np.float32))
    ds.variables['P_TOP'] = _Variable('P_TOP', (), np.array(5000., np.float32))
    monkeypatch.setattr(netcdf_bridge, 'open_dataset', lambda path: ds)
    with pytest.raises(ValueError, match='inconsistent C3H'):
        read_wrfinput_identity(tmp_path/'wrfinput_d01')
