"""Bounded kernel contract checks, not forecast or observational evidence."""
from types import SimpleNamespace
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("separate_lake_mask", [False, True])
def test_mynn_humidity_conversion_and_surface_masks(separate_lake_mask):
    import cupy as cp
    from woof.tc.config import DEFAULTS
    from woof.tc.surface import SurfaceBundle
    shape = (1, 4)
    surface_names = ('znt ust ustm mol hfx qfx qsfc zol psim psih fm fh lh '
                     'u10 v10 th2 t2 q2 chs chs2 cqs2 flhc flqc rmol cd cda '
                     'ck cka ch qstar cpm gz1oz0').split()
    outputs = []
    for scheme in (91, 5):
        fields = {k: cp.zeros(shape, cp.float32) for k in surface_names}
        fields.update(psfc=cp.full(shape, 95000., cp.float32),
                      tsk=cp.full(shape, 302., cp.float32),
                      xland=cp.asarray([[2, 1, 2, 2]], cp.float32),
                      lakemask=cp.asarray([[0, 0, 1, 0]], cp.float32),
                      xice=cp.asarray([[0, 0, 0, .5]], cp.float32))
        fields['u10'].fill(60.)
        fields['qsfc'].fill(.024 if scheme == 91 else .024/1.024)
        atmosphere = {k: cp.full((1, *shape), v, cp.float32) for k, v in
                      dict(u=60., v=0., temperature=300., qv=.02,
                           pressure=95000., dz=20.).items()}
        cfg = SimpleNamespace(**(DEFAULTS | dict(
            sf_sfclay_physics=scheme, tc_flux_curve='bell2012')))
        bundle = SurfaceBundle(cfg, fields)
        if separate_lake_mask:
            bundle.lake_mask_override = fields["lakemask"].copy()
            fields["lakemask"].fill(0)
        bundle.exchange(atmosphere, fields, SimpleNamespace(**fields))
        actual = {k: cp.asnumpy(fields[k]) for k in
                  ('qfx','q2','ust','hfx','mol','qstar','chs','flhc','cpm')}
        outputs.append(actual)
        np.testing.assert_allclose(actual['ust'][0, 0], 60*np.sqrt(.0024), rtol=2e-6)
        for name in ('qfx','q2','ust','hfx'):
            assert np.all(actual[name][0, 1:] == 0), name
        expected = 95000/(287*300*(1+.608*.02)) * .001 * 60 * (.024-.02)
        np.testing.assert_allclose(actual['qfx'][0, 0], expected, rtol=3e-6)
        np.testing.assert_allclose(bundle.diag['cdn'].get()[0, 0], .0024, rtol=2e-6)
        np.testing.assert_allclose(bundle.diag['chn'].get()[0, 0], .001, rtol=2e-6)
    np.testing.assert_allclose(outputs[0]['qfx'], outputs[1]['qfx'], rtol=3e-6)
    np.testing.assert_allclose(outputs[0]['q2'], outputs[1]['q2'], rtol=3e-6)
    # MYNN's virtual temperature and specific-humidity scales are consumed
    # by the next surface call, so they cannot be left at MM5's values.
    assert outputs[1]['qstar'][0, 0] < 0
    assert outputs[1]['mol'][0, 0] != outputs[0]['mol'][0, 0]
    np.testing.assert_allclose(outputs[1]['cpm'][0, 0], 1004.5*(1+.84*.02), rtol=2e-6)
