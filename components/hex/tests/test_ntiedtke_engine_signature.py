"""Bind MPAS timesteps through both supported engine signature spellings."""
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys.ntiedtke_v842 import Constants, _pipeline


@pytest.mark.parametrize('stage', ['ntiedtke_mfub', 'ntiedtke_closure'])
@pytest.mark.parametrize('timestep', ['ztmst', 'dt', None])
def test_args_for_engine_timestep_signature(monkeypatch, stage, timestep):
    from woof.core import ntiedtke as engine

    # Use the installed engine's complete signature and real args_for binder.
    # Only allocation is replaced: this test does no kernel execution.
    signature = tuple(timestep if name in ('dt', 'ztmst') else name
                      for name in engine.NT_STAGE_SIGNATURE[stage]
                      if timestep is not None or name not in ('dt', 'ztmst'))
    signatures = dict(engine.NT_STAGE_SIGNATURE, **{stage: signature})
    monkeypatch.setattr(engine, 'NT_STAGE_SIGNATURE', signatures)

    def allocate(self, **kwargs):
        self.scalars = {name: np.float32(value) for name, value in kwargs.items()}
        self._base = {}
        self.geometry = object()
        self.w = SimpleNamespace(_stage_args={}, scr=object(), scr_i=object(),
                                 bind=lambda name, base: (name, base))

    def allocate_stages(self, geometry):
        self.geometry = geometry
        self._cp = np

    monkeypatch.setattr(engine.NtPipeline, '__init__', allocate)
    monkeypatch.setattr(engine.NtStages, '__init__', allocate_stages)
    pipeline = _pipeline(2, 4, 60., 3, 1, Constants())
    if timestep is None:
        with pytest.raises(ValueError, match=f'{stage}: engine signature lacks timestep argument dt or ztmst'):
            pipeline.args_for(stage)
        return
    args = pipeline.args_for(stage)
    assert args[signature.index(timestep)] == np.float32(180.)
    # The extra MPAS sublimation constant retains its position and value.
    assert args[signature.index('xlv') + 1] == np.float32(Constants().xls)
    pipeline.scalars['ztmst'] = np.float32(240.)
    assert pipeline.args_for(stage)[signature.index(timestep)] == np.float32(240.)
