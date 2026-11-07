from pathlib import Path
from datetime import datetime
import hashlib
import json
import subprocess
import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = requires_gpu


def irrigation_state(cp, monkeypatch, option):
    import woof.core.physics as physics
    from test_restart import _physics_state
    original = physics.initialize_physics
    def initialize(state,cfg,**kw):
        if option:
            kw.update(irrigation=np.ones((cfg.ny,cfg.nx),np.float32),
                      irr_ph_field=np.zeros((cfg.ny,cfg.nx),np.int32))
        return original(state,cfg,**kw)
    with monkeypatch.context() as patch:
        patch.setattr(physics,'initialize_physics',initialize)
        state,cfg,driver=_physics_state(cp,mp_physics=6,cu_physics=0,
            sf_surf_irr_scheme=option,irr_daily_amount=25.,irr_num_hours=24,
            irr_start_julianday=0,irr_end_julianday=366)
    state.qc.fill(0)
    return state,cfg,driver


@pytest.mark.parametrize('option',[1,2,3])
def test_real_irrigation_restart_roundtrip(option,monkeypatch,tmp_path):
    import cupy as cp
    from woof.core.dycore import run_steps
    from woof.io import restart
    from test_restart import _assert_restart_equal
    a,config,_=irrigation_state(cp,monkeypatch,option)
    run_steps(a,config,12)
    reference=restart.write_restart(tmp_path/'reference.npz',a,config)
    b,_,_=irrigation_state(cp,monkeypatch,option)
    run_steps(b,config,6)
    mid=restart.write_restart(tmp_path/'mid.npz',b,config)
    resumed,_,_=irrigation_state(cp,monkeypatch,option)
    restart.restore_restart(mid,resumed,config)
    run_steps(resumed,config,6)
    final=restart.write_restart(tmp_path/'final.npz',resumed,config)
    _assert_restart_equal(final,reference)
    for key in ('irrigation','irr_ph_field','irrigation_channel'):
        assert cp.array_equal(resumed.physics.fields[key],a.physics.fields[key])


def test_default_full_fields_equal_original_noah(monkeypatch,tmp_path):
    import cupy as cp
    import woof.core.kernels as kernels
    from woof.core.dycore import run_steps
    from woof.io import restart
    original=subprocess.check_output(['git','show','839830f1ccfe0ac7d211fe4a1ad66c5a7df5ecd9:woof/core/kernels/noah.cu'],text=True)
    unit=kernels._unit_text
    def base(name,kernel_dir=kernels._KDIR):
        return original if name=='noah' else unit(name,kernel_dir)
    kernels.get_kernel.cache_clear(); kernels.load_module.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(kernels,'_unit_text',base)
        a,cfg,_=irrigation_state(cp,monkeypatch,0)
        run_steps(a,cfg,8)
        reference={k:restart._host(v).tobytes() for k,v in {**restart.state_manifest(a),**restart._scratch_manifest(a),**restart._driver_manifest(a.physics)}.items()}
    kernels.get_kernel.cache_clear(); kernels.load_module.cache_clear()
    b,_,_=irrigation_state(cp,monkeypatch,0)
    run_steps(b,cfg,8)
    actual={k:restart._host(v).tobytes() for k,v in {**restart.state_manifest(b),**restart._scratch_manifest(b),**restart._driver_manifest(b.physics)}.items()}
    assert actual==reference
    receipt={k:hashlib.sha256(v).hexdigest() for k,v in sorted(actual.items())}
    (tmp_path/'default-full-field-hashes.json').write_text(json.dumps(receipt,indent=2))
