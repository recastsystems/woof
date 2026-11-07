"""Stock WRF go routing preserves clocks, checkpoints and ensemble admission."""
from argparse import Namespace
from pathlib import Path
import pytest
from woof.go_cli import _go_wrfinput_main, GoRefusal

def args(**kw):
    return Namespace(**(dict(wrfinput=Path('native'),outdir=Path('result'),
        run_seconds=10800,restart=Path('resume.nc'),soil_source=None,
        benchmark=True,allow_shared_gpu=True,render_products='none',
        checkpoint_at='2024-05-25_19:00:00',members=None,
        keep_member_files=None) | kw))

def test_stock_go_preserves_run_arguments(monkeypatch):
    import woof.wrfinput_forecast as worker
    received={}
    def run(source,out,**kw):
        received.update(source=source,out=out,**kw)
        return 0
    monkeypatch.setattr(worker,'run_wrf_forecast',run)
    assert _go_wrfinput_main(args())==0
    assert received==dict(source=Path('native'),out=Path('result'),
        run_seconds=10800,restart=Path('resume.nc'),soil_source=None,
        benchmark=True,allow_shared_gpu=True,render_products='none',
        checkpoint_at='2024-05-25_19:00:00')

def test_native_one_trajectory_cannot_duplicate_ensemble_members():
    with pytest.raises(ValueError,match='members|member'):
        _go_wrfinput_main(args(members=2))


def test_native_restart_is_forwarded(monkeypatch):
    import woof.wrfinput_forecast as worker
    received = {}
    monkeypatch.setattr(worker, 'run_wrf_forecast',
                        lambda source, out, **kw: received.update(kw) or 0)
    _go_wrfinput_main(args(restart_from=Path('wrfrst_d01')))
    assert received['restart_from'] == Path('wrfrst_d01')

def test_native_and_config_domain_owners_cannot_mix():
    with pytest.raises(GoRefusal,match='CONFIG'):
        _go_wrfinput_main(args(config=Path('other.toml')))
