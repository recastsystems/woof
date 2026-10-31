"""The two WRF-input doors draw the run they just integrated.

`woof run --wrfinput` and `woof run --met-em` reach the very same
`run_prepared_tree` every other prepared door reaches, and the render
capability is route-agnostic -- but these two launchers dropped the
product flags on the floor: nothing armed the early render, nothing ran
the finalize render, and a finished forecast published no picture, no
first-products receipt and no `latest-run.txt`.  These tests hold the
flags to the worker argv, the arming to the shared decision, and the
finalize render to `woof go`'s own stage.
"""
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from woof.filesystem_paths import canonical_path


class _Lock:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _supervised(monkeypatch, calls):
    """Recorders for everything the supervised parent branch touches."""

    import subprocess

    from woof import go_cli, supervisor, wrfinput_door

    monkeypatch.setattr(go_cli, 'render_extra_missing', lambda: None)
    monkeypatch.setattr(wrfinput_door, 'resolve_wrfinput_run', lambda *args, **kwargs: SimpleNamespace(experiment=None))
    monkeypatch.setattr(supervisor, 'select_gpu',
                        lambda uuid=None: calls.setdefault('gpu', []).append(uuid)
                        or SimpleNamespace(uuid='GPU-0'))
    monkeypatch.setattr(supervisor, 'GPUFileLock', _Lock)
    monkeypatch.setattr(supervisor, 'preflight_exclusive_gpu',
                        lambda *args, **kwargs: calls.setdefault('preflight', []).append(args))

    def run(command, **kwargs):
        calls['argv'] = list(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, 'run', run)


def _pair(argv, flag):
    """The flag and its value, as the argv actually spells them."""

    index = argv.index(flag)
    return argv[index:index + 2]


def test_the_supervised_wrfinput_worker_carries_the_product_flags(tmp_path, monkeypatch):
    from woof.wrfinput_forecast import run_wrf_forecast

    calls = {}
    _supervised(monkeypatch, calls)
    outdir = tmp_path / 'out'
    assert run_wrf_forecast(tmp_path / 'wrf', outdir, render_products='refl',
                            render_dir=outdir / 'pics') == 0
    argv = calls['argv']
    assert _pair(argv, '--products') == ['--products', 'refl']
    assert _pair(argv, '--render-dir') == ['--render-dir', str((outdir / 'pics').resolve())]


def test_the_supervised_metem_worker_carries_the_product_flags(tmp_path, monkeypatch):
    from woof import metem_door, metem_forecast
    from test_metem_forecast import _launcher_stub_run

    calls = {}
    _supervised(monkeypatch, calls)
    monkeypatch.setattr(metem_door, 'resolve_metem_run',
                        lambda directory, **kwargs: _launcher_stub_run())
    outdir = tmp_path / 'out'
    assert metem_forecast.run_metem_forecast(
        tmp_path / 'met', outdir, render_products='all',
        render_dir=outdir / 'pics') == 0
    argv = calls['argv']
    assert _pair(argv, '--products') == ['--products', 'all']
    assert _pair(argv, '--render-dir') == ['--render-dir', str((outdir / 'pics').resolve())]


class _FakePopen:
    """The render subprocess, drawing one picture into its ``--out``."""

    calls: list[list[str]] = []

    def __init__(self, command, **kwargs):
        _FakePopen.calls.append(list(command))
        self.pid = 4242
        self.returncode = 0
        out = Path(command[command.index('--out') + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / 'refl_0000.png').write_bytes(b'\x89PNG\r\n\x1a\n')
        # Empty pipes, read while the render runs.
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def wait(self, timeout=None):
        return self.returncode

    def communicate(self):
        return '', ''


def _worker_door(monkeypatch, tmp_path, *, frames=True):
    """Everything the in-process worker body calls, replaced by stubs."""

    import subprocess

    from woof import go_cli, prepared_domain_tree_forecast, wrfinput_door, wrfinput_forecast

    recorded = {'kwargs': None}
    monkeypatch.setattr(go_cli, 'render_extra_missing', lambda: None)
    monkeypatch.setattr(subprocess, 'Popen', _FakePopen)
    _FakePopen.calls = []
    monkeypatch.setattr(
        wrfinput_door, 'resolve_wrfinput_run',
        lambda directory, **kwargs: SimpleNamespace(
            substitution_report=SimpleNamespace(substitutions=())))
    import datetime

    inputs = SimpleNamespace(experiment=SimpleNamespace(
        start_time=datetime.datetime(2026, 5, 17, 18, tzinfo=datetime.timezone.utc)))
    def prepare(run, directory, **kwargs):
        inputs.prepared_root = Path(directory)
        return inputs

    monkeypatch.setattr(wrfinput_forecast, 'prepare_wrf_run', prepare)
    monkeypatch.setattr(wrfinput_forecast, 'WrfInitialization', lambda inputs: object())

    def run_prepared_tree(inputs, *, output_directory, **kwargs):
        recorded['kwargs'] = dict(kwargs)
        recorded['outdir'] = canonical_path(output_directory)
        if frames:
            wrfout = Path(output_directory) / 'wrfout'
            wrfout.mkdir(parents=True, exist_ok=True)
            (wrfout / 'wrfout_d01_2026-05-17_18_00_00').write_bytes(b'CDF')
        return {'status': 'ok'}

    monkeypatch.setattr(prepared_domain_tree_forecast, 'run_prepared_tree',
                        run_prepared_tree)
    return recorded


def test_the_wrfinput_worker_arms_the_early_render_and_draws_the_run(tmp_path, monkeypatch):
    from woof.wrfinput_forecast import run_wrf_forecast

    recorded = _worker_door(monkeypatch, tmp_path)
    outdir = tmp_path / 'out'
    assert run_wrf_forecast(tmp_path / 'wrf', outdir, exclusive_gpu=False,
                            render_products='refl') == 0
    # (a) the shared trigger, armed by presence of a product spec
    trigger = recorded['kwargs']['first_products']
    assert trigger is not None and trigger.render_products == 'refl'
    # (b) the render argv go_cli composes names the frame and the products,
    #     and draws into the ONE folder the early render was armed with
    argv = _FakePopen.calls[-1]
    assert _pair(argv, '--out') == ['--out', str(trigger.render_dir)]
    assert any(token.endswith('wrfout_d01_2026-05-17_18_00_00') for token in argv)
    assert _pair(argv, '--products') == ['--products', 'refl']
    # (c) the pointer names the folder the pictures are in
    pointer = outdir / 'png' / 'latest-run.txt'
    assert pointer.exists()
    folder = outdir / 'png' / pointer.read_text(encoding='utf-8').strip()
    assert folder.is_dir() and list(folder.glob('*.png'))


def test_no_products_means_no_trigger_and_no_render(tmp_path, monkeypatch):
    from woof.wrfinput_forecast import run_wrf_forecast

    recorded = _worker_door(monkeypatch, tmp_path)
    outdir = tmp_path / 'out'
    assert run_wrf_forecast(tmp_path / 'wrf', outdir, exclusive_gpu=False,
                            render_products='none') == 0
    assert 'first_products' not in recorded['kwargs']
    assert _FakePopen.calls == []
    assert not (outdir / 'png').exists()


def test_absent_products_still_draws_the_default_catalog(tmp_path, monkeypatch):
    """Fixed means default-on: no flag draws the catalog, and only late."""

    from woof.wrfinput_forecast import run_wrf_forecast

    recorded = _worker_door(monkeypatch, tmp_path)
    outdir = tmp_path / 'out'
    assert run_wrf_forecast(tmp_path / 'wrf', outdir, exclusive_gpu=False) == 0
    # Off by ABSENCE early ...
    assert 'first_products' not in recorded['kwargs']
    # ... and drawn at the end, with no --products of its own.
    argv = _FakePopen.calls[-1]
    assert '--products' not in argv
    assert (outdir / 'png' / 'latest-run.txt').exists()


def test_an_install_that_cannot_draw_says_so_before_the_card(tmp_path, monkeypatch, capsys):
    from woof import go_cli, supervisor, wrfinput_door
    from woof.wrfinput_forecast import run_wrf_forecast

    order = []
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(wrfinput_door, 'resolve_wrfinput_run', lambda *args, **kwargs: SimpleNamespace(experiment=None))
    monkeypatch.setattr(go_cli, 'render_extra_missing',
                        lambda: order.append('asked') or 'the rust render engine is not available')
    monkeypatch.setattr(supervisor, 'select_gpu',
                        lambda uuid=None: order.append('gpu') or SimpleNamespace(uuid='GPU-0'))
    monkeypatch.setattr(supervisor, 'GPUFileLock', _Lock)
    monkeypatch.setattr(supervisor, 'preflight_exclusive_gpu', lambda *a, **k: None)
    import subprocess
    monkeypatch.setattr(subprocess, 'run',
                        lambda command, **kwargs: SimpleNamespace(returncode=0))
    assert run_wrf_forecast(tmp_path / 'wrf', tmp_path / 'out') == 0
    assert order == ['asked', 'gpu']
    printed = capsys.readouterr().out
    assert 'this run will draw no pictures' in printed
    assert 'woof setup' in printed


def test_the_supervised_relaunch_does_not_repeat_the_readiness_pair(tmp_path, monkeypatch, capsys):
    """One run that cannot draw says so once, where the card is taken.

    The parent announces at plan review and then re-launches this same
    door as a child, so the pair was printed twice into one terminal.
    """

    from woof import go_cli, metem_forecast, provenance_gate, wrfinput_forecast

    monkeypatch.setattr(go_cli, 'render_extra_missing',
                        lambda: 'the rust render engine is not available')
    assert wrfinput_forecast.announce_render_readiness('woof run --wrfinput')
    assert 'this run will draw no pictures' in capsys.readouterr().out
    assert wrfinput_forecast.announce_render_readiness('woof run --wrfinput',
                                                       announce=False)
    assert capsys.readouterr().out == ''
    # And the re-launched child is the process that keeps quiet.
    monkeypatch.setattr(provenance_gate, 'announce', lambda door, **kwargs: None)
    seen = {}
    monkeypatch.setattr(wrfinput_forecast, 'run_wrf_forecast',
                        lambda *args, **kwargs: seen.update(kwargs) or 0)
    monkeypatch.setattr(metem_forecast, 'run_metem_forecast',
                        lambda *args, **kwargs: seen.update(kwargs) or 0)
    for door, flag in ((wrfinput_forecast, '--wrfinput'),
                       (metem_forecast, '--met-em')):
        argv = [flag, str(tmp_path / 'in'), '--outdir', str(tmp_path / 'out')]
        assert door.main(argv) == 0
        assert seen['relaunched'] is False
        assert door.main(argv + ['--_worker']) == 0
        assert seen['relaunched'] is True


def test_an_install_that_cannot_draw_claims_no_run_folder(tmp_path, monkeypatch):
    """No engine, no pictures, and so no stamped folder to open.

    The finalize stage skips itself by name on this install, so a run
    folder claimed for it would be an empty directory per run that
    nothing ever writes into and `latest-run.txt` never names.
    """

    from woof import go_cli
    from woof.wrfinput_forecast import run_wrf_forecast

    recorded = _worker_door(monkeypatch, tmp_path)
    monkeypatch.setattr(go_cli, 'render_extra_missing',
                        lambda: 'the rust render engine is not available')
    outdir = tmp_path / 'out'
    assert run_wrf_forecast(tmp_path / 'wrf', outdir, exclusive_gpu=False,
                            render_products='refl') == 0
    assert recorded['kwargs']['first_products'] is not None
    assert not (outdir / 'png').exists()
    assert _FakePopen.calls == []


@pytest.mark.parametrize('mask', ['', '-1'])
def test_wrfinput_supervisor_preserves_hidden_cuda_without_a_lease(tmp_path, monkeypatch, mask):
    import subprocess
    from woof.ensemble import supervised_devices
    from woof.wrfinput_forecast import run_wrf_forecast
    calls = {}
    _supervised(monkeypatch, calls)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', mask)
    def refuse_lease(*args, **kwargs):
        pytest.fail('hidden-device process attempted a physical GPU lease')
    monkeypatch.setattr(supervised_devices, 'input_device_lease', refuse_lease)
    def child(command, **kwargs):
        assert '--_worker' in command
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == mask
        return SimpleNamespace(returncode=19)
    monkeypatch.setattr(subprocess, 'run', child)
    assert run_wrf_forecast(tmp_path/'wrf',tmp_path/'out',render_products='none') == 19
    assert 'gpu' not in calls
