"""``woof run`` on the config route pins its preparation backend (A157).

The route used to ask for ``auto`` unconditionally, and ``auto`` prepares on
the CPU when the card read 50% busy or could not hold the preparation
(before the load-independent selection fix).  Two arms of one comparison on a shared card
could therefore start from different preparations, and the CPU and card
preparations differ from t = 0.  ``[case_data] preprocess_backend`` and
``woof run --preprocess-backend`` now name it; silence keeps ``auto``.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof import case_data, cli, runtime, supervisor
from woof.case_data import load_experiment_case

from test_case_data import _CASE_DATA_TOML, make_case_toml


def _case(tmp_path, line=None):
    text = _CASE_DATA_TOML if line is None else _CASE_DATA_TOML + line + "\n"
    return make_case_toml(tmp_path, case_data=text)


# --------------------------------------------------------------------
# The TOML key
# --------------------------------------------------------------------

def test_silence_is_auto_and_changes_nothing(tmp_path):
    _exp, data = load_experiment_case(_case(tmp_path))
    assert data.preprocess_backend is None


@pytest.mark.parametrize("backend", ["cuda", "cpu", "auto"])
def test_the_key_names_the_backend(tmp_path, backend):
    _exp, data = load_experiment_case(
        _case(tmp_path, f'preprocess_backend = "{backend}"'))
    assert data.preprocess_backend == backend


@pytest.mark.parametrize("value,match", [
    ('"gpu"', "must be one of"),
    ('"CPU"', "must be one of"),
    ("1", "must be a string"),
])
def test_an_unknown_backend_is_refused_by_name(tmp_path, value, match):
    with pytest.raises(ValueError, match=match):
        load_experiment_case(_case(tmp_path, f"preprocess_backend = {value}"))


def test_the_key_has_a_declared_row():
    from woof.config import declared_key_rows
    row = declared_key_rows()["case_data"]["preprocess_backend"]
    assert row["type"] == "string" and row["default"] is None
    assert "auto" in row["doc"]


# --------------------------------------------------------------------
# The root preparation receives it
# --------------------------------------------------------------------

def _root_preparation(monkeypatch, tmp_path, data, store_request=None):
    """Run prepare_root_experiment_case with its source work substituted."""
    exp, _ = load_experiment_case(_case(tmp_path))
    start = exp.start_time
    captured = {}
    monkeypatch.setattr(runtime, "experiment_grid", lambda *_a: "grid")
    monkeypatch.setattr(runtime, "GeogSelection", SimpleNamespace(
        from_case_data=lambda *_a, **_k: "selection"))
    monkeypatch.setattr(runtime, "forcing_schedule", lambda *_a: (start,))
    monkeypatch.setattr(runtime, "forcing_decode_report", lambda *_a: "")
    from woof.core import cam_ozone
    from woof.ingest import boundary_stream
    monkeypatch.setattr(cam_ozone, "cam_ozone_setup", lambda **_k: None)
    monkeypatch.setattr(boundary_stream, "say_prepared_sealed", lambda *_a: None)

    def prepare(cfg, **kwargs):
        captured.update(kwargs)
        return "prepared"

    monkeypatch.setattr(runtime, "prepare_real_case", prepare)
    result = runtime.prepare_root_experiment_case(
        exp, data, input_catalog=SimpleNamespace(valid_times=(start,)),
        forcing_by_time={start: object()},
        **({"store_request": store_request} if store_request else {}))
    assert result == "prepared"
    return captured


def test_silence_keeps_prepare_real_case_on_its_auto_default(monkeypatch, tmp_path):
    assert inspect.signature(runtime.prepare_real_case).parameters[
        "preprocess_backend"].default == "auto"
    _exp, data = load_experiment_case(_case(tmp_path))
    captured = _root_preparation(monkeypatch, tmp_path, data)
    assert "preprocess_backend" not in captured


@pytest.mark.parametrize("backend", ["cuda", "cpu", "auto"])
def test_a_declared_backend_reaches_the_root_preparation(monkeypatch, tmp_path, capsys, backend):
    _exp, data = load_experiment_case(
        _case(tmp_path, f'preprocess_backend = "{backend}"'))
    captured = _root_preparation(monkeypatch, tmp_path, data)
    assert captured["preprocess_backend"] == backend
    assert f"preparation backend: {backend}, as declared" in capsys.readouterr().out


def test_a_declared_backend_also_pins_the_host_store(monkeypatch, tmp_path):
    from woof.ingest.case_store import CaseStoreRequest
    _exp, data = load_experiment_case(
        _case(tmp_path, 'preprocess_backend = "cpu"'))
    request = CaseStoreRequest(tmp_path / "store")
    assert request.backend == "auto"
    captured = _root_preparation(monkeypatch, tmp_path, data,
                                 store_request=request)
    assert captured["store_request"].backend == "cpu"


def test_prepare_real_case_asks_for_the_named_backend():
    # The one auto request on this route now reads the parameter.
    source = inspect.getsource(runtime.prepare_real_case)
    assert "requested_backend = (preprocess_backend if store_request is None" in source
    assert '"auto" if store_request is None' not in source


# --------------------------------------------------------------------
# woof run --preprocess-backend
# --------------------------------------------------------------------

def test_the_run_flag_parses_and_defaults_to_the_config():
    parser = cli.build_parser()
    assert parser.parse_args(["run", "case.toml"]).preprocess_backend is None
    args = parser.parse_args(["run", "case.toml", "--preprocess-backend", "cpu"])
    assert args.preprocess_backend == "cpu"
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "case.toml", "--preprocess-backend", "gpu"])


@pytest.mark.parametrize("route", ["--met-em", "--wrfinput"])
def test_the_flag_is_refused_where_it_would_be_dropped(tmp_path, route, capsys):
    with pytest.raises(SystemExit) as caught:
        cli._dispatch_argv(["run", route, str(tmp_path),
                            "--preprocess-backend", "cpu"])
    assert caught.value.code == 2
    assert "would be dropped" in capsys.readouterr().err


def test_the_flag_is_refused_on_a_legacy_run_config(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy.toml"
    legacy.write_text('[run]\ncase = "not-a-case"\n', encoding="utf-8")
    monkeypatch.setattr(cli, "load_config",
                        lambda *_a: pytest.fail("the legacy loader ran"))
    args = argparse.Namespace(
        command="run", config=legacy, wrfinput=None, met_em=None,
        preprocess_backend="cpu", no_supervise=True)
    with pytest.raises(ValueError, match="does not read it"):
        cli._dispatch(args)


def test_the_unsupervised_run_hands_the_flag_to_the_runtime(tmp_path, monkeypatch):
    config = _case(tmp_path, 'preprocess_backend = "cuda"')
    seen = []
    from woof import config as config_module
    monkeypatch.setattr(config_module, "validate_experiment_preparation",
                        lambda *_a: None)

    def run_experiment(exp, data, outdir, **_kwargs):
        seen.append(data.preprocess_backend)
        return SimpleNamespace(wrfout_paths=(), completed_seconds=0.0,
                               nan_free=True)

    monkeypatch.setattr(runtime, "run_experiment", run_experiment)
    base = dict(command="run", config=config, wrfinput=None, met_em=None,
                no_supervise=True, outdir=tmp_path / "out", restart=None,
                health_debug=False)
    assert cli._dispatch(argparse.Namespace(**base, preprocess_backend="cpu")) == 0
    assert cli._dispatch(argparse.Namespace(**base, preprocess_backend=None)) == 0
    # The flag overrides the key; silence keeps the key's own.
    assert seen == ["cpu", "cuda"]


def test_the_supervised_run_hands_the_flag_to_every_worker(tmp_path, monkeypatch):
    seen = {}

    def supervised(*_args, **kwargs):
        seen.update(kwargs)
        heartbeat = SimpleNamespace(model_elapsed_seconds=0.0, outer_step=0,
                                    last_checkpoint=None, config_digest="d")
        return SimpleNamespace(run_id="r", attempts=1, heartbeat=heartbeat,
                               stdout_logs=(), stderr_logs=())

    monkeypatch.setattr(supervisor, "supervise_experiment", supervised)
    monkeypatch.setattr(supervisor, "_current_transition_receipt",
                        lambda *_a: (None, None))
    args = argparse.Namespace(
        command="run", config=tmp_path / "experiment.toml",
        outdir=tmp_path / "out", restart=None, gpu_uuid=None,
        supervisor_max_restarts=3, prep_timeout=None, allow_shared_gpu=False,
        health_debug=False, directory_input_hash=None,
        preprocess_backend="cpu")
    assert supervisor.supervise_from_cli(args) == 0
    assert seen["preprocess_backend"] == "cpu"
    # resume and branch carry no such flag, and hand the config's own on.
    del args.preprocess_backend
    assert supervisor.supervise_from_cli(args) == 0
    assert seen["preprocess_backend"] is None


def test_the_worker_command_and_parser_carry_the_flag(tmp_path):
    base = dict(restart=None, health_debug=False)
    plain = supervisor._worker_command(
        tmp_path / "c.toml", tmp_path / "p.toml", tmp_path, **base)
    assert "--preprocess-backend" not in plain
    pinned = supervisor._worker_command(
        tmp_path / "c.toml", tmp_path / "p.toml", tmp_path, **base,
        preprocess_backend="cpu")
    assert pinned[:len(plain)] == plain
    assert pinned[len(plain):] == ["--preprocess-backend", "cpu"]
    parsed = supervisor._parser().parse_args(pinned[3:])
    assert parsed.preprocess_backend == "cpu"
    assert supervisor._parser().parse_args(plain[3:]).preprocess_backend is None


def test_the_worker_overrides_the_config_before_the_runtime(tmp_path, monkeypatch):
    config = _case(tmp_path, 'preprocess_backend = "cuda"')
    payload = config.read_bytes()
    outdir = tmp_path / "out"
    outdir.mkdir()
    captured = tmp_path / "captured.toml"
    captured.write_bytes(payload)
    seen = []

    class _Progress:
        last_phase = "worker-start"
        last_step = 0
        last_wrfout = None
        last_checkpoint = None

        def __init__(self, *_args, **_kwargs):
            pass

        def preparing(self, stage):
            self.last_phase = stage

        def failed(self):
            pass

    class _Stop(Exception):
        pass

    def run_experiment(_exp, data, *_args, **_kwargs):
        seen.append(data.preprocess_backend)
        raise _Stop

    monkeypatch.setattr(supervisor, "RuntimeHeartbeat", _Progress)
    monkeypatch.setattr(supervisor, "_validate_worker_resolved_input_inventory",
                        lambda *_a: None)
    monkeypatch.setattr(supervisor, "write_failure_capsule",
                        lambda path, **_kwargs: Path(path))
    monkeypatch.setattr(runtime, "run_experiment", run_experiment)
    monkeypatch.delenv(supervisor.INPUT_AUTHORITIES_ENV, raising=False)
    for name, value in {
            "WOOF_RUN_ID": "run-id",
            "WOOF_CONFIG_DIGEST": hashlib.sha256(payload).hexdigest(),
            "WOOF_STARTED_AT_UTC": "2026-09-30T00:00:00Z",
            "WOOF_GPU_UUID": "GPU-test", "WOOF_GPU_DRIVER": "610.74",
            "WOOF_GPU_NAME": "RTX 5070 Ti",
            "WOOF_INPUT_HASHES_JSON": "{}"}.items():
        monkeypatch.setenv(name, value)
    for flag, expected in (("cpu", "cpu"), (None, "cuda")):
        with pytest.raises(_Stop):
            supervisor._worker_main(argparse.Namespace(
                config=config, config_payload=captured, outdir=outdir,
                restart=None, health_debug=False, preprocess_backend=flag))
        assert seen[-1] == expected
