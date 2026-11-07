"""One admission question, priced the same way at every run door.

Sharing a GPU is a VRAM question, and the answer is the run's priced
reservation measured against the memory the device reports free
(:func:`woof.supervisor.shared_gpu_admission`).  A door that asks that
function WITHOUT a reservation can no longer refuse anything: it returns
``admitted-unpriced-run``.  So for one card, one co-tenant and one
configuration, an unpriced door says yes exactly where the supervisor says
no, which is the same configuration answered two ways.

These are the doors that must agree, checked one by one and then as a
class, because six of them were left unpriced when the first two were
fixed:

* the multi-run worker and its check worker (plan review itself),
* the ``--met-em`` and ``--wrfinput`` run doors,
* the dedicated 500 m runner's allocation gate,
* the dual-run queue's politeness gate, whose exit-1 branch this made
  unreachable so that it always printed CLEAR and started an arm onto a
  card it should have waited out.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = (REPOSITORY_ROOT / "tools" / "gf_real74_dual" / "gpu_free_gate.py")


class _NullLock:
    """Stand in for the UUID file lock without touching the filesystem."""

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _price_recorder(seen: dict, value):
    """Record what a door priced, and hand it back one fixed number.

    ``source`` is the forcing source a door that knows it prices with
    (:func:`woof.supervisor.priced_reservation_bytes`).
    """

    def price(configuration, *, source=None):
        seen["priced"] = configuration
        seen["source"] = source
        return value

    return price


def _stub_multi_run(monkeypatch, multi_run, seen, uuid, price):
    # Worker admission scenarios stay independent of the local card.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid)
    monkeypatch.setattr(multi_run, "GPUFileLock", _NullLock)
    monkeypatch.setattr(multi_run, "priced_reservation_bytes",
                        _price_recorder(seen, price))
    monkeypatch.setattr(multi_run, "preflight_exclusive_gpu",
                        lambda uuid, **kwargs: seen.update(preflight=kwargs))
    monkeypatch.setattr(multi_run.importlib, "import_module",
                        lambda name: SimpleNamespace(main=lambda arguments: 0))


def _stub_supervisor(monkeypatch, supervisor, seen, price):
    # Both resolved-input doors use the same card and subprocess stand-ins.
    # A CPU-only suite hides real cards. This scenario supplies a fake
    # visible card so it exercises pricing rather than hidden-device relaunch.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-alpha")
    monkeypatch.setattr(supervisor, "select_gpu",
                        lambda uuid: SimpleNamespace(uuid="GPU-alpha"))
    monkeypatch.setattr(supervisor, "GPUFileLock", _NullLock)
    monkeypatch.setattr(supervisor, "priced_reservation_bytes",
                        _price_recorder(seen, price))
    monkeypatch.setattr(supervisor, "preflight_exclusive_gpu",
                        lambda uuid, **kwargs: seen.update(preflight=kwargs))
    monkeypatch.setattr(subprocess, "run",
                        lambda *_args, **_kwargs: SimpleNamespace(returncode=0))


def _gate_module():
    spec = importlib.util.spec_from_file_location(
        "gf_real74_dual_gpu_free_gate", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_run_door_prices_the_card_it_asks_about():
    """No call site may ask the admission question without a reservation.

    An unpriced call cannot raise, so it is not a door at all; it is a
    door that was removed while still looking like one.  This is the
    guard for the class rather than for the six sites that had it.
    """

    unpriced = []
    for area in ("woof", "tools"):
        for path in sorted((REPOSITORY_ROOT / area).rglob("*.py")):
            # Vendored crate sources carry their own helper scripts, which
            # are not this tree's doors and need not parse under this
            # interpreter.
            if any("vendor" in part
                   for part in path.relative_to(REPOSITORY_ROOT).parts):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = (getattr(node.func, "id", None)
                        or getattr(node.func, "attr", None))
                if name != "preflight_exclusive_gpu":
                    continue
                if not any(keyword.arg == "reservation_bytes"
                           for keyword in node.keywords):
                    unpriced.append(
                        f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}")
    assert unpriced == [], (
        "these doors ask the shared-card question with no reservation, so "
        "they can never refuse what woof run refuses: "
        + ", ".join(unpriced))


def test_multi_run_worker_prices_its_admission_from_its_experiment_config(
        tmp_path, monkeypatch):
    from woof import multi_run

    outdir = tmp_path / "out"
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    config = tmp_path / "experiment.toml"
    config.write_text("[experiment]\nname='fixture'\n", encoding="utf-8")

    seen: dict[str, object] = {}
    _stub_multi_run(monkeypatch, multi_run, seen, 'GPU-alpha', 9 * 2 ** 30)

    assert multi_run._locked_module_main(
        gpu_uuid="GPU-alpha",
        module_name="woof.prepared_domain_tree_forecast",
        outdir=outdir,
        inputs=(prepared, config),
        arguments=(
            "--prepared-root", str(prepared),
            "--experiment-config", str(config),
            "--outdir", str(outdir))) == 0
    assert Path(str(seen["priced"])) == config
    assert seen["preflight"]["reservation_bytes"] == 9 * 2 ** 30


def test_multi_run_worker_prices_the_boundary_its_forecast_carries(
        tmp_path, monkeypatch):
    """The reservation carries the source the runner forecasts from.

    The source's analysed hydrometeors ride the root's boundary tables
    (woof.boundary_fields), so a multi-run of HRRR-forced forecasts
    priced with no source reserved a shared card short by those tables.
    The single-domain runner names its source (``--source``); a tree
    runner's prepared root names it in its own document; a root prepared
    from a user's own mapping answers with the mapping document it
    copied into its evidence, since the name ``mapped`` publishes none.
    Red with the source left out of the reservation: every case below
    priced with ``None``.
    """
    import json

    from woof import multi_run
    from woof.boundary_fields import source_boundary_species
    from woof.prepared_domain_tree_forecast import HIERARCHY_SCHEMA

    seen: dict[str, object] = {}
    monkeypatch.setattr(multi_run, "priced_reservation_bytes",
                        _price_recorder(seen, 5 * 2 ** 30))
    config = tmp_path / "experiment.toml"
    single = "woof.prepared_single_domain_forecast"

    def priced(module, root, *extra):
        seen.clear()
        arguments = ["--prepared-root", str(root),
                     "--experiment-config", str(config), *extra]
        if module == single:
            arguments += ["--wps-namelist", str(tmp_path / "namelist.wps")]
        assert multi_run._worker_reservation_bytes(
            module, arguments) == 5 * 2 ** 30
        assert Path(str(seen["priced"])) == config
        return seen["source"]

    named = tmp_path / "named"
    named.mkdir()
    assert priced(single, named, "--source", "hrrr") == "hrrr"

    tree_root = tmp_path / "tree"
    tree_root.mkdir()
    (tree_root / "receipt.json").write_text(
        json.dumps({"schema": HIERARCHY_SCHEMA}), encoding="utf-8")
    source = priced("woof.prepared_domain_tree_forecast", tree_root)
    assert source == "hrrr" and source_boundary_species(source)

    mapping = {"format": "grib2", "fields": {
        "cloud_water_mixing_ratio": {}, "snow_mixing_ratio": {}}}
    mapped = tmp_path / "mapped"
    (mapped / "source-evidence").mkdir(parents=True)
    (mapped / "source-evidence" / "mapping.json").write_text(
        json.dumps(mapping), encoding="utf-8")
    source = priced(single, mapped, "--source", "mapped")
    assert source == mapping
    assert source_boundary_species(source) == ("qc", "qs")

    # A root that names nothing prices water vapour only, and never
    # refuses: the runner admits its forecast again on the cache.
    unnamed = tmp_path / "unnamed"
    unnamed.mkdir()
    assert priced("woof.prepared_domain_tree_forecast", unnamed) is None


def test_multi_run_check_worker_prices_the_configuration_it_reviews(
        tmp_path, monkeypatch):
    """Plan review prices what the run door prices, or it reviews nothing.

    ``woof cli check --alloc`` IS the plan review.  A review that clears
    a shared card the run door refuses minutes later has reviewed a
    different question than the run.
    """

    from woof import multi_run

    config = tmp_path / "experiment.toml"
    config.write_text("[experiment]\nname='fixture'\n", encoding="utf-8")

    seen: dict[str, object] = {}
    _stub_multi_run(monkeypatch, multi_run, seen, 'GPU-beta', 3 * 2 ** 30)

    assert multi_run._locked_check_main(
        gpu_uuid="GPU-beta", config=config, mode="alloc") == 0
    assert Path(str(seen["priced"])) == config
    assert seen["preflight"]["reservation_bytes"] == 3 * 2 ** 30


def test_metem_run_door_prices_the_experiment_it_already_resolved(
        tmp_path, monkeypatch):
    import woof.metem_door as metem_door
    import woof.supervisor as supervisor
    from woof import metem_forecast
    from tests.test_metem_forecast import _launcher_stub_run

    # The door plan-reviews the resolved run (window, vertical ladder,
    # analyzed-field capability, the --outdir claim) before it prices a
    # card, so the run it resolves has to be one that review can read.
    run = _launcher_stub_run()
    experiment = run.experiment
    monkeypatch.setattr(
        metem_door, "resolve_metem_run", lambda directory, **kwargs: run)

    seen: dict[str, object] = {}
    _stub_supervisor(monkeypatch, supervisor, seen, 11 * 2 ** 30)

    source = tmp_path / "source"
    source.mkdir()
    assert metem_forecast.run_metem_forecast(
        source, tmp_path / "forecast") == 0
    assert seen["priced"] is experiment
    assert seen["preflight"]["reservation_bytes"] == 11 * 2 ** 30


def test_wrfinput_run_door_prices_the_experiment_the_directory_declares(
        tmp_path, monkeypatch):
    import woof.supervisor as supervisor
    import woof.wrfinput_door as wrfinput_door
    from woof import wrfinput_forecast
    from tests.test_metem_forecast import _launcher_stub_run

    run = _launcher_stub_run()
    experiment = run.experiment
    monkeypatch.setattr(
        wrfinput_door, "resolve_wrfinput_run", lambda directory, **kwargs: run)

    seen: dict[str, object] = {}
    _stub_supervisor(monkeypatch, supervisor, seen, 5 * 2 ** 30)

    # The --outdir claim refuses an output inside the input tree, so the
    # two are siblings here as they are on a real command line.
    source = tmp_path / "source"
    source.mkdir()
    assert wrfinput_forecast.run_wrf_forecast(
        source, tmp_path / "forecast") == 0
    assert seen["priced"] is experiment
    assert seen["preflight"]["reservation_bytes"] == 5 * 2 ** 30


def test_real74_500m_gate_prices_the_configuration_it_launches(
        tmp_path, monkeypatch):
    from woof import supervisor
    from tools import run_real74_nssl2_500m as runner

    config = tmp_path / "effective.toml"
    config.write_text("config\n", encoding="utf-8")

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        supervisor, "select_gpu",
        lambda uuid: SimpleNamespace(uuid="GPU-test", driver_version="1",
                                     name="test"))
    monkeypatch.setattr(
        supervisor, "priced_reservation_bytes",
        _price_recorder(seen, 13 * 2 ** 30))
    monkeypatch.setattr(
        supervisor, "preflight_exclusive_gpu",
        lambda uuid, **kwargs: seen.update(preflight=kwargs))
    monkeypatch.setattr(
        runner.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=3, stdout="{}", stderr="allocation failed"))

    with pytest.raises(RuntimeError, match="failed closed"):
        runner.gpu_allocation_preflight(config, tmp_path / "run", "GPU-test")
    assert Path(str(seen["priced"])) == config
    assert seen["preflight"]["reservation_bytes"] == 13 * 2 ** 30


def _stub_gate_supervisor(monkeypatch, *, reservation, outcome):
    import woof.supervisor as supervisor

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        supervisor, "select_gpu",
        lambda uuid: SimpleNamespace(uuid="GPU-alpha"))
    monkeypatch.setattr(
        supervisor, "priced_reservation_bytes",
        lambda configuration: reservation)

    def answer(uuid, **kwargs):
        seen.update(kwargs)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(supervisor, "preflight_exclusive_gpu", answer)
    return seen


def test_gate_waits_exactly_while_the_run_door_would_refuse(
        tmp_path, monkeypatch, capsys):
    import woof.supervisor as supervisor

    config = tmp_path / "arm.toml"
    config.write_text("[experiment]\n", encoding="utf-8")
    refusal = supervisor.GPUPreflightError(
        "GPU GPU-alpha cannot admit this run beside its CUDA co-tenant(s) "
        "(pid=41001 name='python.exe' memory=30000MiB): the run's priced "
        "reservation is 30.00 GiB and the device reports 1.00 GiB free")
    seen = _stub_gate_supervisor(
        monkeypatch, reservation=30 * 2 ** 30, outcome=refusal)

    assert _gate_module().main(["--config", str(config)]) == 1
    assert seen["reservation_bytes"] == 30 * 2 ** 30
    printed = capsys.readouterr().out
    assert "BUSY" in printed
    assert "pid=41001" in printed


def test_gate_clears_as_soon_as_the_run_door_would_admit(
        tmp_path, monkeypatch, capsys):
    config = tmp_path / "arm.toml"
    config.write_text("[experiment]\n", encoding="utf-8")
    receipt = {
        "verdict": "admitted",
        "cotenants": "pid=41001 name='python.exe' memory=2000MiB",
        "device_free_bytes": 28 * 2 ** 30,
    }
    seen = _stub_gate_supervisor(
        monkeypatch, reservation=8 * 2 ** 30, outcome=receipt)

    assert _gate_module().main(["--config", str(config)]) == 0
    assert seen["reservation_bytes"] == 8 * 2 ** 30
    printed = capsys.readouterr().out
    assert "CLEAR" in printed
    assert "8.00 GiB" in printed
    assert "28.00 GiB" in printed


def test_gate_with_no_configuration_says_so_and_still_waits(
        monkeypatch, capsys):
    """No configuration is not a licence to print CLEAR beside a co-tenant.

    The gate cannot price what it was not given, so it names the
    co-tenant, names ``--config`` as the way to the priced answer, and
    waits.  Printing CLEAR here is what started an arm onto a held card.
    """

    receipt = {
        "verdict": "admitted-unpriced-run",
        "cotenants": "pid=41001 name='python.exe' memory=30000MiB",
        "device_free_bytes": 1 * 2 ** 30,
    }
    seen = _stub_gate_supervisor(
        monkeypatch, reservation=None, outcome=receipt)

    assert _gate_module().main([]) == 1
    assert seen["reservation_bytes"] is None
    printed = capsys.readouterr().out
    assert "BUSY" in printed
    assert "pid=41001" in printed
    assert "--config" in printed


def test_gate_clears_an_exclusive_card_with_no_configuration(
        monkeypatch, capsys):
    receipt = {"verdict": "exclusive", "cotenants": "",
               "device_free_bytes": None}
    _stub_gate_supervisor(monkeypatch, reservation=None, outcome=receipt)

    assert _gate_module().main([]) == 0
    assert "CLEAR" in capsys.readouterr().out


def test_sweep_queue_gates_on_the_configuration_its_arm_launches(
        tmp_path, monkeypatch):
    from tools import da_sweep_run

    commands: list[list[str]] = []

    def record(command, **_kwargs):
        commands.append(list(command))
        return SimpleNamespace(returncode=0, stdout="gate: CLEAR", stderr="")

    monkeypatch.setattr(da_sweep_run.subprocess, "run", record)
    clear, _detail = da_sweep_run.gate_clear(
        Path("gate.py"), tmp_path / "gate.log", tmp_path / "arm.toml")
    assert clear
    assert commands[0][-2:] == ["--config", str(tmp_path / "arm.toml")]

    commands.clear()
    da_sweep_run.gate_clear(Path("gate.py"), tmp_path / "gate.log")
    assert "--config" not in commands[0]


def test_sweep_queue_binds_the_gate_configuration_tokens(tmp_path):
    from tools import da_sweep_run

    bound = da_sweep_run.expand_tokens(
        "${REPO}/configs/arm.toml", run_dir=tmp_path, repo=Path("/repo"),
        case_root=None)
    assert bound == "/repo/configs/arm.toml"


def test_supervisor_prices_the_attached_card_census(monkeypatch):
    import woof.core.preflight as pf
    import woof.core.streaming as streaming
    import woof.supervisor as supervisor

    profile = pf.DeviceLocalMemoryProfile(
        name="5070 Ti fixture", multiprocessor_count=70,
        max_threads_per_multiprocessor=1536, default_stack_limit_bytes=1024)
    payload = {"total_bytes": 16 * 2**30}
    monkeypatch.setattr(pf, "device_memory_probe_subprocess", lambda: payload)
    monkeypatch.setattr(pf, "profile_from_device_probe", lambda p: profile if p is payload else None)
    machine = SimpleNamespace(device_profile=profile)
    seen = {}
    def machine_from_reading(**kwargs):
        seen.update(kwargs)
        return machine
    monkeypatch.setattr(streaming, "planner_machine", machine_from_reading)
    def estimate(exp, *, machine, source):
        assert source == "gfs"
        return SimpleNamespace(peak_envelope_bytes=machine.device_profile.reservation_bytes(65136))
    monkeypatch.setattr(pf, "admission_estimate", estimate)
    price = supervisor.priced_reservation_bytes(object(), source="gfs")
    assert price == profile.reservation_bytes(65136)
    assert price < pf.MEASURED_LOCAL_MEMORY_PROFILE.reservation_bytes(65136)
    assert seen == dict(vram_bytes=16 * 2**30, name=profile.name, device_profile=profile)


def test_supervisor_keeps_reference_pricing_when_card_probe_is_unread(monkeypatch):
    import woof.core.preflight as pf
    import woof.supervisor as supervisor
    monkeypatch.setattr(pf, "device_memory_probe_subprocess", lambda: None)
    def estimate(exp, *, source):
        return SimpleNamespace(peak_envelope_bytes=123)
    monkeypatch.setattr(pf, "admission_estimate", estimate)
    assert supervisor.priced_reservation_bytes(object()) == 123
