"""`woof go --prepare-only` on a two-domain config prepares and stops.

The breakage, measured on box B on 2026-10-06 with gpuwm 2.8.6: the
two-domain Toronto 750 m city template (hrrr-prs, 750 m nest) launched
with --prepare-only ran the whole 12 h forecast anyway (27 history
writes, 441 s), while a single-domain GFS config honoured the flag.  The
HRRR route went from the door to the registered run-plan launch, which
has no prepare-only option, so the chain was never told.

These drive the real door (``woof.cli.main``) and the real native HRRR
chain on an authored nested configuration, with the stages captured at
the ``run_stage`` seam and every forecast entry point armed to fail the
test.  The other flags checked here are the ones the same door branch
dropped: --dry-run, --outdir/--run-stamp, and the inputs-already-prepared
flags that routed a prepare-only run to a forecasting launch.

CPU-only.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest

import woof.capabilities as capabilities
import woof.go_cli as go_cli
import woof.prepared_domain_tree_forecast as tree
import woof.prepared_single_domain_forecast as single
import woof.runplan as runplan_module
from woof.cli import main
from woof.runplan import generate_intent_config

from test_runplan_hrrr_tree import _plan


#: A fetched-source receipt in the real fetch's shape.  Since abf04433ed8
#: (parallel cache reuse) the chain snapshots it (launch_supplements.
#: hrrr_source_manifest) right after the fetch, so a placeholder byte is a
#: malformed manifest that stops the chain before the preparation stage.
_FETCHED_SUMS = "0" * 64 + "  hrrr.t12z.wrfprsf00.grib2\n"


def _nested_config(tmp_path: Path) -> Path:
    """A two-domain native HRRR config exactly as `woof domain` writes it."""

    plan = _plan(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        config, _ = generate_intent_config(plan, destination=tmp_path / "authored")
    from woof.experiment import load_experiment

    assert len(load_experiment(config).domains) == 2
    return config


def _forecast_tripwires(monkeypatch) -> None:
    """Every way a forecast can start fails the test, and no card is asked for."""

    def forecast(*_args, **_kwargs):
        raise AssertionError("a forecast step ran under --prepare-only")

    for module, name in ((runplan_module, "_hrrr_tree_forecast"),
                         (runplan_module, "_chain_render"),
                         (runplan_module, "execute_plan"),
                         (tree, "main"), (single, "main"),
                         (go_cli, "_registered_launch")):
        monkeypatch.setattr(module, name, forecast)

    def card(*_args, **_kwargs):
        raise AssertionError("--prepare-only asked for the forecast's card")

    monkeypatch.setattr(go_cli, "_require_forecast_device", card)
    monkeypatch.setattr(go_cli, "memory_gate", card)
    monkeypatch.setattr(capabilities, "require_for_command", card)
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    monkeypatch.delenv("WOOF_CONTINUATION_PREFIX", raising=False)


def _captured_stages(monkeypatch) -> list[tuple[str, list[str]]]:
    staged: list[tuple[str, list[str]]] = []

    def run_stage(label, command, **_kwargs):
        staged.append((label, [str(part) for part in command]))
        if label == "hierarchy":
            root = Path(command[command.index("--output-root") + 1])
            root.mkdir(parents=True, exist_ok=True)
            (root / "receipt.json").write_text(json.dumps(
                {"schema": "gpuwm-native-hrrr-hierarchy-direct-v1",
                 "status": "PASS"}), encoding="utf-8")

    def fetch(arguments, run_dir, **_kwargs):
        out = Path(arguments[arguments.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "SHA256SUMS").write_text(_FETCHED_SUMS, encoding="utf-8")
        staged.append(("fetch", [str(a) for a in arguments]))
        return {}

    monkeypatch.setattr(go_cli, "run_stage", run_stage)
    monkeypatch.setattr(runplan_module, "_run_fetch", fetch)
    return staged


def _go(argv, capsys) -> tuple[int, str]:
    rc = main(["go", *argv])
    said = capsys.readouterr()
    assert rc == 0, said.err
    return rc, said.out


def test_prepare_only_on_a_two_domain_config_runs_no_forecast_step(
        tmp_path, monkeypatch, capsys):
    config = _nested_config(tmp_path)
    _forecast_tripwires(monkeypatch)
    staged = _captured_stages(monkeypatch)
    out = tmp_path / "out"

    rc, said = _go([str(config), "--prepare-only", "--products", "none",
                    "--outdir", str(out)], capsys)

    assert rc == 0
    # fetch, the root preparation and the hierarchy stage ran; nothing after.
    assert [label for label, _ in staged] == ["fetch", "prepare", "hierarchy"]
    lines = said.strip().splitlines()
    result = json.loads(lines[-1])
    assert result["status"] == "PREPARED"
    # The terminal says the stages; the event records go to their file only.
    assert "go: prepare" in lines
    assert not any("schema_version" in line for line in lines[:-1])
    prepared = Path(result["prepared_root"])
    assert prepared.name == "hrrr-hierarchy"
    assert (prepared / "receipt.json").is_file()
    # No history was written anywhere under the run.
    assert not list(out.rglob("wrfout_*"))
    assert not (prepared.parent / "run").exists()


def test_prepare_only_on_a_two_domain_hrrr_prs_config_reaches_its_chain_told(
        tmp_path, monkeypatch, capsys):
    """The box B config's own source: hrrr-prs takes the staged chain, and
    2.8.6 handed it to the forecasting run-plan launch instead."""

    config = _nested_config(tmp_path)
    text = config.read_text(encoding="utf-8")
    head, _, tail = text.partition("[fetch]")
    _body, sep, rest = tail.partition("\n[")
    # The Toronto template's own [fetch] shape: hrrr-prs publishes whole
    # objects and refuses an area crop.
    body = ('\nsource = "hrrr-prs"\ncycle = "2024-05-03T12"\nhours = 6\n'
            'cadence = 1\n')
    config.write_text(head + "[fetch]" + body + sep + rest, encoding="utf-8")
    _forecast_tripwires(monkeypatch)
    calls = []

    def chain(plan, *, config_path, exp, observer, run_dir, prepare_only):
        calls.append((len(exp.domains), prepare_only))
        return {"prepared_root": str(run_dir / "chain" / "prep")}

    monkeypatch.setattr(runplan_module, "_staged_chain", chain)
    rc, said = _go([str(config), "--prepare-only", "--outdir", str(tmp_path / "out")],
                   capsys)
    assert rc == 0
    assert calls == [(2, True)]
    assert json.loads(said.strip().splitlines()[-1])["status"] == "PREPARED"


def test_prepare_only_claims_a_stamped_run_folder_under_outdir(
        tmp_path, monkeypatch, capsys):
    """--outdir is the case folder and the run gets its own run-... child,
    as on every other go route; --run-stamp off puts it in --outdir itself."""

    config = _nested_config(tmp_path)
    _forecast_tripwires(monkeypatch)
    _captured_stages(monkeypatch)
    out = tmp_path / "out"

    rc, said = _go([str(config), "--prepare-only", "--outdir", str(out)], capsys)
    assert rc == 0
    prepared = Path(json.loads(said.strip().splitlines()[-1])["prepared_root"])
    run = prepared.parents[1]
    assert run.parent == out.resolve()
    assert run.name.startswith("run-")

    flat = tmp_path / "flat"
    rc, said = _go([str(config), "--prepare-only", "--outdir", str(flat),
                    "--run-stamp", "off"], capsys)
    assert rc == 0
    prepared = Path(json.loads(said.strip().splitlines()[-1])["prepared_root"])
    assert prepared.parents[1] == flat.resolve()


def test_two_prepare_only_launches_in_one_second_get_two_run_folders(
        tmp_path, monkeypatch, capsys):
    """Release review 17 (2.9.7): the folder was only predicted before the
    admission and then made with exist_ok=True, so a second launch of the
    same second that predicted while the first was being admitted ran in
    the first one's folder. Launch B runs whole inside launch A's admission."""

    import datetime

    from woof import run_stamp
    from woof import runplan as runplan_module

    config = _nested_config(tmp_path)
    _forecast_tripwires(monkeypatch)
    _captured_stages(monkeypatch)
    frozen = datetime.datetime(2026, 10, 24, 8, 0, 0, tzinfo=datetime.timezone.utc)
    monkeypatch.setattr(run_stamp, "utcnow", lambda: frozen)
    out = tmp_path / "out"

    admit = runplan_module.disk_admission_refusal
    inner = []

    def admission(*args, **kwargs):
        if not inner:
            inner.append(None)
            rc = main(["go", str(config), "--prepare-only", "--outdir", str(out)])
            assert rc == 0
            inner[0] = Path(json.loads(capsys.readouterr().out.strip().splitlines()[-1])
                            ["prepared_root"]).parents[1]
        return admit(*args, **kwargs)

    monkeypatch.setattr(runplan_module, "disk_admission_refusal", admission)
    rc, said = _go([str(config), "--prepare-only", "--outdir", str(out)], capsys)
    outer = Path(json.loads(said.strip().splitlines()[-1])["prepared_root"]).parents[1]
    assert inner[0] is not None
    assert outer != inner[0], "two launches shared one run folder"
    assert {outer.parent, inner[0].parent} == {out.resolve()}
    assert run_stamp.latest(out).resolve() == outer.resolve()


def test_prepare_only_dry_run_spends_nothing(tmp_path, monkeypatch, capsys):
    config = _nested_config(tmp_path)
    _forecast_tripwires(monkeypatch)
    staged = _captured_stages(monkeypatch)
    out = tmp_path / "out"

    rc, said = _go([str(config), "--prepare-only", "--dry-run",
                    "--outdir", str(out)], capsys)

    assert rc == 0
    assert staged == []
    assert not out.exists()
    assert "2 domain(s); fetch -> prepare" in said
    assert "no forecast" in said


@pytest.mark.parametrize("flag", ["--wps-namelist", "--prepared-root", "--restart"])
def test_prepare_only_with_prepared_inputs_is_refused_not_forecast(
        tmp_path, monkeypatch, capsys, flag):
    """These flags routed the door to the forecasting run-plan launch above
    the prepare-only branch, so a prepare-only command forecast them."""

    config = _nested_config(tmp_path)
    _forecast_tripwires(monkeypatch)
    staged = _captured_stages(monkeypatch)
    named = tmp_path / "named"
    named.mkdir()
    argv = [str(config), "--prepare-only", flag, str(named),
            "--outdir", str(tmp_path / "out")]
    if flag == "--restart":
        argv += ["--prepared-root", str(named)]

    rc = main(["go", *argv])

    assert rc != 0
    assert staged == []
    assert "--prepare-only" in capsys.readouterr().err


def test_prepare_only_on_a_case_data_config_is_refused_not_forecast(
        tmp_path, monkeypatch, capsys):
    """A [case_data] config runs on the experiment route, which prepares
    inside the forecast with no seal to stop at."""

    config = _nested_config(tmp_path)
    config.write_text(config.read_text(encoding="utf-8")
                      + '\n[case_data]\nforcing = "nowhere"\n', encoding="utf-8")
    _forecast_tripwires(monkeypatch)
    staged = _captured_stages(monkeypatch)

    rc = main(["go", str(config), "--prepare-only",
               "--outdir", str(tmp_path / "out")])

    assert rc != 0
    assert staged == []
    assert "[case_data]" in capsys.readouterr().err
