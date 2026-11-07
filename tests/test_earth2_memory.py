"""Measured capacity gates apply to new inference, after reusable receipts."""
from types import SimpleNamespace
import json

import pytest

from woof import earth2_run, earth2_job


def test_capacity_is_row_driven_and_names_the_selected_model(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-selected")
    commands = []

    def query(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="15841, 15000\n")

    monkeypatch.setattr(earth2_run.subprocess, "run", query)
    row = dict(min_vram_gib=30, vram_note="Measured single allocation 20.37 GiB.")
    with pytest.raises(ValueError, match="another-model.*20.37 GiB"):
        earth2_run.require_inference_vram("another-model", row)
    assert commands[0][-1] == "--id=GPU-selected"
    assert earth2_run.require_inference_vram("small-model", {}) is None


def test_nominal_32gb_card_can_meet_measured_floor(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(earth2_run.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout="32607, 32000\n"))
    receipt = earth2_run.require_inference_vram("large-model", dict(min_vram_gib=30))
    assert receipt["total_mib"] == 32607
    assert receipt["minimum_gib"] == 30


def test_reused_fetch_does_not_enter_inference_capacity_gate(tmp_path, monkeypatch):
    from woof import earth2_fetch_cache
    from woof.earth2_door import build_spec
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(build_spec(model="fcn3", cycle="2024-09-24T12",
                                         hours=1, storm="auto", nest="2.5km")))
    source = tmp_path / "completed-fetch"

    def reusable(spec, out, input_dir):
        assert spec["hours"] == 6
        assert input_dir == source
        return {"cache_reuse": True}

    monkeypatch.setattr(earth2_fetch_cache, "reuse", reusable)


    def unexpected(**kwargs):
        pytest.fail("a verified reusable receipt must avoid new model inference")

    monkeypatch.setattr(earth2_run, "run_model", unexpected)
    monkeypatch.setattr(earth2_run, "require_inference_vram", unexpected)
    earth2_job._fetch(path, tmp_path / "output", source)


@pytest.mark.parametrize('model_id, minimum', [
    pytest.param('added-table-row', 21, id='added-table-row-21'),
])
def test_fresh_fetch_refuses_before_runtime_downloads_or_output(tmp_path, monkeypatch,
                                                             model_id, minimum):
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    from woof import earth2_fetch_cache
    from woof.earth2_door import build_spec
    row = dict(earth2_run.resolve_model("fcn3")[1], min_vram_gib=minimum)
    spec = build_spec(model="fcn3", cycle="2024-09-24T12", hours=1,
                      storm="auto", nest="2.5km")
    spec["model"] = model_id
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    output = tmp_path / "output"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-capacity-fixture")
    monkeypatch.setattr(earth2_run, "resolve_model", lambda name: (model_id, row))
    monkeypatch.setattr(earth2_fetch_cache, "reuse", lambda *a: None)
    monkeypatch.setattr(earth2_run.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout="15841, 15000\n"))

    def unexpected(*args, **kwargs):
        pytest.fail("a memory refusal must precede runtime loading and downloads")

    monkeypatch.setattr(earth2_run, "earth2_python", unexpected)
    monkeypatch.setattr(earth2_run, "_donor", unexpected)
    monkeypatch.setattr(earth2_fetch_cache, "remember", unexpected)
    with pytest.raises(ValueError, match=rf"{model_id}: inference needs {minimum} GiB.*16,552 MiB"):
        earth2_job._fetch(path, output)
    assert not output.exists()


def test_fcn3_row_admits_by_its_measured_peak(monkeypatch):
    # Defect (2.9.4 follow-up, test host 2026-10-21): the runner's DISCO band patch did not apply to
    # the pinned torch-harmonics 0.8.1, FCN3 peaked at 60,212 MiB and the 59 GiB floor refused
    # every 32 GB card. With the 0.8.1 layout patched it measured a 16,552 MiB card peak under
    # a 30,000 MiB lease, so 32 GB (and 24 GB) cards admit and a 16 GB card still refuses.
    row = earth2_run.resolve_model("fcn3")[1]
    assert 16552 <= row["min_vram_gib"] * 1024 <= 30000
    assert "16,552 MiB" in row["vram_note"]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(earth2_run.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout="16303, 15841\n"))
    with pytest.raises(ValueError, match="fcn3: inference needs 17 GiB"):
        earth2_run.require_inference_vram("fcn3", row)
    monkeypatch.setattr(earth2_run.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout="32607, 32000\n"))
    assert earth2_run.require_inference_vram("fcn3", row)["minimum_gib"] == 17
