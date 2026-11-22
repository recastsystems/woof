"""WOOF_PIPELINE_WORKERS overrides the HRRR decoder worker count without
breaking the preparation's receipt check.

Breakage it prevents (a rented 8-card box, 2026-10-27): the override rewrote the
receipt's "requested" to the environment value, so
prepare_hrrr_wrf._validated_worker_receipts (which compares "requested" with
the count it passed) refused every preparation after ~10 min with
"HRRR decoder worker receipt is malformed".
"""
from tools import hrrr_pipeline


def test_override_selects_env_count_and_keeps_requested(monkeypatch):
    monkeypatch.setenv(hrrr_pipeline.PIPELINE_WORKERS_ENV, "4")
    workers, receipt = hrrr_pipeline.resolve_workers("8")
    assert workers == 4
    assert receipt["selected"] == 4
    assert receipt["requested"] == "8"
    assert receipt["override"] == {"env": hrrr_pipeline.PIPELINE_WORKERS_ENV, "value": "4"}
    assert hrrr_pipeline.PIPELINE_WORKERS_ENV in receipt["policy"]


def test_no_override_receipt_unchanged(monkeypatch):
    monkeypatch.delenv(hrrr_pipeline.PIPELINE_WORKERS_ENV, raising=False)
    workers, receipt = hrrr_pipeline.resolve_workers("8")
    assert workers == 8
    assert receipt["requested"] == "8" and receipt["policy"] == "explicit"
    assert "override" not in receipt
