"""dist46 item 1: an answer's valid time from the plan's kept facts, not a whole parse of plan.json a target."""
import inspect
import json
from types import SimpleNamespace

import pytest

from woof.hex import ad_doors as d
from woof.hex import ad_plan_facts as pf


# ------------------------------------------------------------------------------------- item 1: the valid time
@pytest.fixture
def dated_plan(tmp_path, monkeypatch):
    monkeypatch.setenv(pf.FACTS_ENV, str(tmp_path / "facts"))
    pf._MEMO.clear()
    plan = tmp_path / "plan"
    (plan / "recording" / "payloads").mkdir(parents=True)
    doc = {"allocations": [{"id": 3, "bytes": 8, "initial": {"path": "recording/payloads/a.bin"}}],
           "physics": {"dt_seconds": 15, "state_xtime": "2023-03-31_18:00:00\u0000\u0000"}, "modules": [], "blocks": {}}
    (plan / "plan.json").write_text(json.dumps(doc))
    (plan / "recording" / "payloads" / "a.bin").write_bytes(b"\0" * 8)
    parses = []
    real = json.loads

    def loads(s, *a, **k):
        text = s if isinstance(s, str) else s.decode("utf-8", "replace") if isinstance(s, bytes) else ""
        if '"blocks"' in text and '"state_xtime"' in text:   # plan.json itself, not the kept facts file
            parses.append(1)
        return real(s, *a, **k)

    monkeypatch.setattr(json, "loads", loads)
    yield SimpleNamespace(plan=plan, parses=parses)
    pf._MEMO.clear()


def test_an_answers_valid_time_comes_from_the_plans_kept_facts_not_a_parse_of_plan_json(dated_plan):
    """THE BREAKAGE: every answer parsed the whole plan.json (708 MB at the 3 km box: 3.4 s and 4,271 MiB of host
    memory on one machine) to read one date, in no named stage (2026-11-14)."""
    fmap = SimpleNamespace(doc={}, source=dated_plan.plan / "alloc-map.json")
    assert d._valid_time(fmap, 0, 15.0) == "2023-03-31T18:00:00"
    assert d._valid_time(fmap, 4, 15.0) == "2023-03-31T18:01:00"
    assert d._valid_time(fmap, 59, 15.0) == "2023-03-31T18:14:45"
    assert len(dated_plan.parses) <= 1, "plan.json parsed more than once for the date"
    pf._MEMO.clear()            # a later command (another process): the date is kept, nothing is parsed
    dated_plan.parses.clear()
    from woof.hex.ad import rows as _rows
    getattr(_rows, "forget_plan_docs", lambda: None)()
    assert d._valid_time(fmap, 0, 15.0) == "2023-03-31T18:00:00"
    assert dated_plan.parses == []
    assert pf.plan_facts(dated_plan.plan).how == "kept"


def test_the_valid_time_lookup_is_a_named_stage_of_the_answer():
    src = inspect.getsource(d._sensitivity_answer)
    assert 'ctx.stage("valid time (the plan\'s kept facts)"' in src
    assert src.index("_valid_time(") < src.index('ctx.stage("valid time')
    assert 'json.loads((base / "plan.json")' not in inspect.getsource(d._valid_time)
