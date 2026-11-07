from contextlib import nullcontext
from datetime import datetime
import json
from types import SimpleNamespace

import pytest

from woof.ensemble.admission import EnsembleMemoryModel, MemoryComponent
from woof.ensemble.packing import CardBudget
from woof.ensemble.production import PreparedEnsembleSession
from woof.ensemble.runtime_context import current_capture


#: These sessions run copies of one prepared input on purpose: they test the
#: session's packing, clocks and receipts, not a forecast ensemble.  A
#: session with no member source refuses N > 1 unless its caller says so.
COPIES = "engine mechanics test: every member runs the one prepared input on purpose"


class Collector:
    def __init__(self):
        self.rows = []
        self.finished = False

    def submit(self, **row):
        self.rows.append(row)

    def finish_run(self):
        self.finished = True
        return {"frames": len(self.rows)}

    def require_complete(self):
        return {}


def inputs():
    cfg = SimpleNamespace(dt=3, use_adaptive_time_step=True, mp_physics=16)
    exp = SimpleNamespace(run_seconds=60, start_time=datetime(2024, 1, 1),
                          root=SimpleNamespace(run=cfg))
    return SimpleNamespace(experiment=exp, boundary_interval_seconds=3600,
                           posted_source=object(), shared_static=object())


def session(tmp_path, *, request=None, **kwargs):
    if "input_provider" not in kwargs:
        kwargs.setdefault("identical_members", COPIES)
    return PreparedEnsembleSession({"members": 3} if request is None else request, output_directory=tmp_path,
        cards=(CardBudget(0, 1000),),
        memory_model=EnsembleMemoryModel((MemoryComponent("ordinary", "ordinary", fixed_bytes=100),)),
        device_scope=lambda _: nullcontext(), **kwargs)


def test_original_member_fallback_reuses_preparation_and_keeps_cfg_and_source(tmp_path):
    prepared, collector = inputs(), Collector()
    seen = []
    def runner(member_inputs, *, output_directory, first_products, **kw):
        assert member_inputs is prepared
        assert member_inputs.posted_source is prepared.posted_source
        assert member_inputs.shared_static is prepared.shared_static
        assert member_inputs.experiment.root.run.dt == 3
        assert member_inputs.experiment.root.run.use_adaptive_time_step
        assert member_inputs.experiment.root.run.mp_physics == 16
        assert first_products is None
        capture = current_capture()
        seen.append(capture.member_id)
        capture.submit(state=object(), streamed=prepared.posted_source,
                       metadata={}, refl_field=None, valid_time=prepared.experiment.start_time,
                       grid_id=1)
        return {"status": "PASS", "clock": [3, 4, 6], "wrfout_count": 0}
    result = session(tmp_path, collector=collector).run_prepared(runner, prepared)
    assert result["status"] == "PASS"
    assert seen == [0, 1, 2]
    assert [row["member_id"] for row in collector.rows] == seen
    assert collector.finished
    assert result["member_history_files"] == []
    assert result["packing"]["waves"] == 3
    assert current_capture() is None


def test_thin_diagnostics_bind_default_member_seeds_and_actual_input_authority(tmp_path, monkeypatch):
    from woof.ensemble import batch_product_output
    from woof.ensemble.seeds import member_seed
    prepared = inputs()
    prepared.source = "gfs"
    prepared.authority_sha256 = {"source_manifest": "a" * 64}
    prepared.prepared_head_sha256 = "b" * 64
    captured, calendars = {}, []
    class ThinCollector(Collector):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured.update(kwargs)
            self.member_archive = SimpleNamespace(expect_member_forecast=lambda member, exp:
                calendars.append((member, exp)))
    monkeypatch.setattr(batch_product_output, "HeadlineDiagnosticCollector", ThinCollector)
    run = PreparedEnsembleSession({"members": 2, "base_seed": 67, "retain_member_diagnostics": True},
        output_directory=tmp_path, renderer=tmp_path / "renderer", identical_members=COPIES,
        cards=(CardBudget(0, 1000),), device_scope=lambda _: nullcontext(),
        memory_model=EnsembleMemoryModel((MemoryComponent("ordinary", "ordinary", fixed_bytes=100),)))
    receipt = run.run_prepared(lambda *args, **kwargs: {"status": "PASS"}, prepared)
    assert receipt["status"] == "PASS"
    assert calendars == [(0, prepared.experiment), (1, prepared.experiment)]
    metadata = captured["member_metadata"]
    assert [row["seed"] for row in metadata] == [member_seed(67, 0), member_seed(67, 1)]
    assert [row["member_id"] for row in metadata] == [0, 1]
    assert all(row["prepared_authority_sha256"] == prepared.authority_sha256 for row in metadata)
    assert all(row["prepared_head_sha256"] == prepared.prepared_head_sha256 for row in metadata)


def test_provider_is_called_for_each_global_member_without_ignored_descriptors(tmp_path):
    prepared, collector = inputs(), Collector()
    provided = []
    def provider(*, shared_inputs, member_id, request):
        assert shared_inputs is prepared
        provided.append(member_id)
        return SimpleNamespace(experiment=shared_inputs.experiment, member=member_id)
    run = session(tmp_path, collector=collector, input_provider=provider)
    def runner(member_inputs, **kw):
        assert member_inputs.member == current_capture().member_id
        return {"status": "PASS"}
    run.run_prepared(runner, prepared)
    assert provided == [0, 1, 2]
    # Listed sources that no door binds are refused by name, with the door
    # that does bind a source per member.
    with pytest.raises(ValueError) as refused:
        PreparedEnsembleSession({"members": 1, "sources": ["external"]},
                                output_directory=tmp_path)
    assert "this door binds none of them" in str(refused.value)
    assert "Next: put the same list in a file and run woof ensemble CONFIG --trajectories FILE" in str(refused.value)


def test_wofs_ladder_session_opens_before_its_runtime_provider_is_bound(tmp_path):
    """The config-driven WoFS ladder (ERA5) names a source family and binds its
    member provider in run_experiment, after the session opens: the session must
    open (WOOF 2026-11-06, 21 members from era5, refused as "binds none of them"),
    and a run that reaches members with no provider is still refused by name."""
    ladder = {"members": 3, "recipe": "wofs", "sources": [{"source": "era5", "members": 3}]}
    run = PreparedEnsembleSession(ladder, output_directory=tmp_path)
    assert run.request.recipe == "wofs" and run.request.sources
    with pytest.raises(ValueError) as refused:
        run.run_prepared(lambda *a, **k: {"status": "PASS"}, inputs())
    assert "the wofs recipe needs each member's own prepared source" in str(refused.value)


def test_failure_records_partial_roster_and_never_calls_products_complete(tmp_path):
    prepared, collector = inputs(), Collector()
    seen = []
    def runner(*args, **kw):
        member = current_capture().member_id
        seen.append(member)
        if member == 1:
            raise RuntimeError("non-finite member")
        return {"status": "PASS"}
    with pytest.raises(RuntimeError, match="non-finite member"):
        session(tmp_path, collector=collector).run_prepared(runner, prepared)
    manifest = json.loads((tmp_path / "ensemble-run.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["members_completed"] == [0]
    assert seen == [0, 1]
    assert not collector.finished
    assert current_capture() is None


def lineage_request():
    from woof.ensemble.lineage import build_lineage
    return {"members": 3, "provenance": [build_lineage({"member": member,
        "source": "gefs", "source_member": f"p{member+1:02d}",
        "cycle": "2024-01-01T00", "seed": 42+member, "rung": "source-member"})
        for member in range(3)]}


@pytest.mark.parametrize("outcome", [None, RuntimeError("member failure"), KeyboardInterrupt()])
def test_lineage_files_cover_success_failed_and_unstarted_members(tmp_path, outcome):
    run = session(tmp_path, request=lineage_request(), collector=Collector())
    seen = []
    def runner(*args, output_directory, **kwargs):
        member = current_capture().member_id
        seen.append(member)
        # Lineage exists before the runner, including for members not yet run.
        for row in run.request.provenance:
            retained = tmp_path / "members" / f"member-{row['member']:04d}" / "member-provenance.json"
            assert json.loads(retained.read_text()) == row
        assert output_directory == tmp_path / "members" / f"member-{member:04d}"
        if member == 1 and outcome is not None:
            raise outcome
        return {"status": "PASS"}
    if outcome is None:
        result = run.run_prepared(runner, inputs())
        assert result["status"] == "PASS" and seen == [0, 1, 2]
    else:
        with pytest.raises(type(outcome)) as caught:
            run.run_prepared(runner, inputs())
        assert caught.value is outcome and seen == [0, 1]
        result = json.loads((tmp_path / "ensemble-run.json").read_text())
        assert result["status"] == ("interrupted" if isinstance(outcome, KeyboardInterrupt) else "failed")
    for row in run.request.provenance:
        retained = tmp_path / "members" / f"member-{row['member']:04d}" / "member-provenance.json"
        assert json.loads(retained.read_text()) == row
    assert result["request"]["provenance"] == list(run.request.provenance)


def test_lineage_directory_collision_is_refused_before_any_runner(tmp_path):
    occupied = tmp_path / "members" / "member-0001"
    occupied.mkdir(parents=True)
    marker = occupied / "member-provenance.json"
    marker.write_text('{"owner":"existing run"}')
    seen = []
    run = session(tmp_path, request=lineage_request(), collector=Collector())
    with pytest.raises(FileExistsError):
        run.run_prepared(lambda *args, **kwargs: seen.append(True), inputs())
    assert seen == [] and marker.read_text() == '{"owner":"existing run"}'
    assert not (tmp_path / "ensemble-run.json").exists()


def test_failing_member_receipt_is_not_reported_as_success(tmp_path):
    with pytest.raises(RuntimeError, match="failing forecast receipt"):
        session(tmp_path, collector=Collector()).run_prepared(
            lambda *args, **kw: {"status": "FAIL"}, inputs())
    manifest = json.loads((tmp_path / "ensemble-run.json").read_text())
    assert manifest["members_completed"] == []


def test_sparse_source_roster_binds_each_initial_and_boundary_owner(tmp_path):
    shared, collector = inputs(), Collector()
    member_inputs = {member: SimpleNamespace(experiment=shared.experiment,
        initial=object(), boundaries=(object(), object(), object())) for member in (19, 3, 44)}
    roster = SimpleNamespace(members=tuple(SimpleNamespace(member_id=member) for member in member_inputs),
        receipts=(), select=lambda ids: tuple(SimpleNamespace(inputs=member_inputs[member]) for member in ids),
        receipt=lambda: {"member_order": [19, 3, 44]})
    observed = []
    def runner(prepared, **kw):
        member = current_capture().member_id
        assert prepared is member_inputs[member]
        assert prepared.initial is member_inputs[member].initial
        assert prepared.boundaries is member_inputs[member].boundaries
        observed.append(member)
        return {"status": "PASS"}
    result = session(tmp_path, collector=collector, member_roster=roster).run_prepared(runner, shared)
    assert observed == [19, 3, 44]
    assert result["member_order"] == [19, 3, 44]
    assert result["members_completed"] == [3, 19, 44]


def test_cyclone_hook_consumes_completed_member_histories_before_pass(tmp_path, monkeypatch):
    from dataclasses import replace
    from woof.ensemble import cyclone_products
    collector = Collector()
    collector.history_ledger = object()
    owner = session(tmp_path, collector=collector)
    owner.request = replace(owner.request, cyclone_products={})
    seen = []
    def products(root, trajectories, request, **kwargs):
        assert collector.finished
        assert kwargs['history_ledger'] is collector.history_ledger
        assert kwargs['member_order'] == (0, 1, 2)
        seen.append(root)
        return {'status': 'complete', 'tracks': 'cyclone/tracks/tracks.json'}
    # The ledger validator belongs to the writer, outside this hook check.
    collector.history_ledger = SimpleNamespace(validate_retirements=lambda: [])
    monkeypatch.setattr(cyclone_products, 'render_roster', products)
    result = owner.run_prepared(lambda *a, **kw: {'status': 'PASS'}, inputs())
    assert result['status'] == 'PASS'
    assert result['products']['cyclone']['status'] == 'complete'
    assert set(result['timing_seconds']) == {'forecast', 'products', 'total'}
    assert result['timing_seconds']['total'] >= result['timing_seconds']['products'] >= 0
    saved = json.loads((tmp_path / 'ensemble-run.json').read_text())
    assert saved['products']['cyclone'] == result['products']['cyclone']
    assert saved['timing_seconds'] == result['timing_seconds']
    assert seen == [tmp_path]
