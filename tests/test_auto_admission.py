"""Auto selection must use the same configured admission as its caller."""
from dataclasses import asdict, replace
import json

import pytest

from woof.core import preflight as pf, streaming as st
from woof.experiment import DomainConfig, experiment_config_document
from tilestream import autoplan as ap
from tilestream.test_ledger_gate import _exp

GIB = 1024 ** 3


def _windows_estimate(exp, **kwargs):
    # Exercise the measured WDDM arithmetic on either test host, without a GPU.
    return replace(pf.estimate_experiment(exp, **kwargs), envelope_family="windows")


def _tree(root_n=704, child_n=64):
    exp = _exp(root_n, "auto")
    root = exp.root
    run = replace(root.run, grid_id=2, nx=child_n, ny=child_n,
                  dx=root.run.dx / 3, dy=root.run.dy / 3, dt=root.run.dt / 3,
                  nested=True, specified=False)
    child = DomainConfig(grid_id=2, parent_id=1, i_parent_start=20,
                         j_parent_start=20, parent_grid_ratio=3,
                         parent_time_step_ratio=3,
                         history_interval_s=root.history_interval_s, run=run)
    return replace(exp, domains=(root, child))


@pytest.mark.parametrize("n,free_gib", [(704, 23.5), (960, 31.4)])
def test_old_empirical_fit_cannot_strand_an_auto_forecast(n, free_gib):
    exp = _exp(n, "auto")
    machine = ap.Machine(int(free_gib * GIB), 256 * GIB)
    estimate = _windows_estimate(exp)
    # Discriminating control: the old decision really did select resident.
    old = st.decide(exp.root.run, replace(exp.tiles, resident_context=None),
                    machine=machine)
    assert not old.stream
    assert estimate.peak_envelope_bytes > machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
    decision = st.decide(exp.root.run, exp.tiles, machine=machine,
                         resident_estimate=estimate)
    assert decision.stream
    envelope = st.streamed_envelope(exp.root.run, exp.tiles,
                                    machine=machine, decision=decision)
    assert envelope.peak_vram_bytes <= machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
    assert decision.detail["resident_admission"]["envelope_bytes"] == estimate.peak_envelope_bytes
    assert decision.detail["resident_admission"]["budget_bytes"] == machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES


@pytest.mark.parametrize("n,free_gib,streams", [(512, 23.5, False),
                                                (816, 23.5, True),
                                                (976, 31.4, True)])
def test_safe_resident_and_adjacent_streaming_roads_remain(n, free_gib, streams):
    exp = _exp(n, "auto")
    machine = ap.Machine(int(free_gib * GIB), 256 * GIB)
    decision = st.decide(exp.root.run, exp.tiles, machine=machine,
                         resident_estimate=_windows_estimate(exp))
    assert decision.stream is streams


def test_an_admitted_small_resident_grid_does_not_pay_the_tile_reserve_twice():
    exp = _exp(200, "auto")
    machine = ap.Machine(int(7.25 * GIB), 128 * GIB)
    estimate = _windows_estimate(exp)
    assert estimate.peak_envelope_bytes < machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
    # The coarse, context-free tile footprint still has its own use, but it
    # must not overturn the complete configuration's admitted resident road.
    coarse = st.decide(exp.root.run, replace(exp.tiles, resident_context=None), machine=machine)
    assert coarse.stream
    selected = st.decide(exp.root.run, exp.tiles, machine=machine,
                         resident_estimate=estimate)
    assert not selected.stream
    assert selected.resident_bytes == estimate.peak_envelope_bytes
    assert selected.budget_bytes == machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
    assert selected.detail["host_claim_bytes"] == 0


def test_exact_phase_estimate_is_forwarded_once_with_actual_operands(monkeypatch):
    exp = replace(_exp(704, "auto"), column_chunk=157,
                  vertical=replace(_exp(704).vertical, p_top=10000.0))
    machine = ap.Machine(int(23.5 * GIB), 256 * GIB)
    original = pf.estimate_experiment
    calls = []
    estimates = []
    def counted(value, **kwargs):
        calls.append((value, kwargs))
        result = replace(original(value, **kwargs), envelope_family="windows")
        estimates.append(result)
        return result
    monkeypatch.setattr(pf, "estimate_experiment", counted)
    monkeypatch.setattr(ap.Machine, "detect", lambda **kw: pytest.fail("second device observation"))
    phases = pf.estimate_phases(exp, source=None, machine=machine,
        forcing_intervals=47, profile=pf.MEASURED_LOCAL_MEMORY_PROFILE, column_chunk=91)
    assert len(calls) == 1
    assert calls[0][0] is exp
    assert calls[0][1]["forcing_intervals"] == 47
    assert calls[0][1]["profile"] is pf.MEASURED_LOCAL_MEMORY_PROFILE
    assert calls[0][1]["column_chunk"] == 91
    assert phases.forecast is estimates[0]
    assert phases.forecast.column_chunk == 91
    # The configured snapshot has the non-default scientific controls too.
    context = exp.tiles.resident_context.experiment
    assert context.column_chunk == 157 and context.vertical.p_top == 10000.0


def test_context_is_finite_and_not_traversed_by_public_identity():
    from woof.core.model import restart_identity_payload
    exp = _tree()
    assert exp.tiles.resident_context.experiment.tiles.resident_context is None
    assert all(dc.tiles is None for dc in exp.tiles.resident_context.experiment.domains)
    assert "resident_context" in asdict(exp)["tiles"]  # finite, even for raw callers
    original_document = experiment_config_document(exp)
    original_identity = restart_identity_payload(exp)
    class DoNotCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("derived planner state entered public serialization")
    object.__setattr__(exp, "tiles", replace(exp.tiles,
        resident_context=st.ResidentAdmissionContext(DoNotCopy())))
    assert experiment_config_document(exp) == original_document
    assert restart_identity_payload(exp) == original_identity
    text = json.dumps(original_document, default=str)
    assert "resident_context" not in text and "radiation_context" not in text
    assert original_document["tiles"] == exp.tiles.to_mapping()


def test_tree_just_over_budget_keeps_the_earlier_resident_preference():
    exp = _tree()
    estimate = _windows_estimate(exp)
    machine = ap.Machine(estimate.peak_envelope_bytes + pf.EXTERNAL_MARGIN_BYTES - 1,
                         256 * GIB)
    rows = {}
    outcome = st.decide_tree(st._config_tree_nodes(exp.domains), exp.tiles,
                            machine=machine, decisions=rows, resident_estimate=estimate)
    assert not rows[1].stream and rows[2].stream
    assert outcome.resident_subset_envelope_bytes < estimate.peak_envelope_bytes
    assert outcome.resident_subset_envelope_bytes <= machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
    assert rows[2].detail["resident_admission"]["preference_changed"]


def test_streaming_a_tiny_child_does_not_hide_an_oversized_resident_parent():
    exp = _tree()
    estimate = _windows_estimate(exp)
    nodes = st._config_tree_nodes(exp.domains)
    root_bound = st._resident_subset_envelope(estimate, nodes, {1})
    machine = ap.Machine(root_bound + pf.EXTERNAL_MARGIN_BYTES - 1, 256 * GIB)
    rows = {}
    result = st.decide_tree(nodes, exp.tiles, machine=machine, decisions=rows,
                           resident_estimate=estimate)
    assert rows[1].stream
    assert not rows[2].stream  # mixed road fits; do not stream everything
    assert result.resident_subset_envelope_bytes < root_bound


def test_declared_budget_is_not_charged_external_margin_twice():
    # 24.25 GiB: the configured envelope is 24.08 GiB since the WRF exact
    # chain's history rows (2.9.1: P_HYD and RTHRATEN planes, eight surface
    # planes) joined it, so the declared budget still sits just above the
    # envelope and a second external margin would still flip it to streaming.
    budget = int(24.25 * GIB)
    exp = _exp(704, "auto", vram_budget_bytes=budget)
    estimate = _windows_estimate(exp)
    machine = ap.Machine(64 * GIB, 256 * GIB)
    decision = st.decide(exp.root.run, exp.tiles, machine=machine, resident_estimate=estimate)
    assert decision.detail["resident_admission"]["budget_bytes"] == budget
    assert not decision.stream


def test_explicit_roads_do_not_price_or_probe_resident_admission(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("explicit road consulted configured auto admission")
    monkeypatch.setattr(pf, "estimate_experiment", forbidden)
    monkeypatch.setattr(ap.Machine, "detect", forbidden)
    cfg = _exp(704).root.run
    assert not st.decide(cfg, st.OFF).stream
    pinned = st.StreamingOptions(mode="on", tile_nx=176, tile_ny=176, nbuffers=2)
    decision = st.decide(cfg, pinned)
    assert decision.stream and decision.tile_nx == 176 and decision.nbuffers == 2
    assert st.cold_planning_machine(replace(_exp(704), tiles=pinned)) is None


def test_cold_machine_is_captured_once_and_reused_during_geometry_refinement(monkeypatch):
    from types import SimpleNamespace
    from woof import runtime
    exp = _exp(704, "auto")
    machine = ap.Machine(24 * GIB, 256 * GIB)
    calls = []
    monkeypatch.setattr(ap.Machine, "detect", lambda **kw: calls.append(kw) or machine)
    captured = st.cold_planning_machine(exp)
    assert captured is machine and len(calls) == 1
    cfg = replace(exp.root.run, use_adaptive_time_step=True)
    prepared = SimpleNamespace(cfg=cfg, grid=SimpleNamespace(
        mapfac_u=lambda: [[1.0]], mapfac_v=lambda: [[1.0]]))
    estimate = _windows_estimate(exp)
    observed = []
    monkeypatch.setattr(st, "decide", lambda cfg, options, **kw: observed.append(kw) or "decision")
    _, result = runtime._refine_single_streaming_plan(prepared, exp.tiles, object(), captured, estimate)
    assert result == "decision" and observed == [{"machine": machine, "resident_estimate": estimate}]
    assert len(calls) == 1


def test_mixed_fit_pays_for_streamed_marginal_and_corridor_beside_resident():
    exp = _tree(child_n=256)
    estimate = _windows_estimate(exp)
    nodes = st._config_tree_nodes(exp.domains)
    root_bound = st._resident_subset_envelope(estimate, nodes, {1})
    budget = root_bound + 1024 ** 2
    rows = {}
    result = st.decide_tree(nodes, exp.tiles,
        machine=ap.Machine(budget + pf.EXTERNAL_MARGIN_BYTES, 256 * GIB),
        decisions=rows, resident_estimate=estimate)
    # The old max-only admission selected a resident root plus >2GiB of
    # child tile/corridor allocations with merely1MiB beside the root bound.
    assert rows[1].stream
    streamed_claims = sum(int(d.detail.get(key, 0))
        for d in rows.values() if d.stream
        for key in ("claim_bytes", "corridor_claim_bytes"))
    if any(not d.stream for d in rows.values()):
        assert result.configured_mixed_envelope_bytes == (
            result.resident_subset_envelope_bytes + streamed_claims)
    assert result.configured_mixed_envelope_bytes <= budget


def test_immutable_process_floor_prunes_many_auto_preferences(monkeypatch):
    exp = _tree(64, 32)
    children = tuple(replace(exp.domains[1], grid_id=gid,
                            run=replace(exp.domains[1].run, grid_id=gid))
                     for gid in range(2, 11))
    exp = replace(exp, domains=(exp.root, *children))
    original = st._decide_tree
    calls = []
    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(st, "_decide_tree", counted)
    with pytest.raises(st.StreamingRefused, match="shared process/radiation floor"):
        st.decide_tree(st._config_tree_nodes(exp.domains), exp.tiles,
                      machine=ap.Machine(GIB, 256 * GIB))
    assert len(calls) == 1  # previously1025 identical impossible walks


def test_auto_ordinary_dispatch_takes_one_admission_before_the_fetch(tmp_path, monkeypatch):
    """The single-domain ``auto`` route prices ONE admission, and early.

    THE GUARD THIS RETIRES pinned the opposite order.  It required this
    route to build the input catalog, decode every forcing snapshot and
    only then price ``estimate_experiment(exp, forcing_intervals=47)`` --
    the schedule's ACTUAL retained boundary count, which is knowable only
    after the case has been fetched.  That was a real reading of the
    route, and it was the defect: a refusal taken there is a refusal the
    user has already paid the download for, and the plan review that
    admitted the same domain minutes earlier priced it from the shared
    admission instead, so the two could disagree.

    What is pinned now is the contract that replaced it: the admission is
    :func:`woof.core.preflight.admission_estimate`'s arguments and
    nothing else -- the experiment's own ``column_chunk`` and the
    machine's device profile -- it is taken ONCE, and it is taken before
    the catalog is built.  The retained interval count did not become
    wrong; it became a different question, and it stays with the run's
    memory LEDGER, which is priced after the fetch because that is when it
    can be.  WHAT THAT COSTS, measured: the boundary tables the count
    sizes are real device residency, and an admission that prices them at
    the library's default cadence rather than the case's own under-states
    them.  It under-states them identically on the review and at the door,
    which is the property this lane exists to hold; closing the remaining
    gap means giving :func:`admission_estimate` a cadence both surfaces
    derive the same way, and that is not this seam.
    """
    from datetime import timedelta
    from types import SimpleNamespace
    from woof import runtime
    from woof.ingest import case_store, preflight as ingest_preflight
    from test_runtime import _fixture_pair
    exp, data = _fixture_pair(tmp_path)
    exp = replace(exp, tiles=st.StreamingOptions(mode="auto"))
    events = []
    machine = ap.Machine(24 * GIB, 256 * GIB)
    monkeypatch.setattr(ap.Machine, "detect", lambda **kw: events.append("cold") or machine)
    catalog = SimpleNamespace(valid_times=tuple(exp.start_time + timedelta(hours=i)
                                               for i in range(48)), excluded_valid_times=())
    monkeypatch.setattr(ingest_preflight, "build_input_catalog",
                        lambda data: events.append("catalog") or catalog)
    monkeypatch.setattr(runtime, "forcing_snapshots", lambda *a: events.append("decode") or {})
    monkeypatch.setattr(runtime, "forcing_schedule", lambda *a: catalog.valid_times)
    marker = object()
    priced = []
    def price(value, **kwargs):
        assert value is exp
        # The [case_data] arm names no forcing source, so its tables
        # carry water vapour only.  It runs before any land cover exists,
        # so it hands the estimate no urban column reading (A176: only a
        # prepared door has one).
        assert kwargs == {"column_chunk": exp.column_chunk, "profile": None,
                          "boundary_species": (), "urban_columns": None}
        priced.append(kwargs)
        events.append("price")
        return marker
    monkeypatch.setattr(pf, "estimate_experiment", price)
    def choose(cfg, options, **kwargs):
        assert kwargs == {"machine": machine, "resident_estimate": marker}
        events.append("decide")
        return st.StreamingDecision(True, "test", 8, 8, 2, 16)
    monkeypatch.setattr(st, "decide", choose)
    monkeypatch.setattr(case_store, "initialization_resources", lambda options: {})
    class ReachedInitializer(Exception):
        pass
    def prepare(*args, **kwargs):
        events.append("prepare")
        assert kwargs["store_request"].backend == "auto"
        raise ReachedInitializer
    monkeypatch.setattr(runtime, "prepare_experiment_case", prepare)
    with pytest.raises(ReachedInitializer):
        runtime.run_experiment(exp, data, tmp_path / "out")
    assert events == ["cold", "price", "decide", "catalog", "decode", "prepare"]
    assert len(priced) == 1, "the route priced its admission more than once"


def test_all_streamed_tree_still_prices_retained_global_workspaces():
    # On 13.5 GiB the only all-streamed road tiles the child at 4.86x
    # redundancy, which auto reached through the retired fallback that
    # lifted the planner's limit.  The limit now binds auto, so this
    # fixture asks for that tiling by the explicit knob; what it pins is
    # the workspace pricing, not the choice.
    exp = _tree(child_n=256)
    exp = replace(exp, tiles=replace(exp.tiles, max_redundancy=False))
    estimate = _windows_estimate(exp)
    nodes = st._config_tree_nodes(exp.domains)
    empty = st._resident_subset_envelope(estimate, nodes, set())
    assert empty > estimate.scratch_arena_bytes + estimate.dycore_state_workspace_bytes
    assert empty == replace(estimate, domains=(), legacy_call_peak_by_domain=()).peak_envelope_bytes
    rows = {}
    result = st.decide_tree(nodes, exp.tiles,
        machine=ap.Machine(int(13.5 * GIB), 256 * GIB), decisions=rows,
        resident_estimate=estimate)
    assert all(row.stream for row in rows.values())
    assert result.configured_mixed_envelope_bytes >= empty
    assert result.configured_mixed_envelope_bytes <= 13 * GIB


def test_auto_honours_the_redundancy_limit_and_real_shortages_stay_errors():
    """The retired guard pinned auto's slow-tiling fallback; this pins its end.

    It used to require that auto, refused by the planner's redundancy
    limit, drop the limit and return the slower tiling with a warning.
    That fallback streamed a 206x204x49 domain in 1,190 tiles at 49.95x,
    237-547 s per step against 0.6-1.9 s resident (measured 2026-09-26),
    so auto now keeps the limit (tests/test_streamed_auto_tiling.py
    pins what it does instead) and only an explicit
    ``[tiles] max_redundancy = false`` accepts such a tiling.
    """
    cfg = _exp(48).root.run
    machine = ap.Machine(32 * GIB, 256 * GIB)
    with pytest.raises(ap.CannotPlan) as hard:
        ap.plan(cfg, machine, prefer_resident=False, max_redundancy=4.0)
    assert "redundancy" in hard.value.detail
    assert hard.value.detail["redundancy"] > ap.MAX_REDUNDANCY
    assert hard.value.detail["ntiles"] >= 1 and hard.value.detail["limit"] == 4.0
    auto = st.StreamingOptions(mode="auto")
    with pytest.raises(ap.CannotPlan) as refused:
        st.decide(cfg, auto, machine=machine, allow_resident=False)
    assert "redundancy" in refused.value.detail
    lifted = st.decide(cfg, replace(auto, max_redundancy=False),
                       machine=machine, allow_resident=False)
    assert lifted.stream and lifted.redundancy > ap.MAX_REDUNDANCY
    with pytest.raises(ap.CannotPlan) as shortage:
        st.decide(cfg, auto, machine=replace(machine, vram_bytes=1),
                  allow_resident=False)
    assert "redundancy" not in shortage.value.detail


def test_tree_public_dispatch_preserves_cold_machine_and_domain_options(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from woof import runtime
    from woof.core import model as model_module

    exp = _tree(64, 32)
    child = replace(exp.domains[1], tiles=st.StreamingOptions(mode="off"))
    exp = replace(exp, domains=(exp.root, child))
    machine = ap.Machine(24 * GIB, 256 * GIB)
    model = SimpleNamespace(_input_catalog=object())
    # run_experiment writes report.json from the summary since 384a7aa6c9a
    # (development branch), so the stub carries the summary
    # fields that report reads, as RealCaseRunSummary and ExperimentRunSummary do.
    data = object()
    result = SimpleNamespace(completed_seconds=float(exp.run_seconds), wrfout_paths=(),
                             nan_free=True, hydro_receipts=None)
    events = []
    monkeypatch.setattr(ap.Machine, "detect", lambda **kw: events.append("cold") or machine)
    def build(value, companion):
        assert value is exp and companion is data
        assert value.domains[1].tiles.mode == "off"
        events.append("build")
        return model
    monkeypatch.setattr(model_module, "build_experiment", build)
    monkeypatch.setattr(runtime, "resolved_tree_config_report", lambda *args: "test report")
    def execute(value, companion, outdir, built, **kwargs):
        assert value is exp and companion is data and built is model
        assert kwargs["planning_machine"] is machine
        assert value.tiles.mode == "auto" and value.domains[1].tiles.mode == "off"
        events.append("execute")
        return result
    monkeypatch.setattr(runtime, "_run_built_experiment", execute)
    assert runtime.run_experiment(exp, data, tmp_path / "tree") is result
    assert events == ["cold", "build", "execute"]
    import json
    report = json.loads((tmp_path / "tree" / "report.json").read_text())
    assert report["status"] == "PASS" and report["domain_count"] == 2
