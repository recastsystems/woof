"""The card memory a fit priced travels as a record, on every route New forecast drives.

New forecast showed how close a draft sits to its card ("needs 4.9 of 14.5 GiB") by reading a line the
engine printed.  Only ``woof check`` prints that line, and the wizard runs the check only when the run's
inputs are already on disk.  A GFS draft has none to wait for; an ERA5 draft's forcing is fetched when the
run starts, so its check is deferred, the line never came, and the page showed the card name alone for
every ERA5 draft, at 80 or 100 levels as at 49.

The wizard prices every draft before it writes the configuration, so that is where the figure is taken:
``woof run-plan --resolve`` answers with a ``memory`` record, in bytes, and the page reads the record.

A draft too big for its card was the same defect on the refusal side.  The page read how big it was from
the refusal's words, which say ``at X GiB peak envelope`` for a source whose preprocessing is priced (ERA5,
GFS) and ``needs X GiB peak envelope ... is NOT PRICED`` for one whose preprocessing is not (HRRR, RAP,
ECMWF open data).  Only the first was read, so a too-big HRRR draft said the engine could not fit the box
and to read the server's log.  The refusal now carries the same two figures as a document on the machine
channel, and the page reads the document.

These tests hold the records to the figures the wizard and the check print, on every route, and hold the
page's reader to the records.  No network and no card: a declared card is priced on the CPU.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from woof.gui.api import Api, memory_fit, region_polygon
from woof.gui.files import write_json
from woof.gui.jobs import Runner
from woof.gui.server import build_server, serve_in_thread
from woof.runplan import PLAN_SCHEMA, generate_intent_config, load_plan, resolve_plan
from test_gui_server import request

GIB = float(1 << 30)

# CUDA usable totals measured on physical 16 GB and 32 GB reference cards.
# The unmeasured 24 GB tier retains its declared total.
_CUDA_TOTAL_BYTES = {16: 16611278848, 32: 33711521792}


def _card_terms(gib):
    """Runtime probe terms with the same available memory a declared tier uses."""
    from woof.domain_wizard import card_assumed_free_gib

    return {"sm_count": 170, "max_threads_per_sm": 1536, "registers_per_sm": 65536,
            "warp_size": 32, "free_bytes": int(card_assumed_free_gib(gib) * GIB),
            "total_bytes": _CUDA_TOTAL_BYTES.get(gib, int(gib * GIB)), "device_name": f"{gib} GB card"}


@pytest.fixture
def isolated_mynn(monkeypatch):
    """Keep runtime choices and their published widths private to each fit regression."""
    import sys

    from woof.core import mynn_pbl_scratch as scratch

    monkeypatch.delenv(scratch.MYNN_PBL_COLUMN_CHUNK_ENV, raising=False)
    monkeypatch.setattr(scratch, "_RESOLVED", {})
    monkeypatch.setattr(scratch, "_PINNED", None)
    monkeypatch.setattr(scratch, "_TILE_WALKED", {})
    for name in scratch._CHUNK_MODULES:
        if name in sys.modules:
            module = sys.modules[name]
            monkeypatch.setattr(module, "MYNN_PBL_COLUMN_CHUNK", module.MYNN_PBL_COLUMN_CHUNK)
    return scratch


def test_capacity_tier_memory_rows_preserve_physical_totals_and_unmeasured_fallback(
        monkeypatch, isolated_mynn):
    from importlib.resources import files

    scratch = isolated_mynn
    monkeypatch.setattr(scratch, "probe_mynn_card", lambda device=None: pytest.fail("unexpected host probe"))
    table = json.loads((files("woof") / "authorities" / "mynn-card-memory.v1.json").read_text(encoding="utf-8"))
    assert table["schema"] == "gpuwm.mynn-card-memory.v1"
    rows = {row["capacity_gib"]: row["cuda_total_bytes"] for row in table["rows"]}
    assert {gib: rows[gib] for gib in _CUDA_TOTAL_BYTES} == _CUDA_TOTAL_BYTES
    assert 24 not in rows
    for gib, total in [*_CUDA_TOTAL_BYTES.items(), (24, 24 * 1024 ** 3)]:
        assert scratch.mynn_pricing_total_bytes(gib) == total
    assert scratch._RESOLVED == {}


@pytest.mark.parametrize("gib, measured", [(16, False), (32, False), (24, False),
                                         (16, True), (32, True)])
def test_sizing_and_emitted_check_use_raw_measurements_before_tier_references(
        monkeypatch, isolated_mynn, gib, measured):
    from woof import domain_wizard

    scratch = isolated_mynn
    monkeypatch.setattr(scratch, "probe_mynn_card", lambda device=None: pytest.fail("unexpected host probe"))
    free = _card_terms(gib)["free_bytes"]
    total = int(gib * GIB) if measured else _card_terms(gib)["total_bytes"]
    assert scratch.mynn_pricing_total_bytes(gib, measured=measured) == total
    card = {**_card_terms(gib), "total_bytes": total}
    expected = scratch.choose_mynn_column_chunk(49, card=card, environ={}).chunk
    monkeypatch.setattr(domain_wizard, "_sizing_phases_for_card",
                        lambda *args, **kwargs: scratch.resolve_mynn_column_chunk(49))
    monkeypatch.setattr(domain_wizard, "_check_emitted_config_for_card",
                        lambda *args, **kwargs: scratch.resolve_mynn_column_chunk(49))
    published = scratch.MYNN_PBL_COLUMN_CHUNK
    profile = object() if measured else None
    assert domain_wizard._sizing_phases(object(), free_bytes=free, vram_gib=gib, profile=profile) == expected
    sizing = domain_wizard.SizingBudget(gib, free, profile, None, measured=measured)
    assert domain_wizard._check_emitted_config(Path("unused.toml"), sizing) == expected
    if measured:
        declared = domain_wizard.SizingBudget(gib, free, None, None)
        assert domain_wizard._check_emitted_config(
            Path("unused.toml"), declared, remote_hardware=True) == expected
    assert scratch._RESOLVED == {} and scratch.MYNN_PBL_COLUMN_CHUNK == published


@pytest.mark.parametrize("nz", [49, 59])
@pytest.mark.parametrize("total_gib, free_gib, source", [
    (24, 22, "measured-vram-bounded"),
    (32, 0.125, "measured-minimum"),
    (128, 127, "measured"),
])
def test_target_context_uses_each_runtime_memory_policy_at_each_depth(
        monkeypatch, isolated_mynn, nz, total_gib, free_gib, source):
    scratch = isolated_mynn
    card = {**_card_terms(total_gib), "free_bytes": int(free_gib * GIB)}
    runtime = scratch.choose_mynn_column_chunk(nz, card=card, environ={})
    shared = scratch.mynn_column_chunk_for_memory(
        nz, total_bytes=card["total_bytes"], free_bytes=card["free_bytes"], environ={})
    assert (shared.chunk, shared.source, shared.workspace_bytes, shared.vram_ceiling) == (
        runtime.chunk, runtime.source, runtime.workspace_bytes, runtime.vram_ceiling)
    assert runtime.source == source
    # A host memo's old override provenance must not widen the target's tile buffers.
    stale = scratch.choose_mynn_column_chunk(
        nz, card=_card_terms(16), environ={scratch.MYNN_PBL_COLUMN_CHUNK_ENV: "32768"})
    scratch._RESOLVED[nz] = stale
    published = scratch.MYNN_PBL_COLUMN_CHUNK
    with scratch.mynn_pricing_memory(total_bytes=card["total_bytes"], free_bytes=card["free_bytes"]):
        assert scratch.resolve_mynn_column_chunk(nz) == runtime.chunk
        assert scratch.resolve_mynn_tile_column_chunk(nz) == min(
            runtime.chunk, scratch.MYNN_PBL_COLUMN_CHUNK_MINIMUM)
    assert scratch._RESOLVED == {nz: stale} and scratch.MYNN_PBL_COLUMN_CHUNK == published
    assert scratch._TILE_WALKED == {}


@pytest.mark.parametrize("control", ["environment", "pin"])
def test_target_context_honors_operator_width_for_resident_and_tile_pricing(
        monkeypatch, isolated_mynn, control):
    scratch = isolated_mynn
    stale = scratch.choose_mynn_column_chunk(49, card=_card_terms(16), environ={})
    scratch._RESOLVED[49] = stale
    if control == "environment":
        width = 32768
        monkeypatch.setenv(scratch.MYNN_PBL_COLUMN_CHUNK_ENV, str(width))
    else:
        width = 12288
        scratch.pin_mynn_column_chunk(width)
    memo = dict(scratch._RESOLVED)
    published = scratch.MYNN_PBL_COLUMN_CHUNK
    target = _card_terms(32)
    with scratch.mynn_pricing_memory(total_bytes=target["total_bytes"], free_bytes=target["free_bytes"]):
        for nz in (49, 59):
            assert scratch.resolve_mynn_column_chunk(nz) == width
            assert scratch.resolve_mynn_tile_column_chunk(nz) == width
    assert scratch._RESOLVED == memo and scratch.MYNN_PBL_COLUMN_CHUNK == published
    assert scratch._TILE_WALKED == {}


def test_nested_target_context_restores_outer_and_runtime_after_an_exception(monkeypatch, isolated_mynn):
    scratch = isolated_mynn
    host, outer, inner = (_card_terms(gib) for gib in (16, 32, 24))
    memo = {nz: scratch.choose_mynn_column_chunk(nz, card=host, environ={}) for nz in (49, 59)}
    scratch._RESOLVED.update(memo)
    published = scratch.MYNN_PBL_COLUMN_CHUNK
    monkeypatch.setattr(scratch, "probe_mynn_card", lambda device=None: pytest.fail("unexpected host probe"))
    outer_widths = [scratch.choose_mynn_column_chunk(nz, card=outer, environ={}).chunk for nz in (49, 59)]
    inner_widths = [scratch.choose_mynn_column_chunk(nz, card=inner, environ={}).chunk for nz in (49, 59)]
    with scratch.mynn_pricing_memory(total_bytes=outer["total_bytes"], free_bytes=outer["free_bytes"]):
        assert [scratch.resolve_mynn_column_chunk(nz) for nz in (49, 59)] == outer_widths
        with pytest.raises(RuntimeError, match="interrupted pricing"):
            with scratch.mynn_pricing_memory(total_bytes=inner["total_bytes"], free_bytes=inner["free_bytes"]):
                assert [scratch.resolve_mynn_column_chunk(nz) for nz in (49, 59)] == inner_widths
                raise RuntimeError("interrupted pricing")
        assert [scratch.resolve_mynn_column_chunk(nz) for nz in (49, 59)] == outer_widths
    assert [scratch.resolve_mynn_column_chunk(nz) for nz in (49, 59)] == [memo[nz].chunk for nz in (49, 59)]
    assert scratch._RESOLVED == memo and scratch.MYNN_PBL_COLUMN_CHUNK == published


def test_concurrent_target_contexts_do_not_share_widths_or_change_runtime(monkeypatch, isolated_mynn):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    scratch = isolated_mynn
    host = _card_terms(24)
    memo = {nz: scratch.choose_mynn_column_chunk(nz, card=host, environ={}) for nz in (49, 59)}
    scratch._RESOLVED.update(memo)
    published = scratch.MYNN_PBL_COLUMN_CHUNK
    monkeypatch.setattr(scratch, "probe_mynn_card", lambda device=None: pytest.fail("unexpected host probe"))
    barrier = Barrier(2, timeout=10)

    def price(gib):
        card = _card_terms(gib)
        expected = [scratch.choose_mynn_column_chunk(nz, card=card, environ={}).chunk for nz in (49, 59)]
        with scratch.mynn_pricing_memory(total_bytes=card["total_bytes"], free_bytes=card["free_bytes"]):
            barrier.wait()
            actual = [scratch.resolve_mynn_column_chunk(nz) for nz in (49, 59)]
            barrier.wait()
            assert actual == expected
        assert [scratch.resolve_mynn_column_chunk(nz) for nz in (49, 59)] == [memo[nz].chunk for nz in (49, 59)]
        return actual

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(price, gib) for gib in (16, 32)]
        left, right = [future.result() for future in futures]
    assert left != right
    assert scratch._RESOLVED == memo and scratch.MYNN_PBL_COLUMN_CHUNK == published
    assert scratch._TILE_WALKED == {}

#: The wizard's one-line verdict, printed on every route (woof/domain_wizard.py, sizing_summary).
_SIZING = re.compile(r"peak envelope ([\d.]+) GiB of a ([\d.]+) GiB budget")
#: The check's verdict, printed only where the check runs (woof/core/preflight.py, check_main).
_CHECKED = re.compile(r"BINDING PHASE: \S+ needs ([\d.]+) GiB; whole-process budget ([\d.]+) GiB")

#: The wizard's too-big sentence, in both of its wordings: the peak envelope comes first in each, the budget
#: after EXCEEDS (woof/core/preflight.py, the phase estimate's verdict).
_REFUSED = re.compile(r"([\d.]+) GiB peak envelope.*?EXCEEDS the ([\d.]+) GiB budget", re.S)

#: The ERA5 draft that showed no memory: 600 by 600 km at 3 km with 80 levels, sized for a 16 GB card.
_DRAFT = {"name": "", "lat": 35.3, "lon": -97.6, "width_km": 600.0, "height_km": 600.0, "hours": 6,
          "dx_km": 3.0, "nz": 80, "card": "16gb", "profile": None, "products": None}
#: A draft too big for an 8 GB card: 2000 by 2000 km at 1 km with 100 levels, over 340 GiB on every source.
_TOO_BIG = {"width_km": 2000.0, "height_km": 2000.0, "dx_km": 1.0, "nz": 100, "card": "8gb"}


def _gui_plan(tmp_path: Path, *, source: str, route: str, cycle: str, **changes) -> Path:
    """The plan New forecast writes for a fit: its own plan document and the box it draws."""

    draft = {**_DRAFT, **changes, "source": source, "route": route, "cycle": cycle}
    folder = tmp_path / source
    folder.mkdir()
    write_json(folder / "region.geojson",
               region_polygon(draft["lat"], draft["lon"], draft["width_km"], draft["height_km"]))
    path = folder / "plan.json"
    write_json(path, Api.plan_document(draft, folder))
    return path


def _gib(value: int) -> float:
    return round(value / GIB, 2)


def test_an_era5_draft_is_priced_as_a_record_though_its_check_waits_for_the_download(tmp_path, capsys):
    resolution, _exp, data = resolve_plan(
        load_plan(_gui_plan(tmp_path, source="era5", route="experiment", cycle="1999-05-03T18")),
        require_inputs=False)
    printed = capsys.readouterr().out
    # The route that broke: it reads forcing files the run fetches when it starts, so the wizard defers the
    # check until they are on disk and no check line was printed to read.
    assert data is not None and data.forcing
    sizing = _SIZING.search(printed)
    assert sizing, printed[-2000:]

    record = resolution["memory"]
    assert record is not None, "the ERA5 fit carried no memory record"
    assert (_gib(record["peak_envelope_bytes"]), _gib(record["budget_bytes"])) == (
        float(sizing.group(1)), float(sizing.group(2)))
    assert record["sizing_basis"] == "declared-capacity" and record["vram_gib"] == 16
    assert 0 < record["peak_envelope_bytes"] <= record["budget_bytes"]

    # What Review and the levels step show, read from the record.
    memory = memory_fit(resolution)
    assert memory == {"need_gib": float(sizing.group(1)), "budget_gib": float(sizing.group(2)), "fits": True}
    assert memory["budget_gib"] == 14.54


def test_a_gfs_draft_record_is_the_figure_its_check_prints(tmp_path, capsys):
    resolution, _exp, data = resolve_plan(
        load_plan(_gui_plan(tmp_path, source="gfs", route="prepared", cycle="2024-05-03T12")),
        require_inputs=False)
    printed = capsys.readouterr().out
    assert data is None
    checked = _CHECKED.search(printed)
    assert checked, printed[-2000:]

    # The figure the page read off this line before is the figure the record now carries.
    record = resolution["memory"]
    assert (_gib(record["peak_envelope_bytes"]), _gib(record["budget_bytes"])) == (
        float(checked.group(1)), float(checked.group(2)))
    assert memory_fit(resolution) == {"need_gib": float(checked.group(1)),
                                      "budget_gib": float(checked.group(2)), "fits": True}


def test_a_hrrr_fit_record_prices_the_boundary_tables_its_phases_price(tmp_path, monkeypatch):
    """The record's allocation estimate is priced on the tables its envelope is.

    HRRR publishes five hydrometeor masses on every frame and the root's
    boundary carries them.  The fit's phase envelope priced those tables;
    the itemized estimate the record's ``alloc_estimate_bytes`` (and the
    printed sizing table) come from did not, so one record carried two
    forecasts.  Red before the A92 follow-up.
    """
    from woof import domain_wizard
    from woof.boundary_fields import boundary_hydrometeor_fields, source_boundary_species

    seen = []
    fit_memory = domain_wizard.fit_memory

    def spy(estimate, phases, budget_bytes, sizing):
        seen.append((estimate, phases))
        return fit_memory(estimate, phases, budget_bytes, sizing)

    monkeypatch.setattr(domain_wizard, "fit_memory", spy)
    resolution, exp, _data = resolve_plan(
        load_plan(_gui_plan(tmp_path, source="hrrr", route="prepared", cycle="2026-09-24T12")),
        require_inputs=False)
    # The emitted root carries the tables: its scheme holds masses HRRR publishes.
    assert boundary_hydrometeor_fields(exp.root.run, source_boundary_species("hrrr"))
    ((estimate, phases),) = seen
    forecast = phases.forecast
    assert estimate.domains[0].category_bytes("lbc") == forecast.domains[0].category_bytes("lbc")
    assert estimate.alloc_estimate_bytes == forecast.alloc_estimate_bytes
    assert estimate.peak_envelope_bytes == forecast.peak_envelope_bytes
    assert resolution["memory"]["alloc_estimate_bytes"] == forecast.alloc_estimate_bytes


@pytest.mark.parametrize("target_gib, expected_width", [(16, 16384), (24, 28672), (32, 36864)])
@pytest.mark.parametrize("host_gib", [None, 16, 32], ids=["no-host-card", "16gb-host", "32gb-host"])
@pytest.mark.parametrize("stale_memo", [False, True], ids=["fresh-runtime", "cached-runtime"])
def test_named_card_fit_prices_the_runtime_width_without_changing_the_hosts_memo(
        tmp_path, monkeypatch, isolated_mynn, target_gib, expected_width, host_gib, stale_memo):
    """Every named tier prices its own scratch even on an absent or differently sized host."""
    from woof import domain_wizard
    from woof.core import preflight as pf
    from woof.physics_compat import THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID

    scratch = isolated_mynn
    host = None if host_gib is None else _card_terms(host_gib)
    probes = []

    def host_probe(device=None):
        probes.append(device)
        return host

    monkeypatch.setattr(scratch, "probe_mynn_card", host_probe)
    if stale_memo:
        scratch._RESOLVED[49] = scratch.choose_mynn_column_chunk(49, card=host, environ={})
    probes.clear()
    memo = dict(scratch._RESOLVED)
    published = scratch.MYNN_PBL_COLUMN_CHUNK
    captured = []
    estimate_phases = domain_wizard.estimate_phases

    def priced(exp, **kwargs):
        widths = [pf.mynn_pbl_column_chunk(dc.run) for dc in exp.domains]
        phases = estimate_phases(exp, **kwargs)
        captured.append((kwargs, phases, widths))
        return phases

    monkeypatch.setattr(domain_wizard, "estimate_phases", priced)
    resolution, exp, _data = resolve_plan(load_plan(_gui_plan(
        tmp_path, source="gfs", route="prepared", cycle="2024-05-03T12", nz=49,
        width_km=300, height_km=300, dx_km=0.75, card=f"{target_gib}gb",
        profile=THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID)), require_inputs=False)
    assert captured
    assert all(widths == [expected_width] for _kwargs, _phases, widths in captured)
    assert probes == [], "declared-card pricing consulted the host card"
    assert scratch._RESOLVED == memo and scratch.MYNN_PBL_COLUMN_CHUNK == published
    assert scratch._TILE_WALKED == {}

    # Run-time pricing follows the real engine resolver with a target-card probe.
    # The separate real-card receipt checks the same path without this emulated probe.
    target = _card_terms(target_gib)
    monkeypatch.setattr(scratch, "probe_mynn_card", lambda device=None: target)
    monkeypatch.setattr(scratch, "_RESOLVED", {})
    kwargs, fitted, _widths = captured[-1]
    runtime = pf.estimate_phases(exp, **kwargs)
    assert scratch._RESOLVED[49].chunk == expected_width
    assert runtime.forecast.alloc_estimate_bytes == fitted.forecast.alloc_estimate_bytes
    assert runtime.peak_envelope_bytes == fitted.peak_envelope_bytes
    assert resolution["memory"]["peak_envelope_bytes"] == runtime.peak_envelope_bytes
    assert resolution["memory"]["alloc_estimate_bytes"] == runtime.forecast.alloc_estimate_bytes


@pytest.mark.parametrize("target_gib", [16, 24, 32])
@pytest.mark.parametrize("host_gib", [None, 16, 32], ids=["no-host-card", "16gb-host", "32gb-host"])
@pytest.mark.parametrize("headroom_bytes", [1, -1], ids=["fits-by-one-byte", "too-big-by-one-byte"])
def test_named_card_polygon_fit_acceptance_matches_the_runtime_budget(
        tmp_path, monkeypatch, capsys, isolated_mynn, target_gib, host_gib, headroom_bytes):
    """A fixed footprint is accepted and refused on either side of the runtime envelope."""
    from woof import domain_wizard
    from woof.cli import main
    from woof.core import preflight as pf
    from woof.physics_compat import THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID

    scratch = isolated_mynn
    target = _card_terms(target_gib)
    monkeypatch.setattr(scratch, "probe_mynn_card", lambda device=None: target)
    draft = dict(source="gfs", route="prepared", cycle="2024-05-03T12", nz=49,
                 width_km=300, height_km=300, dx_km=0.75, card=f"{target_gib}gb",
                 profile=THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID)
    seed = tmp_path / "seed"
    seed.mkdir()
    _resolution, exp, _data = resolve_plan(load_plan(_gui_plan(seed, **draft)), require_inputs=False)
    capsys.readouterr()
    interval = domain_wizard.source_forcing_interval_seconds("gfs")
    runtime = pf.estimate_phases(exp, source="gfs", forcing_interval_seconds=interval,
                                 ingest_forcing_interval_seconds=interval, vram_gib=target_gib)
    envelope = runtime.peak_envelope_bytes
    width = scratch._RESOLVED[49].chunk
    free_bytes = envelope + pf.EXTERNAL_MARGIN_BYTES + headroom_bytes
    assert 0 < free_bytes < target["total_bytes"]
    limited = {**target, "free_bytes": free_bytes}
    # This proof changes the whole-process budget, not the scratch-width regime.
    assert scratch.choose_mynn_column_chunk(49, card=limited, environ={}).chunk == width

    host = None if host_gib is None else _card_terms(host_gib)
    monkeypatch.setattr(scratch, "probe_mynn_card", lambda device=None: host)
    monkeypatch.setattr(scratch, "_RESOLVED", {
        49: scratch.choose_mynn_column_chunk(49, card=host, environ={})})
    memo = dict(scratch._RESOLVED)
    published = scratch.MYNN_PBL_COLUMN_CHUNK
    sizing = domain_wizard.SizingBudget(target_gib, free_bytes, None, None)
    monkeypatch.setattr(domain_wizard, "resolve_sizing_budget", lambda *args, **kwargs: sizing)
    path = _gui_plan(tmp_path, **draft)
    rc = main(["run-plan", str(path), "--resolve"])
    out, err = capsys.readouterr()
    document = json.loads(out)
    record = document["memory"]
    assert record["peak_envelope_bytes"] == envelope
    assert record["budget_bytes"] == envelope + headroom_bytes
    assert scratch._RESOLVED == memo and scratch.MYNN_PBL_COLUMN_CHUNK == published
    if headroom_bytes > 0:
        assert rc == 0 and memory_fit(document)["fits"] is True, err
    else:
        assert rc == 2 and memory_fit(document)["fits"] is False
        assert document["kind"] == "memory" and "EXCEEDS" in document["error"]


def test_resolve_prints_the_record_on_its_machine_channel(tmp_path, capsys):
    from woof.cli import build_parser
    from woof.runplan import run_plan_main

    plan = _gui_plan(tmp_path, source="era5", route="experiment", cycle="1999-05-03T18")
    assert run_plan_main(build_parser().parse_args(["run-plan", str(plan), "--resolve"])) == 0
    document = json.loads(capsys.readouterr().out)
    assert set(document["memory"]) >= {"peak_envelope_bytes", "budget_bytes", "binding_phase", "sizing_basis"}
    assert memory_fit(document)["fits"] is True


def test_a_plan_naming_its_own_config_has_no_fitted_memory(tmp_path):
    era5 = load_plan(_gui_plan(tmp_path, source="era5", route="experiment", cycle="1999-05-03T18"))
    config, _ = generate_intent_config(era5, destination=tmp_path / "written")
    path = tmp_path / "own-config.json"
    path.write_text(json.dumps({"schema": PLAN_SCHEMA, "name": "own", "route": "experiment",
                                "config": {"path": str(config)}, "output_root": str(tmp_path / "run")}),
                    encoding="utf-8")
    resolution, _exp, _data = resolve_plan(load_plan(path), require_inputs=False)
    assert resolution["memory"] is None
    assert memory_fit(resolution) is None


@pytest.mark.parametrize("document", [{}, {"memory": None}, {"memory": {"budget_bytes": 1}},
                                      {"memory": {"peak_envelope_bytes": 1, "budget_bytes": 0}}])
def test_a_document_without_a_usable_record_shows_the_grid_alone(document):
    assert memory_fit(document) is None


def test_the_record_is_read_in_bytes_and_shown_in_gib():
    record = {"peak_envelope_bytes": int(5.92 * GIB), "budget_bytes": int(14.54 * GIB)}
    assert memory_fit({"memory": record}) == {"need_gib": 5.92, "budget_gib": 14.54, "fits": True}
    over = {"peak_envelope_bytes": int(40.97 * GIB), "budget_bytes": int(14.54 * GIB)}
    assert memory_fit({"memory": over})["fits"] is False


@pytest.mark.parametrize("source, route, cycle, words", [
    # Preprocessing not priced for the source: the wording the page could not read.
    ("hrrr", "prepared", "2026-09-24T12", "is NOT PRICED here"),
    ("rap", "prepared", "2026-09-24T12", "is NOT PRICED here"),
    ("ecmwf-open-data", "prepared", "2026-09-24T12", "is NOT PRICED here"),
    # Preprocessing priced: the wording it could.
    ("era5", "experiment", "1999-05-03T18", "is the memory-binding phase at"),
])
def test_a_draft_too_big_for_its_card_is_refused_with_its_figures(tmp_path, capsys, source, route, cycle, words):
    from woof.cli import main

    plan = _gui_plan(tmp_path, source=source, route=route, cycle=cycle, **_TOO_BIG)
    assert main(["run-plan", str(plan), "--resolve"]) == 2
    out, err = capsys.readouterr()
    # The sentence is still the refusal a person reads, on stderr.
    assert "run plan 'config.intent' does not fit" in err and words in err

    document = json.loads(out)
    assert (document["schema"], document["kind"], document["created"]) == (
        "arwen.configuration-error.v1", "memory", False)
    # The whole refusal, both layers, without the marker between them.
    assert words in document["error"] and "[[explain]]" not in document["error"]
    sentence = _REFUSED.search(document["error"])
    record = document["memory"]
    assert record["binding_phase"] == "forecast"
    assert (_gib(record["peak_envelope_bytes"]), _gib(record["budget_bytes"])) == (
        float(sentence.group(1)), float(sentence.group(2)))

    memory = memory_fit(document)
    assert memory == {"need_gib": float(sentence.group(1)), "budget_gib": 6.75, "fits": False}
    assert memory["need_gib"] > 340


def test_estimate_refuses_a_too_big_draft_with_the_same_figures(tmp_path, capsys):
    from woof.cli import main

    # The price of a plan resolves it first, so a draft too big for its card is refused there, and the
    # figures reach the machine channel of --estimate as they reach that of --resolve.
    plan = _gui_plan(tmp_path, source="hrrr", route="prepared", cycle="2026-09-24T12", **_TOO_BIG)
    assert main(["run-plan", str(plan), "--estimate"]) == 2
    out, err = capsys.readouterr()
    assert "run plan 'config.intent' does not fit" in err and "is NOT PRICED here" in err

    document = json.loads(out)
    assert (document["schema"], document["kind"]) == ("arwen.configuration-error.v1", "memory")
    sentence = _REFUSED.search(document["error"])
    memory = memory_fit(document)
    assert memory == {"need_gib": float(sentence.group(1)), "budget_gib": 6.75, "fits": False}
    assert memory["need_gib"] > 340


def test_a_refusal_that_is_not_the_cards_prints_no_document(tmp_path, capsys):
    from woof.cli import main

    # Too few levels is refused before any layout is priced.
    plan = _gui_plan(tmp_path, source="era5", route="experiment", cycle="1999-05-03T18", nz=2)
    assert main(["run-plan", str(plan), "--resolve"]) == 2
    out, err = capsys.readouterr()
    assert out == "" and "--nz must be at least 4" in err


def test_new_forecast_says_how_big_a_too_big_hrrr_draft_is(tmp_path, monkeypatch):
    """The page server and the engine it runs, as New forecast asks them: no fake runner."""

    # The start's availability is asked in the page server's own process; no server is asked in a test.
    monkeypatch.setattr("woof.fetch._head_ok", lambda url: True)
    monkeypatch.setattr("woof.source_availability.quick_head", lambda url: True)
    server = build_server(tmp_path / "runs", port=0, runner=Runner(tmp_path / "jobs"), token="t" * 43)
    serve_in_thread(server)
    try:
        response, body = request(server, "POST", "/api/create/fit", timeout=600, body={
            "source": "hrrr", "cycle": "2026-09-24T12", "lat": 35.3, "lon": -97.6, "hours": 6,
            "width_km": 2000, "height_km": 2000, "dx_km": 1, "nz": 100, "card": "8gb"})
    finally:
        server.shutdown()
        server.server_close()
    assert response.status == 422, body
    memory = body["memory"]
    # 348 GiB when pinned (340-360).  A first-parent bisect to 2.8.8 reads
    # 416.61 GiB after HRRR-suite pricing moves, the large ones bf8a9a71d
    # (the HRRR default suite became MYNN PBL and surface layer with RUC,
    # 346 to 398), 0542178eb (the GSD MYNN generation, to 408.7),
    # a64145961 (the sixth-order filter's workspace priced, to 414.7) and
    # cd2d67020 (to 416.6).  development branch gave the one-card
    # road the A13 pool residue bound this HRRR suite carries on the
    # ranked road (2.25 GiB in place of the ratio's 13 per cent of the
    # subtotal), to 377.2.
    assert memory["fits"] is False and memory["budget_gib"] == 6.75 and 365 < memory["need_gib"] < 390, memory
    assert body["message"] == (f"Too big for the 8 GB card: this grid needs about {memory['need_gib']:.1f} GiB "
                               "and 6.8 GiB is usable."), body


@pytest.mark.parametrize("failure,status", [
    ("timeout", 502), ("missing", 404), ("path", 400),
])
def test_fit_page_preserves_runner_failure_contract(tmp_path, monkeypatch, failure, status):
    """Exercise the farm failures directly instead of relying on worker timing."""
    import subprocess
    from woof.gui.api import PathRefused

    runner = Runner(tmp_path / "jobs")
    api = Api(tmp_path / "runs", runner=runner, token="t" * 43, port=0,
              version="test", bind="127.0.0.1")

    def fail(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], timeout=10)
        if failure == "missing":
            raise FileNotFoundError(2, "missing engine", "engine-python")
        raise PathRefused("invalid path")

    monkeypatch.setattr("gpuwm.gui.jobs.subprocess.run", fail)
    monkeypatch.setattr(api, "_get", lambda *_args: api._cached("failure", ["engine-python"], 1))
    try:
        reply = api.handle("GET", "/api/failure", {}, b"")
        assert reply.status == status and reply.body["ok"] is False
    finally:
        api.close()
