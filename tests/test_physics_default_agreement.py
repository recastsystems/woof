"""One default row, read by every door, and a run that never refuses the suite a door chose.

The default a run takes with no suite named is one table row
(:data:`woof.physics_menu.SPACING_DEFAULTS`, read through
:func:`woof.physics_menu.default_profile_for` at the run's finest grid and
domain count).  Five doors answer "which suite does this draft run": the
physics check (New forecast's Physics step and ``woof physics-catalog
--check``), New forecast's own draft reading, the assistant's planner,
``woof domain`` (the file it writes) and the run (the suite the chain's
stages are told the file is).  They disagreed twice on this line: the
check read the source's own default below 1 km, and the planner named the
source's own default on a sub-km ladder, which the run asserted on every
domain and the domain-tree forecast refused after the download and the
preparation (``cu_physics selected 0 expected 1`` on d02).

The sweep below asks every door for every source the wizard plans, on
grids below and above 1 km, with one domain and with nests, and fails on
any disagreement.  For each suite a door can choose there (none named, the
source's own default, the sub-km row's suite) it writes the file the run
would write and holds the run to it: a suite the run asserts must pass the
domain-tree forecast's own check on every domain, and the root must carry
the suite chosen.  The local DA door (``woof local-da``) is swept the same
way on its 3 km and 750 m rungs, for every source it plans unaided.
"""

from __future__ import annotations

from contextlib import ExitStack
from functools import lru_cache
import json
from pathlib import Path
import re

import pytest

from woof.physics_compat import (
    THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
    single_domain_runtime_switches,
)

#: The grids New forecast sends, as draft keys: one grid below and above
#: 1 km, a chain under a custom root below and above 1 km, and the named
#: ladders the assistant chooses from, the deepest reaching 500 m.
GRIDS = {
    "750 m": {"dx_km": 0.75},
    "3 km": {"dx_km": 3.0},
    "2.25 km with a 750 m nest": {"dx_km": 2.25, "chain": "3"},
    "9 km with a 3 km nest": {"dx_km": 9.0, "chain": "3"},
    "ladder 12-3": {"ladder": "12-3"},
    "ladder 12-3-1-0.5": {"ladder": "12-3-1-0.5"},
}

#: The switches that make a suite what it is, compared on the written root.
#: Cumulus is left out: a door may leave the root's cumulus to the grid.
SUITE_KEYS = ("mp_physics", "bl_pbl_physics", "sf_sfclay_physics", "sf_surface_physics")

CYCLE = "2026-09-20T12"

#: A box every ladder above fits on a 24 GB card, down to its 500 m nest.
BOX_KM = 150.0


@lru_cache(maxsize=None)
def _menus():
    """The source rows the assistant reads, built as the GUI's /api/sources builds them."""

    from woof.runplan import physics_profile_menu, source_inventory

    menu = {row["source_id"]: row for row in physics_profile_menu()["sources"]}
    rows = {}
    for row in source_inventory()["sources"]:
        plan = row.get("run_plan") or {}
        routes = plan.get("intent_routes") or []
        if not plan.get("intent_supported") or plan.get("requires_source_root") or not routes:
            continue
        own = menu.get(row["source_id"]) or {}
        rows[row["source_id"]] = {
            "id": row["source_id"], "route": "prepared" if "prepared" in routes else routes[0],
            "default_profile": own.get("default_profile_id"),
            "spacing_defaults": own.get("spacing_defaults") or [],
            "profiles": [item["profile_id"] for item in own.get("profiles") or [] if item.get("admissible")]}
    return rows


def _page_sources():
    from woof.source_adapters import wizard_planable_source_ids

    return [source for source in wizard_planable_source_ids() if source in _menus()]


def _served(source, when):
    """``when``, or the same hour two days before the end of the source's last
    publication era when ``when`` lies outside every era.

    A dated publication row (the GEFS v11 archive layouts,
    development branch) serves only its own era, so a recent epoch asks it
    for objects it never published.  Read from the route table, so the
    next dated row needs no edit here.
    """

    from datetime import timedelta
    from woof import fetch_routes

    if source not in fetch_routes.route_ids():
        return when
    route = fetch_routes.route_for(source)
    # A producer that runs fewer cycles (the reforecast's 00Z only) is read
    # at its latest cycle at or before the hour.
    hours = [hour for hour in route.cycle_hours if hour <= when.hour]
    when = when.replace(hour=max(hours) if hours else min(route.cycle_hours))
    if not route.publication_eras:
        return when
    naive = when.replace(tzinfo=None)
    if any(era.valid_from <= naive and (era.valid_until is None or naive < era.valid_until)
           for era in route.publication_eras):
        return when
    era = max(route.publication_eras, key=lambda era: era.valid_from)
    if era.valid_until is None:
        # An open era that began after the fixture's epoch (ECMWF's 0.1 degree
        # open data, published from 2026-10-20) is read just inside its start.
        moved = (era.valid_from + timedelta(days=2)).replace(hour=when.hour)
    else:
        moved = (era.valid_until - timedelta(days=2)).replace(hour=when.hour)
    return moved.replace(tzinfo=when.tzinfo)


def _first_lead(source):
    """The first lead the source's route publishes (3 for the GEFSv12
    reforecast, which has no f000), where a draft's window opens."""

    from woof import fetch_routes

    return (fetch_routes.route_for(source).first_lead
            if source in fetch_routes.route_ids() else 0)


def _draft(source, grid, profile=None):
    """New forecast's draft of a BOX_KM box inside the source's coverage."""

    from datetime import datetime, timedelta
    from woof import cf_archive_fetch

    lat, lon = _point(source)
    cycle = _served(source, datetime.fromisoformat(CYCLE)).strftime("%Y-%m-%dT%H")
    if source in cf_archive_fetch.sources():
        # A closed analysis archive needs a representative epoch inside
        # its declared coverage, including this fixture's final boundary.
        bounds = cf_archive_fetch.row(source)
        first = datetime.fromisoformat(bounds["coverage_start"])
        last = datetime.fromisoformat(bounds["coverage_end"])
        anchor = max(first, min(datetime.fromisoformat(CYCLE), last - timedelta(hours=3)))
        cf_archive_fetch.validate_window(source, anchor, 3)
        cycle = anchor.strftime("%Y-%m-%dT%H")
    return {"name": "agreement", "source": source, "cycle": cycle, "hours": 3, "card": "24gb",
            "lat": lat, "lon": lon, "width_km": BOX_KM, "height_km": BOX_KM,
            "start_hour": _first_lead(source), "products": None, "profile": profile,
            "dx_km": grid.get("dx_km"), "ladder": grid.get("ladder"), "chain": grid.get("chain")}


def _point(source):
    from woof.source_adapters import source_coverage_window
    from woof.source_coverage import window_centre

    return window_centre(source_coverage_window(source)) or (37.62, -122.2)


def _written(tmp_path, draft, tag):
    """The file the run writes from New forecast's draft: the page's plan, resolved by `woof domain`."""

    from woof.gui.api import CreateMixin, region_polygon
    from woof.runplan import build_plan, generate_intent_config

    rundir = tmp_path / tag
    rundir.mkdir()
    polygon = region_polygon(draft["lat"], draft["lon"], draft["width_km"], draft["height_km"])
    (rundir / "region.geojson").write_text(json.dumps(polygon), encoding="utf-8")
    document = CreateMixin.plan_document({**draft, "route": _menus()[draft["source"]]["route"]}, rundir)
    plan = build_plan(document, source="agreement", base_dir=rundir, sha256="0" * 64)
    config, _ = generate_intent_config(plan, destination=rundir / "generated")
    return plan, Path(config)


def _held_to_the_run(plan, config: Path, source: str):
    """What the run asserts on this file, having checked the stages that receive the assertion accept it."""

    from woof.experiment import load_experiment
    from woof.prepared_domain_tree_forecast import validate_physics_profile
    from woof.prepared_single_domain_forecast import named_profile_config_conflicts
    from woof.runplan import _asserted_profile, _canonical_source_id

    asserted = _asserted_profile(plan, config_path=config)
    if asserted is not None:
        canonical = _canonical_source_id(source)
        text = config.read_text(encoding="utf-8")
        # The preparer's materialization and the forecast's check (every domain) both accept it.
        assert named_profile_config_conflicts(text, source=canonical, profile=asserted) == []
        validate_physics_profile(load_experiment(config), source=canonical, profile=asserted)
    return asserted


def _named_licence_opt_in(source: str) -> tuple[str, ...]:
    """The licence(s) a licence-gated row's own refusal names, else none.

    Rows whose weights are not permissive refuse every default path and
    name the exact ``--accept-licence`` value that opts in.  A sweep over
    every planable row opts in with exactly that value, as a user would;
    the refusal itself is held by the licensing tests.
    """

    import shlex

    from woof.earth2_licensing import check_argv

    try:
        check_argv(["domain", "--source", source])
    except ValueError as refusal:
        match = re.search(r"--accept-licence (.+)$", str(refusal))
        assert match, refusal
        return tuple(shlex.split(match.group(1)))
    return ()


@pytest.fixture
def licence_opt_in():
    """Opt a test into a row's named licence for the test's duration."""

    from woof.earth2_licensing import licence_scope

    with ExitStack() as stack:
        yield lambda source: stack.enter_context(
            licence_scope(_named_licence_opt_in(source)))


def _suite(profile):
    switches = single_domain_runtime_switches(profile)
    return tuple(int(switches[key]) for key in SUITE_KEYS)


@pytest.mark.parametrize("source", _page_sources())
def test_every_door_names_one_default_and_the_run_takes_every_suite_a_door_chooses(
        tmp_path, source, licence_opt_in):
    licence_opt_in(source)
    from woof import physics_catalog as pc
    from woof.experiment import load_experiment
    from woof.gui.api import PhysicsMixin, draft_default_suite, draft_domains, draft_finest_dx_m
    from woof.gui.assistant.plan import grid_default_profile, ladder_default_profile
    from woof.physics_menu import default_profile_for
    from types import SimpleNamespace

    row = _menus()[source]
    below = []
    for label, grid in GRIDS.items():
        where = f"{source} on {label}"
        draft = _draft(source, grid)
        finest_m, domains = draft_finest_dx_m(draft), draft_domains(draft)
        table = default_profile_for(source, finest_m, domains)

        # The Physics step's check, sent the way the page sends it.
        reply = PhysicsMixin.physics_check(SimpleNamespace(), {**draft, "dx_km": grid.get("dx_km")}, True)
        request = json.loads(reply.body["argv"][reply.body["argv"].index("--check") + 1])
        verdict = pc.check(request)
        # The assistant's planner, on the source row it reads.
        planner = grid_default_profile(row, finest_m, domains)
        answers = {"check": verdict["default_suite"], "new forecast": draft_default_suite(draft),
                   "assistant": planner}
        if grid.get("ladder"):
            answers["assistant's ladder"] = ladder_default_profile(row, grid["ladder"])
        # The file `woof domain` writes from the draft with no suite named.
        plan, config = _written(tmp_path, draft, f"{label}-unnamed".replace(" ", "_"))
        text = config.read_text(encoding="utf-8")
        # Its header names the suite it bound: "# PHYSICS: <id>: <what it is>".
        answers["woof domain"] = next(line.split(":")[1].strip() for line in text.splitlines()
                                       if line.startswith("# PHYSICS:"))
        assert set(answers.values()) == {table}, (where, answers)
        assert pc.experiment_grid(text)["finest_dx_km"] == pytest.approx(finest_m / 1000.0), where
        assert pc.experiment_grid(text)["domains"] == domains, where
        exp = load_experiment(config)
        assert _suite(table) == tuple(int(getattr(exp.root.run, key)) for key in SUITE_KEYS), where
        # The check reads the root the file runs: its spacing, and so its cumulus.  A ladder's root was read at
        # the check's 3 km probe spacing, showing cumulus off where the 12 km root runs Kain-Fritsch.
        assert verdict["dx_km"] == pytest.approx(pc.experiment_grid(text)["dx_km"]), where
        assert (verdict["resolved"]["cumulus"] == "off") == (int(exp.root.run.cu_physics) == 0),             (where, verdict["resolved"]["cumulus"], exp.root.run.cu_physics)
        assert _held_to_the_run(plan, config, source) in (None, table), where
        below.append((table == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID, finest_m < 1000.0))

        # Every suite a door chooses here: the source's own default and the sub-km row's, as the
        # assistant sends them (unnamed when it is this grid's default, named otherwise).
        for chosen in {row["default_profile"], THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID} - {table}:
            if chosen not in row["profiles"]:
                continue
            sent = None if chosen == planner else chosen
            plan, config = _written(tmp_path, _draft(source, grid, sent),
                                    f"{label}-{chosen}".replace(" ", "_"))
            exp = load_experiment(config)
            assert _suite(chosen) == tuple(int(getattr(exp.root.run, key)) for key in SUITE_KEYS), \
                (where, chosen)
            held = _held_to_the_run(plan, config, source)
            # One grid written from a named suite is that suite, so the run still holds it to it.
            assert held == (chosen if domains == 1 and sent else held) and held in (None, chosen),                 (where, chosen, held)
    if row["default_profile"] == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID:
        # A source whose own default is this suite takes it on every grid.
        assert all(sub_km for sub_km, _finer in below), (source, below)
    elif THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID in row["profiles"]:
        # The spacing override binds below 1 km when the source default differs.
        assert all(sub_km == finer for sub_km, finer in below), (source, below)


#: Local DA rungs (woof.local_da.derive_rung): rung 1 runs 3 km and rung 10 is the first at 750 m.
LOCAL_DA_SCALES = (1, 10)


def _local_da_sources():
    """The sources the local DA door plans with no inputs supplied, from the catalogue its review reads."""

    from woof.background_contract import catalog

    return [row["source"] for row in catalog()["sources"]
            if "automatic" in (row.get("initialization_modes") or [])]


@pytest.mark.parametrize("source", _local_da_sources())
def test_the_local_da_door_binds_the_default_row_at_its_rungs_grid(source):
    """A local DA rung reads the one default row at its own spacing, as the physics check does.

    Its configuration read the row with no grid, so a 750 m rung bound the source's own default (YSU and
    Noah) where every other door binds the sub-km row (Thompson, MYNN and RUC), and every cycle restarted
    through a four-layer soil the sub-km suite does not run.
    """

    from datetime import datetime, timedelta, timezone

    from woof import physics_catalog as pc
    from woof.config import validated_soil_layer_count
    from woof.local_da import Card, Request, configuration, derive_rung
    from woof.physics_menu import default_profile_for

    # A settled cycle every catalogued source still serves: two days back, at 12Z.
    epoch = (datetime.now(timezone.utc) - timedelta(days=2)).replace(hour=12, minute=0, second=0, microsecond=0)
    epoch = _served(source, epoch)
    below = []
    for scale in LOCAL_DA_SCALES:
        where = f"{source} at local DA rung {scale}"
        request = Request(epoch=epoch.strftime("%Y-%m-%dT%H:%M:%SZ"), point=tuple(_point(source)),
                          card=Card(vram_gib=24.), source=source, scale=scale)
        rung = derive_rung(request, scale)
        text, _wps, exp, _background = configuration(request, rung, background_probe=lambda url: True)
        grid = pc.experiment_grid(text)
        assert grid["finest_dx_km"] == pytest.approx(rung["dx_m"] / 1000.0) and grid["domains"] == 1, where
        table = default_profile_for(source, rung["dx_m"], grid["domains"])
        assert pc.check({"source": source, **grid})["default_suite"] == table, where
        root = exp.root.run
        assert _suite(table) == tuple(int(getattr(root, key)) for key in SUITE_KEYS), (where, table)
        # The soil every cycle restarts through is the bound land surface's own.
        assert int(root.num_soil_layers) == validated_soil_layer_count(root.sf_surface_physics), where
        below.append((table == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID, rung["dx_m"] < 1000.0))
    if default_profile_for(source, 3000.0, 1) == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID:
        # A source whose own default is this suite takes it on both rungs.
        assert all(sub_km for sub_km, _finer in below), (source, below)
    elif default_profile_for(source, 750.0, 1) == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID:
        # The spacing override binds on 750 m when the coarse default differs.
        assert all(sub_km == finer for sub_km, finer in below), (source, below)


def test_analysis_archive_catalog_preserves_inventory_scoped_initialization():
    """A downloadable archive remains preparable from inspected times.

    The catalog must not offer initialization without supplied inputs when
    the selection owner has no declared analysis publication window.
    """
    from datetime import datetime, timedelta, timezone
    from woof.background_contract import capability, plan

    facts = capability("20crv3-cf")
    assert facts["preparable"] is True
    assert facts["initialization_modes"] == ["prepared"]
    assert facts["time_axis"] == "analysis_times"
    assert facts["cycle_grid"] is None
    init = datetime(2010, 1, 1, 12, tzinfo=timezone.utc)
    arguments = dict(init=init, now=init + timedelta(days=2), run_seconds=2700.)
    with pytest.raises(ValueError, match="supplied native time inventory"):
        plan("20crv3-cf", **arguments)
    times = (init, init + timedelta(hours=3))
    selected = plan("20crv3-cf", **arguments, inventory_times=times)
    assert selected.source == facts["source"]
    assert selected.valid_times == tuple(time.isoformat() for time in times)
    assert selected.acquisition == "supplied"
    assert selected.fetch_hints() is None


@pytest.mark.parametrize("ladder", ["12-3-1-0.5", "auto"])
def test_the_assistant_choosing_the_sources_own_default_on_a_sub_km_ladder_runs_it(tmp_path, monkeypatch, ladder):
    """The real planner picks Morrison with KF (gfs's own default) on a ladder reaching 500 m.

    It is not what that ladder runs unnamed, so the plan names it, and the run asserted it on every domain:
    the domain-tree forecast refused it after the download and the preparation, because the wizard turns
    cumulus off on the nests and damps them by the depth ladder.  The run now takes the file the wizard
    wrote from it, Morrison on the root, and asserts nothing the forecast refuses.

    The system row names a 32 GB card.  The fit must reach the 500 m nest with MYNN at that card's
    36,864 columns even off a card, or with a stale 16 GB runtime choice.  Pricing the off-card cap
    instead made auto stop at 1 km, where the source's own default is the unnamed one and nothing
    is named.
    """

    from datetime import datetime, timezone
    from types import SimpleNamespace

    from woof.experiment import load_experiment
    from woof.gui.assistant.plan import Planner
    from woof.core import mynn_pbl_scratch as scratch, preflight as pf

    card = "32gb"
    monkeypatch.delenv(scratch.MYNN_PBL_COLUMN_CHUNK_ENV, raising=False)
    # CUDA usable total measured on the physical 16 GB reference card.
    stale = scratch.mynn_column_chunk_for_memory(
        49, total_bytes=16611278848, free_bytes=15 * 1024 ** 3, environ={})
    assert stale.chunk == 16384
    monkeypatch.setattr(scratch, "_RESOLVED", {49: stale})
    monkeypatch.setattr(scratch, "_PINNED", None)
    monkeypatch.setattr(scratch, "_TILE_WALKED", {})
    memo = dict(scratch._RESOLVED)
    published = scratch.MYNN_PBL_COLUMN_CHUNK

    def no_host_probe(device=None):
        pytest.fail("the planner's declared-card fit consulted the host card")

    monkeypatch.setattr(scratch, "probe_mynn_card", no_host_probe)
    column_chunk = pf.mynn_pbl_column_chunk
    priced_widths = []

    def priced(cfg, **kwargs):
        width = column_chunk(cfg, **kwargs)
        if int(cfg.nx) * int(cfg.ny) >= 36864:
            priced_widths.append((int(cfg.nz), width))
        return width

    monkeypatch.setattr(pf, "mynn_pbl_column_chunk", priced)
    source = "gfs"
    row = _menus()[source]
    own = row["default_profile"]
    assert own != THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID

    fits = []

    def fit(payload, dry):
        # The page's Check the fit: the grids of the file the fields resolve to.
        fits.append(dict(payload))
        _plan, config = _written(tmp_path, {"products": None, "profile": None, "dx_km": None, **payload},
                                 f"fit{len(fits)}")
        spacings = [domain.run.dx / 1000.0 for domain in load_experiment(config).domains]
        return SimpleNamespace(body={"fit": {"domains": [{"dx_km": dx} for dx in spacings], "words": "fits."}})

    sources_row = {"id": source, "name": "GFS", "coverage": None, "horizon_hours": 384, "step_hours": 1,
                   "default_profile": own, "spacing_defaults": row["spacing_defaults"],
                   "profiles": [{"id": own, "summary": "Morrison"},
                                {"id": THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID, "summary": "Thompson, MYNN, RUC"}]}
    api = SimpleNamespace(sources=lambda: {"default_cycle": CYCLE, "sources": [sources_row],
                                           "ladders": ["12", "12-3", "12-3-1", "12-3-1-0.5", "auto"]},
                          system=lambda: {"card": card, "devices": [{"name": "card"}]}, fit=fit)
    picks = {"day": "now", "box_size": "300", "ladder": ladder, "source": source, "physics": own,
             "machine": "this-computer"}

    def decide(question, state):
        return {"question": question.id, "choice": picks[question.id], "reason": "",
                "options": dict(question.options), "probability": 1.0}

    planner = Planner(None, decide, lambda name, args, fn: fn(), api,
                      now=datetime(2026, 9, 20, 14, tzinfo=timezone.utc))
    lat, lon = _point(source)
    fields = planner.plan("storms", place={"lat": lat, "lon": lon, "place": "here", "finest_km": None})["fields"]
    # The fitted or named ladder reaches 500 m, where the unnamed default is the sub-km row's: own is named.
    if ladder == "auto":
        # Fitted before the physics question, with no physics named: the run the field left alone makes.
        assert "profile" not in fits[0]
        # The fit priced the sub-km suite's MYNN at the width the named card walks, not at the off-card cap.
        assert priced_widths and set(priced_widths) == {(49, 36864)}
        assert scratch._RESOLVED == memo and scratch.MYNN_PBL_COLUMN_CHUNK == published
    assert fields["profile"] == own and fields["ladder"] == ladder
    plan, config = _written(tmp_path, dict(fields), "run")
    assert min(domain.run.dx for domain in load_experiment(config).domains) == pytest.approx(500.0)
    root = load_experiment(config).root.run
    assert _suite(own) == tuple(int(getattr(root, key)) for key in SUITE_KEYS)
    assert _held_to_the_run(plan, config, source) in (None, own)
