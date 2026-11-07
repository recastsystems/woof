"""Moving nests on run-plan's front door: derived, reported, priced.

1.8.4 sealed the statics corridor at preparation and taught the tree
runner to honour a ``[relocation]`` follow source when one is present.
What a MACHINE caller could not do was produce such a preparation from a
plan document: run-plan's own tests never drove a follow config, and
nothing in ``--resolve`` or ``--estimate`` mentioned a corridor, so a
front end had no way to know a plan would build one, what it would cost,
or -- on the HRRR chain -- that it would fail.

The corridor flag itself turned out to be already flowing: run-plan's
prepared route hands the config to ``go_main``, whose prepare stage
derives ``--statics-corridor`` from the config's own follow predicate.
:func:`test_a_follow_plan_composes_the_corridor_prepare_command` is the
test that was missing under that claim -- it drives the real chain and
reads the composed command -- and the rest of this file is the
reporting, pricing and refusal that genuinely were not there.

CPU-only: the chain is captured at ``go_cli._run_stage``, the seam every
other prepared-route test uses, and no stage subprocess is spawned.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

import woof.runplan as runplan_module
from woof import go_cli, run_stamp
from woof.cli import main as cli_main
from woof.runplan import (PLAN_SCHEMA, PlanError, estimate_plan, load_plan,
                           resolve_plan)

_PROFILE = "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"


# ---------------------------------------------------------------------------
# Fixtures: a wizard-emitted config, and the same config grown a nest
# ---------------------------------------------------------------------------

def _emit(directory: Path, name: str, source: str) -> Path:
    """One wizard-emitted config per source, with that source's physics.

    The GFS arm names the certified profile explicitly; the HRRR arm
    takes the wizard's own ``--source hrrr`` default, because the
    nested HRRR route refuses a suite with ``cu_physics != 0`` and the
    wizard says so at emission.
    """
    out = directory / f"{name}.toml"
    argv = [
        "domain", "--point=35.3,-97.5", "--card", "24gb", "--ladder", "12",
        "--source", source, "--cycle", "2026-07-29T18", "--hours", "6",
        "--out", str(out)]
    if source == "gfs":
        argv += ["--physics-profile", _PROFILE]
    assert cli_main(argv) == 0
    return out


def _with_child(config: Path, *, follow: bool) -> str:
    """The emitted config plus a child, and optionally an itinerary.

    The two arms differ ONLY by the ``[relocation]`` block, so every
    "with follow / without follow" pair below is comparing the presence
    of a moving nest and nothing else.
    """
    from woof.experiment import load_experiment

    dt = float(load_experiment(config).root.run.dt)
    text = config.read_text(encoding="utf-8") + """
[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 30
j_parent_start = 30
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = 45
ny = 45
history_interval_s = 3600.0
"""
    if not follow:
        return text
    return text + f"""
[relocation]
enabled = true
grid_id = 2

[[relocation.move]]
at_seconds = {dt * 2:.1f}
di_parent_cells = 1
dj_parent_cells = 0
"""


@pytest.fixture(scope="module")
def gfs_tree(tmp_path_factory):
    """A two-domain GFS config that follows, and its still twin."""

    directory = tmp_path_factory.mktemp("gfs-tree")
    base = _emit(directory, "base", "gfs")
    made = {}
    for label, follow in (("follow", True), ("still", False)):
        path = directory / f"{label}.toml"
        path.write_text(_with_child(base, follow=follow), encoding="utf-8")
        path.with_suffix(".namelist.wps").write_bytes(
            base.with_suffix(".namelist.wps").read_bytes())
        made[label] = path
    return made


@pytest.fixture(autouse=True)
def _two_plan_directories(tmp_path):
    """Tests that compare two arms write one plan into each of these."""

    (tmp_path / "a").mkdir(exist_ok=True)
    (tmp_path / "b").mkdir(exist_ok=True)


def _plan(tmp_path: Path, config: Path, name: str = "corridor-plan") -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": name, "route": "prepared",
        "config": {"path": str(config)},
        "output_root": str(tmp_path / "out")}), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. The composed preparation, from the real chain
# ---------------------------------------------------------------------------

def _drive_to_prepare(plan_path: Path, monkeypatch) -> dict[str, list[str]]:
    """Run the prepared route for real, stopping at the prepare stage.

    Every stage is captured at ``_run_stage`` -- the same seam the rest
    of the prepared-route suite uses -- so the commands recorded here
    are the ones the chain composed, not ones this test rebuilt.  The
    manifest stage writes its artifact because the stage AFTER it
    refuses without one, and that refusal is not what is under test.
    """
    captured: dict[str, list[str]] = {}
    readiness_calls = []

    def _ready():
        # Command composition stops before every real stage. Isolate the
        # launch's separate kernel-readiness check at its own boundary too;
        # its actual failures remain covered by test_go_native_launch.
        readiness_calls.append(True)

    monkeypatch.setattr(go_cli, "_require_forecast_device", _ready)

    def _stage(label, command, **kwargs):
        captured[label] = list(command)
        if label == "manifest":
            manifest = Path(command[command.index("--manifest-out") + 1])
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("{}", encoding="utf-8")
        if label == "prepare":
            raise go_cli.GoStageFailed(7)

    monkeypatch.setattr(go_cli, "_run_stage", _stage)
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: Path("bridge"))
    monkeypatch.setattr(go_cli, "geography_refusal", lambda root, **_: None)
    monkeypatch.setattr(go_cli, "memory_gate", lambda plan, **kw: {
        "verdict": "the forecast is the memory-binding phase at 9.00 GiB",
        "refuse": False, "warn": False, "free_bytes": 30 * 1024 ** 3})

    plan = load_plan(plan_path)
    resolution, exp, data = resolve_plan(plan, require_inputs=False)

    class _Observer:
        def enter_stage(self, *args, **kwargs):
            pass

        def finish_stage(self, *args, **kwargs):
            pass

        def warn(self, *args, **kwargs):
            pass

    with pytest.raises(RuntimeError):
        runplan_module._execute_prepared_route(
            plan, exp=exp, data=data, config_path=plan.config_path,
            observer=_Observer())
    assert readiness_calls == [True]
    return captured


def test_a_follow_plan_composes_the_corridor_prepare_command(
        tmp_path, gfs_tree, monkeypatch):
    """One plan document in, a corridor-sealing preparation out.

    The claim this file exists to hold: a caller who submits a plan
    naming a follow config gets a bundle the tree runner will accept,
    without having to know the flag exists.
    """
    captured = _drive_to_prepare(
        _plan(tmp_path, gfs_tree["follow"]), monkeypatch)

    assert set(captured) == {"authority", "fetch", "manifest", "prepare"}
    prepare = captured["prepare"]
    assert "--statics-corridor" in prepare
    # Bare, not scoped to a grid id: the preparation reads that as
    # "every child domain", which is what corridor_estimate prices.
    flag = prepare.index("--statics-corridor")
    assert prepare[flag + 1].startswith("--")
    # And it is the rw-wps preparation that carries it, not some other
    # stage that happens to mention a corridor.
    assert "woof.source_cli" in prepare


def test_a_still_nest_emits_exactly_what_it_always_did(
        tmp_path, gfs_tree, monkeypatch):
    """The no-follow arm is byte-identical to today's emission.

    Not "does not contain --statics-corridor": the WHOLE composed
    chain, stage for stage and token for token, against the same tree
    without the [relocation] block.  A corridor that leaked into an
    ordinary nested run would change what every existing plan prepares.
    """
    # The two arms are two separate drives, so each allocates its own
    # run stamp off the wall clock.  Left to the clock, this row failed
    # whenever the pair straddled a second -- observed on test host 1 as
    # run-20260818-185602Z against run-20260818-185603Z on the authority
    # stage -- which is a red that says nothing about corridors.  The
    # boundary is FORCED here rather than avoided, so the masking below
    # is exercised on every run instead of once in a while, and a mask
    # that stopped covering the stamp fails immediately rather than
    # intermittently.
    clock = [datetime.datetime(2026, 8, 18, 18, 56, 2,
                               tzinfo=datetime.timezone.utc)]
    monkeypatch.setattr(run_stamp, "utcnow", lambda: clock[0])

    still = _drive_to_prepare(
        _plan(tmp_path / "a", gfs_tree["still"]), monkeypatch)
    clock[0] += datetime.timedelta(seconds=1)
    follow = _drive_to_prepare(
        _plan(tmp_path / "b", gfs_tree["follow"]), monkeypatch)

    assert "--statics-corridor" not in still["prepare"]

    # The two arms necessarily differ in their output root, in the name
    # of the config file they were emitted to, and in the run stamp each
    # drive allocated; all three are erased so what is left is the argv
    # SHAPE, which must be identical.
    #
    # The stamp is masked with run_stamp.STAMP_RE -- the product's own
    # definition of "is this a run folder" -- rather than a pattern
    # spelled here, so the mask cannot drift from the names run_stamp.py
    # emits.  Masking hides nothing this row is looking for: the stamp is
    # derived from the wall clock and the init time, and a corridor
    # cannot reach either, so it can only ever differ for the reason the
    # forced tick above makes it differ.
    def _shape(tokens: list[str], root: Path, label: str) -> list[str]:
        shaped = []
        for token in tokens:
            token = (token.replace(str(root), "<ROOT>").replace("\\", "/")
                     .replace(f"/{label}.", "/<CONFIG>."))
            shaped.append("/".join(
                "<RUN-STAMP>" if run_stamp.STAMP_RE.match(part) else part
                for part in token.split("/")))
        return shaped

    still_shape = {stage: _shape(tokens, tmp_path / "a", "still")
                   for stage, tokens in still.items()}
    follow_shape = {stage: _shape(tokens, tmp_path / "b", "follow")
                    for stage, tokens in follow.items()}
    assert set(still_shape) == set(follow_shape)
    for stage in ("authority", "fetch", "manifest"):
        assert still_shape[stage] == follow_shape[stage], stage
    assert (still_shape["prepare"]
            == [token for token in follow_shape["prepare"]
                if token != "--statics-corridor"])


# ---------------------------------------------------------------------------
# 2. --resolve: the caller can SEE the corridor before launching
# ---------------------------------------------------------------------------

def test_resolve_reports_the_corridor_and_names_its_basis(tmp_path,
                                                          gfs_tree):
    resolution, _exp, _data = resolve_plan(
        load_plan(_plan(tmp_path, gfs_tree["follow"])), require_inputs=False)

    assert resolution["moving_nest"] == {
        "chain": "prepared:go",
        "delivery": "statics_corridor",
        "relocation_grid_id": 2,
        "statics_corridor": True,
    }
    entry, = [item for item in resolution["automatic_resolutions"]
              if item.get("key") == "statics_corridor"]
    assert entry["scope"] == "preparation"
    assert entry["value"] is True
    assert entry["basis"] == "relocation_follow"
    assert "--statics-corridor" in entry["note"]


def test_a_still_plan_says_nothing_about_corridors(tmp_path, gfs_tree):
    """No follow source, no new resolution and no new record.

    A plan that moves nothing must resolve exactly as it did before
    this feature existed; a null `moving_nest` and an absent resolution
    are how that is stated.
    """
    resolution, _exp, _data = resolve_plan(
        load_plan(_plan(tmp_path, gfs_tree["still"])), require_inputs=False)

    assert resolution["moving_nest"] is None
    assert not [item for item in resolution["automatic_resolutions"]
                if item.get("key") == "statics_corridor"]


# ---------------------------------------------------------------------------
# 3. --estimate: the corridor is priced before it is built
# ---------------------------------------------------------------------------

def test_estimate_prices_the_corridor_off_the_real_arithmetic(tmp_path,
                                                              gfs_tree):
    from woof.experiment import load_experiment
    from woof.static.corridor import (CORRIDOR_BYTES_PER_CELL,
                                       CORRIDOR_PLANES_PER_CELL)

    plan = load_plan(_plan(tmp_path, gfs_tree["follow"]))
    estimate = estimate_plan(plan)
    corridor = estimate["corridor"]
    root = load_experiment(gfs_tree["follow"]).root

    entry, = corridor["domains"]
    assert entry["domain"] == "d02"
    assert entry["grid_id"] == 2 and entry["parent_id"] == 1
    # The ground the nest can REACH at child resolution, not the parent's
    # whole extent (root.nx*3 x root.ny*3, which is what this priced and
    # sealed before 2.8).  The itinerary moves d02 one parent cell east
    # and nowhere else, so its corridor is its 45x45 footprint widened by
    # three child cells on the east side, starting at its own placement.
    assert entry["corridor_nx"] == 45 + 3 < root.run.nx * 3
    assert entry["corridor_ny"] == 45 < root.run.ny * 3
    assert entry["window_child_cells"] == [(30 - 1) * 3, (30 - 1) * 3,
                                           48, 45]
    assert entry["whole_frame"] is False
    assert entry["cells"] == entry["corridor_nx"] * entry["corridor_ny"]
    assert entry["cells"] > 45 * 45

    assert entry["planes_per_cell"] == CORRIDOR_PLANES_PER_CELL
    assert entry["bytes_per_cell"] == CORRIDOR_BYTES_PER_CELL
    assert entry["host_bytes"] == entry["cells"] * CORRIDOR_BYTES_PER_CELL
    assert corridor["host_bytes"] == entry["host_bytes"] > 0
    # host_gib is rounded to four places, which a corridor this small
    # (a few megabytes) resolves only to the last place.
    assert corridor["host_gib"] == pytest.approx(
        corridor["host_bytes"] / 1024 ** 3, abs=5e-5)
    assert "No GPU residency" in corridor["basis"]


def test_the_corridor_costs_no_vram(tmp_path, gfs_tree, monkeypatch):
    """Disk and host only -- the VRAM figure must not move.

    The corridor is cropped on the host, so a plan that seals one and
    the same plan without it are the same size on the card.  A front
    end that priced it into the VRAM budget would refuse runs that fit.

    The estimate reads the card in the machine through its probe, and
    ``device_free_bytes`` is that reading at the moment of the call, so
    two estimates on a shared card compare two moments rather than two
    plans: on the release node's RTX 4090 the pair read 24402984960
    against 22796763136 bytes free (proof/node-reds-276).  With the
    local card banned both price the declared profile, and what is
    compared is the plan.
    """
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    moving = estimate_plan(load_plan(_plan(tmp_path / "a",
                                           gfs_tree["follow"])))
    still = estimate_plan(load_plan(_plan(tmp_path / "b",
                                          gfs_tree["still"])))

    assert moving["vram"] == still["vram"]
    assert moving["corridor"]["host_bytes"] > 0
    assert still["corridor"]["host_bytes"] == 0
    assert still["corridor"]["domains"] == []
    assert "no [relocation] follow source" in still["corridor"]["basis"]


def test_every_child_is_priced_not_only_the_one_that_moves(tmp_path,
                                                           gfs_tree):
    """Bare ``--statics-corridor`` covers every child, so price them all.

    The flag run-plan composes takes no grid-id list, and the
    preparation reads that as "every child domain".  An estimate that
    priced only the relocating nest would under-report a three-domain
    tree's disk cost by the corridors it is about to write anyway.
    """
    from woof.static.corridor import validated_corridor_selection

    plan = load_plan(_plan(tmp_path, gfs_tree["follow"]))
    _resolution, exp, _data = resolve_plan(plan, require_inputs=False)

    priced = [entry["grid_id"]
              for entry in estimate_plan(plan)["corridor"]["domains"]]
    # The preparation's OWN selection for the bare flag, asked rather
    # than restated here.
    assert priced == list(validated_corridor_selection(exp, "all"))


# ---------------------------------------------------------------------------
# 4. The HRRR chain: was refused here, now feeds a moving nest too
# ---------------------------------------------------------------------------

def _hrrr_follow(tmp_path: Path) -> Path:
    base = _emit(tmp_path, "hrrr-base", "hrrr")
    config = tmp_path / "hrrr-follow.toml"
    config.write_text(_with_child(base, follow=True), encoding="utf-8")
    return config


def test_a_follow_config_on_the_hrrr_chain_resolves_and_seals(tmp_path):
    """The refusal this replaces was about EMISSION, not the runner.

    The tree runner never cared which preparation sealed the corridor
    -- it reads the corridor set out of whichever hierarchy document
    matched its pinned digest.  What was missing was an HRRR stage that
    wrote one.  With ``woof.hrrr_hierarchy_direct --statics-corridor``
    the delivery table says the same word for both prepared chains, and
    this plan resolves instead of being turned away.
    """
    resolution, _exp, _data = resolve_plan(
        load_plan(_plan(tmp_path, _hrrr_follow(tmp_path))),
        require_inputs=False)

    assert resolution["moving_nest"] == {
        "chain": "prepared:hrrr",
        "delivery": "statics_corridor",
        "relocation_grid_id": 2,
        "statics_corridor": True,
    }
    entry, = [item for item in resolution["automatic_resolutions"]
              if item.get("key") == "statics_corridor"]
    assert entry["value"] is True
    assert entry["basis"] == "relocation_follow"
    # The note names the stage that actually carries the flag on THIS
    # chain.  "the prepare stage" is the GFS sentence and would send a
    # reader to tools.prepare_hrrr_wrf, which seals no corridor and
    # would refuse the flag.
    assert "hrrr_hierarchy_direct" in entry["note"]
    assert "rw-wps" not in entry["note"]


def test_the_gfs_note_still_names_rw_wps(tmp_path, gfs_tree):
    """The other half of the same guard: neither chain inherits the
    other's stage name."""

    resolution, _exp, _data = resolve_plan(
        load_plan(_plan(tmp_path, gfs_tree["follow"])), require_inputs=False)
    entry, = [item for item in resolution["automatic_resolutions"]
              if item.get("key") == "statics_corridor"]
    assert "rw-wps" in entry["note"]
    assert "hrrr_hierarchy_direct" not in entry["note"]


def test_the_hrrr_plan_is_priced_by_the_same_chain_agnostic_arithmetic(
        tmp_path):
    """`corridor_estimate` reads the domain tree, never the chain.

    Verified rather than assumed: the SAME experiment priced as a GFS
    plan and as an HRRR plan must produce identical figures, because the
    corridor's size is a property of the geometry and the static field
    inventory and of nothing else.
    """
    from woof.runplan import corridor_estimate, follow_statics_decision

    plan = load_plan(_plan(tmp_path, _hrrr_follow(tmp_path)))
    _resolution, exp, _data = resolve_plan(plan, require_inputs=False)

    hrrr = corridor_estimate(
        exp, follow_statics_decision(exp, chain="prepared:hrrr"))
    gfs = corridor_estimate(
        exp, follow_statics_decision(exp, chain="prepared:go"))
    assert hrrr == gfs
    assert hrrr["host_bytes"] > 0

    # And the front door reports that same block for the HRRR plan.
    assert estimate_plan(plan)["corridor"] == hrrr


def test_the_hrrr_estimate_costs_no_vram(tmp_path, monkeypatch):
    """Host and disk only, on this chain as on the other.

    The local card is banned for the reason the test above states: the
    comparison is of two plans, not of two readings of a shared card.
    """
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    moving = estimate_plan(load_plan(_plan(tmp_path / "a",
                                           _hrrr_follow(tmp_path))))
    base = _emit(tmp_path / "b", "hrrr-base", "hrrr")
    still_config = tmp_path / "b" / "hrrr-still.toml"
    still_config.write_text(_with_child(base, follow=False),
                            encoding="utf-8")
    still = estimate_plan(load_plan(_plan(tmp_path / "b", still_config)))

    assert moving["vram"] == still["vram"]
    assert moving["corridor"]["host_bytes"] > 0
    assert still["corridor"]["host_bytes"] == 0


def test_a_still_hrrr_tree_says_nothing_about_corridors(tmp_path):
    """Only a MOVING nest pays for one."""

    base = _emit(tmp_path, "hrrr-base", "hrrr")
    config = tmp_path / "hrrr-still.toml"
    config.write_text(_with_child(base, follow=False), encoding="utf-8")

    resolution, _exp, _data = resolve_plan(
        load_plan(_plan(tmp_path, config)), require_inputs=False)
    assert resolution["moving_nest"] is None
    assert not [item for item in resolution["automatic_resolutions"]
                if item.get("key") == "statics_corridor"]


def test_every_preparation_contract_supplies_moving_nest_statics():
    """Mapped hierarchy uses the same sealed corridor as the other chains."""
    from woof.source_cli import preparation_runners, preparation_statics

    assert all(row.moving_statics()["delivery"] == "statics_corridor"
               for row in preparation_runners().values())
    assert preparation_statics("prepared:staged")["delivery"] == "statics_corridor"


# ---------------------------------------------------------------------------
# 5. The completeness guard on the delivery table
# ---------------------------------------------------------------------------

def test_every_chain_declares_where_a_moving_nest_gets_its_statics():
    """A chain cannot be added without answering the question.

    The table is the whole mechanism: a chain missing from it would
    otherwise fall through to whatever the chain happened to do, which
    on a prepared route means preparing a bundle its own forecast stage
    refuses.  Every key `_chain_key` can produce is checked here
    against the routes that actually exist.
    """
    from woof.runplan import ROUTES, _chain_key
    from woof.source_cli import preparation_runners, preparation_statics
    from woof.source_adapters import source_adapters

    reachable = set()
    for route in ROUTES:
        # The prepared route forks on the config's [fetch].source; every
        # other route is its own chain.  Every registered id is walked,
        # so a chain a registry row can reach cannot hide from this
        # completeness gate.
        for source in (None, *(a.source_id for a in source_adapters())):
            reachable.add(_chain_key(route, source))
    # Explicit prepared_root bypasses source preparation and verifies an
    # already sealed corridor through the same tree runner.
    reachable.add("prepared:existing")
    assert reachable == {row.chain for row in preparation_runners().values()} | {"prepared:existing"}
    assert all(preparation_statics(chain)["delivery"] is not None for chain in reachable)


@pytest.mark.parametrize("missing", ["hierarchy_schema", "corridor_stage"])
def test_missing_preparation_outputs_name_the_actual_requirement(missing):
    from dataclasses import replace
    from woof.source_cli import preparation_runners

    row = next(iter(preparation_runners().values()))
    answer = replace(row, **{missing: None}).moving_statics()
    assert answer["delivery"] is None
    assert "implementation" in answer["reason"]
    assert ("hierarchy" if missing == "hierarchy_schema" else "corridor writer") in answer["reason"]
    assert "supply a prepared hierarchy with verified corridors" in answer["reason"]


def test_every_corridor_chain_names_the_stage_that_carries_the_flag():
    from woof.source_cli import preparation_runners, preparation_statics

    chains = {row.chain for row in preparation_runners().values()} - {"experiment"}
    assert all(preparation_statics(chain)["stage"] for chain in chains)
    assert "source_cli" in preparation_statics("prepared:go")["stage"]
    assert "hrrr_hierarchy_direct" in preparation_statics("prepared:hrrr")["stage"]
    assert "mapped_direct" in preparation_statics("prepared:staged")["stage"]


def test_the_retained_refusal_still_says_something_useful():
    """No chain reaches it today; it must still work when one does.

    Machinery kept "for later" and never exercised is machinery that
    has quietly rotted by the time later arrives.  This calls the
    refusal directly, on a chain with no bespoke detail, and requires
    the generic sentence to name the problem and all three remedies.
    """
    from woof.runplan import _follow_unsupported_refusal

    message = _follow_unsupported_refusal("prepared:someday",
                                          relocation_grid_id=3)
    assert "[relocation] follow source on d03" in message
    assert "'prepared:someday' chain cannot supply the statics" in message
    assert "seals no child-resolution statics corridor" in message
    assert "remedy:" in message
    # The remedy names the required preparation contract, independent of source.
    assert "prepared chain that seals a corridor" in message
    assert "[case_data]" in message
    assert "bounds-only [relocation]" in message


def test_a_chain_with_no_declared_delivery_refuses_rather_than_guesses():
    from woof.runplan import follow_statics_decision

    class _Relocation:
        enabled, follow, moves, grid_id = True, "vortex", (), 2

    class _Experiment:
        relocation = _Relocation()

    with pytest.raises(PlanError) as refusal:
        follow_statics_decision(_Experiment(), chain="prepared:invented")
    assert "preparation implementation in the dispatcher" in str(refusal.value)


def test_the_dispatch_and_the_refusal_read_the_same_chain(tmp_path,
                                                          gfs_tree,
                                                          monkeypatch):
    """`_execute_prepared_route` branches on `_chain_key`, not a copy.

    If the dispatch grew its own test of the source, a config could be
    judged corridor-capable at resolve time and then sent down the
    chain that cannot build one.  Proven by making `_chain_key` claim
    everything is HRRR and watching the GFS plan land in `_hrrr_chain`.
    """
    plan = load_plan(_plan(tmp_path, gfs_tree["follow"]))
    # Resolved FIRST, against the real chain key: this plan is a
    # legitimate corridor-bearing GFS plan, and resolve_plan agrees.
    _resolution, exp, data = resolve_plan(plan, require_inputs=False)

    seen: list[str] = []
    monkeypatch.setattr(runplan_module, "_chain_key",
                        lambda route, source: "prepared:hrrr")
    monkeypatch.setattr(
        runplan_module, "_hrrr_chain",
        lambda *a, **k: seen.append("hrrr") or {})
    runplan_module._execute_prepared_route(
        plan, exp=exp, data=data, config_path=plan.config_path,
        observer=None)
    assert seen == ["hrrr"]


# ---------------------------------------------------------------------------
# 6. The predicate all three doors read
# ---------------------------------------------------------------------------

def test_one_predicate_feeds_go_the_printed_line_and_run_plan(gfs_tree):
    """`woof go`, the pasted rw-wps line and run-plan cannot disagree.

    They used to hold three copies of the same sentence, kept equal by
    a drift test.  Now there is one function and the drift test proves
    the obvious; this test names the callers so a fourth reader cannot
    quietly reintroduce a copy.
    """
    from woof.experiment import load_experiment
    from woof.static.corridor import config_declares_follow_source

    following = load_experiment(gfs_tree["follow"])
    still = load_experiment(gfs_tree["still"])
    assert config_declares_follow_source(following) is True
    assert config_declares_follow_source(still) is False

    # go's plan and run-plan's decision, from that one predicate.
    plan = go_cli.plan_from_config(gfs_tree["follow"],
                                   outdir=gfs_tree["follow"].parent / "go")
    assert plan["statics_corridor"] is True
    decision = runplan_module.follow_statics_decision(
        following, chain="prepared:go")
    assert decision["statics_corridor"] is True
    assert runplan_module.follow_statics_decision(
        still, chain="prepared:go") is None


def test_the_same_predicate_feeds_both_hrrr_doors(tmp_path):
    """The driven HRRR chain and the wizard's printed one, one function.

    Two more readers of the same sentence arrived with the HRRR
    emission: run-plan's ``_hrrr_chain`` (which composes the hierarchy
    stage's argv) and ``domain_wizard.hrrr_route_commands`` (which
    prints that same stage for a reader to paste).  A printed chain that
    disagreed with the driven one would hand somebody a paste that
    prepares a bundle the last line of the same paste refuses.
    """
    import inspect

    from woof import domain_wizard
    from woof.experiment import load_experiment
    from woof.static.corridor import config_declares_follow_source

    base = _emit(tmp_path, "hrrr-base", "hrrr")
    follow_path = tmp_path / "hrrr-follow.toml"
    follow_path.write_text(_with_child(base, follow=True), encoding="utf-8")
    still_path = tmp_path / "hrrr-still.toml"
    still_path.write_text(_with_child(base, follow=False), encoding="utf-8")
    following = load_experiment(follow_path)
    still = load_experiment(still_path)

    assert config_declares_follow_source(following) is True
    assert config_declares_follow_source(still) is False
    assert runplan_module.follow_statics_decision(
        following, chain="prepared:hrrr")["statics_corridor"] is True
    assert runplan_module.follow_statics_decision(
        still, chain="prepared:hrrr") is None

    # The printed chain: flag present for the mover, absent for the
    # still nest, and on the hierarchy stage rather than the preparer.
    # Both stages are now spelled `woof prep --source hrrr` (the
    # shipped door, not the modules behind it), so the two are told
    # apart by --root-preparation, which is what makes one of them the
    # hierarchy.
    printed = domain_wizard.hrrr_route_commands(
        base, following, profile=None, data_dir=str(tmp_path / "data"))
    assert "--statics-corridor" in printed
    root_block, hierarchy_block = printed.split("--root-preparation")
    assert "--statics-corridor" in hierarchy_block
    assert "--statics-corridor" not in root_block
    assert "--statics-corridor" not in domain_wizard.hrrr_route_commands(
        base, still, profile=None, data_dir=str(tmp_path / "data"))

    # And both HRRR doors reach the predicate itself rather than
    # re-deriving it: neither source mentions `relocation.follow`.
    for function in (domain_wizard.hrrr_route_commands,
                     runplan_module._hrrr_chain):
        source = inspect.getsource(function)
        assert "config_declares_follow_source" in source
        assert "relocation.follow" not in source


def test_the_tree_runner_asks_the_predicate_too():
    """The door the whole mechanism exists to satisfy held its own copy.

    A reading of ``[relocation]`` that the preparation doors agreed on
    could have differed in the one place that decides whether a bundle
    is accepted -- so the runner asks the same function.
    """
    import inspect

    from woof import prepared_domain_tree_forecast as runner

    source = inspect.getsource(runner.preflight_prepared_tree)
    assert "config_declares_follow_source(exp)" in source
    assert "exp.relocation.follow is not None" not in source


def test_bounds_only_relocation_is_not_a_moving_nest(gfs_tree, tmp_path):
    """Enabled [relocation] with no itinerary needs no corridor."""

    from woof.static.corridor import config_declares_follow_source

    config = tmp_path / "bounds.toml"
    config.write_text(
        gfs_tree["still"].read_text(encoding="utf-8")
        + "\n[relocation]\nenabled = true\ngrid_id = 2\n",
        encoding="utf-8")
    config.with_suffix(".namelist.wps").write_bytes(
        gfs_tree["still"].with_suffix(".namelist.wps").read_bytes())

    from woof.experiment import load_experiment
    assert config_declares_follow_source(load_experiment(config)) is False

    resolution, _exp, _data = resolve_plan(
        load_plan(_plan(tmp_path, config)), require_inputs=False)
    assert resolution["moving_nest"] is None
