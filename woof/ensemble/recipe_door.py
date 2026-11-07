"""The front door for source-trajectory and seeded surface-state ensemble members.

``woof ensemble CONFIG --recipe time-lagged`` (and ``go`` and ``run``, or
``[ensemble] recipe`` in the config) lands here.  The door owns no model
code and no preparation code:

* the member plan is :func:`woof.ensemble.recipes.build_recipe`, the same
  plan ``python -m woof.ensemble.recipes`` prints;
* each member's source trajectory is fetched and prepared by the ordinary
  preparation chain its source already has
  (:func:`woof.regional_preparation.preparation_chains`), from a copy of
  the config whose ``[fetch]`` table names that trajectory;
* the members run through the existing ensemble session
  (:class:`woof.ensemble.production.PreparedEnsembleSession`), which is
  handed each member's own prepared inputs.

Everything a request can be refused for is decided before the run folder
is claimed and before the first member is fetched: the member plan, every
member's own config and chain plan, and the gates the ordinary door asks
(the card, the memory envelope, the geography tree, the renderer, the
products and the disk).  ``--dry-run`` stops after the plan and the member
configs, and spends nothing.

A source recipe member is a real source trajectory. Surface-state recipes
share their unchanged source preparation and apply named, seeded GPU
perturbations independently after each member's ordinary initialization.
Named member rosters share each matching prepared authority and keep
different land selectors and soil layouts in separately bound banks.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from woof.stage_timing import ledger_main


#: Host bytes one member preparation is charged before any preparation of
#: this launch has finished and been measured.  Defect it guards (test host
#: 2026-10-18, AW36): the door started all 32 distinct HRRR preparations at
#: once on a 192-core box whose container cgroup allows 256 GiB, and the
#: kernel killed two producers (exit -9, cgroup oom_kill 2), which stopped
#: the whole ensemble.  After the first preparation ends, the measured peak
#: of the largest finished preparation process replaces this figure.
MEMBER_PREPARATION_HOST_BYTES = 16 * 1024**3
#: Share of the measured host headroom member preparations may hold at once.
MEMBER_PREPARATION_HOST_SHARE = 0.7


class _PreparationAdmission:
    """Admit member preparations against host headroom, not only CPU count.

    Each running preparation is charged ``estimate`` bytes: the configured
    figure until a preparation has finished, then the largest peak resident
    size of any finished preparation process (``RUSAGE_CHILDREN``) with a
    quarter of margin.  A preparation starts when the charges of the running
    ones plus its own fit ``MEMBER_PREPARATION_HOST_SHARE`` of the headroom
    measured when admission opened; one preparation always runs, so a small
    host is slower, never refused.  Unreadable host memory leaves the CPU
    bound alone, as before.
    """

    def __init__(self, cpu_workers):
        import threading
        from woof.ingest.preparation_workers import host_available_bytes
        try:
            headroom = host_available_bytes()
        except Exception:  # noqa: BLE001 - unreadable RAM is unknown, never a refusal
            headroom = None
        self.cpu_workers = max(1, int(cpu_workers))
        self.budget = (None if headroom is None
                       else int(headroom * MEMBER_PREPARATION_HOST_SHARE))
        self.running = 0
        self.peak_running = 0
        self.measured_peak = None
        self.condition = threading.Condition()

    def estimate(self):
        if self.measured_peak:
            return max(1, int(self.measured_peak * 1.25))
        return MEMBER_PREPARATION_HOST_BYTES

    def limit(self):
        if self.budget is None:
            return self.cpu_workers
        return max(1, min(self.cpu_workers, self.budget // self.estimate()))

    def _measure(self):
        try:
            import resource
            import sys
            peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            peak = int(peak) * (1 if sys.platform == "darwin" else 1024)
        except Exception:  # noqa: BLE001 - no reading keeps the current estimate
            return
        if peak > 0:
            self.measured_peak = peak

    def __enter__(self):
        with self.condition:
            while self.running >= self.limit():
                self.condition.wait()
            self.running += 1
            self.peak_running = max(self.peak_running, self.running)
        return self

    def __exit__(self, *exc):
        with self.condition:
            self.running -= 1
            self._measure()
            self.condition.notify_all()
        return False

    def receipt(self):
        return {"host_budget_bytes": self.budget, "estimate_bytes": self.estimate(),
                "measured_child_peak_bytes": self.measured_peak,
                "limit_now": self.limit(), "peak_concurrent": self.peak_running,
                "basis": "min of CPU workers and host headroom share / per-preparation peak"}


#: Box fetch sockets one source download needs to make headway; the most
#: downloads that run at once is the box's fetch connection cap over this.
DOWNLOAD_MIN_STREAMS = 4
#: Seconds a newly admitted download is given to lease its sockets before
#: the free-socket probe decides whether another one may start.
DOWNLOAD_SETTLE_SECONDS = 5.0


def _folder_bytes(root):
    total = 0
    for directory, _folders, files in os.walk(root):
        for name in files:
            try:
                total += os.stat(os.path.join(directory, name)).st_size
            except OSError:
                pass
    return total


class _DownloadAdmission:
    """Source downloads run beside the host-RAM preparation slots, bounded by sockets.

    A download holds no host-RAM preparation slot (:class:`_PreparationAdmission`):
    it is network work.  Its own bound is the box's fetch sockets
    (:mod:`woof.fetch_connections`, shared with every woof process on the
    box): one download always runs; another starts only while the running
    ones leave at least ``DOWNLOAD_MIN_STREAMS`` box sockets idle (a link the
    running downloads already fill gains nothing from one more, and a socket
    starved download only waits on the slot budget), and never more than the
    cap over ``DOWNLOAD_MIN_STREAMS`` at once.  Waiters start in member order,
    so the earliest members' inputs arrive first.  The concrete breakage it
    prevents: a cold 36-member HRRR ensemble (box J, 2026-10-24, 3,286 s) ran
    each leader's download inside its host-RAM slot (10 at once), so 10 of 20
    downloads queued behind decodes while the link sat idle.
    """

    def __init__(self, *, clock=None, free_slots=None, cap=None):
        import threading
        import time as _time
        from woof import fetch_connections
        self._clock = clock or _time.monotonic
        self._free_slots = free_slots or fetch_connections.free_slots
        self.cap = int(cap if cap is not None else fetch_connections.max_connections())
        self.limit = max(1, self.cap // DOWNLOAD_MIN_STREAMS)
        self.running = 0
        self.peak_running = 0
        self.admitted = 0
        self._waiting = set()
        self._last_admit = None
        self._first_start = None
        self._last_end = None
        self.bytes = 0
        self.condition = threading.Condition()

    def _admissible(self):
        if self.running == 0:
            return True
        if self.running >= self.limit:
            return False
        if self._clock() - self._last_admit < DOWNLOAD_SETTLE_SECONDS:
            return False
        try:
            return self._free_slots() >= DOWNLOAD_MIN_STREAMS
        except Exception:  # noqa: BLE001 - an unreadable socket table admits by the cap alone
            return True

    def acquire(self, order):
        with self.condition:
            self._waiting.add(order)
            try:
                while not (order == min(self._waiting) and self._admissible()):
                    self.condition.wait(timeout=1.0)
            finally:
                self._waiting.discard(order)
            self.running += 1
            self.admitted += 1
            self.peak_running = max(self.peak_running, self.running)
            self._last_admit = self._clock()
            if self._first_start is None:
                self._first_start = self._last_admit
            self.condition.notify_all()

    def release(self, folder=None):
        moved = 0 if folder is None else _folder_bytes(folder)
        with self.condition:
            self.running -= 1
            self.bytes += moved
            self._last_end = self._clock()
            self.condition.notify_all()

    def receipt(self):
        span = (None if self._first_start is None or self._last_end is None
                else round(self._last_end - self._first_start, 3))
        return {"box_fetch_connection_cap": self.cap, "most_at_once": self.limit,
                "min_idle_sockets_to_start_another": DOWNLOAD_MIN_STREAMS,
                "downloads": self.admitted, "peak_concurrent": self.peak_running,
                "folder_bytes": self.bytes, "span_seconds": span,
                "mean_mib_per_second": (None if not span else round(self.bytes / span / 2**20, 2)),
                "basis": "outside the host-RAM preparation slots; started while box fetch sockets sit idle"}


class _MemberStages:
    """Which bound one member preparation is under, from the stages it enters.

    Its fetch stage (a leader's download) runs under the download admission
    and outside the host-RAM slots; the first stage after it (the decode,
    whatever the chain calls it) takes a host-RAM slot, so a decode starts
    only once its inputs are on disk.  A follower's fetch only verifies its
    leader's folder, so it takes no download admission.  Leaving the
    download releases the folder's followers.
    """

    def __init__(self, *, admission, downloads, order, leader, folder, fetched):
        import threading
        import time as _time
        self._clock = _time.monotonic
        self._lock = threading.RLock()
        self.admission, self.downloads = admission, downloads
        self.order, self.leader, self.folder, self.fetched = order, leader, folder, fetched
        self.download_held = self.slot_held = False
        self.timings = {}
        self._download_began = None

    def __call__(self, event, fields):
        if event != "stage_started":
            return
        with self._lock:
            if fields.get("stage") == "fetch":
                if self.leader and not (self.download_held or self.slot_held
                                        or "download_seconds" in self.timings):
                    waited = self._clock()
                    self.downloads.acquire(self.order)
                    self._download_began = self._clock()
                    self.timings["download_admission_wait_seconds"] = round(
                        self._download_began - waited, 3)
                    self.download_held = True
                return
            self._enter_slot()

    def _leave_download(self):
        if self.download_held:
            self.download_held = False
            self.timings["download_seconds"] = round(self._clock() - self._download_began, 3)
            self.downloads.release(self.folder)
        if self.fetched is not None:
            self.fetched.set()

    def _enter_slot(self):
        if self.slot_held:
            return
        self._leave_download()
        waited = self._clock()
        self.admission.__enter__()
        self.slot_held = True
        self.timings["preparation_slot_wait_seconds"] = round(self._clock() - waited, 3)

    def close(self):
        with self._lock:
            self._leave_download()
            if self.slot_held:
                self.slot_held = False
                self.admission.__exit__(None, None, None)

#: The recipes a front door can select.  ``input-ensemble`` is the
#: automatic choice for a source with operational members and
#: ``recentered`` has no observation-calibrated amplitude, so neither is
#: a door option here.
RECIPES = ("wofs", "time-lagged", "multi-model", "surface-state", "member-roster")

RECEIPT_NAME = "ensemble-recipe.json"
RECEIPT_SCHEMA = "gpuwm-ensemble-recipe-door.v1"
#: Where the traceback of a failure that is not a stage's own exit goes,
#: beside the receipt that names it.
FAILURE_LOG_NAME = "ensemble-recipe-failure.log"


class RecipeRefusal(ValueError):
    """A recipe request this door will not run, in one sentence."""


def requested(args) -> bool:
    """Did the command line ask for a recipe?  The config is asked separately."""
    return getattr(args, "recipe", None) is not None or getattr(args, "trajectories", None) is not None


def configured(config) -> bool:
    """Does the config's own ``[ensemble]`` table select a recipe?"""
    import tomllib

    from woof.config_authority import CONFIG_SOURCE_ENV, read_config_authority

    path = Path(config)
    # Read through the config authority, as the run itself reads it.  A
    # multi-run child is bound to the bytes its supervisor captured; the
    # original file may have been edited since, and reading it here let
    # those later bytes decide whether the child runs as a recipe
    # ensemble or as one forecast.  Outside a capture the authority reads
    # the file itself.
    if not os.environ.get(CONFIG_SOURCE_ENV) and not path.is_file():
        return False
    try:
        payload = read_config_authority(path).payload
        table = tomllib.loads(payload.decode("utf-8-sig")).get("ensemble")
    except (OSError, ValueError, tomllib.TOMLDecodeError):
        return False
    return isinstance(table, dict) and (table.get("recipe") is not None
        or bool(table.get("trajectories")) or bool(table.get("member_variants")))


def refuse_continuation(command, config) -> None:
    """``woof resume`` and ``woof branch`` on a config that selects a recipe.

    Breakage it prevents: a checkpoint holds ONE prepared trajectory, so
    the continued run would be N copies of it under a config that asks
    for N different source trajectories.  The session refused that only
    at the forecast stage, after the card was locked and a worker started.
    """
    if config is None or not configured(config):
        return
    raise RecipeRefusal(
        f"woof {command} continues one prepared trajectory from its checkpoint, and "
        f"{Path(config).name} selects an ensemble recipe, whose members each come from "
        "their own source trajectory: every continued member would be a copy of the "
        "checkpointed one. Next: woof ensemble CONFIG --recipe time-lagged starts a "
        f"recipe ensemble; remove [ensemble] recipe and trajectories to {command} this run")


def refuse_unsupervised(args) -> None:
    """Refuse the ``woof run`` supervision flags a recipe does not consume.

    A recipe ensemble under ``woof run`` runs in the calling process,
    ahead of the supervisor: no worker, no card lock, no fresh-process
    recovery and no checkpoint to continue.  Breakage it prevents: each of
    these flags was parsed and read by nothing, so the run exited 0
    without doing what its command line said, most visibly a ``--gpu-uuid``
    pin on a multi-card host whose members then ran on every visible card.
    """
    given = [(flag, why) for flag, stated, why in (
        ("--restart", getattr(args, "restart", None) is not None,
         "it continues one forecast from its checkpoint, and a recipe ensemble "
         "prepares and starts every member from its own source, so the run "
         "would be a fresh fetch and forecast in place of the continuation"),
        ("--gpu-uuid", getattr(args, "gpu_uuid", None) is not None,
         "it pins and locks one card for a supervised worker, and the members "
         "run in this process on every visible card with no lock, so they "
         "would land on cards the pin excludes (set CUDA_VISIBLE_DEVICES to "
         "the cards this ensemble may use)"),
        ("--supervisor-max-restarts", getattr(args, "supervisor_max_restarts", 3) != 3,
         "there is no supervised worker to restart, so a member that fails "
         "ends the run whatever the count"),
        ("--prep-timeout", getattr(args, "prep_timeout", None) is not None,
         "no supervisor watches a preparation heartbeat on this route, so "
         "the timeout would never fire"),
        ("--allow-shared-gpu", bool(getattr(args, "allow_shared_gpu", False)),
         "no exclusive-card check runs on this route for it to relax"),
        ("--health-debug", bool(getattr(args, "health_debug", False)),
         "the phase health hooks are not handed to the member forecasts, so "
         "the run would report no attribution as if none were needed"),
        ("--preprocess-backend", getattr(args, "preprocess_backend", None) is not None,
         "it pins a [case_data] config's preparation, and each member is "
         "prepared by its source's own chain, which does not read it"),
        ("--directory-input-hash", getattr(args, "directory_input_hash", None) is not None,
         "it sets how a supervised run binds its declared directory inputs, "
         "and this route starts no supervised run to bind them"),
    ) if stated]
    if given:
        raise RecipeRefusal(
            "An ensemble recipe under woof run runs in this process, unsupervised "
            "and with no checkpoint to resume, and does not use "
            + ", ".join(flag for flag, _ in given) + ": "
            + "; ".join(f"{flag}: {why}" for flag, why in given)
            + ". Next: omit " + ("it" if len(given) == 1 else "them") + ".")


def load_trajectories(path):
    """``--trajectories FILE``: a JSON or TOML list of ``{source, cycle[, member]}``.

    The reader is :func:`woof.ensemble.recipes.load_trajectories`, which
    the planning door shares; this door's refusals stay its own class.
    """
    from woof.ensemble.recipes import load_trajectories as load

    return load(path, refusal=RecipeRefusal)


def flag_overrides(args) -> dict:
    """The ``--recipe`` and ``--trajectories`` flags as request overrides."""
    listed = getattr(args, "trajectories", None)
    return {"recipe": getattr(args, "recipe", None),
            "family_weights": getattr(args, "family_weights", None),
            "trajectories": None if listed is None else load_trajectories(listed)}


def _utc(value):
    from woof.ensemble.recipes import trajectory_time

    return trajectory_time(value, refusal=RecipeRefusal)


def plan_recipe(request, payload, experiment, *, cycle=None):
    """The member plan for this config, without downloading or touching a card.

    ``cycle`` asks for the same plan at another base cycle, the window
    moved with it and nothing written: ``--cycle`` with ``--readiness``
    answers for that cycle's member windows without re-timing the config.

    A plain member count (no recipe named) whose plan is refused leads with
    what the request would otherwise have been, N copies of one forecast
    (:func:`woof.ensemble.member_inputs.planned_refusal`).  Only the
    plan's own refusals take that lead.  The gates the door asks after the
    plan (the card, the memory envelope, the geography tree, the renderer,
    the disk) are not about where the members come from, and a card
    refusal that opened with "N copies of one forecast" right under a
    printed member plan said the opposite of what had happened.
    """
    try:
        return _member_plan(request, payload, experiment, cycle)
    except RecipeRefusal as refusal:
        from woof.ensemble.member_inputs import needs_member_sources, planned_refusal
        if not needs_member_sources(request):
            raise
        raise RecipeRefusal(planned_refusal(request, refusal)) from refusal


def _member_plan(request, payload, experiment, cycle):
    from woof.ensemble.recipes import SourceTrajectory, build_recipe

    fetch = payload.get("fetch")
    if not isinstance(fetch, dict) or not {"source", "cycle"} <= fetch.keys():
        # Breakage it prevents: a [case_data] config names the local files
        # of ONE trajectory, so every member would run the same inputs and
        # the ensemble would report spread it does not have.
        raise RecipeRefusal(
            "an ensemble recipe fetches and prepares each member's own source "
            "trajectory, and this config has no [fetch] source and cycle. "
            "Next: woof domain --help")
    start = _utc(experiment.start_time)
    end = start + timedelta(seconds=float(experiment.run_seconds))
    lead = fetch.get("forecast_start_hour", 0)
    base_cycle = _utc(str(fetch["cycle"]))
    if base_cycle + timedelta(hours=int(lead)) != start:
        raise RecipeRefusal(
            "[fetch].cycle plus forecast_start_hour is not [experiment].start_time, "
            "so the base member's own window is ambiguous. Next: regenerate the "
            "config with woof domain")
    if cycle is not None:
        moved = _utc(cycle)
        start, end, base_cycle = start + (moved - base_cycle), end + (moved - base_cycle), moved
    # The config's own source member is the base trajectory's, and a
    # time-lagged roster keeps it at every lagged cycle.  A multi-model
    # roster runs the listed trajectories, each naming its own member.
    member = fetch.get("member") if request.recipe in ("time-lagged", "surface-state", "member-roster") else None
    try:
        trajectories = tuple(SourceTrajectory(str(item["source"]), _utc(item["cycle"]), item.get("member"))
                             for item in request.trajectories)
        # A plain member count (no recipe named) is the planner's automatic
        # choice: the operational ensemble the source's adapter row declares.
        variants = tuple(getattr(request, "member_variants", ()) or ())
        recipe = build_recipe(source=str(fetch["source"]), cycle=base_cycle, start=start, end=end,
                            count=request.members, base_seed=request.base_seed,
                            kind=request.recipe or "auto", trajectories=trajectories,
                             member=None if member is None else str(member),
                             perturbation=request.perturbation, member_variants=variants)
        from dataclasses import replace
        return replace(recipe, selection_roster=tuple(request.trajectories))
    except RecipeRefusal:
        raise
    except ValueError as error:
        if request.recipe is None:
            from woof.ensemble.member_inputs import no_automatic_members
            raise RecipeRefusal(no_automatic_members(
                str(fetch["source"]), request.members, error)) from error
        raise RecipeRefusal(str(error)) from error


def _crop_box(raw, experiment, source):
    """The domain's own crop box for ``source``, or None when its fetch takes none.

    A member from another model does not inherit the config's ``area``:
    that box was bounded by the config's own source.  It is computed from
    the domain the way ``woof domain`` computes it, bounded by this
    source's coverage, so it exists whether or not the config's own
    ``[fetch]`` table carries one.
    """
    from woof import domain_wizard as wizard

    if not wizard.source_fetch_takes_a_crop_box(source):
        return None
    root = experiment.root.run
    return wizard.fetch_area_hint(raw["projection"], root.nx, root.ny,
                                  source=source, root_dx_m=root.dx)


def member_fetch(raw, experiment, recipe, member) -> dict:
    """The ``[fetch]`` table that acquires one member's trajectory.

    A member of the config's own source keeps the config's table and
    changes its cycle, its start lead and its source member.  A member
    from another model gets that model's own table: its forcing cadence
    and, where its fetch crops at the publisher, the domain's crop box.
    The config's product and provider hints do not carry over to another
    model: that model's own fetch route answers those.
    """
    from woof.source_adapters import get_source_adapter

    authored = (raw.get("ensemble") or {}).get("member_fetches")
    if authored:
        return dict(authored[member.index])
    trajectory = member.trajectory
    first, last = recipe.acquisition_window(trajectory)
    stamp = trajectory.cycle.strftime("%Y-%m-%dT%H")
    if trajectory.source == recipe.base.source:
        fetch = dict(raw["fetch"])
        fetch["cycle"] = stamp
        # Authored forcing can bracket a shorter forecast on its native clock.
        # Keep a covering cycle-start window instead of shrinking it off clock.
        if not (first == 0 and not fetch.get("forecast_start_hour", 0)
                and int(fetch.get("hours", 0)) >= last):
            fetch["hours"] = last - first
        fetch.pop("forecast_start_hour", None)
        if first:
            fetch["forecast_start_hour"] = first
    else:
        cadence = get_source_adapter(trajectory.source).forcing_interval_seconds
        fetch = {"source": trajectory.source, "cycle": stamp, "hours": last - first,
                 **({} if cadence is None or get_source_adapter(trajectory.source).acquisition_owner
                    else {"cadence": max(1, int(cadence) // 3600)}),
                 **({"forecast_start_hour": first} if first else {})}
        area = _crop_box(raw, experiment, trajectory.source)
        if area is not None:
            fetch["area"] = area
    fetch.pop("member", None)
    if trajectory.member is not None:
        fetch["member"] = trajectory.member
    selected = recipe.selection_roster[member.index] if recipe.selection_roster else {}
    if selected.get("fetch_root"):
        fetch["reuse_fetch"] = str(Path(selected["fetch_root"]).expanduser().resolve())
    return fetch


def member_label(member) -> str:
    trajectory = member.trajectory
    return (f"member {member.index} ({trajectory.source} {trajectory.cycle:%Y-%m-%dT%H}Z"
            + ("" if trajectory.member is None else f" member {trajectory.member}") + ")")


def member_configuration(raw, config, wps, recipe, member, directory):
    """Write the config and WPS namelist that prepare one member's trajectory.

    The domain and clock remain the original document. Source recipes
    change only the ``[fetch]`` table (:func:`member_fetch`): it names this
    member's source and cycle and the lead at which that cycle reaches the
    shared start time.  A member from another model takes that model's own
    forcing cadence, and its WPS namelist is rewritten for it by the
    ordinary writer. A named member roster also sets that member's existing
    land selectors before the ordinary source preparation reads them.
    """
    from woof.experiment import load_experiment
    from woof.toml_document import emit_experiment_toml

    trajectory = member.trajectory
    selected = recipe.selection_roster[member.index] if recipe.selection_roster else {}
    if selected.get("tracks") and not selected.get("configuration"):
        raise RecipeRefusal("track-selected trajectory needs its own cyclone-setup configuration; "
                            "using the shared config would seed every moving nest on the same track")
    if selected.get("configuration"):
        import tomllib
        chosen = Path(selected["configuration"])
        if not chosen.is_file() or hashlib.sha256(chosen.read_bytes()).hexdigest() != selected.get("configuration_sha256"):
            raise RecipeRefusal("member configuration differs from the selected cyclone-setup bytes")
        config = chosen
        raw = tomllib.loads(chosen.read_text(encoding="utf-8"))
    original = load_experiment(config)
    if selected.get("configuration"):
        from woof.ensemble.recipes import SourceTrajectory
        fetch = raw.get("fetch", {})
        actual = SourceTrajectory(fetch["source"], _utc(fetch["cycle"]), fetch.get("member"))
        if actual.identity != trajectory.identity:
            raise RecipeRefusal("member configuration belongs to another source trajectory")
        from datetime import timezone
        if (original.start_time.replace(tzinfo=timezone.utc) != recipe.start
                or original.run_seconds != (recipe.end-recipe.start).total_seconds()):
            raise RecipeRefusal("member configuration belongs to another forecast window")
    changed = deepcopy(raw)
    if recipe.member_variants:
        from woof.ensemble.member_variants import variant_configuration
        changed = variant_configuration(changed, recipe.member_variants[member.index])
    changed.pop("ensemble", None)
    changed["fetch"] = member_fetch(raw, original, recipe, member)
    from woof.ensemble.wofs_recipe import per_member_route, selection_owner
    from woof.source_adapters import get_source_adapter
    adapter = get_source_adapter(trajectory.source)
    owner = selection_owner(adapter)
    if (not (raw.get("ensemble") or {}).get("member_fetches")
            and owner is not None and hasattr(owner, "batch_fetch_hints")
            and trajectory.member is not None):
        # Explicit source recipes share the same batch contract as the WoFS
        # ladder. Bind the complete family/cycle roster before the first fetch.
        peers = [peer for peer in recipe.members if (
            peer.trajectory.source == trajectory.source
            and peer.trajectory.cycle == trajectory.cycle)]
        family_key = hashlib.sha256(trajectory.source.encode()).hexdigest()[:12]
        root = (Path(config).resolve().parent / str(raw["fetch"].get("out", "data")) /
                family_key / trajectory.cycle.strftime("%Y%m%dT%H%M%S")).resolve()
        fetches = [owner.member_fetch_hints(member_fetch(raw, original, recipe, peer),
            peer.trajectory.member, window_seconds=(recipe.end - trajectory.cycle).total_seconds(),
            root=root) for peer in peers]
        bound = owner.batch_fetch_hints(fetches)
        if len(bound) != len(peers):
            raise RecipeRefusal("selection owner changed the planned member count")
        changed["fetch"] = bound[peers.index(member)]
    if per_member_route(trajectory.source) and not adapter.ensemble_preparation_owner:
        changed.pop("case_data", None)
    static = changed.get("static")
    if isinstance(static, dict) and isinstance(static.get("highres"), dict):
        # Only a declared [static.highres] has a cache_root to pin: a
        # [static] that names only a source parses to a disabled carrier
        # holding the default root, and writing that root alone made a
        # [static.highres] with no `enabled`, which refused every member
        # of a base that names a static source (what `woof domain`
        # emits for a source whose metadata declares one).
        from woof.static.highres_production import parse_static_table
        carrier = parse_static_table(changed["static"], source=str(config),
                                     base_dir=Path(config).resolve().parent)
        if carrier is not None and getattr(carrier, "cache_root", None) is not None:
            changed.setdefault("static", {}).setdefault("highres", {})["cache_root"] = str(
                carrier.cache_root.resolve())
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "experiment.toml"
    if adapter.ensemble_preparation_owner:
        from importlib import import_module
        import_module(adapter.ensemble_preparation_owner).configure_member(
            changed, target=target, original=Path(config))
    target.write_text(emit_experiment_toml(changed), encoding="utf-8")
    experiment = load_experiment(target)
    namelist = directory / "experiment.namelist.wps"
    if selected.get("configuration"):
        from woof.companion_domains import configuration_wps_namelist
        namelist = configuration_wps_namelist(target, experiment, raw=changed, into=directory)
    elif trajectory.source == recipe.base.source:
        namelist.write_bytes(Path(wps).read_bytes())
    else:
        from woof.companion_domains import candidate_wps_text
        namelist.write_text(candidate_wps_text(changed, original, experiment, target,
                                               original_wps=Path(wps)), encoding="utf-8")
    return target, namelist, experiment


@dataclass(frozen=True)
class MemberPlan:
    """One member, reviewed: its trajectory, its fetch table and its chain."""

    member: object
    fetch: dict
    chain: str
    start_lead: int
    experiment: object
    #: The reviewed copy of the member's config, in the review's scratch
    #: folder: there while the gates are asked, gone before the run.
    config: Path
    preparation_key: str = ""
    variant_name: str | None = None
    memory_key: str = ""
    source_key: str = ""
    physics_tail: str = ""

    @property
    def line(self) -> str:
        trajectory = self.member.trajectory
        return (f"member {self.member.index}"
                + ("" if self.variant_name is None else f" ({self.variant_name})")
                + f": {trajectory.source} "
                f"{trajectory.cycle:%Y-%m-%dT%H}Z"
                + ("" if trajectory.member is None else f" member {trajectory.member}"))


def _posting(options) -> dict:
    """The host pin and posting rule a door carries into every member's fetch."""
    return {key: options[key] for key in ("transport", "as_posted", "late_after_minutes")
            if options.get(key) is not None}


def review_members(raw, config, wps, recipe, *, scratch, options=None):
    """Build every member's config and ask its chain's own plan, before any fetch.

    Breakage it prevents: a member its chain cannot prepare (a table its
    fetch refuses, a host its source cannot pin, a domain outside its
    grid, a chain plan with a missing key) was found only when the loop
    reached it, after every earlier member had been downloaded and
    prepared.  The configs are written to ``scratch``, a folder outside
    the output folder that the caller discards; the run writes them again
    into its own folder.
    """
    import tomllib

    from woof import regional_preparation, runplan
    from woof.go_cli import pinned_transport
    from woof.source_coverage import config_source_coverage_refusal

    options = options or {}
    posting = _posting(options)
    chains = regional_preparation.preparation_chains()
    scratch = Path(scratch)
    plans = []
    for member in recipe.members:
        source = member.trajectory.source
        try:
            target, _namelist, experiment = member_configuration(
                raw, config, wps, recipe, member, scratch / f"member-{member.index:03d}")
            document = tomllib.loads(target.read_text(encoding="utf-8"))
            table = document["fetch"]
            chain = regional_preparation.member_preparation_chain(source)
            if chain not in chains:
                raise RecipeRefusal(f"its source's route ({chain}) has no prepare-only chain")
            pinned_transport(table, posting.get("transport"))
            uncovered = config_source_coverage_refusal(experiment, source)
            if uncovered is not None:
                raise RecipeRefusal(uncovered)
            # The chain's own review of the config, read from its row: for
            # the native chain its planner (the keys its stages need, the
            # preparation preconditions and the profile), for the hourly
            # chain the namelists it runs from.
            review = regional_preparation.preparation_chain_reviews().get(chain)
            if review is not None:
                review(target, experiment, raw=document, scratch=scratch, posting=posting)
        except ValueError as error:
            raise RecipeRefusal(
                f"{member_label(member)} cannot be prepared: {error}") from error
        from woof.ensemble.member_variants import preparation_binding_key, preparation_binding_parts
        binding = preparation_binding_parts(member.trajectory.identity, document, experiment=experiment)
        variant_name = None
        if recipe.member_variants:
            variant = recipe.member_variants[member.index]
            variant_name = variant["name"]
        plans.append(MemberPlan(member, table, chain,
                                 recipe.acquisition_window(member.trajectory)[0],
                                 experiment, target, binding.key, variant_name,
                                 preparation_binding_key(source,
                                     {key: value for key, value in document.items() if key != "fetch"},
                                     experiment=experiment), binding.source_key, binding.physics_tail))
    if recipe.member_variants:
        from woof.ensemble.member_variants import require_distinct_variants
        try:
            require_distinct_variants(raw, recipe.member_variants,
                                     experiments=[plan.experiment for plan in plans],
                                     trajectories=tuple(member.trajectory for member in recipe.members))
        except ValueError as error:
            raise RecipeRefusal(str(error)) from error
    return tuple(plans)


class _MemberPreparationLedger:
    """One member's preparation spans in the ledger its siblings share.

    Members prepare concurrently in one process, so their same-named spans
    (fetch.inputs, prep.inputs, post.orchestration) pair by ``span_id``, the
    way :func:`woof.ensemble.timeline.member_timeline` tags each member's
    forecast steps.
    """

    def __init__(self, ledger, member_id):
        self._ledger, self._member_id = ledger, member_id

    def emit(self, stage, phase="mark", *, attrs=None, **clocks):
        return self._ledger.emit(stage, phase, attrs={
            **(attrs or {}), "member_id": self._member_id,
            "span_id": f"ensemble-prepare-{self._member_id}"}, **clocks)


def _fetch_seconds(directories, start_unix, end_unix):
    """Wall seconds of [start, end] in which at least one member was fetching.

    Read from each member preparation's own stage events, so the ensemble's
    fetch and prepare split is measured, not apportioned.
    """
    from woof.timeline.report import union_seconds

    spans = []
    for directory in directories:
        try:
            lines = (Path(directory) / "prepare-events.jsonl").read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        opened = None
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("stage") != "fetch" or not isinstance(event.get("emitted_unix_ms"), (int, float)):
                continue
            at = event["emitted_unix_ms"] / 1000.0
            if event.get("event") == "stage_started":
                opened = at
            elif event.get("event") == "stage_finished" and opened is not None:
                begin, end = max(opened, start_unix), min(at, end_unix)
                if end > begin:
                    spans.append((begin, end))
                opened = None
    return union_seconds(spans)


def member_download_request(config, experiment, options=None):
    """The request a member's chain fetch makes, and its cache key.

    Keyed on the request the chain's fetch stage makes: the model top this
    config's ladder needs and the host the door pins.
    """
    import tomllib

    from woof.go_cli import fetch_request, managed_download_key, pin_request, pinned_transport

    posting = _posting(options or {})
    hints = tomllib.loads(Path(config).read_text(encoding="utf-8"))["fetch"]
    pinned, _basis = pinned_transport(hints, posting.get("transport"))
    request = pin_request(fetch_request(hints, p_top=experiment.vertical.p_top), pinned)
    return request, managed_download_key(request)


class _SharedDownloads:
    """One download per shared source folder; its other members wait for it.

    Members whose preparations make the same fetch request (one earlier
    HRRR cycle under several physics tails, say) share one download
    folder.  The first preparation of a folder is its downloader; every
    other one waits here, before it takes a preparation slot, until the
    downloader has left its download (its first stage after the fetch, or
    the end of its preparation, whichever is first; :class:`_MemberStages`),
    and then runs its own chain, whose fetch finds the folder complete and
    only verifies it.  The concrete breakage it prevents: a
    36-member HRRR ensemble (2026-10-24) ran every such fetch at once; the
    losers sat in their CPU slots on the folder's writer lock, and member
    4 refused after 600 s while its sibling was still downloading.
    """

    def __init__(self):
        import threading
        self._guard = threading.Lock()
        self._folders = {}

    def claim(self, key, member_id, folder):
        """``(leader, row)`` for this request key; ``folder()`` names it once."""
        import threading
        with self._guard:
            row = self._folders.get(key)
            if row is not None:
                return False, row
            row = {"folder": folder(), "member_id": member_id, "fetched": threading.Event()}
            self._folders[key] = row
            return True, row


def prepare_member(config, experiment, *, source, directory, downloads, geog_root,
                   recipe_sha256, options=None, composition_bank=None,
                   composition_source_key=None, member_id=None, download_dir=None,
                   stage_gate=None):
    """Prepare every domain from one trajectory through its ordinary source chain.

    The chain consumes this member's complete experiment, including its
    nests.  Its prepare-only result names the sealed tree when there is
    more than one domain, so no child can inherit another member's inputs.
    ``download_dir`` is the folder the door chose for this request (see
    :class:`_SharedDownloads`); ``stage_gate(event, fields)`` sees every
    event before it is written and may block there (:class:`_MemberStages`
    holds a stage until its admission lets it start).
    """
    from woof import regional_preparation, runplan
    from woof.go_cli import managed_download_dir

    posting = _posting(options or {})
    chain = regional_preparation.member_preparation_chain(source)
    owner = regional_preparation.preparation_chains()[chain]
    if download_dir is None:
        request, _key = member_download_request(config, experiment, options)
        download_dir = managed_download_dir(downloads, request)
    run_options = {"geog_root": str(geog_root), "supplement": [],
                   "data_dir": str(download_dir), **posting}
    plan = SimpleNamespace(run_options=run_options, config_intent={}, sha256=recipe_sha256)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    from contextlib import nullcontext
    from woof.mapped_composition import shared_composition_source
    scope = (nullcontext() if composition_source_key is None else
             shared_composition_source(composition_source_key, composition_bank))
    canaries = (options or {}).get("_head_canaries")
    check, stop_watch = ((lambda: None), (lambda: None))
    if canaries is not None:
        check, stop_watch = canaries.watch(directory, source=source, config=config,
                                          namelist=(options or {}).get("_canary_namelist"))
    class _CanaryEvents(runplan.EventStream):
        def emit(self, event, **fields):
            # A deterministic canary refusal stops the producer at its next
            # progress event instead of paying for every remaining boundary.
            check()
            if stage_gate is not None:
                stage_gate(event, fields)
            return super().emit(event, **fields)
    events = _CanaryEvents(directory / "prepare-events.jsonl", mirror=None)
    observer = runplan.RunObserver(events)
    if member_id is not None:
        observer._timeline = _MemberPreparationLedger(observer._timeline, member_id)
    try:
        with scope:
            result = owner(plan, config_path=Path(config), exp=experiment,
                           observer=observer, run_dir=directory / "prepare",
                           prepare_only=True)
        # This member's timeline ends with its preparation.  Left to the
        # command's exit, its idle post.orchestration span ran through the
        # whole ensemble forecast and its products (450 s of a 6-member
        # HRRR run, 2026-10-20).
        observer.close()
    finally:
        stop_watch()
        events.close()
    if canaries is not None:
        check()
        # Sources that publish only a seal retain the same launch check.
        canaries.submit({"source": source, **{key: result[key] for key in
                         ("prepared_root", "experiment_config", "wps_namelist")}})
    return {key: result[key] for key in ("prepared_root", "experiment_config", "wps_namelist")}


def member_inputs(shared_inputs, prepared):
    """The ordinary preflight, applied to another member's prepared bundle."""
    from woof import stage_cli

    root = Path(prepared["prepared_root"]).resolve()
    bundle = stage_cli.resolve_bundle(root)
    arguments = dict(shared_inputs.preflight_arguments)
    config = Path(prepared["experiment_config"])
    if bundle.get("layout") == "tree":
        from woof.prepared_domain_tree_forecast import preflight_prepared_tree

        digests = stage_cli.tree_digests(bundle, config)
        arguments.update(prepared_root=root, experiment_config=config,
                         preparation_receipt_sha256=digests["preparation_receipt"],
                         experiment_config_sha256=digests["experiment_config"])
        # Recipe preparation is complete before forecasting.  A previous
        # member's streamed head must not bind this member's sealed tree.
        arguments.pop("prepared_head_sha256", None)
        return preflight_prepared_tree(**arguments)

    from woof.prepared_single_domain_forecast import preflight_prepared_forecast

    digests = stage_cli.single_domain_digests(bundle)
    arguments.update(source=bundle["source"], prepared_root=root,
                     proof_sha256=digests["proof"],
                     source_manifest_sha256=digests["source_manifest"],
                     prepared_content_sha256=digests["prepared_content"],
                     experiment_config=config,
                     wps_namelist=Path(prepared["wps_namelist"]))
    return preflight_prepared_forecast(**arguments)


def _say(text):
    print(f"ensemble: {text}", flush=True)


def disk_refusal(plans, *, case_root, request, options=None) -> str | None:
    """Why the disk cannot hold the ensemble's distinct source preparations.

    Breakage it prevents: distinct source members download their own
    windows and write their own bundles, so a disk with room for one run
    fills partway through the members. Each distinct trajectory is priced
    once from the single-run admission tables; seeded surface members
    share unchanged preparation. Matching cached downloads are subtracted.
    Member history is charged only when the request keeps member files;
    the aggregate products have no measured size and are left out.
    """
    from woof import disk_budget, runplan
    from woof.go_cli import fetch_request, managed_download_dir, pin_request, pinned_transport

    posting = _posting(options or {})
    rows, basis, distinct, preparation_banks, source_tails, source_streams = [], [], {}, {}, {}, {}
    for plan in plans:
        pinned, _basis = pinned_transport(plan.fetch, posting.get("transport"))
        fetch = pin_request(fetch_request(plan.fetch, p_top=plan.experiment.vertical.p_top), pinned)
        try:
            folder = managed_download_dir(case_root, fetch)
        except (OSError, ValueError):
            folder = None
        row = disk_budget.projected_run_bytes(
            plan.experiment, keep_checkpoints=None, fetch=fetch, chain=plan.chain,
            render=False, render_products="none",
            download_present_bytes=runplan._present_download_bytes(fetch, folder, True))
        rows.append(row)
        distinct.setdefault(plan.member.trajectory.identity, row)
        preparation_banks.setdefault(plan.preparation_key or plan.member.trajectory.identity, row)
        source_key = getattr(plan, "source_key", "") or plan.member.trajectory.identity
        source_tails.setdefault(source_key, set()).add(plan.preparation_key or plan.member.trajectory.identity)
        source_streams[source_key] = max(source_streams.get(source_key, 0),
                                       int(row.get("compose_scratch_min_bytes") or 0))
        basis.append(f"member {plan.member.index}: {row['download']['basis']}")
    stream = max(rows, key=lambda row: int(row.get("compose_scratch_min_bytes") or 0))
    members = len(rows)
    # The default-on hydro preparation (shared by every member) is priced
    # here and built only after this admission passes (woof.hydro.runtime
    # ADMISSION_PENDING), so a refusal spends nothing.
    from woof.hydro.runtime import hydro_disk_estimate
    hydro_config = getattr(plans[0], "config", None) if plans else None
    hydro_bytes, hydro_basis = hydro_disk_estimate(hydro_config) if hydro_config else (0, None)
    if hydro_basis:
        basis.append(hydro_basis)
    projection = {
        "download_bytes": sum(int(row["download_bytes"]) for row in distinct.values()),
        # Shared frame streams sit beside output, even if the caller's
        # ordinary scratch override is on another disk. Conservatively price
        # every multi-tail source, including streams released early below.
        "preparation_bytes": (sum(int(row["preparation_bytes"]) for row in preparation_banks.values()) +
                              sum(size for source, size in source_streams.items() if len(source_tails[source]) > 1)
                              + hydro_bytes),
        "history_bytes": (sum(int(row["history_bytes"]) for row in rows)
                          if request.keep_member_files else 0),
        "checkpoint_bytes": 0, "picture_bytes": 0,
        # Ordinary transient decode scratch is still one stream. Retained
        # source banks on output are accounted for in preparation_bytes.
        "compose_scratch_bytes": int(stream.get("compose_scratch_bytes") or 0),
        "compose_scratch_min_bytes": int(stream.get("compose_scratch_min_bytes") or 0),
        "compose_scratch": stream.get("compose_scratch") or {},
    }
    if (projection["compose_scratch"] or {}).get("composes"):
        from woof.ingest.source_coverage import compose_scratch_override_refusal
        absent = compose_scratch_override_refusal()
        if absent is not None:
            return absent + ". Refused before any download or preparation, so nothing was spent."
    scratch_folder = runplan._compose_scratch_folder(
        runplan._staged_prep_root(Path(case_root)), projection)
    scratch_free = (None if scratch_folder is None
                    or disk_budget.same_disk(scratch_folder, Path(case_root))
                    else disk_budget.free_bytes(scratch_folder))
    refusal = disk_budget.disk_refusal(
        projection, disk_budget.free_bytes(Path(case_root)), scratch_free=scratch_free,
        scratch_folder=scratch_folder, subject=f"this ensemble's {members} members",
        remedy="Free some disk, run fewer members, or name an --outdir on a disk with room")
    if refusal is None:
        return None
    return (refusal[0].upper() + refusal[1:] + ". Refused before the first download and before "
            "the hydro preparation, so nothing was spent. " + "; ".join(basis))


def admit(plans, *, config, geog_root, case_root, request, options=None,
          command="ensemble") -> None:
    """The gates the ordinary door asks before its fetch, asked before N fetches.

    Breakage each prevents is the ordinary door's own (see
    :func:`woof.go_cli._require_forecast_device`,
    :func:`woof.go_cli.memory_gate`,
    :func:`woof.go_cli.geography_refusal`): a card that cannot be opened,
    a config too large for it, a missing geography tree, a missing
    renderer or an unknown product was found on this door only after
    every member had been downloaded and prepared.
    """
    import sys

    from woof import capabilities, go_cli, rustwx

    options = options or {}
    def admit_geography():
        # Each member can select a different static source or land layout.
        # The generic installation inventory would demand unused datasets
        # and disagree with fetch-geog --config on complete published statics.
        configs = tuple(dict.fromkeys(getattr(plan, "config", config) for plan in plans)) or (config,)
        for member_config in configs:
            geography = go_cli.geography_refusal(geog_root, config=member_config)
            if geography is not None:
                raise RecipeRefusal(geography)
    if options.get("prepare_only"):
        # Preparation needs geography and disk space, but no forecast card
        # or renderer. Forecast admission still runs on the launch route.
        admit_geography()
        refusal = disk_refusal(plans, case_root=case_root, request=request, options=options)
        if refusal is not None:
            raise RecipeRefusal(refusal)
        return
    capabilities.require(
        f"woof {command}", capabilities.GPU_RUNTIME,
        before=("Refusing here, before the first member's fetch downloads its "
                "forcing data, rather than at the forecast stage after every "
                "member is prepared."))
    try:
        go_cli._require_forecast_device()
        if not options.get("no_memory_gate"):
            # Once per prepared authority: surface arms share a bank,
            # while different soil layouts retain their own inventory.
            seen = set()
            for plan in plans:
                source = plan.member.trajectory.source
                binding = plan.memory_key or source
                if binding in seen:
                    continue
                seen.add(binding)
                gate = go_cli.memory_gate(
                    {"config": plan.config, "source": source,
                     "cadence": plan.fetch.get("cadence")}, experiment=plan.experiment)
                _say(f"memory ({source}) -- {gate['verdict']}")
                if gate["refuse"]:
                    raise RecipeRefusal(go_cli.memory_refusal_text(gate))
                if gate["warn"]:
                    print("warning: memory admission has limited headroom or an "
                          f"unmeasured phase; see woof check {config} --explain.",
                          file=sys.stderr, flush=True)
                if gate.get("preparation_warning"):
                    print(f"warning: {gate['preparation_warning']}.",
                          file=sys.stderr, flush=True)
        admit_geography()
        try:
            renderer = rustwx.find_renderer()
        except FileNotFoundError as error:
            raise RecipeRefusal(str(error)) from None
        if renderer is None:
            # The ensemble session draws its aggregate maps through the
            # Rust renderer whatever --products says, and refused its
            # absence only when the forecast started.
            raise RecipeRefusal(
                "the ensemble's aggregate maps are drawn by the Rust renderer, and "
                "this install has none.\n" + rustwx.renderer_remedy())
        go_cli.admit_render_products(options.get("render_products"), section=None)
    except go_cli.GoRefusal as refusal:
        raise RecipeRefusal(str(refusal)) from None
    refusal = disk_refusal(plans, case_root=case_root, request=request, options=options)
    if refusal is not None:
        raise RecipeRefusal(refusal)


def _exit_code(error) -> int | None:
    """The exit code a stage's own failure is worth, or None for any other error.

    75 for a source lead later than its budget, 130 for a stop, and the
    stage's own code otherwise: the codes the ordinary door returns, so a
    scheduler reads a recipe launch the way it reads any other.
    """
    if isinstance(error, KeyboardInterrupt):
        return 130
    if isinstance(error, SystemExit):
        # A stop, as run control treats it (a SIGTERM handler raising
        # SystemExit(128 + N), offline_child_run.ChildStopped): the stop's
        # own code, and 130 when it carries none.
        code = error.code
        return code if isinstance(code, int) and not isinstance(code, bool) and code else 130
    if not isinstance(error, Exception):
        return None
    from woof.go_cli import GoStageFailed

    # ``exit_code`` is how a stage failure states its own code
    # (StageExitError, GoInterrupted, ChainInterrupted, SourceBehind);
    # GoStageFailed spells it ``code``.  Any other ``code`` attribute (an
    # HTTP status) is not a stage's exit.
    code = error.code if isinstance(error, GoStageFailed) else getattr(error, "exit_code", None)
    if isinstance(code, bool) or not isinstance(code, int):
        return None
    if code < 0:
        import signal
        return 130 if -code == int(signal.SIGINT) else 1
    return code or 1


def _member_ids(rows):
    """Every member the failing batches of a forecast name, in their order."""
    ids = []
    for row in rows:
        for member in row.get("member_ids") or ():
            if member not in ids:
                ids.append(member)
    return ids


def _forecast_record(forecast_dir):
    """The session's own run record, when the runner returned a code instead of raising.

    ``run/ensemble-run.json`` holds the real error class, its text and the
    failing members; an unreadable or absent record (a runner that refused
    before the session opened) is an empty one.
    """
    try:
        record = json.loads((Path(forecast_dir) / "ensemble-run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(record, dict) or record.get("status") not in ("failed", "interrupted"):
        return {}
    return record


def _notify(observer, event, **fields):
    from woof.go_cli import _notify as notify
    notify(observer, event, **fields)


def _preparation_context_sha256(wps, geog_root, options):
    """Bind relaunch reuse to the same preparation code, geography and fetch policy."""
    repo = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    roots = [repo / "woof" / name for name in
             ("ingest", "static", "mapped_direct.py", "regional_preparation.py",
              "go_cli.py", "runplan.py", "experiment.py", "companion_domains.py", "bridges.py")]
    roots += [repo / "tools" / name for name in
              ("grib1_bridge", "rw_wps", "region_global_dealias", "zarr_bridge")]
    for root in roots:
        files = []
        if root.is_dir():
            for directory, folders, names in os.walk(root):
                folders[:] = sorted(name for name in folders
                                    if name not in ("target", "vendor", "__pycache__", ".git"))
                files.extend(Path(directory) / name for name in names)
        else:
            files = [root]
        for path in sorted(files):
            relative = path.relative_to(repo)
            if (not path.is_file() or any(part in ("target", "vendor", "__pycache__", ".git")
                                         for part in relative.parts)
                    or path.suffix not in (".py", ".rs", ".toml", ".lock", ".h", ".c")):
                continue
            digest.update(relative.as_posix().encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    digest.update(Path(wps).read_bytes())
    digest.update(json.dumps({"geog_root": str(geog_root), "posting": _posting(options)},
                             sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _relaunch_preparations(case_root, root, recipe, context):
    """Read ready members from this case's failed forecast, without mutating it.

    The ordinary forecast preflight still checks each reused bundle's full
    authority and content hashes. This receipt never bypasses those checks.
    """
    from woof import stage_cli
    candidates = [Path(root)] + sorted(Path(case_root).glob("run-*"), reverse=True)
    for directory in dict.fromkeys(candidates):
        try:
            prior = json.loads((directory / RECEIPT_NAME).read_text(encoding="utf-8"))
            if (prior.get("schema") != RECEIPT_SCHEMA or prior.get("recipe_sha256") != recipe.sha256
                    or prior.get("preparation_context_sha256") != context
                    or prior.get("preparation_status") != "ready"
                    or not (prior.get("status") == "prepared" or
                            prior.get("failure", {}).get("stage") == "forecast")):
                continue
            rows = prior["members"]
            if {row["member_id"] for row in rows} != {member.index for member in recipe.members}:
                continue
            reusable = {}
            for row in rows:
                bundle = {key: row[key] for key in ("prepared_root", "experiment_config", "wps_namelist")}
                stage_cli.resolve_bundle(Path(bundle["prepared_root"]))
                if not Path(bundle["experiment_config"]).is_file() or (
                        bundle["wps_namelist"] is not None and not Path(bundle["wps_namelist"]).is_file()):
                    raise ValueError("prepared member authorities are missing")
                reusable[row["preparation_binding_sha256"]] = (bundle, str(directory))
            return reusable
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return {}


@ledger_main
def run_recipe_ensemble(args, request, *, observer=None, options=None) -> int:
    """Plan, review, prepare and run a time-lagged or multi-model ensemble.

    ``options`` is what the calling door carries into every member's
    fetch: ``transport``, ``as_posted`` and ``late_after_minutes``.  The
    doors refuse, by name, the flags of theirs this door does not consume.
    """
    import sys
    import tempfile
    import time
    import tomllib

    from woof import run_stamp, stage_cli
    from woof.ensemble.door import production_run_scope
    from woof.experiment import load_experiment
    from woof.explain import explain_enabled
    from woof.geog_assets import default_geog_root
    from woof.stage_timing import LedgerObserver, bind_ledger

    door_started = time.monotonic()
    options = dict(options or {})
    options["prepare_only"] = bool(getattr(args, "prepare_only", False))
    options.setdefault("render_products", getattr(args, "render_products", None))
    options.setdefault("no_memory_gate", bool(getattr(args, "no_memory_gate", False)))
    command = getattr(args, "command", None) or "ensemble"
    config = Path(args.config).resolve()
    if not config.is_file():
        raise RecipeRefusal(f"{config} does not exist. Next: woof domain --help")
    raw = tomllib.loads(config.read_text(encoding="utf-8-sig"))
    experiment = load_experiment(config)
    recipe = plan_recipe(request, raw, experiment)
    # A selected trajectory roster requests the existing cyclone product path.
    # Retain the ordinary writer histories until maps and tracks consume them.
    from woof.ensemble.cyclone_products import roster_product_request
    cyclone_request = roster_product_request(request.trajectories,
        getattr(args, "cyclone_products", None))
    if cyclone_request is not None:
        from dataclasses import replace
        request = replace(request, cyclone_products=cyclone_request)
    named_wps = getattr(args, "wps_namelist", None)
    if named_wps is not None and not Path(named_wps).is_file():
        raise RecipeRefusal(
            f"--wps-namelist {named_wps} does not exist; each member is prepared "
            "from that WPS namelist. Next: name an existing file, or omit the flag")

    def companion_wps(into: Path) -> Path:
        # The namelist a door wrote beside the configuration, or the one
        # rendered from its own grid into ``into`` (a shipped recipe has
        # none beside it); the staged chain asks the same function.
        if named_wps is not None:
            return Path(named_wps)
        from woof.companion_domains import configuration_wps_namelist

        try:
            return configuration_wps_namelist(config, experiment, raw=raw, into=into)
        except ValueError as error:
            raise RecipeRefusal(str(error)) from None

    geog_root = getattr(args, "geog_root", None)
    geog_root = Path(geog_root).resolve() if geog_root is not None else default_geog_root()
    case_root = Path(args.outdir) if getattr(args, "outdir", None) is not None else (
        config.parent / f"{config.stem}-go")
    with tempfile.TemporaryDirectory(prefix="gpuwm-recipe-") as scratch:
        wps = companion_wps(Path(scratch) / "route-inputs")
        plans = review_members(raw, config, wps, recipe, scratch=scratch, options=options)
        _say(f"{recipe.kind} recipe, {len(recipe.members)} members, valid "
             f"{recipe.start:%Y-%m-%dT%H:%M}Z to {recipe.end:%Y-%m-%dT%H:%M}Z")
        for plan in plans:
            _say(plan.line)
            if recipe.selection_roster:
                row = recipe.selection_roster[plan.member.index]
                layout = "; ".join(f"d{d.grid_id:02d} {d.run.nx}x{d.run.ny}x{d.run.nz} dx={d.run.dx:g}m"
                                   for d in plan.experiment.domains)
                _say(f"  {layout}; weight={row.get('weight', 1/len(plans)):.12g}; "
                     f"stratum={row.get('stratum', 'unstratified')}")
                if row.get("fetch_root"):
                    _say(f"  forcing: {plan.fetch['hours']} h; verified reuse candidate {plan.fetch['reuse_fetch']}")
        if getattr(args, "dry_run", False):
            _say("dry run: nothing was fetched, prepared or run")
            return 0
        admit(plans, config=config, geog_root=geog_root, case_root=case_root,
              request=request, options=options, command=command)
        # Admitted: the hydro door now builds the covering fabric.
        from woof.hydro.runtime import rebind_launch
        rebind_launch(args, admitted=True)
    root = run_stamp.resolve(case_root, init=str(raw["fetch"]["cycle"]),
                             enabled=run_stamp.run_stamp_enabled(args), create=True)
    root = Path(root)
    # The run root's own timeline, bound once the root is claimed, as a
    # single go run binds its own: the ensemble's prepare and forecast
    # stages, every member's preparation and every member's forecast steps
    # land in <run>/timeline.jsonl.  Breakage it prevents: an ensemble wrote
    # no run-level timeline at all; its stage notices reached no ledger, and
    # its forecast appeared only as each member preparation's idle span.
    hosted_observer = observer
    observer = LedgerObserver(observer, bind_ledger(root, product="ensemble"), root,
                              relay_forecast=False)
    wps = companion_wps(root / "route-inputs")
    context = _preparation_context_sha256(wps, geog_root, options)
    reusable = _relaunch_preparations(case_root, root, recipe, context)
    receipt = {"schema": RECEIPT_SCHEMA, "status": "preparing", "config": str(config),
               "preparation_status": "preparing", "preparation_context_sha256": context,
               "recipe": recipe.describe(), "recipe_sha256": recipe.sha256,
               "request": request.receipt(), "members": []}

    def publish():
        temporary = root / (RECEIPT_NAME + ".pending")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
                             encoding="utf-8")
        temporary.replace(root / RECEIPT_NAME)

    #: Where the run is, for the failure record: the stage and its member.
    where = {"stage": "prepare", "member_id": None, "started": time.monotonic()}

    def forecast_members(rows):
        """The members a forecast failure belongs to: the error's own, else the provider's."""
        ids = _member_ids(rows)
        if not ids and where["member_id"] is not None:
            ids = [where["member_id"]]
        return ids

    def subject(ids=()):
        if where["stage"] == "prepare":
            return f"member {where['member_id']}'s preparation"
        if ids:
            from woof.ensemble.execution import member_label as members_label
            return f"the forecast of {members_label(ids)}"
        return "the forecast"

    def failed(error, code, *, stopped=False, error_class=None, message=None, rows=None):
        """Record the failure where a reader of the run folder looks, and say it once.

        A forecast failure names its members from what run control wrote
        into the error (or, for a returned code, into ``ensemble-run.json``):
        ``member_id`` for one member, ``member_ids`` and ``failed_members``
        for a pack or a wave.
        """
        receipt["status"] = "interrupted" if stopped or code == 130 else "failed"
        failure = {"stage": where["stage"], "member_id": where["member_id"],
                   "error_class": error_class if error is None else type(error).__name__,
                   "message": message if error is None else str(error),
                   "exit_code": code}
        if where["stage"] == "forecast":
            if rows is None:
                from woof.ensemble.execution import failed_member_rows
                rows = [] if error is None else failed_member_rows(error)
            ids = forecast_members(rows)
            failure["member_id"] = ids[0] if len(ids) == 1 else None
            failure["member_ids"] = ids
            if rows:
                failure["failed_members"] = rows
        receipt["failure"] = failure
        try:
            publish()
        except OSError:
            pass
        _notify(observer, "stage_end", label=where["stage"], exit_code=code, ok=False,
                elapsed_seconds=time.monotonic() - where["started"], progress=None)

    publish()
    prepared = {}
    lineage_fetches = {}
    prepared_banks = {}
    source_bindings = {}
    for plan in plans:
        source_bindings.setdefault(plan.source_key, set()).add(plan.physics_tail)
    # Single-tail sources need no retained stream. Each multi-tail source is
    # spent and removed immediately after its last distinct tail is sealed.
    composition_banks = {key: tempfile.TemporaryDirectory(prefix="source-bank-", dir=root)
                         for key, tails in source_bindings.items() if len(tails) > 1}
    # The caller may supply a qualified source-head diagnostic. Automatic
    # installation belongs to the preparation integrator: the delivered
    # fixed-clock canary cannot certify ordinary adaptive-clock requests.
    canaries = options.get("_head_canaries")
    try:
        from concurrent.futures import ThreadPoolExecutor
        from contextvars import copy_context
        try:
            cpu_workers = len(os.sched_getaffinity(0))
        except AttributeError:
            cpu_workers = os.cpu_count() or 1
        workers = max(1, min(len(plans), cpu_workers))
        # Concurrent members each own a decoded staging tree. Limit that
        # concurrency on scratch-constrained hosts without restricting the
        # native threads within an individual preparation.
        worker_cap = os.environ.get("WOOF_ENSEMBLE_PREPARATION_WORKERS")
        if worker_cap is not None:
            if not worker_cap.isdecimal() or int(worker_cap) < 1:
                raise ValueError("WOOF_ENSEMBLE_PREPARATION_WORKERS must be a positive integer")
            workers = min(workers, int(worker_cap))
        preparation_started = time.monotonic()
        preparation_started_unix = time.time()
        admission = _PreparationAdmission(workers)
        download_admission = _DownloadAdmission()
        receipt["preparation_execution"] = {"backend": "independent_member_workers",
            "workers": workers, "available_cpu_workers": cpu_workers,
            "requested_worker_cap": worker_cap,
            "host_admission": admission.receipt(),
            "shared_binding_policy": "one preparation per exact source and physics binding"}
        jobs, by_binding = [], {}
        shared_downloads = _SharedDownloads()

        def prepare_job(member, member_config, member_wps, member_experiment, binding, directory):
            started = time.monotonic()
            download = None
            leader = False
            try:
                reused_run = None
                if binding.key in reusable:
                    result, reused_run = reusable[binding.key]
                    _say(f"member {member.index} reuses its ready preparation from {reused_run}")
                    if canaries is not None:
                        canaries.submit({"source": member.trajectory.source, **result})
                else:
                    from woof.go_cli import managed_download_dir
                    request, key = member_download_request(member_config, member_experiment, options)
                    leader, row = shared_downloads.claim(
                        key, member.index, lambda: managed_download_dir(case_root, request))
                    download = {"folder": str(row["folder"]),
                                "downloaded_by_member": row["member_id"], "wait_seconds": 0.0}
                    if not leader:
                        if not row["fetched"].is_set():
                            _say(f"member {member.index} waits for member {row['member_id']}'s "
                                 f"download of {Path(row['folder']).name}")
                        waited = time.monotonic()
                        row["fetched"].wait()
                        download["wait_seconds"] = round(time.monotonic() - waited, 3)
                    # The download runs under its own admission; the decode
                    # takes a host-RAM slot only once its inputs are on disk.
                    gate = _MemberStages(admission=admission, downloads=download_admission,
                                         order=member.index, leader=leader, folder=row["folder"],
                                         fetched=row["fetched"] if leader else None)
                    try:
                        result = prepare_member(
                            member_config, member_experiment, source=member.trajectory.source,
                            directory=directory, downloads=case_root, geog_root=geog_root,
                            composition_bank=(composition_banks[binding.source_key].name
                                              if binding.source_key in composition_banks else None),
                            composition_source_key=binding.source_key,
                            recipe_sha256=recipe.sha256, member_id=member.index, options={**options,
                            "_head_canaries": canaries, "_canary_namelist": member_wps},
                            download_dir=row["folder"], stage_gate=gate)
                    finally:
                        gate.close()
                        download.update(gate.timings)
                return result, reused_run, time.monotonic() - started, download
            except BaseException as error:
                from woof.ensemble.execution import name_failing_members
                name_failing_members(error, (member.index,))
                raise
            finally:
                if leader:
                    # A failed or fetch-less downloader still releases its
                    # followers; each one's own fetch then resumes the folder.
                    row["fetched"].set()

        # One thread per preparation: a thread waiting on a download or a
        # leader's folder holds no CPU, and the host-RAM admission bounds
        # the decodes.  With only ``workers`` threads, later cycles' leaders
        # could not start their downloads until an earlier decode ended.
        pool = ThreadPoolExecutor(max_workers=max(workers, min(len(plans), 512)),
                                  thread_name_prefix="ensemble-prepare")
        try:
            for plan in plans:
                member = plan.member
                directory = root / "members" / f"member-{member.index:03d}"
                member_config, member_wps, member_experiment = member_configuration(
                    raw, config, wps, recipe, member, directory / "source")
                from woof.ensemble.member_variants import preparation_binding_parts
                member_document = tomllib.loads(Path(member_config).read_text(encoding="utf-8"))
                lineage_fetches[member.index] = member_document["fetch"]
                binding = preparation_binding_parts(member.trajectory.identity, member_document,
                                                     experiment=member_experiment)
                original_member = prepared_banks.setdefault(binding.source_key, {}).get(binding.physics_tail)
                if binding.key not in by_binding:
                    _say(f"preparing member {member.index} ({member.trajectory.source} "
                         f"{member.trajectory.cycle:%Y-%m-%dT%H}Z)")
                    _notify(observer, "stage_begin", label="prepare", command=[
                        "woof", command, str(config), "#", member_label(member)])
                    future = pool.submit(copy_context().run, prepare_job, member,
                        member_config, member_wps, member_experiment, binding, directory)
                    by_binding[binding.key] = future
                    prepared_banks[binding.source_key][binding.physics_tail] = member.index
                jobs.append((plan, binding, original_member, by_binding[binding.key]))
            # Publish in roster order. Workers own separate directories and
            # event streams; only this thread replaces the run receipt.
            for plan, binding, original_member, future in jobs:
                member = plan.member
                where.update(stage="prepare", member_id=member.index)
                result, reused_run, elapsed, download = future.result()
                prepared[member.index] = dict(result)
                if original_member is not None:
                    _say(f"member {member.index} reuses member {original_member}'s prepared source; "
                         "its seeded surface state is applied at initialization")
                receipt["members"].append({
                    "member_id": member.index, "seed": member.seed,
                    "source": member.trajectory.source, "cycle": member.trajectory.cycle.isoformat(),
                    "source_member": member.trajectory.member,
                    "trajectory_sha256": member.trajectory.identity,
                    "preparation_binding_sha256": binding.key,
                    "source_binding_sha256": binding.source_key,
                    "physics_tail_sha256": binding.physics_tail,
                    "start_lead_hours": plan.start_lead,
                    "preparation_elapsed_seconds": elapsed,
                    **({} if download is None else {"download": download}),
                    **({} if not recipe.member_variants else
                       {"variant": recipe.member_variants[member.index]}),
                    **({} if original_member is None else
                       {"preparation_reused_from_member": original_member}),
                    **({} if reused_run is None else {"preparation_reused_from_run": reused_run}),
                    **prepared[member.index]})
                publish()
                if original_member is None:
                    _notify(observer, "stage_end", label="prepare", exit_code=0, ok=True,
                            elapsed_seconds=elapsed, progress=None)
        finally:
            for _, _, _, future in jobs:
                future.cancel()
            # No worker can still read a shared bank when it is removed.
            pool.shutdown(wait=True, cancel_futures=True)
        receipt["preparation_execution"]["elapsed_seconds"] = time.monotonic() - preparation_started
        fetched = _fetch_seconds(
            [root / "members" / f"member-{plan.member.index:03d}"
             for plan, binding, original_member, future in jobs
             if original_member is None and future.result()[1] is None],
            preparation_started_unix, time.time())
        #: The door's stage clocks, handed to the session so ensemble-run.json
        #: names where the whole run's wall went (fetch, prepare, forecast,
        #: products), not only the forecast session's own wall.
        stage_clock = {"started": door_started,
                       "plan": preparation_started - door_started,
                       "fetch": fetched,
                       "prepare": receipt["preparation_execution"]["elapsed_seconds"] - fetched}
        receipt["preparation_execution"]["host_admission"] = admission.receipt()
        receipt["preparation_execution"]["download_admission"] = download_admission.receipt()

        for bank in composition_banks.values():
            bank.cleanup()
        if canaries is not None:
            canaries.close()
            receipt["canary_sources"] = list(canaries.bindings)
        from woof.ensemble.lineage import enrich_request, trajectory_records
        from dataclasses import replace
        if not request.provenance:
            records, weights = trajectory_records([
                {"source": member.trajectory.source, "cycle": member.trajectory.cycle.isoformat(),
                 **({"member": member.trajectory.member} if member.trajectory.member is not None else {})}
                for member in recipe.members], base_seed=request.base_seed,
                variants=request.member_variants, stochastic=request.stochastic, weights=request.family_weights)
            request = replace(request, provenance=records, family_weights=weights)
        request = enrich_request(request, [lineage_fetches[member.index] for member in recipe.members])
        receipt["request"] = request.receipt()
        for member in receipt["members"]:
            member["provenance"] = request.provenance[member["member_id"]]
        base = recipe.members[0].index
        receipt["preparation_status"] = "ready"
        if options["prepare_only"]:
            receipt["status"] = "prepared"
            publish()
            _say(f"prepare-only: {len(recipe.members)} members prepared under {root / 'members'}; "
                 f"launch without --prepare-only and with --outdir {root}")
            return 0
        publish()
        where.update(stage="forecast", member_id=None, started=time.monotonic())
        bundle = stage_cli.resolve_bundle(Path(prepared[base]["prepared_root"]))
        forecast_dir = root / "run"
        # Relaunching an explicit failed run folder must not mix its old
        # forecast outputs into this attempt or trip the create-only runner.
        if forecast_dir.exists():
            forecast_dir = root / "forecast-attempts" / "run-001"
            attempt = 1
            while forecast_dir.exists():
                attempt += 1
                forecast_dir = root / "forecast-attempts" / f"run-{attempt:03d}"
        cache = {}

        def provider(*, shared_inputs, member_id, request):
            # Member ``base`` IS the bundle the runner preflighted; every other
            # member is the same preflight applied to its own prepared bundle.
            if member_id == base or prepared[member_id] == prepared[base]:
                return shared_inputs
            binding = tuple(prepared[member_id][name] for name in
                            ("prepared_root", "experiment_config", "wps_namelist"))
            if binding not in cache:
                # A member whose bundle the preflight refuses is that
                # member's forecast failure, and the receipt says which.
                where["member_id"] = member_id
                cache[binding] = member_inputs(shared_inputs, prepared[member_id])
                where["member_id"] = None
            return cache[binding]

        sim = stage_cli.sim_command(
            bundle, experiment_config=Path(prepared[base]["experiment_config"]),
            wps_namelist=(None if prepared[base]["wps_namelist"] is None else
                          Path(prepared[base]["wps_namelist"])), outdir=forecast_dir,
            physics_profile=None, progress_format="jsonl",
            render_products=options.get("render_products"),
            **({"devices": args.devices} if getattr(args, "devices", None) is not None else {}),
            memory_gate=not options.get("no_memory_gate"))
        receipt["status"] = "forecasting"
        receipt["forecast_directory"] = str(forecast_dir)
        publish()
        _say(f"running {len(recipe.members)} members in {forecast_dir}")
        _notify(observer, "stage_begin", label="forecast", command=list(sim))
        if bundle["layout"] == "tree":
            from woof import prepared_domain_tree_forecast as runner
        else:
            from woof import prepared_single_domain_forecast as runner
        # Bind the prepared authority before the real session constructor
        # validates member_variants and active surface controls. Binding it
        # after entering the scope refused a fully prepared named roster.
        with production_run_scope(request, output_directory=forecast_dir,
                                  input_provider=provider,
                                  **({"restart_roster": args.restart_roster}
                                     if getattr(args, "restart_roster", None) is not None else {})) as session:
            if session is not None:
                session.door_stage_clock = {**stage_clock, "forecast_started": where["started"]}
            code = runner.main(sim[3:], observer=hosted_observer)
            if code:
                # The runner said why on its own line (a refusal, or a
                # source lead later than its budget) and returned its code.
                # The session's run record keeps the real error and its
                # members; the receipt takes them from there.
                code = int(code)
                record = _forecast_record(forecast_dir)
                rows = list(record.get("failed_members") or ())
                failed(None, code, stopped=record.get("status") == "interrupted",
                       error_class=record.get("error_type"),
                       message=record.get("error") or f"the forecast stage exited {code}",
                       rows=rows)
                print(f"ensemble: stopped during {subject(forecast_members(rows))} "
                      f"(exit {code}). No later member or stage ran; "
                      f"{root / RECEIPT_NAME} records it.", file=sys.stderr, flush=True)
                return code
            products_receipt = session.completed_products()
            timing_seconds = (getattr(session, "last_manifest", None) or {}).get("timing_seconds")
        _notify(observer, "stage_end", label="forecast", exit_code=0, ok=True,
                elapsed_seconds=time.monotonic() - where["started"], progress=None)
    except BaseException as error:  # noqa: BLE001 - every exit is recorded in the receipt
        if canaries is not None:
            canaries.close(raise_error=False)
        code = _exit_code(error)
        from woof.ensemble.execution import failed_member_rows
        what = subject(forecast_members(failed_member_rows(error))
                       if where["stage"] == "forecast" else ())
        said = (str(error) or type(error).__name__).rstrip(".")
        if isinstance(error, SystemExit):
            # A stop signal, as run control reads it: the session says
            # ``interrupted``, so the receipt does too, with the stop's own
            # code.  One sentence, and the stop keeps the process status.
            failed(error, code, stopped=True)
            print(f"ensemble: interrupted during {what} (exit {code}). No later member "
                  f"or stage ran; {root / RECEIPT_NAME} records it.",
                  file=sys.stderr, flush=True)
            raise
        if code is not None:
            # A stage's own exit: it printed its reason above.  One line
            # and its code, as the ordinary door ends.
            failed(error, code)
            verb = "interrupted" if code == 130 else "stopped"
            print(f"ensemble: {verb} during {what}: {said}. No later member "
                  f"or stage ran; {root / RECEIPT_NAME} records it.",
                  file=sys.stderr, flush=True)
            return code
        failed(error, 2 if isinstance(error, ValueError) else 1)
        if isinstance(error, ValueError):
            # A refusal reached from inside a chain: the command's own
            # boundary prints its sentence at exit 2.
            raise
        if not isinstance(error, Exception):
            raise                              # GeneratorExit
        import traceback
        log = root / FAILURE_LOG_NAME
        try:
            log.write_text(traceback.format_exc(), encoding="utf-8")
            kept = f" Its traceback is in {log}."
        except OSError:
            kept = ""
        print(f"ensemble: {what} failed: {type(error).__name__}: {said}. "
              f"{root / RECEIPT_NAME} records it.{kept}", file=sys.stderr, flush=True)
        if explain_enabled(args):
            traceback.print_exc()
        return 1
    finally:
        for bank in composition_banks.values():
            bank.cleanup()
    receipt["status"] = "complete"
    receipt["products"] = products_receipt
    if timing_seconds is not None:
        receipt["timing_seconds"] = timing_seconds
    publish()
    _say(f"complete: {len(recipe.members)} members, products under {forecast_dir}; "
         f"member plan in {root / RECEIPT_NAME}")
    return 0
