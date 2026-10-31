"""Run a standalone CUDA child from archived woof/WRF parent history.

This is woof's native offline-nest driver.  It consumes ordinary parent
history plus authoritative source-physics evidence, performs SINT cold-start
and lateral-boundary preparation, destroys any need for a live parent, and
advances only the requested child on the GPU.  It never invokes WPS,
``real.exe``, ``wrf.exe``, or ``ndown.exe``.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Mapping

import numpy as np

from woof.io.netcdf_serialization import netCDF4  # first use only (D-10)
from woof import downscale_pricing
from woof.aerosol_source_receipt import aerosol_source_report_entry
from woof.config import (
    load_history_selection,
    radiation_scheme_ids,
    soil_layer_count,
)
from woof.io.history_selection import HISTORY_VOCABULARY
from woof.physics_registry import consumer_rows_by_selector
from woof.io.wrfout import INITIAL_CONDITION_GLOBAL_ATTRS
from woof.explain import warn
from woof.core.microphysics_transition import TRANSITION_ORDER
from woof.offline_child import (
    CHILD_REPORT_PIPELINE,
    DERIVED_CHILD_SURFACE_CAVEAT,
    PARENT_SCHEME_CONTRACT,
    OfflineChildContractError,
    OfflineChildPlacement,
    adapt_child_acoustics,
    bind_parent_physics_from_gpuwm_restart,
    bind_parent_physics_from_wrf_namelist,
    build_offline_child_domain_state,
    build_offline_lateral_boundaries,
    child_epssm_is_auto,
    child_mosaic_refusal,
    child_surface_requirement,
    derive_child_surface_from_parent,
    child_inherits_parent_levels,
    interpolate_parent_initial_state,
    les_child_regime,
    open_parent_history,
    read_child_surface_state,
    require_offline_child_root_forcing,
    require_runnable_child_radiation_from_archive,
    reserve_output_root,
    resolve_child_run_config,
    resolve_child_streaming_options,
    validate_parent_history,
)


_PROJECTION_ATTRS = (
    "MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON", "MOAD_CEN_LAT",
    "CEN_LAT", "CEN_LON", "POLE_LAT", "POLE_LON",
)

_CAPABILITIES = {
    "schema": "gpuwm-offline-child-capabilities-v1",
    "runner": "woof.offline_child_run",
    "status": "IMPLEMENTED_UNVERIFIED",
    "explicit_expert_consent_required": False,
    "parent_producers": ["woof", "stock-wrf"],
    "minimum_parent_frames": 2,
    "physics_evidence": ["gpuwm-restart", "wrf-namelist"],
    # DERIVED from the registry's consumers.offline_child rows, the same
    # source woof.offline_child.OFFLINE_CHILD_MP_PHYSICS is built from, so
    # the capability receipt users read cannot disagree with the gate.
    "same_scheme_mp_physics": sorted(
        int(mp) for mp, row in
        consumer_rows_by_selector("microphysics", "offline_child").items()
        if row.get("same_scheme") is True),
    # A child of a DIFFERENT scheme is converted by the online nest edge's
    # own contract and kernel (woof/core/microphysics_transition.py), run
    # on the archived parent's grid before interpolation, for every ordered
    # pair of these schemes; the one edge without a contract is mp=0 at
    # either end (woof.offline_child.offline_cross_scheme_refusal).
    "cross_scheme_transitions": {
        "mp_physics": sorted(int(mp) for mp in PARENT_SCHEME_CONTRACT),
        "policy": "woof.core.microphysics_transition (the online nest "
                  "edge contract and kernel, run on the parent archive)",
        "order": TRANSITION_ORDER,
    },
    # A child may carry its OWN eta ladder, deeper than the archived
    # parent's, when it declares one (``eta_levels`` in the child config,
    # written by ``woof downscale --child-levels``).  The remap is
    # conservative in dry mass and every water substance
    # (woof/vertical_remap.py) and runs once at preparation on the host; the
    # integration loop is unchanged.  p_top/hybrid_opt/etac stay shared with
    # the parent -- that is what gives the two ladders coincident endpoints.
    "vertical_remapping": "conservative-offline-prepare",
    # A child builds and runs on its OWN static geography by default
    # (woof/offline_child_geography.py: WRF ndown's blend_terrain and
    # rebalance, every frame); --parent-terrain keeps the parent's
    # interpolated terrain, which is what every child ran before.
    "terrain_policy": "child-own-static-geography",
    "terrain_policies": ["child-own-static-geography",
                         "sint-parent-inherited"],
    "forecast_backend": "cuda",
    "preprocess_backends": ["cuda", "cpu"],
    "output_ownership": "create-only",
    # Davies clock bind era: the standalone child binds a DomainClock to
    # its external LBC mirror, so boundary consumers take WRF's
    # post-increment dtbc recurrence exactly like the production tree
    # root (woof/ingest/lateral_bc.py bind_lateral_boundary_clock).
    "boundary_clock_semantics": "wrf-dtbc-bound",
    # Full-physics children (LSM/surface-layer/PBL) take a child-grid
    # surface source when one is given (--child-surface-from, the
    # ndown-equivalent contract, higher fidelity) and otherwise derive
    # one from the parent's own history through WRF's nest-birth
    # operators; mp-only children run without either.
    "full_physics_surface_source":
        "child-grid-file-or-parent-history-interpolated",
    # ``[tiles]`` in the child config: this route wires a streamed-domain
    # builder, so it HONORS the block rather than refusing it.  A child
    # refined out of an archived parent is the domain most likely to outgrow
    # the card it is being run on, and until this it was the one route that
    # could not ask to stream -- the RunConfig schema refused the table
    # outright as unknown.
    "tiles": "honored",
}


def _strict_json(document, **options) -> str:
    """One document as JSON a strict reader can open.

    WHAT BREAKAGE THIS PREVENTS (gate law).  ``NaN`` is not a JSON token:
    RFC 8259 has no spelling for it, and ``JSON.parse``, ``serde_json``,
    ``encoding/json`` and ``jq`` all reject a document carrying one,
    while Python's ``json`` writes and reads it by default.  A document
    written from this module could therefore be opened by the tree that
    wrote it and by nothing else -- and the outcome that writes the most
    of them is the one whose readings are exactly the ones that went.
    ``allow_nan=False`` is this tree's own convention for a receipt
    (``woof/background_contract.py``, ``woof/branch.py``,
    ``woof/case_catalog.py``, ``woof/certify/band.py``,
    ``woof/certify/verdict.py``).

    Its REFUSAL is the wrong outcome here: these documents are written on
    the way out of a run that has already failed, and losing the whole
    report to save one number inverts what the report is for.  So a
    non-finite value that still reaches this function -- a health
    record's own CFL through the layer-geometry channel, a mean over no
    samples -- is written as ``null``, which every reader has, and the
    document stays strict.  The health readings never take that route:
    they arrive in the carrying shape :func:`child_health_log_fields`
    gives them, a number or ``null`` beside a state word.
    """

    try:
        return json.dumps(document, allow_nan=False, **options)
    except ValueError:
        # Written permissively, read back with every non-finite token
        # mapped to null, written strictly.  Token-level, so a numpy
        # float that ``default=`` resolved is covered by the same pass.
        relaxed = json.loads(json.dumps(document, default=_jsonable),
                             parse_constant=lambda _token: None)
        return json.dumps(relaxed, allow_nan=False, **options)


def _log(event: str, **values) -> None:
    print(_strict_json({"event": event, **values}, sort_keys=True),
          flush=True)


from woof.first_products import DEFAULT_WAIT_SECONDS as _EARLY_RENDER_WAIT

#: One sentence for the two doors that ask it (this runner at admission and
#: woof downscale before it opens a frame), so they cannot drift apart.
RENDERER_MISSING_REMEDY = (
    "The Rust renderer is unavailable, so this child's products cannot be "
    "drawn. Next: woof setup, then repeat this command. Use "
    "--render-products none to run the child without pictures.")


class ChildStopped(SystemExit):
    """A stop asked for with SIGTERM, raised where the child was.

    THE BREAKAGE: ``woof downscale`` tells a
    reader to stop it with SIGTERM when Ctrl-C cannot reach it, the page
    follows its own Ctrl+C with SIGTERM a minute later, and the child had
    no handler for it: the process died at the default disposition, exit
    143, with no line printed, no event, no ``report.json``, no banner and
    a run manifest carrying only its start, so every reader of the folder
    saw a run that was still going.  :func:`stop_on_signal` raises this
    instead, and the child's stop path records the stop.

    A ``SystemExit``, so nothing between the signal and that path that
    catches ``Exception`` can swallow it, and a door that does not catch
    it exits with ``128 +`` the signal: 143 for SIGTERM, the status a
    shell reports for a process SIGTERM killed.
    """

    def __init__(self, signal_number: int):
        self.signal_number = int(signal_number)
        try:
            self.signal_name = signal.Signals(self.signal_number).name
        except ValueError:
            self.signal_name = f"signal {self.signal_number}"
        super().__init__(128 + self.signal_number)

    def __str__(self) -> str:
        return f"stopped by {self.signal_name}"


#: How many :func:`stop_on_signal` blocks are open: the ``woof
#: downscale`` door opens one and the child it runs opens another, and
#: only the outer one installs anything.
_STOP_DEPTH = 0


def _raise_stop(signal_number, _frame) -> None:
    raise ChildStopped(signal_number)


@contextlib.contextmanager
def stop_on_signal(command: str = "woof downscale"):
    """A stop signal ends this child through its stop path, and says so.

    SIGTERM is turned into :class:`ChildStopped`, raised where the child
    was, exactly as Python turns SIGINT into ``KeyboardInterrupt``; the
    child's stop path (:func:`_record_stop`) then writes the stop into
    the event stream, ``report.json``, the banner and the run manifest.
    Both signals are reported on stderr first by the same report the
    supervisor installs (:func:`woof.signal_report.report_on_signal`),
    in its wording for a door whose runs are stopped this way.

    A disposition somebody else set is left alone: SIGTERM is taken only
    from its default, and a SIGINT a shell set to ignore stays ignored.
    Restored on the way out, because tests and embedders call the door
    repeatedly in one interpreter.  Nested blocks install once.
    """

    global _STOP_DEPTH
    if _STOP_DEPTH:
        yield
        return
    from woof.signal_report import report_on_signal

    replaced = None
    number = getattr(signal, "SIGTERM", None)
    try:
        if number is not None and signal.getsignal(number) is signal.SIG_DFL:
            signal.signal(number, _raise_stop)
            replaced = signal.SIG_DFL
    except (ValueError, OSError):
        # Not the main thread, or no signal support here: the child
        # runs exactly as it did before this existed.
        replaced = None
    _STOP_DEPTH += 1
    try:
        with report_on_signal(command, stoppable=True):
            yield
    finally:
        _STOP_DEPTH -= 1
        if replaced is not None:
            try:
                signal.signal(number, replaced)
            except (ValueError, OSError):
                pass


@contextlib.contextmanager
def _stop_signals_held():
    """While a stop is being recorded, a second stop signal is absorbed.

    The run is already stopping, and the recording takes a few seconds
    at most (the renders are ended on a 2 s grace).  A second Ctrl-C or
    SIGTERM landing in the middle of it used to raise again inside the
    code writing the banner and the report, leaving half of them.
    SIGKILL still ends the process at once, which is what the desktop
    sends 5 s after its stop.
    """

    held = []
    for name in ("SIGINT", "SIGTERM"):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            earlier = signal.getsignal(number)
            if earlier is signal.SIG_IGN or earlier is None:
                continue
            signal.signal(number, lambda _number, _frame: None)
        except (ValueError, OSError):
            continue
        held.append((number, earlier))
    try:
        yield
    finally:
        for number, earlier in held:
            try:
                signal.signal(number, earlier)
            except (ValueError, OSError):
                pass


class _ChildProgress:
    """The offline child's own run manifest and native event stream.

    A child used to publish ``report.json`` at the end and a stream of
    ``{"event": ...}`` lines on stdout, and nothing bound those lines to
    the process that wrote them.  No run browser could show a downscale
    in flight, and a finished one could not be listed beside the
    forecasts it came from.

    So this publishes THE SAME TWO RECEIPTS every other WOOF run
    publishes -- ``gpuwm.run-manifest.v1`` beside the frames and
    ``gpuwm.run-plan.event.v1`` in ``events.jsonl`` -- rather than
    inventing a third progress shape for one route.  Every existing
    reader (the controller's local progress reader, the desktop's live
    view) then works on a child unchanged, and the stream is mirrored to
    stdout so the job log keeps the lines it always had.

    Every method is a no-op until :meth:`start` has run, so a refusal
    raised before the contracts pass still costs nothing.
    """

    def __init__(self) -> None:
        self.events = None
        self.manifest_path = None
        self.run_id = None
        self.outdir = None
        #: The plan the child's pictures are drawn from -- the very dict
        #: :func:`woof.go_cli._render_stage` is handed, so the early
        #: render and the finalize one cannot drift apart in what they
        #: draw or where they put it.  ``None`` when no products were
        #: asked for.
        self.render_plan = None
        self._stage = None
        #: The stage the stream last opened, whichever way it was opened
        #: (the forecast loop emits its own ``stage_started``), so a stop
        #: can say where it landed.
        self._current_stage = None
        self._stage_phases: list[str] = []
        self._stage_started_wall = None
        self._started_wall = time.perf_counter()
        self._first_products = None
        self._first_products_seconds = None
        #: Every frame after the analysis, drawn as it lands
        #: (:mod:`woof.live_products`), armed beside the early render.
        self._live_products = None
        #: The any-point column store, written as frames land (:mod:`woof.live_columns`).
        self._columns = None
        #: Which of the child's frames carry the 3-D state, from its own [output]
        #: (:class:`woof.live_columns.StateFrames`); ``None`` takes every frame.
        self.columns_frames = None
        #: How far the forecast got, from the last progress sample that
        #: carried a step number.  ``None`` until the first one, which
        #: is the state a child refused before its first step is in --
        #: and the banner says so rather than quoting a zero nobody
        #: measured.
        self._stopped = None
        #: The history frames this run has committed, in order, named
        #: for the banner a stopped run leaves.
        self._frames: list[str] = []
        #: The domain a coarse progress sample is attributed to.  A
        #: child is one domain and :meth:`start` learns which.
        self._root_domain = 1
        #: Set by :class:`woof.runplan._GoObserver`, which is what the
        #: shared render stage reports through on every route.
        self._render_summary = None

    def start(self, *, outdir: Path, child_config: Path, ratio: int,
              start_time, parent: dict, name: str) -> None:
        from datetime import datetime, timezone
        import uuid

        from woof import proc_identity, runplan
        from woof.supervisor import atomic_write_json

        self.outdir = Path(outdir)
        events_path = self.outdir / "events.jsonl"
        # The stream first: the manifest names this file, and a reader
        # that finds the manifest must find the file it points at.
        self.events = runplan.EventStream(events_path)
        self.run_id = f"downscale-{uuid.uuid4().hex}"
        config = Path(child_config).resolve()
        document = {
            "schema": runplan.MANIFEST_SCHEMA,
            "name": name,
            "route": "downscale",
            "run_id": self.run_id,
            "pid": os.getpid(),
            # With its creation time, so a reader can tell this run from a
            # later program given the same pid (see woof.proc_identity).
            "process": proc_identity.identify(os.getpid()),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            # A child's "plan" IS its configuration: this route has no
            # run-plan document, and the reader's binding is the same
            # one either way -- name the source and hash it.
            "plan_source": f"woof downscale {config}",
            "plan_sha256": _sha256(config),
            "run_dir": str(self.outdir),
            "outputs_dir": str(self.outdir),
            "events_path": str(events_path),
            "events_schema": runplan.EVENT_SCHEMA,
            # No supervisor heartbeat on this route.  Null rather than a
            # path to a file nothing writes: a reader that opened it
            # would wait forever for a first sample.
            "progress_path": None,
            "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "parent": dict(parent),
        }
        self.manifest_path = self.outdir / runplan.MANIFEST_FILENAME
        atomic_write_json(self.manifest_path, document)
        self.emit("resolved_plan", config_source=str(config),
                  config_sha256=document["plan_sha256"])

    def emit(self, event: str, **fields) -> None:
        # Read on the way past, because this is the one funnel every
        # route's events go through: the forecast loop emits
        # ``model_progress`` straight through here, and a banner written
        # on a failure path cannot ask a loop that already unwound how
        # far it got.
        if event == "model_progress":
            self._note_progress(fields)
        elif event == "stage_started":
            self._current_stage = fields.get("stage")
        elif event == "output_committed":
            path = fields.get("path")
            if path is not None:
                self._frames.append(Path(str(path)).name)
        if self.events is not None:
            self.events.emit(event, **fields)

    def _note_progress(self, fields) -> None:
        """Keep the last sample that names a STEP.

        The render stage's coarse heartbeat emits the same tag with a
        model clock and no step (:meth:`stage_progress`), and taking it
        would leave a stopped forecast reporting the render's second as
        the second it stopped at.

        The two SECONDS stay real and the two COUNTS stay integers.  A
        step count carried as a float reaches the banner as a float and
        was printed as one, which is how the sentence a reader opens
        after losing a forecast came to hold a mantissa.
        """

        if fields.get("outer_step") is None:
            return

        def number(key):
            value = fields.get(key)
            return None if value is None else float(value)

        def count(key):
            value = fields.get(key)
            return None if value is None else int(float(value))

        self._stopped = {"model_seconds": number("model_seconds"),
                         "run_seconds": number("run_seconds"),
                         "step": count("outer_step"),
                         "total_steps": count("total_steps")}

    def failed(self, error: BaseException | None = None, *,
               stage: str = "forecast") -> None:
        """This run failed.  Both callers of that sentence reach here.

        The runner's progress protocol calls ``failed()`` with no
        argument -- :meth:`woof.runplan.RunObserver.failed` takes none,
        and :meth:`woof.runplan._GoObserver.failed` forwards to it --
        while this module's own exit paths call it with the exception
        they caught.  A signature that took only the second spelling
        made this object incompatible with the very surface it is handed
        to, and the incompatibility was reachable: it is a
        ``TypeError`` raised on a failure path, which is where a
        readable message matters most.
        """

        if error is None:
            self.emit("failed", stage=stage)
            return
        # ONE SENTENCE, the same one the banner and the report's
        # ``failure.summary`` carry, with the class name in front.  This
        # message is what a run view puts on one line, and a whitespace
        # join of the whole exception turned a multi-paragraph refusal --
        # a survey block, a table of health checks, a command to type --
        # or a contract error's paragraphs into 1,600 characters of prose
        # with the sentence that matters buried at the front of it.
        # :func:`_stop_reason` chooses the sentence (a refusal's own
        # ``summary`` where it composed one, the first sentence of the
        # text otherwise); the whole text is in ``report.json`` under
        # ``failure.message``, so nothing is lost by saying it once here.
        body = _stop_reason(error)
        message = " ".join(f"{type(error).__name__}: {body}".split())
        self.emit("failed", stage=stage, message=message[:1600])

    # -- the run-plan observer surface the shared render stage drives --
    #
    # `woof.runplan._finish_render` is the finalize render EVERY other
    # route uses, and it reports through `_GoObserver`, which asks its
    # observer for exactly these five things.  Supplying them here is
    # what lets a child's pictures be drawn by that function rather than
    # by a second render stage written for this route -- two renderers
    # for one product set is how a downscaled run ends up drawing a
    # different catalog, into a different folder, than the forecast it
    # was cut from.

    def warn(self, code: str, message: str, **fields) -> None:
        self.emit("warning", code=code, message=message, **fields)

    def stage_progress(self, *, phase: str, elapsed_seconds: float,
                       model_seconds: float, status=None) -> None:
        """One coarse sample from a stage that runs out of process.

        :meth:`woof.runplan._GoObserver.stage_heartbeat` calls this
        whenever the running stage's progress file carries a model
        clock, so an observer that lacks it raises ``AttributeError``
        out of a heartbeat -- inside the render stage of a finished
        forecast.

        Spelled as :meth:`woof.runplan.RunObserver.stage_progress`
        spells it, down to ``source``: one event tag, one field set, so
        a reader of a downscaled run's stream and a reader of a
        forecast's are the same reader.
        """

        speed = (round(model_seconds / elapsed_seconds, 4)
                 if elapsed_seconds > 0.0 and model_seconds > 0.0 else None)
        self.emit("model_progress", domain=self._root_domain,
                  model_seconds=float(model_seconds),
                  wall_seconds=round(float(elapsed_seconds), 6),
                  speed_x=speed, step_ms=None, phase=phase,
                  status=status, source="stage_progress_file")

    def enter_stage(self, stage: str, *, phase: str | None = None) -> None:
        """Close whatever stage is open and open ``stage``.

        Re-entering the open stage is a no-op, exactly as
        :meth:`woof.runplan.RunObserver.enter_stage` treats it: the
        finalize render is announced by its caller AND by the stage
        hook, and one render is one ``stage_started``.
        """

        if stage == self._stage:
            return
        self.finish_stage()
        self._stage = stage
        self._stage_phases = [] if phase is None else [phase]
        self._stage_started_wall = time.perf_counter()
        fields = {"stage": stage}
        if phase is not None:
            fields["phase"] = phase
        self.emit("stage_started", **fields)

    def finish_stage(self, **fields) -> None:
        """Close the open stage, carrying the render summary on finalize."""

        if self._stage is None:
            return
        stage, started = self._stage, self._stage_started_wall
        self._stage = None
        if stage == "finalize" and self._render_summary is not None:
            fields.setdefault("render_summary", self._render_summary)
        self.emit("stage_finished", stage=stage,
                  wall_seconds=round(time.perf_counter() - started, 6),
                  phases=list(self._stage_phases), **fields)

    def arm_render(self, *, outdir, render_products) -> dict | None:
        """Arm this child's pictures: the plan and the renders that use it.

        The plan is built ONCE, here, and every render reads it: the
        early render of the analysis frame, the render of every later
        frame as it lands, and the finalize one
        :func:`_finish_child_render` runs.  ``render_products`` absent
        or ``none`` arms nothing and leaves :attr:`render_plan` ``None``,
        which is the single answer to "does this run draw?".

        That second answer is not decided here.  This door hands the
        plan to :func:`woof.first_products.arm`, which is where the
        whole decision lives for every door, so a route cannot grow a
        second answer to "did this run ask for pictures".

        THE DEFECT THE SECOND RENDER CLOSES: only the analysis frame was
        drawn while the child ran, and every other frame waited for the
        forecast to end.  Measured on a 250 m child: its second frame
        landed at 10:40 and was drawn at 13:16, with every other hour,
        and the run emitted no ``live_products_ready`` while the forecast
        it was cut from emitted 12.  The run-plan route draws each frame
        as it lands (:class:`woof.live_products.LiveProducts`); a child
        now gets the same object, on the same plan, sharing bounded slots
        with an exclusive early render, and the finalize
        render draws only what it cannot prove was drawn.

        Both renders start in a process group of their own
        (``own_group``), because this child ends them itself when it is
        stopped (:meth:`halt_renders`).  In the child's group, the Stop
        that the desktop and the terminal send to the whole group killed
        a render mid-organisation, and the child published what it had
        staged.
        """

        from woof.first_products import arm
        from woof.live_products import (LiveProducts, early_render_runner,
                                          shared_render_slots)

        root = Path(outdir)
        plan = {"run": root, "wrfout_dir": root,
                "render": root / "png",
                "render_products": str(render_products)}
        slot, early_slot = shared_render_slots()
        trigger = arm(plan, report=self._first_products_ready,
                      warn=self.warn, runner=early_render_runner,
                      slot=early_slot, own_group=True)
        if trigger is None:
            self.render_plan = None
            return None
        self.render_plan = plan
        self._first_products = trigger
        self._live_products = LiveProducts(
            plan, report=self._live_products_ready, warn=self.warn,
            first=trigger, slot=slot, own_group=True)
        return self.render_plan

    @property
    def first_products(self):
        """The armed early render, read by the finalize stage."""

        return self._first_products

    @property
    def first_products_seconds(self) -> float | None:
        """Time to first plot, or ``None`` if no early render published."""

        return self._first_products_seconds

    @property
    def live_products(self):
        """The render of every frame as it lands, read by finalize."""

        return self._live_products

    def stop_live_products(self, *, halt: bool = False) -> dict | None:
        """Stop drawing frames as they land.

        ``halt`` is a stopped run: nothing queued is drawn and the render
        in flight is ended.  Otherwise the queue is finished first, which
        a child that failed on its own still gets, so the frames it wrote
        before the failure are pictures too.  The finalize stage
        stops it itself (:func:`woof.go_cli._render_stage`).
        """

        self.stop_columns(halt=halt)
        live = self._live_products
        if live is None:
            return None
        return live.halt() if halt else live.stop()

    def stop_columns(self, *, halt: bool = False) -> None:
        """Settle column receipts before publishing the terminal event.

        Columns run even without pictures. Leaving them to close() put a
        columns_written event after the stop and abandoned no-picture passes.
        """
        columns, self._columns = self._columns, None
        if columns is not None:
            try:
                s = columns.halt() if halt else columns.stop()
                self.emit("columns_written", files=s["files"], bytes=s["bytes"], seconds=s["seconds"],
                          failed=len(s["failed"]), out=s["out"])
            except Exception as error:  # noqa: BLE001 - a store never fails the child
                self.warn("columns_not_written", str(error))

    def _column_frame(self, fields) -> None:
        """The child's any-point column store (rw-store v2), written beside its frames for every output hour."""
        try:
            if self._columns is None and self.outdir is not None:
                from woof.live_columns import LiveColumns, columns_enabled
                if not columns_enabled():
                    return
                self._columns = LiveColumns(Path(self.outdir) / "columns", warn=self.warn,
                                            frames=self.columns_frames)
            if self._columns is not None:
                self._columns.frame_committed(domain=int(fields["domain"]), valid_time=fields["valid_time"],
                                              path=fields["path"])
        except Exception as error:  # noqa: BLE001 - telemetry never fails
            self.warn("columns_not_written", f"committed frame could not be queued: {error}",
                      frame=str(fields.get("path")))

    def halt_renders(self) -> bool:
        """End every render this child started, for a stop.

        Nothing queued is drawn and nothing more is published; the early
        render, the render of the frame that landed last and the
        finalize render in flight are each ended rather than waited for
        (:meth:`woof.first_products.FirstProducts.halt`,
        :meth:`woof.live_products.LiveProducts.halt`,
        :func:`woof.go_cli.end_stage_processes`), because the desktop
        kills a stopped run 5 s after asking and the stop still has to be
        written down.  The every-frame render is closed first and joined
        last: its worker may be waiting on the early render, which is
        ended in between.

        ``True`` once no render of this run is running, which is what
        licenses tidying the picture folder
        (:func:`woof.first_products.discard_stopped_render`).
        """

        from woof.go_cli import end_stage_processes

        first = self._first_products
        live = self._live_products
        if live is not None:
            live.halt(timeout=0)
        ended = True
        if first is not None:
            ended = first.halt() and ended
        if live is not None:
            live.halt()
            ended = not live.running and ended
        self.stop_columns(halt=True)
        return end_stage_processes() and ended

    def stopped(self, *, stage: str, message: str, signal_name: str,
                exit_code: int) -> None:
        """This run was stopped by request.  The stream's last word.

        The run-plan vocabulary's own spelling of a stop, which every
        reader already keys on: a ``failed`` event carrying
        ``interrupted: true`` and the exit code
        (:func:`woof.runplan.execute_plan` writes the same fields, and
        the page reads that pair as Stopped).  A tag of its own would be
        a line no existing reader recognises as the end of a run.

        THE BREAKAGE: a Stop from the desktop or
        the terminal was published as ``failed`` with the message
        ``KeyboardInterrupt: KeyboardInterrupt``, which is neither a
        reason nor something a reader can act on.
        """

        self.emit("failed", stage=stage, message=message[:1600],
                  interrupted=True, exit_code=int(exit_code),
                  signal=signal_name)

    def end(self, state: str) -> None:
        """Write how and when this run ended into its run manifest.

        The manifest carried only ``started_at_utc``, so a folder whose
        run had stopped read as one still running to anything that read
        the folder rather than the process.
        ``end_state`` is ``completed``, ``failed`` or ``stopped``, beside
        ``ended_at_utc``.  Everything else in the document is kept as it
        was written.  Best effort: nothing raises.
        """

        if self.manifest_path is None:
            return
        from datetime import datetime, timezone

        from woof.supervisor import atomic_write_json

        try:
            document = json.loads(
                Path(self.manifest_path).read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                return
            document["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            document["end_state"] = str(state)
            atomic_write_json(self.manifest_path, document)
        except (OSError, ValueError):
            pass

    def _live_products_ready(self, entry) -> None:
        """One frame is readable as pictures, spelled as run-plan spells it."""

        self.emit(
            "live_products_ready", domain=entry["domain"],
            valid_time=entry["valid_time"], frame=entry["frame"],
            pictures=entry["pictures"],
            render_seconds=entry["render_seconds"],
            queued=entry["queued"], complete=entry.get("complete", True))

    def _first_products_ready(self, receipt) -> None:
        """The early render published.  This is the TTFP number."""

        elapsed = round(time.perf_counter() - self._started_wall, 6)
        self._first_products_seconds = elapsed
        self.emit(
            "first_products_ready",
            domain=receipt["domain"], valid_time=receipt["valid_time"],
            frame=receipt["frame"], paths=list(receipt["paths"]),
            render_products=receipt["render_products"],
            render_seconds=receipt["render_seconds"],
            seconds_from_plan_accepted=elapsed,
            complete=receipt.get("complete", True))

    def output_committed(self, **fields) -> None:
        """One child history frame is durable.  It is drawn as it lands.

        The event is the same one this route always emitted; the
        dispatch beside it is what makes each frame a picture while the
        rest of the forecast is still integrating, as it is on every
        other route.  The analysis frame goes to the early render, which
        decides for itself that only the first frame wins; every other
        frame is queued for the every-frame render and this returns at
        once.  Nothing here raises into the forecast loop.
        """

        # The child's own domain, learned from the frames it commits:
        # this route has one, and a coarse progress sample has to name
        # it rather than the root of a hierarchy it does not have.
        self._root_domain = int(fields["domain"])
        self.emit("output_committed", **fields)
        self._column_frame(fields)
        claimed = False
        if self._first_products is not None:
            claimed = bool(self._first_products.frame_committed(
                domain=fields["domain"], valid_time=fields["valid_time"],
                path=fields["path"]))
        live = self._live_products
        if live is None:
            return
        try:
            live.frame_committed(domain=int(fields["domain"]),
                                 valid_time=fields["valid_time"],
                                 path=fields["path"], draw=not claimed)
        except Exception as error:  # noqa: BLE001 - telemetry never fails
            self.warn(
                "live_products_failed",
                f"a committed frame could not be queued for drawing "
                f"({type(error).__name__}: {error}); the end-of-run render "
                "draws it", frame=str(fields["path"]))

    def wait_early_render(self, *, timeout: float | None = _EARLY_RENDER_WAIT) -> None:
        """Join the early render, wherever this run is exiting from.

        A render dispatched on a worker thread outlives nothing: the
        subprocess it owns is still drawing when the process that armed
        it returns.  The finalize stage waits for exactly this reason,
        and every OTHER exit -- a refused forecast, a raised contract, a
        render stage that failed -- has the same obligation, so the wait
        lives on the one method all of them pass through.
        """

        if self._first_products is not None:
            self._first_products.wait(timeout=timeout)

    def early_pictures(self) -> int:
        """How many pictures the early render has published, after joining it."""

        if self._first_products is None:
            return 0
        # The bounded wait: this is a count for a sentence, and a wedged
        # renderer must not hold a run that already failed.  The receipt
        # lists what the early render published under "written".
        receipt = self._first_products.wait()
        written = receipt.get("written") if isinstance(receipt, dict) else None
        return len(written) if isinstance(written, list) else 0

    def pictures_drawn(self) -> tuple[int | None, str | None]:
        """``(pictures on disk, why they could not be counted)``.

        COUNTED from the tree rather than inferred from the early
        render's receipt.  A ``--series`` render draws frame by frame
        and the renderer exits nonzero at the END, when its batch
        summary carries a failure, so a run whose render stage failed
        has usually drawn every frame's other products: measured on the
        shipped 2.7.5 wheel, a 13-frame child whose two requested snow
        variables could not be drawn left 143 pictures on disk while its
        capsule said the rest were not drawn.  A reader who believed
        that re-drew six hours of frames for pictures already there.

        TWO ANSWERS, because a count that could not be TAKEN is not a
        count of zero.  A tree whose listing fails -- a permission wall,
        a dead mount, a path that is a regular file -- used to come back
        as ``0`` and print as "this run has no pictures", which is the
        sentence that sends a reader to re-draw a whole child.  It now
        comes back as ``(None, the error)`` and the caller says the tree
        could not be read.  A directory that is simply not there is
        still the empty case and still counts zero: a render stage that
        never created its output directory drew nothing, and that is a
        reading rather than a failure to read.

        Nothing raises: this is a number for a sentence in a refusal
        that is already being raised.
        """

        from woof.first_products import count_pictures

        if self.render_plan is None:
            return 0, None
        try:
            root = Path(self.render_plan["render"])
        except (KeyError, TypeError, ValueError) as error:
            return None, f"this run recorded no picture directory ({error})"
        # ONE counter for the capsule, the banner, the render summary
        # and the report, so a reader cannot be handed two numbers for
        # one tree, and neither route can lose the difference between an
        # empty folder and one nothing could list.
        return count_pictures(root)

    def keep_early_render(self, why: str, *, requested: bool = False,
                          discard: bool = False) -> dict:
        """Keep what the early render drew, under a did-not-finish banner.

        THE DECISION, recorded where it is enforced: a child that does
        not finish KEEPS its pictures.  The early render draws the
        analysis frame while the forecast is still integrating, and this
        door used to remove every picture it had published the moment
        the run stopped -- so a child that stopped part way through its
        forecast left its reader nothing at all to look at.
        What the pictures were missing was never the pictures: it was
        the verdict beside them.  So the verdict is written beside them,
        in the banner at the top of the render directory, in the render
        summary and in this run's event stream, and nothing is removed.
        IN THE REPORT TOO: a child that stops inside its forecast
        publishes ``report.json`` whatever stopped it, carrying its
        failure capsule or, for a stop that composed none, the sentence
        this banner carries with the whole of what was raised
        (:func:`_stop_capsule`); and both stop arms put what this
        returned into that document's ``products`` block as ``status``
        ``KEPT`` with the count on disk and the banner's path, so the
        folder, the summary and the document say one thing about one
        run.

        The wait is UNBOUNDED, as the removal's was, for the same two
        reasons in their new order: the render thread owns a subprocess
        that must not outlive this process, and the count in the banner
        has to be the count of a render that finished drawing rather
        than of one caught mid-publish.

        ``{"pictures": n, "pictures_error": None, "banner": path or
        None, ...}`` comes back; nothing raises.  ``pictures`` is
        ``None`` with the error beside it when the tree could not be
        listed, which is not the same reading as a tree with nothing in
        it and is not printed as one.

        A run with no render at all keeps the same shape with a count of
        zero: it drew nothing because it was asked for nothing, which is
        a reading and not a failure to read.

        A STOP (``requested``) has already ended its renders
        (:meth:`halt_renders`) and is not held here: the unbounded wait
        is for a run that failed on its own.  ``discard`` removes what a
        stopped render left before anything is counted
        (:func:`woof.first_products.discard_stopped_render`), and is
        passed only when every render has ended.
        """

        if self._first_products is None or self.render_plan is None:
            return {"pictures": 0, "pictures_error": None,
                    "banner": None, "summary": None}
        from woof.first_products import keep

        if not requested:
            self.wait_early_render(timeout=None)
        render_dir = Path(self.render_plan["render"])
        kept = keep(render_dir, why=why, stopped=self._stopped,
                    frames=list(self._frames), requested=requested,
                    discard=discard)
        # A `warning` carrying its own code, because the event
        # vocabulary is a closed schema shared with every reader of
        # every route (:data:`woof.runplan.EVENT_TAGS` for the tag and
        # :data:`woof.runplan.WARNING_CODES` for the code) -- either one
        # invented for one route is a record nothing can read.
        self.warn(
            "early_render_kept", _kept_sentence(kept, render_dir),
            render=str(render_dir), pictures=kept["pictures"],
            banner=kept["banner"], status=kept["status"], why=why)
        return kept

    def close(self, *, wait: bool = True) -> None:
        # The floor under wait_early_render: every exit path closes the
        # stream, so every exit path waits.  And nothing is drawn after
        # the stream closes: a run that got here without its finalize
        # stage stopping the every-frame render (a finalize that raised
        # before it drew) halts it now rather than leave a render running
        # past the process that asked for it.  A stop has ended its
        # renders already and passes ``wait=False``: the desktop kills it
        # 5 s after asking.
        self.stop_live_products(halt=True)
        if wait:
            self.wait_early_render()
        if self.events is not None:
            self.events.close()
            self.events = None


#: Folder names every prepared route uses for its layout rather than for
#: the run's identity: ``<run>/wrfout/`` holds the frames, ``<run>`` is
#: itself called ``run`` under a stamped folder, and a run-plan forecast
#: keeps its whole chain under ``<run>/chain/`` (its frames land in
#: ``<run>/chain/run/wrfout/``).  A child named after one of these would
#: be "Downscale of wrfout" or "Downscale of chain".
_LAYOUT_FOLDERS = frozenset({"wrfout", "run", "chain"})


def _run_manifest(folder: Path) -> dict | None:
    """The run manifest in ``folder`` when it carries a name, else None."""

    from woof import runplan

    try:
        document = json.loads((Path(folder) / runplan.MANIFEST_FILENAME)
                              .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(document, dict)
            or document.get("schema") != runplan.MANIFEST_SCHEMA):
        return None
    name = document.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    return document


def _binds(outer: Path, document: dict, producer: Path) -> bool:
    """True when the run in ``outer`` recorded ``producer`` as its native run.

    Compared by the names below ``outer`` rather than by the whole path,
    so a run folder copied to another disk or machine keeps the binding
    its manifest recorded where it ran.
    """

    native = document.get("native_run")
    recorded = native.get("run_dir") if isinstance(native, dict) else None
    if not isinstance(recorded, str):
        return False
    try:
        inside = producer.relative_to(outer).parts
    except ValueError:
        return False
    tail = tuple(part for part in recorded.replace("\\", "/").split("/") if part)
    return bool(inside) and tail[-len(inside):] == inside


def _parent_identity(frame: Path) -> tuple[str, str | None]:
    """The name a child is called after, and the route of the run it names.

    The parent run's own name first: the ``name`` in its run manifest,
    which is the name a run browser shows for that run.  It is looked
    for from the frame's folder upward, through layout folders and
    stamped run folders (a run-plan forecast's chain claims a stamped
    folder under ``<run>/chain/``), and no further than the first folder
    that is neither.  A run found there that another run recorded as its
    native producer (the ``woof go`` run inside a run-plan forecast's
    chain) gives way to that run, because that is the run a person
    started.

    Where no manifest says, the nearest ancestor that is not a layout
    folder names the parent, so the stamped run folder names it on the
    prepared routes and the history directory itself does everywhere
    else (a WRF parent, a copied folder).  The route is then ``None``.

    The walk starts from the frame's absolute path: ``woof downscale
    chain/run/wrfout`` run from inside the parent's run folder hands the
    frames on as typed, and a relative path runs out of folders at
    ``.`` before it reaches the run folder and its manifest.  ``abspath``
    rather than ``resolve``, so a folder reached through a link keeps
    the name it was given.
    """

    from woof.run_stamp import is_run_folder

    frame = Path(os.path.abspath(frame))
    found = None
    for ancestor in frame.parents:
        if not ancestor.name:
            break
        document = _run_manifest(ancestor)
        if document is not None:
            found = ancestor, document
            break
        if ancestor.name not in _LAYOUT_FOLDERS and not is_run_folder(ancestor):
            break
    if found is not None:
        folder, document = found
        for outer in folder.parents:
            if not outer.name:
                break
            binder = _run_manifest(outer)
            if binder is not None and _binds(outer, binder, folder):
                folder, document = outer, binder
        route = document.get("route")
        return (document["name"].strip(),
                route if isinstance(route, str) else None)
    for ancestor in frame.parents:
        if ancestor.name and ancestor.name not in _LAYOUT_FOLDERS:
            return ancestor.name, None
    return frame.parent.name, None


def _parent_label(frame: Path) -> str:
    """The parent name a child is called after (see :func:`_parent_identity`)."""

    return _parent_identity(frame)[0]


def child_run_name(frame: Path, *, grid_id: int, ratio: int, dx: float) -> str:
    """``Downscale of <parent> · d03 ×3 · 1.33 km``: the run browser's name.

    ``<parent>`` is the parent run's own name as its manifest gives it
    (see :func:`_parent_identity`).  A parent that is itself a downscale
    already reads ``Downscale of <run> · d02 ×3 · 4 km``, so its child
    extends that name with its own grid rather than wrapping it in a
    second "Downscale of".

    Three significant figures on the spacing, so a grandchild at a third
    of 4 km reads as 1.33 km rather than a long fraction, and 4 km stays
    ``4 km``.
    """
    grid = (f"d{int(grid_id):02d} ×{int(ratio)} "
            f"· {float(dx) / 1000.0:.3g} km")
    parent, route = _parent_identity(frame)
    if route == "downscale":
        return f"{parent} · {grid}"
    return f"Downscale of {parent} · {grid}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_receipt(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise OfflineChildContractError(
            f"offline-child input is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _verify_file_receipts(
        receipts: list[dict[str, object]], *, label: str) -> None:
    for receipt in receipts:
        path = Path(str(receipt["path"]))
        observed = _file_receipt(path)
        if observed["bytes"] != receipt["bytes"] or (
                observed["sha256"] != receipt["sha256"]):
            raise OfflineChildContractError(
                f"{label} changed while the offline child was running: {path}")


def _exact_steps(seconds: float, dt: float, label: str) -> int:
    raw = float(seconds) / float(dt)
    rounded = int(round(raw))
    if rounded < 1 or not np.isclose(raw, rounded, rtol=0.0, atol=1e-8):
        raise OfflineChildContractError(
            f"{label}/dt must be a positive integer, got {raw}: the "
            f"child integrates in whole steps of dt = {float(dt):g} s and "
            f"would never land on that instant; set {label} to a value that is "
            f"a whole multiple of dt")
    return rounded


def checkpoint_schedule(steps: int, restart_steps: int | None) -> frozenset:
    """The step indices a child writes a checkpoint at.

    Every whole multiple of ``restart_steps`` inside the run, and the last
    step always: a run with ``restart_interval_s`` unset or zero writes
    exactly one checkpoint, at its end, which is what it always wrote,
    now under a name the next downscale can discover.
    """
    due = {int(steps)}
    if restart_steps is not None and int(restart_steps) > 0:
        due.update(range(int(restart_steps), int(steps) + 1,
                         int(restart_steps)))
    return frozenset(due)


@dataclass(frozen=True)
class ChildCadence:
    """Every step count a child integrates on, as whole steps of ``dt``."""

    steps: int
    output_steps: int
    restart_steps: int | None
    health_steps: int | None
    checkpoint_due: frozenset


def child_cadence(cfg, *, health_interval_seconds: float | None = None
                  ) -> ChildCadence:
    """Check the child's clock once, for the plan review and the run alike.

    ``run_seconds``, ``output_interval_s`` and ``restart_interval_s`` must
    each be a whole number of ``dt`` steps, and so must the health interval
    when the caller has one.  A derived child's step is chosen to land on
    every one of these (``woof.downscale.child_clock_step``: the largest
    step no longer than the parent's ``dt`` over the ratio that they are
    all whole numbers of); a hand-written ``--child-config`` can break
    it, and that used to be
    found only at run start.  ``woof downscale`` reviews with this
    function and the runner integrates on its answer, so the two doors
    cannot disagree about the same clock.  ``restart_interval_s`` unset or
    zero means one checkpoint, at the end.
    """
    steps = _exact_steps(cfg.run_seconds, cfg.dt, "run_seconds")
    output_steps = _exact_steps(
        cfg.output_interval_s, cfg.dt, "output_interval_s")
    restart_steps = (
        _exact_steps(cfg.restart_interval_s, cfg.dt, "restart_interval_s")
        if float(cfg.restart_interval_s) > 0.0 else None)
    health_steps = (
        None if health_interval_seconds is None else
        _exact_steps(health_interval_seconds, cfg.dt,
                     "health_interval_seconds"))
    return ChildCadence(
        steps=steps, output_steps=output_steps, restart_steps=restart_steps,
        health_steps=health_steps,
        checkpoint_due=checkpoint_schedule(steps, restart_steps))


def child_checkpoint_retention(requested: int | None = None) -> int | None:
    """How many complete checkpoint sets a child keeps; ``None`` keeps every one.

    ``requested`` is ``--keep-checkpoints`` (0 keeps every set).  Without
    it, the run-plan knob decides when it is set
    (:data:`woof.resume.KEEP_CHECKPOINTS_ENV`, which ``woof run-plan``
    and ``woof go`` export for what they start), and otherwise the one set
    every forecast route keeps (:data:`woof.resume.DEFAULT_KEEP_CHECKPOINTS`).

    One set is all a child needs: a child is never resumed (it is re-run,
    :func:`woof.resume.offline_child_resume_refusal`), and a downscale
    from this child binds its physics from the newest set.  Keeping every
    hourly set was the breakage: an 11 hour 250 m child kept 11 sets of
    2.59 GB, 28.5 GB beside 14 GB of history, with nothing warning first.
    """
    from woof.resume import (
        DEFAULT_KEEP_CHECKPOINTS, KEEP_CHECKPOINTS_ENV, checkpoint_retention)

    if requested is not None:
        if int(requested) < 0:
            raise OfflineChildContractError(
                f"--keep-checkpoints {requested} is negative; 0 keeps every "
                "checkpoint set")
        return int(requested) or None
    if os.environ.get(KEEP_CHECKPOINTS_ENV, "").strip():
        try:
            return checkpoint_retention()
        except ValueError as error:
            raise OfflineChildContractError(str(error)) from None
    return DEFAULT_KEEP_CHECKPOINTS


class ChildCheckpoints:
    """A child's checkpoint sets: each written whole, then the older ones removed.

    A child's set is one file, its own grid.  ``write`` publishes one set
    at the path it is handed (:func:`woof.io.restart.write_restart`
    writes a temporary file and renames it), and only then are the sets
    beyond the newest ``keep`` removed (``None`` keeps every one), so the
    directory always holds one complete set and holds at most ``keep`` + 1
    while a set is being written.
    """

    def __init__(self, outdir: Path, *, grid_id: int, keep: int | None,
                 write) -> None:
        self.outdir = Path(outdir)
        self.grid_id = int(grid_id)
        self.keep = keep
        self._write = write
        self.written: list[Path] = []
        self.retired: list[Path] = []

    def emit(self, valid_time) -> tuple[Path, list[Path]]:
        """Write the set valid at ``valid_time``; returns it and the sets it retired."""
        from woof.io.restart import restart_filename
        from woof.resume import retire_superseded_checkpoints

        path = Path(self._write(self.outdir / restart_filename(
            valid_time, domain=f"d{self.grid_id:02d}")))
        self.written.append(path)
        # 0 is "keep every set" to the retirer; None would read the
        # environment again instead of the answer this run settled on.
        retired = retire_superseded_checkpoints(self.outdir, self.keep or 0)
        self.retired.extend(retired)
        return path, retired

    @property
    def last(self) -> Path | None:
        return self.written[-1] if self.written else None

    def on_disk(self) -> list[Path]:
        return [path for path in self.written if path.exists()]


def child_history_frames(cadence: ChildCadence) -> int:
    """How many history frames a child on ``cadence`` writes.

    The initial frame, every whole ``output_interval_s``, and the last
    step when the run does not end on one.
    """
    steps, every = int(cadence.steps), int(cadence.output_steps)
    return 1 + steps // every + (1 if steps % every else 0)


def child_disk_remedy(projection: dict) -> str:
    """The ways out of a child that would not fit on its disk, in this door's own flags.

    Only the ways that shrink THIS projection are named.  Fewer products
    are offered whenever the child draws pictures: the shared per-product
    table also prices fewer products when no renderer catalog is available.
    Fewer checkpoint sets are offered only when it holds more than the
    two that keeping one needs.  Naming a flag that leaves the figure where
    it is sent a user whose child drew nothing back to --render-products
    for the same refusal.
    """
    ways = ["Free some disk", "pick a smaller child (--child-size) or a shorter one (--hours)",
            "write its history less often (--output-interval-seconds)"]
    if projection.get("picture_bytes"):
        ways.append("draw fewer products (--render-products; none draws nothing)")
    if int(projection.get("checkpoint_sets_held") or 0) > 2:
        ways.append("keep one checkpoint set (--keep-checkpoints 1)")
    ways.append("put --out on a disk with more room")
    return ", ".join(ways[:-1]) + ", or " + ways[-1]


def child_disk_projection(cfg, cadence: ChildCadence, *,
                          keep_checkpoints: int | None, render_products,
                          outdir: Path, history_selection=None) -> dict:
    """What this child will write, against the free space where it will write it.

    ONE function for the plan review and the run, as :func:`child_cadence`
    is: ``woof downscale`` puts this block in the plan and refuses on it
    before the child starts, and the runner door, which no plan review
    stands in front of, refuses on the same answer.  ``refusal`` is the
    sentence when the projection is larger than the free space, else None.
    Pictures use the same measured per-product, horizontal-grid table as
    forecasts. ``pictures_per_frame`` reports the catalog count when it
    is available; it does not choose the pricing model.
    """
    from woof import disk_budget

    projection = disk_budget.projected_child_bytes(
        cfg, history_frames=child_history_frames(cadence),
        checkpoints_written=len(cadence.checkpoint_due),
        keep_checkpoints=keep_checkpoints,
        render_products=render_products, history_selection=history_selection)
    free = disk_budget.free_bytes(Path(outdir))
    refusal = disk_budget.disk_refusal(
        projection, free, subject="this child", remedy=child_disk_remedy(projection))
    return dict(projection, keep_checkpoints=int(keep_checkpoints or 0),
                free_bytes=free,
                fits=None if free is None else refusal is None,
                refusal=refusal)


def _memory_snapshot(cp) -> dict[str, int]:
    pool = cp.get_default_memory_pool()
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    return {
        "pool_used_bytes": int(pool.used_bytes()),
        "pool_reserved_bytes": int(pool.total_bytes()),
        "device_free_bytes": int(free_bytes),
        "device_total_bytes": int(total_bytes),
    }


def _child_boundary_clock(cfg, *, lbc_interval_seconds: float, steps: int,
                          output_steps: int):
    """One bound DomainClock for the standalone child (Davies bind era).

    The production tree binds the root's DomainClock to the external LBC
    mirror so every Davies consumer takes WRF's post-increment ``dtbc``
    recurrence (dyn_em/solve_em.F:371-372) with interval selection from
    the solve-entry time.  The offline child is its own root, so it
    constructs the same integer-tick clock from the child config and the
    proven parent cadence; the runner drives the executor's exact
    per-step recurrence (seam reset -> prepare_step -> solve -> advance).
    """
    from fractions import Fraction
    import math

    from woof.core.clock import DomainClock, DomainTicks

    dt = Fraction(cfg.dt).limit_denominator(1_000_000)
    if float(dt) != float(cfg.dt):
        raise OfflineChildContractError(
            f"child dt={cfg.dt!r} is not exactly rational within 1e-6; "
            "the bound boundary clock requires an exact tick lattice")
    interval = Fraction(lbc_interval_seconds).limit_denominator(1_000_000)
    if float(interval) != float(lbc_interval_seconds):
        raise OfflineChildContractError(
            f"parent cadence {lbc_interval_seconds!r} s is not exactly "
            "rational within 1e-6")
    tick_den = math.lcm(dt.denominator, interval.denominator)
    step_ticks = int(dt * tick_den)
    interval_ticks = int(interval * tick_den)
    if interval_ticks % step_ticks != 0:
        raise OfflineChildContractError(
            f"parent cadence {lbc_interval_seconds:g} s is not a whole "
            f"number of child steps (dt={cfg.dt:g} s); the boundary seam "
            "must fall on a child step boundary")
    spec = DomainTicks(
        grid_id=int(cfg.grid_id), parent_id=0, parent_time_step_ratio=1,
        step_ticks=step_ticks, dt_fp32=np.float32(cfg.dt),
        history_ticks=int(output_steps) * step_ticks,
        restart_ticks=None, radt_ticks=None, stepra=None,
        cudt_ticks=None, stepcu=None, bldt_ticks=None, stepbl=None,
        lbc_interval_ticks=interval_ticks)
    return DomainClock(spec, tick_den, int(steps) * step_ticks)


def _initialize_child_physics(child, cfg, initial, surface, start_time, *,
                              terrain_drag_static=None):
    """Attach the child physics driver with an accurate warm start.

    mp-only children keep the established default initialization.  A
    radiation scheme needs the child latitude/longitude and UTC start
    time (SINT of the parent's XLAT/XLONG unless the surface source
    carries the child's own).  Land-surface/surface-layer/PBL schemes
    require a child-grid surface source: soil state and land identity
    are never fabricated from scalar defaults on a real-data child.
    """
    from woof.core.physics import initialize_physics
    from woof.static.orographic import required_static_fields
    from woof.ingest.ruc_mosaic import ruc_mosaic_physics_inputs
    from woof.ingest.lake_physics import lake_physics_inputs

    drag_fields = required_static_fields(getattr(cfg, "topo_wind", 0),
                                        getattr(cfg, "gwd_opt", 0))
    drag_kwargs = {}
    if drag_fields:
        missing = [name for name in drag_fields
                   if terrain_drag_static is None or name not in terrain_drag_static]
        if missing:
            raise OfflineChildContractError(
                f"the downscaled child's terrain drag needs static fields {missing}; "
                "build its own geography with the terrain-drag options set.")
        drag_kwargs["terrain_drag_static"] = terrain_drag_static

    needs_surface = bool(cfg.sf_surface_physics or cfg.sf_sfclay_physics
                         or cfg.bl_pbl_physics)
    ra_lw, ra_sw = radiation_scheme_ids(cfg)
    radiation_active = bool(ra_lw or ra_sw)
    if needs_surface and surface is None:
        # The predicate and the sentence live in one place
        # (offline_child.child_surface_requirement) so the front door's
        # early refusal and this late guard cannot drift apart.
        raise OfflineChildContractError(child_surface_requirement(cfg))
    if surface is None and not radiation_active:
        return initialize_physics(child, cfg, **drag_kwargs)

    if surface is not None and "XLAT" in surface.fields:
        lat = np.asarray(surface.fields["XLAT"], dtype=np.float64)
        lon = np.asarray(surface.fields["XLONG"], dtype=np.float64)
    else:
        lat = np.asarray(initial.fields["XLAT"], dtype=np.float64)
        lon = np.asarray(initial.fields["XLONG"], dtype=np.float64)

    radiation = None
    if 4 in radiation_scheme_ids(cfg):
        from woof.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
        if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
            from woof.core.cam_ozone import ROUTING_CHILD_GRID_CLIMATOLOGY
            from woof.core.radiation_composition import make_radiation
            # OZONE ON THIS ROUTE, under o3input = 2.
            #
            # There is no resident parent to interpolate from: this is the
            # ndown-equivalent offline path and it stamps parent_id = 0 on
            # its own DomainTicks precisely because no parent domain is in
            # memory.  What WRF does for exactly this domain is the answer,
            # and WRF's answer is the climatology on the domain's own grid:
            # an offline child is configured as a WRF ROOT (specified =
            # true, nested = false, lateral boundaries read from a file),
            # and for id == 1 WRF calls oznini on that domain's own XLAT
            # (phys/module_physics_init.F:2203-2212) and evaluates
            # ozn_time_int/ozn_p_int on its own columns
            # (phys/module_radiation_driver.F:1801-1823).  Only a resident
            # NEST is handed the parent's field, through the rdf=(p2c)
            # forcing stream the Registry declares on o3rad
            # (Registry/Registry.EM_COMMON:1264).
            #
            # So the constructor takes its root branch, which is the
            # bit-ported chain on the child grid, and the route DECLARES
            # what that is.  The declaration is the whole difference from
            # what this did before the refusal was installed: it used to
            # evaluate the same field and then report
            # "ozone_routing": "root-climatology", which told a reader the
            # domain was a root rather than a refinement of an archived
            # forecast.  It now reports "child-grid-climatology", in the
            # radiation identity that reaches every checkpoint manifest and
            # in the run's report.json.
            radiation = make_radiation(
                cfg, start_time, lat, lon,
                p_top=float(initial.receipt["p_top"]),
                ozone_routing=(ROUTING_CHILD_GRID_CLIMATOLOGY
                               if cfg.o3input == 2 else None))

    if surface is None:
        return initialize_physics(
            child, cfg, radiation=radiation,
            radiation_start_time=start_time,
            radiation_latitude=lat, radiation_longitude=lon, **drag_kwargs)

    from woof.core.landuse import (initialize_landuse, ruc_fractional_seaice,
                                    usemonalb_landuse_inputs)
    fields = surface.fields
    identity = surface.identity
    xice = fields.get("SEAICE", fields.get("XICE"))
    if xice is None:
        xice = np.zeros_like(fields["LANDMASK"])
    # usemonalb's monthly ALBBCK and SNOALB come from the child's own static
    # build (ALBEDO12M interpolated to the start date, real.exe
    # module_initialize_real.F:1192, :1235-1239), exactly as the case-data
    # roads take them.  A child-grid file or a parent-derived surface carries
    # no monthly albedo, and running the table albedo under the switch
    # would be the configuration it replaces, so that is refused by name.
    albedo_inputs = {}
    if bool(getattr(cfg, "usemonalb", False)):
        if terrain_drag_static is None or any(
                terrain_drag_static.get(name) is None
                for name in ("ALBEDO12M", "SNOALB", "LANDMASK")):
            raise OfflineChildContractError(
                "usemonalb=true on a downscaled child needs the monthly "
                "background albedo ALBEDO12M and the snow albedo SNOALB of "
                "the child's own static build; this child's surface "
                "source (a child-grid file or the parent-derived surface) "
                "carries neither, so landuse_init would run LANDUSE.TBL's "
                "seasonal albedo under a switch that names the monthly "
                "field. Build the child's own geography, or set "
                "usemonalb=false")
        albedo_inputs = usemonalb_landuse_inputs(
            cfg, terrain_drag_static, start_time)
    landuse = initialize_landuse(
        fields["LU_INDEX"], soil_type=fields["ISLTYP"],
        urban_legend=int(getattr(cfg, "sf_urban_physics", 0)) > 0,
        landmask=fields["LANDMASK"], snow=fields["SNOW"], xice=xice,
        valid_time=start_time, cen_lat=float(np.mean(lat)),
        mminlu=str(identity["MMINLU"]), iswater=int(identity["ISWATER"]),
        islake=int(identity["ISLAKE"]), isice=int(identity["ISICE"]),
        isoilwater=int(identity["ISOILWATER"]),
        # real.exe's landmask/soil-category reconciliation decides a
        # disagreeing column from its soil temperature, then its SST.
        soil_temperature=fields["TSLB"], sst=fields.get("SST"),
        # landuse_init reads the threshold the surface runs
        # (module_physics_init.F:1463-1465): 0.02 under fractional_seaice
        # = 1, the value the RUC seam and the lake take on this road too.
        fractional_seaice=ruc_fractional_seaice(cfg),
        **albedo_inputs)
    driver = initialize_physics(
        child, cfg, landuse=landuse, tsk=fields["TSK"],
        landuse_dataset=str(identity["MMINLU"]),
        soil_temperature=fields["TSLB"], soil_moisture=fields["SMOIS"],
        liquid_moisture=fields.get("SH2O"),
        ivgtyp=fields["LU_INDEX"], isltyp=fields["ISLTYP"],
        vegfra=fields["VEGFRA"], tmn=fields["TMN"], xice=xice,
        snow=fields["SNOW"],
        snow_depth=fields.get("SNOWH", np.zeros_like(fields["SNOW"])),
        pblh=fields.get("PBLH", 0.0),
        radiation=radiation, radiation_start_time=start_time,
        radiation_latitude=lat, radiation_longitude=lon,
        **ruc_mosaic_physics_inputs(
            cfg, fields if terrain_drag_static is None else terrain_drag_static,
            landuse_attrs=identity, xice=xice,
            processed=terrain_drag_static is None,
            fractional_seaice=ruc_fractional_seaice(cfg)),
        **lake_physics_inputs(
            cfg, fields if terrain_drag_static is None else terrain_drag_static),
        **drag_kwargs)
    # Seed time-zero surface diagnostics from the child-grid source; the
    # first model step replaces them through SFCLAY/LSM/PBL in WRF
    # ordering (same convention as the experiment path's warm seed).
    import cupy as cp
    # A child on its OWN geography carries the land-model fields a
    # standalone run of its grid initializes from its static build (leaf
    # area on the start date, the greenness range, the snow albedo), the
    # way woof.ingest.hrrr_physics seeds them; a surface that carries
    # none (a child-grid file, the parent-interpolated surface) leaves the
    # driver's own initialization, as before.
    for source_name, field_name in (("LAI", "lai"), ("SHDMIN", "shdmin"),
                                    ("SHDMAX", "shdmax")):
        value = fields.get(source_name)
        if value is not None and field_name in driver.fields:
            driver.fields[field_name][...] = cp.asarray(
                value, dtype=cp.float32)
    if (int(cfg.sf_surface_physics) == 3 and cfg.rdlai2d
            and terrain_drag_static is not None
            and terrain_drag_static.get("LAI12M") is not None
            and "lai" in driver.fields):
        from woof.core.landuse import surface_leaf_area
        driver.fields["lai"][...] = cp.asarray(
            surface_leaf_area(cfg, terrain_drag_static["LAI12M"], start_time),
            dtype=cp.float32)
    if (int(cfg.sf_surface_physics) == 3 and cfg.usemonalb
            and "snoalb" in driver.fields):
        from woof.core.landuse import surface_snow_albedo
        driver.fields["snoalb"][...] = cp.asarray(
            surface_snow_albedo(cfg, terrain_drag_static, None), dtype=cp.float32)
    noah_params = getattr(driver, "noah_params", None)
    if (fields.get("SNOALB") is not None and noah_params is not None
            and "snoalb" in driver.fields):
        from woof.core.noah import noah_initial_snow_albedo
        driver.fields["snoalb"][...] = cp.asarray(
            noah_initial_snow_albedo(
                fields["SNOALB"], np.rint(fields["LU_INDEX"]).astype(np.int64),
                noah_params, rdmaxalb=cfg.rdmaxalb),
            dtype=cp.float32)
    for source_name, field_name in (
            ("PSFC", "psfc"), ("T2", "t2"), ("Q2", "q2"), ("TH2", "th2"),
            ("U10", "u10"), ("V10", "v10"), ("UST", "ust")):
        value = fields.get(source_name)
        if value is not None and field_name in driver.fields:
            driver.fields[field_name][...] = cp.asarray(
                value, dtype=cp.float32)
    return driver


def _child_ozone_routing(driver):
    """Where this child's radiation took its ozone from, for the report.

    One of :data:`woof.core.cam_ozone.OZONE_ROUTINGS`, or ``None`` when the
    child's radiation carries no ozone routing at all: no radiation, or a
    scheme that builds its own gas profile rather than reading the WRF CAM
    climatology.  ``None`` is a third state and is reported as one, because
    a missing key would read as "not recorded" for a run that has an answer.
    """
    from woof.core.radiation_composition import legacy_radiation_adapter
    scheme = getattr(driver, "radiation_callable", None)
    adapter = None if scheme is None else legacy_radiation_adapter(scheme)
    return None if adapter is None else adapter.ozone_routing


def _jsonable(value):
    """Plain JSON types for a receipt built from mappings, tuples and numpy."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _create_output_root(path: Path) -> Path:
    """Reserve one output tree without ever adopting a prior run's."""
    return reserve_output_root(path, flag="--outdir")


def _survey_nonfinite_child(streaming, stepper, child) -> dict:
    """The field survey behind a non-finite refusal, which cannot raise.

    THE WHOLE POINT of this survey is that a diagnostic died in front
    of the refusal it was decorating: 2.7.5 coerced a ``None`` CFL with
    ``float()`` in the health line immediately before "offline child
    became non-finite at step N" (``offline_child_run.py`` lines 1460
    and 1477 of the 2.7.5 tree) and the reader got a ``TypeError``
    traceback instead of the sentence.  A survey that can fail -- a device out of memory on a card
    that just filled itself, a carrier in a shape this does not expect --
    must therefore fail INTO the capsule and never over it.  A survey
    that could not be taken says so in the capsule, with its own error,
    and the refusal is raised either way.
    """

    try:
        streaming.refresh_streamed_state(stepper, child)
        from woof.core.dycore import nonfinite_field_survey

        return nonfinite_field_survey(child)
    except BaseException as error:  # noqa: BLE001 - see the docstring
        return {"fields": [], "surveyed": [],
                "error": " ".join(f"{type(error).__name__}: {error}".split())}


#: What a health reading IS, carried beside the reading itself.  A
#: number is ``"measured"``; a reading nothing ever produced is
#: ``"not computed"``; a reading that exists and is not finite is
#: ``"non-finite"``.  For both non-numbers the reading itself is
#: ``null``, which is what keeps every document this outcome writes
#: openable by a strict JSON reader.
READING_MEASURED = "measured"
READING_NOT_COMPUTED = "not computed"
READING_NON_FINITE = "non-finite"

#: The two spellings that are not a quantity, so not a thing a unit can
#: be put after.  One tuple, read by the row renderer and by the state
#: preserver, because a guard written against one spelling and not the
#: other is the defect this replaces.
READING_STATES_WITHOUT_A_NUMBER = (READING_NON_FINITE, READING_NOT_COMPUTED)


def _health_reading(value):
    """One health number as ``(value, state)``, never a non-finite float.

    The distinction is kept because it IS the reading: ``"non-finite"``
    is a number that went, and ``"not computed"`` is a number nothing
    ever produced (:func:`woof.core.dycore.decode_stability_record`
    computes no CFL from fields that are not finite).  Flattening both to
    ``null`` would say the same thing about a field that blew up and a
    field nobody measured, which on this outcome is the whole reading.
    """

    if value is None:
        return None, READING_NOT_COMPUTED
    value = float(value)
    if not math.isfinite(value):
        return None, READING_NON_FINITE
    return value, READING_MEASURED


def child_health_log_fields(record) -> dict:
    """The health numbers one child step records, from its stability record.

    THE CARRYING SHAPE.  ``w_max`` and ``cfl`` are each a number or
    ``null``, and ``w_max_state`` / ``cfl_state`` beside them is
    ``"measured"``, ``"non-finite"`` or ``"not computed"``.  A non-finite
    reading never leaves here as a float, so the ``child_step`` event
    line and ``report.json`` are strict JSON on exactly the run that
    produces one, and the human row keeps both meanings rather than
    printing the same null twice.

    ``cfl`` travels as ``None``, never through ``float()``: the record's
    ``cfl`` IS ``None`` by construction whenever ``nan`` is true --
    :func:`woof.core.dycore.decode_stability_record` computes no CFL
    from fields that are not finite -- and coercing it raised

        TypeError: float() argument must be a string or a real number,
        not 'NoneType'

    in the health line immediately before the refusal that names the
    breakage -- lines 1460 and 1477 of the 2.7.5 tree.  So a child
    that went non-finite, which is the one event this record exists to
    report, died with that traceback at exit 1 instead of with the
    capsule :func:`describe_nonfinite_child` composes -- which reads this
    record's ``w_max`` and ``cfl`` back over the last several checks, so
    the ``None`` this function is careful to pass through is a value the
    refusal prints as "not computed" rather than a hole in the trend.
    """

    cfl, cfl_state = _health_reading(record["cfl"])
    w_max, w_max_state = _health_reading(record["w_max"])
    return {"nan": bool(record["nan"]),
            "cfl": cfl, "cfl_state": cfl_state,
            "w_max": w_max, "w_max_state": w_max_state}


def w_max_location(record, *, nz: int, ny: int, nx: int) -> dict | None:
    """Where one health record's |w| maximum is, and how far from an edge.

    ``record["w_argmax"]`` is the flat index of the maximum on w's own
    ``(nz + 1, ny, nx)`` grid, which both health kernels reduce on every
    check (:func:`woof.core.dycore.decode_stability_record`).  Returned
    as ``{"cell": {"k", "j", "i"}, "edge": name, "edge_cells": n}``: the
    lateral edge nearest the cell and how many cells in from it it sits,
    0 being the boundary row itself.  ``None`` when the record carries no
    finite maximum or no index inside the grid, because a place for a
    number that went is not a place.
    """

    w_max = record.get("w_max")
    index = record.get("w_argmax")
    if index is None or w_max is None:
        return None
    try:
        if not math.isfinite(float(w_max)):
            return None
        nz, ny, nx, index = int(nz), int(ny), int(nx), int(index)
    except (TypeError, ValueError):
        return None
    if ny <= 0 or nx <= 0 or not 0 <= index < (nz + 1) * ny * nx:
        return None
    k, rest = divmod(index, ny * nx)
    j, i = divmod(rest, nx)
    edge, cells = min((("south", j), ("north", ny - 1 - j),
                       ("west", i), ("east", nx - 1 - i)),
                      key=lambda item: item[1])
    return {"cell": {"k": k, "j": j, "i": i}, "edge": edge,
            "edge_cells": cells}


def child_health_trend_row(*, step, model_seconds, record, cfg) -> dict:
    """One health check as the run loop keeps it for the capsule.

    Both readings in the carrying shape :func:`child_health_log_fields`
    produces, plus WHERE the |w| maximum was (``w_max_cell``) and the
    lateral edge nearest it (``w_max_edge``), both ``null`` on a check
    whose maximum was not measured.  The place is kept because it is the
    one location a blow-up leaves that means anything: the survey taken
    after the fields have gone can only box what had already spread, and
    the last measured maximum's place is the nearest thing to where the
    climb started.
    """

    fields = child_health_log_fields(record)
    where = (w_max_location(record, nz=cfg.nz, ny=cfg.ny, nx=cfg.nx)
             if fields["w_max_state"] == READING_MEASURED else None)
    return {"step": int(step),
            "model_seconds": float(model_seconds),
            "w_max": fields["w_max"],
            "w_max_state": fields["w_max_state"],
            "cfl": fields["cfl"],
            "cfl_state": fields["cfl_state"],
            "w_max_cell": None if where is None else where["cell"],
            "w_max_edge": (None if where is None else
                           {"edge": where["edge"],
                            "cells": where["edge_cells"]})}


#: How far back the non-finite capsule reads the health record, in MODEL
#: seconds.  Five minutes: long enough that a reader is shown a trend
#: rather than two samples, short enough that every number quoted belongs
#: to the blow-up rather than to the calm hour before it.
NONFINITE_TREND_SECONDS = 300.0

#: The most checks that window is ever spelled out as, whatever the
#: cadence.  A one-second health cadence would otherwise quote 301 rows
#: into a capsule a run view shows a few lines of.
NONFINITE_TREND_CHECKS_MAX = 12


def nonfinite_trend_checks(cadence_seconds) -> int:
    """How many health checks cover the window, FROM the record's cadence.

    The window is a duration, not a row count: at the default 60-second
    cadence it is six checks and at a 300-second one it is two, and both
    cover the same five minutes of model time.  Deriving it from the
    cadence is what keeps the sentence "over the last 300 model seconds"
    true for a run that chose its own ``--health-interval-seconds``.

    The capsule quotes this many checks PLUS the one that found the
    fields gone, which is the row the window is measured back from and
    carries no w_max and no CFL of its own.
    """

    try:
        cadence = float(cadence_seconds)
    except (TypeError, ValueError):
        return 2
    if not math.isfinite(cadence) or cadence <= 0.0:
        return 2
    return max(2, min(NONFINITE_TREND_CHECKS_MAX,
                      int(math.ceil(NONFINITE_TREND_SECONDS / cadence)) + 1))


def _number(value, digits: int = 4) -> str:
    """One measured number, at the precision a reader can check it to."""

    if value is None:
        return "not computed"
    value = float(value)
    if not math.isfinite(value):
        return "non-finite"
    return f"{value:.{digits}g}"


def _reading(row, name: str, digits: int = 4) -> str:
    """One health reading out of a trend row, as a reader reads it.

    The state word wins when the row carries one, because the value
    beside it is ``null`` for BOTH non-numbers and the value alone can no
    longer tell them apart.  A row built before the state words existed,
    or by hand, still reads correctly: :func:`_number` spells a bare
    ``None`` "not computed" and a bare non-finite float "non-finite".
    """

    state = row.get(f"{name}_state")
    if state in READING_STATES_WITHOUT_A_NUMBER:
        return state
    return _number(row.get(name), digits)


def _health_row(row) -> dict:
    """One trend row with both readings in the carrying shape.

    :func:`describe_nonfinite_child` is handed the run loop's own trend,
    whose rows already come from :func:`child_health_log_fields` -- and
    it is also the composer a test, a sibling door or a later route can
    call with rows of its own.  Normalising here is what makes the
    capsule a document that cannot carry a non-finite float whoever built
    the row, which is the property ``report.json``, the ``child_step``
    line and the run-plan ``failed`` event are all written under.
    """

    row = dict(row)
    for name in ("w_max", "cfl"):
        if name not in row and f"{name}_state" not in row:
            continue
        value, state = _health_reading(row.get(name))
        given = row.get(f"{name}_state")
        # A row that already carries its state word keeps it: with the
        # value null for both non-numbers, re-deriving the state from the
        # value alone would turn every "non-finite" row into a
        # "not computed" one.
        if value is None and given in READING_STATES_WITHOUT_A_NUMBER:
            state = given
        row[name] = value
        row[f"{name}_state"] = state
    return row


def _plain_list(names) -> str:
    """``[a, b, c]`` as ``a, b and c``; one name as itself."""

    names = [str(name) for name in names]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _survey_box(entry: dict) -> dict:
    """A surveyed carrier's bounding box, axes in (k, j, i) order."""

    box = {str(label): [int(bounds[0]), int(bounds[1])]
           for label, bounds in (entry.get("bounding_box") or {}).items()}
    order = {"k": 0, "j": 1, "i": 2}
    return dict(sorted(box.items(), key=lambda item: order.get(item[0], 3)))


def _survey_edges(entry: dict) -> list[str]:
    """The lateral edges a surveyed box reaches, on the carrier's own grid.

    Read from the entry when the survey wrote them, and derived from its
    box and shape when it did not, so an entry built by hand or by an
    older survey is judged the same way.
    """

    from woof.core.dycore import nonfinite_box_edges

    if entry.get("edges") is not None:
        return [str(edge) for edge in entry["edges"]]
    return nonfinite_box_edges(entry.get("bounding_box"), entry.get("shape"))


def _box_text(box: dict) -> str:
    return ", ".join(f"{label} {bounds[0]}-{bounds[1]}"
                     for label, bounds in box.items())


def _box_cell(box: dict) -> dict | None:
    """The one cell a box is, when it is one cell; otherwise ``None``."""

    if not box or any(bounds[0] != bounds[1] for bounds in box.values()):
        return None
    return {label: bounds[0] for label, bounds in box.items()}


def _edge_clause(edges, *, cell: bool) -> str:
    if not edges:
        return ""
    plural = "s" if len(edges) > 1 else ""
    if cell:
        return f", on the {_plain_list(edges)} edge{plural}"
    return f", a box that touches the {_plain_list(edges)} edge{plural}"


def _survey_line(entry: dict) -> str:
    """One surveyed carrier, as the capsule prints it.

    One cell is named and nothing else, because a bounding box around a
    single cell is that cell written twice.  Many cells are a count and
    the box they fall inside, which is the difference between a column
    that went, a plume that went, and a field that has gone entirely.  No
    single cell of many is named: which of them went first is not
    something a survey taken afterwards can know.
    """

    from woof.core.dycore import format_survey_cell

    box = _survey_box(entry)
    edges = _survey_edges(entry)
    cell = _box_cell(box)
    if int(entry["count"]) == 1 and cell is not None:
        return (f"{entry['field']}: 1 cell at {format_survey_cell(cell)}"
                + _edge_clause(edges, cell=True))
    return (f"{entry['field']}: {int(entry['count']):,} cells of "
            f"{int(entry['size']):,}, all inside {_box_text(box)}"
            + _edge_clause(edges, cell=False))


def _nonfinite_extent(fields) -> tuple[dict, list[str], int]:
    """The box every surveyed carrier's non-finite cells fall inside.

    The union of the carriers' own boxes, the edges any of them reaches,
    and the cells counted between them.  A staggered carrier's box is on
    its own grid, one wider along its stagger, so the union is the range
    of indices the non-finite values occupy and not a claim about any one
    field.
    """

    from woof.core.dycore import NONFINITE_SURVEY_EDGES

    box: dict = {}
    reached: set[str] = set()
    total = 0
    for entry in fields:
        total += int(entry["count"])
        reached.update(_survey_edges(entry))
        for label, (low, high) in _survey_box(entry).items():
            if label in box:
                box[label] = [min(box[label][0], low),
                              max(box[label][1], high)]
            else:
                box[label] = [low, high]
    order = {"k": 0, "j": 1, "i": 2}
    box = dict(sorted(box.items(), key=lambda item: order.get(item[0], 3)))
    edges = [name for _axis, _end, name in NONFINITE_SURVEY_EDGES
             if name in reached]
    return box, edges, total


def _found_text(fields) -> str:
    """What the check found, in the words the survey can stand behind.

    The fields, the box they fall inside and how many cells, and whether
    that box touches a lateral edge.  One cell is named only when the
    whole non-finite set IS one cell.
    """

    from woof.core.dycore import format_survey_cell

    if not fields:
        return "non-finite values"
    names = [str(entry["field"]) for entry in fields]
    box, edges, total = _nonfinite_extent(fields)
    cell = _box_cell(box)
    if cell is not None:
        return (f"{_plain_list(names)} non-finite at one cell, "
                f"{format_survey_cell(cell)}"
                + _edge_clause(edges, cell=True))
    if len(names) == 1:
        counted = f" in {total:,} cell{'s' if total != 1 else ''}"
    else:
        counted = f", {total:,} cells between them"
    return (f"{_plain_list(names)} non-finite{counted}, all inside "
            f"{_box_text(box)}" + _edge_clause(edges, cell=False))


def _edge_distance_text(edge: dict | None) -> str:
    if not edge:
        return ""
    cells = int(edge["cells"])
    if cells == 0:
        return f", on the {edge['edge']} edge"
    return (f", {cells} cell{'s' if cells != 1 else ''} in from the "
            f"{edge['edge']} edge")


def describe_nonfinite_child(*, step, total_steps, model_seconds,
                             run_seconds, cadence_seconds, trend, survey,
                             regime=None, render_command=None) -> dict:
    """The capsule a reader is handed when a child stops being finite.

    WHAT BREAKAGE THIS PREVENTS (gate law).  "offline child became
    non-finite at step 6624" was the whole of what a blown-up child said,
    and it is the one sentence that sends a reader back to re-run the
    thing to learn anything at all: it names no field, no place in the
    grid, no model time, and none of the trend the health record was
    already holding.  On the run this was built from, that record held
    six checks of w_max climbing 10.73, 13.22, 15.77, 18.12, 21.05,
    22.97 m/s over the five minutes before the end, with the CFL never
    leaving the 0.19-0.21 band -- a reading that says plainly the time
    step was not what ran out -- and every one of those numbers was
    dropped at the moment it mattered.

    WHAT THE SENTENCE CLAIMS, and no more.  The fields found non-finite
    at the health check after step N, the box they fall inside with its
    cell count, and whether that box touches a lateral edge.  It used to
    name "the cell W went non-finite at", which was the lowest
    memory-order index of the whole non-finite set: always that set's
    lowest level and southmost row, so a block that had spread down to
    the ground read as a blow-up at k=0 on its south edge, and W, which
    is simply first in the survey's list, read as the field that failed
    first.  A single cell is named only when the set is one cell.  The
    last measured |w| maximum's place, which the trend rows carry, is
    the nearest thing to an origin the record holds, and the message
    says it as that.

    Returns the capsule as a document: ``summary`` is the one sentence
    the run-plan ``failed`` event carries and a run view shows first,
    ``message`` is the whole refusal, and the rest are the same facts as
    fields, so the sentence and ``report.json`` cannot disagree.

    THE CARRYING SHAPE, which this document is written under and which
    :func:`child_health_log_fields` produces: every ``trend`` row holds
    ``w_max`` and ``cfl`` as a number or ``null``, with ``w_max_state``
    and ``cfl_state`` beside them spelling ``"measured"``,
    ``"non-finite"`` or ``"not computed"``.  A row handed in any other
    shape is normalised to it here (:func:`_health_row`), so no reading
    this capsule carries can reach a serializer as a non-finite float --
    which is what lets ``report.json`` and the ``child_step`` line be
    written with ``allow_nan=False``, the receipt convention of the rest
    of this tree, on precisely the outcome whose readings went.
    """

    quoted = [_health_row(row) for row in
              list(trend)[-(nonfinite_trend_checks(cadence_seconds) + 1):]]
    finite = [row for row in quoted
              if row.get("w_max_state") == READING_MEASURED]
    fields = list((survey or {}).get("fields") or [])
    survey_error = (survey or {}).get("error")
    box, edges, _total = _nonfinite_extent(fields)

    if len(finite) >= 2:
        span = (float(finite[-1]["model_seconds"])
                - float(finite[0]["model_seconds"]))
        lead = ("w_max ran "
                + ", ".join(_reading(row, "w_max") for row in finite)
                + f" m/s over the {span:g} model seconds before ")
    elif len(finite) == 1:
        lead = (f"w_max read {_reading(finite[0], 'w_max')} m/s one check "
                "before ")
    else:
        lead = ""
    # The step and the model second come BEFORE the survey's findings:
    # a run view shows one line of this sentence, and those are the two
    # facts the old refusal had.
    summary = (f"The child blew up: {lead}the health check after step "
               f"{int(step)} of {int(total_steps)} (model second "
               f"{_number(model_seconds, 8)} of {_number(run_seconds, 8)}) "
               f"found {_found_text(fields)}.")

    paragraphs = [summary]
    if survey_error:
        paragraphs.append(
            "The field survey could not be taken, so this capsule names no "
            f"cell: {survey_error}")
    elif fields:
        paragraphs.append(
            "Non-finite fields at that check, listed dynamics first and "
            "then moisture, which is not the order they failed in:\n"
            + "\n".join(f"  {_survey_line(entry)}" for entry in fields))
    else:
        paragraphs.append(
            "No allocated carrier was still non-finite when the survey ran, "
            "so the reading came from the health record's own maxima (u, w "
            "and theta') and nothing narrower.")

    before = [row for row in finite if int(row["step"]) < int(step)]
    located = [row for row in before if row.get("w_max_cell")]
    when = []
    if before:
        # u, w and theta' only: those three maxima are all the check
        # reads, so a moisture carrier the survey lists may have gone
        # long before and the sentence must not date it.
        previous = int(before[-1]["step"])
        when.append(
            f"The check after step {previous}, {int(step) - previous} "
            "steps earlier, found u, w and theta' finite; the check reads "
            "only those three, and a survey taken afterwards cannot say "
            "which cell or which field went first.")
    elif fields:
        when.append("A survey taken after the fields have gone cannot say "
                    "which cell or which field went first.")
    last_w_max = None
    if located:
        from woof.core.dycore import format_survey_cell

        row = located[-1]
        last_w_max = {"step": int(row["step"]),
                      "model_seconds": float(row["model_seconds"]),
                      "w_max": row["w_max"],
                      "cell": dict(row["w_max_cell"]),
                      "edge": (None if not row.get("w_max_edge")
                               else dict(row["w_max_edge"]))}
        when.append(
            f"The last |w| maximum measured, {_reading(row, 'w_max')} m/s "
            f"at the check after step {int(row['step'])}, was at "
            f"{format_survey_cell(row['w_max_cell'])}"
            f"{_edge_distance_text(row.get('w_max_edge'))}.")
    if when:
        paragraphs.append(" ".join(when))

    if quoted:
        rows = []
        for row in quoted:
            w_max = _reading(row, "w_max")
            # The unit rides the NUMBER, and the guard is on the RENDERED
            # text rather than on one spelling of it: "w_max non-finite
            # m/s" and "w_max not computed m/s" both read as a quantity
            # with a unit, and the whole point of either row is that
            # there is no quantity.  Written against the one spelling,
            # this guard covered the field that went and missed the field
            # nothing measured.
            unit = "" if w_max in READING_STATES_WITHOUT_A_NUMBER else " m/s"
            if unit and row.get("w_max_cell"):
                from woof.core.dycore import format_survey_cell

                unit += f" at {format_survey_cell(row['w_max_cell'])}"
            rows.append(
                f"  step {int(row['step'])}"
                f"  model second {_number(row['model_seconds'], 8)}"
                f"  w_max {w_max}{unit}"
                f"  CFL {_reading(row, 'cfl')}")
        cadence_text = (f", {_number(cadence_seconds, 6)} model seconds apart"
                        if cadence_seconds else "")
        paragraphs.append(
            f"The last {len(quoted)} health checks{cadence_text}:\n"
            + "\n".join(rows))

    if regime is not None:
        paragraphs.append("This child's shape: " + regime["statement"])

    if render_command:
        paragraphs.append(
            "Next: every frame the run did reach is on disk and can be "
            "drawn by hand:\n  " + render_command)
    else:
        paragraphs.append(
            "Next: every frame the run did reach is on disk; `woof render` "
            "draws them, and docs/public/DOWNSCALE.md says what a child "
            "that did not finish leaves behind.")

    return {
        "kind": "non-finite",
        "summary": summary,
        "message": "\n".join(paragraphs),
        "step": int(step),
        "total_steps": int(total_steps),
        "model_seconds": float(model_seconds),
        "run_seconds": float(run_seconds),
        "health_cadence_seconds": (None if cadence_seconds is None
                                   else float(cadence_seconds)),
        "fields": fields,
        "nonfinite_box": box or None,
        "nonfinite_edges": edges,
        "last_w_max": last_w_max,
        "surveyed": list((survey or {}).get("surveyed") or []),
        "survey_error": survey_error,
        "trend": quoted,
        "les_regime": regime,
        "render_command": render_command,
    }


class OfflineChildNonFinite(OfflineChildContractError):
    """A child whose own fields stopped being finite mid-integration.

    A RUN-TIME refusal, not an admission one, and it shares the contract
    error's class for one reason: that class is a ``ValueError``, and
    ``woof.cli`` prints a ``ValueError`` as one message at exit 2 with
    no traceback.  Raised as a bare ``RuntimeError`` this capsule would
    reach its reader as the tail of a stack trace, which is the shape the
    2.7.5 failure already had and the reason none of it was readable.

    ``capsule`` is the document :func:`describe_nonfinite_child` built and
    ``summary`` is its first sentence, which is what the run-plan
    ``failed`` event carries.
    """

    def __init__(self, capsule: dict) -> None:
        super().__init__(capsule["message"])
        self.capsule = dict(capsule)
        self.summary = str(capsule["summary"])


def _parent_grid_metadata(path: Path) -> tuple[float, float, dict[str, object]]:
    """Parent geometry, plus the lineage attributes the child inherits.

    A downscaled child's initial state is the parent's history, so the
    child's initial condition descends from whatever the parent's did.
    The parent's initial-condition provenance is therefore copied
    forward verbatim -- it describes the ROOT of the lineage, which is
    the fact a published child chart must not lose.  The child's own
    time zero stays in ``START_DATE``, where WRF puts it.  A parent that
    carries no provenance (a stock-WRF archive, or a pre-1.4.1 file)
    hands the child nothing, and the child says nothing rather than
    inventing an analysis.  Opened through :func:`open_parent_history`,
    so a frame that cannot be read is refused naming it.
    """
    with open_parent_history(path, netCDF4.Dataset) as dataset:
        try:
            dx = float(dataset.getncattr("DX"))
            dy = float(dataset.getncattr("DY"))
        except AttributeError as exc:
            raise OfflineChildContractError(
                f"{path} lacks authoritative DX/DY attributes") from exc
        present = set(dataset.ncattrs())
        attrs = {
            name: dataset.getncattr(name)
            for name in (*_PROJECTION_ATTRS, *INITIAL_CONDITION_GLOBAL_ATTRS)
            if name in present
        }
    return dx, dy, attrs


def _history_refl_10cm(state, cfg, ticks):
    """Consume mature reflectivity, or write WRF's tick-zero initial field."""
    from woof.core.refl import consume_refl_10cm, refl_10cm_is_stashed, analysis_refl_10cm
    from woof.core.physics_inventory import REFL_10CM_MICROPHYSICS
    if refl_10cm_is_stashed(state):
        return consume_refl_10cm(state)
    if ticks == 0 and state.qv is not None and cfg.mp_physics in REFL_10CM_MICROPHYSICS:
        return analysis_refl_10cm(state, cfg, shape=(cfg.nz, cfg.ny, cfg.nx))
    return None


def _output_fields(state, initial, refl_field=None,
                   surface=None) -> dict[str, np.ndarray]:
    import cupy as cp
    from woof.io.wrfout import state_frame

    result = state_frame(state, include_diagnostic_pressure=True)
    if refl_field is not None:
        result["REFL_10CM"] = cp.asnumpy(refl_field)
    result.update({
        "MAPFAC_M": cp.asnumpy(state.msft),
        "MAPFAC_U": cp.asnumpy(state.msfu),
        "MAPFAC_V": cp.asnumpy(state.msfv),
        "F": cp.asnumpy(state.f),
        "E": cp.asnumpy(state.e),
        "SINALPHA": cp.asnumpy(state.sina),
        "COSALPHA": cp.asnumpy(state.cosa),
        "XLAT": np.asarray(initial.fields["XLAT"], dtype=np.float32),
        "XLONG": np.asarray(initial.fields["XLONG"], dtype=np.float32),
    })
    if surface is not None:
        # The two static land fields the experiment routes take from the
        # geography (woof.runtime._metadata_frame) and this route has no
        # geography for.  It has something better: the child-grid surface
        # source it was warm-started from, which is where its land identity
        # legitimately comes from.  Written verbatim, so a child's own
        # history is itself a valid --child-surface-from file -- the same
        # completeness the parent's history now has, one generation down.
        result.update({
            "LANDMASK": np.asarray(surface.fields["LANDMASK"],
                                   dtype=np.float32),
            "LU_INDEX": np.asarray(surface.fields["LU_INDEX"],
                                   dtype=np.float32),
        })
    return result


def _write_frame(path: Path, state, cfg, initial, valid_time,
                 projection_attrs: dict[str, object], placement,
                 refl_field=None, surface=None,
                 history_selection=None, simulation_start=None,
                 history_parent_start=None) -> None:
    from woof.io.wrfout import WrfoutWriter

    attrs = dict(projection_attrs)
    # A segment of a moving step-down (woof.moving_downscale) writes the
    # history of ONE moving nest: the forecast's start, and the segment's
    # place on its parent's lattice rather than in the cropped window it
    # was run in.  Written as the window place and the segment's own start,
    # every segment read to the renderer as a separate run, so no window
    # that spans a segment start (1 h rain, the 1 h maxima) was drawn on
    # the moving 1 km and 333 m grids of a WOOF setup-following run at any
    # hour (production run_54ce6aa98bdfdfe9, 2026-10-27).
    start = simulation_start or initial.valid_time
    i_start, j_start = (
        (placement.i_parent_start, placement.j_parent_start)
        if history_parent_start is None else history_parent_start)
    if surface is not None:
        # The land-use table identity, forwarded from the child's own
        # surface source.  read_child_surface_state requires these four as
        # EVIDENCE rather than assuming a table, so a child history file
        # that omitted them could not seed a grandchild however complete its
        # fields were -- and inventing them here would be exactly the
        # assumption that refusal exists to prevent.  ISOILWATER rides along
        # because stock WRF writes it too (share/output_wrf.F:973).
        identity = surface.identity
        attrs.update({
            "MMINLU": str(identity["MMINLU"]),
            "ISWATER": np.int32(identity["ISWATER"]),
            "ISLAKE": np.int32(identity["ISLAKE"]),
            "ISICE": np.int32(identity["ISICE"]),
            "ISOILWATER": np.int32(identity["ISOILWATER"]),
        })
    attrs.update({
        "GRID_ID": np.int32(cfg.grid_id),
        "PARENT_ID": np.int32(0),
        "I_PARENT_START": np.int32(i_start),
        "J_PARENT_START": np.int32(j_start),
        "PARENT_GRID_RATIO": np.int32(placement.parent_grid_ratio),
        "DT": np.float32(cfg.dt),
        "HYBRID_OPT": np.int32(cfg.hybrid_opt),
        "ETAC": np.float32(cfg.etac),
        # One forecast, one start: a moving step-down's segments continue
        # the forecast begun at the child's start the way a moving nest
        # continues across its moves, so START_DATE (which XTIME counts
        # from) is that start on every segment too, as in a moving nest's
        # history.
        "START_DATE": start.strftime("%Y-%m-%d_%H:%M:%S"),
        "SIMULATION_START_DATE": start.strftime("%Y-%m-%d_%H:%M:%S"),
        "GPUWM_OFFLINE_CHILD": np.int32(1),
    })
    frame = _output_fields(state, initial, refl_field=refl_field,
                           surface=surface)
    # The child's own [output] selection, from its own child config.  A
    # refined child at a fraction of the parent's spacing is exactly the
    # domain whose history fills a disk, and `woof downscale` is the one
    # RunConfig-TOML route that writes a wrfout, so the table is real
    # here rather than validated-and-ignored.  ``None`` -- and the FULL
    # default -- leaves the frame and the header byte-identical.
    if history_selection is not None:
        frame, history_attrs = history_selection.apply(frame)
        attrs.update(history_attrs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with WrfoutWriter(
            path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
            dx=cfg.dx, dy=cfg.dy,
            title="woof native standalone offline child",
            global_attrs=attrs,
            # The soil axis is the selected LSM's geometry, not a constant.
            soil_layers=soil_layer_count(cfg)) as writer:
        writer.write_frame(valid_time.strftime("%Y-%m-%d_%H:%M:%S"), frame)


def run(args: argparse.Namespace) -> dict[str, object]:
    """One offline child, with its progress receipts published.

    The receipts wrap the whole run so that the LAST event a reader sees
    is always terminal: ``completed`` for a run that produced a report,
    ``failed`` carrying the sentence for one that did not -- including a
    refusal raised before the model started, where the stream exists but
    is empty and the emit is a no-op.
    """

    from woof import ownership
    from woof.offline_child import output_owner_path

    # A folder this call reserves itself (the direct door; ``woof
    # downscale`` reserves before it gets here and releases after) gives
    # up its owner file when the run ends, however it ends.  Only a claim
    # this call took is released: one an outer caller holds stays held.
    outdir = getattr(args, "outdir", None)
    owner_path = None if outdir is None else output_owner_path(outdir)
    held_before = None if owner_path is None else ownership.held_claim(owner_path)
    try:
        # A stop is a way this run ends, recorded like the others: SIGTERM
        # raises where the child is, as SIGINT always has, and both reach
        # the stop path below (:func:`stop_on_signal`).
        with stop_on_signal():
            return _run_with_receipts(args)
    finally:
        held_after = None if owner_path is None else ownership.held_claim(owner_path)
        if held_after is not None and held_after is not held_before:
            held_after.release()


def _run_with_receipts(args: argparse.Namespace) -> dict[str, object]:
    progress = _ChildProgress()
    try:
        report = _run(args, progress)
    except BaseException as error:
        if _stopped_by_user(error):
            # A STOP IS NOT A FAILURE.  It was published as one -- event
            # ``failed``, report FAIL, "Why it stopped: KeyboardInterrupt"
            # -- and a SIGTERM was not published at all.
            with _stop_signals_held():
                _record_stop(progress, error)
            raise
        # Drawing stops before anything is counted, so the banner, the
        # report and the folder agree and no picture lands after them.
        # A stop draws nothing more (the desktop kills a run 5 s after
        # asking it to stop); a child that failed on its own finishes
        # the frames it already wrote.
        progress.stop_live_products(halt=_stopped_by_user(error))
        # The run did not finish, so it KEEPS what it drew and says so
        # twice: the early render's pictures stay where they are under a
        # banner naming the stop, and the same facts go into the report
        # this run leaves behind.  Every way a child stops reaches here
        # -- non-finite, a refusal raised mid-run, an interrupt -- so
        # there is one answer rather than one per cause, and the wait
        # inside the keep is what stops the render subprocess outliving
        # this process.
        kept = progress.keep_early_render(_stop_reason(error))
        # AND A RUN THAT BLEW UP LEAVES A DOCUMENT.  Every other outcome
        # of this route writes ``report.json`` -- a pass, a pass whose
        # pictures failed, a forecast its own health check refused -- and
        # the one outcome that wrote nothing was the one a reader most
        # needs to read afterwards, because the process was gone and the
        # event stream was all that was left of it.  Written from what
        # stopped the run (:func:`_stop_capsule`) and from what the keep
        # left on disk, so the banner, the sentence and the document are
        # the same facts.  EVERY stop, not only the ones that composed a
        # capsule: gating the document on the capsule left the interrupt,
        # the dead mount and the contract error raised mid-run writing a
        # banner over kept pictures and no report beside it, which is
        # this same hole three exception classes narrower.
        _publish_failure_report(progress, _stop_capsule(error), kept=kept)
        progress.failed(error)
        progress.end("failed")
        progress.close()
        raise
    if str(report["result"]) != "PASS":
        progress.stop_live_products()
        kept = progress.keep_early_render(
            "the child's own health check refused this forecast")
        if progress.render_plan is not None:
            from woof.first_products import DID_NOT_FINISH_STATUS

            # TWO fields, because they answer two questions.  ``status``
            # is the PICTURES' own verdict in this block's existing
            # vocabulary beside DRAWN and FAILED; ``run_status`` is the
            # RUN's state, and it is the same string the banner and the
            # render summary carry, so a reader keys on one value across
            # all three documents.
            _record_products(
                progress, report, status="KEPT",
                run_status=DID_NOT_FINISH_STATUS,
                reason=_did_not_finish_capsule(kept),
                pictures_on_disk=kept["pictures"],
                pictures_on_disk_error=kept.get("pictures_error"),
                banner=kept["banner"],
                render_command=_render_command_text(progress.render_plan))
    else:
        try:
            _finish_child_render(progress, report=report)
        except BaseException as error:
            if _stopped_by_user(error):
                # Stopped while the pictures were being drawn: the
                # forecast finished, so its report keeps its verdict and
                # records the stop beside it.
                with _stop_signals_held():
                    _record_stop(progress, error, report=report)
                if isinstance(error, (KeyboardInterrupt, ChildStopped)):
                    raise
                # The render stage's own spelling of a Ctrl-C
                # (GoInterrupted) is an ordinary exception, which the CLI
                # boundary would print as a traceback at exit 1; the stop
                # leaves this door as the Ctrl-C it was, at exit 130.
                raise KeyboardInterrupt from error
            # Before the failure event, so the stream a reader tails
            # ends on the failure rather than on a picture published by
            # a thread that was still running when it was raised.
            progress.stop_live_products(halt=True)
            progress.wait_early_render()
            progress.failed(error, stage="finalize")
            progress.end("failed")
            progress.close()
            raise
    progress.emit(
        "completed", stage="forecast", result=report["result"],
        outputs=len(report.get("outputs", []) or []),
        # Time to first plot and the render's own summary ride the
        # terminal event, exactly as they do on the chain's, so one
        # reader shape serves a forecast and a downscaled forecast.
        first_products_seconds=progress.first_products_seconds,
        **({"render_summary": progress._render_summary}
           if progress._render_summary is not None else {}))
    progress.end("completed" if str(report["result"]) == "PASS"
                 else "failed")
    progress.close()
    return report


def _stopped_by_user(error: BaseException) -> bool:
    """Whether the child ended because it was asked to stop.

    The interrupt spellings the run-plan door already recognises
    (:func:`woof.runplan._is_interrupt`: a ``KeyboardInterrupt``, and a
    stage that answered the same Ctrl-C), and :class:`ChildStopped`,
    which is what a SIGTERM becomes here (:func:`stop_on_signal`).
    """

    from woof.runplan import _is_interrupt

    return isinstance(error, ChildStopped) or _is_interrupt(error)


def _stop_signal(error: BaseException) -> tuple[str, int]:
    """``(signal name, exit code)`` for a stop.

    SIGTERM exits 143 and SIGINT 130, the shell's 128 + the signal, so a
    stop reads the same to a script whichever signal asked for it.
    """

    if isinstance(error, ChildStopped):
        return error.signal_name, int(error.code)
    return "SIGINT", 130


def _stop_why(signal_name: str) -> str:
    """The banner's ``Why it stopped`` line for a stop."""

    return f"it was stopped by request ({signal_name})"


def _stop_message(progress: "_ChildProgress", *, signal_name: str,
                  pictures: int | None, finished: bool) -> str:
    """The one sentence a stop publishes: the event, the report, the log.

    Where the forecast got to, and what is on disk, in the banner's own
    wording (:func:`woof.first_products.banner_text`), so the three
    documents say one thing.
    """

    from woof.first_products import _stop_sentence

    frames = len(progress._frames)
    if finished:
        where = ("The forecast had finished and every frame is on disk; "
                 "the pictures were being drawn.")
    else:
        where = _stop_sentence(progress._stopped)
        where += (f"  The {frames} frame{'s' if frames != 1 else ''} "
                  "written before the stop "
                  f"{'are' if frames != 1 else 'is'} on disk."
                  if frames else "  No frame was written before the stop.")
    if pictures is None:
        held = ""
    elif pictures:
        had, verb, _them = _pictures_phrase(pictures)
        held = f"  {had} drawn before the stop {verb} kept."
    else:
        held = ""
    return f"Stopped by request ({signal_name}).  {where}{held}"


def _record_stop(progress: "_ChildProgress", error: BaseException, *,
                 report: dict | None = None) -> None:
    """Write a stop down everywhere a finished or failed run is written.

    In order: every render is ended (:meth:`_ChildProgress.halt_renders`)
    so nothing is drawn after the stop; what a stopped render left is
    removed and the rest is kept under a banner that says the run was
    stopped by request (:meth:`_ChildProgress.keep_early_render`);
    ``report.json`` records ``result`` ``STOPPED`` with a ``stop`` block
    and the pictures' ``products`` block; the stream ends on the stop
    (:meth:`_ChildProgress.stopped`); and the run manifest gets its end.

    ``report`` is the finished forecast's own report, for a stop that
    landed while the pictures were being drawn: the forecast's verdict
    stands, and the stop is recorded beside it.

    Every step is best effort, because a stop that could not write one
    document must still write the others and still exit.
    """

    from woof.first_products import (
        DID_NOT_FINISH_STATUS, count_pictures, discard_stopped_render)

    signal_name, exit_code = _stop_signal(error)
    stage = ("finalize" if report is not None
             else progress._current_stage or "forecast")
    try:
        ended = progress.halt_renders()
    except Exception:  # noqa: BLE001 - the stop is still written down
        ended = False
    command = (None if progress.render_plan is None
               else _render_command_text(progress.render_plan))
    try:
        if report is None:
            kept = progress.keep_early_render(
                _stop_why(signal_name), requested=True, discard=ended)
            pictures = kept.get("pictures")
            message = _stop_message(progress, signal_name=signal_name,
                                    pictures=pictures, finished=False)
            document = {"result": "STOPPED",
                        "pipeline": CHILD_REPORT_PIPELINE}
            products = dict(
                status="KEPT", run_status=DID_NOT_FINISH_STATUS,
                reason=_did_not_finish_capsule(kept),
                pictures_on_disk=pictures,
                pictures_on_disk_error=kept.get("pictures_error"),
                banner=kept.get("banner"),
                discarded_unfinished=kept.get("discarded", 0),
                render_command=command)
        else:
            discarded = 0
            if progress.render_plan is not None and ended:
                discarded = len(discard_stopped_render(
                    Path(progress.render_plan["render"]))["removed"])
            pictures, uncounted = ((0, None) if progress.render_plan is None
                                   else progress.pictures_drawn())
            message = _stop_message(progress, signal_name=signal_name,
                                    pictures=pictures, finished=True)
            document = report
            products = dict(
                status="STOPPED", run_status="stopped",
                reason=("the render was stopped by request before it "
                        "finished; the forecast finished and every frame "
                        "is on disk, so the rest draw with the command "
                        "beside this"),
                pictures_on_disk=pictures,
                pictures_on_disk_error=uncounted,
                discarded_unfinished=discarded,
                render_command=command)
        document["stop"] = {
            "requested": True, "signal": signal_name,
            "exit_code": exit_code, "stage": stage, "summary": message,
            "stopped_at": (None if progress._stopped is None
                           else dict(progress._stopped))}
        _record_products(progress, document, **products)
    except Exception:  # noqa: BLE001 - the stop is still written down
        message = f"Stopped by request ({signal_name})."
    try:
        progress.stopped(stage=stage, message=message,
                         signal_name=signal_name, exit_code=exit_code)
    finally:
        progress.end("stopped")
        progress.close(wait=False)


def _first_sentence(error: BaseException) -> str:
    """The refusal's first sentence, for the banner's ``Why it stopped``.

    One sentence rather than the whole capsule: a banner is read at a
    glance, the full text is already on the stream's ``failed`` event,
    and a capsule that runs to three paragraphs would bury the four
    facts this file exists to state.  An exception with nothing to say
    -- an interrupt is the ordinary one -- is named by its type instead
    of leaving the line blank.
    """

    text = " ".join(str(error).split())
    if not text:
        return type(error).__name__
    head = text.split(". ")[0].strip()
    return (head or text)[:400]


def _stop_reason(error: BaseException) -> str:
    """Why the run stopped, in the one sentence the banner carries.

    BOTH rules survive, one calling the other, because they answer two
    different exceptions.  A refusal that composed a capsule has already
    chosen which of its sentences is the one to read first and published
    it as ``summary`` -- :class:`OfflineChildNonFinite` does, and its
    capsule runs to paragraphs whose first sentence is not the one a
    reader needs.  Everything else that reaches the stop arm carries no
    such attribute: an interrupt, an ``OSError`` from a dead mount, a
    contract error raised mid-run.  Those are read by
    :func:`_first_sentence` off the exception's own text.

    The run-plan ``failed`` event (:meth:`_ChildProgress.failed`) and the
    report's ``failure.summary`` are built from this same function, so the
    banner, the report and the event say the same sentence about one run,
    the event with the exception's class name in front of it.
    """

    summary = getattr(error, "summary", None)
    if isinstance(summary, str) and summary.strip():
        return " ".join(summary.split())[:400]
    return _first_sentence(error)


def _stop_capsule(error: BaseException) -> dict:
    """The ``failure`` block of a stopped child's report, whatever stopped it.

    A refusal that composed a capsule publishes it verbatim: it holds
    the field that went, the cell, the model second and the trend the
    health record was keeping, and the report is where a reader goes for
    all of it.

    EVERYTHING ELSE GETS A DOCUMENT TOO, of the three keys every reader
    of this file keys on.  ``summary`` is the one sentence the banner
    and the run-plan ``failed`` event carry (the event with the class
    name in front), so the three say one thing;
    ``message`` is the whole of what the exception said, VERBATIM, line
    breaks included -- a contract error raised mid-run composes
    paragraphs, and running them onto one line here would leave the only
    copy of them unreadable -- which for an interrupt, an ``OSError``
    from a dead mount or such an error is all there is; ``error_type``
    names the class,
    because "KeyboardInterrupt" is a fact about the stop and an empty
    string is not.  Without this a reader who opened a folder of kept
    pictures under a banner found no document beside them on exactly
    the stops that leave no other trace, and could not tell a run that
    stopped from one that never wrote its report.
    """

    capsule = getattr(error, "capsule", None)
    if isinstance(capsule, Mapping):
        return dict(capsule)
    message = str(error).strip()
    return {"summary": _stop_reason(error),
            "message": message or type(error).__name__,
            "error_type": type(error).__name__}


def _pictures_phrase(count: int) -> tuple[str, str, str]:
    """``("1 picture", "is", "it")`` or ``("20 pictures", "are", "them")``.

    ONE pair for every sentence this module writes about a kept render,
    because a count made singular over a verb or a pronoun that stayed
    plural is the same defect as the parenthesised plural it replaced,
    and two sentences that disagree with each other about one run are
    worse than either.  The banner's own clause comes from
    :func:`woof.first_products.banner_text` for the same reason.
    """

    if count == 1:
        return "1 picture", "is", "it"
    return f"{count} pictures", "are", "them"


def _kept_sentence(kept: dict, render_dir) -> str:
    """The warning's message: what was kept, and where the banner is.

    The event stream is read by run views as well as by logs, so this
    message agrees with itself on number exactly as the report's capsule
    and the banner do -- including on the third outcome, where the tree
    could not be listed and there is no number to agree about.
    """

    unreadable = kept.get("pictures_error")
    if kept.get("pictures") is None:
        return ("the child did not finish; its picture directory could not "
                f"be listed ({unreadable}), so what is in "
                f"{render_dir} is not known from here, and the banner at "
                "the top of it says where the forecast stopped")
    count = int(kept.get("pictures") or 0)
    if not count:
        return ("the child did not finish; no picture had been drawn yet, "
                f"and the banner at the top of {render_dir} says where the "
                "forecast stopped")
    had, verb, them = _pictures_phrase(count)
    return (f"the child did not finish; the {had} drawn while it ran "
            f"{verb} kept, and the banner beside {them} says where the "
            "forecast stopped")


def _did_not_finish_capsule(kept: dict) -> str:
    """What the report says about a stopped child's pictures.

    Its ``Next`` points at the pictures and the banner, NOT at redrawing
    the frames.  The reader of this sentence has just lost a forecast;
    what is useful to them is the pictures already on disk of the part
    that ran, and redrawing frames they can already see is work that
    answers nothing.  The command to draw them again is in the block
    beside this sentence for the reader who wants it.
    """

    where = kept.get("render") or "this run's picture directory"
    if kept.get("pictures") is None:
        # NOT "there is none to keep".  Nothing was removed and nothing
        # was counted: the tree could not be listed, and the sentence
        # that says a run has no pictures is the one that sends a reader
        # off to re-draw a child whose pictures are sitting behind the
        # error quoted here.
        return (f"this child did not finish.  {where} could not be listed "
                f"({kept.get('pictures_error')}), so what it holds is "
                "unknown; nothing was removed from it, and the frames this "
                "run did write and its checkpoints are on disk.")
    count = int(kept.get("pictures") or 0)
    if not count:
        return ("this child did not finish.  No picture had been drawn yet, "
                "so there is none to keep; the frames it did write and its "
                "checkpoints are on disk.")
    had, verb, them = _pictures_phrase(count)
    banner = kept.get("banner")
    if banner is None:
        return (f"this child did not finish; the {had} drawn while it ran "
                f"{verb} kept.  Next: open {where}.  Every picture there is "
                "of a frame written before the forecast stopped.")
    return (f"this child did not finish; the {had} drawn while it ran "
            f"{verb} kept.  Next: open {where} to see "
            f"{them}, and {Path(banner).name} beside {them} says where the "
            "forecast stopped and that every picture there is from before "
            "it.")


def _render_command_text(render_plan: dict) -> str:
    """The render command for this plan, as a reader would type it."""

    from woof.go_cli import printable, render_command

    return printable(render_command(render_plan))


def _renderer_said(text: str) -> str:
    """The render stage's last lines, indented under one heading.

    Indented rather than run into the sentence: these are the engine's
    own lines and a reader has to be able to tell them from ours, and a
    capsule read in a run view has no other formatting to do it with.

    The block carries no leading space and no trailing newline.  The
    caller joins the refusal's paragraphs with one newline each, so
    this heading and the ``Next:`` clause under it each begin a line
    of their own instead of running on from the sentence before.
    """

    body = "\n".join(f"  {line}" for line in text.splitlines() if line.strip())
    return f"The render stage said:\n{body}" if body else ""


def _record_products(progress: "_ChildProgress", report: dict,
                     **block) -> None:
    """Record what became of this child's pictures, in its own report.

    ``report.json`` is the document that says what this run produced,
    and it used to say ``PASS`` beside an empty picture tree whenever
    the render stage failed -- while the process exited 1 with a
    traceback.  One run cannot have two verdicts: ``result`` stays the
    forecast's, which passed, and this block is the pictures' own.

    The directory is the one the RUN resolved, read off the progress
    object rather than off the arguments: a run that never got as far
    as opening its stream has no report to amend either.
    """

    report["products"] = dict(block)
    if progress.outdir is None:
        return
    _publish_report(report, Path(progress.outdir))


def _publish_report(report: dict, outdir: Path) -> None:
    """Write ``report.json`` through a rename, from its one writer.

    Strictly (:func:`_strict_json`), because this document is read by
    whatever a reader has: a blown-up child's report carrying the token
    ``NaN`` is refused by ``JSON.parse``, ``serde_json``,
    ``encoding/json`` and ``jq`` alike, and this is the only outcome that
    writes that document.
    """

    temporary = outdir / "report.json.tmp"
    temporary.write_text(
        _strict_json(report, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, outdir / "report.json")


def _publish_failure_report(progress: "_ChildProgress", capsule: dict, *,
                            kept: dict) -> dict:
    """``report.json`` for a child that did not finish.

    The same two-document rule the render lane keeps
    (:func:`_finish_child_render`): the refusal a reader is shown and the
    report a reader opens afterwards carry the SAME facts, because one
    run cannot have two accounts of why it stopped.  ``result`` is the
    forecast's verdict, ``failure`` is what stopped the run as
    :func:`_stop_capsule` states it -- the refusal's own capsule
    verbatim where there is one -- and
    ``products`` is the pictures' own outcome, which for a run that did
    not finish is that they are KEPT: the block names the count on disk
    and the banner that stands over them, from the dict
    :meth:`_ChildProgress.keep_early_render` returned, so the document
    and the folder cannot disagree about how many pictures there are.
    """

    from woof.first_products import DID_NOT_FINISH_STATUS

    report = {
        "result": "FAIL",
        "pipeline": CHILD_REPORT_PIPELINE,
        "failure": dict(capsule),
    }
    # TWO fields, because they answer two questions, exactly as they do
    # on the arm holding a report whose result is not PASS: ``status``
    # is the PICTURES' own verdict beside DRAWN and FAILED, and
    # ``run_status`` is the RUN's state, the one string the banner and
    # the render summary carry too.
    _record_products(
        progress, report, status="KEPT",
        run_status=DID_NOT_FINISH_STATUS,
        reason=_did_not_finish_capsule(kept),
        pictures_on_disk=kept.get("pictures"),
        pictures_on_disk_error=kept.get("pictures_error"),
        banner=kept.get("banner"),
        render_command=(None if progress.render_plan is None
                        else _render_command_text(progress.render_plan)))
    return report


def _finish_child_render(progress: "_ChildProgress", *,
                         report: dict) -> None:
    """Draw a finished child's products through the shared render stage.

    :func:`woof.runplan._finish_render` is the finalize render every
    other route runs: it opens the finalize stage, hands the plan to
    :func:`woof.go_cli._render_stage`, and refuses -- naming the render
    command to run by hand -- when a requested product set produced
    nothing.  A child gets that function, not a copy of it.

    A stage that EXITS NONZERO is a different outcome from one that drew
    nothing, and it is the outcome a product the renderer cannot draw
    for these frames produces.  That is a
    :class:`woof.go_cli.GoStageFailed`, which subclasses ``Exception``
    and nothing the CLI boundary recognises -- so it left this door
    printing a traceback at exit 1 over a ``report.json`` that said
    ``PASS``.  It becomes this door's own refusal here: one sentence,
    exit 2, with the command that draws the finished frames by hand, and
    a ``products`` block in the report so the two documents agree.
    """

    if progress.render_plan is None:
        progress.stop_columns()
        return
    from woof.go_cli import GoStageFailed
    from woof.runplan import _finish_render

    try:
        _finish_render(progress.render_plan, observer=progress,
                       door="downscale")
    except GoStageFailed as failure:
        if failure.code == 130 or failure.code == -int(signal.SIGINT):
            # The render answered the Ctrl-C its whole process group got
            # before this process observed its own: that is the user's
            # stop, not a render that failed.
            from woof.go_cli import GoInterrupted

            raise GoInterrupted("render", None) from failure
        command = _render_command_text(progress.render_plan)
        early = progress.early_pictures()
        drawn, uncounted = progress.pictures_drawn()
        # WHAT IS ON DISK, counted.  This sentence used to report the
        # early render's count and call everything after it undrawn,
        # which a series render makes false: the renderer fails at its
        # batch summary, after drawing every frame it could.
        #
        # THREE OUTCOMES, not two.  A tree that could not be listed says
        # so and names the error; it used to count zero and print as an
        # empty tree, which is the one sentence that tells a reader to
        # re-draw everything.
        if uncounted is not None:
            outcome = ("this run's picture tree could not be read "
                       f"({uncounted}), so what it holds is unknown")
        elif drawn:
            outcome = ("this run's pictures are incomplete: "
                       + (f"{drawn} pictures are" if drawn != 1
                          else "1 picture is") + " on disk"
                       + (f", {early} of them drawn early from the first "
                          "frame" if early else "")
                       + ", and at least one product was not drawn")
        else:
            outcome = "this run has no pictures"
        # THE RENDERER'S OWN LAST LINES, in the refusal itself.  This
        # sentence used to carry the command and the exit code and
        # nothing else, and the stage's output went only to a terminal
        # -- which a desktop run view does not have.  A reader was shown
        # a 24-product render command and left to re-run the whole thing
        # to find out which product had failed.  The stage already
        # composes this tail for its event; the exception carries it, so
        # the refusal says what the renderer said.
        said = (failure.diagnostic or "").strip()
        _record_products(
            progress, report, status="FAILED",
            reason=(f"the render stage exited {failure.code}; the "
                    "forecast itself passed and its frames are on disk"),
            render_command=command, drawn_early=early,
            pictures_on_disk=drawn, pictures_on_disk_error=uncounted,
            renderer_output=said)
        # ONE newline between paragraphs, so the heading above and
        # the "Next:" clause below each begin a line of their own.
        # The sentence that opens this refusal already runs past the
        # 220 characters a run view shows of a first line, so the
        # break pushes nothing off that used to be visible.
        paragraphs = [
            "The child integrated and its frames are on disk, but the "
            f"render stage exited {failure.code}, so {outcome}.",
            *([_renderer_said(said)] if said else []),
            "Next: draw the saved frames by hand, which names the "
            "product that could not be drawn:\n  " + command,
        ]
        raise OfflineChildContractError(
            "\n".join(paragraphs)) from failure
    progress.finish_stage()
    _record_products(
        progress, report, status="DRAWN",
        render_products=str(progress.render_plan.get("render_products")),
        render_summary=progress._render_summary,
        render_command=_render_command_text(progress.render_plan))


#: Two-dimensional-looking names that start with Q and are not species.
_NON_SPECIES_Q = frozenset({"Q2", "QFX", "QSFC", "QKE", "QKE_ADV"})


def _interpolation_price(contract, placement, cfg):
    """The device price of interpolating this parent onto this child."""
    from woof.ingest.preparation_price import price_downscale_interpolation

    frame = contract.frames[0]
    dimensions = dict(frame.dimensions)
    species = sum(1 for name in frame.variables
                  if name.startswith("Q") and name not in _NON_SPECIES_Q)
    return price_downscale_interpolation(
        parent_nx=int(placement.parent_nx), parent_ny=int(placement.parent_ny),
        parent_nz=int(dimensions.get("bottom_top", cfg.nz)),
        parent_fields=8 + species, child_cfg=cfg)


def _run(args: argparse.Namespace,
         progress: "_ChildProgress") -> dict[str, object]:
    import cupy as cp
    from woof.ingest.preprocess_backend import decide_preparation_device
    from woof.core import streaming
    from woof.core.refl import consume_refl_10cm, refl_10cm_is_stashed
    from woof.ingest.lateral_bc import (
        attach_streaming_lateral_boundaries,
        bind_lateral_boundary_clock,
        lateral_boundary_reload_count,
        lateral_boundary_resident_bytes,
    )
    from woof.acoustic_adaptation import acoustic_receipt
    from woof.io.restart import auto_epssm_header, write_restart
    from woof.io.wrfout import wrfout_filename

    started = time.perf_counter()
    # Pictures are this route's default, so "this computer cannot draw"
    # is admitted HERE -- before the archived parent is read and the
    # child is integrated -- and not discovered after the forecast,
    # where the only outcome left is a completed child reported as a
    # failure.  `woof go` refuses its own chain at the same point, for
    # the same reason, and names the same two ways out.
    from woof.first_products import early_render_requested

    render_products = getattr(args, "render_products", None)
    if early_render_requested(render_products):
        from woof.go_cli import render_extra_missing

        missing = render_extra_missing()
        if missing is not None:
            from woof.explain import layered

            raise OfflineChildContractError(layered(
                RENDERER_MISSING_REMEDY, missing))
    # ``outdir_reserved`` means the caller already applied the same
    # never-adopt reservation in this process (``woof downscale`` does,
    # so the config it derives can live inside the run it describes).
    # Absent, this is the reservation.
    outdir = (Path(args.outdir).resolve()
              if getattr(args, "outdir_reserved", False)
              else _create_output_root(args.outdir))
    # The config the run is actually built on, resolved through the one
    # function `woof downscale`'s plan review calls, so a --child-levels
    # ladder reaches this door with the same answer the review printed.
    cfg = resolve_child_run_config(
        args.child_config, child_levels=getattr(args, "child_levels", None))
    # Read at ADMISSION, beside the config it belongs to, and before any
    # parent frame is opened: mode = 'on' streams unconditionally and needs
    # no card to be knowable, so a malformed block fails here rather than
    # after the whole archive has been interpolated.  Resolved through the
    # same shared function for the same reason as the config above.
    tiles = resolve_child_streaming_options(
        args.child_config, getattr(args, "tiles", None))
    # Read at ADMISSION as well, and for the same reason: an unknown
    # variable name or a history_vars/history_drop clash refuses HERE,
    # before a parent archive is opened, rather than at the first frame.
    history_selection = load_history_selection(args.child_config)
    history_selection.warn_lost_products(
        HISTORY_VOCABULARY, where=f"child d{cfg.grid_id:02d}")
    from woof.live_columns import StateFrames
    progress.columns_frames = StateFrames.for_selection(history_selection)
    require_offline_child_root_forcing(cfg)
    # WHICH GEOGRAPHY the child is built on, resolved at admission through
    # the one function plan review resolved it with, so the two doors
    # cannot answer differently; an own-geography child whose WPS_GEOG
    # tree is not there is refused HERE, before a parent frame is opened.
    from woof.offline_child_geography import (
        ChildGeographyError, build_child_geography,
        load_child_static_policy, require_geog_tree,
        surface_on_child_geography)

    try:
        static_policy = load_child_static_policy(
            args.child_config, cfg,
            terrain=getattr(args, "child_terrain", None),
            geog_root=getattr(args, "geog_root", None),
            highres_cache=getattr(args, "highres_cache", None))
        require_geog_tree(static_policy, cfg=cfg)
    except (ChildGeographyError, ValueError) as error:
        raise OfflineChildContractError(str(error)) from error
    # The same radiation ladder rule plan review asks, asked again here on
    # the direct runner door, which no plan review stands in front of --
    # and asked BEFORE the archive is opened for interpolation rather than
    # after the whole of it has been read.
    require_runnable_child_radiation_from_archive(
        cfg, (args.parent_history[0] if args.parent_history else None))
    # How many checkpoint sets this child keeps, settled once: the door
    # hands its own answer on, and the runner door reads the same knob.
    keep_checkpoints = child_checkpoint_retention(
        getattr(args, "keep_checkpoints", None))
    if args.parent_restart is not None:
        binding = bind_parent_physics_from_gpuwm_restart(args.parent_restart)
    else:
        binding = bind_parent_physics_from_wrf_namelist(
            args.parent_namelist, domain_id=args.parent_domain_id)
    contract = validate_parent_history(
        args.parent_history,
        max_boundary_interval_seconds=args.max_boundary_interval_seconds,
        physics_binding=binding)
    parent_file_receipts = [
        _file_receipt(frame.path) for frame in contract.frames]
    dims = contract.frames[0].dimensions
    # A child at a DIFFERENT level count than its parent is now built by the
    # conservative vertical remap (woof/vertical_remap.py), which requires
    # the child to name the ladder it wants.  A level count on its own is
    # refused rather than filled in: make_vertical_coord's default is a
    # UNIFORM ladder, so a child that asked only for "more levels" off a
    # stretched parent would silently get a different atmosphere, not a finer
    # sampling of the same one.
    if int(cfg.nz) != int(dims["bottom_top"]) and cfg.eta_levels is None:
        raise OfflineChildContractError(
            f"child nz={cfg.nz} differs from parent nz={dims['bottom_top']} "
            "but the child config names no eta_levels: a deeper child has to "
            "declare the ladder it wants, because a bare level count would "
            "be filled in with a uniform ladder and the child would start "
            "from a different atmosphere than its parent, not a finer "
            "sampling of it.  Add eta_levels to the child config (woof "
            "downscale --child-levels writes one for you).")
    if float(cfg.run_seconds) > (
            contract.end_time - contract.start_time).total_seconds():
        raise OfflineChildContractError(
            "child run_seconds exceeds the archived parent forcing window")
    placement = OfflineChildPlacement(
        parent_nx=int(dims["west_east"]),
        parent_ny=int(dims["south_north"]),
        child_nx=int(cfg.nx), child_ny=int(cfg.ny),
        parent_grid_ratio=int(args.parent_grid_ratio),
        i_parent_start=int(args.i_parent_start),
        j_parent_start=int(args.j_parent_start))
    parent_dx, parent_dy, projection_attrs = _parent_grid_metadata(
        contract.frames[0].path)
    expected_dx = parent_dx / placement.parent_grid_ratio
    expected_dy = parent_dy / placement.parent_grid_ratio
    if not np.isclose(cfg.dx, expected_dx, rtol=2e-7, atol=1e-6):
        raise OfflineChildContractError(
            f"child dx={cfg.dx} != parent DX/ratio={expected_dx}")
    if not np.isclose(cfg.dy, expected_dy, rtol=2e-7, atol=1e-6):
        raise OfflineChildContractError(
            f"child dy={cfg.dy} != parent DY/ratio={expected_dy}")
    # THE ACOUSTIC RULE the child's own ground needs (A181), read off the
    # terrain it integrates before a frame is interpolated: the
    # off-centering floor and the substep count every prepared door
    # applies.  The same function plan review called, so a chosen epssm
    # below the floor is refused here on the runner door too, and every
    # step below runs on the adapted config.
    cfg, acoustic = adapt_child_acoustics(
        cfg, child_config_path=args.child_config,
        frame_path=contract.frames[0].path, placement=placement)
    _log("child_acoustics", **acoustic.receipt())
    # The child's clock, checked by the same function the plan review
    # checked it with (child_cadence), so a cadence that is not a whole
    # number of steps was refused when the child was reviewed and cannot
    # surface here for the first time.  The child's OWN restart cadence is
    # honoured: ``restart_interval_s`` rides into every derived child
    # config verbatim from the parent, and the door's own refusal tells a
    # user "the parent needs restart_interval_s inside its window to be
    # downscalable" -- yet the child wrote one final checkpoint under a
    # name no discovery recognised, so no downscaled run was ever
    # downscalable.  A setting accepted and silently dropped is a defect;
    # checkpoint_due is the cadence.
    cadence = child_cadence(
        cfg, health_interval_seconds=float(args.health_interval_seconds))
    # Noah mosaic, before the parent archive is interpolated: this route
    # has no door that builds its tiles (offline_child.child_mosaic_refusal).
    mosaic_refusal = child_mosaic_refusal(cfg)
    if mosaic_refusal is not None:
        raise OfflineChildContractError(mosaic_refusal)
    # THE DISK, before the parent archive is interpolated and before the
    # child takes a step: the same projection the plan review refused on,
    # asked again here for the runner door, which no review stands in
    # front of.  A child that fills its disk stops partway with a torn
    # frame and can stop other work on that disk.
    disk = child_disk_projection(
        cfg, cadence, keep_checkpoints=keep_checkpoints,
        render_products=render_products, outdir=outdir,
        history_selection=history_selection)
    if disk["refusal"] is not None:
        from woof.explain import layered

        raise OfflineChildContractError(layered(
            disk["refusal"][0].upper() + disk["refusal"][1:] + ".",
            "Refused before the child started.  Bytes per cell: "
            f"{disk['basis']}.  Pictures: {disk['picture_basis']}."))
    # The parent tape's own level count, off the dimensions
    # ``validate_parent_history`` already read, so the regime statement in
    # a refusal costs no second open of the archive.  ``None`` from a tape
    # that does not name the dimension, which only makes the sentence
    # shorter.
    parent_levels = contract.frames[0].dimensions.get("bottom_top")
    parent_levels = None if parent_levels is None else int(parent_levels)
    steps = cadence.steps
    output_steps = cadence.output_steps
    health_steps = cadence.health_steps
    checkpoint_due = cadence.checkpoint_due
    surface = None
    geography = None
    if static_policy.own:
        # THE CHILD'S OWN STATIC GEOGRAPHY, built before the parent archive
        # is interpolated, so a footprint the static sources cannot cover
        # refuses before the minutes of preparation are spent.
        try:
            geography = build_child_geography(
                contract.frames[0].path, placement, cfg, static_policy,
                valid_time=contract.start_time)
        except ChildGeographyError as error:
            raise OfflineChildContractError(str(error)) from error
        _log("child_geography", **_jsonable(dict(geography.receipt)))
    surface_from = getattr(args, "child_surface_from", None)
    if surface_from is not None:
        surface = read_child_surface_state(
            surface_from, child_ny=int(cfg.ny), child_nx=int(cfg.nx),
            num_soil_layers=soil_layer_count(cfg))
    elif child_surface_requirement(cfg) is not None and geography is not None:
        # The child's land identity is its own static build and its land
        # state the parent's, put on that land after the initial state
        # gives the blended terrain and the rebalanced surface pressure
        # (below).  The parent frame's surface inventory is asked for
        # there, before any boundary frame is interpolated.
        pass
    elif child_surface_requirement(cfg) is not None:
        # RESOLVED HERE, before interpolate_parent_initial_state and
        # build_offline_lateral_boundaries spend minutes on the parent
        # archive.  The guard in _initialize_child_physics used to be the
        # only one on this route, and it fires AFTER that work -- so a
        # child that could never start still paid for the preprocessing
        # first.  Defect #275: derive the child surface from the
        # parent's own history (WRF's input_from_file = .false. route)
        # rather than refusing for a file no command in the product
        # could produce for a config-driven parent.
        try:
            surface = derive_child_surface_from_parent(
                contract.frames[0].path, placement=placement,
                num_soil_layers=soil_layer_count(cfg))
        except OfflineChildContractError as error:
            raise OfflineChildContractError(
                f"{child_surface_requirement(cfg)}\n"
                f"  and this parent archive cannot supply one either: "
                f"{error}") from error
        warn("child surface state interpolated from the parent's own "
             "history rather than built on the child grid -- "
             + DERIVED_CHILD_SURFACE_CAVEAT,
             why="This is WRF's input_from_file = .false. route for a "
                 "nest with no wrfinput of its own (med_nest_initial's "
                 "med_interp_domain), run through the Registry's masked "
                 "land interpolator.")
    surface_file_receipts = (
        [] if surface is None else [_file_receipt(surface.path)])
    _log("contract_pass", frames=len(contract.frames),
         cadence_seconds=contract.interval_seconds,
         geometry_sha256=contract.geometry_sha256,
         source_physics=dict(binding.receipt()),
         target_mp_physics=int(cfg.mp_physics),
         child_shape=[cfg.nz, cfg.ny, cfg.nx],
         child_spacing_m=[cfg.dy, cfg.dx])

    # THE [tiles] DECISION, taken HERE on a COLD card and never again.  The
    # same function ``woof downscale`` reviewed this child with
    # (woof.downscale_pricing.price_child): the configured envelope from
    # the itemized estimator, judged against a machine measured before this
    # process has allocated a byte on the device.  It used to be taken after
    # the initial state, the boundary tables and the physics driver had
    # filled the card, with no estimate and no machine, so the tile planner
    # measured what was LEFT and charged the whole rung's fixed cost against
    # it: a 138x138x49 child the review had admitted at 2.90 GiB against
    # 7.32 GiB free was refused at "no tile fits in 3.49 GiB" after the
    # whole archive had been interpolated.  A card that is genuinely too
    # small refuses here, before anything is interpolated or allocated on
    # the device, with the measured figure and the way out.
    #
    # The card is read on EVERY [tiles] setting, off included, because it
    # is what the child is priced on as well as what [tiles] is decided on.
    # With [tiles] off the runner used to price on no card at all, so the
    # estimator fell back to the 170-SM reference profile and report.json
    # carried an envelope larger than the card the child then ran on
    # (17,033,346,128 B for a 552x552x49 child on a 15.47 GiB card that the
    # review and a --tiles=auto run both priced at 14,922,267,728 B).
    from tilestream.autoplan import CannotPlan

    try:
        card = downscale_pricing.cold_card(tiles)
        # The estimator's default forcing model, as the fitted sizing and
        # the review price it: the child's boundary intervals are streamed
        # from the host one at a time, so counting every archived interval
        # as retained on the device would price this child a third above
        # what it holds and stream a child that fits.
        pricing = downscale_pricing.price_child(
            cfg, tiles, machine=card.machine, profile=card.profile,
            basis=downscale_pricing.MEASURED_BASIS)
    except CannotPlan as error:
        raise OfflineChildContractError(str(error)) from error
    tiles = pricing.options
    streaming_decision = pricing.decision
    _log("child_streaming_decision", **pricing.plan_entry())
    admission_machine = streaming.cold_admission_machine(card.machine,
                                                         options=tiles)
    try:
        # A pinned tiling, priced on this card before its buffers exist.
        streaming.admit_pinned_road(cfg, tiles, streaming_decision,
                                    machine=admission_machine,
                                    resident_estimate=pricing.estimate)
    except CannotPlan as error:
        raise OfflineChildContractError(str(error)) from error
    if pricing.estimate is not None:
        # THE RESIDENT CHILD, ADMITTED BEFORE ITS FIRST ALLOCATION.  With
        # [tiles] off the decision above admits nothing, so the child state
        # and the physics driver attached after it (_initialize_child_physics
        # below) went onto the card unpriced and a child too big for it died
        # in CUDA after interpolation.  The same estimate the plan review
        # printed, weighed against the card the decision was read on.
        streaming.admit_resident_road(
            None, streaming_decision, machine=admission_machine,
            estimate=pricing.estimate,
            what="this downscaled child, held resident on the card")

    # PUBLISHED HERE: after every contract that can refuse this child has
    # passed and before the first minute of preprocessing is spent, so a
    # reader watching the directory sees a run it can trust, and sees it
    # from the beginning of the work rather than the end.
    progress.start(
        outdir=outdir, child_config=Path(args.child_config),
        ratio=int(placement.parent_grid_ratio),
        start_time=contract.start_time,
        parent={
            # Absolute, because a run browser finds the parent run by the
            # folder names in this path; a relative one typed at the
            # command line names no run folder at all.  The checkpoint
            # the same way, so the record names one file wherever read.
            "run_dir": os.path.abspath(Path(contract.frames[0].path).parent),
            "restart": (None if args.parent_restart is None
                        else os.path.abspath(args.parent_restart)),
            "frames": len(contract.frames),
            "cadence_seconds": float(contract.interval_seconds),
        },
        name=child_run_name(
            Path(contract.frames[0].path), grid_id=int(cfg.grid_id),
            ratio=int(placement.parent_grid_ratio), dx=float(cfg.dx)))
    # Armed beside the manifest, so the analysis frame -- written before
    # a single step is integrated -- becomes a picture while the child
    # is still running, as it does on every other route.
    progress.arm_render(outdir=outdir, render_products=render_products)
    progress.emit("stage_started", stage="initialize", phase="preprocess")

    # WHERE THE INTERPOLATION RUNS, PRICED BEFORE IT ALLOCATES.  The
    # boundary interpolation holds every parent field on the PARENT's
    # extent and levels, which a child-sized price cannot see.  auto (the
    # default) interpolates on the CPU when that does not fit the card's
    # free memory; an explicit cuda that cannot fit is refused by name
    # (A65).  The child state itself is the forecast's, on the card either
    # way, and was priced above.
    preprocess_backend, preprocess_selection = decide_preparation_device(
        args.preprocess_backend, _interpolation_price(contract, placement, cfg))
    if preprocess_selection is not None:
        _log("preprocess_backend_selection", **preprocess_selection)

    initial = interpolate_parent_initial_state(
        contract.frames[0].path, placement,
        physics_binding=binding, target_mp_physics=cfg.mp_physics,
        backend=preprocess_backend,
        child_eta_levels=cfg.eta_levels, child_cfg=cfg,
        geography=geography)
    if geography is not None:
        _log("child_terrain", **_jsonable(dict(initial.receipt["terrain"])))
        if (surface is None
                and child_surface_requirement(cfg) is not None):
            surface = surface_on_child_geography(
                contract.frames[0].path, placement=placement,
                geography=geography,
                num_soil_layers=soil_layer_count(cfg),
                valid_time=contract.start_time,
                terrain_child=initial.fields["HGT"],
                psfc=initial.fields["PSFC"])
            surface_file_receipts = [_file_receipt(surface.path)]
    conversion = initial.receipt.get("conversion")
    if conversion is not None:
        _log("microphysics_conversion",
             source_mp_physics=int(conversion["source_mp_physics"]),
             target_mp_physics=int(conversion["target_mp_physics"]),
             policy_id=conversion["policy_id"],
             species_action_counts=dict(conversion["species_action_counts"]),
             device=conversion["device"],
             host_chunked=bool(conversion["host_chunked"]))
    prepared = build_offline_lateral_boundaries(
        contract, placement,
        target_mp_physics=cfg.mp_physics,
        backend=preprocess_backend,
        child_eta_levels=cfg.eta_levels, child_cfg=cfg,
        spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone,
        geography=geography)
    # A float32 SINT of a number moment can round across zero.  When it
    # does, say so with the numbers: the cells touched, the tolerance and
    # the most negative value are what tell a reader whether they watched
    # rounding get cleaned up or an interpolation start to go wrong.  An
    # empty account emits nothing, so silence here means it landed clean.
    initial_clamp = initial.receipt.get("positive_definite_clamp") or {}
    if initial_clamp:
        _log("initial_positive_definite_clamp",
             fields={name: dict(account)
                     for name, account in initial_clamp.items()})
    child = build_offline_child_domain_state(initial, cfg)
    attach_streaming_lateral_boundaries(child, prepared.boundaries)
    # Davies clock bind (production semantics): boundary consumers take
    # WRF's post-increment dtbc recurrence from a bound integer-tick
    # clock, exactly like the experiment tree's root.
    clock = _child_boundary_clock(
        cfg, lbc_interval_seconds=contract.interval_seconds,
        steps=steps, output_steps=output_steps)
    bind_lateral_boundary_clock(child, clock)
    driver = _initialize_child_physics(child, cfg, initial, surface,
                                       initial.valid_time,
                                       **({"terrain_drag_static": geography.fields}
                                          if geography is not None and
                                          (cfg.topo_wind or cfg.gwd_opt
                                           or cfg.mosaic_lu or cfg.mosaic_soil
                                           or cfg.sf_lake_physics
                                           or cfg.usemonalb
                                           or cfg.rdlai2d) else {}))
    ozone_routing = _child_ozone_routing(driver)
    # A WARM START (woof.offline_child_warm): an earlier child of this one,
    # at a nearby placement, carries what it integrated over the ground both
    # share.  Applied here -- after the cold-start child and its driver are
    # whole, before the stepper fills any store -- so a moving step-down's
    # segment continues the forecast instead of spinning up again.
    warm_receipt = None
    warm = getattr(args, "warm_start", None)
    if warm is not None:
        from woof.offline_child_warm import WarmStartRefusal, apply_warm_start

        try:
            warm_receipt = apply_warm_start(
                child, cfg, driver, checkpoint=warm["checkpoint"],
                shift=warm["shift"], strip=warm.get("strip"))
        except WarmStartRefusal as error:
            raise OfflineChildContractError(str(error)) from error
        _log("child_warm_start", **_jsonable(warm_receipt))
    cp.cuda.runtime.deviceSynchronize()
    # ``[tiles]``, wired exactly the way the prepared front doors wire it
    # (woof.prepared_single_domain_forecast: decide ONCE, hand the decision
    # to make_stepper, record it).  With no block this is
    # ``woof.core.dycore.step`` ITSELF -- the same function object the loop
    # below has always called -- so a child that configures nothing is
    # byte-for-byte the run it was before this seam existed.
    #
    # The DECISION was taken above, before preprocessing, on the cold card.
    # The STEPPER is made here, AFTER the physics driver is attached, never
    # before: the builder fills the store from the PREPARED state and builds
    # every tile buffer with the domain's own physics selectors, so a
    # stepper made against a bare DomainState would carry a different
    # inventory than the domain it is meant to be.
    stepper = streaming.make_stepper(
        child, cfg, tiles, decision=streaming_decision,
        build=streaming.standalone_domain_builder(grid_id=int(cfg.grid_id)))
    streaming_report = streaming.streaming_receipt(
        tiles, {int(cfg.grid_id): streaming_decision})
    if streaming_report:
        _log("child_tiles", **streaming_report)
    # ``stability_report`` ITSELF when resident; the per-tile fold when
    # streamed.  The state is not where a streamed domain lives, so the
    # whole-field reduction would inspect the snapshot that filled the store
    # and pass forever.
    child_stability = streaming.stability_observer(stepper)
    boundary_bytes = lateral_boundary_resident_bytes(child)
    child_memory_initial = _memory_snapshot(cp)
    child_pool_reserved_peak = child_memory_initial["pool_reserved_bytes"]
    _log("child_launch", pid=os.getpid(), steps=steps,
         boundary_intervals=len(prepared.boundaries.intervals),
         boundary_device_resident_bytes=boundary_bytes,
         boundary_device_reload_count=lateral_boundary_reload_count(child),
         memory=child_memory_initial)

    output_paths: list[Path] = []

    def emit_output() -> None:
        valid = initial.valid_time + timedelta(
            seconds=float(clock.elapsed_seconds))
        # THE HISTORY CADENCE, which is where StreamedDomain.refresh_state
        # says this belongs.  A streamed domain's forecast is in the pinned
        # host store and this DomainState is the snapshot that filled it, so
        # without the copy every frame after the cold-start one would be the
        # initial condition under a later timestamp -- correct inventory,
        # correct Times, no forecast.  Zero and a getattr when resident.
        streaming.refresh_streamed_state(stepper, child)
        refl = _history_refl_10cm(child, cfg, clock.ticks)
        path = outdir / wrfout_filename(valid, domain_id=cfg.grid_id)
        _write_frame(path, child, cfg, initial, valid,
                     projection_attrs, placement, refl_field=refl,
                     surface=surface, history_selection=history_selection,
                     simulation_start=getattr(args, "simulation_start", None),
                     history_parent_start=getattr(args, "history_parent_start",
                                                  None))
        output_paths.append(path)
        # History-interval reset of the UP_HELI_MAX window (no-op unless
        # the child config enables nwp_diagnostics; the synchronous
        # writer above snapshotted the accumulator already).
        from woof.core.uh_diag import reset_up_heli_max
        reset_up_heli_max(child)
        _log("child_output", elapsed_seconds=float(clock.elapsed_seconds),
             path=str(path), bytes=path.stat().st_size)
        progress.output_committed(domain=int(cfg.grid_id),
                                  valid_time=valid.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                  path=str(path), bytes=path.stat().st_size)

    # A DISCOVERABLE set (woof.io.restart.restart_filename's instant
    # naming, the one woof.resume.discover_checkpoint_sets recognises), so
    # this run can be the parent of the next downscale, and only the
    # newest ``keep_checkpoints`` of them stay on disk.
    # The scheme conversion this child was born through, source and target
    # named, so a record built from this restart carries both ends of the
    # seam; absent for a same-scheme child.  And whether its epssm was the
    # model's choice, so a grandchild derived from it inherits that label
    # (A181); absent for a written one.  A header with neither is what it
    # always was.
    restart_header = {
        **({} if conversion is None else {
            "offline_microphysics_conversion": _jsonable(conversion)}),
        **auto_epssm_header(
            int(cfg.grid_id),
            (int(cfg.grid_id),) if child_epssm_is_auto(args.child_config)
            else ())}
    checkpoints = ChildCheckpoints(
        outdir, grid_id=cfg.grid_id, keep=keep_checkpoints,
        write=lambda path: write_restart(
            path, child, cfg, tree_header=restart_header or None))
    carriers_refreshed = 0

    def emit_checkpoint() -> None:
        # The streamed state is refreshed first, exactly as the history
        # writer does: a streamed domain's forecast lives in the pinned
        # host store and this DomainState is the snapshot that filled it,
        # so without the copy every checkpoint would be the initial
        # condition under a later clock.  Zero and a getattr when resident.
        nonlocal carriers_refreshed
        carriers_refreshed = streaming.refresh_streamed_state(stepper, child)
        valid = initial.valid_time + timedelta(
            seconds=float(clock.elapsed_seconds))
        path, retired = checkpoints.emit(valid)
        _log("child_checkpoint", elapsed_seconds=float(clock.elapsed_seconds),
             path=str(path), bytes=path.stat().st_size,
             retired=[str(old) for old in retired])

    progress.emit("stage_started", stage="forecast", phase="integrate")
    emit_output()
    step_seconds = []
    child_health = child_stability(child, cfg)
    # THE HEALTH RECORD'S OWN HISTORY, kept because the capsule reads it
    # back.  The per-step line already carried w_max and the CFL and then
    # dropped them on the floor, so the refusal at the end of a blow-up
    # could say nothing about the climb that produced it.  Bounded by the
    # window the capsule quotes, so a 69,120-step run holds a dozen rows
    # and not 480.
    trend: list[dict] = []
    trend_depth = nonfinite_trend_checks(
        None if health_steps is None else health_steps * float(cfg.dt))
    for step_index in range(1, steps + 1):
        # The executor's exact per-step recurrence (core/clock.py
        # execute_schedule): dtbc zeroes at every external interval seam
        # including t=0, prepare_step applies WRF's post-increment
        # ``grid%dtbc = grid%dtbc + grid%dt`` before the solve, and the
        # calendar advances after it.
        if clock.lbc_reset_due():
            clock.mark_force()
        clock.prepare_step()
        output_due = step_index % output_steps == 0 or step_index == steps
        step_started = time.perf_counter()
        stepper(child, cfg, refl_10cm_due=output_due)
        cp.cuda.runtime.deviceSynchronize()
        step_seconds.append(time.perf_counter() - step_started)
        clock.advance()
        if step_index % health_steps == 0 or step_index == steps:
            child_health = child_stability(child, cfg)
            health_fields = child_health_log_fields(child_health)
            # Both readings in the carrying shape the health decoder
            # produced them in, state words included, and WHERE the |w|
            # maximum was: the row goes into the capsule and from there
            # into report.json, a number that went is null with its state
            # beside it, and the last measured maximum's place is the one
            # location the capsule can offer as near an origin.
            health_row = child_health_trend_row(
                step=step_index, model_seconds=clock.elapsed_seconds,
                record=child_health, cfg=cfg)
            trend.append(health_row)
            del trend[:-(trend_depth + 1)]
            memory = _memory_snapshot(cp)
            child_pool_reserved_peak = max(
                child_pool_reserved_peak, memory["pool_reserved_bytes"])
            _log("child_step", step=step_index, total_steps=steps,
                 elapsed_seconds=float(clock.elapsed_seconds),
                 **health_fields,
                 w_max_cell=health_row["w_max_cell"],
                 w_max_edge=health_row["w_max_edge"],
                 boundary_device_reload_count=lateral_boundary_reload_count(child),
                 memory=memory,
                 wall_seconds=time.perf_counter() - started)
            progress.emit("model_progress", domain=int(cfg.grid_id),
                          model_seconds=float(clock.elapsed_seconds),
                          run_seconds=float(cfg.run_seconds),
                          outer_step=int(step_index), total_steps=int(steps),
                          wall_seconds=time.perf_counter() - started,
                          # The reader that offers "downscale from this
                          # run" takes its checkpoint directory from here,
                          # as it does for every other route.
                          **({"last_checkpoint": str(checkpoints.last)}
                             if checkpoints.last is not None else {}))
            if child_health["nan"]:
                # WHICH field, WHERE, and the climb that got there -- the
                # survey taken once, here, on the way out.  A streamed
                # child's forecast lives in the pinned host store and this
                # DomainState is the snapshot that filled it, so the
                # refresh comes first for exactly the reason the history
                # writer does it: without it the survey would read the
                # initial condition and report a perfectly finite field
                # over a run that had just gone non-finite.
                survey = _survey_nonfinite_child(streaming, stepper, child)
                raise OfflineChildNonFinite(describe_nonfinite_child(
                    step=step_index, total_steps=steps,
                    model_seconds=float(clock.elapsed_seconds),
                    run_seconds=float(cfg.run_seconds),
                    cadence_seconds=(None if health_steps is None
                                     else health_steps * float(cfg.dt)),
                    trend=trend, survey=survey,
                    regime=les_child_regime(
                        cfg,
                        inherits_parent_levels=child_inherits_parent_levels(
                            cfg,
                            child_levels_spec=getattr(
                                args, "child_levels", None),
                            parent_levels=parent_levels),
                        parent_levels=parent_levels),
                    render_command=(
                        None if progress.render_plan is None
                        else _render_command_text(progress.render_plan))))
        if output_due:
            emit_output()
        if step_index in checkpoint_due:
            # The final step is always due, so the run's last checkpoint
            # is the instant-named set the next downscale discovers.  The
            # refresh inside emit_checkpoint is what keeps a streamed
            # child's checkpoint from being the analysis under a later
            # clock, stated there rather than inferred from the output
            # cadence happening to coincide.
            emit_checkpoint()
    restart = checkpoints.last
    sample = np.asarray(step_seconds, dtype=np.float64)
    warm = sample[1:] if sample.size > 1 else sample
    _verify_file_receipts(
        parent_file_receipts, label="parent history input")
    _verify_file_receipts(
        surface_file_receipts, label="child surface source")
    report = {
        "result": "PASS" if not child_health["nan"] else "FAIL",
        "pipeline": CHILD_REPORT_PIPELINE,
        "online_parent_present_during_child": False,
        "parent_frames": [str(frame.path) for frame in contract.frames],
        "parent_frame_receipts": parent_file_receipts,
        "parent_geometry_sha256": contract.geometry_sha256,
        "parent_physics_binding": dict(binding.receipt()),
        "child_config": str(args.child_config.resolve()),
        "child_config_sha256": _sha256(args.child_config.resolve()),
        # The file hash AND the ladder actually integrated, side by side:
        # `--child-levels` can replace the supplied file's eta_levels, so
        # the hash alone no longer answers "which grid was this run on?".
        "child_levels_override": (
            None if getattr(args, "child_levels", None) is None
            else str(args.child_levels)),
        "effective_nz": int(cfg.nz),
        "effective_eta_levels": (
            None if cfg.eta_levels is None
            else [float(value) for value in cfg.eta_levels]),
        "target_mp_physics": int(cfg.mp_physics),
        # The acoustic rule read off the child's own ground (A181): the
        # epssm and substep count it ran, and why, as the prepared doors
        # record theirs.
        "acoustic_substeps": acoustic_receipt([acoustic]),
        "placement": {
            "parent_grid_ratio": placement.parent_grid_ratio,
            "i_parent_start": placement.i_parent_start,
            "j_parent_start": placement.j_parent_start,
            "child_nx": placement.child_nx,
            "child_ny": placement.child_ny,
        },
        "preprocess_backend": preprocess_backend,
        "preprocess_backend_selection": preprocess_selection,
        # Which cadence flag the invoker gave (audit finding 5): True
        # means the ceiling was the archive's own cadence, accepted via
        # --accept-parent-cadence; False means an explicit
        # --max-boundary-interval-seconds.  The effective interval the
        # child was forced at is boundary_clock.lbc_interval_seconds.
        "boundary_cadence_provenance": {
            "accepted_parent_cadence": bool(
                getattr(args, "accepted_parent_cadence", False)),
            "max_boundary_interval_seconds": float(
                args.max_boundary_interval_seconds),
            "effective_interval_seconds": float(contract.interval_seconds),
        },
        "boundary_clock": {
            "semantics": "wrf-dtbc-bound",
            "tick_den": int(clock.tick_den),
            "step_ticks": int(clock.spec.step_ticks),
            "lbc_interval_seconds": float(contract.interval_seconds),
            "final_ticks": int(clock.ticks),
        },
        # WHERE THE OZONE CAME FROM, said rather than inferred.  Under
        # o3input = 2 this route evaluates the packaged CAM climatology on
        # the CHILD's own grid (WRF's own answer for a domain configured as
        # a root, which is what an offline child is), so a reader never has
        # to work out from parent_id whether a parent field was involved.
        # None means the child's radiation carries no ozone routing.
        "child_ozone_routing": ozone_routing,
        # A segment of a moving step-down: the earlier child it continued
        # and what was carried.  None for a cold start.
        "warm_start": _jsonable(warm_receipt),
        "child_surface_source": (
            None if surface is None else dict(surface.receipt)),
        # WHICH TERRAIN the child ran on: its own static geography (the
        # default) or its parent's, with the static build and the blend
        # and rebalance receipts when it was its own.
        "child_terrain": {
            "policy": dict(static_policy.receipt()),
            "geography": (None if geography is None
                          else _jsonable(dict(geography.receipt))),
            "initial": _jsonable(initial.receipt.get("terrain")),
        },
        "child_surface_file_receipts": surface_file_receipts,
        # The online nest edge's own contract receipt for a child of a
        # different scheme (species actions, closure constants, source and
        # target identity), plus where this lane ran it; None when the
        # child inherited its parent's scheme.
        "microphysics_conversion": _jsonable(conversion),
        "boundary_conversion_frames": (
            0 if conversion is None else sum(
                1 for receipt in prepared.frame_receipts
                if receipt.get("conversion") is not None)),
        "preparation_seconds": prepared.preparation_seconds,
        # ``{}`` for a child that configures no [tiles], which is what keeps
        # every report written before this seam existed byte-identical.  A
        # configured child gets the per-grid verdict, and ``streamed_any``
        # is the field to assert on: a mode='auto' child that declined and
        # an unconfigured one are otherwise indistinguishable.
        "tiles": streaming_report,
        # The decision the run was integrated with, and what it was judged
        # on: the same block the plan review wrote, so a reader can hold
        # the two documents side by side and find one answer.
        "streaming": pricing.plan_entry(),
        "streamed_carriers_refreshed": carriers_refreshed,
        # The checkpoints this run leaves on disk, written on the child's
        # own restart_interval_s under the instant naming the next
        # downscale discovers; the last one is the run's end state.  How
        # many it wrote and how many it keeps are beside them, and the
        # disk block is the projection the run was admitted on.
        "restart_interval_s": float(cfg.restart_interval_s),
        "checkpoints": [str(path) for path in checkpoints.on_disk()],
        "checkpoints_written": len(checkpoints.written),
        "keep_checkpoints": int(keep_checkpoints or 0),
        "disk": disk,
        "boundary_intervals": len(prepared.boundaries.intervals),
        "boundary_device_resident_bytes": boundary_bytes,
        "boundary_device_reload_count": lateral_boundary_reload_count(child),
        "child_steps": steps,
        "child_simulated_seconds": float(child.elapsed_seconds),
        "child_step_warm_mean_seconds": float(warm.mean()),
        "child_memory_initial": child_memory_initial,
        "child_pool_reserved_peak_bytes": child_pool_reserved_peak,
        "child_health": child_health,
        "outputs": [str(path) for path in output_paths],
        "output_receipts": [_file_receipt(path) for path in output_paths],
        "final_restart": str(restart),
        "final_restart_sha256": _sha256(restart),
        "final_restart_receipt": _file_receipt(restart),
        "wall_seconds": time.perf_counter() - started,
    }
    # THE AEROSOL SOURCE, on the one route that cannot name a dataset.
    # This child is not a real-data initialization: every transported
    # aerosol scalar (QNWFA/QNIFA) and both surface emission fields
    # (QNWFA2D/QNIFA2D) are interpolated out of the archived PARENT
    # history frame, so this run never resolves a climatology and has no
    # ``RealInitResult.aerosol_initialization`` to publish.  Saying
    # nothing would leave an mp=28 child's report looking exactly like a
    # wsm6 child's; saying "no dataset" would be a claim this process
    # cannot make.  So it reports the third state -- applicable, not
    # recorded here -- and names the report that does hold the answer.
    report.update(aerosol_source_report_entry(
        {}, mp_physics=cfg.mp_physics,
        when_unrecorded=(
            "this child's aerosol state is INHERITED, not initialized: "
            "QNWFA/QNIFA and the QNWFA2D/QNIFA2D surface emission fields "
            "are interpolated from the archived parent history frames "
            "listed under parent_frames, and this process runs no aerosol "
            "dataset resolution of its own. Which source filled the "
            "parent's fields -- WRF's monthly WIF climatology or "
            "thompson_init's synthetic profile -- is recorded in the "
            "PARENT run's report, and is the answer for this child too")))
    _publish_report(report, outdir)
    _log("complete", **report)
    return report


def _parser() -> argparse.ArgumentParser:
    from woof.resume import checkpoint_sets_argument

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-history", type=Path, nargs="+", required=True)
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--parent-restart", type=Path)
    evidence.add_argument("--parent-namelist", type=Path)
    parser.add_argument("--parent-domain-id", type=int, default=1)
    parser.add_argument("--child-config", type=Path, required=True)
    parser.add_argument("--parent-grid-ratio", type=int, required=True)
    parser.add_argument("--i-parent-start", type=int, required=True)
    parser.add_argument("--j-parent-start", type=int, required=True)
    parser.add_argument("--max-boundary-interval-seconds", type=float,
                        required=True)
    parser.add_argument("--accepted-parent-cadence", action="store_true",
                        help="provenance marker: the ceiling above was "
                             "taken from the parent archive's own cadence "
                             "(woof downscale --accept-parent-cadence) "
                             "rather than chosen explicitly; recorded in "
                             "report.json")
    parser.add_argument("--child-surface-from", type=Path, default=None,
                        help="child-grid wrfinput/history file supplying "
                             "land identity and soil warm-start state "
                             "(required for surface-physics children)")
    parser.add_argument("--preprocess-backend", choices=("cuda", "cpu", "auto"),
                        default="auto")
    parser.add_argument("--parent-terrain", action="store_const",
                        const="parent", default=None, dest="child_terrain",
                        help="run the child on its parent's interpolated "
                             "terrain, land use and soil instead of its own "
                             "static geography (the default)")
    parser.add_argument("--geog-root", type=Path, default=None,
                        help="the WPS_GEOG tree the child's own static "
                             "geography is built from (default: the child "
                             "config's [static] geog_root, else the staged "
                             "tree)")
    parser.add_argument("--highres-cache", type=Path, default=None,
                        help="the folder the child's high-resolution terrain "
                             "tiles and land-cover rasters are read from and "
                             "cached in (default: the child config's "
                             "[static.highres] cache_root, else the engine's "
                             "per-user cache)")
    parser.add_argument("--health-interval-seconds", type=float, default=60.0)
    parser.add_argument("--render-products", default=None, metavar="LIST",
                        dest="render_products",
                        help="which products this child's frames are drawn "
                             "into <outdir>/png: a comma-separated list of "
                             "catalog slugs, 'all', or 'none'.  Absent draws "
                             "nothing, because this runner is the engine "
                             "door; `woof downscale` is the door that "
                             "defaults to drawing")
    parser.add_argument("--keep-checkpoints", type=checkpoint_sets_argument,
                        default=None, dest="keep_checkpoints", metavar="N",
                        help="how many complete checkpoint sets the child "
                             "keeps in --outdir (default 1, the newest, "
                             "which a downscale from this child binds to); "
                             "0 keeps every set")
    parser.add_argument("--no-memory-gate", action="store_true",
                        dest="no_memory_gate",
                        help="run a child whose priced peak envelope exceeds "
                             "this card's free memory anyway: the envelope "
                             "is an upper bound and the card's own "
                             "allocation then decides; a child state too big "
                             "to build at all is still refused")
    parser.add_argument("--warm-from", type=Path, default=None,
                        dest="warm_from", metavar="CHECKPOINT",
                        help="carry this earlier child's state onto the "
                             "ground it shares with this one "
                             "(woof.offline_child_warm); needs --warm-shift")
    parser.add_argument("--warm-shift", default=None, dest="warm_shift",
                        metavar="DI,DJ",
                        help="this child's placement minus the earlier "
                             "one's, in child cells")
    parser.add_argument("--warm-strip", type=Path, default=None,
                        dest="warm_strip", metavar="NPZ",
                        help="accumulated precipitation for the freshly "
                             "covered strip, per slot")
    parser.add_argument("--outdir", type=Path, required=True)
    return parser


def warm_start_argument(warm_from, warm_shift, warm_strip=None):
    """The runner's ``warm_start`` entry from the three flags, or None."""
    if warm_from is None and warm_shift is None:
        if warm_strip is not None:
            raise OfflineChildContractError(
                "--warm-strip belongs to a warm start: pass --warm-from and "
                "--warm-shift with it")
        return None
    if warm_from is None or warm_shift is None:
        raise OfflineChildContractError(
            "a warm start needs both --warm-from CHECKPOINT and --warm-shift "
            "DI,DJ (the move in child cells)")
    from woof.offline_child_warm import parse_warm_shift

    try:
        shift = parse_warm_shift(warm_shift)
    except ValueError as error:
        raise OfflineChildContractError(str(error)) from None
    return {"checkpoint": Path(warm_from), "shift": shift,
            "strip": None if warm_strip is None else Path(warm_strip)}


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--show-capabilities"]:
        print(json.dumps(_CAPABILITIES, sort_keys=True))
        return 0
    args = _parser().parse_args(arguments)
    args.warm_start = warm_start_argument(
        args.warm_from, args.warm_shift, args.warm_strip)
    from woof.core.resident_admission import memory_gate_override

    try:
        with memory_gate_override(args.no_memory_gate):
            report = run(args)
    except KeyboardInterrupt:
        # The contract `woof.cli.main` keeps for every subcommand: one
        # line and exit 130, not a traceback.  The stop itself is already
        # written into the run's own records.
        print("woof.offline_child_run: stopped by request (SIGINT); the "
              "run folder records the stop.", file=sys.stderr)
        return 130
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
