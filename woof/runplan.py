"""A versioned plan in, a structured event stream out.

Every other front door in this package talks to a person: it prints a
resolved-config report, a progress line, a refusal sentence.  A program
driving woof as a subprocess -- a GUI, a scheduler, a fleet controller
-- has to read those same facts, and until now its only route to them
was to parse the prose.  Prose is not an interface.  It gets rewritten
for clarity, and the rewrite silently breaks the consumer.

``woof run-plan PLAN.json`` is the interface.  A caller writes ONE JSON
document naming which existing route to execute and which existing
config to execute it with, and gets back an append-only JSONL stream in
which every fact the human output carries is a typed field on a typed
event.  Nothing about the model changes: the plan is an ENVELOPE over
the config system, it is resolved by the same
:func:`woof.case_data.load_experiment_case` seam ``woof run`` uses,
and it is executed by the same :func:`woof.runtime.run_experiment`.
There is no second config format and no forked validation.

The three documents
-------------------

``gpuwm.run-plan.v1`` -- the plan.  ``schema``, ``name``, ``route``,
``config`` (``{"path": ...}`` or ``{"inline": "..."}``), and the
optional ``fetch`` / ``output_root`` / ``run_options``.  Unknown
top-level keys and unknown schema ids are refused, not ignored: a
dropped key runs a default under the name of your value.

``gpuwm.run-plan.event.v1`` -- one JSON object per line of
``<run_dir>/events.jsonl``, mirrored verbatim to stdout.  Every line
carries ``schema_version``, a monotonic ``sequence``, ``emitted_unix_ms``
and an ``event`` tag; the event's own fields are flattened alongside.
The tags are :data:`EVENT_TAGS`, and a ``warning`` event's own ``code``
is one of :data:`WARNING_CODES`.

``gpuwm.run-manifest.v1`` -- ``<run_dir>/run-manifest.json``, written
before any work starts.  It carries this process's pid and the absolute
path of every stream a consumer may want, INCLUDING the two the rest of
the package already owns.

Reattach: read the heartbeat, do not own the pipe
-------------------------------------------------

This module publishes no progress state of its own.
:mod:`woof.supervisor` already writes ``run-progress.json``
(``gpuwm.run-progress/v1``) atomically on every step, and it stays the
only writer of it: :class:`RunObserver` COMPOSES with
:class:`woof.supervisor.RuntimeHeartbeat` rather than replacing it, so
a run-plan run leaves exactly the same heartbeat a ``woof run`` leaves.

So a consumer that attaches to a run already in flight does three
things, in this order:

1. Read ``run-manifest.json`` for the paths and the pid.
2. Read ``run-progress.json`` for CURRENT state.  That file is the
   authoritative anchor -- it is atomically republished, it is what the
   supervisor's own recovery reads, and it is one small read rather
   than a replay.
3. Replay ``events.jsonl`` from byte zero for HISTORY (it is the
   complete record, never rotated or truncated), then tail it for live
   detail.

A consumer that treats the event stream as the anchor will be wrong
exactly once: after a crash between the last event flush and the
process exit.  The heartbeat is the thing that is durable by design.

Query modes
-----------

``--resolve``, ``--estimate`` and ``--probe`` answer a front end's three
pre-flight questions without running anything, each as one JSON document
on stdout.  They live here rather than in the front end because the
answers are derived from this package's own machinery -- the config
loaders, :mod:`woof.core.preflight`'s VRAM itemization,
:mod:`woof.doctor`'s estate checks -- and a front end that
reimplemented them would be reporting its own arithmetic under woof's
name.  Where this package has no measured number for something
(wall-time for an arbitrary configuration), the field is ``null`` with
its ``basis`` stated.  Nothing is estimated by guess.

Nothing silent
--------------

``resolved_plan`` carries ``automatic_resolutions``: one entry for every
value this pipeline chose on its own -- an omitted plan key taking its
default, a schema default filling an unspelled config key, a per-domain
timestep derived down the nest ratio chain.  A consumer can render that
list and a reader can see, before the run, every number nobody typed.
"""

from __future__ import annotations

from woof.explain import warning_scope as _review_warning_scope

import argparse
import contextlib
from contextvars import ContextVar
import copy
import dataclasses
import functools
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import threading
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from woof.explain import layered, split as _split_message
from woof.stage_timing import (ledger_main, bind_ledger, ForecastLedger,
                               register_ledger_observer, current_ledger, end_admission)
from woof.timeline.ledger import Ledger
from woof.timeline.taxonomy import RUN_STAGES, GO_STAGES


#: The plan document a caller writes.
PLAN_SCHEMA = "gpuwm.run-plan.v1"

#: The ``schema_version`` on every line of the event stream.  Its shape
#: -- envelope keys first, event-specific fields flattened after -- is
#: the one already proven by the Rust side of this stack, so a reader
#: written for one stream needs no second framing for the other.
EVENT_SCHEMA = "gpuwm.run-plan.event.v1"

#: The attach manifest.
MANIFEST_SCHEMA = "gpuwm.run-manifest.v1"

#: The documents this module answers ``--resolve``/``--estimate``/
#: ``--probe``/``--catalog``/``--sources``/``--physics-profiles`` with.
#: Versioned for the same reason the others are: a front end pins what
#: it parses.
RESOLVE_SCHEMA = "gpuwm.run-plan.resolved.v1"
ESTIMATE_SCHEMA = "gpuwm.run-plan.estimate.v1"
PROBE_SCHEMA = "gpuwm.run-plan.probe.v1"
CATALOG_SCHEMA = "gpuwm.run-plan.catalog.v1"
SOURCES_SCHEMA = "gpuwm.run-plan.sources.v1"
PHYSICS_PROFILES_SCHEMA = "gpuwm.run-plan.physics-profiles.v1"

EVENTS_FILENAME = "events.jsonl"
MANIFEST_FILENAME = "run-manifest.json"

#: The five stages a run passes through, in order.  ``fetch`` is skipped
#: when the plan declares no ``[fetch]``; every other stage always runs
#: and always emits its pair of events, so a consumer's stage timeline
#: has no holes to interpret.
STAGES = ("fetch", "prepare", "initialize", "forecast", "finalize")

#: Every event tag this module will ever emit.  A consumer switching on
#: ``event`` can be exhaustive against this tuple.
EVENT_TAGS = (
    "plan_accepted", "resolved_plan", "stage_started", "stage_finished",
    "model_progress", "output_committed", "first_products_ready",
    "live_products_ready", "grib2_export_progress",
    # The any-point column store (woof.live_columns): each frame written, then the total.
    "columns_progress", "columns_written",
    "fetch_started", "fetch_progress", "fetch_completed",
    # Chained preparation: the prepared head is published (the forecast
    # starts beside the rest of the preparation), and later the seal.
    "prepare_head_ready", "prepare_sealed",
    # ... and the forecast waiting at a seam for an interval not yet
    # prepared, with how long it waited (``cause: preparation``).
    "boundary_wait_started", "boundary_wait_finished",
    # Runs as the source posts: the window's posting schedule, each lead
    # as one host first holds it and as it is fetched and verified, the
    # run blocked on a lead (every 60 s while it lasts, so a reader
    # tailing the stream can tell a wait from a hang), and the terminal
    # record of a lead later than its budget, just before `failed`.
    "posting_schedule", "lead_posted", "lead_ready",
    "source_wait_started", "source_wait_progress", "source_wait_finished",
    "source_behind",
    "warning", "completed", "failed",
)

#: The fields each posting and wait event carries (design A136, 3.5); a
#: route that emits one of these tags emits every field, ``None`` where
#: the value is not known.
POSTING_EVENT_FIELDS = {
    "posting_schedule": (
        "source", "member", "cycle", "as_posted", "shape", "streams", "why",
        "late_after_minutes", "start_needs", "expected_ready_at",
        "expected_final_at", "leads", "schedule_path", "table_sha256"),
    "lead_posted": (
        "source", "cycle", "lead", "valid_time", "expected_at",
        "first_seen_at", "minutes_after_expected", "posted_when_first_asked",
        "endpoint"),
    "lead_ready": (
        "source", "cycle", "lead", "valid_time", "bytes", "fetch_seconds",
        "marker_sha256"),
    "source_wait_started": (
        "phase", "source", "cycle", "lead", "valid_time", "expected_at",
        "late_at", "waited_seconds", "model_elapsed_seconds",
        "model_valid_time", "interval", "reason"),
    "source_wait_progress": (
        "phase", "source", "cycle", "lead", "valid_time", "expected_at",
        "late_at", "waited_seconds", "model_elapsed_seconds",
        "model_valid_time", "interval", "reason"),
    "source_wait_finished": (
        "phase", "source", "cycle", "lead", "waited_seconds",
        "first_seen_at", "model_elapsed_seconds", "model_valid_time"),
    "boundary_wait_started": (
        "interval", "reason", "cause", "model_elapsed_seconds",
        "model_valid_time"),
    "boundary_wait_finished": (
        "interval", "seconds", "cause", "model_elapsed_seconds",
        "model_valid_time"),
    "source_behind": (
        "source", "cycle", "lead", "valid_time", "expected_at", "late_at",
        "late_after_minutes", "last_answer", "model_elapsed_seconds",
        "model_valid_time", "frames_kept", "checkpoint"),
}


def source_behind_fields(details: Mapping[str, Any]) -> dict[str, Any]:
    """A ``source_behind`` event's fields from the late lead's record."""

    return {key: details.get(key)
            for key in POSTING_EVENT_FIELDS["source_behind"]}

#: Every ``code`` a ``warning`` event carries, and what each one means.
#:
#: ``warning`` is one tag in :data:`EVENT_TAGS`, and the thing a reader
#: actually switches on is the code inside it.  That code was a free
#: string documented nowhere, so a route could invent one and the record
#: it wrote was a line nothing could key on.  This is the vocabulary:
#: a new code is added HERE, with its meaning, in the same commit as the
#: route that emits it, and ``tests/test_runplan.py`` fails a code that
#: reaches the stream without a line in this table.
#:
#: The one family spelled by prefix is the native producer's relay,
#: ``native_producer_<tag>``: it forwards the inner run's own terminal
#: tags under this door's stream, so its codes are as open as
#: :data:`EVENT_TAGS` itself.
WARNING_CODES = {
    "local_publish_failed": "rendered pictures could not be committed to the local static web root",
    "chain_stage_failed":
        "one stage of a chained run failed; the chain's own reader is "
        "told which stage before the run's terminal event",
    "secondary_source_behind":
        "a source lead timed out after preparation or forecast already "
        "failed; the first failure remains the run's cause, and the "
        "source_behind field records the later lead",
    "compose_scratch_may_not_fit":
        "said before the download: the decoded frame stream a regional "
        "source's preparation stages may not fit the disk that holds its "
        "scratch folder (`folder`), where only the layers the atmospheric "
        "window cannot crop are certain; the run is not refused, and "
        "WOOF_COMPOSE_SCRATCH moves the stream to a disk with room",
    "cycle_frame_failed":
        "a boundary `woof cycle` kept could not be written as a frame, "
        "so it is not drawn; its anchor and receipt are unaffected",
    "cycle_pictures_none":
        "`woof cycle` draws nothing for this run, and the message says "
        "why: the parent's planes cannot be placed on a map (an MPAS "
        "mesh, or a grid with no coordinates named for it)",
    "early_render_kept":
        "this run did not finish and the pictures drawn while it ran "
        "were KEPT rather than removed; the event carries how many are "
        "on disk and the path of the banner beside them that states "
        "where the forecast stopped",
    "event_tail_recovered":
        "this run's event history ended in a line cut short by a writer "
        "that was stopped; the next writer moved those bytes to the file "
        "the event names (preserved_path) and the history continues after "
        "its last whole record",
    "first_products_empty":
        "the first committed frame produced no picture for the requested "
        "products, so nothing was published early and the finalize stage "
        "draws it with the rest",
    "first_products_failed":
        "the early render raised; the finalize stage draws every frame "
        "as it would have without one",
    "first_products_incomplete":
        "the early render drew some pictures and then exited nonzero; "
        "they are kept and the finalize stage draws the frame again",
    "first_products_not_dispatched":
        "no frame reached the early render before the forecast ended, so "
        "there was no early picture to publish",
    "first_products_timeout":
        "the early render did not finish within its wait, so the "
        "finalize stage stopped holding a finished forecast for it; that "
        "render was ended and publishes nothing",
    "grib2_export_failed":
        "native GRIB2 export failed; the message names the cause and "
        "the GRIB2 output is incomplete",
    "columns_not_armed":
        "the any-point column store (woof.live_columns) could not start; "
        "the run goes on and writes no store, and the message says why",
    "columns_frames_unknown":
        "the prepared runner could not read which frames carry the 3-D "
        "state from the experiment (the message says why), so the "
        "any-point column store tries every frame on the history cadence; "
        "the run is unaffected",
    "columns_not_written":
        "one committed frame's any-point column store file was not "
        "written (the frame is named); the run and its other frames are "
        "unaffected, and soundings at that hour answer \"no column store\"",
    "live_products_empty":
        "a frame drawn as it landed produced no picture; the end-of-run "
        "render draws it with the rest",
    "live_products_failed":
        "drawing a frame as it landed raised; the end-of-run render draws "
        "it with the rest",
    "live_products_incomplete":
        "a frame drawn as it landed drew some pictures and then the "
        "renderer exited nonzero; they are kept and the end-of-run render "
        "draws the frame again",
    "live_products_timeout":
        "the frame being drawn when the forecast ended did not finish "
        "within its wait, so its render was ended and publishes nothing; "
        "the end-of-run render draws every frame it cannot prove was "
        "published",
    "live_products_stopped":
        "drawing a frame as it landed was stopped before it finished "
        "(the render exited on an interrupt), so none of it was "
        "published; the frame is on disk",
    "first_products_stopped":
        "drawing the first committed frame early was stopped before it "
        "finished (the render exited on an interrupt), so none of it was "
        "published; the frame is on disk",
    "forecast_output_recovery":
        "an earlier attempt's output remains beside this one's; the "
        "event names both directories",
    "kernel_compile_progress":
        "the kernel loader compiled one GPU module for this card (it was "
        "not in the kernel cache); the event carries the module, its "
        "seconds and the running count and seconds, and the words "
        "'compiling GPU kernels' for a page to show",
    "inline_config_materialized":
        "an inline config was written to a file because this route binds "
        "its configuration by path",
    "library_warning":
        "a library this run drives raised a warning; the action and the "
        "reason are carried verbatim",
    "native_producer_completed":
        "the native producer finished and the caller is collecting its "
        "receipts",
    "native_producer_failed":
        "the native producer failed; its own message is carried",
    "preparation_progress":
        "a coarse sample from a preparation phase that runs out of "
        "process",
    "route_inputs_rendered":
        "a native HRRR run found only part of the route's namelist set "
        "beside its configuration, so it wrote the whole set from the "
        "configuration into its run folder; the event names the files "
        "beside it that it did not read (`unread`), the ones that were "
        "missing, and where the set it ran was written (`written`)",
    "render_scratch_left":
        "a stage that draws ended (passed or failed) with working stores "
        "its renders could not remove beside the delivery; the event "
        "carries the scratch root, how many stores and bytes there were, "
        "how many the chain removed once every render had exited, and how "
        "many it could not and marked for the next run in the same folder",
    "render_scratch_swept":
        "a run removed working stores an earlier run in the same folder "
        "marked because it could not remove them when it ended; the event "
        "carries the folder and how many went",
    "render_basemap_missing":
        "the renderer this run drives has no map assets, so every picture "
        "is drawn with no coastlines, borders or state lines; the message "
        "says so, `remedy` carries the pip line that restores them, and "
        "`render_stage` says whether it was found while the forecast drew "
        "(`as-drawn`) or at `finalize`",
    "unmapped_pipeline_phase":
        "the pipeline reported a preparation phase this door has no "
        "stage for; it is attributed to the open stage rather than "
        "dropped",
    "forecast_restarted":
        "the hosted forecast ended one attempt and starts again in this "
        "process (a head-bound tree whose terrain clock moved runs again "
        "on its sealed preparation); `reason` says why, and the attempt's "
        "outputs are kept beside the new ones",
    "stability_retry":
        "a static nested forecast failed its full-state health check and "
        "bounded checkpoint recovery acted on it: it rewinds to the last "
        "proven checkpoint and runs again with half-sized adaptive step "
        "caps, at most twice; `recovery` carries the retry number, its "
        "phase (validating_restore, resumed, refused or interrupted), the "
        "checkpoint, the cause, the per-domain clock_policy_changes and "
        "the path of the stability-recovery.json receipt",
}

#: The one code family spelled by prefix rather than in full.
WARNING_CODE_PREFIXES = ("native_producer_",)

#: Envelope keys an event's own fields may not shadow.
_ENVELOPE_KEYS = frozenset({
    "schema_version", "sequence", "emitted_unix_ms", "event"})

# A prepared front door can enter another native run-plan in this process.
# The scope belongs to that call, not to a global current run or a PID lookup.
_PREPARED_PARENT: ContextVar[Any] = ContextVar("gpuwm_prepared_parent", default=None)

_TOP_LEVEL_KEYS = frozenset({
    "schema", "name", "route", "config", "fetch", "output_root",
    "run_options"})
_REQUIRED_KEYS = ("schema", "name", "route", "config")
_CONFIG_KEYS = frozenset({"path", "inline", "intent"})
_FETCH_KEYS = frozenset({"args"})

#: ``config.intent`` keys, and the ``woof domain`` flag each one is.
#:
#: A mapping and not a schema.  Intent is not a third config format --
#: it is the wizard's own question list, spelled as JSON so a front end
#: can build it from typed fields instead of assembling a command line.
#: Every value is validated by the wizard's REAL parser and every
#: refusal is the wizard's own, so this table is the only thing that
#: could go stale, and a key whose flag disappears fails loudly at the
#: parser rather than being quietly dropped.
#:
#: ``--out`` is deliberately absent: run-plan owns where the generated
#: config lands, and a plan that could redirect it would be able to
#: write outside its own run directory.
_INTENT_FLAGS = {
    "point": "--point",
    "polygon": "--polygon",
    "buffer_km": "--buffer-km",
    "projection": "--projection",
    "name": "--name",
    "card": "--card",
    "vram_gib": "--vram-gib",
    "ladder": "--ladder",
    "root_dx_km": "--root-dx",
    "chain": "--chain",
    "physics_profile": "--physics-profile",
    "cumulus": "--cumulus",
    "physics_choices": "--physics-choices",
    "hours": "--hours",
    "source": "--source",
    "cycle": "--cycle",
    "forecast_start_hour": "--forecast-start-hour",
    "member": "--member",
    "cadence": "--cadence",
    "era5_product": "--era5-product",
    "era5_provider": "--era5-provider",
    "data_dir": "--data-dir",
    "forcing": "--forcing",
    "vtable": "--vtable",
    "geog_root": "--geog-root",
    "history_interval_s": "--history-interval",
    "nest_history_interval_s": "--nest-history-interval",
    "nz": "--nz",
    "isftcflx": "--isftcflx",
    "clock": "--clock",
    "tiles": "--tiles",
    "ack": "--ack",
    "point_extent_km": "--point-extent-km",
}

#: How each intent key actually reaches the chain that executes it.
#:
#: This table exists because three keys did not reach it at all.  The
#: wizard accepts ``--geog-root``, ``--data-dir``, ``--forcing`` and
#: ``--vtable`` and writes the last three into ``[case_data]`` -- a
#: table it only emits for ERA5.  On the prepared route (gfs) there is
#: no ``[case_data]``, so those values were validated, accepted, and
#: then silently dropped: a plan naming a non-default geography tree ran
#: against the default one.
#:
#: The values:
#:
#: ``"config"``      the wizard bakes it into the generated TOML, and
#:                   every consumer reads it from there.
#: ``"go:--flag"``   it does NOT survive into the config on the prepared
#:                   route and must be forwarded to ``woof go``.
#: ``"case_data"``   it lands in ``[case_data]``, so it is meaningful
#:                   only on a route that has one, and is refused
#:                   loudly on a route that does not.
#:
#: Every key in :data:`_INTENT_FLAGS` must appear here; a test fails if
#: one does not, so a key cannot be added without answering "and how
#: does that reach the thing that runs?".
_INTENT_DELIVERY = {
    "point": "config",
    "polygon": "config",
    "buffer_km": "config",
    "projection": "config",
    "name": "config",
    "card": "config",
    "vram_gib": "config",
    "ladder": "config",
    "root_dx_km": "config",
    "chain": "config",
    "physics_profile": "config",
    # Whether the named suite's root cumulus is the suite's or the
    # grid's: the wizard writes the resulting cu_physics into the config.
    "cumulus": "config",
    # Schemes picked in place of the suite's own (the physics composer's
    # checked choices): the wizard writes them into the config the way
    # `woof physics-catalog --into` writes a mix, on every size its fit
    # tries.
    "physics_choices": "config",
    "hours": "config",
    "source": "config",
    "cycle": "config",
    "forecast_start_hour": "config",
    "member": "config",
    "cadence": "config",
    "era5_product": "config",
    "era5_provider": "config",
    "history_interval_s": "config",
    "nest_history_interval_s": "config",
    # The vertical level count: the wizard resamples its own eta ladder to
    # it and writes both into the generated config, and it prices the fit
    # at that count.
    "nz": "config",
    "isftcflx": "config",
    # How the run steps: the wizard writes use_adaptive_time_step into
    # [shared] for an adaptive clock and nothing for a fixed one.
    "clock": "config",
    # The streaming mode: the wizard sizes the fit with it and writes the
    # resulting [tiles] table into the generated config.
    "tiles": "config",
    # Governed-experiment declarations, one id per item: the wizard writes
    # them verbatim into [experiment].acknowledgements.
    "ack": "config",
    # The largest root extent a point request is sized to: it shapes the
    # grid the wizard writes, and nothing downstream reads it again.
    "point_extent_km": "config",
    # `go` defaults its data directory to <outdir>/data and never reads
    # the [fetch].out hint the wizard wrote, so this has to travel as a
    # flag or it does not travel.
    "data_dir": "go:--data-dir",
    # The wizard saves the geography root for every source. The plan
    # also forwards its explicit root as a launch override.
    "geog_root": "go:--geog-root",
    # Read by the HRRR preparer as a flag; on the gfs chain the wizard
    # bakes the resulting physics into the config and `go` needs no
    # flag, so this is declared config-delivered and the HRRR arm reads
    # it off the plan directly.
    "physics_profile": "config",
    # ERA5's declared inputs.  The prepared chain fetches its own GRIB
    # and takes its Vtable from the bridge, so there is nothing for
    # these to mean there and no flag to carry them.
    "forcing": "case_data",
    "vtable": "case_data",
}

#: The filename the generated config takes inside the run directory.
GENERATED_CONFIG_NAME = "intent-config.toml"

def intent_drivability() -> dict[str, dict[str, Any]]:
    """Return the shared source preparation facts used by input validation."""
    from woof.source_drivability import intent_drivability as shared_drivability
    return shared_drivability()


#: ``woof run``'s own documented default output directory.  A plan that
#: omits ``output_root`` lands where the command it wraps would have.
DEFAULT_OUTPUT_ROOT = Path("out") / "run"

#: Pipeline preparation phases, mapped to the stage each one belongs to.
#: The literals are :func:`woof.runtime._preparation_progress`'s own --
#: this table reads them, it does not author them.  A phase absent here
#: does not silently land in whichever stage happens to be open: it
#: emits a ``warning`` naming itself, so a pipeline that grows a phase
#: is reported rather than mis-filed.
#:
#: The three prepared-runner phases belong to ``forecast``: every prepared
#: chain opens that stage itself before it hands the runner over, and the
#: runner's first acts are to check the prepared bundle and load it.  They
#: were missing here, so every prepared ``woof go`` printed two of these
#: warnings (A155); tests/test_runplan.py holds this table to every phase
#: the pipeline's source emits.
_PHASE_STAGES = {
    "quarantine-wrfout": "prepare",
    "resolve-terrain-clock": "prepare",
    "resolve-schedule": "prepare",
    "prepare-case": "prepare",
    "build-domain-tree": "prepare",
    "initialize-health-validator": "initialize",
    "cold-start-wrfout": "initialize",
    "validate-checkpoint": "initialize",
    "restore-checkpoint": "initialize",
    "restore-tree-checkpoint": "initialize",
    "restore-nest-lifecycle": "initialize",
    "initialize-domain-writers": "initialize",
    "initial-health-gate": "initialize",
    "validate-prepared-inputs": "forecast",
    "restore-prepared-domain": "forecast",
    "restore-prepared-domain-tree": "forecast",
}


class StageExitError(RuntimeError):
    """A CLI stage returned a failure after writing its own diagnostic."""

    def __init__(self, stage: str, code: int):
        self.stage, self.exit_code = stage, int(code)
        if code < 0:
            import signal
            try:
                status = signal.Signals(-code).name
            except ValueError:
                status = f"signal {-code}"
            message = f"{stage} stopped by {status}."
        else:
            message = f"{stage} failed (exit {code})."
        super().__init__(message)


class ChainInterrupted(RuntimeError):
    """``woof go`` answered a stop with 130 while running the chain.

    Raised by the prepared route in place of a failure so that
    :func:`execute_plan` exits 130: the stop is the user's, not a defect
    of the chain, and the desktop reads the exit code back from the
    worker's receipt to decide whether the run is "stopped" or "failed".
    """

    exit_code = 130

    def __init__(self, stage: str):
        self.stage = stage
        super().__init__(f"interrupted during {stage}")


class PlanError(ValueError):
    """A plan document this front door refuses to execute.

    ``ValueError`` because that is what every refusal in this package
    travels as, and what :func:`woof.cli.main` prints as one sentence
    at exit 2 rather than as a traceback.

    ``memory`` is set on the refusal of an intent whose grid does not fit
    its card: the wizard's :meth:`~woof.domain_wizard.DomainFitError.memory_record`,
    which the query modes of :func:`run_plan_main` print as the refusal's
    document.

    ``remedy`` is what to do, for a refusal whose fix is not an edit to
    the plan document: the ``failed`` event carries it (:func:`_remedy`)
    in place of the class's plan-document line.  A run refused for a
    geography tree never set up on the computer told the page to fix
    its plan document, which could not help.
    """
    memory: dict[str, Any] | None = None

    def __init__(self, message: object = "", *, remedy: str | None = None):
        super().__init__(message)
        self.remedy = remedy


# ---------------------------------------------------------------------------
# The plan document
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunPlan:
    """One validated plan, with every path made absolute."""

    name: str
    route: str
    config_path: Path | None
    config_inline: str | None
    config_intent: Mapping[str, Any] | None
    config_base_dir: Path
    fetch_arguments: tuple[str, ...] | None
    output_root: Path
    run_options: Mapping[str, Any]
    sha256: str
    source: str
    automatic_resolutions: tuple[Mapping[str, Any], ...]

    @property
    def run_dir(self) -> Path:
        """The one directory this run writes into.

        ``output_root`` IS the run directory, not a parent to derive one
        under: deriving would mean inventing a directory name from the
        plan's ``name``, and a caller who cannot predict where its
        outputs land cannot collect them.
        """

        return self.output_root

    @property
    def config_kind(self) -> str:
        """``"path"``, ``"inline"`` or ``"intent"``."""

        if self.config_path is not None:
            return "path"
        return "inline" if self.config_inline is not None else "intent"

    def config_bytes(self) -> bytes:
        """The config TOML this plan names, as bytes.

        One accessor for the two spellings that ARE a config already.
        ``intent`` is not one of them: it has to be generated first,
        which needs a directory to generate into, so it is resolved by
        :func:`resolve_plan` rather than read here.
        """

        if self.config_inline is not None:
            return self.config_inline.encode("utf-8")
        if self.config_path is None:
            raise PlanError(
                "an intent plan has no config bytes until the wizard has "
                "generated them; resolve_plan does that")
        try:
            return Path(self.config_path).read_bytes()
        except OSError as error:
            # An unreadable config.path is a PLAN defect, not a crash.
            # This read is the first thing every mode does -- --resolve,
            # --estimate and the execution road all reach it through
            # resolve_plan -- so an OSError escaping here left the two
            # query modes printing a bare FileNotFoundError traceback at
            # exit 1, and the execution road emitting a `failed` event
            # whose remedy named a declared [case_data] input when the
            # file that was missing is the config itself.  PlanError is
            # a ValueError, which woof.cli.main prints as one sentence
            # at exit 2 -- the exit code every other refusal in this
            # front door already uses.
            raise PlanError(layered(
                f"run plan {self.source} names config.path "
                f"{self.config_path}, which woof run-plan cannot read: "
                f"{error.strerror or error}.  Nothing was started.  "
                "Point 'config.path' at the TOML on disk -- a relative "
                "path resolves against the plan's own directory, not "
                "the working directory -- or carry the config text "
                "itself in 'config.inline'.",
                "That file IS the configuration, so every mode reads it "
                "before anything else: --resolve, --estimate and the "
                "execution road all reach it through resolve_plan.  "
                "There is no half-answer to give from a plan whose "
                "config is absent, and the estimate and VRAM figures a "
                "front end shows next are derived from it.")) from error


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{label} must be a non-empty string")
    result = value.strip()
    if any(character in result for character in "\r\n"):
        raise PlanError(f"{label} must fit on one line")
    return result


def _reject_unknown(mapping: Mapping[str, Any], known, label: str) -> None:
    unknown = sorted(set(mapping) - set(known))
    if not unknown:
        return
    raise PlanError(layered(
        f"{label} does not have the key(s) {unknown}; no key is "
        "ignored, because a dropped key runs a default under the name "
        "of your value.",
        f"Known {label} keys: {sorted(known)}."))


def build_plan(raw: Mapping[str, Any], *, source: str,
               base_dir: str | Path, sha256: str) -> RunPlan:
    """Validate a parsed plan document and build the :class:`RunPlan`.

    Separate from :func:`load_plan` for the reason
    :func:`woof.experiment.build_experiment` is separate from
    :func:`woof.experiment.load_experiment`: the validation is the
    part a test wants to reach without a file on disk.
    """

    if not isinstance(raw, dict):
        raise PlanError("run plan must be a JSON object")
    _reject_unknown(raw, _TOP_LEVEL_KEYS, "run plan")
    missing = [key for key in _REQUIRED_KEYS if key not in raw]
    if missing:
        raise PlanError(
            f"run plan {source} is missing required key(s) {missing}")
    if raw["schema"] != PLAN_SCHEMA:
        raise PlanError(layered(
            f"{source} is not a {PLAN_SCHEMA} document: schema is "
            f"{raw['schema']!r}.",
            "The schema id carries this document's version.  A reader "
            "that accepted an unrecognized id would be guessing which "
            "fields it holds, which is how a plan runs as something "
            "other than what it says."))

    base = Path(base_dir)
    resolutions: list[dict[str, Any]] = []

    name = _nonempty_string(raw["name"], "run plan 'name'")
    route = _nonempty_string(raw["route"], "run plan 'route'")
    if route not in ROUTES:
        raise PlanError(layered(
            f"run plan {source} names route {route!r}, which this build "
            f"does not have.",
            "Known routes: " + ", ".join(
                f"{key} ({value.summary})"
                for key, value in sorted(ROUTES.items())) + "."))

    config = raw["config"]
    if not isinstance(config, dict):
        raise PlanError("run plan 'config' must be an object with "
                        "exactly one of 'path', 'inline' or 'intent'")
    _reject_unknown(config, _CONFIG_KEYS, "run plan 'config'")
    spelled = sorted(set(config) & _CONFIG_KEYS)
    if len(spelled) != 1:
        raise PlanError(layered(
            "run plan 'config' must carry exactly ONE of 'path' (a TOML "
            "on disk), 'inline' (the TOML text itself) or 'intent' (the "
            f"shape of the run, for the wizard to write), got {spelled}.",
            "They are three ways of saying the same thing, so two of "
            "them is a question about which one wins, and this front "
            "door does not answer questions like that by picking."))
    config_path = None
    config_inline = None
    config_intent = None
    if spelled == ["path"]:
        config_path = _absolute(config["path"], base,
                                "run plan 'config.path'")
    elif spelled == ["inline"]:
        if not isinstance(config["inline"], str) or not config["inline"]:
            raise PlanError("run plan 'config.inline' must be non-empty "
                            "TOML text")
        config_inline = config["inline"]
    else:
        config_intent = _build_intent(config["intent"], route=route,
                                      base=base)
        resolutions.append({
            "scope": "config", "key": "generated_by",
            "value": "woof domain", "basis": "intent_route",
            "note": "the config is written by the wizard from this "
                    "plan's intent, not authored by the caller; its "
                    "full text is on the resolved_plan event"})

    fetch_arguments = None
    fetch = raw.get("fetch")
    if fetch is not None:
        if not isinstance(fetch, dict):
            raise PlanError("run plan 'fetch' must be an object")
        _reject_unknown(fetch, _FETCH_KEYS, "run plan 'fetch'")
        if "args" not in fetch:
            raise PlanError(
                "run plan 'fetch' must carry 'args': the argv list "
                "`woof fetch` itself takes.  The flags are not "
                "restated here, because a second spelling of them "
                "would be a second thing to keep in step.")
        if not isinstance(fetch["args"], list) or not fetch["args"]:
            raise PlanError("run plan 'fetch.args' must be a non-empty "
                            "list of argv strings")
        fetch_arguments = tuple(
            _nonempty_string(argument, f"run plan 'fetch.args[{index}]'")
            for index, argument in enumerate(fetch["args"]))
        _validate_fetch_arguments(fetch_arguments)

    if "output_root" in raw:
        output_root = _absolute(raw["output_root"], base,
                                "run plan 'output_root'")
    else:
        output_root = (base / DEFAULT_OUTPUT_ROOT).resolve()
        resolutions.append({
            "scope": "plan", "key": "output_root",
            "value": str(output_root), "basis": "front_door_default",
            "note": "woof run's own default --outdir, resolved against "
                    "the plan's directory"})

    options = raw.get("run_options", {})
    if not isinstance(options, dict):
        raise PlanError("run plan 'run_options' must be an object")
    known_options = ROUTES[route].run_options | {"devices"}
    _reject_unknown(options, known_options, "run plan 'run_options'")
    resolved_options: dict[str, Any] = {}
    for key in sorted(known_options):
        if key == "devices" and key not in options:
            continue
        if key in options:
            resolved_options[key] = _run_option(key, options[key], base)
            continue
        resolved_options[key] = _RUN_OPTION_DEFAULTS[key]
        resolutions.append({
            "scope": "run_options", "key": key,
            "value": _RUN_OPTION_DEFAULTS[key], "basis": "schema_default"})

    if (config_intent or {}).get("physics_choices") and resolved_options.get("physics_profile"):
        # The preparer holds a config to a named suite switch for switch,
        # and the choices are written over that suite, so the run would be
        # refused at preparation, after its download.
        raise PlanError(layered(
            f"run plan {source} asserts run_options.physics_profile "
            f"{resolved_options['physics_profile']!r} and also picks "
            "config.intent.physics_choices, which the wizard writes over "
            "that suite, so the preparer would refuse the config it makes.",
            "Name the suite the choices change as "
            "config.intent.physics_profile instead; it is the base, and "
            "nothing asserts it."))
    section_refusal = _section_refusal(resolved_options)
    if section_refusal is not None:
        raise PlanError(section_refusal)
    plan = RunPlan(
        name=name, route=route, config_path=config_path,
        config_inline=config_inline, config_intent=config_intent,
        config_base_dir=base,
        fetch_arguments=fetch_arguments, output_root=output_root,
        run_options=resolved_options, sha256=sha256, source=source,
        automatic_resolutions=tuple(resolutions))
    if resolved_options.get("grib2"):
        _grib2_member_contract(plan, b"" if config_intent is not None else plan.config_bytes())
    if route == "prepared":
        if resolved_options.get("restart") and not resolved_options.get("prepared_root"):
            raise PlanError(
                "Prepared restart needs run_options.prepared_root naming the existing "
                "bundle that wrote this checkpoint (woof go --prepared-root DIR). "
                "Nothing will be fetched or prepared for a resume.")
        if resolved_options.get("wps_namelist") and not resolved_options.get("prepared_root"):
            raise PlanError("run_options.wps_namelist requires prepared_root; "
                            "it names the WPS authority of an existing bundle")
        if resolved_options.get("prepared_root"):
            if fetch_arguments is not None:
                raise PlanError("An existing prepared_root consumes no fetch stage; "
                                "remove the plan's fetch block to use that bundle")
            from woof.fetch import pinned_host

            # Breakage it prevents: a pinned host nothing asks, so the run
            # would not use the host its plan names.  ``auto`` pins none,
            # so a plan spelling out the default passes, as ``woof go
            # --prepared-root DIR --transport auto`` does.
            if pinned_host(resolved_options.get("transport")) is not None:
                raise PlanError("An existing prepared_root consumes no fetch stage, "
                                "so run_options.transport would pin a host nothing "
                                "asks; remove it to use that bundle")
            # Breakage it prevents: the bundle's cycle is the one it was
            # prepared from, so a cycle here would be read by nothing and
            # the run would start at another time than its plan names.
            if resolved_options.get("cycle") is not None:
                raise PlanError("An existing prepared_root consumes no fetch stage, "
                                "so run_options.cycle would name a cycle nothing "
                                "fetches; remove it to use that bundle")
            _validate_prepared_output(plan, require_empty=True)
    return plan


def _grib2_member_contract(plan: RunPlan, payload: bytes) -> None:
    """Refuse member-colliding archives before source or device work."""
    if not plan.run_options.get("grib2"):
        return
    from woof.ensemble.door import request_for_payload
    from woof.grib2_live import validate_member_contract
    try:
        request = request_for_payload(payload, override=plan.run_options.get("ensemble"))
        validate_member_contract(True, request)
    except ValueError as error:
        raise PlanError(str(error)) from None


def _section_refusal(options: Mapping[str, Any]) -> str | None:
    """Refuse an ``xsec:`` product the plan names no line for, or ``None``.

    WHAT BREAKAGE THIS PREVENTS (gate law): every render this plan's run
    draws drops a section term it has no line for, so the forecast ran in
    full and the term drew nothing, and a request of only section terms
    ended on "nothing left to draw" after the whole forecast.  The plan
    is the one place the run can still be told before anything is
    fetched.
    """

    products = options.get("render_products")
    text = "" if products is None else str(products).strip()
    if not text or text.lower() == "none" or options.get("render_section"):
        return None
    from woof.rustwx import split_section_spec

    sections = split_section_spec(text)[1]
    if not sections:
        return None
    return ("run plan 'run_options.render_products' names "
            + ", ".join(repr(term) for term in sections)
            + ", a vertical section, and the plan names no line to cut it "
              "along, so it would draw nothing after the whole forecast. "
              "Next: set run_options.render_section to lat,lon,lat,lon or a "
              "JSON file holding {start, end} or a {points, extend_km} "
              "polyline, or drop the term.")


def _build_intent(intent: object, *, route: str,
                  base: Path | None = None) -> dict[str, Any]:
    """Validate a ``config.intent`` block into wizard-flag arguments.

    Shape only.  Every VALUE is left to the wizard's own parser -- a
    latitude, a ladder name, a physics profile id and a cycle are all
    things ``woof domain`` already refuses precisely, and a second
    opinion here would be a second thing to keep in step.
    """

    if not isinstance(intent, dict):
        raise PlanError("run plan 'config.intent' must be an object")
    _reject_unknown(intent, _INTENT_FLAGS, "run plan 'config.intent'")
    for required in ("cycle",):
        if required not in intent:
            raise PlanError(
                f"run plan 'config.intent' must carry {required!r}; "
                "`woof domain` requires it and cannot guess one")
    if not ({"point", "polygon"} & set(intent)):
        raise PlanError(
            "run plan 'config.intent' must carry 'point' (LAT,LON) or "
            "'polygon' (a GeoJSON path): the wizard sizes a domain "
            "around a place, and there is no default place")

    if not ROUTES[route].needs_case_data:
        stranded = sorted(
            key for key in intent
            if _INTENT_DELIVERY.get(key) == "case_data")
        if stranded:
            raise PlanError(layered(
                f"run plan 'config.intent' sets {stranded}, which the "
                f"{route!r} route has nowhere to put.",
                "`woof domain` writes those into [case_data], and it "
                "writes no [case_data] for this route's sources -- so "
                "they would be accepted here and then silently dropped, "
                "which is how a run uses a default nobody chose.  The "
                "prepared chain fetches its own forcing and takes its "
                "Vtable from the bridge; neither is yours to set."))

    source = intent.get("source", "era5")
    _refuse_undrivable_intent_source(str(source), route=route)
    from woof.fetch import validate_fetch_hints
    intent = _keyless_era5_default(intent, str(source))
    selection = {key: intent[key] for key in
                 ("member", "cadence", "era5_product", "era5_provider") if key in intent}
    if selection:
        if selection.get("era5_product") == "ensemble_members":
            selection["retrieve"] = True
        try:
            validate_fetch_hints(dict(selection, source=source), source="config.intent")
        except ValueError as error:
            raise PlanError(str(error)) from error
    intent = dict(intent)
    if isinstance(intent.get("physics_profile"), str):
        from woof.physics_registry import canonical_template_id

        # The wizard reads an old profile ID as its current ID; the chain's
        # own assertion and the manifest read this value directly.
        intent["physics_profile"] = canonical_template_id(intent["physics_profile"])
    if base is not None:
        # A relative file in a plan means a file beside the plan, as the
        # rest of the plan's paths do; left relative it would be read
        # from wherever the plan happened to be launched.
        for key in _INTENT_PATH_KEYS & set(intent):
            value = intent[key]
            if isinstance(value, (list, tuple)):
                intent[key] = [_beside_plan(item, base) for item in value]
            else:
                intent[key] = _beside_plan(value, base)
    return intent


#: The intent keys whose values are files or folders on disk.  Every
#: consumer of the intent reads these values -- the wizard, and after it
#: the preparation stages that take ``data_dir`` and ``geog_root`` from
#: the intent directly -- so they are made absolute once, when the plan
#: is built, rather than at each reader.
_INTENT_PATH_KEYS = frozenset(
    {"polygon", "data_dir", "forcing", "vtable", "geog_root"})


def _beside_plan(value: object, base: Path) -> object:
    """A relative path string made absolute against the plan's folder.

    Normalised without asking the file system, because ``forcing`` may
    be a glob pattern the wizard expands later, and a pattern is not a
    path that can be resolved.
    """

    if not isinstance(value, str) or not value.strip():
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return os.path.abspath(Path(base) / path)


def _keyless_era5_default(intent: Mapping[str, Any], source: str) -> dict[str, Any]:
    """An ERA5 intent that names no provider reads the analysis-ready store.

    The wizard's own default provider is the Copernicus CDS, a keyed job
    API: an intent (the point-and-date door every front end drives) that
    left the provider unsaid failed at acquisition on any computer
    without a CDS key, after the plan had been accepted.  The ARCO store
    holds the same reanalysis from 1940 with no key.  A named provider,
    and the ensemble members only the CDS serves, are kept as written.
    """

    from woof.source_adapters import get_source_adapter

    try:
        canonical = get_source_adapter(source).source_id
    except ValueError:
        return dict(intent)
    if (canonical != "era5" or "era5_provider" in intent
            or intent.get("era5_product") == "ensemble_members"):
        return dict(intent)
    return {**intent, "era5_provider": "arco"}


def _refuse_undrivable_intent_source(source: str, *, route: str) -> None:
    """Refuse an intent source this route cannot drive, naming the fact.

    Three shapes, each with its own true sentence:

    * a registered source no route can drive from an intent -- the
      refusal is the DERIVED one from :func:`intent_drivability`, which
      names the missing registry fact (no acquisition route, a member
      set, no forcing cadence...), never "unknown source";
    * a registered source another route drives -- the refusal names
      that route, so the remedy is one edited field;
    * a name the registry does not hold at all -- the wizard's own
      vocabulary, relayed by naming the door that lists what exists.
    """

    from woof.source_adapters import get_source_adapter

    try:
        canonical = get_source_adapter(source).source_id
    except ValueError as error:
        raise PlanError(layered(
            f"run plan 'config.intent' names source {source!r}, which "
            "is not in the source registry.",
            f"{error}  `woof run-plan --sources` lists every "
            "registered source and whether an intent can drive it.")
        ) from error
    verdict = intent_drivability().get(canonical)
    if verdict is None:                                  # pragma: no cover
        raise PlanError(
            f"run plan 'config.intent' names source {source!r}, which "
            "the registry holds but the drivability derivation did not "
            "answer for; that is a defect in this front door")
    if route in verdict["routes"]:
        return
    if verdict["routes"]:
        other = ", ".join(f'route = "{name}"'
                          for name in verdict["routes"])
        raise PlanError(layered(
            f"run plan 'config.intent' names source {source!r}, which "
            f"the {route!r} route cannot drive.",
            f"An intent for {canonical!r} runs on {other}: "
            + ("its emission carries a [case_data] table the "
               "config-driven route consumes directly."
               if "experiment" in verdict["routes"] else
               "its emission carries no [case_data] table, so it runs "
               "on the prepared chain, not the config-driven one.")))
    raise PlanError(layered(
        f"run plan 'config.intent' names source {source!r}, which no "
        "run-plan route can drive from an intent.",
        str(verdict["refusal"])))


@functools.lru_cache(maxsize=1)
def _repeated_wizard_flags() -> frozenset[str]:
    """The ``woof domain`` flags that are repeated once per value.

    Read off the wizard's own parser (``action="append"``), so a flag
    that becomes repeatable there is spelled correctly here with no edit.
    """

    from woof.domain_wizard import register_cli as register_wizard

    parser = register_wizard(argparse.ArgumentParser().add_subparsers())
    return frozenset(
        option for action in parser._actions  # noqa: SLF001 - argparse
        if isinstance(action, argparse._AppendAction)  # noqa: SLF001
        for option in action.option_strings)


def intent_arguments(intent: Mapping[str, Any], *, out: Path
                     ) -> list[str]:
    """The ``woof domain`` argv one intent block spells.

    Exposed because a front end that wants to show the reader the
    command behind their form has to be able to ask for it, and
    reconstructing it from the flag table would be a second copy.
    """

    arguments: list[str] = []
    repeated = _repeated_wizard_flags()
    for key in sorted(intent):
        flag = _INTENT_FLAGS[key]
        value = intent[key]
        if isinstance(value, bool):
            raise PlanError(
                f"run plan 'config.intent.{key}' must be a value, not a "
                "flag; every wizard option this front door exposes "
                "takes one")
        if isinstance(value, (list, tuple)):
            if flag in repeated:
                # An append-style flag takes ONE value per occurrence:
                # '--ack A B' is refused by argparse, '--ack A --ack B'
                # is the list.
                for item in value:
                    arguments.extend((flag, str(item)))
                continue
            arguments.append(flag)
            arguments.extend(str(item) for item in value)
            continue
        if isinstance(value, Mapping):
            # An object (physics_choices) travels as the JSON the flag
            # takes, not as Python's spelling of a dict.
            arguments.extend((flag, json.dumps(value, sort_keys=True, separators=(",", ":"))))
            continue
        arguments.extend((flag, str(value)))
    arguments.extend(("--out", str(out)))
    return arguments


#: Every ``run_options`` key this module understands, with the value a
#: plan that omits it gets.  A route declares which subset it supports;
#: an option the route does not support is refused, never accepted and
#: dropped.
_RUN_OPTION_DEFAULTS: dict[str, Any] = {
    "ensemble": None,
    "device": None,
    "dry_run": False,
    "restart": None,
    "prepared_root": None,
    "wps_namelist": None,
    "health_debug": False,
    "verify_visuals": True,
    "grib2": False,
    "data_dir": None,
    "initial_inputs": None,
    "geog_root": None,
    "physics_profile": None,
    "supplement": [],
    # `woof render --products`' own spec: a comma-separated product
    # list, or `all`, or `none` to skip rendering entirely.  Absent
    # leaves the chain's default set exactly as it was.  NOT an intent
    # key: intent is the wizard's flag list one-for-one, and the wizard
    # writes configs rather than pictures -- there is no --render-products
    # flag for it to mirror.
    "render_products": None,
    "publish_root": None,
    "publish_cmd": None,
    # `woof render --section`'s own value: the line every `xsec:` term
    # of `render_products` is cut along, carried to every render the run
    # draws.  A file is made absolute against the plan's directory.
    # Absent draws no section, and an `xsec:` term with no line is
    # refused when the plan is built (:func:`_section_refusal`).
    "render_section": None,
    # Complete checkpoint sets the run keeps in its directory; 0 keeps
    # every one.  One is enough to resume, and keeping every hourly set
    # filled a 58 GB disk nine hours into a 12 hour 1 km run.
    "keep_checkpoints": 1,
    # `woof go --transport`: the one host the fetch stage pins, winning
    # over the config's [fetch] transport.  Absent leaves the table's own
    # value, or the source's ladder when the table names none.
    "transport": None,
    # `woof go --whole-cycle` (false) and `--late-after-minutes`: the
    # fetch stage's posting rule and lateness budget, winning over the
    # config's [fetch] as_posted and late_after_minutes (DESIGN A136
    # 3.2).  Absent leaves the table's own, or the default: as posted,
    # with the source row's budget.
    "as_posted": None,
    "late_after_minutes": None,
    # `woof go --cycle`: the cycle the run starts from, winning over the
    # config's [fetch] cycle (or an intent's cycle) the way --transport
    # does; `latest` is resolved once, up front, under the run's posting
    # rule.  A config named by path is re-timed into the run directory
    # (:func:`plan_at_cycle`).  Absent runs the config's own cycle.  A
    # site schedule names it rather than launching `latest` (DESIGN A136
    # 3.7).
    "cycle": None,
    # A concrete assertion for supplied inputs, checked after acquisition.
    # Unlike cycle, this never retimes a config or changes a fetch request.
    "input_cycle": None,
    # `woof go --tiles` on a launch of an existing prepared bundle: the
    # [tiles] mode its single-domain forecast runs under, relayed as the
    # runner's own --tiles (the bundle's config is hash-bound and carries
    # no [tiles]).  Absent: the launch config's enabled [tiles], else the
    # one its --prepare-only launch recorded beside the bundle, else none.
    "forecast_tiles": None,
}


def _run_option(key: str, value: object, base: Path) -> Any:
    label = f"run plan 'run_options.{key}'"
    if key == "ensemble":
        if value is None:
            return None
        from woof.ensemble.request import EnsembleRequest
        try:
            return EnsembleRequest.from_mapping(value).receipt()
        except (ValueError, TypeError) as error:
            raise PlanError(f"{label}: {error}") from error
    if key == "devices":
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise PlanError(f"{label} must be a positive integer slab count")
        return value
    if key == "supplement":
        from woof.launch_supplements import bindings
        try:
            return bindings(value, base=base)
        except ValueError as error:
            raise PlanError(str(error)) from error
    if key in ("dry_run", "health_debug", "grib2", "verify_visuals"):
        if not isinstance(value, bool):
            raise PlanError(f"{label} must be true or false")
        return value
    if key == "device":
        if value is None:
            return None
        if isinstance(value, bool):
            raise PlanError(f"{label} must be a GPU index or full GPU UUID")
        if isinstance(value, int):
            if value < 0:
                raise PlanError(f"{label} GPU index must be nonnegative")
            return str(value)
        selector = _nonempty_string(value, label)
        if selector.isdigit() or selector.startswith("GPU-"):
            return selector
        raise PlanError(
            f"{label} must be a nonnegative GPU index or full GPU UUID, "
            f"got {selector!r}")
    if key == "forecast_tiles":
        if value is not None and value not in ("off", "auto", "on"):
            raise PlanError(f"{label} must be 'off', 'auto' or 'on'")
        return value
    if key == "keep_checkpoints":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PlanError(f"{label} must be a whole number of checkpoint "
                            "sets, 0 to keep every one")
        return value
    if key == "publish_root":
        return None if value is None else str(_absolute(_nonempty_string(value, label), base, label))
    if key == "publish_cmd":
        return None if value is None else _nonempty_string(value, label)
    if key == "render_products":
        return None if value is None else _nonempty_string(value, label)
    if key == "render_section":
        if value is None:
            return None
        from woof.rustwx import is_section_line, section_line_problem

        text = _nonempty_string(value, label)
        section = (text if is_section_line(text)
                   else str(_absolute(text, base, label)))
        problem = section_line_problem(section)
        if problem is not None:
            raise PlanError(f"{label}: {problem}")
        return section
    if key == "physics_profile":
        if value is None:
            return None
        from woof.physics_registry import canonical_template_id

        # An old profile ID is read as its current ID here, once, so the
        # preparation's receipt check, the prepared-run conflict check and
        # the manifest's component record all see the ID the registry keys.
        return canonical_template_id(_nonempty_string(value, label))
    if key == "as_posted":
        if value is not None and not isinstance(value, bool):
            raise PlanError(f"{label} must be true or false")
        return value
    if key in ("cycle", "input_cycle"):
        if value is None:
            return None
        text = _nonempty_string(value, label)
        if key == "cycle" and text.lower() == "latest":
            return "latest"
        try:
            datetime.strptime(text, "%Y-%m-%dT%H")
        except ValueError:
            raise PlanError(f"{label} = {value!r} must be YYYY-MM-DDTHH (UTC) "
                            "or 'latest'") from None
        return text
    if key == "late_after_minutes":
        if value is None:
            return None
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or float(value) <= 0):
            # A zero budget fails every lead a second late.
            raise PlanError(f"{label} must be a positive number of minutes")
        return float(value)
    if key == "transport":
        if value is None:
            return None
        from woof.fetch import FETCH_TRANSPORTS
        if value not in FETCH_TRANSPORTS:
            raise PlanError(f"{label} = {value!r} is not a host `woof fetch "
                            f"--transport` takes; it takes one of {list(FETCH_TRANSPORTS)}")
        return value
    if key in ("restart", "data_dir", "geog_root", "prepared_root", "wps_namelist", "initial_inputs"):
        return None if value is None else str(_absolute(value, base, label))
    raise PlanError(f"{label} is not a run option this build understands")


# ---------------------------------------------------------------------------
# Moving nests: which chain can feed one, and at what price
# ---------------------------------------------------------------------------

# Moving-statics capability belongs to source_cli's preparation dispatcher.
# The runner's artifact preflight remains responsible for actual input validity.
def _chain_key(route: str, config_source: str | None) -> str:
    """Which chain this plan dispatches to.

    The prepared route is three chains wearing one name:
    :func:`_hrrr_chain` drives the native HRRR tools,
    :func:`_staged_chain` drives the fetch->prep->sim staged route for
    the packaged mapped-composition sources, and everything else goes
    to ``woof go``'s rw-wps chain.  :func:`_execute_prepared_route`
    branches on this function rather than on its own copy of the test,
    so the chain that RUNS and the chain the refusal above was decided
    for are the same chain by construction.

    The fork is decided by the SOURCE'S REGISTRY ROW -- the same
    :func:`intent_drivability` derivation the resolve-time refusals
    read -- never by a list of model names here.  A source the
    derivation cannot place (unregistered, or undrivable) falls to the
    ``go`` chain, whose own refusal names its orchestrated set; that is
    the behaviour every such config already had.
    """
    if route != "prepared":
        return route
    from woof.source_drivability import candidate_route_chain as shared_chain

    return shared_chain(config_source)


def prepared_chain_for_source(source: str, *, source_root=None) -> str:
    """The native launch chain declared by this source's capabilities.

    Unlike _chain_key's legacy fallback, a human launch must not send an
    undrivable source into a different decoder and discover it after fetch.
    """
    canonical = _canonical_source_id(source)
    verdict = drivability_for(source) or None
    if verdict is not None and "prepared" in verdict["routes"]:
        if verdict.get("requires_source_root") and source_root is None:
            raise PlanError(
                f"{canonical} requires local input bytes; no acquisition route "
                "can obtain them. Supply data_dir in the plan, --data-dir DIR "
                "with woof go, or [fetch].source_root.")
        return _chain_key("prepared", canonical)
    reason = ("no source is registered with that name" if verdict is None
              else verdict["refusal"] or
              "its input route uses a [case_data] configuration")
    raise PlanError(layered(
        f"{source} has no automatic native launch route. Next: woof sources",
        reason))


def candidate_route_chain(source: object) -> str:
    """The chain a candidate emitted for this source will dispatch to.

    :func:`prepared_chain_for_source` answers the same question for a
    LAUNCH, and refuses a source no chain can drive.  A door deciding
    which files to write beside a candidate must not refuse on that, so
    it asks the shared answer instead -- the same one :func:`_chain_key`
    above returns for the prepared route, so the companions a candidate
    is given and the companions its run reads cannot be decided
    differently.  Kept here under its own name because this module is
    where a reader of the dispatcher looks for it.
    """
    from woof.source_drivability import candidate_route_chain as shared_chain

    return shared_chain(source)


def drivability_for(source: object) -> dict[str, Any]:
    """The drivability verdict for a config's own spelling of a source.

    One derivation, shared with the doors that publish a configuration
    without importing the dispatcher: see
    :func:`woof.source_drivability.drivability_for` for why every door
    asks through it rather than looking a registry id up itself.
    """

    from woof.source_drivability import drivability_for as shared_verdict

    return shared_verdict(source)


def _local_input_hints(hints: Mapping[str, Any]) -> tuple[str, str, Any]:
    """``source``, ``cycle`` and ``hours`` of a local-input ``[fetch]``.

    A local-input source is prepared from bytes already on disk, and
    these three are what say WHICH bytes: the source names the
    preparation contract, the cycle names the analysis time the files
    carry, and hours names the window they have to cover.  Indexing them
    raised ``KeyError`` from inside the review, which reaches a reader as
    a traceback naming a dictionary key rather than as a refused plan
    naming the missing line.
    """

    missing = [key for key in ("source", "cycle", "hours")
               if hints.get(key) is None]
    if missing:
        raise PlanError(
            "[fetch] is missing " + ", ".join(missing) + ": this source is "
            "prepared from local input bytes, and source, cycle and hours "
            "are what name which bytes.\n"
            "  what to do: add " + ", ".join(f"{key} = ..." for key in missing)
            + " to the config's [fetch] table.")
    return str(hints["source"]), str(hints["cycle"]), hints["hours"]


def _canonical_source_id(source: str) -> str:
    """The registry id for a name or alias, or the name unchanged."""

    from woof.source_adapters import get_source_adapter

    try:
        return get_source_adapter(source).source_id
    except ValueError:
        return source


def source_follow_statics(source: str) -> dict[str, Any]:
    """Whether a moving nest can be fed when SOURCE drives this front door.

    The two derivations the dispatch itself uses, read in the order the
    dispatch reads them: :func:`intent_drivability` places the row on a
    chain -- the same placement :func:`_chain_key` makes when it turns a
    plan into a dispatch -- and :func:`woof.source_cli.preparation_statics` says how
    that chain delivers the child-resolution statics a moving nest
    travels over.

    An AUTHORING door asks this before it emits a following nest, and
    :func:`follow_statics_decision` asks it again at resolve time, so
    the two cannot reach different verdicts about the same
    configuration: there is one table and one derivation between them.
    Without this seam an author emitted a following nest for a source
    whose chain seals no corridor, and the reader met the refusal at the
    run door instead of on the screen where the source was chosen.

    TWO ANSWERS WEAR ``delivery = None`` and they are not the same
    answer.  A row placed on a chain whose preparation seals no corridor
    is launchable, and only its nest is refused.  A row placed on NO
    chain is not launchable at all: :func:`prepared_chain_for_source` --
    the function ``woof go`` itself calls before it considers anything
    in the configuration -- refuses the source, and a sentence about
    statics would send the reader to fix the wrong thing.  That refusal
    is returned verbatim as ``launch_refusal`` (``None`` for every row
    that does reach a chain) so the authoring door can quote the run
    door rather than paraphrase it.

    ``delivery`` is the table's own word (``None`` on both of the above);
    ``reason`` is why, in the vocabulary of whichever derivation
    answered.
    """

    canonical = _canonical_source_id(source)
    verdict = intent_drivability().get(canonical)
    chain = (verdict or {}).get("chain")
    launch_refusal = None
    if chain is None:
        # Asked of the run door's own gate, not reconstructed here, so
        # the sentence the reader is shown at authoring is the sentence
        # the launch will raise.
        try:
            prepared_chain_for_source(canonical)
        except PlanError as error:
            launch_refusal = _split_message(str(error))[0]
    from woof.source_cli import preparation_statics

    statics = preparation_statics(chain, source=canonical) if chain else None
    delivery = statics["delivery"] if statics else None
    if delivery is not None:
        reason = None
    elif chain is None:
        reason = ((verdict or {}).get("refusal")
                  or f"no registry row named {canonical!r} reaches a chain "
                     "this front door dispatches to")
    else:
        reason = statics["reason"]
    return {"source": canonical, "chain": chain, "delivery": delivery,
            "integrates_moving_nest": delivery is not None, "reason": reason,
            "launch_refusal": launch_refusal}


def follow_statics_decision(exp, *, chain: str, source: str | None = None) -> dict[str, Any] | None:
    """How a moving nest gets its statics on ``chain``, or ``None``.

    ``None`` means the config declares no moving nest, which is most of
    them: a plan that is not relocating anything is not priced, not
    annotated, and its composed commands are untouched.

    Otherwise a record stating the delivery, whether the preparation
    must seal a corridor, and -- when the chain cannot feed a moving
    nest at all -- the refusal to raise.
    """
    from woof.static.corridor import config_declares_follow_source

    if not config_declares_follow_source(exp):
        return None
    from woof.source_cli import preparation_statics

    try:
        delivery = preparation_statics(chain, source=source)["delivery"]
    except ValueError as error:
        raise PlanError(str(error)) from None
    from woof.static.corridor import declared_follow_subjects, moving_grid_ids

    # THE DECLARATIONS, not the movers.  `moving_grid_ids` answers which
    # grids move -- per-domain followers, the [relocation] mover AND its
    # containment ancestor -- and a refusal built from it told the reader
    # to delete a follow table from a domain that carries none.
    subjects = declared_follow_subjects(exp)
    followers = list(subjects["follower_grid_ids"])
    cohort = bool(followers)
    grid_ids = sorted(moving_grid_ids(exp))
    grid_id = grid_ids[0] if cohort else int(exp.relocation.grid_id)
    return {
        **({"follower_grid_ids": followers} if cohort else {}),
        "chain": chain,
        "delivery": delivery,
        "relocation_grid_id": grid_id,
        "statics_corridor": delivery in {"statics_corridor", "retained_corridor"},
        # The subject of the refusal is every table the CONFIG declares a
        # follow source in, so the withdrawal it names withdraws all of
        # them: either one left standing keeps the predicate True and
        # brings the same refusal back word for word.
        "refusal": (None if delivery is not None
                    else _follow_unsupported_refusal(
                        chain, follower_grid_ids=followers,
                        relocation_grid_id=subjects["relocation_grid_id"],
                        source=source)),
    }


#: What ``automatic_resolutions`` says about each supported delivery.
#: A caller reads this to know, BEFORE launching, whether the
#: preparation it is about to pay for will seal a corridor.
_CORRIDOR_RESOLUTION_NOTE = {
    "statics_corridor":
        "the config declares a [relocation] follow source on d{grid_id:02d}, "
        "so {stage} is composed with --statics-corridor and the "
        "bundle will carry sealed child-resolution statics over the "
        "ground each child can reach; without it "
        "woof-prepared-tree-forecast refuses this config at its "
        "preflight.  Derived from the config, not from a run option: "
        "there is no way to ask for a moving nest and separately forget "
        "the statics it moves onto.  See --estimate for the size.",
    "case_data_ingest":
        "the config declares a [relocation] follow source on d{grid_id:02d}, "
        "and this route holds the geography source for the whole run, so "
        "each footprint's statics are rebuilt at move time.  No corridor "
        "is prepared and none is needed",
}


def _follow_unsupported_refusal(chain: str, *, follower_grid_ids=(),
                                relocation_grid_id: int | None = None,
                                source: str | None = None) -> str:
    """Why this chain cannot run a moving nest, and what will.

    THE SUBJECT IS READ FROM THE DECLARATIONS, through
    :func:`~woof.static.corridor.declared_follow_subjects`, and never
    from the mover set.  A per-domain follower lives in
    ``[[domain]].follow``, and the configuration that declares one may
    carry no ``[relocation]`` table at all; naming that table sent the
    reader to edit something their file has not got, and a sentence
    built from the movers additionally named a follow table on the
    containment ancestor, which declares nothing.

    THE WAY OUT WITHDRAWS EVERY DECLARATION IT NAMES.  Both tables can
    be live at once, and
    :func:`~woof.static.corridor.config_declares_follow_source` stays
    True while either remains, so a way out that retired one of two
    cleared nothing and the refusal came back word for word.

    ``follower_grid_ids`` are the grids whose ``[[domain]].follow`` the
    config declares; ``relocation_grid_id`` is the grid a declared
    ``[relocation]`` follow source moves, ``None`` when the config
    declares none.
    """
    from woof.core.nest_lifecycle import FOLLOWER_TABLE
    from woof.source_cli import preparation_statics

    try:
        detail = preparation_statics(chain, source=source)["reason"]
    except ValueError:
        detail = "its preparation seals no child-resolution statics corridor and it holds no geography source at run time"
    ids = tuple(int(gid) for gid in follower_grid_ids)
    if not ids and relocation_grid_id is None:
        raise ValueError(
            "name the table this config declares its follow source in: "
            "with neither a [[domain]].follow grid nor a [relocation] "
            "grid there is no subject, and the sentence would describe a "
            "table the reader cannot be told to edit")
    where = " and ".join(f"d{gid:02d}" for gid in ids)
    one = len(ids) == 1
    delete_follow = (
        f"delete the follow table from {where}, which leaves "
        f"{'that nest' if one else 'those nests'} where this config places "
        f"{'it' if one else 'them'} and needs no corridor")
    drop_relocation = (
        "drop the follow source for a bounds-only [relocation], which "
        "does not move the nest and needs neither")
    if ids and relocation_grid_id is not None:
        # BOTH tables live.  Naming one of them would state as fact that
        # the other is not there, and withdrawing one would clear
        # nothing.
        declares = (f"a {FOLLOWER_TABLE} on {where} AND a [relocation] "
                    f"follow source on d{relocation_grid_id:02d}")
        predicate = "same pair of follow declarations"
        withdraw = (f"or withdraw BOTH declarations -- {delete_follow}, and "
                    f"{drop_relocation} -- because either one left standing "
                    "keeps this config a moving nest and brings this refusal "
                    "back word for word")
    elif ids:
        declares = f"a {FOLLOWER_TABLE} on {where}"
        predicate = "same follow declaration"
        withdraw = (f"or {delete_follow} (a bounds-only [relocation] may "
                    "stand in its place, but on its own it clears nothing "
                    f"while any {FOLLOWER_TABLE} remains)")
    else:
        declares = f"a [relocation] follow source on d{relocation_grid_id:02d}"
        predicate = "same [relocation] predicate"
        withdraw = f"or {drop_relocation}"
    return (
        f"this plan's config declares {declares}, and the {chain!r} chain "
        f"cannot supply the statics a moving nest needs: {detail} -- so it "
        "is refused here instead, before the fetch.\n"
        "  remedy: run this config on a prepared chain that seals a "
        "corridor; run-plan composes --statics-corridor for "
        f"itself, from this {predicate}; or run it on "
        "the `experiment` route with a "
        "[case_data] block, which holds the geography source and "
        f"rebuilds each footprint's statics at move time; {withdraw}.")


def follow_unsupported_refusal(chain: str, follower_grid_ids, *,
                               source: str | None = None) -> str:
    """The sentence the run door WILL raise, for an authoring door to quote.

    An authoring door that emits a following nest on a chain sealing no
    corridor has to say what the launch will say.  It asks for the
    sentence here rather than writing a second one: the paraphrase that
    used to stand in its place drifted from the refusal it paraphrased,
    and offered a way out that this door's own configuration shape
    cannot take.  ``follower_grid_ids`` are the grids whose
    ``[[domain]].follow`` the authored configuration declares.

    PER-DOMAIN FOLLOWERS ONLY, which is the shape every authoring door
    in this package emits: a configuration that also carries a live
    ``[relocation]`` follow source is a file someone has since edited,
    and the run door reads its subjects off that file through
    :func:`follow_statics_decision` rather than off this quote.
    """
    ids = tuple(int(gid) for gid in follower_grid_ids)
    if not ids:
        raise ValueError(
            "name the grid ids whose [[domain]].follow the configuration "
            "declares: an empty list would describe a [relocation] table "
            "that configuration does not carry")
    return _follow_unsupported_refusal(chain, follower_grid_ids=ids,
                                       source=source)


# ---------------------------------------------------------------------------
# [tiles]: which chain streams, and which grid of it
# ---------------------------------------------------------------------------

#: What a configured ``[tiles]`` table reaches on each chain this front
#: door dispatches to.
#:
#: ``"tree"``       the chain's forecast stage wires a builder for EVERY
#:                  grid the config asks to stream -- root through
#:                  :func:`~woof.core.streaming.standalone_domain_builder`
#:                  or :func:`~woof.core.streaming.prepared_domain_builder`,
#:                  nests through the latter's child road (per-buffer
#:                  packed nest tables, ``nest_stream.make_nest_tile_hook``)
#:                  -- and it honours the per-domain ``[tiles]`` table, so
#:                  which end of a coupling edge streams is the config's
#:                  choice, including both endpoints streamed.
#: ``"unrouted"``   the chain reads ``exp.tiles`` at NO point, so any
#:                  enabled mode is a request nothing will ever act on.
#:
#: Every chain :func:`_chain_key` can return must appear here; a test
#: fails if one does not, so a chain cannot be added without answering
#: "and what does [tiles] do on it?".
_STREAMING_DELIVERY: dict[str, str] = {
    # WAS "unrouted", and the word was earned: woof.runtime.run_experiment
    # read exp.tiles nowhere and refused it at its own front door.  It
    # wires the builders now -- builders_for_tree on the tree arm,
    # standalone_domain_builder on the single-domain arm -- so this front
    # door must stop relaying a refusal the route no longer raises.  A
    # stale "unrouted" here would refuse at resolve time, before the run
    # directory, a config the route would have run.
    "experiment": "tree",
    "prepared:go": "tree",
    "prepared:hrrr": "tree",
    # The staged chain's forecast is the same single-domain runner the
    # HRRR chain hosts, reached through the same --tiles flag; its tree
    # arm reads the user's own config, which carries [tiles] itself.
    "prepared:staged": "tree",
    "prepared:existing": "tree",
}


#: The chains whose fetch stage acquires the pinned monthly WIF aerosol
#: dataset (``woof fetch --wif``) before any forcing byte, so plan review
#: may defer that machine precondition to them rather than refuse it.
#: ``prepared:hrrr`` and ``prepared:staged`` add ``wif`` to their fetch
#: hints; ``prepared:go`` composes ``--wif`` in
#: :func:`woof.go_cli.fetch_command` from the same
#: :func:`woof.preparation_assets.wif_fetch_domains` answer.  Breakage the
#: entry for ``prepared:go`` answers (2.8.8 acceptance D-01): every GFS,
#: ECMWF and other global mp=28 config refused on a box without a
#: hand-staged dataset, because only the two native chains were listed.
#: ``prepared:existing`` prepares nothing and ``experiment`` fetches from
#: the plan's own arguments, so neither is here.
_WIF_ACQUIRING_CHAINS = frozenset({"prepared:go", "prepared:hrrr",
                                   "prepared:staged"})


def streaming_decision(exp, *, chain: str) -> dict[str, Any] | None:
    """What ``[tiles]`` will do on ``chain``, or ``None``.

    ``None`` means the config configures no ``[tiles]``, which is nearly
    all of them: an unconfigured plan is not annotated, not refused, and
    its composed commands are untouched -- the same emptiness contract
    :func:`woof.core.streaming.identity_payload_entry` keeps.

    Otherwise a record naming the chain, what it can stream, the grids
    it cannot, and -- when the combination cannot stream at all -- the
    refusal to raise.

    THE DEFECT THIS ANSWERS.  ``[tiles]`` used to reach the HRRR chain
    and be dropped without a word: the single-domain arm hands its
    forecast the authority the PREPARER published
    (:func:`tools.hrrr_single_domain_benchmark._experiment_tables`),
    which is built in code and has never carried a ``[tiles]`` table, so
    a user's block was read by run-plan, reported in ``--resolve``, and
    then silently replaced by a document that does not mention it.  A
    run configured to stream integrated resident, and the only evidence
    was the absence of a line in the log.  The plumbing is now real (see
    :func:`_hrrr_chain`); this is the other half -- the combinations
    that CANNOT stream, refused here from the config alone rather than
    discovered at the first tile buffer, minutes and two preparations
    downstream.
    """
    from woof.core import streaming

    from woof.core.devices import refuse_unrouted_devices, validate_tree_devices
    try:
        if chain not in {"prepared:go", "prepared:hrrr", "prepared:staged",
                         "prepared:existing"}:
            refuse_unrouted_devices(exp, f"run-plan {chain}")
        elif len(exp.domains) != 1:
            # The prepared chains hand a tree to the tree runner, which
            # splits it (woof sim relays the table); what it cannot split
            # is refused here by name, before any stage runs.
            validate_tree_devices(exp)
        elif getattr(getattr(exp, "relocation", None), "enabled", False):
            refuse_unrouted_devices(exp, f"run-plan {chain} moving nests")
    except ValueError as error:
        raise PlanError(str(error)) from error
    options = getattr(exp, "tiles", None) or streaming.OFF
    if not options.enabled:
        return None
    if chain not in _STREAMING_DELIVERY:
        raise PlanError(
            f"run-plan cannot say what [tiles] does on chain {chain!r}: "
            "it is not in the streaming delivery table, and guessing "
            "would either refuse a config that streams or accept one "
            "whose forecast stage will never read the table")
    delivery = _STREAMING_DELIVERY[chain]
    root = int(exp.root.grid_id)
    nests = tuple(int(dc.grid_id) for dc in exp.domains if dc.parent_id != 0)
    relocation = getattr(exp, "relocation", None)
    moving = (None if relocation is None or not getattr(
        relocation, "enabled", False) else int(relocation.grid_id))
    return {
        "chain": chain,
        "delivery": delivery,
        "mode": options.mode,
        "store": options.store,
        "streamable_grid_id": None if delivery == "unrouted" else root,
        # The nests this chain CANNOT stream.  On a "tree" delivery that
        # is none of them -- the child road is wired and which end streams
        # is the per-domain [tiles] table's answer, not this table's.
        "resident_grid_ids": [] if delivery == "tree" else list(nests),
        "relocation_grid_id": moving,
        # ``moving`` is reported above but not passed down: the moving
        # domain's own sentence now belongs to the core refusal, which
        # reads exp.relocation itself rather than being told about it.
        "refusal": _streaming_refusal(
            exp, chain, delivery, options, root=root, nests=nests),
    }


def _streaming_refusal(exp, chain: str, delivery: str, options, *, root: int,
                       nests: tuple[int, ...]) -> str | None:
    """Return a concrete unrouted or moving-child operation refusal."""
    if delivery == "unrouted":
        # The core's own sentence, raised as this front door's refusal.
        # Calling it rather than restating it is the point: a route that
        # learns to stream stops refusing here on the same day it stops
        # refusing there, without anyone remembering this file exists.
        from types import SimpleNamespace

        from woof.core import streaming

        try:
            streaming.refuse_unrouted_streaming(
                SimpleNamespace(tiles=options), f"{chain!r}",
                consults_the_seam=False)
        except streaming.StreamingRefused as refusal:
            return str(refusal)
        # It did not refuse, so the core no longer agrees that this
        # chain is unrouted -- which means the route learned to read
        # [tiles] and _STREAMING_DELIVERY was not updated with it.
        # Refused rather than accepted: this function's caller has no
        # resolution note for a delivery that does not exist, and an
        # accepted plan here would be one whose forecast stage nobody
        # has checked reads the table.
        return layered(
            f"run-plan lists the {chain!r} chain as reading [tiles] at "
            "no point, but woof.core.streaming.refuse_unrouted_streaming "
            f"accepted mode = '{options.mode}' for it.",
            "The two disagree, so one of them is stale.  If that route "
            "now wires a streamed-domain builder, give it a delivery in "
            "woof.runplan._STREAMING_DELIVERY and a note in "
            "_STREAMING_RESOLUTION_NOTE saying which grid it can stream.")
    if options.mode != "on" or not nests:
        return None
    from woof.core import streaming

    try:
        streaming.refuse_streamed_nests(exp, source="this config")
    except streaming.StreamingRefused as refusal:
        return str(refusal)
    return None


#: What ``automatic_resolutions`` says about a configured ``[tiles]``.
#: A caller reads this to know, BEFORE launching, which grid will
#: actually stream -- because the run itself cannot tell them: a grid
#: that declined to stream is ABSENT from the stepper dict, and absent
#: is exactly what an unconfigured grid looks like.
_STREAMING_RESOLUTION_NOTE = {
    "root_only":
        "[tiles] mode = '{mode}' (store = '{store}') is carried to the "
        "forecast stage of the {chain} chain, which wires "
        "streaming.builders_for_tree and streams for real.  Only "
        "d{root:02d} can stream: a nest's forcing is rebuilt from its "
        "parent rather than tabulated, so prepared_domain_builder "
        "refuses one.{nests}  Nothing here binds the restart identity -- "
        "streaming.identity_payload_entry contributes nothing on purpose, "
        "so a checkpoint written streamed resumes resident and one "
        "written resident resumes streamed",
    "tree":
        "[tiles] mode = '{mode}' (store = '{store}') is carried to the "
        "forecast stage of the {chain} chain, which wires "
        "streaming.builders_for_tree over the whole domain tree and "
        "streams for real.  ANY grid can stream here, d{root:02d} "
        "included: the root through its own tabulated boundaries, a nest "
        "through the child road (per-buffer packed nest tables refilled "
        "when the rolling generation moves).  Which end of a coupling "
        "edge streams is this config's choice -- put `tiles = {{ mode = "
        "\"off\" }}` on the [[domain]] you want resident -- and mode = "
        "'auto' answers it by pricing every domain against one budget, "
        "reserving what the domains below it need before a streamed "
        "domain picks its tile. Parent and child can both stream through "
        "their shared coupling corridors.{nests} Nothing here binds the restart "
        "identity -- streaming.identity_payload_entry contributes nothing "
        "on purpose, so a checkpoint written streamed resumes resident "
        "and one written resident resumes streamed",
}


def corridor_estimate(exp, decision: Mapping[str, Any] | None
                      ) -> dict[str, Any]:
    """What the sealed corridor will cost, priced before it is built.

    The corridor is the one preparation artifact whose size a caller
    cannot infer from the domain sizes it already has -- it covers the
    ground the nest can reach at CHILD resolution, so a modest nest that
    can travel far is hundreds of megabytes.  A front end that launches a
    moving-nest plan without showing that number is hiding the largest
    single thing the preparation will write.

    Priced through the preparation's OWN child selection
    (:func:`woof.static.corridor.validated_corridor_selection`) and the
    corridor module's own arithmetic
    (:func:`woof.static.corridor.planned_corridor_cost`), so the figure shown
    before the run and the artifact written during it come from one
    source rather than from an estimate that agrees with it today.

    Chain-agnostic by construction: nothing here reads the chain, only
    the experiment's own domain tree, so the same arithmetic prices a
    GFS corridor and an HRRR one.  The ``decision`` is consulted for
    WHETHER a corridor is sealed, never for how big it is.
    """
    if decision is None or not decision["statics_corridor"]:
        return {
            "domains": [], "host_bytes": 0, "host_gib": 0.0,
            "basis": ("this config declares no [relocation] follow "
                      "source, so no statics corridor is prepared"
                      if decision is None else
                      "a moving nest on this chain is fed by "
                      f"{decision['delivery']}, which seals no corridor"),
        }
    from woof.static.corridor import (planned_corridor_cost,
                                       validated_corridor_selection)

    # `--statics-corridor` is passed bare, which the preparation reads
    # as "every child domain" -- so every child is priced, not only the
    # one [relocation] names.  Each at the frame and reach window the
    # preparation will build it at, through the same planner.
    by_id = {int(domain.grid_id): domain for domain in exp.domains}
    domains = []
    for grid_id in validated_corridor_selection(exp, "all"):
        cost = planned_corridor_cost(exp, by_id[grid_id])
        cost["domain"] = f"d{grid_id:02d}"
        domains.append(cost)
    total = sum(entry["host_bytes"] for entry in domains)
    return {
        "domains": domains,
        "host_bytes": int(total),
        "host_gib": round(total / 1024 ** 3, 4),
        "basis": (
            "each child's corridor covers the ground its footprint can "
            "reach over the run -- the declared footprint widened by what "
            "its follow settings, itinerary and reach_speed_m_s let it "
            "and every moving ancestor travel, clipped to its frame "
            "(window_child_cells; whole_frame says when that is all of "
            "it) -- at the child's resolution, carrying the native "
            "static contract's "
            f"{domains[0]['planes_per_cell']} float64 planes, so "
            f"{domains[0]['bytes_per_cell']} bytes per corridor cell; "
            "counted from the same field inventory the build is "
            "shape-checked against.  DISK and HOST are the same figure "
            "to within container headers: the cache is an uncompressed "
            "NPZ of exactly these arrays.  No GPU residency -- a "
            "corridor is cropped on the host, so the VRAM estimate "
            "above is unchanged by it."),
    }


def _absolute(value: object, base: Path, label: str) -> Path:
    text = _nonempty_string(value, label)
    path = Path(text)
    return path if path.is_absolute() else (base / path).resolve()


def _validate_fetch_arguments(arguments: Sequence[str]) -> None:
    """Refuse a ``fetch.args`` list ``woof fetch`` would refuse anyway.

    The check is the REAL parser, built from
    :func:`woof.cli.build_parser` -- there is no second copy of the
    fetch flag table here, so a flag added there is accepted here on the
    same commit, and a typo is refused before the run claims a
    directory rather than an hour later.
    """

    _parse_fetch_arguments(arguments)


def domain_size_floor() -> dict[str, Any]:
    """The smallest domain this engine will size, from the engine.

    Studio asked for the number.  It is DERIVED here rather than
    copied: :func:`woof.domain_wizard._dims_for_scale` at the fit
    loop's own ``_MIN_SCALE`` is what the wizard actually bottoms out
    at, so this answer moves when the wizard moves.  A constant
    transcribed into a front end is a number that is right until
    somebody tunes the bracket, and then wrong silently and forever.

    Reaching for two private names is the price of that, and it is the
    right way round: a stale duplicate is worse than a tight coupling
    that breaks loudly.
    """

    from woof import domain_wizard

    nx, ny = domain_wizard._dims_for_scale(  # noqa: SLF001 - see docstring
        domain_wizard._MIN_SCALE, ())[0]  # noqa: SLF001
    return {
        "root_mass_points": {"nx": int(nx), "ny": int(ny)},
        "nest_span_mass_points": 12,
        "clearance_rows": domain_wizard._CLEARANCE_ROWS,  # noqa: SLF001
        "basis": (
            "the wizard's fit loop bisects grid scale between _MIN_SCALE "
            "and _MAX_SCALE; _MIN_SCALE is the smallest layout that "
            "still hosts the deepest ladder with full Davies/blend "
            "clearance.  A nest span below 12 mass points is refused "
            "outright.  Domain size is FITTED from the ladder and the "
            "VRAM budget -- there is no nx/ny input to set."),
    }


def load_plan(path: str | Path) -> RunPlan:
    """Read, hash and validate one plan document."""

    plan_path = Path(path)
    try:
        payload = plan_path.read_bytes()
    except OSError as error:
        raise PlanError(f"run plan {plan_path} could not be read: "
                        f"{error}") from error
    digest = hashlib.sha256(payload).hexdigest()
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlanError(f"run plan {plan_path} is invalid JSON: "
                        f"{error}") from error
    return build_plan(raw, source=str(plan_path),
                      base_dir=plan_path.resolve().parent, sha256=digest)


# ---------------------------------------------------------------------------
# The event stream
# ---------------------------------------------------------------------------


def _last_sequence(path: Path) -> int:
    """The highest sequence already in an event file, or 0.

    Tolerant by design: this runs before anything is written, against a
    file that may not exist, may be empty, or may end in a torn line
    from a killed run.  None of those is a reason to refuse to START a
    stream -- they are things :func:`read_events` reports to whoever
    reads it.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return 0
    highest = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        sequence = record.get("sequence") if isinstance(record, dict) else None
        if isinstance(sequence, int) and sequence > highest:
            highest = sequence
    return highest


def event_owner_path(path: Path) -> Path:
    """The owner file that says which process writes ``path``."""

    return Path(path).with_name(f".{Path(path).name}.owner")


def _event_record(line: bytes) -> dict | None:
    try:
        record = json.loads(line.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(record, dict) or not isinstance(
            record.get("sequence"), int):
        return None
    return record


def _repair_torn_tail(path: Path) -> Path | None:
    """Cut an event file back to its last whole record.  Owner only.

    A writer killed mid-write leaves an unterminated fragment (or, after
    an older release appended onto one, trailing lines that do not
    parse).  Those bytes are copied to a ``.torn-<time>`` file beside the
    stream, so nothing a reader might want is destroyed, and the stream
    is truncated to the end of its last valid record.  A final record
    that is whole and only lacks its newline is completed instead.
    Invalid lines BEFORE a valid record are left for :func:`read_events`
    to report: they are not a torn tail, and hiding them would hide a
    real fault.  Returns the side file when bytes were moved.
    """

    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    keep = 0
    position = 0
    while position < len(data):
        newline = data.find(b"\n", position)
        if newline < 0:
            if _event_record(data[position:]) is not None:
                with path.open("ab") as stream:
                    stream.write(b"\n")
                return None
            break
        line = data[position:newline]
        if not line.strip() or _event_record(line) is not None:
            keep = newline + 1
        position = newline + 1
    if keep >= len(data):
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    side = path.with_name(f"{path.name}.torn-{stamp}")
    side.write_bytes(data[keep:])
    with path.open("r+b") as stream:
        stream.truncate(keep)
    return side


def _jsonable(value: object) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


class EventStream:
    """Append-only JSONL, to a file and to a mirror, under one lock.

    The lock is not decoration.  ``output_committed`` is raised on the
    per-domain wrfout writer's own daemon thread
    (:class:`woof.io.wrfout.AsyncDomainWrfoutWriter`), so two threads
    genuinely do reach :meth:`emit` at once, and a JSONL line that
    interleaves with another is not recoverable by any reader.

    Every line is flushed as it is written.  A consumer tailing the file
    or reading the mirrored pipe sees each event when it happens, not
    when a buffer happens to fill.
    """

    #: Default for ``mirror``.  A sentinel rather than ``None`` because
    #: ``None`` has to be able to mean "no mirror at all" -- a caller
    #: that wanted the file only and got stdout anyway would have no
    #: spelling left for what it asked for.
    MIRROR_STDOUT = object()

    def __init__(self, path: str | Path, *, mirror=MIRROR_STDOUT):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mirror = sys.stdout if mirror is EventStream.MIRROR_STDOUT \
            else mirror
        self._lock = threading.Lock()
        self._native_parent_listener = None
        # One writer per stream, across processes.  Two launches into one
        # output folder used to both open this file for append: their
        # records interleaved with repeated sequence numbers, the second
        # manifest replaced the first, and replay refused the history.
        # The claim is taken BEFORE the file is read or repaired, so the
        # repair below never edits bytes a live writer is still adding.
        from woof import ownership

        try:
            self._claim = ownership.claim(
                event_owner_path(self.path), purpose="run event stream")
        except ownership.OwnershipError as error:
            raise PlanError(
                f"{self.path.parent} is in use by "
                f"{ownership.describe_holder(error.holder)}, which is "
                "writing this run's events.  Wait for it to finish, or "
                "choose another output folder."
                + ownership.recovery_words(error)) from None
        try:
            # Continue an existing stream rather than restarting its
            # numbering.  The file is opened for APPEND, so a second run
            # into the same directory -- a resume, or a caller that
            # reused a run_dir -- would otherwise write a record numbered
            # 1 after a record numbered 7, and read_events would refuse
            # the whole file as reordered.  A torn final line from a
            # killed writer is moved aside first, or the next record
            # would be appended onto it and the whole history would stop
            # replaying.
            self.repaired_tail = _repair_torn_tail(self.path)
            self._sequence = _last_sequence(self.path)
            self._stream = self.path.open("a", encoding="utf-8",
                                          newline="\n")
            if self.repaired_tail is not None:
                # Said in the stream itself: a replay otherwise cannot
                # tell that bytes left the history, or where they went.
                self.emit(
                    "warning", code="event_tail_recovered",
                    message=("the event history ended in a line cut short "
                             "by a writer that was stopped; those bytes "
                             "were moved beside it and the history "
                             "continues after its last whole record"),
                    preserved_path=str(self.repaired_tail))
        except BaseException:
            stream = getattr(self, "_stream", None)
            if stream is not None and not stream.closed:
                stream.close()
            self._claim.release()
            raise

    @property
    def sequence(self) -> int:
        """The sequence number of the last emitted event (0 before any)."""

        return self._sequence

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append one event; return the record exactly as written."""

        if event not in EVENT_TAGS:
            raise PlanError(
                f"{event!r} is not a run-plan event tag; known tags: "
                f"{list(EVENT_TAGS)}")
        shadowed = sorted(_ENVELOPE_KEYS & set(fields))
        if shadowed:
            raise PlanError(
                f"event {event!r} may not carry the envelope key(s) "
                f"{shadowed}")
        with self._lock:
            self._sequence += 1
            record: dict[str, Any] = {
                "schema_version": EVENT_SCHEMA,
                "sequence": self._sequence,
                "emitted_unix_ms": int(time.time() * 1000),
                "event": event,
            }
            record.update(fields)
            line = _strict_event_line(record)
            self._stream.write(line + "\n")
            self._stream.flush()
            if self._mirror is not None:
                self._mirror.write(line + "\n")
                self._mirror.flush()
            # Preserve the native stream's order even when different domain
            # writers emit concurrently. The parent has its own stream lock.
            if self._native_parent_listener is not None:
                self._native_parent_listener(record, line + "\n")
        return record

    @property
    def closed(self) -> bool:
        """Whether this stream's file is closed (its owner is done with it)."""

        return self._stream.closed

    def close(self) -> None:
        with self._lock:
            try:
                if not self._stream.closed:
                    self._stream.flush()
                    self._stream.close()
            finally:
                self._claim.release()

    def __enter__(self) -> "EventStream":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def read_events(path: str | Path, *,
                allow_partial_tail: bool = False) -> list[dict[str, Any]]:
    """Replay one event stream from the top.

    This is the whole of reattach's history half, and it is a plain
    read: the file is append-only and never rotated, so byte zero to
    EOF IS the run.

    A torn final line means the writer died between opening the write
    and flushing it.  That is refused by default rather than trimmed,
    because a reader that silently drops a partial line cannot tell the
    difference between "the run is still going" and "the run died
    here".  ``allow_partial_tail`` is the caller saying it has already
    established which.
    """

    events: list[dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            if number == len(lines) and allow_partial_tail:
                break
            raise PlanError(
                f"{path} line {number} is not valid JSON: {error}"
            ) from error
        if record.get("schema_version") != EVENT_SCHEMA:
            raise PlanError(
                f"{path} line {number} is not a {EVENT_SCHEMA} record: "
                f"schema_version is {record.get('schema_version')!r}")
        expected = len(events) + 1
        if record.get("sequence") != expected:
            raise PlanError(
                f"{path} line {number} has sequence "
                f"{record.get('sequence')!r}, expected {expected}; the "
                "stream is append-only and its sequence is dense, so a "
                "gap is a lost or reordered line, never a skipped one")
        events.append(record)
    return events


# ---------------------------------------------------------------------------
# The observer: woof's own progress protocol, rendered as events
# ---------------------------------------------------------------------------


class RunObserver:
    """The ``progress_callback`` the runtime already accepts.

    Composition, not replacement.  Every call is forwarded verbatim to
    the supervisor's :class:`~woof.supervisor.RuntimeHeartbeat`, which
    stays the ONLY writer of ``run-progress.json``: a run-plan run
    leaves the same heartbeat a ``woof run`` leaves, byte-shaped the
    same way, readable by the same recovery code.  This class adds the
    event stream on top of it and publishes no run state of its own.

    It implements the runtime's whole duck-typed surface --
    ``__call__``, ``preparing``, ``starting``, ``writing``, ``written``,
    ``finalizing``, ``complete``, ``failed`` -- plus ``output_committed``,
    the one hook this work added (:func:`woof.runtime._output_committed`).
    """

    def __init__(self, events: EventStream, *, heartbeat=None,
                 root_domain: int = 1, accepted_wall: float | None = None):
        self._events = events
        #: This run's stream, for a stage that emits records of its own
        #: rather than only opening and closing (the fetch stage relays
        #: one record per file through it).
        self.events = events
        self._heartbeat = heartbeat
        self._root_domain = int(root_domain)
        #: The monotonic reading taken when ``plan_accepted`` was
        #: emitted.  Passed in rather than read here because this object
        #: is built several steps into ``execute_plan``, and every wall
        #: time this run reports is measured from the instant the plan
        #: was accepted -- not from the instant an observer happened to
        #: exist.  Defaulted for callers that build one directly.
        self._accepted_wall = (time.perf_counter() if accepted_wall is None
                               else float(accepted_wall))
        #: Set by :meth:`arm_first_products` when a prepared chain wants
        #: its first frame rendered as it lands.  ``None`` -- the default
        #: and the whole of the ``experiment`` route -- means the
        #: finalize stage is the only render there has ever been.
        self._first_products = None
        #: Set beside it: every committed frame of every grid drawn as it
        #: lands (:mod:`woof.live_products`), on whenever the end of the
        #: run would draw pictures.
        self._live_products = None
        self._grib2_export = None
        self._grib2_base_out = None
        self._grib2_attempt = 1
        #: The any-point column store written as frames land (:mod:`woof.live_columns`), on by default.
        self._columns = None
        #: The plan both were armed with, to arm them again when the
        #: hosted forecast restarts (:meth:`restarting`).
        self._render_plan = None
        #: Time to first plot, once there is one.  Kept so the run's
        #: ``completed`` event can carry the headline number too: a
        #: reader comparing runs should not have to scan the stream for
        #: the one line that has it.
        self._first_products_seconds: float | None = None
        self._stage: str | None = None
        self._stage_phases: list[str] = []
        self._stage_started_wall = 0.0
        self._forecast_started_wall: float | None = None
        self._forecast_started_model: float | None = None
        self._committed = 0
        self._progress_events = 0
        #: The last model time this observer saw.  The chain summary's
        #: only route-independent source for how far the run got: the
        #: single-domain runner publishes it in progress.json and the
        #: tree runner does not publish it at all.
        self._last_model_seconds: float | None = None
        self._render_summary: dict[str, Any] | None = None
        # An events sink without a file (an embedding caller, a test double)
        # gets a ledger that records only when WOOF_TIME_LEDGER names one.
        events_path = getattr(events, "path", None)
        self._timeline = current_ledger(Ledger(
            None if events_path is None else Path(events_path).parent / "timeline.jsonl"))
        self._timeline_forecast = None
        self._timeline_idle = False
        self._timeline_closed = False
        register_ledger_observer(self)

    def close(self):
        """C1 closes any failed stage and records the main-flow finish."""
        if self._timeline_closed:
            return
        # The command's ledger closes every observer at exit, after the
        # door that owns this observer's event stream may already have
        # closed it (a preparation door's ``finally``).  A stage a failure
        # left open is then ended on the timeline only: writing its
        # ``stage_finished`` record to the closed file raised "I/O operation
        # on closed file", which replaced the run's own error and exit code
        # (a failed ensemble member preparation ended "woof ensemble: I/O
        # operation on closed file." at exit 2 instead of its refusal at 1).
        self.finish_stage(record=not getattr(self._events, "closed", False))
        if self._timeline_idle:
            self._timeline.emit("post.orchestration", "end")
            self._timeline_idle = False
        self._timeline.emit("finish", "mark")
        self._timeline_closed = True

    # -- stage bookkeeping --------------------------------------------

    @property
    def stage(self) -> str | None:
        """The stage currently open, or ``None`` before/after the run."""

        return self._stage

    @property
    def outputs_committed(self) -> int:
        return self._committed

    @property
    def last_model_seconds(self) -> float | None:
        return self._last_model_seconds

    def enter_stage(self, stage: str, *, phase: str | None = None) -> None:
        """Close whatever stage is open and open ``stage``."""

        if stage not in STAGES:
            raise PlanError(f"{stage!r} is not a run stage; known stages: "
                            f"{list(STAGES)}")
        if stage == self._stage:
            return
        self.finish_stage()
        if stage == "forecast":
            # A hydro fabric preparing beside this launch's fetch and
            # preparation (woof.hydro.runtime._Beside) is bound before any
            # forecast runner starts; nothing pending, nothing waits.
            hydro = sys.modules.get("woof.hydro.runtime")
            if hydro is not None:
                hydro.join_beside()
        end_admission(self._timeline)
        if self._timeline_idle:
            self._timeline.emit("post.orchestration", "end")
            self._timeline_idle = False
        self._stage = stage
        self._stage_phases = [] if phase is None else [phase]
        self._stage_started_wall = time.perf_counter()
        payload: dict[str, Any] = {"stage": stage}
        if phase is not None:
            payload["phase"] = phase
        self._events.emit("stage_started", **payload)
        self._timeline.emit(RUN_STAGES[stage], "begin", attrs={"phase": phase, "role": "envelope"})
        if stage == "forecast":
            # Native prepared routes use <run>/chain/run. Hosted go uses
            # its plan's run directory, attached by the main flow below.
            forecast_dir = getattr(self, "_timeline_forecast_dir",
                                   self._events.path.parent / "chain" / "run")
            self._timeline_forecast = ForecastLedger(self._timeline, forecast_dir)
            self._timeline_forecast.start(watch=False)

    def finish_stage(self, *, record: bool = True, **fields: Any) -> None:
        """Close the open stage, if any.

        ``record=False`` ends it on the timeline without a ``stage_finished``
        event, for a stream its owner has already closed (:meth:`close`).
        """

        if self._stage is None:
            return
        stage = self._stage
        self._stage = None
        if self._timeline_forecast is not None:
            self._timeline_forecast.finish()
            self._timeline_forecast = None
        self._timeline.emit(RUN_STAGES[stage], "end", attrs={"phase": list(self._stage_phases), "role": "envelope"})
        self._timeline.emit("post.orchestration", "begin")
        self._timeline_idle = True
        if not record:
            return
        if stage == "finalize" and self._render_summary is not None:
            fields.setdefault("render_summary", self._render_summary)
        self._events.emit(
            "stage_finished", stage=stage,
            wall_seconds=round(
                time.perf_counter() - self._stage_started_wall, 6),
            phases=list(self._stage_phases), **fields)

    def warn(self, code: str, message: str, **fields: Any) -> None:
        self._events.emit("warning", code=code, message=message, **fields)

    # -- the runtime's progress protocol ------------------------------

    def preparing(self, phase: str) -> None:
        if self._heartbeat is not None:
            self._heartbeat.preparing(phase)
        stage = _PHASE_STAGES.get(phase)
        if stage is None:
            self.warn(
                "unmapped_pipeline_phase",
                f"the pipeline reported preparation phase {phase!r}, "
                "which this front door has no stage for; it is being "
                "attributed to the open stage rather than dropped",
                phase=phase, stage=self._stage)
            self._stage_phases.append(phase)
            return
        if stage == self._stage:
            self._stage_phases.append(phase)
            return
        self.enter_stage(stage, phase=phase)

    def starting(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.starting()

    # The runtime announces a write between two steps (``writing`` then
    # ``written``) and each beat after the last step (``finalizing``) only
    # when its progress object has those hooks
    # (:func:`woof.supervisor.writing_progress`,
    # :func:`woof.runtime._finalizing_progress`).  Without them here the
    # heartbeat never heard of either: a run-plan or hosted-go forecast
    # published "integrating" through an 81 s last frame write and the
    # read-back of its checkpoint, so the write was timed as a model step.

    def writing(self, phase: str, *, work_bytes: int | None = None) -> None:
        hook = getattr(self._heartbeat, "writing", None)
        if hook is not None:
            hook(phase, work_bytes=work_bytes)

    def written(self) -> None:
        hook = getattr(self._heartbeat, "written", None)
        if hook is not None:
            hook()

    # A forecast at a seam waiting for a boundary interval
    # (:class:`woof.ingest.boundary_stream.SeamWaits`): the heartbeat
    # says ``waiting:source`` or ``waiting:preparation`` with its wait
    # record, so a supervisor times the wait by its own bound.

    def waiting(self, on: str, **record: Any) -> None:
        hook = getattr(self._heartbeat, "waiting", None)
        if hook is not None:
            hook(on, **record)

    def waited(self) -> None:
        hook = getattr(self._heartbeat, "waited", None)
        if hook is not None:
            hook()

    def restarting(self, reason: str) -> None:
        """The hosted forecast starts again in this process.

        The heartbeat publishes its restart record
        (:func:`woof.supervisor.restart_attempt`).  The renders this
        observer armed read the last attempt's frames, which the runner
        moves aside next, so they are ended and waited for first
        (:func:`woof.first_products.halt_renders_and_wait`, in
        :meth:`stop_live_products`' order: the every-frame render closed
        first and joined last) and armed again on the same plan for the
        new attempt's frames.  The speed baseline and the frame count
        start again with the new attempt.
        """

        live, first = self._live_products, self._first_products
        if live is not None or first is not None:
            from woof.first_products import halt_renders_and_wait

            if live is not None:
                live.halt(timeout=0)
            # A bounded halt may return with a reader still alive.  Keep the
            # old render handles and progress intact in that case: the runner
            # is about to rewind history, so a new attempt cannot start until
            # every reader of the old frames has ended.  Try both readers even
            # when the first one fails to stop.
            first_stopped = (first is None or halt_renders_and_wait(first))
            live_stopped = (live is None or halt_renders_and_wait(live))
            if not first_stopped or not live_stopped:
                raise RuntimeError(
                    "forecast restart refused: a render is still reading "
                    "the previous attempt's history; refusing output rewind")

        # The any-point column store, the same way.  THE BREAKAGE: left
        # armed, its writer remembered every frame path it had written, so
        # the new attempt's frames at the same paths were skipped and the
        # store kept the failed attempt's columns for every rewound hour;
        # and its writer could still be reading a frame the runner moves
        # aside next (on Windows that move fails on the open file).  The
        # queued frames are dropped, the frame in hand is waited for (its
        # write is killed at its own timeout), and a fresh writer is armed
        # on the same folder after the restart record.
        columns = self._columns
        if columns is not None:
            summary = self.stop_columns(halt=True, timeout=columns.restart_wait_seconds)
            if summary is not None and not summary.get("stopped", True):
                raise RuntimeError(
                    "forecast restart refused: the column store is still "
                    "reading the previous attempt's history; refusing output rewind")

        hook = getattr(self._heartbeat, "restarting", None)
        if hook is not None:
            hook(reason)
        if columns is not None:
            self.arm_columns(columns.out, frames=columns.frames)
        if self._grib2_export is not None:
            self.stop_grib2(halt=True)
            self._grib2_export = None
            self._grib2_attempt += 1
            base = self._grib2_base_out
            self.arm_grib2(base.with_name(f"{base.name}-attempt-{self._grib2_attempt:02d}"))
        if live is not None or first is not None:
            self._first_products = None
            self._live_products = None
            if self._render_plan is not None:
                self.arm_first_products(self._render_plan)
        self._forecast_started_wall = None
        self._forecast_started_model = None
        self._last_model_seconds = None
        self._committed = 0
        self.warn("forecast_restarted",
                  f"the forecast starts again: {reason}", reason=str(reason))

    #: Set by :meth:`source_behind`: the run ends with ``source_behind``
    #: then ``failed`` (``SourceBehind``, exit 75).
    source_behind_record: dict[str, Any] | None = None
    secondary_source_behind_record: dict[str, Any] | None = None

    def source_behind(self, details: Mapping[str, Any]) -> None:
        """The late lead that ended the forecast; said with ``failed``."""

        self.source_behind_record = dict(details)

    def finalizing(self, phase: str, *, work_bytes: int | None = None) -> None:
        if self._timeline_forecast is not None:
            self._timeline_forecast.read()
        hook = getattr(self._heartbeat, "finalizing", None)
        if hook is not None:
            hook(phase, work_bytes=work_bytes)

    def __call__(self, *, model_elapsed_seconds: float, outer_step: int,
                 last_durable_wrfout=None, last_checkpoint=None,
                 phase: str = "synchronized-step",
                 step_wall_seconds: float = 0.0, **extra: Any) -> None:
        if self._timeline_forecast is not None:
            self._timeline_forecast.poll()
        if self._heartbeat is not None:
            self._heartbeat(
                model_elapsed_seconds=model_elapsed_seconds,
                outer_step=outer_step,
                last_durable_wrfout=last_durable_wrfout,
                last_checkpoint=last_checkpoint, phase=phase,
                step_wall_seconds=step_wall_seconds, **extra)
        model_seconds = float(model_elapsed_seconds)
        self._last_model_seconds = model_seconds
        if self._stage != "forecast":
            self.enter_stage("forecast", phase=phase)
        if self._forecast_started_wall is None:
            # Armed on the FIRST progress call, not on entering the
            # stage.  Every prepared chain opens `forecast` itself
            # before handing the runner over, so keying this off the
            # stage transition meant it never armed there: a live
            # nested run published 181 progress events with speed_x
            # null and wall_seconds 0.0 on every one of them.
            self._forecast_started_wall = time.perf_counter()
            self._forecast_started_model = model_seconds
        wall_seconds = (
            0.0 if self._forecast_started_wall is None
            else time.perf_counter() - self._forecast_started_wall)
        advanced = model_seconds - (self._forecast_started_model or 0.0)
        payload: dict[str, Any] = {
            "domain": self._root_domain,
            "outer_step": int(outer_step),
            "model_seconds": model_seconds,
            "wall_seconds": round(wall_seconds, 6),
            "phase": phase,
        }
        # speed_x is a rate, and a rate over no elapsed wall is not a
        # large number, it is an undefined one.  Reported as null rather
        # than as an infinity a consumer would have to special-case.
        payload["speed_x"] = (round(advanced / wall_seconds, 4)
                              if wall_seconds > 0.0 and advanced > 0.0
                              else None)
        step_ms = float(step_wall_seconds) * 1000.0
        payload["step_ms"] = round(step_ms, 3) if step_ms > 0.0 else None
        if last_checkpoint is not None:
            payload["last_checkpoint"] = str(last_checkpoint)
        # `domain` above is the ROOT clock, and on a tree that is only
        # part of the answer: the nests advance on their own clocks.
        # Present only when there is more than one, so the single-domain
        # route's events are unchanged and a consumer can treat the key's
        # absence as "the root IS the tree".
        clocks = extra.get("domain_clocks")
        if isinstance(clocks, dict) and len(clocks) > 1:
            # Each grid's summed host wall of its own steps, when the
            # executor measured it: the one number that says which grid
            # of a nested run sets its pace.
            walls = extra.get("domain_step_wall")
            walls = walls if isinstance(walls, dict) else {}
            payload["domains"] = [
                {"domain": int(grid_id), "model_seconds": float(seconds),
                 **({"step_wall_seconds": round(float(walls[grid_id]), 3)}
                    if grid_id in walls else {})}
                for grid_id, seconds in sorted(clocks.items())]
        self._progress_events += 1
        self._events.emit("model_progress", **payload)

    def stage_progress(self, *, phase: str, elapsed_seconds: float,
                       model_seconds: float, status=None) -> None:
        """Model progress observed from a stage's published progress file.

        The coarse arm of ``model_progress``, for a route whose stage
        runs as a subprocess: the numbers are the stage's own, read from
        the artifact it republishes, and the cadence is that stage's
        rather than every step.  It carries ``step_ms: null`` and says
        where it came from in ``source``, so a consumer can tell a
        polled sample from a per-step one instead of assuming.
        """

        speed = (round(model_seconds / elapsed_seconds, 4)
                 if elapsed_seconds > 0.0 and model_seconds > 0.0 else None)
        self._events.emit(
            "model_progress", domain=self._root_domain,
            model_seconds=float(model_seconds),
            wall_seconds=round(float(elapsed_seconds), 6),
            speed_x=speed, step_ms=None, phase=phase,
            status=status, source="stage_progress_file")

    # -- time to first plot -------------------------------------------

    @property
    def first_products(self):
        """The armed early render, or ``None``.

        The finalize stage reads this off whatever observer it was given
        -- directly or through :class:`_GoObserver` -- to collect the
        render before deciding what is left to draw.
        """

        return self._first_products

    @property
    def first_products_seconds(self) -> float | None:
        """Time to first plot, or ``None`` if no early render published."""

        return self._first_products_seconds

    @property
    def live_products(self):
        """The render of every frame as it lands, or ``None``."""

        return self._live_products

    def arm_grib2(self, out) -> None:
        """Start native output independently of the selected map products."""
        if self._grib2_export is not None:
            return
        from woof.grib2_live import LiveGrib2Export
        out = Path(out)
        if self._grib2_base_out is None:
            self._grib2_base_out = out
        self._grib2_export = LiveGrib2Export(
            out, report=self._grib2_progress, warn=self.warn)
        self._grib2_progress({"event": "armed", "out": str(out), "attempt": self._grib2_attempt})

    def arm_columns(self, out, *, experiment=None, frames=None) -> None:
        """Write the run's any-point column store (rw-store v2) for every output hour as frames land.

        ``experiment`` (or ``frames`` itself) names which frames carry the 3-D state
        (:class:`woof.live_columns.StateFrames`); the others are not attempted."""
        if self._columns is not None:
            return
        from woof.live_columns import LiveColumns, StateFrames
        try:
            if frames is None and experiment is not None:
                frames = StateFrames.for_experiment(experiment)
            self._columns = LiveColumns(Path(out), report=self._columns_progress, warn=self.warn,
                                        frames=frames)
        except Exception as error:  # noqa: BLE001 - a store that cannot start never stops the run
            self.warn("columns_not_armed", f"no column store for this run: {type(error).__name__}: {error}")

    def _columns_progress(self, event) -> None:
        self._events.emit("columns_progress", progress=dict(event))

    def stop_columns(self, *, halt: bool = False, timeout: float | None = None) -> dict | None:
        """End the column store: ``halt`` drops what is queued (a stopped or restarting run), otherwise the queue
        and each grid's last frame are written. Both waits are bounded (:meth:`woof.live_columns.LiveColumns.stop`)."""
        columns, self._columns = self._columns, None
        if columns is None:
            return None
        if halt:
            summary = columns.halt() if timeout is None else columns.halt(timeout)
        else:
            summary = columns.stop(timeout)
        self._events.emit("columns_written", files=summary["files"], bytes=summary["bytes"],
                          seconds=summary["seconds"], failed=len(summary["failed"]), out=summary["out"])
        return summary

    def _grib2_progress(self, event) -> None:
        self._events.emit("grib2_export_progress", progress=dict(event))

    def stop_grib2(self, *, halt: bool = False) -> dict | None:
        export = self._grib2_export
        if export is None:
            return None
        return export.halt() if halt else export.stop()

    def stop_live_products(self, *, halt: bool = False) -> dict | None:
        """Stop drawing frames as they land.

        ``halt`` is a stopped run: nothing queued is drawn and the render
        in flight is ended (:meth:`woof.live_products.LiveProducts.halt`).
        Otherwise the queue is finished first, which is what a run that
        failed on its own still gets.

        BOTH renders this observer armed, not the every-frame one alone.
        THE BREAKAGE: the early render of the analysis frame is a second
        worker with a render process of its own, and it was left running.
        A stopped run's analysis picture went on drawing and could still
        publish after the stop, and a run that failed walked out on it
        with its process alive and its scratch in the picture folder.

        On a halt the every-frame render is closed first and joined last,
        as :meth:`woof.live_products.LandingRenders.halt` and a
        downscaled child's ``halt_renders`` do: its worker may be waiting
        on the early render, which is ended in between.  Otherwise the
        early render is collected after the queue, with the same bounded
        wait the finalize stage gives it.
        """

        live = self._live_products
        first = self._first_products
        if halt:
            if live is not None:
                live.halt(timeout=0)
            if first is not None:
                first.halt()
            return None if live is None else live.halt()
        summary = None if live is None else live.stop()
        if first is not None:
            first.wait()
        return summary

    def arm_first_products(self, render_plan) -> None:
        """Render the first committed frame as it lands, not at finalize.

        ``render_plan`` is the dict the finalize stage will hand
        ``go_cli._render_stage``: the same output directory and the same
        product spec, so the early render and the late one cannot drift
        apart in what they draw or where they put it.

        Silently does nothing when this run named no products, which is
        the default.  A caller therefore does not have to ask whether
        the feature applies before arming -- the answer lives in one
        place, :func:`woof.first_products.early_render_requested`.
        """

        from woof.ensemble.runtime_context import current_session
        if current_session() is not None:
            return
        from woof.first_products import (FirstProducts,
                                          early_render_requested)
        from woof.live_products import (LiveProducts, early_render_runner,
                                          live_render_requested,
                                          shared_render_slots)

        self._render_plan = render_plan
        products = render_plan.get("render_products")
        slot, early_slot = shared_render_slots()
        if early_render_requested(products):
            self._first_products = FirstProducts(
                render_plan, report=self._first_products_ready,
                warn=self.warn, runner=early_render_runner, slot=early_slot)
        if live_render_requested(products):
            self._live_products = LiveProducts(
                render_plan, report=self._live_products_ready,
                warn=self.warn, first=self._first_products, slot=slot)

    def _live_products_ready(self, entry) -> None:
        """One frame of one grid is readable as pictures."""

        self._events.emit(
            "live_products_ready", domain=entry["domain"],
            valid_time=entry["valid_time"], frame=entry["frame"],
            pictures=entry["pictures"],
            render_seconds=entry["render_seconds"],
            queued=entry["queued"], complete=entry.get("complete", True))

    def _first_products_ready(self, receipt) -> None:
        """The early render published.  This is the TTFP number.

        Emitted from the render's own worker thread, which is why
        ``EventStream`` holds a lock: this and ``model_progress`` from
        the forecast genuinely do reach it at once.
        """

        elapsed = round(time.perf_counter() - self._accepted_wall, 6)
        self._first_products_seconds = elapsed
        if self._live_products is not None:
            self._live_products.publish_first(receipt)
        self._events.emit(
            "first_products_ready",
            domain=receipt["domain"], valid_time=receipt["valid_time"],
            frame=receipt["frame"], paths=list(receipt["paths"]),
            render_products=receipt["render_products"],
            render_seconds=receipt["render_seconds"],
            seconds_from_plan_accepted=elapsed,
            complete=receipt.get("complete", True))

    def frame_submitted(self, *, domain: int, valid_time) -> None:
        if self._live_products is not None:
            self._live_products.frame_submitted(domain=domain, valid_time=valid_time)

    def output_committed(self, *, domain: int, valid_time, path) -> None:
        """One wrfout is durable on disk.  Raised from the writer.

        On the domain-tree route this arrives on the per-domain writer
        thread, after ``WrfoutWriter.close`` has fsynced, self-validated
        and renamed the temporary onto its final name -- so the event is
        emitted for a file that exists and passes its own inventory
        check, never for one that is merely queued.

        When an early render is armed, the FIRST root-domain frame also
        dispatches it.  That frame is the analysis: the history alarm is
        true at t = 0, so it is written before a single step is
        integrated, and it is the picture a reader has been waiting the
        whole download and preparation for.
        """

        self._committed += 1
        self._events.emit(
            "output_committed", domain=int(domain),
            valid_time=(valid_time.isoformat()
                        if isinstance(valid_time, datetime)
                        else str(valid_time)),
            path=str(path))
        export = self._grib2_export
        if export is not None:
            try:
                export.frame_committed(domain=int(domain), valid_time=valid_time, path=path)
            except Exception as error:  # output callbacks must not stop stepping
                self.warn("grib2_export_failed", f"committed frame could not be queued: {error}", frame=str(path))
        columns = self._columns
        if columns is not None:
            try:
                columns.frame_committed(domain=int(domain), valid_time=valid_time, path=path)
            except Exception as error:  # noqa: BLE001 - output callbacks must not stop stepping
                self.warn("columns_not_written", f"committed frame could not be queued: {error}", frame=str(path))
        trigger = self._first_products
        claimed = False
        if trigger is not None and int(domain) == self._root_domain:
            # Guarded here as well as inside the trigger.  This method is
            # reached from `runtime._output_committed`, which -- unlike
            # the async writer's own call site -- does not wrap the
            # callback, so anything raised here would land in the model
            # loop.
            try:
                claimed = bool(trigger.frame_committed(
                    domain=domain, valid_time=valid_time, path=path))
            except Exception as error:  # noqa: BLE001 - telemetry never fails
                self.warn(
                    "first_products_not_dispatched",
                    "the early render of the first frame could not be "
                    f"started ({type(error).__name__}: {error}); the "
                    "finalize stage is unaffected")
        live = self._live_products
        if live is None:
            return
        try:
            # Every grid, every frame; the one the early render claimed
            # is left to it and not queued here.
            live.frame_committed(domain=int(domain), valid_time=valid_time,
                                 path=path, draw=not claimed)
        except Exception as error:  # noqa: BLE001 - telemetry never fails
            self.warn(
                "live_products_failed",
                f"a committed frame could not be queued for drawing "
                f"({type(error).__name__}: {error}); the end-of-run render "
                "draws it", frame=str(path))

    def complete(self, model_elapsed_seconds: float) -> None:
        self.stop_grib2()
        try:
            self.stop_columns()
        except Exception as error:  # noqa: BLE001
            self.warn("columns_not_written", str(error))
        if self._heartbeat is not None:
            self._heartbeat.complete(model_elapsed_seconds)

    def failed(self) -> None:
        try:
            self.stop_grib2()
        except Exception as error:
            self.warn("grib2_export_failed", str(error))
        try:
            self.stop_columns()
        except Exception as error:  # noqa: BLE001
            self.warn("columns_not_written", str(error))
        if self._heartbeat is not None:
            self._heartbeat.failed()


# ---------------------------------------------------------------------------
# Resolution: the snapshot and every value nobody typed
# ---------------------------------------------------------------------------


def _config_snapshot(exp, data) -> dict[str, Any]:
    """The fully resolved configuration, as JSON.

    ``dataclasses.asdict`` over the loaded config pair: the snapshot is
    the objects the model will actually run, not a re-reading of the
    TOML, so a value the loader derived appears here at its derived
    value.
    """

    from woof.experiment import experiment_config_document
    snapshot = {
        "experiment": experiment_config_document(exp),
        # None on the prepared route: those configs declare no
        # [case_data], because their inputs are bound by the prepared
        # cache rather than named in the TOML.  Reported as null rather
        # than omitted, so the key's absence is never ambiguous.
        "case_data": None if data is None else dataclasses.asdict(data),
    }
    return json.loads(json.dumps(snapshot, default=_jsonable))


def _schema_default_resolutions(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every ``ExperimentConfig`` field the document did not spell.

    Mechanical, from the dataclass itself: a field with a declared
    default that the TOML did not set was chosen by the schema, not by
    the author, and that is exactly what ``automatic_resolutions``
    exists to say out loud.

    "Did not set" spans BOTH places an author can write one of these
    fields.  Several are authored as their own top-level table --
    ``[relocation]``, ``[projection]``, ``[perturbation]`` -- and reading
    only inside ``[experiment]`` reported them as schema defaults with
    the schema's value attached, which for a moving-nest plan says
    ``relocation.enabled = false`` about a config whose nest follows a
    storm.  The run was right and the ``configuration`` block in the same
    event was right; only this list lied, and it lied to exactly the
    programmatic caller it exists for.
    """

    from woof.experiment import ExperimentConfig

    table = raw.get("experiment")
    spelled = set(table) if isinstance(table, dict) else set()
    # A top-level table whose name IS a field name is the author spelling
    # that field.  Taken from the document rather than from a second list
    # of "the tabular ones", so a table added later needs no edit here.
    spelled |= {name for name in raw if name != "experiment"}
    resolutions = []
    for field in dataclasses.fields(ExperimentConfig):
        if field.name in {"devices", "simulated_radar"}:
            # OFF contributes no new schema row to an existing plan.
            continue
        if field.name == "physics_params" and field.default is None:
            # Absent constants contribute no schema-default row. An active
            # set is carried by the resolved configuration snapshot instead.
            continue
        if field.default is dataclasses.MISSING:
            continue
        if field.name in spelled:
            continue
        resolutions.append({
            "scope": "experiment", "key": field.name,
            "value": _jsonable_scalar(field.default),
            "basis": "schema_default"})
    return resolutions


def _strict_event_line(record: dict[str, Any]) -> str:
    """One event record as a line a strict JSON reader can open.

    WHAT BREAKAGE THIS PREVENTS (gate law).  ``NaN`` is not a JSON token
    -- RFC 8259 has no spelling for it -- so ``JSON.parse``,
    ``serde_json``, ``encoding/json`` and ``jq`` all refuse a line
    carrying one, while Python's ``json`` writes it by default.  This
    stream is the machine-readable account of a run, and the events most
    likely to carry a number that went are the ones on the way out of a
    failure, which is where a reader needs the stream most.

    A REFUSAL would be the wrong outcome: an event that cannot be written
    is an event that is lost, from a writer that runs inside failure
    handling.  So the strict spelling is tried first, and a record that
    still holds a non-finite number is written with ``null`` in its place
    rather than dropped, which every reader has a value for.
    """

    try:
        return json.dumps(record, default=_jsonable, allow_nan=False)
    except ValueError:
        # Token-level, so a numpy float resolved by ``default=`` is
        # covered by the same pass as a plain one.
        relaxed = json.loads(json.dumps(record, default=_jsonable),
                             parse_constant=lambda _token: None)
        return json.dumps(relaxed, allow_nan=False)


def _jsonable_scalar(value: object) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _jsonable(value)


def _timestep_resolutions(exp) -> list[dict[str, Any]]:
    """The per-domain timestep, and how each one came to be.

    The root domain's step is the ``time_step`` (plus its optional
    rational correction) the author wrote.  Every child's is DERIVED by
    dividing the parent's down ``parent_time_step_ratio`` -- a number
    nobody typed, which changes the model's answer, and which until now
    appeared only inside a printed report.
    """

    resolutions = []
    for domain in exp.domains:
        derived = domain.parent_id != 0
        resolutions.append({
            "scope": "domain", "grid_id": domain.grid_id, "key": "dt",
            "value": float(domain.run.dt),
            "exact": str(exp.dt_exact(domain.grid_id)),
            "basis": ("derived_from_parent_time_step_ratio" if derived
                      else "declared_time_step"),
            "note": (f"parent d{domain.parent_id:02d} step divided by "
                     f"parent_time_step_ratio={domain.parent_time_step_ratio}"
                     if derived else
                     f"time_step={domain.time_step} "
                     f"+{domain.time_step_fract_num}/"
                     f"{domain.time_step_fract_den}")})
    return resolutions


def generate_intent_config(plan: RunPlan, *, destination: Path,
                           memory: dict[str, Any] | None = None
                           ) -> tuple[Path, list[dict[str, Any]]]:
    """Write this plan's intent out as a config, using ``woof domain``.

    Delegation, not composition.  The wizard exposes clean pieces --
    ``fit_ladder``, ``render_config`` -- but between them sits
    ``domain_main``'s orchestration: cycle resolution, projection
    selection by latitude, the fetch area hint and its per-source
    coverage gate, forcing cadence, the ``[case_data]`` block, profile
    resolution, and a full round trip through the real loader before
    anything is written.  Calling the pieces would mean re-deriving all
    of that here, which is the forked config logic this front door
    exists to avoid.  So the wizard's own entry point runs, through the
    wizard's own parser, and this function only supplies ``--out`` and
    reads what landed.

    The wizard is a talker -- it prints its sizing table, its resolved
    cycle, its gray-zone advisories, its next steps.  Its caller has
    already redirected stdout to stderr, so all of that reaches the
    reader and none of it reaches the machine channel.  The one number
    a front end needs from that talk, the card memory the fit priced,
    comes back as a record instead: ``memory``, when given, receives
    :func:`woof.domain_wizard.fit_memory`'s record.
    """

    from woof.cli import build_parser

    destination.mkdir(parents=True, exist_ok=True)
    out = destination / GENERATED_CONFIG_NAME
    intent = dict(plan.config_intent)
    if drivability_for(intent.get("source")).get("requires_source_root") and not intent.get("data_dir"):
        intent["data_dir"] = str((plan.config_base_dir / "data" / plan.name).resolve())
    arguments = intent_arguments(intent, out=out)
    try:
        args = build_parser().parse_args(["domain", *arguments])
    except SystemExit as stop:
        raise PlanError(layered(
            "run plan 'config.intent' is not a valid `woof domain` "
            "request; argparse refused it above.",
            "The intent block is handed to the wizard's own parser "
            "verbatim, so anything `woof domain` accepts is accepted "
            "here and nothing else is.")) from stop
    args.interactive = False
    from woof.domain_wizard import DomainFitError, domain_main

    try:
        code = domain_main(args, memory=memory)
    except DomainFitError as error:
        # The one refusal a front end most needs the numbers from: the
        # requested shape does not fit the requested card.  The wizard's
        # own sentence is carried verbatim -- it already names the
        # budget, the layout it bottomed out at and what to change --
        # and the structural floor is attached beside it so a form can
        # bound its own inputs instead of guessing.  The figures in that
        # sentence travel beside it as a record (``memory``), so a front
        # end reads how much a too-big draft needs from the record and
        # not from words that differ by source.
        refusal = PlanError(layered(
            f"run plan 'config.intent' does not fit: {error}",
            "The smallest domain this engine will size is "
            f"{json.dumps(domain_size_floor(), indent=2)}"))
        refusal.memory = error.memory_record()
        raise refusal from error
    except ValueError as error:
        # Every other refusal the wizard raises, including the loader's
        # own when it round-trips the emitted bytes before writing them
        # -- an output cadence that is not a whole number of a domain's
        # steps arrives here.  Re-raised as a PlanError so it reaches
        # the caller as a refused plan rather than a failed run, with
        # the engine's sentence intact.
        raise PlanError(
            f"run plan 'config.intent' was refused: {error}") from error
    if code:
        raise PlanError(
            f"`woof domain` exited {code} for this plan's intent; the "
            "refusal it printed is above")
    if not out.is_file():
        raise PlanError(
            f"`woof domain` reported success but wrote no {out.name}")

    resolutions = [{
        "scope": "config", "key": "generated_config",
        "value": str(out), "basis": "intent_route",
        "note": "written by `woof domain " + " ".join(arguments) + "`"}]
    # The wizard resolves `latest` into a concrete cycle and writes THAT
    # into the emitted [fetch] table (never the query -- domain_wizard
    # :3210).  Reading it back is how the resolved cycle is reported
    # without resolving it a second time and risking a different answer.
    emitted = tomllib.load(io.BytesIO(out.read_bytes()))
    hints = emitted.get("fetch") or {}
    if "cycle" in hints:
        requested = str(plan.config_intent.get("cycle", ""))
        resolutions.append({
            "scope": "fetch", "key": "cycle",
            "value": hints["cycle"],
            "basis": ("resolved_latest"
                      if requested.strip().lower() == "latest"
                      else "declared"),
            "note": (
                _latest_cycle_note(str(plan.config_intent.get(
                    "source", "era5")))
                if requested.strip().lower() == "latest"
                else "as declared in the intent")})
    for domain in emitted.get("domain") or ():
        if "nx" in domain and "ny" in domain:
            resolutions.append({
                "scope": "domain", "grid_id": domain.get("grid_id"),
                "key": "nx_ny",
                "value": [domain["nx"], domain["ny"]],
                "basis": "fitted_to_vram_budget",
                "note": "domain size is fitted by the wizard's estimator "
                        "loop from the ladder and the card budget; it is "
                        "not an input"})
    return out, resolutions


def declared_inputs(data) -> list[dict[str, str]]:
    """Every file/directory a ``[case_data]`` block declares, and whether
    it is there yet.

    Used twice: to report readiness in a planning answer, and as the
    gate after the fetch stage.  One function, so "what this run needs"
    cannot mean two different sets.
    """

    entries: list[tuple[str, Path, str]] = [
        *(("forcing", path, "file") for path in data.forcing),
        ("vtable", data.vtable, "file"),
        ("wps_namelist", data.wps_namelist, "file"),
        ("geog_root", data.geog_root, "directory"),
    ]
    overlay = getattr(data, "water_temperature_overlay", None)
    if overlay is not None:
        entries.append(("water_temperature_overlay", overlay, "file"))
    orography = getattr(data, "source_orography", None)
    if orography is not None and getattr(orography, "path", None):
        entries.append(("source_orography", orography.path, "file"))
    return [{
        "role": role, "path": str(path), "kind": kind,
        "present": path.is_dir() if kind == "directory" else path.is_file(),
    } for role, path, kind in entries]


@_review_warning_scope()
def resolve_plan(plan: RunPlan, *, generate_into: Path | None = None,
                 require_inputs: bool = True) -> dict[str, Any]:
    """Load a plan's config through the real seam and describe it.

    Everything ``--resolve`` prints, and everything the ``resolved_plan``
    event carries, is built here -- one function, so the document a
    caller inspects before a run and the event it receives during one
    cannot drift apart.
    """

    from woof.case_data import load_experiment_case_bytes

    resolutions = list(plan.automatic_resolutions)
    generated_text = None
    memory: dict[str, Any] = {}
    scratch: tempfile.TemporaryDirectory | None = None
    try:
        if plan.config_intent is not None:
            # A query mode generates into a throwaway directory and
            # hands the text back in its document; a run generates into
            # its own run directory, where the config and the WPS
            # namelist the [case_data] block references have to STAY --
            # they are inputs to the run and provenance afterwards.
            if generate_into is None:
                scratch = tempfile.TemporaryDirectory(prefix="gpuwm-intent-")
                destination = Path(scratch.name)
            else:
                destination = generate_into
            warnings_generated: list[dict[str, str]] = []
            with collect_warnings(warnings_generated):
                generated, generated_resolutions = generate_intent_config(
                    plan, destination=destination, memory=memory)
            resolutions.extend(generated_resolutions)
            payload = generated.read_bytes()
            generated_text = payload.decode("utf-8")
            config_source = str(generated)
            base_dir = generated.parent
        else:
            warnings_generated = []
            payload = plan.config_bytes()
            config_source = (
                str(plan.config_path) if plan.config_path is not None
                else f"{plan.source}#config.inline")
            base_dir = (plan.config_path.parent
                        if plan.config_path is not None
                        else plan.config_base_dir)

        _grib2_member_contract(plan, payload)
        # Imported here rather than at module scope, on this file's own
        # convention for woof.core.streaming: the run-plan front door is
        # reached by every route, and the streaming module must stay a
        # thing a resident plan never pays for.
        from woof.core.streaming import StreamingRefused

        warnings: list[dict[str, str]] = list(warnings_generated)
        try:
            with collect_warnings(warnings):
                if ROUTES[plan.route].needs_case_data:
                    case_payload = payload
                    geog_override = plan.run_options.get("geog_root")
                    if geog_override is not None:
                        from woof.branch import emit_experiment_toml

                        effective = tomllib.loads(payload.decode("utf-8"))
                        if isinstance(effective.get("case_data"), dict):
                            effective["case_data"]["geog_root"] = geog_override
                            case_payload = emit_experiment_toml(effective).encode("utf-8")
                            resolutions.append({
                                "scope": "case_data", "key": "geog_root",
                                "value": geog_override, "basis": "run_options.geog_root",
                                "note": "The explicit run option replaces the config's "
                                        "geography root; the effective input is listed below."})
                    exp, data = load_experiment_case_bytes(
                        case_payload, source=config_source, base_dir=base_dir,
                        require_inputs=require_inputs)
                else:
                    # The prepared route's configs declare no [case_data]:
                    # their inputs are bound by the prepared cache, not
                    # named in the TOML.  experiment_from_text is the
                    # wizard's own round-trip loader for exactly that shape
                    # -- same build_experiment seam, [fetch] validated,
                    # [case_data] split off if present.
                    from woof.domain_wizard import experiment_from_text

                    exp = experiment_from_text(
                        payload.decode("utf-8"), source=config_source)
                    data = None
        except StreamingRefused as refusal:
            # build_experiment refuses [tiles] mode = 'on' over a nested
            # tree, and it does so as StreamingRefused -- a RuntimeError,
            # which this front door would print as a traceback.  Every
            # other refusal here travels as PlanError (a ValueError) so
            # that woof.cli.main prints one sentence and exits 2, and a
            # config-shaped refusal must not be the exception.  The
            # message is carried verbatim: it is the core's sentence, and
            # restating it here is exactly the drift _streaming_refusal
            # below already refuses to introduce.
            raise PlanError(str(refusal)) from None
        raw = tomllib.load(io.BytesIO(payload))
    finally:
        if scratch is not None:
            scratch.cleanup()

    if plan.run_options.get("devices") is not None:
        from dataclasses import replace
        from woof.core.devices import override_device_count
        exp = replace(exp, devices=override_device_count(exp.devices, plan.run_options["devices"]))
    # Before any stage runs: a multi-card split on an interpreter whose lock
    # is on steps its ranks one at a time (woof.free_threading).
    from woof.free_threading import refuse_multi_card_under_gil
    try:
        door = " ".join(str(plan.source).split()[:2])
        refuse_multi_card_under_gil(
            exp.devices.count, door if door.startswith("woof ") else "woof run-plan")
    except ValueError as error:
        raise PlanError(str(error)) from error
    source = config_source
    inputs = [] if data is None else declared_inputs(data)
    resolutions.extend(_schema_default_resolutions(raw))
    resolutions.extend(_timestep_resolutions(exp))
    resolutions.append({
        "scope": "execution", "key": "execution_mode",
        "value": "in_process", "basis": "front_door_contract",
        "note": "run-plan integrates in THIS process rather than "
                "re-executing under woof's own supervisor, so the pid "
                "in run-manifest.json is the pid doing the model work "
                "and the caller owns restart policy"})

    existing_bundle = _existing_prepared_bundle(plan)
    recipe_refusal = _recipe_plan_refusal(plan, raw, existing_bundle)
    if recipe_refusal is not None:
        raise PlanError(recipe_refusal)
    if plan.route == "prepared" and existing_bundle is None:
        hints = raw.get("fetch") or {}
        if {"source", "cycle"} <= hints.keys():
            from woof.go_cli import checked_config_fetch_cycle

            try:
                checked_config_fetch_cycle(hints, start_time=exp.start_time)
            except ValueError as error:
                raise PlanError(str(error)) from error
        from woof.go_cli import pinned_transport

        try:
            pinned, basis = pinned_transport(hints, plan.run_options.get("transport"))
        except ValueError as error:
            raise PlanError(str(error)) from error
        if pinned is not None:
            resolutions.append({
                "scope": "fetch", "key": "transport", "value": pinned,
                "basis": basis,
                "note": "the one host the fetch stage asks; --transport "
                        "(run_options.transport) wins over [fetch] transport"})
    chain = ("prepared:existing" if existing_bundle is not None else
             _chain_key(plan.route, (raw.get("fetch") or {}).get("source")))
    if chain == "prepared:hrrr":
        # The namelists the HRRR chain runs from, asked of the same
        # function the chain calls, so `woof go --dry-run` and plan
        # review refuse what the real run would: a configuration with no
        # companions beside it whose settings those namelists cannot
        # carry used to pass the dry run and fail three seconds into the
        # real one.
        from woof.hrrr_route_inputs import HrrrRouteInputError, run_route_inputs

        try:
            run_route_inputs(Path(config_source), exp, raw=raw)
        except HrrrRouteInputError as refusal:
            raise PlanError(str(refusal)) from None
    if chain == "prepared:staged":
        # The WPS namelist the staged chain runs from, asked of the same
        # function the chain calls, so `woof go --dry-run` and plan
        # review refuse a configuration the run could not render one for.
        from woof.companion_domains import configuration_wps_namelist

        try:
            configuration_wps_namelist(Path(config_source), exp, raw=raw)
        except ValueError as refusal:
            raise PlanError(str(refusal)) from None
        from woof import fetch_routes
        hints = raw.get("fetch") or {}
        source_id = fetch_routes.canonical_source(str(hints.get("source", "")))
        if source_id in fetch_routes.route_ids():
            route = fetch_routes.route_for(source_id)
            if route.members is not None:
                requested = hints.get("member")
                member, token = fetch_routes.resolve_member(
                    route, None if requested is None else str(requested))
                resolutions.append({
                    "scope": "fetch", "key": "member",
                    "value": {"id": member, "token": token},
                    "basis": "route_default" if requested is None else "declared"})
    from woof.source_drivability import local_input_requested
    if chain == "prepared:staged" and local_input_requested(raw.get("fetch") or {}):
        from woof.local_preparation import review_local_inputs
        hints = raw.get("fetch") or {}
        try:
            snapshot = review_local_inputs(hints,
                data_dir=plan.run_options.get("data_dir") or (plan.config_intent or {}).get("data_dir"),
                base_dir=base_dir, supplements=plan.run_options.get("supplement", ()))
        except ValueError as error:
            raise PlanError(str(error)) from error
        local_root = Path(snapshot["source_root"])
        resolutions.append({
            "scope": "preparation", "key": "local_inputs",
            "value": {"source_root": str(local_root), "sha256": snapshot["sha256"],
                      "file_count": len(snapshot["files"])},
            "basis": "local_source_root"})
    if plan.run_options.get("supplement"):
        from woof.go_cli import config_fetch_request, managed_download_dir
        from woof.launch_supplements import validate_route
        validate_route(plan.run_options["supplement"], chain=chain)
        hints = raw.get("fetch") or {}
        intent = plan.config_intent or {}
        # The config's own root counts here too.  A local-input source
        # has no managed download cache to fall back to, and reading the
        # supplement bindings against one would check them against an
        # empty directory the reader never named.
        source_root = (plan.run_options.get("data_dir") or intent.get("data_dir")
                       or hints.get("source_root")
                       or managed_download_dir(
                           plan.run_dir, _pinned_fetch_hints(plan, config_fetch_request(raw))))
        validate_route(plan.run_options["supplement"], chain=chain,
                       source_root=source_root)
    if existing_bundle is not None:
        unused = {key: plan.run_options[key] for key in ("data_dir", "geog_root")
                  if plan.run_options.get(key) is not None}
        if unused:
            warnings.append({
                "action": "Existing prepared bundle selected; preparation paths "
                          + ", ".join(unused) + " are retained but unused.",
                "why": "The simulation consumes the sealed bundle; no fetch, "
                       "geography rebuild or preparation is requested."})
            resolutions.append({"scope": "execution", "key": "unused_preparation_paths",
                                "value": unused, "basis": "run_options.prepared_root"})
        profile = plan.run_options.get("physics_profile")
        if profile is not None:
            from woof.prepared_single_domain_forecast import named_profile_config_conflicts
            conflicts = named_profile_config_conflicts(
                payload.decode("utf-8"), source=existing_bundle["source"], profile=profile)
            if conflicts:
                raise PlanError(f"Named physics_profile {profile!r} differs from the exact "
                                f"prepared-run configuration: {conflicts}. "
                                "Omit the profile assertion to run the config as written.")
        resolutions.append({
            "scope": "execution", "key": "prepared_root",
            "value": str(existing_bundle["document"].parent),
            "basis": "run_options.prepared_root",
            "note": "Use the existing bundle without fetch or preparation; "
                    "the simulation runner verifies its payload and setup."})

    # WHAT THIS MACHINE CANNOT PREPARE, asked here rather than found
    # later.  This is the route the desktop launches every forecast
    # through, and the experiment is in hand the moment it loads -- while
    # `_execute_prepared_route` and the experiment route both reach their
    # own fetch stage first, and `initialize_real`'s per-domain floor is
    # downstream of the whole downloaded cycle.  So an mp=28 config with
    # external lateral boundaries on a machine without
    # QNWFA_QNIFA_SIGMA_MONTHLY.dat used to pass review, pay for 10-15 GB
    # of transfer and be refused afterwards.  It is refused HERE, on the
    # near side of the fetch and of the run root: `--resolve` is plan
    # review and answers with the sentence and both ways out, and a run
    # never reaches its fetch stage.  Raised as PlanError for the same
    # reason StreamingRefused is converted above -- woof.cli prints one
    # sentence and exits 2, and a config-shaped refusal must not be a
    # traceback.  The sentence and the inventory are woof.config's; this
    # door writes no version of its own.
    # ... AND ONLY OF A CHAIN THAT PREPARES.  A sealed bundle is already
    # prepared: `_existing_prepared_forecast` hands it to the runner,
    # which verifies the payload and reads no WIF dataset
    # (`woof/prepared_single_domain_forecast.py`).  Asking this of a
    # prepared:existing chain refused an mp=28 bundle prepared on a
    # machine that HAD the dataset and consumed on one that does not --
    # a refusal for an input the run never opens, which is the half of
    # the gate law that says a gate must name the breakage it prevents.
    # `woof go --prepared-root` reaches the same chain through this
    # function, so the guard covers that door too; `woof run` has no
    # existing-bundle route and always prepares.
    from woof.config import validate_experiment_preparation

    if existing_bundle is None:
        from woof.preparation_assets import wif_fetch_domains, wif_fetch_resolution

        # Only a chain that actually fetches may defer this dependency.
        # Preparation and initialization still require the acquired dataset.
        fetch_hints = raw.get("fetch") or {}
        pending_wif = (wif_fetch_domains(exp, fetch_hints)
                       if chain in _WIF_ACQUIRING_CHAINS
                       and not local_input_requested(fetch_hints) else ())
        try:
            validate_experiment_preparation(exp, pending_wif_domains=pending_wif)
        except ValueError as refusal:
            raise PlanError(str(refusal)) from None
        if pending_wif:
            resolutions.append(wif_fetch_resolution(pending_wif))
        # A root the [fetch] source's grid does not reach, on the same
        # terms: the source row declares its coverage, the config holds
        # the root, and the preparation otherwise refuses it only after
        # the whole cycle is downloaded and decoded.
        from woof.source_coverage import config_source_coverage_refusal

        uncovered = config_source_coverage_refusal(
            exp, (raw.get("fetch") or {}).get("source"))
        if uncovered is not None:
            raise PlanError(uncovered)

    # A moving nest, decided and REPORTED before anything is fetched.
    # The chain is read off the config's own [fetch] table, which is
    # what `_execute_prepared_route` dispatches on, so the chain judged
    # here is the chain that runs.  A plan with no follow source adds
    # nothing to this document.
    moving_source = (raw.get("fetch") or {}).get("source")
    decision = follow_statics_decision(exp, chain=chain, source=moving_source)
    if decision is not None:
        from woof.source_cli import preparation_statics

        if decision["refusal"] is not None:
            raise PlanError(decision["refusal"])
        # Published without the refusal slot: reaching here means there
        # was none, and a null field inviting a caller to test it would
        # imply this document ever carries a live one.
        decision = {key: value for key, value in decision.items()
                    if key != "refusal"}
        resolutions.append({
            "scope": "preparation", "key": "statics_corridor",
            "value": decision["statics_corridor"],
            "basis": "relocation_follow",
            "note": (
                "The existing bundle must carry verified statics corridors; "
                "the simulation runner checks them before execution."
                if existing_bundle is not None else (
                "Declared followers " + ", ".join(
                    f"d{gid:02d}" for gid in decision["follower_grid_ids"])
                + (" require verified statics corridors; preparation includes "
                   "--statics-corridor for each moving subtree."
                   if decision["statics_corridor"] else
                   " rebuild footprint statics from the retained geography source.")
                if "follower_grid_ids" in decision else
                _CORRIDOR_RESOLUTION_NOTE[decision["delivery"]].format(
                    grid_id=decision["relocation_grid_id"],
                    stage=preparation_statics(decision["chain"], source=moving_source)["stage"])))})

    # [tiles], on the same terms and in the same place.  A configured
    # streaming mode that the chain cannot honour is refused HERE --
    # from the config, before the fetch -- rather than at the first tile
    # buffer, which on the HRRR chain is after a download, two
    # preparations and a whole resident tree construction.  A plan that
    # configures no [tiles] adds nothing to this document.
    tiles = streaming_decision(exp, chain=chain)
    if tiles is not None:
        if tiles["refusal"] is not None:
            raise PlanError(tiles["refusal"])
        tiles = {key: value for key, value in tiles.items()
                 if key != "refusal"}
        resolutions.append({
            # NOT "tiles".  `_schema_default_resolutions` already emits
            # that key -- scope "experiment", value the whole default
            # StreamingOptions -- for a config that spells no [tiles].
            # The two are mutually exclusive today, so a consumer keyed
            # on the name would never see both; it would just see one
            # key whose value is sometimes a table of defaults and
            # sometimes a mode, which is a shape it cannot parse.
            "scope": "execution", "key": "tiles_delivery",
            "value": tiles["mode"],
            "basis": "experiment_config",
            "note": _STREAMING_RESOLUTION_NOTE[tiles["delivery"]].format(
                mode=tiles["mode"], store=tiles["store"], chain=chain,
                root=tiles["streamable_grid_id"],
                nests=("" if not tiles["resident_grid_ids"] else
                       "  d"
                       + ", d".join(f"{grid:02d}" for grid
                                    in tiles["resident_grid_ids"])
                       + " therefore run resident, and mode = 'auto' "
                         "reaches that same refusal for any of them "
                         "autoplan says does not fit."))})

    return {
        "schema": RESOLVE_SCHEMA,
        "plan": {
            "name": plan.name, "route": plan.route,
            "source": plan.source, "sha256": plan.sha256,
            "config_kind": plan.config_kind,
            "config_source": source,
            "config_sha256": hashlib.sha256(payload).hexdigest(),
            "run_dir": str(plan.run_dir),
            "fetch_args": (list(plan.fetch_arguments)
                           if plan.fetch_arguments else None),
            "run_options": dict(plan.run_options),
        },
        # The generated TOML, verbatim.  An intent plan's caller never
        # typed this text, so it is the one thing they cannot look up:
        # a front end shows it, and a reader can check what the wizard
        # decided on their behalf before anything runs.
        "generated_config": generated_text,
        # The moving-nest decision as a record, beside the sentence
        # `automatic_resolutions` carries: a front end that wants to
        # draw a corridor toggle reads this, and one that just prints
        # the resolutions gets the same fact in prose.  ``null`` when
        # the config moves no nest.
        "moving_nest": decision,
        # The [tiles] decision as a record, for the same reason and on
        # the same terms: ``null`` when the config configures none, and
        # otherwise the grid that CAN stream beside the grids that
        # cannot.  A run cannot answer this for itself -- a grid that
        # declined to stream is absent from the stepper dict, and absent
        # is what an unconfigured grid looks like too -- so a front end
        # that wants to show which domain will stream has to be told
        # before the run rather than after it.
        "tiles": tiles,
        # The card memory the wizard fitted an intent plan to, as bytes:
        # the binding phase's peak envelope beside the budget it was
        # held to.  A front end shows how close the draft is to its card
        # from this record, whichever route the source takes; it was
        # read off printed lines before, and the route that defers
        # `woof check` until its inputs are fetched (ERA5) printed none
        # of them.  ``null`` for a plan that names its own config:
        # nothing was fitted, and ``--estimate`` prices that config.
        "memory": memory or None,
        "configuration": _config_snapshot(exp, data),
        "declared_inputs": inputs,
        "inputs_present": all(entry["present"] for entry in inputs),
        "domain_size_floor": domain_size_floor(),
        # What the run will write, download and preparation included,
        # from the same projection execute_plan refuses on before its
        # download.
        "disk": _disk_projection(plan, exp, raw=raw, data=data,
                                 fetch_arguments=plan.fetch_arguments,
                                 base_dir=base_dir),
        "automatic_resolutions": resolutions,
        "warnings": warnings,
    }, exp, data


def _plan_recipe(plan: RunPlan, raw: Mapping[str, Any]) -> str | None:
    """The member-source recipe this plan's ensemble request names, or None.

    ``run_options.ensemble`` wins over the configuration's ``[ensemble]``
    table, as it does when the request is built for the run.  A trajectory
    list alone is the multi-model recipe.
    """

    table = plan.run_options.get("ensemble")
    if table is None and isinstance(raw, Mapping):
        table = raw.get("ensemble")
    if not isinstance(table, Mapping):
        return None
    if table.get("recipe") is not None:
        return str(table["recipe"])
    if table.get("member_variants"):
        return "member-roster"
    return "multi-model" if table.get("trajectories") else None


#: Run options a recipe request does not consume, each with the breakage a
#: silent acceptance would cause.  The recipe door fetches and prepares
#: every member by its own source's chain (:mod:`woof.ensemble.recipe_door`).
_RECIPE_UNCONSUMED_OPTIONS = {
    "supplement": "it binds one donor file of one trajectory, and every member "
                  "is prepared from its own trajectory",
    "data_dir": "it names one existing download, and a recipe downloads one "
                "window per member into its own request cache",
    "physics_profile": "it asserts a suite to one chain's preparer, and each "
                       "member is prepared by its source's own chain, which is "
                       "handed no assertion",
    "render_section": "the ensemble draws its aggregate maps and no stage of "
                      "it cuts a vertical section",
}


def _recipe_plan_refusal(plan: RunPlan, raw: Mapping[str, Any],
                         existing_bundle) -> str | None:
    """Why this plan's recipe request cannot run as planned, or None.

    Asked at plan resolution, so ``--resolve`` and a run both answer
    before anything is fetched.  Breakage it prevents: a recipe on the
    experiment route, or over an existing prepared bundle, ran its whole
    download or restore for ONE trajectory and was refused only at the
    forecast stage, where the session finds no member sources; and a run
    option the recipe door does not consume was accepted and read by
    nothing.
    """

    recipe = _plan_recipe(plan, raw)
    if recipe is None or recipe == "wofs" and plan.route == "experiment":
        return None
    door = f"woof ensemble CONFIG --recipe {recipe}"
    if plan.route != "prepared":
        return (f"the {recipe} ensemble recipe fetches and prepares each member's "
                f"own source trajectory, and the {plan.route!r} route runs the one "
                "trajectory its [case_data] files hold: every member would be a "
                f"copy of it. Next: {door} on a config with a [fetch] table "
                "(woof domain writes one)")
    if existing_bundle is not None or plan.run_options.get("restart") is not None:
        return (f"the {recipe} ensemble recipe prepares each member's own source "
                "trajectory, and run_options.prepared_root / restart name one "
                "prepared trajectory: every member would be a copy of it. Next: "
                "remove prepared_root and restart from the plan")
    intent = plan.config_intent or {}
    given = [key for key in _RECIPE_UNCONSUMED_OPTIONS
             if plan.run_options.get(key) not in (None, [], ())
             or (key == "data_dir" and intent.get("data_dir"))]
    if given:
        return (f"the {recipe} ensemble recipe does not use "
                + ", ".join(f"run_options.{key}" for key in given) + ": "
                + "; ".join(f"{key}: {_RECIPE_UNCONSUMED_OPTIONS[key]}" for key in given)
                + ", so it would be read by nothing. Next: remove "
                + ("it" if len(given) == 1 else "them") + " from the plan")
    return None


def _planned_download(plan: RunPlan, raw: Mapping[str, Any], data, *,
                      fetch_arguments: Sequence[str] | None,
                      run_dir: Path | None = None,
                      base_dir: Path | None = None
                      ) -> tuple[dict[str, Any] | None, Path | None, bool]:
    """The request this plan's run downloads with, where it lands, and whether that folder is keyed to it.

    One answer for ``--resolve``, ``--estimate`` and the refusal before the
    fetch.  A plan's own fetch block is the request when it has one.  The
    prepared route downloads from its config's ``[fetch]`` table, into
    ``data_dir`` or the managed directory under the run directory; that
    table was never read here, which is why the review said "no [fetch]
    in this plan" for a plan that downloaded 21 GB.  The experiment route
    downloads the declared forcing its ``[fetch]`` table names when the
    forcing is not on disk.  A reused prepared bundle downloads nothing.

    The managed folder is keyed to the request, so all of it is this
    request's download, a half-finished one included.  Any other folder
    was named by hand and may hold anything.
    """

    from woof import download_budget

    if _existing_prepared_bundle(plan) is not None:
        return None, None, False
    if fetch_arguments is not None:
        request = download_budget.request_from_arguments(list(fetch_arguments))
        out = (request or {}).get("out")
        return request, (Path(out) if out else None), False
    hints = raw.get("fetch") if isinstance(raw, Mapping) else None
    if not isinstance(hints, Mapping) or not hints.get("source"):
        return None, None, False
    if plan.route == "prepared":
        from woof.go_cli import config_fetch_request

        # The request the chain's fetch stage makes, model top and pinned
        # host included, so the price counts the levels that top adds and
        # the objects that host serves.
        request = _pinned_fetch_hints(plan, config_fetch_request(dict(raw)))
        if request.get("source_root"):
            return request, None, False
        intent = plan.config_intent or {}
        data_dir = plan.run_options.get("data_dir") or intent.get("data_dir")
        if data_dir:
            return request, Path(data_dir), False
        from woof.go_cli import managed_download_dir

        root = run_dir if run_dir is not None else plan.run_dir
        try:
            return request, managed_download_dir(root, request), True
        except (OSError, ValueError):
            return request, root / "downloads", False
    try:
        arguments = declared_forcing_fetch(
            raw, data, base_dir=base_dir,
            cache_root=run_dir if run_dir is not None else plan.run_dir)
    except PlanError:
        # The run refuses these inputs itself, before its fetch; the table
        # still says what would be downloaded.
        return dict(hints), None, False
    if arguments is None:
        return None, None, False
    request = download_budget.request_from_arguments(arguments)
    out = (request or {}).get("out")
    return request, (Path(out) if out else None), False


def _preparation_chain(plan: RunPlan, raw: Mapping[str, Any]) -> str | None:
    """The chain whose preparation this run writes, or None when it reuses a bundle."""

    if _existing_prepared_bundle(plan) is not None:
        return None
    if plan.route != "prepared":
        return plan.route
    return _chain_key(plan.route, ((raw or {}).get("fetch") or {}).get("source"))


def _refuse_one_input_ensemble(plan: RunPlan, raw: Mapping[str, Any]) -> None:
    """Refuse, before the fetch, N > 1 members that would all run this plan's one input.

    Breakage it prevents: the experiment route, the native and staged
    chains and an existing prepared bundle hold ONE trajectory's inputs.
    N > 1 members on them are N copies of one forecast, and the ensemble
    session only finds that out at its first member, after the download
    and the preparation.  The ``go`` chain is not refused here: ``woof
    go`` hands a member count to the door that plans each member's own
    source, and refuses it itself where no plan exists.
    """

    request = _member_request(plan, raw)
    if request is None or _preparation_chain(plan, raw) == "prepared:go":
        return
    from woof.ensemble import member_inputs

    try:
        member_inputs.refuse_one_input(
            request, "This plan's route prepares one trajectory, so every member would run it.")
    except ValueError as error:
        raise PlanError(str(error)) from error


def _member_request(plan: RunPlan, raw: Mapping[str, Any]):
    """The ensemble request this plan makes, or None: ``run_options.ensemble`` over the config's table."""

    value = plan.run_options.get("ensemble")
    if value is None:
        value = raw.get("ensemble") if isinstance(raw, Mapping) else None
    if value is None:
        return None
    from woof.ensemble.request import EnsembleRequest

    return EnsembleRequest.from_mapping(value)


def _go_plans_members(plan: RunPlan, raw: Mapping[str, Any]) -> bool:
    """Does this plan hand a plain member count to the door that plans each member's source?

    True for N > 1 members with no recipe named on the ``go`` chain:
    ``woof go`` runs them as the source's operational ensemble, so the
    windows the run fetches are the members', not the config's own.
    """

    from woof.ensemble import member_inputs

    return (member_inputs.needs_member_sources(_member_request(plan, raw))
            and _preparation_chain(plan, raw) == "prepared:go")


def _present_download_bytes(request: Mapping[str, Any] | None, directory: Path | None,
                            keyed: bool) -> int:
    """What of this request's download already lies in ``directory``: it is not written again.

    All of a folder keyed to the request.  In a folder named by hand, only
    the files a fetch receipt of this same request names
    (:func:`woof.download_budget.present_bytes`): counting everything in
    it priced the download of a user's folder of unrelated files at 0.
    """

    from woof import download_budget

    if directory is None:
        return 0
    if not keyed:
        return download_budget.present_bytes(directory, request)
    try:
        if not directory.is_dir():
            return 0
        return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    except OSError:
        return 0


def _disk_projection(plan: RunPlan, exp, *, raw: Mapping[str, Any], data,
                     fetch_arguments: Sequence[str] | None,
                     run_dir: Path | None = None,
                     download_keyed: bool | None = None,
                     base_dir: Path | None = None) -> dict[str, Any]:
    """What this plan's run writes, from :func:`woof.disk_budget.projected_run_bytes`.

    ``download_keyed`` overrides whether the download folder is keyed to
    the request (:func:`_planned_download`), for a caller that knows: a
    chain handing its fetch arguments over names a managed folder the
    arguments alone cannot tell from one named by hand.
    """
    from woof import disk_budget

    keep = int(plan.run_options.get("keep_checkpoints") or 0)
    request, directory, keyed = _planned_download(
        plan, raw, data, fetch_arguments=fetch_arguments, run_dir=run_dir,
        base_dir=base_dir)
    if download_keyed is not None and directory is not None:
        keyed = bool(download_keyed)
    return dict(disk_budget.projected_run_bytes(
        exp, keep_checkpoints=keep or None, fetch=request,
        chain=_preparation_chain(plan, raw), render=_plan_draws_pictures(plan),
        render_products=_plan_render_products(plan) or "none",
        download_present_bytes=_present_download_bytes(request, directory, keyed),
        resume_seconds=_resume_seconds(plan)),
        keep_checkpoints=keep, download_dir=None if directory is None else str(directory))


def _resume_seconds(plan: RunPlan) -> float | None:
    """The model time the run resumes from, read off its checkpoint, or None.

    Disk admission prices only what a resumed run writes after its
    checkpoint.  A checkpoint whose header cannot be read here is left to
    the restore, which refuses it by name; admission then prices the
    whole run, as it does for a cold start.
    """
    checkpoint = plan.run_options.get("restart")
    if checkpoint is None:
        return None
    from woof.io.restart import _admissible_elapsed_seconds, read_restart_header

    try:
        header = read_restart_header(Path(checkpoint))
        return _admissible_elapsed_seconds(
            header.get("elapsed_seconds"), f"restart file {checkpoint}")
    except Exception:
        return None


class DiskRefusal(str):
    """A disk admission's refusal: its words, and the folders it names as the place to act.

    A string, so a door raises it as its own refusal unchanged.
    ``folders`` are the folders the remedy points at (the compose scratch
    folder a frame stream does not fit in, or the one a
    ``WOOF_COMPOSE_SCRATCH`` names that is not there): a page that hides
    machine paths still shows these, or its remedy names nowhere.
    """

    folders: tuple[str, ...] = ()

    def __new__(cls, words: str, folders: Sequence[str] = ()):
        refusal = super().__new__(cls, words)
        refusal.folders = tuple(str(folder) for folder in folders)
        return refusal


def disk_admission_refusal(plan: RunPlan, exp, *, raw: Mapping[str, Any], data,
                           fetch_arguments: Sequence[str] | None, run_dir: Path,
                           download_keyed: bool | None = None,
                           prep_root: Path | None = None,
                           warn: Callable[[str, str, str | None], None] | None = None,
                           hydro_config=None, hydro_args=None,
                           ) -> DiskRefusal | None:
    """The refusal for a run its disks cannot hold, or None: the one disk admission.

    Asked before the run's download, and by ``woof go``'s own GFS chain
    before it claims its run folder too.  THE BREAKAGE: a run whose
    download, preparation, history, checkpoints and pictures do not fit
    fills its disk partway, ends with nothing usable, and can stop other
    work on that disk; a 1 km run filled its disk at hour nine of twelve.
    ``woof run-plan`` refused such a run, but the GFS chain ``woof go``
    runs by itself enters no run plan, so it claimed its folder and
    started the download on a disk with no room.

    ``run_dir`` is where the run writes (it need not exist yet; its
    nearest existing parent is measured).  A download that lands on
    another disk is compared with that disk, and the rest of the run with
    the run directory's (:func:`woof.disk_budget.disk_refusal`).

    The preparation's decoded frame stream is priced and checked here
    too, on the disk that holds its compose scratch folder: beside
    ``prep_root`` (where the preparation writes; the staged chain's
    ``chain/prep`` when not given) or wherever ``WOOF_COMPOSE_SCRATCH``
    points.  THE BREAKAGE: the stream is sized by the SOURCE grid (a GEM
    GDPS 48 hour window kept whole stages about 82 GB), and the engine refused it
    only after the whole download and the first valid time's decode.  A
    ``WOOF_COMPOSE_SCRATCH`` naming no folder is refused here as well:
    the preparation refuses it only once it starts composing.

    ``warn`` is told ``(message, detail, folder)`` when a windowed
    source's stream may not fit where its certain part does
    (:func:`woof.disk_budget.disk_warning`); such a run is admitted.
    """

    from woof import disk_budget

    from woof.hydro.runtime import pending_launch_args
    pending_hydro = pending_launch_args()
    if hydro_config is None and pending_hydro is not None:
        hydro_config = getattr(pending_hydro, "config", None)
        hydro_args = pending_hydro
    projection = _disk_projection(
        plan, exp, raw=raw, data=data, fetch_arguments=fetch_arguments,
        run_dir=run_dir, download_keyed=download_keyed)
    hydro_basis = None
    if hydro_config is not None:
        from woof.hydro.runtime import hydro_disk_estimate
        from woof.hydro.door import cache_root
        hydro_bytes, hydro_basis = hydro_disk_estimate(hydro_config, hydro_args)
        hydro_root = cache_root(hydro_config)
        if hydro_bytes and not disk_budget.same_disk(hydro_root, run_dir):
            hydro_free = disk_budget.free_bytes(hydro_root)
            if hydro_free is not None and hydro_bytes > hydro_free:
                return DiskRefusal(layered(
                    f"Hydro preparation would write about {hydro_bytes / 2**30:.1f} GiB "
                    f"into {hydro_root}, which has {hydro_free / 2**30:.1f} GiB free, "
                    "so it would stop partway when that disk fills.",
                    "Refused before the download and before hydro preparation, so nothing was spent. "
                    "Free some disk in the hydro cache. " + (hydro_basis or "")),
                    folders=(str(hydro_root),))
        else:
            projection["preparation_bytes"] = int(projection.get("preparation_bytes") or 0) + hydro_bytes
    if (projection.get("compose_scratch") or {}).get("composes"):
        from woof.ingest.source_coverage import (COMPOSE_SCRATCH_ENV,
                                                  compose_scratch_override_refusal)
        absent = compose_scratch_override_refusal()
        if absent is not None:
            return DiskRefusal(layered(
                absent + ".",
                "Refused before any download or preparation, so nothing was spent."),
                folders=(os.environ[COMPOSE_SCRATCH_ENV],))
    download_dir = projection.get("download_dir")
    download_free = (None if not download_dir
                     or disk_budget.same_disk(Path(download_dir), run_dir)
                     else disk_budget.free_bytes(Path(download_dir)))
    scratch_folder = _compose_scratch_folder(
        _staged_prep_root(run_dir) if prep_root is None else prep_root, projection)
    scratch_free = (None if scratch_folder is None
                    or disk_budget.same_disk(scratch_folder, run_dir)
                    else disk_budget.free_bytes(scratch_folder))
    run_free = disk_budget.free_bytes(run_dir)
    refusal = disk_budget.disk_refusal(
        projection, run_free, download_free=download_free,
        scratch_free=scratch_free, scratch_folder=scratch_folder)
    stream_basis = (f"Frame stream: {projection['compose_scratch']['basis']}.  "
                    if scratch_folder is not None else "")
    if refusal is not None:
        return DiskRefusal(layered(
            refusal[0].upper() + refusal[1:] + ".",
            ("Refused before the download and before hydro preparation, so nothing was spent.  "
             if hydro_basis else "Refused before the download, so nothing was spent.  ")
            +
            f"Download: {projection['download']['basis']}.  "
            f"Preparation: {projection['preparation']['basis']}.  "
            + stream_basis
            + (f"Hydro: {hydro_basis}.  " if hydro_basis else "")
            + f"Bytes per cell: {disk_budget.MEASURED}."),
            folders=((str(scratch_folder),)
                     if scratch_folder is not None and str(scratch_folder) in refusal
                     else ()))
    caution = disk_budget.disk_warning(
        projection, run_free, download_free=download_free,
        scratch_free=scratch_free, scratch_folder=scratch_folder)
    if caution is not None and warn is not None:
        warn(caution[0].upper() + caution[1:] + ".", stream_basis.strip(),
             None if scratch_folder is None else str(scratch_folder))
    return None


class collect_warnings:
    """Capture :func:`woof.explain.warn` output as structured records.

    The library's warnings are one-line stderr sentences by design, and
    they stay that way.  This attaches a sink to the same call so a
    machine consumer receives them as fields instead of having to
    recognize them in a text stream.
    """

    def __init__(self, sink: list[dict[str, str]]):
        self._sink = sink
        self._observer = None

    def __enter__(self) -> "collect_warnings":
        from woof import explain

        def observer(record: Mapping[str, str]) -> None:
            self._sink.append(dict(record))

        self._observer = observer
        explain.add_warning_observer(observer)
        return self

    def __exit__(self, *exc_info) -> None:
        from woof import explain

        if self._observer is not None:
            explain.remove_warning_observer(self._observer)
            self._observer = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Route:
    """One existing front door, named so a plan can ask for it.

    Route names are generic on purpose.  A route is a way of running the
    model, never a particular experiment: nothing here may name a case.
    """

    name: str
    summary: str
    run_options: frozenset[str]
    execute: Callable[..., Any]
    #: Whether this route's configs carry a ``[case_data]`` table.  The
    #: config-driven route requires one (it IS the declared-input
    #: contract); the prepared route's configs have none, because the
    #: prepared cache binds the inputs instead.
    needs_case_data: bool = True


def _experiment_render_products(plan: RunPlan) -> str | None:
    """The products the experiment route draws, or ``None`` for no render.

    An intent plan is the page's and the wizard's door, and the prepared
    route draws its pictures unless told not to.  Without this default an
    ERA5 start, which runs here, finished its forecast and left the
    forecast page with nothing to show.  A config the caller authored
    keeps the old default of no render.
    """
    products = plan.run_options.get("render_products")
    if products is None and plan.config_intent is not None:
        from woof.first_products import DEFAULT_RENDER_PRODUCTS
        products = DEFAULT_RENDER_PRODUCTS
    return products


def _plan_draws_pictures(plan: RunPlan) -> bool:
    """Whether the route this plan runs draws any picture at all.

    The disk projection reads this, so a run is charged for pictures only
    when its route will draw them.  ``none`` draws nothing on every
    route.  Unset, the prepared route's chain draws its default set, and
    the experiment route draws what :func:`_experiment_render_products`
    resolves, which is nothing for an authored config.
    """
    products = _plan_render_products(plan)
    return products is not None and str(products).strip().lower() != "none"


def _plan_render_products(plan: RunPlan) -> str | None:
    """The product request used by both disk admission and route defaults."""
    if plan.route == "experiment":
        return _experiment_render_products(plan)
    from woof.first_products import DEFAULT_RENDER_PRODUCTS
    products = plan.run_options.get("render_products")
    return DEFAULT_RENDER_PRODUCTS if products is None else products


def _execute_experiment_route(plan: RunPlan, *, exp, data, config_path,
                              observer: RunObserver) -> Mapping[str, Any]:
    """The config-driven experiment route: what ``woof run CONFIG`` runs.

    One call, into the same :func:`woof.runtime.run_experiment` the
    ``run`` subcommand reaches, with this module's observer in the
    ``progress_callback`` slot that function already had.
    """

    from woof import runtime

    products = _experiment_render_products(plan)
    render_plan = None
    if products is not None and str(products).strip().lower() != "none":
        render_plan = {"run": plan.run_dir, "wrfout_dir": plan.run_dir,
                       "render": plan.run_dir / "png", "render_products": products,
                       "publish_root": plan.run_options.get("publish_root") or plan.run_dir / "www",
                       "publish_cmd": plan.run_options.get("publish_cmd"),
                       "hydro": plan.run_options.get("hydro") or "auto",
                       "render_section": plan.run_options.get("render_section"),
                       **({"restart": plan.run_options["restart"]}
                          if plan.run_options.get("restart") is not None else {})}
        observer.arm_first_products(render_plan)
    restart = plan.run_options.get("restart")
    with _preparation_relay(observer):
        summary = runtime.run_experiment(
            exp, data, plan.run_dir,
            restart=None if restart is None else Path(restart),
            progress_callback=observer,
            health_debug=bool(plan.run_options.get("health_debug")))
    if render_plan is not None:
        _finish_render(render_plan, observer=observer)
    return {
        "wrfout_count": len(summary.wrfout_paths),
        "completed_seconds": float(summary.completed_seconds),
        "nan_free": bool(summary.nan_free),
        "restarted": restart is not None,
        "hydro": getattr(summary, "hydro_receipt", None),
    }


#: ``woof go``'s six stage labels, mapped to the run-plan stage each
#: belongs to.  The chain's order is authority, fetch, manifest,
#: prepare, forecast, render -- so ``prepare`` opens, closes for the
#: download, and opens again.  Stages repeat on this route and the
#: pairs stay strictly ordered; a consumer keyed on "which stage is
#: open" reads it correctly either way.  The go label itself rides on
#: the event as the stage's phase, so nothing is flattened away.
_GO_STAGES = {
    "authority": "prepare",
    "fetch": "fetch",
    "manifest": "prepare",
    "prepare": "prepare",
    "forecast": "forecast",
    "render": "finalize",
}


class _GoObserver:
    """``woof go``'s stage hooks, rendered onto the run-plan stages.

    ``go`` reports at stage granularity and hands over the running
    stage's own published progress file; the forecast stage, hosted in
    process, additionally drives this object as the RUNNER's progress
    callback -- so the same object is both the chain observer and the
    model observer, and the events interleave in real order.
    """

    def __init__(self, observer: RunObserver, *, door: str = "go"):
        self._observer = observer
        self._timeline = getattr(observer, "_timeline", Ledger())
        #: The command the reader typed, for the one warning below that
        #: quotes it.  `woof downscale` runs this same render stage,
        #: and a warning in ITS event stream naming `woof go` sends a
        #: reader to a command they did not run.
        self._door = door
        self.failure: dict[str, Any] | None = None
        #: The chain stage `woof go` most recently opened, so a stop
        #: can be reported against the stage it landed in.
        self.current_stage: str | None = None
        #: Whether `woof go`'s chained preparation published its head,
        #: so the end of its preparation stage is said as the seal.
        self._head_ready = False
        #: The start wait said on the heartbeat before the forecast
        #: begins: ``(lead, since_utc)`` of the lead the as-posted fetch
        #: waits for, or ``None`` (:meth:`_start_wait`).
        self._start_waiting: tuple[Any, str] | None = None
        self._forecast_begun = False

    # -- woof go's chain hooks ---------------------------------------

    def bind_timeline(self, run, ledger):
        self._timeline = ledger
        self._observer._timeline = ledger
        self._observer._timeline_forecast_dir = Path(run)

    def stage_begin(self, *, label: str, command) -> None:
        self._timeline.emit(GO_STAGES.get(label, "prep." + label.replace(" ", "_")),
                                      "begin", attrs={"label": label})
        self.current_stage = label
        if label == "forecast":
            # From here the forecast says its own waits (its seams).
            self._forecast_begun = True
            self._end_start_wait()
        self._observer.enter_stage(_GO_STAGES[label], phase=label)

    def _start_wait(self, schedule: Mapping[str, Any]) -> None:
        """Before the forecast begins, the run waits on a start need's lead.

        The as-posted fetch runs beside the preparation (``woof go``'s
        GFS chain) and its schedule is that stage's progress file.  The
        run cannot start without its start needs (DESIGN A136 3.3, the
        schedule's ``start_needs``), so a start-need lead of the window's
        own source in state ``waiting`` (polled, not posted by any host)
        is the run's wait: the heartbeat says ``waiting:source`` with the
        lead and its scheduled and late times, as the forecast says a wait
        at a seam.  The breakage this prevents: launched before the first
        leads post (the site's launch rule), the run's heartbeat and GUI
        export said ``preparing:worker-start`` for as long as the source
        kept it waiting.

        Any other lead the fetch waits for is not the run's wait (DESIGN
        3.5: ``waiting:source`` is said when the run itself is blocked on a
        lead): the fetch polls later leads as the preparation builds the
        head from start needs that are in, and saying ``waiting:source``
        then hid the preparation's own progress behind a false wait.
        """

        if self._forecast_begun:
            return
        from woof.chain_events import start_wait_row

        # The same row the run's stream says as its `phase: start`
        # source wait (GoChainEvents._relay_start_wait).
        row = start_wait_row(schedule)
        if row is None:
            self._end_start_wait()
            return
        lead = row.get("lead")
        if self._start_waiting is None or self._start_waiting[0] != lead:
            self._start_waiting = (
                lead, datetime.now(timezone.utc).isoformat())
        self.waiting("source", since_utc=self._start_waiting[1],
                     lead=lead, expected_at=row.get("expected_at"),
                     late_at=row.get("late_at"))

    def _end_start_wait(self) -> None:
        if self._start_waiting is not None:
            self._start_waiting = None
            self.waited()

    def prepare_head_ready(self, *, head_sha256: str) -> None:
        # `woof go`'s chained preparation (the GFS chain) published its
        # head and the forecast starts beside the rest of it.  Said on the
        # run's stream as the staged route says it, and its seal at the
        # end of the preparation stage below, so a run page says how many
        # boundary times are ready; the page of a plain GFS run never knew
        # its forecast ran beside its preparation.
        self._head_ready = True
        self._observer.events.emit("prepare_head_ready",
                                   head_sha256=str(head_sha256))

    def stage_heartbeat(self, *, label: str, elapsed_seconds: float,
                        progress) -> None:
        # A subprocess stage can only be observed through what it
        # publishes.  The forecast stage runs in process and reports
        # per step through __call__ below, so this is the accurate
        # coarse signal for the stages that do not.
        if not isinstance(progress, dict):
            return
        from woof.source_posting import SCHEDULE_SCHEMA

        if label == "fetch" and progress.get("schema") == SCHEDULE_SCHEMA:
            self._start_wait(progress)
            return
        if progress.get("schema") == "gpuwm.prepare-progress/v1":
            preparation = {key: progress[key] for key in
                           ("schema", "status", "phase", "phase_index", "phases_total", "elapsed_seconds")
                           if key in progress}
            self._observer.warn("preparation_progress", "Preparing forecast inputs",
                                phase=label, preparation=preparation)
        model_seconds = progress.get("model_elapsed_seconds")
        if not isinstance(model_seconds, (int, float)):
            return
        self._observer.stage_progress(
            phase=label, elapsed_seconds=elapsed_seconds,
            model_seconds=float(model_seconds),
            status=progress.get("status"))

    def stage_failed(self, *, label: str, exit_code: int, diagnostic: str) -> None:
        # The first failure is the run's.  An as-posted fetch runs beside
        # the preparation and the forecast (DESIGN A136 2.5), so a second
        # stage can fail after the first (a preparation that ends because
        # its fetch failed); the later one is a consequence, and a record
        # naming it sent the reader to the wrong stage.
        if self.failure is None or self.failure.get("stage") == label:
            self.failure = {"stage": label, "exit_code": exit_code,
                            "diagnostic": diagnostic}

    def chain_failed(self, *, label: str, exit_code: int,
                     diagnostic: str) -> None:
        # `woof go` names the stage whose failure is the chain's when it
        # decided that after the stages beside it ended: the as-posted
        # fetch failed first and the preparation, told by its failure
        # record, may have ended before the fetch did.
        self.failure = {"stage": label, "exit_code": exit_code,
                        "diagnostic": diagnostic}

    def stage_warning(self, *, label: str, code: str, message: str,
                      **fields) -> None:
        # A warning the chain composed from an artifact after a stage
        # exited (a render's leftover working stores, today): the stage's
        # own stderr is captured and never reaches this stream.
        self._observer.warn(code, message, stage=label, **fields)
    def warn(self, code: str, message: str, **fields: Any) -> None:
        """A chain stage's warning, onto the run's own event stream."""

        self._observer.warn(code, message, **fields)

    def stage_end(self, *, label: str, exit_code: int, ok: bool,
                  elapsed_seconds: float, progress) -> None:
        self._timeline.emit(GO_STAGES.get(label, "prep." + label.replace(" ", "_")),
                                      "end", attrs={"label": label, "ok": bool(ok), "exit_code": exit_code})
        if label == "render" and isinstance(progress, dict) and progress.get("schema") == "gpuwm.render-summary.v1":
            self._observer._render_summary = dict(progress)
        if label == "prepare" and ok and self._head_ready:
            self._head_ready = False
            self._observer.events.emit("prepare_sealed", stage=label)
        if not ok:
            if self.failure is None:
                self.failure = {"stage": label, "exit_code": exit_code}
            self._observer.warn(
                "chain_stage_failed",
                f"`woof {self._door}` stage {label!r} exited "
                f"{exit_code}; no later stage ran, because each consumes "
                "the previous one's output",
                stage=label, exit_code=exit_code)

    def stage_secondary_end(self, *, label: str, exit_code: int, ok: bool,
                            elapsed_seconds: float, progress,
                            diagnostic: str) -> None:
        """Record the later stage's exit without replacing the run's failure."""

        events = getattr(self._observer, "events", None)
        emit = getattr(events, "emit", None)
        if callable(emit):
            # This stage ran beside the foreground. Closing the observer's
            # current stage here would close the primary forecast instead.
            emit("stage_finished", stage=_GO_STAGES[label], phase=label,
                 wall_seconds=round(float(elapsed_seconds), 6), phases=[label],
                 secondary=True, ok=bool(ok), exit_code=int(exit_code),
                 diagnostic=diagnostic)

    # -- the runner's progress protocol, for the hosted forecast ------

    def __call__(self, **event) -> None:
        self._observer(**event)

    def preparing(self, phase: str) -> None:
        self._observer.preparing(phase)

    # Forwarded when the wrapped observer has them: `woof downscale`
    # wraps its child's own progress object, which has none of the three.

    def writing(self, phase: str, *, work_bytes: int | None = None) -> None:
        hook = getattr(self._observer, "writing", None)
        if hook is not None:
            hook(phase, work_bytes=work_bytes)

    def written(self) -> None:
        hook = getattr(self._observer, "written", None)
        if hook is not None:
            hook()

    def waiting(self, on: str, **record) -> None:
        hook = getattr(self._observer, "waiting", None)
        if hook is not None:
            hook(on, **record)

    def waited(self) -> None:
        hook = getattr(self._observer, "waited", None)
        if hook is not None:
            hook()

    def restarting(self, reason: str) -> None:
        hook = getattr(self._observer, "restarting", None)
        if hook is not None:
            hook(reason)

    def source_behind(self, details) -> None:
        hook = getattr(self._observer, "source_behind", None)
        if hook is not None:
            hook(details)

    @property
    def source_behind_record(self):
        return getattr(self._observer, "source_behind_record", None)

    @property
    def events(self):
        # The hosted forecast says its seam waits on the run's stream.
        return getattr(self._observer, "events", None)

    def finalizing(self, phase: str, *, work_bytes: int | None = None) -> None:
        hook = getattr(self._observer, "finalizing", None)
        if hook is not None:
            hook(phase, work_bytes=work_bytes)

    def frame_submitted(self, **fields) -> None:
        self._observer.frame_submitted(**fields)

    def output_committed(self, **fields) -> None:
        self._observer.output_committed(**fields)

    # -- time to first plot, forwarded verbatim -----------------------

    @property
    def first_products(self):
        return self._observer.first_products

    @property
    def live_products(self):
        return self._observer.live_products

    def arm_first_products(self, render_plan) -> None:
        self._observer.arm_first_products(render_plan)

    def starting(self) -> None:
        self._observer.starting()

    def complete(self, model_elapsed_seconds: float) -> None:
        self._observer.complete(model_elapsed_seconds)

    def failed(self) -> None:
        self._observer.failed()


def _pinned_fetch_hints(plan: RunPlan, hints: Mapping[str, Any], *,
                        pin_cycle: bool = True) -> dict[str, Any]:
    """The config's ``[fetch]`` hints with ``run_options.transport`` over them.

    The run option is ``woof go --transport`` and wins over the table the
    way the flag does.  A host the config's source cannot pin is refused
    here, in the fetch's own words, before the fetch stage starts.
    ``auto`` in either pins nothing, so the hints then carry no
    ``transport`` at all and key the unpinned request's download.
    """

    from woof.go_cli import pin_request, pinned_transport

    merged = dict(hints)
    try:
        pinned, _basis = pinned_transport(merged, plan.run_options.get("transport"))
    except ValueError as error:
        raise PlanError(str(error)) from error
    merged = pin_request(merged, pinned)
    # The posting rule and budget the run options name win over the
    # table the way --transport does (`woof go --whole-cycle`,
    # `--late-after-minutes`), and so does the cycle (`--cycle`): a plan
    # executed with one runs a config re-timed to it (:func:`plan_at_cycle`),
    # and a query mode asks about the same window without writing one.
    for key in ("as_posted", "late_after_minutes", "cycle"):
        if key == "cycle" and not pin_cycle:
            continue
        value = plan.run_options.get(key)
        if value is not None:
            merged[key] = value
    if merged.get("as_posted") is False:
        merged.pop("late_after_minutes", None)
    return merged


def _fetch_arguments_from_hints(hints: Mapping[str, Any],
                                *, out: Path) -> list[str]:
    """The ``woof fetch`` argv one ``[fetch]`` hints table spells.

    The wizard's own words for that table are "keys mirror `woof fetch`
    flags", so the mapping is mechanical: ``forecast_start_hour``
    becomes ``--forecast-start-hour``.  Nothing is filtered by a list
    kept here -- the argv is handed to woof's real fetch parser, which
    accepts exactly what `woof fetch` accepts and refuses the rest.
    """

    from woof.initial_source import boundary_fetch_request
    hints = boundary_fetch_request(hints)
    arguments: list[str] = []
    for key in sorted(hints):
        if key == "out":
            continue          # run-plan owns where the data lands
        value = hints[key]
        if key == "as_posted":
            # The one boolean key whose false is a flag of its own: the
            # fetch is as posted by default, so false must be said.
            arguments.append("--as-posted" if value else "--whole-cycle")
            continue
        if isinstance(value, bool):
            if value:
                arguments.append("--" + key.replace("_", "-"))
            continue
        arguments += ["--" + key.replace("_", "-"), str(value)]
    from woof import fetch_endpoints
    from woof.fetch import GFS_CONTAINER_SOURCES
    if (fetch_endpoints.policy_uses_aws(str(hints.get("source", "")))
            and hints.get("source") in GFS_CONTAINER_SOURCES):
        from woof.fetch import GFS_RECORDS_MODE
        arguments += ["--mode", GFS_RECORDS_MODE]
    return arguments + ["--out", str(out)]


def _config_for_declared_fetch(plan: RunPlan, resolution: Mapping[str, Any],
                               run_dir: Path) -> dict[str, Any]:
    """The configuration whose [fetch] recipe acquires this run's declared forcing.

    An intent plan's configuration is the one the wizard generated into
    the run directory, not bytes the plan carries.  Reading only a
    plan's own bytes skipped every intent: an ERA5 intent on the
    config-driven route was accepted, wrote a [fetch] recipe naming its
    forcing, and then refused at the input gate for the file that recipe
    was never asked to download.  A relative [fetch].out is relative to
    the configuration's folder, as [case_data] is (D-11), so it is
    anchored there and returned absolute: the generated config of an
    intent lives in the run directory, a named config in its own folder.
    The working directory the run was started from does not decide where
    the download lands.
    """

    if plan.config_intent is None:
        document = tomllib.loads(plan.config_bytes().decode("utf-8"))
        base_dir = (plan.config_path.parent if plan.config_path is not None
                    else plan.config_base_dir)
    else:
        document = tomllib.loads(str(resolution.get("generated_config") or ""))
        base_dir = run_dir
    hints = document.get("fetch")
    if isinstance(hints, dict) and hints.get("out"):
        from woof.case_data import config_fetch_out
        hints["out"] = str(config_fetch_out(document, base_dir=Path(base_dir)))
    return document


def declared_forcing_fetch(payload: Mapping[str, Any], data, *,
                           base_dir: Path | None,
                           cache_root: Path | None = None) -> list[str] | None:
    """Acquire missing declared forcing through the config's own fetch recipe.

    ``base_dir`` is the configuration's folder, which a relative
    ``[fetch].out`` is relative to (D-11).  It has no default, so no
    caller can fall back to the working directory by omission; ``None``
    is for a payload whose out the caller has already made absolute, and
    a relative out with ``None`` is refused rather than guessed.

    ``cache_root`` is the run's output root.  When the declared forcing
    is not on disk, a run that has one downloads it into that root's
    managed cache (``<root>/downloads/<request key>``, the folder the GFS
    and HRRR routes use) rather than into ``[fetch].out`` beside the
    configuration, and the run binds its forcing there
    (:func:`rebind_fetched_forcing`).  The breakage it prevents: an ERA5
    ``woof go CONFIG --outdir DIR`` wrote its 34 MB download into the
    configuration's folder and nothing under DIR, so a read-only or shared
    configuration folder failed or was written into.  Forcing already on
    disk at the declared path is used where it is.
    """
    if data is None:
        return None
    hints = payload.get("fetch")
    # Existing EDA bytes still need request/digest/native-member validation.
    # The acquisition reuses an exact verified receipt without a network call.
    eda = isinstance(hints, dict) and hints.get("source") == "era5" and hints.get("era5_product") == "ensemble_members"
    if all(path.is_file() for path in data.forcing) and not eda:
        return None
    if not isinstance(hints, dict) or hints.get("source") != "era5":
        return None
    from woof.fetch import (ERA5_COMBINED_NAMES, era5_combined_name,
                             era5_forcing_name_disagreement)
    provider = hints.get("era5_provider", "cds")
    if provider not in ERA5_COMBINED_NAMES:
        raise PlanError("[fetch].era5_provider must be "
                        + " or ".join(repr(name) for name in sorted(ERA5_COMBINED_NAMES)) + ".")
    if not hints.get("out"):
        raise PlanError("ERA5 data is missing. Set [fetch].out to the directory containing the declared forcing file.")
    # A relative fetch.out is relative to the configuration's folder, as
    # the forcing the case-data loader resolved is (D-11).
    from woof.case_data import resolve_fetch_out
    if base_dir is None and not Path(str(hints["out"])).expanduser().is_absolute():
        raise PlanError("[fetch].out is relative, and this caller did not say which "
                        "configuration it belongs to; resolve it against the "
                        "configuration's folder before asking for the fetch.")
    out = resolve_fetch_out(hints["out"], base_dir=Path(base_dir or "."),
                            forcing=data.forcing)
    expected = (out / era5_combined_name(provider)).resolve()
    from woof.case_data import same_case_data_path
    if len(data.forcing) != 1 or not same_case_data_path(data.forcing[0], expected):
        # NAME THE FILE.  "Keep both paths on the same ERA5 combined
        # file" told a reader that two declarations disagreed and left
        # them to work out which one was wrong -- and the answer is not
        # symmetric: the provider decides what the fetch publishes, so
        # the forcing declaration is the side that follows.  The wrong
        # name is what a config written for the ARCO provider used to
        # carry, and this refusal was the whole of what a user saw.
        detail = era5_forcing_name_disagreement(data.forcing, provider=provider)
        raise PlanError(
            "ERA5 data is missing, and [fetch].out does not produce the "
            "file named by [case_data].forcing. "
            + (detail if detail is not None else
               f"[fetch].out publishes {str(expected)!r}; [case_data].forcing "
               f"names {[str(path) for path in data.forcing]}.")
            + " Re-author the configuration with `woof domain` to bind both "
              "to the same file.")
    if cache_root is not None and not all(path.is_file() for path in data.forcing):
        from woof.go_cli import GoRefusal, managed_download_dir
        try:
            out = managed_download_dir(Path(cache_root), dict(hints)).resolve()
        except GoRefusal as error:
            raise PlanError(str(error)) from None
    arguments = _fetch_arguments_from_hints(hints, out=out)
    from woof.cli import _join_negative_coordinates
    arguments = _join_negative_coordinates(arguments)
    if "--retrieve" not in arguments:
        arguments.append("--retrieve")
    _validate_fetch_arguments(arguments)
    return arguments


def rebind_fetched_forcing(data, fetch_arguments: Sequence[str] | None):
    """``data`` with missing declared forcing bound to where this run's fetch puts it.

    :func:`declared_forcing_fetch` downloads missing forcing into the run's
    own cache (``--out`` of ``fetch_arguments``), under the same file name
    the configuration declares.  Returns ``(data, resolution)``; the
    resolution is ``None`` when nothing was rebound (forcing on disk where
    declared, several forcing files, or a fetch publishing to the declared
    folder itself).
    """

    if data is None or not fetch_arguments or len(data.forcing) != 1:
        return data, None
    declared = Path(data.forcing[0])
    arguments = [str(value) for value in fetch_arguments]
    if declared.is_file() or "--out" not in arguments:
        return data, None
    index = arguments.index("--out")
    if index + 1 >= len(arguments):
        return data, None
    target = (Path(arguments[index + 1]) / declared.name).resolve()
    from woof.case_data import same_case_data_path
    if same_case_data_path(declared, target):
        return data, None
    return dataclasses.replace(data, forcing=(target,)), {
        "scope": "case_data", "key": "forcing", "value": [str(target)],
        "basis": "run output root download cache",
        "note": f"the declared forcing {str(declared)!r} is not on disk, so this "
                "run downloads it under its own output root and reads it there"}


def _prepare_stage(root: Path, *, arguments: Sequence[Any],
                   stated: Mapping[str, Any],
                   run, built=None) -> dict[str, Any]:
    """Run a preparation, or reuse the one already at ``root``.

    THE recovery seam.  A plan that failed at the forecast stage leaves
    a fetched data directory and a finished prepared bundle behind it;
    re-running the same plan into the same ``output_root`` used to die
    at the preparer's own create-only refusal ("refusing existing
    output root"), which is a correct refusal in the preparer's terms
    and a dead end in the user's -- their only way back to a working
    forecast was a fresh run directory and the gigabytes of download
    that come with it.

    So the decision is taken here, before the preparer is called, from
    the artifacts the previous preparation published:
    :func:`woof.stage_reuse.decide` compares this run's arguments, its
    stated input digests and this engine's own source identity against
    what the bundle records.  A bundle that still describes this run is
    reused; one that does not is moved aside -- never deleted -- and
    rebuilt from the forcing already on disk, which costs the tens of
    seconds a preparation costs rather than the minutes a re-fetch does.

    The returned receipt is what the stage event carries, so which of
    the two happened, and why, is on the record rather than inferred
    from a timing.

    ``built``, when given, is called with that receipt the moment
    ``run`` returns, which is when the preparer has sealed its bundle:
    before the reuse binding, which reads this engine's identity through
    git and digests the stage's arguments, so a chain says when its seal
    landed rather than when this stage's bookkeeping ended.
    """

    from woof import stage_reuse

    decision = dict(stage_reuse.decide(
        root, stated=stated, arguments=arguments))
    if decision["decision"] == stage_reuse.REUSE:
        return decision
    if decision["decision"] == stage_reuse.REBUILD:
        decision["superseded"] = stage_reuse.supersede(root)
    run()
    if built is not None:
        built(dict(decision))
    stage_reuse.write_binding(root, arguments=arguments, stated=stated)
    return decision


class _ChainSeal:
    """``prepare_sealed``, said once, when a chained preparation's seal lands.

    It needs both of two things, in whichever order they happen: the
    forecast bound to the head (so the stream says ``prepare_head_ready``
    first), and the preparer returned with its bundle sealed.  The seal
    is reported by :func:`_prepare_stage` through ``built``, the moment
    the preparer returns.  The breakage this prevents: reported when the
    whole preparation call returned, the seal waited for the stage's
    reuse binding (git and argument digests), and when that outlasted
    the forecast, the event said the seal came at the forecast's end.
    """

    def __init__(self, observer: RunObserver, prep_root: Path):
        self._observer = observer
        self._prep_root = prep_root
        self._lock = threading.Lock()
        self._head = False
        self._prepared: dict[str, Any] | None = None
        self._emitted = False

    def head_bound(self) -> None:
        with self._lock:
            self._head = True
            self._emit()

    def sealed(self, prepared: Mapping[str, Any]) -> None:
        # The first report stands: a preparation's own return after
        # ``built`` said it adds nothing but the later timing.
        with self._lock:
            if self._prepared is None:
                self._prepared = dict(prepared)
            self._emit()

    def _emit(self) -> None:
        if self._emitted or not self._head or self._prepared is None:
            return
        self._emitted = True
        events = getattr(self._observer, "events", None)
        if events is not None:
            events.emit("prepare_sealed", prepared_root=str(self._prep_root),
                        prepared=self._prepared)


def _clear_forecast_output(forecast_dir: Path, *,
                           observer: RunObserver) -> Path:
    """Reserve a forecast path without moving any earlier output address.

    A retry owns another generation containing its forecast and pictures.
    Prior receipts and their absolute frame paths remain together. The
    caller must use the returned path for execution, rendering and summary.
    """
    forecast_dir = Path(forecast_dir)
    forecast_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        forecast_dir.mkdir()
        return forecast_dir
    except FileExistsError:
        pass
    sequence = 1
    while True:
        generation = forecast_dir.with_name(
            f"{forecast_dir.name}-attempt-{sequence:03d}")
        try:
            generation.mkdir()
            break
        except FileExistsError:
            sequence += 1
    output = generation / forecast_dir.name
    output.mkdir()
    try:
        observer.events.emit(
            "warning", code="forecast_output_recovery",
            message=(f"Earlier output remains at {forecast_dir}. "
                     f"This attempt writes to {output}; prior receipt "
                     "addresses and pictures are preserved."),
            previous_path=str(forecast_dir), path=str(output))
    except Exception:            # noqa: BLE001 - telemetry never fails a run
        pass
    return output


def _validate_prepared_output(plan: RunPlan, *, require_empty: bool = False,
                              launch_files: tuple[Path, ...] = ()) -> None:
    """An explicit bundle launch owns new output, never an input's directory."""
    prepared = plan.run_options.get("prepared_root")
    if prepared is None:
        return
    output = plan.run_dir.resolve()
    protected = [Path(prepared).resolve()]
    protected.extend(Path(path).resolve() for path in
                     (plan.config_path, plan.run_options.get("wps_namelist"))
                     if path is not None)
    restart = plan.run_options.get("restart")
    if restart is not None:
        # The siblings form one checkpoint set. Preserve its containing run.
        protected.append(Path(restart).resolve().parent)
    for path in protected:
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise PlanError(
                f"Output {output} overlaps a prepared/config/WPS/checkpoint input {path}. "
                "Choose a fresh sibling output directory; the existing run is kept.")
    if require_empty and output.exists():
        allowed = {path.resolve() for path in launch_files}
        held = [path for path in output.iterdir()
                if not (path.is_file() and not path.is_symlink() and path.resolve() in allowed)]
        if held:
            raise PlanError(f"Output {output} already contains an earlier run or files: "
                            + ", ".join(path.name for path in held)
                            + ". Choose a fresh output directory for this prepared launch.")


def _existing_prepared_bundle(plan: RunPlan) -> dict | None:
    """Identify the declared format; the runner still verifies all payloads."""
    root = plan.run_options.get("prepared_root")
    if root is None:
        if plan.route == "prepared" and plan.run_options.get("restart"):
            raise PlanError("Prepared restart requires the existing prepared_root; "
                            "a fresh preparation cannot replace checkpoint identity")
        return None
    from woof import stage_cli

    bundle = stage_cli.resolve_bundle(Path(root))
    stage_cli._restart_flags(bundle["layout"], plan.run_options.get("restart"), False)
    stage_cli._health_debug_flags(bundle["layout"], bool(plan.run_options.get("health_debug")))
    if bundle["layout"] == "single" and not plan.run_options.get("wps_namelist"):
        raise PlanError("A single prepared bundle needs run_options.wps_namelist "
                        "(woof go --wps-namelist PATH), the exact WPS file "
                        "its preparation bound. No replacement is generated.")
    return bundle


def _prepared_launch_tiles(plan: RunPlan, exp, bundle: Mapping[str, Any]):
    """The [tiles] an existing single-domain bundle's forecast runs under.

    In order: ``run_options.forecast_tiles`` (``woof go --tiles``), the launch
    config's own enabled [tiles], then the table a ``--prepare-only``
    launch recorded beside the bundle (``woof.go_cli.LAUNCH_TILES_NAME``).
    None of them enters the bundle's identity: the runner takes the mode
    as its own ``--tiles``.  Breakage this closes: the two-step form
    (``--prepare-only``, then ``--prepared-root``) of a domain the card
    cannot hold resident was refused at the forecast with "[tiles] mode =
    'auto'" as its remedy, and nothing on that launch could say it (a full
    1800x1060 HRRR grid at 107 GiB on a 96 GiB card).  A tree bundle is
    refused by name if a mode is asked of it (``stage_cli.streaming_flags``):
    its [tiles] lives in its experiment table.
    """

    from woof.core.streaming import StreamingOptions

    mode = plan.run_options.get("forecast_tiles")
    if mode is not None:
        return StreamingOptions.from_mapping({"mode": mode}, source="woof go --tiles")
    if bundle.get("layout") != "single":
        return None
    declared = getattr(exp, "tiles", None)
    if declared is not None and declared.enabled:
        return declared
    from woof.go_cli import LAUNCH_TILES_NAME

    recorded = Path(bundle["document"]).parent / LAUNCH_TILES_NAME
    if not recorded.is_file():
        return None
    try:
        document = json.loads(recorded.read_text(encoding="utf-8"))
        options = StreamingOptions.from_mapping(document["tiles"], source=str(recorded))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PlanError(f"{recorded} is not a launch [tiles] record: {error}") from error
    print(f"go: [tiles] mode = '{options.mode}' from {recorded} (the "
          "--prepare-only launch's config); --tiles overrides it")
    return options if options.enabled else None


def _existing_prepared_forecast(plan: RunPlan, *, config_path: Path,
                                observer: RunObserver, run_dir: Path,
                                exp=None) -> Mapping[str, Any]:
    """Run the existing sim operation without a fetch or preparation retry."""
    from woof import stage_cli

    _validate_prepared_output(plan)
    bundle = _existing_prepared_bundle(plan)
    assert bundle is not None
    tiles = _prepared_launch_tiles(plan, exp, bundle)
    restart = plan.run_options.get("restart")
    wps = plan.run_options.get("wps_namelist")
    forecast_dir = run_dir / "chain" / "run"
    command = stage_cli.sim_command(
        bundle, experiment_config=config_path,
        wps_namelist=None if wps is None else Path(wps),
        outdir=forecast_dir, physics_profile=plan.run_options.get("physics_profile"),
        progress_format="jsonl", restart=None if restart is None else Path(restart),
        **({"devices": plan.run_options["devices"]}
           if plan.run_options.get("devices") is not None else {}),
        health_debug=bool(plan.run_options.get("health_debug")),
        tiles=tiles)
    observer.arm_first_products(_chain_render_plan(
        plan, forecast_dir=forecast_dir, run_dir=run_dir))
    observer.enter_stage("forecast", phase="restore" if restart else "forecast")
    # No retry/supersede: the runner claims a new output directory itself.
    _staged_forecast(command[3:], layout=bundle["layout"], observer=observer)
    result = dict(_chain_render(plan, forecast_dir=forecast_dir,
                                run_dir=run_dir, observer=observer))
    result.update(prepared_root=str(bundle["document"].parent),
                  source=bundle["source"])
    return result


def _asserted_profile(plan: RunPlan, *, config_path) -> str | None:
    """The suite a chain's own stages are told the config IS, or None.

    An explicit ``run_options.physics_profile`` is always asserted.  The
    intent's suite is asserted only where the configuration the wizard
    wrote from it IS that suite on every domain, which the preparer's own
    conflict predicate answers, the one ``woof go`` derives its
    assertion with.  The wizard departs from a named suite on purpose: a
    tree's nests turn cumulus off below the gray zone and take the
    ``diff_6th_factor`` depth ladder, and a root whose cumulus the intent
    left to the grid (``cumulus = "grid"``) turns it off there.  Such a
    config runs as its own switches, which are the suite on the root and
    the wizard's rules on the nests.  Asserting the suite instead made the
    domain-tree forecast refuse every named suite on a tree after the
    download and the preparation (``cu_physics selected 0 expected 1`` on
    d02).  An intent carrying ``physics_choices`` is never asserted: the
    wizard wrote the picked schemes over the suite's own, so the config is
    that mix, not the suite.
    """

    explicit = plan.run_options.get("physics_profile")
    if explicit:
        return str(explicit)
    intent = plan.config_intent or {}
    profile = intent.get("physics_profile")
    if not profile or intent.get("physics_choices"):
        return None
    from woof.prepared_single_domain_forecast import named_profile_config_conflicts

    source = _canonical_source_id(str(intent.get("source", "era5")))
    if named_profile_config_conflicts(Path(config_path).read_text(encoding="utf-8"),
                                      source=source, profile=str(profile)):
        return None
    return str(profile)


def _hrrr_chain(plan: RunPlan, *, config_path: Path, exp,
                observer: RunObserver, run_dir: Path,
                prepare_only: bool = False) -> Mapping[str, Any]:
    """The documented HRRR chain: fetch, prepare, forecast.

    This is the native HRRR chain used by both run-plan and go.
    Preparation uses ``tools/prepare_hrrr_wrf``; the HRRR tools
    read the four namelist/JSON files the wizard writes beside the
    config rather than the TOML itself.  So the stages and their order
    here are the wizard's own printed chain
    (:func:`woof.domain_wizard.hrrr_route_commands`), driven rather
    than printed, with each stage's refusals left entirely alone.

    ONE thing is added to that chain: ``--wps-namelist``.  The runner's
    HRRR manifest role inventory requires a ``wps_namelist`` role
    (prepared_single_domain_forecast.py:2181 -- the prepared-cache
    identity's ``namelist_sha256`` IS that file's digest on this
    route), and the preparer only records the role if it is handed the
    file.  The printed chain never passed it, so the bundle it produced
    could not be read by the single-domain runner at all, and HRRR was
    sent to a benchmark script instead.  Passing it makes a
    single-domain HRRR bundle structurally identical to a GFS one at
    the run step -- same runner, same digests, same observer.
    """

    from woof.fetch import sha256_file
    from woof.go_cli import managed_download_dir, proof_digests, run_stage
    from woof.hrrr_route_inputs import (HrrrRouteInputError, route_input_paths,
                                         run_route_inputs)

    raw = tomllib.load(io.BytesIO(config_path.read_bytes()))
    # The namelists this route runs from: the complete set a door wrote
    # beside the configuration, or, where there is none, the set this run
    # writes from the configuration into its own folder.  The same
    # function answered resolve_plan's question before anything was
    # fetched, so a configuration refused here was refused at the door.
    route_report: dict[str, Any] = {}
    try:
        inputs = run_route_inputs(config_path, exp, raw=raw,
                                  into=run_dir / "chain" / "route-inputs",
                                  report=route_report)
    except HrrrRouteInputError as refusal:
        raise PlanError(str(refusal)) from None
    beside = route_input_paths(config_path)
    disagreeing = route_report.get("disagreeing") or ()
    if disagreeing:
        # A set written before the configuration last changed: the root
        # preparation reads the configuration, so the hierarchy would have
        # refused this set after the fetch (WOOF canary run_0771cd8814b6e7f8,
        # "[shared] gsd_diagnostics = 1" added after `woof domain`).
        observer.warn(
            "route_inputs_rendered",
            f"the namelists beside {config_path.name} were written before "
            f"it last changed: their d01 holds {', '.join(disagreeing)} "
            f"differently from {config_path.name}, which the root "
            "preparation reads, so the run wrote the whole set from the "
            f"configuration into {inputs['namelist_input'].parent}. Next: "
            "to keep a set in step, re-save the configuration through the "
            "door that wrote it",
            disagreeing=list(disagreeing),
            written=str(inputs["namelist_input"].parent))
    # A partial set is not read: a namelist.input edited by hand beside a
    # configuration whose other companions are missing does not run, and
    # the run says so rather than leaving the edit to look applied.  The
    # WPS namelist is the exception, because the rendered set keeps it.
    unread = [path.name for role, path in beside.items()
              if role != "wps_namelist" and path.is_file()
              and inputs[role] != path]
    if unread and not disagreeing:
        missing = [path.name for path in beside.values() if not path.is_file()]
        observer.warn(
            "route_inputs_rendered",
            f"{', '.join(unread)} beside {config_path.name} "
            f"{'is' if len(unread) == 1 else 'are'} not what this run "
            f"reads: the set there is incomplete ({', '.join(missing)} "
            "missing), so the run wrote the whole set from the "
            f"configuration into {inputs['namelist_input'].parent}. Next: "
            "to run edited namelists, put the whole set beside the "
            "configuration, as the door that saved it writes it",
            unread=unread, missing=missing,
            written=str(inputs["namelist_input"].parent))
    hints = _pinned_fetch_hints(plan, raw.get("fetch") or {})
    intent = plan.config_intent or {}
    data_dir = Path(plan.run_options.get("data_dir")
                    or intent.get("data_dir") or managed_download_dir(run_dir, hints))
    geog_root = plan.run_options.get("geog_root") or intent.get("geog_root")
    if geog_root is None:
        from woof.geog_assets import geog_root_from_tables

        geog_root = geog_root_from_tables(raw, base_dir=config_path.parent)
    _stage_static_sources(raw, geog_root, base_dir=config_path.parent)
    # Not created here, and deliberately so: the preparer refuses an
    # --output-root that already exists ("refusing existing output
    # root"), which is its own create-only guarantee.  A re-run into the
    # same output_root meets that refusal, so this chain decides BEFORE
    # calling it whether the bundle already there is this run's -- see
    # _prepare_stage.
    prep_root = run_dir / "chain" / "hrrr-root-prep"
    forecast_dir = run_dir / "chain" / "run"

    # -- fetch ---------------------------------------------------------
    # AS POSTED (A136 L7c (b)): the native route's fetch runs beside
    # its preparation, which starts once the fetch has scheduled the window,
    # decodes each lead as its marker appears, publishes its head on the
    # first two leads and writes the source manifest at its seal.  The
    # fetch beside is ``None`` otherwise, and the window is fetched first.
    from woof.preparation_assets import (analyzed_aerosol_fetch_hints,
                                          wif_fetch_domains)

    if wif_fetch_domains(exp, hints):
        hints = {**hints, "wif": True}
    hints = analyzed_aerosol_fetch_hints(exp, hints)
    beside = _native_fetch_beside(plan, hints, exp, data_dir=data_dir,
                                  run_dir=run_dir, observer=observer,
                                  prepare_only=prepare_only)
    posting = None if beside is None else beside.posting
    if posting is None:
        if beside is None:
            observer.enter_stage("fetch", phase="fetch")
            fetch_report = _run_fetch(
                _fetch_arguments_from_hints(hints, out=data_dir), run_dir,
                events=getattr(observer, "events", None), posting_relay=True)
        else:
            # The fetch scheduled nothing (or ended first): its window is
            # here whole, and the preparation binds it as before.
            try:
                fetch_report = beside.result()
            finally:
                beside.close()
        manifest = data_dir / "SHA256SUMS"
        if not manifest.is_file():
            raise PlanError(
                f"the HRRR fetch wrote no {manifest}, so the preparation "
                "stage has nothing to bind against")
        from woof.launch_supplements import hrrr_source_manifest
        manifest = hrrr_source_manifest(
            plan.run_options.get("supplement", ()), source_root=data_dir,
            fetched_manifest=manifest, output=run_dir / "chain" / "hrrr-source-SHA256SUMS")
        observer.finish_stage(fetch=fetch_report)
    else:
        manifest = None
        observer.finish_stage(fetch={"arguments": beside.arguments,
                                     "as_posted": True,
                                     "posting": str(posting)})
    # The chem sources (HRRR-Smoke's analysed and boundary smoke, the fire
    # emissions) before preparation, as the staged chain does: preparation
    # never downloads, so the packaged HRRR recipe on this chain stopped at
    # its chem initial state with ChemSourceUnavailable 'hrrr-native-smoke'
    # (a rented box, 2026-10-21 06Z).
    _staged_chem_fetch(exp, observer)

    # -- prepare -------------------------------------------------------
    observer.enter_stage("prepare", phase="prepare")
    # The two stages spell the same instant differently: [fetch] carries
    # YYYY-MM-DDTHH (what `woof fetch` takes) and the preparer takes
    # YYYY-MM-DD_HH:MM:SS (what the wizard's printed chain passes it).
    # Converted through the real parser rather than by slicing the
    # string, so a cadence rule the parser enforces is enforced here.
    from woof.fetch import parse_cycle

    cycle = parse_cycle(
        str(hints["cycle"]), "hrrr").strftime("%Y-%m-%d_%H:%M:%S")
    # Preparation overrides the document's full-frame cadence, not the
    # shorter alarm used to schedule its separate quick-look stream.
    cadence = int(exp.root.full_history_interval_s or exp.root.history_interval_s)
    prepare = [
        sys.executable, "-m", "tools.prepare_hrrr_wrf",
        "--source-root", str(data_dir),
        # As posted the seal writes the source manifest from the leads'
        # markers; otherwise the preparation binds the fetch's.
        *(("--as-posted", str(posting)) if manifest is None else (
            "--source-manifest", str(manifest),
            "--source-manifest-sha256", sha256_file(manifest))),
        "--experiment-config", str(config_path),
        "--domain-spec", str(inputs["target_domain"]),
        "--namelist-input", str(inputs["namelist_input"]),
        # THE addition.  Without it the emitted bundle has no
        # wps_namelist role and the runner refuses it outright.
        "--wps-namelist", str(inputs["wps_namelist"]),
        "--geog-root", str(geog_root),
        "--cycle", cycle,
        "--run-seconds", str(int(exp.run_seconds)),
        "--history-interval-seconds", str(cadence),
        "--skip-stock-wrf-export",
        "--output-root", str(prep_root),
    ]
    prepare += [token for binding in plan.run_options.get("supplement", ())
                for token in ("--supplement", binding)]
    profile = _asserted_profile(plan, config_path=config_path)
    if profile:
        # Passed only when the plan states it.  The route owns its own
        # physics gate and the emitted TOML records physics as numbers
        # rather than a profile id, so there is nothing to recover from
        # a config.path plan -- and a default invented at this layer
        # would silently outrank the preparer's own.  Where the two
        # disagree the runner's identity check refuses, loudly, which is
        # the gate doing its job.
        prepare += ["--physics-profile", str(profile)]
    lead = hints.get("forecast_start_hour")
    if lead:
        prepare += ["--forecast-start-hour", str(int(lead))]
    def preparation(built=None):
        return _prepare_stage(
            prep_root, arguments=prepare,
            # As posted there is no source manifest before the seal; the
            # arguments name the posting folder of this run's fetch.
            stated={**({} if manifest is None else
                       {"source_manifest_sha256": sha256_file(manifest)}),
                    "namelist_sha256": sha256_file(inputs["namelist_input"])},
            run=lambda: run_stage("prepare", prepare, explain=False,
                                  progress=prep_root / "progress.json",
                                  observer=_GoObserver(observer)),
            built=built)

    if not prepare_only:
        forecast_dir = _clear_forecast_output(forecast_dir, observer=observer)
    # Armed before either forecast arm, with the very dict `_chain_render`
    # will hand the finalize stage below, so the frame rendered as it
    # lands and the frames rendered at the end agree on both the output
    # directory and the product spec by construction.
    if not prepare_only:
        observer.arm_first_products(
            _chain_render_plan(plan, forecast_dir=forecast_dir, run_dir=run_dir))

    def hierarchy(*, observe_stage=True):
        # A nested HRRR run takes a THIRD stage between the root
        # preparation and the forecast, and then a different runner.
        from woof.static.corridor import config_declares_follow_source

        return _hrrr_hierarchy_stage(
            prep_root=prep_root, inputs=inputs, hints=hints,
            geog_root=geog_root, manifest=manifest, cycle=cycle,
            run_dir=run_dir, observer=observer,
            # THE predicate, not a copy of it: the same function
            # `follow_statics_decision` consulted at resolve time, so a
            # plan reported as corridor-bearing seals one and a plan
            # reported as still does not.
            statics_corridor=config_declares_follow_source(exp),
            acknowledgements=tuple(exp.acknowledgements),
            lifecycle_config=(config_path if any(
                getattr(dc, name, None) is not None for dc in exp.domains
                for name in ("spawn", "follow", "retire", "rearm")) else None),
            observe_stage=observe_stage)

    if len(exp.domains) == 1 and not prepare_only:
        # CHAINED PREPARATION (A136 L7c): the native preparation publishes
        # the bundle's head once its start state exists and one boundary
        # interval per hour after it, so the forecast starts on the head
        # while the later hours are built.  ``None`` is a bundle published
        # at its seal or reused whole, which runs below exactly as before.
        # As posted, the fetch beside it is the chain's: a stage that fails
        # while it runs stops it, and its own failure is the chain's when
        # it came first (a lead later than its budget is exit 75).
        with _beside_settled(beside):
            prepared, chained = _hrrr_single_chain(
                prep_root=prep_root, preparation=preparation,
                forecast_dir=forecast_dir, exp=exp, observer=observer)
            if beside is not None:
                beside.result()
        if chained is not None:
            return _chain_render(plan, forecast_dir=forecast_dir,
                                 run_dir=run_dir, observer=observer)
    elif len(exp.domains) > 1 and (not prepare_only or os.environ.get("WOOF_CONTINUATION_PREFIX")):
        # A native tree chains on its root preparation's head (A136 L7c):
        # the hierarchy stage builds the children on the root's start state
        # and publishes the tree's head, relays the root's boundary
        # intervals as they are written, and seals once the root seals.
        # ``None`` is a tree published at its seal or reused whole, which
        # runs below as before.
        tree_root = run_dir / "chain" / "hrrr-hierarchy"
        with _beside_settled(beside):
            prepared, chained = _hrrr_tree_chain(
                prep_root=prep_root, tree_root=tree_root,
                preparation=preparation,
                hierarchy=lambda: hierarchy(observe_stage=False),
                config_path=config_path, forecast_dir=forecast_dir,
                observer=observer, devices=plan.run_options.get("devices"),
                prepare_only=prepare_only)
            if beside is not None:
                beside.result()
        if chained is not None:
            if prepare_only:
                return _prepared_chain_result(tree_root, config_path, None)
            return _chain_render(plan, forecast_dir=forecast_dir,
                                 run_dir=run_dir, observer=observer)
        observer.finish_stage(hierarchy_root=str(tree_root),
                              prepared=prepared)
        _hrrr_tree_forecast(
            tree_root=tree_root, config_path=config_path,
            forecast_dir=forecast_dir, observer=observer,
            devices=plan.run_options.get("devices"))
        return _chain_render(plan, forecast_dir=forecast_dir,
                            run_dir=run_dir, observer=observer)
    else:
        prepared = preparation()

    if len(exp.domains) > 1:
        tree_root = hierarchy()
        if prepare_only:
            return _prepared_chain_result(tree_root, config_path, None)
        _hrrr_tree_forecast(
            tree_root=tree_root, config_path=config_path,
            forecast_dir=forecast_dir, observer=observer,
            devices=plan.run_options.get("devices"))
        return _chain_render(plan, forecast_dir=forecast_dir,
                            run_dir=run_dir, observer=observer)

    # The preparer publishes its own handoff -- prepared_root, the three
    # digests, and the PUBLISHED authority paths -- into
    # public-wrapper-result.json.  Read it rather than re-deriving any
    # of it: that is the same relay-from-the-artifact rule `go` follows,
    # and every part of it is a value this layer must not compute.
    #
    # Three things here are not guessable and were each wrong first
    # time:
    #   * proof.json lives at the OUTPUT ROOT, not inside the prepared
    #     cache -- the bundle root IS --output-root;
    #   * --prepared-root is therefore that root too, because the
    #     runner's HRRR_BUNDLE_PATHS are relative to it;
    #   * --experiment-config / --wps-namelist must be the PUBLISHED
    #     copies (experiment.toml, namelist.wps) and not the wizard's
    #     originals, because the runner checks each supplied file's
    #     NAME and digest against the portable source manifest.
    wrapper = _read_json_object(prep_root / "public-wrapper-result.json")
    handoff = wrapper.get("portable_bundle")
    if not isinstance(handoff, dict):
        raise PlanError(layered(
            f"the HRRR preparation published no portable bundle in "
            f"{prep_root / 'public-wrapper-result.json'}.",
            "That bundle -- proof.json, the role-keyed source manifest "
            "and the experiment authority -- is what the forecast stage "
            "binds, and the preparer only publishes it when handed "
            "--wps-namelist, which this chain does pass.  Its own "
            "output is above."))
    prepared_root = Path(handoff["prepared_root"])
    # Cross-check the relayed digests against the proof on disk, using
    # go's own reader.  Cheap, and it catches a handoff that does not
    # describe the artifacts beside it.
    digests = proof_digests(prepared_root)
    for key, relayed in (("proof", "proof_sha256"),
                         ("source_manifest", "source_manifest_sha256"),
                         ("prepared_content", "prepared_content_sha256")):
        if handoff.get(relayed) != digests[key]:
            raise PlanError(
                f"the HRRR preparation's published {relayed} does not "
                f"match {prepared_root / 'proof.json'}; the bundle and "
                "the proof beside it disagree")
    observer.finish_stage(prepared_root=str(prepared_root),
                          bundle=str(prep_root /
                                     "public-wrapper-result.json"),
                          prepared=prepared)

    if prepare_only:
        return _prepared_chain_result(prepared_root, Path(handoff['experiment_config']),
                                      Path(handoff['wps_namelist']))

    # -- forecast ------------------------------------------------------
    # In process, so the runner's per-step progress and its per-wrfout
    # landing hook reach the observer.  Same runner and same flags the
    # GFS chain uses; only --source and the bundle differ.
    argv = [
        "--source", "hrrr",
        "--prepared-root", str(prepared_root),
        "--proof-sha256", digests["proof"],
        "--source-manifest-sha256", digests["source_manifest"],
        "--prepared-content-sha256", digests["prepared_content"],
        "--experiment-config", str(handoff["experiment_config"]),
        "--wps-namelist", str(handoff["wps_namelist"]),
        "--io-mode", "history", "--outdir", str(forecast_dir),
    ]
    # [tiles], and the ONE flag on this argv that does not come out of
    # the bundle.
    #
    # THE DEFECT.  --experiment-config above is the authority the
    # PREPARER published, not the config the user wrote: it is rendered
    # from tables that stage builds in code
    # (tools/hrrr_single_domain_benchmark.py _experiment_tables), and
    # those tables are {experiment, projection, shared, domain} -- there
    # has never been a [tiles] among them.  So a user's [tiles] block
    # reached run-plan, was validated, was reported by --resolve, and
    # then vanished at exactly this line: the forecast loaded a document
    # that does not mention it and ran resident, with nothing in the log
    # to say the mode had been asked for.
    #
    # Forwarded as a FLAG rather than published into that document on
    # purpose.  The published authority is hash-bound -- the runner
    # compares its name and sha256 against the portable source manifest
    # -- so putting [tiles] in it would make the execution mode part of
    # the prepared bundle's identity, and a bundle prepared streamed
    # could then not be re-run resident.  That is the exact coupling
    # streaming.identity_payload_entry exists to prevent: it contributes
    # NOTHING to the restart identity so that a forecast which outgrew
    # its card can resume on the machine it outgrew.
    #
    # Only when the mode is enabled, so an ordinary run composes the
    # argv it has always composed, token for token.
    if exp.tiles.enabled:
        from woof.stage_cli import streaming_flags
        argv += streaming_flags("single", tiles=exp.tiles)
    if getattr(getattr(exp, "devices", None), "enabled", False):
        from woof.stage_cli import devices_flags
        argv += devices_flags("single", options=exp.devices)
    if getattr(getattr(exp, "simulated_radar", None), "enabled", False):
        from woof.simulated_radar_config import execution_flags
        argv += execution_flags(exp.simulated_radar)
    # [radar_heating] crosses to the door the same way: beside the
    # hash-bound experiment, so the preparation is the unheated control's.
    if getattr(getattr(exp, "radar_heating", None), "enabled", False):
        from woof.da.forecast_heating import execution_flags as heating_flags
        argv += heating_flags(exp.radar_heating)
    observer.enter_stage("forecast", phase="forecast")
    from woof import prepared_single_domain_forecast as runner

    code = runner.main(argv, observer=observer)
    if code:
        raise StageExitError("forecast", code)

    # -- render --------------------------------------------------------
    # Shared with the tree arm above: `render_products` -- including
    # `none` -- must mean exactly the same thing however the forecast
    # was produced, and the wizard's printed HRRR chain has no render
    # step at all, so this is the one place the sources are made to
    # agree.
    return _chain_render(plan, forecast_dir=forecast_dir, run_dir=run_dir,
                        observer=observer)


class _NativeFetchBeside:
    """A native HRRR chain's as-posted fetch, run beside its preparation.

    The fetch stage this chain has always run (:func:`_run_fetch`, in this
    process, its per-file transfer events on the run's stream), on a thread
    of its own so the preparation can start once the fetch has scheduled the
    window (``posting``).  When it ends without a marker for every lead it
    says so in ``posting/failed.json`` (:func:`woof.go_cli._record_fetch_end`),
    so a preparation waiting on a lead ends by name; the posting folder is
    also relayed onto the run's stream (``posting_schedule``, ``lead_posted``,
    ``lead_ready`` and the start waits).  A fetch in this process cannot be
    stopped from outside: when the chain fails first, it ends with the
    process, as the fetch stage did before.
    """

    def __init__(self, arguments, *, run_dir: Path, fetch_plan, events):
        import contextvars

        from woof import go_cli

        self.arguments = list(arguments)
        self.fetch_plan = dict(fetch_plan)
        self.folder = go_cli.posting_folder(self.fetch_plan)
        self.launched = time.time()
        self.posting: Path | None = None
        self.report: dict | None = None
        self._error: BaseException | None = None
        self.relay = None
        if events is not None:
            from woof import chain_events

            self.relay = chain_events.HostedPostingRelay(
                events, data_dir=self.fetch_plan["data"])
            self.relay.start(since_unix_ms=int(self.launched * 1000))
        context = contextvars.copy_context()
        self._thread = threading.Thread(
            target=context.run, args=(self._run, run_dir, events),
            name="hrrr-fetch-beside", daemon=True)
        self._thread.start()

    def _run(self, run_dir, events) -> None:
        from woof import go_cli

        try:
            self.report = _run_fetch(self.arguments, run_dir, events=events)
        except BaseException as failure:  # noqa: BLE001 - raised by result()
            self._error = failure
        finally:
            go_cli._record_fetch_end(self.folder, self.fetch_plan,
                                     since=self.launched, error=self._error)

    def await_schedule(self) -> Path | None:
        """The posting folder once this fetch has scheduled its window.

        ``None`` when the fetch ended first (its window was here whole, or
        it failed: :meth:`result` raises that).
        """

        from woof import go_cli

        while True:
            if go_cli._fresh_schedule(self.folder, self.fetch_plan,
                                      since=self.launched):
                self.posting = self.folder
                return self.posting
            if not self._thread.is_alive():
                if go_cli._fresh_schedule(self.folder, self.fetch_plan,
                                          since=self.launched):
                    self.posting = self.folder
                return self.posting
            time.sleep(0.25)

    def said_failure(self) -> bool:
        """Whether the fetch failed, or wrote that it is failing (``failed.json``)."""

        from woof.source_posting import POSTING_FAILED_NAME

        return (self._error is not None
                or (self.folder / POSTING_FAILED_NAME).is_file())

    def result(self, timeout: float | None = None) -> dict | None:
        """Wait for the fetch; raise its failure if it failed."""

        self._thread.join(timeout)
        if self._error is not None:
            raise self._error
        return self.report

    def close(self) -> None:
        if self.relay is not None:
            self.relay.stop()
            self.relay = None


def native_whole_window_reason(*, domains: int, donors: bool) -> str | None:
    """Why a native HRRR run fetches its whole window before it prepares.

    ``None`` for a run without a PMSL donor, which prepares as its
    leads post.  Each reason names what reads every lead before the start.
    """

    if donors:
        return ("a native HRRR run with a PMSL donor fetches its whole window "
                "before it prepares: the donor is bound into the source "
                "manifest before any hour is decoded")
    return None


def _native_fetch_beside(plan: RunPlan, hints, exp, *, data_dir: Path,
                         run_dir: Path, observer: RunObserver,
                         prepare_only: bool):
    """Start the native chain's fetch beside its preparation, or ``None``.

    As posted unless the run said ``--whole-cycle`` (``as_posted`` false),
    for a run to a forecast. A PMSL donor is said by
    :func:`native_whole_window_reason`; a preparation-only door stays sealed.
    """

    if hints.get("as_posted") is False or (prepare_only and not os.environ.get("WOOF_CONTINUATION_PREFIX")):
        return None
    reason = native_whole_window_reason(
        domains=len(exp.domains),
        donors=bool(plan.run_options.get("supplement")))
    if reason is not None:
        # The shape and the reason, said once: this run still chains on
        # its root's head, after the whole window is fetched.
        print(f"run: {reason}", flush=True)
        return None
    from woof.fetch import parse_cycle

    fetch_plan = {"source": "hrrr", "data": data_dir,
                  "cycle": parse_cycle(str(hints["cycle"]), "hrrr").strftime(
                      "%Y-%m-%dT%H")}
    observer.enter_stage("fetch", phase="fetch")
    beside = _NativeFetchBeside(
        _fetch_arguments_from_hints(hints, out=data_dir), run_dir=run_dir,
        fetch_plan=fetch_plan, events=getattr(observer, "events", None))
    try:
        posting = beside.await_schedule()
    except BaseException:
        beside.close()
        raise
    if posting is not None:
        print("run: the HRRR fetch runs beside the preparation, which "
              "starts on the window's first leads and waits for each later "
              "one as it posts", flush=True)
    return beside


#: How long a fetch that failed, or said it was failing, before the stage
#: beside it failed is given to end, so its own failure is the chain's
#: (``woof.go_cli._SAID_FAILURE_GRACE_SECONDS``, the same allowance).
_FETCH_SAID_FAILURE_GRACE_SECONDS = 30.0


@contextlib.contextmanager
def _beside_settled(beside):
    """The chain's failure while an as-posted fetch runs beside it.

    When the fetch had failed, or said it was failing
    (``posting/failed.json``), before the preparation or forecast failed,
    they failed because of it and its failure is the chain's: a lead later
    than its budget ends the run as
    :class:`woof.ingest.boundary_stream.SourceBehind` (exit 75, the lead
    named).  Otherwise the stage that failed first is the chain's failure.
    """

    if beside is None or beside.posting is None:
        # No fetch beside the chain: it ended before the preparation began.
        yield
        return
    try:
        yield
    except BaseException as error:
        fetch_first = beside.said_failure()
        beside.close()
        if not fetch_first:
            raise
        from woof.ingest.boundary_stream import (
            POSTING_FAILED_NAME, SOURCE_BEHIND_CODE, SourceBehind,
            read_replaced_json,
        )

        try:
            record = read_replaced_json(beside.posting / POSTING_FAILED_NAME)
        except (OSError, ValueError):
            record = None
        if (isinstance(record, dict)
                and record.get("code") == SOURCE_BEHIND_CODE
                and not isinstance(error, SourceBehind)):
            raise SourceBehind(record) from error
        try:
            beside.result(timeout=_FETCH_SAID_FAILURE_GRACE_SECONDS)
        except BaseException as failed:  # noqa: BLE001 - the chain's failure
            raise failed from error
        raise
    finally:
        beside.close()


def _hrrr_single_chain(*, prep_root: Path, preparation, forecast_dir: Path,
                       exp, observer: RunObserver):
    """Run the native HRRR preparation beside a forecast started on its head.

    ``preparation(built=...)`` runs ``tools.prepare_hrrr_wrf`` to its seal
    (or reuses a finished bundle) and calls ``built`` as the seal lands;
    the forecast is the single-domain runner bound to
    the head (``--prepared-head-sha256``), which waits at a seam for an
    hour not built yet and holds the seal to the head at the end.  Returns
    ``(prepared, forecast)``; ``forecast`` is ``None`` when no chained head
    was published, and the caller then runs the sealed bundle as before.
    """

    from woof import stage_cli
    from woof.ingest.boundary_stream import run_chained

    seal = _ChainSeal(observer, prep_root)

    def chained_preparation():
        result = preparation(built=seal.sealed)
        seal.sealed(result)
        return result

    def chained_forecast(head_sha256):
        if head_sha256 is None:
            return None
        bundle = stage_cli.resolve_head_bundle(prep_root, head_sha256)
        observer.finish_stage(prepared_root=str(prep_root),
                              prepared={"chained": True,
                                        "head_sha256": head_sha256})
        events = getattr(observer, "events", None)
        if events is not None:
            events.emit("prepare_head_ready", head_sha256=head_sha256)
        seal.head_bound()
        from woof.hrrr_prepared_bundle import (
            EXPERIMENT_CONFIG_NAME, WPS_NAMELIST_NAME)

        # The published authorities the head's proof binds by name and
        # digest, as the sealed arm relays them from the handoff.  An
        # as-posted head names no source manifest (its seal writes it), and
        # the runner binds its input plan instead.
        manifest_sha256 = bundle["source_manifest_sha256"]
        argv = [
            "--source", "hrrr",
            "--prepared-root", str(prep_root),
            "--prepared-head-sha256", head_sha256,
            *(() if manifest_sha256 is None else (
                "--source-manifest-sha256", str(manifest_sha256))),
            "--experiment-config", str(prep_root / EXPERIMENT_CONFIG_NAME),
            "--wps-namelist", str(prep_root / WPS_NAMELIST_NAME),
            "--io-mode", "history", "--outdir", str(forecast_dir),
        ]
        if exp.tiles.enabled:
            argv += stage_cli.streaming_flags("single", tiles=exp.tiles)
        if getattr(getattr(exp, "simulated_radar", None), "enabled", False):
            from woof.simulated_radar_config import execution_flags
            argv += execution_flags(exp.simulated_radar)
        if getattr(getattr(exp, "radar_heating", None), "enabled", False):
            from woof.da.forecast_heating import execution_flags as heating_flags
            argv += heating_flags(exp.radar_heating)
        observer.enter_stage("forecast", phase="forecast")
        from woof import prepared_single_domain_forecast as runner

        code = runner.main(argv, observer=observer)
        if code:
            raise StageExitError("forecast", code)
        return bundle

    return run_chained(prepared_root=prep_root, prepare=chained_preparation,
                       forecast=chained_forecast, observer=observer)


def _hrrr_tree_chain(*, prep_root: Path, tree_root: Path, preparation,
                     hierarchy, config_path: Path, forecast_dir: Path,
                     observer: RunObserver, devices: int | None = None,
                     prepare_only: bool = False):
    """Run the native HRRR root preparation and hierarchy beside a forecast.

    ``preparation`` runs the root preparation to its seal and
    ``hierarchy`` the hierarchy stage; the hierarchy starts as soon as the
    root preparation publishes a head of this run (or, when it publishes at
    its seal or is reused whole, once it has finished), and chains on that
    head itself (``woof.hrrr_hierarchy_direct``).  The forecast is the
    tree runner bound to the tree's head (``--prepared-head-sha256``).
    Returns ``(prepared, forecast)``; ``forecast`` is ``None`` when the
    tree was published at its seal or reused whole, and the caller runs
    the sealed tree as before.

    A hierarchy stage stopped by its forecast (``stop.json``) stops the root
    preparation too, which is the breakage this prevents: a stopped run
    that waited for the root to build every later hour for nothing.  Any
    other hierarchy failure leaves the root preparation running to its
    seal, so a retry reuses it, as :func:`run_chained` leaves a producer
    whose forecast failed.
    """

    import contextvars
    import threading

    from woof.ingest.boundary_stream import (
        STOP_NAME, fresh_chained_head, request_stop, run_chained, stream_dir)

    seal_lock = threading.Lock()
    seal_state: dict[str, Any] = {"head": False, "emitted": False}

    def emit_sealed_once():
        if seal_state["emitted"] or not seal_state["head"] \
                or "prepared" not in seal_state:
            return
        seal_state["emitted"] = True
        events = getattr(observer, "events", None)
        if events is not None:
            events.emit("prepare_sealed", prepared_root=str(tree_root),
                        prepared=seal_state["prepared"])

    def tree_preparation():
        box: dict[str, Any] = {}
        started = time.time()

        def root():
            try:
                box["result"] = preparation()
            except BaseException as error:  # noqa: BLE001 - re-raised below
                box["error"] = error

        context = contextvars.copy_context()
        worker = threading.Thread(target=context.run, args=(root,),
                                  name="hrrr-root-preparation", daemon=True)
        worker.start()
        while worker.is_alive() \
                and fresh_chained_head(prep_root, since=started) is None:
            worker.join(0.5)
        if not worker.is_alive():
            worker.join()
            if "error" in box:
                raise box["error"]
        try:
            tree = hierarchy()
        except BaseException as error:
            if (stream_dir(tree_root) / STOP_NAME).exists() \
                    or _is_interrupt(error) or not isinstance(error, Exception):
                request_stop(prep_root, "the tree's forecast ended: "
                                        f"{type(error).__name__}: {error}")
            worker.join()
            if "error" in box and not _is_interrupt(error):
                raise box["error"] from error
            raise
        worker.join()
        if "error" in box:
            raise box["error"]
        result = {"root": box.get("result"), "hierarchy_root": str(tree),
                  "tree_root": str(tree_root)}
        with seal_lock:
            seal_state["prepared"] = result
            emit_sealed_once()
        return result

    def tree_forecast(head_sha256):
        if head_sha256 is None:
            return None
        observer.finish_stage(prepared_root=str(tree_root),
                              prepared={"chained": True,
                                        "head_sha256": head_sha256})
        events = getattr(observer, "events", None)
        if events is not None:
            events.emit("prepare_head_ready", head_sha256=head_sha256)
        with seal_lock:
            seal_state["head"] = True
            emit_sealed_once()
        if prepare_only:
            return head_sha256
        _hrrr_tree_forecast(tree_root=tree_root, config_path=config_path,
                            forecast_dir=forecast_dir, observer=observer,
                            head_sha256=head_sha256, devices=devices)
        return head_sha256

    return run_chained(prepared_root=tree_root, prepare=tree_preparation,
                       forecast=tree_forecast)


def _hrrr_hierarchy_stage(*, prep_root: Path, inputs: Mapping[str, Path],
                          hints: Mapping[str, Any], geog_root,
                          manifest: Path | None, cycle: str, run_dir: Path,
                          observer: RunObserver,
                          statics_corridor: bool = False,
                          acknowledgements: Sequence[str] = (),
                          lifecycle_config: Path | None = None,
                          observe_stage: bool = True) -> Path:
    """Build d02..dNN from the sealed root preparation.

    The stage the GFS tree does not have.  rw-wps is not on this path at
    all: the root preparation above is ``tools.prepare_hrrr_wrf``, and
    this turns its sealed d01 into a hierarchy the tree runner can
    execute.

    Nine required flags, and they are not the preparer's -- notably
    ``--stock-wrf-namelist-input``, which rw-wps rejects outright.  That
    is why this composes its own argv rather than sharing a builder with
    either neighbour: two tools that take *almost* the same flags are
    exactly where a shared builder starts passing one of them something
    it refuses.

    ``statics_corridor`` adds the tenth, and it is where the HRRR chain
    answers a moving nest.  On the GFS chain the corridor flag rides the
    rw-wps prepare stage; here the root preparer knows nothing of the
    children, so the flag belongs to THIS stage -- the one holding
    ``--geog-root`` and the child geometries.  The caller derives the
    boolean from the corridor module's own follow predicate, so the
    plan that was resolved as corridor-bearing is the plan that seals
    one.
    """

    from woof.fetch import sha256_file
    from woof.go_cli import run_stage

    if manifest is None:
        # The root head binds the posting plan and the admitted start
        # leads. Its seal writes both documents; the hierarchy reads the
        # same head and completes their identities at its own seal.
        from woof.ingest.boundary_stream import live_chained_head
        from tools.prepare_hrrr_wrf import POSTED_SOURCE_MANIFEST

        root_head = live_chained_head(prep_root)
        manifest = prep_root / POSTED_SOURCE_MANIFEST
        if root_head is None:
            manifest_sha = sha256_file(manifest)
        else:
            posted = root_head["basis"].get("as_posted")
            if not posted:
                raise PlanError("the native as-posted hierarchy has no root "
                                "input plan to bind its source manifest")
            manifest = prep_root / posted["documents"]["source_manifest"]["path"]
            manifest_sha = root_head["basis"]["cache"]["identity"][
                "source_manifest_sha256"]
    else:
        manifest_sha = sha256_file(manifest)

    tree_root = run_dir / "chain" / "hrrr-hierarchy"
    command = [
        sys.executable, "-m", "woof.hrrr_hierarchy_direct",
        "--root-preparation", str(prep_root),
        "--root-domain-spec", str(inputs["target_domain"]),
        "--wps-namelist", str(inputs["wps_namelist"]),
        "--namelist-input", str(inputs["namelist_input"]),
        # rw-wps has no such flag; this stage requires it.
        "--stock-wrf-namelist-input", str(inputs["stock_namelist_input"]),
        "--geog-root", str(geog_root),
        "--source-manifest", str(manifest),
        "--source-manifest-sha256", manifest_sha,
        "--cycle", cycle,
        "--output-root", str(tree_root),
    ]
    if statics_corridor:
        # Bare, exactly as the GFS chain passes it: the preparation
        # reads that as "every child domain", which is also what
        # corridor_estimate priced for this plan.
        command.append("--statics-corridor")
    # The config's [experiment] acknowledgements.  This stage reads the
    # namelists, which have no spelling for them, so without the flag a
    # tree on a suite that requires one (every shortwave-only suite
    # requires constant-downward-longwave-v1) was refused at the
    # hierarchy import for the declaration its own config carries, after
    # the fetch and the root preparation.
    for acknowledgement in acknowledgements:
        command += ["--ack", str(acknowledgement)]
    if lifecycle_config is not None:
        # The namelists cannot spell spawn/follow/retire/rearm, so the
        # stage takes them from the config: a dormant storm slot's
        # corridor must reach the whole parent, and a follower's its reach.
        command += ["--lifecycle-config", str(lifecycle_config)]
    # Only when nonzero.  This stage raises on a negative lead, and
    # raises again if a lead is passed beside the deprecated
    # --valid-time; passing a bare 0 is legal but says nothing, and the
    # chain reads better without it.
    lead = hints.get("forecast_start_hour")
    if lead:
        command += ["--forecast-start-hour", str(int(lead))]

    if observe_stage:
        # Not in a chained tree: its forecast opens its stage at the tree's
        # head, while this stage is still relaying the root's intervals.
        observer.enter_stage("prepare", phase="hierarchy")
    # Reuse is a whole-hierarchy decision: the canonical domain manifest,
    # every cache payload, static/corridor/export artifact, and the prepared
    # root input must still match the completed stage's binding.
    stated = {"source_manifest_sha256": manifest_sha}

    def run_hierarchy():
        run_stage("hierarchy", command, explain=False,
                  progress=tree_root / "progress.json",
                  observer=_GoObserver(observer))
        # Recovery compares the sealed documents, so record the digest
        # the finished hierarchy consumed rather than its head placeholder.
        from woof.ingest.boundary_stream import is_as_posted_placeholder

        if is_as_posted_placeholder(manifest_sha):
            sealed_sha = sha256_file(manifest)
            command[command.index("--source-manifest-sha256") + 1] = sealed_sha
            stated["source_manifest_sha256"] = sealed_sha

    prepared = _prepare_stage(
        tree_root, arguments=command,
        stated=stated, run=run_hierarchy)
    if not tree_root.is_dir():
        raise PlanError(
            f"the HRRR hierarchy stage wrote no {tree_root}; its own "
            "output is above")
    if observe_stage:
        observer.finish_stage(hierarchy_root=str(tree_root),
                              prepared=prepared)
    return tree_root


def _hrrr_tree_forecast(*, tree_root: Path, config_path: Path,
                        forecast_dir: Path,
                        observer: RunObserver,
                        head_sha256: str | None = None,
                        devices: int | None = None) -> None:
    """The same tree runner the GFS tree route drives.

    The relay is the same shape too -- one preparation-receipt digest
    plus the experiment config's own -- and it reaches the same
    schema-matched document resolver.  Only the filename underneath
    differs: this hierarchy writes ``receipt.json`` where rw-wps writes
    ``proof.json``, and because the resolver matches on SCHEMA rather
    than on filename order it needed no change to find it.

    The tool also prints ``preparation_receipt_sha256`` on stdout.  It
    is not read: the tool computes that value as the sha256 of
    ``receipt.json``'s bytes, so hashing the artifact gives the
    identical digest without making a printed line essential.

    ``[tiles]`` NEEDS NO FLAG HERE, and that is a fact about this argv
    rather than an omission.  ``--experiment-config`` below is the
    USER'S config -- the file they wrote or the wizard emitted, bound by
    its own digest on the next line -- so ``exp.tiles`` at the tree
    runner is already the table they typed, verbatim.  The single-domain
    arm needs ``--tiles`` only because IT hands over the authority the
    preparer published instead, and that document has no [tiles] table
    to carry.  Adding a flag here as well would give one table two
    sources on one route, which is how the two arms end up streaming
    differently from the same config.  Pinned by a test rather than left
    to this comment.
    """

    from woof.fetch import sha256_file

    if head_sha256 is None:
        from woof.go_cli import _hierarchy_document

        binding = ["--preparation-receipt-sha256",
                   sha256_file(_hierarchy_document(tree_root))]
    else:
        # A chained tree: the runner restores from the head and binds the
        # seal at the end of the run.
        binding = ["--prepared-head-sha256", str(head_sha256)]
    argv = [
        "--prepared-root", str(tree_root),
        *binding,
        "--experiment-config", str(config_path),
        "--experiment-config-sha256", sha256_file(config_path),
        # `--devices N` is a COUNT override of the [devices] table the
        # user's config already carries to the runner (the reason [tiles]
        # needs no flag above), so it is relayed as the count alone.
        *([] if devices is None else ["--devices", str(int(devices))]),
        "--io-mode", "history", "--outdir", str(forecast_dir),
    ]
    observer.enter_stage("forecast", phase="forecast")
    from woof import prepared_domain_tree_forecast as runner

    code = runner.main(argv, observer=observer)
    if code:
        raise StageExitError("forecast", code)


def _chain_render_plan(plan: RunPlan, *, forecast_dir: Path,
                       run_dir: Path) -> dict[str, Any]:
    """The plan dict a staged chain's render stage runs on.

    One function because it has several readers: the finalize render
    below, and the early render armed before each chain's forecast.
    Two copies of this literal is exactly how a run ends up publishing
    its first frame into one directory and the rest into another.
    """

    return {"run": forecast_dir, "render": forecast_dir.parent / "png",
            "publish_root": plan.run_options.get("publish_root") or forecast_dir.parent / "www",
            "publish_cmd": plan.run_options.get("publish_cmd"),
            "hydro": plan.run_options.get("hydro") or "auto",
            "render_products": plan.run_options.get("render_products"),
            "render_section": plan.run_options.get("render_section"),
            # The data source the run started from, as go's own plan
            # carries it: the finalize render draws the run beside that
            # source's model when the comparison engine knows it
            # (go_cli._reference_compare_stage).  Without it a staged
            # chain -- the route `woof go` takes for HRRR -- drew no
            # comparison sheets at all.
            **({"source": source}
               if (source := _plan_fetch_source(plan)) is not None else {}),
            **({"restart": plan.run_options["restart"]}
               if plan.run_options.get("restart") is not None else {})}


def _plan_fetch_source(plan: RunPlan) -> str | None:
    """The config's ``[fetch] source``, or None when the plan names none.

    An intent plan has no config bytes until the wizard has generated
    them, and an unreadable config has already been refused by the
    road that ran it; neither has a source to give here.
    """

    # A plan stand-in that carries only run options (the early render's
    # callers and their fixtures) has no config to read a source from.
    config_bytes = getattr(plan, "config_bytes", None)
    if config_bytes is None:
        return None
    try:
        raw = tomllib.load(io.BytesIO(config_bytes()))
    except (PlanError, OSError, tomllib.TOMLDecodeError):
        return None
    source = (raw.get("fetch") or {}).get("source")
    return str(source) if source else None


def _chain_render(plan: RunPlan, *, forecast_dir: Path, run_dir: Path,
                  observer: RunObserver) -> Mapping[str, Any]:
    """go's render stage against a staged chain's output, then the summary."""

    render_plan = _chain_render_plan(plan, forecast_dir=forecast_dir, run_dir=run_dir)
    _finish_render(render_plan, observer=observer)
    return _chain_summary(forecast_dir.parent, observer=observer)


def _finish_render(render_plan: dict, *, observer: RunObserver,
                   door: str = "go") -> None:
    """Finish the shared render stage and refuse an unfulfilled request.

    ``door`` is the command the reader typed, carried down to the one
    line the stage addresses them with -- this function serves
    ``woof go`` and ``woof downscale``, and a failure under one must
    not name the other.
    """
    from woof.go_cli import _render_stage, printable, render_command

    observer.enter_stage("finalize", phase="render")
    rendered = _render_stage(render_plan, explain=False,
                             observer=_GoObserver(observer, door=door),
                             door=door)
    if not rendered and str(render_plan.get("render_products") or "").strip().lower() != "none":
        raise PlanError(
            "Forecast completed, but requested pictures were not produced. "
            "Next: run woof setup, then render the saved forecast:\n  "
            + printable(render_command(render_plan)))


def _stage_static_sources(raw, geog_root, *, base_dir) -> None:
    """Fetch the published static files this run reads and is missing.

    Called by each native chain before its forcing fetch, so a recipe
    naming ``[static] source = "hrrr-conus-v4"`` (or a source whose
    metadata selects it) gets the file through ``fetch-geog
    --static-source``'s own fetch rather than a refusal at static
    preparation after the download.  The lines go to stderr as notes, the
    form a launch relays to the terminal, and never onto a stream a
    run-plan consumer reads as events.
    """

    from woof.geog_assets import GeogFetchError
    from woof.static.external_source import (StaticSourceError,
                                              stage_required_static_sources)

    def say(line: str) -> None:
        line = line if line.startswith(("note:", "warning:")) else f"note: {line}"
        print(line, file=sys.stderr, flush=True)

    try:
        stage_required_static_sources(raw, geog_root, base_dir=base_dir,
                                      progress=say)
    except (GeogFetchError, StaticSourceError) as error:
        raise PlanError(str(error)) from None


def _staged_prep_root(run_dir: Path) -> Path:
    """Where the staged chain's preparation writes, and so where its compose scratch goes.

    One function because two readers need the same answer: the chain
    itself, and the disk check before the download, which measures the
    disk the preparation's frame stream will be staged on
    (:func:`woof.ingest.source_coverage.compose_scratch_folder`).
    """

    return Path(run_dir) / "chain" / "prep"


def _compose_scratch_folder(prep_root: Path, projection: Mapping[str, Any]) -> Path | None:
    """The folder a preparation writing ``prep_root`` stages its frame stream in, or None when it stages none."""

    if not (projection.get("compose_scratch_bytes") or projection.get("compose_scratch_min_bytes")):
        return None
    from woof.ingest.source_coverage import compose_scratch_folder

    return compose_scratch_folder(prep_root)


def _run_prep(arguments: Sequence[str]) -> None:
    """Run ``woof prep`` -- the preprocessing stage -- in this process.

    Through the real parser and the real dispatch, exactly as
    :func:`_run_fetch` drives ``woof fetch``: there is no second
    preparation implementation here and there must never be one.
    Separate from its caller so a test can observe the composed argv
    without preparing anything.
    """

    from woof.cli import build_parser
    from woof.ingest.source_coverage import recorded_preparation_refusal

    args = build_parser().parse_args(["prep", *arguments])
    # The preparation door prints its refusal and returns 78; the chain
    # raises that refusal itself, so the run's failed event carries its
    # sentence and remedy instead of "prepare failed (exit 78)."
    with recorded_preparation_refusal() as refused:
        code = args.func(args)
    if code:
        refusal = refused()
        if refusal is not None:
            raise refusal
        raise StageExitError("prepare", code)


# ---------------------------------------------------------------------------
# The staged chain as posted (DESIGN A136 2.5): the fetch beside the preparation
# ---------------------------------------------------------------------------

#: How much earlier than the fetch's launch a file it writes may be dated
#: and still be this fetch's (a filesystem keeping two-second times).
_POSTED_FRESH_SLACK_SECONDS = 2.0
#: How often the staged chain looks for the as-posted fetch's files.
_POSTED_POLL_SECONDS = 0.2
#: How often a start wait is said again on the heartbeat while it lasts.
_POSTED_WAIT_SAY_SECONDS = 5.0


def _staged_beside_refusal(hints: Mapping[str, Any], exp, *,
                           separate_initial: bool = False) -> str | None:
    """Why the staged chain fetches this window whole before preparing, or ``None``.

    ``None``: the fetch runs beside the preparation, which starts on the
    window's start needs and waits for each later lead's marker
    (``mapped_direct --as-posted``), and the forecast binds its head.  One
    domain or a tree: an as-posted mapped tree's head prepares every child
    under the input plan and its seal holds each child to that head
    (``mapped_direct._prepare_chained_mapped_tree``,
    ``boundary_stream.verify_as_posted_tree_children``).  Each reason names
    what reads the whole window first:

    * the run asked for the whole cycle (``as_posted = false``);
    * the source's preparation reads a whole fetched window, or normalizes
      every input file before its decode
      (:func:`woof.source_cli.as_posted_refusal`);
    * an ensemble member: its member is verified over every input file
      before the preparation (:func:`woof.forcing_member.verify_handoff`);
    * a separate initial analysis (``--initial-inputs`` or the recipe's
      ``[fetch] initial_source``): the mapped preparation checks it against
      the complete forcing window before the start (``mapped_direct``).
    """

    if hints.get("as_posted") is False:
        return "the run asked for the whole cycle"
    if separate_initial:
        # Named breakage (2.9.4, HRRR recipe from NOAA's HRRR f00 + RAP
        # boundaries): the default as-posted fetch either dropped the
        # separate analysis (the run started from RAP's 13 km state: 2.4x
        # the cloud water, f01 surface shortwave -23 W/m2 against NOAA) or,
        # once the analysis reached the preparation, was refused there.
        return ("its separate initial analysis is checked against the "
                "complete forcing window before the start")
    from woof.source_cli import as_posted_refusal

    refusal = as_posted_refusal(str(hints.get("source")))
    if refusal is not None:
        return refusal
    # Retired guard (2026-10-29, WOOF canary run_9498bc66b5fdd08c): a
    # domain tree used to fetch its whole window first here, written
    # 2026-09-30 when mapped_direct refused an as-posted tree.  That refusal
    # was retired on 2026-10-02 (d7ab2f3d7f: the tree's head binds the input
    # plan and its seal re-binds every child), but this one stayed, so every
    # nested run on a mapped source waited for its cycle's LAST lead to post
    # before preparing: the nyc-500m HRRR template launched 07:42Z on the
    # still-posting 06Z cycle sat 14.5 min in fetch until f048 posted 07:57Z
    # (19.1 min click to first map).
    from woof.forcing_member import member_contract

    try:
        member = member_contract(str(hints["source"]), hints.get("member"))
    except (KeyError, ValueError):
        member = None
    if member is not None:
        return ("its ensemble member is verified over every input file "
                "before the preparation")
    return None


def _posted_manifest_path(data_dir: Path, prep_root: Path) -> Path:
    """Where an as-posted staged preparation seals its input manifest.

    Beside the fetched files, because the seal holds each manifest row to
    the lead marker that names the same file from the fetch folder
    (``mapped_direct._PostedMappedSource``), under a name of this run's
    own, so the standalone ``prep-command.txt`` manifest in the shared
    download and another run's are left as they are.
    """

    digest = hashlib.sha256(
        str(Path(prep_root).resolve()).encode("utf-8")).hexdigest()[:12]
    return Path(data_dir) / f"inputs-{digest}.json"


class _FetchBesideEvents:
    """The run's stream, as the fetch beside the preparation relays onto it.

    The fetch's transfer sink is ambient to this process
    (:func:`woof.progress.event_sink`), and the preparation runs in it at
    the same time, so its step records reach the fetch's relay too.  They
    are on the stream already (:func:`_preparation_relay`); said twice they
    would read as two preparations.
    """

    def __init__(self, events):
        self._events = events

    def emit(self, event: str, **fields: Any):
        if event == "warning" and fields.get("code") == "preparation_progress":
            return None
        return self._events.emit(event, **fields)


class _PostedFetch:
    """The as-posted fetch, run in this process on its own thread.

    The staged chain runs its stages in this process (``woof fetch``'s own
    handler through :func:`_run_fetch`, ``woof prep`` through
    :func:`_run_prep`), so its as-posted fetch does too: the same call, on a
    thread, beside the preparation that waits on its lead markers.  While
    the run waits for its start needs, a start-need lead the fetch waits
    for is the run's ``waiting:source`` heartbeat record, by the rule
    ``woof go``'s hosted chain uses (:meth:`_GoObserver._start_wait`).

    A chain that fails stops it (:meth:`stop`): the fetch ends at its next
    wait (:func:`woof.fetch_as_posted.stop_event`), and a fetch that ends
    with a lead unmarked says so in ``posting/failed.json``
    (:func:`woof.go_cli._record_fetch_end`), so a preparation still
    waiting on a marker ends by name.
    """

    def __init__(self, arguments: Sequence[str], run_dir: Path, *,
                 data_dir: Path, hints: Mapping[str, Any],
                 observer: RunObserver):
        import contextvars

        from woof import fetch_as_posted
        from woof.source_posting import POSTING_DIRNAME

        self.data_dir = Path(data_dir)
        self.folder = self.data_dir / POSTING_DIRNAME
        cycle = str(hints.get("cycle") or "")
        #: What :func:`woof.go_cli._record_fetch_end` matches the
        #: schedule against; ``latest`` takes the schedule's own cycle.
        self.plan = {"data": str(self.data_dir), "source": str(hints["source"]),
                     "cycle": None if cycle == "latest" else cycle[:13]}
        self.view = _GoObserver(observer)
        self.launched = time.time()
        self.stop_event = fetch_as_posted.stop_event(self.data_dir)
        self._box: dict[str, Any] = {}
        self._said_wait = 0.0
        context = contextvars.copy_context()
        self._thread = threading.Thread(
            target=context.run,
            args=(self._run, list(arguments), run_dir,
                  getattr(observer, "events", None)),
            name="run-plan-posted-fetch", daemon=True)
        self._thread.start()

    def _run(self, arguments, run_dir, events) -> None:
        from woof import fetch_as_posted
        from woof.go_cli import _record_fetch_end

        error = None
        try:
            self._box["report"] = _run_fetch(
                arguments, run_dir,
                events=None if events is None else _FetchBesideEvents(events))
        except BaseException as failure:  # noqa: BLE001 - see result()
            error = failure
            if not self.stop_event.is_set():
                self._box["error"] = failure
        finally:
            _record_fetch_end(self.folder, self.plan, since=self.launched,
                              error=error, stopped=self.stop_event.is_set())
            fetch_as_posted.release_stop(self.data_dir)

    def running(self) -> bool:
        return self._thread.is_alive()

    def failure(self) -> BaseException | None:
        return self._box.get("error")

    def said_failure(self) -> bool:
        """Whether the fetch failed, or said it is failing (its ``failed.json``)."""

        from woof.source_posting import POSTING_FAILED_NAME

        return ("error" in self._box
                or (self.folder / POSTING_FAILED_NAME).is_file())

    def result(self) -> dict[str, Any]:
        """Wait for the fetch to end; its report, or its failure raised."""

        self._thread.join()
        if "error" in self._box:
            raise self._box["error"]
        return dict(self._box.get("report") or {})

    def stop(self, *, grace: float = 30.0) -> None:
        """End the fetch at its next wait: the run it fetched for has failed."""

        self.stop_event.set()
        self._thread.join(grace)

    def settle(self, *, fetch_first: bool,
               grace: float = 30.0) -> BaseException | None:
        """After the chain failed: the chain's failure, if it is the fetch's.

        ``fetch_first``: the fetch had failed, or said it was failing,
        before the preparation or forecast did, which then failed because
        of it; its failure is the chain's once it ends.  Otherwise the run
        is over and so is its download: the fetch is stopped.
        """

        if fetch_first:
            self._thread.join(grace)
            failed = self.failure()
            if failed is not None:
                return failed
        self.stop(grace=grace)
        return None

    # -- the files it publishes -------------------------------------------

    def _fresh(self, path: Path) -> dict[str, Any] | None:
        try:
            if Path(path).stat().st_mtime \
                    < self.launched - _POSTED_FRESH_SLACK_SECONDS:
                return None
        except OSError:
            return None
        from woof.ingest.boundary_stream import read_replaced_json

        try:
            payload = read_replaced_json(Path(path))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def schedule(self) -> dict[str, Any] | None:
        """This fetch's posting schedule of this window, once it has one."""

        from woof.source_posting import POSTING_SCHEDULE_NAME

        schedule = self._fresh(self.folder / POSTING_SCHEDULE_NAME)
        if schedule is None or schedule.get("source") != self.plan["source"]:
            return None
        cycle = str(schedule.get("cycle") or "")[:13]
        if self.plan["cycle"] is None:
            self.plan["cycle"] = cycle
        return schedule if cycle == self.plan["cycle"] else None

    def handoff(self) -> dict[str, Any] | None:
        """The ``prep-arguments.json`` this fetch wrote as posted, once written."""

        from woof import fetch_routes

        document = self._fresh(self.data_dir / fetch_routes.PREP_ARGUMENTS_NAME)
        if (document is None
                or document.get("schema") != fetch_routes.PREP_ARGUMENTS_SCHEMA
                or not document.get("as_posted")):
            return None
        return document

    def start_wait(self, schedule: Mapping[str, Any]) -> None:
        """The run's start wait from the schedule, said every few seconds."""

        now = time.monotonic()
        if now - self._said_wait >= _POSTED_WAIT_SAY_SECONDS:
            self._said_wait = now
            self.view._start_wait(schedule)

    def forecast_begins(self) -> None:
        """From here the forecast says its own waits (its seams)."""

        self.view._forecast_begun = True
        self.view._end_start_wait()

    def await_handoff(self) -> tuple[Path | None, dict[str, Any] | None]:
        """The posting folder and the handoff the preparation starts from.

        ``(None, None)`` when the fetch ended without scheduling a wait (its
        window was already here whole, or its source is not waited on): the
        window is fetched, and the chain prepares it whole.  The handoff is
        the one this fetch writes once the start needs and donors are in,
        before its later leads move (DESIGN A136 2.3 step 2).  A fetch that
        fails first raises its failure.
        """

        schedule = None
        while True:
            schedule = self.schedule() or schedule
            if schedule is not None:
                document = self.handoff()
                if document is not None:
                    self.view._end_start_wait()
                    return self.folder, document
                self.start_wait(schedule)
            if not self.running():
                schedule = self.schedule() or schedule
                document = self.handoff() if schedule is not None else None
                self.view._end_start_wait()
                self.result()
                return ((self.folder, document) if document is not None
                        else (None, None))
            time.sleep(_POSTED_POLL_SECONDS)


def _say_posted_source_behind(observer, folder: Path, *, primary: bool = True) -> None:
    """A late lead: the run's cause, or a timeout after its first failure.

    The fetch writes ``posting/failed.json`` with ``code: source_behind``
    and the lead's times before it exits 75; the preparation then ends on
    that record in its own process, so this run sees two exit codes and no
    lead.  Said to the observer here, unless the forecast already said it
    at a seam with its model time, so the run emits ``source_behind`` and
    exits 75 as DESIGN A136 3.6 asks, naming the lead.
    """

    if getattr(observer, "source_behind_record", None) is not None:
        return
    from woof.source_posting import POSTING_FAILED_NAME

    record = _read_json_object(Path(folder) / POSTING_FAILED_NAME)
    if record.get("code") != "source_behind":
        return
    if not primary:
        # A forecast can fail before a later lead times out while its
        # preparation continues to the seal. Keep that timeout as the
        # second fact, without replacing the failure that ended the run.
        observer.secondary_source_behind_record = dict(record)
        observer.warn("secondary_source_behind", "After the run failed: " +
                      (record.get("message") or "a later source lead timed out"),
                      source_behind=source_behind_fields(record))
        return
    hook = getattr(observer, "source_behind", None)
    if hook is not None:
        hook({**record, "model_elapsed_seconds": None,
              "model_valid_time": None, "frames_kept": None,
              "checkpoint": None})


def _settle_posted_fetch(fetch: _PostedFetch, stopped: BaseException, *,
                         fetch_first: bool) -> None:
    """After the chain beside the as-posted fetch failed: stop it, or raise its failure.

    The fetch's failure is the chain's when it failed, or said it was
    failing (``posting/failed.json``), before the preparation or forecast
    did, which then failed because of it; a lead past its budget is said by
    the preparation's own ``SourceBehind``, which stands.  Otherwise the
    fetch is stopped.  Not on an interrupt: the interrupt contract kills
    no child, and a stop reaches this process's fetch with the rest of it.
    ``fetch_first`` was sampled at the first callback exception, before
    waiting for the preparation's seal could let a later fetch failure in.
    """

    if isinstance(stopped, (KeyboardInterrupt, ChainInterrupted)):
        return
    from woof.ingest.boundary_stream import SourceBehind

    failed = fetch.settle(fetch_first=fetch_first)
    if failed is not None and not isinstance(stopped, SourceBehind):
        raise failed from None


def _staged_chem_fetch(exp, observer) -> None:
    """Fetch the experiment's data-store chem sources over its own box and window."""
    import subprocess
    import sys

    # The chem table read directly (go_cli is the GPU front door, which the
    # standalone preparation package does not stage): the enabled sources a
    # data store serves, as go_cli.chem_data_store_sources selects them.
    from datetime import timedelta

    from woof.chem_table import chem_names, load_sets
    run = exp.domains[0].run
    sets = chem_names(getattr(run, "chem_sets", None))
    if not sets:
        return
    names = chem_names(getattr(run, "chem_sources", None), "chem_sources")
    table = load_sets(sets, names)
    from woof.data_store_fetch import is_listed_emission
    # RAVE (a published-listing emission source) rides the same fetch, as
    # go_cli.chem_data_store_sources selects it: the staged chain dropped it,
    # so the HRRR recipe's fire hours were first fetched at the forecast's
    # first fire step (a rented box, 2026-10-21 18:47Z, "chem fetch: rap-native-
    # smoke, hrrr-native-smoke").
    sources = tuple(name for name in names
                    if name in table.sources
                    and ((table.sources[name].acquisition is not None
                          and table.sources[name].acquisition.get("kind") != "published_files")
                         or is_listed_emission(table.sources[name])))
    if not sources:
        return
    from woof.static.projection import grids_from_projection_config
    lat, lon = grids_from_projection_config(exp)[0].latlon_mass()
    margin = 0.5
    area = ",".join(f"{value:.3f}" for value in (
        float(lat.min()) - margin, float(lon.min()) - margin,
        float(lat.max()) + margin, float(lon.max()) + margin))
    start = exp.start_time.isoformat()
    end = (exp.start_time + timedelta(seconds=float(exp.run_seconds))).isoformat()
    from woof.chem_boundary_remap import boundary_rows
    # Initial-only sources need one bracket. Fetching the full forecast
    # otherwise decodes many gigabytes that no later frame consumes.
    windows = {}
    for name in sources:
        initial = boundary_rows(table, name, available=names, frame="initial")
        lateral = boundary_rows(table, name, available=names, frame="boundary")
        emission = any(ref["source"] == name
                       for row in table.rows for ref in row.emissions)
        source_end = start if initial and not lateral and not emission else end
        windows.setdefault(source_end, []).append(name)
    for source_end, group in windows.items():
        command = [sys.executable, "-m", "woof.data_store_fetch", "fetch"]
        for name in group:
            command.extend(("--source", name))
        command.extend(("--area", area, "--start", start, "--end", source_end))
        print(f"chem fetch: {', '.join(group)} over {area} through {source_end}", flush=True)
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise PlanError(
                f"the chem fetch for {', '.join(group)} failed (exit "
                f"{completed.returncode}): {completed.stderr.strip()[-2000:]}")


def _staged_chain(plan: RunPlan, *, config_path: Path, exp,
                  observer: RunObserver, run_dir: Path,
                  prepare_only: bool = False, reviewed_inputs=None,
                  defer_initial: bool = False) -> Mapping[str, Any]:
    """The staged route for a packaged mapped source: fetch, prep, sim.

    The documented per-stage chain for every packaged
    mapped-composition source -- ``woof fetch`` (a table acquisition
    route), ``woof prep`` (rw-wps's declarative mapped arm, its
    packaged profile already bound), then the prepared forecast runner
    -- driven rather than printed, with each stage's refusals left
    entirely alone.  Nothing here knows a model's name: which sources
    arrive at this function is :func:`_chain_key`'s answer, derived
    from the registry row, and every argument below is composed from an
    artifact a stage published.

    The preparation's arguments are the FETCH ROUTE'S OWN bound
    handoff: the table route writes ``prep-arguments.json`` beside the
    bytes it verified (the ordered ``--input-list``, every
    ``--supplement`` role binding, the manifest authoring flag), and
    this chain appends explicit caller-supplied supplements and the four
    caller-owned paths: the WPS namelist, experiment config, geography
    root and output root. Automatic source bindings remain the fetch
    route's, and the preparer validates all supplied donor roles.
    Authored input manifests belong beside this chain's preparation, leaving
    any standalone preparation's manifest in the fetch folder unchanged.
    """

    from woof import fetch_routes
    from woof.go_cli import config_fetch_request, managed_download_dir

    raw = tomllib.load(io.BytesIO(config_path.read_bytes()))
    # The [fetch] table plus the model top the config's ladder needs, when
    # the source's registry row says its fetch has to be asked for it, and
    # the host run_options.transport pins over the table's.
    # Execution already retimed the config. Reapplying its nominal cycle
    # here would erase the boundary source's resolved cycle offset.
    hints = _pinned_fetch_hints(plan, config_fetch_request(raw), pin_cycle=False)
    intent = plan.config_intent or {}
    data_dir = Path(plan.run_options.get("data_dir")
                    or intent.get("data_dir") or managed_download_dir(run_dir, hints))
    geog_root = plan.run_options.get("geog_root") or intent.get("geog_root")
    if geog_root is None:
        from woof.geog_assets import geog_root_from_tables

        geog_root = geog_root_from_tables(raw, base_dir=config_path.parent)
    _stage_static_sources(raw, geog_root, base_dir=config_path.parent)
    # The WPS namelist the mapped preparation binds by digest: the one a
    # door wrote beside the config under its own stem, or, where there is
    # none (a shipped recipe, a hand-written or imported config), the one
    # this run renders from the config into its own folder.  The same
    # function answered resolve_plan's question before anything was
    # fetched.
    from woof.companion_domains import configuration_wps_namelist

    try:
        namelist = configuration_wps_namelist(
            config_path, exp, raw=raw, into=run_dir / "chain" / "route-inputs")
    except ValueError as refusal:
        raise PlanError(str(refusal)) from None
    prep_root = _staged_prep_root(run_dir)
    forecast_dir = run_dir / "chain" / "run"
    #: The as-posted fetch running beside the preparation (_PostedFetch),
    #: its posting folder, its handoff and the relay of its posting files
    #: onto the run's stream; ``None`` when the window is fetched (or
    #: found) whole first.
    beside = posting = posted_handoff = posting_relay = None

    # The analysis uses the same acquisition and handoff as every mapped
    # source. An explicit inventory wins over the recipe's automatic fetch.
    from woof.initial_source import initial_fetch_request, initial_inputs_from_fetch
    initial_inputs = plan.run_options.get("initial_inputs")
    initial_hints = initial_fetch_request(raw.get("fetch", {}))
    if initial_inputs is None and initial_hints is not None and not defer_initial:
        initial_hints = _pinned_fetch_hints(plan, initial_hints, pin_cycle=False)
        # An analysis is one complete object set, never a streaming window.
        initial_hints["as_posted"] = False
        initial_hints.pop("late_after_minutes", None)
        initial_dir = managed_download_dir(run_dir, initial_hints)
        observer.enter_stage("fetch", phase="initial-fetch")
        initial_report = _run_fetch(
            _fetch_arguments_from_hints(initial_hints, out=initial_dir), run_dir,
            events=getattr(observer, "events", None))
        initial_inputs = initial_inputs_from_fetch(
            initial_hints["source"], initial_dir,
            run_dir / "chain" / "initial-inputs.json")
        observer.finish_stage(fetch=initial_report, initial_inputs=str(initial_inputs))

    # -- fetch ---------------------------------------------------------
    observer.enter_stage("fetch", phase="fetch")
    local_snapshot = None
    from woof.source_drivability import local_input_requested
    if local_input_requested(hints):
        from woof.local_preparation import (
            inspect_local_inputs, publish_local_handoff, resolve_source_root)
        from woof.fetch import parse_cycle
        local_source, local_cycle, local_hours = _local_input_hints(hints)
        data_dir = resolve_source_root(
            hints, data_dir=plan.run_options.get("data_dir") or intent.get("data_dir"),
            base_dir=config_path.parent)
        if reviewed_inputs is None:
            snapshot = inspect_local_inputs(
                local_source, data_dir,
                cycle=parse_cycle(local_cycle, local_source),
                hours=local_hours, cadence=hints.get("cadence"),
                start_hour=hints.get("forecast_start_hour", 0),
                supplements=plan.run_options.get("supplement", ()))
        else:
            from woof.local_preparation import verify_local_snapshot
            from datetime import timedelta
            from math import ceil
            from woof.source_adapters import get_source_adapter
            snapshot = reviewed_inputs
            verify_local_snapshot(snapshot)
            cadence = hints.get("cadence") or int(
                get_source_adapter(local_source).forcing_interval_seconds / 3600)
            initial = parse_cycle(local_cycle, local_source) + timedelta(
                hours=hints.get("forecast_start_hour", 0))
            times = [(initial + timedelta(hours=i * cadence)).isoformat()
                     for i in range(ceil(float(local_hours) / cadence) + 1)]
            if (snapshot.get("source") != local_source
                    or Path(snapshot.get("source_root", "")).resolve() != data_dir.resolve()
                    or snapshot.get("cycle") != local_cycle
                    or snapshot.get("valid_times") != times):
                raise PlanError("The reviewed local inputs describe another source or window; restore the reviewed handoff.")
        local_snapshot = snapshot
        handoff_path = publish_local_handoff(snapshot, run_dir / "chain" / "local-inputs")
        fetch_report = {"source": hints["source"], "mode": "local",
                        "network_used": False, "input_sha256": snapshot["sha256"],
                        "file_count": len(snapshot["files"])}
    else:
        from woof.preparation_assets import (analyzed_aerosol_fetch_hints,
                                              wif_fetch_domains)

        if wif_fetch_domains(exp, hints):
            hints = {**hints, "wif": True}
        hints = analyzed_aerosol_fetch_hints(exp, hints)
        # Acquisition publishes complete extended paths on Windows. Keep the
        # same directory spelling when reading its handoff and writing the
        # verified member list, including cache roots beyond MAX_PATH.
        from woof.filesystem_paths import io_path
        data_dir = io_path(data_dir)
        fetch_arguments = _fetch_arguments_from_hints(hints, out=data_dir)
        if _staged_beside_refusal(hints, exp,
                                  separate_initial=initial_inputs is not None) is None:
            # AS POSTED (DESIGN A136 2.5): the fetch runs beside the
            # preparation, which starts on the window's start needs and
            # waits for each later lead's marker; its seal writes the input
            # manifest, and the forecast starts at its head.
            beside = _PostedFetch(fetch_arguments, run_dir,
                                  data_dir=data_dir, hints=hints,
                                  observer=observer)
            events = getattr(observer, "events", None)
            if events is not None:
                from woof.chain_events import HostedPostingRelay

                # The fetch's schedule and leads on the run's stream
                # (posting_schedule, lead_posted, lead_ready), and its
                # start wait (source_wait_* with phase start).
                posting_relay = HostedPostingRelay(events, data_dir=data_dir)
                posting_relay.start(
                    since_unix_ms=int(beside.launched * 1000))
            try:
                posting, posted_handoff = beside.await_handoff()
                if posting is not None and "--author-input-manifest" not in (
                        posted_handoff.get("argv") or ()):
                    # Breakage it prevents: without --as-posted the
                    # preparation would read the fetch folder as a whole
                    # window while its later leads are still posting; the
                    # route handoff always names where the manifest goes,
                    # so one without it is not a route's.  The fetch is
                    # stopped below, not left downloading for a refused run.
                    raise PlanError(
                        f"the as-posted fetch's "
                        f"{fetch_routes.PREP_ARGUMENTS_NAME} names no "
                        "--author-input-manifest, so the preparation beside "
                        "it has nowhere its seal writes the window's "
                        "manifest; run with --whole-cycle, or repair the "
                        "handoff")
            except BaseException:
                fetch_first = beside.said_failure()
                if posting_relay is not None:
                    posting_relay.stop()
                if beside.running():
                    beside.stop()
                _say_posted_source_behind(observer, beside.folder,
                                          primary=fetch_first)
                raise
            if posting is None:
                # The window was here whole (or its source is not waited
                # on): the fetch has ended, and the chain prepares it whole.
                fetch_report = beside.result()
                beside = None
                if posting_relay is not None:
                    posting_relay.stop()
                    posting_relay = None
            else:
                fetch_report = {"source": hints["source"], "as_posted": True,
                                "posting": str(posting),
                                "arguments": list(fetch_arguments)}
        else:
            fetch_report = _run_fetch(
                fetch_arguments, run_dir,
                events=getattr(observer, "events", None), posting_relay=True)
        handoff_path = data_dir / fetch_routes.PREP_ARGUMENTS_NAME
    # The data-store chem sources (HRRR-Smoke's analysed and boundary smoke)
    # are fetched here as the prepared:go chain fetches them (go_cli
    # chem_fetch_command): preparation never downloads, so the HRRR recipe
    # on this staged chain stopped at "Initialize root forcing states" with
    # ChemSourceUnavailable 'hrrr-native-smoke' (a rented box, 2026-10-21 16:32Z).
    _staged_chem_fetch(exp, observer)
    # Sample at the first complete callback's exception, before run_chained
    # waits for a failed forecast's preparation to seal. A later fetch
    # timeout during that hold cannot become the run's primary failure.
    first_failure: dict[str, Any] = {}
    first_failure_lock = threading.Lock()

    def failing(error: BaseException) -> None:
        if beside is not None:
            with first_failure_lock:
                if "fetch_first" not in first_failure:
                    first_failure["fetch_first"] = beside.said_failure()
                    first_failure["error"] = error

    try:
        handoff = (posted_handoff if posting is not None
                   else _read_json_object(handoff_path))
        if handoff.get("schema") != fetch_routes.PREP_ARGUMENTS_SCHEMA:
            raise PlanError(
                f"the fetch left no readable {fetch_routes.PREP_ARGUMENTS_NAME} "
                f"in {data_dir}, so the preparation stage has no bound "
                "argument handoff to compose from; the table routes write "
                "one beside every verified fetch")
        supplements = plan.run_options.get("supplement", ())
        supplied_roles = {binding.split("=", 1)[0] for binding in supplements}
        unfetched = set(handoff.get("unbound_supplement_roles") or []) - supplied_roles
        if unfetched:
            raise PlanError(
                f"the fetch handoff {handoff_path} leaves the supplement "
                f"role(s) {sorted(unfetched)} unbound. Supply each with "
                "woof go CONFIG --supplement ROLE=PATH; the fetch route's notes in "
                f"{fetch_routes.PREP_COMMAND_NAME} say what each one needs")
        from woof.prep_handoff import preparation_arguments
        from woof.forcing_member import verify_handoff, prepare_verified
        arguments = preparation_arguments(handoff)
        # Member staging can change the input tree. Bind the receipt to the
        # selected tree the preparer will consume, never to the prior upstream list.
        bound_handoff = dict(handoff, argv=arguments)
        member_receipt = verify_handoff(hints, bound_handoff, out=data_dir)
        observer.finish_stage(fetch=fetch_report, member_verification=member_receipt)

        # -- prepare -------------------------------------------------------
        observer.enter_stage("prepare", phase="prepare")
        if member_receipt is not None:
            arguments[arguments.index("--input-list") + 1] = member_receipt["input_list"]
        if posting is not None:
            # As posted, the seal writes the manifest beside the fetched files
            # (where the lead markers name them), under this run's own name.
            arguments[arguments.index("--author-input-manifest") + 1] = str(
                _posted_manifest_path(data_dir, prep_root))
            arguments += ["--as-posted", str(posting)]
        elif "--author-input-manifest" in arguments:
            # The standalone prep command owns the fetch folder's manifest.
            # Member staging changes the input paths, so author this chain's
            # manifest beside its preparation instead of overwriting that binding.
            arguments[arguments.index("--author-input-manifest") + 1] = str(
                run_dir / "chain" / "inputs.json")
        arguments += [token for binding in supplements
                      for token in ("--supplement", binding)]
        if initial_inputs is not None:
            arguments += ["--initial-inputs", str(initial_inputs)]
        arguments += [
            # The four paths the handoff declares are the caller's.
            "--wps-namelist", str(namelist),
            "--experiment-config", str(config_path),
            "--geog-root", str(geog_root),
            "--output-root", str(prep_root),
        ]
        from woof.source_cli import preparation_statics
        from woof.static.corridor import config_declares_follow_source

        if config_declares_follow_source(exp):
            # Bind this requirement before reuse is decided. A sealed stationary
            # bundle cannot satisfy the newly requested moving corridor.
            arguments.append(preparation_statics("prepared:staged", source=hints["source"])["option"])
        # Same recovery seam as the HRRR chain, and the same function: this
        # route's preparer is create-only too, so a re-run into the same
        # output_root has to decide before calling it whether the bundle
        # already there is this run's.  The route contributes only what it
        # can state exactly -- the fetch handoff it composed the argv from
        # -- because a member a chain cannot compute must not become a
        # silent pass.
        def verify_local():
            if local_snapshot is not None:
                from woof.local_preparation import verify_local_snapshot
                listing = (Path(arguments[arguments.index("--input-list") + 1])
                           if "--input-list" in arguments else None)
                verify_local_snapshot(local_snapshot, input_list=listing)

        def run_preparation():
            result = _run_prep(arguments)
            # Verify before the stage writes its completion/reuse binding.
            verify_local()
            return result

        verify_local()

        member_notes: dict[str, Any] = {}

        def preparation(built=None):
            return prepare_verified(member_receipt, prep_root, lambda: _prepare_stage(
                prep_root, arguments=arguments,
                stated={}, run=run_preparation,
                built=(None if built is None
                       else lambda receipt: built({**receipt, **member_notes}))),
                notes=member_notes)

        from woof import stage_cli

        # ``prepare_sealed`` is emitted when the preparation seals, from the
        # preparation's own thread, not when the forecast returns: emitted
        # after the forecast it put the seal at the end of the run, so the
        # events of a chained run could not say when its preparation ended.
        seal = _ChainSeal(observer, prep_root)

        def chained_preparation():
            try:
                result = preparation(built=seal.sealed)
                seal.sealed(result)
                return result
            except BaseException as error:
                failing(error)
                raise

        def chained_forecast(head_sha256):
            try:
                return forecast_at_head(head_sha256)
            except BaseException as error:
                failing(error)
                raise

        def forecast_at_head(head_sha256):
            # CHAINED PREPARATION: the forecast starts at the prepared head and
            # the preparation builds the later boundary intervals beside it
            # (woof.ingest.boundary_stream).  ``None`` is a tree published
            # sealed, which runs below exactly as before.
            if head_sha256 is None:
                return None
            observer.finish_stage(prepared_root=str(prep_root),
                                  prepared={"chained": True,
                                            "head_sha256": head_sha256},
                                  member_verification=member_receipt)
            events = getattr(observer, "events", None)
            if events is not None:
                events.emit("prepare_head_ready", head_sha256=head_sha256)
            if posting_relay is not None:
                # The head read every start need: a lead the fetch asks for
                # again from here is not the run's start wait.
                posting_relay.head_ready()
            seal.head_bound()
            head_bundle = stage_cli.resolve_head_bundle(prep_root, head_sha256)
            _staged_forecast_stage(head_bundle)
            return head_bundle

        def _staged_forecast_stage(bundle):
            # A retry owns another generation of the forecast path; the render
            # below reads the one this forecast used.
            nonlocal forecast_dir
            target = forecast_dir = _clear_forecast_output(
                forecast_dir, observer=observer)
            profile = _asserted_profile(plan, config_path=config_path)
            command = stage_cli.sim_command(
                bundle, experiment_config=config_path,
                wps_namelist=namelist if bundle["layout"] == "single" else None,
                outdir=target,
                physics_profile=None if profile is None else str(profile),
                progress_format="jsonl",
                **({"devices": plan.run_options["devices"]}
                   if plan.run_options.get("devices") is not None else {}),
                devices_options=(exp.devices if getattr(
                    getattr(exp, "devices", None), "enabled", False) else None),
                tiles=(exp.tiles if exp.tiles.enabled
                       and bundle["layout"] == "single" else None))
            observer.arm_first_products(
                _chain_render_plan(plan, forecast_dir=target, run_dir=run_dir))
            if beside is not None:
                beside.forecast_begins()
            observer.enter_stage("forecast", phase="forecast")
            _staged_forecast(command[3:], layout=str(bundle["layout"]),
                             observer=observer)

        chained = None
        # `woof prep` runs in this process (_run_prep), so its step events reach
        # the run's stream through one listener around it.
        with _preparation_relay(observer):
            if not prepare_only:
                from woof.ingest.boundary_stream import run_chained

                # One domain or a tree: the forecast starts at the head a
                # chained preparation publishes.  A mapped or GFS-series
                # tree chains on either backend, CPU or card, and the tree
                # runner starts its forecast on that head, a storm-following
                # tree included (its statics corridor is built into the
                # head); a tree published at its seal (native HRRR) runs
                # below.
                prepared, chained = run_chained(
                    prepared_root=prep_root, prepare=chained_preparation,
                    forecast=chained_forecast, observer=observer)
            else:
                prepared = preparation()
    except BaseException as stopped:
        with first_failure_lock:
            first = dict(first_failure)
            fetch_first = (first_failure["fetch_first"]
                           if "fetch_first" in first_failure
                           else beside is not None and beside.said_failure())
        if posting_relay is not None:
            # What the fetch published before the chain stopped reaches the
            # run's stream, and nothing after it does.
            posting_relay.stop()
            posting_relay = None
        if beside is not None:
            _say_posted_source_behind(observer, beside.folder,
                                      primary=fetch_first)
            _settle_posted_fetch(beside, stopped, fetch_first=fetch_first)
            if not fetch_first and first and not _is_interrupt(stopped):
                raise first["error"]
        raise
    if beside is not None:
        # The seal read every lead's marker, so the fetch has published its
        # whole window; it ends here, and a failure it met after that is
        # still the chain's.
        try:
            beside.result()
        finally:
            if posting_relay is not None:
                posting_relay.stop()
                posting_relay = None
    verify_local()
    if chained is not None:
        return _chain_render(plan, forecast_dir=forecast_dir,
                             run_dir=run_dir, observer=observer)
    observer.finish_stage(prepared_root=str(prep_root), prepared=prepared,
                          member_verification=member_receipt)

    # -- forecast ------------------------------------------------------
    # The bundle speaks for itself: `resolve_bundle` reads which source
    # prepared it and whether it is one domain or a tree, and
    # `sim_command` relays the digests the runner re-derives and binds.
    # The experiment config and namelist handed over are the SAME files
    # the preparation consumed -- the mapped proof records their
    # receipts, so the runner's identity check passes on exactly them.
    bundle = stage_cli.resolve_bundle(prep_root)
    if prepare_only:
        return _prepared_chain_result(prep_root, config_path, namelist, bundle=bundle)
    _staged_forecast_stage(bundle)

    # -- render --------------------------------------------------------
    return _chain_render(plan, forecast_dir=forecast_dir, run_dir=run_dir,
                         observer=observer)



def _prepared_chain_result(root: Path, config: Path, namelist: Path | None, *, bundle=None) -> dict:
    """Relay a prepared bundle and its authority paths without forecasting."""
    from woof.stage_cli import resolve_bundle
    from woof.hydro.retention import pin_current
    pin_current(root)
    bundle = resolve_bundle(root) if bundle is None else bundle
    return {'schema': 'gpuwm-preparation-result-v1', 'prepared_root': str(root.resolve()),
            'bundle': bundle, 'experiment_config': str(config.resolve()),
            'wps_namelist': None if namelist is None else str(namelist.resolve()),
            'forecast_started': False}


def _staged_forecast(argv: list[str], *, layout: str,
                     observer: RunObserver) -> None:
    """The prepared forecast runner, hosted in this process.

    In process for the same reason the HRRR chain hosts its forecast:
    the runner's per-step progress and its per-wrfout landing hook
    reach the observer directly, so time-to-first-plot is real.
    """

    from woof.hydro.retention import pin_current
    if "--prepared-root" in argv:
        pin_current(argv[argv.index("--prepared-root") + 1])
    if layout == "tree":
        from woof import prepared_domain_tree_forecast as runner
    else:
        from woof import prepared_single_domain_forecast as runner

    code = runner.main(argv, observer=observer)
    if code:
        raise StageExitError("forecast", code)


def _chain_summary(chain: Path, *,
                   observer: "RunObserver | None" = None) -> dict[str, Any]:
    """A finished chain's completion signals, from its own artifacts.

    The two runners do not leave the same receipts.  The single-domain
    one writes ``progress.json`` and ``report.json``; the tree one
    writes its run receipt and progress under ``evidence/`` plus a capsule.
    Reading only the first pair made a completed nested run report
    ``completed_seconds: 0.0`` and ``status: null`` -- and, because the
    heartbeat is fed from this summary, published a ``complete``
    heartbeat whose model time was zero beside an outer_step of 180.

    ``completed_seconds`` comes from published progress, with the observer
    as a fallback while that progress has not been published.  Nothing is inferred: where a receipt
    does not state a status, this says so rather than assuming a PASS
    from the absence of a failure.

    ``chain`` is the folder the chain was pointed at.  ``woof go`` stamps
    a run folder under it by default (:mod:`woof.run_stamp`) and writes
    ``run/`` and ``png/`` there, so when ``chain/run`` does not exist the
    summary reads the run folder go claimed.  It read ``chain/run``
    alone, and a prepared run that wrote two frames and passed completed
    with ``wrfout_count: 0``, ``status: null`` and ``nan_free: null``.
    """

    if not (chain / "run").is_dir():
        from woof import run_stamp as run_stamp_module

        claimed = run_stamp_module.latest(chain)
        if claimed is not None:
            chain = claimed
    forecast = chain / "run"
    progress = _read_json_object(forecast / "progress.json")
    if not progress:
        progress = _read_json_object(forecast / "evidence" / "progress.json")
    report_path = forecast / "report.json"
    report = _read_json_object(report_path)
    if not report:
        report_path = forecast / "evidence" / "run-receipt.json"
        report = _read_json_object(report_path)
    capsule_path = forecast / "certification-capsule.json"
    capsule = _read_json_object(capsule_path)
    # Readiness JSON receipts repeat the history basename under ready/.
    wrfouts = sorted(path for path in forecast.glob("**/wrfout_*")
                     if path.is_file() and path.suffix != ".json")

    frames = capsule.get("output", {}).get("frames")
    restart_contract = report.get("restart_contract") or {}
    if not isinstance(restart_contract, dict):
        restart_contract = {}
    completed_seconds = progress.get("model_elapsed_seconds")
    if not isinstance(completed_seconds, (int, float)) and observer is not None:
        completed_seconds = observer.last_model_seconds
    status = report.get("status") or progress.get("status")
    return {
        "wrfout_count": (len(wrfouts) if wrfouts
                         else len(frames) if isinstance(frames, list)
                         else int(progress.get("frame_count") or 0)),
        "completed_seconds": _finite_seconds(completed_seconds),
        "nan_free": None if status is None else status == "PASS",
        "status": status,
        "status_basis": (
            "the runner's own report" if status is not None
            else "this runner publishes a certification capsule rather "
                 "than a status report; the capsule's presence is not "
                 "read as a verdict here"),
        "chain_root": str(chain),
        "forecast_root": str(forecast),
        "render_root": str(chain / "png"),
        "report": str(report_path) if report_path.is_file() else None,
        "certification_capsule": (str(capsule_path)
                                  if capsule_path.is_file() else None),
        "restarted": bool(restart_contract.get("restart_input")),
        "restart_input": restart_contract.get("restart_input"),
    }


class _PreparedRunRelay:
    """Bind a native child run and relay its durable facts to its caller.

    The child's original manifest, events and summary remain authoritative.
    Every relayed fact names its exact original line; the caller publishes
    its own heartbeat through the existing supervisor callback.
    """

    def __init__(self, observer: RunObserver, chain: Path, config_path: Path):
        self.observer = observer
        self.chain = chain.resolve()
        self.config_path = config_path.resolve()
        self.config_sha256 = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        self.native = None
        self.summary = None
        self.failure = None
        self.resolved = False

    def attach(self, plan: RunPlan, events: EventStream, manifest_path: Path) -> None:
        from woof.supervisor import atomic_write_json
        run_dir = plan.run_dir.resolve()
        if (self.native is not None or run_dir.parent != self.chain
                or events.path.resolve() != run_dir / EVENTS_FILENAME
                or plan.config_path is None or plan.config_path.resolve() != self.config_path
                or hashlib.sha256(plan.config_bytes()).hexdigest() != self.config_sha256
                or plan.source != f"woof go {self.config_path}"):
            raise PlanError("The prepared native producer does not match this run's owned chain and exact configuration")
        manifest = _read_json_object(manifest_path)
        if (manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("pid") != os.getpid()
                or Path(manifest.get("run_dir", "")).resolve() != run_dir
                or manifest.get("plan_sha256") != plan.sha256):
            raise PlanError("The native producer manifest does not identify this process and plan")
        self.native = {"schema": "gpuwm.native-run-binding.v1", "run_id": manifest["run_id"],
                       "pid": manifest["pid"], "run_dir": str(run_dir),
                       "manifest_path": str(manifest_path.resolve()),
                       "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                       "events_path": str(events.path.resolve()),
                       "progress_path": manifest["progress_path"],
                       "config_source": str(self.config_path), "config_sha256": self.config_sha256}
        parent_manifest = self.observer.events.path.parent / MANIFEST_FILENAME
        parent = _read_json_object(parent_manifest)
        if parent:
            if parent.get("schema") != MANIFEST_SCHEMA or parent.get("pid") != os.getpid():
                raise PlanError("The prepared caller manifest does not identify this process")
            parent["native_run"] = dict(self.native)
            atomic_write_json(parent_manifest, parent)
        events._native_parent_listener = self.receive

    def receive(self, record: Mapping[str, Any], line: str) -> None:
        native_source = {**self.native, "sequence": record["sequence"],
                         "emitted_unix_ms": record["emitted_unix_ms"],
                         "event_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest()}
        event = record["event"]
        fields = {key: value for key, value in record.items() if key not in _ENVELOPE_KEYS}
        observer = self.observer
        if event in {"plan_accepted", "resolved_plan"}:
            if event == "resolved_plan":
                if (record.get("config_sha256") != self.config_sha256
                        or Path(record.get("config_source", "")).resolve() != self.config_path):
                    raise PlanError("The native producer resolved a different configuration")
                self.resolved = True
            observer.events.emit("warning", code="native_producer_" + event,
                message="The prepared route attached its native run." if event == "plan_accepted" else
                        "The native producer resolved the caller's exact configuration.",
                native_source=native_source)
            return
        if event == "failed":
            self.failure = dict(record)
            observer.events.emit("warning", code="native_producer_failed",
                message=record.get("message", "The native producer failed"),
                native_source=native_source, native_failure=fields)
            return
        if not self.resolved:
            # A preparation warning may precede resolution; model/output
            # facts require the exact config binding before they can relay.
            if event != "warning":
                raise PlanError("The native producer reported execution before configuration resolution")
        if event == "completed":
            if record.get("dry_run") is not False or not isinstance(record.get("summary"), dict):
                raise PlanError("The native producer did not publish an executed completion summary")
            self.summary = dict(record["summary"])
            if isinstance(record.get("render_summary"), dict):
                observer._render_summary = dict(record["render_summary"])
            observer.events.emit("warning", code="native_producer_completed",
                message="The native producer completed; the caller is collecting its receipts.",
                native_source=native_source, native_summary=self.summary)
            return
        if event == "stage_started":
            observer._stage = str(record["stage"])
            observer._stage_started_wall = time.perf_counter()
            observer._stage_phases = [record["phase"]] if record.get("phase") else []
            if observer._heartbeat is not None:
                if observer._stage == "finalize":
                    observer._heartbeat.finalizing("native-finalize")
                else:
                    observer._heartbeat.preparing(str(record.get("phase") or record["stage"]))
        elif event == "stage_finished":
            if observer._stage == record.get("stage"):
                observer._stage = None
        elif event == "model_progress":
            observer._last_model_seconds = float(record["model_seconds"])
            observer._progress_events += 1
            if observer._heartbeat is not None:
                beat = _read_json_object(Path(self.native["progress_path"]))
                if (beat.get("run_id") != self.native["run_id"]
                        or beat.get("config_digest") != self.config_sha256
                        or beat.get("pid") != os.getpid()
                        or beat.get("model_elapsed_seconds") != record["model_seconds"]):
                    raise PlanError("The native progress event does not match its own heartbeat")
                observer._heartbeat(model_elapsed_seconds=beat["model_elapsed_seconds"],
                    outer_step=beat["outer_step"], last_durable_wrfout=beat.get("last_durable_wrfout"),
                    last_checkpoint=beat.get("last_checkpoint"), phase=str(record.get("phase") or "native-progress"))
        elif event == "output_committed":
            observer._committed += 1
        elif event == "first_products_ready":
            elapsed = round(time.perf_counter() - observer._accepted_wall, 6)
            observer._first_products_seconds = elapsed
            fields["native_seconds_from_plan_accepted"] = fields.get("seconds_from_plan_accepted")
            fields["seconds_from_plan_accepted"] = elapsed
        observer.events.emit(event, **fields, native_source=native_source)


def _execute_prepared_route(plan: RunPlan, *, exp, data, config_path,
                            observer: RunObserver) -> Mapping[str, Any]:
    """The native/prepared route: ``woof go``'s chain, in this process.

    ``woof go`` IS the documented sequence for this route -- authority,
    fetch, manifest, prepare, forecast, render, with the integrity
    digests relayed between them out of each stage's artifacts.  This
    function does not re-implement one step of it.  It builds the same
    argparse namespace the ``go`` subcommand builds, from this plan, and
    hands ``go_main`` an observer.

    The plan's own ``[fetch]`` block is not used here: ``go`` runs the
    fetch itself, from the ``[fetch]`` hints the wizard wrote into the
    config, which is where the resolved cycle and the sized area
    already live.
    """

    from woof.cli import build_parser
    from woof.go_cli import chain_io_root
    # Every chain below reads and writes under the run folder, from this
    # process and from its children.  A run folder so deep that its run
    # tree would pass the Windows path limit is spelled in its extended
    # form, which opens at any length; `woof go` makes the same choice
    # again for its own folder once it knows whether it downloads.
    run_dir = chain_io_root(plan.run_dir, downloads=False)
    if _existing_prepared_bundle(plan) is not None:
        return _existing_prepared_forecast(
            plan, config_path=Path(config_path), observer=observer, run_dir=run_dir,
            exp=exp)

    from woof.go_cli import go_main

    # Which chain this config is on.  Read from the config's own [fetch]
    # table rather than from the plan, so a config.path plan lands on the
    # same chain its emission targeted.
    raw = tomllib.load(io.BytesIO(Path(config_path).read_bytes()))
    # Through _chain_key, not a second copy of the same test: the
    # follow-statics refusal in resolve_plan was decided for whichever
    # chain that function names, and this is where the naming becomes a
    # dispatch.  One function, so a config cannot be judged as one chain
    # and then run as the other.
    chain = _chain_key(plan.route, (raw.get("fetch") or {}).get("source"))
    # A member-source recipe is not one chain's run: each member is fetched
    # and prepared by ITS source's chain.  `woof go` owns that door for
    # every source (woof.go_cli._go_recipe), so a recipe request takes the
    # go arm below whatever chain the config's own source is on.  Breakage
    # it prevents: the two chains dispatched here fetched and prepared the
    # config's one trajectory and were refused at the forecast stage, where
    # the ensemble session found no member sources.
    from woof.ensemble.runtime_context import current_session

    session = current_session()
    recipe = None if session is None else session.request.recipe
    if chain == "prepared:hrrr" and recipe is None:
        return _hrrr_chain(plan, config_path=Path(config_path), exp=exp,
                           observer=observer, run_dir=run_dir)
    if chain == "prepared:staged" and recipe is None:
        return _staged_chain(plan, config_path=Path(config_path), exp=exp,
                             observer=observer, run_dir=run_dir)

    tokens = ["go", str(config_path), "--outdir",
              str(run_dir / "chain")]
    if plan.run_options.get("devices") is not None:
        tokens += ["--devices", str(plan.run_options["devices"])]
    # Every intent key whose delivery is a `woof go` flag, forwarded.
    # Driven off _INTENT_DELIVERY rather than written out here, so a key
    # that gains a flag is carried by declaring it in one table instead
    # of by remembering to edit this function too.
    intent = plan.config_intent or {}
    for key, delivery in sorted(_INTENT_DELIVERY.items()):
        if not delivery.startswith("go:"):
            continue
        # An explicit run option wins over the intent's copy: it is the
        # later and more specific statement, and it is the only way a
        # config.path plan can say these at all.
        value = plan.run_options.get(key) or intent.get(key)
        if value:
            tokens += [delivery.split(":", 1)[1], str(value)]
    if plan.run_options.get("transport") is not None:
        tokens += ["--transport", str(plan.run_options["transport"])]
    # The posting rule and budget reach this chain's fetch stage the way
    # they reach the other chains' (through _pinned_fetch_hints there):
    # without them run_options.as_posted = false was accepted and read by
    # nothing on this route.  The cycle needs no flag: config_path is
    # already the configuration re-timed to it (:func:`plan_at_cycle`).
    if plan.run_options.get("as_posted") is False:
        tokens.append("--whole-cycle")
    elif plan.run_options.get("late_after_minutes") is not None:
        tokens += ["--late-after-minutes",
                   f"{float(plan.run_options['late_after_minutes']):g}"]
    args = build_parser().parse_args(tokens)
    # Not a `woof go` CLI flag: it is stamped onto the namespace that
    # go_main reads, the same way go_main reads --outdir.  Adding a flag
    # to `woof go` for it is a separate decision about that command's
    # surface, and this front door does not get to make it.
    args.render_products = plan.run_options.get("render_products")
    args.publish_root = plan.run_options.get("publish_root")
    args.publish_cmd = plan.run_options.get("publish_cmd")
    # The line the section products are cut along, stamped the same way;
    # `woof go --section` is the typed spelling of the same value.
    args.render_section = plan.run_options.get("render_section")
    args.grib2 = bool(plan.run_options.get("grib2"))
    # No tree keyword any more: `woof go` itself dispatches a
    # multi-domain config to the tree runner, so this front door and
    # the interactive one now enter the same chain by the same call.
    relay = _PreparedRunRelay(observer, run_dir / "chain", Path(config_path))
    chain_observer = _GoObserver(observer)
    token = _PREPARED_PARENT.set(relay)
    try:
        code = go_main(args, observer=chain_observer)
    finally:
        _PREPARED_PARENT.reset(token)
    if code == INTERRUPT_EXIT_CODE:
        # `woof go` answered a stop: it caught its own interrupt, said
        # so in one sentence and returned 130.  Carried up as the stop
        # it is, so this front door exits 130 too, instead of as a
        # failure of the chain (which is what "exited 130" used to
        # become one frame up, and what the desktop then read back from
        # the worker's receipt as "failed").
        raise ChainInterrupted(chain_observer.current_stage or "the chain")
    if code:
        if chain_observer.failure is not None:
            failure = chain_observer.failure
            detail = (failure.get("diagnostic") or
                      (relay.failure or {}).get("message"))
            raise RuntimeError(
                f"The {failure['stage']} stage failed (exit {failure['exit_code']}). "
                "No later stage ran." + (f"\n{detail}" if detail else ""))
        raise RuntimeError(
            f"`{' '.join(tokens)}` exited {code}; the stage that stopped "
            "the chain is named in the failed event's warning above" +
            (f": {relay.failure['message']}" if relay.failure and relay.failure.get("message") else ""))
    if relay.native is not None:
        if relay.summary is None:
            raise PlanError("The native producer exited without its completion summary")
        return relay.summary
    # The chain's own completion signals, from the artifacts it leaves
    # -- `go`'s standing rule, and the only accurate source here: this
    # function did not integrate anything, the hosted runner did, and it
    # publishes what it finished.
    #
    # These used to be None, and None reached `heartbeat.complete`,
    # whose float() raised INSIDE the try that emits `failed`.  So every
    # successful prepared run ended by announcing failure, after `go`
    # had already printed its validity PASS.  A consumer that trusts the
    # contract marked every good run failed.
    return _chain_summary(run_dir / "chain", observer=observer)


ROUTES: dict[str, Route] = {
    "prepared": Route(
        name="prepared",
        summary="the native prepared-cache route: authority, fetch, "
                "manifest, preparation, forecast and render, in the "
                "documented order (what `woof go CONFIG` executes) -- "
                "for sources the config-driven route cannot decode",
        run_options=frozenset({*_RUN_OPTION_DEFAULTS, "data_dir",
                               "geog_root", "physics_profile",
                               "render_products"}),
        needs_case_data=False,
        execute=_execute_prepared_route),
    "experiment": Route(
        name="experiment",
        summary="the config-driven experiment route: one experiment TOML "
                "with its [case_data] inputs, prepared and integrated in "
                "this process (what `woof run CONFIG` executes)",
        run_options=frozenset(_RUN_OPTION_DEFAULTS)
        - {"data_dir", "physics_profile", "prepared_root", "wps_namelist", "supplement",
           "transport", "as_posted", "late_after_minutes", "cycle", "forecast_tiles",
           # The staged chain's separately fetched analysis; this route reads
           # its [case_data] inputs only, so a plan naming one is refused
           # rather than silently ignored.
           "initial_inputs"},
        execute=_execute_experiment_route),
}


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def _intent_check_grid(intent: Mapping[str, Any]) -> dict[str, Any]:
    """The physics check's grid keys for an intent: its root, and with nests its finest spacing and domain count.

    The root is ``root_dx_km``, or the wizard's own 12 km root (every
    ``ladder``'s, and a bare intent's), so the check reads the root's
    schemes where the run has them rather than at its 3 km probe spacing.
    The finest spacing and domain count come from the ``ladder`` or the
    ``chain``; a single grid, and ``auto`` before its fit chose a depth,
    carry the root alone.
    """

    from woof.domain_wizard import LADDER_RATIOS, ROOT_DX_M, finest_spacing_m

    ladder = intent.get("ladder")
    chain = intent.get("chain")
    if ladder:
        ratios, root_m = LADDER_RATIOS.get(str(ladder), ()), ROOT_DX_M
    else:
        values = chain if isinstance(chain, (list, tuple)) else str(chain or "").split(",")
        ratios = tuple(int(value) for value in values if str(value).strip())
        root_m = float(intent.get("root_dx_km") or ROOT_DX_M / 1000.0) * 1000.0
    root = {"dx_km": root_m / 1000.0}
    if not ratios or min(ratios) < 1:
        return root
    return {**root, "finest_dx_km": finest_spacing_m(root_m, ratios) / 1000.0, "domains": len(ratios) + 1}


def manifest_physics(plan: RunPlan, *,
                     generated_config: str | None = None) -> dict[str, Any]:
    """The physics suite this plan states, as the manifest records it.

    A front end comparing two runs, or a person asking later what a run
    used, reads it here without opening the prepared tree.  Only a
    suite the plan STATES is named: an unstated plan leaves the choice
    to its route's own default (see the preparation stage), and writing
    a default here would be a second answer that could disagree with
    the one the preparer takes.

    ``generated_config`` is the configuration an intent plan resolved
    to.  Schemes picked with no suite change the default at the run's
    finest grid, and that grid is read from this file when given: an
    ``auto`` ladder's depth is chosen by its fit, so before resolution
    its record says the base is pending (:func:`refresh_manifest_physics`
    writes it once the plan resolves).
    """

    intent = plan.config_intent or {}
    profile = plan.run_options.get("physics_profile") or intent.get("physics_profile")
    if not profile and plan.config_kind in ("path", "inline"):
        # A configuration that is already a file states its physics as
        # switches, which is how a mix no named suite matches reaches a
        # run (woof physics-catalog --into).  Recorded from the file.
        try:
            from woof.physics_catalog import experiment_physics
            from woof.physics_registry import registry_sha256

            document = experiment_physics(plan.config_bytes().decode("utf-8"))
            from woof.smoke_receipts import config_receipt
            document.update(config_receipt(plan.config_bytes().decode("utf-8")))
            document["stated_by"] = f"the {plan.config_kind} configuration's own switches"
            document["physics_registry_sha256"] = registry_sha256()
            return document
        except (PlanError, ValueError, KeyError, TypeError) as error:
            return {"suite": None, "stated_by": f"the {plan.config_kind} configuration's own switches",
                    "unresolved": str(error)}
    if intent.get("physics_choices"):
        # Picked schemes over a suite (the physics composer's mix): the
        # plan states the choices, and the check names what they resolve
        # to at the plan's root spacing, and the suite they make, if any.
        choices = intent["physics_choices"]
        document = {"suite": None, "stated_by": "plan", "choices": choices,
                    "base_suite": str(profile) if profile else None}
        try:
            from woof.physics_catalog import check
            from woof.physics_registry import registry_sha256

            request: dict[str, Any] = {"choices": choices, "source": intent.get("source")}
            if profile:
                request["suite"] = str(profile)
            if intent.get("root_dx_km"):
                request["dx_km"] = float(intent["root_dx_km"])
            # With no suite named the choices change the default at the
            # plan's finest grid, which is what the wizard writes them over.
            request.update(_intent_check_grid(intent))
            if generated_config:
                from woof.physics_catalog import experiment_grid

                # The grid the resolved file runs: its root spacing, its
                # finest grid and how many grids the fit gave it.
                request.update(experiment_grid(generated_config))
            elif str(intent.get("ladder") or "") == "auto":
                # On 2.8.1 before this the record read the source's own
                # default here and named a YSU, Noah and KF suite for a
                # fitted 500 m ladder whose file runs MYNN and RUC.
                document["unresolved"] = (
                    "the auto ladder's depth is chosen by its fit, so the "
                    "suite these picks change is recorded once the plan "
                    "resolves")
                return document
            verdict = check(request)
            document["base_suite"] = verdict.get("base_suite")
            document["suite"] = verdict.get("named_suite")
            document["components"] = dict(verdict.get("resolved") or {})
            document["switches"] = dict(verdict.get("changed_from_suite") or {})
            document["physics_registry_sha256"] = registry_sha256()
            if not verdict.get("valid"):
                document["unresolved"] = str(verdict.get("words"))
        except (ValueError, KeyError, TypeError) as error:
            document["unresolved"] = str(error)
        return document
    if not profile:
        return {"suite": None,
                "stated_by": "the route's default for its source" if plan.config_kind == "intent"
                else f"the {plan.config_kind} configuration's own switches"}
    document: dict[str, Any] = {"suite": str(profile), "stated_by": "plan"}
    if intent.get("cumulus") and not plan.run_options.get("physics_profile"):
        # "grid": the suite's root cumulus is off below the
        # convection-permitting spacing, as the generated config says.
        document["cumulus"] = str(intent["cumulus"])
    try:
        from woof.physics_compat import single_domain_runtime_switches
        from woof.physics_registry import physics_registry, registry_sha256

        template = physics_registry()["templates"].get(str(profile)) or {}
        document["components"] = dict(template.get("components") or {})
        document["switches"] = dict(single_domain_runtime_switches(str(profile)))
        document["physics_registry_sha256"] = registry_sha256()
    except (KeyError, ValueError) as error:
        document["unresolved"] = str(error)
    return document


def refresh_manifest_physics(path: Path, plan: RunPlan,
                             generated_config: str | None) -> None:
    """Record an intent plan's physics again from the file it resolved to.

    The manifest is published before resolution, and an intent's picks
    change the default at the grid the wizard fits, which for an ``auto``
    ladder only the fit knows.  Rewritten only when the record changes.
    """

    if plan.config_kind != "intent" or not generated_config:
        return
    from woof.supervisor import atomic_write_json

    physics = manifest_physics(plan, generated_config=generated_config)
    document = _read_json_object(path)
    if not document or document.get("physics") == physics:
        return
    document["physics"] = physics
    atomic_write_json(path, document)


def write_manifest(plan: RunPlan, *, run_dir: Path, events_path: Path,
                   run_id: str, started_at_utc: str) -> Path:
    """Publish the attach manifest before any work starts.

    It names every stream a consumer may want, including the two this
    module does not own: the supervisor's heartbeat and its failure
    capsule.  A front end should never have to know those filenames.
    """

    from woof import proc_identity
    from woof.provenance_gate import receipt_block
    from woof.supervisor import (FAILURE_CAPSULE_NAME,
                                  FAILURE_CAPSULE_SCHEMA, HEARTBEAT_NAME,
                                  HEARTBEAT_SCHEMA, atomic_write_json)

    document = {
        "schema": MANIFEST_SCHEMA,
        "name": plan.name,
        "route": plan.route,
        "run_id": run_id,
        "pid": os.getpid(),
        # The pid alone names a process only until it ends: after a crash
        # or a reboot another program can hold it.  Its creation time (and
        # boot) is what lets a front end tell this run from that program
        # before it reports the run alive or signals it.
        "process": proc_identity.identify(os.getpid()),
        "started_at_utc": started_at_utc,
        # WHICH TREE is executing this plan.  A front end reattaching to
        # a run, or comparing two runs, has to be able to answer that
        # from the manifest alone -- the pid and the run_id say which
        # process, and nothing here said which CODE until this field.
        "provenance": receipt_block(),
        "plan_source": plan.source,
        "plan_sha256": plan.sha256,
        "run_dir": str(run_dir),
        "outputs_dir": str(run_dir),
        "events_path": str(run_dir / EVENTS_FILENAME),
        "events_schema": EVENT_SCHEMA,
        "progress_path": str(run_dir / HEARTBEAT_NAME),
        "progress_schema": HEARTBEAT_SCHEMA,
        "failure_capsule_path": str(run_dir / FAILURE_CAPSULE_NAME),
        "failure_capsule_schema": FAILURE_CAPSULE_SCHEMA,
        "reattach": (
            "read progress_path for CURRENT state, replay events_path "
            "from byte zero for HISTORY, then tail it for live detail; "
            "the heartbeat is the durable anchor, the event stream is "
            "the fine-grained feed"),
        "physics": manifest_physics(plan),
    }
    path = run_dir / MANIFEST_FILENAME
    if plan.config_kind in ("path", "inline"):
        from woof.smoke_receipts import config_receipt
        try:
            config_text = plan.config_bytes().decode("utf-8")
        except PlanError:
            # An absent configuration is the road's to class, inside its event
            # boundary, as a plan defect (tests/test_runplan.py, the execution
            # road); the manifest records no smoke receipt for a file it lacks.
            config_text = None
        if config_text is not None:
            document.update(config_receipt(config_text))
    atomic_write_json(path, document)
    return path


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


#: Remedies for the failure classes whose cause REALLY is known from the
#: class alone: a plan document that failed validation, and a declared
#: input that is not where the config says.  Absent is ``None``, never a
#: guess: a wrong remedy costs a reader more than no remedy.
#:
#: ``ModuleNotFoundError`` is deliberately NOT here any more.  It used
#: to map to the CuPy install line, so a plan run that died on a missing
#: ``wrf``, ``scipy`` or ``shapefile`` told the caller's event stream to
#: install a GPU wheel -- a remedy that is wrong, and wrong in the one
#: channel a front end shows a user verbatim.  Import failures are
#: answered by :func:`woof.capabilities.remedy_for_error`, which reads
#: the MODULE the failure names.
#:
#: The ``PlanError`` line is for the plan document.  A refusal whose fix
#: lies elsewhere (an input the computer has to set up once) carries its
#: own ``remedy`` or states one in its message, which :func:`_remedy`
#: reads first.
_REMEDIES = {
    "PlanError": "fix the plan document and re-run; nothing was started",
    "FileNotFoundError": "a declared input is not at the path the config "
                         "names; `woof check CONFIG` names all of them",
}


def _remedy(error: BaseException) -> str | None:
    """The remedy for a failed plan run: what is missing, then the class.

    Order matters.  An import failure is asked about FIRST and answered
    from the module it names, so the class table can never speak for a
    dependency it cannot see.
    """

    from woof import capabilities

    derived = capabilities.remedy_for_error(error)
    if derived is not None:
        return derived
    # A failure that knows its own remedy carries it (the fetch
    # backbone's do); a failed event used to say `remedy: null` for a
    # network cut-off the reader could act on.
    carried = getattr(error, "remedy", None)
    if isinstance(carried, str) and carried:
        return carried
    stated = stated_remedy(str(error))
    if stated is not None:
        return stated
    return _REMEDIES.get(type(error).__name__)


def stated_remedy(message: str) -> str | None:
    """The ``  remedy: <command>`` a refusal writes in its action half, with what qualifies it.

    Refusals across the engine end their action half with that line and
    give alternatives as ``  # ...`` comments below it (a missing
    ``gfs_grib2_bridge``: ``remedy: woof setup``).  The ``failed``
    event's ``remedy`` was null for every one of them, so the web page
    showed the refusal's first line and nothing to do.  Lines indented
    deeper than the label continue it; a blank line or the next label
    ends it.  ``None`` when the message states no remedy.  ``woof go``
    reads it too, so it does not print a remedy its refusal already gave.
    """

    lines = _split_message(message)[0].splitlines()
    for index, line in enumerate(lines):
        body = line.strip()
        if not body.lower().startswith("remedy:"):
            continue
        indent = len(line) - len(line.lstrip())
        kept = [body[len("remedy:"):].strip()]
        for follow in lines[index + 1:]:
            deeper = len(follow) - len(follow.lstrip()) > indent
            if not follow.strip() or not (deeper or follow.strip().startswith("#")):
                break
            kept.append(follow.strip())
        return "\n".join(part for part in kept if part) or None
    return None


def missing_inputs_refusal(missing: Sequence[Mapping[str, Any]], *,
                           before_download: bool = False) -> PlanError:
    """The refusal for declared inputs that are not on disk, with its remedy.

    ``missing`` is :func:`declared_inputs` rows.  The remedy says what to
    do for each kind that is missing, and the refusal carries it, so the
    ``failed`` event's ``remedy`` is that and not the plan-document line:
    a geography tree nobody set up on this computer is not fixed by
    editing the plan, and the page told a user whose tree was missing to
    do exactly that.  The action half names the same next step, so a
    terminal reader gets it without ``--explain``.
    """

    from woof.geog_assets import WRF_FETCH_COMMAND

    roles = sorted({str(entry["role"]) for entry in missing})
    steps = []
    if "geog_root" in roles:
        steps.append("set up the geography data once on this computer "
                     f"with {WRF_FETCH_COMMAND}")
    others = [role for role in roles if role not in ("geog_root", "forcing")]
    if others:
        one = len(others) == 1
        steps.append("put " + ", ".join(others) + " where the config's "
                     f"[case_data] names {'it' if one else 'them'}, or point "
                     f"[case_data] at where {'it is' if one else 'they are'}")
    if "forcing" in roles:
        steps.append("put the forcing files where [case_data] names them, "
                     "or give the plan a [fetch] block that downloads them")
    remedy = "; ".join(steps)
    remedy = remedy[:1].upper() + remedy[1:] + "."
    what = ("declared input(s) this run needs are not on disk"
            + (", and the download does not supply them" if before_download
               else "")
            + ": " + ", ".join(f"{entry['role']} {entry['path']}"
                               for entry in missing) + ".")
    why = ("Refused before the download, so nothing was spent." if before_download
           else "The config names them in [case_data].  A plan with a "
                "[fetch] block downloads its own; without one, the data "
                "has to be there before the run starts.")
    return PlanError(layered(what + "\n  remedy: " + remedy, why),
                     remedy=remedy)


@ledger_main
def execute_plan(plan: RunPlan, *, events: EventStream) -> int:
    """Run one plan to completion, or to its ``failed`` event.

    Returns the process exit code: 0 when a ``completed`` event was the
    last line, nonzero when a ``failed`` one was.
    """

    from woof.supervisor import (HEARTBEAT_NAME, RuntimeHeartbeat,
                                  utc_now)

    run_dir = plan.run_dir
    _validate_prepared_output(
        plan, require_empty=True,
        # The stream's owner file is this launch's own claim on the
        # folder, written by the stream before this check can run.
        launch_files=(Path(events.path), event_owner_path(Path(events.path)),
                      run_dir / "launch.log", run_dir / HEARTBEAT_NAME))
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at_utc = utc_now()
    run_id = hashlib.sha256(
        f"{plan.sha256}:{started_at_utc}:{os.getpid()}".encode("utf-8")
    ).hexdigest()[:16]
    os.environ.setdefault("WOOF_RUN_ID", run_id)
    bind_ledger(run_dir)
    manifest_path = write_manifest(
        plan, run_dir=run_dir, events_path=events.path, run_id=run_id,
        started_at_utc=started_at_utc)
    native_parent = _PREPARED_PARENT.get()
    if native_parent is not None:
        native_parent.attach(plan, events, manifest_path)

    from woof.provenance_gate import receipt_block

    events.emit(
        "plan_accepted", name=plan.name, route=plan.route,
        plan_source=plan.source, plan_sha256=plan.sha256,
        run_dir=str(run_dir), manifest_path=str(manifest_path),
        events_path=str(events.path), pid=os.getpid(), run_id=run_id,
        # The first line of the stream names the tree, so a consumer
        # that only ever tails events.jsonl never has to open the
        # manifest to learn which code produced what follows.
        provenance=receipt_block())
    # Time to first plot is measured from HERE -- the instant this run
    # was accepted -- because that is the instant the person who launched
    # it started waiting.  Taken immediately after the event so the two
    # cannot drift by whatever the next few resolution steps cost.
    accepted_wall = time.perf_counter()

    stage = "preflight"
    observer: RunObserver | None = None
    try:
        # BEFORE the fetch, the resolve and the device.  A plan run that
        # cannot import the runtime it is about to integrate on must say
        # so on the first line of its event stream, not after it has
        # spent the caller's bandwidth -- and it must say so with the
        # remedy, because the caller here is usually a program relaying
        # our words to somebody else.
        #
        # `dry_run` is exempt by the same rule `woof go --dry-run` is:
        # it resolves the plan and stops before any device work, so it
        # is exactly the thing a reader with no runtime should be able
        # to run.
        if not plan.run_options.get("dry_run"):
            from woof import capabilities

            capabilities.require(
                "woof run-plan", capabilities.GPU_RUNTIME,
                before=("Refusing at plan acceptance, before the fetch "
                        "stage downloads anything and before a device is "
                        "selected."))
        stage = "prepare"
        device = plan.run_options.get("device")
        if device is not None:
            # Before anything can import cupy: the mask is read at
            # context creation, so setting it afterwards would select
            # nothing while appearing to.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

        # run_options.cycle (`woof go --cycle`): the configuration is
        # re-timed into the run directory before anything reads it, and
        # `latest` there is answered once, like the fetch's own below.
        plan = plan_at_cycle(plan, run_dir)
        # A `latest` cycle is resolved BEFORE resolution reports, so
        # the resolved_plan event carries the concrete cycle rather than
        # the question the caller asked.
        fetch_arguments = plan.fetch_arguments
        cycle_resolutions: list[dict[str, Any]] = []
        if fetch_arguments is not None:
            stage = "fetch"
            fetch_arguments, cycle_resolutions, cycle_warnings = \
                resolve_fetch_cycle(fetch_arguments)
            for warning in cycle_warnings:
                events.emit("warning", **warning)
        stage = "prepare"

        # An intent plan generates into the run directory: the config
        # and the WPS namelist its [case_data] names are inputs to this
        # run, not scratch.
        # Resolved WITHOUT requiring the inputs to be on disk.  They may
        # not be yet: the fetch stage below is what puts them there, and
        # a plan that fetches its own forcing would otherwise be refused
        # for the absence of the thing it was about to download.  The
        # gate still happens -- after the fetch, before the model.
        resolution, exp, data = resolve_plan(
            plan, generate_into=run_dir, require_inputs=False)
        # The physics record reads the grid the plan resolved to.
        refresh_manifest_physics(manifest_path, plan,
                                 resolution.get("generated_config"))
        if fetch_arguments is None:
            fetch_arguments = declared_forcing_fetch(
                _config_for_declared_fetch(plan, resolution, run_dir), data,
                base_dir=None, cache_root=run_dir)
            if fetch_arguments is not None:
                cycle_resolutions.append({"scope": "fetch", "key": "args",
                    "value": fetch_arguments, "basis": "configuration.fetch"})
        data, rebound = rebind_fetched_forcing(data, fetch_arguments)
        if rebound is not None:
            cycle_resolutions.append(rebound)
        for warning in resolution["warnings"]:
            events.emit("warning", code="library_warning",
                        message=warning["action"], detail=warning["why"])
        events.emit(
            "resolved_plan",
            configuration=resolution["configuration"],
            automatic_resolutions=(
                list(resolution["automatic_resolutions"])
                + cycle_resolutions),
            generated_config=resolution["generated_config"],
            config_kind=plan.config_kind,
            config_sha256=resolution["plan"]["config_sha256"],
            config_source=resolution["plan"]["config_source"],
            domain_size_floor=resolution["domain_size_floor"],
            declared_inputs=resolution["declared_inputs"],
            inputs_present=resolution["inputs_present"],
            run_options=dict(plan.run_options))

        # BEFORE the fetch, and before a dry run reports the plan as
        # runnable: members that would all run this plan's one prepared
        # input are refused while nothing has been spent.
        _refuse_one_input_ensemble(
            plan, _config_for_declared_fetch(plan, resolution, run_dir))

        if plan.run_options.get("dry_run"):
            events.emit(
                "completed", dry_run=True, run_dir=str(run_dir),
                receipt_path=str(manifest_path),
                summary={"executed": False,
                         "reason": "run_options.dry_run resolved the plan "
                                   "and stopped before any device work"})
            return 0

        # BEFORE the fetch: a run that cannot finish on this disk, or that
        # lacks an input no download supplies, is refused while nothing has
        # been spent.  The geography tree was found missing only after a
        # nine minute ERA5 download, and a 1 km run filled its disk at hour
        # nine of twelve.
        if fetch_arguments is not None and data is not None:
            unfetched = [entry for entry in declared_inputs(data)
                         if not entry["present"] and entry["role"] != "forcing"]
            if unfetched:
                raise missing_inputs_refusal(unfetched, before_download=True)
        from woof.resume import KEEP_CHECKPOINTS_ENV
        keep = int(plan.run_options.get("keep_checkpoints") or 0)
        os.environ[KEEP_CHECKPOINTS_ENV] = str(keep)
        # The one disk admission, the frame stream the preparation stages
        # in its compose scratch folder included; a stream that may not
        # fit is said here, before the download, and the run goes on.
        def scratch_caution(message: str, detail: str, folder: str | None) -> None:
            events.emit("warning", code="compose_scratch_may_not_fit",
                        message=message, detail=detail, folder=folder)

        refusal = disk_admission_refusal(
            plan, exp, raw=_config_for_declared_fetch(plan, resolution, run_dir),
            data=data, fetch_arguments=fetch_arguments, run_dir=run_dir,
            warn=scratch_caution)
        if refusal is not None:
            error = PlanError(refusal)
            if refusal.folders:
                error.folders = refusal.folders
            raise error

        from woof.hydro.runtime import pending_launch_args, rebind_launch
        if (hydro_args := pending_launch_args()) is not None:
            rebind_launch(hydro_args, admitted=True)

        heartbeat = RuntimeHeartbeat(
            run_dir / HEARTBEAT_NAME, run_id=run_id,
            config_sha256=resolution["plan"]["config_sha256"],
            started_at_utc=started_at_utc)
        observer = RunObserver(
            events, heartbeat=heartbeat, root_domain=exp.root.grid_id,
            accepted_wall=accepted_wall)
        if plan.run_options.get("grib2"):
            observer.arm_grib2(run_dir / "grib2")
        from woof.live_columns import columns_enabled
        if columns_enabled(plan.run_options):
            observer.arm_columns(run_dir / "columns", experiment=exp)
        heartbeat.starting()

        if fetch_arguments is not None:
            stage = "fetch"
            observer.enter_stage("fetch")
            observer.finish_stage(
                fetch=_run_fetch(fetch_arguments, run_dir, events=events,
                                 posting_relay=True))

        # The gate the resolution above deferred.  Everything the config
        # declares must be on disk before the model starts; whatever was
        # going to supply it has now run.  Named in one refusal rather
        # than discovered one file at a time inside preparation.
        # Only where there is a [case_data] block to gate on.  The
        # prepared route's config declares no inputs -- its chain
        # fetches and binds them itself, and each of its stages refuses
        # what the previous one did not produce.
        missing = [] if data is None else [
            entry for entry in declared_inputs(data)
            if not entry["present"]]
        if missing:
            raise missing_inputs_refusal(missing)

        cycle_receipt = None
        if plan.run_options.get("input_cycle") is not None:
            from woof.input_cycle import verify
            from woof.supervisor import atomic_write_json

            cycle_receipt = verify(
                plan.run_options["input_cycle"],
                prepared_root=plan.run_options.get("prepared_root"),
                restart=plan.run_options.get("restart"), data=data,
                launch_start=exp.start_time)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["input_cycle"] = cycle_receipt
            atomic_write_json(manifest_path, manifest)

        # The route is handed a config that is a FILE.  `woof go` takes
        # a path, the prepared chain binds that path's digest into every
        # stage, and an inline config has nowhere to be one -- so it is
        # materialized here, into the run directory, where it is also
        # the run's own provenance afterwards.
        config_path = Path(resolution["plan"]["config_source"])
        if not config_path.is_file():
            config_path = run_dir / GENERATED_CONFIG_NAME
            config_path.write_bytes(plan.config_bytes())
            events.emit(
                "warning", code="inline_config_materialized",
                message=f"the inline config was written to {config_path} "
                        "because this route binds a config file by path",
                path=str(config_path))

        stage = "prepare"
        # The pipeline opens prepare/initialize/forecast itself, through
        # the phases it already reports; the observer maps them.  Only
        # finalize is this front door's own, because the pipeline has no
        # word for it.
        from woof.ensemble.door import request_for_config, production_run_scope
        from woof.verification_visuals import verification_scope
        ensemble_request = request_for_config(config_path,
            override=plan.run_options.get("ensemble"))
        with _kernel_compile_relay(observer), verification_scope(
                plan.run_options.get("verify_visuals", True)), production_run_scope(
                ensemble_request, output_directory=run_dir):
            summary = ROUTES[plan.route].execute(
                plan, exp=exp, data=data, config_path=config_path,
                observer=observer)
        if cycle_receipt is not None:
            summary = {**summary, "input_cycle": cycle_receipt}

        stage = "finalize"
        observer.enter_stage("finalize")
        grib2_summary = observer.stop_grib2()
        # Every grid's last frame too when the run ended off the hour; the files are already on disk.
        observer.stop_columns()
        receipts = _receipts(run_dir)
        observer.finish_stage(receipts=receipts)
        heartbeat.complete(_finite_seconds(
            summary.get("completed_seconds")))

        events.emit(
            "completed", dry_run=False, run_dir=str(run_dir),
            receipt_path=receipts.get("certification_capsule",
                                      str(manifest_path)),
            receipts=receipts,
            outputs_committed=observer.outputs_committed,
            # Time to first plot, repeated here from
            # `first_products_ready` so the headline number is on the
            # line every reader already reads.  Null when this run
            # published no products early.
            first_products_seconds=observer.first_products_seconds,
            summary=dict(summary),
            **({"grib2_summary": grib2_summary} if grib2_summary is not None else {}),
            **({"render_summary": observer._render_summary} if observer._render_summary is not None else {}))
        return 0
    except BaseException as error:  # noqa: BLE001 - every exit is an event
        interrupted = _is_interrupt(error)
        if observer is not None:
            stage = observer.stage or stage
            # A stopped or failed run stops drawing too; what is drawn
            # stays (the did-not-finish contract keeps pictures).  A run
            # the user stopped draws nothing more: finishing the queue
            # could hold it for minutes, and the desktop kills it 5 s
            # after asking.
            try:
                observer.stop_grib2(halt=interrupted)
            except Exception:  # the failure is already carried by the event
                pass
            try:
                observer.stop_live_products(halt=interrupted)
            except Exception:  # noqa: BLE001 - the failure is the event
                pass
            try:
                observer.stop_columns(halt=interrupted)
            except Exception:  # noqa: BLE001 - the failure is the event
                pass
            observer.finish_stage(outcome="failed")
            observer.failed()
        # The folders a refusal names as the place to act (the scratch
        # folder a frame stream did not fit in): a page that hides machine
        # paths still shows these, or its remedy names nowhere.
        folders = [str(folder) for folder in getattr(error, "folders", ()) or ()]
        # A source lead later than its budget ends the run with the lead
        # named and exit 75 (the same launch succeeds once it posts); every
        # other failure keeps exit 1.  The record comes from the forecast
        # (the observer heard it) or from the error itself.
        from woof.ingest.boundary_stream import (
            SOURCE_BEHIND_EXIT_CODE, SourceBehind, source_behind_sentence,
        )
        behind = (None if interrupted or observer is None
                  else getattr(observer, "source_behind_record", None))
        if (behind is None and not interrupted
                and isinstance(error, SourceBehind)):
            behind = dict(error.details)
        error_class = type(error).__name__
        message = str(error)
        exit_code = (INTERRUPT_EXIT_CODE if interrupted
                     else getattr(error, "exit_code", None))
        if behind is not None:
            events.emit("source_behind", **source_behind_fields(behind))
            error_class, exit_code = "SourceBehind", SOURCE_BEHIND_EXIT_CODE
            message = source_behind_sentence(behind)
        events.emit(
            "failed", stage=stage, error_class=error_class,
            message=message, run_dir=str(run_dir),
            exit_code=exit_code,
            interrupted=interrupted,
            remedy=_remedy(error),
            receipts=_receipts(run_dir),
            **({"secondary_source_behind":
                dict(observer.secondary_source_behind_record)}
               if observer is not None and
               observer.secondary_source_behind_record is not None else {}),
            **({"folders": folders} if folders else {}))
        if interrupted:
            return INTERRUPT_EXIT_CODE
        if behind is not None:
            return SOURCE_BEHIND_EXIT_CODE
        # A prepared forecast refused before integration is an argument/input
        # refusal, not a failed forecast. Other chain failures keep their
        # established exit 1; their stage status remains in the failed event.
        if isinstance(error, StageExitError) and error.stage == "forecast" and exit_code == 2:
            return 2
        from woof.io.restart import RestartMismatchError

        if isinstance(error, RestartMismatchError):
            return 2
        return 1


#: The shell's 128 + SIGINT, the exit code every long-running command
#: here answers a stop with (``woof.go_cli.INTERRUPT_EXIT_CODE`` is the
#: same number; spelled here so this module does not import that one to
#: read it).
INTERRUPT_EXIT_CODE = 130


def _is_interrupt(error: BaseException) -> bool:
    """Whether ``error`` is the user's stop rather than a failure.

    Three spellings of one gesture reach :func:`execute_plan`.  A
    ``KeyboardInterrupt`` is the Ctrl-C that lands between stages.
    ``woof.go_cli.GoInterrupted`` is the one that lands while
    ``run_stage`` is waiting on a stage subprocess, which is where a
    stop during fetch, prepare, forecast or render always lands: the
    whole foreground process group received the signal, this process
    included.  A ``StageExitError`` carrying 130, or the negative of
    SIGINT, is a stage that answered the same signal itself before this
    process observed its own.  All three exit 130.

    The breakage this names: only the first spelling used to exit 130,
    the other two exited 1, and the desktop's saved-run reader calls
    130 "stopped" and every other nonzero code "failed".  A forecast
    stopped during its render was therefore offered for downscaling
    while the launcher still held the job (it knew it had asked for the
    stop) and withdrawn, with no reason on the row, the next time the
    desktop opened and read the run back from its receipts.
    """

    if isinstance(error, KeyboardInterrupt):
        return True
    code = getattr(error, "exit_code", None)
    if not isinstance(code, int):
        return False
    if code == INTERRUPT_EXIT_CODE:
        return True
    import signal
    return code < 0 and -code == int(signal.SIGINT)


def _cycle_spacing_hours(grid, cycle) -> int | None:
    """Hours from ``cycle`` to the next init on ``grid``, or ``None``.

    The spacing a producer actually runs at, read off its own hour set
    rather than assumed to be one of two numbers.  ICON-EU's three
    hours and GEM-GDPS's twelve are both real, and a shared constant was
    wrong for both.
    """

    if grid is None:
        return None
    hours = sorted(grid.hours)
    if len(hours) == 1:
        return 24
    following = next((hour for hour in hours if hour > cycle.hour), None)
    if following is None:
        return 24 - cycle.hour + hours[0]
    return following - cycle.hour


def _latest_cycle_note(source: str) -> str:
    """How ``latest`` was answered for this source, in this source's terms.

    Two mechanisms, and a note that claimed the wrong one is worse than
    no note: a reader shown "probed the mirrors" for a source with no
    mirrors to probe would go looking for a network step that never
    happened.  The mechanism is read off the same predicate the resolver
    dispatches on, so the two can never disagree.
    """

    from woof.fetch import cycle_is_probeable

    if cycle_is_probeable(source):
        return ("`latest` probed the mirrors for the newest cycle "
                "complete through the end of the requested window; the "
                "concrete cycle is recorded so this run is reproducible")
    return ("`latest` resolved from this source's declared publication "
            "delay -- it publishes no object a probe can ask for -- so "
            "the cycle is the newest its registry row says exists; the "
            "concrete cycle is recorded so this run is reproducible")


def _parse_fetch_arguments(arguments: Sequence[str]):
    """``fetch.args`` read by the real ``woof fetch`` parser."""

    from woof.cli import parse_fetch_arguments

    try:
        return parse_fetch_arguments(arguments)
    except SystemExit as stop:
        raise PlanError(layered(
            "run plan 'fetch.args' is not a valid `woof fetch` "
            "argument list; argparse refused it above.",
            "The list is handed to woof's own fetch parser verbatim, "
            "so anything `woof fetch` accepts is accepted here and "
            "nothing else is.")) from stop


def _with_fetch_cycle(arguments: Sequence[str], cycle: str) -> list[str]:
    """``arguments`` with every spelling of ``--cycle`` set to ``cycle``.

    Split (``--cycle latest``), joined (``--cycle=latest``) and the
    abbreviations argparse accepts (``--cyc latest``) are all rewritten,
    so a repeated option cannot leave an unresolved ``latest`` behind for
    argparse's last-one-wins rule to pick.
    """

    def names_cycle(token: str) -> bool:
        name = token.split("=", 1)[0]
        return len(name) >= 4 and "--cycle".startswith(name)

    rewritten: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == "--":
            rewritten.extend(arguments[index:])
            break
        if names_cycle(token):
            name = token.split("=", 1)[0]
            if "=" in token:
                rewritten.append(f"{name}={cycle}")
            else:
                rewritten.extend((name, cycle))
                index += 1
            index += 1
            continue
        rewritten.append(token)
        index += 1
    return rewritten


#: Where a run re-timed by ``run_options.cycle`` keeps its configuration
#: and the namelists beside it, inside the run directory.
CYCLE_CONFIG_DIRNAME = "cycle-config"


def plan_at_cycle(plan: RunPlan, destination: Path) -> RunPlan:
    """``plan`` at ``run_options.cycle`` (``woof go --cycle``), or as it is.

    ``latest`` is resolved once, here, by the fetch's own resolver under
    the plan's posting rule, and the concrete cycle is recorded in
    ``automatic_resolutions``.  An intent plan hands it to the wizard as
    its ``--cycle``.  A configuration named by path is re-timed by
    :func:`woof.companion_setups.retime_to_cycle` (start time, delayed
    nests, the WPS and route namelists) into
    ``destination/cycle-config/``, which is the run directory for a run
    and a scratch folder for a query, and the plan continues on that file;
    ``fetch.args`` a plan carries get the same cycle.

    Refused, naming the breakage: an inline configuration (there is no
    file or namelist beside it to re-time, so the fetch would move and
    the forecast would not).
    """

    cycle = plan.run_options.get("cycle")
    if cycle is None:
        return plan
    if plan.config_intent is not None:
        intent = dict(plan.config_intent)
        intent["cycle"] = cycle
        return dataclasses.replace(plan, config_intent=intent)
    if plan.config_path is None:
        raise PlanError(
            "run plan 'run_options.cycle' re-times the configuration file and "
            "the namelists beside it, and an inline configuration has no "
            "file or namelist to re-time, so its fetch would move and its "
            "forecast would not. Next: name the configuration by "
            "'config.path', or write the cycle into it.")
    from woof.fetch import parse_cycle
    from woof.go_cli import config_fetch_request

    document = tomllib.loads(plan.config_bytes().decode("utf-8-sig"))
    hints = document.get("fetch")
    if not isinstance(hints, dict) or not {"source", "cycle"} <= hints.keys():
        raise PlanError(
            f"run plan 'run_options.cycle' names a cycle, and {plan.config_path} "
            "has no [fetch] source and cycle to move it from")
    source = str(hints["source"])
    resolutions: list[dict[str, Any]] = []
    concrete = str(cycle)
    if concrete == "latest":
        clock_hints = config_fetch_request(document)
        if hints.get("initial_source"):
            # The forecast clock follows its initial analysis, even when
            # the lateral source has newer cycles available.
            clock_hints = {"source": hints["initial_source"], "cycle": "latest",
                           "hours": 0, "as_posted": False}
        arguments = _fetch_arguments_from_hints(
            _pinned_fetch_hints(plan, clock_hints),
            out=Path("latest"))
        arguments, resolutions, _warnings = resolve_fetch_cycle(arguments)
        concrete = _parse_fetch_arguments(arguments).cycle
    try:
        moment = parse_cycle(concrete, source)
        own = parse_cycle(str(hints["cycle"]), source)
    except ValueError as error:
        raise PlanError(f"run plan 'run_options.cycle': {error}") from error
    options = dict(plan.run_options)
    options["cycle"] = concrete
    fetch_arguments = plan.fetch_arguments
    if fetch_arguments is not None:
        fetch_arguments = tuple(_with_fetch_cycle(list(fetch_arguments), concrete))
    config_path = plan.config_path
    if moment == own:
        note = "the configuration's own cycle; nothing was re-timed"
    else:
        from woof.companion_setups import retime_to_cycle

        out = Path(destination) / CYCLE_CONFIG_DIRNAME / plan.config_path.name
        try:
            result = retime_to_cycle(plan.config_path, moment, out)
        except ValueError as error:
            raise PlanError(
                f"run plan 'run_options.cycle' = {concrete}: {error}") from error
        config_path = Path(result["config_path"])
        note = (f"{plan.config_path} re-timed to start {result['start_time']} "
                f"as {config_path}, with its namelists rendered again")
    resolutions.append({"scope": "fetch", "key": "cycle", "value": concrete,
                        "basis": "run_options.cycle", "note": note})
    return dataclasses.replace(
        plan, config_path=config_path, run_options=options,
        fetch_arguments=fetch_arguments,
        automatic_resolutions=tuple(plan.automatic_resolutions) + tuple(resolutions))


def plan_readiness(plan: RunPlan, *, no_probe: bool = False
                   ) -> tuple[dict[str, Any], int]:
    """``woof run-plan --readiness``: the window this plan's fetch asks for.

    The plan's own ``fetch.args`` when it carries them, otherwise the
    configuration's ``[fetch]`` table with the run options over it, read
    by the fetch command's own parser, so the answer is about the fetch
    the run would start.
    """

    from woof.fetch import readiness_for_fetch

    if plan.run_options.get("prepared_root") is not None:
        from woof.stage_cli import readiness_for_bundle

        if plan.config_path is None:
            raise PlanError("Prepared readiness requires its bound config file.")
        return readiness_for_bundle(Path(plan.run_options["prepared_root"]),
            experiment_config=plan.config_path,
            wps_namelist=plan.run_options.get("wps_namelist"),
            physics_profile=plan.run_options.get("physics_profile"),
            restart=plan.run_options.get("restart"), devices=plan.run_options.get("devices"))
    arguments = (list(plan.fetch_arguments)
                 if plan.fetch_arguments is not None else None)
    cycle = plan.run_options.get("cycle")
    if arguments is not None and cycle is not None:
        # run_options.cycle wins over the plan's own fetch.args, as it
        # does when the plan runs (:func:`plan_at_cycle`).
        arguments = _with_fetch_cycle(arguments, str(cycle))
    if plan.config_intent is not None and cycle is not None:
        plan = dataclasses.replace(
            plan, config_intent={**plan.config_intent, "cycle": cycle})
    if arguments is None:
        if plan.config_intent is None:
            text = plan.config_bytes().decode("utf-8-sig")
        else:
            resolution, _exp, _data = resolve_plan(plan, require_inputs=False)
            text = str(resolution.get("generated_config") or "")
        document = tomllib.loads(text)
        hints = document.get("fetch")
        if not isinstance(hints, dict) or not {"source", "cycle"} <= hints.keys():
            # Breakage it prevents: a plan with no download would be
            # answered for a window nobody fetches.
            raise PlanError(
                "--readiness answers for the window a plan fetches, and this "
                "plan's configuration has no [fetch] source and cycle")
        if plan.route == "prepared" and (_plan_recipe(plan, document) is not None
                                         or _go_plans_members(plan, document)):
            # A recipe fetches one window per member: the answer is for
            # all of them, from the reader `woof go --readiness` uses.
            # A plain member count on the go chain is the same run (the
            # source's operational ensemble).  Breakage it prevents: it
            # was answered for the config's own source, ready at one time
            # for a run whose members post at another.
            from woof.domain_wizard import experiment_from_text
            from woof.ensemble.door import request_for_payload
            from woof.go_cli import recipe_readiness

            try:
                request = request_for_payload(
                    text.encode("utf-8"), override=plan.run_options.get("ensemble"))
                return recipe_readiness(
                    request, document,
                    experiment_from_text(text, source=str(plan.config_path or plan.source)),
                    cycle=None if cycle is None else str(cycle),
                    posting={key: plan.run_options[key]
                             for key in ("as_posted", "late_after_minutes")
                             if plan.run_options.get(key) is not None},
                    transport=plan.run_options.get("transport"), no_probe=no_probe)
            except ValueError as error:
                raise PlanError(f"--readiness: {error}") from error
        arguments = _fetch_arguments_from_hints(
            _pinned_fetch_hints(plan, hints), out=Path("readiness"))
        from woof.go_cli import _local_input_readiness

        local = _local_input_readiness(hints)
        if local is not None:
            return local
    parsed = _parse_fetch_arguments(arguments)
    try:
        return readiness_for_fetch(parsed, no_probe=no_probe)
    except ValueError as error:
        raise PlanError(f"--readiness: {error}") from error


def resolve_fetch_cycle(arguments: Sequence[str]
                        ) -> tuple[list[str], list[dict[str, Any]],
                                   list[dict[str, Any]]]:
    """Turn a ``--cycle latest`` into the concrete cycle it means.

    ``woof fetch`` resolves ``latest`` itself, and correctly.  It is
    done HERE anyway, before the fetch runs, for one reason: a plan that
    records ``latest`` records a QUESTION, and the answer changes every
    six hours.  Resolving once, up front, and writing the concrete cycle
    into ``automatic_resolutions`` is what makes the receipt reproducible
    -- and it removes the window in which this front door reports one
    cycle while the fetch downloads the next.

    It is the same rule the wizard already applies to its emitted
    ``[fetch]`` table: "the RESOLVED cycle, never the literal 'latest'"
    (woof/domain_wizard.py:3210).

    ``latest`` is matched case-insensitively.  ``woof fetch`` itself
    compares bare equality while the wizard and the interactive door
    lower-case first, so ``--cycle Latest`` is accepted by two of the
    three front doors today; a machine interface should not inherit
    that coin flip, and the value handed onward is the canonical one.
    """

    from woof import fetch
    from woof.source_cycles import cycle_grid_for

    arguments = list(arguments)
    # The request is read by the fetch command's own parser, once, and
    # every value below comes from what it accepted: searching the raw
    # list for split-form flags missed `--source=gfs` (and resolved
    # another source's cycle) and `--cycle=latest` (and never resolved).
    parsed = _parse_fetch_arguments(arguments)
    if parsed.cycle is None or parsed.cycle.strip().lower() != "latest":
        return arguments, [], []
    try:
        source, last_hour, options = fetch.latest_cycle_request(parsed)
        # A request the resolver refuses (a host the source does not
        # publish on, a lead or member the route does not carry) is the
        # plan's to fix, and says so the way every other plan refusal
        # does.  A cycle that is simply not published yet is not the
        # plan's fault, and stays a RuntimeError.
        cycle = fetch.resolve_latest_cycle(source, last_hour, **options)
    except ValueError as error:
        raise PlanError(f"run plan 'fetch.args': {error}") from error
    concrete = cycle.strftime("%Y-%m-%dT%H")
    arguments = _with_fetch_cycle(arguments, concrete)
    if _parse_fetch_arguments(arguments).cycle != concrete:
        raise PlanError(
            "run plan 'fetch.args': the resolved cycle could not be "
            "written back into the fetch arguments")

    resolutions = [{
        "scope": "fetch", "key": "cycle", "value": concrete,
        "basis": "resolved_latest",
        "note": ("the newest cycle whose start needs (the analysis and the "
                 "first boundary lead, with what the route fetches beside "
                 "them) are posted; the fetch takes each later lead as it "
                 "posts" if options.get("as_posted") else
                 "the newest cycle whose objects for the final requested "
                 "hour are all published; a partially uploaded cycle "
                 "cannot win, so the window is complete by construction")}]

    # A cycle that is not the newest one the clock allows means newer
    # cycles exist and are still publishing.  That is normal and not an
    # error -- but it means the run initializes from older data than the
    # caller may assume, which is worth one line.  Reported as a
    # warning, never a refusal.
    warnings: list[dict[str, Any]] = []
    # "Older than it needs to be" is measured in THIS source's cycles,
    # not in a step two model names shared.  A grid the resolver just
    # used is the one that says how far apart this producer's inits are;
    # a source that declares none cannot be judged late and is not.
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    grid = cycle_grid_for(source)
    behind = int((now - cycle).total_seconds() // 3600)
    step = _cycle_spacing_hours(grid, cycle)
    if step is not None and behind >= 2 * step + grid.delay(cycle, last_hour):
        warnings.append({
            "code": "latest_cycle_is_not_the_newest",
            "message": (
                f"--cycle latest resolved to {concrete}Z for {source}, "
                f"which is about {behind} h old; newer cycles exist but "
                f"are not yet published through hour {last_hour}"),
            "cycle": concrete, "source": source,
            "age_hours": behind, "last_hour": last_hour})
    return arguments, resolutions, warnings


def _kernel_compile_relay(observer: "RunObserver"):
    """Put the loader's compile events on this run's stream.

    The forecast is hosted in this process, so the kernel loader's
    :func:`woof.kernel_compile_notice.observe_module_compile` reaches an
    ambient sink installed here.  Each compiled module becomes one
    ``warning`` event, code ``kernel_compile_progress``, tagged with the
    stage that is open.
    """

    from woof import progress as progress_mod
    from woof.kernel_compile_notice import COMPILE_PROGRESS_CODE

    def relay(event: str, **fields: Any) -> None:
        if event != "warning" or fields.get("code") != COMPILE_PROGRESS_CODE:
            return
        fields = dict(fields)
        code = fields.pop("code")
        message = fields.pop("message")
        observer.warn(code, message, stage=observer.stage, **fields)

    return progress_mod.event_sink(relay)


def _preparation_relay(observer):
    """Put an in-process preparation's step events on this run's stream.

    The preparation reports each step it takes (:func:`woof.progress.
    prep_stage`, and :func:`woof.progress.prep_progress` for a counted
    step) to whoever listens in the process.  Installed around the one
    call of a route that prepares in this process, so each record lands
    once, as a ``warning`` with code ``preparation_progress``.  The staged
    route had no listener, so a run page showed only "preparing" from the
    download to the first model step.
    """

    from woof import progress as progress_mod

    def relay(event: str, **fields: Any) -> None:
        if event == "warning" and fields.get("code") == "preparation_progress":
            events = getattr(observer, "events", None)
            if events is not None:
                events.emit(event, **fields)

    return progress_mod.event_sink(relay)


def _run_fetch(arguments: Sequence[str], run_dir: Path, *,
               events: "EventStream | None" = None,
               posting_relay: bool = False) -> dict[str, Any]:
    """Execute the plan's fetch through ``woof fetch``'s own handler.

    Returns what the fetch actually did.  ``fetch_main`` answers only
    with an exit code, but it leaves a ``fetch-manifest.json`` beside
    the data naming the cycle, the hours and the files -- so the report
    comes from the receipt the fetch itself wrote, not from re-deriving
    anything here.

    ``events`` is this run's stream.  The fetch runs IN-PROCESS, so
    installing an ambient transfer sink around it (see
    :func:`woof.progress.event_sink`) is enough to put a
    ``fetch_started``/``fetch_progress``/``fetch_completed`` record on
    the stream per file -- without threading a stream handle down
    through every route signature in the fetch family.  The stage used
    to report only ``stage_started`` and, some minutes later,
    ``stage_finished``, so a front end had nothing to draw in between.

    ``posting_relay`` is for a fetch that runs as a stage of its own,
    before the preparation: its ``posting/`` folder (the schedule, each
    lead's marker, the start need it waits on) is carried onto the
    stream for as long as it runs, as ``posting_schedule``,
    ``lead_posted``, ``lead_ready`` and the ``phase: start`` source
    waits (:class:`woof.chain_events.HostedPostingRelay`, the relay a
    fetch beside the preparation already has).  The breakage it
    prevents, measured live on 2026-10-01 (woof go on an hrrr-prs tree
    and on an ecmwf-open-data domain, both hosted by this module): the
    fetch waited for its start needs and dated every lead as it posted,
    and the run's stream said none of it, only transfer progress, so a
    page following ``events.jsonl`` could not tell a wait on the source
    from a slow download and had no lead to show.  A caller that relays
    the posting folder itself leaves it off.
    """

    from woof.cli import parse_fetch_arguments
    from woof import progress as progress_mod

    def relay(event: str, **fields: Any) -> None:
        # Telemetry never fails a fetch: an emit that refuses (an
        # unknown tag, a closed stream) is dropped for that record.
        try:
            events.emit(event, **fields)
        except Exception:            # noqa: BLE001 - see above
            pass

    args = parse_fetch_arguments(arguments)
    posting = None
    if (posting_relay and events is not None
            and getattr(args, "out", None) is not None):
        from woof.chain_events import HostedPostingRelay

        posting = HostedPostingRelay(events, data_dir=args.out)
        posting.start(since_unix_ms=int(time.time() * 1000))
    try:
        if events is None:
            code = args.func(args)
        else:
            with progress_mod.event_sink(relay):
                code = args.func(args)
    finally:
        if posting is not None:
            # Every lead the fetch dated lands before the stage closes.
            posting.stop()
    if code:
        raise StageExitError("fetch", code)

    from woof.fetch import FETCH_MANIFEST_NAME

    report: dict[str, Any] = {"arguments": list(arguments)}
    out = getattr(args, "out", None)
    if out is not None:
        manifest = Path(out) / FETCH_MANIFEST_NAME
        if manifest.is_file():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                report["manifest_path"] = str(manifest)
                for key in ("cycle", "source", "forecast_hours",
                            "payload_bytes"):
                    if key in payload:
                        report[key] = payload[key]
                report["transfers"] = _fetch_transfer_split(payload)
    return report


def _fetch_transfer_split(payload: Mapping[str, Any]) -> dict[str, Any]:
    """How much of this fetch came off the network, and how much did not.

    THE question a retry has to answer.  ``woof fetch`` verifies an
    existing payload -- request identity, GRIB envelope, record count,
    recorded digest -- and skips what passes, so re-running a plan into
    a data directory it already filled moves no bytes at all.  The stage
    used to report only the cycle and the total payload size, which is
    the same line either way, so a user watching a retry had no way to
    know the three gigabytes were not being pulled again.

    Counted over the rows that SAY which they were.  Each transfer
    records ``downloaded`` explicitly; the receipt files a fetch writes
    beside the payload (its checksum list, its own manifest) do not, and
    they are reported separately rather than being folded into either
    number, because a receipt is not something a user waited for.
    """

    files = payload.get("files")
    files = files if isinstance(files, list) else []
    downloaded = verified = receipts = 0
    downloaded_bytes = verified_bytes = 0
    seconds = 0.0
    timed = False
    for entry in files:
        if not isinstance(entry, dict):
            continue
        size = entry.get("bytes")
        size = int(size) if isinstance(size, (int, float)) else 0
        stated = entry.get("downloaded")
        elapsed = entry.get("seconds")
        if isinstance(elapsed, (int, float)):
            seconds += float(elapsed)
            timed = True
        if stated is True:
            downloaded += 1
            downloaded_bytes += size
        elif stated is False:
            verified += 1
            verified_bytes += size
        else:
            receipts += 1
    if downloaded and verified:
        summary = (f"{downloaded} file(s) downloaded, {verified} already on "
                   f"disk and verified ({verified_bytes:,} B not "
                   "re-downloaded)")
    elif verified and not downloaded:
        summary = (f"skipped {verified} file(s) already on disk and verified "
                   f"({verified_bytes:,} B not re-downloaded)")
    elif downloaded:
        summary = f"downloaded {downloaded} file(s), {downloaded_bytes:,} B"
    else:
        summary = "this fetch recorded no per-file transfer state"
    return {
        "downloaded_files": downloaded,
        "downloaded_bytes": downloaded_bytes,
        "verified_files": verified,
        "verified_bytes": verified_bytes,
        "receipt_files": receipts,
        "seconds": round(seconds, 6) if timed else None,
        "summary": summary,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    """One JSON object from disk, or ``{}``.

    A stage that has not written its receipt yet is not an error to the
    reader of it, so an absent or half-written file reads as empty
    rather than raising -- the caller decides what a missing signal
    means.
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _finite_seconds(value: object) -> float:
    """A model-time figure a heartbeat will accept, from anything.

    ``supervisor.Heartbeat`` requires a finite, non-negative float and
    refuses anything else at construction.  A route that has no number
    to give -- because its work happened in a subprocess, or because a
    receipt was not written -- must not be able to turn a completed run
    into a crash inside the arm that emits ``failed``.  That is exactly
    what happened once, so the coercion lives at the boundary rather
    than in each route's good intentions.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return 0.0
    return max(0.0, number)


def _receipts(run_dir: Path) -> dict[str, str]:
    """The receipt artifacts that actually exist, by role.

    Only files present on disk are listed.  A path to a receipt that was
    never written is worse than its absence: the consumer opens it.
    """

    from woof.certify.capsule import CAPSULE_FILENAME
    from woof.supervisor import FAILURE_CAPSULE_NAME, HEARTBEAT_NAME

    known = {
        "certification_capsule": CAPSULE_FILENAME,
        "progress": HEARTBEAT_NAME,
        "failure_capsule": FAILURE_CAPSULE_NAME,
        "manifest": MANIFEST_FILENAME,
        "microphysics_transitions": "microphysics-transitions.json",
        "feedback_provenance": "feedback-provenance.json",
        "initial_perturbation": "initial-perturbation.json",
    }
    found = {}
    for role, filename in known.items():
        path = run_dir / filename
        if path.is_file():
            found[role] = str(path)
    return found


# ---------------------------------------------------------------------------
# Query modes
# ---------------------------------------------------------------------------


def _estimate_planner_machine(exp, probe, profile=None):
    """Use the estimate's device observation when ``[tiles]`` needs it.

    ``mode = "auto"`` with no pinned tiling is the planner's decision and
    the planner needs a machine. The same observation also prices resident
    plans' device-dependent non-pool terms and radiation workspace widths.

    ``profile`` is that same observation's device half, carried onto the
    machine because the tree admission takes its device from the machine
    and from nowhere else.
    """
    options = getattr(exp, "tiles", None)
    if options is None or getattr(options, "mode", "off") == "off":
        return None
    if getattr(options, "tile_nx", None) is not None:
        return None                  # pinned: the configuration IS the plan
    from woof.core.streaming import planner_machine

    return planner_machine(
        vram_bytes=None if probe is None else int(probe["free_bytes"]),
        name="run-plan estimate probe", device_profile=profile)


#: The resident itemizer's basis: the sentence a plan that does not
#: stream is given for its VRAM figures.  The device half of the
#: arithmetic is the card's census (device_profile beside it), so the
#: same plan on the same card is the same figure on every reading and
#: on every surface that prices it.
_RESIDENT_VRAM_BASIS = (
    "woof.core.preflight.estimate_phases, whose forecast term is "
    "estimate_experiment -- it sums every domain and shares the scratch "
    "arena across a tree; the envelope is its peak_envelope_bytes (the "
    "estimator `woof check` reports on the same device_profile; no "
    "device context is created in this process)")

#: What a streamed plan is told instead, naming the mechanism rather than
#: leaving a reader to wonder why the figure shrank.
_STREAMED_VRAM_BASIS = (
    "woof.core.preflight.estimate_phases, whose forecast term is the "
    "STREAMED envelope this config's [tiles] table resolves to: the card "
    "holds nbuffers tile buffers of the compute window, not the whole "
    "domain, so the resident itemization describes a run that will not "
    "happen.  Priced from the same tilestream.autoplan Footprint the run "
    "attaches with, and quoted at the RADIATION PEAK -- the tile buffers "
    "in streamed.vram_bytes plus the measured RRTMGP per-call transient "
    "in streamed.radiation_transient_bytes, which is on the card from the "
    "first radiation step onward; the domain itself lives in host_bytes "
    "of host RAM -- the pinned store and arena (pinned_bytes) plus the "
    "domain's lateral forcing series every tile edge is cut from "
    "(boundary_table_bytes) -- which is a REQUIREMENT of this plan and "
    "not a spare figure")


def _vram_estimate(estimate, streamed, exp) -> dict[str, Any]:
    """The ``vram`` section, priced as the run will actually be allocated.

    ``peak_envelope_bytes`` keeps its meaning across both branches -- it
    is THE number to compare against a card -- and only gains accuracy;
    ``envelope_basis`` is what lets a caller render WHICH figure it got.
    """
    if streamed is None:
        return {
            "domains": len(exp.domains),
            "estimate_bytes": int(estimate.alloc_estimate_bytes),
            "estimate_gib": round(
                estimate.alloc_estimate_bytes / 1024 ** 3, 4),
            # The figure a nested plan must be judged on.  alloc_estimate
            # is the POOL REQUEST; the envelope is what the machine has
            # to have free, and it is the tree-aware one --
            # machine_peak_envelope_bytes adds a per-nest term
            # (nests = domains - 1) and, on WDDM, takes the measured
            # footprint floor.  A tree priced on alloc_estimate alone
            # reads as fitting a card it does not fit.
            "peak_envelope_bytes": int(estimate.peak_envelope_bytes),
            "peak_envelope_gib": round(
                estimate.peak_envelope_bytes / 1024 ** 3, 4),
            "envelope_basis": "resident",
            "basis": _RESIDENT_VRAM_BASIS,
        }
    # THE RADIATION PEAK.  ``vram_bytes`` is what the tiling holds between
    # radiation calls; the RRTMGP call's measured per-process transient is
    # on the card too, and a front end that draws "this run needs N GiB"
    # off the hold sizes a card the run meets the transient on.
    peak = int(streamed.peak_vram_bytes)
    return {
        "domains": len(exp.domains),
        # ONE FIGURE, not two.  The streamed envelope is a whole-process
        # number by construction -- CUDA context, the rung's per-process
        # fixed cost and the tile buffers, with autoplan's safety factor
        # -- and tilestream's Footprint offers no pool/non-pool split to
        # derive a separate request from.  Reporting the resident pool
        # request beside a streamed envelope would put two figures about
        # two different runs in one section, and the ratio between them
        # would mean nothing.
        "estimate_bytes": peak,
        "estimate_gib": round(peak / 1024 ** 3, 4),
        "peak_envelope_bytes": peak,
        "peak_envelope_gib": round(peak / 1024 ** 3, 4),
        "envelope_basis": "streamed",
        "basis": _STREAMED_VRAM_BASIS,
        "streamed": _streamed_vram_section(estimate, streamed),
    }


def _streamed_vram_section(estimate, streamed) -> dict[str, Any]:
    """The ``streamed`` block, in the shape the road actually has.

    A NESTED tree walked by ``streaming.steppers_for_tree`` takes a MIXED
    road -- each domain resident or streamed against the budget its
    predecessors left -- so it has no single tiling to report, and the
    tile keys below describe one streamed domain.  Emitting them for a
    tree would have a reader take the child's tile for the tree's; the
    per-domain roads and claims go out as ``rows`` instead, off the same
    walk (:class:`woof.core.streaming.TreeRoadPlan`).
    """
    section = {
        "resident_envelope_bytes": int(estimate.peak_envelope_bytes),
        # The two terms of the peak above, so a caller can render the
        # steady hold beside it without re-deriving either.
        "vram_bytes": int(streamed.vram_bytes),
        "radiation_transient_bytes":
            int(streamed.radiation_transient_bytes),
        "host_bytes": int(streamed.host_bytes),
        "host_budget_bytes": streamed.host_budget_bytes,
    }
    rows = getattr(streamed, "rows", None)
    if rows is not None:
        section["road"] = "mixed (nested tree)"
        section["rows"] = [dict(row) for row in rows]
        return section
    section.update({
        "road": "streamed (single domain)",
        # The two parts of host_bytes above, so a front end can say which
        # part is page-locked.
        "pinned_bytes": int(streamed.pinned_bytes),
        "boundary_table_bytes": int(streamed.boundary_table_bytes),
        "tile_nx": streamed.tile_nx, "tile_ny": streamed.tile_ny,
        "window_nx": streamed.window_nx,
        "window_ny": streamed.window_ny,
        "nbuffers": streamed.nbuffers, "halo": streamed.halo,
        "rung": streamed.rung, "write_mode": streamed.write_mode,
        # How many tiles a step sweeps and the halo work they do: the two
        # numbers a streamed step's pace follows.  A review that showed the
        # tile alone quoted a 1,190-tile sweep at 49.95x as an ordinary
        # streamed run (measured 2026-09-26).
        "ntiles": int(getattr(streamed, "ntiles", 0) or 0),
        "redundancy": round(float(getattr(streamed, "redundancy", 0.0) or 0.0), 4),
    })
    return section


def _execution_estimate(phases, exp, machine) -> dict[str, Any]:
    """Describe the selected memory estimate without mistaking fallback for resident."""
    from woof.core import streaming

    options = getattr(exp, "tiles", None) or streaming.OFF
    road = getattr(phases, "tree_road", None)
    refusal = None if road is None else road.refusal
    streamed = phases.streamed
    resolved = streamed is not None and refusal is None
    reason = None
    # Why auto took the road it took, in its own words, where this estimate
    # asked it: a resident answer inside the external margin says which
    # tiling it declined and why, and a review that dropped that sentence
    # would show a resident plan over its budget with no explanation.
    tiles_reason = None
    if not resolved and refusal is None:
        if road is not None:
            resolved = bool(road.priced)
        elif len(exp.domains) == 1:
            # The table that GOVERNS this domain, not the tree-wide one: a
            # root carrying its own tiles = {...} was reported resolved
            # here without deciding anything, while the run door decided
            # it on the domain's table.
            if not streaming.options_for_domain(exp.domains[0], options).enabled:
                resolved = True
            elif machine is not None:
                # Reuse the estimate's observation and resident arithmetic.
                # Never probe again to distinguish a resident auto decision
                # from the conservative fallback after a planner refusal.
                # With the phases' own boundary source, so the tables it
                # publishes are in this decision as they are in the phases.
                try:
                    decision = streaming.cold_single_domain_decision(
                        exp, machine=machine,
                        source=getattr(phases, "boundary_source", None))
                    resolved = not decision.stream
                    tiles_reason = decision.reason
                    if decision.stream:
                        reason = "The selected tile plan could not be priced."
                except Exception as error:
                    refusal = str(error)
        elif options.mode == "off" and all(
                streaming.options_for_domain(d, options).mode == "off"
                for d in exp.domains):
            resolved = True
    if not resolved and refusal is None and reason is None:
        reason = "The execution plan is unavailable for this memory estimate."
    return {
        "schema": "arwen.execution-memory.v1",
        "configured_mode": options.mode,
        "configured_tiles": options.to_mapping(),
        "resolved": resolved,
        "streamed_forecast": (streamed is not None if resolved else None),
        "planner_refusal": refusal,
        "unresolved_reason": reason,
        "selected_forecast_envelope_bytes": (int(phases.forecast_envelope_bytes)
                                              if resolved else None),
        "resident_reference_bytes": int(phases.forecast.peak_envelope_bytes),
        "host_bytes": (None if streamed is None else int(streamed.host_bytes)),
        "tree_road": None if road is None else road.to_json(),
        "tiles_reason": tiles_reason,
    }


def estimate_plan(plan: RunPlan) -> dict[str, Any]:
    """What this plan will cost, from measured machinery only.

    VRAM comes from :mod:`woof.core.preflight`'s itemization -- the
    same arithmetic ``woof check`` reports, on the CPU, with no CUDA
    context created IN THIS PROCESS. One short-lived subprocess observes
    the local device for its non-pool terms, radiation workspace widths,
    and tile planner; its context dies with it. Output-frame
    COUNTS are exact.  Wall time is
    ``null``: this package has no measured rate for an arbitrary
    configuration, and a front end showing an invented duration would
    be showing woof's name on a number woof never measured.

    ``[tiles]`` REPLACES THE FORECAST TERM, through
    :func:`woof.core.preflight.estimate_phases` rather than the resident
    itemizer underneath it.  Every term ``estimate_experiment`` sums
    itemizes a domain RESIDENT in VRAM, so a streamed plan was quoted a
    figure for a run that was not going to happen -- and a front end that
    renders this document verbatim then reports "exceeds free VRAM" for
    exactly the small cards streaming exists to serve.  ``envelope_basis``
    says which of the two figures the caller got.

    The INGEST phase is deliberately not priced here (``source=None``):
    this document has never carried a preprocessing term and quietly
    growing one would move ``peak_envelope_bytes`` under callers who
    compare it against a card.  ``woof check`` is the surface that
    prices both phases.  The recorded forcing source is still named as
    the phases' ``boundary_source``, so the forecast term carries the
    hydrometeor boundary tables that source publishes, as the check's
    forecast term does.

    ONE FIGURE FOR ONE PLAN ON ONE CARD.  Every input the device
    contributes is a constant of the card -- its name, shader census,
    default stack limit, compile platform and capacity -- and the
    document carries each of them (``device_profile``,
    ``device_total_bytes``), so a reader can price the same plan through
    :func:`woof.core.preflight.estimate_experiment` on the stated
    device and get this figure to the byte, and two readings of one
    plan on one card are one figure.  The free-memory sample
    (``device_free_bytes``) is the one thing here that moves between
    readings; it reaches the ``[tiles]`` planner and nothing else, so a
    resident plan's figure never follows it, and it is stated so a
    streamed plan's two readings can be told apart by their receipts.
    """

    from woof.core.pace import estimate_pace
    from woof.core.preflight import (
        DEFAULT_FORCING_INTERVAL_SECONDS, case_forcing_schedule,
        device_memory_probe_subprocess, estimate_phases,
        profile_from_device_probe, recorded_forcing_interval_seconds,
        recorded_forcing_source)

    resolution, exp, data = resolve_plan(plan, require_inputs=False)
    projection = resolution["disk"]
    payload = resolution.get("generated_config")
    if payload is None:
        payload = plan.config_bytes().decode("utf-8")
    raw = tomllib.loads(payload)
    if data is not None:
        forcing_interval, intervals = case_forcing_schedule(data, exp)
    else:
        # The cadence `woof check` and `woof go` price this file at:
        # the declared one, else the recorded producer's published one.
        # Reading only the declared key priced a producer that takes no
        # cadence flag at the 21,600 s default here while the check
        # priced the same file at the producer's own cadence.
        forcing_interval = recorded_forcing_interval_seconds(raw)
        intervals = None
    # THE TABLES THE RUN HOLDS, as `woof check` prices this file: the
    # root's boundary carries the analysed hydrometeors the recorded
    # source publishes.  Named as the boundary source only, so this
    # document still carries no ingest term.  Priced with no source, it
    # quoted a config whose source publishes them below the check's
    # figure for the same file on the same card.
    boundary_source = recorded_forcing_source(raw, priced_only=False)
    from woof.boundary_fields import source_boundary_species
    probe = device_memory_probe_subprocess()
    profile = profile_from_device_probe(probe)
    total = None if probe is None else probe.get("total_bytes")
    total = (int(total) if isinstance(total, int)
             and not isinstance(total, bool) and total > 0 else None)
    capacity = None if total is None else total / 1024 ** 3
    free = None if probe is None else probe.get("free_bytes")
    free = (int(free) if isinstance(free, int) and not isinstance(free, bool)
            else None)
    machine = _estimate_planner_machine(exp, probe, profile)
    phases = estimate_phases(
        exp, source=None, boundary_source=boundary_source,
        machine=machine, profile=profile, vram_gib=capacity,
        forcing_interval_seconds=(forcing_interval if forcing_interval is not None else
                                  DEFAULT_FORCING_INTERVAL_SECONDS),
        forcing_intervals=intervals)
    estimate = phases.forecast
    streamed = phases.streamed
    # ONE DECISION PER DOCUMENT.  The envelope this document quotes is
    # handed to the pace rather than re-derived, so the tiling the reader
    # is priced for and the tiling the pace describes are the same one --
    # under ``mode = "auto"`` a second derivation genuinely disagrees when
    # the card's occupancy moves between the two calls.  No probe of our
    # own: this function promises to create no CUDA context, so the pace
    # takes the planner machine already in hand and falls back to the
    # configured ``[tiles] vram_budget_bytes`` when there is none.
    # THE ROOT'S ROAD, not the tree's: the pace model prices the root and
    # charges every nest resident, so it reads a single domain's tiling.
    # See PhaseMemoryEstimate.pace_streamed.
    pace = estimate_pace(exp, streamed=phases.pace_streamed, machine=machine)
    # Taken from the resolution rather than re-derived, so the corridor
    # a caller was told about and the corridor it is quoted a price for
    # are one decision.  A chain that cannot feed a moving nest has
    # already refused inside resolve_plan, above.
    corridor = corridor_estimate(exp, resolution["moving_nest"])
    frames = []
    output_rows = {int(row["grid_id"]): row for row in projection["domains"]}
    for domain in exp.domains:
        interval = float(domain.history_interval_s)
        output_row = output_rows[int(domain.grid_id)]
        frames.append({
            "domain": domain.grid_id,
            "history_interval_s": interval,
            "frames": output_row["history_frames"],
            "history_frame_bytes": output_row["history_frame_bytes"],
            "history_selection": output_row["history_selection"],
            "nx": domain.run.nx, "ny": domain.run.ny, "nz": domain.run.nz,
        })
    return {
        "schema": ESTIMATE_SCHEMA,
        "plan": resolution["plan"],
        "execution": _execution_estimate(phases, exp, machine),
        "vram": {
            **_vram_estimate(estimate, streamed, exp),
            "phase_scope": "forecast",
            "device_basis": ("measured local device" if profile is not None
                             else "conservative reference; local device unmeasured"),
            "device_profile": (None if profile is None else
                               dataclasses.asdict(profile)),
            # The two figures the probe read beside the profile: the
            # capacity the estimate was priced with (vram_gib above)
            # and the free sample the [tiles] planner was given.
            "device_total_bytes": total,
            "device_free_bytes": free,
            "forcing_interval_seconds": (forcing_interval if forcing_interval is not None else
                                         DEFAULT_FORCING_INTERVAL_SECONDS),
            "retained_forcing_intervals": intervals,
            # The hydrometeor masses the recorded source publishes, which
            # the root's boundary tables are priced with (those the
            # microphysics carries), stated beside the device and the
            # cadence so the figure can be repriced from the document.
            "boundary_species": list(source_boundary_species(boundary_source)),
        },
        # THE WHOLE DISK, the download and the preparation included: the
        # projection run-plan refuses on before its download.  This used to
        # be null beside a download that was "no [fetch] in this plan"
        # whenever the prepared route's config carried the [fetch] table,
        # and that plan downloaded 21 GB and wrote about 40 GB more.
        "disk": {
            "frames": frames,
            "total_frames": sum(entry["frames"] for entry in frames),
            "bytes": projection["total_bytes"],
            "download_bytes": projection["download_bytes"],
            "preparation_bytes": projection["preparation_bytes"],
            "history_bytes": projection["history_bytes"],
            "checkpoint_bytes": projection["checkpoint_bytes"],
            "picture_bytes": projection["picture_bytes"],
            # The preparation's decoded frame stream, staged while it runs
            # and removed before the forecast writes: sized by the SOURCE
            # grid, so no other figure here reaches it.
            "compose_scratch_bytes": projection["compose_scratch_bytes"],
            "checkpoint_sets_held": projection["checkpoint_sets_held"],
            "unpriced": projection["unpriced"],
            "basis": "frame counts use the writer's domain start, cadence, "
                     "history window and resume clock; history bytes use "
                     "the selected writer variables and their grid shapes; "
                     "other bytes are the "
                     "download, the preparation and the frame stream it "
                     "stages (woof/download_budget.py) "
                     "and the checkpoints and pictures "
                     f"(woof/disk_budget.py: {projection['basis']})"
                     + ("; not priced: " + ", ".join(projection["unpriced"])
                        if projection["unpriced"] else ""),
        },
        # The preparation's largest single artifact when a nest moves,
        # and absent-by-arithmetic when none does.  Disk AND host: the
        # runner loads the corridor whole at preflight.
        "corridor": corridor,
        # The download, from the request the run will actually make: the
        # plan's own fetch block, or the config's [fetch] table on the
        # prepared route, priced from the sizes measured for its source.
        "download": {
            key: projection["download"].get(key)
            for key in ("bytes", "transfer_bytes", "objects", "leads",
                        "source", "mode", "present_bytes", "basis")},
        # THE PACE, which is the figure a user acts on and the one this
        # document used to leave out.  A streamed plan on a small card is
        # priced correctly, routed correctly, started -- and then looks
        # exactly like a stall, because nothing said a step would cost
        # seconds.  ``expected_pace`` says it before the run, on the road
        # the run will actually take, and names the column count that
        # would put this card back on the fast road.
        "expected_pace": (None if pace is None else pace.to_json()),
        "wall_time": {
            # STILL NULL, and deliberately.  ``expected_pace`` is a
            # BRACKET off measured runs on other cards and other grids;
            # an exact second count here would launder it into a
            # measurement of THIS configuration, which nobody has made.
            "seconds": None,
            "basis": "this package publishes no exact rate for an "
                     "arbitrary configuration; expected_pace carries the "
                     "measured bracket and its provenance, and the "
                     "model_progress events carry the real one from the "
                     "first step",
        },
        "automatic_resolutions": resolution["automatic_resolutions"],
    }


#: One process, one ``rw_wrfbatch --list-products``, keyed by the renderer
#: binary that answered it.
#:
#: The catalog is asked at PLAN REVIEW, and plan review is asked several
#: times over one door: ``woof go`` checks the ``--products`` spelling,
#: the desktop's picker fills its product list, ``woof downscale`` does
#: both again for the child.  Each of those spawned the renderer and
#: waited for it.  The answer cannot change while the binary that gives
#: it does not, so the binary IS the key: its path plus the mtime and
#: size of that file.  Restage a renderer and the next call asks it
#: afresh; nothing is carried across processes, so a cached answer can
#: never outlive the install that produced it.
_RENDER_CATALOG_CACHE: dict[tuple, dict[str, Any]] = {}


def _render_catalog_key() -> tuple | None:
    """The identity of the renderer a catalog would come from.

    ``None`` when there is no renderer to key on -- an install with none
    staged is a case the catalog answers with a refusal document, and a
    refusal is not cached: the remedy it names is "stage the renderer",
    and the whole point of following it is that the next call answers
    differently.
    """
    try:
        from woof import rustwx

        renderer = Path(rustwx.find_renderer())
        stat = renderer.stat()
    except Exception:
        return None
    return (str(renderer), int(stat.st_mtime_ns), int(stat.st_size))


def render_catalog() -> dict[str, Any]:
    """What may be put in ``render_products``, as JSON.

    Cached per process against the renderer binary's own identity; see
    :data:`_RENDER_CATALOG_CACHE`.  The copy handed back is the caller's
    own, so a consumer that annotates the document cannot edit what the
    next caller reads.
    """
    key = _render_catalog_key()
    if key is not None:
        cached = _RENDER_CATALOG_CACHE.get(key)
        if cached is not None:
            return copy.deepcopy(cached)
    document = _read_render_catalog()
    if key is not None and document.get("products"):
        _RENDER_CATALOG_CACHE[key] = copy.deepcopy(document)
    return document


#: What the ``local_run`` block of the catalog rests on.
LOCAL_RUN_CATALOG_BASIS = (
    "the renderer's own fileless verdict on the wrfout import lane: a "
    "product is offered when every field it needs is one that import "
    "writes and, for a run of known length, when its time window closes "
    "within the run")


def local_run_catalog(rows, run_hours: float | None = None) -> dict | None:
    """The products a LOCAL run can draw, from the renderer's WRFOUT rows.

    ``products`` is ``render_catalog()["products"]`` narrowed to what a
    wrfout can ever carry, each with the first forecast hour it can exist
    at; ``unavailable`` names every other product with the engine's own
    reason.  ``run_hours`` narrows it again to a run of that length: a
    window that closes after the run ends is unavailable to that run, in
    those words.

    WHAT BREAKAGE THIS PREVENTS (gate law): the catalog a local run was
    offered listed every product of every model -- ensemble and blend
    families included -- and the default preset built from it asked each
    run for three products no wrfout carries the fields of.  Every run
    drew 20 of 24 pictures and said nothing about the other three.

    A per-product property read off the engine, never a list kept here:
    a product the engine can draw from a wrfout is offered with no edit
    to this file, and no source or model is named.  ``None`` for a
    renderer that published no such rows, which a caller treats as "not
    asked" rather than as "nothing drawable".
    """

    if not rows:
        return None
    products, unavailable = [], {}
    for slug, kind, verdict, minimum_hour, detail in rows:
        if verdict != "drawable":
            unavailable[slug] = detail or "no wrfout import writes its fields"
            continue
        try:
            first = int(minimum_hour) if minimum_hour else 0
        except ValueError:
            first = 0
        if run_hours is not None and first > run_hours:
            unavailable[slug] = (
                f"its time window first closes at forecast hour {first}, and "
                f"this run is {run_hours:g} h long")
            continue
        products.append({"name": slug, "kind": kind, "minimum_hour": first})
    return {"products": products, "unavailable": unavailable,
            "run_hours": run_hours, "basis": LOCAL_RUN_CATALOG_BASIS}


def plan_render_catalog(plan) -> dict[str, Any]:
    """``render_catalog()`` with its ``local_run`` block narrowed to ``plan``.

    What ``run-plan PLAN --catalog`` answers: the products THIS run can
    draw, which is the list a picker for this plan should offer.  The
    run's length is the resolved configuration's own ``run_seconds``.
    """

    document = render_catalog()
    local = document.get("local_run")
    if not isinstance(local, dict):
        return document
    _resolution, exp, _data = resolve_plan(plan, require_inputs=False)
    hours = float(exp.run_seconds) / 3600.0
    rows = [(row["name"], row["kind"], "drawable", str(row["minimum_hour"]), "")
            for row in local["products"]]
    narrowed = local_run_catalog(rows, run_hours=hours)
    narrowed["unavailable"] = {**local["unavailable"], **narrowed["unavailable"]}
    document["local_run"] = narrowed
    return document


def _read_render_catalog() -> dict[str, Any]:
    """Ask the renderer itself what it can draw.

    The renderer's own answer, asked rather than transcribed.  Which
    engine speaks is part of the answer, not an implementation detail,
    so the engine is named in the document.

    ``render.py`` already refuses to keep a second copy of the rust
    catalog for exactly this reason; this keeps that promise across the
    machine seam too.

    There is one engine to ask now.  ``--engine auto`` stopped degrading
    to the matplotlib engine when the render law's one-fallback clause
    was enforced (audit F7), so a box with no usable ``rw_wrfbatch``
    answers ``engine: null``, ``products: null`` and the staging remedy
    in ``error`` -- which is a document a picker can act on, unlike the
    five-name list it used to be handed from a different catalog.
    """

    from woof.render import _resolve_engine, matplotlib_workaround_notice

    document: dict[str, Any] = {
        "schema": CATALOG_SCHEMA,
        "spec": "a comma-separated list of the names below, or 'all'; "
                "'none' skips rendering entirely",
        "skip_token": "none",
    }
    try:
        engine, _why = _resolve_engine("auto")
    except (RuntimeError, FileNotFoundError) as refusal:
        # `auto` no longer degrades to the matplotlib engine (render
        # law, audit F7), so "which products may I ask for" has no
        # answer on a box with no renderer.  The refusal is the answer,
        # carried into the document rather than raised through a JSON
        # front door -- a picker that gets `products: null` plus the
        # staging remedy can say so; one that gets a traceback cannot.
        document.update({
            "engine": None,
            "engine_notice": None,
            "products": None,
            "error": str(refusal).split("[[explain]]")[0].strip(),
        })
        return document
    document["engine"] = engine
    document["engine_notice"] = matplotlib_workaround_notice(engine)

    import subprocess

    from woof import rustwx

    try:
        renderer = rustwx.find_renderer()
        result = subprocess.run(
            [str(renderer), "--list-products"], capture_output=True,
            text=True, errors="replace", env=rustwx.renderer_env(),
            timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        document["products"] = None
        document["error"] = f"{type(error).__name__}: {error}"
        return document
    if result.returncode != 0:
        document["products"] = None
        document["error"] = (result.stderr or "").strip() or             f"renderer exited {result.returncode}"
        return document
    # The renderer INDENTS its product lines and leaves its header and
    # footers flush left.  That is the discriminator, not a guess about
    # which words look like slugs -- and the footer declares the count,
    # so the parse is CHECKED rather than trusted.  A disagreement is
    # reported instead of silently returning a short list to a picker.
    products, groups, declared = [], [], None
    lane_rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        if line.startswith("WRFOUT\t"):
            fields = line.split("\t")
            if len(fields) >= 6:
                lane_rows.append(fields[1:6])
            continue
        if line.startswith((" ", "	")):
            products.append({"name": line.strip()})
            continue
        head, sep, tail = line.partition(":")
        if sep and head.strip() == "group keywords":
            groups = [word.strip() for word in tail.split(",") if word.strip()]
            continue
        name, sep, value = line.partition("=")
        if sep and name.strip() == "selectable_slugs":
            try:
                declared = int(value.strip())
            except ValueError:
                declared = None
    document["products"] = products
    document["group_keywords"] = groups
    document["source"] = "the rust renderer's own --list-products"
    document["local_run"] = local_run_catalog(lane_rows)
    if declared is not None and declared != len(products):
        document["parse_warning"] = (
            f"the renderer declared {declared} selectable slugs and this "
            f"read {len(products)}; the raw output is carried below")
        document["raw"] = result.stdout
    return document


def _source_coverage_facts(window: Any) -> dict[str, Any] | None:
    """A registry coverage window as JSON, or ``None`` for a global source.

    Type-agnostic on purpose: every window type in
    :mod:`woof.source_coverage` answers ``envelope``/``centre``/
    ``describe``, so a window shape added later needs no arm here.  The
    ``grid`` block is the dataclass verbatim, which is where a consumer
    finds whatever a particular window type declares beyond the corners.
    """

    if window is None:
        return None
    south, west, north, east = window.envelope()
    centre_lat, centre_lon = window.centre()
    return {
        "kind": type(window).__name__,
        "south": south, "west": west, "north": north, "east": east,
        "centre_lat": centre_lat, "centre_lon": centre_lon,
        "describe": window.describe(),
        "grid": dataclasses.asdict(window),
    }


def _source_fetch_facts(fetch_routes: Any, source_id: str,
                        adapter: Any = None) -> dict[str, Any]:
    """Which transport, if any, can bring this source's bytes.

    The adapter acquisition owner, then three set memberships decide
    the answer, never a name: the route table, legacy transports and
    named refusals. ``route_for`` is asked ONLY on the two arms that have no
    transport, because it raises "has its own transport" for a legacy
    row -- true, and not a refusal.  Reporting that sentence as one
    would send a reader hunting for a fetch route that already works.

    ``adapter`` is the registry row itself when the caller holds it (the
    inventory does): a row appended to the registry has no entry in the
    alias map built at import, so resolving it again by id refused the
    very row the arbitrary acceptance test grafts on.
    """

    from importlib import import_module
    from woof.source_adapters import get_source_adapter

    if adapter is None:
        adapter = get_source_adapter(source_id)
    if adapter.acquisition_owner:
        owner = import_module(adapter.acquisition_owner)
        return {"kind": "acquisition_owner", "route_id": None,
                "refusal": None, "posting": None,
                "acquisition_owner": adapter.acquisition_owner,
                **owner.acquisition_facts(source_id)}
    if source_id in set(fetch_routes.route_ids()):
        return {"kind": "table_route", "route_id": source_id, "refusal": None,
                "posting": _source_posting_facts(source_id)}
    if source_id in set(fetch_routes.LEGACY_ROUTE_SOURCES):
        return {"kind": "legacy_transport", "route_id": None, "refusal": None,
                "posting": _source_posting_facts(source_id)}
    kind = ("refused" if source_id in set(fetch_routes.refusal_ids())
            else "none")
    refusal = None
    try:
        fetch_routes.route_for(source_id)
    except Exception as error:  # noqa: BLE001 - a query mode reports
        refusal = str(error).split("[[explain]]")[0].strip()
    return {"kind": kind, "route_id": None, "refusal": refusal, "posting": None}


def _source_posting_facts(source_id: str) -> dict[str, Any] | None:
    """How the source's cycles post: the route table's posting block and rules.

    What a site plans a launch from without asking a host (DESIGN A136,
    3.7): the shape, the lateness budget and poll ceiling, and the
    ``publication_lag`` rules each lead's ``expected_at`` comes from.  A
    query mode reports, so a lookup that fails says so in the block.
    """

    try:
        from woof.source_posting import declaration

        return declaration(source_id)
    except Exception as error:  # noqa: BLE001 - a query mode reports
        return {"error": f"{type(error).__name__}: {error}"}


def source_inventory() -> dict[str, Any]:
    """Every registered source and what its registry row declares, as JSON.

    THE picker document.  Front ends used to carry their own list of
    models -- three hand-written rows in one case -- which meant a
    source added to the registry was invisible until someone edited a
    GUI, and a row that changed maturity kept its old colour forever.
    This asks the registry instead, exactly as ``--catalog`` asks the
    renderer rather than transcribing its products.

    Every field below is COPIED from a registry table.  There is no
    dict keyed by model name and no branch on a source id anywhere in
    this function, which is what makes adding a model table work: a row
    appended to ``woof.source_adapters._ADAPTERS`` appears here with
    zero code change, and ``tests/test_runplan.py`` grafts one on to
    prove it.

    Three shapes are worth naming for whoever renders this:

    ``display_name`` is the registry's own DISPLAY-NAME column, and
    ``title`` now carries it too (it used to repeat ``source_id``, which
    is why every consumer showed raw ids and any front end that wanted a
    human name kept its own id-to-name lookup -- a per-model table).  A
    row that declares no name reads back as its id, so a grafted row
    still renders.  The name is not invented here: inventing it here
    would be the same forbidden table one repo to the left.

    ``credentials`` is the registry's CREDENTIAL column, resolved
    against THIS box: what the row declares it needs configured, where
    that lives here, whether it is there, and what breaks if it is not.
    A row that declares nothing says exactly that -- not "this source is
    public", which is a promise about a provider the registry cannot
    make.  Existence only ever leaves this door; the credential's value
    is never read.

    ``run_plan.intent_supported`` is narrower than ``maturity.runnable``
    and says so on every row.  The registry can decode far more sources
    than the run-plan INTENT door can drive from a point-and-cycle, and
    a picker that hid the difference would offer launches the resolver
    refuses.  Rows the intent cannot drive are still listed, with the
    engine's own reasons attached, because a truthful "not from here"
    is worth more than a short menu.

    Nothing here raises.  A registry lookup that fails puts its
    exception text in the affected row's ``error`` (or the envelope's,
    when the manifest itself is unavailable) and the document still
    prints -- the same shape :func:`render_catalog` uses for a box with
    no renderer.  A traceback out of a JSON front door is a defect.
    """

    from woof import __version__
    from woof import fetch_routes
    from woof.source_adapters import (source_adapters,
                                       source_capability_manifest,
                                       wizard_planable_source_ids)
    from woof.source_credentials import credentials_block

    document: dict[str, Any] = {
        "schema": SOURCES_SCHEMA,
        "gpuwm_version": str(__version__),
        "registry_schema": None,
        "source_count": None,
        "runnable_source_count": None,
        "readiness_rule": None,
        "certification_rule": None,
        "routes": {name: route.summary
                   for name, route in sorted(ROUTES.items())},
        "sources": [],
    }
    try:
        manifest = source_capability_manifest()
        document.update({
            "registry_schema": manifest["schema"],
            "source_count": manifest["source_count"],
            "runnable_source_count": manifest["runnable_source_count"],
            "readiness_rule": manifest["readiness_rule"],
            "certification_rule": manifest["certification_rule"],
        })
    except Exception as error:  # noqa: BLE001 - a query mode reports
        document["error"] = f"{type(error).__name__}: {error}"

    try:
        planable = set(wizard_planable_source_ids())
    except Exception as error:  # noqa: BLE001 - a query mode reports
        planable = set()
        document.setdefault("error", f"{type(error).__name__}: {error}")

    try:
        drivability = intent_drivability()
    except Exception as error:  # noqa: BLE001 - a query mode reports
        drivability = {}
        document.setdefault("error", f"{type(error).__name__}: {error}")

    rows: list[dict[str, Any]] = []
    for adapter in source_adapters():
        try:
            verdict = drivability.get(
                adapter.source_id,
                {"routes": [], "chain": None,
                 "refusal": "the intent drivability derivation is "
                            "unavailable; its error is on the envelope"})
            rows.append({
                "source_id": adapter.source_id,
                # The registry's display-name column, with its declared
                # fallback to the id; `title` repeats it for consumers
                # already reading that key.  See the docstring.
                "display_name": adapter.display_title,
                "title": adapter.display_title,
                # What the row declares must be configured before its
                # bytes can be acquired, resolved against this box.
                "credentials": credentials_block(adapter.credentials),
                "aliases": list(adapter.aliases),
                "upstream_model_id": adapter.upstream_model_id,
                "source_kind": adapter.source_kind.value,
                # forecast, analysis or reanalysis: what the bytes are,
                # for a person choosing between sources.
                "record_kind": adapter.record_kind,
                "file_family": adapter.file_family,
                "decoder": adapter.decoder,
                "default_product": adapter.default_product,
                "required_products": list(adapter.required_products),
                "max_forecast_hour": adapter.max_forecast_hour,
                "upstream_ingest": adapter.upstream_ingest,
                "forcing_interval_seconds": adapter.forcing_interval_seconds,
                "packaged_profile": adapter.packaged_profile,
                "member_set": adapter.member_set,
                "composition_requirement": adapter.composition_requirement,
                "runner": adapter.runner,
                "notes": adapter.notes,
                "wizard_planable": adapter.source_id in planable,
                # The registry's own words, unmapped.  It records an
                # AdapterStatus and two rule sentences, and that is the
                # whole maturity fact; a front end colours them, it does
                # not translate them into a second vocabulary.
                "maturity": {
                    "status": adapter.status.value,
                    "runnable": adapter.runnable,
                    "stock_wrf_gate": adapter.stock_wrf_gate,
                    "field_mapping": adapter.field_mapping,
                    "level_mapping": adapter.level_mapping,
                    "cadence_mapping": adapter.cadence_mapping,
                },
                "coverage": _source_coverage_facts(adapter.coverage_window),
                "fetch": _source_fetch_facts(fetch_routes, adapter.source_id,
                                             adapter),
                "run_plan": {
                    "intent_routes": sorted(verdict["routes"]),
                    "intent_supported": bool(verdict["routes"]),
                    # The chain the prepared route would dispatch this
                    # source to, and -- for an undrivable row -- the
                    # derived refusal naming the missing registry fact.
                    # A front end relays the sentence; it does not
                    # translate it.
                    "intent_chain": verdict["chain"],
                    "intent_refusal": verdict["refusal"],
                    "requires_source_root": bool(verdict.get("requires_source_root")),
                    "source_root_reason": verdict.get("source_root_reason"),
                },
            })
        except Exception as error:  # noqa: BLE001 - a query mode reports
            # One unreadable row does not cost a picker the other
            # thirty.  The row is present, named, and carries why.
            rows.append({
                "source_id": getattr(adapter, "source_id", None),
                "error": f"{type(error).__name__}: {error}",
            })
    try:
        from woof.composition_source import inventory_rows
        document["composition_sources"] = inventory_rows()
    except Exception as error:
        document["composition_sources"] = []
        document["composition_error"] = f"{type(error).__name__}: {error}"
    document["sources"] = rows
    return json.loads(json.dumps(document, default=_jsonable))


def physics_profile_menu() -> dict[str, Any]:
    """Which physics suite each registered source can run, as one document.

    The source x profile cross-product, evaluated against the SAME
    admissibility rules that refuse at emission -- the route's physics
    gate, the registry's land-surface offer, the nocturnal-validity
    class, and the component-owned vertical bounds.  Owner design,
    2026-08-20: "why cant every model have multiple unique working
    default".

    This exists because a picker had no way to ask.  WOOF Studio ships
    a physics list hand-typed out of woof 1.7.1, which cannot know that
    the native HRRR route refuses every Kain-Fritsch suite, so it offers
    launches that refuse -- and the refusal a user then met named
    another suite that the same route refuses.  Both halves are the same
    missing object: nobody could ask what a source can actually run.

    :mod:`woof.physics_menu` owns the derivation; this function is the
    envelope, exactly as :func:`source_inventory` is the envelope over
    :func:`intent_drivability`.  There is no dict keyed by model name
    and no branch on a source id or a profile id anywhere in either, so
    a row appended to ``woof.source_adapters._ADAPTERS`` and a suite
    appended to ``woof.domain_wizard.WIZARD_PHYSICS_PROFILES`` both
    appear here with zero code change; ``tests/test_physics_menu.py``
    grafts one of each to prove it.

    Two shapes are worth naming for whoever renders this:

    ``profiles`` is the source-INDEPENDENT half -- what a suite runs,
    its nocturnal class, its registry maturity, the level-count window
    every one of its components accepts.  ``sources[].profiles`` is the
    per-source half and carries the six facts a picker needs per cell:
    ``profile_id``, ``admissible``, ``why_not`` (the route's own refusal
    sentence, verbatim), ``is_default``, ``day_only`` and ``maturity``.
    The two lists are parallel and in the same order, so a front end can
    zip them.

    ``sources[].nocturnal_remedy`` is the answer to "this suite meets a
    night window, now what" FOR THIS SOURCE.  It names a suite the
    source's route admits, and when that suite is the source's own
    default it says to stop passing the flag rather than naming an id
    that will change.  The wizard's refusal reads this same field, so
    the printed remedy and the menu cannot drift.

    Nothing here raises.  A lookup that fails puts its exception text in
    the affected row's ``error`` (or the envelope's) and the document
    still prints -- a traceback out of a JSON front door is a defect.
    """

    from woof import __version__
    from woof import physics_menu
    from woof.physics_registry import registry_sha256
    from woof.source_adapters import (source_adapters,
                                       source_capability_manifest)

    document: dict[str, Any] = {
        "schema": PHYSICS_PROFILES_SCHEMA,
        "gpuwm_version": str(__version__),
        "registry_schema": None,
        "physics_registry_sha256": None,
        "source_count": None,
        "profile_count": None,
        "admissibility_rules": [],
        "profiles": [],
        "sources": [],
    }
    try:
        document["registry_schema"] = source_capability_manifest()["schema"]
    except Exception as error:  # noqa: BLE001 - a query mode reports
        document["error"] = f"{type(error).__name__}: {error}"
    try:
        document["physics_registry_sha256"] = registry_sha256()
        document["admissibility_rules"] = physics_menu.admissibility_rules()
    except Exception as error:  # noqa: BLE001 - a query mode reports
        document.setdefault("error", f"{type(error).__name__}: {error}")

    profiles: list[dict[str, Any]] = []
    for profile in physics_menu.shipped_profiles():
        try:
            profiles.append(physics_menu.profile_facts(profile))
        except Exception as error:  # noqa: BLE001 - a query mode reports
            # One unreadable suite does not cost a picker the other
            # eleven.  The row is present, named, and carries why.
            profiles.append({
                "profile_id": profile,
                "error": f"{type(error).__name__}: {error}",
            })
    document["profiles"] = profiles
    document["profile_count"] = len(profiles)

    from woof.domain_wizard import profile_route_blocker

    rows: list[dict[str, Any]] = []
    for adapter in source_adapters():
        try:
            rows.append(physics_menu.source_menu(
                adapter.source_id, display_name=adapter.display_title,
                blocker=profile_route_blocker))
        except Exception as error:  # noqa: BLE001 - a query mode reports
            rows.append({
                "source_id": getattr(adapter, "source_id", None),
                "error": f"{type(error).__name__}: {error}",
            })
    document["sources"] = rows
    document["source_count"] = len(rows)
    return json.loads(json.dumps(document, default=_jsonable))


def probe_environment(*, readiness: bool = True) -> dict[str, Any]:
    """This machine's device inventory and readiness, as one JSON document.

    The DEVICE half is read through NVML (``nvidia-smi``) and never
    through a CUDA context: capacity is the one device question that
    must be answerable without standing one up, and a front end asking
    "can I run?" must not become a compute contender on the card it is
    asking about.  That half is always safe to poll.

    The READINESS half delegates to :func:`woof.doctor.collect_checks`,
    which verifies the estate for real rather than by presence -- and
    that includes a short-lived subprocess that imports CuPy and runs a
    2x2 matmul, which DOES create a context on the card.  Said plainly
    here because a caller polling a busy card needs to know which half
    costs something: pass ``readiness=False`` (``--no-readiness``) for
    the NVML-only document.
    """

    from woof import __version__
    from woof.provenance_gate import receipt_block

    document: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "gpuwm_version": str(__version__),
        # "Can I run?" is half a question without "what would run?".
        # A front end that probes one box and then launches on it needs
        # both, and gpuwm_version above is the metadata claim, not the
        # tree.
        "provenance": receipt_block(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "pid": os.getpid(),
    }

    devices: list[dict[str, Any]] = []
    device_error = None
    try:
        from woof.core.preflight import (device_physical_total_bytes,
                                          device_wide_used_bytes)
        from woof.supervisor import query_gpus

        total = device_physical_total_bytes()
        used = device_wide_used_bytes()
        for identity in query_gpus():
            devices.append({
                "index": identity.index,
                "uuid": identity.uuid,
                "name": identity.name,
                "driver_version": identity.driver_version,
                "memory_total_bytes": total,
                "memory_used_bytes": used,
                "memory_free_bytes": (None if total is None
                                      else max(0, total - used)),
            })
    except Exception as error:  # noqa: BLE001 - a probe reports, never raises
        device_error = f"{type(error).__name__}: {error}"
    document["devices"] = devices
    document["device_query_error"] = device_error
    document["device_query_basis"] = (
        "NVML via nvidia-smi; no CUDA context is created by this probe. "
        "memory_total is the card's, memory_used is device-wide across "
        "every process, so free is what a new run could actually claim.")

    if not readiness:
        document["readiness"] = {
            "collected": False,
            # Not asked is not ready.  Null, explicitly, for the same
            # reason the error arm below is null: a consumer must be
            # able to tell "unknown" from "no".
            "ready": None,
            "basis": "readiness was not requested; the estate check "
                     "creates a CUDA context and this document is the "
                     "poll-safe half.  `ready` is null, meaning UNKNOWN"}
    else:
        try:
            from woof import capabilities
            from woof.doctor import blocking_gaps, collect_checks

            checks = collect_checks()
            # READY MEANS VERIFIED READY.  This field used to be
            # `not blocking_gaps(checks)`, and doctor carries the CuPy
            # check as non-blocking on purpose (an install that has not
            # opted into a GPU wheel is not a broken install) -- so a
            # bare install answered `"ready": true, "blocking_gaps": 0`
            # to a front end whose very next call is `woof run`, which
            # then refuses.  A probe that prints a green light over a
            # hole is worse than one that says nothing.
            #
            # The requirements a RUN needs are asked here directly, by
            # the same registry the run's own front door refuses with,
            # so the two cannot disagree.
            unmet = capabilities.unmet_run_requirements()
            document["readiness"] = {
                "collected": True,
                "checks": [dataclasses.asdict(check)
                           if dataclasses.is_dataclass(check)
                           else dict(check.__dict__) for check in checks],
                "gaps": sum(1 for check in checks
                            if check.status == "missing"),
                "blocking_gaps": len(blocking_gaps(checks)),
                "ready": not blocking_gaps(checks) and not unmet,
                "unmet_run_requirements": [
                    {"module": item.module,
                     "distribution": item.distribution,
                     "extras": list(item.extras),
                     "needed_for": item.unlocks,
                     "remedy": item.remedy} for item in unmet],
                "basis": "woof doctor's own checks, which verify by "
                         "execution and therefore create a CUDA context, "
                         "plus the run front door's own capability "
                         "requirements; `ready` is true only when both "
                         "are satisfied",
            }
        except Exception as error:  # noqa: BLE001 - a probe reports, never raises
            # UNKNOWN, never ready.  `ready` is present and null so a
            # consumer reading the field gets a third answer rather than
            # a missing key it might treat as false -- or, worse, an
            # absent-means-fine default.
            document["readiness"] = {
                "collected": False,
                "ready": None,
                "error": f"{type(error).__name__}: {error}",
                "basis": "readiness could not be established; `ready` is "
                         "null, which means UNKNOWN and never READY"}
    document["routes"] = {
        name: route.summary for name, route in sorted(ROUTES.items())}
    document["schemas"] = {
        "plan": PLAN_SCHEMA, "event": EVENT_SCHEMA,
        "manifest": MANIFEST_SCHEMA, "resolve": RESOLVE_SCHEMA,
        "estimate": ESTIMATE_SCHEMA, "probe": PROBE_SCHEMA,
        "catalog": CATALOG_SCHEMA, "sources": SOURCES_SCHEMA,
        "physics_profiles": PHYSICS_PROFILES_SCHEMA}
    return json.loads(json.dumps(document, default=_jsonable))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_plan_main(args: argparse.Namespace) -> int:
    """``woof run-plan`` and ``python -m woof.runplan``.

    Exit codes: 0 when the last event was ``completed``, 1 when it was
    ``failed``, 130 on a Ctrl-C (the shell's 128 + SIGINT, matching
    every other long-running command here).  The query modes exit 0 on
    a printed document and 2 on a refusal, argparse's own convention.
    """

    # Bound before any redirect: this is the machine channel, and the
    # document below is the only thing allowed onto it.
    machine_channel = sys.stdout

    def answer(document) -> int:
        machine_channel.write(json.dumps(
            document, indent=2, sort_keys=True, default=_jsonable) + "\n")
        machine_channel.flush()
        return 0

    if getattr(args, "catalog", False):
        # With a PLAN, the catalog's local_run block is narrowed to that
        # run's length: the products THIS run can draw.
        with contextlib.redirect_stdout(sys.stderr):
            document = (render_catalog() if args.plan is None
                        else plan_render_catalog(load_plan(args.plan)))
        return answer(document)
    if getattr(args, "sources", False):
        # Same redirect, same reason as --catalog above: the registry
        # imports print, and stdout is the machine channel.
        with contextlib.redirect_stdout(sys.stderr):
            document = source_inventory()
        return answer(document)
    if getattr(args, "physics_profiles", False):
        # Same redirect, same reason as --catalog above: the physics and
        # source registries import print, and stdout is the machine
        # channel.
        with contextlib.redirect_stdout(sys.stderr):
            document = physics_profile_menu()
        return answer(document)
    if getattr(args, "probe", False):
        with contextlib.redirect_stdout(sys.stderr):
            document = probe_environment(
                readiness=not getattr(args, "no_readiness", False))
        return answer(document)
    if args.plan is None:
        raise PlanError(
            "woof run-plan needs a PLAN.json, or one of --probe / "
            "--catalog / --sources / --physics-profiles (which need no "
            "plan)")

    plan = load_plan(args.plan)
    try:
        if getattr(args, "resolve", False):
            # The redirect covers resolution, not just execution: an
            # intent plan runs the wizard, and the wizard prints -- its
            # resolved cycle, its gray-zone advisories, its fit notes.
            # All of that belongs to the reader on stderr; the document
            # is the answer.
            with contextlib.redirect_stdout(sys.stderr),                     tempfile.TemporaryDirectory(prefix="gpuwm-cycle-") as scratch:
                resolution, _exp, _data = resolve_plan(
                    plan_at_cycle(plan, Path(scratch)), require_inputs=False)
            return answer(resolution)
        if getattr(args, "estimate", False):
            with contextlib.redirect_stdout(sys.stderr),                     tempfile.TemporaryDirectory(prefix="gpuwm-cycle-") as scratch:
                document = estimate_plan(plan_at_cycle(plan, Path(scratch)))
            return answer(document)
        if getattr(args, "readiness", False):
            with contextlib.redirect_stdout(sys.stderr):
                document, code = plan_readiness(
                    plan, no_probe=getattr(args, "no_probe", False))
            answer(document)
            return code
    except PlanError as error:
        # A draft too big for its card is refused with its figures: the
        # machine channel gets them as the memory refusal document every
        # memory refusal prints (``woof.configuration_recovery``), and
        # the sentence still goes to stderr at exit 2.  Without it a
        # front end had only the sentence, and read no figure from the
        # one that says preprocessing is not priced for the source.
        if error.memory is not None:
            from woof.configuration_recovery import error_document

            machine_channel.write(json.dumps(
                error_document(error), sort_keys=True) + "\n")
            machine_channel.flush()
        raise

    run_dir = plan.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    # The EventStream binds the REAL stdout here, before the redirect
    # below moves everyone else's to stderr.  That split is the whole
    # promise of this front door: stdout is the machine channel and
    # carries JSONL and nothing else.
    #
    # It is not hypothetical.  The pipeline prints its resolved-config
    # report (runtime.py:1793/1852), its feedback warning, and the
    # wizard prints its resolved cycle -- all with plain print(), all
    # correct for a person, all landing in the middle of the stream a
    # consumer is calling json.loads on line by line.  The dry-run path
    # never reaches any of them, which is exactly why the subprocess
    # test that covered stdout purity passed while a real run did not.
    with EventStream(run_dir / EVENTS_FILENAME) as events:
        with contextlib.redirect_stdout(sys.stderr):
            return execute_plan(plan, events=events)


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    """Register the machine-facing execution front door."""

    parser = subparsers.add_parser(
        "run-plan",
        help="execute one versioned run plan and emit a structured "
             "event stream (JSONL to <run_dir>/events.jsonl and to "
             "stdout) that a program can consume without parsing any "
             "human output")
    parser.add_argument(
        "plan", type=Path, nargs="?", default=None, metavar="PLAN.json",
        help=f"a {PLAN_SCHEMA} document: which route to execute, which "
             "config to execute it with, and where the outputs land")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resolve", action="store_true",
        help="print the fully resolved configuration plus every "
             "automatic resolution as one JSON document, and run "
             "nothing")
    mode.add_argument(
        "--estimate", action="store_true",
        help="print this plan's VRAM estimate, output-frame counts, "
             "download and disk bytes as one JSON document, and run "
             "nothing")
    mode.add_argument(
        "--readiness", action="store_true",
        help="print gpuwm.readiness.v1 for the window this plan fetches "
             "and run nothing: exit 0 ready (or nothing to probe), 75 not "
             "yet (with expected_ready_at and retry_after_seconds), 2 "
             "refused (the window can never start)")
    parser.add_argument(
        "--no-probe", dest="no_probe", action="store_true",
        help="with --readiness: compute the schedule from the source "
             "table only and ask no host")
    mode.add_argument(
        "--catalog", action="store_true",
        help="print the renderer's product catalog as one JSON "
             "document -- what may be put in the render_products run "
             "option, and in local_run the products a local run can "
             "draw -- and run nothing; needs no plan, and a PLAN narrows "
             "local_run to that run's length")
    mode.add_argument(
        "--sources", action="store_true",
        help="print the source registry as one JSON document -- every "
             "registered source, what each one's row declares, and which "
             "run-plan route can drive it from an intent -- and run "
             "nothing; needs no plan")
    mode.add_argument(
        "--physics-profiles", dest="physics_profiles", action="store_true",
        help="print the per-source physics menu as one JSON document -- "
             "every registered source crossed with every shipped physics "
             "suite, saying which pairings this product can actually "
             "prepare, why each refused one is refused, which suite that "
             "source's bare run binds, and which suites run shortwave "
             "with longwave off -- and run nothing; needs no plan")
    mode.add_argument(
        "--probe", action="store_true",
        help="print this machine's device inventory and runtime-estate "
             "readiness as one JSON document; needs no plan.  The "
             "device inventory is NVML only and creates no CUDA "
             "context; the readiness half runs `woof doctor`'s checks, "
             "which verify by execution and do create one")
    parser.add_argument(
        "--no-readiness", dest="no_readiness", action="store_true",
        help="with --probe, report the device inventory only: the "
             "NVML-only half, safe to poll on a card that is busy")
    parser.set_defaults(func=run_plan_main)


def _module_entry(argv: Sequence[str] | None = None) -> int:
    """``python -m woof.runplan PLAN.json``.

    Delegates the WHOLE invocation to :func:`woof.cli.main`, not just
    the parser.  A second entry point that only shared the parser would
    also need its own refusal print boundary, and the first version of
    this function proved why that is a bug rather than duplication: it
    printed the ``[[explain]]`` sentinel straight to the terminal on a
    layered refusal, because choosing a layer is the boundary's job and
    this function was not it.  One boundary, two spellings.
    """

    from woof.cli import main

    tokens = list(sys.argv[1:] if argv is None else argv)
    return main(["run-plan", *tokens])


if __name__ == "__main__":
    sys.exit(_module_entry())


__all__ = [
    "CATALOG_SCHEMA", "DEFAULT_OUTPUT_ROOT", "ESTIMATE_SCHEMA", "EVENTS_FILENAME",
    "EVENT_SCHEMA", "EVENT_TAGS", "MANIFEST_FILENAME", "MANIFEST_SCHEMA",
    "PHYSICS_PROFILES_SCHEMA", "POSTING_EVENT_FIELDS",
    "PLAN_SCHEMA", "PROBE_SCHEMA", "RESOLVE_SCHEMA", "ROUTES",
    "SOURCES_SCHEMA", "STAGES",
    "WARNING_CODES", "WARNING_CODE_PREFIXES",
    "GENERATED_CONFIG_NAME",
    "DiskRefusal", "EventStream", "PlanError", "Route", "RunObserver", "RunPlan",
    "build_plan", "collect_warnings", "corridor_estimate",
    "source_behind_fields",
    "declared_inputs", "disk_admission_refusal", "domain_size_floor",
    "missing_inputs_refusal",
    "estimate_plan", "execute_plan", "follow_statics_decision",
    "follow_unsupported_refusal",
    "generate_intent_config", "physics_profile_menu", "render_catalog",
    "local_run_catalog", "plan_render_catalog", "LOCAL_RUN_CATALOG_BASIS",
    "source_inventory",
    "intent_arguments", "load_plan", "prepared_chain_for_source",
    "probe_environment", "read_events",
    "register_cli", "resolve_fetch_cycle", "resolve_plan",
    "run_plan_main", "stated_remedy", "streaming_decision", "write_manifest",
]
