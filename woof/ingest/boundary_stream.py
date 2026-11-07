"""Chained preparation: the forecast starts while later boundaries are built.

Boundary interval k needs only forcing times k and k+1.  A preparation used
to build every forcing time, pack every interval and publish the whole
prepared tree in one rename before the model could take its first step.
This module splits the prepared tree into three parts so the model can
start once the start state and the next forcing time exist:

* the HEAD: everything the start time makes (static fields, receipts, the
  non-boundary arrays of ``prepared-cache/``) plus ``boundary-stream/
  head.json``, which carries the full interval schedule.  It is published
  with the route's own single rename, early;
* one SEGMENT per interval: its arrays written straight into
  ``prepared-cache/`` under the file numbers the one-shot writer would have
  given them, then ``boundary-stream/segments/{k:05d}.json`` written last.
  The marker is the only ready signal, so the rule works across processes
  and across machines (a copy loop copies arrays first, marker last);
* the SEAL: the same ``prepared-cache/header.json`` the one-shot writer
  writes (so ``content_sha256`` is unchanged), then the companion files,
  then ``proof.json`` last, which stays every existing consumer's
  completion marker.

A tree whose segments all exist before the run starts is the same source,
already complete: there is one writer (:class:`PreparedTreeWriter`), one
reader (:class:`StreamedIntervals`) and one wait, used by every route that
builds boundaries from forcing times.

The early publication is on by default (see :data:`CHAINED_DEFAULT`);
``WOOF_CHAINED_PREP=0`` turns it off for diagnosis, and then the same
writer publishes the head at the seal, so the run starts after preparation
exactly as before.  An installation with no forecast (the RW-WPS
preparation package, see :func:`forecast_installed`) always publishes at
the seal.
"""

from __future__ import annotations

from collections.abc import Sequence
import contextvars
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import threading
import time
import traceback
from typing import Callable, Mapping

import numpy as np


#: True inside a forcing-time build that runs beside others
#: (:meth:`PreparedTreeWriter.stream_forcing_times` with workers > 1).
_CONCURRENT_FORCING_BUILD = contextvars.ContextVar("concurrent_forcing_build", default=False)


def concurrent_forcing_build() -> bool:
    """Whether this forcing-time build runs beside others on the writer's pool.

    Alone, a route's build reads its source through a shared one-deep cache;
    beside others it packs its own copy (woof.mapped_direct), so the packs
    run side by side too instead of one at a time under a lock.
    """
    return _CONCURRENT_FORCING_BUILD.get()


STREAM_DIRNAME = "boundary-stream"
HEAD_NAME = "head.json"
SEGMENTS_DIRNAME = "segments"
PRODUCER_NAME = "producer.json"
FAILED_NAME = "failed.json"
STOP_NAME = "stop.json"
PROOF_NAME = "proof.json"
HEAD_SCHEMA = "gpuwm-boundary-stream-head-v1"
SEGMENT_SCHEMA = "gpuwm-boundary-stream-segment-v1"
CHAINED_ENV = "WOOF_CHAINED_PREP"

#: How often the producer refreshes ``producer.json``.
HEARTBEAT_SECONDS = 15.0
#: How often a waiting consumer republishes its wait reason.  The ``woof
#: run`` supervisor reads a moving heartbeat as alive, so a wait that
#: publishes nothing for longer than its hang threshold would be killed as
#: a hang while the producer is still working.
WAIT_REPORT_SECONDS = 5.0
#: The shortest heartbeat age that counts as a silent producer.  The limit
#: grows to three times the slowest forcing-time build seen so far, so a
#: slow machine is not mistaken for a dead one.
SILENT_FLOOR_SECONDS = 120.0

#: Proof keys only the seal can write: they bind the complete cache, the
#: companion WRF export and the wall times of the whole preparation.  The
#: head carries the proof WITHOUT them (``proof_head``); the sealed proof
#: must equal the head's proof plus exactly these keys, which is what lets a
#: runner validate everything else at the head and the rest at the seal.
SEAL_ONLY_PROOF_KEYS = frozenset({
    "prepared_cache", "export", "initialization_artifacts",
    "timing_seconds", "proof_content_sha256", "boundary_stream",
    # Runtime telemetry completes only after every forcing time is built.
    # It changes no array, scientific setting or cache identity.
    "preparation_parallelism", "forcing_stage_timings",
    # A domain tree's one-shot artifact tree and its companion WRF
    # hierarchy are written at the seal.  No single-domain proof carries
    # these keys, so a single domain's contract is unchanged.  A tree's
    # statics corridor is not among them: it needs no boundary time, so a
    # chained tree builds it into its head (hierarchy-head/statics-corridor)
    # and the head's proof binds it, which is what lets a moving nest's
    # forecast start at the head.
    "artifact_receipt", "wrf_manifest",
    # An as-posted preparation's record of the leads it waited for
    # (DESIGN A136 2.4 item 6); no one-shot proof carries it.
    "posting",
    # A native HRRR domain tree's record of the root preparation it was
    # joined to: the root's sealed cache content digest, which exists
    # only once the root preparation seals.  No other proof carries it.
    "root_preparation",
})

#: An as-posted head (DESIGN A136 2.4 item 5 and the L3 design ruling):
#: the head is written from the start leads, before the input manifest
#: exists, so it binds the INPUT PLAN (every lead, every object name and
#: every non-lead input the manifest will bind) and the start leads'
#: object digests instead.  The seal writes the one-shot identity and the
#: one-shot proof; :func:`verify_seal` accepts exactly the identity keys
#: below and the head's declared seal-authored proof keys changing, and
#: holds every sealed manifest row to its lead's posted marker and the
#: plan recomputed from the sealed manifest to the head's.
AS_POSTED_SCHEMA = "gpuwm.as-posted-head.v1"
INPUT_PLAN_SCHEMA = "gpuwm.input-plan.v1"
#: The seal's record of every lead the preparation consumed, as its
#: posted marker named it (``boundary-stream/posted-leads.json``), so a
#: verifier without the fetch folder can hold the manifest to it.
POSTED_LEADS_NAME = "posted-leads.json"
POSTED_LEADS_SCHEMA = "gpuwm.posted-leads.v1"
#: The cache identity keys the L3 design ruling lets an as-posted seal
#: write: the input manifest's and the composition receipt's digests,
#: which an as-posted head cannot know.  Its head carries
#: :func:`as_posted_placeholder` in each; the seal the digest.  A route
#: whose identity carries the input manifest's digest under another name
#: too (GFS: ``bridge_manifest_sha256`` and ``source_manifest_sha256``)
#: declares those keys manifest-bound in its head, and the seal may write
#: there only that same manifest digest.  Any other identity difference,
#: and any placeholder left in the sealed identity, is refused.
AS_POSTED_IDENTITY_KEYS = frozenset({
    "input_manifest_sha256", "composition_receipt_sha256",
})
#: The prefix of :func:`as_posted_placeholder`.
AS_POSTED_PLACEHOLDER_PREFIX = "as-posted:"
#: How a document an as-posted seal writes is held to what the head and
#: its segments bound, row by row (``basis.as_posted.documents``).  A
#: route whose identity carries another document's digest (native HRRR:
#: ``bridge_manifest_sha256`` is the decoded bridge's SHA256SUMS,
#: ``source_manifest_sha256`` the fetch's) declares those keys
#: document-bound: the head carries the plan's placeholder, the seal writes
#: exactly the digest the sealed input manifest names for that document,
#: and every row of the document must equal a per-lead record bound as
#: the lead was read.  ``posted_objects``: each row is an object a lead's
#: posted marker named (name and digest), and every such object has a row.
#: ``decoded_leads``: the rows outside ``fixed_rows`` are exactly the
#: union of the per-lead decoded-file records each segment bound.
DOCUMENT_ROW_RULES = ("posted_objects", "decoded_leads")

#: ``head.layout`` for a prepared domain tree (a single domain omits it).
LAYOUT_DOMAIN_TREE = "domain_tree"
#: Where a domain tree's head keeps every domain's artifacts: the children
#: complete, the root's static files and its streamed prepared cache.
HIERARCHY_HEAD_DIRNAME = "hierarchy-head"
#: Where a domain tree's seal writes the one-shot artifact tree.
SEALED_HIERARCHY_DIRNAME = "hierarchy-artifacts"


class BoundaryStreamError(RuntimeError):
    """A streamed boundary source cannot deliver the interval asked for."""


class BoundaryProducerFailed(BoundaryStreamError):
    """The producer wrote ``failed.json``: the run ends with its reason."""


class BoundaryProducerSilent(BoundaryStreamError):
    """The producer stopped refreshing its heartbeat without failing."""


class BoundaryStreamStopped(BoundaryStreamError):
    """The consumer wrote ``stop.json``; the producer exits unsealed."""


class PostedWaitStopped(BoundaryStreamError):
    """The owner of a wait for a posted lead ended it (``PostedLeads.wait``).

    Without it a waiter on its own thread polled the posting folder until
    the lead posted or the fetch failed, long after the preparation that
    started it had stopped (the A136 L7c admitter's thread outlived a
    stopped admitter and read ``fNNN.json``, ``failed.json`` and
    ``schedule.json`` of a finished preparation's folder).
    """


class StreamedClockChanged(BoundaryStreamError):
    """A later boundary interval moved the terrain-derived clock.

    Raised by the tree runner's clock guard
    (``prepared_domain_tree_forecast.StreamedClockGuard``) as the interval
    loads.  The run
    so far stepped on a clock a sealed run of the same tree would not
    choose, so it is not that forecast; the runner restarts on the seal.
    """


#: The exit code of a run a source left behind: a lead passed its
#: ``late_at`` (``posting.late_after_minutes`` past its scheduled time).
#: The shell's EX_TEMPFAIL, because the same launch succeeds once the lead
#: posts; every door (fetch --as-posted, go, run-plan) exits with it.
SOURCE_BEHIND_EXIT_CODE = 75
SOURCE_BEHIND_CODE = "source_behind"
#: What a ``source_behind`` record names about the late lead.
SOURCE_BEHIND_FIELDS = ("source", "cycle", "lead", "valid_time",
                        "expected_at", "late_at", "late_after_minutes",
                        "last_answer")
#: What a waiting producer's heartbeat names about the lead it waits on
#: (``producer.json`` ``waiting_for``).  ``state`` and ``first_seen_at``
#: are the fetch's own word on the lead (its schedule row), refreshed while
#: the wait lasts, so a wait reason says what the fetch knows: not posted,
#: not asked for yet, or posted and still downloading.
WAITING_FOR_FIELDS = ("source", "cycle", "lead", "valid_time",
                      "expected_at", "late_at", "since_utc", "state",
                      "first_seen_at", "last_answer")
#: How often a source wait is said again on the event stream while it
#: lasts: nothing else moves in the stream during a wait, and a reader
#: that only tails it cannot otherwise tell a wait from a hang.
SOURCE_WAIT_PROGRESS_SECONDS = 60.0
#: The runner's own record of every wait event it said, one JSON object
#: per line, in its output directory.  ``woof go`` runs the forecast as a
#: subprocess with no event stream, and relays this file onto its own.
WAIT_LOG_NAME = "waits.jsonl"


def _lead_words(lead) -> str:
    try:
        return f"f{int(lead):03d}"
    except (TypeError, ValueError):
        return f"lead {lead}"


def _clock_words(text) -> str:
    """``2026-09-30T15:53:04Z`` -> ``15:53Z``; anything else verbatim."""

    try:
        instant = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return str(text)
    if instant.tzinfo is not None:
        instant = instant.astimezone(timezone.utc)
    return instant.strftime("%H:%MZ")


def _fetch_knows(cause: Mapping[str, object]) -> str:
    """What the fetch knows of a lead a run waits on, from its schedule row.

    ``posted`` once a host has held it (``first_seen_at``, or the row's
    ``posted`` state): the run then waits on the download, not on the
    publisher.  ``unasked`` while the row is still ``scheduled``: the
    fetch has not reached the lead (a late launch moving its posted
    backlog), so whether it is posted is not known.  Otherwise
    ``not_posted`` (``waiting``: the fetch announced it as not posted
    yet), which is also the word for a record with no state, as before
    the fetch said one.  ``unheard`` when the fetch's last answer about
    the lead (the row's ``last_answer``) is that its host could not be
    heard.  The breakage this prevents: a seam wait said
    "gfs f060 is not posted yet" for a lead the fetch had seen posted 23 s
    earlier and was downloading, "not posted" for a lead the fetch had
    not asked about at all, and "not posted" or "not fetched" for a lead
    whose host did not answer (DESIGN A136 3.6: something the engine could
    not see is never reported as the publisher being late).
    """

    if cause.get("first_seen_at") or cause.get("state") == "posted":
        return "posted"
    if cause.get("last_answer") == "not_heard":
        return "unheard"
    if cause.get("state") == "scheduled":
        return "unasked"
    return "not_posted"


def lead_wait_words(cause: Mapping[str, object]) -> str:
    """The state half of a wait line: ``not posted yet (scheduled from about 15:53Z)``."""

    known = _fetch_knows(cause)
    if known == "posted":
        seen = cause.get("first_seen_at")
        return ("posted" + (f" at {_clock_words(seen)}" if seen else "")
                + ", still downloading")
    words = {"unasked": "not fetched yet",
             "unheard": "not fetched; its host cannot be heard"}.get(
                 known, "not posted yet")
    if cause.get("expected_at"):
        words += (" (scheduled from about "
                  f"{_clock_words(cause.get('expected_at'))})")
    return words


def source_wait_reason(cause: Mapping[str, object]) -> str:
    """The reason a run waiting on a source lead gives, from that lead's record.

    ``cause`` names ``source``, ``lead``, ``expected_at`` and ``late_at``
    and, when the fetch has said them, ``state``, ``first_seen_at`` and
    ``last_answer`` (a producer's ``waiting_for``, or the fetch's schedule
    row for a start need; :func:`_fetch_knows`).  One wording for the seam wait and the
    start wait, so the two cannot describe the same lead differently.
    """

    lead = f"{cause.get('source')} {_lead_words(cause.get('lead'))}"
    when = (f"scheduled from about {_clock_words(cause.get('expected_at'))}; "
            f"late at {_clock_words(cause.get('late_at'))}")
    known = _fetch_knows(cause)
    if known == "posted":
        seen = cause.get("first_seen_at")
        return (f"{lead} has posted"
                + (f" (first seen {_clock_words(seen)})" if seen else "")
                + " and is still downloading")
    if known == "unasked":
        return f"{lead} is not fetched yet; the fetch has not reached it ({when})"
    if known == "unheard":
        return (f"{lead} is not fetched: its host cannot be heard, so whether "
                f"it is posted is not known ({when})")
    return f"{lead} is not posted yet ({when})"


def _model_words(elapsed, valid) -> str:
    seconds = int(round(float(elapsed)))
    words = f"{seconds // 3600}:{(seconds % 3600) // 60:02d}"
    if valid:
        try:
            instant = datetime.fromisoformat(str(valid).replace("Z", "+00:00"))
            words += f" (valid {instant.strftime('%Y-%m-%dT%HZ')})"
        except ValueError:
            words += f" (valid {valid})"
    return words


def source_behind_sentence(details: Mapping[str, object]) -> str:
    """The refusal a late lead ends a run with, naming the lead.

    A lead the engine could not hear about is never reported as the
    publisher being late: the sentence says the host was not heard.
    """

    source = details.get("source") or "the source"
    lead = _lead_words(details.get("lead"))
    answer = details.get("last_answer")
    verb = {"not_heard": "could not be heard from about",
            "failed_verification": "posted but did not verify by"}.get(
                str(answer), "has not posted by")
    budget = details.get("late_after_minutes")
    budget_words = ("its budget" if budget is None
                    else f"{float(budget):g} min")
    head = (f"{source} {lead} of the {details.get('cycle')} cycle "
            f"{verb} {_clock_words(details.get('late_at'))}, "
            f"{budget_words} after its scheduled time "
            f"({_clock_words(details.get('expected_at'))}; the source "
            f"table's late_after_minutes is "
            f"{'unset' if budget is None else f'{float(budget):g}'}).")
    if details.get("model_elapsed_seconds") is not None:
        head += (" The forecast stopped at "
                 f"{_model_words(details['model_elapsed_seconds'], details.get('model_valid_time'))}; "
                 "the frames through that time are kept.")
    return (head + f"\nwhat to do: launch the same config again once {lead} "
            "posts (it resumes from the fetched prefix), or raise [fetch] "
            "late_after_minutes for this source.")


class SourceBehind(BoundaryProducerFailed):
    """A source lead passed its ``late_at``: the run stops with exit 75.

    ``details`` carries :data:`SOURCE_BEHIND_FIELDS`, and the forecast
    adds where it stopped (``model_elapsed_seconds``, ``model_valid_time``,
    ``frames_kept``, ``checkpoint``) before the error leaves the runner.
    """

    code = SOURCE_BEHIND_CODE
    exit_code = SOURCE_BEHIND_EXIT_CODE

    def __init__(self, details: Mapping[str, object] | None = None,
                 message: str | None = None):
        details = dict(details or {})
        self.details = {key: details.get(key) for key in SOURCE_BEHIND_FIELDS}
        for key in ("model_elapsed_seconds", "model_valid_time",
                    "frames_kept", "checkpoint"):
            if key in details:
                self.details[key] = details[key]
        super().__init__(message or source_behind_sentence(self.details))

    def at(self, **where) -> "SourceBehind":
        """The same refusal, saying where the forecast stopped."""

        return SourceBehind({**self.details, **where})


#: Whether preparations chain when ``WOOF_CHAINED_PREP`` is unset.  On:
#: the GPU proof runs of 2026-09-28 wrote byte-identical history files
#: chained, unchained and on the line before this change, for a single
#: mapped domain, a nested tree, a tiled root, a tiled child, a GFS domain,
#: a restart before and after the seal and a downscaled child.  The
#: variable stays as the off switch for diagnosis.
CHAINED_DEFAULT = True


def chained_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether a preparation publishes its head before its seal.

    ``WOOF_CHAINED_PREP=1`` (or on/true/yes) turns it on and ``0`` (or
    off/false/no) off; unset, :data:`CHAINED_DEFAULT` decides.  Every
    stage subprocess inherits the variable.
    """

    value = (os.environ if environ is None else environ).get(CHAINED_ENV)
    if value is None or not str(value).strip():
        return CHAINED_DEFAULT
    return str(value).strip().lower() not in {"0", "off", "false", "no"}


def chained_resident_forcing_slots(exp, environ: Mapping[str, str] | None = None) -> int | None:
    """Forcing intervals a chained ``woof go`` forecast holds on the card.

    A one-domain, one-card forecast started on a chained preparation's head
    streams its root's lateral intervals through one device slot
    (:func:`woof.ingest.lateral_bc.attach_streaming_lateral_boundaries`),
    so it holds 1.  ``None`` means the whole schedule: chaining off
    (``WOOF_CHAINED_PREP=0``, a sealed restore), a nested tree, or a split
    across cards.  The go gate and ``woof check`` both read this, so the two
    never disagree about one card: the check priced all 19 hourly hrrr-prs
    intervals of an 18 h full-grid HRRR (8.24 GiB of boundary tables) and
    refused it on a 96 GB card by 0.5 GiB while ``woof go`` prices one slot.
    """
    if not chained_enabled(environ) or len(exp.domains) != 1:
        return None
    devices = getattr(exp, "devices", None)
    if getattr(devices, "enabled", False):
        return None
    return 1


#: What a chained head needs on the machine that writes it: the forecast's
#: memory admission (:func:`chained_admission` reads it from
#: ``woof.core.preflight``) and a forecast executor to bind the head.  The
#: RW-WPS preparation package ships neither.
CHAINED_FORECAST_MODULES = ("woof.core.preflight", "woof.core.model")

#: The decision a writer records when :func:`forecast_installed` is false.
PREPARATION_ONLY_REASON = (
    "this installation prepares inputs and carries no forecast to start on "
    "the head, so the tree is published at its seal")


def forecast_installed() -> bool:
    """Whether this installation can run a forecast on a chained head.

    False in the RW-WPS preparation package, which stages this module for
    the era5, gfs and mapped routes but none of
    :data:`CHAINED_FORECAST_MODULES`.  There a CUDA preparation's
    admission died on ``No module named 'woof.core.preflight'`` right
    after its start time was built, and a CPU one declined chaining only
    because its RAM reader lives in that same missing module.
    """

    for name in CHAINED_FORECAST_MODULES:
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def _say(line: str) -> None:
    """One line on stderr: a hosting door (run-plan) owns stdout."""

    print(line, file=sys.stderr, flush=True)


def _sealed_line(reason: str) -> str:
    return f"prepare: {reason}; the forecast starts after preparation"


#: Why a preparation is published at its seal rather than its head, one
#: row per kind of preparation that builds every forcing time before its
#: forecast can start.  Each is said once, where that preparation builds
#: its forcing (:func:`say_prepared_sealed`), so a run that did not start
#: early says why.
SEALED_REASONS = {
    # Every domain tree chains: a mapped or GFS-series tree on either
    # backend (its children need only the start time, so they are prepared
    # into the head, and the seal re-reads every start state from the head:
    # TreeStartStates), and a native HRRR tree on its root preparation's
    # head (woof.hrrr_hierarchy_direct relays the root's intervals).  Its
    # forecast starts on that head (the tree runner's
    # --prepared-head-sha256), so no tree row is left here.
    "water_overlay": (
        "chained preparation not used: the water-temperature overlay binds "
        "its receipt over every forcing time into the cache identity, so "
        "the head cannot exist before the last forcing time"),
    "met_em": (
        "chained preparation not used: the met_em route writes each "
        "domain's prepared cache whole"),
    "experiment_run": (
        "chained preparation not used: this route builds its forcing "
        "times in the forecast's own process, before the model starts"),
}


def say_prepared_sealed(kind: str) -> None:
    """Say why this preparation is published at its seal, not its head.

    ``kind`` is a row of :data:`SEALED_REASONS`.  The line is the one a
    chained route prints when it declines, on stderr.  Silent when
    ``WOOF_CHAINED_PREP=0`` turned chaining off, because then nothing
    was expected to start early.
    """

    reason = SEALED_REASONS[kind]
    if chained_enabled():
        _say(_sealed_line(reason))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value) -> str:
    from woof.ingest.prepared_cache import _canonical as canonical

    return canonical(value)


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None


#: The waits between reads of a file another process keeps replacing: the
#: writer's own permission-error backoff
#: (:func:`woof.filesystem_paths.replace_file_with_retry`), so a read
#: outlasts one replace however the two interleave.  About 2 s in all: the
#: hosted Windows CI of the 2.9.2 publish showed a polling reader and a
#: replacing writer colliding for longer than half a second.
_REPLACED_READ_BACKOFF_SECONDS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.19, 0.25, 0.25, 0.5, 0.5)


def read_replaced_json(path, *, sleep: Callable[[float], None] = time.sleep):
    """A JSON file another process replaces while this one reads it.

    The as-posted fetch rewrites ``posting/schedule.json`` (and the growing
    ``fetch-manifest.json``) with an atomic replace each time a lead moves.
    On Windows a read that meets the replace fails with a permission error
    for a moment, although the file is whole before and after.  The
    breakage this prevents: such a read answered "no route table" and
    failed an as-posted seal.  A read is tried again on any ``OSError``
    (and on a document that does not parse) for about half a second; the
    last error is then raised, so a file that is really missing or broken
    is still said by name.
    """

    # Opened with delete sharing (filesystem_paths.open_shared_read): on
    # Windows a plain open would hold the name against the writer's replace,
    # and a reader that polls would starve it (2.9.2 publish, hosted CI).
    from woof.filesystem_paths import read_shared_bytes

    path = Path(path)
    for delay in (*_REPLACED_READ_BACKOFF_SECONDS, None):
        try:
            return json.loads(read_shared_bytes(path).decode("utf-8"))
        except (OSError, ValueError):
            if delay is None:
                raise
            sleep(delay)


def stream_dir(root) -> Path:
    return Path(root) / STREAM_DIRNAME


def segment_marker_path(root, index: int) -> Path:
    return stream_dir(root) / SEGMENTS_DIRNAME / f"{int(index):05d}.json"


def input_plan(manifest: Mapping[str, object], *,
               lead_role_prefix: str, route_table_sha256: str,
               derived_roles=(), fixed_rows=None) -> dict:
    """The input plan a manifest implies: itself with every lead's payload digest blanked.

    ``manifest`` is a route's input manifest document, whose ``files``
    maps a role to ``{"name", "sha256"}``; a role spelled
    ``<lead_role_prefix><digits>`` is one lead's payload.  An as-posted
    head builds the manifest it will seal with those digests unknown
    (``None``) and binds this plan's digest; the seal recomputes it from
    the sealed manifest, so every lead, every object name, the source
    block and every non-lead input (the decoder, the namelist, the
    experiment, the series) are held at the head, and only the payload
    digests, which the posted markers bind lead by lead, arrive later.
    ``derived_roles`` are non-lead inputs whose bytes are written from the
    lead rows as the leads arrive (a series file listing every lead's
    object), so their digest is blanked too: the lead rows the plan keeps
    fix their content, and the route's seal checks it against them.
    ``route_table_sha256`` is the route table the as-posted fetch planned
    and fetched the leads under (its ``schedule.json`` ``table_sha256``,
    DESIGN A136 2.4 item 5); the seal recomputes the plan with the table
    the fetch's schedule names then, so leads fetched under another table
    are another plan.
    """

    if not _is_sha256(route_table_sha256):
        raise BoundaryStreamError(
            f"an input plan binds the route table the fetch planned its "
            f"leads under, and {route_table_sha256!r} is not a sha256")
    if fixed_rows is not None:
        return _composition_inputs_plan(
            manifest, route_table_sha256=route_table_sha256,
            fixed_rows=fixed_rows)
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise BoundaryStreamError("an input manifest carries no files table")
    blanked = {}
    for role, spec in files.items():
        spec = dict(spec)
        if (lead_payload_lead(role, lead_role_prefix) is not None
                or str(role) in set(derived_roles)):
            spec["sha256"] = None
        blanked[str(role)] = spec
    return {"schema": INPUT_PLAN_SCHEMA,
            "route_table_sha256": str(route_table_sha256),
            "manifest": json.loads(_canonical({**dict(manifest),
                                               "files": blanked}))}


#: The sections of a composition-inputs manifest
#: (``gpuwm-mapped-composition-inputs-v1``) whose rows are source data: a
#: lead's objects are among them, and an as-posted plan blanks those rows.
COMPOSITION_DATA_SECTIONS = ("primary_files", "supplements")


def composition_data_rows(manifest: Mapping[str, object]) -> list[dict]:
    """Every data row of a composition-inputs manifest, in document order."""

    rows = []
    primary = manifest.get("primary_files")
    supplements = manifest.get("supplements")
    if not isinstance(primary, list) or not isinstance(supplements, Mapping):
        raise BoundaryStreamError(
            "a composition-inputs manifest carries no primary_files list or "
            "supplements table")
    rows.extend(primary)
    for role in sorted(supplements):
        value = supplements[role]
        rows.extend(value if isinstance(value, list) else [value])
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise BoundaryStreamError(
                "a composition-inputs manifest row names no path")
    return rows


def _composition_inputs_plan(manifest: Mapping[str, object], *,
                             route_table_sha256: str, fixed_rows) -> dict:
    """:func:`input_plan` of a composition-inputs manifest.

    Every data row (the primary files and the supplements) is a lead's
    object unless it is in ``fixed_rows``, the inputs the head read whole
    (a donor analysis fetched before the first lead); a lead row keeps its
    path, and its byte count and digest are blanked, because the posted
    marker of its lead binds them.  Everything else (the mapping and
    composition digests, the member, the provenance and decoder rows) is
    kept whole.
    """

    fixed = {str(path) for path in fixed_rows}

    def blank(row):
        row = dict(row)
        if str(row.get("path")) not in fixed:
            row["bytes"] = None
            row["sha256"] = None
        return row

    document = dict(manifest)
    composition_data_rows(document)  # shape check
    document["primary_files"] = [blank(row) for row in
                                 manifest["primary_files"]]
    document["supplements"] = {
        role: ([blank(row) for row in value] if isinstance(value, list)
               else blank(value))
        for role, value in dict(manifest["supplements"]).items()}
    return {"schema": INPUT_PLAN_SCHEMA,
            "route_table_sha256": str(route_table_sha256),
            "fixed_rows": sorted(fixed),
            "manifest": json.loads(_canonical(document))}


def _is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def posted_lead_marker_sha256(marker: Mapping[str, object]) -> str:
    """The digest a head, a segment and the seal bind a posted-lead marker by."""

    return hashlib.sha256(
        _canonical(dict(marker)).encode("utf-8")).hexdigest()


def lead_payload_lead(role, prefix: str) -> int | None:
    """The lead a manifest role is the payload of, or ``None``."""

    role = str(role)
    if not role.startswith(prefix):
        return None
    rest = role[len(prefix):]
    return int(rest) if rest.isdigit() else None


def input_plan_sha256(plan: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(plan).encode("utf-8")).hexdigest()


def as_posted_placeholder(plan_sha256: str) -> str:
    """What an as-posted head's identity carries where the seal puts a manifest digest."""

    return f"{AS_POSTED_PLACEHOLDER_PREFIX}{plan_sha256}"


def is_as_posted_placeholder(value) -> bool:
    """Whether ``value`` is exactly :func:`as_posted_placeholder` of some plan digest."""

    return (isinstance(value, str)
            and value.startswith(AS_POSTED_PLACEHOLDER_PREFIX)
            and _is_sha256(value[len(AS_POSTED_PLACEHOLDER_PREFIX):]))


def _placeholders_left(identity, path=()) -> list[str]:
    """Every dotted path of ``identity`` that still holds a placeholder."""

    if isinstance(identity, Mapping):
        found = []
        for key in sorted(identity):
            found += _placeholders_left(identity[key], path + (key,))
        return found
    if isinstance(identity, (list, tuple)):
        found = []
        for index, item in enumerate(identity):
            found += _placeholders_left(item, path + (index,))
        return found
    if (isinstance(identity, str)
            and identity.startswith(AS_POSTED_PLACEHOLDER_PREFIX)):
        return [".".join(str(part) for part in path) or "identity"]
    return []


def _identity_changes(head, sealed, path=()) -> list[tuple]:
    """Every leaf where two identities differ, or a structural difference."""

    if isinstance(head, Mapping) and isinstance(sealed, Mapping):
        if set(head) != set(sealed):
            return [(path, "keys", sorted(set(head) ^ set(sealed)))]
        changes = []
        for key in sorted(head):
            changes += _identity_changes(head[key], sealed[key], path + (key,))
        return changes
    if head != sealed:
        return [(path, head, sealed)]
    return []


def check_as_posted_identity(head_identity, sealed_identity, *,
                             plan_sha256: str,
                             manifest_sha256: str,
                             manifest_bound=("input_manifest_sha256",),
                             document_bound: Mapping[str, str] | None = None,
                             ) -> list[str]:
    """Refuse any sealed identity change an as-posted head does not allow.

    Allowed: a key named in :data:`AS_POSTED_IDENTITY_KEYS` or in
    ``manifest_bound`` whose head value is this plan's placeholder and
    whose sealed value is a digest.  ``manifest_bound`` names the route's
    identity keys that carry the input manifest's own digest
    (``input_manifest_sha256`` always does), and each must be exactly
    ``manifest_sha256``, the digest of the manifest the seal wrote, so a
    key the ruling does not name changes only to that same digest.  A
    placeholder left anywhere in the sealed identity is refused: the seal
    would publish a cache no one-shot preparation of the same bytes can
    name.  ``document_bound`` maps an identity key that carries another
    document's digest to the digest the sealed input manifest names for
    that document (:data:`DOCUMENT_ROW_RULES`); each such key may change
    from the placeholder only to exactly that digest.  Returns the dotted
    paths that changed.
    """

    if not _is_sha256(manifest_sha256):
        raise BoundaryStreamError(
            "an as-posted seal names the input manifest it wrote, whose "
            f"digest its identity carries; {manifest_sha256!r} is not one")
    placeholder = as_posted_placeholder(plan_sha256)
    bound = set(manifest_bound) | {"input_manifest_sha256"}
    documents = dict(document_bound or {})
    for key, digest in documents.items():
        if not _is_sha256(digest):
            raise BoundaryStreamError(
                f"an as-posted seal names the document whose digest identity "
                f"key {key!r} carries, and {digest!r} is not a digest")
    # Compared as the header will hold them (a tuple and a list of the
    # same values are one identity).
    head_identity = json.loads(_canonical(head_identity))
    sealed_identity = json.loads(_canonical(sealed_identity))
    changed = []
    for path, before, after in _identity_changes(head_identity,
                                                 sealed_identity):
        where = ".".join(str(part) for part in path) or "identity"
        if before == "keys":
            raise BoundaryStreamError(
                f"the sealed cache identity adds or drops {after} at {where}, "
                "which an as-posted head does not allow")
        key = path[-1] if path else None
        if ((key not in AS_POSTED_IDENTITY_KEYS and key not in bound
             and key not in documents)
                or before != placeholder or not _is_sha256(after)):
            raise BoundaryStreamError(
                f"the sealed cache identity differs from the as-posted head "
                f"at {where}, which only a manifest or composition-receipt "
                "digest may")
        if key in documents:
            if after != documents[key]:
                raise BoundaryStreamError(
                    f"the sealed cache identity names {after} at {where}, "
                    f"not the document the sealed input manifest names there "
                    f"({documents[key]})")
        elif key in bound and after != manifest_sha256:
            raise BoundaryStreamError(
                f"the sealed cache identity names manifest {after} at "
                f"{where}, not the sealed input manifest {manifest_sha256}")
        changed.append(where)
    left = _placeholders_left(sealed_identity)
    if left:
        raise BoundaryStreamError(
            f"the sealed cache identity still carries the as-posted "
            f"placeholder at {', '.join(left)}; the seal writes the digest "
            "there, or the cache names inputs no preparation read")
    return changed


def decoded_lead_record_sha256(record: Mapping[str, str]) -> str:
    """The digest a segment binds one lead's decoded-file record by."""

    return hashlib.sha256(_canonical(dict(record)).encode("utf-8")).hexdigest()


def document_bound_digests(posted: Mapping[str, object],
                           manifest: Mapping[str, object]) -> dict[str, str]:
    """``{identity key: digest}`` an as-posted seal's document-bound keys take.

    ``posted`` is a head's ``basis.as_posted``; each of its
    ``document_bound_identity_keys`` names the input-manifest role of the
    document whose digest the key carries, and ``manifest`` is the sealed
    input manifest, which names that document's digest.
    """

    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    digests = {}
    for key, role in dict(posted.get("document_bound_identity_keys")
                          or {}).items():
        spec = (files or {}).get(role)
        digest = spec.get("sha256") if isinstance(spec, Mapping) else None
        if not _is_sha256(digest):
            raise BoundaryStreamError(
                f"the sealed input manifest names no {role} digest, which the "
                f"as-posted head's identity key {key!r} is bound to")
        digests[str(key)] = str(digest)
    return digests


def _sum_rows(path: Path) -> dict[str, str]:
    """``{name: sha256}`` of a ``sha256sum`` document, refusing a malformed one."""

    rows = {}
    for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        digest, separator, name = line.partition("  ")
        if not separator or not _is_sha256(digest) or not name \
                or name in rows:
            raise BoundaryStreamError(
                f"{path} line {number} is not one sha256sum row")
        rows[name] = digest
    return rows


def hold_document_rows(root, *, posted: Mapping[str, object],
                       manifest: Mapping[str, object],
                       markers: Mapping[str, Mapping[str, object]],
                       decoded: Mapping[str, Mapping[str, str]]) -> None:
    """Hold each document an as-posted seal wrote to its per-lead records.

    ``posted`` is the head's ``basis.as_posted``; its ``documents`` map an
    input-manifest role to ``{path, lead_rows, fixed_rows}``
    (:data:`DOCUMENT_ROW_RULES`).  The file must be the digest the sealed
    manifest names for that role; ``markers`` and ``decoded`` are the
    posted-lead markers and decoded-file records the seal recorded, by
    lead.  The breakage this prevents: a seal that writes a document (and
    so an identity digest) for bytes no lead's marker or decode named.
    """

    files = manifest.get("files") or {}
    for role, document in sorted(dict(posted.get("documents") or {}).items()):
        path = Path(root) / str(document["path"])
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise BoundaryStreamError(
                f"the as-posted seal wrote no {role} document at {path}: "
                f"{error}") from None
        named = (files.get(role) or {}).get("sha256")
        if digest != named:
            raise BoundaryStreamError(
                f"{path} is {digest}, not the {role} digest the sealed input "
                f"manifest names ({named})")
        rows = _sum_rows(path)
        fixed = {str(name) for name in document.get("fixed_rows") or ()}
        rule = document.get("lead_rows")
        if rule == "posted_objects":
            expected = {}
            for lead, marker in markers.items():
                for item in marker.get("objects") or ():
                    expected[str(item.get("name"))] = item.get("sha256")
            lead_rows = {name: value for name, value in rows.items()
                         if name not in fixed}
            if lead_rows != expected:
                differ = sorted(set(lead_rows.items()) ^ set(expected.items()))
                raise BoundaryStreamError(
                    f"{path} rows are not the objects the leads' posted "
                    f"markers named (first difference {differ[:1]})")
        elif rule == "decoded_leads":
            expected = {}
            for lead, record in decoded.items():
                expected.update(dict(record))
            lead_rows = {name: value for name, value in rows.items()
                         if name not in fixed}
            if lead_rows != expected:
                differ = sorted(set(lead_rows.items()) ^ set(expected.items()))
                raise BoundaryStreamError(
                    f"{path} rows are not the files each lead's decode "
                    f"recorded as it was read (first difference "
                    f"{differ[:1]})")
        else:
            raise BoundaryStreamError(
                f"the as-posted head names row rule {rule!r} for {role}, "
                f"not one of {DOCUMENT_ROW_RULES}")
        missing = sorted(fixed - set(rows))
        if missing:
            raise BoundaryStreamError(
                f"{path} has no row for {missing}, which the head named as "
                "its fixed rows")


def _leaves_named(identity, key, path=()) -> list[tuple[tuple, object]]:
    found = []
    if isinstance(identity, Mapping):
        for name in sorted(identity):
            if name == key and not isinstance(identity[name],
                                              (Mapping, list, tuple)):
                found.append((path + (name,), identity[name]))
            found += _leaves_named(identity[name], key, path + (name,))
    elif isinstance(identity, (list, tuple)):
        for index, item in enumerate(identity):
            found += _leaves_named(item, key, path + (index,))
    return found


def posted_identity_leaf(identity, key: str) -> str:
    """The one digest a sealed identity carries under ``key``.

    An as-posted head's user metadata may name a digest its identity also
    carries (the mapped engine's ``composition_receipt_sha256``); its seal
    writes there the value the sealed identity holds, so a key the
    identity does not carry exactly once, as a digest, is refused.
    """

    leaves = _leaves_named(json.loads(_canonical(identity)), str(key))
    if len(leaves) != 1 or not _is_sha256(leaves[0][1]):
        raise BoundaryStreamError(
            f"an as-posted seal writes user metadata {key!r} from the sealed "
            f"identity's one {key!r} digest, and the identity carries "
            f"{len(leaves)} such values")
    return str(leaves[0][1])


def head_sha256(head: Mapping[str, object]) -> str:
    """The digest a runner is pinned to: sha256 of the head's basis."""

    return hashlib.sha256(
        _canonical(head["basis"]).encode("utf-8")).hexdigest()


def read_head(root, *, expected_sha256: str | None = None) -> dict:
    """Load ``head.json`` and check its digest (and the caller's pin)."""

    path = stream_dir(root) / HEAD_NAME
    head = _read_json(path)
    if (not isinstance(head, dict) or head.get("schema") != HEAD_SCHEMA
            or not isinstance(head.get("basis"), dict)):
        raise BoundaryStreamError(f"{path} is not a readable {HEAD_SCHEMA}")
    digest = head_sha256(head)
    if head.get("head_sha256") != digest:
        raise BoundaryStreamError(f"{path} fails its own head digest")
    if expected_sha256 is not None and digest != str(expected_sha256).lower():
        raise BoundaryStreamError(
            f"{path} is head {digest}, not the pinned head {expected_sha256}")
    return head


def bind_head(root, head_sha256: str, *,
              require_manifest: bool = False) -> dict:
    """The head a forecast is started on, checked the same way at every door.

    Refused (``BoundaryStreamError``, which each door wraps in its own
    refusal) when the head fails its digest or the pin, when the
    preparation under it will never finish (failed, stopped or silent,
    :func:`unfinished_tree_reason`), which is the breakage this prevents:
    a forecast started on boundaries that never arrive; and, with
    ``require_manifest``, when the head names no source manifest for the
    forecast to bind.  An as-posted head names none by design: it binds
    its input plan (``basis.as_posted``), which its head digest covers, and
    the manifest its seal writes is held to that plan
    (:func:`verify_as_posted_seal`).
    """

    root = Path(root)
    head = read_head(root, expected_sha256=head_sha256)
    reason = unfinished_tree_reason(root)
    if reason is not None:
        raise BoundaryStreamError(
            f"the prepared head in {root} will never be sealed: {reason}")
    if require_manifest and head["basis"].get("as_posted") is None \
            and not isinstance(
            head["basis"].get("input_manifest_sha256"), str):
        raise BoundaryStreamError(
            f"the prepared head in {root} names no source manifest, so the "
            "forecast has nothing to bind it to")
    return head


def live_chained_head(root) -> dict | None:
    """The chained head of a preparation still being produced, or ``None``.

    ``None`` for a sealed tree, a tree published at its seal, and a
    preparation that will never finish.
    """

    root = Path(root)
    if prepared_tree_complete(root) or unfinished_tree_reason(root):
        return None
    stream = stream_dir(root)
    if (stream / FAILED_NAME).exists() or (stream / STOP_NAME).exists():
        return None
    try:
        head = read_head(root)
    except BoundaryStreamError:
        return None
    if not (head.get("decision") or {}).get("chained", False):
        return None
    return head


def proof_document_name(head: Mapping[str, object] | None) -> str:
    """The file a head's seal writes its proof to.

    ``proof.json`` unless the head names another (``basis.proof_name``): a
    native HRRR domain tree's sealed document has always been
    ``receipt.json``, and its chained head keeps that name so the sealed
    tree is the one-shot tree file for file.
    """

    if head is None:
        return PROOF_NAME
    return str((head.get("basis") or {}).get("proof_name") or PROOF_NAME)


def prepared_tree_complete(root) -> bool:
    """A prepared tree is complete when its seal document exists.

    That is ``proof.json``, or the document its head names
    (:func:`proof_document_name`).  The one test for "prepared", so no code
    reads "the output root exists" as "the preparation finished" once a
    head can be published early.
    """

    root = Path(root)
    if (root / PROOF_NAME).is_file():
        return True
    head = _read_json(stream_dir(root) / HEAD_NAME)
    if not isinstance(head, dict) or not isinstance(head.get("basis"), dict):
        return False
    name = proof_document_name(head)
    return name != PROOF_NAME and (root / name).is_file()


def unfinished_tree_reason(root, *, now: float | None = None) -> str | None:
    """Why ``root`` is a head without a seal that nothing will finish.

    ``None`` means the tree is complete, absent, or still being produced by
    a live producer.  Otherwise the reason names the marker that decided it.
    """

    root = Path(root)
    if not root.exists() or prepared_tree_complete(root):
        return None
    stream = stream_dir(root)
    if not (stream / HEAD_NAME).is_file():
        return None
    failed = _read_json(stream / FAILED_NAME)
    if isinstance(failed, dict):
        return f"its producer failed: {failed.get('reason')}"
    stop = _read_json(stream / STOP_NAME)
    if isinstance(stop, dict):
        return f"its forecast stopped it: {stop.get('reason')}"
    beat = _read_json(stream / PRODUCER_NAME)
    age = _heartbeat_age(beat, now=now)
    if age is None or age > _silence_limit(beat):
        return ("its producer has been silent for "
                f"{'an unknown time' if age is None else f'{age:.0f} s'}")
    return None


def remove_unfinished_tree(root, *, log=None) -> bool:
    """Remove an unfinished tree so the next preparation can rebuild it.

    It is this tool's own staging product: a head published early whose
    producer failed, was stopped or went silent.  Refused for anything
    else, which is the breakage this prevents: deleting a complete
    preparation or one a live producer is still writing.
    """

    reason = unfinished_tree_reason(root)
    if reason is None:
        return False
    (log or _say)(f"prepare: removing the unfinished preparation {root} "
                  f"({reason}) and building it again")
    shutil.rmtree(root)
    return True


#: Share of the HOST RAM a chained forecast and its producer leave
#: unclaimed (:func:`host_admission`).  The card's side prices the
#: forecast's whole-process envelope instead (:func:`chained_admission`).
CHAINED_HEADROOM = 0.10
_GIB = float(1 << 30)


def producer_device_bytes(backend: str, *, selection=None) -> int | None:
    """What a CUDA producer holds right after building one forcing time.

    The CuPy pool keeps the blocks a build used, so its total after the
    start time is built is that build's device footprint. Chunked preparation
    also reserves its admitted peak: the next hour may require a larger
    chunk than the initial state's measured pool. ``None`` for a host producer.
    """

    if str(backend) != "cuda":
        return None
    bound = None
    if isinstance(selection, Mapping) and selection.get("chunking"):
        fit = selection.get("device_fit") or {}
        if fit.get("need_bytes") is not None:
            bound = int(fit["need_bytes"])
    try:
        import cupy

        return max(int(cupy.get_default_memory_pool().total_bytes()), bound or 0)
    except Exception:  # noqa: BLE001 - no device, nothing to measure
        return bound


def _measure_card() -> tuple[int, int] | None:
    try:
        import cupy

        free, _total = cupy.cuda.runtime.memGetInfo()
        return int(free), int(cupy.get_default_memory_pool().total_bytes())
    except Exception:  # noqa: BLE001 - no device to measure
        return None


def prepared_head_urban_columns(experiment, root_static, *, child_results=()):
    """The prepared head's urban count, using the runner's land-cover price."""
    if not forecast_installed():
        return None
    from woof.core.urban_state import bem_workspace_counted

    if not any(bem_workspace_counted(dc.run) for dc in experiment.domains):
        return None
    from types import SimpleNamespace
    from woof.prepared_domain_tree_forecast import tree_urban_columns

    domains = [SimpleNamespace(grid_id=int(experiment.root.grid_id),
                               static_fields=root_static)]
    domains.extend(SimpleNamespace(grid_id=int(result.domain.grid_id),
                                   static_fields=getattr(result, "static_fields", None))
                   for result in child_results
                   if getattr(result, "domain", None) is not None)
    return tree_urban_columns(SimpleNamespace(experiment=experiment,
                                              domains=domains))


def chained_admission(*, experiment, backend: str,
                      device_bytes: int | None = None,
                      card: tuple[int, int] | None = None,
                      source=None, urban_columns=None) -> dict:
    """Whether a forecast may run beside this producer; see ``admit``.

    ``source`` is what the producer prepares from (a registered name, or
    the mapping document a mapped route reads), so the forecast is priced
    with the analysed hydrometeor tables that source puts on its boundary.
    ``urban_columns`` is the prepared head's land-cover count; an unknown
    domain retains the configuration's BEP+BEM workspace upper bound.

    The forecast is priced at its whole-process PEAK ENVELOPE, the number
    ``woof check`` and the resident door refuse on (A163: one figure on
    every surface), and the card keeps
    :data:`~woof.core.preflight.EXTERNAL_MARGIN_BYTES` for other
    processes, as those doors do.  It used to be the pool estimate
    against 90% of the card (:data:`CHAINED_HEADROOM`): the 10% stood in
    for the forecast's CUDA context and local-memory backing store, which
    the envelope itemizes, and on a 16 GiB card it was 1.6 GiB against
    1.6-1.9 GiB of itemized non-pool residency, so a pair could be
    admitted that the forecast's own envelope does not fit.  The
    breakage this prevents is that out-of-memory, in the forecast or in
    the producer beside it.
    """

    ranked = getattr(getattr(experiment, "devices", None), "enabled", False)
    ranked_price = None
    if ranked:
        from woof.core.devices_memory import estimate_devices, estimate_devices_tree

        price = (estimate_devices_tree if len(experiment.domains) > 1
                 else estimate_devices)
        ranked_price = price(experiment, streaming_boundaries=True, source=source)
    host_terms = ({} if ranked_price is None else {
        # The usual head/series estimate does not include the pinned store,
        # geography and seam staging held by a ranked forecast.
        "forecast_store_host_bytes": int(ranked_price["host_store_bytes"])
                                     + int(ranked_price["host_staging_bytes"]),
    })
    if str(backend) != "cuda":
        # The card is not shared; host RAM is, and write_head prices it
        # for every backend (host_admission).
        return {"admitted": True, "device": "host", **host_terms,
                "reason": "the producer prepares on the host, so the "
                          "forecast's card is not shared"}
    from woof.core.preflight import EXTERNAL_MARGIN_BYTES, admission_estimate

    pricing = {"source": source}
    if urban_columns is not None:
        pricing["urban_columns"] = urban_columns
    if ranked_price is None:
        forecast_bytes = int(admission_estimate(
            experiment, **pricing).peak_envelope_bytes)
        producer_card = None
    else:
        producer_card = int(experiment.devices.device_ids()[0])
        if card is None:
            try:
                import cupy

                producer_card = int(cupy.cuda.Device().id)
            except Exception:  # noqa: BLE001 - measurement below refuses
                pass
        forecast_bytes = next((int(row["total_bytes"])
                               for row in ranked_price["cards"]
                               if int(row["card"]) == producer_card), 0)
    card = _measure_card() if card is None else card
    if card is None or device_bytes is None:
        return {"admitted": False, "device": "cuda", **host_terms,
                "forecast_bytes": forecast_bytes,
                "reason": ("chained preparation not admitted: the producer "
                           "prepares on the GPU and its memory could not be "
                           "measured, so a forecast beside it could run "
                           "both out of memory")}
    free, pool = card
    budget = int(free + pool) - int(EXTERNAL_MARGIN_BYTES)
    need = forecast_bytes + int(device_bytes)
    admitted = need <= budget
    return {
        "admitted": admitted, "device": "cuda",
        "forecast_bytes": forecast_bytes, "producer_bytes": int(device_bytes),
        **host_terms,
        **({} if producer_card is None else {"producer_card": producer_card}),
        "budget_bytes": budget,
        "reason": (
            f"forecast {forecast_bytes / _GIB:.2f} GiB + preparation "
            f"{device_bytes / _GIB:.2f} GiB fit {budget / _GIB:.2f} GiB "
            "on the GPU" if admitted else
            f"chained preparation not admitted: forecast "
            f"{forecast_bytes / _GIB:.2f} GiB + preparation "
            f"{device_bytes / _GIB:.2f} GiB > {budget / _GIB:.2f} GiB on "
            "the GPU"),
    }


def process_memory_bytes() -> tuple[int, int] | None:
    """This process's ``(resident, peak resident)`` bytes, or ``None``.

    Linux reads ``VmRSS`` and ``VmHWM`` from ``/proc/self/status``;
    Windows reads the working set and its peak.  Anything else is
    unmeasured.
    """

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _Counters()
            counters.cb = ctypes.sizeof(counters)
            psapi = ctypes.WinDLL("psapi")
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(
                    kernel32.GetCurrentProcess(), ctypes.byref(counters),
                    counters.cb):
                return None
            return (int(counters.WorkingSetSize),
                    int(counters.PeakWorkingSetSize))
        except Exception:  # noqa: BLE001 - unmeasured, said so by None
            return None
    values = {}
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                name, _, rest = line.partition(":")
                if name in {"VmRSS", "VmHWM"}:
                    values[name] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    if set(values) != {"VmRSS", "VmHWM"}:
        return None
    return values["VmRSS"], values["VmHWM"]


def _host_available() -> int | None:
    try:
        from woof.core.preflight import host_available_bytes

        return host_available_bytes()
    except Exception:  # noqa: BLE001 - unmeasured, said so by None
        return None


#: Host RAM a forecast process holds whatever its domain: the interpreter,
#: NumPy and CuPy, the CUDA context and runtime, the physics tables and the
#: host copies a restore makes on the way to the card.  Measured as
#: ``memory.cpu_peak_rss_bytes`` in the ``report.json`` of whole forecasts
#: on domains whose own arrays are a few hundred MiB at most: 1.34 GiB at
#: 80 x 64 x 32, 1.58 GiB at 73 x 73 x 49, 1.86 GiB at 200 x 200 x 49 and
#: 1.87 GiB at 162 x 162 x 49.  The floor sits above all four.  With the
#: head and the whole boundary series added
#: (:func:`forecast_host_bytes`), the 1792 x 1024 x 55 CONUS forecast of
#: 2026-09-28 prices at 13.6 GiB against its measured 11.2 GiB peak.
FORECAST_HOST_FLOOR_BYTES = 2 * (1 << 30)


def forecast_host_bytes(*, head_payload_bytes: int,
                        interval_host_bytes: int | None,
                        intervals: int) -> dict:
    """What a chained forecast holds in host RAM, in its three parts.

    The head's arrays, read into host RAM by the restore; the whole
    boundary series, because :class:`StreamedIntervals` keeps every
    interval it loads for the rest of the run; and
    :data:`FORECAST_HOST_FLOOR_BYTES`, the process itself.
    ``interval_host_bytes`` is one loaded interval (``None`` when it could
    not be priced, and then so is the total).
    """

    series = (None if interval_host_bytes is None
              else int(interval_host_bytes) * int(intervals))
    return {
        "head_payload_bytes": int(head_payload_bytes),
        "boundary_series_bytes": series,
        "process_floor_bytes": FORECAST_HOST_FLOOR_BYTES,
        "total_bytes": (None if series is None else
                        int(head_payload_bytes) + series
                        + FORECAST_HOST_FLOOR_BYTES),
    }


def host_admission(*, forecast_bytes: int | None,
                   producer_bytes: int | None = None,
                   available_bytes: int | None = None,
                   producer_peak_bytes: int | None = None,
                   producer_peak_required: bool = False,
                   producer_decode_bytes: int | None = 0) -> dict:
    """Whether the machine's RAM holds a forecast beside its producer.

    Before chaining, host RAM held the preparation and then the forecast;
    chained, it holds both at once, on every producer backend.  The
    breakage this prevents is a machine that runs out of RAM mid-run with
    both of them half done (on Linux the kernel kills a process from
    outside, with no woof message).  ``available_bytes`` is the RAM
    available right after the start time was built, so everything the
    producer keeps is already out of it; ``producer_bytes`` is what the
    producer's builds take on top of that (its measured peak resident
    size less its resident size now). A bounded host-retained producer
    also supplies its priced absolute ``producer_peak_bytes``: the larger
    of that future bound and the observed peak is reduced by current RSS,
    whose bytes are already excluded from available RAM. ``forecast_bytes`` is what the
    forecast process holds in host RAM (:func:`forecast_host_bytes`), or
    ``None`` when it could not be priced.  Admitted with 10% of the
    available RAM left over, otherwise the tree is published at its seal
    as before.
    """

    available = (_host_available() if available_bytes is None
                 else int(available_bytes))
    memory = None
    if producer_bytes is None or producer_peak_bytes is not None:
        memory = process_memory_bytes()
        producer_bytes = (None if memory is None
                          else max(0, memory[1], int(producer_peak_bytes or 0)) - memory[0])
        if producer_bytes is not None:
            producer_bytes = max(0, producer_bytes)
    if producer_peak_required and producer_peak_bytes is None:
        producer_bytes = None
    # A later posted source can launch a separate native decoder. Its RSS
    # is absent from this process's VmHWM. Although decode precedes REAL,
    # boundaries and source metadata can remain from earlier builds. Keep
    # the parent growth allowance beside the whole child reservation.
    if producer_decode_bytes is None:
        producer_bytes = None
    elif producer_bytes is not None:
        producer_bytes = int(producer_bytes) + max(0, int(producer_decode_bytes))
    if available is None or producer_bytes is None or forecast_bytes is None:
        return {"admitted": False, "memory": "host",
                "forecast_host_bytes": (None if forecast_bytes is None
                                        else int(forecast_bytes)),
                "producer_decode_host_bytes": producer_decode_bytes,
                "reason": ("chained preparation not admitted: the future source "
                           "decoder's host peak has no measured, enforceable native "
                           "budget; the forecast starts after preparation"
                           if producer_decode_bytes is None else
                           "chained preparation not admitted: this "
                           "machine's available RAM, the producer's own "
                           "or the forecast's could not be measured, so a "
                           "forecast beside it could run the machine out "
                           "of RAM")}
    budget = int(available * (1.0 - CHAINED_HEADROOM))
    need = int(forecast_bytes) + int(producer_bytes)
    admitted = need <= budget
    return {
        "admitted": admitted, "memory": "host",
        "forecast_host_bytes": int(forecast_bytes),
        "producer_host_bytes": int(producer_bytes),
        "producer_decode_host_bytes": int(producer_decode_bytes),
        **({} if producer_peak_bytes is None else {
            "producer_priced_peak_bytes": int(producer_peak_bytes),
            "producer_resident_bytes": int(memory[0]),
            "producer_observed_peak_bytes": int(memory[1]),
        }),
        "host_budget_bytes": budget,
        "reason": (
            f"forecast {forecast_bytes / _GIB:.2f} GiB + preparation "
            f"{producer_bytes / _GIB:.2f} GiB fit {budget / _GIB:.2f} GiB "
            "of host RAM" if admitted else
            f"chained preparation not admitted: forecast "
            f"{forecast_bytes / _GIB:.2f} GiB + preparation "
            f"{producer_bytes / _GIB:.2f} GiB > {budget / _GIB:.2f} GiB of "
            "available host RAM"),
    }


def request_stop(root, reason: str) -> bool:
    """Ask a producer to exit unsealed: its forecast ended first.

    Writes ``stop.json`` only beside a head without a seal; the producer
    reads it between forcing times and on each heartbeat.
    """

    root = Path(root)
    if prepared_tree_complete(root) or not (
            stream_dir(root) / HEAD_NAME).is_file():
        return False
    try:
        _write_json_atomic(stream_dir(root) / STOP_NAME, {
            "reason": str(reason), "stopped_utc": _utc_now()})
    except OSError:
        return False
    return True


#: The preparation threads :func:`run_chained` is running, by output root,
#: so a forecast in the same process can tell a producer that ended from
#: one that is still building (see :meth:`StreamedIntervals._producer_verdict`).
_LOCAL_PRODUCERS: dict[str, threading.Thread] = {}
_LOCAL_PRODUCERS_LOCK = threading.Lock()


def _root_key(root) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(root)))


def _local_producer(root) -> threading.Thread | None:
    with _LOCAL_PRODUCERS_LOCK:
        return _LOCAL_PRODUCERS.get(_root_key(root))


def _head_created_epoch(head: Mapping[str, object]) -> float | None:
    try:
        created = datetime.fromisoformat(str(head["created_utc"]))
    except (KeyError, TypeError, ValueError):
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return created.timestamp()


def run_chained(*, prepared_root, prepare: Callable[[], object],
                forecast: Callable[[str | None], object],
                on_head: Callable[[str], None] | None = None,
                poll_seconds: float = 0.5, observer=None):
    """Run ``prepare`` beside ``forecast``; the forecast starts at the head.

    The one orchestration every chain uses.  ``prepare`` runs the
    preparation to its seal on a worker thread (the preparation itself is
    whatever the chain already runs: a stage subprocess, or the preparer
    in a spawned process); this thread waits until
    ``<prepared_root>/boundary-stream/head.json`` is written by THIS
    preparation or the preparation ends, then calls
    ``forecast(head_sha256)`` for a chained head, or ``forecast(None)``
    for a tree that was published sealed or reused whole (so the caller
    binds its proof as before).

    Only a head created after this call started is bound, which is the
    breakage this prevents: a retry that reuses its output root finds the
    previous attempt's head there, and a forecast bound to it dies on that
    attempt's failure marker or runs on a preparation the retry is about
    to replace.

    A producer that fails reaches the waiting forecast through
    ``failed.json`` and, in this process, through the worker thread
    having ended.  A forecast that fails is said at once and stops its
    producer (:func:`_stop_for_failed_forecast`), so the run ends with the
    forecast's error instead of after every later forcing time is built.
    A stop asks the producer to stop (``stop.json``), waits for it to exit
    and re-raises:
    ``KeyboardInterrupt``, ``SystemExit``, and every exception
    :func:`woof.runplan._is_interrupt` reads as the user's stop, which
    is how a Ctrl-C reaches this thread from ``woof go``
    (``GoInterrupted``) and ``woof run-plan``.  The breakage this
    prevents: a stopped ``woof go`` reported a failed forecast, left the
    producer running and did not return until it sealed.

    Returns ``(prepare_result, forecast_result)``.
    """

    root = Path(prepared_root)
    box: dict[str, object] = {}
    started_epoch = time.time()

    def produce():
        try:
            box["result"] = prepare()
        except BaseException as error:  # noqa: BLE001 - re-raised below
            box["error"] = error
            _mark_failed_if_unsealed(root, error, since=started_epoch)

    # In the caller's context, as the preparation would be on the caller's
    # own thread: a host's output redirect (woof go's launch log) is a
    # context variable, and a thread starts without it.  The breakage this
    # prevents: under `woof go` a chained preparation missed the redirect,
    # so its preparer ran behind a second host with its own log and each
    # step was said twice under --explain.
    context = contextvars.copy_context()
    worker = threading.Thread(target=context.run, args=(produce,),
                              name="chained-preparation", daemon=True)
    key = _root_key(root)
    with _LOCAL_PRODUCERS_LOCK:
        _LOCAL_PRODUCERS[key] = worker
    try:
        worker.start()
        head_path = stream_dir(root) / HEAD_NAME
        head = None
        while worker.is_alive():
            if head_path.is_file():
                head = _fresh_head(root, since=started_epoch)
                if head is not None:
                    break
            worker.join(poll_seconds)
        if head is not None and not (head.get("decision") or {}).get(
                "chained", False):
            # Published at its seal: nothing to start early.
            head = None
        if head is None:
            worker.join()
            if "error" in box:
                raise box["error"]
            return box.get("result"), forecast(None)
        if on_head is not None:
            on_head(str(head["head_sha256"]))
        try:
            result = forecast(str(head["head_sha256"]))
        except BaseException as error:
            if isinstance(error, Exception) and not _is_stop(error):
                _stop_for_failed_forecast(root, worker, error)
                raise
            request_stop(root, f"the forecast ended: "
                               f"{type(error).__name__}: {error}")
            worker.join()
            raise
        worker.join()
        if "error" in box:
            raise box["error"]
        return box.get("result"), result
    finally:
        with _LOCAL_PRODUCERS_LOCK:
            if _LOCAL_PRODUCERS.get(key) is worker:
                del _LOCAL_PRODUCERS[key]


def _stop_for_failed_forecast(root, worker: threading.Thread,
                              error: BaseException) -> None:
    """Say a failed forecast at once, stop its producer, and let it raise.

    The failure is said the moment the forecast raises, the producer is
    asked to stop (``stop.json``) and is waited for, so the run ends with
    the forecast's own error, its cards released, instead of building every
    later forcing time first.

    Named breakage: a forecast that failed 25 s after its preflight (test host
    run 2a and test host r1b/r1d, 2026-10-21, a store refusal) left its producer
    building all 48 remaining boundary times for 19-24 min while the run
    said waiting:preparation with every card idle, and only then raised;
    the wait read as a streaming regression.  A retry rebuilds the stopped
    preparation's later times, which is minutes; hiding a failure is not.
    """

    _say(f"prepare: the forecast failed ({type(error).__name__}: {error}); "
         "stopping the preparation")
    if worker.is_alive():
        request_stop(root, f"the forecast failed: "
                           f"{type(error).__name__}: {error}")
    worker.join()


def _is_stop(error: BaseException) -> bool:
    """Whether a forecast exception is the user's stop, not a failure."""

    from woof.runplan import _is_interrupt

    return _is_interrupt(error)


def _fresh_head(root, *, since: float) -> dict | None:
    """The readable head in ``root`` if it was created at or after ``since``."""

    try:
        head = read_head(root)
    except BoundaryStreamError:
        return None
    created = _head_created_epoch(head)
    if created is None or created < since:
        return None
    return head


def fresh_chained_head(root, *, since: float) -> dict | None:
    """The chained head ``root`` got at or after ``since``, or ``None``.

    What :func:`run_chained` waits for, for a chain with a stage between
    two preparations (a native HRRR tree's hierarchy starts on its root
    preparation's head): a head from an earlier attempt, or one published
    at its seal, is not one to start on.
    """

    head = _fresh_head(root, since=since)
    if head is None or not (head.get("decision") or {}).get(
            "chained", False):
        return None
    return head


def _mark_failed_if_unsealed(root, error: BaseException, *,
                             since: float) -> None:
    """Write ``failed.json`` for a head this run published and never sealed.

    The preparation normally writes it itself (:meth:`PreparedTreeWriter
    .fail`); this covers the producer that could not, such as a stage
    subprocess killed from outside, so a forecast in another process ends
    by name instead of waiting for the heartbeat to go stale.
    """

    root = Path(root)
    stream = stream_dir(root)
    if (prepared_tree_complete(root) or (stream / FAILED_NAME).exists()
            or _fresh_head(root, since=since) is None):
        return
    try:
        _write_json_atomic(stream / FAILED_NAME, {
            "reason": str(error) or type(error).__name__,
            "stage": "producing",
            "error_type": type(error).__name__,
            "failed_utc": _utc_now(),
        })
    except OSError:
        pass


def _heartbeat_age(beat, *, now: float | None = None) -> float | None:
    if not isinstance(beat, dict):
        return None
    try:
        updated = float(beat["updated_epoch"])
    except (KeyError, TypeError, ValueError):
        return None
    return max(0.0, (time.time() if now is None else now) - updated)


def _same_process(beat) -> bool:
    return (isinstance(beat, dict) and beat.get("pid") == os.getpid()
            and beat.get("host") == socket.gethostname())


def _silence_limit(beat) -> float:
    slowest = 0.0
    if isinstance(beat, dict):
        try:
            slowest = float(beat.get("slowest_build_seconds") or 0.0)
        except (TypeError, ValueError):
            slowest = 0.0
    return max(SILENT_FLOOR_SECONDS, 3.0 * slowest)


# ---------------------------------------------------------------------------
# The producer
# ---------------------------------------------------------------------------


class _ConsoleAfterReader:
    """A console stream that outlives the process reading it.

    Writes go to ``stream`` until it reports the reader gone (a broken
    pipe); from then on they are dropped, since nothing is left to read
    them.  Everything else is the wrapped stream's.
    """

    def __init__(self, stream):
        self._stream = stream
        self.reader_gone = False

    def _lost(self) -> None:
        self.reader_gone = True
        self._stream = open(os.devnull, "w", encoding="utf-8")

    def write(self, text):
        if not self.reader_gone:
            try:
                return self._stream.write(text)
            except (BrokenPipeError, ConnectionResetError):
                self._lost()
        return len(text)

    def flush(self):
        if not self.reader_gone:
            try:
                self._stream.flush()
            except (BrokenPipeError, ConnectionResetError):
                self._lost()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def keep_console_after_reader_exit() -> None:
    """Let a chained producer survive the death of its console's reader.

    A preparation stage prints its progress into a pipe its parent reads,
    and in a ``woof go`` chain that parent also hosts the forecast.  When
    that process dies (killed, out of memory, a driver crash), the
    producer's next line raised a broken pipe and the preparation failed,
    which is the breakage this prevents: a failed forecast leaves its
    producer running to the seal so a retry reuses it, and a checkpoint
    written before the seal resumes only on that same preparation.
    Idempotent, and a no-op for a console nobody closes.
    """

    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is not None and not isinstance(stream, _ConsoleAfterReader):
            setattr(sys, name, _ConsoleAfterReader(stream))


class PreparedTreeWriter:
    """Write a prepared tree as a head, one segment per interval, a seal.

    ``staging`` is the route's own staging directory and ``output_root``
    the name its single rename publishes.  With ``chained`` on,
    :meth:`write_head` performs that rename right after the head is
    written; otherwise :meth:`publish` performs it after ``proof.json``, so
    the unchained run is today's run through the same code.  Either way
    :attr:`root` is where the tree currently lives, and every path a route
    writes after the head is taken from it.
    """

    def __init__(self, *, staging, output_root, identity,
                 chained: bool | None = None,
                 sealed_forcing_extension: bool = False,
                 cache_name: str = "prepared-cache",
                 publish: Callable[[Path, Path], None] | None = None,
                 proof_name: str = PROOF_NAME):
        from woof.ingest.prepared_cache import PreparedCacheStream
        from woof.ensemble.posted_preparation import (
            current_posted_domain, bind_current_prepared_identity)
        member_source = current_posted_domain()
        if member_source is not None:
            identity = bind_current_prepared_identity(identity)
            self._posted_member_source = member_source

        self.staging = Path(staging)
        self.output_root = Path(output_root)
        self.chained = chained_enabled() if chained is None else bool(chained)
        self.cache_name = str(cache_name)
        #: The document :meth:`publish` writes last; a head names it in
        #: ``basis.proof_name`` only when it is not ``proof.json``.
        self.proof_name = str(proof_name)
        self._publish_tree = publish or (lambda src, dst: os.replace(src, dst))
        self.published = False
        self.head: dict | None = None
        self.head_sha256: str | None = None
        self._cache = PreparedCacheStream(
            self.staging / self.cache_name, identity=identity,
            sealed_forcing_extension=sealed_forcing_extension)
        self._heartbeat: _Heartbeat | None = None
        self._segments_written = 0
        self._build_seconds: list[float] = []
        self._decision: dict[str, object] = {"chained": self.chained}
        self.forecast_installed = forecast_installed()
        if not self.forecast_installed:
            # Nothing here can bind the head, and admission cannot be
            # priced without the forecast (see forecast_installed).
            self.chained = False
            self._decision = {"chained": False,
                              "reason": PREPARATION_ONLY_REASON}
        self.head_seconds: float | None = None
        self._started = time.perf_counter()
        #: What the heartbeat says this producer is doing: ``producing``,
        #: or ``waiting_for_source`` with the lead in ``_waiting_for``.
        self._state = "producing"
        self._waiting_for: dict | None = None
        #: The last lead a source wait ended on, with when it was first
        #: seen posted, so a consumer closing its own wait can say so.
        self._arrived: dict | None = None
        # Only selected native physical providers populate these. Ordinary
        # heads and segment bytes omit the entire optional binding.
        self._physical_receipts = None
        self._physical_seen = {}

    def waiting_for_source(self, waiting_for: Mapping[str, object]) -> None:
        """Say on the heartbeat that this producer waits for a source lead.

        ``waiting_for`` names :data:`WAITING_FOR_FIELDS`.  A forecast
        waiting at a seam reads it to say its wait is on the source
        (``waiting:source``) rather than on the preparation.  Said again
        for the lead already waited on (the fetch's word on it changed),
        the wait keeps the time it began.
        """

        record = {key: waiting_for.get(key) for key in WAITING_FOR_FIELDS}
        if record["since_utc"] is None:
            held = self._waiting_for
            same = held is not None and all(
                held.get(key) == record[key]
                for key in ("source", "cycle", "lead"))
            record["since_utc"] = held["since_utc"] if same else _utc_now()
        self._state, self._waiting_for = "waiting_for_source", record
        if self._heartbeat is not None:
            self._heartbeat.beat()

    def source_arrived(self, *, first_seen_at: str | None = None) -> None:
        """End a :meth:`waiting_for_source`: the lead posted, building resumes."""

        if self._waiting_for is not None:
            self._arrived = {
                "source": self._waiting_for.get("source"),
                "cycle": self._waiting_for.get("cycle"),
                "lead": self._waiting_for.get("lead"),
                "first_seen_at": first_seen_at,
            }
        self._state, self._waiting_for = "producing", None
        if self._heartbeat is not None:
            self._heartbeat.beat()

    @property
    def root(self) -> Path:
        return self.output_root if self.published else self.staging

    @property
    def cache_path(self) -> Path:
        return self.root / self.cache_name

    @property
    def stream_path(self) -> Path:
        return stream_dir(self.root)

    def decline_chaining(self, reason: str) -> None:
        """Publish at the seal instead of the head, with the named reason."""

        if self.head is not None:
            raise RuntimeError("chaining is decided before the head")
        if self.chained:
            _say(_sealed_line(reason))
        self.chained = False
        self._decision = {"chained": False, "reason": str(reason)}

    def admit(self, *, experiment, backend: str,
              device_bytes: int | None = None,
              card: tuple[int, int] | None = None,
              source=None, urban_columns=None, preprocess_selection=None,
              future_decode_host_bytes: int | None = 0) -> dict:
        """Admit the forecast and this producer on one machine, or decline.

        The breakage this prevents is both processes out of memory mid-run:
        a chained forecast starts on the card while the producer is still
        building forcing times.  A host (CPU) producer never shares the
        forecast's device and is admitted.  A CUDA producer is admitted
        when the forecast's peak envelope plus the producer's own
        measured holding for one forcing time (``device_bytes``, the
        memory pool after the start time was built) fits what the card
        can give both, less the other-process margin every memory door
        keeps; otherwise the head is published
        at the seal and the run starts after preparation, as before.
        ``card`` is ``(free_bytes, pool_bytes)`` when the caller measured
        it; otherwise it is measured here.  ``source`` is what this
        producer prepares from, whose published hydrometeors the
        forecast's boundary tables carry (:func:`chained_admission`).  An
        installation with no forecast has nothing to admit and keeps the
        decision it recorded at construction.
        """

        if not self.forecast_installed:
            return dict(self._decision)
        decision = chained_admission(
            experiment=experiment, backend=backend,
            device_bytes=device_bytes, card=card, source=source,
            urban_columns=urban_columns)
        decision["producer_decode_host_bytes"] = future_decode_host_bytes
        if (backend == "cuda" and isinstance(preprocess_selection, Mapping)
                and preprocess_selection.get("chunking")):
            host_fit = preprocess_selection.get("host_fit") or {}
            decision["producer_host_peak_bytes"] = host_fit.get("producer_peak_bytes")
            decision["producer_host_peak_required"] = True
            # Chunked preparation releases its pool between batches. The
            # forecast must leave room for the next batch even when those
            # bytes are free at the instant it queries the card.
            decision["producer_reserve_bytes"] = producer_device_bytes(
                backend, selection=preprocess_selection)
            if "producer_card" not in decision:
                try:
                    import cupy

                    decision["producer_card"] = int(cupy.cuda.Device().id)
                except Exception:  # noqa: BLE001 - admission already refused
                    pass
            if "producer_card" in decision:
                from woof.core.device_probe import cuda_device_identity

                identity = cuda_device_identity(decision["producer_card"])
                decision["producer_device_identity"] = identity
            if (decision.get("producer_reserve_bytes") is not None
                    and "producer_card" in decision):
                # A forecast may select its cards or host store after this
                # preparation starts. Publish the bounded producer contract;
                # the consumer prices its actual road before allocating.
                decision["configured_forecast_admitted"] = decision["admitted"]
                decision["configured_forecast_reason"] = decision["reason"]
                decision["admitted"] = True
                decision["consumer_admission_required"] = True
                decision["reason"] = (
                    "bounded preparation publishes its head; the forecast "
                    "admits its selected cards and store while reserving "
                    "the producer's next batch")
        if self.chained and not decision["admitted"]:
            self.decline_chaining(decision["reason"])
        self._decision = {"chained": self.chained, **decision}
        return dict(self._decision)

    def write_head(self, *, initial_result, met, surface=None, metadata=None,
                   lbc, proof_head: Mapping[str, object],
                   input_manifest_sha256: str | None = None,
                   reservation: Mapping[str, object] | None = None,
                   forcing=None,
                   tree: Mapping[str, object] | None = None,
                   extra_head_payload_bytes: int = 0,
                   seal_completes: Sequence[str] = (),
                   as_posted: Mapping[str, object] | None = None,
                   ensemble_physical: Mapping[str, object] | None = None) -> str:
        """Write everything the start time makes; publish it when chained.

        ``lbc`` is ``{"spec_bdy_width", "spec_zone", "relax_zone",
        "schedule": [[start, end], ...], "fields": [...]}``.  ``proof_head``
        is the route's proof document without the seal-only keys.
        ``forcing`` is the route's
        :class:`woof.ingest.lateral_bc.StateBoundaryFrames` holding the
        start time, which prices one boundary interval for the host RAM
        admission; a chained head without it is not admitted, because the
        boundary series the forecast will hold would be unpriced.

        ``tree`` makes this a domain tree's head (see
        :func:`domain_tree_head_fields`): ``head.json`` then carries
        ``layout``, ``domains`` and ``children_artifacts``, and the same
        three under ``basis.tree`` so the head digest binds them.
        ``extra_head_payload_bytes`` is what the children's prepared
        caches add to the forecast's host RAM.  ``seal_completes`` names
        user metadata the start time holds only part of, which
        :meth:`seal_cache` completes
        (:meth:`woof.ingest.prepared_cache.PreparedCacheStream.write_head`).

        ``as_posted`` makes this an as-posted head (see
        :data:`AS_POSTED_SCHEMA`): ``{"input_plan", "start_markers",
        "forcing_leads", "seal_authored_proof_keys", "manifest_path",
        "lead_role_prefix", "derived_roles",
        "manifest_bound_identity_keys"}``.  ``input_plan`` is
        :func:`input_plan` of the manifest the seal will write;
        ``start_markers`` maps each lead the head's own decode read (the
        start lead, and on a backlog the leads posted with it) to its posted
        marker, whose digest the head binds. A tree relaying a root head
        may instead pass ``start_marker_sha256``, its already bound lead
        digest map; the seal holds those digests to the complete markers.
        The two forms are mutually exclusive. ``forcing_leads`` is the lead
        of each forcing time in order, so interval k's segment marker binds
        the markers of leads ``forcing_leads[k]`` and ``[k + 1]``
        (:meth:`bind_posted_leads`); and ``seal_authored_proof_keys`` are
        the proof keys that read every lead, which the head proof leaves
        out and the seal adds.  ``input_manifest_sha256`` is then ``None``:
        the head binds the plan's digest instead.
        """

        physical_block = None
        if ensemble_physical is not None:
            from woof.ensemble import physical_boundary
            physical_block = physical_boundary.make_head(dict(ensemble_physical), lbc["schedule"])
            initial = physical_block["initial_receipt"]
            source = self._cache.identity.get("source_identity", {})
            if source.get("ensemble_posted_physical_input") != initial["binding"]:
                raise ValueError("prepared identity does not bind its initial physical member input")
            metadata = dict(metadata or {})
            if physical_boundary.KEY in metadata or physical_boundary.KEY in seal_completes:
                raise ValueError("physical boundary metadata belongs to its checked native writer")
            metadata[physical_boundary.KEY] = physical_boundary.head_catalog(physical_block)
            seal_completes = (*seal_completes, physical_boundary.KEY)
            self._physical_seen = {0: initial}
        elif self._physical_receipts is not None:
            raise ValueError("native physical receipts require their provider head binding")
        stray = sorted(set(proof_head) & SEAL_ONLY_PROOF_KEYS)
        if stray:
            raise ValueError(f"the head proof carries seal-only keys {stray}")
        posted_block = None
        if as_posted is not None:
            if input_manifest_sha256 is not None:
                raise ValueError(
                    "an as-posted head binds its input plan, never an input "
                    "manifest digest, which does not exist before the last "
                    "lead")
            keys = sorted(str(key) for key in
                          as_posted["seal_authored_proof_keys"])
            stray = sorted(set(proof_head) & set(keys))
            if stray:
                raise ValueError(
                    f"the as-posted head proof carries seal-authored keys "
                    f"{stray}")
            plan = json.loads(_canonical(dict(as_posted["input_plan"])))
            forcing_leads = [int(lead) for lead in as_posted["forcing_leads"]]
            if lbc is not None and len(forcing_leads) \
                    != len(lbc["schedule"]) + 1:
                raise ValueError(
                    f"an as-posted head names {len(forcing_leads)} forcing "
                    f"leads for {len(lbc['schedule'])} boundary intervals; "
                    "each interval's segment binds the leads of its two "
                    "times")
            if "start_marker_sha256" in as_posted:
                if "start_markers" in as_posted:
                    raise ValueError(
                        "an as-posted head takes start markers or their "
                        "bound digests, never both")
                start_digests = {}
                for lead, digest in dict(as_posted[
                        "start_marker_sha256"]).items():
                    if isinstance(lead, bool) or not (
                            isinstance(lead, int) and lead >= 0
                            or isinstance(lead, str) and lead.isdecimal()):
                        raise ValueError(
                            "an as-posted head's start-marker digest names "
                            f"a nonnegative integer lead, not {lead!r}")
                    key = str(int(lead))
                    if key in start_digests or not _is_sha256(digest):
                        raise ValueError(
                            "an as-posted head binds each start lead once "
                            f"to its marker's SHA-256 digest, not {lead!r}: "
                            f"{digest!r}")
                    start_digests[key] = str(digest)
            else:
                start_digests = {
                    str(int(lead)): posted_lead_marker_sha256(marker)
                    for lead, marker
                    in sorted(dict(as_posted["start_markers"]).items())}
            posted_block = {
                "schema": AS_POSTED_SCHEMA,
                "input_plan": plan,
                "input_plan_sha256": input_plan_sha256(plan),
                "start_marker_sha256": start_digests,
                "forcing_leads": forcing_leads,
                "seal_authored_proof_keys": keys,
                "manifest_path": str(as_posted["manifest_path"]),
                "lead_role_prefix": str(as_posted["lead_role_prefix"]),
                "derived_roles": sorted(
                    str(role) for role in as_posted.get("derived_roles", ())),
                "manifest_bound_identity_keys": sorted(
                    str(key) for key in as_posted.get(
                        "manifest_bound_identity_keys",
                        ("input_manifest_sha256",))),
            }
            # Named only by a route whose manifest is a composition-inputs
            # document (the mapped engine), so a GFS head's block, and so
            # its digest, is what it was.
            if as_posted.get("fixed_rows") is not None:
                posted_block["fixed_rows"] = sorted(
                    str(path) for path in as_posted["fixed_rows"])
            if as_posted.get("proof_manifest_key") is not None:
                posted_block["proof_manifest_key"] = str(
                    as_posted["proof_manifest_key"])
            # Named only by a route whose identity carries other documents'
            # digests (native HRRR), so every other head's block, and its
            # digest, is what it was (DOCUMENT_ROW_RULES).
            if as_posted.get("document_bound_identity_keys"):
                posted_block["document_bound_identity_keys"] = {
                    str(key): str(role) for key, role in dict(
                        as_posted["document_bound_identity_keys"]).items()}
                documents = {}
                for role, document in dict(as_posted["documents"]).items():
                    rule = str(document["lead_rows"])
                    if rule not in DOCUMENT_ROW_RULES:
                        raise ValueError(
                            f"an as-posted document's row rule is one of "
                            f"{DOCUMENT_ROW_RULES}, not {rule!r}")
                    documents[str(role)] = {
                        "path": str(document["path"]), "lead_rows": rule,
                        "fixed_rows": sorted(str(name) for name in
                                             document.get("fixed_rows") or ())}
                unknown = sorted(set(posted_block[
                    "document_bound_identity_keys"].values()) - set(documents))
                if unknown:
                    raise ValueError(
                        f"an as-posted head binds identity keys to documents "
                        f"{unknown} it does not say how to hold")
                posted_block["documents"] = documents
                posted_block["decoded_rows"] = any(
                    document["lead_rows"] == "decoded_leads"
                    for document in documents.values())
            if as_posted.get("posted_user_metadata"):
                keys = sorted(str(key) for key in
                              as_posted["posted_user_metadata"])
                placeholder = as_posted_placeholder(
                    posted_block["input_plan_sha256"])
                user = dict(metadata or {})
                wrong = [key for key in keys if user.get(key) != placeholder]
                if wrong:
                    raise ValueError(
                        f"an as-posted head's user metadata {wrong} must hold "
                        "the plan's placeholder, which its seal replaces with "
                        "the digest the sealed identity carries")
                posted_block["posted_user_metadata"] = keys
        # Named only when a route completes metadata at its seal, so every
        # other route calls its cache stream exactly as before.
        completes = ({"seal_completes": tuple(seal_completes)}
                     if seal_completes else {})
        cache_head = self._cache.write_head(
            initial_result=initial_result, met=met, surface=surface,
            metadata=metadata, lbc=lbc, **completes)
        cache_head["directory"] = self.cache_name
        if self.chained:
            # Host RAM, on every backend: the forecast process holds the
            # head's arrays, every boundary interval it loads and its own
            # working set while the producer keeps building beside it.
            forecast = forecast_host_bytes(
                head_payload_bytes=(int(cache_head["payload_bytes"])
                                    + int(extra_head_payload_bytes)),
                interval_host_bytes=(None if forcing is None
                                     else forcing.interval_host_bytes),
                intervals=len(lbc["schedule"]))
            store_bytes = int(self._decision.get("forecast_store_host_bytes", 0))
            if store_bytes:
                forecast["store_and_staging_bytes"] = store_bytes
                if forecast["total_bytes"] is not None:
                    forecast["total_bytes"] += store_bytes
            host = {**host_admission(
                forecast_bytes=forecast["total_bytes"],
                producer_peak_bytes=self._decision.get("producer_host_peak_bytes"),
                producer_peak_required=self._decision.get("producer_host_peak_required", False),
                producer_decode_bytes=self._decision.get("producer_decode_host_bytes", 0)),
                    "forecast_host_parts": forecast}
            if not host["admitted"]:
                self.decline_chaining(host["reason"])
            self._decision = {**self._decision, "chained": self.chained,
                              "host": host}
        basis = {
            "schema": HEAD_SCHEMA,
            "cache": cache_head,
            "proof_head": json.loads(_canonical(proof_head)),
            "input_manifest_sha256": input_manifest_sha256,
        }
        # Added only for a tree, so a single domain's head basis (and so
        # its digest) is what it was before trees chained.
        if tree is not None:
            basis["tree"] = json.loads(_canonical(dict(tree)))
        # Named only when the seal writes another document than proof.json,
        # so every other head's basis (and digest) is unchanged.
        if self.proof_name != PROOF_NAME:
            basis["proof_name"] = self.proof_name
        # Added only as posted, by the same rule.
        if posted_block is not None:
            basis["as_posted"] = posted_block
        if physical_block is not None:
            basis["ensemble_physical"] = physical_block
        head = {
            "schema": HEAD_SCHEMA,
            "basis": basis,
            "created_utc": _utc_now(),
            "decision": dict(self._decision),
            "reservation": dict(reservation or {}),
        }
        if tree is not None:
            # Mirrored at the top level for readers; basis.tree is the copy
            # the digest binds.
            head.update({key: basis["tree"][key] for key in (
                "layout", "domains", "children_artifacts")})
        from woof.ingest.stream_resume import preserve_child_receipts
        preserve_child_receipts(self.root, head)
        head["head_sha256"] = head_sha256(head)
        _write_json_atomic(self.stream_path / HEAD_NAME, head)
        if as_posted is not None:
            from woof.ingest.stream_resume import POSTED_PREFIX_DIRNAME
            for lead, marker in dict(as_posted.get("start_markers") or {}).items():
                _write_json_atomic(self.stream_path / POSTED_PREFIX_DIRNAME
                                   / posted_lead_marker_name(int(lead)), dict(marker))
        (self.stream_path / SEGMENTS_DIRNAME).mkdir(parents=True, exist_ok=True)
        self.head = head
        self.head_sha256 = head["head_sha256"]
        self.head_seconds = time.perf_counter() - self._started
        if self.chained:
            # From here on a forecast may bind this head, so the producer
            # must outlive the process reading its console.
            keep_console_after_reader_exit()
            # The first heartbeat is written before the rename, so no
            # reader ever sees a published head without a producer.
            heartbeat = _Heartbeat(self.stream_path, self)
            self._publish_tree(self.staging, self.output_root)
            self.published = True
            self._record_source_head_publication()
            self._cache.move(self.cache_path)
            heartbeat.path = self.stream_path
            self._heartbeat = heartbeat
            self._heartbeat.start()
            _say(f"prepare: head published at {self.output_root}; the "
                 "forecast may start while the remaining boundary "
                 "intervals are built")
        return self.head_sha256

    def _record_source_head_publication(self) -> None:
        # Observe the existing atomic publication; no stream or array changes.
        from woof.mapped_composition import _SHARED_COMPOSITION
        binding = _SHARED_COMPOSITION.get()
        if binding is not None:
            from woof.timeline.ledger import emit
            emit("prep.head.publish", attrs={"source_key": binding[0],
                "head_sha256": self.head_sha256, "path": str(self.output_root)}, product="ensemble")

    def note_build_seconds(self, seconds: float) -> None:
        """Record one forcing time's build wall (sizes the silence limit)."""

        self._build_seconds.append(float(seconds))

    @property
    def times_built(self) -> int:
        return len(self._build_seconds)

    @property
    def slowest_build_seconds(self) -> float:
        return max(self._build_seconds, default=0.0)

    def check_stop(self) -> None:
        """Raise when the consumer asked this producer to stop."""

        stop = _read_json(self.stream_path / STOP_NAME)
        if isinstance(stop, dict):
            raise BoundaryStreamStopped(
                "the forecast stopped this preparation: "
                f"{stop.get('reason', 'no reason given')}")

    #: An as-posted preparation's posted-lead markers by lead, as it reads
    #: them (:meth:`bind_posted_leads`); ``None`` for any other head.
    _posted_markers: Mapping[int, Mapping[str, object]] | None = None

    def bind_physical_receipts(self, receipts: Mapping[int, Mapping[str, object]]) -> None:
        """Observe the full immutable binding of every native forcing knot."""
        if self.head is not None:
            raise RuntimeError("physical receipt ownership must be bound before the prepared head")
        if not isinstance(receipts, Mapping):
            raise TypeError("physical receipts must be indexed by native forcing knot")
        self._physical_receipts = receipts

    def bind_posted_leads(self, markers: Mapping[int, Mapping[str, object]]
                          ) -> None:
        """Hand an as-posted head's writer the lead markers its route reads.

        ``markers`` is the route's own mapping from a lead to its posted
        marker, filled as each lead is waited for; each segment then binds
        the digests of the markers of the two times it spans (DESIGN A136
        2.4 item 4), and the seal holds every manifest row to them.
        """

        self._posted_markers = markers

    #: An as-posted head's decoded-file records by lead
    #: (:meth:`bind_decoded_leads`); ``None`` for any other head.
    _decoded_records: Mapping[int, Mapping[str, str]] | None = None

    def bind_decoded_leads(self, records: Mapping[int, Mapping[str, str]]
                           ) -> None:
        """Hand the writer each lead's decoded-file record as the route reads it.

        ``records`` maps a lead to ``{row name: sha256}`` of the files its
        decode wrote, filled when the lead is taken; each segment then binds
        the records of the two leads it spans beside their posted markers,
        and the seal holds the decoded document's rows to them
        (``decoded_leads``, :data:`DOCUMENT_ROW_RULES`).
        """

        self._decoded_records = records

    def _segment_decoded(self, index: int) -> dict | None:
        posted = (self.head or {}).get("basis", {}).get("as_posted")
        if posted is None or not posted.get("decoded_rows"):
            return None
        bound = {}
        for lead in posted["forcing_leads"][int(index):int(index) + 2]:
            record = (self._decoded_records or {}).get(int(lead))
            if record is None:
                raise RuntimeError(
                    f"interval {index} spans lead {lead}, whose decoded files "
                    "this preparation has not recorded; an as-posted segment "
                    "binds the decoded record of each lead it was built from")
            bound[str(int(lead))] = decoded_lead_record_sha256(record)
        return bound

    def _segment_leads(self, index: int) -> dict | None:
        posted = (self.head or {}).get("basis", {}).get("as_posted")
        if posted is None:
            return None
        leads = posted["forcing_leads"][int(index):int(index) + 2]
        bound = {}
        for lead in leads:
            marker = (self._posted_markers or {}).get(int(lead))
            if marker is None:
                raise RuntimeError(
                    f"interval {index} spans lead {lead}, whose posted "
                    "marker this preparation has not read; an as-posted "
                    "segment binds the markers of the leads it was built "
                    "from")
            bound[str(int(lead))] = posted_lead_marker_sha256(marker)
        return bound

    def write_segment(self, index: int, interval, *,
                      relay_marker: Mapping[str, object] | None = None) -> dict:
        """Write interval ``index``'s arrays, then its ready marker.

        ``relay_marker`` is a root interval's marker already checked by
        :class:`StreamedIntervals`. A tree carries its posted and decoded
        lead digests into its own marker before the root's seal supplies
        the complete records. The caller holds the root seal to those
        consumed markers; this tree's seal holds the relayed digests to
        the records it copies from that seal.
        """

        if self.head is None:
            raise RuntimeError("a segment needs its head first")
        self.check_stop()
        if relay_marker is None:
            leads = self._segment_leads(index)
            decoded = self._segment_decoded(index)
        else:
            if (relay_marker.get("schema") != SEGMENT_SCHEMA
                    or relay_marker.get("index") != int(index)
                    or relay_marker.get("start_seconds")
                    != float(interval.start_seconds)
                    or relay_marker.get("end_seconds")
                    != float(interval.end_seconds)
                    or sorted(relay_marker.get("fields") or ())
                    != sorted(interval.fields)):
                raise BoundaryStreamError(
                    f"relayed segment {index} is not the root interval this "
                    "tree read, so its source records cannot bind it")
            posted = self.head["basis"].get("as_posted")
            expected = (set() if posted is None else {
                str(int(lead)) for lead in posted["forcing_leads"][
                    int(index):int(index) + 2]})
            records = {}
            for name, needed in (
                    ("posted_leads", posted is not None),
                    ("decoded_leads", bool(posted and
                                           posted.get("decoded_rows")))):
                value = relay_marker.get(name)
                if needed:
                    if (not isinstance(value, Mapping)
                            or set(value) != expected
                            or any(not _is_sha256(digest)
                                   for digest in value.values())):
                        raise BoundaryStreamError(
                            f"relayed segment {index}'s {name} does not bind "
                            f"exactly leads {sorted(expected, key=int)} to "
                            "their record digests, so it cannot name the "
                            "inputs this interval read")
                    records[name] = dict(value)
                elif value is not None:
                    raise BoundaryStreamError(
                        f"relayed segment {index} carries {name} this tree's "
                        "head did not bind")
            leads = records.get("posted_leads")
            decoded = records.get("decoded_leads")
        physical = self.head["basis"].get("ensemble_physical")
        physical_records = None
        if physical is not None:
            from woof.ensemble import physical_boundary
            if relay_marker is None or self._physical_receipts is not None:
                physical_records = physical_boundary.segment_records(
                    physical, int(index), self._physical_receipts or {}, self._physical_seen)
                if relay_marker is not None and "ensemble_physical" in relay_marker:
                    relayed = physical_boundary.validate_segment(
                        physical, int(index), relay_marker["ensemble_physical"], self._physical_seen)
                    if relayed != physical_records:
                        raise BoundaryStreamError("relayed physical endpoints differ from this member's native inputs")
            else:
                physical_records = physical_boundary.validate_segment(
                    physical, int(index), relay_marker.get("ensemble_physical"), self._physical_seen)
        elif relay_marker is not None and "ensemble_physical" in relay_marker:
            raise BoundaryStreamError("relayed physical boundary inputs have no matching provider head")
        segment = self._cache.write_segment(int(index), interval)
        marker = {
            "schema": SEGMENT_SCHEMA,
            "head_sha256": self.head_sha256,
            **segment,
        }
        if leads is not None:
            # Only as posted, so every other segment marker is unchanged.
            marker["posted_leads"] = leads
        if decoded is not None:
            # Only for a head that holds a decoded document's rows.
            marker["decoded_leads"] = decoded
        if physical_records is not None:
            marker["ensemble_physical"] = physical_records
        member_source = getattr(self, "_posted_member_source", None)
        if member_source is not None:
            from woof.ensemble.posted_preparation import (
                posted_binding_from_identity, validate_posted_member_segment)
            bound = member_source.bind_segment(int(index))
            if relay_marker is not None and relay_marker.get("ensemble_member_input") != bound:
                raise BoundaryStreamError("relayed member segment differs from the consumed native member inputs")
            validate_posted_member_segment(bound,
                posted_binding_from_identity(self._cache.identity), index=int(index))
            marker["ensemble_member_input"] = bound
        if leads is not None and self._posted_markers is not None:
            from woof.ingest.stream_resume import POSTED_PREFIX_DIRNAME
            for lead in leads:
                posted_marker = self._posted_markers.get(int(lead))
                if posted_marker is not None:
                    _write_json_atomic(self.stream_path / POSTED_PREFIX_DIRNAME
                                       / posted_lead_marker_name(int(lead)), dict(posted_marker))
        _write_json_atomic(segment_marker_path(self.root, index), marker)
        self._segments_written += 1
        return marker

    def stream_forcing_times(self, *, count: int,
                             build_forcing_time: Callable[[int], tuple],
                             forcing, times: Sequence,
                             release: Callable[[], None] | None = None,
                             workers: int = 1,
                             ) -> None:
        """Build forcing times 1..count-1 and write each interval as it closes.

        The one loop every route that builds boundaries from forcing times
        runs after its head.  ``build_forcing_time(index)`` returns the
        route's ``(met, initialized)`` for that time, built exactly as the
        route builds it; ``forcing`` is the route's
        :class:`woof.ingest.lateral_bc.StateBoundaryFrames`, which already
        holds the start time.  Interval k is written as soon as time k+1
        exists, then time k is released, so at most two forcing times are
        held and the files are numbered in the order the one-shot writer
        numbered them.  ``release`` hands the build's device or pool memory
        back after each time.  A domain tree's seal reads the root's whole
        boundary set back from the sealed cache (:class:`TreeStartStates`),
        so no interval is held here after it is written.

        ``workers`` greater than 1 builds that many forcing times at once on
        threads (each with a copy of the caller's context), while the
        intervals are still added, written and released here in index order.
        Each time's build is the same call on the same inputs, so the
        written bytes are those of the serial loop; at most ``workers`` + 1
        times are held.  Named breakage: a GEFS member preparation on a
        rented RTX PRO 6000 spent 42 s of its 70 s here on one core, nine
        independent forcing times in a row, with the card at 39 % and 17 of
        30 cores idle (KCCX 2026-10-16).
        """

        from woof.progress import prep_progress

        workers = max(1, min(int(workers), int(count) - 1))
        if workers > 1:
            self._stream_forcing_times_concurrently(
                count=int(count), build_forcing_time=build_forcing_time,
                forcing=forcing, times=times, release=release,
                workers=workers)
            return
        for index in range(1, int(count)):
            built = time.perf_counter()
            met, initialized = build_forcing_time(index)
            forcing.add_state(initialized.state, index=index)
            del met, initialized
            if release is not None:
                release()
            self.note_build_seconds(time.perf_counter() - built)
            interval = forcing.interval(index - 1, times)
            self.write_segment(index - 1, interval)
            del interval
            forcing.release(index - 1)
            # Said per interval: this loop is most of a preparation's wall on
            # a long run, and a chained forecast is already stepping beside
            # it, so "boundary times 3 of 16 ready" is what a watcher needs.
            prep_progress("root_boundaries", label="Boundary times",
                          done=index, count=int(count) - 1)

    def _stream_forcing_times_concurrently(self, *, count, build_forcing_time,
                                           forcing, times, release, workers):
        """The serial loop's adds, writes and releases, in its order; builds in parallel."""

        from concurrent.futures import ThreadPoolExecutor
        from woof.progress import prep_progress

        def timed(index):
            started = time.perf_counter()
            built = build_forcing_time(index)
            return built, time.perf_counter() - started

        pending = {}
        following = 1
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="forcing-time") as pool:
            def submit_next():
                nonlocal following
                if following < count:
                    # A fresh copy per task: the preparation's math scope,
                    # setup owners and posted context are context variables.
                    context = contextvars.copy_context()
                    context.run(_CONCURRENT_FORCING_BUILD.set, True)
                    pending[following] = pool.submit(context.run, timed, following)
                    following += 1

            try:
                while len(pending) < workers and following < count:
                    submit_next()
                for index in range(1, count):
                    (met, initialized), seconds = pending.pop(index).result()
                    submit_next()
                    forcing.add_state(initialized.state, index=index)
                    del met, initialized
                    if release is not None:
                        release()
                    self.note_build_seconds(seconds)
                    interval = forcing.interval(index - 1, times)
                    self.write_segment(index - 1, interval)
                    del interval
                    forcing.release(index - 1)
                    prep_progress("root_boundaries", label="Boundary times",
                                  done=index, count=count - 1)
            except BaseException:
                for future in pending.values():
                    future.cancel()
                raise

    def write_intervals(self, intervals) -> None:
        """Write intervals a route already holds: a source already complete."""

        for index, interval in enumerate(intervals):
            self.write_segment(index, interval)

    def seal_cache(self, *, identity=None,
                   manifest_sha256: str | None = None,
                   completed_metadata: Mapping[str, object] | None
                   = None,
                   document_sha256: Mapping[str, str] | None = None,
                   physical_provider_seal: Mapping[str, object] | None = None) -> dict:
        """Write ``header.json``; return the one-shot writer's receipt.

        ``identity`` is the one-shot identity an as-posted head's seal
        writes and ``manifest_sha256`` the digest of the input manifest
        that seal wrote (see :func:`check_as_posted_identity`); a head that
        is not as posted seals the identity it was written under.
        ``completed_metadata`` completes the user metadata the head named
        in ``seal_completes`` (checked by :func:`verify_seal`).
        ``document_sha256`` is ``{identity key: digest}`` for the head's
        document-bound keys (:func:`document_bound_digests`).
        """

        # Named only when a route completes metadata at its seal, so every
        # other head seals with the call it always made.
        physical = (self.head or {}).get("basis", {}).get("ensemble_physical")
        if physical is not None:
            from woof.ensemble import physical_boundary
            if physical_provider_seal is None:
                raise ValueError("physical member inputs must verify every source seal before cache completion")
            completed_metadata = dict(completed_metadata or {})
            if physical_boundary.KEY in completed_metadata:
                raise ValueError("physical boundary completion belongs to its checked native writer")
            completed_metadata[physical_boundary.KEY] = physical_boundary.complete_catalog(
                physical, self._physical_seen, physical_provider_seal)
        elif physical_provider_seal is not None:
            raise ValueError("a physical provider seal has no matching prepared head")
        member_source = getattr(self, "_posted_member_source", None)
        if member_source is not None:
            from woof.ensemble.posted_preparation import bind_current_prepared_identity
            member_source.seal()
            if identity is not None:
                identity = bind_current_prepared_identity(identity)
        completes = ({} if completed_metadata is None
                     else {"completed_metadata": completed_metadata})
        if identity is None:
            return self._cache.seal(**completes)
        posted = (self.head or {}).get("basis", {}).get("as_posted")
        if posted is None:
            raise RuntimeError(
                "only an as-posted head's seal writes another identity")
        if set(document_sha256 or {}) != set(
                posted.get("document_bound_identity_keys") or {}):
            raise RuntimeError(
                "an as-posted seal names the digest of every document its "
                "head bound an identity key to, and only those")
        check_as_posted_identity(
            self.head["basis"]["cache"]["identity"], identity,
            plan_sha256=posted["input_plan_sha256"],
            manifest_sha256=manifest_sha256,
            manifest_bound=posted["manifest_bound_identity_keys"],
            document_bound=document_sha256)
        keys = posted.get("posted_user_metadata") or ()
        if keys:
            # Each is the digest the sealed identity carries under the
            # same name, so the header's metadata and identity agree as a
            # one-shot preparation's do.
            completes["posted_user_metadata"] = {
                key: posted_identity_leaf(identity, key) for key in keys}
        return self._cache.seal(identity=identity, **completes)

    def write_posted_leads(self, markers: Mapping[int, Mapping[str, object]],
                           *, route_table_sha256: str,
                           decoded: Mapping[int, Mapping[str, str]] | None
                           = None) -> dict:
        """Record every lead this preparation consumed, as its marker named it.

        ``markers`` maps a lead to its posted marker
        (:class:`PostedLeads`), kept whole so a verifier recomputes the
        digest the head and the segments bound; ``route_table_sha256`` is
        the route table the fetch's schedule names at the seal.  Written
        beside the head, so a verifier holds the sealed manifest to these
        rows without the fetch folder.  ``decoded`` (a head with a
        ``decoded_leads`` document) is each lead's decoded-file record,
        kept whole beside its marker for the same reason.
        """

        record = {
            "schema": POSTED_LEADS_SCHEMA,
            "head_sha256": self.head_sha256,
            "route_table_sha256": str(route_table_sha256),
            "leads": {
                str(int(lead)): {
                    "marker_sha256": posted_lead_marker_sha256(marker),
                    "marker": json.loads(_canonical(dict(marker))),
                }
                for lead, marker in sorted(dict(markers).items())
            },
        }
        for lead, value in sorted(dict(decoded or {}).items()):
            row = record["leads"].get(str(int(lead)))
            if row is None:
                raise RuntimeError(
                    f"lead {lead} has a decoded record and no posted marker")
            row["decoded"] = {str(name): str(digest)
                              for name, digest in sorted(dict(value).items())}
        _write_json_atomic(self.stream_path / POSTED_LEADS_NAME, record)
        return record

    def publish(self, proof: dict) -> dict:
        """Write ``proof.json`` last (and the tree's rename when unchained)."""

        from woof.ingest.stream_resume import preserve_seal_metadata
        proof = preserve_seal_metadata(self.root, proof)
        if self.head is None:
            raise RuntimeError("the seal needs its head first")
        sealed_keys = _seal_keys(self.head)
        stray = {key: value for key, value in proof.items()
                 if key not in sealed_keys}
        if json.loads(_canonical(stray)) != self.head["basis"]["proof_head"]:
            differing = sorted(
                key for key in set(stray) | set(self.head["basis"]["proof_head"])
                if json.loads(_canonical(stray.get(key)))
                != self.head["basis"]["proof_head"].get(key))
            raise RuntimeError(
                "the sealed proof differs from the head it was published "
                f"under in {differing}")
        _write_json_atomic(self.root / self.proof_name, proof)
        if not self.published:
            self._publish_tree(self.staging, self.output_root)
            self.published = True
            self._record_source_head_publication()
        self._stop_heartbeat(final="sealed")
        return proof

    def boundary_stream_proof(self) -> dict:
        """The one proof field the stream adds: the head it sealed."""

        return {"head_sha256": self.head_sha256}

    def fail(self, error: BaseException) -> None:
        """Record the producer's failure so a waiting forecast can end."""

        try:
            self._record_failure(error)
        finally:
            self._stop_heartbeat(
                final=("stopped" if isinstance(error, BoundaryStreamStopped)
                       else "failed"))

    def _record_failure(self, error: BaseException) -> None:
        if self.published:
            reason = (str(error) or type(error).__name__)
            stage = ("stopped" if isinstance(error, BoundaryStreamStopped)
                     else "producing")
            record = {
                "reason": reason,
                "stage": stage,
                "error_type": type(error).__name__,
                "traceback_tail": "".join(traceback.format_exception(
                    type(error), error, error.__traceback__))[-4000:],
                "failed_utc": _utc_now(),
            }
            if isinstance(error, SourceBehind):
                # Named, so the waiting forecast ends with exit 75 and the
                # lead rather than as a failed preparation.
                record["code"] = SOURCE_BEHIND_CODE
                record["details"] = dict(error.details)
            try:
                _write_json_atomic(self.stream_path / FAILED_NAME, record)
            except (OSError, TypeError, ValueError):
                pass

    def _stop_heartbeat(self, *, final: str) -> None:
        if self._heartbeat is not None:
            self._heartbeat.stop(final=final)
            self._heartbeat = None


class _Heartbeat(threading.Thread):
    def __init__(self, path: Path, writer: PreparedTreeWriter):
        super().__init__(name="boundary-stream-heartbeat", daemon=True)
        self.path = Path(path)
        self.writer = writer
        self._done = threading.Event()
        self.beat()

    def beat(self, state: str | None = None) -> None:
        state = self.writer._state if state is None else state
        record = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "state": state,
            "times_built": self.writer.times_built,
            "segments_written": self.writer._segments_written,
            "slowest_build_seconds": self.writer.slowest_build_seconds,
            "updated_utc": _utc_now(),
            "updated_epoch": time.time(),
        }
        if state == "waiting_for_source" and self.writer._waiting_for:
            record["waiting_for"] = dict(self.writer._waiting_for)
        if self.writer._arrived is not None:
            record["arrived"] = dict(self.writer._arrived)
        try:
            _write_json_atomic(self.path / PRODUCER_NAME, record)
        except OSError:
            pass

    def run(self) -> None:
        while not self._done.wait(HEARTBEAT_SECONDS):
            self.beat()

    def stop(self, *, final: str) -> None:
        self._done.set()
        self.beat(final)


#: The as-posted fetch's folder and files (``<fetch out>/posting/``), the
#: names :mod:`woof.source_posting` declares.  Copied, not imported: the
#: preparation package stages this module without the fetch's tables, and
#: a test holds the two copies equal.
POSTING_DIRNAME = "posting"
POSTING_SCHEDULE_NAME = "schedule.json"
POSTING_FAILED_NAME = "failed.json"
#: The schema of one verified lead's marker (``posting/fNNN.json``), as
#: :data:`woof.fetch_as_posted.MARKER_SCHEMA` writes it.
POSTED_LEAD_SCHEMA = "gpuwm.posted-lead.v1"
#: How often a preparation waiting on a lead looks for its marker.  A
#: marker is a local file the fetch writes, so this costs a stat, never a
#: request to the source's host.
POSTED_LEAD_POLL_SECONDS = 1.0


def posted_lead_marker_name(lead: int) -> str:
    return f"f{int(lead):03d}.json"


class PostedLeads:
    """A preparation's wait for the leads an as-posted fetch publishes.

    ``folder`` is the fetch's ``posting/`` folder, ``source`` and
    ``cycle`` (``YYYY-MM-DDTHH``) the window it fetches.  :meth:`wait`
    blocks until lead N's marker (``fNNN.json``, written after the lead's
    objects are fetched and verified) is there and returns it.  While it
    waits, the producer heartbeat says ``waiting_for_source`` with the
    lead's scheduled and late times from ``schedule.json``
    (:meth:`PreparedTreeWriter.waiting_for_source`), so a forecast at a
    seam says its wait is on the source.

    The preparation never asks a host itself: whether a lead is posted,
    late or unheard is the fetch loop's answer, and only
    ``posting/failed.json`` ends a wait.  So a host that answers "not
    there" in any form (a 404, or the 403 NOMADS gives for a path under a
    cycle directory it has not created yet) is a lead not posted yet,
    never a failed preparation.  ``failed.json`` with ``code:
    source_behind`` raises :class:`SourceBehind` (exit 75, naming the
    lead); any other failure, or a fetch that ended (``fetch_alive``
    answers False) without the lead and without a failure record, raises
    :class:`BoundaryProducerFailed` naming the lead, so the forecast does
    not wait for a marker nothing will write.

    Modelled on the wait of finalprep/producer's scheduled inputs (a
    wait per lead, a heartbeat naming the lead, a late lead failed by
    name), without its decode store: the markers are the fetch loop's own.
    """

    def __init__(self, folder, *, source: str, cycle: str,
                 writer: "PreparedTreeWriter | None" = None,
                 fetch_alive: Callable[[], bool] | None = None,
                 poll_seconds: float = POSTED_LEAD_POLL_SECONDS,
                 sleep: Callable[[float], None] = time.sleep):
        self.folder = Path(folder)
        self.source = str(source)
        self.cycle = str(cycle)
        self.writer = writer
        self.fetch_alive = fetch_alive
        self.poll_seconds = float(poll_seconds)
        self.sleep = sleep
        #: Every wait this preparation made: ``{lead, seconds}`` in order
        #: (the ``posting.waits`` a proof records).
        self.waits: list[dict[str, object]] = []

    # -- reading ---------------------------------------------------------

    def marker(self, lead: int) -> dict | None:
        """Lead ``lead``'s verified marker, or ``None`` if it is not there.

        A marker from another window (a folder reused across cycles) is
        refused by name rather than read as this one's lead.
        """

        path = self.folder / posted_lead_marker_name(lead)
        record = _read_json(path)
        if not isinstance(record, dict):
            return None
        if record.get("schema") != POSTED_LEAD_SCHEMA:
            raise BoundaryProducerFailed(
                f"{path} is not a posted-lead marker (schema "
                f"{record.get('schema')!r})")
        if (record.get("source") != self.source
                or record.get("cycle") != self.cycle
                or record.get("lead") != int(lead)):
            raise BoundaryProducerFailed(
                f"{path} marks {record.get('source')} "
                f"{_lead_words(record.get('lead'))} of the "
                f"{record.get('cycle')} cycle, not {self.source} "
                f"{_lead_words(lead)} of the {self.cycle} cycle this "
                "preparation waits for")
        return record

    def posted(self, leads: Sequence[int]) -> list[int]:
        """The leads of ``leads`` whose markers are there now, in order."""

        return [int(lead) for lead in leads
                if (self.folder / posted_lead_marker_name(lead)).is_file()]

    def _schedule_row(self, lead: int) -> dict:
        # Replaced by the fetch each time a lead moves: read through the
        # replace, so the heartbeat names the lead's scheduled and late
        # times rather than none.
        try:
            schedule = read_replaced_json(self.folder / POSTING_SCHEDULE_NAME)
        except (OSError, ValueError):
            schedule = None
        rows = schedule.get("leads") if isinstance(schedule, dict) else None
        for row in rows or ():
            if isinstance(row, dict) and row.get("lead") == int(lead):
                return row
        return {}

    def route_table_sha256(self) -> str:
        """The route table the fetch planned this window's leads under.

        Read from ``schedule.json`` (``table_sha256``), which the fetch
        writes before any lead; an as-posted plan binds it (DESIGN A136 2.4
        item 5).  The fetch replaces the file as leads move, so it is read
        through a replace (:func:`read_replaced_json`).
        """

        path = self.folder / POSTING_SCHEDULE_NAME
        try:
            schedule = read_replaced_json(path)
        except (OSError, ValueError) as error:
            raise BoundaryProducerFailed(
                f"{path} is not readable ({error}); the as-posted fetch "
                "writes it before any lead, and an as-posted plan binds the "
                "route table it names") from None
        digest = (schedule.get("table_sha256")
                  if isinstance(schedule, dict) else None)
        if not _is_sha256(digest):
            raise BoundaryProducerFailed(
                f"{path} names no route table (table_sha256), which the "
                "as-posted fetch writes before any lead; an as-posted plan "
                "binds the table its leads were fetched under")
        return str(digest)

    def _failure(self, lead: int) -> BoundaryProducerFailed | None:
        record = _read_json(self.folder / POSTING_FAILED_NAME)
        if not isinstance(record, dict):
            return None
        if record.get("code") == SOURCE_BEHIND_CODE:
            return SourceBehind(record)
        return BoundaryProducerFailed(
            f"the fetch of {self.source} {self.cycle} failed before "
            f"{_lead_words(lead)} was fetched: "
            f"{record.get('message') or record.get('reason') or record}")

    # -- waiting ---------------------------------------------------------

    def _waiting_for(self, lead: int, row: Mapping[str, object]) -> dict:
        """The heartbeat's record of a wait on ``lead`` (:data:`WAITING_FOR_FIELDS`)."""

        return {"source": self.source, "cycle": self.cycle, "lead": lead,
                "valid_time": row.get("valid_time"),
                "expected_at": row.get("expected_at"),
                "late_at": row.get("late_at"), "since_utc": None,
                "state": row.get("state"),
                "first_seen_at": row.get("first_seen_at"),
                "last_answer": row.get("last_answer")}

    def wait(self, lead: int, *, stop: threading.Event | None = None) -> dict:
        """Block until lead ``lead`` is fetched and verified; its marker.

        ``stop`` is the owner's own end to the wait: once it is set the
        wait raises :class:`PostedWaitStopped` before its next look at the
        posting folder, so a waiter on a thread of its own ends with the
        preparation that started it.
        """

        lead = int(lead)
        started = time.monotonic()
        waiting = False
        record = None
        try:
            while True:
                if stop is not None and stop.is_set():
                    raise PostedWaitStopped(
                        f"the wait for {self.source} {_lead_words(lead)} of "
                        f"the {self.cycle} cycle was stopped by its owner")
                record = self.marker(lead)
                if record is not None:
                    break
                failure = self._failure(lead)
                if failure is not None:
                    raise failure
                if self.fetch_alive is not None and not self.fetch_alive():
                    # Asked once more: the fetch may have written the
                    # marker (or its failure) just before it ended.
                    record = self.marker(lead)
                    if record is not None:
                        break
                    failure = self._failure(lead)
                    if failure is not None:
                        raise failure
                    raise BoundaryProducerFailed(
                        f"the fetch of {self.source} {self.cycle} ended "
                        f"without fetching {_lead_words(lead)} and without "
                        "saying why; the preparation cannot build the "
                        "boundary time that needs it")
                if self.writer is not None:
                    self.writer.check_stop()
                row = self._schedule_row(lead)
                if not waiting:
                    waiting = True
                    cause = self._waiting_for(lead, row)
                    _say(f"prepare: waiting for {self.source} "
                         f"{_lead_words(lead)} of the {self.cycle} cycle, "
                         f"{lead_wait_words(cause)}")
                    if self.writer is not None:
                        self.writer.waiting_for_source(cause)
                elif row and ((row.get("state"), row.get("first_seen_at"),
                               row.get("last_answer"))
                              != (cause["state"], cause["first_seen_at"],
                                  cause["last_answer"])):
                    # The fetch's word on the lead changed (asked, posted):
                    # the heartbeat says it, so the seam's reason does.
                    cause = self._waiting_for(lead, row)
                    if _fetch_knows(cause) in ("posted", "unheard"):
                        _say(f"prepare: {self.source} {_lead_words(lead)} "
                             f"of the {self.cycle} cycle "
                             f"{lead_wait_words(cause)}")
                    if self.writer is not None:
                        self.writer.waiting_for_source(cause)
                self.sleep(self.poll_seconds)
        finally:
            if waiting and self.writer is not None:
                self.writer.source_arrived(
                    first_seen_at=None if record is None
                    else record.get("first_seen_at"))
        if waiting:
            self.waits.append({"lead": lead, "seconds": round(
                time.monotonic() - started, 3)})
        return record


def domain_tree_head_fields(domain_labels: Sequence[str], *,
                            root_cache: str,
                            children_receipts: Mapping[str, str] | None = None
                            ) -> dict:
    """What a domain tree's head adds to ``head.json`` (and ``basis.tree``).

    ``layout`` is :data:`LAYOUT_DOMAIN_TREE`; ``domains`` the domain
    labels root first (``d01``, ``d02`` ...); ``children_artifacts`` the
    folder under the prepared root that holds every domain's head
    artifacts, laid out like the sealed ``hierarchy-artifacts/``:
    ``domains/dNN/{receipt.json, native-static.npz, geometry-receipt.json,
    prepared-cache/}`` complete for each child, and for the root its
    static files plus the streamed ``prepared-cache/`` (``root_cache``,
    the head's ``basis.cache.directory``) whose header is written at the
    seal.  ``children_receipts`` is ``{"dNN": sha256 of the child's
    receipt.json}``, which binds every child's head files into the head
    digest (the receipt carries each file's own digest).
    """

    labels = [str(label) for label in domain_labels]
    if len(labels) < 2 or labels[0] != "d01":
        raise ValueError("a domain tree head needs d01 and at least one child")
    root = f"{HIERARCHY_HEAD_DIRNAME}/domains/d01"
    return {
        "layout": LAYOUT_DOMAIN_TREE,
        "domains": labels,
        "children_artifacts": HIERARCHY_HEAD_DIRNAME,
        "root": {
            "prepared_cache": str(root_cache),
            "static_cache": f"{root}/native-static.npz",
            "geometry_receipt": f"{root}/geometry-receipt.json",
        },
        **({"children_receipts": dict(sorted(children_receipts.items()))}
           if children_receipts else {}),
    }


#: What a start state's head restore gives back itself; every other
#: attribute of an initialization result is kept by :class:`TreeStartStates`.
_RESTORED_RESULT_NAMES = frozenset({
    "state", "coord", "base", "surface_pressure", "surface_qv"})


def _array_valued(value) -> bool:
    return isinstance(value, np.ndarray) or hasattr(
        value, "__cuda_array_interface__")


def _attributes(value) -> dict:
    from dataclasses import fields, is_dataclass

    found = dict(getattr(value, "__dict__", None) or {})
    if is_dataclass(value) and not isinstance(value, type):
        found.update({item.name: getattr(value, item.name)
                      for item in fields(value)})
    return found


def _shell(value, *, restored) -> dict:
    """``value``'s attributes but its arrays and the ``restored`` names."""

    return {name: item for name, item in _attributes(value).items()
            if name not in restored and not _array_valued(item)}


class TreeStartStates:
    """A chained domain tree's start states, released at the head, re-read at the seal.

    The head holds every domain's start state: the root's in its streamed
    prepared cache, each child's in its complete artifact set under
    ``hierarchy-head/domains/dNN``.  :meth:`release` keeps only what those
    arrays do not carry (the initialization receipts and moisture floors,
    the met fields' receipts, a child's domain, grid, soil and static
    fields), so no start state stays resident while the root's later
    forcing times are built.  :meth:`reread` restores each state on the
    host from the head, through the checks a forecast restoring the head
    makes (each array's digest, the header digest, the setup fingerprint),
    and the seal writes the one-shot tree from them with the one-shot
    writer.  :meth:`require_sealed_is_head` then holds every sealed
    domain's prepared cache to the head's.

    The breakage this prevents: on the card, the start state held under
    every later build doubled the residency the one-shot tree's start-last
    order exists to avoid, so a CUDA-prepared tree (the auto backend's
    choice on a GPU box) could not chain; on the host it held the root and
    every child for the whole stream.
    """

    def __init__(self, *, root, root_met, children):
        self._root = root
        self._root_met = root_met
        self._children = tuple(children)

    @classmethod
    def release(cls, *, root_result, root_met, child_results,
                child_content_sha256: Mapping[str, str]) -> "TreeStartStates":
        """Keep what the head's arrays do not carry; drop every array.

        ``child_content_sha256`` is ``{"dNN": content_sha256}`` of each
        child's head prepared cache, as the head's child receipts record
        it; :meth:`reread` holds each re-read child to it.
        """

        children = []
        for child in child_results:
            label = f"d{int(child.domain.grid_id):02d}"
            children.append({
                "label": label,
                "content_sha256": str(child_content_sha256[label]),
                "child": _shell(child, restored={
                    "state", "real", "horizontal"}),
                "real": _shell(child.real, restored=_RESTORED_RESULT_NAMES),
                "met": _shell(child.horizontal, restored={"fields"}),
            })
        return cls(root=_shell(root_result, restored=_RESTORED_RESULT_NAMES),
                   root_met=_shell(root_met, restored={"fields"}),
                   children=children)

    @staticmethod
    def _restore(directory: Path, *, domain, grid, identity,
                 content_sha256: str, nested: bool):
        from woof.ingest.prepared_cache import (
            PreparedCacheReader, restore_prepared_cache)
        from woof.native_wrf_contract import load_native_static_cache

        cfg = domain.run
        reader = PreparedCacheReader(directory / "prepared-cache",
                                     expected_identity=identity)
        if reader.content_sha256 != content_sha256:
            raise RuntimeError(
                f"the head's d{int(domain.grid_id):02d} prepared cache is "
                f"not the one the head recorded ({reader.content_sha256} "
                f"against {content_sha256}), so the tree cannot be sealed "
                "from it")
        static = load_native_static_cache(
            directory / "native-static.npz", grid, cfg.ny, cfg.nx)
        return restore_prepared_cache(
            reader.path, expected_identity=identity, cfg=cfg, static=static,
            allow_nested_without_lbc=nested, reader=reader, array_module=np,
            # This is a byte-preserving preparation read, not entry into
            # the strict dynamics. Re-expressing theta here changes the
            # arrays sealed from the already-published child head.
            adopt_dynamics_carrier=False)

    @staticmethod
    def _result(shell, restored):
        from types import SimpleNamespace

        result = restored.initial_result
        return SimpleNamespace(**shell, **{
            name: getattr(result, name) for name in _RESTORED_RESULT_NAMES})

    def reread(self, root, *, exp, grids, root_identity,
               root_content_sha256: str):
        """``(root_result, root_met, root_boundaries, child_results)`` on the host.

        ``root`` is where the tree lives now (the writer's ``root``);
        ``root_identity`` and ``root_content_sha256`` are the root's head
        cache identity and its sealed content digest.
        """

        from types import SimpleNamespace

        domains = Path(root) / HIERARCHY_HEAD_DIRNAME / "domains"
        restored = self._restore(
            domains / "d01", domain=exp.domains[0], grid=grids[0],
            identity=root_identity, content_sha256=root_content_sha256,
            nested=False)
        root_result = self._result(self._root, restored)
        root_met = SimpleNamespace(**self._root_met,
                                   fields=restored.met.fields)
        return (root_result, root_met, restored.boundaries,
                self.reread_children(root, exp=exp))

    def reread_children(self, root, *, exp) -> tuple:
        """Every child's start state, restored on the host from the head.

        The half of :meth:`reread` a tree needs when its root is re-read
        from elsewhere: a native HRRR tree joins the root preparation's own
        sealed cache, exactly as its one-shot tree does, and only its
        children come from ``hierarchy-head/``.
        """

        from types import SimpleNamespace

        domains = Path(root) / HIERARCHY_HEAD_DIRNAME / "domains"
        by_label = {f"d{int(domain.grid_id):02d}": domain
                    for domain in exp.domains}
        children = []
        for child in self._children:
            directory = domains / child["label"]
            header = _read_json(directory / "prepared-cache" / "header.json")
            if not isinstance(header, dict) or "identity" not in header:
                raise RuntimeError(
                    f"the head's {child['label']} prepared cache has no "
                    "readable header, so the tree cannot be sealed from it")
            shell = child["child"]
            restored_child = self._restore(
                directory, domain=by_label[child["label"]],
                grid=shell["grid"], identity=header["identity"],
                content_sha256=child["content_sha256"], nested=True)
            real = self._result(child["real"], restored_child)
            children.append(SimpleNamespace(
                **shell, state=real.state, real=real,
                horizontal=SimpleNamespace(
                    **child["met"], fields=restored_child.met.fields)))
        return tuple(children)

    def require_sealed_is_head(self, artifact_receipt: Mapping[str, object],
                               *, root_content_sha256: str,
                               as_posted: Mapping[str, object] | None = None
                               ) -> None:
        """Refuse a sealed tree whose prepared caches are not the head's.

        A forecast may already be integrating the head's arrays when the
        seal is written; a sealed domain that differed from them would bind
        the run to a tree it did not integrate.

        ``as_posted`` is ``{"root", "head", "manifest_sha256"}`` for an
        as-posted tree: its head prepared every child under the input
        plan's placeholder, so a sealed child carries another content
        digest than its head twin.  Each child is then held to that twin
        by :func:`verify_as_posted_tree_children` (the same arrays,
        metadata and static files, the identity changed only where the
        manifest digest goes), and the receipt must record the sealed
        cache that check read.
        """

        expected = {"d01": root_content_sha256, **{
            child["label"]: child["content_sha256"]
            for child in self._children}}
        if as_posted is not None:
            checked = verify_as_posted_tree_children(
                as_posted["root"], head=as_posted["head"],
                manifest_sha256=str(as_posted["manifest_sha256"]))
            for child in self._children:
                record = checked.get(child["label"]) or {}
                if record.get("head_content_sha256") \
                        != child["content_sha256"]:
                    raise RuntimeError(
                        f"the head's {child['label']} prepared cache is not "
                        "the one the head recorded, so the sealed tree "
                        "cannot be held to it")
                expected[child["label"]] = record["sealed_content_sha256"]
        sealed = {
            f"d{int(domain['grid_id']):02d}": domain["artifacts"][
                "prepared_cache"]["content_sha256"]
            for domain in artifact_receipt["domains"]}
        differing = sorted(label for label in expected | sealed
                           if expected.get(label) != sealed.get(label))
        if differing:
            raise RuntimeError(
                "the sealed tree's prepared caches differ from its head's in "
                f"{differing}, so a forecast started on the head would be "
                "bound to arrays it did not integrate")


# ---------------------------------------------------------------------------
# The consumer
# ---------------------------------------------------------------------------


def _instant(text) -> datetime | None:
    try:
        instant = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return (instant.replace(tzinfo=timezone.utc) if instant.tzinfo is None
            else instant.astimezone(timezone.utc))


def wait_cause(beat, *, needed_valid_time: datetime | None = None
               ) -> dict | None:
    """The source lead a producer waits on, when it is why a seam waits.

    ``beat`` is the producer heartbeat.  A producer ``waiting_for_source``
    on the lead the waiting interval needs, or on an earlier one, is the
    reason (``on: source``); otherwise, ``None``, the wait is on the
    preparation.  ``needed_valid_time`` is the valid time at the end of
    the interval the forecast waits for; without it any source wait of
    this producer counts, since it builds forcing times in order.
    """

    if not isinstance(beat, dict) or beat.get("state") != "waiting_for_source":
        return None
    waiting = beat.get("waiting_for")
    if not isinstance(waiting, dict):
        return None
    if needed_valid_time is not None:
        valid = _instant(waiting.get("valid_time"))
        if valid is not None and valid > needed_valid_time:
            return None
    return {"on": "source",
            **{key: waiting.get(key) for key in WAITING_FOR_FIELDS}}


class SeamWaits:
    """One forecast's waits for boundary intervals, said in three places.

    The ``on_wait`` callback of :class:`StreamedIntervals`.  Each report
    names a cause: a source lead not posted yet (``on: source``, from the
    producer heartbeat) or the interval not built yet (``preparation``).
    A wait whose cause changes (the lead posts, then the build runs)
    closes one record and opens the other.  Each record is said:

    * on the event stream (``emit``) and in :data:`WAIT_LOG_NAME`:
      ``source_wait_started``, ``source_wait_progress`` every
      :data:`SOURCE_WAIT_PROGRESS_SECONDS`, ``source_wait_finished``; or
      ``boundary_wait_started`` / ``boundary_wait_finished`` with
      ``cause: preparation``; each with the model time reached;
    * in ``progress.json``, through ``publish(block)`` (``None`` once the
      wait ends): the ``waiting`` block;
    * on the supervisor heartbeat, through the observer's ``waiting`` /
      ``waited`` hooks (``waiting:source`` or ``waiting:preparation``),
      so a watchdog times the wait by its own bound.

    ``model_time(interval)`` returns ``{phase, model_elapsed_seconds,
    model_valid_time, interval}`` for where the model stands; ``phase``
    is ``start`` before the model has stepped (model time and interval
    ``None``) and ``seam`` after.
    """

    def __init__(self, *, emit: Callable[..., None] | None = None,
                 observer=None,
                 publish: Callable[[dict | None], None] | None = None,
                 model_time: Callable[[int | None], dict] | None = None,
                 say: Callable[[str], None] | None = None,
                 log_path=None, producer_path=None,
                 clock: Callable[[], float] = time.monotonic):
        self._emit = emit
        self._observer = observer
        self._publish = publish
        self._model_time = model_time
        self._say = say
        self.log_path = None if log_path is None else Path(log_path)
        self.producer_path = (None if producer_path is None
                              else Path(producer_path))
        self._clock = clock
        self._open: dict | None = None
        self._lock = threading.Lock()
        #: Every record this run closed: (on, lead or interval, seconds).
        self.closed: list[tuple[str, object, float]] = []

    # -- the callback ----------------------------------------------------

    def __call__(self, waiting: dict | None) -> None:
        with self._lock:
            if waiting is None:
                self._close()
                self._hook("waited")
                self._call(self._publish, None)
                return
            self._report(waiting)

    def abandon(self) -> None:
        """Drop the open record unsaid: the wait ended in a refusal.

        A late lead did not arrive, so no ``*_finished`` event is said;
        the heartbeat goes back to the status the wait interrupted before
        the run writes its seam checkpoint and its failure.
        """

        with self._lock:
            self._open = None
            self._hook("waited")

    # -- records ---------------------------------------------------------

    def _report(self, waiting: Mapping[str, object]) -> None:
        now = self._clock()
        cause = (waiting.get("cause")
                 if isinstance(waiting.get("cause"), dict) else None)
        on = "source" if cause is not None else "preparation"
        interval = waiting.get("interval")
        if cause is not None:
            key = (on, cause.get("source"), cause.get("cycle"),
                   cause.get("lead"))
        else:
            key = (on, interval)
        if self._open is not None and self._open["key"] != key:
            self._close()
        if self._open is None:
            where = self._where(interval)
            record = {
                "key": key, "on": on, "since": now,
                "since_utc": _utc_now(), "last_said": now,
                "interval": where.get("interval"),
                "phase": where.get("phase", "seam"),
                "model_elapsed_seconds": where.get("model_elapsed_seconds"),
                "model_valid_time": where.get("model_valid_time"),
                "reason": str(waiting.get("reason") or ""),
                "cause": dict(cause or {}),
            }
            self._open = record
            self._started(record)
        record = self._open
        record["reason"] = str(waiting.get("reason") or record["reason"])
        waited = now - record["since"]
        due = now - record["last_said"] >= SOURCE_WAIT_PROGRESS_SECONDS
        if on == "source" and due:
            record["last_said"] = now
            self._event("source_wait_progress",
                        **self._source_fields(record, waited))
        self._call(self._publish, self._block(record, waited))
        self._hook("waiting", on,
                   since_utc=record["since_utc"],
                   lead=record["cause"].get("lead"),
                   expected_at=record["cause"].get("expected_at"),
                   late_at=record["cause"].get("late_at"))

    def _where(self, interval) -> dict:
        unknown = {"phase": "seam", "interval": interval,
                   "model_elapsed_seconds": None, "model_valid_time": None}
        if self._model_time is None:
            return unknown
        try:
            return dict(self._model_time(interval))
        except Exception:  # noqa: BLE001 - telemetry never fails a run
            return unknown

    @staticmethod
    def _source_fields(record, waited) -> dict:
        cause = record["cause"]
        return {
            "phase": record["phase"], "source": cause.get("source"),
            "cycle": cause.get("cycle"), "lead": cause.get("lead"),
            "valid_time": cause.get("valid_time"),
            "expected_at": cause.get("expected_at"),
            "late_at": cause.get("late_at"),
            "waited_seconds": round(float(waited), 3),
            "model_elapsed_seconds": record["model_elapsed_seconds"],
            "model_valid_time": record["model_valid_time"],
            "interval": record["interval"],
            "reason": record["reason"],
        }

    def _started(self, record) -> None:
        if record["on"] == "source":
            self._event("source_wait_started",
                        **self._source_fields(record, 0.0))
            cause = record["cause"]
            if record["model_elapsed_seconds"] is None:
                where = "before its first step"
            else:
                where = "at " + _model_words(record["model_elapsed_seconds"],
                                             record["model_valid_time"])
            self._line(f"forecast: waiting {where} for {cause.get('source')} "
                       f"{_lead_words(cause.get('lead'))}, "
                       f"{lead_wait_words(cause)}")
            return
        self._event("boundary_wait_started", interval=record["interval"],
                    reason=record["reason"], cause="preparation",
                    model_elapsed_seconds=record["model_elapsed_seconds"],
                    model_valid_time=record["model_valid_time"])
        self._line(f"prepared forecast: waiting: {record['reason']}")

    def _close(self) -> None:
        record, self._open = self._open, None
        if record is None:
            return
        waited = self._clock() - record["since"]
        if record["on"] == "source":
            cause = record["cause"]
            self.closed.append(("source", cause.get("lead"), waited))
            self._event(
                "source_wait_finished", phase=record["phase"],
                source=cause.get("source"), cycle=cause.get("cycle"),
                lead=cause.get("lead"), waited_seconds=round(waited, 3),
                first_seen_at=self._first_seen_at(cause),
                model_elapsed_seconds=record["model_elapsed_seconds"],
                model_valid_time=record["model_valid_time"])
            self._line(f"forecast: {cause.get('source')} "
                       f"{_lead_words(cause.get('lead'))} arrived after "
                       f"{waited:.0f} s; stepping")
            return
        self.closed.append(("preparation", record["interval"], waited))
        self._event("boundary_wait_finished", interval=record["interval"],
                    seconds=round(waited, 3), cause="preparation",
                    model_elapsed_seconds=record["model_elapsed_seconds"],
                    model_valid_time=record["model_valid_time"])
        self._line(f"prepared forecast: boundary interval "
                   f"{record['interval']} arrived after {waited:.1f} s")

    def _first_seen_at(self, cause) -> str | None:
        if self.producer_path is None:
            return None
        beat = _read_json(self.producer_path)
        arrived = beat.get("arrived") if isinstance(beat, dict) else None
        if (isinstance(arrived, dict)
                and arrived.get("lead") == cause.get("lead")
                and arrived.get("cycle") == cause.get("cycle")):
            return arrived.get("first_seen_at")
        return None

    @staticmethod
    def _block(record, waited) -> dict:
        cause = record["cause"]
        return {
            "reason": record["reason"],
            "waited_seconds": float(waited),
            "on": record["on"],
            "phase": record["phase"],
            "since_utc": record["since_utc"],
            "source": cause.get("source"),
            "cycle": cause.get("cycle"),
            "lead": cause.get("lead"),
            "valid_time": cause.get("valid_time"),
            "expected_at": cause.get("expected_at"),
            "late_at": cause.get("late_at"),
            "interval": record["interval"],
        }

    # -- sinks -----------------------------------------------------------

    def record_source_behind(self, error: "SourceBehind") -> None:
        """Log the terminal ``source_behind`` record (``woof go`` relays it).

        Not emitted on the event stream here: the door that ends the run
        says it once, from the error, just before ``failed``.
        """

        self._log("source_behind", dict(error.details))

    def _event(self, event: str, **fields) -> None:
        if self._emit is not None:
            try:
                self._emit(event, **fields)
            except Exception:  # noqa: BLE001 - telemetry never fails a run
                pass
        self._log(event, fields)

    def _log(self, event: str, fields: Mapping[str, object]) -> None:
        if self.log_path is None:
            return
        line = json.dumps({"event": event, **fields,
                           "emitted_unix_ms": int(time.time() * 1000)},
                          default=str)
        try:
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except OSError:
            pass

    def _hook(self, name: str, *args, **kwargs) -> None:
        hook = getattr(self._observer, name, None)
        if hook is None:
            return
        try:
            hook(*args, **kwargs)
        except Exception:  # noqa: BLE001 - telemetry never fails a run
            pass

    @staticmethod
    def _call(function, *args) -> None:
        if function is None:
            return
        try:
            function(*args)
        except Exception:  # noqa: BLE001 - telemetry never fails a run
            pass

    def _line(self, text: str) -> None:
        if self._say is not None:
            self._say(text)


class StreamedIntervals(Sequence):
    """The lazy interval sequence over a prepared tree's segments.

    ``len`` is the declared count and :attr:`bounds` the schedule, so an
    interval search never loads an interval it does not return.
    ``self[k]`` waits for segment k's marker, loads and hash-checks its
    arrays against the marker's manifest rows, validates it against
    interval 0's layout, and returns the same object on every later call
    (the device slot reloads by object identity).  Iteration walks every
    index, so a consumer that needs the whole set simply waits for it.
    """

    def __init__(self, root, *, head: Mapping[str, object],
                 on_wait: Callable[[dict | None], None] | None = None,
                 poll_seconds: float = 0.2,
                 validate: Callable[[object], None] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 start_time: datetime | None = None):
        self.root = Path(root)
        self.head = head
        #: The run's start valid time, which dates each interval's end so
        #: a producer's source wait can be matched to the interval asked.
        self.start_time = start_time
        if start_time is not None and start_time.tzinfo is None:
            self.start_time = start_time.replace(tzinfo=timezone.utc)
        self.head_sha256 = str(head["head_sha256"])
        cache = head["basis"]["cache"]
        from woof.ensemble.posted_preparation import (
            posted_binding_from_identity, validate_posted_member_head)
        member_binding = posted_binding_from_identity(cache.get("identity", {}))
        if member_binding is not None:
            validate_posted_member_head(member_binding, identity=cache["identity"])
            self._posted_member_binding = member_binding
        lbc = cache.get("lbc")
        if not isinstance(lbc, dict) or not lbc.get("schedule"):
            raise BoundaryStreamError(
                f"{self.root} declares no external boundary schedule")
        self.lbc = lbc
        self.bounds = tuple(
            (float(start), float(end)) for start, end in lbc["schedule"])
        self.fields = tuple(lbc["fields"])
        self.cache_path = self.root / str(cache["directory"])
        self.on_wait = on_wait
        self.poll_seconds = float(poll_seconds)
        self.validate = validate
        self._clock = clock
        self._loaded: dict[int, object] = {}
        self._markers: dict[int, dict] = {}
        self._physical_seen = {}
        physical = head["basis"].get("ensemble_physical")
        if physical is not None:
            from woof.ensemble import physical_boundary
            physical_boundary.validate_head(physical, lbc["schedule"])
            self._physical_seen[0] = physical["initial_receipt"]
        self._lock = threading.RLock()
        #: Intervals loading on a background thread: index -> (box, done).
        self._prefetching: dict[int, tuple[dict, threading.Event]] = {}
        #: One entry per wait the run actually took: (index, seconds).
        self.waits: list[tuple[int, float]] = []

    # Sequence protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self.bounds)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[k] for k in range(*index.indices(len(self))))
        k = int(index)
        if k < 0:
            k += len(self)
        if not 0 <= k < len(self):
            raise IndexError(index)
        with self._lock:
            cached = self._loaded.get(k)
            if cached is not None:
                return cached
            pending = getattr(self, "_prefetching", {}).get(k)
        if pending is not None:
            # Loaded (read, hash-checked, validated) on the prefetch thread;
            # a failure there is this call's failure, raised here.
            box, done = pending
            done.wait()
            with self._lock:
                self._prefetching.pop(k, None)
                if "error" in box:
                    raise box["error"]
                return self._loaded.setdefault(k, box["interval"])
        with self._lock:
            cached = self._loaded.get(k)
            if cached is not None:
                return cached
            marker = self.require(k)
            interval = self._load(k, marker)
            self._loaded[k] = interval
            return interval

    def prefetch(self, index: int) -> bool:
        """Load interval ``index`` on a background thread if it is posted.

        The same load ``self[index]`` performs (marker, arrays read and
        hash-checked, layout and clock validated), started early so the
        stepping thread finds it done.  Only an interval whose marker is
        already posted, and one at a time; ``self[index]`` joins it and
        raises its failure.  Breakage this prevents: every hourly interval
        was read, hashed, copied and validated on the stepping thread at
        the hour boundary, a stall of every card at every forecast hour
        (test host, 8 x RTX 5090, 2026-10-21).  Returns whether a load started.
        """
        k = int(index)
        if not 0 <= k < len(self):
            return False
        with self._lock:
            if (k in self._loaded or self._prefetching
                    or not self.is_ready(k)):
                return False
            box: dict = {}
            done = threading.Event()
            self._prefetching[k] = (box, done)

        def load():
            try:
                box["interval"] = self._load(k, self.require(k))
            except BaseException as error:  # noqa: BLE001 - raised by self[k]
                box["error"] = error
            finally:
                done.set()

        threading.Thread(target=load, name=f"lbc-prefetch-{k}",
                         daemon=True).start()
        return True

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]

    def release(self, index: int) -> None:
        """Let interval ``index`` go; a later ``self[index]`` loads it again.

        For a reader that passes each interval on once (a native HRRR tree
        relaying its root preparation's intervals into its own d01 stream),
        so no interval stays held after it is written.
        """

        with self._lock:
            self._loaded.pop(int(index), None)

    # Readiness ---------------------------------------------------------

    def is_ready(self, index: int) -> bool:
        return segment_marker_path(self.root, index).is_file()

    def ready_prefix(self) -> int:
        """How many intervals, from the first, have their markers."""

        count = 0
        while count < len(self) and self.is_ready(count):
            count += 1
        return count

    def sealed(self) -> bool:
        return prepared_tree_complete(self.root)

    def _wait_reason(self, index: int) -> str:
        start, end = self.bounds[index]
        return (f"boundary interval {index} ({start:g} s to {end:g} s) is "
                "not prepared yet")

    def _wait_report(self, index: int | None, waited: float, *,
                     reason: str | None = None) -> dict:
        """One wait report: its reason, and its cause from the producer."""

        needed = None
        if self.start_time is not None:
            end = self.bounds[-1 if index is None else index][1]
            needed = self.start_time + timedelta(seconds=float(end))
        cause = wait_cause(_read_json(stream_dir(self.root) / PRODUCER_NAME),
                           needed_valid_time=needed)
        if cause is not None:
            reason = source_wait_reason(cause)
        elif reason is None:
            reason = self._wait_reason(index)
        return {"reason": reason, "interval": index,
                "waited_seconds": waited, "cause": cause}

    def _producer_verdict(self, index: int,
                          ready: Callable[[], bool] | None = None) -> None:
        stream = stream_dir(self.root)
        failed = _read_json(stream / FAILED_NAME)
        if isinstance(failed, dict):
            if failed.get("code") == SOURCE_BEHIND_CODE:
                details = failed.get("details")
                raise SourceBehind(details if isinstance(details, dict)
                                   else {})
            raise BoundaryProducerFailed(
                "the boundary producer failed: "
                f"{failed.get('reason', 'no reason recorded')}")
        worker = _local_producer(self.root)
        if (worker is not None and not worker.is_alive()
                and not (ready is not None and ready())):
            # The preparation thread of this process ended and what this
            # wait needs never arrived: a producer that could not even
            # write failed.json (a full disk) must not leave the run
            # waiting forever.
            raise BoundaryProducerFailed(
                "the boundary producer ended without preparing interval "
                f"{index}")
        beat = _read_json(stream / PRODUCER_NAME)
        if isinstance(beat, dict) and beat.get("state") in {"failed",
                                                            "stopped"}:
            raise BoundaryProducerFailed(
                f"the boundary producer {beat.get('state')} after building "
                f"time {beat.get('times_built')}; interval {index} will "
                "never arrive")
        age = _heartbeat_age(beat)
        if _same_process(beat):
            # A producer on a thread of this process is judged by the
            # thread itself (above), and a long call holding the
            # interpreter would stall its heartbeat and this check alike:
            # its age proves nothing here.
            age = None
        if age is not None and age > _silence_limit(beat):
            built = beat.get("times_built") if isinstance(beat, dict) else None
            raise BoundaryProducerSilent(
                f"boundary producer silent for {age:.0f} s after building "
                f"time {built}; interval {index} will never arrive")
        if beat is None and not (stream / HEAD_NAME).is_file():
            raise BoundaryProducerSilent(
                f"{self.root} has no boundary producer and no marker for "
                f"interval {index}")

    def require(self, index: int) -> dict:
        """Wait for segment ``index``'s marker and return it."""

        k = int(index)
        marker = self._markers.get(k)
        if marker is not None:
            return marker
        path = segment_marker_path(self.root, k)
        started = None
        last_report = None
        while not path.is_file():
            now = self._clock()
            if started is None:
                started = now
            if last_report is None or now - last_report >= WAIT_REPORT_SECONDS:
                last_report = now
                if self.on_wait is not None:
                    self.on_wait(self._wait_report(k, now - started))
            self._producer_verdict(k, ready=path.is_file)
            time.sleep(self.poll_seconds)
        if started is not None:
            waited = self._clock() - started
            self.waits.append((k, waited))
            if self.on_wait is not None:
                self.on_wait(None)
        marker = _read_json(path)
        if (not isinstance(marker, dict)
                or marker.get("schema") != SEGMENT_SCHEMA
                or marker.get("head_sha256") != self.head_sha256
                or int(marker.get("index", -1)) != k):
            raise BoundaryStreamError(
                f"segment marker {path} does not belong to head "
                f"{self.head_sha256}")
        if [float(marker["start_seconds"]), float(marker["end_seconds"])] \
                != list(self.bounds[k]):
            raise BoundaryStreamError(
                f"segment {k} bounds differ from the head's schedule")
        physical = self.head["basis"].get("ensemble_physical")
        if physical is not None:
            from woof.ensemble import physical_boundary
            try:
                physical_boundary.validate_segment(physical, k, marker.get("ensemble_physical"),
                                                   self._physical_seen)
            except (ValueError, KeyError, TypeError) as error:
                raise BoundaryStreamError(f"segment {k} physical source binding failed: {error}") from error
        elif "ensemble_physical" in marker:
            raise BoundaryStreamError("segment carries physical inputs its prepared head did not bind")
        member_binding = getattr(self, "_posted_member_binding", None)
        if member_binding is not None:
            from woof.ensemble.posted_preparation import validate_posted_member_segment
            validate_posted_member_segment(marker.get("ensemble_member_input"), member_binding, index=k)
        elif marker.get("ensemble_member_input") is not None:
            raise BoundaryStreamError("segment carries member physical inputs its prepared head did not bind")
        self._markers[k] = marker
        return marker

    def wait_sealed(self, *, timeout: float | None = None) -> None:
        """Wait for ``proof.json``; the producer's verdict ends a dead wait."""

        started = self._clock()
        last_report = None
        while not self.sealed():
            now = self._clock()
            if last_report is None or now - last_report >= WAIT_REPORT_SECONDS:
                last_report = now
                if self.on_wait is not None:
                    self.on_wait(self._wait_report(
                        None, now - started,
                        reason="the preparation is not sealed yet"))
            self._producer_verdict(len(self) - 1, ready=self.sealed)
            if timeout is not None and now - started > timeout:
                raise BoundaryStreamError(
                    f"{self.root} was not sealed within {timeout:g} s")
            time.sleep(self.poll_seconds)
        if self.on_wait is not None and last_report is not None:
            self.on_wait(None)

    def stop(self, reason: str) -> None:
        """Ask the producer to exit unsealed (the run ended first)."""

        request_stop(self.root, reason)

    # Loading -----------------------------------------------------------

    def _load(self, k: int, marker: Mapping[str, object]):
        from woof.ingest.lateral_bc import (
            BoundaryInterval, FieldBoundary, RationalTimeLaw, SideBoundary,
            record_built_end_frame,
        )
        from woof.ingest.prepared_cache import (
            PreparedCacheCorruptError, interval_built_end_frame,
            read_manifest_array,
        )

        arrays = marker["arrays"]
        if sorted(marker["fields"]) != sorted(self.fields):
            raise BoundaryStreamError(
                f"segment {k} carries fields {sorted(marker['fields'])}, "
                f"not the head's {sorted(self.fields)}")
        fields = {}
        for name in marker["fields"]:
            sides = {}
            for side_name in ("west", "east", "south", "north"):
                prefix = f"lbc/{k}/{name}/{side_name}"
                laws = [f"{prefix}/rational_time_v1/{coefficient}"
                        for coefficient in ("quadratic", "denominator_rate")]
                present = [key in arrays for key in laws]
                if any(present) and not all(present):
                    raise PreparedCacheCorruptError(
                        f"segment {k} {prefix} has an incomplete rational "
                        "time law")
                law = (RationalTimeLaw(*(read_manifest_array(
                    self.cache_path, key, arrays[key]) for key in laws))
                    if all(present) else None)
                sides[side_name] = SideBoundary(
                    read_manifest_array(self.cache_path, f"{prefix}/value",
                                        arrays[f"{prefix}/value"]),
                    read_manifest_array(self.cache_path, f"{prefix}/tendency",
                                        arrays[f"{prefix}/tendency"]),
                    law)
            fields[name] = FieldBoundary(**sides)
        # The end frame the builder recorded rides the marker (A140b), so a
        # streamed interval hashes to the forcing row of the sealed cache's
        # interval, and a checkpoint written before the seal records it.
        interval = record_built_end_frame(BoundaryInterval(
            float(marker["start_seconds"]), float(marker["end_seconds"]),
            fields), interval_built_end_frame(
                marker, where=f"segment {k} marker"))
        if k > 0:
            _require_same_layout(self[0], interval, k)
        if self.validate is not None:
            self.validate(interval)
        return interval

    def consumed_markers(self) -> dict[int, dict]:
        return dict(self._markers)


def _require_same_layout(first, interval, index: int) -> None:
    from woof.ingest.lateral_bc import _boundary_field_shape

    def layout(value):
        return tuple(
            (name,) + _boundary_field_shape(value.fields[name]) + tuple(
                getattr(value.fields[name], side).time_law is not None
                for side in ("west", "east", "south", "north"))
            for name in sorted(value.fields))

    if layout(interval) != layout(first):
        raise BoundaryStreamError(
            f"boundary interval {index} has a different inventory or side "
            "layout than interval 0")


class ClockBasis:
    """What the long-step derivation read, before it chose a clock.

    Kept so a run streaming its root's boundaries can ask the same
    derivation again as each interval arrives (the tree runner's
    ``StreamedClockGuard``).
    ``experiment`` is the experiment before the clock was applied,
    ``acoustic`` the substep derivation, ``readers`` and ``statics`` each
    domain's cache reader and static fields by grid id, and ``reach`` the
    corridor terrain of each following nest.
    """

    __slots__ = ("experiment", "acoustic", "readers", "statics", "reach")

    def __init__(self, *, experiment, acoustic, readers, statics, reach):
        self.experiment = experiment
        self.acoustic = tuple(acoustic)
        self.readers = readers
        self.statics = statics
        self.reach = reach


def derived_clock(receipt) -> dict[int, tuple]:
    """The clock a derivation chose for each domain, without its reading.

    ``receipt`` is a :func:`woof.terrain_clock.clock_receipt`.  The
    reading (which instant held the strongest crest wind, and its speed)
    may differ between two derivations that choose the same clock; the
    clock is what the integration runs.
    """

    return {
        int(row["grid_id"]): (
            json.dumps(row.get("dt_s"), sort_keys=True),
            int(row.get("step_division", 1)),
            int(row.get("time_step_sound", 0)),
            json.dumps(row.get("max_time_step_s"), sort_keys=True),
            row.get("min_time_step_sound"))
        for row in (receipt or {}).get("domains", ())}


def keep_interval_check(intervals, check) -> None:
    """Chain ``check`` after the per-interval check the series already has.

    A streaming attachment can install its own layout check as the
    series' ``validate`` hook, replacing what was there; the breakage this
    prevents is a clock check installed before it that silently stopped
    running.  Calling it again with the same check is a no-op.
    """

    if check is None:
        return
    attached = getattr(intervals, "validate", None)
    if attached is check or getattr(attached, "_keeps", None) is check:
        return

    def validate(interval):
        if attached is not None:
            attached(interval)
        check(interval)

    validate._keeps = check
    intervals.validate = validate


def streamed_boundaries(root, *, head, on_wait=None, validate=None,
                        start_time=None):
    """A :class:`LateralBoundaries` whose intervals stream from ``root``."""

    from woof.ingest.lateral_bc import LateralBoundaries

    intervals = StreamedIntervals(root, head=head, on_wait=on_wait,
                                  validate=validate, start_time=start_time)
    lbc = intervals.lbc
    return LateralBoundaries(
        intervals, int(lbc["spec_bdy_width"]), int(lbc["spec_zone"]),
        int(lbc["relax_zone"]))


def _seal_keys(head: Mapping[str, object]) -> frozenset:
    """The proof keys only this head's seal may write."""

    posted = head["basis"].get("as_posted") or {}
    return SEAL_ONLY_PROOF_KEYS | frozenset(
        posted.get("seal_authored_proof_keys") or ())


def hold_composition_rows(manifest: Mapping[str, object],
                           markers: Mapping[str, Mapping[str, object]], *,
                           fixed_rows) -> None:
    """Every lead row of a composition-inputs manifest is a posted object.

    A row the head did not read whole (not in ``fixed_rows``) must be an
    object some lead's marker named: the same path (a manifest sealed as
    posted sits in the fetch folder, whose paths the markers name), bytes
    and digest.  A marker names its fetched ``objects`` and, for a route
    that composes a lead's parts into one file, that file under
    ``composed``; a composed primary's row is held to the latter.
    """

    posted = {}
    for lead, marker in markers.items():
        for key in ("objects", "composed"):
            for item in marker.get(key) or ():
                posted.setdefault(str(item.get("name")), []).append(
                    (lead, item.get("bytes"), item.get("sha256")))
    fixed = {str(path) for path in fixed_rows}
    for row in composition_data_rows(manifest):
        path = str(row["path"])
        if path in fixed:
            continue
        named = [entry for entry in posted.get(path, ())
                 if entry[1:] == (row.get("bytes"), row.get("sha256"))]
        if not named:
            raise BoundaryStreamError(
                f"sealed manifest row {path} ({row.get('bytes')} bytes, "
                f"{row.get('sha256')}) is not an object any lead's posted "
                "marker named")


def verify_as_posted_seal(root, *, head: Mapping[str, object],
                          proof: Mapping[str, object],
                          header: Mapping[str, object]) -> dict:
    """Hold an as-posted seal to its head (the L3 design ruling).

    The sealed manifest, with the route table the fetch's schedule named
    at the seal, implies the head's input plan; each of its lead rows is
    the object the lead's posted marker named; the start leads' markers
    are the ones the head bound and every segment's are the ones its two
    times were built from; the proof names that manifest; and the cache
    identity changed only where the manifest's digest goes.
    """

    root = Path(root)
    posted = head["basis"]["as_posted"]
    manifest_path = root / posted["manifest_path"]
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, ValueError) as error:
        raise BoundaryStreamError(
            f"the as-posted seal wrote no readable input manifest at "
            f"{manifest_path}: {error}") from None
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    record = _read_json(stream_dir(root) / POSTED_LEADS_NAME)
    if (not isinstance(record, dict)
            or record.get("schema") != POSTED_LEADS_SCHEMA
            or record.get("head_sha256") != head["head_sha256"]):
        raise BoundaryStreamError(
            f"the as-posted seal left no {POSTED_LEADS_NAME} for this head")
    prefix = posted["lead_role_prefix"]
    plan = input_plan(manifest, lead_role_prefix=prefix,
                      route_table_sha256=record.get("route_table_sha256"),
                      derived_roles=posted.get("derived_roles") or (),
                      fixed_rows=posted.get("fixed_rows"))
    if input_plan_sha256(plan) != posted["input_plan_sha256"]:
        raise BoundaryStreamError(
            f"the sealed input manifest {manifest_path} is not the input "
            "plan the head bound (a lead, an object name, a non-lead input "
            "or the fetch's route table differs)")
    proof_key = str(posted.get("proof_manifest_key")
                    or "input_manifest_sha256")
    named = proof
    for part in proof_key.split("."):
        named = named.get(part) if isinstance(named, Mapping) else None
    if named != manifest_sha256:
        raise BoundaryStreamError(
            f"the sealed proof names input manifest {named}, not the one "
            f"the seal wrote ({manifest_sha256})")
    markers = {}
    for lead, row in (record.get("leads") or {}).items():
        marker = row.get("marker") if isinstance(row, Mapping) else None
        if (not isinstance(marker, Mapping)
                or posted_lead_marker_sha256(marker)
                != row.get("marker_sha256")):
            raise BoundaryStreamError(
                f"{POSTED_LEADS_NAME} records lead {lead} with a marker "
                "that is not the digest it names")
        markers[str(lead)] = dict(marker)
    if posted.get("fixed_rows") is not None:
        hold_composition_rows(manifest, markers,
                               fixed_rows=posted["fixed_rows"])
    else:
        for role, spec in manifest["files"].items():
            lead = lead_payload_lead(role, prefix)
            if lead is None:
                continue
            objects = {str(item.get("name")): item.get("sha256") for item in
                       (markers.get(str(lead)) or {}).get("objects") or ()}
            if objects.get(spec.get("name")) != spec.get("sha256"):
                raise BoundaryStreamError(
                    f"sealed manifest row {role} ({spec.get('name')}, "
                    f"{spec.get('sha256')}) is not the object lead {lead}'s "
                    "posted marker named")
    decoded = {}
    if posted.get("documents") is not None:
        for lead, row in (record.get("leads") or {}).items():
            if posted.get("decoded_rows") and isinstance(row, Mapping) \
                    and isinstance(row.get("decoded"), Mapping):
                decoded[str(lead)] = dict(row["decoded"])
        hold_document_rows(root, posted=posted, manifest=manifest,
                           markers=markers, decoded=decoded)
    digests = {lead: row["marker_sha256"]
               for lead, row in (record.get("leads") or {}).items()}
    for lead, digest in (posted.get("start_marker_sha256") or {}).items():
        if digests.get(str(lead)) != digest:
            raise BoundaryStreamError(
                f"start lead {lead}'s marker at the seal is not the one the "
                "head bound")
    forcing_leads = posted["forcing_leads"]
    for k in range(len(head["basis"]["cache"]["lbc"]["schedule"])):
        segment = _read_json(segment_marker_path(root, k)) or {}
        spans = {str(int(lead)): digests.get(str(int(lead)))
                 for lead in forcing_leads[k:k + 2]}
        if segment.get("posted_leads") != spans:
            raise BoundaryStreamError(
                f"segment {k} was built from lead markers "
                f"{segment.get('posted_leads')}, not the ones the seal "
                f"records for leads {sorted(spans, key=int)}")
        if posted.get("decoded_rows"):
            records = {str(int(lead)): (
                None if str(int(lead)) not in decoded
                else decoded_lead_record_sha256(decoded[str(int(lead))]))
                for lead in forcing_leads[k:k + 2]}
            if segment.get("decoded_leads") != records:
                raise BoundaryStreamError(
                    f"segment {k} was built from decoded leads "
                    f"{segment.get('decoded_leads')}, not the records the "
                    f"seal holds for leads {sorted(records, key=int)}")
    changed = check_as_posted_identity(
        head["basis"]["cache"]["identity"], header.get("identity"),
        plan_sha256=posted["input_plan_sha256"],
        manifest_sha256=manifest_sha256,
        manifest_bound=posted["manifest_bound_identity_keys"],
        document_bound=document_bound_digests(posted, manifest))
    left = _placeholders_left(proof)
    if left:
        # The sealed proof is the one-shot proof of the same bytes; a
        # placeholder in it names a plan where a digest belongs.
        raise BoundaryStreamError(
            f"the sealed proof still carries the as-posted placeholder at "
            f"{', '.join(left)}")
    return {"input_manifest_sha256": manifest_sha256,
            "identity_changed": changed}


def _header_content_sha256(header: Mapping[str, object]) -> str:
    """A prepared cache header's content digest, recomputed from its basis."""

    basis = {key: header.get(key) for key in (
        "schema", "identity", "metadata", "arrays", "payload_bytes")}
    return hashlib.sha256(_canonical(basis).encode("utf-8")).hexdigest()


def verify_as_posted_tree_children(root, *, head: Mapping[str, object],
                                   manifest_sha256: str) -> dict:
    """Hold each sealed child of an as-posted tree to the child its head prepared.

    An as-posted tree's head prepares every child before the input
    manifest exists, so each child's identity carries the input plan's
    placeholder where the manifest digest goes (the children bind
    ``input_plan_sha256``); the seal writes the one-shot tree
    (``hierarchy-artifacts/``) with the manifest's digest.  Each sealed
    child must be its head twin (``hierarchy-head/domains/dNN``, whose
    receipt the head digest binds, and whose cache header must carry its
    own content digest and be the cache that receipt records) in
    everything but that: the same array table and payload, cache metadata
    completed only where the head named a posted identity digest, the
    same static cache and geometry receipt, a header whose
    content digest is its own, the identity changed only as
    :func:`check_as_posted_identity` allows, and no placeholder left
    anywhere in the sealed header or receipt.  The breakage this prevents:
    a forecast started on the head integrates the head's children; a
    sealed child that differed from them, or that still named the plan,
    would bind the run's receipt to a tree it did not run or to inputs no
    preparation read.  Returns ``{"dNN": {"head_content_sha256",
    "sealed_content_sha256", "identity_changed"}}``, the head digest being
    the one the head binds through the child's receipt.
    """

    root = Path(root)
    posted = head["basis"]["as_posted"]
    document_digests = {}
    if posted.get("document_bound_identity_keys"):
        manifest_path = root / posted["manifest_path"]
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (OSError, ValueError) as error:
            raise BoundaryStreamError(
                "the as-posted tree's child identities require the readable "
                f"sealed input manifest at {manifest_path}: {error}") from None
        observed = hashlib.sha256(manifest_bytes).hexdigest()
        if observed != manifest_sha256:
            raise BoundaryStreamError(
                f"the as-posted tree's input manifest is {observed}, not "
                f"the sealed input manifest {manifest_sha256}")
        document_digests = document_bound_digests(posted, manifest)
    tree = head["basis"].get("tree")
    if not isinstance(tree, Mapping):
        raise BoundaryStreamError(
            "only a domain tree's head has children to hold to its seal")
    labels = [str(label) for label in tree.get("domains") or ()][1:]
    receipts = dict(tree.get("children_receipts") or {})
    if sorted(receipts) != sorted(labels):
        raise BoundaryStreamError(
            f"the as-posted tree head binds child receipts "
            f"{sorted(receipts)}, not every child {labels}")
    found = {}
    for label in labels:
        head_dir = root / HIERARCHY_HEAD_DIRNAME / "domains" / label
        sealed_dir = root / SEALED_HIERARCHY_DIRNAME / "domains" / label
        try:
            receipt_bytes = (head_dir / "receipt.json").read_bytes()
        except OSError as error:
            raise BoundaryStreamError(
                f"the head's {label} receipt is not readable: {error}"
            ) from None
        if hashlib.sha256(receipt_bytes).hexdigest() != receipts[label]:
            raise BoundaryStreamError(
                f"the head's {label} receipt is not the one the head binds")
        head_header = _read_json(head_dir / "prepared-cache" / "header.json")
        sealed_header = _read_json(
            sealed_dir / "prepared-cache" / "header.json")
        sealed_receipt = _read_json(sealed_dir / "receipt.json")
        if not all(isinstance(value, dict) and "identity" in value
                   for value in (head_header, sealed_header)) \
                or not isinstance(sealed_receipt, dict):
            raise BoundaryStreamError(
                f"the as-posted tree's {label} has no readable head and "
                "sealed cache headers and receipt")
        try:
            recorded = json.loads(receipt_bytes)["artifacts"][
                "prepared_cache"]["content_sha256"]
        except (ValueError, TypeError, KeyError):
            recorded = None
        if _header_content_sha256(head_header) \
                != head_header.get("content_sha256") \
                or recorded != head_header.get("content_sha256"):
            # The head digest binds the receipt and the receipt the cache:
            # a head header that is not the one recorded would hand the
            # forecast, and its restart identity, a child the head never
            # bound.
            raise BoundaryStreamError(
                f"the head's {label} prepared cache is not the one its "
                "receipt records")
        head_metadata = json.loads(_canonical(head_header.get("metadata")))
        posted_user = posted.get("posted_user_metadata") or ()
        user = (head_metadata.get("user")
                if isinstance(head_metadata, Mapping) else None)
        if isinstance(user, Mapping):
            user = dict(user)
            for name in posted_user:
                if name not in user:
                    continue
                if user[name] != as_posted_placeholder(
                        posted["input_plan_sha256"]):
                    raise BoundaryStreamError(
                        f"the as-posted head's {label} user metadata "
                        f"{name!r} is not the plan's placeholder")
                user[name] = posted_identity_leaf(sealed_header["identity"],
                                                   name)
            head_metadata["user"] = user
        expected_header = {**head_header, "metadata": head_metadata}
        differing = [key for key in ("schema", "metadata", "arrays",
                                     "payload_bytes")
                     if expected_header.get(key) != sealed_header.get(key)]
        for name in ("native-static.npz", "geometry-receipt.json"):
            try:
                same = (hashlib.sha256((head_dir / name).read_bytes())
                        .hexdigest()
                        == hashlib.sha256((sealed_dir / name).read_bytes())
                        .hexdigest())
            except OSError:
                same = False
            if not same:
                differing.append(name)
        if differing:
            raise BoundaryStreamError(
                f"the sealed {label} differs from the child its as-posted "
                f"head prepared in {differing}")
        if _header_content_sha256(sealed_header) \
                != sealed_header.get("content_sha256"):
            raise BoundaryStreamError(
                f"the sealed {label} cache header fails its content digest")
        changed = check_as_posted_identity(
            head_header["identity"], sealed_header["identity"],
            plan_sha256=posted["input_plan_sha256"],
            manifest_sha256=manifest_sha256,
            manifest_bound=posted["manifest_bound_identity_keys"],
            document_bound=document_digests)
        left = (_placeholders_left(sealed_header)
                + _placeholders_left(sealed_receipt))
        if left:
            raise BoundaryStreamError(
                f"the sealed {label} still carries the as-posted placeholder "
                f"at {', '.join(left)}")
        found[label] = {
            "head_content_sha256": head_header.get("content_sha256"),
            "sealed_content_sha256": sealed_header["content_sha256"],
            "identity_changed": changed,
        }
    return found


#: The head cache record naming user metadata its seal completes
#: (:data:`woof.ingest.prepared_cache.SEAL_COMPLETES_KEY`).
SEAL_COMPLETES_KEY = "seal_completes_user_metadata"


def _require_completed_user(sealed, head_user, completes) -> None:
    """The sealed user metadata is the head's, completed only where named.

    A key the head named in ``seal_completes`` may gain entries at the
    seal; every entry the head wrote must be there unchanged, and every
    other key must be equal.  The breakage this prevents: a seal that
    rewrites a start-time receipt a head-bound forecast already checked.
    """

    if not isinstance(sealed, Mapping) or set(sealed) != set(head_user):
        raise BoundaryStreamError(
            "the sealed header metadata 'user' differs from the head")
    for key, value in head_user.items():
        if key not in completes:
            if sealed[key] != value:
                raise BoundaryStreamError(
                    f"the sealed header metadata 'user' {key!r} differs "
                    "from the head")
            continue
        if not isinstance(sealed[key], Mapping) or any(
                sealed[key].get(name) != entry
                for name, entry in value.items()):
            raise BoundaryStreamError(
                f"the sealed header metadata 'user' {key!r} changes an entry "
                "the head wrote")


def verify_seal(root, *, head: Mapping[str, object],
                consumed: Mapping[int, Mapping[str, object]] | None = None
                ) -> dict:
    """Check a sealed tree against the head it was published under.

    The header's ``content_sha256`` must equal the recomputation from the
    head's arrays plus every segment marker, ``proof.json`` must name this
    head, and every marker a forecast consumed must be the marker the seal
    counted.  Returns ``{"proof_sha256", "content_sha256", "head_sha256"}``.
    """

    from woof.ingest.prepared_cache import PREPARED_CACHE_SCHEMA

    root = Path(root)
    head_digest = str(head["head_sha256"])
    proof_path = root / proof_document_name(head)
    proof = _read_json(proof_path)
    if not isinstance(proof, dict):
        raise BoundaryStreamError(f"{proof_path} is not readable")
    named = (proof.get("boundary_stream") or {}).get("head_sha256")
    if named != head_digest:
        raise BoundaryStreamError(
            f"{proof_path} seals head {named}, not the pinned head "
            f"{head_digest}")
    stray = {key: value for key, value in proof.items()
             if key not in _seal_keys(head)}
    if json.loads(_canonical(stray)) != head["basis"]["proof_head"]:
        raise BoundaryStreamError(
            f"{proof_path} differs from the head proof outside the seal keys")
    cache = head["basis"]["cache"]
    cache_path = root / str(cache["directory"])
    header = _read_json(cache_path / "header.json")
    if not isinstance(header, dict) or header.get("schema") \
            != PREPARED_CACHE_SCHEMA:
        raise BoundaryStreamError(f"{cache_path} has no sealed header")
    arrays = dict(cache["arrays"])
    payload = int(cache["payload_bytes"])
    physical = head["basis"].get("ensemble_physical")
    physical_seen = {}
    if physical is not None:
        from woof.ensemble import physical_boundary
        try:
            physical_boundary.validate_head(physical, cache["lbc"]["schedule"])
            if (cache["metadata"].get("user", {}).get(physical_boundary.KEY)
                    != physical_boundary.head_catalog(physical)):
                raise ValueError("physical head cache metadata differs from its provider binding")
            if (cache["identity"].get("source_identity", {}).get("ensemble_posted_physical_input")
                    != physical["initial_receipt"]["binding"]):
                raise ValueError("physical initial input differs from its prepared identity")
            physical_seen[0] = physical["initial_receipt"]
        except (ValueError, KeyError, TypeError) as error:
            raise BoundaryStreamError(f"physical source head verification failed: {error}") from error
    for k in range(len(cache["lbc"]["schedule"])):
        marker = _read_json(segment_marker_path(root, k))
        if not isinstance(marker, dict) \
                or marker.get("head_sha256") != head_digest:
            raise BoundaryStreamError(f"segment {k} marker is missing at seal")
        if consumed and k in consumed and consumed[k] != marker:
            raise BoundaryStreamError(
                f"segment {k} changed after the forecast consumed it")
        if physical is not None:
            try:
                physical_boundary.validate_segment(physical, k, marker.get("ensemble_physical"), physical_seen)
            except (ValueError, KeyError, TypeError) as error:
                raise BoundaryStreamError(f"physical source segment {k} verification failed: {error}") from error
        elif "ensemble_physical" in marker:
            raise BoundaryStreamError("sealed segment carries physical inputs absent from its prepared head")
        arrays.update(marker["arrays"])
        payload += int(marker["payload_bytes"])
    if header.get("arrays") != arrays or int(header.get("payload_bytes", -1)) \
            != payload:
        raise BoundaryStreamError(
            "the sealed header's array table is not the head plus its "
            "segments")
    basis = {key: header[key] for key in (
        "schema", "identity", "metadata", "arrays", "payload_bytes")}
    content = hashlib.sha256(_canonical(basis).encode("utf-8")).hexdigest()
    if content != header.get("content_sha256"):
        raise BoundaryStreamError("the sealed header fails its content digest")
    if physical is not None:
        try:
            physical_boundary.validate_catalog(physical,
                header["metadata"].get("user", {}).get(physical_boundary.KEY), physical_seen)
        except (ValueError, KeyError, TypeError) as error:
            raise BoundaryStreamError(f"physical source seal verification failed: {error}") from error
    as_posted = None
    if head["basis"].get("as_posted") is not None:
        as_posted = verify_as_posted_seal(root, head=head, proof=proof,
                                          header=header)
        if head["basis"].get("tree") is not None:
            # A tree's children bound the plan at the head too; each is
            # held to its sealed twin here.
            as_posted["children"] = verify_as_posted_tree_children(
                root, head=head,
                manifest_sha256=as_posted["input_manifest_sha256"])
    elif header.get("identity") != cache["identity"]:
        raise BoundaryStreamError("the sealed header names another identity")
    completes = set(cache.get(SEAL_COMPLETES_KEY) or ())
    posted_user = ((head["basis"].get("as_posted") or {})
                   .get("posted_user_metadata") or ())
    for key, value in cache["metadata"].items():
        if key == "user" and posted_user:
            # An as-posted head held the plan's placeholder where the seal
            # writes the digest its sealed identity carries (see
            # PreparedTreeWriter.seal_cache); nothing else may change.
            placeholder = as_posted_placeholder(
                head["basis"]["as_posted"]["input_plan_sha256"])
            value = dict(value)
            for name in posted_user:
                if value.get(name) != placeholder:
                    raise BoundaryStreamError(
                        f"the as-posted head's user metadata {name!r} is not "
                        "the plan's placeholder")
                value[name] = posted_identity_leaf(header.get("identity"),
                                                   name)
        if key == "user" and completes:
            _require_completed_user(header["metadata"].get(key), value,
                                    completes)
        elif header["metadata"].get(key) != value:
            raise BoundaryStreamError(
                f"the sealed header metadata {key!r} differs from the head")
    digest = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    sealed = {"proof_sha256": digest, "content_sha256": content,
              "head_sha256": head_digest}
    if as_posted is not None:
        sealed["as_posted"] = as_posted
    return sealed


__all__ = [
    "HIERARCHY_HEAD_DIRNAME", "LAYOUT_DOMAIN_TREE", "domain_tree_head_fields",
    "AS_POSTED_IDENTITY_KEYS", "AS_POSTED_SCHEMA", "INPUT_PLAN_SCHEMA",
    "COMPOSITION_DATA_SECTIONS", "composition_data_rows",
    "hold_composition_rows", "posted_identity_leaf",
    "POSTED_LEADS_NAME", "POSTED_LEADS_SCHEMA", "as_posted_placeholder",
    "check_as_posted_identity", "input_plan", "input_plan_sha256",
    "DOCUMENT_ROW_RULES", "decoded_lead_record_sha256",
    "document_bound_digests", "hold_document_rows",
    "is_as_posted_placeholder", "SEALED_HIERARCHY_DIRNAME",
    "lead_payload_lead", "verify_as_posted_seal",
    "verify_as_posted_tree_children",
    "AS_POSTED_PLACEHOLDER_PREFIX", "posted_lead_marker_sha256",
    "POSTED_LEAD_SCHEMA", "POSTING_DIRNAME", "POSTING_FAILED_NAME",
    "POSTING_SCHEDULE_NAME", "PostedLeads", "PostedWaitStopped",
    "posted_lead_marker_name", "read_replaced_json",
    "ClockBasis", "StreamedClockChanged", "derived_clock",
    "keep_interval_check",
    "BoundaryProducerFailed", "BoundaryProducerSilent", "BoundaryStreamError",
    "BoundaryStreamStopped", "CHAINED_DEFAULT", "CHAINED_ENV", "FAILED_NAME",
    "SOURCE_BEHIND_CODE", "SOURCE_BEHIND_EXIT_CODE", "SOURCE_BEHIND_FIELDS",
    "SourceBehind", "SeamWaits", "WAIT_LOG_NAME", "WAITING_FOR_FIELDS",
    "source_behind_sentence", "source_wait_reason", "wait_cause",
    "SOURCE_WAIT_PROGRESS_SECONDS",
    "FORECAST_HOST_FLOOR_BYTES", "HEAD_NAME", "HEAD_SCHEMA", "PRODUCER_NAME",
    "PreparedTreeWriter", "SEAL_ONLY_PROOF_KEYS", "SEGMENT_SCHEMA",
    "STOP_NAME", "STREAM_DIRNAME", "StreamedIntervals", "bind_head",
    "chained_enabled", "forecast_host_bytes", "forecast_installed",
    "head_sha256", "host_admission", "live_chained_head",
    "fresh_chained_head", "prepared_tree_complete", "proof_document_name",
    "process_memory_bytes", "read_head", "remove_unfinished_tree",
    "SEALED_REASONS", "request_stop", "run_chained", "say_prepared_sealed",
    "segment_marker_path", "stream_dir", "streamed_boundaries",
    "unfinished_tree_reason", "verify_seal",
]
