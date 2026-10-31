"""The pipeline's stages, each invocable on its own terms.

WRF's usability model is separate executables -- ungrib, metgrid, real,
wrf -- that a person drives from their own script, each reading and
writing files at a documented boundary.  ``woof go`` welded the
equivalent stages shut inside one route: the capability was all there,
but the only supported way in was the whole chain, with our fetch, our
directory layout and our render on the end of it.  A collaborator
running his own pipeline said so in as many words -- he wants to pull
his own data, author his own namelist, run OUR preprocessing on HIS
inputs, then run OUR simulation alone, with nothing fetching and
nothing plotting.

This module is that unbundling.  Three stages, three front doors:

``woof prep``
    Preprocessing, on inputs the caller supplies.  It is not a
    reimplementation of anything: it ADOPTS :mod:`woof.source_cli`'s
    parser through argparse ``parents=`` and calls that module's own
    dispatch, so ``rw-wps`` and ``woof prep`` are one program with two
    spellings and cannot drift.  Nothing here fetches.

``woof sim``
    The forecast, on a prepared tree that already exists.  It reads the
    bundle's own document -- ``proof.json`` or ``receipt.json`` -- to
    learn which source prepared it and whether it is a single domain or
    a tree, relays the digests that document carries to the runner, and
    calls the runner in this process so every line the model prints
    reaches the caller's terminal unbuffered by a pipe.  No fetch, no
    render, no network: :func:`sim_main` reaches dispatch without
    importing :mod:`woof.fetch` at all, and
    ``tests/test_stage_seams.py`` asserts exactly that in a fresh
    interpreter.

``woof render``
    Already stood alone -- it takes wrfout files and writes PNGs -- and
    keeps its own front door in :mod:`woof.render`.  It is named here
    because the contract document treats the three as one family.

``woof go`` is unchanged and stays: it is the average user's front
door.  What changes is that it is now provably a COMPOSITION of these
stages rather than the only way to reach them -- the equivalence gates
in ``tests/test_stage_seams.py`` assert that the command ``go`` composes
for its preparation stage is the command ``woof prep`` runs, and that
the command ``go`` composes for its forecast stage is the command
``woof sim`` runs, for both the single-domain and the tree arm.

**The digest relay is not a convenience and it is not a weakening.**
Every stage of this pipeline refuses inputs the previous stage did not
produce, by comparing a digest the caller supplies against one it
recomputes.  ``woof sim`` reads those digests off the artifacts on
disk and hands them to the runner; the runner still recomputes and
still refuses.  What a caller no longer has to be is a checksum
courier.  A third party who would rather carry them by hand can see the
exact command with ``--print-command`` and run it themselves -- that is
the documented boundary, and it is a real one.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import json
import os
import sys
from pathlib import Path
from typing import Mapping, NamedTuple

from woof import run_stamp
from woof.file_digest import sha256_file

#: The two runner modules, as importable module names.  A wheel has no
#: ``tools/`` directory, so a script path would be a promise only a git
#: checkout keeps -- the same reason ``woof go`` stopped naming one.
SINGLE_DOMAIN_RUNNER = "woof.prepared_single_domain_forecast"
TREE_RUNNER = "woof.prepared_domain_tree_forecast"


def missing_forecast_runners() -> tuple[str, ...]:
    """The runner modules this installation does not carry.

    Both, in the standalone RW-WPS preparation package, which stages this
    module and no forecast.  There :func:`_schema_index`, and so
    :func:`resolve_bundle`, cannot import the tables it reads, so a
    caller asks this first rather than resolving a bundle it cannot run.

    A runner counts only when it is in the directories of the package
    that is running (its ``__path__``), not wherever the import system
    would find it.  An editable woof install adds a meta-path finder
    that maps every ``woof.<name>`` to its checkout, so with the staged
    package first on the path ``find_spec`` returned the checkout's tree
    runner, the handoff went on to resolve the bundle, and importing that
    runner against the staged ``woof.core`` failed ("No module named
    'woof.core.adaptive_clock'").
    """

    missing = []
    for name in (SINGLE_DOMAIN_RUNNER, TREE_RUNNER):
        parent = name.rpartition(".")[0]
        package = sys.modules.get(parent) or importlib.import_module(parent)
        found = importlib.machinery.PathFinder.find_spec(
            name, list(package.__path__)) is not None
        if not found:
            missing.append(name)
    return tuple(missing)


#: Filenames a preparation stage may leave as its top-level document,
#: in the order they are looked for.  Both spellings are real: the
#: GFS/ERA5/mapped routes write ``proof.json`` and the tree routes write
#: ``receipt.json``.  Which SCHEMAS count is not restated here --
#: it is read from the runners' own tables by
#: :func:`_schema_index`, because a second list of schemas in this file
#: is the enumeration drift this tree keeps paying for.
#:
#: These are the BINDABLE documents: each carries the digests a runner
#: re-derives and refuses on.  A route may also leave a document that
#: says only "this preparation finished" -- see
#: :data:`ROUTE_COMPLETION_RECEIPTS`.
BUNDLE_DOCUMENTS = ("proof.json", "receipt.json")

#: The authorities a preparation publishes into the bundle root, under
#: the names the forecast stage binds them by.  Spelled here because
#: this module is the one that documents the boundary object; the HRRR
#: writer's own ``EXPERIMENT_CONFIG_NAME``/``WPS_NAMELIST_NAME`` are
#: asserted equal to these in ``tests/test_hrrr_wizard_door.py``.
PREPARED_EXPERIMENT_CONFIG = "experiment.toml"
PREPARED_WPS_NAMELIST = "namelist.wps"


def prep_front_door(source: str) -> str:
    """``woof prep --source X``: the preprocessing stage, named once."""

    return f"woof prep --source {source}"


def staged_route_commands(
        source: str, *,
        prep_arguments: "tuple[str, ...]" = ("...",),
        prepared_root: str = "DIR",
        outdir: str = "OUT",
        experiment_config: str | None = None,
        wps_namelist: bool = True,
        indent: str = "",
        wrap: bool = False) -> tuple[str, str]:
    """``(prep line, sim line)``: the two-command route, spelled ONCE.

    Three doors tell a reader how to run a source whose route is
    prepare-then-simulate: ``woof go``'s refusal for a source it does
    not orchestrate, ``woof sim``'s refusal for a finished tree that
    published no authorities, and the closing block of ``woof domain``.
    All three had their own copy of the same two commands, and the
    wizard's copy had drifted furthest -- it printed the INTERNALS
    (``python -m tools.prepare_hrrr_wrf``, a ``tools/`` script path no
    wheel contains) plus two ``<printed by ...>`` digest placeholders
    that ``woof sim`` reads off the bundle by itself.

    So the spelling lives here, once, and every door renders from it.
    The parameters are the only things that legitimately differ between
    a refusal (placeholders: ``DIR``, ``OUT``, ``...``) and an emission
    (real paths, one flag per line).

    ``wps_namelist`` is False for the tree arm: the tree runner binds
    its preparation receipt instead, and printing a flag that runner
    does not read is how a reader learns the printed chain is
    approximate.
    """

    arguments = tuple(prep_arguments) + (f"--output-root {prepared_root}",)
    config = (f"{prepared_root}/{PREPARED_EXPERIMENT_CONFIG}"
              if experiment_config is None else experiment_config)
    sim_arguments = [f"--experiment-config {config}"]
    if wps_namelist:
        sim_arguments.append(
            f"--wps-namelist {prepared_root}/{PREPARED_WPS_NAMELIST}")
    sim_arguments.append(f"--outdir {outdir}")
    heads = (prep_front_door(source), f"woof sim {prepared_root}")
    if not wrap:
        return tuple(  # type: ignore[return-value]
            indent + " ".join((head, *items))
            for head, items in zip(heads, (arguments, sim_arguments)))
    continuation = " \\\n" + indent + "    "
    return tuple(  # type: ignore[return-value]
        indent + head + continuation + continuation.join(items)
        for head, items in zip(heads, (arguments, sim_arguments)))


class RouteCompletionReceipt(NamedTuple):
    """What one preparation route leaves to say it finished.

    A completion receipt is not a bindable document: it records that the
    route ran to the end, not the digests a forecast binds.  Reading it
    is how this stage tells "the preparation was interrupted" apart from
    "the preparation finished and published no portable authorities",
    which are different situations with different remedies and used to
    get the same sentence.
    """

    #: Filename at the top of the prepared root.
    document: str
    #: The source whose front door writes it.  The COMMAND is derived
    #: from it (:attr:`route`) rather than stored beside it: a row that
    #: spelled its own front door is a fourth copy of the route, which
    #: is the drift this module's helper exists to end.
    source: str
    #: Where the receipt records whether the preparation completed, and
    #: the value that means it did.
    status_key: str
    finished: str
    #: Where it records the portable authorities it published, if any.
    #: ``None``/absent there means the route finished but left nothing
    #: this stage can bind.
    authorities_key: str
    #: Where the route records WHY it published none, when it tried and
    #: could not.  Relayed verbatim so the reader meets the preparation
    #: stage's own sentence rather than this stage's guess at it.
    refusal_key: str

    @property
    def route(self) -> str:
        """The front door that writes it, as a reader would type it."""

        return prep_front_door(self.source)

    def remedy(self) -> str:
        """The two commands that prepare this route again and run it."""

        prep, sim = staged_route_commands(self.source)
        return (f"`{prep}` writes {BUNDLE_DOCUMENTS[0]}, "
                f"{PREPARED_EXPERIMENT_CONFIG} and {PREPARED_WPS_NAMELIST} "
                f"into that root, and `{sim}` runs it")


#: Completion receipts this stage knows how to read.
#:
#: One row today, and it exists because the refusal below used to state
#: a universal this tree does not keep: "A preparation that finished
#: writes one [of proof.json, receipt.json]".  The native HRRR
#: preparation writes NEITHER at the top of its output root -- its
#: completion receipt is ``public-wrapper-result.json``, and the portable
#: authorities a forecast binds are published beside it.  MEASURED
#: 2026-08-18, Linux shakeout: a prep that exited 0 with
#: ``"status": "PASS"`` and a full ``wrf-native-input`` export was
#: refused as "a partial or interrupted one", which sent the reader
#: looking for a crash that had not happened.
#:
#: A row, not an arm: adding a route means adding four strings, and a
#: route whose preparation writes a bindable document needs no row at
#: all.
ROUTE_COMPLETION_RECEIPTS = (
    RouteCompletionReceipt(
        document="public-wrapper-result.json",
        source="hrrr",
        status_key="status", finished="PASS",
        authorities_key="portable_bundle",
        refusal_key="portable_bundle_refusal"),
)

#: Chunk size for the relay's digests.  Same value :mod:`woof.fetch`
#: uses; spelled here rather than imported because importing the fetch
#: module to run a forecast is precisely the welding this seam removes,
#: and ``tests/test_stage_seams.py`` fails if it comes back.
_DIGEST_CHUNK = 1 << 20


class StageRefusal(ValueError):
    """An input this stage will not run on."""


def _sha256(path: Path) -> str:
    """``sha256`` of one file, streamed."""
    return sha256_file(path, block_bytes=_DIGEST_CHUNK)


# ---------------------------------------------------------------------------
# Stage 1: preprocessing, on the caller's own inputs
# ---------------------------------------------------------------------------

def prep_main(args) -> int:
    """``woof prep``: the preprocessing stage, alone.

    The namespace was produced by :func:`woof.source_cli._parser`'s own
    actions (adopted as an argparse parent), so it is exactly the
    namespace ``rw-wps`` would have produced, and it is handed to that
    module's own dispatch.  There is no second preprocessing
    implementation here and there must never be one.
    """

    from woof import source_cli

    parser = getattr(args, "_stage_parser", None)
    if parser is None:                                  # pragma: no cover
        raise StageRefusal(
            "woof prep was dispatched without its own parser, so a "
            "usage error would print the wrong usage line")
    return source_cli.dispatch(args, parser=parser, program="woof prep")


# ---------------------------------------------------------------------------
# Stage 2: the forecast, on a prepared tree that already exists
# ---------------------------------------------------------------------------

def _schema_index() -> dict[str, dict[str, object]]:
    """``schema -> {source, layout}``, read from the runners' own tables.

    Three tables, none of them restated here:

    * ``prepared_single_domain_forecast._PROOF_SCHEMA`` (plus its
      ``_LEGACY_PROOF_SCHEMAS``) -- a DIRECT single-domain preparation.
    * ``prepared_domain_tree_forecast._HIERARCHY_DOCUMENTS`` -- a
      multi-domain preparation, which also carries the source.
    * ``prepared_single_domain_forecast._HIERARCHY_PROOF_SCHEMA`` --
      the same hierarchy documents, mapped back to their source for the
      case where a caller runs d01 alone out of a tree bundle.

    Importing the two runner modules costs about a fifth of a second and
    pulls in neither CuPy nor :mod:`woof.fetch`, which is what lets
    ``woof sim --print-command`` stay a cheap, offline question.
    """

    from woof import prepared_domain_tree_forecast as tree
    from woof import prepared_single_domain_forecast as single

    index: dict[str, dict[str, object]] = {}

    def record(schema: str, source: str, layout: str) -> None:
        entry = index.setdefault(
            schema, {"source": source, "layout": layout, "sources": []})
        entry["layout"] = layout
        sources = entry["sources"]
        assert isinstance(sources, list)
        if source not in sources:
            sources.append(source)
        # The first source registered stays the default answer, so a
        # bundle nothing disambiguates reads exactly as it always did.
        entry.setdefault("source", source)

    for source, schema in single._PROOF_SCHEMA.items():  # noqa: SLF001
        record(schema, source, "single")
    for source, schemas in single._LEGACY_PROOF_SCHEMAS.items():  # noqa: SLF001
        for schema in schemas:
            record(schema, source, "single")
    # The tree table wins for the hierarchy schemas: those documents are
    # what the tree runner reads, and it owns the mapping.
    for entry in tree._HIERARCHY_DOCUMENTS:             # noqa: SLF001
        record(str(entry["schema"]), str(entry["source"]), "tree")
    return index


def _route_completion(root: Path):
    """The route completion receipt this root carries, and what it says.

    Returns ``(receipt, payload)`` for the first row whose document is
    present and readable as a JSON object, or ``None``.
    """

    for receipt in ROUTE_COMPLETION_RECEIPTS:
        document = Path(root) / receipt.document
        if not document.is_file():
            continue
        try:
            payload = json.loads(document.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue                     # unreadable: it says nothing
        if isinstance(payload, dict):
            return receipt, payload
    return None


def _unbindable_root_refusal(root: Path) -> str:
    """Why this root carries nothing the forecast stage can bind.

    Three different situations used to share one sentence, and two of
    them were slandered by it.  "A preparation that finished writes one;
    a partial or interrupted one does not" is a universal, and the
    native HRRR preparation is the counter-example: it finishes, exports
    ``wrfinput``/``wrfbdy``, and writes its completion receipt under a
    name this list does not contain.  A reader whose prep had just
    printed ``"status": "PASS"`` was told the tree was interrupted.

    So the route's own receipt is consulted before the sentence is
    chosen, and each of the three gets the one that is true of it:

    * a receipt that records a FINISHED preparation with no portable
      authorities -- complete, not runnable here, and the remedy is to
      publish them, not to prepare again from the source bytes;
    * a receipt that records an UNfinished one -- named by the status
      the route itself recorded;
    * no receipt at all -- the original refusal, unchanged, teeth
      intact.
    """

    found = _route_completion(root)
    if found is None:
        return (
            f"{root} carries none of {', '.join(BUNDLE_DOCUMENTS)}, and no "
            "preparation route in this install left a completion receipt in "
            "it either, so nothing in it says which preparation produced it "
            "or what the forecast should bind against.  A preparation that "
            "finished leaves one or the other; a partial or interrupted one "
            "leaves neither, and a partial tree must not be run.")

    receipt, payload = found
    document = Path(root) / receipt.document
    status = payload.get(receipt.status_key)
    if status != receipt.finished:
        return (
            f"{document} records this preparation as "
            f"{receipt.status_key}={status!r}, not {receipt.finished!r}, so "
            f"`{receipt.route}` did not finish and the tree it left behind "
            "must not be run.  Nothing here can tell how far it got; "
            "prepare it again and let it reach the end.")
    # Read the field rather than asserting it: this arm is only reached
    # when no bindable document exists, so the authorities are absent
    # either way -- but a sentence that NAMES a field must have looked
    # at it, or a tree whose receipt and directory disagree gets told
    # something untrue about its own receipt.
    published = payload.get(receipt.authorities_key)
    if published:
        return (
            f"{document} records portable authorities under "
            f"`{receipt.authorities_key}`, but none of "
            f"{', '.join(BUNDLE_DOCUMENTS)} is in {root} -- the receipt "
            "and the directory disagree, so something removed or moved "
            "the published files after the preparation wrote them.  "
            "Restore them from wherever this tree was copied, or prepare "
            "it again; nothing here can rebuild a proof from a receipt "
            "that only describes one.")
    recorded = payload.get(receipt.refusal_key)
    why = (
        f"  the preparation recorded why: {recorded}\n"
        if isinstance(recorded, str) and recorded else
        f"  This release publishes them on every `{receipt.route}` run, so "
        "a tree with none was prepared by an earlier one, where they were "
        "opt-in behind --wps-namelist.\n")
    return (
        f"{document} records a `{receipt.route}` preparation that FINISHED "
        f"({receipt.status_key}={status!r}) -- so this tree is complete, not "
        f"partial -- but its `{receipt.authorities_key}` is empty, which "
        "means the run that produced it published none of the portable "
        f"authorities the forecast stage binds: {', '.join(BUNDLE_DOCUMENTS)}"
        " and the experiment config and WPS namelist they are bound to.\n"
        + why
        + "  remedy: prepare it again with this release -- "
        + receipt.remedy() + "\n"
        "  # the tree on disk is not damaged; nothing here can mint the "
        "digests a forecast binds out of a preparation that did not record "
        "them")


def resolve_bundle(prepared_root: Path) -> dict:
    """What a prepared tree says about itself.

    THE contract boundary of the simulation stage, and the reason
    ``woof sim`` needs no config, no cycle, no area and no source flag:
    a preparation stage writes a top-level document that names its own
    schema, and that schema names both the source that prepared it and
    whether it is one domain or a tree.  Asking the bundle beats asking
    the caller to remember, and it beats asking the config -- the config
    is what the RUN was planned from, the bundle is what actually got
    prepared, and the second one is the input the forecast consumes.

    Returns ``{document, schema, source, layout, domains}``.
    """

    root = Path(prepared_root)
    if not root.is_dir():
        raise StageRefusal(
            f"{root} is not a directory, so there is no prepared tree "
            "here to run.  `woof prep --output-root DIR` writes one; "
            "`woof sim DIR` runs it.")
    for name in BUNDLE_DOCUMENTS:
        document = root / name
        if document.is_file():
            break
    else:
        continued = _continuation_head_bundle(root)
        if continued is not None:
            return continued
        raise StageRefusal(_unbindable_root_refusal(root))
    try:
        payload = json.loads(document.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise StageRefusal(
            f"{document} is not readable JSON ({error}), so this "
            "prepared tree cannot be identified") from None
    if not isinstance(payload, dict):
        raise StageRefusal(
            f"{document} is not a JSON object, so it is not a "
            "preparation document")
    schema = payload.get("schema")
    index = _schema_index()
    if not isinstance(schema, str) or schema not in index:
        raise StageRefusal(
            f"{document} declares schema {schema!r}, which no runner in "
            "this install reads.  Known preparation schemas are:\n  "
            + "\n  ".join(sorted(index))
            + "\n  # a bundle prepared by a different woof release is "
              "not runnable here; prepare it again with this one")
    entry = index[schema]
    source = _resolve_packaged_source(root, entry)
    domains = payload.get("domain_count")
    if not isinstance(domains, int) or isinstance(domains, bool):
        domains = 1 if entry["layout"] == "single" else None
    return {
        "document": document,
        "schema": schema,
        "source": source,
        "layout": entry["layout"],
        "domains": domains,
        "payload": payload,
    }


def _continuation_head_bundle(root: Path) -> dict | None:
    """A restored forcing prefix, bound at the head its descriptor pins.

    A checkpoint taken while the boundaries were still streaming stores a
    forcing PREFIX (``woof.ingest.stream_resume seal-prefix``): the head,
    every interval the restart needs, and ``continuation.json``, which pins
    the head digest and every file's digest.  It has no seal document yet:
    the continuation producer relays the later intervals into it and copies
    the seal in last.  So it is bound exactly as the original chained
    forecast was, at its head (``--prepared-head-sha256``).

    THE BREAKAGE (run_515470d250a1060a, 2026-10-26, GFS streamed 3 km,
    woof 2.9.7): ``go --prepared-root PREFIX --restart`` refused such a
    prefix as "carries none of proof.json, receipt.json", so a forecast
    that failed mid-run, or a spot resume on a new box, could not continue
    from its checkpoint.  The refusal's teeth are kept where they belong:
    the prefix must pass its descriptor and file digests, and
    :func:`resolve_head_bundle` (``bind_head``) still refuses a head whose
    producer failed, was stopped or is silent, so a partial tree that no
    live continuation is finishing is never run.

    ``None`` when ``root`` carries no continuation descriptor.
    """

    from woof.ingest.boundary_stream import BoundaryStreamError
    from woof.ingest.stream_resume import DESCRIPTOR_NAME, verify_prefix

    if not (root / DESCRIPTOR_NAME).is_file():
        return None
    try:
        descriptor, _head = verify_prefix(root)
    except (BoundaryStreamError, OSError, ValueError, KeyError) as error:
        raise StageRefusal(
            f"{root} is a restored forcing prefix whose "
            f"{DESCRIPTOR_NAME} does not verify ({error}), so its bytes are "
            "not the ones the checkpoint bound and it must not be run") from None
    return resolve_head_bundle(root, str(descriptor["head_sha256"]))


def resolve_head_bundle(prepared_root: Path, head_sha256: str) -> dict:
    """What a chained preparation's HEAD says about itself.

    The same answer as :func:`resolve_bundle`, read from
    ``boundary-stream/head.json`` before ``proof.json`` exists: the head
    carries the proof without its seal keys, so its schema names the
    source and the layout exactly as the sealed proof will.  A domain
    tree's head (``basis.tree``) answers ``layout: tree``.  The returned
    bundle carries ``head_sha256``, which :func:`sim_command` relays as
    ``--prepared-head-sha256`` to whichever runner the layout names.
    """

    from woof.ingest.boundary_stream import BoundaryStreamError, bind_head

    root = Path(prepared_root)
    try:
        head = bind_head(root, head_sha256)
    except BoundaryStreamError as error:
        raise StageRefusal(str(error)) from None
    payload = dict(head["basis"]["proof_head"])
    schema = payload.get("schema")
    index = _schema_index()
    if not isinstance(schema, str) or schema not in index:
        raise StageRefusal(
            f"the prepared head in {root} declares schema {schema!r}, which "
            "no runner in this install reads")
    entry = index[schema]
    from woof.ingest.boundary_stream import LAYOUT_DOMAIN_TREE

    tree = head["basis"].get("tree")
    layout = ("tree" if isinstance(tree, dict)
              and tree.get("layout") == LAYOUT_DOMAIN_TREE else "single")
    if entry["layout"] != layout:
        # The breakage this prevents: a head whose proof says one layout
        # and whose basis says another, handed to the runner of either.
        raise StageRefusal(
            f"the prepared head in {root} carries a {entry['layout']} proof "
            f"but a {layout} head, so no runner can bind it")
    domains = 1
    if layout == "tree":
        domains = payload.get("domain_count")
        if not isinstance(domains, int) or isinstance(domains, bool)                 or domains != len(tree.get("domains") or ()):
            raise StageRefusal(
                f"the prepared head in {root} names {tree.get('domains')} "
                f"but its proof counts {domains!r} domains")
    posted = head["basis"].get("as_posted")
    plan = posted.get("input_plan") if isinstance(posted, dict) else None
    planned_manifest = plan.get("manifest") if isinstance(plan, dict) else None
    return {
        "document": root / "boundary-stream" / "head.json",
        "root": root,
        "schema": schema,
        "source": _resolve_packaged_source(
            root, entry, manifest=planned_manifest),
        "layout": layout,
        "domains": domains,
        "payload": payload,
        "head_sha256": str(head["head_sha256"]),
        "source_manifest_sha256": head["basis"].get("input_manifest_sha256"),
    }


def unsealed_head_bundle(prepared_root: Path) -> dict | None:
    """The head bundle of a chained preparation that has not sealed yet.

    ``None`` unless the tree carries a chained head, no ``proof.json`` and
    no failure or stop marker, which is the breakage this prevents: a
    forecast started on a preparation that will never finish.
    """

    from woof.ingest.boundary_stream import live_chained_head

    root = Path(prepared_root)
    head = live_chained_head(root)
    if head is None:
        return None
    return resolve_head_bundle(root, head["head_sha256"])


#: Where a mapped preparation copies the input manifest it consumed.
_MAPPED_EVIDENCE_MANIFEST = "source-evidence/input-manifest.json"

#: Where it copies the two authorities that say WHICH mapping it used.
_MAPPED_EVIDENCE_MAPPING = "source-evidence/mapping.json"
_MAPPED_EVIDENCE_COMPOSITION = "source-evidence/composition.json"


def packaged_source_of(prepared_root: Path, *,
                       manifest: Mapping[str, object] | None = None) -> str | None:
    """Which packaged profile prepared this tree, by its own bytes.

    Every packaged source writes the SAME proof schema, because they are
    one declarative route with different tables -- so the schema cannot
    name the source and this does it the only exact way there is: the
    mapping and composition documents the preparation copied into its
    evidence directory are compared, byte for byte, against the digests
    each shipped profile pins.  That is the identical binding the forecast
    runner enforces, asked one stage earlier so the answer at the door and
    the answer at depth cannot disagree.

    A mapping that differs from the pin only in an admission-only
    declaration (:func:`woof.source_authorities.bound_mapping_refusal`)
    is still that profile's: a preparation made before a release added
    one decodes the same frames, and naming it ``mapped`` relabelled a
    packaged source's forecast as a caller's own mapping (A166).  The
    runner then holds its boundary spacing to the packaged target.

    ``None`` means no shipped profile matches, which is exactly what a
    caller-authored mapping looks like.

    An as-posted head has no sealed manifest yet. Its caller supplies the
    manifest from the digest-bound head's input plan instead, so the same
    authority checks identify the source before and after the seal.
    """

    from woof.source_adapters import packaged_profile_sources
    from woof.source_authorities import (bound_mapping_refusal,
                                          packaged_authority_sha256)

    mapping = Path(prepared_root) / _MAPPED_EVIDENCE_MAPPING
    composition = Path(prepared_root) / _MAPPED_EVIDENCE_COMPOSITION
    if not (mapping.is_file() and composition.is_file()):
        return None
    if manifest is None:
        manifest_path = Path(prepared_root) / _MAPPED_EVIDENCE_MANIFEST
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None  # the sealed evidence reader diagnoses invalid content

    from woof.prepared_source_schemas import source_schemas
    schemas = source_schemas()
    observed_composition = _sha256(composition)
    mapping_bytes = mapping.read_bytes()
    for source, profile_id in packaged_profile_sources().items():
        pins = packaged_authority_sha256(profile_id)
        if (observed_composition == pins["composition"]
                and isinstance(manifest, Mapping)
                and manifest.get("schema") == schemas.get(source)
                and bound_mapping_refusal(profile_id, mapping_bytes) is None):
            return source
    return None


def _resolve_packaged_source(prepared_root: Path, entry: dict, *,
                             manifest: Mapping[str, object] | None = None) -> str:
    """The source a bundle was prepared from, disambiguated when it must be."""

    from woof.prepared_source_schemas import mapped_sources
    mapped = mapped_sources()

    sources = entry.get("sources") or [entry["source"]]
    if not mapped.intersection(sources):
        return str(entry["source"])
    identified = packaged_source_of(prepared_root, manifest=manifest)
    if identified in mapped:
        return str(identified)
    # The shared mapped proof schema identifies a route, not a model.
    # Missing/changed authorities are diagnosed by the sealed evidence reader.
    return "mapped"


def boundary_pricing_source(prepared_root: Path, source: str | None = None):
    """What a forecast of this prepared root prices its boundary tables from.

    ``source`` when the caller holds the runner's own ``--source``;
    otherwise the source the root's document names
    (:func:`resolve_bundle`).  A root prepared from a user's own mapping
    (``mapped``) answers with the mapping document it copied into its
    evidence, whose ``fields`` table says which hydrometeors ride its
    boundary (:func:`woof.boundary_fields.source_boundary_species`); the
    name alone publishes none.  ``None`` when nothing says, which prices
    water vapour only.  Never raises: a price is advisory, and the runner
    admits its forecast again on the cache it restores.
    """

    root = Path(prepared_root)
    if source is None:
        try:
            source = resolve_bundle(root)["source"]
        except Exception:  # noqa: BLE001 - pricing is advisory, never a gate
            return None
    if source != "mapped":
        return source
    try:
        mapping = json.loads(
            (root / _MAPPED_EVIDENCE_MAPPING).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return source
    return mapping if isinstance(mapping, dict) else source


def single_domain_digests(bundle: dict) -> dict:
    """The three digests the single-domain runner binds.

    The digest OF the document, and the two digests carried INSIDE it.
    Read from the file rather than recomputed from the inputs, because
    that is what makes this a RELAY: the runner recomputes both sides
    and refuses on any difference, and a caller that computed them
    itself would be checking its own arithmetic against itself.
    """

    payload = bundle["payload"]
    cache = payload.get("prepared_cache")
    if bundle["layout"] == "tree":
        # Relay the root identity already bound by this proof. The reader
        # verifies it against the canonical manifest, domain receipt and
        # actual cache; the stage does not manufacture a new identity.
        artifacts = payload.get("artifact_receipt")
        domains = artifacts.get("domains") if isinstance(artifacts, dict) else None
        roots = [row for row in domains
                 if isinstance(row, dict) and type(row.get("grid_id")) is int
                 and row["grid_id"] == 1] if isinstance(domains, list) else []
        if (len(roots) != 1 or roots[0].get("parent_id") != 0
                or roots[0].get("schema") != "gpuwm-native-domain-artifact-build-v1"
                or roots[0].get("status") != "READY"):
            raise StageRefusal(
                f"{bundle['document']} carries no unambiguous READY d01 "
                "artifact receipt for --runner single.")
        root_artifacts = roots[0].get("artifacts")
        cache = (root_artifacts.get("prepared_cache")
                 if isinstance(root_artifacts, dict) else None)
    content = cache.get("content_sha256") if isinstance(cache, dict) else None
    if not isinstance(content, str):
        raise StageRefusal(
            f"{bundle['document']} carries no prepared-cache identity "
            "for the requested single-domain forecast.")
    manifest = _source_manifest_digest(bundle, payload)
    if manifest is None:
        raise StageRefusal(
            f"{bundle['document']} names no source input manifest, so "
            "the forecast stage has nothing to bind its forcing data "
            "against.  A preparation that finished always records one.")
    return {
        "proof": _sha256(bundle["document"]),
        "source_manifest": manifest,
        "prepared_content": content,
    }


def _source_manifest_digest(bundle: dict, payload: dict) -> str | None:
    """Where the source manifest's digest lives, on either route.

    The named-source routes (GFS/ERA5/HRRR) put it at the proof's top
    level as ``input_manifest_sha256``.  The MAPPED routes do not: the
    caller-facing manifest is the evidence copy the preparation
    published (``source-evidence/input-manifest.json``), which is the
    exact document the forecast runner re-hashes against
    ``--source-manifest-sha256``.  For every generic packaged profile
    that copy is byte-identical to the composition receipt's sealed
    input manifest; for a named-source member route the two are
    DIFFERENT documents -- the evidence carries the route's own member
    manifest while the receipt seals the bridged composition-inputs
    twin -- so relaying the receipt's digest handed the runner a value
    its own gate refuses.  Hashing the published evidence file is the
    one answer that is right on both shapes, and the receipt record
    stays as the fallback for a tree with no evidence copy.

    Reading only the top level made this seam refuse every real mapped
    bundle with the hierarchy message above -- a refusal that named the
    wrong thing, on the route the whole arbitrary-source story runs on.
    Caught by composing against a bundle the runner's own preflight
    accepts instead of against a proof this seam wrote for itself.
    """

    top_level = payload.get("input_manifest_sha256")
    if isinstance(top_level, str):
        return top_level
    evidence = bundle["document"].parent / _MAPPED_EVIDENCE_MANIFEST
    if evidence.is_file():
        return _sha256(evidence)
    composition = payload.get("source_composition")
    if not isinstance(composition, dict):
        return None
    record = composition.get("input_manifest")
    if not isinstance(record, dict):
        return None
    digest = record.get("sha256")
    return digest if isinstance(digest, str) else None


def tree_digests(bundle: dict, experiment_config: Path) -> dict:
    """The two digests the tree runner binds.

    One preparation receipt, and the experiment config's own digest --
    which the single-domain runner takes on trust from the proof and
    the tree runner does not.
    """

    config = Path(experiment_config)
    if not config.is_file():
        raise StageRefusal(
            f"{config} is not a readable file, and the tree runner "
            "binds the experiment config by digest, so it cannot be "
            "omitted or guessed")
    return {
        "preparation_receipt": _sha256(bundle["document"]),
        "experiment_config": _sha256(config),
    }


def _progress_flags(progress_format: str | None) -> list[str]:
    """``--progress-format`` as the runner spells it, or nothing.

    Nothing is not the same as a default spelled out: omitting the flag
    leaves the choice to the runner, so this seam does not pin a value
    the runner is free to move.
    """

    if progress_format is None:
        return []
    return ["--progress-format", str(progress_format)]


def _restart_flags(layout: str, restart: Path | None,
                   sealed_forcing_extension: bool) -> list[str]:
    """Forward canonical checkpoints; extension still needs hierarchy receipts."""
    if sealed_forcing_extension and layout != "tree":
        raise StageRefusal(
            "sealed forcing extension requires the prepared hierarchy artifact format; "
            "a single prepared bundle binds its complete configuration and stop time.")
    return (["--restart", str(restart)] if restart is not None else []) + (
        ["--sealed-forcing-extension"] if sealed_forcing_extension else [])


def _health_debug_flags(layout: str, enabled: bool) -> list[str]:
    return ["--health-debug"] if enabled else []


def _devices_table_argument(text):
    """``woof sim --devices-table JSON`` as validated :class:`DeviceOptions`.

    The [tiles] precedent (``--tiles JSON``): a prepared bundle binds its
    experiment TOML byte for byte, so a split cannot be asked for by editing
    that file -- the forecast refuses a config whose bytes differ from the
    receipt.  The table travels beside it instead, validated here by the same
    ``DeviceOptions.from_mapping`` the TOML parser uses, and binds nothing.
    It is the only way to name ``ids`` for such a bundle, which the one-card
    proof needs (``{"count": 2, "ids": [0, 0]}``).
    """
    if text is None:
        return None
    from woof.core.devices import DeviceOptions
    try:
        table = json.loads(text)
        return DeviceOptions.from_mapping(table, source="--devices-table")
    except (ValueError, TypeError) as error:
        raise StageRefusal(f"--devices-table refused: {error}") from error


def devices_flags(layout: str, *, options=None):
    """``--devices-table`` for the owning runner: single domain and tree alike.

    The tree runner reads ``domains`` (which grids run split) from the same
    table; the single-domain runner refuses a ``domains`` list that leaves its
    one grid out.
    """
    if options is None or not options.enabled:
        return []
    return ["--devices-table", json.dumps(options.to_mapping(), sort_keys=True)]


def streaming_flags(layout: str, *, tiles=None,
                    stream_init: str | None = None) -> list[str]:
    """Validate and encode the owning runner's execution-only overrides."""
    if tiles is None and stream_init is None:
        return []
    if layout != "single":
        raise StageRefusal(
            "prepared tree streaming uses the experiment's [tiles] table; "
            "--tiles and --stream-init overrides belong to the single-domain "
            "runner and cannot be silently applied to a tree")
    from woof import prepared_single_domain_forecast as single
    from woof.core.streaming import StreamingOptions
    try:
        if isinstance(tiles, StreamingOptions):
            tiles = tiles.to_mapping()
        raw = tiles if isinstance(tiles, str) else (
            None if tiles is None else json.dumps(tiles, sort_keys=True))
        options = single._streaming_options_argument(raw)
        if stream_init is not None and stream_init not in single.STREAM_INIT_CHOICES:
            raise ValueError(f"--stream-init must be one of {single.STREAM_INIT_CHOICES}")
    except (TypeError, ValueError) as error:
        raise StageRefusal(f"streaming --tiles/--stream-init: {error}") from error
    return ([] if options is None else
            ["--tiles", json.dumps(options.to_mapping(), sort_keys=True)]) + (
        [] if stream_init is None else ["--stream-init", stream_init])


def sim_command(bundle: dict, *, experiment_config: Path,
                wps_namelist: Path | None, outdir: Path,
                physics_profile: str | None = None,
                io_mode: str = "history",
                progress_format: str | None = None,
                runner: str = "auto",
                restart: Path | None = None,
                sealed_forcing_extension: bool = False,
                health_debug: bool = False,
                render_products: str | None = None,
                render_dir: Path | None = None,
                tiles=None, stream_init: str | None = None, devices: int | None = None, devices_options=None,
                simulated_radar=None,
                memory_gate: bool = True) -> list[str]:
    """The exact runner command this prepared tree needs.

    This is the seam's published boundary.  ``woof sim
    --print-command`` prints it verbatim, so a caller who would rather
    drive the runner from their own script -- with their own logging,
    their own scheduler, their own restart policy -- can copy one line
    and never call ``woof sim`` again.  A front door that can hand out
    its own replacement is a front door nobody is trapped behind.

    ``progress_format`` controls the hosting caller's stdout protocol.
    Scientific settings and explicitly requested restart or diagnostic
    operands use this same command builder for every caller.
    ``None`` -- the default, and what ``woof sim`` uses --
    leaves the runner's own per-step ``Timing for main:`` lines on
    stdout, which is the whole point of the standalone door.  ``woof
    go`` passes ``"jsonl"`` because it OWNS the runner's stdout: its
    subprocess arm would otherwise buffer tens of megabytes of
    discarded lines, and ``woof run-plan`` hosting the runner in
    process reserves that channel for its own event stream.  The
    equivalence gate in ``tests/test_stage_seams.py`` pins that this
    flag is the only difference for otherwise identical operands.
    """

    layout = bundle["layout"] if runner == "auto" else runner
    if layout not in {"single", "tree"}:
        raise StageRefusal(f"unknown runner arm {layout!r}")
    stream_flags = streaming_flags(layout, tiles=tiles, stream_init=stream_init)
    stream_flags += devices_flags(layout, options=devices_options)
    from woof.simulated_radar_config import execution_flags as radar_flags
    try:
        stream_flags += radar_flags(simulated_radar)
    except (ValueError, TypeError) as error:
        raise StageRefusal(str(error)) from error
    if devices is not None:
        stream_flags += ["--devices", str(devices)]
    profile_flags = ([] if physics_profile is None else
                     ["--physics-profile", str(physics_profile)])
    restart_flags = _restart_flags(layout, restart, sealed_forcing_extension)
    health_flags = _health_debug_flags(layout, health_debug)
    render_flags = ([] if render_products is None else
                    ["--render-products", str(render_products)])
    if render_dir is not None:
        render_flags += ["--render-dir", str(render_dir)]
    # Only when asked, so an ordinary command is the one it always was.
    gate_flags = [] if memory_gate else ["--no-memory-gate"]
    config = Path(experiment_config)
    if layout == "tree":
        digests = tree_digests(bundle, config)
        # A chained tree bound at its head: the runner restores from the
        # head and binds the seal at the end, as the single domain does.
        binding = (["--prepared-head-sha256", str(bundle["head_sha256"])]
                   if bundle.get("head_sha256") is not None else
                   ["--preparation-receipt-sha256",
                    digests["preparation_receipt"]])
        return [sys.executable, "-m", TREE_RUNNER,
                "--prepared-root", str(bundle.get(
                    "root", bundle["document"].parent)),
                *binding,
                "--experiment-config", str(config),
                "--experiment-config-sha256", digests["experiment_config"],
                # The split travels to the tree runner exactly as to the
                # single-domain one (streaming_flags refuses tree [tiles]
                # overrides above, so this is the devices table alone).
                *profile_flags, *stream_flags,
                *restart_flags, *health_flags, *render_flags,
                *gate_flags, *_progress_flags(progress_format),
                "--io-mode", io_mode, "--outdir", str(outdir)]
    if wps_namelist is None:
        raise StageRefusal(
            "a single-domain forecast binds the WPS namelist by digest "
            "through its proof, so --wps-namelist is required.  It is "
            "the same namelist.wps the preparation stage consumed.")
    if bundle.get("head_sha256") is not None:
        # A chained preparation bound at its head: the proof and cache
        # digests do not exist yet, and the runner checks them at the seal.
        # An as-posted head names no manifest: it binds its input plan, and
        # its seal writes the manifest, held to that plan.
        binding = ["--prepared-head-sha256", str(bundle["head_sha256"]),
                   *(() if bundle.get("source_manifest_sha256") is None
                     else ("--source-manifest-sha256",
                           str(bundle["source_manifest_sha256"])))]
    else:
        digests = single_domain_digests(bundle)
        binding = ["--proof-sha256", digests["proof"],
                   "--source-manifest-sha256", digests["source_manifest"],
                   "--prepared-content-sha256", digests["prepared_content"]]
    return [sys.executable, "-m", SINGLE_DOMAIN_RUNNER,
            "--source", str(bundle["source"]),
            "--prepared-root", str(bundle.get("root",
                                              bundle["document"].parent)),
            *binding,
            "--experiment-config", str(config),
            "--wps-namelist", str(Path(wps_namelist)),
            *profile_flags, *stream_flags,
            *restart_flags, *health_flags, *render_flags, *gate_flags,
            *_progress_flags(progress_format),
            "--io-mode", io_mode, "--outdir", str(outdir)]


#: What makes a directory a FINISHED-OR-RUNNING forecast's tree rather
#: than an empty folder: the runner's own top-level products.  Named
#: rather than "is it non-empty" so a caller who pre-created their
#: output directory is not refused for it.
_RUN_ARTIFACTS = ("report.json", "wrfout", "progress.jsonl", "receipts")


def preflight_bundle(bundle: dict, *, experiment_config: Path,
                     wps_namelist: Path | None = None, physics_profile=None,
                     restart: Path | None = None, devices=None):
    """The forecast's actual CPU validation, before a query or a launch.

    Compose the shipped command and use its parser to keep digest bindings
    identical to sim. This performs no output allocation or device contact.
    """
    command = sim_command(bundle, experiment_config=experiment_config,
                          wps_namelist=wps_namelist, physics_profile=physics_profile,
                          restart=restart, devices=devices,
                          outdir=Path(bundle["document"]).parent / ".readiness-unused")
    if bundle["layout"] == "tree":
        from woof import prepared_domain_tree_forecast as runner

        args = runner.build_parser().parse_args(command[3:])
        inputs = runner.preflight_prepared_tree(
            prepared_root=args.prepared_root,
            preparation_receipt_sha256=args.preparation_receipt_sha256,
            prepared_head_sha256=args.prepared_head_sha256,
            experiment_config=args.experiment_config,
            experiment_config_sha256=args.experiment_config_sha256,
            physics_profile=args.physics_profile, devices=args.devices)
    else:
        from woof import prepared_single_domain_forecast as runner

        args = runner.build_parser().parse_args(command[3:])
        # Defect: parser defaults are None, while the normal forecast main
        # resolves omitted clock flags from its bound config before preflight.
        runner._resolve_clock_arguments(args)
        inputs = runner.preflight_prepared_forecast(
            source=args.source, prepared_root=args.prepared_root,
            proof_sha256=args.proof_sha256,
            source_manifest_sha256=args.source_manifest_sha256,
            prepared_content_sha256=args.prepared_content_sha256,
            prepared_head_sha256=args.prepared_head_sha256,
            experiment_config=args.experiment_config, wps_namelist=args.wps_namelist,
            physics_profile=args.physics_profile,
            expert_acknowledgements=tuple(args.ack),
            run_seconds=args.run_seconds, history_interval_seconds=args.history_interval_seconds,
            domain_bundle=args.domain_bundle, devices=args.devices)
    if restart is not None:
        # The standalone preparation package intentionally carries no
        # restart/forecast runtime. Keep that dependency on its optional runner.
        single = runner if bundle["layout"] == "single" else runner.prepared_single
        single.preflight_restart_algorithms(restart, inputs.experiment.root.run)
    return inputs


def readiness_for_bundle(prepared_root: Path, *, experiment_config: Path,
                         wps_namelist=None, physics_profile=None, restart=None,
                         devices=None) -> tuple[dict, int]:
    from datetime import datetime, timezone
    from woof import source_readiness as readiness

    source, why = "prepared", None
    try:
        bundle = resolve_bundle(Path(prepared_root))
        source = str(bundle["source"])
        preflight_bundle(bundle, experiment_config=experiment_config,
                         wps_namelist=wps_namelist, physics_profile=physics_profile,
                         restart=restart, devices=devices)
    except (ValueError, OSError) as refused:
        why = str(refused)
    document = readiness._refusal_document(source, None, "prepared-inputs", None,
                                         False, why or "", datetime.now(timezone.utc))
    if why is None:
        document.update(state="ready", ready=True, refusal=None)
    document["prepared_root"] = str(prepared_root)
    return document, readiness.REFUSED_EXIT if why else readiness.READY_EXIT


def _occupied_by(outdir: Path) -> list[str]:
    """Which run artifacts already sit in ``outdir``."""

    return [name for name in _RUN_ARTIFACTS if (Path(outdir) / name).exists()]


def claim_run_dir(args, bundle: dict, *, claim: bool = True) -> Path:
    """This forecast's own output folder, under the caller's ``--outdir``.

    A forecast writes ``wrfout`` frames named for their valid time, one
    ``report.json`` and one progress stream.  Run the same prepared tree
    twice into one directory and the second run's frames land among the
    first's under identical names while ``report.json`` ends up
    describing only the second -- a renderer pointed at that tree then
    draws one product series out of two different runs, with nothing in
    any file saying so.  That is the breakage the stamped folder
    prevents, and the refusal below is the same breakage stated for the
    one path that can still reach it.

    ``--outdir`` pointed at an existing run folder, or at anything under
    one, is honoured verbatim (the run's identity is already
    established: :func:`woof.run_stamp.owning_run`), so that path
    REFUSES when the folder already holds a forecast rather than writing
    into it.  Asked through that one predicate and not through the
    folder's own name, because ``--print-command`` names the folder with
    :func:`woof.run_stamp.resolve` and the real run makes it with
    :func:`woof.run_stamp.allocate`: two spellings of "is this already
    a run's tree?" is how the printed ``--outdir`` and the written one
    come apart, which is the exact contract ``--print-command`` exists
    to keep.

    ``claim=False`` NAMES the folder without making it, which is what
    ``--print-command`` needs: that flag's own contract is that asking
    the question spends nothing, and a directory is something.  The
    refusal above is still evaluated, because a question whose answer is
    "this would overwrite a finished run" should say so rather than
    print a line that will fail.
    """

    outdir = Path(args.outdir)
    enabled = run_stamp.run_stamp_enabled(args)
    owning = run_stamp.owning_run(outdir)
    if enabled and owning is None:
        init = run_stamp.bundle_init(bundle.get("payload"))
        if not claim:
            return run_stamp.resolve(outdir, init=init, create=False)
        return run_stamp.allocate(outdir, init=init)
    occupied = _occupied_by(outdir)
    if occupied:
        raise StageRefusal(
            f"{outdir} already holds a forecast's output "
            f"({', '.join(occupied)}), and this stage would write a "
            "second run's frames into it under the same names.  One "
            "directory cannot hold two runs: the wrfout frames merge "
            "silently, report.json ends up describing only the last "
            "run, and a render of that tree draws one product series "
            "out of two forecasts.\n"
            + ("  remedy: pass --outdir "
               f"{owning.parent}, which claims a fresh "
               "run-... folder beside this one\n"
               if owning is not None else
               "  remedy: drop --run-stamp off and each run claims its "
               f"own {run_stamp.RUN_PREFIX}... folder under {outdir}\n")
            + f"  # or move {outdir} aside if that run is finished with")
    if claim:
        outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def sim_main(args) -> int:
    """``woof sim``: the forecast, alone.

    Nothing is fetched. Rendering stays off unless products are requested;
    that opt-in draws each output frame of every grid as it lands, on the
    runner's own render worker.  The line ``woof prep`` prints asks for
    the default set.
    The runner is called IN THIS PROCESS rather than spawned, which is
    what puts its per-step output on the caller's terminal as it
    happens instead of behind a pipe -- the whole point of running the
    simulation on its own is watching it run.
    """

    from woof.explain import explain_enabled, render

    try:
        from woof.ensemble.calibration_admission import (
            refuse_public_arguments, refuse_unopened_table)
        try:
            refuse_public_arguments(args)
            # This stage runs the one forecast its prepared bundle holds
            # and opens no ensemble session.  2.8.4 refused an [ensemble]
            # table here; reading it and running one forecast would drop
            # the table without a word.
            refuse_unopened_table(getattr(args, "experiment_config", None),
                                  door="woof sim")
        except ValueError as refusal:
            # Before the bundle is opened or a run folder is claimed.
            raise StageRefusal(str(refusal)) from refusal
        try:
            bundle = resolve_bundle(args.prepared_root)
        except StageRefusal:
            # A chained preparation still producing its later boundary
            # intervals (here, or copied here arrays first and markers
            # last): the forecast binds its head and waits at each seam.
            bundle = unsealed_head_bundle(args.prepared_root)
            if bundle is None:
                raise
            print(f"sim: {args.prepared_root} is still being prepared; "
                  f"binding its head {bundle['head_sha256']} and starting "
                  "the forecast beside it", file=sys.stderr)
        layout = bundle["layout"] if args.runner == "auto" else args.runner
        # Check the actual bundle adapter before allocating an output folder.
        _restart_flags(layout, getattr(args, "restart", None),
                       getattr(args, "sealed_forcing_extension", False))
        _health_debug_flags(layout, getattr(args, "health_debug", False))
        # The run folder is resolved BEFORE the command is composed, so
        # the --outdir in the printed line is the one this stage would
        # actually write to.  --print-command only NAMES it: that flag's
        # own contract is that asking the question spends nothing.
        # Compose and validate before allocating anything. Allocation may
        # choose a different collision suffix; only that operand changes.
        outdir = claim_run_dir(args, bundle, claim=False)
        command = sim_command(
            bundle,
            experiment_config=args.experiment_config,
            wps_namelist=getattr(args, "wps_namelist", None),
            outdir=outdir,
            physics_profile=getattr(args, "physics_profile", None),
            io_mode=getattr(args, "io_mode", "history"),
            progress_format=getattr(args, "progress_format", None),
            runner=getattr(args, "runner", "auto"),
            restart=getattr(args, "restart", None),
            sealed_forcing_extension=getattr(args, "sealed_forcing_extension", False),
            health_debug=getattr(args, "health_debug", False),
            render_products=getattr(args, "render_products", None),
            render_dir=getattr(args, "render_dir", None),
            tiles=getattr(args, "tiles", None),
            devices=getattr(args, "devices", None),
            devices_options=_devices_table_argument(
                getattr(args, "devices_table", None)),
            simulated_radar=getattr(args, "simulated_radar_table", None),
            stream_init=getattr(args, "stream_init", None),
            memory_gate=not getattr(args, "no_memory_gate", False))
        if not getattr(args, "print_command", False):
            outdir = claim_run_dir(args, bundle, claim=True)
            command[command.index("--outdir") + 1] = str(outdir)
    except StageRefusal as refusal:
        print(render(str(refusal), explain=explain_enabled(args),
                     command="woof sim"), file=sys.stderr)
        return 2

    if getattr(args, "print_command", False):
        # One line for the shell it is printed in: PowerShell on Windows,
        # where a quoted interpreter path followed by -m is a syntax
        # error, and a POSIX shell elsewhere.
        from woof.prep_output import shell_command

        print(shell_command(command))
        return 0

    layout = bundle["layout"] if args.runner == "auto" else args.runner
    requested = getattr(args, "render_products", None)
    plots = requested is not None and str(requested).strip().lower() not in {"", "none"}
    # Said as what this run does.  It used to read "no render on this
    # route" whenever the flag was absent, although both runners draw
    # every output frame as it lands when asked, and "first-frame plots"
    # when it was given, although they draw every frame.
    if plots:
        render_note = (f"no fetch; drawing {requested} from each output "
                       "frame as it lands")
    elif requested is None:
        render_note = ("no fetch; pass --render-products to draw each "
                       "output frame as it lands")
    else:
        render_note = f"no fetch; frame maps disabled (--render-products {requested})"
    if os.environ.get("WOOF_VERIFY_VISUALS", "1").lower() not in {"0", "false", "off"}:
        render_note += "; observation verification at finish as reports are available"
    print(f"sim: {bundle.get('root', bundle['document'].parent)} -- "
          f"{bundle['schema']} "
          f"(source {bundle['source']}, "
          f"{'domain tree' if layout == 'tree' else 'single domain'}), "
          + render_note)
    if outdir != Path(args.outdir):
        print(f"sim: run folder "
              f"{run_stamp.relative_to_case(outdir, args.outdir)} under "
              f"{args.outdir}")
    # In-process, and by module rather than by path: `python -m` would
    # buy nothing here except a pipe between the model's output and the
    # terminal that is watching it.
    if layout == "tree":
        from woof import prepared_domain_tree_forecast as runner_module
    else:
        from woof import prepared_single_domain_forecast as runner_module
    return runner_module.main(command[3:])


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_cli(subparsers) -> None:
    """Register ``prep`` and ``sim`` on the woof command surface."""

    from woof import source_cli

    prep = subparsers.add_parser(
        "prep",
        parents=[source_cli._parser(  # noqa: SLF001 - the donor, on purpose
            add_help=False, include_version=False)],
        help="preprocessing alone, on inputs you supply -- your GRIB or "
             "NetCDF files, your namelist.wps, your experiment config -- "
             "writing a prepared tree that `woof sim` runs.  Fetches "
             "nothing; the same program as the standalone rw-wps entry "
             "point, spelled as a woof subcommand",
        description="Preprocessing, on inputs the caller supplies.  This "
                    "is the same program as the `rw-wps` console script, "
                    "sharing one parser and one implementation; it "
                    "downloads nothing and renders nothing.")
    prep.set_defaults(func=prep_main)
    # The dispatch calls `parser.error`, and the usage line a reader
    # sees has to be the usage line of the command they typed.
    prep.set_defaults(_stage_parser=prep)

    sim = subparsers.add_parser(
        "sim",
        help="the forecast alone, on a prepared tree that already "
             "exists -- no fetching; rendering is opt-in.  The "
             "bundle's own proof/receipt says which source prepared it "
             "and whether it is one domain or a tree",
        description="Run a prepared tree and exit.  The digests each "
                    "runner binds are read off the bundle's own "
                    "document and relayed; every identity check the "
                    "runner makes is unchanged.  --print-command emits "
                    "the exact runner line so you can drive it "
                    "yourself.")
    from woof.output_formats import add_arguments
    add_arguments(sim)
    sim.add_argument("prepared_root", type=Path, metavar="PREPARED_ROOT",
                     help="a prepared tree written by `woof prep "
                          "--output-root`, by the rw-wps console script, "
                          "or by `woof go`'s preparation stage")
    sim.add_argument("--experiment-config", type=Path, required=True,
                     metavar="TOML", dest="experiment_config",
                     help="the experiment TOML this preparation was "
                          "bound to (the tree runner binds its digest; "
                          "the single-domain runner binds it through "
                             "the proof)")
    from woof.simulated_radar_config import add_execution_argument
    add_execution_argument(sim)
    sim.add_argument("--wps-namelist", type=Path, default=None,
                     metavar="WPS", dest="wps_namelist",
                     help="the namelist.wps this preparation consumed; "
                          "required for a single-domain forecast, "
                          "unused by the tree runner")
    sim.add_argument("--outdir", type=Path, required=True, metavar="DIR",
                     help="where this forecast's output goes.  By "
                          "default it is the parent of one timestamped "
                          "run folder per forecast, so running the same "
                          "prepared tree twice never merges two runs' "
                          "wrfout frames and report.json.  Point it at "
                          "an existing run-... folder and that folder "
                          "is used as given -- and refused if a "
                          "forecast is already in it")
    run_stamp.add_argument(sim, option="--outdir",
                           artifacts="wrfout, report.json and receipts")
    sim.add_argument("--restart", type=Path, default=None, metavar="RST",
                     help="resume a prepared single-domain checkpoint or any "
                          "member of a hierarchy checkpoint set into a fresh "
                          "output folder; the runtime validates config, inputs and "
                          "checkpoint identity without changing settings")
    sim.add_argument("--health-debug", action="store_true",
                     help="write per-step health diagnostics with the prepared hierarchy runner")
    sim.add_argument("--sealed-forcing-extension", action="store_true",
                     help="use the existing prepared-tree append-only forcing "
                          "prefix contract when writing or restoring checkpoints")
    from woof.prepared_single_domain_forecast import STREAM_INIT_CHOICES
    sim.add_argument("--devices", type=int, default=None, metavar="N",
                     help="resident slab count; replaces [devices] count "
                          "(on a tree, for every grid [devices] domains names)")
    sim.add_argument("--devices-table", default=None, metavar="JSON",
                     dest="devices_table",
                     help="[devices] table as JSON (count, grid, ids, transport, "
                          "and on a tree domains); validated by the runner, "
                          "without modifying the prepared configuration or its "
                          "digests")
    sim.add_argument("--tiles", default=None, metavar="JSON",
                     help="single-domain streaming override as a JSON [tiles] "
                          "mapping; validated by the runner, without modifying "
                          "the prepared configuration or its digests")
    sim.add_argument("--stream-init", choices=STREAM_INIT_CHOICES, default=None,
                     help="single-domain streamed initialization: auto prices "
                          "both roads; resident or store forces that road")
    sim.add_argument("--physics-profile", default=None, metavar="ID",
                     dest="physics_profile",
                     help="optional assertion that the hash-bound "
                          "experiment IS this shipped suite; omitted, "
                          "the experiment's own suite runs as written")
    sim.add_argument("--render-products", default=None, metavar="SPEC",
                     help="render selected products from every committed "
                          "history frame of every grid as it lands, while the "
                          "forecast runs: comma-separated catalog selectors, "
                          "'all', or 'none'. Omitted means no rendering")
    sim.add_argument("--render-dir", type=Path, default=None, metavar="DIR",
                     help="picture directory (default OUTDIR/png); "
                          "ignored without --render-products")
    sim.add_argument("--no-memory-gate", action="store_true",
                     dest="no_memory_gate",
                     help="restore a forecast whose priced peak envelope "
                          "exceeds this card's free memory anyway, as "
                          "`woof go --no-memory-gate` does: the envelope is "
                          "an upper bound and the card's own allocation then "
                          "decides; a model state too big to build at all is "
                          "still refused")
    sim.add_argument("--io-mode", default="history", choices=("history",),
                     dest="io_mode",
                     help="history output (the only mode this seam "
                          "offers; `--io-mode none` is a runner-level "
                          "diagnostic, reachable through "
                          "--print-command)")
    sim.add_argument("--runner", default="auto",
                     choices=("auto", "single", "tree"),
                     help="which runner arm to use.  'auto' (default) "
                          "reads it off the bundle's own schema and "
                          "domain count; the explicit values exist for "
                          "a caller who knows better and wants to be "
                          "refused precisely when they do not")
    sim.add_argument("--progress-format", default=None,
                     choices=("text", "jsonl", "off"),
                     dest="progress_format",
                     help="how the run reports progress.  Omitted, the "
                          "runner's own default applies, which is the "
                          "WRF-shaped `Timing for main:` line per step "
                          "per domain on this terminal -- watching it "
                          "run is the reason to run the stage alone.  "
                          "`jsonl` is what `woof go` passes, because it "
                          "owns the runner's stdout; pass it here when "
                          "you are hosting this stage the same way")
    sim.add_argument("--print-command", action="store_true",
                     dest="print_command",
                     help="print the exact runner command, with every "
                          "digest filled in, and exit without running "
                          "it -- the documented boundary a third-party "
                          "script writes to.  The --outdir in that line "
                          "is this run's own timestamped folder, named "
                          "but not created: asking the question spends "
                          "nothing, and the runner makes the directory "
                          "when you run the line.  The line is quoted for "
                          "PowerShell on Windows and for a POSIX shell "
                          "elsewhere")
    sim.set_defaults(func=sim_main)
    return None


__all__ = [
    "BUNDLE_DOCUMENTS", "ROUTE_COMPLETION_RECEIPTS", "RouteCompletionReceipt",
    "SINGLE_DOMAIN_RUNNER", "StageRefusal",
    "TREE_RUNNER", "boundary_pricing_source", "prep_main", "register_cli",
    "resolve_bundle",
    "sim_command", "sim_main", "single_domain_digests", "tree_digests",
]
