"""The pipeline's three stages, each reachable and each usable alone.

A collaborator running WOOF-ArWen inside his own pipeline reported the defect
this file is the gate for: he wants to pull his own data, author his own
namelist, run OUR preprocessing on HIS inputs, then run OUR simulation
alone with nothing fetching and nothing plotting.  ``woof go`` welded
those stages shut -- the capability was all present, but the only
supported way in was the whole chain.

So these are DOOR tests, in the project's sense: a headline feature owns
a permanent test that the front door exists, that a worked example
reaches it, and that the seam is a real boundary rather than a private
call.  Four properties:

1. **Reachability.**  ``woof prep`` and ``woof sim`` are on the real
   command surface, take ``--help``, and dispatch.
2. **One implementation, two spellings.**  ``woof prep`` adopts
   :mod:`woof.source_cli`'s parser rather than restating it, so the
   subcommand and the standalone ``rw-wps`` script cannot drift.
3. **The seam is a boundary.**  ``woof sim --print-command`` emits the
   exact runner line a third party can run themselves, composed from
   digests read off the bundle's own artifacts.
4. **``go`` is a composition, not a second route.**  The command ``go``
   composes for its forecast stage is byte-identical to the one ``woof
   sim`` composes for the same prepared tree -- on the single-domain arm
   AND the tree arm.  If those ever diverge, ``go`` has stopped being an
   orchestration of the stages and become a parallel implementation,
   which is the exact failure the unbundling was for.

And one negative property that is the user's own words made
machine-checkable: running the simulation stage must not import the
fetch machinery.  "no data pulling" is not a promise in prose here, it
is an assertion about ``sys.modules`` in a fresh interpreter.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from woof import go_cli, stage_cli
from woof.cli import build_parser, main as cli_main
from host_shell_words import host_shell_words


# ---------------------------------------------------------------------------
# Fixtures: prepared trees, faked at the boundary the seam actually reads
# ---------------------------------------------------------------------------
#
# The seam's whole contract is "a preparation stage leaves a top-level
# document that names its own schema".  That is what these build.  No
# GPU, no GRIB and no rw-wps run is needed to test the BOUNDARY, and a
# test that needed them would not be a door test, it would be a campaign.

def _single_domain_bundle(root: Path, *, source: str = "gfs") -> Path:
    """A prepared tree shaped like a finished single-domain preparation."""

    from woof.prepared_single_domain_forecast import _PROOF_SCHEMA

    root.mkdir(parents=True, exist_ok=True)
    (root / "proof.json").write_text(json.dumps({
        "schema": _PROOF_SCHEMA[source],
        "status": "READY",
        "input_manifest_sha256": "11" * 32,
        "prepared_cache": {"content_sha256": "22" * 32},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def _tree_bundle(root: Path, *, source: str = "gfs", domains: int = 3) -> Path:
    """A prepared tree shaped like a finished hierarchy preparation."""

    from woof.prepared_single_domain_forecast import _HIERARCHY_PROOF_SCHEMA

    root.mkdir(parents=True, exist_ok=True)
    (root / "proof.json").write_text(json.dumps({
        "schema": _HIERARCHY_PROOF_SCHEMA[source],
        "status": "READY",
        "domain_count": domains,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def _authority(root: Path) -> tuple[Path, Path]:
    """The experiment/WPS pair a preparation was bound to."""

    root.mkdir(parents=True, exist_ok=True)
    config = root / "experiment.toml"
    wps = root / "namelist.wps"
    config.write_text("[experiment]\nname = 'seam'\n", encoding="utf-8")
    wps.write_text("&share\n/\n", encoding="utf-8")
    return config, wps


# ---------------------------------------------------------------------------
# 1. Reachability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("command", ["prep", "sim"])
def test_the_stage_is_on_the_real_command_surface(command):
    """The door exists.  Engine-proven is not shipped."""

    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices  # noqa: SLF001
    assert command in choices, (
        f"`woof {command}` is not registered, so the stage has no front "
        "door and the capability does not exist for a user")


@pytest.mark.parametrize("command", ["prep", "sim", "render"])
def test_every_stage_takes_help_without_touching_anything(command):
    """``--help`` on each stage exits 0 and prints its own usage."""

    result = subprocess.run(
        [sys.executable, "-m", "woof.cli", command, "--help"],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert f"woof {command}" in result.stdout


def test_prep_dispatches_to_the_preprocessing_stage(capsys):
    """A worked example that runs: ask preprocessing what it supports."""

    assert cli_main(["prep", "--show-source", "mapped"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_id"] == "mapped"
    assert payload["runnable"] is True


# ---------------------------------------------------------------------------
# 2. One implementation behind two spellings
# ---------------------------------------------------------------------------

def test_prep_adopts_the_preprocessing_parser_rather_than_restating_it():
    """Every ``rw-wps`` flag is a ``woof prep`` flag, by construction.

    Not "the lists happen to match today": ``prep`` is registered with
    the source CLI's own parser as an argparse ``parents=`` donor, so a
    flag added to one is a flag on the other with no edit here.  This
    asserts the consequence, which is what a user experiences.
    """

    from woof import source_cli

    donor = {option
             for action in source_cli._parser()._actions  # noqa: SLF001
             for option in action.option_strings}
    parser = build_parser()
    prep = parser._subparsers._group_actions[0].choices["prep"]  # noqa: SLF001
    adopted = {option
               for action in prep._actions  # noqa: SLF001
               for option in action.option_strings}
    # `--version` is deliberately dropped: woof owns `woof version`.
    missing = donor - adopted - {"--version"}
    assert not missing, (
        f"woof prep is missing preprocessing flags {sorted(missing)}, so "
        "the subcommand and the rw-wps console script are two programs")


def test_the_preprocessing_stage_has_one_body_behind_both_doors():
    """``rw-wps`` parses and calls the same dispatch ``prep`` calls."""

    from woof import source_cli

    assert hasattr(source_cli, "dispatch")
    assert source_cli.main.__module__ == source_cli.dispatch.__module__


# ---------------------------------------------------------------------------
# 3. The seam is a real, published boundary
# ---------------------------------------------------------------------------

def test_sim_reads_the_bundles_own_document_for_source_and_layout(tmp_path):
    """No ``--source``, no config, no cycle: the bundle says what it is."""

    root = _single_domain_bundle(tmp_path / "prepared", source="era5")
    bundle = stage_cli.resolve_bundle(root)
    assert bundle["source"] == "era5"
    assert bundle["layout"] == "single"

    tree = _tree_bundle(tmp_path / "tree", source="gfs", domains=4)
    resolved = stage_cli.resolve_bundle(tree)
    assert resolved["source"] == "gfs"
    assert resolved["layout"] == "tree"
    assert resolved["domains"] == 4


def test_sim_print_command_emits_a_runnable_line_and_spends_nothing(tmp_path):
    """The documented boundary: one line a third-party script can run."""

    root = _single_domain_bundle(tmp_path / "prepared")
    config, wps = _authority(tmp_path / "authority")
    assert cli_main([
        "sim", str(root), "--experiment-config", str(config),
        "--wps-namelist", str(wps), "--outdir", str(tmp_path / "run"),
        "--print-command"]) == 0
    # Nothing ran: no output directory was created by asking the question.
    assert not (tmp_path / "run").exists()


def test_the_printed_command_carries_the_digests_off_the_artifacts(tmp_path):
    """A relay, not a re-derivation.

    The two digests INSIDE ``proof.json`` are transported verbatim, and
    the digest OF ``proof.json`` is the file's own sha256.  The runner
    still recomputes every one of them and still refuses on any
    difference; what the caller no longer has to be is a courier.
    """

    root = _single_domain_bundle(tmp_path / "prepared")
    config, wps = _authority(tmp_path / "authority")
    command = stage_cli.sim_command(
        stage_cli.resolve_bundle(root), experiment_config=config,
        wps_namelist=wps, outdir=tmp_path / "run")
    assert command[command.index("--source-manifest-sha256") + 1] == "11" * 32
    assert command[command.index("--prepared-content-sha256") + 1] == "22" * 32
    assert (command[command.index("--proof-sha256") + 1]
            == stage_cli._sha256(root / "proof.json"))  # noqa: SLF001


def test_a_partial_prepared_tree_is_refused_naming_what_is_missing(tmp_path):
    """A refusal, not a warning: a partial tree must not be run."""

    root = tmp_path / "half"
    root.mkdir()
    with pytest.raises(stage_cli.StageRefusal) as refusal:
        stage_cli.resolve_bundle(root)
    sentence = str(refusal.value)
    assert "proof.json" in sentence
    # The teeth: nothing here says "finished", and the reader is told the
    # tree must not be run.
    assert "partial or interrupted one leaves neither" in sentence
    assert "must not be run" in sentence


# ---------------------------------------------------------------------------
# 3a. "No bindable document" is three situations, not one
# ---------------------------------------------------------------------------
#
# The refusal above used to be the ONLY answer, and it stated a
# universal: "A preparation that finished writes one".  The native HRRR
# preparation is the counter-example -- it finishes, exports
# wrfinput/wrfbdy, and writes its completion receipt under a name that
# list does not contain.  MEASURED 2026-08-18 on the Linux shakeout:
#
#   proof/prep carries none of proof.json, receipt.json, ... A
#   preparation that finished writes one; a partial or interrupted one
#   does not, and a partial tree must not be run.
#
# while proof/prep/public-wrapper-result.json read {"status": "PASS",
# "portable_bundle": null} and the export was complete.

def _hrrr_native_root(root: Path, *, status: str = "PASS",
                      portable_bundle=None, refusal=None) -> Path:
    """A native HRRR output root, by its own completion receipt."""

    root.mkdir(parents=True)
    (root / "public-wrapper-result.json").write_text(
        json.dumps({"status": status, "portable_bundle": portable_bundle,
                    "portable_bundle_refusal": refusal,
                    "wrf_input": str(root / "wrf-native-input")},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    return root


def test_a_finished_hrrr_tree_without_authorities_is_not_called_partial(
        tmp_path):
    """The measured misdiagnosis, closed.

    A complete preparation must not be described as interrupted: the
    reader who meets that sentence goes looking for a crash that never
    happened.  What is actually true is that this tree carries no
    portable authorities to bind, and the remedy is to publish them --
    which this release does on every run.
    """

    root = _hrrr_native_root(tmp_path / "prep")
    with pytest.raises(stage_cli.StageRefusal) as refusal:
        stage_cli.resolve_bundle(root)
    sentence = str(refusal.value)
    assert "FINISHED" in sentence
    assert "complete, not partial" in sentence
    assert "interrupted" not in sentence
    # It names the receipt it read, the route that wrote it, and the two
    # commands that produce a runnable tree.
    assert "public-wrapper-result.json" in sentence
    assert "woof prep --source hrrr" in sentence
    assert "woof sim DIR --experiment-config DIR/experiment.toml" in sentence


def test_a_publication_the_preparation_refused_is_relayed_verbatim(tmp_path):
    """Both ends of the chain say the same thing.

    When the preparation tried to publish and could not, it records why.
    Repeating that sentence beats inventing a second explanation here --
    this stage does not know what went wrong and must not guess.
    """

    root = _hrrr_native_root(
        tmp_path / "prep",
        refusal="HrrrBundleError: a prepared case needs at least two "
                "forcing frames")
    with pytest.raises(stage_cli.StageRefusal) as refusal:
        stage_cli.resolve_bundle(root)
    assert ("a prepared case needs at least two forcing frames"
            in str(refusal.value))


def test_a_receipt_that_disagrees_with_the_directory_says_so(tmp_path):
    """A sentence that names a field has to have read it.

    A receipt claiming published authorities in a root that has none is
    a third thing again -- files removed after the fact -- and telling
    that reader "your run published none" would be a statement about
    their receipt that their receipt contradicts.
    """

    root = _hrrr_native_root(
        tmp_path / "prep",
        portable_bundle={"proof_sha256": "0" * 64})
    with pytest.raises(stage_cli.StageRefusal) as refusal:
        stage_cli.resolve_bundle(root)
    sentence = str(refusal.value)
    assert "the receipt and the directory disagree" in sentence
    assert "published none" not in sentence


def test_an_unfinished_route_receipt_keeps_the_refusals_teeth(tmp_path):
    """A receipt that records a failure is still a partial tree."""

    root = _hrrr_native_root(tmp_path / "prep", status="FAIL")
    with pytest.raises(stage_cli.StageRefusal) as refusal:
        stage_cli.resolve_bundle(root)
    sentence = str(refusal.value)
    assert "did not finish" in sentence
    assert "must not be run" in sentence
    assert "FINISHED" not in sentence


def test_a_bundle_from_an_unknown_release_is_refused_naming_its_schema(tmp_path):
    root = tmp_path / "alien"
    root.mkdir()
    (root / "proof.json").write_text(
        json.dumps({"schema": "somebody-elses-proof-v9"}), encoding="utf-8")
    with pytest.raises(stage_cli.StageRefusal) as refusal:
        stage_cli.resolve_bundle(root)
    assert "somebody-elses-proof-v9" in str(refusal.value)


def _mapped_evidence(root: Path, *, schema: str, profile: str | None = None) -> None:
    """What a mapped preparation copies into its evidence directory.

    ``profile`` names a PACKAGED profile whose shipped mapping and
    composition are copied verbatim, which is what makes the tree a
    packaged one; without it the two authorities are a caller's own and
    match no shipped digest, which is what makes it a user's mapping.
    """

    from woof.source_authorities import packaged_authorities

    evidence = root / "source-evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "input-manifest.json").write_text(
        json.dumps({"schema": schema}), encoding="utf-8")
    if profile is None:
        for name in ("mapping.json", "composition.json"):
            (evidence / name).write_text(
                json.dumps({"schema": "rw-wps.mapping.v1", "name": "mine"}),
                encoding="utf-8")
        return
    authorities = packaged_authorities(profile)
    for name, role in (("mapping.json", "mapping"),
                       ("composition.json", "composition")):
        (evidence / name).write_bytes(authorities[role].read_bytes())


def test_a_users_own_mapping_resolves_without_claiming_a_packaged_source(tmp_path):
    from woof.mapped_composition import INPUT_MANIFEST_SCHEMA

    root = _single_domain_bundle(tmp_path / "prepared", source="20crv3")
    _mapped_evidence(root, schema=INPUT_MANIFEST_SCHEMA)
    bundle = stage_cli.resolve_bundle(root)
    assert bundle["source"] == "mapped"
    assert bundle["layout"] == "single"


def test_the_packaged_mapped_route_is_not_caught_by_that_refusal(tmp_path):
    """The other direction: a real 20CRv3 bundle still resolves.

    A refusal that fires on the route it is meant to protect is worse
    than no refusal, so it is tested both ways.
    """

    from woof.prepared_single_domain_forecast import (
        _MAPPED_PACKAGED_PROFILE, _SOURCE_SCHEMA)

    root = _single_domain_bundle(tmp_path / "prepared", source="20crv3")
    _mapped_evidence(root, schema=_SOURCE_SCHEMA["20crv3"],
                     profile=_MAPPED_PACKAGED_PROFILE["20crv3"])
    bundle = stage_cli.resolve_bundle(root)
    assert bundle["source"] == "20crv3"
    assert bundle["layout"] == "single"


def _named_by_bytes(profiles: dict) -> dict:
    """``source -> the source its prepared bytes identify``.

    Model-table rows (one family row, one row per model) share their
    family's packaged profile, so their preparations are byte-identical
    and the bytes name the first registry row holding that profile -- the
    family.  The model itself is the acquisition receipt's fact, not the
    mapping's.  Every distinct profile still names exactly its own row.
    """

    first: dict = {}
    for source, profile in profiles.items():
        first.setdefault(profile, source)
    return {source: first[profile] for source, profile in profiles.items()}


def test_two_packaged_profiles_are_told_apart_by_their_own_authorities(
    tmp_path,
):
    """One proof schema, two shipped sources, resolved by bytes.

    Every packaged profile writes ``gpuwm-mapped-direct-wrf-proof-v1``, so
    the schema cannot say which source prepared a tree.  The mapping and
    composition documents the preparation copied CAN, and exactly: they
    are compared against each shipped profile's pinned digests, which is
    the same binding the forecast runner enforces.  Without this a 20CRv3
    NetCDF bundle would be run through the GRIB2 member certificate and
    refused on a manifest shape it never had.
    """

    from woof.prepared_single_domain_forecast import (
        _MAPPED_PACKAGED_PROFILE, _SOURCE_SCHEMA)

    named = _named_by_bytes(_MAPPED_PACKAGED_PROFILE)
    for source, profile in _MAPPED_PACKAGED_PROFILE.items():
        root = _single_domain_bundle(tmp_path / source, source="20crv3")
        _mapped_evidence(root, schema=_SOURCE_SCHEMA[source], profile=profile)
        assert stage_cli.packaged_source_of(root) == named[source]
        assert stage_cli.resolve_bundle(root)["source"] == named[source]


def _mapped_posted_head(root: Path, *, source: str = "rap",
                        packaged: bool = True) -> dict:
    """The source seam of a digest-bound head, before its manifest exists."""

    import time

    from woof.ingest import boundary_stream
    from woof.prepared_single_domain_forecast import (
        _MAPPED_PACKAGED_PROFILE, _PROOF_SCHEMA, _SOURCE_SCHEMA)

    schema = _SOURCE_SCHEMA[source]
    _mapped_evidence(
        root, schema=schema,
        profile=_MAPPED_PACKAGED_PROFILE[source] if packaged else None)
    (root / "source-evidence" / "input-manifest.json").unlink()
    head = {"schema": boundary_stream.HEAD_SCHEMA, "basis": {
        "proof_head": {"schema": _PROOF_SCHEMA[source]},
        "as_posted": {"input_plan": {"manifest": {"schema": schema}}}}}
    head["head_sha256"] = boundary_stream.head_sha256(head)
    stream = root / "boundary-stream"
    stream.mkdir()
    (stream / "head.json").write_text(json.dumps(head), encoding="utf-8")
    (stream / "producer.json").write_text(
        json.dumps({"updated_epoch": time.time()}), encoding="utf-8")
    return head


def test_a_packaged_as_posted_head_keeps_the_sealed_sources_identity(tmp_path):
    """A head binds its planned manifest before the seal writes its copy."""

    from woof.prepared_single_domain_forecast import (
        _MAPPED_PACKAGED_PROFILE, _SOURCE_SCHEMA)

    config, wps = _authority(tmp_path / "authority")
    named = _named_by_bytes(_MAPPED_PACKAGED_PROFILE)
    for source, profile in _MAPPED_PACKAGED_PROFILE.items():
        root = tmp_path / source
        head = _mapped_posted_head(root, source=source)
        assert not (root / "proof.json").exists()
        assert not (root / "source-evidence" / "input-manifest.json").exists()
        bundle = stage_cli.resolve_head_bundle(root, head["head_sha256"])
        assert bundle["source"] == named[source]
        command = stage_cli.sim_command(
            bundle, experiment_config=config, wps_namelist=wps,
            outdir=tmp_path / "run")
        assert command[command.index("--source") + 1] == named[source]
        # Once its manifest exists, the same authorities still name it.
        _single_domain_bundle(root, source=source)
        _mapped_evidence(root, schema=_SOURCE_SCHEMA[source], profile=profile)
        assert stage_cli.resolve_bundle(root)["source"] == bundle["source"]


def test_a_callers_as_posted_head_keeps_its_generic_mapped_identity(tmp_path):
    root = tmp_path / "custom"
    head = _mapped_posted_head(root, packaged=False)
    bundle = stage_cli.resolve_head_bundle(root, head["head_sha256"])
    assert bundle["source"] == "mapped"


@pytest.mark.parametrize("changed", ["mapping", "composition", "schema"])
def test_a_head_claims_no_packaged_source_when_its_authorities_differ(
        tmp_path, changed):
    from woof.ingest import boundary_stream

    root = tmp_path / "changed"
    head = _mapped_posted_head(root)
    if changed == "schema":
        head["basis"]["as_posted"]["input_plan"]["manifest"]["schema"] = "unknown"
        head["head_sha256"] = boundary_stream.head_sha256(head)
        (root / "boundary-stream" / "head.json").write_text(
            json.dumps(head), encoding="utf-8")
    else:
        evidence = root / "source-evidence" / f"{changed}.json"
        authority = json.loads(evidence.read_bytes())
        if changed == "mapping":
            authority["target"]["target_vertical_levels"] += 1
        else:
            authority["name"] = "custom"
        evidence.write_text(json.dumps(authority), encoding="utf-8")
    bundle = stage_cli.resolve_head_bundle(root, head["head_sha256"])
    assert bundle["source"] == "mapped"


def test_a_changed_head_plan_is_refused_before_its_source_is_resolved(tmp_path):
    root = tmp_path / "changed-plan"
    head = _mapped_posted_head(root)
    head["basis"]["as_posted"]["input_plan"]["manifest"]["schema"] = "unknown"
    (root / "boundary-stream" / "head.json").write_text(
        json.dumps(head), encoding="utf-8")
    with pytest.raises(stage_cli.StageRefusal, match="fails its own head digest"):
        stage_cli.resolve_head_bundle(root, head["head_sha256"])


def test_an_earlier_release_s_preparation_keeps_its_packaged_source(tmp_path):
    """A166: an admission-only mapping change does not relabel a bundle.

    A159 added ``accept_boundary_interval_multiples`` to ten packaged
    mappings.  A tree prepared before it carries the old document, whose
    digest no longer matches the pin; the door named it ``mapped``, so a
    hrrr-prs forecast ran and reported as a caller's own mapping.  The
    old document resolves to its profile; one that decodes differently
    still does not.
    """

    from woof.prepared_single_domain_forecast import (
        _MAPPED_PACKAGED_PROFILE, _SOURCE_SCHEMA)
    from woof.source_authorities import BOUNDARY_MULTIPLES_KEY

    named = _named_by_bytes(_MAPPED_PACKAGED_PROFILE)
    for source, profile in _MAPPED_PACKAGED_PROFILE.items():
        root = _single_domain_bundle(tmp_path / source, source="20crv3")
        _mapped_evidence(root, schema=_SOURCE_SCHEMA[source], profile=profile)
        mapping = root / "source-evidence" / "mapping.json"
        document = json.loads(mapping.read_bytes())
        target = document["target"]
        if target.pop(BOUNDARY_MULTIPLES_KEY, None) is None:
            target[BOUNDARY_MULTIPLES_KEY] = True
        mapping.write_text(json.dumps(document, indent=2), encoding="utf-8")
        assert stage_cli.packaged_source_of(root) == named[source]
        assert stage_cli.resolve_bundle(root)["source"] == named[source]
        target["target_vertical_levels"] = int(
            target.get("target_vertical_levels") or 0) + 1
        mapping.write_text(json.dumps(document, indent=2), encoding="utf-8")
        assert stage_cli.packaged_source_of(root) is None
        assert stage_cli.resolve_bundle(root)["source"] == "mapped"


def test_runner_single_requires_a_bound_root_artifact_receipt(tmp_path):
    """A hierarchy without a recorded root cache cannot relay its identity."""

    tree = _tree_bundle(tmp_path / "tree")
    config, wps = _authority(tmp_path / "authority")
    with pytest.raises(stage_cli.StageRefusal) as refusal:
        stage_cli.sim_command(
            stage_cli.resolve_bundle(tree), experiment_config=config,
            wps_namelist=wps, outdir=tmp_path / "run", runner="single")
    assert "no unambiguous READY d01 artifact receipt" in str(refusal.value)


# ---------------------------------------------------------------------------
# 4. `go` is a composition of the stages, not a second route
# ---------------------------------------------------------------------------

def _go_plan(tmp_path, *, domains: int = 1):
    """The plan ``go`` builds, without running any of its stages."""

    out = tmp_path / "cfg.toml"
    argv = ["domain", "--point=35.3,-97.5", "--card", "24gb",
            "--ladder", "12" if domains == 1 else "12-3-1",
            "--source", "gfs", "--cycle", "2026-07-29T18", "--hours", "6",
            "--out", str(out),
            "--physics-profile", "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"]
    assert cli_main(argv) == 0
    return go_cli.plan_from_config(out, outdir=tmp_path / "go")


def _without_progress(command: list[str]) -> list[str]:
    """``command`` with one ``--progress-format VALUE`` pair removed.

    Refuses rather than returning the command unchanged when the pair is
    absent: this helper exists to subtract a difference that is supposed
    to be there, and silently subtracting nothing would turn the
    "nothing else differs" assertion into a tautology.
    """

    try:
        i = command.index("--progress-format")
    except ValueError:                                  # pragma: no cover
        raise AssertionError(
            "expected `woof go` to pass --progress-format; it did not, so "
            "the equivalence gate below would be comparing go against "
            "itself") from None
    return command[:i] + command[i + 2:]


def test_gos_forecast_stage_is_exactly_gpuwm_sim_single_domain(tmp_path):
    """THE equivalence gate for the single-domain arm."""

    plan = _go_plan(tmp_path)
    prepared = _single_domain_bundle(Path(plan["prepared"]))
    config, wps = _authority(Path(plan["authority"]))

    from_go = go_cli.forecast_command(plan, go_cli.proof_digests(prepared))

    def seam(**extra):
        return stage_cli.sim_command(
            stage_cli.resolve_bundle(prepared),
            experiment_config=config, wps_namelist=wps, outdir=plan["run"],
            physics_profile=plan["profile"], **extra)

    assert from_go == seam(progress_format="jsonl"), (
        "`woof go`'s forecast stage and `woof sim` compose different "
        "commands, so go is a second implementation rather than a "
        "composition of the stages")
    # ...and the progress transport is the ONLY thing go is allowed to
    # differ on.  Without this second assertion the parameter above
    # would be a hole: any future divergence could be waved through by
    # passing one more keyword.
    assert _without_progress(from_go) == seam(), (
        "`woof go` differs from `woof sim` on something other than "
        "--progress-format")


def test_gos_forecast_stage_is_exactly_gpuwm_sim_tree(tmp_path):
    """THE equivalence gate for the domain-tree arm."""

    plan = _go_plan(tmp_path, domains=3)
    prepared = _tree_bundle(Path(plan["prepared"]))
    config, _wps = _authority(Path(plan["authority"]))

    from_go = go_cli.tree_forecast_command(plan)

    def seam(**extra):
        return stage_cli.sim_command(
            stage_cli.resolve_bundle(prepared), experiment_config=config,
            wps_namelist=None, outdir=plan["run"], **extra)

    assert from_go == seam(progress_format="jsonl"), (
        "`woof go`'s tree forecast stage and `woof sim` compose "
        "different commands")
    assert _without_progress(from_go) == seam(), (
        "`woof go`'s tree stage differs from `woof sim` on something "
        "other than --progress-format")


def test_go_still_prints_its_whole_chain_and_runs_nothing(tmp_path, capsys, monkeypatch):
    """``go`` is the average user's front door and must not regress."""
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: tmp_path / "bridge")

    out = tmp_path / "cfg.toml"
    assert cli_main([
        "domain", "--point=35.3,-97.5", "--card", "24gb", "--ladder", "12",
        "--source", "gfs", "--cycle", "2026-07-29T18", "--hours", "6",
        "--out", str(out), "--physics-profile",
        "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"]) == 0
    capsys.readouterr()
    assert cli_main(["go", str(out), "--outdir", str(tmp_path / "go"),
                     "--dry-run"]) == 0
    printed = capsys.readouterr().out
    for label in ("1. authority", "2. fetch", "3. manifest", "4. prepare",
                  "5. forecast", "6. render"):
        assert label in printed, f"woof go no longer prints {label}"
    assert not (tmp_path / "go").exists()


# ---------------------------------------------------------------------------
# The negative property: "no data pulling", asserted rather than promised
# ---------------------------------------------------------------------------

_NO_FETCH_PROBE = """
import json, sys
import woof.stage_cli as stage
from types import SimpleNamespace
args = SimpleNamespace(
    prepared_root=sys.argv[1], experiment_config=sys.argv[2],
    wps_namelist=sys.argv[3], outdir=sys.argv[4], physics_profile=None,
    io_mode="history", runner="auto", print_command=True, explain=False)
rc = stage.sim_main(args)
print(json.dumps({
    "rc": rc,
    "fetch": "woof.fetch" in sys.modules,
    "urllib_request": "urllib.request" in sys.modules,
    "http_client": "http.client" in sys.modules,
}), file=sys.stderr)
"""


def test_the_simulation_stage_never_imports_the_fetch_machinery(tmp_path):
    """"no data pulling", as an assertion about a fresh interpreter.

    The stage reaching dispatch without :mod:`woof.fetch` in
    ``sys.modules`` is a stronger statement than any prose promise:
    that module is the forcing-data download machinery, and the
    simulation stage must not be able to reach it even by accident.
    ``woof go``'s relay helper imports it for its ``sha256_file``,
    which is exactly why this seam carries its own digest helper
    instead of borrowing that one.

    ``urllib.request`` is deliberately NOT asserted absent, and the
    accuracy matters: :mod:`woof.table_assets`, which the single-domain
    runner imports for its Thompson-table check, imports it at module
    scope.  That is a pre-existing property of the runner, it predates
    this seam, and claiming otherwise here would be a test that passes
    by describing something else.  What the network claim rests on is
    the behavioural test below.

    The probe drives :mod:`woof.stage_cli` rather than
    :mod:`woof.cli`, because the aggregate CLI imports every
    subcommand's registrar at module scope and so pulls the download
    stack in for ``woof version`` too.  That is the aggregator's
    property, not this stage's; the stage is the unit a third party
    imports, and it is clean.
    """

    root = _single_domain_bundle(tmp_path / "prepared")
    config, wps = _authority(tmp_path / "authority")
    result = subprocess.run(
        [sys.executable, "-c", _NO_FETCH_PROBE, str(root), str(config),
         str(wps), str(tmp_path / "run")],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stderr.strip().splitlines()[-1])
    assert report["rc"] == 0
    assert report["fetch"] is False, (
        "woof sim imported woof.fetch, so the simulation stage still "
        "drags the download machinery in")


_NO_SOCKET_PROBE = """
import sys

# Import the front door FIRST, so module-scope imports (ssl subclasses
# socket.socket, and would break under a poisoned class) all complete.
# What is being tested is whether the COMMAND touches the network, not
# whether importing Python's stdlib does.
from woof.cli import main

import socket


def _refuse(*args, **kwargs):
    raise RuntimeError("the simulation stage reached for the network")


socket.socket.connect = _refuse
socket.socket.connect_ex = _refuse
socket.create_connection = _refuse
socket.getaddrinfo = _refuse

raise SystemExit(main([
    "sim", sys.argv[1], "--experiment-config", sys.argv[2],
    "--wps-namelist", sys.argv[3], "--outdir", sys.argv[4],
    "--print-command"]))
"""


def test_the_simulation_stage_opens_no_socket_on_the_real_cli_route(tmp_path):
    """The network is not merely unused, it is unavailable.

    Every way of reaching a remote host raises in this child, so any
    implicit lookup, telemetry call or lazy download on the ``woof
    sim`` path -- through the real front door, not the library -- ends
    the process nonzero.  Exit 0 is the proof, and it is the reporter's
    "no data pulling" stated in the one form a script can check.
    """

    root = _single_domain_bundle(tmp_path / "prepared")
    config, wps = _authority(tmp_path / "authority")
    result = subprocess.run(
        [sys.executable, "-c", _NO_SOCKET_PROBE, str(root), str(config),
         str(wps), str(tmp_path / "run")],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        "woof sim did not survive an unavailable network:\n"
        + result.stderr)
    assert stage_cli.SINGLE_DOMAIN_RUNNER in result.stdout


# ---------------------------------------------------------------------------
# The contract document: a third party must not have to read our source
# ---------------------------------------------------------------------------

_CONTRACT = (Path(__file__).resolve().parent.parent
             / "docs" / "public" / "PIPELINE-STAGES.md")


def test_the_stage_contract_document_ships():
    """A seam nobody can find the boundary of is not a seam."""

    assert _CONTRACT.is_file(), (
        f"{_CONTRACT.name} is missing, so each stage's input/output "
        "contract exists only in this repository's source")
    text = _CONTRACT.read_text(encoding="utf-8")
    for stage in ("woof prep", "woof sim", "woof render"):
        assert stage in text, f"{stage} is undocumented in {_CONTRACT.name}"


def test_every_command_the_contract_names_is_a_real_subcommand():
    """Doc drift, caught as a test rather than as a user's dead end."""

    import re

    text = _CONTRACT.read_text(encoding="utf-8")
    parser = build_parser()
    known = set(parser._subparsers._group_actions[0].choices)  # noqa: SLF001
    named = set(re.findall(r"`woof ([a-z][a-z-]*)", text))
    unknown = named - known
    assert not unknown, (
        f"{_CONTRACT.name} tells a reader to run {sorted(unknown)}, which "
        "this CLI does not have")


def test_the_contract_continues_both_prepared_layouts_through_the_real_parser(tmp_path):
    """The contract said the single prepared runner had no checkpoint
    restore and `woof go` no checkpoint operand; both exist, so a reader
    of the contract believed a single-domain forecast could not resume.
    Every continuation command the section documents must reach its
    runner with --restart intact, and the single-domain one must exist."""

    import re

    text = _CONTRACT.read_text(encoding="utf-8")
    heading = "### Continuing a prepared forecast"
    assert heading in text, f"{_CONTRACT.name} has no prepared continuation section"
    section = text.split(heading, 1)[1].split("\n### ", 1)[0]
    sims = [" ".join(block.replace("\\\n", " ").split())
            for block in re.findall(r"```sh\n(.*?)```", section, flags=re.S)
            if block.lstrip().startswith("woof sim ")]
    single, tree = _single_domain_bundle(tmp_path / "single"), _tree_bundle(tmp_path / "tree")
    config, wps = _authority(tmp_path / "authority")
    checkpoint = tmp_path / "previous" / "gpuwmrst_d01_2026-01-01_03:00:00.npz"
    layouts = set()
    for command in sims:
        tokens = shlex.split(command)[2:]
        assert "--restart" in tokens, command
        layout = "single" if "--wps-namelist" in tokens else "tree"
        layouts.add(layout)
        values = {"--experiment-config": str(config), "--wps-namelist": str(wps),
                  "--restart": str(checkpoint), "--outdir": str(tmp_path / layout / "out")}
        argv = [str(single if layout == "single" else tree)]
        for flag in tokens[1::2]:
            argv += [flag, values[flag]]
        result = subprocess.run(
            [sys.executable, "-m", "woof.cli", "sim", *argv, "--print-command"],
            capture_output=True, text=True, check=False)
        assert result.returncode == 0, (command, result.stderr)
        printed = host_shell_words(result.stdout.strip().splitlines()[-1])
        runner = (stage_cli.SINGLE_DOMAIN_RUNNER if layout == "single"
                  else stage_cli.TREE_RUNNER)
        assert printed[1:3] == ["-m", runner], printed
        assert printed[printed.index("--restart") + 1] == str(checkpoint)
    assert layouts == {"single", "tree"}, sims
    go = next(line for line in section.split("`")
              if line.startswith("woof go ") and "--restart" in line)
    parsed = build_parser().parse_args(go.split()[1:])
    assert parsed.prepared_root is not None and parsed.restart is not None


def test_the_readme_points_at_the_contract():
    readme = (Path(__file__).resolve().parent.parent / "README.md"
              ).read_text(encoding="utf-8")
    assert "docs/public/PIPELINE-STAGES.md" in readme


def test_the_printed_command_is_a_line_a_shell_can_run(tmp_path):
    """``--print-command`` output round-trips through the shell it is
    printed for: ``shlex`` for a POSIX line, PowerShell's own parser for
    the Windows one."""

    root = _single_domain_bundle(tmp_path / "prepared")
    config, wps = _authority(tmp_path / "authority")
    result = subprocess.run(
        [sys.executable, "-m", "woof.cli", "sim", str(root),
         "--experiment-config", str(config), "--wps-namelist", str(wps),
         "--outdir", str(tmp_path / "run"), "--print-command"],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    tokens = host_shell_words(result.stdout.strip())
    assert tokens[1:3] == ["-m", stage_cli.SINGLE_DOMAIN_RUNNER]
    assert "--outdir" in tokens
