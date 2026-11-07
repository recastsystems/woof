"""A GFS domain tree prepares chained, like a mapped one (A136 L7b).

CPU orchestration gates, on the fixtures of test_gfs_initial_perturbation:
the real GFS door, experiment loader, manifest verifier, prepared-tree
writer and head digest run; decoding, array numerics, the hierarchy's child
preparation and the one-shot artifact writer are doubles.  The arrays
themselves are held byte-equal between the chained and one-shot trees by
the real-data identity proof on test host 4.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof import gfs_direct, stage_cli
from woof.experiment import load_experiment
from woof.ingest.boundary_stream import (
    HIERARCHY_HEAD_DIRNAME, LAYOUT_DOMAIN_TREE, SEAL_ONLY_PROOF_KEYS,
    read_head, segment_marker_path)

from test_gfs_initial_perturbation import (
    _config, _cpu_preparation, _inputs, _with_bubble)


class _RecordingCacheStream:
    """The prepared-cache writer, recorded: head, segments, seal."""

    def __init__(self, directory, *, identity, **_kwargs):
        self.directory = Path(directory)
        self.identity = identity

    def move(self, directory):
        self.directory = Path(directory)

    def write_head(self, **kwargs):
        self.directory.mkdir(parents=True)
        return {"identity": {}, "metadata": dict(kwargs["metadata"] or {}),
                "arrays": {}, "payload_bytes": 0, "lbc": kwargs["lbc"],
                "setup_core_fingerprint": "0" * 64}

    def write_segment(self, index, interval):
        return {"index": index,
                "start_seconds": float(interval.start_seconds),
                "end_seconds": float(interval.end_seconds),
                "fields": sorted(interval.fields), "arrays": {},
                "payload_bytes": 0, "prefix": {}}

    def seal(self):
        (self.directory / "header.json").write_text(
            json.dumps({"identity": self.identity}, sort_keys=True,
                       default=str), encoding="utf-8")
        return {"schema": "gpuwm-prepared-cache-v1", "status": "BUILT",
                "content_sha256": "c" * 64, "array_count": 1,
                "payload_bytes": 1}


def _chained_tree_doubles(monkeypatch, exp):
    """The chained arm's hierarchy seams, recorded in ``seen``."""

    import woof.ingest.prepared_cache as prepared_cache_module

    children = tuple(f"d{int(domain.grid_id):02d} start"
                     for domain in exp.domains[1:])
    seen = {"children_written": [], "released": [], "reread": [],
            "children": children}
    tree_head = SimpleNamespace(
        child_results=children,
        forcing_identity={"forcing_hours": (0, 3)},
        static_receipt={}, source_coverage_receipt={}, topology_receipt={},
        statics_corridor_receipt=None,
        bound_source_identity=lambda identity: {
            **dict(identity), "hierarchy_bound": True})

    def head(**kwargs):
        seen["head"] = kwargs
        seen["head_root_state"] = kwargs["root_initial_result"].state.thp.copy()
        return tree_head

    def static_files(directory, *, domain, grid, static_fields):
        (Path(directory) / "native-static.npz").write_bytes(b"static")
        (Path(directory) / "geometry-receipt.json").write_text(
            "{}", encoding="utf-8")
        return {"sha256": "b" * 64}, {}

    def child_artifacts(domain_root, *, exp, child_results, **kwargs):
        seen["children_written"].append((tuple(child_results), kwargs))
        builds = []
        for domain in exp.domains[1:]:
            receipt = {"grid_id": int(domain.grid_id), "artifacts": {
                "prepared_cache": {"payload_bytes": 64,
                                   "content_sha256": "d" * 64}}}
            folder = Path(domain_root) / f"d{int(domain.grid_id):02d}"
            folder.mkdir(parents=True)
            (folder / "receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8")
            builds.append(SimpleNamespace(receipt=receipt))
        return builds

    def binding(**kwargs):
        seen["binding"] = kwargs
        return SimpleNamespace(identity={"source": "gfs-chained-tree-test"},
                               metadata={"source_adapter": "gfs"})

    class StartStates:
        @classmethod
        def release(cls, **kwargs):
            seen["released"].append(kwargs)
            return cls()

        def reread(self, root, **kwargs):
            seen["reread"].append((Path(root), kwargs))
            state = SimpleNamespace(lateral_boundaries="whole series")
            return (SimpleNamespace(state=state, reread=True),
                    SimpleNamespace(fields={}), "whole series",
                    ("children from the head",))

        def require_sealed_is_head(self, receipt, *, root_content_sha256):
            seen["sealed_is_head"] = (receipt, root_content_sha256)

    def seal(sealed_head, **kwargs):
        seen["seal"] = (sealed_head, kwargs)
        Path(kwargs["artifact_output"]).mkdir(parents=True)
        return SimpleNamespace(
            hierarchy=SimpleNamespace(
                moisture_floor_receipts={},
                artifacts=SimpleNamespace(receipt={"fixture": True}),
                wrf_manifest={"status": "NOT_REQUESTED"},
                timings_seconds={"initialize_children": 0.1}),
            # The real seal copies the head's corridor set and returns
            # its receipt.
            statics_corridor_receipt=sealed_head.statics_corridor_receipt)

    monkeypatch.setattr(
        prepared_cache_module, "PreparedCacheStream", _RecordingCacheStream)
    monkeypatch.setattr(gfs_direct, "_canonical_surface", lambda soil: {})
    monkeypatch.setattr(
        gfs_direct, "prepare_regular_source_hierarchy_head", head)
    monkeypatch.setattr(gfs_direct, "write_domain_static_files", static_files)
    monkeypatch.setattr(
        gfs_direct, "write_child_domain_artifacts", child_artifacts)
    monkeypatch.setattr(gfs_direct, "root_domain_artifact_binding", binding)
    monkeypatch.setattr(gfs_direct, "TreeStartStates", StartStates)
    monkeypatch.setattr(gfs_direct, "seal_regular_source_hierarchy", seal)
    monkeypatch.setattr(gfs_direct, "hierarchy_moisture_floor_receipts",
                        lambda *operands: {})
    return seen, tree_head


def _without_seal_keys(proof):
    return {key: value for key, value in proof.items()
            if key not in SEAL_ONLY_PROOF_KEYS}


@pytest.mark.parametrize("domains", [2, 3])
@pytest.mark.parametrize("bubble", [False, True])
def test_a_gfs_domain_tree_chains_through_prepare_gfs_wrf(
        tmp_path, monkeypatch, domains, bubble):
    """The GFS door publishes a tree's head at its start time.

    The breakage this prevents: every nested `woof go` from GFS waited for
    its whole preparation before the forecast started (the retired
    ``gfs_domain_tree`` sealed reason), while a mapped tree started on its
    head.  The start time is built first, the children go into the head,
    one segment per root interval follows, and the seal writes the one-shot
    tree from the states the head holds.  The chained proof is the
    one-shot proof plus the stream's own keys, so a forecast bound to the
    head and one bound to the sealed tree read the same document.
    """

    config = _config(tmp_path, domains=domains)
    if bubble:
        config = _with_bubble(config)
    exp = load_experiment(config)

    # The one-shot arm, chaining off: the reference document.
    with monkeypatch.context() as patch:
        patch.setenv("WOOF_CHAINED_PREP", "0")
        oneshot_captures = _cpu_preparation(patch, exp)
        oneshot = gfs_direct.prepare_gfs_wrf(
            **_inputs(tmp_path, config, "oneshot"))
    assert len(oneshot_captures["hierarchies"]) == 1

    monkeypatch.setenv("WOOF_CHAINED_PREP", "1")
    # The host admission is priced by its own tests (test_boundary_stream);
    # read here it took the machine's real free RAM and refused the chain
    # on a loaded 24-core box with 1.2 GiB available.
    from woof.ingest import boundary_stream
    monkeypatch.setattr(boundary_stream, "_host_available",
                        lambda: 64 * 1024 ** 3)
    captures = _cpu_preparation(monkeypatch, exp)
    seen, tree_head = _chained_tree_doubles(monkeypatch, exp)
    arguments = _inputs(tmp_path, config, "chained")
    proof = gfs_direct.prepare_gfs_wrf(**arguments)
    root = arguments["output_root"]

    # The one-shot hierarchy call is the unchained route's alone.
    assert captures["hierarchies"] == []
    # START FIRST: the start time is built before the later one, and it is
    # the state the children and the head are prepared from.
    times = [valid for valid, _payload in captures["root_initializations"]]
    assert times == sorted(times) and len(times) == 2
    start_payload = captures["root_initializations"][0][1]
    np.testing.assert_array_equal(seen["head_root_state"], start_payload)
    # The head sees the whole series, so a delayed nest's start time is
    # there when its child is prepared.
    assert [s.valid_time for s in seen["head"]["snapshots"]] == times
    assert seen["head"]["source_name"] == "GFS"
    assert seen["head"]["forcing_hours"] == [0, 3]

    published = read_head(root)
    assert published["decision"]["chained"] is True
    assert published["layout"] == LAYOUT_DOMAIN_TREE
    assert published["domains"] == [
        f"d{int(domain.grid_id):02d}" for domain in exp.domains]
    assert published["basis"]["cache"]["directory"] == (
        f"{HIERARCHY_HEAD_DIRNAME}/domains/d01/prepared-cache")
    receipts = published["basis"]["tree"]["children_receipts"]
    assert sorted(receipts) == published["domains"][1:]
    for label, digest in receipts.items():
        child = root / HIERARCHY_HEAD_DIRNAME / "domains" / label
        assert hashlib.sha256(
            (child / "receipt.json").read_bytes()).hexdigest() == digest
    # The children were prepared from the start time into the head, bound
    # to the hierarchy's own source identity.
    ((written, child_kwargs),) = seen["children_written"]
    assert written == seen["children"]
    assert child_kwargs["source_identity"]["hierarchy_bound"] is True
    assert seen["binding"]["source_identity"]["hierarchy_bound"] is True
    # Every start state is released at the head and re-read at the seal.
    (released,) = seen["released"]
    assert released["child_results"] == seen["children"]
    assert set(released["child_content_sha256"]) == set(receipts)
    ((reread_root, reread),) = seen["reread"]
    assert reread_root == root
    assert reread["root_identity"] == {"source": "gfs-chained-tree-test"}
    assert reread["root_content_sha256"] == "c" * 64

    # One segment per root interval, then the seal on the whole series.
    assert segment_marker_path(root, 0).is_file()
    assert not segment_marker_path(root, 1).exists()
    sealed_head, sealed = seen["seal"]
    assert sealed_head is tree_head
    # The seal's children are the ones re-read from the head.
    assert sealed_head.child_results == ("children from the head",)
    assert sealed["root_boundaries"] == "whole series"
    assert sealed["root_initial_result"].reread is True
    assert sealed["artifact_output"] == root / "hierarchy-artifacts"
    assert sealed["wrf_output"] == root / "wrf-native-input"
    assert sealed["stock_wrf_export"] == "optional"
    # The unbound identity: the seal binds it through the head.
    assert "hierarchy_bound" not in sealed["source_identity"]
    assert seen["sealed_is_head"] == ({"fixture": True}, "c" * 64)

    # The proof on disk is the one returned, and it names its head.
    assert json.loads((root / "proof.json").read_text()) == proof
    assert proof["boundary_stream"] == {
        "head_sha256": published["head_sha256"]}
    assert proof["schema"] == gfs_direct.HIERARCHY_PROOF_SCHEMA
    # The two arms write one document: the head's proof equals the
    # one-shot proof less the seal-only keys, the deferred bubble included.
    assert published["basis"]["proof_head"] == json.loads(json.dumps(
        _without_seal_keys(oneshot)))
    assert _without_seal_keys(proof) == _without_seal_keys(oneshot)
    assert set(proof) - set(oneshot) == {"boundary_stream"}
    assert ("initial_perturbation" in proof) is bubble

    # The forecast doors bind this head as a tree, and the sealed tree as
    # the same tree.
    bundle = stage_cli.resolve_head_bundle(root, published["head_sha256"])
    assert bundle["layout"] == "tree" and bundle["domains"] == domains
    command = stage_cli.sim_command(
        bundle, experiment_config=config, wps_namelist=None,
        outdir=tmp_path / "forecast-not-started")
    assert command[2] == "woof.prepared_domain_tree_forecast"
    assert command[command.index("--prepared-head-sha256") + 1] == (
        published["head_sha256"])
    assert stage_cli.resolve_bundle(root)["layout"] == "tree"


def test_a_failure_after_the_head_marks_the_published_tree_failed(
        tmp_path, monkeypatch):
    """A seal that fails leaves the published head marked failed.

    The breakage this prevents: a forecast already stepping on the head
    would wait at its next seam for a preparation that had died.
    """

    config = _config(tmp_path, domains=2)
    exp = load_experiment(config)
    monkeypatch.setenv("WOOF_CHAINED_PREP", "1")
    _cpu_preparation(monkeypatch, exp)
    _chained_tree_doubles(monkeypatch, exp)

    def broken_seal(*_args, **_kwargs):
        raise RuntimeError("the one-shot tree writer failed")

    monkeypatch.setattr(gfs_direct, "seal_regular_source_hierarchy",
                        broken_seal)
    arguments = _inputs(tmp_path, config, "failed")
    with pytest.raises(RuntimeError, match="one-shot tree writer failed"):
        gfs_direct.prepare_gfs_wrf(**arguments)
    root = arguments["output_root"]
    failed = json.loads(
        (root / "boundary-stream" / "failed.json").read_text())
    assert "one-shot tree writer failed" in failed["reason"]
    assert not (root / "proof.json").exists()
    # The staging directory was the published head; nothing is left beside it.
    assert not list(root.parent.glob(".d-*"))


def test_go_says_no_sealed_reason_for_a_gfs_tree():
    """go's GFS arm hands a tree to run_chained like a single domain.

    The retired ``gfs_domain_tree`` row told every nested GFS run that its
    forecast would start after preparation; the preparation now says so
    itself when it declines (admission, or chaining off).
    """

    from woof.ingest import boundary_stream

    source = Path(gfs_direct.__file__).with_name("go_cli.py").read_text(
        encoding="utf-8")
    assert "gfs_domain_tree" not in source
    assert "gfs_domain_tree" not in boundary_stream.SEALED_REASONS


def test_a_moving_gfs_tree_binds_its_statics_corridor_at_the_head(
        tmp_path, monkeypatch):
    """A136 L7d: a GFS tree whose nest moves carries its corridor at the head.

    The breakage this prevents: a moving or cyclone GFS tree prepared
    chained, but its forecast still waited for the seal (10.9 s measured
    on L7b's moving tree), because the statics corridor the nest re-grounds
    over was written only there.  The head is asked to build the corridor
    into hierarchy-head/, its proof binds the set receipt, the seal copies
    it from there, and the sealed proof equals the one-shot proof less the
    stream's own key.
    """

    corridor = {"schema": "gpuwm-statics-corridor-set-v1",
                "status": "READY", "domains": {"d02": {"cache": {
                    "path": "d02.npz", "bytes": 1, "sha256": "e" * 64}}}}
    config = _config(tmp_path, domains=2)
    exp = load_experiment(config)

    with monkeypatch.context() as patch:
        patch.setenv("WOOF_CHAINED_PREP", "0")
        _cpu_preparation(patch, exp)
        oneshot_hierarchy = (
            gfs_direct.initialize_and_export_regular_source_hierarchy)

        def with_corridor(**kwargs):
            assert kwargs["statics_corridor"] == "all"
            result = oneshot_hierarchy(**kwargs)
            result.statics_corridor_receipt = corridor
            return result

        patch.setattr(
            gfs_direct, "initialize_and_export_regular_source_hierarchy",
            with_corridor)
        oneshot = gfs_direct.prepare_gfs_wrf(
            **_inputs(tmp_path, config, "oneshot"), statics_corridor="all")
    assert oneshot["statics_corridor"] == corridor

    monkeypatch.setenv("WOOF_CHAINED_PREP", "1")
    _cpu_preparation(monkeypatch, exp)
    seen, tree_head = _chained_tree_doubles(monkeypatch, exp)
    tree_head.statics_corridor_receipt = corridor
    arguments = _inputs(tmp_path, config, "chained")
    proof = gfs_direct.prepare_gfs_wrf(**arguments, statics_corridor="all")
    root = arguments["output_root"]

    # The head is built with its corridor, into the head's folder.
    assert seen["head"]["statics_corridor"] == "all"
    assert seen["head"]["head_artifacts"].name == HIERARCHY_HEAD_DIRNAME
    published = read_head(root)
    assert published["basis"]["proof_head"]["statics_corridor"] == corridor
    assert published["basis"]["proof_head"] == json.loads(json.dumps(
        _without_seal_keys(oneshot)))
    # The seal copies the head's set from the published head.
    _sealed_head, sealed = seen["seal"]
    assert sealed["head_artifacts"] == root / HIERARCHY_HEAD_DIRNAME
    assert proof["statics_corridor"] == corridor
    assert _without_seal_keys(proof) == _without_seal_keys(oneshot)
    assert set(proof) - set(oneshot) == {"boundary_stream"}


def test_go_prepares_a_gfs_tree_beside_its_as_posted_fetch():
    """A136 L3 (v): a GFS tree's preparation starts on its start leads.

    go ran every tree's preparation after its whole window was fetched,
    because the tree preparation refused to run as posted; the tree now
    binds the input plan at its head, so the tree takes the single
    domain's chain, and --whole-cycle keeps the old order for both.
    """

    from woof import go_cli

    for domains in (1, 2, 3):
        plan = {"as_posted": None, "domains": domains, "source": "gfs"}
        assert go_cli.posts_beside_preparation(plan)
        assert not go_cli.posts_beside_preparation({**plan, "as_posted": False})
