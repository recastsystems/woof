"""Every publication phase uses one exact eight-file distribution contract."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import re

import pytest
import yaml

# (Recast WOOF) the engine's own PyPI publication: four engine wheels and the companion pairs, named by the
# engine's distributions. Recast WOOF publishes its five files through its own release workflows, and the
# rename gives this tool a distribution ("woof") that is not published, so the contract does not apply here.
pytest.skip("the engine's four-wheel publication contract; Recast WOOF publishes through its own workflows",
            allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
VERSION = "9.8.7"
TAG = "v" + VERSION
COMMIT = "a" * 40
REPOSITORY = "example-org/example-repo"


@pytest.fixture(scope="module")
def publication():
    spec = importlib.util.spec_from_file_location("dist_shape_publication", ROOT / "tools/promote_prepared_release.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(name):
    return {"filename": name, "bytes": len(name), "sha256": hashlib.sha256(name.encode()).hexdigest()}


@pytest.fixture
def manifest():
    names = [f"gpuwm-{VERSION}-py3-none-any.whl", f"gpuwm-{VERSION}-py3-none-manylinux_2_28_x86_64.whl",
             f"gpuwm-{VERSION}-py3-none-win_amd64.whl", f"gpuwm-{VERSION}.tar.gz",
             f"woof_data-{VERSION}-py3-none-any.whl", f"woof_data-{VERSION}.tar.gz",
             f"gpuwm_kernels-{VERSION}-py3-none-any.whl", f"gpuwm_kernels-{VERSION}.tar.gz",
             f"gpuwm_kernels_startup-{VERSION}-py3-none-any.whl", f"gpuwm_kernels_startup-{VERSION}.tar.gz"]
    carried = [row("gpuwm-carried-physics-v9.8.6-v9.8.7.json"),
               row("gpuwm-carried-physics-v9.8.6-v9.8.7.verification.json")]
    return {"schema": "arwen.publication-assets.v1", "engine_source_revision": COMMIT,
            "desktop_source_revision": "d" * 40,
            "carried_physics": {"schema": "arwen.carried-release-asset.v1", "release_id": "e" * 64,
                                "release": carried[0], "verification": carried[1]},
            "github": {"repository": REPOSITORY, "target_version": TAG, "assets": [
                row(f"gpuwm-bridges-{TAG}-linux-x86_64.zip"), row(f"gpuwm-bridges-{TAG}-win-x86_64.zip"),
                row("bridge-bundle-manifest.json"), *carried]}, "pypi": {"artifacts": [row(name) for name in names]}}


def test_one_manifest_defines_the_engine_four_and_companion_pair(publication, manifest):
    plan = publication.load_manifest(manifest, TAG, COMMIT, REPOSITORY)
    assert len(plan["pypi"]["woof"]) == 4
    assert len(plan["pypi"]["recast-woof-data"]) == 2
    assert len(plan["pypi"]["gpuwm-kernels"]) == 2
    assert len(plan["pypi"]["gpuwm-kernels-startup"]) == 2
    assert {row["filename"] for rows in plan["pypi"].values() for row in rows} == {
        row["filename"] for row in manifest["pypi"]["artifacts"]}


@pytest.mark.parametrize("missing", range(10))
def test_every_distribution_is_required_at_capture(publication, manifest, missing):
    manifest["pypi"]["artifacts"].pop(missing)
    with pytest.raises(publication.PublicationError, match="exactly"):
        publication.load_manifest(manifest, TAG, COMMIT, REPOSITORY)


@pytest.mark.parametrize("filename", ["gpuwm-9.8.7-py3-none-linux_x86_64.whl", "gpuwm-9.8.7-py3-none-manylinux_2_28_aarch64.whl",
                                      "gpuwm-9.8.8-py3-none-any.whl", "gpuwm-data-9.8.7.tar.gz"])
def test_an_extra_or_substituted_platform_is_not_silently_published(publication, manifest, filename):
    manifest["pypi"]["artifacts"].append(row(filename))
    with pytest.raises(publication.PublicationError, match="exactly"):
        publication.load_manifest(manifest, TAG, COMMIT, REPOSITORY)


def test_native_and_universal_phases_are_disjoint_and_cover_the_engine(publication, manifest):
    plan = publication.load_manifest(manifest, TAG, COMMIT, REPOSITORY)
    rows = plan["pypi"]["woof"]
    native = {row["filename"] for row in publication.phase_rows(rows, "woof", "native")}
    pure = {row["filename"] for row in publication.phase_rows(rows, "woof", "pure")}
    assert native == {f"gpuwm-{VERSION}-py3-none-manylinux_2_28_x86_64.whl", f"gpuwm-{VERSION}-py3-none-win_amd64.whl"}
    assert pure == {f"gpuwm-{VERSION}-py3-none-any.whl", f"gpuwm-{VERSION}.tar.gz"}
    assert not native & pure
    assert native | pure == {row["filename"] for row in rows}
    assert publication.phase_rows(rows, "woof", "all") == rows


@pytest.mark.parametrize("phase", ["native", "pure"])
@pytest.mark.parametrize("project", ["recast-woof-data", "gpuwm-kernels", "gpuwm-kernels-startup"])
def test_companion_pair_cannot_use_an_engine_phase(publication, manifest, phase, project):
    plan = publication.load_manifest(manifest, TAG, COMMIT, REPOSITORY)
    with pytest.raises(publication.PublicationError, match="only engine"):
        publication.phase_rows(plan["pypi"][project], project, phase)


def command(steps, operation, project, phase):
    matches = []
    for index, step in enumerate(steps):
        script = step.get("run", "")
        if not re.search(r"promote_prepared_release\.py\s+" + re.escape(operation) + r"\s", script):
            continue
        if not re.search(r"--project\s+" + re.escape(project) + r"(?:\s|$)", script):
            continue
        selected = re.search(r"--phase\s+(\S+)", script)
        if (selected.group(1) if selected else "all") == phase:
            matches.append(index)
    assert len(matches) == 1, (operation, project, phase, matches)
    return matches[0]


def test_index_visibility_gates_companion_then_native_then_universal_uploads():
    from _woof_snapshot import engine_workflow_reason
    if (reason := engine_workflow_reason(".github/workflows/publish.yml")) is not None:
        pytest.skip(reason)
    workflow = yaml.safe_load((ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    # gpuwm-kernels-startup publishes from its own job (environment pypi-startup), which
    # publish needs: every companion is indexed before the first engine wheel uploads.
    needs = jobs["publish"]["needs"]
    assert "publish_startup" in ([needs] if isinstance(needs, str) else needs)
    for job, phases in (("publish_startup", [("startup", "gpuwm-kernels-startup", "all", "all")]),
                        ("publish", [("data", "recast-woof-data", "all", "all"), ("kernels", "gpuwm-kernels", "all", "all"),
                                     ("native", "woof", "native", "native"), ("pure", "woof", "pure", "all")])):
        steps = jobs[job]["steps"]
        previous_wait = -1
        for identifier, project, stage_phase, wait_phase in phases:
            stage = command(steps, "stage-missing", project, stage_phase)
            assert steps[stage]["id"] == identifier
            uploads = [i for i, step in enumerate(steps)
                       if step.get("uses", "").startswith("pypa/gh-action-pypi-publish@")
                       and step.get("if") == f"steps.{identifier}.outputs.upload_required == 'true'"]
            assert len(uploads) == 1
            wait = command(steps, "wait-index", project, wait_phase)
            assert previous_wait < stage < uploads[0] < wait
            assert "if" not in steps[wait], "a retry with no new uploads must still prove the complete index"
            previous_wait = wait
        assert len([step for step in steps
                    if step.get("uses", "").startswith("pypa/gh-action-pypi-publish@")]) == len(phases)


def test_native_index_gate_allows_missing_universal_files_but_rejects_foreign_bytes(publication, manifest):
    plan = publication.load_manifest(manifest, TAG, COMMIT, REPOSITORY)
    rows = plan["pypi"]["woof"]
    native = publication.phase_rows(rows, "woof", "native")
    payload = {"info": {"name": "woof", "version": VERSION}, "urls": [
        {"filename": row["filename"], "size": row["bytes"], "digests": {"sha256": row["sha256"]}}
        for row in native]}
    result = publication.wait_for_index("woof", VERSION, rows, lambda *args: (200, deepcopy(payload)),
                                        clock=lambda: 0, timeout=5, required=native)
    assert result["status"] == "PASS" and result["files"] == 2 and result["attempts"] == 1
    payload["urls"].append({"filename": f"gpuwm-{VERSION}-py3-none-any.whl", "size": 1, "digests": {"sha256": "f" * 64}})
    with pytest.raises(publication.PublicationError, match="differs"):
        publication.wait_for_index("woof", VERSION, rows, lambda *args: (200, payload), clock=lambda: 0,
                                    timeout=5, required=native)
