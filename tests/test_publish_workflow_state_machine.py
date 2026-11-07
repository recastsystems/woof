"""The parsed workflow and fake controller endpoints enforce publication order.

No test performs a network request or invokes the real publication entrypoint.
The callable controller is exercised with synthetic proofs and local byte files.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
MANIFEST_SHA = "b" * 64
VERSION = "9.8.7"


@pytest.fixture(scope="module")
def workflow():
    from _woof_snapshot import engine_workflow_reason
    reason = engine_workflow_reason(".github/workflows/publish.yml")
    if reason is not None:
        pytest.skip(reason)
    return yaml.safe_load((ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def publication():
    spec = importlib.util.spec_from_file_location("state_machine_publication", ROOT / "tools/promote_prepared_release.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def needs(job):
    value = job.get("needs", [])
    return {value} if isinstance(value, str) else set(value)


def ancestors(jobs, name):
    found = set()
    pending = list(needs(jobs[name]))
    while pending:
        parent = pending.pop()
        assert parent != name, "publication job graph has a cycle"
        if parent not in found:
            found.add(parent)
            pending.extend(needs(jobs[parent]))
    return found


def scripts(job):
    return "\n".join(step.get("run", "") for step in job["steps"])


def test_public_mutation_waits_for_exact_assets_and_both_platforms(workflow):
    jobs = workflow["jobs"]
    ordered = ["controller", "release_authority", "prepare", "test", "qualify", "assets", "publish_startup", "publish",
               "release"]
    for predecessor, successor in zip(ordered, ordered[1:]):
        assert predecessor in ancestors(jobs, successor), (predecessor, successor)
    matrix = jobs["qualify"]["strategy"]["matrix"]["include"]
    assert {row["platform"] for row in matrix} == {"linux-x86_64", "win-x86_64"}
    assert len(matrix) == 2
    assert jobs["qualify"]["strategy"]["fail-fast"] is False


def test_release_events_and_dispatch_share_a_serialized_authority(workflow):
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 parses on as true.
    assert triggers["release"]["types"] == ["published"]
    assert "pull_request" in triggers
    dispatch = triggers["workflow_dispatch"]["inputs"]
    assert dispatch["release_tag"]["required"] is True
    assert "publication_manifest_sha256" in dispatch
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert "github.ref" in workflow["concurrency"]["group"]
    for name, job in workflow["jobs"].items():
        if name != "controller":
            assert job["if"] == "github.event_name != 'pull_request'"


def test_oidc_and_github_write_are_in_separate_jobs(workflow):
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    # One trusted-publisher environment per job: PyPI refuses two identical pending
    # publishers, so gpuwm-kernels-startup is bound to pypi-startup in its own job.
    publishers = {"publish": "pypi", "publish_startup": "pypi-startup"}
    for name, environment in publishers.items():
        assert jobs[name]["permissions"] == {"id-token": "write"}
        assert jobs[name]["environment"] == environment
    for name, job in jobs.items():
        permissions = job.get("permissions", workflow["permissions"])
        if name not in publishers:
            assert permissions.get("id-token") != "write"
        if permissions.get("contents") == "write":
            assert name in {"release_authority", "assets"}
    for name in publishers:
        publisher = jobs[name]
        assert "GH_TOKEN" not in json.dumps(publisher)
        assert not any(step.get("uses", "").startswith("actions/checkout@") for step in publisher["steps"])
        for step in publisher["steps"]:
            if step.get("uses", "").startswith("pypa/gh-action-pypi-publish@"):
                assert not {"password", "user", "skip-existing"} & set(step.get("with", {}))


def test_startup_companion_uploads_only_under_its_own_environment(workflow):
    jobs = workflow["jobs"]
    startup = scripts(jobs["publish_startup"])
    assert "--project gpuwm-kernels-startup" in startup
    assert re.findall(r"--project\s+(\S+)", startup) == ["gpuwm-kernels-startup", "gpuwm-kernels-startup"]
    assert "gpuwm-kernels-startup" not in scripts(jobs["publish"])
    assert "publish_startup" in needs(jobs["publish"])
    assert "publish_startup" in needs(jobs["release"])
    uploads = [step for step in jobs["publish_startup"]["steps"]
               if step.get("uses", "").startswith("pypa/gh-action-pypi-publish@")]
    assert len(uploads) == 1 and uploads[0]["if"] == "steps.startup.outputs.upload_required == 'true'"


def test_release_jobs_execute_the_tested_controller_after_hash_verification(workflow):
    jobs = workflow["jobs"]
    assert "controller_sha256" in jobs["test"]["outputs"]
    for name, job in jobs.items():
        if name in {"controller", "test"}:
            continue
        steps = job["steps"]
        uses = [i for i, step in enumerate(steps)
                if re.search(r"python\s+(?:tools|controller)/promote_prepared_release\.py\s", step.get("run", ""))]
        source = "controller" if name in {"release_authority", "ci", "prepare"} else "test"
        checks = [i for i, step in enumerate(steps)
                  if "sha256sum --check --strict" in step.get("run", "")
                  and step.get("env", {}).get("CONTROLLER_SHA256") == "${{ needs." + source + ".outputs.controller_sha256 }}"]
        assert uses and len(checks) == 1, name
        assert checks[0] < min(uses), name
        if name in {"prepare", "qualify"}:
            checkouts = [step for step in steps if step.get("uses", "").startswith("actions/checkout@")]
            assert len(checkouts) == 1
            assert checkouts[0]["with"]["ref"] == "${{ needs.release_authority.outputs.commit }}"
            assert checkouts[0]["with"]["persist-credentials"] is False
        else:
            controllers = [step for step in steps if step.get("with", {}).get("name") == "publication-controller"]
            assert len(controllers) == 1
            assert controllers[0]["uses"].startswith("actions/download-artifact@")
        if name != "release_authority":
            assert job["env"]["PUBLICATION_COMMIT"] == "${{ needs.release_authority.outputs.commit }}"
            assert job["env"]["PUBLICATION_MANIFEST_SHA256"] == "${{ needs.release_authority.outputs.manifest_sha256 }}"


def test_actions_and_artifact_handoffs_are_pinned_to_this_run(workflow):
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            action = step.get("uses")
            if action:
                assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", action), action
            if action and action.startswith("actions/download-artifact@"):
                assert not {"run-id", "repository", "github-token"} & set(step.get("with", {}))
            if action and action.startswith("actions/upload-artifact@"):
                assert step["with"]["if-no-files-found"] == "error"


def test_promotion_uses_prepared_bytes_and_keeps_existing_packaging_checks(workflow):
    jobs = workflow["jobs"]
    for name, job in jobs.items():
        if name != "test":
            assert not re.search(r"python\s+-m\s+build|cargo\s+build|gh\s+release\s+(?:upload|create|delete)|git\s+(?:push|tag)", scripts(job)), name
    test_script = scripts(jobs["test"])
    selected = {"publish_workflow_state_machine", "release_version_declaration", "publish_dist_shape_agreement",
                "verify_release_artifacts", "package_data_coverage", "prepared_publication"}
    assert set(re.findall(r"tests/test_(\w+)\.py", test_script)) == selected
    assert "cargo build" not in test_script and "rustup" not in test_script and "-e " not in test_script
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    nightly = ci["jobs"]["publication_contracts"]
    assert nightly["if"] == "github.event_name == 'schedule'"
    assert ci.get("on", ci.get(True))["schedule"] == [{"cron": "0 9 * * *"}]
    # Keep the entire former tag-time selection, including the fetcher callers.
    previous = {"bridge_fetch", "table_fetch", "package_data_coverage", "companion_distribution",
        "configs_are_packaged", "doctor", "doctor_nexrad", "cli", "tui_cli", "source_adapters",
        "namelist_compat", "native_wrf_distribution", "prepared_publication", "carried_channel",
        "carried_release_packet", "publish_workflow_state_machine", "publish_dist_shape_agreement",
        "release_version_declaration", "release_bump_version", "precut_gate_completion", "gpu_pin_set",
        "bridge_bundle_adopt", "verify_source_bridge_pins", "verify_release_artifacts", "native_build_inputs",
        "workflow_rehearsals", "workflow_job_replay", "pypi_capacity", "publish_preflight", "vendor_cache_ignore"}
    assert set(re.findall(r"tests/test_(\w+)\.py", scripts(nightly))) == previous


#: What marks a listed test file as one that stages through the fetcher:
#: table staging (``fetch_tables``) and static assets (``geog_assets``) are
#: pin-checked by rw_fetch since 2.8.9, and the other two name it directly.
FETCHER_USE = ("fetch_tables", "geog_assets", "rw_fetch", "find_fetch_bin")


def test_nightly_builds_the_fetcher_before_the_files_that_need_it(workflow):
    """The job that roots the publish chain has the one native its files use.

    The breakage, named: the v2.8.9 publish run 38124857348 failed in this
    job on five tests of tests/test_table_fetch.py ("rw_fetch is missing"),
    after the tag was public and the draft release created. ci.yml had a
    fetcher on the leg that builds no natives; this job had not, and nothing
    compared the two. This file runs in ci.yml on every push.
    """
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    steps = ci["jobs"]["publication_contracts"]["steps"]
    runners = [i for i, step in enumerate(steps) if "python -m pytest" in step.get("run", "")]
    assert len(runners) == 1, runners
    tests = steps[runners[0]]
    listed = re.findall(r"tests/\S+\.py", tests["run"])
    needing = [name for name in listed
               if any(mark in (ROOT / name).read_text(encoding="utf-8") for mark in FETCHER_USE)]
    assert "tests/test_table_fetch.py" in needing, needing
    command = "rustup run 1.94.0 cargo build --release --locked --offline -p rw-fetch"
    builders = [i for i, step in enumerate(steps) if command in step.get("run", "")]
    assert len(builders) == 1 and builders[0] < runners[0], (
        f"the test job runs {needing} with no earlier step that builds rw_fetch")
    build = steps[builders[0]]
    assert "if" not in build and "if" not in tests
    assert "rustup toolchain install 1.94.0 --profile minimal" in build["run"]
    # Outside the checkout, and named to the tests: nothing is left to the
    # resolution ladder or to whatever a runner image happens to carry.
    target = build["env"]["CARGO_TARGET_DIR"]
    assert target.startswith("${{ runner.temp }}/"), target
    assert tests["env"]["WOOF_RW_FETCH"] == target + "/release/rw_fetch"
    # The same toolchain pin and cargo flags as ci.yml's fetcher-only leg.
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for line in build["run"].splitlines():
        assert line.strip() in ci, line


def test_publish_selection_has_no_unbuilt_stage1_native_dependencies(workflow):
    job = workflow["jobs"]["test"]
    # This job has one supported runner/interpreter; the platform matrix is
    # qualify, which consumes the sealed packet instead of running these tests.
    assert job["runs-on"] == "ubuntu-24.04" and "strategy" not in job
    setup = [step for step in job["steps"] if step.get("uses", "").startswith("actions/setup-python@")]
    assert len(setup) == 1 and setup[0]["with"]["python-version"] == "3.11"
    listed = set(re.findall(r"tests/\S+\.py", scripts(job)))
    native = {line.split("#", 1)[0].strip() for line in
              (ROOT / "tools/battery/stage1_native_files.txt").read_text().splitlines()}
    native.discard("")
    assert not listed & native, f"new publish tests need additional native builds: {sorted(listed & native)}"


def test_docker_publisher_upload_paths_are_workspace_relative(workflow):
    placed = {"data": "publish", "kernels": "publish", "startup": "publish_startup", "native": "publish",
              "pure": "publish"}
    for phase, job in placed.items():
        steps = workflow["jobs"][job]["steps"]
        stage = next(step for step in steps if step.get("id") == phase)
        upload = next(step for step in steps if step.get("if") == f"steps.{phase}.outputs.upload_required == 'true'")
        directory = upload["with"]["packages-dir"].rstrip("/")
        assert directory and not PurePosixPath(directory).is_absolute()
        assert not any(token in directory for token in ("$", "..", "\\", ":")), directory
        assert re.search(r"--out\s+[\"']?" + re.escape(directory) + r"[\"']?(?:\s|$)", stage["run"])


def test_packet_smoke_consumes_upload_bytes_before_any_public_mutation(workflow):
    jobs = workflow["jobs"]
    job = jobs["test"]
    assert needs(job) == {"controller", "release_authority", "prepare"}
    assert job["timeout-minutes"] == 8
    assert job["env"]["RELEASE_VERSION"] == "${{ needs.release_authority.outputs.version }}"
    downloads = [step["with"] for step in job["steps"] if step.get("uses", "").startswith("actions/download-artifact@")]
    assert downloads == [{"name": "release-proof", "path": "${{ runner.temp }}/release-proof"}]
    assert 'publish_packet_smoke.py install --dists "$RUNNER_TEMP/release-proof/dists"' in scripts(job)
    assert 'publish_packet_smoke.py check --dists "$RUNNER_TEMP/release-proof/dists"' in scripts(job)
    assert '--dists "$RUNNER_TEMP/release-proof/dists"' in scripts(jobs["publish"])
    # The shared _chain_static_staging_kept_off_the_network fixture imports
    # numpy before every selected metadata contract, even on a bare venv.
    assert "python -m pip install pytest pyyaml 'setuptools>=77' wheel 'numpy>=2'" in scripts(job)
    for name in ("qualify", "assets", "publish_startup", "publish", "release"):
        assert "test" in ancestors(jobs, name)
    uploads = [step for step in jobs["controller"]["steps"] if step.get("uses", "").startswith("actions/upload-artifact@")]
    assert len(uploads) == 1 and uploads[0]["with"]["name"] == "publication-controller"
    assert 'sha256sum tools/promote_prepared_release.py' in scripts(job)


@pytest.fixture
def controller_case(publication, monkeypatch, tmp_path):
    filenames = {
        "recast-woof-data": [f"woof_data-{VERSION}-py3-none-any.whl", f"woof_data-{VERSION}.tar.gz"],
        "gpuwm-kernels": [f"gpuwm_kernels-{VERSION}-py3-none-any.whl", f"gpuwm_kernels-{VERSION}.tar.gz"],
        "gpuwm-kernels-startup": [f"gpuwm_kernels_startup-{VERSION}-py3-none-any.whl",
                                  f"gpuwm_kernels_startup-{VERSION}.tar.gz"],
        "woof": [f"gpuwm-{VERSION}-py3-none-any.whl", f"gpuwm-{VERSION}-py3-none-manylinux_2_28_x86_64.whl",
                  f"gpuwm-{VERSION}-py3-none-win_amd64.whl", f"gpuwm-{VERSION}.tar.gz"],
    }
    dists = tmp_path / "dists"
    dists.mkdir()
    pypi = {}
    for project, names in filenames.items():
        pypi[project] = []
        for name in names:
            payload = ("proven synthetic " + name).encode()
            (dists / name).write_bytes(payload)
            pypi[project].append({"filename": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    captured = {"commit": COMMIT, "manifest_sha256": MANIFEST_SHA, "repository": "example-org/example-repo",
                "tag": "v" + VERSION, "release_id": 42, "prerelease": False}
    proven = {"captured": captured, "plan": {"version": VERSION, "pypi": pypi}}
    monkeypatch.setattr(publication, "proof", lambda path: deepcopy(proven))
    events = []
    state = {"draft": True, "indexes": {project: (404, None) for project in pypi}}

    def fetch(project, version, timeout=30):
        assert version == VERSION
        events.append(("index", project))
        return deepcopy(state["indexes"][project])

    class FakeGitHub:
        def __init__(self, repository):
            assert repository == captured["repository"]
        def json(self, suffix, *, data=None):
            assert suffix == "/releases/42"
            assert data == {"draft": False, "prerelease": False}
            events.append(("patch", suffix))
            state["draft"] = False
            return {"id": 42}

    def remote(client, selected, *, require_public):
        assert selected == captured
        events.append(("remote", require_public))
        assert not require_public or state["draft"] is False
        return {"id": 42, "draft": state["draft"], "immutable": True}

    monkeypatch.setattr(publication, "fetch_index", fetch)
    monkeypatch.setattr(publication, "GitHub", FakeGitHub)
    monkeypatch.setattr(publication, "verify_remote", remote)
    monkeypatch.setattr(publication, "output_values", lambda values: events.append(("outputs", values)))
    smokes = tmp_path / "smokes"
    for platform in ("linux-x86_64", "win-x86_64"):
        publication.write_json(smokes / platform / "SMOKE.json", {
            "platform": platform, "status": "PASS", "commit": COMMIT, "manifest_sha256": MANIFEST_SHA})
    args = SimpleNamespace(proof=tmp_path / "proof", smokes=smokes, out=tmp_path / "PUBLIC.json")
    return SimpleNamespace(args=args, proven=proven, events=events, state=state, dists=dists)


def payload(project, rows):
    return {"info": {"name": project, "version": VERSION}, "urls": [
        {"filename": row["filename"], "size": row["bytes"], "digests": {"sha256": row["sha256"]}}
        for row in rows]}


def test_draft_promotion_checks_both_indexes_before_the_only_write(publication, controller_case):
    case = controller_case
    publication.promote(case.args)
    assert case.events == [("index", "recast-woof-data"), ("index", "gpuwm-kernels"),
                           ("index", "gpuwm-kernels-startup"), ("index", "woof"), ("remote", False),
                           ("patch", "/releases/42"), ("remote", True)]
    assert publication.read_json(case.args.out)["public_before_pypi"] is True


def test_an_already_public_retry_makes_no_github_write(publication, controller_case):
    case = controller_case
    case.state["draft"] = False
    publication.promote(case.args)
    assert not any(event[0] == "patch" for event in case.events)
    assert publication.read_json(case.args.out)["status"] == "PASS"


@pytest.mark.parametrize("project", ["recast-woof-data", "gpuwm-kernels", "gpuwm-kernels-startup", "woof"])
def test_a_pypi_collision_prevents_github_promotion(publication, controller_case, project):
    case = controller_case
    collision = payload(project, case.proven["plan"]["pypi"][project])
    collision["urls"][0]["digests"]["sha256"] = "f" * 64
    case.state["indexes"][project] = (200, collision)
    with pytest.raises(publication.PublicationError, match="differs"):
        publication.promote(case.args)
    assert not any(event[0] in {"remote", "patch"} for event in case.events)
    assert not case.args.out.exists()


@pytest.mark.parametrize("field,value", [("status", "FAIL"), ("commit", "c" * 40),
                                         ("manifest_sha256", "d" * 64), ("platform", "linux-x86_64")])
def test_invalid_platform_proof_stops_before_any_remote_access(publication, controller_case, field, value):
    case = controller_case
    path = case.args.smokes / "win-x86_64/SMOKE.json"
    row = publication.read_json(path)
    row[field] = value
    publication.write_json(path, row)
    with pytest.raises(publication.PublicationError):
        publication.promote(case.args)
    assert case.events == []
    assert not case.args.out.exists()


def test_stage_rechecks_every_local_file_before_copying_any(publication, controller_case):
    case = controller_case
    bad = case.proven["plan"]["pypi"]["woof"][-1]["filename"]
    (case.dists / bad).write_bytes(b"tampered")
    args = SimpleNamespace(proof=case.args.proof, dists=case.dists, project="recast-woof-data", phase="all", out=case.args.out)
    with pytest.raises(publication.PublicationError, match="artifact changed"):
        publication.stage_missing(args)
    assert not args.out.exists()


def test_partial_index_retry_stages_only_missing_exact_bytes(publication, controller_case):
    case = controller_case
    rows = case.proven["plan"]["pypi"]["woof"]
    native = [row for row in rows if row["filename"].endswith(("manylinux_2_28_x86_64.whl", "win_amd64.whl"))]
    case.state["indexes"]["woof"] = (200, payload("woof", native[:1]))
    args = SimpleNamespace(proof=case.args.proof, dists=case.dists, project="woof", phase="native", out=case.args.out)
    publication.stage_missing(args)
    assert [path.name for path in args.out.iterdir()] == [native[1]["filename"]]
    assert (args.out / native[1]["filename"]).read_bytes() == (case.dists / native[1]["filename"]).read_bytes()
    assert case.events[-1] == ("outputs", {"upload_required": "true", "missing_count": "1"})


def test_complete_index_retry_does_not_upload_again(publication, controller_case):
    case = controller_case
    rows = case.proven["plan"]["pypi"]["recast-woof-data"]
    case.state["indexes"]["recast-woof-data"] = (200, payload("recast-woof-data", rows))
    args = SimpleNamespace(proof=case.args.proof, dists=case.dists, project="recast-woof-data", phase="all", out=case.args.out)
    publication.stage_missing(args)
    assert list(args.out.iterdir()) == []
    assert case.events[-1] == ("outputs", {"upload_required": "false", "missing_count": "0"})


# --------------------------------------------------------------------------
# publication waits for ci on the commit it publishes
# --------------------------------------------------------------------------
#
# ci failed on the public repository for 2.7.6, 2.7.7 and 2.8.0 while this
# workflow succeeded on all three: nothing here read ci.


def test_publication_depends_on_ci_passing_on_the_published_commit(workflow):
    jobs = workflow["jobs"]
    gate = jobs["ci"]
    assert needs(gate) == {"controller", "release_authority"}
    assert gate["permissions"] == {"actions": "read"}
    run = scripts(gate)
    assert re.search(r"promote_prepared_release\.py ci-passed .*--commit \"\$PUBLICATION_COMMIT\"", run), run
    assert "GH_TOKEN" in json.dumps(gate["steps"])
    # Every job that verifies, qualifies, promotes or uploads waits for it.
    for name in jobs:
        if name not in {"controller", "release_authority", "ci"}:
            assert "ci" in ancestors(jobs, name), name


def _run(sha, status="completed", conclusion="success", run_id=1, branch="main", event="push"):
    timestamp = (datetime(2026, 10, 1, tzinfo=timezone.utc)
                 + timedelta(seconds=run_id if type(run_id) is int else 0)).isoformat()
    return {"id": run_id, "head_sha": sha, "status": status, "conclusion": conclusion,
            "event": event, "head_branch": branch, "run_attempt": 1,
            "created_at": timestamp, "run_started_at": timestamp, "updated_at": timestamp}


def test_ci_verdicts_read_only_this_commit_and_fail_closed(publication):
    judge = publication.judge_ci_runs
    other = "c" * 40
    assert judge([], COMMIT)[0] == "pending"
    assert judge([_run(other)], COMMIT)[0] == "pending"
    assert judge([_run(COMMIT, status="in_progress", conclusion=None)], COMMIT)[0] == "pending"
    assert judge([_run(COMMIT), _run(COMMIT, "queued", None, 2)], COMMIT)[0] == "pending"
    assert judge([_run(COMMIT)], COMMIT)[0] == "passed"
    assert judge([_run(COMMIT), _run(COMMIT, conclusion="cancelled", run_id=2)], COMMIT)[0] == "refused"
    verdict, why = judge([_run(COMMIT), _run(COMMIT, conclusion="failure", run_id=7)], COMMIT)
    assert verdict == "refused" and "7=failure" in why
    assert judge([_run(COMMIT, conclusion="cancelled")], COMMIT)[0] == "refused"
    assert judge([_run(COMMIT, conclusion="timed_out")], COMMIT)[0] == "refused"
    assert judge([_run(other, conclusion="failure"), _run(COMMIT)], COMMIT)[0] == "passed"


def test_a_successful_retry_supersedes_an_older_failure_but_a_new_failure_is_fatal(publication):
    judge = publication.judge_ci_runs
    failed = _run(COMMIT, conclusion="failure", run_id=1)
    retry = _run(COMMIT, run_id=2, event="workflow_dispatch")
    assert judge([failed, retry], COMMIT)[0] == "passed"
    assert judge([retry, failed], COMMIT)[0] == "passed"
    assert judge([retry, failed, _run(COMMIT, conclusion="failure", run_id=3)], COMMIT)[0] == "refused"
    assert judge([retry, failed, _run(COMMIT, "in_progress", None, run_id=3)], COMMIT)[0] == "pending"


def test_a_later_attempt_of_an_older_run_id_supersedes_historical_failure(publication):
    earlier_id = dict(_run(COMMIT, run_id=1), run_attempt=2,
                      run_started_at="2026-10-01T01:00:00Z", updated_at="2026-10-01T01:01:00Z")
    newer_id = _run(COMMIT, conclusion="failure", run_id=2)
    assert publication.judge_ci_runs([newer_id, earlier_id], COMMIT)[0] == "passed"
    assert publication.judge_ci_runs([newer_id, dict(earlier_id, conclusion="failure")], COMMIT)[0] == "refused"


@pytest.mark.parametrize("status", ["queued", "requested", "waiting", "pending"])
def test_a_queued_retry_waits_even_when_its_old_started_timestamp_precedes_a_pass(publication, status):
    retry = dict(_run(COMMIT, status=status, conclusion=None, run_id=1), run_attempt=2,
                 updated_at="2026-10-01T01:00:00Z")
    assert publication.judge_ci_runs([_run(COMMIT, run_id=2), retry], COMMIT)[0] == "pending"


def test_an_unorderable_attempt_cannot_qualify_a_commit(publication):
    with pytest.raises(publication.PublicationError, match="timestamp"):
        publication.judge_ci_runs([dict(_run(COMMIT), run_started_at="invalid")], COMMIT)


def test_a_pr_merge_check_cannot_qualify_the_release_commit(publication):
    judge = publication.judge_ci_runs
    pr = _run(COMMIT, run_id=10, event="pull_request")
    assert judge([pr], COMMIT)[0] == "pending"
    assert judge([pr, _run(COMMIT, conclusion="failure")], COMMIT)[0] == "refused"
    assert judge([_run(COMMIT), dict(pr, conclusion="failure")], COMMIT)[0] == "passed"


@pytest.mark.parametrize("conclusion", [None, "neutral", "skipped", "unknown", "cancelled", "failure"])
def test_only_an_explicit_success_can_pass_the_latest_check(publication, conclusion):
    assert publication.judge_ci_runs([_run(COMMIT, conclusion=conclusion)], COMMIT)[0] == "refused"


def test_an_invalid_run_id_cannot_forge_check_order(publication):
    with pytest.raises(publication.PublicationError, match="positive run id"):
        publication.judge_ci_runs([_run(COMMIT, run_id=None)], COMMIT)


def test_ci_inventory_follows_pagination_before_judging_a_commit(publication):
    seen = []
    rows = [_run(COMMIT, run_id=i + 2, event="pull_request") for i in range(100)]

    class FakeGitHub:
        def json(self, suffix):
            seen.append(suffix)
            return {"workflow_runs": rows if suffix.endswith("&page=1") else [_run(COMMIT)], "total_count": 101}

    runs = publication._ci_workflow_runs(FakeGitHub(), COMMIT)
    assert len(seen) == 2 and "page=2" in seen[1]
    assert publication.judge_ci_runs(runs, COMMIT)[0] == "passed"


def _jobs(gate="completed", gate_conclusion="success", cpu="in_progress", cpu_conclusion=None):
    names = ["syntax", "imports (ubuntu-24.04)", "imports (windows-2025)",
             "Static qualification (Linux UNQUALIFIED)", "Static qualification (Windows exact comparisons)"]
    rows = [{"name": name, "status": gate, "conclusion": gate_conclusion} for name in names]
    return rows + [{"name": f"{job} ({os})", "status": cpu, "conclusion": cpu_conclusion}
                   for job in ("cpu", "native", "oracles") for os in ("ubuntu-24.04", "windows-2025")]


def _ci_gate(publication, monkeypatch, tmp_path, pages, jobs=None):
    seen = []
    jobs = _jobs(gate="in_progress", gate_conclusion=None) if jobs is None else jobs

    class FakeGitHub:
        def __init__(self, repository):
            self.repository = repository

        def json(self, suffix, *, data=None):
            assert data is None, "the ci gate never writes"
            if "/jobs?" in suffix:
                assert suffix.startswith("/actions/runs/1/attempts/1/jobs"), suffix
                return {"jobs": jobs}
            seen.append(suffix)
            return {"workflow_runs": pages[min(len(seen), len(pages)) - 1]}

    monkeypatch.setattr(publication, "GitHub", FakeGitHub)
    now = [0.0]
    args = SimpleNamespace(repository="owner/repo", commit=COMMIT, timeout=300.0, interval=60.0,
                           out=tmp_path / "CI-PASSED.json")
    run = lambda: publication.ci_passed(args, sleep=lambda s: now.__setitem__(0, now[0] + s),
                                        clock=lambda: now[0])
    return run, seen, args


def test_the_ci_gate_waits_for_running_ci_then_records_the_pass(publication, monkeypatch, tmp_path):
    pages = [[_run(COMMIT, "in_progress", None)], [_run(COMMIT, "in_progress", None)], [_run(COMMIT)]]
    run, seen, args = _ci_gate(publication, monkeypatch, tmp_path, pages)
    run()
    assert len(seen) == 3
    assert all(f"head_sha={COMMIT}" in suffix and "/actions/workflows/ci.yml/runs" in suffix for suffix in seen)
    record = json.loads(args.out.read_text(encoding="utf-8"))
    assert record["status"] == "PASS" and record["commit"] == COMMIT


def test_the_ci_gate_refuses_a_failed_run_and_gives_up_on_one_that_never_ends(publication, monkeypatch, tmp_path):
    run, _seen, args = _ci_gate(publication, monkeypatch, tmp_path, [[_run(COMMIT, conclusion="failure")]],
                                jobs=_jobs(gate_conclusion="failure", cpu="completed", cpu_conclusion="success"))
    with pytest.raises(publication.PublicationError, match="did not pass"):
        run()
    assert not args.out.exists()
    run, seen, args = _ci_gate(publication, monkeypatch, tmp_path, [[_run(COMMIT, "in_progress", None)]])
    with pytest.raises(publication.PublicationError, match="gave up"):
        run()
    assert len(seen) == 6 and not args.out.exists()


def test_ci_is_judged_on_its_fast_gate_jobs_and_the_long_batteries_do_not_gate(publication, monkeypatch, tmp_path):
    """Project ruling 2026-10-20: the release must not wait ~2 h for the hosted cpu batteries that already ran locally."""
    judge = publication.judge_ci_runs
    running = dict(_run(COMMIT, "in_progress", None), jobs=_jobs())
    assert judge([running], COMMIT)[0] == "passed"
    cpu_red = dict(_run(COMMIT, conclusion="failure"), jobs=_jobs(cpu="completed", cpu_conclusion="failure"))
    assert judge([cpu_red], COMMIT)[0] == "passed"
    gate_red = dict(_run(COMMIT, conclusion="failure"), jobs=_jobs(gate_conclusion="failure"))
    assert judge([gate_red], COMMIT)[0] == "refused"
    waiting = dict(_run(COMMIT, "in_progress", None), jobs=_jobs(gate="in_progress", gate_conclusion=None))
    assert judge([waiting], COMMIT)[0] == "pending"
    one_leg = dict(_run(COMMIT, "in_progress", None),
                   jobs=[job for job in _jobs() if job["name"] != "imports (windows-2025)"])
    verdict, why = judge([one_leg], COMMIT)
    assert verdict == "pending" and "imports" in why
    skipped = dict(_run(COMMIT, conclusion="failure"), jobs=_jobs(gate_conclusion="skipped"))
    assert judge([skipped], COMMIT)[0] == "refused"
    run, seen, args = _ci_gate(publication, monkeypatch, tmp_path, [[_run(COMMIT, "in_progress", None)]],
                               jobs=_jobs())
    run()
    assert len(seen) == 1 and json.loads(args.out.read_text(encoding="utf-8"))["status"] == "PASS"
