"""The capsule: one schema, one builder, one set of emitting routes.

The tests hold the three things a receipt can quietly lose -- coverage of the
published pin set, a shared builder behind every exit, and an accurate account of
what was not measured -- and each is paired with the mutation that must break
it.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import jsonschema
import pytest

from woof.certify import capsule as capsule_module
from woof.certify.capsule import (CAPSULE_FILENAME, CAPSULE_SCHEMA_ID,
                                   EMISSION_SITES, REQUIRED_SECTIONS,
                                   SCHEMA_PATH, CapsuleValidationError,
                                   build_capsule, emit_run_capsule,
                                   load_certification_capsule,
                                   validate_certification_capsule)
from woof.certify.pins import (COMPOUND_DOC_ROWS, PIN_KEYS, PINS,
                                PIN_TABLE_DOC, PIN_TABLE_HEADING, resolve_pins)

REPO = Path(__file__).resolve().parents[1]
DETERMINISM = REPO / PIN_TABLE_DOC


def _published_pin_rows() -> list[str]:
    """First column of every row of the published pin table."""
    lines = DETERMINISM.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines)
                 if PIN_TABLE_HEADING in line)
    rows: list[str] = []
    started = False
    for line in lines[start:]:
        if line.startswith("| Pin |"):
            started = True
            continue
        if not started:
            continue
        if line.startswith("|---"):
            continue
        if not line.startswith("|"):
            break
        rows.append(line.split("|")[1].strip())
    return rows


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _stub_capsule(**kwargs):
    """A capsule whose ONLY defect is the one the calling test introduces.

    The kernel manifest is filled through the real recorder before building.
    A test process compiles nothing, so ``build_capsule`` would otherwise
    snapshot an empty manifest -- and the certification path now refuses
    that on its own, which would make every refusal test below pass for the
    wrong reason.  A guard that pre-empts the refusal a test is aiming at is
    a test that no longer measures what it names.
    """
    from woof.certify.kernel_manifest import record_module

    record_module("woof.core.kernels:stub",
                  source="__device__ void stub() {}",
                  options=["-std=c++17"])
    return build_capsule(emission_site="supervisor:success",
                         require_gpu=False, **kwargs)


# --- F3-AC1: the pin set is covered, present, and accurately statused --------

def test_the_published_pin_table_has_eleven_rows_two_of_them_compound():
    rows = _published_pin_rows()
    assert len(rows) == 11, rows
    assert set(COMPOUND_DOC_ROWS) <= set(rows)
    assert len(COMPOUND_DOC_ROWS) == 2


def test_every_pin_item_quotes_a_published_row_and_the_rows_are_all_covered():
    rows = _published_pin_rows()
    assert [pin.doc_row for pin in PINS if pin.doc_row_part is None or True]
    for pin in PINS:
        assert pin.doc_row in rows, (
            f"pin {pin.key} quotes a row that is not published: {pin.doc_row}")
    assert {pin.doc_row for pin in PINS} == set(rows)
    assert len(PINS) == len(rows) + len(COMPOUND_DOC_ROWS) == 13


def test_the_schema_enumerates_exactly_the_pin_keys():
    stack = _schema()["properties"]["numerical_stack"]
    assert tuple(stack["required"]) == PIN_KEYS
    assert set(stack["properties"]) == set(PIN_KEYS)
    assert stack["additionalProperties"] is False


def test_the_schema_is_a_valid_json_schema():
    schema = _schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$id"] == CAPSULE_SCHEMA_ID


def test_every_pin_is_present_with_a_declared_status_and_a_reason_if_absent():
    capsule = _stub_capsule()
    stack = capsule["numerical_stack"]
    assert tuple(stack) == PIN_KEYS
    for key, entry in stack.items():
        assert entry["status"] in {"resolved", "unavailable"}, key
        if entry["status"] == "unavailable":
            assert entry["reason"], key


def test_the_code_section_records_the_version_and_the_verbatim_commit():
    import woof
    from woof.supervisor import git_commit

    capsule = _stub_capsule()
    assert capsule["code"]["gpuwm_version"] == woof.__version__
    recorded = capsule["code"]["git_commit"]
    assert recorded == git_commit()
    assert re.fullmatch(r"[0-9a-f]{40}", recorded) or recorded.startswith(
        "unavailable: ")


def test_a_capsule_with_a_pin_removed_is_refused():
    capsule = _stub_capsule()
    capsule["numerical_stack"].pop("numpy_version")
    with pytest.raises(CapsuleValidationError):
        validate_certification_capsule(capsule)


def test_the_certification_path_refuses_an_unresolved_pin():
    """Mutation control 3: an unresolved pin cannot reach certification."""
    capsule = _stub_capsule()
    validate_certification_capsule(capsule)  # the plain path accepts it
    with pytest.raises(CapsuleValidationError, match="not.*resolved"):
        validate_certification_capsule(capsule, certification_path=True)


# --- 4090 stress finding: accurate statuses on a no-git (pip install) tree ---

_NO_GIT_SENTINEL = ("unavailable: fatal: not a git repository "
                    "(or any of the parent directories): .git")


def _simulate_no_git(monkeypatch):
    """A pip-installed tree: the package imports, git answers nothing."""
    import woof.supervisor as supervisor

    monkeypatch.setattr(supervisor, "git_commit",
                        lambda: _NO_GIT_SENTINEL)


def test_resolved_never_wraps_an_unavailable_payload(monkeypatch):
    """The published-wheel capsule carried arwen_version_and_commit with
    status "resolved" wrapping the value "unavailable: fatal: not a git
    repository".  A resolved status is a claim that something measured
    the value; a payload that says unavailable refutes the claim in the
    same breath.  On a no-git tree the pin is accurately unavailable, and
    the half that DOES exist -- the version, from package metadata -- is
    still bound in the entry."""
    _simulate_no_git(monkeypatch)
    stack = resolve_pins(require_gpu=False)
    entry = stack["arwen_version_and_commit"]
    assert entry["status"] == "unavailable"
    assert "not a git repository" in entry["reason"]
    import woof

    assert entry["value"] == {"version": woof.__version__}
    # The general law, over the whole pin set: no resolved entry ever
    # carries the unavailable sentinel inside its payload.
    for key, item in stack.items():
        if item["status"] == "resolved":
            assert "unavailable:" not in json.dumps(item["value"]), key


def test_a_git_tree_still_resolves_the_commit_pin(monkeypatch):
    import woof.supervisor as supervisor

    monkeypatch.setattr(supervisor, "git_commit", lambda: "ab" * 20)
    stack = resolve_pins(require_gpu=False)
    entry = stack["arwen_version_and_commit"]
    assert entry["status"] == "resolved"
    assert entry["value"]["git_commit"] == "ab" * 20


def test_a_no_git_capsule_still_binds_config_and_input_bytes(monkeypatch):
    """DETERMINISM.md lists config bytes and input bytes as essential
    pins, and both exist regardless of git; the stress capsule reported
    them unavailable because the pip route never handed them over."""
    _simulate_no_git(monkeypatch)
    capsule = build_capsule(
        emission_site="prepared_single_domain_forecast",
        require_gpu=False,
        run_context={
            "config_bytes": {"path": "experiment.toml", "sha256": "0" * 64},
            "input_artifact_bytes": {"experiment_config": "0" * 64},
            "runner_route_and_io_mode": {
                "route": "prepared_single_domain_forecast",
                "io_mode": "history"},
            "output_and_diagnostic_mode": {
                "io_mode": "history", "history_interval_seconds": 3600.0},
        },
        input_bytes={"entries": {
            "experiment_config:experiment.toml": {
                "algorithm": "sha256", "digest": "0" * 64},
        }})
    stack = capsule["numerical_stack"]
    assert stack["config_bytes"]["status"] == "resolved"
    assert stack["config_bytes"]["value"]["sha256"] == "0" * 64
    assert stack["input_artifact_bytes"]["status"] == "resolved"
    assert capsule["input_bytes"]["entries"]
    assert capsule["input_bytes"].get("status") != "unavailable"
    validate_certification_capsule(capsule)


def test_both_prepared_routes_hand_the_capsule_config_and_input_bytes():
    """The seam, held structurally: each prepared runner's
    emit_run_capsule call passes an input_bytes section and a
    config_bytes run-context pin.  (The runners need a prepared cache
    and a GPU to execute, so the call site is what a CPU test can pin.)
    """
    for name in ("woof.prepared_single_domain_forecast",
                 "woof.prepared_domain_tree_forecast"):
        path = REPO / (name.replace(".", "/") + ".py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)
                 and node.func.id == "emit_run_capsule"]
        assert calls, f"{name} no longer emits a capsule"
        for call in calls:
            keywords = {keyword.arg: keyword.value
                        for keyword in call.keywords}
            assert "input_bytes" in keywords, (
                f"{name} emits a capsule without binding input_bytes")
            context = keywords.get("run_context")
            assert isinstance(context, ast.Dict), name
            context_keys = {key.value for key in context.keys
                            if isinstance(key, ast.Constant)}
            assert "config_bytes" in context_keys, (
                f"{name} emits a capsule without binding config_bytes")
            assert "input_artifact_bytes" in context_keys, name


# --- F2: the stub measures nothing it cannot measure ------------------------

def test_the_stub_populates_no_stack_field_it_did_not_measure():
    capsule = _stub_capsule()
    stack = capsule["numerical_stack"]
    for key in ("gpu_identity", "cuda_driver_version", "cuda_toolkit_nvrtc"):
        assert stack[key]["status"] == "unavailable"
        assert stack[key]["value"] is None
        assert "does not probe" in stack[key]["reason"]


def test_the_builder_uses_the_probe_the_bridge_receipt_uses():
    import woof.gpu_stack_identity as probe
    import woof.native_wrf_distribution as bridge

    assert (capsule_module.gpu_cuda_stack_identity
            is bridge.gpu_cuda_stack_identity
            is probe.gpu_cuda_stack_identity)


def test_run_context_keys_outside_the_pin_set_are_refused():
    with pytest.raises(KeyError):
        resolve_pins({"not_a_pin": 1}, require_gpu=False)


# --- F2: four-site parity, and the control that must fail all four ---------

EMITTERS = (
    "woof.runtime",
    "woof.supervisor",
    "woof.prepared_single_domain_forecast",
    "woof.prepared_domain_tree_forecast",
)


def _emitter_modules():
    import importlib

    return {name: importlib.import_module(name) for name in EMITTERS}


def test_all_four_sites_emit_through_the_same_builder_object():
    for name, module in _emitter_modules().items():
        assert module.emit_run_capsule is emit_run_capsule, name


def test_all_four_sites_name_a_declared_emission_site():
    def declared_sites(expression):
        if isinstance(expression, ast.Name) and expression.id == "emission_site":
            return set()  # The common front-door wrapper forwards this argument.
        if isinstance(expression, ast.IfExp):
            return declared_sites(expression.body) | declared_sites(expression.orelse)
        assert isinstance(expression, ast.Constant), ast.dump(expression)
        assert isinstance(expression.value, str), ast.dump(expression)
        return {expression.value}

    used = set()
    for name in EMITTERS:
        path = REPO / (name.replace(".", "/") + ".py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if name == "woof.runtime":
            owners = {node.name: node for node in tree.body
                      if isinstance(node, ast.FunctionDef)}
            calls = [node for node in ast.walk(owners["_run_experiment"])
                     if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name)
                     and node.func.id == "_run_built_experiment"]
            assert {any(kw.arg == "prepared_steppers" for kw in call.keywords)
                    for call in calls} == {True, False}
            labels = [kw.value for node in ast.walk(owners["_run_built_experiment"])
                      if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name)
                      and node.func.id == "_emit_front_door_capsule"
                      for kw in node.keywords if kw.arg == "emission_site"]
            expected = ast.parse(
                "'runtime.run_experiment:single-domain' if prepared_steppers "
                "is not None else 'runtime.run_experiment:domain-tree'",
                mode="eval").body
            assert len(labels) == 1
            assert ast.dump(labels[0]) == ast.dump(expected)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in {"emit_run_capsule",
                                         "_emit_front_door_capsule"}):
                for keyword in node.keywords:
                    if keyword.arg == "emission_site":
                        used.update(declared_sites(keyword.value))
    assert used == set(EMISSION_SITES), used


def test_control_dropping_a_required_field_is_caught_at_all_four_sites(
        tmp_path, monkeypatch, capsys):
    """Mutation control 1: remove one required field from the shared builder.

    Warn-not-block posture: on the run path (certification_path=False)
    the failed emission is one warning line and NO capsule file -- a
    finished forecast is never destroyed by its audit artifact.  The
    control still bites at every site: no site may silently write a
    capsule missing the field, and the certification path stays fatal.
    """
    original = capsule_module.build_capsule

    def crippled(**kwargs):
        document = original(**kwargs)
        document.pop("numerical_stack")
        return document

    monkeypatch.setattr(capsule_module, "build_capsule", crippled)
    for index, module in enumerate(_emitter_modules().values()):
        outdir = tmp_path / str(index)
        result = module.emit_run_capsule(
            outdir, emission_site="supervisor:success", require_gpu=False)
        err = capsys.readouterr().err
        assert result is None
        assert not (outdir / CAPSULE_FILENAME).exists()
        assert "warning:" in err
        assert "certification capsule not written" in err
        assert "numerical_stack" in err
    # KEEP-HARD negative: with certification explicitly requested the
    # same failure refuses.
    with pytest.raises(CapsuleValidationError):
        capsule_module.emit_run_capsule(
            tmp_path / "cert", emission_site="supervisor:success",
            certification_path=True, require_gpu=False)


def test_a_missing_jsonschema_cannot_kill_a_completed_forecast(
        tmp_path, monkeypatch, capsys):
    """The field failure: a pip install without jsonschema ran a whole
    forecast, then died at capsule emission.  Emission must warn and
    return None; the certify path must name the remedy, not traceback."""
    import builtins

    real_import = builtins.__import__

    def no_jsonschema(name, *args, **kwargs):
        if name == "jsonschema":
            raise ModuleNotFoundError("No module named 'jsonschema'",
                                      name="jsonschema")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "jsonschema", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_jsonschema)
    outdir = tmp_path / "run"
    result = capsule_module.emit_run_capsule(
        outdir, emission_site="supervisor:success", require_gpu=False)
    err = capsys.readouterr().err
    assert result is None
    assert not (outdir / CAPSULE_FILENAME).exists()
    assert "certification capsule not written" in err
    assert "jsonschema" in err
    # Certification asked for explicitly: fatal, with the remedy named.
    with pytest.raises(CapsuleValidationError, match="pip install"):
        capsule_module.emit_run_capsule(
            tmp_path / "cert", emission_site="supervisor:success",
            certification_path=True, require_gpu=False)


def test_the_uncrippled_builder_emits_through_all_four_sites(tmp_path):
    for index, module in enumerate(_emitter_modules().values()):
        outdir = tmp_path / str(index)
        path = module.emit_run_capsule(
            outdir, emission_site="supervisor:success", require_gpu=False)
        assert path.name == CAPSULE_FILENAME
        document = load_certification_capsule(path)
        assert document["schema"] == CAPSULE_SCHEMA_ID
        assert tuple(sorted(document)) == tuple(sorted(REQUIRED_SECTIONS))


# --- D-14: content-mode geography on the certification path ---------------

def _certifiable_capsule():
    capsule = _stub_capsule()
    for entry in capsule["numerical_stack"].values():
        entry["status"] = "resolved"
        entry.pop("reason", None)
        if entry["value"] is None:
            entry["value"] = "recorded-by-the-test"
    return capsule


def test_the_certification_path_refuses_an_inventory_mode_directory_input():
    capsule = _certifiable_capsule()
    capsule["input_bytes"] = {"entries": {
        "geography:/data/geog": {"algorithm": "sha256-directory-inventory",
                                 "digest": "0" * 64},
    }}
    with pytest.raises(CapsuleValidationError, match="content"):
        validate_certification_capsule(capsule, certification_path=True)


def test_the_certification_path_accepts_a_content_mode_directory_input():
    capsule = _certifiable_capsule()
    capsule["input_bytes"] = {"entries": {
        "geography:/data/geog": {"algorithm": "sha256-directory-content",
                                 "digest": "0" * 64},
        "forcing:/data/f.nc": {"algorithm": "sha256", "digest": "1" * 64},
    }}
    validate_certification_capsule(capsule, certification_path=True)


# --- D-19: the manifest's two halves --------------------------------------

def test_the_schema_requires_the_cpu_half_and_states_the_gpu_half():
    entry = _schema()["properties"]["kernel_manifest"]["additionalProperties"]
    assert set(entry["required"]) == {"source_sha256", "options",
                                      "compiled_image"}
    image = entry["properties"]["compiled_image"]
    assert set(image["properties"]["status"]["enum"]) == {"resolved",
                                                          "unavailable"}
    assert image["properties"]["kind"]["enum"] == ["ptx", "cubin"]


# --- F3-AC3: the digest observes the trajectory, it does not join it -------

def test_the_digest_is_computed_after_the_final_sync_and_the_writer_drain():
    """Both execution owners finish I/O and sync before either state digest."""
    source = (REPO / "woof" / "runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef)}
    def digest_lines(body):
        return [node.lineno for node in ast.walk(body)
                if isinstance(node, ast.Call)
                and ((isinstance(node.func, ast.Name)
                      and node.func.id == "canonical_state_digest")
                     or (isinstance(node.func, ast.Attribute)
                         and node.func.attr == "canonical_digest"))]

    # run_experiment delegates ordinary singles to integrate_prepared_case,
    # and both adaptive singles and domain trees to _run_built_experiment.
    # Since the simulated-radar merge (development branch, e4d945f49, merged
    # at 794dafd16) the single arm hands integrate_prepared_case to
    # woof.simulated_radar.run_with_radar as its runner rather than calling
    # it by name.  The wrapper only calls it; its radar thread reads history
    # files the writer has already committed, never the device state, and
    # it computes no digest, so the digest is still integrate_prepared_case's
    # own, after that function's final sync.  The runner handed over counts
    # as delegated, and the wrapper is held to computing no digest.
    delegated = set()
    for node in ast.walk(functions["_run_experiment"]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            delegated.add(node.func.id)
            if (node.func.id == "run_with_radar" and node.args
                    and isinstance(node.args[0], ast.Name)):
                delegated.add(node.args[0].id)
    assert {"integrate_prepared_case", "_run_built_experiment"} <= delegated
    radar = ast.parse((REPO / "woof" / "simulated_radar.py").read_text(
        encoding="utf-8"))
    wrapper = next(node for node in ast.walk(radar)
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "run_with_radar")
    assert not digest_lines(wrapper)

    assert not digest_lines(functions["run_experiment"])
    assert not digest_lines(functions["_run_experiment"])
    for name in ("integrate_prepared_case", "_run_built_experiment"):
        body = functions[name]
        sync = [node.lineno for node in ast.walk(body)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "deviceSynchronize"]
        digest = digest_lines(body)
        assert sync and digest, name
        assert min(digest) > max(sync), (
            f"{name} computes the digest at line {min(digest)}, at or before "
            f"its final device synchronization at {max(sync)}")
    drain = [node.lineno for node in ast.walk(functions["_run_built_experiment"])
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "drain"]
    digest = digest_lines(functions["_run_built_experiment"])
    assert drain and min(digest) > max(drain)


def test_the_digest_instrumentation_can_be_switched_off_for_the_control_pair():
    from woof.runtime import TRAJECTORY_DIGEST_ENV, trajectory_digest_enabled

    assert trajectory_digest_enabled() is True
    import os

    previous = os.environ.get(TRAJECTORY_DIGEST_ENV)
    try:
        os.environ[TRAJECTORY_DIGEST_ENV] = "0"
        assert trajectory_digest_enabled() is False
        os.environ[TRAJECTORY_DIGEST_ENV] = "1"
        assert trajectory_digest_enabled() is True
    finally:
        if previous is None:
            os.environ.pop(TRAJECTORY_DIGEST_ENV, None)
        else:
            os.environ[TRAJECTORY_DIGEST_ENV] = previous


def test_both_run_summaries_carry_the_digest_field():
    from woof.runtime import ExperimentRunSummary, RealCaseRunSummary

    assert "trajectory_digest" in RealCaseRunSummary.__dataclass_fields__
    assert "trajectory_digest" in ExperimentRunSummary.__dataclass_fields__


def test_the_gpu_identity_uuid_is_bounded_to_the_sixteen_uuid_bytes():
    """The capsule's ``gpu_identity`` UUID is the 16 bytes that are the
    UUID -- the same over-read the FTZ probe fixed (``cudaDeviceProp``
    places ``luid`` right after ``uuid``, and CuPy's conversion stops at
    the first zero byte, not the array bound), bounded the same way."""
    from woof.certify.pins import CUDA_UUID_BYTES, device_uuid_hex

    uuid16 = bytes(range(1, 17))
    assert CUDA_UUID_BYTES == 16
    assert device_uuid_hex(uuid16) == uuid16.hex()
    # No trailing byte -- LUID or otherwise -- can reach the pin.
    for tail in (b"\x9e\x5d\x01", b"\xff" * 8, b"\x00" * 4):
        assert device_uuid_hex(uuid16 + tail) == uuid16.hex()
    for shape in (bytearray, memoryview):
        assert device_uuid_hex(shape(uuid16 + b"\x01\x02\x03")) \
            == uuid16.hex()
    # Absent property stays absent; an already-str value passes through.
    assert device_uuid_hex(None) is None
    assert device_uuid_hex("GPU-abc") == "GPU-abc"


def test_cupy_pin_resolves_whichever_wheel_the_box_installed(monkeypatch):
    """``cupy_version`` names the box's CuPy, not the CUDA-12 wheel.

    The pin used to look up the literal distribution ``cupy-cuda12x``,
    so a CUDA-13 box -- the box the ``gpu-cu13`` extra exists for --
    certified with ``cupy_version`` unavailable while running CuPy
    kernels.  Each spelling pip can resolve must satisfy the pin, and
    the value stays a bare version string because the FTZ receipt and
    its claim-site anchor record it as one.
    """
    import importlib.metadata as im

    from woof.certify import pins as pins_module

    class _Dist:
        def __init__(self, name, version):
            self.metadata = {"Name": name}
            self.version = version

    for name in ("cupy-cuda13x", "cupy-cuda12x", "cupy"):
        monkeypatch.setattr(
            im, "distributions", lambda name=name: iter([
                _Dist("numpy", "2.1.0"), _Dist(name, "14.0.1")]))
        assert pins_module._installed_cupy_version() == "14.0.1", name

    # Several at once (a broken but observable pip state): deterministic,
    # highest-sorting wheel name, never an exception.
    monkeypatch.setattr(im, "distributions", lambda: iter([
        _Dist("cupy-cuda12x", "13.6.0"), _Dist("cupy-cuda13x", "14.0.1")]))
    assert pins_module._installed_cupy_version() == "14.0.1"

    # No CuPy at all: _package_pins records unavailable with the reason,
    # exactly as the old lookup did on a CuPy-less box.
    monkeypatch.setattr(im, "distributions", lambda: iter([
        _Dist("numpy", "2.1.0")]))
    entries = {}
    pins_module._package_pins(entries)
    assert entries["cupy_version"]["status"] == "unavailable"
    assert "no CuPy distribution" in entries["cupy_version"]["reason"]
