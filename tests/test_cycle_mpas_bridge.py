"""The bridge's contract, tested without a GPU.

Every test here is about the *structure* that makes the closed loop
possible, not about the physics: that the forecast process cannot import
the cycling spine, that the anchor is the only channel, and that a
boundary is stamped from evidence rather than from optimism.
"""

from __future__ import annotations

import json
import shutil
import site
import subprocess
import sys
import sysconfig
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

from woof.cycle import mpas_bridge
from woof.cycle.anchor import prognostic_sha256, read_anchor, write_anchor
from mpas_cycle_bridge import anchor_codec as codec


# --------------------------------------------------------------------------
# rule 1: the bridge package carries no woof import, provably


def test_bridge_package_is_gpuwm_free():
    receipt = mpas_bridge.verify_bridge_purity()
    assert receipt["gpuwm_imports"] == 0
    assert "worker.py" in receipt["modules_scanned"]
    assert "portbind.py" in receipt["modules_scanned"]
    assert "anchor_codec.py" in receipt["modules_scanned"]


def _bridge_copy(tmp_path, monkeypatch) -> Path:
    """A private copy of the bridge package for the purity check to read.

    The intruder tests used to write their intruder into the installed
    package itself, so a parallel test in another worker that launched the
    bridge in that window was refused for a module it never had (2.9.10
    pass 3 R5, test_cycle_port_package[mpas_port]: offences
    ``_test_intruder2.py``).  verify_bridge_purity reads the package beside
    ``mpas_bridge.mpas_cycle_bridge.__file__``, so pointing that at a copy
    keeps the real package untouched.
    """
    real = Path(mpas_bridge.mpas_cycle_bridge.__file__).resolve().parent
    copy = tmp_path / "mpas_cycle_bridge"
    shutil.copytree(real, copy, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(mpas_bridge, "mpas_cycle_bridge", types.SimpleNamespace(
        __file__=str(copy / "__init__.py")))
    return copy


def test_purity_check_refuses_an_innocent_spine_import(tmp_path, monkeypatch):
    """RED ON REVERT: add the import a contributor would add, get refused.

    This is the failure mode the whole bridge exists to prevent -- a
    helpful ``from woof.cycle.anchor import read_anchor`` inside the
    worker, which claims the woof name before the port can pin its
    frozen Arwen and makes the port refuse eight frames deep.
    """
    package = _bridge_copy(tmp_path, monkeypatch)
    intruder = package / "_test_intruder.py"
    intruder.write_text(
        "from woof.cycle.anchor import read_anchor\n", encoding="utf-8")
    try:
        with pytest.raises(mpas_bridge.BridgeRefusal) as excinfo:
            mpas_bridge.verify_bridge_purity()
    finally:
        intruder.unlink()
    offences = excinfo.value.observed["offences"]
    assert offences[0]["module"] == "_test_intruder.py"
    assert offences[0]["imports"] == "woof.cycle.anchor"
    # And it is clean again the moment the intruder is gone.
    assert mpas_bridge.verify_bridge_purity()["gpuwm_imports"] == 0


def test_a_plain_gpuwm_import_is_caught_too(tmp_path, monkeypatch):
    package = _bridge_copy(tmp_path, monkeypatch)
    intruder = package / "_test_intruder2.py"
    intruder.write_text("import woof\n", encoding="utf-8")
    try:
        with pytest.raises(mpas_bridge.BridgeRefusal):
            mpas_bridge.verify_bridge_purity()
    finally:
        intruder.unlink()


# --------------------------------------------------------------------------
# rule 2: the runtime guard refuses the spine inside a forecast process


GUARD_PROBE = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, {root!r})
    from mpas_cycle_bridge.spine_guard import install_guard, SpineImportRefused

    guard = install_guard()
    out = {{}}
    for name in ("woof.cycle.anchor", "woof.cycle", "woof"):
        try:
            __import__(name)
            out[name] = "IMPORTED"
        except SpineImportRefused as error:
            out[name] = "REFUSED: " + str(error)[:80]
        except ImportError as error:
            out[name] = "IMPORTERROR: " + str(error)[:80]
    # Arming lets the pinned tree through the top-level gate, but never
    # the spine namespace.
    guard.arm({root!r})
    try:
        __import__("woof.cycle.anchor")
        out["armed:woof.cycle.anchor"] = "IMPORTED"
    except SpineImportRefused as error:
        out["armed:woof.cycle.anchor"] = "REFUSED: " + str(error)
    print(json.dumps(out))
""")


def _probe(source: str) -> dict:
    env = mpas_bridge.child_environment()
    completed = subprocess.run([sys.executable, "-c", source], env=env,
                               cwd=str(mpas_bridge.bridge_root()),
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_guard_refuses_the_spine_even_with_the_repo_on_the_path():
    """The spine tree is importable and the guard refuses it anyway.

    The probe deliberately puts the repository root on ``sys.path`` --
    the exact thing that makes ``woof.cycle`` resolvable -- so this
    tests the guard, not the absence of the tree.
    """
    result = _probe(GUARD_PROBE.format(root=str(mpas_bridge.bridge_root())))
    assert result["woof.cycle.anchor"].startswith("REFUSED")
    assert result["woof.cycle"].startswith("REFUSED")
    assert result["woof"].startswith("REFUSED")
    # Once armed, the top-level gate opens for the pinned tree -- and the
    # spine namespace is STILL refused, which is the rule that matters:
    # it is not a startup ordering trick, it holds for the whole run.
    armed = result["armed:woof.cycle.anchor"]
    assert armed.startswith("REFUSED")
    assert "cycling spine" in armed
    assert "anchor on disk" in armed


def test_guard_refuses_installation_when_gpuwm_is_already_live():
    source = textwrap.dedent(f"""
        import json, sys
        sys.path.insert(0, {str(mpas_bridge.bridge_root())!r})
        import woof.cycle.anchor  # the damage, done before the guard
        from mpas_cycle_bridge.spine_guard import (install_guard,
                                                   SpineImportRefused)
        try:
            install_guard()
            print(json.dumps({{"result": "INSTALLED"}}))
        except SpineImportRefused as error:
            print(json.dumps({{"result": "REFUSED", "why": str(error)[:120]}}))
    """)
    result = _probe(source)
    assert result["result"] == "REFUSED"
    assert "already imported" in result["why"]


# --------------------------------------------------------------------------
# rule 2b: the bootstrap's scrub is free from a checkout and must not fire
# from an installation


def _running_library_dirs() -> list[str]:
    """Where THIS interpreter keeps its installed distributions.

    Asked of the interpreter rather than guessed from the path's shape,
    the same way the bootstrap under test asks.  Used to give a throwaway
    environment the dependencies the worker imports at module scope; the
    test below says why inheriting the base interpreter's library is not
    enough.
    """

    found: set[str] = set()
    paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        entry = paths.get(key)
        if entry:
            found.add(str(Path(entry).resolve()))
    getter = getattr(site, "getsitepackages", None)
    if getter is not None:
        try:
            found.update(str(Path(entry).resolve()) for entry in getter())
        except Exception:         # site disabled (-S): nothing to report
            pass
    return sorted(found)


BOOTSTRAP_PROBE = textwrap.dedent("""
    import json, runpy, sys
    module = runpy.run_path({worker!r})
    report = {{
        "root": str(module["_REPO_ROOT"]),
        "root_is_library": module["_ROOT_IS_LIBRARY"],
        "root_on_path": str(module["_REPO_ROOT"]) in sys.path,
    }}
    {extra}
    print(json.dumps(report))
""")


def test_a_checkout_bridge_still_drops_its_repository_root():
    """From a checkout the scrub costs nothing, so it still happens.

    The root of a checkout holds the live spine tree and nothing the
    forecast needs, so dropping it makes the spine unreachable as well as
    refused.  That half of the bootstrap is unchanged and this pins it.
    """
    probe = BOOTSTRAP_PROBE.format(worker=str(mpas_bridge.worker_path()),
                                   extra="")
    result = _probe(probe)
    assert result["root_is_library"] is False
    assert result["root_on_path"] is False


def test_an_installed_bridge_keeps_its_library_directory_on_the_path(tmp_path):
    """Installed, the root IS site-packages and the scrub must not fire.

    The breakage: the bootstrap dropped its own root unconditionally, so
    an installed worker dropped site-packages and took every other
    installed distribution with it.  The pinned Arwen tree imports cupy
    at module scope, so the worker died "No module named 'cupy'" at the
    seed leg of cycle 1 while the same commit run from a checkout was
    fine -- numpy, already in sys.modules by that line, hid it.  Here the
    bridge is installed into a throwaway environment beside a stand-in
    dependency, and the stand-in must still import once the bootstrap has
    run.
    """
    environment = tmp_path / "env"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip",
                    str(environment)],
                   check=True, capture_output=True, timeout=600)
    libraries = sorted(environment.glob("**/site-packages"))
    assert len(libraries) == 1, libraries
    library = libraries[0]
    shutil.copytree(mpas_bridge.worker_path().parent,
                    library / "mpas_cycle_bridge")
    (library / "stand_in_dependency.py").write_text(
        "VALUE = 'installed beside the bridge'\n", encoding="utf-8")
    # The worker imports numpy at module scope, by way of anchor_codec, so
    # the throwaway environment has to resolve it.  ``--system-site-packages``
    # cannot supply it: a venv's base is the interpreter it was BUILT from,
    # so a child venv inherits the base's library while numpy normally lives
    # in the running venv.  Measured on a uv-provisioned 3.11: the child's
    # sys.path named the base interpreter's site-packages and not this one,
    # and the probe died "No module named 'numpy'" -- so the assertion below
    # passed only on a box whose base interpreter happened to carry numpy.
    # A .pth names this interpreter's own library directories instead; site
    # reads it even under PYTHONSAFEPATH, and its entries land after the
    # throwaway library, so every assertion below still measures whether the
    # bootstrap kept its own root.
    (library / "_running_interpreter.pth").write_text(
        "\n".join(_running_library_dirs()) + "\n", encoding="utf-8")
    python = environment / "Scripts" / "python.exe"
    if not python.exists():
        python = environment / "bin" / "python"

    probe = BOOTSTRAP_PROBE.format(
        worker=str(library / "mpas_cycle_bridge" / "worker.py"),
        extra=("import stand_in_dependency\n"
               "report['dependency'] = stand_in_dependency.VALUE"))
    completed = subprocess.run([str(python), "-c", probe],
                               env=mpas_bridge.child_environment(),
                               capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["root"] == str(library)
    assert result["root_is_library"] is True
    assert result["root_on_path"] is True
    assert result["dependency"] == "installed beside the bridge"


# --------------------------------------------------------------------------
# rule 3: the child environment cannot reach the spine tree


def test_child_environment_never_forwards_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(mpas_bridge.bridge_root()))
    env = mpas_bridge.child_environment()
    assert "PYTHONPATH" not in env
    assert env["PYTHONSAFEPATH"] == "1"


def test_child_environment_keeps_the_cuda_variables(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SOME_PERSONAL_SHELL_THING", "yes")
    env = mpas_bridge.child_environment()
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "SOME_PERSONAL_SHELL_THING" not in env


# --------------------------------------------------------------------------
# the anchor is the only channel: one encoding, both sides


def test_codec_and_spine_agree_on_the_state_hash(tmp_path):
    """The worker and the spine must hash a state identically.

    They do because they run the same function -- this test is what
    keeps that true if somebody re-implements one of them.
    """
    rng = np.random.default_rng(11)
    prognostic = {
        "rho": rng.random((4, 9), dtype=np.float32),
        "rho_theta": rng.random((4, 9), dtype=np.float32),
        "rho_u": rng.random((4, 13), dtype=np.float32),
        "rho_w": rng.random((5, 9), dtype=np.float32),
        "scalars": rng.random((3, 4, 9), dtype=np.float32),
    }
    assert prognostic_sha256(prognostic) == codec.prognostic_sha256(prognostic)

    path = tmp_path / f"round-trip{codec.SUFFIX[codec.ARRAY_FORMAT]}"
    codec.write_arrays(path, prognostic)
    assert codec.prognostic_sha256(codec.read_arrays(path)) == \
        codec.prognostic_sha256(prognostic)


def test_worker_reads_a_spine_written_anchor_without_gpuwm(tmp_path):
    """Cross-lane, in the strict sense: the reader is not the writer.

    The spine writes the anchor with ``write_anchor``; the codec -- the
    only anchor code the forecast process is allowed to run -- reads it
    back and must see the same state, the same sidecar, the same seam
    and the same increment.
    """
    rng = np.random.default_rng(5)
    prognostic = {name: rng.random(shape, dtype=np.float32) for name, shape in
                  (("rho", (4, 9)), ("rho_theta", (4, 9)), ("rho_u", (4, 13)),
                   ("rho_w", (5, 9)), ("scalars", (3, 4, 9)))}
    derived = {name: rng.random((4, 9), dtype=np.float32) for name in
               ("theta_m", "exner", "density_perturbation",
                "rho_theta_perturbation", "pressure_perturbation",
                "normal_velocity", "vertical_velocity")}
    seam = {"backend_field": rng.random((4, 9), dtype=np.float32)}
    increment = {"rho_theta": np.full((4, 9), 0.25, dtype=np.float32)}

    published = write_anchor(
        tmp_path, cycle_index=1, anchor_ticks=1000,
        valid_time="2026-08-12T06:00:00Z", parent_kind="mpas-cuda-frames",
        prognostic=prognostic, derived=derived, seam=seam, mesh_id="x1.test",
        analysis={"state": "PENDING", "arrays": increment})

    manifest = codec.read_manifest(published)
    assert manifest["parent"]["prognostic_sha256"] == \
        prognostic_sha256(prognostic)
    back = codec.read_member(published, manifest["parent"]["path"])
    assert codec.prognostic_sha256(back) == prognostic_sha256(prognostic)
    assert set(codec.read_member(published, manifest["derived"]["path"])) == \
        set(derived)
    assert set(codec.read_member(published,
                                 manifest["seam"]["path"])) == set(seam)
    got = codec.read_member(published, manifest["analysis"]["path"])
    np.testing.assert_array_equal(got["rho_theta"], increment["rho_theta"])
    # And the spine's own reader agrees, so the two views cannot drift.
    assert read_anchor(published).manifest["parent"]["prognostic_sha256"] == \
        manifest["parent"]["prognostic_sha256"]


def test_codec_refuses_an_uncommitted_anchor(tmp_path):
    staging = tmp_path / "anchors" / "anchor_001_half.tmp"
    staging.mkdir(parents=True)
    (staging / "anchor.json").write_text("{}", encoding="utf-8")
    with pytest.raises(codec.CodecRefusal) as excinfo:
        codec.read_manifest(staging)
    assert "COMMIT" in str(excinfo.value)


def test_codec_refuses_a_tampered_anchor(tmp_path):
    rng = np.random.default_rng(7)
    prognostic = {name: rng.random(shape, dtype=np.float32) for name, shape in
                  (("rho", (2, 3)), ("rho_theta", (2, 3)), ("rho_u", (2, 4)),
                   ("rho_w", (3, 3)), ("scalars", (1, 2, 3)))}
    derived = {"theta_m": rng.random((2, 3), dtype=np.float32)}
    published = write_anchor(
        tmp_path, cycle_index=1, anchor_ticks=1000,
        valid_time="2026-08-12T06:00:00Z", parent_kind="mpas-cuda-frames",
        prognostic=prognostic, derived=derived, mesh_id="x1.test")
    victim = published / codec.MANIFEST_NAME
    victim.write_text(victim.read_text(encoding="utf-8") + "\n",
                      encoding="utf-8")
    with pytest.raises(codec.CodecRefusal) as excinfo:
        codec.read_manifest(published)
    assert "hash" in str(excinfo.value)


# --------------------------------------------------------------------------
# the stamp is graded from evidence, on the spine side


def _complete_manifest(**overrides):
    manifest = {
        "steps_executed": 8,
        "rehydration": "_construct_device_stack(state=, saved_diagnostics=, "
                       "backend_restart=)",
        "rehydrated_sha256": "a" * 64,
        "analysed_sha256": "a" * 64,
        "arwen_pin": {"obeyed": True},
    }
    manifest.update(overrides)
    return manifest


def test_a_complete_segment_earns_the_closed_stamp():
    verdict = mpas_bridge.stamp_for_segment(_complete_manifest(),
                                            steps_requested=8)
    assert verdict["parent_kind"] == mpas_bridge.CLOSED_KIND
    assert verdict["evidence_gaps"] == []
    assert verdict["analysis_reentered_dycore"] is True


@pytest.mark.parametrize('overrides, needle', [
    pytest.param({'rehydrated_sha256': 'b' * 64}, 'did NOT re-enter the dycore', id='overrides5-did NOT re-enter the dycore'),
])
def test_missing_evidence_downgrades_the_stamp(overrides, needle):
    """A stamp that overstates is worse than a missing feature.

    Each of these is a way a run can look finished and not be one, and
    each must produce ``mpas-cuda-frames`` -- the accurate label -- with
    the gap named.
    """
    verdict = mpas_bridge.stamp_for_segment(_complete_manifest(**overrides),
                                            steps_requested=8)
    assert verdict["parent_kind"] == mpas_bridge.FRAMES_KIND
    assert verdict["analysis_reentered_dycore"] is False
    assert any(needle in gap for gap in verdict["evidence_gaps"]), \
        verdict["evidence_gaps"]


def test_a_crashed_arm_cannot_earn_a_stamp():
    """The 0.22-second-crash shape, refused.

    Two arms once crashed in a fraction of a second, differed slightly,
    sailed past an exact-0.0 guard and published ``speedup: 1.0073`` with
    ``meets_gate: true``.  Divergence is not evidence.  An arm with no
    step receipts gets nothing.
    """
    verdict = mpas_bridge.stamp_for_segment(
        {"steps_executed": None, "arwen_pin": {}}, steps_requested=8)
    assert verdict["parent_kind"] == mpas_bridge.FRAMES_KIND
    assert len(verdict["evidence_gaps"]) >= 3


# --------------------------------------------------------------------------
# launching refuses loudly


def test_launch_refuses_and_names_the_log_when_the_worker_dies(tmp_path):
    with pytest.raises(mpas_bridge.BridgeRefusal) as excinfo:
        mpas_bridge.launch(phase="seed", port_root=tmp_path / "nowhere",
                           port_config=tmp_path / "nothing.json", steps=1,
                           out=tmp_path / "out", timeout=300)
    observed = excinfo.value.observed
    assert Path(observed["log"]).is_file()
    assert observed["returncode"] != 0
    assert "worker.log" in observed["log"]


def test_segment_reader_refuses_a_tampered_segment(tmp_path):
    root = tmp_path / "seg"
    root.mkdir()
    payload = root / f"parent_prognostic{codec.SUFFIX[codec.ARRAY_FORMAT]}"
    codec.write_arrays(payload, {"rho": np.zeros((2, 2), dtype=np.float32)})
    (root / "segment.json").write_text(json.dumps({
        "files": {payload.name: "0" * 64}}) + "\n", encoding="utf-8")
    with pytest.raises(mpas_bridge.BridgeRefusal) as excinfo:
        mpas_bridge.read_segment(root)
    assert "hash" in str(excinfo.value)
