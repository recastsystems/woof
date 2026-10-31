# Replaces: test_standalone_native_smoke_forecast_annotation_still_guards_staging (now test_standalone_forecast_supervisor_annotation_still_guards_staging).
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

from woof import __version__
from woof.gfs_direct import _git_source_identity
from woof.native_wrf_distribution import (
    BRIDGE_NAMES,
    BRIDGE_WORKSPACES,
    BUNDLED_BRIDGES,
    CPU_BACKEND_LIBRARY,
    CUDA_KERNEL_SOURCES,
    HRRR_HELPERS,
    PYTHON_DISTRIBUTION,
    RUNTIME_SCHEMA,
    WINDOWS_CPU_BACKEND_LIBRARY,
    _BRIDGE_ABI_MARKERS,
    _BRIDGE_USAGE_MARKERS,
    _cpu_backend_self_test,
    _numeric_version,
    _runtime_dependency_modules,
    add_bridge_options,
    bridge_identity,
    bridge_inputs,
    cpu_backend_library_name,
    distribution_contract,
)
from tools.build_native_wrf_distribution import (
    FORBIDDEN_WHEEL_PAYLOADS,
    _wheel_public_path_violations,
    _write_deterministic_tar,
)
from tools.build_rw_wps_release import (
    NotAGitCheckout,
    _cargo_release_environment,
    _run as _run_release_subprocess,
    _stage_rw_wps_python_project,
    _staged_internal_imports,
    _staged_verification_imports,
)


from tools.build_native_wrf_windows_distribution import (
    _dynamic_msvc_runtime_imports,
    _require_static_msvc_runtime,
    _write_deterministic_zip,
)
from tools.smoke_rw_wps_cpu_install import _safe_extract, _safe_extract_zip


ROOT = Path(__file__).parents[1]


def _run_python(*arguments, **options):
    return subprocess.run([sys.executable, *arguments], capture_output=True,
                          text=True, check=False, **options)


def _stage_or_skip(destination):
    """Stage the standalone project, or say why the tree cannot.

    Staging copies tracked files only, so a tree with no git index
    cannot be staged at all -- the question has no answer there.  That
    is a property of the directory the suite is running in, not a
    finding about the product, and it must not read as one: the Linux
    pre-cut gate ran against a `git archive` extraction and got a
    subprocess traceback ending in `exit status 128`, which looked like
    a staging bug for as long as it took to read the command.

    CI's own checkout is a clone, so this never skips there.
    """

    try:
        return _stage_rw_wps_python_project(destination)
    except NotAGitCheckout as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def staged_project(tmp_path_factory):
    """The standalone project, staged once for the module (once per
    pytest-xdist worker), as ``(path, receipt)``.

    Staging copies and import-scans about 735 tracked files, 3 s on an
    idle desktop and longer on a loaded battery host, and fifteen tests
    here used to stage a copy each to read it.  They read this one.  A
    test that writes into the staged tree takes ``staged_copy``; the two
    tests whose subject is a refused staging call the real staging.
    """

    destination = tmp_path_factory.mktemp("rw-wps-staging") / "rw-wps-python"
    receipt = _stage_or_skip(destination)
    return destination, receipt


@pytest.fixture
def staged_copy(staged_project, tmp_path):
    """A private copy of the shared staging, for a test that writes into it."""

    source, receipt = staged_project
    destination = tmp_path / "rw-wps-python"
    shutil.copytree(source, destination,
                    ignore=shutil.ignore_patterns("__pycache__"))
    return destination, receipt


def test_distribution_builder_runs_directly_outside_checkout(tmp_path):
    completed = _run_python(str(ROOT / 'tools' / 'build_native_wrf_distribution.py'), '--help', cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert "Build a hash-bound Linux woof" in completed.stdout
    assert "native-WRF runtime distribution." in completed.stdout


@pytest.mark.parametrize(
    ("script", "marker"),
    (
        ("build_rw_wps_release.py", "complete Linux RW-WPS runtime"),
        ("smoke_rw_wps_cpu_install.py", "CPU backend without CUDA"),
    ),
)
def test_standalone_release_tools_run_directly_outside_checkout(
    tmp_path, script, marker,
):
    completed = _run_python(str(ROOT / 'tools' / script), '--help', cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert marker in completed.stdout


def test_release_builder_routes_child_progress_away_from_receipt_stdout(
        monkeypatch):
    observed = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "tools.build_rw_wps_release.subprocess.run", fake_run)
    _run_release_subprocess(["builder", "--progress"])

    assert observed["argv"] == ["builder", "--progress"]
    assert observed["check"] is True
    assert observed["stdout"] is sys.stderr
    assert observed["stderr"] is sys.stderr


def test_release_builder_stdout_is_one_strict_json_document(
        tmp_path, monkeypatch, capsys):
    import tools.build_rw_wps_release as builder

    def fake_build(_args):
        print("pip/cargo progress", file=sys.stderr)
        return {"schema": "rw-wps-build-test-v1", "status": "PASS"}

    monkeypatch.setattr(builder, "build_release", fake_build)
    assert builder.main([
        "--output-dir", str(tmp_path / "runtime"),
        "--archive", str(tmp_path / "runtime.tar.gz"),
    ]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "schema": "rw-wps-build-test-v1", "status": "PASS"}
    assert "pip/cargo progress" in captured.err


def test_windows_release_tool_runs_directly_outside_checkout(tmp_path):
    completed = _run_python(str(ROOT / 'tools' / 'build_rw_wps_windows_release.py'), '--help', cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert "complete Windows x86_64 CPU RW-WPS runtime" in completed.stdout


def test_clean_install_smoke_rejects_archive_traversal(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    payload = b"owned"
    with tarfile.open(archive, "w:gz") as output:
        info = tarfile.TarInfo("rw-wps/../../outside")
        info.size = len(payload)
        output.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe archive path"):
        _safe_extract(archive, tmp_path / "extract")
    assert not (tmp_path / "outside").exists()


def test_clean_install_smoke_rejects_zip_traversal_and_case_alias(tmp_path):
    traversal = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(traversal, "w") as output:
        output.writestr("rw-wps/../../outside", "owned")
    with pytest.raises(ValueError, match="unsafe archive path"):
        _safe_extract_zip(traversal, tmp_path / "extract-traversal")
    assert not (tmp_path / "outside").exists()

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as output:
        output.writestr("rw-wps/File.txt", "first")
        output.writestr("rw-wps/file.txt", "second")
    with pytest.raises(ValueError, match="duplicate archive path"):
        _safe_extract_zip(duplicate, tmp_path / "extract-duplicate")


def test_release_archive_is_byte_identical_across_build_roots(tmp_path):
    roots = []
    archives = []
    for index in range(2):
        source = tmp_path / f"build-{index}" / "rw-wps-runtime"
        (source / "bin").mkdir(parents=True)
        executable = source / "bin" / "rw-wps"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        (source / "manifest.json").write_text(
            '{"status":"READY"}\n', encoding="utf-8")
        # Filesystem clocks and parent locations are explicitly outside the
        # archive identity.
        os.utime(source / "manifest.json", (1000 + index, 2000 + index))
        archive = tmp_path / f"result-{index}" / "rw-wps-runtime.tar.gz"
        archive.parent.mkdir()
        _write_deterministic_tar(source, archive)
        roots.append(source)
        archives.append(archive)
    assert roots[0].parent != roots[1].parent
    assert archives[0].read_bytes() == archives[1].read_bytes()


def test_windows_release_zip_is_byte_identical_across_build_roots(tmp_path):
    archives = []
    for index in range(2):
        source = tmp_path / f"windows-build-{index}" / "rw-wps-runtime"
        (source / "bin").mkdir(parents=True)
        (source / "bin" / "rw-wps.cmd").write_bytes(
            b"@echo off\r\nexit /b 0\r\n")
        (source / "manifest.json").write_text(
            '{"status":"READY"}\n', encoding="utf-8")
        os.utime(source / "manifest.json", (1000 + index, 2000 + index))
        archive = tmp_path / f"windows-result-{index}" / "rw-wps-runtime.zip"
        archive.parent.mkdir()
        _write_deterministic_zip(source, archive)
        archives.append(archive)
    assert archives[0].read_bytes() == archives[1].read_bytes()


def test_windows_runtime_rejects_dynamic_msvc_crt_markers(tmp_path):
    static = tmp_path / "static.exe"
    static.write_bytes(b"MZ\x00KERNEL32.dll\x00")
    assert _dynamic_msvc_runtime_imports(static) == []
    assert _require_static_msvc_runtime(static) == []

    dynamic = tmp_path / "dynamic.exe"
    dynamic.write_bytes(b"MZ\x00VCRUNTIME140.dll\x00")
    assert _dynamic_msvc_runtime_imports(dynamic) == ["vcruntime140.dll"]
    with pytest.raises(RuntimeError, match="imports dynamic MSVC CRT"):
        _require_static_msvc_runtime(dynamic)


def test_rust_release_build_remaps_checkout_and_ignores_host_flags(
    tmp_path, monkeypatch,
):
    source = tmp_path / "checkout with spaces"
    target = tmp_path / "cargo-target"
    monkeypatch.setenv("RUSTFLAGS", "-C target-cpu=native")
    monkeypatch.setenv("CARGO_ENCODED_RUSTFLAGS", "caller-specific")

    environment = _cargo_release_environment(
        source_root=source,
        target_dir=target,
        source_date_epoch="1784709589",
    )

    assert "RUSTFLAGS" not in environment
    flags = environment["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
    assert flags[0] == (
        f"--remap-path-prefix={source.resolve()}=/usr/src/rw-wps"
    )
    if sys.platform == "win32":
        assert flags[1:] == [
            "-C", "link-arg=/Brepro",
            "-C", "target-feature=+crt-static",
        ]
    else:
        assert flags == [flags[0]]
    assert environment["CARGO_TARGET_DIR"] == str(target)
    assert environment["SOURCE_DATE_EPOCH"] == "1784709589"


def test_native_wrf_contract_is_versioned_and_explicit():
    contract = distribution_contract("linux-x86_64")
    windows_contract = distribution_contract("windows-x86_64")
    assert contract["schema"] == RUNTIME_SCHEMA
    assert contract["gpuwm_version"] == __version__
    assert contract["python_distribution"] == PYTHON_DISTRIBUTION == "rw-wps"
    assert contract["runtime_forbidden"] == ["WPS", "real.exe"]
    assert contract["platform"]["python"] == ">=3.11"
    assert contract["platform"]["cuda_runtime_family"] == "12.x"
    assert contract["platform"]["identifier"] == "linux-x86_64"
    assert windows_contract["platform"] == {
        "identifier": "windows-x86_64",
        **windows_contract["supported_platforms"]["windows-x86_64"],
    }
    assert windows_contract["platform"]["operating_system"] == "Windows"
    assert windows_contract["platform"]["shell"] == "PowerShell 5.1+"
    assert windows_contract["platform"]["cuda_runtime_family"] is None
    assert windows_contract["platform"]["msvc_runtime"] == "statically linked"
    assert contract["public_controls"]["hrrr"]["cadence_seconds"] == 3600
    assert "seal_hrrr_native_bridge.py" in HRRR_HELPERS
    assert "prepare_hrrr_wrf.py" in HRRR_HELPERS
    assert "download_hrrr_native_subset.py" in HRRR_HELPERS
    assert "common.cuh" in CUDA_KERNEL_SOURCES
    assert "diagnostics.cu" in CUDA_KERNEL_SOURCES
    assert "vert_interp.cu" in CUDA_KERNEL_SOURCES
    assert "thompson.cu" not in CUDA_KERNEL_SOURCES
    assert "dycore.cu" not in CUDA_KERNEL_SOURCES
    assert contract["bundled_native_libraries"] == [CPU_BACKEND_LIBRARY]
    assert windows_contract["bundled_native_libraries"] == [
        WINDOWS_CPU_BACKEND_LIBRARY]
    assert contract["bundled_native_libraries_by_platform"][
        "windows-x86_64"] == WINDOWS_CPU_BACKEND_LIBRARY
    assert contract["supported_platforms"]["windows-x86_64"][
        "backends"] == ["cpu"]
    assert contract["bundled_native_bridges"] == list(BRIDGE_NAMES)
    assert "grib2_inventory" in BRIDGE_NAMES
    assert "grib2_dump" in BRIDGE_NAMES
    assert contract["preprocess_backends"]["cpu"] \
        == "rust-parallel-fp32-v1"
    assert "parallel CPU" in contract["public_controls"]["gfs"]["preprocessing"]
    assert "parallel CPU" in contract["public_controls"]["era5"]["preprocessing"]
    assert "max_dom=4" in contract["public_controls"]["20crv3"]["domain"]
    assert "packaged" in contract["public_controls"]["20crv3"]["authorities"]
    assert "d01 through d04" in (
        contract["public_controls"]["mapped"]["real_gates"]
    )
    assert "max_dom" in contract["public_controls"]["mapped"]["domain"]
    assert "explicit" in contract["public_controls"]["hrrr"]["domain"]
    assert "explicit" in contract["public_controls"]["gfs"]["domain"]
    assert "explicit" in contract["public_controls"]["era5"]["domain"]
    assert "frozen 49" not in json.dumps(contract["public_controls"])


def test_runtime_selects_platform_cpu_library_names():
    assert cpu_backend_library_name("Linux") == CPU_BACKEND_LIBRARY
    assert cpu_backend_library_name("Windows") == WINDOWS_CPU_BACKEND_LIBRARY
    with pytest.raises(RuntimeError, match="unsupported"):
        cpu_backend_library_name("Plan9")


def test_installed_launcher_skips_device_gate_for_cpu_and_nonexecuting_modes():
    launcher = (
        Path(__file__).parents[1] / "tools" / "gpuwm_native_wrf_launcher.sh"
    ).read_text(encoding="utf-8")
    assert "--author-only|--dry-run" in launcher
    assert "--list-sources" in launcher
    assert "--show-source=*" in launcher
    assert "--show-support-matrix" in launcher
    assert "--namelist-support-report" in launcher
    assert "--preprocess-backend=cpu|--preprocess-backend=auto" in launcher
    assert "cpu|auto) skip_gpu=1" in launcher
    assert "runtime_check+=(--skip-gpu)" in launcher


def test_installed_launcher_keeps_the_entire_python_chain_immutable():
    launcher = (
        Path(__file__).parents[1] / "tools" / "gpuwm_native_wrf_launcher.sh"
    ).read_text(encoding="utf-8")
    no_user_site = "export PYTHONNOUSERSITE=1"
    no_bytecode = "export PYTHONDONTWRITEBYTECODE=1"
    first_python = 'PYTHONPATH="$runtime" "$python_path" -P -m'

    assert no_user_site in launcher
    assert no_bytecode in launcher
    assert launcher.index(no_user_site) < launcher.index(first_python)
    assert launcher.index(no_bytecode) < launcher.index(first_python)


def test_installer_and_runtime_support_explicit_no_gpu_authoring_mode():
    installer = (
        Path(__file__).parents[1] / "tools" / "install_gpuwm_native_wrf.sh"
    ).read_text(encoding="utf-8")
    assert "--skip-gpu" in installer
    assert "-name 'rw_wps-*.whl'" in installer
    assert "gpuwm-*.whl" not in installer
    assert "runtime_check+=(--skip-gpu)" in installer
    assert "cupy" not in _runtime_dependency_modules(False)
    assert "matplotlib" not in _runtime_dependency_modules(False)
    assert _runtime_dependency_modules(True)["cupy-cuda12x"] == "cupy"


def test_windows_installer_and_launcher_are_fail_closed():
    installer = (ROOT / "tools" / "install_gpuwm_native_wrf_windows.ps1").read_text(
        encoding="utf-8")
    launcher = (ROOT / "tools" / "gpuwm_native_wrf_launcher_windows.ps1").read_text(
        encoding="utf-8")
    for marker in (
        "Get-FileHash", "ReparsePoint", "Duplicate SHA256SUMS path",
        "Refusing existing WOOF_INSTALL_ROOT", "--no-index", "--no-deps",
        "--skip-gpu", '".rw-{0}-{1}"', "Substring(0, 8)",
        "projected path length", "[System.IO.Directory]::Move",
        "importlib.util.cache_from_source",
        "str(sys.flags.optimize)",
        "$cacheProjectionScript | & $python -",
        "[System.Math]::Max($partial.Length, $target.Length)",
    ):
        assert marker in installer
    assert "& $python -c" not in installer
    assert ".rw-wps-runtime.partial-" not in installer
    assert "Move-Item -LiteralPath $partial" not in installer
    for marker in (
        "Get-FileHash", "ReparsePoint", "WOOF_PYTHON differs",
        "--preprocess-backend=cuda", "gpuwm_preprocess_cpu.dll",
        "--skip-gpu", '$env:PYTHONNOUSERSITE = "1"',
        '$env:PYTHONDONTWRITEBYTECODE = "1"',
    ):
        assert marker in launcher
    first_python = launcher.index("& $python -B -P -m")
    assert launcher.index('$env:PYTHONNOUSERSITE = "1"') < first_python
    assert launcher.index('$env:PYTHONDONTWRITEBYTECODE = "1"') < first_python
    assert launcher.count("& $python -B -P -m") == 2


def test_standalone_parser_resolves_a_named_parameter_set_without_forecast_runtime(
        tmp_path, staged_copy):
    """The staged parser carries its registry and fixed-clock dependencies."""
    import tomllib
    from test_physics_params import _EXPERIMENT

    staged, receipt = staged_copy
    assert "woof/physics_params.py" in receipt["files"]
    assert "woof/physics_params_registry_v1.json" in receipt["files"]
    assert "woof/core/adaptive_clock.py" in receipt["files"]
    assert "woof/core/adaptive_timestep.py" in receipt["files"]
    assert "woof/render_compare.py" not in receipt["files"]
    metadata = tomllib.loads((staged / "pyproject.toml").read_text(encoding="utf-8"))
    assert "physics_params_registry_v1.json" in metadata["tool"]["setuptools"]["package-data"]["woof"]
    version = metadata["project"]["version"]
    info = staged / f"rw_wps-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: rw-wps\nVersion: {version}\n",
                                   encoding="utf-8")
    (info / "RECORD").write_text("woof/__init__.py,,\n", encoding="utf-8")
    set_path = tmp_path / "member.toml"
    set_path.write_text('[physics_params]\nname = "member"\nvalues = {"mynn.prandtl" = 0.8}\n',
                        encoding="utf-8")
    script = r"""
import os
from pathlib import Path
import sys
import tomllib
from woof.experiment import build_experiment, experiment_config_document
from woof import physics_params
from woof.core import adaptive_clock, adaptive_timestep

root = Path(os.environ["RW_WPS_STAGED_ROOT"]).resolve()
assert Path(physics_params.__file__).resolve().is_relative_to(root)
assert Path(adaptive_clock.__file__).resolve().is_relative_to(root)
assert Path(adaptive_timestep.__file__).resolve().is_relative_to(root)
assert adaptive_clock.wrf_num_sound_steps(20.0, 3000.0, 3000.0, 1.0443) == 6
exp = build_experiment(tomllib.loads(os.environ["RW_WPS_EXPERIMENT"]), "staged-parameter-test")
assert exp.physics_params.name == "member"
assert exp.physics_params.changed() == ("mynn.prandtl",)
assert experiment_config_document(exp)["physics_params"] == physics_params.document(exp.physics_params)
for module in ("woof.core.physics", "woof.core.ruc", "woof.core.ruc_gpu", "woof.core.model"):
    assert module not in sys.modules, module
"""
    environment = os.environ.copy()
    environment["GPUWM_NO_LOCAL_GPU"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(staged)
    environment["RW_WPS_STAGED_ROOT"] = str(staged)
    # A prescribed surface with radiation off needs no forecast-only
    # LSM or radiation modules. The selected coefficient belongs to MYNN.
    environment["RW_WPS_EXPERIMENT"] = _EXPERIMENT.replace(
        "sf_surface_physics = 3", "sf_surface_physics = 0").replace(
        "num_soil_layers = 6", "num_soil_layers = 4").replace(
        "ra_lw_physics = 4", "ra_lw_physics = 0").replace(
        "ra_sw_physics = 4", "ra_sw_physics = 0").replace(
        'ra_rrtmg_variant = "rrtmg_legacy"', 'ra_rrtmg_variant = "rte-rrtmgp"').replace(
        'wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-legacy-v1"',
        'wrf_rrtmg_compatibility = "none"')
    environment["WOOF_PHYSICS_PARAMS"] = str(set_path)
    completed = _run_python('-P', '-c', script, cwd=tmp_path, env=environment)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_standalone_terrain_preparation_reads_measured_maps_without_forecast(
        staged_project):
    """Non-flat preparation reaches the staged map readers and their data."""
    import tomllib

    staged, receipt = staged_project
    names = {
        "terrain_clock_map.json", "terrain_clock_map_local_face.json",
        "terrain_clock_adaptive_local_face.json",
    }
    assert {f"woof/{name}" for name in names} <= set(receipt["files"])
    metadata = tomllib.loads((staged / "pyproject.toml").read_text(encoding="utf-8"))
    assert names <= set(metadata["tool"]["setuptools"]["package-data"]["woof"])
    assert not [item for item in receipt["optional_internal_imports"]
                if item["module"] in {"woof.acoustic_adaptation",
                                      "woof.terrain_clock",
                                      "woof.terrain_clock_local"}]
    script = """
from pathlib import Path
import sys
import numpy as np
from woof import acoustic_adaptation, terrain_clock, terrain_clock_local
from woof.static.lambert import LambertGrid
from woof.static.terrain_autosmooth import measured_slope_limit, prepare_fields

root = Path.cwd().resolve()
for module in (acoustic_adaptation, terrain_clock, terrain_clock_local):
    assert Path(module.__file__).resolve().is_relative_to(root)
grid = LambertGrid(25, 0, 20, 30, 0, 500, 500, 25, 25)
limit, arms = measured_slope_limit(grid)
assert limit == .85 and arms
# A non-flat field reaches the map instead of taking the flat early return.
terrain = np.zeros((24, 24))
terrain[10:14, 10:14] = 20
fields = {"HGT_M": terrain}
assert prepare_fields(fields, grid, domain_id=1) is fields
table = terrain_clock_local.candidate_map(arms)
assert table.rows
assert terrain_clock_local.adaptive_document()["rows"]
for name in ("woof.core.model", "woof.core.dycore", "woof.core.physics", "cupy"):
    assert name not in sys.modules, name
"""
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(staged), CUDA_VISIBLE_DEVICES="",
                       GPUWM_NO_LOCAL_GPU="1", PYTHONDONTWRITEBYTECODE="1")
    completed = _run_python('-P', '-c', script, cwd=staged, env=environment)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_standalone_python_project_excludes_forecast_executor(staged_project):
    staged, receipt = staged_project
    files = set(receipt["files"])

    assert receipt["distribution"] == "rw-wps"
    assert receipt["forecast_executor_files"] == []
    assert receipt["verification_imports"] == []
    assert receipt["unresolved_internal_imports"] == []
    assert receipt["optional_internal_imports"]
    assert all(item["optional_reason"]
               for item in receipt["optional_internal_imports"])
    assert {
        "woof/multi_run.py", "woof/stream.py",
    } <= FORBIDDEN_WHEEL_PAYLOADS
    assert not (FORBIDDEN_WHEEL_PAYLOADS & files)
    assert not any(name.startswith("woof/verify/") for name in files)
    assert "woof/source_drivability.py" in files
    assert "woof/core/track_boundary.py" in files
    assert "woof/obs/goes_window.py" not in files
    assert "woof/source_cli.py" in files
    assert "woof/physics_registry.py" in files
    assert "woof/physics_registry_v2.json" in files
    assert "woof/mapped_direct.py" in files
    assert "woof/twentycrv3_direct.py" in files
    # The packaged profiles a wheel user can name on `--source`, shipped as
    # the table data they are.  `woof/twentycrv3.py` used to be asserted
    # here: a 1,185-line per-model NetCDF-CF decoder with no consumer,
    # replaced by these three documents plus one adapter row.
    for role in ("mapping", "composition", "provenance"):
        assert f"woof/authorities/rw-wps-20crv3-netcdf.{role}.json" in files
        assert (
            f"woof/authorities/rw-wps-20crv3-member-grib2.{role}.json"
            in files)
    assert "woof/core/state.py" in files
    assert "woof/core/nest_fields.py" in files
    assert "woof/core/ozone_contract.py" in files
    assert "woof/core/inflow_perturbation.py" in files
    assert "woof/ensemble/surface_controls.py" in files
    assert "woof/ensemble/member_variants.py" in files
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(staged), CUDA_VISIBLE_DEVICES="",
                       GPUWM_NO_LOCAL_GPU="1", PYTHONDONTWRITEBYTECODE="1")
    script = """
from pathlib import Path
from woof.ensemble import member_variants, surface_controls
assert Path(surface_controls.__file__).resolve().is_relative_to(Path.cwd())
assert Path(member_variants.__file__).resolve().is_relative_to(Path.cwd())
assert surface_controls.shared_surface_options(
    {"kind": "surface-state", "sst_offset_k": [-1.0, 1.0]}, 2
) == {"kind": "surface-state", "sst_offset_k": [-1.0, 1.0]}
variants = member_variants.normalize_member_variants([
    {"name": "land_a", "physics": {"sf_surface_physics": 2, "num_soil_layers": 4}},
    {"name": "land_b", "physics": {"sf_surface_physics": 3, "num_soil_layers": 9},
     "surface": {"soil_moisture_scale": 0.9, "sst_offset_k": 1.0}},
], 2)
assert variants[0]["physics"] == {"sf_surface_physics": 2, "num_soil_layers": 4}
assert variants[1]["surface"]["sst_offset_k"] == 1.0
import sys
from woof import kernel_aot, kernel_aot_build
from woof.certify import compile_platform
assert Path(kernel_aot.__file__).resolve().is_relative_to(Path.cwd())
assert Path(kernel_aot_build.__file__).resolve().is_relative_to(Path.cwd())
assert callable(compile_platform.nvrtc_banner)
assert "woof.supervisor" not in sys.modules
assert "cupy" not in sys.modules
"""
    completed = _run_python('-P', '-c', script, cwd=staged, env=environment)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "woof/core/attribute_tracking.py" in files
    assert "woof/toml_document.py" in files
    assert "woof/prepared_source_schemas.py" in files
    for name in ("metem_forecast", "wrfinput_forecast", "launchpad_api", "tui_worker",
                 "remote_cli", "remote_worker", "research_workspaces",
                 "starter_template", "tui_products", "case_catalog",
                 "case_catalog_import", "companion_query", "companion_domains",
                 "companion_setups",
                 "companion_forcing", "configuration_recovery", "remote_artifacts",
                 "remote_input_transfer", "remote_plan", "remote_processed",
                 "render", "render_receipts", "background_contract", "regional_preparation",
                 "local_da", "local_da_fetch", "local_da_observations", "local_da_runtime",
                 "cyclone_seed", "cyclone_sources"):
        assert f"woof/{name}.py" not in files
    assert "woof/ingest/case_store.py" not in files
    assert "woof/ingest/relocation_continuation.py" not in files
    assert "woof/core/thompson_contract.py" in files
    assert "woof/core/nssl2_contract.py" in files
    assert "woof/core/kernels/vert_interp.cu" in files
    assert "woof/offline_child.py" not in files
    assert "woof/offline_child_geography.py" not in files
    assert "woof/offline_child_run.py" not in files
    assert "woof/offline_child_smoke.py" not in files
    assert "woof/multi_run.py" not in files
    assert "woof/stream.py" not in files
    # `woof resume` locates a forecast checkpoint for the supervisor's
    # `run --restart` dispatch: nothing here imports it, no entry point
    # exposes it, and its own lookups are woof.supervisor (forbidden
    # above) and woof.io.restart (not staged).  A preprocessing wheel
    # has no checkpoints to resume from.
    assert "woof/resume.py" not in files
    # The saved history a resumed forecast's first pictures read: its
    # importers are the excluded go and live-render doors, and it reaches
    # woof.io.restart, woof.resume and woof.live_products.  Staged, it
    # made this staging refuse outright.
    assert "woof/restart_render.py" not in files
    # Auto's card-load probe ships as its own leaf and the forecast memory
    # preflight that re-exports it stays out, so the backend selector
    # reaches nothing this package omits: no optional import stands in for
    # the load reading.
    assert "woof/core/device_probe.py" in files
    assert "woof/core/preflight.py" not in files
    assert not [item for item in receipt["optional_internal_imports"]
                if item["path"] == "woof/ingest/preprocess_backend.py"]
    # The run disk projection and its download and preparation pricing are
    # read only by the excluded run-plan door, and the pricing parses a
    # fetch argv with woof.cli's parser: staged, the 2.8.0 cut battery
    # found an unresolved import of woof.cli here.
    assert "woof/disk_budget.py" not in files
    assert "woof/download_budget.py" not in files
    # The source table reads ArchiveWindow from the date guidance at import,
    # so the guidance ships; its availability answer reaches the domain
    # wizard, which does not, and that one import is named optional.
    assert "woof/source_availability.py" in files
    assert {(item["path"], item["module"]) for item in receipt["optional_internal_imports"]} >= {
        ("woof/source_availability.py", "woof.domain_wizard")}
    # The initial-state perturbation ships with the ingest package; the
    # radiation ceiling import is optional. A configuration may also need
    # this module's cloud-optics metadata, in which case the package-derived
    # closure stages it and leaves its forecast-only imports unreachable.
    assert "woof/ingest/init_perturbation.py" in files
    if "woof/core/rrtmgp.py" in files:
        assert "woof/core/rrtmgp.py" in receipt["import_closure_files"]
        assert any(item["path"] == "woof/core/rrtmgp.py"
                   and item["module"] == "woof.core.physics"
                   for item in receipt["optional_internal_imports"])
    else:
        assert {(item["path"], item["module"]) for item in receipt["optional_internal_imports"]} >= {
            ("woof/ingest/init_perturbation.py", "woof.core.rrtmgp")}
    # Preparation builds and seals the moving-nest statics corridor, so the
    # corridor module ships.  Its check of a corridor sealed under an older
    # build contract reads the relocation planner and the relocation
    # initializer's overlap rule; only the prepared-tree forecast runner
    # loads a corridor, so those two stay out and the imports are named
    # optional.  Unnamed, they made this staging refuse outright.
    assert "woof/static/corridor.py" in files
    assert "woof/core/nest_relocation.py" not in files
    assert "woof/ingest/relocation_init.py" not in files
    assert "woof/prepared_domain_tree_forecast.py" not in files
    assert {(item["path"], item["module"]) for item in receipt["optional_internal_imports"]} >= {
        ("woof/static/corridor.py", "woof.core.nest_relocation"),
        ("woof/static/corridor.py", "woof.ingest.relocation_init")}
    # Static preparation checks its slope against the measured terrain map.
    # Ship its CPU readers while leaving forecast execution and live physics
    # cadence out of this preprocessing package.
    assert "woof/acoustic_adaptation.py" in files
    assert "woof/terrain_clock.py" in files
    assert "woof/terrain_clock_local.py" in files
    assert "woof/core/adaptive_clock.py" in files
    assert "woof/core/adaptive_timestep.py" in files
    assert "woof/core/physics.py" not in files
    # lateral_bc.py no longer reaches woof.core.dycore: the exact chain moved
    # WRF_MUTS_SLOT into woof.state_serialization_contract (2.9.1), which the
    # standalone project already ships.
    optional = {(item["path"], item["module"]) for item in receipt["optional_internal_imports"]}
    assert ("woof/core/adaptive_clock.py", "woof.core.physics") in optional
    assert ("woof/ingest/lateral_bc.py", "woof.core.dycore") not in optional
    # The chained writer ships with the era5, gfs and mapped routes; the
    # forecast admission it reaches only when it chains does not, and a
    # preparation-only install never chains, so that import is optional.
    assert "woof/ingest/boundary_stream.py" in files
    assert "woof/core/urban_state.py" not in files
    assert {(item["path"], item["module"]) for item in receipt["optional_internal_imports"]} >= {
        ("woof/ingest/boundary_stream.py", "woof.core.preflight"),
        ("woof/ingest/boundary_stream.py", "woof.core.urban_state"),
        ("woof/ingest/boundary_stream.py", "woof.prepared_domain_tree_forecast")}
    # doctor, on the other hand, belongs: a preprocessing install is
    # exactly the one that needs to be told which bridge is missing.  It
    # is here because its WPS_GEOG check reads the dataset list from
    # woof.geog_assets, which stages the tree, rather than from the
    # domain wizard, which imports the CUDA front door.
    assert "woof/doctor.py" in files
    # The resolved-scheme vertical preflight ships with preprocessing:
    # rw-wps must refuse an nz a selected component cannot run.
    assert "woof/physics_vertical_contract.py" in files
    assert "tools/hrrr_state_proof.py" not in files
    assert "tools/hrrr_two_domain_forecast.py" not in files
    assert "LICENSE" in files
    assert "NOTICE" in files
    metadata = (staged / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "rw-wps"' in metadata
    assert 'license = { file = "LICENSE" }' in metadata
    assert (staged / "LICENSE").read_text(encoding="utf-8").startswith(
        "                              Apache License"
    )
    assert "matplotlib" not in metadata


def test_standalone_python_project_imports_without_source_tree_or_cupy(
        tmp_path, staged_project):
    staged, _receipt = staged_project
    script = r"""
from importlib.abc import MetaPathFinder
from pathlib import Path
import os
import sys

class RejectExternalModules(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = (
            "cupy",
            "woof.cli",
            "woof.companion_query",
            "woof.companion_domains",
            "woof.companion_setups",
            "woof.companion_forcing",
            "woof.configuration_recovery",
            "woof.core.model",
            "woof.core.physics",
            "woof.multi_run",
            "woof.runtime",
            "woof.remote_artifacts",
            "woof.remote_input_transfer",
            "woof.remote_plan",
            "woof.remote_processed",
            "woof.render",
            "woof.render_receipts",
            "woof.stream",
            "woof.supervisor",
            "woof.verify",
        )
        if any(fullname == name or fullname.startswith(name + ".")
               for name in forbidden):
            raise ModuleNotFoundError(f"blocked RW-WPS-external module: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalModules())
root = Path(os.environ["RW_WPS_STAGED_ROOT"]).resolve()
import woof.source_cli
import woof.era5_direct
from woof.fetch import validate_fetch_hints
from woof.source_drivability import intent_drivability
assert intent_drivability()["gfs"]["routes"]
validate_fetch_hints({"source": "gfs", "hours": 3}, source="standalone control")
assert "woof.runplan" not in sys.modules
import woof.gfs_direct
import woof.hrrr_hierarchy_direct
import woof.mapped_direct
import woof.twentycrv3_direct
import woof.twentycrv3_wrf
# The estate report is part of this surface: a preprocessing install is
# the one that most needs to be told which bridge is missing, and it
# must reach that conclusion without the CUDA front door the domain
# wizard drags in.
import woof.doctor
import tools.hrrr_single_domain_benchmark
from woof.core.nest_interp import register_nest, sint
import numpy as np

# Exercise the shared preparation contracts, not only their imports. They
# previously reached forecast-only owners when a real config/input used them.
from dataclasses import replace
from datetime import datetime
import tomllib
from woof.config import RunConfig
from woof.experiment import experiment_from_run_config
from woof.core.ozone_contract import cam_ozone_domain_ids
from woof.ingest.analyzed_numbers import metgrid_number_targets
from woof.experiment_document import render_experiment_document
from woof.prepared_source_schemas import mapped_sources, source_schemas
from woof.core.storm_tracking import build_follow_config
from woof.core.attribute_tracking import validate_attribute_domains
from types import SimpleNamespace

cfg = RunConfig(nx=28, ny=28, nz=12, dx=3000., dy=3000., ztop=16000.,
                dt=3., run_seconds=30.)
exp = experiment_from_run_config(cfg, datetime(2000, 6, 1, 12))
child = replace(exp.root, grid_id=2, parent_id=1, parent_grid_ratio=3,
                i_parent_start=8, j_parent_start=8,
                run=replace(cfg, nx=16, ny=16, dx=1000., dy=1000., dt=1.,
                            ra_lw_physics=4, ra_sw_physics=4,
                            ra_rrtmg_variant="rrtmg_legacy", o3input=2))
assert cam_ozone_domain_ids(replace(exp, domains=(exp.root, child))) == {1, 2}
assert metgrid_number_targets(replace(cfg, moist=True, mp_physics=28)) == {
    'QNI': 'ni', 'QNC': 'nc', 'QNR': 'nr', 'QNWFA': 'nwfa', 'QNIFA': 'nifa'}
raw = {'experiment': {'name': 'prepared'}, 'domain': [{'grid_id': 1}],
       'static': {'highres': {'path': "Ana's terrain", 'enabled': True}}}
assert tomllib.loads(render_experiment_document(raw)) == raw
assert 'mapped' in mapped_sources()
assert source_schemas()['mapped'] == 'gpuwm-mapped-composition-inputs-v1'
assert 'woof.core.preflight' not in sys.modules
assert 'woof.core.cam_ozone' not in sys.modules
assert 'woof.branch' not in sys.modules
assert 'woof.prepared_single_domain_forecast' not in sys.modules

# The ensemble package's preparation side is staged and its orchestration is
# not.  Every streamed preparation asks these hooks whether a member input is
# bound, and the namelist importer reads the stochastic contract; with the
# package absent, or with a door that imported woof.ensemble.cycle beside any
# submodule, each was a ModuleNotFoundError in a preparation binding no member.
from woof.ensemble.posted_preparation import (
    current_posted_domain, current_posted_preparation)
from woof.ensemble.runtime_preparation import current_runtime_preparation
assert current_posted_domain() is None
assert current_posted_preparation() is None
assert current_runtime_preparation() is None
import woof.namelist_stochastic
from woof.ensemble import physical_boundary, physical_store
for name in ("cycle", "engine", "member", "request", "batch_products"):
    assert f"woof.ensemble.{name}" not in sys.modules, name
    assert not (root / "woof" / "ensemble" / f"{name}.py").exists(), name
# An experiment config carrying [ensemble] still builds past the table:
# preparation consumes nothing from it, and its validator lives with the
# forecast this wheel omits.  The builder leaves it for the ensemble door, so
# the refusal below is about the rest of the document, not about the table.
from woof.experiment import build_experiment
try:
    build_experiment({"ensemble": {"members": 2, "unknown": 1}}, "standalone")
except ValueError as error:
    assert "must carry an [experiment] table" in str(error), error
else:
    raise AssertionError("an empty experiment document was built")
# A [grid]/[dynamics]/[run] config carrying the table is refused by name here
# as in the full distribution: no door runs an ensemble from one (2.8.4
# refused the table as unknown), and the refusal needs nothing this wheel
# omits.  Read and dropped, the table prepared one input for a run that the
# forecast install then refuses.
from woof.config import load_config
ensemble_config = Path.cwd() / "ensemble.toml"
ensemble_config.write_text(
    "[grid]\nnx = 28\nny = 28\nnz = 12\ndx = 3000.0\ndy = 3000.0\n"
    "ztop = 16000.0\n[run]\ndt = 3.0\nrun_seconds = 30.0\n"
    "[ensemble]\nmembers = 2\n", encoding="utf-8")
try:
    load_config(ensemble_config)
except ValueError as error:
    assert "carries an [ensemble] table" in str(error), error
    assert "remove the [ensemble] table to run this one forecast" in str(error), error
else:
    raise AssertionError("the [ensemble] table of a RunConfig was read and dropped")
assert "woof.ensemble.request" not in sys.modules
# The planning door reads a multi-model member list with what this wheel
# carries.  Its reader lived in the forecast door (woof.ensemble.recipe_door),
# which is not staged, so the staging refused the import as unresolved.
import contextlib
import io
import json
from woof.ensemble import recipes
listed = Path.cwd() / "members.json"
listed.write_text(json.dumps([{"source": "hrrr", "cycle": "2026-10-01T18"},
                              {"source": "rap", "cycle": "2026-10-01T18"}]),
                  encoding="utf-8")
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    assert recipes.main(["--source", "hrrr", "--cycle", "2026-10-01T18:00:00+00:00",
                         "--hours", "1", "--members", "2", "--recipe", "multi-model",
                         "--trajectories", str(listed)]) == 0
assert [member["trajectory"]["source"]
        for member in json.loads(printed.getvalue())["members"]] == ["hrrr", "rap"]
assert "woof.ensemble.recipe_door" not in sys.modules
assert not (root / "woof" / "ensemble" / "recipe_door.py").exists()
# A posted provider member needs the forecast runner's preflight and says so.
from woof.ensemble.posted_native import checked_source_inputs
try:
    checked_source_inputs(None, source="gfs", experiment_config="x",
                          wps_namelist="y")
except RuntimeError as error:
    assert "preparation-only installation" in str(error), error
else:
    raise AssertionError("a provider member was not refused by name")

# A staged config reader must retain attribute-following validation, including
# its refusal, while the runtime UI and executor imports remain blocked.
follow = build_follow_config({
    'field': 'attribute', 'attribute': 'theta', 'extremum': 'max',
    'reduction': 'column_max', 'threshold': 301., 'search_margin_cells': 10,
    'min_shift_cells': 1, 'max_shift_cells': 6, 'cooldown_seconds': 20.,
}, 'standalone-config')
validate_attribute_domains(
    [replace(exp.root, follow=None),
     replace(child, follow=SimpleNamespace(tracker=follow))],
    SimpleNamespace(follow=None))
try:
    validate_attribute_domains(
        [replace(exp.root, follow=None),
         replace(child, follow=SimpleNamespace(tracker=replace(follow,
             reduction='model_level', model_level=cfg.nz)))],
        SimpleNamespace(follow=None))
except ValueError as error:
    assert 'outside source d01 mass levels' in str(error)
else:
    raise AssertionError('staged attribute validation accepted a missing level')

capabilities = tools.hrrr_single_domain_benchmark.runner_capabilities()
assert capabilities["readiness"] == \
    "PREPARATION_ONLY_FORECAST_EXECUTOR_OMITTED"
assert capabilities["modes"]["prepare-only"]["available"] is True
assert capabilities["modes"]["forecast"]["available"] is False
assert "woof.core.model" in \
    capabilities["modes"]["forecast"]["missing_executor_modules"]
assert capabilities["standalone_rw_wps_wheel"][
    "forecast_executor_included"] is False

registration = register_nest(
    nri=3, nrj=3, i_parent_start=4, j_parent_start=3,
    child_nx=9, child_ny=9, parent_nx=12, parent_ny=12,
    stagger="", wrapper="interp",
)
parent = np.linspace(
    -2.0, 4.0, 2 * registration.nyp * registration.nxp,
    dtype=np.float32,
).reshape(2, registration.nyp, registration.nxp)
child = sint(parent, registration)
assert child.shape == (2, registration.nyc, registration.nxc)
assert np.isfinite(child).all()
for module in (
    woof.source_cli,
    woof.era5_direct,
    woof.gfs_direct,
    woof.hrrr_hierarchy_direct,
    woof.mapped_direct,
    woof.twentycrv3_direct,
    woof.twentycrv3_wrf,
    woof.doctor,
    tools.hrrr_single_domain_benchmark,
):
    assert Path(module.__file__).resolve().is_relative_to(root)
# Importing doctor is not the check; running the geography check is.
# The old spelling reached woof.domain_wizard from inside the function
# body, where an import test that only imports modules never sees it.
woof.doctor._geog_tree_checks(Path(os.environ["RW_WPS_STAGED_ROOT"])
                               / "no-such-geog-root")
# and the call really did take the lazy-import path, so the absences
# below are evidence rather than an artifact of nothing having run
assert "woof.geog_assets" in sys.modules
for name in (
    "cupy", "woof.cli", "woof.core.model", "woof.core.physics",
    "woof.domain_wizard", "woof.multi_run", "woof.resume",
    "woof.runtime", "woof.stream", "woof.supervisor", "woof.verify",
    "woof.remote_cli", "woof.remote_worker", "woof.research_workspaces",
    "woof.starter_template", "woof.tui_products",
):
    assert name not in sys.modules
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(staged)
    environment["RW_WPS_STAGED_ROOT"] = str(staged)
    completed = _run_python('-P', '-c', script, cwd=tmp_path, env=environment)
    assert completed.returncode == 0, completed.stderr


def test_the_full_distribution_validates_the_ensemble_table_at_load(tmp_path):
    """Leaving the table unvalidated is a property of the staged wheel alone.

    The breakage this prevents: build_experiment skips the ``[ensemble]``
    validator where ``woof.ensemble.request`` is not installed.  If that
    presence check ever answered False in the full distribution, every
    ensemble table would load unchecked and an unknown key would reach the
    batched forecast.

    load_config asks no such question: a ``[grid]``/``[dynamics]``/``[run]``
    config opens no ensemble session at any door, so its table is refused
    by name in every installation, as 2.8.4 refused it as unknown.
    """

    from woof import ensemble
    from woof.config import load_config
    from woof.experiment import build_experiment

    assert ensemble.request_installed()
    with pytest.raises(ValueError, match="unknown ensemble"):
        build_experiment({"ensemble": {"members": 2, "unknown": 1}}, "full")
    config = tmp_path / "ensemble.toml"
    config.write_text(
        "[grid]\nnx = 28\nny = 28\nnz = 12\ndx = 3000.0\ndy = 3000.0\n"
        "ztop = 16000.0\n[run]\ndt = 3.0\nrun_seconds = 30.0\n"
        "[ensemble]\nmembers = 2\nunknown = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"carries an \[ensemble\] table") as refused:
        load_config(config)
    assert "opens no ensemble session" in str(refused.value)


def test_the_planning_door_reads_its_member_list_without_the_forecast_door(tmp_path):
    """``python -m woof.ensemble.recipes --trajectories FILE`` imports no forecast door.

    The breakage this prevents: the planning module is staged into the
    preparation-only wheel, and its ``--trajectories`` reader was imported
    from ``woof.ensemble.recipe_door``, which is not.  The staging of the
    wheel refused that as an unresolved internal import, so the wheel could
    not be built.  This runs in any tree, staged or not.
    """

    listed = tmp_path / "members.json"
    listed.write_text(json.dumps([{"source": "hrrr", "cycle": "2026-10-01T18"},
                                  {"source": "rap", "cycle": "2026-10-01T18"}]),
                      encoding="utf-8")
    script = (
        "import sys\n"
        "from woof.ensemble import recipes\n"
        "code = recipes.main(['--source', 'hrrr', '--cycle', '2026-10-01T18:00:00+00:00',\n"
        "                     '--hours', '1', '--members', '2', '--recipe', 'multi-model',\n"
        "                     '--trajectories', sys.argv[1]])\n"
        "assert code == 0, code\n"
        "assert 'woof.ensemble.recipe_door' not in sys.modules\n")
    completed = _run_python(
        '-c', script, str(listed),
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": os.pathsep.join(
            [str(ROOT), os.environ.get("PYTHONPATH", "")])}, timeout=120)
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert [member["trajectory"]["source"] for member in plan["members"]] == ["hrrr", "rap"]


def test_standalone_auto_backend_reads_the_card_load(tmp_path, staged_project):
    """The staged package prices its preparation against the card's load.

    Auto's load probe lived in the forecast memory preflight, which this
    package does not carry, so the package kept a certified card another
    program had nearly filled or held busy, where woof prepared on the
    CPU.  The probe here is the real one, run as its own subprocess from
    the staged tree against a stand-in CuPy that reports the card's
    memory; CUDA_VISIBLE_DEVICES=-1 keeps the machine's own nvidia-smi
    out of it.  Whether the preparation fits is its price against that
    free memory (A65), and the price is computed in the staged tree from
    a mapped preparation's decoded inventory: while the price read its
    inventories from the forecast preflight, this import raised
    ModuleNotFoundError here and stopped every preparation.  The fitting
    half also holds the older fix: a probe that cannot be imported must
    not read as a CuPy that could not load.
    """

    staged, _receipt = staged_project
    shadow = tmp_path / "shadow"
    (shadow / "cupy").mkdir(parents=True)
    (shadow / "cupy" / "__init__.py").write_text(
        "import os\n"
        "\n"
        "GIB = 1024 ** 3\n"
        "\n"
        "\n"
        "class _Runtime:\n"
        "    @staticmethod\n"
        "    def memGetInfo():\n"
        "        return int(os.environ['SHADOW_FREE_GIB']) * GIB, 32 * GIB\n"
        "\n"
        "    @staticmethod\n"
        "    def getDeviceProperties(device):\n"
        "        return {'name': 'shadow card', 'multiProcessorCount': 64,\n"
        "                'maxThreadsPerMultiProcessor': 1536}\n"
        "\n"
        "    @staticmethod\n"
        "    def deviceGetLimit(limit):\n"
        "        return 1024\n"
        "\n"
        "\n"
        "class cuda:\n"
        "    runtime = _Runtime\n",
        encoding="utf-8")
    script = r"""
from pathlib import Path
from types import SimpleNamespace
import os

GIB = 1024 ** 3
root = Path(os.environ["RW_WPS_STAGED_ROOT"]).resolve()
assert not (root / "woof" / "core" / "preflight.py").exists()
import woof.ingest.preprocess_backend as backend
assert Path(backend.__file__).resolve().is_relative_to(root)

runtime = SimpleNamespace(getDeviceCount=lambda: 1, getDevice=lambda: 0,
                          runtimeGetVersion=lambda: 13020)
class ShadowCuda(backend.CudaPreprocessBackend):
    @property
    def array_module(self):
        return SimpleNamespace(__version__="14.2.0",
                               cuda=SimpleNamespace(runtime=runtime))

cpu = SimpleNamespace(name="cpu")
backend.CudaPreprocessBackend = ShadowCuda
backend.ParallelCpuPreprocessBackend = lambda **_: cpu
backend._gpu_runtime_installed = lambda: True
from woof.ingest import preparation_workers
preparation_workers.host_available_bytes = lambda: 1024 * GIB

from woof.config import RunConfig
from woof.ingest.preparation_price import price_forcing_preparation

run = RunConfig(nx=896, ny=512, nz=59, dx=6000.0, dy=6000.0, ztop=20000.0,
                dt=30.0, run_seconds=21600.0, terrain_opt=1, moist=True,
                mp_physics=8, km_opt=4, bl_pbl_physics=1, specified=True,
                spec_bdy_width=5, sf_sfclay_physics=91, sf_surface_physics=2)
exp = SimpleNamespace(domains=[SimpleNamespace(run=run)])
level = SimpleNamespace(shape=(40, 1059, 1799))
snapshot = SimpleNamespace(fields={
    "TT": level, "UU": level, "VV": level, "RH": level, "GHT": level,
    "PSFC": SimpleNamespace(shape=(1059, 1799))})
price = price_forcing_preparation("mapped", exp, [snapshot] * 7)
# The stand-in card is 32 GiB. No free memory cannot hold even a batch;
# 30 GiB free admits the host-retained, bounded CUDA preparation.
assert 2 * GIB < price.need_bytes < 30 * GIB, price.need_bytes

# A card another program has filled no longer sends auto to the CPU
# (f80839b3f6, CPU and CUDA preparation must give the same child inputs):
# the preparation waits for the memory and stays on the card.  The stand-in
# card frees 30 GiB at the first wait.
import time
waits = []
_sleep = time.sleep
def _release(seconds):
    waits.append(seconds)
    os.environ["SHADOW_FREE_GIB"] = "30"
os.environ["SHADOW_FREE_GIB"] = "0"
time.sleep = _release
try:
    chosen = backend.resolve_preprocess_backend("auto", price=price)
finally:
    time.sleep = _sleep
assert chosen is not cpu, chosen.selection
assert chosen.name == "cuda", chosen.selection
assert waits, "a full card must be waited for, not traded for the CPU"

os.environ["SHADOW_FREE_GIB"] = "30"
chosen = backend.resolve_preprocess_backend("auto", price=price)
from woof.ingest.bounded_cuda import BoundedCudaPreprocessBackend
assert isinstance(chosen, BoundedCudaPreprocessBackend), chosen.selection
assert "certified" in chosen.selection["reason"], chosen.selection
assert chosen.selection["device_load"]["free_bytes"] == 30 * GIB, chosen.selection
fit = chosen.selection["device_fit"]
assert fit["fits"] is True, fit
assert fit["need_bytes"] < price.need_bytes, fit
assert fit["unchunked_need_bytes"] == price.need_bytes, fit
assert fit["need_bytes"] == sum(fit["terms"].values()), fit
assert chosen.selection["chunking"]["retained_arrays"] == "host"
assert fit["free_bytes"] == 30 * GIB, fit

import woof.core.device_probe as probe
assert Path(probe.__file__).resolve().is_relative_to(root)
"""
    environment = os.environ.copy()
    environment.pop("WOOF_NATIVE_DISTRIBUTION_MANIFEST", None)
    environment.pop("GPUWM_NO_LOCAL_GPU", None)
    # WOOF's spelling of the same switch: woof/_env.py maps it back onto the
    # engine's name at import, so the probe would still see local CUDA forbidden.
    environment.pop("WOOF_NO_LOCAL_GPU", None)
    environment.pop("PYTHONSAFEPATH", None)
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["PYTHONPATH"] = os.pathsep.join((str(shadow), str(staged)))
    environment["RW_WPS_STAGED_ROOT"] = str(staged)
    completed = _run_python('-P', '-c', script, cwd=tmp_path, env=environment)
    assert completed.returncode == 0, completed.stderr
    assert "could not be loaded" not in completed.stderr
    assert "waiting for" in completed.stderr, completed.stderr


def test_standalone_preparation_publishes_at_its_seal(tmp_path, staged_project):
    """The staged chained writer admits a CUDA producer without a forecast.

    The package stages the chained writer for its era5, gfs and mapped
    routes but not the forecast, so a CUDA preparation died at its
    admission on the missing woof.core.preflight.  Run against the staged
    tree, with chaining asked for, the writer declines it by name and
    prices nothing.
    """

    staged, _receipt = staged_project
    script = r"""
from pathlib import Path
import os
import sys

root = Path(os.environ["RW_WPS_STAGED_ROOT"]).resolve()
from woof.ingest import boundary_stream
assert Path(boundary_stream.__file__).resolve().is_relative_to(root)
assert not boundary_stream.forecast_installed()
staging = Path(os.environ["RW_WPS_WORK"]) / ".tmp-tree"
staging.mkdir()
writer = boundary_stream.PreparedTreeWriter(
    staging=staging, output_root=staging.parent / "tree", identity={})
expected = {"chained": False,
            "reason": boundary_stream.PREPARATION_ONLY_REASON}
assert writer.chained is False
for backend in ("cuda", "cpu"):
    decision = writer.admit(experiment=object(), backend=backend,
                            device_bytes=1 << 30, card=(1 << 40, 0))
    assert decision == expected, decision
assert "woof.core.preflight" not in sys.modules
"""
    environment = os.environ.copy()
    environment["WOOF_CHAINED_PREP"] = "1"
    environment["PYTHONPATH"] = str(staged)
    environment["RW_WPS_STAGED_ROOT"] = str(staged)
    environment["RW_WPS_WORK"] = str(tmp_path)
    completed = _run_python('-P', '-c', script, cwd=tmp_path, env=environment)
    assert completed.returncode == 0, completed.stderr


def test_standalone_package_boundary_scan_rejects_verify_imports(tmp_path):
    package = tmp_path / "staged" / "woof"
    package.mkdir(parents=True)
    (package / "direct.py").write_text(
        "from woof.verify.npref import np_sint\n", encoding="utf-8")
    (package / "dynamic.py").write_text(
        "import importlib\n"
        "importlib.import_module('woof.verify.metrics')\n",
        encoding="utf-8",
    )

    assert _staged_verification_imports(tmp_path / "staged") == [
        {
            "path": "woof/direct.py",
            "line": 1,
            "module": "woof.verify.npref",
            "kind": "from",
        },
        {
            "path": "woof/dynamic.py",
            "line": 2,
            "module": "woof.verify.metrics",
            "kind": "dynamic",
        },
    ]


def test_standalone_package_boundary_scan_rejects_omitted_internal_imports(
        tmp_path):
    package = tmp_path / "staged" / "woof"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "direct.py").write_text(
        "from woof.io.restart import setup_fingerprint\n",
        encoding="utf-8",
    )

    assert _staged_internal_imports(
            tmp_path / "staged", source_root=ROOT) == [{
        "path": "woof/direct.py",
        "line": 1,
        "module": "woof.io.restart",
        "kind": "from",
    }]


def test_standalone_package_boundary_scan_reads_from_package_import_module(
        tmp_path):
    """``from woof.core import pace`` imports woof.core.pace.

    The scan read it as an import of woof.core, which is staged, so an
    unstaged submodule imported that way passed and the staged package
    raised ImportError at the call (A126: host_libm, and the staged map
    projections with it).  A name the source tree has
    no module for is an attribute and is not flagged; a staged submodule
    passes; a missing package is reported once, as itself.
    """

    staged = tmp_path / "staged"
    core = staged / "woof" / "core"
    core.mkdir(parents=True)
    for package in (staged / "woof", core):
        (package / "__init__.py").write_text("", encoding="utf-8")
    (core / "constants.py").write_text("", encoding="utf-8")
    (core / "user.py").write_text(
        "from woof.core import constants, pace\n"
        "from . import pace as relative_pace\n"
        "from woof.core import not_a_module_anywhere\n"
        "from woof.io import restart\n",
        encoding="utf-8",
    )
    assert (ROOT / "woof" / "core" / "pace.py").is_file()
    assert (ROOT / "woof" / "io" / "restart.py").is_file()

    assert _staged_internal_imports(staged, source_root=ROOT) == [
        {"path": "woof/core/user.py", "line": 1,
         "module": "woof.core.pace", "kind": "from"},
        {"path": "woof/core/user.py", "line": 2,
         "module": "woof.core.pace", "kind": "from"},
        {"path": "woof/core/user.py", "line": 4,
         "module": "woof.io", "kind": "from"},
    ]


def test_standalone_staging_refuses_an_unstaged_module_imported_from_its_package(
        tmp_path, monkeypatch):
    """The real staging refuses ``from woof.core import pace``.

    woof/core/streaming.py imports the pace model that way, and pace is
    not staged.  With its written reason removed from
    _OPTIONAL_STAGED_IMPORTS the staging refuses by name, which it did
    not do while the scan read the import as one of woof.core.
    """

    import tools.build_rw_wps_release as release

    key = ("woof/core/streaming.py", "woof.core.pace")
    assert key in release._OPTIONAL_STAGED_IMPORTS
    monkeypatch.setattr(release, "_OPTIONAL_STAGED_IMPORTS", {
        k: v for k, v in release._OPTIONAL_STAGED_IMPORTS.items() if k != key})
    with pytest.raises(RuntimeError) as refused:
        _stage_or_skip(tmp_path / "rw-wps-python")
    message = str(refused.value)
    assert "unresolved internal imports" in message
    assert "'module': 'woof.core.pace'" in message
    assert "'path': 'woof/core/streaming.py'" in message


def test_standalone_scan_reads_from_gpuwm_import_verify(tmp_path):
    package = tmp_path / "staged" / "woof"
    package.mkdir(parents=True)
    (package / "direct.py").write_text(
        "from woof import verify\n", encoding="utf-8")

    assert _staged_verification_imports(tmp_path / "staged") == [{
        "path": "woof/direct.py",
        "line": 1,
        "module": "woof.verify",
        "kind": "from",
    }]


def test_standalone_fire_static_keeps_metadata_without_forecast_commands(
        staged_project):
    staged, _receipt = staged_project
    for name in ("fire_ideal.py", "sfire_debug.py"):
        assert not (staged / "woof" / name).exists()
    for name in ("core/sfire_clock.py", "io/sfire_schema.py",
                 "static/sfire.py", "ingest/wrfinput_sfire.py"):
        assert (staged / "woof" / name).is_file(), name
    import tomllib
    project = tomllib.loads((staged / "pyproject.toml").read_text())
    assert project["project"]["scripts"]["gpuwm-sfire-static"] == (
        "woof.static.sfire:main")
    script = r"""
import importlib.abc, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
class NoCuPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] == 'cupy':
            raise ImportError('standalone fire-static metadata must not require CuPy')
sys.meta_path.insert(0, NoCuPy())
from woof.core import sfire_clock
from woof.io import sfire_schema, wrf_output_schema
from woof.static import sfire
from woof.ingest import wrfinput_sfire
for module in (sfire_clock, sfire_schema, wrf_output_schema, sfire, wrfinput_sfire):
    assert pathlib.Path(module.__file__).resolve().is_relative_to(root)
assert 'NFUEL_CAT' in sfire_schema.SFIRE_REGISTRY_FIELDS
assert 'FIRE_AREA' in wrf_output_schema.HISTORY_FIELDS_BY_NETCDF_NAME
try: sfire.main(['--help'])
except SystemExit as stopped: assert stopped.code == 0
else: raise AssertionError('the staged static helper did not expose its CLI')
"""
    result = _run_python('-I', '-c', script, str(staged))
    assert result.returncode == 0, result.stdout + result.stderr


def test_standalone_forecast_supervisor_annotation_still_guards_staging(
        tmp_path, monkeypatch, staged_project):
    import tools.build_rw_wps_release as release
    # The symbol-aware closure can now stage chem_sfire.import_native as a
    # leaf, so removing that old annotation no longer crosses an executor
    # boundary. Supervisor is still forbidden, regardless of its imports.
    # The unmodified tree stages: ``staged_project`` is that staging.
    key = ("woof/certify/capsule.py", "woof.supervisor")
    monkeypatch.setattr(release, "_OPTIONAL_STAGED_IMPORTS", {
        name: reason for name, reason in release._OPTIONAL_STAGED_IMPORTS.items()
        if name != key})
    with pytest.raises(RuntimeError, match="unresolved internal imports") as refused:
        _stage_or_skip(tmp_path / "missing-annotation")
    assert "woof.supervisor" in str(refused.value)
    assert "woof/certify/capsule.py" in str(refused.value)


def _finish_a_staged_preparation(tmp_path, staged, preamble=""):
    """Run a mapped preparation to its handoff through the staged door.

    ``staged`` is the staged project (the module's shared staging).  A
    stand-in adapter child writes a finished bundle; ``preamble`` runs in
    the child interpreter before woof is imported.
    """

    from test_source_adapters import _mapped_args

    root = tmp_path / "prepared"
    argv = _mapped_args()
    argv[argv.index("--output-root") + 1] = str(root)
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import json, pathlib, sys\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "root.mkdir()\n"
        "(root / 'proof.json').write_text(json.dumps("
        "{'schema': 'gpuwm-mapped-composition-proof-v1', 'status': 'READY'}))\n"
        "(root / 'experiment.toml').write_text('[experiment]\\nname = \"x\"\\n')\n"
        "(root / 'namelist.wps').write_text('&share\\n/\\n')\n",
        encoding="utf-8",
    )
    script = preamble + r"""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

staged = Path(os.environ["RW_WPS_STAGED_ROOT"]).resolve()
import woof.source_cli as source_cli
assert Path(source_cli.__file__).resolve().is_relative_to(staged)
source_cli._mapped_command = lambda args: [
    sys.executable, os.environ["RW_WPS_ADAPTER"], str(args.output_root)]
code = source_cli.main(json.loads(os.environ["RW_WPS_ARGV"]))
assert code == 0, code
from woof.ingest.boundary_stream import prepared_head_urban_columns
urban = SimpleNamespace(domains=(SimpleNamespace(
    run=SimpleNamespace(sf_urban_physics=3)),))
assert prepared_head_urban_columns(urban, {}) is None
assert "woof.core.urban_state" not in sys.modules
for name in ("woof.prepared_single_domain_forecast",
             "woof.prepared_domain_tree_forecast"):
    assert name not in sys.modules, name
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(staged)
    environment["RW_WPS_STAGED_ROOT"] = str(staged)
    environment["RW_WPS_ADAPTER"] = str(adapter)
    environment["RW_WPS_ARGV"] = json.dumps(argv)
    completed = _run_python('-P', '-c', script, cwd=tmp_path, env=environment)
    assert completed.returncode == 0, completed.stderr
    assert "Traceback" not in completed.stderr, completed.stderr
    assert "prep: complete" in completed.stdout
    assert "this installation prepares inputs only" in completed.stdout
    assert "woof.prepared_domain_tree_forecast" in completed.stdout
    assert f"The prepared tree is {root}" in completed.stdout
    assert "Run the forecast" not in completed.stdout
    assert (root / "proof.json").is_file()
    return completed


def test_standalone_preparation_finishes_without_the_forecast_runners(
        tmp_path, staged_project):
    """A finished preparation hands off without a forecast to hand to.

    The package stages prep_output and stage_cli but neither forecast
    runner, and the handoff resolved the bundle through the runners'
    schema tables: every finished `rw-wps` preparation ended in
    "ImportError: cannot import name 'prepared_domain_tree_forecast'"
    right after "prep: complete".  Run through the staged rw-wps door,
    with a stand-in adapter child that writes a finished bundle.
    """

    _finish_a_staged_preparation(tmp_path, staged_project[0])


#: The finder an editable woof install (``pip install -e .``, as the
#: publish workflow's test job installs it) appends to ``sys.meta_path``,
#: in setuptools' own shape: the top-level package maps to the checkout,
#: and an immediate child the path finders miss resolves there.
_EDITABLE_GPUWM_FINDER = r"""
import importlib.util as _util
import os as _os
from importlib.machinery import PathFinder as _PathFinder
from pathlib import Path as _Path
import sys as _sys


class _EditableGpuwmFinder:
    MAPPING = {"woof": str(_Path(_os.environ["RW_WPS_CHECKOUT"]) / "woof")}

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        parent = fullname.rpartition(".")[0]
        if parent in cls.MAPPING:
            return _PathFinder.find_spec(fullname, path=[cls.MAPPING[parent]])
        return None


# Keep normal staged-path resolution first, then this fixture's checkout,
# ahead of any real editable install already registered by the gate venv.
_sys.meta_path.insert(_sys.meta_path.index(_PathFinder) + 1, _EditableGpuwmFinder)
_spec = _util.find_spec("woof.prepared_domain_tree_forecast")
print("CHECKOUT RUNNER", _spec.origin)
"""


def test_standalone_preparation_ignores_a_runner_another_tree_provides(
        tmp_path, monkeypatch, staged_project):
    """A runner only another woof tree has is a runner this one lacks.

    With woof installed editable (publish.yml's test job) and the staged
    package first on the path, ``importlib.util.find_spec`` found the
    checkout's tree runner through the editable finder, so the handoff
    resolved the bundle and importing that runner against the staged
    woof.core failed on its forecast-only dependencies.
    The finder here is setuptools' shape; the handoff must end on the
    preparation-only line and import neither runner.
    """

    checkout_runner = ROOT / "woof" / "prepared_domain_tree_forecast.py"
    assert checkout_runner.is_file()
    monkeypatch.setenv("RW_WPS_CHECKOUT", str(ROOT))
    completed = _finish_a_staged_preparation(
        tmp_path, staged_project[0], preamble=_EDITABLE_GPUWM_FINDER)
    first = completed.stdout.splitlines()[0]
    # The finder is in effect: it hands the checkout's runner to find_spec.
    assert first.startswith("CHECKOUT RUNNER "), completed.stdout
    assert Path(first.removeprefix("CHECKOUT RUNNER ")).resolve() == (
        checkout_runner.resolve())


def test_the_standalone_package_reports_its_own_distribution_version(
        tmp_path, staged_project):
    """A clean rw-wps install knows its version.

    woof/__init__.py read the ``woof`` distribution alone, and a clean
    install of the standalone package carries only ``rw-wps``: it
    reported 0+unknown, and its installed runtime check refused every
    clean install ("rw-wps version mismatch: metadata=2.8.1,
    module=0+unknown").  Run in the staged tree with only an rw-wps
    distribution on the path (``-S``: no site-packages).
    """

    import woof

    assert PYTHON_DISTRIBUTION in woof.DISTRIBUTION_NAMES
    assert woof.DISTRIBUTION_NAMES[0] == woof.DISTRIBUTION_NAME
    staged, _receipt = staged_project
    site = tmp_path / "site"
    info = site / "rw_wps-9.8.7.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rw-wps\nVersion: 9.8.7\n",
        encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(staged), str(site)))
    completed = _run_python(
        '-S', '-P', '-c', 'import woof; print(woof.__version__); print(woof.__file__)',
        cwd=tmp_path, env=environment)
    assert completed.returncode == 0, completed.stderr
    version, location = completed.stdout.splitlines()
    assert version == "9.8.7"
    assert Path(location).resolve().is_relative_to(staged.resolve())


def test_a_staged_package_reads_its_own_version_beside_an_installed_gpuwm(
        tmp_path, staged_copy):
    """The staged package's runtime check passes beside another woof.

    install_gpuwm_native_wrf.sh installs the rw-wps wheel with ``pip
    install --target`` and runs its runtime check with only that target
    on PYTHONPATH, so an interpreter whose site-packages also carries
    woof (a venv, or a --user install) put a second woof distribution
    on the path.  woof/__init__.py read the first distribution NAMED
    woof, reported that install's number, and the check refused ("rw-wps
    version mismatch: metadata=9.8.7, module=1.2.3").  The staged tree
    here carries the rw-wps .dist-info pip --target writes beside it and a
    later site directory carries an installed woof, package and all.
    """

    staged, _receipt = staged_copy
    info = staged / "rw_wps-9.8.7.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rw-wps\nVersion: 9.8.7\n",
        encoding="utf-8")
    (info / "RECORD").write_text(
        _record_row("woof/__init__.py", staged / "woof" / "__init__.py")
        + f"{info.name}/METADATA,,\n{info.name}/RECORD,,\n",
        encoding="utf-8")
    site = tmp_path / "site-packages"
    (site / "woof").mkdir(parents=True)
    other = site / "woof" / "__init__.py"
    other.write_bytes(b"__version__ = '1.2.3'\n")
    other_info = site / "gpuwm-1.2.3.dist-info"
    other_info.mkdir()
    (other_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: woof\nVersion: 1.2.3\n",
        encoding="utf-8")
    (other_info / "RECORD").write_text(
        _record_row("woof/__init__.py", other)
        + f"{other_info.name}/METADATA,,\n{other_info.name}/RECORD,,\n",
        encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(staged), str(site)))
    program = (
        "import json, woof;"
        "from woof.native_wrf_distribution import _installed_record_receipt;"
        "print(json.dumps({'version': woof.__version__,"
        " 'file': woof.__file__,"
        " 'receipt': _installed_record_receipt()}))")
    completed = _run_python('-S', '-P', '-c', program, cwd=tmp_path, env=environment)
    assert "version mismatch" not in completed.stderr, completed.stderr
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert Path(payload["file"]).resolve().is_relative_to(staged.resolve())
    assert payload["version"] == "9.8.7"
    assert payload["receipt"]["distribution_name"] == PYTHON_DISTRIBUTION
    assert payload["receipt"]["distribution_version"] == "9.8.7"
    assert payload["receipt"]["record_file_count"] == 1


def _record_row(name, path):
    import base64

    digest = base64.urlsafe_b64encode(
        hashlib.sha256(path.read_bytes()).digest()).decode().rstrip("=")
    return f"{name},sha256={digest},{path.stat().st_size}\n"


@pytest.mark.parametrize("layout", ["target", "home"])
def test_the_installed_record_check_finds_the_console_scripts_pip_wrote(
        tmp_path, monkeypatch, layout):
    """The runtime check verifies every hashed RECORD row, scripts too.

    install.sh installs the wheel with ``pip install --target``, whose
    RECORD names the console scripts ``../../bin/<name>`` (relative to
    the temporary home's lib/python) while pip moves them to
    ``<target>/bin``.  Python 3.11 lists those rows and the check looked
    two levels above the target, so every clean install there failed
    ("installed wheel file is missing"); from Python 3.12 dist.files
    drops rows missing at the literal path, so a deleted file passed.
    ``home`` is ``pip install --home``, where the scripts do stay there.
    """

    import importlib.metadata

    import woof.native_wrf_distribution as distribution

    if layout == "target":
        root = tmp_path / "runtime"
        scripts = root / "bin"
    else:
        root = tmp_path / "home" / "lib" / "python"
        scripts = tmp_path / "home" / "bin"
    info = root / f"rw_wps-{distribution.__version__}.dist-info"
    info.mkdir(parents=True)
    scripts.mkdir(parents=True)
    (root / "woof").mkdir()
    module = root / "woof" / "entry.py"
    module.write_bytes(b"def main():\n    return 0\n")
    script = scripts / "rw-wps"
    script.write_bytes(b"#!/usr/bin/python3\nfrom gpuwm.entry import main\n")
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rw-wps\n"
        f"Version: {distribution.__version__}\n", encoding="utf-8")
    (info / "RECORD").write_text(
        _record_row("woof/entry.py", module)
        + _record_row("../../bin/rw-wps", script)
        + "woof/__pycache__/entry.cpython-311.pyc,,\n"
        + f"{info.name}/RECORD,,\n", encoding="utf-8")
    installed = importlib.metadata.PathDistribution(info)
    monkeypatch.setattr(distribution, "metadata", SimpleNamespace(
        distribution=lambda name: installed))

    receipt = distribution._installed_record_receipt()
    assert receipt["record_file_count"] == 2

    script.write_bytes(b"#!/usr/bin/python3\nraise SystemExit(1)\n")
    with pytest.raises(RuntimeError, match="RECORD mismatch for ../../bin/rw-wps"):
        distribution._installed_record_receipt()
    script.unlink()
    with pytest.raises(FileNotFoundError, match="installed wheel file is missing"):
        distribution._installed_record_receipt()


def test_standalone_radar_grid_writer_is_staged(tmp_path, staged_project):
    """woof.obs.write_radar_grid writes, or refuses by name.

    The staged woof.obs exports the radar-grid writer, whose default
    engine opens woof.io.classic_product; the package did not stage it,
    so the call raised ImportError.  Where the Rust NetCDF writer loads,
    the staged package writes a product and, where the Rust reader also
    loads, reads it back; where the writer does not load, the refusal is
    the named one, not an ImportError.
    """

    staged, _receipt = staged_project
    script = r"""
import os
from pathlib import Path
import sys

staged = Path(os.environ["RW_WPS_STAGED_ROOT"]).resolve()
sys.path.append(os.environ["RW_WPS_TESTS"])
import test_obs_radar_grid as helpers
import woof.io.classic_product
import woof.obs
from woof.io import nc_writer_bridge
from woof.obs.grid_product import ObsGridWriterUnavailable
for module in (woof.io.classic_product, woof.obs):
    assert Path(module.__file__).resolve().is_relative_to(staged), module

grid = helpers._grid(nx=21, ny=21)
volume = helpers._volume(grid, reflectivity=[5.0, 25.0, 45.0],
                         velocity=[-10.0, 0.0, 10.0])
observations, params = helpers._gridded(grid, volume)
path = Path(os.environ["RW_WPS_WORK"]) / "radar-grid.nc"
if nc_writer_bridge.unavailable_reason() is None:
    receipt = woof.obs.write_radar_grid(
        path, observations, grid, valid_time="2026-07-28T20:03:16Z",
        params=params)
    assert receipt["schema"] == woof.obs.RADAR_GRID_SCHEMA
    assert path.stat().st_size == receipt["bytes"]
    # The read-back needs the Rust NetCDF reader, a separate native: a
    # bridge folder can hold the writer without it, so it is checked only
    # where the reader resolves.
    from woof import netcdf_bridge
    if netcdf_bridge.find_netcdf_bin() is not None:
        read = woof.obs.read_radar_grid(
            path, expected_grid_identity=grid.identity_sha256())
        assert read["schema"] == receipt["schema"]
        assert read["dims"] == receipt["dims"]
    print("WROTE", receipt["bytes"])
else:
    try:
        woof.obs.write_radar_grid(
            path, observations, grid, valid_time="2026-07-28T20:03:16Z",
            params=params)
    except ObsGridWriterUnavailable as error:
        print("REFUSED", error)
    else:
        raise AssertionError("wrote without a loadable Rust writer")
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(staged)
    environment["RW_WPS_STAGED_ROOT"] = str(staged)
    environment["RW_WPS_TESTS"] = str(ROOT / "tests")
    environment["RW_WPS_WORK"] = str(tmp_path)
    # The radar-grid fixture superobs its volume before the writer under
    # test sees it, and since 2.8.7 superob refuses rather than falls back
    # to numpy when rw-superob is absent.  The staged tree is Python only,
    # so the library's own ladder there searches a tools/rustwx with no
    # build in it and the fixture died before reaching the writer (public
    # CI 37665304517, cpu ubuntu-24.04 and windows-2025).  Where this
    # checkout built the library, the fixture uses that build.  publish.yml's
    # test job and the Windows cpu job build no natives and run this file;
    # there the fixture states the numpy reference opt-out the library
    # documents.  Either way the subject, the staged writer or its named
    # refusal, is unchanged.
    from woof.obs import superob_bridge
    try:
        built = superob_bridge.resolve_superob_bridge()
    except superob_bridge.SuperobBridgeMissing:
        environment[superob_bridge.SUPEROB_PYTHON_ENV] = "1"
    else:
        environment[superob_bridge.SUPEROB_BRIDGE_ENV] = str(built)
        environment.pop(superob_bridge.SUPEROB_PYTHON_ENV, None)
    completed = _run_python('-P', '-c', script, cwd=tmp_path, env=environment)
    assert completed.returncode == 0, completed.stderr
    # Without the Rust static-fields library (publish.yml's test job
    # builds no natives) the grid helper's projection prints its
    # "[static] WORKAROUND" line first, so the outcome is found by line.
    outcomes = [line for line in completed.stdout.splitlines()
                if line.startswith(("WROTE ", "REFUSED "))]
    assert len(outcomes) == 1, completed.stdout


def test_cpu_native_export_modules_import_without_cupy():
    """The Rust/NumPy route must not import GPU or forecast executors."""

    script = r"""
from importlib.abc import MetaPathFinder
import sys

class RejectCupy(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = (
            "cupy",
            "woof.cli",
            "woof.core.model",
            "woof.runtime",
            "woof.supervisor",
        )
        if any(fullname == name or fullname.startswith(name + ".")
               for name in forbidden):
            raise ModuleNotFoundError(
                f"blocked RW-WPS-external module: {fullname}")
        return None

sys.meta_path.insert(0, RejectCupy())
import numpy as np
from woof.config import RunConfig
from woof.core.state import DomainState
import woof.era5_direct
import woof.gfs_direct
import woof.hrrr_hierarchy_direct
import woof.mapped_direct
import woof.twentycrv3_wrf

cfg = RunConfig(
    nx=3, ny=2, nz=2, dx=1000.0, dy=1000.0, ztop=10000.0,
    dt=1.0, run_seconds=1.0, moist=True, mp_physics=6,
)
state = DomainState(cfg, array_module=np)
assert isinstance(state.u, np.ndarray)
try:
    DomainState(cfg)
except RuntimeError as error:
    assert "CuPy is required for CUDA forecast state" in str(error)
else:
    raise AssertionError("default CUDA state did not reject missing CuPy")
assert "cupy" not in sys.modules
for module in (
    "woof.cli", "woof.core.model", "woof.runtime", "woof.supervisor"
):
    assert module not in sys.modules
"""
    completed = _run_python('-c', script, cwd=ROOT)
    assert completed.returncode == 0, completed.stderr


def test_cpu_backend_self_test_executes_parallel_fp32_arithmetic():
    class Plan:
        def apply(self, source, *, method, workers):
            assert source.dtype.name == "float32"
            assert method == "bilinear"
            assert workers in (1, 3)
            return source.reshape(-1)[[0, 1]].mean(dtype=source.dtype)[None, None] \
                + source.dtype.type([-0.5, 1.0, 2.5])[None, :]

    class Backend:
        def indexed_plan(self, source_shape, y, x):
            assert source_shape == (2, 2)
            assert y.dtype.name == x.dtype.name == "float32"
            return Plan()

    receipt = _cpu_backend_self_test(Backend())
    assert receipt["status"] == "PASS"
    assert receipt["worker_counts"] == [1, 3]
    assert receipt["output_values"] == [[1.0, 2.5, 4.0]]
    assert len(receipt["output_sha256"]) == 64


def test_runtime_version_parser_handles_local_and_post_releases():
    assert _numeric_version("13.0.0") >= (13, 0)
    assert _numeric_version("1.7.4.post1") >= (1, 6)


def test_wheel_release_audit_rejects_developer_profile_paths(tmp_path):
    wheel = tmp_path / "rw_wps_fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("woof/public.py", "ROOT = '/case/input'\n")
        # Assembled from fragments on purpose: this repository's own
        # release-snapshot scan reads every shipped file for exactly this
        # marker, and a literal here would trip it on the fixture.
        archive.writestr(
            "woof/private.md",
            "staged under C:/" + "Users/example/Downloads/private\n",
        )
    assert _wheel_public_path_violations(wheel) == [{
        "path": "woof/private.md",
        # Assembled, not written literally: see the fixture above.
        "marker": "c:/" + "users/",
        "kind": "Windows user profile",
    }]


def _fake_executable_bridge(
    path: Path, marker: bytes | None,
) -> None:
    path.write_bytes(b"\x7fELF" + (marker or b"") + b"\0fixture")
    path.chmod(0o755)


def _fake_pe_bridge(path: Path, marker: bytes) -> None:
    payload = bytearray(0x84)
    payload[:2] = b"MZ"
    payload[0x3c:0x40] = (0x80).to_bytes(4, "little")
    payload[0x80:0x84] = b"PE\0\0"
    payload.extend(marker)
    path.write_bytes(payload)


@pytest.mark.parametrize(
    ("name", "usage"),
    (
        ("grib2_inventory", "usage: grib2_inventory INPUT.grib2 [--decode]"),
        ("grib2_dump", "usage: grib2_dump INPUT.grib2 INDEX OUTPUT.f32"),
    ),
)
def test_generic_grib2_bridge_identity_binds_tabular_abi(
    tmp_path, monkeypatch, name, usage,
):
    monkeypatch.setattr(
        "woof.native_wrf_distribution.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2,
            stdout="",
            stderr=usage,
        ),
    )
    bridge = tmp_path / name
    _fake_executable_bridge(bridge, _BRIDGE_ABI_MARKERS[name])
    identity = bridge_identity(bridge, name)
    assert identity["tabular_abi_marker_sha256"] == hashlib.sha256(
        _BRIDGE_ABI_MARKERS[name]
    ).hexdigest()

    _fake_executable_bridge(bridge, None)
    with pytest.raises(RuntimeError, match="incompatible tabular ABI"):
        bridge_identity(bridge, name)


def test_bridge_identity_accepts_pe_and_preserves_executable_suffix(
    tmp_path, monkeypatch,
):
    bridge = tmp_path / "grib2_inventory.exe"
    _fake_pe_bridge(bridge, _BRIDGE_ABI_MARKERS["grib2_inventory"])

    def run(argv, **_kwargs):
        assert Path(argv[0]).suffix == ".exe"
        return SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="usage: grib2_inventory INPUT.grib2 [--decode]",
        )

    monkeypatch.setattr(
        "woof.native_wrf_distribution.subprocess.run",
        run,
    )
    identity = bridge_identity(bridge, "grib2_inventory")
    assert identity["binary_format"] == "pe"


def test_bridge_identity_executes_frozen_bytes_and_rejects_source_drift(
    tmp_path, monkeypatch
):
    bridge = tmp_path / "grib2_inventory"
    _fake_executable_bridge(
        bridge,
        _BRIDGE_ABI_MARKERS["grib2_inventory"],
    )
    original = bridge.read_bytes()

    def run(argv, **_kwargs):
        executed = Path(argv[0])
        assert executed != bridge
        assert executed.read_bytes() == original
        bridge.write_bytes(original + b"changed")
        return SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="usage: grib2_inventory INPUT.grib2 [--decode]",
        )

    monkeypatch.setattr(
        "woof.native_wrf_distribution.subprocess.run",
        run,
    )
    with pytest.raises(RuntimeError, match="changed during identity probe"):
        bridge_identity(bridge, "grib2_inventory")


def test_bridge_identity_translates_probe_timeout(tmp_path, monkeypatch):
    bridge = tmp_path / "grib2_inventory"
    _fake_executable_bridge(
        bridge,
        _BRIDGE_ABI_MARKERS["grib2_inventory"],
    )
    monkeypatch.setattr(
        "woof.native_wrf_distribution.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(args[0], timeout=10)
        ),
    )
    with pytest.raises(RuntimeError, match="identity probe timed out"):
        bridge_identity(bridge, "grib2_inventory")


def test_bridge_identity_translates_noexec_probe_error(tmp_path, monkeypatch):
    bridge = tmp_path / "grib2_inventory"
    _fake_executable_bridge(
        bridge,
        _BRIDGE_ABI_MARKERS["grib2_inventory"],
    )
    monkeypatch.setattr(
        "woof.native_wrf_distribution.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("noexec")
        ),
    )
    with pytest.raises(RuntimeError, match="temporary filesystem may be mounted noexec"):
        bridge_identity(bridge, "grib2_inventory")


def test_gfs_provenance_prefers_bound_distribution_manifest(tmp_path, monkeypatch):
    from conftest import complete_runtime_manifest

    manifest = tmp_path / "manifest.json"
    document = complete_runtime_manifest()
    document["source"].update({"commit": "a" * 40, "tree": "b" * 40})
    manifest.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setenv("WOOF_NATIVE_DISTRIBUTION_MANIFEST", str(manifest))
    identity = _git_source_identity()
    assert identity["identity_source"] == "gpuwm-native-distribution-manifest"
    assert identity["commit"] == "a" * 40
    assert identity["distribution_manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()).hexdigest()


def test_a_freshly_sealed_artifact_carries_the_distribution_version():
    """The stale constant was not merely printed -- it was SEALED IN.

    Native runtime contracts, prepared-cache writer identity and the
    standalone RW-WPS wheel's own pyproject all stamped ``0.1.1`` on a
    1.1.1 release.  Two of those feed gates that compare a wheel's
    metadata version against ``woof.__version__``, so the stale
    constant was one truthful version away from refusing its own seal.
    """

    from importlib import metadata
    import woof
    from tools.build_rw_wps_release import _standalone_pyproject

    distribution = metadata.version("recast-woof")
    for platform in ("linux-x86_64", "windows-x86_64"):
        contract = distribution_contract(platform)
        assert contract["gpuwm_version"] == distribution, platform

    # The standalone wheel's version is the one the sealed-runtime
    # receipt compares against; it must be stamped, never typed.
    stamped = [line for line in _standalone_pyproject().splitlines()
               if line.startswith("version = ")]
    assert stamped == [f'version = "{distribution}"'], stamped
    assert woof.__version__ == distribution


def test_a_freshly_written_prepared_cache_stamps_this_release(tmp_path):
    from importlib import metadata

    from woof.ingest.prepared_cache import (
        CACHE_WRITER_KEY, cache_writer_version)

    header = {CACHE_WRITER_KEY: {"gpuwm_version": __version__}}
    assert cache_writer_version(header) == metadata.version("recast-woof")


# ---------------------------------------------------------------------------
# The standalone rw-wps bundle's declaration
# ---------------------------------------------------------------------------


def test_the_standalone_bundle_carries_every_bridge_a_shipped_source_names():
    """A route that names a binary must be a route the bundle can run.

    THE defect this file is here to stop from recurring.  2.7.5 added the
    icosahedral remapper ``gdt101_remap`` to the bridge bundle for ICON
    global, and the standalone rw-wps bundle did not grow it, because its
    contents were a hand-kept tuple with no tie to the routes that need
    them.  It shipped as a known limit: ``woof doctor`` reported
    ``MISSING bridge gdt101_remap: not staged`` on an install carrying
    exactly what the bundle declares, and ``rw-wps --source icon-global``
    refused rather than falling back.

    The binding is to the packaged authorities, not to a literal, so it
    is the arbitrary test and not a patch: a producer on a new native
    grid arrives as a normalization document naming the binary that
    reads it, and this fails the moment that binary is outside the
    bundle -- before a release can ship a source the bundle cannot run.
    """

    from woof import source_authorities, source_normalization

    named = {
        source_normalization.load_normalization(name).bridge
        for name in source_authorities.packaged_normalizer_ids()
    }
    assert named, "no packaged source declares an input-normalization binary"
    assert "gdt101_remap" in named
    assert named <= set(BRIDGE_NAMES), (
        "the standalone rw-wps bundle does not carry "
        f"{sorted(named - set(BRIDGE_NAMES))}, which a packaged source's "
        "input normalization resolves and refuses without")


def test_every_declared_bridge_is_one_row_and_nothing_is_written_per_bridge():
    """One row is the whole declaration: name, workspace, option, env, probe.

    The five-places shape this replaces lost two bridges.  These are the
    surfaces a row must reach, checked together so none of them can go
    stale on its own again.
    """

    from woof.bridges import BRIDGE_ENV
    from woof.rustwx_fetch import FETCH_ENV, FETCH_NAME

    # The row's environment variable must be the one the module that
    # RESOLVES that bridge actually reads.  Both launchers are checked
    # against the row below, so a row holding its own copy of the name
    # would let the variable move in the resolver while the row, the
    # launchers and every test stayed green and agreed with each other
    # about a variable nothing reads.
    resolver_env = {**BRIDGE_ENV, FETCH_NAME: FETCH_ENV}
    assert BRIDGE_NAMES == tuple(b.name for b in BUNDLED_BRIDGES)
    assert len(set(BRIDGE_NAMES)) == len(BRIDGE_NAMES)
    assert BRIDGE_WORKSPACES == ("tools/grib1_bridge", "tools/rustwx")
    for bridge in BUNDLED_BRIDGES:
        assert bridge.name in resolver_env, (
            f"{bridge.name} is declared bundled but no module resolves it "
            "from an environment variable, so the launchers would bind a "
            "name nothing reads")
        assert bridge.env_var == resolver_env[bridge.name], (
            f"{bridge.name}'s row binds {bridge.env_var} and its resolver "
            f"reads {resolver_env[bridge.name]}")
        assert bridge.workspace in BRIDGE_WORKSPACES
        assert bridge.option.startswith("--")
        assert bridge.dest == bridge.option[2:].replace("-", "_")
        assert bridge.env_var.startswith("WOOF_")
        assert bridge.usage_marker == f"usage: {bridge.name}"
        assert _BRIDGE_USAGE_MARKERS[bridge.name] == bridge.usage_marker
        assert bridge.consumer.strip() == bridge.consumer and bridge.consumer

    remapper, = [b for b in BUNDLED_BRIDGES if b.name == "gdt101_remap"]
    assert remapper.workspace == "tools/grib1_bridge"
    assert remapper.env_var == "WOOF_GDT101_REMAP"
    assert _BRIDGE_ABI_MARKERS["gdt101_remap"] \
        == b"arwen.gdt101-regional-remap.v1"


def test_the_declared_bridges_are_the_ones_both_builders_take_and_stage():
    """The builders' option surface is generated, so it cannot drift.

    ``tools/build_rw_wps_release.py`` handed the packager a namespace
    that never named ``rw_fetch``; the packager read ``args.rw_fetch``
    and died with an ``AttributeError`` after the wheel and the Rust had
    already been built.  Both builders now take their options from the
    table, and a namespace missing a row is refused by name.
    """

    import argparse

    parser = argparse.ArgumentParser()
    add_bridge_options(parser)
    parsed = parser.parse_args([
        argument
        for bridge in BUNDLED_BRIDGES
        for argument in (bridge.option, f"/built/{bridge.name}")
    ])
    resolved = bridge_inputs(parsed)
    assert list(resolved) == list(BRIDGE_NAMES)
    assert resolved["gdt101_remap"].name == "gdt101_remap"

    short = argparse.Namespace(**{
        bridge.dest: Path(f"/built/{bridge.name}")
        for bridge in BUNDLED_BRIDGES if bridge.name != "gdt101_remap"
    })
    with pytest.raises(ValueError, match="gdt101_remap"):
        bridge_inputs(short)


@pytest.mark.parametrize(
    ("script", "staging", "cpu_backend"),
    (
        ("build_rw_wps_release.py",
         "**{bridge.dest: native / bridge.name",
         "native / CPU_BACKEND_LIBRARY"),
        ("build_rw_wps_windows_release.py",
         '**{bridge.dest: native / f"{bridge.name}.exe"',
         "native / WINDOWS_CPU_BACKEND_LIBRARY"),
    ),
)
def test_the_release_builder_compiles_every_workspace_the_table_names(
        script, staging, cpu_backend):
    """rw_fetch builds in the renderer workspace, not the decoder one.

    The shipped builders compiled ``tools/grib1_bridge`` only, so even a
    namespace that had named ``rw_fetch`` would have pointed at a file
    cargo never produced.  The build plan is the table's workspace
    column, read at build time.

    BOTH release builders, because there are two and only one of them
    was mended first: the Linux command was read off the table while
    ``tools/build_rw_wps_windows_release.py`` kept its single manifest
    and its five hand-written literals, and the new refusal in
    :func:`bridge_inputs` then stopped the Windows archive from being
    cut at all.  A guard that reads one of two files bans the shape in
    one of two files.
    """

    root = Path(__file__).resolve().parents[1]
    source = (root / "tools" / script).read_text(encoding="utf-8")
    assert "for workspace in BRIDGE_WORKSPACES:" in source
    assert staging in source
    assert cpu_backend in source
    for literal in ("grib1_bridge=native /", "gfs_bridge=native /",
                    "hrrr_bridge=native /", "grib2_dump=native /"):
        assert literal not in source, (
            f"a per-bridge literal is back in tools/{script}; that is "
            "the shape that dropped rw_fetch and then the remapper")
    for literal in ('"grib1_bridge" / "Cargo.toml"',
                    '"gpuwm_preprocess_cpu.dll"',
                    '"libgpuwm_preprocess_cpu.so"'):
        assert literal not in source, (
            f"tools/{script} hard-codes {literal} instead of reading the "
            "declaration, which is how one workspace and five bridges "
            "became the whole build plan")
    for workspace in BRIDGE_WORKSPACES:
        manifest = root / workspace / "Cargo.toml"
        assert manifest.is_file(), f"declared workspace has no {manifest}"


def test_both_installed_launchers_bind_every_declared_bridge():
    """The bundle's Python lives under ``runtime/``, so the ladder needs these.

    ``<root>/libexec/bridges`` is resolved beside the PACKAGE, which in
    an installed bundle is ``<root>/runtime``; the launcher's explicit
    environment binding is what actually reaches ``<root>/libexec``.
    Both launchers bound five of the six bridges before this, so a
    ``rw-wps`` run resolved the fetch backbone through some other rung
    or not at all.
    """

    tools = Path(__file__).resolve().parents[1] / "tools"
    posix = (tools / "gpuwm_native_wrf_launcher.sh").read_text(encoding="utf-8")
    windows = (tools / "gpuwm_native_wrf_launcher_windows.ps1").read_text(
        encoding="utf-8")
    for bridge in BUNDLED_BRIDGES:
        assert f'export {bridge.env_var}="$root/libexec/bridges/' \
            f'{bridge.name}"' in posix, bridge.name
        separator = chr(92)
        assert (f'$env:{bridge.env_var} = Join-Path $Root '
                f'"libexec{separator}bridges{separator}'
                f'{bridge.name}.exe"') in windows, bridge.name


@pytest.mark.parametrize("argv, message", [
    ([], "--bridge-dir is required unless --contract is used"),
    (["--contract", "--receipt", "r.json"],
     "--contract cannot be combined with runtime verification inputs"),
])
def test_runtime_check_usage_errors_reach_the_user(argv, message, monkeypatch,
                                                   capsys):
    """``woof-wrf-runtime-check`` prints its usage error and exits 2.

    The named breakage: ``main`` called ``parser.error`` on a parser it
    never bound, so a bare ``woof-wrf-runtime-check`` (or ``--contract``
    with a runtime input) died on ``NameError: name 'parser' is not
    defined`` instead of saying what was missing (shipped in 2.8.7;
    found by tests/test_no_undefined_names.py).
    """
    import woof.native_wrf_distribution as runtime_check
    import woof.provenance_gate as provenance_gate

    monkeypatch.setattr(provenance_gate, "announce_for_main", lambda name: None)
    with pytest.raises(SystemExit) as caught:
        runtime_check.main(argv)
    assert caught.value.code == 2
    assert message in capsys.readouterr().err


def test_preparation_package_carries_ocean_configuration_without_gpu(
        tmp_path, staged_project):
    project, _receipt = staged_project
    asset = project / "woof/data/surface_ocean.json"
    assert asset.read_bytes() == (ROOT / "woof/data/surface_ocean.json").read_bytes()
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from woof.config import RunConfig
from woof.surface_ocean import ocean_namelist_defaults
cfg = RunConfig(nx=8, ny=8, nz=8, dx=9000, dy=9000,
                ztop=12000, dt=10, run_seconds=20, sf_ocean_physics=1)
for key, value in ocean_namelist_defaults().items():
    assert getattr(cfg, key) == value, key
assert 'cupy' not in sys.modules
"""
    env = dict(os.environ, GPUWM_NO_LOCAL_GPU="1", PYTHONPATH="")
    result = _run_python('-P', '-c', code, str(project), cwd=tmp_path, env=env)
    assert result.returncode == 0, result.stdout + result.stderr


def test_runtime_identity_keeps_a_git_status_timeout_unverified(monkeypatch):
    """The receipt fallback remains truthful when the status subprocess times out."""
    from woof import provenance

    monkeypatch.setattr(provenance, "git_executable", lambda: "git")
    monkeypatch.setattr("woof.runtime_manifest.git_checkout_root", lambda root: root)
    monkeypatch.setattr(provenance.subprocess, "run", lambda *args, **kwargs: (
        _ for _ in ()).throw(subprocess.TimeoutExpired(args[0], timeout=10)))
    identity = provenance.git_identity(ROOT)
    assert identity is not None and identity["dirty"] is None
    assert "timed out" in identity["dirty_unknown_reason"]
    assert not provenance.worktree_is_clean(identity)
    receipt = provenance.Provenance(str(ROOT / "woof"), str(ROOT), "source", git=identity)
    assert "worktree unverified" in receipt.banner()
