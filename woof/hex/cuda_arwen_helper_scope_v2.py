"""Bind exact helper, SW CUDA and ozone sources outside physical identity.

Only complete measured sets are admitted: the 2.8.7 baseline and candidate, and the
published-engine rows in data/profiles/engine-helper-sets.v1.json. The original
sixteen-file engine manifest and physical restart contract are unchanged.
"""
from contextlib import nullcontext, contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path

_SOURCE_SETS = json.loads(Path(__file__).with_name("radiation_source_sets.json").read_text())
PUBLISHED_SOURCES = _SOURCE_SETS["baseline"]
DEVICE_SOURCES = _SOURCE_SETS["candidate"]
BASELINE_HELPERS = {name: digest for name, digest in PUBLISHED_SOURCES.items()
                    if name.startswith("woof/core/") and name.endswith(".py")}
SW_CUDA = "woof/core/kernels/rrtmg_sw.cu"
OZONE = "woof/ingest/wrf_ozone.py"
DEVICE_PY = "woof/ingest/ozone_pressure_device.py"
DEVICE_CUDA = "woof/ingest/ozone_pressure_device.cu"
DEVICE_OPTIONS = ("-std=c++17", "--ftz=false", "--fmad=false", "--prec-div=true")
ADMITTED_SOURCE_SETS = {
    "published-2.8.7": PUBLISHED_SOURCES,
    "sw32-ozone-device": DEVICE_SOURCES,
}
RUNTIME_DESCRIPTIONS = {
    "published-2.8.7": "published 2.8.7 coalesced shortwave and CPU ozone pressure",
    "sw32-ozone-device": "2.8.7 coalesced shortwave, scoped radiation readback and device ozone pressure",
}
PUBLISHED_MODES = ("published-2.8.7",)
EXECUTION_DEPENDENCIES = {}

# Measured engine helper sets, one row each in data/profiles/engine-helper-sets.
# A source row does not claim unchanged arithmetic or forecast output.
# THE BREAKAGE THIS PREVENTS: the 2.8.8 line rebinds one slab in rrtmg_lw.py,
# so every hex forecast on it refused here as an unknown source set.  A row
# binds the same six sources as the baseline, never fewer, and runs the
# published (non-candidate) path.
_HELPER_ROWS = json.loads(Path(__file__).with_name("data").joinpath(
    "profiles", "engine-helper-sets.v1.json").read_text(encoding="utf-8"))["sets"]
for _row in _HELPER_ROWS:
    if set(_row["sources"]) != set(PUBLISHED_SOURCES):
        raise RuntimeError("an engine helper row must bind exactly the published helper set: "
                           + _row["id"])
    if _row["id"] in ADMITTED_SOURCE_SETS:
        raise RuntimeError("engine helper row id is declared twice: " + _row["id"])
    ADMITTED_SOURCE_SETS[_row["id"]] = dict(_row["sources"])
    RUNTIME_DESCRIPTIONS[_row["id"]] = _row["description"]
    PUBLISHED_MODES = (*PUBLISHED_MODES, _row["id"])
    dependencies = _row.get("execution_dependencies", {})
    if any(not name.startswith("woof/") or not name.endswith(".py")
           or ".." in Path(name).parts or name in PUBLISHED_SOURCES
           for name in dependencies):
        raise RuntimeError("invalid engine execution dependency: " + _row["id"])
    EXECUTION_DEPENDENCIES[_row["id"]] = dict(dependencies)

def _named_engine_remedy(root):
    """Name the engine being refused and the way out (plan call 14).

    The dependency range admits the floor's whole minor, so an engine pip
    resolves can carry a helper set this distribution never measured.  The
    refusal names that engine by version, names the one this port was
    measured on, and gives both remedies, instead of two digests and no
    version (the 2026-08-27 user walk).
    """
    from . import engine_pin

    try:
        found = engine_pin.declared_version(Path(root)) or engine_pin.installed_version() or "of unknown version"
        wanted = engine_pin.wanted_version()
    except Exception as error:  # a refusal must still be readable on a broken estate
        return f"the installed engine could not be named ({error})"
    return (f"the installed engine is woof {found} at {root}; this woof hex admits the "
            f"helper sets measured above and was measured on woof {wanted}. Either "
            f'pip install "recast-woof=={wanted}", or install the woof hex release that admits '
            f"woof {found} (each engine release that moves this seam ships one)")


def _source_set_sha256(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True,
                          separators=(",", ":")).encode()).hexdigest()


def _inside(root, name):
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root):
        raise RuntimeError("WOOF-ArWen execution source escapes its engine tree: " + name)
    return path


@dataclass(frozen=True)
class ArwenHelperScopeV2:
    root: Path
    candidate: bool
    mode: str
    files: dict
    imported_paths: dict
    source_paths: dict
    provider: object = None
    execution_dependencies: dict = None

    def receipt(self):
        return {"schema": "hex-busy.helper-execution/v4",
                "mode": self.mode,
                "runtime_description": RUNTIME_DESCRIPTIONS[self.mode],
                "phase1_event_enabled": self.candidate,
                "source_set_sha256": _source_set_sha256(self.files),
                "source_files": dict(self.files),
                "execution_dependencies": dict(self.execution_dependencies or {}),
                "execution_source_set_sha256": _source_set_sha256(
                    {**self.files, **(self.execution_dependencies or {})}),
                "source_paths": dict(self.source_paths),
                "imported_paths": dict(self.imported_paths),
                "sw_cuda_compile_options": ["-std=c++17", "--ftz=false"],
                "device_pressure_enabled": self.provider is not None,
                "device_provider": self.provider.receipt() if self.provider is not None else None,
                "device_provider_snapshot_timing": "adapter builds helper receipt before the current phase1",
                "physical_restart_identity_changed": False}

    def phase1(self, *, n_columns, n_levels):
        if not self.candidate:
            return nullcontext()
        module = importlib.import_module("woof.core.rrtmg_lw")
        if Path(module.__file__).resolve(strict=True) != _inside(self.root, "woof/core/rrtmg_lw.py"):
            raise RuntimeError("LW helper was imported from another engine tree")
        @contextmanager
        def active():
            with ExitStack() as stack:
                if self.provider is not None:
                    stack.enter_context(self.provider.phase1())
                event = stack.enter_context(module.hex_busy_phase1_event_v2(n_columns, n_levels))
                yield event
        return active()


def bind_arwen_helpers_v2(root):
    root = Path(root).resolve(strict=True)
    paths = {name: _inside(root, name) for name in PUBLISHED_SOURCES}
    found = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    device_prefix = {name: DEVICE_SOURCES[name] for name in PUBLISHED_SOURCES}
    device_files_present = any((root / name).exists() or (root / name).is_symlink()
                               for name in (DEVICE_PY, DEVICE_CUDA))
    if found == device_prefix or device_files_present:
        for name in (DEVICE_PY, DEVICE_CUDA):
            paths[name] = _inside(root, name)
            found[name] = hashlib.sha256(paths[name].read_bytes()).hexdigest()
    mode = next((label for label, expected in ADMITTED_SOURCE_SETS.items() if found == expected), None)
    if mode is None:
        moved = [name for name in found if found[name] != DEVICE_SOURCES.get(name)]
        raise RuntimeError("unknown or mixed woof execution source set (admitted: "
                           + ", ".join(ADMITTED_SOURCE_SETS) + "); differs from the 2.8.7 device set at: "
                           + ", ".join(moved) + "; " + _named_engine_remedy(root))
    # Bind the exact source transformer as well as the six original helpers.
    # A matching radiation wrapper must never admit a different transformer.
    dependencies = EXECUTION_DEPENDENCIES.get(mode, {})
    for name, expected in dependencies.items():
        path = _inside(root, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("unknown or mixed woof execution dependency: " + name)
        paths[name] = path
    legacy = importlib.import_module("woof.core.rrtmg_legacy")
    if Path(legacy.__file__).resolve(strict=True) != _inside(root, "woof/core/rrtmg_legacy.py"):
        raise RuntimeError("pinned legacy consumer was imported from another engine tree")
    aliases = {"rrtmg_lw": "_lw", "rrtmg_sw": "_sw", "rrtmg_mcica": "_mcica",
               "rrtmg_legacy_prep": "_prep"}
    imported_paths = {}
    module_names = (*BASELINE_HELPERS, OZONE) + ((DEVICE_PY,) if mode == 'sw32-ozone-device' else ())
    for name in module_names:
        module = importlib.import_module(name[:-3].replace("/", "."))
        actual = Path(module.__file__).resolve(strict=True)
        if actual != paths[name]:
            raise RuntimeError("WOOF-ArWen execution module was imported from another engine tree: " + name)
        if name in BASELINE_HELPERS and getattr(legacy, aliases[actual.stem]) is not module:
            raise RuntimeError("pinned legacy helper binding is stale: " + name)
        imported_paths[name] = str(actual)
    for name in dependencies:
        module = importlib.import_module(name[:-3].replace("/", "."))
        actual = Path(module.__file__).resolve(strict=True)
        if actual != paths[name]:
            raise RuntimeError("WOOF-ArWen execution dependency was imported from another engine tree: " + name)
        imported_paths[name] = str(actual)
    # CudaSW compiles exactly this sibling file. Bind both its path and bytes.
    sw_module = importlib.import_module("woof.core.rrtmg_sw")
    compile_path = (Path(sw_module.__file__).parent / "kernels/rrtmg_sw.cu").resolve(strict=True)
    if compile_path != paths[SW_CUDA]:
        raise RuntimeError("SW CUDA compile source is outside the bound engine tree")
    provider = None
    if mode == "sw32-ozone-device":
        helper = importlib.import_module("woof.ingest.ozone_pressure_device")
        if Path(helper.cuda_source_path()).resolve(strict=True) != paths[DEVICE_CUDA]:
            raise RuntimeError("device ozone CUDA load source is outside the bound engine tree")
        if (helper.SOURCE_SHA256 != found[DEVICE_CUDA] or helper.HELPER_SHA256 != found[DEVICE_PY]
                or hashlib.sha256(helper.SOURCE.encode()).hexdigest() != found[DEVICE_CUDA]):
            raise RuntimeError("imported device ozone source cache is stale")
        if tuple(helper.NVRTC_OPTIONS) != DEVICE_OPTIONS:
            raise RuntimeError("device ozone compile options differ from the reviewed source")
        provider = helper.create_backend_provider()
    return ArwenHelperScopeV2(root, mode not in PUBLISHED_MODES, mode, found,
                             imported_paths, {name: str(path) for name, path in paths.items()}, provider,
                             dict(dependencies))
