"""Default-on hydro door: decide at preparation whether a fabric covers the domain.

Owner, 2026-10-18: hydro is default on for every forecast and every ensemble
member whenever a hydrofabric covers the domain; ``--hydro off`` is the
opt-out; where no fabric exists hydro is off and the receipt names the gap.

This module is orchestration only. It reads the provider table
(``woof/data/hydro/fabric-providers.v1.json``), the experiment's parent grid,
and runs the providers' own Rust tools (fabric builders, ``rw_hydrofabric
prepare``) as subprocesses. A new fabric source is a table row, never a new
code path here. No hydrologic number is computed in Python.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import contextlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

SCHEMA = "gpuwm-hydro-door.v1"
TABLE_SCHEMA = "gpuwm-hydro-fabric-providers-v1"
TABLE = Path(__file__).resolve().parents[1] / "data/hydro/fabric-providers.v1.json"
ENV_TABLE = "WOOF_HYDRO_FABRIC_PROVIDERS"      # extra rows, appended after the shipped table
ENV_CACHE = "WOOF_HYDRO_FABRIC_CACHE"
ENV_DECISION = "WOOF_HYDRO_DOOR_DECISION_JSON"  # launcher -> runner/member processes
PREPARED_KEYS = ("realization", "stages", "driver_factory")
LATTICE = 33  # parent-grid sample points per axis for coverage fractions


class HydroDoorRefusal(ValueError):
    """Explicit ``--hydro`` with no covering fabric: the request cannot be met."""


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


# -- provider table ---------------------------------------------------------

def load_providers(table=None):
    """Shipped rows, then any rows from ``$WOOF_HYDRO_FABRIC_PROVIDERS``."""
    documents = [json.loads(Path(table or TABLE).read_text())]
    extra = os.environ.get(ENV_TABLE)
    if table is None and extra:
        documents.append(json.loads(Path(extra).read_text()))
    head = documents[0]
    rows = []
    for document in documents:
        if document.get("schema") != TABLE_SCHEMA:
            raise ValueError(f"hydro fabric provider table needs schema {TABLE_SCHEMA}")
        rows.extend(document["providers"])
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate hydro fabric provider ids make the door ambiguous")
    rows.sort(key=lambda row: (int(row["priority"]), row["id"]))
    return {"coupled_parent_lsm": head["coupled_parent_lsm"], "budget": head["budget"],
            "providers": rows, "table_sha256": _digest(documents)}


# -- domain geometry --------------------------------------------------------

def _wrap180(value):
    return (value + 180.0) % 360.0 - 180.0


def domain_geometry(config_path):
    """Parent-grid footprint from the experiment's own projection resolver."""
    import numpy as np
    from woof.experiment import load_experiment
    from woof.static.projection import grids_from_projection_config

    experiment = load_experiment(config_path)
    grid = grids_from_projection_config(experiment)[0]
    run = experiment.domains[0].run
    nx, ny = int(run.nx), int(run.ny)
    xs = np.linspace(1.0, float(nx), LATTICE)
    ys = np.linspace(1.0, float(ny), LATTICE)
    x, y = np.meshgrid(xs, ys)
    lat, lon = grid.ij_to_latlon(x.ravel(), y.ravel())
    lat = np.asarray(lat, dtype=np.float64).ravel()
    lon = np.asarray(lon, dtype=np.float64).ravel()
    from .parent_geometry import bind
    from woof.hydrofabric import parent_grid_contract
    contract = parent_grid_contract(bind(grid))
    return geometry_from_points(lat.tolist(), lon.tolist(), center_lon=float(grid.cen_lon),
                                parent_lsm=int(run.sf_surface_physics),
                                grid_key=_digest({"definition": grid.definition(), "nx": nx, "ny": ny}),
                                nx=nx, ny=ny, dx_m=float(run.dx),
                                parent_grid_contract=contract,
                                time_origin=experiment.start_time.isoformat() + "Z",
                                duration_seconds=float(experiment.run_seconds),
                                interval_seconds=float(run.bldt * 60.0 or run.dt),
                                adaptive_clock=bool(getattr(run, "use_adaptive_time_step", False)))


def geometry_from_points(lat, lon, *, center_lon, parent_lsm, grid_key, **extra):
    relative = [_wrap180(value - center_lon) for value in lon]
    west = _wrap180(center_lon + min(relative))
    east = _wrap180(center_lon + max(relative))
    return {"south": min(lat), "north": max(lat), "west": west, "east": east,
            "crosses_dateline": west > east, "points": list(zip(lat, lon)),
            "parent_lsm": parent_lsm, "grid_key": grid_key, **extra}


def _inside(region, lat, lon):
    if region == "global":
        return True
    west, south, east, north = region
    if not south <= lat <= north:
        return False
    return west <= lon <= east if west <= east else (lon >= west or lon <= east)


def covered_fraction(coverage, points):
    regions = ["global"] if coverage == "global" else coverage
    inside = sum(1 for lat, lon in points if any(_inside(r, lat, lon) for r in regions))
    return inside / len(points) if points else 0.0


# -- availability probes (each runs the provider's own tool) -----------------

def _executable(spec):
    """The provider tool from the bridge ladder; a wheel install stages it on demand."""
    from woof import bridges

    def found():
        candidates = bridges.artifact_candidates(spec["env"], spec["filename"],
                                                 crate_relative=spec.get("crate"))
        return next((path for path in candidates
                     if path.is_file() and os.access(path, os.X_OK)), None)

    path = found()
    if path is None and not os.environ.get(spec["env"]) and bridges.fetch_on_demand(spec["filename"]):
        path = found()
    return path


def _expand(value):
    if value is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    return None if "$" in expanded else Path(expanded)


def _builder_path(need):
    path = _expand(os.environ.get(need["env"]) or need.get("default"))
    return None if path is None else path.resolve()


def probe(row):
    """Every reason this row cannot run today; empty means available."""
    reasons = []
    if row["kind"] == "prepared":
        manifest = _expand(row.get("manifest"))
        if manifest is None or not manifest.is_file():
            reasons.append(f"prepared manifest not present: {row.get('manifest')}")
        return reasons
    builder = row["builder"]
    exe = _executable(builder["executable"])
    if exe is None:
        spec = builder["executable"]
        reasons.append(f"builder executable {spec['filename']} not found "
                       f"(set {spec['env']} or build {spec.get('crate')})")
    else:
        try:
            usage = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
            text = usage.stdout + usage.stderr
        except (OSError, subprocess.TimeoutExpired) as error:
            text = ""
            reasons.append(f"builder executable {exe} did not answer: {error}")
        if text and not re.search(rf"(^|[\s|:]){re.escape(builder['subcommand'])}(\s|$|\|)", text):
            reasons.append(f"{exe.name} has no '{builder['subcommand']}' subcommand yet "
                           f"({builder.get('pending', 'builder not implemented')})")
        table = builder.get("source_table")
        if table:
            path = _repo_root() / table["path"]
            if not path.is_file():
                reasons.append(f"source table {table['path']} is not in this tree")
            else:
                admit = subprocess.run([str(exe), "admit", str(path), table["row"]],
                                       capture_output=True, text=True, timeout=30)
                if admit.returncode:
                    detail = (admit.stderr or admit.stdout).strip().splitlines()
                    reasons.append(f"source row {table['row']} not admitted: "
                                   f"{detail[-1] if detail else 'exit ' + str(admit.returncode)}")
    for need in builder.get("paths", ()):
        resolved = _builder_path(need)
        # A path with an acquisition hook is filled at preparation for the
        # footprint (woof.hydro.acquire); only an unresolvable root is a gap.
        if resolved is None or (not resolved.is_dir() and not need.get("acquire")):
            reasons.append(f"{need['label']} not available (set {need['env']})")
    # Python decoders a row's acquisition needs that a bare install does not
    # carry (an explicit extra). Checked here, before anything is fetched, so a
    # missing decoder rejects the row instead of failing after its download.
    for module in row.get("python_modules", ()):
        import importlib.util
        if importlib.util.find_spec(module["module"]) is None:
            reasons.append(f"{module['module']} ({module['licence']}) is not installed: "
                           f"pip install 'recast-woof[{module['extra']}]'")
    if row.get("preparation"):
        from .provider import recipe_path
        for name in ("parameters", "hydrofabric", "reservoirs", "initial_state"):
            spec = row["preparation"].get(name)
            if spec is not None and _executable(spec) is None:
                reasons.append(f"provider preparation needs {spec['filename']} ({spec['env']})")
        try:
            path = recipe_path(row)
            if not path.is_file():
                reasons.append(f"same-network parameter source recipe missing: {path}")
        except (OSError, ValueError) as error:
            reasons.append(f"parameter source recipe unavailable: {error}")
    return reasons


def row_realization(row, parent_lsm):
    """The coupled realization for this parent LSM: ``realization_by_parent_lsm``
    picks per LSM (Noah, RUC and Noah-MP publish runoff differently), else the
    row's single ``realization``."""
    by_lsm = row.get("realization_by_parent_lsm")
    if by_lsm:
        return by_lsm.get(str(parent_lsm))
    return row.get("realization")


def _realization_reasons(row, parent_lsm=None, adaptive_clock=False):
    name = row_realization(row, parent_lsm)
    if not name and row.get("realization_by_parent_lsm"):
        return [f"no coupled realization for parent sf_surface_physics={parent_lsm} "
                f"(rows: {sorted(row['realization_by_parent_lsm'])})"]
    if not name:
        return [f"no admitted coupled realization: {row.get('realization_gap', 'none named')}"]
    from .config import resolve_configuration
    try:
        resolve_configuration(name)
    except ValueError as error:
        return [str(error)]
    return []


def _coupling_reasons(row, geometry):
    """Reject invalid clocks before even an availability probe can fetch tools.

    Divisibility is not a constraint: the prepared provider conservatively
    splits producer volumes onto a fixed routing clock, including the tail.
    """
    reasons = []
    for key in ('interval_seconds', 'duration_seconds'):
        value = geometry.get(key)
        if key in geometry and (type(value) not in (int, float) or
                                 not math.isfinite(value) or value <= 0):
            reasons.append(f'coupling_interval_invalid: {key} must be finite and positive; '
                           'nonpositive/nonfinite clocks cannot advance runoff or publish history')
    policy = row.get('preparation')
    if policy:
        value = policy.get('routing_interval_seconds')
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            reasons.append('coupling_routing_interval_invalid: routing_interval_seconds must be '
                           'finite and positive; routing cannot advance with an invalid dt')
        elif 3600 % value:
            reasons.append('coupling_spinup_interval_unsupported: hourly initial-water replay needs '
                           'whole fixed routing substeps within each hour')
    return reasons


def evaluate(row, geometry, *, probe_row=None):
    probe_row = probe_row or probe  # the module's probe at call time
    fraction = covered_fraction(row["coverage"], geometry["points"])
    reasons = []
    minimum = float(row.get("min_covered_fraction", 0.0))
    if fraction <= 0.0 or fraction < minimum:
        reasons.append(f"covers {fraction:.3f} of the parent domain; row needs >= "
                       f"{minimum:g} and > 0")
    if "parent_lsm" in row and geometry["parent_lsm"] not in row["parent_lsm"]:
        reasons.append("provider exchange requires its declared parent LSM; this row supports "
                       + str(row["parent_lsm"]))
    licence = row["data_licence"]
    if licence["status"] != "verified":
        reasons.append(f"data licence {licence['status']} ({licence['name']}): {licence['reason']}")
    reasons.extend(_realization_reasons(row, geometry["parent_lsm"],
                                        geometry.get("adaptive_clock", False)))
    clock_reasons = _coupling_reasons(row, geometry)
    reasons.extend(clock_reasons)
    if fraction > 0.0 and not clock_reasons:
        try:
            reasons.extend(probe_row(row))
        except Exception as error:  # noqa: BLE001 - a failing probe is a named gap
            reasons.append(f"availability probe failed: {type(error).__name__}: {error}")
    return {"id": row["id"], "owner": row["owner"], "kind": row["kind"],
            "covered_fraction": round(fraction, 6),
            "realization": row_realization(row, geometry["parent_lsm"]), "reasons": reasons,
            "verdict": "available" if not reasons else "rejected"}


# -- the decision -------------------------------------------------------------

def explicitly_prepared(settings):
    return bool(settings) and any(settings.get(key) for key in PREPARED_KEYS)


def _summary(geometry):
    if geometry is None:
        return None
    keys = ("south", "north", "west", "east", "crosses_dateline", "parent_lsm", "grid_key",
            "nx", "ny", "dx_m", "parent_grid_contract",
            "time_origin", "duration_seconds", "interval_seconds", "adaptive_clock")
    return {key: geometry[key] for key in keys if key in geometry}


def decide(*, settings, requested, config_path=None, table=None, geometry=None,
           probe_row=None, geometry_reader=None, excluded_providers=()):
    """Return the door decision for one forecast; never touches a GPU.

    ``requested`` is the CLI value: None (default), "" (bare ``--hydro``),
    "off", or a settings path. ``settings`` is the [hydro] table already read
    by the activation policy (woof.ensemble.optional_routing.activation_settings).
    """
    settings = dict(settings or {})
    base = {"schema": SCHEMA, "requested": requested}
    if requested == "off" or settings.get("activation_reason") == "cli_off":
        return {**base, "enabled": False, "reason": "cli_off", "gaps": []}
    if explicitly_prepared(settings):
        return {**base, "enabled": bool(settings.get("enabled")),
                "reason": settings.get("activation_reason", "configuration_enabled"),
                "fabric": "explicit settings", "gaps": []}
    if settings.get("activation_reason") == "configuration_off":
        return {**base, "enabled": False, "reason": "configuration_off", "gaps": []}
    if requested is None and not settings.get("enabled"):
        # Hydro is opt-in (Owner, 2026-10-22: a first forecast in a new region downloaded 4.6 GB of hydro inputs and
        # spent minutes single-core on the spin-up): a bare run has no hydro; `--hydro on` (or a [hydro] table with
        # enabled = true) turns it on.
        return {**base, "enabled": False, "reason": "default_off", "gaps": []}
    providers = load_providers(table)
    base["table_sha256"] = providers["table_sha256"]
    if geometry is None:
        try:
            reader = geometry_reader or domain_geometry
            geometry = reader(config_path) if config_path is not None else None
        except Exception as error:  # noqa: BLE001 - the gap names the exact failure
            return {**base, "enabled": False, "reason": "no_domain_geometry",
                    "gaps": [{"owner": "hydro-world-door",
                              "gap": f"parent grid unreadable from {config_path}: "
                                     f"{type(error).__name__}: {error}"}]}
    if geometry is None:
        return {**base, "enabled": False, "reason": "no_domain_geometry",
                "gaps": [{"owner": "hydro-world-door",
                          "gap": "no experiment configuration, so no parent grid to cover"}]}
    base["footprint"] = _summary(geometry)
    candidates = [evaluate(row, geometry, probe_row=probe_row) for row in providers["providers"]]
    for candidate in candidates:
        if candidate['id'] in excluded_providers:
            candidate['reasons'].append('preparation failed earlier in this launch')
            candidate['verdict'] = 'rejected'
    base["candidates"] = candidates
    allowed = providers["coupled_parent_lsm"]
    if geometry["parent_lsm"] not in allowed["values"]:
        return {**base, "enabled": False, "reason": "parent_lsm_unsupported",
                "gaps": [{"owner": allowed["owner"],
                          "gap": f"parent sf_surface_physics={geometry['parent_lsm']}; "
                                 f"{allowed['reason']}"}]}
    chosen = next((c for c in candidates if c["verdict"] == "available"), None)
    if chosen is None:
        gaps = [{"owner": c["owner"], "provider": c["id"], "gap": "; ".join(c["reasons"])}
                for c in candidates if c["covered_fraction"] > 0.0] or [
                {"owner": "hydro-world-fabric",
                 "gap": "no provider row covers this domain"}]
        return {**base, "enabled": False, "reason": "no_covering_fabric", "gaps": gaps}
    row = next(r for r in providers["providers"] if r["id"] == chosen["id"])
    return {**base, "enabled": True, "reason": "fabric_covered", "provider": row["id"],
            "provider_kind": row["kind"], "realization": chosen["realization"],
            "manifest_binding": row.get("manifest_binding", "fabric_manifest"),
            "coverage_scope": row.get('coverage_scope', 'whole parent domain'),
            "attribution": row["data_licence"].get("attribution"), "gaps": [],
            "_row": row}


# -- materialisation at preparation -------------------------------------------

CACHE_DIRNAME = "hydro-fabric"


def cache_root(config_path=None):
    """Where prepared fabrics and their warm water state are built and reused.

    ``$WOOF_HYDRO_FABRIC_CACHE`` when set, else ``<case-data root>/hydro-fabric``:
    the user data root ``woof setup`` and ``fetch-geog`` stage into
    (:func:`woof.case_data.case_data_root`). Entries are keyed by the
    preparation request, not by the configuration, so a second run on the same
    domain from any configuration reuses the build. Never beside the
    configuration: that wrote ~1.5 GB of start-up state into a checkout's
    ``configs/.gpuwm-hydro`` on the gfs_12km quickstart and fails outright where
    ``configs/`` is read-only. ``config_path`` is accepted for callers and unused.
    """
    configured = os.environ.get(ENV_CACHE)
    if configured and configured.strip():
        return Path(configured.strip()).expanduser()
    from woof.case_data import case_data_root
    return case_data_root().expanduser() / CACHE_DIRNAME


def _fill(template, values):
    return [str(part).format(**values) for part in template]


@contextlib.contextmanager
def _build_lock(path):
    """Hold an exclusive lock on ``path`` across processes (created when absent)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as handle:
        try:
            import fcntl
        except ImportError:
            fcntl = None
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return
        import msvcrt
        handle.seek(0)
        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                break
            except OSError:  # LK_LOCK gives up after 10 s; a fabric build holds it for minutes
                continue
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def source_footprint(row, footprint):
    """Limit native source work to its coverage; retain the parent-grid contract."""
    if row.get("coverage_policy") != "source-land":
        return footprint
    coverage = row["coverage"]
    if len(coverage) != 1:
        raise RuntimeError(f"{row['id']} source-land coverage requires one contiguous box")
    bounds = coverage[0]
    clipped = dict(footprint)
    for key, limit, operation in zip(
            ("west", "south", "east", "north"), bounds,
            (max, max, min, min)):
        clipped[key] = operation(footprint[key], limit)
    if clipped["west"] >= clipped["east"] or clipped["south"] >= clipped["north"]:
        raise RuntimeError(f"{row['id']} has no source coverage in the parent domain")
    return clipped


def _build_location(root, request):
    """Reuse a complete matching build; preserve incomplete attempts on retry.

    A failed initial-water fetch can leave a valid fabric and parameter output.
    Running their non-overwriting producers into that directory again refuses
    every subsequent forecast, so retries select a fresh owned sibling.
    """
    attempt = 0
    while True:
        candidate = root if not attempt else root.with_name(f"{root.name}-retry-{attempt:04d}")
        try:
            if json.loads((candidate / "door-request.json").read_text()) == request:
                manifest = Path(json.loads((candidate / "door-prepared.json").read_text())["manifest"])
                if manifest.is_file() and manifest.resolve().is_relative_to(candidate.resolve()):
                    return candidate, manifest
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if not candidate.exists() or (candidate.is_dir() and not any(candidate.iterdir())):
            return candidate, None
        attempt += 1


def preparation_request(row, footprint):
    """The cache identity and inputs, resolved without acquiring or building."""
    paths, acquirers = {}, []
    for need in row["builder"].get("paths", ()):
        if "argument" in need:
            resolved = _builder_path(need)
            if resolved is not None and need.get("acquire"):
                acquirers.append((need, resolved))
            elif resolved is None or not resolved.is_dir():
                raise RuntimeError(f"{need['label']} not available (set {need['env']})")
            paths[need["argument"]] = str(resolved)
    request = {"preparation": row.get("preparation"), "provider": row["id"], "footprint": footprint,
               "builder": row["builder"], "data_licence": row["data_licence"],
               "paths": paths}
    if row.get("preparation"):
        from .provider import preparation_identity
        request["source_recipe"] = preparation_identity(row)
    return request, acquirers


def materialize(decision, *, config_path, runner=subprocess.run, prepare=None):
    """Build (or reuse) the covering fabric, one builder per provider at a time.

    Breakage it prevents: a fabric built ahead of its launch (``woof hydro
    ahead``, the WOOF box while it waits for a prepared bundle) and the
    launch's own build beside its fetch would write one cache entry at once.
    The second builder waits here and then reuses what the first one built.
    """
    row = decision["_row"]
    with _build_lock(cache_root(config_path) / row["id"] / ".materialize.lock"):
        return _materialize(decision, config_path=config_path, runner=runner, prepare=prepare)


def _materialize(decision, *, config_path, runner=subprocess.run, prepare=None):
    """Build (or reuse) the covering fabric and return the composed settings.

    The returned settings select the provider's realization with the prepared
    manifest bound; children read them through WOOF_NEXTGEN_SETTINGS.
    """
    row = decision["_row"]
    footprint = decision["footprint"]
    source_domain = source_footprint(row, footprint)
    from .retention import reference, reclaim
    started = time.perf_counter()
    build = {"provider": row["id"], "kind": row["kind"]}
    if row["kind"] == "prepared":
        manifest = _expand(row["manifest"])
        out = cache_root(config_path) / row["id"] / _digest(footprint)[:16]
        reference(out / "hydro-door.toml", root=cache_root(config_path))
        build["reused"] = True
    else:
        out = cache_root(config_path) / row["id"] / _digest(footprint)[:16]
        # Publish a live reference under the reclaimer's lock before reading
        # an existing start. Otherwise a concurrent launch could unlink it
        # between reading its receipt and binding the reused settings.
        reference(out / "hydro-door.toml", root=cache_root(config_path))
        receipt = out / "door-prepared.json"
        manifest = None
        if receipt.is_file():
            manifest = Path(json.loads(receipt.read_text())["manifest"])
        request, acquirers = preparation_request(row, footprint)
        paths = request["paths"]
        # A different recipe or initialization policy needs fresh preparation.
        # Keep the legacy footprint address only when its complete stamp matches;
        # rebuilding into it would overwrite an existing fabric and warm state.
        legacy_stamp = out / "door-request.json"
        if not (manifest is not None and manifest.is_file() and legacy_stamp.is_file()
                and json.loads(legacy_stamp.read_text()) == request):
            out = cache_root(config_path) / row["id"] / _digest(request)[:16]
            reference(out / "hydro-door.toml", root=cache_root(config_path))
            receipt = out / "door-prepared.json"
            manifest = (Path(json.loads(receipt.read_text())["manifest"])
                        if receipt.is_file() else None)
        out, manifest = _build_location(out, request)
        # A retry sibling is this launch's start too: reference it before use.
        reference(out / "hydro-door.toml", root=cache_root(config_path))
        receipt = out / "door-prepared.json"
        stamp = out / "door-request.json"
        # The fabric and its parameters do not depend on the forecast's start
        # time: they are kept once per domain and request (hard links into
        # each start's folder), and a new cycle on the same domain fetches
        # nothing and rebuilds only initial water and routing. Breakage it
        # prevents: every cycle and every new tip rebuilt the whole fabric and
        # parameters (2.2 GB and about 3 minutes for the Oklahoma WoFS box).
        static_request = dict(request, footprint={k: v for k, v in footprint.items()
                                                  if k not in _TIME_KEYS})
        static_dir = cache_root(config_path) / row["id"] / ("static-" + _digest(static_request)[:16])
        static_ready = (row.get("preparation") is not None
                        and (static_dir / "static-ready.json").is_file())
        if (manifest is not None and manifest.is_file() and stamp.is_file()
                and json.loads(stamp.read_text()) == request):
            build["reused"] = True
        elif static_ready:
            out.mkdir(parents=True, exist_ok=True)
            _link_static(static_dir, out)
            build["static_reused"] = str(static_dir)
            if "parent_grid_contract" in footprint:
                (out / "parent-grid.json").write_text(json.dumps(footprint["parent_grid_contract"]["grid"], indent=2) + "\n")
            config = {"geopackage": str(out / row["builder"].get("geopackage", "fabric.gpkg")),
                      "output": str(out / "prepared"),
                      "data_license": row["data_licence"]["name"]}
            if "parent_grid_contract" in footprint:
                config["grid"] = footprint["parent_grid_contract"]["grid"]
                config["parent_grid_contract"] = footprint["parent_grid_contract"]
            (out / "prepare.json").write_text(json.dumps(config, indent=2) + "\n")
            from .provider import prepare as prepare_provider
            prepared = prepare_provider(row=row, footprint=source_domain, out=out, config=config,
                                        runner=runner, static_ready=True)
            if isinstance(prepared, Mapping) and not prepared.get("manifest") and prepared.get("output"):
                prepared = dict(prepared, manifest=str(Path(prepared["output"]) / "manifest.json"))
            manifest = Path(prepared["manifest"]).resolve()
            if not manifest.is_relative_to(out.resolve()) or not manifest.is_file():
                raise RuntimeError(f"hydro preparation wrote no manifest inside {out}")
            receipt.write_text(json.dumps(prepared, indent=2, sort_keys=True) + "\n")
            stamp.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
            build["reused"] = False
        else:
            out.mkdir(parents=True, exist_ok=True)
            # The initial-water source (the spin-up's ERA5 forcing days) reads
            # the footprint's area only: it downloads beside the fabric's own
            # sources and build instead of after them (provider.start_water_prefetch).
            water_prefetch = None
            if row.get("preparation"):
                from .provider import start_water_prefetch
                water_prefetch = start_water_prefetch(row, footprint, out)
            exe = _executable(row["builder"]["executable"])
            table = row["builder"].get("source_table") or {}
            sources = None
            for need, resolved in acquirers:
                import importlib
                module, _, entry = need["acquire"].partition(":")
                recipe = None
                if row.get("preparation"):
                    from .provider import recipe_path
                    recipe = json.loads(recipe_path(row).read_text())
                sources, acquired = getattr(importlib.import_module(module), entry)(
                    footprint=source_domain, data_root=resolved, builder_exe=exe,
                    builder_row=table.get("row", ""), recipe=recipe)
                build.setdefault("acquisition", []).append(acquired)
            values = {"exe": exe, "out": out, "row": table.get("row", ""),
                      "table": _repo_root() / table["path"] if table else "",
                      "west": source_domain["west"], "south": source_domain["south"],
                      "east": source_domain["east"], "north": source_domain["north"]}
            values.update(paths)
            if "parent_grid_contract" in footprint:
                (out / "parent-grid.json").write_text(json.dumps(footprint["parent_grid_contract"]["grid"], indent=2) + "\n")
            argv = _fill(row["builder"]["argv"], values)
            result = runner(argv, capture_output=True, text=True,
                            timeout=int(row["builder"].get("timeout_seconds", 1800)))
            if result.returncode:
                raise RuntimeError(f"hydro fabric builder {row['id']} failed: "
                                   f"{(result.stderr or result.stdout).strip()[-2000:]}")
            build["builder_argv"] = argv
            build["builder_seconds"] = time.perf_counter() - started
            config = {"geopackage": str(out / row["builder"].get("geopackage", "fabric.gpkg")),
                      "output": str(out / "prepared"),
                      "data_license": row["data_licence"]["name"]}
            if "parent_grid_contract" in footprint:
                config["grid"] = footprint["parent_grid_contract"]["grid"]
                config["parent_grid_contract"] = footprint["parent_grid_contract"]
            (out / "prepare.json").write_text(json.dumps(config, indent=2) + "\n")
            if row.get("preparation"):
                from .provider import prepare as prepare_provider
                prepared = prepare_provider(row=row, footprint=source_domain, out=out,
                                            config=config, runner=runner, sources=sources,
                                            water_prefetch=water_prefetch)
            else:
                if prepare is None:
                    from woof.hydrofabric import prepare
                prepared = prepare(out / "prepare.json")
            if isinstance(prepared, Mapping) and not prepared.get("manifest") and prepared.get("output"):
                prepared = dict(prepared, manifest=str(Path(prepared["output"]) / "manifest.json"))
            if not isinstance(prepared, Mapping) or not prepared.get("manifest"):
                raise RuntimeError("rw_hydrofabric prepare returned no manifest path")
            manifest = Path(prepared["manifest"]).resolve()
            if not manifest.is_relative_to(out.resolve()):
                raise RuntimeError("rw_hydrofabric prepare manifest escapes the requested output")
            if not manifest.is_file():
                raise RuntimeError(f"rw_hydrofabric prepare wrote no manifest at {manifest}")
            receipt.write_text(json.dumps(prepared, indent=2, sort_keys=True) + "\n")
            stamp.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
            build["reused"] = False
            if row.get("preparation"):
                _keep_static(out, static_dir)
    build["manifest"] = str(manifest)
    build["seconds"] = time.perf_counter() - started
    # Where the initial water came from travels with the decision into the
    # run receipt: reanalysis spin-up, live analysis soil over a carried
    # spin-up, or the named era5_carry_over fallback.
    try:
        stages = json.loads((out / "provider-stages.json").read_text())
        water = [s for s in stages if s.get("stage") == "initial-water"][-1]["receipt"]["initial_water"]
        build["initial_water"] = {key: value for key, value in water.items() if key != "live"} | (
            {"live_valid_time": water["live"]["valid_time"], "live_url": water["live"]["fetch"]["url"]}
            if "live" in water else {})
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        pass
    realization = decision.get("realization") or row_realization(row, footprint.get("parent_lsm"))
    composed = {"enabled": True, "realization": realization,
                "settings_bindings": {decision["manifest_binding"]: str(manifest)},
                "activation_reason": "fabric_covered"}
    if row.get("preparation"):
        from .provider import composition
        composed.update(composition(manifest=manifest, footprint=footprint,
                                    realization=realization))
    out.mkdir(parents=True, exist_ok=True)
    settings_path = out / "hydro-door.toml"
    settings_path.write_text(to_toml(composed))
    reference(settings_path, root=cache_root(config_path))
    if not build.get("reused"):
        (out / "retention-managed.json").write_text(json.dumps({"schema": "gpuwm-hydro-retained-start.v1"}) + "\n")
    try:
        build["retention"] = reclaim(cache_root(config_path))
    except OSError as error:
        build["retention"] = {"blocked": str(error), "removed": [], "reclaimed_bytes": 0}
    return composed, settings_path, build


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("hydro door settings must be finite")
        return repr(value)
    if isinstance(value, Mapping):
        return "{ " + ", ".join(json.dumps(str(k)) + " = " + _toml_value(v)
                                 for k, v in value.items()) + " }"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return json.dumps(str(value))


def to_toml(settings):
    """Flat [hydro] table plus one level of subtables; enough for door output."""
    lines, tables = ["[hydro]"], []
    for key, value in settings.items():
        if isinstance(value, Mapping):
            tables.append((key, value))
        else:
            lines.append(f"{key} = {_toml_value(value)}")
    for key, table in tables:
        lines.append(f"\n[hydro.{key}]")
        lines.extend(f"{name} = {_toml_value(item)}" for name, item in table.items())
    return "\n".join(lines) + "\n"


# -- receipts and process hand-off --------------------------------------------

def public(decision):
    """Receipt form: drop the internal row reference."""
    if decision is None:
        return None
    return {key: deepcopy(value) for key, value in decision.items() if not key.startswith("_")}


def publish_decision(decision):
    os.environ[ENV_DECISION] = json.dumps(public(decision), sort_keys=True)


def launch_line(decision):
    """The one line `woof go` prints beside the memory verdict: hydro on or
    off, the fabric and its coverage of the parent grid, or the reason off.

    Breakage it prevents: on the 2.9.3 headline HRRR run hydro was off with
    no_covering_fabric and only the receipt said so.
    """
    if decision is None:
        return "go: hydro -- off: no door decision for this launch"
    if decision.get("reason") == "admission_pending":
        return "go: hydro -- waiting for disk and card admission; preparation has not started"
    candidates = {c["id"]: c for c in decision.get("candidates", ())}
    if decision.get("enabled"):
        provider = decision.get("provider") or decision.get("fabric") or "explicit settings"
        covered = candidates.get(provider, {}).get("covered_fraction")
        cover = "" if covered is None else f", covering {covered:.3f} of the parent grid"
        if decision.get('coverage_scope') not in (None, 'whole parent domain'):
            cover += '; ' + decision['coverage_scope']
        build = decision.get("build") or {}
        beside = ""
        if build.get("beside") and build.get("seconds") is not None:
            beside = (f" beside the fetch and the preparation (the forecast waited "
                      f"{build.get('waited_seconds', 0.0):.0f} s for it)")
        when = ("; prepared when the launch runs (dry run)" if build.get("deferred")
                else "; preparing beside the fetch and the preparation" if build.get("pending")
                else "; reused" + beside if build.get("reused")
                else f"; prepared in {build['seconds']:.0f} s" + beside if build.get("seconds") is not None
                else "")
        realization = decision.get("realization")
        water = build.get("initial_water") or {}
        initial = ("" if not water else
                   f"; initial water {water['method']}" + (" (FALLBACK)" if water.get("fallback") else "")
                   + (" (spin-up {ran} of {requested} h)".format(**water["spinup_hours"])
                      if isinstance(water.get("spinup_hours"), dict) else ""))
        return (f"go: hydro -- on: fabric {provider}{cover}"
                + (f", realization {realization}" if realization else "") + when + initial)
    gaps = "; ".join(f"[{gap.get('provider') or gap.get('owner')}] {gap.get('gap')}"
                     for gap in decision.get("gaps", ()))
    return f"go: hydro -- off ({decision.get('reason')})" + (f": {gaps}" if gaps else "")


def inherited_decision():
    text = os.environ.get(ENV_DECISION)
    return None if not text else json.loads(text)


#: Footprint keys a forecast's start time sets; the fabric and parameters do not read them.
_TIME_KEYS = ("time_origin", "duration_seconds", "interval_seconds", "adaptive_clock")

#: The time-independent artifacts of a provider preparation (builder and parameters).
_STATIC_FILES = ("fabric.gpkg", "fabric.gpkg.dams.json", "fabric.gpkg.terrain.json", "builder-receipt.json",
                 "parameterized.gpkg", "parameter-request.json", "static-gauges.json")
_STATIC_DIRS = ("parameters",)


def _link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def _keep_static(out, static_dir):
    """Keep this preparation's fabric and parameters for later starts on the domain."""
    if (static_dir / "static-ready.json").is_file():
        return
    staging = static_dir.with_name(static_dir.name + f".partial.{os.getpid()}")
    for name in _STATIC_FILES:
        if (out / name).is_file():
            _link(out / name, staging / name)
    for name in _STATIC_DIRS:
        for path in (out / name).rglob("*") if (out / name).is_dir() else ():
            if path.is_file():
                _link(path, staging / path.relative_to(out))
    gauges = {}
    if (out / "static-gauges.json").is_file():
        gauges = json.loads((out / "static-gauges.json").read_text()).get("gauges") or {}
    (staging / "static-ready.json").write_text(json.dumps({"static_dir": str(static_dir), "from": str(out),
                                                           "gauges": gauges}) + "\n")
    try:
        os.replace(staging, static_dir)
    except OSError:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)


def _link_static(static_dir, out):
    for path in static_dir.rglob("*"):
        if path.is_file():
            _link(path, out / path.relative_to(static_dir))


def _failure_memo(decision, config_path):
    """Where a failed fabric preparation for this footprint is recorded."""
    return (cache_root(config_path) / decision["provider"] / _digest(decision["footprint"])[:16]
            / "preparation-failed.json")


def resolve_launch(settings, *, requested, config_path, table=None, plan_only=False,
                   prepared_launch=False, decision=None, _failed=()):
    """Launcher entry: the decision, the settings to run, and the settings file.

    Returns (settings, settings_path_or_None, decision). Explicit ``--hydro``
    without a covering fabric raises HydroDoorRefusal naming every gap.
    ``plan_only`` (a dry run) records which fabric covers the domain and
    builds nothing. Breakage it prevents: ``woof go --dry-run`` on a
    continental domain fetched the hydro sources and built and spun up the
    fabric before printing its plan. ``decision`` is :func:`decide`'s answer
    when the caller already holds it (the launcher that prepares the fabric
    beside the fetch, woof.hydro.runtime), so the door does not decide twice.
    """
    if decision is None or _failed:
        decision = decide(settings=settings, requested=requested, config_path=config_path,
                          table=table, excluded_providers=[item['provider'] for item in _failed])
    if _failed:
        decision['preparation_fallbacks'] = list(_failed)

    def fallback(gap):
        row = decision.get('_row') or {}
        target = row.get('fallback_provider')
        available = any(c['id'] == target and c['verdict'] == 'available'
                        for c in decision.get('candidates', ()))
        if target and available:
            import sys
            print(f"hydro: {row['id']} preparation unavailable ({gap}); trying {target}", file=sys.stderr)
            return resolve_launch(settings, requested=requested, config_path=config_path,
                                  table=table, plan_only=plan_only, prepared_launch=prepared_launch,
                                  _failed=(*_failed, {'provider': row['id'], 'gap': gap}))
        return None
    if decision["reason"] == "fabric_covered" and plan_only:
        decision.pop("_row", None)
        decision["build"] = {"deferred": "dry run: the covering fabric is prepared when this launch runs"}
        return dict(settings or {}), None, decision
    memo = _failure_memo(decision, config_path) if decision["reason"] == "fabric_covered" else None
    if memo is not None and prepared_launch and memo.is_file() and requested != "":
        # A forecast from a prepared root takes the preparation's own answer:
        # it never fetches again what its --prepare-only step already tried.
        # Breakage it prevents: the Milton forecast retried 739 hydro inputs
        # its preparation had given up on, after the preparation's 516 s.
        recorded = json.loads(memo.read_text())
        alternative = fallback(recorded.get('gap', '') + ' (recorded at preparation; not fetched again)')
        if alternative is not None:
            return alternative
        decision.pop("_row", None)
        decision.update(enabled=False, reason="fabric_preparation_failed",
                        gaps=[{"owner": recorded.get("owner"), "provider": decision["provider"],
                               "gap": recorded.get("gap", "") + " (recorded by this footprint's preparation at "
                                      + recorded.get("at", "?") + "; not fetched again)"}])
        settings = dict(settings or {})
        settings["enabled"] = False
        settings["activation_reason"] = decision["reason"]
        return settings, None, decision
    if decision["reason"] == "fabric_covered":
        try:
            composed, path, build = materialize(decision, config_path=config_path)
            if memo is not None:
                memo.unlink(missing_ok=True)
        except Exception as error:  # noqa: BLE001 - default runs keep the forecast, loudly
            alternative = fallback(f'{type(error).__name__}: {error}')
            if alternative is not None:
                return alternative
            if requested == "":
                raise HydroDoorRefusal(f"--hydro was requested but preparing the covering fabric "
                                       f"{decision['provider']} failed: {type(error).__name__}: {error}") from error
            # Default-on run: the forecast still runs, hydro is off for this run
            # only and the receipt and stderr name the failure and its owner.
            import sys
            message = f"{type(error).__name__}: {error}"
            print(f"hydro: preparing fabric {decision['provider']} failed; this forecast runs "
                  f"without hydro: {message}", file=sys.stderr)
            row = decision.pop("_row")
            decision.update(enabled=False, reason="fabric_preparation_failed",
                            gaps=[{"owner": row["owner"], "provider": row["id"], "gap": message}])
            try:
                memo.parent.mkdir(parents=True, exist_ok=True)
                memo.write_text(json.dumps({"owner": row["owner"], "gap": message,
                                            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + chr(10))
            except OSError:
                pass
            settings = dict(settings or {})
            settings["enabled"] = False
            settings["activation_reason"] = decision["reason"]
            return settings, None, decision
        decision["build"] = build
        merged = dict(settings or {})
        merged.update(composed)
        return merged, path, decision
    if requested == "" and not decision["enabled"] and decision["reason"] != "cli_off":
        raise HydroDoorRefusal(
            "--hydro was requested but no hydrofabric covers this domain: "
            + "; ".join(f"[{gap['owner']}] {gap['gap']}" for gap in decision["gaps"]))
    # The decision travels in WOOF_HYDRO_DOOR_DECISION_JSON, never inside the
    # settings: settings feed the realization identity sealed in checkpoints.
    settings = dict(settings or {})
    if not decision["enabled"] and not explicitly_prepared(settings):
        settings["enabled"] = False
        settings["activation_reason"] = decision["reason"]
    return settings, None, decision
