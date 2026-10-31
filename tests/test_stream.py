"""CPU contracts for the operational chunked streaming controller."""

from __future__ import annotations

from collections import Counter
import dataclasses
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import os
import subprocess
import sys
import textwrap
import threading
import tomllib

import numpy as np
import pytest

from woof.experiment import load_experiment
from woof.fetch import HRRR_DEFAULT_MODE, hrrr_object_url
from woof.ingest.lateral_bc import (
    BoundaryInterval, FieldBoundary, LateralBoundaries, SideBoundary,
)
from woof.ingest.prepared_cache import write_prepared_cache
from woof.ingest.hrrr_target import load_hrrr_target_domain
from woof.io.restart import read_restart_header
from woof.namelist_import import parse_namelist
from woof import stream
from test_prepared_cache import _fixture as _prepared_cache_fixture


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_experiment(path: Path, *, delayed_child=False,
                      proof_geometry=False) -> None:
    child_start = (
        "start_time = 2026-07-23T00:30:00\n"
        if delayed_child else "")
    if proof_geometry:
        nz = 49
        eta_levels = ", ".join(
            f"{1.0 - level / nz:.12g}" for level in range(nz + 1))
        root_nx, root_ny = 72, 70
        child_nx, child_ny = 90, 72
        child_history_interval_s = 900.0
    else:
        nz = 8
        eta_levels = \
            "1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0"
        root_nx, root_ny = 100, 100
        child_nx, child_ny = 90, 90
        child_history_interval_s = 1800.0
    path.write_text(
        textwrap.dedent(f"""
        [experiment]
        name = "stream-tree"
        start_time = 2026-07-23T00:00:00
        run_seconds = 3600.0
        restart_interval_s = 3600.0

        [projection]
        map_proj = "lambert"
        ref_lat = 40.0
        ref_lon = -83.0
        truelat1 = 30.0
        truelat2 = 60.0
        stand_lon = -84.0

        [shared]
        nz = {nz}
        ztop = 12000.0
        p_top = 10000.0
        eta_levels = [{eta_levels}]
        hybrid_opt = 2
        etac = 0.2
        map_proj = 1
        moist = true
        mp_physics = 6

        [[domain]]
        grid_id = 1
        parent_id = 0
        i_parent_start = 1
        j_parent_start = 1
        parent_grid_ratio = 1
        parent_time_step_ratio = 1
        nx = {root_nx}
        ny = {root_ny}
        dx = 12000.0
        time_step = 60
        history_interval_s = 3600.0

        [[domain]]
        grid_id = 2
        parent_id = 1
        i_parent_start = 31
        j_parent_start = 31
        parent_grid_ratio = 3
        parent_time_step_ratio = 3
        {child_start.rstrip()}
        nx = {child_nx}
        ny = {child_ny}
        history_interval_s = {child_history_interval_s}
        """).lstrip(),
        encoding="utf-8",
    )


def _make_plan(tmp_path: Path, *, cycle_count=2, target_lead=2,
               delayed_child=False, proof_geometry=False,
               cache_enabled=False):
    experiment = tmp_path / "experiment.toml"
    _write_experiment(
        experiment, delayed_child=delayed_child,
        proof_geometry=proof_geometry)
    root_nx, root_ny, nz = (
        (72, 70, 49) if proof_geometry else (100, 100, 8))
    domain = tmp_path / "domain.json"
    _write_json(domain, {
        "schema": "gpuwm-hrrr-target-domain-v1",
        "name": "generic-stream-root",
        "map_proj": "lambert",
        "nx": root_nx,
        "ny": root_ny,
        "nz": nz,
        "dx_m": 12000.0,
        "dy_m": 12000.0,
        "ref_lat": 40.0,
        "ref_lon": -83.0,
        "truelat1": 30.0,
        "truelat2": 60.0,
        "stand_lon": -84.0,
        "time_step_seconds": 60,
        "spec_bdy_width": 5,
        "spec_zone": 1,
        "relax_zone": 4,
        "surface_fallback_radius_cells": 8,
    })
    wps = tmp_path / "namelist.wps"
    wps.write_text("&share\n max_dom = 2,\n/\n", encoding="utf-8")
    namelist = textwrap.dedent("""
        &time_control
         run_hours = 1,
        /
        &domains
         max_dom = 2,
        /
    """).lstrip()
    native = tmp_path / "namelist.input"
    stock = tmp_path / "namelist.stock.input"
    native.write_text(namelist, encoding="utf-8")
    stock.write_text(namelist, encoding="utf-8")
    geog = tmp_path / "geog"
    geog.mkdir()
    cache = tmp_path / "fetch-cache"
    fetch_section = ""
    if cache_enabled:
        cache.mkdir()
        fetch_section = textwrap.dedent(f"""
            [fetch]
            cache_dir = "{cache.name}"
        """)
    plan_path = tmp_path / "stream.toml"
    target_line = ("" if target_lead is None
                   else f"target_lead = {target_lead}\n")
    plan_path.write_text(
        textwrap.dedent(f"""
        schema = "{stream.PLAN_SCHEMA}"

        [stream]
        work_root = "work"
        cycle = "latest"
        cycle_count = {cycle_count}
        {target_line.rstrip()}
        poll_seconds = 1.0
        wait_timeout_seconds = 300.0

        {fetch_section.rstrip()}

        [prepare]
        experiment_config = "{experiment.name}"
        domain_spec = "{domain.name}"
        wps_namelist = "{wps.name}"
        namelist_input = "{native.name}"
        stock_wrf_namelist_input = "{stock.name}"
        geog_root = "{geog.name}"
        physics_profile = "generic-profile-v1"
        pipeline_workers = 2
        child_workers = 2

        [run]
        io_mode = "history"
        health_debug = true
        """).lstrip(),
        encoding="utf-8",
    )
    return stream.load_stream_plan(plan_path)


def test_omitted_target_defaults_to_one_retained_generation(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=None)

    assert plan.target_lead == 1


@pytest.mark.parametrize("syntax", [
    "basic-quoted", "literal-quoted", "escaped", "header-comments",
    "multiline",
])
def test_materialized_quoted_experiment_advances_every_declared_clock(
        tmp_path, syntax):
    plan = _make_plan(tmp_path)
    text = plan.experiment_config.read_text(encoding="utf-8")
    for parent_id in (0, 1):
        text = text.replace(
            f"parent_id = {parent_id}\n",
            f"parent_id = {parent_id}\n"
            "start_time = 2026-07-23T00:00:00 # domain clock\n")
    text = text.replace(
        "start_time = 2026-07-23T00:00:00\n",
        "start_time = 2026-07-23T00:00:00 # cycle clock\n")
    text = text.replace(
        "run_seconds = 3600.0\n",
        "run_seconds = 3600.0 # forecast duration\n")
    if syntax in ("basic-quoted", "literal-quoted", "escaped"):
        def quote(name):
            if syntax == "literal-quoted":
                return f"'{name}'"
            if syntax == "escaped":
                name = f"\\u{ord(name[0]):04x}{name[1:]}"
            return f'"{name}"'

        text = re.sub(
            r"(?m)^(\s*)(\[\[?)([A-Za-z_]+)(\]\]?)",
            lambda match: (match[1] + match[2] + quote(match[3])
                           + match[4]), text)
        text = re.sub(
            r"(?m)^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=)",
            lambda match: match[1] + quote(match[2]) + match[3], text)
    elif syntax == "header-comments":
        text = re.sub(
            r"(?m)^(\s*\[\[?[A-Za-z_]+\]\]?)$",
            r"\1 # retained table comment", text)
    else:
        text = text.replace(
            'name = "stream-tree"',
            'name = """stream-tree\n[experiment]\n'
            'start_time = 1900-01-01T00:00:00\n[[domain]]\n"""')
        text = text.replace(
            "eta_levels = [1.0,", "eta_levels = [\n# ] is only a comment\n1.0,")
        text = text.replace(
            "run_seconds = 3600.0 # forecast duration",
            'run_seconds = """\n3600\n""" # forecast duration')
    plan.experiment_config.write_text(text, encoding="utf-8")
    plan = stream.load_stream_plan(plan.path)
    before = tomllib.loads(text)
    cycle = datetime(2026, 7, 23, 2)
    destination = tmp_path / "materialized" / "experiment.toml"

    stream._materialize_experiment(plan, destination, cycle=cycle, lead=2)

    rendered = destination.read_text(encoding="utf-8")
    expected = before
    expected["experiment"]["start_time"] = cycle
    expected["experiment"]["run_seconds"] = 7200
    for domain in expected["domain"]:
        domain["start_time"] = cycle
    assert tomllib.loads(rendered) == expected
    loaded = load_experiment(destination)
    assert loaded.start_time == cycle
    assert loaded.run_seconds == 7200
    assert [loaded.domain_start_time(domain.grid_id)
            for domain in loaded.domains] == [cycle, cycle]
    assert rendered.count("# domain clock") == 2
    assert "# cycle clock" in rendered
    assert "# forecast duration" in rendered
    assert plan.experiment_config.read_text(encoding="utf-8") == text


def test_stream_controller_refuses_ambient_child_registry_override(
        tmp_path, monkeypatch):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=1)
    monkeypatch.setenv(
        stream._PINNED_PHYSICS_REGISTRY_ENV, str(tmp_path / "other.json"))
    monkeypatch.setenv(
        stream._PINNED_PHYSICS_REGISTRY_SHA256_ENV, "a" * 64)

    with pytest.raises(ValueError, match="reserved for stream child"):
        stream.load_stream_plan(plan.path)


def _run_stream_cli(plan: Path) -> subprocess.CompletedProcess[str]:
    """`woof stream PLAN`, with the provenance banner silenced.

    Every one of these cases asserts that stderr carries the refusal and
    NOTHING else -- one line, no traceback -- and that is only true of a
    caller who asked for quiet.  The provenance banner is one line on
    stderr at every front door by design, so once it landed these three
    read as three broken refusals when the refusals were exactly right.
    `WOOF_PROVENANCE_BANNER=0` is the banner's own documented switch and
    is how tests/test_hrrr_domain_cli.py pins the same property.
    """

    environment = dict(os.environ)
    environment["WOOF_PROVENANCE_BANNER"] = "0"
    return subprocess.run(
        [sys.executable, "-m", "woof.cli", "stream", str(plan)],
        cwd=Path(__file__).resolve().parents[1], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        env=environment)


def test_stream_cli_missing_plan_is_one_line_exit_two(tmp_path):
    result = _run_stream_cli(tmp_path / "definitely-missing-plan.toml")

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith(
        "woof stream: stream plan load refused:")
    assert "Traceback" not in result.stderr
    assert len(result.stderr.splitlines()) == 1


def test_stream_cli_directory_plan_is_one_line_exit_two(tmp_path):
    result = _run_stream_cli(tmp_path)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith(
        "woof stream: stream plan load refused:")
    assert "Traceback" not in result.stderr
    assert len(result.stderr.splitlines()) == 1


def test_stream_cli_missing_declared_prepare_input_is_one_line_exit_two(
        tmp_path):
    plan = tmp_path / "missing-input-stream.toml"
    plan.write_text(textwrap.dedent(f"""
        schema = "{stream.PLAN_SCHEMA}"

        [stream]
        work_root = "work"

        [prepare]
        experiment_config = "missing-experiment.toml"
        domain_spec = "missing-domain.json"
        wps_namelist = "missing-namelist.wps"
        namelist_input = "missing-namelist.input"
        stock_wrf_namelist_input = "missing-stock-namelist.input"
        geog_root = "missing-geog"
        physics_profile = "generic-profile-v1"
    """).lstrip(), encoding="utf-8")

    result = _run_stream_cli(plan)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith(
        "woof stream: stream plan load refused:")
    assert "[prepare] experiment_config" in result.stderr
    assert "Traceback" not in result.stderr
    assert len(result.stderr.splitlines()) == 1


def _arg(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


class FakeBackend:
    """Deterministic public-stage double which still writes real artifacts."""

    def __init__(self, cycles, *, fail_hierarchy_once=False):
        self.cycles = list(cycles)
        self._next_cycle = 1
        self._now = datetime(2026, 7, 23, tzinfo=timezone.utc)
        self.fail_hierarchy_once = fail_hierarchy_once
        self.downloads = Counter()
        self.fetch_calls = []
        self.commands = []
        self.latest_calls = 0
        self.next_cycle_calls = []
        self.availability_calls = []

    def now(self):
        value = self._now
        self._now += timedelta(seconds=7)
        return value

    def resolve_latest(self):
        self.latest_calls += 1
        return self.cycles[0]

    def wait_for_next_cycle(self, after, *, timeout_s, poll_s):
        self.next_cycle_calls.append((after, timeout_s, poll_s))
        value = self.cycles[self._next_cycle]
        self._next_cycle += 1
        assert value > after
        self._now += timedelta(minutes=50)
        return value

    def wait_for_hour(self, cycle, lead, *, timeout_s, poll_s):
        del timeout_s, poll_s
        self.availability_calls.append((cycle, lead))
        self._now += timedelta(minutes=5)
        observed = stream._iso(self.now())
        products = {}
        for product in ("wrfnat", "wrfprs"):
            products[product] = {
                "transport": "fixture",
                "object_url": f"fixture://{product}/f{lead:03d}",
                "index_url": f"fixture://{product}/f{lead:03d}.idx",
                "object": {
                    "url": f"fixture://{product}/f{lead:03d}",
                    "content_length_bytes": 1024 + lead,
                    "remote_last_modified": observed,
                    "etag": None,
                    "first_observed_at": observed,
                },
                "index": {
                    "url": f"fixture://{product}/f{lead:03d}.idx",
                    "content_length_bytes": 64,
                    "remote_last_modified": observed,
                    "etag": None,
                    "first_observed_at": observed,
                },
            }
        return {
            "first_observed_at": observed,
            "remote_ready_last_modified_at": observed,
            "object_content_length_bytes": 2 * (1024 + lead),
            "products": products,
        }

    def free_disk_bytes(self, _path):
        return 1024 ** 4

    def fetch_prefix(self, plan, cycle, lead, source_root):
        del plan
        source_root.mkdir(parents=True, exist_ok=True)
        self.fetch_calls.append((cycle, lead))
        rows = []
        files = []
        for hour in range(lead + 1):
            for role, product in (("atmosphere", "wrfnat"),
                                  ("soil", "wrfprs")):
                data = source_root / f"{role}-f{hour:03d}.bin"
                if not data.exists():
                    data.write_bytes(
                        f"{cycle:%Y%m%d%H}:{hour}:{role}\n".encode())
                    if role == "atmosphere":
                        self.downloads[(cycle, hour)] += 1
                digest = hashlib.sha256(data.read_bytes()).hexdigest()
                rows.append(f"{digest}  {data.name}")
                files.append({
                    "name": data.name,
                    "role": role,
                    "forecast_hour": hour,
                    "bytes": data.stat().st_size,
                    "sha256": digest,
                    "url": f"fixture://{product}/f{hour:03d}",
                    "transport": "fixture",
                })
        (source_root / "SHA256SUMS").write_text(
            "\n".join(rows) + "\n", encoding="utf-8")
        sums = source_root / "SHA256SUMS"
        files.append({
            "name": sums.name,
            "role": "checksums",
            "forecast_hour": None,
            "bytes": sums.stat().st_size,
            "sha256": hashlib.sha256(sums.read_bytes()).hexdigest(),
            "url": None,
            "transport": None,
        })
        manifest = source_root / "fetch-manifest.json"
        _write_json(manifest, {
            "schema": "gpuwm-fetch-manifest-v1",
            "source": "hrrr",
            "cycle": cycle.strftime("%Y-%m-%dT%H:00:00Z"),
            "forecast_hours": list(range(lead + 1)),
            "files": files,
        })
        return manifest

    def run_command(self, argv, *, stage):
        argv = list(argv)
        self.commands.append((stage, argv))
        if "woof.source_cli" in argv and "--root-preparation" not in argv:
            output = Path(_arg(argv, "--output-root"))
            cycle = datetime.strptime(
                _arg(argv, "--valid-time"), "%Y-%m-%d_%H:%M:%S")
            lead = int(_arg(argv, "--forecast-end-hour"))
            hours = list(range(lead + 1))
            report = output / "native" / "preparation-report" / "report.json"
            native_root = output / "native"
            bridge = native_root / "native-bridge"
            bridge.mkdir(parents=True)
            bridge_payload = bridge / "decoded.bin"
            bridge_payload.write_bytes(
                f"{cycle.isoformat()}:{hours}".encode("ascii"))
            (bridge / "SHA256SUMS").write_text(
                f"{hashlib.sha256(bridge_payload.read_bytes()).hexdigest()}"
                f"  {bridge_payload.name}\n", encoding="utf-8")
            static = output / "native-static.npz"
            with static.open("wb") as handle:
                np.savez(handle, DUMMY=np.ones((1,), dtype=np.float32))
            _write_json(output / "native-static-receipt.json", {
                "status": "PASS",
            })
            _write_json(output / "native-geometry-receipt.json", {
                "status": "PASS",
            })
            initial, met, base_boundaries = _prepared_cache_fixture()
            template = base_boundaries.intervals[0]
            base_side = template.fields["u"].west
            intervals = []
            for index in range(lead):
                value = (base_side.value
                         + index * 3600.0 * base_side.tendency)
                side = SideBoundary(value, base_side.tendency)
                intervals.append(BoundaryInterval(
                    index * 3600.0, (index + 1) * 3600.0,
                    {"u": FieldBoundary(side, side, side, side)}))
            boundaries = LateralBoundaries(
                tuple(intervals), base_boundaries.spec_bdy_width,
                base_boundaries.spec_zone, base_boundaries.relax_zone)
            initial.state.lateral_boundaries = boundaries
            source_sums = Path(_arg(argv, "--source-manifest"))
            namelist = Path(_arg(argv, "--namelist-input"))
            identity = {
                "bridge_manifest_sha256": hashlib.sha256(
                    (bridge / "SHA256SUMS").read_bytes()).hexdigest(),
                "source_manifest_sha256": hashlib.sha256(
                    source_sums.read_bytes()).hexdigest(),
                "static_cache_sha256": hashlib.sha256(
                    static.read_bytes()).hexdigest(),
                "namelist_sha256": hashlib.sha256(
                    namelist.read_bytes()).hexdigest(),
                "domain_config": {
                    "run": {"run_seconds": float(lead * 3600),
                            "specified": True, "nested": False}},
                "forcing_hours": hours,
                "source_identity": {
                    "adapter": "fixture", "source_cycle": cycle.isoformat(),
                    "source_forecast_hours": hours,
                    "model_forcing_hours": hours,
                },
            }
            cache_receipt = write_prepared_cache(
                native_root / "prepared-cache", identity=identity,
                initial_result=initial, met=met, boundaries=boundaries,
                metadata={"source_forecast_hours": hours},
                sealed_forcing_extension=True)
            predecessor = (
                Path(_arg(argv, "--extend-root-preparation")).resolve()
                if "--extend-root-preparation" in argv else None)
            _write_json(output / "public-wrapper-result.json", {
                "status": "PASS",
                "source_cycle": cycle.isoformat(),
                "model_start_time": cycle.isoformat(),
                "source_forecast_hours": hours,
                "model_forcing_hours": hours,
                "forcing_hours": hours,
                "history_interval_seconds": float(
                    _arg(argv, "--history-interval-seconds")),
                "preparation_report": str(report.resolve()),
                "physics": {"profile": _arg(argv, "--physics-profile")},
                "prepared_cache_contract": {
                    "mode": "sealed-prefix-v1",
                    "operation": ("extend-one-hour"
                                  if predecessor is not None else "initial"),
                    "predecessor": (
                        None if predecessor is None else str(predecessor)),
                    "content_sha256": cache_receipt["content_sha256"],
                },
            })
            _write_json(report, {
                "status": "PASS",
                "source_cycle": cycle.isoformat(),
                "model_start_time": cycle.isoformat(),
                "source_forecast_hours": hours,
                "model_forcing_hours": hours,
                "target_domain_sha256": load_hrrr_target_domain(
                    _arg(argv, "--domain-spec")).identity_sha256(),
                "input": {
                    "bridge": str(bridge.resolve()),
                    "bridge_manifest_sha256": identity[
                        "bridge_manifest_sha256"],
                    "source_manifest_sha256": identity[
                        "source_manifest_sha256"],
                    "source_forecast_hours": hours,
                    "model_forcing_hours": hours,
                    "forcing_hours": hours,
                },
                "prepared_cache": cache_receipt,
            })
            return
        if "woof.source_cli" in argv:
            if self.fail_hierarchy_once:
                self.fail_hierarchy_once = False
                raise RuntimeError("simulated hierarchy interruption")
            output = Path(_arg(argv, "--output-root"))
            root = Path(_arg(argv, "--root-preparation"))
            wrapper = json.loads(
                (root / "public-wrapper-result.json").read_text())
            source_sums = Path(_arg(argv, "--source-manifest"))
            wps = Path(_arg(argv, "--wps-namelist"))
            native = Path(_arg(argv, "--namelist-input"))
            stock = Path(_arg(argv, "--stock-wrf-namelist-input"))
            root_header = json.loads((
                root / "native" / "prepared-cache" / "header.json"
            ).read_text())
            _write_json(output / "receipt.json", {
                "schema": "gpuwm-native-hrrr-hierarchy-direct-v1",
                "status": "PASS",
                "valid_time": datetime.strptime(
                    _arg(argv, "--valid-time"),
                    "%Y-%m-%d_%H:%M:%S").isoformat(),
                "domain_count": 2,
                "forcing_hours": wrapper["forcing_hours"],
                "provenance": {
                    "source_manifest_sha256": hashlib.sha256(
                        source_sums.read_bytes()).hexdigest(),
                    "wps_namelist_sha256": hashlib.sha256(
                        wps.read_bytes()).hexdigest(),
                    "native_namelist_input_sha256": hashlib.sha256(
                        native.read_bytes()).hexdigest(),
                    "stock_wrf_namelist_input_sha256": hashlib.sha256(
                        stock.read_bytes()).hexdigest(),
                    "root_static_receipt_sha256": hashlib.sha256(
                        (root / "native-static-receipt.json").read_bytes()
                    ).hexdigest(),
                    "root_prepared_content_sha256": root_header[
                        "content_sha256"],
                },
            })
            artifact_root = output / "hierarchy-artifacts"
            manifest_domains = []
            for grid_id in (1, 2):
                bundle = artifact_root / "domains" / f"d{grid_id:02d}"
                bundle.mkdir(parents=True)
                shutil.copytree(
                    root / "native" / "prepared-cache",
                    bundle / "prepared-cache")
                shutil.copy2(
                    root / "native-static.npz",
                    bundle / "native-static.npz")
                _write_json(bundle / "geometry-receipt.json", {
                    "geometry": {"grid_id": grid_id}})
                _write_json(bundle / "receipt.json", {
                    "status": "READY", "grid_id": grid_id})
                manifest_domains.append({
                    "grid_id": grid_id,
                    "prepared_cache":
                        f"domains/d{grid_id:02d}/prepared-cache",
                    "static_cache":
                        f"domains/d{grid_id:02d}/native-static.npz",
                    "geometry_receipt":
                        f"domains/d{grid_id:02d}/geometry-receipt.json",
                })
            _write_json(artifact_root / "domain-artifacts.json", {
                "schema": "gpuwm-native-domain-artifacts-v1",
                "domains": manifest_domains,
            })
            _write_json(
                artifact_root / "receipt.json",
                {"status": "READY", "domain_count": 2,
                 "grid_ids": [1, 2]},
            )
            return
        assert "woof.prepared_domain_tree_forecast" in argv
        output = Path(_arg(argv, "--outdir"))
        experiment = load_experiment(_arg(argv, "--experiment-config"))
        cycle = experiment.start_time
        lead = int(experiment.run_seconds // 3600)
        set_id = f"{cycle:%Y%m%dT%H}-f{lead:03d}"
        fingerprint = f"fixture-{cycle:%Y%m%dT%H}-f{lead:03d}"
        checkpoints = []
        for grid_id in (1, 2):
            checkpoint = output / (
                f"gpuwmrst_d{grid_id:02d}_{cycle:%Y-%m-%d_%H}_00_00"
                f"__{set_id}.npz")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            header = {
                "checkpoint_set_id": set_id,
                "domain_ids": [1, 2],
                "grid_id": grid_id,
                "elapsed_seconds": float(lead * 3600),
                "experiment_fingerprint": fingerprint,
                "forcing_extension_mode": "sealed-prefix-v1",
            }
            with checkpoint.open("wb") as handle:
                np.savez(handle, __gpuwm_restart_header__=np.frombuffer(
                    json.dumps(header).encode(), dtype=np.uint8))
            checkpoints.append(checkpoint)
        health = {
            "d01": {"ok": True},
            "d02": {"ok": True},
        }
        frame = output / "history" / f"frame-f{lead:03d}.bin"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(f"history:{cycle.isoformat()}:{lead}".encode())
        frame_row = {
            "path": str(frame.resolve()),
            "bytes": frame.stat().st_size,
            "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
        }
        _write_json(output / "evidence" / "run-receipt.json", {
            "schema": "gpuwm-prepared-domain-tree-forecast-v1",
            "status": "PASS",
            "experiment": {
                "start_time": cycle.isoformat(),
                "run_seconds": float(lead * 3600),
                "fingerprint": fingerprint,
            },
            "restart_contract": {
                "mode": "sealed-forcing-extension",
                "restart_input": (
                    str(Path(_arg(argv, "--restart")).resolve())
                    if "--restart" in argv else None),
            },
            "health": {"initial": health, "final": health},
            "input": {
                "prepared_root": str(Path(
                    _arg(argv, "--prepared-root")).resolve()),
                "forcing_hours": list(range(lead + 1)),
                "authority_sha256": {
                    "experiment_config": hashlib.sha256(Path(
                        _arg(argv, "--experiment-config")).read_bytes()
                    ).hexdigest(),
                    "preparation_receipt": hashlib.sha256(Path(
                        _arg(argv, "--prepared-root"),
                        "receipt.json").read_bytes()).hexdigest(),
                },
            },
            "output": {
                "io_mode": _arg(argv, "--io-mode"),
                "frame_count": 1,
                "total_bytes": frame.stat().st_size,
                "files": [frame_row],
                "last_checkpoint": str(checkpoints[0].resolve()),
            },
        })


def test_one_invocation_runs_two_cycles_and_durable_hourly_chain(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=2, target_lead=2)
    cycles = [datetime(2026, 7, 23, hour) for hour in (1, 2)]
    backend = FakeBackend(cycles)

    result = stream.run_stream(plan, backend=backend, progress=lambda _: None)

    assert result["status"] == "PASS"
    assert result["completed_cycle_count"] == 2
    assert backend.latest_calls == 1
    assert [row[0] for row in backend.next_cycle_calls] == [cycles[0]]
    assert backend.fetch_calls == [
        (cycles[0], 1), (cycles[0], 2),
        (cycles[1], 1), (cycles[1], 2),
    ]
    assert set(backend.downloads.values()) == {1}
    assert set(backend.downloads) == {
        (cycle, hour) for cycle in cycles for hour in range(3)
    }

    runs = [argv for stage, argv in backend.commands
            if "sealed forecast" in stage]
    assert len(runs) == 4
    assert all("--sealed-forcing-extension" in argv for argv in runs)
    assert ["--restart" in argv for argv in runs] == [False, True, False, True]

    for cycle in cycles:
        cycle_root = plan.work_root / "cycles" / cycle.strftime("%Y%m%dT%H")
        first = json.loads(
            (cycle_root / "legs" / "f001" / "chain-link.json").read_text())
        second = json.loads(
            (cycle_root / "legs" / "f002" / "chain-link.json").read_text())
        assert second["previous_link_sha256"] == first["link_sha256"]
        assert second["restart_input"] == first["checkpoint_root"]
        assert any(item["role"].startswith("restart_input.d")
                   for item in second["artifacts"])
        assert len(second["artifacts"]) > len(first["artifacts"])
        assert load_experiment(
            cycle_root / "legs" / "f001" / "configs" / "experiment.toml"
        ).run_seconds == 3600.0
        assert load_experiment(
            cycle_root / "legs" / "f002" / "configs" / "experiment.toml"
        ).run_seconds == 7200.0
        summary = json.loads((cycle_root / "chain-summary.json").read_text())
        assert summary["completed_leads"] == [1, 2]
        assert all(row["validity"] == "PASS" for row in summary["timeline"])
        assert all(row["forcing_set_first_observed_at"]
                   for row in summary["timeline"])
        for row in summary["timeline"]:
            assert row["root_preparation_started_at"]
            assert row["root_preparation_completed_at"]
            assert row["root_preparation_seconds"] >= 0.0
            assert row["hierarchy_preparation_started_at"]
            assert row["hierarchy_preparation_completed_at"]
            assert row["hierarchy_preparation_seconds"] >= 0.0
            assert row["preparation_completed_at"] == \
                row["hierarchy_preparation_completed_at"]
            assert row["forecast_started_at"]
            assert row["forecast_completed_at"] == row["leg_completed_at"]

    assert result["cycles"][1]["previous_cycle_link_sha256"] == \
        result["cycles"][0]["cycle_link_sha256"]

    before = (
        len(backend.fetch_calls), len(backend.commands),
        len(backend.availability_calls), backend.latest_calls,
    )
    adopted = stream.run_stream(
        plan, backend=backend, progress=lambda _: None)
    assert adopted == result
    assert (
        len(backend.fetch_calls), len(backend.commands),
        len(backend.availability_calls), backend.latest_calls,
    ) == before

    checkpoint = Path(result["cycles"][0]["chain_summary"]["path"]).parent \
        / "legs" / "f002" / "run"
    checkpoint = next(checkpoint.glob("gpuwmrst_d01_*.npz"))
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed or disappeared"):
        stream.run_stream(plan, backend=backend, progress=lambda _: None)


def test_completed_production_replay_deep_verifies_without_gpu(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=1)
    cycle = datetime(2026, 7, 23, 1)
    built = stream.run_stream(
        plan, backend=FakeBackend([cycle]), progress=lambda _: None)

    class VerificationOnlyProductionBackend(stream.ProductionBackend):
        def gpu_allocation(self, _plan):
            raise AssertionError(
                "completed verification-only replay entered GPU allocation")

    replay = VerificationOnlyProductionBackend(progress=lambda _: None)
    assert stream.run_stream(
        plan, backend=replay, progress=lambda _: None) == built

    bridge_manifest = (
        plan.work_root / "cycles" / cycle.strftime("%Y%m%dT%H") /
        "legs" / "f001" / "root-preparation" / "native" /
        "native-bridge" / "SHA256SUMS")
    bridge_manifest.write_bytes(bridge_manifest.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="changed or disappeared"):
        stream.run_stream(plan, backend=replay, progress=lambda _: None)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("completed_cycle_count", 999, "completed count disagrees"),
        ("latest_cycle_link_sha256", "0" * 64, "tail disagrees"),
        ("status", "RUNNING", "already contains the requested"),
        ("requested_cycle_count", 999, "another plan or schema"),
    ),
)
def test_completed_replay_refuses_forged_outer_program_without_gpu(
        tmp_path, field, value, match):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=1)
    cycle = datetime(2026, 7, 23, 1)
    stream.run_stream(
        plan, backend=FakeBackend([cycle]), progress=lambda _: None)
    program_path = plan.work_root / "stream-summary.json"
    forged = json.loads(program_path.read_text())
    forged[field] = value
    _write_json(program_path, forged)

    class VerificationOnlyProductionBackend(stream.ProductionBackend):
        def gpu_allocation(self, _plan):
            raise AssertionError(
                "forged completed replay entered GPU allocation")

    with pytest.raises(ValueError, match=match):
        stream.run_stream(
            plan,
            backend=VerificationOnlyProductionBackend(
                progress=lambda _: None),
            progress=lambda _: None)


def test_completed_replay_refuses_nonhourly_outer_cycle_chain_without_gpu(
        tmp_path):
    plan = _make_plan(tmp_path, cycle_count=2, target_lead=1)
    cycles = [datetime(2026, 7, 23, hour) for hour in (1, 2)]
    built = stream.run_stream(
        plan, backend=FakeBackend(cycles), progress=lambda _: None)
    first = built["cycles"][0]
    first_path = Path(first["chain_summary"]["path"])
    first_summary = json.loads(first_path.read_text())
    duplicate = stream._cycle_record(
        first_summary, first_path, first["cycle_link_sha256"])
    forged = stream._program_payload(
        plan, [first, duplicate], status="PASS",
        started_at=built["started_at"])
    _write_json(plan.work_root / "stream-summary.json", forged)

    class VerificationOnlyProductionBackend(stream.ProductionBackend):
        def gpu_allocation(self, _plan):
            raise AssertionError(
                "nonhourly completed replay entered GPU allocation")

    with pytest.raises(ValueError, match="not exact hourly successors"):
        stream.run_stream(
            plan,
            backend=VerificationOnlyProductionBackend(
                progress=lambda _: None),
            progress=lambda _: None)


@pytest.mark.parametrize(
    ("corruption", "match"),
    (
        ("cache-header", "changed or disappeared"),
        ("cache-array-append", "changed or disappeared"),
        ("cache-array-flip", "changed or disappeared"),
        ("bridge-manifest", "changed or disappeared"),
        ("bridge-payload", "SHA256SUMS payload changed or disappeared"),
        ("hierarchy-receipt", "changed or disappeared"),
        ("hierarchy-cache-array", "changed or disappeared"),
        ("hierarchy-static-delete", "changed or disappeared"),
        ("named-frame", "output frame changed or disappeared"),
    ),
)
def test_resume_reverifies_deep_leg_artifacts(
        tmp_path, corruption, match):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=1)
    cycle = datetime(2026, 7, 23, 2)
    backend = FakeBackend([cycle])
    result = stream.run_stream(
        plan, backend=backend, progress=lambda _: None)
    assert result["status"] == "PASS"
    leg = (plan.work_root / "cycles" / cycle.strftime("%Y%m%dT%H") /
           "legs" / "f001")
    cache = leg / "root-preparation" / "native" / "prepared-cache"
    if corruption == "cache-header":
        target = cache / "header.json"
    elif corruption.startswith("cache-array"):
        header = json.loads((cache / "header.json").read_text())
        target = cache / next(iter(header["arrays"].values()))["file"]
    elif corruption.startswith("bridge-"):
        bridge = leg / "root-preparation" / "native" / "native-bridge"
        if corruption == "bridge-manifest":
            target = bridge / "SHA256SUMS"
        else:
            target = next(
                path for path in bridge.rglob("*")
                if path.is_file() and path.name != "SHA256SUMS")
    elif corruption == "hierarchy-receipt":
        target = leg / "prepared-hierarchy" / "receipt.json"
    elif corruption == "hierarchy-cache-array":
        hierarchy_cache = (
            leg / "prepared-hierarchy" / "hierarchy-artifacts" /
            "domains" / "d01" / "prepared-cache")
        hierarchy_header = json.loads((
            hierarchy_cache / "header.json").read_text())
        target = hierarchy_cache / next(iter(
            hierarchy_header["arrays"].values()))["file"]
    elif corruption == "hierarchy-static-delete":
        target = (
            leg / "prepared-hierarchy" / "hierarchy-artifacts" /
            "domains" / "d01" / "native-static.npz")
    else:
        receipt = json.loads((
            leg / "run" / "evidence" / "run-receipt.json").read_text())
        target = Path(receipt["output"]["files"][0]["path"])
    original = target.read_bytes()
    if corruption == "hierarchy-static-delete":
        target.unlink()
    elif corruption == "cache-array-flip":
        changed = bytearray(original)
        changed[-1] ^= 1
        target.write_bytes(changed)
    else:
        target.write_bytes(original + b"corrupt")
    before = (len(backend.fetch_calls), len(backend.commands))

    with pytest.raises((ValueError, RuntimeError), match=match):
        stream.run_stream(plan, backend=backend, progress=lambda _: None)

    assert (len(backend.fetch_calls), len(backend.commands)) == before


def test_interruption_resumes_without_refetching_verified_prefix(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=2)
    cycle = datetime(2026, 7, 23, 3)
    backend = FakeBackend([cycle], fail_hierarchy_once=True)

    with pytest.raises(RuntimeError, match="simulated hierarchy interruption"):
        stream.run_stream(plan, backend=backend, progress=lambda _: None)

    assert backend.downloads == Counter({(cycle, 0): 1, (cycle, 1): 1})
    availability = (
        plan.work_root / "cycles" / cycle.strftime("%Y%m%dT%H")
        / "legs" / "f001" / "availability.json")
    observed_bytes = availability.read_bytes()

    result = stream.run_stream(
        plan, backend=backend, progress=lambda _: None)

    assert result["status"] == "PASS"
    assert set(backend.downloads.values()) == {1}
    assert backend.downloads[(cycle, 2)] == 1
    assert backend.fetch_calls == [(cycle, 1), (cycle, 1), (cycle, 2)]
    assert availability.read_bytes() == observed_bytes
    assert backend.latest_calls == 1


def test_production_fetch_is_rust_full_file_contiguous_prefix(
        tmp_path, monkeypatch):
    calls = {}
    source = tmp_path / "source"

    def resolve(requested, *, progress):
        calls["resolved"] = (requested, progress)
        return "rust", Path("fetch-engine")

    def prior(root, **kwargs):
        calls["prior"] = (root, kwargs)

    def fetch(**kwargs):
        calls["fetch"] = kwargs
        return source / "fetch-manifest.json"

    monkeypatch.setattr(stream, "resolve_fetch_engine", resolve)
    monkeypatch.setattr(stream, "check_prior_request", prior)
    monkeypatch.setattr(stream, "fetch_hrrr", fetch)
    def progress(_):
        return None
    backend = stream.ProductionBackend(progress=progress)
    cycle = datetime(2026, 7, 23, 4)
    plan = type("Plan", (), {
        "area": None,
        "wait_timeout_seconds": 12.0,
        "cache_dir": None,
    })()

    backend.preflight_fetch(plan)
    result = backend.fetch_prefix(plan, cycle, 3, source)

    assert result == source / "fetch-manifest.json"
    assert calls["resolved"] == ("rust", progress)
    assert calls["fetch"]["engine"] == "rust"
    assert calls["fetch"]["engine_bin"] == Path("fetch-engine")
    assert calls["fetch"]["mode"] == HRRR_DEFAULT_MODE
    assert calls["fetch"]["hours"] == (0, 1, 2, 3)
    assert calls["fetch"]["transport"] == "s3"
    assert calls["fetch"]["wait"] is True


def test_advancing_production_stream_preflights_fetch_before_wait_or_gpu(
        tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=1)
    events = []

    class MissingFetchBackend(stream.ProductionBackend):
        def preflight_fetch(self, _plan):
            events.append("fetch-preflight")
            raise RuntimeError("required Rust fetcher is absent")

        def resolve_latest(self):
            events.append("resolve-latest")
            raise AssertionError("cycle selection preceded fetch preflight")

        def gpu_allocation(self, _plan):
            events.append("gpu-allocation")
            raise AssertionError("GPU allocation preceded fetch preflight")

    with pytest.raises(RuntimeError, match="required Rust fetcher is absent"):
        stream.run_stream(
            plan, backend=MissingFetchBackend(progress=lambda _: None),
            progress=lambda _: None)

    assert events == ["fetch-preflight"]


def test_production_availability_observes_both_objects_and_indexes():
    cycle = datetime(2026, 7, 23, 5)
    observed = datetime(2026, 7, 23, 6, 7, tzinfo=timezone.utc)
    available = set()
    for product in ("wrfnat", "wrfprs"):
        url = hrrr_object_url(cycle, 1, product, transport="s3")
        available.update((url, url + ".idx"))
    backend = stream.ProductionBackend(
        progress=lambda _: None,
        probe=lambda url: url in available,
        now=lambda: observed,
        monotonic=lambda: 0.0,
    )

    result = backend.wait_for_hour(cycle, 1, timeout_s=10.0, poll_s=1.0)

    assert result["first_observed_at"] == "2026-07-23T06:07:00Z"
    assert result["remote_ready_last_modified_at"] is None
    assert set(result["products"]) == {"wrfnat", "wrfprs"}
    assert result["products"]["wrfnat"]["transport"] == "s3"
    assert result["products"]["wrfprs"]["transport"] == "s3"


def test_availability_refuses_a_different_effective_fetch_transport():
    observation = {
        "products": {
            product: {
                "transport": "s3",
                "object_url": f"https://s3/{product}",
                "index_url": f"https://s3/{product}.idx",
            }
            for product in ("wrfnat", "wrfprs")
        },
    }
    manifest = {"files": [
        {
            "role": role, "forecast_hour": 1,
            "transport": ("nomads" if role == "atmosphere" else "s3"),
            "url": f"https://s3/{product}", "sha256": "a" * 64,
            "bytes": 1,
        }
        for role, product in (("atmosphere", "wrfnat"),
                              ("soil", "wrfprs"))
    ]}

    with pytest.raises(ValueError, match="effective wrfnat"):
        stream._bind_fetch_transport(observation, manifest, lead=1)


def test_s3_fetch_binding_rechecks_remote_identity_and_downloaded_bytes(
        tmp_path):
    products = {}
    files = []
    for role, product in (("atmosphere", "wrfnat"), ("soil", "wrfprs")):
        url = f"https://s3.example/{product}"
        payload = f"full-{product}".encode("ascii")
        name = f"{product}.grib2"
        (tmp_path / name).write_bytes(payload)
        products[product] = {
            "transport": "s3", "object_url": url,
            "index_url": url + ".idx",
            "object": {
                "url": url, "content_length_bytes": len(payload),
                "etag": f'"{product}-etag"'},
            "index": {
                "url": url + ".idx", "content_length_bytes": 17,
                "etag": f'"{product}-index-etag"'},
        }
        files.append({
            "role": role, "forecast_hour": 1, "transport": "s3",
            "url": url, "name": name, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    before = {"products": products}
    after = json.loads(json.dumps(before))

    bound = stream._bind_fetch_transport(
        before, {"files": files}, lead=1, source_root=tmp_path,
        post_fetch_observation=after)

    assert bound["effective_fetch"]["wrfnat"][
        "stable_remote_identity"]["pre_fetch"]["etag"] == \
        '"wrfnat-etag"'
    after["products"]["wrfnat"]["object"]["etag"] = '"changed"'
    with pytest.raises(ValueError, match="URL/Content-Length/ETag changed"):
        stream._bind_fetch_transport(
            before, {"files": files}, lead=1, source_root=tmp_path,
            post_fetch_observation=after)


def test_watcher_close_joins_and_forbids_late_publication(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=1)
    cycle = datetime(2026, 7, 23, 7)
    cycle_root = tmp_path / "watcher-cycle"

    class ClosingBackend:
        def __init__(self):
            self.started = threading.Barrier(3)

        def wait_for_hour(self, cycle, lead, *, timeout_s, poll_s,
                          stop_event=None):
            del cycle, lead, timeout_s, poll_s
            self.started.wait(timeout=5.0)
            assert stop_event.wait(timeout=5.0)
            return {
                "first_observed_at": "2026-07-23T07:00:00Z",
                "remote_ready_last_modified_at": None,
                "object_content_length_bytes": 1,
                "products": {},
            }

    backend = ClosingBackend()
    watcher = stream._AvailabilityWatcher(
        cycle_root, backend=backend, cycle=cycle, target=1,
        plan=plan, progress=lambda _: None)
    watcher.start()
    backend.started.wait(timeout=5.0)
    watcher.close()

    assert all(not thread.is_alive() for thread in watcher.threads)
    assert not list(cycle_root.rglob("*.json"))


def test_watcher_partial_start_failure_closes_earlier_workers(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=2)
    cycle = datetime(2026, 7, 23, 7)
    cycle_root = tmp_path / "partial-start-cycle"
    # f000 starts a worker before start() reaches this malformed f001
    # observation.  The failure must cancel and join that earlier worker.
    _write_json(
        cycle_root / "legs" / "f001" / "availability.json",
        {"schema": "not-a-stream-observation"})

    class BlockingBackend:
        def __init__(self):
            self.started = threading.Event()
            self.stopped = threading.Event()

        def wait_for_hour(self, cycle, lead, *, timeout_s, poll_s,
                          stop_event=None):
            del cycle, lead, timeout_s, poll_s
            self.started.set()
            assert stop_event.wait(timeout=5.0)
            self.stopped.set()
            raise RuntimeError("cancelled")

    backend = BlockingBackend()
    watcher = stream._AvailabilityWatcher(
        cycle_root, backend=backend, cycle=cycle, target=2,
        plan=plan, progress=lambda _: None)

    with pytest.raises(ValueError, match="identity mismatch"):
        watcher.start()

    assert backend.started.is_set()
    assert backend.stopped.is_set()
    assert all(not thread.is_alive() for thread in watcher.threads)
    assert not (cycle_root / "availability" / "f000.json").exists()


def test_production_next_cycle_probes_exact_hour_not_latest():
    after = datetime(2026, 7, 23, 11)
    expected = after + timedelta(hours=1)
    seen = []
    available = set()
    for product in ("wrfnat", "wrfprs"):
        url = hrrr_object_url(expected, 0, product, transport="s3")
        available.update((url, url + ".idx"))

    def probe(url):
        seen.append(url)
        return url in available

    backend = stream.ProductionBackend(
        progress=lambda _: None, probe=probe,
        now=lambda: datetime(2026, 7, 23, 12, 5, tzinfo=timezone.utc),
        monotonic=lambda: 0.0)

    assert backend.wait_for_next_cycle(
        after, timeout_s=30.0, poll_s=1.0) == expected
    assert seen
    assert all(f"hrrr.{expected:%Y%m%d}" in url for url in seen)
    assert all(f"t{expected:%H}z" in url for url in seen)


def test_terminal_crash_window_recovers_to_pass(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=1)
    cycle = datetime(2026, 7, 23, 6)
    backend = FakeBackend([cycle])
    started_at = stream._iso(backend.now())
    stream._claim_work_root(plan)
    stream._atomic_json(
        plan.work_root / "stream-summary.json",
        stream._program_payload(
            plan, [], status="RUNNING", started_at=started_at),
    )
    stream._run_cycle(
        plan, backend=backend, progress=lambda _: None)
    command_count = len(backend.commands)

    def forbidden_gpu_allocation(_plan):
        raise AssertionError(
            "terminal crash recovery entered GPU allocation")

    backend.gpu_allocation = forbidden_gpu_allocation

    recovered = stream.run_stream(
        plan, backend=backend, progress=lambda _: None)

    assert recovered["status"] == "PASS"
    assert recovered["completed_cycle_count"] == 1
    assert len(backend.commands) == command_count
    on_disk = json.loads(
        (plan.work_root / "stream-summary.json").read_text())
    assert on_disk == recovered


def test_complete_marker_adoption_refuses_cycle_summary_mismatch_without_gpu(
        tmp_path):
    plan = _make_plan(tmp_path, cycle_count=2, target_lead=1)
    cycles = [datetime(2026, 7, 23, hour) for hour in (6, 7)]
    backend = FakeBackend(cycles)
    started_at = stream._iso(backend.now())
    stream._claim_work_root(plan)
    first = stream._run_cycle(
        plan, backend=backend, progress=lambda _: None)
    first_path = (
        plan.work_root / "cycles" / cycles[0].strftime("%Y%m%dT%H")
        / "chain-summary.json")
    first_record = stream._cycle_record(first, first_path, None)
    stream._atomic_json(
        plan.work_root / "stream-summary.json",
        stream._program_payload(
            plan, [first_record], status="RUNNING", started_at=started_at),
    )
    active_path = plan.work_root / "active-cycle.json"
    active = json.loads(active_path.read_text())
    active["cycle"] = cycles[1].strftime("%Y-%m-%dT%H")
    _write_json(active_path, active)

    def forbidden_gpu_allocation(_plan):
        raise AssertionError(
            "invalid COMPLETE-marker adoption entered GPU allocation")

    backend.gpu_allocation = forbidden_gpu_allocation
    with pytest.raises(ValueError, match="active marker disagrees"):
        stream.run_stream(plan, backend=backend, progress=lambda _: None)


def test_final_cycle_crash_window_recovers_outer_predecessor(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=2, target_lead=1)
    cycles = [datetime(2026, 7, 23, hour) for hour in (7, 8)]
    backend = FakeBackend(cycles)
    started_at = stream._iso(backend.now())
    stream._claim_work_root(plan)
    first = stream._run_cycle(
        plan, backend=backend, progress=lambda _: None)
    first_path = (
        plan.work_root / "cycles" / cycles[0].strftime("%Y%m%dT%H")
        / "chain-summary.json")
    first_record = stream._cycle_record(first, first_path, None)
    stream._atomic_json(
        plan.work_root / "stream-summary.json",
        stream._program_payload(
            plan, [first_record], status="RUNNING", started_at=started_at),
    )
    stream._run_cycle(
        plan, backend=backend, progress=lambda _: None,
        after_cycle=cycles[0])
    command_count = len(backend.commands)

    recovered = stream.run_stream(
        plan, backend=backend, progress=lambda _: None)

    assert recovered["status"] == "PASS"
    assert recovered["completed_cycle_count"] == 2
    assert recovered["cycles"][1]["previous_cycle_link_sha256"] == \
        recovered["cycles"][0]["cycle_link_sha256"]
    assert len(backend.commands) == command_count


def test_cycle_selection_refuses_leapfrog_and_stale_active(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=2, target_lead=1)
    after = datetime(2026, 7, 23, 9)
    backend = FakeBackend([after, after + timedelta(hours=2)])
    with pytest.raises(ValueError, match="no leapfrogging"):
        stream._select_cycle(plan, backend, after_cycle=after)

    stream._atomic_json(plan.work_root / "active-cycle.json", {
        "schema": stream.ACTIVE_SCHEMA,
        "status": "ACTIVE",
        "plan_identity_sha256": plan.identity_sha256,
        "cycle": after.strftime("%Y-%m-%dT%H"),
        "target_lead": 1,
    })
    with pytest.raises(ValueError, match="immediate hourly successor"):
        stream._select_cycle(plan, backend, after_cycle=after)


def test_a_delayed_child_stream_plan_refuses_before_it_spends(tmp_path):
    """Every stream leg runs the PREPARED-tree runner, which restores its
    domains from caches and cannot bring one to life mid-run.  Task #205
    fixed the activation-epoch REFL_10CM stash on the `woof run` route;
    this route still has no activation callback, so the plan refuses at
    load -- before a cycle is fetched or prepared -- and says which door
    does run it."""
    with pytest.raises(ValueError) as caught:
        _make_plan(
            tmp_path, cycle_count=1, target_lead=1, delayed_child=True)
    message = str(caught.value)
    assert "woof stream route does not implement delayed nest " \
           "activation" in message
    assert "`woof run`" in message


def test_child_start_at_first_leg_endpoint_keeps_its_own_refusal(tmp_path):
    """Ordering pin: the stream's own endpoint gate is more specific than
    the route refusal below it, so a child starting AT the first leg's
    end keeps the sentence about the checkpoint it would break."""
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=1)
    source = plan.experiment_config
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "run_seconds = 3600.0", "run_seconds = 7200.0").replace(
            "parent_time_step_ratio = 3",
            "parent_time_step_ratio = 3\n"
            "start_time = 2026-07-23T01:00:00"),
        encoding="utf-8")

    with pytest.raises(ValueError, match="start before the first one-hour"):
        stream.load_stream_plan(plan.path)


@pytest.mark.parametrize("field", [
    "experiment_config", "domain_spec", "wps_namelist",
    "namelist_input", "stock_wrf_namelist_input",
])
def test_plan_authority_drift_before_pin_is_refused(tmp_path, field):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=1)
    authority = getattr(plan, field)
    authority.write_bytes(authority.read_bytes() + b"\n")
    backend = FakeBackend([datetime(2026, 7, 23, 13)])

    with pytest.raises(ValueError, match=f"authority {field} changed"):
        stream.run_stream(plan, backend=backend, progress=lambda _: None)

    assert not backend.fetch_calls
    assert not backend.commands


def test_geog_manifest_appearance_after_plan_load_is_refused(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=1)
    _write_json(plan.geog_manifest_path, {"schema": "geog-authority-v1"})

    with pytest.raises(ValueError, match="geog authority manifest changed"):
        stream.run_stream(
            plan, backend=FakeBackend([datetime(2026, 7, 23, 14)]),
            progress=lambda _: None)


def test_pinned_authority_prevents_mid_program_template_drift(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=2)
    original = plan.experiment_config

    class MutatingBackend(FakeBackend):
        changed = False

        def run_command(self, argv, *, stage):
            super().run_command(argv, stage=stage)
            if stage.startswith("f001 sealed") and not self.changed:
                self.changed = True
                original.write_text(
                    original.read_text(encoding="utf-8").replace(
                        "mp_physics = 6", "mp_physics = 8"),
                    encoding="utf-8")

    cycle = datetime(2026, 7, 23, 15)
    backend = MutatingBackend([cycle])
    result = stream.run_stream(
        plan, backend=backend, progress=lambda _: None)

    assert result["status"] == "PASS"
    cycle_root = plan.work_root / "cycles" / cycle.strftime("%Y%m%dT%H")
    assert load_experiment(
        cycle_root / "legs" / "f001" / "configs" / "experiment.toml"
    ).root.run.mp_physics == 6
    assert load_experiment(
        cycle_root / "legs" / "f002" / "configs" / "experiment.toml"
    ).root.run.mp_physics == 6
    changed_plan = stream.load_stream_plan(plan.path)
    with pytest.raises(ValueError, match="belongs to another plan"):
        stream.run_stream(
            changed_plan, backend=backend, progress=lambda _: None)


def test_child_consumes_pinned_registry_during_mutate_use_restore(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=1)
    packaged = next(
        path for label, path, _digest in plan.authority_sources
        if label == "physics_registry")
    original = tmp_path / "mutable-physics-registry.json"
    original_bytes = packaged.read_bytes()
    original.write_bytes(original_bytes)
    expected = hashlib.sha256(original_bytes).hexdigest()
    authorities = tuple(
        (label, original, expected) if label == "physics_registry"
        else (label, path, digest)
        for label, path, digest in plan.authority_sources)
    plan = dataclasses.replace(plan, authority_sources=authorities)
    stream._claim_work_root(plan)
    pinned = stream._pin_plan_authorities(plan)
    backend = stream.ProductionBackend(progress=lambda _: None)
    backend.bind_plan_authorities(pinned)
    marker = tmp_path / "child-registry.txt"
    script = (
        "from pathlib import Path; import hashlib; "
        "from woof.physics_registry import _REGISTRY_PATH; "
        f"Path({str(marker)!r}).write_text("
        "str(_REGISTRY_PATH) + '\\n' + "
        "hashlib.sha256(_REGISTRY_PATH.read_bytes()).hexdigest())"
    )

    original.write_bytes(b"{}\n")
    try:
        # The endpoint check cannot catch a mutate/use/restore race.  The
        # child instead opens the immutable snapshot path supplied in env.
        stream._verify_pinned_authorities(pinned)
        backend.run_command(
            [stream.sys.executable, "-c", script], stage="registry probe")
    finally:
        original.write_bytes(original_bytes)

    child_path, child_digest = marker.read_text().splitlines()
    assert Path(child_path).resolve() != original.resolve()
    assert Path(child_path).is_file()
    assert child_digest == expected
    assert original.read_bytes() == original_bytes

    snapshot = Path(child_path)
    snapshot_bytes = snapshot.read_bytes()
    snapshot.write_bytes(snapshot_bytes + b"\n")
    try:
        with pytest.raises(stream.subprocess.CalledProcessError):
            backend.run_command(
                [stream.sys.executable, "-c",
                 "import woof.physics_registry"],
                stage="altered registry probe")
    finally:
        snapshot.write_bytes(snapshot_bytes)


def test_copied_stale_run_is_quarantined_and_cannot_skip_advancement(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=2)
    cycle = datetime(2026, 7, 23, 16)
    backend = FakeBackend([cycle])
    stream._claim_work_root(plan)
    stream._run_cycle(plan, backend=backend, progress=lambda _: None)
    cycle_root = plan.work_root / "cycles" / cycle.strftime("%Y%m%dT%H")
    first_run = cycle_root / "legs" / "f001" / "run"
    second_leg = cycle_root / "legs" / "f002"
    second_run = second_leg / "run"
    saved = second_leg / "run-original"
    second_run.rename(saved)
    shutil.copytree(first_run, second_run)
    (second_leg / "chain-link.json").unlink()
    prior_state = json.loads((second_leg / "state.json").read_text())
    prior_completed = prior_state["stages"]["run"]["completed_at"]

    summary = stream._run_cycle(
        plan, backend=backend, progress=lambda _: None)

    assert summary["status"] == "PASS"
    assert list(second_leg.glob("run.interrupted-*"))
    new_state = json.loads((second_leg / "state.json").read_text())
    assert new_state["stages"]["run"]["completed_at"] > prior_completed
    link = json.loads((second_leg / "chain-link.json").read_text())
    assert link["checkpoint_root"] != link["restart_input"]
    assert read_restart_header(Path(link["checkpoint_root"]))[
        "elapsed_seconds"] == 7200.0


def test_disk_gate_refuses_before_any_fetch_or_child_process(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=2, target_lead=4)
    cycle = datetime(2026, 7, 23, 17)

    class FullDiskBackend(FakeBackend):
        def free_disk_bytes(self, _path):
            return 1

    backend = FullDiskBackend([cycle, cycle + timedelta(hours=1)])
    with pytest.raises(RuntimeError, match="disk-capacity refusal before fetch"):
        stream.run_stream(plan, backend=backend, progress=lambda _: None)

    assert not backend.fetch_calls
    assert not backend.commands
    receipts = list(plan.work_root.glob("disk-capacity-refusals/*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text())
    assert payload["status"] == "REFUSED"
    assert payload["basis"]["target_lead"] == 4
    assert payload["basis"]["cycle_count"] == 2
    assert payload["projected_generation_bytes"] > 0
    assert payload["fixed_margin_bytes"] == 2 * 1024 ** 3


def test_a_work_root_without_hard_links_streams_every_hour(
        tmp_path, monkeypatch):
    """A drive with no hard links (exFAT) streams every hour.

    From the second hour on each preparation reuses the previous hour's
    files.  On such a drive they are now copied and proven byte-identical
    instead of linked (woof.filesystem_paths.link_or_copy_verified), so
    the stream no longer refuses the work root before its first fetch --
    a refusal that cited only the copy's disk cost.
    """
    import errno
    import os

    def no_hard_links(source, destination, *args, **kwargs):
        if os.name == "nt":
            raise OSError(errno.EINVAL, "Incorrect function", str(source), 1,
                          str(destination))
        raise OSError(errno.EPERM, "Operation not permitted", str(source),
                      None, str(destination))

    monkeypatch.setattr(os, "link", no_hard_links)
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=2)
    cycle = datetime(2026, 7, 23, 17)
    backend = FakeBackend([cycle])

    result = stream.run_stream(plan, backend=backend, progress=lambda _: None)

    assert result["status"] == "PASS"
    assert backend.downloads == Counter(
        {(cycle, 0): 1, (cycle, 1): 1, (cycle, 2): 1})


@pytest.mark.usefixtures("classic_history_frames")
def test_node_budget_accepts_enforced_source_envelope_and_refuses_below_bound(
        tmp_path):
    # The accepted Node B deployment geometry: d01 72x70x49 hourly history,
    # d02 90x72x49 at 15-minute history cadence, four leads over two exact
    # hourly cycles.  This is the proof geometry rather than the tiny default
    # controller fixture, so this gate protects the deployment it authorizes.
    plan = _make_plan(
        tmp_path, cycle_count=2, target_lead=4, proof_geometry=True,
        cache_enabled=True)
    cycle = datetime(2026, 7, 23, 17)
    observed = [
        {"object_content_length_bytes": 2 * 1024 ** 3},
        {"object_content_length_bytes": 3 * 1024 ** 3},
    ]
    node_free = 246_116_417_536

    class FixedDisk(FakeBackend):
        def __init__(self, free):
            super().__init__([cycle, cycle + timedelta(hours=1)])
            self.free = free

        def free_disk_bytes(self, _path):
            return self.free

    receipt = stream._disk_capacity_gate(
        plan.work_root / "disk-capacity.json", plan=plan,
        backend=FixedDisk(node_free), cycle=cycle, lead=1,
        observations=observed)
    required = receipt["conservative_retained_requirement_bytes"]

    # A190 retires the 64/48-field history assumption and takes history out
    # of the blanket 2x factor, which still doubles the retained
    # preparation/checkpoint reservation. Reprice the fixed proof geometry
    # independently from the stream controller: four retained preparations
    # and their 14/44 history frames per cycle. Each tick-zero frame writes
    # the initial reflectivity array too.
    from woof import disk_budget
    from woof.io.history_selection import resolve

    preparation = sum(
        domain.run.nx * domain.run.ny * (domain.run.nz * 4 * 192 + 4 * 96)
        for domain in plan.experiment.domains)
    history = 0
    for domain, frames in zip(plan.experiment.domains, (14, 44), strict=True):
        selection = resolve(plan.experiment.output, domain.output)
        mature = disk_budget.history_frame_bytes(domain.run, selection)
        history += frames * mature
    expected_generation = 2 * (4 * 2 * preparation + history)

    assert receipt["status"] == "PASS"
    assert receipt["free_bytes_observed"] == node_free
    assert receipt["projected_source_output_bytes"] == 85_899_345_920
    assert receipt["projected_cache_copy_bytes"] == 85_899_345_920
    assert receipt["projected_generation_bytes"] == expected_generation
    assert receipt["fixed_margin_bytes"] == 2_147_483_648
    assert required == 2 * 85_899_345_920 + expected_generation + 2_147_483_648
    assert receipt["free_bytes_after_requirement"] == node_free - required
    assert receipt["volume_layout"]["kind"] == \
        "shared-work-cache-volume"
    assert receipt["work_volume"]["required_bytes"] == required
    assert receipt["cache_volume"]["required_bytes"] == required
    assert receipt["work_volume"]["free_bytes_observed"] == node_free
    assert receipt["cache_volume"]["free_bytes_observed"] == node_free
    assert receipt["projected_generation_bytes"] / 1024 ** 3 == \
        pytest.approx(expected_generation / 1024 ** 3)
    assert receipt["basis"]["history_cadence_seconds"] == {
        "d01": 3600.0,
        "d02": 900.0,
    }
    assert receipt["basis"]["full_file_hour_reservation_cap_bytes"] == \
        8 * 1024 ** 3

    exact_root = tmp_path / "exact-space-work"
    exact_plan = dataclasses.replace(plan, work_root=exact_root)
    exact = stream._disk_capacity_gate(
        exact_root / "disk-capacity.json", plan=exact_plan,
        backend=FixedDisk(required), cycle=cycle, lead=1,
        observations=observed)
    assert exact["status"] == "PASS"
    assert exact["free_bytes_after_requirement"] == 0

    low_root = tmp_path / "low-space-work"
    low_plan = dataclasses.replace(plan, work_root=low_root)
    with pytest.raises(RuntimeError, match="disk-capacity refusal"):
        stream._disk_capacity_gate(
            low_root / "disk-capacity.json", plan=low_plan,
            backend=FixedDisk(required - 1), cycle=cycle, lead=1,
            observations=observed)


def test_split_volume_capacity_gates_cache_independently(tmp_path):
    plan = _make_plan(
        tmp_path, cycle_count=1, target_lead=1, cache_enabled=True)
    cycle = datetime(2026, 7, 23, 17)
    observations = [
        {"object_content_length_bytes": 2 * 1024 ** 3},
        {"object_content_length_bytes": 3 * 1024 ** 3},
    ]
    projected_cache = 16 * 1024 ** 3

    generation = stream._estimated_generation_bytes(plan, 1)
    work_required = projected_cache + generation + 2 * 1024 ** 3

    class SplitDisk(FakeBackend):
        def __init__(self, work_free, cache_free):
            super().__init__([cycle])
            self.work_free = work_free
            self.cache_free = cache_free

        def disk_volume_identity(self, path):
            return ("cache-volume" if Path(path).resolve()
                    == plan.cache_dir.resolve() else "work-volume")

        def free_disk_bytes(self, path):
            return (self.cache_free if Path(path).resolve()
                    == plan.cache_dir.resolve() else self.work_free)

    receipt = stream._disk_capacity_gate(
        plan.work_root / "disk-capacity.json", plan=plan,
        backend=SplitDisk(work_required, projected_cache),
        cycle=cycle, lead=1,
        observations=observations)
    assert receipt["status"] == "PASS"
    assert receipt["volume_layout"]["kind"] == \
        "split-work-cache-volumes"
    assert receipt["cache_volume"]["required_bytes"] == projected_cache
    assert receipt["work_volume"]["required_bytes"] == work_required
    assert receipt["work_volume"]["remaining_bytes_after_requirement"] == 0
    assert receipt["cache_volume"][
        "remaining_bytes_after_requirement"] == 0

    low_cache_root = tmp_path / "split-cache-refusal"
    low_cache_plan = dataclasses.replace(plan, work_root=low_cache_root)
    with pytest.raises(RuntimeError, match="disk-capacity refusal"):
        stream._disk_capacity_gate(
            low_cache_root / "disk-capacity.json", plan=low_cache_plan,
            backend=SplitDisk(work_required, projected_cache - 1),
            cycle=cycle, lead=1,
            observations=observations)
    refusal = next(
        (low_cache_root / "disk-capacity-refusals").glob("*.json"))
    payload = json.loads(refusal.read_text())
    assert payload["work_volume"]["identity"] == "work-volume"
    assert payload["work_volume"]["free_bytes_observed"] == work_required
    assert payload["cache_volume"]["free_bytes_observed"] == \
        projected_cache - 1
    assert payload["cache_volume"]["identity"] == "cache-volume"
    assert payload["cache_volume"][
        "remaining_bytes_after_requirement"] == -1
    assert payload["status"] == "REFUSED"

    low_work_root = tmp_path / "split-work-refusal"
    low_work_plan = dataclasses.replace(plan, work_root=low_work_root)
    with pytest.raises(RuntimeError, match="disk-capacity refusal"):
        stream._disk_capacity_gate(
            low_work_root / "disk-capacity.json", plan=low_work_plan,
            backend=SplitDisk(work_required - 1, projected_cache),
            cycle=cycle, lead=1, observations=observations)
    refusal = next(
        (low_work_root / "disk-capacity-refusals").glob("*.json"))
    payload = json.loads(refusal.read_text())
    assert payload["work_volume"]["identity"] == "work-volume"
    assert payload["work_volume"]["free_bytes_observed"] == \
        work_required - 1
    assert payload["work_volume"][
        "remaining_bytes_after_requirement"] == -1
    assert payload["cache_volume"]["identity"] == "cache-volume"
    assert payload["cache_volume"]["free_bytes_observed"] == projected_cache
    assert payload["status"] == "REFUSED"


def test_first_fetch_headroom_prices_f000_and_f001_with_cache_copies(tmp_path):
    plan = _make_plan(
        tmp_path, cycle_count=1, target_lead=1, cache_enabled=True)
    cycle = datetime(2026, 7, 23, 17)
    two_hours = [{"object_content_length_bytes": 1024 ** 3}] * 2
    generation = stream._estimated_generation_bytes(plan, 1)
    two_required = 4 * 1024 ** 3 + generation + 512 * 1024 ** 2

    class SameDisk(FakeBackend):
        def __init__(self, available):
            super().__init__([cycle])
            self.available = available

        def free_disk_bytes(self, _path):
            return self.available

    passed = stream._disk_headroom_gate(
        tmp_path / "two-hour-headroom.json", plan=plan,
        backend=SameDisk(two_required), cycle=cycle, lead=1,
        observations=two_hours)
    passed_payload = json.loads(passed.read_text())
    assert passed_payload["source_hours_priced"] == 2
    assert passed_payload["source_prefix_state"] == \
        "unverified-full-prefix-replacement"
    assert passed_payload["work_volume"][
        "remaining_bytes_after_requirement"] == 0
    with pytest.raises(RuntimeError, match="disk-headroom refusal"):
        stream._disk_headroom_gate(
            tmp_path / "two-hour-low-headroom.json", plan=plan,
            backend=SameDisk(two_required - 1), cycle=cycle, lead=1,
            observations=two_hours)
    refusal = next((tmp_path / "disk-headroom-refusals").glob("*.json"))
    payload = json.loads(refusal.read_text())
    assert payload["source_hours_priced"] == 2
    assert payload["projected_source_output_bytes"] == 2 * 1024 ** 3
    assert payload["projected_cache_copy_bytes"] == 2 * 1024 ** 3
    assert payload["work_volume"]["required_bytes"] == two_required


def test_split_volume_headroom_gates_each_volume_at_exact_boundary(tmp_path):
    plan = _make_plan(
        tmp_path, cycle_count=1, target_lead=2, cache_enabled=True)
    cycle = datetime(2026, 7, 23, 17)
    source_bytes = 3 * 1024 ** 3

    work_required = (
        source_bytes + stream._estimated_generation_bytes(plan, 2)
        + 512 * 1024 ** 2)

    class SplitHeadroom(FakeBackend):
        def __init__(self, work_free, cache_free):
            super().__init__([cycle])
            self.work_free = work_free
            self.cache_free = cache_free

        def disk_volume_identity(self, path):
            return ("cache-volume" if Path(path).resolve()
                    == plan.cache_dir.resolve() else "work-volume")

        def free_disk_bytes(self, path):
            return (self.cache_free if Path(path).resolve()
                    == plan.cache_dir.resolve() else self.work_free)

    passed = stream._disk_headroom_gate(
        tmp_path / "split-headroom-pass.json", plan=plan,
        backend=SplitHeadroom(work_required, source_bytes), cycle=cycle,
        lead=2,
        observations=[{"object_content_length_bytes": source_bytes}],
        prior_fetch_prefix_verified=True)
    payload = json.loads(passed.read_text())
    assert payload["status"] == "PASS"
    assert payload["work_volume"]["remaining_bytes_after_requirement"] == 0
    assert payload["cache_volume"][
        "remaining_bytes_after_requirement"] == 0

    with pytest.raises(RuntimeError, match="disk-headroom refusal"):
        stream._disk_headroom_gate(
            tmp_path / "split-cache-low-headroom.json", plan=plan,
            backend=SplitHeadroom(work_required, source_bytes - 1),
            cycle=cycle, lead=2,
            observations=[{"object_content_length_bytes": source_bytes}],
            prior_fetch_prefix_verified=True)
    refusal = next((tmp_path / "disk-headroom-refusals").glob("*.json"))
    payload = json.loads(refusal.read_text())
    assert payload["work_volume"]["identity"] == "work-volume"
    assert payload["work_volume"]["free_bytes_observed"] == work_required
    assert payload["cache_volume"]["required_bytes"] == source_bytes
    assert payload["cache_volume"]["free_bytes_observed"] == source_bytes - 1
    assert payload["cache_volume"]["identity"] == "cache-volume"
    assert payload["cache_volume"][
        "remaining_bytes_after_requirement"] == -1

    work_refusal_root = tmp_path / "split-work-low"
    with pytest.raises(RuntimeError, match="disk-headroom refusal"):
        stream._disk_headroom_gate(
            work_refusal_root / "split-work-low-headroom.json", plan=plan,
            backend=SplitHeadroom(work_required - 1, source_bytes),
            cycle=cycle, lead=2,
            observations=[{"object_content_length_bytes": source_bytes}],
            prior_fetch_prefix_verified=True)
    refusal = next(
        (work_refusal_root / "disk-headroom-refusals").glob("*.json"))
    payload = json.loads(refusal.read_text())
    assert payload["work_volume"]["identity"] == "work-volume"
    assert payload["work_volume"]["free_bytes_observed"] == \
        work_required - 1
    assert payload["work_volume"][
        "remaining_bytes_after_requirement"] == -1
    assert payload["cache_volume"]["identity"] == "cache-volume"
    assert payload["cache_volume"]["free_bytes_observed"] == source_bytes


@pytest.mark.parametrize("prior_damage", ("corrupt", "partial"))
def test_f002_unverified_prior_prefix_prices_full_replacement_before_fetch(
        tmp_path, prior_damage):
    plan = _make_plan(
        tmp_path, cycle_count=1, target_lead=2, cache_enabled=True)
    cycle = datetime(2026, 7, 23, 17)
    terminal_bytes = 2 * (1024 + 2)
    full_prefix_bytes = sum(2 * (1024 + hour) for hour in range(3))
    terminal_only_requirement = (
        2 * terminal_bytes + stream._estimated_generation_bytes(plan, 2)
        + 512 * 1024 ** 2)

    class DamagedPriorPrefix(FakeBackend):
        damaged = False

        def run_command(self, argv, *, stage):
            super().run_command(argv, stage=stage)
            if stage == "f001 sealed forecast" and not self.damaged:
                source = (
                    plan.work_root / "cycles" / cycle.strftime("%Y%m%dT%H")
                    / "source" / "atmosphere-f000.bin")
                if prior_damage == "corrupt":
                    source.write_bytes(source.read_bytes() + b"corrupt")
                else:
                    source.unlink()
                self.damaged = True

        def free_disk_bytes(self, _path):
            first_link = (
                plan.work_root / "cycles" / cycle.strftime("%Y%m%dT%H")
                / "legs" / "f001" / "chain-link.json")
            if self.damaged and first_link.is_file():
                return terminal_only_requirement
            return 1024 ** 4

    backend = DamagedPriorPrefix([cycle])
    with pytest.raises(RuntimeError, match="disk-headroom refusal before fetch"):
        stream.run_stream(plan, backend=backend, progress=lambda _: None)

    assert backend.fetch_calls == [(cycle, 1)]
    leg = (plan.work_root / "cycles" / cycle.strftime("%Y%m%dT%H")
           / "legs" / "f002")
    refusal = next((leg / "disk-headroom-refusals").glob("*.json"))
    payload = json.loads(refusal.read_text())
    assert payload["fetch_prefix_verified_before_gate"] is False
    assert payload["prior_fetch_prefix_verified_before_gate"] is False
    assert payload["source_prefix_state"] == \
        "unverified-full-prefix-replacement"
    assert payload["source_hour_pricing_policy"] == "observed-full-prefix"
    assert payload["source_hours_observed"] == 3
    assert payload["source_hours_priced"] == 3
    assert payload["projected_source_output_bytes"] == full_prefix_bytes
    assert payload["projected_cache_copy_bytes"] == full_prefix_bytes
    assert payload["work_volume"]["free_bytes_observed"] == \
        terminal_only_requirement
    assert payload["work_volume"][
        "remaining_bytes_after_requirement"] == \
        -2 * (full_prefix_bytes - terminal_bytes)


def test_observed_source_hour_above_enforced_envelope_refuses_before_fetch(
        tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=1)
    cycle = datetime(2026, 7, 23, 17)

    class OversizeHourBackend(FakeBackend):
        def wait_for_hour(self, *args, **kwargs):
            observed = super().wait_for_hour(*args, **kwargs)
            observed["object_content_length_bytes"] = \
                stream._SOURCE_HOUR_RESERVATION_BYTES + 1
            return observed

    backend = OversizeHourBackend([cycle])
    with pytest.raises(RuntimeError, match="above the enforced.*envelope"):
        stream.run_stream(plan, backend=backend, progress=lambda _: None)

    assert not backend.fetch_calls
    assert not backend.commands


def test_nonempty_unowned_work_root_is_refused_without_moving_it(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=1)
    plan.work_root.mkdir()
    sentinel = plan.work_root / "belongs-to-user.txt"
    sentinel.write_text("leave me", encoding="utf-8")
    backend = FakeBackend([datetime(2026, 7, 23, 17)])

    with pytest.raises(ValueError, match="unowned nonempty"):
        stream.run_stream(plan, backend=backend, progress=lambda _: None)

    assert sentinel.read_text(encoding="utf-8") == "leave me"
    assert not backend.fetch_calls
    assert not backend.commands


def test_unowned_stage_is_refused_not_quarantined(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=1)
    cycle = datetime(2026, 7, 23, 17)
    stream._claim_work_root(plan)
    foreign = (plan.work_root / "cycles" / cycle.strftime("%Y%m%dT%H") /
               "legs" / "f001" / "root-preparation")
    foreign.mkdir(parents=True)
    sentinel = foreign / "foreign.txt"
    sentinel.write_text("not stream-owned", encoding="utf-8")
    backend = FakeBackend([cycle])

    with pytest.raises(ValueError, match="unowned stream stage"):
        stream._run_cycle(plan, backend=backend, progress=lambda _: None)

    assert sentinel.read_text(encoding="utf-8") == "not stream-owned"
    assert not list(foreign.parent.glob("root-preparation.interrupted-*"))


def test_production_backend_uses_supervisor_uuid_lock_and_cuda_mask(
        tmp_path, monkeypatch):
    import woof.supervisor as supervisor

    plan = dataclasses.replace(
        _make_plan(tmp_path, cycle_count=1, target_lead=1),
        gpu_uuid="GPU-fixture", allow_shared_gpu=False)
    gpu = supervisor.GPUIdentity(
        "GPU-fixture", "999.1", "fixture device", 1)
    events = []

    class Lock:
        def __init__(self, uuid, *, path, run_id):
            events.append(("lock", uuid, Path(path), run_id))

        def __enter__(self):
            events.append(("enter",))
            return self

        def __exit__(self, *exc):
            events.append(("exit",))

    monkeypatch.setattr(supervisor, "select_gpu", lambda requested: gpu)
    monkeypatch.setattr(supervisor, "default_lock_path",
                        lambda _uuid: tmp_path / "gpu.lock")
    monkeypatch.setattr(supervisor, "GPUFileLock", Lock)
    monkeypatch.setattr(
        supervisor, "preflight_exclusive_gpu",
        lambda uuid, **kwargs: events.append(("preflight", uuid, kwargs)))
    launched = {}
    monkeypatch.setattr(
        stream.subprocess, "run",
        lambda argv, **kwargs: launched.update(
            {"argv": argv, "env": kwargs["env"]}))
    backend = stream.ProductionBackend(progress=lambda _: None)

    with backend.gpu_allocation(plan) as receipt:
        backend.run_command(["fixture-worker"], stage="fixture")
        assert launched["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-fixture"
        assert launched["env"]["WOOF_GPU_UUID"] == "GPU-fixture"

    assert receipt["resolved_uuid"] == "GPU-fixture"
    assert receipt["lock_path"] == str((tmp_path / "gpu.lock").resolve())
    assert [event[0] for event in events].count("preflight") == 2
    assert events[-1] == ("exit",)


def test_program_capacity_is_reserved_once_as_free_space_decreases(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=3)
    cycle = datetime(2026, 7, 23, 17)

    class DecreasingDiskBackend(FakeBackend):
        def __init__(self, cycles):
            super().__init__(cycles)
            self.disk_calls = 0

        def free_disk_bytes(self, _path):
            self.disk_calls += 1
            return 8 * 1024 ** 4 - self.disk_calls * 128 * 1024 ** 3

    backend = DecreasingDiskBackend([cycle])
    result = stream.run_stream(
        plan, backend=backend, progress=lambda _: None)

    assert result["status"] == "PASS"
    reservation = json.loads(
        (plan.work_root / "disk-capacity.json").read_text())
    assert reservation["schema"] == "gpuwm-stream-disk-capacity-v4"
    assert reservation["basis"]["future_hour_size_policy"] == \
        "enforced-fixed-upper-envelope-v1"
    assert reservation["projected_source_bytes"] == \
        stream._SOURCE_HOUR_RESERVATION_BYTES * 4
    assert backend.disk_calls == 4  # one program reservation + three legs


def test_mid_program_external_disk_consumption_refuses_next_fetch(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=2)
    cycle = datetime(2026, 7, 23, 17)

    class ConsumedDiskBackend(FakeBackend):
        def __init__(self, cycles):
            super().__init__(cycles)
            self.disk_calls = 0

        def free_disk_bytes(self, _path):
            self.disk_calls += 1
            return 8 * 1024 ** 4 if self.disk_calls <= 2 else 1

    backend = ConsumedDiskBackend([cycle])
    with pytest.raises(RuntimeError, match="disk-headroom refusal before fetch"):
        stream.run_stream(plan, backend=backend, progress=lambda _: None)

    assert backend.fetch_calls == [(cycle, 1)]
    refusals = list(plan.work_root.glob(
        "cycles/*/legs/f002/disk-headroom-refusals/*.json"))
    assert len(refusals) == 1
    assert json.loads(refusals[0].read_text())["status"] == "REFUSED"


def test_resume_rechecks_stale_passed_headroom_before_fetch(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=1)
    cycle = datetime(2026, 7, 23, 17)

    class CrashWindowBackend(FakeBackend):
        def __init__(self, cycles):
            super().__init__(cycles)
            self.low_space = False

        def free_disk_bytes(self, _path):
            return 1 if self.low_space else 8 * 1024 ** 4

        def fetch_prefix(self, plan, cycle, lead, source_root):
            if not self.low_space:
                raise RuntimeError("crash after headroom PASS")
            return super().fetch_prefix(plan, cycle, lead, source_root)

    backend = CrashWindowBackend([cycle])
    with pytest.raises(RuntimeError, match="crash after headroom PASS"):
        stream.run_stream(plan, backend=backend, progress=lambda _: None)
    headroom = (plan.work_root / "cycles" / cycle.strftime("%Y%m%dT%H") /
                "legs" / "f001" / "disk-headroom.json")
    assert json.loads(headroom.read_text())["status"] == "PASS"

    backend.low_space = True
    with pytest.raises(RuntimeError, match="disk-headroom refusal before fetch"):
        stream.run_stream(plan, backend=backend, progress=lambda _: None)

    # The stale PASS is preserved as evidence, but it did not authorize the
    # resumed fetch after free space changed.
    assert json.loads(headroom.read_text())["status"] == "PASS"
    assert backend.fetch_calls == []
    assert len(list(headroom.parent.glob(
        "disk-headroom-refusals/*.json"))) == 1


def test_post_fetch_crash_resume_prices_only_generation_at_exact_boundary(
        tmp_path):
    plan = _make_plan(
        tmp_path, cycle_count=1, target_lead=1, cache_enabled=True)
    cycle = datetime(2026, 7, 23, 17)
    resume_requirement = (
        stream._estimated_generation_bytes(plan, 1) + 512 * 1024 ** 2)

    class PostFetchCrashBackend(FakeBackend):
        def __init__(self, cycles):
            super().__init__(cycles)
            self.resume_at_exact_boundary = False
            self.crashed = False

        def free_disk_bytes(self, _path):
            if self.resume_at_exact_boundary:
                return resume_requirement
            return 1024 ** 4

        def run_command(self, argv, *, stage):
            if stage == "f001 root preparation" and not self.crashed:
                self.crashed = True
                raise RuntimeError("crash after verified fetch")
            return super().run_command(argv, stage=stage)

    backend = PostFetchCrashBackend([cycle])
    with pytest.raises(RuntimeError, match="crash after verified fetch"):
        stream.run_stream(plan, backend=backend, progress=lambda _: None)

    source = (plan.work_root / "cycles" / cycle.strftime("%Y%m%dT%H") /
              "source")
    assert stream._fetch_prefix_is_verified(source, cycle=cycle, lead=1)
    backend.resume_at_exact_boundary = True
    result = stream.run_stream(plan, backend=backend, progress=lambda _: None)

    assert result["status"] == "PASS"
    leg = (plan.work_root / "cycles" / cycle.strftime("%Y%m%dT%H") /
           "legs" / "f001")
    attempts = list((leg / "disk-headroom-attempts").glob("pass-*.json"))
    assert len(attempts) == 1
    receipt = json.loads(attempts[0].read_text())
    assert receipt["resume_recheck"] is True
    assert receipt["fetch_prefix_verified_before_gate"] is True
    assert receipt["source_hours_observed"] == 2
    assert receipt["source_hours_priced"] == 0
    assert receipt["source_hours_requiring_write"] == 0
    assert receipt["projected_source_output_bytes"] == 0
    assert receipt["projected_cache_copy_bytes"] == 0
    assert receipt["work_volume"]["required_bytes"] == resume_requirement
    assert receipt["work_volume"][
        "remaining_bytes_after_requirement"] == 0
    assert backend.fetch_calls == [(cycle, 1), (cycle, 1)]
    assert set(backend.downloads.values()) == {1}


def test_incomplete_manifest_inventory_cannot_authorize_zero_write_resume(
        tmp_path):
    plan = _make_plan(
        tmp_path, cycle_count=1, target_lead=1, cache_enabled=True)
    cycle = datetime(2026, 7, 23, 17)
    backend = FakeBackend([cycle])
    source = tmp_path / "source"
    manifest_path = backend.fetch_prefix(plan, cycle, 1, source)
    assert stream._fetch_prefix_is_verified(source, cycle=cycle, lead=1)

    missing = source / "atmosphere-f001.bin"
    missing.unlink()
    sums = source / "SHA256SUMS"
    sums.write_text("\n".join(
        row for row in sums.read_text().splitlines()
        if "atmosphere-f001.bin" not in row) + "\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = [
        row for row in manifest["files"]
        if not (row.get("forecast_hour") == 1
                and row.get("role") == "atmosphere")]
    checksum = next(
        row for row in manifest["files"] if row.get("role") == "checksums")
    checksum["bytes"] = sums.stat().st_size
    checksum["sha256"] = hashlib.sha256(sums.read_bytes()).hexdigest()
    _write_json(manifest_path, manifest)

    verified = stream._fetch_prefix_is_verified(
        source, cycle=cycle, lead=1)
    assert verified is False
    generation_only = (
        stream._estimated_generation_bytes(plan, 1) + 512 * 1024 ** 2)

    class ExactGenerationOnly(FakeBackend):
        def free_disk_bytes(self, _path):
            return generation_only

    with pytest.raises(RuntimeError, match="disk-headroom refusal"):
        stream._disk_headroom_gate(
            tmp_path / "incomplete-prefix-headroom.json", plan=plan,
            backend=ExactGenerationOnly([cycle]), cycle=cycle, lead=1,
            observations=[
                backend.wait_for_hour(
                    cycle, hour, timeout_s=1.0, poll_s=1.0)
                for hour in (0, 1)],
            fetch_prefix_verified=verified)


def test_watcher_records_future_hour_during_long_prior_leg(tmp_path):
    plan = _make_plan(tmp_path, cycle_count=1, target_lead=4)
    cycle = datetime(2026, 7, 23, 18)

    class ConcurrentAvailabilityBackend(FakeBackend):
        def __init__(self, cycles):
            super().__init__(cycles)
            self.release_four = threading.Event()
            self.four_observed = threading.Event()

        def wait_for_hour(self, cycle, lead, *, timeout_s, poll_s,
                          stop_event=None):
            if lead == 4:
                assert self.release_four.wait(timeout=5.0)
                if stop_event is not None and stop_event.is_set():
                    raise RuntimeError("watcher stopped")
            result = super().wait_for_hour(
                cycle, lead, timeout_s=timeout_s, poll_s=poll_s)
            if lead == 4:
                self.four_observed.set()
            return result

        def run_command(self, argv, *, stage):
            if stage.startswith("f003 sealed"):
                self.release_four.set()
                assert self.four_observed.wait(timeout=5.0)
                self._now += timedelta(minutes=15)
            super().run_command(argv, stage=stage)

    backend = ConcurrentAvailabilityBackend([cycle])
    result = stream.run_stream(
        plan, backend=backend, progress=lambda _: None)
    timeline = json.loads(Path(
        result["cycles"][0]["chain_summary"]["path"]).read_text())[
            "timeline"]
    third, fourth = timeline[2], timeline[3]
    assert fourth["forcing_set_first_observed_at"] < \
        third["leg_completed_at"]
    assert fourth[
        "previous_leg_completed_before_this_lead_first_observed"] is False
    assert fourth[
        "previous_leg_completed_before_remote_ready_last_modified"] is False


class _NoLock:
    """The supervisor's UUID lock, without the file, for door tests."""

    def __init__(self, uuid, *, path, run_id):
        self.uuid = uuid

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def stream_warnings():
    """Every woof.explain warning raised inside one test, as records."""

    from woof import explain

    seen = []
    explain.add_warning_observer(seen.append)
    try:
        yield seen
    finally:
        explain.remove_warning_observer(seen.append)


def _shared_card(monkeypatch, tmp_path, *, cotenant_mib=30000):
    """One measured card carrying one CUDA co-tenant.

    Returns the mutable device reading, so a test can make the co-tenant
    grow after the stream has started.
    """

    import woof.core.preflight as preflight
    import woof.supervisor as supervisor

    gpu = supervisor.GPUIdentity("GPU-fixture", "999.1", "fixture device", 1)
    processes = (supervisor.GPUProcess(
        gpu.uuid, 41001, "python.exe", cotenant_mib, "C"),)
    device = {"total": 0, "used": 0}
    monkeypatch.setattr(supervisor, "select_gpu", lambda requested: gpu)
    monkeypatch.setattr(supervisor, "default_lock_path",
                        lambda _uuid: tmp_path / "gpu.lock")
    monkeypatch.setattr(supervisor, "GPUFileLock", _NoLock)
    monkeypatch.setattr(supervisor, "query_compute_processes",
                        lambda uuid: processes)
    monkeypatch.setattr(supervisor, "_ADMISSION_WARNED", set())
    monkeypatch.setattr(preflight, "device_physical_total_bytes",
                        lambda **kwargs: device["total"])
    monkeypatch.setattr(preflight, "device_wide_used_bytes",
                        lambda **kwargs: device["used"])
    monkeypatch.setattr(stream.subprocess, "run",
                        lambda argv, **kwargs: None)
    return device


def test_stream_prices_its_card_from_the_function_gpuwm_run_prices_from(
        tmp_path, monkeypatch):
    """One configuration, one card, one answer at both run doors.

    The stream door passed no reservation at all, so the shared-card
    admission returned early as 'admitted-unpriced-run' and could never
    refuse: `woof run` refused a configuration beside a co-tenant that
    `woof stream` admitted.  Both doors price from
    woof.supervisor.priced_reservation_bytes now, and the per-command
    call decides nothing.
    """

    import woof.supervisor as supervisor

    plan = dataclasses.replace(
        _make_plan(tmp_path, cycle_count=1, target_lead=1),
        gpu_uuid="GPU-fixture")
    priced = supervisor.priced_reservation_bytes(plan.experiment,
                                                 source="hrrr")
    assert priced is not None and priced > 0
    # Priced from the plan's loaded experiment and from its configuration
    # file: the same number, so the stream cannot price what it does not run.
    assert supervisor.priced_reservation_bytes(
        plan.experiment_config, source="hrrr") == priced
    # With the source it prepares from: the analysed hydrometeors HRRR
    # publishes ride the root's boundary tables, and a reservation priced
    # without them is short by exactly those tables.
    assert priced > supervisor.priced_reservation_bytes(plan.experiment)

    calls = []
    monkeypatch.setattr(
        supervisor, "select_gpu",
        lambda requested: supervisor.GPUIdentity(
            "GPU-fixture", "999.1", "fixture device", 1))
    monkeypatch.setattr(supervisor, "default_lock_path",
                        lambda _uuid: tmp_path / "gpu.lock")
    monkeypatch.setattr(supervisor, "GPUFileLock", _NoLock)
    monkeypatch.setattr(supervisor, "preflight_exclusive_gpu",
                        lambda uuid, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(stream.subprocess, "run",
                        lambda argv, **kwargs: None)

    backend = stream.ProductionBackend(progress=lambda _: None)
    with backend.gpu_allocation(plan):
        backend.run_command(["fixture-worker"], stage="one")
        backend.run_command(["fixture-worker"], stage="two")

    assert [call["reservation_bytes"] for call in calls] == [priced] * 3
    # The admission decides once, before the first stage command.
    assert calls[0].get("decide", True) is True
    assert [call["decide"] for call in calls[1:]] == [False, False]


def test_a_stream_that_does_not_fit_beside_a_cotenant_is_refused_at_the_door(
        tmp_path, monkeypatch):
    import woof.supervisor as supervisor

    plan = dataclasses.replace(
        _make_plan(tmp_path, cycle_count=1, target_lead=1),
        gpu_uuid="GPU-fixture")
    priced = supervisor.priced_reservation_bytes(plan.experiment,
                                                 source="hrrr")
    device = _shared_card(monkeypatch, tmp_path)
    device["total"] = priced + 4 * 1024 ** 3
    device["used"] = 5 * 1024 ** 3  # free = priced - 1 GiB

    backend = stream.ProductionBackend(progress=lambda _: None)
    with pytest.raises(supervisor.GPUPreflightError) as caught:
        with backend.gpu_allocation(plan):
            backend.run_command(["fixture-worker"], stage="one")
    message = str(caught.value)
    assert "priced reservation" in message
    assert "pid=41001" in message
    assert "free" in message


def test_a_started_stream_says_the_shared_card_once_and_is_never_re_refused(
        tmp_path, monkeypatch, stream_warnings):
    import woof.supervisor as supervisor

    plan = dataclasses.replace(
        _make_plan(tmp_path, cycle_count=1, target_lead=1),
        gpu_uuid="GPU-fixture")
    priced = supervisor.priced_reservation_bytes(plan.experiment,
                                                 source="hrrr")
    device = _shared_card(monkeypatch, tmp_path)
    device["total"] = priced + 8 * 1024 ** 3
    device["used"] = 2 * 1024 ** 3  # free = priced + 6 GiB

    backend = stream.ProductionBackend(progress=lambda _: None)
    with backend.gpu_allocation(plan):
        backend.run_command(["fixture-worker"], stage="one")
        backend.run_command(["fixture-worker"], stage="two")
        # One shared card, one sentence, however many stage commands run.
        assert len(stream_warnings) == 1
        assert "priced reservation" in stream_warnings[0]["action"]
        # The co-tenant grows past what this run reserved AFTER the stream
        # has started.  That is news, so it is named again -- and it is
        # not a refusal, because the run has already produced output.
        device["used"] = device["total"] - 1024 ** 3
        backend.run_command(["fixture-worker"], stage="three")

    assert len(stream_warnings) == 2
    assert "already started" in stream_warnings[-1]["action"]
