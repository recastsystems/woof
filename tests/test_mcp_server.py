"""The MCP server, driven over actual stdio by the SDK's own client.

DEMO PROOF, not a mock walk: every test here speaks the real protocol
to a real ``python -m woof.mcp`` subprocess, which itself shells out
to the real CLI doors -- verify against the artifact, both hops.  The
suite proves initialize/list-tools, the synchronous doors (doctor,
sources, estimate on a SHIPPED config), the refusal-verbatim contract,
the full async job lifecycle on a fast real render job that needs no
card, and the GPU-lock refusal with the lock genuinely held.

CPU-only: ``GPUWM_NO_LOCAL_GPU=1`` rides the server environment, so no
test opens a device; anything card-shaped stays out of this file.

Execution needs the ``mcp`` SDK (the ``[mcp]`` extra). It is loaded by
the execution fixture so collection and the battery census retain the
whole test inventory when the optional SDK is absent.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from contextlib import AsyncExitStack
from pathlib import Path

from woof import proc_identity

import numpy as np
import pytest

@pytest.fixture(scope="module", autouse=True)
def mcp_sdk():
    pytest.skip("Recast WOOF carries no MCP server: woof.mcp keeps only the GPU lock")
    pytest.importorskip("mcp", reason="the server needs the [mcp] extra "
                        "(pip install recast-woof[mcp])")
    global ClientSession, StdioServerParameters, stdio_client
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The shipped config the estimate leg prices -- a wheel carries it
#: (configs/ ships), and `woof check` accepts it with a declared
#: budget on a box with no visible card.
SHIPPED_CONFIG = "configs/gfs_12km_quickstart.toml"

#: Every engine-tier tool the server promises; list_tools must serve
#: each one.  Transcribed here deliberately: a tool that silently
#: leaves the listing is a front door that fell off.
ENGINE_TOOLS = {
    "arwen_doctor", "arwen_sources", "arwen_capabilities",
    "arwen_plan_domain", "arwen_estimate", "arwen_fetch", "arwen_prep",
    "arwen_forecast", "arwen_render", "arwen_list_products",
    "arwen_run_summary", "arwen_cells", "arwen_cells_catalog",
    "job_status", "job_events", "job_result", "job_cancel", "job_list",
}


class McpClient:
    """One live stdio session to the real server, callable from sync tests.

    The SDK client is async; this hosts it on a dedicated event-loop
    thread so the whole module shares ONE server process instead of
    paying the interpreter start-up per test.
    """

    def __init__(self, jobs_dir: Path):
        self.jobs_dir = jobs_dir
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True)
        self._thread.start()
        # ONE task owns the whole session lifetime: anyio cancel scopes
        # must exit in the task that entered them, so open and close
        # both happen inside _serve and close() only trips its event.
        import concurrent.futures
        self._ready: concurrent.futures.Future = concurrent.futures.Future()
        self._serving = asyncio.run_coroutine_threadsafe(
            self._serve(), self._loop)
        self.init_result = self._ready.result(timeout=120)

    async def _serve(self):
        env = {str(k): str(v) for k, v in os.environ.items()}
        env["GPUWM_NO_LOCAL_GPU"] = "1"
        env["WOOF_MCP_JOBS_DIR"] = str(self.jobs_dir)
        # A dry run checks the WPS_GEOG tree as the launch does (2.8.8), so
        # the server sees a staged stand-in tree, not the host's default.
        from _staged_geog import staged_case_data_env
        env = staged_case_data_env(self.jobs_dir / "case-data", base=env)
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "woof.mcp"],
            env=env, cwd=str(REPO_ROOT))
        self._close_event = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(
                    stdio_client(params))
                self.session = await stack.enter_async_context(
                    ClientSession(read, write))
                self._ready.set_result(await self.session.initialize())
                await self._close_event.wait()
        except BaseException as error:  # pragma: no cover - harness
            if not self._ready.done():
                self._ready.set_exception(error)
            raise

    def _run(self, coroutine, timeout: float):
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result(timeout=timeout)

    def list_tools(self):
        return self._run(self.session.list_tools(), timeout=60)

    def call(self, name: str, arguments: dict | None = None, *,
             timeout: float = 300.0):
        return self._run(
            self.session.call_tool(name, arguments or {}), timeout=timeout)

    def close(self):
        try:
            self._loop.call_soon_threadsafe(self._close_event.set)
            self._serving.result(timeout=30)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10)


def payload(result) -> dict:
    """A tool result's structured document, however the SDK framed it."""

    assert not result.isError, _error_text(result)
    if result.structuredContent is not None:
        document = result.structuredContent
        if set(document) == {"result"}:
            return document["result"]
        return document
    return json.loads(result.content[0].text)


def _error_text(result) -> str:
    return " ".join(block.text for block in result.content
                    if getattr(block, "text", None))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    jobs_dir = tmp_path_factory.mktemp("mcp-jobs")
    live = McpClient(jobs_dir)
    yield live
    live.close()


# -------------------------------------------------------------------------
# initialize / list-tools
# -------------------------------------------------------------------------
def test_initialize_names_the_server(client):
    assert client.init_result.serverInfo.name == "arwen"


def test_list_tools_serves_every_engine_tool(client):
    names = {tool.name for tool in client.list_tools().tools}
    missing = ENGINE_TOOLS - names
    assert not missing, (
        f"engine tools missing from the listing: {sorted(missing)} -- a "
        "tool that is not listed is a front door that fell off.")


def test_every_tool_declares_a_schema_and_a_description(client):
    for tool in client.list_tools().tools:
        assert tool.inputSchema.get("type") == "object", tool.name
        assert (tool.description or "").strip(), (
            f"{tool.name} has no description; a driving agent chooses "
            "tools by these sentences.")


# -------------------------------------------------------------------------
# synchronous doors
# -------------------------------------------------------------------------
def test_arwen_doctor_runs_the_real_probes(client):
    document = payload(client.call("arwen_doctor", timeout=600))
    assert document["exit_code"] in (0, 1)
    checks = document["checks"]
    assert isinstance(checks, list) and len(checks) >= 5
    for check in checks:
        assert "name" in check and "status" in check
    assert document["healthy"] == (document["exit_code"] == 0)


def test_arwen_sources_serves_the_registry(client):
    document = payload(client.call("arwen_sources"))
    registry = document["registry"]
    assert registry["runnable_source_count"] >= 10
    assert "routes" in registry


def test_arwen_estimate_prices_the_shipped_config(client):
    document = payload(client.call("arwen_estimate", {
        "config": SHIPPED_CONFIG,
        "budget_gib": 10.75, "vram_gib": 12.0}))
    assert document["exit_code"] == 0 and document["fits"] is True
    assert "d01" in document["report"]["domains"]


def test_forecast_dry_run_prints_the_chain(client, tmp_path):
    document = payload(client.call("arwen_forecast", {
        "config": SHIPPED_CONFIG, "mode": "go", "dry_run": True,
        "outdir": str(tmp_path / "dryrun")}))
    text = "\n".join(document["commands"])
    for stage in ("authority", "fetch", "prepare", "forecast", "render"):
        assert stage in text, f"dry run names no {stage} stage:\n{text}"


# -------------------------------------------------------------------------
# refusals travel verbatim
# -------------------------------------------------------------------------
def test_a_cli_refusal_sentence_reaches_the_agent_verbatim(client):
    result = client.call("arwen_estimate",
                         {"config": "no-such-config.toml"})
    assert result.isError
    assert ("does not exist; pass the experiment .toml that "
            "`woof domain` wrote." in _error_text(result)), (
        "the door's one-sentence refusal was not carried verbatim: "
        + _error_text(result))


def test_an_unknown_job_is_refused_by_name(client):
    result = client.call("job_status", {"job_id": "job-never-launched"})
    assert result.isError
    assert "no job named job-never-launched" in _error_text(result)


# -------------------------------------------------------------------------
# the job lifecycle, on a fast real render job (no card)
# -------------------------------------------------------------------------
_NZ, _NY, _NX = 4, 12, 16


def _wrfout_fixture(root: Path) -> Path:
    """One-frame wrfout via the project's own writer (as test_render.py)."""

    from woof.io.wrfout import WrfoutWriter

    rng = np.random.default_rng(11)
    lat = np.tile(np.linspace(38.0, 40.0, _NY)[:, None], (1, _NX))
    lon = np.tile(np.linspace(-98.0, -95.0, _NX)[None, :], (_NY, 1))
    frame = {
        "T": np.zeros((_NZ, _NY, _NX), np.float32),
        "MU": np.zeros((_NY, _NX), np.float32),
        "REFL_10CM": rng.uniform(-20.0, 65.0,
                                 (_NZ, _NY, _NX)).astype(np.float32),
        "T2": rng.uniform(280.0, 300.0, (_NY, _NX)).astype(np.float32),
        "U10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "V10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "RAINC": rng.uniform(0.0, 5.0, (_NY, _NX)).astype(np.float32),
        "RAINNC": rng.uniform(0.0, 30.0, (_NY, _NX)).astype(np.float32),
        "OLR": rng.uniform(90.0, 320.0, (_NY, _NX)).astype(np.float32),
        "XLAT": lat.astype(np.float32),
        "XLONG": lon.astype(np.float32),
        "HGT": np.zeros((_NY, _NX), np.float32),
        "SINALPHA": np.zeros((_NY, _NX), np.float32),
        "COSALPHA": np.ones((_NY, _NX), np.float32),
    }
    path = root / "wrfout_d01_1974-04-03_18-00-00.nc"
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ,
                      dx=1000.0, dy=1000.0) as writer:
        writer.write_frame("1974-04-03_18:00:00", frame)
    return path


def _wait_until_done(client, job_id: str, *, budget_s: float = 240.0):
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        status = payload(client.call("job_status", {"job_id": job_id}))
        if status["state"] != "running":
            return status
        time.sleep(0.5)
    pytest.fail(f"job {job_id} still running after {budget_s}s")


def test_the_full_job_lifecycle_on_a_real_render(client, tmp_path):
    """launch -> status -> events (cursored) -> result -> artifact."""

    pytest.importorskip(
        "wrf", reason="the render door needs the wrf package (wrf-rust)")
    wrfout = _wrfout_fixture(tmp_path)
    out_dir = tmp_path / "png"

    launch = payload(client.call("arwen_render", {
        "wrfout": [str(wrfout)], "out_dir": str(out_dir),
        "products": "t2", "engine": "matplotlib", "timeidx": "0"}))
    job_id = launch["job_id"]
    assert launch["gpu"] is False

    listed = payload(client.call("job_list"))
    assert any(job["job_id"] == job_id for job in listed["jobs"])

    status = _wait_until_done(client, job_id)
    assert status["state"] == "exited", status
    assert status["exit_code"] == 0, status

    # The cursor contract: a second read from next_cursor returns only
    # what is new, and a drained stream on an exited job reports eof.
    events = payload(client.call("job_events",
                                 {"job_id": job_id, "stream": "stdout"}))
    assert events["present"] is True
    again = payload(client.call("job_events", {
        "job_id": job_id, "stream": "stdout",
        "cursor": events["next_cursor"]}))
    assert again["lines"] == [] and again["eof"] is True

    result = payload(client.call("job_result", {"job_id": job_id}))
    assert result["ok"] is True and result["exit_code"] == 0

    pngs = list(out_dir.rglob("*.png"))
    assert pngs, ("the render job exited 0 and left no PNG under "
                  f"{out_dir}; a job result is not a receipt for work "
                  "that did not happen")

    # Cancel after the fact is refused by state, and deletes nothing.
    cancel = client.call("job_cancel", {"job_id": job_id})
    assert cancel.isError
    assert "not running" in _error_text(cancel)
    assert pngs[0].is_file()


def test_arwen_list_products_reads_the_fixture(client, tmp_path):
    pytest.importorskip(
        "wrf", reason="the render door needs the wrf package (wrf-rust)")
    wrfout = _wrfout_fixture(tmp_path)
    document = payload(client.call("arwen_list_products", {
        "wrfout": str(wrfout), "engine": "matplotlib"}, timeout=300))
    assert document["exit_code"] in (0, 1)
    assert any("t2" in line for line in document["listing"])


# -------------------------------------------------------------------------
# GPU arbitration: one job per card, refusal names the holder
# -------------------------------------------------------------------------
def test_a_second_gpu_launch_is_refused_naming_the_running_job(client):
    """The lock genuinely held (by this live process), launch refused."""

    lock_path = client.jobs_dir / "gpu.lock"
    lock_path.write_text(json.dumps({
        "job_id": "job-test-holder",
        # This pytest process's own pid: alive for the whole test, so
        # the liveness check cannot reclaim the lock underneath it.
        "wrapper_pid": os.getpid(),
        # ...and its identity: a bare pid names no process (it could be reused).
        "wrapper_process": proc_identity.identify(os.getpid()),
        "created_utc": "2026-08-31T00:00:00+00:00",
    }), encoding="utf-8")
    try:
        result = client.call("arwen_forecast", {
            "config": SHIPPED_CONFIG, "mode": "run"})
        assert result.isError, (
            "a GPU launch went through while the card lock was held; "
            "two CUDA contexts on one card is the exact breakage the "
            "lock exists to prevent")
        text = _error_text(result)
        assert "job-test-holder" in text, text
        assert "refused" in text, text
    finally:
        lock_path.unlink(missing_ok=True)


# -------------------------------------------------------------------------
# the registered surface is the engine tier, exactly
# -------------------------------------------------------------------------
def test_the_engine_tier_is_the_whole_registered_surface():
    """build_server registers the engine tier and nothing else.

    The server wraps the CLI doors; a tool appearing outside the
    ``engine`` tier would be a registration surface no door owns and no
    listing promises.  Asserted in process (no stdio hop) so the tier
    layout itself is pinned, not just the flattened tool listing.
    """

    from woof.mcp.jobs import JobManager
    from woof.mcp.server import build_server

    _server, tiers = build_server(JobManager(Path("nowhere-jobs")))
    assert set(tiers) == {"engine", "suite"}
    assert set(tiers["engine"]) == ENGINE_TOOLS
