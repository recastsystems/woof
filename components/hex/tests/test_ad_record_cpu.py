# SPDX-License-Identifier: Apache-2.0
"""woof.hex.ad.record on the CPU: the per-run retention list, the shipped files, and the command surface.

The retention test is the per-date defect: a start date the shipped list does not cover leaves
'upload_needs_live_rule' words at seal, and the run's own list must take exactly the uploads the seal named that are
equal in both recordings (never a pointer table that differs between them)."""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from woof.hex.ad.record import env


def _trace(directory: Path, rows: list[dict]) -> Path:
    directory.mkdir(parents=True)
    with gzip.open(directory / "driver-1.jsonl.gz", "wt") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    return directory


def _upload(seq, src, n, sha, step=1):
    return [{"type": "MemcpyHtoD_v2", "sequence": seq, "b": src, "count": n},
            {"type": "HostUploadBytes", "sequence": seq + 1, "source": src, "bytes": n, "sha256": sha, "step": step}]


def test_retention_takes_only_named_uploads_equal_in_both(tmp_path):
    const, table_a, table_b = "c" * 64, "a" * 64, "b" * 64
    a = _trace(tmp_path / "rec-a-recorder", _upload(10, 0x1000, 4096, const) + _upload(20, 0x2000, 64, table_a))
    b = _trace(tmp_path / "rec-b-recorder", _upload(11, 0x1100, 4096, const) + _upload(21, 0x2100, 64, table_b))
    base = tmp_path / "base.txt"
    base.write_text("# base\n" + "d" * 64 + " 8\n")
    residual = tmp_path / "stage2-residual.json"
    residual.write_text(json.dumps({"residual": {"upload_needs_live_rule": [["x", "y", 10]]}}))
    out = tmp_path / "retention-1.txt"
    done = subprocess.run([sys.executable, str(env.TOOLS / "gen_immutable.py"), str(a), str(b), str(base), str(out),
                           "test", str(residual)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    entries = [l.split() for l in out.read_text().splitlines() if l and not l.startswith("#")]
    assert entries == [["d" * 64, "8"], [const, "4096"]]
    assert all(len(l) < 255 for l in out.read_text().splitlines())  # the tracer reads lines with fgets(line, 256)


def test_retention_refuses_a_named_upload_that_differs_between_recordings(tmp_path):
    a = _trace(tmp_path / "rec-a-recorder", _upload(20, 0x2000, 64, "a" * 64))
    b = _trace(tmp_path / "rec-b-recorder", _upload(21, 0x2100, 64, "b" * 64))
    base = tmp_path / "base.txt"
    base.write_text("# base\n")
    residual = tmp_path / "r.json"
    residual.write_text(json.dumps({"residual": {"upload_needs_live_rule": [["x", "y", 20]]}}))
    done = subprocess.run([sys.executable, str(env.TOOLS / "gen_immutable.py"), str(a), str(b), str(base),
                           str(tmp_path / "o.txt"), "test", str(residual)], capture_output=True, text=True)
    assert done.returncode != 0 and "not equal in both recordings" in done.stderr


def test_shipped_files_are_present():
    for name in ("driver_trace_cupti_held.c", "driver_trace_cupti_fallback.h", "upload_capture_thread.h", "d2h_capture.h",
                 "upload_capture.h", "observe_cublas_geam.c", "observe_library_calls.cc", "sha256.cu"):
        assert (env.NATIVE / name).is_file(), name
    for name in ("immutable/hrrrv5-x1.2562.txt", "cindexer-pairs-base.json", "digest/sha256-sm_89.cubin.gz",
                 "digest/sha256-sm_120.cubin.gz", "verified/image-constants.json",
                 "verified/recovered-lw-source-proof.json", "verified/recovered-lw-source.cu.gz"):
        assert (env.DATA / name).is_file(), name
    for script in ("record_observer.py", "observe_slotted_census.py", "catalog_v10.py", "make_partial_plan.py",
                   "seal_phys_stage2.py", "seal_phys_stage3.py", "finalize_package.py", "p1fix_package.py",
                   "gen_immutable.py", "prove_cindexer_pairs.py", "make_alloc_map.py", "make_frame_states.py"):
        assert (env.TOOLS / script).is_file(), script


def test_no_fixed_host_paths_in_the_shipped_tree():
    # The project's own folder names are read from the private pattern file (``host_tokens``; HEX_PRIVATE_PATTERNS
    # names it, and it is never packaged), beside the generic roots.
    pattern = ("/" + "home/", "/" + "work/ad")  # split: hygiene scans grep for them
    private = os.environ.get("HEX_PRIVATE_PATTERNS")
    if private:
        pattern += tuple(json.loads(Path(private).read_text(encoding="utf-8"))["host_tokens"])
    for path in env.HERE.rglob("*"):
        if path.is_file() and path.suffix not in (".cubin", ".gz", ".pyc"):
            text = path.read_text(errors="replace")
            assert not any(p in text for p in pattern), path


def test_record_refuses_an_uncatalogued_suite_and_a_partial_step(tmp_path):
    init = tmp_path / "init.nc"
    init.write_bytes(b"x")
    base = [sys.executable, "-m", "woof.hex.ad.record", "record", "--mesh", "x1.2562", "--init", str(init),
            "--grid", str(init), "--out", str(tmp_path / "pkg"), "--runner", sys.executable]
    done = subprocess.run(base + ["--suite", "mesoscale_reference", "--dt", "1200", "--hours", "1"],
                          capture_output=True, text=True)
    assert done.returncode == 3 and "launch catalog" in done.stderr
    done = subprocess.run(base + ["--dt", "700", "--hours", "1"], capture_output=True, text=True)
    assert done.returncode == 3 and "whole number" in done.stderr


@pytest.mark.parametrize("command", ["build", "record", "replay", "retain"])
def test_help(command):
    done = subprocess.run([sys.executable, "-m", "woof.hex.ad.record", command, "--help"], capture_output=True, text=True)
    assert done.returncode == 0 and "usage" in done.stdout


def test_recorder_source_includes_the_cupti_param_structs_once():
    """cupti.h includes generated_cuda_meta.h, which has no include guard. THE BREAKAGE THIS PREVENTS: the recorder
    included it a second time, so building recorder.so failed with "redefinition of struct cuInit_params_st" and every
    door --init stopped before recording (box-d, CUPTI 13.5.25 wheel, 2026-11-11; the CUDA 13.4.92 toolkit headers
    are the same bytes)."""
    import re
    source = (env.NATIVE / "driver_trace_cupti_held.c").read_text()
    includes = re.findall(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", source, flags=re.M)
    assert "cupti.h" in includes
    assert "generated_cuda_meta.h" not in includes


def test_the_sealer_finds_its_source_sites_by_text_in_the_shipped_hexcore():
    """Three sealer rules belong to source lines of hexcore: the long-wave wrapper's call of _layer_groups (the
    observer's own reads), its one host read `sizes = hist.get()`, and the RRFS rate-table upload. THE BREAKAGE THIS
    PREVENTS: the rules named line numbers (rrtmg_lw_v842.py:1190 and :1019, driver_rrfs.py:2657); the files grew, and
    a 120 km door recording from --init refused at the seal with device_to_host_consumer 2 and
    upload_needs_live_rule 1 (box-d d120-d and d120-f, 2026-11-11). The rules now match file and text, so each text
    must be in the shipped source exactly once, and the tools must not name those lines again."""
    hexcore = env.hexcore_src()
    lw = (hexcore / "mpasphys" / "rrtmg_lw_v842.py").read_text().splitlines()
    assert sum("_layer_groups(" in line and not line.lstrip().startswith("def ") for line in lw) == 1
    assert sum("sizes = hist.get()" in line for line in lw) == 1
    driver = (hexcore / "mpasphys" / "rrfs" / "driver_rrfs.py").read_text()
    assert driver.count("table = cp.asarray(np.array([array.data.ptr for array in table_arrays], dtype=np.uint64))") == 1
    classify = (env.TOOLS / "classify_d2h_checked.py").read_text()
    stage2 = (env.TOOLS / "seal_phys_stage2.py").read_text()
    assert "'_layer_groups('" in classify and "'sizes = hist.get()'" in classify
    assert "startswith('woof/hex/mpasphys/rrtmg_lw_v842.py:" not in classify
    assert "'rrtmg_lw_v842.py:1019' in site" not in classify
    assert "'mpasphys/rrfs/driver_rrfs.py:2657'" not in stage2
    assert "table = cp.asarray(np.array([array.data.ptr for array in table_arrays], dtype=np.uint64))" in stage2


def test_the_door_receipt_says_what_a_plan_recorded(tmp_path):
    """plan_recording reads a package's own record-receipt.json and manifest.json: a door recording says radiation and
    land were recorded live and carries no manifest; a package built by the recording tools names its manifest."""
    from woof.hex.ad_doors import plan_recording
    door = tmp_path / "door"
    door.mkdir()
    (door / "record-receipt.json").write_text(json.dumps({"schema": "hexad.record-receipt/v1", "numerics": "strict",
                                                          "hold_slots": "", "residual": {}}))
    said = plan_recording(door)
    assert said["manifest"] is None and said["radiation_and_land"] == "recorded live" and said["numerics"] == "strict"
    lane = tmp_path / "lane"
    lane.mkdir()
    (lane / "record-receipt.json").write_text(json.dumps({"schema": "x", "numerics": "strict"}))
    (lane / "manifest.json").write_text(json.dumps({"name": "int7 radiation-frozen, land live"}))
    said = plan_recording(lane)
    assert said["manifest"]["name"] == "int7 radiation-frozen, land live" and len(said["manifest"]["sha256"]) == 64
    assert "radiation_and_land" not in said
    assert plan_recording(tmp_path / "none") == {"manifest": None, "recorder_schema": None, "numerics": None}


# ---- the card-to-host capture and the payload check (x1.40962 --init, 2026-11-11: a kept file named for 163,848 zero
# ---- bytes held other bytes, and the sealed plan refused its own replay with a bare "SHA-256 mismatch")

def _package_with_payloads(root: Path, payloads: dict[str, bytes]) -> Path:
    """A package folder whose plan names every payload given as {name it is filed under: bytes it holds}."""
    (root / "recording" / "payloads").mkdir(parents=True)
    for name, data in payloads.items():
        (root / "recording" / "payloads" / f"{name}.gz").write_bytes(gzip.compress(data, 1))
    ops = [{"kind": "upload", "data": {"path": f"recording/payloads/{n}.gz", "sha256": n}} for n in payloads]
    (root / "plan.json").write_text(json.dumps({"blocks": {"step_01": ops}}))
    return root


def test_the_payload_check_refuses_a_planted_bad_payload_by_file_name(tmp_path):
    import hashlib
    import struct
    from woof.hex.ad.record import chain
    good = b"edge weights" * 999
    zeros = bytes(163848)
    table = struct.pack("<40962i", *([5] * 12 + [6] * 40950))  # what the bad file of the real recording held
    zero_name = hashlib.sha256(zeros).hexdigest()
    assert zero_name == "2f9fea7963dc2be4aa68c110b25231e96ce5cca421a61b67acf5c4b3c56d5c5a"
    package = _package_with_payloads(tmp_path / "bad", {hashlib.sha256(good).hexdigest(): good, zero_name: table})
    # a file the plan does not name is not the plan's business, whatever it holds
    (package / "recording" / "payloads" / ("0" * 64 + ".gz")).write_bytes(gzip.compress(b"not named", 1))
    result = chain.check_payloads(package)
    assert result == {"checked": 2, "bad": [[zero_name + ".gz", hashlib.sha256(table).hexdigest()]]}
    run = chain.Run(tmp_path / "work")
    with pytest.raises(SystemExit) as refusal:
        chain.payload_gate(run, package)
    line = (tmp_path / "work" / "record.log").read_text()
    assert f"recording/payloads/{zero_name}.gz holds bytes that hash to {hashlib.sha256(table).hexdigest()}" in line
    assert "1 of 2 plan-named payload files" in line and refusal.value.code not in (0, None)
    # the same package with the file holding the bytes of its name passes, and a cut-off file is refused by name
    whole = _package_with_payloads(tmp_path / "good", {hashlib.sha256(good).hexdigest(): good, zero_name: zeros})
    assert chain.payload_gate(chain.Run(tmp_path / "work2"), whole) == {"checked": 2, "bad": []}
    cut = whole / "recording" / "payloads" / f"{zero_name}.gz"
    cut.write_bytes(cut.read_bytes()[:40])
    bad = chain.check_payloads(whole)["bad"]
    assert len(bad) == 1 and bad[0][0] == zero_name + ".gz" and bad[0][1].startswith("unreadable")


def test_record_checks_the_payloads_before_the_package_is_made_whole():
    source = (Path(env.__file__).parent / "chain.py").read_text()
    body = source[source.index("def record(o)"):]
    assert body.index("package(o, run, pair, retention, pairs)") < body.index("payload_gate(run, o.out)") \
        < body.index("internalize(o, run)") < body.index('replay(o.out, o.steps')


def _harness(tmp_path: Path, *flags: str) -> Path:
    import shutil
    cc = shutil.which("gcc") or shutil.which("cc")
    if not cc:
        pytest.skip("no C compiler")
    exe = tmp_path / ("harness" + "".join(flags).replace("-D", "-"))
    built = subprocess.run([cc, "-O2", "-Wall", *flags, f"-I{env.NATIVE}",
                            str(Path(__file__).parent / "data" / "ad_record" / "d2h_harness.c"), "-o", str(exe),
                            "-lpthread", "-lz", "-lcrypto"], capture_output=True, text=True)
    if built.returncode and ("zlib.h" in built.stderr or "openssl" in built.stderr or "cannot find -l" in built.stderr):
        pytest.skip("zlib or libcrypto development files missing")
    assert built.returncode == 0, built.stderr
    return exe


def _misnamed(root: Path) -> list[str]:
    import hashlib
    return sorted(p.name for p in (root / "payloads").glob("*.gz")
                  if hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest() != p.name[:-3])


def test_a_kept_card_to_host_read_is_named_for_the_bytes_it_holds_while_the_buffer_changes(tmp_path):
    """The capture hashes and writes ONE private copy. A second thread rewrites the host buffer all the time."""
    exe = _harness(tmp_path)
    root = tmp_path / "rec"
    (root / "payloads").mkdir(parents=True)
    done = subprocess.run([str(exe), "store", str(root), "400", "163848"], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(done.stdout) == {"stored": 400, "errors": 0}
    kept = list((root / "payloads").glob("*.gz"))
    assert kept and not _misnamed(root)
    assert not [p for p in (root / "payloads").iterdir() if p.name.startswith(".")]  # no temporary file is left


def test_an_asynchronous_card_to_host_read_is_waited_for_before_it_is_hashed(tmp_path):
    """MemcpyDtoHAsync with no stream (the recorded case): the bytes land only when the context is waited for."""
    import hashlib
    import struct
    exe = _harness(tmp_path)
    root = tmp_path / "rec"
    (root / "payloads").mkdir(parents=True)
    done = subprocess.run([str(exe), "wait", str(root)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    rows = {r["case"]: r for r in map(json.loads, done.stdout.splitlines())}
    landed = struct.pack("<1024i", *([5] * 12 + [6] * 1012))
    assert rows["async-no-stream"] == {"case": "async-no-stream", "sha256": hashlib.sha256(landed).hexdigest(),
                                       "stream_waits": 0, "context_waits": 1}
    assert rows["async-no-stream"]["sha256"] != hashlib.sha256(bytes(4096)).hexdigest()
    assert rows["async-stream"] == {"case": "async-stream", "stream_waits": 1, "context_waits": 0}
    assert rows["sync"] == {"case": "sync", "stream_waits": 0, "context_waits": 0}
    assert rows["async-no-driver"]["rc"] != 0
    assert not _misnamed(root)
    # every card-to-host call site says whether its copy is asynchronous
    source = (env.NATIVE / "driver_trace_cupti_held.c").read_text()
    assert source.count("trace_d2h_bytes(d,n,t,1)") == 1 and source.count("trace_d2h_bytes(d,n,NULL,0)") == 1
    assert source.count("trace_d2h_bytes((void *)(uintptr_t)(d),n,t,1)") == 1
    assert "SHA256((const unsigned char *)host, bytes, digest)" not in source


def test_every_recorded_allocation_is_born_zero():
    """x1.40962 --init: the 504 bytes past a 163,848-byte array, and a block nothing uses, held stale bytes in the
    recording and zero in the replay; the strict replay refused 16 allocations whose written bytes were all equal."""
    source = (env.NATIVE / "driver_trace_cupti_held.c").read_text()
    assert '#define TRACE_MemAlloc(p,n) do { if (!status) trace_born_zero(*p,n,NULL,0);' in source
    assert '#define TRACE_MemAllocAsync(p,n,s) do { if (!status) trace_born_zero(*p,n,s,1);' in source
    body = source[source.index("static void trace_born_zero(unsigned long long address, size_t bytes, void *stream, int asynchronous) {"):]
    body = body[:body.index("\n}\n")]
    assert '"cuMemsetD8_v2"' in body and '"cuMemsetD8Async"' in body and "binder_internal = 1;" in body
    assert 'trace_basic("AllocationBornZeroError"' in body  # a fill that fails is in the trace, never silent


def test_a_plain_allocation_is_zero_before_the_forecast_gets_it():
    """A 120 km recording: one of two forecasts stopped at step 1 with 'wind reconstruction failed geometry
    validation (invalid word 1)'. cuMemsetD8_v2 runs on the legacy stream and returns before it has run; the forecast
    also writes on non-blocking streams, which that stream is not ordered against. The fill is waited for there; the
    stream-ordered fill of cuMemAllocAsync is on the allocation's own stream and is not waited for."""
    source = (env.NATIVE / "driver_trace_cupti_held.c").read_text()
    body = source[source.index("static void trace_born_zero(unsigned long long address, size_t bytes, void *stream, int asynchronous) {"):]
    body = body[:body.index("\n}\n")]
    assert 'resolve(driver, "cuStreamSynchronize")' in body
    fill = body.index("fill(address, 0, bytes)")
    wait = body.index("if (!rc && !asynchronous) rc = wait_stream ? wait_stream((void *)CU_STREAM_LEGACY) : -1;")
    assert fill < wait < body.index("binder_internal = previous;")
    assert "wait_stream(stream)" not in body   # the stream-ordered fill is not waited for
    assert 'trace_basic("AllocationBornZeroError"' in body   # a fill or wait that fails is in the trace


def _cgroup(directory: Path, *, current: int, cache: int = 0, high: str = "max", top: str = "max") -> str:
    directory.mkdir(parents=True)
    (directory / "memory.current").write_text(f"{current}\n")
    (directory / "memory.stat").write_text(f"anon {current - cache}\nactive_file {cache // 2}\n"
                                           f"inactive_file {cache - cache // 2}\nshmem 0\n")
    (directory / "memory.high").write_text(f"{high}\n")
    (directory / "memory.max").write_text(f"{top}\n")
    return str(directory)


def test_the_packaging_pool_is_sized_from_the_job_s_own_memory_limit(tmp_path, monkeypatch):
    """A 12 km recording on NOAA's hrrrv5 deck, 2026-11-16: inside a scope of MemoryHigh=20G MemoryMax=40G on a 512 GB
    host the packaging step read no limit (it looked only at the root of the cgroup tree), started 16 workers and
    stalled at the soft limit until the recording's time limit ended it; outside the scope it took 20 s."""
    monkeypatch.syspath_prepend(str(env.TOOLS))
    monkeypatch.delenv("P1_PKG_WORKERS", raising=False)
    import p1fix_package as tool
    gib = 1 << 30
    # the process's own cgroup is read from its own line, nearest directory first, up to the root
    own = tmp_path / "proc-cgroup"
    own.write_text("0::/user.slice/user-1000.slice/run-7.scope\n")
    dirs = tool._cgroup_dirs(str(own), "/sys/fs/cgroup")
    assert dirs == ["/sys/fs/cgroup/user.slice/user-1000.slice/run-7.scope", "/sys/fs/cgroup/user.slice/user-1000.slice",
                    "/sys/fs/cgroup/user.slice", "/sys/fs/cgroup"]
    # the scope of that recording: 37 GiB of file cache charged (the two recordings), 1 GiB of its own memory
    scope = _cgroup(tmp_path / "t" / "slice" / "scope", current=38 * gib, cache=37 * gib, high=str(20 * gib),
                    top=str(40 * gib))
    parent = _cgroup(tmp_path / "t2" / "slice", current=90 * gib, cache=40 * gib, top=str(114 * gib))
    root = str(tmp_path / "t3")           # the root holds no memory files
    (tmp_path / "t3").mkdir()
    assert tool.cgroup_room_bytes([scope, parent, root]) == 19 * gib      # the soft limit, less what cannot be dropped
    assert tool.cgroup_room_bytes([parent, root]) == 64 * gib             # a parent's hard limit alone
    assert tool.cgroup_room_bytes([root]) is None                         # no limit anywhere: the host decides
    # the pool: 75 residue jobs of 2 x 980 MB each; half of the room in flight, never the host's 16
    peak = 2 * 980 * (1 << 20)
    assert tool.pool_workers(75, peak, free=327 * gib) == min(16, os.cpu_count() or 4)
    assert tool.pool_workers(75, peak, free=tool.cgroup_room_bytes([scope, parent, root])) == min(4, os.cpu_count() or 4)
    monkeypatch.setattr(tool, "cgroup_room_bytes", lambda dirs=None: 3 * gib)
    assert tool.free_memory_bytes() <= 3 * gib
    assert tool.pool_workers(75, peak) == 1


def test_every_tool_the_chain_runs_is_in_the_shipped_tree():
    """A 12 km recording on NOAA's hrrrv5 deck, 2026-11-16: a recorder staged by hand onto a rented card host recorded
    for 338 s and then died in 1 s at the seal, because the layout tool the chain opens only after the card
    (make_layout_source.py) had not been copied. The release recorder ships its tools as package files; this names
    every script chain.py and cli.py run, so none can be
    left out of a build unseen."""
    import re
    here = Path(env.__file__).parent
    named = set()
    for source in ("chain.py", "cli.py"):
        named |= set(re.findall(r'_py\("([\w.]+\.py)"\)', (here / source).read_text()))
        named |= set(re.findall(r'env\.TOOLS / "([\w.]+\.py)"', (here / source).read_text()))
    assert {"make_layout_source.py", "p1fix_package.py", "finalize_package.py", "make_alloc_map.py"} <= named
    missing = sorted(n for n in named if not (env.TOOLS / n).is_file())
    assert not missing, missing


def test_the_sealer_finds_its_output_step_sites_by_text_in_the_shipped_hexcore(tmp_path, monkeypatch):
    """A 12 km recording from the sealed 0.6.0 candidate's wheel, 2026-11-16: the window holds one output time
    (step 15) and the pair could not seal: 27 history downloads, the isobaric literals and the pool re-pack after the
    diagnostics reset were each refused, because the rules named lines of an older copy of the forward
    (streams_output_v842.py:394, isobaric.py:199/205/229/235, device_pools_v842.py:104) and the shipped files had
    moved (426 and 445, 201/207/231/237, 106). The rules now find each site by its text; every rule site must be in
    the shipped source, and a download from either line of _queue_download must take the history rule."""
    monkeypatch.syspath_prepend(str(env.TOOLS))
    import seal_phys_stage2 as stage2
    hexcore = env.hexcore_src()
    alias, verified = stage2.shipped_sites(hexcore)
    assert verified == set(stage2.SITE_TEXTS), sorted(set(stage2.SITE_TEXTS) - verified)
    assert set(stage2.SOURCE_LITERALS) | {stage2.KISS_JUMP_SITE} | {s for s, _ in stage2.RESET_ZERO_SITES} <= verified
    by_site = {}
    for shipped, site in alias.items():
        by_site.setdefault(site, []).append(shipped)
    # one shipped line a site, but for the history stream: its two downloads are both lines of _queue_download
    history = "mpasphys/streams_output_v842.py:426"
    assert "woof/hex/" + history in stage2.HISTORY_OUTPUT_SITES
    assert {site: len(lines) for site, lines in by_site.items() if len(lines) != 1} == {history: 2}
    source = (hexcore / "mpasphys" / "streams_output_v842.py").read_text()
    body = source[source.index("def _queue_download(self, cp, queued):"):]
    body = body[:body.index("\n    def ", 10)]
    for text in stage2.SITE_TEXTS[history]:
        assert body.count(text) == 1 and source.count(text) == 1, text
    # the rule itself: a download whose stack starts at either shipped line is an output-only download
    op = {"kind": "device_to_host", "stream": 1, "source": {"allocation": 7, "offset": 0}, "bytes": 64,
          "asynchronous": True, "recorded_sequence": 5}
    for shipped in by_site[history]:
        stack = f"woof/hex/{shipped};woof/hex/mpasphys/runtime_v842.py:1"
        assert stage2.history_output_rule(op, stack) is None or shipped == history   # by line number alone: refused
        live = stage2.history_output_rule(op, stack, alias)
        assert live and live["host_consumer"] == "observational_output" and live["site"] == f"woof/hex/{shipped}"
    assert stage2.history_output_rule(op, "woof/hex/mpasphys/runtime_v842.py:1;x", alias) is None
    # a literal upload at a site whose text is not in the given source is refused; a file that moved keeps its rule
    import struct
    site = "mpasdiag/isobaric.py:231"
    text, values = stage2.SOURCE_LITERALS[site]
    payload = struct.pack(f"<{len(values)}f", *values)
    upload = {"kind": "upload", "destination": {"allocation": 3, "offset": 0}, "data": {"path": "x", "sha256": "0"}}
    assert stage2.literal_upload_rule(upload, site, "outside", payload, verified)["proof"]["rule"] == "source_literal_float32"
    assert stage2.literal_upload_rule(upload, site, "outside", payload, set()) is None
    moved = tmp_path / "hexcore" / "mpasdiag"
    moved.mkdir(parents=True)
    (moved / "isobaric.py").write_text("\n" * 40 + "    " + text + "\n")
    alias2, verified2 = stage2.shipped_sites(tmp_path / "hexcore")
    assert alias2 == {"mpasdiag/isobaric.py:41": site} and verified2 == {site}


def test_the_first_output_step_moisture_slot_upload_has_a_site_and_a_rule(tmp_path, monkeypatch):
    """The 12 km release example by the release recorder (port rowrec r2, a test host, 2026-11-16) refused at seal with
    one 'upload_needs_live_rule' word at step 15: the isobaric diagnostics' moisture slot (current_qv) is uploaded by
    cp.array, which the observer did not see, so the upload had no site. The observer now records cp.array of a host
    ndarray (never of a list, whose stacked pointer table keeps no site), and the resident-output-initial rule takes
    that line of the shipped file by its text."""
    monkeypatch.syspath_prepend(str(env.TOOLS))
    import numpy as np
    import observe_host_asarray as host
    import seal_phys_stage2 as stage2
    hexcore = env.hexcore_src()
    rel = stage2.RESIDENT_OUTPUT_INITIAL[0]
    lines = (hexcore / rel).read_text().splitlines()
    for text in stage2.RESIDENT_OUTPUT_INITIAL[1:]:
        hits = [i + 1 for i, line in enumerate(lines) if text in line]
        assert len(hits) == 1, (text, hits)
        assert stage2.resident_output_initial(f"{rel}:{hits[0]}", hexcore)
    assert not stage2.resident_output_initial(f"{rel}:1", hexcore)

    class Out:
        def __init__(self, a):
            self.nbytes, self.shape, self.dtype = a.nbytes, a.shape, a.dtype
            self.data = type("P", (), {"ptr": 4096})()

    calls = []

    class FakeCupy:
        @staticmethod
        def asarray(value, dtype=None, order=None):
            calls.append("asarray")
            return Out(np.asarray(value, dtype=dtype))

        @staticmethod
        def array(value, dtype=None, copy=True, order="K"):
            calls.append("array")
            return Out(np.asarray(value, dtype=dtype))

    class Sink:
        def __init__(self):
            import threading
            self.lock, self.rows = threading.Lock(), []
            self.host_asarray_output = self

        def write(self, line):
            self.rows.append(json.loads(line))

        def flush(self):
            pass

    sink = Sink()
    monkeypatch.setattr(host.base, "forecast_step", lambda: 15)
    cp = FakeCupy()
    host.install_upload_hooks(cp, np, sink)
    import importlib.util
    mod = tmp_path / "hexcore" / "mpasphys" / "qv_site.py"
    mod.parent.mkdir(parents=True)
    mod.write_text("def first(cp, initial):\n"
                   "    return cp.array(initial[1], dtype=initial.dtype, copy=True, order='C')\n"
                   "def other(cp, initial):\n"
                   "    cp.asarray(initial[2], order='C')\n"
                   "    return cp.array([initial[0], initial[1]])\n")
    spec = importlib.util.spec_from_file_location("qv_site", mod)
    site = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(site)
    initial = np.arange(3 * 59 * 7, dtype=np.float32).reshape(3, 59, 7)
    site.first(cp, initial)
    site.other(cp, initial)                 # the list form may stack device arrays: no record
    assert calls == ["array", "asarray", "array"]
    assert len(sink.rows) == 2
    assert all(r["success"] and r["input"]["bytes"] == 59 * 7 * 4 for r in sink.rows)
    assert [(Path(r["stack"][0]["path"]).name, r["stack"][0]["line"]) for r in sink.rows] == [
        ("qv_site.py", 2), ("qv_site.py", 4)]
    assert sink.rows[0]["input"]["sha256_c"] == __import__("hashlib").sha256(initial[1].tobytes()).hexdigest()


def test_the_receipt_names_the_deck_the_recorded_forecast_ran(tmp_path):
    """The 3 km recording of 2026-11-16 ran radiation every 15 minutes (the plain door's per-mesh row) where the deck
    that matches the reference model on that box runs it every 60; no file of the package said which. The receipt now
    carries the forecast's own words."""
    from woof.hex.ad.record import chain
    end = {"schema": "mpas-native-forecast-v1", "completed": True,
           "physics": {"suite": "hrrrv5", "config": {"config_radtlw_interval": "00:30:00", "config_radtsw_interval": "00:30:00",
                                                    "config_convection_scheme": "cu_ntiedtke", "config_gfl_sub3d": 1}},
           "namelist": {"physics": {"config_radtlw_interval": "00:30:00", "config_radtsw_interval": "00:30:00"}}}
    door = ("woof hex: NOAA hrrrv5 deck for this mesh (12000 m nominal): rrfs-workflow config.meshdep conus12km row "
            "config_radtlw_interval=00:30:00, config_radtsw_interval=00:30:00, config_number_of_sub_steps=2")
    log = tmp_path / "record-a.log"
    log.write_text("seed_payloads: manifest 665\n" + door + "\nnot json {\n" + json.dumps(end) + "\n")
    deck = chain.deck_of(log)
    assert deck["config_radtlw_interval"] == "00:30:00" and deck["config_radtsw_interval"] == "00:30:00"
    assert deck["config_convection_scheme"] == "cu_ntiedtke" and deck["config_gfl_sub3d"] == 1
    assert deck["door_line"] == door and "end record" in deck["source"]
    # two places of one record that disagree are both kept, never one chosen
    end["namelist"]["physics"]["config_radtlw_interval"] = "00:15:00"
    log.write_text(json.dumps(end) + "\n")
    assert chain.deck_of(log)["config_radtlw_interval"] == ["00:30:00", "00:15:00"]
    # a log with no end record: null, and the chain says so in its log
    log.write_text("woof hex: stopped\n")
    assert chain.deck_of(log) is None and chain.deck_of(tmp_path / "absent.log") is None
    source = (Path(env.__file__).parent / "chain.py").read_text()
    assert '"deck": deck,' in source and 'deck = deck_of(o.work / "record-a.log")' in source
