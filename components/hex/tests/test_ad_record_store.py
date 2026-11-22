# SPDX-License-Identifier: Apache-2.0
"""The lean boundary store (no boundary bytes kept for no reader; the recorder writes the residue digests)."""
from __future__ import annotations

import gzip
import hashlib
import json

from woof.hex.ad.record import chain, env


def _tool(monkeypatch):
    monkeypatch.syspath_prepend(str(env.TOOLS))
    import p1fix_package as tool
    return tool


def test_the_recorder_digests_every_boundary_and_keeps_bytes_only_where_a_held_write_landed():
    source = (env.NATIVE / "driver_trace_cupti_held.c").read_text()
    assert "step<=73" not in source and "    if (step>1) {" in source          # plans past 72 steps can seal
    assert "int binder_residue_rows(const unsigned long long *pointers" in source
    assert 'BoundaryMaskedDigest' in source
    assert "if (worker->lean && !allocation->keep) { free(bytes); continue; }" in source
    assert 'lean=payloads && !strcmp(payloads,"digest")' in source


def test_the_store_word(monkeypatch):
    monkeypatch.delenv("WOOF_HEX_RECORD_STORE", raising=False)
    assert chain.BOUNDARY_STORES[chain.boundary_store()] == "digest"
    monkeypatch.setenv("WOOF_HEX_RECORD_STORE", "full")
    assert chain.BOUNDARY_STORES[chain.boundary_store()] == "1"


def test_the_recorder_s_residue_digest_stands_for_the_payload_and_must_agree_with_it(tmp_path, monkeypatch):
    tool = _tool(monkeypatch)
    data = bytes(range(256)) * 64
    ranges = [{"offset": 100, "bytes": 50, "array": "cost"}, {"offset": 4000, "bytes": 96, "array": "warp_work"}]
    masked = bytearray(data)
    for r in ranges:
        masked[r["offset"]:r["offset"] + r["bytes"]] = bytes(r["bytes"])
    sha, want = hashlib.sha256(data).hexdigest(), hashlib.sha256(bytes(masked)).hexdigest()
    path = tmp_path / f"{sha}.gz"
    job = (3, 414, ranges, sha, str(path))
    line = {"step": 3, "base": 1 << 40, "bytes": len(data), "sha256": sha, "masked_sha256": want,
            "ranges": [[100, 50], [4000, 96]]}
    # lean: no payload, the recorder's line
    assert tool.recorded_masked(job, line, None) == (None, want, "recorder")
    # full: both, and they agree (the payload's own rule, masked_payload)
    path.write_bytes(gzip.compress(data, 1))
    loaded = tool.masked_payload(job)
    assert loaded == (None, want)
    assert tool.recorded_masked(job, line, loaded) == (None, want, "payload=recorder")
    # a disagreement, other ranges, other bytes, or neither source: refused, never sealed
    assert tool.recorded_masked(job, dict(line, masked_sha256="0" * 64), loaded)[0].startswith("refusing")
    assert tool.recorded_masked(job, dict(line, ranges=[[100, 50]]), None)[0].startswith("refusing")
    assert tool.recorded_masked(job, dict(line, sha256="1" * 64), None)[0].startswith("refusing")
    path.unlink()
    assert tool.recorded_masked(job, None, None)[0].startswith("refusing")


def test_the_residue_digests_are_read_in_the_boundary_pass(tmp_path, monkeypatch):
    tool = _tool(monkeypatch)
    trace = tmp_path / "driver-1.jsonl.gz"
    rows = [{"type": "Boundary", "a": 2, "clock_ns": 7},
            {"type": "BoundaryAllocationDigest", "step": 1, "base": 5, "bytes": 8, "sha256": "a" * 64},
            {"type": "BoundaryMaskedDigest", "step": 1, "base": 5, "bytes": 8, "sha256": "a" * 64,
             "masked_sha256": "b" * 64, "ranges": [[0, 4]]}]
    trace.write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode()))
    assert tool.boundary_clock(trace, 2) == 7
    assert tool.masked_digests(trace)[(1, 5, 8)]["masked_sha256"] == "b" * 64


def test_a_boundary_file_refresh_upload_has_a_site_rule(monkeypatch):
    """A 30 km recording (25 steps from 12Z): 17 'upload_needs_live_rule' words at step 25, the 13Z boundary file's
    values uploaded onto the boundary support (cuda_regional_forecast_v841.py _derive_on_support)."""
    monkeypatch.syspath_prepend(str(env.TOOLS))
    import seal_phys_stage2 as stage2
    src = env.hexcore_src()
    lines = (src / stage2.LBC_FILE_VALUES[0]).read_text().splitlines()
    at = [i + 1 for i, line in enumerate(lines) if stage2.LBC_FILE_VALUES[1] in line]
    assert len(at) == 1 and "_derive_on_support" in "\n".join(lines[at[0] - 25:at[0]])
    site = f"{stage2.LBC_FILE_VALUES[0]}:{at[0]}"
    assert stage2.resident_output_initial(site, src, stage2.LBC_FILE_VALUES)
    assert not stage2.resident_output_initial(f"{stage2.LBC_FILE_VALUES[0]}:{at[0] - 1}", src, stage2.LBC_FILE_VALUES)
    assert not stage2.resident_output_initial(site, src)   # not the resident output's rule
