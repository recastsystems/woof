"""``hoststore.host_memory`` answers on every platform the product ships on.

THE DEFECT THIS EXISTS FOR
--------------------------
``tilestream/hoststore.py`` read ``/proc/meminfo`` unconditionally, so on
Windows -- the platform the product's own front door ships on, and the one
the cut box runs -- EVERY ``HostDomainStore`` construction died in
``check_allocatable`` with ``FileNotFoundError: /proc/meminfo`` before a
single byte was pinned.  Seven of ``tilestream/test_hoststore.py``'s eight
gates failed on that one line, and the out-of-core host store -- the module
that makes a larger-than-VRAM domain possible at all -- was unusable on the
machine it was built for.  ``tilestream/autoplan.py`` had already grown the
``GlobalMemoryStatusEx`` branch for exactly this reason (its module
docstring: "a planner that had no host source on Windows did not fall back
to a guess, it RAISED"); the store never inherited it.

This file is SEPARATE from ``test_hoststore.py`` on purpose: that module's
helpers import cupy, so the whole module is auto-marked ``gpu`` and the
stage-1 leg (GPUWM_NO_LOCAL_GPU=1, the Windows cut box) skips it.  The
host-memory read needs no card, and the box that needs this gate most is
exactly the one the gpu shard never runs on -- so the gate lives where the
CPU leg can see it.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sys

import pytest

from tilestream import hoststore as HS


def test_host_memory_answers_on_this_platform() -> None:
    """The whole defect: this call must not need procfs to exist."""
    mem = HS.host_memory()
    assert set(mem) == {"total", "available", "free"}, sorted(mem)
    for key, value in mem.items():
        assert isinstance(value, int), (key, type(value))


def test_host_memory_values_are_sane() -> None:
    """A source that answered zeros would wave every allocation through
    ``check_allocatable``'s fraction cap (0.47 * 0 = 0 refuses everything)
    or refuse everything (available 0), so the values are held to physical
    sense, not just to existing."""
    mem = HS.host_memory()
    assert mem["total"] >= (1 << 30), (
        f"MemTotal {mem['total']} B is under 1 GiB; no supported box is "
        "that small, so the source is misreading")
    assert 0 < mem["available"] <= mem["total"], mem
    assert 0 <= mem["free"] <= mem["total"], mem


@pytest.mark.skipif(os.name != "nt", reason="the Windows branch is only "
                    "measurable on Windows")
def test_windows_branch_does_not_open_procfs() -> None:
    """On Windows the answer comes from ``GlobalMemoryStatusEx``, full stop.

    Pinned by construction rather than by mocking ``open``: the Windows
    reader is called directly and must agree with ``host_memory`` -- if a
    procfs read ever came back in front of it, the two would still agree,
    but ``host_memory`` would already have raised on the box running this
    test, which has no ``/proc``.
    """
    direct = HS._host_memory_windows()
    assert direct["total"] >= (1 << 30)
    assert 0 < direct["available"] <= direct["total"]
    via_public = HS.host_memory()
    # Same source, sampled twice: totals are constant, available drifts.
    assert via_public["total"] == direct["total"]


def test_linux_parser_still_reads_meminfo_text() -> None:
    """The /proc/meminfo PARSER, held to a fixture so the Linux arm cannot
    regress while the Windows arm is being fixed.  Pure text, no procfs."""
    sample = ("MemTotal:       98467840 kB\n"
              "MemFree:        61906944 kB\n"
              "MemAvailable:   96468992 kB\n"
              "Buffers:              0 kB\n")
    parsed = HS._parse_meminfo(sample.splitlines())
    assert parsed["MemTotal"] == 98467840 * 1024
    assert parsed["MemAvailable"] == 96468992 * 1024
    assert parsed["MemFree"] == 61906944 * 1024


def test_check_allocatable_runs_on_this_platform(monkeypatch) -> None:
    """The consumer that actually died on Windows, exercised end to end:
    a tiny request must pass all three gates, an absurd one must be
    refused by the fraction cap -- both without touching procfs and
    without allocating anything.

    The machine figures are a stand-in, not the live box: on the 2.9.2
    rehearsal host (123 GB) this read 7.49 GiB available under the
    concurrent legs, below the 8 GiB reserve, and refused the 1 MiB
    request.  The gates are what is measured here, not the box's load.
    """
    monkeypatch.setattr(HS, "host_memory",
                        lambda: {"total": 64 << 30, "available": 32 << 30})
    HS.check_allocatable(1 << 20)                     # 1 MiB: always fine
    with pytest.raises(HS.HostMemoryExhausted):
        HS.check_allocatable(1 << 50)                 # 1 PiB: never fine


# --------------------------------------------------------------------------
# the store reads what its cgroup allows, not the whole host
# --------------------------------------------------------------------------

#: The stand-in cgroup, meminfo and membership files every host-memory
#: reader is held to: the planner's and the renderer's (``rw_host_memory``),
#: and here the store's and its diagnostics'.
_HOST_MEMORY_CASES = json.loads(
    (Path(__file__).resolve().parents[1] / "tools" / "rustwx" / "crates"
     / "rw-host-memory" / "src" / "host_memory_cgroup_cases.json")
    .read_text(encoding="utf-8"))["cases"]
_LIMITED = [case for case in _HOST_MEMORY_CASES if case["limit"] is not None]


def _stand_in(tmp_path, monkeypatch, case) -> dict[str, int]:
    """Point the planner's walk and meminfo path at one case's files, on
    the procfs route a container has; the case's meminfo, parsed."""
    from tilestream import autoplan

    root = tmp_path / "cgroup"
    root.mkdir()
    for relative, text in case["cgroup"].items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    meminfo = tmp_path / "meminfo"
    meminfo.write_bytes(case["meminfo"].encode("ascii"))
    membership = tmp_path / "proc-self-cgroup"
    if case["proc_self_cgroup"] is not None:
        membership.write_bytes(case["proc_self_cgroup"].encode("ascii"))
    monkeypatch.setattr(autoplan, "_CGROUP_ROOT", str(root))
    monkeypatch.setattr(autoplan, "_PROC_SELF_CGROUP", str(membership))
    monkeypatch.setattr(autoplan, "_PROC_MEMINFO", str(meminfo))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv(HS.PINNED_CEILING_ENV, raising=False)
    return {key: int(value) * 1024 for key, value in re.findall(
        r"^(\w+):\s+(\d+) kB$", case["meminfo"], re.MULTILINE)}


@pytest.mark.parametrize("case", _HOST_MEMORY_CASES,
                         ids=[case["name"] for case in _HOST_MEMORY_CASES])
def test_the_store_reads_the_memory_its_cgroup_allows(
        tmp_path, monkeypatch, case):
    """THE BREAKAGE: ``host_memory`` read ``/proc/meminfo`` with no cgroup
    cap.  In a systemd scope with ``MemoryMax=2G`` on a 30 GiB worker it
    read 30.55 GiB total and 25.97 GiB available, so ``check_allocatable``
    with no ``budget_bytes`` admitted a 3 GiB pinned store and
    ``pinned_ceiling_bytes`` put the wall at 15.27 GiB.  The total is the
    smallest limit on the process's cgroup path and the available is the
    least room under any of them, the table the planner and the renderer
    answer.  With no MemAvailable line the store keeps its MemFree reading,
    capped the same way, where the table's readers answer unknown."""
    info = _stand_in(tmp_path, monkeypatch, case)
    mem = HS.host_memory()

    total = info["MemTotal"]
    expected_total = total if case["limit"] is None else min(case["limit"], total)
    assert mem["total"] == expected_total
    if case["available"] is not None:
        assert mem["available"] == case["available"]
    else:
        assert "MemAvailable" not in info, case["name"]
        free = info["MemFree"]
        assert mem["available"] == (free if case["headroom"] is None
                                    else min(free, case["headroom"]))
    assert 0 <= mem["free"] <= mem["total"], mem
    assert HS.pinned_ceiling_bytes() == int(
        HS.PINNED_CEILING_FRACTION * expected_total)


@pytest.mark.parametrize("case", _LIMITED,
                         ids=[case["name"] for case in _LIMITED])
def test_a_store_past_the_room_under_its_limit_is_refused_naming_it(
        tmp_path, monkeypatch, case):
    """The consumer: with no ``budget_bytes`` a store one byte past the
    room under the limit is refused, and the refusal names the limit its
    figure came from, because the host's MemAvailable behind it would have
    admitted a page-locked store the limit cannot hold.  Where the host's
    own available figure is the smaller one (the host has less free than
    the limit's room), the limit did not set the figure and is not named."""
    _stand_in(tmp_path, monkeypatch, case)
    room = HS.host_memory()["available"]
    with pytest.raises(HS.HostMemoryExhausted) as excinfo:
        HS.check_allocatable(room + 1, reserve_bytes=0)
    message = str(excinfo.value)
    named = f"{case['limit'] / HS.GIB:.2f} GiB memory cgroup limit" in message
    capped_by_limit = room < HS._host_memory_uncapped()["available"]
    assert named == capped_by_limit, (case["name"], message)


def test_the_fraction_gate_takes_the_limit_not_the_host(tmp_path, monkeypatch):
    """A 1.5 GiB store inside a 2 GiB scope clears the room (1.625 GiB) but
    is over 47% of the limit, so it is refused; against the host's 512 GiB
    MemTotal it cleared both gates."""
    case = next(case for case in _HOST_MEMORY_CASES
                if case["name"].startswith("cgroup v2 limit on the process's own scope"))
    _stand_in(tmp_path, monkeypatch, case)
    request = 3 * HS.GIB // 2
    assert request < HS.host_memory()["available"]
    with pytest.raises(HS.HostMemoryExhausted) as excinfo:
        HS.check_allocatable(request, reserve_bytes=0)
    assert "of the machine's 2.00 GiB under this process's 2.00 GiB" in str(
        excinfo.value)
    HS.check_allocatable(int(0.47 * 2 * HS.GIB), reserve_bytes=0)


def test_the_endurance_trace_reads_its_own_cgroup_usage(tmp_path, monkeypatch):
    """THE BREAKAGE: the endurance trace's cgroup column read the mount
    root's ``memory.current``, which a cgroup v2 root does not have, so in a
    systemd scope it was NaN from the first step.  It reads the process's
    own cgroup now, and NaN only when no level on the path reports usage."""
    from tilestream import endure

    scope = next(case for case in _HOST_MEMORY_CASES
                 if case["name"].startswith("cgroup v2 limit on the process's own scope"))
    (tmp_path / "scope").mkdir()
    _stand_in(tmp_path / "scope", monkeypatch, scope)
    assert endure._cgroup_gib() == 536870912 / 2 ** 30

    unread = next(case for case in _HOST_MEMORY_CASES
                  if case["name"] == "a limit whose usage cannot be read is the room")
    (tmp_path / "unread").mkdir()
    _stand_in(tmp_path / "unread", monkeypatch, unread)
    assert math.isnan(endure._cgroup_gib())
