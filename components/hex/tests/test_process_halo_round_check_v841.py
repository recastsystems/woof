"""Rank-process halo rounds: headers only in a schema's first round, the round check in shared memory (no GPU needed).

Breakage it removes: conus3km on 8 x RTX PRO 6000, one process per card, rank 0's host waited 48.9 s per forecast
hour (about 272 ms of each 861 ms step) for its neighbours' per-round socket headers, which carried only the round
check after each schema's first round (V8X6000 lease mcs2-tmr-8c, 2026-10-29, MULTICARD/SCALING-2.md section 4).
Breakage the check still prevents: two ranks that disagree on the halo schedule (tag or field schema, or how many
rounds they run) mix halos or hang a card on a flag instead of stopping with the disagreement named.
"""

import inspect

import pytest

from woof.hex import partition_process_transport_v841 as process_transport
from woof.hex.mpasphys import process_ranks_v842
from woof.hex.partition_samehost_transport_v841 import SameHostTransportError

RoundCheck = process_transport.RoundCheck


def _pair():
    """Ranks 0 and 1 joined by one flag page, as RankProcesses makes one memfd page per neighbouring pair."""
    page = bytearray(process_transport._FLAG_PAGE)
    low, high = process_transport._ROUND_RECORD_OFFSET
    return RoundCheck(0, {1: (page, low, high)}), RoundCheck(1, {0: (page, high, low)}), page


def _digest(name):
    return name.encode().ljust(16, b".")


def _run(check, names, start=1):
    for round_, name in enumerate(names, start):
        check.advance(round_, _digest(name), tag=name)


def test_agreeing_ranks_compare_every_round_once_without_waiting():
    a, b, _ = _pair()
    names = [f"round-{index}" for index in range(1, 41)]
    _run(a, names[:25])                  # rank 0's host runs 25 rounds ahead: nothing to wait for
    _run(b, names[:30])                  # rank 1 catches up and passes it, comparing as it goes
    _run(a, names[25:], start=26)
    _run(b, names[30:], start=31)
    a.close()
    b.close()
    assert a.comparisons + b.comparisons >= 2
    # The history reaches back only to the last round the other end may still publish.
    assert len(a.history) == 1 and len(b.history) == 1


def test_a_schema_disagreement_stops_the_end_that_gets_there_second():
    a, b, _ = _pair()
    _run(a, ["dyn-1", "dyn-2", "dyn-3"])
    with pytest.raises(SameHostTransportError, match="disagree on the halo schedule at or before round 3"):
        _run(b, ["dyn-1", "dyn-2", "OTHER"])


def test_a_disagreement_seen_late_is_still_caught_by_the_chain():
    a, b, _ = _pair()
    _run(a, ["dyn-1", "OTHER", "dyn-3", "dyn-4", "dyn-5"])   # rank 0 is ahead past the bad round
    _run(b, ["dyn-1", "dyn-2", "dyn-3", "dyn-4"])            # rank 1 cannot compare a round it has not reached
    with pytest.raises(SameHostTransportError, match="at or before round 5"):
        _run(b, ["dyn-5"], start=5)


def test_a_rank_that_runs_past_a_closed_neighbour_stops():
    a, b, _ = _pair()
    _run(a, ["dyn-1", "dyn-2", "dyn-3"])
    a.finish()
    a.close()
    _run(b, ["dyn-1", "dyn-2", "dyn-3"])
    with pytest.raises(SameHostTransportError, match="closed after 3 halo rounds, but rank 1 is at round 4"):
        _run(b, ["dyn-4"], start=4)


def test_a_rank_that_closes_behind_a_neighbour_stops():
    a, b, _ = _pair()
    _run(b, ["dyn-1", "dyn-2", "dyn-3", "dyn-4"])
    _run(a, ["dyn-1", "dyn-2", "dyn-3"])
    a.finish()
    with pytest.raises(SameHostTransportError, match="is at halo round 4, but rank 0 closed after 3"):
        a.close()


def test_a_record_read_mid_write_is_skipped_not_compared():
    a, b, page = _pair()
    _run(a, ["dyn-1"])
    offset = process_transport._ROUND_RECORD_OFFSET[0] + 8   # the chain bytes of rank 0's record
    page[offset] ^= 0xFF
    assert b.read(0) is None
    _run(b, ["dyn-1"])                    # nothing to compare yet; no false disagreement
    assert b.comparisons == 0


def test_headers_cross_only_in_a_schemas_first_round_and_when_an_fd_must():
    transport = process_transport.ProcessHaloTransport.__new__(process_transport.ProcessHaloTransport)
    announced, heard = set(), set()
    # (schema, packet bytes) per round, the same on both ends: 'a' and 'b' alternate, 'empty' carries no bytes.
    schedule = [("a", 64), ("b", 32), ("a", 64), ("empty", 0), ("b", 32), ("a", 64), ("empty", 0)]
    sent, read = [], []
    send_buffers, recv_buffers = set(), set()
    for round_, (schema, size) in enumerate(schedule, 1):
        creates = bool(size) and schema not in send_buffers          # the sender makes its buffer: an fd crosses
        if creates:
            send_buffers.add(schema)
        if transport._header_due(announced, 1, schema, creates):
            sent.append(round_)
        lacks = bool(size) and schema not in recv_buffers             # the receiver has no buffer mapped yet
        if transport._header_due(heard, 0, schema, lacks):
            read.append(round_)
            if size:
                recv_buffers.add(schema)
    assert sent == read == [1, 2, 4]


def test_mpas_route_rank0_aborts_its_halo_transport_when_a_child_fails():
    source = inspect.getsource(process_ranks_v842.run_process_ranks)
    assert "rank0.get('transport')" in source and "holder=rank0" in source
