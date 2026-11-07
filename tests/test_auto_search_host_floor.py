"""Impossible host stores must not multiply the automatic subset search."""
from dataclasses import replace
import tomllib

import pytest

from woof.branch import emit_experiment_toml
from woof.experiment import load_experiment
from woof.core import preflight as pf, streaming as st
from tilestream import autoplan as ap
from test_check_nested_mixed_road import _nested_auto_tiles

GIB = 1024 ** 3


def _experiment(tmp_path, domains=4, *, host_gib=2, write_mode="ring",
                root_n=300, child_n=600, root_off=False):
    raw = tomllib.loads(_nested_auto_tiles(tmp_path).read_text(encoding="utf-8"))
    raw["domain"][0].update(nx=root_n, ny=root_n)
    raw["domain"][1].update(nx=child_n, ny=child_n)
    raw["domain"] = [raw["domain"][0]] + [
        dict(raw["domain"][1], grid_id=gid) for gid in range(2, domains + 1)]
    if host_gib is not None:
        raw["tiles"]["host_budget_bytes"] = int(host_gib * GIB)
    raw["tiles"]["write_mode"] = write_mode
    if root_off:
        raw["domain"][0]["tiles"] = {"mode": "off"}
    path = tmp_path / "validated.toml"
    path.write_text(emit_experiment_toml(raw), encoding="utf-8")
    return load_experiment(path)


def _walk(exp, free_gib=24):
    estimate = pf.estimate_experiment(exp)
    nodes = st._config_tree_nodes(exp.domains)
    machine = ap.Machine(int(free_gib * GIB), 256 * GIB)
    rows = {}
    result = st.decide_tree(nodes, exp.tiles, machine=machine, decisions=rows,
                           resident_estimate=estimate)
    return result, rows


@pytest.mark.parametrize("domains", [4, 8, 12, 16])
def test_host_impossible_choices_terminate_after_one_real_walk(tmp_path, monkeypatch, domains):
    exp = _experiment(tmp_path, domains)
    estimate = pf.estimate_experiment(exp)
    nodes = st._config_tree_nodes(exp.domains)
    budget = int(23.5 * GIB)
    # The existing global/process floor pruning cannot decide this case.
    assert st._resident_subset_envelope(estimate, nodes, set()) < budget
    assert st._tree_process_overhead_bytes(nodes) + st._tree_radiation_transient_bytes(nodes) < budget
    child_floor = st.radiation_footprint(exp.domains[1].run, exp.tiles).store_bytes(600 * 600 * 49)
    assert child_floor > exp.tiles.host_budget_bytes
    original = st._decide_tree
    calls = []
    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(st, "_decide_tree", counted)
    with pytest.raises(st.StreamingRefused, match="must remain resident") as error:
        _walk(exp)
    assert "host allowance" in str(error.value)
    assert "required resident envelope" in str(error.value)
    assert len(calls) == 1  # Twelve domains previously repeated 4,097 real walks.


def test_host_impossible_streaming_can_keep_that_domain_resident(tmp_path):
    exp = _experiment(tmp_path, 2, root_n=704, child_n=63)
    root_floor = st.radiation_footprint(exp.root.run, exp.tiles).store_bytes(704 * 704 * 49)
    assert root_floor > exp.tiles.host_budget_bytes
    # The 63x63 child's every tiling does more than 4x the necessary work.
    # Auto keeps the redundancy limit (since the 1,190-tile road of
    # 2026-09-26), so with the root held resident by its host store there
    # is no streamed road left, and the card's 21.75 GiB holds the tree: it
    # runs resident inside the external margin and says so.  (22 GiB until
    # A163: the measured forecast margin is 1.13 of the subtotal, not the
    # plan's 1.15, which prices this 15.71 GiB-subtotal tree at 21.59 GiB,
    # and at 22 GiB that fits outright.)  RULINGS 1 (history CLDFRA, 2.8.7)
    # prices each domain's held radiation cloud-fraction buffer, which
    # moves the resident tree to 21.81 GiB: the card here is 22 GiB, whose
    # 21.5 GiB admission budget the tree still exceeds while fitting the
    # card, the same inside-the-margin case as before.  The WRF exact chain's
    # history rows (2.9.1: P_HYD and RTHRATEN planes, eight surface planes)
    # move it to 22.15 GiB: the card is 22.5 GiB, whose 22 GiB admission
    # budget the tree exceeds while fitting the card.  The speed2 held-storage
    # sharing (c563a6f8860) moves it to 21.85 GiB: the card is 22.25 GiB,
    # whose 21.75 GiB admission budget the tree exceeds while fitting the
    # card (read at 22.0, 22.25, 22.3 and 22.4 GiB at 3d4bfffd093; from
    # 22.4 GiB the envelope fits the budget outright).  97e9a3ca061
    # (cap-hail): automatic HAILCAST on the 3-km child loads its 7,392-B
    # frame, the widest in the process, and its diagnostic slots, moving the
    # tree to 22.41 GiB: the card is 22.75 GiB, whose 22.25 GiB admission
    # budget the tree exceeds while fitting the card (from 23.0 GiB it fits
    # the budget outright).  The NWP diagnostics port (9ebcd8d0ac6, fitted
    # in fix/seal4-fit) prices its time-t density and output windows on the
    # child, moving the tree to 22.55 GiB; the auto walk keeps its 22.75 GiB
    # card (resident inside the margin), and the knob walk moves from 22.25
    # to 22.5 GiB, whose 22.0 GiB budget the 21.97 GiB mixed road fits
    # (measured 22.25: refused; 22.5 to 23.0: this road; 23.25: resident).
    assert st._inbound_stream_tiling(st._config_tree_nodes(exp.domains)[1],
                                     exp.tiles) is None
    result, rows = _walk(exp, free_gib=22.75)
    assert not rows[1].stream and not rows[2].stream
    assert "inside the" in rows[2].reason and "redundancy limit" in rows[2].reason
    # The road this test is about -- the host-blocked root resident beside
    # a streamed child -- is the one the explicit knob asks for by name.
    exp = replace(exp, tiles=replace(exp.tiles, max_redundancy=False))
    result, rows = _walk(exp, free_gib=22.5)
    assert not rows[1].stream and rows[2].stream
    assert result.configured_mixed_envelope_bytes <= int(22.0 * GIB)
    assert result.host_spent_bytes <= exp.tiles.host_budget_bytes


def test_two_shadow_stores_are_required_while_ring_alternative_remains(tmp_path):
    # 23.5 GiB free: this tree prices 23.27 GiB resident at A163's measured
    # 1.13 forecast margin, so on the 24 GiB this test used before it no
    # longer needs a streamed road at all.
    ring = _experiment(tmp_path, 3, host_gib=6, root_off=True)
    result, rows = _walk(ring, free_gib=23.5)
    assert not rows[1].stream and any(row.stream for row in rows.values())
    assert result.host_spent_bytes <= 6 * GIB
    shadow = replace(ring, tiles=replace(ring.tiles, write_mode="shadow"))
    with pytest.raises(st.StreamingRefused, match="must remain resident"):
        _walk(shadow, free_gib=23.5)


def test_ample_host_preserves_the_existing_fitting_alternative(tmp_path):
    exp = _experiment(tmp_path, 8, host_gib=None)
    # 24.5 GiB since the NWP diagnostics port priced its child cost (fixed in
    # fix/seal4-fit): at 24 GiB the second domain streams too (measured 24 and
    # 24.25: [F, T, ...]; 24.5 and 24.75: these roads).
    result, rows = _walk(exp, free_gib=24.5)
    # Same roads as the committed-before control on this public configuration.
    assert [row.stream for row in rows.values()] == [False, False, True, True, True, True, True, True]
    assert result.configured_mixed_envelope_bytes <= int(24.0 * GIB)
    assert result.host_spent_bytes <= result.host_budget_bytes


def test_all_resident_fit_needs_no_streamed_host_allowance(tmp_path):
    exp = _experiment(tmp_path, 2, host_gib=0.001)
    result, rows = _walk(exp, free_gib=64)
    assert not any(row.stream for row in rows.values())
    assert result.host_spent_bytes == 0