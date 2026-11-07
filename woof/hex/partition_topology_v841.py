"""Place unchanged METIS ranks on measured PCIe links using integer halo counts.

The objective is lexicographic: minimize cross-socket traffic first, then
traffic outside a switch pair. No field storage, partition ownership, local
entity order, or numerical operation is changed. Unknown links remain unknown.
"""
from __future__ import annotations

import itertools
import os
import re
import subprocess
from typing import Any, Sequence


def _bus(value: str) -> str:
    parts = value.strip().lower().split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid PCI bus address {value!r}")
    return f"{int(parts[0], 16):04x}:{parts[1]}:{parts[2]}"


def parse_topology(text: str, inventory: str, buses: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Map physical nvidia-smi rows to CUDA-visible devices by PCI address."""
    physical = {}
    for line in inventory.splitlines():
        index, bus = line.split(",", 1)
        physical[_bus(bus)] = f"GPU{int(index.strip())}"
    labels = [physical[_bus(bus)] for bus in buses]
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    lines = [line.split() for line in clean.splitlines()]
    header = next(row for row in lines if row and row[0] == "GPU0")
    columns = [token for token in header if re.fullmatch(r"GPU\d+", token)]
    rows = {row[0]: row[1:1 + len(columns)] for row in lines
            if len(row) > len(columns) and re.fullmatch(r"GPU\d+", row[0])}
    result = tuple(tuple(rows[src][columns.index(dst)] for dst in labels) for src in labels)
    allowed = {"X", "PIX", "PXB", "PHB", "NODE", "SYS"}
    for i, row in enumerate(result):
        for j, link in enumerate(row):
            if link not in allowed and not re.fullmatch(r"NV\d+", link):
                raise ValueError(f"unrecognized GPU topology link {link!r}")
            if link != result[j][i] or (i == j and link != "X"):
                raise ValueError("GPU topology is not symmetric with a self-link diagonal")
    return result


def discover_topology(devices: Sequence[int]) -> tuple[tuple[tuple[str, ...], ...] | None, str]:
    """Read topology only; no CUDA allocation or peer enable takes place here."""
    if any(os.environ.get(name, "") not in ("", "0")
           for name in ("WOOF_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU")):
        return None, "local GPU disabled; retain rank order"
    try:
        import cupy as cp
        buses = [cp.cuda.Device(device).pci_bus_id for device in devices]
        def query(arguments):
            return subprocess.run([*__import__('woof.gpu_smi', fromlist=['command']).command(), *arguments], capture_output=True,
                                  text=True, check=True, timeout=15).stdout
        links = parse_topology(query(["topo", "-m"]),
                               query(["--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"]), buses)
        return links, "nvidia-smi topology mapped through CUDA-visible PCI addresses"
    except (ImportError, OSError, subprocess.SubprocessError, RuntimeError, ValueError, KeyError, StopIteration, IndexError) as error:
        return None, f"topology unavailable ({type(error).__name__}); retain rank order"


def place_ranks(tables: Sequence[Any], devices: Sequence[int], links=None) -> dict[str, Any]:
    """Find the exact best rank placement for up to eight distinct cards.

Each undirected weight is the two directional cell and edge halo index
counts. The receipt calls these entity transfers, not bytes or bandwidth.
Every field-specific byte count still comes from the actual transport.
"""
    devices = tuple(map(int, devices))
    n = len(tables)
    if n != len(devices) or not 1 <= n <= 8 or len(set(devices)) != n:
        raise ValueError("topology placement needs 1..8 distinct cards and one table per rank; "
                         "prevents attaching a halo plan to an absent or duplicate card")
    basis = "supplied topology"
    if links is None:
        links, basis = discover_topology(devices)
    identity = tuple(range(n))
    if links is None:
        return {"devices": list(devices), "rank_to_slot": list(identity), "basis": basis,
                "objective": None, "baseline_objective": None, "qualified_speedup": False}
    links = tuple(tuple(row) for row in links)
    if len(links) != n or any(len(row) != n for row in links):
        raise ValueError("topology dimensions do not match the card grant; prevents rank/card misrouting")
    tiers = []
    for i, row in enumerate(links):
        tier = []
        for j, link in enumerate(row):
            if row[j] != links[j][i] or (i == j and link != "X"):
                raise ValueError("topology must be symmetric with self links; prevents invalid card placement")
            if link in ("X", "PIX", "PXB") or re.fullmatch(r"NV\d+", link):
                tier.append((0, 0))
            elif link in ("PHB", "NODE"):
                tier.append((0, 1))
            elif link == "SYS":
                tier.append((1, 1))
            else:
                raise ValueError(f"unknown topology link {link!r}; prevents guessing cross-socket routing")
        tiers.append(tier)
    weights = [[0] * n for _ in range(n)]
    for rank, table in enumerate(tables):
        if table.rank != rank:
            raise ValueError("halo tables are not in rank order; prevents assigning the wrong mesh to a card")
        for peer in table.peers:
            other = int(peer.peer)
            if not 0 <= other < n or other == rank:
                raise ValueError("invalid halo peer; prevents nonexistent or self rank transfers")
            weights[min(rank, other)][max(rank, other)] += len(peer.cell_send) + len(peer.edge_send)
    edges = [(i, j, weights[i][j]) for i in range(n) for j in range(i + 1, n) if weights[i][j]]
    def objective(order):
        cross = outside = 0
        for i, j, weight in edges:
            a, b = tiers[order[i]][order[j]]
            cross += weight * a
            outside += weight * b
        return cross, outside
    # Keeping identity on ties avoids relabeling an already optimal cached run.
    best = identity
    baseline = score = objective(identity)
    for order in itertools.permutations(identity):
        candidate = objective(order)
        if candidate < score:
            best, score = order, candidate
    return {"devices": [devices[slot] for slot in best], "rank_to_slot": list(best),
            "basis": basis, "links": [list(row) for row in links],
            "objective": list(score), "baseline_objective": list(baseline),
            "objective_units": "cross-socket, outside-switch-pair entity transfers",
            "traffic": weights, "search": "all rank/card permutations, identity wins ties",
            "qualified_speedup": False}
