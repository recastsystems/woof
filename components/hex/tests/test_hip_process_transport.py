from types import SimpleNamespace
import numpy as np
import pytest

from woof.hex.process_transport import broadcast_tree


def table(*peers):
    return SimpleNamespace(peers=[SimpleNamespace(peer=p, cell_send=np.array([0]),
        cell_recv=np.array([0]), edge_send=np.array([],dtype=int),
        edge_recv=np.array([],dtype=int)) for p in peers])


def test_bootstrap_tree_reaches_all_eight_ranks_without_non_neighbour_link():
    tables = [table(1), table(0, 2), table(1, 3), table(2, 4),
              table(3, 5), table(4, 6), table(5, 7), table(6)]
    assert broadcast_tree(tables) == {0: None, **{i: i-1 for i in range(1, 8)}}


def test_disconnected_bootstrap_fails_before_any_socket_wait():
    with pytest.raises(ValueError, match='connected'):
        broadcast_tree([table(1), table(0), table()])


def test_amd_never_loads_the_nvidia_collective_library(monkeypatch):
    from woof import gpu_smi
    from woof.hex import nccl_runtime
    monkeypatch.setattr(gpu_smi, 'is_amd', lambda: True)
    monkeypatch.setenv('WOOF_HEX_NCCL_LIBRARY', '/wrong/libnccl.so')
    monkeypatch.setenv('ROCM_PATH', '/private/rocm')
    candidates = nccl_runtime._candidates()
    assert candidates[0] == '/private/rocm/lib/librccl.so.1'
    assert all('nccl' not in p for p in candidates)


def test_nccl_transport_imports_against_release_halo_tables():
    from woof.hex.partition_nccl_transport_v841 import NcclHaloTransport
    from woof.hex.partition_samehost_tables_v841 import active_peers
    halo = table(1, 2)
    for kind in ('cell_send', 'cell_recv', 'edge_send', 'edge_recv'):
        setattr(halo.peers[1], kind, np.array([], dtype=int))
    assert active_peers(halo) == (1,)
    assert callable(NcclHaloTransport)


def test_eight_xgmi_cards_keep_direct_peer_routes_across_numa():
    from woof.hex.partition_samehost_transport_v841 import choose_copy_paths
    cards = list(range(8))
    peers = {(a,b): True for a in cards for b in cards if a != b}
    nodes = {card: card // 4 for card in cards}
    links = {pair: {'type': 'xgmi'} for pair in peers}
    cuda = choose_copy_paths(cards, peers, numa_nodes=nodes)
    amd = choose_copy_paths(cards, peers, numa_nodes=nodes, links=links)
    assert cuda[0,4] == 'staged'
    assert {value for pair,value in amd.items() if pair[0] != pair[1]} == {'peer'}
