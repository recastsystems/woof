"""Prevent AMD topology from inheriting the NVIDIA PCIe NUMA workaround."""
from tilestream.ranks import choose_transports


def test_xgmi_crosses_cpu_numa_without_host_staging():
    cards = list(range(8))
    peers = {(a, b): True for a in cards for b in cards if a != b}
    links = {pair: {'type': 'xgmi', 'hops': 1} for pair in peers}
    nodes = {a: a // 4 for a in cards}
    cuda = choose_transports(cards, 'auto', peers, nodes)
    hip = choose_transports(cards, 'auto', peers, nodes, links)
    assert {v for pair, v in cuda.items() if pair[0] != pair[1]} == {'staged'}
    assert {v for pair, v in hip.items() if pair[0] != pair[1]} == {'peer'}


def test_unknown_link_or_missing_reverse_peer_uses_pinned_host():
    peers = {(0, 1): True, (1, 0): True}
    paths = choose_transports([0, 1], 'auto', peers, {0: 0, 1: 1}, {})
    assert paths[0, 1] == paths[1, 0] == 'host'
    paths = choose_transports([0, 1], 'auto', {(0, 1): True},
                              links={(0, 1): {'type': 'xgmi'}})
    assert paths[0, 1] == paths[1, 0] == 'host'


def test_prebuilt_target_matches_device_family(monkeypatch):
    from types import SimpleNamespace
    import pytest
    from woof.hip_backend import architecture
    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(
        is_hip=True, getDevice=lambda: 0,
        getDeviceProperties=lambda _: {'gcnArchName': b'gfx942:sramecc+:xnack-'})))
    monkeypatch.delenv('WOOF_HIP_ARCH', raising=False)
    assert architecture(cp) == 'gfx942:sramecc+:xnack-'
    monkeypatch.setenv('WOOF_HIP_ARCH', 'gfx942')
    assert architecture(cp) == 'gfx942'
    monkeypatch.setenv('WOOF_HIP_ARCH', 'gfx90a')
    with pytest.raises(ValueError, match='does not match device'):
        architecture(cp)
