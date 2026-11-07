"""The HIP stream-wait mask is an actual fifth ABI argument."""
from types import SimpleNamespace


def test_hip_stream_wait_passes_mask(monkeypatch):
    from woof import gpu_smi, hip_runtime
    from woof.hex.partition_process_transport_v841 import _Driver
    calls = []
    class Function:
        def __call__(self, *args): calls.append(args); return 0
    hip = SimpleNamespace(hipStreamWaitValue32=Function(), hipStreamWriteValue32=Function(),
                          hipHostGetDevicePointer=Function())
    monkeypatch.setattr(gpu_smi, 'is_amd', lambda: True)
    monkeypatch.setattr(hip_runtime, 'library', lambda: hip)
    driver = _Driver()
    driver.wait_geq(SimpleNamespace(ptr=4), 0x1000, 7)
    assert len(hip.hipStreamWaitValue32.argtypes) == 5
    assert len(calls[0]) == 5 and calls[0][-1] == 0xFFFFFFFF


def test_collective_close_precedes_child_acknowledgements(monkeypatch):
    from woof.hex.drivers.samehost_rank_group_v841 import ProcessRankGroup
    import woof.hex.rank_processes as ranks
    events = []
    class Transport:
        collective_close = True
        def close(self): events.append('transport')
    class Link:
        def send(self, value): events.append(value[0])
        def recv(self):
            # A child cannot acknowledge until every communicator closes.
            assert 'transport' in events
            events.append('ack')
            return ('closed', {'rank': 1})
    group = SimpleNamespace(step_links={1: Link()}, rank0_responded=lambda: None,
        join=lambda timeout: {1: 0}, close=lambda: None)
    run = ProcessRankGroup.__new__(ProcessRankGroup)
    run.started, run.failed, run.size = True, False, 2
    run.group, run.ctx = group, SimpleNamespace(transport=Transport())
    monkeypatch.setattr(ranks, 'process_cpu_record', lambda rank: {'rank': rank})
    run.close()
    assert events == ['close', 'transport', 'ack', 'transport']
    assert run.records == [{'rank': 0}, {'rank': 1}]
    assert not run.started
