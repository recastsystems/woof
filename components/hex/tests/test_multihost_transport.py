"""Exercise the recovered network buffer route and AMD launch environment."""
from concurrent.futures import ThreadPoolExecutor
import socket
import sys
from types import SimpleNamespace

import pytest

import numpy as np

from woof.hex import multihost
from woof.hex.rank_processes import Link


@pytest.mark.parametrize('joined', (False, True))
def test_joined_host_owns_output_scratch_and_measurement(tmp_path, monkeypatch, joined):
    original = [sys.executable, '-m', 'woof.hex.drivers.forecast_receipt']
    for option in ('--report', '--scratch', '--out'):
        original += ([option+'='+str(tmp_path/option[2:])] if joined
                     else [option, str(tmp_path/option[2:])])
    monkeypatch.setattr(sys, 'orig_argv', original)
    session = SimpleNamespace(out=tmp_path/'out')
    command = multihost.LaunchSession._joined_argv(session, 1)
    host = tmp_path/'out/.multihost/host1'
    for option, target in (('--out', host), ('--scratch', host/'scratch'),
                           ('--report', host/'measurement.json')):
        if joined:
            assert option+'='+str(target) in command
        else:
            assert command[command.index(option)+1] == str(target)
    assert sys.orig_argv == original


def test_tcp_large_buffers_preserve_every_bit():
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0)); listener.listen(1)
    client = socket.create_connection(listener.getsockname())
    server, _ = listener.accept(); listener.close()
    sender, receiver = Link(client, timeout=20), Link(server, timeout=20)
    words = np.arange(1 << 20, dtype=np.uint32)
    words[:4] = [0, 0x80000000, 0x7fc00001, 0xff800000]
    floats = words.view(np.float32)
    try:
        assert sender.inband and receiver.inband
        with ThreadPoolExecutor(1) as pool:
            sent = pool.submit(sender.send, {'state': floats, 'clock': (7, 120)})
            received = receiver.recv(); sent.result()
        assert received['clock'] == (7, 120)
        assert received['state'].tobytes() == floats.tobytes()
    finally:
        sender.close(); receiver.close()


def test_amd_runtime_environment_crosses_hosts_without_card_masks():
    for name in ('RCCL_', 'HIP_', 'HSA_', 'ROCM_', 'ROCR_'):
        assert name in multihost._FORWARD_PREFIXES
    for name in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
        assert name in multihost._NOT_FORWARDED
    assert 'HIPCC' in multihost._FORWARD_NAMES
