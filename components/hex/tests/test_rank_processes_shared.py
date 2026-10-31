"""SharedObjects: the global source mesh reaches every child rank once, with the same words (woof.hex.rank_processes).

THE COST IT REMOVES (NOAA RRFS v2 conus3km, 8 rank processes, V8X6000, 2026-10-29): every child request carried the
whole global mesh (29.0 of 30.5 GB), copied into 7 memfds while the cards waited, and every child held its own copy.
These check that a request sent with a shared reference unpickles to the same values in a separate process, that a
child's write stays in that child, and that a request without the shared object is unchanged.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import rank_processes as rp

pytestmark = pytest.mark.skipif(not rp.supported()[0], reason="needs Linux memfd and fd passing")

CHILD = """
import json, sys
import numpy as np
from woof.hex import rank_processes as rp
view = rp.SharedObjectsView(int(sys.argv[1]))
source = view.resolve('source')
source.big[0] = -1.0          # copy-on-write: stays in this process
again = view.resolve('source')
print(json.dumps(dict(big=source.big[1:6].tolist(), small=source.small.tolist(), name=source.name,
                      same_object=again is source, nbytes=int(source.big.nbytes))))
"""


def _source():
    rng = np.random.default_rng(7)
    return SimpleNamespace(big=rng.standard_normal(3 << 18).astype(np.float32),   # 3 MiB: out of band
                           small=np.arange(5, dtype=np.int64), name='global')


def test_child_maps_the_same_words_and_writes_stay_private():
    source = _source()
    shared = rp.SharedObjects({'source': source})
    try:
        result = subprocess.run([sys.executable, '-c', CHILD, str(shared.fd)], pass_fds=(shared.fd,),
                                capture_output=True, text=True, timeout=120)
    finally:
        fd = shared.fd
        shared.close()
    assert result.returncode == 0, result.stderr
    got = json.loads(result.stdout)
    assert got['big'] == source.big[1:6].tolist()
    assert got['small'] == [0, 1, 2, 3, 4] and got['name'] == 'global' and got['same_object']
    assert got['nbytes'] == source.big.nbytes
    assert source.big[0] != -1.0
    assert fd >= 0 and shared.fd == -1


def test_request_carries_a_reference_and_resolves_to_the_shared_object():
    source = _source()
    local = SimpleNamespace(partition_source_mesh=source, own=np.ones(1 << 19, np.float32))
    shared = rp.SharedObjects({'source': source})
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sender, receiver = rp.Link(a, timeout=20, name='0-1'), rp.Link(b, timeout=20, name='1-0')
    try:
        sender.send(dict(local=local), shared.persistent_id)
        view = rp.SharedObjectsView(os.dup(shared.fd))   # the view closes its fd once mapped
        received = receiver.recv(view.resolve)
        # The source's 3 MiB did not travel in the message; the local array did.
        assert sender.sent_bytes < source.big.nbytes + local.own.nbytes
        mesh = received['local'].partition_source_mesh
        assert np.array_equal(mesh.big.view(np.uint32), source.big.view(np.uint32))
        assert np.array_equal(received['local'].own, local.own)
    finally:
        shared.close()
        sender.close()
        receiver.close()


def test_plain_send_is_unchanged():
    source = _source()
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sender, receiver = rp.Link(a, timeout=20, name='0-1'), rp.Link(b, timeout=20, name='1-0')
    try:
        sender.send(dict(source=source))
        received = receiver.recv()
        assert np.array_equal(received['source'].big, source.big)
    finally:
        sender.close()
        receiver.close()
