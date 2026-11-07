# SPDX-License-Identifier: Apache-2.0
"""Select a process halo transport without changing CUDA's default route.

ROCm's stream-wait-value API requires hipMallocSignalMemory, whereas the
CUDA process route waits on registered mmap words. Use RCCL's stream-ordered
send/recv on AMD instead of passing unsupported mapped words to HIP. The
pack/unpack kernels and packet schemas are shared with the CUDA transport.
"""
from __future__ import annotations

import os


def broadcast_tree(tables):
    """A deterministic spanning tree using only existing halo socket links."""
    from .partition_samehost_tables_v841 import active_peers
    parents = {0: None}
    pending = [0]
    for rank in pending:
        for peer in active_peers(tables[rank]):
            if peer not in parents:
                parents[peer] = rank
                pending.append(peer)
    if len(parents) != len(tables):
        raise ValueError('RCCL bootstrap requires a connected halo graph')
    return parents


def process_halo_transport(tables, devices, rank, links, *, xp, timeout=1800., **options):
    requested = os.environ.get('WOOF_HEX_PROCESS_TRANSPORT', 'auto').lower()
    if requested not in ('auto', 'shared', 'nccl', 'rccl'):
        raise ValueError(f'unknown WOOF_HEX_PROCESS_TRANSPORT {requested!r}')
    amd = bool(xp.cuda.runtime.is_hip)
    if requested in ('auto', 'shared') and not amd:
        from .partition_process_transport_v841 import ProcessHaloTransport
        return ProcessHaloTransport(tables, devices, rank, links, xp=xp, timeout=timeout, **options)
    if requested == 'shared':
        raise ValueError('HIP mapped host flags are not stream signal memory; use the default RCCL process transport')
    from . import nccl_runtime
    from .partition_nccl_transport_v841 import NcclHaloTransport
    parents = broadcast_tree(tables)
    try:
        for sock, _fd in links.values():
            sock.settimeout(timeout)
        if rank == 0:
            uid = nccl_runtime.unique_id()
        else:
            sock = links[parents[rank]][0]
            uid = bytearray()
            while len(uid) < nccl_runtime.UNIQUE_ID_BYTES:
                chunk = sock.recv(nccl_runtime.UNIQUE_ID_BYTES - len(uid))
                if not chunk:
                    raise RuntimeError('a halo peer stopped during RCCL bootstrap')
                uid.extend(chunk)
            uid = bytes(uid)
        for child, parent in parents.items():
            if parent == rank:
                links[child][0].sendall(uid)
        with xp.cuda.Device(devices[rank]):
            communicator = nccl_runtime.NcclCommunicator(len(tables), uid, rank, timeout=timeout)
            try:
                return NcclHaloTransport(tables, devices, rank, communicator, xp=xp, timeout=timeout, **options)
            except BaseException:
                communicator.abort('halo construction failed')
                communicator.destroy()
                raise
    finally:
        # The existing supervisor created these fds; after bootstrap only
        # NCCL/RCCL owns the halo data path. The separate control links stay.
        for sock, fd in links.values():
            sock.close()
            os.close(fd)
