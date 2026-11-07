#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise real peer copies, stream/event ordering, pinned DMA and NCCL/RCCL.

The 8x MI300X admission is explicit; --hardware nvidia is a translation
proof and never counts as an AMD hardware result.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import mmap
from pathlib import Path
import time

import numpy as np


def host_api(cp):
    from woof import hip_runtime
    from tilestream.gather import _Memcpy3DParms
    native = bool(cp.cuda.runtime.is_hip)
    # Native bindings are also exercised by the HIP-on-NVIDIA shim.
    import os
    native = native or bool(os.environ.get('WOOF_HIP_RUNTIME_LIBRARY'))
    rt = cp.cuda.runtime
    page = mmap.mmap(-1, 4096)
    address = ctypes.addressof(ctypes.c_char.from_buffer(page))
    register = hip_runtime.host_register if native else rt.hostRegister
    unregister = hip_runtime.host_unregister if native else rt.hostUnregister
    register(address, 4096, 3)
    try:
        if native:
            assert hip_runtime.mapped_pointer(address) != 0
        host = np.ndarray((4, 8, 16), dtype=np.uint32, buffer=page)
        host[:] = np.arange(host.size, dtype=np.uint32).reshape(host.shape)
        device = cp.zeros_like(cp.asarray(host))
        # Nonzero x/y offsets, padded rows, and a three-plane copy catch ABI
        # errors that a contiguous transfer or round trip would conceal.
        p = _Memcpy3DParms()
        p.srcPtr.ptr, p.srcPtr.pitch, p.srcPtr.xsize, p.srcPtr.ysize = address, 64, 16, 8
        p.dstPtr.ptr, p.dstPtr.pitch, p.dstPtr.xsize, p.dstPtr.ysize = device.data.ptr, 64, 16, 8
        p.srcPos.x, p.srcPos.y = 8, 1
        p.dstPos.x, p.dstPos.y = 12, 2
        p.extent.width, p.extent.height, p.extent.depth = 32, 4, 3
        p.kind = rt.memcpyHostToDevice
        stream = cp.cuda.Stream(non_blocking=True)
        hip_runtime.memcpy3d_function(rt)(ctypes.addressof(p), stream.ptr)
        stream.synchronize()
        expected = np.zeros_like(host)
        expected[:3, 2:6, 3:11] = host[:3, 1:5, 2:10]
        assert cp.asnumpy(device).tobytes() == expected.tobytes(), 'pitched-copy ABI/offset mismatch'
    finally:
        unregister(address)
        page.close()
    return dict(registered_mapped_host=True, pitched_copy_exact=True, native_hip_api=native)


def run(cards, hardware, mib=16):
    import woof
    from woof.hip_backend import install
    install()
    import cupy as cp
    from woof.hex import nccl_runtime
    rt = cp.cuda.runtime
    if cards > rt.getDeviceCount():
        raise RuntimeError(f'{cards} cards requested, {rt.getDeviceCount()} visible')
    inventory = []
    for card in range(cards):
        with cp.cuda.Device(card):
            props = rt.getDeviceProperties(card)
            free, total = rt.memGetInfo()
            arch = props.get('gcnArchName', b'').decode() if isinstance(props.get('gcnArchName'), bytes) else str(props.get('gcnArchName', ''))
            row = dict(card=card, name=props['name'].decode(), architecture=arch,
                       wave_size=props['warpSize'], free_bytes=free, total_bytes=total)
            if hardware == 'mi300x':
                assert rt.is_hip and arch.startswith('gfx942'), row
                assert props['warpSize'] == 64 and total >= 180 * (1 << 30), row
            # Probe uses under 64 MiB/card, irrespective of the forecast.
            assert free > 256 * (1 << 20), row
            inventory.append(row)
    links = {}
    if rt.is_hip:
        from woof.hip_runtime import peer_links
        links = peer_links(range(cards))
    peers = []
    size = mib * (1 << 20)
    for source in range(cards):
        for destination in range(cards):
            if source == destination:
                continue
            assert rt.deviceCanAccessPeer(source, destination), (source, destination)
            if hardware == 'mi300x':
                assert links[source, destination]['type'] == 'xgmi', links
            with cp.cuda.Device(source):
                try:
                    rt.deviceEnablePeerAccess(destination)
                except rt.CUDARuntimeError as error:
                    if error.status != 704:
                        raise
                data = cp.arange(size // 4, dtype=cp.uint32)
                produced = cp.cuda.Event(disable_timing=True)
                produced.record()
            with cp.cuda.Device(destination):
                try:
                    rt.deviceEnablePeerAccess(source)
                except rt.CUDARuntimeError as error:
                    if error.status != 704:
                        raise
                target = cp.zeros(size // 4, dtype=cp.uint32)
                stream = cp.cuda.Stream(non_blocking=True)
                stream.wait_event(produced)
                started = cp.cuda.Event()
                ended = cp.cuda.Event()
                started.record(stream)
                for _ in range(8):
                    rt.memcpyPeerAsync(target.data.ptr, destination, data.data.ptr, source, size, stream.ptr)
                ended.record(stream)
                ended.synchronize()
                seconds = cp.cuda.get_elapsed_time(started, ended) / 1000.
                assert cp.asnumpy(target).tobytes() == np.arange(size // 4, dtype=np.uint32).tobytes()
                peers.append(dict(source=source, destination=destination, bytes=size*8,
                                  seconds=seconds, GBps=size*8/seconds/1e9,
                                  exact=True, link=links.get((source, destination))))
    with cp.cuda.Device(0):
        host = host_api(cp)
    uid = nccl_runtime.unique_id()
    def rank(card):
        with cp.cuda.Device(card):
            communicator = nccl_runtime.NcclCommunicator(cards, uid, card, timeout=60.)
            try:
                send = cp.full(4096, card + 1, dtype=cp.int32)
                receive = cp.empty_like(send)
                stream = cp.cuda.get_current_stream()
                began = time.perf_counter()
                communicator.allreduce(send.data.ptr, receive.data.ptr, send.size, 'int32', 'max', stream.ptr)
                stream.synchronize()
                assert bool(cp.all(receive == cards).item()), 'all-reduce max mismatch'
                return dict(rank=card, seconds=time.perf_counter()-began, exact=True)
            finally:
                communicator.destroy()
    with ThreadPoolExecutor(max_workers=cards) as pool:
        collectives = list(pool.map(rank, range(cards)))
    return dict(schema='gpuwm.multicard-probe/v1', hardware=hardware, cards=inventory,
                peers=peers, host_api=host, collective_library=nccl_runtime.library_path(),
                collective_version=nccl_runtime.version(), allreduce=collectives)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cards', type=int, default=8)
    parser.add_argument('--hardware', choices=('mi300x', 'nvidia'), default='mi300x')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cards, args.hardware)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
