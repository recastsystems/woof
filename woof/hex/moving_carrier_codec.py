"""Bounded parallel lossless storage of owned moving-boundary payloads.

Protocol 5 retains array dtype, order, aliases and mutability. Its buffers are
compressed in independent 8 MiB blocks with standard-library zlib, without
changing any floating-point value or omitting restart state. The caller seals
and verifies the complete file hash before loading this trusted pickle.
"""
from concurrent.futures import ThreadPoolExecutor
import pickle
import struct
import zlib

from .host_parallel import threads

MAGIC = b"HXCPZ001"
BLOCK = 8 << 20
U64 = struct.Struct("<Q")


def dump(payload, stream):
    """Write a carrier. The compressed blocks are produced on all cores and written in file order.

    The block size, zlib level and order are unchanged, so the file bytes do not depend on the worker
    count. THE COST THIS REMOVES: eight workers in lockstep batches (compress a batch, then write it)
    left the export of a 3.5 GB carrier at ~1-2 cores for ~20 s at every move (a two-card cloud machine, 2026-10-24).
    """
    from collections import deque
    buffers = []
    encoded = pickle.dumps(payload, protocol=5, buffer_callback=buffers.append)
    stream.write(MAGIC + U64.pack(len(encoded)) + U64.pack(len(buffers)))
    stream.write(encoded)
    workers = threads()
    raw_bytes = len(encoded)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hex-carrier") as pool:
        inflight = deque()

        def drain(limit):
            while len(inflight) > limit:
                compressed = inflight.popleft().result()
                stream.write(U64.pack(len(compressed)))
                stream.write(compressed)
        for buffer in buffers:
            view = buffer.raw()
            raw_bytes += len(view)
            drain(0)  # every block of the previous buffer is written before this buffer's length
            stream.write(U64.pack(len(view)))
            for start in range(0, len(view), BLOCK):
                inflight.append(pool.submit(_compress, view[start:start + BLOCK]))
                drain(2 * workers)
        drain(0)
    return dict(codec="protocol5-zlib-blocks-v1", buffers=len(buffers),
                raw_bytes=raw_bytes, block_bytes=BLOCK, workers=workers)


def _compress(block):
    return zlib.compress(block, level=1)


def _read(stream, length):
    data = stream.read(length)
    if len(data) != length:
        raise ValueError("truncated moving carrier")
    return data


def _size(stream):
    return U64.unpack(_read(stream, U64.size))[0]


def _inflate(item):
    compressed, target = item
    decoder = zlib.decompressobj()
    data = decoder.decompress(compressed, len(target) + 1)
    if (len(data) != len(target) or not decoder.eof
            or decoder.unused_data or decoder.unconsumed_tail):
        raise ValueError("invalid moving carrier block")
    target[:] = data


def load(stream, *, before_unpickle=None):
    """Decode a carrier; ``before_unpickle()`` (the caller's hash check) runs before any unpickling.

    The buffers are inflated into uninitialised memory on all cores (no serial zero fill of a
    multi-GB bytearray first) and stay writable, as protocol 5 buffers were; the words are the same.
    """
    if stream.read(len(MAGIC)) != MAGIC:
        if before_unpickle is not None:
            before_unpickle()
        stream.seek(0)
        return pickle.load(stream)  # legacy owned carrier
    import numpy as np
    from collections import deque
    try:
        metadata_bytes, count = _size(stream), _size(stream)
        encoded = _read(stream, metadata_bytes)
        buffers = []
        workers = threads()
        # Reading the next compressed block overlaps the inflation of the ones in flight (bounded queue).
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hex-carrier") as pool:
            inflight = deque()
            for _ in range(count):
                buffer = np.empty(_size(stream), dtype=np.uint8)
                view = memoryview(buffer)
                for start in range(0, len(view), BLOCK):
                    inflight.append(pool.submit(_inflate, (_read(stream, _size(stream)), view[start:start + BLOCK])))
                    while len(inflight) > 2 * workers:
                        inflight.popleft().result()
                buffers.append(buffer)
            while inflight:
                inflight.popleft().result()
        if stream.read(1):
            raise ValueError("trailing bytes in moving carrier")
    except Exception:
        # A damaged file is reported as the receipt mismatch it is, when the caller can check.
        if before_unpickle is not None:
            before_unpickle()
        raise
    if before_unpickle is not None:
        before_unpickle()
    return pickle.loads(encoded, buffers=buffers)
