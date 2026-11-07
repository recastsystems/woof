"""Periodic Python stacks without the native faulthandler timer's frame walk.

The CPython 3.12 native timer crashed in dump_frame while a moving forecast
created and retired worker threads. sys._current_frames() gives this Python
sampler owned frame references, so formatting cannot follow freed frames.
Fatal-signal reporting and external native debuggers remain independent.
"""
from __future__ import annotations

import sys
import threading
import traceback


def dump_stacks(stream=None):
    """Retain the frame snapshot until all stacks have been formatted."""
    stream = sys.stderr if stream is None else stream
    frames = sys._current_frames()
    blocks = ['Periodic Python stack snapshot:\n']
    for ident, frame in sorted(frames.items()):
        blocks.append(f'Thread 0x{ident:x} (most recent call last):\n')
        blocks.extend(traceback.format_stack(frame))
    stream.write(''.join(blocks))
    stream.flush()


def start(interval_seconds=180.0, *, stream=None):
    """Start a daemon sampler; return its stop event and thread for shutdown."""
    if interval_seconds <= 0:
        raise ValueError('stack sampling interval must be positive')
    stop = threading.Event()

    def sample():
        while not stop.wait(interval_seconds):
            dump_stacks(stream)

    thread = threading.Thread(target=sample, name='hex-stack-watch', daemon=True)
    thread.start()
    return stop, thread
