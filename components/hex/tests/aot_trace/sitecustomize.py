"""Test-only NVRTC call receipts, activated only inside a granted card hold."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import threading
import time


def install():
    destination = os.environ.get("WOOF_HEX_AOT_TRACE_DIR")
    if not destination or not os.environ.get("CARD_QUEUE_CARDS"):
        return
    from cupy_backends.cuda.libs import nvrtc
    create = nvrtc.createProgram
    compile_program = nvrtc.compileProgram
    destroy = nvrtc.destroyProgram
    programs = {}
    events = []
    lock = threading.RLock()
    def create_program(source, *args, **kwargs):
        program = create(source, *args, **kwargs)
        blob = source.encode() if isinstance(source, str) else bytes(source)
        with lock:
            programs[program] = {"source_sha256": hashlib.sha256(blob).hexdigest(),
                                 "nonempty": bool(blob.strip()), "source_bytes": len(blob)}
        return program
    def compile_receipt(program, options=()):
        started = time.monotonic()
        error = None
        try:
            return compile_program(program, options)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            with lock:
                events.append({**programs.get(program, {"nonempty": None}),
                               "t_start": started, "seconds": time.monotonic() - started,
                               "pid": os.getpid(), "error": error,
                               "options": [x.decode() if isinstance(x, bytes) else str(x) for x in options]})
    def destroy_program(program):
        try:
            return destroy(program)
        finally:
            with lock:
                programs.pop(program, None)
    def save():
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        (root / f"nvrtc-{os.getpid()}.json").write_text(json.dumps(events, indent=2) + "\n")
    nvrtc.createProgram = create_program
    nvrtc.compileProgram = compile_receipt
    nvrtc.destroyProgram = destroy_program
    atexit.register(save)


install()
