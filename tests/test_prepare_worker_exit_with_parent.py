"""Boundary-preparation workers die with their parent.

Breakage this prevents (Updraft factory, a rented 8-card box, 2026-10-27): a killed HRRR
preparation left its spawn-pool boundary workers blocked on the executor's call
queue for 8.5 h, each holding an 866 MiB CUDA context the card pool could not see.
The test is that exact shape: the real ``_prepare_worker_init`` pool, parent
SIGKILLed while its workers are busy.
"""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PARENT = textwrap.dedent("""
    import multiprocessing, sys, time
    from concurrent.futures import ProcessPoolExecutor
    import tools.hrrr_single_domain_benchmark as h

    if __name__ == "__main__":
        ex = ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn"),
            initializer=h._prepare_worker_init,
            initargs=({}, {}, (1.0, 0.0), 2000.0, 5, None, None))
        futures = [ex.submit(time.sleep, 600) for _ in range(2)]
        while len(ex._processes) < 2:
            time.sleep(0.1)
        time.sleep(3)
        print(" ".join(str(pid) for pid in ex._processes), flush=True)
        time.sleep(600)
""")


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
    except OSError:
        return True
    return state != "Z"


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL of the parent is the POSIX failure shape")
def test_workers_exit_when_parent_is_killed(tmp_path):
    script = tmp_path / "parent.py"
    script.write_text(PARENT)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(ROOT)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    parent = subprocess.Popen([sys.executable, str(script)], cwd=tmp_path, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    workers = []
    try:
        line = parent.stdout.readline()
        assert line.strip(), parent.stderr.read()
        workers = [int(pid) for pid in line.split()]
        assert len(workers) == 2 and all(_alive(pid) for pid in workers)
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=30)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and any(_alive(pid) for pid in workers):
            time.sleep(0.2)
        assert not any(_alive(pid) for pid in workers), f"orphaned workers survived: {workers}"
    finally:
        if parent.poll() is None:
            parent.kill()
        for pid in workers:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
