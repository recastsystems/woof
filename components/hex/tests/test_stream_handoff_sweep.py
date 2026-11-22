"""Dead runs' tmpfs stream handoff folders are swept; live and unnamed ones are left (hex 0.4.15, 20331da).

THE BREAKAGE THIS PINS (global16 hex-fit16, verification host, 2026-11-06): killed 60 km runs left 1.46 GB each in
/dev/shm (tmpfs is RAM), which fed the host out-of-memory kills of the next runs on a 30 GB host.
"""
import os
import subprocess
import sys

from woof.hex import restart_v842


def test_sweep_removes_dead_owner_folders_only(tmp_path):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    gone = tmp_path / f".mpas-handoff-p{dead.pid}-abc"
    live = tmp_path / f".mpas-handoff-p{os.getppid()}-def"
    mine = tmp_path / f".mpas-handoff-p{os.getpid()}-ghi"
    unnamed = tmp_path / ".mpas-handoff-xyz"
    for folder in (gone, live, mine, unnamed):
        folder.mkdir()
        (folder / "words").write_bytes(b"0" * 16)
    restart_v842._SWEPT_HANDOFF_ROOTS.discard(str(tmp_path))
    restart_v842._sweep_dead_handoffs(tmp_path)
    assert not gone.exists()
    assert live.exists() and mine.exists() and unnamed.exists()
