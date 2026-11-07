"""Run one forecast with an explicit disk limit and asynchronous resource receipts.

Launch this inside the pool lease. No CUDA synchronization is introduced. The
guard observes only its child tree and terminates those exact PIDs on a disk
limit, retaining a failure receipt. The collector is launched separately so it
can finish reducing closed frames after the producer exits.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import psutil


def usage(root):
    size = 0
    for folder, directories, files in os.walk(root, followlinks=False):
        directories[:] = [d for d in directories if not (Path(folder) / d).is_symlink()]
        for name in files:
            path = Path(folder) / name
            try:
                if not path.is_symlink():
                    size += path.stat().st_size
            except FileNotFoundError:
                pass
    return size


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--stop-gib', type=float, default=28,
                   help='stop below the configured hard cap, allowing pending writes')
    p.add_argument('--hard-cap-gb', type=float, default=40,
                   help='decimal GB cap recorded in this run receipt')
    p.add_argument('--max-wall-seconds', type=float, default=None)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not 0 < a.stop_gib * (1 << 30) < a.hard_cap_gb * 1_000_000_000:
        p.error("stop threshold must be positive and below the declared hard cap")
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        p.error('a producer command is required')
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    device_info = subprocess.run(['nvidia-smi',
        '--query-gpu=index,uuid,name,memory.total,driver_version', '--format=csv,noheader'],
        capture_output=True, text=True).stdout.strip().splitlines()
    process = subprocess.Popen(command)
    parent = psutil.Process(process.pid)
    reason, peak_bytes, peak_mib = None, 0, 0
    with a.receipt.with_suffix('.resources.jsonl').open('w') as stream:
        while process.poll() is None:
            try:
                children = parent.children(recursive=True)
            except psutil.NoSuchProcess:
                children = []
            owned = [parent, *children]
            pids = {c.pid for c in owned}
            cpu = []
            for child in owned:
                try:
                    times = child.cpu_times()
                    cpu.append(dict(pid=child.pid, cpu_seconds=times.user+times.system,
                                    rss_bytes=child.memory_info().rss, status=child.status()))
                except psutil.NoSuchProcess:
                    pass
            used = usage(a.root)
            peak_bytes = max(peak_bytes, used)
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                '--format=csv,noheader,nounits'], capture_output=True, text=True)
            vram = 0
            for line in result.stdout.splitlines():
                parts = [v.strip() for v in line.split(',')]
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and int(parts[0]) in pids:
                    vram += int(parts[1])
            peak_mib = max(peak_mib, vram)
            stream.write(json.dumps(dict(elapsed=time.monotonic()-began, bytes=used,
                vram_mib=vram, pids=sorted(pids), processes=cpu))+'\n')
            stream.flush()
            if used >= a.stop_gib * (1 << 30) or (a.max_wall_seconds is not None
                    and time.monotonic()-began >= a.max_wall_seconds):
                reason = ('disk guard reached configured stop threshold'
                          if used >= a.stop_gib * (1 << 30) else 'explicit wall limit')
                for child in reversed(owned):
                    try:
                        child.terminate()  # psutil verifies PID identity before signalling
                    except psutil.NoSuchProcess:
                        pass
                # Popen must reap its own direct child. Letting psutil reap it
                # first makes Popen see ECHILD and can replace SIGTERM with 0.
                try:
                    process.wait(timeout=10)
                    alive = []
                except subprocess.TimeoutExpired:
                    alive = [parent]
                _, descendants_alive = psutil.wait_procs(children, timeout=1)
                alive.extend(descendants_alive)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                break
            try:
                remaining = (20 if a.max_wall_seconds is None else
                             max(0, a.max_wall_seconds-(time.monotonic()-began)))
                process.wait(timeout=min(20, remaining))
            except subprocess.TimeoutExpired:
                pass
    rc = process.wait()
    temporary = a.receipt.with_suffix('.part')
    temporary.write_text(json.dumps(dict(command=command, returncode=rc, reason=reason,
        wall_seconds=time.monotonic()-began, peak_bytes=peak_bytes, peak_vram_mib=peak_mib,
        stop_gib=a.stop_gib, hard_lane_cap_gb=a.hard_cap_gb, devices=device_info,
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_commit=os.environ.get('WOOF_HURRICANE_SOURCE_COMMIT')), indent=2)+'\n')
    temporary.replace(a.receipt)
    return (1 if reason and rc == 0 else rc) if rc >= 0 else 128-rc


if __name__ == '__main__':
    raise SystemExit(main())
