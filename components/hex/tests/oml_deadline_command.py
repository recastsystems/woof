"""Run an admitted proof only when its budget fits the host retirement deadline."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import signal
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finish-before', required=True)
    parser.add_argument('--budget-seconds', type=int, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    deadline = datetime.fromisoformat(args.finish_before)
    if deadline.tzinfo is None or args.budget_seconds <= 0 or not command:
        raise ValueError('Explicit UTC deadline, positive budget and command required')
    remaining = (deadline-datetime.now(timezone.utc)).total_seconds()
    if remaining < args.budget_seconds:
        print(json.dumps(dict(executed=False, forecast_acceptance=False,
                              reason='host retirement deadline',
                              finish_before=args.finish_before,
                              required_budget_seconds=args.budget_seconds)), flush=True)
        return 75
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait(timeout=args.budget_seconds)
    except subprocess.TimeoutExpired:
        # Stop only this command's explicit PIDs before the card is given back.
        # A killed verifier must not leave its CUDA forecast running underneath.
        descendants = {process.pid}
        parents = {}
        for directory in Path('/proc').iterdir():
            if not directory.name.isdigit():
                continue
            try:
                parents[int(directory.name)] = int((directory/'stat').read_text().split(') ', 1)[1].split()[1])
            except (FileNotFoundError, ProcessLookupError):
                continue
        while True:
            more = descendants | {pid for pid, parent in parents.items() if parent in descendants}
            if more == descendants:
                break
            descendants = more
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in sorted(descendants, reverse=True):
                try:
                    if os.getsid(pid) == process.pid:
                        os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        process.wait()
        print(json.dumps(dict(forecast_acceptance=False,
                              reason='proof exceeded admitted execution budget')), flush=True)
        return 124


if __name__ == '__main__':
    raise SystemExit(main())
