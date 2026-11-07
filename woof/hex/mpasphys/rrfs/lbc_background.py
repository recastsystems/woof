"""Own a detached boundary job until every scheduled file is durable."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys


def main(argv=None):
    from ...lbc_stream import atomic_json

    worker, specification = (sys.argv[1:] if argv is None else argv)
    specification = Path(specification)
    job = json.loads(specification.read_text())
    output = Path(job["config"]["out_dir"])
    try:
        completed = subprocess.run([worker], input=json.dumps(job).encode(), stdout=subprocess.PIPE)
        if completed.returncode:
            raise RuntimeError(f"boundary worker exited {completed.returncode}; log {specification.parent / 'producer.log'}")
        receipt = json.loads(completed.stdout)
        atomic_json(specification.parent / "complete.json", receipt)
        return 0
    except Exception as error:
        atomic_json(output / "lbc-failed.json", {"error": str(error)})
        print(str(error), file=sys.stderr, flush=True)
        return 1
    finally:
        shutil.rmtree(specification.parent / "met", ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
