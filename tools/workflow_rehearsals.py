"""Census of every workflow job and its pre-push rehearsal route.

R exercises publication controllers with recorded/local fixtures. It never
performs publication. Optional GPU dispatch is checked as a workflow contract;
device qualification remains a separate release gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
# Session job keys are executable routes in ci-repro-289.sh.
ENTRIES = {
    "ci.yml": {
        "publication_contracts": ("R", "nightly contract selection and fetcher setup fixtures"),
        "syntax": ("S", "compileall under declared Python floor"),
        "imports": ("I", "import sweep; Linux plus Windows contract"),
        "cpu": ("B", "C", "Stage 1 Linux; Windows partition emulation"),
        "oracles": ("D", "must-run oracle decks"),
        "native": ("A", "offline cargo gates"),
        "static_qualification": ("Q", "Linux result; Windows contract"),
        "gpu": ("R", "optional dispatch contract only; no GPU execution"),
    },
    "publish.yml": {
        "controller": ("R", "controller hash and same-run artifact handoff fixtures"),
        "test": ("P", "actual YAML steps, explicit local packet, fresh checkout, Python and home"),
        "release_authority": ("R", "capture authority and packet fixtures"),
        "ci": ("R", "commit-bound CI judgement fixtures"),
        "prepare": ("R", "packet, distribution and native proof fixtures"),
        "qualify": ("R", "platform installation smoke fixtures"),
        "assets": ("R", "promotion order, refusal and retry fixtures"),
        "publish_startup": ("R", "startup companion staging, index, retry and upload-path fixtures"),
        "publish": ("R", "staging, index, retry and upload-path fixtures"),
        "release": ("R", "final release and index fixtures"),
    },
}
CONTRACT_FILES = (
    "tests/test_ci_workflow.py",
    "tests/test_prepared_publication.py",
    "tests/test_publish_workflow_state_machine.py",
    "tests/test_publish_dist_shape_agreement.py",
    "tests/test_native_wrf_distribution.py",
    "tests/test_workflow_rehearsals.py",
    "tests/test_workflow_job_replay.py",
    "tests/test_pypi_capacity.py",
    "tests/test_publish_preflight.py",
    "tests/test_vendor_cache_ignore.py",
)


def census(root=ROOT):
    found = {}
    for path in sorted((root / ".github/workflows").glob("*.y*ml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        found[path.name] = document["jobs"]
    actual = {(f, job) for f, jobs in found.items() for job in jobs}
    declared = {(f, job) for f, jobs in ENTRIES.items() for job in jobs}
    if actual != declared:
        raise ValueError(f"workflow rehearsal coverage differs: missing={sorted(actual - declared)}; stale={sorted(declared - actual)}")
    return [{"workflow": f, "job": job, "routes": list(ENTRIES[f][job][:-1]),
             "scope": ENTRIES[f][job][-1], "runs_on": value.get("runs-on"),
             "matrix": value.get("strategy", {}).get("matrix"),
             # Git's text checkout can use CRLF on the release desktop.
             "workflow_sha256": hashlib.sha256((root / ".github/workflows" / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
            for f, jobs in found.items() for job, value in jobs.items()]


def record(root, commit, results):
    rows = census(root)
    required = {route for row in rows for route in row["routes"]}
    failed = sorted(route for route in required if results.get(route) != 0)
    if failed:
        raise ValueError(f"pre-push rehearsal incomplete or failed: {failed}")
    return {"schema": "arwen.workflow-rehearsals.v1", "status": "PASS", "commit": commit,
            "jobs": rows, "results": results,
            "limits": "runner image differences, Windows emulation, secrets and actual uploads are not covered"}


def require_receipt(path, root, commit):
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = record(root, commit, receipt["results"])
    if any(receipt.get(key) != expected[key] for key in ("schema", "status", "commit", "jobs")):
        raise ValueError("workflow rehearsal receipt does not bind to this commit and workflow census")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--commit")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--contracts", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.contracts:
            census(args.root)
            import subprocess
            return subprocess.run([sys.executable, "-m", "pytest", "-q", "-m",
                "not gpu and not slow and not network", *CONTRACT_FILES], cwd=args.root).returncode
        result = record(args.root, args.commit, json.loads(args.results.read_text())) if args.results else census(args.root)
        text = json.dumps(result, indent=2) + "\n"
        if args.out:
            args.out.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f"REFUSED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
