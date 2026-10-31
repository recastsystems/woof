"""Request orchestration only. All acquisition, decoding and numbers are Rust."""
from pathlib import Path
import argparse
import json
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("scratch", type=Path)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    examples = Path(__file__).parent / "examples"
    sources = json.loads((examples / "sources.json").read_text())
    cases = json.loads((examples / "cases.json").read_text())
    configs = args.scratch / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    for case in cases:
        request = {**case, "operation": "stage", "threads": args.threads,
                   "sources": sources, "output": str(args.scratch / "cases" / case["case"])}
        path = configs / (case["case"] + "-stage.json")
        path.write_text(json.dumps(request, indent=2) + "\n")
        with (args.scratch / (case["case"] + "-stage.log")).open("w") as log:
            result = subprocess.run([str(args.binary), str(path)], stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        print(json.dumps({"case": case["case"], "returncode": result.returncode}), flush=True)

if __name__ == "__main__":
    main()
