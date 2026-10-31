"""Complete a staged helper bundle from the runtime's decoder inventory.

A source snapshot over an older wheel's libexec can omit newly required
decoders. Build the source crate offline and copy only the missing helpers
before a forecast acquires its GPU lease. Existing helpers stay byte-for-byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crate", type=Path, required=True)
    parser.add_argument("--libexec", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--cargo", default="cargo")
    args = parser.parse_args()
    from woof.hex.mpasphys.resources_build import DECODERS, decoder_filename

    args.libexec.mkdir(parents=True, exist_ok=True)
    names = {name: decoder_filename(name) for name in DECODERS}
    missing = [name for name, filename in names.items()
               if not (args.libexec / filename).is_file()]
    command = None
    if missing:
        command = [args.cargo, "build", "--release", "--locked", "--offline",
                   "--jobs", str(os.cpu_count() or 1), "--manifest-path", str(args.crate),
                   "--target-dir", str(args.target_dir)]
        subprocess.run(command, check=True)
        for name in missing:
            source = args.target_dir / "release" / names[name]
            with tempfile.NamedTemporaryFile(dir=args.libexec, delete=False) as handle:
                temporary = Path(handle.name)
            try:
                shutil.copy2(source, temporary)
                temporary.chmod(0o755)
                os.replace(temporary, args.libexec / names[name])
            finally:
                temporary.unlink(missing_ok=True)
    helpers = {name: {"path": str(args.libexec / filename),
                      "sha256": digest(args.libexec / filename),
                      "staged": name in missing}
               for name, filename in names.items()}
    receipt = {"schema": "gpuwm-hex.stage-physics-decoders/v1",
               "crate_sha256": digest(args.crate), "command": command,
               "helpers": helpers}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"staged": missing, "verified_helpers": len(helpers),
                      "receipt": str(args.receipt)}))


if __name__ == "__main__":
    main()
