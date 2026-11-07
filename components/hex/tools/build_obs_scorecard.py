"""Build the hex scorer against the installed engine's existing Rust readers.

This creates a private Cargo project, never edits the engine workspace.
Run on a compute box, not on the desktop.
"""
import argparse
import json
import os
import shlex
from pathlib import Path
import subprocess
import tomllib
import shutil


def write_project(engine: Path, out: Path) -> Path:
    """Write the scorer's private Cargo project under ``out`` and return its manifest.

    tools/helper_licences.py reads the same project, so the licence notice
    covers exactly the crates this build links.
    """
    root = engine.resolve() / "tools/rustwx"
    out.mkdir(parents=True, exist_ok=True)
    dependencies = {
        "netcrust": root / "vendor/netcrust",
        "grib-core": root.parent / "grib1_bridge/vendor/grib-core",
        "rw-wrfbatch": root / "crates/rw-wrfbatch",
        "rw-obs": root / "crates/rw-obs",
        "rw-isobaric": root / "crates/rw-isobaric",
        "rustwx-core": root / "crates/rustwx-core",
        "rustwx-io": root / "crates/rustwx-io",
    }
    source = Path(__file__).resolve().parents[1] / "rust/obs-scorecard/main.rs"
    manifest = ('[package]\nname="hex-obs-scorecard"\nversion="0.1.0"\nedition="2021"\npublish=false\n'
                '[[bin]]\nname="hex_obs_scorecard"\npath=' + json.dumps(str(source)) + '\n'
                '[dependencies]\nserde_json="1"\nchrono="0.4"\n'
                # IGRA2 station zips (the engine's rw_igra2 keeps NCEI's .txt.zip); rw-obs already locks flate2.
                'flate2={version="1",default-features=false,features=["rust_backend"]}\n')
    for name, path in dependencies.items():
        if not (path / "Cargo.toml").is_file():
            raise FileNotFoundError(path / "Cargo.toml")
        manifest += name + '={path=' + json.dumps(str(path)) + '}\n'
    workspace = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    for registry, rows in workspace.get("patch", {}).items():
        manifest += '\n[patch.' + json.dumps(registry) + ']\n'
        for name, row in rows.items():
            if set(row) != {"path"}:
                raise ValueError(f"unsupported engine patch metadata for {name}")
            manifest += name + '={path=' + json.dumps(str((root / row["path"]).resolve())) + '}\n'
    target = out / "Cargo.toml"
    target.write_text(manifest, encoding="utf-8")
    lock = out / "Cargo.lock"
    if not lock.exists():
        shutil.copyfile(Path(__file__).resolve().parents[1] / "rust/obs-scorecard/Cargo.lock", lock)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--install", action="store_true", help="stage the native binary in this hex package's libexec for wheel packaging")
    args = parser.parse_args()
    root = args.engine.resolve() / "tools/rustwx"
    source = Path(__file__).resolve().parents[1] / "rust/obs-scorecard/main.rs"
    target = write_project(args.engine, args.out)
    # Cargo's target override is used by development branch builds to keep scratch
    # off the source disk. Copy from that same location after compilation.
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", args.out.resolve() / "target"))
    if not target_dir.is_absolute():
        target_dir = root / target_dir
    env = dict(os.environ)
    # Rust panic locations otherwise embed the random build-via-sdist root.
    # Keep caller flags and remap the three source/project roots consistently.
    encoded = env.get("CARGO_ENCODED_RUSTFLAGS")
    flags = encoded.split("\x1f") if encoded is not None else shlex.split(env.get("RUSTFLAGS", ""))
    for path, stable in ((args.engine.resolve(), "/gpuwm"),
                         (Path(__file__).resolve().parents[1], "/gpuwm-hex"),
                         (args.out.resolve(), "/hex-obs-scorecard-build")):
        flags.append(f"--remap-path-prefix={path}={stable}")
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    code = subprocess.call(["cargo", "test" if args.test else "build", "--release", "--offline", "--locked",
                            "--jobs", os.environ.get("CARGO_BUILD_JOBS", "8"),
                            "--target-dir", str(target_dir), "--manifest-path", str(target)], cwd=root, env=env)
    if code == 0 and args.install:
        destination = source.parents[2] / "src/hexcore/libexec"
        destination.mkdir(parents=True, exist_ok=True)
        executable = "hex_obs_scorecard.exe" if __import__("os").name == "nt" else "hex_obs_scorecard"
        shutil.copyfile(target_dir / "release" / executable, destination / executable)
        (destination / executable).chmod(0o755)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
