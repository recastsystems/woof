"""Choose wide-family oracle cases by the WRF source lines they execute.

usage: select_cases.py WRF_SOURCE_ROOT BUILD_DIR N_CANDIDATES [--write]

build.sh must have run in BUILD_DIR (it leaves the harness, constants and
stub objects there).  This compiles phys/module_cu_kf.F once more with
--coverage, runs every curated case and N_CANDIDATES wide candidates
through it one process each, reads gcov's per-line counts, and keeps, in
candidate order, every candidate that executes a line of the scheme no
earlier case executed.  With --write it records the kept indices in
selected_wide.json beside this file, which gen_cases.py then appends to
the fixture's case set.  It always writes BUILD_DIR/kf-old-coverage.txt:
the scheme's executable lines the final case set never reaches.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True,
                          **kw)


def lines_of(cov_dir: Path, binary: Path, cases: Path, index: int,
             gcno: Path) -> frozenset[int]:
    prefix = cov_dir / f"p{index}"
    if prefix.exists():
        shutil.rmtree(prefix)
    prefix.mkdir(parents=True)
    env = dict(os.environ, GCOV_PREFIX=str(prefix),
               GCOV_PREFIX_STRIP="99")
    subprocess.run([str(binary), str(cases), str(index), os.devnull],
                   env=env, capture_output=True)
    gcda = prefix / "module_cu_kf.gcda"
    if not gcda.exists():
        raise RuntimeError(f"no coverage data for case {index}")
    shutil.copy(gcno, prefix / "module_cu_kf.gcno")
    out = run(["gcov", "-j", "-o", str(prefix), "module_cu_kf.F"],
              cwd=prefix)
    del out
    report = next(prefix.glob("*.gcov.json.gz"))
    data = json.loads(gzip.decompress(report.read_bytes()))
    executed = set()
    for entry in data["files"]:
        if not entry["file"].endswith("module_cu_kf.F"):
            continue
        for line in entry["lines"]:
            if line["count"] > 0:
                executed.add(line["line_number"])
    shutil.rmtree(prefix)
    return frozenset(executed)


def well_posed(build_dir: Path, cases: Path, index: int) -> bool:
    """Arms A, N (and B when built) agree word for word and stay finite.

    A candidate fails this when WRF's own arithmetic leaves the defined
    path: TPMIX's secant step divides 0 by 0 for some extreme parcels
    (module_cu_kf.F:2381), and once a NaN is in the column the result
    depends on the build (the -O2 production object orders MIN/MAX
    operands differently) and on stack contents.  Such a column has no
    single WRF answer to grade a port against, so it is not admitted.
    """
    import numpy as np

    outs = []
    for arm in ("A", "N", "B"):
        exe = build_dir / f"run_kf_old_{arm}"
        if not exe.exists():
            continue
        target = build_dir / "coverage" / f"wp-{arm}-{index}.bin"
        proc = subprocess.run([str(exe), str(build_dir / "coverage" /
                                             cases.name), str(index),
                               str(target)], capture_output=True)
        if proc.returncode != 0:
            outs.append(b"fatal")
            continue
        outs.append(target.read_bytes())
        target.unlink()
    if any(out != outs[0] for out in outs):
        return False
    if outs[0] == b"fatal":
        return True
    floats = np.frombuffer(outs[0], "<f4", (len(outs[0]) - 12) // 4, 12)
    return bool(np.all(np.isfinite(floats)))


def all_executable(cov_dir: Path, binary: Path, cases: Path, gcno: Path):
    prefix = cov_dir / "all"
    prefix.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GCOV_PREFIX=str(prefix), GCOV_PREFIX_STRIP="99")
    subprocess.run([str(binary), str(cases), "0", os.devnull], env=env,
                   capture_output=True)
    shutil.copy(gcno, prefix / "module_cu_kf.gcno")
    run(["gcov", "-j", "-o", str(prefix), "module_cu_kf.F"], cwd=prefix)
    data = json.loads(gzip.decompress(
        next(prefix.glob("*.gcov.json.gz")).read_bytes()))
    return sorted(line["line_number"] for entry in data["files"]
                  if entry["file"].endswith("module_cu_kf.F")
                  for line in entry["lines"])


def main() -> None:
    source_root = Path(sys.argv[1]).resolve()
    build_dir = Path(sys.argv[2]).resolve()
    ncand = int(sys.argv[3])
    write = "--write" in sys.argv[4:]
    cov_dir = build_dir / "coverage"
    if cov_dir.exists():
        shutil.rmtree(cov_dir)
    cov_dir.mkdir()
    src = cov_dir / "module_cu_kf.F"
    shutil.copy(source_root / "phys" / "module_cu_kf.F", src)
    run(["gfortran", "-c", "-O0", "-cpp", "-ffree-form",
         "-ffree-line-length-none", "--coverage", "-I", str(build_dir),
         "-J", str(cov_dir), "-o", "module_cu_kf.o", "module_cu_kf.F"],
        cwd=cov_dir)
    binary = cov_dir / "run_cov"
    run(["gfortran", "--coverage", "-o", str(binary),
         str(build_dir / "run_kf_old.o"), str(cov_dir / "module_cu_kf.o"),
         str(build_dir / "module_model_constants.o"),
         str(build_dir / "stub_wrf.o")])
    gcno = cov_dir / "module_cu_kf.gcno"
    real74 = HERE.parent.parent / "tests" / "data" / \
        "kf_real74_12z_columns.npz"
    base = cov_dir / "base.bin"
    saved = HERE / "selected_wide.json"
    moved = None
    if saved.exists():
        moved = saved.with_suffix(".json.bak")
        saved.rename(moved)
    try:
        run([sys.executable, str(HERE / "gen_cases.py"), str(base),
             "--real74", str(real74)])
    finally:
        if moved is not None:
            moved.rename(saved)
    cand = cov_dir / "cand.bin"
    run([sys.executable, str(HERE / "gen_cases.py"), str(cand),
         "--candidates", str(ncand)])
    nbase = json.loads(base.with_suffix(".json").read_text()).__len__()

    with ThreadPoolExecutor(max_workers=48) as pool:
        base_lines = list(pool.map(
            lambda i: lines_of(cov_dir, binary, base, i, gcno),
            range(nbase)))
        cand_lines = list(pool.map(
            lambda i: lines_of(cov_dir, binary, cand, i, gcno),
            range(ncand)))
    covered = set().union(*base_lines)
    print(f"curated set ({nbase} cases) executes {len(covered)} lines")
    kept = []
    rejected = []
    for index, executed in enumerate(cand_lines):
        new = executed - covered
        if new:
            if not well_posed(build_dir, cand, index):
                rejected.append(index)
                continue
            kept.append(index)
            covered |= executed
            print(f"  candidate {index}: +{len(new)} lines "
                  f"({min(new)}..{max(new)})")
    print(f"kept {len(kept)} of {ncand} candidates; "
          f"{len(covered)} lines executed; rejected as ill-posed "
          f"(arms disagree or non-finite): {rejected}")
    executable = all_executable(cov_dir, binary, base, gcno)
    source = src.read_text().splitlines()
    missed = [n for n in executable if n not in covered]
    with (build_dir / "kf-old-coverage.txt").open("w") as fh:
        fh.write(f"# phys/module_cu_kf.F lines gcov counts as executable: "
                 f"{len(executable)}; executed by the fixture's cases: "
                 f"{len(set(executable) & covered)}; never executed: "
                 f"{len(missed)}\n")
        for n in missed:
            fh.write(f"{n:5d}: {source[n - 1].rstrip()}\n")
    if write:
        saved.write_text(json.dumps({
            "seed_note": "gen_cases.wide_candidate(seed, index)",
            "candidates_scanned": ncand,
            "rejected_ill_posed": rejected,
            "candidates": kept}, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {saved}")


if __name__ == "__main__":
    main()
