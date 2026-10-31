"""The boundary door writes its times in several rw_mpas_lbc processes at once.

THE BREAKAGE: rw_mpas_lbc writes the times of one call one after another on one core (111 % CPU, 16.9 s for the
7 hourly times of a 6 h 96->3 km region, an RTX 5090 host, 2026-10-29), and hourly table-source boundaries multiply the times.
One process per time wrote byte-identical files in 4.1 s.
"""
import argparse
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from woof.hex import hrrr_intermediate as door

pytestmark = pytest.mark.skipif(os.name == "nt", reason="WPS intermediate names carry ':', which NTFS reads as a stream")

FAKE = """#!{python}
import json, sys
from pathlib import Path
argv = sys.argv[1:]
out = Path(argv[argv.index("--out-dir") + 1])
times = [argv[i + 1].split("=")[0] for i, a in enumerate(argv) if a == "--interval"]
for t in times:
    (out / ("lbc." + t.replace(":", ".") + ".nc")).write_text("boundary " + t + " " + argv[argv.index("--fg-interval-seconds") + 1])
Path(argv[argv.index("--receipt") + 1]).write_text(json.dumps({{"times": times}}))
"""


def _arguments(tmp_path: Path, workers):
    engine = tmp_path / "rw_mpas_lbc"
    engine.write_text(FAKE.format(python=sys.executable))
    engine.chmod(engine.stat().st_mode | stat.S_IXUSR)
    met = tmp_path / "met"
    met.mkdir()
    for hour in range(18, 25):
        day, hh = (7, hour) if hour < 24 else (8, 0)
        (met / f"MET:2026-10-{day:02d}_{hh:02d}").write_bytes(b"x" * 64)
    grid = tmp_path / "init.nc"
    grid.write_bytes(b"g" * 64)
    return argparse.Namespace(
        lbc_exe=engine, grid=grid, met_dir=met, out_dir=tmp_path / "lbc", start_time="2026-10-07_18:00:00",
        stop_time="2026-10-08_00:00:00", nfglevels=34, extrap_airtemp="lapse-rate", use_spechumd="no",
        theta_adv_order=3, coef_3rd_order=0.25, oned_underflow="preserve", physics_mode="exact", stream=False,
        workers=workers)


@pytest.mark.parametrize("workers", [1, 3, 7])
def test_every_time_lands_once_whatever_the_process_count(tmp_path, workers, monkeypatch):
    from datetime import datetime

    arguments = _arguments(tmp_path, workers)
    held = [(datetime.strptime(p.name[4:], "%Y-%m-%d_%H"), p) for p in sorted(arguments.met_dir.iterdir())]
    monkeypatch.setattr(door, "intermediate_valid_times", lambda _: held)  # the stand-ins carry no WPS header
    assert door.run_lbc(arguments) == 0
    out = tmp_path / "lbc"
    names = sorted(p.name for p in out.glob("lbc.*.nc"))
    assert names == [f"lbc.2026-10-07_{h:02d}.00.00.nc" for h in range(18, 24)] + ["lbc.2026-10-08_00.00.00.nc"]
    assert all(p.read_text().endswith(" 3600") for p in out.glob("lbc.*.nc"))
    assert not list(out.glob(".part-*"))
    receipt = json.loads((out / "lbc-door-receipt.json").read_text())
    # The door runs at most one process per CPU this process may use (lbc_workers); a 4-CPU CI runner gave
    # 4 for 7 requested (woof hex public ci, 0.4.10 main 765a6eff, 2026-10-30).
    from woof.hex.cpu_budget import available_cpus
    assert receipt["processes"] == min(workers, available_cpus(), 16) and len(receipt["files"]) == 7


def test_workers_follow_cpus_and_memory(monkeypatch, tmp_path):
    from woof.hex import cpu_budget, products_door

    grid, met = tmp_path / "g", tmp_path / "m"
    grid.write_bytes(b"g" * 1000)
    met.write_bytes(b"m" * 1000)
    monkeypatch.setattr(cpu_budget, "available_cpus", lambda: 24)
    monkeypatch.setattr(products_door, "available_memory_bytes", lambda: 100_000)
    assert door.lbc_workers(7, grid, [met]) == 7
    assert door.lbc_workers(49, grid, [met]) == 16
    monkeypatch.setattr(products_door, "available_memory_bytes", lambda: 8_000)  # 2 x 2,000 bytes fit in half
    assert door.lbc_workers(7, grid, [met]) == 2
    monkeypatch.setattr(products_door, "available_memory_bytes", lambda: None)
    assert door.lbc_workers(7, grid, [met]) == 1
    assert door.lbc_workers(7, grid, [met], requested=3) == 3
