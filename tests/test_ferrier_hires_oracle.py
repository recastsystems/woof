"""Ferrier-Aligo (WRF mp_physics 5 and 15) against WRF v4.6.1's own Fortran.

The WRF half is tools/ferrier_hires_wrf461_oracle: byte-unmodified
``phys/module_mp_fer_hires.F`` driven column by column (``run_ferrier.F90``
for FER_HIRES, ``advect/run_ferrier_advect.F90`` for FER_HIRES_ADVECT) and,
for the init, the same module with a dump subroutine appended
(``advect/build_init_dump.sh``).  Fixtures are under
tests/data/oracles/ferrier_hires (mp=5, 72 columns) and
tests/data/oracles/ferrier_hires_advect (mp=15, 153 columns; init dump).

Strict arithmetic (GPUWM_WRF_EXACT=1) must hold 0 ULP on every output
field.  The GPU legs run the port in a subprocess because the strict
switch is read once, at import.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tools" / "ferrier_hires_wrf461_oracle"
MP5 = ROOT / "tests" / "data" / "oracles" / "ferrier_hires"
MP15 = ROOT / "tests" / "data" / "oracles" / "ferrier_hires_advect"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cmp():
    return _load(HARNESS / "compare_ferrier.py", "_fh_compare")


# ---------------------------------------------------------------------------
# CPU: fixtures, generator, table loader
# ---------------------------------------------------------------------------

def _sums(folder: Path) -> dict[str, str]:
    out = {}
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            out[parts[1]] = parts[0]
    return out


def test_advect_fixtures_match_the_oracle_run():
    sums = _sums(MP15)
    for name, ref in (("columns_inputs.csv", "columns_inputs.csv"),
                      ("levels.csv", "O0/levels.csv"),
                      ("surface.csv", "O0/surface.csv")):
        got = hashlib.sha256((MP15 / name).read_bytes()).hexdigest()
        assert got == sums[ref], name
    # WRF's own GNU flags wrote the same bits as -O0.
    assert sums["O2vec/levels.csv"] == sums["O0/levels.csv"]
    assert sums["O2vec/surface.csv"] == sums["O0/surface.csv"]


def test_advect_generator_reproduces_the_committed_inputs(tmp_path):
    _load(HARNESS / "advect" / "make_columns_advect.py",
          "_fh_mkadv").main(tmp_path)
    assert (tmp_path / "columns_inputs.csv").read_bytes() == \
        (MP15 / "columns_inputs.csv").read_bytes()


def test_advect_column_set_covers_the_rime_bounds():
    rows = [line.split(",") for line in
            (MP15 / "columns_inputs.csv").read_text().splitlines()
            if line.startswith("C,")]
    assert len(rows) >= 64
    regimes = {r[2] for r in rows}
    for needed in ("convective", "snow", "high_terrain", "freezing_rain",
                   "ocean", "rimef_bounds", "long_run", "negative"):
        assert needed in regimes
    assert sum(r[2] == "rimef_bounds" for r in rows) == 16


def test_table_loader_pins_wrfs_bytes(tmp_path):
    from woof.core import fer_hires as fh
    tables = fh.load_table()
    assert tables["MASSR"].size == fh.NTAB and tables["VEL_RF"].size == 328
    blob = bytearray((fh.packaged_table_root() / fh.TABLE_NAME).read_bytes())
    blob[100] ^= 1
    (tmp_path / fh.TABLE_NAME).write_bytes(bytes(blob))
    with pytest.raises(fh.FerrierTableError,
                       match="corrupt physics table, identity mismatch"):
        fh.load_table(tmp_path)


def test_variant_is_registry_metadata():
    from woof.core import fer_hires as fh
    assert fh.variant_for(15) == "advected_rime"
    assert fh.variant_for(8) is None
    assert 15 in fh.ferrier_selectors()


# ---------------------------------------------------------------------------
# GPU: init, columns, planted bit, default arithmetic
# ---------------------------------------------------------------------------

def _run_port(inputs: Path, out: Path, *, strict: bool) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    if strict:
        env["GPUWM_WRF_EXACT"] = "1"
    else:
        env.pop("GPUWM_WRF_EXACT", None)
    subprocess.run([sys.executable, str(HARNESS / "port_columns.py"),
                    str(inputs), str(out)], check=True, env=env, cwd=ROOT)


def _assert_init_matches_wrf_module_variables():
    from woof.core import fer_hires as fh
    rows = list(csv.DictReader(open(MP15 / "init_constants.csv")))
    for dt in sorted({r["tag"] for r in rows}):
        want = {r["name"]: r["hex"] for r in rows if r["tag"] == dt}
        buf = fh.device_tables(float(dt[4:]))
        got = fh.scalar_values(buf)
        for name, value in got.items():
            assert f"{int(value.view(np.uint32)):08X}" == want[name], (dt, name)
        host = buf.get()
        for name in ("TBPVS", "TBPVS0"):
            o = {"TBPVS": fh.OFF_TBPVS, "TBPVS0": fh.OFF_TBPVS0}[name]
            digest = hashlib.sha256(
                host[o:o + fh.NX].astype("<f4").tobytes()).hexdigest()
            assert digest == want[name + "_sha256"].lower(), (dt, name)


@pytest.mark.gpu
def test_init_matches_wrf_module_variables():
    # Exact tables belong to the proof mode; production keeps fast numerics.
    env = dict(os.environ, GPUWM_WRF_EXACT="1", PYTHONPATH=str(ROOT))
    subprocess.run([sys.executable, "-c",
        "import runpy; runpy.run_path(" + repr(str(Path(__file__).resolve()))
        + ")[\"_assert_init_matches_wrf_module_variables\"]()"],
        check=True, env=env, cwd=ROOT)


@pytest.mark.gpu
def test_advect_strict_is_zero_ulp_on_every_field(tmp_path):
    _run_port(MP15 / "columns_inputs.csv", tmp_path, strict=True)
    assert _cmp().compare(MP15, tmp_path, MP15 / "columns_inputs.csv")


@pytest.mark.gpu
def test_mp5_arm_matches_the_qualified_stock_dependency(tmp_path):
    _run_port(MP5 / "columns_inputs.csv", tmp_path, strict=True)
    assert _cmp().compare(MP5, tmp_path, MP5 / "columns_inputs.csv")


@pytest.mark.gpu
def test_a_planted_bit_in_the_variant_path_is_caught(tmp_path):
    """The unmutated kernel and a copy one ULP off on the advected rime
    write-back, both compiled here with the strict options: the control
    must pass and the mutant must fail, so the oracle sees one bit."""
    import cupy as cp
    from woof.core import kernels as K
    from woof.wrf_exact import STRICT_OPTIONS
    port = _load(HARNESS / "port_columns.py", "_fh_port")
    target = "qrimef[idx] = QIo * FrimeF_o;"
    results = {}
    for tag, text in (("control", target),
                      ("mutant", "qrimef[idx] = __uint_as_float("
                                 "__float_as_uint(QIo * FrimeF_o) ^ 1u);")):
        kdir = tmp_path / tag / "k"
        kdir.mkdir(parents=True)
        for f in K._KDIR.iterdir():
            if f.suffix in (".cuh", ".cu"):
                shutil.copy(f, kdir / f.name)
        src = (kdir / "fer_hires.cu").read_text(encoding="utf-8")
        assert src.count(target) == 1
        (kdir / "fer_hires.cu").write_text(src.replace(target, text),
                                           encoding="utf-8")
        mod = cp.RawModule(code=K.module_source("fer_hires", kernel_dir=kdir),
                           options=("-std=c++17",) + STRICT_OPTIONS)
        out = tmp_path / tag / "out"
        port.run(MP15 / "columns_inputs.csv", out,
                 kernel=mod.get_function("fer_hires_column"))
        results[tag] = _cmp().compare(MP15, out, MP15 / "columns_inputs.csv")
    assert results == {"control": True, "mutant": False}
