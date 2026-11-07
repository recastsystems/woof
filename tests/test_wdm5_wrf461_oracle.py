"""WDM5 stock WRF diagnostic fixture and pinned coefficient words."""
from __future__ import annotations

import gzip
import importlib.util
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "oracles" / "wdm5"
TOOLS = ROOT / "tools" / "wdm5_wrf461_oracle"
KERNELS = ROOT / "woof" / "core" / "kernels"
CONFIGS = ("dt20",)
FLT_MIN = np.float32(1.17549435e-38)


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(
        f"wdm5_oracle_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module through sys.modules.
    sys.modules[spec.name] = module
    sys.path.insert(0, str(TOOLS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(TOOLS))
    return module


IO = _tool("oracle_io")


def _gunzip(src: Path, dst: Path) -> Path:
    dst.write_bytes(gzip.decompress(src.read_bytes()))
    return dst


def _case(tmp_path: Path, config: str):
    path = _gunzip(DATA / f"{config}.in.gz", tmp_path / f"{config}.in")
    return path, IO.read_input(path)


def _reference(tmp_path: Path, config: str, arm: str, case):
    path = _gunzip(DATA / f"{config}.{arm}.out.gz",
                   tmp_path / f"{config}.{arm}.out")
    return IO.read_output(path, case.ncol, case.nz, case.nsteps)


def _subnormal_columns(levels) -> np.ndarray:
    first = next(iter(levels.values()))
    flags = np.zeros(first.shape[1], bool)
    for name in ("qv", "qc", "qr", "qi", "qs", "nn", "nc", "nr"):
        a = np.abs(np.asarray(levels[name], np.float32))
        flags |= ((a > 0) & (a < FLT_MIN)).any(axis=0)
    return flags


def _columns_fed_a_subnormal(case, reference_steps) -> np.ndarray:
    """Columns where a subnormal value ENTERS some call: from the input, or
    from the scheme's own output of an earlier step (WRF's words, which the
    strict build reproduces)."""
    fed = _subnormal_columns(case.levels)
    for levels, _ in reference_steps[:-1]:
        fed |= _subnormal_columns(levels)
    return fed


def _differing(woof_steps, ref_steps, *, skip=(), scale_reference_radii=True):
    """Per-field count of differing words, and the columns that differ."""
    counts, columns = {}, None
    for (wl, ws), (rl, rs) in zip(woof_steps, ref_steps):
        for name in IO.OUTPUT_LEVEL_FIELDS + IO.OUTPUT_SURFACE_FIELDS:
            if name in skip:
                continue
            level = name in IO.OUTPUT_LEVEL_FIELDS
            w = np.asarray((wl if level else ws)[name], np.float32)
            r = np.asarray((rl if level else rs)[name], np.float32)
            if scale_reference_radii and name in ("re_cloud", "re_ice",
                                                  "re_snow"):
                # WRF's metres, scaled by RRTMG's own REAL multiply.
                r = (r * np.float32(1.0e6)).astype(np.float32)
            diff = w.view(np.uint32) != r.view(np.uint32)
            colhit = diff.any(axis=0) if level else diff
            columns = colhit if columns is None else (columns | colhit)
            counts[name] = counts.get(name, 0) + int(diff.sum())
    return counts, columns


def _dump() -> dict[str, int]:
    words = {}
    for line in (DATA / "wdm5init.txt").read_text().splitlines():
        name, bits, _ = line.split()
        words[name] = int(bits, 16)
    return words


def _f32_bits(text: str) -> int:
    return int(np.float32(float(text.rstrip("f"))).view(np.uint32))


# ---------------------------------------------------------------------------
# Provenance and the coefficient words (CPU)
# ---------------------------------------------------------------------------

def test_the_oracle_data_names_the_pinned_wrf_source():
    sums = (DATA / "oracle-sha256sums.txt").read_text(encoding="utf-8")
    assert ("0ee4071f1bc85f985130f3b4cf1f5a9acf87f23078e5e06b169d3681ff13c01c"
            "  <wrf-4.6.1-checkout>/phys/module_mp_wdm5.F") in sums
    assert (DATA / "wrf-commit.txt").read_text().strip() == (
        "d66e442fccc04111067e29274c9f9eaccc3cef28")
    assert "13.3.0" in (DATA / "compiler.txt").read_text()
    # The linked reference stays on scalar libm.
    report = (DATA / "libmvec-report.txt").read_text()
    assert "_ZGV" not in report.split("# -O2")[0]


def test_the_kernels_bake_wdm5init_s_own_words():
    """Every coefficient literal the WDM5 path reads is the float32 word WRF's
    wdm5init (or gfortran's compile-time fold) leaves, not a float64 product
    rounded once."""
    dump = _dump()
    names = {
        "WDM_PI": "pi", "WDM_PIDNC": "pidnc", "WDM_PIDNR": "pidnr",
        "WDM_PVTR": "pvtr", "WDM_PVTRN": "pvtrn", "WDM_PRECR1": "precr1",
        "WDM_PRECR2": "precr2", "WDM_PVTS": "pvts", "WDM_PIDN0S": "pidn0s",
        "WDM_PACRC": "pacrc", "WDM_PRECS1": "precs1", "WDM_PRECS2": "precs2",
        "WDM_ROQIMAX": "roqimax", "WDM_RSLOPECMAX": "rslopecmax",
        "WDM_RSLOPEC2MAX": "rslopec2max", "WDM_RSLOPEC3MAX": "rslopec3max",
        "WDM_RSLOPERMAX": "rslopermax", "WDM_RSLOPERBMAX": "rsloperbmax",
        "WDM_RSLOPER2MAX": "rsloper2max", "WDM_RSLOPER3MAX": "rsloper3max",
        "WDM_RSLOPESMAX": "rslopesmax", "WDM_RSLOPESBMAX": "rslopesbmax",
        "WDM_RSLOPES2MAX": "rslopes2max", "WDM_RSLOPES3MAX": "rslopes3max",
        "WDM_CDM2": "cdm2", "WDM_ACTR3": "fold_actr3",
        "WDM_CBRT24": "fold_24cbrt", "WDM5_XLV1": "xlv1",
        "WDM5_1E20O16": "fold_1e20o16", "WDM5_XAM_R": "xam_r",
        "WDM5_XAM_S": "xam_s", "WDM5_XOBMR": "xobmr", "WDM5_XOCMS": "xocms",
        "WDM5_XCRG4": "xcrg4",
    }
    source = ((KERNELS / "wdm_common.cuh").read_text(encoding="utf-8")
              + (KERNELS / "wdm5.cu").read_text(encoding="utf-8"))
    defines = dict(re.findall(r"#define (WDM5?_[A-Z0-9_]+)\s+([0-9.eE+-]+f)\b",
                              source))
    for macro, word in names.items():
        assert macro in defines, macro
        assert _f32_bits(defines[macro]) == dump[word], (macro, word)
    # The radar module's tuples WDM5 relies on being exact.
    assert dump["xcrg3"] == dump["xcsg3"] == _f32_bits("6.0")
    assert dump["xcrg4"] == dump["xcsg4"]
    assert dump["xcre2"] == _f32_bits("1.0") and dump["xcre4"] == _f32_bits("7.0")
    assert dump["xcse1"] == _f32_bits("4.0") and dump["xcse4"] == _f32_bits("7.0")
    assert dump["xorg2"] == _f32_bits("1.0") and dump["xobms"] == dump["xobmr"]
    # 0.176/0.93 as refl10cm_wdm5 folds it.
    assert "0.189247310f" in source and dump["fold_0176o093"] == 0x3E41CA0C


def test_the_reflectivity_words_are_wrf_s():
    from woof.core.refl import WDM5_RADAR_WORDS

    text = (DATA / "wdm5init.txt").read_text()
    dbl = {line.split()[0]: int(line.split()[1], 16)
           for line in text.splitlines() if len(line.split()[1]) == 16}
    assert WDM5_RADAR_WORDS == {name: dbl[name] for name in WDM5_RADAR_WORDS}


def test_the_reflectivity_bin_tables_are_wrf_s():
    """radar_init's snow bins and Simpson weights, computed on this host,
    are WRF's words (the device kernel reads them from woof.core.refl)."""
    if platform.libc_ver()[0] != "glibc":
        pytest.skip("the bin tables are graded on a glibc host")
    from woof.core.refl import radar_init

    rc = radar_init()
    text = (DATA / "wdm5init.txt").read_text()
    dbl = {line.split()[0]: int(line.split()[1], 16)
           for line in text.splitlines()}
    for name, table in (("xxDs", rc.xxds), ("xdts", rc.xdts),
                        ("simpson", rc.simpson)):
        got = np.asarray(table, np.float64).view(np.uint64)
        want = [dbl[f"{name}{n + 1:02d}"] for n in range(50)]
        assert [int(v) for v in got] == want, name


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("config", ("dt20", "native_step2"))
def test_device_matches_the_stock_wrf_fixture(tmp_path, config):
    from woof.numerics import is_strict
    if not is_strict():
        pytest.skip("bitwise native replay requires GPUWM_NUMERICS=strict")
    inp, case = _case(tmp_path, config)
    out = tmp_path / "gpu.out"
    _tool("run_woof").run(inp, out)
    counts, _ = _differing(IO.read_output(out, case.ncol, case.nz, case.nsteps),
                           _reference(tmp_path, config, "stock", case))
    assert not any(counts.values()), counts


def test_comparator_detects_a_one_bit_change():
    compare = _tool("compare")
    a = np.ones(4, np.float32)
    b = a.copy()
    b.view(np.uint32)[2] ^= np.uint32(1)
    hit, ulp, _ = compare.compare_arrays(a, b)
    assert hit.tolist() == [False, False, True, False]
    assert ulp[2] == 1


def test_wdm5_admission_preserves_the_surface_compiler_policy():
    from woof.core.kernels import module_options
    expected = ("-std=c++17", "--fmad=false", "--ftz=false")
    for name in ("sfclay", "myjsfc", "mynn_surface", "mynn_pbl"):
        assert module_options(name) == expected, name
