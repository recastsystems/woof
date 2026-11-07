"""PX table provenance and scalar launch validation, CPU only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from woof.core import pxlsm

ORACLE = Path(__file__).resolve().parents[1] / "woof/data/pxlsm/oracle"

def test_every_table_word_equals_the_wrf_object_dump():
    rows = pxlsm.load_pxlsm_tables()
    count = 0
    for line in (ORACLE / "pxlsm-tables.txt").read_text().splitlines():
        dataset, row, n, *words = line.split()
        got = rows[dataset][pxlsm.PX_TABLE_ROWS.index(row)]
        want = np.array([int(w, 16) for w in words], dtype=np.uint32)
        assert len(want) == int(n)
        np.testing.assert_array_equal(got.view(np.uint32), want)
        assert not rows[dataset].flags.writeable
        count += len(want)
    assert count == 9 * (24 + 28 + 40 + 50 + 61)
    assert hashlib.sha256(pxlsm.tables_path().read_bytes()).hexdigest() == pxlsm.PXLSM_TABLES_SHA256

def test_one_changed_table_bit_is_refused_before_device_allocation(tmp_path, monkeypatch):
    doc = json.loads(pxlsm.tables_path().read_text())
    word = np.array([doc["datasets"]["USGS"]["Z00"][0]], dtype=np.float32).view(np.uint32)
    word[0] ^= np.uint32(1)
    doc["datasets"]["USGS"]["Z00"][0] = float(word.view(np.float32)[0])
    path = tmp_path / "tables.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(pxlsm, "tables_path", lambda: path)
    pxlsm.load_pxlsm_tables.cache_clear()
    try:
        with pytest.raises(ValueError, match="SHA-256"):
            pxlsm.load_pxlsm_tables()
    finally:
        pxlsm.load_pxlsm_tables.cache_clear()

@pytest.mark.parametrize("nlcat", [20, 21, 24, 28, 40, 50, 61])
def test_device_table_geometry(nlcat):
    table = pxlsm.px_device_table(nlcat)
    assert table.shape == (9, nlcat)
    assert table.dtype == np.float32 and table.flags.c_contiguous

def test_unknown_category_count_names_the_missing_tables():
    with pytest.raises(ValueError, match="Unknown Land Use Category"):
        pxlsm.px_dataset(25)

def valid_call():
    return dict(dt=10., dtbl=10., curr_secs=0., itimestep=1,
                smois_init=1, soil_nudge=0, modis_veg=0, anal_interval=0,
                domain_shape=(8, 12), tile_origin=(0, 0),
                tile_shape=(8, 12), dzs=(.01, .99))

@pytest.mark.parametrize("name,value", [
    ("dt", 0.), ("dt", float("nan")), ("dtbl", 0.),
    ("curr_secs", float("inf")), ("itimestep", True), ("itimestep", 0),
    ("soil_nudge", 2), ("domain_shape", (0, 12)), ("tile_origin", (1, 0)),
    ("dzs", (.01,)), ("dzs", (.01, 0.))])
def test_bad_scalar_call_is_refused_before_cuda_import(name, value):
    args = valid_call()
    args[name] = value
    with pytest.raises(ValueError):
        pxlsm.validate_pxlsm_call(**args)

def test_nudging_needs_a_positive_integer_analysis_interval():
    args = valid_call()
    args.update(soil_nudge=1)
    with pytest.raises(ValueError, match="analysis interval"):
        pxlsm.validate_pxlsm_call(**args)
    args["anal_interval"] = 3600
    pxlsm.validate_pxlsm_call(**args)

def test_model_dt_and_surface_interval_remain_distinct():
    args = valid_call()
    args.update(dt=10., dtbl=60., smois_init=2)
    pxlsm.validate_pxlsm_call(**args)

def test_pxlsm_uses_the_shared_no_ftz_no_fma_compile_route():
    from woof.core import kernels
    assert kernels.module_options("pxlsm") == ("-std=c++17", "--fmad=false")
    assert not any("ftz=true" in x for x in kernels.module_options("pxlsm"))
    source = kernels.module_source("pxlsm")
    assert "#define PA(a, b) __fadd_rn((a), (b))" in source
    assert "#define PPOW(x, y) gfk_pow((x), (y))" in source

def test_every_kernel_constant_word_matches_the_compiled_oracle():
    import re
    from woof.core import kernels
    source = (Path(kernels.__file__).parent / "pxlsm.cu").read_text()
    table = source.split("};", 1)[0]
    entries = re.findall(r"0x([0-9A-Fa-f]+)u,?\s*//\s*(\d+)\s+(\w+)\s*=", table)
    oracle = {line.split()[0]: int(line.split()[1], 16)
              for line in (ORACLE / "literals.txt").read_text().splitlines()}
    assert len(entries) == len(oracle) == 140
    for word, index, name in entries:
        assert int(word, 16) == oracle[name], (index, name)

def test_pxlsm_package_assets_are_included_in_distribution():
    import tomllib
    repo = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads((repo / "pyproject.toml").read_text())
    entries = manifest["tool"]["setuptools"]["package-data"]["woof"]
    import fnmatch
    for path in (repo / "woof/data/pxlsm").rglob("*"):
        if path.is_file() and "__pycache__" not in str(path):
            relative = str(path.relative_to(repo / "woof")).replace("\\", "/")
            assert any(fnmatch.fnmatch(relative, pattern) for pattern in entries), relative
