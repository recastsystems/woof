"""WRF v4.8.0 MYNN-SFC b66187f5 (mynn_sfclay_variant = "mynnsfc_b66187f").

The column code is the woof hex transcription of the same MYNN-SFC commit
with WRF's gravity.  The bitwise proof is the module replay in
tools/mynnsfc_wrf480_oracle (real A195-shape HRRR-recipe captures against
the native WRF 4.8.0 strict referee's own objects); these tests pin the
source words that replay depends on, the option refusals, and, on a GPU, the
committed native-referee words for one cropped real capture.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
UNIT = ROOT / "woof" / "core" / "kernels" / "mynnsfc_b66187f.cu"
ORACLE = ROOT / "tests" / "data" / "oracles" / "mynnsfc480"


def _unit() -> str:
    return UNIT.read_text(encoding="utf-8")


def test_unit_carries_wrf_gravity_words():
    text = _unit()
    # WRF module_model_constants g = 9.81; MPAS gravity 9.80616 must be gone.
    assert text.count("0x1.39eb860000000p+3f") == 9
    assert text.count("0x1.a1887a0000000p-4f") == 4      # g_inv = 1./grav
    assert text.count("0x1.761c140000000p-10f") == 1     # charnock/grav
    for mpas in ("0x1.39cc100000000p+3f", "0x1.a1b2560000000p-4f",
                 "0x1.7641960000000p-10f"):
        assert mpas not in text


def test_atan_is_the_glibc_239_word():
    from woof.core.mynnsfc_b66187f import module_source
    src = module_source()
    assert "#define SL_ATAN mynn_glibc_atanf" in src
    # The CORE-MATH header (glibc 2.41+) is not assembled.
    assert "CORE-MATH MIT transcriptions" not in src


def test_citations_name_wrf_480_tree():
    text = _unit()
    cites = re.findall(r"MYNN-SFC/(MPAS/)?module_sf_mynnsfc_(\w+)\.F90:\d+", text)
    assert len(cites) >= 200
    assert not re.search(r"MPAS-Model-rrfs|@1d6e370e", text)


@pytest.mark.parametrize("key,value,needle", [
    ("sf_mynn_sfcflux_land", 2, "qstar"),
    ("sf_mynn_sfcflux_land", 4, "ivegsrc"),
    ("sf_mynn_sfcflux_water", 6, "defines"),
    ("shalwater_z0", 1, "water_depth"),
])
def test_refusals_name_the_breakage(key, value, needle):
    from woof.config import ConfigurationRefused
    from woof.core.mynnsfc_b66187f import validate_mynnsfc_b66187f
    cfg = SimpleNamespace(sf_mynn_sfcflux_land=0, sf_mynn_sfcflux_water=1,
                          shalwater_z0=0)
    setattr(cfg, key, value)
    with pytest.raises(ConfigurationRefused, match=needle):
        validate_mynnsfc_b66187f(cfg)


def test_presence_masks_follow_the_wrf_call_sites():
    from woof.core import mynnsfc_b66187f as m
    bit = lambda name: 1 << m.PRESENCE.index(name)
    for name in ("restart", "flag_lsm", "lakemask", "water_depth", "ck",
                 "cka", "cd", "cda", "ustm", "shalwater_z0"):
        assert m.WRF_PRESENT & bit(name)
    for name in ("cycling", "ivegsrc", "vegtype", "sigmaf", "tsurf",
                 "stress", "hflx", "qflx", "fm"):
        assert not m.WRF_PRESENT & bit(name)
    for name in ("shalwater_z0", "lakemask", "water_depth"):
        assert not m.WRF_SEA_PRESENT & bit(name)
    assert len(m.FIELDS) == 68


def test_reference_map():
    from woof.core.mynnsfc_b66187f import MYNN_SFCLAY_BY_WRF_REFERENCE
    assert MYNN_SFCLAY_BY_WRF_REFERENCE == {
        "4.7.1": "wrf_461", "4.8.0": "mynnsfc_b66187f"}


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(p.stem for p in ORACLE.glob("*.capture.npz"))
                         if ORACLE.is_dir() else [])
def test_gpu_matches_native_referee_words(name):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # noqa: BLE001
        pytest.skip("no CUDA device")
    sys.path.insert(0, str(ROOT / "tools" / "mynnsfc_wrf480_oracle"))
    import replay
    cap = dict(np.load(ORACLE / f"{name}.npz"))
    native = dict(np.load(ORACLE / f"{name.removesuffix('.capture')}.native.npz"))
    gpu = replay.run_gpu(cap)
    result = replay.compare(cap, native, gpu)
    if int(cap["mode"]) == 1:
        # CH at sea-ice cells: WRF 4.8.0 blends the never-assigned wrapper
        # local CH_SEA (module_surface_driver.F:5631, :5681); no ice cells
        # are in these captures.
        assert not np.any(cap["xice"] >= cap["xice_threshold"])
    assert result["different"] == 0, {
        k: v for k, v in result["fields"].items() if v["different"]}
