"""MMM-physics generation for hex's YSU and revised MM5 (mmm_version.py; MMM a4baf7f -> 550b5b4)."""
from dataclasses import dataclass, field
from types import MappingProxyType

import pytest

from woof.hex.mpasphys import mmm_version as mmm
from woof.hex.mpasphys import sfclay_mm5_v842 as mm5
from woof.hex.mpasphys import ysu_v842 as ysu


@dataclass(frozen=True)
class _Config:
    selectors: dict
    backends: dict
    mode: str = "correct"
    mmm: object = None
    inactive: tuple = ()

    def active(self, slot):
        return slot not in self.inactive


def _config(pbl="bl_ysu", sfc="sf_monin_obukhov_rev", backend="hex-cuda", **kw):
    return _Config({"pbl": pbl, "surface_layer": sfc}, {"pbl": backend, "surface_layer": backend}, **kw)


def test_defaults_are_latest_in_correct_mode_and_the_pin_in_exact_mode():
    assert mmm.resolve({}, mode="correct")["mmm_version"] == "550b5b4"
    assert mmm.resolve({}, mode="exact")["mmm_version"] == "a4baf7f"
    assert mmm.resolve({"mmm_version": "pinned"}, mode="correct")["mmm_version"] == "a4baf7f"
    assert mmm.resolve({"mmm_version": "wrf_480"}, mode="exact")["mmm_version"] == "550b5b4"
    assert not mmm.resolve({}, mode="correct")["kim_tofd"]


def test_kim_tofd_needs_550b5b4_and_the_revised_layer():
    with pytest.raises(ValueError, match="kim_tofd needs"):
        mmm.resolve({"mmm_version": "a4baf7f", "kim_tofd": "true"}, mode="correct")
    with pytest.raises(ValueError, match="sf_monin_obukhov_rev"):
        mmm.configure(_config(sfc="sf_mynn"), {"kim_tofd": "true"})
    assert mmm.configure(_config(), {"kim_tofd": ".true."}).mmm["kim_tofd"]


def test_native_cpu_slots_run_the_pin_and_refuse_a_named_550b5b4():
    assert mmm.configure(_config(backend="native-cpu"), {}).mmm["mmm_version"] == "a4baf7f"
    with pytest.raises(ValueError, match="GPU port only"):
        mmm.configure(_config(backend="native-cpu"), {"mmm_version": "550b5b4"})


def test_unknown_key_and_bad_version_are_refused():
    with pytest.raises(ValueError, match="unknown MMM-physics options"):
        mmm.resolve({"mmm_versoin": "550b5b4"}, mode="correct")
    with pytest.raises(ValueError, match="mmm_version must be"):
        mmm.resolve({"mmm_version": "63c806d"}, mode="correct")


def test_ysu_550b5b4_is_the_pinned_text_with_exactly_the_top_down_cap():
    header = ysu.shared_device_math_header()
    pinned, latest = ysu.cuda_source(header), ysu.cuda_source(header, "550b5b4")
    assert pinned == ysu.cuda_source(header, "a4baf7f")
    a, b = pinned.splitlines(), latest.splitlines()
    assert len(a) == len(b)
    changed = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(changed) == 2  # the cited comment and the one statement
    assert "ysu_minf(ysu_mul(0x1.99999a0000000p+0f, tmp1), 0x1.99999a0000000p-2f)" in changed[1][1]


def test_mm5_form_drag_off_is_the_pinned_kernel_text():
    assert mm5._cuda_text(()) == mm5._cuda_text((), False)
    on = mm5._cuda_text((), True)
    assert "MMM 550b5b4:716" in on and "MMM 550b5b4:780" in on
    # upstream sf_sfclayrev.F90:712 (dx_factor from itself, never used) is not transcribed
    assert "dx_factor" not in on


def test_slot_call_without_a_choice_runs_the_pin():
    class Call:
        pass
    assert mmm.of(Call())["mmm_version"] == "a4baf7f"
    call = Call()
    call.mmm = MappingProxyType({"mmm_version": "550b5b4", "kim_tofd": False, "tofd_factor": 0.003})
    assert mmm.of(call)["mmm_version"] == "550b5b4"
