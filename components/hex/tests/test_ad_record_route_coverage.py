# SPDX-License-Identifier: Apache-2.0
"""A recording runs on the route whose kernels have derivatives, says which, and refuses a plan with a kernel kind that
has none when it is sealed (no card).

THE BREAKAGE (2026-11-18): a 12 km plan recorded with the 0.6.0 recorder sealed and replayed exactly and then
failed the release's own check (whole-state dot 0.228 over 15 minutes; the gradient promised 0.71 to 0.77 of the real
change): the forecast ran cupy_getitem_mask (the transport's density check, a float gather with no derivative), and a
recording through --init on the 12 km row also ran the four-kernel regional column solve, which has no derivative
rows. Nothing said so until a window ran.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from woof.hex import ad_doors as d
from woof.hex.ad.record import chain, coverage, env, route


def _plan(path: Path, kinds: dict[str, int]) -> Path:
    ops = [{"kind": "kernel", "function": k} for k, n in kinds.items() for _ in range(n)]
    path.write_text(json.dumps({"blocks": {"step_01": ops + [{"kind": "synchronization"}]}}))
    return path


def test_the_default_route_is_the_derivative_forms():
    assert route.default() == "derivative"
    words = route.words("derivative")
    # the plain forms the door table already asked for, and the two a 12 km --init recording lacked
    for knob, value in {"WOOF_HEX_PAIR_ADD": "plain", "WOOF_HEX_OZONE_MONTHS": "all", "WOOF_HEX_EDGE_HOIST": "0",
                        "WOOF_HEX_MYNNEDMF_LANES": "16", "WOOF_HEX_REGIONAL_ACOUSTIC_SPLIT": "0",
                        "WOOF_HEX_MEMBER_MEMORY": "single", "WOOF_HEX_MEM_ACOUSTIC": "0"}.items():
        assert words[knob] == value
    assert route.words("forecast") == {}
    assert set(route.table()["routes"]["derivative"]["why"]) == set(words)


def test_the_route_words_win_in_every_forecast_of_the_recording(tmp_path, monkeypatch):
    monkeypatch.setenv("WOOF_HEX_REGIONAL_ACOUSTIC_SPLIT", "1")
    monkeypatch.setattr(env, "runtime_env", lambda card: {"CUDA_VISIBLE_DEVICES": str(card), **__import__("os").environ})
    o = types.SimpleNamespace(work=tmp_path, compile_workers=2, xdg=None, lam=None, route="derivative")
    assert chain._base_env(o, "0")["WOOF_HEX_REGIONAL_ACOUSTIC_SPLIT"] == "0"
    o.route = "forecast"
    assert chain._base_env(o, "0")["WOOF_HEX_REGIONAL_ACOUSTIC_SPLIT"] == "1"


def test_the_recorder_command_takes_the_route():
    out = subprocess.run([sys.executable, "-m", "woof.hex.ad.record", "record", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "--route" in out.stdout and "derivative" in out.stdout


def test_the_door_hands_its_route_to_the_recorder(tmp_path, monkeypatch):
    row = dict(d.load_table()["rows"]["hrrrv5-any"], mesh="x1.40962", dt_s=450)
    for name in ("grid.nc", "oro.nc", "init.nc"):
        (tmp_path / name).write_bytes(b"x")
    seen = {}

    def run(argv, stdout=None, stderr=None, env=None, **kw):
        seen["argv"] = argv
        return types.SimpleNamespace(returncode=7)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.delenv(d.RECORDER_ENV, raising=False)
    monkeypatch.setattr(d, "resolve_runner", lambda named: tmp_path / "ad-binder")
    for asked, want in ((None, "derivative"), ("forecast", "forecast")):
        args = types.SimpleNamespace(grid=str(tmp_path / "grid.nc"), oro=str(tmp_path / "oro.nc"), steps=None,
                                     out=str(tmp_path / f"out-{want}"), init=str(tmp_path / "init.nc"), card="0",
                                     cards=None, runner=None, record_route=asked)
        with pytest.raises(d.AdDoorError, match="recording from"):
            d.record_plan(args, tmp_path, row)
        at = seen["argv"].index("--route")
        assert seen["argv"][at + 1] == want


def test_a_kind_with_no_derivative_is_named(tmp_path):
    plan = _plan(tmp_path / "plan.json", {"acoustic_coefficients_v841": 3, "cupy_getitem_mask": 15,
                                          "acoustic_column_sweep_regional_v841": 2,
                                          "cupy_isfinite__float32_bool": 15, "cupy_maximum__int32_int32_int32": 15,
                                          "cupy_copy__bool_bool": 30, "upper_cam_f32": 4, "upper_rayleigh_f32": 4,
                                          "ntiedtke_cuascn": 1, "tempo_warp_steps": 1, "validate_density_v841": 1})
    result = coverage.check(plan)
    assert result["missing"] == {"cupy_getitem_mask": 15, "acoustic_column_sweep_regional_v841": 2}
    assert result["by"]["checked"] >= 1 and result["by"]["no-float-word"] >= 3 and result["by"]["rule"] == 2
    assert any(x.startswith("ntiedtke_cuascn (hrrrv5-q12") for x in result["held_by_name"])
    words = coverage.refusal(result)
    assert "cupy_getitem_mask (15 launches)" in words and "nothing was sealed" in words


def test_the_example_plans_table_holds_neither_defect_kind():
    doc = json.loads(coverage.KINDS_FILE.read_text())
    assert len(doc["plans"]) == 3 and len(doc["kinds"]) > 200
    for kind in ("cupy_getitem_mask", "acoustic_column_pre_regional_v841", "acoustic_column_sweep_regional_v841"):
        assert kind not in doc["kinds"]
    assert "acoustic_column_solve_regional_v841" in doc["kinds"] and "validate_density_v841" in doc["kinds"]


def test_float_results_are_not_taken_for_bool_or_integer():
    # kinds no example plan runs, so the name rule alone decides
    for kind in ("cupy_greater__float64_float64_bool", "cupy_copy__int16_int64", "cupy_bitwise_or__uint8_uint8_uint8",
                 "cupy_copy__float64_float32x", "cupy_subtract__float64_float64_float64", "cupy_getitem_mask"):
        assert kind not in coverage.checked_kinds()
    assert coverage.why_covered("cupy_greater__float64_float64_bool") == "no-float-word"
    assert coverage.why_covered("cupy_copy__int16_int64") == "no-float-word"
    assert coverage.why_covered("cupy_bitwise_or__uint8_uint8_uint8") == "no-float-word"
    assert coverage.why_covered("cupy_subtract__float64_float64_float64") is None
    assert coverage.why_covered("cupy_getitem_mask") is None


def test_the_seventy_two_step_cap_is_gone():
    root = Path(chain.__file__).parent
    assert "step<=72" not in (root / "native" / "driver_trace_cupti_held.c").read_text()
    assert "digest_step <= 72" not in (root / "tools" / "catalog_v10.py").read_text()
    assert "max_steps > 72" not in (root / "tools" / "make_partial_plan.py").read_text()


def _launch(kind: str, addresses: list) -> dict:
    """A kernel launch whose pointer arguments name these (allocation, offset) addresses, after two scalar words."""
    args = [{"kind": "bytes", "bytes_hex": "3b000000"}, {"kind": "bytes", "bytes_hex": "83220000"}]
    args += [{"kind": "device", "target": {"allocation": a, "offset": o}} for a, o in addresses]
    return {"kind": "kernel", "function": kind, "arguments": args}


def test_a_kind_run_on_its_own_input_is_refused(tmp_path):
    # THE BREAKAGE (2026-11-18): with the dycore's short array lifetimes recover_edges_f32 wrote flux_u
    # (argument 7) over its input ru_avg_p (argument 5); its reverse rule adds into b_ru_avg_p and then clears
    # b_flux_u, so on one address the input's share was lost: whole-state dot 0.239 after one step at 12 km.
    alias = [_launch("recover_edges_f32", [(49, 0), (425, 6604), (391, 0), (398, 27), (391, 0)])]
    apart = [_launch("recover_edges_f32", [(49, 0), (425, 6604), (391, 0), (398, 27), (367, 0)])]
    # a pair the example plans themselves run (theta_edge_flux_f32's arguments 6 and 7) is not refused
    known = [_launch("theta_edge_flux_f32", [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (5, 0)])]
    for ops, refused in ((alias, True), (apart, False), (known, False)):
        plan = tmp_path / "plan.json"
        plan.write_text(json.dumps({"blocks": {"step_01": ops, "step_02": ops}}))
        result = coverage.check(plan)
        assert coverage.refuses(result) is refused
    plan.write_text(json.dumps({"blocks": {"step_01": alias, "step_02": alias}}))
    result = coverage.check(plan)
    assert result["missing"] == {} and result["aliased"] == {"recover_edges_f32": {"pairs": [[4, 6]], "launches": 2}}
    words = coverage.refusal(result)
    assert "recover_edges_f32 (arguments 4/6 at one address, 2 launches)" in words and "nothing was sealed" in words


def test_the_example_plans_never_run_the_recover_kinds_on_their_own_inputs():
    doc = json.loads(coverage.KINDS_FILE.read_text())
    for kind in ("recover_cells_f32", "recover_edges_f32", "recover_interfaces_f32"):
        assert kind in doc["kinds"] and "aliased_args" not in doc["kinds"][kind]
    assert doc["kinds"]["theta_edge_flux_f32"]["aliased_args"] == [[6, 7], [8, 9]]


def test_the_derivative_route_keeps_the_earlier_dycore_lifetimes():
    from woof.hex.cuda_acoustic_v841 import short_lifetimes_enabled
    assert route.words("derivative")["WOOF_HEX_MEM_ACOUSTIC"] == "0"
    assert "recover_edges_f32" in route.table()["routes"]["derivative"]["why"]["WOOF_HEX_MEM_ACOUSTIC"]
    old = __import__("os").environ.pop("WOOF_HEX_MEM_ACOUSTIC", None)
    try:
        assert short_lifetimes_enabled()   # the forecast's own default is unchanged
        __import__("os").environ["WOOF_HEX_MEM_ACOUSTIC"] = "0"
        assert not short_lifetimes_enabled()
    finally:
        __import__("os").environ.pop("WOOF_HEX_MEM_ACOUSTIC", None)
        if old is not None:
            __import__("os").environ["WOOF_HEX_MEM_ACOUSTIC"] = old
