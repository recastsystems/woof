"""WSM6 fast numerics: plain operators, CUDA binary32 libm, lanes over levels; strict text untouched (CPU only)."""
import hashlib

import pytest

from woof.hex.mpasphys import wsm6_v842 as w

HEADER = ("__device__ float mp_exp(float x) {return thompson_aa_expf(x);}\n"
          "__device__ float mp_log(float x) {return thompson_aa_logf(x);}\n"
          "__device__ float mp_log10(float x) {return ubm_log10f(x);}\n"
          "__device__ float mp_pow(float x,float y) {return tc_pow(x,y);}\n")


def test_strict_text_is_the_column_per_thread_source():
    strict = w.cuda_source(HEADER, max_levels=55, numerics="strict")
    assert "mp_wsm6_run_coop" not in strict and "WsP<float>" in strict
    assert "__fadd_rn" in strict and "thompson_aa_expf(x);}" in strict


def test_fast_text_swaps_operators_maths_and_runs_lanes_over_levels():
    fast = w.cuda_source(HEADER, max_levels=55, numerics="fast")
    for op in "+-*/":
        assert f"{{return a{op}b;}}" in fast
    assert "return expf(x);" in fast and "return powf(x,y);" in fast
    assert "thompson_aa_expf(x);}" not in fast
    # level loops across lanes: mp_wsm6_run 25, the four slopes 1 each, sedimentation 18 + 19, radar 5
    assert fast.count("(lanes over levels)") == 71
    assert "slope_wsm6_coop(S, qrs_tmp" in fast and "mp_wsm6_run_coop(S,local" in fast
    assert "nislfv_rain_plm6_coop(S, idim" in fast and "slope_graup_coop(S, qr2" in fast
    assert "WsP<float>" not in fast  # native one-struct-per-column workspace


def test_strict_coop_is_forced_only_on_request(monkeypatch):
    monkeypatch.setenv("WOOF_HEX_WSM6_COOP", "1")
    assert "mp_wsm6_run_coop" in w.cuda_source(HEADER, max_levels=55, numerics="strict")
    monkeypatch.setenv("WOOF_HEX_WSM6_COOP", "0")
    assert "mp_wsm6_run_coop" not in w.cuda_source(HEADER, max_levels=55, numerics="fast")


FUNCTION = """__device__ void demo(Wsm6State &S, float *a, float *b, int kts, int kte) {
int i;
int k;
float x;
float s;
  for (k=kts; k<=kte; k+=1) { // demo:1
  x = a[((k)-(kts))]; // demo:2
  b[((k)-(kts))] = x; // demo:3
   } // demo:4
  for (k=kts; k<=kte; k+=1) { // demo:5
  b[((k)-(kts))] = a[((k+1)-(kts))]; // demo:6
   } // demo:7
  for (k=kts; k<=kte; k+=1) { // demo:8
  b[((k+1)-(kts))] = b[((k)-(kts))]; // demo:9
   } // demo:10
  s = 0.f; // demo:11
  for (k=kts; k<=kte; k+=1) { // demo:12
  s = add(s, a[((k)-(kts))]); // demo:13
   } // demo:14
  b[((1)-(kts))] = s; // demo:15
}
"""


def test_only_own_level_loops_go_across_lanes():
    out = w._coop_function(FUNCTION, "demo", ())
    # demo:1 and demo:5 (writes b at its own level, reads a anywhere); demo:8 writes b at k+1 and the
    # demo:12 sum carries s: both stay in program order, the shared write on lane 0
    assert out.count("(lanes over levels)") == 2
    assert out.count("if (lane == 0)") == 1  # demo:8 to demo:15, one lane-0 group
    assert "s = __shfl_sync(0xffffffffu, s, 0);" in out


def test_a_value_left_behind_by_a_level_loop_is_refused():
    leftover = FUNCTION.replace("  s = 0.f; // demo:11", "  s = x; // demo:11")
    with pytest.raises(ValueError, match="left behind"):
        w._coop_function(leftover, "demo", ())


def test_pack_inventory_composes_wsm6_per_numerics():
    strict = hashlib.sha256(w.cuda_source(HEADER, max_levels=59, numerics="strict").encode()).hexdigest()
    fast = hashlib.sha256(w.cuda_source(HEADER, max_levels=59, numerics="fast").encode()).hexdigest()
    assert strict != fast
    with pytest.raises(ValueError):
        w._fast_maths(HEADER.replace("thompson_aa_logf", "other_logf"))
