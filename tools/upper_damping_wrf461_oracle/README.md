# Upper-damping column oracle against WRF 4.6.1

Grades WOOF's two upper-damping pieces against WRF 4.6.1's own compiled Fortran,
word for word:

- `damp_opt = 3` (Klemp, Dudhia and Hassiotis 2008 implicit w damping), which WRF
  applies inside `advance_w` (dyn_em/module_small_step_em.F:1445-1458) and WOOF inside
  the `advance_w_phi` / `advance_w_phi_msf` CUDA kernels (woof/core/kernels/acoustic.cu).
- `w_damping = 1`, WRF `w_damp` (dyn_em/module_big_step_utilities_em.F), WOOF
  `dycore.apply_w_damping` and the `w_damp` kernel (woof/core/kernels/openbc.cu), plus
  the `max_vert_cfl` / `max_horiz_cfl` it returns (WOOF's `w_cfl_stat` reduction).

The fork-only `upper_wind_limiter_form = noaa_wrf39` is not in WRF 4.6.1 and is not
graded here.

## Pieces

| file | what it does |
|---|---|
| `columns.py` | 96 columns (12 x 8, periodic), 49 levels on the real hybrid coordinate of the committed small-step fixture. Rows: convective (2), stable night, snow and ice, ocean (ht = 0), high terrain (2, surfaces 1.5 to 4.3 km), edge cases (a column at rest, signed zeros, w of 1e-30, 60 m/s at the top, alternating signs, a 200 m raised top). Each column is integrated hydrostatically from its own surface, so the damping layer `htop - zdamp` starts at a different height in every column. `edge_zdamp` picks a zdamp whose float32 `htop - zdamp` equals one level's height exactly (WRF's `hk .ge. hbot` boundary). |
| `build.py` | Compiles the pinned, unmodified WRF 4.6.1 source twice: `O0` (-O0 -ffp-contract=off) and `stock` (WRF's own GNU flags, -O2 -ftree-vectorize -funroll-loops, plus -fPIC). advance_w comes in with the other seven small-step routines through the generated C wrappers of `tools/smallstep_wrf471_oracle/build.py`; `w_damp` is cut verbatim out of module_big_step_utilities_em.F into a one-routine module with data-only stand-ins for the framework names it uses. `--as-built` compares the floating-point instruction histogram and external symbols of advance_w and w_damp in a compiled wrf.exe tree with the stock build here. |
| `run_oracle.py` | 8 advance_w cases x 2 map variants (unity map runs `advance_w_phi`, map factors run `advance_w_phi_msf`), and 5 w_damp cases x 2 time steps x 2 map variants. Every output word is compared: a, alpha, gamma, w, ph, t_2ave, muts, muave, p, al and the history-weighted p for advance_w; rw_tend, max_vert_cfl and max_horiz_cfl for w_damp. Writes a JSON receipt and, with `--fixture`, the test fixture. |
| `probe_signed_zero.py` | Drives one column to an all-negative-zero acoustic state under a negative w_save, to see whether WRF's below-layer identity `(w - 0*m*w_save)/1` (which WOOF skips) can change a signed zero. |

advance_w cases: `hrrr_damp` (dampcoef 0.2, zdamp 5000 m, dtau 4.5 s = 18 s / 4),
`hrrr_control` (damp_opt 0), `deep_weak` (0.05, 8000 m, 2 s), `shallow_strong` (1.0,
2500 m, 1 s), `whole_column` (zdamp 25 km, every level damped), `hbot_on_level`,
`rigid_lid` (top_lid), `dry_loading` (no moisture loading).
w_damp cases: (w_crit_cfl, zadvect_implicit) = (1, 0), (1, 1), (2, 0), (2, 1), (1.5, 1),
at dt 18 s and 30 s, with Omega cells whose Courant number is exactly 1.0, 1.5 and 2.0 to
the last bit (WRF's test is strict, so those must not be damped).

## Rerun (box W1, CPU build, one GPU through the mutex)

```bash
T=/work/pverify/upper-damping/src; source /work/pverify/base/env.sh; cd $T
python tools/upper_damping_wrf461_oracle/build.py /work/pverify/wrf-build/src/WRF-4.6.1 \
    /work/pverify/upper-damping/oracle --as-built /work/pverify/wrf-build/serial/WRF-4.6.1
card-lease upper-damping --cards 1 --est 600 --wait 1800 bash -c '
  source /work/pverify/base/env.sh; export T=/work/pverify/upper-damping/src
  export PYTHONPATH=$T:$T/tests; cd $T
  GPUWM_WRF_EXACT=1 python tools/upper_damping_wrf461_oracle/run_oracle.py \
      --oracle /work/pverify/upper-damping/oracle --output strict.json \
      --fixture tests/data/upper_damping_wrf461.npz
  python tools/upper_damping_wrf461_oracle/run_oracle.py \
      --oracle /work/pverify/upper-damping/oracle --output default.json
  GPUWM_WRF_EXACT=1 python -m pytest -q tests/test_upper_damping_wrf461_oracle.py
  python -m pytest -q tests/test_upper_damping_wrf461_oracle.py'
```

The fixture test (`tests/test_upper_damping_wrf461_oracle.py`) needs no Fortran: it
replays WOOF on the committed inputs and compares with the committed WRF words.

## Receipts

`receipts/strict.json`, `receipts/default.json`, `receipts/probe-strict.json` and
`receipts/build-receipt.json` are the runs of record (RTX PRO 6000 Blackwell, NVRTC 12.9,
gfortran 13.3.0, glibc 2.39).

`ph_unchanged` in the advance_w receipts is a harness self-check, not a WRF comparison:
it compares WOOF's zero-length diagnosis launch (dtau = 0, used only to isolate the
equation of state) with its own input phi. It differs only on the 49 negative-zero phi
words of edge column (7, 1), where `-0 + 0` gives `+0` (0 ULP by value). The summary
lists it apart from the graded outputs.
