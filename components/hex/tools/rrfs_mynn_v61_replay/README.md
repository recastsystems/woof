# MYNN v6.1 module replay (RRFS 44c0849 host)

Referee tools for the selectable MYNN suite versions (`config_mynn_version` on the `rrfs-44c0849` pin set).
CPU Fortran here is a referee tool only; it is never on a run path.

## MYNN-SFC v6.1.0 (`sf_mynnsfc_v610`)

1. `sfc_observer.py MPAS_44C0849_TREE LANE` writes the external argument observer for `mynnsfc_driver`
   (`make` in `LANE/sfc-observer/mynnsfc_driver`).
2. `sfc_capture.py c1|c2|c3 LANE` runs the RRFS referee of record (44c0849 strict + NULLIFY x2 + RUC ilnb=0 x2,
   a128a1e8, labelled patched) on x1.2562 hrrrv5, GFS 2023-03-31 12Z, 1 process, three 720 s steps, ordinary and
   observed, and requires 0 different output words between them (observer passivity).
   - c1: cold start.
   - c2: restart from c1's +36 min restart.
   - c3: the cycled start (`config_do_DAcycling`). Native 44c0849 segfaults in RRTMG SW (`taumol_sw`) at step 1
     of this start, before the surface layer runs, so c3 has no whole-model capture. The replay instead re-runs
     c2's captured inputs with restart=false and cycling=true.
3. `sfc_referee.py v600|v610 OUT` builds standalone MYNN-SFC v6.0.0 (b66187f) or v6.1.0 (a038a9e) with the
   44c0849 strict compile lines (v6.1.0 adds the host's `-Dmpas`) behind the bind(c) bridge
   (`tests/mpas_rrfs_mynnsfc/native_mynnsfc_bridge.F90`, and `native_mynnsfc_bridge_v610.F90` without `flag_iter`).
4. `WOOF_HEX_NUMERICS=strict sfc_replay.py LANE OUT.json [--timing N]` runs the replay. It needs 0 different words
   in every comparison:
   - the standalone v6.0.0 build against the captured whole-model POST (harness witness);
   - `sf_mynnsfc_v600` against that POST;
   - `sf_mynnsfc_v610` against the standalone v6.1.0 build, through both the plain and the resident-door launch
     arguments.

## MYNN-EDMF v6.1.1 (`bl_mynnedmf_v611`)

`src/hexcore/mpasphys/rrfs/bl_mynnedmf_v611.py` is generated, not hand-edited: NCAR/MYNN-EDMF v6.1.1
(2b2da61b05a8, Apache-2.0) through the same Fortran-to-CUDA translator as the pinned c6d08709 module.

1. Generation, from a clean checkout of NCAR/MYNN-EDMF at 2b2da61:
   `tests/rrfs_mynnedmf/generate_backend.py --layout v6 --root MYNN-EDMF --target src/hexcore/mpasphys/rrfs/bl_mynnedmf_v611.py --receipt R.json --model-pin 44c0849 --scheme-pin 2b2da61`.
   The v6 layout adds, opt-in: the MPAS preprocessor table (`-Dmpas`, as the RRFS host's physics Makefile passes
   COREDEF), ALLOCATE/DEALLOCATE of arena arrays, assumed-shape dummies (memory bounds, or a launch extent for a bare
   `:`), FUNCTION ... RESULT, optional written scalars (`MynnOptionalRef`), MAXVAL(MASK=), and routines reachable
   from `mynnedmf_driver` only. Regenerating the pinned module with the extended translator is byte-identical.
   `edmf-sources.json` lists every upstream file with its hash.
2. Captures on the RRFS referee of record (a128a1e8, labelled patched), x1.2562 hrrrv5, GFS 2023-03-31 12Z,
   1 process, three 720 s steps each, with observer passivity: `edmf_capture_chain.sh` (`edmf_capture.py`).
   - C1 cold start; C2 restart from C1's +36 min restart; C3 the RRFS DA-cycled start (do_DAcycling true,
     do_restart false, the da_state stream as the initial state, written by a jedi_da twin of C1).
   - Each set captures `mynnedmf_driver` PRE/POST and `MPAS_to_physics` (its `pres2_hyd_p` is the v6 driver's `pint`,
     as ufs-community/MPAS-Model noaa/develop de016e0 driver_pbl.F:1748-1828 binds it).
3. Referees: `edmf_referee.py build --pin pinned|v6` builds the tag's MPAS driver standalone with the 44c0849
   gnu-strict compile lines (v6 adds `-Dmpas`) behind a generated bind(c) bridge.
4. Replay (`edmf_replay.py`):
   - `witness`: the standalone pinned build against the captured native POST (every defined word; the five TKE-budget
     outputs the c6d08709 driver copies from never-written locals with tke_budget=0 are named and excluded);
   - `referee`: the standalone v6.1.1 build on the v6 arguments `bl_mynnedmf_v611.native_driver_arguments` builds;
   - `gpu`: the strict image against that POST at lane widths 1, 4 and 16.
5. `edmf_witness_undefined.py`: the C3 first step reads an undefined native local (below).
6. `edmf_time_stage.py`: v6.1.1 against c6d08709, fast numerics, interleaved on one card, CUDA events.

Results (RTX 5090, strict):

- Harness witness: 0 different of 15,771,675 words per set (C1, C2, C3).
- C1 and C2: 0 different of 18,077,475 words per set at every lane width.
- C3 steps 2-3: 0 different of 12,051,650 words at every lane width.
- C3 step 1 is native-undefined. At initflag = 1 with restart false the v6.1.1 driver stages no turbulence state
  (src/module_bl_mynnedmf_driver.F90:646-670), and with cycling true the core reads `maxval(qke1(:))` of that
  never-written automatic array (src/module_bl_mynnedmf.F90:462-467) to decide whether to initialise TKE.
  - The same standalone referee on the same inputs gives different words with a different stack (631 words).
  - With one labelled witness edit, `qke1(kts:kte) = zero` at the staging point (the value the zeroed arena
    holds), the referee equals the GPU module in every word.
  - The default referee's garbage usually skips the initialisation, so native starts the cycled PBL from zero TKE
    (exch_h at most 0.047 m2/s over the mesh); the GPU module initialises it as on a cold start.

Forecast and speed (2026-11-09, RTX 5090, fast numerics):

- 24 h x1.40962 120 km hrrrv5, GFS 2023-03-31 12Z, 1 process, mynn-v6.1 beside the pinned default (same deck and
  init): both complete (120 steps). Forecast runtime 42.9 s vs 47.2 s. The rw_wrfbatch sheets at +6/+12/+18/+24 h look
  normal: the same systems in the same places, with cell-scale differences in convective echoes. The domain
  series (night-land 2 m T and Td, PBLH, HFX; mynn-sfc-v6.1.0 as a third arm) puts the EDMF change at
  0.03-0.08 K cooler and 0.08-0.12 C lower 2 m Td at night over land, consistent with dew/frost deposition no
  longer being deleted (E9). The SFC change alone is close to zero.
- Option off: the pinned default on this branch against the branch base (5527f559), same 1 h run: 76,804,109 words,
  0 different.
- MYNN-EDMF stage, v6.1.1 / c6d08709, interleaved CUDA events on the same inputs: median 0.82 at 2,562 columns and
  1.05 at 40,992 columns (each column repeated 16 times), against the 1.2 bar.
- 8 rank-threads on one 32 GB card is refused by admission at 120 km (32.2 GB priced), so the forecast ran as
  1 process.

Default (2026-11-09): `mynn-v6.1` is the default on the rrfs-44c0849 route (2026-11-08, latest
upstream as default). "pinned default" above means the default when those runs were made; `pinned` is now
named explicitly, and the bitwise gates and paper reruns against native 44c0849 pass
`--physics config_mynn_version=pinned` on the hex side.
