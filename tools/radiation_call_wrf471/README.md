# WRF 4.7.1 radiation call phase replay

`ra_call_offset=-1` selects WRF's old fixed-step phase; zero is its default new
phase. For STEPRA=5 the calls are 1,5,10,... and 1,6,11,... respectively.
The offset also controls CAM absorption refresh and held solar bookkeeping on
continuation. Adaptive radiation timing keeps its time-based decision.

This selector adds no GPU operation between radiation calls. Changing phase can
change how many radiation calls fall inside a finite forecast. The scheduler's
cost and the radiation-call count are recorded by `profile_go.py` inside the
whole forecast; its two phase predicates alternate order on successive steps.

On the RTX PRO 6000 Blackwell real-input 60-step forecast, the interleaved
default and old-phase predicate medians were 0.941 and 0.9565 microseconds:
1.0165x, or 0.0155 microseconds more per scheduling decision. There is no
additional GPU step between radiation calls. This 12-minute case made 13
radiation calls with the old phase versus 12 with the default; native WRF
made the same calls. These timings measure the selector, not a radiation
scheme speed comparison. Evidence: the lane report's ra-final-old.stage.json.

Kept real-input captures and native outputs:
`~/weather/data/datasets/paper/wrf-ports/port-madwrf-raoffset/ra_call_offset/`.
The manifest there pins the strict native executable, source, commands and
hashes. Native WRF emits `CALL rrtmg_lw` after its own driver decision with
`debug_level=100`; no source instrumentation changes that decision.

Run `python tools/radiation_call_wrf471/replay.py CAPTURE_ROOT` to compare all
120 native decision words and the corresponding retained last-call indices.
This is a selector replay, not a radiation-flux or whole-model parity claim.
No native words are excluded. `prepare_cases.py SOURCE ROOT` prepares the two
12-minute, 12-second-step real-input native cases, with `radt=1` minute.

The whole-run profiler is invoked as:
`python tools/radiation_call_wrf471/profile_go.py STAGE.json CASE --outdir OUT --benchmark`.
It uses CUDA events around the existing radiation stage without synchronizing
at every step. Its CPU predicate timings include the same Python timer overhead
in both interleaved arms. It changes no model field.
