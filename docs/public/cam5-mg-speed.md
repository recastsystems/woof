# CAM5.1 MG performance option

CAM5.1 Morrison–Gettelman microphysics is an explicit `mp_physics = 11`
option. It is off by default. Its measured fast stage cost is 21.32 times
Morrison MP10 on a shared RTX PRO 6000 Blackwell Server Edition. Select it
when the CAM cloud fraction and aerosol physics are needed. This row uses
the measured native physics cost exception; it does not meet the 1.2 times
speed target.

A single interleaved series on the same real 124 by 198 by 49 tile gave
these median CUDA event times. Compilation, allocation and input reset
were outside the events. The card had two other compute processes, so the
absolute times and the size of the gain remain subject to contention.

| Stage | Before, ms | After, ms | After / MP10 |
|---|---:|---:|---:|
| CAM5.1 MG | 226.60 | 177.11 | 21.32 |
| Morrison MP10 | 8.31 | 8.31 | 1.00 |

The fast path now reuses saturation and cloud fraction results only when
their complete changing binary64 inputs are bit-identical. Fixed pressure,
ice mass, land and snow inputs remain local to the call. The cache rejects
NaN keys and distinguishes signed zeros. It uses the existing device
workspace and leaves strict preprocessing unchanged. Fast output changed
109 of 26,589,816 words in the timing tile, consistent with its separate
compiler arithmetic policy; strict replay is checked separately.

The stock native WRF call was also profiled on that tile. CPU time excludes
MPI startup and input/output. The phase instrument reproduces all
26,589,816 native output words and the entire original reference packet.

| Native work | CPU seconds |
|---|---:|
| MG cloud macrophysics | 0.481078 |
| MG aerosol activation and nucleation | 1.484738 |
| MG microphysics | 1.809233 |
| Whole MG driver | 4.139827 |
| Whole MP10 driver | 0.805217 |

The cloud and aerosol phases alone cost 2.44 times the whole MP10 call.
They are extra native equations, measured at the calls to
`Macrophysics_simple`, `microp_aero_ts` and `mmicro_pcond` in
`phys/module_mp_cammgmp_driver.F`. Native CAM explicitly imports
`r8 => shr_kind_r8` in `phys/module_cam_mp_cldwat2m_micro.F:22` and
`phys/module_cam_mp_microp_aero.F:26`;
`phys/module_cam_shr_kind_mod.F:20` defines it with
`selected_real_kind(12)`. The outer WRF fields remain default REAL32.
Binary64 alone is not the exception proof: the measured extra cloud and
aerosol phases are.

This does not attribute the full GPU ratio to physics. Nsight Compute
also shows an implementation cost: 255 registers per thread, local-memory
spills and low achieved occupancy in the column kernel. Nsight Systems
shows two chunk launches, device health work and no per-call table upload.
Further faithful GPU work can reduce the remaining cost.

The strict path retains the correctly rounded MIT CORE-MATH cosine and
arccosine. The untouched native replay still has the 71 previously
declared rounding differences. Correctly rounded native math is a separate
instrument under the existing ruling, not an untouched-native parity
claim. See `cam5-suite.md` for that ruling and the suite interface.

Exact commands, raw samples, profiles, replay results and the whole fast
forecast are recorded in `spd-cam5-mg.md` and `evidence/spd-cam5-mg/` under
the private WRF port reports. The native replay packets and manifest are
retained in `~/weather/data/datasets/paper/wrf-ports/spd-cam5-mg/`.
