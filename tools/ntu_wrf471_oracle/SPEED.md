# NTU measured cost and opt-in scope

`ntu-mp56` (`mp_physics=56`) is an explicit component override. It must
never become a default. Its native work exceeds the nearest existing
multi-moment scheme, MY2 (`mp_physics=9`), on the captured real forecast
tile. The 1.2x speed bar is not met.

On one shared RTX PRO 6000 Blackwell Server Edition, a single batch of
seven alternating before/MY2/after/MY2 calls gives median NTU 81.031487 ms
and MY2 0.365760 ms: **221.542779x**. Copies and compilation are outside
the events. A proposed fast-only reference-alias change increased NTU to
104.615074 ms and was reverted. The final runtime source is byte-identical
to the pre-change source. The older RTX 4090 measurement is 66.827202 ms
versus 0.189440 ms, or 352.761840x. These are different cards and shared
loads, not a before/after speedup.

An actual GNU Fortran 12.5 REAL32 profile of the unchanged WRF v4.7.1
module, with observation-only counters, records **274,625 native adaptive
iterations**, at most 299 in one outer 20-second step. The earlier count
397 is the sum over outer steps for the stiffest cell. The native call
profile records 80,568 SMALL_DT calls, 3,280,736 DYEQU calls, 222,642 GAMMP
calls and 31,871 ACTIVA calls. MY2 does not run these NTU aerosol/shape
and adaptive moment closures. The profiled NTU call uses 0.146541 CPU
seconds versus MY2 0.018914 seconds under the same profile flags. These
include profiling overhead; they are evidence of extra native work,
not a GPU lower bound. The counted native return is still 0/2,048,705
differing words on the original packet.

The native recurrence and step limits are module_mp_ntu.F:2230-2244 and
4103-4133; the SMALL_DT body begins at 3267. WRF REAL is 32-bit here.
GAMLN explicitly declares DOUBLE PRECISION at 648, promotes XX with DBLE
at 655, and converts back with SNGL at 667. ACTIVA also declares explicit
double temporaries at 2697. No blanket binary64 exception is claimed.

A separate Nsight Systems capture finds three NTU kernels, no copies or
host synchronization between them, cached constants and pointer tables,
and prephysics accounting for 98.7% of the kernel time. That capture used
a separately rebuilt real packet, not the exact replay packet. Nsight
Compute on the original packet, RTX 4090, records prephysics at 255
registers/thread, 608 local bytes/thread, 333,691 spill requests, 16.67%
theoretical occupancy and 5.76% achieved occupancy. Average active threads
per warp are 4.13. DRAM use is only 0.06% of peak. These implementation
and workload limits remain. This evidence does not prove all 221.54x is
unavoidable physics cost.

The speed-lane report carries commands, configs, raw event samples,
native profile sources, Nsight reports, strict replay and whole-forecast
results at `WRF-PORTS-2026-11-01/spd-mp56-ntu.md`, with evidence under
`evidence/spd-mp56-ntu/`. Strict assembled source remains unchanged.
