# WRF 4.7.1 MYNN sub-options

Nondefault WRF 4.7.1 MYNN controls select `bl_mynn_version="wrf_471"`.
Ordinary forecasts retain `wrf_461` and its defaults. The 4.7.1 source has
changed throughout the solver; the two generations are not aliases.

The source is WRF commit `f52c197ed39d12e087d02c50f412d90d418f6186`.
`module_bl_mynnedmf.F` SHA256 is
`6e4a7d5b35ce46f01591f2c1d58e545380d546e654b4a59ee1bcf99cfbce2d72`;
`module_bl_mynnedmf_common.F` SHA256 is
`02af5bd4b97903ce3cd44458bdd218c504fa6034b7d20dfffc983b88bff5038e`.
These retain WRF's public-domain statement. `erff.cuh` retains its Sun
permission notice; see the repository NOTICE and licenses directory.

The offline recipe is `tools/mynn_wrf471_port/{lower,lower_cuda,generalize,coalesce}.py`.
Native REAL constants are rounded before lowering, private arrays use a
priced device arena interleaved across columns, and every Fortran loop keeps its arithmetic order. Run coalesce.py on the generalized source before shipping it; scalar pointers keep stride one and private array pointers retain the column stride through array sections. Compound assignments retain the original operand type.
`schema49.json` and `schema50.json` describe affine vertical dimensions;
the generated source was independently checked against a 64-level lowering.
No Fortran compiler or f2c is used during a forecast.

`prewarm.py --nz 49 --arch sm_120` builds strict and fast images with
parallel offline optimization. Each receipt binds the complete source,
vertical size, architecture, arithmetic mode and image bytes. Another
vertical size or architecture requires its own image or CUDA nvcc.

`tke_budget=1` publishes QSHEAR, QBUOY, QDISS, QWT and DTKE. The switch
belongs to `&physics`. `bl_mynn_output=1` publishes the ten native plume
fields. The other controls implement the Registry's closure 2.5/2.6/3.0,
cloud-PDF 0/1/2, mixing length 0/1/2, EDMF, momentum, TKE, scalar mixing,
cloud mixing, total-water mixing and cloud-radiation switches. Selector
6 is refused by WRF 4.7.1 itself; MYNN is selector 5.

Retained real native inputs and references live at
`~/weather/data/datasets/paper/wrf-ports/port-mynn-subopts/`.
The lane report records qualification, timings, forecast commands and
map review. A raw column replay alone does not qualify a forecast.

Fast mode inlines native routines to avoid their large device-call frames. Strict mode keeps the native calls and libm helpers. Fast mode uses CUDA FP32 elementary functions. The cold native rmol local is read before assignment; the deterministic zero frame and its whole-native context witness are recorded in qualification.json and the retained paper evidence.

Fast mode splits the native driver at complete-statement boundaries and
launches nine kernels through a cached CUDA graph. Eight independent updrafts
run in separate workers; each level's flux additions keep native plume order.
Private carriers with no enabled reader are skipped. Native environmental
accumulators read only by the compiled-false subsidence branch are removed.
`coalesce.py` applies `plume_parallel.py` and `fold_locals.py` after `phases.py`.
Cross-phase locals live in the existing priced arena; the dead UPQV vector
holds the plume split's bounded state. Strict single-kernel and strict graph
replays both compare every native output word. Graphs are rebuilt runtime
handles, kept outside serialized forecast state.

The 14 qualified changed settings pass strict single-kernel and graph
replay against 90 retained native columns each, with 0/234270 differing
native output words and no exclusions. Each has a real fast forecast and
a reviewed Rust map. Interleaved same-card RTX 4090 timing gives:

| Setting | Stage/default MYNN |
|---|---:|
| `bl_mynn_closure=2.5` | 0.6729x |
| `bl_mynn_closure=3.0` | 0.7238x |
| `bl_mynn_cloudmix=0` | 0.8458x |
| `bl_mynn_cloudpdf=0` | 0.7786x |
| `bl_mynn_cloudpdf=1` | 0.7970x |
| `bl_mynn_edmf=0` | 0.5306x |
| `bl_mynn_edmf_mom=0` | 0.8165x |
| `bl_mynn_edmf_tke=1` | 0.7966x |
| `bl_mynn_mixlength=0` | 0.6698x |
| `bl_mynn_mixqt=1` | 0.7854x |
| `bl_mynn_mixscalars=1` | 0.8612x |
| `bl_mynn_output=1` | 0.7815x |
| `icloud_bl=0` | 0.7060x |
| `tke_budget=1` | 0.7742x |

The nearest baseline is ordinary WRF 4.6.1 MYNN with default sub-options.
Six baseline arms and 14 option arms use one commit and card; the first
step of each is excluded, and all 54 warm baseline samples are retained.
The baseline median is 14.651739 ms. These two-minute cases establish
execution and stage cost, not forecast skill. Exact commands, immutable
commit, retained proof paths and the cold-local witness are recorded in
`qualification.json` and the lane report. Default forecasts retain WRF
4.6.1; omitting scalar mixing preserves its existing value 0 with a notice
that native WRF 4.7.1 defaults to 1.
