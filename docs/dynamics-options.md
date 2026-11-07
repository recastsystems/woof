# WRF 4.7.1 dynamics options

The default dynamics path keeps its existing arithmetic. New transport selectors choose separate kernel specializations. WENO momentum runs only on the final Runge–Kutta stage, as in WRF 4.7.1. Its strict mode keeps native precision and assignment order; normal forecasts use FP32 weights normalized before reciprocal evaluation.

| Option | Native words checked | Inside-forecast cost / default | Evidence folder |
|---|---:|---:|---|
| momentum_adv_opt=3 | 3,344,700 | 1.107856× momentum_adv_opt=1 | weno-fastdiv |
| rk_ord=2 | 4,410,000 scalar updates plus actual acoustic schedule | 0.928308× rk_ord=3 full step | rk2 |
| h_mom_adv_order=0 | 2,219,700 u/v | 0.702770× h_mom_adv_order=5 | hmom0-node4 |
| h_mom_adv_order=4 | 2,219,700 u/v | 0.929126× h_mom_adv_order=5 | hmom4-node4 |
| h_mom_adv_order=2 | 2,219,700 u/v | 1.000745× h_mom_adv_order=5 | hmom2 |
| h_mom_adv_order=3 | 2,219,700 u/v | 0.957341× h_mom_adv_order=5 | hmom3 |
| v_mom_adv_order=2 | 2,219,700 u/v | 0.954613× v_mom_adv_order=3 | vmom2-node4 |
| v_mom_adv_order=4 | 2,219,700 u/v | 0.998997× v_mom_adv_order=3 | vmom4-node4 |
| v_mom_adv_order=6 | 2,219,700 u/v | 0.983325× v_mom_adv_order=3 | vmom6-node4 |
| v_sca_adv_order=2 | 3,330,000 scalar/W/PD | 0.866304× v_sca_adv_order=3 | vsca2-node4 |
| h_mom_adv_order=6 | 2,219,700 u/v | 0.997757× h_mom_adv_order=5 | hmom6-node4 |
| v_sca_adv_order=4 | 3,330,000 scalar/W/PD | 1.020343× v_sca_adv_order=3 | vsca4-node4 |
| v_sca_adv_order=6 | 3,330,000 scalar/W/PD | 1.029014× v_sca_adv_order=3 | vsca6-node4 |
| m_opt=1 | 28,420,000 raw; 56,448 production geometry/stress | 1.119862× m_opt=0 mixing stage | mopt1-fixed-node4 |
| h_sca_adv_order=2 | 3,321,496 defined scalar/W/PD/geopotential | 0.802553× h_sca_adv_order=5 | hsca2 |
| h_sca_adv_order=3 | 3,336,000 scalar/W/PD/geopotential | 1.125270× h_sca_adv_order=5 | hsca3 |
| h_sca_adv_order=4 | 3,336,000 scalar/W/PD/geopotential | 0.999290× h_sca_adv_order=5 | hsca4 |
| h_sca_adv_order=6 | 3,336,000 scalar/W/PD/geopotential | 1.046713× h_sca_adv_order=5 | hsca6 |

Only horizontal scalar order 2 excludes native undefined words, as described below. These are real-input operator replays, not a claim that the complete GPU forecast equals a complete native WRF forecast. Fast forecast timing alternates option and default launches on identical inputs on the same card and commit, inside a two-hour `woof go` forecast. The first three calls of each stage are excluded. New scalar rows use warm stage medians weighted by observed call counts, including geopotential transport; full elapsed sample sums remain in their receipts. Pressure, 10 m wind and reflectivity maps were rendered with `rw_wrfbatch` through `woof.rustwx` and opened at hours 0, 1 and 2. The case is largely clear, so convective checking is limited.

The permanent native inputs and reference outputs are kept at:

`<paper-archive>/wrf-ports/port-dyn-options/<row>/`

Each row has `MANIFEST.md` with native commit, strict build flags, command, file sizes and SHA-256 hashes. Replay receipts and GPU/reference word arrays are in the sibling `replay/<row>/` folder. Forecast logs, stage samples, all nine PNGs and map reviews are in:

`<reports>/WRF-PORTS-2026-11-01/evidence/port-dyn-options/<evidence folder>/`

To rerun on a CUDA machine with the engine installed, set `GPUWM_WRF_EXACT=1` and run:

```
python tools/port_dyn_options/replay_capture.py <paper root>/<row>/capture receipt.json
```

Create the receipt's parent directory first. The script assembles the owned MPI tiles, calls the production launchers, compares all float words and writes the full arrays beside the receipt. Strict native WRF is v4.7.1 commit `f52c197ed39d12e087d02c50f412d90d418f6186`, REAL32, GNU Fortran dmpar, `-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`. Capture code and the whole native run recipe are in `tools/port_dyn_options/`.

Native constraints still apply. Horizontal scalar order 0 cannot use WRF positive-definite scalar transport. WENO momentum selects its own horizontal WENO flux, independently of `h_mom_adv_order`. Omitted legacy engine `h_sca_adv_order` retains its former behavior; an explicit order 2 selects native second-order scalar transport and binds that choice in checkpoints.

The order-2 positive-definite native routine leaves horizontal fluxes unwritten on the two outside y rows, then reads them in the divergence (`dyn_em/module_advect_em.F:7176–7240`). The permanent `hsca2-init-witness/` runs the unchanged public routine twice on each captured real MPI tile, with strict flags plus `-finit-real=zero` or `-finit-real=snan`. Exactly 14,504 words change, matching the unwritten band; no other word changes. The mask SHA-256 is `64e12dff5d667c74ee46762e34e450ee1e554ffdeb857e89c47f0a60604c38c3`. Replay with `--undefined-witness <paper root>/hsca2-init-witness/undefined-mask.npz`; the script validates its full shape and exact band and keeps raw and defined-only results. Native full-run allocator witnesses are also retained. All other scalar orders exclude no words. Separate geopotential replay: `GPUWM_WRF_EXACT=1 python tools/port_dyn_options/phi_replay.py replay <engine> <paper root>/phi-replay/<row>`.

The registry advertises only added rows with retained replay evidence and a real forecast. Work on other orders remains separate until their full evidence is complete. The native NBA module has a restrictive source header. A paper-based replacement must be opt-in and labelled “written from papers, not bitwise to WRF”.

Stress output replay includes the production deformation geometry with retained real WRF inputs. The separate full-domain raw stress replay checks native call arguments and outputs. Native K is supplied in the geometry test to isolate the added calculation. The native PBL-on branch leaves M33 zero. Rerun with `GPUWM_WRF_EXACT=1 python tools/port_dyn_options/stress_pipeline.py run <paper root>/m_opt-1-production/stress-pipeline/oracle.so receipt-folder --fixture <paper root>/m_opt-1-production/deformation-real_open.npz`. Its build command and source hashes are in that row's MANIFEST.md. All six history fields are prewarmed and recorded.

RK2 proof also keeps the native acoustic count, acoustic timestep and RK timestep for every stage of the whole one-minute run. The replay calls production scalar bookkeeping, ordinary update and positive-definite source folding. Its cost alternates complete RK2/RK3 steps of two real forecasts with separate evolving states; a pipe token prevents the two GPU steps from overlapping. All 600 full-step samples are saved; the ratio above compares warm medians. This full-step timing includes physics cadence.
