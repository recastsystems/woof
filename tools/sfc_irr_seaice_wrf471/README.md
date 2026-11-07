# WRF 4.7.1 Noah and sea-ice qualification

This port reuses the earlier Noah irrigation and UA work and adds the native
fractional-ice driver and prescribed sea-ice controls. WRF-shipped code retains
WRF's public-domain treatment. The native referee is WRF v4.7.1 commit
f52c197ed39d12e087d02c50f412d90d418f6186, GNU Fortran 13.3, REAL32,
`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`, MPI 16.

Kept native inputs and outputs are under
`~/weather/data/datasets/paper/wrf-ports/port-sfc-irr-seaice/`.
Each row has a MANIFEST.md and SHA256SUMS.json with commands, pins, sizes and
hashes; shared binary layouts and native build files are kept beside the rows.
The real case is 2019-11-26 12Z CONUS, 424 x 299 x 49, with 424 ice cells.
Option controls are explicit: irrigation is 20 percent on crop cells, prescribed
ice is 3 m, snow is from real SNOWH, prescribed albedo is 0.65, and existing
ice cells are set to fraction 0.5 for the fractional case. These controls are
not claimed to be observed data. Native inputs are retained unchanged.

| Row | Strict replay, differing/compared words | Default-fast stage ratio |
|---|---:|---:|
| ua_phys=true | 0/42,977,064, Noah plus ice | 0.966x Noah land |
| sf_surf_irr_scheme=1 | 0/5,760 module; 0/41,455,752 Noah plus ice | 0.999x Noah land |
| sf_surf_irr_scheme=2 | 0/5,760 module; 0/41,455,752 Noah plus ice | 1.020x Noah land |
| sf_surf_irr_scheme=3 | 0/752,004 module; 0/41,455,752 Noah plus ice | 0.985x Noah land |
| seaice_albedo_opt=1 | 0/15,593,448 ice | 1.012x Noah land, with snow/thickness controls |
| seaice_albedo_opt=2 | 0/15,593,448 ice | 0.952x Noah land |
| seaice_snowdepth_opt=1 | 0/15,593,448 ice | 1.012x Noah land, with albedo-1/thickness controls |
| seaice_thickness_opt=1 | 0/15,593,448 ice | 1.012x Noah land, with albedo-1/snow controls |
| fractional_seaice=1, Noah | 0/23,579,064 surface; 0/23,580,336 blend; 0/41,455,752 land plus ice | **opt-in: 1.526x surface plus land** |
| fractional_seaice=1, Noah-MP | 0/23,579,064 surface; 0/23,580,336 blend; 0/15,593,448 ice | **opt-in: 1.958x surface layer** |

Fractional ice remains an explicit, default-off selector. It runs two native
surface calls. Native CPU_TIME around those calls measures 1.966x the first
call alone. GPU timing uses CUDA events, medians of steps 3 through 10 of each
200-second forecast, divided by the mean of adjacent option-off arm medians.
The nearest scheme is the same land model, option off, same commit/card.
Card 0 was shared; receipts retain its free memory and identity. Noah-MP land
medians varied with contention: the apparent 0.806x combined-stage ratio is
not a speedup claim. The conservative surface-layer cost is stated above.
No double-precision replacement is used.

These row proofs use the default revised MM5 surface layer (sf_sfclay_physics=1).
Adapters for other admitted surface-layer schemes do not carry a claim that
this case qualified every cross-product. UA is Noah-only. Channel irrigation
is Noah-only and is separate from Noah-MP opt_irr. Input fields and timing
controls required by a selected option are checked at the importer and run
boundaries; no whole-cell substitution occurs for Noah or Noah-MP.

The native revised MM5 wrapper declares USTM_SEA but reads it before assignment
(kept module_surface_driver-capture.F lines 6341 and 6595). A second unchanged
native run with MALLOC_PERTURB_=64 changes exactly 1,272 USTM words on ice.
Every other captured surface output agrees. The fractional replay excludes
only those witnessed USTM words. It applies no tolerance or further output
mask. The kept witness captures are part of each fractional manifest.

Run replays from an engine checkout with the CUDA Python environment:

```sh
P="$HOME/weather/data/datasets/paper/wrf-ports/port-sfc-irr-seaice"
export GPUWM_WRF_EXACT=1
python tools/sfc_irr_seaice_wrf471/replay.py "$P/ua-phys" \
  --layout "$P/layout" --tables "$P/native-build" --out replay-ua.json
python tools/sfc_irr_seaice_wrf471/replay_irrigation.py "$P/irr1" --out replay-irr1.json
python tools/sfc_irr_seaice_wrf471/replay.py "$P/irr1" \
  --layout "$P/layout" --tables "$P/native-build" --out replay-land-irr1.json
# Use irr2 or irr3 for the other irrigation rows.
python tools/sfc_irr_seaice_wrf471/replay.py "$P/frac-noah" \
  --layout "$P/layout" --tables "$P/native-build" --out replay-ice-noah.json
python tools/sfc_irr_seaice_wrf471/replay_weighting.py "$P/frac-noah" \
  --layout "$P/layout" --out replay-blend-noah.json
W=$(mktemp -d)
ln -s "$P/frac-noah/witness-captures" "$W/captures"
python tools/sfc_irr_seaice_wrf471/replay_fractional.py "$P/frac-noah" \
  --layout "$P/layout" --undefined-witness "$W" --out replay-frac-noah.json
# Repeat the fractional commands with frac-noahmp and its own witness.
```

`prepare_case.py`, the four capture instruments and `instrument_profile.py`
reproduce the native cases and hooks. Full native build and MPI commands are
in the kept manifests. Forecast commands, configs, timing samples, native
profiles, strict replay JSON and viewed Rust maps are in the lane report's
evidence directory. Default-off Noah and Noah-MP output field arrays are
byte-identical to the pre-port tree across initial and evolved outputs:
1,302,058,678 and 1,355,811,702 bytes respectively. Container timestamps are
not compared. Retain the native captures permanently.

The production kernel prewarm includes every added module. Compiler-bound
AOT images are built from production requests with `measure_frames.py`
(`WOOF_AOT_RECORD=.../{pid}.json`) and `tools/build_kernel_aot.py --workers 8`;
source and compiler hashes prevent stale image reuse. Measured per-thread
frames are at most 224 bytes in fast and strict images, below the existing
320-byte admission ceiling.
