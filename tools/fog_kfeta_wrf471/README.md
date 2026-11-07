The fog and KF trigger port reuses the qualified fog implementation from
337cd155a78310cab9d7c29c967a27c11766c3fe and the KF trigger work carried by
ports-final 017c04cea. The numerical authority is public-domain WRF 4.7.1,
commit f52c197ed39d12e087d02c50f412d90d418f6186.

The kept real cases are at
`~/weather/data/datasets/paper/wrf-ports/port-fog-kfeta/grav1-kf2/` and
`~/weather/data/datasets/paper/wrf-ports/port-fog-kfeta/grav2-kf3/`.
Each MANIFEST.md records the captured bytes, native executable hash, build,
run command and replay command. Captures are permanent reference inputs and
outputs. Do not delete them when cleaning regenerated forecast output.

To reproduce capture generation, use an otherwise unmodified native tree:

```sh
python3 tools/fog_kfeta_wrf471/instrument_native.py /path/to/WRF-4.7.1
cd /path/to/WRF-4.7.1
# GNU dmpar, REAL32; configure.wrf in the kept native evidence records all flags.
# FCOPTIM = -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize
./compile -j 8 em_real > native-build.log 2>&1
cd /path/to/real-case
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WOOF_CAPTURE_DIR="$PWD/captures"
mkdir -p "$WOOF_CAPTURE_DIR"
mpirun --mca btl self,tcp --mca pml ob1 -np 8 --bind-to core \
  /path/to/WRF-4.7.1/main/wrf.exe > native-tcp.log 2>&1
```

Both cases use the kept 150 by 150 by 49 real input, 3 km spacing, 12 s
steps, MP8, YSU, MM5 surface layer, Noah, legacy RRTMG and CU1. Native runs
last 12 minutes. Surface/PBL captures occur after the sixth call with cloud
water, with a thirtieth-call fallback. KF is captured at KTAU=31, including
native W0AVG and the real RQVFTEN from dynamics. Instrumentation adds stream
I/O only; the kept capture-only.patch shows every change.

From the engine root, on a GPU with enough free memory:

```sh
GPUWM_WRF_EXACT=1 python3 tools/fog_kfeta_wrf471/replay.py \
  /path/to/grav1-kf2/captures --landuse MODIFIED_IGBP_MODIS_NOAH \
  --out replay-gs1-tr2.json
GPUWM_WRF_EXACT=1 python3 tools/fog_kfeta_wrf471/replay.py \
  /path/to/grav2-kf3/captures --landuse MODIFIED_IGBP_MODIS_NOAH \
  --out replay-gs2-tr3.json
```

The comparison uses uint32 words, including signed zeros. All physical
tiles must be present. Fog compares VDFG, FGDP, DFGDP and the full DQC
result. Trigger 2 also compares the two full perturbation arrays. KF compares
all six output rate profiles and six column outputs where native's entry
NCA is less than half DT: these are the columns on which native calls the
column routine. Other columns retain held tendencies and are outside this
column replay. No native undefined words are excluded from those comparisons.

Native surface captures in these cases have cloud water below its deposition
threshold. The nonzero fog settling witness is the PBL flux divergence.
Trigger 3 has active rain-producing convection. These facts bound the replay;
they do not establish forecast skill or a surface-fog weather case.

Both fog surface passes require native MMINLU `USGS` or
`MODIFIED_IGBP_MODIS_NOAH`. With sf_sfclay_physics=0 native returns before
this surface check; its PBL settling pass still runs without land-use
categories (module_surface_driver.F:1489-1490). Native resets grav_settling to zero for MP28/29
(share/module_check_a_mundo.F:2474-2487); the importer and run door name that
reset when refusing a nonzero request. Only CU1 consumes kfeta_trigger 2/3
(phys/module_cu_kfeta.F:254-398 and :991-1017). Other cumulus schemes ignore
the native key, so their nondefault requests are refused as nonconsuming.
