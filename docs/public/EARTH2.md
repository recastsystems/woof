# WOOF Earth-2 quickstart

The example scripts use `woof earth2`; set `WOOF_BIN=woof` for the engine spelling.

These runs used Earth2Studio 0.19.0. For a separate environment, set
`WOOF_EARTH2_PYTHON` to that environment's Python executable.

FCN3 inference needs 17 GiB of free GPU memory, a 24 GB card or larger (measured peak 16,552 MiB; smaller cards refuse before any download).

Wind Gust AFNO is refused because its published weights use the NVIDIA Community Model License.

Run a short AI-driven WOOF regional forecast on a 16 GB card:

```sh
woof earth2 --model ucast --cycle 2024-09-24T12 --hours 12 \
  --area 24,-90,30,-84 --out earth2-forecast
```

The runnable version is `examples/earth2/forecast.sh --short --out earth2-forecast`.
This short version uses UCast's 12 hour cadence. UCast inference peaked at
4.746 GiB of allocated GPU memory; the regional run with receipt-checked forcing
reuse completed in 146.3 seconds at 4,467 MiB peak device memory above baseline.
SFNO failed on the 16 GB card: its process used 14.1 GiB and its next
1.49 GiB allocation had only 1.36 GiB free.

Run a short storm nest on a 16 GB card using a completed FCN3 fetch:

```sh
woof earth2 --model fcn3 --storm auto --nest 2.5km --card 16gb \
  --cycle 2024-09-24T12 --hours 6 --earth2-input /path/to/completed-fetch --out earth2-storm
```

Fresh FCN3 inference requires the larger card described above. This six hour
short version reuses a completed larger-card fetch and runs tracking,
preparation and the 2.5 km nest on the 16 GB card. Keep the original receipt,
request and named forecast/analysis files together; reuse checks their receipt.
The input must match the model and cycle, contain every requested lead, and
cover the authored domain and moving corridor. A bounded explicit input defines
the area available to automatic storm search. An incompatible input refuses
before inference; omitting it keeps the normal model acquisition path.
The runnable version is
`examples/earth2/storm_nest.sh --short --fetch-receipt /path/to/earth2-run.json --out earth2-storm`.
It completed in 297.7 seconds at 3,218 MiB peak device
memory above baseline. The measured automatic center was 16.396 N, 99.821 W.

Run a short mixed ensemble on the same card:

```sh
woof earth2 --model ucast --mix ucast:1,gefs:1 \
  --cycle 2024-09-24T12 --hours 12 --area 24,-90,30,-84 --out earth2-mix
```

This small-N, 12 hour version uses one UCast member and one GEFS control.
The runnable version is `examples/earth2/mixed_ensemble.sh --short --out earth2-mix`.
It completed with receipt-checked acquisition reuse in 284.2 seconds at
6,633 MiB peak device memory above baseline. These two members demonstrate
the mixed-source workflow; they do not establish ensemble calibration.

Fetch the public FCN3 weights with the example below. It reads the model's
repository from the model authority and passes `token=False` to Hugging Face.

```sh
python3 examples/earth2/hf_fetch.py --model fcn3 --out earth2-weights
```

Use a Python environment with `huggingface_hub` installed and a new output
directory. The example writes `hf-fetch-receipt.json`, including the resolved
repository revision, file sizes and elapsed download time. It creates no account.


Export the regional forecast for Earth2Studio and calculate weights-free
10 m wind speed with its DerivedWS diagnostic:

```sh
woof earth2 --model ucast --export earth2 --diagnose derived-ws10m \
  --cycle 2024-09-24T12 --hours 12 --area 24,-90,30,-84 --out earth2-export
woof render --cf earth2-export/diagnostics/derived-ws10m/diagnostic.nc \
  --variable ws10m --bounds=-90,-84,24,30 --out earth2-export/diagnostic-maps \
  --receipt earth2-export/diagnostic-map-receipt.json
```

The runnable version is `examples/earth2/export_diagnostic.sh --short --out earth2-export`.
This 12 hour short version completed, including its native diagnostic map,
with receipt-checked forcing reuse in 151.2 seconds at
2,926 MiB peak device memory above baseline. DerivedWS needs no weights;
the map displays wind speed in knots.

<!-- earth2-models:begin -->
Generated from `woof/authorities/earth2-models.v1.json`.
Authority SHA-256: `6b5087de93d9240cadf7f595a87f1543ec356f9fce6ce5490cf0c8f1e6d88477`.

These are authority declarations, not a list of measured forecasts.
Gated rows declared here: none.
Account-required rows declared here: none.

| ID | Model | Step (h) | Licence declared for weights | Use | Weight access declared | Profile | Inference memory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ace2 | ACE2ERA5 | 6 | Apache-2.0 | See declared licence | No account declared | Not declared | Capacity floor not declared |
| aifs | AIFS | 6 | CC BY 4.0 | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| aifs-ens | AIFSENS | 6 | CC BY 4.0 | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| aifs2 | AIFS2 | 6 | CC BY 4.0 | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| aifs2-ens | AIFS2ENS | 6 | CC BY 4.0 | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| atlas | Atlas CRPS, selected member | 6 | NVIDIA Open Model License | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| atlas-crps | Atlas CRPS | 6 | NVIDIA Open Model License | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| atlas-si | Atlas stochastic interpolant | 6 | NVIDIA Open Model License | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| aurora | Aurora | 6 | MIT | Refused by runtime | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| aurora15 | Aurora1p5 | 1 | MIT | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| aurora15-6h | Aurora1p5_6h | 6 | MIT | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| aurora15-ensemble | Aurora1p5Ensemble | 1 | MIT | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| aurora15-ensemble-6h | Aurora1p5Ensemble_6h | 6 | MIT | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| corrdiff | CorrDiff (km-scale diffusion downscaling) | Not declared | Not declared | See declared licence | Not declared | Not declared | Capacity floor not declared |
| derived-ws10m | Derived 10 m wind speed | Not declared | Apache-2.0 (Earth2Studio code; no weights) | See declared licence | No account declared | Not declared | Capacity floor not declared |
| dlesym | DLESyMLatLon | 6 | Apache-2.0 | See declared licence | No account declared | Not declared | Capacity floor not declared |
| dlwp | DLWP | 6 | unresolved: NGC checkpoint terms | See declared licence | No account declared | Not declared | Capacity floor not declared |
| fcn | FCN | 6 | unresolved: checkpoint has no explicit weight licence | See declared licence | No account declared | Not declared | Capacity floor not declared |
| fcn3 | FCN3 | 6 | Apache-2.0 | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | 17.0 GiB free; FCN3 inference needs 17 GiB of free GPU memory (a 24 GB card or larger) on the pinned stack (earth2studio 0.19.0, torch-harmonics 0.8.1 a632ca7, torch 2.14.1 cu13). Measured on an RTX PRO 6000 96 GB, 2026-10-21, woof fetch --source earth2-fcn3 --cycle 2024-10-08T00 --hours 24, one member, batch_size 1, under a 30,000 MiB card lease: card memory peak 16,552 MiB (process-tree nvidia-smi sampler), torch max_memory_allocated 13.969 GiB, max_memory_reserved 15.512 GiB (earth2-run.json). 17 GiB is that measured peak rounded up. The earlier 59 GiB floor (60,212 MiB card peak, 49.3 GiB allocated) measured the runner's DISCO band patch not applying to torch-harmonics 0.8.1, so its whole expanded convolution input and the einsum's permuted copy were resident at once; woof/earth2_runner.py now contracts that layout one channel slice and latitude band at a time. |
| fengwu | FengWu | 6 | unresolved: FengWu weight terms | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| fuxi | FuXi | 6 | unresolved: FuXi weight terms | See declared licence | No account declared | earth2-pl13-rh-netcdf-v1 | Capacity floor not declared |
| fuxi-s2s | FuXiS2S | 24 | CC BY-NC-ND 4.0 | Research-only | No account declared | Not declared | Capacity floor not declared |
| gencast-mini | GenCastMini | 12 | CC BY 4.0 | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| graphcast-operational | GraphCastOperational | 6 | CC BY 4.0 | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| graphcast-small | GraphCastSmall | 6 | CC BY 4.0 | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| interpmodafno | InterpModAFNO | 1 | NVIDIA Community Model License | See declared licence | No account declared | Not declared | Capacity floor not declared |
| pangu | Pangu6 | 6 | CC BY-NC-SA 4.0 | Research-only | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| sfno | SFNO | 6 | Apache-2.0 | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| stormcast | StormCast | 1 | Apache-2.0 | See declared licence | No account declared | earth2-stormcast-hrrr-netcdf-v1 | Capacity floor not declared |
| stormcast-conus | StormCastCONUS | 1 | OpenMDW-1.1 | See declared licence | No account declared | earth2-stormcast-conus-hrrr-netcdf-v1 | Capacity floor not declared |
| ucast | UCast | 12 | Apache-2.0 | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| weathernext2 | WeatherNext2Cyclones | 6 | CC BY 4.0 | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| weathernext2-mini | WeatherNext2CyclonesMini | 6 | CC BY 4.0 | See declared licence | No account declared | earth2-pl13-q-netcdf-v1 | Capacity floor not declared |
| windgust | Wind Gust AFNO | 1 | NVIDIA Community Model License | Explicit --accept-licence required | Not declared | Not declared | Capacity floor not declared |
<!-- earth2-models:end -->

Regenerate and check this table from the repository root:

```sh
python3 tools/earth2_models_doc.py
python3 tools/earth2_models_doc.py --check
```

Restricted diagnostic weights remain off by default. The existing spelling
`--accept-licence "NVIDIA Community Model License"` enables the licence gate
for `earth2 --diagnose windgust` and both `earth2-diagnose` forms. Acceptance
does not waive upstream production restrictions or bypass missing channels,
account requirements or other runtime refusals. The default export recipe uses
`--diagnose derived-ws10m`, which needs Apache-2.0 code and no weights.
