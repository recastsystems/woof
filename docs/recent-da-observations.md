# Recent Gulf and southeast observations

`tools/da_recent_observations.py` is a public box-side adapter. It has four doors: `inventory`, `fetch`, `prepare`, and `truth`. Inventory reads only anonymous public object metadata. Fetch has one destination lock and one sequential downloader. Preparation runs native front doors sequentially. No credentials or SDK are consulted.

The case starts at `2026-10-03T21:00:00Z`. Hourly positive member legs analyse at 22Z, 23Z and 00Z, or continue through 03Z for six hours. The free forecast then lasts six hours. An observation at the initial 21Z cannot be used as a zero-length forecast leg.

The observation case JSON uses schema `da-rerun.recent-observations.v1`, `model_start_utc`, `analysis_times_utc`, `observation_cutoff_utc`, `bbox` in W,S,E,N, and at least two explicit `radar_sites`. Optional `model_requests` is a list of public bucket/key pairs, and `surface_networks` freezes the native station-network roster. The box operator must freeze the radar roster against the actual prepared domain. The initial metadata roster is a broad acquisition list, not a domain coverage proof.

## Native builds

Use the current engine export, its `.engine-export-sha` and matching bridge source stamp. Build the engine's whole native roster, the six workspaces the installers build (`docs/install.md`); the radar front door, the station and MRMS readers and the region-global solver this page needs are among them, and a partial build leaves `woof doctor` reporting the other default routes missing:

```bash
cd "$DA_ENGINE"
for workspace in tools/grib1_bridge tools/rustwx tools/zarr_bridge tools/rw_wps tools/region_global_dealias tools/nextgen_hydrofabric tools/nextgen_forcing tools/nextgen_verify tools/hydro_world tools/hydro_world_params tools/nextgen_reservoir_prepare tools/hydro_world_init tools/ocean_prepare tools/rw-bathy tools/wind-table tools/nextgen_output tools/rw_icechunk; do
  (cd "$workspace" && cargo fetch --locked && cargo build --release --offline --locked -j 16)
done
export WOOF_RW_NEXRAD="$DA_ENGINE/tools/rustwx/target/release/rw_nexrad"
export WOOF_RW_ASOS="$DA_ENGINE/tools/rustwx/target/release/rw_asos"
export WOOF_RW_MRMS="$DA_ENGINE/tools/rustwx/target/release/rw_mrms"
export WOOF_DEALIAS_REGION_BRIDGE="$DA_ENGINE/tools/region_global_dealias/target/release/libregion_global_dealias.so"
cd "$DA_ENGINE"
```

The native radar decoder and region-global solver are required. A missing named solver is a refusal. The adapter does not silently substitute the mask-only or VAD paths. The same 250 km and 20 degree range/elevation authority goes to decode and superobbing. RHO dual-pol QC, censored clear-air support, the public 4/3-earth beam geometry, terrain-relative layer support, per-radar beam vectors and the public coherence rejection are retained. The public path has no terrain beam-blockage model or attenuation correction; no such treatment is claimed.

## Full case, one fetch controller

Stage the frozen observation case JSON, then:

```bash
python tools/da_recent_observations.py inventory --case recent-observation-case.json \
  --out "$DA_ROOT/receipts/recent-inventory.json"
python tools/da_recent_observations.py fetch --inventory "$DA_ROOT/receipts/recent-inventory.json" \
  --root /workspace/shared-prepared/da-rerun-286/recent/raw \
  --receipt "$DA_ROOT/receipts/recent-fetch.json" --max-gib 10
python tools/da_recent_observations.py prepare --inventory "$DA_ROOT/receipts/recent-inventory.json" \
  --fetch-receipt "$DA_ROOT/receipts/recent-fetch.json" \
  --grid-wrfout /workspace/shared-prepared/da-rerun-286/recent/prepared/grid.nc \
  --out /workspace/shared-prepared/da-rerun-286/recent/observations
python tools/da_recent_observations.py truth --inventory "$DA_ROOT/receipts/recent-inventory.json" \
  --fetch-receipt "$DA_ROOT/receipts/recent-fetch.json" \
  --out /workspace/shared-prepared/da-rerun-286/recent/truth --center-lon -89 --center-lat 31
```

`grid.nc` must come from the actual prepared model state, including its real terrain and PH/PHB interfaces. A synthetic stretched georeference is not equivalent. Use the case builder's emitted path rather than inventing this filename. On a host where `/workspace/shared-prepared` is not writable, author the case with `--shared-root` naming a writable directory and keep the single shared copy there. The rented box keeps the standard shared directory.

If the selected objects are already present, add `fetch --reuse-receipt OLD_COMPLETE_FETCH.json` while writing a fresh new receipt. The controller requires a completed ancestor, exact path and object metadata, then hashes the actual cached file before reuse. It preserves `fetched_at_utc`, binds the ancestor file SHA and reports downloaded/reused counts separately. Changed, partial or untracked cache data refuses. It never overwrites the ancestor receipt or stamps old bytes as newly fetched. The case JSON stays unchanged.

`prepare` validates every inventory object against its fetched byte count and SHA, selects each site's newest candidate whose LastModified is at/before that analysis, then requires a complete native sweep roster and measured radial end at/before the analysis. A volume's filename is its start, not its completion. Measured end age is at most 900 seconds. Native decode/verify must pass. A contributing radar must carry usable on-grid Z and Vr, a balanced region-global finite-gate account and at least one actual native library call, plus dual-pol QC accounting. Refusals stay in each `slot-NN/radar-attempts.json`. At least two radars must contribute.

Surface preparation freezes the native IEM station table. The case author emits a network list; the adapter normalizes that metadata to the native CLI's CSV argument without editing case authority. A caller may also give the equivalent CSV string. For each hour, `rw_asos fetch` requests only `[analysis-900 s, analysis]`, then the native decoder performs unit conversion, gross-error QC and matching. The adapter rejects any future, stale or wrong-slot report. Hourly seams are individually verified. Their station and report metadata are joined without changing observation values, and the union is reverified natively. It states no across-window completeness claim. Original METAR publication time is unknown, so the result is an archive replay and does not establish operational latency.

The final `observations/slots.json` contains `obs`, `grid_wrfout`, `leg_seconds=3600`, `analysis_time`, field/file hashes and the combined `surface_obs`. Put these exact slots and surface path in the regional run manifest. Include `surface_dewpoint_error_k: 1.0` only when requesting the pinned dewpoint operator. Native source hashes, decoder hashes, superob parameters, inventory hash and fetch receipt hash enter the preparation receipt.

Truth fetch selects temporal brackets around the exact scoring hours `[0,1]`, `[2,3]` and `[5,6]`. The case must separately freeze `truth_bbox` and native `truth_padding_m` of at least 75000 m. Truth decoding refuses reuse of the model bbox. The native MRMS decoder writes PrecipRate, primary QC composite and RQI packs on that identical independent truth extent. `truth/truth.json` is directly usable by `tools/regional_rain_score.py`. Native sentinel and bitmap masks are preserved. RQI is retained separately because no unstated RQI threshold should define a new score. Historical selected storm footprints or independent gauges require separate frozen inputs. Native observed footprints are built before model intersection. A clipped outer-edge object or footprint outside model support remains pending.

## Bounded real-data smoke

The smoke uses only the first 22Z slot. It never claims three or six analysed hours or six-hour verification. Supply the minimal six model requests separately so the same fetch controller obtains both model and radar objects:

```bash
python tools/da_recent_observations.py inventory --case recent-observation-smoke.json \
  --first-slot-only --model-requests recent-smoke-model-requests.json \
  --out "$DA_ROOT/receipts/recent-smoke-inventory.json"
python tools/da_recent_observations.py fetch --inventory "$DA_ROOT/receipts/recent-smoke-inventory.json" \
  --root "$SHARED/raw" --receipt "$DA_ROOT/receipts/recent-smoke-fetch.json" --max-gib 3
python tools/da_recent_observations.py prepare --inventory "$DA_ROOT/receipts/recent-smoke-inventory.json" \
  --fetch-receipt "$DA_ROOT/receipts/recent-smoke-fetch.json" --grid-wrfout "$ACTUAL_GRID" \
  --out "$SHARED/observations"
```

It emits `status=prepared-smoke-first-slot`, `forecast_fork_utc=2026-10-03T22:00:00Z`, the requested full-case fork separately, and `scientific_gate=pending`. It does not fetch MRMS. `truth` refuses a smoke inventory.
