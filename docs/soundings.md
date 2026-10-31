# Soundings from a run

`woof soundings` draws the website sounding window from ordinary WRF history files or rw-store sounding volumes. It accepts a single run or an ensemble with `members/<member-id>/` directories. No cycling or assimilation directory structure is required.

```sh
woof soundings ./run --point point-a=35,-97 --point point-b=36,-98
```

The default renders the ensemble mean and an all-member view every hour. The all-member view includes thin temperature/dewpoint profiles and hodographs, a P10–P90 envelope, and the mean profile. Each window includes the original parcel, wind/shear, severe-weather, thermodynamic and other diagnostic panels.

```sh
woof soundings ./run --point 35,-97 --mode mean --mode all --mode member000 \
  --out ./render --setting theme=light
```

Use the actual member directory name with `--mode`. For a deterministic run its identifier is `det`. `--domain d01` selects one domain. `--interval-minutes 60` samples the available valid times hourly from the first time; it does not synthesize missing forecast times. All ensemble members must have a profile at each selected time.

A named station can be supplied directly as `--station NAME=LAT,LON`, or looked up in an explicit JSON station catalogue:

```json
{"TEST": [35.0, -97.0]}
```

```sh
woof soundings ./run --station TEST --stations ./stations.json
```

Files go to `RUN/render/<domain>/soundings/<valid-day>/`, or the same layout beneath `--out`. The daily `index.html` shows the sequence. Each PNG has JSON containing the profile, indices and the site's diagnostic tables; the all-member JSON also contains individual profiles and percentiles. `soundings-manifest.json` records pictures and render timings.

To request the product as part of an ensemble run, add this to its run configuration:

```toml
[ensemble]
members = 2
sounding_points = ["point-a=35,-97", "point-b=36,-98"]
```

This retains member history files and renders the sounding product after the ensemble completes. Other source and ensemble options still apply.

The adapter reads NetCDF in Rust, rotates WRF grid-relative winds before bilinear sampling, converts WRF water-vapour mixing ratio to dewpoint, and includes surface anchors and pressure vertical velocity. Ensemble statistics use a common log-pressure ladder without extrapolating below ground or above the model. rw-store inputs use physical exact-time metadata or a recognizable UTC run origin for older whole-hour stores.

The renderer is the ported egui/wgpu website window, including sharprs/sharppyrs, rather than a separate chart implementation. Packages include `rw_soundings`. For a source build:

```sh
cd tools/rustwx
cargo build --offline --locked --release -p rw-soundings
```

A batch of up to 8 pictures per CPU thread (32 with the default 4 threads) draws on the built-in CPU rasteriser on those threads, without starting a GPU device. A larger batch renders on a high-performance GPU through Vulkan (Metal or DX12 on macOS and Windows), then GL, and otherwise on the CPU rasteriser, so a headless server with no graphics stack still draws every picture. On Linux, GPU rendering needs the Vulkan loader, which some server and cloud images leave out:

```sh
sudo apt-get install -y libvulkan1     # Debian/Ubuntu; dnf install vulkan-loader on RHEL/Fedora
```

`woof doctor` names the device large batches will use (`rw_soundings --probe-device` prints the same line). `GPUWM_SOUNDINGS_RENDERER=cpu` forces the CPU rasteriser for every batch and `GPUWM_SOUNDINGS_RENDERER=gpu` puts every batch on the GPU and refuses to run without a GPU device. Both dark (default) and light themes use contrasting diagnostic headers and map outlines.
