# Without a GPU

The forecast loop needs an NVIDIA card. Everything upstream of it does
not, and this page is the shortest path a GPU-less machine can walk end
to end: install, size a domain, fetch real bytes, and preprocess them
into a prepared bundle a card can pick up.

It exists because the accurate answer to "what can I evaluate on this
laptop?" used to be undocumented, so the first refusal a GPU-less
reader met was also the first news that a card was required.

Every command and every number below was run on one Windows 11 box
against live GFS on 2026-08-18. Your times will vary with network and
disk; the shape will not.

## What a machine without a card can and cannot reach

| door | without a card | why |
|---|---|---|
| `woof setup`, `woof doctor` | **yes** | staging and estate reporting touch no device |
| `woof fetch`, `woof fetch-geog` | **yes** | downloads and hashes |
| `woof domain` | **yes, with `--card` or `--vram-gib`** | a declared tier is sizing arithmetic; without one the wizard measures the local card and refuses when there is none |
| `woof check` | **yes, with `--budget-gib` and `--vram-gib`** | it says on its face that it is estimating for hardware not present |
| `woof prep` | **yes** | preprocessing runs on the deterministic parallel CPU backend, and the bare default selects it |
| `woof sim`, `woof go`, `woof run` | **no** | the forecast model is CUDA-only |
| `woof render` | **only with a wrfout in hand** | rendering reads a finished forecast; a GPU-less box cannot produce one, but it can draw one produced elsewhere |

So a GPU-less box reaches **prepared model inputs**, not weather images.
That is a real deliverable (it is the expensive, fiddly half of the
pipeline, and the bundle it writes is portable) but it is not a
forecast, and nothing on this page will produce a picture of the
weather.

## The walkthrough

### 0. Install and stage

```bash
pip install recast-woof
woof setup
```

No CuPy extra is needed for anything on this page. `setup` stages the
decode bridges and the physics tables, then reports the estate. It
closes by saying it did **not** stage the WPS_GEOG static geography,
which the next step needs:

```bash
woof fetch-geog          # ~2.7 GB down, ~31 GiB unpacked, resumable
```

If a WPS_GEOG tree already exists on the machine, skip this and pass
`--geog-root` to step 3 instead.

### 1. Size a domain against a card you do not have

```bash
woof domain --point 35.3,-97.5 --card 12gb \
  --source gfs --cycle latest --hours 3 --out configs/cpuwalk.toml \
  --data-dir data/cpuwalk --explain
```

**`--card` is what makes this work.** Declaring a tier tells the wizard
to size for a machine that need not be this one, so it never probes for
a local device. Omit it and the wizard measures the local card, and on
a box with none, refuses and names the choice.

**`--explain` is what prints step 2.** Without it the wizard ends with
the one `woof go` line that fetches, prepares and forecasts in a
single run, which needs a card. With it, the wizard prints the separate
`woof fetch` line this page runs next. `--data-dir data/cpuwalk` makes
that line download into the folder the later steps read.

Measured: 2.2 s. It writes `cpuwalk.toml` and `cpuwalk.namelist.wps`,
runs the memory preflight against the declared 12 GiB, and prints the
next commands with the area and cycle already filled in.

### 2. Fetch real bytes

```bash
woof fetch --source gfs --cycle 2026-08-18T18 --hours 3 \
  --area 7.89,-132.71,61.63,-62.29 --p-top-pa 5000 --out data/cpuwalk
```

Paste the `woof fetch` line step 1 printed as step 1 of its `next:`
list: the area, the resolved cycle and the model top are its own, and
its `--out` is the `data/cpuwalk` folder, spelled as a full path. `--p-top-pa 5000` is the config's 50 hPa top:
without it the fetch stops at 100 hPa and step 3 refuses the folder.
Measured: 2 files, 21.3 MB, 7.5 s on the 100 hPa ladder this walk first
ran with. The same box on the 2026-09-27 12Z cycle took 21.0 MB without
the flag and 22.7 MB with it, in 9.5 s.

Then bind those bytes to the config, which is what the front door reads:

```bash
woof fetch --source gfs --author-front-door-manifest --out data/cpuwalk \
  --wps-namelist configs/cpuwalk.namelist.wps \
  --experiment-config configs/cpuwalk.toml
```

Measured: 1.0 s. It prints the complete `prep` command with the manifest
digest already filled in, so nothing is copied by hand.

### 3. Preprocess on the CPU

Paste the line step 2 printed:

```bash
woof prep --source gfs \
  --gfs-series data/cpuwalk/gfs-series.tsv \
  --cycle 2026-08-18_18:00:00 \
  --wps-namelist configs/cpuwalk.namelist.wps \
  --experiment-config configs/cpuwalk.toml \
  --source-manifest data/cpuwalk/gfs-input-manifest.json \
  --source-manifest-sha256 <the digest step 2 printed> \
  --physics-profile morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1 \
  --geog-root $GPUWM_CASE_DATA_ROOT/WPS_GEOG \
  --output-root out/cpuwalk-prepared
```

**No backend flag.** The default is `--preprocess-backend auto`, and
where CUDA is unusable `auto` selects the CPU backend and says so in one
line:

```
warning: preprocess backend auto: <what it found>, so source-grid/WRF-real
preprocessing runs on the deterministic parallel CPU backend
```

That backend is the packaged Rust bridge, checked against the CUDA path by
the preprocessing parity suite. The documented CPU/GPU preparation comparison
found start-state temperature differences of about 5e-4 K, not bit identity
([TILES.md](TILES.md)); a forecast can amplify them. Prepare both arms of a
comparison with the same `--preprocess-backend`. `--preprocess-backend cpu`
selects this backend explicitly.

Measured: 8.9 s of preprocessing inside a 10.4 s command (static build
3.7 s, decode 0.8 s, initialize 2.5 s, export 1.0 s) writing 700 MiB
under `out/cpuwalk-prepared`:

```
proof.json  source-input-manifest.json  geometry-receipt.json
native-static.npz  wrf-native-input/  prepared-cache/
decoder-inventory.tsv  decoder-gate.tsv  decoder-sha256.tsv
```

`prep` closes by printing the forecast command with all three digests
filled in. That is the line a machine with a card runs.

### 4. Confirm the sizing, still without a card

```bash
woof check configs/cpuwalk.toml --budget-gib 8.23 --vram-gib 12
```

Measured: 1.2 s. Both flags are required here: without them `check`
measures this machine's free VRAM, which is the question a GPU-less box
cannot answer. With them it prices the run against the declared card and
labels the result plainly (`ESTIMATE FOR HARDWARE NOT PRESENT`) so the
margin is never mistaken for a measurement.

## Where it stops

The next command is the forecast, and it is the one that needs a card:

```bash
python -m woof.prepared_single_domain_forecast --source gfs \
  --prepared-root out/cpuwalk-prepared ...
```

The model integrates on CUDA. There is no CPU fallback for it and none
is planned: the project is a GPU model, and a CPU dynamical core would
be a second implementation to keep bit-identical rather than a
convenience.

What the prepared bundle is good for:

- **Move it to a machine with a card.** It is self-contained and
  digest-bound; the receiving box runs the command `prep` printed.
- **Inspect it.** `proof.json`, the geometry receipt and the decoder
  tables answer what was decoded, from which records, at what
  resolution, with which physics authority: the questions that
  normally need a completed run to settle.
- **Prove the route before renting.** Everything above is the part that
  fails on a misconfigured domain, a stale cycle or a missing geog tile.
  Getting a clean prepared bundle on a laptop means the expensive box
  only has to integrate.

## Rendering someone else's forecast

`woof render` reads wrfout files; it never runs the model. Given a
wrfout produced elsewhere, a GPU-less box can draw products from it:
the production renderer is a CPU Rust binary, staged by `woof
fetch-bridges` (part of `woof setup`). `woof doctor` names it if it is
missing. This is a way to look at a colleague's run, not a way to make
one.

## See also

- [FIRST-LIGHT.md](FIRST-LIGHT.md): the same path on a machine with a
  card, through to rendered products.
- [HARDWARE.md](HARDWARE.md): what fits on which card, and where the
  sizing model's safety factor comes from.
- [GLOSSARY.md](GLOSSARY.md): front-door manifest, prepared bundle,
  authority and the rest, mapped to their WPS equivalents.
