# woof.hex.ad.record: record a trajectory, seal it, replay it

The differentiable engine does not re-implement the forecast. It replays the production hex forward from a recording
of it, and the tangent and adjoint run over that replay. This module makes the recording and seals it into a plan
package (the tape) that the replay runner (`ad-binder replay`), the tangent and the adjoint read.

## Install

```sh
pip install "recast-woof[ad]"          # gpu-cu13 plus nvidia-cuda-cupti and nvidia-cuda-crt (CUPTI and its headers)
```

The recorder libraries are built from the C sources in `native/` on first use, so the host needs a C/C++ compiler
(`CC`/`CXX`, else gcc/g++) and the zlib and OpenSSL libcrypto development libraries. The replay runner is, in this order:
`--runner`, `HEXAD_RUNNER`, the wheel's own runner (`woof/hex/libexec/ad-binder`), or `ad-binder` on `PATH`.

## Record one hour of x1.2562 hrrrv5 on one card

> **In WOOF 1.1.7** recording a plan from a start of your own (`--init FILE`, `--init example`) is not supported: the derivative commands run on the example rows' own plans (`--row`).

```sh
python -m woof.hex.ad.record record --mesh x1.2562 --suite hrrrv5 \
    --init init.nc --grid x1.2562.grid.nc --oro x1.2562.ugwp_oro_data.nc \
    --dt 1200 --hours 1 --out pkg --card 0 --serial
python -m woof.hex.ad.record replay pkg --card 0
```

Arguments after `--` go to `woof hex forecast` unchanged. `--serial` records A then B on one card; without it A runs
on `--card` and B on `--card-b`. `--numerics fast` is the default; `--numerics strict` is the named proof run.

## What `record` does

1. **build**: the three LD_PRELOAD libraries (CUPTI recorder, cuBLAS and CUB/Thrust observers), cached by source hash.
2. **warm**: one ordinary forward of the same configuration, untraced, so resources, the kernel store and the CuPy cache
   are filled and no traced forward compiles or downloads.
3. **record**: the forward twice under the recorder (A and B), radiation and land live, every boundary payload kept.
4. **seal**: A against B. Catalog, correlation, linearisation, library audits, partial plan, device-to-host
   classification, stage 2 (every launch argument and upload explained) and stage 3 (frames, per-step reference).
5. **retention**: a start date the shipped retention list does not cover leaves `upload_needs_live_rule` words. The
   step<=1 uploads the seal names that are equal in A and B are added to this run's own list (`retention.txt`), and
   the pair is recorded once more with it.
6. **pairs**: unproved CIndexer scratch is proved from this recording (at most two rounds) into the package's own
   `cindexer-pairs.json`.
7. **package**: `plan.json` and its assets, `alloc-map.json` and `scale-table.json`, `record-receipt.json` (wall time
   per stage, numerics label, library and runner sha256, and `deck`: the radiation intervals and the cumulus setting the
   recorded forecast itself reports).
8. **replay**: strict replay of the new plan: rc 0 and every boundary hash equal to its own reference.

Any other residual, a failed stage or a replay mismatch is one `DEFECT: woof.hex.ad.record: ...` line and exit 3. There
is no fallback to an older package and no unchecked replay.

## Rows

The sealer catalogs the `hrrrv5` and `convection_permitting` suites. Retention lists ship for hrrrv5 on x1.2562 and
x1.40962; any other mesh or date builds its own list in step 5. Every package records its row, numerics, residual
and replay result in `record-receipt.json`.

## Layout

| path | what |
|---|---|
| `cli.py`, `chain.py` | the command and the record/seal/retention/replay chain |
| `env.py`, `build.py` | finding the toolkit, CUPTI and CuPy in the install; building the recorder libraries |
| `native/` | recorder and observer C sources, the boundary digest kernel source |
| `tools/` | the observer front end and the sealer stages (scripts, run by `chain.py`) |
| `data/` | retention lists, the base CIndexer pairs, verified source records, prebuilt digest cubins (sm_89, sm_120) |
