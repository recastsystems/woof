# The woof hex test battery

Three tiers. The split is not by speed, it is by **what a machine has to own**
to run a tier in full, and each tier that cannot run says which thing is
missing rather than dying somewhere unrecognisable.

The lists in this directory are the source of truth for what a tier covers.
CI reads them; a queue script reads them; nobody retypes a file list into a
workflow, because a retyped list drifts and the drift is invisible until a
gate that was supposed to run has not been running for a month.

---

## Tier 1 (`cpu_files.txt`) runs anywhere

```sh
PYTHONPATH=src python -m pytest -q -rs -m "not gpu and not bigcard and not assets" \
    $(grep -v '^#' tools/battery/cpu_files.txt)
```

Needs nothing but Python and the declared dependencies. This is the tier a
GitHub-hosted runner runs on every push and pull request, and it is the tier
that catches what a public cut can actually break: the packaging declaration,
the front doors' argument surfaces and refusals, the proof-guard ordering and
pin structure, the multi-GPU partition executor's pure logic.

`test_native_dry_agreement.py` also runs one NumPy dry step when the
rebuildable native reference is present. Set `HEX_NATIVE_DRY_REFERENCE`
to the work root produced by `woof hex verify-dycore`. Without it, `-rs`
prints the missing-reference reason and the command that creates it.

It deliberately does not import CuPy. Any test whose module or function
reaches CuPy is auto-marked `gpu` by `tests/conftest.py` (by AST, not by
trusting an author to remember a decorator) so it cannot leak into this tier
by omission.

## Tier 2 (`asset_gates.txt`) needs the byte-pinned authority files

```sh
PYTHONPATH=src python -m pytest -q -m assets $(grep -v '^#' tools/battery/asset_gates.txt)
```

About 6.9 GiB of mesh, static, init and native-history files, pinned by byte
count and SHA-256 (masked over the netCDF `file_id` nonce for the history
files, which is random per write). **They ship with no fetch path.** They live
relative to the checkout root, on whichever machine holds them, under
`work/v841-vr-static/` and
`work/v841-full-physics-gf-gwdo-native-authority-20260820a/`.

Without them these tests skip with the count and the first missing path
named, because the failure they used to produce (`FileNotFoundError` on a
path inside the checkout that has never existed on this machine) reads as a
broken checkout rather than a missing asset.

## Tier 3 (`gpu_gates.txt`) needs a big card

```sh
PYTHONPATH=src python -m pytest -q -m bigcard        # the capacity preflight
# then the harnesses named in gpu_gates.txt
```

The x4.163842 full-physics tier peaks at **20,446 MiB** on a 170 SM card and
the gate asks for **22,202.8 MiB free**, that peak plus the card's own margin,
so in practice it still needs a 32 GiB device: a 24 GiB part running
anything else rarely offers 21.7 GiB. The gate retypes neither figure:
`tests/conftest.py` reads `device_admission.native_device_floor_bytes()` and
both the `bigcard` marker text and the skip message print what it returns, so
this requirement cannot drift from the one the run itself applies. Ask
`device_admission.model_for_card(card).required_bytes(163_842)` for any other
card: the number is a property of the card, not of the mesh alone. (The
"about 26.4 GiB resident" this tier used to be described by is superseded; it
predates the Grell-Freitas frame cut and the radiation-chunk narrowing.) On a
smaller card it does not refuse a check: it dies inside a CuPy allocation
part-way through a run, after burning the time it took to get there.
`tests/test_device_capacity.py` is the cheap version of that discovery and
runs first.

**This tier is not in GitHub Actions and will not be.** GitHub-hosted runners
have no CUDA device, and a tier that is silently skipped in CI is worse than a
tier that is declared external: the first reports green while proving nothing.
The reason is named here, in `gpu_gates.txt`, and in the CI workflow itself.

Set `WOOF_HEX_NO_LOCAL_GPU=1` (or `GPUWM_NO_LOCAL_GPU=1`, honoured so a box
configured for the engine behaves the same) to ban device contact outright:
the gates skip by name and `CUDA_VISIBLE_DEVICES` is set to `-1` so that a
transitive allocation the markers cannot see still finds no device.

## Both routes (`both_routes.json`) one real run per capability family, on each physics route

```sh
HEX_GPU_LEASE=<pool admission program> PYTHONPATH=src python tools/both_routes_gate.py --work DIR \
    --staged GFS_DIR --geog WPS_GEOG --best-track IBTRACS.csv --storm-staged GFS_F000 [--jobs 3] [--clean]
PYTHONPATH=src python tools/both_routes_gate.py --plan     # every command, nothing run
```

Every hex capability has to work on the WOOF-physics route and on the MPAS-physics route, and a fix to one
route has to reach the other. This leg runs each capability family once on each route: a 120 km globe for 1-2 h
(global door with live frames, maps and export; keep-as-init export with `HGT`, radar and the column store;
2 cards = 1 card; restart with SPPT+SKEB = straight; IAU; JEDI background; chemistry; balance series; trigger
nests arming; a member roster), a small variable-resolution globe with and without local time stepping, a
limited-area country point and a storm-following moving region. One leg is TIMED: `lts-speed` runs the 96 -> 3 km
zoom globe (the site's default hex shape) for 3 h with the fixed step and again with a bare local-time-stepping
request, back to back on the same card, and is red unless the LTS arm's warm forecast hour (frame to frame, f000
excluded) is at least `min_speedup` times faster; its receipt keeps both rates, the door's plan choice and the
host load at each start, so a red on a loaded box can be told apart from a regression. Breakage it prevents:
LTS at 0.74-0.81x of the fixed step on that mesh in 0.4.5-0.4.9 with no red anywhere (HEX-LTS-REGRESSION
2026-10-27). Each check names what it proves; the receipt
(`DIR/both-routes.json`) and matrix (`DIR/both-routes.md`) carry the verdict, median step time and sampled peak
VRAM per route.

It never holds a card exclusively. GPU steps go through `HEX_GPU_LEASE` (invoked as
`<program> --vram-mib N [--cards K] <lane> -- <command>`, K only for the two-card leg) or the site GPU queue
without `--proof`, so a timed arm in flight keeps its card. The CPU tier's `tests/test_both_routes_gate.py`
parses every command with the real parser, so a renamed flag is red there, before any card is asked for.
