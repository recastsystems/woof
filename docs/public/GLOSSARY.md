# Glossary, for people who know WPS

This project's preprocessing does what `geogrid`/`ungrib`/`metgrid`/
`real.exe` do, but it does not use their names, and the new names are
not guessable. If you arrived from a working WRF case, this page is the
translation table.

`namelist.wps`, `namelist.input` and `WPS_GEOG` are unchanged and stay
authoritative. What replaced the four executables is one door with a
different vocabulary.

## The chain, side by side

| WPS/WRF | here | notes |
|---|---|---|
| `geogrid.exe` | the **static build** inside `woof prep` | reads the same `WPS_GEOG` tree; writes its own geometry receipt instead of `geo_em` files |
| `ungrib.exe` + Vtable | **decode**, driven by a **mapping** | no intermediate files; the Vtable's job is done by the mapping's selectors |
| `metgrid.exe` | **horizontal interpolation** inside `woof prep` | no `met_em` files; the result goes straight into the prepared bundle |
| `real.exe` | **initialize** inside `woof prep` | produces `wrfinput`/`wrfbdy` equivalents under `wrf-native-input/` |
| `wrf.exe` | `woof sim` | CUDA-only; this is the one stage that needs a card |
| all four, scripted | `woof prep` | one command, one output root, one proof |
| the whole thing plus plotting | `woof go` | authority, fetch, manifest, prep, sim, render: the same six commands, run in order |

The single most useful sentence: **`prep` is all of WPS plus `real.exe`;
`sim` is `wrf.exe`.** Everything below elaborates one of those two.

## The vocabulary

**Mapping.** The table that says how to read one model's files: which
records carry which fields, at which levels, in which units. It is what
a Vtable is for, generalized past GRIB (a mapping also covers NetCDF
sources) and it is data, not code. Adding a new model is a mapping, not
a new code path. `woof prep --show-source MODEL` prints the one a named
source uses; `--list-sources` names all of them.

**Descriptor.** The mapping's companion: the geometry and cadence of the
source grid, so the reader knows what shape the records it selected are
in. Together, mapping + descriptor are the full "how to read this model"
statement.

**Composition.** One prepared state assembled from more than one source:
a global model for the upper air and a regional one for the surface,
say, or an analysis stitched to a forecast tail. WPS has no equivalent;
the nearest thing is running `ungrib` twice with different prefixes and
hoping `metgrid`'s interpolation order does what you meant. Here it is
declared in a file and the receipt says which field came from which
source.

**Supplement role.** A named slot a composition binds an extra file
into: `--supplement ROLE=PATH`. The role is the composition's word for
"the thing that plays this part" (soil, sea surface, orography) so
the same composition works with different files supplying the same role.

**Contributing mapping.** When a composition draws on a source that has
its own mapping, `--contributing-mapping ROLE=PATH` binds it. The
composition says what goes where; the contributing mapping says how to
read it.

**Front-door manifest.** The list of exactly which input files this
preparation will read, with a SHA-256 for each, plus a digest over the
whole list. `woof fetch --author-front-door-manifest` writes it; `woof
prep` takes it as `--source-manifest` with its `--source-manifest-sha256`
and refuses if a byte moved. WPS has nothing like it: `ungrib` reads
whatever `link_grib.csh` linked, and nothing records what that was.
This is the "front door": the one place the outside world's bytes enter,
named and hashed.

**Authority materialization.** Resolving the physics your config selects
into an explicit, complete statement, *before* preprocessing rather than
after, and writing it beside the run. A shipped physics profile supplies
every switch your config is silent about and never overrides one it
states; where the two disagree the step refuses and names each key. The
WPS analogue is the part of `namelist.input` `real.exe` happens to read,
except that here the resolved suite is a file with a digest, and every
later stage is bound to it.

**Prepared bundle** (also **prepared root**, `--prepared-root`). What
`prep` writes and `sim` reads: the model-ready state, its static fields,
its decoder receipts and its proof. The rough analogue of a directory of
`met_em` files plus `wrfinput`/`wrfbdy`, except that it is self-contained
and digest-bound, so it can be moved to another machine and the receiving
runner can prove it is the same bundle.

**Proof** (`proof.json`). The receipt at the root of a prepared bundle:
what was decoded, from which records, onto which geometry, under which
physics authority. `prep` prints the `sim` command with the proof's
digest already filled in, and `sim` refuses a bundle whose digest does
not match. There is no WPS equivalent: this is the part that makes a
preparation reproducible rather than merely repeatable.

**Mapped engine.** The code that actually decodes mapped source bytes.
The default is the Rust engine. `--mapped-engine python` selects the
slower Python decode path and is a documented **workaround**, kept
reachable so a decode the Rust engine gets wrong has a way around it
while the defect is fixed, not a mode to prefer.

**Preprocess backend.** Which processor runs the interpolation inside
`prep`: `cuda`, `cpu`, or `auto` (the default). `auto` selects the CPU
backend wherever CUDA is unusable and says so in one line. The CPU
backend is the packaged Rust bridge, checked against the CUDA path by the
preprocessing parity suite. Their prepared start states are close but not
bit-identical ([TILES.md](TILES.md)); use the same backend for both arms of
a forecast comparison. See [WITHOUT-A-GPU.md](WITHOUT-A-GPU.md).

**Run stamp.** The timestamped folder every run of `go`, `sim` and
`render` writes into, so two runs of one configuration never overwrite
each other. `run-<launch>Z_i<init>Z`. See
[run-output-folders.md](../run-output-folders.md).

**Domain wizard** (`woof domain`). Sizes a nest ladder against a card's
VRAM and emits both `namelist.wps` and the experiment config, consistent
with each other. It replaces the arithmetic you would otherwise do by
hand before writing `&geogrid`.

## Words that mean what you expect

`namelist.wps`, `namelist.input`, `WPS_GEOG`, `wrfout`, `wrfinput`,
`wrfbdy`, cycle, lead, nest, parent ratio, `e_vert`, `dx`, `dt`. These
were not renamed and are not redefined.

One thing worth knowing about the outermost domain: `d01` has no
parent, and the two halves of a WRF pair spell that differently. The
WPS user guide asks for `parent_id = 1` in `namelist.wps`, WRF's
`namelist.input` uses `0`, and namelist generators write either in
either file. `woof import-namelist` and the namelist support report
take `0` or `1` for `d01` in both files, and list the spelling they
normalized as an advisory; any other value is refused, because the
outermost domain cannot have a parent. See
[CONFIGURATION.md](CONFIGURATION.md) for the rest of the domain-layout
mapping.

## Bringing a real case over

`woof prep --namelist-support-report` reads your unchanged
`namelist.wps`/`namelist.input` pair and reports, key by key, what is
supported here and what stock WRF does with it: two separate verdicts,
so a key that differs is visible rather than silently absorbed.

`woof import-namelist` translates the pair into a config, and closes
with a report of every key by section, the provenance of each pin, the
substitutions it made by name, and what it could not implement.

Start in a new output directory. Numerical identity against your
existing WPS/real output is not promised (interpolation and masked-donor
policies are explicit here and may differ while remaining structurally
comparable) so compare inventories first and fields second.

## See also

- [FIRST-LIGHT.md](FIRST-LIGHT.md): the whole path, with timings.
- [PIPELINE-STAGES.md](PIPELINE-STAGES.md): what each stage does and
  what it leaves behind.
- [WITHOUT-A-GPU.md](WITHOUT-A-GPU.md): how far the chain runs with no
  card in the machine.
- [WRF-INTEROP.md](WRF-INTEROP.md), where this model deliberately
  diverges from WRF, and why.
