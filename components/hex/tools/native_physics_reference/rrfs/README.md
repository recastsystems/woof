# Pinned RRFS native reference

Primary model and gitlinks are in `pins.json`. This extends the shared native
reference flag editor, public-input downloader and immutable publisher. Neither
the primary reference repositories nor any physics arithmetic is edited.

Use the remote runner for builds and native production under the pool mechanics. The
integrator's later host rule prohibits a the archive host native production fallback;
the archive host is limited to the completed one-time cross-host check. All builds use
`taskset -c 8-23`, with `-j4` supplied by
`build_reference.sh`. Source the lane environment and set `MPAS_UNIT` first.
Arguments are work directory, pinned source, recipe (`gnu-strict` or `gnu-o3`),
core (`atmosphere` or `init_atmosphere`), MPICH prefix, PnetCDF prefix and pinned
MPAS-Data directory. The actual compiler wrappers capture every object command,
including nested NOAA makes. `audit_flags.py` requires zero exceptions. The last
completed compilation determines an object's provenance after a rebuild.

`prepare_build.py` exports verified git archives, preventing copied submodule
gitdir files from pointing back into the reference. Build-only diffs propagate
strict flags into C/C++, disable redundant network checkout, propagate MPAS
preprocessor flags into SMIOL, and compile the unchanged TEMPO v3 table program
against the atmosphere's MPAS precision modules. The obsolete MPAS TEMPO utility
is not adapted. An explicitly requested incremental retry can set
`RRFS_RESUME_BUILD=1`; it must use the existing pinned exported build tree.

After the object audit passes, `publish_build.py BUILD REF --core CORE --recipe
RECIPE` atomically publishes and seals the binary, tables, source diff and audit.
Each publication appends its hashed manifest and invocation notes to `READY.md`.
A build receipt is not a forecast identity receipt.

`verify_native_reference.py --pins-document <REFERENCE-PINS.json> --out <receipt>`
authenticates the complete five-build authority and four native table recipes
recorded in the `rrfs` pins block. It hashes every published member, independently
checks every audited object's final captured compiler invocation and archive
copy provenance, and binds each table to its actual utility, audit and toolchain.
Historical compiler paths are checked as metadata because an archive transfer
can relocate a publication. Raw object bodies are not archived or rehashed by
this verifier. Missing recipes, groups, single-precision flags, strict flags,
changed bytes, diagnostic authorities and mismatched table recipes fail.
Success proves published build/table provenance; it does not run weather or
establish physics replay, forecast identity or lane acceptance. Put its receipt
under the lane's receipts directory and temporary work under the prescribed
`TMPDIR`.

`cache_ci.py` verifies archives downloaded by the shared `download_inputs.py`,
extracts the fork's three GSL CI cases, copies their original decks, and publishes
all inputs and byte hashes. Downloaded TEMPO tables are explicitly labelled;
they do not substitute for the strict native table build.

`generate_decks.py` takes explicit source/workflow paths and runtime JSON. Required
runtime keys are `start`, `duration`, `output_interval`, `restart_interval`, `dt`,
`substeps`, `mesh`, `limited_area`, `graph_prefix`, `lbc_interval_hours`,
`input_filenames` and `workflow_variables`. The last dictionary resolves all
remaining workflow placeholders, including radiation cadence, soil levels, DA,
I/O and stochastic switches. Source registries and control.F determine suites
and choices; `deck-specs.json` records companion and variant rows. Included
registries are expanded with their relative paths. Nested stream lists are copied
and hashed. Unsupported workflow namelist groups are preserved verbatim and
identified in the index, rather than assigned invented model semantics.

`cache_cases.py` takes an explicit index with optional `decode` and `cases` rows.
Decode rows contain named paired GRIB inputs, start/stop and interval; ungrib
concatenates each pgrb2/pgrb2b pair as the workflow does. Case rows select workflow
namelist/stream templates, substitutions, input names, decoded directory, ranks,
expected outputs and publication path. Static, invariant, case7 and case9 all run
the audited strict init executable; no Python interpolation or field repair is
performed. Run CPU jobs under the lane cpuset. Keep raw input provenance and
immutable case products; discard duplicate concatenated input bytes after decode.

`run_tables.py` runs the audited strict TEMPO utility with explicit MPI ranks,
hashes all three v3 tables and publishes them only after successful completion.
Its manifest binds the utility and both table-build objects to the compiler audit.

`run_forecast.py` runs a generated deck with an explicit case file mapping and
ordered table directories. It verifies the executable, audit and every nested
deck file against their manifests before running MPI. Successful runs and native
failures both retain an exit/status receipt with every emitted file hash. The
receipt records resolved options, rank count, CPU affinity and selected table
overrides; it does not infer an identity tier from successful execution.

Tests: `python -m pytest tree/tests/test_rrfs_native_reference.py -q` from the hex
tree. The real pinned-source deck test skips when external sources are absent;
compiler receipt and include-resolution tests remain portable.

`generate_case_matrix.py` expands an explicit request into static/invariant,
case7 and every hourly case9 stage, including both forecast endpoints. Soil
companions, dates, mesh inputs, geography and clocks are request rows, with no
case-specific defaults. Precomputed static/invariant files remain explicit.
`run_case_matrix.py` invokes the same native case runner, saves real failure
receipts, and reuses native decoded intermediates only after byte verification.
When a verification directory is named, it also requires each input's S3 witness.

`download_verified_inputs.py` extends f0's public downloader. A directory lock
prevents overlapping resumable downloads. Every downloaded or reused object
must match the public S3 Content-Length and whole-object MD5 ETag; the receipt
then records SHA256. It never interprets or repairs weather bytes.

`catalog_decks.py` copies and indexes original source decks verbatim, including
native XML peculiarities. `record_pins.py` updates only the rrfs block in the
shared pins document from immutable publications whose file hashes still match.

The optional last build argument `secondary` exports c3f78a5 and its own physics
gitlinks from local history. The optional stochastic gitlink is recorded as
inactive because this secondary dry comparison uses STOCHASTIC_PHYSICS=false.
Every other gitlink is exported at its pin. The primary remains the referee.
Compiler temporary files use the caller's prescribed lane TMPDIR. Archive-extracted
libphys objects must match an audited compiled object's basename and SHA256.

`publish_deck_matrix.py` expands explicit mesh/date runtime rows, publishes each
group immediately and writes the top-level decks/INDEX.json with all forced
companions and immutable paths. Optional selected_names is a table selection
over existing source-derived rows. It does not add a suite-specific code path.
`publish_forecast.py` verifies every output/deck hash against a completed real
native receipt before sealing the forecast under pin-set/recipe/mesh/date/deck/
rank/label. Runtime input bytes stay in their case/table caches.

`measure_dycore.py run` accepts a common hashed dry-deck manifest, explicit input
mapping and a published strict executable. Both primary/secondary RRFS and f0's
existing v8.4.2 executable can run it without rebuilding f0. `compare` requires
completed same-host/same-rank runs with identical input/deck hashes, invokes the
Rust comparator on every field/stream, and records the first differing field,
word and integration step. A source-version difference is a measurement, not a
forecast acceptance gate. No field exclusions or numerical tolerance are used.

Diagnostic initializer builds use the ninth `build_reference.sh` argument
`bounds-snan`: strict recipe plus `-g -fcheck=all -finit-real=snan` on unchanged
native source. They remain private witnesses; reference publication rejects them.
`publish_inputs.py` seals raw public inputs only after copied bytes match every
S3 checksum witness; native ungrib decodes these unchanged bytes.


`transport_inputs.py` transports raw input chunks through the remote runner's 200 MB mirror
limit, verifies the publication digest at both ends and exposes only complete
files. Build-host results return through the runner's output folder; permanent publications are on the archive host.
`run_forecast_matrix.py` executes explicit ready rows without blocking unrelated
rows, retains terminal failures and publishes only completed native forecasts.
Restart rows name their successful parent's emitted file. The supplementary
immutable index name is explicit request data; published indexes are never
replaced. Native restart mode and clock are explicit runtime rows. Cross-host
identity uses `identity.py --mode cross-host`, fixed equal ranks and every stream.

Canonical initialization remains subject to unchanged native behavior. Case
receipts record SOURCE-PINS and host; a diagnostic initializer's weather outputs
cannot be published as the primary reference. The patch under diagnostics/
contains exactly two optional aerosol pointer NULLIFY statements at fork
mpas_init_atm_cases.F:4414 and :7979. It is private counterfactual evidence only,
not part of either primary recipe. Applying it to a private offline export keeps
all strict compiler flags; it does not establish parity with an aborting pristine
initializer. Full case7/case9 witness and comparisons are in lane receipts.
Table pins cover the immutable tables/ publications as well as native/ builds.

`field_inventory/` builds the Rust `mpas-rrfs-field-inventory` executable with
native NetCDF. It inventories every floating variable's original IEEE words,
including NaN payloads and signed infinities, without normalizing or rewriting
data. Run it on each emitted file, retain its JSON, and bind the file, executable
and source hashes in the orchestration receipt. Finiteness is diagnostic evidence
and does not establish forecast parity. Build/test through the build host `cargo build --offline --release -j4` and `cargo test --offline -j4`.

`decode_hourly.py` takes explicit valid-time rows, verified raw-file SHA256/size,
an immutable source publication hash, native ungrib/Vtable hashes, and an explicit
rsync source/argument list. It fetches one paired input at a time, rejects changed
bytes before invoking native Fortran, and retires only the copied raw pair after
intermediate hashes are recorded. Each hourly manifest retains the original
source pins. No suite, date, region or mesh is selected by generic defaults.

Dry source-version measurements retain the exact nonfatal native missing
diagnostics-interval messages in their completion witness. Every other logged
ERROR/CRITICAL/FATAL, including the witnessed SMIOL ENOSPC error, still rejects
completion. This does not change the forecast publisher's completion policy.

`recover-measurement` preserves an original rejected wrapper receipt and log,
requires native exit0/completion with no unrecognized errors, verifies the strict
executable/audit/toolchain and every original emitted-file hash, and requires a
matching disposal hash for retired files. Explicit retained timestamps include
every field. Its new manifest is labelled measurement-only; no forecast or
full-output identity is inferred from the retained subset.

Static/invariant rows omit GFS intermediates only when their rendered workflow
namelist explicitly sets `config_met_interp`, `config_input_sst` and
`config_blend_bdy_terrain` false.
Meteorological stages still require unchanged native decoded inputs. This is an
orchestration dependency correction, with no native options or defaults changed.

`stage_geography.py` verifies explicit archive size/SHA256, runs native tar for
both metadata inspection and extraction, rejects traversal and link entries,
and checks the caller's work-directory byte limit before creating its output.
Its index binds archive paths/hashes, extracted size, native tar hash and the
exact listing; extraction rechecks that index. All geography dataset choices
remain request data. The NCAR source archives are retained on the archive host's immutable
data disk even while their disposable extracted copy is used on the build host.

For large geography archives, `extract --sparse --reserve-bytes N` sends each
already inspected regular tar member through native `cp --sparse=always`.
Zero byte runs become filesystem holes; reads return the original bytes.
The extractor checks the next member's full logical size against the remaining
allocation budget before creating it, and records physical allocation and both
native tool hashes. A byte preservation test includes nonzero data on both sides
of a large zero run and verifies refusal before an oversized member is created.

Hourly decoding supports an explicit `archive_sink`: transfer each finished
endpoint to the archive host, invoke `receive_hourly.py` with its native manifest/input/
decoder hashes, and retire local intermediate bytes only after the receiver
verifies and seals every file. Corrupt transfers retain the original local
output. The INDEX records each permanent publication. Explicit storage rows
reserve both raw-pair copies, the maximum decoded output and concurrent-work
headroom before execution. This avoids accumulating the roughly2GB endpoints
in lane scratch; permanent archives stay on the archive host's data disk.
The runner resolves its working directory, decoder and Vtable before changing
the native process directory, so relative command-line paths remain valid.

After every requested endpoint is sealed, `receive_hourly.py --index INDEX
--destination ARCHIVE --ready READY` revalidates every child publication, native
manifest, source pin and intermediate-file hash before sealing the parent index.
The immutable parent gives subsequent case/LBC producers one authenticated
inventory covering both forecast endpoints; it does not certify initialization
or forecast identity.


The completed dry source measurement is published under `ref/rrfs/forecasts/dry-source-version-x1.40962-20230331-v1` (publication SHA256 `4bea97a7e658a88234cf60190ab89f08db22f4f5b118633bdd6de7e16f41b1d0`). All three strict runs complete 24 h on the build host, MPI1, common x1.40962 input and dt360. Primary 1d6e370e first differs from v8.4.2 in `exner` at step 1, index `[0,0,3]`, bits `3f7ebc4b` versus `3f7ebc4a`. Secondary c3f78a5 has zero differing stored words across all fields at the retained initial, first-step and final snapshots. Intermediate outputs retain hashes; this publication does not establish their field identity. Original manifests and the measurement-only recovery witnesses remain archived.


`run_forecast.py` materializes native-writable case inputs, records their pre-MPI hashes, and inventories both changed and same-byte rewritten inputs as emitted output. Immutable table inputs remain linked. `identity.py` compares exactly the authenticated native emitted-file inventory through the Rust comparator. `workflow_deck.py` renders explicit workflow stream overlays and companions from an authenticated source deck.

`replay_io.py --request <JSON> --out <new work directory>` executes an explicit cold phase and two identical restart phases, with every completed phase transferred to the archive host through `receive_forecast.py` before scratch retirement. Request data supplies clocks, settings, tables and their hashes, rank count, storage reservation, MPI/comparator paths, and archive-host environment/receiver paths. Cold restart and DA companion files remain private inputs until both reruns finish. Native completion and full-stream rerun identity are required. This infrastructure has39 passing checks; actual restarted native forecast acceptance is separately recorded in receipts. `receive_case.py` authenticates transferred static/case products against the pristine strict initializer and geography manifest.


The complete original GSL workflow-stream IO replay is sealed at `ref/rrfs/forecasts/native-io-replay-original-ci-v1`, publication SHA256 `244f5ee3558b66f2e2ad33aed883aaa53c24a7b48b5f60bd4eed40516b674d42`. All three cases complete 1 h cold and two identical 1 h restarts on the build host, MPI1, strict generated TEMPO tables, `config_jedi_da=true`, `config_do_DAcycling=false`. Every emitted field in all13 NetCDF files per restart has zero differing stored words (28,681,352 words per case), including the rewritten DA input/output; source ingress remains unchanged. Literal file bytes differ. This is an IO-runner witness, with canonical12+12h and DAcycling=true acceptance recorded separately.


The separate `config_do_DAcycling=true` original GSL replay is sealed at `ref/rrfs/forecasts/native-da-cycling-original-ci-v1` (publication SHA256 `f4e042a4254f2a59cec9f0484180323e78e5b1292941c8a1ec328a1c4273f61b`). All three cases complete nine strict native phases in total, MPI1 on the build host, 1 h cold and two 1 h resumes. All13 NetCDF files and28,681,352 stored words per restarted rerun have zero differences, including rewritten DA companions. Source ingress remains unchanged; literal file bytes differ. Canonical12+12h,24h and Hex acceptance remain incomplete.


The original GSL GFS winter case also completes 12 h cold and two 12 h restarts with DA cycling enabled, MPI1 on the build host. Its native LBC input spans48h. The sealed index is `ref/rrfs/forecasts/native-hour12-restart-original-gsl-gfs-v1` (publication SHA256 `a6be81a5800e0b1db50663eeeabf4959d99c0b2ce0b3f3642e7863911eed2186`). Every27 emitted NetCDF files and42,957,831 stored words in the restarted reruns match; literal file bytes differ. Explicit `da_state` native `output_timelevels="0 12 24"` emits the required restart companion clocks. This is the original imported GSL mesh/date scope, not canonical cases or Hex parity. Native continuity against a continuous24h forecast is measured separately.


Native restart continuity differs on the original GSL GFS case: strict MPI1 on the build host, DA cycling enabled, continuous24h versus12h+12h has15,541,584 different stored words across all27 resumed same-clock files (42,957,831 words). The first difference is `cape` in the first post-restart diagnostic file. Both complete emitted inventories and earlier unmatched continuous clocks are retained at `ref/rrfs/forecasts/native-original-gsl-gfs-24h-restart-continuity-v1`, publication SHA256 `f602e40e95e2dcb96dd50ce8a60b297e5b848dcefee4eca0ecfe063664199fd5`. Only the restart/start/duration clock controls differ; all other scientific configuration, source, tables, host and rank agree. This is a native continuity measurement, separate from zero-word restarted rerun identity and from canonical/Hex acceptance.


Native first-guess files remain required for workflow static/invariant templates
when boundary terrain blending is enabled, independently of meteorological
interpolation and SST input (`mpas_init_atm_cases.F:4468-4478` at 1d6e370e).
All three options must be explicitly false to omit decoded inputs. Complete
compressed geography images can be remounted with `reuse_compressed_view` after
authenticating the original whole-member native-cmp receipt, archives, images,
indexes and mount tool. New views stay read-only and are released on exit.

`summarize_finiteness.py` validates the actual Rust inventory keys (`nan`,
`positive_infinity`, `negative_infinity`, `first_nonfinite`) and requires an
inventory for every emitted file. It reports native nonfinite words without
changing or excluding them. A missing generic `nonfinite` key is not a zero
count. The separately labelled chemistry-on CONUS12km case pair is a measured
additional option configuration, not the workflow chemistry-off canonical case.
The complete diagnostic-companion supporting matrix records 25 successful
restart reruns and six native cold failures; its index does not certify the
required canonical 24-hour or Hex comparisons.


`receive_geography_images.py` authenticates every original archive and native
compressed image against the pinned whole-member byte-verification receipt
before publishing on the permanent host. `create_region_reference.py` replays
the explicitly requested unchanged upstream `create_region` tool twice with
pinned source files, polygon and global grid; it records original output hashes
and native headers. The tool computes the subset; no mesh processing is added
to the launcher. Both native grid and graph bytes must match. The region/tool
choices are request data, and the region still requires fork-native static,
invariant, initial-condition and hourly boundary-condition stages.

`run_case_pipeline.py` executes an explicit dependency-ordered native case
request. Each hourly endpoint is pinned to its immutable publication and kept
only while stages use it. The Rust IEEE inventory covers every native output;
nonfinite values are recorded without disabling a scheme. Every successful stage
is transferred to the permanent archive host and authenticated before private
outputs are retired; static/invariant dependencies remain until their last use.
`receive_case.py --retire-output-root PATH` independently checks every transferred
output against the sealed archive before deleting any received output. Failure
receipts preserve actual native status and do not create completed products.

Case9 rows include the same monthly TEMPO aerosol runtime file as case7, as
`exrrfs_lbc.sh` requires. Chemistry choices come from explicit workflow companion
templates and source-authorized request replacements, not generic defaults.

`publish_sparse.py` supplies native GNU cp as the copy adapter to a separately
loaded copy of f0's unchanged publisher. All original file selection, SHA256,
READY publication and sealing code remains f0's. Explicit `--reflink=never
--sparse=always` represents original zero byte runs as filesystem holes, with
every source/destination SHA256 checked. No weather arrays or file headers are
normalized. The archived SPARSE-COPY-RECIPE.json pins the native tool and options.
Existing immutable publications are never changed.

`stage_geography.add_pinned_datasets` authenticates every original member of an
explicit additional dataset before adding a geography path. It preserves the
base compressed-view receipt hash and records the supplementary inventory and
source authority. The fork's GWD statistics read USGS `landuse_30s` independently
of the static interpolation's selected land-use dataset
(`mpas_init_atm_gwd.F:603-647` at `1d6e370e`); a MODIS archive alone does not supply
that dependency. The original f0 USGS files can be reused after checking all
member hashes, without copying, transforming or replacing any weather field.

`run_published_decks.py` extends `replay_matrix.py` with explicit groups of
published native input and deck companions. Each request pins source case
options, strict initializer binary, deck publications and runtime tables. Each
case member is checked against both the sealed publication and native manifest;
soil-layer counts must match the selected deck. Missing hourly inputs remain
pending until the actual native publication exists. Every strict/O3 forecast
uses the unchanged runner and all workflow stream lists. Failed native rows are
retained in the complete attempt summary. This command's status reflects native
completion; identity claims require the matrix's actual full-word comparisons.

When a forecast host uses an authenticated private copy of a sealed build, a
`replay_matrix.py` row may name `archive_build` for the build's path on the
archive host. `run_published_decks.py` accepts the corresponding request mapping
`archive_builds`. The receiver still checks the actual binary, flag audit,
source pins and toolchain hashes against the forecast manifest before publishing.
The private copy and archive build must therefore contain the same audited bytes.

A published-deck group can supply explicit `replays`: ordered phases with a
simple `name`, positive `ranks`, optional configuration `overlay`, `input_files`
(path/SHA authority for a native partition), and earlier-phase `from_outputs`
or `compare` mappings. Each phase expands independently for the same deck and
recipe. The matrix consumes actual archived checkpoints and compares all native
fields; no phase may substitute another deck's or recipe's checkpoint. Extra
inputs cannot overwrite initialized weather. Clocks, filenames and scientific
settings are explicit request data; ordinary deck groups retain their behavior.

`compare_recipes.py` measures every emitted native field across two archived
recipes. A request binds both publication/manifest hashes, binary and toolchain
authorities, the Rust comparator, and every differing generated table hash.
Host, rank, weather inputs, scientific settings and stream contracts must match.
The measurement retains recipe table differences and reports all word changes;
it grants no same-recipe or Hex identity claim. Each pair's authenticated private
weather copies are retired after comparison, leaving the the archive host archives intact.
### Bounded native output archival

`replay_matrix.py` accepts explicit per-run `readonly_inputs` and
`output_archive` data. Each linked case filename must match exclusively `input`
streams; DA and restart `input;output` streams retain private writable copies.
The archive helper waits for a later native `Begin timestep` clock, checks every
raw field digest and stored clock with the Rust comparator, transfers the
original NetCDF bytes, authenticates the remote SHA256 and the unchanged local
file, then retires the private copy. Explicit `output_timelevels` retain the
initial DA state and checkpoint companions. Final collection occurs after
native exit. Unknown files stay local and enter the ordinary final manifest.

`run_published_decks.py` also accepts byte-authenticated
`ancillary_publications` for invariant and orography companions, and passes
explicit storage choices from group data. Acquired files use native rsync sparse
storage; every original publication hash is verified. The existing receiver
checks the complete output inventory and seals all output bytes before retiring
received staging copies. `ARCHIVED-OUTPUTS.json` records original hashes,
inventories, clocks and transfer authorities. The archive host owns the permanent archive.

Qualification: `native-closed-byte-archive-qualification-v1` is a complete
24-hour strict native hrrrv5 CONUS-cut execution on MPI1 on the build host from committed
`b6023ff`: all 55 NetCDF streams and 77,891,334 stored words match the original
copy-all baseline. Publication SHA256
`7024fa17cfeabf74d778747453bc28aad92d59ab173df4b5b5cea66b19bcebee`.
The initializer is the explicitly labeled smoke/dust-ON support branch.

# Native Thompson lookup companions

`run_tables.py --kind thompson --stage-only` runs the existing fork
`build_tables` utility and retains its four original unformatted lookup files.
The utility binary and both utility objects must match the audited primary
build. `--kind tempo_v3` remains the default. Neither route edits native source.
Source authority is `src/core_atmosphere/utils/build_tables.F:10-19` and
`atmphys_build_tables_thompson.F:37-114` at `1d6e370e`.

Run table construction on the authorized CPU host. Stage-only products are
private transfer inputs; authenticate and publish them on the archive host before using
them as an archive authority. Preserve the recipe in the manifest and link the
original basenames into forecast scratch. A table build is supporting evidence;
the integrator owns merged-tree forecast acceptance under the owner's 03:00Z ruling.
Table rerun identity is measured on the original bytes, separately from weather
forecast identity. `record_pins.py` records both TEMPO and Thompson publications
and checks that every manifest table hash belongs to the sealed publication.



`publish_diagnostic_build.py BUILD PRISTINE_SOURCE REF --core atmosphere`
seals a separate bounds/SNaN diagnostic under `REF/diagnostics/`. It reaudits
all objects, requires `-g -fcheck=all -finit-real=snan` on every Fortran object
including extracted archive copies, and compares each original code file with
Git archive bytes at the fork and every submodule pin. C/C++ retain strict
flags. The diagnostic publisher records all provenance but authorizes no word
exclusion. `publish_build.py` continues to refuse diagnostic binaries as primary
references. Native-defect policy requires actual checked full-model affected-word
measurements in addition to the build; a successful build alone is insufficient.

The separate `bounds-snan-integer-poison` diagnostic adds
`-finit-integer=-2147483647` to the existing checks. Select it as the ninth
argument to `build_reference.sh`, after `primary`, and build through
a bounded build slot with the same pinned toolchain and inputs. The compiler audit
requires the named flag on every Fortran object and rejects a later integer
initialization override. The manifest records the poison and hashes the shared
diagnostic flag definition. Source arithmetic, primary binaries, C/C++ flags
and the earlier real-SNaN publication are unchanged. A compiler flag does not
prove that a particular allocatable integer array was initialized; the owner
must capture and compare its actual boundary state. r13's original full
real-SNaN controls failed the same affected-word-set relation; that failed
witness remains evidence, without an exclusion or production source repair.

The complete source-unchanged diagnostic atmosphere is published at
`ref/rrfs/diagnostics/4b06780d2cbfd4a8bc732817ae6de31cf64ca1f65e885abaf9e0c1e820191863`.
Its publication SHA256 is
`108069d37cd964c92801c74762e4ebe1d50121cb5a506e6482f7f5b41df5d90f`;
all 691 objects passed the audit. The binary SHA256 is
`2e41c29be65af879be14bfcb0509119d6152d1a5c3ef54934f2c67a880c8ad67`.
This the build host diagnostic supports same-host checked runs alongside the unchanged
primary executable. It supplies no affected-word mask or exclusion ruling.

The original measured GFL ingress contract is published at
`ref/rrfs/diagnostics/original-measured-gfl-checked-ingress-contract-v1`
(PUBLICATION SHA256 `9d967c0f07ef6b67e1cb96af49ed91b8bfd906280261044cd03238759daea243`).
It binds the full source-unchanged checked atmosphere to six exact original CI
GFL decks, their public input/table publications, the original two-rank
partition, and 24 untouched input summaries from 48 ordinary native executions.
Read `INDEX.json` and `README.txt` to materialize the exact scientific ingress.
Original public TEMPO table bytes differ from regenerated recipe tables and
remain required for this contract. The recorded ordinary runner hash differs
from the file at its recorded commit; historical per-run binary hash fields
were absent. Both provenance limitations are preserved. This publication runs
no weather forecast and proves no checked affected-word set or exclusion.

Small-global RUC9 companions now cover both brief dates: six authentic static,
invariant and case7 stages, eight NetCDF files, in
`ref/rrfs/cases/pristine-global-small-both-date-ruc9-index-v1`
(PUBLICATION SHA256 `2ff834c377b7fd670eb73c34f55d9f63db571e31bf267130a6d963d0445c3a93`).
The index authenticates original stage dependencies and finite output evidence.
These are explicitly smoke/dust-ON initialization companions on MPI1 on the build host;
canonical OFF and forecast identity are separate outstanding requirements.

`verify_diagnostic_contract.py PUBLICATION.json --expected-sha256 SHA --out
RECEIPT.json` authenticates the sealed original measured GFL ingress contract,
its diagnostic binary, generated decks/partition and required shared input
publication membership and bytes. Add `--runtime DIR --contract ID --ranks N`
to check a materialized runtime before launching the owner's diagnostic.
The verifier launches no weather executable and grants no native-defect
exclusion or physics identity. It preserves the recorded runner discrepancy.


The full integer-poison diagnostic is sealed at
`ref/rrfs/diagnostics/f54f0bb02b9fd0a504ec41873c4fc5f8bb90d8a49f28cdf18e130e127a505c13`,
publication SHA256 `39939c13df272c4dd262b04d11759e7d8ab016ca74674f2d84b20639ab13955a`,
binary SHA256 `0e2eb4e28fc3e4661e0407ac17c73c1b5a2a3ee4068f1e50bf2ef915b8515d8a`.
All 691 objects passed and all nine pinned source groups matched Git bytes.
The six unchanged original measured GFL ingress contracts are rebound at
`ref/rrfs/diagnostics/original-gfl-integer-poison-checked-ingress-contract-v1`,
publication SHA256 `cdfb5e7fd09c52c64729db7d6b4dc82843603b5d8a9942073a8b678dd21650ce`.
All 12 real runtime contracts (six deck rows, MPI1/2) verified 374 required
input roles through the committed verifier, without executing weather.
The original public TEMPO tables and failed real-SNaN witness remain unchanged.
This supplies the requested diagnostic build and exact ingress; actual wrapper
state capture, affected-word attribution and any exclusion remain unproven.

`verify_case_companions.py PUBLICATION.json --expected-sha256 SHA --out
RECEIPT.json` authenticates a complete published static/invariant/case7 index.
It checks every original stage member, the pristine strict initializer's binary,
source pins and audit, original-to-archived manifest bindings, static-to-case7
dependencies, soil/vertical levels, clocks and host, and the archived Rust
finiteness summaries and their word counts. All mesh/date choices come from the
index. It reads original bytes without decoding or changing weather fields and
launches no native forecast. Its receipt establishes case input provenance;
canonical OFF initialization and physics/Hex identity remain separate claims.

The completed GFL integer-state ruling review is sealed at
`ref/rrfs/diagnostics/native-gfl-integer-state-ruling-evidence-v1`.
`diagnostics/GFL-STATE-AUTHORITY-PROPOSAL.json` binds the exact pinned wrapper
and deep-module source, the original primary and integer-poison build
publications, the owner's 24 native comparisons and 12 complete checked
boundary receipts. The checked runs observed the named integer poison in
147,960 congestus entry words, with no unexpected values. The affected-word
test failed all 72 exact set comparisons; it authorizes no exclusion.

At the time of that pristine witness, the adjacent
`native-gfl-state-authority-proposal.patch` proposed initializing `ierrm(i)` beside
`ierr(i)` in the fork's wrapper and remained unapplied pending the owner's ruling.
The owner approved exactly that correction on 2026-10-14 07:45Z; its separately
labelled strict authority is described below. The pristine witness remains
unchanged. Captured-state substitution is diagnostic evidence, without
production replay acceptance.
The review authenticates existing metadata and build members; it does not
replay the historical raw captures or native executions.
The complete pristine RUC9 global companion catalogue now covers both lane
meshes and both dates: `cases/pristine-global-both-date-ruc9-companions-index-v1`
under the shared RRFS reference root. Its publication SHA256 is
`08b7b84db2db3749083b62b2648ff577e810f26210204f03a7c9987de221eb25`.
The twelve original static/invariant/case7 stages contain sixteen NetCDF files;
the original Rust inventories report 90,823,244 finite case7 floating words.
The existing committed `verify_case_companions.py` authenticates the complete
catalogue, original strict initializer, stage bodies and static dependencies:

```bash
python tools/native_physics_reference/rrfs/verify_case_companions.py \
  $MPAS_LANES/ref/rrfs/cases/pristine-global-both-date-ruc9-companions-index-v1/PUBLICATION.json \
  --expected-sha256 08b7b84db2db3749083b62b2648ff577e810f26210204f03a7c9987de221eb25 \
  --out "$TMPDIR/ruc9-companion-verification.json"
```

This is native input provenance at MPI1 on the build host. These case7 inputs explicitly
enable smoke and dust. Canonical OFF initialization, cross-host or rank
identity, physics replay and RRFS Hex acceptance are not established by this
catalogue. No forecast was launched to publish or verify it.

The complete retained foundation image sequence has a separate visual review
at `forecasts/foundation-v842-existing-sequence-review-v1`, publication SHA256
`24c53af78d44f5d94c09eab17a57fbd5c97b808e85290d7ae6173b253de9f459`.
All twenty-four original PNGs were reopened, all three panels examined, and the
available 6/12/24-hour evolution compared. The review records per-image hashes,
actual observations, each scale's white threshold and the coarse maps' limits.
These are existing v8.4.2 foundation images; the review provides supporting
evidence and does not supply an RRFS visual forecast. No new image or forecast
was generated, and no intermediate sequence or absent time series is claimed.



The complete authority verifier is sealed at
`ref/rrfs/diagnostics/native-reference-authority-byte-verifier-v1`
(publication SHA256 `e4821202f19e2564315ab6d8d48474316e6b77d2be9dd677e6e5f9997c82ae8e`).
The copied service and its dependencies reproduce the committed result for five
native builds, 2,285 captured object commands and 14 native table files. This
publication contains provenance evidence only; native execution, raw-object
rehashing and numerical acceptance are outside its scope.


The complete original global companion index is sealed at
`ref/rrfs/cases/pristine-global-both-date-all-soil-companions-index-v1`
(publication SHA256 `e73c06c881920666a719898ea78c0e53b31145351e896ecb4d07277d006b1206`).
It covers both lane meshes, both dates, RUC9 and four-soil initialization:
24 pristine native stages, 32 NetCDF bodies and 180,878,248 finite case7 words.
Run the committed `verify_case_companions.py` with that publication/hash and
`--out "$TMPDIR/complete-global-companions.json"` to authenticate all stage,
source, static-dependency and archived Rust finiteness evidence. This is
MPI1 on the build host chemistry-on input support. Canonical chemistry-off, rerun/rank/
cross-host identity, physics replay and RRFS Hex acceptance remain unproven.

The owner approved two narrowly scoped authorities on 2026-10-14 07:45Z.
`ruled_reference.py` stages and publishes them separately under `native/derivatives/`.
The initializer has exactly the two approved NULLIFY statements and the exact
approved binary hash; it may generate canonical case 7/9 only. Each case and any
forecast using it binds the initializer patch, binary and ruling, labelled
“init derivative (owner ruling 2026-10-14 07:45Z)”. The GFL strict referee has only
the approved ierrm initialization and is labelled “RRFS fork 1d6e370e + ierrm init
(owner ruling 2026-10-14 07:45Z)”. Pristine strict/O3 witnesses stay published.
`record_pins.py` records these under `source_corrected_builds`; they never replace
the five unchanged entries in `native_builds`. `cache_cases.py` authenticates the
initializer scope and `run_forecast.py` recovers its authority from actual selected
input manifests. Forecast publication verifies those bindings again.

Restart-only ingress inherits the approved initializer ruling from the completed
parent forecast and authenticates the selected restart bytes. Raw input transport
keeps the same authority per content address in `INPUT-AUTHORITIES.json`; aliases
of identical bytes cannot erase an existing initializer ruling.

`verification_manifest.py` produces the verifier's existing `mpas-native-run-v1`
contract from an unchanged sealed RRFS forecast publication. Its request pins the
publication, primary inputs and every stream-list/table companion. The native
completion banner, actual selected input bytes and every output member are
authenticated before Rust describes all stored record times. The generated
contract preserves the initializer ruling and any corrected atmosphere authority.
This is comparison metadata; r4 supplies the actual RRFS launcher and acceptance.

The single matching corrected-GFL reference requested by r13 is complete at
`ref/rrfs/identity/approved-gfl-canonical-single-visual-24h-reference-v1`,
publication SHA256
`586319eaccd9fc1a10f0ced353e85dc10f8ed72a4a0fc0fac6fca96d3561a7f1`.
It binds the separately labelled approved GFL atmosphere and canonical init
derivative, both full patch/binary/ruling authorities, and the original native
forecast publication `08fa788e0336ed64b0f1a0149d05f8921df74fa62c8b14d6f3b606c88a0fc59d`.
The native MPI1 on the build host forecast completed 24 hours on x1.2562/March, RUC9,
59 model levels, hrrrv5 companions with explicit GFL/sub3d1, 720-second steps.
All 77 NetCDF outputs were described by Rust and admitted by the unchanged
verifier: hourly da_state/diagnostics/output at steps 0 through 120 by 5, and
restart at 60 and 120. The sealed verification deck and native-run contract
provide r13's exact native comparison; r4 supplies the actual Hex route.
This establishes native completion and authenticated metadata, without a
Hex comparison, image review, cross-host or rank-invariance claim.

`canonical_catalogue.py` joins explicitly pinned canonical input products to
selected grid/static/invariant/UGWP files, boundary endpoints, tables and deck
indexes. It authenticates the sealed recorded native verification, original
case metadata and exact selected companion bytes. Every native stage must
belong to exactly one requested group; boundary endpoints must agree with their
native clocks and invariant/static dependencies. Dates, meshes and catalogue
membership remain request data. The service does not decode weather fields or
recompute archived finiteness and does not claim a numerical replay.

```bash
python tools/native_physics_reference/rrfs/canonical_catalogue.py \
  --request <explicit catalogue request.json> --out "$TMPDIR/catalogue.json"
```

The complete canonical catalogue is READY-listed at
`ref/rrfs/cases/approved-canonical-complete-companion-catalogue-v1`,
PUBLICATION SHA256
`4d0df206496973c437848efe4050ea36637c3152436f9b145295dd9fbb4483bf`.
Its 12 mesh/date/soil groups account for all 112 native case7/9 stages and
212 NetCDF files from the four approved source products, including 25 hourly
boundary endpoints for each regional date. It selects the exact retained
static/grid/invariant/UGWP files, 21 forecast tables and four generated deck
indexes. The sealed source evidence records 34,229,051,562 finite floating
words and zero nonfinite values; catalogue assembly authenticates that evidence
and rehashes selected companions, without rereading weather outputs or
claiming forecast identity. Every canonical case retains the approved
initializer derivative label, patch, binary and full ruling.


The one matching Noah-MP reference requested by r17 is complete at
`ref/rrfs/identity/approved-noahmp-canonical-single-visual-24h-reference-v1`,
PUBLICATION SHA256
`bcf1535ea930322fdfb7403dee38dfecdec0fc41f9ecbba598de527e0d80d2a7`.
The unchanged primary strict atmosphere completed 24 hours on MPI1 on the build host,
March x1.2562, canonical four-soil input, 59 model levels, hrrrv5 with
`sf_noahmp`, at the workflow's 60-second timestep and four acoustic substeps.
All 21 selected table bodies match the pinned owner request, including its
GSL UGWP limb table and final strict TEMPO tables. The scientific namelist
matches the owner; three stream ingress filename aliases are recorded.
The actual native grid is the canonical fork soil4 static file; consume the
published `verification-request.json` and `native-request.json` for exact
input, grid and table paths.

The 163-member original forecast publication is
`0fefcac401de6dd01186ec91cf1688b994ca477562fb41e032dad81c1bcca82b`.
Rust described all 77 NetCDF outputs and the unchanged verifier admitted
hourly da_state/diagnostics/output at steps 0 through 1440 by 60, with restart
at steps 720 and 1440. The independent verifier authenticates the archive,
selected bytes, source authority and record contract from a supplied checkout.
This is native completion and comparison metadata, without numerical replay,
forecast regeneration, Hex execution, visual review or cross-host identity.
Every input and derived forecast retains the full separately approved
initializer patch/binary/ruling label; the atmosphere has no source correction.
The interrupted first attempt remains a witness of a companion-selection
orchestration error, without a completed-reference or native-defect claim.
`workflow_deck.py` records timestep and substep overlays in both the actual
namelist and runtime metadata; this changes no scientific defaults.


The one owner-matching MYJ reference requested by r15 is complete at
`ref/rrfs/identity/myj-owner-matching-single-visual-24h-reference-v1`,
PUBLICATION SHA256
`b17a57d66370093f89a934ce7184c27ba7217c17089e13f44038281f03fa6e9f`.
The unchanged primary strict atmosphere completed 24 hours on MPI1 on the build host,
March x1.2562, MYJ with its native sf_myj companion and TEMPO ML disabled,
at the owner's original 720-second timestep. Scientific namelist and streams
match exact owner bytes, as do all 20 selected table bodies, including the
original public GSL TEMPO and UGWP tables. The owner's original optional
smoke/dust-on input, ancestor grid and original initializer publication are
preserved separately in `cases/myj-owner-matching-optional-inputs-v1`,
publication `9d2d4c9a69a8d2de874a10a9d4e9d21224e5b92f408ff36eb5d96d8b683e29f8`.
This input has no canonical workflow-off or approved initializer derivative claim.

The original 113-member forecast publication is
`97bd8e10687e2c4f53e4b40c3319e65b522bcdde492d2880ed9e9a43ce84d3f7`.
Rust described and the unchanged verifier admitted all 52 NetCDF outputs:
hourly diagnostics/output steps 0 through 120 by 5 and restart at 60 and 120.
DA-state is absent with the owner's `config_jedi_da=False`.
The sealed native-run and comparison deck bind actual selected inputs and
tables; the input filename alias used by the verifier preserves exact bytes.
Independent supplied-checkout verification authenticates the original archive,
owner selection and actual record contract, without numerical replay,
forecast regeneration, Hex execution, cross-host identity or visual acceptance.


A sealed availability snapshot at
`ref/rrfs/diagnostics/all-cached-native-availability-v1`, publication
`8e951debd5b5a5c4678d1b1a1399bdd8e7cb56ad05bb02f5cbf2648a53b77bc1`,
authenticates all 211 current forecast-root publication metadata files and
189 declared native manifests. All declared members are present, including
5,415 NetCDF outputs across 160 strict and 29 O3 forecast publications;
22 supplementary publications are explicitly separate. Missing legacy runtime
metadata remains null. This checks archive availability, without rehashing
weather bodies or implying full requested matrix coverage, scientific identity
or image review.
### Archived native field comparisons

`recompare_published.py` authenticates complete native forecast publications and
uses the pinned native Rust comparator on every emitted NetCDF field. Requests
name each immutable publication, comparison mode and comparator digest. A
replica mapping changes only storage paths; publication and output hashes stay
mandatory. Initializer and GFL derivative authorities and labels must match.
`--stage` copies unchanged bodies for bounded the build host transport; no weather
decoding or arithmetic occurs in Python. Numerical execution exits nonzero for
any differing field word, including a rank-spread witness.

The historical primary strict hrrrv5 March x1.2562 evidence is sealed at
`ref/rrfs/identity/primary-global-hrrrv5-archived-field-recomparison-v2`,
publication SHA256
`03c768e493fd6c7dba85bc1f9889b4c1d9b6c7bc3a5b8ee91f1f806d91a7215f`.
An actual new Rust comparison reproduces 55 files / 171,500,889 stored words
with zero differences for the MPI1 24-hour repeat, and 27 files / 89,029,098
words with zero differences for repeated MPI1 12-hour resumes. Rank 1 versus
24 differs in 78,747,166 words across 280 fields, first in
`updraft_helicity_max`. Literal file bytes differ in all three comparisons.
Original optional smoke/dust-ON initializer input scope remains separate from
the canonical aerosol-OFF inputs.

Run its `recompare-native-reference-identity.sh <hex checkout>` on the archive host to
stage copies, execute the two fixed-rank comparisons on the build host, and retire
private copies on both hosts. The script was actually rerun successfully
against committed code. The original v1 runner's physical checkout path failed
after host transport; that witness is preserved, and v2 uses the same logical
lane path on both hosts. No extra forecast was executed. This rechecks archived
native field words; it is not a native-tap module replay, Hex parity, forecast
regeneration or visual acceptance. Door runs against native must explicitly
select `--physics-mode exact`; this native archive tool launches no Hex door.


### RRFS foundation reference delivery

The integrator's explicit 2026-10-14 11:35Z directive accepts r0's published
reference tooling and receipts for the RRFS foundation batch. The canonical
one-hour reference was consumed by the completed exact-mode fork-core Hex run
with all 13 physics slots on native CPU. This acceptance does not assert whole
forecast equality: the consumer compared 40,525,062 words and found 11,760,244
differences. Rank spread and restart-continuity differences remain recorded.
The initializer derivative and GFL corrected referee retain their independent
the owner's 07:45Z labels; pristine builds remain available.

The delivery includes an authenticated copy of the consumer's eight fine-scale
Rust sheets, personally inspected by r0. This is one-hour, coarse-mesh foundation
evidence, with visible local differences, rather than a new scheme forecast.
The shared split-stream sheet entry and step-zero terrain/reconstructed-wind
differences are open owner follow-ups. The lander runs G0/G1/G2 separately.
Use the explicitly scoped authority, case, native-contract and archived field
comparison scripts in the delivery; none is renamed to imply a CUDA module
replay or regenerated forecast. New parity door commands must select
`--physics-mode exact`.

The 2026-10-14 07:45Z ruling record (`DREW-RULING.json` in each derivative publication) is kept in
the reference store, not in this tree; `ruled_reference.py` reads it from `RRFS_SOURCE_AUTHORITY_RULING`
and still binds its exact bytes by SHA-256. Recorded labels in published receipts are unchanged.

### N-rank references (RRFS route on several ranks)

Native RRFS-MPAS is not bitwise across rank counts (1 vs 2 ranks: 52.7M of 129.4M words differ over 6 h on
x1.2562 hrrrv5), so an N-rank hex claim compares hex on N same-host ranks with native on N MPI ranks and states N.

1. Partition: `python -m woof.hex.drivers.samehost_layouts_v841 --grid x1.2562.grid.nc --ranks N --output-dir parts/ranksN`
   (METIS k-way, seed 841, K=3 rings). The forecast door builds the same files itself when `--samehost-layouts` and
   `--samehost-part` are absent (cached under `~/.cache/gpuwm-hex/samehost-layouts/<grid sha256>/`).
2. Native: `run_forecast.py ... --case CASE --ranks N`, where CASE is the canonical case JSON plus
   `"x1.2562.graph.info.part.N": parts/ranksN/x1.2562.graph.info.part.N`.
3. hex: `woof hex forecast ... --physics-mode exact --numerics strict --samehost-devices 0,1[,2,3]`
   (`0,0` runs two rank threads on one card for testing).
4. Compare, native first: `rank_compare.py NATIVE_RUN HEX_RUN receipt.json`.

Measured 2026-10-28 (x1.2562, 2023-03-31 12Z, strict, rank threads sharing one RTX 5090): 1 h at N=2 and N=4, and
6 h at N=2 and N=4 (survey probe), differ from native at the same N only in `tsnav`, a RUC local native leaves
uninitialised (it differs on 1 rank too). An init whose nwfa is zero on one rank's cells checks the TEMPO aerosol
init's global extrema: 37 of 45,947,194 words differ (all `tsnav`) at N=2 over 2 steps.
