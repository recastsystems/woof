# MPAS oracle stream v1

The existing native `match_dump` pattern is extended to preserve type, shape,
Fortran lower bounds, step, call event, MPI rank and `indexToCellID` context.
All integers and words are explicitly little endian regardless of the MPAS
recipe's big endian option. No Fortran sequential record markers are present.

Header: eight ASCII bytes `ORCL0001`, then 21 signed int32 words:

1. type (1 real32, 2 real64, 3 int32, 4 logical32, 5 character bytes, 6 int64,
   7 complex32, 8 complex64)
2. element width in bytes (complex32 is 8, complex64 is 16)
3. rank (0 through 5)
4. five extents, unused extents equal one
5. five lower bounds
6. model step, per-key event, MPI rank
7. number of global IDs, cell axis, level axis (axes one based, zero unmapped)
8. fixed string width (256), reserved zero

Three space-padded 256-byte strings follow: key, phase, dummy/field name.
Then int32 global IDs, followed by raw Fortran array-order words. For scalar
characters, extent one is the character length and width is one byte.
For character arrays, width is the character element length; extents and
lower bounds describe the array, and a differing word means a differing string.
IDs zero mean a dummy/halo element without established global identity;
receipts never label them as a physical column.
Complex payloads preserve raw real and imaginary component bits in Fortran
storage order. Comparison counts each IEEE component as a word and computes
ULP distance per component; the first difference names its real/imaginary
component and the containing element's cell ID, level and indices. Column
extraction retains both components. No complex arithmetic decodes the payload.

Filename: `<key>.s<step:010>.e<event:010>.r<rank:06>.<phase>.<name>.bin`.
The directory must already exist. `status=new` prevents silent overwriting.
`MPAS_ORACLE_KEYS` and `MPAS_ORACLE_STEPS` are comma-separated exact keys and
integer steps; an unset or empty value or `*` selects all. `MPAS_ORACLE_DIR`
unset/empty disables dumping. Call counters still advance while disabled so
selection does not rename later captures. Callers must set IDs, step and rank
using `oracle_set_context` before entering a scheme; unknown axes stay zero.

A complete replay includes every supplied IN/INOUT argument before the call,
every OUT/INOUT argument afterward, absent optional arguments, hidden state,
and source/compiler pins. The inventory separately records argument and
hidden-state completeness. A comparison receipt is never forecast proof.

Pool boundary records include member kind, descriptor association, field type,
rank, time levels, and active/present flags before any values are read. Inactive
fields and absent arrays therefore remain visible. Supplied optional arrays
imported from native modules retain their allocation/association guards;
omitted optional dummies are listed separately in the call inventory.
New boundary records retain ordered member names and native field/dimension
names and sizes. Driver contexts include block configs and dimensions even when
the native call does not pass dimensions explicitly. Global IDs and native
solve-range dimensions distinguish owned cells from halo storage.

`state__physics_init` selects all initialization tables. Names carry the
`state.<module>.` prefix so equal variable names in different modules stay
distinct. A selected step zero is required. Initialization records are separate
from pre/post scheme captures. The whole native `physics_init` execution
(MPAS mpas_atmphys_init.F:83-481) also opens a selected step-zero scope before
its first executable statement. Supported local SAVE/DATA/COMMON and allocation
hooks from initialized modules record first PRE and final POST values, including
nested initialization calls. Every native return closes that scope after its
local POST. Module tables still emit their existing `init` snapshots. The
source inventory lists these initialization hooks; unobserved executions and
unsupported storage remain incomplete.

New captures also record entry storage for supplied OUT arguments. Native
schemes can leave portions unwritten, including halo levels and option-dependent
diagnostics; replay initializes those words from entry storage before calling
the original routine. Older inventories without `entry_storage` retain their
shape-only OUT allocation behavior. Output comparisons never seed from outputs.

Local SAVE/DATA/COMMON hooks retain the original storage. Within each active selected
scope they record first-entry values and buffer final-exit values for each
routine; replay restores once on that routine's first entry. Driver scopes
remain active around nested scheme scopes, including unselected schemes.
Each scope keeps its own routine counts and allocation ordinals. A replay
restores only PRE inputs and routes computed records only to its own scope;
nested hooks still update the enclosing hidden-state capture. Module state is
captured at initialization and around each scheme call.

Radiation scopes also observe O3DATA's four original local DATA tables
(module_ra_rrtmg_lw.F:12334-12444) through a generated alternate Fortran ENTRY.
The ENTRY shares the native declarations and SAVE storage and receives the
native integer bounds needed for automatic-array allocation. It restores PRE
and records actual POST storage without executing O3DATA's numerical body.
Both radiation callers prime that storage only within an active oracle scope;
ordinary O3DATA calls still update the final POST values. This supplies unchanged
hidden state when a nighttime column skips O3DATA, without copying native POST
values or inventing a duplicate table. Generation refuses changed SAVE or bound
declarations. Inventories identify these accessors, and the source audit treats
their records as storage evidence, never numerical execution or branch coverage.
Other unobserved local state remains incomplete.

GNU 15 rejects coverage compilation of either tail or pre-body alternate ENTRY
layouts with `coverage-invalid-line-number` in its generated forwarding function.
Coverage recipes retain the warning with `-Wno-error=coverage-invalid-line-number`;
strict arithmetic flags are unchanged. Native line/branch counters and replay
outputs are checked independently, and accessor function metadata cannot prove
execution of O3DATA's numerical body. Coverage remains partial.

MPAS/ESMF interval
records contain primitive clock fields and calendar association/value fields;
unset reference time has no descendant records. Calendar pointers within a
captured module carry association and alias classes: zero means absent, equal
positive classes mean shared storage, and equal-valued independent targets keep
distinct classes. Restoration preserves those relationships within the module;
Represented aliases to ESMF default/Gregorian/no-leap calendars use negative classes -1/-2/-3. Restore the ESMF global target tables before the consuming physics module.
The native MPAS/ESMF clock graph uses typed target ordinals for internal shared
targets and cycles. Restoration reuses compatible native scalar targets and
arrays with matching lower bounds and extents, preserving existing references
such as domain%clock. A target already assigned to another captured ordinal
cannot be reused for a distinct node; local calendars cannot overwrite global
provider targets. Replaced arrays and other external aliases still require
separate evidence. Capture and restore do not serialize raw addresses.
New driver-boundary captures include private primitive module state from the
callee's inventoried transitive module graph, alongside the pre/post pools.
Each boundary inventories its source callee, captured modules, unavailable
external modules and unsupported declarations. Boundary scopes now retain supported local SAVE/DATA/COMMON and allocation
entry inputs from their nested calls. Unsupported local descriptors, external
modules and storage aliases remain explicitly incomplete. Core/dynamics boundary modules
are captured at their own entries/exits; physics_init imports only physics/framework/external
modules so it cannot create a dependency cycle through its native callers.

The replay door normally invokes `<executable> <input-prefix> <output-prefix>`
once per capture. `--whole-capture` instead invokes
`<executable> <dump-directory> <fresh-result-directory>` once. It requires a
single-site dump containing exactly the selected captures. Each call's output
set, schema and words are independently compared afterward. The executable
creates the result directory and emits the original capture basenames. With
`--backend hex-cuda`, that one invocation holds the GPU queue for the entire job.
Sealed `scheme_aliases` supply public names without changing native call keys.
Sealed `scheme_groups` map a public name to several native schemes. Groups and
aliases can compose; cycles are refused before execution. `sfclay_mm5` selects
both `sfclay` and `sfclayrev` through this table. Existing immutable captures
need a new owned metadata view to add a group; their numeric records stay intact.
Directory-mode comparison indexes filenames once; it still checks every
capture's exact output set and delegates payload comparison to Rust.
Standalone native replay invokes one process and rejects any other
`--process-count`. When sealing, this option records the capture's fixed count.
Older manifests without that
metadata remain explicitly unknown in replay receipts.

`publish_dump.py` verifies a sealed capture before moving its owned directory
under `<campaign>/<pin-set>/<suite>/<dump-root-hash>`. An exclusive local lock
serializes publication and accounts for every campaign payload against a
100,000,000,000-byte cap. Existing publications are immutable; retirement is an
explicit owner action after retaining receipts. Spread and coverage receipts
belong next to the current shared capture.
`publish_dump.py --archive-index INDEX --consolidated-archive RECEIPT
--capture-receipt CAPTURE --inventory INVENTORY` also publishes a completed
compressed capture collection. It authenticates the compressed archive, the
completed producer/index provenance, and every invocation's immutable manifest
and independent member-restoration receipt. Payload and metadata are hard links;
active readers keep their original paths. The cap includes compressed archives,
metadata and publication receipts, counting shared inodes once. A failed replay
is preserved as a failed replay; archive authentication grants no forecast proof.
When the campaign directory is on another disk of its host, `--archive-store` names
an owned store on the payload's disk. The immutable campaign entry then links
to that sealed store; cap accounting follows sealed catalogue links and still
counts each inode once. Archives never move to a build host.
The catalogue lives at `<campaign>/<pin-set>/<suite>/<archive-sha256>` and contains
`PUBLICATION.json`, `complete.tar.zst`, `capture-index.json`, `capture.json`,
`inventory.json`, `consolidated.archive.json` and every invocation's manifest and
archive receipt. Source names inside those receipts remain unchanged provenance.
`replay/run_capture_matrix.py --from-archives --archive-publication PUBLICATION`
uses the published directory as `--archive-root`, authenticates all published
members on their canonical host, and reads from that directory while validating
original producer paths. Pass the catalogue's index, consolidated receipt,
capture receipt and inventory through the existing corresponding options. GNU
tar/zstd transports opaque records; Rust alone decodes and compares field words.
GNU/Intel pairwise spreads are debugging evidence only, never a proof bar
(since 2026-10-12). Forecast acceptance fixes the process count and
requires the bitwise or witnessed maths-library tier of the current policy.
Native compiler spread receipts retain all six recipe pairs per array. They use
compact JSON; if the complete receipt would exceed 9 MB, a v2 index authenticates
numbered `spread.arrays-*.json` shards by SHA256 and byte length. No array is
omitted to meet the delivery cap. These receipts are debugging evidence only.
Every new compiler comparison requires recorded, equal positive replay counts;
its array shards carry those counts and the debugging-only scope themselves.
# Native interface evidence matrix

`replay/audit_matrix.py --plan PLAN.json --out RECEIPT.json` checks every
interface in a generated replay build manifest. Each site supplies strict,
GNU O3, Intel O3, Intel valuesafe and coverage replay receipts, plus its
same-input spread receipt. It authenticates every spread member and shard,
requires every native output array and all six compiler pairs, and requires
zero strict/coverage differences. Legacy receipts without process metadata
need a hash-bound completed batch witness that explicitly names the same
receipt and one replay process. Original receipts remain unchanged.

`replay/collect_coverage.py --matrix-receipt RECEIPT.json` can consume that
legacy process witness while binding actual gcov counters, notes and native
source mappings. Coverage is cumulative; partial branch coverage, missing
hidden state and whole-model acceptance remain explicitly separate from the
matrix's successful replay checks. Compiler spreads remain debugging evidence.

Synthetic profiles in `replay/profiles/` supply typed native input assignments.
`generate_fixture.py` restores the original PRE storage, applies those
assignments in Fortran, and captures the unchanged native scheme's results.
Optional `pre_assertions` and `post_assertions` are logical expressions over
mandatory arguments. Preconditions run after perturbation; postconditions run
immediately after the native call, before the replay writes its argument,
pool or module POST records. Instrumented callee hooks may already have written
local-state POST records during that call; a producer must require successful
fixture exit before sealing any expected outputs. Array sections can limit a scenario
assertion to the native active levels while dumps and comparisons retain halo
storage. Assertions preserve module argument aliases and reject undefined OUT
inputs, optional argument reads and statement injection.

For coverage collected on another host, `--compiled-build-root` names the
original directory embedded in gcov notes. `--build-root` holds the copied
instrumented sources, notes and isolated counters. Both paths are retained in
receipts; mapping still authenticates the pristine and instrumented source
hashes and rejects sources outside the declared compiler tree. Set a fresh
`GCOV_PREFIX` when separate scenario counters are required.

`replay/project_input_reads.py` accepts one sealed selected-column capture and
successful `strace -f -s 8192 -e trace=openat` executions of all five native
recipes. The trace index binds the input root, capture, executable, trace and
independent replay receipt hashes. Strict and coverage witnesses must be
bitwise. The tool retains every argument, expected POST and module/local state
record, plus every allocation PRE record opened by any native recipe. It can
remove only allocation-entry PRE records that none of those executions opened.
Numeric extraction and comparison remain in the Rust helper. Execute all five
recipes again on the projected capture and compare every array with the
original results before publishing it. This proves input closure for those
executions; hidden-state, branch and forecast completeness remain separate.

# Source declaration audit and replay resources

`replay/audit_state.py` reparses the pinned native module sources with the current
declaration parser and compares both declaration roots and emitted leaves with
the sealed capture inventory. It then checks PRE/POST record membership,
including allocation and pointer presence through the Rust stream reader.
Source hashes, inventory hashes and audit/parser hashes are recorded. This
detects older inventories that silently omitted declarations, such as the
complex SAVE fields `m_w_0` and `m_i_0` in `physics_mmm/mp_radar.F90:43`.
A passing audit certifies only the known physics module declarations and their
captured membership. Local SAVE, framework globals, external aliases and
conditional derived state still require independent evidence; the audit never
promotes `hidden_state_complete`.

Native replay children use the host's permitted hard stack limit, and receipts
record the parent limits and effective child limits. The parent process remains
unchanged. This supports compiler-generated temporary storage while restoring
large tables; the native source, compiler flags, executable and captured input
words are unchanged. A resource correction does not certify numerical identity.

The source audit also reparses local SAVE/DATA/COMMON declarations and compares
them with the generated hook inventory. For each observed routine it requires
all first-entry PRE and final-exit POST records, rejecting unknown local records.
A routine with neither phase stays explicitly unobserved: absence never proves
that the routine did not execute. Local execution coverage and global hidden
state remain separate fields from matching the source declaration inventory.


Automatic entry storage requests in `entry-storage.v1.json` name native source
arrays whose allocator-return contents are read before all words are written.
The generator validates each declaration and inserts `oracle_allocation` after
compiler allocation, before the routine's first executable statement. PRE names
retain source declaration lines and scope ordinals. Replay restores the observed
entry words before native arithmetic; expected POST storage is never an input.
The current requests cover CAM `tauxcl` and `tauxci` at
`physics_wrf/module_ra_cam.F:364-365`, copied to diagnostics at lines 678-679.
Changing these declarations to SAVE, pointer, allocatable or dummy storage is a
source error requiring a new capture design. The eight-call synthetic CAM
qualification of 7d15d28 counts all 22,145,892 output/state words and reproduces
them exactly under strict GNU and coverage. That capture is diagnostic evidence,
not a replacement for older captures or a native forecast identity witness.

Framework modules use private external transport interfaces, avoiding a physics
USE dependency during framework compilation. The generator captures supported
mutable `mpas_timekeeping` and ESMF globals, including Calendar/BaseTime table
leaves with native precision and lower bounds. `framework_state_check.py` builds
against the actual model archive and restores initialization records twice,
checking every emitted field through the Rust comparator. This check certifies
the represented globals and aliases; unobserved local SAVE calls and unsupported
clock descendants remain explicit completeness gaps.


The source audit also reparses the automatic entry-storage request table and
compares source declarations, line tags, dimensions and PRE-only scope ordinals
with the sealed inventory. A direct callee must have its entry records; a POST
record cannot satisfy that requirement. Nonexecuted nested hooks stay explicitly
unobserved. This check authenticates the requested automatic inputs, not every
possible automatic allocation. The fresh CAM eight-call capture has matching
module/local/request inventories and all requested PRE records. Its complete
source audit still reports local-state source obligations and unobserved local
routines; successful automatic-input membership does not override those gaps.

## Fork replay extension

The shared generator accepts `--pin-set rrfs`. The pin table authenticates the
model and physics submodule commits, selects NOAA MPAS wrappers, and excludes
other model interfaces and MUSICA. `prepare_build.py --sources-only` prepares
owned instrumented modules for standalone work without asserting that a strict
native reference build exists. Full preparation still requires its reference.

RRFS wrappers may call other tapped wrappers. Capture scopes nest; each parent
retains the first hidden routine input and final hidden output across child
calls. Standalone replay suppresses child call-site taps while preserving the
actual native calls and hidden-state hooks. Replay restores each hidden input
once per scope. Input and output bytes retain the existing transport format.

Fresh allocated arrays and native automatic temporary arrays have additional
entry records named `allocation.<native-source-tag>.n<invocation:010>`. Hooks
capture bytes immediately after allocation or at routine entry, before native
initialization or use. Replay performs native allocation and restores these
inputs for each invocation. Every enabled ancestor keeps its own invocation
sequence, so a parent replay retains successive child allocations even when
child call-site taps are suppressed. Entry records are never reconstructed
from dumped outputs. Unsupported allocation options, types and pure routines
remain explicit source-located inventory issues.

The RRFS entry inventory also includes primitive scalar and array OUT dummies
within these native modules. A native branch can leave an OUT dummy unwritten;
replay restores its observed caller-entry bytes before that branch executes.
Optional dummies are captured only when present, including their bounds.
Declaration-only callback interfaces are untouched, while empty executable
routines capture at their native END statement. True assumed-size OUT arrays
remain unresolved without a caller extent; finite dimension expressions such
as `n*1` retain their actual native bounds. The inventory distinguishes
`dummy-out` and `automatic-local` storage and retains every unsupported entry.

Changed driver boundaries also capture primitive state in their callee module
closure, including private state. Configuration pools restore native MPAS field
descriptors, ranks, time levels and iterator position. Derived aliases, external
callback context and unresolved state remain completeness issues; successful
array replay alone does not establish full boundary or forecast identity.

`column_oracle_replay.py --ref rrfs` requires a sealed RRFS source family. The
default remains `mpas`, and cross-family replay or sealing is rejected.

Synthetic columns use `replay/prepare_synthetic.py` with explicit native
Fortran expressions in a profile JSON. The fixture uses the callee's declared
argument names and bounds; caller-local pointer expressions are not available
in that standalone program. Profiles and generated sources are hashed in its
synthetic manifest. Captures and replay still use native Fortran and the Rust
word comparator. These are debugging inputs, not forecast proof.

`replay/run_synthetic.py` executes every prepared profile under every built
recipe. It retains per-array comparisons and executable/source hashes, records
the host and fixed one-process scope, and retires owned run files after each
receipt. Required native namelists or tables are staged with `--input-file`;
their hashes are recorded and each fixture gets a fresh working directory.
Native input or replay failure makes the matrix command exit nonzero.

`replay/prepare_framework.py` prepares every shared generated argument replay
against an owned complete model build, including SMOKE and driver-local
routines that need actual framework pools. Supply the model's recorded external
link arguments as a JSON array. Build via `make -j4` (through `HEX_BUILD_QUEUE` on a shared
host). The source hash
check also applies to source-only framework builds; their manifest explicitly
records that no published strict reference authenticated their forecast.

Archived replay accepts `--replay-recipe gnu-strict|gnu-o3|coverage`. Alternate
recipes require `--replay-inventory`, `--replay-artifacts` and `--replay-audit`.
The runner authenticates every audited native object, generated replay source,
build manifest and executable before reading captures. Capture and replay
inventories must agree in every field except `instrumentation_sha256`, whose
owned Makefile flags differ by recipe; generated native source hashes, argument
interfaces, state inventories and fork pins must still agree. Capture provenance
keeps its original strict recipe. Replay receipts and the matrix's `replay_build`
record the actual executable recipe and its authenticated metadata hashes.

GNU coverage from native package wrapper copies is mapped through the generated
source inventory (`coverage.py --inventory`). The compiled copy, inventoried
wrapper and recorded hash must agree, and exactly one pristine source path must
resolve. Receipts preserve both compiled and native paths. A changed or ambiguous
copy is a coverage authentication failure, rather than an inferred source match.

Changed-boundary replay can call `oracle_pool_restore_in_place(prefix,pool)`
after the actual native framework has initialized its context. It restores
primitive values and field attributes through the existing descriptors, without
recreating target storage. Cross-pool array aliases, descriptor pointers and
ancillary I/O pointers therefore retain their native identities. Captured member
order, association, type, rank, time-level count, extents and lower bounds must
match that context; an incompatible replay context is an error. This API does
not construct callbacks, infer missing state or prove the context is complete.
Each top-level restoration refreshes capture context independently of previous
captures. Fresh reconstruction remains available through `oracle_pool_restore`.
Native mesh contexts borrow the persistent pointer returned by
`mpas_pool_get_array` through `oracle_set_native_context`; that storage must live
through the driver scope. The standalone `oracle_set_context` API owns a copy.
Switching to an owned replay context never frees borrowed native mesh storage.

`replay/prepare_passivity_variant.py` prepares owned strict diagnostic builds
from authenticated frozen objects. Whole-module body restoration retains
generated headers and serializers; `--whole-native-source` instead restores
complete pristine source files. Framework, ESMF and core replacements remain in
their actual native libraries. `--parent-variant` authenticates and reuses a
previous sealed build. `--seal` binds sources, dependencies, recipes and all
compiled products. Compile its Makefile with `make -j4` (through `HEX_BUILD_QUEUE` on a
shared host).
`replay/run_passivity_variant.py` runs complete pristine/variant deck pairs,
compares every written field with Rust and retires its owned runtime only after
authentication. These are host/count-specific diagnostics, never a Hex gate,
an exclusion or permission to use non-passive captures as pristine authority.
Native bounds diagnostics must retain their changed compiler flags, whole-module
source and executable hashes, original input manifest, process count, actual host
and complete native trace. Map an instrumented line back to the exact pinned
statement before naming the defect. An observed bounds error proves that read was
invalid; it does not prove that the same affected words caused a replay mismatch
or a later forecast failure. Keep those comparisons and all their words counted.
For example, all three original later MYNN calls reach `DOWNA(K-1,I)` with `K=1`
in pinned `physics_mmm/bl_mynn_subroutines.F90:6235`. Their bounds-checked diagnostic
exits do not justify excluding the 1,709 differing words of strict call 15.
Any proposed exclusion still requires the exact repeat/host/process-count and
affected-word witnesses and the owner's native-defect ruling (proof-bar ruling 1).

Framework SAVE inventory includes every source module under src/framework,
including the private native I/O reservation ledger (mpas_io_units.F:29) and
pool error level (mpas_pool_routines.F:108). Supported primitive globals use
the existing external transport bridge, preserving framework-before-physics
compilation. Unsupported timer and stream topology roots remain explicit
inventory issues. Capturing supported globals does not establish global hidden
state completeness or authorize excluding any original failed words.

Framework allocator hooks use the same typed external bridge as framework SAVE
records. The full native `mpas_string_utils.F:34-69` module compiles before the
physics transport module; its `subStrings` allocation at line 56 captures PRE
contents before native substring assignment. Replay restores those contents and
then executes the original routine. Missing PRE remains an explicit replay
failure. Refreshing an owned failed build restores every candidate framework
source before regenerating hooks, so a partial overlay cannot accumulate them.

The whole native logger target at `mpas_log.F:54` uses the source-derived
`mpas_log_types.inc:11-43` layout: all counters, task/core/domain metadata,
output/error file fields, presence and represented pointer alias classes.
Restoration preserves an existing domain logger root, reuses compatible file
targets, repairs split/shared aliases and initializes newly allocated child
pointers before reading them. Actual full native module fixtures exercise
logging APIs and two mutation/restore passes in fresh processes, including
absent roots and missing PRE refusal. External file-target aliases and live
file-descriptor positions remain explicit completeness gaps; captured unit
numbers and filenames do not establish those properties.

Public scheme names use the sealed alias definitions first, then the current
`scheme_aliases.json` lookup catalog. Existing sealed captures retain every native
call-site key, file and root hash. A newer public name, such as `ntiedtke` for
`cu_ntiedtke_driver`, therefore selects old products without rewriting them.
Replay and batch receipts bind the current catalog SHA256 and its mappings, the
sealed alias-table hash, the requested name and the resolved native schemes.
This is lookup provenance; it never changes scientific inputs or outputs.

The source-state audit reparses the whole known native module closure and its local
SAVE/DATA/COMMON hooks. Broad declaration-time capture observations are classified
as represented storage only when that routine's complete source-derived hook
inventory matches and has no unsupported local declarations. The receipt retains
every original observation separately. A missing hook, changed declaration, missing
SAVE name or unsupported descriptor stays unresolved. This classification does not
prove execution: observed PRE/POST pairs and numerical execution coverage remain
separate, and a static getter never satisfies numerical execution coverage.

Native self-check profiles may contain an explicit `cases` matrix. Its cases share
one source-built, strictly audited oracle executable and one stated process count;
each case independently runs plain and dump-on forecasts, compares every written
field through the Rust verifier and checks framework initialization restoration.
No case can override the host, recipe or process count. A failed case fails the
matrix. Each completed case authenticates and retires its disposable outputs before
the next starts. Returned metadata exceeding the receipt limit is preserved as
ordered, independently compressed byte chunks with original and per-chunk hashes;
no records are dropped to fit the limit. This is the instrumentation neutrality
self-check, not a replacement for real-door forecast or hidden-state completeness.
Transport copies authenticated namelist, streams and stream-list controls only.
Case, native build and table manifests independently bind runtime data. Putting
data or output hashes under a deck path refuses explicitly; this avoids flattening
large table symlinks into a transfer that the runner cannot carry.
Coverage matrices use `replay/merge_coverage.py --collection <coverage/index.json>`
for each completed capture, with the pristine `--source-root` and authenticated
instrumented `--build-root`. The tool checks every collection, replay, raw gcov
and source mapping before combining counters. Source, generated source, gcno,
compiler and process count bind the observation; distinct builds stay separate.
Branch identity includes function, native line, occurrence and block IDs.
Generated instrumentation is excluded. Remaining native branches and unmapped
native lines are retained, and a coverage union never asserts forecast proof.


### Source-verified allocation controls

The coverage matrix first authenticates each collection against its original
sealed mapping and raw gcov hashes. It then maps an expanded native single-line
`IF (.NOT.ALLOCATED(a)) ALLOCATE(a(...))` back to its pinned source line only
when the tagged original line, array name, condition and allocation bounds all
match. Both the native condition and the allocation's compiler runtime checks
retain their actual counter/block identities. The observation hook and generated
`ENDIF` remain excluded. Duplicate or changed expansions are not mapped.
Historical coverage collections remain unchanged; the combined matrix records
its recovered mappings and still reports every unobserved native branch.

Full-framework recipe preparation emits a separate owned native tree for each
GNU/Intel profile. Each completed build binds the full object inventory, final
model binary, original compiler commands, flags and generated launcher hashes.
`replay/prepare_native.py --replay-root` can compile every generated interface
against those actual MPAS framework archives; numerical framework stubs are not
substitutes. Library paths follow the installed prefix layout. Native AUTOCLEAN
is used when recorded build options differ. Build receipts establish compilation
only; replay comparison, native counters and forecast identity require their own
executions, always at an explicit process count. Compiler differences remain
debugging evidence and never supply a tolerance or acceptance bar.

The source-state membership audit interprets captured logical controls through
the Rust word reader. Known source-bound ASSOCIATED/ALLOCATED conditions and
scalar logical components govern descendant membership in native nested IF
order. A false outer control authorizes absent descendants; a missing, cyclic
or unsupported control remains unverified. Derived pointer roots require both
presence and alias records. Record membership does not certify external alias
semantics, cyclic typed clock topology, or external file-descriptor positions.
The complete native mpas_log.F target-state regression checks independent,
shared and absent roots through capture and two restores, and refuses missing
root controls, missing aliases and missing PRE.


The original March x1.2562 convection-permitting real-door forecast from clean
4a0287a completes 72 steps at one process. Official verify-mpas compares
545,733,285 written field words with zero differences; 3,066 missing or
incompatible samples remain recorded. All 24 Rust difference sheets at
6/12/24 hours were opened and reviewed with the fixed fine difference scales;
field patterns, contours and wind barbs agree and no coloured difference pattern
is visible. Receipts real-door-convection-v72-final/ and
real-door-convection-v72-visual-review.json bind the actual snapshot, original
native publication, renderer, scale table and every PNG. This is the joint
native-CPU baseline; no CUDA slot or global hidden-state completeness is inferred.


The source-state audit now walks the represented complete MPAS/ESMF clock graph
in the generated native reader's declaration order. Six separate typed pointer
registries preserve internal aliases and cycles; new nodes require consecutive
ordinals. Calendar provider references, native negative array bounds and inactive
TimeInterval descendants retain their original meaning. Missing leaves, orphan
nodes, invalid provider classes and forward references prevent membership
certification. Controls and record layouts are decoded only by the Rust reader.
The native fourteen-type capture/two-restore regression and complete logger
regression pass together at one process in native-whole-clock-membership-v82/.
The original scalar-character rank check failure remains recorded separately.
This certifies represented membership and internal topology; external aliases,
file-descriptor positions and whole-model hidden-state closure remain unproven.

Complete-build gcov collection retains owned generated transport and replay
sources separately from pristine native MPAS sources. Their source bytes,
compiler counters and notes remain authenticated, and their receipts state
`native_branches_counted=0`. A source outside the owned compilation root, or a
missing generated source, still refuses collection. Old sealed mappings are
recomputed under their original options before native allocation-control
mapping is added. A generated-only matrix cannot pass by an empty `all()`;
matrices with unmapped sources retain those witnesses and stay incomplete.

The six compiled collector checks and the complete two-suite production
collection are sealed in native-complete-coverage-collection-tests-v89/ and
native-complete-build-coverage-v93/. At one process, the latter authenticates
108 mesoscale and 109 convection objects from 72 actual native replay calls.
Its 107 native source groups observe 8,538 of 48,392 branches; 34 generated
source reports add no native branches. Coverage remains incomplete. Separate
per-entry counter runs in native-scheme-isolated-coverage-v87/ reproduce all
1,432,092,983 observed output words exactly. These counters cover the linked
native universe for each entry, including routines it did not execute. The
original summary's 78 build members are 39 executables plus 39 gcov notes;
the sealed correction records 39 compiled scheme entries and 19 observed ones.
No compiler spread, unobserved entry, external state or forecast is certified
by these coverage collections.

Four-recipe pairwise debugging can reuse authenticated original comparisons
with `spread.py --reuse-authenticated-comparisons`. Identical computed output
hashes prove exact equality; an output matching the common native reference
allows reuse of the other output's measured comparison. Two distinct changed
outputs require their authenticated record bytes and a new Rust comparison.
Their separate distances from native never determine their pairwise distance.
Every pair retains its evidence type and both computed output hashes. The
option does not change the proof bar or turn compiler spread into acceptance.

The complete one-process first-step debug matrix covers 72 native calls,
47,835 arrays and 287,010 compiler pairs. The 824 pairs with distinct changed
outputs were compared by Rust after 138 complete native call recoveries
reproduced every original computed record hash. No expected POST storage was
seeded. Canonical receipts/native-all-entry-pairwise-debug-v107/ preserves all
per-array pairs, source/executable bindings and output-retirement witnesses.
The original four-recipe numeric matrix and current three compiled/refusal
checks are sealed separately in native-all-entry-debug-recipes-v102/ and
native-pairwise-spread-tests-v106/. Coverage and global hidden state remain
incomplete; these compiler differences are debugging evidence only.

Native timer capture follows `mpas_timer.F:45`, the complete three types in
`mpas_timer_types.inc:1–19` and `mpas_dmpar_types.inc:9–31`, and six typed
pointer registries. It preserves cycles, shared targets, each array view's
lower bounds, allocation presence and primitive words. The framework uses
external bridges to avoid a dependency cycle with the physics modules.
MPI handles are opaque native storage: the generated code supports the
pinned MPI F08 declarations and integer declarations through `TRANSFER`,
and checks their native storage size. It does not recreate MPI resources.

The complete native timer fixture in native-whole-timer-membership-v125/
captures an existing graph and restores it twice, then repeats from a null
root. Six represented graph datasets and 4,496 words match exactly at one
process. Missing leaves, orphan nodes, invalid forward references and
inconsistent alias extents refuse. The original MPI F08 compile failure is
preserved in native-whole-timer-membership-v123-failure/. The second host returned
exit 75; these checks used the authorized proof-host fallback and actual native
framework libraries. External aliases, MPI resource recreation and future
wall-clock readings remain unproven. This fixture is not a forecast proof.

Stream connectivity state follows all eight saved roots in
`mpas_stream_manager.F:96–103,4894–5397` and every native component in
`mpas_field_types.inc:246–276`. One traversal preserves aliases across roots,
linked-node cycles, raw primitive storage, pointer presence and each array
view's bounds. Generated framework bridges avoid a module dependency cycle.
Capture hooks surround `MPAS_writeStream` after native prewrite indexing and
before native postwrite restoration; the original call order is unchanged.
Select `MPAS_ORACLE_KEYS=stream_reindex` to capture populated saved roots.
Native fields which the prewrite routine never initializes remain observed
storage; no default values are invented. Non-null external pointees are named
in membership audits and refuse standalone restoration, while capture itself
continues. External aliases and native uninitialized-metadata stability remain
unproven.

The actual native fixture `native-whole-stream-membership-v131/` compares
4,368 words over six graph datasets at process1, with zero differences. It
restores both live and absent roots twice, preserves cross-root aliases and
bounds, and checks missing leaves, orphan nodes, invalid references and
unrepresented external pointers. Four selected checks pass, with no skip.
The original schema failure is retained separately. The native final link now
groups libdycore, libops and libframework, resolving generated graph bridges
referenced from framework objects after the physics archive's first scan.
Object arithmetic flags remain unchanged. Full integration compilation and
real-input replay receipts are separate from this fixture.

`replay/real_input_check.py` authenticates a permanent native input archive,
extracts stated actual calls with tar/zstd, generates their native standalone
replays from the original sealed interfaces, and compares every output word
through Rust. A subset manifest retains its original parent root and archive
hash. The profile binds the original native build and every retained native artifact.
For older captures, replay uses their original generated state schema: an old
capture does not acquire newly generated timer or stream records. The checker
requires the candidate transport and standalone replay ABI files to match that
artifact, and records the other generator differences explicitly. Current
framework graph checks and the full strict build are separate evidence.
Input records and computed records are hashed before retirement. The checker
runs one process, accepts only exact words and never starts a forecast.

The delivery entry points are `receipts/replay.sh HEX_CHECKOUT` and its
`receipts/reproduce.sh` wrapper. They regenerate three stated real native
cloudiness/LW/SW calls from the permanent authenticated archive, compare every
output word at one process, and execute seven current native framework checks.
The profile is adjacent to the entry point, so the sealed delivery can run on
a lander checkout. Neither entry point submits a whole forecast. This is the
2026-10-14 03:00Z service landing scope; it does not claim all branches or
external-state closure. Compiler spread remains debugging only.
