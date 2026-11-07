# Physics parameter sets

Registered constants can be selected for a forecast without changing its
prepared inputs. With no set, kernel source, parameter tables and public
configuration retain their default bytes. No parameter identity or output
attribute is added. No tuned set is a new default.

A set file contains:

```toml
[physics_params]
name = "parameter-member"
values = { "mynn.prandtl" = 0.8, "ruc.z0.short" = 0.6 }
```

An experiment may carry the same table, or reference a relative set file
with `[physics_params] set = "member.toml"`. Unknown constants, invalid
values, conflicting set sources and constants whose schemes run on no
domain refuse before the forecast. Ranges and exact CUDA source sites live
in `woof/physics_params_registry_v1.json`. CASE(1) mixing-length parameters
do not change CASE(2). Roughness rows exclude seasonal crops whose roughness
receives a subtractive seasonal adjustment.

For members sharing a preparation, keep the experiment file and prepared
root unchanged. Launch each forecast in its own process with
`WOOF_PHYSICS_PARAMS` naming the absolute set-file path. Set this variable
only for the member forecast, after preparation. An absent variable selects
the ordinary default member. Parsing resolves the file through
`woof.physics_params.environment_set()`; programmatic validation uses
`parse_table()` or `make_set()`. Values are canonical float32 values, and
`document()` supplies the name, values, registry digest and set digest.

Direct Python `run_experiment()` and model construction bind the validated
set before preparation. Execution of an already-built tree retains its
original set and refuses a first selection or a switch after construction.

Future ensemble recipe support needs an optional member set-file reference.
The recipe resolves the path relative to its own file, validates the set,
checks its schemes against the member experiment, and records `document()`
in the member receipt. It passes the absolute path through the child
process environment. This member option does not enter the preparation
cache key: all registered edits are forecast-time constants. The launch
must sanitize an inherited `WOOF_PHYSICS_PARAMS` for default members and
must not compile kernels in a shared parent process before setting a
member's environment. Members use separate output and checkpoint roots.

The set digest binds the experiment identity used by checkpoints. Outputs
carry `WOOF_PHYSICS_PARAMS` and `GPUWM_PHYSICS_PARAMS_SHA256`; edited RUC
bundles retain pinned input-table hashes and record each changed cell.
For the regional family, fixed scalar names and float32 dtypes determine CUDA
source and compilation; values bind once to per-module device constant memory.
Changing values while retaining those names reuses the executable cache.
The setup copy precedes every forecast stream; no per-step copy, lookup or
synchronization is added. An inactive family preserves the original CUDA bytes.
RUC table arithmetic runs on the GPU after pinned-table validation. One
process cannot switch to a different set after compiling kernels or
issuing tables; a new member requires a new process.


The regional fractional interface is also available from a stock WRF case:

```sh
WOOF_PHYSICS_PARAMS=/absolute/path/coefficients.toml woof go \
  --wrfinput /absolute/path/wrf-case --outdir /absolute/path/forecast
```

For example, `coefficients.toml` may contain:

```toml
[physics_params.knobs]
"physics.mynn.b1" = 0.01
"physics.thompson.auto_amplitude" = 0.01
```

The native namelist still selects the schemes and required pairings. MYNN
EDMF and GF shallow convection cannot both be enabled, as in native WRF.
The seven `mynn.length2` controls select the `gsd_41` generation with
mixing length 2; they are refused on `wrf_461`, whose equations differ.
Rain and graupel fall-speed controls remain held until their native
initialization and collision tables can be regenerated together.

One process uses the same immutable coefficients for every participating
domain and ensemble member. Distinct member wind perturbations work with
an imported native case through `--ensemble-request`; an unperturbed
request that would produce duplicate members still refuses. Independent
per-member coefficient sets use separate forecast processes. Per-domain
and per-cell coefficient fields are not implemented by this interface.

The 2026-11-01 lane evidence includes a three-hour native-input forecast,
both nest feedback modes, checkpoint continuation, two distinct ensemble
members, and an all-zero run whose four hourly history files are
byte-identical to the unselected baseline. Component replay coverage and
its limits are recorded in `docs/public/PHYSICS.md`; this does not qualify
all 33 controls or establish forecast skill.
