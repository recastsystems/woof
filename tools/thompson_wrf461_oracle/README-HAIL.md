# Thompson graupel-hail, mp_physics=38

The CUDA hail-aware implementation was recovered from
`902b8f7f8298ad8127deb6abb8f0f3ca989696fc` and brought forward to the
current carrier, IO and physics interfaces. It remains
`implemented-unverified`: the inspected forecast is not an observation-based
skill qualification.

The sources are public WRF `phys/module_mp_thompson.F`,
`phys/module_mp_radar.F` and `run/CCN_ACTIVATE.BIN`, distributed under
WRF's [public-domain statement](https://github.com/wrf-model/WRF/blob/v4.7.1/LICENSE.txt).
Their licence was checked before recovery. No competitor implementation
was used. Retain WRF's licence and attribution with its assets.

The 240,308-byte Thompson module is byte-identical in v4.6.1 and v4.7.1,
SHA-256 `fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802`.
Changing that version pin cannot explain the recovered strict differences.
`woof/core/thompson_contract.py` binds the coefficients, CCN table and
four generated table assets. The rain-graupel table has nine density slabs.
Generated large tables remain outside source control.

## Runtime and stock namelists

The production adapter compiles the aerosol CUDA kernels with
`THOMPSON_AA_HAIL=1`; the option-off kernel source is retained.
Fast numerics are the default. Strict arithmetic is the proof opt-in
`GPUWM_WRF_EXACT=1`. No host numerical column calculation, device readback
or synchronization was added to the step path. Redundant Exner and zero-fill
launches were removed from MP38. Table hashes are checked once per startup
policy, rather than on every step.

Prognostic `ng` and `qvolg` accompany graupel mass through state allocation,
RK/acoustic transport, boundaries, history, restart/checkpoint and ordinary
ensemble members. Both same-scheme nest feedback modes carry the moments.
Mixed-scheme nest edges and offline-child conversion remain refused:
there is no density-consistent number/volume conversion for those paths.
The old fork generation without prognostic volume is refused by name.

Stock import preserves selector 38 and the global native controls
`force_read_thompson`, `write_thompson_tables` and
`write_thompson_mp38table`. A validated complete cache is reused read-only.
Force-read refuses a missing asset; missing hail tables require the explicit
MP38 generation permission. Disabling persistence builds missing assets in
private temporary storage without changing the selected cache. Invalid
existing assets are never overwritten. Default controls do not change
option-off config/restart identities. These controls are admitted for MP38;
explicit MP8/28 use remains refused because their distinct cache loader does
not implement them.

The native MP38 caller leaves all three effective-radius flags false
(`module_physics_init.F:985-1024`). MP38 therefore preserves inactive radii
and skips the radius routine's tiny-cloud cleanup. Native real initialization
does not seed MP38's absent hydrometeor number moments; input mass is
preserved, with initial radar history retaining the native zero field until
the first diagnostic. These fixes leave MP28 unchanged.

## Native proof and recovered fixtures

The lane's strict native WRF v4.7.1 build uses
`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`.
`capture_native_hail.py` adds output-only writes. Its
`--require-graupel` mode waits for a diagnostic-due column with pre-existing
graupel mass, number and volume. A manifest binds every captured CSV to the
public source, instrumented executable, namelist and run command.
No native arithmetic expression is replaced.

The original two wet 49-level columns, captured at timestep 42 with GNU
Fortran 13.3/glibc 2.39, now replay with **0 / 1,876 differing state words
and 0 / 1,372 literal tendency words**. All 294 literal Exner, temperature
and thickness entry words are exact. These entries have no graupel.
A second strict native v4.7.1 run (GNU Fortran 15.2/glibc 2.43) captured
two more 49-level columns at timesteps 42 and 80, with pre-existing graupel
mass, number and volume. That replay is also exact: **0 / 1,876 state words
and 0 / 1,372 literal tendency words**, with all 294 module-entry words exact.
The columns contain 17/28 nonzero graupel-mass levels and 12/13 volume
levels; peak graupel mass is 3.337e-8/1.109e-8 kg/kg. This is active graupel
coverage, not a captured large-hail storm. Across both captures the total is
0 / 3,752 state and 0 / 2,744 tendency words.

The replay consumes literal pressure, theta, vertical velocity, species,
surface state and native module-entry Exner/temperature/thickness. It also
reports the independent differences that reconstructing thickness through
forecast geopotential would introduce; it does not perturb native inputs
to erase those differences.

```sh
GPUWM_WRF_EXACT=1 python tools/thompson_wrf461_oracle/validate_hail.py \
  --backend gpu --fixtures /path/to/native-capture \
  --capture-manifest /path/to/native-capture/manifest.json \
  --out strict-native-replay.json
```

The five shipped synthetic fixtures are separate regression diagnostics.
The recovered "43 strict differences" meant 19 state field/case comparisons
plus 24 tendency comparisons, containing 33 and 55 unequal words.
The arithmetic corrections reduce that to 16 and 18 comparisons, containing
30 / 2,315 unequal state words and 38 / 1,680 tendency words. Maxima are
4 and 136 ULP; warm-melt is exact. No native undefined behavior is claimed.
The fast synthetic comparison has 33 and 76 unequal words; its largest hail
vapor tendency relative difference is 1.631724e-5 (absolute 6.17e-12).
The previous 1e-5 diagnostic tolerance was not widened.

```sh
GPUWM_WRF_EXACT=1 python tools/thompson_wrf461_oracle/validate_hail.py \
  --backend gpu --fixtures woof/data/thompson/oracle-hail --out strict-hail.json
```

The synthetic command exits 1 for the declared residue. It is not the
real-native proof and must not be described as bitwise conformance.

## Whole runs and transport checks

The final `woof go` run used MP38/YSU/classic-MM5/Noah/KF/legacy-RRTMG,
12 km, 60 by 48 by 49 cells, 2024-05-06 18–21 UTC, fast numerics.
It completed all 147 adaptive steps, four hourly history frames and hourly
checkpoints. Graupel mass, number and volume develop during the forecast.
Two distinct surface-state ensemble members also completed the full three
hours through the ordinary production prepared runner and Rust probability
products. The unqualified packed-member composition uses the existing
ordinary-member fallback.

Rust maps of reflectivity over all four hours and temperature were inspected
beside native WRF and MRMS. Both model forecasts produce broader, weaker
precipitation too far east of the narrower observed storm line. Temperature
patterns are broadly similar; no temperature observations were fetched.
Native starts from the common Rust-exported prepared state, not fresh
WPS/real.exe input. External hydrometeor and vertical-velocity boundary
tendencies are absent from that exporter, so whole-model forcing/trajectory
parity is not claimed. Automatic observation certification also remains
unavailable with the installed verifier's time-parser ABI; the manual Rust
comparison panels are the visual evidence.

`tests/test_thompson_hail_continuation_gpu.py` compares 40 full model steps
with 20 + checkpoint + fresh restore + 20, including live hail moments;
every serialized word agrees. `tests/test_thompson_hail_nests_gpu.py` drives
the real nested executor with feedback 0 and 1, two parent/six child steps,
checks live moments and checkpoint inventories, then restores both domains
and obtains identical carried words and clocks. These transport tests are
idealized inputs; they are not a second real nested forecast.

The whole option-off control runs MP28 for the same three hours against
candidate `4123f1916e3c95984862cd9b21f2fc1e8dd5f850` on identical sealed
inputs. All four history files are byte-identical, as is the final canonical
carried state. The lane report records the full commands, configs, source
identities, performance samples, figures and limitations under
`<reports>/WRF-PORTS-2026-11-01/evidence/port-mp38-thompson-gh/`.

## GPU stage timing

On the shared RTX 5090, the final fast-numerics comparison tiles the two
active-graupel native columns to 49 by 48 by 60 cells. Sixteen alternating
MP28/MP38 pairs give median CUDA physics times of **0.842432 / 0.969824 ms**,
a **1.151219 ratio**. This meets the 1.2x stage budget. The identical
production adapter kernel sequences are captured into CUDA graphs for this
measurement, with state reset outside the events, to remove host scheduling
gaps. Graph capture is a timing method here, not a new production execution
mode. All shared-card samples are retained, including interrupted samples.
These are GPU physics times, not whole-model step or CLI wall times.

Separate launcher events identify the additional hail fallout work:
graupel sedimentation is 0.053952 ms for MP28 and 0.168672 ms for MP38,
which transports mass, number and volume with density-dependent velocities.
Cold-network times are 0.256992 / 0.249792 ms; warm-network times are
0.116544 / 0.113312 ms. These separate samples can contain host gaps and
nested wrapper events, so they must not be summed into the graph total.
The lane retains the full stage trace, interleaved samples and helper as
`resume-stage-profile.json`, `resume-stage-profile.log` and
`helpers/profile_stage.py`.
