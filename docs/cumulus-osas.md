# Old GFS SAS, cu_physics = 95

Selector 95 runs the complete WRF v4.6.1 OSAS column driver on CUDA,
including deep OSASCNV, shallow SHALCV, MSTADBT3, TRIDI2T3 and the
native-word gfs_funcphys tables. The source authority is official WRF
commit `d66e442fccc04111067e29274c9f9eaccc3cef28`. Every source and derived
table has its license and hash recorded in
`tools/osas_wrf461_oracle/source-pins.json`; WRF's distributed public-domain
statement is retained alongside it. No competitor source is used. The OSAS module, all three GFS support
modules and WRF preprocessor are byte-identical at WRF v4.7.1 revision
`f52c197ed39d`; the lane source-equivalence receipt records each hash.

Fast binary64 CUDA arithmetic is the default. `GPUWM_WRF_EXACT=1` selects
the strict proof path. OSAS retains native binary64 column precision even
in fast mode. Strict support uses the existing pinned permissive arithmetic
helpers. Device-built tables contain 94,662 binary64 words with SHA-256
`46b826ff2e7519cd89149ca9c351bc8aa5819f5e1a1812dea1f306f5e6735a61`.
Their host digest read occurs once at setup. Column stepping has no host
numeric processing or per-step host synchronization.

The stock WRF namelist importer admits `cu_physics = 95` and preserves
positive `cudt`. Held tendencies and PRATEC account for rain on intervening
steps. Adaptive stepping resets the interval as WRF does. There is no
PBL pairing requirement in the native OSAS field contract. Moisture is
required, and at least three mass levels are required by the native column
indexing. The selectable composition is
`wsm6-ysu-mm5-noah-osas-rrtmg-legacy-v1`.

`woof go --wrfinput REAL_EXE_DIRECTORY --run-seconds 10800 --outdir RUN`
uses that directory's producing `namelist.input` and the shared WRF-input
forecast runner. This runner retains its existing domain-tree, supervision,
Rust input/output and product paths. The native-input directory must contain
its lateral forcing and the static assets required by the selected suite.

Two production choices are explicit. WRF reseeds its cloud-top draw from
the clock per row; the default port uses model step and the persistent
domain-column identity for restart-stable counter draws. Strict replay
accepts the captured native XKT2 input, or WRF's REGTEST value 0.1 without
an injection. When rain fully evaporates, default mode restores cloud
condensate with temperature and vapor, fixing the native backout's retained
detrainment. Strict replay reproduces native backout. Both choices belong
to checkpoint identity. A default forecast is not a native bitwise trajectory.

Persistent state includes held rates, PRATEC, RAINCV, CUTOP/CUBOT and an
optional injected draw. The draw is a domain plane. Reused GPU tiles gather
global column identities and carry the draw and diagnostics with the other
state. Cloud diagnostics map to native cumulus CUTOP/CUBOT, not radiation
HTOP/HBOT. Those four native restart variables are explicit history
extensions emitted by the Rust writer. Native restart import also handles
annual SHDAVG as the native Noah setup climatology, without treating it as
an unconsumed prognostic field.

Strict cold streaming reserves transport buffers before the first acoustic
producer, with a persisted cold marker. First-step diagnosis ignores this
reserved capacity until real acoustic mass is produced. The marker travels
with tile clocks, checkpoints and graph topology keys. It prevents a reused
tile from treating allocation as evidence that an acoustic step already ran.
Ordinary resident archives omit the optional marker.

The live native-run replay compares all eight CU_OSAS outputs over 300 real
columns and 49 mass levels: 60,000 float32 words, zero differing words,
including 58 raining columns. The captured native XKT2 is unchanged.
Analytic chained calls add 172,800 output words and 258,048 binary64 stage
words, all equal. Committed fixtures and tests are under
`tests/data/oracles/osas` and `tests/test_cu_osas_wrf461_parity.py`.
Checkpoint tests require whole-archive byte identity for uninterrupted
40 steps versus 20 steps, restore, and 20 steps, in both arithmetic modes
and with and without an injected draw. GPU tests also cover cold and warm
reused tiles, both resident nest feedback modes, and ensemble execution.

The delivered lane report records the real 300 x 300 x 49, 3 km forecast
for 2026-10-16 09 to 12 UTC, native WRF comparison and actual ASOS station
verification at 10, 11 and 12 UTC. Rust-rendered paired maps cover
2 m temperature/dewpoint, 10 m wind, composite reflectivity and accumulated
precipitation. This is one suite and one case. The independent scientific
evidence axis for the OSAS option remains `none`: the suite comparison
does not isolate OSAS skill. No radar score or general forecast-skill
improvement is claimed. The report and receipts are in
`WRF-PORTS-2026-11-01/evidence/port-cu95-osas` in the lane delivery.

A measured speed repair separates low-register driver preparation and
level-parallel tendency output from column physics. Within OSASCNV, two
sets of independent environmental passes are fused without changing any
native arithmetic expression or the ascending level dependence. The
15-entry critical-work table lives in constant memory. The existing
bounded binary64 workspace and stream ordering carry all intermediates;
no new persistent state or host synchronization is introduced.

On a shared RTX 4090, 24 alternating before/after/New-SAS stage samples
on the same 90,000-column, 49-level row-tiled real input measure 9.076736 ms
before and 8.479232 ms after, versus 7.217664 ms for New SAS 96. The ratio
falls from 1.257573 to **1.174789**, meeting the 1.2 target without a physics
cost exception. The older 1.262169 result is superseded. This workload
repeats one captured native row and fixes New-SAS-only PBLH/HFX/QFX to
1000 m/0/0, as the original timing did.

On the RTX 5090, all 90,000 distinct columns captured from the first real
GO call, with its actual surface/PBL fields, measure 4.714720 ms before,
4.205712 ms after, and 5.424224 ms for New SAS: **0.775357** after.
Both comparisons include packing, counter RNG, workspace, physics and
carrier updates. All 18,000,000 captured fast output words remain equal.
The fresh real native replay retains 60,000 words with zero differences;
the native capture/output archive is retained with the lane's speed proof.
Nsight Systems traces attribute the work to device kernels; host syncs
shown at capture end belong to the profiling harness, outside the stage.

Not measured here: a native-restart continuation forecast, a moving-nest
real case, multi-case observation skill, or performance on cards beyond the two above.
