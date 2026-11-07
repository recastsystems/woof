# RRTMG research controls: 2026-11-01 receipt

All four controls are usable through the existing experiment parameter table or
`WOOF_PHYSICS_PARAMS`. The obsolete `held-license` metadata is removed. AER's
2016 BSD-3-Clause relicence applies; existing full BSD, WRF and preserved historical
notices remain in `licenses/`. No numerical production code or default changes.
These are original coefficient experiments with no stock WRF namelist keys.
This is not a papers rewrite and does not establish bitwise native equivalence
for the tuned full forecast or upgrade the radiation scheme maturity label.

```toml
[physics_params.knobs]
"physics.rrtmg.radius_liquid" = 0.01
"physics.rrtmg.radius_ice" = 0.01
"physics.rrtmg.radius_snow" = 0.01
"physics.rrtmg.overlap_break_fraction" = 0.25
```

Radius offsets multiply the selected microphysics radius by `1 + offset` after
unit conversion and before the existing floors; the ice optical bound remains
140 micrometres. Each radius offset admits −0.2 through 0.2, with zero default.
Overlap admits 0 through 1, default zero: a deterministic column/g-point selector
mixes the maximum-random and random native overlap endpoints. It does not add a
new native WRF option. Requires legacy option-4 LW/SW, `use_mp_re=1` for radii,
and the existing declared cloud/overlap conditions in the control table.

Each radius control at +1% matches the WRF-gated CPU preparation wrapper with
independently rescaled physical inputs on 16 cloudy real forecast columns per
side: 507,936 compared words per control, zero differing words. This is wrapper
replay, not a newly instrumented native Fortran run with a matching control.
Nine additional synthetic radius checks cover −20%, +1%, +20% and the snow
fallback. Their largest relative difference is 9.45e-8 from multiplication and
unit-conversion rounding; the declared comparison limit is 2e-6.

Overlap fractions 0, 0.25 and 1 each compare 1,999,296 words with zero differences
against independently selected WRF-gated NumPy endpoints, in staged g-point,
chunked fused column and whole fused column layouts. Tests are executable in
`tests/test_rrtmg_tuning_controls.py`; all focused control and McICA tests finished
with 255 passed and 26 skipped. Skips are recorded, not counted as device passes.

The stock real tutorial namelist runs through `woof go --wrfinput` with option
4/4, MP8, YSU, Noah2, 73×60×34 mass cells, 30 km, dt180 s, from 2000-01-24 12 to
15 UTC. Native WRF 4.6.1, untuned GPU and all-four-tuned GPU complete all 60 steps.
The four hourly all-zero files are byte-identical to the pre-change branch.
The native reference uses the original WRF option, without the research controls.

All 18 Rust maps (temperature and shortwave at 13, 14 and 15 UTC for native,
untuned and tuned runs) were inspected. The continental cold pool, warm Atlantic,
coastal gradient and southeastern cloud shading remain aligned. Tuning changes
shortwave locally in broken cloud, as expected from changing particle optical
sizes and overlap selection. At 15 UTC, tuned minus untuned T2 RMSE/max is
0.00916/0.177 K; SWDOWN RMSE/max is 11.563/160.309 W m−2. Tuned minus native T2
RMSE is 0.04497 K and SWDOWN RMSE is 12.590 W m−2. These are measured differences,
not proof that each whole-forecast difference has one isolated cause.

ASOS temperature comparison uses the same nearest report within 30 minutes per
station and nearest grid point within 25 km for each model; reports outside
−80..125 F are excluded from numerical scores. At 15 UTC, 267 stations give
RMSE 1.85084 K native, 1.85449 K untuned and 1.85441 K tuned. Unscreened observation
dots remain visible in the maps, including one implausibly hot report. There is
no observed radiation comparison and no calibrated skill claim.

Both nest feedback modes advance one parent and four child steps on the real
1974-04-03 cold-input case (250×200×49 parent, 80×80×49 child, ratio4). The 13 UTC
GPU checkpoint resumes through 15 UTC with byte-identical 14 and 15 UTC files.
The original native 13 UTC `wrfrst` imports and advances two steps. A two-member
ordinary ensemble advances 360 seconds with distinct final canonical state
hashes; its retained diagnostic frame is at initialization only. These are
wiring checks, not a full nested ensemble forecast.

Three alternating trials of 31 warmed CUDA-event samples on 512×34 captured real
columns time complete device preparation and full McICA materialization. Median
of trial medians: untuned LW 0.51843 ms, SW 0.61072 ms; tuned LW 0.49709 ms,
SW 0.60026 ms. Combined ratio 0.97183. Common optical transfer is excluded and
unchanged. The six complete three-hour wall samples range from 2.6 to 19.9 s
under shared load; their median tuned/untuned ratio is 1.93 and is not a stable
physics timing. No claim of a full-forecast 1.2x guarantee is made.

Evidence root: `<reports>/WRF-PORTS-2026-11-01/evidence/lic-audit/`.
Every launch has a JSON command/environment receipt and a log. Relevant files:
`rrtmg-tests-final.log`, `real-radius-*-results.json`, `gpu-tuned-go.json`,
`native-stock-cached-endian-final.json`, `forecast-metrics.json`,
`off-byte-identity.json`, `restart-byte-identity.json`, `ensemble-distinct.json`,
`gpu-cold-nest-{0,1}.json`, `gpu-native-warm.json`, `rrtmg-timing-*-results.json`,
`timing-samples.json`, and `render-summary.json`. Verification scripts and exact
namelists are retained there. Earlier failed setup commands remain disclosed.
