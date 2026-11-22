# Trailer B. Picture captions and inspection record

All weather pictures are unaltered `woof.rustwx` → `rw_wrfbatch` output, 1400 × 1000 pixels. The gallery pairs w, Earth-relative relative vertical vorticity and column-maximum reflectivity. Heights are 500, 1,000 and 3,000 m AGL. Read the printed unit multiplier: vorticity often uses 10⁻⁶ s⁻¹, weak w uses 10⁻³ m/s. Color limits are selected separately for each map. Click a picture for its full size.

## Handoff, 2026-10-16 00:00 UTC

The first gallery section compares the 60 km hex export with initialized 3 km WOOF-ArWen at the same UTC, at 1,000 m AGL, with reflectivity beneath. The source covers a broader region. Both have nonzero resolved vertical motion. The import is not a bitwise restart: horizontal/vertical interpolation and interface reconstruction change individual values. Source reflectivity is at the −35 dBZ clear-air floor; initialized WOOF-ArWen diagnostic reflectivity is 0 dBZ. Those distinct diagnostic values must not be read as storm growth. This pair proves the reachable import path and visible fields, not survival of a mature storm through a moving seam.

Viewed: the source and initialized WOOF-ArWen w at 1,000 m, source vertical vorticity at 1,000 m, and both initialization reflectivity maps. Also viewed WOOF-ArWen vertical vorticity at 1,000 m at 01:00 UTC: weak broad gradients with finer terrain-related texture, not a compact tornadic vortex.

The source 03:00 UTC reflectivity map was also viewed. It has a broad precipitation band from Texas northeastward; it is three hours after the clear initialization. The measured two-hour fine LES box near 36°N, 91°W remains calm. Broad-source extrema and cropped-target extrema therefore have different spatial support.

## Full 111 m fallback, 00:15 UTC

Viewed all seven fields at this time: w and vertical vorticity at each named height, plus reflectivity. Weak vertical-motion waves and vorticity near the western/northern sides are startup and adjustment features. The w color limits are about −0.14 to +0.19 m/s at 500 m, −0.06 to +0.09 m/s at 1,000 m, and −0.005 to +0.045 m/s at 3,000 m. Reflectivity is the uniform −35 dBZ floor. There is no mature rotating storm in these pictures.

## Regional export

Viewed the real regional-export 00:00 UTC w at 1,000 m AGL. It shows a smooth weak field, approximately −0.002 to +0.030 m/s. The renderer's 1.112 km label is the target geographic-grid spacing; it is not the source mesh resolution. This export window samples coarse cells, with mean nearest-source distance 22.94 km and maximum 37.60 km. The mesh has a sub-km core elsewhere.

## Full 111 m fallback, 02:00 UTC

Viewed all seven final maps: w and Earth-relative vertical vorticity at 500/1000/3000 m AGL, plus reflectivity. Weak wave-like vertical motion and narrow vorticity bands enter from the western side and cross the box. These are adjustment structures without a co-located mature precipitation/updraft core; the reflectivity remains at the −35 dBZ floor. At the named heights, interior maxima are w = 0.4896/0.3182/0.2285 m/s and |vertical vorticity| = 0.001318/0.001071/0.0003660 s⁻¹. Larger edge colors can lie in the excluded rim. A vorticity maximum alone does not establish a tornado.

Viewed the full three-domain time series at all three heights, covering all 25 times through two hours. The three footprints differ (120/60/30 km). Vertical motion remains below 1 m/s at these heights; area above 5 m/s is zero throughout. Fine-domain vorticity has intermittent maxima, consistent with the narrow bands visible in the maps. This is a finite, stable calm run, not a mature storm benchmark.

## Time series

The source/3 km chart was viewed. It compares different footprints and unequal native history cadence (source three-hourly; WOOF-ArWen five-minutely). Its max w and max absolute vorticity are useful audit values, not storm-object continuity metrics. Updraft area with w ≥ 5 m/s is zero. Full-tree charts and exact raw receipts accompany the final gallery. Statistics exclude a two-cell boundary rim; extrema are horizontal extrema at the stated height, not a whole-volume maximum.

## Texas precipitation-band handoff, 03:00–05:00 UTC

Viewed the 03Z source and initialized 3 km WOOF-ArWen w maps at 1,000 m AGL. The broad positive w pattern over Texas and neighboring subsidence remain visible at initialization. The source map has a much broader footprint; the cropped target preserves the broad pattern through horizontal/vertical interpolation.

Viewed all four selected band reflectivity times: 03:00, 03:05, 04:00 and 05:00 UTC. The initial 0 dBZ diagnostic precedes physics diagnosis. At 03:05 the broad Texas band is visible above 50 dBZ; through 04–05Z it develops a more organized line extending toward Oklahoma, with roughly 50–60 dBZ cores. This is a precipitation-system comparison across changed resolution and physics, not a byte-exact restart or Xenia validation.

Viewed all six final band w/vorticity maps at 500/1000/3000 m AGL. At 3,000 m the narrow uplift band follows the precipitation line. Lower levels have discrete stronger cells and terrain-related patterns toward the southwest. Vorticity is patchy and banded; these maps do not establish a compact storm vortex. Final max w is 2.884/5.314/10.216 m/s; max |vertical vorticity| is 0.002664/0.002250/0.002994 s⁻¹. Area above 5 m/s is 0/9.261/92.600 km².

Viewed the band source/target time-series charts at 1,000 and 3,000 m AGL. The target develops sustained stronger updrafts; 3,000 m updraft area becomes substantial late in the run. The two source samples are three hours apart; the target samples are five minutes apart over two hours. Straight connectors are plotting guides, with no synthesized source measurements. Source and target maxima cover different geographic footprints. The 500 m chart is provided using the same analysis workflow.

## What remains

The producer must provide actual moving Xenia full-column exports at five-minute cadence, each with its own mesh/init, on one invariant geographic grid and terrain. A mature storm track must select the 25-frame two-hour interval and fixed 30 km box. Then repeat these maps and time series on matched footprints and inspect seam continuity. A following nest is extra integration only if that measured storm exits the fixed box.
