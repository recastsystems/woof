# TC vitals reader

`rw_tc_vitals` reads the NCEP fixed-column TCVITALS format and writes `tc-vitals/1` JSON. Build with `cargo build --release --locked --manifest-path tools/tc_vitals/Cargo.toml`. Parsing runs in Rayon; `RAYON_NUM_THREADS` controls concurrency.

```
rw_tc_vitals --input syndat_tcvitals --storm AL092022 --cycle 2022092600 --output vitals.json
```

Filters require an exact report time. Conflicting revisions fail by default. `--revision first` or `last` explicitly chooses file order, which does **not** establish operational availability. No nearest-time, best-track or future-record fallback exists. Identical records are coalesced. Malformed records fail the request unless `--allow-rejected` is explicit; that option records every rejection. Pre-1999 two-digit years require `--century 19` or another explicit century.

Each record contains source line and SHA-256, reporting organization, storm identifiers, UTC report time, centre, motion, minimum and environmental pressure, outer closed-isobar radius, maximum wind, RMW, R34/50/64 and source flags. Units are Pa, m, m/s and degrees. Quadrants are NE, SE, SW, NW. Negative missing sentinels become JSON null; reported zero is preserved. Consumers must distinguish these. Missing observations must not be assimilated as zero. There are no invented uncertainties; the DA configuration supplies observation error and representativeness error. This parser's structural QC does not replace DA innovation QC.

The source envelope contains the complete input SHA-256, byte count, revision policy, duplicate/rejection receipts and parallel parsing wall time. Archive report times alone cannot prove information was available at initialization. Historical retrospective comparisons must report this limitation or establish an issue-time provenance separately.

Format: https://www.emc.ncep.noaa.gov/emc/pages/infrastructure/obs-data/tcvitals_description.php . Independently implemented, Apache-2.0. No external source code copied. NOAA/NWS data are public domain unless individually marked otherwise: https://www.weather.gov/disclaimer/ . Public per-storm archive: `https://ftp.nhc.noaa.gov/atcf/archive/YYYY/tcvitals/ID-tcvitals-arch.dat`. Coverage must be checked for each cycle; a successful download does not imply full coverage.
