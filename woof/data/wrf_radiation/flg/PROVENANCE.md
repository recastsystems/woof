# WRF Fu-Liou-Gu source and coefficient provenance

Licence checked at the public source before completing the column port:

- Source: https://raw.githubusercontent.com/wrf-model/WRF/v4.7.1/phys/module_ra_flg.F
- Identical native pin: https://raw.githubusercontent.com/wrf-model/WRF/v4.6.1/phys/module_ra_flg.F
- Both source SHA256: `c4977cdf2ddf3f2279c9be7e1fc659ef23bbb37376e5cf11f9a98a73d33b0266`.
- Licence source: https://raw.githubusercontent.com/wrf-model/WRF/v4.7.1/README.md, lines 5 to 22. SHA256: `077da78ac41e9d6dfb7e94de1c3a5adb129aaf89a29470f26be92d9cd33929f6`.
- The WRF public-domain statement covers this distributed source. The FLG file has no additional restrictive licence. The retained notice is `licenses/LICENSE-WRF-public-domain.txt`.

All 198 table entries in `manifest.json` are extracted from that source,
including its module DATA and routine-local DATA. No external coefficient
library is required. `tools/flg_wrf461_oracle/extract_tables.py` generates a
Fortran extractor; `package_tables.py` records shapes, offsets and hashes.
The source commit is WRF v4.6.1 `d66e442fccc04111067e29274c9f9eaccc3cef28`.

- `tables.bin`: 159,708 bytes, SHA256 `d93bdc9b5ddd33cb6ca61704fb5cddf768f0d922244e07169d28905f6a3b3c23`.
- `manifest.json`: SHA256 `06c467ffbe95732b72dc32075a984233fda8ebe3207fe5d04164937fbc7492b5`.

The source's `ice6.dps` declaration has no DATA initialization. The pinned
gfortran object exposes 24 zero BSS words; the manifest names that origin.
The stock forecast uses `nice=2`, so this compiler-scoped ice-6 detail does
not affect the real-column replay. No competitor code was used.

Strict elementary-function headers retain their existing permissive
notices, including Arm MIT and CORE-MATH MIT in `licenses/`.
