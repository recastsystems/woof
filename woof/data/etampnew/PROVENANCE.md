# ETAMPNEW table provenance

`ETAMPNEW_DATA.expanded_rain` is WRF v4.6.1's own `run/ETAMPNEW_DATA.expanded_rain`
(tag v4.6.1, commit `d66e442fccc04111067e29274c9f9eaccc3cef28`), byte-identical:

| | bytes | sha-256 |
| --- | ---: | --- |
| WRF v4.6.1 `run/` | 43,400 | `28e60e6c83eb55889c4ab65eaa33e9fc3b557e5ab45c3dde7ee7153c2ef95a13` |
| packaged here | 43,400 | `28e60e6c83eb55889c4ab65eaa33e9fc3b557e5ab45c3dde7ee7153c2ef95a13` |

It is a Fortran unformatted sequential file written big-endian with 4-byte
record markers.  `ETANEWinit` (phys/module_mp_etanew.F:2266-2281) reads its
first twelve records (VENTR1, VENTR2, ACCRR, MASSR, VRAIN, RRATE, VENTI1,
VENTI2, ACCRI, MASSI, VSNOWI, VEL_RF); a thirteenth record of 35 words is
never read.  `tools/etampnew_seed.rs` validates both markers of every record.
`gpuwm/core/etanew_tables.py` validates the source and seeded hashes.

The same file is what WRF's Ferrier-Aligo scheme (mp_physics = 5,
`fer_hires_init`) opens. The first twelve seeded records use the same
word order and lengths as the mp=5 port. Its saturation and rate setup
uses a different scalar layout and is not substituted into this scheme.

WRF is NCAR/UCAR public domain and is attributed in `NOTICE`
(`LICENSE-WRF.txt` beside this file).

The .f32le device table is seeded by tools/etampnew_seed.rs. It decodes record markers and moves every word to little endian without floating point operations. Python selects only hash-validated seeds.

Both table files live in gpuwm/data/ferrier/tables, the mp=5 port's canonical directory. The mp=95 loader accepts GPUWM_FERRIER_TABLE_ROOT as well as GPUWM_ETAMPNEW_TABLE_ROOT (the latter takes precedence). Both schemes therefore select the same source bytes and twelve-record lookup prefix.

Both public rain variants are supplied through the shared ferrier/tables path.
The same Rust decoder preserves each variant's record words independently.
The original first six rain records have 401 words (MDR 50:450);
the expanded variant has 951 words (MDR 50:1000). Both have 951-word
ice records, whose low bits are not assumed equal between source files.
load_tables(variant="original_rain") selects the original lookup arrays.
Stock ETANEWinit always selects expanded_rain. No padding or interpolation
substitutes the shorter rain table for that WRF selection.
Source and seed hashes for both files are in MANIFEST.sha256.
