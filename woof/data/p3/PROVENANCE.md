# P3 lookup table provenance

`tables/p3_lookupTable_1.dat-v5.4_2momI` is WRF's own
`run/p3_lookupTable_1.dat-v5.4_2momI` **with CRLF line endings**.  P3
v4.5.2 intends table-1 version 5.4_2momI (phys/module_mp_p3.F:138), and
`phys/module_mp_p3.F` is byte-identical across WRF v4.6.1, v4.7.1 and
v4.8.0 (sha-256
`716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6`), so
this table is current for the newest WRF as well as for the frozen v4.6.1
reference bundle this copy came from.

WRF is NCAR/UCAR public domain and is attributed in `NOTICE`.

## Prognostic droplet number source audit, 2026-11-01

The selector 51 port uses WRF's public source, not a competitor port.
The source and table were checked at these upstream locations before
restoring the implementation:

- [WRF v4.7.1 public-domain statement](https://github.com/wrf-model/WRF/blob/v4.7.1/LICENSE.txt).
- [P3 source](https://github.com/wrf-model/WRF/blob/v4.7.1/phys/module_mp_p3.F).
- [Two-moment ice table](https://github.com/wrf-model/WRF/blob/v4.7.1/run/p3_lookupTable_1.dat-v5.4_2momI).

Neither source nor table has a separate restrictive notice. The checked
v4.7.1 source is byte-identical to v4.6.1 with the source hash above.
The checked v4.7.1 table has the LF hash in the table below and equals the
packaged table after CRLF normalization. Selector 51 needs no new table.
This audit adds no claim about native forecast or GPU replay validation.

## The CRLF divergence, measured 2026-08-28

This file is NOT byte-identical to WRF's, and the earlier wording here
that said it was is corrected:

| | bytes | sha-256 | line endings |
| --- | ---: | --- | --- |
| WRF v4.6.1 `run/` | 1,606,038 | `be1ab6fb03481e376e47c6c79d808af5d8ab069f2b242931e9c54801bad4ae84` | LF |
| packaged here | 1,637,040 | `5cb47f2ad15b726abeacac21b5877c737178526051661473a465315348867c2f` | CRLF |

The difference is exactly 31,002 bytes for exactly 31,002 records: one CR
per line, added when the file was copied through a Windows tool.  Both
files carry the same 167,002 tokens.

**It changes no number.**  WRF's own `p3_init` was compiled with gfortran
15.2.0 and run against each file in turn; the `itab` and `itabcoll`
module arrays it produced were byte-identical, because a Fortran
list-directed READ treats the CR as trailing blank.  Our parser produces
those same bytes from either file too (the digests are pinned as
`FORTRAN_ITAB_SHA256` / `FORTRAN_ITABCOLL_SHA256` in
`gpuwm/core/p3_tables.py`, and `tests/test_p3_table_refusals.py` holds
them).

**It does have one consequence.**  `GPUWM_P3_TABLE_ROOT` invites an
operator to point at a mirror, and the most obvious mirror on any machine
with WRF on it is WRF's own `run/` directory -- which this pin refuses.
The refusal names the CRLF fact when it recognises the upstream digest,
so the operator is not left hunting for corruption.  Repinning to the
upstream bytes instead is a RESTART-IDENTITY decision, not a loader one:
the digest is written into every mp=50 checkpoint's setup identity
(`gpuwm/io/restart.py::_p3_setup_identity`), so moving it would make
every existing mp=50 checkpoint refuse to resume.

## Two-category interaction table

`tables/p3_lookupTable_2.dat-v5.3` is WRF v4.6.1's own `run/` file,
byte-identical with LF endings. The source pin is WRF commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`.
- Size: 15,250,032 bytes.
- SHA-256: `c9103041ead719cf9e2e215cf846e0ef81b2004f29386156444a079af6b24d97`.
- Parsed `itabcolli1` followed by `itabcolli2`, in Fortran element order:
  `13a745a56fd11a0239fa0d573bf480abf5e3da44b7e435d28c0b5f59b9ef00a6`.

WRF reads this table only for `nCat>1` (`module_mp_p3.F:474-519`).
The table parser is checked against the Fortran initialization dump from
`tools/p3_wrf461_oracle/table2_dump.sh`; the packaged and parsed digests
are asserted in `tests/test_p3_2cat_port.py`. One-category initialization
still loads only table 1. Two-category checkpoints identify both assets.

The module and both tables were checked at the public WRF source on
2026-11-01. WRF's public-domain notice is retained in `LICENSE-WRF.txt`.
The upstream [notice](https://github.com/wrf-model/WRF/blob/v4.6.1/LICENSE.txt)
covers this release; neither the module header nor either table carries a
separate restrictive notice. The two-category table is the
[upstream run file](https://github.com/wrf-model/WRF/blob/v4.6.1/run/p3_lookupTable_2.dat-v5.3).
A fresh strict native initialization matched all 500,000 parsed table-2
words, with SHA-256 `13a745a56fd11a0239fa0d573bf480abf5e3da44b7e435d28c0b5f59b9ef00a6`.
This initialization check is separate from a real-input physics replay.

## Three-moment ice

`tables/p3_lookupTable_1.dat-v5.4_3momI.xz` contains the exact LF text
from WRF v4.7.1 `run/p3_lookupTable_1.dat-v5.4_3momI`. Its 17,886,038
uncompressed bytes have SHA-256
`9a3c57ecc09498802c8d7cb3931dbb0200dcf0f51466b3e288120c080271e6dc`.
The packaged 614,968-byte xz stream has SHA-256
`18ef2429958eee71c84b37f7212a8ec62eb61549043bfabe84f6dad82a836b69`.
The official source and table carry WRF's public-domain statement and no
separate restriction. The source pin above is unchanged in WRF v4.7.1.

`rw-p3` validates the version, length and text SHA-256 before parsing
independent blocks in parallel. It also reads the identical uncompressed
WRF run/ copy. The Fortran array hashes in `p3_tables.py` pin all 825,000
parsed words. Rust startup uses `rw-libm` and parallel independent rain
rows, retaining WRF's sequential sum within each row.

The decoder `lzma-rust2` 0.16.1 is Apache-2.0. Its new dependencies `crc`
3.3.0 and `crc-catalog` 2.4.0 are MIT OR Apache-2.0. Their complete licence
texts and source checksums are retained in `tools/rustwx/vendor/crates-io/`.
