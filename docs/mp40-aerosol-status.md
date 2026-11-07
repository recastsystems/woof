# MP40 and aerosol coupling status

Audit date: 2026-11-01. Candidate base:
`4123f1916e3c95984862cd9b21f2fc1e8dd5f850`.

The MP40 implementation is qualified on this candidate for `aercu_opt=0`
within the real-run scope below. Positive `aercu_opt` now uses a public NASA replacement, **written from papers,
not bitwise to WRF**; see [the preparation and mapping](public-cumulus-aerosol.md).
The original external data licence remains unresolved as documented below.

## Native source and licence

The repository's native compatibility pin is WRF v4.6.1,
`d66e442fccc04111067e29274c9f9eaccc3cef28`
(`woof/wrf461_compatibility.py:23`). The requested forward comparison is
WRF v4.7.1. The two upstream copies of
`phys/module_mp_morr_two_moment_aero.F` are byte-identical, with SHA-256
`d512aa16ad862b04c7278540a207a56aa10955e052bd9b9bc870e9a4f0812180`.
WRF distributes this code under its
[public-domain notice](https://github.com/wrf-model/WRF/blob/v4.7.1/LICENSE.txt).
No competitor implementation was used.

## External aerosol table

The [native namelist documentation](https://github.com/wrf-model/WRF/blob/v4.7.1/run/README.namelist#L889-L899)
requires `CESM_RCP4.5_Aerosol_Data.dat` for both `aercu_opt=1` and `2`.
The [native admission check](https://github.com/wrf-model/WRF/blob/v4.7.1/share/module_check_a_mundo.F#L1407-L1435)
requires both `cu_physics=11` and `mp_physics=40` on every active domain.
It also sets `alevsiz_cu=30`, `no_src_types_cu=10` and `scalar_pblmix=1`.
The [native reader](https://github.com/wrf-model/WRF/blob/v4.7.1/phys/module_physics_init.F#L5521-L5551)
stops if the table is missing or cannot be opened. A zero-filled aerosol
provider would not implement this contract.

The complete [official archive](https://www2.mmm.ucar.edu/wrf/src/wrf_files/CESM_RCP4.5_Aerosol_Data.tar.gz)
was streamed for an inventory without saving the data. Its compressed
size is 1,455,870,920 bytes and its SHA-256 is
`1dc033ca2aece62c90649e3771226b8fe4350b02023ada8a4c5222cc428449ab`.
It contains exactly two regular files:

| File | Bytes |
| --- | ---: |
| `CESM_RCP4.5_Current_Aerosol_Data.dat` | 875,890,612 |
| `CESM_RCP4.5_Future_Aerosol_Data.dat` | 875,890,612 |

There is no licence, notice or README member. The
[CESM data management plan](https://files.cesm.ucar.edu/management/docs/cesm-data-management-plan-2021.pdf)
describes general CESM data terms as CC BY-SA 4.0 on page 8. It does not
identify this archive or either of its files. The
[CESM software copyright page](https://www.cesm.ucar.edu/models/cesm2/copyright)
does not establish a permissive licence for these data files either.
WRF's code notice alone does not resolve the external data provenance.

The exact table licence remains unverified. Under the permissive-only
shipping requirement, these original data files remain excluded. The fresh
NASA climatology supplies the shared interface without opening these files.

## Resumed implementation

The prior MP40 ref `e6542667dc4a93b0ac1f9dd220ba49a7348472eb` is integrated
forward, preserving the candidate's newer scheme entries and proof policies.
CUDA carries the column processes, sedimentation, finalization and reflectivity.
Forecasts use fast numerics by default; strict arithmetic is an explicit proof
option. No aerosol table is read for `aercu_opt=0`.

The older constructed-column oracle remains a component check. The additional
`instrument_native.py` and `replay_native.py` tools capture and replay actual
columns from a strict native WRF run. The completed real-run and lifecycle results are recorded below and in the
lane report.

Positive coupling uses the public NASA provider. Its interpolation status flows
through the resident asynchronous health ledger before publication.

## Completed real-run qualification

Strict native inputs captured during a real six-hour WRF run replayed with
**516,186 words and zero differences**: 726 columns, 44 levels, six sampled
times, `morr_rimed_ice=1`, `aercu_opt=0`. The independent one-bit negative
control detected one difference. No native-undefined residue was needed.
These samples have native radar flag zero; this real-input replay does not
claim to qualify the reflectivity branch. The existing constructed-column
oracle checks that branch separately. Graupel is nearly absent in this case.

The ordinary stock-namelist `woof go` forecast and native WRF both completed
2021-06-27 12–18 UTC over Florida, 199×199×44, 3 km, dt=18 s. Rust maps were
inspected beside native and MRMS at 13, 15 and 18 UTC. The model patterns are
close; individual cells differ, and both models have coverage and placement
errors against MRMS. This is implementation evidence, not a forecast-skill
claim. Fast numerics remained the forecast default.

A real 15 UTC checkpoint continuation matched all 201 serialized arrays,
110,794,444 elements, with zero differing words at 18 UTC. Same-scheme live
nests passed both feedback modes and checkpoint continuation, with 1,373,272
words and zero differences per mode. Archived-parent MP40 continuation passed
with nonzero QNC preserved into both child time levels. Two ordinary CUDA
ensemble members completed and wrote byte-identical corresponding histories.
The packed ensemble adapter and mixed-scheme passive-QNC closure remain
unqualified; the latter is refused explicitly.

On the RTX 5090, the median fast microphysics stage was 2.8455 ms versus
6.4421 ms for MP10 at 44×199×199 over 30 event-timed repetitions (ratio
0.4417). Candidate-default MP9 and MP10 full-physics state comparisons were
byte-identical over 12 steps. See the compact
[qualification receipt](../tools/morrison_aero_wrf461_oracle/qualification.json)
and the port report kept with the maintainers.

The registry retains `experimental-runtime` maturity and `scientific_evidence=none`:
this run proves the stated implementation scope and does not establish
independent conservation or forecast-skill qualification for the full scheme.
