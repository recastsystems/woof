# RW-WPS community support matrix

This page summarizes the machine-readable
`woof/native_wrf_support_v1.json`. The JSON file is authoritative. “Writer”
means the state inventory and NetCDF exporter exist; “stock-WRF gated” means
unchanged WRF v4.6.1 opened the exact hash-bound files and advanced.

## Sources

| Source/path | Implemented envelope | Strongest retained stock-WRF evidence |
|---|---|---|
| HRRR CONUS named profile | Hourly f00..f12; Lambert; explicit eta; static one-way d01..d21 parser/export envelope | d01..d06 |
| GFS 0.25° named profile | Uniform 1/3-hour pressure-level series; Lambert hierarchy implemented through d06 | Single domain; named hierarchy beyond that is not live-gated |
| GFS mapped profile | Current exact GRIB2 mapping/composition; declared `max_dom=4` | d01..d04 |
| ERA5 named GRIB1 profile | Uniform combined series; Lambert hierarchy implemented through d06 | Single 250×200×49 domain |
| Generic mapped GRIB1 | Descriptor/Vtable or sealed mapping | Exact historical ERA5 single-domain contract only |
| Generic mapped GRIB2 | Descriptor/Vtable or sealed mapping | Exact current GFS d01..d04 contract only |
| Generic mapped NetCDF | Explicit descriptor/mapping | Exact historical ERA5 single-domain contract only |
| 20CRv3 member GRIB2 | Named `rw-wps --source 20crv3` exact-member adapter with wheel-packaged authorities and shared native hierarchy preparation; mapping ceiling `max_dom=4` | Exact private-sample decode/preparation evidence only; live native/stock-WRF gate pending |
| Other registry sources | Inventory/refusal messages only unless composed through `mapped` | None |

A readable file is not necessarily a complete WRF state. New mappings must
declare atmosphere, surface, soil, time/accumulation, wind rotation, missing
data, hydrometeor, and target policies. They do not inherit another mapping's
evidence.

## Geometry and execution

| Capability | Status |
|---|---|
| Lambert conformal | Implemented, both hemispheres; matched-run verified in the northern hemisphere only: southern hemisphere is oracle-verified + smoke-run verified, not matched-run verified |
| Mercator; polar stereographic (both poles) | Implemented; oracle-verified + smoke-run verified (binary64 gates against the pinned WRF v4.6.1 `module_llxy.F` oracle plus GPU smoke integrations), not matched-run verified |
| Antimeridian-crossing domains | Implemented (Mercator/polar/Lambert); same oracle + smoke maturity |
| Pole-containing or pole-touching domains | Refused; lat-lon source interpolation and static-tile windowing are not pole-capable |
| Latitude/longitude (cylindrical), rotated grids | Unsupported; fail closed |
| Static one-way nests | Implemented; strongest live gate d06 HRRR |
| Delayed child starts | Implemented in namelist import, native hierarchy artifacts, WOOF-ArWen tree execution, wrfout provenance, and restart/resume. Each start must be an exact parent-step boundary and an external-forcing seam; synthetic end-to-end gated, not a new stock-WRF certification claim |
| Boundary forcing cadence | Generic mapped hierarchies accept positive uniform whole-second cadence, including 5 minutes, when the cadence is an exact integer number of d01 steps. Named-source routes retain their genuine source-product cadence limits |
| Moving nests, two-way feedback, vertical refinement | Unsupported; fail closed |
| Domain count | Common parser/export 1..21; explicit regression at 6; route/mapping/live-evidence ceilings still apply |
| Horizontal dimensions/placement | Dynamic, with WPS/WRF equality, ratio divisibility, parent containment, and positive-spacing checks |
| Native static geography | Complete global WPS 5-minute category handling implemented; optional provenance-bound high-resolution raster prototype implemented but not improvement/stock-WRF certified |
| Vertical coordinate | Shared explicit eta, `hybrid_opt=2`; structural 35/49/80-level gates; source-top coverage required |
| CUDA preprocessing | Implemented and numerically compared |
| Rust/NumPy CPU preprocessing and setup/export | Implemented for ERA5/GFS/20CRv3/mapped; clean-machine real-data matrix incomplete |
| `auto` backend | CUDA 12.x or 13.x device when available, otherwise CPU; also CPU at measured GPU utilization >=50%, when the preparation's priced device memory exceeds the card's free memory, or when CUDA cannot open the card, with measurements, the price and the reason in the receipt |
| HRRR common CPU/CUDA selector | Not yet wired in the public named driver |
| Sealed Linux archive | x86-64 CPU/CUDA assembly and clean CPU install gated |
| Sealed Windows archive | x86-64 CPU-only builder/PowerShell installer implemented; HRRR shell lane and Windows real-source/stock-WRF gates pending |

## Physics and products

| Capability | Status |
|---|---|
| WSM6 + YSU + classic MM5 option 91 + four-layer Noah | Stock-WRF gated slice: retained receipts report that unchanged WRF v4.6.1 opened the exact files and advanced 5 to 60 model seconds. This is an interoperability gate, not an accuracy certification; [receipt scope and binary caveats](wrf-interop-evidence.md). |
| Thompson (`mp_physics=8`) writer | Inventory/direct writer implemented; exact hash-bound `f9760c9` current-GFS d01..d04 files advanced unchanged WRF v4.6.1 by 10 seconds; broader envelopes are not certified |
| Morrison (`mp_physics=10`) writer | Inventory/direct writer implemented; exact hash-bound `f9760c9` current-GFS d01..d04 files advanced unchanged WRF v4.6.1 by 10 seconds; broader envelopes are not certified |
| Other PBL/surface/LSM/microphysics state | Unsupported unless explicitly inventoried |
| `wrfinput_d01..dNN` | Atomic hierarchy writer implemented within route ceiling |
| `wrfbdy_d01` | Implemented; root only |
| Auxiliary lower-boundary files | Inventory not complete |
| Deterministic receipts/SHA-256 | Implemented |

## Release blockers

- the public named-HRRR CPU selector and broader named ERA5/GFS hierarchy
  proof coverage;
- live 20CRv3 and Windows live-source/stock-WRF gates;
- clean-machine live-source/stock-WRF coverage for the dedicated `rw-wps`
  distribution beyond the retained exact gates;
- redistributable clean-room fixtures and second-machine CPU/CUDA CI;
- current-hash stock-WRF gates outside the listed envelopes;
- complete auxiliary/physics state inventory and generic-mapping
  certification;
- dependency/SBOM/signing and security review; and
- an owner-approved repository license.
