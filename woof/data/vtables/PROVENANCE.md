# gpuwm/data/vtables

## Vtable.ERA5_CDO

- sha256: `64282b5b35ac7302e274f764327923080883f164f4e605ef06529d1baef6620e`
- WPS-format Vtable describing ERA5 GRIB1 parameters (pressure-level +
  single-level, ECMWF table-128 codes) for the native GRIB1 ingest route.
  Derived from the WPS v4.6.1 `ungrib/Variable_Tables/Vtable.ERA-interim.pl`
  family, adjusted for CDS ERA5 retrievals (level-type 112 soil layers are
  additionally aliased by `gpuwm/ingest/grib.py` for native CDS files).
- Byte-identical to the reference bundle's `era5_grib/Vtable.ERA5_CDO`,
  the exact schema the certified real-data lineage decoded with; the
  canonical parameter/spelling mapping is pinned by
  `gpuwm/ingest/grib.py::_CANONICAL_SPECS`, which fails loudly on drift.
- Packaged so `woof domain` can emit relocatable configs that declare a
  vtable without referencing a machine-local bundle path.  WPS Vtables
  carry the WRF public-domain license.
