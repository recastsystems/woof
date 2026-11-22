# WRF v4.6.1 MYNN/LSM ownership probes

`transcribe_pairing.py` is an independent NumPy/FP32 statement transcription
of the ownership-only portions of:

- `phys/module_surface_driver.F:2359-2400,5244-5560,7117-7208` (MYNN
  surface call, fractional-sea-ice staging, and ice/SST component
  temperatures);
- `phys/module_surface_driver.F:3127-3181,3324-3370` and
  `phys/noahmp/drivers/wrf/module_sf_noahmpdrv.F:1206-1285`
  (Noah-MP write-back and post-LSM diagnostics);
- `phys/module_surface_driver.F:3500-3528,3579-3592`,
  `phys/module_sf_ruclsm.F:219-230,284-307`, and
  `phys/module_sf_sfcdiags_ruclsm.F:7-146`
  (RUC write-back and post-LSM diagnostics).

The pinned authority is WRF tag `v4.6.1`, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`; its Noah-MP submodule is
`848f54ad3d28c4303151fe5ad83724e232694422`.

Pinned source SHA-256:

- `module_surface_driver.F`:
  `128af41014f36528bdfc2ef48e7a3452bcdbbe02fade19b6173c9fabc27277b8`
- `module_sf_ruclsm.F`:
  `c07673ca656398d056ef7696496299b51f25167c2335d4f60ce2d862afdf8925`
- `module_sf_sfcdiags_ruclsm.F`:
  `6d21be6a275f2b19c35f36e3fc6a013529f487363b5c6cb8c5e950ee7b159e27`
- `module_sf_noahmpdrv.F`:
  `2201527e9febdd8d0f6cd56e4cc3ad22a180a656a6eafc93252ba4658369114b`

The probe imports no WOOF-ArWen modules. It is intentionally limited to driver
assignments and diagnostics; it does not duplicate either land-surface
column solver.
