# WRF v4.6.1 surface-forcing probes

`transcribe_surface_forcing.py` is an independent FP32 statement
transcription of the coupling-only portions of:

- `phys/module_sf_ruclsm.F:611-652`
- `phys/module_surface_driver.F:3461-3473,3530-3572`
- `phys/noahmp/drivers/wrf/module_sf_noahmpdrv.F:776-789`

The pinned authority is WRF tag `v4.6.1`, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`; its Noah-MP submodule is
`848f54ad3d28c4303151fe5ad83724e232694422`.

The probe imports no WOOF-ArWen modules. Tests use it as a source oracle over
multiple rate combinations and fractional ice values.
