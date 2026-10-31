# Urban infra oracle fixtures, WRF v4.7.1

## init/ -- urban_param_init + urban_var_init

Six cases, `opt<sf_urban_physics>_lcz<use_wudapt_lcz>` for options 1-3 and
both legends, written by `tools/urban_wrf471_oracle/run_init.F90` through
`oracle_io.F90` (format: `gpuwm/verify/urban_oracle.py`).  Each case is one
process: `urban_param_init` allocates its module tables once, at the first
call's category count, so two tables in one process would overrun them.

Built with

    bash tools/urban_wrf471_oracle/build.sh ~/work/urban-wrf471/WRF BUILD

on an x86-64 host, 2026-09-30:

* WRF tree: wrf-model/WRF tag v4.7.1, commit
  `f52c197ed39d12e087d02c50f412d90d418f6186`; every compiled source matched
  `tools/urban_wrf471_oracle/SOURCES.sha256` before compilation.
* Compiler: GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0, `-O0`, WRF's phys/
  defines (`-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4
  -DDWORDSIZE=8 -DLWORDSIZE=4`).  No `_ZGV*` symbol in any -O0 object; the
  libmvec positive control emitted one.
* C library: glibc 2.43 (Ubuntu GLIBC 2.43-2ubuntu2.4).  The derived table
  values call `expf` and `powf`; gpuwm evaluates them with its glibc 2.39
  transcription (`gpuwm.core.noahmp_libm`) and matches every one with zero
  ULP, so the two glibc versions agree on these arguments.

SHA-256 of what built them:

    623868c74c4b9d579e9c3811e9c334d731394c2afbfea7d693221626fbf0b0ea  phys/module_sf_urban.F
    c51ddd86871f81d5f132e63ddab2240e23886891356a7964debd4eafda9e500f  phys/module_sf_bep.F
    42fe129dde0ccba24b84a64c1393e582204a7ced938568527beeb0c4f2a20b36  phys/module_sf_bep_bem.F
    5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062  share/module_model_constants.F
    2ed7dc6e90e0fe442ffee84512b4998d31c8ec3400d3a7ab6078404065f784a6  frame/module_wrf_error.F
    5811226b3db503ae02d8b35cb5cb10e0e90804e451f0f4a5b76a274ee64773d0  run/URBPARM.TBL
    ab08e3f79d2f5d9d329aa2c953de4e7d94a71741baa0a50f1ecc145c6f81d9ab  run/URBPARM_LCZ.TBL
    738ff01251fe17808d20d471852eda93a9c262c21831ef7b34ba15f23d206d89  tools/urban_wrf471_oracle/run_init.F90
    1653e767df483cf937d11afb8f2e7ee785d5d833ae9469a40901919ed4938de7  tools/urban_wrf471_oracle/oracle_io.F90
    f31f5b26def1a17163ce20f770db749b80ef2f45102d3eefb2dfcc6280623c53  tools/urban_wrf471_oracle/stub_wrf.F90

Gates: `tests/test_urban_tables.py` (84 table names x 6 cases, 0 ULP) and
`tests/test_urban_init_wrf471_parity.py` (every Registry array
`urban_var_init` writes, 0 ULP; arrays WRF leaves at the -7 sentinel must
stay at gpuwm's allocation zero).
