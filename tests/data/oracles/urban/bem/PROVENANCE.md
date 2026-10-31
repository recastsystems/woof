# BEP+BEM oracle, WRF v4.7.1 (`sf_urban_physics = 3`)

Five fixtures, each the complete dump of one run of
`tools/urban_wrf471_oracle/run_bep_bem.F90` against the byte-unmodified WRF
v4.7.1 sources pinned in `tools/urban_wrf471_oracle/SOURCES.sha256`
(copied from the design folder's `wrf-src/SOURCES.sha256`):

| source | sha256 |
| --- | --- |
| `phys/module_sf_bep_bem.F` | `42fe129dde0ccba24b84a64c1393e582204a7ced938568527beeb0c4f2a20b36` |
| `phys/module_sf_bem.F` | `7bc761ee592feadbaeb974b4271b956044eb02c5a4f3bf0e77af4fbabf0de8eb` |
| `phys/module_sf_urban.F` | `623868c74c4b9d579e9c3811e9c334d731394c2afbfea7d693221626fbf0b0ea` |
| `phys/module_bep_bem_helper.F` | `79403e10104e23fc2a44eb7cf9c0a33c10fabed3b963c9ccb0ea0bfd29072b6d` |
| `share/module_model_constants.F` | `5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062` |
| `frame/module_wrf_error.F` | `2ed7dc6e90e0fe442ffee84512b4998d31c8ec3400d3a7ab6078404065f784a6` |
| `run/URBPARM.TBL` | `5811226b3db503ae02d8b35cb5cb10e0e90804e451f0f4a5b76a274ee64773d0` |
| `run/URBPARM_LCZ.TBL` | `ab08e3f79d2f5d9d329aa2c953de4e7d94a71741baa0a50f1ecc145c6f81d9ab` |

| file | table | `use_wudapt_lcz` | raw dump sha256 (`data.bin`) |
| --- | --- | --- | --- |
| `bep_bem_stock.npz` | `URBPARM.TBL` as shipped | 0 | `e2ba27ec4ad4fa1db46ad7a5cb7636e6898572892a0cdb1bb506d0fff3cb390a` |
| `bep_bem_lcz.npz` | `URBPARM_LCZ.TBL` as shipped | 1 | `cfb23dce468e57b5a92387308469530ff6755e60902c25331bc9624c163da68a` |
| `bep_bem_gr1pv.npz` | `URBPARM.TBL` + sed below (sha256 `ae51a089...`) | 0 | `4fabeeb6e73db6c05971fd92863e019825949ff4a6e65ac5ee6dbdc8c1e916c6` |
| `bep_bem_gr2.npz` | `URBPARM.TBL` + sed below (sha256 `709ad680...`) | 0 | `f1ddff6842da353644427f19cef78f9fdc2e9eeeec7e0dce8de80aa9ac661e02` |
| `bep_bem_long.npz` | `URBPARM.TBL` as shipped, 30 calls | 0 | `6a9b3d8833db9ed1eeb379c782df7d2405db7e040aab7a12e1cb6fc52aab8f55` |

The two derived tables switch on the arms WRF ships switched off, so that a
gate exists for them:

```
gr1pv: GR_FLAG:1  GR_TYPE: 1  GR_FRAC_ROOF:0.5,0.3,0.6  PV_FRAC_ROOF: 0.3,0.2,0.4
       IRHO: hours 12-15 and 23-24 on   TIME_ON: 8., 0., 6.   TIME_OFF: 18., 24., 20.
       SW_COND: 1, 0, 1
gr2:   GR_FLAG:1  (GR_TYPE 2 as shipped)  GR_FRAC_ROOF:0.4,0.8,0.2  IRHO: every hour on
```

Each run: 12 columns (one j row), 30 mass levels, `dt = 60 s`, four
consecutive `BEP_BEM` calls (30 for `long`).  `urban_param_init` reads the table,
`urban_var_init` (option 3, not a restart) builds the initial state from the
columns' `IVGTYP`, `TSK`, `TSLB`, `SMOIS` and handed-in `FRC_URB2D`, and the
driver zeroes `EMISS/RL_UP/RS_ABS/GRDFLX_URB` and `B_Q_BEP` on every column
before every call exactly as `module_sf_noahdrv.F:1639-1647` (and
`module_sf_noahmpdrv.F:3646-3656`) do.  Column design (`run_bep_bem.F90`):
two rural columns, the stock `ISURBAN -> UTYPE` mapping and the explicit LCZ
rows, handed-in fractions 0 (the table-fraction arm), 0.01, 0.5, 0.6, 0.7,
0.8, 0.9, 0.95, 0.99, 1.0, local hours from night through dusk including the
24 h wrap of `nhourday` inside the four steps, wet (`RAINBL` 0.2 to 2 mm) and
dry, stable and unstable first levels, and two columns with gridded
morphology (`HGT_URB2D > 0`, a height histogram in `HI_URB2D`).

Array names are the dump names with `/` written as `__`; every array keeps
Fortran index order (`(ncol, k, 1)` for a WRF `(ims:ime, k, jms:jme)`
field).  `tbl__*` are the `module_sf_urban` table arrays after
`urban_param_init`; `init__*` the `urban_var_init` inputs; `state0__*` the
state after `urban_var_init`; `stepN__*` the inputs of call N and
`stepN__out__*` its outputs; `final__*` the state after the last call.
Nothing in the files is a hand-computed expectation.

Toolchain: GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0, glibc 2.43
(Ubuntu, x86-64), `-O0` with WRF's own defines
(`-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4
-DDWORDSIZE=8 -DLWORDSIZE=4`).  Libm/libgcc symbols the `-O0` objects of
`module_sf_bem.F` and `module_sf_bep_bem.F` import: `acosf asinf atanf cosf
expf log10f logf powf sinf tanf __powisf2`; no `_ZGV*`.

Compiler spread, measured: the same sources at WRF's own
`-O2 -ftree-vectorize -funroll-loops` import `_ZGVbN4vv_powf` (glibc libmvec)
and `sincosf`, and differ from the `-O0` fixtures in `grdflx_urb` (at most
80 ULP, on the 30th call of `long`; 36 over the four-call fixtures) and
`rl_up` (at most 4 ULP) only; every other array of every step of every
variant, every prognostic layer after 30 calls included, is bit-identical
between the two builds.

Reproduce (measured 2026-09-30: all five `data.bin` hashes above
come back bit-identical):

```
bash tools/urban_wrf471_oracle/build.sh ~/work/urban-wrf471/WRF BUILD
bash tools/urban_wrf471_oracle/run_bep_bem.sh BUILD ~/work/urban-wrf471/WRF/run OUT
python tools/urban_wrf471_oracle/bem_dump_to_npz.py OUT gpuwm/data/urban/oracle/bem
```

`build.sh` itself runs `run_bep_bem` once with no variant arguments, which
is the `stock` fixture (`BUILD/fixtures/bep_bem/data.bin`, same hash).
