# Operational-branch form oracles (the HRRR door)

The HRRR door (`woof/data/physics_sources/request-defaults.v1.toml`) selects
forms that exist only in the operational RAP/HRRR branch, NOAA-EMC/HRRR tag
v4.1.21 (commit 40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827,
`sorc/hrrr_wrfarw.fd/WRFV3.9`). None of them is in stock WRF v4.6.1, so the
referee is the branch's own code. `fetch_fork.py` downloads the files from
that public commit on the box and checks each against its pinned sha256.

| form | oracle | the branch's code compiled |
|---|---|---|
| `mp_zero_out = 2`, `mp_zero_out_thresh = 1e-12`, `mp_zero_out_all = 1` | `zero_out_oracle.py` | `phys/module_microphysics_zero_out.F` unchanged, called as `solve_em.F:4070-4141` calls it |
| `alb_sol = 1` | `albsol_oracle.py` | `radconst`, `calc_coszen` and `module_radiation_driver.F:781-789, 1013-1014, 1032-1034, 1038-1063` verbatim |
| `upper_wind_limiter_form = noaa_wrf39` | `upper_wind_regimes.py` | `module_small_step_em.F:1705-1736` verbatim |
| `diff_6th_form = noaa_wrf39`, `diff_6th_factor2` | `diff6_regimes.py` | `sixth_order_diffusion` verbatim (tools/wrf_diffusion_oracle/diff6_fork_oracle.py), and the per-row step expression of `module_em.F:1216-1217` |
| `rrtmg_cloud_optics_form = noaa_wrf39` | `cloud_form_transplant.py` + tools/rrtmg_legacy_column_oracle (`fork-cloud` arm) | WRF v4.6.1 RRTMG with the branch's two wrapper cloud rules in place of v4.6.1's |
| `ruc_soilprop`, `ruc_irrigation`, `ruc_snow` = `wrf_45`, `ruc_qvg_cold_start = air`, `ruc_2m_diagnostic = log_profile` | tools/ruc_lsm_gpu_oracle `--lineage fork` | the branch's `module_sf_ruclsm.F` and `module_sf_sfcdiags_ruclsm.F` unchanged, inside its surface-driver arm |

Every WOOF side is the production device path. Every comparison is word for
word (uint32), with the ULP distance on the ordered float32 line.

## Rerun on a box (CPU for Fortran, GPU through the mutex)

    T=<tree>; source /work/pverify/base/env.sh
    F=<dir>; python $T/tools/fork_forms_oracle/fetch_fork.py $F
    R=<runs>
    python $T/tools/fork_forms_oracle/zero_out_oracle.py build $F $R/zero_out/build
    python $T/tools/fork_forms_oracle/albsol_oracle.py build $F $R/albsol/build
    python $T/tools/fork_forms_oracle/upper_wind_regimes.py build $F $R/upper_wind/build
    python $T/tools/fork_forms_oracle/diff6_regimes.py build $F/dyn_em/module_big_step_utilities_em.F $R/diff6/build
    bash $T/tools/rrtmg_legacy_column_oracle/build.sh <WRF-4.6.1 root> $R/rrtmg/build fork-cloud fork
    python $T/tools/rrtmg_legacy_column_oracle/make_columns.py $R/rrtmg/cols
    (cd $R/rrtmg/build/fork-cloud && ./rad_column_oracle $R/rrtmg/cols/in_REGIME.bin $R/rrtmg/wrf/fork-cloud/out_REGIME.bin)

Then, through `card-lease`, once with `GPUWM_WRF_EXACT=1` and once
without:

    python $T/tools/fork_forms_oracle/zero_out_oracle.py run $R/zero_out/build $R/zero_out/out
    python $T/tools/fork_forms_oracle/albsol_oracle.py run $R/albsol/build $R/albsol/out [--before]
    python $T/tools/fork_forms_oracle/upper_wind_regimes.py run $R/upper_wind/build $R/upper_wind/out
    python $T/tools/fork_forms_oracle/diff6_regimes.py run $R/diff6/build $R/diff6/out
    RRTMG_CLOUD_OPTICS_FORM=noaa_wrf39 python $T/tools/rrtmg_legacy_column_oracle/run_woof.py $R/rrtmg/cols $R/rrtmg/woof-noaa
    python $T/tools/rrtmg_legacy_column_oracle/compare.py $R/rrtmg/cols $R/rrtmg/wrf $R/rrtmg/woof-noaa strict fork-cloud

RUC (`WRF_SRC` is the fetched branch tree; `RUC_TABLES` the WRF v4.6.1
`run/` directory, the tables WOOF loads):

    WRF_SRC=$F RUC_TABLES=<WRF-4.6.1>/run MUTEX_TAG=<lane> \
      bash $T/tools/ruc_lsm_gpu_oracle/run_case.sh $R/ruc/fork-hrrr-strict strict --nzs 9 --mosaic 1 --rdlai2d 1 --lineage fork

`pack_fixtures.py` packs the branch's words into `tests/data/oracles/fork_forms`,
and `tests/test_noaa_wrf39_fork_forms_oracle.py` reruns the WOOF side on any
card without Fortran.
