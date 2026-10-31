# Experimental HRRR-like MPAS physics

These profiles are opt-in. They do not claim to reproduce operational HRRR or to improve observation skill. The default remains the unchanged MPAS convection-permitting suite.

```
woof hex forecast ... --physics-mode correct --physics-profile hrrr-like-aerosol
woof hex forecast ... --physics-mode correct --physics-profile hrrr-like
```

Both use MPAS-A v8.4.2 aerosol-aware Thompson and MYNN cloud coupling. `hrrr-like` additionally selects `config_mynn_mixlength=1`, `config_mynn_closure=2.6`, and `config_mynn_edmf_mom=1`. The rows live in `woof/hex/data/profiles/mpas-physics-suites.v1.json`, under `experimental_profiles`. Adding another supported combination means adding a row. Native `--physics KEY=VALUE` and explicit namelist values override the row defaults; unsupported selector combinations are rejected. Exact mode, conflicting suites, unknown names and CPU backends are rejected for these experimental rows. Bare forecasts and native suite receipts do not gain profile options.

The prepared init must carry `nc`, `nifa`, and `nwfa`. Use the existing `initialize_species` preparation with the aerosol species row and an verified WIF monthly climatology. No synthetic aerosol fallback is added. The whole forecast receipt records `hex_physics_profile` and `config_hex_icloud_bl`, as well as all resolved native options. A full cold-start run initializes surface aerosol emissions with the native MPAS formula; restarts preserve their carried emissions.

Cloud coupling follows the modern phase-separated, grid-mean MYNN carrier semantics in WRF v4.6.1 `phys/module_radiation_driver.F:1403-1429`: retain the diagnosed fraction on step 1, replace it with the MYNN fraction thereafter; add subgrid liquid/ice only below the resolved-condensate thresholds and above the subgrid-fraction threshold. Radiation copies change; dynamical condensate and MYNN diagnostics do not. The earlier unconditional-add/max-fraction experiment is superseded. This is not the older HRRRv4 single in-cloud condensate representation.

The aerosol column now reuses classic Thompson's bounded GPU workspace transformation. Arithmetic and column order stay unchanged. `WOOF_HEX_THOMPSON_WORKSPACE=0` selects the old local-memory implementation for replay. `WOOF_HEX_THOMPSON_TILE_COLUMNS` bounds storage. On the tested 55-level case, the kernel local frame falls from 46,288 to 64 bytes. The default storage fix is independent of profile selection.

Remaining differences from HRRR include the MPAS dycore, native MPAS aerosol Thompson, MPAS MYNN, Noah rather than cycled RUC, and the absence of HRRR analysis and aerosol assimilation. Existing correction gaps are reported by the driver, including the aerosol precipitable-water diagnostic. No land-state or DA improvement is claimed by these rows.

Tests, from this artifact root:

```
PYTHONPATH=. python tests/test_profiles.py
PYTHONPATH=. python tests/test_cloud_coupling.py
gfortran -shared -fPIC -O0 -ffp-contract=off tests/fixtures/aerosol_emissions_oracle.f90 -o /OWN/WORK/emissions.so
AEROSOL_EMISSIONS_ORACLE=/OWN/WORK/emissions.so PYTHONPATH=. python tests/test_aerosol_emissions.py
gfortran -shared -fPIC -O0 -ffp-contract=off tests/fixtures/cloud_merge_oracle.f90 -o /OWN/WORK/cloud.so
CLOUD_MERGE_ORACLE=/OWN/WORK/cloud.so PYTHONPATH=. python tests/test_cloud_merge_oracle.py
# Through the permitted GPU allocator, with existing certified physics resources:
PYTHONPATH=. python tests/test_aerosol_workspace_gpu.py
CLOUD_MERGE_ORACLE=/OWN/WORK/cloud.so PYTHONPATH=. python tests/test_cloud_merge_oracle.py --gpu
```

Changes marked 2026-10-28: profile table and routing, MYNN/radiation coupling, aerosol column storage, and native cold aerosol emissions. This artifact is modified from the installed 0.4.8 package, not an unmodified MPAS or WRF release. Original distribution notices remain with the artifact. MPAS source uses its BSD licence; WRF v4.6.1/v4.7.1 licence places those releases in the public domain; the hex artifact declares Apache-2.0. See the lane REPORT.md for source URLs, terms, exact run evidence and skill limits.
