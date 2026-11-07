# TC ocean history diagnostics

With `--ocean-mode pwp`, set `WOOF_TC_OCEAN_DIAGNOSTICS=1` to emit
`ocean_ohc26`, `ocean_d26`, `ocean_d26_censored`, `ocean_ild`, `ocean_mld`
and `ocean_barrier_layer` at the normal output cadence. Install the matching
woof branch that supplies `woof.ocean.tc_diagnostics`.

All numerical work runs on CUDA in native owned-cell order; the existing
history mapping restores output order. OHC uses the configured PWP density
and heat capacity. MLD uses its linear equation of state, a 10 m reference
and a 0.03 kg m-3 density increment. ILD uses a 0.2 K temperature decrease.
Barrier thickness is max(ILD - MLD, 0).

OHC is in kJ cm-2, depths are metres. A 26 C isotherm below the column is
censored: D26 is missing and OHC is a lower bound. Unresolved mixed-layer or
isothermal crossings remain missing, as does barrier thickness if either is
missing. Dry columns are missing. This avoids mistaking the 200 m model
bottom for an observed thermocline.

The flag is opt-in pending whole-storm observational qualification. With the
flag absent, no diagnostic arrays are created and model state is unchanged.
References: doi:10.1175/2007WAF2006111.1; doi:10.1029/2004JC002378.
