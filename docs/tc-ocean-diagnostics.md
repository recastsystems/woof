# Upper-ocean diagnostics

Set `WOOF_TC_OCEAN_DIAGNOSTICS=1` for a run with the PWP ocean enabled.
The normal history writer emits these fields at its existing output cadence:

| Field | Units | Meaning |
|---|---|---|
| OCEAN_OHC26 | kJ cm-2 | Heat above 26 C down to the first downward crossing |
| OCEAN_D26 | m | Depth of that crossing |
| OCEAN_D26_CENSORED | 1 | One when the ocean column ends above the crossing |
| OCEAN_ILD | m | First 0.2 K cooling from the 10 m reference |
| OCEAN_MLD | m | First 0.03 kg m-3 density increase from the 10 m reference |
| OCEAN_BARRIER_LAYER | m | max(ILD - MLD, 0) |

OHC is a **lower bound** where D26 is censored. D26, ILD and MLD are missing
when their crossings are unresolved; barrier thickness is missing if either
layer depth is unresolved. Land, lakes and invalid profiles are missing.
The field is surface-connected: a deeper isolated warm inversion is excluded.

The CUDA calculation reconstructs values linearly at the layer centres, with
constant values between the end centres and the surface/bottom. It uses the
model's live density, not a separate seawater equation of state. OHC uses
1025 kg m-3 and 3985 J kg-1 K-1. The disk planner includes the same six fields.
No diagnostic work runs at atmospheric timestep cadence. With the environment
variable absent, this feature creates no arrays and changes no model state.

The public `woof.ocean.tc_diagnostics.diagnose` API also accepts arbitrary
CUDA column arrays in (..., level) order. If no density profile is supplied,
it uses the caller's linear density constants. Hex uses this option to match
its PWP equation of state and writes lowercase field names.

References implemented independently from published definitions:
[OHC](https://doi.org/10.1175/2007WAF2006111.1) and
[mixed-layer thresholds](https://doi.org/10.1029/2004JC002378).
These diagnostics do not establish forecast skill or authorize a TC default.
