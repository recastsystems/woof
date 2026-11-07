# Joined radiation direct-beam handoff

The joined radiation field gather omitted `swddir`, which is also the
request for the solver's direct/diffuse surface split, and its memory plan
omitted the corresponding bank. On real native-WRF clear column `c00021`
from `tools/rrtmg_wrf461_oracle/sw_fixtures/fixtures_real.npz` (SHA256
`18c6667f20fc6f22bc96b13cb3785bc3bbd0bbad908895cb49fa95b3d1ab0a14`),
native legacy RRTMG supplies 919.865234375 W/m² direct and
61.42169189453125 W/m² diffuse; WRF's radiation driver passes these
separately to the surface (module_radiation_driver.F:712,2633 and
module_surface_driver.F:3139). The old joined RRTMGP call instead returned
no SWDDIR, losing its own solver's 929.8901977539062 W/m² beam. Including
the request and pricing its bank returns that beam and 61.7899169921875
W/m² diffuse, with zero differing words against ordinary RRTMGP across
192 surface-output words for two members. The distinct RRTMGP spectrum
is not claimed bitwise equal to legacy WRF RRTMG. The clear column's
unused zero native radii are represented by inert positive WOOF radii;
atmospheric and solar inputs remain from the real fixture. This default-on
transport fix applies with CU0 and is independent of CU99. The exact
command, old-field-omission reconstruction, native numbers and actual CUDA
results are `fifth/reproduce_swddir.py`, `fifth/reproduce_swddir.json` and
`fifth/swddir-real-workspace.log` in
`WRF-PORTS-2026-11-01/evidence/port-cu99-kfold/`.
