"""EXPERIMENTAL (WOOF-ArWen v1.2): analysis starts, ensemble and operator support.

Nothing in this package is on a certified forecast path, and nothing here is
wired into a default route.  The forecast executable, the physics driver, the
product side, and every certified verification case run exactly as they did
before this package existed; these modules are libraries a caller must reach
for explicitly.  Every public entry point stamps ``status="experimental"``
into its provenance so a downstream product can refuse to ship a field that
passed through it.

What is here since 3.0.3.  The local data assimilation that lived in this
package (``woof local-da``, the LETKF, the static-covariance analysis, the
radar, satellite and point-observation adapters, the reflectivity nudging,
the spread repair and the cycle drivers) was removed in gpuwm 3.0.3: it was
experimental and no longer maintained.  What stays is used by forecasts,
ensembles and verification, and keeps its import path:

- :mod:`woof.da.analysis_door` -- ``woof run --wrfinput DIR --analysis
  BUNDLE``: a forecast started from an analysis made elsewhere, with
  :mod:`woof.da.analysis_boundary`, :mod:`woof.da.maspt`,
  :mod:`woof.da.host_analysis` and :mod:`woof.da.hydrostatic`.
- :mod:`woof.da.iau` -- the incremental analysis update forcing the dycore
  and the tiled road apply.
- :mod:`woof.da.forecast_heating` and :mod:`woof.da.radar_tten` -- radar
  latent heating in a plain forecast (``[radar_heating]``).
- :mod:`woof.da.perturb`, :mod:`woof.da.ensemble_stats` and
  :mod:`woof.da.moments` -- ensemble initial-condition perturbations.
- :mod:`woof.da.positivity` -- the hydrometeor positivity policy of the
  ensemble cycle seam (:mod:`woof.ensemble.cycle`).
- :mod:`woof.da.enprod` -- ``woof enprod``, the ensemble products.
- :mod:`woof.da.obsop` and :mod:`woof.da.obsop_cwp` -- simulated
  reflectivity, radial velocity and cloud water path on the model grid,
  read by verification and by the observation front doors.
- :mod:`woof.da.background` -- the background source registry the regional
  preparation reads.

The deliberate absence of any submodule import here is a design statement, and
it is essential twice over.  A ``from woof.da import *`` that pulled in
cupy at import time would make every symbol here a GPU dependency for callers
that only wanted a dataclass, and would break the test suite's ``-m "not
gpu"`` guarantee (see ``tests/conftest.py``).
Import the module you mean.

EXPERIMENTAL.  Interfaces here are not covered by the v1 stability promise and
may change without a deprecation cycle.
"""

from __future__ import annotations

#: Stamped into every provenance dict this package emits.
STATUS = "experimental"

#: Consumers that refuse experimental code can gate on this.
EXPERIMENTAL = True

__all__ = ["STATUS", "EXPERIMENTAL"]
