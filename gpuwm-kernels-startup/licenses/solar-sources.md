# Solar calculation references and terms

The Rust implementation uses the public equations below. It does not call pvlib.
The pvlib BSD notice is included for its equation implementations consulted during
implementation. No restrictive library or observation dataset is a dependency.

- PVWatts v5 DC and inverter equations (8-11): https://pvwatts.nrel.gov/downloads/pvwattsv5.pdf . Equation-based implementation; no source copied from the report. Uses Faiman temperature and isotropic POA, explicitly recorded in every receipt, rather than claiming full PVWatts software equivalence.
- Erbs hourly decomposition and isotropic transposition: https://raw.githubusercontent.com/pvlib/pvlib-python/main/pvlib/irradiance.py . BSD-3-Clause; licence at https://raw.githubusercontent.com/pvlib/pvlib-python/main/LICENSE . "Redistribution and use in source and binary forms, with or without modification". Full notice: solar-pvlib-BSD-3-Clause.txt.
- Single-axis backtracking equations (level cross-axis terrain) and axis convention: https://raw.githubusercontent.com/pvlib/pvlib-python/main/pvlib/tracking.py . Same BSD-3-Clause notice.
- PVWatts inverter equation implementation reference: https://raw.githubusercontent.com/pvlib/pvlib-python/main/pvlib/inverter.py . Same BSD-3-Clause notice.
- Faiman thermal equation and default coefficients: https://raw.githubusercontent.com/pvlib/pvlib-python/main/pvlib/temperature.py . Same BSD-3-Clause notice. The row exposes the coefficients. This is a module-temperature estimate used as cell temperature, rather than Fuentes thermal inertia.
- Haurwitz empirical clear-sky GHI: https://raw.githubusercontent.com/pvlib/pvlib-python/main/pvlib/clearsky.py . Same BSD-3-Clause notice. It does not account for measured aerosol or water vapour.
- NOAA UTC solar position equations: https://gml.noaa.gov/grad/solcalc/solareqns.PDF . NOAA government information, public domain under https://gml.noaa.gov/about/disclaimer.html . "The information on government servers are in the public domain".
- NOAA SURFRAD and SOLRAD data, acknowledged in each acquisition receipt: https://gml.noaa.gov/about/disclaimer.html . Same public-domain terms. Data retain their provenance and are not presented as official model output. Formats: https://gml.noaa.gov/grad/surfrad/ and https://gml.noaa.gov/aftp/data/radiation/solrad/README_SOLRAD.txt . Post-2015 one-minute rows are the defaults; QC=0 only, missing records remain missing.

The real-size verification replay uses public NOAA forecast messages acquired
from the NOAA NODD bucket, not third-party forecast code. Terms at
https://registry.opendata.aws/noaa-hrrr-pds/ say: "NOAA data disseminated through NODD are open to the public and can be used as desired."
NOAA attribution is retained in the acquisition receipt; derived solar output
is identified as a calculation and does not imply NOAA endorsement.
