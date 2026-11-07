# Atmosphere and spectral-wave exchange

Original Apache-2.0 implementation from published equations. No ocean-model
implementation code was read or translated for these changes.

The parent atmosphere mass grid drives the waves. Integer blocks restrict
earth-relative winds and stress with water-cell area weights. Vectors rotate
before averaging. The inverse map is piecewise constant and masks land and ice.
This numerical map is an explicit first-order exchange choice; it does not add
wave energy or momentum to the atmosphere when roughness_feedback is false.
Moving children retain the atmosphere's own surface laws and never reuse stale
parent wave indices. Wave feedback on moving children is not claimed.

The spectral solver supplies effective roughness from its wave-supported stress.
The relevant stress formulation is Janssen (1991), Journal of Physical
Oceanography 21, 1631-1642, DOI
10.1175/1520-0485(1991)021<1631:QLTOWW>2.0.CO;2. The coupling layer does not apply
the stress enhancement a second time. The solver's numerical details and source
pack citations belong to waves/SCIENCE.md.

High-wind momentum drag follows the envelope in Holthuijsen, Powell and Pietrzak
(2012), Journal of Geophysical Research 117, C09003, DOI 10.1029/2012JC007983,
equation (1) and Table 2. In neutral conditions:

    1000 Cd = max(0.7, min(a + b (U10/27.5)^c, d (1-(U10/54)^e)))
    z0_limit = 10 exp(-0.4/sqrt(Cd))

The table's coefficients interpolate linearly in directional spreading. For
(a,b,c), the no/collocated/opposing-swell row is (1.05,1.25,1.4) at spread <=30
or >=80 degrees; the cross-swell row is (0.7,1.1,6) at 50 degrees. For (d,e),
the endpoint row is (2.3,10) at spread <=45 or >=55 degrees and (8.2,2.5) at
50 degrees. The implementation is in roughness.py beside the equation citation.
It needs the spectrum-integrated circular spread, not a partition's width.

This is an upper envelope on spectral roughness, not a replacement of all drag
by an empirical wind curve. A smoothstep transition from 20 to 27.5 m/s joins
the uncapped spectral roughness to the capped solution. Those joining endpoints
are an explicit numerical choice, not a published fitted equation. The heat and
moisture exchange formulations, including selected tropical-cyclone scalar flux
options, remain owned by the atmosphere surface layer.

The saturation motivation is Powell, Vickery and Reinhold (2003), Nature 422,
279-283, DOI 10.1038/nature01481, and Donelan et al. (2004), Geophysical Research
Letters 31, L18306, DOI 10.1029/2004GL019460. These papers establish why simply
extrapolating increasing drag is inappropriate. Curcic and Haus (2020), GRL 47,
e2020GL087647, DOI 10.1029/2020GL087647 revises the laboratory saturation data;
the 2004 laboratory threshold is therefore not treated here as an exact
universal limit. The 2012 envelope is itself a preliminary observational
approximation, and this coupling still requires whole-case validation.

The older 72 h coupled comparison changed SST, waves and surge together. Its
intensity degradation cannot establish a wave-drag cause independently. Calm
wave initialization and an unconstrained bulk roughness extrapolation are
identified limitations to remove, not a claim of measured causal attribution.

Timing uses CUDA events on the same stream and host enqueue timestamps. Events
are harvested at output cadence; no step waits for device completion. A filled
event ring makes the timing receipt incomplete rather than adding a hidden wait.
