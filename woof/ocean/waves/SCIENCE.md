Current action solver (release candidate, not yet qualified)
==========================================================

`create_solver` selects `action.ActionSolver`. The legacy `WaveModel` described
below remains available for existing direct callers. It is not the new source
pack implementation. `source_packs.json` records the candidate constants and
scientific choices, not a claim of skill against observations.

The new state is N=E/(2*pi*f), integrated with df dtheta. It is compacted to
wet cells. Finite-volume geographic propagation uses MUSCL MC reconstruction
and SSP-RK2. Shared face fluxes cancel on adjacent cells. The full anisotropic
garden-sprinkler diffusion tensor uses a fixed 3600 s spreading horizon in both
structured and Voronoi designs. This is a numerical approximation to wave age;
it requires the basin swell arrival test. A common donor flux budget preserves
positivity without clipping cells independently. Geographic CFL includes both
advection and diffusion. Depth-driven refraction and great-circle turning use
the inherited implicit cyclic solve. Local time-varying depth adds conservative
implicit frequency advection, with closed frequency endpoints. No current input
is supplied by the present shared ocean ABI; current refraction is not claimed.

Sources use a fixed interval, diagonal semi-implicit integration and the
time-scaled increment limiter. There is no growth-selected time loop. Two
source half-intervals surround geographic transport. Wind input, dissipation,
linear seeding and DIA enter one increment.
The former separate source update followed by a donor-capped DIA update need
not preserve an equilibrium of the summed physical source terms. The new
diagonal includes the analytic derivative of the discrete DIA quartets,
including interpolation and deposition weights. Loss coefficients are frozen
in that diagonal linearization; off-diagonal terms remain omitted. The HJ
increment and nonnegative-energy limits stay active. This source update is
not an exactly conservative implicit solve of DIA in isolation.
The default pack remains provisionally st4. Initial two-storm comparisons
favour its height error over janssen, but do not satisfy every reference metric
or complete the required basin qualification.
The st4 pack follows directional saturation, cumulative breaking, wind sheltering
and laminar/turbulent swell friction. It chooses the published constant friction
branch fe=.006, not the Grant-Madsen variable-friction branch. The janssen pack
uses the positive-square-root wavenumber mean and revised whitecapping constants.
Both use a .002 m cap on the base roughness, a declared high-wind modification
to uncapped BAJ. Effective roughness can exceed the base cap through wave stress.
The resolved wind stress includes a numerical quadrature of the f^-5 tail.
The corrected calculation accumulates tail stress across all directions before
advancing to the next tail frequency. The former per-direction accumulation
did not apply the shared residual-stress vector in the sheltering equation.
Tail insertion uses the pack's published cutoff rule. Bottom friction and
depth-induced breaking remain active for both packs.

The revised growth candidate extends the DIA work grid with a diagnostic f^-5
tail far enough to include every quartet feeding the retained grid. The cubic
coefficient remains C*g^-4*f^11 for per-Hz variance density. Source moments
include analytic tail integrals above the last bin edge. Stress-tail quadrature
starts at that edge as well. The janssen cutoff uses the windsea inverse-frequency
mean, with cutoff 2.5*fmean as specified in TM509 equation 9. The previous
code incorrectly retained the superseded cycle-4 4*fPM floor. Removing that
floor is an equation correction; its growth result still needs qualification. A diagnostic tail exchanges energy
with the resolved range, so closed finite-grid DIA conservation is not claimed
for this extended calculation.

Revised smoke 82b17b1b66 completed 72 h but failed the growth gate: duration
Hs*g/U10^2 at 10 m/s was .1851 (st4) and .1800 (janssen). At 20 m/s it was
.2472 and .2429. Both 10 m/s curves undergrew before full development, and
the shortest-fetch heights remained too high. Those results authorize no
skill claim. Subsequent joint-source and collective-tail-stress calculations
removed the large high-frequency trapping/downshift defect but still fail
growth qualification. A 240 h duration check at aee3b6f264 gives normalized
height about .205 for both 10 m/s packs, versus .297/.280 at 20 m/s. The same
eight-arm smoke at e7e0490659 with fixed dt=60 s completed in 356.13 s and
retained that bias. Refining the fixed step did not resolve the short-fetch
excess or the late equilibrium error. No coefficient sweep was used.

Initial standalone aa5e7cc9f3 completed both source packs for the two storms.
For the first storm, Hs RMSE is .545/.639 m versus .495 m for the common-time
GFS-Wave comparison. For the second, it is .552/.682 versus .682 m, but peak
period and direction errors are larger than GFS-Wave. Spectral peaks are too
broad and windsea/swell mode switches differ visibly. The basin arm timed out
at 156 of 192 hours. These results are incomplete qualification, not a release
skill claim. Exact cases, pictures, input identities and commands remain in
the lane's delivery evidence rather than generic model defaults.

The historical growth numbers above are not a measurement on the final source.
The current finite growth evidence and its limits are recorded in
validation/GROWTH-REFEREES.md. The ST4 paper (2010 JPO40 p1925) explicitly
describes continued slow growth, so the former 240-hour stationary PM gate
is retired. This changes the referee, with no source coefficient change.

The growth comparison stops empirical power laws at their stated limits and shows
the PM references (Hs*g/U10^2=.24-.27 and fp*U10/g=.13), without imposing
them as an ST4 stationary limit. References: Pierson and
Moskowitz (1964), JGR 69, 5181-5190, DOI 10.1029/JZ069i024p05181; Alves et al.
(2003), JPO 33, 1301-1323, DOI 10.1175/1520-0485(2003)033<1301:RTPALF>2.0.CO;2.

Watershed topology follows ascent to peaks on a single smoothed spectrum;
all statistics integrate the unsmoothed variance. The peak criterion c < 1.5 U10 cos(theta-wind) identifies wind sea.
Adjacent systems merge by the published spread, saddle and low-variance rules
described below. The three strongest remaining swell systems stay distinct;
residual systems join the nearest retained peak. Partition variances sum to
total variance. Tp fits a parabola through three frequency-density bins.
Dp is the energy-weighted direction at the peak frequency; dir uses all bins.
Bulk calm and dry values are zero; unresolved active initialization is an error.

Primary equation references for the new code:

* Holthuijsen (2007), Waves in Oceanic and Coastal Waters, Cambridge University
  Press, DOI 10.1017/CBO9780511618536. Dispersion and action-space rays.
* van Leer (1979), Journal of Computational Physics 32, 101-136,
  DOI 10.1016/0021-9991(79)90145-1. Limited linear reconstruction.
* Shu and Osher (1988), Journal of Computational Physics 77, 439-471,
  DOI 10.1016/0021-9991(88)90177-5. SSP time integration.
* Booij and Holthuijsen (1987), Journal of Computational Physics 68, 307-326,
  DOI 10.1016/0021-9991(87)90060-X. Sub-bin dispersion tensor.
* Janssen (1991), Journal of Physical Oceanography 21, 1631-1642,
  DOI 10.1175/1520-0485(1991)021<1631:QLTOWW>2.0.CO;2. Wind and wave stress.
* Ardhuin et al. (2010), Journal of Physical Oceanography 40, 1917-1941,
  DOI 10.1175/2010JPO4324.1. Equations 8-24, Tables A1-A2. Constant-fe branch
  and Table A2's saturation coefficient are chosen explicitly. Preprint
  https://arxiv.org/abs/0907.4240 and the published paper were consulted.
* Ardhuin, Chapron and Collard (2009), Geophysical Research Letters 36, L06607,
  DOI 10.1029/2008GL037030. Swell dissipation. This is the swell-observation
  paper; JPO 39, 2820 concerns surface currents, a different paper.
* Bidlot, Janssen and Abdalla (2007), ECMWF Technical Memorandum 509,
  DOI 10.21957/m97gmhqze. Equations 4-13, whitecapping and limiter constants.
  https://www.ecmwf.int/sites/default/files/elibrary/2007/73707-revised-formulation-ocean-wave-dissipation-and-its-model-impact.pdf
* Hersbach and Janssen (1999), Journal of Atmospheric and Oceanic Technology
  16, 884-892, DOI 10.1175/1520-0426(1999)016<0884:IOTSFB>2.0.CO;2.
  Joint source increment with fully implicit weighting, equation 2 with alpha=1.
  The numerical derivation is also stated in the published ECMWF IFS Cycle
  40r1 scientific documentation, Part VII, chapter 5 equations 5.4-5.8:
  https://www.ecmwf.int/sites/default/files/elibrary/2014/9207-part-vii-ecmwf-wave-model.pdf
  Only published equations were consulted, never model implementation code.
* Cavaleri and Malanotte-Rizzoli (1981), Journal of Geophysical Research 86,
  10961-10973, DOI 10.1029/JC086iC11p10961. Linear seeding. The low-frequency
  filter uses the finite fully-developed frequency following Tolman (1992),
  Journal of Physical Oceanography 22, 1095-1111,
  DOI 10.1175/1520-0485(1992)022<1095:EONOTP>2.0.CO;2.
* Hasselmann et al. (1985), Journal of Physical Oceanography 15, 1378-1391,
  DOI 10.1175/1520-0485(1985)015<1378:CAPOTN>2.0.CO;2. DIA.
* Komen et al. (1994), Dynamics and Modelling of Ocean Waves, Cambridge
  University Press, DOI 10.1017/CBO9780511628955. Finite-depth DIA scaling.
* Hasselmann et al. (1973), Deutsche Hydrographische Zeitschrift supplement
  A8(12), JONSWAP. No DOI verified for the original report. Primary text:
  https://pure.mpg.de/rest/items/item_3262854_4/component/file_3282032/content
* Battjes and Janssen (1978), Proceedings of ICCE 16, 569-587,
  DOI 10.9753/icce.v16.32. Depth-induced breaking.
* Hanson and Phillips (2001), Journal of Atmospheric and Oceanic Technology
  18, 277-293, DOI 10.1175/1520-0426(2001)018<0277:ASAOOW>2.0.CO;2.
  Watershed partitioning. Reconstruction coefficients and their references
  remain solely in the data lane's partition_shapes.v1.json.

Qualification remains unfinished until growth, three real-case comparisons,
visual inspection and stage/memory measurements are recorded in the lane report.

The memory correction preserves the source equations and spectra. Cell-local
source and watershed intermediates share a 1024-cell arena; the prior wind-input
sign is packed into one bit per spectral bin. Structured transport borrows
three arrays beside the action state, and one transport array temporarily holds
directional turning. Dispersion is solved once per frequency rather than once
per direction. Allocation receipts deduplicate aliased CUDA owners and report
both live and reserved pool bytes. The resource target is about 25 KB per wet
cell on real-size grids; measured evidence is required before claiming it.

Legacy solver equations and numerical choices
===============================

All CUDA and Rust code in this package is original. Published equations were
consulted; no WW3, SWAN or ecWAM implementation was copied or translated.

The prognostic spectrum is elevation variance E(f,theta), m^2 Hz^-1 rad^-1,
with travel direction theta counterclockwise from east and absolute intrinsic
frequency sigma = 2 pi f. Bathymetry is static and currents are absent, so
multiplication of the action equation by sigma gives the conservative variance
balance. Spatial fluxes use finite-depth group velocity, spherical cell areas,
and upwind face values. Directional flux includes great-circle rotation and
bathymetric refraction. Frequency shifting is zero with static depth and no
current. Coastlines absorb outgoing variance; open edges have zero incident
spectrum unless a caller provides incident edge spectra. Spatial propagation is first-order explicit upwind. Directional turning uses
conservative backward-Euler upwind, solved as a cyclic tridiagonal system on
CUDA, one thread per cell/frequency ring. Its M-matrix retains positivity and
directional variance without a directional Courant restriction; coastal depth
gradients therefore do not force subsecond steps on every ocean cell. Frequency
is geometric and direction periodic.
Dispersion sigma^2 = g k tanh(k h) is solved by Newton iteration on CUDA.

Wind uses the WAM cycle 4 Janssen formulation: B = beta rho_a/rho_w
(u*/c)^2 max(cos(theta-theta_w),0)^2 sigma, with beta = 1.2/kappa^2 lambda
(log lambda)^4, lambda = g z_e/c^2 exp(kappa c/|u* cos|), lambda < 1.
The log profile and wave stress determine u* and effective z_e using eight
underrelaxed iterations. z_e = z0/sqrt(1 - tau_w/tau), z0 = .01 u*^2/g plus
the viscous smooth-surface contribution .11 nu/u*. Wave stress is the magnitude
of rho_w g integral (k/sigma) B E (cos theta,sin theta) df dtheta.
The stress ratio is bounded at .95 to keep the surface-layer fixed point finite.
A Cavaleri-Malanotte-Rizzoli linear source seeds calm water, with the
Pierson-Moskowitz low-frequency filter. The per-radian-frequency source's 1/(2pi)
is removed when converting to per-Hz density. Constants: kappa=.41,
rho_a=1.225 kg/m^3, rho_w=1025 kg/m^3, nu=1.5e-5 m^2/s, g=9.80665 m/s^2.

Whitecapping is -Gamma sigma_tilde (k/k_tilde) E, Gamma = 4.10e-5
(.5 + .5 k/k_tilde) (k_tilde sqrt(m0)/sqrt(.00302))^4.
sigma_tilde is the inverse variance-weighted inverse sigma; k_tilde is the
inverse square of the weighted inverse square root of k. JONSWAP bottom
friction is -.038 (sigma/g/sinh(kh))^2 E. Depth-limited breaking uses
Hmax=.73h, (1-Qb)/log(Qb)=-8m0/Hmax^2 and a sink
-.25 Qb f_mean Hmax^2 E/m0. The probability root is solved in log space.
Wind growth is exponential and dissipation implicit. A source step uses B dt <=
.2 from the current diagnosed growth bound, in addition to the propagation CFL
and configured maximum step. The shared asynchronous ocean ABI instead performs
that growth-limited local subcycling inside each cell block on CUDA, with spatial
transport substeps chosen from a conservative group-speed envelope. It uses the
same source equations, and evaluates DIA at each transport substep. All source
terms are on by default.

DIA uses two mirrored resonant configurations, lambda=.25, angles
(+11.48,-33.56) degrees and their negatives, Cnl=3e7, and f^11/g^4 scaling.
With E in per-Hz units the cubic transfer is
Cnl f^11/g^4 [E0^2(Eplus/1.25^4 + Eminus/.75^4)
-2 E0 Eplus Eminus/(1-.25^2)^4]. A positive transfer removes two action units
from the central bin and adds one at each partner. Physical-frequency linear
interpolation makes the partner frequencies sum to 2 f0; deposition uses the
same weights and converts action back to variance at each recipient's actual
frequency. Thus integrated action and variance conserve to floating-point
summation error. Discrete angular interpolation does not exactly conserve vector
momentum. Finite-depth correction uses kd=max(.5,.75 k_tilde h) and
1+5.5/kd (1-5kd/6) exp(-1.25kd).

The finite frequency domain contains complete in-band quadruplets only; no
extrapolated high-frequency tail is imposed. Whole-quadruplet donor limits and a
cell-wide positive update factor retain both discrete invariants while preventing
negative bins. This limiter changes transfer magnitude if the step is too long;
receipts record its minimum. The default grid is 36 frequencies from .035 Hz
with ratio1.1 and 36 directions, covering approximately .035 to .984 Hz.
The statistical partition is an age criterion, not watershed partitions of each
spectral peak. Hs=4 sqrt(m0); Tp is the inverse of the peak of direction-integrated
spectral density, and direction is the integrated meteorological FROM bearing.

Primary technical references consulted (equations, not implementation code):

* SWAN technical documentation, wind equations 2.33-2.43:
  https://swanmodel.sourceforge.io/online_doc/swantech/node15.html
* Whitecapping, bottom friction and depth breaking:
  https://swanmodel.sourceforge.io/online_doc/swantech/node16.html
* DIA configurations and depth scaling, equations 2.76-2.80:
  https://swanmodel.sourceforge.io/online_doc/swantech/node18.html
* NOAA WAVEWATCH III technical manual, spectral balance and DIA discussion:
  https://polar.ncep.noaa.gov/waves/wavewatch/manual.v4.18.pdf

Numerical conservation and positivity tests establish implementation properties.
They do not establish forecast skill. A storm run and public-buoy comparison must
accompany qualification, with calm spin-up, resolution, boundary effects, the
ERA5 wind representation and the observation definitions stated explicitly.

Takeover correction: cached transport and partitions
---------------------------------------------------
Structured MUSCL slopes are calculated once per cell and bin, and a shared
face flux once per interface, in six-frequency tiles. Coast faces retain the
contracted absorbing condition: outgoing action leaves, with zero incoming
spectrum. An intermediate closed-wall candidate was rejected before any whole
forecast. The CFL sums positive outgoing characteristics and the diffusion diagonal,
removing the previous all-face speed bound and extra diffusion factor of four.
The SSP safety coefficient remains .4. This changes temporal truncation error;
it is assessed separately from the cached-flux algebra comparison.
Directional turning solves the same cyclic backward-Euler system using float32
parallel cyclic reduction and the same rank-one corner correction. Reference:
Hockney (1965), JACM 12,95-113, DOI 10.1145/321250.321259.
Wind closure retains eight iterations and all published coefficients. Ring
stress is calculated once before the frequency prefix sum; iterations share a
single launch. Warp reductions change addition order, not spectral measures.

The output partitioner uses the peak criterion c < 1.5 U10 cos(delta) from
Hanson and Phillips (2001), JAOT 18,277-293,
DOI 10.1175/1520-0426(2001)018<0277:AAOOSD>2.0.CO;2, section 2a.
Adjacent swell basins merge using their model-spectrum settings kappa=.5 and
zeta=.75, updating variance and frequency-vector moments after each merge.
The low-energy merge threshold .0025 m2 is the published variance threshold
reviewed by Portilla et al. (2009), JAOT 26,107-122,
DOI 10.1175/2008JTECHO609.1, section 2a and Table 1. It is not tuned to a case.
Residual basins beyond the three output swells join the nearest retained
spectral peak, rather than all joining a possibly empty third swell. All
resolved variance is retained. Wind-input classification used for source tails
remains the distinct TM509 criterion.

Frequency-group transport uses six-frequency groups with depth-independent
maximum cg=0.6 g/sigma at the lowest group frequency, and the unchanged .4
SSP safety factor applied to outgoing advection plus the diffusion diagonal.
Each group takes ceil(call_dt/group_dt) equal pieces; boundary spectra are
evaluated at its own midpoint. Implicit direction turning uses the same group
interval. Geometry-dependent turning rates are evaluated in the directional solver.
A rejected intermediate build computed them into scratch before transport,
which overwrote that scratch. No production run used that build.
This follows the frequency-specific propagation scheduling described in
Tolman et al. (2002), Weather and Forecasting 17,311-333,
doi:10.1175/1520-0434(2002)017<0311:DAIOWG>2.0.CO;2, section 2.
The operator and source constants are unchanged; truncation differences from
the former common smallest-frequency interval must be recorded. Wind closure
uses one warp per cell to avoid idle warps during sequential tail integration;
all eight iterations and all 24 tail rings remain active.

Cell-local surface closure now spans the whole wet domain, borrowing the
transport stage array for beta while propagation is inactive. Source means
and diagnostic tails are also launched over the domain. The DIA/saturation
arenas remain tiled. This reorders independent cells only; every cell retains
its two eight-iteration closures per source half and final diagnostic closure.
The 1024-cell source tile had left only about six active single-warp blocks
per multiprocessor during the long sequential stress tail. No extra full
spectral array or changed source coefficient is needed to remove that limit.

Partition topology uses on-chip scratch up to 1600 spectral bins. Larger
exposed spectral configurations use the identical block algorithm in a bounded
global-memory tile, preserving every frequency and direction. The standard
36-by-36 grid does not allocate that additional scratch.

## Angular correction, 17:55Z cycle correction 2

The first spherical packet exposed excess directional diffusion in the first-order backward-Euler donor operator. Replace it by conservative piecewise-linear reconstruction with the monotonized-central slope limiter (van Leer1977,JCP23,276-299, DOI10.1016/0021-9991(77)90095-X) and SSPRK2 (Gottlieb and Shu1998,Math.Comp67,73-85, DOI10.1090/S0025-5718-98-00913-2). Ring-local angular substeps enforce CFL<=0.45 on the analytic face rate. This is default for both angular entry points and does not alter source coefficients. The spherical reference now averages exact backward rays across each directional bin with16-point Gauss quadrature; the first reference used point samples and could alias the narrow arrival. This reference correction is reported separately from the transport correction.


Optional point-budget instrumentation records actual split-operator energy increments on device, including both diagnostic tail overwrites, geographical transport (with boundary influence), angular transport, and frequency transport. Source terms are split into wind input, seed, DIA, bottom loss, depth breaking, whitecapping, cumulative breaking and swell loss. Raw and applied increments are both retained; applied terms use the same joint diagonal denominator and increment limiter, and their sum is checked against the actual source update. These are fixed-cell Eulerian budgets, not material ray-following conservation statements. Output is only at the configured field cadence. The observer never writes a prognostic field or closure iteration state.

## Swell-friction revision after the Pacific term budget

The 2016 GRL paper, doi:10.1002/2015GL067566, equations3-4 and section3, corrects
the factor-two viscous expression and reports fe=0.0037 as the median of1050
independent tracks. The constant-friction branch now uses this population value,
replacing0.006, which was chosen inside the older small-sample range. This is an
external empirical revision, not a fit to the present cases or proof of a universal
friction constant. The turbulent16*sigma² form is unchanged. Cdsv=1.2 remains
relative to corrected viscous theory; the Reynolds switch and all other sources
are unchanged. The source-free packet validation tools remain.

Evidence: cycle-1900/diagnosis.json and budget/*.json in the lane. At downstream
points,0.055-0.100Hz integrated swell sinks are1.446617 and0.589303m² over192h;
whitecapping and cumulative breaking in that band are zero. DIA is+0.180701 and
-0.020360m². This identifies swell friction as the dominant local physical sink,
not the entire model-minus-GFS error. Eulerian geographic changes are not material
dissipation. Arbitrary-point GFS E(f) is reconstructed from partitions and labelled;
actual station spectra and NDBC decide the final whole-run bar.
