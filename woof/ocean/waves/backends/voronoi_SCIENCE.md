# Voronoi geographic transport

Original Apache-2.0 implementation from published equations. No external wave
model implementation was read, translated or incorporated.

The conserved variable is action density per Hz and radian in wet-cell order.
Each face contributes one signed flux to both adjacent cells. Face length is
dvEdge, normal centre separation is dcEdge, and the control volume is areaCell.
Least-squares gradients use spherical logarithmic-map displacements in each
cell's local east/north frame. Linear face reconstruction uses the multidimensional
limiter of Barth and Jespersen (1989), *27th Aerospace Sciences Meeting*,
AIAA 89-0366, DOI 10.2514/6.1989-366. This conference publication has no journal.
Primary metadata: https://ntrs.nasa.gov/citations/19890037939 .

Each face belongs to a dyadic cell-size class selected by the smaller adjacent
area/perimeter length. The geographic operator is the sum of these face-class
operators. A palindromic half/full/half composition follows Strang (1968),
*SIAM Journal on Numerical Analysis* 5, 506-517, DOI 10.1137/0705041.
Each class takes its own CFL-limited SSPRK2 substeps, following Shu and Osher
(1988), *Journal of Computational Physics* 77, 439-471,
DOI 10.1016/0021-9991(88)90177-5. Each class updates only its incident cells.
This is an operator-splitting design, not a claim to implement a published
asynchronous flux-register algorithm. Smooth-order and interface errors require
measurement. Limiter activation can reduce local accuracy near extrema.

An outgoing-flux budget scales a donor's fluxes by the same nonnegative factor
on both sides of every face. This algebraic safeguard conserves internal action;
it is not a post-update clipping or a physical source term. The normal CFL is
chosen so this safeguard should be inactive for resolved smooth transport.
One float32 ULP is reserved in its outgoing budget to avoid a negative
roundoff residual at complete drainage; incoming flux uses that same factor.

Directional components are transported between tangent frames, not as scalar
bin numbers. Li (2011), *Monthly Weather Review* 139, 1536-1555,
DOI 10.1175/2010MWR3196.1, explains the failure of that scalar assumption at
high latitude. Li (2012), *Journal of Computational Physics* 231, 8262-8277,
DOI 10.1016/j.jcp.2012.08.007, treats spherical wave propagation and polar
direction references. The implementation here uses parallel transport through
the actual shared edge rather than reproducing an SMC grid or its code.

For source and target unit normals p,q, a source tangent vector v is carried
along their short great-circle arc by v' = v - (v.q)/(1+p.q) (p+q). Its angle
in the target east/north basis defines a cyclic direction-bin shift. Both
cells' bases are transported to the edge. Linear periodic angular weights
are nonnegative and sum to one. The face's signed flux uses the left cell's
direction quadrature; its gain/loss in the other cell uses the transpose of
the same interpolation weights. Donor budgets are formed in each donor's
own bins before their factor is applied to both sides. Thus summing all
directions cancels each internal face, including a face adjoining a pole.
Gradients sample neighbouring spectra in the matching transported direction;
the spatial diffusion tensor and velocity are resolved in the edge frame.
The mesh's stored longitude supplies an ordinary tangent basis at a pole.

Garden-sprinkler dispersion uses the directional projection of the tensor in
Booij and Holthuijsen (1987), *Journal of Computational Physics* 68, 307-326,
DOI 10.1016/0021-9991(87)90060-X:
D_parallel = delta_cg^2 T / 12 and D_perpendicular = cg^2 delta_theta^2 T / 12.
The normal derivative uses the two cell centres; the tangential derivative
uses averaged reconstructed gradients. Their projection supplies the complete
tensor flux, including its cross term. Shared donor limiting handles outgoing
budgets without breaking the internal-face flux balance.
The current common spreading horizon is configurable, initially 3600 seconds.
That horizon is a numerical choice pending coordination with the structured
backend and swell verification, not an observed physical wave age.

Open edges consume incident action only on inflow. A dry neighbor absorbs
outgoing action and supplies none. This follows the core CONTRACT.md resolution
of SPEC's contradictory closed/absorbing-coast wording. No reflecting wall is
claimed. Depth changes retain the prepared wet topology. The optional
`turning_handled=True` backend hook suppresses only the core's analytic
great-circle term, since edge transport already accounts for frame rotation.
Frequency evolution, bathymetric refraction and physical sources remain in
the core. No latitude clipping, masked polar cell or tiny global step is used.

Validation status, 2026-10-30: the native 40,962-cell polar packet check completed
322 steps (26.77 simulated hours), with relative integrated-energy error
6.1922195e-8, no negative/nonfinite action, and 93.9214 percent of the energy
more than two degrees beyond the pole. The centroid lag was 0.552945 degrees;
the packet broadened. This measures operator behavior, not observational skill.
Evidence in the lane delivery: receipts/pole-crossing.json and
pictures/pole-crossing.png; command: tree/tools/waves_polar_proof.py.

The initial coupled global forecast completed 192 hours with 27,914 wet cells,
1,152 atmosphere steps and 2,304 wave calls at 36 by 36 spectral bins. Wave
event spans were 411.8193 ms per atmosphere step, including shared-card stalls.
Tracked wave arrays occupied 55,187.25 bytes per wet cell, of which 42,412.78
belonged to this backend. Hs RMSE was 1.31265 m on 2,832 common buoy pairs,
versus 0.58849 m for the operational reference. Distant swell height was weak.
Coastal sampling corrections and the shared source/transport comparison remain
necessary before attributing that error to this backend or claiming release
qualification. Evidence: pacific-120/forecast/waves/receipt.json,
pacific-120/paper/verification-full-reference/receipt.json and the delivery's
VISUAL-REVIEW.md. These values describe the initial run at engine 07e9159b8fa;
they must not be presented as measurements of later shared-core changes.
