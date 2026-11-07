# Coupled validation inputs

The first case uses a fixed 4 km Lambert parent, 384 x 320 x 49, centered at
25 N, 87 W, from 2024-10-07 12 UTC through 2024-10-10 06 UTC. It covers the
eastern Gulf, the initial center and the Florida crossing. The explicit
convection configuration uses no cumulus parameterization on the 4 km grid.
All other listed physics remain enabled in both arms. The selected surface
option is the base engine's isftcflx=2; a separate later high-wind exchange
implementation is not silently assumed to be present in this base.

The normal wave door defaults to integer coarsening 2, giving 192 x 160 cells
at nominal 8 km for these cases, 36 x 36 bins and a 300 s coupling interval.
The completed 66-hour ON run measured 3.675431 wave device ms per atmosphere
step, 3.247% of its inclusive step, with 25361 wet cells on the shared RTX
5090. This is a measured resource choice, not a resolution optimum. Output is hourly for buoy series and full-field hashes;
the map sequence uses six-hourly frames.

Run both scored arms from the same prepared atmosphere:

    woof go milton.toml --waves on --waves-options '{"roughness_feedback":true}'
    woof go milton.toml --waves on --waves-options '{"roughness_feedback":false}'

The run options control only wave feedback. Neither arm starts waves from
calm water. Automatic wave preparation fetches one initialization cycle's
forecast partitions and records its cycle and hashes. Scores against buoys
and the operational reference must use matching valid times and locations.

The separate full-history waves-off reference is authorized as the proof's
unscored reference. Run it at the same size and compare every atmosphere field
with feedback OFF. The qualified proof compares 13,635,873,963 typed words
across all 67 hourly frames with zero differences. Retain all raw output
behind the fold candidate until acceptance, with the lane retention marker.
The delivery packet records commands, binary hashes and complete receipts.

Roughness feedback defaults OFF. Both full ON trajectories pass, but feedback
raises hourly common-pair Hs RMSE from 0.599941 to 0.726488 m in the first
case (341 pairs) and from 0.492942 to 0.516138 m in the second (712 pairs).
Tp and wrapped peak-direction RMSE also worsen in both hourly comparisons.
The first-case wind and track errors improve slightly with feedback, while
pressure error does not; the initial intensity deficit remains large. The
actual surface stress diagnostic confirms the implemented high-wind drag
decline reaches the atmosphere. Feedback ON remains explicitly selectable
for experiments. This choice is based on skill evidence, with all physics
otherwise preserved. Native receipts and reviewed figures are in the lane's
paper evidence under each coupled case's SCORES.md and atmosphere folder.
