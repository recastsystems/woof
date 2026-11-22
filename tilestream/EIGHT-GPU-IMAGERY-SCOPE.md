# The 8-GPU box: one or two rendered timesteps, not a forecast

The owner has scoped phase 2 down. He wants **one or two plot timesteps at the
biggest domain the box can run** (proof that the machine can hold and integrate
something enormous) and then the box gets destroyed. He is paying by the hour
and is winding rentals down.

So do NOT queue a multi-hour forecast. Budget roughly **30-60 minutes of
integration**, then render, then stop. If a configuration cannot produce a frame
in that window, make the domain smaller rather than the run longer: the point of
the image is the domain size, and a 2,000-cell-per-side domain at one valid time
proves it better than a small domain at twenty.

## Initialise from HRRR, not ERA5, and this is the whole trick

A 1-2 frame deliverable cannot afford spin-up, and the two analyses behave
completely differently:

* **ERA5 is ~31 km.** Nothing convective exists in it. Storms have to grow from
  the mesoscale environment, which takes **hours** of model time: the 2011 case
  running elsewhere needed t+1.5 h before reflectivity was worth looking at, and
  its best frames are at t+3 h. Wrong tool for two frames.
* **HRRR is already 3 km and convection-allowing.** Its analysis *contains
  storms*. Initialise from an HRRR analysis at a time when the event is already
  under way and the very first output frame has organised convection in it,
  because the convection came in with the initial condition and the model only
  has to keep it accurate.

So pick an HRRR-era case at its peak, 2021-12-10 (the Quad-State derecho, a
spectacular long-lived QLCS, nocturnal so it is unmistakable) or 2019-05-28
(Plains outbreak), and initialise ON the event rather than before it.

Label it accurately: a short-lead run from a convection-allowing analysis is a
legitimate and common configuration, but the caption must say the lead time so
nobody reads a 30-minute forecast as a 12-hour one.

## What the frames must carry

Use WOOF-ArWen's own renderer (`woof render --engine rust`; see `RENDERING.md`: do
not hand-roll matplotlib). Composite reflectivity is the primary field; updraft
helicity over it is the supercell signature.

State on the figure or in a README beside it: domain in cells and km, dx, dt, the
analysis and its valid time, the lead time, how many GPUs, the wall clock, and
(the actual point) **what a single card does with the same configuration**.
Attempt the single-card run and record the failure verbatim. That contrast is the
deliverable; the pretty picture is the vehicle.

## Check for seams before you ship it

The domain is decomposed across eight cards. Look for straight-line
discontinuities on sub-domain boundaries, in reflectivity AND in a smooth field
such as surface pressure where a seam is far easier to see. A visible seam in a
marketing image would be both a bug and an embarrassment.
