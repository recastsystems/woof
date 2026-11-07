# Native storm intensity products

TC moving runs keep the consensus position for motion decisions. Their reported
intensity comes from the raw minimum surface pressure over native water cells
within 150 km of that position, followed by the maximum 10 m wind within 150 km
of the pressure cell. Native ocean terrain is at sea level. No smoothing or
subcell fitting enters these intensity values. An inland storm with no water
cell in that search retains its motion fix and reports missing intensity.

The controller uses the current member's native land mask. At every successful
diagnostic time it writes:

- `storm-track.fixes.jsonl`: full-precision `gpuwm.tc-field-fix.v1` records,
  including field position, pressure, instantaneous wind and NE/SE/SW/NW
  34/50/64 kt radii. Position-track receipts retain the separate consensus fix.
- `storm-track.atcf`: integer-hour NEQ rows with aid `HEXG`, when the case's
  `storm.storm_id`, `storm.atcf_id` or authored track identifier is an ATCF `BBNNYYYY` identifier.
- `storm-track.swath.f32` and `.json`: a persistent global 0.1-degree geographic
  grid of sampled cell wind maxima. Metadata declares shape, units and spacing;
  NaN means no source cell sampled that bin. This is consultation-time sampling,
  not an every-step maximum or an interpolated wind surface.

These products require the matching woof native TC diagnostics bridge. The
shared Rust reducer handles both structured grids and the complete hex mesh.
The products are diagnostics; they do not establish forecast skill or calibrated
wind probabilities.
