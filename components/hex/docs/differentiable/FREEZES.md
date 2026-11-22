# The freezes

The places where the runner and the generated derivative images deliberately do **not** take the exact derivative
of the forward: what each freeze freezes and where it lives. The holds a row chooses on top of these (radiation and
cloud fraction, the boundary layer's turbulence closure and plume, the mixing length, the lateral-boundary driving data, the 12 km row's
species outputs and cumulus scheme, the 3 km cloud route's riming terms, the mountain-drag closure on the 120 km
rows) are listed per row in [GUIDE.md](GUIDE.md) section 1 ("Scope, by row") and [LIMITS.md](LIMITS.md)
("Held pieces").

## The rule

- A non-differentiable switch whose exact derivative cannot be used (a flux limiter, a closure that blows the
  tangent up) is **frozen from the trajectory**: its coefficients are taken from the forward and treated as passive
  in the tangent and the adjoint. This is the standard operational treatment of non-differentiable limiters.
- A freeze keeps the forward value bit for bit and drops the derivative: the tangent is 0 and the adjoint tape has no
  node there.
- Every comparison still decides on the forward value (the executed branch), and the primal of every entry is
  unchanged (the per-launch identity check shows 0 differing words).
- Each freeze is stated in the window receipt and in the operator's docstring.
- The radiation storage scale at the end is not a freeze: it is an exact rescaling, listed here because it lives in
  the same places.

## Summary

| freeze | switch | where | freezes |
|---|---|---|---|
| FCT limiter | `--fct-frozen 1` (window flag) | wrules.rs `fct_frozen`, window.cu `win_fct_minmax_frozen_{tl,ad}` | fct_scale's scale_in / scale_out; fct_minmax_source's bounds |
| MYNN-EDMF closure | `gen_whole.py --freeze closure` (generator default) | `mw_frz(x)` in the P3 image | stability functions, mixing length, diffusivities, PBL height, plumes |
| GSL drag closure | `p4-gen ugwp ... --freeze=closure` | `p4_frz(x)` in the P4 ugwp image | Ri, N^2, their reference-layer averages, BL N, stress limiters |
| TEMPO closure, mask 0xc6u | `tempo-ad-gen --freeze closure` (`DEFAULT_MASK`) | freeze.rs points in the TEMPO image | number re-derivations and clamps, snow / ice / graupel deposition coefficients |
| radiation clamps | built into the radiation rows | the rows' .ad.cu | fmaxf clamps, radius defaults, gates |
| radiation 2^72 storage scale | built into 4 entries and the P7 slot | rlw_inatm, rlw_setcoef, rsw_inatm_layers_b, rsw_setcoef_b, schemes.rs | none (exact) |

## FCT limiter (`--fct-frozen 1`)

- `fct_scale`: the limiter coefficients get zero tangent and their cotangent is dropped (an exact zero map).
- `fct_minmax_source`: keeps the exact derivative of `source_old` (transported state) and gives the limiter bounds
  none.
- The other FCT kernels run their own entries, so the limiter applies the trajectory's weights linearly.
- Receipt: `"fct_frozen": true` and an `approximation` string.
- Why it is needed: the exact limiter tangent goes non-finite within 4 to 16 steps of an hrrrv5 window; without the
  freeze there is no 18-step operator.

## MYNN-EDMF closure (`gen_whole.py --freeze closure`)

- Frozen: the PBL height and its sigma factors; the level-2 and level-2.5/3 stability functions with the
  counter-gradient terms; the mixing length; the diffusivities and exchange coefficients; every output of the updraft
  and downdraft plumes and the top-down cloud radiation term; the carried el_pbl, sh3d, sm3d and pblh at entry.
- Live: the implicit vertical diffusion with the frozen diffusivities, the TKE and variance prediction, the
  mass-flux transport with the frozen plume fluxes, the subgrid cloud closure and the surface similarity
  (`closure+cloud` also freezes the last two).
- `--freeze none` reproduces the exact-branch output byte for byte.
- The 18-step tangent stays near the held size (0.77x), against 4.9e21x unfrozen.

## GSL drag closure (`p4-gen ugwp ... --freeze=closure`)

- Frozen: Ri and N^2 after the 1e-5 floor and their reference-layer averages (so bnv, brvf and the blocking pe), the
  BL N and the stress limiters; the integer selections never carry a derivative.
- Live: winds, density, terrain statistics, the Froude number through ulow, efact, coefm and the base stress.
- `closure+froude` also freezes the Froude number.

## TEMPO closure, mask 0xc6u (`tempo-ad-gen --freeze closure`)

- Mask 0xc6u = groups 1 (clamp_number), 2 (number_rederive), 6 (snow deposition coefficient) and 7 (ice/graupel
  deposition coefficient). Groups 0, 3, 4 and 5 are selectable; groups 8-23 are diagnostic only.
- Each freeze point is pinned by source line and required text; the generator refuses when a line does not hold its
  text.
- Live: masses, temperature, vapour, the saturation ratios, the process tendencies' explicit dependence on them,
  sedimentation and the depletion limiter.
- Short-window (1-3 step) reflectivity gradients are better with the unfrozen tangent (`--freeze none`).

## Radiation clamps

The radiation rows differentiate the executed branch. Their fmaxf clamps, radius defaults, the condensate gates, the
top layer of the SW post-processing and the night branch of the SW outputs carry no derivative.

## The 2^72 molecule-unit storage scale (exact)

At small adjoint scales most cotangents of the molecule-unit words (coldry, wbrodl/wbroad, wkl) were float32
subnormal. Their shadows hold t·2^-72 and b·2^72, applied in double before the single float32 rounding, so the scale
is exact. Per-launch FD harnesses see those four entries' molecule-unit outputs in the 2^72 unit.
