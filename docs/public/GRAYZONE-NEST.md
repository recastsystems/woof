# The gray-zone parent chain: shipped with findings, not blessed

`configs/les_nest_250m_grayzone.toml` is the shipped nested-LES tree
(`configs/les_nest_250m_km3.toml`, measured in
[LES.md](LES.md) §2 "Nested, on a real case") with exactly one resolved
field changed: the 750 m parent runs Shin-Hong
(`bl_pbl_physics = 11`) instead of YSU. `tests/test_grayzone_nest_config.py`
pins that the two configs resolve identically everywhere else, so any
difference between the two runs is the parent closure and nothing else.

**The configuration is not blessed.** The P4 interface campaign scored it
against the instrument registered before any SH-parent run existed, two of
the four screens read outside their registered bands, and under the
registered rule (AC-P4.2: any screen outside its band is a finding that
blocks the blessing, not a number to re-band) those findings block the
blessing. The findings section below is the status of record; the numbers
live in
`docs/superpowers/receipts/grayzone/P4-CAMPAIGN-20260804.md`.

## Findings (the campaign's verdict on this configuration)

- **Finding 1: the partition handoff read on the flat-baseline side at
  every scored frame, by a factor of ~3.** d02's measured mixed-layer
  subgrid TKE fraction (0.16–0.22) sits far below both the registered
  direction floor and the Honnert band the idealized ladder put Shin-Hong
  inside at the same scale (~0.83 at 800 m). The committed data also
  shows d02's own mid-CBL resolved variance roughly doubled against the
  YSU baseline over the same footprint: the opposite sign of the ladder.
  Two candidate readings are on the record, unadjudicated: the instrument
  transplant (Honnert's decomposition over a real terrain-bearing
  footprint counts mesoscale variance the idealized ladder box never
  contained) or genuinely different real-case behaviour of the scheme at
  750 m. Discriminating between them is an owner adjudication.
- **Finding 2: the rim/far-field w-variance ratio out by 2.6 % at one
  frame (22Z)**, inside its ceiling at the other three. 22Z is also the
  frame the D90 receipt's parent control failed flatness on and the YSU
  baseline's own highest rim frame; recorded as context, not as an
  excuse.
- **In-band, both directions reported:** the spectral overlay sits inside
  its ceiling at every frame and tighter than the YSU baseline (the
  child inherits its large scales through SH-parent boundaries at least
  as cleanly as through YSU's) and the far-field statistics show no
  degradation (four-frame mean delta −0.0003 against a −0.033 floor).
  The nest interface itself is healthy; what moved is the parent's own
  gray-zone partition.

The campaign's controls were clean: the YSU baseline rerun reproduced the
committed D90 run A byte for byte, and the SH-parent determinism pair was
bitwise identical (the no-ECC corruption screen).

## The per-domain reading, and why each domain is what it is

| domain | dx | closure | why |
|---|---|---|---|
| d01 | 3 km | **YSU, km_opt 4** (default) | The acceptance ladder's 3200 m rung was advisory-only, never gated: Shin-Hong's own boundary-layer depth shallows there and the rung sits on block-degenerate ground. Switching the coarse parent to 11 is an **owner option, not the default**. |
| d02 | 750 m | **Shin-Hong 11, km_opt 4** | 750 m is inside the gray zone (dx between ~2 km and the child). This is exactly the `sweep_config(dx, 11)` configuration the ladder scored (Shin-Hong vertical transport, horizontal Smagorinsky) and every gated rung 1600/800/400/200/100 m landed inside its pre-registered Honnert (2011) band ([receipt](receipts/grayzone/PHASE1-SHINHONG-20260803.md)). At d02's scale the ladder read subgrid TKE fraction ≈ 0.83 (800 m rung) where every flat closure read ≈ 0.35, and Finding 1 above is that the real case did not reproduce this. |
| d03 | 250 m | **PBL off, km_opt 3** | Unchanged from the shipped LES child block, byte for byte: `c_s = 0.25`, `mix_isotropic = 1`, `mix_upper_bound = 0.1`, `isfflx = 1`. |

> **`mix_isotropic` moved from 0 to 1 on 2026-08-09, in both files at
> once.** At 0 the 250 m child sat at `mix_upper_bound·(dz_max/dx)² =
> 0.702`, 2.8× the explicit horizontal diffusion limit: the same
> criterion that read 4.23 on a 100 m tornado child and aborted that run
> ([LES.md](LES.md) §4). **Every number on this page and in the P4
> receipts was measured at `mix_isotropic = 0`**; that configuration is
> archived at `configs/frozen/les_nest_250m_grayzone.toml` and
> `configs/frozen/les_nest_250m_km3.toml`, whose as-run digests
> `d3eb6c70c0…` and `2a3a279b6d…` are what the receipts name. Both files
> gained a radiation declaration at 1.8.8 so they would keep loading, which
> moved their own digests and no physics selector;
> `configs/frozen/README.md` carries both numbers and what a reproduction
> passes. The
> shipped pair has not been re-scored, and the change does not touch the
> findings below: it is applied identically to both files, so the two
> trees still differ by exactly the d02 `bl_pbl_physics` row and the
> comparison the campaign made is still a comparison of one row.

The per-domain reading was cut from the idealized ladder: a parent whose
dx lies between ~2 km and its LES child selects `bl_pbl_physics = 11`
with `km_opt = 4`; a parent at or above ~3 km stays YSU unless the owner
elects otherwise; the LES child stays PBL-off with its LES closure.
Whether that rule survives on real cases is exactly what Finding 1 puts
to the adjudication.

## What Shin-Hong on d02 hands the child, and what stays on d02

- **A `km_opt=2` LES child runs under a Shin-Hong parent.** A
  `km_opt=2` child cold-starts its own TKE under any parent
  ([LES.md](LES.md) §4); only a `km_opt=2` parent adds a
  not-yet-verified warning. A Shin-Hong parent is `km_opt = 4`: its
  e_sgs is a published diagnostic, not the closure's prognostic
  carrier, so a `km_opt=2` child under it runs on the same terms as
  under any `km_opt=4` parent.
- **The scheme's subgrid TKE stays on d02.** Shin-Hong computes its
  own SGS TKE every step and publishes it as `state.e_sgs`
  (written to its wrfout as `TKE_SHINHONG`). Like WRF, where `tke` has
  no nest-interpolation (`i`) and no feedback (`f`) Registry flag, that
  field stays on its own domain. The child cold-starts whatever closure
  energy it carries, exactly as it does under a YSU parent.

## Bounds inherited from the shipped tree

Every bound in [LES.md](LES.md) §4 applies unchanged: the 49-level shared
vertical grid (this is coarse LES at the gray-zone edge; vertical
resolution, not dx, is the binding constraint), the one-way nesting, and
the fetch/spin-up finding, the D90 measurement on the shipped case
(development branch, `INFLOW-FETCH-D90-2026-08-03.md`) found the 250 m
child's inflow spin-up exceeds most of its own domain, so whole-domain
child scores on either parent chain are fetch-weighted averages until the
inflow generator lands.

## Status

The configuration ships as a configuration: importable, per-domain,
refused nowhere it should not be, and pinned against the shipped tree by
`tests/test_grayzone_nest_config.py`. It ships **with the findings
above, unblessed**: the registered interface instrument
(`docs/superpowers/receipts/grayzone/P4-INTERFACE-REGISTRATION-20260804.md`)
scored it, two screens read out of band, and the blessing waits on the
owner's adjudication of Finding 1. No band was widened and no number was
softened to say otherwise.
