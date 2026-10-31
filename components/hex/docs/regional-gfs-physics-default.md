# Regional GFS cold-start physics

The regional GFS recipe selects `--physics-mode correct --physics-profile hrrr-like-aerosol` on the MPAS route, with `convection_permitting` as its base suite. The profile enables corrected boundary-layer cloud coupling and aerosol-aware Thompson. It retains native MYNN settings.

The three first-match rows in `data/profiles/physics-default.v1.json` are:

| Row | Applies to |
| --- | --- |
| `regional-gfs-cold-zoom-globe` | Generated global meshes with regional refinement, initialized from GFS |
| `regional-gfs-cold-limited-area` | Cold limited-area regions driven by GFS, excluding kept hex parents |
| `regional-gfs-cold-storm-following` | GFS-initialized moving families and their fixed controls |

Each row requires source `gfs` or `gfs-gdex` and a cold start. Quasi-uniform global requests, other sources, warm parents and standalone forecasts keep their existing defaults. The source door must still support the requested source cadence and coverage.

The promotion follows three regional warm-season cases: Vermont 2023-07-08, Plains 2024-05-06 and Southeast 2023-06-14. The aerosol profile improved FSS in 27/27 comparisons, CSI in 8/9 and area error in 7/9. This is evidence for the scoped recipe, not HRRR equivalence or validation of global or cycling defaults.

Cold initialization forwards the profile to the existing verified monthly WIF initializer. It prepares `nc`, `nifa` and `nwfa`, records the profile and WIF hash in the init provenance, and refuses an unavailable climatology. Cached regional and moving cold inputs without that profile preparation are rebuilt. Native soil donors remain separate from meteorology decoding and reach the existing categorical relative-wetness remap. Later limited-area boundary hours do not request an analysis-only land donor. The current donor table supports analysis-time cold initialization only.

Explicit physics selections take precedence. `--physics-suite convection_permitting` keeps the native suite; `--physics-mode exact` keeps the native proof configuration. Their argument lists, namelists and configuration receipts are tested against the pre-promotion state. Moving-run overrides also supersede implicit case defaults. The optional `hrrr-like` profile with additional MYNN changes remains separately selectable.

For a high-level door, pass these choices through `--forecast-arg`, for example `--forecast-arg=--physics-mode=exact`. For a moving case, put them in `physics_args` or pass the moving run's forecast arguments. Runtime physics receipts name `hex_physics_profile`; regional and zoom plans also name the selected profile.

The categorical cold-soil remap ships in the same release.


Landing and product synchronization: the WOOF hex form, Live Setups and API defaults must change together with the hex release. Each must select `correct` plus `hrrr-like-aerosol` only for the three regional GFS cold-start rows listed above. Preserve explicit native and exact overrides, and leave warm/global requests outside this qualification. The release lead owns coordinating that rollout; this branch does not itself update those applications.

Retained regressions are named follow-ups, all owned by the **hex physics lead**:

| Follow-up | Retained regression | Required follow-up evidence |
| --- | --- | --- |
| `HEX-PHYS-VERMONT-40-CSI` | Vermont 40 dBZ CSI falls from 0.004 to 0 | Diagnose missed strong echoes against the retained three-case score and sheets; document any fix on the same scoring window. |
| `HEX-PHYS-PLAINS-ONSET` | Plains strong-echo onset moves from 17Z to 18Z | Diagnose the one-hour initiation delay and verify onset after a proposed physics change. |
| `HEX-PHYS-18Z-LH-BIAS` | Plains and Southeast retain 18Z latent-heat bias | Diagnose land/cloud contributions against the retained 18Z flux table and document a corrected flux comparison. |

These are retained limitations, not passing gates or completed fixes. Source: the three-case evaluation table and its verdict. The same-card performance landing receipt is recorded in `regional-gfs-landing-receipt.md`.
