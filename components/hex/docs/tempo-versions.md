# TEMPO versions on the RRFS physics route

The RRFS microphysics slot (`config_microp_scheme = 'mp_tempo'`: hrrrv5 and every TEMPO suite) runs one
of two TEMPO versions, named in `&physics_mp_tempo`:

| `config_tempo_version` | TEMPO | Use |
|---|---|---|
| `'v3.2.5'` (default) | NCAR/TEMPO `tempo_v3.2.5` (c62699fb660e, 2026-10-06 release) | every correct-mode run (the door default) that does not name a version |
| `'v3.0.3'` | NCAR/TEMPO 9adf9ef, the version both RRFS pins (1d6e370e, 44c0849) carry | the pinned referee: every `--physics-mode exact` run that names no version (the bitwise RRFS results and the RRFS landing gate), or any run that names it |

Exact mode reproduces the route's pinned native fork bit for bit, so it takes the pin's TEMPO unless the namelist
names a version (the route table's `exact_mode_defaults`); a native deck's own namelist, which has no
`config_tempo_version`, therefore runs v3.0.3 against native in exact mode.

Any other value is refused by name. `config_tempo_hailhyperaware` (default `.true.`) switches v3.2.5's hail
category; it is off whenever `config_tempo_hailaware` is off, as in TEMPO itself. With `'v3.0.3'` no line of
the v3.2.5 module is compiled or run, and the v3.0.3 kernels are the 0.4.15 ones byte for byte.

```text
&physics_mp_tempo
    config_tempo_version = 'v3.0.3'
/
```

or, on the forecast door, `--physics config_tempo_version=v3.0.3`.

## What v3.2.5 changes

- A hail category (hail-hyperaware): each call diagnoses a hail fraction from dense, large graupel, gives the hail
  its own size distribution, fall speed, sedimentation, riming, rain collection, sublimation, melting, depletion
  limits, reflectivity and precipitation, and merges it back into graupel at the end of the call (surface graupel
  precipitation and the hail-size diagnostic include it).
- Cloud ice and ice number sediment in every substep; v3.0.3 fell them once with a 1/substeps flux, so ice fell
  that many times too slowly at long time steps.
- Koop aqueous freezing reads the supersaturation the caller passes (v3.0.3 was off by one in water activity);
  nucleated ice now leaves the ice-friendly aerosol; the nucleation cap counts snow number.
- The graupel volume from rain-ice collection carries its missing factor of 1000; frozen-drop volumes use the
  densest graupel class; frozen drops raise the graupel density when they dominate its growth.
- Cloud condensation skips a level at exactly 100 % humidity instead of stopping for the rest of the column.
- Rain and graupel fall speeds stay fixed through the sedimentation substeps; the graupel fall-speed coefficient
  and melting ventilation, the melting number loss and the riming efficiency above freezing follow upstream.
- First-step initialisation (aerosol profiles, graupel number and volume) applies to every column of the first
  step, not only the first column of each process.

The lookup tables are the same files for both versions: rebuilt at `tempo_v3.2.5` with the strict recipe they are
byte-identical to the v3.0.3 rebuild.

One upstream read of an unset variable (`final_weight`, main:1755, where main:1752 did not set it) is defined
here as 1, no scaling; the native code the port is replayed against carries the same definition, labelled patched.

The start-frame reflectivity a correct-mode TEMPO forecast writes into its f000 frame (`tempo_refl_v303`) follows
the run's version: with v3.2.5 it splits the start state's graupel into hail as the v3.2.5 step does and adds the
hail term. Fast numerics runs v3.2.5 as the (level, cell) phase kernels plus the warp tail
(`tempo_phase_v325`), the same layout as v3.0.3's.

A run whose TEMPO is not the RRFS pin writes `hex_tempo_version` and `hex_tempo_hailhyperaware` as global
attributes on every stream it writes; a pinned (v3.0.3) run writes neither, so exact-mode headers stay native's.
`python -m woof.hex.mpasphys.rrfs.tempo_refl_v303 FILE` (and `rediagnose_file`) takes the version from those
attributes when `--tempo-version` is not given; a file without them is read as v3.0.3, and the tool prints which
version it used.
