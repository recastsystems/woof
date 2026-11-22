"""Correct-mode fixes of defects in the pinned RRFS fork schemes (``physics.mode = 'correct'``, the default).

Each row is one defect that hex copies faithfully from the pinned fork source in exact mode, confirmed against
the upstream fix named in ``upstream``.  Exact mode never reads this table: the exact CUDA sources stay the pinned
transcription byte for byte, so exact mode is still bit for bit with native.

A row is applied when correct mode runs its slot with that scheme on hex-cuda and the slot implementation declares
the id (the suite table's ``cuda_entries[...]["corrections"]``).  The scheme module carries the source change
(``tempo_v303.corrected_sources``, ``tempo_v325.corrected_column``, ``bl_mynnedmf_*.module_cuda``).  A row whose
slot runs the native-cpu fork library is a named gap (the pinned Fortran carries no correction), printed at
start-up and in the receipt.  ``physics.correct_skip`` takes these ids like the v8.4.2 ones.

Scheme versions.  Correct mode's defaults are the latest upstream releases (TEMPO v3.2.5; on the 44c0849 pin set
MYNN-EDMF v6.1.1 and MYNN-SFC v6.1.0), and a release already carries every upstream fix older than its tag.  A
row's ``carried_by`` names the version modules whose own text has the fix: when the slot runs one of them the row
is neither applied nor a gap (``carried`` lists it for the receipt).  So a text pair is applied only where the
defect is still in the running source: the pinned schemes (TEMPO v3.0.3, MYNN-EDMF c6d08709 / c942418, MYNN-SFC
v6.0.0), which correct mode runs on the older pin set and wherever a run names the pins, and TEMPO v3.2.5 for
TP4 and TP5, whose upstream fixes are later than the tag.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

#: Owner of every row: the hex RRFS fork route (this table and the scheme modules that carry the changes).
OWNER = "hex RRFS fork corrections (woof.hex.mpasphys.rrfs.fork_corrections)"

FORK_CORRECTIONS: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "MY1": {
        "slot": "pbl", "schemes": ("bl_mynnedmf",),
        "carried_by": ("rrfs.bl_mynnedmf_v611",),
        "upstream": "NCAR/MYNN-EDMF f3c9d9e (2026-11-04, 'bugfix for allowing negative moisture surface fluxes')",
        "what": "the surface vapour-flux limiter keeps downward fluxes (dew and frost), bounded so the lowest level "
                "keeps 10 % of its vapour: max(qvflux, min(1e-8 - 0.9 sqv, 0) / (dtz rhosfc / rho))",
        "breakage": "min(0.9 sqv - 1e-8, 0) / dtz is 0 whenever sqv > 1.1e-8 kg/kg, so max(qvflux, 0) deletes every "
                    "downward surface moisture flux: dew and frost never take vapour out of the air"},
    "MY2": {
        "slot": "pbl", "schemes": ("bl_mynnedmf",),
        "carried_by": ("rrfs.bl_mynnedmf_v611",),
        "upstream": "NCAR/MYNN-EDMF eb5029c (2026-11-04, 'bug fix of rhosfc')",
        "what": "the surface air density is psfc / (R_d T (1 + 0.608 qv)), the virtual temperature",
        "breakage": "rhosfc = psfc / (R_d (T + 0.608 qv)) adds 0.608 qv kelvin instead of scaling T, so the surface "
                    "density that multiplies every surface flux into the lowest level is about 0.608 qv too high "
                    "(1 % at 16 g/kg)"},
    "MY3": {
        "slot": "pbl", "schemes": ("bl_mynnedmf",),
        "carried_by": ("rrfs.bl_mynnedmf_v611",),
        "upstream": "NCAR/MYNN-EDMF e447691 (in v6.1.0: the theta tendency of the ice branch from the mixed condensate)",
        "what": "with ice on, dth = (thl + xlvcp/exner sqc2 + xlscp/exner sqi2 - th) / delt: the cloud water and ice "
                "after mixing, as the liquid-only branch and the condensate tendencies dqc and dqi use",
        "breakage": "the ice branch took sqc and sqi from before mixing, so the latent heat of the condensate the "
                    "scheme itself mixes and condenses never reached theta while its water did: the heat and water "
                    "tendencies of one step disagree wherever the boundary layer holds cloud"},
    "MY4": {
        "slot": "pbl", "schemes": ("bl_mynnedmf",),
        "carried_by": ("rrfs.bl_mynnedmf_v611",),
        "upstream": "NCAR/MYNN-EDMF 87af8c6 (the unified driver: condensate converted back with 1/(1 - sqv))",
        "what": "the driver returns the mixed cloud water, ice and snow to mixing ratios as sq / (1 - sqv), the "
                "inverse of its own sq = q / (1 + qv)",
        "breakage": "sq (1 + sqv) is not that inverse (1 + qv = 1 / (1 - sqv)), so every PBL call returned each "
                    "condensate about sqv^2 short: 1.25e-4 of a column's cloud water a call, up to 3.0e-4 in a "
                    "cell, about 1.8 per cent over 59 steps for a cell that keeps its cloud (model audit D19)"},
    "MY5": {
        "slot": "pbl", "schemes": ("bl_mynnedmf",),
        "carried_by": ("rrfs.bl_mynnedmf_v611",),
        "upstream": "NCAR/MYNN-EDMF 27da0f0 (in v6.1.0: PBL-height search start and the surface potential temperature)",
        "what": "th_sfc = ts / exnerg (the surface Exner function), and the second PBL-height search starts at max(kthv + 1, kts + 1), the level above the minimum virtual temperature",
        "breakage": "the second search ran from kts + 1 whatever kthv was (its k = kthv + 1 was overwritten by the loop head), and th_sfc used the lowest level's Exner function. Small and local on the 5 March 2022 12 km start: at the first call the PBL height changes in 2 of 13,808 columns (at most 60 m); the 25 columns under 30 m are under 30 m with and without it"},
    "SF1": {
        "slot": "surface_layer", "schemes": ("sf_mynnsfclay",),
        "carried_by": ("rrfs.sf_mynnsfc_v610",),
        "upstream": "NCAR/MYNN-SFC f0c1839 (2026-10-29, 'fix the cold start condition check')",
        "what": "the step-1 cold-start reset (pblh from ust, hfx, qfx and mol to zero) runs only on a cold start: "
                "itimestep == 1 and not (restart or DA cycling)",
        "breakage": "(.not. restart .or. .not. cycling) is true unless both are set, so a restart or a DA-cycled "
                    "start wipes the surface heat and moisture fluxes, the Obukhov scale and the PBL height it was "
                    "handed at its first step"},
    "TP1": {
        "slot": "microphysics", "schemes": ("mp_tempo",),
        "carried_by": ("rrfs.tempo_v325",),
        "upstream": "NCAR/TEMPO 881614f (TEMPO v3.0.5: RETURN -> CYCLE in cloud_condensation)",
        "what": "cloud condensation skips an exactly saturated level (|ssatw| < eps) and goes on up the column",
        "breakage": "the first exactly saturated level RETURNs from cloud_condensation, so no level above it "
                    "condenses or evaporates cloud water in that substep"},
    "TP2": {
        "slot": "microphysics", "schemes": ("mp_tempo",),
        "carried_by": ("rrfs.tempo_v325",),
        "upstream": "NCAR/TEMPO 92aafb9 (#42, 'Bugfixes for graupel density prefactor')",
        "what": "graupel volume from rain-ice collection is meters3_to_liters * prg_rci / rho_i, as every other "
                "graupel volume source",
        "breakage": "pbg_rci lacks the meters3_to_liters factor, so graupel made by rain-ice collection gets a "
                    "thousandth of its volume (its density pinned at the clamp)"},
    "TP4": {
        "slot": "microphysics", "schemes": ("mp_tempo",),
        "carried_by": (),
        "upstream": "NCAR/TEMPO 71588d1 (#62, 'Add additional depletion checks to tempo')",
        "what": "the cloud-water over-depletion limiter scales the riming graupel volume terms (pbg_scw, pbg_gcw) by "
                "the same ratio as their mass terms",
        "breakage": "the limiter cuts the riming mass (prg_scw, prg_gcw) and leaves its volume, so a cloud-limited "
                    "riming step adds graupel volume for mass it never adds (graupel density pulled down)"},
    "TP3": {
        "slot": "microphysics", "schemes": ("mp_tempo",),
        "carried_by": ("rrfs.tempo_v325",),
        "upstream": "NCAR/TEMPO 4b40a93 (#38, TEMPO v3.1.1, 'Bugfix for graupel and aerosol initialization')",
        "what": "the first-call initialisation (graupel number and volume from graupel mass, absent aerosols) runs "
                "in every column of the first call",
        "breakage": "first_call_main is a SAVE logical cleared after the first column, so only column 1 of each "
                    "rank gets graupel number and volume; every other column starts graupel at zero number and "
                    "volume"},
    "TP5": {
        "slot": "microphysics", "schemes": ("mp_tempo",),
        "carried_by": (),
        "upstream": "NCAR/TEMPO ace0eae (#59, 2026-11-13, 'Fix flipped snow moment calculation')",
        "what": "snow_moments takes the mass moment as the second moment when bm_s = 2 (TEMPO's value) and runs "
                "the temperature fit only for another bm_s, as upstream now does.  Not carried: upstream says the "
                "fix needs a new rain-snow collection table; the table here is still the pinned build's, made with "
                "the flipped moments",
        "breakage": "the two branches are the wrong way round, so with bm_s = 2 every snow moment (number, the "
                    "fall, riming, deposition and reflectivity moments) is built on a fitted second moment "
                    "(smob / a)^(1 / b) in place of smob itself"},
    "TP6": {
        "slot": "microphysics", "schemes": ("mp_tempo",),
        "carried_by": (),
        "upstream": "ufs-community/ccpp-physics Thompson module_mp_thompson.F90 (the lineage of TEMPO's activation): "
                    "activ_ncloud(temp(k), w1d(k)+rand3, nwfa(k), lsml) passes the land-sea mask; NCAR/TEMPO v3.2.5 "
                    "and main call activate_cloud_number(temp(k), w1d(k), nwfa(k)) with no land in cloud_condensation",
        "what": "droplet activation takes the cell's land mask (MPAS xland: 1 land, 2 water or sea ice), so TEMPO's own "
                "floor on the activated fraction applies: 0.15 of the aerosol number over water and sea ice, 0 over "
                "land (activate_cloud_number's lower_lim_nuc_frac)",
        "breakage": "with land absent lower_lim_nuc_frac stays 0 and the water floor never runs: where the resolved "
                    "vertical velocity sits at the activation table's floor (0.011 m/s, coarse cells) marine "
                    "stratocumulus activates about 21 droplets per cm3 from about 600 aerosol per cm3 and drizzles, a "
                    "false light-echo sheet over the oceans (18 h zoom-globe run of 19 Nov 2026: 29.7 per cent of the "
                    "cells off California at 5 dBZ or more at 18 h, 11.4 per cent with the floor)"},
})


#: The module each ``config_tempo_version`` runs on hex-cuda (tempo_v303.TEMPO_VERSIONS).
TEMPO_MODULES: Mapping[str, str] = MappingProxyType({"v3.0.3": "rrfs.tempo_v303", "v3.2.5": "rrfs.tempo_v325"})


def _state(finding, row, selectors, backends, hex_backend, off_backend, skip, running):
    """'applied', 'gap', 'carried' or None for one row in one resolved configuration."""
    slot = row["slot"]
    scheme = selectors.get(slot)
    if finding in skip or scheme not in row["schemes"] or backends.get(slot) in (None, off_backend):
        return None
    if backends[slot] != hex_backend:
        return "gap"
    return "carried" if (running or {}).get(slot) in row["carried_by"] else "applied"


def active(selectors: Mapping[str, str], backends: Mapping[str, Any], hex_backend: Any,
           off_backend: Any, skip, running: Mapping[str, str] | None = None
           ) -> tuple[frozenset[str], tuple[tuple[str, str, str, Any], ...]]:
    """(applied ids, gaps) for one resolved fork configuration in correct mode.

    A row is applied when its slot runs one of its schemes on hex-cuda in a source that still has the defect;
    on another live backend it is a gap (finding, slot, scheme, backend).  ``running`` maps a slot to the
    version module it runs (the MYNN version overlay, ``TEMPO_MODULES``); a row that module carries is neither
    (``carried``).  Skipped ids are neither.  With no ``running`` every slot runs its pinned scheme.
    """
    applied, gaps = set(), []
    for finding, row in FORK_CORRECTIONS.items():
        state = _state(finding, row, selectors, backends, hex_backend, off_backend, skip, running)
        if state == "applied":
            applied.add(finding)
        elif state == "gap":
            slot = row["slot"]
            gaps.append((finding, slot, selectors.get(slot), backends[slot]))
    return frozenset(applied), tuple(gaps)


def carried(selectors: Mapping[str, str], backends: Mapping[str, Any], hex_backend: Any,
            off_backend: Any, running: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    """(finding, module) for each row the running scheme version already carries in its own text."""
    return tuple((finding, running[row["slot"]]) for finding, row in FORK_CORRECTIONS.items()
                 if _state(finding, row, selectors, backends, hex_backend, off_backend, frozenset(), running)
                 == "carried")
