"""Which WRF release WOOF-ArWen is pinned against: ``wrf_reference``.

WOOF-ArWen's bitwise referee moved from WRF 4.7.1 to WRF v4.8.0 (released
2026-06-08).  4.7.1 stays selectable as the previous pin.  This module is
the one authority for that choice: data plus resolvers, importing nothing
heavy at module scope.

* ``wrf_reference`` is one user-facing key: ``[shared]`` of an experiment
  TOML, ``[run]`` of a legacy RunConfig TOML, ``woof import-namelist
  --wrf-reference`` and the WOOF/API field that writes the TOML key.
* It is resolved ONCE, when the configuration is loaded, into the
  per-scheme generation keys (``solar_generation``, ``ysu_version`` ...)
  that the named WRF release runs.  Those keys already bind the restart
  and run identities, so the selector adds no field of its own to
  :class:`woof.config.RunConfig`, and an omitted selector changes nothing.
* A per-scheme key written beside it still wins, as ``bl_mynn_version``
  does today.
* Omitted, the engine runs its defaults: the newest generation of every
  scheme (decided 2026-11-08: defaults are the latest upstream versions,
  pins are for bitwise proof).  ``"4.8.0"`` pins the schemes WRF 4.8.0
  itself ships, ``"4.7.1"`` the ones WRF 4.7.1 ships.  Which generation
  each release runs is the importer's release table
  (:data:`woof.namelist_import.WRF_RELEASE_GENERATIONS`): one table for
  the namelist door and the run door, so the two cannot disagree.
* The namelist vocabulary follows the same choice: the 4.8.0 Registry
  table for a 4.8 release (and for a bare ``4``, the newest), the 4.7.1
  table for 4.7.1, every older release and the WRF 3 line
  (:func:`registry_reference_for_wrf_version`).

The ports that change words (GFL, MYNN-EDMF 8f8e8b5, MYNN-SFC, KIM
drag, Shin-Hong, Fitch induction, TEMPO, lightning) each add their row to
``WRF_RELEASE_GENERATIONS`` and move their keys out of
:data:`WRF480_ONLY_KEYS` when the consumer lands.
"""
from __future__ import annotations

from typing import MutableMapping

#: The WRF releases ArWen can be pinned against, newest first.
WRF_REFERENCES = ("4.8.0", "4.7.1")

#: The Registry vocabulary read when nothing names a release.
WRF_REFERENCE_DEFAULT = "4.8.0"

#: The tags and commits the packaged Registry tables were extracted from.
WRF_REFERENCE_SOURCES = {
    "4.8.0": {"tag": "v4.8.0",
              "commit": "06d4240ae989cc3e50af412bb472df3d9048783c"},
    "4.7.1": {"tag": "v4.7.1",
              "commit": "f52c197ed39d12e087d02c50f412d90d418f6186"},
}

#: The key's spelling in every door.
WRF_REFERENCE_KEY = "wrf_reference"


def check_wrf_reference(value, *, where: str = WRF_REFERENCE_KEY) -> str:
    """``value`` as a known reference, or a refusal naming the choices."""
    text = value.strip().lstrip("vV") if isinstance(value, str) else None
    if text not in WRF_REFERENCES:
        raise ValueError(
            f"{where} = {value!r}: name a WRF release WOOF-ArWen is pinned "
            f"against, one of {list(WRF_REFERENCES)} (omit it to run the "
            "engine defaults, the newest generation of every scheme)")
    return text


def registry_reference_for_wrf_version(wrf_version) -> str:
    """The Registry table a namelist written for ``wrf_version`` is read with.

    A dotted release before 4.8 (4.6.1, 4.7, 4.7.1) and the WRF 3 line
    read the 4.7.1 table, the one every import used before 4.8.0 was
    packaged; a release at or after 4.8 and a bare ``4`` read the 4.8.0
    table.
    """
    from woof.namelist_import import wrf_line_of, wrf_release_of

    release = wrf_release_of(wrf_version)
    if release is None:
        return "4.7.1" if wrf_line_of(wrf_version) == "3" else "4.8.0"
    return "4.8.0" if release[:2] >= (4, 8) else "4.7.1"


#: Selectors whose ENGINE default is deliberately not the code stock WRF
#: runs (a science default, not a generation), with what every pinned
#: reference runs.  ``ruc_soilprop``: the engine default is the WRF
#: v4.0-4.5 / operational RAP/HRRR SOILPROP (woof.core.ruc_tier), while
#: stock WRF 4.6.1, 4.7.1 and 4.8.0 all ship the total-porosity form
#: (``phys/module_sf_ruclsm.F`` is byte-identical across the three).
#: Named breakage: the 4.8.0 HRRR-recipe proof (A195) ran the engine form
#: and left the referee at step 1 on 824 of 1,600 RUC columns (the
#: A195 gate, 2026-11-09).
REFERENCE_STOCK_SELECTORS = {"ruc_soilprop": "wrf_461"}


def reference_generations(reference: str) -> dict[str, str]:
    """The per-scheme generation keys ``reference`` resolves to.

    The release-table rows where the engine default is newer than that
    release's code, plus :data:`REFERENCE_STOCK_SELECTORS`, the selectors
    whose engine default is not stock WRF's code at all.
    """
    from woof.namelist_import import wrf_release_generations

    return {**wrf_release_generations(check_wrf_reference(reference)),
            **REFERENCE_STOCK_SELECTORS}


def resolve_wrf_reference(table: MutableMapping, *, where: str
                          ) -> str | None:
    """Pop ``wrf_reference`` from a config table and resolve it in place.

    Absent: returns None and touches nothing, so an existing configuration
    keeps its bytes and its identity.  Present: each generation key the
    named release runs is written into ``table`` unless the table already
    sets it (an explicit per-scheme key wins); the reference is returned.
    """
    if WRF_REFERENCE_KEY not in table:
        return None
    reference = check_wrf_reference(table.pop(WRF_REFERENCE_KEY),
                                    where=f"{WRF_REFERENCE_KEY} in {where}")
    for key, generation in reference_generations(reference).items():
        table.setdefault(key, generation)
    return reference


def wrf_reference_receipt(cfg) -> dict:
    """What a run receipt records: the pin the generation keys amount to.

    ``wrf_reference`` names the newest reference whose every generation
    row ``cfg`` runs, or ``"custom"`` when the keys mix releases;
    ``scheme_generations`` lists the keys themselves.
    """
    from dataclasses import fields

    from woof.config import RunConfig
    from woof.namelist_import import WRF_RELEASE_GENERATIONS

    keys = [key for key, _since, _old in WRF_RELEASE_GENERATIONS]
    defaults = {field.name: field.default for field in fields(RunConfig)
                if field.name in keys}
    generations = {key: getattr(cfg, key, None) for key in keys}
    matched = "custom"
    for reference in WRF_REFERENCES:
        pinned = reference_generations(reference)
        if all(generations[key] == pinned.get(key, defaults.get(key))
               for key in keys):
            matched = reference
            break
    return {"wrf_reference": matched, "scheme_generations": generations}


#: Namelist keys WRF 4.8.0's Registry declares and WRF 4.7.1's does not,
#: with the port that owns each.  Under the 4.8.0 vocabulary a key
#: written at its WRF 4.8.0 Registry default is consumed and recorded with
#: ``note`` (what ArWen does with it today); any other value is refused by
#: name, naming the owner, because no ArWen code runs it yet.  The port of a
#: consumer moves its keys out of this table.  New keys the
#: importer reads elsewhere (bl_mynn_mixaerosols, bl_mynn_mixnumcon,
#: windfarm_induction, the Noah-MP runoff split, &cordex) are not listed.
WRF480_ONLY_KEYS: dict[tuple[str, str], dict[str, str]] = {
    **{("physics", key): {
        "owner": "the MYNN-EDMF port",
        "note": "WRF 4.8.0 MYNN-EDMF 8f8e8b5 option; WOOF-ArWen's MYNN "
                "generations do not read it"}
       for key in ("bl_mynn_edmf_dd", "bl_mynn_ess")},
    **{("physics", key): {
        "owner": "the MYNN surface-layer port",
        "note": "WRF 4.8.0 MYNN-SFC b66187f roughness-length option; "
                "WOOF-ArWen's sf_sfclay_physics=5 is the in-tree WRF 4.7.1 "
                "module_sf_mynn, which has no such option"}
       for key in ("sf_mynn_sfcflux_land", "sf_mynn_sfcflux_water")},
    **{("physics", key): {
        "owner": "the KIM form-drag port",
        "note": "WRF 4.8.0 KIM turbulent form drag in revised MM5 "
                "(kim_tofd switches it on, tofd_factor scales it)"}
       for key in ("kim_tofd", "tofd_factor")},
    **{("dynamics", key): {
        "owner": "the KIM gravity-wave drag port",
        "note": "WRF 4.8.0 KIM gravity-wave drag control; WOOF-ArWen's "
                "gwd_opt=1 is the WRF 4.7.1 YSU GWDO, which does not read it"}
       for key in ("gwd_dx_factor", "gwd_if_nonhyd")},
    **{("physics", key): {
        "owner": "the Shin-Hong port",
        "note": "WRF 4.8.0 Shin-Hong option; read only with "
                "bl_pbl_physics = 11"}
       for key in ("shinhong_scu_mixing", "shinhong_nonlocal_flux",
                   "shinhong_ke_dissipation")},
    **{("physics", key): {
        "owner": "the TEMPO port",
        "note": "WRF 4.8.0 TEMPO flag; read only with mp_physics = 88"}
       for key in ("tempo_aerosolaware", "tempo_hailaware")},
    **{("physics", key): {
        "owner": "the dynamic-lightning port",
        "note": "WRF 4.8.0 dynamic-lightning control (dyn_lightning_option "
                "= 1 switches the scheme on)"}
       for key in ("dyn_lightning_option", "num_light_periods", "coul_pos",
                   "coul_neg", "coul_neu", "j_pos", "j_neg", "j_neu")},
    ("physics", "num_radn_layers"): {
        "owner": "the Noah-MP clean re-implementation program",
        "note": "Noah-MP v5 SNICAR band count; WOOF-ArWen runs the WRF-lineage "
                "v4.x Noah-MP, which has no such control"},
    **{("noah_mp", key): {
        "owner": "the Noah-MP clean re-implementation program",
        "note": "Noah-MP v5 option (UCAR licence, not taken); WOOF-ArWen runs "
                "the WRF-lineage v4.x Noah-MP, which has no such selector"}
       for key in ("opt_compact", "opt_tksno", "opt_scf", "opt_wetland",
                   "snicar_bandnumber_opt", "snicar_solarspec_opt",
                   "snicar_snowoptics_opt", "snicar_dustoptics_opt",
                   "snicar_rtsolver_opt", "snicar_snowshape_opt",
                   "snicar_use_aerosol", "snicar_snowbc_intmix",
                   "snicar_snowdust_intmix", "snicar_use_oc",
                   "snicar_aerosol_readtable")},
}

#: Rows of :data:`WRF480_ONLY_KEYS` whose port has landed (WRF-480
#: assembly, port/wrf-480).  The importer reads these keys itself under the
#: 4.8.0 vocabulary (the ports' own translations); under the 4.7.1
#: vocabulary a key WRF 4.7.1's Registry does not declare is refused by
#: name, as wrf.exe 4.7.1 fails its namelist read on it.  Named breakage:
#: the ports' importers read their keys unconditionally, so a 4.7.1 import
#: would have accepted a key wrf.exe 4.7.1 itself rejects.
WRF480_PORTED_OWNERS = ("the MYNN surface-layer port", "the KIM form-drag port",
                        "the KIM gravity-wave drag port", "the Shin-Hong port",
                        "the dynamic-lightning port")
WRF480_PORTED_KEYS: dict[tuple[str, str], dict[str, str]] = {
    key: WRF480_ONLY_KEYS.pop(key) for key in [
        key for key, row in WRF480_ONLY_KEYS.items()
        if row["owner"].startswith(WRF480_PORTED_OWNERS)]}


#: The WRF 4.8.0 ``&cordex`` group (Registry/registry.cordex).  Its
#: diagnostics are compiled only under ``CORDEXDIAG``, which the default
#: WRF build does not set and ArWen does not carry; with ``output_cordex``
#: at 0 every key of the group is inert.
CORDEX_GROUP = "cordex"
CORDEX_SWITCH = "output_cordex"

#: WRF 4.8.0 splits Noah-MP's ``opt_run`` into surface and subsurface
#: runoff selectors.  ArWen's WRF-lineage v4.x Noah-MP has the one
#: selector, which sets both; equal values map onto it.
NOAHMP_RUNOFF_SPLIT = ("opt_runsrf", "opt_runsub")
NOAHMP_RUNOFF_KEY = "opt_run"
NOAHMP_RUNOFF_SPLIT_OWNER = ("the Noah-MP clean re-implementation program "
                             "(its runoff-split kernel)")

#: WRF 4.8.0's physics-suite cells that differ from 4.7.1's
#: (share/module_check_a_mundo.F setup_physics_suite): the tropical
#: suite's surface layer moves from old MM5 (91) to revised MM5 (1).
PHYSICS_SUITE_CHANGES_480 = {"tropical": {"sf_sfclay_physics": 1}}
PHYSICS_SUITE_LINES_480 = {"conus": 3167, "tropical": 3184,
                           "unrecognized": 3199}
