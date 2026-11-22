"""Write woof/data/chem/species/gocart.json and the GOCART set files.

Every number is WRF-Chem v4.7.1's (or GSL's for mp_coupling), with its
file:line.  Run from the worktree root.  Each row's optics object is the
merge of the two pinned optics contracts under
recast-woof-data/woof_data/data/chem/optics/ (see _optics below).
"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")


def _optics(root):
    """name -> the row's ``optics`` object, or None.

    WRF-Chem's GOCART volume optics (gocart_simple_optics_rows.json, the
    fields woof.core.chem_optics.pack_rows reads) joined with RTE-RRTMGP's
    MERRA lookup (gocart_simple-merra.json, the fields
    woof.core.chem_rrtmgp_aerosol.pack_rows reads).  The WRF file's
    merra_type/merra_bin are descriptive labels; the RRTMGP file's integer
    type index and bin replace them, and a row with no MERRA type carries
    none (it adds nothing to the RRTMGP lookup).
    """
    base = root / "recast-woof-data" / "woof_data" / "data" / "chem" / "optics"
    wrf = {r["name"]: r["optics"] for r in json.loads(
        (base / "gocart_simple_optics_rows.json").read_text())}
    merra = {r["name"]: r["optics"] for r in json.loads(
        (base / "gocart_simple-merra.json").read_text())}
    out = {}
    for name in wrf.keys() | merra.keys():
        o = dict(wrf.get(name) or {})
        o.pop("merra_type", None)
        o.pop("merra_bin", None)
        o.update(merra.get(name) or {})
        out[name] = o or None
    return out


OPTICS = _optics(root)

EPS = 1e-16  # WRF epsilc, chem/module_data_radm2.F:5

GS, GL, DU, GP = "gocart_simple", "gocart_lite", "dust", "gocart_primary"
CAMS = "cams_aq"

# The CAMS global forecast's aerosol mixing ratios as boundary entries of the
# GOCART rows (lane/aq-cams's derivation, one entry per row, with its bin
# weights and provenance).  An entry acts only when chem_sources enables
# cams-global; otherwise the row keeps its default inflow.
CAMS_BOUNDARY = json.loads(
    (Path(__file__).resolve().parent / "cams_gocart_boundary.json").read_text(
        encoding="utf-8"))["entries"]

# phys/module_data_gocart_dust.F:18-21 (den_dust, reff_dust, ipoint, frac_s),
# :28 (distr_dust); chem/chemics_init.F:1742 (ch_dust = 0.8D-9 into a REAL).
# chem/module_gocart_dust_afwa.F:357-361 (the per-bin VIS_DUST coefficients)
DUST = [
    # reff_m,   den,    ipoint, frac_s, distr,    vis
    (0.73e-6, 2500.0, 3, 0.1, 1.074e-1, 1.470e-6),
    (1.4e-6, 2650.0, 2, 0.25, 1.012e-1, 7.877e-7),
    (2.4e-6, 2650.0, 2, 0.25, 2.078e-1, 4.623e-7),
    (4.5e-6, 2650.0, 2, 0.25, 4.817e-1, 2.429e-7),
    (8.0e-6, 2650.0, 2, 0.25, 1.019e-1, 1.387e-7),
]
# chem/module_data_gocart_seas.F:2-5 (ra, rb in um; den_seas; reff_seas)
SEAS = [(0.1, 0.5, 0.30e-6), (0.5, 1.5, 1.00e-6), (1.5, 5.0, 3.25e-6),
        (5.0, 10.0, 7.50e-6)]
# GSL get_niwfa, mp_thompson.F90:1062-1066: dust bins 1-5 -> nifa;
# GOCART sea salt 1-4 = MERRA sea-salt bins 7-10 (same edges 0.1-0.5,
# 0.5-1.5, 1.5-5, 5-10 um) -> nwfa group 0 (x9.); sulfate -> group 1 (x5);
# hydrophilic OC -> group 2 (x8).
NIFA_DUST = (4.0737762, 30.459203, 153.45048, 1011.5142, 5683.3501)
NWFA_SEAS = (0.2907854, 12.91224, 206.2216, 4326.23)
GSL = ("ccpp-physics 3e6660c6 physics/MP/Thompson/mp_thompson.F90:1062-1066 "
       "(get_niwfa, Apache-2.0)")

VERTMX = "mixing.vertmx"


def row(name, output, long_name, units, phase, family, sets, inflow, processes,
        prov, **kw):
    r = {"name": name, "output_name": output, "long_name": long_name,
         "units": units, "phase": phase, "family": family, "sets": sets,
         "transported": kw.pop("transported", True), "floor": EPS,
         "default_inflow": inflow,
         "molar_mass_g_mol": kw.pop("molar_mass_g_mol", None),
         "processes": processes,
         "emissions": kw.pop("emissions", []), "boundary": kw.pop("boundary", []),
         "drydep": kw.pop("drydep", None), "settling": kw.pop("settling", None),
         "wetdep_ls_alpha": kw.pop("wetdep_ls_alpha", 0.0),
         "aging": kw.pop("aging", None),
         "optics": OPTICS.get(name, kw.pop("optics", None)),
         "mp_coupling": kw.pop("mp_coupling", None),
         "exclusive_group": None, "provenance": prov}
    r.update(kw)
    return r


DRY_AER = {"scheme": "gocart_aerosol"}
INFLOW = ("default_inflow: bdy_chem_value_gocart, "
          "chem/module_input_chem_data.F:1267-1298")
WETDEP = ("wetdep_ls_alpha: chem/module_wetdep_ls.F:39-46 (chem_opt >= 300)")


def edgar(field, weight, prov):
    return {"source": "edgar-v81", "field": field, "weight": weight,
            "vertical": "surface", "provenance": prov}


rows = []
# ---- sulfur ----------------------------------------------------------------
# so2 is ONE row shared by the GOCART sulfur sets and the CAMS air-quality set
# (the table refuses a species defined twice).  GOCART oxidizes it
# (chem.sulfur, gated to the two GOCART sulfur sets by process_sets); the CAMS
# set carries it with no chemistry.  Every set deposits it through Wesely, as
# WRF-Chem's GOCART_SIMPLE path does (wesely_driver for the gases before
# gocart_drydep_driver, chem/dry_dep_driver.F:377-399); the Wesely block and
# the CAMS boundary entry are lane/aq-cams's (dep_init's general branch).
SO2_WESELY = {
    "scheme": "wesely",
    "wesely": {
        "hstar": 253000.0, "dhr": 5816.0, "f0": 0.0,
        "dratio": 1.9783283, "scpr23": 1.4261369, "arm": "sulfur_dioxide",
        "from": "hstar chem/module_dep_simple.F:2198, dhr :2475, f0 :2687, "
                "dvj 0.126 :2902, dratio and scpr23 :3478-3489; SO2 resistance "
                "arm rc :1081-1150 (Registry dname so2, registry.chem:1580)"}}
SO2_CAMS = {
    "source": "cams-global", "fields": ["sulphur_dioxide"], "weights": [1.0],
    "conversion": "kg_kg_moist_to_ppmv_dry",
    "provenance": "CAMS mass mixing ratio per moist air is taken to dry air as "
                  "q/(1-q_v) with the same frame's CAMS specific humidity, then "
                  "to ppmv as q_dry * mwdry / M_i * 1e6 with mwdry = 28.966 "
                  "g/mol (share/module_model_constants.F:34) and M_i this row's "
                  "molar_mass_g_mol."}
rows.append(row(
    "so2", "so2", "SO2 mixing ratio", "ppmv", "gas", "sulfur_gas",
    [GS, GL, CAMS], 5.0e-6,
    ["emission.inventory", "chem.sulfur", "drydep.wesely", VERTMX],
    "registry.chem:3610 row so2; " + INFLOW + "; molar mass 64.066 (WRF-Chem's "
    "GOCART value; the CAMS conversion and the mass ledger read it); EDGAR "
    "weight kg m-2 s-1 -> mol km-2 hr-1 = 1e6*3600/0.064066; dry deposition "
    "through wesely_driver (chem/dry_dep_driver.F:377) with dep_init's general "
    "(non-MOZART, non-CBM4/CB05) branch numbers, the dratio and scpr23 float32 "
    "words dep_init derives from dvj (tests/test_chem_wesely_wrf471_parity.py "
    "holds them to the Fortran dump); chem.sulfur only in the GOCART sulfur "
    "sets (process_sets), so the CAMS set carries SO2 without chemistry; "
    "wetdep_ls_alpha 0 (large-scale wet scavenging skips gases, "
    "chem/module_wetdep_ls.F:39-46)",
    molar_mass_g_mol=64.066, sulfur_role="so2", drydep=SO2_WESELY,
    process_sets={"chem.sulfur": [GS, GL]},
    boundary=[SO2_CAMS],
    emissions=[edgar("so2", 1.0e6 * 3600.0 / 0.064066,
                     "EDGAR v8.1 SO2 total flux, kg m-2 s-1 to WRF emis_ant "
                     "mol km-2 hr-1 (emissions_driver.F:1601-1603)")]))
rows.append(row(
    "sulf", "sulf", "SULF mixing ratio (sulfate)", "ppmv", "aerosol",
    "sulfate", [GS, GL], 3.0e-6,
    ["chem.sulfur", "drydep.gocart", VERTMX, "wetdep.ls", "optics.gocart",
     "coupling.thompson"],
    "registry.chem:3610 row sulf; " + INFLOW + "; " + WETDEP + " (1.0); "
    "molar mass 96.0576 (sum_pm_gocart mwso4, chem/module_gocart_aerosols.F:85); "
    "gocart_drydep_driver hands sulf the aerosol velocity "
    "(chem/module_gocart_drydep.F:116)",
    molar_mass_g_mol=96.0576, sulfur_role="so4", drydep=DRY_AER,
    wetdep_ls_alpha=1.0,
    mp_coupling={"target": "nwfa", "group": 1, "order": 0, "group_factor": 5,
                 "unit_mass": 0.3053104, "from": GSL}))
rows.append(row(
    "dms", "dms", "DMS mixing ratio", "ppmv", "gas", "sulfur_gas", [GS], 1.0e-6,
    ["chem.sulfur", VERTMX],
    "registry.chem:3610 row dms; " + INFLOW + "; no emission until a DMS "
    "seawater source row exists (dmsemis_opt refused); ddvel(dms) = 0 for "
    "GOCART_SIMPLE (chem/module_dep_simple.F:437)",
    molar_mass_g_mol=62.13, sulfur_role="dms"))
rows.append(row(
    "msa", "msa", "MSA mixing ratio", "ppmv", "gas", "sulfur_gas", [GS], 1.0e-6,
    ["chem.sulfur", "drydep.gocart", VERTMX],
    "registry.chem:3610 row msa; " + INFLOW + "; aerosol deposition velocity "
    "(chem/module_gocart_drydep.F:117)",
    molar_mass_g_mol=96.109, sulfur_role="msa", drydep=DRY_AER))
# ---- unspeciated PM ----------------------------------------------------------
for name, out, long_name, inflow in (("p25", "P25", "other gocart primary pm25", 1.0),
                                     ("p10", "P10", "other gocart primary pm10", 1.0e-12)):
    rows.append(row(
        name, out, long_name, "ug kg-1", "aerosol", "pm_other", [GS], inflow,
        ["drydep.gocart", VERTMX, "wetdep.ls", "optics.gocart"],
        "registry.chem:3610; " + INFLOW + "; " + WETDEP + " (0.5); no "
        "inventory row yet (unspeciated PM needs PM minus BC and OC)",
        drydep=DRY_AER, wetdep_ls_alpha=0.5))
# ---- carbon ------------------------------------------------------------------
AGE = "chem/module_gocart_aerosols.F:155 (r1 = 4.63D-6 s-1, 2.5 d) and :70 (oc2 += 8*dBC2)"
rows.append(row(
    "bc1", "BC1", "Hydrophobic Black Carbon", "ug kg-1", "aerosol", "bc",
    [GS, GL, GP], 1.0e-2,
    ["emission.inventory", "aging.gocart", "drydep.gocart", VERTMX,
     "optics.gocart"],
    "registry.chem:3610; " + INFLOW + "; " + WETDEP + " (bc1 skipped); "
    "all anthropogenic BC enters bc1 (emissions_driver.F:1604-1605); " + AGE,
    drydep=DRY_AER,
    aging={"to": "bc2", "efold_s": 1.0 / 4.63e-6, "rate_per_s": 4.63e-6,
           "coproduct": {"to": "oc2", "factor": 8.0}, "from": AGE},
    emissions=[edgar("bc", 1.0e9, "EDGAR v8.1 BC total flux, kg m-2 s-1 to "
                     "ug m-2 s-1 (emissions_driver.F:1604-1605)")]))
rows.append(row(
    "bc2", "BC2", "Hydrophilic Black Carbon", "ug kg-1", "aerosol", "bc",
    [GS, GL, GP], 1.0e-2,
    # aging.gocart writes this row (bc1's aged mass arrives here), so the
    # row names it: the driver books a process's change only on its rows.
    ["aging.gocart", "drydep.gocart", VERTMX, "wetdep.ls", "optics.gocart"],
    "registry.chem:3610; " + INFLOW + "; " + WETDEP + " (0.8)",
    drydep=DRY_AER, wetdep_ls_alpha=0.8))
rows.append(row(
    "oc1", "OC1", "Hydrophobic Organic Carbon", "ug kg-1", "aerosol", "oc",
    [GS, GL, GP], 1.0e-2,
    ["emission.inventory", "aging.gocart", "drydep.gocart", VERTMX,
     "optics.gocart"],
    "registry.chem:3610 (WRF's long name says Black Carbon, a Registry typo); "
    + INFLOW + "; " + WETDEP + " (oc1 skipped); " + AGE,
    drydep=DRY_AER,
    aging={"to": "oc2", "efold_s": 1.0 / 4.63e-6, "rate_per_s": 4.63e-6,
           "from": AGE},
    emissions=[edgar("oc", 1.0e9, "EDGAR v8.1 OC total flux (carbon mass), "
                     "kg m-2 s-1 to ug m-2 s-1 (emissions_driver.F:1606-1607)")]))
rows.append(row(
    "oc2", "OC2", "Hydrophilic Organic Carbon", "ug kg-1", "aerosol", "oc",
    [GS, GL, GP], 1.0e-2,
    # aging.gocart writes this row (oc1's aged mass and bc aging's
    # co-product), so the row names it.
    ["aging.gocart", "drydep.gocart", VERTMX, "wetdep.ls", "optics.gocart",
     "coupling.thompson"],
    "registry.chem:3610; " + INFLOW + "; " + WETDEP + " (0.8)",
    drydep=DRY_AER, wetdep_ls_alpha=0.8,
    mp_coupling={"target": "nwfa", "group": 2, "order": 0, "group_factor": 8,
                 "unit_mass": 0.3232698, "from": GSL}))
# ---- dust --------------------------------------------------------------------
for i, (reff, den, ipoint, frac, distr, vis) in enumerate(DUST, start=1):
    rows.append(row(
        f"dust_{i}", f"DUST_{i}", f"dust size bin {i}: {reff*1e6:g}um effective radius",
        "ug kg-1", "aerosol", "dust", [GS, GL, DU, GP], 1.0e-12,
        ["emission.dust", "drydep.gocart", "settling.gocart", VERTMX,
         "wetdep.ls", "optics.gocart", "coupling.thompson"],
        "registry.chem:3610; " + INFLOW + "; " + WETDEP + " (0.5); bin "
        "constants phys/module_data_gocart_dust.F:18-21, :28",
        drydep=DRY_AER, wetdep_ls_alpha=0.5,
        settling={"radius_m": reff, "density_kg_m3": den, "growth": "none",
                  "from": "phys/module_data_gocart_dust.F:18-19"},
        dust_emission={"erod_class": ipoint, "frac_s": frac, "ch_dust": 0.8e-9,
                       "afwa_distr": distr, "afwa_vis_coefficient": vis,
                       "from": "phys/module_data_gocart_dust.F:20-21, :28; "
                               "chem/chemics_init.F:1742; "
                               "chem/module_gocart_dust_afwa.F:357-361"},
        mp_coupling={"target": "nifa", "group": 0, "order": i, "group_factor": 1,
                     "unit_mass": NIFA_DUST[i - 1], "from": GSL}))
# ---- sea salt ----------------------------------------------------------------
for i, (ra, rb, reff) in enumerate(SEAS, start=1):
    rows.append(row(
        f"seas_{i}", f"SEAS_{i}", f"sea-salt size bin {i}: {reff*1e6:g}um effective radius",
        "ug kg-1", "aerosol", "seasalt", [GS, GL, GP], 1.0e-12,
        ["emission.seasalt", "drydep.gocart", "settling.gocart", VERTMX,
         "wetdep.ls", "optics.gocart", "coupling.thompson"],
        "registry.chem:3610; " + INFLOW + "; " + WETDEP + " (1.0); bin "
        "constants chem/module_data_gocart_seas.F:2-5",
        drydep=DRY_AER, wetdep_ls_alpha=1.0,
        settling={"radius_m": reff, "density_kg_m3": 2200.0,
                  "growth": "gerber_seasalt",
                  "from": "chem/module_data_gocart_seas.F:4-5; Gerber growth "
                          "chem/module_gocart_settling.F:249, :262, :328-331"},
        seasalt_emission={"r_lo_um": ra, "r_hi_um": rb,
                          "from": "chem/module_data_gocart_seas.F:2-3"},
        mp_coupling={"target": "nwfa", "group": 0, "order": i + 6,
                     "group_factor": 9.0, "unit_mass": NWFA_SEAS[i - 1],
                     "from": GSL + "; GOCART seas_" + str(i) + " = MERRA-2 "
                     "sea-salt bin " + str(i + 6)}))

for r in rows:
    entry = CAMS_BOUNDARY.get(r["name"])
    if entry is not None:
        r["boundary"] = [*r["boundary"], entry]
unknown = sorted(set(CAMS_BOUNDARY) - {r["name"] for r in rows})
if unknown:
    raise SystemExit(f"cams_gocart_boundary.json names no GOCART row {unknown}")

species = {"schema": "gpuwm.chem.species.v1", "owner": "aq-gocart",
           "description": "The GOCART aerosol family of WRF-Chem v4.7.1 "
           "(chem_opt=300 gocart_simple package, registry.chem:4022), one row "
           "per species; the sets gocart_simple, gocart_lite, dust and "
           "gocart_primary select among them.",
           "rows": rows}
(root / "woof/data/chem/species").mkdir(parents=True, exist_ok=True)
(root / "woof/data/chem/species/gocart.json").write_text(
    json.dumps(species, indent=2) + "\n", encoding="utf-8", newline="\n")

SETS = {
    GS: ("WRF-Chem's GOCART_SIMPLE package, all 19 species (registry.chem:4022).",
         300, "registry.chem:4022 package gocart_simple chem_opt==300"),
    GL: ("GOCART-lite: GOCART_SIMPLE without DMS and MSA (no DMS seawater "
         "source row) and without the unspeciated P25/P10 (no anthropogenic "
         "PM row); 15 species.", None,
         "WOOF-ArWen's default GOCART (DESIGN 1.4); a subset of chem_opt=300"),
    DU: ("Dust only: the five GOCART dust bins (chem_opt=401).", 401,
         "registry.chem:4031 package dust chem_opt==401"),
    GP: ("GOCART primary aerosols: dust, sea salt, BC and OC with no sulfur "
         "rows, so no oxidant source (and no keyed download) is needed; 13 "
         "species.", None,
         "WOOF-ArWen subset of chem_opt=300 for runs without an oxidant source "
         "(DESIGN 6.4: 'the user can run gocart_lite minus the sulfur rows')"),
}
for name, (desc, opt, prov) in SETS.items():
    (root / "woof/data/chem/sets").mkdir(parents=True, exist_ok=True)
    (root / f"woof/data/chem/sets/{name}.json").write_text(json.dumps({
        "schema": "gpuwm.chem.set.v1", "name": name, "description": desc,
        "wrf_chem_opt": opt, "wrf_tracer_opt": None, "provenance": prov},
        indent=2) + "\n", encoding="utf-8", newline="\n")
print(len(rows), "rows")
