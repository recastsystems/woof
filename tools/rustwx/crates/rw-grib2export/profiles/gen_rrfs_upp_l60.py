"""Build rrfs_mpas_postcntrl_L60.json, the RRFS v2 UPP field set the
rw_rrfs_upp exporter writes, from NOAA's own control tables.

Inputs (configuration tables only; no UPP source code is read or used):
  rrfs-workflow 570c81f fix/upp/rrfs_mpas_postcntrl_L60.xml   the field list
  rrfs-workflow 570c81f fix/upp/post_avblflds.xml             per-field defaults
  rrfs-workflow 570c81f fix/upp/params_grib2_tbl_new          pname -> d.c.n
(rrfs-workflow 570c81f pins UPP 1296eeb, whose parm/ copies are the same tables.)
Only fields in the runtime flat file (fix/upp/postxconfig-NT_L60.txt) are kept:
two GSD lightning-threat rows of the XML have no GRIB2 identity and are not in
the flat file UPP reads.

usage: python -I gen_rrfs_upp_l60.py SRC_DIR OUT_JSON
"""
import json
import sys
import xml.etree.ElementTree as ET

# Fixed-surface names of the control files -> WMO FM 92 code table 4.5
# (NCEP local entries 200+ per NCEP's GRIB2 documentation, table 4.5).
LEVEL = {
    "surface": 1, "cloud_base": 2, "cloud_top": 3, "0C_isotherm": 4,
    "lvl_of_adiab_cond_from_sfc": 5, "top_of_atmos": 8, "entire_atmos": 10,
    "level_free_convection": 14, "isothermal": 20, "isobaric_sfc": 100,
    "mean_sea_lvl": 101, "spec_hgt_lvl_above_grnd": 103, "sigma_lvl": 104,
    "hybrid_lvl": 105, "depth_bel_land_sfc": 106, "spec_pres_above_grnd": 108,
    "entire_atmos_single_lyr": 200, "hghst_trop_frz_lvl": 204,
    "bound_lyr_cloud_lyr": 211, "low_cloud_lyr": 214, "cloud_ceilng": 215,
    "planetary_bound_lyr": 220, "mid_cloud_lyr": 224, "high_cloud_lyr": 234,
    "equil_lvl": 247,
}
# What the UPP 1296eeb referee writes where the tables alone do not say (rrfs-products
# integration, 2026-10-29: UPP 1296eeb built and run exactly as RRFS v2's
# exrrfs_upp.sh runs it, IOFORM netcdfpara, MODELNAME RAPR, SUBMODELNAME MPAS, on
# MPASSIT 58fc18d output of the mv3km hrrrv5 history, f003 and f006):
# * the entire-atmosphere fields keep the table's level 200 (REFC, VIL, total TCDC,
#   TCOLW, TCOLI included; the earlier level-10 guess came from HRRR's GRIB2, which
#   the RRFS v2 path does not follow), so OBSERVED_LEVEL is empty;
# * isobaric levels are UPP's default pressure set (itag gives no po list, kpo 0:
#   2 Pa .. 100000 Pa), so the control file's 101320 Pa level is never written;
# * the run-total surface and subsurface runoff carry a one-hour window ending at
#   the lead (forecast time lead-1, length 1 h) over the run total, as UPP labels them.
OBSERVED_LEVEL = {}
UPP_DEFAULT_PRESSURE = {
    2., 3., 4., 7., 10., 20., 40., 70., 100., 200., 500., 700., 1000., 2000., 3000., 5000., 7000., 7500.,
    *[10000. + 2500. * k for k in range(37)],
}
WINDOW = {"ACM_SSRUN_ON_SURFACE": "last-hour-label", "ACM_BGRUN_ON_SURFACE": "last-hour-label"}
STAT = {"MAX": 2, "MIN": 3, "ACM": 1, "AVE": 0}
FILES = {"POSTPRS": "prslev", "POSTNAT": "natlev", "POSTTWO": "testbed"}


def floats(text):
    return [float(x) for x in (text or "").split()]


def main(src, out):
    l60 = ET.parse(f"{src}/fix_upp_rrfs_mpas_postcntrl_L60.xml").getroot()
    avbl = {}
    for p in ET.parse(f"{src}/fix_upp_post_avblflds.xml").getroot().iter("param"):
        d = {c.tag: (c.text or "").strip() for c in p}
        avbl[d["shortname"]] = d
    table = {}
    for line in open(f"{src}/fix_upp_params_grib2_tbl_new"):
        f = line.split()
        if len(f) >= 5:
            table.setdefault(f[4], []).append(tuple(int(x) for x in f[:4]))
    flat = open(f"{src}/fix_upp_postxconfig-NT_L60.txt").read().split("\n")
    runtime = set(x.strip() for x in flat)
    fields = []
    for ps in l60.iter("paramset"):
        datset = ps.find("datset").text.strip()
        for p in ps.iter("param"):
            name = p.find("shortname").text.strip()
            if name not in runtime:
                continue
            d = dict(avbl.get(name, {}))
            d.update({c.tag: (c.text or "").strip() for c in p})
            local = 1 if d.get("table_info") == "NCEP" else 0
            ents = table.get(d["pname"], [])
            pick = [e for e in ents if e[3] == local] or ents
            dcn = pick[0]
            t1 = d.get("fixed_sfc1_type")
            t2 = d.get("fixed_sfc2_type")
            s1 = int(d.get("scale_fact_fixed_sfc1", "0") or 0)
            s2 = int(d.get("scale_fact_fixed_sfc2", "0") or 0)
            lev1 = [v / 10 ** s1 for v in floats(d.get("level"))]
            if LEVEL[t1] == 100:
                lev1 = [v for v in lev1 if v in UPP_DEFAULT_PRESSURE]
            lev2 = [v / 10 ** s2 for v in floats(d.get("level2"))]
            tmpl = int((d.get("pdstmpl") or "tmpl4_0").split("_")[1])
            fields.append({
                "file": FILES[datset],
                "datset": datset,
                "shortname": name,
                "pname": d["pname"],
                "discipline": dcn[0], "category": dcn[1], "number": dcn[2],
                "ncep_local": bool(dcn[3]),
                "level_type": OBSERVED_LEVEL.get(name, LEVEL[t1]),
                "level_type_table": LEVEL[t1],
                "levels": lev1,
                "level2_type": LEVEL[t2] if t2 else None,
                "levels2": lev2,
                "template": tmpl,
                "stat": STAT.get(d.get("stats_proc")) if d.get("stats_proc") else None,
                "aerosol": d.get("aerosol_type") or None,
                "upp_scale": float(d["scale"]),
                "window": WINDOW.get(name),
            })
    doc = {
        "profile": "rrfs-mpas-postcntrl-L60",
        "source": "NOAA-EMC/rrfs-workflow 570c81f fix/upp (rrfs_mpas_postcntrl_L60.xml, post_avblflds.xml, "
                  "params_grib2_tbl_new; UPP 1296eeb); runtime set = postxconfig-NT_L60.txt",
        "files": {"prslev": "POSTPRS + POSTTWO", "natlev": "POSTNAT + POSTTWO", "testbed": "POSTTWO"},
        "fields": fields,
    }
    json.dump(doc, open(out, "w", newline="\n"), indent=1)
    print(len(fields), "fields")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
