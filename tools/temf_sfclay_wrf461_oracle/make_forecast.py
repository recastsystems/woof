"""Create a small TEMF forecast namelist from public WRF v4.6.1 em_real."""
from pathlib import Path
import argparse
import hashlib
import json
from woof.fortran_namelist import parse_namelist_text
from woof.namelist_import import import_namelists

def text(groups):
    out=[]
    for section,fields in groups.items():
        out.append("&"+section)
        for key,values in fields.items():
            def val(x):
                if isinstance(x,bool): return ".true." if x else ".false."
                if isinstance(x,str): return "'"+x+"'"
                return str(x)
            out.append(" "+key+" = "+", ".join(val(x) for x in values)+",")
        out.append("/")
    return "\n".join(out)+"\n"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--wrf-source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--cycle",default="2026-10-08T00")
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    original=a.wrf_source/"test/em_real/namelist.input"
    original_bytes=original.read_bytes()
    source=parse_namelist_text(original_bytes.decode())
    effective={g:{k:[v[0]] for k,v in fields.items()} for g,fields in source.items()}
    overrides={
        "time_control":{"run_days":0,"run_hours":0,"run_minutes":10,
            "run_seconds":0,"start_year":2026,"start_month":10,"start_day":8,
            "start_hour":0,"end_year":2026,"end_month":10,"end_day":8,
            "end_hour":0,"end_minute":10,"start_minute":0,
            "interval_seconds":10800,"history_interval":10,
            "restart":False,"restart_interval":60},
        "domains":{"time_step":10,"max_dom":1,"e_we":33,"e_sn":25,
            "e_vert":31,"eta_levels":[round(1.-k/30.,8) for k in range(31)],"p_top_requested":10000,"dx":12000.,"dy":12000.,
            "parent_id":0,"grid_id":1,"i_parent_start":1,"j_parent_start":1,
            "parent_grid_ratio":1,"parent_time_step_ratio":1,"feedback":0},
        "physics":{"mp_physics":6,"cu_physics":0,"ra_lw_physics":4,
            "ra_sw_physics":4,"bl_pbl_physics":10,"sf_sfclay_physics":10,
            "sf_surface_physics":2,"radt":15,"bldt":0,"cudt":0,
            "num_soil_layers":4,"fractional_seaice":0},
        "dynamics":{"use_theta_m":0,"gwd_opt":0},
        "bdy_control":{"specified":True,"nested":False}}
    for group,rows in overrides.items():
        for key,value in rows.items(): effective.setdefault(group,{})[key]=value if isinstance(value,list) else [value]
    effective["physics"].pop("physics_suite",None)
    # The public template's input level count describes its old met_em.
    effective["domains"].pop("num_metgrid_levels",None)
    effective["domains"].pop("dzstretch_s",None)
    wps=parse_namelist_text((Path(__file__).resolve().parents[2]/
        "configs/gfs_12km_quickstart.namelist.wps").read_text())
    wps["share"].update({"start_date":["2026-10-08_00:00:00"],
                         "end_date":["2026-10-08_00:10:00"]})
    wps["geogrid"]["e_we"]=[33];wps["geogrid"]["e_sn"]=[25]
    (a.out/"namelist.wps").write_text(text(wps))
    (a.out/"namelist.input").write_text(text(effective))
    (a.out/"public-namelist-original.input").write_bytes(original_bytes)
    config,report=import_namelists(a.out/"namelist.wps",a.out/"namelist.input",
                                  name="temf-public-column-forecast")
    config+="\n[fetch]\nsource = 'gfs'\ncycle = '"+a.cycle+"'\nhours = 3\ncadence = 3\narea = '31,-102,40,-93'\nout = '/workspace/shared-prepared/temf-public-gfs-2026100800'\n"
    (a.out/"forecast.toml").write_text(config)
    (a.out/"namelist-provenance.json").write_text(json.dumps({
        "public_source":"https://github.com/wrf-model/WRF/blob/v4.6.1/test/em_real/namelist.input",
        "original_sha256":hashlib.sha256(original_bytes).hexdigest(),
        "overrides":overrides,"removed":{"physics":["physics_suite"],
             "domains":["num_metgrid_levels","dzstretch_s"]},
        "effective_input_sha256":hashlib.sha256(text(effective).encode()).hexdigest(),
        "run_seconds":600,"grid":[32,24,30],
        "source_cycle":a.cycle,"declared_divergences":["RTE+RRTMGP for WRF 4/4",
            "TEMF six documented fixes and WOOF correctly rounded math"]
    },indent=2)+"\n")
    print("FORECAST_CONFIG_READY")
if __name__=="__main__":main()
