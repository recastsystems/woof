"""Named opt-in hurricane front door; configurations remain ordinary experiments."""
from datetime import datetime
import json
from pathlib import Path

ARMS = {
    "stock": dict(isftcflx=2),
    "green-zhang": dict(isftcflx=2, tc_flux_curve="gz-option2"),
    "cblast": dict(isftcflx=2, tc_flux_curve="bell2012"),
    "bao": dict(isftcflx=2, tc_flux_curve="gz-option2", tc_spray="bao2000"),
    "barr": dict(isftcflx=2, tc_flux_curve="gz-option2", tc_spray="barr2023"),
}
FOLLOW = dict(field="pressure", threshold=1100., level_hpa=0., radius_km=60.,
    search_margin_cells=20, min_shift_cells=2, max_shift_cells=12,
    cooldown_seconds=3600., cadence_seconds=900., max_move_parent_cells=12,
    min_overlap_fraction=.7, pressure_center="minimum", fallback_level_hpa=850.,
    fallback_deficit_m=25.)


def value(v):
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {value(x)}" for k, x in v.items()) + " }"
    return json.dumps(v, allow_nan=False)


def configuration(*, start, latitude, longitude, hours=72., arm="green-zhang",
                  les=False, horizontal_length_m=0., wave_roughness=False,
                  fetch_m=100000., parent_dx_m=9000.):
    """Moving 9/3/1 or 12/3/1 km grids, with optional 200/40 m LES grids.

    Davis et al. (2008), doi:10.1175/2007MWR2085.1; Wang (2001),
    doi:10.1175/1520-0493(2001)129<1370:AESOTC>2.0.CO;2.
    Uses the engine's existing relocation and tracker, without truth steering.
    Fine grids are a configuration option, not a claim of resolved TSVs.
    Rotunno et al. (2009), doi:10.1175/2009BAMS2884.1.
    """
    if parent_dx_m not in (9000., 12000.):
        raise ValueError("parent_dx_m must be 9000 or 12000 for the 3/1 km children")
    from woof.domain_wizard import _ETA_LEVELS
    outer_ratio = int(parent_dx_m / 3000.)
    shared = dict(nz=49, eta_levels=list(_ETA_LEVELS), ztop=20000., p_top=5000., hybrid_opt=2, etac=.2,
        base_temp=290., time_step_sound=4, emdiv=.01, hypsometric_opt=2,
        h_sca_adv_order=5, smdiv=.1, moist_adv_opt=1, w_damping=1,
        damp_opt=3, zdamp=5000., dampcoef=.2, spec_zone=1, relax_zone=4,
        bldt=0., nwp_diagnostics=1, bl_pbl_physics=1, diff_opt=2,
        diff_6th_opt=2, diff_6th_slopeopt=1, epssm=.5, km_opt=4,
        moist=True, moist_cq=True, morr_rimed_ice=1, mp_physics=10,
        num_soil_layers=4, ra_lw_physics=4, ra_physics=0,
        ra_rrtmg_variant="rte-rrtmgp", ra_sw_physics=4, sf_sfclay_physics=91,
        sf_surface_physics=2, terrain_opt=1, top_lid=False,
        wrf_rrtmg_compatibility="wrf-rrtmg-4-4-to-rte-rrtmgp-v2", map_proj=1,
        tc_horizontal_length_m=horizontal_length_m, tc_wave_roughness=wave_roughness,
        tc_fetch_m=fetch_m, **ARMS[arm])
    tables = dict(experiment=dict(name="tc", start_time=start, run_seconds=hours*3600,
                    feedback=0, smooth_option=0, blend_width=5, spec_bdy_width=5,
                    restart_interval_s=0.),
        projection=dict(map_proj="lambert", ref_lat=latitude, ref_lon=longitude,
                        truelat1=latitude-10, truelat2=latitude+10, stand_lon=longitude),
        shared=shared, tiles=dict(mode="off"),
        output=dict(history_vars=["T2", "Q2", "PSFC", "U10", "V10", "TSK",
             "HFX", "LH", "QFX", "UST", "LANDMASK", "P", "PB", "PH", "QVAPOR",
             "U", "V", "W", "REFL_10CM", "RAINNC"]),
        fetch=dict(source="gefs", cycle=start.strftime("%Y-%m-%dT%H"), hours=int(hours),
                   out="inputs/gefs", cadence=3, member="c00"))
    domains=[]
    for idx,ratio in enumerate([1,outer_ratio,3]+([5,5] if les else []),1):
        domain=dict(grid_id=idx, parent_id=idx-1, parent_grid_ratio=ratio,
            parent_time_step_ratio=ratio, nx=240, ny=240,
            i_parent_start=1 if idx==1 else 1+(240-240//ratio)//2,
            j_parent_start=1 if idx==1 else 1+(240-240//ratio)//2,
            specified=idx==1, nested=idx!=1, history_interval_s=3600.,
            radt=12., cu_physics=1 if idx==1 else 0, cudt_minutes=parent_dx_m/2000., diff_6th_factor=.1)
        if idx==1:
            domain.update(dx=parent_dx_m,time_step=int(parent_dx_m/200.))
        else:
            domain["follow"]={**FOLLOW, "track":dict(path=f"storm-track.d{idx:02d}.csv")}
        if idx>=4:
            domain.update(bl_pbl_physics=0, km_opt=3, mix_isotropic=1,
                          tc_horizontal_length_m=0.)
            # An explicit fine-grid seed/extent remains user-editable. This
            # footprint covers the core only; no eyewall-TSV claim is made.
        domains.append(domain)
    lines=["# Opt-in TC experiment. Review footprint, source vortex and LES coverage."]
    for name,table in tables.items():
        lines += [f"\n[{name}]"]+[f"{k} = {value(v)}" for k,v in table.items()]
    for domain in domains:
        lines += ["\n[[domain]]"]+[f"{k} = {value(v)}" for k,v in domain.items()]
    return "\n".join(lines)+"\n"


def register_cli(sub):
    parser=sub.add_parser("tc",help="opt-in tropical-cyclone configurations and landfall diagnostics")
    commands=parser.add_subparsers(dest="tc_command",required=True)
    init=commands.add_parser("init",help="write and validate a following-nest experiment")
    init.add_argument("--start",type=datetime.fromisoformat,required=True)
    init.add_argument("--latitude",type=float,required=True)
    init.add_argument("--longitude",type=float,required=True)
    init.add_argument("--hours",type=float,default=72.)
    init.add_argument("--arm",choices=ARMS,default="green-zhang")
    init.add_argument("--les",action="store_true")
    init.add_argument("--horizontal-length-m",type=float,default=0.)
    init.add_argument("--wave-roughness",action="store_true")
    init.add_argument("--fetch-m",type=float,default=100000.)
    init.add_argument("--parent-dx-m",type=float,choices=(9000.,12000.),default=9000.)
    init.add_argument("--out",type=Path,required=True)
    init.set_defaults(func=initialize)
    curves=commands.add_parser("curves",help="show the equation rows, observations and extensions")
    curves.set_defaults(func=lambda args: print_curves())
    decay=commands.add_parser("decay",help="inland wind ratios from an ordered track CSV")
    decay.add_argument("--input",type=Path,required=True)
    decay.add_argument("--out",type=Path,required=True)
    decay.set_defaults(func=decay_main)
    vortices=commands.add_parser("vortices",help="GPU threshold candidates from Rust storm-diagnostics output")
    vortices.add_argument("--input",type=Path,required=True)
    vortices.add_argument("--height",type=float,default=500.)
    vortices.add_argument("--dx",type=float,required=True)
    vortices.add_argument("--dy",type=float,required=True)
    vortices.add_argument("--latitude",type=float,required=True)
    vortices.add_argument("--out",type=Path,required=True)
    vortices.set_defaults(func=vortices_main)


def decay_main(args):
    import csv
    from .diagnostics import inland_decay
    with args.input.open() as f:
        rows=inland_decay(csv.DictReader(f))
    with args.out.open('x') as f:
        json.dump(dict(schema='gpuwm-tc-decay.v1',rows=rows),f,indent=2)
        f.write('\n')
    return 0


def vortices_main(args):
    import cupy as cp
    from woof.netcdf_bridge import open_dataset
    from .diagnostics import vortex_candidates
    with open_dataset(args.input) as data:
        w=cp.asarray(data.variables[f'STORM_W_{args.height:.0f}M'][:]).squeeze()
        zeta=cp.asarray(data.variables[f'STORM_ZETA_{args.height:.0f}M'][:]).squeeze()
    mask=vortex_candidates(w,zeta,height_m=args.height,dx_m=args.dx,dy_m=args.dy,
                           latitude=args.latitude,xp=cp)
    # Only the small final candidate index product crosses to the host.
    indices=cp.argwhere(mask).get().tolist()
    with args.out.open('x') as f:
        json.dump(dict(schema='gpuwm-tc-vortex-candidates.v1', source=str(args.input),
            height_m=args.height, grid_m=[args.dy,args.dx], indices_yx=indices,
            interpretation='Threshold candidates; review 1--2 km circulation diameter '
                           'and lifetime in wind frames. This is not a tornado forecast.'),f,indent=2)
        f.write('\n')
    return 0


def print_curves():
    from .curves import CURVES
    print(json.dumps(dict(CURVES),indent=2))
    return 0


def initialize(args):
    from woof.experiment import build_experiment_from_config_tables
    import tomllib
    text=configuration(start=args.start,latitude=args.latitude,longitude=args.longitude,
        hours=args.hours,arm=args.arm,les=args.les,horizontal_length_m=args.horizontal_length_m,
        wave_roughness=args.wave_roughness,fetch_m=args.fetch_m,parent_dx_m=args.parent_dx_m)
    # Validate before creating an output, including every inherited domain selector.
    build_experiment_from_config_tables(tomllib.loads(text),source=str(args.out),
                                       base_dir=args.out.resolve().parent)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x") as f:
        f.write(text)
    print(f"Wrote {args.out}; prepare and run with woof go {args.out}")
    return 0
