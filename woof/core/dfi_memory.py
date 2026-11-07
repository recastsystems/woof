"""Startup-only DFI device storage, priced before a resident restore.

Array names follow ModelDfiState, DeviceDfiState and DfiPhysicsInitialization.
History uses the existing complete field inventory. Calendar and descriptor
entries are explicit upper bounds, not measured allocator corrections.
"""
from math import ceil


def startup_shapes(domain, *, step_seconds=None):
    cfg = domain.run
    if not cfg.dfi_opt:
        return {}
    from woof.config import soil_layer_count
    from woof.core.dfi_model import MOIST_NAMES, SCALAR_NAMES
    from woof.ingest.wrfinput import active_moisture_map
    from woof.core.dfi_runtime import lifecycle
    from woof.io.history_layout import history_frame_bytes
    from woof.wrf_exact import ENABLED, DIAGNOSTICS_ENABLED

    nz,ny,nx = cfg.nz,cfg.ny,cfg.nx
    mass,full,surface = (nz,ny,nx),(nz+1,ny,nx),(ny,nx)
    soil = (soil_layer_count(cfg),ny,nx)
    active = active_moisture_map(cfg)
    moisture = (1+sum(n in active for n in MOIST_NAMES),*mass)
    scalar = (1+sum(n in active for n in SCALAR_NAMES),*mass)
    shapes = {'diagnostic/ww':full,'diagnostic/php':mass,'diagnostic/ph0':full,
        'binding/moist':moisture,'binding/scalar':scalar,
        'binding/re_cloud':mass,'binding/re_ice':mass,'binding/re_snow':mass,
        'saved/moist':moisture,'saved/scalar':scalar,'saved/rh':mass,
        'saved/re_cloud':mass,'saved/re_ice':mass,'saved/re_snow':mass}
    for name in ('t_2','php','p','tke_2','al','alt','pb'):
        shapes['saved/'+name] = mass
    for name in ('w_2','ww','phb','ph0','ph_2'):
        shapes['saved/'+name] = full
    shapes.update({'saved/mu_2':surface,'saved/u_2':(nz,ny,nx+1),
                   'saved/v_2':(nz,ny+1,nx)})
    for name in ('snow','snowh','snowc','canwat','tsk'):
        shapes['saved/'+name] = surface
    for name in ('tslb','smois'):
        shapes['saved/'+name] = soil
    if cfg.sf_surface_physics == 3:
        for name in ('qvg','tsnav','soilt1'):
            shapes['saved/'+name] = surface
        for name in ('smfr3d','keepfr3dflag'):
            shapes['saved/'+name] = soil
    for name in ('pb','alb','thb','dphb_resid'):
        shapes['base/'+name] = mass
    shapes.update({'base/phb':full,'base/mub':surface})
    if not ENABLED:
        shapes['binding/t_2'] = mass
    if not DIAGNOSTICS_ENABLED:
        shapes['binding/p'] = mass
    # Missing model TKE uses the native inactive carrier; a selected TKE
    # scheme aliases its existing carrier. Charging the mass array covers both.
    shapes['binding/inactive_tke_bound'] = mass
    for name in ('effc','effi','effs'):
        shapes['base/'+name+'_bound'] = mass
    phases,_ = lifecycle(cfg.dfi_opt,cfg.dfi_bckstop_seconds,cfg.dfi_fwdstop_seconds)
    dt = (step_seconds if step_seconds is not None else
          cfg.time_step_dfi if cfg.time_step_dfi > 0 else cfg.dt)
    steps = max(ceil(abs(p.stop-p.start)/dt) for p in phases)
    shapes['coefficients'] = (steps+1,)
    # uint64 descriptor rows and driver graph metadata are charged in
    # four-byte units. 64 KiB per solve bounds the added DFI event/row
    # descriptors without folding them into unrelated forecast storage.
    shapes['graph_descriptor_bound'] = (16384*(steps+1),)
    if cfg.ra_physics or cfg.ra_lw_physics or cfg.ra_sw_physics:
        # Per-time coszen, 64-bit day/night indices and the shipped
        # climatology interpolation. The native climatology has fewer
        # than 128 levels; model+128 also covers the model extensions.
        shapes['calendar_bound'] = (steps+1,ny,nx,nz+128+3)
    if cfg.dfi_write_dfi_history:
        interval = float(domain.history_interval_s)
        count = max(1 if interval <= 0 else
                    1+int(abs(p.stop-p.start)//interval) for p in phases)
        frame = history_frame_bytes(cfg,include_reflectivity=False)
        shapes['history_snapshots_bound'] = (count*((frame+3)//4),)
    return shapes
