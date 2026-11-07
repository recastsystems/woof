"""Carry coupled ocean columns across moving meshes on CUDA.

Water-only area overlap preserves column inventories on unchanged water
support. Coastline/analysis-support changes are measured separately; no
uniform temperature correction disguises their budget. No prescribed-ocean
path imports CUDA or changes a carrier.
"""
from copy import deepcopy

_STATIC = {'greenfrac', 'albedo12m', 'landmask', 'ocean_coriolis'}
_CELL_FIRST = {'ocean_pwp_columns', 'ocean_pwp_diagnostics'}


def remap_surface(payload, fresh, weights, layout, target_mesh, donors, cache):
    states = [b.get('surface_updates') for b in payload['backends']]
    selected = fresh.get('surface_updates')
    if selected is None or selected.get('ocean') is None:
        if any(s is not None and s.get('ocean') is not None for s in states):
            raise ValueError('moving target lost its coupled ocean')
        return None
    if any(s is None or s.get('ocean') is None for s in states):
        raise ValueError('moving ocean requires every source owner state')
    for state in states:
        if state['ocean']['options'] != selected['ocean']['options']:
            raise ValueError('moving ocean parameters or initial source changed')
        if {k:v for k,v in state['identity'].items() if k!='cells'} != {
                k:v for k,v in selected['identity'].items() if k!='cells'}:
            raise ValueError('moving ocean stream clock changed')
        if set(state['arrays']) != set(selected['arrays']):
            raise ValueError('moving ocean carrier inventory changed')
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    source_n = len(weights.source_area)
    target_n = len(weights.target_area)
    work = cache.setdefault('coupled_ocean', {}).setdefault(cp.cuda.Device().id, {})
    if not work:
        # Owner-only gathers are device assignments, including the unusual
        # cell-first (cell,variable,depth) PWP storage.
        fields = {}
        for key, first in states[0]['arrays'].items():
            shape = first.shape[1:] if key in _CELL_FIRST else first.shape[:-1]
            result = cp.empty((*shape,source_n),dtype=first.dtype)
            seen = cp.zeros(source_n,dtype=cp.int32)
            for state, old_layout in zip(states,payload['layouts']):
                a=cp.asarray(state['arrays'][key]);a=cp.moveaxis(a,0,-1) if key in _CELL_FIRST else a
                if a.shape[:-1]!=shape or a.shape[-1]!=len(old_layout.cell_l2g):
                    raise ValueError('unknown moving ocean layout: '+key)
                count=old_layout.n_owned_cells
                ids=cp.asarray(old_layout.cell_l2g[:count],dtype=cp.int64)
                result[...,ids]=a[...,:count];cp.add.at(seen,ids,1)
            if not bool(cp.all(seen==1)):raise ValueError('moving ocean owners must cover every source cell once')
            fields[key]=result
        overlap=csr_matrix(weights.overlap)
        water=(fields['ocean_mask']>0).astype(cp.float64)
        covered=overlap@water
        def mean(a, mask=water, denominator=covered):
            flat=cp.where(mask>0,a,0).reshape(-1,source_n)
            value=(overlap@flat.T).T.reshape(*a.shape[:-1],target_n)
            return value/cp.maximum(denominator,cp.finfo(cp.float64).tiny)
        mapped={key:mean(a) for key,a in fields.items() if key.startswith('ocean_') and key not in _STATIC}
        def rotate(u,v,mask,denominator):
            old_lat=cp.asarray(payload['mesh'].arrays['latCell']);old_lon=cp.asarray(payload['mesh'].arrays['lonCell'])
            new_lat=cp.asarray(target_mesh.arrays['latCell']);new_lon=cp.asarray(target_mesh.arrays['lonCell'])
            east=(-cp.sin(old_lon),cp.cos(old_lon),cp.zeros_like(old_lon))
            north=(-cp.sin(old_lat)*cp.cos(old_lon),-cp.sin(old_lat)*cp.sin(old_lon),cp.cos(old_lat))
            te=(-cp.sin(new_lon),cp.cos(new_lon),cp.zeros_like(new_lon))
            tn=(-cp.sin(new_lat)*cp.cos(new_lon),-cp.sin(new_lat)*cp.sin(new_lon),cp.cos(new_lat))
            xyz=[mean(u*east[j]+v*north[j],mask,denominator) for j in range(3)]
            return sum(xyz[j]*te[j] for j in range(3)),sum(xyz[j]*tn[j] for j in range(3))
        if 'ocean_hu' in mapped:
            mapped['ocean_hu'],mapped['ocean_hv']=rotate(fields['ocean_hu'],fields['ocean_hv'],water,covered)
        pwp_covered=None
        pwp_failures=None
        if 'ocean_pwp_columns' in mapped:
            active=(fields['ocean_pwp_diagnostics'][3]>.5).astype(cp.float64)
            pwp_covered=overlap@active
            columns=mean(fields['ocean_pwp_columns'],active,pwp_covered)
            columns[2],columns[3]=rotate(fields['ocean_pwp_columns'][2],fields['ocean_pwp_columns'][3],active,pwp_covered)
            mapped['ocean_pwp_columns']=columns
            # Preserve any exhausted mixing flag among contributing water
            # columns. A nearest land donor can miss a failed contributor.
            rows=cp.repeat(cp.arange(target_n,dtype=cp.int64),cp.diff(overlap.indptr))
            pwp_failures=cp.zeros(target_n,dtype=fields['ocean_pwp_diagnostics'].dtype)
            contributions=cp.where((overlap.data>0)&(active[overlap.indices]>0),
                                   fields['ocean_pwp_diagnostics'][2,overlap.indices],0)
            cp.maximum.at(pwp_failures,rows,contributions)
        work.update(fields=fields,mapped=mapped,covered=covered,pwp_covered=pwp_covered,pwp_failures=pwp_failures)
    ids=cp.asarray(layout.cell_l2g,dtype=cp.int64)
    donor_ids=cp.asarray(donors,dtype=cp.int64)
    output=deepcopy({k:v for k,v in selected.items() if k!='arrays'})
    output['arrays']={}
    eligible=(cp.asarray(selected['arrays']['ocean_mask'])>0)&(work['covered'][ids]>0)
    for key,old in selected['arrays'].items():
        target=cp.asarray(old)
        if key in _STATIC:
            output['arrays'][key]=target
            continue
        if key.startswith('ocean_'):
            value=work['mapped'][key][...,ids]
            if key in _CELL_FIRST:value=cp.moveaxis(value,-1,0)
            mask=eligible.reshape((-1,)+(1,)*(target.ndim-1)) if key in _CELL_FIRST else eligible
            if key=='ocean_mask':value=cp.ones_like(target)
            if key=='ocean_pwp_columns':
                mask=mask&(work['pwp_covered'][ids]>0)[:,None,None]
            value=cp.where(mask,value,target)
        else:
            # Deep-soil/calendar carriers stay coherent with the existing
            # geography-compatible land donor; SST is republished below.
            value=work['fields'][key][...,donor_ids]
        output['arrays'][key]=value.astype(target.dtype)
    if 'ocean_pwp_columns' in output['arrays']:
        a=output['arrays']
        carried=eligible&(work['pwp_covered'][ids]>0)
        active=(carried|(cp.asarray(selected['arrays']['ocean_pwp_diagnostics'])[:,3]>.5))&(a['ocean_area']==0)&(a['ocean_mask']>0)
        a['ocean_pwp_diagnostics'][:,3]=active.astype(a['ocean_pwp_diagnostics'].dtype)
        # Diagnostics are recomputed by PWP; never average a categorical flag
        # or turn a nonzero mixing-failure count into a fractional success.
        a['ocean_pwp_diagnostics'][:,2]=cp.where(carried,work['pwp_failures'][ids],a['ocean_pwp_diagnostics'][:,2])
        p=selected['ocean']['options']['pwp']
        density=p['density']*(-p['thermal_expansion']*a['ocean_pwp_columns'][:,0,:]+p['haline_contraction']*a['ocean_pwp_columns'][:,1,:])
        jumps=density[:,1:]-density[:,:-1]>p['density_jump']
        depth=cp.where(jumps.any(axis=1),jumps.argmax(axis=1)+1,density.shape[1])*p['layer_m']
        a['ocean_pwp_diagnostics'][:,0]=cp.where(active,depth,a['ocean_pwp_diagnostics'][:,0])
        a['ocean_depth']=cp.where(active,depth,a['ocean_depth']).astype(a['ocean_depth'].dtype)
        a['ocean_temperature']=cp.where(active,a['ocean_pwp_columns'][:,0,0],a['ocean_temperature'])
    for key in ('alarms','february','counts','last_record'):
        output[key]=deepcopy(states[0][key])
    output['ocean']['steps']=states[0]['ocean']['steps']
    fresh['surface_updates']=output
    # Use physical area, not the wet overlap denominator, to expose coastline
    # changes in the actual model inventory. Owner cells only avoid halo counts.
    count=layout.n_owned_cells;own=ids[:count]
    report={'method':'CUDA water-only area overlap; Earth-vector current rotation',
            'mapped_owned_water_cells':int(cp.sum(eligible[:count])),
            'uncovered_owned_water_cells':int(cp.sum((cp.asarray(selected['arrays']['ocean_mask'])[:count]>0)&~eligible[:count])),
            'support_note':'Conservative on unchanged water support; coastline/support changes are explicit and uncorrected'}
    if 'ocean_pwp_columns' in output['arrays']:
        p=selected['ocean']['options']['pwp'];factor=p['density']*p['heat_capacity']*p['layer_m']
        source=work['fields']['ocean_pwp_columns'][0];mask=work['fields']['ocean_pwp_diagnostics'][3]>.5
        source_heat=cp.sum(cp.where(mask,cp.sum(source-273.15,axis=0),0)*cp.asarray(weights.source_area))*factor
        target=output['arrays']['ocean_pwp_columns'][:count,0,:]
        target_mask=output['arrays']['ocean_pwp_diagnostics'][:count,3]>.5
        target_heat=cp.sum(cp.where(target_mask,cp.sum(target-273.15,axis=1),0)*cp.asarray(weights.target_area)[own])*factor
        source_salt=cp.sum(cp.where(mask,cp.sum(work['fields']['ocean_pwp_columns'][1],axis=0),0)*cp.asarray(weights.source_area))*p['density']*p['layer_m']/1000
        target_salt=cp.sum(cp.where(target_mask,cp.sum(output['arrays']['ocean_pwp_columns'][:count,1,:],axis=1),0)*cp.asarray(weights.target_area)[own])*p['density']*p['layer_m']/1000
        report.update(source_global_sensible_heat_above_273_15_j=float(source_heat),target_owned_sensible_heat_above_273_15_j=float(target_heat),
            source_global_salt_mass_kg=float(source_salt),target_owned_salt_mass_kg=float(target_salt),
            newly_initialized_owned_pwp_cells=int(cp.sum(active[:count]&~carried[:count])))
    return report


def restore_surface_backend(backend, payload, weights, layout, target_mesh, cache):
    """Run after attach_host: the target surface service does not exist earlier."""
    if not any((b.get('surface_updates') or {}).get('ocean') is not None
               for b in payload['backends']):
        return None
    service = getattr(backend, '_surface_updates', None)
    if service is None:
        raise ValueError('moving target did not attach its coupled ocean')
    fresh = {'surface_updates': service.state(device=True)}
    donors = cache['ocean_surface_donors'][id(layout)]
    receipt = remap_surface(payload, fresh, weights, layout, target_mesh, donors, cache)
    service.restore(fresh['surface_updates'])
    # The coupled ocean owns ocean ice: its carried area is the concentration.
    # The member's role words came from its cold geography and its xicem from a
    # land-state donor; bring both to the carried ice before the conversion
    # runs, so MPAS's threshold-crossing conversion sees no false crossing and
    # no sea point is left classified as glacier or land (moving_physics.
    # settle_ocean_roles names the the storm breakage).
    import cupy as cp
    from .moving_physics import settle_ocean_roles
    f = backend._seam._driver.fields
    ocean = service.ocean.array('mask').reshape(f['xice'].shape) > 0
    wet = ocean & (f['lakemask'] < .5)
    f['xice'][...] = cp.where(ocean, service.ocean.array('area').reshape(f['xice'].shape), f['xice'])
    # What the conversion alone would have left (the defect, counted on owned
    # columns): a sea point below the ice threshold still carrying xland 1 with
    # a donor xicem below the threshold is never reconverted, so Noah-MP runs
    # it as land or glacier; a land column whose donor xicem is icy is turned
    # into open water by the ice-to-water branch.
    count = layout.n_owned_cells
    thr = cp.float32(service.threshold)
    x, xl = f['xice'].reshape(-1)[:count], f['xland'].reshape(-1)[:count]
    o, w = service.owned['xicem'][:count], wet.reshape(-1)[:count]
    stranded = int(cp.count_nonzero(w & (x < thr) & (xl < 1.5) & (o < thr)))
    land_to_water = int(cp.count_nonzero(~ocean.reshape(-1)[:count] & (x < thr) & (xl < 1.5) & (o >= thr)))
    settled = settle_ocean_roles(f, wet, service.threshold, service.isice, service.iswater, cp)
    x = f['xice'].reshape(-1)
    service.owned['xicem'][...] = cp.where(x >= thr, x, 0)
    service.owned['seaice'][...] = (service.owned['xicem'] > 0).astype(service.owned['seaice'].dtype)
    after = int(cp.count_nonzero(wet.reshape(-1)[:count] & (x[:count] < thr) & (f['xland'].reshape(-1)[:count] < 1.5)))
    if receipt is not None:
        receipt['surface_roles'] = dict(
            method='wet columns take the role of the carried ocean ice; xicem set to it (no false crossing)',
            role_columns_changed=settled, unsettled_sea_points_as_land_or_glacier=stranded,
            unsettled_land_turned_water=land_to_water, sea_points_as_land_or_glacier_after=after)
    service.ocean.publish(convert=True)
    service.refresh_classification()
    return receipt
