"""Carry resident column physics between meshes and recomputed LTS classes.

Owner columns alone form the source. Halo and scratch columns are rebuilt
from those owners. Integer categories use the largest-overlap donor. Floating
SAVE fields use the same positive overlap operator as the atmosphere.
"""
from copy import deepcopy
import numpy as np

CATEGORIES = {'fields/xland', 'fields/xice', 'fields/landmask', 'fields/lakemask'}
STATIC = {'fields/xland', 'fields/xice', 'fields/landmask', 'fields/lakemask',
          'fields/isltyp', 'fields/ivgtyp', 'fields/shdmax', 'fields/shdmin',
          'fields/tmn', 'fields/vegfra', 'fields/albbck', 'fields/embck',
          # The surface albedo and emissivity are blended over the sea-ice fraction, and xice above is the new
          # member's own: carrying them from the old mesh pairs one cell's blended albedo with another's ice fraction,
          # and RUC's de-blend then returns an ice albedo above 1. THE BREAKAGE THIS PREVENTS (HRRR-shape setup run,
          # a check of run_2bfb0e1e82195f53, 2026-10-25): the first move stopped with "RUC CUDA snow preparation
          # alb must be below 1". The previous-step ice fraction follows xice for the same reason.
          'fields/albedo', 'fields/emiss', 'fields/xicem'}
# Noah-MP's stores and layer geometry form one column state. Independently
# averaging them while selecting integer snow layers from a donor breaks
# the solver's water accounting. Carry this group from one compatible donor.
LAND_STATE = {'fields/'+name for name in (
    'snow','snowh','canwat','tslb','smois','sh2o','lai',
    'tvxy','tgxy','canicexy','canliqxy','eahxy','tahxy','cmxy','chxy','fwetxy',
    'sneqvoxy','alboldxy','qsnowxy','qrainxy','wslakexy','waxy','xsaixy','taussxy',
    'isnowxy','pgsxy','tsnoxy','snicexy','snliqxy','zsnsoxy')}
LAND_CATEGORIES = ('fields/xland','fields/xice','fields/isltyp','fields/ivgtyp')
VECTOR_PAIRS=(('pbl/du','pbl/dv'),('seam/du','seam/dv'),('fields/u10','fields/v10'))+tuple(
    (f'driver/{scheme}_tendencies/ru',f'driver/{scheme}_tendencies/rv')
    for scheme in ('pbl','radiation','cumulus'))


# Classify by the sealed field contract, never by a coincidental soil/level
# dimension match (reduced tests can have four atmospheric and soil levels).
ATMOSPHERIC_FIELDS = {'diag/cldfra', 'fields/exch_h', 'fields/exch_m',
    'radiation/o33d_grid', 'cumulus/w0avg', 'driver/rthratenlw', 'driver/rthratensw'}
ATMOSPHERIC_PREFIXES = ('pbl/', 'seam/', 'state/', 'driver/pbl_tendencies/',
    'driver/radiation_tendencies/', 'driver/cumulus_tendencies/')


def atmospheric_intensive(values, weights):
    mapper = getattr(weights, 'layer_intensive', weights.intensive)
    return mapper(values)


def field_intensive(values, weights, key):
    if key.startswith(ATMOSPHERIC_PREFIXES) and key not in ATMOSPHERIC_FIELDS and hasattr(weights, 'source_dzw') and \
            values.shape[0] not in (len(weights.source_dzw), len(weights.source_dzw) + 1):
        # A column seam (the RUC/MYNN suites) carries its surface and soil state under seam/ too: a leading axis that
        # is neither the mass levels nor their interfaces (1 for a surface field, 9 RUC soil levels) is not
        # atmospheric, and maps across the cells only. THE BREAKAGE THIS PREVENTS (owner run run_2bfb0e1e82195f53,
        # HRRR-shape setup-following 3 km, 2026-10-25): the first move stopped with "height map layer field has
        # invalid source dimensions".
        return weights.intensive(values)
    if key in ATMOSPHERIC_FIELDS or key.startswith(ATMOSPHERIC_PREFIXES):
        # Known column rates use the mass grid. Fail on an unknown vertical
        # layout rather than silently treating it as computational levels.
        if hasattr(weights, 'source_dzw') and values.shape[0] == len(weights.source_dzw) + 1:
            shape = values.shape
            flat = values.reshape(shape[0], -1, shape[-1])
            mapped = np.stack([weights.interface_intensive(flat[:, j])
                               for j in range(flat.shape[1])], axis=1)
            return mapped.reshape((mapped.shape[0], *shape[1:-1], mapped.shape[-1]))
        return atmospheric_intensive(values, weights)
    return weights.intensive(values)


def local_vector_bases(mesh):
    lat=np.asarray(mesh.arrays['latCell'],dtype=float)
    lon=np.asarray(mesh.arrays['lonCell'],dtype=float)
    east=np.stack((-np.sin(lon),np.cos(lon),np.zeros_like(lon)),axis=-1)
    north=np.stack((-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)),axis=-1)
    return east,north


def carry_horizontal_vector(u,v,weights,source_mesh,target_mesh, *, atmospheric=False):
    if u.dtype!=v.dtype:
        raise ValueError('physics vector components have incompatible dtypes')
    y_stagger=(u.ndim==3 and v.ndim==3 and u.shape[1]==1 and v.shape[1]==2 and
               u.shape[0]==v.shape[0] and u.shape[2]==v.shape[2])
    if y_stagger:
        # The column seam has one physical y column and two duplicated y
        # faces. Verify that contract before collapsing and rebuilding it.
        if not np.array_equal(v[:,0],v[:,1]):
            raise ValueError('physics pseudo y-edge copies differ')
        mapped_u,mapped_v=carry_horizontal_vector(u[:,0],v[:,0],weights,source_mesh,target_mesh,atmospheric=atmospheric)
        return mapped_u[:,None,:],np.repeat(mapped_v[:,None,:],2,axis=1)
    if u.shape!=v.shape:
        raise ValueError(f'physics vector components have incompatible layouts: {u.shape}, {v.shape}')
    east,north=local_vector_bases(source_mesh)
    target_east,target_north=local_vector_bases(target_mesh)
    mapper = (lambda x: atmospheric_intensive(x, weights)) if atmospheric else weights.intensive
    components=[mapper(u*east[:,axis]+v*north[:,axis]) for axis in range(3)]
    mapped_u=sum(value*target_east[:,axis] for axis,value in enumerate(components))
    mapped_v=sum(value*target_north[:,axis] for axis,value in enumerate(components))
    return mapped_u.astype(u.dtype),mapped_v.astype(v.dtype)


def coupled_column_faces(du,dv):
    """Native unit-mass column seam's periodic C-grid interpolation."""
    if du.shape!=dv.shape or du.ndim!=3 or du.shape[1]!=1 or du.dtype!=dv.dtype:
        raise ValueError('raw column PBL rates have incompatible layouts')
    ru=np.float32(.5)*(du+np.roll(du,1,axis=2))
    return np.concatenate((ru,ru[:,:,:1]),axis=2),np.concatenate((dv,dv[:,:1,:]),axis=1)


def gather_owner_columns(arrays, layouts, n_cells):
    shape = arrays[0].shape[:-1]
    result = np.empty((*shape, n_cells), dtype=arrays[0].dtype)
    seen = np.zeros(n_cells, dtype=bool)
    padding = []
    for array, layout in zip(arrays, layouts):
        local = len(layout.cell_l2g)
        if array.shape[:-1] != shape or array.shape[-1] not in (local, local+1):
            raise ValueError('physics field has an unknown column layout')
        count = layout.n_owned_cells
        ids = np.asarray(layout.cell_l2g[:count], dtype=np.int64)
        if np.any(seen[ids]):
            raise ValueError('physics owner columns overlap')
        seen[ids] = True
        result[..., ids] = array[..., :count]
        if array.shape[-1] == local+1:
            padding.append(float(np.max(abs(array[..., -1]))))
    if not np.all(seen):
        raise ValueError('physics owner columns do not cover the globe')
    return result, padding


def largest_donors(weights):
    """Each target cell's largest-overlap donor (the first one on ties), as ``np.argmax`` per CSR row.

    Vectorized over rows: the row maximum by ``np.maximum.reduceat`` and the first stored position
    holding it by ``np.minimum.reduceat`` give the same donor ``np.argmax`` picks. THE COST THIS
    REMOVES: the Python loop over ~98,000 rows ran on one core at every move, once per field that
    needs it on the MPAS route. A map with an empty or non-finite row keeps the loop.
    """
    matrix=weights.overlap
    cached=getattr(weights,'_largest_donors',None)
    if cached is not None and cached[0] is matrix:
        return cached[1].copy()  # callers may refine their copy (coherent surface donors)
    source=matrix
    if getattr(matrix,'format',None)!='csr':
        matrix=matrix.tocsr()
    indptr=np.asarray(matrix.indptr,dtype=np.int64)
    data=np.asarray(matrix.data)
    if matrix.shape[0]==0 or np.any(np.diff(indptr)<=0) or not np.all(np.isfinite(data)):
        result=np.array([matrix.indices[matrix.indptr[i]+np.argmax(
            matrix.data[matrix.indptr[i]:matrix.indptr[i+1]])]
            for i in range(matrix.shape[0])], dtype=np.int64)
    else:
        starts=indptr[:-1]
        best=np.maximum.reduceat(data,starts)
        rows=np.repeat(np.arange(matrix.shape[0]),np.diff(indptr))
        positions=np.where(data==best[rows],np.arange(len(data)),len(data))
        first=np.minimum.reduceat(positions,starts)
        result=np.asarray(matrix.indices,dtype=np.int64)[first]
    try:
        weights._largest_donors=(source,result.copy())
    except AttributeError:
        pass
    return result


def carry_field(values, weights, *, categorical=False, donors=None):
    if categorical or np.issubdtype(values.dtype, np.integer):
        return values[..., largest_donors(weights) if donors is None else donors]
    return weights.intensive(values).astype(values.dtype)


def mapped_owner_field(payload, weights, key, cache=None, *, target_mesh=None):
    """Map a sealed owner field once, then slice it into every new class."""
    if cache is None:
        cache={}
    identity=(id(payload),id(weights),id(target_mesh))
    if cache.setdefault('identity',identity)!=identity:
        raise ValueError('physics field cache belongs to another remap')
    fields=cache.setdefault('fields',{})
    if key not in fields:
        pair=next((pair for pair in VECTOR_PAIRS if key in pair),None)
        if pair is not None:
            if target_mesh is None:
                raise ValueError('physics vector remap requires target geography')
            old=[gather_owner_columns([source['seam']['arrays'][component] for source in payload['backends']],
                 payload['layouts'],len(weights.source_area)) for component in pair]
            mapped=carry_horizontal_vector(old[0][0],old[1][0],weights,payload['mesh'],target_mesh,
                atmospheric=pair[0] != 'fields/u10')
            for index,component in enumerate(pair):fields[component]=(mapped[index],old[index][1],False)
            return fields[key]
        values,padding=gather_owner_columns(
            [source['seam']['arrays'][key] for source in payload['backends']],
            payload['layouts'],len(weights.source_area))
        categorical=key in CATEGORIES or np.issubdtype(values.dtype,np.integer)
        if categorical and 'donors' not in cache:
            cache['donors']=largest_donors(weights)
        mapped = (carry_field(values,weights,categorical=True,donors=cache.get('donors'))
                  if categorical else field_intensive(values,weights,key).astype(values.dtype))
        fields[key]=(mapped,padding,categorical)
    return fields[key]


def prefetch_owner_fields(payload, weights, keys, cache, *, target_mesh=None):
    """Map every not-yet-cached owner field on all cores; the class loop then only slices.

    Each field is the same ``mapped_owner_field`` work on its own inputs, written into the shared
    cache only by this thread, so every mapped word is the one the sequential loop produced.
    THE COST THIS REMOVES: the first class's loop mapped ~100 persistent physics fields one after
    another on one core, 6.2 s of a resumed WOOF storm-following segment (box-aws2x 2026-10-24).
    """
    from .host_parallel import run_tasks
    if cache is None:
        return
    identity=(id(payload),id(weights),id(target_mesh))
    if cache.setdefault('identity',identity)!=identity:
        raise ValueError('physics field cache belongs to another remap')
    fields=cache.setdefault('fields',{})
    pending,seen=[],set()
    for key in keys:
        pair=next((pair for pair in VECTOR_PAIRS if key in pair),None)
        group=pair if pair is not None else (key,)
        if key in fields or group in seen:
            continue
        seen.add(group)
        pending.append(key)
    if not pending:
        return
    categorical_needed=any(
        key in CATEGORIES or np.issubdtype(payload['backends'][0]['seam']['arrays'][key].dtype,np.integer)
        for key in pending if not any(key in pair for pair in VECTOR_PAIRS))
    if categorical_needed and 'donors' not in cache:
        cache['donors']=largest_donors(weights)
    # One atmospheric field first: it builds the map's lazily cached device geometry once.
    first=next((key for key in pending if key in ATMOSPHERIC_FIELDS or key.startswith(ATMOSPHERIC_PREFIXES)),None)
    if first is not None:
        mapped_owner_field(payload,weights,first,cache,target_mesh=target_mesh)
        pending.remove(first)

    def compute(key):
        local={'identity':identity,'fields':{},'donors':cache.get('donors')}
        if local['donors'] is None:
            local.pop('donors')
        mapped_owner_field(payload,weights,key,local,target_mesh=target_mesh)
        return local['fields']
    for result in run_tasks([lambda key=key: compute(key) for key in pending]):
        for key,value in result.items():
            fields.setdefault(key,value)


def sea_ice_threshold(identity):
    """The seam's own sea-ice classification threshold.

    The Arwen and Noah-MP seams seal an explicit xice_threshold; the Noah
    suite seams seal fractional_seaice (0/1) instead, whose threshold is
    MPAS's own rule (mpas_atmphys_vars / framework init: 0.02 when
    config_frac_seaice, else 0.5).  Breakage it fixes: every Thompson (suite
    seam) moving member died at its first move with KeyError xice_threshold
    (ensemble-07, 2026-10-21).
    """
    if identity.get('xice_threshold') is not None:
        return float(identity['xice_threshold'])
    if 'xice_threshold' in identity:
        # The column seams of the RUC suites (HRRR shape) seal the key with no value: they run fractional sea ice
        # (config_frac_seaice), whose MPAS threshold is 0.02, the value the hex driver passes every column seam.
        # Breakage it fixes: the HRRR-shape setup run run_f3e336fa8500473d died at its first move with
        # "float() argument must be ... not 'NoneType'".
        fractional = identity.get('fractional_seaice')
        if fractional in (0, 1):
            return 0.02 if int(fractional) else 0.5
        from woof.hex.drivers.run_cuda_v841_full_physics_x4 import ARWEN_XICE_THRESHOLD
        return float(ARWEN_XICE_THRESHOLD)
    if 'xice_threshold' in identity.get('surface',()):
        # The column-physics seam (mpas-column-physics-v1) seals the NAMES of
        # its initialize_physics surface inputs, not their values; the hex
        # driver passes xice_threshold=ARWEN_XICE_THRESHOLD on every such
        # seam (run_cuda_v841_full_physics_x4 surface inputs).  Breakage it
        # fixes: Thompson moving members still died at their first move with
        # "seals neither xice_threshold nor fractional_seaice" (moving
        # smoke-thompson-08, 2026-10-21).
        from woof.hex.drivers.run_cuda_v841_full_physics_x4 import ARWEN_XICE_THRESHOLD
        return float(ARWEN_XICE_THRESHOLD)
    fractional=identity.get('fractional_seaice')
    if fractional in (0,1):
        return 0.02 if int(fractional) else 0.5
    raise KeyError('seam identity seals neither xice_threshold nor fractional_seaice')


#: The surface role words MPAS's sea-ice conversion writes for a sea point
#: (mpas_atmphys_update_surface.F:196-313, woof.hex.mpasphys.surface_update_v842
#: sst_update): ice takes the land-ice class, soil 16 and xland 1; open water
#: takes the water class, soil 14 and xland 2. Vegetation is zero either way.
OCEAN_ROLE_FIELDS = ('xland', 'ivgtyp', 'isltyp', 'vegfra', 'albbck', 'embck')


def surface_land_use(backend):
    """(isice, iswater) of the backend's vegetation dataset, as its surface service reads them."""
    noahmp = getattr(getattr(getattr(backend, '_seam', None), '_driver', None), 'noahmp_params', None)
    if noahmp is not None:
        return int(noahmp.land_use.isice), int(noahmp.land_use.iswater)
    from woof.core.noahmp import load_noahmp_parameters
    from woof.core.noahmp_runtime import DEFAULT_VEGETATION_DATASET
    _, veg = load_noahmp_parameters().vegetation_groups(DEFAULT_VEGETATION_DATASET)
    return int(veg.scalar('ISICE')), int(veg.scalar('ISWATER'))


def settle_ocean_roles(fields, wet, threshold, isice, iswater, xp=np):
    """Give every wet column the surface role its CURRENT ice concentration implies.

    A moved member's role words (xland, vegetation and soil class, background
    albedo) come from its own cold geography, i.e. the initial ice extent on the
    new mesh, while its ice concentration is carried. Where the two disagree a
    sea point is left half converted: xice below the threshold with xland 1 and
    the land-ice class classifies as GLACIER (or ordinary land), so Noah-MP runs
    a glacier column on ocean state. MPAS's own conversion cannot repair it,
    because it fires only on a threshold crossing against the carried xicem.
    Breakage it fixes: the storm hex storm-following, box M 2026-10-27, every move
    flipped polar columns water/sea ice/glacier, the history census refused, the
    Noah-MP glacier water budget failed and the next move's sea-ice remap
    refused. Returns the number of wet columns whose role changed.
    """
    # Written in place: the driver's arrays are bound into its restart manifest.
    ice = wet & (fields['xice'] >= xp.float32(threshold))
    water = wet & ~ice
    values = dict(xland=(1, 2), ivgtyp=(isice, iswater), isltyp=(16, 14), vegfra=(0, 0),
                  albbck=(.80, .08), embck=(.98, .98))
    changed = xp.zeros(wet.shape, dtype=bool)
    for name in OCEAN_ROLE_FIELDS:
        if name not in fields:
            continue
        old = fields[name]
        on_ice, on_water = (xp.asarray(v, dtype=old.dtype) for v in values[name])
        new = xp.where(ice, on_ice, xp.where(water, on_water, old)).astype(old.dtype)
        if name in ('xland', 'ivgtyp', 'isltyp'):
            changed |= new != old
        old[...] = new
    return int(xp.count_nonzero(changed))


def land_state_donors(payload,weights,layout,fresh,target_mesh,cache):
    """One local coherent donor, preserving land, sea-ice and glacier roles."""
    from scipy.spatial import cKDTree
    from .cuda_arwen_physics_v841 import _ISICE_TABLE
    sources=cache.setdefault('land_sources',{})
    for key in LAND_CATEGORIES:
        if key not in sources:
            sources[key]=gather_owner_columns([b['seam']['arrays'][key] for b in payload['backends']],
                                              payload['layouts'],len(weights.source_area))[0]
    categories=np.stack([sources[key].reshape(-1) for key in LAND_CATEGORIES],axis=-1)
    count=len(layout.cell_l2g)
    targets=np.stack([fresh[key].reshape(-1)[:count] for key in LAND_CATEGORIES],axis=-1)
    threshold=sea_ice_threshold(payload['backends'][0]['seam']['identity'])
    categories[:,0]=categories[:,0]>=1.5;targets[:,0]=targets[:,0]>=1.5
    categories[:,1]=categories[:,1]>=threshold;targets[:,1]=targets[:,1]>=threshold
    categories=np.column_stack((categories,categories[:,3]==_ISICE_TABLE))
    targets=np.column_stack((targets,targets[:,3]==_ISICE_TABLE))
    if 'ocean_ice_fields' in cache:
        # Native ocean ice has its own conservative volume/snow owner below.
        # Land donors must retain glacier roles; ocean donors need not match
        # the cold geography's obsolete ice/vegetation classification.
        if 'fields/landmask' not in sources:
            sources['fields/landmask']=gather_owner_columns(
                [b['seam']['arrays']['fields/landmask'] for b in payload['backends']],
                payload['layouts'],len(weights.source_area))[0]
        ocean=sources['fields/landmask'].reshape(-1)<.5
        target_ocean=fresh['fields/landmask'].reshape(-1)[:count]<.5
        categories[:,0]=ocean;targets[:,0]=target_ocean
        categories[ocean,1]=0;targets[target_ocean,1]=0
        categories[ocean,4]=0;targets[target_ocean,4]=0
    if 'donors' not in cache:cache['donors']=largest_donors(weights)
    defaults=cache['donors']
    ids=defaults[layout.cell_l2g].copy()
    mismatch=np.any(categories[ids]!=targets,axis=1)
    fallback=0;distance_max=0.;relaxed={}
    trees=cache.setdefault('land_category_trees',{})
    for column in np.flatnonzero(mismatch):
        row=int(layout.cell_l2g[column]);start,end=weights.overlap.indptr[row:row+2]
        candidates=weights.overlap.indices[start:end]
        found=False;tier=None
        for fields in ((0,1,2,3,4),(0,1,2,4),(0,1,3,4),(0,1,4)):
            compatible=np.all(categories[candidates][:,fields]==targets[column,list(fields)],axis=1)
            if not np.any(compatible):continue
            positions=np.flatnonzero(compatible)
            best=positions[np.argmax(weights.overlap.data[start:end][positions])]
            ids[column]=candidates[best];tier=fields;found=True;break
        if found:
            if len(tier)<5:
                name=','.join(LAND_CATEGORIES[i] for i in tier if i<4)
                relaxed[name]=relaxed.get(name,0)+1
            continue
        fields=(0,1,4)
        category=tuple(targets[column,list(fields)])
        if category not in trees:
            available=np.flatnonzero(np.all(categories[:,fields]==targets[column,list(fields)],axis=1))
            if not len(available):
                raise ValueError(f'no carried land column matches new land/ice/glacier role {category}')
            coordinates=np.stack([payload['mesh'].arrays[k][available] for k in ('xCell','yCell','zCell')],axis=-1)
            coordinates/=np.linalg.norm(coordinates,axis=1,keepdims=True)
            trees[category]=(available,cKDTree(coordinates))
        available,tree=trees[category]
        point=np.array([target_mesh.arrays[k][row] for k in ('xCell','yCell','zCell')],dtype=float)
        point/=np.linalg.norm(point)
        chord,index=tree.query(point)
        ids[column]=available[index];fallback+=1
        name=','.join(LAND_CATEGORIES[i] for i in fields if i<4)
        relaxed[name]=relaxed.get(name,0)+1
        distance_max=max(distance_max,float(2*np.arcsin(min(chord/2,1))*
                                           payload['mesh'].attrs['sphere_radius']/1000))
    return ids,dict(columns=count,dominant_category_mismatch_columns=int(mismatch.sum()),
                   changed_dominant_donors=int(np.count_nonzero(ids!=defaults[layout.cell_l2g])),
                   nearest_compatible_fallback_columns=fallback,
                   nearest_compatible_fallback_distance_max_km=distance_max,
                   relaxed_category_donor_columns=relaxed,xice_threshold=threshold,
                   categories=list(LAND_CATEGORIES))


def restore_remapped_backend(backend, payload, weights, layout, *, base_dt, rate, mapped_cache=None,target_mesh=None):
    from .moving_seaice import complete_source_ice, remap_ocean_ice, SEAICE_FIELDS
    if mapped_cache is None:
        mapped_cache = {}
    if 'ice_complete_payload' not in mapped_cache:
        mapped_cache['ice_complete_payload'] = complete_source_ice(payload)
    payload = mapped_cache['ice_complete_payload']
    fresh=backend.restart_state()
    sources=payload['backends']
    # Only the sealed column census changes. Never bypass an engine/physics
    # contract mismatch by manufacturing a fresh identity around old data.
    for source in sources:
        if {k:v for k,v in source['identity'].items() if k!='constructor_identity_sha256'} != {
                k:v for k,v in fresh['identity'].items() if k!='constructor_identity_sha256'}:
            raise ValueError('remapped physics engine or contract changed')
        changed={'n_columns','dt','stepra','stepbl','stepcu'}
        old_selection={k:v for k,v in source['seam']['identity'].items() if k not in changed}
        new_selection={k:v for k,v in fresh['seam']['identity'].items() if k not in changed}
        if old_selection != new_selection:
            difference={key:{'source':old_selection.get(key),'target':new_selection.get(key)}
                        for key in old_selection.keys()|new_selection.keys()
                        if old_selection.get(key)!=new_selection.get(key)}
            raise ValueError(f'remapped physics selection changed: {difference}')
    keys=set().union(*(set(source['seam']['arrays']) for source in sources))
    # A local-time-step class with no ocean column carries no sea-ice state (fields/ice_*): those fields exist only in
    # the classes that hold water cells. Such a class's columns are land, which holds no ice, so its share of an ice
    # field is zero. THE BREAKAGE THIS PREVENTS (owner run run_1227117932ed36db, HRRR-shape setup-following 3 km, GFS
    # 2026-10-23 18Z f354, 2026-10-24): the first move stopped with 'source physics classes have different persistent
    # fields' (4 classes; 16 fields/ice_* present in 2). Any other missing field is still refused.
    for source in sources:
        arrays=source['seam']['arrays']
        missing=keys-set(arrays)
        if any(not key.startswith('fields/ice_') for key in missing):
            raise ValueError('source physics classes have different persistent fields: '
                             + ', '.join(sorted(k for k in missing if not k.startswith('fields/ice_'))))
        columns=int(source['seam']['identity'].get('n_columns') or 0)
        for key in missing:
            like=next(s['seam']['arrays'][key] for s in sources if key in s['seam']['arrays'])
            arrays[key]=np.zeros((*tuple(like.shape)[:-1],columns),dtype=like.dtype)
    if set(fresh['seam']['arrays'])-keys:
        raise ValueError('target physics needs fields absent from the boundary')
    receipt=[]
    result=dict(fresh['seam']['arrays'])
    local=len(layout.cell_l2g)
    if mapped_cache is None:mapped_cache={}
    ice_fields = None
    if SEAICE_FIELDS <= keys:
        if 'ocean_ice_fields' not in mapped_cache:
            target_landmask = mapped_cache.get('target_landmask')
            if target_landmask is None:
                if not np.array_equal(layout.cell_l2g, np.arange(len(weights.target_area))):
                    raise ValueError('partitioned ice remap needs global target landmask')
                target_landmask = fresh['seam']['arrays']['fields/landmask']
            ice_fields, ice_receipt = remap_ocean_ice(payload, weights, target_mesh, target_landmask)
            mapped_cache['ocean_ice_fields'] = ice_fields
            mapped_cache['ocean_ice_receipt'] = ice_receipt
        ice_fields = mapped_cache['ocean_ice_fields']
        fresh_arrays = fresh['seam']['arrays']
        fresh_arrays['fields/xice'] = ice_fields['fields/xice'][..., layout.cell_l2g]
        wet = ice_fields['fields/ice_ocean_mask'][..., layout.cell_l2g] > 0
        if np.any(wet & (fresh_arrays['fields/lakemask'] >= .5)):
            raise ValueError('target ocean geography unexpectedly includes lake columns')
        threshold = sea_ice_threshold(sources[0]['seam']['identity'])
        fresh_arrays['fields/xland'] = np.where(wet & (fresh_arrays['fields/xice'] < threshold),
                                                2, 1).astype(np.float32)
        roles = {name: np.array(fresh_arrays['fields/'+name]) for name in ('xice',)+OCEAN_ROLE_FIELDS
                 if 'fields/'+name in fresh_arrays}
        isice, iswater = surface_land_use(backend)
        mapped_cache.setdefault('settled_ocean_roles', 0)
        mapped_cache['settled_ocean_roles'] += settle_ocean_roles(roles, wet, threshold, isice, iswater)
        for name, value in roles.items():
            fresh_arrays['fields/'+name] = value
    land_donors,land_receipt=land_state_donors(payload,weights,layout,fresh['seam']['arrays'],target_mesh,mapped_cache)
    if any((b.get('surface_updates') or {}).get('ocean') is not None for b in sources):
        mapped_cache.setdefault('ocean_surface_donors', {})[id(layout)] = land_donors
    prefetch_owner_fields(payload,weights,[key for key in sorted(keys)
                                           if not (ice_fields is not None and key in ice_fields)
                                           and key not in LAND_STATE],
                          mapped_cache,target_mesh=target_mesh)
    for key in sorted(keys):
        if ice_fields is not None and key in ice_fields:
            selected=ice_fields[key][..., layout.cell_l2g];padding=[];categorical=False
        elif key in LAND_STATE:
            land_fields=mapped_cache['land_sources']
            if key not in land_fields:
                land_fields[key],padding=gather_owner_columns([b['seam']['arrays'][key] for b in payload['backends']],
                                                          payload['layouts'],len(weights.source_area))
            else:padding=[]
            selected=np.ascontiguousarray(land_fields[key][...,land_donors]);categorical=True
        else:
            mapped,padding,categorical=mapped_owner_field(payload,weights,key,mapped_cache,target_mesh=target_mesh)
            selected=mapped[..., np.asarray(layout.cell_l2g,dtype=np.int64)]
        if key in fresh['seam']['arrays']:
            target=fresh['seam']['arrays'][key]
            if target.shape[:-1]!=selected.shape[:-1] or target.shape[-1] not in (local,local+1):
                raise ValueError(f'unsupported target physics layout: {key}')
            if target.shape[-1]==local+1:
                # Coupled x-momentum has a periodic alias of its first face.
                # Other pseudo columns remain non-owning duplicate padding.
                alias=selected[...,:1] if key.endswith('_tendencies/ru') else selected[...,-1:]
                selected=np.concatenate((selected,alias),axis=-1)
        else:
            # Existing lazily born ozone/cumulus caches are restart-authorized.
            if key not in ('radiation/o33d_grid','cumulus/w0avg') and key not in SEAICE_FIELDS:
                raise ValueError(f'unknown lazy physics carrier: {key}')
        method='largest-overlap donor' if categorical else 'conservative area overlap'
        if key in LAND_STATE:method='local coherent Noah-MP column donor preserving land/ice/glacier role; prefer overlapping soil and vegetation matches'
        if any(key in pair for pair in VECTOR_PAIRS):method='earth Cartesian overlap, target east/north projection'
        if hasattr(weights, 'layer_intensive') and (key in ATMOSPHERIC_FIELDS or key.startswith(ATMOSPHERIC_PREFIXES)):
            method += '; physical-height atmospheric interpolation'
        if key in STATIC:
            selected=fresh['seam']['arrays'][key]
            method='new geography constructor, persistent dynamical fields carried separately'
        if ice_fields is not None and key in ice_fields:
            method='ocean-only inventory transfer; binary geography mask; consistent ice area and thickness'
        result[key]=selected
        receipt.append(dict(field=key,method=method,source_padding_abs_max=padding,
                            shape=list(selected.shape)))
    if ice_fields is not None:
        # Keep the LSM-facing tile snow consistent with its prognostic owner.
        a=ice_fields['fields/xice'][...,layout.cell_l2g]
        wet=ice_fields['fields/ice_ocean_mask'][...,layout.cell_l2g]>0
        snow=np.divide(ice_fields['fields/ice_snow_mass'][...,layout.cell_l2g],a,
                       out=np.zeros_like(a),where=a>0)
        for key,value in (('fields/snow',snow),('fields/snowh',snow/330)):
            if key in result:
                result[key]=np.where(wet,value,result[key]).astype(result[key].dtype)
                next(item for item in receipt if item['field']==key)['method'] += '; ocean tile snow from carried ice inventory'
    # The driver's held x-face rates depend on local column order. Rebuild
    # them from the carried physical raw PBL rates, using native interpolation.
    if 'pbl/du' in result and 'pbl/dv' in result:
        ru,rv=coupled_column_faces(result['pbl/du'],result['pbl/dv'])
        for key,value in (('driver/pbl_tendencies/ru',ru),('driver/pbl_tendencies/rv',rv)):
            if key in result:
                if result[key].shape!=value.shape:raise ValueError('rebuilt PBL face cache extent differs')
                result[key]=value
                next(item for item in receipt if item['field']==key)['method']='native face interpolation of remapped raw PBL rates'
    scalars=deepcopy(sources[0]['seam']['scalars'])
    for source in sources:
        for key in ('elapsed_seconds','carriers','kf_history_time'):
            # A column seam keeps no Kain-Fritsch clock: compare what the seam holds (absent on all is equal).
            if source['seam']['scalars'].get(key)!=scalars.get(key):
                raise ValueError(f'physics caches have incompatible boundary clocks: {key}')
    step=float(payload['time_seconds'])/(base_dt*rate)
    if step!=int(step):
        raise ValueError('physics boundary does not lie on the new class clock')
    scalars['step_index']=int(step)
    fresh['seam']=dict(identity=fresh['seam']['identity'],arrays=result,scalars=scalars)
    fresh['adapter']=deepcopy(sources[0]['adapter'])
    backend.restore_restart_state(fresh)
    wave_states = [source.get('waves') for source in sources]
    if any(value is not None for value in wave_states):
        if any(value is None for value in wave_states):
            raise ValueError('moving physics owners disagree on wave mode')
        if any(value['clock'] != wave_states[0]['clock'] for value in wave_states):
            raise ValueError('moving physics owners carry different wave clocks')
        backend._wave_move_restart = (wave_states[0], payload['mesh'], weights)
    return dict(fields=receipt,land_state=land_receipt,sea_ice=mapped_cache.get('ocean_ice_receipt'),
                ocean_role_columns_settled=mapped_cache.get('settled_ocean_roles', 0),step_index=int(step),time_seconds=payload['time_seconds'],
                metadata_note='backend call counters retained from coarsest owner class; cadence index rebuilt')


def remap_step_carriers(payload, weights, layouts):
    """Carry physics-to-dynamics feedback without reinitializing accumulators."""
    previous=[]
    if any(item is None for item in payload['previous']):
        if not all(item is None for item in payload['previous']):
            raise ValueError('partial previous-surface carrier')
        previous=[None]*len(layouts)
    else:
        keys=set(payload['previous'][0])
        global_values={key: weights.intensive(gather_owner_columns(
            [item[key] for item in payload['previous']],payload['layouts'],len(weights.source_area))[0]).astype(
                payload['previous'][0][key].dtype) for key in keys}
        previous=[{key: np.ascontiguousarray(value[...,layout.cell_l2g])
                   for key,value in global_values.items()} for layout in layouts]
    forcing=[]
    if any(item is None for item in payload['forcing']):
        if not all(item is None for item in payload['forcing']):
            raise ValueError('partial GF feedback carrier')
        forcing=[None]*len(layouts)
    else:
        global_values={key: atmospheric_intensive(gather_owner_columns(
            [item[key] for item in payload['forcing']],payload['layouts'],len(weights.source_area))[0], weights).astype(
                payload['forcing'][0][key].dtype) for key in ('rthdynten','rqvdynten')}
        forcing=[dict(time_seconds=payload['forcing'][0]['time_seconds'],
                      **{key:np.ascontiguousarray(value[...,layout.cell_l2g])
                         for key,value in global_values.items()}) for layout in layouts]
    # Spectral coefficients and their AR clock are independent of either mesh.
    return dict(previous=previous,forcing=forcing,stochastic=payload.get("stochastic"))
