"""Complete GFL fork convection driver slice, using resident CUDA MPAS pools.

RRFS fork MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F at 1d6e370e:
611-629,668-710,871-914,978-1080,1219-1255,1288-1433.
"""
from . import cu_gfl_bf6a4b0 as gfl

PREPARED = {'u':'u_p','v':'v_p','w':'w_p','t':'t_p','q':'qv_p','rho':'rho_p',
            'p':'pres_hyd_p','pi':'pi_p','p8w':'pres2_hyd_p','dz8w':'dz_p',
            'qc3d':'qc_p','qi3d':'qi_p','maxmf':'maxmfbl_p'}
SURFACE = {'htop':'cutop','hbot':'cubot','xmb_total':'xmb_total','xmb_shallow':'xmb_shallow',
           'k22_shallow':'k22_shallow','kbcon_shallow':'kbcon_shallow','ktop_shallow':'ktop_shallow',
           'vcpool':'vcpool','sig_deep':'sigma_deep','sig_deep_far':'sigma_deep_far',
           'raincv':'raincv','pratec':'cuprec','rainncv':'rainncv','hfx':'hfx','qfx':'qfx',
           'gsw':'gsw','kpbl':'kpbl'}
LEVEL = {'rthcuten':'rthcuten','rqvcuten':'rqvcuten','rqccuten':'rqccuten','rqicuten':'rqicuten',
         'rucuten':'rucuten','rvcuten':'rvcuten','rthften':'rthdynten','rqvften':'rqvdynten',
         'rthblten':'rthblten','rqvblten':'rqvblten'}
DIAG_LEVEL = {'qc_cu':'qc_cu','qi_cu':'qi_cu','cldfrac_cu':'cldfrac_cu','refl10cm_cu':'refl10cm_cu'}
PREPARED_LEVEL = {'rthcuten':'rthcuten_p','rqvcuten':'rqvcuten_p','rqccuten':'rqccuten_p',
                  'rqicuten':'rqicuten_p','rucuten':'rucuten_p','rvcuten':'rvcuten_p',
                  'rthften':'rthdynten_p','rqvften':'rqvdynten_p','rthblten':'rthblten_p',
                  'rqvblten':'rqvblten_p','qc_cu':'qccu_p','qi_cu':'qicu_p',
                  'cldfrac_cu':'cldfrac_cu_p','refl10cm_cu':'refl10cm_cu_p'}
CARRIERS = ('rthcuten','rqvcuten','rucuten','rvcuten')
HALO_FIELDS = ('sigma_deep','sigma_deep_far','sub3d_rucuten','sub3d_rvcuten','sub3d_rthcuten','sub3d_rqvcuten')
HALO_GROUP = {'name':'physics:sub3d_cuten','fields':HALO_FIELDS,'timeLevel':1,'haloLayers':(2,)}


def bind_native_pools(provider,call):
    """RRFS fork MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:668-710,1336-1367.

    The session owner supplies resident halo-aware views for spreading. The
    ordinary solve-only provider cannot represent this driver read boundary.
    Native prepared fields retain their nVertLevels+1 allocation words.
    """
    dims=provider.session.layout.dimensions
    # Whole-entity (halo) views only where the driver reads neighbours: config_gfl_sub3d > 0 spreads subsidence
    # over two rings after the physics:sub3d_cuten exchange (driver_convection.F:1336-1433). With sub3d <= 0 (the
    # Registry default) every read and write is the cell's own, so the owned views every other stage uses serve it.
    # THE BREAKAGE THIS FIXES (hrrrv5 + storm scheme on the resident GPU glue, 2026-10-31): the whole views were
    # always taken, so a word the history/diag streams or physics_get_tend had already bound in the owned extent
    # (vcpool, raincv, the sub3d_* tendencies) refused the step: "cached extent conflict for
    # diag_physics.vcpool level 1: requested (90321,), cached (90320,)".
    neighbours=int(call.options['config_gfl_sub3d'])>0
    if neighbours and getattr(provider,'resident',False):
        raise ValueError('config_gfl_sub3d > 0 (GF-Li neighbour subsidence spreading) is not on the resident GPU '
                         'glue yet: its two-ring reads need the whole-entity words of six fields that the output '
                         'streams and physics_get_tend hold in the owned extent; run config_gfl_sub3d = 0 (the '
                         'Registry default) or the non-resident provider')
    view=provider.resident_pool if neighbours else provider.__getitem__
    return {'prepared':provider['prepared'],'mesh':view('mesh'),
            'sfc_input':view('sfc_input'),'diag_physics':view('diag_physics'),
            'tend_physics':view('tend_physics'),'state':view('state'),
            'nCellsSolve':dims['nCellsSolve'],'nVertLevels':dims['nVertLevels'],
            'dimensions':dims,
            # The native driver reads the core config pool, not physics options.
            # MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:978-981.
            'config_len_disp':provider.session.layout.configs['config_len_disp'],
            'constants':provider['constants'],'scalar_state':provider['scalar_state'],
            'exchange_halo_group':provider.exchange_halo_group}


def from_mpas(pools,call):
    """RRFS fork MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:871-914,978-1080.

    CUDA copies marshal native storage; all driver arithmetic executes kernels.
    Top interface words are copied from the prepared arrays when available and
    never coupled back into native pool arrays with only active levels.
    """
    import cupy as cp
    import numpy as np
    n,nz=int(pools['nCellsSolve']),int(pools['nVertLevels'])
    prepared,mesh,diag,tend=pools['prepared'],pools['mesh'],pools['diag_physics'],pools['tend_physics']
    a={}
    def level(name,array,held):
        """Marshal MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:978-1080."""
        result=cp.empty((n,nz+1),dtype=cp.float32)
        result[:,:nz]=array[:nz,:n].T
        result[:,nz]=held[nz,:n]
        a[name]=result
    for name,source in PREPARED.items():
        value=prepared[source]
        if name=='maxmf':a[name]=cp.ascontiguousarray(value[:n])
        else:a[name]=cp.ascontiguousarray(value[:nz+1,:n].T)
    for name,source in SURFACE.items():a[name]=cp.ascontiguousarray(diag[source][:n]).copy()
    for name,source in LEVEL.items():level(name,tend[source],prepared[PREPARED_LEVEL[name]])
    for name,source in DIAG_LEVEL.items():level(name,diag[source],prepared[PREPARED_LEVEL[name]])
    # Native allocation resets active sub3d fields on every convection call.
    # mpas_atmphys_driver_convection.F:248-263.
    for name in CARRIERS:
        field='sub3d_'+name
        level(field,cp.zeros((nz,n),dtype=cp.float32),prepared[field+'_p'])
    a['gdc']=a['qc_cu'];a['gdc2']=a['qi_cu']
    a['areacell']=cp.ascontiguousarray(mesh['areaCell'][:n]);a['ht']=cp.ascontiguousarray(mesh['zgrid'][0,:n])
    a['xland']=cp.ascontiguousarray(pools['sfc_input']['xland'][:n]);a['bilbc']=cp.ascontiguousarray(mesh['bdyMaskCell'][:n])
    a['dxcell']=cp.empty(n,dtype=cp.float32)
    k=gfl._module(cp.cuda.runtime.getDevice())
    if n:k.get_function('gfl_driver_dx')(((n+127)//128,),(128,),
        (np.int32(n),np.float32(pools['config_len_disp']),mesh['meshDensity'],a['dxcell']))
    a['rthraten']=cp.empty((n,nz+1),dtype=cp.float32)
    a['rthraten'][:,nz]=prepared['rthraten_p'][nz,:n]
    if n:k.get_function('gfl_driver_radiation')(((n*nz+127)//128,),(128,),
        (np.int32(n),np.int32(nz),np.int32(tend['rthratenlw'].shape[1]),tend['rthratenlw'],tend['rthratensw'],a['rthraten']))
    scalar=pools['scalar_state'];const=pools['constants'];num_chem=int(scalar['num_chem'])
    values={'itimestep':call.itimestep,'dt':scalar['dt_dyn'],'xlv':const['xlv'],'cp':const['cp'],
            'g':const['gravity'],'r_v':const['R_v'],'ichoice_deep':call.options['config_gfconv_closure_deep'],
            'ichoice_shallow':call.options['config_gfconv_closure_shallow'],'ishallow_g3':1,
            'sub3d':max(0,int(call.options['config_gfl_sub3d'])),'num_chem':num_chem}
    for name,value in values.items():a[name]=cp.asarray([value],dtype=cp.int32 if name in gfl.INTEGER_FIELDS else cp.float32)
    if num_chem:
        # The caller's chemistry slice is the native chemistry_start:end view.
        dims=pools['dimensions']
        chemistry=pools['state']['scalars'][int(dims['chemistry_start'])-1:int(dims['chemistry_end'])]
        a['chem3d_in']=cp.empty((n,num_chem,nz+1),dtype=cp.float32)
        a['chem3d_in'][...,:nz]=chemistry[:num_chem,:nz,:n].transpose(2,0,1)
        a['chem3d_in'][...,nz]=pools['prepared']['chem_p'][:num_chem,nz,:n].T
    else:a['chem3d_in']=cp.empty((n,0,nz+1),dtype=cp.float32)
    return a


def to_mpas(a,pools,call):
    """RRFS fork MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:1219-1255,1288-1433."""
    import cupy as cp
    n,nz=int(pools['nCellsSolve']),int(pools['nVertLevels'])
    diag,tend=pools['diag_physics'],pools['tend_physics']
    for name in ('raincv','pratec'):diag[SURFACE[name]][:n]=a[name]
    for name in ('rthcuten','rqvcuten','rqccuten','rqicuten','rucuten','rvcuten'):tend[name][:nz,:n]=a[name][:,:nz].T
    for name in ('htop','hbot','xmb_total','xmb_shallow','k22_shallow','kbcon_shallow','ktop_shallow','vcpool','sig_deep','sig_deep_far'):
        diag[SURFACE[name]][:n]=a[name]
    # Native writes back ktop_deep_p unchanged, and does not couple qc3d/qi3d here.
    for name,source in DIAG_LEVEL.items():diag[source][:nz,:n]=a[name][:,:nz].T
    for name in CARRIERS:tend['sub3d_'+name][:nz,:n]=a['sub3d_'+name][:,:nz].T
    if int(pools['scalar_state']['num_chem']):
        dims=pools['dimensions']
        chemistry=pools['state']['scalars'][int(dims['chemistry_start'])-1:int(dims['chemistry_end'])]
        chemistry[:,:nz,:n]=a['chem3d_in'][...,:nz].transpose(1,2,0)
    # Exchange the actual resident carriers before packing neighbor words.
    option=int(call.options['config_gfl_sub3d'])
    if option>0:pools['exchange_halo_group'](HALO_GROUP['name'])
    values=cp.ascontiguousarray(cp.stack([tend[name][:nz,:].T for name in CARRIERS]))
    subs=cp.ascontiguousarray(cp.stack([tend['sub3d_'+name][:nz,:].T for name in CARRIERS]))
    mesh={name:cp.ascontiguousarray(pools['mesh'][name].T) if name in ('cellsOnCell','cellsOnCellsOnCell')
          else pools['mesh'][name] for name in ('cellsOnCell','cellsOnCellsOnCell','nEdgesOnCell',
                                               'nCellsOnCellsOnCell','bdyMaskCell')}
    gfl._spread_after_halo(values,subs,mesh=mesh,sigma_deep=diag['sigma_deep'],
        sigma_deep_far=diag['sigma_deep_far'],config_gfl_sub3d=option,nCellsSolve=n)
    for index,name in enumerate(CARRIERS):tend[name][:nz,:n]=values[index,:n,:].T


def run_slot(call):
    """RRFS fork MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:611-629,668-710,871-1433."""
    pools=call.pools;args=from_mpas(pools,call)
    # Native driver clears activity before invoking every convection scheme.
    pools['prepared']['cu_act_flag'][:int(pools['nCellsSolve'])].fill(False)
    result=gfl.launch_columns(args,ncol=int(pools['nCellsSolve']),nz=int(pools['nVertLevels']),
        pbl_scheme=str(call.options['config_pbl_scheme']),row_dx_first=args['dxcell'][:1])
    to_mpas(args,pools,call)
    return result
