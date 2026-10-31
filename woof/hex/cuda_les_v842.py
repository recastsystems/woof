"""Resident LES frame at the MPAS RK boundary.

Native ordering: mpas_atm_time_integration.F:6331-6390,6581-6606,
6724-6801,6881-6905. Array permutation/padding is storage-only.
"""
from collections import Counter

import numpy as np

from .mixing_v842_les import (LesDissipationV842, ROUTINES, _extent,
    les_boundary_arguments, les_model_from_string, les_surface_from_string)


class ResidentLesV842(LesDissipationV842):
    """Own native-layout scratch and retain RK1 tendencies through RK2/3."""

    def __init__(self, mesh, vertical, profiles, config, scalar_names, t_init):
        # MPAS v8.4.2 adds the floor; the NOAA v8.4.1 route predates it.
        super().__init__(prognostic_viscosity_floor=getattr(config, 'prognostic_les_viscosity_floor', 0.01) != 0.0)
        import cupy as cp
        from .mixing import compute_mesh_mixing_scaling, resolve_config_len_disp
        from .mixing_v841 import initialize_deformation_weights_v841

        self.cp, self.config = cp, config
        self.names = tuple(scalar_names)
        self.counts = Counter()
        a = mesh.arrays
        nc, ne, nv = mesh.n_cells, mesh.n_edges, mesh.n_vertices
        nl = vertical.n_vert_levels
        f = self.frame = dict(ncells=nc, nedges=ne, nvertices=nv,
            nvertlevels=nl, maxedges=a['edgesOnCell'].shape[1],
            vertexdegree=a['edgesOnVertex'].shape[1], num_scalars=len(self.names),
            num_scalars_dummy=len(self.names), cellstart=1, cellend=nc,
            cellsolvestart=1, cellsolveend=nc, edgestart=1, edgeend=ne,
            edgesolvestart=1, edgesolveend=ne, vertexstart=1, vertexend=nv,
            dynamics_substep=1, les_model_opt=les_model_from_string(config.config_les_model),
            les_surface_opt=les_surface_from_string(config.config_les_surface),
            mix_scalars=int(config.config_mix_scalars),
            config_mix_full=int(config.config_mix_full),
            c_s=np.float32(config.config_smagorinsky_coef), prandtl_inv=np.float32(1),
            config_len_disp=resolve_config_len_disp(mesh, config.config_len_disp, dtype=np.float32),
            v_mom_eddy_visc2=np.float32(config.config_v_mom_eddy_visc2),
            v_theta_eddy_visc2=np.float32(config.config_v_theta_eddy_visc2),
            dt=np.float32(config.config_dt), invdt=np.float32(1)/np.float32(config.config_dt))
        for key in ('config_visc4_2dsmag','config_del4u_div_factor',
                    'config_surface_drag_coefficient','config_surface_heat_flux',
                    'config_surface_moisture_flux'):
            f[key] = np.float32(getattr(config,key))
        for name in ('qv','qc','tke'):
            f['index_'+name] = self.names.index(name)+1 if name in self.names else 0
        if f['les_model_opt'] == 2 and not f['index_tke']:
            raise ValueError('prognostic LES requires a transported TKE scalar carrier')

        schema = les_boundary_arguments()
        for key, spec in schema.items():
            if spec['dims']:
                dims = [str(nc+1) if d == ':' else d for d in spec['dims']]
                shape = tuple(_extent(d, f) for d in dims)
                dtype = cp.float32 if spec['type']=='float' else cp.int32
                f[key] = cp.zeros(shape, dtype=dtype, order='F')
            elif spec['type']=='float' and spec['intent']!='in':
                f[key] = cp.zeros(1, dtype=cp.float32)
        self.initialized = set(key for key,spec in schema.items() if not spec['dims'])

        # mesh.py stores zero-based connectivity. Native reserves entity n+1.
        for key, name, count in (
            ('edgesoncell','edgesOnCell',ne),('cellsonedge','cellsOnEdge',nc),
            ('verticesonedge','verticesOnEdge',nv),('edgesonvertex','edgesOnVertex',ne)):
            raw = np.asarray(a[name])
            self.set(key, np.where(raw<0,count+1,raw+1).astype(np.int32).T)
        self.set('nedgesoncell', np.asarray(a['nEdgesOnCell'],np.int32))
        for key,name in (('angleedge','angleEdge'),('dcedge','dcEdge'),('dvedge','dvEdge')):
            self.set(key,np.asarray(a[name],np.float32))
        for key,name in (('invareacell','areaCell'),('invareatriangle','areaTriangle'),
                         ('invdcedge','dcEdge'),('invdvedge','dvEdge')):
            self.set(key,np.float32(1)/np.asarray(a[name],np.float32))
        eoc, eov = np.asarray(a['edgesOnCell']), np.asarray(a['edgesOnVertex'])
        cell_sign = np.zeros(eoc.shape,np.float32)
        valid = np.arange(eoc.shape[1])[None, :] < np.asarray(a['nEdgesOnCell'])[:, None]
        cell_sign[valid] = np.where(
            np.asarray(a['cellsOnEdge'])[eoc[valid], 0] ==
            np.broadcast_to(np.arange(nc)[:, None], eoc.shape)[valid], 1, -1)
        vertex_sign = np.where(a['verticesOnEdge'][eov,1]==np.arange(nv)[:,None],1,-1)
        self.set('edgesoncell_sign',cell_sign.T)
        self.set('edgesonvertex_sign',vertex_sign.astype(np.float32).T)
        scaling=compute_mesh_mixing_scaling(mesh,config_h_ScaleWithMesh=config.config_h_ScaleWithMesh,dtype=np.float32)
        self.set('meshscalingdel2',scaling.del2)
        self.set('meshscalingdel4',scaling.del4)
        weights=initialize_deformation_weights_v841(mesh,dtype=np.float32)
        self.deformation_weights = weights
        for suffix in ('c2','s2','cs','c','s'):
            self.set('deformation_coef_'+suffix,getattr(weights,'coef_'+suffix).T)
        for name in ('zgrid','zz','rdzu','rdzw','fzm','fzp'):
            value=np.asarray(getattr(vertical,name),np.float32)
            self.set(name,value[:nl] if value.ndim==1 else value)
        self.set('u_init',profiles.u_init)
        self.set('v_init',profiles.v_init)
        self.set('t_init',t_init)
        # Native scratch and default-off surface buffers begin at zero.
        for name in ('eddy_visc_horz','eddy_visc_vert','bn2','prandtl_3d_inv',
                     'delsq_u','delsq_vorticity','delsq_divergence','delsq_w','delsq_theta',
                     'tend_u_euler','tend_w_euler','tend_theta_euler','ustm','hfx','qfx'):
            self.initialized.add(name)
        self.frame['bv_freq2']=self.frame['bn2']
        self.frame['ureconstructzonal']=self.frame['ucell']
        self.frame['ureconstructmeridional']=self.frame['vcell']

    def set(self, name, value):
        """Copy actual words into native storage; leave its sentinel untouched."""
        target=self.frame[name]
        value=self.cp.asarray(value,dtype=target.dtype)
        if value.shape == target.shape:
            target[...] = value
        elif value.shape == target.shape[:-1]+(target.shape[-1]-1,):
            target[...,:-1] = value
        else:
            raise ValueError(f'LES {name}: {value.shape} cannot fill {target.shape}')
        self.initialized.add(name)

    def dispatch(self, routine):
        """Count completed launches, not requested steps."""
        args=ROUTINES[routine]['args']
        missing=set(args)-self.initialized
        if missing:
            raise RuntimeError(f'LES {routine} has unbound fields: {sorted(missing)}')
        self.call(routine,**{key:self.frame[key] for key in args})
        self.counts[routine]+=1

    def begin(self, driver, state, saved, diagnostics, moist, sources, substep, winds):
        """MPAS time_integration.F:6331-6390, before pressure seed."""
        f=self.frame
        f['dynamics_substep']=substep
        for key,value in (
            ('u',saved.normal_velocity),('v',diagnostics.tangential_velocity),
            ('w',saved.vertical_velocity),('theta_m',saved.theta_m),('exner',saved.exner),
            ('pressure_b',driver.atmosphere.reference.pressure_base),
            ('pp',saved.pressure_perturbation),('rho_zz',state.rho),
            ('rho_edge',diagnostics.h_edge),('divergence',diagnostics.divergence),
            ('vorticity',diagnostics.vorticity),('scalars',state.scalars),
            ('qtot',moist.qtot),('ucell',winds.u_p),('vcell',winds.v_p)):
            self.set(key,value)
        self.initialized.update(('bv_freq2','ureconstructzonal','ureconstructmeridional'))
        if substep == 1:
            self.set('tend_scalars',sources.scalars)
        f['tend_u_euler'].fill(0)
        if f['les_model_opt']:
            self.dispatch('calculate_n2')
            self.dispatch('les_models')
        elif self.config.config_horiz_mixing == '2d_smagorinsky':
            from dataclasses import fields
            from .mixing_v841 import V841MixingConfig
            horizontal = driver.horizontal
            if getattr(horizontal, '_deformation_v841', None) is None:
                horizontal.attach_deformation_weights_v841(self.deformation_weights)
            cfg = V841MixingConfig(**{
                field.name: getattr(self.config, field.name)
                for field in fields(V841MixingConfig)
            })
            kdiff, h4, _, _, _ = horizontal.compute_smagorinsky_coefficients_v841(
                saved.normal_velocity, diagnostics.tangential_velocity,
                dt=self.config.config_dt, config=cfg,
            )
            self.set('eddy_visc_horz', kdiff)
            f['h_mom_eddy_visc4'].fill(h4)
            f['h_theta_eddy_visc4'].fill(h4)
            self.counts['smagorinsky_2d'] += 1
        else:
            # Native 2d_fixed branch, time_integration.F:6344-6358.
            f['eddy_visc_horz'][:,:-1].fill(self.config.config_h_theta_eddy_visc2)
            f['h_mom_eddy_visc4'].fill(self.config.config_h_mom_eddy_visc4)
            f['h_theta_eddy_visc4'].fill(self.config.config_h_theta_eddy_visc4)
            self.counts['fixed_2d'] += 1
        # Native clamps negative TKE in les_models before transport.
        state.scalars[...] = f['scalars'][...,:-1]
        coef=self.config.config_mpas_cam_coef
        if coef>0:
            n=getattr(self.config,'config_number_cam_damping_levels',4)  # as cuda_driver's 2-D path
            # Native literals 4.0*2.0833*len*coef, then level taper.
            base=np.float32(np.float32(np.float32(4)*np.float32(2.0833))*f['config_len_disp'])*np.float32(coef)
            for k in range(f['nvertlevels']-n,f['nvertlevels']):
                value=base*(np.float32(1)-np.float32(f['nvertlevels']-k-1)/np.float32(n))
                f['eddy_visc_horz'][k,:-1]=self.cp.maximum(f['eddy_visc_horz'][k,:-1],value)

    def tendencies(self, seed_u, sources):
        """Seed u first; retain separate w mixing before pressure subtraction."""
        self.set('tend_u_euler',seed_u)
        for routine in ('u_dissipation_3d','w_dissipation_3d','scalar_dissipation_3d_les'):
            self.dispatch(routine)
        sources.scalars[...] = self.frame['tend_scalars'][...,:-1]
        return tuple(self.cp.ascontiguousarray(self.frame[key][...,:-1])
                     for key in ('tend_u_euler','tend_w_euler','tend_theta_euler'))

    def receipt(self, completed_steps):
        return dict(backend='hex-cuda',model=self.config.config_les_model,
            completed_steps=completed_steps,routine_call_counts=dict(self.counts))
