"""Native hydrostatic spherical RK3 integration with resident private storage.

The longitude FFTs, pole prescriptions, directional metrics and acoustic
transitions are WRF's global equations. A CUDA graph owns the entire step;
Python constructs its launch schedule once, before the forecast clock starts.
"""
import numpy as np


class GlobalHydrostaticStepper:
    def __init__(self, fields, *, dt, sound_steps=4, filter_lat=45.,
                 smdiv=.1, emdiv=.01, epssm=.1, hypsometric_opt=1,
                 v_mom_order=3, v_sca_order=3, phi_adv_z=1, lw=31, owner=None):
        import cupy as cp
        from woof.core.global_geometry import GlobalGeometry
        from woof.core.global_advection import GlobalAdvection
        from woof.core.global_acoustic import GlobalAcousticTransport
        from woof.core.global_hydrostatic import (HydrostaticDiagnostics,
            HydrostaticAcoustic, HydrostaticVerticalVelocity)
        from woof.core.global_polar import PoleFaces, PolarFFT
        from woof.core.global_rotation import GlobalRotation
        from woof.core.kernels import load_module_no_ftz
        nz, ny, nx = fields['t'].shape
        if sound_steps < 2 or sound_steps % 2 or not np.isfinite(dt) or dt <= 0:
            raise ValueError('global RK3 requires a positive dt and an even acoustic count >=2')
        if lw not in (0,31):
            raise ValueError('global dry step requires LW 0 or 31; other physics needs its global driver binding')
        self.fields, self.dims = fields, tuple(np.int32(v) for v in (nx,ny,nz))
        self.dt, self.sound_steps, self.lw = float(dt), sound_steps, lw
        self.v_mom_order, self.v_sca_order, self.phi_adv_z = v_mom_order,v_sca_order,phi_adv_z
        self.elapsed_seconds = 0.
        def allocate(shape,name,dtype=cp.float32):
            return (cp.zeros(shape,dtype) if owner is None else
                    owner.scratch(shape,'global_rk_'+name,dtype=dtype))
        def resident(value,name):
            result=allocate(value.shape,name,value.dtype);result[...]=value
            return result
        profiles = lambda names: resident(cp.stack([fields[n] for n in names.split()]),'profiles_'+names.replace(' ','_'))
        # All scalars are copied at setup, never inside a captured step.
        scalars = cp.asnumpy(fields['scalars'])
        self.ptop, dx, dy, cf1, cf2, cf3, cfn, cfn1 = map(float,scalars)
        self.spacing = np.float32(1)/np.float32(dx), np.float32(1)/np.float32(dy)
        def u_metric(source):
            result=allocate((ny,nx+1),'metric_'+('ux' if source is fields['tx'] else 'uy'))
            result[:,:nx]=source;result[:,nx]=source[:,0]
            return result
        if 'ux' not in fields:fields['ux']=u_metric(fields['tx'])
        if 'uy' not in fields:fields['uy']=u_metric(fields['ty'])
        self.metrics=tuple(fields[n] for n in ('tx','ty','ux','uy','vx','vy','vxinv'))
        p8=profiles('c1h c2h c1f c2f dnw fnm fnp rdnw')
        self.geometry=GlobalGeometry(nx,ny,nz,profiles=p8,metrics=self.metrics,
            rdx=self.spacing[0],rdy=self.spacing[1],cfn=cfn,cfn1=cfn1)
        self.advection=GlobalAdvection(nx,ny,nz,profiles=profiles('rdnw fnm fnp'))
        self.transport=GlobalAcousticTransport(nx,ny,nz,
            profiles=profiles('c1h c2h dnw fnm fnp rdnw'),metrics=self.metrics,
            rdx=self.spacing[0],rdy=self.spacing[1],emdiv=emdiv,epssm=epssm)
        self.hydro=HydrostaticDiagnostics(nx,ny,nz,
            constants=resident(cp.asarray((100000,300,self.ptop,287,np.float32(461.6)/np.float32(287),
                                  -np.float32(717.5)/np.float32(1004.5)),cp.float32),'hydro_constants'),
            profiles=profiles('c1h c2h c3h c4h c3f c4f dnw rdnw rdn'),
            use_theta_m=0,hypsometric_opt=hypsometric_opt)
        self.acoustic_hydro=HydrostaticAcoustic(nx,ny,nz,
            profiles=resident(cp.stack([fields[n][:nz] for n in ('c1h','c2h','c3h','dnw')]),'acoustic_hydro_profiles'),smdiv=smdiv)
        self.w_operators=[HydrostaticVerticalVelocity(nx,ny,nz,
            constants=resident(cp.asarray((9.81,dt*factor,cf1,cf2,cf3,*self.spacing),cp.float32),'w_constants_'+str(factor)),
            profiles=profiles('c1f c2f')) for factor in (1/3,.5,1)]
        self.poles=PoleFaces(nx,ny,nz)
        # Each distinct geometry owns a plan and work buffers. Reusing a
        # plan sequentially within one graph does not alias member storage.
        self.filters={}
        def fft_allocator(kind):
            return lambda shape,name,dtype: allocate(shape,'fft_'+kind+'_'+name,dtype)
        self.filters['scalar']=PolarFFT(nx,ny,nz,latitude=fields['lat'],filter_lat=filter_lat,allocator=fft_allocator('scalar'))
        self.filters['u']=PolarFFT(nx,ny,nz,latitude=fields['lat'],filter_lat=filter_lat,staggered_x=True,allocator=fft_allocator('u'))
        vlat=allocate((ny+1,nx),'vlat');vlat[:ny]=fields['lat']
        # clat's northern extra row is zero in the native allocation;
        # the real module capture witnesses it. It is not a mass latitude.
        vlat[ny]=0
        self.filters['v']=PolarFFT(nx,ny+1,nz,latitude=vlat,filter_lat=filter_lat,dvlat=np.float32(90)/np.float32(ny),allocator=fft_allocator('v'))
        self.filters['w']=PolarFFT(nx,ny,nz+1,latitude=fields['lat'],filter_lat=filter_lat,allocator=fft_allocator('w'))
        self.filters['mass']=PolarFFT(nx,ny,1,latitude=fields['lat'],filter_lat=filter_lat,allocator=fft_allocator('mass'))
        self.work={}
        shapes={'s':(nz,ny,nx),'u':(nz,ny,nx+1),'v':(nz,ny+1,nx),
                'w':(nz+1,ny,nx),'m':(ny,nx),'mu':(ny,nx+1),'mv':(ny+1,nx)}
        def work_array(name,kind):
            self.work[name]=allocate(shapes[kind],name)
        for name,kind in (('ru','u'),('rv','v'),('rw','w'),('ww','w'),('divv','s'),
            ('muu','mu'),('muv','mv'),('mut','m'),('muts','m'),('muus','mu'),('muvs','mv'),
            ('mu0','m'),('u0','u'),('v0','v'),('w0','w'),('t0','s'),('ph0','w'),
            ('us','u'),('vs','v'),('ws','w'),('ts','s'),('phs','w'),('mus','m'),('ww1','w'),
            ('c2a','s'),('pm1','s'),('ru_tend','u'),('rv_tend','v'),('rw_tend','w'),
            ('t_tend','s'),('ph_tend','w'),('mu_tend','m'),('muave','m'),('mudf','m'),
            ('t_ave','s'),('dummy_mass','m'),('exner','s'),('temperature','s'),('ps','m'),
            ('heat','s'),('held','s')):
            work_array(name,kind)
        if 'rthraten' in fields:self.work['heat']=fields['rthraten']
        self.work['moist']=allocate((0,nz,ny,nx),'moist')
        self.work['height_halo']=allocate((ny+2,nx+2),'height_halo')
        for name,levels in (('p',nz),('pb',nz),('al',nz),('alt',nz),('ph',nz+1),('ts',nz)):
            self.work[name+'_halo']=allocate((levels,ny+2,nx+2),name+'_halo')
        self.work['mudf_halo']=allocate((ny+2,nx+2),'mudf_halo')
        for name,levels,rows in (('ru',nz,ny),('rv',nz,ny+1),('ww',nz+1,ny),('q',nz,ny),('qv',nz,ny+1)):
            self.work[name+'_zonal']=allocate((levels,rows,nx+7),name+'_zonal')
        self.work['rotation_halo']=allocate((4*ny+nz*(ny+1)+(nz+1)*ny,),'rotation_halo')
        sina=allocate((ny,nx),'sina');sina.fill(0)
        cosa=allocate((ny,nx),'cosa');cosa.fill(1)
        self.rotation=GlobalRotation(nx,ny,nz,metrics=self.metrics[:6],
            geography=(fields['f'],fields['e'],sina,cosa,fields['lat']),
            profiles=profiles('fnm fnp'),zonal_halo=self.work['rotation_halo'])
        module=load_module_no_ftz('global_geometry', no_fmad=True)
        self.physics_prepare=module.get_function('global_physics_prepare')
        self.hold_heating=module.get_function('global_hold_heating')
        self.add_held=module.get_function('global_add_held')
        forcing=load_module_no_ftz('idealized_forcing', no_fmad=True)
        self.relax_temperature=forcing.get_function('relax_temperature')
        self.relax_momentum=forcing.get_function('relax_momentum')
        self.graph=None

    def _diagnose(self):
        f,w=self.fields,self.work
        self.hydro.apply(mu=f['mu'],muts=w['muts'],alb=f['alb'],pb=f['pb'],t=f['t'],
            phb=f['phb'],moist=w['moist'],p=f['p'],al=f['al'],ph=f['ph'],
            full_pressure=f['pressure'],alt=f['alt'])

    def _acoustic_diagnose(self,first):
        f,w=self.fields,self.work
        self.acoustic_hydro.apply(mu=f['mu'],mut=w['muts'],alt=f['alt'],
            theta_coupled=f['t'],theta_ref=w['ts'],c2a=w['c2a'],p=f['p'],al=f['al'],
            ph=f['ph'],pm1=w['pm1'],first=first)

    def _filter(self,name,field):
        self.filters[name].apply(field[None] if field.ndim==2 else field)

    def _enqueue(self):
        import cupy as cp
        f,w=self.fields,self.work
        nx,ny,nz=map(int,self.dims);mp=nx*ny
        for stage,(factor,substeps) in enumerate(((1/3,1),(.5,self.sound_steps//2),(1,self.sound_steps))):
            self.poles.apply(f['v']);self.poles.apply(w['v0'])
            self.geometry.mass_faces(f['mu'],f['mub'],w['mut'],w['muu'],w['muv'])
            self.geometry.couple(muu=w['muu'],muv=w['muv'],mut=w['mut'],u=f['u'],v=f['v'],w=f['w'],
                ru=w['ru'],rv=w['rv'],rw=w['rw'])
            self.geometry.omega(mu=f['mu'],mub=f['mub'],u=f['u'],v=f['v'],divv=w['divv'],ww=w['ww'])
            cp.add(f['al'],f['alb'],out=f['alt'])
            cp.add(f['p'],f['pb'],out=f['pressure'])
            self.poles.apply(f['v'],copy_neighbor=True)
            self.poles.apply(w['v0'],copy_neighbor=True)
            for name in ('ru_tend','rv_tend','rw_tend','t_tend','ph_tend','mu_tend'):
                w[name].fill(0)
            if stage==0:
                self.physics_prepare(((nz*mp+127)//128,),(128,),(*self.dims,
                    np.float32(287)/np.float32(1004.5),f['p'],f['pb'],f['ph'],f['phb'],f['t'],
                    f['pressure'],w['exner'],w['temperature'],w['ps']))
                w['heat'].fill(0)
                if self.lw==31:
                    self.relax_temperature(((nz*mp+255)//256,),(256,),(*self.dims,
                        f['lat'],f['pressure'],w['ps'],w['exner'],w['temperature'],
                        np.float32(np.pi)/np.float32(180),np.float32(287),np.float32(1004.5),w['heat']))
                self.hold_heating(((nz*mp+127)//128,),(128,),(*self.dims,
                    f['c1h'],f['c2h'],w['mut'],w['heat'],w['held']))
            for name,stagger in (('ru',True),('rv',False),('ww',False)):
                self.advection.pack(w[name],w[name+'_zonal'],stagger_x=stagger)
            for kind,name,metric,tendency in (('scalar','t','tx','t_tend'),('u','u','ux','ru_tend'),('v','v','vy','rv_tend')):
                q=w['qv_zonal'] if kind=='v' else w['q_zonal']
                self.advection.pack(f[name],q,stagger_x=kind=='u')
                self.advection.apply(kind=kind,q=q,ru=w['ru_zonal'],rv=w['rv_zonal'],ww=w['ww_zonal'],
                    metric=f[metric],vx=f['vx'],tendency=w[tendency],rdx=self.spacing[0],rdy=self.spacing[1],
                    vorder=self.v_sca_order if kind=='scalar' else self.v_mom_order)
            self.geometry.rhs_phi(order=5,phi_adv_z=self.phi_adv_z,muu=w['muu'],muv=w['muv'],
                u=f['u'],v=f['v'],ww=w['ww'],ph=f['ph'],phb=f['phb'],tendency=w['ph_tend'])
            self.geometry.pressure(muu=w['muu'],muv=w['muv'],ph=f['ph'],alt=f['alt'],p=f['p'],pb=f['pb'],al=f['al'],
                ru_tend=w['ru_tend'],rv_tend=w['rv_tend'])
            self.rotation.apply(ru=w['ru'],rv=w['rv'],rw=w['rw'],u=f['u'],v=f['v'],
                ru_tend=w['ru_tend'],rv_tend=w['rv_tend'],rw_tend=w['rw_tend'])
            if self.lw==31:
                self.transport.pack(f['p'],w['p_halo']);self.transport.pack(f['pb'],w['pb_halo'])
                for axis,name in enumerate(('ru','rv')):
                    self.relax_momentum(((w[name].size+255)//256,),(256,),(*self.dims,
                        np.int32(axis),np.int32(nx+1 if axis==0 else ny),w['p_halo'],w['pb_halo'],w[name],w[name+'_tend']))
            self.add_held(((nz*mp+127)//128,),(128,),(np.int32(nz*mp),f['ty'],w['held'],w['t_tend'],np.int32(mp)))
            if stage==0:
                for name in ('u','v','w','t','ph','mu'):
                    w[name+'0'][...]=f[name]
                w['muus'][...]=w['muu'];w['muvs'][...]=w['muv'];w['mudf'].fill(0)
            else:
                self.geometry.mass_faces(w['mu0'],f['mub'],w['dummy_mass'],w['muus'],w['muvs'],mode=2)
            self.geometry.prepare_mass(mub=f['mub'],mu0=w['mu0'],mu=f['mu'],mu_save=w['mus'],muts=w['muts'])
            w['ww1'][...]=w['ww']
            self.geometry.prepare(muu=w['muu'],muv=w['muv'],mut=w['mut'],muus=w['muus'],muvs=w['muvs'],muts=w['muts'],
                u0=w['u0'],v0=w['v0'],w0=w['w0'],t0=w['t0'],ph0=w['ph0'],pb=f['pb'],p=f['p'],alt=f['alt'],
                u=f['u'],v=f['v'],w=f['w'],t=f['t'],ph=f['ph'],us=w['us'],vs=w['vs'],ws=w['ws'],ts=w['ts'],phs=w['phs'],c2a=w['c2a'])
            self._acoustic_diagnose(True)
            self.transport.pack(w['ts'],w['ts_halo']);self.transport.pack(f['alt'],w['alt_halo'])
            self.transport.pack(f['pb'],w['pb_halo'])
            subdt=np.float32(self.dt*factor/substeps)
            for _ in range(substeps):
                for name in ('p','al','ph'):
                    self.transport.pack(f[name],w[name+'_halo'])
                self.transport.pack(w['mudf'],w['mudf_halo'])
                self.transport.advance_uv(dt=subdt,muu=w['muu'],muv=w['muv'],mudf=w['mudf_halo'],
                    ph=w['ph_halo'],alt=w['alt_halo'],p=w['p_halo'],pb=w['pb_halo'],al=w['al_halo'],
                    ru_tend=w['ru_tend'],rv_tend=w['rv_tend'],u=f['u'],v=f['v'])
                self._filter('v',f['v']);self._filter('u',f['u'])
                self.transport.advance_mass_theta(dt=subdt,muu=w['muu'],muv=w['muv'],mut=w['mut'],
                    u=f['u'],v=f['v'],u_ref=w['us'],v_ref=w['vs'],theta_ref=w['ts_halo'],ww_ref=w['ww1'],
                    theta_tend=w['t_tend'],mu_tend=w['mu_tend'],mu=f['mu'],muts=w['muts'],muave=w['muave'],
                    mudf=w['mudf'],ww=w['ww'],theta=f['t'],theta_ave=w['t_ave'],divv=w['divv'])
                self._filter('scalar',f['t']);self._filter('mass',f['mu'])
                cp.add(w['mut'],f['mu'],out=w['muts'])
                self._acoustic_diagnose(False)
            self.geometry.mass_faces(w['muts'],f['mub'],w['dummy_mass'],w['muus'],w['muvs'],mode=1)
            self.geometry.finish(muu=w['muu'],muv=w['muv'],mut=w['mut'],muus=w['muus'],muvs=w['muvs'],muts=w['muts'],
                us=w['us'],vs=w['vs'],ws=w['ws'],ts=w['ts'],phs=w['phs'],mus=w['mus'],ww1=w['ww1'],
                u=f['u'],v=f['v'],w=f['w'],t=f['t'],ph=f['ph'],mu=f['mu'],ww=w['ww'])
            for name,kind in (('v','v'),('u','u'),('t','scalar'),('mu','mass'),('w','w'),('ph','w')):
                self._filter(kind,f[name])
            self._filter('mass',w['mut'])
            self._diagnose()
            if stage<2:
                self.w_operators[stage].apply(mut=w['muts'],msftx=f['tx'],msfty=f['ty'],height_halo=w['height_halo'],
                    u=f['u'],v=f['v'],ph_tend=w['ph_tend'],ph_new=f['ph'],ph_old=w['ph0'],w=f['w'])
        self._diagnose()
        self.w_operators[2].apply(mut=w['muts'],msftx=f['tx'],msfty=f['ty'],height_halo=w['height_halo'],
            u=f['u'],v=f['v'],ph_tend=w['ph_tend'],ph_new=f['ph'],ph_old=w['ph0'],w=f['w'])

    def capture(self):
        """Warm libraries and bind one graph without advancing the initial state."""
        import cupy as cp
        names=('u','v','w','t','ph','mu','p','al','alt','pressure')
        saved={name:self.fields[name].copy() for name in names}
        saved_heat=self.work['heat'].copy()
        self._enqueue()
        for name in names:
            self.fields[name][...]=saved[name]
        self.work['heat'][...]=saved_heat
        cp.cuda.get_current_stream().synchronize()
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture()
            self._enqueue()
            self.graph=stream.end_capture()

    def step(self):
        if self.graph is None:
            raise RuntimeError('global step graph must be constructed before advancing')
        self.graph.launch()
        self.elapsed_seconds+=self.dt
