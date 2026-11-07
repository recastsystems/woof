"""GPU finite-volume geometry for lat/lon and orthogonal projected mass grids.

Coordinates and rotations are earth relative. Compaction, geometry and stencil
preparation execute on CUDA. Only validation/bounds synchronize at construction.
"""
from pathlib import Path
import math
import numpy as np


class StructuredBackend:
    def __init__(self, grid, bathymetry, *, kind="latlon", map_factor=None,
                 gse_spreading_seconds=3600., spectral=None):
        import cupy as cp
        self.cp, self.name = cp, kind
        self.gse_seconds=float(gse_spreading_seconds)
        self.ratio=float((spectral or {}).get("ratio",1.1))
        self.nd=int((spectral or {}).get("directions",36))
        if not math.isfinite(self.gse_seconds) or self.gse_seconds<=0:
            raise ValueError("GSE spreading horizon must be positive and finite")
        self.full_shape = (grid.ny, grid.nx)
        ny, nx = self.full_shape
        if min(nx, ny) < 2 or kind not in ("latlon", "curvilinear"):
            raise ValueError("structured waves require >=2x2 latlon or curvilinear grid")
        def plane(a, name):
            if not hasattr(a, "__cuda_array_interface__"):
                raise TypeError(f"{name} must be CUDA float32")
            a = cp.asarray(a)
            if a.shape != self.full_shape or a.dtype != cp.float32 or not a.flags.c_contiguous:
                raise ValueError(f"{name} must be contiguous float32 {self.full_shape}")
            if not bool(cp.all(cp.isfinite(a))):
                raise ValueError(f"{name} is nonfinite")
            return a
        lat, lon = plane(grid.latitude, "latitude"), plane(grid.longitude, "longitude")
        depth = plane(bathymetry.depth, "depth")
        wet = (plane(bathymetry.wet, "wet") > .5) & (depth > 0)
        self.wet_to_full = cp.flatnonzero(wet).astype(cp.int32)
        self.ncell = int(self.wet_to_full.size)
        if not self.ncell:
            raise ValueError("wave grid has no wet cells")
        n = self.ncell
        self.full_to_wet = cp.full(nx*ny, -1, cp.int32)
        self.full_to_wet[self.wet_to_full] = cp.arange(n, dtype=cp.int32)
        self.lon = cp.ascontiguousarray(lon.ravel()[self.wet_to_full])
        self.lat = cp.ascontiguousarray(lat.ravel()[self.wet_to_full])
        self.depth = cp.ascontiguousarray(depth.ravel()[self.wet_to_full])
        y, x = cp.divmod(self.wet_to_full, nx)
        # Four directed faces W,E,S,N. Each interior face uses identical metric
        # and opposite normal on its two incident cells, so fluxes cancel.
        full_nb = cp.stack((self.wet_to_full-1, self.wet_to_full+1,
                            self.wet_to_full-nx, self.wet_to_full+nx), axis=1)
        outside = cp.stack((x == 0, x == nx-1, y == 0, y == ny-1), axis=1)
        safe_nb = cp.clip(full_nb, 0, nx*ny-1)
        self.neighbor = cp.where(outside, -1, self.full_to_wet[safe_nb]).astype(cp.int32)
        edge = cp.any(outside, axis=1)
        self.boundary_cells = cp.flatnonzero(edge).astype(cp.int32)
        slots = cp.full(n, -1, cp.int32)
        slots[self.boundary_cells] = cp.arange(self.boundary_cells.size, dtype=cp.int32)
        self.boundary_slot = cp.where(outside, slots[:, None], -1).astype(cp.int32)
        if kind == "latlon":
            if not bool(cp.allclose(lat, lat[:, :1], atol=1e-5)) or not bool(cp.allclose(lon, lon[:1, :], atol=1e-5)):
                raise ValueError("latlon needs rectilinear coordinates; use curvilinear")
            dl = cp.diff(lon[0])
            dl = (dl + 180) % 360 - 180
            dp = cp.diff(lat[:, 0])
            if not bool(cp.all(dl > 0)) or not bool(cp.all(dp > 0)) or not bool(cp.allclose(dl, dl[0], atol=2e-5)) or not bool(cp.allclose(dp, dp[0], atol=2e-5)):
                raise ValueError("latlon axes must be regular and increasing, with cyclic longitude allowed")
            dlon, dlat = float(dl[0])*math.pi/180, float(dp[0])*math.pi/180
            phi = self.lat * cp.float32(math.pi/180)
            if float(cp.max(cp.abs(phi))) + .5*dlat >= math.pi/2:
                raise ValueError("latlon cell faces must not cross a pole")
            radius = 6371000.
            dx = radius*cp.cos(phi)*dlon
            dy = cp.full(n, radius*dlat, cp.float32)
            self.area = (radius**2*dlon*(cp.sin(phi+.5*dlat)-cp.sin(phi-.5*dlat))).astype(cp.float32)
            self.length = cp.stack((dy, dy, radius*cp.cos(phi-.5*dlat)*dlon,
                                    radius*cp.cos(phi+.5*dlat)*dlon), axis=1).astype(cp.float32)
            self.normal_east = cp.broadcast_to(cp.asarray([-1,1,0,0],cp.float32), (n,4)).copy()
            self.normal_north = cp.broadcast_to(cp.asarray([0,0,-1,1],cp.float32), (n,4)).copy()
            self.distance = cp.stack((dx,dx,dy,dy),axis=1).astype(cp.float32)
            self.ca, self.sa = cp.ones(n,cp.float32), cp.zeros(n,cp.float32)
        else:
            factor = map_factor if map_factor is not None else getattr(grid,"map_factor",None)
            if factor is None:
                raise ValueError("curvilinear propagation requires mass-grid map_factor")
            mf = plane(factor,"map_factor").ravel()
            if not bool(cp.all(mf > 0)) or not (math.isfinite(grid.dx) and math.isfinite(grid.dy) and min(grid.dx,grid.dy)>0):
                raise ValueError("positive projected spacing and map factors required")
            ca = plane(grid.cosalpha,"cosalpha").ravel()
            sa = plane(grid.sinalpha,"sinalpha").ravel()
            if not bool(cp.allclose(ca*ca+sa*sa,1,atol=2e-4)):
                raise ValueError("grid rotation must be a unit vector")
            cell = self.wet_to_full
            self.ca,self.sa = ca[cell].copy(),sa[cell].copy()
            dx,dy = grid.dx/mf[cell],grid.dy/mf[cell]
            self.area = (dx*dy).astype(cp.float32)
            neighbor_mf = cp.where(outside,mf[cell,None],mf[safe_nb])
            distance_x = .5*grid.dx*(1/mf[cell,None]+1/neighbor_mf)
            distance_y = .5*grid.dy*(1/mf[cell,None]+1/neighbor_mf)
            self.distance = cp.concatenate((distance_x[:,:2],distance_y[:,2:]),axis=1).astype(cp.float32)
            self.length = cp.concatenate((distance_y[:,:2],distance_x[:,2:]),axis=1).astype(cp.float32)
            fc = ca[cell,None]+cp.where(outside,ca[cell,None],ca[safe_nb])
            fs = sa[cell,None]+cp.where(outside,sa[cell,None],sa[safe_nb])
            norm = cp.hypot(fc,fs)
            if bool(cp.any(norm < 1e-6)):
                raise ValueError("adjacent grid rotations are antipodal")
            fc,fs = fc/norm,fs/norm
            self.normal_east = cp.stack((-fc[:,0],fc[:,1],fs[:,2],-fs[:,3]),axis=1).astype(cp.float32)
            self.normal_north = cp.stack((-fs[:,0],fs[:,1],-fc[:,2],fc[:,3]),axis=1).astype(cp.float32)
        self.depth_gradient_east = cp.empty(n,cp.float32)
        self.depth_gradient_north = cp.empty(n,cp.float32)
        self._cfl_length = float(cp.min(self.area/cp.sum(self.length,axis=1)))
        self._diffusion_metric=float(cp.max(cp.sum(self.length/self.distance,axis=1)/self.area))
        self._buffers = None
        source = Path(__file__).with_name("structured.cu").read_text()
        options=("--std=c++17","--use_fast_math")
        self.module=cp.RawModule(code=source,options=options)
        from woof.certify.kernel_manifest import record_module
        record_module("woof.ocean.waves:structured",source=source,options=options,module=self.module)
        self._gradient=self.module.get_function("wave_depth_gradient")
        self._transport=self.module.get_function("wave_structured_transport")
        self._slopes=self.module.get_function("wave_structured_slopes")
        self._faces=self.module.get_function("wave_structured_faces")
        self._gather=self.module.get_function("wave_structured_gather")
        self._cached=None
        self.set_depth(self.depth,cp.cuda.get_current_stream())

    def max_dt(self,cg_max):
        if not math.isfinite(cg_max) or cg_max <= 0:
            raise ValueError("group-speed envelope must be positive and finite")
        delta=max(2*math.pi/self.nd,math.sqrt(self.ratio)-1/math.sqrt(self.ratio))
        diffusion=(cg_max*delta)**2*self.gse_seconds/12
        # Sum outgoing characteristics, not all four absolute face speeds.
        # The diffusion diagonal is D*sum(L/d)/A, without the old extra 4.
        # SSP coefficient remains .4; the donor limiter still enforces positivity.
        cp=self.cp
        a=cp.arange(self.nd,dtype=cp.float32)*(2*math.pi/self.nd)
        rate=cp.zeros(self.ncell,cp.float32)
        for angle in a:
            velocity=cp.maximum(cp.cos(angle)*self.normal_east+cp.sin(angle)*self.normal_north,0)
            rate=cp.maximum(rate,cp.sum(velocity*self.length,axis=1)/self.area)
        diagonal=cp.sum(self.length/self.distance,axis=1)/self.area
        return .4/float(cp.max(cg_max*rate+diffusion*diagonal))

    def frequency_steps(self,first_hz,ratio,nf):
        """Constructor-only envelopes, independent of changing water depth.

        Tolman et al. (2002), WAF 17,311-333,
        doi:10.1175/1520-0434(2002)017<0311:DAIOWG>2.0.CO;2.
        Six-frequency groups match the scratch tile. Each uses its slowest
        frequency and the analytic maximum over all possible depths.
        """
        return tuple((first,min(6,nf-first),self.max_dt(
            .6*9.80665/(2*math.pi*first_hz*ratio**first)))
            for first in range(0,nf,6))

    def set_depth(self,depth,stream):
        if depth is not self.depth:
            self.cp.copyto(self.depth,depth)
        self._gradient(((self.ncell+127)//128,),(128,),
            (self.depth,self.neighbor,self.distance,self.ca,self.sa,
             self.depth_gradient_east,self.depth_gradient_north,np.int32(self.ncell)),stream=stream)

    def advect(self,state,group_velocity,dt,stream,incident_boundary):
        return self.advect_group(state,group_velocity,dt,stream,incident_boundary,
                                 0,state.shape[1])

    def advect_group(self,state,group_velocity,dt,stream,incident_boundary,
                     first_frequency,frequency_count):
        cp=self.cp
        if self._buffers is None:
            self._buffers=(cp.empty_like(state),cp.empty_like(state),cp.empty_like(state))
            self._budget=cp.empty_like(state)
        # Constructor warmup must allocate these before graph capture. Both core
        # and backend own persistent arrays; never overwrite the input pointer.
        available=[a for a in self._buffers if a.data.ptr != state.data.ptr]
        stage,result=available[:2]
        nf,nd=state.shape[1:]
        n=state.size
        if self._cached is None:
            # Construct after laboratory geometry overrides, before graph capture.
            sides=cp.arange(4,dtype=cp.int32)[None,:]
            owned=(sides%2==1)|(self.neighbor<0)
            owner=cp.flatnonzero(owned).astype(cp.int32)
            ids=cp.full((self.ncell,4),-1,cp.int32)
            ids.ravel()[owner]=cp.arange(owner.size,dtype=cp.int32)
            for side in (0,2):
                neighbor=self.neighbor[:,side]
                ids[:,side]=cp.where(neighbor>=0,ids[cp.maximum(neighbor,0),side+1],ids[:,side])
            count=min(6,nf)*nd
            slopes=cp.empty((2,self.ncell,count),cp.float32)
            flux=cp.empty((owner.size,count),cp.float32)
            self._cached=(owner,ids,slopes,flux,count)
        owner,ids,slopes,flux,tile=self._cached
        incident=np.uint64(0) if incident_boundary is None else incident_boundary
        end=(first_frequency+frequency_count)*nd
        for first in range(first_frequency*nd,end,tile):
            count=min(tile,end-first);size=self.ncell*count
            integers=tuple(np.int32(x) for x in (self.ncell,nf*nd,first,count))
            for source,dest,final in ((state,stage,0),(stage,result,1)):
                self._slopes(((size+127)//128,),(128,),
                    (source,self.neighbor,self.distance,slopes,*integers),stream=stream)
                self._faces(((owner.size*count+127)//128,),(128,),
                    (source,slopes,group_velocity,self.depth,self.neighbor,self.boundary_slot,owner,
                     self.normal_east,self.normal_north,self.length,self.distance,self.ca,self.sa,incident,flux,
                     *tuple(np.int32(x) for x in (self.ncell,owner.size,nf,nd,first,count)),
                     np.float32(self.gse_seconds),np.float32(math.sqrt(self.ratio)-1/math.sqrt(self.ratio))),stream=stream)
                common=(source,state,dest,self.depth,self.neighbor,ids,self.area,flux,self._budget,*integers,
                        np.float32(dt),np.int32(final))
                for phase in (0,1):
                    self._gather(((size+127)//128,),(128,),(*common,np.int32(phase)),stream=stream)
        return result

    def bind_workspace(self,stage,result,budget):
        """Borrow core-owned scratch; only the returned result survives a call."""
        if stage.shape!=result.shape or stage.shape!=budget.shape or stage.shape[0]!=self.ncell:
            raise ValueError("structured transport workspace shape mismatch")
        if len({a.data.ptr for a in (stage,result,budget)})!=3:
            raise ValueError("structured transport workspace must use distinct buffers")
        self._buffers=(stage,result);self._budget=budget
