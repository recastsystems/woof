"""Native WRF v4.7.1 firebrand particle kernels and persistent state."""
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

PARTICLE_COORDS=("fs_p_x","fs_p_y","fs_p_z")
PARTICLE_PROPERTIES=("fs_p_mass","fs_p_diam","fs_p_effd","fs_p_temp","fs_p_tvel")
PARTICLE_IDENTIFIERS=("fs_p_id","fs_p_src","fs_p_dt")
COARSE_REAL_FIELDS=("fs_fire_area","fs_fuel_spotting_risk","fs_count_landed_all",
                    "fs_count_landed_hist","fs_frac_landed","fs_spotting_lkhd")
COARSE_INTEGER_FIELDS=("fs_landing_mask","fs_gen_inst")


@dataclass(frozen=True)
class SpottingOptions:
    fs_array_maxsize:int=100000
    fs_firebrand_gen_lim:int=0
    fs_firebrand_gen_dt:int=5
    fs_firebrand_gen_levels:int=5
    fs_firebrand_gen_maxhgt:int=50
    fs_firebrand_gen_levrand:bool=False
    fs_firebrand_gen_levrand_seed:int=1
    fs_firebrand_gen_mom3d_dt:int=4
    fs_firebrand_gen_prop_diam:float=10.0
    fs_firebrand_gen_prop_effd:float=10.0
    fs_firebrand_gen_prop_temp:float=900.0
    fs_firebrand_gen_prop_tvel:float=0.0
    fs_firebrand_dens:float=513000.0
    fs_firebrand_dens_char:float=299000.0
    fs_firebrand_max_life_dt:int=200
    fs_firebrand_land_hgt:float=0.15
    fuel_crosswalk:bool=False
    trackember:bool=False

    @classmethod
    def from_config(cls,cfg):
        default=cls()
        return cls(**{key:getattr(cfg,key,getattr(default,key)) for key in asdict(default)})

    def validate(self):
        if self.fs_array_maxsize<1 or self.fs_firebrand_gen_levels<1:
            raise ValueError("firebrand capacity and generation levels must be positive to allocate particle state")
        for key in ("fs_firebrand_gen_prop_diam","fs_firebrand_gen_prop_effd","fs_firebrand_gen_prop_temp",
                    "fs_firebrand_dens","fs_firebrand_dens_char"):
            if not np.isfinite(getattr(self,key)) or getattr(self,key)<=0:
                raise ValueError(f"{key} must be finite and positive for firebrand mass, burnout and settling")


class SpottingState:
    """Native particles, generation clocks and landing diagnostics per domain."""
    def __init__(self,coarse_shape,sr_x,sr_y,options=None,*,field_allocator=None,field_storage=None):
        import cupy as cp
        self.options=options or SpottingOptions()
        self.options.validate()
        self.coarse_shape=tuple(map(int,coarse_shape))
        self.sr_x,self.sr_y=int(sr_x),int(sr_y)
        if min(*self.coarse_shape,self.sr_x,self.sr_y)<1:
            raise ValueError("firebrand geometry must contain positive atmosphere dimensions and refinement")
        ny,nx=self.coarse_shape
        self.fine_shape=(ny*self.sr_y,nx*self.sr_x)
        self.padded_shape=(ny+8,nx+8)
        self.coordinates=cp.zeros((3,self.options.fs_array_maxsize),cp.float32)
        self.properties=cp.zeros((5,self.options.fs_array_maxsize),cp.float32)
        self.identifiers=cp.zeros((3,self.options.fs_array_maxsize),cp.int32)
        self.control=cp.zeros(3,cp.int32)
        self._field_allocator=cp.empty if field_allocator is None else field_allocator
        layout={**{key:(self.padded_shape,cp.float32) for key in COARSE_REAL_FIELDS},
                **{key:(self.padded_shape,cp.int32) for key in COARSE_INTEGER_FIELDS},
                "fs_fire_rosdt":(self.fine_shape,cp.float32)}
        if field_storage is None:
            self.data={key:self._field_allocator(shape,dtype) for key,(shape,dtype) in layout.items()}
            for value in self.data.values():
                value.fill(0)
        else:
            if set(field_storage)!=set(layout):
                raise ValueError("firebrand field storage must retain every native landing and release field")
            self.data=dict(field_storage)
            for key,(shape,dtype) in layout.items():
                value=self.data[key]
                if value.shape!=shape or value.dtype!=dtype or not value.flags.c_contiguous:
                    raise ValueError(f"firebrand field storage {key} has incompatible native geometry or dtype")
        self.last_gen_dt=0
        self.count_reset=False
        self.step_count=0

    @property
    def interior(self):
        ny,nx=self.coarse_shape
        return (slice(4,ny+4),slice(4,nx+4))

    def arrays(self):
        return {**{key:self.coordinates[i] for i,key in enumerate(PARTICLE_COORDS)},
            **{key:self.properties[i] for i,key in enumerate(PARTICLE_PROPERTIES)},
            **{key:self.identifiers[i] for i,key in enumerate(PARTICLE_IDENTIFIERS)},
            **self.data,"control":self.control}

    def metadata(self):
        return dict(version=1,coarse_shape=list(self.coarse_shape),sr_x=self.sr_x,sr_y=self.sr_y,
            options=asdict(self.options),last_gen_dt=self.last_gen_dt,count_reset=self.count_reset,
            step_count=self.step_count)

    def validate_restart(self,arrays,metadata):
        expected=self.metadata()
        for key in ("version","coarse_shape","sr_x","sr_y","options"):
            if metadata.get(key)!=expected[key]:
                raise ValueError(f"firebrand restart {key} differs from immutable spotting geometry or controls")
        if set(arrays)!=set(self.arrays()):
            raise ValueError("firebrand restart must contain every particle, identifier and landing field")
        for key,value in self.arrays().items():
            if arrays[key].shape!=value.shape or arrays[key].dtype!=value.dtype:
                raise ValueError(f"firebrand restart {key} has incompatible native shape or dtype")
        for key in ("last_gen_dt","step_count"):
            if type(metadata.get(key)) is not int or metadata[key]<0:
                raise ValueError(f"firebrand restart {key} must be a nonnegative integer clock")
        if type(metadata.get("count_reset")) is not bool:
            raise ValueError("firebrand restart count_reset must be boolean")

    def restore(self,arrays,metadata):
        import cupy as cp
        self.validate_restart(arrays,metadata)
        self.coordinates=cp.ascontiguousarray(cp.stack([arrays[key] for key in PARTICLE_COORDS]))
        self.properties=cp.ascontiguousarray(cp.stack([arrays[key] for key in PARTICLE_PROPERTIES]))
        self.identifiers=cp.ascontiguousarray(cp.stack([arrays[key] for key in PARTICLE_IDENTIFIERS]))
        self.control=cp.ascontiguousarray(cp.asarray(arrays["control"],dtype=cp.int32))
        self.data={key:cp.ascontiguousarray(cp.asarray(arrays[key])) for key in self.data}
        self.last_gen_dt=metadata["last_gen_dt"]
        self.count_reset=metadata["count_reset"]
        self.step_count=metadata["step_count"]

    def output_fields(self):
        import cupy as cp
        result={key.upper():self.data[key][self.interior] for key in (*COARSE_REAL_FIELDS,*COARSE_INTEGER_FIELDS)}
        result.update(FS_FIRE_ROSDT=self.data["fs_fire_rosdt"],
            FS_GEN_IDMAX=self.control[0],FS_LAST_GEN_DT=cp.asarray(self.last_gen_dt,dtype=cp.int32),
            FS_COUNT_RESET=cp.asarray(int(self.count_reset),dtype=cp.int32))
        if self.options.trackember:
            result.update({key.upper():value for key,value in self.arrays().items() if key.startswith("fs_p_")})
        return result

    def _generate(self,fine,step):
        import cupy as cp
        o=self.options
        ny,nx=self.coarse_shape
        n=self.fine_shape[0]*self.fine_shape[1]
        potential=self._field_allocator(self.fine_shape,cp.float32)
        release=cp.empty((3,0),cp.float32)
        prop=cp.empty((5,0),cp.float32)
        selection=cp.zeros(2,cp.int32)
        def select(capacity):
            _launch("sfire_spotting_select",1,(self.data["fs_fire_rosdt"],fine["fgip"],potential,release,prop,
            self.data["fs_gen_inst"],selection,*map(np.int32,(self.fine_shape[1],self.fine_shape[0],nx,ny,
            self.sr_x,self.sr_y,o.fs_firebrand_gen_lim,o.fs_firebrand_gen_levels,capacity,fine["fgip"].strides[0]//4)),
            *map(np.float32,(o.fs_firebrand_gen_maxhgt,o.fs_firebrand_gen_prop_diam,o.fs_firebrand_gen_prop_effd,
                            o.fs_firebrand_gen_prop_temp,o.fs_firebrand_gen_prop_tvel,o.fs_firebrand_dens))))
        # Count first. Equal release potentials can exceed the requested
        # approximate rank, so a guessed limit would truncate native releases.
        select(0)
        points=int(selection[0].item())
        if points:
            length=points*o.fs_firebrand_gen_levels
            release=self._field_allocator((3,length),cp.float32);release.fill(0)
            prop=self._field_allocator((5,length),cp.float32);prop.fill(0)
            select(length)
            seed=(o.fs_firebrand_gen_levrand_seed+1)*step if o.fs_firebrand_gen_levrand else 0
            release,prop=release_heights(release,prop,points=points,levels=o.fs_firebrand_gen_levels,
                seed=seed,land_height=o.fs_firebrand_land_hgt,field_allocator=self._field_allocator)
            self.coordinates,self.properties,self.identifiers,_,self.control=generate_particles(
                self.coordinates,self.properties,self.identifiers,release,prop,self.control)
        self.data["fs_fire_rosdt"].fill(0)
        self.last_gen_dt=0

    def advance_native(self,fields,fine,*,step,history_alarm=False,advector=None):
        """Advance after the completed atmosphere interval and final halos.

        ``fields`` contains prepared padded meteorology. Fine fields are
        physical cells: BURNT_AREA_DT, FGIP, FMC_G, NFUEL_CAT and FIRE_AREA.
        The native source overwrites its spread accumulator each call.
        """
        import cupy as cp
        o=self.options
        if o.fs_firebrand_gen_lim<=0:
            return
        if type(step) is not int or step!=self.step_count+1:
            raise ValueError("firebrand advancement must follow its checkpointed integer atmosphere cadence")
        fine={key:cp.asarray(fine[key],dtype=cp.float32) for key in
              ("burnt_area_dt","fgip","fmc_g","nfuel_cat","fire_area")}
        if any(value.shape!=self.fine_shape for value in fine.values()):
            raise ValueError("firebrand generation fields must cover every physical refined fire cell")
        if any(value.strides[1]!=4 or value.strides[0]<4*self.fine_shape[1] for value in fine.values()):
            raise ValueError("firebrand fine planes must retain contiguous native rows and a valid row stride")
        if self.count_reset:
            for key in ("fs_count_landed_hist","fs_fuel_spotting_risk","fs_gen_inst"):
                self.data[key].fill(0)
            self.count_reset=False
        self.data["fs_landing_mask"].fill(0)
        cp.copyto(self.data["fs_fire_rosdt"],fine["burnt_area_dt"])
        self.last_gen_dt+=1
        if self.last_gen_dt>=o.fs_firebrand_gen_dt:
            self._generate(fine,step)
        active=int(self.control[1].item())
        if active:
            coords=cp.ascontiguousarray(self.coordinates[:,:active])
            prop=cp.ascontiguousarray(self.properties[:,:active])
            transport = advect_particles if advector is None else advector
            coords,prop,status=transport(coords,self.identifiers[2,:active],prop,fields,float(fields["dt"]),
                origin=(-3,-3),tile=(1,self.coarse_shape[1],1,self.coarse_shape[0]),
                land_height=o.fs_firebrand_land_hgt,momentum_steps=o.fs_firebrand_gen_mom3d_dt,
                brand_density=o.fs_firebrand_dens,char_density=o.fs_firebrand_dens_char)
            self.coordinates[:,:active]=coords;self.properties[:,:active]=prop
            all_status=cp.zeros(o.fs_array_maxsize,cp.int32);all_status[:active]=status
            delta=self._field_allocator(self.padded_shape,cp.float32)
            delta.fill(0)
            _launch("sfire_spotting_remove_deposit",o.fs_array_maxsize,(self.coordinates,self.properties,self.identifiers,
                all_status,delta,*map(np.int32,(o.fs_array_maxsize,self.coarse_shape[1],self.coarse_shape[0],o.fs_firebrand_max_life_dt)),
                np.float32(o.fs_firebrand_land_hgt)))
            _launch("sfire_spotting_add_counts",delta.size,(self.data["fs_count_landed_all"],self.data["fs_count_landed_hist"],delta,np.int32(delta.size)))
            _launch("sfire_spotting_compact",1,(self.coordinates,self.properties,self.identifiers,self.control,np.int32(o.fs_array_maxsize)))
        if history_alarm and step>1:
            self.count_reset=True
            _launch("sfire_spotting_history",1,(*[fine[key] for key in ("fire_area","fgip","fmc_g","nfuel_cat")],
                *[self.data[key] for key in ("fs_fire_area","fs_fuel_spotting_risk","fs_count_landed_hist",
                   "fs_landing_mask","fs_frac_landed","fs_spotting_lkhd")],
                *map(np.int32,(self.coarse_shape[1],self.coarse_shape[0],self.sr_x,self.sr_y,o.fuel_crosswalk,
                    *[fine[key].strides[0]//4 for key in ("fire_area","fgip","fmc_g","nfuel_cat")]))))
        self.step_count=step

    def advance_atmosphere(self,state,cfg,fire,*,dt=None,history_alarm=False):
        """Final-step hook using compact atmosphere state and current fire fields."""
        if self.options.fs_firebrand_gen_lim<=0:
            return
        if getattr(fire,"_tile_spotting_owner",False):
            return
        fields=prepare_atmosphere(state,cfg)
        fields["dt"]=float(cfg.dt if dt is None else dt)
        grid=fire.grid if hasattr(fire,"grid") else fire
        fine={key:grid.data[key][grid.interior] for key in ("burnt_area_dt","fgip","fmc_g","nfuel_cat","fire_area")}
        self.advance_native(fields,fine,step=self.step_count+1,history_alarm=history_alarm)

MODULE_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")
MODULE_KEY = "woof.core.sfire_spotting:sfire_spotting"


def module_source(kernel_dir=None):
    from woof.core.kernels import _preamble
    root = Path(__file__).parent / "kernels" if kernel_dir is None else Path(kernel_dir)
    return _preamble(root)+"".join((root/name).read_text(encoding="utf-8") for name in
        ("glibc_flt32.cuh", "glibc_flt64.cuh", "sfire_spotting.cu"))


@lru_cache(maxsize=None)
def _module(device):
    import cupy as cp
    from cupy.cuda import compiler
    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile
    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(source, MODULE_OPTIONS, None, "sfire_spotting.cu")
        result = cp.cuda.function.Module()
        result.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS, module=None)
    return result


def _launch(name, n, args):
    import cupy as cp
    _module(cp.cuda.Device().id).get_function(name)(((n+127)//128,), (128,), args)


def properties(inputs, *, density=513000.0,original_property=False):
    import cupy as cp
    src = cp.ascontiguousarray(cp.asarray(inputs, dtype=cp.float32))
    if src.ndim != 2 or src.shape[0] != 4:
        raise ValueError("firebrand input properties need diameter, effective diameter, temperature and terminal velocity rows")
    n = src.shape[1]
    out = cp.empty((5,n), cp.float32)
    _launch("sfire_spotting_property", n, (src, out, np.int32(n), np.float32(density),np.int32(original_property)))
    return out


def particle_physics(prop, height, dt, pressure, density, temperature, wind, *,
                     mode="coupled", brand_density=513000.0, char_density=299000.0):
    import cupy as cp
    prop = cp.ascontiguousarray(cp.asarray(prop, dtype=cp.float32))
    if prop.ndim != 2 or prop.shape[0] != 5:
        raise ValueError("firebrand state needs mass, diameter, effective diameter, temperature and terminal velocity rows")
    n = prop.shape[1]
    arrays = [cp.ascontiguousarray(cp.asarray(a, dtype=cp.float32)) for a in
              (height, pressure, density, temperature, wind)]
    if any(a.shape != (n,) for a in arrays):
        raise ValueError("particle meteorology and height must align with every firebrand")
    modes = {"burnout":0, "termvel":1, "coupled":2}
    if mode not in modes:
        raise ValueError("firebrand physics mode must be burnout, termvel or coupled")
    _launch("sfire_spotting_physics", n,
        (prop,*arrays,np.int32(n),np.int32(modes[mode]),np.float32(dt),
         np.float32(brand_density),np.float32(char_density)))
    return prop, arrays[0]


def interpolate_box(coordinates, corners):
    import cupy as cp
    coordinates = cp.ascontiguousarray(cp.asarray(coordinates, dtype=cp.float32))
    corners = cp.ascontiguousarray(cp.asarray(corners, dtype=cp.float32))
    if coordinates.ndim != 2 or coordinates.shape[0] != 3 or corners.shape != (8, coordinates.shape[1]):
        raise ValueError("particle interpolation needs three coordinate and eight corner rows")
    n = coordinates.shape[1]
    out = cp.empty(n, cp.float32)
    _launch("sfire_spotting_interp_box",n,(coordinates,corners,out,np.int32(n)))
    return out


def sample_at_particles(coordinates, fields, *, origin, tile):
    import cupy as cp
    coordinates = cp.ascontiguousarray(cp.asarray(coordinates, dtype=cp.float32))
    if coordinates.ndim != 2 or coordinates.shape[0] != 3:
        raise ValueError("firebrand sampling requires three particle coordinate rows")
    arrays = [cp.ascontiguousarray(cp.asarray(fields[k], dtype=cp.float32)) for k in
              ("u","v","w","pressure","theta","density","height")]
    shape = arrays[0].shape
    if len(shape) != 3 or any(a.shape != shape for a in arrays):
        raise ValueError("particle meteorology must share one padded native three-dimensional allocation")
    nz, ny, nx = shape
    ml, mb = origin
    is_, ie, js, je = tile
    n = coordinates.shape[1]
    out = cp.empty((7,n), cp.float32)
    status = cp.zeros(n, cp.int32)
    _launch("sfire_spotting_sample",n,(coordinates,*arrays,out,status,
        *map(np.int32,(n,nx,ny,nz,ml,mb,is_-4,ie-1+4,js-4,je-1+4))))
    if bool(cp.any(status)):
        raise ValueError("particle sampling would read beyond the horizontal allocation or model-top interface")
    return out


def advect_particles(coordinates, life, prop, fields, dt, *, origin, tile,
                     land_height=0.15, momentum_steps=4, brand_density=513000.0,
                     char_density=299000.0, original_cleanup=False):
    """Two-pass transport, burnout and settling with source staggered offsets.

    Burnt particles are removed before deposition. ``original_cleanup``
    exposes the unmodified source control where a fully burnt particle can
    instead enter the landing count.
    """
    import cupy as cp
    coordinates = cp.ascontiguousarray(cp.asarray(coordinates, dtype=cp.float32))
    prop = cp.ascontiguousarray(cp.asarray(prop, dtype=cp.float32))
    if coordinates.ndim != 2 or coordinates.shape[0] != 3:
        raise ValueError("particle advection requires three coordinate rows")
    n = coordinates.shape[1]
    if prop.shape != (5,n):
        raise ValueError("particle advection requires five aligned property rows")
    life = cp.ascontiguousarray(cp.asarray(life, dtype=cp.int32))
    if life.shape != (n,):
        raise ValueError("particle lifetimes must align with coordinates")
    arrays = [cp.ascontiguousarray(cp.asarray(fields[k], dtype=cp.float32)) for k in
              ("u","v","w","pressure","theta","density","height")]
    nz, ny, nx = arrays[0].shape
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("particle advection meteorology must share one padded allocation")
    metric = [cp.ascontiguousarray(cp.asarray(fields[k], dtype=cp.float32)) for k in ("metric_x","metric_y")]
    if any(a.shape != (ny,nx) for a in metric):
        raise ValueError("particle horizontal metrics must align with meteorology")
    out = cp.empty_like(coordinates)
    status = cp.zeros(n, cp.int32)
    _launch("sfire_spotting_advect",n,(coordinates,life,prop,*arrays,*metric,out,status,
        *map(np.int32,(n,nx,ny,nz,*origin,*tile,momentum_steps,original_cleanup)),
        np.float32(dt),np.float32(land_height),np.float32(brand_density),np.float32(char_density)))
    return out, prop, status


def release_heights(coordinates, prop, *, points, levels, seed=0, land_height=0.15,field_allocator=None):
    import cupy as cp
    coords = cp.ascontiguousarray(cp.asarray(coordinates, dtype=cp.float32))
    prop = cp.ascontiguousarray(cp.asarray(prop, dtype=cp.float32))
    n = int(points)*int(levels)
    if coords.shape != (3,n) or prop.shape != (5,n) or points<1 or levels<1:
        raise ValueError("firebrand release arrays must contain every source point at each generation level")
    random = (cp.empty if field_allocator is None else field_allocator)((n,),cp.float32)
    random.fill(0)
    _launch("sfire_spotting_release_heights",1,(coords,prop,random,np.int32(points),np.int32(levels),np.int32(seed),np.float32(land_height)))
    return coords, prop


def generate_particles(coordinates, prop, ident, release, release_prop, control, *,
                       release_life=None, release_source=None, source_prefix=1000000):
    import cupy as cp
    arrays = [cp.ascontiguousarray(cp.asarray(a, dtype=cp.float32)) for a in (coordinates,prop,release,release_prop)]
    coords,prop,release,release_prop = arrays
    ident = cp.ascontiguousarray(cp.asarray(ident, dtype=cp.int32))
    control = cp.ascontiguousarray(cp.asarray(control, dtype=cp.int32))
    capacity,nrel = coords.shape[1],release.shape[1]
    if coords.shape!=(3,capacity) or prop.shape!=(5,capacity) or ident.shape!=(3,capacity) or control.shape!=(3,) or release.shape!=(3,nrel) or release_prop.shape!=(5,nrel):
        raise ValueError("firebrand generation requires aligned packed particles, releases and counters")
    imported = release_life is not None or release_source is not None
    if imported and (release_life is None or release_source is None):
        raise ValueError("imported particles require source and lifetime records together")
    life = cp.zeros(nrel,cp.int32) if not imported else cp.ascontiguousarray(cp.asarray(release_life,dtype=cp.int32))
    source = cp.zeros(nrel,cp.int32) if not imported else cp.ascontiguousarray(cp.asarray(release_source,dtype=cp.int32))
    if life.shape!=(nrel,) or source.shape!=(nrel,):
        raise ValueError("imported firebrand records must align with release positions")
    _launch("sfire_spotting_generate",1,(coords,prop,ident,release,release_prop,life,source,control,
        *map(np.int32,(capacity,nrel,imported,source_prefix))))
    if int(control[2].item()):
        raise ValueError("firebrand insertion requires contiguous active particles and valid positive release positions")
    return coords,prop,ident,release,control


def approximate_order(values, order):
    import cupy as cp
    values = cp.ascontiguousarray(cp.asarray(values,dtype=cp.float32)).ravel()
    if values.size<1:
        raise ValueError("native release ranking requires at least one candidate")
    out = cp.empty((),cp.float32)
    _launch("sfire_spotting_order",1,(values,out,np.int32(values.size),np.int32(order)))
    return out


def neighbor_packets(coordinates,prop,ident,mask,neighbors,*,tile):
    """Stable native eight-neighbor packets with one destination per brand.

    Packet views retain source X/Y/Z/property row order and integer
    ID/SRC/DT row order. Missing edge neighbors receive no packet.
    """
    import cupy as cp
    coords=cp.ascontiguousarray(cp.asarray(coordinates,dtype=cp.float32))
    prop=cp.ascontiguousarray(cp.asarray(prop,dtype=cp.float32))
    ident=cp.ascontiguousarray(cp.asarray(ident,dtype=cp.int32))
    mask=cp.ascontiguousarray(cp.asarray(mask,dtype=cp.int32))
    neighbors=cp.ascontiguousarray(cp.asarray(neighbors,dtype=cp.int32))
    n=coords.shape[1]
    if coords.shape!=(3,n) or prop.shape!=(5,n) or ident.shape!=(3,n) or mask.shape!=(n,) or neighbors.shape!=(8,):
        raise ValueError("firebrand neighbor exchange requires aligned particle rows and the native eight-neighbor topology")
    real=cp.empty((8,n),cp.float32)
    integer=cp.empty((3,n),cp.int32)
    counts=cp.zeros(8,cp.int32)
    _launch("sfire_spotting_pack_neighbors",1,(coords,prop,ident,mask,neighbors,real,integer,counts,
        *map(np.int32,(n,*tile))))
    sizes=counts.get().tolist()
    packets=[]
    offset=0
    for size in sizes:
        packets.append((cp.ascontiguousarray(real[:,offset:offset+size]),
                        cp.ascontiguousarray(integer[:,offset:offset+size])))
        offset+=size
    return tuple(packets),counts


def concatenate_neighbor_packets(packets):
    """Concatenate arrivals in the native neighbor order without field math."""
    import cupy as cp
    if len(packets)!=8:
        raise ValueError("firebrand receive order must include all eight native neighbor slots")
    real=[];integer=[]
    for floating,ids in packets:
        floating=cp.ascontiguousarray(cp.asarray(floating,dtype=cp.float32))
        ids=cp.ascontiguousarray(cp.asarray(ids,dtype=cp.int32))
        if floating.ndim!=2 or floating.shape[0]!=8 or ids.shape!=(3,floating.shape[1]):
            raise ValueError("firebrand receive packet must keep all native real and integer property rows")
        real.append(floating);integer.append(ids)
    return cp.ascontiguousarray(cp.concatenate(real,axis=1)),cp.ascontiguousarray(cp.concatenate(integer,axis=1))


def exchange_neighbor_packets(comm,packets,neighbors,*,return_device=True,tag_base=16000):
    """Exchange native typed packets with posted receives before large sends.

    ``comm`` implements MPI's Isend/Irecv interface. Buffers stay alive
    until every request completes. The neighbor slot order matches the
    original helper, including absent neighbors and empty messages.
    """
    if len(packets)!=8:
        raise ValueError("firebrand MPI exchange requires the eight native neighbor slots")
    def host(value):
        if hasattr(value,"__cuda_array_interface__"):
            import cupy as cp
            return cp.asnumpy(value)
        return np.asarray(value)
    neighbors=host(neighbors)
    if neighbors.shape!=(8,) or neighbors.dtype!=np.int32:
        raise ValueError("firebrand MPI topology requires eight native int32 neighbor identifiers")
    send=[]
    for real,integer in packets:
        real,integer=np.ascontiguousarray(host(real)),np.ascontiguousarray(host(integer))
        if (real.ndim!=2 or real.shape[0]!=8 or real.dtype!=np.float32
                or integer.shape!=(3,real.shape[1]) or integer.dtype!=np.int32):
            raise ValueError("firebrand MPI packets require aligned native float32 and int32 property rows")
        send.append((real,integer))
    sent_counts=[np.asarray([real.shape[1]],np.int32) for real,_ in send]
    received_counts=[np.zeros(1,np.int32) for _ in range(8)]
    requests=[]
    for edge,rank in enumerate(neighbors.tolist()):
        if rank>=0:
            requests.append(comm.Irecv(received_counts[edge],source=rank,tag=tag_base))
    for edge,rank in enumerate(neighbors.tolist()):
        if rank>=0:
            requests.append(comm.Isend(sent_counts[edge],dest=rank,tag=tag_base))
    for request in requests:
        request.Wait()
    received=[];requests=[]
    for edge,rank in enumerate(neighbors.tolist()):
        count=int(received_counts[edge][0])
        if count<0:
            raise ValueError("firebrand MPI receive count cannot be negative")
        real=np.empty((8,count),np.float32);integer=np.empty((3,count),np.int32)
        received.append((real,integer))
        if rank>=0:
            requests.extend((comm.Irecv(real,source=rank,tag=tag_base+1),
                             comm.Irecv(integer,source=rank,tag=tag_base+2)))
    for edge,rank in enumerate(neighbors.tolist()):
        if rank>=0:
            requests.extend((comm.Isend(send[edge][0],dest=rank,tag=tag_base+1),
                             comm.Isend(send[edge][1],dest=rank,tag=tag_base+2)))
    for request in requests:
        request.Wait()
    if return_device:
        import cupy as cp
        return tuple((cp.asarray(real),cp.asarray(integer)) for real,integer in received)
    return tuple(received)


def particle_blocks(coordinates,*,coarse_shape,block_shape):
    """Device-classify active brands into bounded atmosphere-column groups."""
    import cupy as cp
    coords=cp.ascontiguousarray(cp.asarray(coordinates,dtype=cp.float32))
    if coords.ndim!=2 or coords.shape[0]!=3:
        raise ValueError("firebrand column groups require three aligned coordinate rows")
    ny,nx=map(int,coarse_shape);by,bx=map(int,block_shape)
    if min(ny,nx,by,bx)<1:
        raise ValueError("firebrand column groups require positive domain and block dimensions")
    out=cp.empty(coords.shape[1],cp.int32)
    _launch("sfire_spotting_particle_blocks",out.size,(coords,out,*map(np.int32,(out.size,nx,ny,bx,by))))
    return out


def prepare_native_fields(inputs, *, p_top, rdx, rdy=None, use_theta_m=False,theta_total=False):
    """Prepare native particle meteorology, retaining source vertical order.

    Inputs share a padded W-level allocation, including the unused top
    mass slot. Moist species use their source order in a leading axis.
    The ground-relative height conversion preserves the bottom altitude
    before changing any level.
    """
    import cupy as cp
    names=("ph","phb","p","pb","theta_pert","moist","al","alb","msftx","msfty",
           "muts","c1h","c2h","dnw","fnm","fnp")
    source=dict(inputs)
    source.setdefault("msfty",source["msftx"])
    arrays={k:cp.ascontiguousarray(cp.asarray(source[k],dtype=cp.float32)) for k in names}
    nz,ny,nx=arrays["ph"].shape
    shape=(nz,ny,nx)
    if nz<3 or any(arrays[k].shape!=shape for k in ("phb","p","pb","theta_pert","al","alb")):
        raise ValueError("particle column preparation requires aligned padded native vertical fields")
    if arrays["moist"].ndim!=4 or arrays["moist"].shape[1:]!=shape:
        raise ValueError("particle column preparation requires source-ordered moisture species")
    if any(arrays[k].shape!=(ny,nx) for k in ("msftx","msfty","muts")):
        raise ValueError("particle column mass and metrics must align with horizontal meteorology")
    if any(arrays[k].size<nz-1 for k in ("c1h","c2h","dnw")) or any(arrays[k].size<nz for k in ("fnm","fnp")):
        raise ValueError("particle column coefficients must cover every physical vertical level")
    out={k:cp.empty(shape,cp.float32) for k in ("height","pressure","theta","density","p8w")}
    out.update(metric_x=cp.empty((ny,nx),cp.float32),metric_y=cp.empty((ny,nx),cp.float32))
    _launch("sfire_spotting_prepare",nx*ny,(*[arrays[k] for k in names],
        *[out[k] for k in ("height","pressure","theta","density","metric_x","metric_y","p8w")],
        *map(np.int32,(nx*ny,nz,arrays["moist"].shape[0],use_theta_m,theta_total)),
        np.float32(p_top),np.float32(rdx),np.float32(rdx if rdy is None else rdy)))
    out.update({k:cp.ascontiguousarray(cp.asarray(source[k],dtype=cp.float32)) for k in ("u","v","w")})
    return out


def prepare_atmosphere(state,cfg):
    """Device-copy compact staggered state into native firebrand columns."""
    import cupy as cp
    nz,ny,nx=state.thp.shape
    if (ny,nx)!=(cfg.ny,cfg.nx):
        raise ValueError("firebrand atmosphere dimensions differ from its domain configuration")
    shape=(nz+1,ny+8,nx+8)
    packed={key:cp.empty(shape,cp.float32) for key in
        ("u","v","w","ph","phb","p","pb","theta_pert","al","alb")}
    species=[getattr(state,key,None) for key in ("qv","qc","qr","qi","qs","qg","qh")]
    species=[value for value in species if value is not None]
    moist=cp.ascontiguousarray(cp.stack(species)) if species else cp.zeros((1,nz,ny,nx),cp.float32)
    nm=moist.shape[0]
    packed["moist"]=cp.empty((nm,*shape),cp.float32)
    packed["msftx"]=cp.empty(shape[1:],cp.float32)
    packed["muts"]=cp.empty(shape[1:],cp.float32)
    profiles=sum(int(getattr(state,key).ndim==1)<<i for i,key in enumerate(("phb","pb","thb","alb")))
    inputs=[cp.ascontiguousarray(cp.asarray(getattr(state,key),dtype=cp.float32)) for key in
            ("u","v","w","php","phb","p","pb","thp","thb")]
    inputs.extend((moist,*[cp.ascontiguousarray(cp.asarray(getattr(state,key),dtype=cp.float32)) for key in
                          ("al","alb","msft","mup","mub2d")]))
    _launch("sfire_spotting_pack_atmosphere",int(np.prod(shape)),(*inputs,
        *[packed[key] for key in ("u","v","w","ph","phb","p","pb","theta_pert","moist","al","alb","msftx","muts")],
        *map(np.int32,(nx,ny,nz,nm,profiles))))
    for key in ("c1h","c2h","dnw","fnm","fnp"):
        value=cp.ascontiguousarray(cp.asarray(getattr(state,key),dtype=cp.float32))
        if value.size==nz:
            padded=cp.zeros(nz+1,cp.float32);padded[:nz]=value;value=padded
        packed[key]=value
    return prepare_native_fields(packed,p_top=float(state.p_top),rdx=float(np.float32(1.)/np.float32(cfg.dx)),
        rdy=float(np.float32(1.)/np.float32(cfg.dy)),theta_total=True,
        use_theta_m=bool(getattr(state,"use_theta_m",0)))
