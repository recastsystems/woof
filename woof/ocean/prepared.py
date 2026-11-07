"""Bind the Rust spectral preparation contract to the coupled device solver."""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path


class PreparedWaves:
    def __init__(self, path):
        self.path=Path(path).resolve()
        self.manifest=json.loads((self.path/"manifest.json").read_text())
        m=self.manifest
        if m.get("schema")!="gpuwm.waves.prepared/v1" or m.get("status")!="complete":
            raise ValueError("waves need a complete gpuwm.waves.prepared/v1 manifest")
        self.manifest_sha256=hashlib.sha256((self.path/"manifest.json").read_bytes()).hexdigest()
        self.arrays={}

    def array(self, name):
        if name in self.arrays:return self.arrays[name]
        from .door import upload
        row=self.manifest["arrays"][name]
        source=(self.path/row["path"]).resolve()
        if not source.is_relative_to(self.path):
            raise ValueError(f"prepared wave array escapes its folder: {name}")
        with source.open("rb") as stream:
            digest=hashlib.file_digest(stream,"sha256").hexdigest()
        if digest!=row["sha256"]:
            raise ValueError(f"prepared wave array hash differs: {name}")
        self.arrays[name]=upload(source,row["shape"],dtype=row["dtype"])
        return self.arrays[name]

    def coupler(self, options):
        import cupy as cp
        from .interface import OceanGrid,Bathymetry,validate_fields
        from .coupling import DeviceCoupler
        from .waves import create_solver
        m=self.manifest;g=m["grid"]
        shape=(g["ny"],g["nx"])
        def plane(name):return cp.ascontiguousarray(self.array(name).reshape(shape),dtype=cp.float32)
        grid=OceanGrid(g["nx"],g["ny"],float(g["dx"]),float(g["dy"]),
                       *(plane(name) for name in ("lat","lon","cosalpha","sinalpha")))
        wet=plane("wet");depth=plane("depth")
        # Prepared depth and wet topology are the authority. Coastal is a
        # device diagnostic for the legacy ocean ABI, not a changed wet mask.
        padded=cp.pad(wet,1,constant_values=0)
        coastal=(wet>.5)&((padded[:-2,1:-1]<.5)|(padded[2:,1:-1]<.5)|
                          (padded[1:-1,:-2]<.5)|(padded[1:-1,2:]<.5))
        bathy=Bathymetry(-depth,depth,wet,coastal.astype(cp.float32),
                        m["bathymetry_datum"],str(self.path/"manifest.json"))
        grid.validate();validate_fields(bathy,grid)
        table_path=Path(__file__).parent/"waves"/"partition_shapes.v1.json"
        table=json.loads(table_path.read_text())
        solver=create_solver(grid,bathy,config={
            "backend":g["kind"],"map_factor":plane("map_factor"),
            "spectral":{"frequencies":options["nf"],"directions":options["nd"]},
            "source_pack":options["source_pack"],"time_seconds":float(m["initial_valid_time"]),
            "partition_shapes_sha256":hashlib.sha256(json.dumps(table,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        })
        solver.initialize_partitions(self.array("initial_partitions"),
                                     valid_time=float(m["initial_valid_time"]),shape_table=table)
        timeline=BoundaryTimeline(self,solver,table)
        timeline.bind(float(m["initial_valid_time"]))
        station_ids=options["stations"]
        stations=[row for row in m["stations"] if station_ids=="auto" or row["id"] in station_ids]
        if station_ids!="auto" and set(station_ids)!={row["id"] for row in stations}:
            raise ValueError("requested wave stations are absent from the prepared domain")
        solver.set_stations(cp.asarray([row["cell"] for row in stations],dtype=cp.int32))
        coupler=DeviceCoupler(grid,bathy,waves=solver,interval=options["interval_seconds"],
                              roughness_feedback=options["roughness_feedback"])
        from .interface import AtmosphereForcing
        coupler.buffers['u10'][...]=self.array('u10_east')[0].reshape(shape)
        coupler.buffers['v10'][...]=self.array('v10_north')[0].reshape(shape)
        coupler.buffers['rho_air'].fill(1.225)
        coupler.buffers['psfc'].fill(101325.)
        coupler.buffers['ustar'].fill(.1)
        coupler.forcing=AtmosphereForcing(**coupler.buffers,time_seconds=float(m['initial_valid_time']))
        solver.initialize_surface(coupler.forcing)
        coupler.surface_roughness(coupler.z0,coupler.buffers['ustar'])
        coupler.timeline=timeline
        # A longer restart may extend the boundary window. Bind immutable
        # geometry/initialization and verify the old boundary prefix separately;
        # request paths or a longer forcing list are not physical identity.
        field_table=Path(m['request']['field_table'])
        coupler.prepared_identity={
            "initial_valid_time":m['initial_valid_time'],
            "arrays":{key:m['arrays'][key]['sha256'] for key in
                      ('lat','lon','depth','wet','area','map_factor','cosalpha','sinalpha','initial_partitions','boundary_cells')},
            "field_table_sha256":hashlib.sha256(field_table.read_bytes()).hexdigest(),
            "forecast_cycle":m['request'].get('forecast_cycle'),
            "partition_shapes_sha256":solver.partition_shapes_sha256}
        coupler.station_metadata=stations
        return coupler


class BoundaryTimeline:
    def __init__(self, prepared, solver, table):
        import cupy as cp
        self.solver=solver;self.table=table
        self.times=tuple(float(v) for v in prepared.manifest["boundary_times"])
        if (len(self.times)<2 or any(not math.isfinite(v) for v in self.times)
                or any(b<=a for a,b in zip(self.times,self.times[1:]))):
            raise ValueError("wave boundary times must be finite and strictly increasing")
        indices=prepared.array("boundary_cells")
        count=solver.backend.full_to_wet.size
        if indices.dtype!=cp.int32 or indices.ndim!=1:
            raise ValueError("prepared wave boundary_cells must be int32 full-cell indices")
        if bool(cp.any((indices<0)|(indices>=count))) or cp.unique(indices).size!=indices.size:
            raise ValueError("prepared wave boundary indices are invalid or repeated")
        lookup=cp.full(count,-1,cp.int32);lookup[indices]=cp.arange(indices.size,dtype=cp.int32)
        expected=solver.backend.wet_to_full[solver.backend.boundary_cells]
        order=lookup[expected]
        if expected.size!=indices.size or bool(cp.any(order<0)):
            raise ValueError("prepared boundary cells differ from the solver open boundary")
        self.partitions=cp.ascontiguousarray(prepared.array("boundary_partitions")[:,order])
        if self.partitions.shape!=(len(self.times),indices.size,4,4):
            raise ValueError("wave boundary partition shape differs from times and cells")
        self.index=None
        self.source=prepared.path/prepared.manifest['arrays']['boundary_partitions']['path']
        self.source_row=prepared.manifest['arrays']['boundary_partitions']

    def checkpoint_identity(self):
        return {"times":list(self.times),"shape":self.source_row['shape'],"sha256":self.source_row['sha256']}

    def validate_identity(self, incoming):
        if not isinstance(incoming,dict):raise ValueError('wave restart lacks boundary identity')
        count=len(incoming.get('times',[]))
        shape=incoming.get('shape',[])
        if (count<2 or list(self.times[:count])!=incoming['times'] or len(shape)!=4
                or shape[0]!=count or shape[1:]!=self.source_row['shape'][1:]):
            raise ValueError('wave restart boundary grid or historical time knots differ')
        if count==len(self.times):digest=self.source_row['sha256']
        else:
            remaining=math.prod(shape)*4;hasher=hashlib.sha256()
            with self.source.open('rb') as stream:
                while remaining:
                    block=stream.read(min(remaining,1024*1024))
                    if not block:raise ValueError('wave boundary prefix is truncated')
                    hasher.update(block);remaining-=len(block)
            digest=hasher.hexdigest()
        if digest!=incoming.get('sha256'):
            raise ValueError('wave restart historical boundary partitions differ')

    def bind(self, time_seconds):
        if not self.times[0]<=time_seconds<self.times[-1]:
            raise ValueError("wave boundary does not cover the next integration interval")
        index=bisect_right(self.times,time_seconds)-1
        if index!=self.index:
            self.solver.set_boundary_partitions(self.partitions[index],self.partitions[index+1],
                time0=self.times[index],time1=self.times[index+1],shape_table=self.table)
            self.index=index

    def step(self, dt):
        remaining=float(dt)
        while remaining>1.e-8:
            now=self.solver.time_seconds
            self.bind(now)
            delta=min(remaining,self.times[self.index+1]-now)
            self.solver.step(delta)
            remaining=max(0.,remaining-delta)
