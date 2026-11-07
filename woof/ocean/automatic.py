"""Automatic normal-run preparation. Rust owns acquisition and data processing."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime,timedelta,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid


def native_binary():
    from woof import bridges
    name=bridges.executable_name("rw_waves_coupled_prepare")
    def find():return bridges.find_artifact("GPUWM_RW_WAVES_COUPLED_PREPARE",name,crate_relative="tools/waves_data")
    found=find()
    if found is None and bridges.fetch_on_demand("rw_waves_coupled_prepare"):found=find()
    if found is None:
        raise FileNotFoundError("rw_waves_coupled_prepare is required for --waves on; install this release's wave preparation bridge")
    return Path(found)


def _sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


@contextmanager
def _lock(path):
    # Kernel-owned locks release after a crash. A later ordinary launch can
    # retry an interrupted preparation without deleting a stale lock marker.
    with path.open("a+b") as stream:
        if os.name=="nt":
            import msvcrt
            if stream.seek(0,2)==0:stream.write(b"0");stream.flush()
            stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_LOCK,1)
            try:yield
            finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(stream,fcntl.LOCK_UN)


def _grid_file(path,grid,latitude,longitude,state):
    import cupy as cp
    from woof.io.nc_writer_bridge import ClassicSchema
    schema=ClassicSchema("cdf2")
    y=schema.def_dim("y",latitude.shape[0]);x=schema.def_dim("x",latitude.shape[1])
    sina,cosa=grid.rotation_m()
    fields={"latitude":latitude,"longitude":longitude,"cosalpha":cosa,"sinalpha":sina,"map_factor":state.msft}
    ids={name:schema.def_var(name,"f4",(y,x)) for name in fields}
    schema.put_global_attr("Conventions","CF-1.10")
    with schema.create(path) as writer:
        for name,array in fields.items():
            # This is a device byte transfer into the Rust writer, not Python
            # NetCDF encoding, regridding or numerical preparation.
            # Grid rotations can remain NumPy arrays even when the mass-grid
            # coordinates and map factors are already on the device.
            device=cp.asarray(array,dtype=cp.float32)
            writer.write_var(ids[name],cp.asnumpy(cp.ascontiguousarray(device)))


def prepare_atmosphere(*,grid,latitude,longitude,state,physics,start_time,run_seconds,options):
    from .prepared import PreparedWaves
    if state is None:raise ValueError("waves preparation requires the atmosphere map factors")
    if isinstance(start_time,str):start_time=datetime.fromisoformat(start_time.replace("Z","+00:00"))
    if start_time.tzinfo is None:start_time=start_time.replace(tzinfo=timezone.utc)
    start=start_time.astimezone(timezone.utc);end=start+timedelta(seconds=run_seconds)
    cache=Path(os.environ.get("WOOF_WAVES_CACHE",str(Path.home()/".woof"/"cache"/"waves"))).resolve()
    cache.mkdir(parents=True,exist_ok=True)
    binary=native_binary()
    catalog=Path(__file__).with_name("coupled_sources.v1.json")
    shared={role:os.environ[key] for role,key in
            (("bathymetry","WOOF_WAVES_BATHYMETRY"),("stations","WOOF_WAVES_STATIONS")) if os.environ.get(key)}
    with _lock(cache/"preparation.lock"):
        request_dir=cache/"requests"/uuid.uuid4().hex;request_dir.mkdir(parents=True)
        grid_file=request_dir/"atmosphere-grid.nc"
        _grid_file(grid_file,grid,latitude,longitude,state)
        identity={"source_grid_sha256":_sha(grid_file),"start":start.isoformat(),"end":end.isoformat(),
                  "coarsening":options["coarsening"],"stations":options["stations"],
                  "catalog_sha256":_sha(catalog),"binary_sha256":_sha(binary),
                  "shared_inputs":{key:_sha(value) for key,value in shared.items()}}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        entry=cache/"prepared"/key;receipt_path=entry/"receipt.json"
        # A checkpoint-connected shorter leg uses the existing longer forcing
        # window on this identical grid/cycle. Do not prepare another copy just
        # because its stop time is earlier. All source and tool hashes still
        # have to agree, and PreparedWaves verifies the selected array bytes.
        if not receipt_path.is_file():
            physical={k:v for k,v in identity.items() if k!='end'}
            covering=[]
            for candidate in (cache/'prepared').glob('*/receipt.json'):
                previous=json.loads(candidate.read_text())
                prior=previous.get('identity',{})
                if (previous.get('status')=='complete' and
                        {k:v for k,v in prior.items() if k!='end'}==physical and
                        datetime.fromisoformat(prior['end'])>=end):
                    covering.append((prior['end'],candidate))
            if covering:receipt_path=min(covering)[1]
        if receipt_path.is_file():
            receipt=json.loads(receipt_path.read_text())
        else:
            entry.mkdir(parents=True,exist_ok=True)
            # A failed preparation gets its own attempt folder. A retry can
            # reuse source downloads without overwriting any partial arrays.
            attempt=entry/uuid.uuid4().hex
            request={"output":str(attempt),"input_cache":str(cache/"inputs"),
                     "source_catalog":str(catalog),"start":start.isoformat(),"end":end.isoformat(),
                     "coarsening":options["coarsening"],"stations":options["stations"],"shared_inputs":shared,
                     "threads":os.cpu_count() or 1,
                     "grid":{"kind":"curvilinear","file":str(grid_file),"latitude":"latitude","longitude":"longitude",
                             "cosalpha":"cosalpha","sinalpha":"sinalpha","map_factor":"map_factor","dx":grid.dx,"dy":grid.dy}}
            request_path=request_dir/"prepare.json"
            request_path.write_text(json.dumps(request,indent=2)+"\n")
            with (request_dir/"prepare.log").open("w") as log:
                subprocess.run([str(binary),str(request_path)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
            receipt=json.loads((attempt/"receipt.json").read_text())
            receipt["identity"]=identity;receipt["request_path"]=str(request_path)
            temporary=receipt_path.with_suffix(".json.new")
            temporary.write_text(json.dumps(receipt,indent=2)+"\n");temporary.replace(receipt_path)
        if receipt.get("status")!="complete" or receipt.get("source_grid_sha256")!=identity["source_grid_sha256"]:
            raise ValueError("automatic wave preparation receipt does not match the atmosphere grid")
        prepared=PreparedWaves(receipt["prepared"])
        if prepared.manifest["initial_valid_time"]!=start.timestamp() or prepared.manifest["boundary_times"][-1]<end.timestamp():
            raise ValueError("prepared wave spectra do not cover the atmosphere run window")
        coupler=prepared.coupler(options)
        coupler.render_interval=float(options['render_interval_seconds'])
        coupler.preparation_receipt=receipt
        from .products import WaveOutput
        coupler.output=WaveOutput(cache/"runs"/uuid.uuid4().hex,coupler)
        return coupler
