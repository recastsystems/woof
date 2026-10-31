"""Native analysis carriers and GPU forcing-boundary refresh."""
from pathlib import Path
import hashlib
import json
import os
import time
from bisect import bisect_right
from types import SimpleNamespace
import numpy as np

FIELDS = ('u', 'v', 'thp', 'thb', 'pressure', 'php', 'phb', 'qv')


def series_path(root, grid_id=1):
    root = Path(root)
    return root.with_name(root.name + '-nudging') / f'd{grid_id:02d}'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    part = path.with_suffix(path.suffix + '.part')
    part.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    os.replace(part, path)


def selected(exp, grid_id=None):
    options = getattr(exp, 'nudging', None)
    return None if options is None else options.for_domain(exp.root.grid_id if grid_id is None else grid_id)


def analysis_writer(root, exp, times, source_sha256, domain=None):
    domain = exp.root if domain is None else domain
    if selected(exp, domain.grid_id) is None:
        return None
    return AnalysisWriter(root, exp, times, source_sha256, domain=domain)


def analysis_writer_from_request(args, exp, chain, times):
    """Bind a native preparer's portable output to its source manifest."""
    if selected(exp) is None:
        return None
    if not args.prepare_only:
        raise ValueError('NUDGING_NATIVE_ANALYSIS_STORE: native nudging needs prepared go inputs so its analyses remain with the portable forcing bundle')
    if getattr(args,'as_posted',None) is not None:
        raise ValueError('NUDGING_UNSEALED_ANALYSIS_IDENTITY: an as-posted input plan changes its source digest at seal; prepare a complete source window with --whole-cycle before nudging')
    published=getattr(args,'publish_experiment_config',None)
    root=Path(chain['output_root']) if chain is not None else Path(published).parent if published is not None else None
    source_sha=getattr(args,'source_manifest_sha256',None)
    if root is None or not source_sha:
        raise ValueError('NUDGING_PORTABLE_SOURCE_IDENTITY: native analyses require a portable output root and a sealed source-manifest digest')
    return analysis_writer(root,exp,times,source_sha)


class AnalysisWriter:
    """Publish initialized carriers through the Rust parallel array writer."""
    def __init__(self, root, exp, times, source_sha256, *, domain=None):
        from woof.ingest.prepared_writer import native_writer, batch_budget
        domain = exp.root if domain is None else domain
        cfg = domain.run
        controls = selected(exp, domain.grid_id)
        if controls is None:
            raise ValueError('analysis writer requires an enabled domain')
        if cfg.use_adaptive_time_step:
            raise ValueError('NUDGING_ADAPTIVE_CLOCK: captured nudging needs the actual step clock; use a fixed time step')
        offsets = [(t - times[0]).total_seconds() for t in times]
        if times[0] != exp.start_time:
            raise ValueError('NUDGING_ANALYSIS_ORIGIN: the first analysis must match the experiment start time')
        if len(offsets) < 2 or offsets[0] != 0 or any(b <= a for a,b in zip(offsets,offsets[1:])):
            raise ValueError('nudging analyses must have ordered, distinct bracketing times')
        if any(abs(t/cfg.dt-round(t/cfg.dt)) > 1.e-8 for t in offsets):
            raise ValueError('NUDGING_FORCING_CLOCK: analysis boundaries must coincide with model steps')
        indices = list(range(len(offsets)))
        if controls.gfdda_interval_m is not None:
            cadence = controls.gfdda_interval_m*60.
            indices = [i for i,t in enumerate(offsets) if abs(t/cadence-round(t/cadence)) < 1.e-8]
            chosen = [offsets[i] for i in indices]
            if len(chosen) < 2 or any(abs(b-a-cadence)>1.e-6 for a,b in zip(chosen,chosen[1:])):
                raise ValueError('NUDGING_ANALYSIS_CADENCE: gfdda_interval_m requests analysis times absent from the source frames')
            offsets = chosen
        required = min(exp.run_seconds, controls.gfdda_end_h*3600.) if controls.gfdda_end_h is not None else exp.run_seconds
        if offsets[-1] < required:
            raise ValueError('NUDGING_ANALYSIS_COVERAGE: source frames do not bracket the requested nudging period')
        if (controls.gfdda_end_h is not None and controls.gfdda_end_h*3600. < exp.run_seconds
                and controls.gfdda_end_h*3600. not in offsets):
            raise ValueError('NUDGING_END_ANALYSIS: gfdda_end_h must coincide with an analysis time so the end ramp uses the requested final analysis')
        self.source_indices = {source:target for target,source in enumerate(indices)}
        self.root = series_path(root, domain.grid_id)
        self.root.mkdir(parents=True, exist_ok=True)
        self.header = dict(schema='gpuwm-nudging-analysis-v2', start_time=exp.start_time.isoformat(), offsets=offsets, shape=[cfg.nz,cfg.ny,cfg.nx], source_sha256=source_sha256)
        marker = self.root / 'series.json'
        if marker.exists() and json.loads(marker.read_text()) != self.header:
            raise ValueError('nudging analysis directory belongs to different forcing or geometry')
        atomic_json(marker, self.header)
        self.writer = native_writer()
        if self.writer is None:
            raise RuntimeError('NUDGING_NATIVE_WRITER: install a runtime with the Rust prepared-array writer')
        self.workers, self.batch_bytes = batch_budget()
        self.seconds = 0.

    def write(self, index, initialized):
        from woof.ingest.prepared_writer import write_arrays
        if index not in self.source_indices:
            return
        index = self.source_indices[index]
        started = time.perf_counter()
        arrays = {name:getattr(initialized.state,name) for name in FIELDS if name != 'pressure'}
        arrays['pressure'] = initialized.total_pressure
        if any(not isinstance(value,np.ndarray) for value in arrays.values()):
            import cupy as cp
            arrays = {name:cp.asnumpy(value) for name,value in arrays.items()}
        rows = [(self.root/f'{index:05d}-{name}.npy.part',arrays[name]) for name in FIELDS]
        hashes = write_arrays(self.writer,rows,workers=self.workers)
        files = {}
        for name,(part,value),sha in zip(FIELDS,rows,hashes):
            path = part.with_suffix('')
            os.replace(part,path)
            files[name] = dict(file=path.name,sha256=sha,shape=list(value.shape),dtype=str(value.dtype))
        atomic_json(self.root/f'{index:05d}.json',dict(files=files,offset_seconds=self.header['offsets'][index]))
        self.seconds += time.perf_counter()-started


def allocate_held_tendencies(state,cfg,controls):
    """Allocate persistent cadence buffers before restoring a checkpoint."""
    if controls is None or controls.cadence_steps(cfg.dt) == 1:
        return
    import cupy as cp
    for name,target in (('u',state.u),('v',state.v),('temperature',state.thp),('geopotential',state.php),('qv',state.qv)):
        if controls.coefficient(name) > 0:
            key = 'nudging_tendency_'+name
            if getattr(state,key,None) is None:
                setattr(state,key,cp.zeros_like(target))


class PreparedNudging:
    """Refresh endpoints only when the existing lateral forcing slot changes."""
    def __init__(self,state,cfg,exp,prepared_root,source_sha256,*,grid_id=None):
        from woof.ingest.lateral_bc import attach_streaming_lateral_boundaries
        from .forecast import attach
        grid_id = exp.root.grid_id if grid_id is None else grid_id
        self.controls = selected(exp,grid_id)
        self.state,self.cfg = state,cfg
        self.root = series_path(prepared_root,grid_id)
        self.header = json.loads((self.root/'series.json').read_text())
        if (self.header['schema'] != 'gpuwm-nudging-analysis-v2' or self.header['start_time'] != exp.start_time.isoformat() or self.header['shape'] != [cfg.nz,cfg.ny,cfg.nx] or self.header['source_sha256'] != source_sha256):
            raise ValueError('nudging analysis identity differs from the prepared forecast')
        self.offsets = self.header['offsets']
        self.loads = 0
        self.refresh_seconds = 0.
        self.frame_digests = {}
        self.plan = None
        elapsed = state.elapsed_seconds
        if cfg.specified:
            attach_streaming_lateral_boundaries(state,state.lateral_boundaries,allow_prestaged=False)
            state.elapsed_seconds = elapsed
        interval = self.analysis_interval(elapsed)
        self.active_interval = (interval.start_seconds,interval.end_seconds)
        allocate_held_tendencies(state,cfg,self.controls)
        self.plan = attach(state,cfg,self.controls,self.endpoints(interval),interval_start=interval.start_seconds,interval_end=interval.end_seconds)
        state._spectral_nudging_interval = self.replace

    def frame(self,index):
        import cupy as cp
        from woof.ingest.prepared_mmap import map_npy_readonly
        from woof.ingest.prepared_writer import native_hasher,hash_arrays,batch_budget
        marker = self.root/f'{index:05d}.json'
        meta = json.loads(marker.read_text())
        if meta['offset_seconds'] != self.offsets[index] or set(meta['files']) != set(FIELDS):
            raise ValueError('nudging analysis inventory or clock differs')
        host = {}
        for name in FIELDS:
            entry = meta['files'][name]
            path = self.root/f'{index:05d}-{name}.npy'
            if entry['file'] != path.name:
                raise ValueError('nudging analysis file name differs')
            value = map_npy_readonly(path)
            if list(value.shape) != entry['shape'] or str(value.dtype) != entry['dtype']:
                raise ValueError('nudging analysis shape or precision differs')
            host[name] = value
        hasher = native_hasher()
        if hasher is None:
            raise RuntimeError('NUDGING_NATIVE_HASHER: analysis integrity requires the native prepared-array hasher')
        hashes = hash_arrays(hasher,list(host.values()),workers=batch_budget()[0])
        if any(sha != meta['files'][name]['sha256'] for name,sha in zip(FIELDS,hashes)):
            raise ValueError('nudging analysis checksum failed')
        values = {name:cp.asarray(a,dtype=cp.float32) for name,a in host.items()}
        for name in ('thb','phb'):
            if values[name].ndim == 1:
                values[name] = values[name][:,None,None]
        self.frame_digests[str(index)] = digest(marker)
        temperature = values['thp']+values['thb']
        if self.controls.convention == 'regional':
            temperature *= (values['pressure']/cp.float32(100000.))**cp.float32(287./1004.5)
        return dict(u=values['u'],v=values['v'],temperature=temperature,geopotential=values['php']+values['phb'],qv=values['qv'])

    def endpoints(self,interval):
        a,b = self.frame(self.offsets.index(interval.start_seconds)),self.frame(self.offsets.index(interval.end_seconds))
        return {name:(a[name],b[name]) for name in a if self.controls.coefficient(name)>0}

    def analysis_interval(self,elapsed):
        end = self.controls.gfdda_end_h
        when = elapsed if end is None else min(elapsed,end*3600.)
        index = min(len(self.offsets)-2,max(0,bisect_right(self.offsets,when)-1))
        if end is not None and when >= end*3600. and index > 0 and self.offsets[index] == end*3600.:
            index -= 1
        return SimpleNamespace(start_seconds=self.offsets[index],end_seconds=self.offsets[index+1])

    def replace(self,interval):
        started = time.perf_counter()
        elapsed = interval.start_seconds
        interval = self.analysis_interval(elapsed)
        identity = (interval.start_seconds,interval.end_seconds)
        if identity == self.active_interval:
            return
        self.plan.replace_interval(self.endpoints(interval),elapsed_seconds=elapsed,dt_seconds=self.cfg.dt,interval_start=interval.start_seconds,interval_end=interval.end_seconds)
        self.active_interval = identity
        self.loads += 1
        self.refresh_seconds += time.perf_counter()-started

    def receipt(self):
        self.plan.check_at_seam()
        count = int(self.plan.stage_count.get()[0])
        stages = {name:float(value)*1.e-6/max(1,count) for name,value in zip(self.plan.bindings,self.plan.stage_sums_ns.get())}
        return dict(enabled=True,config=self.controls.to_dict(),forcing_refreshes=self.loads,forcing_refresh_host_seconds=self.refresh_seconds,forcing_frame_sha256=self.frame_digests,pbl_mask='live production physics PBL height',timing_scope='included in forecast and lateral forcing boundary timings',operator_steps=count,operator_ms_per_step=stages,refresh_ms_per_step=self.refresh_seconds*1000/max(1,count))

    def detach(self):
        self.state._spectral_nudging_interval = None
        self.state._spectral_nudging_graph = None


def validate_fixed_domains(exp):
    """Prepared analysis grids cannot follow a relocated/replaced state."""
    if getattr(exp,'nudging',None) is None:
        return
    for domain in exp.domains:
        if selected(exp,domain.grid_id) is None:
            continue
        if (getattr(domain,'spawn',None) is not None or
                getattr(domain,'follow',None) is not None or
                (domain.start_time is not None and domain.start_time != exp.start_time)):
            raise ValueError('NUDGING_FIXED_ANALYSIS_GRID: a delayed, spawned or moving domain replaces the state and geometry bound by its prepared analyses; use a fixed domain starting with the root')
        relocation=getattr(exp,'relocation',None)
        if domain.parent_id and relocation is not None and relocation.enabled:
            raise ValueError('NUDGING_FIXED_ANALYSIS_GRID: moving nests need analyses remapped at each move; fixed-grid analyses cannot follow them')


def write_nested_analyses(root,exp,times,source_sha256,*,catalog,root_grid,root_at,
                          source_orography=None,workers=8,preprocess_backend='cpu',
                          cpu_bridge=None,sfcp_to_sfcp=True,soil_layer_contract=None):
    """Use the existing native child initializer at each source analysis time."""
    chosen=[d for d in exp.domains[1:] if selected(exp,d.grid_id) is not None]
    if not chosen:
        return
    from dataclasses import replace
    from woof.ingest.nest_init import ParentInitView,initialize_child_chain_parallel
    validate_fixed_domains(exp)
    writers={d.grid_id:AnalysisWriter(root,exp,times,source_sha256,domain=d) for d in chosen}
    indices=sorted(set().union(*(set(w.source_indices) for w in writers.values())))
    for index in indices:
        instant=times[index]
        domains=tuple(replace(d,start_time=instant) for d in exp.domains)
        clocked=replace(exp,start_time=instant,domains=domains)
        _met,initialized=root_at(index)
        parent=ParentInitView(cfg=domains[0],grid=root_grid,state=initialized.state)
        children=initialize_child_chain_parallel(clocked,parent,catalog,source_orography,
            workers=workers,preprocess_backend=preprocess_backend,cpu_bridge=cpu_bridge,
            state_backend='preprocess',sfcp_to_sfcp=sfcp_to_sfcp,
            soil_layer_contract=soil_layer_contract)
        for domain,child in zip(domains[1:],children,strict=True):
            if domain.grid_id in writers:
                writers[domain.grid_id].write(index,child.real)


class PreparedTreeNudging:
    """Fan out the root's existing forcing seam to explicitly selected nests."""
    def __init__(self,model,nodes,inputs,steppers):
        exp=inputs.experiment
        validate_fixed_domains(exp)
        self.owners=[]
        chosen=[node for gid,node in nodes.items() if selected(exp,gid) is not None]
        if any(int(node.cfg.grid_id) in steppers for node in chosen):
            raise ValueError('NUDGING_FULL_HORIZONTAL_FFT: a selected domain must be resident; tile-local FFTs change the requested wavelengths')
        root=model.root
        if selected(exp,root.cfg.grid_id) is None:
            from woof.ingest.lateral_bc import attach_streaming_lateral_boundaries
            elapsed=root.state.elapsed_seconds
            attach_streaming_lateral_boundaries(root.state,root.state.lateral_boundaries,allow_prestaged=False)
            root.state.elapsed_seconds=elapsed
        bundles={b.grid_id:b for b in inputs.domains}
        for node in chosen:
            owner=PreparedNudging(node.state,node.cfg.run,exp,inputs.prepared_root,
                bundles[node.cfg.grid_id].cache_identity['source_manifest_sha256'],grid_id=node.cfg.grid_id)
            self.owners.append(owner)
        self.root=root.state
        self.root._spectral_nudging_interval=self.replace

    def replace(self,interval):
        for owner in self.owners:
            owner.replace(interval)

    def receipt(self):
        return {str(owner.cfg.grid_id):owner.receipt() for owner in self.owners}

    def detach(self):
        for owner in self.owners:
            owner.detach()
        self.root._spectral_nudging_interval=None
