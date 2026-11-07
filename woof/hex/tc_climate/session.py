"""Device diagnostics at existing forecast history boundaries, default off."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import asdict
import json
from pathlib import Path
import time

from . import DeviceTracker, Recipe, genesis_potential_index, qualify_tracks


def device_leaves(value):
    out = {}
    def visit(item):
        if isinstance(item, Mapping):
            for key, val in item.items():
                if isinstance(val, Mapping):
                    visit(val)
                elif hasattr(val, '__cuda_array_interface__'):
                    if key in out:
                        raise ValueError(f'duplicate diagnostic field {key}')
                    out[key] = val
    visit(value)
    return out


def validate_schedule(recipe, schedule, dt_seconds):
    """Require all diagnostic boundaries in history before allocating CUDA."""
    stride = recipe.cadence_hours * 3600 / dt_seconds
    if not stride.is_integer():
        raise ValueError('TC cadence must be an integral number of model steps')
    expected = set(range(0, int(schedule['steps']) + 1, int(stride)))
    if expected.difference(schedule['capture_steps']):
        raise ValueError('TC diagnostics need history at every cadence boundary, including hour zero')
    return len(expected)


class ForecastSession:
    def __init__(self, mesh, terrain, ocean, recipe_path, frame_count, start_time, initial_surface):
        import cupy as cp
        self.cp = cp
        self.recipe = Recipe.load(recipe_path)
        self.tracker = DeviceTracker(mesh.arrays['cellsOnCell'], mesh.arrays['nEdgesOnCell'],
            mesh.arrays['latCell'], mesh.arrays['lonCell'], terrain, self.recipe)
        self.ocean = cp.ascontiguousarray(cp.asarray(ocean), cp.int32)
        # Before the first surface call, the backend has zero placeholders.
        # Use the same initialized winds as the engine's hour-zero history;
        # upload once during setup without modifying atmospheric state.
        self.initial_wind = cp.hypot(cp.asarray(initial_surface['u10'], dtype=cp.float32),
                                    cp.asarray(initial_surface['v10'], dtype=cp.float32))
        if self.ocean.shape != (self.tracker.n,):
            raise ValueError('ocean mask shape differs from the mesh')
        self.pi_module = cp.RawModule(code=Path(__file__).with_name('potential_intensity.cu').read_text(),
                                     options=('--std=c++17',))
        self.frames = []
        self.max_frames = frame_count
        self.start_time = str(start_time)
        self.enqueue_seconds = 0.
        self.events = []

    def sample(self, prepared, diagnostic, atmosphere, time_hours):
        cp, r, tracker = self.cp, self.recipe, self.tracker
        # History may be more frequent than tracker cadence. This scalar clock
        # is already on the host; no device scalar is read here.
        if abs(time_hours/r.cadence_hours-round(time_hours/r.cadence_hours)) > 1e-6:
            return {}
        if len(self.frames) >= self.max_frames:
            raise ValueError('TC record allocation exceeded the planned number of samples')
        started = time.perf_counter()
        event_start, event_end = cp.cuda.Event(), cp.cuda.Event()
        event_start.record()
        from ..s2s_products import interp_log_pressure
        from ..mpasdiag.isobaric import compute_slp
        n = tracker.n
        if prepared.pres_p.shape[1] != n:
            raise ValueError('TC sampler needs an unpadded global atmosphere')
        nz = prepared.pres_p.shape[0]
        if nz > 128:
            raise ValueError('PI local sounding workspace supports at most 128 levels')
        p = cp.ascontiguousarray(prepared.pres_p.T)
        temperature = cp.ascontiguousarray(prepared.t_p.T)
        vapor = cp.ascontiguousarray(prepared.species_p['qv'].T)
        heights = cp.ascontiguousarray(cp.asarray(atmosphere.vertical.zgrid.T), cp.float32)
        slp = compute_slp(cp.ascontiguousarray(p*.01), temperature, heights, vapor, pascals=True)
        z = cp.ascontiguousarray(prepared.zmid_p.T)
        thickness = cp.ascontiguousarray(interp_log_pressure(cp,p,z,30000)-interp_log_pressure(cp,p,z,50000))
        fields = device_leaves(diagnostic)
        # SST may be a prescribed surface-series field. tsk is the sea-surface
        # skin temperature only where the immutable ocean mask is true.
        sst = fields.get('sst', fields.get('sstsk', fields.get('tsk')))
        if sst is None or 'u10' not in fields or 'v10' not in fields:
            raise ValueError('TC diagnostics require resident SST/skin temperature and 10 m winds')
        wind = cp.ascontiguousarray((self.initial_wind if time_hours == 0 else
                                    cp.hypot(fields['u10'], fields['v10'])).reshape(n), cp.float32)
        self.frames.append(tracker.observe(time_hours,slp,thickness,wind))
        output = {'tc_mslp': cp.where(cp.isfinite(slp) & (slp > 0), slp, cp.float32(-9999))}
        if abs(time_hours/24-round(time_hours/24)) < 1e-6:
            u, v = cp.ascontiguousarray(prepared.u_p.T), cp.ascontiguousarray(prepared.v_p.T)
            u850, v850 = interp_log_pressure(cp,p,u,85000), interp_log_pressure(cp,p,v,85000)
            shear = cp.hypot(interp_log_pressure(cp,p,u,20000)-u850,
                             interp_log_pressure(cp,p,v,20000)-v850)
            t600 = interp_log_pressure(cp,p,temperature,60000)
            q600 = interp_log_pressure(cp,p,vapor,60000)
            e600 = q600*600/(.622+q600)
            rh = 100*e600/(6.112*cp.exp(17.67*(t600-273.15)/(t600-29.65)))
            # Supersaturation is represented as saturation, with missing inputs
            # preserved. The standalone GPI API instead rejects invalid RH.
            rh = cp.clip(rh,0,100)
            pi, status = cp.empty(n,cp.float32),cp.empty(n,cp.int32)
            self.pi_module.get_function('potential_intensity')(((n+127)//128,), (128,),
                (n,nz,prepared.pres_p,prepared.t_p,prepared.species_p['qv'],
                 cp.ascontiguousarray(sst.reshape(n), cp.float32),slp,self.ocean,
                 cp.float32(r.pi_ckcd),cp.float32(r.pi_wind_reduction),
                 cp.float32(r.pi_top_hpa),pi,status))
            eta = cp.empty(n,cp.float32)
            self.pi_module.get_function('vorticity850')(((n+127)//128,), (128,),
                (n,tracker.graph.shape[1],tracker.graph,tracker.counts,tracker.lat,tracker.lon,
                 cp.ascontiguousarray(u850),cp.ascontiguousarray(v850),eta))
            gpi = genesis_potential_index(cp,eta,rh,pi,shear)
            # Finite CF missing-value sentinel keeps the atmospheric snapshot
            # physical gate intact. Land/invalid PI is never published as zero.
            output.update({'tc_gpi':cp.where(cp.isfinite(gpi),gpi,cp.float32(-9999)),
                      'tc_potential_intensity':cp.where(cp.isfinite(pi),pi,cp.float32(-9999)),
                      'tc_pi_status':status.astype(cp.float32)})
        self.enqueue_seconds += time.perf_counter()-started
        event_end.record()
        self.events.append((event_start, event_end))
        return output

    def finish(self, out):
        """After integration, before device retirement: download small records."""
        cp = self.cp
        records = []
        candidate_counts = []
        for frame in self.frames:
            candidate_counts.append(int(cp.asnumpy(frame['candidate_count'])[0]))
            errors = int(cp.asnumpy(frame['errors'])[0])
            if errors:
                raise RuntimeError(f'TC diagnostic capacity overflow, status bits {errors}, candidates {candidate_counts[-1]}; no valid count')
            arrays = {k:cp.asnumpy(v) for k,v in frame.items() if k not in ('time_hours','errors')}
            for i, tid in enumerate(arrays['ids']):
                if tid < 0:
                    continue
                record = {'time_hours':frame['time_hours'],'track_id':int(tid),
                          'cell':int(arrays['cells'][i])}
                for key in ('latitude_deg','longitude_deg','mslp_pa','wind10_ms','terrain_m'):
                    record[key] = float(arrays[key][i])
                records.append(record)
        tracks = qualify_tracks(records,self.recipe)
        device_ms = [float(cp.cuda.get_elapsed_time(a,b)) for a,b in self.events]
        value = {'schema':'gpuwm-hex.tc-climate/v1','recipe':asdict(self.recipe),
                 'start_time':self.start_time,'time_units':'hours since start_time',
                 'mesh_cells':self.tracker.n,
                 'records':records,'qualified_tracks':tracks,
                 'enqueue_seconds':self.enqueue_seconds,
                 'diagnostic_stream_interval_ms':device_ms,
                 'timing_scope':'CPU enqueue wall time; not GPU execution time or added whole-step cost',
                 'samples':len(self.frames),'track_count':len(tracks),
                 'candidate_counts':candidate_counts,
                 'pi_method':'reversible tcpyPI CUDA adaptation with dissipative heating; parameters in recipe',
                 'gpi_cadence_hours':24,'association':'online globally nearest pair, deterministic one-to-one'}
        target = Path(out)/'tc-climate.json'
        target.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        return {'path':str(target),'samples':len(self.frames),'track_count':len(tracks),
                 'candidate_counts':candidate_counts,
                'enqueue_seconds':self.enqueue_seconds,
                'diagnostic_stream_interval_ms':sum(device_ms)}
