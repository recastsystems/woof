"""Instance-local observer of unchanged native scalar stages.

This is an admission instrument, not an active full-step integrator. The
orchestration code object runs unchanged with a private globals dictionary.
Only its scalar transport entry is wrapped. No native/global object is patched.
"""
import hashlib
import types
from pathlib import Path
import numpy as np
from .fullstep_lts import geometric_rates, buffer_rates
from .cuda_courant_fullstep_lts import CudaColumnTransportMonitor

# Re-audited 2026-10-24 (HEX-PHYSICS-PARITY fix group 2, #8): both calls of advance_scalar_transport_cuda_v841 in
# cuda_driver.py (the global path and the regional third stage) still pass source-old rho at position 4, the split
# fluxes at 6 and 7 and the stage duration at 8, with rdzw and rk_step as keywords.  The previous pin (05e87df6...)
# matched no shipped driver since 0.4.1, so the shadow refused every variable-mesh run on both physics routes.
# Re-audited again for 0.4.5: the limited-area suite change to cuda_driver.py touches the moist coefficients only;
# both transport calls keep the same argument positions.  tests/test_fullstep_lts_shadow_pin.py now fails on any
# driver change, so the pin is re-audited at merge time instead of refusing a user's run.
# Re-audited 2026-10-28 (HEX-LTS-REGRESSION, deferred class-step validation): the change adds
# flush_deferred_validation_v841 and a deferred branch around the step's validation read; both transport calls
# keep the same argument positions.
# Re-audited 2026-10-29 (rel/hex-0.4.10 fold HB9, the 44c0849 re-pin): the scalar tendency both calls pass is
# now scalar_source (physics_tendencies.scalars unless NOAA config_scalar_eddy_mix adds its source); both
# transport calls keep the same argument positions.
# Re-audited 2026-10-29 (rel/hex-0.4.10 fold HB11, rrfs/one-tree-fast 39a1af5b): the change lets the limited-area
# lane bind its start levels (_shares_start_levels, regional_state_levels_enabled); a driver with local_timestep
# attached still keeps the images, and neither transport call is touched.
# Re-audited 2026-10-29 (fix/rrfs-shared-dycore-8card): the change adds one branch at the step's validation read
# (deferred_checks_v842 inside an MPAS-physics forecast loop; the LTS list branch still comes first); both transport
# calls keep the same argument positions and neither is touched.
# Re-audited 2026-10-30 (woof hex 0.4.11 fold HB2, feat/hex-moving-resident 78111207): the change adds an optional
# upload_cache passthrough to the from-host constructor only; the LTS list branch still comes first and both transport
# calls keep the same argument positions, neither touched.
NATIVE_DRIVER_SHA256 = 'e8d3c8484928c85f126dfce6bacb35ba215cf59dc7005ded780b96aa639f0f42'


class FullStepShadow:
    def __init__(self, driver, *, dt, max_rate=16, target=.90, global_cell_ids=None, sink=None):
        import inspect
        cp = driver.cp
        self.cp, self.driver, self.dt, self.sink = cp, driver, dt, sink
        self.target = target
        source = Path(inspect.getfile(type(driver)))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != NATIVE_DRIVER_SHA256:
            raise ValueError(f'native driver source {digest} changed: this observer could misidentify source-old density or accepted split-flux arguments; re-audit the native call before monitoring')
        mesh = driver.atmosphere.mesh
        self.monitor = CudaColumnTransportMonitor(cp, mesh, driver.atmosphere.state.rho.shape[0])
        if global_cell_ids is None:
            raise ValueError('shadow needs explicit source-grid global cell IDs; local ordinals may be permuted')
        self.global_cell_ids = np.asarray(global_cell_ids)
        if self.global_cell_ids.dtype.kind not in 'iu' or np.any(self.global_cell_ids <= 0) or self.global_cell_ids.shape != (self.monitor.nc,) or np.unique(self.global_cell_ids).size != self.monitor.nc:
            raise ValueError('cell ID mapping must identify every owned column once')
        def host(name):
            value = getattr(mesh, name)
            return cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.asarray(value)
        self.coe = host('cells_on_edge')
        geometric = geometric_rates(host('dc_edge'), self.coe, self.monitor.nc, max_rate)
        self.raw_geometric = geometric
        self.geometric = buffer_rates(geometric, self.coe)
        self.device_geometric = cp.asarray(self.raw_geometric)
        self.device_coe = cp.asarray(self.coe)
        self.worst = cp.zeros(self.monitor.nc, dtype=cp.float64)
        self.samples = 0
        self.last = None
        self.peak = None
        self.original = driver._step_device_v841
        original = self.original.__func__
        native = original.__globals__['advance_scalar_transport_cuda_v841']
        def observed(*args, **kwargs):
            # Positional arguments are source-old rho, actual split fluxes,
            # and the immutable scalar RK duration in cuda_driver.py:4161.
            self.observe(args[4], args[6], args[7], kwargs['rdzw'],
                         stage=kwargs['rk_step'], stage_dt=float(args[8]),
                         time_seconds=float(driver.atmosphere.state.time_seconds))
            return native(*args, **kwargs)
        private_globals = dict(original.__globals__)
        private_globals['advance_scalar_transport_cuda_v841'] = observed
        instrumented = types.FunctionType(original.__code__, private_globals,
                                         original.__name__, original.__defaults__, original.__closure__)
        instrumented.__kwdefaults__ = original.__kwdefaults__
        driver._step_device_v841 = types.MethodType(instrumented, driver)

    def observe(self, rho, uh, ww, rdzw, *, stage, stage_dt, time_seconds):
        cp = self.cp
        out, levels = self.monitor.observe(rho, uh, ww, rdzw)
        cp.maximum(self.worst, out[0], out=self.worst)
        index = int(cp.argmax(out[0]).item())
        self.samples += 1
        rates = cp.ones(self.monitor.nc, dtype=cp.int32)
        for rate in (1, 2, 4, 8, 16):
            rates[(self.device_geometric >= rate) & (self.dt*rate*out[0] <= self.target)] = rate
        for _ in range(2):
            previous = rates.copy()
            cp.minimum.at(rates,self.device_coe[:,0],previous[self.device_coe[:,1]])
            cp.minimum.at(rates,self.device_coe[:,1],previous[self.device_coe[:,0]])
        row = dict(sample=self.samples, stage=stage, stage_dt=stage_dt, time_seconds=time_seconds,
                   transport_rate=float(out[0,index].item()), cell=index, level=int(levels[index].item()),
                   global_cell_id=int(self.global_cell_ids[index]),
                   stage_depletion=float(stage_dt*out[0,index].item()),
                   outer_depletion=float(self.dt*out[0,index].item()),
                   required_base_dt=float(self.target/out[0,index].item()) if float(out[0,index].item()) > 0 else None,
                   horizontal_rate_max=float(cp.max(out[1]).item()),
                   vertical_rate_max=float(cp.max(out[2]).item()),
                   base_infeasible_cells=int(cp.count_nonzero(self.dt*out[0] > self.target).item()),
                   optimistic_transport_counts={str(r): int(cp.count_nonzero(rates == r).item()) for r in (1,2,4,8,16)})
        self.last = row
        if self.peak is None or row['transport_rate'] > self.peak['transport_rate']:
            self.peak = row.copy()
        if self.sink is not None:
            self.sink(row)
        return row

    def receipt(self):
        from .fullstep_lts import transport_ceiling
        worst = self.cp.asnumpy(self.worst)
        rates, infeasible = transport_ceiling(self.raw_geometric, worst, self.dt, self.target)
        rates = buffer_rates(rates, self.coe)
        index = int(np.argmax(worst))
        negative_gate = None
        if np.any(infeasible) and np.any(self.geometric > 1):
            negative_gate = dict(status='base_transport_infeasible',global_cell_id=int(self.global_cell_ids[index]),
                transport_rate=float(worst[index]),outer_depletion=float(self.dt*worst[index]),
                required_base_dt=float(self.target/worst[index]),
                controlling_sample=self.peak,
                reason='the base local step exceeds the donor bound; promoting the limiter is not a valid timestep admission')
        return dict(schema='gpuwm-hex.fullstep-lts/v1', mode='shadow', samples=self.samples,
                    native_driver_sha256=NATIVE_DRIVER_SHA256, last=self.last,
                    transport_target=self.target,
                    negative_gate=negative_gate,
                    geometric_counts={str(r): int(np.count_nonzero(self.geometric == r)) for r in (1,2,4,8,16)},
                    optimistic_transport_counts={str(r): int(np.count_nonzero(rates == r)) for r in (1,2,4,8,16)},
                    base_infeasible_cells=int(np.count_nonzero(infeasible)),
                    optimistic_cell_work_ratio=float(rates.size/np.sum(1./rates)),
                    admission_complete=False,
                    unproved_gates=['advection/acoustics/mixing/physics admission', 'complete six-hour shadow',
                                   'active interface conservation', 'whole-operator cost', '48-hour score'])

    def detach(self):
        self.driver._step_device_v841 = self.original
