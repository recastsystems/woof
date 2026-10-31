"""Conservative terrain-aligned physical-height integration for moving meshes.

Each horizontal overlap aligns the source column's ground and top with the
target endpoints. Interior target interfaces are then evaluated in the
source's physical height coordinate. Integrated, limited linear profiles
preserve each source column total, including when vertical surfaces differ.
"""
from __future__ import annotations
import os
import numpy as np


def _select_backend(backend):
    """Use CUDA by default when visible; CPU is an explicit reference backend.

    CPU-only preparation still works without importing CuPy when devices are
    hidden. CUDA execution failures propagate instead of retrying on the CPU.
    The retired moving GPU environment switches have no effect.
    """
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("height map backend must be auto, cpu or cuda")
    if backend != "auto":
        return backend
    import os
    if (os.environ.get("CUDA_VISIBLE_DEVICES") == "" or
            any(os.environ.get(key) == "1" for key in
                ("WOOF_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU"))):
        return "cpu"
    try:
        import cupy as cp
    except ImportError:
        return "cpu"
    return "cuda" if cp.cuda.runtime.getDeviceCount() else "cpu"


def _positive_cumulative_roundoff(cumulative, *, absolute_height=False):
    """Close a positive column's cumulative integrals within FP64 roundoff.

    Positive limited profiles have monotone cumulative mass. Independent
    evaluations near an empty layer can reverse that ordering by a few ulps.
    Reconcile only those bounded arithmetic discrepancies, keeping both
    endpoints unchanged, before differencing into layer masses.
    """
    bottom, top = cumulative[0], cumulative[-1]
    if np.any(top < bottom) or (not absolute_height and (np.any(top < 0) or np.any(bottom != 0))):
        raise ValueError('positive cumulative endpoints are invalid')
    # Absolute-height columns can start below the donor ground (negative
    # cumulative mass), or above it (positive). Preserve both endpoints and
    # allow only the same bounded cancellation as an aligned column.
    corrected = np.maximum.accumulate(np.clip(cumulative, bottom[None, :], top[None, :]), axis=0)
    change = abs(corrected-cumulative)
    scale = np.maximum(abs(bottom), abs(top))
    allowance = 128*np.finfo(np.float64).eps*scale
    if np.any(change > allowance[None, :]):
        raise ValueError('positive cumulative reversal exceeds FP64 arithmetic allowance')
    return corrected, dict(changed_interfaces=int(np.count_nonzero(change)),
        maximum_integrated_adjustment=float(change.max()),
        maximum_column_relative_adjustment=float(np.max(np.divide(change,scale[None,:],
            out=np.zeros_like(change),where=scale[None,:]!=0))),
        column_endpoints_preserved=True)


def positive_cumulative_roundoff(cumulative, *, absolute_height=False):
    """Reconcile independent columns in bounded parallel blocks, preserving bytes."""
    from .host_parallel import run_blocks, MIN_PARALLEL_ELEMENTS, threads
    if cumulative.size < MIN_PARALLEL_ELEMENTS or threads() == 1:
        return _positive_cumulative_roundoff(cumulative, absolute_height=absolute_height)
    corrected = np.empty_like(cumulative)
    receipts = []
    def reconcile(block):
        values, receipt = _positive_cumulative_roundoff(
            cumulative[:, block], absolute_height=absolute_height)
        corrected[:, block] = values
        receipts.append(receipt)
    run_blocks(reconcile, cumulative.shape[1], elements=cumulative.size)
    return corrected, dict(
        changed_interfaces=sum(row['changed_interfaces'] for row in receipts),
        maximum_integrated_adjustment=max(row['maximum_integrated_adjustment'] for row in receipts),
        maximum_column_relative_adjustment=max(row['maximum_column_relative_adjustment'] for row in receipts),
        column_endpoints_preserved=True)


def _source_profiles(weights, values, density, positive):
    """Keep each column's reconstruction and cumulative addition order intact.

    Whole-field NumPy reconstruction formerly allocated several full-size
    temporaries and ran on one CPU. Bounded independent column blocks use
    the existing host worker budget, including on the default CUDA route.
    """
    from .host_parallel import run_blocks
    means = np.empty_like(values)
    slopes = np.empty_like(values)
    cumulative = np.empty((len(values)+1, values.shape[1]), dtype=np.float64)
    cumulative[0] = 0.
    def reconstruct(block):
        value = values[:, block]
        dz = weights.source_dz[:, block]
        centres = weights.source_centres[:, block]
        mean = value*weights.source_dzw[:, None]/dz
        slope = np.zeros_like(mean)
        if len(mean)>1:
            left = np.diff(mean, axis=0)/np.diff(centres, axis=0)
            if not weights.endpoint_alignment:
                slope[0] = left[0]; slope[-1] = left[-1]
        if len(mean)>2:
            central = (mean[2:]-mean[:-2])/(centres[2:]-centres[:-2])
            a, b, c = 2*left[:-1], central, 2*left[1:]
            same = (np.signbit(a)==np.signbit(b)) & (np.signbit(b)==np.signbit(c))
            slope[1:-1] = np.where(same, np.copysign(
                np.minimum(np.minimum(abs(a), abs(b)), abs(c)), b), 0.)
        if positive:
            bound = 2*mean/dz
            slope = np.clip(slope, -bound, bound)
            if not weights.endpoint_alignment:
                slope[0] = np.minimum(slope[0], mean[0]/weights.source_bottom_reach[block])
        means[:, block] = mean
        slopes[:, block] = slope
        cumulative[1:, block] = np.cumsum(value*weights.source_dzw[:, None], axis=0)
    run_blocks(reconstruct, values.shape[1], elements=values.size)
    if density is not None:
        slopes[0] = values[0]/density[0]*weights._lowest_density_slope(density)
    return means, slopes, cumulative


class HeightConservativeMap:
    redistributes_vertical = True

    def __init__(self, horizontal, source_zgrid, target_zgrid, source_dzw, target_dzw, *,
                 endpoint_alignment=True, domain_reconciliation='global', domain_support_depth_m=1000.,
                 backend='auto'):
        if domain_reconciliation not in ('global','local'):
            raise ValueError('height map domain reconciliation must be global or local')
        if domain_reconciliation=='local' and endpoint_alignment:
            raise ValueError('local domain reconciliation requires absolute-height transfer')
        if not np.isfinite(domain_support_depth_m) or domain_support_depth_m<=0:
            raise ValueError('local domain support depth must be finite and positive')
        self.backend=_select_backend(backend)
        self.horizontal=horizontal
        self.source_area=horizontal.source_area
        self.target_area=horizontal.target_area
        self.overlap=horizontal.overlap
        self.endpoint_alignment=endpoint_alignment
        self.domain_reconciliation=domain_reconciliation
        self.domain_support_depth_m=float(domain_support_depth_m)
        self.last_domain_reconciliation=None
        self.execution_timings=[]
        self.below_ground=None
        self.source_zgrid=np.asarray(source_zgrid,dtype=np.float64)
        self.target_zgrid=np.asarray(target_zgrid,dtype=np.float64)
        self.source_dzw=np.asarray(source_dzw,dtype=np.float64)
        self.target_dzw=np.asarray(target_dzw,dtype=np.float64)
        sz,tz=self.source_zgrid,self.target_zgrid
        if sz.shape!=(len(self.source_dzw)+1,len(self.source_area)) or tz.shape!=(len(self.target_dzw)+1,len(self.target_area)):
            raise ValueError('height map coordinate dimensions differ from the source/target mass grid')
        if not np.isfinite(sz).all() or not np.isfinite(tz).all() or np.any(np.diff(sz,axis=0)<=0) or np.any(np.diff(tz,axis=0)<=0):
            raise ValueError('height map requires finite increasing physical interfaces')
        if np.any(self.source_dzw<=0) or np.any(self.target_dzw<=0):
            raise ValueError('height map nominal layer thickness must be positive')
        coo=self.overlap.tocoo();self.rows=coo.row;self.cols=coo.col;self.areas=coo.data
        ns=len(self.source_dzw)
        if ns>255:raise ValueError('height map has more than 255 source layers')
        ground_s,top_s=sz[0,self.cols],sz[-1,self.cols]
        ground_t,top_t=tz[0,self.rows],tz[-1,self.rows]
        self.sample_height=np.empty((len(self.target_dzw)+1,len(self.areas)),dtype=np.float64)
        self.sample_layer=np.empty(self.sample_height.shape,dtype=np.uint8)
        def stencil(level):
            # Levels are independent. Bound each worker's donor gathers so
            # a 55-level million-overlap move does not allocate whole-mesh
            # binary-search buffers on every worker at once.
            for start in range(0,len(self.areas),32768):
                stop=min(start+32768,len(self.areas))
                c=self.cols[start:stop]; r=self.rows[start:stop]
                gs=ground_s[start:stop]; gt=ground_t[start:stop]
                ts=top_s[start:stop]; tt=top_t[start:stop]
                height=gs+(tz[level,r]-gt)*(ts-gs)/(tt-gt)
                if not endpoint_alignment:height=tz[level,r].copy()
                elif level==0:height=gs.copy()
                elif level==len(self.target_dzw):height=ts.copy()
                self.sample_height[level,start:stop]=height
                lo=np.zeros(len(height),dtype=np.int32);hi=np.full(len(height),ns,dtype=np.int32)
                for _ in range(int(np.ceil(np.log2(ns+1)))):
                    mid=(lo+hi+1)//2
                    below=sz[mid,c]<=height
                    lo=np.where(below,mid,lo);hi=np.where(below,hi,mid-1)
                self.sample_layer[level,start:stop]=np.minimum(lo,ns-1)
        from concurrent.futures import ThreadPoolExecutor
        from .host_parallel import threads, MIN_PARALLEL_ELEMENTS
        workers=min(threads(),len(self.target_dzw)+1)
        if self.sample_height.size<MIN_PARALLEL_ELEMENTS or workers==1:
            for level in range(len(self.target_dzw)+1):stencil(level)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(stencil,range(len(self.target_dzw)+1)))
        self.source_dz=np.diff(sz,axis=0)
        if not endpoint_alignment and not np.allclose(self.sample_height[-1],top_s,rtol=0,atol=1e-6):
            raise ValueError('absolute-height prototype requires an identical physical model top')
        self.source_centres=.5*(sz[:-1]+sz[1:])
        lowest_height=sz[0].copy()
        np.minimum.at(lowest_height,self.cols,self.sample_height[0])
        self.source_bottom_reach=self.source_centres[0]-lowest_height
        shift=tz[0]-horizontal.intensive(sz[0])
        self.receipt=dict(horizontal.receipt,vertical_remap=dict(
            method='terrain-endpoint-aligned physical-height cumulative integration, MC limited linear source profiles',
            source_layers=ns,target_layers=len(self.target_dzw),
            terrain_alignment_area_rms_m=float(np.sqrt(np.dot(shift*shift,self.target_area)/self.target_area.sum())),
            terrain_alignment_abs_max_m=float(abs(shift).max()),
            total_column_conservation=True,per_computational_level_conservation=False))
        if not endpoint_alignment:
            self.receipt['vertical_remap'].update(method='absolute physical-height limited layer integration with explicit global terrain-domain reconciliation',
                total_column_conservation=False,global_conservation=True,
                terrain_domain_reconciliation='positive products scaled uniformly; signed products receive a uniform computational-density correction; field-specific amounts retained')
            if domain_reconciliation=='local':
                self.receipt['vertical_remap'].update(
                    method='absolute physical-height limited integration with local lower-column domain reconciliation',
                    total_column_conservation=True,
                    terrain_domain_reconciliation='each target column retains its horizontal-overlap source integral; only lower support layers reconciled',
                    domain_support_depth_m=self.domain_support_depth_m)

    def intensive(self,values):
        """Surface/land fields have no atmospheric vertical coordinate."""
        import os
        if self.backend == 'cuda' and os.environ.get('WOOF_HEX_MOVING_HORIZONTAL_GPU','1') != '0':
            from .moving_horizontal_cuda import intensive
            return intensive(self.horizontal, values)
        return self.horizontal.intensive(values)

    def layer_intensive(self, values):
        """Map atmospheric rates in physical height and record host wall time."""
        from .stage_timing import stage
        gpu = self.backend == 'cuda'
        with stage(self.execution_timings, 'physical-height-rate-interpolation',
                   parallelism={'device': 'GPU plus CPU' if gpu else 'CPU', 'host_threads': 1}):
            return self._layer_intensive(values)

    def _layer_intensive(self, values):
        """Sample atmospheric rates at target physical layer centres.

        Rates are intensive, not computational densities. Applying the mass
        remap (including terrain-domain reconciliation) to a rate is wrong.
        Interpolate each donor in height, then take its area-weighted mean.
        Outside the donor's centre range retain its nearest endpoint value.
        Extra axes between level and cell are carried independently.
        """
        values = np.asarray(values, dtype=np.float64)
        ns = len(self.source_dzw)
        if values.ndim < 2 or values.shape[0] != ns or values.shape[-1] != len(self.source_area):
            raise ValueError('height map layer field has invalid source dimensions')
        if not np.isfinite(values).all():
            raise ValueError('height map layer field is not finite')
        if self.backend == 'cuda':
            from .moving_vertical_cuda import layer_integral
            return layer_integral(self, values)
        middle = values.shape[1:-1]
        source = values.reshape(ns, -1, len(self.source_area))
        result = np.empty((len(self.target_dzw), source.shape[1], len(self.target_area)))
        for level in range(len(result)):
            height = .5 * (self.sample_height[level] + self.sample_height[level + 1])
            lo = np.zeros(len(height), dtype=np.int32)
            hi = np.full(len(height), ns - 1, dtype=np.int32)
            for _ in range(int(np.ceil(np.log2(ns + 1)))):
                mid = (lo + hi + 1) // 2
                below = self.source_centres[mid, self.cols] <= height
                lo = np.where(below, mid, lo)
                hi = np.where(below, hi, mid - 1)
            k = np.minimum(lo, max(ns - 2, 0))
            upper = np.minimum(k + 1, ns - 1)
            span = self.source_centres[upper, self.cols] - self.source_centres[k, self.cols]
            fraction = np.clip((height - self.source_centres[k, self.cols]) /
                               np.where(span > 0, span, 1.), 0., 1.)
            for j in range(source.shape[1]):
                carried = source[k, j, self.cols] + fraction * (
                    source[upper, j, self.cols] - source[k, j, self.cols])
                result[level, j] = np.bincount(self.rows, weights=self.areas * carried,
                    minlength=len(self.target_area)) / self.target_area
        return result.reshape((len(self.target_dzw), *middle, len(self.target_area)))

    def interface_intensive(self, values):
        """Interpolate interface diagnostics in the same terrain-aligned height.

        Outside a donor's interface range (absolute heights below its ground,
        where the target ground is lower) retain its nearest endpoint value,
        as :meth:`layer_intensive` does. Never extrapolate: the lowest source
        interfaces carry near-ground values (the native vertical pressure
        residual, terrain-following w), and a linear extension 1-2 km under a
        plateau donor multiplies them by that depth over the first layer.
        THE BREAKAGE: on steep terrain (2025-11-08 12Z smoke case, 240/60 km,
        move at 1 h) a basin target column at 983 m with an 11 % donor on a
        2,741 m plateau carried -4 m/s2 through its lowest kilometre (its
        donors carry 3e-4 there), the balance imposed it, and its surface
        pressure fell 3.2 kPa in the first 15 min (jolt ratio 6.06).
        """
        values = np.asarray(values, dtype=np.float64)
        if values.shape != self.source_zgrid.shape or not np.isfinite(values).all():
            raise ValueError('height map interface field has invalid source dimensions or values')
        if self.backend == 'cuda':
            from .moving_vertical_cuda import interface_integral
            return interface_integral(self, values)
        result = np.empty(self.target_zgrid.shape, dtype=np.float64)
        for level in range(len(result)):
            k, c = self.sample_layer[level], self.cols
            fraction = np.clip((self.sample_height[level] - self.source_zgrid[k, c])
                               / self.source_dz[k, c], 0., 1.)
            carried = values[k, c] + fraction * (values[k + 1, c] - values[k, c])
            result[level] = np.bincount(self.rows, weights=self.areas * carried,
                                       minlength=len(self.target_area)) / self.target_area
        return result

    def mass_intensive(self,values,density=None):
        """Conservatively integrate mass products and record host wall time."""
        result, domain, positivity = self.mass_intensive_receipted(values, density=density)
        self.last_domain_reconciliation = domain
        self.last_positivity_roundoff = positivity
        return result

    def mass_intensive_receipted(self, values, density=None):
        """:meth:`mass_intensive` returning ``(result, domain reconciliation, positivity roundoff)``.

        It leaves ``last_domain_reconciliation`` / ``last_positivity_roundoff`` untouched, so several
        products can be integrated on concurrent threads (moving_balance, moving_state) and each keeps
        its own receipt.  The arithmetic is the same call; only where the two receipts go differs.
        """
        from .stage_timing import stage
        gpu = (self.backend == 'cuda' and len(self.source_dzw) == len(self.target_dzw))
        from .host_parallel import threads
        receipt = {}
        with stage(self.execution_timings, 'physical-height-mass-integration',
                   parallelism={'device': 'GPU plus CPU' if gpu else 'CPU',
                                'host_threads': threads(), 'host_schedule': 'bounded columns'}):
            result = self._mass_intensive(values, density=density, receipt=receipt)
        return result, receipt.get('domain'), receipt.get('positivity')

    def _mass_intensive(self,values,density=None,receipt=None):
        """Map computational mass density, integrating physical layer profiles.

        ``density`` (absolute-height transfer only): the source computational
        dry density of which ``values`` is a product (pass the density itself
        for the dry mass). Its lowest layer is then reconstructed with the
        product's intensive part held at that layer's mean ratio, and, once
        :meth:`extend_below_ground` has been given the donors' lowest
        thermodynamics, the air a donor lends below its own ground is its
        hydrostatic constant-theta extension, carrying its lowest theta, q
        and wind. THE BREAKAGE: every product, the density included, was
        extrapolated with its own lowest-layer gradient. Under a donor 650 m
        above the target ground (2025-11-08 12Z smoke case, 95.8E 26.8N,
        night-time surface inversion) that made the basin column's lowest
        layers 16 K colder than every donor and its wind 8 m/s against
        1-5 m/s; with theta held but the density still extrapolated, the
        column was out of hydrostatic balance and the balanced
        initialization cooled it by the same 15 K. Its surface pressure ran
        at 1.0 Pa/s for 30 min after the move against 0.46-0.59 in the
        fixed run.
        """
        values=np.asarray(values,dtype=np.float64)
        if values.shape!=(len(self.source_dzw),len(self.source_area)):
            raise ValueError('height map mass field has the wrong source dimensions')
        if not np.isfinite(values).all():raise ValueError('height map mass field is not finite')
        if density is not None and self.endpoint_alignment:
            density=None
        if density is not None:
            density=np.asarray(density,dtype=np.float64)
            if density.shape!=values.shape or not np.isfinite(density).all() or np.any(density<=0):
                raise ValueError('height map carrier density must be positive, finite and match the product')
        positive = bool(np.all(values>=0))
        device = (self.backend == 'cuda' and len(self.source_dzw) == len(self.target_dzw)
                  and self.domain_reconciliation != 'local')
        if device:
            from .moving_vertical_cuda import device_profiles_enabled
            device = device_profiles_enabled()
        if device:
            # Profiles, integral, positive roundoff and layer difference on the card, word for word the host
            # path below (moving_vertical_cuda.device_layer_result); the totals and the rescale stay here.
            from .moving_vertical_cuda import device_layer_result
            if receipt is None:
                receipt={}
            receipt['domain']=None
            result, receipt['positivity'] = device_layer_result(self, values, density, positive)
        else:
            means, slopes, cumulative = _source_profiles(self, values, density, positive)
            import os
            if self.backend == 'cuda' and len(self.source_dzw) == len(self.target_dzw):
                from .moving_vertical_cuda import cumulative_integral
                target_cumulative = cumulative_integral(self, means, slopes, cumulative, values, density)
            else:
                target_cumulative=np.zeros((len(self.target_dzw)+1,len(self.target_area)),dtype=np.float64)
                def interface(level):
                    k=self.sample_layer[level];c=self.cols;z=self.sample_height[level]
                    lower=self.source_zgrid[k,c];centre=self.source_centres[k,c]
                    carried=(cumulative[k,c]+means[k,c]*(z-lower)+
                             .5*slopes[k,c]*((z-centre)**2-(lower-centre)**2))
                    if density is not None and self.below_ground is not None:
                        under=z<self.source_zgrid[0,c]
                        if np.any(under):
                            ratio=values[0,c]/density[0,c]
                            carried=np.where(under,-ratio*self.below_ground[level],carried)
                    target_cumulative[level]=np.bincount(self.rows,weights=self.areas*carried,minlength=len(self.target_area))
                # Each target interface is independent and keeps its own arithmetic and bincount order, so the result is
                # byte-identical to one thread. Breakage it prevents: this loop ran on one core (31 s of an 83 s analysis
                # remap onto the derecho 1 km member, box 2026-10-17 01:34Z) while the moving forecast waited; it also
                # runs at every moving-region move. WOOF_HEX_MOVING_HEIGHT_THREADS caps the threads (default: affinity).
                from concurrent.futures import ThreadPoolExecutor
                levels=range(1 if self.endpoint_alignment else 0,len(self.target_dzw))
                threads=int(os.environ.get('WOOF_HEX_MOVING_HEIGHT_THREADS','0')) or len(os.sched_getaffinity(0))
                with ThreadPoolExecutor(max_workers=max(1,min(threads,len(levels)))) as pool:
                    list(pool.map(interface,levels))
                # Exact source endpoint totals own the top. No below-ground loss or
                # above-top extrapolation can change a column's carried mass.
                target_cumulative[-1]=np.bincount(self.rows,weights=self.areas*cumulative[-1,self.cols],
                                                 minlength=len(self.target_area))
            if receipt is None:
                receipt={}
            receipt['positivity']=None
            receipt['domain']=None
            if positive:
                target_cumulative,receipt['positivity']=positive_cumulative_roundoff(
                    target_cumulative, absolute_height=not self.endpoint_alignment)
            result=np.diff(target_cumulative,axis=0)/(self.target_dzw[:,None]*self.target_area)
        if not self.endpoint_alignment:
            before=self.totals(values,source=True)[0]
            raw=self.totals(result,source=False)[0]
            if self.domain_reconciliation=='local':
                return self._reconcile_local_columns(values,result,target_cumulative[-1],before,raw,receipt)
            if positive:
                if raw<=0 and before!=0:raise ValueError('positive absolute-height integral is not positive')
                factor=1. if before==raw==0 else before/raw
                result*=factor
                correction=dict(policy='uniform positive-product rescaling',factor=float(factor))
            else:
                offset=(before-raw)/(self.target_area.sum()*self.target_dzw.sum())
                result+=offset
                correction=dict(policy='uniform signed computational-density correction',offset=float(offset))
            receipt['domain']=dict(correction,source_total=float(before),
                absolute_height_total_before_reconciliation=float(raw),redistributed_total=float(before-raw),
                relative_to_source_abs_integral=float((before-raw)/self.totals(abs(values),source=True)[0])
                    if np.any(values) else 0.)
        return result

    def extend_below_ground(self, theta, exner, loading=None):
        """Give the donors' lowest-layer thermodynamics for the sub-ground extension.

        ``theta`` (the modified potential temperature rho_theta/rho),
        ``exner`` and ``loading`` (1 + total water mixing ratio) are the
        source mass-level fields. Below its ground a donor is extended with
        its lowest theta held: exner rises along g*loading/(cp*theta) down
        from the lowest layer centre, and the dry mass between a height and
        the donor's ground is the hydrostatic pressure difference over
        g*loading. Only an absolute-height transfer has sub-ground heights.
        """
        if self.endpoint_alignment:
            return
        theta=np.asarray(theta,dtype=np.float64)[0]
        exner=np.asarray(exner,dtype=np.float64)[0]
        loading=np.ones_like(theta) if loading is None else np.asarray(loading,dtype=np.float64)[0]
        finite=np.isfinite(theta).all() and np.isfinite(exner).all() and np.isfinite(loading).all()
        if not finite or np.any(theta<=0) or np.any(exner<=0) or np.any(loading<1):
            raise ValueError('sub-ground extension needs positive finite theta, exner and loading')
        if (self.backend=='cuda' and os.environ.get('WOOF_HEX_MOVING_EXTENSION_GPU','1')!='0'
                and os.environ.get('WOOF_HEX_MOVING_POW_GPU','1')!='0'):
            from .moving_math_cuda import ground_extension
            self.below_ground,self._cuda_below_ground=ground_extension(self,theta,exner,loading)
            if np.any(self.below_ground<0):
                raise ValueError('sub-ground extension produced negative mass')
            return
        cp,rgas,gravity=1004.5,287.0,9.80616
        c=self.cols
        rate=gravity*loading[c]/(cp*theta[c])
        centre=self.source_centres[0,c]
        ground=self.source_zgrid[0,c]

        def pressure(z):
            return 1e5*(exner[c]+rate*(centre-z))**(cp/rgas)

        self.below_ground=np.zeros(self.sample_height.shape)
        ground_pressure=pressure(ground)
        def layer(level):
            # Only heights under the donor's ground are extended; heights above it are held at the ground
            # (zero mass), so the power law is never evaluated far above the donor where its base is negative.
            z=np.minimum(self.sample_height[level],ground)
            self.below_ground[level]=(pressure(z)-ground_pressure)/(gravity*loading[c])
        from .host_parallel import run_tasks
        run_tasks([lambda level=level: layer(level) for level in range(len(self.sample_height))], workers=4)
        if np.any(self.below_ground<0):
            raise ValueError('sub-ground extension produced negative mass')

    def _lowest_density_slope(self, density):
        """The lowest-layer physical density slope :meth:`mass_intensive` uses for ``density``."""
        means=density[:2]*self.source_dzw[:2,None]/self.source_dz[:2]
        if len(density)<2:
            return np.zeros(density.shape[1])
        slope=(means[1]-means[0])/(self.source_centres[1]-self.source_centres[0])
        slope=np.clip(slope,-2*means[0]/self.source_dz[0],2*means[0]/self.source_dz[0])
        return np.minimum(slope,means[0]/self.source_bottom_reach)

    def _reconcile_local_columns(self, values, result, target_integrated, before, raw, receipt=None):
        """Return displaced material in the lower part of its own target column.

        This is an opt-in domain correction, not a balance fix. The target
        column integral is the horizontal overlap of complete source columns.
        Unchanged one-donor columns are copied exactly. Positive products use
        a local scale (or deposit into empty support); signed products use a
        local additive correction. Support expands only when required to
        avoid withdrawing more material than the positive support contains.
        """
        nc=len(self.target_area)
        nl=len(self.target_dzw)
        expected=target_integrated/self.target_area
        counts=np.bincount(self.rows,minlength=nc)
        donors=np.full(nc,-1,dtype=np.int64)
        donors[self.rows]=self.cols
        overlap_area=np.zeros(nc)
        overlap_area[self.rows]=self.areas
        unchanged=(counts==1)&(donors>=0)
        safe=np.flatnonzero(unchanged)
        unchanged[safe]&=(overlap_area[safe]==self.target_area[safe])&(
            overlap_area[safe]==self.source_area[donors[safe]])
        safe=np.flatnonzero(unchanged)
        unchanged[safe]&=np.all(self.target_zgrid[:,safe]==self.source_zgrid[:,donors[safe]],axis=0)
        if not np.array_equal(self.source_dzw,self.target_dzw):
            unchanged[:]=False
        result[:,unchanged]=values[:,donors[unchanged]]
        column_raw=np.einsum('lc,l->c',result,self.target_dzw)
        delta=expected-column_raw
        delta[unchanged]=0.
        # Include one full source bottom layer and at least the declared
        # physical depth above either ground. No other horizontal column
        # supplies or receives the reconciled material.
        highest_source_bottom=np.full(nc,-np.inf)
        np.maximum.at(highest_source_bottom,self.rows,self.source_zgrid[1,self.cols])
        support_top=np.maximum(highest_source_bottom,
            self.target_zgrid[0]+self.domain_support_depth_m)
        support_counts=np.maximum(1,np.sum(self.target_zgrid[:-1]<support_top[None,:],axis=0))
        cumulative=np.cumsum(result*self.target_dzw[:,None],axis=0)
        columns=np.arange(nc)
        positive=np.all(values>=0)
        support_mass=cumulative[support_counts-1,columns]
        if positive:
            needs=support_mass+delta<0
            while np.any(needs):
                support_counts[needs]+=1
                if np.any(support_counts>nl):
                    raise ValueError('local positive-product domain reconciliation has insufficient column material')
                support_mass=cumulative[support_counts-1,columns]
                needs=support_mass+delta<0
        support=np.arange(nl)[:,None]<support_counts[None,:]
        thickness=np.sum(support*self.target_dzw[:,None],axis=0)
        if positive:
            nonzero=support_mass>0
            factor=np.ones(nc)
            factor[nonzero]=(support_mass[nonzero]+delta[nonzero])/support_mass[nonzero]
            result*=np.where(support,factor[None,:],1.)
            empty=~nonzero
            if np.any(delta[empty]<0):
                raise ValueError('local positive-product reconciliation cannot withdraw from empty support')
            result+=np.where(support&empty[None,:],(delta/thickness)[None,:],0.)
        else:
            result+=np.where(support,(delta/thickness)[None,:],0.)
        # Copy again so harmless arithmetic cannot alter the exact carry.
        result[:,unchanged]=values[:,donors[unchanged]]
        local_error=np.einsum('lc,l->c',result,self.target_dzw)-expected
        expected_abs=(self.overlap@np.sum(abs(values)*self.source_dzw[:,None],axis=0))/self.target_area
        arithmetic=128*np.finfo(float).eps*expected_abs
        if np.any(abs(local_error)>arithmetic):
            raise ValueError('local target-column integral exceeds its bounded arithmetic allowance')
        if positive and np.any(result<0):
            raise ValueError('local positive-product reconciliation produced negative material')
        if receipt is None:
            receipt={}
        receipt['domain']=dict(
            policy='local lower-column domain reconciliation',positive_product=bool(positive),
            source_total=float(before),absolute_height_total_before_reconciliation=float(raw),
            redistributed_total=float(before-raw),
            redistributed_abs_column_total=float(np.dot(abs(delta),self.target_area)),
            relative_to_source_abs_integral=float((before-raw)/self.totals(abs(values),source=True)[0])
                if np.any(values) else 0.,
            correction_columns=int(np.count_nonzero(delta)),
            unchanged_columns_carried_exactly=int(unchanged.sum()),
            support_depth_m=self.domain_support_depth_m,
            support_layers_max=int(support_counts[delta!=0].max()) if np.any(delta) else 0,
            target_column_integral_error_abs_max=float(abs(local_error).max()))
        return result

    def totals(self,values,*,source):
        areas=self.source_area if source else self.target_area
        dz=self.source_dzw if source else self.target_dzw
        return np.atleast_1d(np.einsum('lc,c,l->',np.asarray(values,dtype=np.float64),areas,dz))

    def conserved(self,values,density=None):
        exact=self.mass_intensive(values,density=density);actual=exact.astype(np.float32)
        before=self.totals(values,source=True);after=self.totals(actual,source=False)
        envelope=self.totals(abs(actual.astype(float)-exact),source=False)
        fp64=self.totals(exact,source=False)-before
        arithmetic=128*np.finfo(float).eps*self.totals(abs(exact),source=False)
        if np.any(abs(fp64)>arithmetic):
            raise ValueError('height remap FP64 integral exceeds its bounded arithmetic allowance')
        if np.any(abs(after-before)>envelope+arithmetic):
            raise ValueError('height remap exceeds global measured cast accounting')
        return actual,dict(before=before.tolist(),after=after.tolist(),absolute_error=(after-before).tolist(),
            rounding_envelope=envelope.tolist(),fp64_error=fp64.tolist(),
            terrain_domain_reconciliation=self.last_domain_reconciliation,
            positivity_roundoff=self.last_positivity_roundoff,
            accounting='whole-column global totals after vertical redistribution; includes nominal thickness')
