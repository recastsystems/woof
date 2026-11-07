"""Member-batched scalar transport using the installed scalar arithmetic.

All scalar launches span the resident member dimension. Species retain their
established operation order; there is no loop over advancing members. Tables
and static coefficients use their admitted ownership. Prepared lateral tables
can supply held and recomputed scalar tendencies. Implicit vertical transport
requires its own column binding.
"""
from __future__ import annotations

from functools import lru_cache
from math import prod

import numpy as np

from woof.ensemble.batch_kernel import (
    KernelSpec, PointerSpec, prepare_batch_kernel_launch, prepare_batch_source_launch,
)
from woof.ensemble.batch_state import BatchedDomainState, BatchStateUnsupported
from woof.ensemble.batch_storage import BatchArraySpec

_F = np.float32
_I = np.int32
_U = np.uint64
_TPB = 128


def workspace_specs(cfg):
    """Price materialized original ufunc operands before resident allocation."""
    if not cfg.moist:
        return ()
    plane = (cfg.ny, cfg.nx)
    specs = [BatchArraySpec('batch_moist_mu0', plane, 'member'),
             BatchArraySpec('batch_moist_mu', plane, 'member')]
    if cfg.moist_adv_opt == 1 and not (cfg.open_x or cfg.open_y):
        shape = (cfg.nz, cfg.ny, cfg.nx)
        specs += [BatchArraySpec('batch_moist_chm0', shape, 'member'),
                  BatchArraySpec('batch_moist_fold_work', shape, 'member')]
    return tuple(specs)


def required_scratch_slots(cfg):
    from woof.core.preflight import scratch_slot_registry
    names = {'rk_ru_m', 'rk_rv_m', 'rk_ww_m', 'moist_rq_t'} if cfg.moist else set()
    if cfg.moist and cfg.moist_adv_opt == 1 and not (cfg.open_x or cfg.open_y):
        names.update(('pd_fxl', 'pd_fxc', 'pd_fyl', 'pd_fyc', 'pd_fzl', 'pd_fzc', 'moist_pd_q0'))
    registry = scratch_slot_registry(cfg)
    if cfg.moist and cfg.specified:
        names.update(name for name in registry if name.startswith('lbc_') and name.endswith('_held'))
    if names - registry.keys():
        raise BatchStateUnsupported('scalar transport scratch inventory changed; audit its allocation lifetimes')
    return {name: np.dtype('float32') for name in sorted(names)}


class _Bindings:
    def __init__(self, state):
        from woof.ensemble.batch_acoustic import _Bindings as InventoryBindings
        self.inventory = InventoryBindings(state, state.cfg)
        self.state = state

    def array(self, name):
        if name not in self.state.storage.arrays:
            raise BatchStateUnsupported('scalar transport needs planned allocation ' + repr(name))
        return self.state.storage.arrays[name]

    def pointer(self, name, array):
        owner, spec = self.inventory.owner(array)
        return PointerSpec(name, spec.ownership, spec.dtype), self.state.storage.pointer_stride_bytes(owner)

    def raw(self, module, entry, names, arrays, suffix, grid, *, block=(_TPB, 1, 1)):
        specs, strides = [], {}
        for name, array in zip(names, arrays, strict=True):
            spec, stride = self.pointer(name, array)
            specs.append(spec)
            strides[name] = stride
        return prepare_batch_kernel_launch(KernelSpec(module, entry, tuple(specs)),
            self.state.members, grid, block, tuple(arrays) + tuple(suffix), pointer_strides=strides)


def prepare_sumflux_launch(state, entry, targets, sources=(), nsub=0):
    """Bind the unchanged three-stagger acoustic accumulation for all members."""
    if entry not in ('zero_sumflux', 'accumulate_sumflux', 'finish_sumflux'):
        raise ValueError('sumflux entry must select zero, accumulation or finish')
    if len(targets) != 3 or len(sources) != (0 if entry == 'zero_sumflux' else 3):
        raise ValueError('sumflux requires exactly the original three staggered fields')
    if entry == 'finish_sumflux' and (type(nsub) is not int or nsub < 1):
        raise ValueError('sumflux finish requires a positive acoustic substep count')
    from woof.ensemble.batch_acoustic import _scalar_array
    if state.members == 1:
        from woof.core.dycore import _prepare_sumflux_launch
        return _prepare_sumflux_launch(entry, tuple(_scalar_array(state, value) for value in targets),
            tuple(_scalar_array(state, value) for value in sources), nsub)
    binding = _Bindings(state)
    names = ('ru_m', 'rv_m', 'ww_m')
    if entry == 'accumulate_sumflux':
        names += ('u_pp', 'v_pp', 'ww_pp')
    elif entry == 'finish_sumflux':
        names += ('ru', 'rv', 'ww')
    sizes = tuple(_U(prod(value.shape[1:])) for value in targets)
    nmax = max(int(size) for size in sizes)
    suffix = ((_F(nsub),) if entry == 'finish_sumflux' else ()) + sizes + (_U(nmax),)
    return binding.raw('acoustic', entry, names, tuple(targets) + tuple(sources), suffix,
                       ((nmax + 255) // 256,), block=(256,))


def prepare_flow_dependent_boundaries(state, fields, u_flux, v_flux, spec_zone, *, inflow_value=0.0):
    """Bind the original flow-dependent perimeter kernel over all members."""
    fields = tuple(fields)
    if not 1 <= len(fields) <= 9:
        raise ValueError('flow-dependent boundary batch requires the original 1..9 fields')
    nz, ny, nx = state.cfg.nz, state.cfg.ny, state.cfg.nx
    expected = (state.members, nz, ny, nx)
    if any(tuple(field.shape) != expected for field in fields):
        raise ValueError('flow-dependent fields must expose every admitted mass-grid member')
    if tuple(u_flux.shape) != (state.members, nz, ny, nx + 1) or tuple(v_flux.shape) != (state.members, nz, ny + 1, nx):
        raise ValueError('flow-dependent fluxes require the original staggered member shapes')
    if spec_zone < 1 or min(ny, nx) <= 2 * spec_zone:
        raise ValueError('spec_zone leaves no flow-dependent boundary interior')
    from woof.ingest import lateral_bc as lateral
    from woof.ensemble.batch_acoustic import _scalar_array
    if state.members == 1:
        def original():
            lateral.apply_flow_dependent_boundaries(tuple(_scalar_array(state, value) for value in fields),
                _scalar_array(state, u_flux), _scalar_array(state, v_flux), spec_zone, inflow_value=inflow_value)
        original.numerical_entries = ('flow_dependent_batch',)
        return original
    binding = _Bindings(state)
    padded = fields + (fields[-1],) * (9 - len(fields))
    frame_count = lateral._perimeter_count(ny, nx, spec_zone)
    names = tuple('f' + str(index) for index in range(9)) + ('u_flux', 'v_flux')
    suffix = tuple(_I(value) for value in (len(fields), spec_zone, nz, ny, nx, frame_count)) + (_F(inflow_value),)
    launch = binding.raw('lbc_flow', 'flow_dependent_batch', names, padded + (u_flux, v_flux), suffix,
                         ((nz * frame_count + 255) // 256,), block=(256,))
    launch.numerical_entries = ('flow_dependent_batch',)
    return launch


@lru_cache(maxsize=None)
def scalar_update_source(has_msf, has_physics, has_fixed, clamp):
    """Embed the installed ElementwiseKernel operation without math edits."""
    from woof.core.moist import _update_scalar_kernel
    import ast
    import inspect
    import textwrap
    syntax = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(_update_scalar_kernel))))
    calls = [node for node in ast.walk(syntax) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'ElementwiseKernel']
    if len(calls) != 1:
        raise BatchStateUnsupported('coupled scalar factory no longer has one compiler-option authority')
    declarations = [keyword.value for keyword in calls[0].keywords if keyword.arg == 'options']
    if len(declarations) != 1:
        raise BatchStateUnsupported('coupled scalar factory compiler options require an explicit audit')
    options = tuple(ast.literal_eval(declarations[0]))
    original = _update_scalar_kernel(has_msf, has_physics, has_fixed, clamp)
    expected = [('q0', 'T', False, True), ('tend', 'T', False, True)]
    expected += [(name, 'T', True, True) for name in ('c1h', 'c2h', 'mu0', 'mu')]
    if has_msf:
        expected.append(('msft', 'T', True, True))
    for flag, name in ((has_physics, 'physics'), (has_fixed, 'fixed')):
        if flag:
            expected.append((name, 'T', False, True))
    expected += [('dt_eff', 'T', False, True), ('ncol', 'int', False, True), ('q', 'T', False, False)]
    actual = [(p.name, p.ctype, bool(p.raw), bool(p.is_const))
              for p in original.in_params + original.out_params]
    if actual != expected or '_ind' in original.operation:
        raise BatchStateUnsupported('coupled scalar update ABI/indexer changed; audit its rounding and ownership')
    parameters = []
    pointers = []
    locals_text = []
    for name, ctype, raw, readonly in expected:
        if name in ('dt_eff', 'ncol'):
            parameters.append(('float' if name == 'dt_eff' else 'int') + ' ' + name)
            continue
        parameter = name if raw else name + '_values'
        parameters.append(('const float* ' if readonly else 'float* ') + parameter)
        pointers.append(parameter)
        if not raw:
            locals_text.append(('const T ' if readonly else 'T& ') + name +
                               (' = ' if readonly else ' = ') + parameter + '[i];')
    parameters.append('int scalar_size')
    source = ('typedef float T;\n' + original.preamble + '\nextern "C" __global__ void scalar_update('
              + ', '.join(parameters) + ') {\n'
              'const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n'
              'if (i >= scalar_size) return;\n' + '\n'.join(locals_text) + '\n'
              + original.operation + '\n}\n')
    return source, tuple(pointers), options


def _prepare_update(binding, q, q0, tend, mu0, mu, dt_eff, *, physics=None, fixed=None,
                    clamp=False, scale_msf=True):
    state = binding.state
    if prod(q.shape[1:]) > np.iinfo(np.int32).max:
        raise BatchStateUnsupported('the installed scalar update uses signed 32-bit per-field indices')
    source, pointer_names, options = scalar_update_source(bool(state.has_msf and scale_msf), physics is not None,
                                                         fixed is not None, clamp)
    values = {'q0_values': q0, 'tend_values': tend, 'c1h': state.c1h, 'c2h': state.c2h,
              'mu0': mu0, 'mu': mu, 'msft': state.msft, 'physics_values': physics,
              'fixed_values': fixed, 'q_values': q, 'dt_eff': _F(dt_eff),
              'ncol': _I(state.cfg.ny * state.cfg.nx), 'scalar_size': _I(prod(q.shape[1:]))}
    pointers, strides = [], {}
    for name in pointer_names:
        spec, stride = binding.pointer(name, values[name])
        pointers.append(spec)
        strides[name] = stride
    spec = KernelSpec('ensemble_moist', 'scalar_update', tuple(pointers), options=options)
    from woof.ensemble.batch_kernel import _entry_parts, _runtime_audit_options
    names = _entry_parts(source, spec, _runtime_audit_options(spec))[-2]
    return prepare_batch_source_launch(source, spec, state.members,
        ((int(values['scalar_size']) + _TPB - 1) // _TPB,), (_TPB,),
        tuple(values[name] for name in names), pointer_strides=strides)


class _ScalarPhysics:
    def __init__(self, state, original):
        self.state, self.original = state, original

    def scalar_for(self, name):
        from woof.ensemble.batch_acoustic import _scalar_array
        return _scalar_array(self.state, self.original.scalar_for(name))


def _scalar_table_view(state, cfg, tables, clock):
    """Give stock N=1 transport metadata over the already admitted table views."""
    from types import MappingProxyType, SimpleNamespace
    from woof.ensemble.batch_acoustic import _scalar_state
    from woof.ingest import lateral_bc as lateral
    from woof.ensemble.batch_boundaries import _state_check
    _state_check(state, cfg, tables)
    selection, evaluated = tables.evaluate(tables.select(clock))
    view = _scalar_state(state)
    view.lateral_boundaries = tables.boundaries[0]
    interval = lateral._DeviceBoundaryInterval(MappingProxyType({name:
        tables.scalar_field(selection, name, evaluated) for name in tables.fields}))
    dt_key = np.float32(tables.dt) if cfg.nested else float(lateral.lateral_boundary_clock_dt(cfg))
    key = (tables.width, cfg.spec_zone, cfg.relax_zone, dt_key, float(tables.spec_exp),
           bool(cfg.nested), float(lateral.relax_timescale_seconds(cfg)))
    selected_clock = SimpleNamespace(elapsed_seconds=clock.elapsed_seconds,
        dt_fp32=clock.dt_fp32, dtbc_launch_fp32=selection.offset)
    view._lateral_boundary_device = lateral._DeviceLateralBoundaries(
        (interval,), MappingProxyType({id(tables.boundaries[0].intervals[selection.interval]): 0}),
        {key: (tables.storage.arrays['weights:fcx'], tables.storage.arrays['weights:gcx'])},
        set(), tables.plan.required_bytes(1), rolling=bool(cfg.nested), clock=selected_clock)
    return view


def prepare_scalars_stage(state, cfg, ru, rv, ww, dt_eff, final, *, apply_relax=True,
                          physics_tendencies=None, fixed_tendencies=None,
                          export_advective_forcing=False, implicit=None, tables=None, clock=None):
    """Prepare original unlimited/PD scalar transport over complete member slabs."""
    if not isinstance(state, BatchedDomainState) or not cfg.moist:
        raise TypeError('batched scalar transport needs an admitted moist member state')
    from woof.core import moist as original
    from woof.ensemble.batch_acoustic import _scalar_array, _scalar_state
    if tables is not None and clock is None:
        raise ValueError('member scalar tables require the common solve-entry boundary clock')
    if state.members == 1:
        fixed = (None if fixed_tendencies is None else
                 {name: _scalar_array(state, value) for name, value in fixed_tendencies.items()})
        physics = None if physics_tendencies is None else _ScalarPhysics(state, physics_tendencies)
        scalar_view = _scalar_state(state) if tables is None else _scalar_table_view(state, cfg, tables, clock)
        def single():
            original.advance_scalars_stage(scalar_view, cfg,
                *(_scalar_array(state, value) for value in (ru, rv, ww)), dt_eff, final,
                apply_relax=apply_relax, physics_tendencies=physics, fixed_tendencies=fixed,
                export_advective_forcing=export_advective_forcing, implicit=implicit)
        return single
    if cfg.tracer_opt == 2:
        raise BatchStateUnsupported(
            'member tracer transport needs independent tracer_adv_opt and native '
            'zero-inflow boundary bindings; use the ordinary member executor')
    if implicit is not None or cfg.zadvect_implicit:
        raise BatchStateUnsupported('member scalar transport needs its own IEVA column solve before applying implicit fluxes')
    if state.lateral_boundaries is not None and tables is None:
        raise BatchStateUnsupported('scalar lateral tables need member-held spec/relax tendencies before transport')
    if tables is not None and clock is None:
        raise ValueError('member scalar tables require the common solve-entry boundary clock')
    if export_advective_forcing and getattr(state, 'rqvften', None) is not None:
        raise BatchStateUnsupported('advective qv export needs its planned member uncoupling carrier')
    import cupy as cp
    from woof.ensemble.batch_operators import prepare_flux_div
    from woof.ensemble.batch_bookkeeping import prepare_bookkeeping
    binding = _Bindings(state)
    if any(state.storage.specs[name].ownership != 'shared' for name in ('c1h', 'c2h', 'rdnw', 'fnm', 'fnp', 'msft')):
        raise BatchStateUnsupported('this scalar transport graph requires byte-verified shared vertical and map coefficients')
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    mu0, mu = (binding.array(name) for name in ('batch_moist_mu0', 'batch_moist_mu'))
    tend = state.scratch((nz, ny, nx), 'moist_rq_t')
    zero = prepare_bookkeeping(((tend, tend),), members=state.members, zero=True)
    forced = cfg.specified or cfg.nested
    bx, by = cfg.open_x or forced, cfg.open_y or forced
    pd = final and cfg.moist_adv_opt == 1 and not (cfg.open_x or cfg.open_y)
    from woof.core.advection import vertical_orders
    vorder_scalar = vertical_orders(cfg)[0]
    # moist.py passes scalar_horizontal_order(cfg) to both species routes
    # (launch_pd_fluxes and launch_flux_div_scalar) since 2582c065c99.
    from woof.core.advection import scalar_horizontal_order
    horder_scalar = scalar_horizontal_order(cfg)
    if pd and horder_scalar not in (2, 3, 4, 5, 6):
        raise ValueError("WRF advect_scalar_pd admits horder 2, 3, 4, 5 or 6")
    pd_defines = () if horder_scalar == 5 else (('GPUWM_HORDER', int(horder_scalar)),)
    supplied = frozenset()
    selection = evaluated = None
    capture_rows = []
    held_fields = {}
    if tables is not None:
        from woof.ensemble.batch_boundaries import _state_check, _relax
        from woof.boundary_fields import HELD_BOUNDARY_FIELDS
        _state_check(state, cfg, tables)
        selection, evaluated = tables.evaluate(tables.select(clock))
        supplied = frozenset(tables.fields)
        if cfg.specified:
            for name in HELD_BOUNDARY_FIELDS:
                if name not in supplied:
                    continue
                held = state.scratch((nz, ny, nx), 'lbc_' + name + '_held')
                held_fields[name] = held
                if apply_relax:
                    capture_rows.append((held, _relax(state, cfg, tables, selection, evaluated,
                        name, held, held, apply_relax=True)))

    def lateral(name, output, source):
        if tables is None or name not in supplied:
            return lambda: None
        return _relax(state, cfg, tables, selection, evaluated, name, output, output,
                      apply_relax=True, source=source, source_mu=state.mup0)
    chm0 = binding.array('batch_moist_chm0') if pd else None
    fold_work = binding.array('batch_moist_fold_work') if pd else None
    q0_fold = state.scratch((nz, ny, nx), 'moist_pd_q0') if pd else None
    rows = []
    buffers = tuple(state.scratch(shape, name) for shape, name in (
        ((nz, ny, nx + 1), 'pd_fxl'), ((nz, ny, nx + 1), 'pd_fxc'),
        ((nz, ny + 1, nx), 'pd_fyl'), ((nz, ny + 1, nx), 'pd_fyc'),
        ((nz + 1, ny, nx), 'pd_fzl'), ((nz + 1, ny, nx), 'pd_fzc'))) if pd else ()
    for name in original.moist_species(state):
        q, q0 = getattr(state, name), getattr(state, name + '0')
        physics = None if physics_tendencies is None else physics_tendencies.scalar_for(name)
        fixed = None if fixed_tendencies is None else fixed_tendencies.get(name)
        held = held_fields.get(name)
        recompute = tables is not None and (cfg.nested or (name in supplied and name not in held_fields))
        lbc_source = tend if recompute and pd else held
        sources = tuple(value for value in (fixed, physics, lbc_source) if value is not None)
        for value in sources:
            binding.inventory.owner(value)
        q0_eff = q0_fold if pd and sources else q0
        lateral_launch = lateral(name, tend, q0) if recompute else None
        lbc_after_msf = recompute or held is not None
        if pd:
            arrays = (q, q0_eff, ru, rv, ww, mu, state.c1h, state.c2h, state.rdnw,
                      state.fnm, state.fnp, state.msft)
            # Output pointers follow the three scalar spacing arguments.
            pointer_names = ('q', 'q0', 'ru', 'rv', 'rw', 'mut', 'c1h', 'c2h',
                             'rdnw', 'fnm', 'fnp', 'msft', 'fxl', 'fxc', 'fyl', 'fyc', 'fzl', 'fzc')
            specs, strides = [], {}
            for parameter, array in zip(pointer_names, arrays + buffers, strict=True):
                spec, stride = binding.pointer(parameter, array)
                specs.append(spec); strides[parameter] = stride
            flux_spec = KernelSpec('pd_advection', 'pd_fluxes', tuple(specs))
            flux_args = arrays + (_F(cfg.dx), _F(cfg.dy), _F(dt_eff)) + buffers + tuple(
                _I(value) for value in (nz, ny, nx, state.has_msf, bx, by, vorder_scalar))
            flux = prepare_batch_kernel_launch(flux_spec, state.members,
                ((nx + 1 + _TPB - 1) // _TPB, ny + 1, nz + 1), (_TPB, 1, 1),
                flux_args, pointer_strides=strides, defines=pd_defines)
            if vorder_scalar == 5:
                primary = flux
                vertical = prepare_batch_kernel_launch(
                    KernelSpec('pd_vertical_sl', 'pd_vertical_sl', tuple(specs)), state.members,
                    ((nx + 1 + _TPB - 1) // _TPB, ny + 1, nz + 1), (_TPB, 1, 1),
                    flux_args, pointer_strides=strides)
                def flux(primary=primary, vertical=vertical):
                    primary()
                    vertical()
            r_arrays = (q0_eff, mu0) + buffers + (state.c1h, state.c2h, state.rdnw, state.msft)
            r_names = ('q0', 'mu_old', 'fxl', 'fxc', 'fyl', 'fyc', 'fzl', 'fzc',
                       'c1h', 'c2h', 'rdnw', 'msft')
            r_specs, r_strides = [], {}
            for parameter, array in zip(r_names + ('tend_out',), r_arrays + (tend,), strict=True):
                spec, stride = binding.pointer(parameter, array)
                r_specs.append(spec); r_strides[parameter] = stride
            r_args = r_arrays + (_F(1.0 / cfg.dx), _F(1.0 / cfg.dy), _F(dt_eff), tend) + tuple(
                _I(value) for value in (nz, ny, nx, state.has_msf, bx, by))
            from woof.wrf_exact import ADVECTION_ENABLED
            if ADVECTION_ENABLED:
                raise BatchStateUnsupported('strict PD mass-history parameters need a separate complete scalar-stage proof')
            renorm = prepare_batch_kernel_launch(KernelSpec('pd_advection', 'pd_renorm_apply', tuple(r_specs)),
                state.members, ((nx + _TPB - 1) // _TPB, ny, nz), (_TPB, 1, 1), r_args,
                pointer_strides=r_strides)
            update = _prepare_update(binding, q, q0_eff, tend, mu0, mu, dt_eff, clamp=True)
        else:
            flux = prepare_flux_div(q, ru, rv, ww, tend, state.rdnw, state.fnm, state.fnp,
                state.msft, dx=cfg.dx, dy=cfg.dy, open_x=bx, open_y=by,
                has_msf=state.has_msf, spec=forced, vorder=vorder_scalar,
                horder=horder_scalar)
            renorm = None
            update = _prepare_update(binding, q, q0, tend, mu0, mu, dt_eff,
                                     physics=physics, fixed=fixed, clamp=final,
                                     scale_msf=not lbc_after_msf)
        rows.append((q0, q0_eff, sources, flux, renorm, update, lateral_launch, held, lbc_after_msf))

    flow = []
    flow_names = ()
    if cfg.specified:
        names = original.moist_species(state)
        qnn_name = 'nn' if getattr(state, 'nn', None) is not None else 'qnn'
        if qnn_name in names:
            from woof.core.microphysics_transition import NSSL2_BACKGROUND_CCN_PER_KG
            inflow = cfg.wdm6_ccn_conc if qnn_name == 'nn' else NSSL2_BACKGROUND_CCN_PER_KG
            flow.append(prepare_flow_dependent_boundaries(state, (getattr(state, qnn_name),),
                                                         ru, rv, cfg.spec_zone, inflow_value=inflow))
        specified_scalars = set(held_fields) | (set(names) & set(supplied))
        flow_names = tuple(name for name in names if name not in ('qv', qnn_name) and name not in specified_scalars)
        for start in range(0, len(flow_names), 9):
            fields = tuple(getattr(state, name) for name in flow_names[start:start + 9])
            flow.append(prepare_flow_dependent_boundaries(state, fields, ru, rv, cfg.spec_zone))

    def launch():
        cp.add(state.mub2d, state.mup0, out=mu0)
        cp.add(state.mub2d, state.mup, out=mu)
        if pd:
            cp.multiply(state.c1h[None, :, None, None], mu0[:, None], out=chm0)
            cp.add(chm0, state.c2h[None, :, None, None], out=chm0)
        for held, capture in capture_rows:
            held.fill(0)
            capture()
        for q0, q0_eff, sources, flux, renorm, update, lateral_launch, held, lbc_after_msf in rows:
            if pd and lateral_launch is not None:
                zero()
                lateral_launch()
            if pd and sources:
                cp.copyto(q0_eff, q0)
                for source in sources:
                    cp.multiply(_F(dt_eff), source, out=fold_work)
                    cp.divide(fold_work, chm0, out=fold_work)
                    cp.add(q0_eff, fold_work, out=q0_eff)
            zero()
            flux()
            if renorm is not None:
                renorm()
                if forced and cfg.spec_zone > 0:
                    size = cfg.spec_zone
                    tend[:, :, :size, :] = 0
                    tend[:, :, -size:, :] = 0
                    tend[:, :, size:-size, :size] = 0
                    tend[:, :, size:-size, -size:] = 0
            else:
                if state.has_msf and lbc_after_msf:
                    cp.multiply(tend, state.msft[None, None], out=tend)
                if lateral_launch is not None:
                    lateral_launch()
                if held is not None:
                    cp.add(tend, held, out=tend)
                    size = cfg.spec_zone
                    if size > 0:
                        tend[:, :, :size, :] = held[:, :, :size, :]
                        tend[:, :, ny - size:, :] = held[:, :, ny - size:, :]
                        tend[:, :, size:ny - size, :size] = held[:, :, size:ny - size, :size]
                        tend[:, :, size:ny - size, nx - size:] = held[:, :, size:ny - size, nx - size:]
            update()
        for boundary in flow:
            boundary()
    launch.transport_receipt = {'members': state.members, 'species': original.moist_species(state),
                                'positive_definite': bool(pd), 'lateral_tables_attached': tables is not None,
                                'flow_dependent_fields': flow_names,
                                'additional_workspace': tuple(spec.name for spec in workspace_specs(cfg))}
    return launch
