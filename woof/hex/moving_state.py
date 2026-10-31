"""Atmospheric remap with explicit FP32 integral accounting.

The supplied map selects horizontal or terrain-aligned physical-height mass
transfer. Both paths reconstruct thermodynamics on the target Jacobian.
"""
from __future__ import annotations
import numpy as np


def reconstruct_cartesian(normal, arrays):
    """Reconstruct with MPAS cell coefficients in the earth Cartesian basis."""
    edges = np.asarray(arrays['edgesOnCell'], dtype=np.int64)
    counts = np.asarray(arrays['nEdgesOnCell'])
    coefficients = np.asarray(arrays['coeffs_reconstruct'], dtype=np.float64)
    if coefficients.shape != (*edges.shape, 3):
        raise ValueError(f'coeffs_reconstruct layout {coefficients.shape} != {(*edges.shape, 3)}')
    result = np.zeros((normal.shape[0], edges.shape[0], 3), dtype=np.float64)
    for slot in range(edges.shape[1]):
        ids = np.flatnonzero(counts > slot)
        if np.any(edges[ids, slot] < 0) or np.any(edges[ids, slot] >= normal.shape[1]):
            raise ValueError('reconstruction has invalid active edge')
        result[:, ids] += normal[:, edges[ids, slot], None] * coefficients[ids, slot]
    return result


def height_remap_map(weights):
    return getattr(weights, 'redistributes_vertical', False)


def remap_atmosphere(payload, weights, *, source_mesh, target_prepared, config, balance_continuity=False):
    from woof.hex.state import PrognosticState
    from woof.hex.driver import (DrySavedDiagnostics, _pressure_exner,
                                _pressure_perturbation, _normal_velocity)
    from woof.hex.acoustic import edge_signs_on_cells
    from woof.hex.rust_hostprep import correct_vertical_momentum
    old = payload['state']
    vertical = target_prepared.vertical
    if not np.array_equal(payload['vertical'].dzw, vertical.dzw):
        raise ValueError('moving remap requires identical nominal vertical levels')
    from .moving_balance import balance_scope
    balanced, balance_reason = balance_scope(weights, old)
    balance_receipt = dict(applied=False, reason=balance_reason)

    from .moving_balance import _mapped
    from .host_parallel import run_tasks
    import time
    began = time.perf_counter()
    stage_seconds = {}

    def timed(name, operation):
        # Wall seconds of each branch for the receipt (no device sync, no copy; the arithmetic is untouched).
        def run():
            start = time.perf_counter()
            try:
                return operation()
            finally:
                stage_seconds[name] = time.perf_counter() - start
        return run

    def product_map(value):
        # Products of the dry density keep the donor's lowest intensive part
        # below its ground (HeightConservativeMap.mass_intensive, density).
        if height_remap_map(weights):
            return weights.mass_intensive(value, density=old['rho'])
        return getattr(weights, 'mass_intensive', weights.intensive)(value)

    def momentum_product(value):
        # product_map without touching the map's shared last-call receipts (it may run on another thread).
        if height_remap_map(weights):
            return _mapped(weights, value, old['rho'])[0]
        return getattr(weights, 'mass_intensive', weights.intensive)(value)

    def momentum_branch():
        # Momentum, divergence diagnostics and the interface w map read only the carrier and the map,
        # never the remapped thermodynamics, so they run beside it (THE COST THIS REMOVES: ~4.5 s of
        # one core after the 18 s thermodynamics, box-aws2x 2026-10-24).  Same calls, same order.
        return _remap_momentum(payload, weights, source_mesh, target_prepared, momentum_product, mass_map)
    if balanced:
        from .moving_balance import remap_balanced_thermodynamics
        from .moving_balance import extend_below_ground as _extend
        # The below-ground extension is given once, before either branch reads the map.
        _extend(weights, old, [i for i, name in enumerate(payload['scalar_names'])
                               if name in ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'qh')])
        mass_map = lambda value: _mapped(weights, value, None)[0]
        (rho,rho_theta,scalars,mass,heat,tracer_receipts,balance_receipt), momentum = run_tasks([
            timed('thermodynamics', lambda: remap_balanced_thermodynamics(payload, weights, target_prepared,
                                                                          extended=True)),
            timed('momentum', momentum_branch)])
        balance_receipt = dict(balance_receipt, applied=True, reason=balance_reason)
        height_remap = True
    else:
        if height_remap_map(weights):
            from .moving_balance import extend_below_ground
            extend_below_ground(weights, old, [i for i, name in enumerate(payload['scalar_names'])
                                               if name in ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'qh')])
            rho, mass = weights.conserved(old['rho'], density=old['rho'])
        else:
            rho, mass = weights.conserved(old['rho'])
        rho_theta, heat = (weights.conserved(old['rho_theta'], density=old['rho']) if height_remap_map(weights)
                           else weights.conserved(old['rho_theta']))
        height_remap = getattr(weights, 'redistributes_vertical', False)
        mass_map = getattr(weights, 'mass_intensive', weights.intensive)
        def totals(value, *, source):
            if height_remap:
                return weights.totals(value, source=source)
            return (value * (weights.source_area if source else weights.target_area)).sum(axis=-1)
        if np.any(rho <= 0) or np.any(rho_theta <= 0):
            raise ValueError('remap produced non-positive atmospheric mass or theta mass')
        scalars = np.empty((old['scalars'].shape[0], *rho.shape), dtype=np.float32)
        tracer_receipts = {}
        for index, name in enumerate(payload['scalar_names']):
            source = old['rho'].astype(np.float64) * old['scalars'][index]
            exact = product_map(source)
            domain_reconciliation=getattr(weights,'last_domain_reconciliation',None)
            scalars[index] = (exact / rho).astype(np.float32)
            actual = rho.astype(np.float64) * scalars[index]
            before = totals(source, source=True)
            after = totals(actual, source=False)
            envelope = totals(abs(actual-exact), source=False)
            map_error = totals(exact, source=False)-before
            arithmetic = 128*np.finfo(np.float64).eps*totals(abs(exact), source=False)
            if np.any(abs(map_error) > arithmetic):
                raise ValueError(f'{name} FP64 integral exceeds its bounded arithmetic allowance')
            if np.any(abs(after-before) > envelope + arithmetic):
                raise ValueError(f'{name} exceeds its measured remap/cast envelope')
            tracer_receipts[name] = dict(before=before.tolist(), after=after.tolist(),
                                        absolute_error=(after-before).tolist(),
                                        rounding_envelope=envelope.tolist(), fp64_error=map_error.tolist())
            if height_remap:
                tracer_receipts[name]['accounting'] = mass['accounting']
                tracer_receipts[name]['terrain_domain_reconciliation']=domain_reconciliation
                tracer_receipts[name]['positivity_roundoff']=weights.last_positivity_roundoff
        momentum = timed('momentum', momentum_branch)()
    branches_done = time.perf_counter()
    (target_vector, source_momentum, mapped_momentum, radial, cells, gather_cells, rho_u, tangent_momentum,
     divergence, w) = momentum
    arrays = target_prepared.mesh.arrays
    interface_rho, interface_zz = np.zeros_like(w), np.zeros_like(w)
    for level in range(1, rho.shape[0]):
        interface_rho[level] = vertical.fzm[level]*rho[level] + vertical.fzp[level]*rho[level-1]
        interface_zz[level] = vertical.fzm[level]*vertical.zz[level] + vertical.fzp[level]*vertical.zz[level-1]
    rho_w = w*interface_rho*interface_zz
    terrain = target_prepared.terrain_metrics
    rho_w = correct_vertical_momentum(
        counts=arrays['nEdgesOnCell'], edges=arrays['edgesOnCell'],
        fzm=vertical.fzm, fzp=vertical.fzp, ru=rho_u, interface_zz=interface_zz,
        signs=edge_signs_on_cells(target_prepared.mesh).astype(np.float32),
        zb_cell=terrain.zb_cell, zb3_cell=terrain.zb3_cell, rw=rho_w)
    continuity_balance = None
    if balance_continuity:
        from .nest_handoff import balance_regional_mass_flux
        old_total_divergence = mass_flux_divergence(old['rho_u'], source_mesh.arrays)
        old_total_divergence += np.diff(np.asarray(old['rho_w'], dtype=np.float64), axis=0) / np.asarray(payload['vertical'].dzw)[:, None]
        rho_u, continuity_balance = balance_regional_mass_flux(
            rho_u, rho_w, mass_map(old_total_divergence), vertical, arrays)
    state = PrognosticState(rho, rho_theta, rho_u, rho_w, scalars, payload['time_seconds'])
    state.validate(n_cells=rho.shape[1], n_edges=rho_u.shape[1], n_vert_levels=rho.shape[0])
    # Re-derive thermodynamics from the carried mass products on the new
    # Jacobian and reference state, using the same engine equations.
    constants = dict(rgas=287.0, cp=1004.5, reference_pressure=100000.0)
    pressure, exner, theta = _pressure_exner(state, vertical, **constants)
    reference = target_prepared.reference
    perturbation, _, _ = _pressure_perturbation(state, reference, vertical, exner=exner, **constants)
    from .moving_velocity import recover_vertical_velocity
    recovered_w = recover_vertical_velocity(state, vertical, arrays, terrain, device_map=weights)
    if np.any(cells < 0):
        gather_rho = np.concatenate((rho, np.zeros_like(rho[:, :1])), axis=1)
        denominator = gather_rho[:, gather_cells[:, 0]] + gather_rho[:, gather_cells[:, 1]]
        if np.any(denominator <= 0):
            raise ValueError('regional remap has an edge without a positive-density adjacent cell')
        normal = rho.dtype.type(2) * rho_u / denominator
    else:
        normal = _normal_velocity(state, target_prepared.mesh)
    saved = DrySavedDiagnostics(theta, exner, rho-reference.rho_base,
                                rho_theta-reference.rho_theta_base, perturbation,
                                normal, recovered_w)
    saved.validate(rho.shape, rho.dtype, rho_u.shape[1])
    def integrated(item):
        dz=np.asarray(vertical.dzw,dtype=float)
        return {key:float(np.asarray(item[key])[0] if height_remap else np.dot(np.asarray(item[key]),dz)) for key in
                ('before','after','absolute_error','rounding_envelope','fp64_error')}
    stage_seconds['branches'] = branches_done - began
    stage_seconds['finish'] = time.perf_counter() - branches_done
    return state, saved, dict(mass=mass, theta_mass=heat, tracers=tracer_receipts, stage_seconds=stage_seconds,
                             mass_total_kg=integrated(mass),
                             theta_mass_total_kg_kelvin=integrated(heat),
                             tracer_totals_kg={name:integrated(item) for name,item in tracer_receipts.items()},
                             coordinate=(weights.receipt['vertical_remap']['method']+', target terrain Jacobian'
                                         if height_remap else 'equal computational levels, target terrain Jacobian'),
                             momentum='earth Cartesian reconstruction, overlap, tangent projection, edge projection',
                             cartesian_momentum_kg_m_s=dict(source=source_momentum.tolist(),
                                 overlap_before_tangent_projection=mapped_momentum.tolist(),
                                 after_tangent_projection=tangent_momentum.tolist()),
                             divergence=divergence, continuity_balance=continuity_balance,
                             balance=balance_receipt,
                             radial_momentum_abs_max=float(abs(radial).max()),
                             pressure_min_pa=float(pressure.min()), pressure_max_pa=float(pressure.max()),
                             mapped_vertical_velocity_abs_max=float(abs(w).max()),
                             vertical_velocity_abs_max=float(abs(recovered_w).max()), damping_added=False)


def _remap_momentum(payload, weights, source_mesh, target_prepared, product_map, mass_map):
    """The carrier's momentum on the target edges, its divergence receipt and the interface w map.

    Reads only the carrier and the map (never the remapped thermodynamics), so remap_atmosphere runs
    it beside the thermodynamic remap.  The body is the previous inline code, unchanged.
    """
    old = payload['state']
    vertical = target_prepared.vertical
    import os
    gpu_momentum = (getattr(weights, 'backend', 'cpu') == 'cuda'
                    and os.environ.get('WOOF_HEX_MOVING_MOMENTUM_GPU','1') != '0')
    if gpu_momentum:
        from .moving_momentum_cuda import reconstruct_mass, divergence as device_divergence
        vector = reconstruct_mass(weights, old['normal_velocity'], old['rho'], source_mesh.arrays)
    else:
        vector = reconstruct_cartesian(old['normal_velocity'], source_mesh.arrays)
        vector *= old['rho'][:, :, None]
    def flux_divergence(value, arrays, kind):
        return (device_divergence(weights, value, arrays, kind=kind) if gpu_momentum
                else mass_flux_divergence(value, arrays))
    from .host_parallel import run_tasks
    # The three component maps and the source divergence are independent calls; run them together.
    *components, expected_divergence = run_tasks(
        [lambda axis=axis: product_map(vector[:, :, axis]) for axis in range(3)] +
        [lambda: mass_map(flux_divergence(old['rho_u'], source_mesh.arrays, 'source'))])
    target_vector = np.stack(components, axis=-1)
    del components
    dz=np.asarray(vertical.dzw,dtype=np.float64)
    source_momentum=np.einsum('lcj,c,l->j',vector,weights.source_area,dz)
    mapped_momentum=np.einsum('lcj,c,l->j',target_vector,weights.target_area,dz)
    arrays = target_prepared.mesh.arrays
    projected = None
    if gpu_momentum and os.environ.get('WOOF_HEX_MOVING_EDGE_PROJECTION_GPU','1') != '0':
        from .moving_momentum_cuda import project_edges
        projected = project_edges(weights, target_vector, arrays)
    if projected is not None:
        target_vector, radial, cells, gather_cells, rho_u = projected
    else:
        centers = np.stack([arrays[name] for name in ('xCell', 'yCell', 'zCell')], axis=-1).astype(np.float64)
        centers /= np.linalg.norm(centers, axis=-1, keepdims=True)
        radial = np.einsum('lcj,cj->lc', target_vector, centers)
        target_vector -= radial[:, :, None] * centers
        cells = np.asarray(arrays['cellsOnEdge'], dtype=np.int64)
        normals = np.asarray(arrays['edgeNormalVectors'], dtype=np.float64)
        if normals.shape != (len(cells), 3):
            raise ValueError(f'edge normal shape {normals.shape} is unsupported')
        # Native regional cold initialization gives the absent cell zero mass.
        # Never let a -1 sentinel gather the last real atmospheric column.
        gather_vector = np.concatenate((target_vector, np.zeros_like(target_vector[:, :1])), axis=1)
        gather_cells = np.where(cells < 0, target_vector.shape[1], cells)
        rho_u = np.einsum('lej,ej->le', (gather_vector[:, gather_cells[:, 0]] + gather_vector[:, gather_cells[:, 1]])*.5,
                          normals).astype(np.float32)
    tangent_momentum=np.einsum('lcj,c,l->j',target_vector,weights.target_area,dz)
    target_divergence=flux_divergence(rho_u, arrays, 'target')
    def divergence_rms(value):
        return float(np.sqrt(np.einsum('lc,c,l->',value*value,weights.target_area,dz)/
                             (weights.target_area.sum()*dz.sum())))
    divergence=dict(coordinate='computational mass-flux divergence',
        units='kg m-3 s-1',expected_overlap_rms=divergence_rms(expected_divergence),
        target_edge_rms=divergence_rms(target_divergence),
        difference_rms=divergence_rms(target_divergence-expected_divergence),
        difference_abs_max=float(abs(target_divergence-expected_divergence).max()),
        target_global_integral_kg_s=np.einsum('lc,c,l->',target_divergence,weights.target_area,dz).item(),
        balance_cleanup_added=False)
    del expected_divergence,target_divergence
    w_map = getattr(weights, 'interface_intensive', weights.intensive)
    w = w_map(old['vertical_velocity']).astype(np.float32)
    return (target_vector, source_momentum, mapped_momentum, radial, cells, gather_cells, rho_u,
            tangent_momentum, divergence, w)


def mass_flux_divergence(edge_flux, arrays):
    """Closed-cell MPAS flux divergence, accumulated independently in FP64."""
    from woof.hex.acoustic import edge_signs_on_cells
    # The helper's public contract accepts a mesh with these connectivity arrays.
    from types import SimpleNamespace
    signs=edge_signs_on_cells(SimpleNamespace(arrays=arrays))
    edges=np.asarray(arrays['edgesOnCell'],dtype=np.int64)
    counts=np.asarray(arrays['nEdgesOnCell'])
    length=np.asarray(arrays['dvEdge'],dtype=np.float64)
    area=np.asarray(arrays['areaCell'],dtype=np.float64)
    result=np.zeros((edge_flux.shape[0],len(area)),dtype=np.float64)
    for slot in range(edges.shape[1]):
        ids=np.flatnonzero(counts>slot);selected=edges[ids,slot]
        if np.any(selected<0) or np.any(selected>=edge_flux.shape[1]):
            raise ValueError('divergence has invalid active edge')
        result[:,ids]+=edge_flux[:,selected].astype(np.float64)*length[selected]*signs[ids,slot]
    return result/area
