"""State transfer across a frozen-far-field move (woof.hex.moving_zone).

Every cell outside the changed zone keeps its polygon, static fields and
vertical coordinate, so its prognostic and saved diagnostic columns are copied
bit for bit, and every face of such a cell keeps its mass flux (sign-corrected
for the face orientation). Only the changed zone is remapped: mass products by
the closed zone overlap in physical height, momentum through the earth-Cartesian
reconstruction. No global rescaling, damping or smoothing touches carried cells.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

RGAS, CP, P0 = 287.0, 1004.5, 100000.0


def _totals(field, area, dzw):
    return float(np.einsum('lc,c,l->', np.asarray(field, dtype=np.float64), area, dzw))


def _project_zone_divergence(rho_u, divergence, expected, zone_cells, free_edges, arrays, area):
    """Solve L phi = A (E - D) on the zone graph (faces with both cells remapped and not carried).

    dF_e = (phi_c0 - phi_c1) / dc_e on those faces gives A_i div_i(dF) = sum_e dv_e/dc_e (phi_i - phi_j),
    a weighted graph Laplacian; carried faces are fixed (Neumann). One factorisation, every level.
    """
    from scipy.sparse import csr_matrix, identity
    from scipy.sparse.linalg import splu
    from scipy.sparse.csgraph import connected_components
    cells = np.asarray(arrays['cellsOnEdge'], dtype=np.int64)
    dv = np.asarray(arrays['dvEdge'], dtype=np.float64)
    dc = np.asarray(arrays['dcEdge'], dtype=np.float64)
    in_zone = np.zeros(len(area), dtype=bool)
    in_zone[zone_cells] = True
    edges = np.flatnonzero(free_edges & in_zone[cells[:, 0]] & in_zone[cells[:, 1]])
    nodes = np.unique(cells[edges].ravel())
    local = np.full(len(area), -1, dtype=np.int64)
    local[nodes] = np.arange(len(nodes))
    a, b = local[cells[edges, 0]], local[cells[edges, 1]]
    weight = dv[edges] / dc[edges]
    n = len(nodes)
    laplacian = csr_matrix((np.concatenate((weight, weight, -weight, -weight)),
                            (np.concatenate((a, b, a, b)), np.concatenate((a, b, b, a)))), shape=(n, n))
    count, labels = connected_components(laplacian, directed=False)
    # Zone positions of the graph nodes inside the (level, zone_cells) arrays.
    position = np.full(len(area), -1, dtype=np.int64)
    position[zone_cells] = np.arange(len(zone_cells))
    rows = position[nodes]
    weights_area = area[nodes]
    rhs = (expected[:, rows] - divergence[:, nodes]) * weights_area[None, :]
    # Each connected component must be compatible (its boundary flux is carried exactly):
    # remove the rounding-level component mean so the singular system is consistent.
    residual_before = []
    for component in range(count):
        members = labels == component
        mean = rhs[:, members].sum(axis=1) / members.sum()
        residual_before.append(float(np.abs(rhs[:, members].sum(axis=1)).max()))
        rhs[:, members] -= mean[:, None]
    scale = float(laplacian.diagonal().mean())
    factor = splu((laplacian + identity(n, format='csr') * (1e-12 * scale)).tocsc())
    phi = factor.solve(np.ascontiguousarray(rhs.T)).T
    correction = (phi[:, a] - phi[:, b]) / dc[edges][None, :]
    receipt = dict(edges=edges, projection_faces=int(len(edges)), projection_cells=int(n),
                   projection_components=int(count),
                   projection_component_imbalance_abs_max=max(residual_before) if residual_before else 0.0,
                   projection_correction_rms=float(np.sqrt(np.mean(correction ** 2))),
                   projection_rho_u_rms=float(np.sqrt(np.mean(rho_u[:, edges].astype(np.float64) ** 2))))
    return correction, receipt


MOIST_SPECIES = ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'qh')
GRAVITY = 9.80616


def _vertical_residual_from_saved(old, vertical, columns, moist):
    """Native w-equation pressure/buoyancy residual of source columns (F:6784 form), interfaces 0..nl.

    rdzu (pp_k - pp_k-1) - (fzm dpdz_k + fzp dpdz_k-1), dpdz = -g (rb qtot + rr (1 + qtot)), from the
    carried saved pressure and density perturbations (exact model diagnostics at the boundary).
    """
    pp = old['pressure_perturbation'][:, columns].astype(np.float64)
    rr = old['density_perturbation'][:, columns].astype(np.float64)
    rb = old['rho'][:, columns].astype(np.float64) - rr
    qtot = sum(old['scalars'][i][:, columns].astype(np.float64) for i in moist) if moist else 0.0
    dpdz = -GRAVITY * (rb * qtot + rr * (1.0 + qtot))
    nl = pp.shape[0]
    residual = np.zeros((nl + 1, pp.shape[1]))
    rdzu = np.asarray(vertical.rdzu, dtype=np.float64)
    fzm, fzp = np.asarray(vertical.fzm, dtype=np.float64), np.asarray(vertical.fzp, dtype=np.float64)
    for k in range(1, nl):
        residual[k] = rdzu[k] * (pp[k] - pp[k - 1]) - (fzm[k] * dpdz[k] + fzp[k] * dpdz[k - 1])
    return residual


def _balance_columns(rho, theta_m, tracer_mass, moist, target_residual, vertical, reference, columns, dzw,
                     *, newton=6, secant=12):
    """Solve each column's density so the native vertical residual equals `target_residual`.

    theta_m and every tracer MASS are held; the column dry mass is held exactly (secant on the
    top density); the native pressure perturbation uses the model's own reference split
    pp = p - zz R rtb exner_base, so a carried source residual (hydrostatic plus any resolved
    non-hydrostatic forcing) is reproduced on the new vertical coordinate.
    """
    rgas, cp = 287.0, 1004.5
    cv = cp - rgas
    zz = np.asarray(vertical.zz, dtype=np.float64)[:, columns]
    rdzu = np.asarray(vertical.rdzu, dtype=np.float64)
    fzm, fzp = np.asarray(vertical.fzm, dtype=np.float64), np.asarray(vertical.fzp, dtype=np.float64)
    rb = np.asarray(reference.rho_base, dtype=np.float64)[:, columns]
    rtb = np.asarray(reference.rho_theta_base, dtype=np.float64)[:, columns]
    exb = np.asarray(reference.exner_base, dtype=np.float64)[:, columns]
    pb = zz * rgas * rtb * exb
    nl = rho.shape[0]
    target_mass = np.einsum('lc,l->c', rho, dzw)

    def pressure(x, k):
        return 100000.0 * (rgas * zz[k] * x * theta_m[k] / 100000.0) ** (cp / cv)

    def qtot_of(x, k):
        return sum(tracer_mass[i][k] for i in moist) / x if moist else 0.0

    def shoot(top):
        out = np.empty_like(rho)
        out[-1] = top
        for k in range(nl - 1, 0, -1):
            upper = out[k]
            qk = qtot_of(upper, k)
            dpdz_k = -GRAVITY * (rb[k] * qk + (upper - rb[k]) * (1 + qk))
            ppk = pressure(upper, k) - pb[k]
            x = rho[k - 1].copy()
            for _ in range(newton):
                # tracer masses fixed: qtot*x = Q is constant, so dpdz linear in x
                big_q = sum(tracer_mass[i][k - 1] for i in moist) if moist else 0.0
                dpdz_lower = -GRAVITY * (rb[k - 1] * big_q / x + (x - rb[k - 1]) + big_q - rb[k - 1] * big_q / x)
                p_lower = pressure(x, k - 1)
                f = rdzu[k] * (ppk - (p_lower - pb[k - 1])) - (fzm[k] * dpdz_k + fzp[k] * dpdz_lower) - \
                    target_residual[k]
                derivative = -rdzu[k] * (cp / cv) * p_lower / x + fzp[k] * GRAVITY
                x = np.maximum(x - f / derivative, 1e-6 * rho[k - 1])
            out[k - 1] = x
        return out

    top0 = rho[-1].copy()
    a = top0
    fa = np.einsum('lc,l->c', shoot(a), dzw) - target_mass
    b = top0 * (1 + 1e-3)
    fb = np.einsum('lc,l->c', shoot(b), dzw) - target_mass
    for _ in range(secant):
        step = np.where(fb != fa, (b - a) / np.where(fb != fa, fb - fa, 1.0) * fb, 0.0)
        a, fa = b, fb
        b = np.maximum(b - step, 1e-6 * top0)
        fb = np.einsum('lc,l->c', shoot(b), dzw) - target_mass
        if np.max(np.abs(fb) / target_mass) < 1e-13:
            break
    result = shoot(b)
    change = np.abs(result / rho - 1)
    receipt = dict(method='native vertical residual reproduced per remapped column (theta_m, tracer masses and '
                          'column dry mass held)',
                   columns=int(rho.shape[1]), column_mass_relative_abs_max=float(np.max(np.abs(fb) / target_mass)),
                   density_relative_change_abs_max=float(change.max()),
                   density_relative_change_rms=float(np.sqrt(np.mean(change ** 2))),
                   top_density_relative_change_abs_max=float(np.max(np.abs(b / top0 - 1))))
    return result, receipt


def remap_zone_atmosphere(payload, zone, sub, *, source_mesh, target_prepared, height='absolute',
                          divergence_projection=True, balance_columns=False):
    """Return (PrognosticState, DrySavedDiagnostics, receipt) on the target member."""
    from woof.hex.state import PrognosticState
    from woof.hex.driver import DrySavedDiagnostics, _pressure_exner, _pressure_perturbation, _normal_velocity
    from woof.hex.acoustic import edge_signs_on_cells
    from woof.hex.rust_hostprep import correct_vertical_momentum
    from .moving_vertical import HeightConservativeMap
    from .moving_state import reconstruct_cartesian, mass_flux_divergence
    from .moving_velocity import recover_vertical_velocity
    old = payload['state']
    sv, tv = payload['vertical'], target_prepared.vertical
    cs, ct = zone.carried_source, zone.carried_target
    zs, zt = zone.remapped_source, zone.remapped_target  # changed polygons + changed vertical coordinate
    es, et, sign = zone.edge_source, zone.edge_target, zone.edge_sign.astype(np.float32)
    nl = old['rho'].shape[0]
    nt = zone.n_target
    if not np.array_equal(sv.dzw, tv.dzw):
        raise ValueError('zone remap requires identical nominal vertical levels')
    dzw = np.asarray(tv.dzw, dtype=np.float64)
    source_area = np.asarray(source_mesh.arrays['areaCell'], dtype=np.float64)
    target_area = np.asarray(target_prepared.mesh.arrays['areaCell'], dtype=np.float64)
    if len(source_area) != zone.n_source or len(target_area) != nt:
        raise ValueError('zone map cell counts differ from the carried and target meshes')
    # Carried columns must have the same vertical coordinate and reference state.
    vertical_identity = dict(
        zgrid_abs_max_m=float(np.abs(np.asarray(sv.zgrid)[:, cs] - np.asarray(tv.zgrid)[:, ct]).max()),
        zz_abs_max=float(np.abs(np.asarray(sv.zz)[:, cs] - np.asarray(tv.zz)[:, ct]).max()),
        area_relative_max=float(np.abs(target_area[ct] / source_area[cs] - 1).max()))
    if vertical_identity['zgrid_abs_max_m'] > 1e-3:
        raise ValueError(f'carried columns changed vertical coordinate: {vertical_identity}')
    zone_map = HeightConservativeMap(sub, np.asarray(sv.zgrid)[:, zs], np.asarray(tv.zgrid)[:, zt], sv.dzw, tv.dzw,
                                     endpoint_alignment=(height != 'absolute'), domain_reconciliation='global')
    rho = np.empty((nl, nt), dtype=np.float32)
    rho[:, ct] = old['rho'][:, cs]
    rho[:, zt], mass = zone_map.conserved(old['rho'][:, zs])
    rho_theta = np.empty_like(rho)
    rho_theta[:, ct] = old['rho_theta'][:, cs]
    rho_theta[:, zt], heat = zone_map.conserved(old['rho_theta'][:, zs])
    if np.any(rho[:, zt] <= 0) or np.any(rho_theta[:, zt] <= 0):
        raise ValueError('zone remap produced non-positive mass or theta mass')
    scalars = np.empty((old['scalars'].shape[0], nl, nt), dtype=np.float32)
    rho_zone = rho[:, zt].astype(np.float64)
    tracer_mass = []
    for index in range(old['scalars'].shape[0]):
        scalars[index][:, ct] = old['scalars'][index][:, cs]
        product = old['rho'][:, zs].astype(np.float64) * old['scalars'][index][:, zs]
        tracer_mass.append(zone_map.mass_intensive(product))
        scalars[index][:, zt] = (tracer_mass[-1] / rho_zone).astype(np.float32)
    balance = None
    if balance_columns:
        # D4: discrete vertical balance of every remapped column (see _balance_columns).
        moist = [i for i, name in enumerate(payload['scalar_names']) if name in MOIST_SPECIES]
        source_residual = _vertical_residual_from_saved(old, sv, zs, moist)
        target_residual = zone_map.interface_intensive(source_residual)
        theta_m = rho_theta[:, zt].astype(np.float64) / rho_zone
        new_rho, balance = _balance_columns(
            rho_zone, theta_m, np.stack(tracer_mass), moist, target_residual, tv, target_prepared.reference,
            zt, dzw)
        rho[:, zt] = new_rho.astype(np.float32)
        rho_theta[:, zt] = (new_rho * theta_m).astype(np.float32)
        for index in range(old['scalars'].shape[0]):
            scalars[index][:, zt] = (tracer_mass[index] / new_rho).astype(np.float32)
    del tracer_mass
    # Momentum: zone cells from the overlap of source cell momentum vectors.
    arrays = target_prepared.mesh.arrays
    vector = reconstruct_cartesian(old['normal_velocity'], source_mesh.arrays)
    vector *= old['rho'][:, :, None]
    target_vector = np.zeros((nl, nt, 3))
    for axis in range(3):
        target_vector[:, zt, axis] = zone_map.mass_intensive(vector[:, zs, axis])
    target_vector[:, ct] = vector[:, cs]
    del vector
    centres = np.stack([arrays[name] for name in ('xCell', 'yCell', 'zCell')], axis=-1).astype(np.float64)
    centres /= np.linalg.norm(centres, axis=-1, keepdims=True)
    radial = np.einsum('lcj,cj->lc', target_vector, centres)
    target_vector -= radial[:, :, None] * centres
    cells = np.asarray(arrays['cellsOnEdge'], dtype=np.int64)
    normals = np.asarray(arrays['edgeNormalVectors'], dtype=np.float64)
    rho_u = np.einsum('lej,ej->le', (target_vector[:, cells[:, 0]] + target_vector[:, cells[:, 1]]) * .5,
                      normals).astype(np.float32)
    del target_vector
    rho_u[:, et] = old['rho_u'][:, es] * sign
    zone_edges = np.ones(zone.n_target_edges, dtype=bool)
    zone_edges[et] = False
    # Divergence of the zone: carried faces keep every carried cell's divergence exactly.
    source_divergence = mass_flux_divergence(old['rho_u'], source_mesh.arrays)
    target_divergence = mass_flux_divergence(rho_u, arrays)
    expected = sub.intensive(source_divergence[:, zs])
    def rms(value, area):
        return float(np.sqrt(np.einsum('lc,c,l->', value * value, area, dzw) / (area.sum() * dzw.sum())))
    divergence = dict(
        carried_cells_abs_max_difference=float(np.abs(target_divergence[:, ct] - source_divergence[:, cs]).max()),
        zone_expected_rms=rms(expected, target_area[zt]),
        zone_target_rms=rms(target_divergence[:, zt], target_area[zt]),
        zone_difference_rms_before_projection=rms(target_divergence[:, zt] - expected, target_area[zt]))
    if divergence_projection:
        # D2: minimal gradient correction on zone-internal faces so each remapped cell's
        # mass-flux divergence (hence its column mass tendency and surface pressure
        # tendency) equals the conservative overlap of the source divergence.
        correction, projection = _project_zone_divergence(
            rho_u, target_divergence, expected, zt, zone_edges, arrays, target_area)
        rho_u[:, projection['edges']] = (rho_u[:, projection['edges']].astype(np.float64) +
                                         correction).astype(np.float32)
        target_divergence = mass_flux_divergence(rho_u, arrays)
        divergence.update({k: v for k, v in projection.items() if k != 'edges'})
    divergence['zone_difference_rms'] = rms(target_divergence[:, zt] - expected, target_area[zt])
    divergence['divergence_projection'] = bool(divergence_projection)
    del source_divergence, target_divergence, expected
    # Vertical velocity: interface heights in the zone, exact columns outside.
    w = np.empty((nl + 1, nt), dtype=np.float32)
    w[:, ct] = old['vertical_velocity'][:, cs]
    w[:, zt] = zone_map.interface_intensive(old['vertical_velocity'][:, zs]).astype(np.float32)
    interface_rho, interface_zz = np.zeros_like(w), np.zeros_like(w)
    for level in range(1, nl):
        interface_rho[level] = tv.fzm[level] * rho[level] + tv.fzp[level] * rho[level - 1]
        interface_zz[level] = tv.fzm[level] * tv.zz[level] + tv.fzp[level] * tv.zz[level - 1]
    terrain = target_prepared.terrain_metrics
    rho_w = correct_vertical_momentum(
        counts=arrays['nEdgesOnCell'], edges=arrays['edgesOnCell'], fzm=tv.fzm, fzp=tv.fzp, ru=rho_u,
        interface_zz=interface_zz, signs=edge_signs_on_cells(target_prepared.mesh).astype(np.float32),
        zb_cell=terrain.zb_cell, zb3_cell=terrain.zb3_cell, rw=w * interface_rho * interface_zz)
    rho_w = np.asarray(rho_w, dtype=np.float32)
    rho_w[:, ct] = old['rho_w'][:, cs]
    state = PrognosticState(rho, rho_theta, rho_u, rho_w, scalars, payload['time_seconds'])
    state.validate(n_cells=nt, n_edges=rho_u.shape[1], n_vert_levels=nl)
    constants = dict(rgas=RGAS, cp=CP, reference_pressure=P0)
    pressure, exner, theta = _pressure_exner(state, tv, **constants)
    reference = target_prepared.reference
    perturbation, _, _ = _pressure_perturbation(state, reference, tv, exner=exner, **constants)
    recovered_w = recover_vertical_velocity(state, tv, arrays, terrain)
    normal_velocity = _normal_velocity(state, target_prepared.mesh)
    # Saved time-level-one diagnostics: exact source values on carried columns and on
    # faces between two carried cells; zone columns derive from the new state.
    for target_array, name in ((theta, 'theta_m'), (exner, 'exner'), (perturbation, 'pressure_perturbation'),
                               (recovered_w, 'vertical_velocity')):
        target_array[:, ct] = old[name][:, cs]
    density_perturbation = rho - reference.rho_base
    rho_theta_perturbation = rho_theta - reference.rho_theta_base
    density_perturbation[:, ct] = old['density_perturbation'][:, cs]
    rho_theta_perturbation[:, ct] = old['rho_theta_perturbation'][:, cs]
    carried_mask = np.zeros(nt, dtype=bool)
    carried_mask[ct] = True
    inner = carried_mask[cells[et, 0]] & carried_mask[cells[et, 1]]
    normal_velocity[:, et[inner]] = old['normal_velocity'][:, es[inner]] * sign[inner]
    saved = DrySavedDiagnostics(theta, exner, density_perturbation, rho_theta_perturbation, perturbation,
                                normal_velocity, recovered_w)
    saved.validate(rho.shape, rho.dtype, rho_u.shape[1])
    # Global FP64 budgets on each member's own metric.
    budgets = {}
    def budget(name, before, after):
        budgets[name] = dict(before=before, after=after, relative=(after - before) / before if before else 0.0)
    budget('dry_mass', _totals(old['rho'], source_area, dzw), _totals(rho, target_area, dzw))
    budget('theta_mass', _totals(old['rho_theta'], source_area, dzw), _totals(rho_theta, target_area, dzw))
    for index, name in enumerate(payload['scalar_names']):
        before = _totals(old['rho'].astype(np.float64) * old['scalars'][index], source_area, dzw)
        after = _totals(rho.astype(np.float64) * scalars[index], target_area, dzw)
        budget(name, before, after)
    carried_mass = dict(before=_totals(old['rho'][:, cs], source_area[cs], dzw),
                        after=_totals(rho[:, ct], target_area[ct], dzw))
    receipt = dict(
        method='frozen-far-field zone transfer: exact carried columns and faces; zone by closed overlap in '
               f'{"absolute physical height" if height == "absolute" else "terrain-aligned height"}',
        vertical_identity=vertical_identity, budgets=budgets, carried_mass=carried_mass,
        zone_mass=mass, zone_theta_mass=heat, divergence=divergence,
        zone_domain_reconciliation=zone_map.last_domain_reconciliation, zone_column_balance=balance,
        carried_cells=int(len(ct)), zone_cells=int(len(zt)), carried_faces=int(len(et)),
        zone_faces=int(zone_edges.sum()),
        zone_vertical_velocity_abs_max=float(np.abs(recovered_w[:, zt]).max()),
        pressure_min_pa=float(pressure.min()), pressure_max_pa=float(pressure.max()),
        radial_momentum_abs_max=float(np.abs(radial).max()), damping_added=False)
    return state, saved, receipt


def resume_zone_host(host, payload, *, zone_path, receipt_path):
    """Replace the target member's initial state by the zone transfer of the sealed carrier."""
    from dataclasses import replace
    from .moving_zone import ZoneMap, CompositeMap
    began = time.perf_counter()
    print('[moving-zone] loading zone map', flush=True)
    zone = ZoneMap.load(zone_path)
    prepared = host['prepared']
    radius = float(prepared.mesh.attrs['sphere_radius'])
    if float(payload['mesh'].attrs['sphere_radius']) != radius:
        raise ValueError('moving forecast earth radii differ')
    if tuple(payload['scalar_names']) != tuple(host['scalar_names']):
        raise ValueError('moving carrier species differ from the running physics')
    source_area = np.asarray(payload['mesh'].arrays['areaCell'], dtype=np.float64)
    target_area = np.asarray(prepared.mesh.arrays['areaCell'], dtype=np.float64)
    if len(source_area) != zone.n_source or len(target_area) != zone.n_target:
        raise ValueError('zone map was built for other members')
    generator = zone.zone_overlap
    from .moving_remap import ConservativeMap
    scaled = ConservativeMap(generator.overlap * radius ** 2, generator.source_area * radius ** 2,
                             generator.target_area * radius ** 2,
                             dict(generator.receipt, effective_area_radius_m=radius))
    zone_physical = scaled.with_physical_areas(source_area[zone.zone_source], target_area[zone.zone_target],
                                               generator_areas=(generator.source_area, generator.target_area),
                                               generator_area_scale=radius ** 2)
    sub = zone.remapped_operator(zone_physical, source_area, target_area)
    composite = CompositeMap(zone, sub, source_area, target_area)
    print('[moving-zone] transferring atmosphere', flush=True)
    stage = time.perf_counter()
    import os
    height = os.environ.get('WOOF_HEX_MOVING_ZONE_HEIGHT', 'absolute')
    state, saved, receipt = remap_zone_atmosphere(payload, zone, sub, source_mesh=payload['mesh'],
                                                  target_prepared=prepared, height=height,
                                                  divergence_projection=os.environ.get(
                                                      'GPUWM_HEX_MOVING_ZONE_DIVERGENCE', '1') != '0',
                                                  balance_columns=os.environ.get(
                                                      'GPUWM_HEX_MOVING_ZONE_BALANCE', '0') == '1')
    selected = replace(prepared, state=state, saved_diagnostics=saved)
    for name in ('advection_coefficients', 'expected_execution_fingerprint'):
        object.__setattr__(selected, name, getattr(prepared, name))
    receipt.update(zone_map=zone.receipt, zone_operator_physical=sub.receipt, time_seconds=payload['time_seconds'],
                   atmosphere_transfer_seconds=time.perf_counter() - stage,
                   total_atmospheric_transfer_seconds=time.perf_counter() - began)
    Path(receipt_path).write_text(json.dumps(receipt, indent=2, default=float) + '\n')
    print('[moving-zone] atmospheric transfer complete ' + json.dumps(
        {k: v['relative'] for k, v in receipt['budgets'].items()}), flush=True)
    return dict(host, prepared=selected), composite, receipt
