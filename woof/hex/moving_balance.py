"""Constrained moving-mesh thermodynamic initialization; never a time filter.

The native vertical pressure/buoyancy operator does not commute with a
conservative remap or the nonlinear equation of state. Carry its acceleration
and use weighted Newton corrections of density/theta mass subject to
that operator and both column integrals. Moist tracer masses are held fixed.
The Newton projection uses a tridiagonal Schur block plus two integral rows:
O(levels * cells) work and storage, with no dense per-column matrices.

This constrains the vertical pressure contribution, not horizontal balance,
transport or total vertical acceleration. Terrain-domain changes can require
real pressure changes; no local projection can remove those while retaining
all the original column budgets.
"""

import os

import numpy as np

#: The saved native diagnostics the carried force is computed from.  A moving
#: checkpoint carries them; an analysis transfer (woof.hex.analysis_remap) reads
#: only prognostic fields from an analysis file and has none.
CARRIED_FORCE_FIELDS = ("pressure_perturbation", "density_perturbation", "exner")

#: Vapour and cloud water: the pair the remapped state is saturation-adjusted on (saturation_adjustment).
SATURATION_SPECIES = ("qv", "qc")
#: Latent heat of vaporisation and dry heat capacity as MPAS-A v8.4.2 Thompson uses them (lvap0, cp).
LATENT_HEAT, DRY_CP, DRY_RGAS, VAPOR_RGAS = 2.5e6, 1004.5, 287.0, 461.6


def liquid_saturation_mixing_ratio(temperature, pressure):
    """Saturation mixing ratio over liquid water as MPAS-A v8.4.2 Thompson computes it (``rslf``).

    Flatau et al. (1992) polynomial, module_mp_thompson.F.
    """
    x = np.maximum(-80.0, np.asarray(temperature, dtype=np.float64) - 273.16)
    esl = .611583699e03 + x * (.444606896e02 + x * (.143177157e01 + x * (.264224321e-1 + x * (
        .299291081e-3 + x * (.203154182e-5 + x * (.702620698e-8 + x * (.379534310e-11
                                                                    + x * (-.321582393e-13))))))))
    pressure = np.asarray(pressure, dtype=np.float64)
    esl = np.minimum(esl, pressure * 0.15)
    return 0.622 * esl / np.maximum(1.0e-4, pressure - esl)


def evaporation(rho, heat, zz, vapor, cloud, *, backend='cpu'):
    """Cloud water (mixing ratio) a cell evaporates to reach liquid saturation, with its latent cooling.

    ``rho`` and ``heat`` are dry density and theta_m mass (``rho_theta``), ``vapor``/``cloud`` mixing
    ratios.  Newton iterations on qv + x = qs(T - L x / cp), at the cell's Exner function; never more
    than the cloud water there.  Returns (x, the cell's temperature before, its Exner function).

    Every operation is elementwise, so level-major fields run in blocks of whole levels on all cores
    (host_parallel.row_map) and each word is the one the single whole-field call writes.  THE COST THIS
    REMOVES: two evaporations per move (donor and target) were 3.5 + 3.75 s of one core in a resumed
    storm-following segment (box-aws2x 2026-10-24, sampled), the largest single-core part of the remap.
    """
    arrays = [np.asarray(value) for value in (rho, heat, zz, vapor, cloud)]
    shape = arrays[0].shape
    if len(shape) != 2 or any(value.shape != shape for value in arrays):
        return _evaporation(*arrays)
    if (backend == 'cuda' and os.environ.get('WOOF_HEX_MOVING_SATURATION_GPU', '1') != '0'
            and all(np.isfinite(value).all() for value in arrays)
            and np.all(arrays[0] > 0) and np.all(arrays[1] > 0) and np.all(arrays[2] > 0)):
        from .moving_saturation_cuda import evaporation as device_evaporation
        return device_evaporation(*arrays)
    from .host_parallel import row_map
    x, temperature, exner = (np.empty(shape, dtype=np.float64) for _ in range(3))

    def block(rows):
        x[rows], temperature[rows], exner[rows] = _evaporation(*(value[rows] for value in arrays))
    row_map(block, shape[0], elements=int(np.prod(shape)) * 8)
    return x, temperature, exner


def _evaporation(rho, heat, zz, vapor, cloud):
    rho = np.asarray(rho, dtype=np.float64)
    heat = np.asarray(heat, dtype=np.float64)
    vapor = np.asarray(vapor, dtype=np.float64)
    cloud = np.maximum(np.asarray(cloud, dtype=np.float64), 0.0)
    exner = (np.asarray(zz, dtype=np.float64) * (DRY_RGAS / 1.0e5) * heat) ** (DRY_RGAS / (DRY_CP - DRY_RGAS))
    pressure = 1.0e5 * exner ** (DRY_CP / DRY_RGAS)
    temperature = heat / rho / (1.0 + VAPOR_RGAS / DRY_RGAS * vapor) * exner
    x = np.zeros_like(cloud)
    for _ in range(4):
        cooled = temperature - LATENT_HEAT / DRY_CP * x
        saturation = liquid_saturation_mixing_ratio(cooled, pressure)
        slope = (liquid_saturation_mixing_ratio(cooled + 0.01, pressure) - saturation) / 0.01
        x = x - (vapor + x - saturation) / (1.0 + LATENT_HEAT / DRY_CP * slope)
        x = np.clip(x, 0.0, cloud)
    return x, temperature, exner


def saturation_adjustment(vapor, cloud, rho, heat, zz, donor_evaporation, total, *, threshold=1.0e-9, backend='cpu'):
    """Evaporate the cloud water the remap itself left in subsaturated air, with its latent cooling.

    ``vapor``/``cloud`` are the target's FP64 water products (dry density times mixing ratio), ``rho`` and
    ``heat`` its dry density and theta_m products, ``donor_evaporation`` the donors' own evaporation
    (``evaporation`` on the source, times dry density) carried by the same overlap, ``total`` the target
    integral.  Returns (vapor, cloud, heat, receipt).

    THE BREAKAGE THIS PREVENTS: the overlap remap carries vapour, cloud water and theta_m each
    conservatively, so mixing a cloudy donor with drier donors (or with the warmer constant-theta air a
    higher donor lends below its ground) leaves cloud water in subsaturated air: 9.4 % of all cloud
    water on the 2025-11-08 12Z smoke case, against 0.06 % in the carried state itself.  Thompson
    evaporated it on its first call after the move, at fixed density, so the cooling (0.4-0.5 K in the
    lowest layers of far-field fog cells at 48.0N 126.3E, 23.5N 104.5E, 29.8N 102.5E) dropped their
    pressure at once and the surface pressure fell and rebounded 200-260 Pa in two 225 s steps
    (1.15 Pa/s, 3.6 times the fixed run).

    The same evaporation, with the same latent cooling, now happens here, before the balanced
    initialization builds the column, so the column starts in balance with it.  Only the excess over
    what the donors would evaporate themselves is adjusted (the subsaturated cloud a running model
    holds between microphysics calls is carried as it is).  Dry mass and each cell's total water are
    unchanged; the phase change moves water from cloud to vapour and lowers theta_m mass by its latent
    cooling, and the receipt declares both.
    """
    rho = np.asarray(rho, dtype=np.float64)
    heat = np.asarray(heat, dtype=np.float64)
    if (backend == 'cuda' and os.environ.get('WOOF_HEX_MOVING_PHASE_GPU','1') != '0'
            and os.environ.get('WOOF_HEX_MOVING_SATURATION_GPU','1') != '0'
            and rho.ndim == 2 and rho.size
            and all(np.shape(x) == rho.shape and np.isfinite(x).all()
                    for x in (vapor,cloud,heat,zz,donor_evaporation))
            and np.isfinite(rho).all() and np.all(rho>0) and np.all(heat>0) and np.all(zz>0)):
        from .moving_saturation_cuda import adjustment as device_adjustment
        return device_adjustment(vapor,cloud,rho,heat,zz,donor_evaporation,total,threshold)
    x, temperature, exner = evaporation(rho, heat, zz, vapor / rho, cloud / rho, backend=backend)
    moved = (x * rho) - donor_evaporation
    moved = np.where(moved > threshold * rho, np.minimum(moved, np.maximum(cloud, 0.0)), 0.0)
    receipt = dict(method="target saturation adjustment of the cloud water the remap left subsaturated beyond "
                          "the donors' own; latent cooling applied before the balanced initialization",
                   cells=int(np.count_nonzero(np.any(moved > 0, axis=0))), layers=int(np.count_nonzero(moved > 0)),
                   cloud_to_vapor_kg=float(total(moved)), cloud_total_kg=float(total(cloud)),
                   largest_mixing_ratio=float(np.max(moved / rho, initial=0.0)))
    if not np.any(moved > 0):
        receipt.update(theta_mass_change_kg_kelvin=0.0, largest_cooling_k=0.0)
        return vapor, cloud, heat, receipt
    q_old = vapor / rho
    q_new = q_old + moved / rho
    cooled = temperature - LATENT_HEAT / DRY_CP * moved / rho
    new_heat = rho * cooled / exner * (1.0 + VAPOR_RGAS / DRY_RGAS * q_new)
    new_heat = np.where(moved > 0, new_heat, heat)
    receipt.update(theta_mass_change_kg_kelvin=float(total(new_heat - heat)),
                   largest_cooling_k=float(np.max(temperature - cooled)))
    return vapor + moved, cloud - moved, new_heat, receipt


def balance_scope(weights, state, environ=None):
    """``(apply, reason)``: whether a remap takes the balanced initialization.

    It applies to every height remap whose payload carries the saved native
    pressure split, unless ``WOOF_HEX_MOVING_BALANCE=0`` selects the previous
    thermodynamic path (kept as a comparison knob; the default is on).  THE
    BREAKAGE THE PAYLOAD CHECK PREVENTS: an analysis transfer has no saved
    pressure split and the balanced path reads it by key, so every analysis
    remap onto a new mesh would fail with KeyError once balance is the
    default.  It keeps its previous path and its receipt says why, by name.
    """
    environ = os.environ if environ is None else environ
    if not getattr(weights, "redistributes_vertical", False):
        return False, "horizontal-only map: the vertical geometry does not change"
    missing = [name for name in CARRIED_FORCE_FIELDS if name not in state]
    if missing:
        return False, ("the payload carries no saved native pressure split (missing "
                       + ", ".join(missing) + "), so there is no source force to carry; "
                       "an analysis transfer takes this path")
    if environ.get("WOOF_HEX_MOVING_BALANCE", "1") == "0":
        return False, ("WOOF_HEX_MOVING_BALANCE=0: the previous thermodynamic path, "
                       "kept as a comparison knob")
    return True, "carried native vertical pressure acceleration"


def project_columns(rho, heat, tracer, acceleration, v, ref, *, backend='cpu'):
    # Minimum relative thermodynamic change subject to native vertical force
    # and independent column mass / theta-mass budgets. Schur complement of
    # a tridiagonal force block and two integral constraints.
    rho, heat, tracer, acceleration = (
        np.asarray(x, dtype=np.float64) for x in (rho, heat, tracer, acceleration)
    )
    if (
        rho.ndim != 2
        or rho.shape[0] < 2
        or heat.shape != rho.shape
        or tracer.shape != rho.shape
    ):
        raise ValueError(
            "moving balance needs matching mass-grid fields and at least two layers"
        )
    if acceleration.shape != (rho.shape[0] + 1, rho.shape[1]):
        raise ValueError("moving balance acceleration must be on interfaces")
    if any(not np.isfinite(x).all() for x in (rho, heat, tracer, acceleration)):
        raise ValueError("moving balance received a non-finite field")
    if np.any(rho <= 0) or np.any(heat <= 0) or np.any(tracer < 0):
        raise ValueError(
            "moving balance needs positive thermodynamics and nonnegative water mass; "
            f"minimum rho={rho.min():.17g}, heat={heat.min():.17g}, water={tracer.min():.17g}"
        )
    if backend not in ('cpu','cuda'):
        raise ValueError('moving balance backend must be cpu or cuda')
    if backend == 'cuda' and os.environ.get('WOOF_HEX_MOVING_NEWTON_GPU','1') != '0':
        from .moving_balance_cuda import project_columns as project_cuda
        return project_cuda(rho,heat,tracer,acceleration,v,ref)
    r, h = rho.copy(), heat.copy()
    dz = np.asarray(v.dzw, dtype=float)
    zz = np.asarray(v.zz, dtype=float)
    rb = np.asarray(ref.rho_base, dtype=float)
    pb = (
        zz
        * 287
        * np.asarray(ref.rho_theta_base, dtype=float)
        * np.asarray(ref.exner_base, dtype=float)
    )
    a = np.asarray(v.rdzu, dtype=float)[1:]
    wm = np.asarray(v.fzm, dtype=float)[1:] * 9.80616
    wp = np.asarray(v.fzp, dtype=float)[1:] * 9.80616
    target_r = dz @ rho
    target_h = dz @ heat
    for it in range(8):
        p = 1e5 * (287 * zz * h / 1e5) ** 1.4
        pp = p - pb
        load = r + tracer - rb
        f = (
            a[:, None] * np.diff(pp, axis=0)
            + wm[:, None] * load[1:]
            + wp[:, None] * load[:-1]
        )
        fm = wm[:, None] / 9.80616
        fp = wp[:, None] / 9.80616
        metric = fm * zz[1:] + fp * zz[:-1]
        ri = fm * r[1:] + fp * r[:-1]
        loading = 1.0 + 0.5 * (tracer[1:] / r[1:] + tracer[:-1] / r[:-1])
        inertia = ri * metric * loading
        f -= acceleration[1:-1] * inertia
        # Force Jacobian coefficients on rho and heat below/above interface.
        rl = wp[:, None] - acceleration[1:-1] * metric * (
            fp * loading - 0.5 * ri * tracer[:-1] / r[:-1] ** 2
        )
        ru = wm[:, None] - acceleration[1:-1] * metric * (
            fm * loading - 0.5 * ri * tracer[1:] / r[1:] ** 2
        )
        deriv = 1.4 * p / h
        hl = -a[:, None] * deriv[:-1]
        hu = a[:, None] * deriv[1:]
        vr = rho * rho
        vh = heat * heat
        diag = (
            rl * rl * vr[:-1] + ru * ru * vr[1:] + hl * hl * vh[:-1] + hu * hu * vh[1:]
        )
        off = ru[:-1] * rl[1:] * vr[1:-1] + hu[:-1] * hl[1:] * vh[1:-1]
        # Normalize the budget equations (no need to use physical area here).
        cr = dz[:, None] / target_r
        ch = dz[:, None] / target_h
        br = rl * vr[:-1] * cr[:-1] + ru * vr[1:] * cr[1:]
        bh = hl * vh[:-1] * ch[:-1] + hu * vh[1:] * ch[1:]
        b = np.stack((f, br, bh), axis=-1)
        d = diag.copy()
        b = b.copy()
        for k in range(1, len(d)):
            mult = off[k - 1] / d[k - 1]
            d[k] -= mult * off[k - 1]
            b[k] -= mult[:, None] * b[k - 1]
        b[-1] /= d[-1, :, None]
        for k in range(len(d) - 2, -1, -1):
            b[k] = (b[k] - off[k, :, None] * b[k + 1]) / d[k, :, None]
        x, ur, uh = b[..., 0], b[..., 1], b[..., 2]
        srr = np.sum(vr * cr * cr, axis=0) - np.sum(br * ur, axis=0)
        shh = np.sum(vh * ch * ch, axis=0) - np.sum(bh * uh, axis=0)
        srh = -np.sum(br * uh, axis=0)
        er = (dz @ r - target_r) / target_r - np.sum(br * x, axis=0)
        eh = (dz @ h - target_h) / target_h - np.sum(bh * x, axis=0)
        det = srr * shh - srh * srh
        lr = (er * shh - eh * srh) / det
        lh = (eh * srr - er * srh) / det
        lf = x - ur * lr - uh * lh
        dr = vr * cr * lr
        dh = vh * ch * lh
        dr[:-1] += vr[:-1] * rl * lf
        dr[1:] += vr[1:] * ru * lf
        dh[:-1] += vh[:-1] * hl * lf
        dh[1:] += vh[1:] * hu * lf
        r -= dr
        h -= dh
        if (
            not np.isfinite(r).all()
            or not np.isfinite(h).all()
            or np.min(r) <= 0
            or np.min(h) <= 0
        ):
            raise ValueError("moving balance produced non-positive thermodynamics")
        if abs(f).max() < 1e-10:
            break
    if abs(f).max() > 1e-8:
        raise ValueError("moving column balance did not converge")
    return (
        r,
        h,
        dict(
            iterations=it + 1,
            residual=float(abs(f).max()),
            mass=float(np.max(abs(dz @ r - target_r) / target_r)),
            heat=float(np.max(abs(dz @ h - target_h) / target_h)),
            rho_change=float(abs(r / rho - 1).max()),
            heat_change=float(abs(h / heat - 1).max()),
        ),
    )


def source_pressure_acceleration(old, vertical, moist):
    """Native vertical pressure/buoyancy residual divided by w inertia.

    The residual sign is the negative of the w tendency. This sign is kept
    on both sides of the constraint. Physical w also includes a terrain/u
    term; only the vertical pressure contribution is transferred here.
    """
    r = old["rho"].astype(np.float64)
    rr = old["density_perturbation"].astype(np.float64)
    pp = old["pressure_perturbation"].astype(np.float64)
    q = sum((old["scalars"][i].astype(np.float64) for i in moist), np.zeros_like(r))
    dp = -9.80616 * (rr + r * q)
    fm = np.asarray(vertical.fzm, dtype=float)[1:, None]
    fp = np.asarray(vertical.fzp, dtype=float)[1:, None]
    zz = np.asarray(vertical.zz, dtype=float)
    inertia = (
        (fm * r[1:] + fp * r[:-1])
        * (fm * zz[1:] + fp * zz[:-1])
        * (1.0 + 0.5 * (q[1:] + q[:-1]))
    )
    result = np.zeros((r.shape[0] + 1, r.shape[1]))
    result[1:-1] = (
        np.asarray(vertical.rdzu, dtype=float)[1:, None] * np.diff(pp, axis=0)
        - fm * dp[1:]
        - fp * dp[:-1]
    ) / inertia
    return result


def _sparse_bracket_rounding(out, exact, measure, mask, target):
    """Solve a sparse failed rounding budget within the original brackets.

    Prefix rounding can miss an admissible combination when a new tracer
    occupies only a few cells. Meet in the middle bounds this fallback to
    two arrays of at most 65536 subset sums; dense fields keep the fast path.
    """
    ids = np.flatnonzero(mask)
    if not 0 < len(ids) <= 32 or np.any(measure.ravel()[ids] < 0):
        return out
    nearest = exact.ravel()[ids].astype(np.float32)
    lower = np.where(nearest > exact.ravel()[ids], np.nextafter(nearest, np.float32(-np.inf)), nearest)
    upper = np.where(nearest < exact.ravel()[ids], np.nextafter(nearest, np.float32(np.inf)), nearest)
    candidate = out.copy()
    candidate.ravel()[ids] = lower
    remaining = target - float(np.sum(candidate.astype(float) * measure))
    quanta = (upper.astype(float)-lower.astype(float))*measure.ravel()[ids]
    split = len(ids)//2

    def sums(values):
        result = np.zeros(1)
        for value in values:
            result = np.concatenate((result, result+value))
        return result

    left, right = sums(quanta[:split]), sums(quanta[split:])
    order = np.argsort(right, kind='stable')
    sorted_right = right[order]
    positions = np.searchsorted(sorted_right, remaining-left)
    best = (float('inf'), 0, 0)
    for offset in (-1, 0):
        selected = np.clip(positions+offset, 0, len(right)-1)
        errors = abs(left+sorted_right[selected]-remaining)
        i = int(np.argmin(errors))
        if errors[i] < best[0]:
            best = (float(errors[i]), i, int(order[selected[i]]))
    bits = best[1] | (best[2] << split)
    for i, index in enumerate(ids):
        if bits & (1 << i):
            candidate.ravel()[index] = upper[i]
    return candidate


def _ordered_bracket_rounding(exact, measure, mask, target, bound):
    """Close a budget the index-order prefix missed, largest admissible quantum first.

    Each eligible value stays at its nearest float or moves once to the other
    float bracketing it, as in the prefix path. The prefix walks candidates in
    index order, so one heavy quantum (a coarse background cell carrying a
    sparse hydrometeor) can be the step that overshoots, leaving a residual no
    later candidate in that direction can reduce. Taking the largest quantum
    that still reduces the residual first, then reversing direction when the
    residual changes sign, closes such budgets with the small quanta. The cast
    error is never larger than the quanta in its own direction, so a solution
    within one float per value exists whenever the quanta are fine enough.
    Breakage it prevents: storm-following run_0bd003fbf0fc2910 (2026-10-26,
    hex 0.4.6, MPAS physics) refused its first move with a residual of twice
    the tolerance and 1061 eligible cells.
    """
    nearest = exact.astype(np.float32)
    out = nearest.copy()
    flat, ex, m = out.ravel(), exact.ravel(), measure.ravel()
    ids = np.flatnonzero(mask)
    if not len(ids) or np.any(m[ids] < 0):
        return out
    near = nearest.ravel()[ids]
    other = np.where(near.astype(float) > ex[ids], np.nextafter(near, np.float32(-np.inf)),
                     np.nextafter(near, np.float32(np.inf)))
    delta = (other.astype(float) - near.astype(float)) * m[ids]
    error = float(target - np.sum(out.astype(float) * measure))
    moved = np.zeros(len(ids), dtype=bool)
    for _ in range(64):
        if abs(error) <= bound:
            break
        cand = np.flatnonzero(~moved & (np.sign(delta) == np.sign(error)) & (delta != 0))
        if not len(cand):
            break
        size = abs(delta[cand])
        order = cand[np.lexsort((ids[cand], -size))]
        size = abs(delta[order])
        sign, j, took = np.sign(error), 0, False
        while j < len(order) and abs(error) > bound and np.sign(error) == sign:
            # quanta at least twice the residual cannot reduce it; skip to the first that can
            j = max(j, int(np.searchsorted(-size, -2.0 * abs(error), side="right")))
            if j >= len(order):
                break
            k = order[j]
            flat[ids[k]] = other[k]
            error -= float(delta[k])
            moved[k] = took = True
            j += 1
        if not took:
            break
    return out


def conservative_float32(exact, measure, target, *, eligible=None, rtol=5e-10):
    """Round within one ULP while closing the integrated product budget.

    Only values that were inexact in the ordinary FP32 cast are eligible.
    Thus identity carries and exact zero tracers stay bitwise untouched.
    A deterministic prefix chooses adjacent floats; a small greedy tail
    resolves the residual. Refuse if FP32 cannot represent the requested
    tolerance instead of treating the observed error as an allowance.
    """
    exact = np.asarray(exact, dtype=float)
    measure = np.broadcast_to(measure, exact.shape)
    out = exact.astype(np.float32)
    total = float(np.sum(out.astype(float) * measure))
    error = float(target - total)
    tolerance = rtol * max(abs(target), float(np.sum(abs(exact) * measure)), 1e-300)
    if abs(error) <= tolerance:
        # The ordinary cast already closes the budget. The later verification
        # evaluates the same sum on the unchanged array, so neither a giant
        # index list nor the second scan can change its verdict or its words.
        return out
    mask = out.astype(float) != exact
    if eligible is not None:
        mask &= np.broadcast_to(eligible, exact.shape)
    ids = np.flatnonzero(mask)
    if abs(error) > tolerance and len(ids):
        flat = out.ravel()
        next_values = np.nextafter(
            flat[ids], np.float32(np.inf if error > 0 else -np.inf)
        )
        quanta = (
            next_values.astype(float) - flat[ids].astype(float)
        ) * measure.ravel()[ids]
        # Only rounding toward the other bracketing float is admissible.
        valid = (next_values.astype(float) - exact.ravel()[ids]) * (
            flat[ids].astype(float) - exact.ravel()[ids]
        ) <= 0
        ids, next_values, quanta = ids[valid], next_values[valid], quanta[valid]
        prefix = np.cumsum(quanta)
        stop = int(np.argmin(abs(np.concatenate(([0.0], prefix)) - error)))
        flat[ids[:stop]] = next_values[:stop]
        error -= float(prefix[stop - 1]) if stop else 0.0
        # Resolve only the unconsumed tail. Each selected value moves once.
        for j in range(stop, len(ids)):
            if abs(error) <= tolerance:
                break
            if abs(error - quanta[j]) < abs(error):
                flat[ids[j]] = next_values[j]
                error -= float(quanta[j])
    actual = float(np.sum(out.astype(float) * measure))
    if abs(actual - target) > max(tolerance, 128 * np.finfo(float).eps * abs(target)):
        out = _sparse_bracket_rounding(out, exact, measure, mask, target)
        actual = float(np.sum(out.astype(float) * measure))
    if abs(actual - target) > max(tolerance, 128 * np.finfo(float).eps * abs(target)):
        ordered = _ordered_bracket_rounding(exact, measure, mask, target, tolerance)
        ordered_actual = float(np.sum(ordered.astype(float) * measure))
        if abs(ordered_actual - target) < abs(actual - target):
            out, actual = ordered, ordered_actual
    if abs(actual - target) > max(tolerance, 128 * np.finfo(float).eps * abs(target)):
        raise ValueError("FP32 moving integral cannot meet its conservation tolerance; "
                         f"target={target:.17g}, actual={actual:.17g}, tolerance={tolerance:.17g}, "
                         f"nonzero={np.count_nonzero(exact)}, eligible={len(ids)}, "
                         f"range=[{exact.min():.17g}, {exact.max():.17g}]")
    return out


def extend_below_ground(weights, old, moist):
    """Hand an absolute-height map the donors' lowest theta, exner and water loading."""
    extend = getattr(weights, "extend_below_ground", None)
    if extend is None or "exner" not in old:
        return
    # Only the lowest layer enters extend_below_ground. Do not materialize
    # all 55 levels of every water species merely to discard levels 1:.
    rho = old["rho"][:1].astype(np.float64)
    loading = 1.0 + sum((old["scalars"][i][:1].astype(np.float64) for i in moist), np.zeros_like(rho))
    extend(old["rho_theta"][:1].astype(np.float64) / rho, old["exner"][:1], loading)


def _mapped(weights, values, density):
    """``(product map, domain reconciliation, positivity roundoff)`` of one mass product.

    A map that can (HeightConservativeMap.mass_intensive_receipted) keeps both receipts out of its
    shared ``last_*`` attributes, so concurrent products cannot read each other's receipts.
    """
    receipted = getattr(weights, "mass_intensive_receipted", None)
    # An instance that replaces its own mass_intensive is honoured (that call, then its receipts).
    if receipted is not None and "mass_intensive" not in getattr(weights, "__dict__", {}):
        return receipted(values, density=density)
    result = weights.mass_intensive(values, density=density)
    return (result, getattr(weights, "last_domain_reconciliation", None),
            getattr(weights, "last_positivity_roundoff", None))


def remap_balanced_thermodynamics(payload, weights, prepared, *, extended=False):
    """Map products, initialize the carried force, then round conservatively.

    ``extended``: the caller already gave the map this payload's below-ground extension.

    THE COST THIS REMOVES (box-aws2x 2026-10-24, resumed storm-following segment, sampled): this
    function ran every product map, both evaporations, the column projection and the tracer casts one
    after another on one thread, 18 s of the remap.  The independent pieces now run concurrently
    (host_parallel.run_tasks / row_map); each is the same call on the same inputs and every sum keeps
    its order, so the carried words are unchanged.
    """
    from types import SimpleNamespace
    from .host_parallel import run_tasks

    old, v, ref = payload["state"], prepared.vertical, prepared.reference
    moist = [
        i
        for i, name in enumerate(payload["scalar_names"])
        if name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
    ]
    import time
    clock = [time.perf_counter()]
    stage_seconds = {}

    def mark(name):
        # Wall seconds of each stage for the receipt (no device sync, no copy; the arithmetic is untouched).
        now = time.perf_counter()
        stage_seconds[name] = now - clock[0]
        clock[0] = now
    if not extended:
        extend_below_ground(weights, old, moist)
    density = old["rho"]
    # The first product runs alone: it builds the map's lazily cached device geometry once.
    rho, mass_domain, positivity = _mapped(weights, old["rho"], density)
    # Vapour and cloud water are adjusted to the target's saturation before the balance (saturation_adjustment);
    # their products are kept for the tracer cast below.
    names = list(payload["scalar_names"])
    adjusted, adjusted_domains, adjustment = {}, {}, None
    declared = dict(heat=0.0)
    saturating = all(name in names for name in SATURATION_SPECIES)
    iv, ic = (names.index(name) for name in SATURATION_SPECIES) if saturating else (None, None)

    def product(i):
        return lambda: _mapped(weights, old["rho"].astype(float) * old["scalars"][i], density)

    def donor_evaporation():
        donor, _, _ = evaporation(old["rho"], old["rho_theta"], payload["vertical"].zz,
                                  old["scalars"][iv], old["scalars"][ic], backend=getattr(weights, 'backend', 'cpu'))
        return _mapped(weights, old["rho"].astype(float) * donor, density)

    # Tasks listed in the sequential program order of the products (that order fixes which positivity
    # receipt a carried tracer reports below).
    plain = [i for i in moist if not (saturating and i in (iv, ic))]
    tasks = [lambda: _mapped(weights, old["rho_theta"], density)]
    if saturating:
        tasks += [product(iv), product(ic), donor_evaporation]
    tasks += [product(i) for i in plain]
    tasks.append(lambda: weights.interface_intensive(source_pressure_acceleration(old, payload["vertical"], moist)))
    mark('density')
    results = run_tasks(tasks)
    mark('products')
    acceleration = results.pop()
    (heat, heat_domain, positivity), results = results[0], results[1:]
    if saturating:
        adjusted[iv], adjusted_domains[iv], _ = results[0]
        adjusted[ic], adjusted_domains[ic], _ = results[1]
        donor, _, positivity = results[2]
        results = results[3:]
        adjusted[iv], adjusted[ic], heat, adjustment = saturation_adjustment(
            adjusted[iv], adjusted[ic], rho, heat, v.zz, donor,
            lambda value: float(weights.totals(value, source=False)[0]), backend=getattr(weights, 'backend', 'cpu'))
        declared = dict(heat=adjustment["theta_mass_change_kg_kelvin"],
                        **{names[iv]: adjustment["cloud_to_vapor_kg"], names[ic]: -adjustment["cloud_to_vapor_kg"]})
    plain_products = {}
    for i, result in zip(plain, results):
        plain_products[i] = result
        positivity = result[2]
    # The total water mass alone enters the balance equations, summed in moist order.  The plain tracers'
    # products are kept for their cast below when they fit a bounded host budget (8 GB); otherwise each is
    # remapped again there, as before.  Either way the cast reads the words of the same call on the same inputs
    # (the map's below-ground extension is set once, above, and the integral has no atomics), so the carried
    # words are unchanged.  THE COST THIS REMOVES: the second height remap of every plain moist tracer (five
    # of the 19 product maps of a convection_permitting move).
    keep_products = (len(plain) * rho.size * 8 <= 8 * 2**30
                     and os.environ.get("WOOF_HEX_MOVING_KEEP_PRODUCTS", "1") != "0")
    water = np.zeros_like(rho)
    for i in moist:
        water += adjusted[i] if i in adjusted else plain_products[i][0]
    retained = plain_products if keep_products else {}
    del plain_products
    acceleration[[0, -1]] = 0.0
    exact_rho, exact_heat = rho.copy(), heat.copy()

    # Each 4096-column block is its own projection (its own Newton iterations), as before; the blocks
    # run concurrently and write disjoint columns.
    def project(sl):
        vs = SimpleNamespace(
            dzw=v.dzw, zz=v.zz[:, sl], rdzu=v.rdzu, fzm=v.fzm, fzp=v.fzp
        )
        rs = SimpleNamespace(
            rho_base=ref.rho_base[:, sl],
            rho_theta_base=ref.rho_theta_base[:, sl],
            exner_base=ref.exner_base[:, sl],
        )
        exact_rho[:, sl], exact_heat[:, sl], record = project_columns(
            rho[:, sl], heat[:, sl], water[:, sl], acceleration[:, sl], vs, rs,
            backend=getattr(weights,'backend','cpu')
        )
        return record
    blocks = [slice(start, min(start + 4096, rho.shape[1])) for start in range(0, rho.shape[1], 4096)]
    mark('saturation_and_water')
    records = run_tasks([lambda sl=sl: project(sl) for sl in blocks])
    mark('column_projection')
    # Do not project an exactly unchanged donor column: its saved native
    # diagnostics, including FP32 evaluation residue, already own its force.
    counts = np.bincount(weights.rows, minlength=len(weights.target_area))
    donors = np.full(len(counts), -1, dtype=int)
    donors[weights.rows] = weights.cols
    unchanged = counts == 1
    cols = np.flatnonzero(unchanged)
    unchanged[cols] &= weights.target_area[cols] == weights.source_area[donors[cols]]
    cols = np.flatnonzero(unchanged)
    unchanged[cols] &= np.all(
        weights.target_zgrid[:, cols] == weights.source_zgrid[:, donors[cols]], axis=0
    )
    cols = np.flatnonzero(unchanged)
    # Cumulative integration can add FP64 cancellation even on an identity
    # stencil. Admit only arithmetic noise, never a finite domain correction.
    tol = 128 * np.finfo(float).eps
    unchanged[cols] &= np.all(
        abs(rho[:, cols] / old["rho"][:, donors[cols]] - 1.0) <= tol, axis=0
    ) & np.all(
        abs(heat[:, cols] / old["rho_theta"][:, donors[cols]] - 1.0) <= tol, axis=0
    )
    cols = np.flatnonzero(unchanged)
    donor = donors[cols]
    old_water = np.zeros((rho.shape[0], len(cols)))
    for i in moist:
        old_water += old["rho"][:, donor].astype(float) * old["scalars"][i][:, donor]
    unchanged[cols] &= np.all(
        abs(water[:, cols] - old_water) <= tol * abs(old_water), axis=0
    )
    # Geometry identity alone does not imply reference-state identity. Check
    # the saved native pressure split before copying any original column.
    cols = np.flatnonzero(unchanged)
    donor = donors[cols]
    old_h = old["rho_theta"][:, donor]
    ex = old["exner"][:, donor]
    hb = ref.rho_theta_base[:, cols]
    pp = (
        v.zz[:, cols]
        * np.float32(287.0)
        * (ex * (old_h - hb) + hb * (ex - ref.exner_base[:, cols]))
    )
    unchanged[cols] &= np.all(
        pp == old["pressure_perturbation"][:, donor], axis=0
    ) & np.all(
        old["rho"][:, donor] - ref.rho_base[:, cols]
        == old["density_perturbation"][:, donor],
        axis=0,
    )
    exact_rho[:, unchanged] = old["rho"][:, donors[unchanged]]
    exact_heat[:, unchanged] = old["rho_theta"][:, donors[unchanged]]
    measure = weights.target_dzw[:, None] * weights.target_area

    def cast_product(source, exact, domain, change=0.0):
        # ``change``: the declared saturation-adjustment change of this total (zero for dry mass).
        before = float(weights.totals(source, source=True)[0])
        error = float(weights.totals(exact, source=False)[0] - before - change)
        allowance = (
            256
            * np.finfo(float).eps
            * float(weights.totals(abs(source), source=True)[0])
        )
        if abs(error) > allowance:
            raise ValueError(
                "balanced moving integral exceeds FP64 arithmetic allowance"
            )
        actual = conservative_float32(
            exact, measure, before + change, eligible=~unchanged[None, :]
        )
        after = float(weights.totals(actual, source=False)[0])
        receipt = dict(
            before=[before],
            after=[after],
            absolute_error=[after - before - change],
            saturation_adjustment=[change],
            rounding_envelope=[
                float(np.sum(abs(actual.astype(float) - exact) * measure))
            ],
            fp64_error=[error],
            terrain_domain_reconciliation=domain,
            accounting="whole-column global totals; constrained balance and conservative FP32 rounding",
        )
        return actual, receipt

    mark('unchanged_columns')
    (rho, mass), (heat, heat_receipt) = run_tasks([
        lambda: cast_product(old["rho"], exact_rho, mass_domain),
        lambda: cast_product(old["rho_theta"], exact_heat, heat_domain, declared["heat"])])
    mark('cast_mass_and_heat')
    tracers = {}
    scalars = np.empty((len(payload["scalar_names"]), *rho.shape), dtype=np.float32)

    def cast_tracer(i, name):
        source = old["rho"].astype(float) * old["scalars"][i]
        if i in adjusted:
            exact, domain = adjusted[i], adjusted_domains[i]
            own_positivity, mapped = None, False
        else:
            kept = retained.pop(i, None)
            exact, domain, own_positivity = kept if kept is not None else _mapped(weights, source, density)
            mapped = True
        change = declared.get(name, 0.0)
        before = float(weights.totals(source, source=True)[0])
        fp64 = float(weights.totals(exact, source=False)[0] - before - change)
        if abs(fp64) > 256 * np.finfo(float).eps * float(
            weights.totals(abs(source), source=True)[0]
        ):
            raise ValueError(
                "balanced tracer integral exceeds FP64 arithmetic allowance"
            )
        # A tracer can have a nonzero terrain-domain correction even when
        # dry mass and theta mass do not. Its copy mask must be independent.
        copy_tracer = unchanged.copy()
        cols = np.flatnonzero(copy_tracer)
        donor_product = source[:, donors[cols]]
        copy_tracer[cols] &= np.all(
            abs(exact[:, cols] - donor_product) <= tol * abs(donor_product), axis=0
        )
        mixing = exact / rho
        mixing[:, copy_tracer] = old["scalars"][i][:, donors[copy_tracer]]
        scalars[i] = conservative_float32(
            mixing, measure * rho, before + change, eligible=~copy_tracer[None, :]
        )
        actual = rho.astype(float) * scalars[i]
        after = float(weights.totals(actual, source=False)[0])
        return mapped, own_positivity, dict(
            before=[before],
            after=[after],
            absolute_error=[after - before - change],
            saturation_adjustment=[change],
            rounding_envelope=[float(np.sum(abs(actual - exact) * measure))],
            fp64_error=[fp64],
            terrain_domain_reconciliation=domain,
            positivity_roundoff=None,
            accounting=mass["accounting"],
        )

    # About eight FP64 level-by-cell temporaries live per tracer task; bound the concurrent ones to 32 GB.
    budget = max(1, int(32 * 2**30 // max(1, 64 * rho.size)))
    cast = run_tasks([lambda i=i, name=name: cast_tracer(i, name)
                      for i, name in enumerate(payload["scalar_names"])], workers=budget)
    # A carried (saturation-adjusted) tracer maps nothing, so it reports the positivity receipt of the
    # product mapped just before it in program order, as the sequential loop did.
    for name, (mapped, own_positivity, receipt) in zip(payload["scalar_names"], cast):
        if mapped:
            positivity = own_positivity
        receipt["positivity_roundoff"] = positivity
        tracers[name] = receipt
    adjusted.clear()
    mark('cast_tracers')
    record = {key: max(row[key] for row in records) for key in records[0]}
    record.update(
        method="constrained native vertical pressure acceleration; column mass, theta mass and water masses retained",
        saturation_adjustment=adjustment,
        unchanged_columns=int(unchanged.sum()),
        horizontal_balance_constrained=False,
        damping_added=False,
        stage_seconds=stage_seconds,
    )
    return rho, heat, scalars, mass, heat_receipt, tracers, record
