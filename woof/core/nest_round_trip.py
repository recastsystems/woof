"""Strict mode: the couple/uncouple round trip WRF's nest force leaves on both domains.

``med_force_domain`` (share/mediation_force_domain.F:111-206) couples the
parent and the nest in place (``couple_or_uncouple_em(.true.)``: x*mass),
interpolates the nest's boundary tables from the coupled fields, and then
uncouples both (``couple_or_uncouple_em(.false.)``: x*(1./mass)).  The
mass does not move between the two calls, so each prognostic word comes
back as ``(x*m)*(1/m)``, which in FP32 is not always x.  WRF does this to
both domains at every parent step.  woof's coupler forms the coupled fields
without writing the state back (woof.core.nest.NestCoupler.force), so
strict mode applies the round trip itself, after the tables are built.

A nest's history frame at an instant it shares with its parent is written
by WRF AFTER the next force (the nest's integrate call follows
med_nest_force), so its words carry that force's round trip, while the
force's boundary tendencies read the words before it.  The coupler cannot
apply it early, so :func:`round_tripped_frame_fields` gives the history
writer round-tripped copies instead (``_wrf_frame_round_trip_due``, set by
the executor's history callback).

Every operation below is one CuPy operation, so it rounds once, in
dyn_em/couple_or_uncouple_em.F's order: half-level mass
``(c1h*mub+c2h)+(c1h*mu)`` for t and every moist/scalar species,
full-level ``(c1f*mub+c2f)+(c1f*mu)`` for ph and (over msfty) w, and the
face averages ``0.5*(...)/msfuy`` and ``2*msfuy/(...)`` for u (msfvx for
v).  On a specified or nested domain the east and north faces take the
one-sided mass and the west and south faces read the halo column, which
holds the edge mass (set_physical_bc2d's zero-gradient copy) as the last
acoustic step left it: woof.core.dycore.WRF_MU_HALO_SLOT once the domain
has stepped, the edge column itself before that.
"""

from __future__ import annotations

from woof.core.nest_fields import nest_field_kinds

#: Levels per launch group, so the transient mass planes stay a slab, not a
#: second copy of the domain.
_LEVEL_BLOCK = 16


def _attr(kind: str) -> str:
    return {"t": "thp", "ph": "php"}.get(kind, kind)


def round_trip_targets(state, run) -> dict:
    """``{state attribute: array}`` for every word the round trip touches."""
    kinds = [k for k in nest_field_kinds(run) if k != "mu"]
    out = {}
    for kind in kinds:
        value = getattr(state, _attr(kind), None)
        if value is not None:
            out[_attr(kind)] = value
    return out


def couple_round_trip(state, run, targets=None) -> None:
    """Apply WRF's force-time couple/uncouple round trip, in place.

    ``targets`` maps state attribute names to the arrays to transform;
    by default the state's own (the force), or copies (a history frame).
    """
    import cupy as cp

    if not (getattr(run, "specified", False) or getattr(run, "nested", False)):
        raise ValueError("the nest force round trip is only defined on a "
                         "specified or nested domain")
    from woof.core.dycore import WRF_MU_HALO_SLOT

    if targets is None:
        targets = round_trip_targets(state, run)
    f32 = cp.float32
    mub, mu = state.mub2d, state.mup
    halo = state.existing_scratch(WRF_MU_HALO_SLOT)
    if halo is None:
        halo = mu
    has_msf = bool(getattr(state, "has_msf", True))
    msft = state.msft if has_msf else None
    msfu = state.msfu if has_msf else None
    msfv = state.msfv if has_msf else None
    half = [a for a in targets if a not in ("u", "v", "w", "php")]
    nz = int(state.thp.shape[0])

    for k0 in range(0, nz + 1, _LEVEL_BLOCK):
        k1 = min(nz + 1, k0 + _LEVEL_BLOCK)
        c1f = state.c1f[k0:k1, None, None]
        c2f = state.c2f[k0:k1, None, None]
        mf = ((c1f * mub[None]) + c2f) + (c1f * mu[None])
        if "php" in targets:
            ph = targets["php"][k0:k1]
            ph *= mf
            ph *= f32(1.0) / mf
        if "w" in targets:
            w = targets["w"][k0:k1]
            if has_msf:
                w *= mf / msft[None]
                w *= msft[None] / mf
            else:
                w *= mf
                w *= f32(1.0) / mf
        del mf

    for k0 in range(0, nz, _LEVEL_BLOCK):
        k1 = min(nz, k0 + _LEVEL_BLOCK)
        c1h = state.c1h[k0:k1, None, None]
        c2h = state.c2h[k0:k1, None, None]
        a = (c1h * mub[None]) + c2h
        b = c1h * mu[None]
        m = a + b
        minv = f32(1.0) / m
        for name in half:
            x = targets[name][k0:k1]
            x *= m
            x *= minv
        del m, minv
        if "u" in targets:
            # Face i (1-based) reads columns i and i-1; face 1 reads the halo.
            a_h = (c1h * mub[None, :, :1]) + c2h
            b_h = c1h * halo[None, :, :1]
            s = ((a[:, :, 1:] + a[:, :, :-1]) + b[:, :, 1:]) + b[:, :, :-1]
            s0 = ((a[:, :, :1] + a_h) + b[:, :, :1]) + b_h
            u = targets["u"][k0:k1]
            inner, west, east = u[:, :, 1:-1], u[:, :, :1], u[:, :, -1:]
            se = a[:, :, -1:] + b[:, :, -1:]
            if has_msf:
                mi, m0, me = msfu[None, :, 1:-1], msfu[None, :, :1], msfu[None, :, -1:]
                inner *= (f32(0.5) * s) / mi
                inner *= (f32(2.0) * mi) / s
                west *= (f32(0.5) * s0) / m0
                west *= (f32(2.0) * m0) / s0
                east *= se / me
                east *= me / se
            else:
                inner *= f32(0.5) * s
                inner *= f32(2.0) / s
                west *= f32(0.5) * s0
                west *= f32(2.0) / s0
                east *= se
                east *= f32(1.0) / se
            del s, s0, se, a_h, b_h
        if "v" in targets:
            a_h = (c1h * mub[None, :1, :]) + c2h
            b_h = c1h * halo[None, :1, :]
            s = ((a[:, 1:, :] + a[:, :-1, :]) + b[:, 1:, :]) + b[:, :-1, :]
            s0 = ((a[:, :1, :] + a_h) + b[:, :1, :]) + b_h
            v = targets["v"][k0:k1]
            inner, south, north = v[:, 1:-1, :], v[:, :1, :], v[:, -1:, :]
            sn = a[:, -1:, :] + b[:, -1:, :]
            if has_msf:
                mi, m0, mn = msfv[None, 1:-1, :], msfv[None, :1, :], msfv[None, -1:, :]
                inner *= (f32(0.5) * s) / mi
                inner *= (f32(2.0) * mi) / s
                south *= (f32(0.5) * s0) / m0
                south *= (f32(2.0) * m0) / s0
                north *= sn / mn
                north *= mn / sn
            else:
                inner *= f32(0.5) * s
                inner *= f32(2.0) / s
                south *= f32(0.5) * s0
                south *= f32(2.0) / s0
                north *= sn
                north *= f32(1.0) / sn
            del s, s0, sn, a_h, b_h
        del a, b


def round_tripped_frame_fields(state, fields: dict) -> dict:
    """History rows of a nest frame that WRF writes after the next force.

    Every row that IS one of the round trip's state arrays is replaced by a
    round-tripped copy; THM is t_2 and takes it, T is th_phy_m_t0 (set at
    the end of the nest's solve) and keeps the state's word.
    """
    import cupy as cp

    run = state._wrf_round_trip_run
    targets = round_trip_targets(state, run)
    copies = {name: cp.array(value, copy=True) for name, value in targets.items()}
    couple_round_trip(state, run, copies)
    by_id = {id(value): copies[name] for name, value in targets.items()}
    out = dict(fields)
    for name, value in fields.items():
        if name == "T":
            continue
        replacement = by_id.get(id(value))
        if replacement is not None:
            out[name] = replacement
    return out


def couple_halo_faces(state, kind: str, out) -> None:
    """Strict mode: a coupled u/v field's west/south face as WRF forms it.

    couple_or_uncouple_em's generic face loop runs over the west (south)
    physical face and reads mu_2 at ids-1 (jds-1): the halo, which holds the
    edge mass as the last acoustic step left it, before spec_bdy_final
    installed the boundary value (woof.core.dycore.WRF_MU_HALO_SLOT).  The
    coupling kernel duplicates the edge column there; the two are usually
    the same word, and when they are not, the nest's boundary tendency
    parts from wrf.exe at that face (the UAE pair's nest, one V word at
    60 s, the first force after a two-way feedback).  ``out`` is the
    coupled field the force just wrote; only its face 0 is rewritten.
    """
    if kind not in ("u", "v"):
        return
    from woof.core.dycore import WRF_MU_HALO_SLOT

    halo = state.existing_scratch(WRF_MU_HALO_SLOT)
    if halo is None:
        return
    f32 = __import__("cupy").float32
    c1h = state.c1h[:, None]
    c2h = state.c2h[:, None]
    has_msf = bool(getattr(state, "has_msf", True))
    if kind == "u":
        mub, mu, mu_h = state.mub2d[None, :, 0], state.mup[None, :, 0], halo[None, :, 0]
        field = state.u[:, :, 0]
        target = out[:, :, 0]
        msf = state.msfu[None, :, 0] if has_msf else None
    else:
        mub, mu, mu_h = state.mub2d[None, 0, :], state.mup[None, 0, :], halo[None, 0, :]
        field = state.v[:, 0, :]
        target = out[:, 0, :]
        msf = state.msfv[None, 0, :] if has_msf else None
    base = (c1h * mub) + c2h
    s = ((base + base) + (c1h * mu)) + (c1h * mu_h)
    ch = f32(0.5) * s
    if msf is not None:
        ch = ch / msf
    target[...] = ch * field


def _keeps_th_phy_m_t0(node) -> bool:
    """Strict mode: does this domain's history T part from its t_2?

    WRF's history T is th_phy_m_t0, which update_phys_fields sets to t_2 at
    the very end of every solve (after_all_rk_steps -> diagnostics_driver,
    phys/module_diagnostics_driver.F:188/1265), whatever the physics.  A
    parent's t_2 then moves again before its next frame -- the force round
    trip and its nests' feedback -- so its T is a copy taken there; every
    other domain's T is its t_2.
    """
    from woof.wrf_exact import ENABLED as wrf_exact

    state = node.state
    if not wrf_exact or not node.children or getattr(state, "thp", None) is None:
        return False
    return getattr(state, "_streamed_domain", None) is None


def th_phy_m_t0_after_step(node) -> None:
    """update_phys_fields: th_phy_m_t0 = t_2 at the end of the solve."""
    if not _keeps_th_phy_m_t0(node):
        return
    import cupy as cp

    state = node.state
    out = getattr(state, "_wrf_th_phy_m_t0", None)
    if out is None or out.shape != state.thp.shape:
        out = state._wrf_th_phy_m_t0 = cp.empty_like(state.thp)
    out[...] = state.thp
