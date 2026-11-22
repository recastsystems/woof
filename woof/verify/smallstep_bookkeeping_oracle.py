"""Small-step bookkeeping measured against the compiled WRF oracle.

Fixture references are words returned by native WRF v4.7.1 routines.  The
port runners call the engine's production launch functions.  Theta has two
representations: WRF couples theta minus 300 K; WOOF-ArWen couples full theta.
Both the native representation comparison and an explicitly labelled
full-theta isolation call are retained, without an acceptance tolerance.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from functools import lru_cache
import re
import numpy as np

from woof.verify.smallstep_oracle import ORACLE_DIR, word_metrics
from woof.verify.wrf471_fixtures import require_fixture_dir

BOOKKEEPING_DIR = ORACLE_DIR / "bookkeeping"


def _append_observer_parameters(source, entry, parameters):
    """Append to every conditional signature, before the function body."""
    match = re.search(r"void " + re.escape(entry) + r"\([^{}]*\{", source)
    if match is None:
        raise ValueError(f"missing kernel signature for {entry}")
    head = match.group()[:-1]
    if head.count(")") not in (1, 2):
        raise ValueError(f"unsupported observer signature for {entry}")
    head = head.replace(")", ", " + parameters + ")")
    return source[:match.start()] + head + "{" + source[match.end():]
OUTPUT_COVERAGE = {
    "small_step_prep": {
        "u_2": "u_pp", "v_2": "v_pp", "w_2": "w_pp", "t_2": "native_th_pp",
        "ph_2": "ph_pp", "mu_2": "mu_pp",
        **{name: name for name in ("u_1", "v_1", "w_1", "t_1", "ph_1", "mu_1",
                                  "u_save", "v_save", "w_save", "t_save", "ph_save", "mu_save",
                                  "muus", "muvs", "muts", "mudf", "ww_save", "c2a")}},
    "small_step_finish": {
        "u_2": "u", "v_2": "v", "w_2": "w", "t_2": "thp", "ph_2": "php",
        "mu_2": "mup", "ww": "ww",
        **{name: name for name in ("mu_1", "mu_save", "muu", "muus", "muv", "muvs", "mut", "muts")}},
    "calc_p_rho": {"al": "al_pp", "p": "p_pp", "ph": "ph_pp", "pm1": "p_pp_old"},
    "sumflux": {name: name for name in ("ru_m", "rv_m", "ww_m")},
}


@lru_cache(maxsize=1)
def _observed_module():
    """Observe registers without changing a single arithmetic expression."""
    import cupy as cp
    from woof.core.kernels import module_source
    source = module_source("dycore")
    uv_start = source.index("void small_step_init_uv(")
    column_start = source.index("void small_step_init_column(")
    finish_start = source.index("void small_step_finish_uv(")
    finish_column_start = source.index("void small_step_finish_column(")
    uv = source[uv_start:column_start]
    column = source[column_start:finish_start]
    uv = _append_observer_parameters(uv, "small_step_init_uv",
                                    "real* oracle_muus, real* oracle_muvs")
    needle = "real mtf = rn_mul(0.5f, rn_add(mt_a, mt_b));"
    assert uv.count(needle) == 2
    uv = uv.replace(needle, needle + "\n        if (k == 0) oracle_muus[(size_t)j * nxf + i] = mtf;", 1)
    offset = uv.index(needle, uv.index(needle) + len(needle))
    uv = uv[:offset] + uv[offset:].replace(
        needle, needle + "\n        if (k == 0) oracle_muvs[(size_t)j * nx + i] = mtf;", 1)
    column = _append_observer_parameters(column, "small_step_init_column",
                                        "real* oracle_muts, real* oracle_c2a")
    needle = "real mus = rn_add(mub2d[c], mup[c]);"
    assert column.count(needle) == 1
    column = column.replace(needle, needle + "\n    oracle_muts[c] = mut;")
    needle = "real c2a = rn_div(rn_mul(GAMMA, p[h]), alt[h]);"
    assert column.count(needle) == 1
    column = column.replace(needle, needle + "\n        oracle_c2a[h] = c2a;")
    finish_uv = source[finish_start:finish_column_start]
    finish_column = source[finish_column_start:]
    finish_uv = _append_observer_parameters(finish_uv, "small_step_finish_uv",
        "real* oracle_muu, real* oracle_muus, real* oracle_muv, real* oracle_muvs")
    needle = "real mnf = rn_mul(0.5f, rn_add(mna, mnb));"
    assert finish_uv.count(needle) == 2
    finish_uv = finish_uv.replace(needle, needle + "\n        if (k == 0) { oracle_muu[(size_t)j * nxf + i] = msf; oracle_muus[(size_t)j * nxf + i] = mnf; }", 1)
    offset = finish_uv.index(needle, finish_uv.index(needle) + len(needle))
    finish_uv = finish_uv[:offset] + finish_uv[offset:].replace(
        needle, needle + "\n        if (k == 0) { oracle_muv[(size_t)j * nx + i] = msf; oracle_muvs[(size_t)j * nx + i] = mnf; }", 1)
    finish_column = _append_observer_parameters(finish_column, "small_step_finish_column",
                                              "real* oracle_mut, real* oracle_muts")
    needle = "real mun = rn_add(mus, mu_pp[c]);"
    assert finish_column.count(needle) == 1
    finish_column = finish_column.replace(needle, needle + "\n    oracle_mut[c] = mus; oracle_muts[c] = mun;")
    instrumented = source[:uv_start] + uv + column + finish_uv + finish_column
    return cp.RawModule(code=instrumented, options=("-std=c++17",))


def load_bookkeeping(path):
    require_fixture_dir(Path(path).parent, "small-step bookkeeping")
    with np.load(path, allow_pickle=False) as a:
        return {key: a[key].copy() for key in a.files}


def _cfg(fixture):
    periodic = bool(fixture["periodic"])
    return SimpleNamespace(
        periodic_x=periodic, periodic_y=periodic,
        open_x=not periodic, open_y=not periodic, specified=False,
        nested=False, spec_zone=1, emdiv=float(fixture.get("emdiv", 0.01)),
        dx=float(fixture["dx"]), dy=float(fixture["dy"]),
    )


def _state(fixture):
    import cupy as cp
    state = SimpleNamespace(**{key[3:]: cp.asarray(value) for key, value in
                               fixture.items() if key.startswith("in_")})
    nz, ny, nx = state.p.shape
    state.has_msf = bool(fixture["has_msf"])
    state.qv = None
    state.tke = None
    state.total_mu = lambda: state.mub2d + state.mup
    scratch = {}
    def allocate(shape, name):
        if name not in scratch:
            scratch[name] = cp.zeros(shape, dtype=cp.float32)
        return scratch[name]
    state.scratch = allocate
    for name, shape in (("u_pp", state.u.shape), ("v_pp", state.v.shape),
                        ("w_pp", state.w.shape), ("th_pp", state.thp.shape),
                        ("ph_pp", state.php.shape), ("mu_pp", state.mup.shape),
                        ("al_pp", state.p.shape), ("p_pp", state.p.shape),
                        ("p_pp_old", state.p.shape)):
        if not hasattr(state, name):
            setattr(state, name, cp.zeros(shape, dtype=cp.float32))
    return state


def port_prep(fixture):
    """Execute time-level copies and the engine's fused prep/EOS launcher."""
    import cupy as cp
    from woof.core.dycore import _init_small_steps, _save_time_t
    state = _state(fixture)
    if int(fixture["rk_step"]) == 1:
        _save_time_t(state)
    _init_small_steps(state, _cfg(fixture))
    cp.cuda.get_current_stream().synchronize()
    values = {name: cp.asnumpy(getattr(state, name)) for name in
              ("u_pp", "v_pp", "w_pp", "th_pp", "ph_pp", "mu_pp",
               "al_pp", "p_pp", "p_pp_old")}
    # WRF T2 is coupled theta-300; this is an explicit representation map,
    # not a tolerance or a replacement reference.  Every operation rounds.
    offset = (fixture["in_c1h"][:, None, None] * values["mu_pp"][None]
              * np.float32(300.0))
    values["native_th_pp"] = values["th_pp"] - offset
    # Snapshots and saves are real retained device arrays in the engine,
    # rather than additional copies allocated by WRF's driver.
    for target, source in (("u", "u"), ("v", "v"), ("w", "w"),
                           ("t", "thp"), ("ph", "php"), ("mu", "mup")):
        values[target + "_1"] = cp.asnumpy(getattr(state, source + "0"))
        values[target + "_save"] = cp.asnumpy(getattr(state, source))
    values["ww_save"] = cp.asnumpy(state.ww)
    mudf = state.scratch(state.mup.shape, "acoustic_mudf")
    mudf[...] = state.mudf
    if int(fixture["rk_step"]) == 1:
        # Exact outer step protocol at dycore.step's stage-one MUDF reset.
        mudf[...] = 0
    values["mudf"] = cp.asnumpy(mudf)
    # The ordinary launcher has already produced the answer above.  Repeat
    # the same production launch path with extra stores of its live
    # registers, and reject any perturbation of its ordinary outputs.
    observed_state = _state(fixture)
    if int(fixture["rk_step"]) == 1:
        _save_time_t(observed_state)
    observed = {"muus": cp.empty((state.mup.shape[0], state.mup.shape[1] + 1), cp.float32),
                "muvs": cp.empty((state.mup.shape[0] + 1, state.mup.shape[1]), cp.float32),
                "muts": cp.empty_like(state.mup), "c2a": cp.empty_like(state.p)}
    import woof.core.dycore as dycore
    ordinary_get_kernel = dycore.get_kernel
    module = _observed_module()
    def get_observed(module_name, name):
        if module_name == "dycore" and name in ("small_step_init_uv", "small_step_init_column"):
            kernel = module.get_function(name)
            extras = (observed["muus"], observed["muvs"]) if name.endswith("uv") else (observed["muts"], observed["c2a"])
            return lambda grid, block, args: kernel(grid, block, (*args, *extras))
        return ordinary_get_kernel(module_name, name)
    try:
        dycore.get_kernel = get_observed
        _init_small_steps(observed_state, _cfg(fixture))
        cp.cuda.get_current_stream().synchronize()
    finally:
        dycore.get_kernel = ordinary_get_kernel
    for name in ("u_pp", "v_pp", "w_pp", "th_pp", "ph_pp", "mu_pp",
                 "al_pp", "p_pp", "p_pp_old"):
        observed_value = cp.asnumpy(getattr(observed_state, name))
        if not np.array_equal(values[name].view(np.uint32), observed_value.view(np.uint32)):
            raise AssertionError("Register observation changed production " + name)
    values.update({name: cp.asnumpy(value) for name, value in observed.items()})
    return values


def port_finish(fixture):
    """Execute the engine's uncoupling launcher, including retained heating."""
    import cupy as cp
    from woof.core.dycore import _finish_small_steps, _sumflux_launch
    state = _state(fixture)
    saved_mu = cp.asnumpy(state.mup)
    _finish_small_steps(state, _cfg(fixture), float(fixture["hdiab_dt"]))
    cp.cuda.get_current_stream().synchronize()
    values = {name: cp.asnumpy(getattr(state, name)) for name in
              ("u", "v", "w", "thp", "php", "mup")}
    values["mu_1"] = cp.asnumpy(state.mup0)
    values["mu_save"] = saved_mu
    # WRF stores post-finish total WW.  The engine reconstructs its total
    # eta flux through scalar sumflux.  With one substep the same actual
    # production path yields precisely this logical WW output.
    targets = (cp.empty_like(state.u), cp.empty_like(state.v), cp.empty_like(state.ww))
    _sumflux_launch("zero_sumflux", targets)
    _sumflux_launch("accumulate_sumflux", targets, (state.u_pp, state.v_pp, state.ww_pp))
    _sumflux_launch("finish_sumflux", targets, (state.u, state.v, state.ww), 1)
    values["ww"] = cp.asnumpy(targets[2])
    observed_state = _state(fixture)
    observed = {name: cp.empty(shape, cp.float32) for name, shape in
                (("muu", state.msfu.shape), ("muus", state.msfu.shape),
                 ("muv", state.msfv.shape), ("muvs", state.msfv.shape),
                 ("mut", state.mup.shape), ("muts", state.mup.shape))}
    import woof.core.dycore as dycore
    ordinary_get_kernel = dycore.get_kernel
    module = _observed_module()
    def get_observed(module_name, name):
        if module_name == "dycore" and name in ("small_step_finish_uv", "small_step_finish_column"):
            kernel = module.get_function(name)
            extras = tuple(observed[key] for key in ("muu", "muus", "muv", "muvs")) if name.endswith("uv") else (observed["mut"], observed["muts"])
            return lambda grid, block, args: kernel(grid, block, (*args, *extras))
        return ordinary_get_kernel(module_name, name)
    try:
        dycore.get_kernel = get_observed
        _finish_small_steps(observed_state, _cfg(fixture), float(fixture["hdiab_dt"]))
        cp.cuda.get_current_stream().synchronize()
    finally:
        dycore.get_kernel = ordinary_get_kernel
    for name in ("u", "v", "w", "thp", "php", "mup"):
        observed_value = cp.asnumpy(getattr(observed_state, name))
        if not np.array_equal(values[name].view(np.uint32), observed_value.view(np.uint32)):
            raise AssertionError("Register observation changed production " + name)
    values.update({name: cp.asnumpy(value) for name, value in observed.items()})
    return values


def port_sumflux(fixture):
    """Call all three actual batched sumflux launch paths."""
    import cupy as cp
    from woof.core.dycore import _sumflux_launch, stage_fluxes
    targets = tuple(cp.asarray(fixture["in_" + name]) for name in
                    ("ru_m", "rv_m", "ww_m"))
    _sumflux_launch("zero_sumflux", targets)
    nsub = int(fixture["nsub"])
    history = {}
    for iteration in range(nsub):
        sources = tuple(cp.asarray(fixture[f"flux_{iteration}_{name}"])
                        for name in ("u", "v", "w"))
        _sumflux_launch("accumulate_sumflux", targets, sources)
        for name, value in zip(("ru_m", "rv_m", "ww_m"), targets):
            history[f"iteration_{iteration+1}_{name}"] = cp.asnumpy(value)
    state = _state(fixture)
    ru, rv, _ = stage_fluxes(state, _cfg(fixture))
    _sumflux_launch("finish_sumflux", targets,
                   (ru, rv, cp.asarray(fixture["in_ww"])), nsub)
    cp.cuda.get_current_stream().synchronize()
    history.update({name: cp.asnumpy(value) for name, value in
                    zip(("ru_m", "rv_m", "ww_m"), targets)})
    return history


def port_emdiv(fixture):
    """Call the production divergence-filter launcher on its saved inputs."""
    import cupy as cp
    from woof.core.dycore import apply_emdiv_filter
    state = _state(fixture)
    mudf = cp.asarray(fixture["in_mudf"])
    previous = cp.full_like(state.mu_pp, np.float32(-73.0))
    apply_emdiv_filter(state, _cfg(fixture), mudf, previous)
    cp.cuda.get_current_stream().synchronize()
    return {"u_pp": cp.asnumpy(state.u_pp),
            "v_pp": cp.asnumpy(state.v_pp),
            "mu_prev": cp.asnumpy(previous)}


PORT_RUNNERS = {"prep": port_prep, "finish": port_finish,
                "sumflux": port_sumflux, "emdiv": port_emdiv}


def measure_bookkeeping(fixture, routine):
    actual = PORT_RUNNERS[routine](fixture)
    return {name: word_metrics(actual[name], value) for key, value in
            fixture.items() if key.startswith("ref_")
            and (name := key[4:]) in actual}


def prep_rounding_trace(fixture):
    """Source-level FP32 trace used only to attribute a measured difference.

    This is not an oracle reference.  Tests first compare production GPU
    outputs with compiled WRF, then demand that these explicit rounding
    trees reproduce the production words.  WRF's native outputs remain
    the independent expected values.
    """
    f = {key[3:]: value for key, value in fixture.items() if key.startswith("in_")}
    periodic = bool(fixture["periodic"])
    mut, mus = f["mub2d"] + f["mup0"], f["mub2d"] + f["mup"]
    def mean(a, axis):
        pads = [(0, 0), (0, 0)]
        pads[axis] = (1, 1)
        p = np.pad(a, pads, mode="wrap" if periodic else "edge")
        left = np.take(p, np.arange(a.shape[axis] + 1), axis=axis)
        right = np.take(p, np.arange(1, a.shape[axis] + 2), axis=axis)
        return np.float32(.5) * (left + right)
    ch, c2 = f["c1h"][:, None, None], f["c2h"][:, None, None]
    ct, cs = ch * mut[None] + c2, ch * mus[None] + c2
    u = ((ch * mean(mut, 1)[None] + c2) * f["u0"]
         - (ch * mean(mus, 1)[None] + c2) * f["u"]) / f["msfu"][None]
    v = ((ch * mean(mut, 0)[None] + c2) * f["v0"]
         - (ch * mean(mus, 0)[None] + c2) * f["v"]) / f["msfv"][None]
    mupp = f["mup0"] - f["mup"]
    ths = f["thb"] + f["thp"]
    thpp = ct * (f["thb"] + f["thp0"]) - cs * ths
    ph = f["php0"] - f["php"]
    sum_al = f["alt"] * (ch * mupp[None]) + f["rdnw"][:, None, None] * (ph[1:] - ph[:-1])
    al = -sum_al / ct
    gamma = np.float32(np.float32(1004.5) / np.float32(717.5))
    c2a = (gamma * f["p"]) / f["alt"]
    pp = c2a * (f["alt"] * (thpp - (ch * mupp[None]) * ths) / (ct * ths) - al)
    return {"u_pp": u, "v_pp": v, "th_pp": thpp, "al_pp": al,
            "p_pp": pp, "p_pp_old": pp.copy()}


def finish_rounding_trace(fixture):
    """Named coupling and heating rounding trees, solely for attribution."""
    f = {key[3:]: value for key, value in fixture.items() if key.startswith("in_")}
    periodic = bool(fixture["periodic"])
    mus = f["mub2d"] + f["mup"]
    mun = mus + f["mu_pp"]
    def mean(a, axis):
        pads = [(0, 0), (0, 0)]
        pads[axis] = (1, 1)
        p = np.pad(a, pads, mode="wrap" if periodic else "edge")
        left = np.take(p, np.arange(a.shape[axis] + 1), axis=axis)
        right = np.take(p, np.arange(1, a.shape[axis] + 2), axis=axis)
        return np.float32(.5) * (left + right)
    ch, c2 = f["c1h"][:, None, None], f["c2h"][:, None, None]
    cf, cf2 = f["c1f"][:, None, None], f["c2f"][:, None, None]
    u = ((ch * mean(mus, 1)[None] + c2) * f["u"] + f["u_pp"] * f["msfu"][None]) / (ch * mean(mun, 1)[None] + c2)
    v = ((ch * mean(mus, 0)[None] + c2) * f["v"] + f["v_pp"] * f["msfv"][None]) / (ch * mean(mun, 0)[None] + c2)
    w = ((cf * mus[None] + cf2) * f["w"] + f["w_pp"] * f["msft"][None]) / (cf * mun[None] + cf2)
    numerator = (ch * mus[None] + c2) * (f["thb"] + f["thp"]) + f["th_pp"]
    if float(fixture["hdiab_dt"]):
        removal = (np.float32(fixture["hdiab_dt"]) * (ch * mus[None] + c2)) * f["h_diabatic"]
        numerator = numerator - removal
    return {"u": u, "v": v, "w": w, "thp": numerator / (ch * mun[None] + c2) - f["thb"],
            "php": f["php"] + f["ph_pp"], "mup": f["mup"] + f["mu_pp"]}


def emdiv_rounding_trace(fixture):
    """Source scalar operator boundaries that explain the reciprocal gap."""
    f = {key[3:]: value for key, value in fixture.items() if key.startswith("in_")}
    periodic = bool(fixture["periodic"])
    c1h = f["c1h"][:, None, None]
    gx = np.float32(-float(fixture["emdiv"]) * float(fixture["dx"])) * (f["mudf"] - np.roll(f["mudf"], 1, axis=1))
    gy = np.float32(-float(fixture["emdiv"]) * float(fixture["dy"])) * (f["mudf"] - np.roll(f["mudf"], 1, axis=0))
    u, v = f["u_pp"].copy(), f["v_pp"].copy()
    if periodic:
        gx = np.concatenate((gx, gx[:, :1]), axis=1) / f["msfu"]
        gy = np.concatenate((gy, gy[:1]), axis=0) / f["msfv"]
        u += c1h * gx[None]
        v += c1h * gy[None]
    else:
        u[:, :, 1:-1] += c1h * (gx[:, 1:] / f["msfu"][:, 1:-1])[None]
        v[:, 1:-1, :] += c1h * (gy[1:] / f["msfv"][1:-1])[None]
    return {"u_pp": u, "v_pp": v}


def sumflux_rounding_trace(fixture):
    """Actual stored input fluxes and scalar mean operator boundaries."""
    result = {}
    nsub = int(fixture["nsub"])
    for name, source, reference in (("ru_m", "u", "ru"), ("rv_m", "v", "rv"), ("ww_m", "w", "ww")):
        summed = np.zeros_like(fixture["in_" + name])
        for iteration in range(nsub):
            summed = summed + fixture[f"flux_{iteration}_{source}"]
        result[name] = (summed / np.float32(nsub)) + fixture["in_" + reference]
    return result


ROUNDING_TRACES = {"prep": prep_rounding_trace, "finish": finish_rounding_trace,
                   "sumflux": sumflux_rounding_trace, "emdiv": emdiv_rounding_trace}
