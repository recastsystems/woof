"""Package native compiled-WRF bookkeeping inputs and output words."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from woof.verify.smallstep_oracle import WRFOracle, load_cases


def _face_neighbors(mu, axis, periodic):
    pad = [(0, 0), (0, 0)]
    pad[axis] = (1, 1)
    padded = np.pad(mu, pad, mode="wrap" if periodic else "edge")
    a = np.take(padded, np.arange(1, mu.shape[axis] + 2), axis=axis)
    b = np.take(padded, np.arange(mu.shape[axis] + 1), axis=axis)
    return a, b


def face_mass(mu, axis, periodic):
    """WOOF-ArWen's declared mean of already-total column masses."""
    a, b = _face_neighbors(mu, axis, periodic)
    return np.float32(0.5) * (a + b)


def wrf_face_mass(mu, mub, axis, periodic):
    """WRF calc_mu_uv's perturbation-first REAL input preparation."""
    a, b = _face_neighbors(mu, axis, periodic)
    ba, bb = _face_neighbors(mub, axis, periodic)
    return np.float32(0.5) * ((a + b) + ba + bb)


def _get(raw, *names):
    for name in names:
        if name in raw:
            return np.ascontiguousarray(raw[name], dtype=np.float32)
    raise KeyError(names)


def base_inputs(raw):
    t = _get(raw, "T", "THM")
    p = _get(raw, "P") + _get(raw, "PB")
    q = raw.get("QVAPOR", np.zeros_like(t))
    # A thermodynamically consistent dry inverse density from the real state.
    # If the common fixture supplies ALT, it is used directly.
    alt = raw.get("ALT")
    if alt is None:
        tk = (t + np.float32(300)) * (p / np.float32(100000)) ** np.float32(287 / 1004.5)
        alt = np.float32(287) * tk * (np.float32(1) + np.float32(1.608) * q) / p
    nz, ny, nx = t.shape
    c1h = _get(raw, "C1H")[:nz]
    c2h = _get(raw, "C2H")[:nz]
    c1f = _get(raw, "C1F")[:nz+1]
    c2f = _get(raw, "C2F")[:nz+1]
    dnw = _get(raw, "DNW")[:nz]
    return {
        "u": _get(raw, "U"), "v": _get(raw, "V"), "w": _get(raw, "W"),
        "thp": t, "php": _get(raw, "PH"), "mup": _get(raw, "MU"),
        "mub2d": _get(raw, "MUB"), "p": p, "alt": np.asarray(alt, np.float32),
        "thb": np.full((nz, ny, nx), np.float32(300)),
        "c1h": c1h, "c2h": c2h, "c1f": c1f, "c2f": c2f,
        "dnw": dnw, "rdnw": _get(raw, "RDNW")[:nz],
        "msfu": _get(raw, "MAPFAC_U", "MSFU", "MSFUY"),
        "msfv": _get(raw, "MAPFAC_V", "MSFV", "MSFVX"),
        "msft": _get(raw, "MAPFAC_M", "MSFT", "MSFTY"),
        **{key.lower(): _get(raw, key) for key in ("C3H", "C4H", "C3F", "C4F")},
    }


def periodic_faces(values, periodic):
    """Load the duplicated physical faces required by a periodic domain."""
    if periodic:
        for key in ("u", "u0", "u_pp"):
            if key in values:
                values[key][:, :, -1] = values[key][:, :, 0]
        for key in ("v", "v0", "v_pp"):
            if key in values:
                values[key][:, -1, :] = values[key][:, 0, :]
        values["msfu"][:, -1] = values["msfu"][:, 0]
        values["msfv"][-1, :] = values["msfv"][0, :]
    return values


def coefficients(o, inputs):
    return {name: o.array(inputs[name]) for name in ("c1h", "c2h", "c1f", "c2f")} | {
        name: o.array(inputs[name])
        for name in ("c3h", "c4h", "c3f", "c4f")}


def maps(o, inputs):
    return {"msfux": o.array(inputs["msfu"]), "msfuy": o.array(inputs["msfu"]),
            "msfvx": o.array(inputs["msfv"]), "msfvy": o.array(inputs["msfv"]),
            "msfvx_inv": o.array(np.float32(1) / inputs["msfv"]),
            "msftx": o.array(inputs["msft"]), "msfty": o.array(inputs["msft"])}


def _stage(inputs, rk_step):
    values = {key: value.copy() for key, value in inputs.items()}
    # One acoustically small, spatially smooth stage displacement of a real
    # state.  It prevents cancellation to zero from disguising a wrong sign.
    for key, amount in (("u", .03125), ("v", -.015625), ("w", .0009765625),
                        ("thp", .0009765625), ("php", .125), ("mup", .0625)):
        values[key + "0"] = values[key].copy()
        if rk_step > 1:
            pattern = np.linspace(-1, 1, values[key].size, dtype=np.float32).reshape(values[key].shape)
            values[key] += np.float32(amount * (rk_step - 1)) * pattern
    return values


def _header(values, periodic, metadata):
    return {"periodic": np.asarray(periodic), "has_msf": np.asarray(True),
            "dx": np.asarray(metadata.get("dx", 3000.0)),
            "dy": np.asarray(metadata.get("dy", 3000.0)),
            **{"in_" + key: value for key, value in values.items()}}


def prep_fixture(o, raw, metadata, rk_step):
    original = base_inputs(raw)
    values = periodic_faces(_stage(original, rk_step), o.periodic)
    values["ww"] = values["w"] * np.float32(10)
    values["mudf"] = np.full_like(values["mup"], np.float32(.003))
    theta_shape = values["thp"].shape
    phi_shape = values["php"].shape
    surface_shape = values["mup"].shape
    args = coefficients(o, values) | maps(o, values)
    # Canonical WRF theta perturbation, with the same 300 K model constant.
    for target, source in (("u", "u"), ("v", "v"), ("w", "w"),
                           ("t", "thp"), ("ph", "php"), ("mu", "mup")):
        args[target + "_1"] = o.array(values[source + "0"])
        args[target + "_2"] = o.array(values[source])
        args[target + "_save"] = o.array(np.zeros_like(values[source]))
    mt = values["mub2d"] + values["mup"]
    args.update(mub=o.array(values["mub2d"]), mut=o.array(mt),
                muu=o.array(wrf_face_mass(values["mup"], values["mub2d"], 1, o.periodic)),
                muv=o.array(wrf_face_mass(values["mup"], values["mub2d"], 0, o.periodic)),
                muus=o.array(np.zeros(surface_shape, np.float32)),
                muvs=o.array(np.zeros(surface_shape, np.float32)),
                muts=o.array(np.zeros(surface_shape, np.float32)),
                mudf=o.array(values["mudf"]),
                ww=o.array(values["ww"]),
                ww_save=o.array(np.zeros(phi_shape, np.float32)),
                c2a=o.array(np.zeros(theta_shape, np.float32)),
                p=o.array(_get(raw, "P")), pb=o.array(_get(raw, "PB")),
                alt=o.array(values["alt"]), rdx=1 / metadata.get("dx", 3000.),
                rdy=1 / metadata.get("dy", 3000.), rk_step=rk_step)
    o.call("small_step_prep", **args)
    fixture = _header(values, o.periodic, metadata)
    fixture["rk_step"] = np.asarray(rk_step)
    for name, key, shape in (("u_pp", "u_2", values["u"].shape),
                             ("v_pp", "v_2", values["v"].shape),
                             ("w_pp", "w_2", phi_shape),
                             ("native_th_pp", "t_2", theta_shape),
                             ("ph_pp", "ph_2", phi_shape),
                             ("mu_pp", "mu_2", surface_shape)):
        fixture["ref_" + name] = o.extract(args[key], shape)
    for target, source in (("u", "u"), ("v", "v"), ("w", "w"),
                           ("t", "thp"), ("ph", "php"), ("mu", "mup")):
        for suffix in ("_1", "_save"):
            fixture["ref_" + target + suffix] = o.extract(args[target + suffix], values[source].shape)
    for name, shape in (("muus", values["msfu"].shape), ("muvs", values["msfv"].shape),
                        ("muts", surface_shape), ("mudf", surface_shape),
                        ("ww_save", phi_shape), ("c2a", theta_shape)):
        fixture["ref_" + name] = o.extract(args[name], shape)
    eos = coefficients(o, values) | dict(
        al=o.array(np.zeros(theta_shape, np.float32)),
        p=o.array(np.zeros(theta_shape, np.float32)), ph=args["ph_2"],
        alt=args["alt"], t_2=args["t_2"], t_1=args["t_save"], c2a=args["c2a"],
        pm1=o.array(np.zeros(theta_shape, np.float32)), mu=args["mu_2"],
        mut=args["muts"], znu=o.array(_get(raw, "ZNU")[:theta_shape[0]]),
        t0=300., rdnw=o.array(values["rdnw"]), dnw=o.array(values["dnw"]),
        smdiv=.1, non_hydrostatic=True, step=0)
    o.call("calc_p_rho", **eos)
    for name, key in (("al_pp", "al"), ("p_pp", "p"), ("p_pp_old", "pm1")):
        fixture["ref_" + name] = o.extract(eos[key], theta_shape)
    # Direct source-routine representation isolation: full theta is a
    # separately labelled input and t0=0 only in this diagnostic call.
    full = {key: value.copy(order="F") if isinstance(value, np.ndarray) else value for key, value in args.items()}
    for target, source in (("u", "u"), ("v", "v"), ("w", "w"),
                           ("ph", "php"), ("mu", "mup")):
        full[target + "_1"] = o.array(values[source + "0"])
        full[target + "_2"] = o.array(values[source])
    full["t_1"] = o.array(values["thp0"] + np.float32(300))
    full["t_2"] = o.array(values["thp"] + np.float32(300))
    o.call("small_step_prep", **full)
    fixture["isolation_th_pp"] = o.extract(full["t_2"], theta_shape)
    isolated_eos = eos | dict(t_2=full["t_2"], t_1=full["t_save"], t0=0.,
                              mu=full["mu_2"], mut=full["muts"], ph=full["ph_2"])
    o.call("calc_p_rho", **isolated_eos)
    for name, key in (("al_pp", "al"), ("p_pp", "p"), ("p_pp_old", "pm1")):
        fixture["isolation_" + name] = o.extract(isolated_eos[key], theta_shape)
    return fixture


def finish_fixture(o, raw, metadata, final):
    values = periodic_faces(_stage(base_inputs(raw), 2), o.periodic)
    nz, ny, nx = values["thp"].shape
    # Small coupled acoustic increments with all real input staggering.
    for name, amount in (("u_pp", .125), ("v_pp", -.0625), ("w_pp", .015625),
                         ("ph_pp", .125), ("mu_pp", .0625), ("th_pp", .25)):
        field = {"u_pp": "u", "v_pp": "v", "w_pp": "w", "ph_pp": "php",
                 "mu_pp": "mup", "th_pp": "thp"}[name]
        values[name] = np.linspace(-amount, amount, values[field].size,
                                   dtype=np.float32).reshape(values[field].shape)
    periodic_faces(values, o.periodic)
    values["h_diabatic"] = np.full((nz, ny, nx), np.float32(.000125))
    values["ww"] = values["w"] * np.float32(10)
    values["ww_pp"] = values["w_pp"].copy()
    # Keep WRF's native theta numerator while explicitly mapping the
    # full-theta engine input, as required by the differing conventions.
    native_pp = values["th_pp"] - (values["c1h"][:, None, None]
                                   * values["mu_pp"][None] * np.float32(300))
    ms = values["mub2d"] + values["mup"]
    mn = ms + values["mu_pp"]
    args = coefficients(o, values) | maps(o, values)
    for target, source, pp in (("u", "u", "u_pp"), ("v", "v", "v_pp"),
                               ("w", "w", "w_pp"), ("t", "thp", "th_pp"),
                               ("ph", "php", "ph_pp"), ("mu", "mup", "mu_pp")):
        args[target + "_1"] = o.array(values[source + "0"])
        args[target + "_2"] = o.array(native_pp if target == "t" else values[pp])
        args[target + "_save"] = o.array(values[source])
    args.update(mut=o.array(ms), muts=o.array(mn),
                muu=o.array(wrf_face_mass(values["mup"], values["mub2d"], 1, o.periodic)),
                muus=o.array(wrf_face_mass(values["mup"] + values["mu_pp"], values["mub2d"], 1, o.periodic)),
                muv=o.array(wrf_face_mass(values["mup"], values["mub2d"], 0, o.periodic)),
                muvs=o.array(wrf_face_mass(values["mup"] + values["mu_pp"], values["mub2d"], 0, o.periodic)),
                ww=o.array(values["ww_pp"]), ww1=o.array(values["ww"]),
                h_diabatic=o.array(values["h_diabatic"]),
                number_of_small_timesteps=6, dts=.5, rk_step=3 if final else 2,
                rk_order=3)
    args.pop("msfvx_inv")
    o.call("small_step_finish", **args)
    fixture = _header(values, o.periodic, metadata)
    fixture["hdiab_dt"] = np.asarray(3. if final else 0.)
    for name, key in (("u", "u_2"), ("v", "v_2"), ("w", "w_2"),
                      ("thp", "t_2"), ("php", "ph_2"), ("mup", "mu_2")):
        fixture["ref_" + name] = o.extract(args[key], values[name].shape)
    fixture["ref_ww"] = o.extract(args["ww"], values["w"].shape)
    for name, shape in (("mu_1", values["mup"].shape), ("mu_save", values["mup"].shape),
                        ("muu", values["msfu"].shape), ("muus", values["msfu"].shape),
                        ("muv", values["msfv"].shape), ("muvs", values["msfv"].shape),
                        ("mut", values["mup"].shape), ("muts", values["mup"].shape)):
        fixture["ref_" + name] = o.extract(args[name], shape)
    isolated = {key: value.copy(order="F") if isinstance(value, np.ndarray) else value
                for key, value in args.items()}
    for target, source, pp in (("u", "u", "u_pp"), ("v", "v", "v_pp"),
                               ("w", "w", "w_pp"), ("t", "thp", "th_pp"),
                               ("ph", "php", "ph_pp"), ("mu", "mup", "mu_pp")):
        isolated[target + "_2"] = o.array(values[pp])
        isolated[target + "_save"] = o.array(values[source] + np.float32(300) if target == "t"
                                              else values[source])
    isolated.update(muu=o.array(face_mass(ms, 1, o.periodic)),
                    muus=o.array(face_mass(mn, 1, o.periodic)),
                    muv=o.array(face_mass(ms, 0, o.periodic)),
                    muvs=o.array(face_mass(mn, 0, o.periodic)))
    o.call("small_step_finish", **isolated)
    for name, key in (("u", "u_2"), ("v", "v_2"), ("w", "w_2"),
                      ("thp", "t_2"), ("php", "ph_2"), ("mup", "mu_2")):
        witness = o.extract(isolated[key], values[name].shape)
        fixture["isolation_" + name] = witness - np.float32(300) if name == "thp" else witness
    return fixture


def sumflux_fixture(o, raw, metadata, nsub):
    values = periodic_faces(base_inputs(raw), o.periodic)
    ms = values["mub2d"] + values["mup"]
    muu = wrf_face_mass(values["mup"], values["mub2d"], 1, o.periodic)
    muv = wrf_face_mass(values["mup"], values["mub2d"], 0, o.periodic)
    engine_muu, engine_muv = face_mass(ms, 1, o.periodic), face_mass(ms, 0, o.periodic)
    ch = values["c1h"][:, None, None]
    c2 = values["c2h"][:, None, None]
    values["ru"] = ((ch * engine_muu[None] + c2) * values["u"]) / values["msfu"][None]
    values["rv"] = ((ch * engine_muv[None] + c2) * values["v"]) / values["msfv"][None]
    values["ww"] = values["w"] * np.float32(10)
    for name, src in (("ru_m", "ru"), ("rv_m", "rv"), ("ww_m", "ww")):
        values[name] = np.full_like(values[src], np.float32(-17))
    args = coefficients(o, values) | maps(o, values)
    for key in ("msftx", "msfty"):
        args.pop(key)
    args.update(u_lin=o.array(values["u"]), v_lin=o.array(values["v"]),
                ww_lin=o.array(values["ww"]), muu=o.array(muu), muv=o.array(muv),
                ru_m=o.array(values["ru_m"]), rv_m=o.array(values["rv_m"]),
                ww_m=o.array(values["ww_m"]), epssm=.1,
                number_of_small_timesteps=nsub)
    fixture = _header(values, o.periodic, metadata)
    fixture["nsub"] = np.asarray(nsub)
    for iteration in range(nsub):
        for key, src in (("ru", "u"), ("rv", "v"), ("ww", "w")):
            flow = np.float32((iteration + 1) * .0009765625) * values[src]
            args[key] = o.array(flow)
            fixture[f"flux_{iteration}_{src}"] = flow
        args["iteration"] = iteration + 1
        o.call("sumflux", **args)
        if iteration + 1 < nsub:
            for name in ("ru_m", "rv_m", "ww_m"):
                fixture[f"ref_iteration_{iteration+1}_{name}"] = o.extract(args[name], values[name].shape)
    for name in ("ru_m", "rv_m", "ww_m"):
        fixture["ref_" + name] = o.extract(args[name], values[name].shape)
    return fixture


def emdiv_fixture(o, raw, metadata, emdiv):
    values = periodic_faces(base_inputs(raw), o.periodic)
    nz, ny, nx = values["p"].shape
    for pp, source in (("u_pp", "u"), ("v_pp", "v")):
        values[pp] = values[source] * np.float32(.0078125)
    values["mu_pp"] = values["mup"] * np.float32(.001)
    periodic_faces(values, o.periodic)
    # A mass tendency trace with real-state spatial structure, not random
    # words.  The source pressure anomaly gives a realistic smooth field.
    values["mudf"] = raw["P"][0] * np.float32(.00003125)
    args = coefficients(o, values) | maps(o, values)
    for key in ("msftx", "msfty"):
        args.pop(key)
    zero3 = o.array(np.zeros((nz, ny, nx), np.float32))
    zerof = o.array(np.zeros((nz+1, ny, nx), np.float32))
    zero2 = o.array(np.zeros((ny, nx), np.float32))
    args.update(u=o.array(values["u_pp"]), v=o.array(values["v_pp"]),
                ru_tend=zero3, rv_tend=zero3, p=zero3, pb=zero3,
                ph=zerof, php=zerof, alt=o.array(np.ones((nz, ny, nx), np.float32)),
                al=zero3, mu=zero2, muu=o.array(values["mub2d"]),
                muv=o.array(values["mub2d"]),
                cqu=o.array(np.ones((nz, ny, nx), np.float32)),
                cqv=o.array(np.ones((nz, ny, nx), np.float32)), mudf=o.array(values["mudf"]),
                rdx=1 / metadata.get("dx", 3000.), rdy=1 / metadata.get("dy", 3000.),
                dts=1., cf1=1., cf2=0., cf3=0.,
                fnm=o.array(np.full(nz+1, np.float32(.5))),
                fnp=o.array(np.full(nz+1, np.float32(.5))),
                rdnw=o.array(values["rdnw"]), emdiv=emdiv, spec_zone=1,
                non_hydrostatic=False, top_lid=False,
                config_flags={"periodic_x": o.periodic, "periodic_y": o.periodic,
                              "open_xs": not o.periodic, "open_xe": not o.periodic,
                              "open_ys": not o.periodic, "open_ye": not o.periodic})
    o.call("advance_uv", **args)
    fixture = _header(values, o.periodic, metadata)
    fixture["emdiv"] = np.asarray(emdiv)
    for name, key in (("u_pp", "u"), ("v_pp", "v")):
        fixture["ref_" + name] = o.extract(args[key], values[name].shape)
    fixture["ref_mu_prev"] = values["mu_pp"].copy()
    return fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "wrf471-smallstep-bookkeeping-v1", "files": {},
                "theta_representation": "canonical WRF theta-300 with explicit WOOF-ArWen full-theta map"}
    for case, raw, metadata in load_cases():
        nz, ny, nx = raw["T"].shape
        for periodic in (False, True):
            o = WRFOracle(a.library, nx, ny, nz, periodic)
            for routine, variants, make in (
                    ("prep", (1, 2, 3), prep_fixture),
                    ("finish", (False, True), finish_fixture),
                    ("sumflux", (1, 3, 6), sumflux_fixture),
                    ("emdiv", (0.01, .015625), emdiv_fixture)):
                for variant in variants:
                    fixture = make(o, raw, metadata, variant)
                    name = f"{case}-{int(periodic)}-{routine}-{variant}.npz"
                    path = a.output / name
                    np.savez_compressed(path, **fixture)
                    manifest["files"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                               "case": case, "routine": routine,
                                               "periodic": periodic, "variant": variant}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
