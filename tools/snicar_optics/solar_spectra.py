"""Incident solar spectra for the snow-albedo option, from AFGL atmospheres
through our RRTMGP shortwave. Re-implemented from published descriptions;
no SNICAR program, SBDART table or upstream spectrum file is an input.

Method (provenance per step in docs/SNOW-PAPERS.md, "Incident spectra"):
1. Atmospheres: AFGL-TR-86-0110 Table 1 (Anderson et al. 1986, US government,
   public domain): mid-latitude winter/summer and sub-Arctic winter/summer,
   0-75 km. Summit summer is sub-Arctic summer truncated at 796 hPa and high
   mountain summer is mid-latitude summer truncated at 556 hPa, the choices
   Flanner et al. 2021 (GMD 14, 7673) section 2.4 publish.
2. Clear sky for direct light, cloudy sky for diffuse light (same section):
   a liquid cloud of optical depth 10 at 500 nm, 10 um effective radius, in
   the layer holding 800 hPa (bottom layer when the surface is above it);
   aerosol optical depth 0.05 with sulfate-like optics. The lower boundary
   is snow of 100 um effective radius, from our own Mie table and our own
   adding-doubling solver (snow_ad_probe).
3. Our BSD RRTMGP (14 SW bands, 224 g-points) gives the band-integrated
   surface direct and diffuse flux for each atmosphere and cosine of the
   solar zenith angle.
4. Within each RRTMGP band the 0.01 um spectral shape is the ASTM G173
   extraterrestrial spectrum times its own published spectral transmittance
   raised to a band exponent fitted so the band integral matches RRTMGP
   (a Beer-law rescaling of the optical path). Diffuse light keeps the
   direct exponent at mu=0.6 for its in-band shape and is scaled to the
   RRTMGP band total.
Run: python solar_spectra.py profiles.json OUT.bin OUT.json
"""
import json
import sys
import hashlib
from pathlib import Path
import numpy as np

MU = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
CASES = (("mid-latitude winter", "mlw", None), ("mid-latitude summer", "mls", None),
         ("sub-Arctic winter", "saw", None), ("sub-Arctic summer", "sas", None),
         ("Summit Greenland summer", "sas", 796.0), ("high mountain summer", "mls", 556.0))
NW = 480
NR = 32
OP = 1504
TOP_KM = 75.0
MAGIC = b"GPWMSPC1"


def levels(prof, psfc):
    z = np.asarray(prof["alt_km"])
    keep = z <= TOP_KM
    p = np.asarray(prof["p_mb"])[keep] * 100.0
    cols = {k: np.asarray(prof[k], np.float64)[keep]
            for k in ("T_K", "h2o_ppmv", "o3_ppmv", "n2o_ppmv", "co_ppmv", "ch4_ppmv")}
    if psfc is not None:
        ps = psfc * 100.0
        lp = np.log(p)
        above = p < ps
        new = {}
        for k, v in cols.items():
            f = v if k == "T_K" else np.log(v)
            val = np.interp(np.log(ps), lp[::-1], f[::-1])
            new[k] = np.concatenate([[val if k == "T_K" else np.exp(val)], v[above]])
        p = np.concatenate([[ps], p[above]])
        cols = new
    return p, cols


def g173(path):
    a = np.loadtxt(path, delimiter=",", skiprows=2)
    lam = a[:, 0] * 1e-3
    etr = a[:, 1]
    direct = a[:, 3]
    trans = np.where(etr > 0, np.clip(direct / np.maximum(etr, 1e-30), 1e-12, 1.0), 1e-12)
    return lam, etr, trans


def bin_integrate(lam, f):
    """Trapezoid integral of f over each 0.01 um bin, 0.2-5 um, split at bin edges."""
    out = np.zeros(NW)
    edges = 0.2 + 0.01 * np.arange(NW + 1)
    fine = np.unique(np.concatenate([lam, edges[(edges > lam[0]) & (edges < lam[-1])]]))
    ff = np.interp(fine, lam, f)
    mid = 0.5 * (fine[1:] + fine[:-1])
    seg = 0.5 * (ff[1:] + ff[:-1]) * np.diff(fine)
    idx = np.floor((mid - 0.2) / 0.01 + 1e-9).astype(int)
    ok = (idx >= 0) & (idx < NW)
    np.add.at(out, idx[ok], seg[ok])
    return out


def main(profiles_path, out_bin, out_json):
    import cupy as cp
    import netCDF4
    from woof.core import rrtmgp as R
    from woof.core import snow_papers as S

    here = Path(__file__).resolve().parent
    lam, etr, trans = g173(here / "data" / "solar.csv")
    profiles = json.load(open(profiles_path))
    sw = R.load_gas_tables("sw")
    cld = R.load_cloud_tables("sw")
    nc = netCDF4.Dataset(Path(R.DATA_DIR) / "rrtmgp-gas-sw-g224.nc")
    wvn = np.asarray(nc.variables["bnd_limits_wavenumber"][:], np.float64)
    nc.close()
    lam_lo, lam_hi = 1e4 / wvn[:, 1], 1e4 / wvn[:, 0]
    nband, ngpt = sw.nband, sw.ngpt
    gpt_band = np.zeros(ngpt, int)
    for b, (s, e) in enumerate(sw.band_lims_gpt):
        gpt_band[s:e + 1] = b

    def band_of(x):
        out = np.full(np.shape(x), -1)
        for b in range(nband):
            out[(x >= lam_lo[b]) & (x < lam_hi[b])] = b
        return out

    fine_band = band_of(lam)
    centres = 0.205 + 0.01 * np.arange(NW)
    bin_band = band_of(centres)
    etr_bins = bin_integrate(lam, etr)

    # 100 um sphere snow lower boundary, our own Mie table and AD solver.
    tab = np.asarray(S._host_table(), np.float64)
    radii = tab[:NR]
    i = int(np.clip(np.searchsorted(radii, 100.0) - 1, 0, NR - 2))
    f = np.log(100.0 / radii[i]) / np.log(radii[i + 1] / radii[i])
    row = lambda r: tab[OP + r * NW * 3:OP + (r + 1) * NW * 3].reshape(NW, 3)
    o = row(i) * (1 - f) + row(i + 1) * f
    ext, ab, gg = o[:, 0], o[:, 1], o[:, 2]
    x = np.zeros((NW * len(MU), 4))
    for k, mu in enumerate(MU):
        x[k * NW:(k + 1) * NW] = np.stack(
            [np.full(NW, 1e3), ab / np.maximum(ext, 1e-30), gg, np.full(NW, mu)], 1)
    xd = cp.asarray(x)
    yd = cp.empty((len(x), 10), cp.float64)
    S._module(cp.cuda.Device().id).get_function("snow_ad_probe")(
        ((len(x) + 63) // 64,), (64,), (xd, yd, np.int32(len(x))))
    y = cp.asnumpy(yd).reshape(len(MU), NW, 10)
    snow_dir = y[:, :, 0]
    snow_dif = y[0, :, 2]

    def band_mean(v):
        out = np.zeros(nband)
        for b in range(nband):
            m = bin_band == b
            w = etr_bins[m]
            # RRTMGP bands below 0.28 um or above 4 um carry little or no
            # G173 support; give them a dark-snow value (no weight reaches
            # the snow spectrum there either).
            out[b] = (v[m] * w).sum() / w.sum() if w.sum() > 0 else 0.05
        return out

    alb_dir = np.stack([band_mean(snow_dir[k]) for k in range(len(MU))])
    alb_dif = band_mean(snow_dif)

    gi = sw.gas_index
    solar = np.asarray(sw.solar_source, np.float64)
    solar = solar / solar.sum() * sw.tsi_default
    results = []
    for title, key, psfc in CASES:
        p, c = levels(profiles[key], psfc)
        plev = p[::-1].copy()                       # top first (top_at_1)
        plev[0] = max(plev[0], float(sw.press_ref.min()) * 1.0001)
        T = c["T_K"][::-1]
        play = np.sqrt(plev[1:] * plev[:-1])
        tlay = 0.5 * (T[1:] + T[:-1])
        nlay = len(play)

        def mean(v):
            v = v[::-1]
            return 0.5 * (v[1:] + v[:-1]) * 1e-6
        h2o = mean(c["h2o_ppmv"])
        vmr = np.zeros((nlay, sw.ngas + 1))
        vmr[:, gi["h2o"]] = h2o / (1 - h2o)          # AFGL ppmv of moist air -> dry
        vmr[:, gi["o3"]] = mean(c["o3_ppmv"])
        vmr[:, gi["n2o"]] = mean(c["n2o_ppmv"])
        vmr[:, gi["co"]] = mean(c["co_ppmv"])
        vmr[:, gi["ch4"]] = mean(c["ch4_ppmv"])
        vmr[:, gi["co2"]] = 415e-6
        vmr[:, gi["o2"]] = 0.2095
        vmr[:, gi["n2"]] = 0.7808
        nm = len(MU)
        ncol = 2 * nm

        def D(a):
            return cp.asarray(np.ascontiguousarray(np.broadcast_to(a, (ncol, *np.shape(a)))), dtype=R.DTYPE)
        gas = R.gas_optics(sw, D(play), D(plev), D(tlay), D(vmr))
        lay = int(np.argmin(np.abs(play - 80000.0))) if plev[-1] > 80000.0 else nlay - 1
        ones = np.zeros((ncol, nlay))
        ones[nm:, lay] = 1.0
        z = cp.zeros((ncol, nlay), R.DTYPE)
        rel = cp.full((ncol, nlay), 10.0, R.DTYPE)
        dg = cp.full((ncol, nlay), 50.0, R.DTYPE)
        unit = R.cloud_optics(cld, cp.asarray(ones, R.DTYPE), z, rel, dg)
        b500 = int(band_of(np.array([0.5]))[0])
        lwp = 10.0 / float(unit.tau[nm, lay, b500])
        cloud = R.cloud_optics(cld, cp.asarray(ones * lwp, R.DTYPE), z, rel, dg)
        # Aerosol: AOD 0.05 at 0.55 um, Angstrom 1.5, ssa 0.99, g 0.7, in the
        # lowest ~2 km (pressure-weighted), sulfate-like (Flanner 2021 s2.4).
        lamc = np.sqrt(lam_lo * lam_hi)
        dpl = np.diff(plev)
        weight = np.where(play > plev[-1] * np.exp(-2.0 / 8.0), dpl, 0.0)
        weight /= weight.sum()
        taua = 0.05 * (lamc / 0.55) ** -1.5
        ta = cp.asarray(np.broadcast_to(weight[None, :, None] * taua[None, None, :], (ncol, nlay, nband)), R.DTYPE)
        ct = cloud.tau + ta
        csa = cp.where(ct > 0, (cloud.tau * cloud.ssa + ta * 0.99) / cp.maximum(ct, 1e-30), 0)
        cg = cp.where(ct * csa > 0, (cloud.tau * cloud.ssa * cloud.g + ta * 0.99 * 0.7) / cp.maximum(ct * csa, 1e-30), 0)
        allsky = R.add_cloud_optics(sw, gas, R.CloudOpticsResult(
            cp.ascontiguousarray(ct.astype(R.DTYPE)), cp.ascontiguousarray(csa.astype(R.DTYPE)),
            cp.ascontiguousarray(cg.astype(R.DTYPE))))
        t2, s2, g2 = R.delta_scale(allsky.tau, allsky.ssa, allsky.g)
        mu = np.concatenate([MU, MU])
        ad = np.ascontiguousarray(np.concatenate([alb_dir, alb_dir])[:, gpt_band])
        af = np.ascontiguousarray(np.broadcast_to(alb_dif[gpt_band], (ncol, ngpt)))
        sdir = np.zeros((ncol, nband))
        sdif = np.zeros((ncol, nband))
        toa = np.zeros((ncol, nband))
        for b in range(nband):
            inc = np.ascontiguousarray(np.broadcast_to(np.where(gpt_band == b, solar, 0.0), (ncol, ngpt)))
            fl = R.sw_rte(t2, s2, g2, cp.asarray(mu, R.DTYPE), cp.asarray(ad, R.DTYPE),
                          cp.asarray(af, R.DTYPE), cp.asarray(inc, R.DTYPE), top_at_1=True)
            dn = cp.asnumpy(fl.flux_dn).astype(np.float64)
            dr = cp.asnumpy(fl.flux_dir).astype(np.float64)
            sdir[:, b] = dr[:, -1]
            sdif[:, b] = dn[:, -1] - dr[:, -1]
            toa[:, b] = dn[:, 0]
        expect = mu[:, None] * np.array([solar[gpt_band == b].sum() for b in range(nband)])[None]
        assert np.allclose(toa, expect, rtol=1e-4), ("RRTMGP TOA flux is not mu*S", toa[0], expect[0])
        results.append(dict(
            title=title, profile=key, psfc_hpa=psfc, nlay=nlay,
            cloud_layer_hpa=float(play[lay] / 100), lwp_gm2=lwp,
            clear_direct=(sdir[:nm] / expect[:nm]).tolist(),
            clear_diffuse=(sdif[:nm] / expect[:nm]).tolist(),
            cloudy_direct=(sdir[nm:] / expect[nm:]).tolist(),
            cloudy_diffuse=(sdif[nm:] / expect[nm:]).tolist(),
            clear_direct_broadband_mu05=float(sdir[6].sum() / expect[6].sum()),
            cloudy_diffuse_broadband_mu05=float(sdif[nm + 6].sum() / expect[nm + 6].sum())))
        print(title, "nlay", nlay, "LWP %.1f g/m2" % lwp,
              "clear direct/TOA mu=0.5: %.3f" % results[-1]["clear_direct_broadband_mu05"],
              "cloudy diffuse/TOA: %.3f" % results[-1]["cloudy_diffuse_broadband_mu05"], flush=True)

    tau_path = -np.log(trans)
    covered = np.array([etr[fine_band == b].sum() > 0 for b in range(nband)])

    def shaped(alpha_band):
        a = np.zeros_like(lam)
        m = fine_band >= 0
        a[m] = alpha_band[fine_band[m]]
        return etr * np.exp(-a * tau_path) * m

    # Trapezoid weights of the irregular G173 grid (0.5, 1 and 5 nm steps),
    # so the fitted band ratio is the same integral bin_integrate takes.
    dl = np.gradient(lam)

    def fit(ratio, b):
        m = fine_band == b
        e, t = etr[m]*dl[m], tau_path[m]
        target = np.clip(ratio, 1e-9, 1.0)
        if e.sum() <= 0 or e.sum() / e.sum() <= target:
            return 0.0
        lo, hi = 0.0, 400.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if (e * np.exp(-mid * t)).sum() / e.sum() > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    words = []
    receipt = dict(mu=MU.tolist(), layout="f32 mu[nmu]; then per case [nmu][480][direct,diffuse]", cases=[])
    etr_band = np.array([etr_bins[bin_band == b].sum() for b in range(nband)])
    for r in results:
        spec = np.zeros((len(MU), NW, 2))
        a06 = np.array([fit(r["clear_direct"][7][b], b) if covered[b] else 0.0 for b in range(nband)])
        alphas = []
        for k, mu in enumerate(MU):
            alpha = np.array([fit(r["clear_direct"][k][b], b) if covered[b] else 0.0 for b in range(nband)])
            alphas.append(alpha.tolist())
            spec[k, :, 0] = mu * bin_integrate(lam, shaped(alpha))
        base = bin_integrate(lam, shaped(a06))
        for k, mu in enumerate(MU):
            scale = np.zeros(NW)
            for b in range(nband):
                m = bin_band == b
                tot = base[m].sum()
                want = r["cloudy_diffuse"][k][b] * mu * etr_band[b]
                scale[m] = want / tot if tot > 0 else 0.0
            spec[k, :, 1] = base * scale
        spec[:, etr_bins <= 0, :] = 0.0
        # Check: every RRTMGP band integral of the shaped direct spectrum
        # matches the RRTMGP ratio within the bin-edge aggregation.
        for k, mu in enumerate(MU):
            for b in range(nband):
                m = bin_band == b
                if etr_band[b] > 0.02 * etr_band.max():
                    got = spec[k, m, 0].sum() / (mu * etr_band[b])
                    want = r["clear_direct"][k][b]
                    assert abs(got - want) < 0.02 + 0.03 * want, (r["title"], mu, b, got, want)
        spec /= spec.sum(axis=1, keepdims=True).max()
        r["alpha"] = alphas
        r["vis_fraction_direct_mu05"] = float(spec[6, centres < 0.7, 0].sum() / spec[6, :, 0].sum())
        r["vis_fraction_diffuse_mu05"] = float(spec[6, centres < 0.7, 1].sum() / spec[6, :, 1].sum())
        receipt["cases"].append(r)
        words.append(spec.astype(np.float32))
    arr = np.concatenate([MU.astype(np.float32)] + [w.reshape(-1) for w in words]).astype("<f4")
    blob = MAGIC + np.array([len(CASES), len(MU), NW], "<i4").tobytes() + arr.tobytes()
    Path(out_bin).write_bytes(blob)
    wd = tab[2 * NR + NW:2 * NR + 2 * NW]
    wi = tab[2 * NR + 2 * NW:2 * NR + 3 * NW]
    receipt.update(
        sha256=hashlib.sha256(blob).hexdigest(), words=int(arr.size), bytes=len(blob),
        g173_vis_fraction_direct=float(wd[centres < 0.7].sum() / wd.sum()),
        g173_vis_fraction_diffuse=float(wi[centres < 0.7].sum() / wi.sum()),
        rrtmgp_band_um=[[float(a), float(b)] for a, b in zip(lam_lo, lam_hi)],
        snow_boundary_albedo_dir_mu05=alb_dir[6].tolist(), snow_boundary_albedo_dif=alb_dif.tolist())
    json.dump(receipt, open(out_json, "w"), indent=1)
    print("wrote", out_bin, receipt["sha256"], "words", arr.size)
    for r in receipt["cases"]:
        print("%-26s VIS fraction direct %.3f diffuse %.3f" % (
            r["title"], r["vis_fraction_direct_mu05"], r["vis_fraction_diffuse_mu05"]))
    print("%-26s VIS fraction direct %.3f diffuse %.3f" % (
        "ASTM G173 (today)", receipt["g173_vis_fraction_direct"], receipt["g173_vis_fraction_diffuse"]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
