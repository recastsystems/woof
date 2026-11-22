#!/usr/bin/env python3
"""Assemble the WP-L7 ArWen-vs-WRF comparison table.

The WOOF-ArWen column is transcribed from committed receipts on
feature/les-integration @ 85c24d6d; every entry carries its source so a reader
can check it.  The WRF column comes from score_wrf_les.py, which imports no
woof code.

No band is cut here.  This program measures and reports; per spec D-L11 a band
is ratified by the owner from committed receipts, and never derived from the
run being scored.
"""
import sys
import json
import os

# ---------------------------------------------------------------- ArWen side
# Values transcribed from the CURRENT ArWen receipts,
# docs/internal/receipts/les/cbl-2026-08-02/{km2,km3}/ on
# feature/les-integration.  The five loose receipts one directory up are an
# earlier code state; docs/public/LES.md:78-81 says explicitly that they do NOT
# correspond to the current table, and an earlier revision of this lane wrongly
# compared against them.
#
# All entries are FINAL-SAMPLE unless the note says otherwise, because that is
# how the ArWen case module computes them.
ARWEN = {
    "match_km3_100m": {
        "_src": "cbl-2026-08-02/km3/cbl_km3_receipt.json",
        "zi_m": (1847.528, "final sample"),
        "wth_res_max_over_qs": (0.843861, "final sample"),
        "entrainment_min_over_qs": (-0.165444,
                                    "wth_total_min_over_qs_mean, 20 samples, "
                                    "sd 0.02966 = 17.9%"),
        "tke_res_max": (2.118383, "final sample"),
        "mass_drift_rel": (2.231955e-10, "whole run"),
        "resolved_fraction_ml": (None, "km_opt=3: no prognostic SGS TKE"),
    },
    "match_km2_100m": {
        "_src": "cbl-2026-08-02/km2/cbl_km2_receipt.json",
        "zi_m": (1728.635, "final sample"),
        "wth_res_max_over_qs": (0.837984, "final sample"),
        "entrainment_min_over_qs": (-0.143392,
                                    "wth_total_min_over_qs_mean, 20 samples, "
                                    "sd 0.02604 = 18.2%"),
        "resolved_fraction_ml": (0.893885, "final sample, index band"),
        "tke_res_max": (2.190961, "final sample"),
        "mass_drift_rel": (6.867556e-11, "whole run"),
        "e_sgs_ml_mean": (0.166157, "e_sgs_volume_final, volume mean"),
    },
    # 50 m receipts committed 2026-08-02 at 8d083ef4 after this lane flagged
    # their absence. That lane re-ran both on the current tree rather than
    # committing the older node copies, reproduced every published number, and
    # got byte-identical physics fields back -- so the pair carries its own
    # dual-run corruption check. Same-instrument comparison is therefore
    # available at 50 m and the scalar fallback is no longer needed.
    "match_km3_50m": {
        "_src": "cbl-2026-08-02/km3fine/cbl_km3fine_receipt.json",
        "zi_m": (1787.904, "final sample"),
        "wth_res_max_over_qs": (0.9040047, "final sample"),
        "entrainment_min_over_qs": (-0.1330549,
                                    "window mean, sd 0.02799 = 21.0%"),
        "tke_res_max": (2.325391, "final sample"),
        "mass_drift_rel": (2.2253017e-08, "whole run"),
        "resolved_fraction_ml": (None, "km_opt=3: e_sgs identically zero, "
                                       "receipt is null deliberately"),
    },
    "match_km2_50m": {
        "_src": "cbl-2026-08-02/km2fine/cbl_km2fine_receipt.json",
        "zi_m": (1814.357, "final sample"),
        "wth_res_max_over_qs": (0.8963442, "final sample"),
        "entrainment_min_over_qs": (-0.1292632,
                                    "window mean, sd 0.02410 = 18.6%"),
        "resolved_fraction_ml": (0.9320623, "final sample, index band"),
        "tke_res_max": (2.531422, "final sample"),
        "mass_drift_rel": (2.0654164e-08, "whole run"),
        "e_sgs_ml_mean": (0.1154856, "e_sgs_volume_final, volume mean"),
    },
}

ARWEN_REFINE = {
    "km_opt=2 resolved TKE fraction": (0.894, 0.932),
    "km_opt=2 resolved flux fraction": (0.838, 0.896),
    "km_opt=3 resolved flux fraction": (0.844, 0.904),
}

# ArWen's OWN spread at fixed configuration, from two committed km_opt=2
# receipts that share nx/ny/nz/dx/ztop/dt/km_opt/run_seconds exactly
# (cbl_km2_receipt.json and cbl_km2budget_receipt.json).  They are not
# bit-identical runs -- their mass drifts differ by 4x -- so the gap between
# them is a floor on what "agreement" can mean for a single-sample statistic.
# Every entry is a final-sample value.
ARWEN_SELF_SPREAD_KM2 = {
    "zi_m": (1689.30, 1768.12),
    "wth_res_max_over_qs": (0.86560, 0.83581),
    "wth_total_min_over_qs": (-0.09342, -0.15351),
    "tke_res_max": (2.20383, 2.21283),
    "resolved_fraction_ml": (0.89188, 0.89459),
    "mass_drift_rel": (1.3735e-10, 3.4338e-11),
    "w_max": (7.55684, 7.42687),
}
# The km_opt=3 receipt (cbl_les2h_receipt.json) is a single run, so no
# equivalent spread exists for it; its final-sample z_i is 1807.75 m against
# the 1689 m quoted as the 30-minute-window value in the same receipt's prose.

ORDER = ["zi_m", "wstar", "theta_ml", "wmax_over_wstar", "flux_fit_intercept",
         "flux_fit_slope", "flux_fit_rms", "entrainment_min_over_qs",
         "wth_res_max_over_qs", "resolved_fraction_ml", "tke_res_max",
         "e_sgs_ml_mean", "updraft_fraction_low", "mass_drift_rel"]


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    a = abs(v)
    if a != 0 and (a < 1e-3 or a >= 1e5):
        return "%.3e" % v
    return "%.4g" % v


# ArWen reports some of these from a 30-minute window (its receipt prose) and
# others from the final sample (its receipt JSON).  The WRF partner has to use
# the SAME convention per metric or the comparison silently compares an average
# against an instantaneous value.
FINAL_SAMPLE_KEYS = {"wth_res_max_over_qs", "resolved_fraction_ml"}


def wrf_value(w, key):
    """Pull the WRF counterpart under WOOF-ArWen's own convention for that metric."""
    fs = w.get("final_sample", {})
    if key == "resolved_fraction_ml":
        if w.get("km_opt") != 2:
            return None
        return fs.get("resolved_fraction_ml_trunc")
    if key in FINAL_SAMPLE_KEYS:
        return fs.get(key, w.get(key))
    if key == "tke_res_max":
        return w.get("tke_res_max_trunc")
    return w.get(key)


def main():
    score_dir = sys.argv[1]
    out = sys.argv[2]
    lines = []
    A = lines.append
    A("# WP-L7 comparison: WOOF-ArWen CBL vs WRF v4.6.1 em_les oracle")
    A("")
    A("WRF column: this lane's fresh em_les build, scored by score_wrf_les.py")
    A("(no woof import). WOOF-ArWen column: committed receipts on")
    A("feature/les-integration @ 85c24d6d.")
    A("")
    A("No pass band is cut here. Differences are reported, not judged.")

    got = {}
    for rid in ("match_km3_100m", "match_km2_100m", "match_km3_50m",
                "match_km2_50m"):
        p = os.path.join(score_dir, rid + ".json")
        if os.path.exists(p):
            got[rid] = json.load(open(p))

    for rid, w in got.items():
        a = ARWEN.get(rid, {})
        A("")
        A("## %s  (km_opt=%s, dx=%s m, %sx%sx%s)"
          % (rid, w.get("km_opt"), w.get("dx_m"), w.get("nx"), w.get("ny"),
             w.get("nz")))
        A("")
        A("WOOF-ArWen source: %s" % a.get("_src", "-"))
        A("")
        A("| metric | WOOF-ArWen | WRF em_les | diff | note |")
        A("|---|---|---|---|---|")
        for key in ORDER:
            av = a.get(key)
            aval, anote = (av if isinstance(av, tuple) else (av, ""))
            wval = wrf_value(w, key)
            if aval is None and wval is None:
                continue
            if aval is not None and wval is not None:
                d = wval - aval
                dtxt = "%+.4g" % d
                if abs(aval) > 1e-12:
                    dtxt += " (%+.1f%%)" % (100.0 * d / abs(aval))
            else:
                dtxt = "-"
            A("| %s | %s | %s | %s | %s |"
              % (key, fmt(aval), fmt(wval), dtxt, anote))
        pw = w.get("power", {})
        A("")
        A("Power receipt (2.4): T* = %.0f s, window = %.2f T*, spin-up = %.2f T*,"
          " T_int = %.0f s, N_t = %.2f, L_x/z_i = %.2f, N_h = %.1f, "
          "frames in window = %d"
          % (pw.get("tstar_s", float("nan")),
             pw.get("window_over_tstar", float("nan")),
             pw.get("spinup_over_tstar", float("nan")),
             pw.get("T_int_s", float("nan")), pw.get("N_t", float("nan")),
             pw.get("L_x_over_zi", float("nan")), pw.get("N_h", float("nan")),
             w.get("n_frames_in_window", 0)))
        sp = w.get("spectrum", {})
        A("")
        A("Spectrum at z = %.0f m: E(k) slope %.3f over bins %s, "
          "Parseval residual %.2e"
          % (sp.get("level_z_m", float("nan")),
             sp.get("slope_Ek", float("nan")), sp.get("fit_bins"),
             sp.get("parseval_residual_rel", float("nan"))))
        A("")
        A("Unstagger-convention sensitivity (WRF side, same run): resolved "
          "fraction %s (truncation, WOOF-ArWen's) vs %s (averaging); tke_res_max "
          "%s vs %s."
          % (fmt(w.get("resolved_fraction_ml_trunc")),
             fmt(w.get("resolved_fraction_ml_avg")),
             fmt(w.get("tke_res_max_trunc")),
             fmt(w.get("tke_res_max_avg"))))

    A("")
    A("## Grid refinement: does WRF move the same way?")
    A("")
    A("| quantity | ArWen 100 m | ArWen 50 m | WOOF-ArWen delta | WRF 100 m | "
      "WRF 50 m | WRF delta |")
    A("|---|---|---|---|---|---|---|")

    def refine_row(label, key, cid, fid):
        ac = ARWEN.get(cid, {}).get(key, (None, ""))[0]
        af = ARWEN.get(fid, {}).get(key, (None, ""))[0]
        wc = wrf_value(got[cid], key) if cid in got else None
        wf = wrf_value(got[fid], key) if fid in got else None
        ad = ("%+.4f" % (af - ac)) if (ac is not None and af is not None) else "-"
        wd = ("%+.4f" % (wf - wc)) if (wc is not None and wf is not None) else "-"
        A("| %s | %s | %s | %s | %s | %s | %s |"
          % (label, fmt(ac), fmt(af), ad, fmt(wc), fmt(wf), wd))

    refine_row("km_opt=2 resolved TKE fraction", "resolved_fraction_ml",
               "match_km2_100m", "match_km2_50m")
    refine_row("km_opt=2 resolved flux fraction", "wth_res_max_over_qs",
               "match_km2_100m", "match_km2_50m")
    refine_row("km_opt=3 resolved flux fraction", "wth_res_max_over_qs",
               "match_km3_100m", "match_km3_50m")

    # WRF-only second refinement pair, 200 m -> 100 m.  ArWen has no runs at
    # 200 m, so this column tests the same property (does the partition move
    # toward resolved as the grid refines?) at a different pair of spacings.
    A("")
    A("### WRF-only second refinement pair, 200 m -> 100 m")
    A("")
    A("WOOF-ArWen published nothing at 200 m, so this tests the property rather "
      "than reproducing WOOF-ArWen's numbers.")
    A("")
    A("| quantity | WRF 200 m | WRF 100 m | delta | moves toward resolved? |")
    A("|---|---|---|---|---|")
    for label, key, c2, c1 in (
            ("km_opt=2 resolved TKE fraction", "resolved_fraction_ml",
             "match_km2_200m", "match_km2_100m"),
            ("km_opt=2 resolved flux fraction", "wth_res_max_over_qs",
             "match_km2_200m", "match_km2_100m"),
            ("km_opt=3 resolved flux fraction", "wth_res_max_over_qs",
             "match_km3_200m", "match_km3_100m")):
        v2 = wrf_value(got[c2], key) if c2 in got else None
        v1 = wrf_value(got[c1], key) if c1 in got else None
        if v2 is None or v1 is None:
            A("| %s | %s | %s | - | - |" % (label, fmt(v2), fmt(v1)))
        else:
            A("| %s | %s | %s | %+.4f | %s |"
              % (label, fmt(v2), fmt(v1), v1 - v2,
                 "yes" if v1 > v2 else "NO"))
    A("")
    A("WOOF-ArWen published refinement pairs (docs/public/LES.md:89-97): " +
      "; ".join("%s %.3f -> %.3f" % (k, v[0], v[1])
                for k, v in ARWEN_REFINE.items()))

    # ------------------------------------------------- ArWen's own spread
    A("")
    A("## Scale check: WOOF-ArWen's own spread at fixed configuration")
    A("")
    A("Two committed WOOF-ArWen km_opt=2 receipts share nx/ny/nz/dx/ztop/dt/km_opt/"
      "run_seconds exactly and are not bit-identical (their mass drifts differ "
      "by 4x). The gap between them is a floor on what agreement can mean for "
      "a single-sample statistic, and the WRF column is this lane's "
      "final-sample value for the same run so the three are like-for-like.")
    A("")
    A("| metric (final sample) | WOOF-ArWen km2 A | WOOF-ArWen km2 B | WOOF-ArWen self-gap | "
      "WRF km2 100 m | inside WOOF-ArWen's own range? |")
    A("|---|---|---|---|---|---|")
    w2 = got.get("match_km2_100m", {})
    fs = w2.get("final_sample", {})
    fs_map = {"zi_m": fs.get("zi_m"),
              "wth_res_max_over_qs": fs.get("wth_res_max_over_qs"),
              "wth_total_min_over_qs": fs.get("wth_total_min_over_qs"),
              "tke_res_max": fs.get("tke_res_max_trunc"),
              "resolved_fraction_ml": fs.get("resolved_fraction_ml_trunc"),
              "mass_drift_rel": w2.get("mass_drift_rel")}
    for k, (va, vb) in ARWEN_SELF_SPREAD_KM2.items():
        wv = fs_map.get(k)
        gap = abs(vb - va)
        if wv is None:
            inside = "-"
        else:
            inside = "yes" if min(va, vb) <= wv <= max(va, vb) else "NO"
        A("| %s | %s | %s | %s | %s | %s |"
          % (k, fmt(va), fmt(vb), fmt(gap), fmt(wv), inside))
    A("")
    A("Note: resolved_fraction_ml is the metric that barely moves between "
      "WOOF-ArWen's two runs (0.8919 vs 0.8946, a gap of 0.0027), which is why it "
      "is the one that can carry a refinement claim; the entrainment minimum "
      "moves by 64% between the same two runs and cannot.")

    txt = "\n".join(lines) + "\n"
    open(out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
