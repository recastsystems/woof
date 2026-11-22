"""The gf_deep_stage capture layout, ONE place.

These lists mirror the LEV_/SCA_/ISCA_ enums and the input-packing enums in
woof/core/kernels/gf_deep.cu, in the same order.  tests/test_gf_deep_cuda.py
(the device gate) and gf_host_parity.py (the no-GPU crosscheck) both import
them from here, so the two graders cannot drift apart -- and a change to the
kernel's layout has exactly one Python file to update.
"""

LEV_FIELDS = [
    "qes", "he", "hes", "qeso", "heo", "heso",
    "qes_cup", "q_cup", "he_cup", "hes_cup", "gamma_cup1",
    "t_cup", "qeso_cup", "qo_cup", "heo_cup", "heso_cup",
    "zo_cup", "po_cup", "gammao_cup", "tn_cup",
    "u_cup", "v_cup", "entr2d_a", "zu_pdf",
    "cd", "entr2d_b", "upme", "upmd", "upmeu", "upmdu",
    "hc", "uc", "vc", "hco", "dby", "dbyo", "dbyt",
    "cdd", "ddme", "ddmd", "ddmeu", "ddmdu", "mentrd2d",
    "hcdo", "ucd", "vcd", "dbydo", "c1d",
    "qcdo", "qrcdo", "pwdo",
    "qco", "qrco", "pwo", "clw_all",
    "cupclw", "cnvwt",
    "dellu", "dellv", "dellah", "dellaq", "dellaqc", "dellat",
    "xhe", "xq", "xt", "xqes", "xhes",
    "xqes_cup", "xq_cup", "xhe_cup", "xhes_cup", "gamma_cupx",
    "xt_cup", "xhc", "xdby",
    "outt_o", "outq_o", "outqc_o",
    "outt_ke", "outu_f", "outv_f",
    "dtempdz",
]

SCA_FIELDS = [
    "zws", "ztexec", "zqexec", "cap_max", "entr_rate", "sig",
    "sig_thresh", "hkb0", "hkbo0", "frh_kb",
    "up_tun", "up_alpha", "up_beta", "up_fzu",
    "dn_tun", "dn_alpha", "dn_beta", "dn_fzu",
    "bud", "beta", "edtmax", "pwevo", "bu",
    "pwavo", "psum", "psumh",
    "aa0", "aa1", "aa1_bl", "tau_ecmwf", "tau_bl", "umean",
    "edt", "edtc1", "edto", "xhkb", "xaa0", "pr7", "mconv2",
] + [f"xf{i}" for i in range(1, 17)] + [f"f{i}" for i in range(1, 11)] + [
    "xf_dicycle", "closure_n", "xmb", "pre",
]

ISCA_FIELDS = [
    "xland1", "kbmax", "k22_0", "k22_1", "kbcon_1",
    "ierr_1", "kstabi", "kstabm", "pmin_lev", "start_level",
    "ktop_pdf", "ktopdby", "kbcon_2", "ierr_2",
    "up_kbadj", "up_kklev", "up_kfinal", "dn_kbadj",
    "ktop_dbyt", "ierr_3", "kzdown", "jmin", "kdet_2",
    "ierr_4", "ierr_5", "ierr_6", "ierr_7",
    "k22x", "kbconx", "ierr2", "ierr3", "ktop", "ierr",
    "kinv1", "kinv2", "kinv3", "kinv4", "kinv5",
    "kinv_clamped",
]

IN_LEV = ["zo", "t2d", "q2d", "tn", "qo", "po", "us", "vs", "rhoi", "omeg_in"]
IN_SCA = ["ter11", "psur", "hfxi", "qfxi", "xlandi", "dx", "ccn", "dt",
          "xmbs", "fzu_up", "fzu_dn"]

# --------------------------------------------------------------------------
# the shallow stage (gf_shallow_stage), gf-shallow-*.csv order
# --------------------------------------------------------------------------
SH_LEV_FIELDS = [
    "qes", "he", "hes", "qeso", "heo", "heso",
    "qes_cup", "q_cup", "he_cup", "hes_cup", "z_cup", "p_cup",
    "gamma_cup0", "t_cup", "qeso_cup", "qo_cup", "heo_cup", "heso_cup",
    "zo_cup", "po_cup0", "gammao_cup", "tn_cup",
    "dtempdz", "entr2d_a", "cd_a", "zu_pdf", "zuo_b",
    "upme", "upmd", "cd_b", "entr2d_b",
    "hc", "hco", "dby", "dbyo", "dbyt", "qco_a", "qrco",
    "pwo", "cupclw", "qco", "cnvwt",
    "dellah", "dellaq", "dellaqc", "dellat",
    "xhe", "xq", "xt", "xqes", "xhes",
    "xqes_cup", "xq_cup", "xhe_cup", "xhes_cup", "gamma_cupx",
    "xt_cup", "po_cupx",
    "xhc", "xdby", "xzu",
    "zuo", "outt", "outq", "outqc",
]

SH_SCA_FIELDS = [
    "buo_flux", "zws", "ztexec", "zqexec", "cap_max",
    "entr_rate", "hkb0", "hkbo0", "hkbo_1", "hkb_2",
    "sh_tun", "sh_alpha", "sh_beta", "sh_fzu", "qaver",
    "aa0", "aa1", "xhkb", "xaa0", "xkshal",
    "xff1", "xff2", "xff3", "blqe", "trash_kb", "xmbmax",
    "xmb", "xmb_out", "pre",
]

SH_ISCA_FIELDS = [
    "xland1", "kbmax", "k22_0", "k22_1", "kbcon_1", "ierr_1",
    "kstabi", "kinv1", "kinv2", "kinv3", "kinv4", "kinv5",
    "kstabi_oob", "start_level", "ierr_231", "ktop_0",
    "ktop_pdf", "kbcon_2", "ierr_2", "sh_kbadj", "sh_kfinal",
    "ktop_3", "k22_3", "ki_dbyt", "ktop_4", "ierr_4", "ierr_5",
    "ierr_6", "k22", "kbcon", "ktop", "ierr",
]

SH_IN_LEV = ["zo", "t2d", "q2d", "tshall", "qshall", "po", "dhdt", "rhoi"]
SH_IN_SCA = ["ter11", "psur", "hfxi", "qfxi", "xlandi", "dt", "fzu_sh"]

# --------------------------------------------------------------------------
# the whole-driver stage (gf_gfdrv_stage), gf-levels/gf-surface order plus
# the post-neg_check seam run_cup_gf.F90 captures
# --------------------------------------------------------------------------
DRV_LEV_FIELDS = [
    "rthcuten", "rqvcuten", "rqccuten", "rqicuten", "dudt_phy", "dvdt_phy",
    "gdc", "gdc2",
    "outt", "outq", "outqc", "outu", "outv",
    "outts", "outqs", "outqcs",
]

DRV_SCA_FIELDS = [
    "raincv", "pratec", "htop", "hbot", "xmb_shallow", "pret", "prets",
    "cuten", "cutens",
]

DRV_ISCA_FIELDS = [
    "ktop_deep", "k22_shallow", "kbcon_shallow", "ktop_shallow",
    "kbcon", "ktop",
]

DRV_IN_LEV = ["u", "v", "w", "t", "qv", "p", "pi", "rho", "dz8w", "p8w",
              "rthften", "rqvften", "rthraten", "rthblten", "rqvblten"]
#: gf.cu's DINS_ order.  The last three are the per-column ``fzu`` override
#: ("<= 0 means compute"), added at 2.7.0 when gamma stopped reproducing
#: glibc's tgammaf (docs/gf_gamma_known_delta.md).  The shipped forecast path
#: leaves them 0; a run graded against a WRF capture pins them from it, which
#: is what ``drv_scalar_inputs(fixture, pin_fzu=True)`` below builds.
DRV_IN_SCA = ["ht", "hfx", "qfx", "xland", "dt", "dx",
              "fzu_up", "fzu_dn", "fzu_sh"]

#: The three DRV_IN_SCA slots that are NOT read from gf-surface.csv.
DRV_FZU_SLOTS = ("fzu_up", "fzu_dn", "fzu_sh")


def captured_fzu(fixture):
    """WRF's own ``fzu`` words for every GFDRV column, as float32 arrays.

    Returns ``(up, dn, sh)``, each shaped ``(ncol,)`` in ``fixture.key``
    order.  The deep pair comes from ``gf-deep-surface.csv`` (keyed by the
    same ``(case, idx, arm)`` triple the fixture is ordered by) and the
    shallow one from ``gf-shallow-surface.csv``, which is keyed by CASE
    alone because CUP_gf_sh takes no dx.

    WHY THIS EXISTS.  WOOF-ArWen's gamma is correctly rounded and glibc's is not,
    so the kernel's computed ``fzu`` is deliberately not WRF's
    (docs/gf_gamma_known_delta.md).  Feeding these words back through the
    ``fzu_override`` slots is what lets the parity suites keep grading the
    other ~4,000 transcribed lines of GFDRV bitwise -- exactly what the CPU
    reference has done since the port landed
    (tests/test_gf_driver_parity.py).
    """
    import csv

    import numpy as np

    from woof.verify.gf_oracle import GF_ORACLE_DIR

    deep = {}
    with (GF_ORACLE_DIR / "gf-deep-surface.csv").open(
            newline="", encoding="ascii") as fh:
        for r in csv.DictReader(fh):
            deep[(int(r["case"]), int(r["idx"]), int(r["arm"]))] = (
                np.float32(r["up_fzu"]), np.float32(r["dn_fzu"]))
    shal = {}
    with (GF_ORACLE_DIR / "gf-shallow-surface.csv").open(
            newline="", encoding="ascii") as fh:
        for r in csv.DictReader(fh):
            shal[int(r["case"])] = np.float32(r["sh_fzu"])

    n = fixture.ncol
    up = np.zeros(n, dtype=np.float32)
    dn = np.zeros(n, dtype=np.float32)
    sh = np.zeros(n, dtype=np.float32)
    for i, row in enumerate(fixture.key):
        trip = tuple(int(v) for v in row)
        up[i], dn[i] = deep[trip]
        sh[i] = shal[trip[0]]
    return up, dn, sh


def drv_scalar_inputs(fixture, pin_fzu: bool):
    """``scin`` for ``gf_gfdrv_stage``, shaped ``(ncol, len(DRV_IN_SCA))``.

    ``pin_fzu=False`` reproduces the SHIPPED forecast path: the three
    override slots are 0 and the kernel computes ``fzu`` from WOOF-ArWen's own
    correctly rounded gamma.  ``pin_fzu=True`` fills them with WRF's captured
    words, which is the only way a GFDRV-boundary comparison against the WRF
    capture can be bitwise.
    """
    import numpy as np

    gs = fixture.surface
    out = np.zeros((fixture.ncol, len(DRV_IN_SCA)), dtype=np.float32)
    for j, name in enumerate(DRV_IN_SCA):
        if name in DRV_FZU_SLOTS:
            continue
        out[:, j] = gs[name].astype(np.float32)
    if pin_fzu:
        up, dn, sh = captured_fzu(fixture)
        out[:, DRV_IN_SCA.index("fzu_up")] = up
        out[:, DRV_IN_SCA.index("fzu_dn")] = dn
        out[:, DRV_IN_SCA.index("fzu_sh")] = sh
    return np.ascontiguousarray(out)


def reference_constant_words():
    """The GFC table's words, re-derived from the CPU reference.

    Order must match the table in woof/core/kernels/gf.cu; both the
    source-parse test and the device-dump test compare against this."""
    import numpy as np

    from woof.verify import gf_deep_ref as R

    F = np.float32
    vals = [
        F(0.0), F(1.0), F(2.0), F(0.5), F(-0.5), R.G, R.CP, R.XLV, R.R_V,
        R.C1, R.FRH_THRESH, R.RH_THRESH,
        F(F(F(1.0) - R.FRH_THRESH) * F(F(1.0) - R.FRH_THRESH)),
        R.BETAJB, R.FLUXTUNE, R.PGCD, R._TINY32,
        F(273.155), F(-20.0), F(273.16), F(373.16),
        F(-9.09718), F(3.56654), F(0.876793), F(-7.90298), F(5.02808),
        F(1.3816e-07), F(11.344), F(0.0081328), F(-3.49149), F(10.0),
        R._LOG10, R._LOG6_OVER_LOG10, R._LOG1013_OVER_LOG10,
        F(0.622), F(1.0e-8), F(1.0e-16), F(9.81), F(1004.0), F(2.5e06),
        F(R.XLV / R.CP), F(F(1.0) / R.XLV),
        F(0.608), F(0.41), F(1.2), F(0.3333), F(75.0), F(20.0), F(25.0),
        F(16.0), F(1.5), F(0.0001), F(7.0e-5), F(3.0e-6), F(0.2), F(3.14),
        F(4000.0), F(1000.0), F(1000.0), F(0.1), F(20.0), F(150.0),
        F(200.0), F(1.3), F(1.0e-9), F(1.0e-6), F(0.9), F(0.4), F(0.013),
        F(0.8), F(2.5), F(4.0), F(2.0), F(4000.0), F(0.6), F(5.0),
        F(0.05), F(0.0015), F(0.02), F(0.4), F(0.015), F(0.004),
        F(273.15), F(0.07), F(7.0), F(1.0061), F(1.23e-2), F(1000.0),
        F(4.0), F(0.1), F(190.0), F(100.0), F(1.0e3), F(1.591), F(0.639),
        F(0.0953), F(0.00496), F(3.281e-3), F(0.02), F(3.0),
        F(0.96729352), F(-0.70034167), F(0.162179896), F(-1.2569798e-2),
        F(4.2772e-4), F(-5.44e-6), F(2.4), F(0.2), F(0.1), F(-0.01),
        F(1.0e-2), F(1.0e-5), F(1.0e-3), F(12.0), F(300.01), F(86400.0),
        F(1.0e9), F(100.0), F(300.0), F(1.0e-08), F(200.0), F(0.01),
        F(150.0),
        # appended with the shallow/driver halves
        F(148.01), F(3000.0), F(125.0), F(9.0e-5), F(2.3), F(0.03),
        F(258.0),
    ]
    import numpy as _np
    return _np.array(
        [int(_np.float32(v).view(_np.uint32)) for v in vals],
        dtype=_np.uint32)
