// Keep large initialization subtrees out of callers to bound NVRTC memory.
// Only call boundaries change; the transcribed arithmetic stays in place.
// Clean WRF 4.6.1 column transcription. No third-party runtime is linked.
#ifndef GD_DEVICE
#define GD_DEVICE __device__
#endif
typedef float real;
typedef double doublereal;
typedef int integer;
typedef int logical;
typedef int ftnlen;
#define TRUE_ 1
#define FALSE_ 0
typedef struct {
    integer ltop[3];
    real ptopc[4], xduo3n[2997]	,
	    xdo3n2[2997]	, xdo3n3[2997]	, xdo3n4[2997]	,
	    prgfdl[81], ab15wd, sko2d, skc1r,
	    sko3r, em1[5040]	,
	    em1wde[5040]	, table1[5040]	, table2[5040]	,
	    table3[5040]	, em3[5040]	, source[420]	,
	    dsrce[420]	, t1[5040], t2[
	    5040], t4[5040], em1v[5040], em1vw[5040],
	    em3v[5040], axsd[310], rsqr, co251[6724]
	    , cdt51[6724]	,
	    c2d51[6724]	, co258[6724]	, cdt58[6724]	,
	    c2d58[6724]	, tbpvs[7501],
	    c1xpvs, c2xpvs, stemp[82], gtemp[82],
	    co231[82], co238[82], c2d31[82], c2d38[82]
	    , cdt31[82], cdt38[82], co271[82], co278[
	    82], c2d71[82], c2d78[82], cdt71[82],
	    cdt78[82], co2m51[81], co2m58[81], cdtm51[
	    81], cdtm58[81], c2dm51[81], c2dm58[81];
} GDTable;
typedef struct {
    real r1, rsin1, rcos1, rcos2;
} GDClock;

typedef struct {
    real co2in1_co2800[6724];
    real co2in1_co2801[6724];
    real co2in1_co2802[6724];
    real co2in1_d2ct10[6724];
    real co2in1_dcdt8[6724];
    real co2in1_co2po[6724];
    real co2in1_dcdt10[6724];
    real co2in1_co2po1[6724];
    real co2in1_co2po2[6724];
    real co2ins_co2800[6724];
    real co2ins_co2801[6724];
    real co2ins_co2802[6724];
    real co2ins_d2ct10[6724];
    real co2ins_dcdt8[6724];
    real co2ins_co2po[6724];
    real co2ins_dcdt10[6724];
    real co2ins_co2po1[6724];
    real co2ins_co2po2[6724];
    real co2int_p[82];
    real co2int_pd[83];
    real co2int_ps[82];
    real co2int_pds[83];
    real co2int_plm[82];
    real co2int_trns[6724];
    real co2o3_t20[20172];
    real co2o3_t21[20172];
    real co2o3_t22[20172];
    real co2o3_t41[166];
    real co2o3_t42[82];
    real co2o3_t43[166];
    real co2o3_t44[82];
    real co2o3_t23[20172];
    real co2o3_co2d1d[486];
    real co2o3_co2d2d[40344];
    real co2o3_co2iq2[40344];
    real co2o3_co2iq3[40344];
    real co2o3_sgtemp[164];
    real co2o3_siglnu[81];
    real co2o3_sglvnu[82];
    real co2ptz_a[328];
    real co2ptz_pd[83];
    real co2ptz_alt[82];
    real co2ptz_temp[82];
    real co2ptz_gtemp[82];
    real co2ptz_press[82];
    real co2ptz_tmpint[328];
    real co2ptz_prsint[82];
    real coeint_etap[109];
    real coeint_path0[109];
    real o3clim_equiv_9[11988];
    real o3clim_p[81];
    real o3clim_ph[82];
    real o3clim_ro31[410];
    real o3clim_ro32[410];
    real o3clim_pstd[81];
    real o3clim_rstd[81];
    real o3clim_duo3n[779];
    real o3clim_o35deg[2997];
    real o3clim_rdata[81];
    real o3clim_tempn[19];
    real o3clim_dduo3n[1539];
    real quadsr_wt[101];
    real sigp_q[81];
    real sigp_qmh[82];
    real sigp_pdt[83];
    real sigp_plm[82];
} GDScratch;
typedef struct {
    real pa[109], transa[11881]	,
	    xa[109], ca[109], eta[109], sexpv[109],
	    core, uexp, sexp, p1, p2,
	    trnslo;
    integer ia, ja, n, status;
    real raw[106929]	;
    real ozone_data[1053];
} GDInitState;
#define gd_init_state__1 (*init)
#define gd_tables__1 (*gt)
#define gd_clock__1 (*clock)


















GD_DEVICE real antemp_(integer *l, real *z__, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init)
{


    real zb[70]	 = { 2.f,3.f,16.5f,21.5f,45.f,51.f,
	    70.f,100.f,200.f,300.f,1.5f,6.5f,13.f,18.f,26.f,36.f,48.f,50.f,
	    70.f,100.f,3.f,10.f,19.f,25.f,32.f,44.5f,50.f,71.f,98.f,200.f,
	    4.7f,10.f,23.f,31.8f,44.f,50.2f,69.2f,100.f,102.f,103.f,1.f,3.2f,
	    8.5f,15.5f,25.f,30.f,35.f,50.f,70.f,100.f,11.f,20.f,32.f,47.f,
	    51.f,71.f,84.852f,90.f,91.f,92.f,11.f,20.f,32.f,47.f,51.f,71.f,
	    84.852f,90.f,91.f,92.f };
    real c__[77]	 = { -6.f,-4.f,-6.7f,4.f,2.2f,1.f,
	    -2.8f,-.27f,0.f,0.f,0.f,-4.f,-6.f,-6.5f,0.f,1.2f,2.2f,2.5f,0.f,
	    -3.f,-.25f,0.f,-3.5f,-6.f,-.5f,0.f,.4f,3.2f,1.6f,-1.8f,-.7f,0.f,
	    0.f,-5.3f,-7.f,0.f,1.4f,3.f,.7f,-3.3f,-.2f,0.f,0.f,0.f,3.f,-3.2f,
	    -6.8f,0.f,-.6f,1.f,1.2f,2.5f,-.7f,-1.2f,0.f,-6.5f,0.f,1.f,2.8f,
	    0.f,-2.8f,-2.f,0.f,0.f,0.f,0.f,-6.5f,0.f,1.f,2.8f,0.f,-2.8f,-2.f,
	    0.f,0.f,0.f,0.f };
    real delta[70]	 = { .5f,.5f,.3f,.5f,1.f,1.f,
	    1.f,1.f,1.f,1.f,.5f,1.f,.5f,.5f,1.f,1.f,2.5f,.5f,1.f,1.f,.5f,.5f,
	    1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,.5f,.3f,1.f,1.f,2.f,1.f,1.5f,1.f,
	    1.f,1.f,.4f,1.5f,.3f,.5f,1.f,1.f,1.f,1.f,1.f,1.f,.3f,1.f,1.f,1.f,
	    1.f,1.f,1.f,1.f,1.f,1.f,.3f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f };
    real tstar[7] = { 300.f,294.f,272.2f,287.f,257.1f,288.15f,288.15f }
	    ;


    integer i__1;
    real ret_val, r__1;





    integer n;
    real x, y, fac, temp, expo, expp, zlog;
    integer nlast;
    real faclog;

    nlast = 10;
    temp = tstar[*l - 1] + c__[*l * 11 - 11] * *z__;
    i__1 = nlast;
    for (n = 1; n <= i__1; ++n) {
	expo = (*z__ - zb[n + *l * 10 - 11]) / delta[n + *l * 10 - 11];
	expp = zb[n + *l * 10 - 11] / delta[n + *l * 10 - 11];
	if (abs(expo) - 50.f <= 0.f) {
	    goto L23;
	} else {
	    goto L24;
	}
L23:
	x = (real)gd_exp(expo);
	y = x + 1.f / x;
	zlog = (real)gd_log(y);
	goto L25;
L24:
	zlog = abs(expo);
L25:
	if (expp - 50.f <= 0.f) {
	    goto L27;
	} else {
	    goto L28;
	}
L27:
	fac = (real)gd_exp(expp) + (real)gd_exp((real)(-expp));
	faclog = (real)gd_log(fac);
	goto L29;
L28:
	faclog = expp;
L29:
	r__1 = (c__[n + 1 + *l * 11 - 12] - c__[n + *l * 11 - 12]) * .5f;
	temp += r__1 * (*z__ + delta[n + *l * 10 - 11] * (zlog - faclog));

    }
    ret_val = temp;
    return ret_val;
}




















 GD_DEVICE GD_NOINLINE int co2in1_(real *t20, real *t21, real *t66, integer *iq,
	integer *l, integer *lp1, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init)
{

    integer i__1, i__2;
    real r__1;


    integer i__, j;

    real c1;
    real c2x;
    real *co2800=work->co2in1_co2800;
    real *co2801=work->co2in1_co2801;
    real *co2802=work->co2in1_co2802;
    real d2ct8[
	    6724];
    integer itin;

    real *d2ct10=work->co2in1_d2ct10;
    real *dcdt8=work->co2in1_dcdt8;
    real *co2po=work->co2in1_co2po;
    integer itin1;

    real *dcdt10=work->co2in1_dcdt10;
    real *co2po1=work->co2in1_co2po1;
    real *co2po2=work->co2in1_co2po2;


    t66 -= 82;
    t21 -= 6807;
    t20 -= 6807;


    itin = 20;
    itin1 = 21;
    i__1 = *lp1;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *lp1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    co2po[i__ + j * 82 - 83] = t20[i__ + (j + 82) * 82];
	    if (*iq == 5) {
		goto L300;
	    }
	    co2po1[i__ + j * 82 - 83] = t20[i__ + (j + 164) * 82];
	    co2po2[i__ + j * 82 - 83] = t20[i__ + (j + 246) * 82];
L300:
	    ;
	}
    }
    i__2 = *lp1;
    for (j = 1; j <= i__2; ++j) {
	i__1 = *lp1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    co2800[i__ + j * 82 - 83] = t21[i__ + (j + 82) * 82];
	    if (*iq == 5) {
		goto L301;
	    }
	    co2801[i__ + j * 82 - 83] = t21[i__ + (j + 164) * 82];
	    co2802[i__ + j * 82 - 83] = t21[i__ + (j + 246) * 82];
L301:
	    ;
	}
    }
    if (*iq == 1) {
	c1 = 1.5f;
	c2x = .5f;
    }
    if (*iq == 2) {
	c1 = 1.6363636363636365f;
	c2x = .63636363636363635f;
    }
    if (*iq == 3) {
	c1 = 1.3846153846153846f;
	c2x = .38461538461538464f;
    }
    if (*iq == 4) {
	c1 = 1.8f;
	c2x = .8f;
    }
    if (*iq == 5) {
	c1 = 1.f;
	c2x = 0.f;
    }
    i__1 = *lp1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = *lp1;
	for (j = 1; j <= i__2; ++j) {
	    co2po[j + i__ * 82 - 83] = c1 * co2po[j + i__ * 82 - 83] - c2x;
	    co2800[j + i__ * 82 - 83] = c1 * co2800[j + i__ * 82 - 83] - c2x;
	    if (*iq == 5) {
		goto L1021;
	    }
	    co2po1[j + i__ * 82 - 83] = c1 * co2po1[j + i__ * 82 - 83] - c2x;
	    co2801[j + i__ * 82 - 83] = c1 * co2801[j + i__ * 82 - 83] - c2x;
	    co2po2[j + i__ * 82 - 83] = c1 * co2po2[j + i__ * 82 - 83] - c2x;
	    co2802[j + i__ * 82 - 83] = c1 * co2802[j + i__ * 82 - 83] - c2x;
L1021:
	    ;
	}
    }
    if (*iq >= 1 && *iq <= 4) {
	i__2 = *lp1;
	for (j = 1; j <= i__2; ++j) {
	    i__1 = *lp1;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		r__1 = (co2801[i__ + j * 82 - 83] - co2802[i__ + j * 82 - 83])
			 * .02f;
		dcdt8[i__ + j * 82 - 83] = r__1 * 100.f;
		r__1 = (co2po1[i__ + j * 82 - 83] - co2po2[i__ + j * 82 - 83])
			 * .02f;
		dcdt10[i__ + j * 82 - 83] = r__1 * 100.f;
		r__1 = (co2801[i__ + j * 82 - 83] + co2802[i__ + j * 82 - 83]
			- co2800[i__ + j * 82 - 83] * 2.f) * .0016f;
		d2ct8[i__ + j * 82 - 83] = r__1 * 1e3f;
		r__1 = (co2po1[i__ + j * 82 - 83] + co2po2[i__ + j * 82 - 83]
			- co2po[i__ + j * 82 - 83] * 2.f) * .0016f;
		d2ct10[i__ + j * 82 - 83] = r__1 * 1e3f;

	    }
	}
    }
    i__1 = *l;
    for (i__ = 1; i__ <= i__1; ++i__) {
	t66[i__ + 162] = co2po[i__ + (i__ + 1) * 82 - 83];
	t66[i__ + 405] = co2800[i__ + (i__ + 1) * 82 - 83];
	if (*iq == 5) {
	    goto L400;
	}
	t66[i__ + 81] = dcdt10[i__ + (i__ + 1) * 82 - 83];
	t66[i__ + 243] = d2ct10[i__ + (i__ + 1) * 82 - 83];
	t66[i__ + 324] = dcdt8[i__ + (i__ + 1) * 82 - 83];
	t66[i__ + 486] = d2ct8[i__ + (i__ + 1) * 82 - 83];
L400:
	;
    }
    return 0;
}




















 GD_DEVICE GD_NOINLINE int co2ins_(real *t22, real *t23, real *t66, integer *iq,
	integer *l, integer *lp1, integer *iflag, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init)
{

    integer i__1, i__2;
    real r__1;


    integer i__, j;

    real c1;
    real c2x;
    real *co2800=work->co2ins_co2800;
    real *co2801=work->co2ins_co2801;
    real *co2802=work->co2ins_co2802;
    real d2ct8[
	    6724];
    real *d2ct10=work->co2ins_d2ct10;
    real *dcdt8=work->co2ins_dcdt8;
    real *co2po=work->co2ins_co2po;
    real *dcdt10=work->co2ins_dcdt10;
    real *co2po1=work->co2ins_co2po1;
    real *co2po2=work->co2ins_co2po2;


    t66 -= 6807;
    t23 -= 6807;
    t22 -= 6807;


    i__1 = *lp1;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *lp1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    co2po[i__ + j * 82 - 83] = t22[i__ + (j + 82) * 82];
	    if (*iq == 5) {
		goto L300;
	    }
	    co2po1[i__ + j * 82 - 83] = t22[i__ + (j + 164) * 82];
	    co2po2[i__ + j * 82 - 83] = t22[i__ + (j + 246) * 82];
L300:
	    ;
	}
    }
    i__2 = *lp1;
    for (j = 1; j <= i__2; ++j) {
	i__1 = *lp1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    co2800[i__ + j * 82 - 83] = t23[i__ + (j + 82) * 82];
	    if (*iq == 5) {
		goto L301;
	    }
	    co2801[i__ + j * 82 - 83] = t23[i__ + (j + 164) * 82];
	    co2802[i__ + j * 82 - 83] = t23[i__ + (j + 246) * 82];
L301:
	    ;
	}
    }
    if (*iq == 1) {
	c1 = 1.5f;
	c2x = .5f;
    }
    if (*iq == 2) {
	c1 = 1.6363636363636365f;
	c2x = .63636363636363635f;
    }
    if (*iq == 3) {
	c1 = 1.3846153846153846f;
	c2x = .38461538461538464f;
    }
    if (*iq == 4) {
	c1 = 1.8f;
	c2x = .8f;
    }
    if (*iq == 5) {
	c1 = 1.f;
	c2x = 0.f;
    }
    i__1 = *lp1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = *lp1;
	for (j = 1; j <= i__2; ++j) {
	    co2po[j + i__ * 82 - 83] = c1 * co2po[j + i__ * 82 - 83] - c2x;
	    co2800[j + i__ * 82 - 83] = c1 * co2800[j + i__ * 82 - 83] - c2x;
	    if (*iq == 5) {
		goto L1021;
	    }
	    co2po1[j + i__ * 82 - 83] = c1 * co2po1[j + i__ * 82 - 83] - c2x;
	    co2801[j + i__ * 82 - 83] = c1 * co2801[j + i__ * 82 - 83] - c2x;
	    co2po2[j + i__ * 82 - 83] = c1 * co2po2[j + i__ * 82 - 83] - c2x;
	    co2802[j + i__ * 82 - 83] = c1 * co2802[j + i__ * 82 - 83] - c2x;
L1021:
	    ;
	}
    }
    if (*iq >= 1 && *iq <= 4) {
	i__2 = *lp1;
	for (j = 1; j <= i__2; ++j) {
	    i__1 = *lp1;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		r__1 = (co2801[i__ + j * 82 - 83] - co2802[i__ + j * 82 - 83])
			 * .02f;
		dcdt8[i__ + j * 82 - 83] = r__1 * 100.f;
		r__1 = (co2po1[i__ + j * 82 - 83] - co2po2[i__ + j * 82 - 83])
			 * .02f;
		dcdt10[i__ + j * 82 - 83] = r__1 * 100.f;
		r__1 = (co2801[i__ + j * 82 - 83] + co2802[i__ + j * 82 - 83]
			- co2800[i__ + j * 82 - 83] * 2.f) * .0016f;
		d2ct8[i__ + j * 82 - 83] = r__1 * 1e3f;
		r__1 = (co2po1[i__ + j * 82 - 83] + co2po2[i__ + j * 82 - 83]
			- co2po[i__ + j * 82 - 83] * 2.f) * .0016f;
		d2ct10[i__ + j * 82 - 83] = r__1 * 1e3f;

	    }
	}
    }
    if (*iq == 1 || *iq == 4) {
	i__1 = *lp1;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *lp1;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		t66[i__ + (j + 82) * 82] = dcdt10[i__ + j * 82 - 83];
		t66[i__ + (j + 164) * 82] = co2po[i__ + j * 82 - 83];
		t66[i__ + (j + 246) * 82] = d2ct10[i__ + j * 82 - 83];
		t66[i__ + (j + 328) * 82] = dcdt8[i__ + j * 82 - 83];
		t66[i__ + (j + 410) * 82] = co2800[i__ + j * 82 - 83];
		t66[i__ + (j + 492) * 82] = d2ct8[i__ + j * 82 - 83];

	    }
	}
    } else {
	i__2 = *lp1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    t66[i__ + 13530] = co2po[i__ * 82 - 82];
	    t66[i__ + 33702] = co2800[i__ * 82 - 82];
	    if (*iq == 5) {
		goto L409;
	    }
	    t66[i__ + 6806] = dcdt10[i__ * 82 - 82];
	    t66[i__ + 20254] = d2ct10[i__ * 82 - 82];
	    t66[i__ + 26978] = dcdt8[i__ * 82 - 82];
	    t66[i__ + 40426] = d2ct8[i__ * 82 - 82];
L409:
	    ;
	}
    }
    return 0;
}






















GD_DEVICE doublereal co2int_c_b2 = 10.;
GD_DEVICE doublereal co2int_c_b3 = .066666666666666666;
GD_DEVICE doublereal co2int_c_b5 = .033333333333333333;

 GD_DEVICE GD_NOINLINE int co2int_(integer *itape, real *t15a, real *t15b, real *
	t22, real *ratio, integer *ir, integer *nmethd, integer *nlevls,
	integer *nlp1, integer *nlp2, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init)
{


    integer i__1, i__2, i__3;





    integer i__, j;

    real *p=work->co2int_p;
    real *pd=work->co2int_pd;
    real *ps=work->co2int_ps;
    integer iap, kkk;

    real *pds=work->co2int_pds;
    real *plm=work->co2int_plm;
    integer nlv, ntap;

    real *trns=work->co2int_trns;
    integer itap1, nlp1v, nlp2v;
    real fact30, fact15;
    extern GD_DEVICE int gd_load_transmission__(integer *, integer *, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init),
	    sintr2_(GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init), coeint_(real *, integer *, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init);
    integer icloop;
    extern GD_DEVICE GD_NOINLINE int quadsr_(integer *, integer *, integer *, real
	    *, real *, real *, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init);


    t22 -= 6807;
    --t15b;
    t15a -= 84;


    gd_init_state__1.pa[0] = 0.f;
    fact15 = (real)pow_dd(&co2int_c_b2, &co2int_c_b3);
    fact30 = (real)pow_dd(&co2int_c_b2, &co2int_c_b5);
    gd_init_state__1.pa[1] = .001f;
    for (i__ = 2; i__ <= 76; ++i__) {
	gd_init_state__1.pa[i__] = gd_init_state__1.pa[i__ - 1] *
		fact15;

    }
    for (i__ = 77; i__ <= 108; ++i__) {
	gd_init_state__1.pa[i__] = gd_init_state__1.pa[i__ - 1] *
		fact30;

    }
    gd_init_state__1.n = 25;
    nlv = *nlevls;
    nlp1v = *nlp1;
    nlp2v = *nlp2;
    if (*ratio == 1.f) {
	goto L621;
    }
    gd_init_state__1.status = 1;
L621:
    itap1 = *itape;
    ntap = 1;
    if (*nmethd == 2) {
	goto L502;
    }
    i__1 = *nlp1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	p[i__ - 1] = t15b[i__];

    }
    i__1 = *nlp1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ps[i__ - 1] = p[i__ - 1];

    }
    goto L503;
L502:
    i__1 = *nlp2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	pd[i__ - 1] = t15a[i__ + 83];

    }
    i__1 = *nlp1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	plm[i__ - 1] = t15a[i__ + 166];

    }
    i__1 = *nlp1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	pds[i__ - 1] = pd[i__];
	ps[i__ - 1] = plm[i__ - 1];

    }
L503:
    icloop = 3;
    if (*ir == 4) {
	icloop = 1;
    }
    i__1 = icloop;
    for (kkk = 1; kkk <= i__1; ++kkk) {
	if (*nmethd == 2) {
	    goto L505;
	}
	i__2 = *nlp1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    p[i__ - 1] = ps[i__ - 1];

	}
	goto L506;
L505:
	i__2 = *nlp1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    pd[i__ - 1] = pds[i__ - 1];
	    p[i__ - 1] = ps[i__ - 1];

	}
L506:
	gd_init_state__1.ia = 108;
	iap = gd_init_state__1.ia + 1;
	if (ntap == 1) {
	    gd_load_transmission__(itape, &kkk, gt, clock, work, init);
	}
	for (j = 109; j >= 1; j += -6) {
	}
	i__2 = iap;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    gd_init_state__1.transa[i__ + i__ * 109 - 110] = 1.f;

	}
	coeint_(ratio, ir, gt, clock, work, init);
	i__2 = *nlp1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = *nlp1;
	    for (j = 1; j <= i__3; ++j) {
		trns[j + i__ * 82 - 83] = 1.f;

	    }
	}
	i__3 = *nlp1;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    i__2 = i__;
	    for (j = 1; j <= i__2; ++j) {
		if (i__ == j) {
		    goto L20;
		}
		gd_init_state__1.p1 = p[j - 1];
		gd_init_state__1.p2 = p[i__ - 1];
		sintr2_(gt, clock, work, init);
		trns[j + i__ * 82 - 83] = gd_init_state__1.trnslo;
L20:
		;
	    }

	}
	i__3 = *nlp1;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    i__2 = *nlp1;
	    for (j = i__; j <= i__2; ++j) {
		trns[j + i__ * 82 - 83] = trns[i__ + j * 82 - 83];

	    }
	}
	if (*nmethd == 1) {
	    goto L2872;
	}
	i__2 = *nlp1;
	for (j = 1; j <= i__2; ++j) {
	    i__3 = *nlp1;
	    for (i__ = 2; i__ <= i__3; ++i__) {
		gd_init_state__1.ia = i__;
		gd_init_state__1.ja = j;
		gd_init_state__1.n = 25;
		if (i__ != j) {
		    gd_init_state__1.n = 3;
		}
		quadsr_(&nlv, &nlp1v, &nlp2v, p, pd, trns, gt, clock, work, init);

	    }

	}
L2872:
	i__2 = *nlp1;
	for (j = 1; j <= i__2; ++j) {
	    i__3 = *nlp1;
	    for (i__ = 1; i__ <= i__3; ++i__) {
		t22[i__ + (j + kkk * 82) * 82] = trns[i__ + j * 82 - 83];

	    }
	}

    }
    return 0;
}






















GD_DEVICE integer co2o3_c__1 = 1;
GD_DEVICE integer co2o3_c__2 = 2;
GD_DEVICE integer co2o3_c__3 = 3;

 GD_DEVICE GD_NOINLINE int co2o3_(real *sfull, real *shalf, real *pptop, integer *l,
	 integer *lp1, integer *lp2, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init)
{

    integer i__1, i__2;


    integer k, k1, k2;

    real *t20=work->co2o3_t20;
    real *t21=work->co2o3_t21;
    real *t22=work->co2o3_t22;
    real *t41=work->co2o3_t41;
    real *t42=work->co2o3_t42;
    real *t43=work->co2o3_t43;
    real *t44=work->co2o3_t44;
    real *t23=work->co2o3_t23;
    integer ir, iq, etarad_unit61__, etarad_unit62__, etarad_unit63__;

    real *co2d1d=work->co2o3_co2d1d;
    real *co2d2d=work->co2o3_co2d2d;
    integer lread;
    extern GD_DEVICE GD_NOINLINE int co2in1_(real *, real *, real *, integer *,
	    integer *, integer *, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init);

    real *co2iq2=work->co2o3_co2iq2;
    real *co2iq3=work->co2o3_co2iq3;
    real ratio;
    extern GD_DEVICE GD_NOINLINE int co2ins_(real *, real *, real *, integer *,
	    integer *, integer *, integer *, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init), co2int_(integer *, real *, real
	    *, real *, real *, integer *, integer *, integer *, integer *,
	    integer *, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init);
    integer ico2tp;
    extern GD_DEVICE GD_NOINLINE int co2ptz_(real *, real *, real *, real *, real *
	    , real *, real *, real *, real *, real *, integer *, integer *,
	    integer *, integer *, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init);
    integer nmethd;

    real *sgtemp=work->co2o3_sgtemp;
    real *siglnu=work->co2o3_siglnu;
    real *sglvnu=work->co2o3_sglvnu;


    --shalf;
    --sfull;


    etarad_unit61__ = 1;
    etarad_unit62__ = 2;
    etarad_unit63__ = 3;
    lread = 0;
    co2ptz_(sgtemp, t41, t42, t43, t44, sglvnu, siglnu, &sfull[1], &shalf[1],
	    pptop, &lread, l, lp1, lp2, gt, clock, work, init);
    i__1 = *lp1;
    for (k = 1; k <= i__1; ++k) {
	gd_tables__1.stemp[k - 1] = sgtemp[k - 1];
	gd_tables__1.gtemp[k - 1] = sgtemp[k + 81];
    }
    ico2tp = etarad_unit61__;
    ir = 1;
    ratio = 1.f;
    nmethd = 2;
    co2int_(&ico2tp, t41, t42, t22, &ratio, &ir, &nmethd, l, lp1, lp2, gt, clock, work, init);
    ir = 1;
    ratio = 1.f;
    nmethd = 1;
    co2int_(&ico2tp, t41, t42, t20, &ratio, &ir, &nmethd, l, lp1, lp2, gt, clock, work, init);
    ir = 1;
    ratio = 1.f;
    nmethd = 2;
    co2int_(&ico2tp, t43, t44, t23, &ratio, &ir, &nmethd, l, lp1, lp2, gt, clock, work, init);
    ir = 1;
    ratio = 1.f;
    nmethd = 1;
    co2int_(&ico2tp, t43, t44, t21, &ratio, &ir, &nmethd, l, lp1, lp2, gt, clock, work, init);
    iq = 1;
    co2in1_(t20, t21, co2d1d, &iq, l, lp1, gt, clock, work, init);
    i__1 = *l;
    for (k = 1; k <= i__1; ++k) {
	gd_tables__1.cdtm51[k - 1] = co2d1d[k - 1];
	gd_tables__1.co2m51[k - 1] = co2d1d[k + 80];
	gd_tables__1.c2dm51[k - 1] = co2d1d[k + 161];
	gd_tables__1.cdtm58[k - 1] = co2d1d[k + 242];
	gd_tables__1.co2m58[k - 1] = co2d1d[k + 323];
	gd_tables__1.c2dm58[k - 1] = co2d1d[k + 404];
    }
    co2ins_(t22, t23, co2d2d, &iq, l, lp1, &co2o3_c__1, gt, clock, work, init);
    i__1 = *lp1;
    for (k1 = 1; k1 <= i__1; ++k1) {
	i__2 = *lp1;
	for (k2 = 1; k2 <= i__2; ++k2) {
	    gd_tables__1.cdt51[k1 + k2 * 82 - 83] = co2d2d[k1 + (k2 +
		    82) * 82 - 6807];
	    gd_tables__1.co251[k1 + k2 * 82 - 83] = co2d2d[k1 + (k2 +
		    164) * 82 - 6807];
	    gd_tables__1.c2d51[k1 + k2 * 82 - 83] = co2d2d[k1 + (k2 +
		    246) * 82 - 6807];
	    gd_tables__1.cdt58[k1 + k2 * 82 - 83] = co2d2d[k1 + (k2 +
		    328) * 82 - 6807];
	    gd_tables__1.co258[k1 + k2 * 82 - 83] = co2d2d[k1 + (k2 +
		    410) * 82 - 6807];
	    gd_tables__1.c2d58[k1 + k2 * 82 - 83] = co2d2d[k1 + (k2 +
		    492) * 82 - 6807];
	}
    }
    ico2tp = etarad_unit62__;
    ir = 2;
    ratio = 1.f;
    nmethd = 2;
    co2int_(&ico2tp, t41, t42, t22, &ratio, &ir, &nmethd, l, lp1, lp2, gt, clock, work, init);
    co2int_(&ico2tp, t43, t44, t23, &ratio, &ir, &nmethd, l, lp1, lp2, gt, clock, work, init);
    iq = 2;
    co2ins_(t22, t23, co2iq2, &iq, l, lp1, &co2o3_c__2, gt, clock, work, init);
    i__1 = *lp1;
    for (k = 1; k <= i__1; ++k) {
	gd_tables__1.cdt31[k - 1] = co2iq2[k - 1];
	gd_tables__1.co231[k - 1] = co2iq2[k + 6723];
	gd_tables__1.c2d31[k - 1] = co2iq2[k + 13447];
	gd_tables__1.cdt38[k - 1] = co2iq2[k + 20171];
	gd_tables__1.co238[k - 1] = co2iq2[k + 26895];
	gd_tables__1.c2d38[k - 1] = co2iq2[k + 33619];
    }
    ico2tp = etarad_unit63__;
    ir = 3;
    ratio = 1.f;
    nmethd = 2;
    co2int_(&ico2tp, t41, t42, t22, &ratio, &ir, &nmethd, l, lp1, lp2, gt, clock, work, init);
    co2int_(&ico2tp, t43, t44, t23, &ratio, &ir, &nmethd, l, lp1, lp2, gt, clock, work, init);
    iq = 3;
    co2ins_(t22, t23, co2iq3, &iq, l, lp1, &co2o3_c__3, gt, clock, work, init);
    i__1 = *lp1;
    for (k = 1; k <= i__1; ++k) {
	gd_tables__1.cdt71[k - 1] = co2iq3[k - 1];
	gd_tables__1.co271[k - 1] = co2iq3[k + 6723];
	gd_tables__1.c2d71[k - 1] = co2iq3[k + 13447];
	gd_tables__1.cdt78[k - 1] = co2iq3[k + 20171];
	gd_tables__1.co278[k - 1] = co2iq3[k + 26895];
	gd_tables__1.c2d78[k - 1] = co2iq3[k + 33619];
    }
    return 0;
}






















GD_DEVICE integer co2ptz_c__6 = 6;
GD_DEVICE real co2ptz_c_b4 = 0.f;

 GD_DEVICE GD_NOINLINE int co2ptz_(real *sgtemp, real *t41, real *t42, real *t43,
	real *t44, real *sglvnu, real *siglnu, real *sfull, real *shalf, real
	*pptop, integer *lread, integer *nl, integer *nlp, integer *nlp2, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init)
{


    real psmax = 1013.25f;


    integer i__1, i__2;
    real r__1, r__2, r__3, r__4;






    real *a=work->co2ptz_a;
    integer m, n;

    real r__;
    real g0;
    real aa;
    real *pd=work->co2ptz_pd;
    real dz;
    integer nq;

    real ht;
    real rk1;
    real rk2;
    real rk3;
    real rk4;
    real *alt=work->co2ptz_alt;
    integer nlm;
    real pcld;
    extern GD_DEVICE GD_NOINLINE int sigp_(real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, integer *,
	     integer *, integer *, integer *, integer *, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init);
    integer nlev;

    real *temp=work->co2ptz_temp;
    integer nint;

    real dlogp;
    real *gtemp=work->co2ptz_gtemp;
    real pstar;
    real *press=work->co2ptz_press;
    real zmass;
    integer ntype;
    real znint, delzap;
    extern GD_DEVICE real antemp_(integer *, real *, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init);

    real *tmpint=work->co2ptz_tmpint;
    real *prsint=work->co2ptz_prsint;


    --shalf;
    --sfull;
    --siglnu;
    --sglvnu;
    --t44;
    t43 -= 84;
    --t42;
    t41 -= 84;
    sgtemp -= 83;


    ntype = 0;

    nlev = *nl;
    delzap = .5f;
    r__ = 8.31432f;
    g0 = 9.80665f;
    zmass = 28.9644f;
    aa = 6356.766f;
    alt[0] = 0.f;
    temp[0] = (real)antemp_(&co2ptz_c__6, &co2ptz_c_b4, gt, clock, work, init);
    pstar = psmax;
    gd_tables__1.ltop[0] = 0;
    gd_tables__1.ltop[1] = 0;
    gd_tables__1.ltop[2] = 0;
    i__1 = *nl;
    for (n = 1; n <= i__1; ++n) {
	pcld = (pstar - *pptop * 10.f) * shalf[n] + *pptop * 10.f;
	if (pcld >= 642.f) {
	    gd_tables__1.ltop[0] = n;
	}
	if (pcld >= 350.f) {
	    gd_tables__1.ltop[1] = n;
	}
	if (pcld >= 150.f) {
	    gd_tables__1.ltop[2] = n;
	}

    }
    nlm = *nl - 1;
    sigp_(&pstar, pd, gtemp, &t41[84], &t42[1], &t43[84], &t44[1], &sglvnu[1],
	     &siglnu[1], &sfull[1], &shalf[1], pptop, lread, nl, nlp, &nlm,
	    nlp2, gt, clock, work, init);
    pd[*nlp2 - 1] = pstar;
    i__1 = *nlp;
    for (n = 1; n <= i__1; ++n) {
	prsint[n - 1] = pd[*nlp2 + 1 - n - 1];

    }
    for (nq = 1; nq <= 4; ++nq) {
	i__1 = *nlp;
	for (n = 2; n <= i__1; ++n) {

	    r__1 = (nq - 1) * .25f;
	    press[n - 1] = prsint[n - 1] + r__1 * (prsint[n - 2] - prsint[n -
		    1]);
	}
	press[0] = prsint[0];
	i__1 = nlev;
	for (n = 1; n <= i__1; ++n) {
	    dlogp = (real)gd_log((real)(press[n - 1] / press[n])) * 7.f;
	    nint = dlogp / delzap;
	    ++nint;
	    znint = (real) nint;
	    r__2 = zmass * 7.f;
	    r__1 = r__2 * g0;
	    dz = r__ * dlogp / (r__1 * znint);
	    ht = alt[n - 1];
	    i__2 = nint;
	    for (m = 1; m <= i__2; ++m) {
		rk1 = (real)antemp_(&co2ptz_c__6, &ht, gt, clock, work, init) * dz;
		r__1 = ht + rk1 * .5f;
		rk2 = (real)antemp_(&co2ptz_c__6, &r__1, gt, clock, work, init) * dz;
		r__1 = ht + rk2 * .5f;
		rk3 = (real)antemp_(&co2ptz_c__6, &r__1, gt, clock, work, init) * dz;
		r__1 = ht + rk3;
		rk4 = (real)antemp_(&co2ptz_c__6, &r__1, gt, clock, work, init) * dz;
		r__4 = rk1 + rk2;
		r__3 = r__4 + rk2;
		r__2 = r__3 + rk3;
		r__1 = r__2 + rk3;
		ht += (r__1 + rk4) * .16666667f;

	    }
	    alt[n] = ht;
	    temp[n] = (real)antemp_(&co2ptz_c__6, &ht, gt, clock, work, init);

	}
	i__1 = *nlp;
	for (n = 1; n <= i__1; ++n) {
	    tmpint[n + nq * 82 - 83] = temp[n - 1];
	    a[n + nq * 82 - 83] = alt[n - 1];

	}

    }
    i__1 = *nlp;
    for (n = 1; n <= i__1; ++n) {
	sgtemp[n + 82] = tmpint[*nlp2 - n - 1];

    }
    i__1 = *nlp;
    for (n = 1; n <= i__1; ++n) {
	sgtemp[n + 164] = gtemp[n - 1];

    }
    return 0;
}






















GD_DEVICE real coeint_c_b8 = 0.f;

 GD_DEVICE int coeint_(real *rat, integer *ir, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init)
{


    real sinv[4] = { 2.74992f,2.12731f,4.38111f,.0832926f };


    real r__1;
    doublereal d__1, d__2, d__3, d__4;





    real f;
    integer i__;
    real r__, f1, f2;
    integer ll, np;

    real xx;
    real pa2;
    real arg1;
    real arg2;
    real *etap=work->coeint_etap;
    extern GD_DEVICE real path_(real *, real *, real *, real *, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init);

    real rexp;
    real *path0=work->coeint_path0;
    real check;
    real patha;
    real pathb;
    real fprime;

    gd_init_state__1.core = 5.f;
    gd_init_state__1.uexp = .9f;
    for (i__ = 1; i__ <= 109; ++i__) {
	pa2 = gd_init_state__1.pa[i__ - 1] * gd_init_state__1.pa[
		i__ - 1];
	r__1 = gd_init_state__1.pa[i__ - 1] * 2e-5f + .505f;
	gd_init_state__1.sexpv[i__ - 1] = r__1 + (pa2 - .25f) * .035f /
		(pa2 + .25f);

    }
    for (i__ = 1; i__ <= 109; ++i__) {
	gd_init_state__1.eta[i__ - 1] = (real)gd_exp((real)(
		-gd_init_state__1.pa[i__ - 1] / 500.f)) * 3.2e-4f;
	etap[i__ - 1] = gd_init_state__1.eta[i__ - 1];

    }
    for (np = 1; np <= 10; ++np) {
	for (i__ = 3; i__ <= 109; ++i__) {
	    gd_init_state__1.sexp = gd_init_state__1.sexpv[i__ -
		    1];
	    r__ = (1. - gd_init_state__1.transa[i__ + (i__ - 2) * 109 -
		    110]) / (1. - gd_init_state__1.transa[i__ + (i__ -
		    1) * 109 - 110]);
	    d__1 = (doublereal) r__;
	    d__2 = (doublereal) (gd_init_state__1.uexp /
		    gd_init_state__1.sexp);
	    rexp = (real)pow_dd(&d__1, &d__2);
	    arg1 = (real)path_(&gd_init_state__1.pa[i__ - 1], &
		    gd_init_state__1.pa[i__ - 3], &
		    gd_init_state__1.core, &gd_init_state__1.eta[
		    i__ - 1], gt, clock, work, init);
	    arg2 = (real)path_(&gd_init_state__1.pa[i__ - 1], &
		    gd_init_state__1.pa[i__ - 2], &
		    gd_init_state__1.core, &gd_init_state__1.eta[
		    i__ - 1], gt, clock, work, init);
	    d__1 = (doublereal) (real)path_(&gd_init_state__1.pa[i__ -
		    1], &gd_init_state__1.pa[i__ - 3], &
		    gd_init_state__1.core, &gd_init_state__1.eta[
		    i__ - 1], gt, clock, work, init);
	    d__2 = (doublereal) gd_init_state__1.uexp;
	    patha = (real)pow_dd(&d__1, &d__2);
	    d__1 = (doublereal) (real)path_(&gd_init_state__1.pa[i__ -
		    1], &gd_init_state__1.pa[i__ - 2], &
		    gd_init_state__1.core, &gd_init_state__1.eta[
		    i__ - 1], gt, clock, work, init);
	    d__2 = (doublereal) gd_init_state__1.uexp;
	    pathb = (real)pow_dd(&d__1, &d__2);
	    r__1 = pathb * pathb;
	    xx = (pathb * rexp - patha) * 2. / (r__1 * rexp - patha * patha);
	    for (ll = 1; ll <= 20; ++ll) {
		f1 = gd_log(xx * patha + 1.);
		f2 = gd_log(xx * pathb + 1.);
		f = f1 / f2 - rexp;
		fprime = (f2 * patha / (xx * patha + 1.) - f1 * pathb / (xx *
			pathb + 1.)) / (f2 * f2);
		xx -= f / fprime;
		check = xx * patha + 1.;
		if (check <= 0.f) {
		    gd_init_state__1.status = 1;
		}

	    }
	    d__1 = 1. - gd_init_state__1.transa[i__ + (i__ - 2) * 109 -
		    110];
	    d__2 = (doublereal) (gd_init_state__1.uexp /
		    gd_init_state__1.sexp);
	    gd_init_state__1.ca[i__ - 1] = gd_pow_double_ptr(&d__1, &d__2) / (gd_log(
		    xx * patha + 1.) + 1e-20);
	    gd_init_state__1.xa[i__ - 1] = xx;

	}
	gd_init_state__1.xa[1] = gd_init_state__1.xa[2];
	gd_init_state__1.xa[0] = gd_init_state__1.xa[2];
	gd_init_state__1.ca[1] = gd_init_state__1.ca[2];
	gd_init_state__1.ca[0] = gd_init_state__1.ca[2];
	for (i__ = 3; i__ <= 109; ++i__) {
	    d__1 = (doublereal) (real)path_(&gd_init_state__1.pa[i__ -
		    1], &coeint_c_b8, &gd_init_state__1.core, &
		    gd_init_state__1.eta[i__ - 1], gt, clock, work, init);
	    d__2 = (doublereal) gd_init_state__1.uexp;
	    path0[i__ - 1] = (real)pow_dd(&d__1, &d__2);
	    path0[i__ - 1] = gd_init_state__1.xa[i__ - 1] * path0[i__ -
		    1] + 1.;

	}
	for (i__ = 1; i__ <= 109; ++i__) {
	    gd_init_state__1.sexp = gd_init_state__1.sexpv[i__ -
		    1];
	    etap[i__ - 1] = gd_init_state__1.eta[i__ - 1];
	    d__1 = (doublereal) (sinv[*ir - 1] / *rat);
	    d__2 = (doublereal) (1.f / gd_init_state__1.sexp);
	    d__3 = (doublereal) (gd_init_state__1.ca[i__ - 1] *
		    gd_init_state__1.xa[i__ - 1]);
	    d__4 = (doublereal) (1.f / gd_init_state__1.uexp);
	    gd_init_state__1.eta[i__ - 1] = (real)pow_dd(&d__1, &d__2) *
		     (real)pow_dd(&d__3, &d__4);

	}

    }
    return 0;
}






















GD_DEVICE doublereal fpvsx_c_b2 = 5.0693392753601074;
GD_DEVICE doublereal fpvsx_c_b3 = .56338030099868774;

GD_DEVICE real fpvsx_(real *t, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init)
{

    real ret_val, r__1;
    doublereal d__1;





    real tr;

    tr = 273.16000366210938f / *t;
    if (*t >= 273.16000366210938f) {
	d__1 = (doublereal) tr;
	r__1 = (real)pow_dd(&d__1, &fpvsx_c_b2) * .61078006029129028f;
	ret_val = r__1 * (real)gd_exp((real)((1.f - tr) * 24.900640487670898f));
    } else {
	d__1 = (doublereal) tr;
	r__1 = (real)pow_dd(&d__1, &fpvsx_c_b3) * .61078006029129028f;
	ret_val = r__1 * (real)gd_exp((real)((1.f - tr) * 23.044143676757812f));
    }
    return ret_val;
}




















 GD_DEVICE int gd_load_transmission__(integer *itape, integer *itemp, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init)
{
    integer i__, j;

    for (j = 1; j <= 109; ++j) {
	for (i__ = 1; i__ <= 109; ++i__) {
	    gd_init_state__1.transa[i__ + j * 109 - 110] =
		    gd_init_state__1.raw[i__ + (j + (*itemp + *itape *
		    3) * 109) * 109 - 47634];
	}
    }
    return 0;
}




















 GD_DEVICE int gpvs_(GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init)
{
    real t, x;
    integer jx;
    real xinc;
    extern GD_DEVICE real fpvsx_(real *, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init);

    xinc = .02f;
    gd_tables__1.c1xpvs = 1.f - 180.f / xinc;
    gd_tables__1.c2xpvs = 1.f / xinc;
    for (jx = 1; jx <= 7501; ++jx) {
	x = (jx - 1) * xinc + 180.f;
	t = x;
	gd_tables__1.tbpvs[jx - 1] = (real)fpvsx_(&t, gt, clock, work, init);
    }
    return 0;
}




















 GD_DEVICE int o3clim_(GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init)
{


    real *o3lo1=init->ozone_data+0;
    real *o3lo2=init->ozone_data+160;
    real *o3lo3=init->ozone_data+320;
    real *o3lo4=init->ozone_data+480;
    real *equiv_1=init->ozone_data+640;

    real *equiv_3=init->ozone_data+890;

    real *equiv_5=init->ozone_data+972;



    integer i__1;
    real r__1, r__2;

    real *equiv_9=work->o3clim_equiv_9;


    integer i__, k, n;

    real *p=work->o3clim_p;
    real a1;
    real b1;
    real b2;
#define p1 (equiv_5)
#define p2 (equiv_5 + 48)
    integer ki, jj, kk;

    real *ph=work->o3clim_ph;
    integer kq;
#define ph1 (equiv_3)
#define ph2 (equiv_3 + 45)
    real avg;
    integer ken;
#define o3o3 (equiv_9)
    integer nkm;

    real *ro31=work->o3clim_ro31;
    real *ro32=work->o3clim_ro32;
#define pxx (equiv_5)
#define pyy (equiv_3)
#define o3hi (equiv_1)
#define o3hi1 (equiv_1)
#define o3hi2 (equiv_1 + 160)
    integer nkmm;

    real *pstd=work->o3clim_pstd;
    real *rstd=work->o3clim_rstd;
#define xrad1 (equiv_9)
#define xrad2 (equiv_9 + 2997)
#define xrad3 (equiv_9 + 5994)
#define xrad4 (equiv_9 + 8991)

    real *duo3n=work->o3clim_duo3n;
    real *o35deg=work->o3clim_o35deg;
    integer ncase;

    real *rdata=work->o3clim_rdata;
    real *tempn=work->o3clim_tempn;
    real *dduo3n=work->o3clim_dduo3n;
    integer iplace, iindex, jindex;

    for (k = 1; k <= 81; ++k) {
	ph[k - 1] = pyy[k - 1] * 1013250.f;
	p[k - 1] = pxx[k - 1] * 1013250.f;
    }
    ph[81] = pyy[81] * 1013250.f;
    for (k = 1; k <= 81; ++k) {
	pstd[k - 1] = p[k - 1];
    }
    for (k = 1; k <= 25; ++k) {
	for (n = 1; n <= 10; ++n) {
	    ro31[n + k * 10 - 11] = o3hi[n + k * 10 - 11];
	    ro32[n + k * 10 - 11] = o3hi[n + k * 10 - 11];
	}
    }
    for (ncase = 1; ncase <= 4; ++ncase) {
	iplace = 2;
	if (ncase == 2) {
	    iplace = 4;
	}
	if (ncase == 3) {
	    iplace = 1;
	}
	if (ncase == 4) {
	    iplace = 3;
	}
	if (ncase == 1 || ncase == 2) {
	    for (k = 26; k <= 41; ++k) {
		for (n = 1; n <= 10; ++n) {
		    ro31[n + k * 10 - 11] = o3lo1[n + (k - 25) * 10 - 11];
		    ro32[n + k * 10 - 11] = o3lo2[n + (k - 25) * 10 - 11];
		}
	    }
	}
	if (ncase == 3 || ncase == 4) {
	    for (k = 26; k <= 41; ++k) {
		for (n = 1; n <= 10; ++n) {
		    ro31[n + k * 10 - 11] = o3lo3[n + (k - 25) * 10 - 11];
		    ro32[n + k * 10 - 11] = o3lo4[n + (k - 25) * 10 - 11];
		}
	    }
	}
	for (kk = 1; kk <= 41; ++kk) {
	    for (n = 1; n <= 10; ++n) {
		duo3n[n + kk * 19 - 20] = ro31[11 - n + kk * 10 - 11];
		duo3n[n + 9 + kk * 19 - 20] = ro32[n + kk * 10 - 11];
	    }
	    duo3n[kk * 19 - 10] = (ro31[kk * 10 - 10] + ro32[kk * 10 - 10]) *
		    .5f;

	}
	if (ncase == 2 || ncase == 4) {
	    for (kk = 1; kk <= 41; ++kk) {
		for (n = 1; n <= 19; ++n) {
		    tempn[n - 1] = duo3n[20 - n + kk * 19 - 20];
		}
		for (n = 1; n <= 19; ++n) {
		    duo3n[n + kk * 19 - 20] = tempn[n - 1];
		}

	    }
	}
	for (n = 1; n <= 19; ++n) {
	    for (kk = 1; kk <= 41; ++kk) {
		rstd[kk - 1] = duo3n[n + kk * 19 - 20];
	    }
	    nkm = 80;
	    nkmm = 78;
	    i__1 = nkmm;
	    for (k = 4; k <= i__1; k += 2) {
		ki = k / 2;
		rdata[k - 1] = (rstd[ki - 1] + rstd[ki]) * .5f - (rstd[ki + 1]
			 - rstd[ki] - rstd[ki - 1] + rstd[ki - 2]) / 16.f;
	    }
	    rdata[1] = (rstd[1] + rstd[0]) * .5f;
	    rdata[nkm - 1] = (rstd[40] + rstd[39]) * .5f;
	    for (k = 1; k <= 81; k += 2) {
		kq = (k + 1) / 2;
		rdata[k - 1] = rstd[kq - 1];
	    }
	    for (kk = 1; kk <= 81; ++kk) {
		dduo3n[n + kk * 19 - 20] = rdata[kk - 1] * .01f;
	    }

	}
	for (kk = 1; kk <= 81; ++kk) {
	    for (n = 1; n <= 19; ++n) {
		o35deg[(n << 1) - 1 + kk * 37 - 38] = dduo3n[n + kk * 19 - 20]
			;
	    }
	    for (n = 1; n <= 18; ++n) {
		o35deg[(n << 1) + kk * 37 - 38] = (dduo3n[n + kk * 19 - 20] +
			dduo3n[n + 1 + kk * 19 - 20]) * .5f;
	    }

	}
	for (jj = 1; jj <= 37; ++jj) {
	    for (ken = 1; ken <= 81; ++ken) {
		o3o3[jj + (ken + iplace * 81) * 37 - 3035] = o35deg[jj + ken *
			 37 - 38];
	    }
	}

    }
    for (i__ = 1; i__ <= 2997; ++i__) {
	r__2 = xrad1[i__ - 1] + xrad2[i__ - 1];
	r__1 = r__2 + xrad3[i__ - 1];
	avg = (r__1 + xrad4[i__ - 1]) * .25f;
	a1 = (xrad2[i__ - 1] - xrad4[i__ - 1]) * .5f;
	b1 = (xrad1[i__ - 1] - xrad3[i__ - 1]) * .5f;
	b2 = (xrad1[i__ - 1] + xrad3[i__ - 1] - (xrad2[i__ - 1] + xrad4[i__ -
		1])) * .25f;
	iindex = (i__ - 1) % 37 + 1;
	jindex = (i__ - 1) / 37 + 1;
	gd_tables__1.xduo3n[iindex + jindex * 37 - 38] = avg;
	gd_tables__1.xdo3n2[iindex + jindex * 37 - 38] = a1;
	gd_tables__1.xdo3n3[iindex + jindex * 37 - 38] = b1;
	gd_tables__1.xdo3n4[iindex + jindex * 37 - 38] = b2;
    }
    for (n = 1; n <= 81; ++n) {
	gd_tables__1.prgfdl[n - 1] = pstd[n - 1] * .1f;
    }
    return 0;
}

#undef xrad4
#undef xrad3
#undef xrad2
#undef xrad1
#undef o3hi2
#undef o3hi1
#undef o3hi
#undef pyy
#undef pxx
#undef o3o3
#undef ph2
#undef ph1
#undef p2
#undef p1





















GD_DEVICE real path_(real *a, real *b, real *c__, real *e, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init)
{

    real ret_val, r__1, r__2;
    doublereal d__1, d__2, d__3, d__4;





    real pexp;

    pexp = 1.f / gd_init_state__1.sexp;
    d__1 = (doublereal) (*a - *b);
    d__2 = (doublereal) pexp;
    r__1 = *a + *b;
    r__2 = *a + *b;
    d__3 = (doublereal) (*a - *b);
    d__4 = (doublereal) (pexp - 1.f);
    ret_val = (real)pow_dd(&d__1, &d__2) * (r__1 + *c__) / (*e * (r__2 + *c__)
	     + (real)pow_dd(&d__3, &d__4));
    return ret_val;
}




















 GD_DEVICE int qintrp_(real *xm, real *x0, real *xp, real *fm, real *f0,
	 real *fp, real *x, real *f, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init)
{
    real a, b, d1, d2, del;

    d1 = (*fp - *f0) / (*xp - *x0);
    d2 = (*fm - *f0) / (*xm - *x0);
    b = (d1 - d2) / (*xp - *xm);
    a = d1 - b * (*xp - *x0);
    del = *x - *x0;
    *f = *f0 + del * (a + del * b);
    return 0;
}




















 GD_DEVICE GD_NOINLINE int quadsr_(integer *nlv, integer *nlp1v, integer *nlp2v,
	real *p, real *pd, real *trns, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init)
{

    integer i__1;


    integer i__, n2;
    real dp;
    integer kk;

    real *wt=work->quadsr_wt;
    integer n2p;
    real pfix, pvary;
    extern GD_DEVICE int sintr2_(GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init);
    real trnsnb;


    trns -= 83;
    --pd;
    --p;


    n2 = gd_init_state__1.n << 1;
    n2p = (gd_init_state__1.n << 1) + 1;
    wt[0] = 1.f;
    i__1 = gd_init_state__1.n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	wt[(i__ << 1) - 1] = 4.f;
	wt[i__ * 2] = 1.f;

    }
    if (gd_init_state__1.n == 1) {
	goto L25;
    }
    i__1 = gd_init_state__1.n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	wt[(i__ << 1) - 2] = 2.f;

    }
L25:
    trnsnb = 0.f;
    dp = (pd[gd_init_state__1.ia] - pd[gd_init_state__1.ia - 1]) /
	     n2;
    pfix = p[gd_init_state__1.ja];
    i__1 = n2p;
    for (kk = 1; kk <= i__1; ++kk) {
	pvary = pd[gd_init_state__1.ia - 1] + (kk - 1) * dp;
	if (pvary >= pfix) {
	    gd_init_state__1.p2 = pvary;
	}
	if (pvary >= pfix) {
	    gd_init_state__1.p1 = pfix;
	}
	if (pvary < pfix) {
	    gd_init_state__1.p1 = pvary;
	}
	if (pvary < pfix) {
	    gd_init_state__1.p2 = pfix;
	}
	sintr2_(gt, clock, work, init);
	trnsnb += gd_init_state__1.trnslo * wt[kk - 1];

    }
    trns[gd_init_state__1.ia + gd_init_state__1.ja * 82] = trnsnb
	    * dp / ((pd[gd_init_state__1.ia] - pd[
	    gd_init_state__1.ia - 1]) * 3.f);
    return 0;
}






















GD_DEVICE doublereal sigp_c_b12 = .2;
GD_DEVICE doublereal sigp_c_b13 = .8;

 GD_DEVICE GD_NOINLINE int sigp_(real *pstar, real *pd, real *gtemp, real *t41,
	real *t42, real *t43, real *t44, real *sglvnu, real *siglnu, real *
	siglv, real *sigly, real *pptop, integer *lread, integer *kd, integer
	*kp, integer *km, integer *kp2, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init)
{

    integer i__1;
    doublereal d__1, d__2;





    integer i__, k;

    real *q=work->sigp_q;
    real *qmh=work->sigp_qmh;
    real *pdt=work->sigp_pdt;
    real *plm=work->sigp_plm;
    real pss;
    real psfc;


    --sigly;
    --siglv;
    --siglnu;
    --sglvnu;
    --t44;
    t43 -= 84;
    --t42;
    t41 -= 84;
    --gtemp;
    --pd;



    if (*pptop <= 0.f) {
	goto L708;
    }
    psfc = 100.f;
    i__1 = *kd;
    for (k = 1; k <= i__1; ++k) {
	sigly[k] = (sigly[k] * (psfc - *pptop) + *pptop) / psfc;

    }
    i__1 = *kp;
    for (k = 1; k <= i__1; ++k) {
	siglv[k] = (siglv[k] * (psfc - *pptop) + *pptop) / psfc;

    }
L708:
    i__1 = *kp;
    for (k = 1; k <= i__1; ++k) {
	sglvnu[k] = siglv[k];
	if (k <= *kd) {
	    siglnu[k] = sigly[k];
	}

    }
    i__1 = *kd;
    for (k = 1; k <= i__1; ++k) {
	q[k - 1] = siglnu[*kd + 1 - k];

    }
    pss = 1013250.f;
    qmh[0] = 0.f;
    qmh[*kp - 1] = 1.f;
    i__1 = *kd;
    for (k = 2; k <= i__1; ++k) {
	qmh[k - 1] = (q[k - 2] + q[k - 1]) * .5f;

    }
    pd[1] = 0.f;
    pd[*kp2] = pss;
    i__1 = *kp;
    for (k = 2; k <= i__1; ++k) {
	pd[k] = q[k - 2] * pss;

    }
    plm[0] = 0.f;
    i__1 = *km;
    for (k = 1; k <= i__1; ++k) {
	plm[k] = (pd[k + 1] + pd[k + 2]) * .5f;

    }
    plm[*kp - 1] = pss;
    i__1 = *kd;
    for (k = 1; k <= i__1; ++k) {
	d__1 = (doublereal) pd[k + 1];
	d__2 = (doublereal) (pd[k + 1] / 3e4f + 1.f);
	gtemp[k] = (real)pow_dd(&d__1, &sigp_c_b12) * (real)pow_dd(&d__2, &sigp_c_b13) /
		 1013250.f;

    }
    gtemp[*kp] = 0.f;
    i__1 = *kp;
    for (i__ = 1; i__ <= i__1; ++i__) {
	pd[i__] *= .001f;
	plm[i__ - 1] *= .001f;

    }
    pd[*kp2] *= .001f;
    i__1 = *kp2;
    for (k = 1; k <= i__1; ++k) {
	t41[k + 83] = pd[k];

    }
    i__1 = *kp;
    for (k = 1; k <= i__1; ++k) {
	t41[k + 166] = plm[k - 1];
	t42[k] = plm[k - 1];

    }
    i__1 = *kp2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	pdt[i__ - 1] = pd[i__];

    }
    pss = 810600.f;
    qmh[0] = 0.f;
    qmh[*kp - 1] = 1.f;
    i__1 = *kd;
    for (k = 2; k <= i__1; ++k) {
	qmh[k - 1] = (q[k - 2] + q[k - 1]) * .5f;

    }
    pd[1] = 0.f;
    pd[*kp2] = pss;
    i__1 = *kp;
    for (k = 2; k <= i__1; ++k) {
	pd[k] = q[k - 2] * pss;

    }
    plm[0] = 0.f;
    i__1 = *km;
    for (k = 1; k <= i__1; ++k) {
	plm[k] = (pd[k + 1] + pd[k + 2]) * .5f;

    }
    plm[*kp - 1] = pss;
    i__1 = *kp;
    for (i__ = 1; i__ <= i__1; ++i__) {
	pd[i__] *= .001f;
	plm[i__ - 1] *= .001f;

    }
    pd[*kp2] *= .001f;
    i__1 = *kp2;
    for (k = 1; k <= i__1; ++k) {
	t43[k + 83] = pd[k];

    }
    i__1 = *kp;
    for (k = 1; k <= i__1; ++k) {
	t43[k + 166] = plm[k - 1];
	t44[k] = plm[k - 1];

    }
    i__1 = *kp2;
    for (i__ = 1; i__ <= i__1; ++i__) {
	pd[i__] = pdt[i__ - 1];

    }
    return 0;
}




















 GD_DEVICE int sintr2_(GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init)
{

    doublereal d__1, d__2;





    integer i__, j, l;
    real ei, ep, ep2;
    integer ip1, jp1;
    real aij, eij;
    integer iii, jjj;
    real tij, uij, tri, ai2j2, ei2j2;
    integer ieta;
    real aijp, aipj, dtdj, eipj, eijp, peta, etap, ti2j2;
    extern GD_DEVICE real path_(real *, real *, real *, real *, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init);
    real cint, ui2j2, prod, tipj, up2p1, tijp, uipj, uijp, trip, xint, aijp2,
	    aip2j, eip2j, eijp2, epip1, tip2j, tijp2, uip2j, uijp2, aipjp,
	    dtdpj, eipjp, epp2p1, epsil, prodi, tipjp, uipjp;
    integer ietap1;
    real aipjp2, aip2jp, eip2jp, eipjp2, epipp1, tip2jp, tipjp2, uipjp2,
	    uip2jp, prodip, pipmpi;
    extern GD_DEVICE int qintrp_(real *, real *, real *, real *, real *
	    , real *, real *, real *, GDTable *gt, GDClock *clock, GDScratch *work, GDInitState *init);

    for (l = 1; l <= 109; ++l) {
	ip1 = l;
	if (gd_init_state__1.p2 - gd_init_state__1.pa[l - 1] <=
		0.f) {
	    goto L65;
	} else {
	    goto L70;
	}
L70:
	;
    }
L65:
    i__ = ip1 - 1;
    if (ip1 == 1) {
	ip1 = 2;
    }
    if (i__ == 0) {
	i__ = 1;
    }
    for (l = 1; l <= 109; ++l) {
	jp1 = l;
	if (gd_init_state__1.p1 - gd_init_state__1.pa[l - 1] <=
		0.f) {
	    goto L75;
	} else {
	    goto L80;
	}
L80:
	;
    }
L75:
    j = jp1 - 1;
    if (jp1 == 1) {
	jp1 = 2;
    }
    if (j == 0) {
	j = 1;
    }
    jjj = j;
    iii = i__;
    j = jjj;
    jp1 = j + 1;
    i__ = iii;
    ip1 = i__ + 1;
    peta = gd_init_state__1.p2;
    for (l = 1; l <= 109; ++l) {
	ietap1 = l;
	if (peta - gd_init_state__1.pa[l - 1] <= 0.f) {
	    goto L85;
	} else {
	    goto L90;
	}
L90:
	;
    }
L85:
    ieta = ietap1 - 1;
    if (ietap1 == 1) {
	ietap1 = 2;
    }
    if (ieta == 0) {
	ieta = 1;
    }
    etap = gd_init_state__1.eta[ieta - 1] + (peta -
	    gd_init_state__1.pa[ieta - 1]) * (
	    gd_init_state__1.eta[ietap1 - 1] -
	    gd_init_state__1.eta[ieta - 1]) / (
	    gd_init_state__1.pa[ietap1 - 1] - gd_init_state__1.pa[
	    ieta - 1]);
    gd_init_state__1.sexp = gd_init_state__1.sexpv[ieta - 1] + (
	    peta - gd_init_state__1.pa[ieta - 1]) * (
	    gd_init_state__1.sexpv[ietap1 - 1] -
	    gd_init_state__1.sexpv[ieta - 1]) / (
	    gd_init_state__1.pa[ietap1 - 1] - gd_init_state__1.pa[
	    ieta - 1]);
    pipmpi = gd_init_state__1.pa[ip1 - 1] - gd_init_state__1.pa[
	    i__ - 1];
    d__1 = (doublereal) (real)path_(&gd_init_state__1.p2, &
	    gd_init_state__1.p1, &gd_init_state__1.core, &etap, gt, clock, work, init);
    d__2 = (doublereal) gd_init_state__1.uexp;
    up2p1 = (real)pow_dd(&d__1, &d__2);
    if (i__ - j <= 0) {
	goto L126;
    } else {
	goto L127;
    }
L126:
    d__1 = gd_init_state__1.ca[ip1 - 1] * gd_log(gd_init_state__1.xa[
	    ip1 - 1] * up2p1 + 1.);
    d__2 = (doublereal) (gd_init_state__1.sexp /
	    gd_init_state__1.uexp);
    trip = gd_pow_double_ptr(&d__1, &d__2);
    d__1 = gd_init_state__1.ca[i__ - 1] * gd_log(gd_init_state__1.xa[
	    i__ - 1] * up2p1 + 1.);
    d__2 = (doublereal) (gd_init_state__1.sexp /
	    gd_init_state__1.uexp);
    tri = gd_pow_double_ptr(&d__1, &d__2);
    gd_init_state__1.trnslo = 1. - ((gd_init_state__1.pa[ip1 - 1]
	    - gd_init_state__1.p2) * tri + (gd_init_state__1.p2 -
	    gd_init_state__1.pa[i__ - 1]) * trip) / pipmpi;
    goto L128;
L127:
    tij = gd_init_state__1.transa[i__ + j * 109 - 110];
    tipj = gd_init_state__1.transa[i__ + 1 + j * 109 - 110];
    tijp = gd_init_state__1.transa[i__ + (j + 1) * 109 - 110];
    tipjp = gd_init_state__1.transa[i__ + 1 + (j + 1) * 109 - 110];
    d__1 = (doublereal) (real)path_(&gd_init_state__1.pa[i__ - 1], &
	    gd_init_state__1.pa[j - 1], &gd_init_state__1.core, &
	    etap, gt, clock, work, init);
    d__2 = (doublereal) gd_init_state__1.uexp;
    uij = (real)pow_dd(&d__1, &d__2);
    d__1 = (doublereal) (real)path_(&gd_init_state__1.pa[i__], &
	    gd_init_state__1.pa[j - 1], &gd_init_state__1.core, &
	    etap, gt, clock, work, init);
    d__2 = (doublereal) gd_init_state__1.uexp;
    uipj = (real)pow_dd(&d__1, &d__2);
    d__1 = (doublereal) (real)path_(&gd_init_state__1.pa[i__ - 1], &
	    gd_init_state__1.pa[j], &gd_init_state__1.core, &etap, gt, clock, work, init)
	    ;
    d__2 = (doublereal) gd_init_state__1.uexp;
    uijp = (real)pow_dd(&d__1, &d__2);
    d__1 = (doublereal) (real)path_(&gd_init_state__1.pa[i__], &
	    gd_init_state__1.pa[j], &gd_init_state__1.core, &etap, gt, clock, work, init)
	    ;
    d__2 = (doublereal) gd_init_state__1.uexp;
    uipjp = (real)pow_dd(&d__1, &d__2);
    prodi = gd_init_state__1.ca[i__ - 1] * gd_init_state__1.xa[
	    i__ - 1];
    prodip = gd_init_state__1.ca[i__] * gd_init_state__1.xa[i__];
    prod = ((gd_init_state__1.pa[i__] - gd_init_state__1.p2) *
	    prodi + (gd_init_state__1.p2 - gd_init_state__1.pa[
	    i__ - 1]) * prodip) / pipmpi;
    xint = ((gd_init_state__1.pa[i__] - gd_init_state__1.p2) *
	    gd_init_state__1.xa[i__ - 1] + (gd_init_state__1.p2 -
	    gd_init_state__1.pa[i__ - 1]) * gd_init_state__1.xa[
	    i__]) / pipmpi;
    cint = prod / xint;
    d__1 = cint * gd_log(xint * uij + 1.);
    d__2 = (doublereal) (gd_init_state__1.sexp /
	    gd_init_state__1.uexp);
    aij = gd_pow_double_ptr(&d__1, &d__2);
    d__1 = cint * gd_log(xint * uijp + 1.);
    d__2 = (doublereal) (gd_init_state__1.sexp /
	    gd_init_state__1.uexp);
    aijp = gd_pow_double_ptr(&d__1, &d__2);
    d__1 = cint * gd_log(xint * uipj + 1.);
    d__2 = (doublereal) (gd_init_state__1.sexp /
	    gd_init_state__1.uexp);
    aipj = gd_pow_double_ptr(&d__1, &d__2);
    d__1 = cint * gd_log(xint * uipjp + 1.);
    d__2 = (doublereal) (gd_init_state__1.sexp /
	    gd_init_state__1.uexp);
    aipjp = gd_pow_double_ptr(&d__1, &d__2);
    eij = tij + aij;
    eipj = tipj + aipj;
    eijp = tijp + aijp;
    eipjp = tipjp + aipjp;
    dtdj = (eijp - eij) / (gd_init_state__1.pa[j] -
	    gd_init_state__1.pa[j - 1]);
    dtdpj = (eipjp - eipj) / (gd_init_state__1.pa[j] -
	    gd_init_state__1.pa[j - 1]);
    epip1 = eij + dtdj * (gd_init_state__1.p1 -
	    gd_init_state__1.pa[j - 1]);
    epipp1 = eipj + dtdpj * (gd_init_state__1.p1 -
	    gd_init_state__1.pa[j - 1]);
    epp2p1 = ((gd_init_state__1.pa[i__] - gd_init_state__1.p2) *
	    epip1 + (gd_init_state__1.p2 - gd_init_state__1.pa[
	    i__ - 1]) * epipp1) / pipmpi;
    d__1 = cint * gd_log(xint * up2p1 + 1.);
    d__2 = (doublereal) (gd_init_state__1.sexp /
	    gd_init_state__1.uexp);
    gd_init_state__1.trnslo = epp2p1 - gd_pow_double_ptr(&d__1, &d__2);
    if (i__ >= 108 || j >= 108) {
	goto L350;
    }
    if (i__ - j - 2 <= 0) {
	goto L350;
    } else {
	goto L355;
    }
L355:
    tip2j = gd_init_state__1.transa[i__ + 2 + j * 109 - 110];
    tip2jp = gd_init_state__1.transa[i__ + 2 + (j + 1) * 109 - 110];
    ti2j2 = gd_init_state__1.transa[i__ + 2 + (j + 2) * 109 - 110];
    tijp2 = gd_init_state__1.transa[i__ + (j + 2) * 109 - 110];
    tipjp2 = gd_init_state__1.transa[i__ + 1 + (j + 2) * 109 - 110];
    d__1 = (doublereal) (real)path_(&gd_init_state__1.pa[i__ + 1], &
	    gd_init_state__1.pa[j - 1], &gd_init_state__1.core, &
	    etap, gt, clock, work, init);
    d__2 = (doublereal) gd_init_state__1.uexp;
    uip2j = (real)pow_dd(&d__1, &d__2);
    d__1 = (doublereal) (real)path_(&gd_init_state__1.pa[i__ - 1], &
	    gd_init_state__1.pa[j + 1], &gd_init_state__1.core, &
	    etap, gt, clock, work, init);
    d__2 = (doublereal) gd_init_state__1.uexp;
    uijp2 = (real)pow_dd(&d__1, &d__2);
    d__1 = (doublereal) (real)path_(&gd_init_state__1.pa[i__], &
	    gd_init_state__1.pa[j + 1], &gd_init_state__1.core, &
	    etap, gt, clock, work, init);
    d__2 = (doublereal) gd_init_state__1.uexp;
    uipjp2 = (real)pow_dd(&d__1, &d__2);
    d__1 = (doublereal) (real)path_(&gd_init_state__1.pa[i__ + 1], &
	    gd_init_state__1.pa[j + 1], &gd_init_state__1.core, &
	    etap, gt, clock, work, init);
    d__2 = (doublereal) gd_init_state__1.uexp;
    ui2j2 = (real)pow_dd(&d__1, &d__2);
    d__1 = (doublereal) (real)path_(&gd_init_state__1.pa[i__ + 1], &
	    gd_init_state__1.pa[j], &gd_init_state__1.core, &etap, gt, clock, work, init)
	    ;
    d__2 = (doublereal) gd_init_state__1.uexp;
    uip2jp = (real)pow_dd(&d__1, &d__2);
    d__1 = cint * gd_log(xint * uijp2 + 1.);
    d__2 = (doublereal) (gd_init_state__1.sexp /
	    gd_init_state__1.uexp);
    aijp2 = gd_pow_double_ptr(&d__1, &d__2);
    d__1 = cint * gd_log(xint * uipjp2 + 1.);
    d__2 = (doublereal) (gd_init_state__1.sexp /
	    gd_init_state__1.uexp);
    aipjp2 = gd_pow_double_ptr(&d__1, &d__2);
    d__1 = cint * gd_log(xint * uip2j + 1.);
    d__2 = (doublereal) (gd_init_state__1.sexp /
	    gd_init_state__1.uexp);
    aip2j = gd_pow_double_ptr(&d__1, &d__2);
    d__1 = cint * gd_log(xint * uip2jp + 1.);
    d__2 = (doublereal) (gd_init_state__1.sexp /
	    gd_init_state__1.uexp);
    aip2jp = gd_pow_double_ptr(&d__1, &d__2);
    d__1 = cint * gd_log(xint * ui2j2 + 1.);
    d__2 = (doublereal) (gd_init_state__1.sexp /
	    gd_init_state__1.uexp);
    ai2j2 = gd_pow_double_ptr(&d__1, &d__2);
    eip2j = tip2j + aip2j;
    eip2jp = tip2jp + aip2jp;
    eijp2 = tijp2 + aijp2;
    eipjp2 = tipjp2 + aipjp2;
    ei2j2 = ti2j2 + ai2j2;
    qintrp_(&gd_init_state__1.pa[j - 1], &gd_init_state__1.pa[j],
	    &gd_init_state__1.pa[j + 1], &eij, &eijp, &eijp2, &
	    gd_init_state__1.p1, &ei, gt, clock, work, init);
    qintrp_(&gd_init_state__1.pa[j - 1], &gd_init_state__1.pa[j],
	    &gd_init_state__1.pa[j + 1], &eipj, &eipjp, &eipjp2, &
	    gd_init_state__1.p1, &ep, gt, clock, work, init);
    qintrp_(&gd_init_state__1.pa[j - 1], &gd_init_state__1.pa[j],
	    &gd_init_state__1.pa[j + 1], &eip2j, &eip2jp, &ei2j2, &
	    gd_init_state__1.p1, &ep2, gt, clock, work, init);
    qintrp_(&gd_init_state__1.pa[i__ - 1], &gd_init_state__1.pa[
	    i__], &gd_init_state__1.pa[i__ + 1], &ei, &ep, &ep2, &
	    gd_init_state__1.p2, &epsil, gt, clock, work, init);
    d__1 = cint * gd_log(xint * up2p1 + 1.);
    d__2 = (doublereal) (gd_init_state__1.sexp /
	    gd_init_state__1.uexp);
    gd_init_state__1.trnslo = epsil - gd_pow_double_ptr(&d__1, &d__2);
L350:
L128:

    return 0;
}
