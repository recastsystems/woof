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
    real ptopc[4], xduo3n[2997]	, xdo3n2[2997]	, xdo3n3[2997]	, xdo3n4[
	    2997]	, prgfdl[81], ab15wd, sko2d, skc1r,
	    sko3r, em1[5040]	, em1wde[5040]	, table1[5040]	, table2[
	    5040]	, table3[5040]	, em3[5040]	, source[420]	, dsrce[420]	, t1[5040], t2[5040]
	    , t4[5040], em1v[5040], em1vw[5040], em3v[5040], axsd[310], rsqr,
	    co251[6724]	, cdt51[6724]	,
	    c2d51[6724]	, co258[6724]	,
	    cdt58[6724]	, c2d58[6724]	,
	    tbpvs[7501], c1xpvs, c2xpvs, stemp[82], gtemp[82], co231[82],
	    co238[82], c2d31[82], c2d38[82], cdt31[82], cdt38[82], co271[82],
	    co278[82], c2d71[82], c2d78[82], cdt71[82], cdt78[82], co2m51[81],
	     co2m58[81], cdtm51[81], cdtm58[81], c2dm51[81], c2dm58[81];
} GDTable;
typedef struct {
    real r1, rsin1, rcos1, rcos2;
} GDClock;

typedef struct {
    real clo89_cldipt[6724];
    real clo89_cldrow[82];
    real e1e290_du[82];
    integer e1e290_it1[245];
    real e1e290_ww1[82];
    real e1e290_ww2[82];
    real e1e290_fyo[82];
    real e1e290_tmp3[82];
    integer e1e290_ival[82];
    real e290_dt[82];
    real e290_du[82];
    real e290_fyo[82];
    real e290_tmp3[82];
    integer e290_ival[82];
    real e2spec_du[82];
    real e2spec_fyo[82];
    real e2spec_tmp3[82];
    integer e2spec_ival[82];
    real e3v88_dt[163];
    real e3v88_du[163];
    integer e3v88_it[163];
    real e3v88_ww1[163];
    real e3v88_ww2[163];
    real e3v88_fxo[163];
    real e3v88_fyo[163];
    real e3v88_tmp3[163];
    integer etara_idat[3];
    real etara_tendl[81];
    real etara_pflip[82];
    real etara_qflip[82];
    real etara_tflip[82];
    real etara_tends[81];
    real etara_cubot[1];
    real etara_cutop[1];
    real etara_qiflip[82];
    real etara_qwflip[82];
    real etara_p8wflip[82];
    real fst88_c__[163];
    real fst88_c2[163];
    real fst88_z1[82];
    real fst88_tc[82];
    real fst88_dt[82];
    real fst88_ss1[82];
    real fst88_ss2[82];
    real fst88_emd[163];
    real fst88_alp[163];
    real fst88_dtc[82];
    real fst88_css[82];
    real fst88_fxo[82];
    real fst88_flx[82];
    real fst88_cts[81];
    real fst88_tpl[163];
    integer fst88_ixo[82];
    real fst88_oss[82];
    real fst88_fac1[82];
    real fst88_dte2[82];
    real fst88_to31d[82];
    real fst88_fbot[1];
    real fst88_csub[163];
    real fst88_rlog[81];
    real fst88_sorc[1230];
    real fst88_pbot[1];
    real fst88_ftop[1];
    real fst88_dtsp[2];
    real fst88_ptop[1];
    real fst88_csub2[163];
    real fst88_e1flx[82];
    real fst88_to3dg[82];
    real fst88_fxoe2[82];
    real fst88_co2sp[82];
    real fst88_avmo3[82];
    real fst88_ctso3[81];
    real fst88_avvo2[82];
    real fst88_vtmp3[82];
    real fst88_to3sp[82];
    real fst88_vsum1[82];
    real fst88_e1cts1[82];
    real fst88_e1ctw1[82];
    real fst88_dsorc[82];
    real fst88_emiss[82];
    real fst88_e1cts2[81];
    real fst88_e1ctw2[81];
    real fst88_flx1e1[1];
    real fst88_excts[81];
    real fst88_gxcts[1];
    real fst88_csour[82];
    real fst88_fxosp[2];
    real fst88_cont1d[82];
    real fst88_delpr1[82];
    real fst88_delpr2[82];
    real fst88_over1d[82];
    real fst88_avpho3[82];
    real fst88_to3spc[81];
    real fst88_heatem[82];
    real fst88_emisdg[82];
    real fst88_delptc[1];
    real fst88_avephi[82];
    real fst88_emspec[2];
    real fst88_contdg[82];
    real fst88_emissb[82];
    real fst88_cnttau[82];
    real fst88_flxnet[82];
    real fst88_totevv[82];
    real lwr88_p[82];
    real lwr88_t[82];
    real lwr88_a1[1];
    real lwr88_vv[81];
    real lwr88_co21[6724];
    real lwr88_co2r[82];
    real lwr88_qh2o[81];
    real lwr88_var1[81];
    real lwr88_var2[81];
    real lwr88_var3[81];
    real lwr88_var4[81];
    real lwr88_emx1[1];
    real lwr88_emx2[1];
    real lwr88_co2r1[82];
    real lwr88_co2r2[82];
    real lwr88_dift[82];
    real lwr88_delp[81];
    real lwr88_tdav[82];
    real lwr88_empl[163];
    real lwr88_d2cd21[82];
    real lwr88_d2cd22[82];
    real lwr88_co2md[81];
    real lwr88_delp2[81];
    real lwr88_co2mr[81];
    real lwr88_toto3[82];
    real lwr88_vsum1[1];
    real lwr88_dco2d1[82];
    real lwr88_vsum3[82];
    real lwr88_dco2d2[82];
    real lwr88_vsum2[1];
    real lwr88_co2m2d[81];
    real lwr88_co2sp1[82];
    real lwr88_co2sp2[82];
    real lwr88_tlsqu[82];
    real lwr88_dco2dt[82];
    real lwr88_co2nbl[81];
    real lwr88_tphio3[82];
    real lwr88_totvo2[82];
    real lwr88_cntval[82];
    real lwr88_tstdav[82];
    real lwr88_totphi[82];
    real lwr88_texpsl[82];
    real o3int_ro31[410];
    real o3int_ro32[410];
    real o3int_rbar[81];
    real o3int_rstd[81];
    real o3int_dduo3[1539];
    real o3int_duo3n[779];
    real o3int_o35deg[2997];
    real o3int_rdata[81];
    real o3int_tempn[19];
    real ozon2d_qo3o3[81];
    real ozon2d_tthan[1];
    integer ozon2d_jjrow[1];
    real radfs_df[82];
    real radfs_uf[82];
    real radfs_qo3[81];
    real radfs_fswl[82];
    real radfs_hswl[82];
    real radfs_ufl[82];
    real radfs_dfl[82];
    real radfs_gdfnbr[1];
    real radfs_cldfac[6724];
    real radfs_gdfndr[1];
    real radfs_eqcmt[82];
    real radfs_gdfvbr[1];
    real radfs_press[82];
    real radfs_gdfvdr[1];
    real radfs_temp[82];
    real radfs_fsw[82];
    real radfs_hsw[82];
    real radfs_rh2o[81];
    real radfs_heatra[81];
    real radfs_tauda[1];
    real radfs_phalf[82];
    real radfs_coszen[1];
    real radfs_grnflx[1];
    real radfs_tthan[1];
    real radfs_do3v[81];
    real radfs_do3vp[81];
    integer radfs_jjrow[1];
    real radfs_topflx[1];
    real radfs_dduo3n[2997];
    real radfs_ddo3n2[2997];
    real radfs_ddo3n3[2997];
    real radfs_alvb[1];
    real radfs_alnb[1];
    real radfs_alvd[1];
    real radfs_alnd[1];
    real radfs_gdfvb[1];
    real radfs_gdfnb[1];
    real radfs_gdfvd[1];
    real radfs_gdfnd[1];
    real radfs_sfcalb[1];
    integer radtn_lmh[1];
    integer radtn_lvl[1];
    real radtn_ozn[81];
    real radtn_tot[1];
    integer radtn_kcld[1];
    real radtn_psfc[1];
    real radtn_pmid[81];
    real radtn_tmid[81];
    real radtn_qmid[81];
    real radtn_czen[1];
    integer radtn_mbot[3];
    integer radtn_mtop[3];
    real radtn_tskn[1];
    real radtn_xlat[1];
    real radtn_cosz[1];
    real radtn_pint[82];
    real radtn_emis[82];
    real radtn_pozn[81];
    real radtn_camt[82];
    integer radtn_kbtm[82];
    integer radtn_ktop[82];
    real radtn_rrcl[984];
    real radtn_ttcl[984];
    real radtn_ccmid[81];
    real radtn_csmid[81];
    real radtn_thmid[81];
    real radtn_fswdn[1];
    real radtn_slmsk[1];
    real radtn_flwup[1];
    integer radtn_nclds[1];
    real radtn_qwmid[81];
    real radtn_fswup[1];
    real radtn_qimid[81];
    real radtn_albedo[1];
    real radtn_cldcfr[3];
    real radtn_cldamt[82];
    real radtn_taudar[1];
    real radtn_flwdns[1];
    real radtn_fswdns[1];
    real radtn_flwups[1];
    real radtn_fswups[1];
    real radtn_fswdnsc[1];
    real spa88_f[81];
    real spa88_x[81];
    real spa88_y[81];
    real spa88_ag[81];
    real spa88_ff[81];
    real spa88_tt[81];
    real spa88_agg[81];
    real spa88_fac1[81];
    real spa88_fac2[81];
    real spa88_ctmp[82];
    real spa88_topm[81];
    real spa88_ctmp2[82];
    real spa88_ctmp3[82];
    real spa88_phitmp[81];
    real spa88_topphi[81];
    real spa88_psitmp[81];
    real swr93_ff[82];
    real swr93_dp[82];
    real swr93_cr[82];
    real swr93_ct[82];
    real swr93_du[82];
    real swr93_ud[82];
    real swr93_pp[82];
    real swr93_ur[82];
    real swr93_pr2[82];
    real swr93_to3[164];
    real swr93_uo3[164];
    real swr93_dfn[82];
    real swr93_ttd[82];
    real swr93_ufn[82];
    real swr93_ttu[82];
    real swr93_ffo3[82];
    real swr93_tco2[164];
    real swr93_uco2[164];
    real swr93_tdo3[82];
    real swr93_duo3[82];
    real swr93_udo3[82];
    real swr93_tmp1[1];
    real swr93_uro3[82];
    real swr93_tuo3[82];
    real swr93_alfa[82];
    real swr93_tcld[82];
    real swr93_refl[1];
    real swr93_secz[1];
    real swr93_tclu[82];
    real swr93_xamt[82];
    real swr93_rray[1];
    real swr93_ffco2[82];
    real swr93_tdcl1[82];
    real swr93_tdcl2[82];
    real swr93_refl2[1];
    real swr93_tdco2[82];
    real swr93_duco2[82];
    real swr93_udco2[82];
    real swr93_ttdb1[82];
    real swr93_urco2[82];
    real swr93_tucl1[82];
    real swr93_tuco2[82];
    real swr93_ttub1[82];
    real swr93_dpcld[82];
    real swr93_alfau[82];
    real swr93_ccmax[1];
    real swr93_pptop[82];
    real swr93_dfnclu[82];
    real swr93_dfntop[12];
    real swr93_dfntrn[82];
    real swr93_ufnclu[82];
    real swr93_ufntrn[82];
} GDScratch;
#define gd_tables__1 (*gt)
#define gd_clock__1 (*clock)














 GD_DEVICE int cal_mon_day__(integer *julday, integer *julyr, integer *
	jmonth, integer *jday, GDTable *gt, GDClock *clock, GDScratch *work)
{


    integer month[12] = { 31,28,31,30,31,30,31,31,30,31,30,31 };

    integer i__;
    logical leap, not_find_date__;
    integer itmpday, itmpmon;

    not_find_date__ = TRUE_;
    itmpday = *julday;
    itmpmon = 1;
    leap = FALSE_;
    if (*julyr % 4 == 0) {
	month[1] = 29;
	leap = TRUE_;
    }
    i__ = 1;
    while(not_find_date__) {
	if (itmpday > month[i__ - 1]) {
	    itmpday -= month[i__ - 1];
	} else {
	    *jday = itmpday;
	    *jmonth = i__;
	    not_find_date__ = FALSE_;
	}
	++i__;
    }
    return 0;
}
















 GD_DEVICE int clo89_(real *cldfac, real *camt, integer *nclds, integer
	*kbtm, integer *ktop, integer *ids, integer *ide, integer *jds,
	integer *jde, integer *kds, integer *kde, integer *ims, integer *ime,
	integer *jms, integer *jme, integer *kms, integer *kme, integer *its,
	integer *ite, integer *jts, integer *jte, integer *kts, integer *kte, GDTable *gt, GDClock *clock, GDScratch *work)
{

    integer i__1, i__2, i__3, i__4, i__5;


    integer i__, j, k, l, k1, k2, kb, nc, ip, iq, ir, kp, kt, lm1, lm2, lm3,
	    lp1, lp2, lp3;
    real xcld;
    integer myie, itop, jtop, myis;

    real *cldipt=work->clo89_cldipt;
    real *cldrow=work->clo89_cldrow;


    ktop -= 2;
    kbtm -= 2;
    --nclds;
    camt -= 2;
    cldfac -= 84;


    l = *kte;
    lp1 = l + 1;
    lp2 = l + 2;
    lp3 = l + 3;
    lm1 = l - 1;
    lm2 = l - 2;
    lm3 = l - 3;
    myis = *its;
    myie = *ite;
    i__1 = myie;
    for (iq = myis; iq <= i__1; iq += 64) {
	itop = iq + 63;
	if (itop > myie) {
	    itop = myie;
	}
	jtop = itop - iq + 1;
	i__2 = jtop;
	for (ip = 1; ip <= i__2; ++ip) {
	    ir = iq + ip - 1;
	    if (nclds[ir] == 0) {
		i__3 = lp1;
		for (j = 1; j <= i__3; ++j) {
		    i__4 = lp1;
		    for (i__ = 1; i__ <= i__4; ++i__) {
			cldipt[i__ + (j + ip * 82) * 82 - 6807] = 1.f;

		    }
		}
	    }
	    if (nclds[ir] >= 1) {
		xcld = 1.f - camt[ir + 2];
		k1 = ktop[ir + 2] + 1;
		k2 = kbtm[ir + 2];
		i__4 = lp1;
		for (j = 1; j <= i__4; ++j) {
		    cldrow[j - 1] = 1.f;

		}
		i__4 = k2;
		for (j = 1; j <= i__4; ++j) {
		    cldrow[j - 1] = xcld;

		}

		i__4 = k1, i__3 = k2 + 1;
		kb = max(i__4,i__3);
		i__4 = lp1;
		for (k = kb; k <= i__4; ++k) {
		    i__3 = lp1;
		    for (kp = 1; kp <= i__3; ++kp) {
			cldipt[kp + (k + ip * 82) * 82 - 6807] = cldrow[kp -
				1];

		    }
		}
		i__3 = lp1;
		for (j = 1; j <= i__3; ++j) {
		    cldrow[j - 1] = 1.f;

		}
		i__3 = lp1;
		for (j = k1; j <= i__3; ++j) {
		    cldrow[j - 1] = xcld;

		}

		i__3 = k1 - 1;
		kt = min(i__3,k2);
		i__3 = kt;
		for (k = 1; k <= i__3; ++k) {
		    i__4 = lp1;
		    for (kp = 1; kp <= i__4; ++kp) {
			cldipt[kp + (k + ip * 82) * 82 - 6807] = cldrow[kp -
				1];

		    }
		}
		if (k2 + 1 <= k1 - 1) {
		    i__4 = k1 - 1;
		    for (j = k2 + 1; j <= i__4; ++j) {
			i__3 = lp1;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    cldipt[i__ + (j + ip * 82) * 82 - 6807] = 1.f;

			}
		    }
		} else if (k1 <= k2) {
		    i__3 = k2;
		    for (j = k1; j <= i__3; ++j) {
			i__4 = lp1;
			for (i__ = 1; i__ <= i__4; ++i__) {
			    cldipt[i__ + (j + ip * 82) * 82 - 6807] = xcld;

			}
		    }
		}
	    }
	    if (nclds[ir] >= 2) {
		i__4 = nclds[ir];
		for (nc = 2; nc <= i__4; ++nc) {
		    xcld = 1.f - camt[ir + (nc + 1)];
		    k1 = ktop[ir + (nc + 1)] + 1;
		    k2 = kbtm[ir + (nc + 1)];
		    i__3 = lp1;
		    for (j = 1; j <= i__3; ++j) {
			cldrow[j - 1] = 1.f;

		    }
		    i__3 = k2;
		    for (j = 1; j <= i__3; ++j) {
			cldrow[j - 1] = xcld;

		    }

		    i__3 = k1, i__5 = k2 + 1;
		    kb = max(i__3,i__5);
		    i__3 = lp1;
		    for (k = kb; k <= i__3; ++k) {
			i__5 = lp1;
			for (kp = 1; kp <= i__5; ++kp) {
			    cldipt[kp + (k + ip * 82) * 82 - 6807] *= cldrow[
				    kp - 1];

			}
		    }
		    i__5 = lp1;
		    for (j = 1; j <= i__5; ++j) {
			cldrow[j - 1] = 1.f;

		    }
		    i__5 = lp1;
		    for (j = k1; j <= i__5; ++j) {
			cldrow[j - 1] = xcld;

		    }

		    i__5 = k1 - 1;
		    kt = min(i__5,k2);
		    i__5 = kt;
		    for (k = 1; k <= i__5; ++k) {
			i__3 = lp1;
			for (kp = 1; kp <= i__3; ++kp) {
			    cldipt[kp + (k + ip * 82) * 82 - 6807] *= cldrow[
				    kp - 1];

			}
		    }
		    if (k1 <= k2) {
			i__3 = k2;
			for (j = k1; j <= i__3; ++j) {
			    i__5 = lp1;
			    for (i__ = 1; i__ <= i__5; ++i__) {
				cldipt[i__ + (j + ip * 82) * 82 - 6807] *=
					xcld;

			    }
			}
		    }

		}
	    }

	}
	i__2 = lp1;
	for (j = 1; j <= i__2; ++j) {
	    i__4 = lp1;
	    for (i__ = 1; i__ <= i__4; ++i__) {
		i__5 = jtop;
		for (ip = 1; ip <= i__5; ++ip) {
		    ir = iq + ip - 1;
		    cldfac[ir + (i__ + j * 82)] = cldipt[i__ + (j + ip * 82) *
			     82 - 6807];

		}
	    }
	}

    }
    return 0;
}
















 GD_DEVICE int e1e290_(real *g1, real *g2, real *g3, real *g4, real *g5,
	 real *emiss, real *fxoe1, real *dte1, real *fxoe2, real *dte2, real *
	avephi, real *temp, real *t, real *h16e1, real *ten, real *hp1, real *
	h28e1, real *haf, integer *ids, integer *ide, integer *jds, integer *
	jde, integer *kds, integer *kde, integer *ims, integer *ime, integer *
	jms, integer *jme, integer *kms, integer *kme, integer *its, integer *
	ite, integer *jts, integer *jte, integer *kts, integer *kte, GDTable *gt, GDClock *clock, GDScratch *work)
{

    integer i__1, i__2;
    real r__1, r__2, r__3, r__4, r__5, r__6;





    integer i__, k, l, ll;

    real *du=work->e1e290_du;

    integer kp;
    integer lm1;
    integer lm2;
    integer lm3;
    integer lp1;
    integer *it1=work->e1e290_it1;
    integer lp2;
    integer lp3;

    real *ww1=work->e1e290_ww1;
    real *ww2=work->e1e290_ww2;
    real *fyo=work->e1e290_fyo;
    integer llm1, llm2, llp1;

    real *tmp3=work->e1e290_tmp3;

    integer *ival=work->e1e290_ival;
    integer myie;
    integer myis;


    t -= 2;
    temp -= 2;
    avephi -= 2;
    dte2 -= 2;
    fxoe2 -= 2;
    dte1 -= 2;
    fxoe1 -= 2;
    emiss -= 2;
    g5 -= 2;
    g4 -= 2;
    g3 -= 2;
    g2 -= 2;
    g1 -= 2;


    l = *kte;
    lp1 = l + 1;
    lp2 = l + 2;
    lp3 = l + 3;
    llp1 = (l << 1) + 1;
    lm1 = l - 1;
    lm2 = l - 2;
    lm3 = l - 3;
    ll = l << 1;
    llm2 = ll - 2;
    llm1 = ll - 1;
    myis = *its;
    myie = *ite;
    i__1 = lp1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    tmp3[i__ + k - 2] = (real)r_lg10(&avephi[i__ + k]) + *h16e1;
	    r__1 = tmp3[i__ + k - 2] * *ten;
	    fyo[i__ + k - 2] = (real)r_int(&r__1);
	    du[i__ + k - 2] = tmp3[i__ + k - 2] - *hp1 * fyo[i__ + k - 2];
	    fyo[i__ + k - 2] = *h28e1 * fyo[i__ + k - 2];
	    ival[i__ + k - 2] = fyo[i__ + k - 2] + fxoe2[i__ + k];
	    r__1 = gd_tables__1.t1[ival[i__ + k - 2] - 1] + du[i__ + k - 2] *
		    gd_tables__1.t2[ival[i__ + k - 2] - 1];
	    emiss[i__ + k] = r__1 + dte2[i__ + k] * gd_tables__1.t4[ival[i__
		    + k - 2] - 1];

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	emiss[i__ + l] = *haf * (emiss[i__ + l] + emiss[i__ + lp1]);

    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	it1[i__ - 1] = fxoe1[i__ + 1];
	ww1[i__ - 1] = *ten - dte1[i__ + 1];
	ww2[i__ - 1] = *hp1;

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    it1[i__ + (k + 1) - 2] = fyo[i__ + k - 2] + fxoe1[i__ + (k + 1)];
	    it1[i__ + (lp2 + k - 1) - 2] = fyo[i__ + k - 2] + fxoe1[i__ + k];
	    ww1[i__ + (k + 1) - 2] = *ten - dte1[i__ + (k + 1)];
	    ww2[i__ + (k + 1) - 2] = *hp1 - du[i__ + k - 2];

	}
    }
    i__1 = l;
    for (kp = 1; kp <= i__1; ++kp) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    it1[i__ + (kp + llp1) - 2] = fyo[i__ + kp - 2] + fxoe1[i__ + 1];

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	r__1 = ww1[i__ - 1] * ww2[i__ - 1];
	r__2 = ww2[i__ - 1] * dte1[i__ + 1];
	g1[i__ + 1] = r__1 * gd_tables__1.em1v[it1[i__ - 1] - 1] + r__2 *
		gd_tables__1.em1v[it1[i__ - 1]];
	g3[i__ + 1] = g1[i__ + 1];

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    r__3 = ww1[i__ + (k + 1) - 2] * ww2[i__ + (k + 1) - 2];
	    r__4 = ww2[i__ + (k + 1) - 2] * dte1[i__ + (k + 1)];
	    r__2 = r__3 * gd_tables__1.em1v[it1[i__ + (k + 1) - 2] - 1] +
		    r__4 * gd_tables__1.em1v[it1[i__ + (k + 1) - 2]];
	    r__5 = ww1[i__ + (k + 1) - 2] * du[i__ + k - 2];
	    r__1 = r__2 + r__5 * gd_tables__1.em1v[it1[i__ + (k + 1) - 2] +
		    27];
	    r__6 = dte1[i__ + (k + 1)] * du[i__ + k - 2];
	    g1[i__ + (k + 1)] = r__1 + r__6 * gd_tables__1.em1v[it1[i__ + (k
		    + 1) - 2] + 28];
	    r__3 = ww1[i__ + k - 2] * ww2[i__ + (k + 1) - 2];
	    r__4 = ww2[i__ + (k + 1) - 2] * dte1[i__ + k];
	    r__2 = r__3 * gd_tables__1.em1v[it1[i__ + (k + lp2 - 1) - 2] - 1]
		    + r__4 * gd_tables__1.em1v[it1[i__ + (k + lp2 - 1) - 2]];
	    r__5 = ww1[i__ + k - 2] * du[i__ + k - 2];
	    r__1 = r__2 + r__5 * gd_tables__1.em1v[it1[i__ + (k + lp2 - 1) -
		    2] + 27];
	    r__6 = dte1[i__ + k] * du[i__ + k - 2];
	    g2[i__ + k] = r__1 + r__6 * gd_tables__1.em1v[it1[i__ + (k + lp2
		    - 1) - 2] + 28];

	}
    }
    i__1 = lp1;
    for (kp = 2; kp <= i__1; ++kp) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    r__3 = ww1[i__ - 1] * ww2[i__ + kp - 2];
	    r__4 = ww2[i__ + kp - 2] * dte1[i__ + 1];
	    r__2 = r__3 * gd_tables__1.em1v[it1[i__ + (ll + kp) - 2] - 1] +
		    r__4 * gd_tables__1.em1v[it1[i__ + (ll + kp) - 2]];
	    r__5 = ww1[i__ - 1] * du[i__ + (kp - 1) - 2];
	    r__1 = r__2 + r__5 * gd_tables__1.em1v[it1[i__ + (ll + kp) - 2] +
		    27];
	    r__6 = dte1[i__ + 1] * du[i__ + (kp - 1) - 2];
	    g3[i__ + kp] = r__1 + r__6 * gd_tables__1.em1v[it1[i__ + (ll + kp)
		     - 2] + 28];

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	r__1 = ww1[i__ - 1] * ww2[i__ - 1];
	r__2 = ww2[i__ - 1] * dte1[i__ + 1];
	g4[i__ + 1] = r__1 * gd_tables__1.em1vw[it1[i__ - 1] - 1] + r__2 *
		gd_tables__1.em1vw[it1[i__ - 1]];

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    r__3 = ww1[i__ + (k + 1) - 2] * ww2[i__ + (k + 1) - 2];
	    r__4 = ww2[i__ + (k + 1) - 2] * dte1[i__ + (k + 1)];
	    r__2 = r__3 * gd_tables__1.em1vw[it1[i__ + (k + 1) - 2] - 1] +
		    r__4 * gd_tables__1.em1vw[it1[i__ + (k + 1) - 2]];
	    r__5 = ww1[i__ + (k + 1) - 2] * du[i__ + k - 2];
	    r__1 = r__2 + r__5 * gd_tables__1.em1vw[it1[i__ + (k + 1) - 2] +
		    27];
	    r__6 = dte1[i__ + (k + 1)] * du[i__ + k - 2];
	    g4[i__ + (k + 1)] = r__1 + r__6 * gd_tables__1.em1vw[it1[i__ + (k
		    + 1) - 2] + 28];
	    r__3 = ww1[i__ + k - 2] * ww2[i__ + (k + 1) - 2];
	    r__4 = ww2[i__ + (k + 1) - 2] * dte1[i__ + k];
	    r__2 = r__3 * gd_tables__1.em1vw[it1[i__ + (k + lp2 - 1) - 2] - 1]
		     + r__4 * gd_tables__1.em1vw[it1[i__ + (k + lp2 - 1) - 2]]
		    ;
	    r__5 = ww1[i__ + k - 2] * du[i__ + k - 2];
	    r__1 = r__2 + r__5 * gd_tables__1.em1vw[it1[i__ + (k + lp2 - 1) -
		    2] + 27];
	    r__6 = dte1[i__ + k] * du[i__ + k - 2];
	    g5[i__ + k] = r__1 + r__6 * gd_tables__1.em1vw[it1[i__ + (k + lp2
		    - 1) - 2] + 28];

	}
    }
    return 0;
}
















 GD_DEVICE int e290_(real *emissb, real *emiss, real *avephi, integer *
	klen, real *fxoe2, real *dte2, real *h16e1, real *hp1, real *h28e1,
	real *haf, real *ten, integer *ids, integer *ide, integer *jds,
	integer *jde, integer *kds, integer *kde, integer *ims, integer *ime,
	integer *jms, integer *jme, integer *kms, integer *kme, integer *its,
	integer *ite, integer *jts, integer *jte, integer *kts, integer *kte, GDTable *gt, GDClock *clock, GDScratch *work)
{

    integer i__1, i__2;
    real r__1;





    integer i__, k, l;

    real *dt=work->e290_dt;
    real *du=work->e290_du;
    integer ll, lm1, lm2, lm3, lp1, lp2, lp3;

    real *fyo=work->e290_fyo;
    integer llm1, llm2, llp1;

    real *tmp3=work->e290_tmp3;

    integer *ival=work->e290_ival;
    integer myie;
    integer myis;


    dte2 -= 2;
    fxoe2 -= 2;
    avephi -= 2;
    emiss -= 2;
    emissb -= 2;


    l = *kte;
    lp1 = l + 1;
    lp2 = l + 2;
    lp3 = l + 3;
    llp1 = (l << 1) + 1;
    lm1 = l - 1;
    lm2 = l - 2;
    lm3 = l - 3;
    ll = l << 1;
    llm2 = ll - 2;
    llm1 = ll - 1;
    myis = *its;
    myie = *ite;
    i__1 = lp2 - *klen;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    tmp3[i__ + k - 2] = (real)r_lg10(&avephi[i__ + (*klen + k - 1)])
		    + *h16e1;
	    r__1 = tmp3[i__ + k - 2] * *ten;
	    fyo[i__ + k - 2] = (real)r_int(&r__1);
	    du[i__ + k - 2] = tmp3[i__ + k - 2] - *hp1 * fyo[i__ + k - 2];
	    fyo[i__ + k - 2] = *h28e1 * fyo[i__ + k - 2];
	    ival[i__ + k - 2] = fyo[i__ + k - 2] + fxoe2[i__ + (*klen + k - 1)
		    ];
	    r__1 = gd_tables__1.t1[ival[i__ + k - 2] - 1] + du[i__ + k - 2] *
		    gd_tables__1.t2[ival[i__ + k - 2] - 1];
	    emiss[i__ + (*klen + k - 1)] = r__1 + dte2[i__ + (*klen + k - 1)]
		    * gd_tables__1.t4[ival[i__ + k - 2] - 1];

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	emiss[i__ + l] = *haf * (emiss[i__ + l] + emiss[i__ + lp1]);

    }
    i__2 = lp1 - *klen;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    dt[i__ + k - 2] = dte2[i__ + (*klen - 1)];
	    ival[i__ + k - 2] = fyo[i__ + k - 2] + fxoe2[i__ + (*klen - 1)];

	}
    }
    i__1 = lp1 - *klen;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    r__1 = gd_tables__1.t1[ival[i__ + k - 2] - 1] + du[i__ + k - 2] *
		    gd_tables__1.t2[ival[i__ + k - 2] - 1];
	    emissb[i__ + (*klen + k - 1)] = r__1 + dt[i__ + k - 2] *
		    gd_tables__1.t4[ival[i__ + k - 2] - 1];

	}
    }
    return 0;
}
















 GD_DEVICE int e2spec_(real *emiss, real *avephi, real *fxosp, real *
	dtsp, real *h16e1, real *ten, real *h28e1, real *hp1, integer *ids,
	integer *ide, integer *jds, integer *jde, integer *kds, integer *kde,
	integer *ims, integer *ime, integer *jms, integer *jme, integer *kms,
	integer *kme, integer *its, integer *ite, integer *jts, integer *jte,
	integer *kts, integer *kte, GDTable *gt, GDClock *clock, GDScratch *work)
{

    integer i__1;
    real r__1;





    integer i__, k;

    real *du=work->e2spec_du;
    real *fyo=work->e2spec_fyo;
    real *tmp3=work->e2spec_tmp3;

    integer *ival=work->e2spec_ival;
    integer myie;
    integer myis;


    dtsp -= 2;
    fxosp -= 2;
    avephi -= 2;
    emiss -= 2;


    myis = *its;
    myie = *ite;
    for (k = 1; k <= 2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    tmp3[i__ + k - 2] = (real)r_lg10(&avephi[i__ + k]) + *h16e1;
	    r__1 = tmp3[i__ + k - 2] * *ten;
	    fyo[i__ + k - 2] = (real)r_int(&r__1);
	    du[i__ + k - 2] = tmp3[i__ + k - 2] - *hp1 * fyo[i__ + k - 2];
	    ival[i__ + k - 2] = *h28e1 * fyo[i__ + k - 2] + fxosp[i__ + k];
	    r__1 = gd_tables__1.t1[ival[i__ + k - 2] - 1] + du[i__ + k - 2] *
		    gd_tables__1.t2[ival[i__ + k - 2] - 1];
	    emiss[i__ + k] = r__1 + dtsp[i__ + k] * gd_tables__1.t4[ival[i__
		    + k - 2] - 1];

	}
    }
    return 0;
}
















 GD_DEVICE int e3v88_(real *emv, real *tv, real *av, real *ten, real *
	hp1, real *h28e1, real *h16e1, integer *ids, integer *ide, integer *
	jds, integer *jde, integer *kds, integer *kde, integer *ims, integer *
	ime, integer *jms, integer *jme, integer *kms, integer *kme, integer *
	its, integer *ite, integer *jts, integer *jte, integer *kts, integer *
	kte, GDTable *gt, GDClock *clock, GDScratch *work)
{

    integer i__1, i__2;
    real r__1, r__2, r__3, r__4, r__5, r__6;





    integer i__, k, l;

    real *dt=work->e3v88_dt;
    real *du=work->e3v88_du;

    integer *it=work->e3v88_it;

    real *ww1=work->e3v88_ww1;
    real *ww2=work->e3v88_ww2;
    real *fxo=work->e3v88_fxo;
    real *fyo=work->e3v88_fyo;
    integer llp1;

    real *tmp3=work->e3v88_tmp3;
    integer myie, myis;


    av -= 2;
    tv -= 2;
    emv -= 2;


    l = *kte;
    llp1 = (l << 1) + 1;
    myis = *its;
    myie = *ite;
    i__1 = llp1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    r__1 = tv[i__ + k] * *hp1;
	    fxo[i__ + k - 2] = (real)r_int(&r__1);
	    tmp3[i__ + k - 2] = (real)r_lg10(&av[i__ + k]) + *h16e1;
	    dt[i__ + k - 2] = tv[i__ + k] - *ten * fxo[i__ + k - 2];
	    r__1 = tmp3[i__ + k - 2] * *ten;
	    fyo[i__ + k - 2] = (real)r_int(&r__1);
	    du[i__ + k - 2] = tmp3[i__ + k - 2] - *hp1 * fyo[i__ + k - 2];
	    it[i__ + k - 2] = fxo[i__ + k - 2] + fyo[i__ + k - 2] * *h28e1;
	    ww1[i__ + k - 2] = *ten - dt[i__ + k - 2];
	    ww2[i__ + k - 2] = *hp1 - du[i__ + k - 2];
	    r__3 = ww1[i__ + k - 2] * ww2[i__ + k - 2];
	    r__4 = ww2[i__ + k - 2] * dt[i__ + k - 2];
	    r__2 = r__3 * gd_tables__1.em3v[it[i__ + k - 2] - 10] + r__4 *
		    gd_tables__1.em3v[it[i__ + k - 2] - 9];
	    r__5 = ww1[i__ + k - 2] * du[i__ + k - 2];
	    r__1 = r__2 + r__5 * gd_tables__1.em3v[it[i__ + k - 2] + 18];
	    r__6 = dt[i__ + k - 2] * du[i__ + k - 2];
	    emv[i__ + k] = r__1 + r__6 * gd_tables__1.em3v[it[i__ + k - 2] +
		    19];

	}
    }
    return 0;
}
















 GD_DEVICE int etara_(real *dt, real *thraten, real *thratenlw, real *
	thratensw, real *cldfra, real *pi3d, real *xland, real *p8w, real *
	dz8w, real *rho_phy__, real *p_phy__, real *t, real *qv, real *qw,
	real *qi, real *qs, real *tsk2d, real *glw, real *rswin, real *gsw,
	real *rswinc, real *rswtoa, real *rlwtoa, real *czmean, real *glat,
	real *glon, real *htop, real *hbot, real *htopr, real *hbotr, real *
	albedo, real *cuppt, real *vegfra, real *snow, real *g, real *gmt,
	integer *nstepra, integer *nphs, integer *itimestep, real *xtime,
	real *julian, integer *julyr, integer *julday, logical *gfdl_lw__,
	logical *gfdl_sw__, real *cfracl, real *cfracm, real *cfrach, real *
	acfrst, integer *ncfrst, real *acfrcv, integer *ncfrcv, integer *ids,
	integer *ide, integer *jds, integer *jde, integer *kds, integer *kde,
	integer *ims, integer *ime, integer *jms, integer *jme, integer *kms,
	integer *kme, integer *its, integer *ite, integer *jts, integer *jte,
	integer *kts, integer *kte, GDTable *gt, GDClock *clock, GDScratch *work)
{

    integer i__1, i__2, i__3;
    real r__1, r__2;





    integer i__, j, k;
    extern GD_DEVICE int cal_mon_day__(integer *, integer *, integer *,
	     integer *, GDTable *gt, GDClock *clock, GDScratch *work);

    integer *idat=work->etara_idat;
    integer jday;
    integer kflip;

    real *tendl=work->etara_tendl;
    extern GD_DEVICE int radtn_(real *, real *, real *, real *, real *,
	     real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, integer *, real *, integer *, real *,
	    real *, real *, real *, real *, real *, integer *, integer *,
	    real *, real *, integer *, integer *, integer *, integer *, real *
	    , real *, real *, real *, real *, real *, real *, real *, real *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *, GDTable *gt, GDClock *clock, GDScratch *work);

    real *pflip=work->etara_pflip;
    real *qflip=work->etara_qflip;
    real *tflip=work->etara_tflip;
    real *tends=work->etara_tends;
    real *cubot=work->etara_cubot;
    integer ihour, ihrst;

    real *cutop=work->etara_cutop;
    real *qiflip=work->etara_qiflip;
    extern GD_DEVICE int solard_(integer *, integer *, integer *,
	    integer *, GDTable *gt, GDClock *clock, GDScratch *work);
    integer jmonth;

    real *qwflip=work->etara_qwflip;
    real *p8wflip=work->etara_p8wflip;


    ncfrcv -= 2;
    acfrcv -= 2;
    ncfrst -= 2;
    acfrst -= 2;
    cfrach -= 2;
    cfracm -= 2;
    cfracl -= 2;
    snow -= 2;
    vegfra -= 2;
    cuppt -= 2;
    albedo -= 2;
    hbotr -= 2;
    htopr -= 2;
    hbot -= 2;
    htop -= 2;
    glon -= 2;
    glat -= 2;
    czmean -= 2;
    rlwtoa -= 2;
    rswtoa -= 2;
    rswinc -= 2;
    gsw -= 2;
    rswin -= 2;
    glw -= 2;
    tsk2d -= 2;
    qs -= 84;
    qi -= 84;
    qw -= 84;
    qv -= 84;
    t -= 84;
    p_phy__ -= 84;
    rho_phy__ -= 84;
    dz8w -= 84;
    p8w -= 84;
    xland -= 2;
    pi3d -= 84;
    cldfra -= 84;
    thratensw -= 84;
    thratenlw -= 84;
    thraten -= 84;


    if (*gfdl_lw__ && *gfdl_sw__) {
	goto L100;
    }
    i__1 = *jme;
    for (j = *jms; j <= i__1; ++j) {
	i__2 = *kme;
	for (k = *kms; k <= i__2; ++k) {
	    i__3 = *ime;
	    for (i__ = *ims; i__ <= i__3; ++i__) {
		cldfra[i__ + (k + j * 82)] = 0.f;
	    }
	}
    }
    i__1 = *kme;
    for (k = *kms; k <= i__1; ++k) {
	kflip = *kme + 1 - k;
	i__2 = *jte;
	for (j = *jts; j <= i__2; ++j) {
	    i__3 = *ite;
	    for (i__ = *its; i__ <= i__3; ++i__) {
		p8wflip[i__ + (k + j * 82) - 84] = p8w[i__ + (kflip + j * 82)]
			;
	    }
	}
    }
    i__1 = *kte;
    for (k = *kts; k <= i__1; ++k) {
	kflip = *kte + 1 - k;
	i__2 = *jte;
	for (j = *jts; j <= i__2; ++j) {
	    i__3 = *ite;
	    for (i__ = *its; i__ <= i__3; ++i__) {
		tflip[i__ + (k + j * 82) - 84] = t[i__ + (kflip + j * 82)];

		r__1 = 0.f, r__2 = qv[i__ + (kflip + j * 82)] / (qv[i__ + (
			kflip + j * 82)] + 1.f);
		qflip[i__ + (k + j * 82) - 84] = max(r__1,r__2);

		r__1 = qw[i__ + (kflip + j * 82)];
		qwflip[i__ + (k + j * 82) - 84] = max(r__1,0.f);

		r__1 = qs[i__ + (kflip + j * 82)];
		qiflip[i__ + (k + j * 82) - 84] = max(r__1,0.f);
		if (TRUE_) {
		    qiflip[i__ + (k + j * 82) - 84] += qi[i__ + (kflip + j *
			    82)];
		}
		pflip[i__ + (k + j * 82) - 84] = p_phy__[i__ + (kflip + j *
			82)];
	    }
	}
    }
    i__1 = *jte;
    for (j = *jts; j <= i__1; ++j) {
	i__2 = *ite;
	for (i__ = *its; i__ <= i__2; ++i__) {
	    cubot[i__ + j - 2] = *kte + 1 - hbot[i__ + j];
	    cutop[i__ + j - 2] = *kte + 1 - htop[i__ + j];
	}
    }
    cal_mon_day__(julday, julyr, &jmonth, &jday, gt, clock, work);
    idat[0] = jmonth;
    idat[1] = jday;
    idat[2] = *julyr;
    ihrst = i_nint(gmt);
    r__1 = *xtime / 60.f;
    ihour = (ihrst + i_nint(&r__1)) % 24;
    solard_(&ihour, &jday, &jmonth, julyr, gt, clock, work);
    radtn_(dt, tflip, qflip, qwflip, qiflip, pflip, p8wflip, &xland[2], &
	    tsk2d[2], &glat[2], &glon[2], cutop, cubot, &albedo[2], &cuppt[2],
	     &acfrcv[2], &ncfrcv[2], &acfrst[2], &ncfrst[2], &vegfra[2], &
	    snow[2], &glw[2], &gsw[2], &rswin[2], &rswinc[2], idat, &ihrst,
	    xtime, julian, nstepra, nstepra, nphs, itimestep, tends, tendl, &
	    cldfra[84], &rswtoa[2], &rlwtoa[2], &czmean[2], &cfracl[2], &
	    cfracm[2], &cfrach[2], ids, ide, jds, jde, kds, kde, ims, ime,
	    jms, jme, kms, kme, its, ite, jts, jte, kts, kte, gt, clock, work);
    if (*gfdl_lw__) {
	i__1 = *jte;
	for (j = *jts; j <= i__1; ++j) {
	    i__2 = *kte;
	    for (k = *kts; k <= i__2; ++k) {
		kflip = *kte + 1 - k;
		i__3 = *ite;
		for (i__ = *its; i__ <= i__3; ++i__) {
		    thratenlw[i__ + (k + j * 82)] = tendl[i__ + (kflip + j *
			    81) - 83] / pi3d[i__ + (k + j * 82)];
		    thratensw[i__ + (k + j * 82)] = tends[i__ + (kflip + j *
			    81) - 83] / pi3d[i__ + (k + j * 82)];
		    thraten[i__ + (k + j * 82)] += thratenlw[i__ + (k + j *
			    82)];
		}
	    }
	}
    }
    if (*gfdl_sw__) {
	i__1 = *jte;
	for (j = *jts; j <= i__1; ++j) {
	    i__2 = *kte;
	    for (k = *kts; k <= i__2; ++k) {
		kflip = *kte + 1 - k;
		i__3 = *ite;
		for (i__ = *its; i__ <= i__3; ++i__) {
		    thratensw[i__ + (k + j * 82)] = tends[i__ + (kflip + j *
			    81) - 83] / pi3d[i__ + (k + j * 82)];
		}
	    }
	}
    }
    i__1 = *jte;
    for (j = *jts; j <= i__1; ++j) {
	i__2 = *ite;
	for (i__ = *its; i__ <= i__2; ++i__) {
	    hbotr[i__ + j] = hbot[i__ + j];
	    htopr[i__ + j] = htop[i__ + j];
	    hbot[i__ + j] = (real) (*kte + 1);
	    htop[i__ + j] = 0.f;
	    cuppt[i__ + j] = 0.f;
	}
    }
L100:
    if (*gfdl_sw__) {
	i__1 = *jte;
	for (j = *jts; j <= i__1; ++j) {
	    i__2 = *kte;
	    for (k = *kts; k <= i__2; ++k) {
		kflip = *kte + 1 - k;
		i__3 = *ite;
		for (i__ = *its; i__ <= i__3; ++i__) {
		    thraten[i__ + (k + j * 82)] += thratensw[i__ + (k + j *
			    82)];
		}
	    }
	}
    }
    return 0;
}
















GD_DEVICE real fpvs_(real *t, GDTable *gt, GDClock *clock, GDScratch *work)
{

    real ret_val, r__1, r__2;


    real xj;
    integer jx;



    r__2 = gd_tables__1.c1xpvs + gd_tables__1.c2xpvs * *t;
    r__1 = max(r__2,1.f);
    xj = min(r__1,7501.f);
    jx = min(xj,7500.f);
    ret_val = gd_tables__1.tbpvs[jx - 1] + (xj - jx) * (gd_tables__1.tbpvs[jx]
	     - gd_tables__1.tbpvs[jx - 1]);
    return ret_val;
}
















 GD_DEVICE int fst88_(real *heatra, real *grnflx, real *topflx, real *
	qh2o, real *press, real *p, real *delp, real *delp2, real *temp, real
	*t, real *cldfac, integer *nclds, integer *ktop, integer *kbtm, real *
	camt, real *co21, real *co2nbl, real *co2sp1, real *co2sp2, real *
	var1, real *var2, real *var3, real *var4, real *cntval, real *toto3,
	real *tphio3, real *totphi, real *totvo2, real *emx1, real *emx2,
	real *empl, real *bo3rnd, real *ao3rnd, real *apcm, real *bpcm, real *
	atpcm, real *btpcm, real *acomb, real *bcomb, real *betacm, real *ten,
	 real *hp1, real *haf, real *one, real *four, real *hm1ez, real *
	radcon, real *quartr, real *two, real *hm6666m2, real *hmp66667, real
	*hmp5, real *hp166666, real *h41666m2, real *radcon1, real *h16e1,
	real *h28e1, real *h25e2, real *h44194m2, real *h1p41819, integer *
	ids, integer *ide, integer *jds, integer *jde, integer *kds, integer *
	kde, integer *ims, integer *ime, integer *jms, integer *jme, integer *
	kms, integer *kme, integer *its, integer *ite, integer *jts, integer *
	jte, integer *kts, integer *kte, GDTable *gt, GDClock *clock, GDScratch *work)
{

    integer i__1, i__2, i__3;
    real r__1, r__2;






    real *c__=work->fst88_c__;
    integer i__, k, l;

    real *c2=work->fst88_c2;
    integer j1, j3;

    real *z1=work->fst88_z1;
    integer kk;

    real *tc=work->fst88_tc;
    real *dt=work->fst88_dt;
    integer ll, kp;
    extern GD_DEVICE int e290_(real *, real *, real *, integer *, real
	    *, real *, real *, real *, real *, real *, real *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, GDTable *gt, GDClock *clock, GDScratch *work);
    integer lm1, lm2, lm3, lp1, lp2, lp3;

    real *ss1=work->fst88_ss1;
    real *ss2=work->fst88_ss2;
    real *emd=work->fst88_emd;
    real *alp=work->fst88_alp;
    real *dtc=work->fst88_dtc;
    real *css=work->fst88_css;
    real *fxo=work->fst88_fxo;
    real *flx=work->fst88_flx;
    real *cts=work->fst88_cts;
    real *tpl=work->fst88_tpl;

    integer *ixo=work->fst88_ixo;

    real *oss=work->fst88_oss;
    real *fac1=work->fst88_fac1;
    real *dte2=work->fst88_dte2;
    integer llm1, llm2, llp1;
    extern GD_DEVICE int e3v88_(real *, real *, real *, real *, real *,
	     real *, real *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, GDTable *gt, GDClock *clock, GDScratch *work), e1e290_(real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, GDTable *gt, GDClock *clock, GDScratch *work);

    real *to31d=work->fst88_to31d;
    real *fbot=work->fst88_fbot;
    real *csub=work->fst88_csub;
    integer kmax, kmin, icnt;

    real *rlog=work->fst88_rlog;
    real *sorc=work->fst88_sorc;
    real *pbot=work->fst88_pbot;
    real *ftop=work->fst88_ftop;
    integer myie;

    real *dtsp=work->fst88_dtsp;
    integer klen, myis;

    real *ptop=work->fst88_ptop;
    extern GD_DEVICE int spa88_(real *, real *, real *, real *, real *,
	     real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, GDTable *gt, GDClock *clock, GDScratch *work);

    real *csub2=work->fst88_csub2;
    real *e1flx=work->fst88_e1flx;
    real *to3dg=work->fst88_to3dg;
    real *fxoe2=work->fst88_fxoe2;
    real *co2sp=work->fst88_co2sp;
    real *avmo3=work->fst88_avmo3;
    real *ctso3=work->fst88_ctso3;
    real *avvo2=work->fst88_avvo2;
    real *vtmp3=work->fst88_vtmp3;
    real *to3sp=work->fst88_to3sp;
    real *vsum1=work->fst88_vsum1;
    real *e1cts1=work->fst88_e1cts1;
    real *e1ctw1=work->fst88_e1ctw1;
    real *dsorc=work->fst88_dsorc;
    real *emiss=work->fst88_emiss;
    real *e1cts2=work->fst88_e1cts2;
    real *e1ctw2=work->fst88_e1ctw2;
    real *flx1e1=work->fst88_flx1e1;
    integer kclds;

    real *excts=work->fst88_excts;
    real *gxcts=work->fst88_gxcts;
    real *csour=work->fst88_csour;
    real *fxosp=work->fst88_fxosp;
    extern GD_DEVICE int e2spec_(real *, real *, real *, real *, real *
	    , real *, real *, real *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, GDTable *gt, GDClock *clock, GDScratch *work);

    real *cont1d=work->fst88_cont1d;
    real *delpr1=work->fst88_delpr1;
    real *delpr2=work->fst88_delpr2;
    real *over1d=work->fst88_over1d;
    real *avpho3=work->fst88_avpho3;
    real *to3spc=work->fst88_to3spc;
    real *heatem=work->fst88_heatem;
    real *emisdg=work->fst88_emisdg;
    real *delptc=work->fst88_delptc;
    real *avephi=work->fst88_avephi;
    real *emspec=work->fst88_emspec;
    real *contdg=work->fst88_contdg;
    real *emissb=work->fst88_emissb;
    real *cnttau=work->fst88_cnttau;
    real *flxnet=work->fst88_flxnet;
    real *totevv=work->fst88_totevv;


    --betacm;
    --bcomb;
    --acomb;
    --btpcm;
    --atpcm;
    --bpcm;
    --apcm;
    --ao3rnd;
    --bo3rnd;
    empl -= 2;
    --emx2;
    --emx1;
    totvo2 -= 2;
    totphi -= 2;
    tphio3 -= 2;
    toto3 -= 2;
    cntval -= 2;
    var4 -= 2;
    var3 -= 2;
    var2 -= 2;
    var1 -= 2;
    co2sp2 -= 2;
    co2sp1 -= 2;
    co2nbl -= 2;
    co21 -= 84;
    camt -= 2;
    kbtm -= 2;
    ktop -= 2;
    --nclds;
    cldfac -= 84;
    t -= 2;
    temp -= 2;
    delp2 -= 2;
    delp -= 2;
    p -= 2;
    press -= 2;
    qh2o -= 2;
    --topflx;
    --grnflx;
    heatra -= 2;


    l = *kte;
    lp1 = l + 1;
    lp2 = l + 2;
    lp3 = l + 3;
    llp1 = (l << 1) + 1;
    lm1 = l - 1;
    lm2 = l - 2;
    lm3 = l - 3;
    ll = l << 1;
    llm2 = ll - 2;
    llm1 = ll - 1;
    myis = *its;
    myie = *ite;
    i__1 = lp1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    r__1 = temp[i__ + k] * *hp1;
	    vtmp3[i__ + k - 2] = (real)r_int(&r__1);
	    fxo[i__ + k - 2] = vtmp3[i__ + k - 2] - 9.f;
	    dt[i__ + k - 2] = temp[i__ + k] - *ten * vtmp3[i__ + k - 2];
	    ixo[i__ + k - 2] = fxo[i__ + k - 2];

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    r__1 = t[i__ + (k + 1)] * *hp1;
	    vtmp3[i__ + k - 2] = (real)r_int(&r__1);
	    fxoe2[i__ + k - 2] = vtmp3[i__ + k - 2] - 9.f;
	    dte2[i__ + k - 2] = t[i__ + (k + 1)] - *ten * vtmp3[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	fxoe2[i__ + lp1 - 2] = fxo[i__ + l - 2];
	dte2[i__ + lp1 - 2] = dt[i__ + l - 2];
	fxosp[i__ - 1] = fxoe2[i__ + lm1 - 2];
	fxosp[i__] = fxo[i__ + lm1 - 2];
	dtsp[i__ - 1] = dte2[i__ + lm1 - 2];
	dtsp[i__] = dt[i__ + lm1 - 2];

    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	i__2 = lp1;
	for (k = 1; k <= i__2; ++k) {
	    vtmp3[i__ + k - 2] = gd_tables__1.source[ixo[i__ + k - 2] - 1];
	    dsorc[i__ + k - 2] = gd_tables__1.dsrce[ixo[i__ + k - 2] - 1];

	}
    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    sorc[i__ + (k + 82) - 84] = vtmp3[i__ + k - 2] + dt[i__ + k - 2] *
		     dsorc[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	i__2 = lp1;
	for (k = 1; k <= i__2; ++k) {
	    vtmp3[i__ + k - 2] = gd_tables__1.source[ixo[i__ + k - 2] + 27];
	    dsorc[i__ + k - 2] = gd_tables__1.dsrce[ixo[i__ + k - 2] + 27];

	}
    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    sorc[i__ + (k + 164) - 84] = vtmp3[i__ + k - 2] + dt[i__ + k - 2]
		    * dsorc[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	i__2 = lp1;
	for (k = 1; k <= i__2; ++k) {
	    vtmp3[i__ + k - 2] = gd_tables__1.source[ixo[i__ + k - 2] + 55];
	    dsorc[i__ + k - 2] = gd_tables__1.dsrce[ixo[i__ + k - 2] + 55];

	}
    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    sorc[i__ + (k + 246) - 84] = vtmp3[i__ + k - 2] + dt[i__ + k - 2]
		    * dsorc[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	i__2 = lp1;
	for (k = 1; k <= i__2; ++k) {
	    vtmp3[i__ + k - 2] = gd_tables__1.source[ixo[i__ + k - 2] + 83];
	    dsorc[i__ + k - 2] = gd_tables__1.dsrce[ixo[i__ + k - 2] + 83];

	}
    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    sorc[i__ + (k + 328) - 84] = vtmp3[i__ + k - 2] + dt[i__ + k - 2]
		    * dsorc[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	i__2 = lp1;
	for (k = 1; k <= i__2; ++k) {
	    vtmp3[i__ + k - 2] = gd_tables__1.source[ixo[i__ + k - 2] + 111];
	    dsorc[i__ + k - 2] = gd_tables__1.dsrce[ixo[i__ + k - 2] + 111];

	}
    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    sorc[i__ + (k + 410) - 84] = vtmp3[i__ + k - 2] + dt[i__ + k - 2]
		    * dsorc[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	i__2 = lp1;
	for (k = 1; k <= i__2; ++k) {
	    vtmp3[i__ + k - 2] = gd_tables__1.source[ixo[i__ + k - 2] + 139];
	    dsorc[i__ + k - 2] = gd_tables__1.dsrce[ixo[i__ + k - 2] + 139];

	}
    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    sorc[i__ + (k + 492) - 84] = vtmp3[i__ + k - 2] + dt[i__ + k - 2]
		    * dsorc[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	i__2 = lp1;
	for (k = 1; k <= i__2; ++k) {
	    vtmp3[i__ + k - 2] = gd_tables__1.source[ixo[i__ + k - 2] + 167];
	    dsorc[i__ + k - 2] = gd_tables__1.dsrce[ixo[i__ + k - 2] + 167];

	}
    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    sorc[i__ + (k + 574) - 84] = vtmp3[i__ + k - 2] + dt[i__ + k - 2]
		    * dsorc[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	i__2 = lp1;
	for (k = 1; k <= i__2; ++k) {
	    vtmp3[i__ + k - 2] = gd_tables__1.source[ixo[i__ + k - 2] + 195];
	    dsorc[i__ + k - 2] = gd_tables__1.dsrce[ixo[i__ + k - 2] + 195];

	}
    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    sorc[i__ + (k + 656) - 84] = vtmp3[i__ + k - 2] + dt[i__ + k - 2]
		    * dsorc[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	i__2 = lp1;
	for (k = 1; k <= i__2; ++k) {
	    vtmp3[i__ + k - 2] = gd_tables__1.source[ixo[i__ + k - 2] + 223];
	    dsorc[i__ + k - 2] = gd_tables__1.dsrce[ixo[i__ + k - 2] + 223];

	}
    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    sorc[i__ + (k + 738) - 84] = vtmp3[i__ + k - 2] + dt[i__ + k - 2]
		    * dsorc[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	i__2 = lp1;
	for (k = 1; k <= i__2; ++k) {
	    vtmp3[i__ + k - 2] = gd_tables__1.source[ixo[i__ + k - 2] + 251];
	    dsorc[i__ + k - 2] = gd_tables__1.dsrce[ixo[i__ + k - 2] + 251];

	}
    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    sorc[i__ + (k + 820) - 84] = vtmp3[i__ + k - 2] + dt[i__ + k - 2]
		    * dsorc[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	i__2 = lp1;
	for (k = 1; k <= i__2; ++k) {
	    vtmp3[i__ + k - 2] = gd_tables__1.source[ixo[i__ + k - 2] + 279];
	    dsorc[i__ + k - 2] = gd_tables__1.dsrce[ixo[i__ + k - 2] + 279];

	}
    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    sorc[i__ + (k + 902) - 84] = vtmp3[i__ + k - 2] + dt[i__ + k - 2]
		    * dsorc[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	i__2 = lp1;
	for (k = 1; k <= i__2; ++k) {
	    vtmp3[i__ + k - 2] = gd_tables__1.source[ixo[i__ + k - 2] + 307];
	    dsorc[i__ + k - 2] = gd_tables__1.dsrce[ixo[i__ + k - 2] + 307];

	}
    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    sorc[i__ + (k + 984) - 84] = vtmp3[i__ + k - 2] + dt[i__ + k - 2]
		    * dsorc[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	i__2 = lp1;
	for (k = 1; k <= i__2; ++k) {
	    vtmp3[i__ + k - 2] = gd_tables__1.source[ixo[i__ + k - 2] + 335];
	    dsorc[i__ + k - 2] = gd_tables__1.dsrce[ixo[i__ + k - 2] + 335];

	}
    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    sorc[i__ + (k + 1066) - 84] = vtmp3[i__ + k - 2] + dt[i__ + k - 2]
		     * dsorc[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	i__2 = lp1;
	for (k = 1; k <= i__2; ++k) {
	    vtmp3[i__ + k - 2] = gd_tables__1.source[ixo[i__ + k - 2] + 363];
	    dsorc[i__ + k - 2] = gd_tables__1.dsrce[ixo[i__ + k - 2] + 363];

	}
    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    sorc[i__ + (k + 1148) - 84] = vtmp3[i__ + k - 2] + dt[i__ + k - 2]
		     * dsorc[i__ + k - 2];

	}
    }
    i__1 = lp1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    r__1 = sorc[i__ + (k + 902) - 84] + sorc[i__ + (k + 984) - 84];
	    ss1[i__ + k - 2] = r__1 + sorc[i__ + (k + 1148) - 84];

	}
    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    csour[i__ + k - 2] = sorc[i__ + (k + 738) - 84] + sorc[i__ + (k +
		    820) - 84];

	}
    }
    i__1 = lp1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    r__2 = temp[i__ + k] * temp[i__ + k];
	    r__1 = r__2 * temp[i__ + k];
	    tc[i__ + k - 2] = r__1 * temp[i__ + k];

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    oss[i__ + (k + 1) - 2] = sorc[i__ + (k + 1067) - 84] - sorc[i__ +
		    (k + 1066) - 84];
	    css[i__ + (k + 1) - 2] = csour[i__ + (k + 1) - 2] - csour[i__ + k
		    - 2];
	    dtc[i__ + (k + 1) - 2] = tc[i__ + (k + 1) - 2] - tc[i__ + k - 2];
	    ss2[i__ + (k + 1) - 2] = ss1[i__ + (k + 1) - 2] - ss1[i__ + k - 2]
		    ;

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    avephi[i__ + k - 2] = totphi[i__ + (k + 1)];

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	avephi[i__ + lp1 - 2] = avephi[i__ + lm1 - 2] + emx1[i__];

    }
    e1e290_(e1cts1, e1cts2, e1flx, e1ctw1, e1ctw2, emiss, fxo, dt, fxoe2,
	    dte2, avephi, &temp[2], &t[2], h16e1, ten, hp1, h28e1, haf, ids,
	    ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite,
	    jts, jte, kts, kte, gt, clock, work);
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    fac1[i__ + k - 2] = bo3rnd[2] * tphio3[i__ + (k + 1)] / toto3[i__
		    + (k + 1)];
	    r__2 = *four * ao3rnd[2];
	    r__1 = fac1[i__ + k - 2] * ((real)gd_sqrt((real)(*one + r__2 * toto3[
		    i__ + (k + 1)] / fac1[i__ + k - 2])) - *one);
	    to3spc[i__ + k - 2] = *haf * r__1;
	    to3sp[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * (to3spc[i__ + k -
		    2] + gd_tables__1.sko3r * totvo2[i__ + (k + 1)])));
	    over1d[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * ((real)gd_sqrt((real)
		    (gd_tables__1.ab15wd * totphi[i__ + (k + 1)])) +
		    gd_tables__1.skc1r * totvo2[i__ + (k + 1)])));
	    cnttau[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * totvo2[i__ + (k +
		    1)]));
	    totevv[i__ + k - 2] = 1.f / cnttau[i__ + k - 2];

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    co2sp[i__ + (k + 1) - 2] = over1d[i__ + k - 2] * co21[i__ + ((k +
		    1) * 82 + 1)];

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    co21[i__ + (k + 83)] *= over1d[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	rlog[i__ - 1] = over1d[i__ - 1] * co2nbl[i__ + 1];

    }
    i__1 = lp1;
    for (k = 2; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    r__2 = tc[i__ - 1] * e1flx[i__ + k - 2] + ss1[i__ - 1] * cnttau[
		    i__ + (k - 1) - 2];
	    r__1 = r__2 + sorc[i__ + 983] * to3sp[i__ + (k - 1) - 2];
	    flx[i__ + k - 2] = (r__1 + csour[i__ - 1] * co2sp[i__ + k - 2]) *
		    cldfac[i__ + (k * 82 + 1)];

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	r__2 = tc[i__ - 1] * e1flx[i__ - 1] + ss1[i__ - 1];
	r__1 = r__2 + sorc[i__ + 983];
	flx[i__ - 1] = r__1 + csour[i__ - 1];

    }
    i__2 = lp1;
    for (kp = 2; kp <= i__2; ++kp) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    r__2 = oss[i__ + kp - 2] * to3sp[i__ + (kp - 1) - 2] + ss2[i__ +
		    kp - 2] * cnttau[i__ + (kp - 1) - 2];
	    r__1 = r__2 + css[i__ + kp - 2] * co21[i__ + (kp + 82)];
	    flx[i__ - 1] += (r__1 + dtc[i__ + kp - 2] * emiss[i__ + (kp - 1)
		    - 2]) * cldfac[i__ + (kp + 82)];

	}
    }
    spa88_(excts, ctso3, gxcts, sorc, csour, &cldfac[84], &temp[2], &press[2],
	     &var1[2], &var2[2], &p[2], &delp[2], &delp2[2], &totvo2[2],
	    to3sp, to3spc, &co2sp1[2], &co2sp2[2], co2sp, &apcm[1], &bpcm[1],
	    &atpcm[1], &btpcm[1], &acomb[1], &bcomb[1], &betacm[1], h25e2,
	    one, h44194m2, h1p41819, haf, hm1ez, two, radcon, ids, ide, jds,
	    jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte,
	    kts, kte, gt, clock, work);
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	vtmp3[i__ - 1] = 1.f;

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    vtmp3[i__ + (k + 1) - 2] = cnttau[i__ + k - 2] * cldfac[i__ + (k
		    + 83)];

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    r__1 = *radcon * delp[i__ + k];
	    cts[i__ + k - 2] = r__1 * (tc[i__ + k - 2] * (e1ctw2[i__ + k - 2]
		    * cldfac[i__ + (k + 83)] - e1ctw1[i__ + k - 2] * cldfac[
		    i__ + (k + 82)]) + ss1[i__ + k - 2] * (vtmp3[i__ + (k + 1)
		     - 2] - vtmp3[i__ + k - 2]));

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    vtmp3[i__ + k - 2] = tc[i__ + k - 2] * (cldfac[i__ + (k + 82)] * (
		    e1cts1[i__ + k - 2] - e1ctw1[i__ + k - 2]) - cldfac[i__ +
		    (k + 83)] * (e1cts2[i__ + k - 2] - e1ctw2[i__ + k - 2]));

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	r__1 = tc[i__ + lp1 - 2] * cldfac[i__ + (lp1 + 82)];
	flx1e1[i__ - 1] = r__1 * (e1cts1[i__ + lp1 - 2] - e1ctw1[i__ + lp1 -
		2]);

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    flx1e1[i__ - 1] += vtmp3[i__ + k - 2];

	}

    }
    i__2 = lm1;
    for (k = 2; k <= i__2; ++k) {
	klen = k;
	i__1 = lp1 - k;
	for (kk = 1; kk <= i__1; ++kk) {
	    i__3 = myie;
	    for (i__ = myis; i__ <= i__3; ++i__) {
		avephi[i__ + (kk + k - 1) - 2] = totphi[i__ + (kk + k)] -
			totphi[i__ + k];

	    }
	}
	i__3 = myie;
	for (i__ = myis; i__ <= i__3; ++i__) {
	    avephi[i__ + lp1 - 2] = avephi[i__ + lm1 - 2] + emx1[i__];

	}
	e290_(emissb, emiss, avephi, &klen, fxoe2, dte2, h16e1, hp1, h28e1,
		haf, ten, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme,
		kms, kme, its, ite, jts, jte, kts, kte, gt, clock, work);
	i__3 = lp1 - k;
	for (kk = 1; kk <= i__3; ++kk) {
	    i__1 = myie;
	    for (i__ = myis; i__ <= i__1; ++i__) {
		avmo3[i__ + (kk + k - 1) - 2] = toto3[i__ + (kk + k)] - toto3[
			i__ + k];
		avpho3[i__ + (kk + k - 1) - 2] = tphio3[i__ + (kk + k)] -
			tphio3[i__ + k];
		avvo2[i__ + (kk + k - 1) - 2] = totvo2[i__ + (kk + k)] -
			totvo2[i__ + k];
		cont1d[i__ + (kk + k - 1) - 2] = cnttau[i__ + (kk + k - 1) -
			2] * totevv[i__ + (k - 1) - 2];

	    }
	}
	i__1 = lp1 - k;
	for (kk = 1; kk <= i__1; ++kk) {
	    i__3 = myie;
	    for (i__ = myis; i__ <= i__3; ++i__) {
		fac1[i__ + (k + kk - 1) - 2] = bo3rnd[2] * avpho3[i__ + (k +
			kk - 1) - 2] / avmo3[i__ + (k + kk - 1) - 2];
		r__2 = *four * ao3rnd[2];
		r__1 = fac1[i__ + (k + kk - 1) - 2] * ((real)gd_sqrt((real)(*one
			+ r__2 * avmo3[i__ + (k + kk - 1) - 2] / fac1[i__ + (
			k + kk - 1) - 2])) - *one);
		vtmp3[i__ + (k + kk - 1) - 2] = *haf * r__1;
		to31d[i__ + (k + kk - 1) - 2] = (real)gd_exp((real)(*hm1ez * (
			vtmp3[i__ + (k + kk - 1) - 2] + gd_tables__1.sko3r *
			avvo2[i__ + (k + kk - 1) - 2])));
		over1d[i__ + (k + kk - 1) - 2] = (real)gd_exp((real)(*hm1ez * ((
			real)gd_sqrt((real)(gd_tables__1.ab15wd * avephi[i__ + (
			k + kk - 1) - 2])) + gd_tables__1.skc1r * avvo2[i__ +
			(k + kk - 1) - 2])));
		co21[i__ + (k + kk + k * 82)] = over1d[i__ + (k + kk - 1) - 2]
			 * co21[i__ + (k + kk + k * 82)];

	    }
	}
	i__3 = lp1;
	for (kp = k + 1; kp <= i__3; ++kp) {
	    i__1 = myie;
	    for (i__ = myis; i__ <= i__1; ++i__) {
		co21[i__ + (k + kp * 82)] = over1d[i__ + (kp - 1) - 2] * co21[
			i__ + (k + kp * 82)];

	    }
	}
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    rlog[i__ + k - 2] = over1d[i__ + k - 2] * co2nbl[i__ + k];

	}
	i__1 = lp1;
	for (kp = k + 1; kp <= i__1; ++kp) {
	    i__3 = myie;
	    for (i__ = myis; i__ <= i__3; ++i__) {
		r__2 = oss[i__ + kp - 2] * to31d[i__ + (kp - 1) - 2] + ss2[
			i__ + kp - 2] * cont1d[i__ + (kp - 1) - 2];
		r__1 = r__2 + css[i__ + kp - 2] * co21[i__ + (kp + k * 82)];
		flx[i__ + k - 2] += (r__1 + dtc[i__ + kp - 2] * emiss[i__ + (
			kp - 1) - 2]) * cldfac[i__ + (kp + k * 82)];

	    }
	}
	i__3 = lp1;
	for (kp = k + 1; kp <= i__3; ++kp) {
	    i__1 = myie;
	    for (i__ = myis; i__ <= i__1; ++i__) {
		r__2 = oss[i__ + k - 2] * to31d[i__ + (kp - 1) - 2] + ss2[i__
			+ k - 2] * cont1d[i__ + (kp - 1) - 2];
		r__1 = r__2 + css[i__ + k - 2] * co21[i__ + (k + kp * 82)];
		flx[i__ + kp - 2] += (r__1 + dtc[i__ + k - 2] * emissb[i__ + (
			kp - 1) - 2]) * cldfac[i__ + (k + kp * 82)];

	    }
	}

    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	tpl[i__ - 1] = temp[i__ + l];
	tpl[i__ + lp1 - 2] = *haf * (t[i__ + lp1] + temp[i__ + l]);
	tpl[i__ + llp1 - 2] = *haf * (t[i__ + l] + temp[i__ + l]);

    }
    i__2 = l;
    for (k = 2; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    tpl[i__ + k - 2] = t[i__ + k];
	    tpl[i__ + (k + l) - 2] = t[i__ + k];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	avephi[i__ - 1] = var2[i__ + l];
	avephi[i__] = var2[i__ + l] + empl[i__ + l];

    }
    e2spec_(emiss, avephi, fxosp, dtsp, h16e1, ten, h28e1, hp1, ids, ide, jds,
	     jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte,
	    kts, kte, gt, clock, work);
    e3v88_(emd, tpl, &empl[2], ten, hp1, h28e1, h16e1, ids, ide, jds, jde,
	    kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts,
	    kte, gt, clock, work);
    i__1 = l;
    for (k = 2; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    emisdg[i__ + k - 2] = emd[i__ + (k + l) - 2] + emd[i__ + k - 2];

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	emspec[i__ - 1] = (emd[i__ - 1] * empl[i__ + 1] - emd[i__ + lp1 - 2] *
		 empl[i__ + lp1]) / emx1[i__] + *quartr * (emiss[i__ - 1] +
		emiss[i__]);
	emisdg[i__ + lp1 - 2] = *two * emd[i__ + lp1 - 2];
	emspec[i__] = *two * (emd[i__ - 1] * empl[i__ + 1] - emd[i__ + llp1 -
		2] * empl[i__ + llp1]) / emx2[i__];

    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	fac1[i__ + l - 2] = bo3rnd[2] * var4[i__ + l] / var3[i__ + l];
	r__2 = *four * ao3rnd[2];
	r__1 = fac1[i__ + l - 2] * ((real)gd_sqrt((real)(*one + r__2 * var3[i__
		+ l] / fac1[i__ + l - 2])) - *one);
	vtmp3[i__ + l - 2] = *haf * r__1;
	to31d[i__ + l - 2] = (real)gd_exp((real)(*hm1ez * (vtmp3[i__ + l - 2] +
		gd_tables__1.sko3r * cntval[i__ + l])));
	over1d[i__ + l - 2] = (real)gd_exp((real)(*hm1ez * ((real)gd_sqrt((real)(
		gd_tables__1.ab15wd * var2[i__ + l])) + gd_tables__1.skc1r *
		cntval[i__ + l])));
	cont1d[i__ + l - 2] = cnttau[i__ + l - 2] * totevv[i__ + lm1 - 2];
	rlog[i__ + l - 2] = over1d[i__ + l - 2] * co2nbl[i__ + l];

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    rlog[i__ + k - 2] = (real)gd_log(rlog[i__ + k - 2]);

	}
    }
    i__1 = lm1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    delpr1[i__ + (k + 1) - 2] = delp[i__ + (k + 1)] * (press[i__ + (k
		    + 1)] - p[i__ + (k + 1)]);
	    alp[i__ + (lp1 + k - 1) - 2] = -(real)gd_sqrt(delpr1[i__ + (k + 1) -
		    2]) * rlog[i__ + (k + 1) - 2];

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    delpr2[i__ + (k + 1) - 2] = delp[i__ + k] * (p[i__ + (k + 1)] -
		    press[i__ + k]);
	    alp[i__ + k - 2] = -(real)gd_sqrt(delpr2[i__ + (k + 1) - 2]) * rlog[
		    i__ + k - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	alp[i__ + ll - 2] = -rlog[i__ + l - 2];
	alp[i__ + llp1 - 2] = -rlog[i__ + l - 2] * (real)gd_sqrt((real)(delp[i__
		+ l] * (p[i__ + lp1] - press[i__ + lm1])));

    }
    i__1 = llp1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    c__[i__ + k - 2] = alp[i__ + k - 2] * (*hmp66667 + alp[i__ + k -
		    2] * (*quartr + alp[i__ + k - 2] * *hm6666m2));

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	co21[i__ + (lp1 + lp1 * 82)] = *one + c__[i__ + l - 2];
	co21[i__ + (lp1 + l * 82)] = *one + (delp2[i__ + l] * c__[i__ + ll -
		2] - (press[i__ + l] - p[i__ + l]) * c__[i__ + llm1 - 2]) / (
		p[i__ + lp1] - press[i__ + l]);
	co21[i__ + (l + lp1 * 82)] = *one + ((p[i__ + lp1] - press[i__ + lm1])
		 * c__[i__ + llp1 - 2] - (p[i__ + lp1] - press[i__ + l]) *
		c__[i__ + l - 2]) / (press[i__ + l] - press[i__ + lm1]);

    }
    i__2 = l;
    for (k = 2; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    co21[i__ + (k + k * 82)] = *one + *haf * (c__[i__ + (lm1 + k) - 2]
		     + c__[i__ + (k - 1) - 2]);

	}
    }
    i__1 = lm1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    csub[i__ + (k + 1) - 2] = cntval[i__ + (k + 1)] * delpr1[i__ + (k
		    + 1) - 2];
	    csub[i__ + (lp1 + k - 1) - 2] = cntval[i__ + k] * delpr2[i__ + (k
		    + 1) - 2];

	}
    }
    i__2 = llm2;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    csub2[i__ + (k + 1) - 2] = gd_tables__1.sko3r * csub[i__ + (k + 1)
		     - 2];
	    c__[i__ + (k + 1) - 2] = csub[i__ + (k + 1) - 2] * (*hmp5 + csub[
		    i__ + (k + 1) - 2] * (*hp166666 - csub[i__ + (k + 1) - 2]
		    * *h41666m2));
	    c2[i__ + (k + 1) - 2] = csub2[i__ + (k + 1) - 2] * (*hmp5 + csub2[
		    i__ + (k + 1) - 2] * (*hp166666 - csub2[i__ + (k + 1) - 2]
		     * *h41666m2));

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	contdg[i__ + lp1 - 2] = c__[i__ + llm1 - 2] + 1.f;
	to3dg[i__ + lp1 - 2] = c2[i__ + llm1 - 2] + 1.f;

    }
    i__1 = l;
    for (k = 2; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    contdg[i__ + k - 2] = *one + *haf * (c__[i__ + k - 2] + c__[i__ +
		    (lm1 + k) - 2]);
	    to3dg[i__ + k - 2] = *one + *haf * (c2[i__ + k - 2] + c2[i__ + (
		    lm1 + k) - 2]);

	}
    }
    i__2 = lp1;
    for (k = 2; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    r__2 = dtc[i__ + k - 2] * emisdg[i__ + k - 2] + ss2[i__ + k - 2] *
		     contdg[i__ + k - 2];
	    r__1 = r__2 + oss[i__ + k - 2] * to3dg[i__ + k - 2];
	    flx[i__ + k - 2] += (r__1 + css[i__ + k - 2] * co21[i__ + (k + k *
		     82)]) * cldfac[i__ + (k + k * 82)];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	r__2 = css[i__ + lp1 - 2] * co21[i__ + (lp1 + l * 82)] + dtc[i__ +
		lp1 - 2] * emspec[i__];
	r__1 = r__2 + oss[i__ + lp1 - 2] * to31d[i__ + l - 2];
	flx[i__ + l - 2] += (r__1 + ss2[i__ + lp1 - 2] * cont1d[i__ + l - 2])
		* cldfac[i__ + (lp1 + l * 82)];
	r__2 = css[i__ + l - 2] * co21[i__ + (l + lp1 * 82)] + oss[i__ + l -
		2] * to31d[i__ + l - 2];
	r__1 = r__2 + ss2[i__ + l - 2] * cont1d[i__ + l - 2];
	flx[i__ + lp1 - 2] += (r__1 + dtc[i__ + l - 2] * emspec[i__ - 1]) *
		cldfac[i__ + (l + lp1 * 82)];

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    r__1 = *radcon * (flx[i__ + (k + 1) - 2] - flx[i__ + k - 2]);
	    heatem[i__ + k - 2] = r__1 * delp[i__ + k];

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    heatra[i__ + k] = heatem[i__ + k - 2] - cts[i__ + k - 2] - ctso3[
		    i__ + k - 2] + excts[i__ + k - 2];

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    r__1 = heatra[i__ + k] * delp2[i__ + k];
	    vsum1[i__ + k - 2] = r__1 * *radcon1;

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	topflx[i__] = flx1e1[i__ - 1] + gxcts[i__ - 1];
	flxnet[i__ - 1] = topflx[i__];

    }
    i__2 = lp1;
    for (k = 2; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    flxnet[i__ + k - 2] = flxnet[i__ + (k - 1) - 2] + vsum1[i__ + (k
		    - 1) - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	grnflx[i__] = flxnet[i__ + lp1 - 2];

    }
    icnt = 0;
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	icnt += nclds[i__];

    }
    if (icnt == 0) {
	goto L6999;
    }
    kclds = nclds[myis];
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {

	i__2 = nclds[i__];
	kclds = max(i__2,kclds);

    }
    i__1 = kclds;
    for (kk = 1; kk <= i__1; ++kk) {
	kmin = lp1;
	kmax = 0;
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    j1 = ktop[i__ + (kk + 1)];
	    j3 = kbtm[i__ + (kk + 1)];
	    if (j3 > j1) {
		ptop[i__ - 1] = p[i__ + j1];
		pbot[i__ - 1] = p[i__ + (j3 + 1)];
		ftop[i__ - 1] = flxnet[i__ + j1 - 2];
		fbot[i__ - 1] = flxnet[i__ + (j3 + 1) - 2];
		delptc[i__ - 1] = (ftop[i__ - 1] - fbot[i__ - 1]) / (ptop[i__
			- 1] - pbot[i__ - 1]);
		kmin = min(kmin,j1);
		kmax = max(kmax,j3);
	    }

	}
	++kmin;
	i__2 = kmax;
	for (k = kmin; k <= i__2; ++k) {
	    i__3 = myie;
	    for (i__ = myis; i__ <= i__3; ++i__) {
		if (ktop[i__ + (kk + 1)] < k && k <= kbtm[i__ + (kk + 1)]) {
		    z1[i__ + k - 2] = (p[i__ + k] - ptop[i__ - 1]) * delptc[
			    i__ - 1] + ftop[i__ - 1];
		    flxnet[i__ + k - 2] = z1[i__ + k - 2];
		}

	    }

	}

    }

L6999:
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    r__1 = *radcon * (flxnet[i__ + (k + 1) - 2] - flxnet[i__ + k - 2])
		    ;
	    heatra[i__ + k] = r__1 * delp[i__ + k];

	}
    }
    return 0;
}
















GD_DEVICE real gausin_(real *xsd, GDTable *gt, GDClock *clock, GDScratch *work)
{

    real ret_val, r__1;





    real a1, a2, b1, b2, b3, rn, sum;

    a1 = *xsd * gd_tables__1.rsqr;

    r__1 = *xsd;
    a2 = (real)gd_exp((real)(r__1 * r__1 * -.5f));
    rn = 1.f;
    b1 = 1.f;
    b2 = 1.f;
    b3 = 1.f;
    sum = 1.f;
    while(b2 > .0010000000474974513f) {
	rn += 1.f;

	r__1 = *xsd;
	b2 = r__1 * r__1 / (rn * 2.f - 1.f);
	b3 = b1 * b2;
	sum += b3;
	b1 = b3;
    }
    r__1 = a1 * a2;
    ret_val = r__1 * sum;
    return ret_val;
}
















 GD_DEVICE int lwr88_(real *heatra, real *grnflx, real *topflx, real *
	press, real *temp, real *rh2o, real *qo3, real *cldfac, real *camt,
	integer *nclds, integer *ktop, integer *kbtm, real *bo3rnd, real *
	ao3rnd, real *apcm, real *bpcm, real *atpcm, real *btpcm, real *acomb,
	 real *bcomb, real *betacm, real *zero, real *one, real *h18e3, real *
	p0inv, real *h6p08108, real *diffctr, real *ginv, real *h3m4, real *
	betinw, real *rath2omw, real *gp0inv, real *p0, real *p0xzp8, real *
	p0xzp2, real *h3m3, real *h1m3, real *h1m2, real *h25e2, real *b0,
	real *b2, real *b1, real *b3, real *haf, real *ten, real *hp1, real *
	four, real *hm1ez, real *radcon, real *quartr, real *two, real *
	hm6666m2, real *hmp66667, real *hmp5, real *hp166666, real *h41666m2,
	real *radcon1, real *h16e1, real *h28e1, real *h44194m2, real *
	h1p41819, integer *ids, integer *ide, integer *jds, integer *jde,
	integer *kds, integer *kde, integer *ims, integer *ime, integer *jms,
	integer *jme, integer *kms, integer *kme, integer *its, integer *ite,
	integer *jts, integer *jte, integer *kts, integer *kte, GDTable *gt, GDClock *clock, GDScratch *work)
{

    integer i__1, i__2, i__3;
    real r__1, r__2;





    integer i__, k, l;

    real *p=work->lwr88_p;
    real *t=work->lwr88_t;
    real *a1=work->lwr88_a1;
    real a2[
	    1];
    integer ll, kp;

    real *vv=work->lwr88_vv;
    integer lm1, lm2, lm3, lp1, lp2, lp3;

    real *co21=work->lwr88_co21;
    real *co2r=work->lwr88_co2r;
    real *qh2o=work->lwr88_qh2o;
    real *var1=work->lwr88_var1;
    real *var2=work->lwr88_var2;
    real *var3=work->lwr88_var3;
    real *var4=work->lwr88_var4;
    real *emx1=work->lwr88_emx1;
    real *emx2=work->lwr88_emx2;
    integer llp1;

    real *co2r1=work->lwr88_co2r1;
    real *co2r2=work->lwr88_co2r2;
    real *dift=work->lwr88_dift;
    real *delp=work->lwr88_delp;
    real *tdav=work->lwr88_tdav;
    real *empl=work->lwr88_empl;
    integer myie;
    extern GD_DEVICE int fst88_(real *, real *, real *, real *, real *,
	     real *, real *, real *, real *, real *, real *, integer *,
	    integer *, integer *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, GDTable *gt, GDClock *clock, GDScratch *work);

    real *d2cd21=work->lwr88_d2cd21;
    real *d2cd22=work->lwr88_d2cd22;
    integer myis;

    real *co2md=work->lwr88_co2md;
    real *delp2=work->lwr88_delp2;
    real *co2mr=work->lwr88_co2mr;
    real *toto3=work->lwr88_toto3;
    real *vsum1=work->lwr88_vsum1;
    real *dco2d1=work->lwr88_dco2d1;
    real *vsum3=work->lwr88_vsum3;
    real *dco2d2=work->lwr88_dco2d2;
    real *vsum2=work->lwr88_vsum2;
    real d2cdt2[
	    82];
    real *co2m2d=work->lwr88_co2m2d;
    real *co2sp1=work->lwr88_co2sp1;
    real *co2sp2=work->lwr88_co2sp2;
    real *tlsqu=work->lwr88_tlsqu;
    real *dco2dt=work->lwr88_dco2dt;
    real *co2nbl=work->lwr88_co2nbl;
    real *tphio3=work->lwr88_tphio3;
    real *totvo2=work->lwr88_totvo2;
    real *cntval=work->lwr88_cntval;
    real *tstdav=work->lwr88_tstdav;
    real *totphi=work->lwr88_totphi;
    real *texpsl=work->lwr88_texpsl;


    --betacm;
    --bcomb;
    --acomb;
    --btpcm;
    --atpcm;
    --bpcm;
    --apcm;
    --ao3rnd;
    --bo3rnd;
    kbtm -= 2;
    ktop -= 2;
    --nclds;
    camt -= 2;
    cldfac -= 84;
    qo3 -= 2;
    rh2o -= 2;
    temp -= 2;
    press -= 2;
    --topflx;
    --grnflx;
    heatra -= 2;


    l = *kte;
    lp1 = l + 1;
    lp2 = l + 2;
    lp3 = l + 3;
    llp1 = (l << 1) + 1;
    lm1 = l - 1;
    lm2 = l - 2;
    lm3 = l - 3;
    ll = l << 1;
    myis = *its;
    myie = *ite;
    i__1 = l;
    for (k = 2; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    p[i__ + k - 2] = *haf * (press[i__ + (k - 1)] + press[i__ + k]);
	    t[i__ + k - 2] = *haf * (temp[i__ + (k - 1)] + temp[i__ + k]);

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	p[i__ - 1] = *zero;
	p[i__ + lp1 - 2] = press[i__ + lp1];
	t[i__ - 1] = temp[i__ + 1];
	t[i__ + lp1 - 2] = temp[i__ + lp1];

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    delp2[i__ + k - 2] = p[i__ + (k + 1) - 2] - p[i__ + k - 2];
	    delp[i__ + k - 2] = *one / delp2[i__ + k - 2];

	}
    }
    i__1 = lp1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    texpsl[i__ + k - 2] = *h18e3 / temp[i__ + k] - *h6p08108;
	    texpsl[i__ + k - 2] = (real)gd_exp(texpsl[i__ + k - 2]);

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    qh2o[i__ + k - 2] = rh2o[i__ + k] * *diffctr;
	    r__1 = *haf * (p[i__ + (k + 1) - 2] + p[i__ + k - 2]);
	    vv[i__ + k - 2] = r__1 * *p0inv;
	    r__1 = delp2[i__ + k - 2] * qh2o[i__ + k - 2];
	    var1[i__ + k - 2] = r__1 * *ginv;
	    r__2 = delp2[i__ + k - 2] * qo3[i__ + k];
	    r__1 = r__2 * *diffctr;
	    var3[i__ + k - 2] = r__1 * *ginv;
	    var2[i__ + k - 2] = var1[i__ + k - 2] * (vv[i__ + k - 2] + *h3m4);
	    var4[i__ + k - 2] = var3[i__ + k - 2] * (vv[i__ + k - 2] + *h3m3);
	    r__2 = texpsl[i__ + k - 2] * rh2o[i__ + k];
	    r__1 = r__2 * var2[i__ + k - 2];
	    cntval[i__ + k - 2] = r__1 * *betinw / (rh2o[i__ + k] + *rath2omw)
		    ;

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	totphi[i__ - 1] = *zero;
	toto3[i__ - 1] = *zero;
	tphio3[i__ - 1] = *zero;
	totvo2[i__ - 1] = *zero;

    }
    i__1 = lp1;
    for (k = 2; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    totphi[i__ + k - 2] = totphi[i__ + (k - 1) - 2] + var2[i__ + (k -
		    1) - 2];
	    toto3[i__ + k - 2] = toto3[i__ + (k - 1) - 2] + var3[i__ + (k - 1)
		     - 2];
	    tphio3[i__ + k - 2] = tphio3[i__ + (k - 1) - 2] + var4[i__ + (k -
		    1) - 2];
	    totvo2[i__ + k - 2] = totvo2[i__ + (k - 1) - 2] + cntval[i__ + (k
		    - 1) - 2];

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	r__2 = qh2o[i__ + l - 2] * press[i__ + l];
	r__1 = r__2 * (press[i__ + l] - p[i__ + l - 2]);
	emx1[i__ - 1] = r__1 * *gp0inv;
	r__2 = qh2o[i__ + l - 2] * press[i__ + l];
	r__1 = r__2 * (p[i__ + lp1 - 2] - press[i__ + l]);
	emx2[i__ - 1] = r__1 * *gp0inv;

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    r__2 = qh2o[i__ + k - 2] * p[i__ + (k + 1) - 2];
	    r__1 = r__2 * (p[i__ + (k + 1) - 2] - press[i__ + k]);
	    empl[i__ + (k + 1) - 2] = r__1 * *gp0inv;

	}
    }
    i__1 = lm1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    r__2 = qh2o[i__ + (k + 1) - 2] * p[i__ + (k + 1) - 2];
	    r__1 = r__2 * (press[i__ + (k + 1)] - p[i__ + (k + 1) - 2]);
	    empl[i__ + (lp2 + k - 1) - 2] = r__1 * *gp0inv;

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	empl[i__ - 1] = var2[i__ + l - 2];
	empl[i__ + llp1 - 2] = empl[i__ + ll - 2];

    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	tstdav[i__ - 1] = *zero;
	tdav[i__ - 1] = *zero;

    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    vsum3[i__ + k - 2] = temp[i__ + k] - gd_tables__1.stemp[k - 1];

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    vsum2[i__ - 1] = gd_tables__1.gtemp[k - 1] * delp2[i__ + k - 2];
	    vsum1[i__ - 1] = vsum2[i__ - 1] * vsum3[i__ + k - 2];
	    tstdav[i__ + (k + 1) - 2] = tstdav[i__ + k - 2] + vsum2[i__ - 1];
	    tdav[i__ + (k + 1) - 2] = tdav[i__ + k - 2] + vsum1[i__ - 1];

	}

    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	a1[i__ - 1] = (press[i__ + lp1] - *p0xzp8) / *p0xzp2;
	a2[i__ - 1] = (*p0 - press[i__ + lp1]) / *p0xzp2;

    }
    i__1 = lp1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    co2r1[i__ + k - 2] = a1[i__ - 1] * gd_tables__1.co231[k - 1] + a2[
		    i__ - 1] * gd_tables__1.co238[k - 1];
	    d2cd21[i__ + k - 2] = *h1m3 * (a1[i__ - 1] * gd_tables__1.c2d31[k
		    - 1] + a2[i__ - 1] * gd_tables__1.c2d38[k - 1]);
	    dco2d1[i__ + k - 2] = *h1m2 * (a1[i__ - 1] * gd_tables__1.cdt31[k
		    - 1] + a2[i__ - 1] * gd_tables__1.cdt38[k - 1]);
	    co2r2[i__ + k - 2] = a1[i__ - 1] * gd_tables__1.co271[k - 1] + a2[
		    i__ - 1] * gd_tables__1.co278[k - 1];
	    d2cd22[i__ + k - 2] = *h1m3 * (a1[i__ - 1] * gd_tables__1.c2d71[k
		    - 1] + a2[i__ - 1] * gd_tables__1.c2d78[k - 1]);
	    dco2d2[i__ + k - 2] = *h1m2 * (a1[i__ - 1] * gd_tables__1.cdt71[k
		    - 1] + a2[i__ - 1] * gd_tables__1.cdt78[k - 1]);

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    co2mr[i__ + k - 2] = a1[i__ - 1] * gd_tables__1.co2m51[k - 1] +
		    a2[i__ - 1] * gd_tables__1.co2m58[k - 1];
	    co2md[i__ + k - 2] = *h1m2 * (a1[i__ - 1] * gd_tables__1.cdtm51[k
		    - 1] + a2[i__ - 1] * gd_tables__1.cdtm58[k - 1]);
	    co2m2d[i__ + k - 2] = *h1m3 * (a1[i__ - 1] * gd_tables__1.c2dm51[
		    k - 1] + a2[i__ - 1] * gd_tables__1.c2dm58[k - 1]);

	}
    }
    i__1 = lp1;
    for (kp = 2; kp <= i__1; ++kp) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    dift[i__ + kp - 2] = tdav[i__ + kp - 2] / tstdav[i__ + kp - 2];

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	co21[i__ - 1] = 1.f;
	co2sp1[i__ - 1] = 1.f;
	co2sp2[i__ - 1] = 1.f;

    }
    i__2 = lp1;
    for (kp = 2; kp <= i__2; ++kp) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    co2r[i__ + kp - 2] = a1[i__ - 1] * gd_tables__1.co251[kp - 1] +
		    a2[i__ - 1] * gd_tables__1.co258[kp - 1];
	    dco2dt[i__ + kp - 2] = *h1m2 * (a1[i__ - 1] * gd_tables__1.cdt51[
		    kp - 1] + a2[i__ - 1] * gd_tables__1.cdt58[kp - 1]);
	    d2cdt2[i__ + kp - 2] = *h1m3 * (a1[i__ - 1] * gd_tables__1.c2d51[
		    kp - 1] + a2[i__ - 1] * gd_tables__1.c2d58[kp - 1]);
	    r__1 = *haf * dift[i__ + kp - 2];
	    co21[i__ + (kp + 82) - 84] = co2r[i__ + kp - 2] + dift[i__ + kp -
		    2] * (dco2dt[i__ + kp - 2] + r__1 * d2cdt2[i__ + kp - 2]);
	    co2r[i__ + kp - 2] = a1[i__ - 1] * gd_tables__1.co251[kp * 82 -
		    82] + a2[i__ - 1] * gd_tables__1.co258[kp * 82 - 82];
	    dco2dt[i__ + kp - 2] = *h1m2 * (a1[i__ - 1] * gd_tables__1.cdt51[
		    kp * 82 - 82] + a2[i__ - 1] * gd_tables__1.cdt58[kp * 82
		    - 82]);
	    d2cdt2[i__ + kp - 2] = *h1m3 * (a1[i__ - 1] * gd_tables__1.c2d51[
		    kp * 82 - 82] + a2[i__ - 1] * gd_tables__1.c2d58[kp * 82
		    - 82]);
	    r__1 = *haf * dift[i__ + kp - 2];
	    co21[i__ + (kp * 82 + 1) - 84] = co2r[i__ + kp - 2] + dift[i__ +
		    kp - 2] * (dco2dt[i__ + kp - 2] + r__1 * d2cdt2[i__ + kp
		    - 2]);

	}
    }
    i__1 = lp1;
    for (k = 2; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    r__1 = *haf * dift[i__ + k - 2];
	    co2sp1[i__ + k - 2] = co2r1[i__ + k - 2] + dift[i__ + k - 2] * (
		    dco2d1[i__ + k - 2] + r__1 * d2cd21[i__ + k - 2]);
	    r__1 = *haf * dift[i__ + k - 2];
	    co2sp2[i__ + k - 2] = co2r2[i__ + k - 2] + dift[i__ + k - 2] * (
		    dco2d2[i__ + k - 2] + r__1 * d2cd22[i__ + k - 2]);

	}
    }
    i__2 = l;
    for (k = 2; k <= i__2; ++k) {
	i__1 = lp1;
	for (kp = k + 1; kp <= i__1; ++kp) {
	    i__3 = myie;
	    for (i__ = myis; i__ <= i__3; ++i__) {
		dift[i__ + kp - 2] = (tdav[i__ + kp - 2] - tdav[i__ + k - 2])
			/ (tstdav[i__ + kp - 2] - tstdav[i__ + k - 2]);
		co2r[i__ + kp - 2] = a1[i__ - 1] * gd_tables__1.co251[kp + k *
			 82 - 83] + a2[i__ - 1] * gd_tables__1.co258[kp + k *
			82 - 83];
		dco2dt[i__ + kp - 2] = *h1m2 * (a1[i__ - 1] *
			gd_tables__1.cdt51[kp + k * 82 - 83] + a2[i__ - 1] *
			gd_tables__1.cdt58[kp + k * 82 - 83]);
		d2cdt2[i__ + kp - 2] = *h1m3 * (a1[i__ - 1] *
			gd_tables__1.c2d51[kp + k * 82 - 83] + a2[i__ - 1] *
			gd_tables__1.c2d58[kp + k * 82 - 83]);
		r__1 = *haf * dift[i__ + kp - 2];
		co21[i__ + (kp + k * 82) - 84] = co2r[i__ + kp - 2] + dift[
			i__ + kp - 2] * (dco2dt[i__ + kp - 2] + r__1 * d2cdt2[
			i__ + kp - 2]);
		co2r[i__ + kp - 2] = a1[i__ - 1] * gd_tables__1.co251[k + kp *
			 82 - 83] + a2[i__ - 1] * gd_tables__1.co258[k + kp *
			82 - 83];
		dco2dt[i__ + kp - 2] = *h1m2 * (a1[i__ - 1] *
			gd_tables__1.cdt51[k + kp * 82 - 83] + a2[i__ - 1] *
			gd_tables__1.cdt58[k + kp * 82 - 83]);
		d2cdt2[i__ + kp - 2] = *h1m3 * (a1[i__ - 1] *
			gd_tables__1.c2d51[k + kp * 82 - 83] + a2[i__ - 1] *
			gd_tables__1.c2d58[k + kp * 82 - 83]);
		r__1 = *haf * dift[i__ + kp - 2];
		co21[i__ + (k + kp * 82) - 84] = co2r[i__ + kp - 2] + dift[
			i__ + kp - 2] * (dco2dt[i__ + kp - 2] + r__1 * d2cdt2[
			i__ + kp - 2]);

	    }
	}

    }
    i__2 = lp1;
    for (k = 2; k <= i__2; ++k) {
	i__3 = myie;
	for (i__ = myis; i__ <= i__3; ++i__) {
	    dift[i__ + k - 2] = *haf * (vsum3[i__ + k - 2] + vsum3[i__ + (k -
		    1) - 2]);
	    co2r[i__ + k - 2] = a1[i__ - 1] * gd_tables__1.co251[k + k * 82 -
		    83] + a2[i__ - 1] * gd_tables__1.co258[k + k * 82 - 83];
	    dco2dt[i__ + k - 2] = *h1m2 * (a1[i__ - 1] * gd_tables__1.cdt51[k
		    + k * 82 - 83] + a2[i__ - 1] * gd_tables__1.cdt58[k + k *
		    82 - 83]);
	    d2cdt2[i__ + k - 2] = *h1m3 * (a1[i__ - 1] * gd_tables__1.c2d51[k
		    + k * 82 - 83] + a2[i__ - 1] * gd_tables__1.c2d58[k + k *
		    82 - 83]);
	    r__1 = *haf * dift[i__ + k - 2];
	    co21[i__ + (k + k * 82) - 84] = co2r[i__ + k - 2] + dift[i__ + k
		    - 2] * (dco2dt[i__ + k - 2] + r__1 * d2cdt2[i__ + k - 2]);

	}
    }
    i__3 = l;
    for (k = 1; k <= i__3; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    r__1 = *haf * vsum3[i__ + k - 2];
	    co2nbl[i__ + k - 2] = co2mr[i__ + k - 2] + vsum3[i__ + k - 2] * (
		    co2md[i__ + k - 2] + r__1 * co2m2d[i__ + k - 2]);

	}
    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__3 = myie;
	for (i__ = myis; i__ <= i__3; ++i__) {
	    if (t[i__ + k - 2] <= *h25e2) {
		tlsqu[i__ + k - 2] = *b0 + (t[i__ + k - 2] - *h25e2) * (*b1 +
			(t[i__ + k - 2] - *h25e2) * (*b2 + *b3 * (t[i__ + k -
			2] - *h25e2)));
	    } else {
		tlsqu[i__ + k - 2] = *b0;
	    }

	}
    }
    i__3 = lp1;
    for (k = 1; k <= i__3; ++k) {
	i__2 = lp1;
	for (kp = 1; kp <= i__2; ++kp) {
	    i__1 = myie;
	    for (i__ = myis; i__ <= i__1; ++i__) {
		co21[i__ + (kp + k * 82) - 84] = co21[i__ + (kp + k * 82) -
			84] * (*one - tlsqu[i__ + kp - 2]) + tlsqu[i__ + kp -
			2];

	    }
	}

    }
    i__3 = lp1;
    for (k = 1; k <= i__3; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    co2sp1[i__ + k - 2] = co2sp1[i__ + k - 2] * (*one - tlsqu[i__ - 1]
		    ) + tlsqu[i__ - 1];
	    co2sp2[i__ + k - 2] = co2sp2[i__ + k - 2] * (*one - tlsqu[i__ - 1]
		    ) + tlsqu[i__ - 1];

	}

    }
    i__3 = l;
    for (k = 1; k <= i__3; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    co2nbl[i__ + k - 2] = co2nbl[i__ + k - 2] * (*one - tlsqu[i__ + k
		    - 2]) + tlsqu[i__ + k - 2];

	}

    }
    fst88_(&heatra[2], &grnflx[1], &topflx[1], qh2o, &press[2], p, delp,
	    delp2, &temp[2], t, &cldfac[84], &nclds[1], &ktop[2], &kbtm[2], &
	    camt[2], co21, co2nbl, co2sp1, co2sp2, var1, var2, var3, var4,
	    cntval, toto3, tphio3, totphi, totvo2, emx1, emx2, empl, &bo3rnd[
	    1], &ao3rnd[1], &apcm[1], &bpcm[1], &atpcm[1], &btpcm[1], &acomb[
	    1], &bcomb[1], &betacm[1], ten, hp1, haf, one, four, hm1ez,
	    radcon, quartr, two, hm6666m2, hmp66667, hmp5, hp166666, h41666m2,
	     radcon1, h16e1, h28e1, h25e2, h44194m2, h1p41819, ids, ide, jds,
	    jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte,
	    kts, kte, gt, clock, work);
    return 0;
}
















 GD_DEVICE int o3int_(real *phalf, real *dduo3n, real *ddo3n2, real *
	ddo3n3, real *ddo3n4, integer *ids, integer *ide, integer *jds,
	integer *jde, integer *kds, integer *kde, integer *ims, integer *ime,
	integer *jms, integer *jme, integer *kms, integer *kme, integer *its,
	integer *ite, integer *jts, integer *jte, integer *kts, integer *kte, GDTable *gt, GDClock *clock, GDScratch *work)
{


    real o3lo1[160]	 = { 14.9f,14.2f,13.3f,
	    12.5f,11.2f,10.3f,9.5f,8.6f,7.5f,7.4f,14.5f,14.1f,13.f,11.8f,
	    10.5f,9.8f,9.2f,7.9f,7.4f,7.4f,11.8f,11.5f,10.9f,10.5f,9.9f,9.6f,
	    8.9f,7.5f,7.2f,7.2f,7.3f,7.7f,7.8f,8.4f,8.4f,8.5f,7.9f,7.4f,7.1f,
	    7.1f,4.1f,4.4f,5.3f,6.6f,6.9f,7.5f,7.4f,7.2f,7.f,6.9f,1.8f,1.9f,
	    2.5f,3.3f,4.5f,5.8f,6.3f,6.3f,6.4f,6.1f,.4f,.5f,.8f,1.2f,2.7f,
	    3.6f,4.6f,4.7f,5.f,5.2f,.1f,.15f,.2f,.5f,1.4f,2.1f,3.f,3.2f,3.5f,
	    3.9f,.07f,.1f,.12f,.3f,1.f,1.4f,1.8f,1.9f,2.3f,2.5f,.06f,.08f,.1f,
	    .15f,.6f,.8f,1.4f,1.5f,1.5f,1.6f,.05f,.05f,.06f,.09f,.2f,.4f,.7f,
	    .8f,.9f,.9f,.05f,.05f,.06f,.08f,.1f,.13f,.2f,.25f,.3f,.4f,.05f,
	    .05f,.05f,.06f,.07f,.07f,.08f,.09f,.1f,.13f,.05f,.05f,.05f,.05f,
	    .06f,.06f,.06f,.06f,.07f,.07f,.05f,.05f,.05f,.05f,.05f,.05f,.05f,
	    .06f,.06f,.06f,.04f,.04f,.04f,.04f,.04f,.04f,.04f,.05f,.05f,.05f }
	    ;
    real o3lo2[160]	 = { 14.8f,14.2f,13.8f,
	    12.2f,11.f,9.8f,8.5f,7.8f,7.4f,6.9f,13.2f,13.f,12.5f,11.3f,10.4f,
	    9.f,7.8f,7.5f,7.f,6.6f,10.6f,10.6f,10.7f,10.1f,9.4f,8.6f,7.5f,7.f,
	    6.5f,6.1f,7.f,7.3f,7.5f,7.5f,7.5f,7.3f,6.7f,6.4f,6.f,5.8f,3.8f,
	    4.f,4.7f,5.f,5.2f,5.9f,5.8f,5.6f,5.5f,5.5f,1.4f,1.6f,2.4f,3.f,
	    3.7f,4.1f,4.6f,4.8f,5.1f,5.f,.4f,.5f,.9f,1.2f,2.f,2.7f,3.2f,3.6f,
	    4.3f,4.1f,.07f,.1f,.2f,.3f,.8f,1.4f,2.1f,2.4f,2.7f,3.f,.06f,.07f,
	    .09f,.15f,.3f,.7f,1.2f,1.4f,1.6f,2.f,.05f,.05f,.06f,.12f,.15f,.3f,
	    .6f,.7f,.8f,.8f,.04f,.05f,.06f,.08f,.09f,.15f,.3f,.4f,.4f,.4f,
	    .04f,.04f,.05f,.055f,.06f,.09f,.12f,.13f,.15f,.15f,.03f,.03f,
	    .045f,.052f,.055f,.06f,.07f,.07f,.06f,.07f,.03f,.03f,.04f,.051f,
	    .052f,.052f,.06f,.06f,.05f,.05f,.02f,.02f,.03f,.05f,.05f,.05f,
	    .04f,.04f,.04f,.04f,.02f,.02f,.02f,.04f,.04f,.04f,.03f,.03f,.03f,
	    .03f };
    real o3lo3[160]	 = { 14.5f,14.f,13.5f,11.3f,
	    11.f,10.f,9.f,8.3f,7.5f,7.3f,13.5f,13.2f,12.5f,11.1f,10.4f,9.7f,
	    8.2f,7.8f,7.4f,6.8f,10.8f,10.9f,11.f,10.4f,10.f,9.6f,7.9f,7.5f,
	    7.f,6.7f,7.3f,7.5f,7.8f,8.5f,9.f,8.5f,7.7f,7.4f,6.9f,6.5f,4.1f,
	    4.5f,5.3f,6.2f,7.3f,7.7f,7.3f,7.f,6.6f,6.4f,1.8f,2.f,2.2f,3.8f,
	    4.3f,5.6f,6.2f,6.2f,6.4f,6.2f,.3f,.5f,.6f,1.5f,2.8f,3.7f,4.5f,
	    4.7f,5.5f,5.6f,.09f,.1f,.15f,.6f,1.2f,2.1f,3.f,3.5f,4.f,4.3f,.06f,
	    .08f,.1f,.3f,.6f,1.1f,1.9f,2.2f,2.9f,3.f,.04f,.05f,.06f,.15f,.45f,
	    .6f,1.1f,1.3f,1.6f,1.8f,.04f,.04f,.04f,.08f,.2f,.3f,.55f,.6f,.75f,
	    .9f,.04f,.04f,.04f,.05f,.06f,.1f,.12f,.15f,.2f,.25f,.04f,.04f,
	    .03f,.04f,.05f,.06f,.07f,.07f,.07f,.08f,.03f,.03f,.04f,.05f,.05f,
	    .05f,.05f,.05f,.05f,.05f,.03f,.03f,.03f,.04f,.04f,.04f,.05f,.05f,
	    .04f,.04f,.02f,.02f,.02f,.04f,.04f,.04f,.04f,.04f,.03f,.03f };
    real o3lo4[160]	 = { 14.2f,13.8f,13.2f,
	    12.5f,11.7f,10.5f,8.6f,7.8f,7.5f,6.6f,12.5f,12.4f,12.2f,11.7f,
	    10.8f,9.8f,7.8f,7.2f,6.5f,6.1f,10.6f,10.5f,10.4f,10.1f,9.6f,9.f,
	    7.1f,6.8f,6.1f,5.9f,7.f,7.4f,7.9f,7.8f,7.6f,7.3f,6.2f,6.1f,5.8f,
	    5.6f,4.2f,4.6f,5.1f,5.6f,5.9f,5.9f,5.9f,5.8f,5.6f,5.3f,2.1f,2.3f,
	    2.6f,2.9f,3.5f,4.3f,4.8f,4.9f,5.1f,5.1f,.7f,.8f,1.f,1.5f,2.f,2.8f,
	    3.5f,3.6f,3.7f,4.f,.15f,.2f,.4f,.5f,.6f,1.4f,2.1f,2.2f,2.3f,2.5f,
	    .08f,.1f,.15f,.25f,.3f,.9f,1.2f,1.3f,1.4f,1.6f,.07f,.08f,.1f,.14f,
	    .2f,.5f,.7f,.9f,.9f,.8f,.05f,.06f,.08f,.12f,.14f,.2f,.35f,.4f,.6f,
	    .5f,.05f,.05f,.08f,.09f,.09f,.09f,.11f,.12f,.15f,.18f,.04f,.05f,
	    .06f,.07f,.07f,.08f,.08f,.08f,.08f,.08f,.04f,.04f,.05f,.07f,.07f,
	    .07f,.07f,.07f,.06f,.05f,.02f,.02f,.04f,.05f,.05f,.05f,.05f,.05f,
	    .04f,.04f,.02f,.02f,.03f,.04f,.04f,.04f,.04f,.04f,.03f,.03f };
    struct {
	real e_1[250];
	} equiv_1 = { .55f, .5f, .45f, .45f, .4f, .35f, .35f, .3f, .3f, .3f,
		.55f, .51f, .46f, .47f, .42f, .38f, .37f, .36f, .35f, .35f,
		.55f, .53f, .48f, .49f, .44f, .42f, .41f, .4f, .38f, .38f,
		.6f, .55f, .52f, .52f, .5f, .47f, .46f, .44f, .42f, .41f,
		.65f, .6f, .55f, .56f, .53f, .52f, .5f, .48f, .45f, .45f,
		.75f, .65f, .6f, .6f, .55f, .55f, .55f, .5f, .48f, .47f, .8f,
		.75f, .75f, .75f, .7f, .7f, .65f, .63f, .6f, .6f, .9f, .85f,
		.85f, .8f, .8f, .75f, .75f, .74f, .72f, .71f, 1.1f, 1.05f,
		1.f, .9f, .9f, .9f, .85f, .83f, .8f, .8f, 1.4f, 1.3f, 1.25f,
		1.25f, 1.25f, 1.2f, 1.15f, 1.1f, 1.05f, 1.f, 1.7f, 1.7f, 1.6f,
		 1.6f, 1.6f, 1.6f, 1.6f, 1.6f, 1.5f, 1.5f, 2.1f, 2.f, 1.9f,
		1.9f, 1.9f, 1.8f, 1.8f, 1.8f, 1.7f, 1.7f, 2.4f, 2.3f, 2.2f,
		2.2f, 2.2f, 2.1f, 2.1f, 2.1f, 2.f, 2.f, 2.7f, 2.5f, 2.5f,
		2.5f, 2.5f, 2.5f, 2.4f, 2.4f, 2.3f, 2.3f, 2.9f, 2.8f, 2.7f,
		2.7f, 2.7f, 2.7f, 2.7f, 2.7f, 2.6f, 2.6f, 3.1f, 3.1f, 3.f,
		3.f, 3.f, 3.f, 3.f, 3.f, 2.9f, 2.8f, 3.3f, 3.4f, 3.4f, 3.6f,
		3.7f, 3.9f, 4.f, 4.1f, 4.f, 3.8f, 3.6f, 3.8f, 3.9f, 4.2f,
		4.7f, 5.3f, 5.6f, 5.7f, 5.5f, 5.2f, 4.1f, 4.3f, 4.7f, 5.2f,
		6.f, 6.7f, 7.f, 6.8f, 6.4f, 6.2f, 5.4f, 5.7f, 6.f, 6.6f, 7.3f,
		 8.f, 8.4f, 7.7f, 7.1f, 6.7f, 6.7f, 6.8f, 7.f, 7.6f, 8.3f,
		10.f, 9.6f, 8.2f, 7.5f, 7.2f, 9.2f, 9.3f, 9.4f, 9.6f, 10.3f,
		10.6f, 10.f, 8.5f, 7.7f, 7.3f, 12.6f, 12.1f, 12.f, 12.1f,
		11.7f, 11.f, 10.f, 8.6f, 7.8f, 7.4f, 14.2f, 13.5f, 13.1f,
		12.8f, 11.9f, 10.9f, 9.8f, 8.5f, 7.8f, 7.5f, 14.3f, 14.f,
		13.4f, 12.7f, 11.6f, 10.6f, 9.3f, 8.4f, 7.6f, 7.3f };

    struct {
	real e_1[82];
	} equiv_3 = { 0.f, 1.027246e-5f, 1.239831e-5f, 1.491845e-5f,
		1.788053e-5f, 2.135032e-5f, 2.540162e-5f, 3.011718e-5f,
		3.558949e-5f, 4.192172e-5f, 4.922875e-5f, 5.763817e-5f,
		6.729146e-5f, 7.834518e-5f, 9.097232e-5f, 1.053635e-4f,
		1.217288e-4f, 1.402989e-4f, 1.61327e-4f, 1.850904e-4f,
		2.119495e-4f, 2.423836e-4f, 2.76898e-4f, 3.160017e-4f,
		3.602623e-4f, 4.103126e-4f, 4.668569e-4f, 5.306792e-4f,
		6.026516e-4f, 6.839018e-4f, 7.759249e-4f, 8.803303e-4f,
		9.987843e-4f, .001133178f, .001285955f, .00146036f,
		.001660001f, .001888764f, .002151165f, .002452466f,
		.002798806f, .003197345f, .003656456f, .004185934f,
		.004797257f, .005503893f, .006321654f, .007269144f,
		.008368272f, .009644873f, .01112946f, .0128581f, .01487354f,
		.01722643f, .01997696f, .0231967f, .02697093f, .03140135f,
		.03660952f, .0427409f, .04996992f, .05848471f, .06847525f,
		.08017242f, .09386772f, .1099026f, .1286765f, .1506574f,
		.1763932f, .2065253f, .2415209f, .2814823f, .3266369f,
		.3774861f, .4345638f, .4984375f, .5697097f, .6490189f,
		.7370409f, .8344896f, .942119f, 1.f };

    struct {
	real e_1[81];
	} equiv_5 = { 9.3e-6f, 1.129521e-5f, 1.360915e-5f, 1.63537e-5f,
		1.95499e-5f, 2.331653e-5f, 2.767314e-5f, 3.277707e-5f,
		3.864321e-5f, 4.547839e-5f, 5.328839e-5f, 6.234301e-5f,
		7.263268e-5f, 8.450696e-5f, 9.793231e-5f, 1.133587e-4f,
		1.30717e-4f, 1.505832e-4f, 1.728373e-4f, 1.982122e-4f,
		2.266389e-4f, 2.59222e-4f, 2.957792e-4f, 3.376068e-4f,
		3.844381e-4f, 4.379281e-4f, 4.976965e-4f, 5.658476e-4f,
		6.418494e-4f, 7.287094e-4f, 8.261995e-4f, 9.380076e-4f,
		.001063498f, .001207423f, .001369594f, .001557141f,
		.001769657f, .002015887f, .00229552f, .002620143f,
		.002989651f, .003419469f, .003909867f, .004481491f,
		.005135272f, .005898971f, .006774619f, .007799763f,
		.008978218f, .01036103f, .01195488f, .01382957f, .01599631f,
		.01855114f, .02151235f, .02501293f, .0290822f, .03390544f,
		.03952926f, .04621349f, .05403168f, .06330472f, .07406807f,
		.08677983f, .1015345f, .1189603f, .1391863f, .1630739f,
		.1908004f, .2235461f, .260941f, .3036404f, .351375f,
		.4055375f, .4656677f, .5335132f, .6083618f, .6923932f,
		.7845676f, .8875882f, 1.f };



    integer i__1, i__2;
    real r__1;


    integer k, l, n;
#define p ((real *)&equiv_5)
#define p1 ((real *)&equiv_5)
#define p2 ((real *)&equiv_5 + 48)
    integer ki, jj, kk;
#define ph ((real *)&equiv_3)
    integer nk, kq, np;
#define ph1 ((real *)&equiv_3)
#define ph2 ((real *)&equiv_3 + 45)
    integer nm1, np2, ken;

    real *ro31=work->o3int_ro31;
    real *ro32=work->o3int_ro32;
    integer nkk, nkp, nkm;
#define o3hi ((real *)&equiv_1)
    real o3rd, o3du;
#define o3hi1 ((real *)&equiv_1)
#define o3hi2 ((real *)&equiv_1 + 160)

    real *rbar=work->o3int_rbar;
    integer nkmm;

    real *rstd=work->o3int_rstd;
    real *dduo3=work->o3int_dduo3;
    real *duo3n=work->o3int_duo3n;
    real o3tot;
    real *o35deg=work->o3int_o35deg;
    real *rdata=work->o3int_rdata;
    integer ncase, itape;

    real *tempn=work->o3int_tempn;
    integer iplace;


    ddo3n4 -= 38;
    ddo3n3 -= 38;
    ddo3n2 -= 38;
    dduo3n -= 38;
    --phalf;


    n = *kte;
    np = n + 1;
    np2 = n + 2;
    nm1 = n - 1;
    nkk = 41;
    nk = 81;
    nkp = nk + 1;
    i__1 = np;
    for (k = 1; k <= i__1; ++k) {

	r__1 = phalf[k] * .01f;
	phalf[k] = r__1 * 1e3f;
    }
    i__1 = nk;
    for (k = 1; k <= i__1; ++k) {
	ph[k - 1] *= 1013250.f;

	p[k - 1] *= 1013250.f;
    }
    ph[nkp - 1] *= 1013250.f;
    for (k = 1; k <= 25; ++k) {
	for (l = 1; l <= 10; ++l) {
	    ro31[l + k * 10 - 11] = o3hi[l + k * 10 - 11];
	    ro32[l + k * 10 - 11] = o3hi[l + k * 10 - 11];

	}
    }
    for (ncase = 1; ncase <= 4; ++ncase) {
	itape = ncase + 50;
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
		for (l = 1; l <= 10; ++l) {
		    ro31[l + k * 10 - 11] = o3lo1[l + (k - 25) * 10 - 11];
		    ro32[l + k * 10 - 11] = o3lo2[l + (k - 25) * 10 - 11];

		}
	    }
	}
	if (ncase == 3 || ncase == 4) {
	    for (k = 26; k <= 41; ++k) {
		for (l = 1; l <= 10; ++l) {
		    ro31[l + k * 10 - 11] = o3lo3[l + (k - 25) * 10 - 11];
		    ro32[l + k * 10 - 11] = o3lo4[l + (k - 25) * 10 - 11];

		}
	    }
	}
	i__1 = nkk;
	for (kk = 1; kk <= i__1; ++kk) {
	    for (l = 1; l <= 10; ++l) {
		duo3n[l + kk * 19 - 20] = ro31[11 - l + kk * 10 - 11];

		duo3n[l + 9 + kk * 19 - 20] = ro32[l + kk * 10 - 11];
	    }
	    duo3n[kk * 19 - 10] = (ro31[kk * 10 - 10] + ro32[kk * 10 - 10]) *
		    .5f;

	}
	if (ncase == 2 || ncase == 4) {
	    i__1 = nkk;
	    for (kk = 1; kk <= i__1; ++kk) {
		for (l = 1; l <= 19; ++l) {
		    tempn[l - 1] = duo3n[20 - l + kk * 19 - 20];

		}
		for (l = 1; l <= 19; ++l) {
		    duo3n[l + kk * 19 - 20] = tempn[l - 1];

		}

	    }
	}
	for (l = 1; l <= 19; ++l) {
	    i__1 = nkk;
	    for (kk = 1; kk <= i__1; ++kk) {

		rstd[kk - 1] = duo3n[l + kk * 19 - 20];
	    }
	    nkm = nk - 1;
	    nkmm = nk - 3;
	    i__1 = nkmm;
	    for (k = 4; k <= i__1; k += 2) {
		ki = k / 2;

		rdata[k - 1] = (rstd[ki - 1] + rstd[ki]) * .5f - (rstd[ki + 1]
			 - rstd[ki] - rstd[ki - 1] + rstd[ki - 2]) / 16.f;
	    }
	    rdata[1] = (rstd[1] + rstd[0]) * .5f;
	    rdata[nkm - 1] = (rstd[nkk - 1] + rstd[nkk - 2]) * .5f;
	    i__1 = nk;
	    for (k = 1; k <= i__1; k += 2) {
		kq = (k + 1) / 2;

		rdata[k - 1] = rstd[kq - 1];
	    }
	    i__1 = n;
	    for (kk = 1; kk <= i__1; ++kk) {
		rbar[kk - 1] = 0.f;
		i__2 = nk;
		for (k = 1; k <= i__2; ++k) {
		    if (ph[k] < phalf[kk]) {
			goto L98;
		    }
		    if (ph[k - 1] > phalf[kk + 1]) {
			goto L98;
		    }
		    if (ph[k] < phalf[kk + 1] && ph[k - 1] < phalf[kk]) {
			rbar[kk - 1] += rdata[k - 1] * (ph[k] - phalf[kk]);
		    }
		    if (ph[k] < phalf[kk + 1] && ph[k - 1] >= phalf[kk]) {
			rbar[kk - 1] += rdata[k - 1] * (ph[k] - ph[k - 1]);
		    }
		    if (ph[k] > phalf[kk + 1] && ph[k - 1] > phalf[kk]) {
			rbar[kk - 1] += rdata[k - 1] * (phalf[kk + 1] - ph[k
				- 1]);
		    }
L98:
		    ;
		}
		rbar[kk - 1] /= phalf[kk + 1] - phalf[kk];
		if (rbar[kk - 1] > 0.f) {
		    goto L99;
		}
		i__2 = nk;
		for (k = 1; k <= i__2; ++k) {
		    if (ph[k - 1] < phalf[kk] && ph[k] >= phalf[kk + 1]) {
			rbar[kk - 1] = rdata[k - 1];
		    }

		}
L99:
		;
	    }
	    o3rd = 0.f;
	    for (kk = 1; kk <= 80; ++kk) {

		o3rd += rdata[kk - 1] * (ph[kk] - ph[kk - 1]);
	    }
	    o3rd += rdata[80] * (p[80] - ph[80]);
	    o3rd /= 980.f;
	    o3tot = 0.f;
	    i__1 = n;
	    for (kk = 1; kk <= i__1; ++kk) {

		o3tot += rbar[kk - 1] * (phalf[kk + 1] - phalf[kk]);
	    }
	    o3tot /= 980.f;
	    o3du = o3tot / 2.144f;
	    i__1 = n;
	    for (kk = 1; kk <= i__1; ++kk) {

		dduo3[l + kk * 19 - 20] = rbar[kk - 1] * .01f;
	    }

	}
	i__1 = n;
	for (kk = 1; kk <= i__1; ++kk) {
	    for (l = 1; l <= 19; ++l) {
		o35deg[(l << 1) - 1 + kk * 37 - 38] = dduo3[l + kk * 19 - 20];

	    }
	    for (l = 1; l <= 18; ++l) {
		o35deg[(l << 1) + kk * 37 - 38] = (dduo3[l + kk * 19 - 20] +
			dduo3[l + 1 + kk * 19 - 20]) * .5f;

	    }

	}
	if (iplace == 1) {
	    for (jj = 1; jj <= 37; ++jj) {
		i__1 = n;
		for (ken = 1; ken <= i__1; ++ken) {
		    dduo3n[jj + ken * 37] = o35deg[jj + ken * 37 - 38];

		}
	    }
	} else if (iplace == 2) {
	    for (jj = 1; jj <= 37; ++jj) {
		i__1 = n;
		for (ken = 1; ken <= i__1; ++ken) {
		    ddo3n2[jj + ken * 37] = o35deg[jj + ken * 37 - 38];

		}
	    }
	} else if (iplace == 3) {
	    for (jj = 1; jj <= 37; ++jj) {
		i__1 = n;
		for (ken = 1; ken <= i__1; ++ken) {
		    ddo3n3[jj + ken * 37] = o35deg[jj + ken * 37 - 38];

		}
	    }
	} else if (iplace == 4) {
	    for (jj = 1; jj <= 37; ++jj) {
		i__1 = n;
		for (ken = 1; ken <= i__1; ++ken) {
		    ddo3n4[jj + ken * 37] = o35deg[jj + ken * 37 - 38];

		}
	    }
	}

    }
    return 0;
}

#undef o3hi2
#undef o3hi1
#undef o3hi
#undef ph2
#undef ph1
#undef ph
#undef p2
#undef p1
#undef p

















 GD_DEVICE int ozon2d_(integer *lk, real *pozn, real *xlat, real *qo3,
	integer *myis, integer *myie, integer *ids, integer *ide, integer *
	jds, integer *jde, integer *kds, integer *kde, integer *ims, integer *
	ime, integer *jms, integer *jme, integer *kms, integer *kme, integer *
	its, integer *ite, integer *jts, integer *jte, integer *kts, integer *
	kte, GDTable *gt, GDClock *clock, GDScratch *work)
{

    integer i__1, i__2, i__3, i__4;
    real r__1, r__2;





    integer i__, k, it;
    real th2, do3v, aphi;
    integer ilog;

    real aplo;
    real *qo3o3=work->ozon2d_qo3o3;
    real do3vp;
    integer nhalf;

    real *tthan=work->ozon2d_tthan;

    integer *jjrow=work->ozon2d_jjrow;
    integer numitr;


    qo3 -= 2;
    --xlat;
    pozn -= 2;


    i__1 = *myie;
    for (i__ = *myis; i__ <= i__1; ++i__) {
	th2 = xlat[i__] * .2f;
	jjrow[i__ - 1] = 19.001f - th2;
	tthan[i__ - 1] = 19 - jjrow[i__ - 1] - th2;
    }
    for (k = 1; k <= 81; ++k) {
	i__1 = *myie;
	for (i__ = *myis; i__ <= i__1; ++i__) {
	    r__2 = gd_tables__1.xduo3n[jjrow[i__ - 1] + k * 37 - 38] +
		    gd_clock__1.rsin1 * gd_tables__1.xdo3n2[jjrow[i__ - 1] +
		    k * 37 - 38];
	    r__1 = r__2 + gd_clock__1.rcos1 * gd_tables__1.xdo3n3[jjrow[i__ -
		    1] + k * 37 - 38];
	    do3v = r__1 + gd_clock__1.rcos2 * gd_tables__1.xdo3n4[jjrow[i__ -
		    1] + k * 37 - 38];
	    r__2 = gd_tables__1.xduo3n[jjrow[i__ - 1] + 1 + k * 37 - 38] +
		    gd_clock__1.rsin1 * gd_tables__1.xdo3n2[jjrow[i__ - 1] +
		    1 + k * 37 - 38];
	    r__1 = r__2 + gd_clock__1.rcos1 * gd_tables__1.xdo3n3[jjrow[i__ -
		    1] + 1 + k * 37 - 38];
	    do3vp = r__1 + gd_clock__1.rcos2 * gd_tables__1.xdo3n4[jjrow[i__
		    - 1] + 1 + k * 37 - 38];
	    qo3o3[i__ + k - 2] = (do3v + tthan[i__ - 1] * (do3vp - do3v)) *
		    1e-4f;
	}
    }
    numitr = 0;
    ilog = 81;
L20:
    ilog = (ilog + 1) / 2;
    if (ilog == 1) {
	goto L25;
    }
    ++numitr;
    goto L20;
L25:
    i__1 = *lk;
    for (k = 1; k <= i__1; ++k) {
	nhalf = 41;
	i__2 = *myie;
	for (i__ = *myis; i__ <= i__2; ++i__) {
	    jjrow[i__ - 1] = nhalf;
	}
	i__2 = numitr;
	for (it = 1; it <= i__2; ++it) {
	    nhalf = (nhalf + 1) / 2;
	    i__3 = *myie;
	    for (i__ = *myis; i__ <= i__3; ++i__) {
		if (pozn[i__ + k] < gd_tables__1.prgfdl[jjrow[i__ - 1] - 2]) {
		    jjrow[i__ - 1] -= nhalf;
		} else if (pozn[i__ + k] >= gd_tables__1.prgfdl[jjrow[i__ - 1]
			 - 1]) {
		    jjrow[i__ - 1] += nhalf;
		}

		i__4 = jjrow[i__ - 1];
		jjrow[i__ - 1] = min(i__4,81);

		i__4 = jjrow[i__ - 1];
		jjrow[i__ - 1] = max(i__4,2);
	    }

	}
	i__2 = *myie;
	for (i__ = *myis; i__ <= i__2; ++i__) {
	    if (pozn[i__ + k] < gd_tables__1.prgfdl[0]) {
		qo3[i__ + k] = qo3o3[i__ - 1];
	    } else if (pozn[i__ + k] > gd_tables__1.prgfdl[80]) {
		qo3[i__ + k] = qo3o3[i__ + 79];
	    } else {
		aplo = (real)gd_log(gd_tables__1.prgfdl[jjrow[i__ - 1] - 2]);
		aphi = (real)gd_log(gd_tables__1.prgfdl[jjrow[i__ - 1] - 1]);
		qo3[i__ + k] = qo3o3[i__ + jjrow[i__ - 1] - 2] + ((real)gd_log(
			pozn[i__ + k]) - aphi) / (aplo - aphi) * (qo3o3[i__ +
			(jjrow[i__ - 1] - 1) - 2] - qo3o3[i__ + jjrow[i__ - 1]
			 - 2]);
	    }

	}

    }
    return 0;
}


















GD_DEVICE real radfs_c_b17 = 0.f;
GD_DEVICE real radfs_c_b18 = 1.f;
GD_DEVICE real radfs_c_b19 = 1800.f;
GD_DEVICE real radfs_c_b20 = 9.8692328265315155e-7f;
GD_DEVICE real radfs_c_b21 = 6.0810809135437012f;
GD_DEVICE real radfs_c_b22 = 1.6599999666213989f;
GD_DEVICE real radfs_c_b23 = .0010197162628173828f;
GD_DEVICE real radfs_c_b24 = 3.0000001424923539e-4f;
GD_DEVICE real radfs_c_b25 = 7.6681098937988281f;
GD_DEVICE real radfs_c_b26 = .62199997901916504f;
GD_DEVICE real radfs_c_b27 = 1.0063817557082189e-9f;
GD_DEVICE real radfs_c_b28 = 1013250.f;
GD_DEVICE real radfs_c_b29 = 810600.125f;
GD_DEVICE real radfs_c_b30 = 202649.90625f;
GD_DEVICE real radfs_c_b31 = .0030000000260770321f;
GD_DEVICE real radfs_c_b32 = .0010000000474974513f;
GD_DEVICE real radfs_c_b33 = .0099999997764825821f;
GD_DEVICE real radfs_c_b34 = 250.f;
GD_DEVICE real radfs_c_b35 = -5.1926410378655419e-5f;
GD_DEVICE real radfs_c_b36 = -1.0680131481421995e-6f;
GD_DEVICE real radfs_c_b37 = -1.8113332043867558e-4f;
GD_DEVICE real radfs_c_b38 = -6.7303517425898463e-8f;
GD_DEVICE real radfs_c_b39 = .5f;
GD_DEVICE real radfs_c_b40 = 10.f;
GD_DEVICE real radfs_c_b41 = .10000000149011612f;
GD_DEVICE real radfs_c_b42 = 4.f;
GD_DEVICE real radfs_c_b43 = -1.f;
GD_DEVICE real radfs_c_b44 = 8.4270000457763672f;
GD_DEVICE real radfs_c_b45 = .25f;
GD_DEVICE real radfs_c_b46 = 2.f;
GD_DEVICE real radfs_c_b47 = -.066666997969150543f;
GD_DEVICE real radfs_c_b48 = -.66667002439498901f;
GD_DEVICE real radfs_c_b49 = -.5f;
GD_DEVICE real radfs_c_b50 = .16666600108146667f;
GD_DEVICE real radfs_c_b51 = .041666600853204727f;
GD_DEVICE real radfs_c_b52 = .11866619437932968f;
GD_DEVICE real radfs_c_b53 = 16.f;
GD_DEVICE real radfs_c_b54 = 28.f;
GD_DEVICE real radfs_c_b55 = .044194001704454422f;
GD_DEVICE real radfs_c_b56 = 1.4181909561157227f;
GD_DEVICE real radfs_c_b60 = 35.f;
GD_DEVICE real radfs_c_b61 = 1224.f;
GD_DEVICE real radfs_c_b65 = 697667.f;
GD_DEVICE real radfs_c_b66 = .21899999678134918f;
GD_DEVICE real radfs_c_b67 = .81599998474121094f;
GD_DEVICE real radfs_c_b70 = .0023499999660998583f;
GD_DEVICE real radfs_c_b71 = .25999999046325684f;
GD_DEVICE real radfs_c_b72 = .012900000438094139f;
GD_DEVICE real radfs_c_b73 = 7.5826502870768309e-4f;
GD_DEVICE real radfs_c_b74 = 103.59999847412109f;
GD_DEVICE real radfs_c_b75 = 1.0820000171661377f;
GD_DEVICE real radfs_c_b76 = -.80500000715255737f;
GD_DEVICE real radfs_c_b77 = 138.60000610351562f;
GD_DEVICE real radfs_c_b78 = .065800003707408905f;
GD_DEVICE real radfs_c_b79 = .021180000156164169f;
GD_DEVICE real radfs_c_b80 = .041999999433755875f;
GD_DEVICE real radfs_c_b81 = 3.2299998565576971e-4f;
GD_DEVICE real radfs_c_b84 = 1.8999999761581421f;
GD_DEVICE real radfs_c_b85 = 50.f;

 GD_DEVICE int radfs_(real *qs, real *pp, real *ppi, real *qqh2o, real *
	tt, real *o3qo3, real *tsfc, real *slmsk, real *albedo, real *xlat,
	real *camt, integer *ktop, integer *kbtm, integer *nclds, real *emcld,
	 real *rrcl, real *ttcl, real *coszro, real *taudar, integer *ibeg,
	integer *ko3, integer *kalb, integer *itimsw, integer *itimlw, real *
	swh, real *hlw, real *flwup, real *fswup, real *fswdn, real *fswdns,
	real *fswups, real *flwdns, real *flwups, real *fswdnsc, integer *ids,
	 integer *ide, integer *jds, integer *jde, integer *kds, integer *kde,
	 integer *ims, integer *ime, integer *jms, integer *jme, integer *kms,
	 integer *kme, integer *its, integer *ite, integer *jts, integer *jte,
	 integer *kts, integer *kte, integer *imd, integer *jmd, integer *
	jndx, GDTable *gt, GDClock *clock, GDScratch *work)
{


    real bpcm[15] = { -1.1352e-5f,-3.23965e-5f,-4.48417e-5f,
	    -2.30779e-5f,-3.61981e-5f,-1.45117e-5f,1.98349e-5f,-4.86529e-5f,
	    -5.5005e-5f,-6.84057e-5f,-4.47093e-5f,-7.7839e-5f,-9.82953e-5f,
	    -7.72497e-5f,-7.48263e-5f };
    real atpcm[15] = { -.00106346f,.00641531f,.0137362f,.00922513f,
	    .0136162f,.0169791f,.0206959f,.0166223f,.0171776f,.0229724f,
	    .027553f,.0302731f,.0281662f,.0199525f,.0370962f };
    real btpcm[15] = { -7.35731e-6f,-2.94149e-5f,-5.05592e-5f,
	    -2.80894e-5f,-4.92972e-5f,-3.41508e-5f,-3.62947e-5f,-2.50487e-5f,
	    -5.21369e-5f,-7.4626e-5f,-7.44124e-5f,-8.81905e-5f,-9.33645e-5f,
	    -6.64045e-5f,-1.1529e-4f };
    real betacm[15] = { 0.f,0.f,0.f,0.f,188.625f,144.293f,174.098f,
	    90.9366f,49.7489f,22.1212f,11.3124f,7.54174f,5.89554f,4.95227f,
	    0.f };
    real abcff[12] = { 4e-5f,4e-5f,.002f,.035f,.377f,1.95f,9.4f,44.6f,
	    190.f,989.f,2706.f,39011.f };
    real pwts[12] = { .5f,.121416f,.0698f,.1558f,.0631f,.0362f,.0243f,
	    .0158f,.0087f,.001467f,.002342f,.001075f };
    real cfco2 = 508.96f;
    real cfo3 = 466.64f;
    real rrayav = .144f;
    real trn[21] = { 0.f,.05f,.1f,.15f,.2f,.25f,.3f,.35f,.4f,.45f,.5f,
	    .55f,.6f,.65f,.7f,.75f,.8f,.85f,.9f,.95f,1.f };
    real za[20] = { 90.f,88.f,86.f,84.f,82.f,80.f,78.f,76.f,74.f,70.f,
	    66.f,62.f,58.f,54.f,50.f,40.f,30.f,20.f,10.f,0.f };
    real dza[19] = { 2.f,2.f,2.f,2.f,2.f,2.f,2.f,2.f,4.f,4.f,4.f,4.f,
	    4.f,4.f,10.f,10.f,10.f,10.f,10.f };
    real ao3rnd[3] = { 54.3368f,2346.76f,38.4881f };
    real bo3rnd[3] = { 5.26064f,9.22424f,4.96515f };
    real acomb[15] = { 15207.f,3321.94f,527.177f,163.124f,268.808f,
	    53.4591f,26.8071f,12.3133f,6.00199f,.640803f,.0501549f,.0167961f,
	    .017811f,.170166f,.00537083f };
    real bcomb[15] = { .152538f,.118677f,.10366f,.100119f,.127518f,
	    .118409f,.0904061f,.0642011f,.062966f,.0643346f,.0717082f,
	    .062973f,.0875182f,.0857907f,.214005f };
    real apcm[15] = { -6.71879e-4f,.00654345f,.0143657f,.00923593f,
	    .0117022f,.0159596f,.01816f,.0145013f,.0170062f,.0233303f,
	    .0256735f,.0274745f,.0279259f,.0197002f,.0349782f };
    struct {
	real e_1[420];
	} equiv_2 = { .061f, .062f, .072f, .087f, .115f, .163f, .235f, .318f,
		.395f, .472f, .542f, .604f, .655f, .693f, .719f, .732f, .73f,
		.681f, .581f, .453f, .425f, .061f, .062f, .07f, .083f, .108f,
		.145f, .198f, .263f, .336f, .415f, .487f, .547f, .595f, .631f,
		 .656f, .67f, .652f, .602f, .494f, .398f, .37f, .061f, .061f,
		.068f, .079f, .098f, .13f, .174f, .228f, .29f, .357f, .424f,
		.498f, .556f, .588f, .603f, .592f, .556f, .488f, .393f, .342f,
		 .325f, .061f, .061f, .065f, .073f, .086f, .11f, .15f, .192f,
		.248f, .306f, .36f, .407f, .444f, .469f, .48f, .474f, .444f,
		.386f, .333f, .301f, .29f, .061f, .061f, .065f, .07f, .082f,
		.101f, .131f, .168f, .208f, .252f, .295f, .331f, .358f, .375f,
		 .385f, .377f, .356f, .32f, .288f, .266f, .255f, .061f, .061f,
		 .063f, .068f, .077f, .092f, .114f, .143f, .176f, .21f, .242f,
		 .272f, .288f, .296f, .3f, .291f, .273f, .252f, .237f, .266f,
		.22f, .061f, .061f, .062f, .066f, .072f, .084f, .103f, .127f,
		.151f, .176f, .198f, .219f, .236f, .245f, .25f, .246f, .235f,
		.222f, .211f, .205f, .2f, .061f, .061f, .061f, .065f, .071f,
		.079f, .094f, .113f, .134f, .154f, .173f, .185f, .19f, .193f,
		.193f, .19f, .188f, .185f, .182f, .18f, .178f, .061f, .061f,
		.061f, .064f, .067f, .072f, .083f, .099f, .117f, .135f, .15f,
		.16f, .164f, .165f, .164f, .162f, .16f, .159f, .158f, .157f,
		.157f, .061f, .061f, .061f, .062f, .065f, .068f, .074f, .084f,
		 .097f, .111f, .121f, .127f, .13f, .131f, .131f, .13f, .129f,
		.127f, .126f, .125f, .122f, .061f, .061f, .061f, .061f, .062f,
		 .064f, .07f, .076f, .085f, .094f, .101f, .105f, .107f, .106f,
		 .103f, .1f, .097f, .096f, .095f, .095f, .095f, .061f, .061f,
		.061f, .06f, .061f, .062f, .065f, .07f, .075f, .081f, .086f,
		.089f, .09f, .088f, .084f, .08f, .077f, .075f, .074f, .074f,
		.074f, .061f, .061f, .06f, .06f, .06f, .061f, .063f, .065f,
		.068f, .072f, .076f, .077f, .076f, .074f, .071f, .067f, .064f,
		 .062f, .061f, .061f, .061f, .061f, .061f, .06f, .06f, .06f,
		.06f, .061f, .062f, .065f, .068f, .069f, .069f, .068f, .065f,
		.061f, .058f, .055f, .054f, .053f, .052f, .052f, .061f, .061f,
		 .06f, .06f, .06f, .06f, .06f, .06f, .062f, .065f, .065f,
		.063f, .06f, .057f, .054f, .05f, .047f, .046f, .045f, .044f,
		.044f, .061f, .061f, .06f, .06f, .06f, .059f, .059f, .059f,
		.059f, .059f, .058f, .055f, .051f, .047f, .043f, .039f, .035f,
		 .033f, .032f, .031f, .031f, .061f, .061f, .06f, .06f, .06f,
		.059f, .059f, .058f, .057f, .056f, .054f, .051f, .047f, .043f,
		 .039f, .036f, .033f, .03f, .028f, .027f, .026f, .061f, .061f,
		 .06f, .06f, .06f, .059f, .059f, .058f, .057f, .055f, .052f,
		.049f, .045f, .04f, .036f, .032f, .029f, .027f, .026f, .025f,
		.025f, .061f, .061f, .06f, .06f, .06f, .059f, .059f, .058f,
		.056f, .053f, .05f, .046f, .042f, .038f, .034f, .031f, .028f,
		.026f, .025f, .025f, .025f, .061f, .061f, .06f, .06f, .059f,
		.058f, .058f, .057f, .055f, .053f, .05f, .046f, .042f, .038f,
		.034f, .03f, .028f, .029f, .025f, .025f, .025f };



    integer i__1, i__2;
    real r__1, r__2, r__3;





    integer i__, k, l, n;

    real *df=work->radfs_df;
    integer ll, iq;

    real *uf=work->radfs_uf;
    integer ir, jx, lm1, lm2;
    real th2;
    integer lp1, lp2;

    real *qo3=work->radfs_qo3;
    integer lp3, lm3;

    real *fswl=work->radfs_fswl;
    real *hswl=work->radfs_hswl;
    real *ufl=work->radfs_ufl;
    real *dfl=work->radfs_dfl;
    real *gdfnbr=work->radfs_gdfnbr;
    real *cldfac=work->radfs_cldfac;
    real *gdfndr=work->radfs_gdfndr;
    real *eqcmt=work->radfs_eqcmt;
    real *gdfvbr=work->radfs_gdfvbr;
    real *press=work->radfs_press;
    real *gdfvdr=work->radfs_gdfvdr;
    real *temp=work->radfs_temp;
    real *fsw=work->radfs_fsw;
    real *hsw=work->radfs_hsw;
    real *rh2o=work->radfs_rh2o;
    real *heatra=work->radfs_heatra;
    real *tauda=work->radfs_tauda;
    real *phalf=work->radfs_phalf;
    real *coszen=work->radfs_coszen;
    real *grnflx=work->radfs_grnflx;
    real *tthan=work->radfs_tthan;
    real *do3v=work->radfs_do3v;
    real *do3vp=work->radfs_do3vp;

    integer *jjrow=work->radfs_jjrow;

    real *topflx=work->radfs_topflx;
    real *dduo3n=work->radfs_dduo3n;
    real *ddo3n2=work->radfs_ddo3n2;
    real *ddo3n3=work->radfs_ddo3n3;
    real ddo3n4[
	    2997];
#define albd ((real *)&equiv_2)
    real ssolar, zen, dzen, alb1, alb2;
#define alb11 ((real *)&equiv_2)
#define alb22 ((real *)&equiv_2 + 147)
#define alb33 ((real *)&equiv_2 + 294)

    real *alvb=work->radfs_alvb;
    real *alnb=work->radfs_alnb;
    real *alvd=work->radfs_alvd;
    real *alnd=work->radfs_alnd;
    real *gdfvb=work->radfs_gdfvb;
    real *gdfnb=work->radfs_gdfnb;
    real *gdfvd=work->radfs_gdfvd;
    real *gdfnd=work->radfs_gdfnd;
    real *sfcalb=work->radfs_sfcalb;
    real rrvco2;
    real rrco2;
    real tdum;
    real albd0;
    real alvd1;
    real alnd1;
    integer llm2, llm1, myis, myie, llp1;
    extern GD_DEVICE int o3int_(real *, real *, real *, real *, real *,
	     integer *, integer *, integer *, integer *, integer *, integer *,
	     integer *, integer *, integer *, integer *, integer *, integer *,
	     integer *, integer *, integer *, integer *, integer *, integer *, GDTable *gt, GDClock *clock, GDScratch *work)
	    , clo89_(real *, real *, integer *, integer *, integer *, integer
	    *, integer *, integer *, integer *, integer *, integer *, integer
	    *, integer *, integer *, integer *, integer *, integer *, integer
	    *, integer *, integer *, integer *, integer *, integer *, GDTable *gt, GDClock *clock, GDScratch *work), lwr88_(
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, integer *, integer *, integer *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, GDTable *gt, GDClock *clock, GDScratch *work), swr93_(
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, integer *,
	     integer *, integer *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, real *, real *, real *, real *, real *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, GDTable *gt, GDClock *clock, GDScratch *work);


    --fswdnsc;
    --flwups;
    --flwdns;
    --fswups;
    --fswdns;
    --fswdn;
    --fswup;
    --flwup;
    hlw -= 2;
    swh -= 2;
    --taudar;
    --coszro;
    ttcl -= 14;
    rrcl -= 14;
    emcld -= 2;
    --nclds;
    kbtm -= 2;
    ktop -= 2;
    camt -= 2;
    --xlat;
    --albedo;
    --slmsk;
    --tsfc;
    o3qo3 -= 2;
    tt -= 2;
    qqh2o -= 2;
    ppi -= 2;
    pp -= 2;
    --qs;


    l = *kte;
    lp1 = l + 1;
    lp2 = l + 2;
    lp3 = l + 3;
    llp1 = (l << 1) + 1;
    lm1 = l - 1;
    lm2 = l - 2;
    lm3 = l - 3;
    ll = l << 1;
    llm2 = ll - 2;
    llm1 = ll - 1;
    myis = *its;
    myie = *ite;
    ssolar = 1.9600000381469727f / (gd_clock__1.r1 * gd_clock__1.r1);
    ssolar *= .97f;
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	ir = i__ + *ibeg - 1;
	th2 = xlat[ir] * .20000000298023224f;
	jjrow[i__ - 1] = 19.000999450683594f - th2;
	tthan[i__ - 1] = 19 - jjrow[i__ - 1] - th2;
	sfcalb[i__ - 1] = albedo[ir];
	press[i__ + lp1 - 2] = qs[ir] * 10.f;
	temp[i__ + lp1 - 2] = (r__1 = tsfc[ir], abs(r__1));
	coszen[i__ - 1] = coszro[ir];
	tauda[i__ - 1] = taudar[ir];

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    ir = i__ + *ibeg - 1;
	    temp[i__ + k - 2] = tt[ir + k];
	    press[i__ + k - 2] = pp[ir + k] * 10.f;
	    rh2o[i__ + k - 2] = qqh2o[ir + k];
	    if (rh2o[i__ + k - 2] < 3.0000001061125658e-6f) {
		rh2o[i__ + k - 2] = 3.0000001061125658e-6f;
	    }

	}
    }
    if (*ko3 == 0) {
	goto L65;
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    qo3[i__ + k - 2] = o3qo3[i__ + *ibeg - 1 + k];

	}
    }
L65:
    if (*kalb > 0) {
	goto L110;
    }
    iq = 11;
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	if (coszen[i__ - 1] > 0.f && slmsk[i__ + *ibeg - 1] > .5f) {

	    r__1 = coszen[i__ - 1];
	    zen = (real)gd_acos((real)(max(r__1,0.f))) * 57.2957763671875f;
	    if (zen >= 74.f) {
		jx = (integer) ((90.f - zen) * .5f + 1.f);
	    }
	    if (zen < 74.f && zen >= 50.f) {
		jx = (integer) ((74.f - zen) * .25f + 9.f);
	    }
	    if (zen < 50.f) {
		jx = (integer) ((50.f - zen) * .10000000149011612f + 15.f);
	    }
	    dzen = -(zen - za[jx - 1]) / dza[jx - 1];
	    alb1 = albd[iq + jx * 21 - 22] + dzen * (albd[iq + (jx + 1) * 21
		    - 22] - albd[iq + jx * 21 - 22]);
	    alb2 = albd[iq + 1 + jx * 21 - 22] + dzen * (albd[iq + 1 + (jx +
		    1) * 21 - 22] - albd[iq + 1 + jx * 21 - 22]);
	    r__1 = (alb2 - alb1) * 20.f;
	    sfcalb[i__ - 1] = alb1 + r__1 * (.53700000047683716f - trn[iq - 1]
		    );
	}

    }
L110:
    if (*ko3 > 0) {
	goto L135;
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	phalf[0] = 0.f;
	phalf[lp1 - 1] = ppi[i__ + *kme];
	i__2 = lm1;
	for (k = 1; k <= i__2; ++k) {
	    phalf[k] = pp[i__ + k];
	}
	o3int_(phalf, dduo3n, ddo3n2, ddo3n3, ddo3n4, ids, ide, jds, jde, kds,
		 kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts,
		kte, gt, clock, work);
	i__2 = l;
	for (k = 1; k <= i__2; ++k) {
	    r__2 = dduo3n[jjrow[i__ - 1] + k * 37 - 38] + gd_clock__1.rsin1 *
		    ddo3n2[jjrow[i__ - 1] + k * 37 - 38];
	    r__1 = r__2 + gd_clock__1.rcos1 * ddo3n3[jjrow[i__ - 1] + k * 37
		    - 38];
	    do3v[i__ + k - 2] = r__1 + gd_clock__1.rcos2 * ddo3n4[jjrow[i__ -
		    1] + k * 37 - 38];
	    r__2 = dduo3n[jjrow[i__ - 1] + 1 + k * 37 - 38] +
		    gd_clock__1.rsin1 * ddo3n2[jjrow[i__ - 1] + 1 + k * 37 -
		    38];
	    r__1 = r__2 + gd_clock__1.rcos1 * ddo3n3[jjrow[i__ - 1] + 1 + k *
		    37 - 38];
	    do3vp[i__ + k - 2] = r__1 + gd_clock__1.rcos2 * ddo3n4[jjrow[i__
		    - 1] + 1 + k * 37 - 38];
	    qo3[i__ + k - 2] = (do3v[i__ + k - 2] + tthan[i__ - 1] * (do3vp[
		    i__ + k - 2] - do3v[i__ + k - 2])) *
		    9.9999997473787516e-5f;

	}

    }
L135:
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	alvd[i__ - 1] = sfcalb[i__ - 1];
	alnd[i__ - 1] = sfcalb[i__ - 1];
	alvb[i__ - 1] = sfcalb[i__ - 1];
	alnb[i__ - 1] = sfcalb[i__ - 1];
	if (slmsk[i__ + *ibeg - 1] < .5f) {
	    if (sfcalb[i__ - 1] <= .5f) {
		albd0 = (.5f - (real)gd_acos(coszen[i__ - 1]) /
			3.1415927410125732f) * -18.f;
		albd0 = (real)gd_exp(albd0);
		alvd1 = (alvd[i__ - 1] - .054313f) / .945687f;
		alnd1 = (alnd[i__ - 1] - .054313f) / .945687f;
		alvb[i__ - 1] = alvd1 + (1.f - alvd1) * albd0;
		alnb[i__ - 1] = alnd1 + (1.f - alnd1) * albd0;

		r__1 = .5f, r__2 = alvb[i__ - 1];
		alvb[i__ - 1] = min(r__1,r__2);

		r__1 = .5f, r__2 = alnb[i__ - 1];
		alnb[i__ - 1] = min(r__1,r__2);
	    }
	}

    }
    for (n = 1; n <= 2; ++n) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    rrcl[i__ + (n + 12)] = alvd[i__ - 1];
	    ttcl[i__ + (n + 12)] = 0.f;

	}
    }
    for (n = 3; n <= 12; ++n) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    rrcl[i__ + (n + 12)] = alnd[i__ - 1];
	    ttcl[i__ + (n + 12)] = 0.f;

	}
    }
    rrvco2 = 3.3000000985339284e-4f;
    rrco2 = rrvco2 * 1.5194497108459473f;

    if (*itimlw == 0) {
	goto L300;
    }
    i__1 = lp1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    eqcmt[i__ + k - 2] = camt[i__ + k] * emcld[i__ + k];

	}
    }
    clo89_(cldfac, eqcmt, &nclds[1], &kbtm[2], &ktop[2], ids, ide, jds, jde,
	    kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts,
	    kte, gt, clock, work);
    lwr88_(heatra, grnflx, topflx, press, temp, rh2o, qo3, cldfac, eqcmt, &
	    nclds[1], &ktop[2], &kbtm[2], bo3rnd, ao3rnd, apcm, bpcm, atpcm,
	    btpcm, acomb, bcomb, betacm, &radfs_c_b17, &radfs_c_b18, &radfs_c_b19, &radfs_c_b20, &
	    radfs_c_b21, &radfs_c_b22, &radfs_c_b23, &radfs_c_b24, &radfs_c_b25, &radfs_c_b26, &radfs_c_b27, &radfs_c_b28, &
	    radfs_c_b29, &radfs_c_b30, &radfs_c_b31, &radfs_c_b32, &radfs_c_b33, &radfs_c_b34, &radfs_c_b35, &radfs_c_b36, &
	    radfs_c_b37, &radfs_c_b38, &radfs_c_b39, &radfs_c_b40, &radfs_c_b41, &radfs_c_b42, &radfs_c_b43, &radfs_c_b44, &
	    radfs_c_b45, &radfs_c_b46, &radfs_c_b47, &radfs_c_b48, &radfs_c_b49, &radfs_c_b50, &radfs_c_b51, &radfs_c_b52, &
	    radfs_c_b53, &radfs_c_b54, &radfs_c_b55, &radfs_c_b56, ids, ide, jds, jde, kds, kde, ims,
	    ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte, gt, clock, work);
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	ir = i__ + *ibeg - 1;
	flwup[ir] = topflx[i__ - 1] * .001f;
	tdum = (temp[i__ + lp1 - 2] + temp[i__ + l - 2]) * .5f;
	r__3 = tdum * 5.6729998476612309e-8f;
	r__2 = r__3 * tdum;
	r__1 = r__2 * tdum;
	flwups[ir] = r__1 * tdum;
	flwdns[ir] = flwups[ir] - grnflx[i__ - 1] * .001f;

    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	tdum = (heatra[i__ + l - 2] + heatra[i__ + lm1 - 2]) * .5f;
	heatra[i__ + l - 2] = tdum;
	heatra[i__ + lm1 - 2] = tdum;
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    hlw[i__ + *ibeg - 1 + k] = heatra[i__ + k - 2] *
		    1.1574000382097438e-5f;

	}
    }
L300:
    if (*itimsw == 0) {
	goto L350;
    }
    swr93_(fsw, hsw, uf, df, fswl, hswl, ufl, dfl, press, coszen, tauda, rh2o,
	     &rrco2, &ssolar, qo3, &nclds[1], &ktop[2], &kbtm[2], &camt[2], &
	    rrcl[14], &ttcl[14], alvb, alnb, alvd, alnd, gdfvb, gdfnb, gdfvd,
	    gdfnd, abcff, pwts, &radfs_c_b60, &radfs_c_b61, &radfs_c_b18, &radfs_c_b17, &radfs_c_b39, &
	    radfs_c_b65, &radfs_c_b66, &radfs_c_b67, &rrayav, &radfs_c_b23, &cfco2, &cfo3, &radfs_c_b46, &
	    radfs_c_b70, &radfs_c_b71, &radfs_c_b72, &radfs_c_b73, &radfs_c_b74, &radfs_c_b75, &radfs_c_b76, &radfs_c_b77, &
	    radfs_c_b78, &radfs_c_b79, &radfs_c_b80, &radfs_c_b81, &radfs_c_b43, &radfs_c_b22, &radfs_c_b84, &radfs_c_b85, &
	    radfs_c_b44, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme,
	     its, ite, jts, jte, kts, kte, gt, clock, work);
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	ir = i__ + *ibeg - 1;
	fswup[ir] = uf[i__ - 1] * .001f;
	fswdn[ir] = df[i__ - 1] * .001f;
	fswups[ir] = uf[i__ + lp1 - 2] * .001f;
	fswdns[ir] = df[i__ + lp1 - 2] * .001f;
	fswdnsc[ir] = dfl[i__ + lp1 - 2] * .001f;
	gdfvdr[ir - 1] = gdfvd[i__ - 1] * .001f;
	gdfndr[ir - 1] = gdfnd[i__ - 1] * .001f;
	gdfvbr[ir - 1] = gdfvb[i__ - 1] * .001f;
	gdfnbr[ir - 1] = gdfnb[i__ - 1] * .001f;

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    swh[i__ + *ibeg - 1 + k] = hsw[i__ + k - 2] *
		    1.1574000382097438e-5f;

	}
    }
L350:
    return 0;
}

#undef alb33
#undef alb22
#undef alb11
#undef albd



















GD_DEVICE doublereal radtn_c_b6 = .28571429848670959;
GD_DEVICE integer radtn_c__1 = 1;
GD_DEVICE integer radtn_c__0 = 0;

 GD_DEVICE int radtn_(real *dt, real *t, real *q, real *qcw, real *qice,
	 real *pflip, real *p8wflip, real *xland, real *tsk2d, real *glat,
	real *glon, real *cutop, real *cubot, real *alb, real *cuppt, real *
	acfrcv, integer *ncfrcv, real *acfrst, integer *ncfrst, real *vegfrc,
	real *sno, real *glw, real *gsw, real *rswin, real *rswinc, integer *
	idat, integer *ihrst, real *xtime, real *julian, integer *nrads,
	integer *nradl, integer *nphs, integer *ntsd, real *tends, real *
	tendl, real *cldfra, real *rswtoa, real *rlwtoa, real *czmean, real *
	cfracl, real *cfracm, real *cfrach, integer *ids, integer *ide,
	integer *jds, integer *jde, integer *kds, integer *kde, integer *ims,
	integer *ime, integer *jms, integer *jme, integer *kms, integer *kme,
	integer *its, integer *ite, integer *jts, integer *jte, integer *kts,
	integer *kte, GDTable *gt, GDClock *clock, GDScratch *work)
{


    logical cncld = TRUE_;
    real cthk[3] = { 2e4f,2e4f,2e4f };
    real cc[10] = { .1f,.2f,.3f,.4f,.5f,.6f,.7f,.8f,.9f,1.f };
    real ppt[10] = { 0.f,.14f,.31f,.7f,1.6f,3.4f,7.7f,17.f,38.f,85.f };
    real abcff[12] = { 4e-5f,4e-5f,.002f,.035f,.377f,1.95f,9.4f,44.6f,
	    190.f,989.f,2706.f,39011.f };


    integer i__1, i__2, i__3, i__4, i__5;
    real r__1, r__2, r__3;
    doublereal d__1;






    integer i__, j, l, n;
    logical new_cloud__;
    real p1, p2, convprate, aa, bb, dd, ee, ff, gg;
    integer ii, nc, lm, ll;
    real wv, cc1, cc2, cl1, cl2, cr1;
    integer lm1, lp1;
    real arg;
    integer imd;
    doublereal ddx, eex;

    integer *lmh=work->radtn_lmh;
    integer *lvl=work->radtn_lvl;

    real *ozn=work->radtn_ozn;
    real *tot=work->radtn_tot;
    real sdm;
    integer lml, jmd;
    logical bit1, bit2;
    integer kbt2, kbt1, kth1, kth2;
    real prs1, prs2, addl;

    integer *kcld=work->radtn_kcld;

    real *psfc=work->radtn_psfc;
    real *pmid=work->radtn_pmid;
    real *tmid=work->radtn_tmid;
    real *qmid=work->radtn_qmid;
    logical long__;

    real *czen=work->radtn_czen;
    doublereal prod;
    logical bitw, bitx, bity, bitz;

    integer *mbot=work->radtn_mbot;
    integer *mtop=work->radtn_mtop;

    real *tskn=work->radtn_tskn;
    real *xlat=work->radtn_xlat;
    real *cosz=work->radtn_cosz;
    real *pint=work->radtn_pint;
    real *emis=work->radtn_emis;
    real *pozn=work->radtn_pozn;
    real *camt=work->radtn_camt;

    integer *kbtm=work->radtn_kbtm;
    integer *ktop=work->radtn_ktop;

    real *rrcl=work->radtn_rrcl;
    real *ttcl=work->radtn_ttcl;
    real time;
    real dayi;
    real hour;
    real rang;
    real apes;
    real pmod;
    real ctau;
    real clfr1;
    real dpcl;
    real qsum;
    real delp;
    real tcld;
    real fctr;
    real clfr;
    real esat;
    real qsat;
    real rhum;
    real qcld;
    real tauc;
    integer myjs, myje, myis, myie, nmod, nlvl, ktop1, ncld, nktp, nbtm,
	    myjs1, myjs2, myje2;

    real *ccmid=work->radtn_ccmid;
    integer myje1, ixsd, nband, jndx;
    extern GD_DEVICE real fpvs_(real *, GDTable *gt, GDClock *clock, GDScratch *work);

    real *csmid=work->radtn_csmid;
    real *thmid=work->radtn_thmid;
    real *fswdn=work->radtn_fswdn;
    real *slmsk=work->radtn_slmsk;
    logical short__;

    real *flwup=work->radtn_flwup;

    integer *nclds=work->radtn_nclds;

    real *qwmid=work->radtn_qwmid;
    real *fswup=work->radtn_fswup;
    real *qimid=work->radtn_qimid;
    real clstp;
    real times;
    real exner;
    real snomm;
    real rhtot;
    integer ntsph, lvlij, lcnvb, lcnvt, malvl, lltop, llbot;
    extern GD_DEVICE int radfs_(real *, real *, real *, real *, real *,
	     real *, real *, real *, real *, real *, real *, integer *,
	    integer *, integer *, real *, real *, real *, real *, real *,
	    integer *, integer *, integer *, integer *, integer *, real *,
	    real *, real *, real *, real *, real *, real *, real *, real *,
	    real *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, integer *, GDTable *gt, GDClock *clock, GDScratch *work), ozon2d_(integer *,
	    real *, real *, real *, integer *, integer *, integer *, integer *
	    , integer *, integer *, integer *, integer *, integer *, integer *
	    , integer *, integer *, integer *, integer *, integer *, integer *
	    , integer *, integer *, integer *, integer *, GDTable *gt, GDClock *clock, GDScratch *work);

    real *albedo=work->radtn_albedo;
    real *cldcfr=work->radtn_cldcfr;
    real cfcmax;
    real *cldamt=work->radtn_cldamt;
    real snofac;
    real cldmax;
    real cfravg;
    real *taudar=work->radtn_taudar;
    real cfsmax;
    integer nradpp;

    real *flwdns=work->radtn_flwdns;
    extern GD_DEVICE int zenith_(real *, real *, real *, integer *,
	    integer *, real *, real *, real *, integer *, integer *, integer *
	    , integer *, integer *, integer *, integer *, integer *, integer *
	    , integer *, integer *, integer *, integer *, integer *, integer *
	    , integer *, integer *, integer *, integer *, integer *, integer *
	    , integer *, GDTable *gt, GDClock *clock, GDScratch *work);

    real *fswdns=work->radtn_fswdns;
    integer itimlw, itimsw;

    real *flwups=work->radtn_flwups;
    integer kntlyr;

    real *fswups=work->radtn_fswups;
    real *fswdnsc=work->radtn_fswdnsc;


    cfrach -= 2;
    cfracm -= 2;
    cfracl -= 2;
    czmean -= 2;
    rlwtoa -= 2;
    rswtoa -= 2;
    cldfra -= 84;
    tendl -= 83;
    tends -= 83;
    --idat;
    rswinc -= 2;
    rswin -= 2;
    gsw -= 2;
    glw -= 2;
    sno -= 2;
    vegfrc -= 2;
    ncfrst -= 2;
    acfrst -= 2;
    ncfrcv -= 2;
    acfrcv -= 2;
    cuppt -= 2;
    alb -= 2;
    cubot -= 2;
    cutop -= 2;
    glon -= 2;
    glat -= 2;
    tsk2d -= 2;
    xland -= 2;
    p8wflip -= 84;
    pflip -= 84;
    qice -= 84;
    qcw -= 84;
    q -= 84;
    t -= 84;


    imd = (*ims + *ime) / 2;
    jmd = (*jms + *jme) / 2;
    myjs = *jts;
    myje = *jte;
    myis = *its;
    myie = *ite;
    myjs1 = *jts;
    myje1 = *jte;
    myjs2 = *jts;
    myje2 = *jte;
    lm = *kte;
    lm1 = lm - 1;
    lp1 = lm + 1;
    i__1 = *jte;
    for (j = *jts; j <= i__1; ++j) {
	i__2 = *ite;
	for (i__ = *its; i__ <= i__2; ++i__) {
	    lmh[i__ + j - 2] = *kme - 1;
	    lvl[i__ + j - 2] = 0;
	}
    }
    r__1 = 3600.f / *dt;
    ntsph = i_nint(&r__1);
    nradpp = min(*nrads,*nradl);
    clstp = nradpp * 1.f / ntsph;
    convprate = 2.4e4f / clstp;
    short__ = TRUE_;
    long__ = TRUE_;
    itimsw = 0;
    itimlw = 0;
    if (short__) {
	itimsw = 1;
    }
    if (long__) {
	itimlw = 1;
    }
    time = *xtime * 60.f;
    zenith_(&time, &dayi, &hour, &idat[1], ihrst, &glon[2], &glat[2], czen, &
	    myis, &myie, &myjs, &myje, ids, ide, jds, jde, kds, kde, ims, ime,
	     jms, jme, kms, kme, its, ite, jts, jte, kts, kte, gt, clock, work);
    addl = 0.f;
    if (idat[3] % 4 == 0) {
	addl = 1.f;
    }
    rang = (dayi - 14.8125f) * 6.2831850051879883f / (addl + 365.f);
    gd_clock__1.rsin1 = (real)gd_sin(rang);
    gd_clock__1.rcos1 = (real)gd_cos(rang);
    gd_clock__1.rcos2 = (real)gd_cos((real)(rang * 2.f));
    if (short__) {
	i__1 = myje;
	for (j = myjs; j <= i__1; ++j) {
	    i__2 = myie;
	    for (i__ = myis; i__ <= i__2; ++i__) {
		czmean[i__ + j] = 0.f;
		tot[i__ + j - 2] = 0.f;
	    }
	}
	i__1 = *nrads;
	i__2 = *nphs;
	for (ii = 0; i__2 < 0 ? ii >= i__1 : ii <= i__1; ii += i__2) {
	    times = *xtime * 60.f + ii * *dt;
	    zenith_(&times, &dayi, &hour, &idat[1], ihrst, &glon[2], &glat[2],
		     czen, &myis, &myie, &myjs, &myje, ids, ide, jds, jde,
		    kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts,
		    jte, kts, kte, gt, clock, work);
	    i__3 = myje;
	    for (j = myjs; j <= i__3; ++j) {
		i__4 = myie;
		for (i__ = myis; i__ <= i__4; ++i__) {
		    if (czen[i__ + j - 2] > 0.f) {
			czmean[i__ + j] += czen[i__ + j - 2];
			tot[i__ + j - 2] += 1.f;
		    }
		}
	    }
	}
	i__2 = myje;
	for (j = myjs; j <= i__2; ++j) {
	    i__1 = myie;
	    for (i__ = myis; i__ <= i__1; ++i__) {
		if (tot[i__ + j - 2] > 0.f) {
		    czmean[i__ + j] /= tot[i__ + j - 2];
		}
	    }
	}
    }
    i__2 = lm;
    for (l = 2; l <= i__2; ++l) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    pozn[i__ + l - 2] = 1.f;
	}
    }
    i__2 = myje;
    for (j = myjs; j <= i__2; ++j) {
	i__1 = lm;
	for (l = 1; l <= i__1; ++l) {
	    i__3 = myie;
	    for (i__ = myis; i__ <= i__3; ++i__) {
		tmid[i__ + l - 2] = t[i__ + (j * 82 + 1)];
		qmid[i__ + l - 2] = 9.999999960041972e-13f;
		qwmid[i__ + l - 2] = 0.f;
		qimid[i__ + l - 2] = 0.f;
		csmid[i__ + l - 2] = 0.f;
		ccmid[i__ + l - 2] = 0.f;
		ozn[i__ + l - 2] = 1.000000013351432e-10f;
		tends[i__ + (l + j * 81)] = 0.f;
		tendl[i__ + (l + j * 81)] = 0.f;
	    }

	}
	for (n = 1; n <= 3; ++n) {
	    i__1 = myie;
	    for (i__ = myis; i__ <= i__1; ++i__) {
		cldcfr[i__ + n - 2] = 0.f;
		mtop[i__ + n - 2] = 0;
		mbot[i__ + n - 2] = 0;
	    }

	}
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    lml = lmh[i__ + j - 2];
	    lvlij = lvl[i__ + j - 2];
	    i__3 = lml;
	    for (l = 1; l <= i__3; ++l) {
		pmid[i__ + (l + lvlij) - 2] = pflip[i__ + (l + j * 82)];
		pint[i__ + (l + lvlij + 1) - 2] = p8wflip[i__ + (l + 1 + j *
			82)];
		d__1 = (doublereal) (1e5f / pmid[i__ + (l + lvlij) - 2]);
		exner = (real)pow_dd(&d__1, &radtn_c_b6);
		tmid[i__ + (l + lvlij) - 2] = t[i__ + (l + j * 82)];
		thmid[i__ + (l + lvlij) - 2] = t[i__ + (l + j * 82)] * exner;

		r__1 = 9.999999960041972e-13f, r__2 = q[i__ + (l + j * 82)];
		qmid[i__ + (l + lvlij) - 2] = max(r__1,r__2);
		qwmid[i__ + (l + lvlij) - 2] = qcw[i__ + (l + j * 82)];
		qimid[i__ + (l + lvlij) - 2] = qice[i__ + (l + j * 82)];
	    }
	    if (lvlij > 0) {
		kntlyr = 0;
		for (l = lvlij; l >= 1; --l) {
		    ++kntlyr;
		    pmid[i__ + l - 2] = p8wflip[i__ + (j * 82 + 1)] - (real) (
			    (kntlyr << 1) - 1) * 5.f;
		    pint[i__ + (l + 1) - 2] = pmid[i__ + l - 2] + 5.f;
		    d__1 = (doublereal) (1e5f / pmid[i__ + l - 2]);
		    exner = (real)pow_dd(&d__1, &radtn_c_b6);
		    thmid[i__ + l - 2] = tmid[i__ + l - 2] * exner;
		}
	    }
	    if (lvlij == 0) {
		pint[i__ - 1] = p8wflip[i__ + (j * 82 + 1)];
	    } else {
		pint[i__ - 1] = pmid[i__ - 1] - 5.f;
	    }

	}
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    psfc[i__ - 1] = p8wflip[i__ + (*kme + j * 82)];
	    d__1 = (doublereal) (psfc[i__ - 1] * 1e-5f);
	    apes = (real)pow_dd(&d__1, &radtn_c_b6);
	    if (xland[i__ + j] - 1.5f > 0.f) {
		tskn[i__ - 1] = -tsk2d[i__ + j];
	    } else {
		tskn[i__ - 1] = tsk2d[i__ + j];
	    }
	    slmsk[i__ - 1] = xland[i__ + j] - 1.f;

	    r__1 = sno[i__ + j];
	    snomm = max(r__1,0.f);

	    r__1 = snomm / .02f;
	    snofac = min(r__1,1.f);
	    albedo[i__ - 1] = alb[i__ + j];
	    xlat[i__ - 1] = glat[i__ + j] * 57.295780181884766f;
	    cosz[i__ - 1] = czmean[i__ + j];

	}
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    lml = lmh[i__ + j - 2];
	    lvlij = lvl[i__ + j - 2];
	    i__3 = lml;
	    for (l = 1; l <= i__3; ++l) {
		ll = l + lvlij;
		wv = qmid[i__ + ll - 2] / (1.f - qmid[i__ + ll - 2]);
		qcld = qwmid[i__ + ll - 2] + qimid[i__ + ll - 2];
		if (qcld <= 9.999999960041972e-13f) {
		    goto L255;
		}
		clfr = 0.f;
		wv = qmid[i__ + ll - 2] / (1.f - qmid[i__ + ll - 2]);
		esat = (real)fpvs_(&tmid[i__ + ll - 2], gt, clock, work) * 1e3f;
		qsat = esat * .62175041437149048f / (pmid[i__ + ll - 2] -
			esat);
		rhum = wv / qsat;
		rhtot = (wv + qcld) / qsat;
		lcnvt = i_nint(&cutop[i__ + j]) + lvlij;
		lcnvt = min(lm,lcnvt);
		lcnvb = i_nint(&cubot[i__ + j]) + lvlij;
		lcnvb = min(lm,lcnvb);
		if (ll >= lcnvt && ll <= lcnvb) {
		    sdm = .039999999105930328f;
		} else {
		    sdm = .0099999997764825821f;
		}
		arg = (rhtot - 1.f) / sdm;
		if (arg <= .004999999888241291f && arg >=
			-.004999999888241291f) {
		    clfr = .5f;
		} else if (arg > .004999999888241291f) {
		    if (arg >= 3.0999999046325684f) {
			clfr = 1.f;
		    } else {
			ixsd = (integer) (arg / .0099999997764825821f + .5f);

			i__4 = 310, i__5 = max(ixsd,1);
			ixsd = min(i__4,i__5);
			clfr = gd_tables__1.axsd[ixsd - 1] + .5f;
		    }
		} else {
		    if (arg <= -3.0999999046325684f) {
			clfr = 0.f;
		    } else {
			ixsd = (integer) (arg / -.0099999997764825821f + .5f);

			i__4 = 310, i__5 = max(ixsd,1);
			ixsd = min(i__4,i__5);
			clfr = .5f - gd_tables__1.axsd[ixsd - 1];
			if (clfr < .0099999997764825821f) {
			    clfr = 0.f;
			}
		    }
		}
		csmid[i__ + ll - 2] = clfr;
L255:
		;
	    }
	}
	if (cncld) {
	    i__1 = myie;
	    for (i__ = myis; i__ <= i__1; ++i__) {
		if (cubot[i__ + j] - cutop[i__ + j] > 1.f) {
		    clfr = cc[0];
		    pmod = cuppt[i__ + j] * convprate;
		    if (pmod > ppt[0]) {
			for (nc = 1; nc <= 10; ++nc) {
			    if (pmod > ppt[nc - 1]) {
				nmod = nc;
			    }
			}
			if (nmod >= 10) {
			    clfr = cc[9];
			} else {
			    cc1 = cc[nmod - 1];
			    cc2 = cc[nmod];
			    p1 = ppt[nmod - 1];
			    p2 = ppt[nmod];
			    clfr = cc1 + (cc2 - cc1) * (pmod - p1) / (p2 - p1)
				    ;
			}
			clfr = min(1.f,clfr);
		    }
		    lvlij = lvl[i__ + j - 2];
		    lcnvt = i_nint(&cutop[i__ + j]) + lvlij;
		    lcnvt = min(lm,lcnvt);
		    lcnvb = i_nint(&cubot[i__ + j]) + lvlij;
		    lcnvb = min(lm,lcnvb);
		    i__3 = lcnvb;
		    for (ll = lcnvt; ll <= i__3; ++ll) {

			r__1 = 0.f, r__2 = 1.f - csmid[i__ + ll - 2];
			arg = max(r__1,r__2);
			ccmid[i__ + ll - 2] = min(arg,clfr);
		    }
		}
	    }
	}
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    i__3 = lm;
	    for (l = 0; l <= i__3; ++l) {
		cldamt[l] = 0.f;
	    }
	    for (nlvl = 1; nlvl <= 3; ++nlvl) {
		cldmax = 0.f;
		malvl = lm;
		lltop = lm + 1 - gd_tables__1.ltop[nlvl - 1] + lvl[i__ + j -
			2];
		if (lltop >= lm) {
		    goto L480;
		}
		if (nlvl > 1) {
		    llbot = lm + 1 - gd_tables__1.ltop[nlvl - 2] - 1 + lvl[
			    i__ + j - 2];
		    llbot = min(llbot,lm1);
		} else {
		    llbot = lm1;
		}
		i__3 = llbot;
		for (l = lltop; l <= i__3; ++l) {

		    r__1 = csmid[i__ + l - 2], r__2 = ccmid[i__ + l - 2];
		    cldamt[l] = max(r__1,r__2);
		    if (cldamt[l] > cldmax) {
			malvl = l;
			cldmax = cldamt[l];
		    }

		}
		cl1 = 0.f;
		cl2 = 0.f;
		kbt1 = llbot;
		kbt2 = llbot;
		kth1 = 0;
		kth2 = 0;
		i__3 = llbot;
		for (ll = lltop; ll <= i__3; ++ll) {
		    l = llbot - ll + lltop;
		    bit1 = FALSE_;
		    cr1 = cldamt[l];
		    bitx = pint[i__ + l - 2] >= gd_tables__1.ptopc[nlvl] &&
			    pint[i__ + l - 2] < gd_tables__1.ptopc[nlvl - 1]
			    && cldamt[l] > 0.f;
		    bit1 = bit1 || bitx;
		    if (! bit1) {
			goto L450;
		    }
		    bity = bitx && kth2 <= 0;
		    bitz = bitx && kth2 > 0;
		    if (bity) {
			kbt2 = l;
			kth2 = 1;
		    }
		    if (bitz) {
			ktop1 = kbt2 - kth2 + 1;
			dpcl = pmid[i__ + kbt2 - 2] - pmid[i__ + ktop1 - 2];
			if (dpcl < cthk[nlvl - 1]) {
			    ++kth2;
			} else {
			    --kbt2;
			}
		    }
		    if (bitx) {
			cl2 = max(cl2,cr1);
		    }
		    bit2 = FALSE_;
		    bity = bitx && (cldamt[l - 1] <= 0.f || pint[i__ + (l - 1)
			     - 2] < gd_tables__1.ptopc[nlvl]);
		    bitz = bity && cl1 > 0.f;
		    bitw = bity && cl1 <= 0.f;
		    bit2 = bit2 || bity;
		    if (! bit2) {
			goto L450;
		    }
		    if (bitz) {
			kbt1 = (integer) ((cl1 * kbt1 + cl2 * kbt2) / (cl1 +
				cl2));
			kth1 = (integer) ((cl1 * kth1 + cl2 * kth2) / (cl1 +
				cl2)) + 1;
			cl1 = cl1 + cl2 - cl1 * cl2;
		    }
		    if (bitw) {
			kbt1 = kbt2;
			kth1 = kth2;
			cl1 = cl2;
		    }
		    if (bity) {
			kbt2 = llbot;
			kth2 = 0;
			cl2 = 0.f;
		    }
L450:
		    ;
		}
		cldcfr[i__ + nlvl - 2] = min(1.f,cl1);

		i__3 = kbt1, i__4 = kbt1 - kth1 + 1;
		mtop[i__ + nlvl - 2] = min(i__3,i__4);
		mbot[i__ + nlvl - 2] = kbt1;
L480:
		;
	    }

	}
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    taudar[i__ - 1] = 1.f;
	}
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    lml = lmh[i__ + j - 2];
	    lvlij = lvl[i__ + j - 2];
	    emis[i__ - 1] = 1.f;
	    ktop[i__ - 1] = lp1;
	    kbtm[i__ - 1] = lp1;
	    camt[i__ - 1] = 1.f;
	    kcld[i__ - 1] = 2;
	    for (nband = 1; nband <= 12; ++nband) {
		rrcl[i__ + (nband + 12) - 14] = 0.f;
		ttcl[i__ + (nband + 12) - 14] = 1.f;
	    }
	    i__3 = lp1;
	    for (l = 2; l <= i__3; ++l) {
		camt[i__ + l - 2] = 0.f;
		ktop[i__ + l - 2] = 1;
		kbtm[i__ + l - 2] = 1;
		emis[i__ + l - 2] = 0.f;
		for (nband = 1; nband <= 12; ++nband) {
		    rrcl[i__ + (nband + l * 12) - 14] = 0.f;
		    ttcl[i__ + (nband + l * 12) - 14] = 1.f;
		}

	    }
	    new_cloud__ = TRUE_;
	    i__3 = lml;
	    for (l = 2; l <= i__3; ++l) {
		ll = lml - l + 1 + lvlij;

		r__1 = ccmid[i__ + ll - 2], r__2 = csmid[i__ + ll - 2];
		clfr = max(r__1,r__2);

		r__1 = ccmid[i__ + (ll + 1) - 2], r__2 = csmid[i__ + (ll + 1)
			- 2];
		clfr1 = max(r__1,r__2);
		if (clfr >= .0099999997764825821f) {
		    if (new_cloud__) {
			if (l == 2 && clfr1 >= .0099999997764825821f) {
			    kbtm[i__ + kcld[i__ - 1] - 2] = ll + 1;
			    camt[i__ + kcld[i__ - 1] - 2] = clfr1;
			} else {
			    kbtm[i__ + kcld[i__ - 1] - 2] = ll;
			    camt[i__ + kcld[i__ - 1] - 2] = clfr;
			}
			new_cloud__ = FALSE_;
		    } else {

			r__1 = camt[i__ + kcld[i__ - 1] - 2];
			camt[i__ + kcld[i__ - 1] - 2] = max(r__1,clfr);
		    }
		} else if (clfr1 >= .0099999997764825821f) {
		    if (l == 2) {
			kbtm[i__ + kcld[i__ - 1] - 2] = ll + 1;
			camt[i__ + kcld[i__ - 1] - 2] = clfr1;
		    }
		    ktop[i__ + kcld[i__ - 1] - 2] = ll + 1;
		    new_cloud__ = TRUE_;
		    ++kcld[i__ - 1];
		    camt[i__ + kcld[i__ - 1] - 2] = 0.f;
		}
	    }
	    nclds[i__ - 1] = kcld[i__ - 1] - 2;
	    ncld = nclds[i__ - 1];
	    if (ncld >= 1) {
		i__3 = ncld + 1;
		for (nc = 2; nc <= i__3; ++nc) {
		    tauc = 0.f;
		    qsum = 0.f;
		    nktp = lp1;
		    nbtm = 0;
		    bitx = camt[i__ + nc - 2] >= .0099999997764825821f;

		    i__4 = nktp, i__5 = ktop[i__ + nc - 2];
		    nktp = min(i__4,i__5);

		    i__4 = nbtm, i__5 = kbtm[i__ + nc - 2];
		    nbtm = max(i__4,i__5);
		    i__4 = nbtm;
		    for (ll = nktp; ll <= i__4; ++ll) {
			if (ll >= ktop[i__ + nc - 2] && ll <= kbtm[i__ + nc -
				2] && bitx) {
			    prs1 = pint[i__ + ll - 2] * .01f;
			    prs2 = pint[i__ + (ll + 1) - 2] * .01f;
			    delp = prs2 - prs1;
			    tcld = tmid[i__ + ll - 2] - 273.14999389648438f;
			    r__1 = qmid[i__ + ll - 2] * delp;
			    qsum += r__1 * (prs1 + prs2) / ((real)gd_sqrt(tmid[
				    i__ + ll - 2]) * 120.1612f);
			    ctau = 0.f;
			    if (ccmid[i__ + ll - 2] >= .0099999997764825821f)
				    {
				if (tcld >= -20.f) {
				    ctau = .079999998211860657f;
				} else {
				    ctau = .049999997019767761f;
				}
			    }
			    r__1 = ctau + qwmid[i__ + ll - 2] * 800.f;
			    ctau = r__1 + qimid[i__ + ll - 2] * 500.f;
			    tauc += delp * ctau;
			}
		    }
		    if (bitx) {
			emis[i__ + nc - 2] = 1.f - (real)gd_exp((real)(tauc *
				-2.4900000095367432f));
		    }
		    if (qsum >= 9.9999997473787516e-6f) {
			for (nband = 1; nband <= 12; ++nband) {
			    if (bitx) {
				prod = abcff[nband - 1] * qsum;
				ddx = tauc / (tauc + prod);
				eex = 1.f - ddx;
				if (abs(eex) >= 1e-8f) {
				    dd = ddx;
				    ee = eex;
				    ff = 1.f - dd * .85f;

				    r__3 = ee * 3.f;
				    r__1 = 50.f, r__2 = (real)gd_sqrt((real)(
					    r__3 * ff)) * tauc;
				    aa = min(r__1,r__2);
				    aa = (real)gd_exp((real)(-aa));
				    bb = ff / ee;
				    gg = (real)gd_sqrt(bb);
				    r__2 = (gg - 1.f) * (gg - 1.f);
				    r__1 = r__2 * aa;
				    dd = (gg + 1.f) * (gg + 1.f) - r__1 * aa;

				    r__1 = 1e-6f, r__2 = (bb - 1.f) * (1.f -
					    aa * aa) / dd;
				    rrcl[i__ + (nband + nc * 12) - 14] = max(
					    r__1,r__2);

				    r__3 = gg * 4.f;
				    r__1 = 1e-6f, r__2 = r__3 * aa / dd;
				    ttcl[i__ + (nband + nc * 12) - 14] = max(
					    r__1,r__2);
				}
			    }

			}
		    }

		}
	    }

	}
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    fctr = pint[i__] / (pint[i__] - pint[i__ - 1]);
	    pozn[i__ - 1] = fctr * (pmid[i__ - 1] - pint[i__ - 1]);
	}
	ozon2d_(&lm, pozn, xlat, ozn, &myis, &myie, ids, ide, jds, jde, kds,
		kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts,
		kte, gt, clock, work);
	jndx = j;
	radfs_(psfc, pmid, pint, qmid, tmid, ozn, tskn, slmsk, albedo, xlat,
		camt, ktop, kbtm, nclds, emis, rrcl, ttcl, cosz, taudar, &
		radtn_c__1, &radtn_c__1, &radtn_c__0, &itimsw, &itimlw, &tends[*its + (*kts + j
		* 81)], &tendl[*its + (*kts + j * 81)], flwup, fswup, fswdn,
		fswdns, fswups, flwdns, flwups, fswdnsc, ids, ide, jds, jde,
		kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte,
		kts, kte, &imd, &jmd, &jndx, gt, clock, work);
	if (long__) {
	    i__1 = myie;
	    for (i__ = myis; i__ <= i__1; ++i__) {
		glw[i__ + j] = flwdns[i__ - 1];
		rlwtoa[i__ + j] = flwup[i__ - 1];
	    }
	}
	if (short__) {
	    i__1 = myie;
	    for (i__ = myis; i__ <= i__1; ++i__) {
		gsw[i__ + j] = fswdns[i__ - 1] - fswups[i__ - 1];
		rswin[i__ + j] = fswdns[i__ - 1];
		rswinc[i__ + j] = fswdnsc[i__ - 1];
		rswtoa[i__ + j] = fswup[i__ - 1];
	    }
	}
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    cfracl[i__ + j] = cldcfr[i__ - 1];
	    cfracm[i__ + j] = cldcfr[i__];
	    cfrach[i__ + j] = cldcfr[i__ + 1];
	    if (cncld) {
		cfsmax = 0.f;
		cfcmax = 0.f;
		i__3 = lmh[i__ + j - 2];
		for (l = 1; l <= i__3; ++l) {
		    ll = l + lvl[i__ + j - 2];

		    r__1 = cfsmax, r__2 = csmid[i__ + ll - 2];
		    cfsmax = max(r__1,r__2);

		    r__1 = cfcmax, r__2 = ccmid[i__ + ll - 2];
		    cfcmax = max(r__1,r__2);
		}
		acfrst[i__ + j] += cfsmax;
		++ncfrst[i__ + j];
		acfrcv[i__ + j] += cfcmax;
		++ncfrcv[i__ + j];
	    } else {
		r__1 = (1.f - cfracl[i__ + j]) * (1.f - cfracm[i__ + j]);
		cfravg = 1.f - r__1 * (1.f - cfrach[i__ + j]);
		acfrst[i__ + j] += cfravg;
		++ncfrst[i__ + j];
	    }
	    lml = lmh[i__ + j - 2];
	    i__3 = lml;
	    for (l = 1; l <= i__3; ++l) {
		ll = lml - l + 1 + lvl[i__ + j - 2];

		r__1 = ccmid[i__ + ll - 2], r__2 = csmid[i__ + ll - 2];
		cldfra[i__ + (l + j * 82)] = max(r__1,r__2);
	    }
	}

    }
    return 0;
}
















 GD_DEVICE int solard_(integer *ihrst, integer *iday, integer *month,
	integer *julyr, GDTable *gt, GDClock *clock, GDScratch *work)
{


    integer jmn = 0;
    real ccr = 1.3e-6f;
    real tpp = 1.55f;
    integer jdor2 = 2415020;


    real r__1;





    real e, t, ec;
    integer jd;
    real em, ep, cr, fjd, dat;
    integer jhr;
    real fjd1, date, year;
    integer iter;
    real dayinc;

    jhr = *ihrst;
    jd = *iday - 32075 + (*julyr + 4800 + (*month - 14) / 12) * 1461 / 4 + (*
	    month - 2 - (*month - 14) / 12 * 12) * 367 / 12 - (*julyr + 4900
	    + (*month - 14) / 12) / 100 * 3 / 4;
    if (jhr < 12) {
	--jd;
	r__1 = (real) jhr * .041666667f + .5f;
	fjd = r__1 + (real) jmn * 6.9444444e-4f;
    } else {

	fjd = (real) (jhr - 12) * .041666667f + (real) jmn * 6.9444444e-4f;
    }
    dayinc = jhr / 24.f;
    r__1 = jd + fjd;
    fjd1 = r__1 + dayinc;
    jd = fjd1;
    fjd = fjd1 - jd;
    dat = (real) (jd - jdor2) - tpp + fjd;
    t = (real) (jd - jdor2) / 36525.f;
    year = t * 3.04e-6f + .25964134f;
    ec = .01675104f - (t * 1.26e-7f + 4.18e-5f) * t;
    year += 365.f;
    date = (real)r_mod(&dat, &year);
    em = date * 6.2831850051879883f / year;
    e = 1.f;
    iter = 0;
L31:
    ep = e - (e - ec * (real)gd_sin(e) - em) / (1.f - ec * (real)gd_cos(e));
    cr = (r__1 = e - ep, abs(r__1));
    e = ep;
    ++iter;
    if (iter > 10) {
	goto L1031;
    }
    if (cr > ccr) {
	goto L31;
    }
L1031:
    gd_clock__1.r1 = 1.f - ec * (real)gd_cos(e);
    return 0;
}
















 GD_DEVICE int spa88_(real *excts, real *ctso3, real *gxcts, real *sorc,
	 real *csour, real *cldfac, real *temp, real *press, real *var1, real
	*var2, real *p, real *delp, real *delp2, real *totvo2, real *to3sp,
	real *to3spc, real *co2sp1, real *co2sp2, real *co2sp, real *apcm,
	real *bpcm, real *atpcm, real *btpcm, real *acomb, real *bcomb, real *
	betacm, real *h25e2, real *one, real *h44194m2, real *h1p41819, real *
	haf, real *hm1ez, real *two, real *radcon, integer *ids, integer *ide,
	 integer *jds, integer *jde, integer *kds, integer *kde, integer *ims,
	 integer *ime, integer *jms, integer *jme, integer *kms, integer *kme,
	 integer *its, integer *ite, integer *jts, integer *jte, integer *kts,
	 integer *kte, GDTable *gt, GDClock *clock, GDScratch *work)
{

    integer i__1, i__2;
    real r__1, r__2, r__3;






    real *f=work->spa88_f;
    integer i__, k, l;

    real *x=work->spa88_x;
    real *y=work->spa88_y;
    real *ag=work->spa88_ag;
    real *ff=work->spa88_ff;
    integer ll;

    real *tt=work->spa88_tt;
    integer lm1, lm2, lm3, lp1, lp2, lp3;

    real *agg=work->spa88_agg;
    real *fac1=work->spa88_fac1;
    real *fac2=work->spa88_fac2;
    integer llm1, llm2, llp1;

    real *ctmp=work->spa88_ctmp;
    integer myie;

    real *topm=work->spa88_topm;
    integer myis;

    real *ctmp2=work->spa88_ctmp2;
    real *ctmp3=work->spa88_ctmp3;
    real *phitmp=work->spa88_phitmp;
    real *topphi=work->spa88_topphi;
    real *psitmp=work->spa88_psitmp;


    --betacm;
    --bcomb;
    --acomb;
    --btpcm;
    --atpcm;
    --bpcm;
    --apcm;
    co2sp -= 2;
    co2sp2 -= 2;
    co2sp1 -= 2;
    to3spc -= 2;
    to3sp -= 2;
    totvo2 -= 2;
    delp2 -= 2;
    delp -= 2;
    p -= 2;
    var2 -= 2;
    var1 -= 2;
    press -= 2;
    temp -= 2;
    cldfac -= 84;
    csour -= 2;
    sorc -= 84;
    --gxcts;
    ctso3 -= 2;
    excts -= 2;


    l = *kte;
    lp1 = l + 1;
    lp2 = l + 2;
    lp3 = l + 3;
    llp1 = (l << 1) + 1;
    lm1 = l - 1;
    lm2 = l - 2;
    lm3 = l - 3;
    ll = l << 1;
    llm2 = ll - 2;
    llm1 = ll - 1;
    myis = *its;
    myie = *ite;
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    x[i__ + k - 2] = temp[i__ + k] - *h25e2;
	    y[i__ + k - 2] = x[i__ + k - 2] * x[i__ + k - 2];

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	ctmp[i__ - 1] = *one;
	ctmp2[i__ - 1] = 1.f;
	ctmp3[i__ - 1] = 1.f;

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    f[i__ + k - 2] = *h44194m2 * (apcm[1] * x[i__ + k - 2] + bpcm[1] *
		     y[i__ + k - 2]);
	    ff[i__ + k - 2] = *h44194m2 * (atpcm[1] * x[i__ + k - 2] + btpcm[
		    1] * y[i__ + k - 2]);
	    ag[i__ + k - 2] = (*h1p41819 + f[i__ + k - 2]) * f[i__ + k - 2] +
		    *one;
	    agg[i__ + k - 2] = (*h1p41819 + ff[i__ + k - 2]) * ff[i__ + k - 2]
		     + *one;

	    r__3 = ag[i__ + k - 2] * ag[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    phitmp[i__ + k - 2] = var1[i__ + k] * (r__1 * r__1);

	    r__3 = agg[i__ + k - 2] * agg[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    psitmp[i__ + k - 2] = var2[i__ + k] * (r__1 * r__1);

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	topm[i__ - 1] = phitmp[i__ - 1];
	topphi[i__ - 1] = psitmp[i__ - 1];

    }
    i__1 = l;
    for (k = 2; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    topm[i__ + k - 2] = topm[i__ + (k - 1) - 2] + phitmp[i__ + k - 2];
	    topphi[i__ + k - 2] = topphi[i__ + (k - 1) - 2] + psitmp[i__ + k
		    - 2];

	}

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    fac1[i__ + k - 2] = acomb[1] * topm[i__ + k - 2];
	    fac2[i__ + k - 2] = fac1[i__ + k - 2] * topm[i__ + k - 2] / (
		    bcomb[1] * topphi[i__ + k - 2]);
	    tt[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * fac1[i__ + k - 2] / (
		    real)gd_sqrt((real)(fac2[i__ + k - 2] + 1.f))));
	    ctmp[i__ + (k + 1) - 2] = tt[i__ + k - 2] * cldfac[i__ + (k + 83)]
		    ;

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    excts[i__ + k] = sorc[i__ + (k + 82)] * (ctmp[i__ + (k + 1) - 2]
		    - ctmp[i__ + k - 2]);

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	r__2 = *haf * delp[i__ + l];
	r__1 = r__2 * (tt[i__ + lm1 - 2] * (p[i__ + lp1] - press[i__ + l]) +
		tt[i__ + l - 2] * (p[i__ + lp1] + press[i__ + l] - *two * p[
		i__ + l]));
	gxcts[i__] = cldfac[i__ + (lp1 + 82)] * (tt[i__ + l - 2] * sorc[i__ +
		(l + 82)] + r__1 * (sorc[i__ + (lp1 + 82)] - sorc[i__ + (l +
		82)]));

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    f[i__ + k - 2] = *h44194m2 * (apcm[2] * x[i__ + k - 2] + bpcm[2] *
		     y[i__ + k - 2]);
	    ff[i__ + k - 2] = *h44194m2 * (atpcm[2] * x[i__ + k - 2] + btpcm[
		    2] * y[i__ + k - 2]);
	    ag[i__ + k - 2] = (*h1p41819 + f[i__ + k - 2]) * f[i__ + k - 2] +
		    *one;
	    agg[i__ + k - 2] = (*h1p41819 + ff[i__ + k - 2]) * ff[i__ + k - 2]
		     + *one;

	    r__3 = ag[i__ + k - 2] * ag[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    phitmp[i__ + k - 2] = var1[i__ + k] * (r__1 * r__1);

	    r__3 = agg[i__ + k - 2] * agg[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    psitmp[i__ + k - 2] = var2[i__ + k] * (r__1 * r__1);

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	topm[i__ - 1] = phitmp[i__ - 1];
	topphi[i__ - 1] = psitmp[i__ - 1];

    }
    i__2 = l;
    for (k = 2; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    topm[i__ + k - 2] = topm[i__ + (k - 1) - 2] + phitmp[i__ + k - 2];
	    topphi[i__ + k - 2] = topphi[i__ + (k - 1) - 2] + psitmp[i__ + k
		    - 2];

	}

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    fac1[i__ + k - 2] = acomb[2] * topm[i__ + k - 2];
	    fac2[i__ + k - 2] = fac1[i__ + k - 2] * topm[i__ + k - 2] / (
		    bcomb[2] * topphi[i__ + k - 2]);
	    tt[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * fac1[i__ + k - 2] / (
		    real)gd_sqrt((real)(fac2[i__ + k - 2] + 1.f))));
	    ctmp[i__ + (k + 1) - 2] = tt[i__ + k - 2] * cldfac[i__ + (k + 83)]
		    ;

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    excts[i__ + k] += sorc[i__ + (k + 164)] * (ctmp[i__ + (k + 1) - 2]
		     - ctmp[i__ + k - 2]);

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	r__2 = *haf * delp[i__ + l];
	r__1 = r__2 * (tt[i__ + lm1 - 2] * (p[i__ + lp1] - press[i__ + l]) +
		tt[i__ + l - 2] * (p[i__ + lp1] + press[i__ + l] - *two * p[
		i__ + l]));
	gxcts[i__] += cldfac[i__ + (lp1 + 82)] * (tt[i__ + l - 2] * sorc[i__
		+ (l + 164)] + r__1 * (sorc[i__ + (lp1 + 164)] - sorc[i__ + (
		l + 164)]));

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    f[i__ + k - 2] = *h44194m2 * (apcm[3] * x[i__ + k - 2] + bpcm[3] *
		     y[i__ + k - 2]);
	    ff[i__ + k - 2] = *h44194m2 * (atpcm[3] * x[i__ + k - 2] + btpcm[
		    3] * y[i__ + k - 2]);
	    ag[i__ + k - 2] = (*h1p41819 + f[i__ + k - 2]) * f[i__ + k - 2] +
		    *one;
	    agg[i__ + k - 2] = (*h1p41819 + ff[i__ + k - 2]) * ff[i__ + k - 2]
		     + *one;

	    r__3 = ag[i__ + k - 2] * ag[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    phitmp[i__ + k - 2] = var1[i__ + k] * (r__1 * r__1);

	    r__3 = agg[i__ + k - 2] * agg[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    psitmp[i__ + k - 2] = var2[i__ + k] * (r__1 * r__1);

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	topm[i__ - 1] = phitmp[i__ - 1];
	topphi[i__ - 1] = psitmp[i__ - 1];

    }
    i__1 = l;
    for (k = 2; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    topm[i__ + k - 2] = topm[i__ + (k - 1) - 2] + phitmp[i__ + k - 2];
	    topphi[i__ + k - 2] = topphi[i__ + (k - 1) - 2] + psitmp[i__ + k
		    - 2];

	}

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    fac1[i__ + k - 2] = acomb[3] * topm[i__ + k - 2];
	    fac2[i__ + k - 2] = fac1[i__ + k - 2] * topm[i__ + k - 2] / (
		    bcomb[3] * topphi[i__ + k - 2]);
	    tt[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * fac1[i__ + k - 2] / (
		    real)gd_sqrt((real)(fac2[i__ + k - 2] + 1.f))));
	    ctmp[i__ + (k + 1) - 2] = tt[i__ + k - 2] * cldfac[i__ + (k + 83)]
		    ;

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    excts[i__ + k] += sorc[i__ + (k + 246)] * (ctmp[i__ + (k + 1) - 2]
		     - ctmp[i__ + k - 2]);

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	r__2 = *haf * delp[i__ + l];
	r__1 = r__2 * (tt[i__ + lm1 - 2] * (p[i__ + lp1] - press[i__ + l]) +
		tt[i__ + l - 2] * (p[i__ + lp1] + press[i__ + l] - *two * p[
		i__ + l]));
	gxcts[i__] += cldfac[i__ + (lp1 + 82)] * (tt[i__ + l - 2] * sorc[i__
		+ (l + 246)] + r__1 * (sorc[i__ + (lp1 + 246)] - sorc[i__ + (
		l + 246)]));

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    f[i__ + k - 2] = *h44194m2 * (apcm[4] * x[i__ + k - 2] + bpcm[4] *
		     y[i__ + k - 2]);
	    ff[i__ + k - 2] = *h44194m2 * (atpcm[4] * x[i__ + k - 2] + btpcm[
		    4] * y[i__ + k - 2]);
	    ag[i__ + k - 2] = (*h1p41819 + f[i__ + k - 2]) * f[i__ + k - 2] +
		    *one;
	    agg[i__ + k - 2] = (*h1p41819 + ff[i__ + k - 2]) * ff[i__ + k - 2]
		     + *one;

	    r__3 = ag[i__ + k - 2] * ag[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    phitmp[i__ + k - 2] = var1[i__ + k] * (r__1 * r__1);

	    r__3 = agg[i__ + k - 2] * agg[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    psitmp[i__ + k - 2] = var2[i__ + k] * (r__1 * r__1);

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	topm[i__ - 1] = phitmp[i__ - 1];
	topphi[i__ - 1] = psitmp[i__ - 1];

    }
    i__2 = l;
    for (k = 2; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    topm[i__ + k - 2] = topm[i__ + (k - 1) - 2] + phitmp[i__ + k - 2];
	    topphi[i__ + k - 2] = topphi[i__ + (k - 1) - 2] + psitmp[i__ + k
		    - 2];

	}

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    fac1[i__ + k - 2] = acomb[4] * topm[i__ + k - 2];
	    fac2[i__ + k - 2] = fac1[i__ + k - 2] * topm[i__ + k - 2] / (
		    bcomb[4] * topphi[i__ + k - 2]);
	    tt[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * fac1[i__ + k - 2] / (
		    real)gd_sqrt((real)(fac2[i__ + k - 2] + 1.f))));
	    ctmp[i__ + (k + 1) - 2] = tt[i__ + k - 2] * cldfac[i__ + (k + 83)]
		    ;

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    excts[i__ + k] += sorc[i__ + (k + 328)] * (ctmp[i__ + (k + 1) - 2]
		     - ctmp[i__ + k - 2]);

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	r__2 = *haf * delp[i__ + l];
	r__1 = r__2 * (tt[i__ + lm1 - 2] * (p[i__ + lp1] - press[i__ + l]) +
		tt[i__ + l - 2] * (p[i__ + lp1] + press[i__ + l] - *two * p[
		i__ + l]));
	gxcts[i__] += cldfac[i__ + (lp1 + 82)] * (tt[i__ + l - 2] * sorc[i__
		+ (l + 328)] + r__1 * (sorc[i__ + (lp1 + 328)] - sorc[i__ + (
		l + 328)]));

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    f[i__ + k - 2] = *h44194m2 * (apcm[5] * x[i__ + k - 2] + bpcm[5] *
		     y[i__ + k - 2]);
	    ff[i__ + k - 2] = *h44194m2 * (atpcm[5] * x[i__ + k - 2] + btpcm[
		    5] * y[i__ + k - 2]);
	    ag[i__ + k - 2] = (*h1p41819 + f[i__ + k - 2]) * f[i__ + k - 2] +
		    *one;
	    agg[i__ + k - 2] = (*h1p41819 + ff[i__ + k - 2]) * ff[i__ + k - 2]
		     + *one;

	    r__3 = ag[i__ + k - 2] * ag[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    phitmp[i__ + k - 2] = var1[i__ + k] * (r__1 * r__1);

	    r__3 = agg[i__ + k - 2] * agg[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    psitmp[i__ + k - 2] = var2[i__ + k] * (r__1 * r__1);

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	topm[i__ - 1] = phitmp[i__ - 1];
	topphi[i__ - 1] = psitmp[i__ - 1];

    }
    i__1 = l;
    for (k = 2; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    topm[i__ + k - 2] = topm[i__ + (k - 1) - 2] + phitmp[i__ + k - 2];
	    topphi[i__ + k - 2] = topphi[i__ + (k - 1) - 2] + psitmp[i__ + k
		    - 2];

	}

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    fac1[i__ + k - 2] = acomb[5] * topm[i__ + k - 2];
	    fac2[i__ + k - 2] = fac1[i__ + k - 2] * topm[i__ + k - 2] / (
		    bcomb[5] * topphi[i__ + k - 2]);
	    r__1 = betacm[5] * totvo2[i__ + (k + 1)];
	    tt[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * (fac1[i__ + k - 2] / (
		    real)gd_sqrt((real)(*one + fac2[i__ + k - 2])) + r__1 *
		    gd_tables__1.sko2d)));
	    ctmp[i__ + (k + 1) - 2] = tt[i__ + k - 2] * cldfac[i__ + (k + 83)]
		    ;

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    excts[i__ + k] += sorc[i__ + (k + 410)] * (ctmp[i__ + (k + 1) - 2]
		     - ctmp[i__ + k - 2]);

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	r__2 = *haf * delp[i__ + l];
	r__1 = r__2 * (tt[i__ + lm1 - 2] * (p[i__ + lp1] - press[i__ + l]) +
		tt[i__ + l - 2] * (p[i__ + lp1] + press[i__ + l] - *two * p[
		i__ + l]));
	gxcts[i__] += cldfac[i__ + (lp1 + 82)] * (tt[i__ + l - 2] * sorc[i__
		+ (l + 410)] + r__1 * (sorc[i__ + (lp1 + 410)] - sorc[i__ + (
		l + 410)]));

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    f[i__ + k - 2] = *h44194m2 * (apcm[6] * x[i__ + k - 2] + bpcm[6] *
		     y[i__ + k - 2]);
	    ff[i__ + k - 2] = *h44194m2 * (atpcm[6] * x[i__ + k - 2] + btpcm[
		    6] * y[i__ + k - 2]);
	    ag[i__ + k - 2] = (*h1p41819 + f[i__ + k - 2]) * f[i__ + k - 2] +
		    *one;
	    agg[i__ + k - 2] = (*h1p41819 + ff[i__ + k - 2]) * ff[i__ + k - 2]
		     + *one;

	    r__3 = ag[i__ + k - 2] * ag[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    phitmp[i__ + k - 2] = var1[i__ + k] * (r__1 * r__1);

	    r__3 = agg[i__ + k - 2] * agg[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    psitmp[i__ + k - 2] = var2[i__ + k] * (r__1 * r__1);

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	topm[i__ - 1] = phitmp[i__ - 1];
	topphi[i__ - 1] = psitmp[i__ - 1];

    }
    i__2 = l;
    for (k = 2; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    topm[i__ + k - 2] = topm[i__ + (k - 1) - 2] + phitmp[i__ + k - 2];
	    topphi[i__ + k - 2] = topphi[i__ + (k - 1) - 2] + psitmp[i__ + k
		    - 2];

	}

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    fac1[i__ + k - 2] = acomb[6] * topm[i__ + k - 2];
	    fac2[i__ + k - 2] = fac1[i__ + k - 2] * topm[i__ + k - 2] / (
		    bcomb[6] * topphi[i__ + k - 2]);
	    r__1 = betacm[6] * totvo2[i__ + (k + 1)];
	    tt[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * (fac1[i__ + k - 2] / (
		    real)gd_sqrt((real)(*one + fac2[i__ + k - 2])) + r__1 *
		    gd_tables__1.sko2d)));
	    ctmp[i__ + (k + 1) - 2] = tt[i__ + k - 2] * cldfac[i__ + (k + 83)]
		    ;

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    excts[i__ + k] += sorc[i__ + (k + 492)] * (ctmp[i__ + (k + 1) - 2]
		     - ctmp[i__ + k - 2]);

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	r__2 = *haf * delp[i__ + l];
	r__1 = r__2 * (tt[i__ + lm1 - 2] * (p[i__ + lp1] - press[i__ + l]) +
		tt[i__ + l - 2] * (p[i__ + lp1] + press[i__ + l] - *two * p[
		i__ + l]));
	gxcts[i__] += cldfac[i__ + (lp1 + 82)] * (tt[i__ + l - 2] * sorc[i__
		+ (l + 492)] + r__1 * (sorc[i__ + (lp1 + 492)] - sorc[i__ + (
		l + 492)]));

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    f[i__ + k - 2] = *h44194m2 * (apcm[7] * x[i__ + k - 2] + bpcm[7] *
		     y[i__ + k - 2]);
	    ff[i__ + k - 2] = *h44194m2 * (atpcm[7] * x[i__ + k - 2] + btpcm[
		    7] * y[i__ + k - 2]);
	    ag[i__ + k - 2] = (*h1p41819 + f[i__ + k - 2]) * f[i__ + k - 2] +
		    *one;
	    agg[i__ + k - 2] = (*h1p41819 + ff[i__ + k - 2]) * ff[i__ + k - 2]
		     + *one;

	    r__3 = ag[i__ + k - 2] * ag[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    phitmp[i__ + k - 2] = var1[i__ + k] * (r__1 * r__1);

	    r__3 = agg[i__ + k - 2] * agg[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    psitmp[i__ + k - 2] = var2[i__ + k] * (r__1 * r__1);

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	topm[i__ - 1] = phitmp[i__ - 1];
	topphi[i__ - 1] = psitmp[i__ - 1];

    }
    i__1 = l;
    for (k = 2; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    topm[i__ + k - 2] = topm[i__ + (k - 1) - 2] + phitmp[i__ + k - 2];
	    topphi[i__ + k - 2] = topphi[i__ + (k - 1) - 2] + psitmp[i__ + k
		    - 2];

	}

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    fac1[i__ + k - 2] = acomb[7] * topm[i__ + k - 2];
	    fac2[i__ + k - 2] = fac1[i__ + k - 2] * topm[i__ + k - 2] / (
		    bcomb[7] * topphi[i__ + k - 2]);
	    r__1 = betacm[7] * totvo2[i__ + (k + 1)];
	    tt[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * (fac1[i__ + k - 2] / (
		    real)gd_sqrt((real)(*one + fac2[i__ + k - 2])) + r__1 *
		    gd_tables__1.sko2d)));
	    ctmp[i__ + (k + 1) - 2] = tt[i__ + k - 2] * cldfac[i__ + (k + 83)]
		    ;

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    excts[i__ + k] += sorc[i__ + (k + 574)] * (ctmp[i__ + (k + 1) - 2]
		     - ctmp[i__ + k - 2]);

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	r__2 = *haf * delp[i__ + l];
	r__1 = r__2 * (tt[i__ + lm1 - 2] * (p[i__ + lp1] - press[i__ + l]) +
		tt[i__ + l - 2] * (p[i__ + lp1] + press[i__ + l] - *two * p[
		i__ + l]));
	gxcts[i__] += cldfac[i__ + (lp1 + 82)] * (tt[i__ + l - 2] * sorc[i__
		+ (l + 574)] + r__1 * (sorc[i__ + (lp1 + 574)] - sorc[i__ + (
		l + 574)]));

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    f[i__ + k - 2] = *h44194m2 * (apcm[8] * x[i__ + k - 2] + bpcm[8] *
		     y[i__ + k - 2]);
	    ff[i__ + k - 2] = *h44194m2 * (atpcm[8] * x[i__ + k - 2] + btpcm[
		    8] * y[i__ + k - 2]);
	    ag[i__ + k - 2] = (*h1p41819 + f[i__ + k - 2]) * f[i__ + k - 2] +
		    *one;
	    agg[i__ + k - 2] = (*h1p41819 + ff[i__ + k - 2]) * ff[i__ + k - 2]
		     + *one;

	    r__3 = ag[i__ + k - 2] * ag[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    phitmp[i__ + k - 2] = var1[i__ + k] * (r__1 * r__1);

	    r__3 = agg[i__ + k - 2] * agg[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    psitmp[i__ + k - 2] = var2[i__ + k] * (r__1 * r__1);

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	topm[i__ - 1] = phitmp[i__ - 1];
	topphi[i__ - 1] = psitmp[i__ - 1];

    }
    i__2 = l;
    for (k = 2; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    topm[i__ + k - 2] = topm[i__ + (k - 1) - 2] + phitmp[i__ + k - 2];
	    topphi[i__ + k - 2] = topphi[i__ + (k - 1) - 2] + psitmp[i__ + k
		    - 2];

	}

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    fac1[i__ + k - 2] = acomb[8] * topm[i__ + k - 2];
	    fac2[i__ + k - 2] = fac1[i__ + k - 2] * topm[i__ + k - 2] / (
		    bcomb[8] * topphi[i__ + k - 2]);
	    r__1 = betacm[8] * totvo2[i__ + (k + 1)];
	    tt[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * (fac1[i__ + k - 2] / (
		    real)gd_sqrt((real)(*one + fac2[i__ + k - 2])) + r__1 *
		    gd_tables__1.sko2d)));
	    ctmp[i__ + (k + 1) - 2] = tt[i__ + k - 2] * cldfac[i__ + (k + 83)]
		    ;

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    excts[i__ + k] += sorc[i__ + (k + 656)] * (ctmp[i__ + (k + 1) - 2]
		     - ctmp[i__ + k - 2]);

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	r__2 = *haf * delp[i__ + l];
	r__1 = r__2 * (tt[i__ + lm1 - 2] * (p[i__ + lp1] - press[i__ + l]) +
		tt[i__ + l - 2] * (p[i__ + lp1] + press[i__ + l] - *two * p[
		i__ + l]));
	gxcts[i__] += cldfac[i__ + (lp1 + 82)] * (tt[i__ + l - 2] * sorc[i__
		+ (l + 656)] + r__1 * (sorc[i__ + (lp1 + 656)] - sorc[i__ + (
		l + 656)]));

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    f[i__ + k - 2] = *h44194m2 * (apcm[9] * x[i__ + k - 2] + bpcm[9] *
		     y[i__ + k - 2]);
	    ff[i__ + k - 2] = *h44194m2 * (atpcm[9] * x[i__ + k - 2] + btpcm[
		    9] * y[i__ + k - 2]);
	    ag[i__ + k - 2] = (*h1p41819 + f[i__ + k - 2]) * f[i__ + k - 2] +
		    *one;
	    agg[i__ + k - 2] = (*h1p41819 + ff[i__ + k - 2]) * ff[i__ + k - 2]
		     + *one;

	    r__3 = ag[i__ + k - 2] * ag[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    phitmp[i__ + k - 2] = var1[i__ + k] * (r__1 * r__1);

	    r__3 = agg[i__ + k - 2] * agg[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    psitmp[i__ + k - 2] = var2[i__ + k] * (r__1 * r__1);

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	topm[i__ - 1] = phitmp[i__ - 1];
	topphi[i__ - 1] = psitmp[i__ - 1];

    }
    i__1 = l;
    for (k = 2; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    topm[i__ + k - 2] = topm[i__ + (k - 1) - 2] + phitmp[i__ + k - 2];
	    topphi[i__ + k - 2] = topphi[i__ + (k - 1) - 2] + psitmp[i__ + k
		    - 2];

	}

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    fac1[i__ + k - 2] = acomb[9] * topm[i__ + k - 2];
	    fac2[i__ + k - 2] = fac1[i__ + k - 2] * topm[i__ + k - 2] / (
		    bcomb[9] * topphi[i__ + k - 2]);
	    r__1 = betacm[9] * totvo2[i__ + (k + 1)];
	    tt[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * (fac1[i__ + k - 2] / (
		    real)gd_sqrt((real)(*one + fac2[i__ + k - 2])) + r__1 *
		    gd_tables__1.sko2d))) * co2sp1[i__ + (k + 1)];
	    ctmp[i__ + (k + 1) - 2] = tt[i__ + k - 2] * cldfac[i__ + (k + 83)]
		    ;

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    excts[i__ + k] += sorc[i__ + (k + 738)] * (ctmp[i__ + (k + 1) - 2]
		     - ctmp[i__ + k - 2]);

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	r__2 = *haf * delp[i__ + l];
	r__1 = r__2 * (tt[i__ + lm1 - 2] * (p[i__ + lp1] - press[i__ + l]) +
		tt[i__ + l - 2] * (p[i__ + lp1] + press[i__ + l] - *two * p[
		i__ + l]));
	gxcts[i__] += cldfac[i__ + (lp1 + 82)] * (tt[i__ + l - 2] * sorc[i__
		+ (l + 738)] + r__1 * (sorc[i__ + (lp1 + 738)] - sorc[i__ + (
		l + 738)]));

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    f[i__ + k - 2] = *h44194m2 * (apcm[10] * x[i__ + k - 2] + bpcm[10]
		     * y[i__ + k - 2]);
	    ff[i__ + k - 2] = *h44194m2 * (atpcm[10] * x[i__ + k - 2] + btpcm[
		    10] * y[i__ + k - 2]);
	    ag[i__ + k - 2] = (*h1p41819 + f[i__ + k - 2]) * f[i__ + k - 2] +
		    *one;
	    agg[i__ + k - 2] = (*h1p41819 + ff[i__ + k - 2]) * ff[i__ + k - 2]
		     + *one;

	    r__3 = ag[i__ + k - 2] * ag[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    phitmp[i__ + k - 2] = var1[i__ + k] * (r__1 * r__1);

	    r__3 = agg[i__ + k - 2] * agg[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    psitmp[i__ + k - 2] = var2[i__ + k] * (r__1 * r__1);

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	topm[i__ - 1] = phitmp[i__ - 1];
	topphi[i__ - 1] = psitmp[i__ - 1];

    }
    i__2 = l;
    for (k = 2; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    topm[i__ + k - 2] = topm[i__ + (k - 1) - 2] + phitmp[i__ + k - 2];
	    topphi[i__ + k - 2] = topphi[i__ + (k - 1) - 2] + psitmp[i__ + k
		    - 2];

	}

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    fac1[i__ + k - 2] = acomb[10] * topm[i__ + k - 2];
	    fac2[i__ + k - 2] = fac1[i__ + k - 2] * topm[i__ + k - 2] / (
		    bcomb[10] * topphi[i__ + k - 2]);
	    r__1 = betacm[10] * totvo2[i__ + (k + 1)];
	    tt[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * (fac1[i__ + k - 2] / (
		    real)gd_sqrt((real)(*one + fac2[i__ + k - 2])) + r__1 *
		    gd_tables__1.sko2d))) * co2sp2[i__ + (k + 1)];
	    ctmp[i__ + (k + 1) - 2] = tt[i__ + k - 2] * cldfac[i__ + (k + 83)]
		    ;

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    excts[i__ + k] += sorc[i__ + (k + 820)] * (ctmp[i__ + (k + 1) - 2]
		     - ctmp[i__ + k - 2]);

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	r__2 = *haf * delp[i__ + l];
	r__1 = r__2 * (tt[i__ + lm1 - 2] * (p[i__ + lp1] - press[i__ + l]) +
		tt[i__ + l - 2] * (p[i__ + lp1] + press[i__ + l] - *two * p[
		i__ + l]));
	gxcts[i__] += cldfac[i__ + (lp1 + 82)] * (tt[i__ + l - 2] * sorc[i__
		+ (l + 820)] + r__1 * (sorc[i__ + (lp1 + 820)] - sorc[i__ + (
		l + 820)]));

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    f[i__ + k - 2] = *h44194m2 * (apcm[11] * x[i__ + k - 2] + bpcm[11]
		     * y[i__ + k - 2]);
	    ff[i__ + k - 2] = *h44194m2 * (atpcm[11] * x[i__ + k - 2] + btpcm[
		    11] * y[i__ + k - 2]);
	    ag[i__ + k - 2] = (*h1p41819 + f[i__ + k - 2]) * f[i__ + k - 2] +
		    *one;
	    agg[i__ + k - 2] = (*h1p41819 + ff[i__ + k - 2]) * ff[i__ + k - 2]
		     + *one;

	    r__3 = ag[i__ + k - 2] * ag[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    phitmp[i__ + k - 2] = var1[i__ + k] * (r__1 * r__1);

	    r__3 = agg[i__ + k - 2] * agg[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    psitmp[i__ + k - 2] = var2[i__ + k] * (r__1 * r__1);

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	topm[i__ - 1] = phitmp[i__ - 1];
	topphi[i__ - 1] = psitmp[i__ - 1];

    }
    i__1 = l;
    for (k = 2; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    topm[i__ + k - 2] = topm[i__ + (k - 1) - 2] + phitmp[i__ + k - 2];
	    topphi[i__ + k - 2] = topphi[i__ + (k - 1) - 2] + psitmp[i__ + k
		    - 2];

	}

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    fac1[i__ + k - 2] = acomb[11] * topm[i__ + k - 2];
	    fac2[i__ + k - 2] = fac1[i__ + k - 2] * topm[i__ + k - 2] / (
		    bcomb[11] * topphi[i__ + k - 2]);
	    r__1 = betacm[11] * totvo2[i__ + (k + 1)];
	    tt[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * (fac1[i__ + k - 2] / (
		    real)gd_sqrt((real)(*one + fac2[i__ + k - 2])) + r__1 *
		    gd_tables__1.sko2d)));
	    ctmp[i__ + (k + 1) - 2] = tt[i__ + k - 2] * cldfac[i__ + (k + 83)]
		    ;

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    excts[i__ + k] += sorc[i__ + (k + 902)] * (ctmp[i__ + (k + 1) - 2]
		     - ctmp[i__ + k - 2]);

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	r__2 = *haf * delp[i__ + l];
	r__1 = r__2 * (tt[i__ + lm1 - 2] * (p[i__ + lp1] - press[i__ + l]) +
		tt[i__ + l - 2] * (p[i__ + lp1] + press[i__ + l] - *two * p[
		i__ + l]));
	gxcts[i__] += cldfac[i__ + (lp1 + 82)] * (tt[i__ + l - 2] * sorc[i__
		+ (l + 902)] + r__1 * (sorc[i__ + (lp1 + 902)] - sorc[i__ + (
		l + 902)]));

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    f[i__ + k - 2] = *h44194m2 * (apcm[12] * x[i__ + k - 2] + bpcm[12]
		     * y[i__ + k - 2]);
	    ff[i__ + k - 2] = *h44194m2 * (atpcm[12] * x[i__ + k - 2] + btpcm[
		    12] * y[i__ + k - 2]);
	    ag[i__ + k - 2] = (*h1p41819 + f[i__ + k - 2]) * f[i__ + k - 2] +
		    *one;
	    agg[i__ + k - 2] = (*h1p41819 + ff[i__ + k - 2]) * ff[i__ + k - 2]
		     + *one;

	    r__3 = ag[i__ + k - 2] * ag[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    phitmp[i__ + k - 2] = var1[i__ + k] * (r__1 * r__1);

	    r__3 = agg[i__ + k - 2] * agg[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    psitmp[i__ + k - 2] = var2[i__ + k] * (r__1 * r__1);

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	topm[i__ - 1] = phitmp[i__ - 1];
	topphi[i__ - 1] = psitmp[i__ - 1];

    }
    i__2 = l;
    for (k = 2; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    topm[i__ + k - 2] = topm[i__ + (k - 1) - 2] + phitmp[i__ + k - 2];
	    topphi[i__ + k - 2] = topphi[i__ + (k - 1) - 2] + psitmp[i__ + k
		    - 2];

	}

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    fac1[i__ + k - 2] = acomb[12] * topm[i__ + k - 2];
	    fac2[i__ + k - 2] = fac1[i__ + k - 2] * topm[i__ + k - 2] / (
		    bcomb[12] * topphi[i__ + k - 2]);
	    r__1 = betacm[12] * totvo2[i__ + (k + 1)];
	    tt[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * (fac1[i__ + k - 2] / (
		    real)gd_sqrt((real)(*one + fac2[i__ + k - 2])) + r__1 *
		    gd_tables__1.sko2d)));
	    ctmp[i__ + (k + 1) - 2] = tt[i__ + k - 2] * cldfac[i__ + (k + 83)]
		    ;

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    excts[i__ + k] += sorc[i__ + (k + 984)] * (ctmp[i__ + (k + 1) - 2]
		     - ctmp[i__ + k - 2]);

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	r__2 = *haf * delp[i__ + l];
	r__1 = r__2 * (tt[i__ + lm1 - 2] * (p[i__ + lp1] - press[i__ + l]) +
		tt[i__ + l - 2] * (p[i__ + lp1] + press[i__ + l] - *two * p[
		i__ + l]));
	gxcts[i__] += cldfac[i__ + (lp1 + 82)] * (tt[i__ + l - 2] * sorc[i__
		+ (l + 984)] + r__1 * (sorc[i__ + (lp1 + 984)] - sorc[i__ + (
		l + 984)]));

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    f[i__ + k - 2] = *h44194m2 * (apcm[13] * x[i__ + k - 2] + bpcm[13]
		     * y[i__ + k - 2]);
	    ff[i__ + k - 2] = *h44194m2 * (atpcm[13] * x[i__ + k - 2] + btpcm[
		    13] * y[i__ + k - 2]);
	    ag[i__ + k - 2] = (*h1p41819 + f[i__ + k - 2]) * f[i__ + k - 2] +
		    *one;
	    agg[i__ + k - 2] = (*h1p41819 + ff[i__ + k - 2]) * ff[i__ + k - 2]
		     + *one;

	    r__3 = ag[i__ + k - 2] * ag[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    phitmp[i__ + k - 2] = var1[i__ + k] * (r__1 * r__1);

	    r__3 = agg[i__ + k - 2] * agg[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    psitmp[i__ + k - 2] = var2[i__ + k] * (r__1 * r__1);

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	topm[i__ - 1] = phitmp[i__ - 1];
	topphi[i__ - 1] = psitmp[i__ - 1];

    }
    i__1 = l;
    for (k = 2; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    topm[i__ + k - 2] = topm[i__ + (k - 1) - 2] + phitmp[i__ + k - 2];
	    topphi[i__ + k - 2] = topphi[i__ + (k - 1) - 2] + psitmp[i__ + k
		    - 2];

	}

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    fac1[i__ + k - 2] = acomb[13] * topm[i__ + k - 2];
	    fac2[i__ + k - 2] = fac1[i__ + k - 2] * topm[i__ + k - 2] / (
		    bcomb[13] * topphi[i__ + k - 2]);
	    r__2 = betacm[13] * totvo2[i__ + (k + 1)];
	    r__1 = fac1[i__ + k - 2] / (real)gd_sqrt((real)(*one + fac2[i__ + k
		    - 2])) + r__2 * gd_tables__1.sko2d;
	    tt[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * (r__1 + to3spc[i__ +
		    k])));
	    ctmp[i__ + (k + 1) - 2] = tt[i__ + k - 2] * cldfac[i__ + (k + 83)]
		    ;

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    excts[i__ + k] += sorc[i__ + (k + 1066)] * (ctmp[i__ + (k + 1) -
		    2] - ctmp[i__ + k - 2]);

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	r__2 = *haf * delp[i__ + l];
	r__1 = r__2 * (tt[i__ + lm1 - 2] * (p[i__ + lp1] - press[i__ + l]) +
		tt[i__ + l - 2] * (p[i__ + lp1] + press[i__ + l] - *two * p[
		i__ + l]));
	gxcts[i__] += cldfac[i__ + (lp1 + 82)] * (tt[i__ + l - 2] * sorc[i__
		+ (l + 1066)] + r__1 * (sorc[i__ + (lp1 + 1066)] - sorc[i__ +
		(l + 1066)]));

    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    f[i__ + k - 2] = *h44194m2 * (apcm[14] * x[i__ + k - 2] + bpcm[14]
		     * y[i__ + k - 2]);
	    ff[i__ + k - 2] = *h44194m2 * (atpcm[14] * x[i__ + k - 2] + btpcm[
		    14] * y[i__ + k - 2]);
	    ag[i__ + k - 2] = (*h1p41819 + f[i__ + k - 2]) * f[i__ + k - 2] +
		    *one;
	    agg[i__ + k - 2] = (*h1p41819 + ff[i__ + k - 2]) * ff[i__ + k - 2]
		     + *one;

	    r__3 = ag[i__ + k - 2] * ag[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    phitmp[i__ + k - 2] = var1[i__ + k] * (r__1 * r__1);

	    r__3 = agg[i__ + k - 2] * agg[i__ + k - 2];

	    r__2 = r__3 * r__3;

	    r__1 = r__2 * r__2;
	    psitmp[i__ + k - 2] = var2[i__ + k] * (r__1 * r__1);

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	topm[i__ - 1] = phitmp[i__ - 1];
	topphi[i__ - 1] = psitmp[i__ - 1];

    }
    i__2 = l;
    for (k = 2; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    topm[i__ + k - 2] = topm[i__ + (k - 1) - 2] + phitmp[i__ + k - 2];
	    topphi[i__ + k - 2] = topphi[i__ + (k - 1) - 2] + psitmp[i__ + k
		    - 2];

	}

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    fac1[i__ + k - 2] = acomb[14] * topm[i__ + k - 2];
	    fac2[i__ + k - 2] = fac1[i__ + k - 2] * topm[i__ + k - 2] / (
		    bcomb[14] * topphi[i__ + k - 2]);
	    r__1 = betacm[14] * totvo2[i__ + (k + 1)];
	    tt[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * (fac1[i__ + k - 2] / (
		    real)gd_sqrt((real)(*one + fac2[i__ + k - 2])) + r__1 *
		    gd_tables__1.sko2d)));
	    ctmp[i__ + (k + 1) - 2] = tt[i__ + k - 2] * cldfac[i__ + (k + 83)]
		    ;

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    excts[i__ + k] += sorc[i__ + (k + 1148)] * (ctmp[i__ + (k + 1) -
		    2] - ctmp[i__ + k - 2]);

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	r__2 = *haf * delp[i__ + l];
	r__1 = r__2 * (tt[i__ + lm1 - 2] * (p[i__ + lp1] - press[i__ + l]) +
		tt[i__ + l - 2] * (p[i__ + lp1] + press[i__ + l] - *two * p[
		i__ + l]));
	gxcts[i__] += cldfac[i__ + (lp1 + 82)] * (tt[i__ + l - 2] * sorc[i__
		+ (l + 1148)] + r__1 * (sorc[i__ + (lp1 + 1148)] - sorc[i__ +
		(l + 1148)]));

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    gxcts[i__] -= excts[i__ + k];

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    r__1 = excts[i__ + k] * *radcon;
	    excts[i__ + k] = r__1 * delp[i__ + k];

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    ctmp2[i__ + (k + 1) - 2] = co2sp[i__ + (k + 1)] * cldfac[i__ + (k
		    + 83)];
	    ctmp3[i__ + (k + 1) - 2] = to3sp[i__ + k] * cldfac[i__ + (k + 83)]
		    ;

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    r__1 = *radcon * delp[i__ + k];
	    ctso3[i__ + k] = r__1 * (csour[i__ + k] * (ctmp2[i__ + (k + 1) -
		    2] - ctmp2[i__ + k - 2]) + sorc[i__ + (k + 1066)] * (
		    ctmp3[i__ + (k + 1) - 2] - ctmp3[i__ + k - 2]));

	}
    }
    return 0;
}
















 GD_DEVICE int swr93_(real *fswc, real *hswc, real *ufswc, real *dfswc,
	real *fswl, real *hswl, real *ufswl, real *dfswl, real *press, real *
	coszro, real *taudar, real *rh2o, real *rrco2, real *ssolar, real *
	qo3, integer *nclds, integer *ktopsw, integer *kbtmsw, real *camt,
	real *crr, real *ctt, real *alvb, real *alnb, real *alvd, real *alnd,
	real *gdfvb, real *gdfnb, real *gdfvd, real *gdfnd, real *abcff, real
	*pwts, real *h35e1, real *h1224e3, real *one, real *zero, real *haf,
	real *h69766e5, real *hp219, real *hp816, real *rrayav, real *ginv,
	real *cfco2, real *cfo3, real *two, real *h235m3, real *hp26, real *
	h129m2, real *h75826m4, real *h1036e2, real *h1p082, real *hmp805,
	real *h1386e2, real *h658m2, real *h2118m2, real *h42m2, real *h323m4,
	 real *hm1ez, real *diffctr, real *o3difctr, real *fifty, real *
	radcon, integer *ids, integer *ide, integer *jds, integer *jde,
	integer *kds, integer *kde, integer *ims, integer *ime, integer *jms,
	integer *jme, integer *kms, integer *kme, integer *its, integer *ite,
	integer *jts, integer *jte, integer *kts, integer *kte, GDTable *gt, GDClock *clock, GDScratch *work)
{

    integer i__1, i__2, i__3;
    real r__1, r__2, r__3, r__4;





    integer i__, k, l, n, j1, j2, j3;

    real *ff=work->swr93_ff;
    real *dp=work->swr93_dp;
    real *cr=work->swr93_cr;
    integer kk;

    real *ct=work->swr93_ct;
    integer ll;

    real *du=work->swr93_du;
    real *ud=work->swr93_ud;
    integer ip;

    real *pp=work->swr93_pp;
    real *ur=work->swr93_ur;
    integer lm1, lm2, lm3, lp1, lp2, lp3;

    real *pr2=work->swr93_pr2;
    real *to3=work->swr93_to3;
    real *uo3=work->swr93_uo3;
    real *dfn=work->swr93_dfn;
    real *ttd=work->swr93_ttd;
    real *ufn=work->swr93_ufn;
    real *ttu=work->swr93_ttu;
    real *ffo3=work->swr93_ffo3;
    real *tco2=work->swr93_tco2;
    real *uco2=work->swr93_uco2;
    real *tdo3=work->swr93_tdo3;
    real *duo3=work->swr93_duo3;
    real *udo3=work->swr93_udo3;
    integer llp1;

    real *tmp1=work->swr93_tmp1;
    real *uro3=work->swr93_uro3;
    real *tuo3=work->swr93_tuo3;
    real *alfa=work->swr93_alfa;
    real *tcld=work->swr93_tcld;
    real *refl=work->swr93_refl;
    real *secz=work->swr93_secz;
    integer myie;

    real *tclu=work->swr93_tclu;
    real *xamt=work->swr93_xamt;
    integer jtop;

    real *rray=work->swr93_rray;
    integer myis;

    real *ffco2=work->swr93_ffco2;
    real *tdcl1=work->swr93_tdcl1;
    real *tdcl2=work->swr93_tdcl2;
    real *refl2=work->swr93_refl2;
    real *tdco2=work->swr93_tdco2;
    real *duco2=work->swr93_duco2;
    real *udco2=work->swr93_udco2;
    real *ttdb1=work->swr93_ttdb1;
    real *urco2=work->swr93_urco2;
    real *tucl1=work->swr93_tucl1;
    real *tuco2=work->swr93_tuco2;
    real *ttub1=work->swr93_ttub1;
    integer myis1;

    real *dpcld=work->swr93_dpcld;
    real *alfau=work->swr93_alfau;
    real *ccmax=work->swr93_ccmax;
    integer kclds;

    real denom;
    real tempf;
    real htemp;
    real tempg;
    real *pptop=work->swr93_pptop;
    real *dfnclu=work->swr93_dfnclu;
    integer nnclds;

    real *dfntop=work->swr93_dfntop;
    real *dfntrn=work->swr93_dfntrn;
    real *ufnclu=work->swr93_ufnclu;
    real *ufntrn=work->swr93_ufntrn;


    --pwts;
    --abcff;
    --gdfnd;
    --gdfvd;
    --gdfnb;
    --gdfvb;
    --alnd;
    --alvd;
    --alnb;
    --alvb;
    ctt -= 14;
    crr -= 14;
    camt -= 2;
    kbtmsw -= 2;
    ktopsw -= 2;
    --nclds;
    qo3 -= 2;
    rh2o -= 2;
    --taudar;
    --coszro;
    press -= 2;
    dfswl -= 2;
    ufswl -= 2;
    hswl -= 2;
    fswl -= 2;
    dfswc -= 2;
    ufswc -= 2;
    hswc -= 2;
    fswc -= 2;


    l = *kte;
    lp1 = l + 1;
    lp2 = l + 2;
    lp3 = l + 3;
    llp1 = (l << 1) + 1;
    lm1 = l - 1;
    lm2 = l - 2;
    lm3 = l - 3;
    ll = l << 1;
    myis = *its;
    myie = *ite;
    myis1 = myis + 1;
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	r__1 = *h1224e3 * coszro[i__];
	secz[i__ - 1] = *h35e1 / (real)gd_sqrt((real)(r__1 * coszro[i__] + *one))
		;
	pp[i__ - 1] = *zero;
	pp[i__ + lp1 - 2] = press[i__ + lp1];
	tmp1[i__ - 1] = *one / press[i__ + lp1];

    }
    i__1 = lm1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    pp[i__ + (k + 1) - 2] = *haf * (press[i__ + (k + 1)] + press[i__
		    + k]);

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    dp[i__ + k - 2] = pp[i__ + (k + 1) - 2] - pp[i__ + k - 2];
	    pr2[i__ + k - 2] = *haf * (pp[i__ + k - 2] + pp[i__ + (k + 1) - 2]
		    );

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    pr2[i__ + k - 2] *= tmp1[i__ - 1];

	}
    }
    for (n = 1; n <= 12; ++n) {
	i__2 = myie;
	for (ip = myis; ip <= i__2; ++ip) {
	    r__3 = *ssolar * *h69766e5;
	    r__2 = r__3 * coszro[ip];
	    r__1 = r__2 * taudar[ip];
	    dfntop[ip + n - 2] = r__1 * pwts[n];

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	rray[i__ - 1] = *hp219 / (*one + *hp816 * coszro[i__]);
	r__1 = (*one - rray[i__ - 1]) * (*one - *rrayav);
	refl[i__ - 1] = rray[i__ - 1] + r__1 * alvb[i__] / (*one - alvd[i__] *
		 *rrayav);

    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	rray[i__ - 1] = .104f / (*one + coszro[i__] * 4.8f);
	r__1 = (*one - rray[i__ - 1]) * (*one - .093f);
	refl2[i__ - 1] = rray[i__ - 1] + r__1 * alvb[i__] / (*one - alvd[i__]
		* .093f);

    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    r__2 = *ginv * rh2o[i__ + k];
	    r__1 = r__2 * dp[i__ + k - 2];
	    du[i__ + k - 2] = r__1 * pr2[i__ + k - 2];
	    r__3 = *rrco2 * *ginv;
	    r__2 = r__3 * *cfco2;
	    r__1 = r__2 * dp[i__ + k - 2];
	    duco2[i__ + k - 2] = r__1 * pr2[i__ + k - 2];
	    r__2 = *ginv * *cfo3;
	    r__1 = r__2 * qo3[i__ + k];
	    duo3[i__ + k - 2] = r__1 * dp[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (ip = myis; ip <= i__1; ++ip) {
	ud[ip - 1] = *zero;
	udco2[ip - 1] = *zero;
	udo3[ip - 1] = *zero;
	uo3[ip - 1] = udo3[ip - 1];
	uco2[ip - 1] = udco2[ip - 1];

    }
    i__1 = lp1;
    for (k = 2; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    ud[i__ + k - 2] = ud[i__ + (k - 1) - 2] + du[i__ + (k - 1) - 2] *
		    secz[i__ - 1];
	    udco2[i__ + k - 2] = udco2[i__ + (k - 1) - 2] + duco2[i__ + (k -
		    1) - 2] * secz[i__ - 1];
	    udo3[i__ + k - 2] = udo3[i__ + (k - 1) - 2] + duo3[i__ + (k - 1)
		    - 2] * secz[i__ - 1];
	    uo3[i__ + k - 2] = udo3[i__ + k - 2];
	    uco2[i__ + k - 2] = udco2[i__ + k - 2];

	}
    }
    i__2 = myie;
    for (ip = myis; ip <= i__2; ++ip) {
	ur[ip + lp1 - 2] = ud[ip + lp1 - 2];
	urco2[ip + lp1 - 2] = udco2[ip + lp1 - 2];
	uro3[ip + lp1 - 2] = udo3[ip + lp1 - 2];
	uo3[ip + (lp1 + lp1) - 2] = uro3[ip + lp1 - 2];
	uco2[ip + (lp1 + lp1) - 2] = urco2[ip + lp1 - 2];

    }
    for (k = l; k >= 1; --k) {
	i__2 = myie;
	for (ip = myis; ip <= i__2; ++ip) {
	    ur[ip + k - 2] = ur[ip + (k + 1) - 2] + du[ip + k - 2] * *diffctr;
	    urco2[ip + k - 2] = urco2[ip + (k + 1) - 2] + duco2[ip + k - 2] *
		    *diffctr;
	    uro3[ip + k - 2] = uro3[ip + (k + 1) - 2] + duo3[ip + k - 2] * *
		    o3difctr;
	    uo3[ip + (lp1 + k) - 2] = uro3[ip + k - 2];
	    uco2[ip + (lp1 + k) - 2] = urco2[ip + k - 2];

	}
    }
    i__2 = ll;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    tco2[i__ + (k + 1) - 2] = *one - *two * (*h235m3 * (real)gd_exp((
		    real)(*hp26 * (real)gd_log((real)(uco2[i__ + (k + 1) - 2] + *
		    h129m2)))) - *h75826m4);

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    tdco2[i__ + (k + 1) - 2] = tco2[i__ + (k + 1) - 2];

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    tuco2[i__ + k - 2] = tco2[i__ + (lp1 + k) - 2];

	}
    }
    r__1 = *h1036e2 * *h1036e2;
    htemp = r__1 * *h1036e2;
    i__1 = ll;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    r__1 = *two * uo3[i__ + (k + 1) - 2];
	    r__4 = htemp * uo3[i__ + (k + 1) - 2];
	    r__3 = r__4 * uo3[i__ + (k + 1) - 2];
	    r__2 = *h1p082 * (real)gd_exp((real)(*hmp805 * (real)gd_log((real)(*one
		    + *h1386e2 * uo3[i__ + (k + 1) - 2])))) + *h658m2 / (*one
		    + r__3 * uo3[i__ + (k + 1) - 2]);
	    to3[i__ + (k + 1) - 2] = *one - r__1 * (r__2 + *h2118m2 / (*one +
		    uo3[i__ + (k + 1) - 2] * (*h42m2 + *h323m4 * uo3[i__ + (k
		    + 1) - 2])));

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    tdo3[i__ + (k + 1) - 2] = to3[i__ + (k + 1) - 2];

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    tuo3[i__ + k - 2] = to3[i__ + (lp1 + k) - 2];

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {

	    r__1 = *fifty, r__2 = abcff[1] * ud[i__ + (k + 1) - 2];
	    ttd[i__ + (k + 1) - 2] = (real)gd_exp((real)(*hm1ez * min(r__1,r__2))
		    );

	    r__1 = *fifty, r__2 = abcff[1] * ur[i__ + k - 2];
	    ttu[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * min(r__1,r__2)));
	    dfn[i__ + (k + 1) - 2] = ttd[i__ + (k + 1) - 2] * tdo3[i__ + (k +
		    1) - 2];
	    ufn[i__ + k - 2] = ttu[i__ + k - 2] * tuo3[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	dfn[i__ - 1] = *one;
	ufn[i__ + lp1 - 2] = dfn[i__ + lp1 - 2];

    }
    i__1 = lp1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    dfswl[i__ + k] = dfn[i__ + k - 2] * dfntop[i__ - 1];
	    r__1 = refl[i__ - 1] * ufn[i__ + k - 2];
	    ufswl[i__ + k] = r__1 * dfntop[i__ - 1];

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	gdfvb[i__] = dfswl[i__ + lp1] * (real)gd_exp((real)(secz[i__ - 1] *
		-.15746f));
	gdfvd[i__] = ((*one - refl2[i__ - 1]) * dfswl[i__ + lp1] - (*one -
		alvb[i__]) * gdfvb[i__]) / (*one - alvd[i__]);
	gdfnb[i__] = *zero;
	gdfnd[i__] = *zero;

    }
    for (n = 2; n <= 12; ++n) {
	if (n == 2) {
	    i__2 = l;
	    for (k = 1; k <= i__2; ++k) {
		i__1 = myie;
		for (i__ = myis; i__ <= i__1; ++i__) {
		    dfn[i__ + (k + 1) - 2] = ttd[i__ + (k + 1) - 2] * tdco2[
			    i__ + (k + 1) - 2];
		    ufn[i__ + k - 2] = ttu[i__ + k - 2] * tuco2[i__ + k - 2];

		}
	    }
	} else {
	    i__1 = l;
	    for (k = 1; k <= i__1; ++k) {
		i__2 = myie;
		for (i__ = myis; i__ <= i__2; ++i__) {

		    r__1 = *fifty, r__2 = abcff[n] * ud[i__ + (k + 1) - 2];
		    dfn[i__ + (k + 1) - 2] = (real)gd_exp((real)(*hm1ez * min(
			    r__1,r__2))) * tdco2[i__ + (k + 1) - 2];

		    r__1 = *fifty, r__2 = abcff[n] * ur[i__ + k - 2];
		    ufn[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * min(r__1,
			    r__2))) * tuco2[i__ + k - 2];

		}
	    }
	}
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    dfn[i__ - 1] = *one;
	    ufn[i__ + lp1 - 2] = dfn[i__ + lp1 - 2];

	}
	i__2 = lp1;
	for (k = 1; k <= i__2; ++k) {
	    i__1 = myie;
	    for (i__ = myis; i__ <= i__1; ++i__) {
		dfswl[i__ + k] += dfn[i__ + k - 2] * dfntop[i__ + n - 2];
		r__1 = alnb[i__] * ufn[i__ + k - 2];
		ufswl[i__ + k] += r__1 * dfntop[i__ + n - 2];

	    }
	}
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    gdfnb[i__] += dfn[i__ + lp1 - 2] * dfntop[i__ + n - 2];

	}

    }
    i__1 = lp1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    fswl[i__ + k] = ufswl[i__ + k] - dfswl[i__ + k];

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    hswl[i__ + k] = *radcon * (fswl[i__ + (k + 1)] - fswl[i__ + k]) /
		    dp[i__ + k - 2];

	}
    }
    kclds = nclds[myis];
    i__1 = myie;
    for (i__ = myis1; i__ <= i__1; ++i__) {

	i__2 = nclds[i__];
	kclds = max(i__2,kclds);

    }
    i__1 = lp1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    dfswc[i__ + k] = dfswl[i__ + k];
	    ufswc[i__ + k] = ufswl[i__ + k];
	    fswc[i__ + k] = fswl[i__ + k];

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    hswc[i__ + k] = hswl[i__ + k];

	}
    }
    if (kclds == 0) {
	return 0;
    }
    i__1 = lp1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    xamt[i__ + k - 2] = camt[i__ + k];

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	nnclds = nclds[i__];
	ccmax[i__ - 1] = *zero;
	if (nnclds <= 0) {
	    goto L470;
	}
	ccmax[i__ - 1] = *one;
	i__1 = nnclds;
	for (k = 1; k <= i__1; ++k) {
	    ccmax[i__ - 1] *= *one - camt[i__ + (k + 1)];

	}
	ccmax[i__ - 1] = *one - ccmax[i__ - 1];
	if (ccmax[i__ - 1] > *zero) {
	    i__1 = nnclds;
	    for (k = 1; k <= i__1; ++k) {
		xamt[i__ + (k + 1) - 2] = camt[i__ + (k + 1)] / ccmax[i__ - 1]
			;

	    }
	}
L470:
	;
    }
    i__2 = lp1;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    ff[i__ + k - 2] = *diffctr;
	    ffco2[i__ + k - 2] = *diffctr;
	    ffo3[i__ + k - 2] = *o3difctr;

	}
    }
    i__1 = myie;
    for (ip = myis; ip <= i__1; ++ip) {
	jtop = ktopsw[ip + (nclds[ip] + 1)];
	i__2 = jtop;
	for (k = 1; k <= i__2; ++k) {
	    ff[ip + k - 2] = secz[ip - 1];
	    ffco2[ip + k - 2] = secz[ip - 1];
	    ffo3[ip + k - 2] = secz[ip - 1];

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	rray[i__ - 1] = *hp219 / (*one + *hp816 * coszro[i__]);
	r__1 = (*one - rray[i__ - 1]) * (*one - *rrayav);
	refl[i__ - 1] = rray[i__ - 1] + r__1 * alvd[i__] / (*one - alvd[i__] *
		 *rrayav);

    }
    i__2 = myie;
    for (ip = myis; ip <= i__2; ++ip) {
	ud[ip - 1] = *zero;
	udco2[ip - 1] = *zero;
	udo3[ip - 1] = *zero;
	uo3[ip - 1] = udo3[ip - 1];
	uco2[ip - 1] = udco2[ip - 1];

    }
    i__2 = lp1;
    for (k = 2; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    ud[i__ + k - 2] = ud[i__ + (k - 1) - 2] + du[i__ + (k - 1) - 2] *
		    ff[i__ + k - 2];
	    udco2[i__ + k - 2] = udco2[i__ + (k - 1) - 2] + duco2[i__ + (k -
		    1) - 2] * ffco2[i__ + k - 2];
	    udo3[i__ + k - 2] = udo3[i__ + (k - 1) - 2] + duo3[i__ + (k - 1)
		    - 2] * ffo3[i__ + k - 2];
	    uo3[i__ + k - 2] = udo3[i__ + k - 2];
	    uco2[i__ + k - 2] = udco2[i__ + k - 2];

	}
    }
    i__1 = myie;
    for (ip = myis; ip <= i__1; ++ip) {
	ur[ip + lp1 - 2] = ud[ip + lp1 - 2];
	urco2[ip + lp1 - 2] = udco2[ip + lp1 - 2];
	uro3[ip + lp1 - 2] = udo3[ip + lp1 - 2];
	uo3[ip + (lp1 + lp1) - 2] = uro3[ip + lp1 - 2];
	uco2[ip + (lp1 + lp1) - 2] = urco2[ip + lp1 - 2];

    }
    for (k = l; k >= 1; --k) {
	i__1 = myie;
	for (ip = myis; ip <= i__1; ++ip) {
	    ur[ip + k - 2] = ur[ip + (k + 1) - 2] + du[ip + k - 2] * *diffctr;
	    urco2[ip + k - 2] = urco2[ip + (k + 1) - 2] + duco2[ip + k - 2] *
		    *diffctr;
	    uro3[ip + k - 2] = uro3[ip + (k + 1) - 2] + duo3[ip + k - 2] * *
		    o3difctr;
	    uo3[ip + (lp1 + k) - 2] = uro3[ip + k - 2];
	    uco2[ip + (lp1 + k) - 2] = urco2[ip + k - 2];

	}
    }
    i__1 = ll;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    tco2[i__ + (k + 1) - 2] = *one - *two * (*h235m3 * (real)gd_exp((
		    real)(*hp26 * (real)gd_log((real)(uco2[i__ + (k + 1) - 2] + *
		    h129m2)))) - *h75826m4);

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    tdco2[i__ + (k + 1) - 2] = tco2[i__ + (k + 1) - 2];

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    tuco2[i__ + k - 2] = tco2[i__ + (lp1 + k) - 2];

	}
    }
    i__2 = ll;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    r__1 = *two * uo3[i__ + (k + 1) - 2];
	    r__4 = htemp * uo3[i__ + (k + 1) - 2];
	    r__3 = r__4 * uo3[i__ + (k + 1) - 2];
	    r__2 = *h1p082 * (real)gd_exp((real)(*hmp805 * (real)gd_log((real)(*one
		    + *h1386e2 * uo3[i__ + (k + 1) - 2])))) + *h658m2 / (*one
		    + r__3 * uo3[i__ + (k + 1) - 2]);
	    to3[i__ + (k + 1) - 2] = *one - r__1 * (r__2 + *h2118m2 / (*one +
		    uo3[i__ + (k + 1) - 2] * (*h42m2 + *h323m4 * uo3[i__ + (k
		    + 1) - 2])));

	}
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    tdo3[i__ + (k + 1) - 2] = to3[i__ + (k + 1) - 2];

	}
    }
    i__2 = l;
    for (k = 1; k <= i__2; ++k) {
	i__1 = myie;
	for (i__ = myis; i__ <= i__1; ++i__) {
	    tuo3[i__ + k - 2] = to3[i__ + (lp1 + k) - 2];

	}
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	cr[i__ - 1] = refl[i__ - 1];

    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	kclds = nclds[i__];
	if (kclds == 0) {
	    goto L581;
	}
	i__2 = kclds + 1;
	for (kk = 2; kk <= i__2; ++kk) {
	    cr[i__ + kk - 2] = crr[i__ + (kk * 12 + 1)] * xamt[i__ + kk - 2];
	    ct[i__ + kk - 2] = *one - (*one - ctt[i__ + (kk * 12 + 1)]) *
		    xamt[i__ + kk - 2];

	}
L581:
	;
    }
    i__1 = myie;
    for (i__ = myis; i__ <= i__1; ++i__) {
	kclds = nclds[i__];
	if (kclds == 0) {
	    goto L591;
	}
	i__2 = kclds;
	for (kk = 1; kk <= i__2; ++kk) {
	    if (kbtmsw[i__ + (kk + 1)] - 1 > ktopsw[i__ + (kk + 1)]) {
		pptop[i__ + kk - 2] = pp[i__ + ktopsw[i__ + (kk + 1)] - 2];
		dpcld[i__ + kk - 2] = *one / (pptop[i__ + kk - 2] - pp[i__ +
			kbtmsw[i__ + (kk + 1)] - 2]);
	    }

	}
L591:
	;
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {

	    r__1 = *fifty, r__2 = abcff[1] * ud[i__ + (k + 1) - 2];
	    ttdb1[i__ + (k + 1) - 2] = (real)gd_exp((real)(*hm1ez * min(r__1,
		    r__2)));

	    r__1 = *fifty, r__2 = abcff[1] * ur[i__ + k - 2];
	    ttub1[i__ + k - 2] = (real)gd_exp((real)(*hm1ez * min(r__1,r__2)));
	    ttd[i__ + (k + 1) - 2] = ttdb1[i__ + (k + 1) - 2] * tdo3[i__ + (k
		    + 1) - 2];
	    ttu[i__ + k - 2] = ttub1[i__ + k - 2] * tuo3[i__ + k - 2];

	}
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	ttd[i__ - 1] = *one;
	ttu[i__ + lp1 - 2] = ttd[i__ + lp1 - 2];

    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	tdcl1[i__ - 1] = ttd[i__ + lp1 - 2];
	tucl1[i__ - 1] = ttu[i__ + lp1 - 2];
	tdcl2[i__ - 1] = tdcl1[i__ - 1];
	dfntrn[i__ - 1] = *one / tdcl1[i__ - 1];
	ufntrn[i__ - 1] = dfntrn[i__ - 1];

    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	kclds = nclds[i__];
	if (kclds == 0) {
	    goto L631;
	}
	i__1 = kclds + 1;
	for (kk = 2; kk <= i__1; ++kk) {
	    tdcl1[i__ + kk - 2] = ttd[i__ + ktopsw[i__ + kk] - 2];
	    tucl1[i__ + kk - 2] = ttu[i__ + ktopsw[i__ + kk] - 2];
	    tdcl2[i__ + kk - 2] = ttd[i__ + kbtmsw[i__ + kk] - 2];

	}
L631:
	;
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	kclds = nclds[i__];
	if (kclds == 0) {
	    goto L641;
	}
	i__1 = kclds + 1;
	for (kk = 2; kk <= i__1; ++kk) {
	    dfntrn[i__ + kk - 2] = *one / tdcl1[i__ + kk - 2];
	    ufntrn[i__ + kk - 2] = *one / tucl1[i__ + kk - 2];

	}
L641:
	;
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	kclds = nclds[i__];
	if (kclds == 0) {
	    goto L651;
	}
	i__1 = kclds;
	for (kk = 1; kk <= i__1; ++kk) {
	    r__1 = tdcl1[i__ + kk - 2] * dfntrn[i__ + (kk + 1) - 2];
	    tclu[i__ + kk - 2] = r__1 * ct[i__ + (kk + 1) - 2];
	    tcld[i__ + kk - 2] = tdcl1[i__ + kk - 2] / tdcl2[i__ + (kk + 1) -
		    2];

	}
L651:
	;
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	kclds = nclds[i__];
	if (kclds == 0) {
	    goto L660;
	}
	alfa[i__ - 1] = cr[i__ - 1];
	alfau[i__ - 1] = *zero;
L660:
	;
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	kclds = nclds[i__];
	if (kclds == 0) {
	    goto L671;
	}
	i__1 = kclds + 1;
	for (kk = 2; kk <= i__1; ++kk) {
	    r__1 = tclu[i__ + (kk - 1) - 2] * tclu[i__ + (kk - 1) - 2];
	    r__3 = tcld[i__ + (kk - 1) - 2] * tcld[i__ + (kk - 1) - 2];
	    r__2 = r__3 * alfa[i__ + (kk - 1) - 2];
	    alfau[i__ + kk - 2] = r__1 * alfa[i__ + (kk - 1) - 2] / (*one -
		    r__2 * cr[i__ + kk - 2]);
	    alfa[i__ + kk - 2] = alfau[i__ + kk - 2] + cr[i__ + kk - 2];

	}
L671:
	;
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	kclds = nclds[i__];
	if (kclds == 0) {
	    goto L680;
	}
	ufnclu[i__ + (kclds + 1) - 2] = alfa[i__ + (kclds + 1) - 2] * tdcl1[
		i__ + (kclds + 1) - 2];
	dfnclu[i__ + (kclds + 1) - 2] = tdcl1[i__ + (kclds + 1) - 2];
L680:
	;
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	kclds = nclds[i__];
	if (kclds == 0) {
	    goto L691;
	}
	for (kk = kclds; kk >= 1; --kk) {
	    ufnclu[i__ + kk - 2] = ufnclu[i__ + (kk + 1) - 2] * alfau[i__ + (
		    kk + 1) - 2] / (alfa[i__ + (kk + 1) - 2] * tclu[i__ + kk
		    - 2]);
	    dfnclu[i__ + kk - 2] = ufnclu[i__ + kk - 2] / alfa[i__ + kk - 2];

	}
L691:
	;
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	kclds = nclds[i__];
	if (kclds == 0) {
	    goto L701;
	}
	i__1 = kclds + 1;
	for (kk = 1; kk <= i__1; ++kk) {
	    ufntrn[i__ + kk - 2] = ufnclu[i__ + kk - 2] * ufntrn[i__ + kk - 2]
		    ;
	    dfntrn[i__ + kk - 2] = dfnclu[i__ + kk - 2] * dfntrn[i__ + kk - 2]
		    ;

	}
L701:
	;
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	kclds = nclds[i__];
	if (kclds == 0) {
	    goto L720;
	}
	j2 = kbtmsw[i__ + 2];
	i__1 = lp1;
	for (k = j2; k <= i__1; ++k) {
	    ufn[i__ + k - 2] = ufntrn[i__ - 1] * ttu[i__ + k - 2];
	    dfn[i__ + k - 2] = dfntrn[i__ - 1] * ttd[i__ + k - 2];

	}
L720:
	;
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	kclds = nclds[i__];
	if (kclds == 0) {
	    goto L760;
	}
	i__1 = kclds + 1;
	for (kk = 2; kk <= i__1; ++kk) {
	    j1 = ktopsw[i__ + kk];
	    j2 = kbtmsw[i__ + (kk + 1)];
	    if (j1 == 1) {
		goto L755;
	    }
	    i__3 = j1;
	    for (k = j2; k <= i__3; ++k) {
		ufn[i__ + k - 2] = ufntrn[i__ + kk - 2] * ttu[i__ + k - 2];
		dfn[i__ + k - 2] = dfntrn[i__ + kk - 2] * ttd[i__ + k - 2];

	    }
	    j3 = kbtmsw[i__ + kk];
	    if (j3 - j1 > 1) {
		tempf = (ufnclu[i__ + kk - 2] - ufn[i__ + j3 - 2]) * dpcld[
			i__ + (kk - 1) - 2];
		tempg = (dfnclu[i__ + kk - 2] - dfn[i__ + j3 - 2]) * dpcld[
			i__ + (kk - 1) - 2];
		i__3 = j3 - 1;
		for (k = j1 + 1; k <= i__3; ++k) {
		    ufn[i__ + k - 2] = ufnclu[i__ + kk - 2] + tempf * (pp[i__
			    + k - 2] - pptop[i__ + (kk - 1) - 2]);
		    dfn[i__ + k - 2] = dfnclu[i__ + kk - 2] + tempg * (pp[i__
			    + k - 2] - pptop[i__ + (kk - 1) - 2]);

		}
	    }
L755:
	    ;
	}
L760:
	;
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	kclds = nclds[i__];
	if (kclds == 0) {
	    goto L770;
	}
	i__1 = lp1;
	for (k = 1; k <= i__1; ++k) {
	    dfswc[i__ + k] = dfn[i__ + k - 2] * dfntop[i__ - 1];
	    ufswc[i__ + k] = ufn[i__ + k - 2] * dfntop[i__ - 1];

	}
L770:
	;
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	kclds = nclds[i__];
	if (kclds == 0) {
	    goto L780;
	}
	tmp1[i__ - 1] = *one - ccmax[i__ - 1];
	gdfvb[i__] = tmp1[i__ - 1] * gdfvb[i__];
	gdfnb[i__] = tmp1[i__ - 1] * gdfnb[i__];
	gdfvd[i__] = tmp1[i__ - 1] * gdfvd[i__] + ccmax[i__ - 1] * dfswc[i__
		+ lp1];
L780:
	;
    }
    for (n = 2; n <= 12; ++n) {
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    kclds = nclds[i__];
	    if (kclds == 0) {
		goto L791;
	    }
	    i__1 = kclds + 1;
	    for (k = 1; k <= i__1; ++k) {
		cr[i__ + k - 2] = crr[i__ + (n + k * 12)] * xamt[i__ + k - 2];
		ct[i__ + k - 2] = *one - (*one - ctt[i__ + (n + k * 12)]) *
			xamt[i__ + k - 2];

	    }
L791:
	    ;
	}
	if (n == 2) {
	    i__2 = myie;
	    for (i__ = myis; i__ <= i__2; ++i__) {
		kclds = nclds[i__];
		if (kclds == 0) {
		    goto L800;
		}
		i__1 = lp1;
		for (kk = 2; kk <= i__1; ++kk) {
		    ttd[i__ + kk - 2] = ttdb1[i__ + kk - 2] * tdco2[i__ + kk
			    - 2];

		}
		i__1 = l;
		for (kk = 1; kk <= i__1; ++kk) {
		    ttu[i__ + kk - 2] = ttub1[i__ + kk - 2] * tuco2[i__ + kk
			    - 2];

		}
L800:
		;
	    }
	} else {
	    i__2 = myie;
	    for (i__ = myis; i__ <= i__2; ++i__) {
		kclds = nclds[i__];
		if (kclds == 0) {
		    goto L810;
		}
		i__1 = lp1;
		for (kk = 2; kk <= i__1; ++kk) {

		    r__1 = *fifty, r__2 = abcff[n] * ud[i__ + kk - 2];
		    ttd[i__ + kk - 2] = (real)gd_exp((real)(*hm1ez * min(r__1,
			    r__2))) * tdco2[i__ + kk - 2];

		}
		i__1 = l;
		for (kk = 1; kk <= i__1; ++kk) {

		    r__1 = *fifty, r__2 = abcff[n] * ur[i__ + kk - 2];
		    ttu[i__ + kk - 2] = (real)gd_exp((real)(*hm1ez * min(r__1,
			    r__2))) * tuco2[i__ + kk - 2];

		}
L810:
		;
	    }
	}
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    kclds = nclds[i__];
	    if (kclds == 0) {
		goto L820;
	    }
	    ttu[i__ + lp1 - 2] = ttd[i__ + lp1 - 2];
	    ttd[i__ - 1] = *one;
L820:
	    ;
	}
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    kclds = nclds[i__];
	    if (kclds == 0) {
		goto L830;
	    }
	    tdcl1[i__ - 1] = ttd[i__ + lp1 - 2];
	    tucl1[i__ - 1] = ttu[i__ + lp1 - 2];
	    tdcl2[i__ - 1] = tdcl1[i__ - 1];
	    dfntrn[i__ - 1] = *one / tdcl1[i__ - 1];
	    ufntrn[i__ - 1] = dfntrn[i__ - 1];
L830:
	    ;
	}
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    kclds = nclds[i__];
	    if (kclds == 0) {
		goto L841;
	    }
	    i__1 = kclds + 1;
	    for (kk = 2; kk <= i__1; ++kk) {
		tdcl1[i__ + kk - 2] = ttd[i__ + ktopsw[i__ + kk] - 2];
		tucl1[i__ + kk - 2] = ttu[i__ + ktopsw[i__ + kk] - 2];
		tdcl2[i__ + kk - 2] = ttd[i__ + kbtmsw[i__ + kk] - 2];

	    }
L841:
	    ;
	}
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    kclds = nclds[i__];
	    if (kclds == 0) {
		goto L851;
	    }
	    i__1 = kclds + 1;
	    for (kk = 2; kk <= i__1; ++kk) {
		dfntrn[i__ + kk - 2] = *one / tdcl1[i__ + kk - 2];
		ufntrn[i__ + kk - 2] = *one / tucl1[i__ + kk - 2];

	    }
L851:
	    ;
	}
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    kclds = nclds[i__];
	    if (kclds == 0) {
		goto L861;
	    }
	    i__1 = kclds;
	    for (kk = 1; kk <= i__1; ++kk) {
		r__1 = tdcl1[i__ + kk - 2] * dfntrn[i__ + (kk + 1) - 2];
		tclu[i__ + kk - 2] = r__1 * ct[i__ + (kk + 1) - 2];
		tcld[i__ + kk - 2] = tdcl1[i__ + kk - 2] / tdcl2[i__ + (kk +
			1) - 2];

	    }
L861:
	    ;
	}
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    kclds = nclds[i__];
	    if (kclds == 0) {
		goto L870;
	    }
	    alfa[i__ - 1] = cr[i__ - 1];
	    alfau[i__ - 1] = *zero;
L870:
	    ;
	}
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    kclds = nclds[i__];
	    if (kclds == 0) {
		goto L881;
	    }
	    i__1 = kclds + 1;
	    for (kk = 2; kk <= i__1; ++kk) {
		r__1 = tclu[i__ + (kk - 1) - 2] * tclu[i__ + (kk - 1) - 2];
		r__3 = tcld[i__ + (kk - 1) - 2] * tcld[i__ + (kk - 1) - 2];
		r__2 = r__3 * alfa[i__ + (kk - 1) - 2];
		alfau[i__ + kk - 2] = r__1 * alfa[i__ + (kk - 1) - 2] / (*one
			- r__2 * cr[i__ + kk - 2]);
		alfa[i__ + kk - 2] = alfau[i__ + kk - 2] + cr[i__ + kk - 2];

	    }
L881:
	    ;
	}
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    kclds = nclds[i__];
	    if (kclds == 0) {
		goto L890;
	    }
	    ufnclu[i__ + (kclds + 1) - 2] = alfa[i__ + (kclds + 1) - 2] *
		    tdcl1[i__ + (kclds + 1) - 2];
	    dfnclu[i__ + (kclds + 1) - 2] = tdcl1[i__ + (kclds + 1) - 2];
L890:
	    ;
	}
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    kclds = nclds[i__];
	    if (kclds == 0) {
		goto L901;
	    }
	    for (kk = kclds; kk >= 1; --kk) {
		denom = alfa[i__ + (kk + 1) - 2] * tclu[i__ + kk - 2];
		if (denom > 1.0000000036274937e-15f) {
		    ufnclu[i__ + kk - 2] = ufnclu[i__ + (kk + 1) - 2] * alfau[
			    i__ + (kk + 1) - 2] / denom;
		} else {
		    ufnclu[i__ + kk - 2] = 0.f;
		}
		if (alfa[i__ + kk - 2] > 1.0000000036274937e-15f) {
		    dfnclu[i__ + kk - 2] = ufnclu[i__ + kk - 2] / alfa[i__ +
			    kk - 2];
		} else {
		    dfnclu[i__ + kk - 2] = 0.f;
		}

	    }
L901:
	    ;
	}
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    kclds = nclds[i__];
	    if (kclds == 0) {
		goto L911;
	    }
	    i__1 = kclds + 1;
	    for (kk = 1; kk <= i__1; ++kk) {
		ufntrn[i__ + kk - 2] = ufnclu[i__ + kk - 2] * ufntrn[i__ + kk
			- 2];
		dfntrn[i__ + kk - 2] = dfnclu[i__ + kk - 2] * dfntrn[i__ + kk
			- 2];

	    }
L911:
	    ;
	}
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    kclds = nclds[i__];
	    if (kclds == 0) {
		goto L930;
	    }
	    j2 = kbtmsw[i__ + 2];
	    i__1 = lp1;
	    for (k = j2; k <= i__1; ++k) {
		ufn[i__ + k - 2] = ufntrn[i__ - 1] * ttu[i__ + k - 2];
		dfn[i__ + k - 2] = dfntrn[i__ - 1] * ttd[i__ + k - 2];

	    }
L930:
	    ;
	}
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    kclds = nclds[i__];
	    if (kclds == 0) {
		goto L970;
	    }
	    i__1 = kclds + 1;
	    for (kk = 2; kk <= i__1; ++kk) {
		j1 = ktopsw[i__ + kk];
		j2 = kbtmsw[i__ + (kk + 1)];
		if (j1 == 1) {
		    goto L965;
		}
		i__3 = j1;
		for (k = j2; k <= i__3; ++k) {
		    ufn[i__ + k - 2] = ufntrn[i__ + kk - 2] * ttu[i__ + k - 2]
			    ;
		    dfn[i__ + k - 2] = dfntrn[i__ + kk - 2] * ttd[i__ + k - 2]
			    ;

		}
		j3 = kbtmsw[i__ + kk];
		if (j3 - j1 > 1) {
		    tempf = (ufnclu[i__ + kk - 2] - ufn[i__ + j3 - 2]) *
			    dpcld[i__ + (kk - 1) - 2];
		    tempg = (dfnclu[i__ + kk - 2] - dfn[i__ + j3 - 2]) *
			    dpcld[i__ + (kk - 1) - 2];
		    i__3 = j3 - 1;
		    for (k = j1 + 1; k <= i__3; ++k) {
			ufn[i__ + k - 2] = ufnclu[i__ + kk - 2] + tempf * (pp[
				i__ + k - 2] - pptop[i__ + (kk - 1) - 2]);
			dfn[i__ + k - 2] = dfnclu[i__ + kk - 2] + tempg * (pp[
				i__ + k - 2] - pptop[i__ + (kk - 1) - 2]);

		    }
		}
L965:
		;
	    }
L970:
	    ;
	}
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    kclds = nclds[i__];
	    if (kclds == 0) {
		goto L980;
	    }
	    i__1 = lp1;
	    for (k = 1; k <= i__1; ++k) {
		dfswc[i__ + k] += dfn[i__ + k - 2] * dfntop[i__ + n - 2];
		ufswc[i__ + k] += ufn[i__ + k - 2] * dfntop[i__ + n - 2];

	    }
L980:
	    ;
	}
	i__2 = myie;
	for (i__ = myis; i__ <= i__2; ++i__) {
	    kclds = nclds[i__];
	    if (kclds == 0) {
		goto L990;
	    }
	    r__1 = ccmax[i__ - 1] * dfn[i__ + lp1 - 2];
	    gdfnd[i__] += r__1 * dfntop[i__ + n - 2];
L990:
	    ;
	}

    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	kclds = nclds[i__];
	if (kclds == 0) {
	    goto L1100;
	}
	i__1 = lp1;
	for (k = 1; k <= i__1; ++k) {
	    dfswc[i__ + k] = tmp1[i__ - 1] * dfswl[i__ + k] + ccmax[i__ - 1] *
		     dfswc[i__ + k];
	    ufswc[i__ + k] = tmp1[i__ - 1] * ufswl[i__ + k] + ccmax[i__ - 1] *
		     ufswc[i__ + k];

	}
L1100:
	;
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	kclds = nclds[i__];
	if (kclds == 0) {
	    goto L1200;
	}
	i__1 = lp1;
	for (kk = 1; kk <= i__1; ++kk) {
	    fswc[i__ + kk] = ufswc[i__ + kk] - dfswc[i__ + kk];

	}
L1200:
	;
    }
    i__2 = myie;
    for (i__ = myis; i__ <= i__2; ++i__) {
	kclds = nclds[i__];
	if (kclds == 0) {
	    goto L1250;
	}
	i__1 = l;
	for (kk = 1; kk <= i__1; ++kk) {
	    hswc[i__ + kk] = *radcon * (fswc[i__ + (kk + 1)] - fswc[i__ + kk])
		     / dp[i__ + kk - 2];

	}
L1250:
	;
    }
    return 0;
}
















 GD_DEVICE int zenith_(real *times, real *dayi, real *hour, integer *
	idat, integer *ihrst, real *glon, real *glat, real *czen, integer *
	myis, integer *myie, integer *myjs, integer *myje, integer *ids,
	integer *ide, integer *jds, integer *jde, integer *kds, integer *kde,
	integer *ims, integer *ime, integer *jms, integer *jme, integer *kms,
	integer *kme, integer *its, integer *ite, integer *jts, integer *jte,
	integer *kts, integer *kte, GDTable *gt, GDClock *clock, GDScratch *work)
{


    integer month[12] = { 31,28,31,30,31,30,31,31,30,31,30,31 };


    integer i__1, i__2;
    real r__1, r__2;





    integer i__, j;
    real ra, tu, dec, day;
    integer knt;
    logical leap;
    real anom, slon, datj0, stim0, difjd, hrang, hrlcl;
    integer kmnth;
    real yfctr, slonm, addday, datjul;
    integer idifyr;
    real sidtim, sinalt, startyr;


    czen -= 2;
    glat -= 2;
    glon -= 2;
    --idat;


    day = 0.f;
    leap = FALSE_;
    if (idat[3] % 4 == 0) {
	month[1] = 29;
	leap = TRUE_;
    }
    if (idat[1] > 1) {
	kmnth = idat[1] - 1;
	i__1 = kmnth;
	for (knt = 1; knt <= i__1; ++knt) {
	    day += (real) month[knt - 1];

	}
    }
    r__1 = day + (real) (idat[2] - 1);
    day = r__1 + ((real) (*ihrst) + *times / 3600.f) / 24.f;
    *dayi = (real) ((integer) day + 1);
    *hour = (day - *dayi + 1.f) * 24.f;
    yfctr = 2e3f - idat[3];
    idifyr = idat[3] - 2000;
    if (idifyr < 0) {
	addday = (real) (idifyr / 4);
    } else {
	addday = (real) ((idifyr + 3) / 4);
    }
    r__2 = idifyr * 365.f + 2451545.f;
    r__1 = r__2 + addday;
    startyr = r__1 - .5f;
    datjul = startyr + day;
    difjd = datjul - 2451545.f;
    slonm = (difjd * .9856474f + 280.46f) * .01745329238474369f + yfctr *
	    6.2831850051879883f;
    anom = (difjd * .9856003f + 357.528f) * .01745329238474369f;
    slon = slonm + ((real)gd_sin(anom) * 1.915f + (real)gd_sin((real)(anom * 2.f)) *
	     .02f) * .01745329238474369f;
    if (slon > 6.2831850051879883f) {
	slon += -6.2831850051879883f;
    }
    dec = (real)gd_asin((real)((real)gd_sin(slon) * (real)gd_sin(.40910518169403076f)))
	    ;
    ra = (real)gd_acos((real)((real)gd_cos(slon) / (real)gd_cos(dec)));
    if (slon > 3.1415925025939941f) {
	ra = 6.2831850051879883f - ra;
    }
    r__1 = startyr + *dayi;
    datj0 = r__1 - 1.f;
    tu = (datj0 - 2451545.f) / 36525.f;
    r__1 = tu * .093103997409343719f + 8640185.f;
    r__2 = tu * -6.1999999161344022e-6f;
    stim0 = tu * (r__1 + r__2 * tu) + 24110.548828125f;
    r__1 = stim0 / 3600.f + yfctr * 24.f;
    sidtim = r__1 + *hour * 1.00273791f;
    r__1 = sidtim * 15.f;
    sidtim = r__1 * .01745329238474369f;
    if (sidtim < 0.f) {
	sidtim += 6.2831850051879883f;
    }
    if (sidtim > 6.2831850051879883f) {
	sidtim += -6.2831850051879883f;
    }
    hrang = sidtim - ra;
    i__1 = *myje;
    for (j = *myjs; j <= i__1; ++j) {
	i__2 = *myie;
	for (i__ = *myis; i__ <= i__2; ++i__) {
	    r__1 = hrang + glon[i__ + j];
	    hrlcl = r__1 + 6.2831850051879883f;
	    r__1 = (real)gd_cos(dec) * (real)gd_cos(hrlcl);
	    sinalt = (real)gd_sin(dec) * (real)gd_sin(glat[i__ + j]) + r__1 * (real)
		    gd_cos(glat[i__ + j]);
	    if (sinalt < 0.f) {
		sinalt = 0.f;
	    }
	    czen[i__ + j] = sinalt;

	}
    }
    if (*dayi > 365.f) {
	if (! leap) {
	    *dayi += -365.f;
	} else if (leap && *dayi > 366.f) {
	    *dayi += -366.f;
	}
    }
    return 0;
}
