// WRF v4 Noah-MP 848f54ad3d28c4303151fe5ad83724e232694422.
// Native PHASECHANGE :5595-5810 and FRH2O :5814-5946, public-domain WRF notice.
// Compiled after the unchanged default thermal unit.

// Independently written SNICAR heating coupling. The full absorbed solar
// budget already enters SSOIL at the surface. Move the penetrating portion
// to HRT's volumetric layer sources, preserving the column energy budget.
__device__ void noahmp_hrt_selected_core(int opt_stc, int opt_tbot, int isnow, const real* zsnso, const real* stc, real tbot, real zbot, const real* df, const real* hcpct, real ssoil, const real* phi, real* ai, real* bi, real* ci, real* rhs, real* botflx);
extern "C" __global__ void noahmp_tsnosoi_snow_papers(
    const real *xs,const int *ixs,const real *solar,real *ys,int ncase,int opt_stc,int opt_tbot){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=ncase)return;
    const real *x=xs+(size_t)c*(5*NMP_NLAY+7);
    real *y=ys+(size_t)c*(NMP_NLAY+1);int isnow=ixs[5*c],top=isnow+3;
    real z[7],t[7],df[7],hc[7],phi[7],ai[7],bi[7],ci[7],rhs[7];
    real penetrating=0.f;
    for(int j=0;j<7;++j){z[j]=x[j];t[j]=x[7+j];df[j]=x[14+j];hc[j]=x[21+j];
        phi[j]=j>top?solar[(size_t)c*7+j]:0.f;penetrating+=phi[j];}
    const real *p=x+35;real bottom;
    if(opt_stc==2 || opt_tbot==1) noahmp_hrt_selected_core(opt_stc,opt_tbot,isnow,z,t,p[0],p[6]-p[4],df,hc,p[1]-penetrating,phi,ai,bi,ci,rhs,&bottom);
    else noahmp_hrt_core(isnow,z,t,p[0],p[6]-p[4],df,hc,p[1]-penetrating,phi,ai,bi,ci,rhs,&bottom);
    noahmp_hstep_core(isnow,p[3],ai,bi,ci,rhs,t);
    for(int j=0;j<7;++j)y[j]=t[j];y[7]=bottom;
}

__device__ real noahmp_frh2o_selected(real tkelv, real smc, real sh2o, real bexp, real psisat, real smcmax)
{
    const real ck = NMP_FRH2O_CONST[0];
    const real blim = NMP_FRH2O_CONST[1];
    const real err = NMP_FRH2O_CONST[2];

    real bx = bexp;                                                // :5860
    if (bexp > blim) bx = blim;                                    // :5866
    int nlog = 0, kcount = 0;

    if (tkelv > SU(NMP_TFRZ, 1.0e-3f)) {                           // :5872
        return smc;                                     // :5873

    }

    // :5878  IF (CK /= 0.0) -- CK is the PARAMETER 8.0, so always taken.
    real swl = SU(smc, sh2o);                                      // :5879
    if (swl > SU(smc, 0.02f)) swl = SU(smc, 0.02f);                // :5883
    if (swl < 0.0f) swl = 0.0f;                                    // :5887
    while (nlog < 10 && kcount == 0) {                             // :5888-5889
        nlog += 1;
        real arg = MU(MU(DV(MU(psisat, NMP_GRAV), NMP_HFUS),
                         r_pow(AD(1.0f, MU(ck, swl)), 2.0f)),
                      r_pow(DV(smcmax, SU(smc, swl)), bx));        // :5891-5893
        real df = SU(r_log(arg), r_log(-DV(SU(tkelv, NMP_TFRZ), tkelv)));
        real denom = AD(DV(MU(2.0f, ck), AD(1.0f, MU(ck, swl))),
                        DV(bx, SU(smc, swl)));                     // :5894
        real swlk = SU(swl, DV(df, denom));                        // :5895
        if (swlk > SU(smc, 0.02f)) swlk = SU(smc, 0.02f);          // :5899
        if (swlk < 0.0f) swlk = 0.0f;                              // :5900
        real dswl = fabsf(SU(swlk, swl));                          // :5905
        swl = swlk;                                                // :5909
        if (dswl <= err) kcount += 1;                              // :5910
    }
    real free = SU(smc, swl);                                      // :5919

    if (kcount == 0) {                                             // :5928
        // :5929-5930 writes a diagnostic through wrf_message; the port has
        // nowhere to write it and drops it.  This arm is NOT bound by the
        // fixture -- see run_thermal.F90's note on the 200000-draw search.
        real fk = MU(r_pow(MU(DV(NMP_HFUS, MU(NMP_GRAV, -psisat)),
                              DV(SU(tkelv, NMP_TFRZ), tkelv)),
                           -DV(1.0f, bx)),
                     smcmax);                                      // :5931-5932
        if (fk < 0.02f) fk = 0.02f;                                // :5933
        free = fminf(fk, smc);                                     // :5934
    }
    return free;
}

__device__ void noahmp_phasechange_frz2_core(
    int isnow, int ist, real dt,
    const real* __restrict__ fact, const real* __restrict__ dzsnso,
    const real* __restrict__ smcmax, const real* __restrict__ psisat,
    const real* __restrict__ bexp,
    real* __restrict__ stc, real* __restrict__ snice,
    real* __restrict__ snliq, real* __restrict__ smc,
    real* __restrict__ sh2o, real* __restrict__ sneqv_io,
    real* __restrict__ snowh_io, real* __restrict__ qmelt_out,
    real* __restrict__ ponding_out, int* __restrict__ imelt)
{
    real sneqv = *sneqv_io;
    real snowh = *snowh_io;
    real qmelt = 0.0f;                                             // :5644
    real ponding = 0.0f;                                           // :5645
    real supercool[NMP_NLAY], mice[NMP_NLAY], mliq[NMP_NLAY];
    real wice0[NMP_NLAY], wmass0[NMP_NLAY], hm[NMP_NLAY], xm[NMP_NLAY];
    for (int i = 0; i < NMP_NLAY; ++i) {
        imelt[i] = 0;
        supercool[i] = 0.0f;                                       // :5648-5650
        mice[i] = 0.0f;
        mliq[i] = 0.0f;
        wice0[i] = 0.0f;
        wmass0[i] = 0.0f;
        hm[i] = 0.0f;
        xm[i] = 0.0f;
    }

    for (int j = isnow + 1; j <= 0; ++j) {                         // :5652-5655
        int s = j + NMP_OFF;
        mice[s] = snice[s];
        mliq[s] = snliq[s];
    }
    for (int j = 1; j <= NMP_NSOIL; ++j) {                         // :5657-5660
        int s = j + NMP_OFF;
        mliq[s] = MU(MU(sh2o[j - 1], dzsnso[s]), 1000.0f);
        mice[s] = MU(MU(SU(smc[j - 1], sh2o[j - 1]), dzsnso[s]), 1000.0f);
    }
    for (int j = isnow + 1; j <= NMP_NSOIL; ++j) {                 // :5662-5669
        int s = j + NMP_OFF;
        imelt[s] = 0;
        hm[s] = 0.0f;
        xm[s] = 0.0f;
        wice0[s] = mice[s];
        wmass0[s] = AD(mice[s], mliq[s]);
    }

    if (ist == 1) {                                                // :5671-5697
        for (int j = 1; j <= NMP_NSOIL; ++j) {
            int s = j + NMP_OFF;
            if (stc[s] < NMP_TFRZ) {                               // :5684
                supercool[s] = noahmp_frh2o_selected(stc[s], smc[j - 1], sh2o[j - 1],
                                  bexp[j - 1], psisat[j - 1], smcmax[j - 1]);
                supercool[s] = MU(MU(supercool[s], dzsnso[s]), 1000.0f);
            }
        }
    }

    for (int j = isnow + 1; j <= NMP_NSOIL; ++j) {                 // :5699-5713
        int s = j + NMP_OFF;
        if (mice[s] > 0.0f && stc[s] >= NMP_TFRZ) imelt[s] = 1;
        if (mliq[s] > supercool[s] && stc[s] < NMP_TFRZ) imelt[s] = 2;
        if (isnow == 0 && sneqv > 0.0f && j == 1) {
            if (stc[s] >= NMP_TFRZ) imelt[s] = 1;
        }
    }

    for (int j = isnow + 1; j <= NMP_NSOIL; ++j) {                 // :5717-5731
        int s = j + NMP_OFF;
        if (imelt[s] > 0) {
            hm[s] = DV(SU(stc[s], NMP_TFRZ), fact[s]);
            stc[s] = NMP_TFRZ;
        }
        // :5723-5730.  FACT is DT/(HCPCT*DZSNSO) at :2497-2499 and therefore
        // strictly positive at every call site, which ties sign(HM) to the
        // IMELT test that produced it and makes both resets unreachable in a
        // forecast.  They are transcribed, and the fixture's fact_sign_probe
        // case binds them with a negative FACT.
        if (imelt[s] == 1 && hm[s] < 0.0f) { hm[s] = 0.0f; imelt[s] = 0; }
        if (imelt[s] == 2 && hm[s] > 0.0f) { hm[s] = 0.0f; imelt[s] = 0; }
        xm[s] = DV(MU(hm[s], dt), NMP_HFUS);
    }

    int one = 1 + NMP_OFF;
    if (isnow == 0 && sneqv > 0.0f && xm[one] > 0.0f) {            // :5735-5752
        real temp1 = sneqv;
        sneqv = fmaxf(0.0f, SU(temp1, xm[one]));
        real propor = DV(sneqv, temp1);
        snowh = fmaxf(0.0f, MU(propor, snowh));
        snowh = fminf(fmaxf(snowh, DV(sneqv, 500.0f)), DV(sneqv, 50.0f));
        real heatr = SU(hm[one], DV(MU(NMP_HFUS, SU(temp1, sneqv)), dt));
        if (heatr > 0.0f) {
            xm[one] = DV(MU(heatr, dt), NMP_HFUS);
            hm[one] = heatr;
        } else {
            xm[one] = 0.0f;
            hm[one] = 0.0f;
        }
        qmelt = DV(fmaxf(0.0f, SU(temp1, sneqv)), dt);
        // :5751  XMF = HFUS*QMELT -- a local nothing reads.
        ponding = SU(temp1, sneqv);
    }

    for (int j = isnow + 1; j <= NMP_NSOIL; ++j) {                 // :5756-5795
        int s = j + NMP_OFF;
        if (imelt[s] > 0 && fabsf(hm[s]) > 0.0f) {
            real heatr = 0.0f;
            if (xm[s] > 0.0f) {                                    // :5760-5762
                mice[s] = fmaxf(0.0f, SU(wice0[s], xm[s]));
                heatr = SU(hm[s], DV(MU(NMP_HFUS, SU(wice0[s], mice[s])), dt));
            } else if (xm[s] < 0.0f) {                             // :5763-5776
                if (j <= 0) {
                    mice[s] = fminf(wmass0[s], SU(wice0[s], xm[s]));
                } else {
                    // :5768 WMASS0 < SUPERCOOL is unreachable: XM<0 implies
                    // IMELT==2 (positive FACT) implies MLIQ > SUPERCOOL, and
                    // WMASS0 = MICE + MLIQ with MICE >= 0.
                    if (wmass0[s] < supercool[s]) {
                        mice[s] = 0.0f;
                    } else {
                        mice[s] = fminf(SU(wmass0[s], supercool[s]),
                                        SU(wice0[s], xm[s]));
                        mice[s] = fmaxf(mice[s], 0.0f);
                    }
                }
                heatr = SU(hm[s], DV(MU(NMP_HFUS, SU(wice0[s], mice[s])), dt));
            }
            mliq[s] = fmaxf(0.0f, SU(wmass0[s], mice[s]));         // :5779
            if (fabsf(heatr) > 0.0f) {                             // :5781-5790
                stc[s] = AD(stc[s], MU(fact[s], heatr));
                if (j <= 0) {
                    if (MU(mliq[s], mice[s]) > 0.0f) stc[s] = NMP_TFRZ;
                    if (mice[s] == 0.0f) {                         // BARLAGE
                        stc[s] = NMP_TFRZ;
                        hm[s + 1] = AD(hm[s + 1], heatr);
                        xm[s + 1] = DV(MU(hm[s + 1], dt), NMP_HFUS);
                    }
                }
            }
            // :5789  XMF = XMF + ... -- a local nothing reads.
            if (j < 1) {                                           // :5791-5793
                qmelt = AD(qmelt,
                           DV(fmaxf(0.0f, SU(wice0[s], mice[s])), dt));
            }
        }
    }

    for (int j = isnow + 1; j <= 0; ++j) {                         // :5797-5800
        int s = j + NMP_OFF;
        snliq[s] = mliq[s];
        snice[s] = mice[s];
    }
    for (int j = 1; j <= NMP_NSOIL; ++j) {                         // :5802-5805
        int s = j + NMP_OFF;
        sh2o[j - 1] = DV(mliq[s], MU(1000.0f, dzsnso[s]));
        smc[j - 1] = DV(AD(mliq[s], mice[s]), MU(1000.0f, dzsnso[s]));
    }

    *sneqv_io = sneqv;
    *snowh_io = snowh;
    *qmelt_out = qmelt;
    *ponding_out = ponding;
}

extern "C" __global__
void noahmp_thermal_phasechange_frz2(const real* __restrict__ xs,
                                const int* __restrict__ ixs,
                                real* __restrict__ ys, int ncase)
{
    const int NIN = 4 * NMP_NLAY + 2 * NMP_NSNOW + 2 * NMP_NSOIL + 3
                    + 3 * NMP_NSOIL;
    const int NOUT = NMP_NLAY + 2 * NMP_NSNOW + 2 * NMP_NSOIL + 4 + NMP_NLAY;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncase) return;
    const real* x = xs + (size_t)idx * NIN;
    real* y = ys + (size_t)idx * NOUT;
    const int* ix = ixs + (size_t)idx * 4;
    int isnow = ix[0];
    int ist = ix[1];
    // ix[2] ILOC and ix[3] JLOC are declared at :5608-5609 and never read.

    real fact[NMP_NLAY], dzsnso[NMP_NLAY], stc[NMP_NLAY];
    real snice[NMP_NSNOW], snliq[NMP_NSNOW];
    real smc[NMP_NSOIL], sh2o[NMP_NSOIL];
    real smcmax[NMP_NSOIL], psisat[NMP_NSOIL], bexp[NMP_NSOIL];
    for (int i = 0; i < NMP_NLAY; ++i) {
        fact[i] = x[i];
        dzsnso[i] = x[NMP_NLAY + i];
        // x[2*NMP_NLAY + i] is HCPCT: declared at :5617, never referenced.
        stc[i] = x[3 * NMP_NLAY + i];
    }
    int base = 4 * NMP_NLAY;
    for (int i = 0; i < NMP_NSNOW; ++i) {
        snice[i] = x[base + i];
        snliq[i] = x[base + NMP_NSNOW + i];
    }
    base += 2 * NMP_NSNOW;
    for (int i = 0; i < NMP_NSOIL; ++i) {
        smc[i] = x[base + i];
        sh2o[i] = x[base + NMP_NSOIL + i];
    }
    base += 2 * NMP_NSOIL;
    real sneqv = x[base];
    real snowh = x[base + 1];
    real dt = x[base + 2];
    base += 3;
    for (int i = 0; i < NMP_NSOIL; ++i) {
        smcmax[i] = x[base + i];
        psisat[i] = x[base + NMP_NSOIL + i];
        bexp[i] = x[base + 2 * NMP_NSOIL + i];
    }

    real qmelt, ponding;
    int imelt[NMP_NLAY];
    noahmp_phasechange_frz2_core(isnow, ist, dt, fact, dzsnso, smcmax, psisat, bexp,
                            stc, snice, snliq, smc, sh2o, &sneqv, &snowh,
                            &qmelt, &ponding, imelt);

    for (int i = 0; i < NMP_NLAY; ++i) y[i] = stc[i];
    base = NMP_NLAY;
    for (int i = 0; i < NMP_NSNOW; ++i) {
        y[base + i] = snice[i];
        y[base + NMP_NSNOW + i] = snliq[i];
    }
    base += 2 * NMP_NSNOW;
    for (int i = 0; i < NMP_NSOIL; ++i) {
        y[base + i] = smc[i];
        y[base + NMP_NSOIL + i] = sh2o[i];
    }
    base += 2 * NMP_NSOIL;
    y[base] = sneqv;
    y[base + 1] = snowh;
    y[base + 2] = qmelt;
    y[base + 3] = ponding;
    base += 4;
    for (int i = 0; i < NMP_NLAY; ++i) y[base + i] = (real)imelt[i];
}


// Native HRT :5375-5473, TSNOSOI :5266-5371, complete selected top/bottom boundary.
__device__ void noahmp_hrt_selected_core(
    int opt_stc, int opt_tbot, int isnow, const real* zsnso, const real* stc, real tbot, real zbot,
    const real* df, const real* hcpct, real ssoil, const real* phi,
    real* ai, real* bi, real* ci, real* rhsts, real* botflx)
{
    real denom[NMP_NLAY], ddz[NMP_NLAY], dtsdz[NMP_NLAY], eflux[NMP_NLAY];
    for (int i = 0; i < NMP_NLAY; ++i) {
        ai[i] = 0.0f;
        bi[i] = 0.0f;
        ci[i] = 0.0f;
        rhsts[i] = 0.0f;
        denom[i] = 0.0f;
        ddz[i] = 0.0f;
        dtsdz[i] = 0.0f;
        eflux[i] = 0.0f;
    }
    *botflx = 0.0f;

    for (int k = isnow + 1; k <= NMP_NSOIL; ++k) {                 // :5424-5449
        int j = k + NMP_OFF;
        if (k == isnow + 1) {                                      // :5425-5430
            denom[j] = -MU(zsnso[j], hcpct[j]);
            real temp1 = -zsnso[j + 1];
            ddz[j] = DV(2.0f, temp1);
            dtsdz[j] = DV(MU(2.0f, SU(stc[j], stc[j + 1])), temp1);
            eflux[j] = SU(SU(MU(df[j], dtsdz[j]), ssoil), phi[j]);
        } else if (k < NMP_NSOIL) {                                // :5431-5436
            denom[j] = MU(SU(zsnso[j - 1], zsnso[j]), hcpct[j]);
            real temp1 = SU(zsnso[j - 1], zsnso[j + 1]);
            ddz[j] = DV(2.0f, temp1);
            dtsdz[j] = DV(MU(2.0f, SU(stc[j], stc[j + 1])), temp1);
            eflux[j] = SU(SU(MU(df[j], dtsdz[j]),
                             MU(df[j - 1], dtsdz[j - 1])), phi[j]);
        } else {                                                   // :5437-5447
            denom[j] = MU(SU(zsnso[j - 1], zsnso[j]), hcpct[j]);
            // :5439 assigns TEMP1 here and never reads it again.
            if (opt_tbot == 1) *botflx = 0.0f;
            else {
            dtsdz[j] = DV(SU(stc[j], tbot),
                          SU(MU(0.5f, AD(zsnso[j - 1], zsnso[j])), zbot));
            *botflx = -MU(df[j], dtsdz[j]);
            }
            eflux[j] = SU(SU(-*botflx, MU(df[j - 1], dtsdz[j - 1])), phi[j]);
        }
    }

    for (int k = isnow + 1; k <= NMP_NSOIL; ++k) {                 // :5451-5471
        int j = k + NMP_OFF;
        if (k == isnow + 1) {                                      // :5452-5460
            ai[j] = 0.0f;
            ci[j] = -DV(MU(df[j], ddz[j]), denom[j]);
            bi[j] = -ci[j];
            if (opt_stc == 2)
                bi[j] = AD(-ci[j], DV(df[j], MU(MU(MU(0.5f, zsnso[j]), zsnso[j]), hcpct[j])));
        } else if (k < NMP_NSOIL) {                                // :5461-5464
            ai[j] = -DV(MU(df[j - 1], ddz[j - 1]), denom[j]);
            ci[j] = -DV(MU(df[j], ddz[j]), denom[j]);
            bi[j] = -AD(ai[j], ci[j]);
        } else {                                                   // :5465-5468
            ai[j] = -DV(MU(df[j - 1], ddz[j - 1]), denom[j]);
            ci[j] = 0.0f;
            bi[j] = -AD(ai[j], ci[j]);
        }
        rhsts[j] = DV(eflux[j], -denom[j]);                        // :5470
    }
}

__device__ void noahmp_tsnosoi_selected_core(int opt_stc, int opt_tbot, int isnow, real dt, real tbot, real ssoil,
                                    real snowh, real zbot,
                                    const real* __restrict__ zsnso,
                                    const real* __restrict__ df,
                                    const real* __restrict__ hcpct,
                                    real* __restrict__ stc,
                                    real* __restrict__ eflxb)
{
    real zbotsno = SU(zbot, snowh);                                // :5314
    real phi[NMP_NLAY];                                            // :5310
    for (int i = 0; i < NMP_NLAY; ++i) phi[i] = 0.0f;

    real ai[NMP_NLAY], bi[NMP_NLAY], ci[NMP_NLAY], rhsts[NMP_NLAY];
    noahmp_hrt_selected_core(opt_stc, opt_tbot, isnow, zsnso, stc, tbot, zbotsno, df, hcpct, ssoil, phi,
                    ai, bi, ci, rhsts, eflxb);                     // :5324
    noahmp_hstep_core(isnow, dt, ai, bi, ci, rhsts, stc);          // :5330
    // :5337-5342 fills the local EFLXB2, which the RETURN at :5346 discards.
}

extern "C" __global__
void noahmp_thermal_tsnosoi_selected(const real* __restrict__ xs,
                            const int* __restrict__ ixs,
                            real* __restrict__ ys, int ncase, int opt_stc, int opt_tbot)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncase) return;
    const real* x = xs + (size_t)idx * (5 * NMP_NLAY + 7);
    real* y = ys + (size_t)idx * (NMP_NLAY + 1);
    // ixs[5*idx + 1..4] are ICE, IST, ILOC, JLOC: none survives :5346.
    int isnow = ixs[(size_t)idx * 5];

    real zsnso[NMP_NLAY], stc[NMP_NLAY], df[NMP_NLAY], hcpct[NMP_NLAY];
    for (int i = 0; i < NMP_NLAY; ++i) {
        zsnso[i] = x[i];
        stc[i] = x[NMP_NLAY + i];
        df[i] = x[2 * NMP_NLAY + i];
        hcpct[i] = x[3 * NMP_NLAY + i];
        // x[4*NMP_NLAY + i] is DZSNSO, read only after the RETURN at :5346.
    }
    int base = 5 * NMP_NLAY;
    real tbot = x[base];
    real ssoil = x[base + 1];
    real dt = x[base + 3];
    real snowh = x[base + 4];
    real zbot = x[base + 6];

    real eflxb;
    noahmp_tsnosoi_selected_core(opt_stc, opt_tbot, isnow, dt, tbot, ssoil, snowh, zbot,
                        zsnso, df, hcpct, stc, &eflxb);

    for (int i = 0; i < NMP_NLAY; ++i) y[i] = stc[i];
    y[NMP_NLAY] = eflxb;
}


// Native ENERGY :2373-2387, after TSNOSOI and before PHASECHANGE.
extern "C" __global__ void noahmp_energy_stc2_snow(
 const real* snowh,const real* fveg,const real* tv,const bool* tile,
 real* tg,real* ts,real* tgv,real* tgb,int n){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 if(snowh[c]>0.05f && tg[c]>NMP_TFRZ){
  tgv[c]=NMP_TFRZ;tgb[c]=NMP_TFRZ;
  if(tile[c] && fveg[c]>0.0f){
   tg[c]=AD(MU(fveg[c],NMP_TFRZ),MU(SU(1.0f,fveg[c]),NMP_TFRZ));
   ts[c]=AD(MU(fveg[c],tv[c]),MU(SU(1.0f,fveg[c]),NMP_TFRZ));
  }else{tg[c]=NMP_TFRZ;ts[c]=NMP_TFRZ;}
 }
}
