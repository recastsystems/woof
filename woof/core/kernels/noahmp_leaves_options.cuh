// WRF-lineage v4 Noah-MP 848f54ad, module_sf_noahmplsm.F:1083-1251.
// Public domain: licenses/LICENSE-WRF-public-domain.txt; no v5 source.
extern "C" __global__
void noahmp_leaf_atm_selected(const real* __restrict__ xs, const int* __restrict__ ixs,
                     real* __restrict__ ys, int ncase, int snf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncase) return;
    (void)ixs;
    const real* x = xs + (size_t)idx * 11;
    real* y = ys + (size_t)idx * 17;

    real sfcprs = x[0], sfctmp = x[1], q2 = x[2];
    real prcpconv = x[3], prcpnonc = x[4], prcpshcv = x[5];
    // x[6], x[7], x[8] are PRCPSNOW / PRCPGRPL / PRCPHAIL, read only inside
    // the IF(OPT_SNF == 4) block at :1217-1228.
    real soldn = x[9], cosz = x[10];

    real pair = sfcprs;                                            // :1144
    real thair = MU(sfctmp, r_pow(DV(sfcprs, pair),
                                  DV(NMP_RAIR, NMP_CPAIR)));       // :1145
    real qair = q2;                                                // :1147
    real eair = DV(MU(qair, sfcprs),
                   AD(0.622f, MU(0.378f, qair)));                  // :1149
    real rhoair = DV(SU(sfcprs, MU(0.378f, eair)),
                     MU(NMP_RAIR, sfctmp));                        // :1150

    real swdown = (cosz <= 0.0f) ? 0.0f : soldn;                   // :1152-1156
    real solad1 = MU(MU(swdown, 0.7f), 0.5f);                      // :1158
    real solai1 = MU(MU(swdown, 0.3f), 0.5f);                      // :1160

    real prcp = AD(AD(prcpconv, prcpnonc), prcpshcv);              // :1163
    real qprecc = MU(0.10f, prcp);                                 // :1169
    real qprecl = MU(0.90f, prcp);
    if(snf==4){qprecc=AD(prcpconv,prcpshcv);qprecl=prcpnonc;}                                 // :1170

    real fp = 0.0f;                                                // :1175
    if (AD(qprecc, qprecl) > 0.0f) {                               // :1176-1177
        fp = DV(AD(qprecc, qprecl), AD(MU(10.0f, qprecc), qprecl));
    }

    real fpice;                                                    // :1183-1195
    if (sfctmp > AD(NMP_TFRZ, 2.5f)) {
        fpice = 0.0f;
    } else if (sfctmp <= AD(NMP_TFRZ, 0.5f)) {
        fpice = 1.0f;
    } else if (sfctmp <= AD(NMP_TFRZ, 2.0f)) {
        fpice = SU(1.0f, AD(-54.632f, MU(0.2f, sfctmp)));
    } else {
        fpice = 0.6f;
    }

    real bdfall = fminf(120.0f,
        AD(67.92f, MU(51.25f, r_exp(DV(SU(sfctmp, NMP_TFRZ), 2.59f)))));  // :1216

    // Native option arms :1197-1245, in their original statement order.
    if(snf==2)fpice=sfctmp>=AD(NMP_TFRZ,2.2f)?0.0f:1.0f;
    if(snf==3)fpice=sfctmp>=NMP_TFRZ?0.0f:1.0f;
    if(snf==4){
        real frozen=AD(AD(x[6],x[7]),x[8]);
        if(prcpnonc>0.0f && frozen>0.0f){
            real ratio=DV(frozen,prcpnonc);
            fpice=1.0f<ratio?1.0f:ratio;
            fpice=0.0f>fpice?0.0f:fpice;
            bdfall=AD(AD(MU(bdfall,DV(x[6],frozen)),MU(500.0f,DV(x[7],frozen))),MU(917.0f,DV(x[8],frozen)));
        }else fpice=0.0f;
    }
    if(snf==5){
        real tdc=SU(sfctmp,NMP_TFRZ);
        tdc=-50.0f>tdc?-50.0f:tdc;tdc=50.0f<tdc?50.0f:tdc;
        real latent=sfctmp>NMP_TFRZ?2.5104e6f:2.8440e6f;
        real gamma=DV(MU(NMP_CPAIR,sfcprs),MU(0.622f,latent));
        real wet=SU(tdc,5.0f);
        for(int j=0;j<10;j++){
            real es=MU(610.8f,r_exp(DV(MU(17.27f,wet),AD(237.3f,wet))));
            wet=SU(wet,DV(SU(es,eair),gamma));
        }
        fpice=DV(1.0f,AD(1.0f,MU(6.99e-5f,r_exp(MU(2.0f,AD(wet,3.97f))))));
    }

    y[0] = thair;
    y[1] = qair;
    y[2] = eair;
    y[3] = rhoair;
    y[4] = qprecc;
    y[5] = qprecl;
    y[6] = solad1;
    y[7] = solad1;                                                 // :1159
    y[8] = solai1;
    y[9] = solai1;                                                 // :1161
    y[10] = swdown;
    y[11] = bdfall;
    y[12] = MU(prcp, SU(1.0f, fpice));                             // :1247
    y[13] = MU(prcp, fpice);                                       // :1248
    y[14] = fp;
    y[15] = fpice;
    y[16] = prcp;
}

