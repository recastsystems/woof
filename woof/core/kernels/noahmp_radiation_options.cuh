// WRF v4 Noah-MP 848f54ad3d28c4303151fe5ad83724e232694422.
// Selected-option extension compiled after the unchanged default unit.
// Public-domain source notice: licenses/LICENSE-WRF-public-domain.txt.

// SNOWALB_BATS :3171-3222. Seven actual transferred MPTABLE parameters:
// BATS_COSZ, BATS_VIS_NEW, BATS_NIR_NEW, BATS_VIS_AGE, BATS_NIR_AGE,
// BATS_VIS_DIR, BATS_NIR_DIR. FSNO is a native dummy, unused in this leaf.
// out: ALBSND(1:2), ALBSNI(1:2).
__device__ void d_snowalb_bats(const float *b, float cosz, float fage, float *o)
{
    float sl1 = FDIV(K_ONE, b[0]);
    float sl2 = FMUL(K_TWO, b[0]);
    float cf1 = FSUB(FDIV(FADD(K_ONE, sl1),
                          FADD(K_ONE, FMUL(sl2, cosz))), sl1);
    float fzen = f_max(cf1, K_ZERO);
    o[2] = FMUL(b[1], FSUB(K_ONE, FMUL(b[3], fage)));
    o[3] = FMUL(b[2], FSUB(K_ONE, FMUL(b[4], fage)));
    o[0] = FADD(o[2], FMUL(FMUL(b[5], fzen), FSUB(K_ONE, o[2])));
    o[1] = FADD(o[3], FMUL(FMUL(b[6], fzen), FSUB(K_ONE, o[3])));
}

// TWOSTREAM opt_rad=2, native :3442. All other ordered arithmetic is retained.
__device__ void d_twostream_dense(const float *p, int ib1, int ic, float *o)
{
    int ib = ib1 - 1;
    float xl = p[0];
    float omegas[2] = { p[1], p[2] };
    float betads = p[3], betais = p[4];
    float cosz = p[5], vai = p[6], fwet = p[7], t = p[8], fveg = p[9];
    float albgrd[2] = { p[10], p[11] };
    float albgri[2] = { p[12], p[13] };
    float rho[2] = { p[14], p[15] };
    float tau[2] = { p[16], p[17] };
    float fab[2] = { p[18], p[19] };
    float fre[2] = { p[20], p[21] };
    float ftd[2] = { p[22], p[23] };
    float fti[2] = { p[24], p[25] };
    float gdir  = p[26];
    float frev[2] = { p[27], p[28] };
    float freg[2] = { p[29], p[30] };
    float bgap = p[31], wgap = p[32];

    float gap, kopen;
    if (vai == K_ZERO) { gap = K_ONE; kopen = K_ONE; }
    else               { gap = K_ZERO; kopen = K_ZERO; }

    float coszi = f_max(K_P001, cosz);
    float chil = f_min(f_max(xl, K_NP4), K_P6);
    if (fabsf(chil) <= K_P01) chil = K_P01;
    float phi1 = FSUB(FSUB(K_HALF, FMUL(K_P633, chil)),
                      FMUL(FMUL(K_P330, chil), chil));
    float phi2 = FMUL(K_P877, FSUB(K_ONE, FMUL(K_TWO, phi1)));
    gdir = FADD(phi1, FMUL(phi2, coszi));
    float ext = FDIV(gdir, coszi);
    float avmu = FDIV(FSUB(K_ONE,
                           FMUL(FDIV(phi1, phi2),
                                glibc_logf(FDIV(FADD(phi1, phi2), phi1)))),
                      phi2);
    float omegal = FADD(rho[ib], tau[ib]);
    float tmp0 = FADD(gdir, FMUL(phi2, coszi));
    float tmp1 = FMUL(phi1, coszi);
    float asu = FMUL(FDIV(FMUL(FMUL(K_HALF, omegal), gdir), tmp0),
                     FSUB(K_ONE,
                          FMUL(FDIV(tmp1, tmp0),
                               glibc_logf(FDIV(FADD(tmp1, tmp0), tmp1)))));
    float betadl = FMUL(FDIV(FADD(K_ONE, FMUL(avmu, ext)),
                             FMUL(FMUL(omegal, avmu), ext)), asu);
    float qh = FDIV(FADD(K_ONE, chil), K_TWO);
    float q = FMUL(qh, qh);
    float betail = FDIV(FMUL(K_HALF,
                             FADD(FADD(rho[ib], tau[ib]),
                                  FMUL(FSUB(rho[ib], tau[ib]), q))), omegal);

    float tmp2;
    if (t > K_TFRZ) {
        tmp0 = omegal; tmp1 = betadl; tmp2 = betail;
    } else {
        tmp0 = FADD(FMUL(FSUB(K_ONE, fwet), omegal), FMUL(fwet, omegas[ib]));
        tmp1 = FDIV(FADD(FMUL(FMUL(FSUB(K_ONE, fwet), omegal), betadl),
                         FMUL(FMUL(fwet, omegas[ib]), betads)), tmp0);
        tmp2 = FDIV(FADD(FMUL(FMUL(FSUB(K_ONE, fwet), omegal), betail),
                         FMUL(FMUL(fwet, omegas[ib]), betais)), tmp0);
    }
    float omega = tmp0, betad = tmp1, betai = tmp2;

    float b = FADD(FSUB(K_ONE, omega), FMUL(omega, betai));
    float c = FMUL(omega, betai);
    tmp0 = FMUL(avmu, ext);
    float d = FMUL(FMUL(tmp0, omega), betad);
    float f = FMUL(FMUL(tmp0, omega), FSUB(K_ONE, betad));
    tmp1 = FSUB(FMUL(b, b), FMUL(c, c));
    float h = FDIV(FSQRT(tmp1), avmu);
    float sigma = FSUB(FMUL(tmp0, tmp0), tmp1);
    if (fabsf(sigma) < K_MPE) sigma = f_sign(K_MPE, sigma);
    float p1 = FADD(b, FMUL(avmu, h));
    float p2 = FSUB(b, FMUL(avmu, h));
    float p3 = FADD(b, tmp0);
    float p4 = FSUB(b, tmp0);
    float s1 = glibc_expf(FMUL(-h, vai));
    float s2 = glibc_expf(FMUL(-ext, vai));
    float u1, u2, u3;
    if (ic == 0) {
        u1 = FSUB(b, FDIV(c, albgrd[ib]));
        u2 = FSUB(b, FMUL(c, albgrd[ib]));
        u3 = FADD(f, FMUL(c, albgrd[ib]));
    } else {
        u1 = FSUB(b, FDIV(c, albgri[ib]));
        u2 = FSUB(b, FMUL(c, albgri[ib]));
        u3 = FADD(f, FMUL(c, albgri[ib]));
    }
    tmp2 = FSUB(u1, FMUL(avmu, h));
    float tmp3 = FADD(u1, FMUL(avmu, h));
    float d1 = FSUB(FDIV(FMUL(p1, tmp2), s1), FMUL(FMUL(p2, tmp3), s1));
    float tmp4 = FADD(u2, FMUL(avmu, h));
    float tmp5 = FSUB(u2, FMUL(avmu, h));
    float d2 = FSUB(FDIV(tmp4, s1), FMUL(tmp5, s1));
    float h1 = FSUB(FMUL(-d, p4), FMUL(c, f));
    float tmp6 = FSUB(d, FDIV(FMUL(h1, p3), sigma));
    float tmp7 = FMUL(FSUB(FSUB(d, c), FMUL(FDIV(h1, sigma), FADD(u1, tmp0))), s2);
    float h2 = FDIV(FSUB(FDIV(FMUL(tmp6, tmp2), s1), FMUL(p2, tmp7)), d1);
    float h3 = -FDIV(FSUB(FMUL(FMUL(tmp6, tmp3), s1), FMUL(p1, tmp7)), d1);
    float h4 = FSUB(FMUL(-f, p3), FMUL(c, d));
    float tmp8 = FDIV(h4, sigma);
    float tmp9 = FMUL(FSUB(u3, FMUL(tmp8, FSUB(u2, tmp0))), s2);
    float h5 = -FDIV(FADD(FDIV(FMUL(tmp8, tmp4), s1), tmp9), d2);
    float h6 = FDIV(FADD(FMUL(FMUL(tmp8, tmp5), s1), tmp9), d2);
    float h7 = FDIV(FMUL(c, tmp2), FMUL(d1, s1));
    float h8 = FDIV(FMUL(FMUL(-c, tmp3), s1), d1);
    float h9 = FDIV(tmp4, FMUL(d2, s1));
    float h10 = FDIV(FMUL(-tmp5, s1), d2);

    float ftds, ftis;
    if (ic == 0) {
        ftds = FADD(FMUL(s2, FSUB(K_ONE, gap)), gap);
        ftis = FMUL(FADD(FADD(FDIV(FMUL(h4, s2), sigma), FMUL(h5, s1)),
                         FDIV(h6, s1)), FSUB(K_ONE, gap));
    } else {
        ftds = K_ZERO;
        ftis = FADD(FMUL(FADD(FMUL(h9, s1), FDIV(h10, s1)),
                         FSUB(K_ONE, kopen)), kopen);
    }
    ftd[ib] = ftds;
    fti[ib] = ftis;

    float fres, freveg, frebar;
    if (ic == 0) {
        fres = FADD(FMUL(FADD(FADD(FDIV(h1, sigma), h2), h3),
                         FSUB(K_ONE, gap)), FMUL(albgrd[ib], gap));
        freveg = FMUL(FADD(FADD(FDIV(h1, sigma), h2), h3), FSUB(K_ONE, gap));
        frebar = FMUL(albgrd[ib], gap);
    } else {
        fres = FADD(FMUL(FADD(h7, h8), FSUB(K_ONE, kopen)),
                    FMUL(albgri[ib], kopen));
        freveg = FADD(FMUL(FADD(h7, h8), FSUB(K_ONE, kopen)),
                      FMUL(albgri[ib], kopen));
        frebar = K_ZERO;
    }
    fre[ib] = fres;
    frev[ib] = freveg;
    freg[ib] = frebar;

    fab[ib] = FSUB(FSUB(FSUB(K_ONE, fre[ib]),
                        FMUL(FSUB(K_ONE, albgrd[ib]), ftd[ib])),
                   FMUL(FSUB(K_ONE, albgri[ib]), fti[ib]));

    o[0] = fab[0];  o[1] = fab[1];  o[2] = fre[0];  o[3] = fre[1];
    o[4] = ftd[0];  o[5] = ftd[1];  o[6] = fti[0];  o[7] = fti[1];
    o[8] = gdir;
    o[9] = frev[0]; o[10] = frev[1]; o[11] = freg[0]; o[12] = freg[1];
    o[13] = bgap;   o[14] = wgap;
}

// TWOSTREAM crown geometry, native :3427-3440.
__device__ void d_twostream_crown(const float *p, int ib1, int ic, float *o, const float *crown)
{
    int ib = ib1 - 1;
    float xl = p[0];
    float omegas[2] = { p[1], p[2] };
    float betads = p[3], betais = p[4];
    float cosz = p[5], vai = p[6], fwet = p[7], t = p[8], fveg = p[9];
    float albgrd[2] = { p[10], p[11] };
    float albgri[2] = { p[12], p[13] };
    float rho[2] = { p[14], p[15] };
    float tau[2] = { p[16], p[17] };
    float fab[2] = { p[18], p[19] };
    float fre[2] = { p[20], p[21] };
    float ftd[2] = { p[22], p[23] };
    float fti[2] = { p[24], p[25] };
    float gdir  = p[26];
    float frev[2] = { p[27], p[28] };
    float freg[2] = { p[29], p[30] };
    float bgap = p[31], wgap = p[32];

    float gap, kopen;
    if (vai == K_ZERO) { gap = K_ONE; kopen = K_ONE; }
    else {
        const float pai = __uint_as_float(0x40490fdbU);
        float rc = crown[0], hvt = crown[1], hvb = crown[2];
        float rc2 = FMUL(rc, rc);
        float denfveg = FDIV(-glibc_logf(f_max(FSUB(K_ONE, fveg), K_P01)),
                            FMUL(pai, rc2));
        float hd = FSUB(hvt, hvb);
        float bb = FMUL(K_HALF, hd);
        float thetap = glibc_atanf(FMUL(FDIV(bb, rc),
                            glibc_tanf(glibc_acosf(f_max(K_P01, cosz)))));
        bgap = glibc_expf(FDIV(FMUL(FMUL(-denfveg, pai), rc2), glibc_cosf(thetap)));
        float fa = FDIV(vai, FMUL(FMUL(FMUL(FMUL(__uint_as_float(0x3faa3d71U), pai),
                      glibc_powf(rc, __uint_as_float(0x40400000U))), FDIV(bb, rc)), denfveg));
        float newvai = FMUL(hd, fa);
        wgap = FMUL(FSUB(K_ONE, bgap), glibc_expf(FDIV(FMUL(-K_HALF, newvai), cosz)));
        gap = f_min(FSUB(K_ONE, fveg), FADD(bgap, wgap));
        kopen = __uint_as_float(0x3d4ccccdU);
    }

    float coszi = f_max(K_P001, cosz);
    float chil = f_min(f_max(xl, K_NP4), K_P6);
    if (fabsf(chil) <= K_P01) chil = K_P01;
    float phi1 = FSUB(FSUB(K_HALF, FMUL(K_P633, chil)),
                      FMUL(FMUL(K_P330, chil), chil));
    float phi2 = FMUL(K_P877, FSUB(K_ONE, FMUL(K_TWO, phi1)));
    gdir = FADD(phi1, FMUL(phi2, coszi));
    float ext = FDIV(gdir, coszi);
    float avmu = FDIV(FSUB(K_ONE,
                           FMUL(FDIV(phi1, phi2),
                                glibc_logf(FDIV(FADD(phi1, phi2), phi1)))),
                      phi2);
    float omegal = FADD(rho[ib], tau[ib]);
    float tmp0 = FADD(gdir, FMUL(phi2, coszi));
    float tmp1 = FMUL(phi1, coszi);
    float asu = FMUL(FDIV(FMUL(FMUL(K_HALF, omegal), gdir), tmp0),
                     FSUB(K_ONE,
                          FMUL(FDIV(tmp1, tmp0),
                               glibc_logf(FDIV(FADD(tmp1, tmp0), tmp1)))));
    float betadl = FMUL(FDIV(FADD(K_ONE, FMUL(avmu, ext)),
                             FMUL(FMUL(omegal, avmu), ext)), asu);
    float qh = FDIV(FADD(K_ONE, chil), K_TWO);
    float q = FMUL(qh, qh);
    float betail = FDIV(FMUL(K_HALF,
                             FADD(FADD(rho[ib], tau[ib]),
                                  FMUL(FSUB(rho[ib], tau[ib]), q))), omegal);

    float tmp2;
    if (t > K_TFRZ) {
        tmp0 = omegal; tmp1 = betadl; tmp2 = betail;
    } else {
        tmp0 = FADD(FMUL(FSUB(K_ONE, fwet), omegal), FMUL(fwet, omegas[ib]));
        tmp1 = FDIV(FADD(FMUL(FMUL(FSUB(K_ONE, fwet), omegal), betadl),
                         FMUL(FMUL(fwet, omegas[ib]), betads)), tmp0);
        tmp2 = FDIV(FADD(FMUL(FMUL(FSUB(K_ONE, fwet), omegal), betail),
                         FMUL(FMUL(fwet, omegas[ib]), betais)), tmp0);
    }
    float omega = tmp0, betad = tmp1, betai = tmp2;

    float b = FADD(FSUB(K_ONE, omega), FMUL(omega, betai));
    float c = FMUL(omega, betai);
    tmp0 = FMUL(avmu, ext);
    float d = FMUL(FMUL(tmp0, omega), betad);
    float f = FMUL(FMUL(tmp0, omega), FSUB(K_ONE, betad));
    tmp1 = FSUB(FMUL(b, b), FMUL(c, c));
    float h = FDIV(FSQRT(tmp1), avmu);
    float sigma = FSUB(FMUL(tmp0, tmp0), tmp1);
    if (fabsf(sigma) < K_MPE) sigma = f_sign(K_MPE, sigma);
    float p1 = FADD(b, FMUL(avmu, h));
    float p2 = FSUB(b, FMUL(avmu, h));
    float p3 = FADD(b, tmp0);
    float p4 = FSUB(b, tmp0);
    float s1 = glibc_expf(FMUL(-h, vai));
    float s2 = glibc_expf(FMUL(-ext, vai));
    float u1, u2, u3;
    if (ic == 0) {
        u1 = FSUB(b, FDIV(c, albgrd[ib]));
        u2 = FSUB(b, FMUL(c, albgrd[ib]));
        u3 = FADD(f, FMUL(c, albgrd[ib]));
    } else {
        u1 = FSUB(b, FDIV(c, albgri[ib]));
        u2 = FSUB(b, FMUL(c, albgri[ib]));
        u3 = FADD(f, FMUL(c, albgri[ib]));
    }
    tmp2 = FSUB(u1, FMUL(avmu, h));
    float tmp3 = FADD(u1, FMUL(avmu, h));
    float d1 = FSUB(FDIV(FMUL(p1, tmp2), s1), FMUL(FMUL(p2, tmp3), s1));
    float tmp4 = FADD(u2, FMUL(avmu, h));
    float tmp5 = FSUB(u2, FMUL(avmu, h));
    float d2 = FSUB(FDIV(tmp4, s1), FMUL(tmp5, s1));
    float h1 = FSUB(FMUL(-d, p4), FMUL(c, f));
    float tmp6 = FSUB(d, FDIV(FMUL(h1, p3), sigma));
    float tmp7 = FMUL(FSUB(FSUB(d, c), FMUL(FDIV(h1, sigma), FADD(u1, tmp0))), s2);
    float h2 = FDIV(FSUB(FDIV(FMUL(tmp6, tmp2), s1), FMUL(p2, tmp7)), d1);
    float h3 = -FDIV(FSUB(FMUL(FMUL(tmp6, tmp3), s1), FMUL(p1, tmp7)), d1);
    float h4 = FSUB(FMUL(-f, p3), FMUL(c, d));
    float tmp8 = FDIV(h4, sigma);
    float tmp9 = FMUL(FSUB(u3, FMUL(tmp8, FSUB(u2, tmp0))), s2);
    float h5 = -FDIV(FADD(FDIV(FMUL(tmp8, tmp4), s1), tmp9), d2);
    float h6 = FDIV(FADD(FMUL(FMUL(tmp8, tmp5), s1), tmp9), d2);
    float h7 = FDIV(FMUL(c, tmp2), FMUL(d1, s1));
    float h8 = FDIV(FMUL(FMUL(-c, tmp3), s1), d1);
    float h9 = FDIV(tmp4, FMUL(d2, s1));
    float h10 = FDIV(FMUL(-tmp5, s1), d2);

    float ftds, ftis;
    if (ic == 0) {
        ftds = FADD(FMUL(s2, FSUB(K_ONE, gap)), gap);
        ftis = FMUL(FADD(FADD(FDIV(FMUL(h4, s2), sigma), FMUL(h5, s1)),
                         FDIV(h6, s1)), FSUB(K_ONE, gap));
    } else {
        ftds = K_ZERO;
        ftis = FADD(FMUL(FADD(FMUL(h9, s1), FDIV(h10, s1)),
                         FSUB(K_ONE, kopen)), kopen);
    }
    ftd[ib] = ftds;
    fti[ib] = ftis;

    float fres, freveg, frebar;
    if (ic == 0) {
        fres = FADD(FMUL(FADD(FADD(FDIV(h1, sigma), h2), h3),
                         FSUB(K_ONE, gap)), FMUL(albgrd[ib], gap));
        freveg = FMUL(FADD(FADD(FDIV(h1, sigma), h2), h3), FSUB(K_ONE, gap));
        frebar = FMUL(albgrd[ib], gap);
    } else {
        fres = FADD(FMUL(FADD(h7, h8), FSUB(K_ONE, kopen)),
                    FMUL(albgri[ib], kopen));
        freveg = FADD(FMUL(FADD(h7, h8), FSUB(K_ONE, kopen)),
                      FMUL(albgri[ib], kopen));
        frebar = K_ZERO;
    }
    fre[ib] = fres;
    frev[ib] = freveg;
    freg[ib] = frebar;

    fab[ib] = FSUB(FSUB(FSUB(K_ONE, fre[ib]),
                        FMUL(FSUB(K_ONE, albgrd[ib]), ftd[ib])),
                   FMUL(FSUB(K_ONE, albgri[ib]), fti[ib]));

    o[0] = fab[0];  o[1] = fab[1];  o[2] = fre[0];  o[3] = fre[1];
    o[4] = ftd[0];  o[5] = ftd[1];  o[6] = fti[0];  o[7] = fti[1];
    o[8] = gdir;
    o[9] = frev[0]; o[10] = frev[1]; o[11] = freg[0]; o[12] = freg[1];
    o[13] = bgap;   o[14] = wgap;
}

__device__ void d_albedo_selected(const float *p, int ist, float *o, bool night_age=false,
                         const float *bats=nullptr, bool dense=false, const float *crown=nullptr, const float *snow=nullptr)
{
    float swemx = p[4];
    float dt = p[24], cosz = p[25], elai = p[26], esai = p[27];
    float tg = p[28], tv = p[29];
    float fsno = p[31], fwet = p[32], sneqvo = p[33], sneqv = p[34];
    float qsnow = p[35], fveg = p[36];
    float albold = p[41], tauss = p[42], fage = p[43];

    float mpe = K_MPE;
    float bgap = K_ZERO, wgap = K_ZERO;
    float albd[2] = { K_ZERO, K_ZERO }, albi[2] = { K_ZERO, K_ZERO };
    float albgrd[2] = { K_ZERO, K_ZERO }, albgri[2] = { K_ZERO, K_ZERO };
    float albsnd[2] = { K_ZERO, K_ZERO }, albsni[2] = { K_ZERO, K_ZERO };
    float fabd[2] = { K_ZERO, K_ZERO }, fabi[2] = { K_ZERO, K_ZERO };
    float ftdd[2] = { K_ZERO, K_ZERO }, ftid[2] = { K_ZERO, K_ZERO };
    float ftii[2] = { K_ZERO, K_ZERO };
    float fsun = K_ZERO;
    // Not zeroed by ALBEDO's init loop (lines 2829-2842): undefined on the
    // COSZ <= 0 exit, which in the reference build means the caller's bytes.
    float frevd[2] = { p[44], p[45] }, frevi[2] = { p[46], p[47] };
    float fregd[2] = { p[48], p[49] }, fregi[2] = { p[50], p[51] };
    float ftdi[2] = { K_ZERO, K_ZERO };
    float gdir = K_ZERO;

    if (snow) { tauss = snow[4]; albold = snow[5]; }
    if (night_age && !snow) {
        float sa_in[10] = { p[0], p[1], p[2], p[3], swemx,
                            dt, tg, sneqvo, sneqv, tauss };
        float sa_out[2];
        d_snow_age(sa_in, sa_out);
        tauss = sa_out[0]; fage = sa_out[1];
    }
    if (cosz > K_ZERO) {
        float rho[2], tau[2], vai = K_ZERO;
        for (int ib = 0; ib < 2; ++ib) {
            vai = FADD(elai, esai);
            float wl = FDIV(elai, f_max(vai, mpe));
            float ws = FDIV(esai, f_max(vai, mpe));
            rho[ib] = f_max(FADD(FMUL(p[11 + ib], wl), FMUL(p[13 + ib], ws)), mpe);
            tau[ib] = f_max(FADD(FMUL(p[15 + ib], wl), FMUL(p[17 + ib], ws)), mpe);
        }

        float sa_in[10] = { p[0], p[1], p[2], p[3], swemx,
                            dt, tg, sneqvo, sneqv, tauss };
        float sa_out[2];
        if (!night_age && !snow) {
            d_snow_age(sa_in, sa_out);
            tauss = sa_out[0];
            fage = sa_out[1];
        }

        if (snow) {
            albsnd[0] = snow[0]; albsnd[1] = snow[1];
            albsni[0] = snow[2]; albsni[1] = snow[3];
        } else if (bats != nullptr) {
            float sb_out[4];
            d_snowalb_bats(bats, cosz, fage, sb_out);
            albsnd[0] = sb_out[0]; albsnd[1] = sb_out[1];
            albsni[0] = sb_out[2]; albsni[1] = sb_out[3];
            // Native opt_alb=1 leaves ALBOLD unchanged.
        } else {
            float sc_in[4] = { swemx, qsnow, dt, albold };
            float sc_out[5];
            d_snowalb_class(sc_in, sc_out);
            albsnd[0] = sc_out[1]; albsnd[1] = sc_out[2];
            albsni[0] = sc_out[3]; albsni[1] = sc_out[4];
            albold = sc_out[0];
        }

        float ga_in[14] = { p[5], p[6], p[7], p[8], p[9], p[10],
                            fsno, p[37], albsnd[0], albsnd[1],
                            albsni[0], albsni[1], cosz, tg };
        float ga_out[4];
        d_groundalb(ga_in, ist, ga_out);
        albgrd[0] = ga_out[0]; albgrd[1] = ga_out[1];
        albgri[0] = ga_out[2]; albgri[1] = ga_out[3];

        for (int ib = 1; ib <= 2; ++ib) {
            float ts_in[33] = {
                p[19], p[20], p[21], p[22], p[23],
                cosz, vai, fwet, tv, fveg,
                albgrd[0], albgrd[1], albgri[0], albgri[1],
                rho[0], rho[1], tau[0], tau[1],
                fabd[0], fabd[1], albd[0], albd[1],
                ftdd[0], ftdd[1], ftid[0], ftid[1],
                gdir, frevd[0], frevd[1], fregd[0], fregd[1], bgap, wgap };
            float ts_out[15];
            if (crown) d_twostream_crown(ts_in, ib, 0, ts_out, crown);
            else if (dense) d_twostream_dense(ts_in, ib, 0, ts_out);
            else d_twostream(ts_in, ib, 0, ts_out);
            fabd[0] = ts_out[0]; fabd[1] = ts_out[1];
            albd[0] = ts_out[2]; albd[1] = ts_out[3];
            ftdd[0] = ts_out[4]; ftdd[1] = ts_out[5];
            ftid[0] = ts_out[6]; ftid[1] = ts_out[7];
            gdir = ts_out[8];
            frevd[0] = ts_out[9];  frevd[1] = ts_out[10];
            fregd[0] = ts_out[11]; fregd[1] = ts_out[12];
            bgap = ts_out[13]; wgap = ts_out[14];

            float ti_in[33] = {
                p[19], p[20], p[21], p[22], p[23],
                cosz, vai, fwet, tv, fveg,
                albgrd[0], albgrd[1], albgri[0], albgri[1],
                rho[0], rho[1], tau[0], tau[1],
                fabi[0], fabi[1], albi[0], albi[1],
                ftdi[0], ftdi[1], ftii[0], ftii[1],
                gdir, frevi[0], frevi[1], fregi[0], fregi[1], bgap, wgap };
            if (crown) d_twostream_crown(ti_in, ib, 1, ts_out, crown);
            else if (dense) d_twostream_dense(ti_in, ib, 1, ts_out);
            else d_twostream(ti_in, ib, 1, ts_out);
            fabi[0] = ts_out[0]; fabi[1] = ts_out[1];
            albi[0] = ts_out[2]; albi[1] = ts_out[3];
            ftdi[0] = ts_out[4]; ftdi[1] = ts_out[5];
            ftii[0] = ts_out[6]; ftii[1] = ts_out[7];
            gdir = ts_out[8];
            frevi[0] = ts_out[9];  frevi[1] = ts_out[10];
            fregi[0] = ts_out[11]; fregi[1] = ts_out[12];
            bgap = ts_out[13]; wgap = ts_out[14];
        }

        float ext = FMUL(FDIV(gdir, cosz),
                         FSQRT(FSUB(FSUB(K_ONE, rho[0]), tau[0])));
        fsun = FDIV(FSUB(K_ONE, glibc_expf(FMUL(-ext, vai))),
                    f_max(FMUL(ext, vai), mpe));
        ext = fsun;
        fsun = (ext < K_P01) ? K_ZERO : ext;
    }

    o[0] = fage; o[1] = albold; o[2] = tauss; o[3] = fsun;
    o[4] = bgap; o[5] = wgap;
    const float *v[15] = { albgrd, albgri, albd, albi, fabd, fabi,
                           ftdd, ftid, ftii, frevd, frevi, fregd, fregi,
                           albsnd, albsni };
    for (int k = 0; k < 15; ++k) { o[6 + 2 * k] = v[k][0]; o[7 + 2 * k] = v[k][1]; }
}

// --------------------------------------------------------------------------
// one kernel per leaf; row-major [n, nin] in, [n, nout] out
// --------------------------------------------------------------------------
__device__ void d_radiation_selected(const float *p, int ist, float *o, bool night_age=false,
                            const float *bats=nullptr, bool dense=false, const float *crown=nullptr, const float *snow=nullptr,
                            const float *absorption=nullptr, float *heating=nullptr)
{
    float alb[36];
    d_albedo_selected(p, ist, alb, night_age, bats, dense, crown, snow);

    const float elai = p[26], esai = p[27];
    const float *solad = p + 52, *solai = p + 54;

    // :2786-2790
    const float fsun = alb[3];
    const float fsha = FSUB(K_ONE, fsun);
    const float laisun = FMUL(elai, fsun);
    const float laisha = FMUL(elai, fsha);
    const float vai = FADD(elai, esai);

    // :2793-2799.  ALBEDO's outputs are laid out
    // fage albold tauss fsun bgap wgap then 15 two-element vectors in the
    // order albgrd albgri albd albi fabd fabi ftdd ftid ftii frevd frevi
    // fregd fregi albsnd albsni, so vector k occupies alb[6+2k], alb[7+2k].
    float s[37];
    s[0] = K_MPE; s[1] = fsun; s[2] = fsha; s[3] = elai; s[4] = vai;
    s[5] = laisun; s[6] = laisha;
    s[7] = solad[0]; s[8] = solad[1];
    s[9] = solai[0]; s[10] = solai[1];
    // SURRAD's order is fabd fabi ftdd ftid ftii albgrd albgri albd albi
    // frevd frevi fregd fregi -- ALBEDO's vectors 4,5,6,7,8,0,1,2,3,9,10,11,12.
    const int order[13] = { 4, 5, 6, 7, 8, 0, 1, 2, 3, 9, 10, 11, 12 };
    for (int k = 0; k < 13; ++k) {
        s[11 + 2 * k] = alb[6 + 2 * order[k]];
        s[12 + 2 * k] = alb[7 + 2 * order[k]];
    }

    float sr[8];
    d_surrad(s, sr);
    if(heating){
        // Fluxes reaching the snow surface after the canopy's multiple
        // reflections, using the same FTDD/FTID/FTII as SURRAD.
        float sum=0.f;
        for(int j=0;j<3;++j){float h=0.f;for(int b=0;b<2;++b){
            float direct=solad[b]*alb[18+b];
            float diffuse=solad[b]*alb[20+b]+solai[b]*alb[22+b];
            h+=p[31]*(direct*absorption[j*4+b]+diffuse*absorption[j*4+2+b]);
        }heating[j]=h;sum+=h;}
        heating[3]=fmaxf(0.f,sr[3]-sum);
        for(int j=4;j<7;++j)heating[j]=0.f;
    }

    o[0] = fsun; o[1] = laisun; o[2] = laisha;
    o[3] = sr[0];  // PARSUN
    o[4] = sr[1];  // PARSHA
    o[5] = sr[2];  // SAV
    o[6] = sr[3];  // SAG
    o[7] = sr[4];  // FSA
    o[8] = sr[5];  // FSR
    o[9] = sr[6];  // FSRV
    o[10] = sr[7]; // FSRG
    o[11] = alb[32]; o[12] = alb[33];   // ALBSND
    o[13] = alb[34]; o[14] = alb[35];   // ALBSNI
    o[15] = alb[4];  o[16] = alb[5];    // BGAP, WGAP
    o[17] = alb[1];  o[18] = alb[2];    // ALBOLD, TAUSS
}

// Existing 56/19-word RADIATION ABI plus a separate actual parameter slab.
// Setup selects the entry point; no host work or extra albedo launch per column.
extern "C" __global__ void noahmp_radiation_batch_bats(const float *in,
    const int *ist, const float *bats, float *out, int n, int night_age)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d_radiation_selected(in + (size_t)i * R_NIN, ist[i],
                    out + (size_t)i * R_NOUT, night_age != 0,
                    bats + (size_t)i * 7);
}

// Native leaf replay ABI: f32[n,9] (seven parameters, COSZ, FAGE), f32[n,4].
extern "C" __global__ void noahmp_rad_snowalb_bats(const float *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float *p = in + (size_t)i * 9;
        d_snowalb_bats(p, p[7], p[8], out + (size_t)i * 4);
    }
}

// Native RAD2 works with either the CLASS or actual BATS albedo parameters.
extern "C" __global__ void noahmp_radiation_batch_dense(const float *in,
    const int *ist, const float *bats, float *out, int n, int night_age)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d_radiation_selected(in + (size_t)i * R_NIN, ist[i],
                    out + (size_t)i * R_NOUT, night_age != 0,
                    bats == nullptr ? nullptr : bats + (size_t)i * 7, true);
}

extern "C" __global__ void noahmp_radiation_batch_crown(const float *in,
    const int *ist, const float *bats, const float *crown, float *out, int n, int night_age)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d_radiation_selected(in + (size_t)i * R_NIN, ist[i],
                    out + (size_t)i * R_NOUT, night_age != 0,
                    bats == nullptr ? nullptr : bats + (size_t)i * 7,
                    false, crown + (size_t)i * 3);
}

// Independently written snow spectrum, shared existing canopy/ground coupling.
extern "C" __global__ void noahmp_radiation_batch_snow_papers(const float *in,
    const int *ist, const float *snow, const float *absorption, float *heating,
    const float *crown, float *out,
    int n, int night_age, int opt_rad)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) d_radiation_selected(in+(size_t)i*R_NIN,ist[i],
        out+(size_t)i*R_NOUT,night_age!=0,nullptr,opt_rad==2,
        crown?crown+(size_t)i*3:nullptr,snow+(size_t)i*6,
        absorption+(size_t)i*28,heating+(size_t)i*7);
}
