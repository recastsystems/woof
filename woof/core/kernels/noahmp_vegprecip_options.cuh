// WRF v4 Noah-MP 848f54ad3d28c4303151fe5ad83724e232694422.
// PHENOLOGY:1255-1386, complete active prescribed-input-LAI branch.
// WRF public-domain notice and source hashes: tools/noahmp_options_wrf461_oracle/source_pin.json.
// Reuses the unchanged default image's ordered arithmetic and libm helpers.
extern "C" __global__ void noahmp_phenology_input_lai(
    int n,
    const float *__restrict__ input_lai,
    const float *__restrict__ input_sai,
    int dynamic,
    const int   *__restrict__ vegtyp,
    const int   *__restrict__ yearlen,
    const int   *__restrict__ iswater,
    const int   *__restrict__ isbarren,
    const int   *__restrict__ isice,
    const int   *__restrict__ urban_flag,
    const float *__restrict__ snowh_a,
    const float *__restrict__ tv_a,
    const float *__restrict__ lat_a,
    const float *__restrict__ julian_a,
    const float *__restrict__ hvt_a,
    const float *__restrict__ hvb_a,
    const float *__restrict__ tmin_a,
    const float *__restrict__ laim_a,      // n x 12, row major
    const float *__restrict__ saim_a,
    float *__restrict__ lai_o,
    float *__restrict__ sai_o,
    float *__restrict__ elai_o,
    float *__restrict__ esai_o,
    float *__restrict__ igs_o,
    float *__restrict__ fb_o)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;

    const float snowh  = snowh_a[c];
    const float tv     = tv_a[c];
    const float lat    = lat_a[c];
    const float julian = julian_a[c];
    const float hvt    = hvt_a[c];
    const float hvb    = hvb_a[c];
    const float tmin   = tmin_a[c];
    const int   ylen   = yearlen[c];

    // WRF v4 PHENOLOGY:1319-1322, CROPTYPE=0, DVEG=7/8/9.
    // LAI is the actual carried input, never the monthly parameter table.
    float lai = input_lai[c];
    float sai = dynamic ? input_sai[c] : vp_fmax(0.05f, __fmul_rn(0.1f, lai));
    if (!dynamic && lai < 0.05f) sai = 0.0f;

    if (sai < 0.05f) sai = 0.0f;
    if (lai < 0.05f || sai == 0.0f) lai = 0.0f;

    if (vegtyp[c] == iswater[c] || vegtyp[c] == isbarren[c] ||
        vegtyp[c] == isice[c]   || urban_flag[c] != 0) {
        lai = 0.0f;
        sai = 0.0f;
    }

    float span = __fsub_rn(hvt, hvb);
    float db = vp_fmin(vp_fmax(__fsub_rn(snowh, hvb), 0.0f), span);
    float fb = __fdiv_rn(db, vp_fmax(1.0e-06f, span));

    if (hvt > 0.0f && hvt <= 1.0f) {
        float snowhc = __fmul_rn(hvt, vp_glibc_expf(-__fdiv_rn(snowh, 0.2f)));
        fb = __fdiv_rn(vp_fmin(snowh, snowhc), snowhc);
    }

    float elai = __fmul_rn(lai, __fsub_rn(1.0f, fb));
    float esai = __fmul_rn(sai, __fsub_rn(1.0f, fb));
    if (esai < 0.05f) esai = 0.0f;
    if (elai < 0.05f || esai == 0.0f) elai = 0.0f;

    float igs = (tv > tmin) ? 1.0f : 0.0f;

    lai_o[c]  = lai;
    sai_o[c]  = sai;
    elai_o[c] = elai;
    esai_o[c] = esai;
    igs_o[c]  = igs;
    fb_o[c]   = fb;
}


// Native CROPTYPE>0 PHENOLOGY, lines 1301-1383.
extern "C" __global__ void noahmp_phenology_crop(
    int n,
    const float *__restrict__ input_lai,
    const float *__restrict__ input_sai,
    int dynamic,
    const int* croptype,
    const int* pgs,
    const int   *__restrict__ vegtyp,
    const int   *__restrict__ yearlen,
    const int   *__restrict__ iswater,
    const int   *__restrict__ isbarren,
    const int   *__restrict__ isice,
    const int   *__restrict__ urban_flag,
    const float *__restrict__ snowh_a,
    const float *__restrict__ tv_a,
    const float *__restrict__ lat_a,
    const float *__restrict__ julian_a,
    const float *__restrict__ hvt_a,
    const float *__restrict__ hvb_a,
    const float *__restrict__ tmin_a,
    const float *__restrict__ laim_a,      // n x 12, row major
    const float *__restrict__ saim_a,
    float *__restrict__ lai_o,
    float *__restrict__ sai_o,
    float *__restrict__ elai_o,
    float *__restrict__ esai_o,
    float *__restrict__ igs_o,
    float *__restrict__ fb_o)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n || croptype[c] == 0) return;

    const float snowh  = snowh_a[c];
    const float tv     = tv_a[c];
    const float lat    = lat_a[c];
    const float julian = julian_a[c];
    const float hvt    = hvt_a[c];
    const float hvb    = hvb_a[c];
    const float tmin   = tmin_a[c];
    const int   ylen   = yearlen[c];

    // WRF v4 PHENOLOGY:1319-1322, CROPTYPE=0, DVEG=7/8/9.
    // LAI is the actual carried input, never the monthly parameter table.
    float lai = input_lai[c];
    float sai = dynamic ? input_sai[c] : vp_fmax(0.05f, __fmul_rn(0.1f, lai));
    if (!dynamic && lai < 0.05f) sai = 0.0f;

    float span = __fsub_rn(hvt, hvb);
    float db = vp_fmin(vp_fmax(__fsub_rn(snowh, hvb), 0.0f), span);
    float fb = __fdiv_rn(db, vp_fmax(1.0e-06f, span));

    if (hvt > 0.0f && hvt <= 1.0f) {
        float snowhc = __fmul_rn(hvt, vp_glibc_expf(-__fdiv_rn(snowh, 0.2f)));
        fb = __fdiv_rn(vp_fmin(snowh, snowhc), snowhc);
    }

    float elai = __fmul_rn(lai, __fsub_rn(1.0f, fb));
    float esai = __fmul_rn(sai, __fsub_rn(1.0f, fb));



    float igs = (pgs[c]>2 && pgs[c]<7) ? 1.0f : 0.0f;

    lai_o[c]  = lai;
    sai_o[c]  = sai;
    elai_o[c] = elai;
    esai_o[c] = esai;
    igs_o[c]  = igs;
    fb_o[c]   = fb;
}
