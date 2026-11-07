// Existing WOOF complex and backscatter helpers from refl.cu.
#define REFL_NRBINS 50



#define RPI 3.1415926535897932384626434

#define R_D 287.0



#define RXAM_R (RPI * 997.0 / 6.0)
#define RXAM_S (100.0 * RPI / 6.0)



#define RGAMMA_4 6.0
#define RGAMMA_7 720.0

#define RPI5 (3.14159 * 3.14159 * 3.14159 * 3.14159 * 3.14159)
#define RLAMDA4 (0.10 * 0.10 * 0.10 * 0.10)

#define RMELT_OUTSIDE 0.9



struct cd { double re, im; };

__device__ __forceinline__ cd cd_make(double re, double im)
{
    cd c; c.re = re; c.im = im; return c;
}

__device__ __forceinline__ cd cd_add(cd a, cd b)
{
    return cd_make(a.re + b.re, a.im + b.im);
}

__device__ __forceinline__ cd cd_sub(cd a, cd b)
{
    return cd_make(a.re - b.re, a.im - b.im);
}

__device__ __forceinline__ cd cd_mul(cd a, cd b)
{
    return cd_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

__device__ __forceinline__ cd cd_div(cd a, cd b)
{
    double d = b.re * b.re + b.im * b.im;
    return cd_make((a.re * b.re + a.im * b.im) / d,
                   (a.im * b.re - a.re * b.im) / d);
}

__device__ __forceinline__ double cd_abs(cd a)
{
    return hypot(a.re, a.im);
}


__device__ __forceinline__ cd cd_sqrt(cd a)
{
    double m = cd_abs(a);
    double re = sqrt(fmax(0.5 * (m + a.re), 0.0));
    double im = sqrt(fmax(0.5 * (m - a.re), 0.0));
    return cd_make(re, a.im < 0.0 ? -im : im);
}


__device__ __forceinline__ cd cd_log(cd a)
{
    return cd_make(log(cd_abs(a)), atan2(a.im, a.re));
}

__device__ __forceinline__ cd cd_scale(double s, cd a)
{
    return cd_make(s * a.re, s * a.im);
}







__device__ cd refl_maxwellgarnett(double vol1, double vol2, double vol3,
                                  cd m1, cd m2, cd m3)
{
    cd m1t = cd_mul(m1, m1);
    cd m2t = cd_mul(m2, m2);
    cd m3t = cd_mul(m3, m3);
    cd d2 = cd_sub(m2t, m1t);
    cd d3 = cd_sub(m3t, m1t);
    cd one = cd_make(1.0, 0.0);
    cd beta2 = cd_mul(cd_div(cd_scale(2.0, m1t), d2),
                      cd_sub(cd_mul(cd_div(m2t, d2),
                                    cd_log(cd_div(m2t, m1t))), one));
    cd beta3 = cd_mul(cd_div(cd_scale(2.0, m1t), d3),
                      cd_sub(cd_mul(cd_div(m3t, d3),
                                    cd_log(cd_div(m3t, m1t))), one));
    cd num = cd_add(cd_scale(1.0 - vol2 - vol3, m1t),
                    cd_add(cd_scale(vol2, cd_mul(beta2, m2t)),
                           cd_scale(vol3, cd_mul(beta3, m3t))));
    cd den = cd_add(cd_make(1.0 - vol2 - vol3, 0.0),
                    cd_add(cd_scale(vol2, beta2), cd_scale(vol3, beta3)));
    return cd_sqrt(cd_div(num, den));
}







__device__ cd refl_m_mix_nested(cd m_a, cd m_i, cd m_w, double volair,
                                double volice, double volwater)
{
    double vol1 = volice / fmax(volice + volwater, 1.0e-10);
    double vol2 = 1.0 - vol1;
    cd mtmp = refl_maxwellgarnett(vol2, 0.0, vol1, m_w, m_a, m_i);
    return refl_maxwellgarnett(1.0 - volair, volair, 0.0,
                               mtmp, m_a, cd_scale(2.0, m_a));
}



__device__ double refl_rayleigh_soak_wetgraupel(
    double x_g, double a_geo, double b_geo, double fmelt,
    double meltratio_outside, cd m_w, cd m_i)
{
    double fm = fmin(fmax(fmelt, 0.0), 1.0);
    double mra = fmin(fmax(meltratio_outside, 0.0), 1.0);
    mra = mra + (1.0 - mra) * fm;
    double x_w = x_g * fm;
    double d_g = a_geo * pow(x_g, b_geo);
    if (d_g < 1.0e-12) return 0.0;
    double vg = RPI / 6.0 * d_g * d_g * d_g;
    double rhog = fmin(fmax(x_g / vg, 10.0), 900.0);
    vg = x_g / rhog;
    double grenz = 1.0 - rhog / 1000.0;
    double volg;
    if (mra <= grenz) {
        volg = vg * (1.0 - mra * fm);
    } else {
        double fmgrenz = (900.0 - rhog)
                       / (mra * 900.0 - rhog + 900.0 * rhog / 1000.0);
        volg = (fm <= fmgrenz) ? (1.0 - mra * fm) * vg
                               : (x_g - x_w) / 900.0 + x_w / 1000.0;
    }
    double d_large = pow(6.0 / RPI * volg, 1.0 / 3.0);
    double volice = (x_g - x_w) / (volg * 900.0);
    double volwater = x_w / (1000.0 * volg);
    double volair = 1.0 - volice - volwater;
    cd m_core = refl_m_mix_nested(cd_make(1.0, 0.0), m_i, m_w,
                                  volair, volice, volwater);
    cd m2 = cd_mul(m_core, m_core);
    double kfac = cd_abs(cd_div(cd_sub(m2, cd_make(1.0, 0.0)),
                                cd_add(m2, cd_make(2.0, 0.0))));
    double d3 = d_large * d_large * d_large;
    return kfac * kfac * RPI5 * d3 * d3 / RLAMDA4;
}
