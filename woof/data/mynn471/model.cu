/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

// Float erf arithmetic from the permissively licensed Sun routine.
// Upstream: glibc-2.39/sysdeps/ieee754/flt-32/s_erff.c
__device__ float mynn471_erff(float x)
{
const float
tiny	    = 1e-30,
half=  5.0000000000e-01, /* 0x3F000000 */
one =  1.0000000000e+00, /* 0x3F800000 */
two =  2.0000000000e+00, /* 0x40000000 */
	/* c = (subfloat)0.84506291151 */
erx =  8.4506291151e-01, /* 0x3f58560b */
/*
 * Coefficients for approximation to  erf on [0,0.84375]
 */
efx =  1.2837916613e-01, /* 0x3e0375d4 */
pp0  =  1.2837916613e-01, /* 0x3e0375d4 */
pp1  = -3.2504209876e-01, /* 0xbea66beb */
pp2  = -2.8481749818e-02, /* 0xbce9528f */
pp3  = -5.7702702470e-03, /* 0xbbbd1489 */
pp4  = -2.3763017452e-05, /* 0xb7c756b1 */
qq1  =  3.9791721106e-01, /* 0x3ecbbbce */
qq2  =  6.5022252500e-02, /* 0x3d852a63 */
qq3  =  5.0813062117e-03, /* 0x3ba68116 */
qq4  =  1.3249473704e-04, /* 0x390aee49 */
qq5  = -3.9602282413e-06, /* 0xb684e21a */
/*
 * Coefficients for approximation to  erf  in [0.84375,1.25]
 */
pa0  = -2.3621185683e-03, /* 0xbb1acdc6 */
pa1  =  4.1485610604e-01, /* 0x3ed46805 */
pa2  = -3.7220788002e-01, /* 0xbebe9208 */
pa3  =  3.1834661961e-01, /* 0x3ea2fe54 */
pa4  = -1.1089469492e-01, /* 0xbde31cc2 */
pa5  =  3.5478305072e-02, /* 0x3d1151b3 */
pa6  = -2.1663755178e-03, /* 0xbb0df9c0 */
qa1  =  1.0642088205e-01, /* 0x3dd9f331 */
qa2  =  5.4039794207e-01, /* 0x3f0a5785 */
qa3  =  7.1828655899e-02, /* 0x3d931ae7 */
qa4  =  1.2617121637e-01, /* 0x3e013307 */
qa5  =  1.3637083583e-02, /* 0x3c5f6e13 */
qa6  =  1.1984500103e-02, /* 0x3c445aa3 */
/*
 * Coefficients for approximation to  erfc in [1.25,1/0.35]
 */
ra0  = -9.8649440333e-03, /* 0xbc21a093 */
ra1  = -6.9385856390e-01, /* 0xbf31a0b7 */
ra2  = -1.0558626175e+01, /* 0xc128f022 */
ra3  = -6.2375331879e+01, /* 0xc2798057 */
ra4  = -1.6239666748e+02, /* 0xc322658c */
ra5  = -1.8460508728e+02, /* 0xc3389ae7 */
ra6  = -8.1287437439e+01, /* 0xc2a2932b */
ra7  = -9.8143291473e+00, /* 0xc11d077e */
sa1  =  1.9651271820e+01, /* 0x419d35ce */
sa2  =  1.3765776062e+02, /* 0x4309a863 */
sa3  =  4.3456588745e+02, /* 0x43d9486f */
sa4  =  6.4538726807e+02, /* 0x442158c9 */
sa5  =  4.2900814819e+02, /* 0x43d6810b */
sa6  =  1.0863500214e+02, /* 0x42d9451f */
sa7  =  6.5702495575e+00, /* 0x40d23f7c */
sa8  = -6.0424413532e-02, /* 0xbd777f97 */
/*
 * Coefficients for approximation to  erfc in [1/.35,28]
 */
rb0  = -9.8649431020e-03, /* 0xbc21a092 */
rb1  = -7.9928326607e-01, /* 0xbf4c9dd4 */
rb2  = -1.7757955551e+01, /* 0xc18e104b */
rb3  = -1.6063638306e+02, /* 0xc320a2ea */
rb4  = -6.3756646729e+02, /* 0xc41f6441 */
rb5  = -1.0250950928e+03, /* 0xc480230b */
rb6  = -4.8351919556e+02, /* 0xc3f1c275 */
sb1  =  3.0338060379e+01, /* 0x41f2b459 */
sb2  =  3.2579251099e+02, /* 0x43a2e571 */
sb3  =  1.5367296143e+03, /* 0x44c01759 */
sb4  =  3.1998581543e+03, /* 0x4547fdbb */
sb5  =  2.5530502930e+03, /* 0x451f90ce */
sb6  =  4.7452853394e+02, /* 0x43ed43a7 */
sb7  = -2.2440952301e+01; /* 0xc1b38712 */


	int hx,ix,i;
	float R,S,P,Q,s,y,z,r;
	hx=__float_as_int(x);
	ix = hx&0x7fffffff;
	if(ix>=0x7f800000) {		/* erf(nan)=nan */
	    i = ((unsigned int)hx>>31)<<1;
	    return (float)(1-i)+one/x;	/* erf(+-inf)=+-1 */
	}

	if(ix < 0x3f580000) {		/* |x|<0.84375 */
	    if(ix < 0x31800000) { 	/* |x|<2**-28 */
	        if (ix < 0x04000000)
		  {
		    /* Avoid spurious underflow.  */
		    float ret = 0.0625f * (16.0f * x + (16.0f * efx) * x);
		    
		    return ret;
		  }
		return x + efx*x;
	    }
	    z = x*x;
	    r = pp0+z*(pp1+z*(pp2+z*(pp3+z*pp4)));
	    s = one+z*(qq1+z*(qq2+z*(qq3+z*(qq4+z*qq5))));
	    y = r/s;
	    return x + x*y;
	}
	if(ix < 0x3fa00000) {		/* 0.84375 <= |x| < 1.25 */
	    s = fabsf(x)-one;
	    P = pa0+s*(pa1+s*(pa2+s*(pa3+s*(pa4+s*(pa5+s*pa6)))));
	    Q = one+s*(qa1+s*(qa2+s*(qa3+s*(qa4+s*(qa5+s*qa6)))));
	    if(hx>=0) return erx + P/Q; else return -erx - P/Q;
	}
	if (ix >= 0x40c00000) {		/* inf>|x|>=6 */
	    if(hx>=0) return one-tiny; else return tiny-one;
	}
	x = fabsf(x);
 	s = one/(x*x);
	if(ix< 0x4036DB6E) {	/* |x| < 1/0.35 */
	    R=ra0+s*(ra1+s*(ra2+s*(ra3+s*(ra4+s*(
				ra5+s*(ra6+s*ra7))))));
	    S=one+s*(sa1+s*(sa2+s*(sa3+s*(sa4+s*(
				sa5+s*(sa6+s*(sa7+s*sa8)))))));
	} else {	/* |x| >= 1/0.35 */
	    R=rb0+s*(rb1+s*(rb2+s*(rb3+s*(rb4+s*(
				rb5+s*rb6)))));
	    S=one+s*(sb1+s*(sb2+s*(sb3+s*(sb4+s*(
				sb5+s*(sb6+s*sb7))))));
	}
	ix=__float_as_int(x);
	z=__int_as_float(ix&0xfffff000);
	r  =  mynn_glibc_expf(-z*z-(float)0.5625)*mynn_glibc_expf((z-x)*(z+x)+R/S);
	if(hx>=0) return one-r/x; else return  r/x-one;
}

#if GPUWM_WRF_EXACT
#define M471_CALL __noinline__
#define M471_LAUNCH
#else
#define M471_CALL __forceinline__
#define M471_LAUNCH __launch_bounds__(128,4)
#endif
typedef int integer;
typedef int logical;
typedef double doublereal;
template<class T> struct M471Ptr;
template<class T> struct M471Ref {
 T *p; int stride;
 __device__ operator T() const{return *p;}
 __device__ T operator=(T x){*p=x;return x;}
 __device__ T operator=(const M471Ref&x){return *this=(T)x;}
 template<class U> __device__ T operator+=(U x){return *this=*p+x;}
 template<class U> __device__ T operator-=(U x){return *this=*p-x;}
 template<class U> __device__ T operator*=(U x){return *this=*p*x;}
 template<class U> __device__ T operator/=(U x){return *this=*p/x;}
 __device__ M471Ptr<T> operator&() const;
};
template<class T> struct M471Ptr {
 T *p; int stride;
 __device__ M471Ptr(T*q=nullptr,int s=1):p(q),stride(s){}
 __device__ T& operator*()const{return *p;}
 __device__ M471Ref<T> operator[](int i)const{return {p+i*stride,stride};}
 __device__ M471Ptr operator+(int i)const{return {p+i*stride,stride};}
 __device__ M471Ptr operator-(int i)const{return {p-i*stride,stride};}
 __device__ M471Ptr& operator+=(int i){p+=i*stride;return *this;}
 __device__ M471Ptr& operator-=(int i){p-=i*stride;return *this;}
 __device__ M471Ptr& operator--(){p-=stride;return *this;}
 __device__ M471Ptr& operator++(){p+=stride;return *this;}
};
template<class T> __device__ M471Ptr<T> M471Ref<T>::operator&()const{return {p,stride};}
template<class T> __device__ M471Ptr<T> m471_array(unsigned char*base,int offset,int n){
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 return {(T*)(base+(unsigned long long)offset*n)+c,n};
}

#define TRUE_ 1
#define FALSE_ 0
#define abs(x) ((x)>=0 ? (x) : -(x))
#define dabs(x) abs(x)
#define dmin(a,b) ((a)<=(b)?(a):(b))
#define dmax(a,b) ((a)>=(b)?(a):(b))
#define min(a,b) dmin(a,b)
#define max(a,b) dmax(a,b)
__device__ float pow_dd(M471Ptr<double> a,M471Ptr<double> b){return mynn_glibc_powf((float)*a,(float)*b);}
__device__ double d_sign(M471Ptr<double> a,M471Ptr<double> b){return *b>=0.?fabs(*a):-fabs(*a);}
__device__ float r_lg10(M471Ptr<float> a){return mynn_glibc_log10f(*a);}
__device__ float erf_(M471Ptr<float> a){
#if GPUWM_WRF_EXACT
 return mynn471_erff(*a);
#else
 return erff(*a);
#endif
}
__device__ float m471_sqrt(float x){return sqrtf(x);}
__device__ double m471_sqrt(double x){return sqrt(x);}
__device__ float m471_exp(float x){return mynn_glibc_expf(x);}
__device__ double m471_exp(double x){return exp(x);}
__device__ float m471_log(float x){return mynn_glibc_logf(x);}
__device__ float m471_atan(float x){return mynn_glibc_atanf(x);}
__device__ float m471_tanh(float x){return mynn_tanhf(x);}
#define sqrt(x) m471_sqrt(x)
#define exp(x) m471_exp(x)
#define log(x) m471_log(x)
#define atan(x) m471_atan(x)
#define tanh(x) m471_tanh(x)
// WRF 4.7.1 public-domain MYNN numerical source; generated offline.



template<int M471_PHASE>
__device__ M471_CALL /* Subroutine */ int mynnedmf_(unsigned char *m471_work, int m471_n, M471Ptr<integer> i__, M471Ptr<integer> j, M471Ptr<integer> initflag, 
	M471Ptr<logical> restart, M471Ptr<logical> cycling, M471Ptr<real> delt, M471Ptr<real> dz1, M471Ptr<real> dx, 
	M471Ptr<real> u1, M471Ptr<real> v1, M471Ptr<real> w1, M471Ptr<real> th1, M471Ptr<real> sqv1, M471Ptr<real> sqc1, M471Ptr<real> sqi1, M471Ptr<real> sqs1, M471Ptr<real> qnc1, M471Ptr<real> qni1, M471Ptr<real> qnwfa1, M471Ptr<real> qnifa1,
	 M471Ptr<real> qnbca1, M471Ptr<real> ozone1, M471Ptr<real> p1, M471Ptr<real> ex1, M471Ptr<real> rho1, M471Ptr<real> tk1, M471Ptr<real> xland, M471Ptr<real> ts, M471Ptr<real> qsfc, M471Ptr<real> ps, M471Ptr<real> ust, M471Ptr<real> ch,
	 M471Ptr<real> hfx, M471Ptr<real> qfx, M471Ptr<real> znt, M471Ptr<real> wspd, M471Ptr<real> uoce, M471Ptr<real> voce, 
	M471Ptr<real> qke1, M471Ptr<real> qke_adv1__, M471Ptr<real> el1, M471Ptr<real> sh1, M471Ptr<real> sm1, M471Ptr<real> kh1, M471Ptr<real> km1, M471Ptr<integer> nchem, M471Ptr<integer> kdvel, M471Ptr<integer> ndvel, M471Ptr<real> chem, M471Ptr<real> vdep, M471Ptr<real> frp, M471Ptr<real> emis_ant_no__, M471Ptr<logical> mix_chem__,
	 M471Ptr<logical> enh_mix__, M471Ptr<logical> rrfs_sd__, M471Ptr<logical> smoke_dbg__, M471Ptr<real> tsq1, M471Ptr<real> qsq1, M471Ptr<real> cov1, M471Ptr<real> du1, M471Ptr<real> dv1, M471Ptr<real> dth1, M471Ptr<real> dqv1, M471Ptr<real> dqc1, M471Ptr<real> dqi1, M471Ptr<real> dqnc1, M471Ptr<real> dqni1, M471Ptr<real> dqs1, 
	M471Ptr<real> dqnwfa1, M471Ptr<real> dqnifa1, M471Ptr<real> dqnbca1, M471Ptr<real> dozone1, M471Ptr<real> rthraten1, M471Ptr<real> pblh, M471Ptr<integer> kpbl, M471Ptr<real> maxwidth, M471Ptr<real> maxmf, 
	M471Ptr<real> ztop_plume__, M471Ptr<real> dqke1, M471Ptr<real> qwt1, M471Ptr<real> qshear1, M471Ptr<real> qbuoy1, M471Ptr<real> qdiss1, M471Ptr<real> qc_bl1__, M471Ptr<real> qi_bl1__, M471Ptr<real> cldfra_bl1__, M471Ptr<logical> bl_mynn_tkeadvect__, M471Ptr<integer> tke_budget__, 
	M471Ptr<integer> bl_mynn_cloudpdf__, M471Ptr<integer> bl_mynn_mixlength__, M471Ptr<integer> icloud_bl__, M471Ptr<real> closure, M471Ptr<integer> bl_mynn_edmf__, M471Ptr<integer> bl_mynn_edmf_mom__, M471Ptr<integer> bl_mynn_edmf_tke__, M471Ptr<integer> bl_mynn_mixscalars__, M471Ptr<integer> bl_mynn_output__, M471Ptr<integer> bl_mynn_cloudmix__, M471Ptr<integer> bl_mynn_mixqt__, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_qt1__, M471Ptr<real> edmf_thl1__, M471Ptr<real> edmf_ent1__, 
	M471Ptr<real> edmf_qc1__, M471Ptr<real> sub_thl1__, M471Ptr<real> sub_sqv1__, M471Ptr<real> det_thl1__, M471Ptr<real> det_sqv1__, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__, M471Ptr<logical> flag_qc__, M471Ptr<logical> flag_qi__, M471Ptr<logical> flag_qnc__, M471Ptr<logical> flag_qni__, M471Ptr<logical> flag_qs__, M471Ptr<logical> flag_qnwfa__, M471Ptr<logical> flag_qnifa__, M471Ptr<logical> flag_qnbca__, M471Ptr<logical> flag_ozone__, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<integer> errflg);
__device__ M471_CALL /* Subroutine */ int mym_initialize__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> xland,
	 M471Ptr<real> dz, M471Ptr<real> dx, M471Ptr<real> zw, M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> thl, M471Ptr<real> qw, 
	M471Ptr<real> zi, M471Ptr<real> theta, M471Ptr<real> thv, M471Ptr<real> thlv, M471Ptr<real> sh, M471Ptr<real> sm, 
	M471Ptr<real> ust, M471Ptr<real> rmol, M471Ptr<real> el, M471Ptr<real> qke, M471Ptr<real> tsq, M471Ptr<real> qsq, 
	M471Ptr<real> cov, M471Ptr<real> psig_bl__, M471Ptr<real> cldfra_bl1__, M471Ptr<integer> bl_mynn_mixlength__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w_dd1__, M471Ptr<real> edmf_a_dd1__, M471Ptr<logical> initialize_qke__, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__);
__device__ M471_CALL /* Subroutine */ int mym_level2__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dz, M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> thl, M471Ptr<real> thv, M471Ptr<real> thlv, M471Ptr<real> qw, M471Ptr<real> ql, 
	M471Ptr<real> vt, M471Ptr<real> vq, M471Ptr<real> dtl, M471Ptr<real> dqw, M471Ptr<real> dtv, M471Ptr<real> gm, M471Ptr<real> gh, M471Ptr<real> sm, M471Ptr<real> sh);
__device__ M471_CALL /* Subroutine */ int mym_length__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> xland, 
	M471Ptr<real> dz, M471Ptr<real> dx, M471Ptr<real> zw, M471Ptr<real> rmol, M471Ptr<real> flt, M471Ptr<real> fltv, M471Ptr<real> flq, M471Ptr<real> vt, M471Ptr<real> vq, M471Ptr<real> u1, M471Ptr<real> v1, M471Ptr<real> qke, M471Ptr<real> dtv, 
	M471Ptr<real> el, M471Ptr<real> pblh, M471Ptr<real> theta, M471Ptr<real> qkw, M471Ptr<real> psig_bl__, M471Ptr<real> cldfra_bl1__, M471Ptr<integer> bl_mynn_mixlength__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w_dd1__, M471Ptr<real> edmf_a_dd1__);
__device__ M471_CALL /* Subroutine */ int boulac_length0__(unsigned char *m471_work, int m471_n, M471Ptr<integer> k, M471Ptr<integer> kts, M471Ptr<integer> kte, 
	M471Ptr<real> zw, M471Ptr<real> dz, M471Ptr<real> qtke, M471Ptr<real> theta, M471Ptr<real> lb1, M471Ptr<real> lb2);
__device__ M471_CALL /* Subroutine */ int boulac_length__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> zw, 
	M471Ptr<real> dz, M471Ptr<real> qtke, M471Ptr<real> theta, M471Ptr<real> lb1, M471Ptr<real> lb2);
__device__ M471_CALL /* Subroutine */ int mym_turbulence__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> xland,
	 M471Ptr<real> closure, M471Ptr<real> dz, M471Ptr<real> dx, M471Ptr<real> zw, M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> thl, M471Ptr<real> thv, M471Ptr<real> thlv, M471Ptr<real> ql, M471Ptr<real> qw, M471Ptr<real> qke, M471Ptr<real> tsq, 
	M471Ptr<real> qsq, M471Ptr<real> cov, M471Ptr<real> vt, M471Ptr<real> vq, M471Ptr<real> rmol, M471Ptr<real> flt, M471Ptr<real> fltv, M471Ptr<real> flq, M471Ptr<real> pblh, M471Ptr<real> theta, M471Ptr<real> sh, M471Ptr<real> sm, M471Ptr<real> el, M471Ptr<real> dfm, M471Ptr<real> dfh, M471Ptr<real> dfq, M471Ptr<real> tcd, M471Ptr<real> qcd, M471Ptr<real> pdk, 
	M471Ptr<real> pdt, M471Ptr<real> pdq, M471Ptr<real> pdc, M471Ptr<real> qwt1, M471Ptr<real> qshear1, M471Ptr<real> qbuoy1, M471Ptr<real> qdiss1, M471Ptr<integer> tke_budget__, M471Ptr<real> psig_bl__, M471Ptr<real> psig_shcu__, M471Ptr<real> cldfra_bl1__, M471Ptr<integer> bl_mynn_mixlength__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w_dd1__, M471Ptr<real> edmf_a_dd1__, 
	M471Ptr<real> tkeprod_dn__, M471Ptr<real> tkeprod_up__, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__);
__device__ M471_CALL /* Subroutine */ int mym_predict__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> closure, 
	M471Ptr<real> delt, M471Ptr<real> dz, M471Ptr<real> ust, M471Ptr<real> flt, M471Ptr<real> flq, M471Ptr<real> pmz, 
	M471Ptr<real> phh, M471Ptr<real> el, M471Ptr<real> dfq, M471Ptr<real> rho, M471Ptr<real> pdk, M471Ptr<real> pdt, M471Ptr<real> pdq, M471Ptr<real> pdc, M471Ptr<real> qke, M471Ptr<real> tsq, M471Ptr<real> qsq, M471Ptr<real> cov, M471Ptr<real> s_aw1__, M471Ptr<real> s_awqke1__, M471Ptr<integer> bl_mynn_edmf_tke__, M471Ptr<real> qwt1, 
	M471Ptr<real> qdiss1, M471Ptr<integer> tke_budget__);
__device__ M471_CALL /* Subroutine */ int mym_condensation__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dx, 
	M471Ptr<real> dz, M471Ptr<real> zw, M471Ptr<real> xland, M471Ptr<real> thl, M471Ptr<real> qw, M471Ptr<real> qv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<real> qs, M471Ptr<real> p, M471Ptr<real> exner, M471Ptr<real> tsq, M471Ptr<real> qsq, 
	M471Ptr<real> cov, M471Ptr<real> sh, M471Ptr<real> el, M471Ptr<integer> bl_mynn_cloudpdf__, M471Ptr<real> qc_bl1__, M471Ptr<real> qi_bl1__, M471Ptr<real> cldfra_bl1__, M471Ptr<real> pblh1, M471Ptr<real> hfx1,
	 M471Ptr<real> vt, M471Ptr<real> vq, M471Ptr<real> th, M471Ptr<real> sgm, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__);
__device__ M471_CALL /* Subroutine */ int mynn_tendencies__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<integer> i__, M471Ptr<real> delt, M471Ptr<real> dz, M471Ptr<real> zw, M471Ptr<real> xland, M471Ptr<real> rho, M471Ptr<real> u, 
	M471Ptr<real> v, M471Ptr<real> th, M471Ptr<real> tk, M471Ptr<real> qv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<real> qs, 
	M471Ptr<real> qnc, M471Ptr<real> qni, M471Ptr<real> psfc, M471Ptr<real> p, M471Ptr<real> exner, M471Ptr<real> thl, 
	M471Ptr<real> sqv, M471Ptr<real> sqc, M471Ptr<real> sqi, M471Ptr<real> sqs, M471Ptr<real> sqw, M471Ptr<real> qnwfa, 
	M471Ptr<real> qnifa, M471Ptr<real> qnbca, M471Ptr<real> ozone, M471Ptr<real> ust, M471Ptr<real> flt, M471Ptr<real> flq, M471Ptr<real> flqv, M471Ptr<real> flqc, M471Ptr<real> wspd, M471Ptr<real> uoce, M471Ptr<real> voce, M471Ptr<real> tsq, M471Ptr<real> qsq, M471Ptr<real> cov, M471Ptr<real> tcd, M471Ptr<real> qcd, M471Ptr<real> dfm, M471Ptr<real> dfh, M471Ptr<real> dfq, M471Ptr<real> du, M471Ptr<real> dv, M471Ptr<real> dth, M471Ptr<real> dqv, M471Ptr<real> dqc, 
	M471Ptr<real> dqi, M471Ptr<real> dqs, M471Ptr<real> dqnc, M471Ptr<real> dqni, M471Ptr<real> dqnwfa, M471Ptr<real> dqnifa, M471Ptr<real> dqnbca, M471Ptr<real> dozone, M471Ptr<real> diss_heat__, M471Ptr<real> s_aw1__, 
	M471Ptr<real> s_awthl1__, M471Ptr<real> s_awqt1__, M471Ptr<real> s_awqv1__, M471Ptr<real> s_awqc1__, 
	M471Ptr<real> s_awu1__, M471Ptr<real> s_awv1__, M471Ptr<real> s_awqnc1__, M471Ptr<real> s_awqni1__, 
	M471Ptr<real> s_awqnwfa1__, M471Ptr<real> s_awqnifa1__, M471Ptr<real> s_awqnbca1__, M471Ptr<real> sd_aw1__, M471Ptr<real> sd_awthl1__, M471Ptr<real> sd_awqt1__, M471Ptr<real> sd_awqv1__, M471Ptr<real> sd_awqc1__, M471Ptr<real> sd_awqi1__, M471Ptr<real> sd_awqnc1__, M471Ptr<real> sd_awqni1__, 
	M471Ptr<real> sd_awqnwfa1__, M471Ptr<real> sd_awqnifa1__, M471Ptr<real> sd_awu1__, M471Ptr<real> sd_awv1__, M471Ptr<real> sub_thl__, M471Ptr<real> sub_sqv__, M471Ptr<real> sub_u__, M471Ptr<real> sub_v__, M471Ptr<real> det_thl__, M471Ptr<real> det_sqv__, M471Ptr<real> det_sqc__, M471Ptr<real> det_u__, M471Ptr<real> det_v__, M471Ptr<logical> flag_qc__, M471Ptr<logical> flag_qi__, 
	M471Ptr<logical> flag_qnc__, M471Ptr<logical> flag_qni__, M471Ptr<logical> flag_qs__, M471Ptr<logical> flag_qnwfa__, M471Ptr<logical> flag_qnifa__, M471Ptr<logical> flag_qnbca__, M471Ptr<logical> flag_ozone__, M471Ptr<real> cldfra_bl1__, M471Ptr<integer> bl_mynn_cloudmix__, 
	M471Ptr<integer> bl_mynn_mixqt__, M471Ptr<integer> bl_mynn_edmf__, M471Ptr<integer> bl_mynn_edmf_mom__, M471Ptr<integer> bl_mynn_mixscalars__);
__device__ M471_CALL /* Subroutine */ int moisture_check__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kte, M471Ptr<real> delt, M471Ptr<real> dp, 
	M471Ptr<real> exner, M471Ptr<real> qv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<real> qs, M471Ptr<real> th, M471Ptr<real> dqv, M471Ptr<real> dqc, M471Ptr<real> dqi, M471Ptr<real> dqs, M471Ptr<real> dth);
__device__ M471_CALL /* Subroutine */ int mynn_mix_chem__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<integer> i__,
	 M471Ptr<real> delt, M471Ptr<real> dz, M471Ptr<real> pblh, M471Ptr<integer> nchem, M471Ptr<integer> kdvel, 
	M471Ptr<integer> ndvel, M471Ptr<real> chem1, M471Ptr<real> vd1, M471Ptr<real> rho, M471Ptr<real> flt, M471Ptr<real> tcd, M471Ptr<real> qcd, M471Ptr<real> dfh, M471Ptr<real> s_aw1__, M471Ptr<real> s_awchem1__, M471Ptr<real> emis_ant_no__, M471Ptr<real> frp, M471Ptr<logical> rrfs_sd__, M471Ptr<logical> enh_mix__, 
	M471Ptr<logical> smoke_dbg__);
__device__ M471_CALL /* Subroutine */ int retrieve_exchange_coeffs__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, 
	M471Ptr<real> dfm, M471Ptr<real> dfh, M471Ptr<real> dz, M471Ptr<real> km1, M471Ptr<real> kh1);
__device__ M471_CALL /* Subroutine */ int tridiag_(unsigned char *m471_work, int m471_n, M471Ptr<integer> n, M471Ptr<real> a, M471Ptr<real> b, M471Ptr<real> c__, M471Ptr<real> d__);
__device__ M471_CALL /* Subroutine */ int tridiag2_(unsigned char *m471_work, int m471_n, M471Ptr<integer> kte, M471Ptr<real> a, M471Ptr<real> b, M471Ptr<real> c__, 
	M471Ptr<real> d__, M471Ptr<real> x);
__device__ M471_CALL /* Subroutine */ int tridiag3_(unsigned char *m471_work, int m471_n, M471Ptr<integer> kte, M471Ptr<real> a, M471Ptr<real> b, M471Ptr<real> c__, 
	M471Ptr<real> d__, M471Ptr<real> x);
__device__ M471_CALL /* Subroutine */ int get_pblh__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> pblh, M471Ptr<real> thv1, M471Ptr<real> qke1, M471Ptr<real> zw1, M471Ptr<real> dz1, M471Ptr<real> landsea, M471Ptr<integer> kpbl);
template<int M471_PHASE>
__device__ M471_CALL /* Subroutine */ int dmp_mf__(unsigned char *m471_work, int m471_n, M471Ptr<integer> i__, M471Ptr<integer> j, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dt, M471Ptr<real> zw1, M471Ptr<real> dz1, M471Ptr<real> p1, M471Ptr<real> rho1, M471Ptr<integer> momentum_opt__, M471Ptr<integer> tke_opt__, M471Ptr<integer> scalar_opt__, M471Ptr<real> u1, 
	M471Ptr<real> v1, M471Ptr<real> w1, M471Ptr<real> th1, M471Ptr<real> thl1, M471Ptr<real> thv1, M471Ptr<real> tk1, 
	M471Ptr<real> qt1, M471Ptr<real> qv1, M471Ptr<real> qc1, M471Ptr<real> qke1, M471Ptr<real> qnc1, M471Ptr<real> qni1, 
	M471Ptr<real> qnwfa1, M471Ptr<real> qnifa1, M471Ptr<real> qnbca1, M471Ptr<real> ex1, M471Ptr<real> vt1, M471Ptr<real> vq1, M471Ptr<real> sgm1, M471Ptr<real> ust, M471Ptr<real> flt, M471Ptr<real> fltv, M471Ptr<real> flq, M471Ptr<real> flqv, M471Ptr<real> pblh, M471Ptr<integer> kpbl, M471Ptr<real> dx, M471Ptr<real> landsea, M471Ptr<real> ts, 
	M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_qt1__, M471Ptr<real> edmf_thl1__,
	 M471Ptr<real> edmf_ent1__, M471Ptr<real> edmf_qc1__, M471Ptr<real> s_aw1__, M471Ptr<real> s_awthl1__,
	 M471Ptr<real> s_awqt1__, M471Ptr<real> s_awqv1__, M471Ptr<real> s_awqc1__, M471Ptr<real> s_awu1__, 
	M471Ptr<real> s_awv1__, M471Ptr<real> s_awqke1__, M471Ptr<real> s_awqnc1__, M471Ptr<real> s_awqni1__, 
	M471Ptr<real> s_awqnwfa1__, M471Ptr<real> s_awqnifa1__, M471Ptr<real> s_awqnbca1__, M471Ptr<real> sub_thl1__, M471Ptr<real> sub_sqv1__, M471Ptr<real> sub_u1__, M471Ptr<real> sub_v1__, M471Ptr<real> det_thl1__, M471Ptr<real> det_sqv1__, M471Ptr<real> det_sqc1__, M471Ptr<real> det_u1__, M471Ptr<real> det_v1__, M471Ptr<integer> nchem, M471Ptr<real> chem1, M471Ptr<real> s_awchem1__, M471Ptr<logical> mix_chem__, M471Ptr<real> qc_bl1__, M471Ptr<real> cldfra_bl1__, M471Ptr<real> qc_bl1_old__, 
	M471Ptr<real> cldfra_bl1_old__, M471Ptr<logical> f_qc__, M471Ptr<logical> f_qi__, M471Ptr<logical> f_qnc__, M471Ptr<logical> f_qni__, M471Ptr<logical> f_qnwfa__, M471Ptr<logical> f_qnifa__, 
	M471Ptr<logical> f_qnbca__, M471Ptr<real> psig_shcu__, M471Ptr<real> maxwidth, M471Ptr<integer> ktop, 
	M471Ptr<real> maxmf, M471Ptr<real> ztop, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__,
	 M471Ptr<real> tkeprod_up__, M471Ptr<real> el1);
__device__ M471_CALL /* Subroutine */ int condensation_edmf__(unsigned char *m471_work, int m471_n, M471Ptr<real> qt, M471Ptr<real> thl, M471Ptr<real> p, M471Ptr<real> zagl, M471Ptr<real> thv, M471Ptr<real> qc);
__device__ M471_CALL /* Subroutine */ int condensation_ddmf__(unsigned char *m471_work, int m471_n, M471Ptr<real> qt, M471Ptr<real> thl, M471Ptr<real> p, M471Ptr<real> zagl, M471Ptr<real> thv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<integer> debug_dd__);
__device__ M471_CALL /* Subroutine */ int condensation_edmf_r__(unsigned char *m471_work, int m471_n, M471Ptr<real> qt, M471Ptr<real> thl, M471Ptr<real> p, M471Ptr<real> zagl, M471Ptr<real> thv, M471Ptr<real> qc);
__device__ M471_CALL /* Subroutine */ int ddmp_mf__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dt, M471Ptr<real> dx,
	 M471Ptr<real> zw, M471Ptr<real> dz, M471Ptr<real> p, M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> th, M471Ptr<real> thl, 
	M471Ptr<real> thv, M471Ptr<real> tk, M471Ptr<real> qt, M471Ptr<real> qv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<real> qnc, M471Ptr<real> qni, M471Ptr<real> qnwfa, M471Ptr<real> qnifa, M471Ptr<real> qke, M471Ptr<real> rho, M471Ptr<real> exner, M471Ptr<real> qc_bl1__, M471Ptr<real> qi_bl1__, M471Ptr<real> cldfra_bl1__, M471Ptr<real> ust, 
	M471Ptr<real> flt, M471Ptr<real> flq, M471Ptr<real> fltv, M471Ptr<real> pblh, M471Ptr<integer> kpbl, M471Ptr<real> edmf_a_dd__, M471Ptr<real> edmf_w_dd__, M471Ptr<real> edmf_qt_dd__, M471Ptr<real> edmf_thl_dd__, M471Ptr<real> edmf_ent_dd__, M471Ptr<real> edmf_qc_dd__, M471Ptr<real> sd_aw__,
	 M471Ptr<real> sd_awthl__, M471Ptr<real> sd_awqt__, M471Ptr<real> sd_awqv__, M471Ptr<real> sd_awqc__, 
	M471Ptr<real> sd_awqi__, M471Ptr<real> sd_awqnc__, M471Ptr<real> sd_awqni__, M471Ptr<real> sd_awqnwfa__, M471Ptr<real> sd_awqnifa__, M471Ptr<real> sd_awu__, M471Ptr<real> sd_awv__, 
	M471Ptr<real> sd_awqke__, M471Ptr<real> tkeprod_dn__, M471Ptr<real> el, M471Ptr<real> rthraten);
__device__ M471_CALL /* Subroutine */ int scale_aware__(unsigned char *m471_work, int m471_n, M471Ptr<real> dx, M471Ptr<real> pblh, M471Ptr<real> psig_bl__, 
	M471Ptr<real> psig_shcu__);
__device__ M471_CALL real esat_blend__(unsigned char *m471_work, int m471_n, M471Ptr<real> t);
__device__ M471_CALL real qsat_blend__(unsigned char *m471_work, int m471_n, M471Ptr<real> t, M471Ptr<real> p);
__device__ M471_CALL real xl_blend__(unsigned char *m471_work, int m471_n, M471Ptr<real> t);
__device__ M471_CALL real phim_(unsigned char *m471_work, int m471_n, M471Ptr<real> zet);
__device__ M471_CALL real phih_(unsigned char *m471_work, int m471_n, M471Ptr<real> zet);
__device__ M471_CALL /* Subroutine */ int topdown_cloudrad__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dz1,
	 M471Ptr<real> zw, M471Ptr<real> fltv, M471Ptr<real> xland, M471Ptr<integer> kpbl, M471Ptr<real> pblh, M471Ptr<real> sqc, M471Ptr<real> sqi, M471Ptr<real> sqw, M471Ptr<real> thl, M471Ptr<real> th1, M471Ptr<real> ex1, M471Ptr<real> p1, 
	M471Ptr<real> rho1, M471Ptr<real> thv, M471Ptr<real> cldfra_bl1__, M471Ptr<real> rthraten, M471Ptr<real> maxkhtopdown, M471Ptr<real> khtopdown, M471Ptr<real> tkeprodtd);
__device__ M471_CALL integer ceiling_(unsigned char *m471_work, int m471_n, M471Ptr<real> x);
/* Table of constant values */

static __device__ integer c__0 = 0;

template<int M471_PHASE>
__device__ M471_CALL /* Subroutine */ int mynnedmf_(unsigned char *m471_work, int m471_n, M471Ptr<integer> i__, M471Ptr<integer> j, M471Ptr<integer> initflag, 
	M471Ptr<logical> restart, M471Ptr<logical> cycling, M471Ptr<real> delt, M471Ptr<real> dz1, M471Ptr<real> dx, 
	M471Ptr<real> u1, M471Ptr<real> v1, M471Ptr<real> w1, M471Ptr<real> th1, M471Ptr<real> sqv1, M471Ptr<real> sqc1, M471Ptr<real> sqi1, M471Ptr<real> sqs1, M471Ptr<real> qnc1, M471Ptr<real> qni1, M471Ptr<real> qnwfa1, M471Ptr<real> qnifa1,
	 M471Ptr<real> qnbca1, M471Ptr<real> ozone1, M471Ptr<real> p1, M471Ptr<real> ex1, M471Ptr<real> rho1, M471Ptr<real> tk1, M471Ptr<real> xland, M471Ptr<real> ts, M471Ptr<real> qsfc, M471Ptr<real> ps, M471Ptr<real> ust, M471Ptr<real> ch,
	 M471Ptr<real> hfx, M471Ptr<real> qfx, M471Ptr<real> znt, M471Ptr<real> wspd, M471Ptr<real> uoce, M471Ptr<real> voce, 
	M471Ptr<real> qke1, M471Ptr<real> qke_adv1__, M471Ptr<real> el1, M471Ptr<real> sh1, M471Ptr<real> sm1, M471Ptr<real> kh1, M471Ptr<real> km1, M471Ptr<integer> nchem, M471Ptr<integer> kdvel, M471Ptr<integer> ndvel, M471Ptr<real> chem, M471Ptr<real> vdep, M471Ptr<real> frp, M471Ptr<real> emis_ant_no__, M471Ptr<logical> mix_chem__,
	 M471Ptr<logical> enh_mix__, M471Ptr<logical> rrfs_sd__, M471Ptr<logical> smoke_dbg__, M471Ptr<real> tsq1, M471Ptr<real> qsq1, M471Ptr<real> cov1, M471Ptr<real> du1, M471Ptr<real> dv1, M471Ptr<real> dth1, M471Ptr<real> dqv1, M471Ptr<real> dqc1, M471Ptr<real> dqi1, M471Ptr<real> dqnc1, M471Ptr<real> dqni1, M471Ptr<real> dqs1, 
	M471Ptr<real> dqnwfa1, M471Ptr<real> dqnifa1, M471Ptr<real> dqnbca1, M471Ptr<real> dozone1, M471Ptr<real> rthraten1, M471Ptr<real> pblh, M471Ptr<integer> kpbl, M471Ptr<real> maxwidth, M471Ptr<real> maxmf, 
	M471Ptr<real> ztop_plume__, M471Ptr<real> dqke1, M471Ptr<real> qwt1, M471Ptr<real> qshear1, M471Ptr<real> qbuoy1, M471Ptr<real> qdiss1, M471Ptr<real> qc_bl1__, M471Ptr<real> qi_bl1__, M471Ptr<real> cldfra_bl1__, M471Ptr<logical> bl_mynn_tkeadvect__, M471Ptr<integer> tke_budget__, 
	M471Ptr<integer> bl_mynn_cloudpdf__, M471Ptr<integer> bl_mynn_mixlength__, M471Ptr<integer> icloud_bl__, M471Ptr<real> closure, M471Ptr<integer> bl_mynn_edmf__, M471Ptr<integer> bl_mynn_edmf_mom__, M471Ptr<integer> bl_mynn_edmf_tke__, M471Ptr<integer> bl_mynn_mixscalars__, M471Ptr<integer> bl_mynn_output__, M471Ptr<integer> bl_mynn_cloudmix__, M471Ptr<integer> bl_mynn_mixqt__, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_qt1__, M471Ptr<real> edmf_thl1__, M471Ptr<real> edmf_ent1__, 
	M471Ptr<real> edmf_qc1__, M471Ptr<real> sub_thl1__, M471Ptr<real> sub_sqv1__, M471Ptr<real> det_thl1__, M471Ptr<real> det_sqv1__, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__, M471Ptr<logical> flag_qc__, M471Ptr<logical> flag_qi__, M471Ptr<logical> flag_qnc__, M471Ptr<logical> flag_qni__, M471Ptr<logical> flag_qs__, M471Ptr<logical> flag_qnwfa__, M471Ptr<logical> flag_qnifa__, M471Ptr<logical> flag_qnbca__, M471Ptr<logical> flag_ozone__, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<integer> errflg) {
    /* Initialized data */

    real karman = .4000000059604644775390625f;
    real p608 = .6083624362945556640625f;
    real zero = 0.f;
    real half = .5f;
    real one = 1.f;
    real p1000mb = 1e5f;
    real gtr = .032700002193450927734375f;
    real xlscp = 2837.232421875f;
    real xlvcp = 2488.80029296875f;
    real cp = 1004.5f;
    real cphm_st__ = 5.f;
    real cphm_unst__ = 16.f;
    real cphh_st__ = 5.f;
    real cphh_unst__ = 16.f;
    real b1 = 24.f;
    real rcp = .2857142984867095947265625f;
    real scaleaware = 1.f;
    real rc0000 = 1200.f;
    real rc0001 = .001f;
    real rc0002 = 270.f;
    real rc0003 = 380.f;
    real rc0004 = 160.f;
    real rc0005 = 3.402823466e38f;
    real rc0006 = 2e-4f;
    real rc0007 = 5.f;
    real rc0008 = 700.f;
    real rc0009 = .01f;
    real rc0010 = 0.f;
    real rc0011 = .84f;
    real rc0012 = 1e-8f;
    real rc0013 = 1e-6f;
    real rc0014 = 20.f;
    real rc0015 = .25f;
    real rc0016 = 1.f;
    real rc0017 = 1.5f;
    real rc0018 = 0.f;
    real rc0019 = .002f;
    real rc0020 = 1e4f;
    real rc0021 = 1e-12f;
    real rc0022 = 4.f;

    /* System generated locals */
    integer dz1_offset, u1_offset, v1_offset, w1_offset, th1_offset, 
	    p1_offset, ex1_offset, rho1_offset, tk1_offset, rthraten1_offset, 
	    sqv1_offset, sqc1_offset, sqi1_offset, sqs1_offset, qni1_offset, 
	    qnc1_offset, qnwfa1_offset, qnifa1_offset, qnbca1_offset, 
	    ozone1_offset, qke1_offset, tsq1_offset, qsq1_offset, cov1_offset,
	     qke_adv1_offset, sh1_offset, sm1_offset, el1_offset, du1_offset, 
	    dv1_offset, dth1_offset, dqv1_offset, dqc1_offset, dqi1_offset, 
	    dqs1_offset, dqni1_offset, dqnc1_offset, dqnwfa1_offset, 
	    dqnifa1_offset, dqnbca1_offset, dozone1_offset, qc_bl1_offset, 
	    qi_bl1_offset, cldfra_bl1_offset, edmf_a1_offset, edmf_w1_offset, 
	    edmf_qt1_offset, edmf_thl1_offset, edmf_ent1_offset, 
	    edmf_qc1_offset, sub_thl1_offset, sub_sqv1_offset, 
	    det_thl1_offset, det_sqv1_offset, qwt1_offset, qshear1_offset, 
	    qbuoy1_offset, qdiss1_offset, dqke1_offset, kh1_offset, 
	    km1_offset, chem_dim1, chem_offset, pattern_spp_pbl1_offset, i__1,
	     i__2;
    real local_r__1; real &r__1=M471_PHASE<0 ? local_r__1 : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+0),m471_n);
    real local_r__2; real &r__2=M471_PHASE<0 ? local_r__2 : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+4),m471_n);
    real local_r__3; real &r__3=M471_PHASE<0 ? local_r__3 : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+8),m471_n);
    real local_r__4; real &r__4=M471_PHASE<0 ? local_r__4 : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+12),m471_n);
    doublereal d__1, d__2;

    /* Builtin functions */
    

    /* Local variables */
    M471Ptr<real> s_awthl1__=m471_array<real>(m471_work,0,m471_n);
M471Ptr<real> s_awqni1__=m471_array<real>(m471_work,(8*(MYNN471_NZ-49)+200),m471_n);
M471Ptr<real> sd_awqt1__=m471_array<real>(m471_work,(16*(MYNN471_NZ-49)+400),m471_n);
M471Ptr<real> sd_awqv1__=m471_array<real>(m471_work,(24*(MYNN471_NZ-49)+600),m471_n);
    
    integer kproblem;
    M471Ptr<real> s_awchem1__=m471_array<real>(m471_work,(32*(MYNN471_NZ-49)+800),m471_n);
M471Ptr<real> sd_awqke1__=m471_array<real>(m471_work,(40*(MYNN471_NZ-49)+1000),m471_n);
M471Ptr<real> sd_awqnc1__=m471_array<real>(m471_work,(48*(MYNN471_NZ-49)+1200),m471_n);
M471Ptr<real> sd_awthl1__=m471_array<real>(m471_work,(56*(MYNN471_NZ-49)+1400),m471_n);
M471Ptr<real> sd_awqni1__=m471_array<real>(m471_work,(64*(MYNN471_NZ-49)+1600),m471_n);
    integer k;
    
    real local_psig_shcu__; real &psig_shcu__=M471_PHASE<0 ? local_psig_shcu__ : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+16),m471_n);
M471Ptr<real> edmf_a_dd1__=m471_array<real>(m471_work,(72*(MYNN471_NZ-49)+1800),m471_n);
M471Ptr<real> edmf_w_dd1__=m471_array<real>(m471_work,(72*(MYNN471_NZ-49)+2000),m471_n);
M471Ptr<real> qc_bl1_old__=m471_array<real>(m471_work,(72*(MYNN471_NZ-49)+2200),m471_n);
M471Ptr<real> qi_bl1_old__=m471_array<real>(m471_work,(72*(MYNN471_NZ-49)+2400),m471_n);
M471Ptr<real> khtopdown=m471_array<real>(m471_work,(72*(MYNN471_NZ-49)+2600),m471_n);
M471Ptr<real> s_awqnbca1__=m471_array<real>(m471_work,(72*(MYNN471_NZ-49)+2800),m471_n);
M471Ptr<real> diss_heat1__=m471_array<real>(m471_work,(80*(MYNN471_NZ-49)+3000),m471_n);
M471Ptr<real> s_awqnifa1__=m471_array<real>(m471_work,(80*(MYNN471_NZ-49)+3200),m471_n);
    integer ic;
    M471Ptr<real> s_awqnwfa1__=m471_array<real>(m471_work,(88*(MYNN471_NZ-49)+3400),m471_n);
M471Ptr<real> qc1=m471_array<real>(m471_work,(96*(MYNN471_NZ-49)+3600),m471_n);
M471Ptr<real> edmf_qc_dd1__=m471_array<real>(m471_work,(96*(MYNN471_NZ-49)+3800),m471_n);
M471Ptr<real> tkeprod_dn__=m471_array<real>(m471_work,(96*(MYNN471_NZ-49)+4000),m471_n);
M471Ptr<real> qi1=m471_array<real>(m471_work,(96*(MYNN471_NZ-49)+4200),m471_n);
M471Ptr<real> qs1=m471_array<real>(m471_work,(96*(MYNN471_NZ-49)+4400),m471_n);
M471Ptr<real> edmf_qt_dd1__=m471_array<real>(m471_work,(96*(MYNN471_NZ-49)+4600),m471_n);
M471Ptr<real> qv1=m471_array<real>(m471_work,(96*(MYNN471_NZ-49)+4800),m471_n);
M471Ptr<real> vq1=m471_array<real>(m471_work,(96*(MYNN471_NZ-49)+5000),m471_n);
M471Ptr<real> vt1=m471_array<real>(m471_work,(96*(MYNN471_NZ-49)+5200),m471_n);
M471Ptr<real> tkeprod_up__=m471_array<real>(m471_work,(96*(MYNN471_NZ-49)+5400),m471_n);
    integer local_ktop_plume__; integer &ktop_plume__=M471_PHASE<0 ? local_ktop_plume__ : *m471_array<integer>(m471_work,(1184*(MYNN471_NZ-49)+85800+96),m471_n);
    M471Ptr<real> zw1=m471_array<real>(m471_work,(96*(MYNN471_NZ-49)+5600),m471_n);
M471Ptr<real> sd_awqnifa1__=m471_array<real>(m471_work,(104*(MYNN471_NZ-49)+5800),m471_n);
    integer imd, jmd;
    real local_cpm; real &cpm=M471_PHASE<0 ? local_cpm : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+20),m471_n);
    real local_flq; real &flq=M471_PHASE<0 ? local_flq : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+24),m471_n);
    real local_flt; real &flt=M471_PHASE<0 ? local_flt : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+28),m471_n);
    real local_phh; real &phh=M471_PHASE<0 ? local_phh : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+32),m471_n);
    real local_zet; real &zet=M471_PHASE<0 ? local_zet : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+36),m471_n);
M471Ptr<real> sd_awqnwfa1__=m471_array<real>(m471_work,(112*(MYNN471_NZ-49)+6000),m471_n);
    
    real local_pmz; real &pmz=M471_PHASE<0 ? local_pmz : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+40),m471_n);
    real local_wsp; real &wsp=M471_PHASE<0 ? local_wsp : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+44),m471_n);
M471Ptr<real> dfh1=m471_array<real>(m471_work,(120*(MYNN471_NZ-49)+6200),m471_n);
M471Ptr<real> dfm1=m471_array<real>(m471_work,(120*(MYNN471_NZ-49)+6400),m471_n);
M471Ptr<real> qcd1=m471_array<real>(m471_work,(120*(MYNN471_NZ-49)+6600),m471_n);
M471Ptr<real> dfq1=m471_array<real>(m471_work,(120*(MYNN471_NZ-49)+6800),m471_n);
M471Ptr<real> tcd1=m471_array<real>(m471_work,(120*(MYNN471_NZ-49)+7000),m471_n);
M471Ptr<real> pdc1=m471_array<real>(m471_work,(120*(MYNN471_NZ-49)+7200),m471_n);
M471Ptr<real> pdk1=m471_array<real>(m471_work,(120*(MYNN471_NZ-49)+7400),m471_n);
M471Ptr<real> pdq1=m471_array<real>(m471_work,(120*(MYNN471_NZ-49)+7600),m471_n);
M471Ptr<real> thl1=m471_array<real>(m471_work,(120*(MYNN471_NZ-49)+7800),m471_n);
M471Ptr<real> pdt1=m471_array<real>(m471_work,(120*(MYNN471_NZ-49)+8000),m471_n);
M471Ptr<real> sgm1=m471_array<real>(m471_work,(120*(MYNN471_NZ-49)+8200),m471_n);
M471Ptr<real> edmf_thl_dd1__=m471_array<real>(m471_work,(120*(MYNN471_NZ-49)+8400),m471_n);
M471Ptr<real> thv1=m471_array<real>(m471_work,(120*(MYNN471_NZ-49)+8600),m471_n);
M471Ptr<real> edmf_ent_dd1__=m471_array<real>(m471_work,(120*(MYNN471_NZ-49)+8800),m471_n);
    
    real local_flqc; real &flqc=M471_PHASE<0 ? local_flqc : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+48),m471_n);
M471Ptr<real> sqw1=m471_array<real>(m471_work,(120*(MYNN471_NZ-49)+9000),m471_n);
    
    real local_rmol; real &rmol=M471_PHASE<0 ? local_rmol : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+92),m471_n);
    if constexpr(M471_PHASE<=0) rmol=0.f;
    real local_sqcg; real &sqcg=M471_PHASE<0 ? local_sqcg : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+52),m471_n);
    real local_flqv; real &flqv=M471_PHASE<0 ? local_flqv : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+56),m471_n);
    real local_fltv; real &fltv=M471_PHASE<0 ? local_fltv : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+60),m471_n);
    real local_red001; real &red001=M471_PHASE<0 ? local_red001 : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+64),m471_n);
    real local_delt2; real &delt2=M471_PHASE<0 ? local_delt2 : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+68),m471_n);
    real local_maxkhtopdown; real &maxkhtopdown=M471_PHASE<0 ? local_maxkhtopdown : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+72),m471_n);
    real local_phi_m__; real &phi_m__=M471_PHASE<0 ? local_phi_m__ : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+76),m471_n);
M471Ptr<real> s_aw1__=m471_array<real>(m471_work,(120*(MYNN471_NZ-49)+9200),m471_n);
M471Ptr<real> thlv1=m471_array<real>(m471_work,(128*(MYNN471_NZ-49)+9400),m471_n);
M471Ptr<real> sd_aw1__=m471_array<real>(m471_work,(128*(MYNN471_NZ-49)+9600),m471_n);
M471Ptr<real> det_u1__=m471_array<real>(m471_work,(136*(MYNN471_NZ-49)+9800),m471_n);
M471Ptr<real> det_v1__=m471_array<real>(m471_work,(136*(MYNN471_NZ-49)+10000),m471_n);
M471Ptr<real> cldfra_bl1_old__=m471_array<real>(m471_work,(136*(MYNN471_NZ-49)+10200),m471_n);
    
    M471Ptr<real> sub_u1__=m471_array<real>(m471_work,(136*(MYNN471_NZ-49)+10400),m471_n);
M471Ptr<real> sub_v1__=m471_array<real>(m471_work,(136*(MYNN471_NZ-49)+10600),m471_n);
M471Ptr<real> s_awu1__=m471_array<real>(m471_work,(136*(MYNN471_NZ-49)+10800),m471_n);
M471Ptr<real> s_awv1__=m471_array<real>(m471_work,(144*(MYNN471_NZ-49)+11000),m471_n);
    integer ixarr1, ixarr2;
    M471Ptr<real> dummy1=m471_array<real>(m471_work,(152*(MYNN471_NZ-49)+11200),m471_n);
M471Ptr<real> dummy2=m471_array<real>(m471_work,(152*(MYNN471_NZ-49)+11400),m471_n);
M471Ptr<real> kzero1=m471_array<real>(m471_work,(152*(MYNN471_NZ-49)+11600),m471_n);
    
    real local_th_sfc__; real &th_sfc__=M471_PHASE<0 ? local_th_sfc__ : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+80),m471_n);
    real local_exnerg; real &exnerg=M471_PHASE<0 ? local_exnerg : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+84),m471_n);
M471Ptr<real> s_awqc1__=m471_array<real>(m471_work,(152*(MYNN471_NZ-49)+11800),m471_n);
    logical initialize_qke__;
    M471Ptr<real> sd_awu1__=m471_array<real>(m471_work,(160*(MYNN471_NZ-49)+12000),m471_n);
M471Ptr<real> sd_awv1__=m471_array<real>(m471_work,(168*(MYNN471_NZ-49)+12200),m471_n);
M471Ptr<real> s_awqt1__=m471_array<real>(m471_work,(176*(MYNN471_NZ-49)+12400),m471_n);
M471Ptr<real> s_awqv1__=m471_array<real>(m471_work,(184*(MYNN471_NZ-49)+12600),m471_n);
    
    real local_psig_bl__; real &psig_bl__=M471_PHASE<0 ? local_psig_bl__ : *m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+85800+88),m471_n);
    logical problem;
    M471Ptr<real> sd_awqc1__=m471_array<real>(m471_work,(192*(MYNN471_NZ-49)+12800),m471_n);
M471Ptr<real> det_sqc1__=m471_array<real>(m471_work,(200*(MYNN471_NZ-49)+13000),m471_n);
M471Ptr<real> sd_awqi1__=m471_array<real>(m471_work,(200*(MYNN471_NZ-49)+13200),m471_n);
M471Ptr<real> s_awqke1__=m471_array<real>(m471_work,(208*(MYNN471_NZ-49)+13400),m471_n);
M471Ptr<real> s_awqnc1__=m471_array<real>(m471_work,(216*(MYNN471_NZ-49)+13600),m471_n);
    

    /* Parameter adjustments */
    --vdep;
    pattern_spp_pbl1_offset = *kts;
    pattern_spp_pbl1__ -= pattern_spp_pbl1_offset;
    det_sqv1_offset = *kts;
    det_sqv1__ -= det_sqv1_offset;
    det_thl1_offset = *kts;
    det_thl1__ -= det_thl1_offset;
    sub_sqv1_offset = *kts;
    sub_sqv1__ -= sub_sqv1_offset;
    sub_thl1_offset = *kts;
    sub_thl1__ -= sub_thl1_offset;
    edmf_qc1_offset = *kts;
    edmf_qc1__ -= edmf_qc1_offset;
    edmf_ent1_offset = *kts;
    edmf_ent1__ -= edmf_ent1_offset;
    edmf_thl1_offset = *kts;
    edmf_thl1__ -= edmf_thl1_offset;
    edmf_qt1_offset = *kts;
    edmf_qt1__ -= edmf_qt1_offset;
    edmf_w1_offset = *kts;
    edmf_w1__ -= edmf_w1_offset;
    edmf_a1_offset = *kts;
    edmf_a1__ -= edmf_a1_offset;
    cldfra_bl1_offset = *kts;
    cldfra_bl1__ -= cldfra_bl1_offset;
    qi_bl1_offset = *kts;
    qi_bl1__ -= qi_bl1_offset;
    qc_bl1_offset = *kts;
    qc_bl1__ -= qc_bl1_offset;
    qdiss1_offset = *kts;
    qdiss1 -= qdiss1_offset;
    qbuoy1_offset = *kts;
    qbuoy1 -= qbuoy1_offset;
    qshear1_offset = *kts;
    qshear1 -= qshear1_offset;
    qwt1_offset = *kts;
    qwt1 -= qwt1_offset;
    dqke1_offset = *kts;
    dqke1 -= dqke1_offset;
    rthraten1_offset = *kts;
    rthraten1 -= rthraten1_offset;
    dozone1_offset = *kts;
    dozone1 -= dozone1_offset;
    dqnbca1_offset = *kts;
    dqnbca1 -= dqnbca1_offset;
    dqnifa1_offset = *kts;
    dqnifa1 -= dqnifa1_offset;
    dqnwfa1_offset = *kts;
    dqnwfa1 -= dqnwfa1_offset;
    dqs1_offset = *kts;
    dqs1 -= dqs1_offset;
    dqni1_offset = *kts;
    dqni1 -= dqni1_offset;
    dqnc1_offset = *kts;
    dqnc1 -= dqnc1_offset;
    dqi1_offset = *kts;
    dqi1 -= dqi1_offset;
    dqc1_offset = *kts;
    dqc1 -= dqc1_offset;
    dqv1_offset = *kts;
    dqv1 -= dqv1_offset;
    dth1_offset = *kts;
    dth1 -= dth1_offset;
    dv1_offset = *kts;
    dv1 -= dv1_offset;
    du1_offset = *kts;
    du1 -= du1_offset;
    cov1_offset = *kts;
    cov1 -= cov1_offset;
    qsq1_offset = *kts;
    qsq1 -= qsq1_offset;
    tsq1_offset = *kts;
    tsq1 -= tsq1_offset;
    chem_dim1 = *kte - *kts + 1;
    chem_offset = *kts + chem_dim1;
    chem -= chem_offset;
    km1_offset = *kts;
    km1 -= km1_offset;
    kh1_offset = *kts;
    kh1 -= kh1_offset;
    sm1_offset = *kts;
    sm1 -= sm1_offset;
    sh1_offset = *kts;
    sh1 -= sh1_offset;
    el1_offset = *kts;
    el1 -= el1_offset;
    qke_adv1_offset = *kts;
    qke_adv1__ -= qke_adv1_offset;
    qke1_offset = *kts;
    qke1 -= qke1_offset;
    tk1_offset = *kts;
    tk1 -= tk1_offset;
    rho1_offset = *kts;
    rho1 -= rho1_offset;
    ex1_offset = *kts;
    ex1 -= ex1_offset;
    p1_offset = *kts;
    p1 -= p1_offset;
    ozone1_offset = *kts;
    ozone1 -= ozone1_offset;
    qnbca1_offset = *kts;
    qnbca1 -= qnbca1_offset;
    qnifa1_offset = *kts;
    qnifa1 -= qnifa1_offset;
    qnwfa1_offset = *kts;
    qnwfa1 -= qnwfa1_offset;
    qni1_offset = *kts;
    qni1 -= qni1_offset;
    qnc1_offset = *kts;
    qnc1 -= qnc1_offset;
    sqs1_offset = *kts;
    sqs1 -= sqs1_offset;
    sqi1_offset = *kts;
    sqi1 -= sqi1_offset;
    sqc1_offset = *kts;
    sqc1 -= sqc1_offset;
    sqv1_offset = *kts;
    sqv1 -= sqv1_offset;
    th1_offset = *kts;
    th1 -= th1_offset;
    w1_offset = *kts;
    w1 -= w1_offset;
    v1_offset = *kts;
    v1 -= v1_offset;
    u1_offset = *kts;
    u1 -= u1_offset;
    dz1_offset = *kts;
    dz1 -= dz1_offset;

    if constexpr(M471_PHASE<0 || M471_PHASE==0) {
    /* Function Body */
    *errflg = 0;
    if (FALSE_) {
	problem = FALSE_;
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
/* Computing 2nd power */
	    r__1 = u1[k];
/* Computing 2nd power */
	    r__2 = v1[k];
	    wsp = (real)sqrt((real)(r__1 * r__1 + r__2 * r__2));
	    if (abs(*hfx) > rc0000 || abs(*qfx) > rc0001 || wsp > rc0002 || 
		    tk1[k] > rc0003 || tk1[k] < rc0004 || sqv1[k] < zero || 
		    sqc1[k] < zero || *ps - p1[*kts] < zero) {
		kproblem = k;
		problem = TRUE_;
	    }
/* L9004: */
	}
/* L9005: */
	if (problem) {
	}
    }
    imd = 10;
    jmd = 10;
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_a1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_w1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_qt1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_thl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_ent1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_qc1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_thl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_sqv1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_thl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_sqv1__[*kts + ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	edmf_a_dd1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	edmf_w_dd1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	edmf_qc_dd1__[ixarr1 - 1] = zero;
    }
    ktop_plume__ = 0;
    *ztop_plume__ = zero;
    *maxwidth = zero;
    *maxmf = zero;
    maxkhtopdown = zero;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	kzero1[ixarr1 - 1] = zero;
    }
    if (*initflag > 0 && ! (*restart)) {
	if (*restart || *cycling) {
	    red001 = -rc0005;
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
/* Computing MAX */
		r__1 = red001, r__2 = qke1[*kts + ixarr1 - 1];
		red001 = max(r__1,r__2);
	    }
	    if (red001 < rc0006) {
		initialize_qke__ = TRUE_;
	    } else {
		initialize_qke__ = FALSE_;
	    }
	} else {
	    initialize_qke__ = TRUE_;
	}
	if (! (*restart) || ! (*cycling)) {
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		sh1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		sm1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		el1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		tsq1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qsq1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		cov1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		cldfra_bl1__[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qc_bl1__[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qi_bl1__[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qke1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qke_adv1__[*kts + ixarr1 - 1] = zero;
	    }
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dqc1[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dqi1[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dqni1[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dqnc1[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dqnwfa1[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dqnifa1[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dqnbca1[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dozone1[*kts + ixarr1 - 1] = zero;
	}
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    qc_bl1_old__[ixarr1 - 1] = zero;
	}
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    cldfra_bl1_old__[ixarr1 - 1] = zero;
	}
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    sgm1[ixarr1 - 1] = zero;
	}
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    vt1[ixarr1 - 1] = zero;
	}
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    vq1[ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    km1[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    kh1[*kts + ixarr1 - 1] = zero;
	}
	if (*tke_budget__ == 1) {
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qwt1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qshear1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qbuoy1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qdiss1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		dqke1[*kts + ixarr1 - 1] = zero;
	    }
	}
	zw1[*kts - 1] = zero;
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    thv1[k - 1] = th1[k] * (one + p608 * sqv1[k]);
	    sqw1[k - 1] = sqv1[k] + sqc1[k] + sqi1[k];
	    thl1[k - 1] = th1[k] - xlvcp / ex1[k] * sqc1[k] - xlscp / ex1[k] *
		     sqi1[k];
	    thlv1[k - 1] = thl1[k - 1] * (one + p608 * sqv1[k]);
	    zw1[k] = zw1[k - 1] + dz1[k];
/* L9007: */
	}
/* L9008: */
	if (initialize_qke__) {
	    i__1 = *kte;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
		r__1 = (*ust * rc0008 - zw1[k - 1]) / (max(*ust,rc0009) * 
			rc0008);
		qke1[k] = rc0007 * *ust * max(r__1,rc0009);
/* L9010: */
	    }
/* L9011: */
	}
	get_pblh__(m471_work,m471_n, kts, kte, pblh, thv1, &qke1[qke1_offset], zw1, &dz1[
		dz1_offset], xland, kpbl);
	if (scaleaware > zero) {
	    scale_aware__(m471_work,m471_n, dx, pblh, &psig_bl__, &psig_shcu__);
	} else {
	    psig_bl__ = one;
	    psig_shcu__ = one;
	}
	mym_initialize__(m471_work,m471_n, kts, kte, xland, &dz1[dz1_offset], dx, zw1, &u1[
		u1_offset], &v1[v1_offset], thl1, &sqv1[sqv1_offset], pblh, &
		th1[th1_offset], thv1, thlv1, &sh1[sh1_offset], &sm1[
		sm1_offset], ust, &rmol, &el1[el1_offset], &qke1[qke1_offset],
		 &tsq1[tsq1_offset], &qsq1[qsq1_offset], &cov1[cov1_offset], &
		psig_bl__, &cldfra_bl1__[cldfra_bl1_offset], 
		bl_mynn_mixlength__, &edmf_w1__[edmf_w1_offset], &edmf_a1__[
		edmf_a1_offset], edmf_w_dd1__, edmf_a_dd1__, &
		initialize_qke__, spp_pbl__, &pattern_spp_pbl1__[
		pattern_spp_pbl1_offset]);
	if (! (*restart)) {
	    if (*bl_mynn_tkeadvect__) {
		i__1 = *kte;
		
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		    qke_adv1__[k] = qke1[k];
/* L9013: */
		}
/* L9014: */
	    }
	}
    }
    if (*bl_mynn_tkeadvect__) {
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    qke1[*kts + ixarr1 - 1] = qke_adv1__[*kts + ixarr1 - 1];
	}
    }
    if (*tke_budget__ == 1) {
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dqke1[*kts + ixarr1 - 1] = qke1[*kts + ixarr1 - 1];
	}
    }
    if (*icloud_bl__ > 0) {
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    cldfra_bl1_old__[*kts + ixarr1 - 2] = cldfra_bl1__[*kts + ixarr1 
		    - 1];
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    qc_bl1_old__[*kts + ixarr1 - 2] = qc_bl1__[*kts + ixarr1 - 1];
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    qi_bl1_old__[*kts + ixarr1 - 2] = qi_bl1__[*kts + ixarr1 - 1];
	}
    } else {
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    cldfra_bl1__[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    qc_bl1__[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    qi_bl1__[*kts + ixarr1 - 1] = zero;
	}
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    cldfra_bl1_old__[ixarr1 - 1] = zero;
	}
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    qc_bl1_old__[ixarr1 - 1] = zero;
	}
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    qi_bl1_old__[ixarr1 - 1] = zero;
	}
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dqc1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dqi1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dqs1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dqni1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dqnc1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dqnwfa1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dqnifa1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dqnbca1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dozone1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_a1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_w1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_qc1__[*kts + ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_aw1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awthl1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqt1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqv1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqc1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awu1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awv1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqke1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqnc1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqni1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqnwfa1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqnifa1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqnbca1__[ixarr1 - 1] = zero;
    }
    i__1 = *nchem;
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= i__1; ++ixarr2) {
	i__2 = *kte + 1 - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
	    s_awchem1__[*kts + ixarr1 - 1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	edmf_a_dd1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	edmf_w_dd1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	edmf_qc_dd1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_aw1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awthl1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awqt1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awqv1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awqc1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awqi1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awqnc1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awqni1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awqnwfa1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awqnifa1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awu1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awv1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awqke1__[ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_thl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_sqv1__[*kts + ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	sub_u1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	sub_v1__[ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_thl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_sqv1__[*kts + ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	det_sqc1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	det_u1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	det_v1__[ixarr1 - 1] = zero;
    }
    zw1[*kts - 1] = zero;
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	zw1[k] = zw1[k - 1] + dz1[k];
	qv1[k - 1] = sqv1[k] / (one - sqv1[k]);
	qc1[k - 1] = sqc1[k] / (one - sqv1[k]);
	qi1[k - 1] = sqi1[k] / (one - sqv1[k]);
	qs1[k - 1] = sqs1[k] / (one - sqv1[k]);
	sqw1[k - 1] = sqv1[k] + sqc1[k] + sqi1[k];
	thl1[k - 1] = th1[k] - xlvcp / ex1[k] * sqc1[k] - xlscp / ex1[k] * 
		sqi1[k];
	thv1[k - 1] = th1[k] * (one + p608 * sqv1[k]);
/* L9016: */
    }
/* L9017: */
    get_pblh__(m471_work,m471_n, kts, kte, pblh, thv1, &qke1[qke1_offset], zw1, &dz1[dz1_offset]
	    , xland, kpbl);
    if (scaleaware > rc0010) {
	scale_aware__(m471_work,m471_n, dx, pblh, &psig_bl__, &psig_shcu__);
    } else {
	psig_bl__ = one;
	psig_shcu__ = one;
    }
    sqcg = zero;
/* Computing MAX */
    r__1 = sqv1[*kts];
    cpm = cp * (one + rc0011 * max(r__1,rc0012));
    d__1 =  (*ps / p1000mb);
    d__2 =  rcp;
    exnerg = (real)pow_dd(&d__1, &d__2);
    flqv = *qfx / rho1[*kts];
    flqc = zero;
    th_sfc__ = *ts / ex1[*kts];
    flq = flqv + flqc;
    flt = *hfx / (rho1[*kts] * cpm) - xlvcp * flqc / ex1[*kts];
    fltv = flt + flqv * p608 * th_sfc__;
/* Computing MAX */
/* Computing 3rd power */
    r__2 = *ust;
    r__1 = r__2 * (r__2 * r__2);
    rmol = -karman * gtr * fltv / max(r__1,rc0013);
    zet = half * dz1[*kts] * rmol;
/* Computing MAX */
    r__1 = zet, r__2 = -rc0014;
    zet = max(r__1,r__2);
    zet = min(zet,rc0014);
    if (FALSE_) {
	if (zet >= zero) {
	    pmz = one + (cphm_st__ - one) * zet;
	    phh = one + cphh_st__ * zet;
	} else {
	    d__1 =  (one - cphm_unst__ * zet);
	    d__2 =  rc0015;
	    pmz = one / (real)pow_dd(&d__1, &d__2) - zet;
	    phh = one / (real)sqrt((real)(one - cphh_unst__ * zet));
	}
    } else {
	phi_m__ = (real)phim_(m471_work,m471_n, &zet);
	pmz = phi_m__ - zet;
	phh = (real)phih_(m471_work,m471_n, &zet);
    }

    }
    if constexpr(M471_PHASE<0 || M471_PHASE==1) {
    mym_condensation__(m471_work,m471_n, kts, kte, dx, &dz1[dz1_offset], zw1, xland, thl1, sqw1,
	     &sqv1[sqv1_offset], &sqc1[sqc1_offset], &sqi1[sqi1_offset], &
	    sqs1[sqs1_offset], &p1[p1_offset], &ex1[ex1_offset], &tsq1[
	    tsq1_offset], &qsq1[qsq1_offset], &cov1[cov1_offset], &sh1[
	    sh1_offset], &el1[el1_offset], bl_mynn_cloudpdf__, &qc_bl1__[
	    qc_bl1_offset], &qi_bl1__[qi_bl1_offset], &cldfra_bl1__[
	    cldfra_bl1_offset], pblh, hfx, vt1, vq1, &th1[th1_offset], sgm1, 
	    spp_pbl__, &pattern_spp_pbl1__[pattern_spp_pbl1_offset]);
    if (FALSE_) {
	topdown_cloudrad__(m471_work,m471_n, kts, kte, &dz1[dz1_offset], zw1, &fltv, xland, 
		kpbl, pblh, &sqc1[sqc1_offset], &sqi1[sqi1_offset], sqw1, 
		thl1, &th1[th1_offset], &ex1[ex1_offset], &p1[p1_offset], &
		rho1[rho1_offset], thv1, &cldfra_bl1__[cldfra_bl1_offset], &
		rthraten1[rthraten1_offset], &maxkhtopdown, khtopdown, 
		tkeprod_dn__);
    } else {
	maxkhtopdown = zero;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    khtopdown[ixarr1 - 1] = zero;
	}
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    tkeprod_dn__[ixarr1 - 1] = zero;
	}
    }

    }
    if constexpr(M471_PHASE<0 || M471_PHASE==2 || M471_PHASE==7 || M471_PHASE==8 || M471_PHASE==9) {
    if (*bl_mynn_edmf__ > 0) {
	dmp_mf__<M471_PHASE>(m471_work,m471_n, i__, j, kts, kte, delt, zw1, &dz1[dz1_offset], &p1[p1_offset]
		, &rho1[rho1_offset], bl_mynn_edmf_mom__, bl_mynn_edmf_tke__, 
		bl_mynn_mixscalars__, &u1[u1_offset], &v1[v1_offset], &w1[
		w1_offset], &th1[th1_offset], thl1, thv1, &tk1[tk1_offset], 
		sqw1, &sqv1[sqv1_offset], &sqc1[sqc1_offset], &qke1[
		qke1_offset], &qnc1[qnc1_offset], &qni1[qni1_offset], &qnwfa1[
		qnwfa1_offset], &qnifa1[qnifa1_offset], &qnbca1[qnbca1_offset]
		, &ex1[ex1_offset], vt1, vq1, sgm1, ust, &flt, &fltv, &flq, &
		flqv, pblh, kpbl, dx, xland, &th_sfc__, &edmf_a1__[
		edmf_a1_offset], &edmf_w1__[edmf_w1_offset], &edmf_qt1__[
		edmf_qt1_offset], &edmf_thl1__[edmf_thl1_offset], &
		edmf_ent1__[edmf_ent1_offset], &edmf_qc1__[edmf_qc1_offset], 
		s_aw1__, s_awthl1__, s_awqt1__, s_awqv1__, s_awqc1__, 
		s_awu1__, s_awv1__, s_awqke1__, s_awqnc1__, s_awqni1__, 
		s_awqnwfa1__, s_awqnifa1__, s_awqnbca1__, &sub_thl1__[
		sub_thl1_offset], &sub_sqv1__[sub_sqv1_offset], sub_u1__, 
		sub_v1__, &det_thl1__[det_thl1_offset], &det_sqv1__[
		det_sqv1_offset], det_sqc1__, det_u1__, det_v1__, nchem, &
		chem[chem_offset], s_awchem1__, mix_chem__, &qc_bl1__[
		qc_bl1_offset], &cldfra_bl1__[cldfra_bl1_offset], 
		qc_bl1_old__, cldfra_bl1_old__, flag_qc__, flag_qi__, 
		flag_qnc__, flag_qni__, flag_qnwfa__, flag_qnifa__, 
		flag_qnbca__, &psig_shcu__, maxwidth, &ktop_plume__, maxmf, 
		ztop_plume__, spp_pbl__, &pattern_spp_pbl1__[
		pattern_spp_pbl1_offset], tkeprod_up__, &el1[el1_offset]);
    } else {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    tkeprod_up__[ixarr1 - 1] = zero;
	}
    }
    if constexpr(M471_PHASE==7 || M471_PHASE==8)return 0;
    if (FALSE_) {
	ddmp_mf__(m471_work,m471_n, kts, kte, delt, dx, zw1, &dz1[dz1_offset], &p1[p1_offset], &
		u1[u1_offset], &v1[v1_offset], &th1[th1_offset], thl1, thv1, &
		tk1[tk1_offset], sqw1, &sqv1[sqv1_offset], &sqc1[sqc1_offset],
		 &sqi1[sqi1_offset], &qnc1[qnc1_offset], &qni1[qni1_offset], &
		qnwfa1[qnwfa1_offset], &qnifa1[qnifa1_offset], &qke1[
		qke1_offset], &rho1[rho1_offset], &ex1[ex1_offset], &qc_bl1__[
		qc_bl1_offset], &qi_bl1__[qi_bl1_offset], &cldfra_bl1__[
		cldfra_bl1_offset], ust, &flt, &flq, &fltv, pblh, kpbl, 
		edmf_a_dd1__, edmf_w_dd1__, edmf_qt_dd1__, edmf_thl_dd1__, 
		edmf_ent_dd1__, edmf_qc_dd1__, sd_aw1__, sd_awthl1__, 
		sd_awqt1__, sd_awqv1__, sd_awqc1__, sd_awqi1__, sd_awqnc1__, 
		sd_awqni1__, sd_awqnwfa1__, sd_awqnifa1__, sd_awu1__, 
		sd_awv1__, sd_awqke1__, tkeprod_dn__, &el1[el1_offset], &
		rthraten1[rthraten1_offset]);
    } else {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    tkeprod_dn__[ixarr1 - 1] = zero;
	}
    }
    delt2 = *delt;
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	r__1 = qc_bl1__[k], r__2 = sqc1[k];
/* Computing MAX */
	r__3 = qi_bl1__[k], r__4 = sqi1[k] + sqs1[k];
	thlv1[k - 1] = (th1[k] - xlvcp / ex1[k] * max(r__1,r__2) - xlscp / 
		ex1[k] * max(r__3,r__4)) * (one + p608 * sqv1[k]);
/* L9019: */
    }
/* L9020: */

    }
    if constexpr(M471_PHASE<0 || M471_PHASE==3) {
    mym_turbulence__(m471_work,m471_n, kts, kte, xland, closure, &dz1[dz1_offset], dx, zw1, &u1[
	    u1_offset], &v1[v1_offset], thl1, thv1, thlv1, &sqc1[sqc1_offset],
	     sqw1, &qke1[qke1_offset], &tsq1[tsq1_offset], &qsq1[qsq1_offset],
	     &cov1[cov1_offset], vt1, vq1, &rmol, &flt, &fltv, &flq, pblh, &
	    th1[th1_offset], &sh1[sh1_offset], &sm1[sm1_offset], &el1[
	    el1_offset], dfm1, dfh1, dfq1, tcd1, qcd1, pdk1, pdt1, pdq1, pdc1,
	     &qwt1[qwt1_offset], &qshear1[qshear1_offset], &qbuoy1[
	    qbuoy1_offset], &qdiss1[qdiss1_offset], tke_budget__, &psig_bl__, 
	    &psig_shcu__, &cldfra_bl1__[cldfra_bl1_offset], 
	    bl_mynn_mixlength__, &edmf_w1__[edmf_w1_offset], &edmf_a1__[
	    edmf_a1_offset], edmf_w_dd1__, edmf_a_dd1__, tkeprod_dn__, 
	    tkeprod_up__, spp_pbl__, &pattern_spp_pbl1__[
	    pattern_spp_pbl1_offset]);

    }
    if constexpr(M471_PHASE<0 || M471_PHASE==4) {
    mym_predict__(m471_work,m471_n, kts, kte, closure, &delt2, &dz1[dz1_offset], ust, &flt, &
	    flq, &pmz, &phh, &el1[el1_offset], dfq1, &rho1[rho1_offset], pdk1,
	     pdt1, pdq1, pdc1, &qke1[qke1_offset], &tsq1[tsq1_offset], &qsq1[
	    qsq1_offset], &cov1[cov1_offset], s_aw1__, s_awqke1__, 
	    bl_mynn_edmf_tke__, &qwt1[qwt1_offset], &qdiss1[qdiss1_offset], 
	    tke_budget__);
    if (TRUE_) {
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
/* Computing MIN */
/* Computing MAX */
	    d__1 =  qke1[k];
	    d__2 =  rc0017;
/* Computing MAX */
	    r__3 = half * (el1[k] + el1[k + 1]);
	    r__2 = rc0016 * (real)pow_dd(&d__1, &d__2) / (b1 * max(r__3,one)) 
		    / cp;
	    r__1 = max(r__2,rc0018);
	    diss_heat1__[k - 1] = min(r__1,rc0019);
/* Computing MAX */
	    r__1 = p1[k];
	    diss_heat1__[k - 1] *= (real)exp((real)(-rc0020 / max(r__1,one)));
/* L9022: */
	}
/* L9023: */
	diss_heat1__[*kte - 1] = rc0010;
    } else {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    diss_heat1__[ixarr1 - 1] = rc0010;
	}
    }

    }
    if constexpr(M471_PHASE<0 || M471_PHASE==5) {
    mynn_tendencies__(m471_work,m471_n, kts, kte, i__, delt, &dz1[dz1_offset], zw1, xland, &
	    rho1[rho1_offset], &u1[u1_offset], &v1[v1_offset], &th1[
	    th1_offset], &tk1[tk1_offset], qv1, qc1, qi1, kzero1, &qnc1[
	    qnc1_offset], &qni1[qni1_offset], ps, &p1[p1_offset], &ex1[
	    ex1_offset], thl1, &sqv1[sqv1_offset], &sqc1[sqc1_offset], &sqi1[
	    sqi1_offset], kzero1, sqw1, &qnwfa1[qnwfa1_offset], &qnifa1[
	    qnifa1_offset], &qnbca1[qnbca1_offset], &ozone1[ozone1_offset], 
	    ust, &flt, &flq, &flqv, &flqc, wspd, uoce, voce, &tsq1[
	    tsq1_offset], &qsq1[qsq1_offset], &cov1[cov1_offset], tcd1, qcd1, 
	    dfm1, dfh1, dfq1, &du1[du1_offset], &dv1[dv1_offset], &dth1[
	    dth1_offset], &dqv1[dqv1_offset], &dqc1[dqc1_offset], &dqi1[
	    dqi1_offset], &dqs1[dqs1_offset], &dqnc1[dqnc1_offset], &dqni1[
	    dqni1_offset], &dqnwfa1[dqnwfa1_offset], &dqnifa1[dqnifa1_offset],
	     &dqnbca1[dqnbca1_offset], &dozone1[dozone1_offset], diss_heat1__,
	     s_aw1__, s_awthl1__, s_awqt1__, s_awqv1__, s_awqc1__, s_awu1__, 
	    s_awv1__, s_awqnc1__, s_awqni1__, s_awqnwfa1__, s_awqnifa1__, 
	    s_awqnbca1__, sd_aw1__, sd_awthl1__, sd_awqt1__, sd_awqv1__, 
	    sd_awqc1__, sd_awqi1__, sd_awqnc1__, sd_awqni1__, sd_awqnwfa1__, 
	    sd_awqnifa1__, sd_awu1__, sd_awv1__, &sub_thl1__[sub_thl1_offset],
	     &sub_sqv1__[sub_sqv1_offset], sub_u1__, sub_v1__, &det_thl1__[
	    det_thl1_offset], &det_sqv1__[det_sqv1_offset], det_sqc1__, 
	    det_u1__, det_v1__, flag_qc__, flag_qi__, flag_qnc__, flag_qni__, 
	    flag_qs__, flag_qnwfa__, flag_qnifa__, flag_qnbca__, flag_ozone__,
	     &cldfra_bl1__[cldfra_bl1_offset], bl_mynn_cloudmix__, 
	    bl_mynn_mixqt__, bl_mynn_edmf__, bl_mynn_edmf_mom__, 
	    bl_mynn_mixscalars__);

    }
    if constexpr(M471_PHASE<0 || M471_PHASE==6) {
    if (*mix_chem__) {
	if (*rrfs_sd__) {
	    mynn_mix_chem__(m471_work,m471_n, kts, kte, i__, delt, &dz1[dz1_offset], pblh, 
		    nchem, kdvel, ndvel, &chem[chem_offset], &vdep[1], &rho1[
		    rho1_offset], &flt, tcd1, qcd1, dfh1, s_aw1__, 
		    s_awchem1__, emis_ant_no__, frp, rrfs_sd__, enh_mix__, 
		    smoke_dbg__);
	} else {
	    mynn_mix_chem__(m471_work,m471_n, kts, kte, i__, delt, &dz1[dz1_offset], pblh, 
		    nchem, kdvel, ndvel, &chem[chem_offset], &vdep[1], &rho1[
		    rho1_offset], &flt, tcd1, qcd1, dfh1, s_aw1__, 
		    s_awchem1__, &zero, &zero, rrfs_sd__, enh_mix__, 
		    smoke_dbg__);
	}
	i__1 = *nchem;
	
#pragma unroll 1
for (ic = 1; ic <= i__1; ++ic) {
	    i__2 = *kte;
	    
#pragma unroll 1
for (k = *kts; k <= i__2; ++k) {
/* Computing MAX */
		r__1 = rc0021, r__2 = chem[k + ic * chem_dim1];
		chem[k + ic * chem_dim1] = max(r__1,r__2);
/* L9028: */
	    }
/* L9029: */
/* L9025: */
	}
/* L9026: */
    }
    retrieve_exchange_coeffs__(m471_work,m471_n, kts, kte, dfm1, dfh1, &dz1[dz1_offset], &km1[
	    km1_offset], &kh1[kh1_offset]);
    if (*tke_budget__ == 1) {
	k = *kts;
/* Computing 3rd power */
	r__1 = *ust;
	qshear1[k] = rc0022 * (r__1 * (r__1 * r__1) * phi_m__ / (karman * dz1[
		k])) - qshear1[k + 1];
/* Computing 3rd power */
	r__1 = *ust;
	qbuoy1[k] = rc0022 * (-(r__1 * (r__1 * r__1)) * zet / (karman * dz1[k]
		)) - qbuoy1[k + 1];
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    dummy1[k - 1] = half * (qshear1[k] + qshear1[k + 1]);
	    dummy2[k - 1] = half * (qbuoy1[k] + qbuoy1[k + 1]);
	    dqke1[k] = half * (qke1[k] - dqke1[k]) / *delt;
/* L9031: */
	}
/* L9032: */
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    qshear1[*kts + ixarr1 - 1] = dummy1[ixarr1 - 1];
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    qbuoy1[*kts + ixarr1 - 1] = dummy2[ixarr1 - 1];
	}
	k = *kte;
	qshear1[k] = zero;
	qbuoy1[k] = zero;
	qwt1[k] = zero;
	qdiss1[k] = zero;
	dqke1[k] = zero;
    }
    if (*bl_mynn_output__ > 0) {
	if (FALSE_) {
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		edmf_a1__[*kts + ixarr1 - 1] = edmf_a_dd1__[*kts + ixarr1 - 2]
			;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		edmf_w1__[*kts + ixarr1 - 1] = edmf_w_dd1__[*kts + ixarr1 - 2]
			;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		edmf_qt1__[*kts + ixarr1 - 1] = edmf_qt_dd1__[*kts + ixarr1 - 
			2];
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		edmf_thl1__[*kts + ixarr1 - 1] = edmf_thl_dd1__[*kts + ixarr1 
			- 2];
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		edmf_ent1__[*kts + ixarr1 - 1] = edmf_ent_dd1__[*kts + ixarr1 
			- 2];
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		edmf_qc1__[*kts + ixarr1 - 1] = edmf_qc_dd1__[*kts + ixarr1 - 
			2];
	    }
	}
    }
    if (FALSE_) {
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    if (*icloud_bl__ > 0) {
		if (cldfra_bl1__[k] < zero || cldfra_bl1__[k] > one) {
		}
	    }
/* L9034: */
	}
/* L9035: */
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	qke_adv1__[*kts + ixarr1 - 1] = qke1[*kts + ixarr1 - 1];
    }

    }
    return 0;
} /* mynnedmf_ */

__device__ M471_CALL /* Subroutine */ int mym_initialize__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> xland,
	 M471Ptr<real> dz, M471Ptr<real> dx, M471Ptr<real> zw, M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> thl, M471Ptr<real> qw, 
	M471Ptr<real> zi, M471Ptr<real> theta, M471Ptr<real> thv, M471Ptr<real> thlv, M471Ptr<real> sh, M471Ptr<real> sm, 
	M471Ptr<real> ust, M471Ptr<real> rmol, M471Ptr<real> el, M471Ptr<real> qke, M471Ptr<real> tsq, M471Ptr<real> qsq, 
	M471Ptr<real> cov, M471Ptr<real> psig_bl__, M471Ptr<real> cldfra_bl1__, M471Ptr<integer> bl_mynn_mixlength__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w_dd1__, M471Ptr<real> edmf_a_dd1__, M471Ptr<logical> initialize_qke__, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__) {
    /* Initialized data */

    real karman = .4000000059604644775390625f;
    real zero = 0.f;
    real half = .5f;
    real one = 1.f;
    real twothirds = .666666686534881591796875f;
    real b1 = 24.f;
    real b2 = 15.f;
    real qkemin = 9.999999747378751635551452636719e-6f;
    real rc0000 = 1.f;
    real rc0001 = 0.f;
    real rc0002 = 1.5f;
    real rc0003 = 2.f;
    real rc0004 = 3.f;
    real rc0005 = 700.f;
    real rc0006 = .01f;
    real rc0007 = 1.f;
    real rc0008 = 100.f;
    real rc0009 = 0.f;
    real rc0010 = .02f;
    real rc0011 = .25f;
    real rc0012 = 125.f;
    real rc0013 = .5f;

    /* System generated locals */
    integer dz_offset, zw_offset, u_offset, v_offset, thl_offset, thlv_offset,
	     qw_offset, cldfra_bl1_offset, edmf_w1_offset, edmf_a1_offset, 
	    edmf_w_dd1_offset, edmf_a_dd1_offset, tsq_offset, qsq_offset, 
	    cov_offset, el_offset, qke_offset, sm_offset, sh_offset, 
	    theta_offset, thv_offset, pattern_spp_pbl1_offset, i__1, i__2;
    real r__1, r__2;
    doublereal d__1, d__2;

    /* Builtin functions */
    

    /* Local variables */
    integer k, l;
    M471Ptr<real> gh=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+13800),m471_n);
M471Ptr<real> gm=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+14000),m471_n);
M471Ptr<real> ql=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+14200),m471_n);
    
    real  b1l, b2l;
M471Ptr<real> vq=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+14400),m471_n);
M471Ptr<real> vt=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+14600),m471_n);
    
    real  phh, elq, phm, elv, vkz, pmz, flt, flq;
M471Ptr<real> pdc=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+14800),m471_n);
M471Ptr<real> pdk=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+15000),m471_n);
M471Ptr<real> pdq=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+15200),m471_n);
M471Ptr<real> pdt=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+15400),m471_n);
M471Ptr<real> dtl=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+15600),m471_n);
M471Ptr<real> dqw=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+15800),m471_n);
M471Ptr<real> dtv=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+16000),m471_n);
M471Ptr<real> qkw=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+16200),m471_n);
    integer lmax;
    real fltv, tmpq;

    /* Parameter adjustments */
    pattern_spp_pbl1_offset = *kts;
    pattern_spp_pbl1__ -= pattern_spp_pbl1_offset;
    edmf_a_dd1_offset = *kts;
    edmf_a_dd1__ -= edmf_a_dd1_offset;
    edmf_w_dd1_offset = *kts;
    edmf_w_dd1__ -= edmf_w_dd1_offset;
    edmf_a1_offset = *kts;
    edmf_a1__ -= edmf_a1_offset;
    edmf_w1_offset = *kts;
    edmf_w1__ -= edmf_w1_offset;
    cldfra_bl1_offset = *kts;
    cldfra_bl1__ -= cldfra_bl1_offset;
    cov_offset = *kts;
    cov -= cov_offset;
    qsq_offset = *kts;
    qsq -= qsq_offset;
    tsq_offset = *kts;
    tsq -= tsq_offset;
    qke_offset = *kts;
    qke -= qke_offset;
    el_offset = *kts;
    el -= el_offset;
    sm_offset = *kts;
    sm -= sm_offset;
    sh_offset = *kts;
    sh -= sh_offset;
    thlv_offset = *kts;
    thlv -= thlv_offset;
    thv_offset = *kts;
    thv -= thv_offset;
    theta_offset = *kts;
    theta -= theta_offset;
    qw_offset = *kts;
    qw -= qw_offset;
    thl_offset = *kts;
    thl -= thl_offset;
    v_offset = *kts;
    v -= v_offset;
    u_offset = *kts;
    u -= u_offset;
    zw_offset = *kts;
    zw -= zw_offset;
    dz_offset = *kts;
    dz -= dz_offset;

    /* Function Body */
    pmz = rc0000;
    phh = rc0000;
    flt = rc0001;
    fltv = rc0001;
    flq = rc0001;
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	ql[k - 1] = zero;
	vt[k - 1] = zero;
	vq[k - 1] = zero;
/* L9004: */
    }
/* L9005: */
    mym_level2__(m471_work,m471_n, kts, kte, &dz[dz_offset], &u[u_offset], &v[v_offset], &thl[
	    thl_offset], &thv[thv_offset], &thlv[thlv_offset], &qw[qw_offset],
	     ql, vt, vq, dtl, dqw, dtv, gm, gh, &sm[sm_offset], &sh[sh_offset]
	    );
    el[*kts] = zero;
    if (*initialize_qke__) {
/* Computing 2nd power */
	r__1 = *ust;
	d__1 =  (b1 * pmz);
	d__2 =  (rc0003 / rc0004);
	qke[*kts] = rc0002 * (r__1 * r__1) * (real)pow_dd(&d__1, &d__2);
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = (*ust * rc0005 - zw[k]) / (max(*ust,rc0006) * rc0005);
	    qke[k] = qke[*kts] * max(r__1,rc0006);
/* L9007: */
	}
/* L9008: */
    }
    d__1 =  (b1 * pmz);
    d__2 =  (rc0007 / rc0004);
    phm = phh * b2 / (real)pow_dd(&d__1, &d__2);
/* Computing 2nd power */
    r__1 = flt / *ust;
    tsq[*kts] = phm * (r__1 * r__1);
/* Computing 2nd power */
    r__1 = flq / *ust;
    qsq[*kts] = phm * (r__1 * r__1);
    cov[*kts] = phm * (flt / *ust) * (flq / *ust);
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	vkz = karman * zw[k];
	el[k] = vkz / (one + vkz / rc0008);
	tsq[k] = rc0009;
	qsq[k] = rc0009;
	cov[k] = rc0009;
/* L9010: */
    }
/* L9011: */
    lmax = 5;
    i__1 = lmax;
    
#pragma unroll 1
for (l = 1; l <= i__1; ++l) {
	mym_length__(m471_work,m471_n, kts, kte, xland, &dz[dz_offset], dx, &zw[zw_offset], 
		rmol, &flt, &fltv, &flq, vt, vq, &u[u_offset], &v[v_offset], &
		qke[qke_offset], dtv, &el[el_offset], zi, &theta[theta_offset]
		, qkw, psig_bl__, &cldfra_bl1__[cldfra_bl1_offset], 
		bl_mynn_mixlength__, &edmf_w1__[edmf_w1_offset], &edmf_a1__[
		edmf_a1_offset], &edmf_w_dd1__[edmf_w_dd1_offset], &
		edmf_a_dd1__[edmf_a_dd1_offset]);
	i__2 = *kte;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__2; ++k) {
	    elq = el[k] * qkw[k - 1];
	    pdk[k - 1] = elq * (sm[k] * gm[k - 1] + sh[k] * gh[k - 1]);
/* Computing 2nd power */
	    r__1 = dtl[k - 1];
	    pdt[k - 1] = elq * sh[k] * (r__1 * r__1);
/* Computing 2nd power */
	    r__1 = dqw[k - 1];
	    pdq[k - 1] = elq * sh[k] * (r__1 * r__1);
	    pdc[k - 1] = elq * sh[k] * dtl[k - 1] * dqw[k - 1];
/* L9016: */
	}
/* L9017: */
	vkz = karman * half * dz[*kts];
	elv = half * (el[*kts + 1] + el[*kts]) / vkz;
	if (*initialize_qke__) {
/* Computing 2nd power */
	    r__1 = max(*ust,rc0010);
	    d__1 =  (b1 * pmz * elv);
	    d__2 =  (rc0003 / rc0004);
	    qke[*kts] = rc0007 * (r__1 * r__1) * (real)pow_dd(&d__1, &d__2);
	}
/* Computing 2nd power */
	r__1 = elv;
	d__1 =  (b1 * pmz / (r__1 * r__1));
	d__2 =  (rc0007 / rc0004);
	phm = phh * b2 / (real)pow_dd(&d__1, &d__2);
/* Computing 2nd power */
	r__1 = flt / *ust;
	tsq[*kts] = phm * (r__1 * r__1);
/* Computing 2nd power */
	r__1 = flq / *ust;
	qsq[*kts] = phm * (r__1 * r__1);
	cov[*kts] = phm * (flt / *ust) * (flq / *ust);
	i__2 = *kte - 1;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__2; ++k) {
	    b1l = b1 * rc0011 * (el[k + 1] + el[k]);
/* Computing MIN */
/* Computing MAX */
	    r__2 = b1l * (pdk[k] + pdk[k - 1]);
	    r__1 = max(r__2,qkemin);
	    tmpq = min(r__1,rc0012);
	    if (*initialize_qke__) {
		d__1 =  tmpq;
		d__2 =  twothirds;
		qke[k] = (real)pow_dd(&d__1, &d__2);
	    }
	    if (qke[k] <= zero) {
		b2l = rc0009;
	    } else {
		b2l = b2 * (b1l / b1) / (real)sqrt(qke[k]);
	    }
	    tsq[k] = b2l * (pdt[k] + pdt[k - 1]);
	    qsq[k] = b2l * (pdq[k] + pdq[k - 1]);
	    cov[k] = b2l * (pdc[k] + pdc[k - 1]);
/* L9019: */
	}
/* L9020: */
/* L9013: */
    }
/* L9014: */
    if (*initialize_qke__) {
	qke[*kts] = rc0013 * (qke[*kts] + qke[*kts + 1]);
	qke[*kte] = qke[*kte - 1];
    }
    tsq[*kte] = tsq[*kte - 1];
    qsq[*kte] = qsq[*kte - 1];
    cov[*kte] = cov[*kte - 1];
    return 0;
} /* mym_initialize__ */

__device__ M471_CALL /* Subroutine */ int mym_level2__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dz, M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> thl, M471Ptr<real> thv, M471Ptr<real> thlv, M471Ptr<real> qw, M471Ptr<real> ql, 
	M471Ptr<real> vt, M471Ptr<real> vq, M471Ptr<real> dtl, M471Ptr<real> dqw, M471Ptr<real> dtv, M471Ptr<real> gm, M471Ptr<real> gh, M471Ptr<real> sm, M471Ptr<real> sh) {
    /* Initialized data */

    real one = 1.f;
    real gtr = .032700002193450927734375f;
    real tv0 = 182.50872802734375f;
    real g1 = .23499999940395355224609375f;
    real b1 = 24.f;
    real c2 = .72899997234344482421875f;
    real c5 = .20000000298023223876953125f;
    real a1 = 1.1800000667572021484375f;
    real c1 = .137067615985870361328125f;
    real a2 = .664521038532257080078125f;
    real g2 = .56412994861602783203125f;
    real ckmod = 1.f;
    real rc0000 = 0.f;
    real rc0001 = .5f;
    real rc0002 = 1e-10f;
    real rc0003 = 1.f;
    real rc0004 = 3.f;
    real rc0005 = 2.f;
    real rc0006 = 4.f;

    /* System generated locals */
    integer dz_offset, u_offset, v_offset, thl_offset, qw_offset, ql_offset, 
	    vt_offset, vq_offset, thv_offset, thlv_offset, dtl_offset, 
	    dqw_offset, dtv_offset, gm_offset, gh_offset, sm_offset, 
	    sh_offset, i__1;
    real r__1, r__2;

    /* Builtin functions */
    

    /* Local variables */
    integer k;
    real f1, f2, rf, ri, rf1, rf2, ri1, ri2, ri3, ri4, abk, afk, rfc, shc, 
	    smc, dtq, dqz, dzk, duz, dtz, vqq, vtt, a2fac;

    /* Parameter adjustments */
    sh_offset = *kts;
    sh -= sh_offset;
    sm_offset = *kts;
    sm -= sm_offset;
    gh_offset = *kts;
    gh -= gh_offset;
    gm_offset = *kts;
    gm -= gm_offset;
    dtv_offset = *kts;
    dtv -= dtv_offset;
    dqw_offset = *kts;
    dqw -= dqw_offset;
    dtl_offset = *kts;
    dtl -= dtl_offset;
    vq_offset = *kts;
    vq -= vq_offset;
    vt_offset = *kts;
    vt -= vt_offset;
    ql_offset = *kts;
    ql -= ql_offset;
    qw_offset = *kts;
    qw -= qw_offset;
    thlv_offset = *kts;
    thlv -= thlv_offset;
    thv_offset = *kts;
    thv -= thv_offset;
    thl_offset = *kts;
    thl -= thl_offset;
    v_offset = *kts;
    v -= v_offset;
    u_offset = *kts;
    u -= u_offset;
    dz_offset = *kts;
    dz -= dz_offset;

    /* Function Body */
    dtl[(0 + (0 + (*kts << 2))) / 4] = rc0000;
    dqw[*kts] = rc0000;
    dtv[*kts] = rc0000;
    gm[*kts] = rc0000;
    gh[*kts] = rc0000;
    sm[*kts] = rc0000;
    sh[*kts] = rc0000;
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	dzk = rc0001 * (dz[k] + dz[k - 1]);
	afk = dz[k] / (dz[k] + dz[k - 1]);
	abk = one - afk;
/* Computing 2nd power */
	r__1 = u[k] - u[k - 1];
/* Computing 2nd power */
	r__2 = v[k] - v[k - 1];
	duz = r__1 * r__1 + r__2 * r__2;
/* Computing 2nd power */
	r__1 = dzk;
	duz /= r__1 * r__1;
	dtz = (thl[k] - thl[k - 1]) / dzk;
	dqz = (qw[k] - qw[k - 1]) / dzk;
	vtt = one + vt[k] * abk + vt[k - 1] * afk;
	vqq = tv0 + vq[k] * abk + vq[k - 1] * afk;
	if (FALSE_) {
	    dtq = vtt * dtz + vqq * dqz;
	} else {
	    dtq = (thlv[k] - thlv[k - 1]) / dzk;
	}
	dtl[k] = dtz;
	dqw[k] = dqz;
	dtv[k] = dtq;
	gm[k] = duz;
	gh[k] = -dtq * gtr;
	ri = -gh[k] / max(duz,rc0002);
	if (ckmod == 1.f) {
	    a2fac = rc0003 / (rc0003 + max(ri,rc0000));
	} else {
	    a2fac = rc0003;
	}
	rfc = g1 / (g1 + g2);
	f1 = b1 * (g1 - c1) + rc0004 * a2 * a2fac * (one - c2) * (one - c5) + 
		rc0005 * a1 * (rc0004 - rc0005 * c2);
	f2 = b1 * (g1 + g2) - rc0004 * a1 * (one - c2);
	rf1 = b1 * (g1 - c1) / f1;
	rf2 = b1 * g1 / f2;
	smc = a1 / (a2 * a2fac) * f1 / f2;
	shc = rc0004 * (a2 * a2fac) * (g1 + g2);
	ri1 = rc0001 / smc;
	ri2 = rf1 * smc;
	ri3 = rc0006 * rf2 * smc - rc0005 * ri2;
/* Computing 2nd power */
	r__1 = ri2;
	ri4 = r__1 * r__1;
/* Computing MIN */
/* Computing 2nd power */
	r__2 = ri;
	r__1 = ri1 * (ri + ri2 - (real)sqrt((real)(r__2 * r__2 - ri3 * ri + 
		ri4)));
	rf = min(r__1,rfc);
	sh[k] = shc * (rfc - rf) / (one - rf);
	sm[k] = smc * (rf1 - rf) / (rf2 - rf) * sh[k];
/* L9004: */
    }
/* L9005: */
    return 0;
} /* mym_level2__ */

__device__ M471_CALL /* Subroutine */ int mym_length__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> xland, 
	M471Ptr<real> dz, M471Ptr<real> dx, M471Ptr<real> zw, M471Ptr<real> rmol, M471Ptr<real> flt, M471Ptr<real> fltv, M471Ptr<real> flq, M471Ptr<real> vt, M471Ptr<real> vq, M471Ptr<real> u1, M471Ptr<real> v1, M471Ptr<real> qke, M471Ptr<real> dtv, 
	M471Ptr<real> el, M471Ptr<real> pblh, M471Ptr<real> theta, M471Ptr<real> qkw, M471Ptr<real> psig_bl__, M471Ptr<real> cldfra_bl1__, M471Ptr<integer> bl_mynn_mixlength__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w_dd1__, M471Ptr<real> edmf_a_dd1__) {
    /* Initialized data */

    real karman = .4000000059604644775390625f;
    real zero = 0.f;
    real half = .5f;
    real one = 1.f;
    real onethird = .3333333432674407958984375f;
    real grav = 9.81000041961669921875f;
    real gtr = .032700002193450927734375f;
    real tv0 = 182.50872802734375f;
    real qmin = 0.f;
    real zmax = 1.f;
    real qkemin = 9.999999747378751635551452636719e-6f;
    real rc0000 = 300.f;
    real rc0001 = 750.f;
    real rc0002 = 100.f;
    real rc0003 = 2.f;
    real rc0004 = .18f;
    real rc0005 = 1e3f;
    real rc0006 = 2.7f;
    real rc0007 = .23f;
    real rc0008 = 5.f;
    real rc0009 = .3f;
    real rc0010 = 1e4f;
    real rc0011 = 2.f;
    real rc0012 = 1e-5f;
    real rc0013 = .03f;
    real rc0014 = 1e10f;
    real rc0015 = 0.f;
    real rc0016 = .2f;
    real rc0017 = 20.f;
    real rc0018 = 1.f;
    real rc0019 = 50.f;
    real rc0020 = .4f;
    real rc0021 = 3.5f;
    real rc0022 = 2.5f;
    real rc0023 = 50.f;
    real rc0024 = 600.f;
    real rc0025 = .005f;
    real rc0026 = .01f;
    real rc0027 = 30.f;
    real rc0028 = 1.5f;
    real rc0029 = 350.f;
    real rc0030 = 25.f;
    real rc0031 = 400.f;
    real rc0032 = 8.f;
    real rc0033 = .001f;
    real rc0034 = 1.f;
    real rc0035 = .1f;
    real rc0036 = .22f;
    real rc0037 = .3f;
    real rc0038 = 10.f;
    real rc0039 = .5f;
    real rc0040 = 1.25f;
    real rc0041 = 1e-4f;
    real rc0042 = 30.f;
    real rc0043 = 150.f;
    real rc0044 = 40.f;
    real rc0045 = 200.f;
    real rc0046 = .25f;
    real rc0047 = 800.f;

    /* System generated locals */
    integer dz_offset, zw_offset, u1_offset, v1_offset, qke_offset, vt_offset,
	     vq_offset, cldfra_bl1_offset, edmf_w1_offset, edmf_a1_offset, 
	    edmf_w_dd1_offset, edmf_a_dd1_offset, qkw_offset, el_offset, 
	    dtv_offset, theta_offset, i__1;
    real r__1, r__2, r__3, r__4, r__5;
    doublereal d__1, d__2;

    /* Builtin functions */
    

    /* Local variables */
    integer k;
    real h1, h2, tau_cloud__, bv, wt, abk, afk, elb, elf, elt, cns, vsc, csl, 
	    dzk, qdz, els, zwk, qkw_elb_min__, alp1, alp2, alp3, alp4, alp5, 
	    alp6, zwk1, ctau, zslh, vflx, pblh2, wt_u1__, wt_u2__, 
	    ugrid, mindz, maxdz, wstar;
M471Ptr<real> qtke=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+16400),m471_n);
    
    real pblh_plus_ent__, elb_mf__, cldavg, el_les__, qkw_mf__, 
	    uonset, minpblh, elt_max__;
M471Ptr<real> thetaw=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+16600),m471_n);
M471Ptr<real> elblavg=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+16800),m471_n);
M471Ptr<real> elblmin=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+17000),m471_n);

    /* Parameter adjustments */
    edmf_a_dd1_offset = *kts;
    edmf_a_dd1__ -= edmf_a_dd1_offset;
    edmf_w_dd1_offset = *kts;
    edmf_w_dd1__ -= edmf_w_dd1_offset;
    edmf_a1_offset = *kts;
    edmf_a1__ -= edmf_a1_offset;
    edmf_w1_offset = *kts;
    edmf_w1__ -= edmf_w1_offset;
    cldfra_bl1_offset = *kts;
    cldfra_bl1__ -= cldfra_bl1_offset;
    qkw_offset = *kts;
    qkw -= qkw_offset;
    theta_offset = *kts;
    theta -= theta_offset;
    el_offset = *kts;
    el -= el_offset;
    dtv_offset = *kts;
    dtv -= dtv_offset;
    qke_offset = *kts;
    qke -= qke_offset;
    v1_offset = *kts;
    v1 -= v1_offset;
    u1_offset = *kts;
    u1 -= u1_offset;
    vq_offset = *kts;
    vq -= vq_offset;
    vt_offset = *kts;
    vt -= vt_offset;
    zw_offset = *kts;
    zw -= zw_offset;
    dz_offset = *kts;
    dz -= dz_offset;

    /* Function Body */
    minpblh = rc0000;
    maxdz = rc0001;
    mindz = rc0000;
    zslh = rc0002;
    csl = rc0003;
    qkw_elb_min__ = rc0004;
    ctau = rc0005;
    if (*bl_mynn_mixlength__ == 0) {
	cns = rc0006;
	alp1 = rc0007;
	alp2 = one;
	alp3 = rc0008;
	alp4 = rc0002;
	alp5 = rc0009;
/* Computing MIN */
	r__1 = rc0010, r__2 = zw[*kte - 2];
	pblh2 = min(r__1,r__2);
/* Computing MAX */
	r__1 = rc0009 * pblh2;
	h1 = max(r__1,mindz);
	h1 = min(h1,maxdz);
	h2 = h1 / rc0011;
/* Computing MAX */
	r__1 = qke[*kts];
	qkw[*kts] = (real)sqrt((real)(max(r__1,qkemin)));
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    afk = dz[k] / (dz[k] + dz[k - 1]);
	    abk = one - afk;
/* Computing MAX */
	    r__1 = qke[k] * abk + qke[k - 1] * afk;
	    qkw[k] = (real)sqrt((real)(max(r__1,qkemin)));
/* L9004: */
	}
/* L9005: */
	elt = rc0012;
	vsc = rc0012;
	k = *kts + 1;
	zwk = zw[k];
L9006:
	if (! (zwk <= pblh2 + h1)) {
	    goto L9007;
	}
	dzk = half * (dz[k] + dz[k - 1]);
/* Computing MAX */
	r__1 = qkw[k] - qmin;
	qdz = max(r__1,rc0013) * dzk;
	elt += qdz * zwk;
	vsc += qdz;
	++k;
	zwk = zw[k];
	goto L9006;
L9007:
	elt = alp1 * elt / vsc;
	vflx = (vt[*kts] + one) * *flt + (vq[*kts] + tv0) * *flq;
	d__1 =  (gtr * elt * max(vflx,zero));
	d__2 =  onethird;
	vsc = (real)pow_dd(&d__1, &d__2);
	el[*kts] = zero;
	zwk1 = zw[*kts + 1];
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    zwk = zw[k];
	    if (dtv[k] > zero) {
		bv = (real)sqrt((real)(gtr * dtv[k]));
		elb = alp2 * qkw[k] / bv * (one + alp3 / alp2 * (real)sqrt((
			real)(vsc / (bv * elt))));
		elf = alp2 * qkw[k] / bv;
	    } else {
		elb = rc0014;
		elf = elb;
	    }
	    if (*rmol > rc0015) {
/* Computing MIN */
		r__1 = zwk * *rmol;
		els = karman * zwk / (one + cns * min(r__1,zmax));
	    } else {
		d__1 =  (one - alp4 * zwk * *rmol);
		d__2 =  rc0016;
		els = karman * zwk * (real)pow_dd(&d__1, &d__2);
	    }
	    wt = half * (real)tanh((real)((zwk - (pblh2 + h1)) / h2)) + half;
/* Computing MIN */
	    r__1 = elb / (elb / elt + elb / els + one);
	    el[k] = min(r__1,elf);
/* L9010: */
	}
/* L9011: */
    } else if (*bl_mynn_mixlength__ == 1) {
/* Computing 2nd power */
	r__1 = u1[*kts];
/* Computing 2nd power */
	r__2 = v1[*kts];
	ugrid = (real)sqrt((real)(r__1 * r__1 + r__2 * r__2));
	uonset = rc0017;
/* Computing MIN */
/* Computing MAX */
	r__3 = zero, r__4 = ugrid - uonset;
	r__1 = rc0018, r__2 = max(r__3,r__4) / rc0019;
	wt_u1__ = one - rc0016 * min(r__1,r__2);
/* Computing MIN */
/* Computing MAX */
	r__3 = zero, r__4 = ugrid - uonset;
	r__1 = rc0018, r__2 = max(r__3,r__4) / rc0019;
	wt_u2__ = one - rc0020 * min(r__1,r__2);
	cns = rc0021;
	alp1 = rc0007;
	alp2 = rc0009;
	alp3 = rc0022 * wt_u2__;
	alp4 = rc0008;
	alp5 = rc0009;
	alp6 = rc0023;
	pblh2 = max(*pblh,rc0000);
/* Computing MAX */
	r__1 = rc0009 * pblh2;
	h1 = max(r__1,rc0000);
	h1 = min(h1,rc0024);
	h2 = h1 / rc0011;
/* Computing MAX */
	r__1 = half * qke[*kts], r__2 = half * qkemin;
	qtke[*kts - 1] = max(r__1,r__2);
	thetaw[*kts - 1] = theta[*kts];
/* Computing MAX */
	r__1 = qke[*kts];
	qkw[*kts] = (real)sqrt((real)(max(r__1,qkemin)));
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    afk = dz[k] / (dz[k] + dz[k - 1]);
	    abk = one - afk;
/* Computing MAX */
	    r__1 = qke[k] * abk + qke[k - 1] * afk;
	    qkw[k] = (real)sqrt((real)(max(r__1,qkemin)));
/* Computing MAX */
/* Computing 2nd power */
	    r__2 = qkw[k];
	    r__1 = half * (r__2 * r__2);
	    qtke[k - 1] = max(r__1,rc0025);
	    thetaw[k - 1] = theta[k] * abk + theta[k - 1] * afk;
/* L9013: */
	}
/* L9014: */
	elt = rc0012;
	vsc = rc0012;
	k = *kts + 1;
	zwk = zw[k];
L9015:
	if (! (zwk <= pblh2 + h1)) {
	    goto L9016;
	}
	dzk = half * (dz[k] + dz[k - 1]);
/* Computing MIN */
/* Computing MAX */
	r__2 = qkw[k] - qmin;
	r__1 = max(r__2,rc0026);
	qdz = min(r__1,rc0027) * dzk;
	elt += qdz * zwk;
	vsc += qdz;
	++k;
	zwk = zw[k];
	goto L9015;
L9016:
	if (*xland - rc0028 >= zero) {
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = ugrid - rc0019;
	    r__1 = one, r__2 = max(r__3,r__4) / rc0030;
	    elt_max__ = rc0029 + rc0002 * min(r__1,r__2);
	} else {
	    elt_max__ = rc0031;
	}
/* Computing MIN */
/* Computing MAX */
	r__2 = alp1 * elt / vsc;
	r__1 = max(r__2,rc0032);
	elt = min(r__1,elt_max__);
	vflx = *fltv;
	d__1 =  (gtr * elt * max(vflx,rc0015));
	d__2 =  onethird;
	vsc = (real)pow_dd(&d__1, &d__2);
	el[*kts] = rc0015;
	zwk1 = zw[*kts + 1];
	boulac_length__(m471_work,m471_n, kts, kte, &zw[zw_offset], &dz[dz_offset], qtke, 
		thetaw, elblmin, elblavg);
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    zwk = zw[k];
/* Computing MAX */
	    r__2 = edmf_a1__[k - 1] * edmf_w1__[k - 1], r__3 = (r__1 = 
		    edmf_a_dd1__[k - 1] * edmf_w_dd1__[k - 1], abs(r__1));
	    qkw_mf__ = max(r__2,r__3);
	    if (dtv[k] > rc0015) {
/* Computing MAX */
		r__1 = (real)sqrt((real)(gtr * dtv[k]));
		bv = max(r__1,rc0033);
		elb_mf__ = alp6 * qkw_mf__ / bv;
		elb_mf__ /= rc0034 + elb_mf__ / rc0002;
/* Computing MAX */
		r__1 = qkw[k];
		elb = alp2 * max(r__1,qkw_elb_min__) / bv * (one + alp3 * (
			real)sqrt((real)(vsc / (bv * elt))));
		elb = max(elb,elb_mf__);
		elb = min(elb,zwk);
/* Computing MAX */
		r__1 = qkw[k];
		elf = one * max(r__1,qkw_elb_min__) / bv;
		elf = max(elf,elb_mf__);
/* Computing MAX */
		r__1 = elblavg[k - 1];
		elblavg[k - 1] = max(r__1,elb_mf__);
	    } else {
		elb = rc0014;
		elf = elb;
	    }
	    if (*rmol > rc0015) {
/* Computing MIN */
		r__1 = zwk * *rmol;
		els = karman * zwk / (one + cns * min(r__1,zmax));
	    } else {
		d__1 =  (one - alp4 * zwk * *rmol);
		d__2 =  rc0016;
		els = karman * zwk * (real)pow_dd(&d__1, &d__2);
	    }
	    wt = half * (real)tanh((real)((zwk - (pblh2 + h1)) / h2)) + half;
/* Computing 2nd power */
	    r__1 = els;
/* Computing 2nd power */
	    r__2 = els;
/* Computing 2nd power */
	    r__3 = elt;
	    el[k] = (real)sqrt((real)(r__1 * r__1 / (rc0034 + r__2 * r__2 / (
		    r__3 * r__3))));
/* Computing MIN */
	    r__1 = el[k];
	    el[k] = min(r__1,elb);
/* Computing MIN */
	    r__1 = el[k];
	    el[k] = min(r__1,elf);
	    if (*xland - rc0028 >= zero) {
		el[k] *= wt_u1__;
	    }
	    el[k] = el[k] * (rc0034 - wt) + alp5 * elblavg[k - 1] * wt;
/* L9019: */
	}
/* L9020: */
    } else if (*bl_mynn_mixlength__ == 2) {
	uonset = rc0021 + dz[*kts] * rc0035;
/* Computing 2nd power */
	r__1 = u1[*kts];
/* Computing 2nd power */
	r__2 = v1[*kts];
	ugrid = (real)sqrt((real)(r__1 * r__1 + r__2 * r__2));
	cns = rc0021;
	alp1 = rc0036;
	alp2 = rc0037;
	alp3 = rc0022;
	alp4 = rc0008;
	alp5 = alp2;
	alp6 = rc0019;
	pblh2 = max(*pblh,rc0000);
/* Computing MAX */
	r__1 = rc0009 * pblh2;
	h1 = max(r__1,rc0000);
	h1 = min(h1,rc0024);
	h2 = h1 * half;
/* Computing MAX */
	r__1 = half * qke[*kts], r__2 = half * qkemin;
	qtke[*kts - 1] = max(r__1,r__2);
/* Computing MAX */
	r__1 = qke[*kts];
	qkw[*kts] = (real)sqrt((real)(max(r__1,qkemin)));
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    afk = dz[k] / (dz[k] + dz[k - 1]);
	    abk = one - afk;
/* Computing MAX */
	    r__1 = qke[k] * abk + qke[k - 1] * afk;
	    qkw[k] = (real)sqrt((real)(max(r__1,qkemin)));
/* Computing 2nd power */
	    r__1 = qkw[k];
	    qtke[k - 1] = half * (r__1 * r__1);
/* L9022: */
	}
/* L9023: */
	elt = rc0012;
	vsc = rc0012;
/* Computing MAX */
	r__1 = *pblh + h1;
	pblh_plus_ent__ = max(r__1,rc0002);
	k = *kts + 1;
	zwk = zw[k];
L9024:
	if (! (zwk <= pblh_plus_ent__)) {
	    goto L9025;
	}
	dzk = half * (dz[k] + dz[k - 1]);
/* Computing MIN */
/* Computing MAX */
	r__2 = qkw[k] - qmin;
	r__1 = max(r__2,rc0013);
	qdz = min(r__1,rc0027) * dzk;
	elt += qdz * zwk;
	vsc += qdz;
	++k;
	zwk = zw[k];
	goto L9024;
L9025:
/* Computing MIN */
/* Computing MAX */
	r__2 = alp1 * elt / vsc;
	r__1 = max(r__2,rc0038);
	elt = min(r__1,rc0031);
	vflx = *fltv;
	d__1 =  (gtr * elt * max(vflx,rc0015));
	d__2 =  onethird;
	vsc = (real)pow_dd(&d__1, &d__2);
	el[*kts] = rc0015;
	zwk1 = zw[*kts + 1];
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    zwk = zw[k];
	    dzk = rc0039 * (dz[k] + dz[k - 1]);
	    cldavg = rc0039 * (cldfra_bl1__[k - 1] + cldfra_bl1__[k]);
/* Computing MAX */
	    r__2 = edmf_a1__[k - 1] * edmf_w1__[k - 1], r__3 = (r__1 = 
		    edmf_a_dd1__[k - 1] * edmf_w_dd1__[k - 1], abs(r__1));
	    qkw_mf__ = max(r__2,r__3);
	    if (dtv[k] > rc0015) {
/* Computing MAX */
		r__1 = (real)sqrt((real)(gtr * dtv[k]));
		bv = max(r__1,rc0033);
/* Computing MAX */
		r__1 = alp2 * qkw[k], r__2 = alp6 * qkw_mf__;
		elb_mf__ = max(r__1,r__2) / bv * (one + alp3 * (real)sqrt((
			real)(vsc / (bv * elt))));
/* Computing MIN */
/* Computing MAX */
		r__2 = alp5 * qkw[k], r__3 = alp6 * qkw_mf__;
		r__1 = max(r__2,r__3) / bv;
		elb = min(r__1,zwk);
		d__1 =  (gtr * *pblh * max(vflx,rc0041));
		d__2 =  onethird;
		wstar = rc0040 * (real)pow_dd(&d__1, &d__2);
/* Computing MIN */
/* Computing MAX */
		r__2 = ctau * wstar / grav;
		r__1 = max(r__2,rc0042);
		tau_cloud__ = min(r__1,rc0043);
		wt = half * (real)tanh((real)((zwk - (pblh2 + h1)) / h2)) + 
			half;
		tau_cloud__ = tau_cloud__ * (rc0034 - wt) + rc0023 * wt;
/* Computing MIN */
/* Computing MAX */
/* Computing MIN */
		r__4 = qtke[k - 1];
		r__2 = tau_cloud__ * (real)sqrt((real)(min(r__4,rc0044))), 
			r__3 = alp6 * qkw_mf__ / bv;
		r__1 = max(r__2,r__3);
		elf = min(r__1,zwk);
	    } else {
		d__1 =  (gtr * *pblh * max(vflx,rc0041));
		d__2 =  onethird;
		wstar = rc0040 * (real)pow_dd(&d__1, &d__2);
/* Computing MIN */
/* Computing MAX */
		r__2 = ctau * wstar / grav;
		r__1 = max(r__2,rc0023);
		tau_cloud__ = min(r__1,rc0045);
		wt = half * (real)tanh((real)((zwk - (pblh2 + h1)) / h2)) + 
			half;
/* Computing MAX */
		r__1 = rc0002, r__2 = dzk * rc0046;
		tau_cloud__ = tau_cloud__ * (rc0034 - wt) + max(r__1,r__2) * 
			wt;
/* Computing MIN */
/* Computing MIN */
		r__2 = qtke[k - 1];
		r__1 = tau_cloud__ * (real)sqrt((real)(min(r__2,rc0044)));
		elb = min(r__1,zwk);
		elf = elb;
		elb_mf__ = elb;
	    }
	    elf /= rc0034 + elf / rc0047;
	    elb_mf__ = max(elb_mf__,rc0026);
	    if (*rmol > rc0015) {
/* Computing MIN */
		r__1 = zwk * *rmol;
		els = karman * zwk / (one + cns * min(r__1,zmax));
	    } else {
		d__1 =  (one - alp4 * zwk * *rmol);
		d__2 =  rc0016;
		els = karman * zwk * (real)pow_dd(&d__1, &d__2);
	    }
	    wt = half * (real)tanh((real)((zwk - (pblh2 + h1)) / h2)) + half;
/* Computing 2nd power */
	    r__1 = els;
/* Computing 2nd power */
	    r__2 = els;
/* Computing 2nd power */
	    r__3 = elt;
/* Computing 2nd power */
	    r__4 = els;
/* Computing 2nd power */
	    r__5 = elb_mf__;
	    el[k] = (real)sqrt((real)(r__1 * r__1 / (rc0034 + r__2 * r__2 / (
		    r__3 * r__3) + r__4 * r__4 / (r__5 * r__5))));
	    el[k] = el[k] * (rc0034 - wt) + elf * wt;
/* L9028: */
	}
/* L9029: */
    }
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	el_les__ = rc0046 * half * (dz[k] + dz[k - 1]);
/* Computing MIN */
	r__1 = el_les__, r__2 = el[k];
	el[k] = el[k] * *psig_bl__ + (rc0034 - *psig_bl__) * min(r__1,r__2);
/* L9031: */
    }
/* L9032: */
    return 0;
} /* mym_length__ */

__device__ M471_CALL /* Subroutine */ int boulac_length0__(unsigned char *m471_work, int m471_n, M471Ptr<integer> k, M471Ptr<integer> kts, M471Ptr<integer> kte, 
	M471Ptr<real> zw, M471Ptr<real> dz, M471Ptr<real> qtke, M471Ptr<real> theta, M471Ptr<real> lb1, M471Ptr<real> lb2) {
    /* Initialized data */

    real gtr = .032700002193450927734375f;
    real rc0000 = 0.f;
    real rc0001 = .5f;
    real rc0002 = 2.f;
    real rc0003 = .1f;
    real rc0004 = 1e3f;

    /* System generated locals */
    integer qtke_offset, dz_offset, theta_offset, zw_offset;
    real r__1, r__2, r__3;

    /* Builtin functions */
    

    /* Local variables */
    real tl, bbb, dld, dlu, zdo, dzt;
    integer izz;
    real zup, zzz, beta;
    integer found;
    real zup_inf__, zdo_sup__;

    /* Parameter adjustments */
    theta_offset = *kts;
    theta -= theta_offset;
    qtke_offset = *kts;
    qtke -= qtke_offset;
    dz_offset = *kts;
    dz -= dz_offset;
    zw_offset = *kts;
    zw -= zw_offset;

    /* Function Body */
    zup = rc0000;
    dlu = zw[*kte + 1] - zw[*k] - dz[*k] * rc0001;
    zzz = rc0000;
    zup_inf__ = rc0000;
    beta = gtr;
    if (*k < *kte) {
	found = 0;
	izz = *k;
L9003:
	if (! (found == 0)) {
	    goto L9004;
	}
	if (izz < *kte) {
	    dzt = dz[izz];
	    zup -= beta * theta[*k] * dzt;
	    zup += beta * (theta[izz + 1] + theta[izz]) * dzt * rc0001;
	    zzz += dzt;
	    if (qtke[*k] < zup && qtke[*k] >= zup_inf__) {
		bbb = (theta[izz + 1] - theta[izz]) / dzt;
		if (bbb != rc0000) {
/* Computing MAX */
/* Computing 2nd power */
		    r__3 = beta * (theta[izz] - theta[*k]);
		    r__1 = rc0000, r__2 = r__3 * r__3 + rc0002 * bbb * beta * 
			    (qtke[*k] - zup_inf__);
		    tl = (-beta * (theta[izz] - theta[*k]) + (real)sqrt((real)
			    (max(r__1,r__2)))) / bbb / beta;
		} else {
		    if (theta[izz] != theta[*k]) {
			tl = (qtke[*k] - zup_inf__) / (beta * (theta[izz] - 
				theta[*k]));
		    } else {
			tl = rc0000;
		    }
		}
		dlu = zzz - dzt + tl;
		found = 1;
	    }
	    zup_inf__ = zup;
	    ++izz;
	} else {
	    found = 1;
	}
	goto L9003;
L9004:
	;
    }
    zdo = rc0000;
    zdo_sup__ = rc0000;
    dld = zw[*k];
    zzz = rc0000;
    if (*k > *kts) {
	found = 0;
	izz = *k;
L9006:
	if (! (found == 0)) {
	    goto L9007;
	}
	if (izz > *kts) {
	    dzt = dz[izz - 1];
	    zdo += beta * theta[*k] * dzt;
	    zdo -= beta * (theta[izz - 1] + theta[izz]) * dzt * rc0001;
	    zzz += dzt;
	    if (qtke[*k] < zdo && qtke[*k] >= zdo_sup__) {
		bbb = (theta[izz] - theta[izz - 1]) / dzt;
		if (bbb != rc0000) {
/* Computing MAX */
/* Computing 2nd power */
		    r__3 = beta * (theta[izz] - theta[*k]);
		    r__1 = rc0000, r__2 = r__3 * r__3 + rc0002 * bbb * beta * 
			    (qtke[*k] - zdo_sup__);
		    tl = (beta * (theta[izz] - theta[*k]) + (real)sqrt((real)(
			    max(r__1,r__2)))) / bbb / beta;
		} else {
		    if (theta[izz] != theta[*k]) {
			tl = (qtke[*k] - zdo_sup__) / (beta * (theta[izz] - 
				theta[*k]));
		    } else {
			tl = rc0000;
		    }
		}
		dld = zzz - dzt + tl;
		found = 1;
	    }
	    zdo_sup__ = zdo;
	    --izz;
	} else {
	    found = 1;
	}
	goto L9006;
L9007:
	;
    }
/* Computing MIN */
    r__1 = dld, r__2 = zw[*k + 1];
    dld = min(r__1,r__2);
    *lb1 = min(dlu,dld);
/* Computing MAX */
    r__1 = rc0003, r__2 = min(dlu,rc0004);
    dlu = max(r__1,r__2);
/* Computing MAX */
    r__1 = rc0003, r__2 = min(dld,rc0004);
    dld = max(r__1,r__2);
    *lb2 = (real)sqrt((real)(dlu * dld));
    if (*k == *kte) {
	*lb1 = rc0000;
	*lb2 = rc0000;
    }
    return 0;
} /* boulac_length0__ */

__device__ M471_CALL /* Subroutine */ int boulac_length__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> zw, 
	M471Ptr<real> dz, M471Ptr<real> qtke, M471Ptr<real> theta, M471Ptr<real> lb1, M471Ptr<real> lb2) {
    /* Initialized data */

    real gtr = .032700002193450927734375f;
    real rc0000 = 1500.f;
    real rc0001 = 0.f;
    real rc0002 = .5f;
    real rc0003 = 2.f;
    real rc0004 = .1f;
    real rc0005 = 1.f;

    /* System generated locals */
    integer qtke_offset, dz_offset, theta_offset, lb1_offset, lb2_offset, 
	    zw_offset, i__1;
    real r__1, r__2, r__3;

    /* Builtin functions */
    

    /* Local variables */
    real tl;
    integer iz;
    real bbb, zdo, dzt;
M471Ptr<real> dld=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+17200),m471_n);
M471Ptr<real> dlu=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+17400),m471_n);
    integer izz;
    real zup, zzz, beta, lmax;
    integer found;
    real zup_inf__, zdo_sup__;

    /* Parameter adjustments */
    lb2_offset = *kts;
    lb2 -= lb2_offset;
    lb1_offset = *kts;
    lb1 -= lb1_offset;
    theta_offset = *kts;
    theta -= theta_offset;
    qtke_offset = *kts;
    qtke -= qtke_offset;
    dz_offset = *kts;
    dz -= dz_offset;
    zw_offset = *kts;
    zw -= zw_offset;

    /* Function Body */
    lmax = rc0000;
    i__1 = *kte;
    
#pragma unroll 1
for (iz = *kts; iz <= i__1; ++iz) {
	zup = rc0001;
	dlu[iz - 1] = zw[*kte + 1] - zw[iz] - dz[iz] * rc0002;
	zzz = rc0001;
	zup_inf__ = rc0001;
	beta = gtr;
	if (iz < *kte) {
	    found = 0;
	    izz = iz;
L9006:
	    if (! (found == 0)) {
		goto L9007;
	    }
	    if (izz < *kte) {
		dzt = dz[izz];
		zup -= beta * theta[iz] * dzt;
		zup += beta * (theta[izz + 1] + theta[izz]) * dzt * rc0002;
		zzz += dzt;
		if (qtke[iz] < zup && qtke[iz] >= zup_inf__) {
		    bbb = (theta[izz + 1] - theta[izz]) / dzt;
		    if (bbb != rc0001) {
/* Computing MAX */
/* Computing 2nd power */
			r__3 = beta * (theta[izz] - theta[iz]);
			r__1 = rc0001, r__2 = r__3 * r__3 + rc0003 * bbb * 
				beta * (qtke[iz] - zup_inf__);
			tl = (-beta * (theta[izz] - theta[iz]) + (real)sqrt((
				real)(max(r__1,r__2)))) / bbb / beta;
		    } else {
			if (theta[izz] != theta[iz]) {
			    tl = (qtke[iz] - zup_inf__) / (beta * (theta[izz] 
				    - theta[iz]));
			} else {
			    tl = rc0001;
			}
		    }
		    dlu[iz - 1] = zzz - dzt + tl;
		    found = 1;
		}
		zup_inf__ = zup;
		++izz;
	    } else {
		found = 1;
	    }
	    goto L9006;
L9007:
	    ;
	}
	zdo = rc0001;
	zdo_sup__ = rc0001;
	dld[iz - 1] = zw[iz];
	zzz = rc0001;
	if (iz > *kts) {
	    found = 0;
	    izz = iz;
L9009:
	    if (! (found == 0)) {
		goto L9010;
	    }
	    if (izz > *kts) {
		dzt = dz[izz - 1];
		zdo += beta * theta[iz] * dzt;
		zdo -= beta * (theta[izz - 1] + theta[izz]) * dzt * rc0002;
		zzz += dzt;
		if (qtke[iz] < zdo && qtke[iz] >= zdo_sup__) {
		    bbb = (theta[izz] - theta[izz - 1]) / dzt;
		    if (bbb != rc0001) {
/* Computing MAX */
/* Computing 2nd power */
			r__3 = beta * (theta[izz] - theta[iz]);
			r__1 = rc0001, r__2 = r__3 * r__3 + rc0003 * bbb * 
				beta * (qtke[iz] - zdo_sup__);
			tl = (beta * (theta[izz] - theta[iz]) + (real)sqrt((
				real)(max(r__1,r__2)))) / bbb / beta;
		    } else {
			if (theta[izz] != theta[iz]) {
			    tl = (qtke[iz] - zdo_sup__) / (beta * (theta[izz] 
				    - theta[iz]));
			} else {
			    tl = rc0001;
			}
		    }
		    dld[iz - 1] = zzz - dzt + tl;
		    found = 1;
		}
		zdo_sup__ = zdo;
		--izz;
	    } else {
		found = 1;
	    }
	    goto L9009;
L9010:
	    ;
	}
/* Computing MIN */
	r__1 = dld[iz - 1], r__2 = zw[iz + 1];
	dld[iz - 1] = min(r__1,r__2);
/* Computing MAX */
	r__1 = rc0004, r__2 = dlu[iz - 1] / (rc0005 + dlu[iz - 1] / lmax);
	dlu[iz - 1] = max(r__1,r__2);
/* Computing MAX */
	r__1 = rc0004, r__2 = dld[iz - 1] / (rc0005 + dld[iz - 1] / lmax);
	dld[iz - 1] = max(r__1,r__2);
/* Computing MIN */
	r__1 = dlu[iz - 1], r__2 = dld[iz - 1];
	lb1[iz] = min(r__1,r__2);
	lb2[iz] = (real)sqrt((real)(dlu[iz - 1] * dld[iz - 1]));
/* L9004: */
    }
/* L9005: */
    lb1[*kte] = lb1[*kte - 1];
    lb2[*kte] = lb2[*kte - 1];
    return 0;
} /* boulac_length__ */

__device__ M471_CALL /* Subroutine */ int mym_turbulence__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> xland,
	 M471Ptr<real> closure, M471Ptr<real> dz, M471Ptr<real> dx, M471Ptr<real> zw, M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> thl, M471Ptr<real> thv, M471Ptr<real> thlv, M471Ptr<real> ql, M471Ptr<real> qw, M471Ptr<real> qke, M471Ptr<real> tsq, 
	M471Ptr<real> qsq, M471Ptr<real> cov, M471Ptr<real> vt, M471Ptr<real> vq, M471Ptr<real> rmol, M471Ptr<real> flt, M471Ptr<real> fltv, M471Ptr<real> flq, M471Ptr<real> pblh, M471Ptr<real> theta, M471Ptr<real> sh, M471Ptr<real> sm, M471Ptr<real> el, M471Ptr<real> dfm, M471Ptr<real> dfh, M471Ptr<real> dfq, M471Ptr<real> tcd, M471Ptr<real> qcd, M471Ptr<real> pdk, 
	M471Ptr<real> pdt, M471Ptr<real> pdq, M471Ptr<real> pdc, M471Ptr<real> qwt1, M471Ptr<real> qshear1, M471Ptr<real> qbuoy1, M471Ptr<real> qdiss1, M471Ptr<integer> tke_budget__, M471Ptr<real> psig_bl__, M471Ptr<real> psig_shcu__, M471Ptr<real> cldfra_bl1__, M471Ptr<integer> bl_mynn_mixlength__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w_dd1__, M471Ptr<real> edmf_a_dd1__, 
	M471Ptr<real> tkeprod_dn__, M471Ptr<real> tkeprod_up__, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__) {
    /* Initialized data */

    real zero = 0.f;
    real half = .5f;
    real one = 1.f;
    real two = 2.f;
    real gtr = .032700002193450927734375f;
    real tv0 = 182.50872802734375f;
    real b1 = 24.f;
    real b2 = 15.f;
    real a1 = 1.1800000667572021484375f;
    real c1 = .137067615985870361328125f;
    real a2 = .664521038532257080078125f;
    real cc3 = .65999996662139892578125f;
    real e1c = 19.73627471923828125f;
    real e2c = 1.91250514984130859375f;
    real e3c = .861627042293548583984375f;
    real e4c = 2.550006866455078125f;
    real e5c = 8.354400634765625f;
    real ckmod = 1.f;
    real rc0000 = 5.f;
    real rc0001 = .5f;
    real rc0002 = 1e-5f;
    real rc0003 = 1e-10f;
    real rc0004 = .76f;
    real rc0005 = 4.f;
    real rc0006 = 7.f;
    real rc0007 = .01f;
    real rc0008 = 6.873f;
    real rc0009 = 0.f;
    real rc0010 = 3.f;
    real rc0011 = 1e-8f;
    real rc0012 = 4.f;
    real rc0013 = .02f;
    real rc0014 = 27.f;
    real rc0015 = 54.f;
    real rc0016 = 9.f;
    real rc0017 = 12.f;
    real rc0018 = 3.f;
    real rc0019 = 18.f;
    real rc0020 = .12f;
    real rc0021 = .3f;
    real rc0022 = .1f;
    real rc0023 = 2.f;
    real rc0024 = .04f;
    real rc0025 = 10.f;
    real rc0026 = 1.5f;
    real rc0027 = 8e3f;
    real rc0028 = 2e3f;
    real rc0029 = .001f;

    /* System generated locals */
    integer dz_offset, zw_offset, u_offset, v_offset, thl_offset, thv_offset, 
	    thlv_offset, qw_offset, ql_offset, vt_offset, vq_offset, 
	    qke_offset, tsq_offset, qsq_offset, cov_offset, cldfra_bl1_offset,
	     edmf_w1_offset, edmf_a1_offset, edmf_w_dd1_offset, 
	    edmf_a_dd1_offset, tkeprod_dn_offset, tkeprod_up_offset, 
	    dfm_offset, dfh_offset, dfq_offset, pdk_offset, pdt_offset, 
	    pdq_offset, pdc_offset, tcd_offset, qcd_offset, el_offset, 
	    qwt1_offset, qshear1_offset, qbuoy1_offset, qdiss1_offset, 
	    sm_offset, sh_offset, theta_offset, pattern_spp_pbl1_offset, i__1;
    real r__1, r__2, r__3;
    doublereal d__1, d__2, d__3, d__4;

    /* Builtin functions */
    

    /* Local variables */
    integer k;
    doublereal e1, e2, e3, e4;
    real  ri;
M471Ptr<real> gh=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+17600),m471_n);
M471Ptr<real> gm=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+17800),m471_n);
    
    real e6c;
    
    real adh, abk, aeh, afk, adm, aem, elh, sh20, auh, shb, dzk, vqq, cw25, vtt, smd, elq, duz, aum, req, 
	    rsl, sm20;
M471Ptr<real> dtl=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+18000),m471_n);
M471Ptr<real> dqw=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+18200),m471_n);
M471Ptr<real> dtv=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+18400),m471_n);
M471Ptr<real> qkw=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+18600),m471_n);
    doublereal c2sq, c3sq;
    real rsl2;
    doublereal q2sq, r2sq, q3sq, t2sq, t3sq, r3sq, eden, ghel;
    real gamq;
    doublereal gmel;
    real gamt, gamv;
    doublereal wden, dlsq;
    real clow, cupp, a2fac;
    doublereal elsq, qdiv, enum__;
    real gmelq, mfmax, prlim, prnum, cldavg, sh25min, sh25max, sm25min, 
	    sm25max, prlimit;

    /* Parameter adjustments */
    pattern_spp_pbl1_offset = *kts;
    pattern_spp_pbl1__ -= pattern_spp_pbl1_offset;
    tkeprod_up_offset = *kts;
    tkeprod_up__ -= tkeprod_up_offset;
    tkeprod_dn_offset = *kts;
    tkeprod_dn__ -= tkeprod_dn_offset;
    edmf_a_dd1_offset = *kts;
    edmf_a_dd1__ -= edmf_a_dd1_offset;
    edmf_w_dd1_offset = *kts;
    edmf_w_dd1__ -= edmf_w_dd1_offset;
    edmf_a1_offset = *kts;
    edmf_a1__ -= edmf_a1_offset;
    edmf_w1_offset = *kts;
    edmf_w1__ -= edmf_w1_offset;
    cldfra_bl1_offset = *kts;
    cldfra_bl1__ -= cldfra_bl1_offset;
    qdiss1_offset = *kts;
    qdiss1 -= qdiss1_offset;
    qbuoy1_offset = *kts;
    qbuoy1 -= qbuoy1_offset;
    qshear1_offset = *kts;
    qshear1 -= qshear1_offset;
    qwt1_offset = *kts;
    qwt1 -= qwt1_offset;
    pdc_offset = *kts;
    pdc -= pdc_offset;
    pdq_offset = *kts;
    pdq -= pdq_offset;
    pdt_offset = *kts;
    pdt -= pdt_offset;
    pdk_offset = *kts;
    pdk -= pdk_offset;
    qcd_offset = *kts;
    qcd -= qcd_offset;
    tcd_offset = *kts;
    tcd -= tcd_offset;
    dfq_offset = *kts;
    dfq -= dfq_offset;
    dfh_offset = *kts;
    dfh -= dfh_offset;
    dfm_offset = *kts;
    dfm -= dfm_offset;
    el_offset = *kts;
    el -= el_offset;
    sm_offset = *kts;
    sm -= sm_offset;
    sh_offset = *kts;
    sh -= sh_offset;
    theta_offset = *kts;
    theta -= theta_offset;
    vq_offset = *kts;
    vq -= vq_offset;
    vt_offset = *kts;
    vt -= vt_offset;
    cov_offset = *kts;
    cov -= cov_offset;
    qsq_offset = *kts;
    qsq -= qsq_offset;
    tsq_offset = *kts;
    tsq -= tsq_offset;
    qke_offset = *kts;
    qke -= qke_offset;
    qw_offset = *kts;
    qw -= qw_offset;
    ql_offset = *kts;
    ql -= ql_offset;
    thlv_offset = *kts;
    thlv -= thlv_offset;
    thv_offset = *kts;
    thv -= thv_offset;
    thl_offset = *kts;
    thl -= thl_offset;
    v_offset = *kts;
    v -= v_offset;
    u_offset = *kts;
    u -= u_offset;
    zw_offset = *kts;
    zw -= zw_offset;
    dz_offset = *kts;
    dz -= dz_offset;

    /* Function Body */
    prlimit = rc0000;
    mym_level2__(m471_work,m471_n, kts, kte, &dz[dz_offset], &u[u_offset], &v[v_offset], &thl[
	    thl_offset], &thv[thv_offset], &thlv[thlv_offset], &qw[qw_offset],
	     &ql[ql_offset], &vt[vt_offset], &vq[vq_offset], dtl, dqw, dtv, 
	    gm, gh, &sm[sm_offset], &sh[sh_offset]);
    mym_length__(m471_work,m471_n, kts, kte, xland, &dz[dz_offset], dx, &zw[zw_offset], rmol, 
	    flt, fltv, flq, &vt[vt_offset], &vq[vq_offset], &u[u_offset], &v[
	    v_offset], &qke[qke_offset], dtv, &el[el_offset], pblh, &theta[
	    theta_offset], qkw, psig_bl__, &cldfra_bl1__[cldfra_bl1_offset], 
	    bl_mynn_mixlength__, &edmf_w1__[edmf_w1_offset], &edmf_a1__[
	    edmf_a1_offset], &edmf_w_dd1__[edmf_w_dd1_offset], &edmf_a_dd1__[
	    edmf_a_dd1_offset]);
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	dzk = rc0001 * (dz[k] + dz[k - 1]);
	afk = dz[k] / (dz[k] + dz[k - 1]);
	abk = one - afk;
/* Computing 2nd power */
	r__1 = el[k];
	elsq = r__1 * r__1;
/* Computing 2nd power */
	r__1 = qkw[k - 1];
	q3sq = r__1 * r__1;
	q2sq = b1 * elsq * (sm[k] * gm[k - 1] + sh[k] * gh[k - 1]);
/* Computing MAX */
	r__1 = sh[k];
	sh20 = max(r__1,rc0002);
/* Computing MAX */
	r__1 = sm[k];
	sm20 = max(r__1,rc0002);
/* Computing MAX */
	r__1 = sh[k];
	sh[k] = max(r__1,rc0002);
/* Computing 2nd power */
	r__1 = u[k] - u[k - 1];
/* Computing 2nd power */
	r__2 = v[k] - v[k - 1];
	duz = r__1 * r__1 + r__2 * r__2;
/* Computing 2nd power */
	r__1 = dzk;
	duz /= r__1 * r__1;
	ri = -gh[k - 1] / max(duz,rc0003);
	if (ckmod == 1.f) {
	    a2fac = one / (one + max(ri,zero));
	} else {
	    a2fac = one;
	}
/* Computing MIN */
	r__1 = rc0004 + rc0005 * max(ri,zero);
	prnum = min(r__1,prlimit);
	if (ri >= one) {
	    prlim = rc0006 * ri;
	} else if (ri >= rc0007 && ri <= one) {
	    prlim = rc0008 * ri + one / (rc0008 * ri);
	} else {
	    prlim = prlimit;
	}
	gmel = gm[k - 1] * elsq;
	ghel = gh[k - 1] * elsq;
	if (FALSE_) {
	    if (sh[k] < rc0009 || sm[k] < rc0009) {
	    }
	}
	dlsq = elsq;
	if (q3sq / dlsq < -gh[k - 1]) {
	    q3sq = -dlsq * gh[k - 1];
	}
	if (q3sq < q2sq) {
	    qdiv = sqrt(q3sq / q2sq);
	    sh[k] *= qdiv;
	    sm[k] *= qdiv;
/* Computing 2nd power */
	    d__1 = qdiv;
	    e1 = q3sq - e1c * ghel * a2fac * (d__1 * d__1);
/* Computing 2nd power */
	    d__1 = qdiv;
	    e2 = q3sq - e2c * ghel * a2fac * (d__1 * d__1);
/* Computing 2nd power */
	    r__1 = a2fac;
/* Computing 2nd power */
	    d__1 = qdiv;
	    e3 = e1 + e3c * ghel * (r__1 * r__1) * (d__1 * d__1);
/* Computing 2nd power */
	    d__1 = qdiv;
	    e4 = e1 - e4c * ghel * a2fac * (d__1 * d__1);
/* Computing 2nd power */
	    d__1 = qdiv;
	    eden = e2 * e4 + e3 * e5c * gmel * (d__1 * d__1);
	    eden = max(eden,1e-20);
	} else {
	    e1 = q3sq - e1c * ghel * a2fac;
	    e2 = q3sq - e2c * ghel * a2fac;
/* Computing 2nd power */
	    r__1 = a2fac;
	    e3 = e1 + e3c * ghel * (r__1 * r__1);
	    e4 = e1 - e4c * ghel * a2fac;
	    eden = e2 * e4 + e3 * e5c * gmel;
	    eden = max(eden,1e-20);
	    qdiv = one;
	    sm[k] = q3sq * a1 * (e3 - rc0010 * c1 * e4) / eden;
	    sh[k] = q3sq * (a2 * a2fac) * (e2 + rc0010 * c1 * e5c * gmel) / 
		    eden;
	}
/* Computing MAX */
	r__1 = (real) (gmel / q3sq);
	gmelq = max(r__1,rc0011);
	sm25max = rc0012;
	sh25max = rc0012;
	sm25min = zero;
	sh25min = zero;
	if (FALSE_) {
	    if (sh[k] < sh25min || sm[k] < sm25min || sh[k] > sh25max || sm[k]
		     > sm25max) {
	    }
	}
	if (sh[k] > sh25max) {
	    sh[k] = sh25max;
	}
	if (sh[k] < sh25min) {
	    sh[k] = sh25min;
	}
/* Computing MAX */
	r__1 = sh[k];
	shb = max(r__1,rc0013);
/* Computing MIN */
	r__1 = sm[k], r__2 = prlim * shb;
	sm[k] = min(r__1,r__2);
	if (*closure >= rc0010) {
/* Computing 2nd power */
	    r__1 = dtl[k - 1];
	    t2sq = qdiv * b2 * elsq * sh[k] * (r__1 * r__1);
/* Computing 2nd power */
	    r__1 = dqw[k - 1];
	    r2sq = qdiv * b2 * elsq * sh[k] * (r__1 * r__1);
	    c2sq = qdiv * b2 * elsq * sh[k] * dtl[k - 1] * dqw[k - 1];
/* Computing MAX */
	    r__1 = tsq[k] * abk + tsq[k - 1] * afk;
	    t3sq = max(r__1,zero);
/* Computing MAX */
	    r__1 = qsq[k] * abk + qsq[k - 1] * afk;
	    r3sq = max(r__1,zero);
	    c3sq = cov[k] * abk + cov[k - 1] * afk;
/* Computing MIN */
	    d__2 = abs(c3sq), d__3 = sqrt(t3sq * r3sq);
	    d__1 = min(d__2,d__3);
	    c3sq = d_sign(&d__1, &c3sq);
	    vtt = one + vt[k] * abk + vt[k - 1] * afk;
	    vqq = tv0 + vq[k] * abk + vq[k - 1] * afk;
	    t2sq = vtt * t2sq + vqq * c2sq;
	    r2sq = vtt * c2sq + vqq * r2sq;
/* Computing MAX */
	    d__1 = vtt * t2sq + vqq * r2sq;
	    c2sq = max(d__1,0.);
	    t3sq = vtt * t3sq + vqq * c3sq;
	    r3sq = vtt * c3sq + vqq * r3sq;
/* Computing MAX */
	    d__1 = vtt * t3sq + vqq * r3sq;
	    c3sq = max(d__1,0.);
/* Computing 2nd power */
	    d__1 = qdiv;
	    cw25 = e1 * (e2 + rc0010 * c1 * e5c * gmel * (d__1 * d__1)) / (
		    rc0010 * eden);
	    dlsq = elsq;
	    if (q3sq / dlsq < -gh[k - 1]) {
		q3sq = -dlsq * gh[k - 1];
	    }
/* Computing 2nd power */
	    r__1 = a2 * a2fac;
/* Computing 2nd power */
	    r__2 = gtr;
	    auh = rc0014 * a1 * (r__1 * r__1) * b2 * (r__2 * r__2);
/* Computing 2nd power */
	    r__1 = a1;
	    aum = rc0015 * (r__1 * r__1) * (a2 * a2fac) * b2 * c1 * gtr;
/* Computing 2nd power */
	    r__1 = a2 * a2fac;
/* Computing 2nd power */
	    r__2 = gtr;
	    adh = rc0016 * a1 * (r__1 * r__1) * (rc0017 * a1 + rc0018 * b2) * 
		    (r__2 * r__2);
/* Computing 2nd power */
	    r__1 = a1;
	    adm = rc0019 * (r__1 * r__1) * (a2 * a2fac) * (b2 - rc0018 * (a2 *
		     a2fac)) * gtr;
/* Computing 2nd power */
	    r__1 = a2 * a2fac;
/* Computing 2nd power */
	    r__2 = a2 * a2fac;
	    aeh = (rc0016 * a1 * (r__1 * r__1) * b1 + rc0016 * a1 * (r__2 * 
		    r__2) * (rc0017 * a1 + rc0018 * b2)) * gtr;
/* Computing 2nd power */
	    r__1 = a1;
	    aem = rc0018 * a1 * (a2 * a2fac) * b1 * (rc0018 * (a2 * a2fac) + 
		    rc0018 * b2 * c1 + (rc0019 * a1 * c1 - b2)) + rc0019 * (
		    r__1 * r__1) * (a2 * a2fac) * (b2 - rc0018 * (a2 * a2fac))
		    ;
	    req = -aeh / aem;
	    rsl = (auh + aum * req) / (rc0018 * adh + rc0018 * adm * req);
	    rsl = rc0020;
	    rsl2 = one - two * rsl;
/* Computing 2nd power */
	    d__1 = qdiv;
	    e2 = q3sq - e2c * ghel * a2fac * (d__1 * d__1);
/* Computing 2nd power */
	    r__1 = a2fac;
/* Computing 2nd power */
	    d__1 = qdiv;
	    e3 = q3sq + e3c * ghel * (r__1 * r__1) * (d__1 * d__1);
/* Computing 2nd power */
	    d__1 = qdiv;
	    e4 = q3sq - e4c * ghel * a2fac * (d__1 * d__1);
/* Computing 2nd power */
	    d__1 = qdiv;
	    eden = e2 * e4 + e3 * e5c * gmel * (d__1 * d__1);
/* Computing 2nd power */
	    r__1 = gtr;
/* Computing 2nd power */
	    d__1 = dlsq;
/* Computing 2nd power */
	    d__2 = qdiv;
/* Computing 2nd power */
	    r__2 = a2fac;
/* Computing 2nd power */
	    d__3 = qdiv;
	    wden = cc3 * (r__1 * r__1) * (d__1 * d__1) / elsq * (d__2 * d__2) 
		    * (e2 * e4c * a2fac - e3c * e5c * gmel * (r__2 * r__2) * (
		    d__3 * d__3));
	    if (wden != zero) {
		clow = q3sq * (rc0020 - cw25) * eden / wden;
		cupp = q3sq * (rc0004 - cw25) * eden / wden;
		if (wden > zero) {
/* Computing MIN */
/* Computing MAX */
		    d__3 = c3sq, d__4 = c2sq + clow;
		    d__1 = max(d__3,d__4), d__2 = c2sq + cupp;
		    c3sq = min(d__1,d__2);
		} else {
/* Computing MAX */
/* Computing MIN */
		    d__3 = c3sq, d__4 = c2sq + clow;
		    d__1 = min(d__3,d__4), d__2 = c2sq + cupp;
		    c3sq = max(d__1,d__2);
		}
	    }
/* Computing 2nd power */
	    d__1 = qdiv;
	    e1 = e2 + e5c * gmel * (d__1 * d__1);
	    eden = max(eden,1e-20);
	    e6c = rc0010 * (a2 * a2fac) * cc3 * gtr * dlsq / elsq;
	    if (t2sq >= zero) {
/* Computing MAX */
		d__1 = qdiv * e6c * (t3sq - t2sq);
		enum__ = max(d__1,0.);
	    } else {
/* Computing MIN */
		d__1 = qdiv * e6c * (t3sq - t2sq);
		enum__ = min(d__1,0.);
	    }
	    gamt = -e1 * enum__ / eden;
	    if (r2sq >= zero) {
/* Computing MAX */
		d__1 = qdiv * e6c * (r3sq - r2sq);
		enum__ = max(d__1,0.);
	    } else {
/* Computing MIN */
		d__1 = qdiv * e6c * (r3sq - r2sq);
		enum__ = min(d__1,0.);
	    }
	    gamq = -e1 * enum__ / eden;
/* Computing MAX */
	    d__1 = qdiv * e6c * (c3sq - c2sq);
	    enum__ = max(d__1,0.);
/* Computing 2nd power */
	    d__1 = qdiv;
/* Computing 2nd power */
	    r__1 = a2fac;
	    smd = dlsq * enum__ * gtr / eden * (d__1 * d__1) * (e3c * (r__1 * 
		    r__1) + e4c * a2fac) * a1 / (a2 * a2fac);
	    gamv = e1 * enum__ * gtr / eden;
	    sm[k] += smd;
	    qdiv = one;
	    if (FALSE_) {
		if (sh[k] < -rc0021 || sm[k] < -rc0021 || qke[k] < -rc0022 || 
			abs(smd) > rc0023) {
		}
	    }
	} else {
	    gamt = zero;
	    gamq = zero;
	    gamv = zero;
	}
	cldavg = half * (cldfra_bl1__[k - 1] + cldfra_bl1__[k]);
/* Computing MAX */
	r__2 = edmf_a1__[k] * edmf_w1__[k], r__3 = (r__1 = edmf_a_dd1__[k] * 
		edmf_w_dd1__[k], abs(r__1));
	mfmax = max(r__2,r__3);
/* Computing MAX */
/* Computing MIN */
	r__3 = rc0025 * mfmax;
	r__1 = sm[k], r__2 = rc0024 * min(r__3,one);
	sm[k] = max(r__1,r__2);
/* Computing MAX */
/* Computing MIN */
	r__3 = rc0025 * mfmax;
	r__1 = sh[k], r__2 = rc0024 * min(r__3,one);
	sh[k] = max(r__1,r__2);
/* Computing MAX */
	r__1 = sm[k], r__2 = rc0024 * min(cldavg,one);
	sm[k] = max(r__1,r__2);
/* Computing MAX */
	r__1 = sh[k], r__2 = rc0024 * min(cldavg,one);
	sh[k] = max(r__1,r__2);
	elq = el[k] * qkw[k - 1];
	elh = elq * qdiv;
	pdk[k] = elq * (sm[k] * gm[k - 1] + sh[k] * gh[k - 1] + gamv) + half *
		 tkeprod_dn__[k] + half * tkeprod_up__[k];
	pdt[k] = elh * (sh[k] * dtl[k - 1] + gamt) * dtl[k - 1];
	pdq[k] = elh * (sh[k] * dqw[k - 1] + gamq) * dqw[k - 1];
	pdc[k] = elh * (sh[k] * dtl[k - 1] + gamt) * dqw[k - 1] * half + elh *
		 (sh[k] * dqw[k - 1] + gamq) * dtl[k - 1] * half;
	tcd[k] = elq * gamt;
	qcd[k] = elq * gamq;
	dfm[k] = elq * sm[k] / dzk;
	dfh[k] = elq * sh[k] / dzk;
	dfq[k] = dfm[k];
	if (*tke_budget__ == 1) {
	    qshear1[k] = elq * sm[k] * gm[k - 1];
	    qbuoy1[k] = elq * (sh[k] * gh[k - 1] + gamv) + half * 
		    tkeprod_dn__[k] + half * tkeprod_up__[k];
	}
/* L9004: */
    }
/* L9005: */
    dfm[*kts] = zero;
    dfh[*kts] = zero;
    dfq[*kts] = zero;
    pdk[*kts] = zero;
    pdt[*kts] = zero;
    pdq[*kts] = zero;
    pdc[*kts] = zero;
    tcd[*kts] = zero;
    qcd[*kts] = zero;
    tcd[*kte] = zero;
    qcd[*kte] = zero;
    i__1 = *kte - 1;
    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	dzk = dz[k];
	tcd[k] = (tcd[k + 1] - tcd[k]) / dzk;
	qcd[k] = (qcd[k + 1] - qcd[k]) / dzk;
/* L9007: */
    }
/* L9008: */
    if (*spp_pbl__ == 1) {
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
/* Computing MAX */
	    r__2 = zw[k] - rc0027;
	    r__1 = (real)exp((real)(-max(r__2,zero) / rc0028));
	    dfm[k] += dfm[k] * pattern_spp_pbl1__[k] * rc0026 * max(r__1,
		    rc0029);
/* Computing MAX */
/* Computing MAX */
	    r__2 = zw[k] - rc0027;
	    r__1 = (real)exp((real)(-max(r__2,zero) / rc0028));
	    dfh[k] += dfh[k] * pattern_spp_pbl1__[k] * rc0026 * max(r__1,
		    rc0029);
/* L9010: */
	}
/* L9011: */
    }
    return 0;
} /* mym_turbulence__ */

__device__ M471_CALL /* Subroutine */ int mym_predict__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> closure, 
	M471Ptr<real> delt, M471Ptr<real> dz, M471Ptr<real> ust, M471Ptr<real> flt, M471Ptr<real> flq, M471Ptr<real> pmz, 
	M471Ptr<real> phh, M471Ptr<real> el, M471Ptr<real> dfq, M471Ptr<real> rho, M471Ptr<real> pdk, M471Ptr<real> pdt, M471Ptr<real> pdq, M471Ptr<real> pdc, M471Ptr<real> qke, M471Ptr<real> tsq, M471Ptr<real> qsq, M471Ptr<real> cov, M471Ptr<real> s_aw1__, M471Ptr<real> s_awqke1__, M471Ptr<integer> bl_mynn_edmf_tke__, M471Ptr<real> qwt1, 
	M471Ptr<real> qdiss1, M471Ptr<integer> tke_budget__) {
    /* Initialized data */

    real rc0004 = 2.f;
    real rc0005 = 0.f;
    real rc0006 = 150.f;
    real rc0007 = 2.5f;
    real rc0008 = 1e-17f;
    real rc0009 = .25f;
    real rc0010 = 3.f;
    real karman = .4000000059604644775390625f;
    real zero = 0.f;
    real one = 1.f;
    real b1 = 24.f;
    real b2 = 15.f;
    real sqfac = 3.f;
    real qkemin = 9.999999747378751635551452636719e-6f;
    real rc0000 = .5f;
    real rc0001 = 1.f;
    real rc0002 = 1e-4f;
    real rc0003 = 2.f;

    /* System generated locals */
    integer dz_offset, dfq_offset, el_offset, rho_offset, pdk_offset, 
	    pdt_offset, pdq_offset, pdc_offset, qke_offset, tsq_offset, 
	    qsq_offset, cov_offset, s_awqke1_offset, s_aw1_offset, 
	    qwt1_offset, qdiss1_offset, i__1;
    real r__1, r__2;

    /* Builtin functions */
    

    /* Local variables */
    M471Ptr<real> a=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+18800),m471_n);
M471Ptr<real> b=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+19000),m471_n);
M471Ptr<real> c__=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+19200),m471_n);
M471Ptr<real> d__=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+19400),m471_n);
    integer k;
    real  b1l, b2l, phm, vkz, pdc1, pdk1, pdq1, pdt1, onoff;
M471Ptr<real> x=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+19600),m471_n);
M471Ptr<real> bp=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+19800),m471_n);
M471Ptr<real> rp=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+20000),m471_n);
M471Ptr<real> qkw=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+20200),m471_n);
M471Ptr<real> dtz=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+20400),m471_n);
M471Ptr<real> df3q=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+20600),m471_n);
M471Ptr<real> kmdz=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+20800),m471_n);
M471Ptr<real> kqdz=m471_array<real>(m471_work,(232*(MYNN471_NZ-49)+21000),m471_n);
M471Ptr<real> rhoz=m471_array<real>(m471_work,(240*(MYNN471_NZ-49)+21200),m471_n);
M471Ptr<real> dzinv=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+21400),m471_n);
    integer ixarr1;
    M471Ptr<real> tke_up__=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+21600),m471_n);
M471Ptr<real> rhoinv=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+21800),m471_n);
    

    /* Parameter adjustments */
    qdiss1_offset = *kts;
    qdiss1 -= qdiss1_offset;
    qwt1_offset = *kts;
    qwt1 -= qwt1_offset;
    s_awqke1_offset = *kts;
    s_awqke1__ -= s_awqke1_offset;
    s_aw1_offset = *kts;
    s_aw1__ -= s_aw1_offset;
    cov_offset = *kts;
    cov -= cov_offset;
    qsq_offset = *kts;
    qsq -= qsq_offset;
    tsq_offset = *kts;
    tsq -= tsq_offset;
    qke_offset = *kts;
    qke -= qke_offset;
    pdc_offset = *kts;
    pdc -= pdc_offset;
    pdq_offset = *kts;
    pdq -= pdq_offset;
    pdt_offset = *kts;
    pdt -= pdt_offset;
    pdk_offset = *kts;
    pdk -= pdk_offset;
    rho_offset = *kts;
    rho -= rho_offset;
    dfq_offset = *kts;
    dfq -= dfq_offset;
    el_offset = *kts;
    el -= el_offset;
    dz_offset = *kts;
    dz -= dz_offset;

    /* Function Body */
    if (*bl_mynn_edmf_tke__ == 0) {
	onoff = zero;
    } else {
	onoff = one;
    }
    vkz = karman * rc0000 * dz[*kts];
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	r__1 = qke[k];
	qkw[k - 1] = (real)sqrt((real)(max(r__1,zero)));
	df3q[k - 1] = sqfac * dfq[k];
	dtz[k - 1] = *delt / dz[k];
/* L9004: */
    }
/* L9005: */
    rhoz[*kts - 1] = rho[*kts];
    rhoinv[*kts - 1] = rc0001 / rho[*kts];
    kqdz[*kts - 1] = rhoz[*kts - 1] * df3q[*kts - 1];
    kmdz[*kts - 1] = rhoz[*kts - 1] * dfq[*kts];
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	rhoz[k - 1] = (rho[k] * dz[k - 1] + rho[k - 1] * dz[k]) / (dz[k - 1] 
		+ dz[k]);
/* Computing MAX */
	r__1 = rhoz[k - 1];
	rhoz[k - 1] = max(r__1,rc0002);
/* Computing MAX */
	r__1 = rho[k];
	rhoinv[k - 1] = rc0001 / max(r__1,rc0002);
	kqdz[k - 1] = rhoz[k - 1] * df3q[k - 1];
	kmdz[k - 1] = rhoz[k - 1] * dfq[k];
/* L9007: */
    }
/* L9008: */
    rhoz[*kte] = rhoz[*kte - 1];
    kqdz[*kte] = rhoz[*kte] * df3q[*kte - 1];
    kmdz[*kte] = rhoz[*kte] * dfq[*kte];
    i__1 = *kte - 1;
    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
/* Computing MAX */
	r__1 = kqdz[k - 1], r__2 = rc0000 * s_aw1__[k];
	kqdz[k - 1] = max(r__1,r__2);
/* Computing MAX */
	r__1 = kqdz[k - 1], r__2 = -rc0000 * (s_aw1__[k] - s_aw1__[k + 1]);
	kqdz[k - 1] = max(r__1,r__2);
/* Computing MAX */
	r__1 = kmdz[k - 1], r__2 = rc0000 * s_aw1__[k];
	kmdz[k - 1] = max(r__1,r__2);
/* Computing MAX */
	r__1 = kmdz[k - 1], r__2 = -rc0000 * (s_aw1__[k] - s_aw1__[k + 1]);
	kmdz[k - 1] = max(r__1,r__2);
/* L9010: */
    }
/* L9011: */
/* Computing 3rd power */
    r__1 = *ust;
    pdk1 = rc0003 * (r__1 * (r__1 * r__1)) * *pmz / vkz;
    phm = rc0003 / *ust * *phh / vkz;
/* Computing 2nd power */
    r__1 = *flt;
    pdt1 = phm * (r__1 * r__1);
/* Computing 2nd power */
    r__1 = *flq;
    pdq1 = phm * (r__1 * r__1);
    pdc1 = phm * *flt * *flq;
    pdk[*kts] = pdk1 - pdk[*kts + 1];
    pdt[*kts] = pdt[*kts + 1];
    pdq[*kts] = pdq[*kts + 1];
    pdc[*kts] = pdc[*kts + 1];
    i__1 = *kte - 1;
    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	b1l = b1 * rc0000 * (el[k + 1] + el[k]);
	bp[k - 1] = rc0004 * qkw[k - 1] / b1l;
	rp[k - 1] = pdk[k + 1] + pdk[k];
/* L9013: */
    }
/* L9014: */
    i__1 = *kte - 1;
    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	a[k - 1] = -dtz[k - 1] * kqdz[k - 1] * rhoinv[k - 1] + rc0000 * dtz[k 
		- 1] * rhoinv[k - 1] * s_aw1__[k] * onoff;
	b[k - 1] = rc0001 + dtz[k - 1] * (kqdz[k - 1] + kqdz[k]) * rhoinv[k - 
		1] + rc0000 * dtz[k - 1] * rhoinv[k - 1] * (s_aw1__[k] - 
		s_aw1__[k + 1]) * onoff + bp[k - 1] * *delt;
	c__[k - 1] = -dtz[k - 1] * kqdz[k] * rhoinv[k - 1] - rc0000 * dtz[k - 
		1] * rhoinv[k - 1] * s_aw1__[k + 1] * onoff;
	d__[k - 1] = rp[k - 1] * *delt + qke[k] + dtz[k - 1] * rhoinv[k - 1] *
		 (s_awqke1__[k] - s_awqke1__[k + 1]) * onoff;
/* L9016: */
    }
/* L9017: */
    a[*kte - 1] = rc0005;
    b[*kte - 1] = rc0001;
    c__[*kte - 1] = rc0005;
    d__[*kte - 1] = qke[*kte];
    tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	r__1 = x[k - 1];
	qke[k] = max(r__1,qkemin);
/* Computing MIN */
	r__1 = qke[k];
	qke[k] = min(r__1,rc0006);
/* L9019: */
    }
/* L9020: */
    if (*tke_budget__ == 1) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    tke_up__[ixarr1 - 1] = rc0000 * qke[*kts + ixarr1 - 1];
	}
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    dzinv[ixarr1 - 1] = rc0001 / dz[*kts + ixarr1 - 1];
	}
	k = *kts;
	qwt1[k] = dzinv[k - 1] * (kqdz[k] * (tke_up__[k] - tke_up__[k - 1]) - 
		kqdz[k - 1] * tke_up__[k - 1] + rc0000 * rhoinv[k - 1] * (
		s_aw1__[k + 1] * tke_up__[k] + (s_aw1__[k + 1] - s_aw1__[k]) *
		 tke_up__[k - 1] + (s_awqke1__[k] - s_awqke1__[k + 1])) * 
		onoff);
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    qwt1[k] = dzinv[k - 1] * (kqdz[k] * (tke_up__[k] - tke_up__[k - 1]
		    ) - kqdz[k - 1] * (tke_up__[k - 1] - tke_up__[k - 2]) + 
		    rc0000 * rhoinv[k - 1] * (s_aw1__[k + 1] * tke_up__[k] + (
		    s_aw1__[k + 1] - s_aw1__[k]) * tke_up__[k - 1] - s_aw1__[
		    k] * tke_up__[k - 2] + (s_awqke1__[k] - s_awqke1__[k + 1])
		    ) * onoff);
/* L9022: */
	}
/* L9023: */
	k = *kte;
	qwt1[k] = dzinv[k - 1] * (-kqdz[k - 1] * (tke_up__[k - 1] - tke_up__[
		k - 2]) + rc0000 * rhoinv[k - 1] * (-s_aw1__[k] * tke_up__[k 
		- 1] - s_aw1__[k] * tke_up__[k - 2] + s_awqke1__[k]) * onoff);
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    qdiss1[*kts + ixarr1 - 1] = bp[ixarr1 - 1] * tke_up__[ixarr1 - 1];
	}
    }
    if (*closure > rc0007) {
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    b2l = b2 * rc0000 * (el[k + 1] + el[k]);
	    bp[k - 1] = rc0004 * qkw[k - 1] / b2l;
	    rp[k - 1] = pdq[k + 1] + pdq[k];
/* L9025: */
	}
/* L9026: */
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * kmdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = rc0001 + dtz[k - 1] * (kmdz[k - 1] + kmdz[k]) * rhoinv[
		    k - 1] + bp[k - 1] * *delt;
	    c__[k - 1] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1];
	    d__[k - 1] = rp[k - 1] * *delt + qsq[k];
/* L9028: */
	}
/* L9029: */
	a[*kte - 1] = -rc0001;
	b[*kte - 1] = rc0001;
	c__[*kte - 1] = rc0005;
	d__[*kte - 1] = rc0005;
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = x[k - 1];
	    qsq[k] = max(r__1,rc0008);
/* L9031: */
	}
/* L9032: */
    } else {
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    if (qkw[k - 1] <= zero) {
		b2l = zero;
	    } else {
		b2l = b2 * rc0009 * (el[k + 1] + el[k]) / qkw[k - 1];
	    }
	    qsq[k] = b2l * (pdq[k + 1] + pdq[k]);
/* L9034: */
	}
/* L9035: */
	qsq[*kte] = qsq[*kte - 1];
    }
    if (*closure >= rc0010) {
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    b2l = b2 * rc0000 * (el[k + 1] + el[k]);
	    bp[k - 1] = rc0004 * qkw[k - 1] / b2l;
	    rp[k - 1] = pdt[k + 1] + pdt[k];
/* L9037: */
	}
/* L9038: */
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * kmdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = rc0001 + dtz[k - 1] * (kmdz[k - 1] + kmdz[k]) * rhoinv[
		    k - 1] + bp[k - 1] * *delt;
	    c__[k - 1] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1];
	    d__[k - 1] = rp[k - 1] * *delt + tsq[k];
/* L9040: */
	}
/* L9041: */
	a[*kte - 1] = -rc0001;
	b[*kte - 1] = rc0001;
	c__[*kte - 1] = rc0005;
	d__[*kte - 1] = rc0005;
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    tsq[k] = x[k - 1];
/* L9043: */
	}
/* L9044: */
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    b2l = b2 * rc0000 * (el[k + 1] + el[k]);
	    bp[k - 1] = rc0004 * qkw[k - 1] / b2l;
	    rp[k - 1] = pdc[k + 1] + pdc[k];
/* L9046: */
	}
/* L9047: */
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * kmdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = rc0001 + dtz[k - 1] * (kmdz[k - 1] + kmdz[k]) * rhoinv[
		    k - 1] + bp[k - 1] * *delt;
	    c__[k - 1] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1];
	    d__[k - 1] = rp[k - 1] * *delt + cov[k];
/* L9049: */
	}
/* L9050: */
	a[*kte - 1] = -rc0001;
	b[*kte - 1] = rc0001;
	c__[*kte - 1] = rc0005;
	d__[*kte - 1] = rc0005;
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    cov[k] = x[k - 1];
/* L9052: */
	}
/* L9053: */
    } else {
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    if (qkw[k - 1] <= zero) {
		b2l = zero;
	    } else {
		b2l = b2 * rc0009 * (el[k + 1] + el[k]) / qkw[k - 1];
	    }
	    tsq[k] = b2l * (pdt[k + 1] + pdt[k]);
	    cov[k] = b2l * (pdc[k + 1] + pdc[k]);
/* L9055: */
	}
/* L9056: */
	tsq[*kte] = tsq[*kte - 1];
	cov[*kte] = cov[*kte - 1];
    }
    return 0;
} /* mym_predict__ */

__device__ M471_CALL /* Subroutine */ int mym_condensation__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dx, 
	M471Ptr<real> dz, M471Ptr<real> zw, M471Ptr<real> xland, M471Ptr<real> thl, M471Ptr<real> qw, M471Ptr<real> qv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<real> qs, M471Ptr<real> p, M471Ptr<real> exner, M471Ptr<real> tsq, M471Ptr<real> qsq, 
	M471Ptr<real> cov, M471Ptr<real> sh, M471Ptr<real> el, M471Ptr<integer> bl_mynn_cloudpdf__, M471Ptr<real> qc_bl1__, M471Ptr<real> qi_bl1__, M471Ptr<real> cldfra_bl1__, M471Ptr<real> pblh1, M471Ptr<real> hfx1,
	 M471Ptr<real> vt, M471Ptr<real> vq, M471Ptr<real> th, M471Ptr<real> sgm, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__) {
    /* Initialized data */

    real cp = 1004.5f;
    real r_d__ = 287.f;
    real r_v__ = 461.600006103515625f;
    real xlv = 2.5e6f;
    real p608 = .6083624362945556640625f;
    real ep_2__ = .621750414371490478515625f;
    real cpv = 1846.4000244140625f;
    real zero = 0.f;
    real half = .5f;
    real one = 1.f;
    real onethird = .3333333432674407958984375f;
    real tice = 240.f;
    real ep_3__ = .378249585628509521484375f;
    real tv0 = 182.50872802734375f;
    real xlvcp = 2488.80029296875f;
    real b2 = 15.f;
    real tliq = 269.f;
    real rr2 = .7071068286895751953125f;
    real rrp = .398942291736602783203125f;
    real rc0000 = .007f;
    real rc0001 = .02f;
    real rc0002 = .025f;
    real rc0003 = .01f;
    real rc0004 = .03f;
    real rc0005 = .83f;
    real rc0006 = 1.1f;
    real rc0007 = 44307.692f;
    real rc0008 = 101325.f;
    real rc0009 = .19f;
    real rc0010 = 10.f;
    real rc0011 = 1500.f;
    real rc0012 = 0.f;
    real rc0013 = 1e-4f;
    real rc0014 = 2.f;
    real rc0015 = .5f;
    real rc0016 = 240.f;
    real rc0017 = 29.f;
    real rc0018 = 1.f;
    real rc0019 = .1f;
    real rc0020 = .03f;
    real rc0021 = 4.f;
    real rc0022 = 1e-10f;
    real rc0023 = 1e-10f;
    real rc0024 = .001f;
    real rc0025 = 1e-13f;
    real rc0026 = 100.f;
    real rc0027 = 500.f;
    real rc0028 = .2f;
    real rc0029 = 300.f;
    real rc0030 = 150.f;
    real rc0031 = 50.f;
    real rc0032 = 200.f;
    real rc0033 = .02f;
    real rc0034 = 1e-9f;
    real rc0035 = .045f;
    real rc0036 = 9.f;
    real rc0037 = 3.f;
    real rc0038 = 1e-6f;
    real rc0039 = .08f;
    real rc0040 = 6.f;
    real rc0041 = .35f;
    real rc0042 = 4.1f;
    real rc0043 = .37f;
    real rc0044 = 2.1f;
    real rc0045 = .4f;
    real rc0046 = .56f;
    real rc0047 = .976f;
    real rc0048 = .55f;
    real rc0049 = .955f;
    real rc0050 = .055f;
    real rc0051 = 1.5f;
    real rc0052 = 2.5f;
    real rc0053 = 1.7f;
    real rc0054 = 3.f;
    real rc0055 = 3.8f;
    real rc0056 = 23.9f;
    real rc0057 = 1.6f;
    real rc0058 = 60.f;
    real rc0059 = .6f;
    real rc0060 = 25.f;
    real rc0061 = .61f;
    real rc0062 = 1.61f;

    /* System generated locals */
    integer dz_offset, zw_offset, p_offset, exner_offset, thl_offset, 
	    qw_offset, qv_offset, qc_offset, qi_offset, qs_offset, tsq_offset,
	     qsq_offset, cov_offset, th_offset, vt_offset, vq_offset, 
	    sgm_offset, qc_bl1_offset, qi_bl1_offset, cldfra_bl1_offset, 
	    sh_offset, el_offset, pattern_spp_pbl1_offset, i__1, i__2;
    real r__1, r__2, r__3, r__4, r__5, r__6;
    doublereal d__1, d__2, d__3;

    /* Builtin functions */
    

    /* Local variables */
    real clim_sfc__, liq_frac__, clim_pbl__;
    
    real qlim_sfc__, qlim_pbl__, clim_trp__, ql_water__, qlim_trp__;
M471Ptr<real> a=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+22000),m471_n);
M471Ptr<real> b=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+22200),m471_n);
    integer k;
    real t, cldfra_rh__, cldfra_rh0__, cldfra_rh1__, bb, pt, qt, xl, wt;
M471Ptr<real> q1=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+22400),m471_n);
M471Ptr<real> ql=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+22600),m471_n);
M471Ptr<real> rh=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+22800),m471_n);
    
    real cldfra_qsq__, eq1, q1k, ht1, ht2, q2p, wt2, cldfra_qsq0__, 
	    cldfra_qsq1__, rac, fng;
M471Ptr<real> bet=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+23000),m471_n);
M471Ptr<real> alp=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+23200),m471_n);
    
    real cpm, dth, qll, qsl, rsl, qmq, dqw, zsl, dzk, qww;
    doublereal c3sq, r3sq, t3sq;
    real beta, clim, sgmc, esat, zagl, qlim, dqsl, sgmq, pblh2, q1_rh__, 
	    alpha, cfmax, rhmax, theta1, theta2, ql_ice__, rhcrit, rh_hack__, 
	    qsat_tk__;
    integer k_tropo__;
    real qw_pert__;

    /* Parameter adjustments */
    pattern_spp_pbl1_offset = *kts;
    pattern_spp_pbl1__ -= pattern_spp_pbl1_offset;
    sgm_offset = *kts;
    sgm -= sgm_offset;
    th_offset = *kts;
    th -= th_offset;
    vq_offset = *kts;
    vq -= vq_offset;
    vt_offset = *kts;
    vt -= vt_offset;
    cldfra_bl1_offset = *kts;
    cldfra_bl1__ -= cldfra_bl1_offset;
    qi_bl1_offset = *kts;
    qi_bl1__ -= qi_bl1_offset;
    qc_bl1_offset = *kts;
    qc_bl1__ -= qc_bl1_offset;
    el_offset = *kts;
    el -= el_offset;
    sh_offset = *kts;
    sh -= sh_offset;
    cov_offset = *kts;
    cov -= cov_offset;
    qsq_offset = *kts;
    qsq -= qsq_offset;
    tsq_offset = *kts;
    tsq -= tsq_offset;
    exner_offset = *kts;
    exner -= exner_offset;
    p_offset = *kts;
    p -= p_offset;
    qs_offset = *kts;
    qs -= qs_offset;
    qi_offset = *kts;
    qi -= qi_offset;
    qc_offset = *kts;
    qc -= qc_offset;
    qv_offset = *kts;
    qv -= qv_offset;
    qw_offset = *kts;
    qw -= qw_offset;
    thl_offset = *kts;
    thl -= thl_offset;
    zw_offset = *kts;
    zw -= zw_offset;
    dz_offset = *kts;
    dz -= dz_offset;

    /* Function Body */
    qlim_sfc__ = rc0000;
    qlim_pbl__ = rc0001;
    qlim_trp__ = rc0002;
    clim_sfc__ = rc0003;
    clim_pbl__ = rc0002;
    clim_trp__ = rc0004;
    rhcrit = rc0005;
    rhmax = rc0006;
    i__1 = *kts;
    
#pragma unroll 1
for (k = *kte - 3; k >= i__1; --k) {
	theta1 = th[k];
	theta2 = th[k + 2];
	d__1 =  (p[k] / rc0008);
	d__2 =  rc0009;
	ht1 = rc0007 * (one - (real)pow_dd(&d__1, &d__2));
	d__1 =  (p[k + 2] / rc0008);
	d__2 =  rc0009;
	ht2 = rc0007 * (one - (real)pow_dd(&d__1, &d__2));
	if ((theta2 - theta1) / (ht2 - ht1) < rc0010 / rc0011 && ht1 < 1.9e4f 
		&& ht1 > 4e3f) {
	    goto L86;
	}
/* L9004: */
    }
/* L9005: */
L86:
/* Computing MAX */
    i__1 = *kts + 2, i__2 = k + 2;
    k_tropo__ = max(i__1,i__2);
    zagl = rc0012;
    if (*bl_mynn_cloudpdf__ == 0) {
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    t = th[k] * exner[k];
	    esat = (real)esat_blend__(m471_work,m471_n, &t);
/* Computing MAX */
	    r__1 = rc0013, r__2 = p[k] - ep_3__ * esat;
	    qsl = ep_2__ * esat / max(r__1,r__2);
/* Computing 2nd power */
	    r__1 = t;
	    dqsl = qsl * ep_2__ * xlv / (r_d__ * (r__1 * r__1));
	    alp[k - 1] = one / (one + dqsl * xlvcp);
	    bet[k - 1] = dqsl * exner[k];
/* Computing MAX */
	    r__1 = tsq[k];
	    t3sq = max(r__1,zero);
/* Computing MAX */
	    r__1 = qsq[k];
	    r3sq = max(r__1,zero);
	    c3sq = cov[k];
/* Computing MIN */
	    d__2 = abs(c3sq), d__3 = sqrt(t3sq * r3sq);
	    d__1 = min(d__2,d__3);
	    c3sq = d_sign(&d__1, &c3sq);
/* Computing 2nd power */
	    r__1 = bet[k - 1];
	    r3sq = r3sq + r__1 * r__1 * t3sq - rc0014 * bet[k - 1] * c3sq;
	    qmq = qw[k] - qsl;
	    sgm[k] = sqrt((max(r3sq,1e-10)));
	    q1[k - 1] = qmq / sgm[k];
	    r__1 = q1[k - 1] * rr2;
	    cldfra_bl1__[k] = rc0015 * (one + (real)erf_(&r__1));
	    q1k = q1[k - 1];
	    eq1 = rrp * (real)exp((real)(-rc0015 * q1k * q1k));
/* Computing MAX */
	    r__1 = cldfra_bl1__[k] * q1k + eq1;
	    qll = max(r__1,zero);
	    ql[k - 1] = alp[k - 1] * sgm[k] * qll;
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = (t - rc0016) / rc0017;
	    r__1 = one, r__2 = max(r__3,r__4);
	    liq_frac__ = min(r__1,r__2);
	    qc_bl1__[k] = liq_frac__ * ql[k - 1];
	    qi_bl1__[k] = (one - liq_frac__) * ql[k - 1];
	    q2p = xlvcp / exner[k];
	    pt = thl[k] + q2p * ql[k - 1];
	    qt = one + p608 * qw[k] - (rc0018 + p608) * (qc_bl1__[k] + 
		    qi_bl1__[k]) * cldfra_bl1__[k];
	    rac = alp[k - 1] * (cldfra_bl1__[k] - qll * eq1) * (q2p * qt - (
		    rc0018 + p608) * pt);
	    vt[k] = qt - one - rac * bet[k - 1];
	    vq[k] = p608 * pt - tv0 + rac;
/* L9007: */
	}
/* L9008: */
    } else if (*bl_mynn_cloudpdf__ == 1 || *bl_mynn_cloudpdf__ == -1) {
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    t = th[k] * exner[k];
	    esat = (real)esat_blend__(m471_work,m471_n, &t);
/* Computing MAX */
	    r__1 = rc0013, r__2 = p[k] - ep_3__ * esat;
	    qsl = ep_2__ * esat / max(r__1,r__2);
/* Computing 2nd power */
	    r__1 = t;
	    dqsl = qsl * ep_2__ * xlv / (r_d__ * (r__1 * r__1));
	    alp[k - 1] = one / (one + dqsl * xlvcp);
	    bet[k - 1] = dqsl * exner[k];
	    if (k == *kts) {
		dzk = rc0015 * dz[k];
	    } else {
		dzk = dz[k];
	    }
/* Computing MAX */
	    i__2 = k - 1;
	    dth = rc0015 * (thl[k + 1] + thl[k]) - rc0015 * (thl[k] + thl[max(
		    i__2,*kts)]);
/* Computing MAX */
	    i__2 = k - 1;
	    dqw = rc0015 * (qw[k + 1] + qw[k]) - rc0015 * (qw[k] + qw[max(
		    i__2,*kts)]);
/* Computing MAX */
/* Computing 2nd power */
	    r__2 = alp[k - 1];
/* Computing MAX */
/* Computing 2nd power */
	    r__4 = el[k];
	    r__3 = r__4 * r__4;
/* Computing MAX */
	    r__5 = sh[k];
/* Computing 2nd power */
	    r__6 = dqw / dzk - bet[k - 1] * (dth / dzk);
	    r__1 = r__2 * r__2 * max(r__3,rc0019) * b2 * max(r__5,rc0020) / 
		    rc0021 * (r__6 * r__6);
	    sgm[k] = (real)sqrt((real)(max(r__1,rc0022)));
	    qmq = qw[k] - qsl;
	    q1[k - 1] = qmq / sgm[k];
	    r__1 = q1[k - 1] * rr2;
	    cldfra_bl1__[k] = rc0015 * (one + (real)erf_(&r__1));
	    q1k = q1[k - 1];
	    eq1 = rrp * (real)exp((real)(-rc0015 * q1k * q1k));
/* Computing MAX */
	    r__1 = cldfra_bl1__[k] * q1k + eq1;
	    qll = max(r__1,zero);
	    ql[k - 1] = alp[k - 1] * sgm[k] * qll;
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = (t - rc0016) / rc0017;
	    r__1 = one, r__2 = max(r__3,r__4);
	    liq_frac__ = min(r__1,r__2);
	    qc_bl1__[k] = liq_frac__ * ql[k - 1];
	    qi_bl1__[k] = (one - liq_frac__) * ql[k - 1];
	    q2p = xlvcp / exner[k];
	    pt = thl[k] + q2p * ql[k - 1];
	    qt = one + p608 * qw[k] - (rc0018 + p608) * (qc_bl1__[k] + 
		    qi_bl1__[k]) * cldfra_bl1__[k];
	    rac = alp[k - 1] * (cldfra_bl1__[k] - qll * eq1) * (q2p * qt - (
		    rc0018 + p608) * pt);
	    vt[k] = qt - one - rac * bet[k - 1];
	    vq[k] = p608 * pt - tv0 + rac;
/* L9010: */
	}
/* L9011: */
    } else if (*bl_mynn_cloudpdf__ == 2 || *bl_mynn_cloudpdf__ == -2) {
	pblh2 = max(rc0010,*pblh1);
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    zagl = zw[k] + half * dz[k];
	    t = th[k] * exner[k];
	    xl = (real)xl_blend__(m471_work,m471_n, &t);
	    qsat_tk__ = (real)qsat_blend__(m471_work,m471_n, &t, &p[k]);
/* Computing MAX */
/* Computing MIN */
	    r__2 = rhmax, r__3 = qw[k] / max(rc0023,qsat_tk__);
	    r__1 = min(r__2,r__3);
	    rh[k - 1] = max(r__1,rc0024);
/* Computing 2nd power */
	    r__1 = t;
	    dqsl = qsat_tk__ * ep_2__ * xlv / (r_d__ * (r__1 * r__1));
	    alp[k - 1] = one / (one + dqsl * xlvcp);
	    bet[k - 1] = dqsl * exner[k];
/* Computing 2nd power */
	    r__1 = t;
	    rsl = xl * qsat_tk__ / (r_v__ * (r__1 * r__1));
	    cpm = cp + qw[k] * cpv;
	    a[k - 1] = one / (one + xl * rsl / cpm);
	    b[k - 1] = a[k - 1] * rsl;
	    qw_pert__ = qw[k] + qw[k] * rc0015 * pattern_spp_pbl1__[k] * (
		    real) (*spp_pbl__);
	    qmq = qw_pert__ - qsat_tk__;
/* Computing MAX */
	    r__1 = qsq[k];
	    r3sq = max(r__1,zero);
/* Computing MAX */
	    d__1 = rc0025, d__2 = sqrt(r3sq);
	    sgm[k] = max(d__1,d__2);
/* Computing MIN */
	    r__1 = sgm[k], r__2 = qw[k] * onethird;
	    sgm[k] = min(r__1,r__2);
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = dz[k] - rc0026;
	    r__1 = one, r__2 = max(r__3,r__4) / rc0027;
	    wt = min(r__1,r__2);
	    sgm[k] += sgm[k] * rc0028 * wt;
	    sgmq = sgm[k];
	    sgmc = sgm[k];
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = zagl - (pblh2 + rc0010);
	    r__1 = one, r__2 = max(r__3,r__4) / rc0029;
	    wt = min(r__1,r__2);
	    clim = clim_pbl__ * (one - wt) + clim_trp__ * wt;
/* Computing MIN */
/* Computing MAX */
	    r__3 = rc0031, r__4 = rc0019 * pblh2;
	    r__1 = rc0030, r__2 = max(r__3,r__4);
	    zsl = min(r__1,r__2);
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = zagl - zsl;
	    r__1 = one, r__2 = max(r__3,r__4) / rc0032;
	    wt = min(r__1,r__2);
	    clim = clim_sfc__ * (one - wt) + clim * wt;
/* Computing MAX */
	    r__1 = sgmc, r__2 = qw[k] * clim;
	    sgmc = max(r__1,r__2);
	    sgmc = max(rc0025,sgmc);
	    if (qmq >= zero) {
/* Computing MAX */
		r__1 = rc0033 * qw[k];
		sgmc = max(r__1,sgmc);
	    }
	    q1[k - 1] = qmq / sgmc;
	    rh_hack__ = rh[k - 1];
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = zagl - pblh2;
	    r__1 = one, r__2 = max(r__3,r__4) / rc0029;
	    wt2 = min(r__1,r__2);
	    if (qi[k] + qs[k] > rc0034 && zagl > pblh2) {
/* Computing MIN */
		r__3 = qi[k] + qs[k];
		r__1 = rhmax, r__2 = rhcrit + wt2 * rc0035 * (rc0036 + (real)
			r_lg10(&r__3));
		rh_hack__ = min(r__1,r__2);
/* Computing MAX */
		r__1 = rh[k - 1];
		rh[k - 1] = max(r__1,rh_hack__);
		q1_rh__ = -rc0037 + rc0037 * (rh[k - 1] - rhcrit) / (one - 
			rhcrit);
/* Computing MAX */
		r__1 = q1_rh__, r__2 = q1[k - 1];
		q1[k - 1] = max(r__1,r__2);
	    }
	    if (qc[k] > rc0038 && zagl > pblh2) {
/* Computing MIN */
		r__1 = rhmax, r__2 = rhcrit + wt2 * rc0039 * (rc0040 + (real)
			r_lg10(&qc[k]));
		rh_hack__ = min(r__1,r__2);
/* Computing MAX */
		r__1 = rh[k - 1];
		rh[k - 1] = max(r__1,rh_hack__);
		q1_rh__ = -rc0037 + rc0037 * (rh[k - 1] - rhcrit) / (one - 
			rhcrit);
/* Computing MAX */
		r__1 = q1_rh__, r__2 = q1[k - 1];
		q1[k - 1] = max(r__1,r__2);
	    }
	    q1k = q1[k - 1];
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = (zagl - (*pblh1 - rc0026)) / rc0032;
	    r__1 = one, r__2 = max(r__3,r__4);
	    wt2 = min(r__1,r__2);
/* Computing MAX */
/* Computing MIN */
	    r__3 = one, r__4 = half + rc0041 * (real)atan((real)(rc0042 * q1k)
		    );
	    r__1 = zero, r__2 = min(r__3,r__4);
	    cldfra_qsq0__ = max(r__1,r__2);
/* Computing MAX */
/* Computing MIN */
	    r__3 = one, r__4 = half + rc0043 * (real)atan((real)(rc0044 * (
		    q1k + rc0045)));
	    r__1 = zero, r__2 = min(r__3,r__4);
	    cldfra_qsq1__ = max(r__1,r__2);
	    cldfra_qsq__ = cldfra_qsq0__ * (one - wt2) + cldfra_qsq1__ * wt2;
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = rc0046 * (real)tanh((real)((rh[k - 1] - 
		    rc0047) / rc0004)) + half;
	    r__1 = one, r__2 = max(r__3,r__4);
	    cldfra_rh0__ = min(r__1,r__2);
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = rc0048 * (real)tanh((real)((rh[k - 1] - 
		    rc0049) / rc0050)) + half;
	    r__1 = one, r__2 = max(r__3,r__4);
	    cldfra_rh1__ = min(r__1,r__2);
	    cldfra_rh__ = zero;
	    cldfra_bl1__[k] = max(cldfra_qsq__,cldfra_rh__);
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = zagl - (pblh2 + rc0010);
	    r__1 = one, r__2 = max(r__3,r__4) / rc0029;
	    wt = min(r__1,r__2);
	    qlim = qlim_pbl__ * (one - wt) + qlim_trp__ * wt;
/* Computing MIN */
/* Computing MAX */
	    r__3 = rc0031, r__4 = rc0019 * pblh2;
	    r__1 = rc0030, r__2 = max(r__3,r__4);
	    zsl = min(r__1,r__2);
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = zagl - zsl;
	    r__1 = one, r__2 = max(r__3,r__4) / rc0032;
	    wt = min(r__1,r__2);
	    qlim = qlim_sfc__ * (one - wt) + qlim * wt;
/* Computing MAX */
	    r__1 = sgmq, r__2 = qw[k] * qlim;
	    sgmq = max(r__1,r__2);
/* Computing MIN */
	    r__1 = sgmq, r__2 = rc0002 * qw[k];
	    ql_water__ = min(r__1,r__2) * cldfra_bl1__[k];
/* Computing MIN */
	    r__1 = sgmq, r__2 = rc0002 * qw[k];
	    ql_ice__ = min(r__1,r__2) * cldfra_bl1__[k];
	    if (cldfra_bl1__[k] < rc0024) {
		ql_ice__ = zero;
		ql_water__ = zero;
		cldfra_bl1__[k] = zero;
	    }
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = (t - tice) / (tliq - tice);
	    r__1 = one, r__2 = max(r__3,r__4);
	    liq_frac__ = min(r__1,r__2);
	    qc_bl1__[k] = liq_frac__ * ql_water__;
	    qi_bl1__[k] = (one - liq_frac__) * ql_ice__;
	    if (k >= k_tropo__) {
		cldfra_bl1__[k] = zero;
		qc_bl1__[k] = zero;
		qi_bl1__[k] = zero;
	    }
	    if (*xland - rc0051 >= zero) {
/* Computing MAX */
		r__1 = q1[k - 1], r__2 = -rc0052;
		q1k = max(r__1,r__2);
	    } else {
/* Computing MAX */
		r__1 = q1[k - 1], r__2 = -rc0014;
		q1k = max(r__1,r__2);
	    }
	    if (q1k >= one) {
		fng = one;
	    } else if (q1k >= -rc0053 && q1k < one) {
		fng = (real)exp((real)(-rc0045 * (q1k - one)));
	    } else if (q1k >= -rc0052 && q1k < -rc0053) {
		fng = rc0054 + (real)exp((real)(-rc0055 * (q1k + rc0053)));
	    } else {
/* Computing MIN */
		r__1 = rc0056 + (real)exp((real)(-rc0057 * (q1k + rc0052)));
		fng = min(r__1,rc0058);
	    }
/* Computing MIN */
	    r__1 = cldfra_bl1__[k];
	    cfmax = min(r__1,rc0059);
/* Computing MIN */
/* Computing MAX */
	    r__2 = rc0060, r__3 = rc0019 * pblh2;
	    r__1 = max(r__2,r__3);
	    zsl = min(r__1,rc0026);
/* Computing MIN */
	    r__1 = zagl / zsl;
	    wt = min(r__1,one);
	    cfmax *= wt;
	    bb = b[k - 1] * t / th[k];
	    qww = one + rc0061 * qw[k];
	    alpha = rc0061 * th[k];
	    beta = th[k] / t * (xl / cp) - rc0062 * th[k];
	    vt[k] = qww - cfmax * beta * bb * fng - one;
	    vq[k] = alpha + cfmax * beta * a[k - 1] * fng - tv0;
/* L9013: */
	}
/* L9014: */
    }
    if (*bl_mynn_cloudpdf__ < 0) {
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    cldfra_bl1__[k] = zero;
	    qc_bl1__[k] = zero;
	    qi_bl1__[k] = zero;
/* L9016: */
	}
/* L9017: */
    }
    ql[*kte - 1] = ql[*kte - 2];
    vt[*kte] = vt[*kte - 1];
    vq[*kte] = vq[*kte - 1];
    qc_bl1__[*kte] = zero;
    qi_bl1__[*kte] = zero;
    cldfra_bl1__[*kte] = zero;
    return 0;
} /* mym_condensation__ */

__device__ M471_CALL /* Subroutine */ int mynn_tendencies__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<integer> i__, M471Ptr<real> delt, M471Ptr<real> dz, M471Ptr<real> zw, M471Ptr<real> xland, M471Ptr<real> rho, M471Ptr<real> u, 
	M471Ptr<real> v, M471Ptr<real> th, M471Ptr<real> tk, M471Ptr<real> qv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<real> qs, 
	M471Ptr<real> qnc, M471Ptr<real> qni, M471Ptr<real> psfc, M471Ptr<real> p, M471Ptr<real> exner, M471Ptr<real> thl, 
	M471Ptr<real> sqv, M471Ptr<real> sqc, M471Ptr<real> sqi, M471Ptr<real> sqs, M471Ptr<real> sqw, M471Ptr<real> qnwfa, 
	M471Ptr<real> qnifa, M471Ptr<real> qnbca, M471Ptr<real> ozone, M471Ptr<real> ust, M471Ptr<real> flt, M471Ptr<real> flq, M471Ptr<real> flqv, M471Ptr<real> flqc, M471Ptr<real> wspd, M471Ptr<real> uoce, M471Ptr<real> voce, M471Ptr<real> tsq, M471Ptr<real> qsq, M471Ptr<real> cov, M471Ptr<real> tcd, M471Ptr<real> qcd, M471Ptr<real> dfm, M471Ptr<real> dfh, M471Ptr<real> dfq, M471Ptr<real> du, M471Ptr<real> dv, M471Ptr<real> dth, M471Ptr<real> dqv, M471Ptr<real> dqc, 
	M471Ptr<real> dqi, M471Ptr<real> dqs, M471Ptr<real> dqnc, M471Ptr<real> dqni, M471Ptr<real> dqnwfa, M471Ptr<real> dqnifa, M471Ptr<real> dqnbca, M471Ptr<real> dozone, M471Ptr<real> diss_heat__, M471Ptr<real> s_aw1__, 
	M471Ptr<real> s_awthl1__, M471Ptr<real> s_awqt1__, M471Ptr<real> s_awqv1__, M471Ptr<real> s_awqc1__, 
	M471Ptr<real> s_awu1__, M471Ptr<real> s_awv1__, M471Ptr<real> s_awqnc1__, M471Ptr<real> s_awqni1__, 
	M471Ptr<real> s_awqnwfa1__, M471Ptr<real> s_awqnifa1__, M471Ptr<real> s_awqnbca1__, M471Ptr<real> sd_aw1__, M471Ptr<real> sd_awthl1__, M471Ptr<real> sd_awqt1__, M471Ptr<real> sd_awqv1__, M471Ptr<real> sd_awqc1__, M471Ptr<real> sd_awqi1__, M471Ptr<real> sd_awqnc1__, M471Ptr<real> sd_awqni1__, 
	M471Ptr<real> sd_awqnwfa1__, M471Ptr<real> sd_awqnifa1__, M471Ptr<real> sd_awu1__, M471Ptr<real> sd_awv1__, M471Ptr<real> sub_thl__, M471Ptr<real> sub_sqv__, M471Ptr<real> sub_u__, M471Ptr<real> sub_v__, M471Ptr<real> det_thl__, M471Ptr<real> det_sqv__, M471Ptr<real> det_sqc__, M471Ptr<real> det_u__, M471Ptr<real> det_v__, M471Ptr<logical> flag_qc__, M471Ptr<logical> flag_qi__, 
	M471Ptr<logical> flag_qnc__, M471Ptr<logical> flag_qni__, M471Ptr<logical> flag_qs__, M471Ptr<logical> flag_qnwfa__, M471Ptr<logical> flag_qnifa__, M471Ptr<logical> flag_qnbca__, M471Ptr<logical> flag_ozone__, M471Ptr<real> cldfra_bl1__, M471Ptr<integer> bl_mynn_cloudmix__, 
	M471Ptr<integer> bl_mynn_mixqt__, M471Ptr<integer> bl_mynn_edmf__, M471Ptr<integer> bl_mynn_edmf_mom__, M471Ptr<integer> bl_mynn_mixscalars__) {
    /* Initialized data */

    real r_d__ = 287.f;
    real p608 = .6083624362945556640625f;
    real zero = 0.f;
    real one = 1.f;
    real grav = 9.81000041961669921875f;
    real xlscp = 2837.232421875f;
    real xlvcp = 2488.80029296875f;
    real rc0000 = 1.f;
    real rc0001 = 100.f;
    real rc0002 = 1e-6f;
    real rc0003 = 8e8f;
    real rc0004 = 5e6f;
    real rc0005 = 2.7e8f;
    real rc0006 = 1e4f;
    real rc0007 = 2e3f;
    real rc0008 = 1e4f;
    real rc0009 = .5f;
    real rc0010 = 1.f;
    real rc0011 = 1e-4f;
    real rc0012 = .5f;
    real rc0013 = .99f;
    real rc0014 = .01f;
    real rc0015 = 0.f;
    real rc0016 = .9f;
    real rc0017 = 1e-8f;
    real rc0018 = 1e-12f;
    real rc0019 = 200.f;
    real rc0020 = 360.f;
    real rc0021 = 160.f;

    /* System generated locals */
    integer s_aw1_offset, s_awthl1_offset, s_awqt1_offset, s_awqnc1_offset, 
	    s_awqni1_offset, s_awqv1_offset, s_awqc1_offset, s_awu1_offset, 
	    s_awv1_offset, s_awqnwfa1_offset, s_awqnifa1_offset, 
	    s_awqnbca1_offset, sd_aw1_offset, sd_awthl1_offset, 
	    sd_awqt1_offset, sd_awqv1_offset, sd_awqc1_offset, 
	    sd_awqi1_offset, sd_awqnc1_offset, sd_awqni1_offset, 
	    sd_awqnwfa1_offset, sd_awqnifa1_offset, sd_awu1_offset, 
	    sd_awv1_offset, sub_thl_offset, sub_sqv_offset, sub_u_offset, 
	    sub_v_offset, det_thl_offset, det_sqv_offset, det_sqc_offset, 
	    det_u_offset, det_v_offset, u_offset, v_offset, th_offset, 
	    tk_offset, qv_offset, qc_offset, qi_offset, qs_offset, qni_offset,
	     qnc_offset, rho_offset, p_offset, exner_offset, dfq_offset, 
	    dz_offset, zw_offset, tsq_offset, qsq_offset, cov_offset, 
	    tcd_offset, qcd_offset, cldfra_bl1_offset, diss_heat_offset, 
	    thl_offset, sqw_offset, sqv_offset, sqc_offset, sqi_offset, 
	    sqs_offset, qnwfa_offset, qnifa_offset, qnbca_offset, 
	    ozone_offset, dfm_offset, dfh_offset, du_offset, dv_offset, 
	    dth_offset, dqv_offset, dqc_offset, dqi_offset, dqs_offset, 
	    dqni_offset, dqnc_offset, dqnwfa_offset, dqnifa_offset, 
	    dqnbca_offset, dozone_offset, i__1;
    real r__1, r__2, r__3;

    /* Builtin functions */
    

    /* Local variables */
    real aero_min__, aero_max__;
    integer kproblem;
    M471Ptr<real> a=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+23400),m471_n);
M471Ptr<real> b=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+23600),m471_n);
M471Ptr<real> c__=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+23800),m471_n);
M471Ptr<real> d__=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+24000),m471_n);
    integer k;
    real t;
M471Ptr<real> x=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+24200),m471_n);
    
    real condensate, th2, tk2, portion_qc__, portion_qi__, wsp, dzk, wsp2, qsat, onoff, 
	    dztop;
M471Ptr<real> dtz=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+24400),m471_n);
M471Ptr<real> qnc2=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+24600),m471_n);
M471Ptr<real> sqc2=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+24800),m471_n);
M471Ptr<real> qni2=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+25000),m471_n);
M471Ptr<real> sqi2=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+25200),m471_n);
M471Ptr<real> sqs2=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+25400),m471_n);
M471Ptr<real> sqv2=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+25600),m471_n);
M471Ptr<real> sqw2=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+25800),m471_n);
M471Ptr<real> delp=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+26000),m471_n);
M471Ptr<real> khdz=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+26200),m471_n);
M471Ptr<real> kmdz=m471_array<real>(m471_work,(256*(MYNN471_NZ-49)+26400),m471_n);
M471Ptr<real> rhoz=m471_array<real>(m471_work,(264*(MYNN471_NZ-49)+26600),m471_n);
M471Ptr<real> qnbca2=m471_array<real>(m471_work,(272*(MYNN471_NZ-49)+26800),m471_n);
M471Ptr<real> qnifa2=m471_array<real>(m471_work,(272*(MYNN471_NZ-49)+27000),m471_n);
M471Ptr<real> qnwfa2=m471_array<real>(m471_work,(272*(MYNN471_NZ-49)+27200),m471_n);
    integer ixarr1;
    real ifa_ht__, nc_min__, wfa_ht__, ni_min__, rhosfc, th_new__, nonloc, qvflux;
M471Ptr<real> rhoinv=m471_array<real>(m471_work,(272*(MYNN471_NZ-49)+27400),m471_n);
    
    real ifa_min__, ifa_max__, wfa_min__, wfa_max__;
    logical problem;
    real ustdiff, ustdrag;
    

    /* Parameter adjustments */
    cldfra_bl1_offset = *kts;
    cldfra_bl1__ -= cldfra_bl1_offset;
    det_v_offset = *kts;
    det_v__ -= det_v_offset;
    det_u_offset = *kts;
    det_u__ -= det_u_offset;
    det_sqc_offset = *kts;
    det_sqc__ -= det_sqc_offset;
    det_sqv_offset = *kts;
    det_sqv__ -= det_sqv_offset;
    det_thl_offset = *kts;
    det_thl__ -= det_thl_offset;
    sub_v_offset = *kts;
    sub_v__ -= sub_v_offset;
    sub_u_offset = *kts;
    sub_u__ -= sub_u_offset;
    sub_sqv_offset = *kts;
    sub_sqv__ -= sub_sqv_offset;
    sub_thl_offset = *kts;
    sub_thl__ -= sub_thl_offset;
    sd_awv1_offset = *kts;
    sd_awv1__ -= sd_awv1_offset;
    sd_awu1_offset = *kts;
    sd_awu1__ -= sd_awu1_offset;
    sd_awqnifa1_offset = *kts;
    sd_awqnifa1__ -= sd_awqnifa1_offset;
    sd_awqnwfa1_offset = *kts;
    sd_awqnwfa1__ -= sd_awqnwfa1_offset;
    sd_awqni1_offset = *kts;
    sd_awqni1__ -= sd_awqni1_offset;
    sd_awqnc1_offset = *kts;
    sd_awqnc1__ -= sd_awqnc1_offset;
    sd_awqi1_offset = *kts;
    sd_awqi1__ -= sd_awqi1_offset;
    sd_awqc1_offset = *kts;
    sd_awqc1__ -= sd_awqc1_offset;
    sd_awqv1_offset = *kts;
    sd_awqv1__ -= sd_awqv1_offset;
    sd_awqt1_offset = *kts;
    sd_awqt1__ -= sd_awqt1_offset;
    sd_awthl1_offset = *kts;
    sd_awthl1__ -= sd_awthl1_offset;
    sd_aw1_offset = *kts;
    sd_aw1__ -= sd_aw1_offset;
    s_awqnbca1_offset = *kts;
    s_awqnbca1__ -= s_awqnbca1_offset;
    s_awqnifa1_offset = *kts;
    s_awqnifa1__ -= s_awqnifa1_offset;
    s_awqnwfa1_offset = *kts;
    s_awqnwfa1__ -= s_awqnwfa1_offset;
    s_awqni1_offset = *kts;
    s_awqni1__ -= s_awqni1_offset;
    s_awqnc1_offset = *kts;
    s_awqnc1__ -= s_awqnc1_offset;
    s_awv1_offset = *kts;
    s_awv1__ -= s_awv1_offset;
    s_awu1_offset = *kts;
    s_awu1__ -= s_awu1_offset;
    s_awqc1_offset = *kts;
    s_awqc1__ -= s_awqc1_offset;
    s_awqv1_offset = *kts;
    s_awqv1__ -= s_awqv1_offset;
    s_awqt1_offset = *kts;
    s_awqt1__ -= s_awqt1_offset;
    s_awthl1_offset = *kts;
    s_awthl1__ -= s_awthl1_offset;
    s_aw1_offset = *kts;
    s_aw1__ -= s_aw1_offset;
    diss_heat_offset = *kts;
    diss_heat__ -= diss_heat_offset;
    dozone_offset = *kts;
    dozone -= dozone_offset;
    dqnbca_offset = *kts;
    dqnbca -= dqnbca_offset;
    dqnifa_offset = *kts;
    dqnifa -= dqnifa_offset;
    dqnwfa_offset = *kts;
    dqnwfa -= dqnwfa_offset;
    dqni_offset = *kts;
    dqni -= dqni_offset;
    dqnc_offset = *kts;
    dqnc -= dqnc_offset;
    dqs_offset = *kts;
    dqs -= dqs_offset;
    dqi_offset = *kts;
    dqi -= dqi_offset;
    dqc_offset = *kts;
    dqc -= dqc_offset;
    dqv_offset = *kts;
    dqv -= dqv_offset;
    dth_offset = *kts;
    dth -= dth_offset;
    dv_offset = *kts;
    dv -= dv_offset;
    du_offset = *kts;
    du -= du_offset;
    dfq_offset = *kts;
    dfq -= dfq_offset;
    dfh_offset = *kts;
    dfh -= dfh_offset;
    dfm_offset = *kts;
    dfm -= dfm_offset;
    qcd_offset = *kts;
    qcd -= qcd_offset;
    tcd_offset = *kts;
    tcd -= tcd_offset;
    cov_offset = *kts;
    cov -= cov_offset;
    qsq_offset = *kts;
    qsq -= qsq_offset;
    tsq_offset = *kts;
    tsq -= tsq_offset;
    ozone_offset = *kts;
    ozone -= ozone_offset;
    qnbca_offset = *kts;
    qnbca -= qnbca_offset;
    qnifa_offset = *kts;
    qnifa -= qnifa_offset;
    qnwfa_offset = *kts;
    qnwfa -= qnwfa_offset;
    sqw_offset = *kts;
    sqw -= sqw_offset;
    sqs_offset = *kts;
    sqs -= sqs_offset;
    sqi_offset = *kts;
    sqi -= sqi_offset;
    sqc_offset = *kts;
    sqc -= sqc_offset;
    sqv_offset = *kts;
    sqv -= sqv_offset;
    thl_offset = *kts;
    thl -= thl_offset;
    exner_offset = *kts;
    exner -= exner_offset;
    p_offset = *kts;
    p -= p_offset;
    qni_offset = *kts;
    qni -= qni_offset;
    qnc_offset = *kts;
    qnc -= qnc_offset;
    qs_offset = *kts;
    qs -= qs_offset;
    qi_offset = *kts;
    qi -= qi_offset;
    qc_offset = *kts;
    qc -= qc_offset;
    qv_offset = *kts;
    qv -= qv_offset;
    tk_offset = *kts;
    tk -= tk_offset;
    th_offset = *kts;
    th -= th_offset;
    v_offset = *kts;
    v -= v_offset;
    u_offset = *kts;
    u -= u_offset;
    rho_offset = *kts;
    rho -= rho_offset;
    zw_offset = *kts;
    zw -= zw_offset;
    dz_offset = *kts;
    dz -= dz_offset;

    /* Function Body */
    nonloc = rc0000;
    nc_min__ = rc0001;
    ni_min__ = rc0002;
    wfa_max__ = rc0003;
    wfa_min__ = rc0004;
    ifa_max__ = rc0005;
    ifa_min__ = rc0006;
    wfa_ht__ = rc0007;
    ifa_ht__ = rc0008;
    dztop = rc0009 * (dz[*kte] + dz[*kte - 1]);
    if (*bl_mynn_edmf_mom__ == 0) {
	onoff = zero;
    } else {
	onoff = one;
    }
    rhosfc = *psfc / (r_d__ * (tk[*kts] + p608 * qv[*kts]));
    dtz[*kts - 1] = *delt / dz[*kts];
    rhoz[*kts - 1] = rho[*kts];
    rhoinv[*kts - 1] = rc0010 / rho[*kts];
    khdz[*kts - 1] = rhoz[*kts - 1] * dfh[*kts];
    kmdz[*kts - 1] = rhoz[*kts - 1] * dfm[*kts];
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	dtz[k - 1] = *delt / dz[k];
	rhoz[k - 1] = (rho[k] * dz[k - 1] + rho[k - 1] * dz[k]) / (dz[k - 1] 
		+ dz[k]);
/* Computing MAX */
	r__1 = rhoz[k - 1];
	rhoz[k - 1] = max(r__1,rc0011);
/* Computing MAX */
	r__1 = rho[k];
	rhoinv[k - 1] = rc0010 / max(r__1,rc0011);
	dzk = rc0012 * (dz[k] + dz[k - 1]);
	khdz[k - 1] = rhoz[k - 1] * dfh[k];
	kmdz[k - 1] = rhoz[k - 1] * dfm[k];
/* L9004: */
    }
/* L9005: */
    rhoz[*kte] = rhoz[*kte - 1];
    khdz[*kte] = rhoz[*kte] * dfh[*kte];
    kmdz[*kte] = rhoz[*kte] * dfm[*kte];
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	delp[k - 1] = rho[k] * grav * dz[k];
/* L9007: */
    }
/* L9008: */
    if (delp[*kts - 1] < rc0012 * delp[*kts]) {
	delp[*kts - 1] = rc0012 * delp[*kts];
    }
    i__1 = *kte - 1;
    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
/* Computing MAX */
	r__1 = khdz[k - 1], r__2 = rc0012 * (s_aw1__[k] + sd_aw1__[k]);
	khdz[k - 1] = max(r__1,r__2);
/* Computing MAX */
	r__1 = khdz[k - 1], r__2 = -rc0012 * (s_aw1__[k] - s_aw1__[k + 1]) - 
		rc0012 * (sd_aw1__[k] - sd_aw1__[k + 1]);
	khdz[k - 1] = max(r__1,r__2);
/* Computing MAX */
	r__1 = kmdz[k - 1], r__2 = rc0012 * (s_aw1__[k] + sd_aw1__[k]);
	kmdz[k - 1] = max(r__1,r__2);
/* Computing MAX */
	r__1 = kmdz[k - 1], r__2 = -rc0012 * (s_aw1__[k] - s_aw1__[k + 1]) - 
		rc0012 * (sd_aw1__[k] - sd_aw1__[k + 1]);
	kmdz[k - 1] = max(r__1,r__2);
/* L9010: */
    }
/* L9011: */
/* Computing MIN */
    r__1 = *ust * *ust;
    ustdrag = min(r__1,rc0013) / *wspd;
/* Computing MIN */
    r__1 = *ust * *ust;
    ustdiff = min(r__1,rc0014) / *wspd;
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dth[*kts + ixarr1 - 1] = zero;
    }
    k = *kts;
    a[k - 1] = -dtz[k - 1] * kmdz[k - 1] * rhoinv[k - 1];
/* Computing 2nd power */
    r__1 = *ust;
    b[k - 1] = rc0010 + dtz[k - 1] * (kmdz[k] + rhosfc * (r__1 * r__1) / *
	    wspd) * rhoinv[k - 1] - rc0012 * dtz[k - 1] * rhoinv[k - 1] * 
	    s_aw1__[k + 1] * onoff - rc0012 * dtz[k - 1] * rhoinv[k - 1] * 
	    sd_aw1__[k + 1] * onoff;
    c__[k - 1] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1] - rc0012 * dtz[k - 1] *
	     rhoinv[k - 1] * s_aw1__[k + 1] * onoff - rc0012 * dtz[k - 1] * 
	    rhoinv[k - 1] * sd_aw1__[k + 1] * onoff;
/* Computing 2nd power */
    r__1 = *ust;
    d__[k - 1] = u[k] + dtz[k - 1] * *uoce * (r__1 * r__1) / *wspd - dtz[k - 
	    1] * rhoinv[k - 1] * s_awu1__[k + 1] * onoff + dtz[k - 1] * 
	    rhoinv[k - 1] * sd_awu1__[k + 1] * onoff + sub_u__[k] * *delt + 
	    det_u__[k] * *delt;
    i__1 = *kte - 1;
    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	a[k - 1] = -dtz[k - 1] * kmdz[k - 1] * rhoinv[k - 1] + rc0012 * dtz[k 
		- 1] * rhoinv[k - 1] * s_aw1__[k] * onoff + rc0012 * dtz[k - 
		1] * rhoinv[k - 1] * sd_aw1__[k] * onoff;
	b[k - 1] = rc0010 + dtz[k - 1] * (kmdz[k - 1] + kmdz[k]) * rhoinv[k - 
		1] + rc0012 * dtz[k - 1] * rhoinv[k - 1] * (s_aw1__[k] - 
		s_aw1__[k + 1]) * onoff + rc0012 * dtz[k - 1] * rhoinv[k - 1] 
		* (sd_aw1__[k] - sd_aw1__[k + 1]) * onoff;
	c__[k - 1] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1] - rc0012 * dtz[k - 
		1] * rhoinv[k - 1] * s_aw1__[k + 1] * onoff - rc0012 * dtz[k 
		- 1] * rhoinv[k - 1] * sd_aw1__[k + 1] * onoff;
	d__[k - 1] = u[k] + dtz[k - 1] * rhoinv[k - 1] * (s_awu1__[k] - 
		s_awu1__[k + 1]) * onoff - dtz[k - 1] * rhoinv[k - 1] * (
		sd_awu1__[k] - sd_awu1__[k + 1]) * onoff + sub_u__[k] * *delt 
		+ det_u__[k] * *delt;
/* L9013: */
    }
/* L9014: */
    a[*kte - 1] = 0.f;
    b[*kte - 1] = rc0010;
    c__[*kte - 1] = rc0015;
    d__[*kte - 1] = u[*kte];
    tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	du[k] = (x[k - 1] - u[k]) / *delt;
/* L9016: */
    }
/* L9017: */
    k = *kts;
    a[k - 1] = -dtz[k - 1] * kmdz[k - 1] * rhoinv[k - 1];
/* Computing 2nd power */
    r__1 = *ust;
    b[k - 1] = rc0010 + dtz[k - 1] * (kmdz[k] + rhosfc * (r__1 * r__1) / *
	    wspd) * rhoinv[k - 1] - rc0012 * dtz[k - 1] * rhoinv[k - 1] * 
	    s_aw1__[k + 1] * onoff - rc0012 * dtz[k - 1] * rhoinv[k - 1] * 
	    sd_aw1__[k + 1] * onoff;
    c__[k - 1] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1] - rc0012 * dtz[k - 1] *
	     rhoinv[k - 1] * s_aw1__[k + 1] * onoff - rc0012 * dtz[k - 1] * 
	    rhoinv[k - 1] * sd_aw1__[k + 1] * onoff;
/* Computing 2nd power */
    r__1 = *ust;
    d__[k - 1] = v[k] + dtz[k - 1] * *voce * (r__1 * r__1) / *wspd - dtz[k - 
	    1] * rhoinv[k - 1] * s_awv1__[k + 1] * onoff + dtz[k - 1] * 
	    rhoinv[k - 1] * sd_awv1__[k + 1] * onoff + sub_v__[k] * *delt + 
	    det_v__[k] * *delt;
    i__1 = *kte - 1;
    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	a[k - 1] = -dtz[k - 1] * kmdz[k - 1] * rhoinv[k - 1] + rc0012 * dtz[k 
		- 1] * rhoinv[k - 1] * s_aw1__[k] * onoff + rc0012 * dtz[k - 
		1] * rhoinv[k - 1] * sd_aw1__[k] * onoff;
	b[k - 1] = rc0010 + dtz[k - 1] * (kmdz[k - 1] + kmdz[k]) * rhoinv[k - 
		1] + rc0012 * dtz[k - 1] * rhoinv[k - 1] * (s_aw1__[k] - 
		s_aw1__[k + 1]) * onoff + rc0012 * dtz[k - 1] * rhoinv[k - 1] 
		* (sd_aw1__[k] - sd_aw1__[k + 1]) * onoff;
	c__[k - 1] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1] - rc0012 * dtz[k - 
		1] * rhoinv[k - 1] * s_aw1__[k + 1] * onoff - rc0012 * dtz[k 
		- 1] * rhoinv[k - 1] * sd_aw1__[k + 1] * onoff;
	d__[k - 1] = v[k] + dtz[k - 1] * rhoinv[k - 1] * (s_awv1__[k] - 
		s_awv1__[k + 1]) * onoff - dtz[k - 1] * rhoinv[k - 1] * (
		sd_awv1__[k] - sd_awv1__[k + 1]) * onoff + sub_v__[k] * *delt 
		+ det_v__[k] * *delt;
/* L9019: */
    }
/* L9020: */
    a[*kte - 1] = 0.f;
    b[*kte - 1] = rc0010;
    c__[*kte - 1] = rc0015;
    d__[*kte - 1] = v[*kte];
    tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	dv[k] = (x[k - 1] - v[k]) / *delt;
/* L9022: */
    }
/* L9023: */
    k = *kts;
    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
    b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k - 1] 
	    - rc0012 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - rc0012 * 
	    dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[k - 1] *
	     rhoinv[k - 1] * s_aw1__[k + 1] - rc0012 * dtz[k - 1] * rhoinv[k 
	    - 1] * sd_aw1__[k + 1];
    d__[k - 1] = thl[k] + dtz[k - 1] * rhosfc * *flt * rhoinv[k - 1] + tcd[k] 
	    * *delt - dtz[k - 1] * rhoinv[k - 1] * s_awthl1__[k + 1] + dtz[k 
	    - 1] * rhoinv[k - 1] * sd_awthl1__[k + 1] + diss_heat__[k] * *
	    delt + sub_thl__[k] * *delt + det_thl__[k] * *delt;
    i__1 = *kte - 1;
    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + rc0012 * dtz[k 
		- 1] * rhoinv[k - 1] * s_aw1__[k] + rc0012 * dtz[k - 1] * 
		rhoinv[k - 1] * sd_aw1__[k];
	b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[k - 
		1] + rc0012 * dtz[k - 1] * rhoinv[k - 1] * (s_aw1__[k] - 
		s_aw1__[k + 1]) + rc0012 * dtz[k - 1] * rhoinv[k - 1] * (
		sd_aw1__[k] - sd_aw1__[k + 1]);
	c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[k - 
		1] * rhoinv[k - 1] * s_aw1__[k + 1] - rc0012 * dtz[k - 1] * 
		rhoinv[k - 1] * sd_aw1__[k + 1];
	d__[k - 1] = thl[k] + tcd[k] * *delt + dtz[k - 1] * rhoinv[k - 1] * (
		s_awthl1__[k] - s_awthl1__[k + 1]) - dtz[k - 1] * rhoinv[k - 
		1] * (sd_awthl1__[k] - sd_awthl1__[k + 1]) + diss_heat__[k] * 
		*delt + sub_thl__[k] * *delt + det_thl__[k] * *delt;
/* L9025: */
    }
/* L9026: */
    a[*kte - 1] = rc0015;
    b[*kte - 1] = rc0010;
    c__[*kte - 1] = rc0015;
    d__[*kte - 1] = thl[*kte];
    tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	thl[k] = x[k - 1];
/* L9028: */
    }
/* L9029: */
    if (*bl_mynn_mixqt__ > 0) {
	k = *kts;
	a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k - 
		1] - rc0012 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - 
		rc0012 * dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[k - 
		1] * rhoinv[k - 1] * s_aw1__[k + 1] - rc0012 * dtz[k - 1] * 
		rhoinv[k - 1] * sd_aw1__[k + 1];
	d__[k - 1] = sqw[k] + dtz[k - 1] * rhosfc * *flq * rhoinv[k - 1] + 
		qcd[k] * *delt - dtz[k - 1] * rhoinv[k - 1] * s_awqt1__[k + 1]
		 + dtz[k - 1] * rhoinv[k - 1] * sd_awqt1__[k + 1];
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + rc0012 * 
		    dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] + rc0012 * dtz[k 
		    - 1] * rhoinv[k - 1] * sd_aw1__[k];
	    b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[
		    k - 1] + rc0012 * dtz[k - 1] * rhoinv[k - 1] * (s_aw1__[k]
		     - s_aw1__[k + 1]) + rc0012 * dtz[k - 1] * rhoinv[k - 1] *
		     (sd_aw1__[k] - sd_aw1__[k + 1]);
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[
		    k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - rc0012 * dtz[k 
		    - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	    d__[k - 1] = sqw[k] + qcd[k] * *delt + dtz[k - 1] * rhoinv[k - 1] 
		    * (s_awqt1__[k] - s_awqt1__[k + 1]) - dtz[k - 1] * rhoinv[
		    k - 1] * (sd_awqt1__[k] - sd_awqt1__[k + 1]);
/* L9031: */
	}
/* L9032: */
	a[*kte - 1] = rc0015;
	b[*kte - 1] = rc0010;
	c__[*kte - 1] = rc0015;
	d__[*kte - 1] = sqw[*kte];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, sqw2);
    } else {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    sqw2[ixarr1 - 1] = sqw[*kts + ixarr1 - 1];
	}
    }
    if (*bl_mynn_mixqt__ == 0) {
	if (*bl_mynn_cloudmix__ > 0 && *flag_qc__) {
	    k = *kts;
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[
		    k - 1] - rc0012 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k 
		    + 1] - rc0012 * dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 
		    1];
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[
		    k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - rc0012 * dtz[k 
		    - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	    d__[k - 1] = sqc[k] + dtz[k - 1] * rhosfc * *flqc * rhoinv[k - 1] 
		    + qcd[k] * *delt - dtz[k - 1] * rhoinv[k - 1] * s_awqc1__[
		    k + 1] + dtz[k - 1] * rhoinv[k - 1] * sd_awqc1__[k + 1] + 
		    det_sqc__[k] * *delt;
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + rc0012 
			* dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] + rc0012 * 
			dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k];
		b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1] + rc0012 * dtz[k - 1] * rhoinv[k - 1] * 
			(s_aw1__[k] - s_aw1__[k + 1]) + rc0012 * dtz[k - 1] * 
			rhoinv[k - 1] * (sd_aw1__[k] - sd_aw1__[k + 1]);
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * 
			dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - rc0012 *
			 dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
		d__[k - 1] = sqc[k] + qcd[k] * *delt + dtz[k - 1] * rhoinv[k 
			- 1] * (s_awqc1__[k] - s_awqc1__[k + 1]) - dtz[k - 1] 
			* rhoinv[k - 1] * (sd_awqc1__[k] - sd_awqc1__[k + 1]) 
			+ det_sqc__[k] * *delt;
/* L9034: */
	    }
/* L9035: */
	    a[*kte - 1] = rc0015;
	    b[*kte - 1] = rc0010;
	    c__[*kte - 1] = rc0015;
	    d__[*kte - 1] = sqc[*kte];
	    tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, sqc2);
	} else {
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
		sqc2[ixarr1 - 1] = sqc[*kts + ixarr1 - 1];
	    }
	}
    }
    if (*bl_mynn_mixqt__ == 0) {
	k = *kts;
	qvflux = *flqv;
	if (qvflux < zero) {
/* Computing MAX */
/* Computing MIN */
	    r__3 = rc0016 * sqv[*kts] - rc0017;
	    r__1 = qvflux, r__2 = min(r__3,zero) / dtz[*kts - 1];
	    qvflux = max(r__1,r__2);
	}
	a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k - 
		1] - rc0012 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - 
		rc0012 * dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[k - 
		1] * rhoinv[k - 1] * s_aw1__[k + 1] - rc0012 * dtz[k - 1] * 
		rhoinv[k - 1] * sd_aw1__[k + 1];
	d__[k - 1] = sqv[k] + dtz[k - 1] * rhosfc * qvflux * rhoinv[k - 1] + 
		qcd[k] * *delt - dtz[k - 1] * rhoinv[k - 1] * s_awqv1__[k + 1]
		 + dtz[k - 1] * rhoinv[k - 1] * sd_awqv1__[k + 1] + sub_sqv__[
		k] * *delt + det_sqv__[k] * *delt;
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + rc0012 * 
		    dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] + rc0012 * dtz[k 
		    - 1] * rhoinv[k - 1] * sd_aw1__[k];
	    b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[
		    k - 1] + rc0012 * dtz[k - 1] * rhoinv[k - 1] * (s_aw1__[k]
		     - s_aw1__[k + 1]) + rc0012 * dtz[k - 1] * rhoinv[k - 1] *
		     (sd_aw1__[k] - sd_aw1__[k + 1]);
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[
		    k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - rc0012 * dtz[k 
		    - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	    d__[k - 1] = sqv[k] + qcd[k] * *delt + dtz[k - 1] * rhoinv[k - 1] 
		    * (s_awqv1__[k] - s_awqv1__[k + 1]) - dtz[k - 1] * rhoinv[
		    k - 1] * (sd_awqv1__[k] - sd_awqv1__[k + 1]) + sub_sqv__[
		    k] * *delt + det_sqv__[k] * *delt;
/* L9037: */
	}
/* L9038: */
	a[*kte - 1] = rc0015;
	b[*kte - 1] = rc0010;
	c__[*kte - 1] = rc0015;
	d__[*kte - 1] = sqv[*kte];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, sqv2);
    } else {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    sqv2[ixarr1 - 1] = sqv[*kts + ixarr1 - 1];
	}
    }
    if (*bl_mynn_cloudmix__ > 0 && *flag_qi__) {
	k = *kts;
	a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k - 
		1] - rc0012 * dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[k - 
		1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	d__[k - 1] = sqi[k] + dtz[k - 1] * rhoinv[k - 1] * sd_awqi1__[k + 1];
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + rc0012 * 
		    dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k];
	    b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[
		    k - 1] + rc0012 * dtz[k - 1] * rhoinv[k - 1] * (sd_aw1__[
		    k] - sd_aw1__[k + 1]);
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[
		    k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	    d__[k - 1] = sqi[k] - dtz[k - 1] * rhoinv[k - 1] * (sd_awqi1__[k] 
		    - sd_awqi1__[k + 1]);
/* L9040: */
	}
/* L9041: */
	a[*kte - 1] = rc0015;
	b[*kte - 1] = rc0010;
	c__[*kte - 1] = rc0015;
	d__[*kte - 1] = sqi[*kte];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, sqi2);
    } else {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    sqi2[ixarr1 - 1] = sqi[*kts + ixarr1 - 1];
	}
    }
    if (FALSE_) {
	k = *kts;
	a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k - 
		1];
	c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	d__[k - 1] = sqs[k];
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[
		    k - 1];
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    d__[k - 1] = sqs[k];
/* L9043: */
	}
/* L9044: */
	a[*kte - 1] = rc0015;
	b[*kte - 1] = rc0010;
	c__[*kte - 1] = rc0015;
	d__[*kte - 1] = sqs[*kte];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, sqs2);
    } else {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    sqs2[ixarr1 - 1] = sqs[*kts + ixarr1 - 1];
	}
    }
    if (*bl_mynn_cloudmix__ > 0 && *flag_qni__ && *bl_mynn_mixscalars__ > 0) {
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = qni[k];
	    qni2[k - 1] = max(r__1,zero);
	    if (sqi[k] > rc0018) {
/* Computing MAX */
		r__1 = qni2[k - 1];
		qni2[k - 1] = max(r__1,ni_min__);
	    }
/* L9046: */
	}
/* L9047: */
	k = *kts;
	a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k - 
		1] - rc0012 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - 
		rc0012 * dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[k - 
		1] * rhoinv[k - 1] * s_aw1__[k + 1] - rc0012 * dtz[k - 1] * 
		rhoinv[k - 1] * sd_aw1__[k + 1];
	d__[k - 1] = qni2[k - 1] - dtz[k - 1] * rhoinv[k - 1] * s_awqni1__[k 
		+ 1] + dtz[k - 1] * rhoinv[k - 1] * sd_awqni1__[k + 1];
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + rc0012 * 
		    dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] + rc0012 * dtz[k 
		    - 1] * rhoinv[k - 1] * sd_aw1__[k];
	    b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[
		    k - 1] + rc0012 * dtz[k - 1] * rhoinv[k - 1] * (s_aw1__[k]
		     - s_aw1__[k + 1]) + rc0012 * dtz[k - 1] * rhoinv[k - 1] *
		     (sd_aw1__[k] - sd_aw1__[k + 1]);
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[
		    k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - rc0012 * dtz[k 
		    - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	    d__[k - 1] = qni2[k - 1] + dtz[k - 1] * rhoinv[k - 1] * (
		    s_awqni1__[k] - s_awqni1__[k + 1]) - dtz[k - 1] * rhoinv[
		    k - 1] * (sd_awqni1__[k] - sd_awqni1__[k + 1]);
/* L9049: */
	}
/* L9050: */
	a[*kte - 1] = rc0015;
	b[*kte - 1] = rc0010;
	c__[*kte - 1] = rc0015;
	d__[*kte - 1] = qni2[*kte - 1];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    qni2[k - 1] = x[k - 1];
	    if (sqi2[k - 1] > rc0018) {
/* Computing MAX */
		r__1 = qni2[k - 1];
		qni2[k - 1] = max(r__1,ni_min__);
	    }
/* L9052: */
	}
/* L9053: */
    } else {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    qni2[ixarr1 - 1] = qni[*kts + ixarr1 - 1];
	}
    }
    if (*bl_mynn_cloudmix__ > 0 && *flag_qnc__ && *bl_mynn_mixscalars__ > 0) {
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = qnc[k];
	    qnc2[k - 1] = max(r__1,zero);
	    if (sqc[k] > rc0018) {
/* Computing MAX */
		r__1 = qnc2[k - 1];
		qnc2[k - 1] = max(r__1,nc_min__);
	    }
/* L9055: */
	}
/* L9056: */
	k = *kts;
	a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k - 
		1] - rc0012 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - 
		rc0012 * dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[k - 
		1] * rhoinv[k - 1] * s_aw1__[k + 1] - rc0012 * dtz[k - 1] * 
		rhoinv[k - 1] * sd_aw1__[k + 1];
	d__[k - 1] = qnc2[k - 1] - dtz[k - 1] * rhoinv[k - 1] * s_awqnc1__[k 
		+ 1] + dtz[k - 1] * rhoinv[k - 1] * sd_awqnc1__[k + 1];
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + rc0012 * 
		    dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] + rc0012 * dtz[k 
		    - 1] * rhoinv[k - 1] * sd_aw1__[k];
	    b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[
		    k - 1] + rc0012 * dtz[k - 1] * rhoinv[k - 1] * (s_aw1__[k]
		     - s_aw1__[k + 1]) + rc0012 * dtz[k - 1] * rhoinv[k - 1] *
		     (sd_aw1__[k] - sd_aw1__[k + 1]);
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[
		    k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - rc0012 * dtz[k 
		    - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	    d__[k - 1] = qnc2[k - 1] + dtz[k - 1] * rhoinv[k - 1] * (
		    s_awqnc1__[k] - s_awqnc1__[k + 1]) - dtz[k - 1] * rhoinv[
		    k - 1] * (sd_awqnc1__[k] - sd_awqnc1__[k + 1]);
/* L9058: */
	}
/* L9059: */
	a[*kte - 1] = rc0015;
	b[*kte - 1] = rc0010;
	c__[*kte - 1] = rc0015;
	d__[*kte - 1] = qnc2[*kte - 1];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    qnc2[k - 1] = x[k - 1];
	    if (sqc2[k - 1] > rc0018) {
/* Computing MAX */
		r__1 = qnc2[k - 1];
		qnc2[k - 1] = max(r__1,nc_min__);
	    }
/* L9061: */
	}
/* L9062: */
    } else {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    qnc2[ixarr1 - 1] = qnc[*kts + ixarr1 - 1];
	}
    }
    if (*bl_mynn_cloudmix__ > 0 && *flag_qnwfa__ && *bl_mynn_mixscalars__ > 0)
	     {
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = qnwfa[k];
	    qnwfa2[k - 1] = max(r__1,zero);
/* L9064: */
	}
/* L9065: */
	k = *kts;
	a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k - 
		1] - rc0012 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - 
		rc0012 * dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[k - 
		1] * rhoinv[k - 1] * s_aw1__[k + 1] - rc0012 * dtz[k - 1] * 
		rhoinv[k - 1] * sd_aw1__[k + 1];
	d__[k - 1] = qnwfa2[k - 1] - dtz[k - 1] * rhoinv[k - 1] * 
		s_awqnwfa1__[k + 1] + dtz[k - 1] * rhoinv[k - 1] * 
		sd_awqnwfa1__[k + 1];
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + rc0012 * 
		    dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] + rc0012 * dtz[k 
		    - 1] * rhoinv[k - 1] * sd_aw1__[k];
	    b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[
		    k - 1] + rc0012 * dtz[k - 1] * rhoinv[k - 1] * (s_aw1__[k]
		     - s_aw1__[k + 1]) + rc0012 * dtz[k - 1] * rhoinv[k - 1] *
		     (sd_aw1__[k] - sd_aw1__[k + 1]);
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[
		    k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - rc0012 * dtz[k 
		    - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	    d__[k - 1] = qnwfa2[k - 1] + dtz[k - 1] * rhoinv[k - 1] * (
		    s_awqnwfa1__[k] - s_awqnwfa1__[k + 1]) - dtz[k - 1] * 
		    rhoinv[k - 1] * (sd_awqnwfa1__[k] - sd_awqnwfa1__[k + 1]);
/* L9067: */
	}
/* L9068: */
	a[*kte - 1] = rc0015;
	b[*kte - 1] = rc0010;
	c__[*kte - 1] = rc0015;
	d__[*kte - 1] = qnwfa2[*kte - 1];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = x[k - 1];
	    qnwfa2[k - 1] = max(r__1,zero);
/* L9070: */
	}
/* L9071: */
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    aero_min__ = wfa_min__ * (real)exp((real)(-zw[k] / wfa_ht__));
	    aero_max__ = wfa_max__ * (real)exp((real)(-zw[k] / wfa_ht__));
/* Computing MIN */
/* Computing MAX */
	    r__2 = aero_min__, r__3 = qnwfa2[k - 1];
	    r__1 = max(r__2,r__3);
	    qnwfa2[k - 1] = min(r__1,aero_max__);
/* L9073: */
	}
/* L9074: */
    } else {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    qnwfa2[ixarr1 - 1] = qnwfa[*kts + ixarr1 - 1];
	}
    }
    if (*bl_mynn_cloudmix__ > 0 && *flag_qnifa__ && *bl_mynn_mixscalars__ > 0)
	     {
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = qnifa[k];
	    qnifa2[k - 1] = max(r__1,zero);
/* L9076: */
	}
/* L9077: */
	k = *kts;
	a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k - 
		1] - rc0012 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - 
		rc0012 * dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[k - 
		1] * rhoinv[k - 1] * s_aw1__[k + 1] - rc0012 * dtz[k - 1] * 
		rhoinv[k - 1] * sd_aw1__[k + 1];
	d__[k - 1] = qnifa2[k - 1] - dtz[k - 1] * rhoinv[k - 1] * 
		s_awqnifa1__[k + 1] + dtz[k - 1] * rhoinv[k - 1] * 
		sd_awqnifa1__[k + 1];
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + rc0012 * 
		    dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] + rc0012 * dtz[k 
		    - 1] * rhoinv[k - 1] * sd_aw1__[k];
	    b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[
		    k - 1] + rc0012 * dtz[k - 1] * rhoinv[k - 1] * (s_aw1__[k]
		     - s_aw1__[k + 1]) + rc0012 * dtz[k - 1] * rhoinv[k - 1] *
		     (sd_aw1__[k] - sd_aw1__[k + 1]);
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[
		    k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - rc0012 * dtz[k 
		    - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	    d__[k - 1] = qnifa2[k - 1] + dtz[k - 1] * rhoinv[k - 1] * (
		    s_awqnifa1__[k] - s_awqnifa1__[k + 1]) - dtz[k - 1] * 
		    rhoinv[k - 1] * (sd_awqnifa1__[k] - sd_awqnifa1__[k + 1]);
/* L9079: */
	}
/* L9080: */
	a[*kte - 1] = rc0015;
	b[*kte - 1] = rc0010;
	c__[*kte - 1] = rc0015;
	d__[*kte - 1] = qnifa2[*kte - 1];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = x[k - 1];
	    qnifa2[k - 1] = max(r__1,zero);
/* L9082: */
	}
/* L9083: */
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    aero_min__ = ifa_min__ * (real)exp((real)(-zw[k] / ifa_ht__));
	    aero_max__ = ifa_max__ * (real)exp((real)(-zw[k] / ifa_ht__));
/* Computing MIN */
/* Computing MAX */
	    r__2 = aero_min__, r__3 = qnifa2[k - 1];
	    r__1 = max(r__2,r__3);
	    qnifa2[k - 1] = min(r__1,aero_max__);
/* L9085: */
	}
/* L9086: */
    } else {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    qnifa2[ixarr1 - 1] = qnifa[*kts + ixarr1 - 1];
	}
    }
    if (*bl_mynn_cloudmix__ > 0 && *flag_qnbca__ && *bl_mynn_mixscalars__ > 0)
	     {
	k = *kts;
	a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[k - 
		1] - rc0012 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] * 
		nonloc;
	c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[k - 
		1] * rhoinv[k - 1] * s_aw1__[k + 1] * nonloc;
	d__[k - 1] = qnbca[k] - dtz[k - 1] * rhoinv[k - 1] * s_awqnbca1__[k + 
		1] * nonloc;
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + rc0012 * 
		    dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] * nonloc;
	    b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[
		    k - 1] + rc0012 * dtz[k - 1] * rhoinv[k - 1] * (s_aw1__[k]
		     - s_aw1__[k + 1]) * nonloc;
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0012 * dtz[
		    k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] * nonloc;
	    d__[k - 1] = qnbca[k] + dtz[k - 1] * rhoinv[k - 1] * (
		    s_awqnbca1__[k] - s_awqnbca1__[k + 1]) * nonloc;
/* L9088: */
	}
/* L9089: */
	a[*kte - 1] = rc0015;
	b[*kte - 1] = rc0010;
	c__[*kte - 1] = rc0015;
	d__[*kte - 1] = qnbca[*kte];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    qnbca2[k - 1] = x[k - 1];
/* L9091: */
	}
/* L9092: */
    } else {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    qnbca2[ixarr1 - 1] = qnbca[*kts + ixarr1 - 1];
	}
    }
    if (*flag_ozone__) {
	k = *kts;
	a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k - 
		1];
	c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	d__[k - 1] = ozone[k];
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = rc0010 + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[
		    k - 1];
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    d__[k - 1] = ozone[k];
/* L9094: */
	}
/* L9095: */
	a[*kte - 1] = rc0015;
	b[*kte - 1] = rc0010;
	c__[*kte - 1] = rc0015;
	d__[*kte - 1] = ozone[*kte];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    dozone[k] = (x[k - 1] - ozone[k]) / *delt;
/* L9097: */
	}
/* L9098: */
    } else {
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dozone[*kts + ixarr1 - 1] = zero;
	}
    }
    if (*bl_mynn_mixqt__ > 0) {
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    th_new__ = thl[k] + xlvcp / exner[k] * sqc[k] + xlscp / exner[k] *
		     sqi[k];
	    t = th_new__ * exner[k];
	    qsat = (real)qsat_blend__(m471_work,m471_n, &t, &p[k]);
	    if (sqc[k] > zero || sqi[k] > zero) {
/* Computing MIN */
		r__1 = sqw2[k - 1];
		sqv2[k - 1] = min(r__1,qsat);
		portion_qc__ = sqc[k] / (sqc[k] + sqi[k]);
		portion_qi__ = sqi[k] / (sqc[k] + sqi[k]);
/* Computing MAX */
		r__1 = sqw2[k - 1] - qsat;
		condensate = max(r__1,zero);
		sqc2[k - 1] = condensate * portion_qc__;
		sqi2[k - 1] = condensate * portion_qi__;
	    } else {
		sqv2[k - 1] = sqw2[k - 1];
		sqi2[k - 1] = zero;
		sqc2[k - 1] = zero;
	    }
/* L9100: */
	}
/* L9101: */
    }
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	dqv[k] = (sqv2[k - 1] - sqv[k]) / *delt;
/* L9103: */
    }
/* L9104: */
    if (*bl_mynn_cloudmix__ > 0) {
	if (*flag_qc__) {
	    i__1 = *kte;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		dqc[k] = (sqc2[k - 1] - sqc[k]) / *delt;
/* L9106: */
	    }
/* L9107: */
	} else {
	    i__1 = *kte;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		dqc[k] = rc0015;
/* L9109: */
	    }
/* L9110: */
	}
	if (*flag_qnc__ && *bl_mynn_mixscalars__ > 0) {
	    i__1 = *kte;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		dqnc[k] = (qnc2[k - 1] - qnc[k]) / *delt;
/* L9112: */
	    }
/* L9113: */
	} else {
	    i__1 = *kte;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		dqnc[k] = rc0015;
/* L9115: */
	    }
/* L9116: */
	}
	if (*flag_qi__) {
	    i__1 = *kte;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		dqi[k] = (sqi2[k - 1] - sqi[k]) / *delt;
/* L9118: */
	    }
/* L9119: */
	} else {
	    i__1 = *kte;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		dqi[k] = rc0015;
/* L9121: */
	    }
/* L9122: */
	}
	if (FALSE_) {
	    i__1 = *kte;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		dqs[k] = (sqs2[k - 1] - sqs[k]) / *delt;
/* L9124: */
	    }
/* L9125: */
	} else {
	    i__1 = *kte;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		dqs[k] = rc0015;
/* L9127: */
	    }
/* L9128: */
	}
	if (*flag_qni__ && *bl_mynn_mixscalars__ > 0) {
	    i__1 = *kte;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		dqni[k] = (qni2[k - 1] - qni[k]) / *delt;
/* L9130: */
	    }
/* L9131: */
	} else {
	    i__1 = *kte;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		dqni[k] = rc0015;
/* L9133: */
	    }
/* L9134: */
	}
    } else {
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    dqc[k] = rc0015;
	    dqnc[k] = rc0015;
	    dqi[k] = rc0015;
	    dqni[k] = rc0015;
	    dqs[k] = rc0015;
/* L9136: */
	}
/* L9137: */
    }
    moisture_check__(m471_work,m471_n, kte, delt, delp, &exner[exner_offset], sqv2, sqc2, sqi2, 
	    sqs2, &thl[thl_offset], &dqv[dqv_offset], &dqc[dqc_offset], &dqi[
	    dqi_offset], &dqs[dqs_offset], &dth[dth_offset]);
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	if (dozone[k] * *delt + ozone[k] < rc0015) {
	    dozone[k] = -ozone[k] * rc0013 / *delt;
	}
/* L9139: */
    }
/* L9140: */
    if (*flag_qi__) {
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    dth[k] = (thl[k] + xlvcp / exner[k] * sqc2[k - 1] + xlscp / exner[
		    k] * sqi2[k - 1] - th[k]) / *delt;
/* L9142: */
	}
/* L9143: */
    } else {
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    dth[k] = (thl[k] + xlvcp / exner[k] * sqc2[k - 1] - th[k]) / *
		    delt;
/* L9145: */
	}
/* L9146: */
    }
    if (*flag_qnwfa__ && *flag_qnifa__ && *bl_mynn_mixscalars__ > 0) {
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    dqnwfa[k] = (qnwfa2[k - 1] - qnwfa[k]) / *delt;
	    dqnifa[k] = (qnifa2[k - 1] - qnifa[k]) / *delt;
/* L9148: */
	}
/* L9149: */
    } else {
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    dqnwfa[k] = rc0015;
	    dqnifa[k] = rc0015;
/* L9151: */
	}
/* L9152: */
    }
    if (*flag_qnbca__ && *bl_mynn_mixscalars__ > 0) {
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    dqnbca[k] = (qnbca2[k - 1] - qnbca[k]) / *delt;
/* L9154: */
	}
/* L9155: */
    } else {
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    dqnbca[k] = rc0015;
/* L9157: */
	}
/* L9158: */
    }
    if (FALSE_) {
	problem = FALSE_;
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
/* Computing 2nd power */
	    r__1 = u[k];
/* Computing 2nd power */
	    r__2 = v[k];
	    wsp = (real)sqrt((real)(r__1 * r__1 + r__2 * r__2));
/* Computing 2nd power */
	    r__1 = u[k] + du[k] * *delt;
/* Computing 2nd power */
	    r__2 = v[k] + du[k] * *delt;
	    wsp2 = (real)sqrt((real)(r__1 * r__1 + r__2 * r__2));
	    th2 = th[k] + dth[k] * *delt;
	    tk2 = th2 * exner[k];
	    if (wsp2 > rc0019 || tk2 > rc0020 || tk2 < rc0021) {
		problem = TRUE_;
		kproblem = k;
	    }
/* L9160: */
	}
/* L9161: */
	if (problem) {
	}
    }
    return 0;
} /* mynn_tendencies__ */

__device__ M471_CALL /* Subroutine */ int moisture_check__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kte, M471Ptr<real> delt, M471Ptr<real> dp, 
	M471Ptr<real> exner, M471Ptr<real> qv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<real> qs, M471Ptr<real> th, M471Ptr<real> dqv, M471Ptr<real> dqc, M471Ptr<real> dqi, M471Ptr<real> dqs, M471Ptr<real> dth) {
    /* Initialized data */

    real zero = 0.f;
    real half = .5f;
    real xlscp = 2837.232421875f;
    real xlvcp = 2488.80029296875f;
    real rc0000 = 1e-20f;
    real rc0001 = 1e-20f;
    real rc0002 = 2.f;

    /* System generated locals */
    integer i__1;
    real r__1, r__2;

    /* Local variables */
    integer k;
    real aa, dum, sum, dqc2, dqi2, dqs2, dqv2, qcmin, qimin, qvmin;

    /* Parameter adjustments */
    --dth;
    --dqs;
    --dqi;
    --dqc;
    --dqv;
    --th;
    --qs;
    --qi;
    --qc;
    --qv;
    --exner;
    --dp;

    /* Function Body */
    qvmin = rc0000;
    qcmin = zero;
    qimin = zero;
    
#pragma unroll 1
for (k = *kte; k >= 1; --k) {
/* Computing MAX */
	r__1 = zero, r__2 = qcmin - qc[k];
	dqc2 = max(r__1,r__2);
/* Computing MAX */
	r__1 = zero, r__2 = qimin - qi[k];
	dqi2 = max(r__1,r__2);
/* Computing MAX */
	r__1 = zero, r__2 = qimin - qs[k];
	dqs2 = max(r__1,r__2);
	dqc[k] += dqc2 / *delt;
	dqi[k] += dqi2 / *delt;
	dqs[k] += dqs2 / *delt;
	dqv[k] -= (dqc2 + dqi2 + dqs2) / *delt;
	dth[k] = dth[k] + xlvcp / exner[k] * (dqc2 / *delt) + xlscp / exner[k]
		 * ((dqi2 + dqs2) / *delt);
	qc[k] += dqc2;
	qi[k] += dqi2;
	qs[k] += dqs2;
	qv[k] = qv[k] - dqc2 - dqi2 - dqs2;
	th[k] = th[k] + xlvcp / exner[k] * dqc2 + xlscp / exner[k] * (dqi2 + 
		dqs2);
/* Computing MAX */
	r__1 = zero, r__2 = qvmin - qv[k];
	dqv2 = max(r__1,r__2);
	dqv[k] += dqv2 / *delt;
	qv[k] += dqv2;
	if (k != 1) {
	    qv[k - 1] -= dqv2 * dp[k] / dp[k - 1];
	    dqv[k - 1] -= dqv2 * dp[k] / dp[k - 1] / *delt;
	}
/* Computing MAX */
	r__1 = qv[k];
	qv[k] = max(r__1,qvmin);
/* Computing MAX */
	r__1 = qc[k];
	qc[k] = max(r__1,qcmin);
/* Computing MAX */
	r__1 = qi[k];
	qi[k] = max(r__1,qimin);
/* Computing MAX */
	r__1 = qs[k];
	qs[k] = max(r__1,qimin);
/* L9004: */
    }
/* L9005: */
    if (dqv2 > rc0001) {
	sum = zero;
	i__1 = *kte;
	
#pragma unroll 1
for (k = 1; k <= i__1; ++k) {
	    if (qv[k] > rc0002 * qvmin) {
		sum += qv[k] * dp[k];
	    }
/* L9007: */
	}
/* L9008: */
	aa = dqv2 * dp[1] / max(rc0001,sum);
	if (aa < half) {
	    i__1 = *kte;
	    
#pragma unroll 1
for (k = 1; k <= i__1; ++k) {
		if (qv[k] > rc0002 * qvmin) {
		    dum = aa * qv[k];
		    qv[k] -= dum;
		    dqv[k] -= dum / *delt;
		}
/* L9010: */
	    }
/* L9011: */
	} else {
	}
    }
    return 0;
} /* moisture_check__ */

__device__ M471_CALL /* Subroutine */ int mynn_mix_chem__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<integer> i__,
	 M471Ptr<real> delt, M471Ptr<real> dz, M471Ptr<real> pblh, M471Ptr<integer> nchem, M471Ptr<integer> kdvel, 
	M471Ptr<integer> ndvel, M471Ptr<real> chem1, M471Ptr<real> vd1, M471Ptr<real> rho, M471Ptr<real> flt, M471Ptr<real> tcd, M471Ptr<real> qcd, M471Ptr<real> dfh, M471Ptr<real> s_aw1__, M471Ptr<real> s_awchem1__, M471Ptr<real> emis_ant_no__, M471Ptr<real> frp, M471Ptr<logical> rrfs_sd__, M471Ptr<logical> enh_mix__, 
	M471Ptr<logical> smoke_dbg__) {
    /* Initialized data */

    real rc0000 = 10.f;
    real rc0001 = 100.f;
    real rc0002 = .5f;
    real rc0003 = 1.f;
    real rc0004 = 1e-4f;
    real rc0005 = .5f;
    real rc0006 = .15f;
    real rc0007 = 1.1f;
    real rc0008 = 2.f;
    real rc0009 = 0.f;

    /* System generated locals */
    integer dfh_offset, dz_offset, tcd_offset, qcd_offset, rho_offset, 
	    s_aw1_offset, chem1_dim1, chem1_offset, s_awchem1_dim1, 
	    s_awchem1_offset, i__1, i__2;
    real r__1, r__2;
    doublereal d__1, d__2;

    /* Builtin functions */
    

    /* Local variables */
    real khdz_old__;
    integer kmaxfire;
    M471Ptr<real> a=m471_array<real>(m471_work,(272*(MYNN471_NZ-49)+27600),m471_n);
M471Ptr<real> b=m471_array<real>(m471_work,(272*(MYNN471_NZ-49)+27800),m471_n);
M471Ptr<real> c__=m471_array<real>(m471_work,(272*(MYNN471_NZ-49)+28000),m471_n);
M471Ptr<real> d__=m471_array<real>(m471_work,(272*(MYNN471_NZ-49)+28200),m471_n);
    integer k;
    real  khdz_back__;
M471Ptr<real> x=m471_array<real>(m471_work,(272*(MYNN471_NZ-49)+28400),m471_n);
    integer ic;
    real dzk, no_threshold__, dztop, 
	    frp_threshold__, pblh_threshold__;
M471Ptr<real> dtz=m471_array<real>(m471_work,(272*(MYNN471_NZ-49)+28600),m471_n);
M471Ptr<real> khdz=m471_array<real>(m471_work,(272*(MYNN471_NZ-49)+28800),m471_n);
M471Ptr<real> rhoz=m471_array<real>(m471_work,(280*(MYNN471_NZ-49)+29000),m471_n);
M471Ptr<real> rhoinv=m471_array<real>(m471_work,(288*(MYNN471_NZ-49)+29200),m471_n);
    
    

    /* Parameter adjustments */
    s_aw1_offset = *kts;
    s_aw1__ -= s_aw1_offset;
    dfh_offset = *kts;
    dfh -= dfh_offset;
    qcd_offset = *kts;
    qcd -= qcd_offset;
    tcd_offset = *kts;
    tcd -= tcd_offset;
    rho_offset = *kts;
    rho -= rho_offset;
    dz_offset = *kts;
    dz -= dz_offset;
    s_awchem1_dim1 = *kte + 1 - *kts + 1;
    s_awchem1_offset = *kts + s_awchem1_dim1;
    s_awchem1__ -= s_awchem1_offset;
    chem1_dim1 = *kte - *kts + 1;
    chem1_offset = *kts + chem1_dim1;
    chem1 -= chem1_offset;
    --vd1;

    /* Function Body */
    no_threshold__ = rc0000;
    frp_threshold__ = rc0000;
    pblh_threshold__ = rc0001;
    dztop = rc0002 * (dz[*kte] + dz[*kte - 1]);
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	dtz[k - 1] = *delt / dz[k];
/* L9004: */
    }
/* L9005: */
    rhoz[*kts - 1] = rho[*kts];
    rhoinv[*kts - 1] = rc0003 / rho[*kts];
    khdz[*kts - 1] = rhoz[*kts - 1] * dfh[*kts];
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	rhoz[k - 1] = (rho[k] * dz[k - 1] + rho[k - 1] * dz[k]) / (dz[k - 1] 
		+ dz[k]);
/* Computing MAX */
	r__1 = rhoz[k - 1];
	rhoz[k - 1] = max(r__1,rc0004);
/* Computing MAX */
	r__1 = rho[k];
	rhoinv[k - 1] = rc0003 / max(r__1,rc0004);
	dzk = rc0005 * (dz[k] + dz[k - 1]);
	khdz[k - 1] = rhoz[k - 1] * dfh[k];
/* L9007: */
    }
/* L9008: */
    rhoz[*kte] = rhoz[*kte - 1];
    khdz[*kte] = rhoz[*kte] * dfh[*kte];
    i__1 = *kte - 1;
    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
/* Computing MAX */
	r__1 = khdz[k - 1], r__2 = rc0005 * s_aw1__[k];
	khdz[k - 1] = max(r__1,r__2);
/* Computing MAX */
	r__1 = khdz[k - 1], r__2 = -rc0005 * (s_aw1__[k] - s_aw1__[k + 1]);
	khdz[k - 1] = max(r__1,r__2);
/* L9010: */
    }
/* L9011: */
    if (*rrfs_sd__ && *enh_mix__) {
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    khdz_old__ = khdz[k - 1];
	    khdz_back__ = *pblh * rc0006 / dz[k];
	    if (*pblh < pblh_threshold__) {
		if (*emis_ant_no__ > no_threshold__) {
/* Computing MAX */
		    r__1 = rc0007 * khdz[k - 1], r__2 = (real)sqrt((real)(*
			    emis_ant_no__ / no_threshold__)) / dz[k] * rhoz[k 
			    - 1];
		    khdz[k - 1] = max(r__1,r__2);
		}
		if (*frp > frp_threshold__) {
		    r__1 = (real)log(*frp);
		    kmaxfire = ceiling_(m471_work,m471_n, &r__1);
/* Computing MAX */
		    d__1 =  (real)log(*frp);
		    d__2 =  rc0008;
		    r__1 = rc0007 * khdz[k - 1], r__2 = (rc0003 - k / (
			    kmaxfire * rc0008)) * ((real)pow_dd(&d__1, &d__2) 
			    - rc0008 * (real)log(*frp)) / dz[k] * rhoz[k - 1];
		    khdz[k - 1] = max(r__1,r__2);
		}
	    }
/* L9013: */
	}
/* L9014: */
    }
    i__1 = *nchem;
    
#pragma unroll 1
for (ic = 1; ic <= i__1; ++ic) {
	k = *kts;
	a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	b[k - 1] = rc0003 + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k - 
		1] - rc0005 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1];
	c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0005 * dtz[k - 
		1] * rhoinv[k - 1] * s_aw1__[k + 1];
	d__[k - 1] = chem1[k + ic * chem1_dim1] - dtz[k - 1] * vd1[ic] * 
		chem1[k + ic * chem1_dim1] - dtz[k - 1] * rhoinv[k - 1] * 
		s_awchem1__[k + 1 + ic * s_awchem1_dim1];
	i__2 = *kte - 1;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__2; ++k) {
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + rc0005 * 
		    dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k];
	    b[k - 1] = rc0003 + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[
		    k - 1] + rc0005 * dtz[k - 1] * rhoinv[k - 1] * (s_aw1__[k]
		     - s_aw1__[k + 1]);
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - rc0005 * dtz[
		    k - 1] * rhoinv[k - 1] * s_aw1__[k + 1];
	    d__[k - 1] = chem1[k + ic * chem1_dim1] + dtz[k - 1] * rhoinv[k - 
		    1] * (s_awchem1__[k + ic * s_awchem1_dim1] - s_awchem1__[
		    k + 1 + ic * s_awchem1_dim1]);
/* L9019: */
	}
/* L9020: */
	a[*kte - 1] = rc0009;
	b[*kte - 1] = rc0003;
	c__[*kte - 1] = rc0009;
	d__[*kte - 1] = chem1[*kte + ic * chem1_dim1];
	tridiag3_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__2 = *kte;
	
#pragma unroll 1
for (k = *kts; k <= i__2; ++k) {
	    chem1[k + ic * chem1_dim1] = x[k - 1];
/* L9022: */
	}
/* L9023: */
/* L9016: */
    }
/* L9017: */
    return 0;
} /* mynn_mix_chem__ */

__device__ M471_CALL /* Subroutine */ int retrieve_exchange_coeffs__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, 
	M471Ptr<real> dfm, M471Ptr<real> dfh, M471Ptr<real> dz, M471Ptr<real> km1, M471Ptr<real> kh1) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = .5f;

    /* System generated locals */
    integer dz_offset, dfm_offset, dfh_offset, km1_offset, kh1_offset, i__1;

    /* Local variables */
    integer k;
    real dzk;

    /* Parameter adjustments */
    kh1_offset = *kts;
    kh1 -= kh1_offset;
    km1_offset = *kts;
    km1 -= km1_offset;
    dz_offset = *kts;
    dz -= dz_offset;
    dfh_offset = *kts;
    dfh -= dfh_offset;
    dfm_offset = *kts;
    dfm -= dfm_offset;

    /* Function Body */
    km1[(0 + (0 + (*kts << 2))) / 4] = rc0000;
    kh1[*kts] = rc0000;
    i__1 = *kte;
    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	dzk = rc0001 * (dz[k] + dz[k - 1]);
	km1[k] = dfm[k] * dzk;
	kh1[k] = dfh[k] * dzk;
/* L9004: */
    }
/* L9005: */
    return 0;
} /* retrieve_exchange_coeffs__ */

__device__ M471_CALL /* Subroutine */ int tridiag_(unsigned char *m471_work, int m471_n, M471Ptr<integer> n, M471Ptr<real> a, M471Ptr<real> b, M471Ptr<real> c__, M471Ptr<real> d__) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 1.f;

    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__;
    real p;
M471Ptr<real> q=m471_array<real>(m471_work,(288*(MYNN471_NZ-49)+29400),m471_n);

    /* Parameter adjustments */
    --d__;
    --c__;
    --b;
    --a;

    /* Function Body */
    c__[(0 + (0 + (*n << 2))) / 4] = rc0000;
    q[0] = -c__[1] / b[1];
    d__[1] /= b[1];
    i__1 = *n;
    
#pragma unroll 1
for (i__ = 2; i__ <= i__1; ++i__) {
	p = rc0001 / (b[i__] + a[i__] * q[i__ - 2]);
	q[i__ - 1] = -c__[i__] * p;
	d__[i__] = (d__[i__] - a[i__] * d__[i__ - 1]) * p;
/* L9004: */
    }
/* L9005: */
    
#pragma unroll 1
for (i__ = *n - 1; i__ >= 1; --i__) {
	d__[i__] += q[i__ - 1] * d__[i__ + 1];
/* L9007: */
    }
/* L9008: */
    return 0;
} /* tridiag_ */

__device__ M471_CALL /* Subroutine */ int tridiag2_(unsigned char *m471_work, int m471_n, M471Ptr<integer> kte, M471Ptr<real> a, M471Ptr<real> b, M471Ptr<real> c__, 
	M471Ptr<real> d__, M471Ptr<real> x) {

    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer k;
    real m;
M471Ptr<real> cp=m471_array<real>(m471_work,(288*(MYNN471_NZ-49)+29600),m471_n);
M471Ptr<real> dp=m471_array<real>(m471_work,(288*(MYNN471_NZ-49)+29800),m471_n);

    /* Parameter adjustments */
    --x;
    --d__;
    --c__;
    --b;
    --a;

    /* Function Body */
    cp[0] = c__[1] / b[1];
    dp[0] = d__[1] / b[1];
    i__1 = *kte;
    
#pragma unroll 1
for (k = 2; k <= i__1; ++k) {
	m = b[k] - cp[k - 2] * a[k];
	cp[k - 1] = c__[k] / m;
	dp[k - 1] = (d__[k] - dp[k - 2] * a[k]) / m;
/* L9004: */
    }
/* L9005: */
    x[*kte] = dp[*kte - 1];
    
#pragma unroll 1
for (k = *kte - 1; k >= 1; --k) {
	x[k] = dp[k - 1] - cp[k - 1] * x[k + 1];
/* L9007: */
    }
/* L9008: */
    return 0;
} /* tridiag2_ */

__device__ M471_CALL /* Subroutine */ int tridiag3_(unsigned char *m471_work, int m471_n, M471Ptr<integer> kte, M471Ptr<real> a, M471Ptr<real> b, M471Ptr<real> c__, 
	M471Ptr<real> d__, M471Ptr<real> x) {

    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer k;

    /* Parameter adjustments */
    --x;
    --d__;
    --c__;
    --b;
    --a;

    /* Function Body */
    
#pragma unroll 1
for (k = *kte - 1; k >= 1; --k) {
	d__[k] -= c__[k] * d__[k + 1] / b[k + 1];
	b[k] -= c__[k] * a[k + 1] / b[k + 1];
/* L9004: */
    }
/* L9005: */
    i__1 = *kte;
    
#pragma unroll 1
for (k = 2; k <= i__1; ++k) {
	d__[k] -= a[k] * d__[k - 1] / b[k - 1];
/* L9007: */
    }
/* L9008: */
    i__1 = *kte;
    
#pragma unroll 1
for (k = 1; k <= i__1; ++k) {
	x[k] = d__[k] / b[k];
/* L9010: */
    }
/* L9011: */
    return 0;
} /* tridiag3_ */

__device__ M471_CALL /* Subroutine */ int get_pblh__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> pblh, M471Ptr<real> thv1, M471Ptr<real> qke1, M471Ptr<real> zw1, M471Ptr<real> dz1, M471Ptr<real> landsea, M471Ptr<integer> kpbl) {
    /* Initialized data */

    real zero = 0.f;
    real one = 1.f;
    real rc0000 = 200.f;
    real rc0001 = 400.f;
    real rc0002 = 9e9f;
    real rc0003 = 0.f;
    real rc0004 = 1.5f;
    real rc0005 = 1.25f;
    real rc0006 = 1e-6f;
    real rc0007 = 40.f;
    real rc0008 = .01f;
    real rc0009 = .5f;
    real rc0010 = 350.f;
    real rc0011 = 10.f;
    real rc0012 = .5f;
    real rc0013 = 1.f;

    /* System generated locals */
    integer thv1_offset, qke1_offset, dz1_offset, zw1_offset, i__1;
    real r__1, r__2, r__3;

    /* Builtin functions */
    

    /* Local variables */
    real sbl_damp__, pblh_tke__, delt_thv__;
    integer k;
    real wt;
    integer ktke;
    real qtke;
    integer kthv;
    real qtkem1, maxqke, tkeeps, minthv, sbl_lim__;

    /* Parameter adjustments */
    dz1_offset = *kts;
    dz1 -= dz1_offset;
    zw1_offset = *kts;
    zw1 -= zw1_offset;
    qke1_offset = *kts;
    qke1 -= qke1_offset;
    thv1_offset = *kts;
    thv1 -= thv1_offset;

    /* Function Body */
    sbl_lim__ = rc0000;
    sbl_damp__ = rc0001;
    *kpbl = 2;
    k = *kts + 1;
    kthv = 1;
    minthv = rc0002;
L9003:
    if (! (zw1[k] <= rc0000)) {
	goto L9004;
    }
    if (minthv > thv1[k]) {
	minthv = thv1[k];
	kthv = k;
    }
    ++k;
    goto L9003;
L9004:
    *pblh = rc0003;
    k = kthv + 1;
    if (*landsea - rc0004 >= zero) {
	delt_thv__ = one;
    } else {
	delt_thv__ = rc0005;
    }
    *pblh = rc0003;
    k = kthv + 1;
    i__1 = *kte - 1;
    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	if (thv1[k] >= minthv + delt_thv__) {
/* Computing MIN */
/* Computing MAX */
	    r__2 = thv1[k] - thv1[k - 1];
	    r__1 = (thv1[k] - (minthv + delt_thv__)) / max(r__2,rc0006);
	    *pblh = zw1[k] - dz1[k - 1] * min(r__1,one);
	}
	if (k == *kte - 1) {
	    *pblh = zw1[*kts + 1];
	}
	if (*pblh != zero) {
	    goto L9008;
	}
/* L9007: */
    }
L9008:
    ktke = 1;
/* Computing MAX */
    r__1 = qke1[*kts];
    maxqke = max(r__1,rc0003);
    tkeeps = maxqke / rc0007;
    tkeeps = max(tkeeps,rc0008);
    pblh_tke__ = rc0003;
    k = ktke + 1;
    i__1 = *kte - 1;
    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
/* Computing MAX */
	r__1 = rc0009 * qke1[k];
	qtke = max(r__1,rc0003);
/* Computing MAX */
	r__1 = rc0009 * qke1[k - 1];
	qtkem1 = max(r__1,rc0003);
	if (qtke <= tkeeps) {
/* Computing MIN */
/* Computing MAX */
	    r__2 = qtkem1 - qtke;
	    r__1 = (tkeeps - qtke) / max(r__2,rc0006);
	    pblh_tke__ = zw1[k] - dz1[k - 1] * min(r__1,one);
/* Computing MAX */
	    r__1 = pblh_tke__, r__2 = zw1[*kts + 1];
	    pblh_tke__ = max(r__1,r__2);
	}
	if (k == *kte - 1) {
	    pblh_tke__ = zw1[*kts + 1];
	}
	if (pblh_tke__ != rc0003) {
	    goto L9011;
	}
/* L9010: */
    }
L9011:
/* Computing MIN */
    r__1 = pblh_tke__, r__2 = *pblh + rc0010;
    pblh_tke__ = min(r__1,r__2);
/* Computing MAX */
/* Computing MAX */
    r__3 = *pblh - rc0010;
    r__1 = pblh_tke__, r__2 = max(r__3,rc0011);
    pblh_tke__ = max(r__1,r__2);
    wt = rc0012 * (real)tanh((real)((*pblh - sbl_lim__) / sbl_damp__)) + 
	    rc0012;
    if (maxqke <= tkeeps) {
    } else {
	*pblh = pblh_tke__ * (rc0013 - wt) + *pblh * wt;
    }
    i__1 = *kte - 1;
    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	if (zw1[k] >= *pblh) {
	    *kpbl = k - 1;
	    goto L9014;
	}
/* L9013: */
    }
L9014:
    return 0;
} /* get_pblh__ */

template<int M471_PHASE>
__device__ M471_CALL /* Subroutine */ int dmp_mf__(unsigned char *m471_work, int m471_n, M471Ptr<integer> i__, M471Ptr<integer> j, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dt, M471Ptr<real> zw1, M471Ptr<real> dz1, M471Ptr<real> p1, M471Ptr<real> rho1, M471Ptr<integer> momentum_opt__, M471Ptr<integer> tke_opt__, M471Ptr<integer> scalar_opt__, M471Ptr<real> u1, 
	M471Ptr<real> v1, M471Ptr<real> w1, M471Ptr<real> th1, M471Ptr<real> thl1, M471Ptr<real> thv1, M471Ptr<real> tk1, 
	M471Ptr<real> qt1, M471Ptr<real> qv1, M471Ptr<real> qc1, M471Ptr<real> qke1, M471Ptr<real> qnc1, M471Ptr<real> qni1, 
	M471Ptr<real> qnwfa1, M471Ptr<real> qnifa1, M471Ptr<real> qnbca1, M471Ptr<real> ex1, M471Ptr<real> vt1, M471Ptr<real> vq1, M471Ptr<real> sgm1, M471Ptr<real> ust, M471Ptr<real> flt, M471Ptr<real> fltv, M471Ptr<real> flq, M471Ptr<real> flqv, M471Ptr<real> pblh, M471Ptr<integer> kpbl, M471Ptr<real> dx, M471Ptr<real> landsea, M471Ptr<real> ts, 
	M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_qt1__, M471Ptr<real> edmf_thl1__,
	 M471Ptr<real> edmf_ent1__, M471Ptr<real> edmf_qc1__, M471Ptr<real> s_aw1__, M471Ptr<real> s_awthl1__,
	 M471Ptr<real> s_awqt1__, M471Ptr<real> s_awqv1__, M471Ptr<real> s_awqc1__, M471Ptr<real> s_awu1__, 
	M471Ptr<real> s_awv1__, M471Ptr<real> s_awqke1__, M471Ptr<real> s_awqnc1__, M471Ptr<real> s_awqni1__, 
	M471Ptr<real> s_awqnwfa1__, M471Ptr<real> s_awqnifa1__, M471Ptr<real> s_awqnbca1__, M471Ptr<real> sub_thl1__, M471Ptr<real> sub_sqv1__, M471Ptr<real> sub_u1__, M471Ptr<real> sub_v1__, M471Ptr<real> det_thl1__, M471Ptr<real> det_sqv1__, M471Ptr<real> det_sqc1__, M471Ptr<real> det_u1__, M471Ptr<real> det_v1__, M471Ptr<integer> nchem, M471Ptr<real> chem1, M471Ptr<real> s_awchem1__, M471Ptr<logical> mix_chem__, M471Ptr<real> qc_bl1__, M471Ptr<real> cldfra_bl1__, M471Ptr<real> qc_bl1_old__, 
	M471Ptr<real> cldfra_bl1_old__, M471Ptr<logical> f_qc__, M471Ptr<logical> f_qi__, M471Ptr<logical> f_qnc__, M471Ptr<logical> f_qni__, M471Ptr<logical> f_qnwfa__, M471Ptr<logical> f_qnifa__, 
	M471Ptr<logical> f_qnbca__, M471Ptr<real> psig_shcu__, M471Ptr<real> maxwidth, M471Ptr<integer> ktop, 
	M471Ptr<real> maxmf, M471Ptr<real> ztop, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__,
	 M471Ptr<real> tkeprod_up__, M471Ptr<real> el1) {
    /* Initialized data */

    real cp = 1004.5f;
    real r_v__ = 461.600006103515625f;
    real p608 = .6083624362945556640625f;
    real ep_2__ = .621750414371490478515625f;
    real cpv = 1846.4000244140625f;
    real zero = 0.f;
    real half = .5f;
    real one = 1.f;
    real onethird = .3333333432674407958984375f;
    real grav = 9.81000041961669921875f;
    real ep_3__ = .378249585628509521484375f;
    real gtr = .032700002193450927734375f;
    real tv0 = 182.50872802734375f;
    real xlvcp = 2488.80029296875f;
    real b1 = 24.f;
    real rc0000 = 2.f;
    real rc0001 = 3.f;
    real rc0002 = .002f;
    real rc0003 = 1.5f;
    real rc0004 = .1f;
    real rc0005 = 1e3f;
    real rc0006 = 300.f;
    real rc0007 = 0.f;
    real rc0008 = 1.2f;
    real rc0009 = .5f;
    real rc0010 = .75f;
    real rc0011 = 1.f;
    real rc0012 = 45.f;
    real rc0013 = .25f;
    real rc0014 = 0.f;
    real rc0015 = .001f;
    real rc0016 = 9e3f;
    real rc0017 = 500.f;
    real rc0018 = 50.f;
    real rc0019 = 1e-5f;
    real rc0020 = .003f;
    real rc0021 = 5e-4f;
    real rc0022 = 1.1f;
    real rc0023 = 400.f;
    real rc0024 = .9f;
    real rc0025 = .01f;
    real rc0026 = .6f;
    real rc0027 = .04f;
    real rc0028 = .04f;
    real rc0029 = .5f;
    real rc0030 = .007f;
    real rc0031 = .02f;
    real rc0032 = 1.9f;
    real rc0033 = .05f;
    real rc0034 = .012f;
    real rc0035 = .03f;
    real rc0036 = 10.f;
    real rc0037 = 13.f;
    real rc0038 = .1f;
    real rc0039 = .4f;
    real rc0040 = .01f;
    real rc0041 = 1e-5f;
    real rc0042 = 1.34f;
    real rc0043 = .58f;
    real rc0044 = 4.f;
    real rc0045 = .8f;
    real rc0046 = 1.2e4f;
    real rc0047 = 3e3f;
    real rc0048 = .3f;
    real rc0049 = .33f;
    real rc0050 = 3e-4f;
    real rc0051 = 1500.f;
    real rc0052 = 4e3f;
    real rc0053 = 5e-6f;
    real rc0054 = .3333f;
    real rc0055 = .15f;
    real rc0056 = .2f;
    real rc0057 = 250.f;
    real rc0058 = 1.25f;
    real rc0059 = 200.f;
    real rc0060 = 2.f;
    real rc0061 = 3.f;
    real rc0062 = 8e-5f;
    real rc0063 = .06f;
    real rc0064 = 2e-4f;
    real rc0065 = 1e-4f;
    real rc0066 = 1e-8f;
    real rc0067 = 1e-6f;
    real rc0068 = 3.402823466e38f;
    real rc0069 = 10.f;
    real rc0070 = 1e-6f;
    real rc0071 = 1e-7f;
    real rc0072 = .61f;
    real rc0073 = 1.61f;
    real rc0074 = 1.f;
    real rc0075 = .36f;
    real rc0076 = 1.55f;
    real rc0077 = 1.8f;
    real rc0078 = 5e-5f;
    real rc0079 = 1.86f;
    real rc0080 = 2.2e-5f;
    real rc0081 = 1.18f;
    real rc0082 = 2.25f;
    real rc0083 = 1.7f;
    real rc0084 = 2.5f;
    real rc0085 = 3.8f;
    real rc0086 = 23.9f;
    real rc0087 = 1.6f;
    real rc0088 = 60.f;
    real rc0089 = 1e-8f;

    /* System generated locals */
    integer pattern_spp_pbl1_offset, u1_offset, v1_offset, w1_offset, 
	    th1_offset, thl1_offset, tk1_offset, qt1_offset, qv1_offset, 
	    qc1_offset, ex1_offset, dz1_offset, thv1_offset, p1_offset, 
	    rho1_offset, qke1_offset, qnc1_offset, qni1_offset, qnwfa1_offset,
	     qnifa1_offset, qnbca1_offset, el1_offset, zw1_offset, 
	    edmf_a1_offset, edmf_w1_offset, edmf_qt1_offset, edmf_thl1_offset,
	     edmf_ent1_offset, edmf_qc1_offset, s_aw1_offset, s_awthl1_offset,
	     s_awqt1_offset, s_awqv1_offset, s_awqc1_offset, s_awqnc1_offset, 
	    s_awqni1_offset, s_awqnwfa1_offset, s_awqnifa1_offset, 
	    s_awqnbca1_offset, s_awu1_offset, s_awv1_offset, s_awqke1_offset, 
	    qc_bl1_offset, cldfra_bl1_offset, qc_bl1_old_offset, 
	    cldfra_bl1_old_offset, tkeprod_up_offset, chem1_dim1, 
	    chem1_offset, s_awchem1_dim1, s_awchem1_offset, vt1_offset, 
	    vq1_offset, sgm1_offset, sub_thl1_offset, sub_sqv1_offset, 
	    sub_u1_offset, sub_v1_offset, det_thl1_offset, det_sqv1_offset, 
	    det_sqc1_offset, det_u1_offset, det_v1_offset, i__1, i__2;
    real r__1, r__2, r__3;
    doublereal d__1, d__2;

    /* Builtin functions */
    

    /* Local variables */
    real exc_heat__;
    
    real qc_plume__, wspd_pbl__, minwidth, a, b, c__, d__, f;
M471Ptr<real> envm_thl__=m471_array<real>(m471_work,(288*(MYNN471_NZ-49)+30000),m471_n);
M471Ptr<real> envm_sqc__=m471_array<real>(m471_work,(288*(MYNN471_NZ-49)+30200),m471_n);
M471Ptr<real> envm_sqv__=m471_array<real>(m471_work,(288*(MYNN471_NZ-49)+30400),m471_n);
    integer k;
    real l, n, b9, cf_thresh__, q1, z0, 
	    exc_moist__, detrateuv;
M471Ptr<real> edmf_chem__=m471_array<real>(m471_work,(288*(MYNN471_NZ-49)+30600),m471_n);
    
    real bb;
    integer overshoot, ic, k50;
    real dl, cn;
    integer ip;
    real wa, pk, wb, wc, uk, un, vn, wn, xl, pt, vk;
    
    real cloud_base__;
    
    real an2, q2p, adjustment, fng, bvf, cpm, qcn, aup, qmq, rsl, qtn, 
	    hux, thp, wlv, qww, qtp, qcp, qsl,
	     frz, dzp, oow, tvs, flx1, flt2, ukm1, maxwidth_dx__, vkm1, nup2, 
	    beta, cdet, fluxportion, esat, qken, qncn, lmax, lmin, wpbl, thln,
	     qnin, wmin, wmax, maxw, thvn, 
	    thvk, atot, dcut, sigq, 
	    temp, red001, red002, red003, 
	    zagl, csub, dxsa, maxwidth_cld__, maxwidth_pbl__, maxwidth_flx__, 
	    fltv2, acfac, ac_mf__, mf_cf__, pgfac, alpha, dlmin, 
	    maxqc, qstar, pwmin, pwmax, wstar;
M471Ptr<real> upa=m471_array<real>(m471_work,(288*(MYNN471_NZ-49)+30800),m471_n);
M471Ptr<real> ent=m471_array<real>(m471_work,(320*(MYNN471_NZ-49)+32400),m471_n);
M471Ptr<real> dzi=m471_array<real>(m471_work,(352*(MYNN471_NZ-49)+33968),m471_n);
M471Ptr<real> upw=m471_array<real>(m471_work,(352*(MYNN471_NZ-49)+34168),m471_n);
M471Ptr<real> upu=m471_array<real>(m471_work,(384*(MYNN471_NZ-49)+35768),m471_n);
M471Ptr<real> upv=m471_array<real>(m471_work,(416*(MYNN471_NZ-49)+37368),m471_n);
M471Ptr<real> upqc=m471_array<real>(m471_work,(448*(MYNN471_NZ-49)+38968),m471_n);
M471Ptr<real> rhoz=m471_array<real>(m471_work,(480*(MYNN471_NZ-49)+40568),m471_n);
M471Ptr<real> upqt=m471_array<real>(m471_work,(480*(MYNN471_NZ-49)+40768),m471_n);
M471Ptr<real> upqv=m471_array<real>(m471_work,(512*(MYNN471_NZ-49)+42368),m471_n);
M471Ptr<real> chemn=m471_array<real>(m471_work,(544*(MYNN471_NZ-49)+43968),m471_n);
M471Ptr<real> upqke=m471_array<real>(m471_work,(544*(MYNN471_NZ-49)+43976),m471_n);
M471Ptr<real> upqnc=m471_array<real>(m471_work,(576*(MYNN471_NZ-49)+45576),m471_n);
M471Ptr<real> upthl=m471_array<real>(m471_work,(608*(MYNN471_NZ-49)+47176),m471_n);
M471Ptr<real> upqni=m471_array<real>(m471_work,(640*(MYNN471_NZ-49)+48776),m471_n);
M471Ptr<real> upthv=m471_array<real>(m471_work,(672*(MYNN471_NZ-49)+50376),m471_n);
    integer ixarr1, ixarr2, ixarr3;
    real thvkm1, bcoeff, qnbcan, csigma, qc_ent__, qnifan, 
	    ac_wsp__, aratio, 
	    qc_sgs__, sigmaw, psig_w__, qnwfan, entexm, sublim, qv_ent__, entexp, 
	    qt_ent__, thstar, dthvdz, dzpmax;
M471Ptr<real> envi_a__=m471_array<real>(m471_work,(704*(MYNN471_NZ-49)+51976),m471_n);
M471Ptr<real> upchem=m471_array<real>(m471_work,(712*(MYNN471_NZ-49)+52176),m471_n);
M471Ptr<real> envm_u__=m471_array<real>(m471_work,(744*(MYNN471_NZ-49)+53776),m471_n);
M471Ptr<real> exneri=m471_array<real>(m471_work,(744*(MYNN471_NZ-49)+53976),m471_n);
M471Ptr<real> envm_v__=m471_array<real>(m471_work,(744*(MYNN471_NZ-49)+54176),m471_n);
M471Ptr<real> envi_w__=m471_array<real>(m471_work,(744*(MYNN471_NZ-49)+54376),m471_n);
    logical superadiabatic;
    real exc_fac__, qc_grid__, detrate, 
	    sigmath, thl_ent__, sigmaqt, 
	    qsat_tk__, detturb;
M471Ptr<real> upqnbca=m471_array<real>(m471_work,(752*(MYNN471_NZ-49)+54576),m471_n);
M471Ptr<real> upqnifa=m471_array<real>(m471_work,(784*(MYNN471_NZ-49)+56176),m471_n);
M471Ptr<real> upqnwfa=m471_array<real>(m471_work,(816*(MYNN471_NZ-49)+57776),m471_n);
M471Ptr<real> edmf_th1__=m471_array<real>(m471_work,(848*(MYNN471_NZ-49)+59376),m471_n);

    /* Parameter adjustments */
    el1_offset = *kts;
    el1 -= el1_offset;
    tkeprod_up_offset = *kts;
    tkeprod_up__ -= tkeprod_up_offset;
    pattern_spp_pbl1_offset = *kts;
    pattern_spp_pbl1__ -= pattern_spp_pbl1_offset;
    cldfra_bl1_old_offset = *kts;
    cldfra_bl1_old__ -= cldfra_bl1_old_offset;
    qc_bl1_old_offset = *kts;
    qc_bl1_old__ -= qc_bl1_old_offset;
    cldfra_bl1_offset = *kts;
    cldfra_bl1__ -= cldfra_bl1_offset;
    qc_bl1_offset = *kts;
    qc_bl1__ -= qc_bl1_offset;
    det_v1_offset = *kts;
    det_v1__ -= det_v1_offset;
    det_u1_offset = *kts;
    det_u1__ -= det_u1_offset;
    det_sqc1_offset = *kts;
    det_sqc1__ -= det_sqc1_offset;
    det_sqv1_offset = *kts;
    det_sqv1__ -= det_sqv1_offset;
    det_thl1_offset = *kts;
    det_thl1__ -= det_thl1_offset;
    sub_v1_offset = *kts;
    sub_v1__ -= sub_v1_offset;
    sub_u1_offset = *kts;
    sub_u1__ -= sub_u1_offset;
    sub_sqv1_offset = *kts;
    sub_sqv1__ -= sub_sqv1_offset;
    sub_thl1_offset = *kts;
    sub_thl1__ -= sub_thl1_offset;
    s_awqnbca1_offset = *kts;
    s_awqnbca1__ -= s_awqnbca1_offset;
    s_awqnifa1_offset = *kts;
    s_awqnifa1__ -= s_awqnifa1_offset;
    s_awqnwfa1_offset = *kts;
    s_awqnwfa1__ -= s_awqnwfa1_offset;
    s_awqni1_offset = *kts;
    s_awqni1__ -= s_awqni1_offset;
    s_awqnc1_offset = *kts;
    s_awqnc1__ -= s_awqnc1_offset;
    s_awqke1_offset = *kts;
    s_awqke1__ -= s_awqke1_offset;
    s_awv1_offset = *kts;
    s_awv1__ -= s_awv1_offset;
    s_awu1_offset = *kts;
    s_awu1__ -= s_awu1_offset;
    s_awqc1_offset = *kts;
    s_awqc1__ -= s_awqc1_offset;
    s_awqv1_offset = *kts;
    s_awqv1__ -= s_awqv1_offset;
    s_awqt1_offset = *kts;
    s_awqt1__ -= s_awqt1_offset;
    s_awthl1_offset = *kts;
    s_awthl1__ -= s_awthl1_offset;
    s_aw1_offset = *kts;
    s_aw1__ -= s_aw1_offset;
    edmf_qc1_offset = *kts;
    edmf_qc1__ -= edmf_qc1_offset;
    edmf_ent1_offset = *kts;
    edmf_ent1__ -= edmf_ent1_offset;
    edmf_thl1_offset = *kts;
    edmf_thl1__ -= edmf_thl1_offset;
    edmf_qt1_offset = *kts;
    edmf_qt1__ -= edmf_qt1_offset;
    edmf_w1_offset = *kts;
    edmf_w1__ -= edmf_w1_offset;
    edmf_a1_offset = *kts;
    edmf_a1__ -= edmf_a1_offset;
    sgm1_offset = *kts;
    sgm1 -= sgm1_offset;
    vq1_offset = *kts;
    vq1 -= vq1_offset;
    vt1_offset = *kts;
    vt1 -= vt1_offset;
    ex1_offset = *kts;
    ex1 -= ex1_offset;
    qnbca1_offset = *kts;
    qnbca1 -= qnbca1_offset;
    qnifa1_offset = *kts;
    qnifa1 -= qnifa1_offset;
    qnwfa1_offset = *kts;
    qnwfa1 -= qnwfa1_offset;
    qni1_offset = *kts;
    qni1 -= qni1_offset;
    qnc1_offset = *kts;
    qnc1 -= qnc1_offset;
    qke1_offset = *kts;
    qke1 -= qke1_offset;
    qc1_offset = *kts;
    qc1 -= qc1_offset;
    qv1_offset = *kts;
    qv1 -= qv1_offset;
    qt1_offset = *kts;
    qt1 -= qt1_offset;
    tk1_offset = *kts;
    tk1 -= tk1_offset;
    thv1_offset = *kts;
    thv1 -= thv1_offset;
    thl1_offset = *kts;
    thl1 -= thl1_offset;
    th1_offset = *kts;
    th1 -= th1_offset;
    w1_offset = *kts;
    w1 -= w1_offset;
    v1_offset = *kts;
    v1 -= v1_offset;
    u1_offset = *kts;
    u1 -= u1_offset;
    rho1_offset = *kts;
    rho1 -= rho1_offset;
    p1_offset = *kts;
    p1 -= p1_offset;
    dz1_offset = *kts;
    dz1 -= dz1_offset;
    zw1_offset = *kts;
    zw1 -= zw1_offset;
    s_awchem1_dim1 = *kte + 1 - *kts + 1;
    s_awchem1_offset = *kts + s_awchem1_dim1;
    s_awchem1__ -= s_awchem1_offset;
    chem1_dim1 = *kte - *kts + 1;
    chem1_offset = *kts + chem1_dim1;
    chem1 -= chem1_offset;

    if constexpr(M471_PHASE<0 || M471_PHASE==2){
    /* Function Body */
    wa = rc0000 / rc0001;
    wb = rc0002;
    wc = rc0003;
    atot = rc0004;
    lmax = rc0005;
    lmin = rc0006;
    dlmin = rc0007;
    dcut = rc0008;
    cf_thresh__ = rc0009;
    fluxportion = rc0010;
    cdet = rc0011 / rc0012;
    dzpmax = rc0006;
    csub = rc0013;
    pgfac = rc0014;
    *ktop = 0;
    *ztop = zero;
    *maxmf = zero;
    *maxwidth = zero;
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upw[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upthl[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upthv[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upqt[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upu[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upv[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upqc[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (FALSE_) { upqv[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*tke_opt__ > 0) { upqke[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*scalar_opt__ > 0) { upqnc[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*scalar_opt__ > 0) { upqni[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*scalar_opt__ > 0) { upqnwfa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*scalar_opt__ > 0) { upqnifa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*scalar_opt__ > 0) { upqnbca[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
    if (*mix_chem__) {
	i__1 = *nchem;
	
#pragma unroll 1
for (ixarr3 = 1; ixarr3 <= i__1; ++ixarr3) {
	    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
		i__2 = *kte + 1 - *kts + 1;
		
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
		    upchem[*kts + ixarr1 - 1 + (ixarr2 + (ixarr3 << 3)) * (1*(MYNN471_NZ-49)+50) 
			    - (9*(MYNN471_NZ-49)+451)] = zero;
		}
	    }
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    ent[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] = rc0015;
	}
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_a1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_w1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_qt1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_thl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_ent1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_qc1__[*kts + ixarr1 - 1] = zero;
    }
    if (*mix_chem__) {
	i__1 = *nchem;
	
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= i__1; ++ixarr2) {
	    i__2 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
		edmf_chem__[*kts + ixarr1 - 1 + ixarr2 * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] = zero;
	    }
	}
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_aw1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awthl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqt1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqv1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqc1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awu1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awv1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqke1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqnc1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqni1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqnwfa1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqnifa1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqnbca1__[*kts + ixarr1 - 1] = zero;
    }
    if (*mix_chem__) {
	i__1 = *nchem;
	
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= i__1; ++ixarr2) {
	    i__2 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
		s_awchem1__[*kts + ixarr1 - 1 + ixarr2 * s_awchem1_dim1] = 
			zero;
	    }
	}
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_thl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_sqv1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_u1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_v1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_thl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_sqv1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_sqc1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_u1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_v1__[*kts + ixarr1 - 1] = zero;
    }
    nup2 = 8.f;
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	tkeprod_up__[*kts + ixarr1 - 1] = zero;
    }
    if (FALSE_) {
    }
    maxw = zero;
    cloud_base__ = rc0016;
    i__1 = *kte - 1;
    
#pragma unroll 1
for (k = 1; k <= i__1; ++k) {
	zagl = zw1[k] + half * dz1[k];
	if (zagl > *pblh + rc0017) {
	    goto L9005;
	}
	wpbl = w1[k];
	if (w1[k] < zero) {
	    wpbl = rc0000 * w1[k];
	}
/* Computing MAX */
	r__1 = maxw, r__2 = abs(wpbl);
	maxw = max(r__1,r__2);
	if (zagl <= rc0018) {
	    k50 = k;
	}
/* Computing MAX */
	r__1 = qc1[k], r__2 = qc_bl1__[k];
	qc_sgs__ = max(r__1,r__2);
	if (qc_sgs__ > rc0019 && cldfra_bl1__[k] >= rc0009 && cloud_base__ == 
		rc0016) {
	    cloud_base__ = zw1[k];
	}
/* L9004: */
    }
L9005:
/* Computing MAX */
    r__1 = zero, r__2 = maxw - one;
    maxw = max(r__1,r__2);
/* Computing MAX */
    r__1 = zero, r__2 = one - maxw;
    psig_w__ = max(r__1,r__2);
    psig_w__ = min(psig_w__,*psig_shcu__);
    fltv2 = *fltv;
    if (psig_w__ == zero && *fltv > zero) {
	fltv2 = -rc0011 * *fltv;
    }
    if (FALSE_) {
    }
    superadiabatic = FALSE_;
    if (*landsea - rc0003 >= zero) {
	hux = -rc0015;
    } else {
	hux = -rc0020;
    }
    tvs = *ts * (one + p608 * qv1[*kts]);
    i__1 = max(1,k50);
    
#pragma unroll 1
for (k = 1; k <= i__1; ++k) {
	if (k == 1) {
	    dthvdz = (thv1[k] - tvs) / (half * dz1[k]);
	    if (dthvdz < hux) {
		superadiabatic = TRUE_;
	    } else {
		superadiabatic = FALSE_;
		goto L9008;
	    }
	} else {
	    hux = -rc0021;
	    dthvdz = (thv1[k] - thv1[k - 1]) / (rc0009 * (dz1[k] + dz1[k - 1])
		    );
	    if (dthvdz < hux) {
		superadiabatic = TRUE_;
	    } else {
		superadiabatic = FALSE_;
		goto L9008;
	    }
	}
/* L9007: */
    }
L9008:
    if (FALSE_) {
    }
/* Computing MIN */
    r__1 = *dx * dcut;
    maxwidth_dx__ = min(r__1,lmax);
/* Computing MIN */
    r__1 = rc0022 * *pblh;
    maxwidth_pbl__ = min(r__1,lmax);
    if (*landsea - rc0003 < zero) {
/* Computing MIN */
/* Computing MAX */
	r__3 = rc0009 * cloud_base__;
	r__1 = lmax, r__2 = max(r__3,rc0023);
	maxwidth_cld__ = min(r__1,r__2);
    } else {
/* Computing MIN */
/* Computing MAX */
	r__3 = rc0024 * cloud_base__;
	r__1 = lmax, r__2 = max(r__3,rc0023);
	maxwidth_cld__ = min(r__1,r__2);
    }
/* Computing MAX */
/* Computing 2nd power */
    r__2 = u1[*kts];
/* Computing 2nd power */
    r__3 = v1[*kts];
    r__1 = r__2 * r__2 + r__3 * r__3;
    wspd_pbl__ = (real)sqrt((real)(max(r__1,rc0025)));
    if (*landsea - rc0003 < zero) {
/* Computing MAX */
/* Computing MIN */
	r__2 = rc0005 * (rc0026 * (real)tanh((real)((*fltv - rc0027) / rc0028)
		) + rc0029);
	r__1 = min(r__2,rc0005);
	maxwidth_flx__ = max(r__1,zero);
    } else {
/* Computing MAX */
/* Computing MIN */
	r__2 = rc0005 * (rc0026 * (real)tanh((real)((*fltv - rc0030) / rc0031)
		) + rc0029);
	r__1 = min(r__2,rc0005);
	maxwidth_flx__ = max(r__1,zero);
    }
    *maxwidth = min(maxwidth_dx__,maxwidth_pbl__);
    *maxwidth = min(*maxwidth,maxwidth_cld__);
    *maxwidth = min(*maxwidth,maxwidth_flx__);
    minwidth = lmin;
    if (FALSE_) {
    }
    if (*maxwidth <= minwidth) {
	nup2 = 0.f;
	*maxwidth = zero;
    }
    if (fltv2 > rc0002 && *maxwidth > minwidth && superadiabatic) {
	cn = zero;
	d__ = -rc0032;
	dl = (*maxwidth - minwidth) / 7.f;
	
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
	    l = minwidth + dl * (real) (ip - 1);
	    d__1 =  l;
	    d__2 =  d__;
	    cn += (real)pow_dd(&d__1, &d__2) * (l * l) / (*dx * *dx) * dl;
/* L9010: */
	}
/* L9011: */
	c__ = atot / cn;
	if (*landsea - rc0003 < zero) {
	    acfac = half * (real)tanh((real)((fltv2 - rc0031) / rc0033)) + 
		    half;
	} else {
	    acfac = half * (real)tanh((real)((fltv2 - rc0034) / rc0035)) + 
		    half;
	}
	if (wspd_pbl__ <= rc0036) {
	    ac_wsp__ = one;
	} else {
/* Computing MIN */
/* Computing MAX */
	    r__2 = wspd_pbl__ - rc0037;
	    r__1 = max(r__2,zero) / rc0036;
	    ac_wsp__ = one - min(r__1,one);
	}
	acfac = min(acfac,ac_wsp__);
	an2 = rc0007;
	
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
	    l = minwidth + dl * (real) (ip - 1);
	    d__1 =  l;
	    d__2 =  d__;
	    n = c__ * (real)pow_dd(&d__1, &d__2);
	    upa[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = n * l * l / (*dx * *dx) * dl;
	    upa[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] *= acfac;
	    an2 += upa[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)];
/* L9013: */
	}
/* L9014: */
	if (FALSE_) {
	}
	z0 = rc0018;
	pwmin = rc0038;
	pwmax = rc0039;
/* Computing MAX */
	d__1 =  (gtr * fltv2 * *pblh);
	d__2 =  onethird;
	r__1 = rc0040, r__2 = (real)pow_dd(&d__1, &d__2);
	wstar = max(r__1,r__2);
	qstar = max(*flq,rc0041) / wstar;
	thstar = *flt / wstar;
	if (*landsea - rc0003 >= zero) {
	    csigma = rc0042;
	} else {
	    csigma = rc0042;
	}
	if (FALSE_) {
	    exc_fac__ = zero;
	} else {
	    if (*landsea - rc0003 >= zero) {
		exc_fac__ = rc0043 * rc0044;
	    } else {
		exc_fac__ = rc0043;
	    }
	}
	exc_fac__ *= ac_wsp__;
	d__1 =  (z0 / *pblh);
	d__2 =  onethird;
	sigmaw = csigma * wstar * (real)pow_dd(&d__1, &d__2) * (one - rc0045 *
		 z0 / *pblh);
	d__1 =  (z0 / *pblh);
	d__2 =  onethird;
	sigmaqt = csigma * qstar * (real)pow_dd(&d__1, &d__2);
	d__1 =  (z0 / *pblh);
	d__2 =  onethird;
	sigmath = csigma * thstar * (real)pow_dd(&d__1, &d__2);
/* Computing MIN */
	r__1 = sigmaw * pwmin;
	wmin = min(r__1,rc0038);
/* Computing MIN */
	r__1 = sigmaw * pwmax;
	wmax = min(r__1,rc0009);
	if (FALSE_) {
	}
	
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
	    wlv = wmin + (wmax - wmin) / nup2 * (real) (ip - 1);
	    upw[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = wmin + (real) ip / 8.f * (wmax - wmin);
	    upu[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (u1[*kts] * dz1[*kts + 1] + u1[*kts + 1] * 
		    dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]);
	    upv[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (v1[*kts] * dz1[*kts + 1] + v1[*kts + 1] * 
		    dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]);
	    upqc[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = zero;
	    exc_heat__ = exc_fac__ * upw[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] * sigmath / sigmaw;
	    upthv[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (thv1[*kts] * dz1[*kts + 1] + thv1[*kts + 1]
		     * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]) + exc_heat__;
	    upthl[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (thl1[*kts] * dz1[*kts + 1] + thl1[*kts + 1]
		     * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]) + exc_heat__;
	    exc_moist__ = exc_fac__ * upw[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] * sigmaqt / sigmaw;
	    upqt[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (qt1[*kts] * dz1[*kts + 1] + qt1[*kts + 1] * 
		    dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]) + exc_moist__;
	    if (*tke_opt__ > 0) { upqke[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (qke1[*kts] * dz1[*kts + 1] + qke1[*kts + 1]
		     * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    if (*scalar_opt__ > 0) { upqnc[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (qnc1[*kts] * dz1[*kts + 1] + qnc1[*kts + 1]
		     * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    if (*scalar_opt__ > 0) { upqni[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (qni1[*kts] * dz1[*kts + 1] + qni1[*kts + 1]
		     * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    if (*scalar_opt__ > 0) { upqnwfa[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (qnwfa1[*kts] * dz1[*kts + 1] + qnwfa1[*
		    kts + 1] * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    if (*scalar_opt__ > 0) { upqnifa[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (qnifa1[*kts] * dz1[*kts + 1] + qnifa1[*
		    kts + 1] * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    if (*scalar_opt__ > 0) { upqnbca[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (qnbca1[*kts] * dz1[*kts + 1] + qnbca1[*
		    kts + 1] * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
/* L9016: */
	}
/* L9017: */
	if (*mix_chem__) {
	    
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
		i__1 = *nchem;
		
#pragma unroll 1
for (ic = 1; ic <= i__1; ++ic) {
		    upchem[(ip + (ic << 3)) * (1*(MYNN471_NZ-49)+50) - (9*(MYNN471_NZ-49)+450)] = (chem1[*kts + ic * 
			    chem1_dim1] * dz1[*kts + 1] + chem1[*kts + 1 + ic 
			    * chem1_dim1] * dz1[*kts]) / (dz1[*kts] + dz1[*
			    kts + 1]);
/* L9022: */
		}
/* L9023: */
/* L9019: */
	    }
/* L9020: */
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    if (FALSE_) { envm_thl__[*kts + ixarr1 - 2] = thl1[*kts + ixarr1 - 1]; }
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    if (FALSE_) { envm_sqv__[*kts + ixarr1 - 2] = qv1[*kts + ixarr1 - 1]; }
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    if (FALSE_) { envm_sqc__[*kts + ixarr1 - 2] = qc1[*kts + ixarr1 - 1]; }
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    if (FALSE_) { envm_u__[*kts + ixarr1 - 2] = u1[*kts + ixarr1 - 1]; }
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    if (FALSE_) { envm_v__[*kts + ixarr1 - 2] = v1[*kts + ixarr1 - 1]; }
	}
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    rhoz[k - 1] = (rho1[k] * dz1[k + 1] + rho1[k + 1] * dz1[k]) / (
		    dz1[k + 1] + dz1[k]);
/* L9025: */
	}
/* L9026: */
	rhoz[*kte - 1] = rho1[*kte];
/* Computing MIN */
/* Computing MAX */
	r__2 = (rc0046 - *dx) / (rc0046 - rc0047);
	r__1 = max(r__2,zero);
	dxsa = one - min(r__1,one);
	
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
	    qcn = zero;
	    overshoot = 0;
	    l = minwidth + dl * (real) (ip - 1);
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
		wmin = rc0048 + l * rc0021;
/* Computing MIN */
/* Computing MAX */
		r__2 = upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		r__1 = max(r__2,wmin);
		ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] = rc0049 / (min(r__1,rc0024) * l);
/* Computing MAX */
		r__1 = ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)];
		ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] = max(r__1,rc0050);
/* Computing MIN */
		r__1 = *pblh + rc0051;
		if (zw1[k] >= min(r__1,rc0052)) {
/* Computing MIN */
		    r__1 = *pblh + rc0051;
		    ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] += (zw1[k] - min(r__1,rc0052)) * 
			    rc0053;
		}
		ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] *= one - pattern_spp_pbl1__[k];
/* Computing MIN */
		r__1 = ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)], r__2 = rc0024 / (zw1[k + 1] - 
			zw1[k]);
		ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] = min(r__1,r__2);
		uk = (u1[k] * dz1[k + 1] + u1[k + 1] * dz1[k]) / (dz1[k + 1] 
			+ dz1[k]);
		ukm1 = (u1[k - 1] * dz1[k] + u1[k] * dz1[k - 1]) / (dz1[k - 1]
			 + dz1[k]);
		vk = (v1[k] * dz1[k + 1] + v1[k + 1] * dz1[k]) / (dz1[k + 1] 
			+ dz1[k]);
		vkm1 = (v1[k - 1] * dz1[k] + v1[k] * dz1[k - 1]) / (dz1[k - 1]
			 + dz1[k]);
		entexp = ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] * (zw1[k + 1] - zw1[k]);
		entexm = entexp * rc0054;
		qtn = upqt[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexp) + qt1[k] 
			* entexp;
		thln = upthl[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexp) + thl1[
			k] * entexp;
		un = upu[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexm) + u1[k] * 
			entexm + dxsa * pgfac * (uk - ukm1);
		vn = upv[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexm) + v1[k] * 
			entexm + dxsa * pgfac * (vk - vkm1);
		if (*tke_opt__ > 0) { qken = upqke[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexp) + qke1[
			k] * entexp; }
		if (*scalar_opt__ > 0) { qncn = upqnc[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexp) + qnc1[
			k] * entexp; }
		if (*scalar_opt__ > 0) { qnin = upqni[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexp) + qni1[
			k] * entexp; }
		if (*scalar_opt__ > 0) { qnwfan = upqnwfa[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexp) + 
			qnwfa1[k] * entexp; }
		if (*scalar_opt__ > 0) { qnifan = upqnifa[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexp) + 
			qnifa1[k] * entexp; }
		if (*scalar_opt__ > 0) { qnbcan = upqnbca[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexp) + 
			qnbca1[k] * entexp; }
		qc_ent__ = qcn;
		qt_ent__ = qtn;
		thl_ent__ = thln;
		if (*mix_chem__) {
		    i__2 = *nchem;
		    
#pragma unroll 1
for (ic = 1; ic <= i__2; ++ic) {
			chemn[ic - 1] = upchem[k - 1 + (ip + (ic << 3)) * (1*(MYNN471_NZ-49)+50) 
				- (9*(MYNN471_NZ-49)+451)] * (rc0011 - entexp) + chem1[k + ic * 
				chem1_dim1] * entexp;
/* L9034: */
		    }
/* L9035: */
		}
		pk = (p1[k] * dz1[k + 1] + p1[k + 1] * dz1[k]) / (dz1[k + 1] 
			+ dz1[k]);
		condensation_edmf__(m471_work,m471_n, &qtn, &thln, &pk, &zw1[k + 1], &thvn, &
			qcn);
		thvk = (thv1[k] * dz1[k + 1] + thv1[k + 1] * dz1[k]) / (dz1[k 
			+ 1] + dz1[k]);
		thvkm1 = (thv1[k - 1] * dz1[k] + thv1[k] * dz1[k - 1]) / (dz1[
			k - 1] + dz1[k]);
		b = grav * (thvn / thvk - one);
		if (b > rc0007) {
		    bcoeff = rc0055;
		} else {
		    bcoeff = rc0056;
		}
		if (upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] < rc0056) {
/* Computing MAX */
		    r__1 = upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
/* Computing MIN */
		    r__2 = zw1[k] - zw1[k - 1];
		    wn = upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + (-rc0000 * ent[k + ip * 
			    (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] * upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + bcoeff * b 
			    / max(r__1,rc0056)) * min(r__2,rc0057);
		} else {
/* Computing MIN */
		    r__1 = zw1[k] - zw1[k - 1];
		    wn = upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + (-rc0000 * ent[k + ip * 
			    (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] * upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + bcoeff * b 
			    / upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) * min(r__1,rc0057);
		}
/* Computing MIN */
		r__1 = rc0058 * (zw1[k] - zw1[k - 1]) / rc0059;
		if (wn > upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + min(r__1,rc0060)) {
/* Computing MIN */
		    r__1 = rc0058 * (zw1[k] - zw1[k - 1]) / rc0059;
		    wn = upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + min(r__1,rc0060);
		}
/* Computing MIN */
		r__1 = rc0058 * (zw1[k] - zw1[k - 1]) / rc0059;
		if (wn < upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] - min(r__1,rc0060)) {
/* Computing MIN */
		    r__1 = rc0058 * (zw1[k] - zw1[k - 1]) / rc0059;
		    wn = upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] - min(r__1,rc0060);
		}
/* Computing MIN */
		r__1 = max(wn,zero);
		wn = min(r__1,rc0061);
		if (k == *kts + 1 && wn == zero) {
		    nup2 = 0.f;
		    goto L9032;
		}
		if (FALSE_) {
		    if (wn >= rc0061) {
		    }
		}
		if (wn <= zero && overshoot == 0) {
		    overshoot = 1;
		    if (thvk - thvkm1 > zero) {
			bvf = (real)sqrt((real)(gtr * (thvk - thvkm1) / dz1[k]
				));
			frz = upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] / (bvf * dz1[k]);
/* Computing MAX */
			r__1 = min(frz,one);
			dzp = dz1[k] * max(r__1,zero);
		    }
		} else {
		    dzp = dz1[k];
		}
/* Computing MIN */
		r__1 = upa[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] / (rc0011 - upa[k - 1 + ip * 
			(1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]);
		aratio = min(r__1,rc0009);
		detturb = rc0062;
/* Computing MAX */
		r__1 = one, r__2 = rc0009 * (wn + upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]);
		oow = -rc0063 / max(r__1,r__2);
/* Computing MIN */
/* Computing MAX */
		r__2 = oow * (wn - upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) / dz1[k];
		r__1 = max(r__2,detturb);
		detrate = min(r__1,rc0064);
/* Computing MIN */
/* Computing MAX */
		r__2 = oow * (wn - upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) / dz1[k];
		r__1 = max(r__2,detturb);
		detrateuv = min(r__1,rc0065);
		if (FALSE_) { envm_thl__[k - 1] += (rc0009 * (thl_ent__ + upthl[k - 1 + ip *
			 (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) - thl1[k]) * detrate * aratio * min(dzp,
			dzpmax); }
/* Computing MAX */
		r__1 = qt_ent__ - qc_ent__;
/* Computing MAX */
		r__2 = upqt[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] - upqc[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]
			;
		qv_ent__ = rc0009 * (max(r__1,rc0007) + max(r__2,rc0007));
		if (FALSE_) { envm_sqv__[k - 1] += (qv_ent__ - qv1[k]) * detrate * aratio * 
			min(dzp,dzpmax); }
		if (upqc[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] > rc0066) {
		    if (qc1[k] > rc0067) {
			qc_grid__ = qc1[k];
		    } else {
			qc_grid__ = qc_bl1__[k];
		    }
/* Computing MAX */
		    r__1 = upa[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * rc0009 * (qcn + upqc[k 
			    - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) - qc_grid__;
		    if (FALSE_) { envm_sqc__[k - 1] += max(r__1,zero) * detrate * aratio * 
			    min(dzp,dzpmax); }
		}
		if (FALSE_) { envm_u__[k - 1] += (rc0009 * (un + upu[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) 
			- u1[k]) * detrateuv * aratio * min(dzp,dzpmax); }
		if (FALSE_) { envm_v__[k - 1] += (rc0009 * (vn + upv[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) 
			- v1[k]) * detrateuv * aratio * min(dzp,dzpmax); }
		if (wn > rc0007) {
		    upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = wn;
		    upthv[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = thvn;
		    upthl[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = thln;
		    upqt[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qtn;
		    upqc[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qcn;
		    upu[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = un;
		    upv[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = vn;
		    if (*tke_opt__ > 0) { upqke[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qken; }
		    if (*scalar_opt__ > 0) { upqnc[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qncn; }
		    if (*scalar_opt__ > 0) { upqni[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qnin; }
		    if (*scalar_opt__ > 0) { upqnwfa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qnwfan; }
		    if (*scalar_opt__ > 0) { upqnifa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qnifan; }
		    if (*scalar_opt__ > 0) { upqnbca[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qnbcan; }
		    upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = upa[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		    if (*mix_chem__) {
			i__2 = *nchem;
			
#pragma unroll 1
for (ic = 1; ic <= i__2; ++ic) {
			    upchem[k + (ip + (ic << 3)) * (1*(MYNN471_NZ-49)+50) - (9*(MYNN471_NZ-49)+451)] = chemn[
				    ic - 1];
/* L9037: */
			}
/* L9038: */
		    }
		    *ktop = max(*ktop,k);
		} else {
		    if (FALSE_) {
		    }
		    goto L9032;
		}
/* L9031: */
	    }
L9032:
	    if (FALSE_) {
		red001 = -rc0068;
		
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
/* Computing MAX */
		    r__1 = red001, r__2 = upw[ixarr1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		    red001 = max(r__1,r__2);
		}
		red002 = rc0068;
		
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
/* Computing MIN */
		    r__1 = red002, r__2 = upa[ixarr1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		    red002 = min(r__1,r__2);
		}
		red003 = -rc0068;
		
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
/* Computing MAX */
		    r__1 = red003, r__2 = upa[ixarr1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		    red003 = max(r__1,r__2);
		}
		if (red001 > rc0069 || red002 < zero || red003 > atot || nup2 
			> 10.f) {
		}
	    }
/* L9028: */
	}
/* L9029: */
    } else {
	nup2 = rc0007;
    }
/* Computing MIN */
    i__1 = *ktop, i__2 = *kte - 1;
    *ktop = min(i__1,i__2);
    if (*ktop == 0) {
	*ztop = zero;
    } else {
	*ztop = zw1[*ktop];
    }
    if (nup2 > 0.f) {
	
#pragma unroll 1
for(k=*kts;k<=*kte-1;++k){
real m471_fold0=s_aw1__[k + 1];
real m471_fold1=s_awthl1__[k + 1];
real m471_fold2=s_awqt1__[k + 1];
real m471_fold3=s_awqc1__[k + 1];
real m471_fold4=s_awqv1__[k + 1];
#pragma unroll 1
for(ip=1;ip<=8;++ip){

		m471_fold0 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[k 
			+ ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * psig_w__;
		m471_fold1 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upthl[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			psig_w__;
		m471_fold2 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[
			k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqt[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * psig_w__;
		qc_plume__ = upqc[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		m471_fold3 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[
			k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * qc_plume__ * psig_w__;
		m471_fold4 = m471_fold2 - m471_fold3;
/* L9043: */
	    
}
s_aw1__[k + 1]=m471_fold0;
s_awthl1__[k + 1]=m471_fold1;
s_awqt1__[k + 1]=m471_fold2;
s_awqc1__[k + 1]=m471_fold3;
s_awqv1__[k + 1]=m471_fold4;
}
/* L9041: */
	if (*momentum_opt__ > 0) {
	    
#pragma unroll 1
for(k=*kts;k<=*kte-1;++k){
real m471_fold0=s_awu1__[k + 1];
real m471_fold1=s_awv1__[k + 1];
#pragma unroll 1
for(ip=1;ip<=8;++ip){

		    m471_fold0 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upu[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    psig_w__;
		    m471_fold1 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upv[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    psig_w__;
/* L9049: */
		
}
s_awu1__[k + 1]=m471_fold0;
s_awv1__[k + 1]=m471_fold1;
}
/* L9047: */
	}
	if (*tke_opt__ > 0) {
	    
#pragma unroll 1
for(k=*kts;k<=*kte-1;++k){
real m471_fold0=s_awqke1__[k + 1];
#pragma unroll 1
for(ip=1;ip<=8;++ip){

		    m471_fold0 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] *
			     upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqke[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] *
			     psig_w__;
/* L9055: */
		
}
s_awqke1__[k + 1]=m471_fold0;
}
/* L9053: */
	}
	if (*mix_chem__) {
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
		    i__2 = *nchem;
		    
#pragma unroll 1
for (ic = 1; ic <= i__2; ++ic) {
			s_awchem1__[k + 1 + ic * s_awchem1_dim1] += rhoz[k - 
				1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[k + ip * (1*(MYNN471_NZ-49)+50) 
				- (1*(MYNN471_NZ-49)+51)] * upchem[k + (ip + (ic << 3)) * (1*(MYNN471_NZ-49)+50) - 
				(9*(MYNN471_NZ-49)+451)] * psig_w__;
/* L9064: */
		    }
/* L9065: */
/* L9061: */
		}
/* L9062: */
/* L9058: */
	    }
/* L9059: */
	}
	if (*scalar_opt__ > 0) {
	    
#pragma unroll 1
for(k=*kts;k<=*kte-1;++k){
real m471_fold0=s_awqnc1__[k + 1];
real m471_fold1=s_awqni1__[k + 1];
real m471_fold2=s_awqnwfa1__[k + 1];
real m471_fold3=s_awqnifa1__[k + 1];
real m471_fold4=s_awqnbca1__[k + 1];
#pragma unroll 1
for(ip=1;ip<=8;++ip){

		    m471_fold0 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] *
			     upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqnc[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] *
			     psig_w__;
		    m471_fold1 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] *
			     upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqni[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] *
			     psig_w__;
		    m471_fold2 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]
			     * upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqnwfa[k + ip * (1*(MYNN471_NZ-49)+50) - 
			    (1*(MYNN471_NZ-49)+51)] * psig_w__;
		    m471_fold3 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]
			     * upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqnifa[k + ip * (1*(MYNN471_NZ-49)+50) - 
			    (1*(MYNN471_NZ-49)+51)] * psig_w__;
		    m471_fold4 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]
			     * upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqnbca[k + ip * (1*(MYNN471_NZ-49)+50) - 
			    (1*(MYNN471_NZ-49)+51)] * psig_w__;
/* L9070: */
		
}
s_awqnc1__[k + 1]=m471_fold0;
s_awqni1__[k + 1]=m471_fold1;
s_awqnwfa1__[k + 1]=m471_fold2;
s_awqnifa1__[k + 1]=m471_fold3;
s_awqnbca1__[k + 1]=m471_fold4;
}
/* L9068: */
	}
	if (s_aw1__[*kts + 1] != rc0007) {
	    dzi[*kts - 1] = half * (dz1[*kts] + dz1[*kts + 1]);
/* Computing MAX */
	    r__1 = s_aw1__[*kts + 1] * (thv1[*kts] - thv1[*kts + 1]) / dzi[*
		    kts - 1];
	    flx1 = max(r__1,rc0070);
	} else {
	    flx1 = zero;
	}
	adjustment = one;
	flt2 = max(*fltv,zero);
	if (flx1 > fluxportion * flt2 / dz1[*kts] && flx1 > zero) {
/* Computing MAX */
	    r__1 = rc0025, r__2 = fluxportion * flt2 / dz1[*kts] / flx1;
	    adjustment = max(r__1,r__2);
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_aw1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awthl1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqt1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqc1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqv1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqnc1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqni1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqnwfa1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqnifa1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqnbca1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    if (*momentum_opt__ > 0) {
		i__1 = *kte + 1 - *kts + 1;
		
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		    s_awu1__[*kts + ixarr1 - 1] *= adjustment;
		}
		i__1 = *kte + 1 - *kts + 1;
		
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		    s_awv1__[*kts + ixarr1 - 1] *= adjustment;
		}
	    }
	    if (*tke_opt__ > 0) {
		i__1 = *kte + 1 - *kts + 1;
		
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		    s_awqke1__[*kts + ixarr1 - 1] *= adjustment;
		}
	    }
	    if (*mix_chem__) {
		i__1 = *nchem;
		
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= i__1; ++ixarr2) {
		    i__2 = *kte + 1 - *kts + 1;
		    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
			s_awchem1__[*kts + ixarr1 - 1 + ixarr2 * 
				s_awchem1_dim1] *= adjustment;
		    }
		}
	    }
	    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
		
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
		    upa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] *= adjustment;
		}
	    }
	}
	if (FALSE_) {
	}
	
#pragma unroll 1
for(k=*kts;k<=*kte-1;++k){
real m471_fold0=edmf_a1__[k];
real m471_fold1=edmf_w1__[k];
real m471_fold2=edmf_qt1__[k];
real m471_fold3=edmf_thl1__[k];
real m471_fold4=edmf_ent1__[k];
real m471_fold5=edmf_qc1__[k];
#pragma unroll 1
for(ip=1;ip<=8;++ip){

		m471_fold0 += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		m471_fold1 += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		m471_fold2 += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqt[k + ip * (1*(MYNN471_NZ-49)+50) - 
			(1*(MYNN471_NZ-49)+51)];
		m471_fold3 += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upthl[k + ip * (1*(MYNN471_NZ-49)+50) - 
			(1*(MYNN471_NZ-49)+51)];
		m471_fold4 += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * ent[k + ip * (1*(MYNN471_NZ-49)+49) - 
			(1*(MYNN471_NZ-49)+50)];
		m471_fold5 += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqc[k + ip * (1*(MYNN471_NZ-49)+50) - 
			(1*(MYNN471_NZ-49)+51)];
/* L9076: */
	    
}
edmf_a1__[k]=m471_fold0;
edmf_w1__[k]=m471_fold1;
edmf_qt1__[k]=m471_fold2;
edmf_thl1__[k]=m471_fold3;
edmf_ent1__[k]=m471_fold4;
edmf_qc1__[k]=m471_fold5;
}
/* L9074: */
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    if (edmf_a1__[k] > rc0007) {
		edmf_w1__[k] /= edmf_a1__[k];
		edmf_qt1__[k] /= edmf_a1__[k];
		edmf_thl1__[k] /= edmf_a1__[k];
		edmf_ent1__[k] /= edmf_a1__[k];
		edmf_qc1__[k] /= edmf_a1__[k];
		edmf_a1__[k] *= psig_w__;
		if (edmf_a1__[k] * edmf_w1__[k] > *maxmf) {
		    *maxmf = edmf_a1__[k] * edmf_w1__[k];
		}
	    }
/* Computing 3rd power */
	    r__2 = (r__1 = edmf_w1__[k], abs(r__1));
/* Computing MAX */
	    r__3 = el1[k];
	    tkeprod_up__[k] = r__2 * (r__2 * r__2) * edmf_a1__[k] / (b1 * max(
		    r__3,rc0038));
/* L9079: */
	}
/* L9080: */
	if (*mix_chem__) {
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
		    i__2 = *nchem;
		    
#pragma unroll 1
for (ic = 1; ic <= i__2; ++ic) {
			edmf_chem__[k + ic * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]
				 * upchem[k + (ip + (ic << 3)) * (1*(MYNN471_NZ-49)+50) - (9*(MYNN471_NZ-49)+451)];
/* L9088: */
		    }
/* L9089: */
/* L9085: */
		}
/* L9086: */
/* L9082: */
	    }
/* L9083: */
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		if (edmf_a1__[k] > rc0007) {
		    i__2 = *nchem;
		    
#pragma unroll 1
for (ic = 1; ic <= i__2; ++ic) {
			edmf_chem__[k + ic * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] /= edmf_a1__[k];
/* L9094: */
		    }
/* L9095: */
		}
/* L9091: */
	    }
/* L9092: */
	}
	if (FALSE_) {
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
		envi_w__[k - 1] = onethird * (edmf_w1__[k - 1] + edmf_w1__[k] 
			+ edmf_w1__[k + 1]);
		envi_a__[k - 1] = onethird * (edmf_a1__[k - 1] + edmf_a1__[k] 
			+ edmf_a1__[k + 1]) * adjustment;
/* L9097: */
	    }
/* L9098: */
	    envi_w__[*kts - 1] = edmf_w1__[*kts];
	    envi_a__[*kts - 1] = edmf_a1__[*kts];
	    envi_w__[*kte - 1] = zero;
	    envi_a__[*kte - 1] = edmf_a1__[*kte];
	    envi_w__[*kte] = zero;
	    envi_a__[*kte] = edmf_a1__[*kte];
	    if (envi_w__[*kts - 1] > rc0024 * dz1[*kts] / *dt) {
		sublim = rc0024 * dz1[*kts] / *dt / envi_w__[*kts - 1];
	    } else {
		sublim = one;
	    }
	    i__1 = *kte;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		temp = envi_a__[k - 1];
		envi_a__[k - 1] = one - temp;
		envi_w__[k - 1] = csub * sublim * envi_w__[k - 1] * temp / (
			rc0011 - temp);
/* L9100: */
	    }
/* L9101: */
	    dzi[*kts - 1] = rc0009 * (dz1[*kts] + dz1[*kts + 1]);
	    sub_thl1__[*kts] = rc0009 * envi_w__[*kts - 1] * envi_a__[*kts - 
		    1] * (rho1[*kts + 1] * thl1[*kts + 1] - rho1[*kts] * thl1[
		    *kts]) / dzi[*kts - 1] / rhoz[*kts - 1];
	    sub_sqv1__[*kts] = rc0009 * envi_w__[*kts - 1] * envi_a__[*kts - 
		    1] * (rho1[*kts + 1] * qv1[*kts + 1] - rho1[*kts] * qv1[*
		    kts]) / dzi[*kts - 1] / rhoz[*kts - 1];
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
		dzi[k - 1] = rc0009 * (dz1[k] + dz1[k + 1]);
		sub_thl1__[k] = rc0009 * (envi_w__[k - 1] + envi_w__[k - 2]) *
			 rc0009 * (envi_a__[k - 1] + envi_a__[k - 2]) * (rho1[
			k + 1] * thl1[k + 1] - rho1[k] * thl1[k]) / dzi[k - 1]
			 / rhoz[k - 1];
		sub_sqv1__[k] = rc0009 * (envi_w__[k - 1] + envi_w__[k - 2]) *
			 rc0009 * (envi_a__[k - 1] + envi_a__[k - 2]) * (rho1[
			k + 1] * qv1[k + 1] - rho1[k] * qv1[k]) / dzi[k - 1] /
			 rhoz[k - 1];
/* L9103: */
	    }
/* L9104: */
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		det_thl1__[k] = cdet * (envm_thl__[k - 1] - thl1[k]) * 
			envi_a__[k - 1] * psig_w__;
		det_sqv1__[k] = cdet * (envm_sqv__[k - 1] - qv1[k]) * 
			envi_a__[k - 1] * psig_w__;
		det_sqc1__[k] = cdet * (envm_sqc__[k - 1] - qc1[k]) * 
			envi_a__[k - 1] * psig_w__;
/* L9106: */
	    }
/* L9107: */
	    if (*momentum_opt__ > 0) {
		sub_u1__[*kts] = rc0009 * envi_w__[*kts - 1] * envi_a__[*kts 
			- 1] * (rho1[*kts + 1] * u1[*kts + 1] - rho1[*kts] * 
			u1[*kts]) / dzi[*kts - 1] / rhoz[*kts - 1];
		sub_v1__[*kts] = rc0009 * envi_w__[*kts - 1] * envi_a__[*kts 
			- 1] * (rho1[*kts + 1] * v1[*kts + 1] - rho1[*kts] * 
			v1[*kts]) / dzi[*kts - 1] / rhoz[*kts - 1];
		i__1 = *kte - 1;
		
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
		    sub_u1__[k] = rc0009 * (envi_w__[k - 1] + envi_w__[k - 2])
			     * rc0009 * (envi_a__[k - 1] + envi_a__[k - 2]) * 
			    (rho1[k + 1] * u1[k + 1] - rho1[k] * u1[k]) / dzi[
			    k - 1] / rhoz[k - 1];
		    sub_v1__[k] = rc0009 * (envi_w__[k - 1] + envi_w__[k - 2])
			     * rc0009 * (envi_a__[k - 1] + envi_a__[k - 2]) * 
			    (rho1[k + 1] * v1[k + 1] - rho1[k] * v1[k]) / dzi[
			    k - 1] / rhoz[k - 1];
/* L9109: */
		}
/* L9110: */
		i__1 = *kte - 1;
		
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		    det_u1__[k] = cdet * (envm_u__[k - 1] - u1[k]) * envi_a__[
			    k - 1] * psig_w__;
		    det_v1__[k] = cdet * (envm_v__[k - 1] - v1[k]) * envi_a__[
			    k - 1] * psig_w__;
/* L9112: */
		}
/* L9113: */
	    }
	}
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    if (FALSE_) { exneri[k - 1] = (ex1[k] * dz1[k + 1] + ex1[k + 1] * dz1[k]) / (
		    dz1[k + 1] + dz1[k]); }
	    edmf_th1__[k - 1] = edmf_thl1__[k] + xlvcp / ex1[k] * edmf_qc1__[
		    k];
	    dzi[k - 1] = rc0009 * (dz1[k] + dz1[k + 1]);
/* L9115: */
	}
/* L9116: */
	i__1 = *kte - 2;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    if (k > *ktop) {
		goto L9119;
	    }
	    if (rc0009 * (edmf_qc1__[k] + edmf_qc1__[k - 1]) > zero && 
		    cldfra_bl1__[k] < cf_thresh__) {
		aup = (edmf_a1__[k] * dzi[k - 2] + edmf_a1__[k - 1] * dzi[k - 
			1]) / (dzi[k - 2] + dzi[k - 1]);
		thp = (edmf_th1__[k - 1] * dzi[k - 2] + edmf_th1__[k - 2] * 
			dzi[k - 1]) / (dzi[k - 2] + dzi[k - 1]);
		qtp = (edmf_qt1__[k] * dzi[k - 2] + edmf_qt1__[k - 1] * dzi[k 
			- 1]) / (dzi[k - 2] + dzi[k - 1]);
		esat = (real)esat_blend__(m471_work,m471_n, &tk1[k]);
/* Computing MAX */
		r__1 = rc0071, r__2 = p1[k] - ep_3__ * esat;
		qsl = ep_2__ * esat / max(r__1,r__2);
		if (edmf_qc1__[k] > zero && edmf_qc1__[k - 1] > zero) {
		    qcp = (edmf_qc1__[k] * dzi[k - 2] + edmf_qc1__[k - 1] * 
			    dzi[k - 1]) / (dzi[k - 2] + dzi[k - 1]);
		} else {
/* Computing MAX */
		    r__1 = edmf_qc1__[k], r__2 = edmf_qc1__[k - 1];
		    qcp = max(r__1,r__2);
		}
		xl = (real)xl_blend__(m471_work,m471_n, &tk1[k]);
		qsat_tk__ = (real)qsat_blend__(m471_work,m471_n, &tk1[k], &p1[k]);
/* Computing 2nd power */
		r__1 = tk1[k];
		rsl = xl * qsat_tk__ / (r_v__ * (r__1 * r__1));
		cpm = cp + qt1[k] * cpv;
		a = rc0011 / (rc0011 + xl * rsl / cpm);
		b9 = a * rsl;
		q2p = xlvcp / ex1[k];
		pt = thl1[k] + q2p * qcp * aup;
		bb = b9 * tk1[k] / pt;
		qww = rc0011 + rc0072 * qt1[k];
		alpha = rc0072 * pt;
		beta = pt * xl / (tk1[k] * cp) - rc0073 * pt;
		if (a > zero) {
/* Computing MIN */
		    r__1 = rc0074 / a;
		    f = min(r__1,rc0044);
		} else {
		    f = one;
		}
		sigq = rc0036 * aup * (qtp - qt1[k]);
/* Computing MAX */
		r__1 = sigq, r__2 = qsat_tk__ * rc0031;
		sigq = max(r__1,r__2);
/* Computing MIN */
		r__1 = sigq, r__2 = qsat_tk__ * rc0013;
		sigq = min(r__1,r__2);
		qmq = a * (qt1[k] - qsat_tk__);
		q1 = qmq / sigq;
		if (*landsea - rc0003 >= zero) {
/* Computing MIN */
/* Computing MAX */
		    r__2 = rc0009 + rc0075 * (real)atan((real)(rc0076 * q1));
		    r__1 = max(r__2,rc0025);
		    mf_cf__ = min(r__1,rc0045);
/* Computing MAX */
		    r__1 = mf_cf__, r__2 = rc0008 * aup;
		    mf_cf__ = max(r__1,r__2);
		} else {
/* Computing MIN */
/* Computing MAX */
		    r__2 = rc0009 + rc0075 * (real)atan((real)(rc0076 * q1));
		    r__1 = max(r__2,rc0025);
		    mf_cf__ = min(r__1,rc0045);
/* Computing MAX */
		    r__1 = mf_cf__, r__2 = rc0077 * aup;
		    mf_cf__ = max(r__1,r__2);
		}
		if (*landsea - rc0003 >= zero) {
		    if (qcp * aup > rc0078) {
			qc_bl1__[k] = rc0079 * (qcp * aup) - rc0080;
		    } else {
			qc_bl1__[k] = rc0081 * (qcp * aup);
		    }
		    cldfra_bl1__[k] = mf_cf__;
		    ac_mf__ = mf_cf__;
		} else {
		    if (qcp * aup > rc0078) {
			qc_bl1__[k] = rc0079 * (qcp * aup) - rc0080;
		    } else {
			qc_bl1__[k] = rc0081 * (qcp * aup);
		    }
		    cldfra_bl1__[k] = mf_cf__;
		    ac_mf__ = mf_cf__;
		}
/* Computing MAX */
		r__1 = q1, r__2 = -rc0082;
		q1 = max(r__1,r__2);
		if (q1 >= one) {
		    fng = one;
		} else if (q1 >= -rc0083 && q1 < one) {
		    fng = (real)exp((real)(-rc0039 * (q1 - one)));
		} else if (q1 >= -rc0084 && q1 < -rc0083) {
		    fng = rc0061 + (real)exp((real)(-rc0085 * (q1 + rc0083)));
		} else {
/* Computing MIN */
		    r__1 = rc0086 + (real)exp((real)(-rc0087 * (q1 + rc0084)))
			    ;
		    fng = min(r__1,rc0088);
		}
		vt1[k] = qww - rc0003 * aup * beta * bb * fng - one;
		vq1[k] = alpha + rc0003 * aup * beta * a * fng - tv0;
	    }
/* L9118: */
	}
L9119:
	;
    }
    if (*ktop > 0) {
	red001 = -rc0068;
	i__1 = *ktop;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
/* Computing MAX */
	    r__1 = red001, r__2 = edmf_qc1__[ixarr1];
	    red001 = max(r__1,r__2);
	}
	maxqc = red001;
	if (maxqc < rc0089) {
	    *maxmf = -rc0011 * *maxmf;
	}
    }
    if (FALSE_) {
    }
    return 0;

    }else{
    if constexpr(M471_PHASE==7){
    /* Function Body */
    wa = rc0000 / rc0001;
    wb = rc0002;
    wc = rc0003;
    atot = rc0004;
    lmax = rc0005;
    lmin = rc0006;
    dlmin = rc0007;
    dcut = rc0008;
    cf_thresh__ = rc0009;
    fluxportion = rc0010;
    cdet = rc0011 / rc0012;
    dzpmax = rc0006;
    csub = rc0013;
    pgfac = rc0014;
    *ktop = 0;
    *ztop = zero;
    *maxmf = zero;
    *maxwidth = zero;
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upw[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upthl[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upthv[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upqt[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upu[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upv[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upqc[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (FALSE_) { upqv[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*tke_opt__ > 0) { upqke[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*scalar_opt__ > 0) { upqnc[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*scalar_opt__ > 0) { upqni[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*scalar_opt__ > 0) { upqnwfa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*scalar_opt__ > 0) { upqnifa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*scalar_opt__ > 0) { upqnbca[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
    if (*mix_chem__) {
	i__1 = *nchem;
	
#pragma unroll 1
for (ixarr3 = 1; ixarr3 <= i__1; ++ixarr3) {
	    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
		i__2 = *kte + 1 - *kts + 1;
		
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
		    upchem[*kts + ixarr1 - 1 + (ixarr2 + (ixarr3 << 3)) * (1*(MYNN471_NZ-49)+50) 
			    - (9*(MYNN471_NZ-49)+451)] = zero;
		}
	    }
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    ent[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] = rc0015;
	}
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_a1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_w1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_qt1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_thl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_ent1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_qc1__[*kts + ixarr1 - 1] = zero;
    }
    if (*mix_chem__) {
	i__1 = *nchem;
	
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= i__1; ++ixarr2) {
	    i__2 = *kte - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
		edmf_chem__[*kts + ixarr1 - 1 + ixarr2 * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] = zero;
	    }
	}
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_aw1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awthl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqt1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqv1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqc1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awu1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awv1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqke1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqnc1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqni1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqnwfa1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqnifa1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqnbca1__[*kts + ixarr1 - 1] = zero;
    }
    if (*mix_chem__) {
	i__1 = *nchem;
	
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= i__1; ++ixarr2) {
	    i__2 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
		s_awchem1__[*kts + ixarr1 - 1 + ixarr2 * s_awchem1_dim1] = 
			zero;
	    }
	}
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_thl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_sqv1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_u1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_v1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_thl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_sqv1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_sqc1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_u1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_v1__[*kts + ixarr1 - 1] = zero;
    }
    nup2 = 8.f;
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	tkeprod_up__[*kts + ixarr1 - 1] = zero;
    }
    if (FALSE_) {
    }
    maxw = zero;
    cloud_base__ = rc0016;
    i__1 = *kte - 1;
    
#pragma unroll 1
for (k = 1; k <= i__1; ++k) {
	zagl = zw1[k] + half * dz1[k];
	if (zagl > *pblh + rc0017) {
	    goto Lsplit9005;
	}
	wpbl = w1[k];
	if (w1[k] < zero) {
	    wpbl = rc0000 * w1[k];
	}
/* Computing MAX */
	r__1 = maxw, r__2 = abs(wpbl);
	maxw = max(r__1,r__2);
	if (zagl <= rc0018) {
	    k50 = k;
	}
/* Computing MAX */
	r__1 = qc1[k], r__2 = qc_bl1__[k];
	qc_sgs__ = max(r__1,r__2);
	if (qc_sgs__ > rc0019 && cldfra_bl1__[k] >= rc0009 && cloud_base__ == 
		rc0016) {
	    cloud_base__ = zw1[k];
	}
/* Lsplit9004: */
    }
Lsplit9005:
/* Computing MAX */
    r__1 = zero, r__2 = maxw - one;
    maxw = max(r__1,r__2);
/* Computing MAX */
    r__1 = zero, r__2 = one - maxw;
    psig_w__ = max(r__1,r__2);
    psig_w__ = min(psig_w__,*psig_shcu__);
    fltv2 = *fltv;
    if (psig_w__ == zero && *fltv > zero) {
	fltv2 = -rc0011 * *fltv;
    }
    if (FALSE_) {
    }
    superadiabatic = FALSE_;
    if (*landsea - rc0003 >= zero) {
	hux = -rc0015;
    } else {
	hux = -rc0020;
    }
    tvs = *ts * (one + p608 * qv1[*kts]);
    i__1 = max(1,k50);
    
#pragma unroll 1
for (k = 1; k <= i__1; ++k) {
	if (k == 1) {
	    dthvdz = (thv1[k] - tvs) / (half * dz1[k]);
	    if (dthvdz < hux) {
		superadiabatic = TRUE_;
	    } else {
		superadiabatic = FALSE_;
		goto Lsplit9008;
	    }
	} else {
	    hux = -rc0021;
	    dthvdz = (thv1[k] - thv1[k - 1]) / (rc0009 * (dz1[k] + dz1[k - 1])
		    );
	    if (dthvdz < hux) {
		superadiabatic = TRUE_;
	    } else {
		superadiabatic = FALSE_;
		goto Lsplit9008;
	    }
	}
/* Lsplit9007: */
    }
Lsplit9008:
    if (FALSE_) {
    }
/* Computing MIN */
    r__1 = *dx * dcut;
    maxwidth_dx__ = min(r__1,lmax);
/* Computing MIN */
    r__1 = rc0022 * *pblh;
    maxwidth_pbl__ = min(r__1,lmax);
    if (*landsea - rc0003 < zero) {
/* Computing MIN */
/* Computing MAX */
	r__3 = rc0009 * cloud_base__;
	r__1 = lmax, r__2 = max(r__3,rc0023);
	maxwidth_cld__ = min(r__1,r__2);
    } else {
/* Computing MIN */
/* Computing MAX */
	r__3 = rc0024 * cloud_base__;
	r__1 = lmax, r__2 = max(r__3,rc0023);
	maxwidth_cld__ = min(r__1,r__2);
    }
/* Computing MAX */
/* Computing 2nd power */
    r__2 = u1[*kts];
/* Computing 2nd power */
    r__3 = v1[*kts];
    r__1 = r__2 * r__2 + r__3 * r__3;
    wspd_pbl__ = (real)sqrt((real)(max(r__1,rc0025)));
    if (*landsea - rc0003 < zero) {
/* Computing MAX */
/* Computing MIN */
	r__2 = rc0005 * (rc0026 * (real)tanh((real)((*fltv - rc0027) / rc0028)
		) + rc0029);
	r__1 = min(r__2,rc0005);
	maxwidth_flx__ = max(r__1,zero);
    } else {
/* Computing MAX */
/* Computing MIN */
	r__2 = rc0005 * (rc0026 * (real)tanh((real)((*fltv - rc0030) / rc0031)
		) + rc0029);
	r__1 = min(r__2,rc0005);
	maxwidth_flx__ = max(r__1,zero);
    }
    *maxwidth = min(maxwidth_dx__,maxwidth_pbl__);
    *maxwidth = min(*maxwidth,maxwidth_cld__);
    *maxwidth = min(*maxwidth,maxwidth_flx__);
    minwidth = lmin;
    if (FALSE_) {
    }
    if (*maxwidth <= minwidth) {
	nup2 = 0.f;
	*maxwidth = zero;
    }
    upqv[0]=minwidth;
    upqv[1]=pgfac;
    upqv[2]=dzpmax;
    upqv[3]=psig_w__;
    upqv[4]=fluxportion;
    upqv[5]=cf_thresh__;
    upqv[6]=atot;
    upqv[7]=cdet;
    upqv[8]=csub;
    upqv[11]=0.f;
    for(int m471_p=0;m471_p<8;++m471_p){upqv[12+m471_p]=0.f;upqv[20+m471_p]=0.f;}
    if (fltv2 > rc0002 && *maxwidth > minwidth && superadiabatic) {
	cn = zero;
	d__ = -rc0032;
	dl = (*maxwidth - minwidth) / 7.f;
	
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
	    l = minwidth + dl * (real) (ip - 1);
	    d__1 =  l;
	    d__2 =  d__;
	    cn += (real)pow_dd(&d__1, &d__2) * (l * l) / (*dx * *dx) * dl;
/* Lsplit9010: */
	}
/* Lsplit9011: */
	c__ = atot / cn;
	if (*landsea - rc0003 < zero) {
	    acfac = half * (real)tanh((real)((fltv2 - rc0031) / rc0033)) + 
		    half;
	} else {
	    acfac = half * (real)tanh((real)((fltv2 - rc0034) / rc0035)) + 
		    half;
	}
	if (wspd_pbl__ <= rc0036) {
	    ac_wsp__ = one;
	} else {
/* Computing MIN */
/* Computing MAX */
	    r__2 = wspd_pbl__ - rc0037;
	    r__1 = max(r__2,zero) / rc0036;
	    ac_wsp__ = one - min(r__1,one);
	}
	acfac = min(acfac,ac_wsp__);
	an2 = rc0007;
	
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
	    l = minwidth + dl * (real) (ip - 1);
	    d__1 =  l;
	    d__2 =  d__;
	    n = c__ * (real)pow_dd(&d__1, &d__2);
	    upa[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = n * l * l / (*dx * *dx) * dl;
	    upa[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] *= acfac;
	    an2 += upa[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)];
/* Lsplit9013: */
	}
/* Lsplit9014: */
	if (FALSE_) {
	}
	z0 = rc0018;
	pwmin = rc0038;
	pwmax = rc0039;
/* Computing MAX */
	d__1 =  (gtr * fltv2 * *pblh);
	d__2 =  onethird;
	r__1 = rc0040, r__2 = (real)pow_dd(&d__1, &d__2);
	wstar = max(r__1,r__2);
	qstar = max(*flq,rc0041) / wstar;
	thstar = *flt / wstar;
	if (*landsea - rc0003 >= zero) {
	    csigma = rc0042;
	} else {
	    csigma = rc0042;
	}
	if (FALSE_) {
	    exc_fac__ = zero;
	} else {
	    if (*landsea - rc0003 >= zero) {
		exc_fac__ = rc0043 * rc0044;
	    } else {
		exc_fac__ = rc0043;
	    }
	}
	exc_fac__ *= ac_wsp__;
	d__1 =  (z0 / *pblh);
	d__2 =  onethird;
	sigmaw = csigma * wstar * (real)pow_dd(&d__1, &d__2) * (one - rc0045 *
		 z0 / *pblh);
	d__1 =  (z0 / *pblh);
	d__2 =  onethird;
	sigmaqt = csigma * qstar * (real)pow_dd(&d__1, &d__2);
	d__1 =  (z0 / *pblh);
	d__2 =  onethird;
	sigmath = csigma * thstar * (real)pow_dd(&d__1, &d__2);
/* Computing MIN */
	r__1 = sigmaw * pwmin;
	wmin = min(r__1,rc0038);
/* Computing MIN */
	r__1 = sigmaw * pwmax;
	wmax = min(r__1,rc0009);
	if (FALSE_) {
	}
	
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
	    wlv = wmin + (wmax - wmin) / nup2 * (real) (ip - 1);
	    upw[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = wmin + (real) ip / 8.f * (wmax - wmin);
	    upu[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (u1[*kts] * dz1[*kts + 1] + u1[*kts + 1] * 
		    dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]);
	    upv[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (v1[*kts] * dz1[*kts + 1] + v1[*kts + 1] * 
		    dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]);
	    upqc[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = zero;
	    exc_heat__ = exc_fac__ * upw[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] * sigmath / sigmaw;
	    upthv[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (thv1[*kts] * dz1[*kts + 1] + thv1[*kts + 1]
		     * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]) + exc_heat__;
	    upthl[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (thl1[*kts] * dz1[*kts + 1] + thl1[*kts + 1]
		     * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]) + exc_heat__;
	    exc_moist__ = exc_fac__ * upw[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] * sigmaqt / sigmaw;
	    upqt[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (qt1[*kts] * dz1[*kts + 1] + qt1[*kts + 1] * 
		    dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]) + exc_moist__;
	    if (*tke_opt__ > 0) { upqke[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (qke1[*kts] * dz1[*kts + 1] + qke1[*kts + 1]
		     * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    if (*scalar_opt__ > 0) { upqnc[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (qnc1[*kts] * dz1[*kts + 1] + qnc1[*kts + 1]
		     * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    if (*scalar_opt__ > 0) { upqni[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (qni1[*kts] * dz1[*kts + 1] + qni1[*kts + 1]
		     * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    if (*scalar_opt__ > 0) { upqnwfa[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (qnwfa1[*kts] * dz1[*kts + 1] + qnwfa1[*
		    kts + 1] * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    if (*scalar_opt__ > 0) { upqnifa[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (qnifa1[*kts] * dz1[*kts + 1] + qnifa1[*
		    kts + 1] * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    if (*scalar_opt__ > 0) { upqnbca[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = (qnbca1[*kts] * dz1[*kts + 1] + qnbca1[*
		    kts + 1] * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
/* Lsplit9016: */
	}
/* Lsplit9017: */
	if (*mix_chem__) {
	    
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
		i__1 = *nchem;
		
#pragma unroll 1
for (ic = 1; ic <= i__1; ++ic) {
		    upchem[(ip + (ic << 3)) * (1*(MYNN471_NZ-49)+50) - (9*(MYNN471_NZ-49)+450)] = (chem1[*kts + ic * 
			    chem1_dim1] * dz1[*kts + 1] + chem1[*kts + 1 + ic 
			    * chem1_dim1] * dz1[*kts]) / (dz1[*kts] + dz1[*
			    kts + 1]);
/* Lsplit9022: */
		}
/* Lsplit9023: */
/* Lsplit9019: */
	    }
/* Lsplit9020: */
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    if (FALSE_) { envm_thl__[*kts + ixarr1 - 2] = thl1[*kts + ixarr1 - 1]; }
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    if (FALSE_) { envm_sqv__[*kts + ixarr1 - 2] = qv1[*kts + ixarr1 - 1]; }
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    if (FALSE_) { envm_sqc__[*kts + ixarr1 - 2] = qc1[*kts + ixarr1 - 1]; }
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    if (FALSE_) { envm_u__[*kts + ixarr1 - 2] = u1[*kts + ixarr1 - 1]; }
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    if (FALSE_) { envm_v__[*kts + ixarr1 - 2] = v1[*kts + ixarr1 - 1]; }
	}
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    rhoz[k - 1] = (rho1[k] * dz1[k + 1] + rho1[k + 1] * dz1[k]) / (
		    dz1[k + 1] + dz1[k]);
/* Lsplit9025: */
	}
/* Lsplit9026: */
	rhoz[*kte - 1] = rho1[*kte];
/* Computing MIN */
/* Computing MAX */
	r__2 = (rc0046 - *dx) / (rc0046 - rc0047);
	r__1 = max(r__2,zero);
	dxsa = one - min(r__1,one);
	

    upqv[9]=dl;upqv[10]=dxsa;upqv[11]=1.f;
    }

    }
    if constexpr(M471_PHASE==8){
    minwidth=upqv[0];
    pgfac=upqv[1];
    dzpmax=upqv[2];
    psig_w__=upqv[3];
    fluxportion=upqv[4];
    cf_thresh__=upqv[5];
    atot=upqv[6];
    cdet=upqv[7];
    csub=upqv[8];

    if(upqv[11]>0.f){
    dl=upqv[9];dxsa=upqv[10];
    integer m471_local_top=0;ktop=M471Ptr<integer>(&m471_local_top);nup2=8.f;
#pragma unroll 1
for (ip = int(blockIdx.y)+1; ip <= int(blockIdx.y)+1; ++ip) {
	    qcn = zero;
	    overshoot = 0;
	    l = minwidth + dl * (real) (ip - 1);
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
		wmin = rc0048 + l * rc0021;
/* Computing MIN */
/* Computing MAX */
		r__2 = upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		r__1 = max(r__2,wmin);
		ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] = rc0049 / (min(r__1,rc0024) * l);
/* Computing MAX */
		r__1 = ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)];
		ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] = max(r__1,rc0050);
/* Computing MIN */
		r__1 = *pblh + rc0051;
		if (zw1[k] >= min(r__1,rc0052)) {
/* Computing MIN */
		    r__1 = *pblh + rc0051;
		    ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] += (zw1[k] - min(r__1,rc0052)) * 
			    rc0053;
		}
		ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] *= one - pattern_spp_pbl1__[k];
/* Computing MIN */
		r__1 = ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)], r__2 = rc0024 / (zw1[k + 1] - 
			zw1[k]);
		ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] = min(r__1,r__2);
		uk = (u1[k] * dz1[k + 1] + u1[k + 1] * dz1[k]) / (dz1[k + 1] 
			+ dz1[k]);
		ukm1 = (u1[k - 1] * dz1[k] + u1[k] * dz1[k - 1]) / (dz1[k - 1]
			 + dz1[k]);
		vk = (v1[k] * dz1[k + 1] + v1[k + 1] * dz1[k]) / (dz1[k + 1] 
			+ dz1[k]);
		vkm1 = (v1[k - 1] * dz1[k] + v1[k] * dz1[k - 1]) / (dz1[k - 1]
			 + dz1[k]);
		entexp = ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] * (zw1[k + 1] - zw1[k]);
		entexm = entexp * rc0054;
		qtn = upqt[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexp) + qt1[k] 
			* entexp;
		thln = upthl[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexp) + thl1[
			k] * entexp;
		un = upu[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexm) + u1[k] * 
			entexm + dxsa * pgfac * (uk - ukm1);
		vn = upv[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexm) + v1[k] * 
			entexm + dxsa * pgfac * (vk - vkm1);
		if (*tke_opt__ > 0) { qken = upqke[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexp) + qke1[
			k] * entexp; }
		if (*scalar_opt__ > 0) { qncn = upqnc[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexp) + qnc1[
			k] * entexp; }
		if (*scalar_opt__ > 0) { qnin = upqni[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexp) + qni1[
			k] * entexp; }
		if (*scalar_opt__ > 0) { qnwfan = upqnwfa[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexp) + 
			qnwfa1[k] * entexp; }
		if (*scalar_opt__ > 0) { qnifan = upqnifa[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexp) + 
			qnifa1[k] * entexp; }
		if (*scalar_opt__ > 0) { qnbcan = upqnbca[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0011 - entexp) + 
			qnbca1[k] * entexp; }
		qc_ent__ = qcn;
		qt_ent__ = qtn;
		thl_ent__ = thln;
		if (*mix_chem__) {
		    i__2 = *nchem;
		    
#pragma unroll 1
for (ic = 1; ic <= i__2; ++ic) {
			chemn[ic - 1] = upchem[k - 1 + (ip + (ic << 3)) * (1*(MYNN471_NZ-49)+50) 
				- (9*(MYNN471_NZ-49)+451)] * (rc0011 - entexp) + chem1[k + ic * 
				chem1_dim1] * entexp;
/* Lsplit9034: */
		    }
/* Lsplit9035: */
		}
		pk = (p1[k] * dz1[k + 1] + p1[k + 1] * dz1[k]) / (dz1[k + 1] 
			+ dz1[k]);
		condensation_edmf__(m471_work,m471_n, &qtn, &thln, &pk, &zw1[k + 1], &thvn, &
			qcn);
		thvk = (thv1[k] * dz1[k + 1] + thv1[k + 1] * dz1[k]) / (dz1[k 
			+ 1] + dz1[k]);
		thvkm1 = (thv1[k - 1] * dz1[k] + thv1[k] * dz1[k - 1]) / (dz1[
			k - 1] + dz1[k]);
		b = grav * (thvn / thvk - one);
		if (b > rc0007) {
		    bcoeff = rc0055;
		} else {
		    bcoeff = rc0056;
		}
		if (upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] < rc0056) {
/* Computing MAX */
		    r__1 = upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
/* Computing MIN */
		    r__2 = zw1[k] - zw1[k - 1];
		    wn = upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + (-rc0000 * ent[k + ip * 
			    (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] * upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + bcoeff * b 
			    / max(r__1,rc0056)) * min(r__2,rc0057);
		} else {
/* Computing MIN */
		    r__1 = zw1[k] - zw1[k - 1];
		    wn = upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + (-rc0000 * ent[k + ip * 
			    (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] * upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + bcoeff * b 
			    / upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) * min(r__1,rc0057);
		}
/* Computing MIN */
		r__1 = rc0058 * (zw1[k] - zw1[k - 1]) / rc0059;
		if (wn > upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + min(r__1,rc0060)) {
/* Computing MIN */
		    r__1 = rc0058 * (zw1[k] - zw1[k - 1]) / rc0059;
		    wn = upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + min(r__1,rc0060);
		}
/* Computing MIN */
		r__1 = rc0058 * (zw1[k] - zw1[k - 1]) / rc0059;
		if (wn < upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] - min(r__1,rc0060)) {
/* Computing MIN */
		    r__1 = rc0058 * (zw1[k] - zw1[k - 1]) / rc0059;
		    wn = upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] - min(r__1,rc0060);
		}
/* Computing MIN */
		r__1 = max(wn,zero);
		wn = min(r__1,rc0061);
		if (k == *kts + 1 && wn == zero) {
		    nup2 = 0.f;
		    goto Lsplit9032;
		}
		if (FALSE_) {
		    if (wn >= rc0061) {
		    }
		}
		if (wn <= zero && overshoot == 0) {
		    overshoot = 1;
		    if (thvk - thvkm1 > zero) {
			bvf = (real)sqrt((real)(gtr * (thvk - thvkm1) / dz1[k]
				));
			frz = upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] / (bvf * dz1[k]);
/* Computing MAX */
			r__1 = min(frz,one);
			dzp = dz1[k] * max(r__1,zero);
		    }
		} else {
		    dzp = dz1[k];
		}
/* Computing MIN */
		r__1 = upa[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] / (rc0011 - upa[k - 1 + ip * 
			(1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]);
		aratio = min(r__1,rc0009);
		detturb = rc0062;
/* Computing MAX */
		r__1 = one, r__2 = rc0009 * (wn + upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]);
		oow = -rc0063 / max(r__1,r__2);
/* Computing MIN */
/* Computing MAX */
		r__2 = oow * (wn - upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) / dz1[k];
		r__1 = max(r__2,detturb);
		detrate = min(r__1,rc0064);
/* Computing MIN */
/* Computing MAX */
		r__2 = oow * (wn - upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) / dz1[k];
		r__1 = max(r__2,detturb);
		detrateuv = min(r__1,rc0065);
		if (FALSE_) { envm_thl__[k - 1] += (rc0009 * (thl_ent__ + upthl[k - 1 + ip *
			 (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) - thl1[k]) * detrate * aratio * min(dzp,
			dzpmax); }
/* Computing MAX */
		r__1 = qt_ent__ - qc_ent__;
/* Computing MAX */
		r__2 = upqt[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] - upqc[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]
			;
		qv_ent__ = rc0009 * (max(r__1,rc0007) + max(r__2,rc0007));
		if (FALSE_) { envm_sqv__[k - 1] += (qv_ent__ - qv1[k]) * detrate * aratio * 
			min(dzp,dzpmax); }
		if (upqc[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] > rc0066) {
		    if (qc1[k] > rc0067) {
			qc_grid__ = qc1[k];
		    } else {
			qc_grid__ = qc_bl1__[k];
		    }
/* Computing MAX */
		    r__1 = upa[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * rc0009 * (qcn + upqc[k 
			    - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) - qc_grid__;
		    if (FALSE_) { envm_sqc__[k - 1] += max(r__1,zero) * detrate * aratio * 
			    min(dzp,dzpmax); }
		}
		if (FALSE_) { envm_u__[k - 1] += (rc0009 * (un + upu[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) 
			- u1[k]) * detrateuv * aratio * min(dzp,dzpmax); }
		if (FALSE_) { envm_v__[k - 1] += (rc0009 * (vn + upv[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) 
			- v1[k]) * detrateuv * aratio * min(dzp,dzpmax); }
		if (wn > rc0007) {
		    upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = wn;
		    upthv[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = thvn;
		    upthl[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = thln;
		    upqt[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qtn;
		    upqc[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qcn;
		    upu[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = un;
		    upv[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = vn;
		    if (*tke_opt__ > 0) { upqke[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qken; }
		    if (*scalar_opt__ > 0) { upqnc[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qncn; }
		    if (*scalar_opt__ > 0) { upqni[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qnin; }
		    if (*scalar_opt__ > 0) { upqnwfa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qnwfan; }
		    if (*scalar_opt__ > 0) { upqnifa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qnifan; }
		    if (*scalar_opt__ > 0) { upqnbca[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qnbcan; }
		    upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = upa[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		    if (*mix_chem__) {
			i__2 = *nchem;
			
#pragma unroll 1
for (ic = 1; ic <= i__2; ++ic) {
			    upchem[k + (ip + (ic << 3)) * (1*(MYNN471_NZ-49)+50) - (9*(MYNN471_NZ-49)+451)] = chemn[
				    ic - 1];
/* Lsplit9037: */
			}
/* Lsplit9038: */
		    }
		    *ktop = max(*ktop,k);
		} else {
		    if (FALSE_) {
		    }
		    goto Lsplit9032;
		}
/* Lsplit9031: */
	    }
Lsplit9032:
	    if (FALSE_) {
		red001 = -rc0068;
		
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
/* Computing MAX */
		    r__1 = red001, r__2 = upw[ixarr1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		    red001 = max(r__1,r__2);
		}
		red002 = rc0068;
		
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
/* Computing MIN */
		    r__1 = red002, r__2 = upa[ixarr1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		    red002 = min(r__1,r__2);
		}
		red003 = -rc0068;
		
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
/* Computing MAX */
		    r__1 = red003, r__2 = upa[ixarr1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		    red003 = max(r__1,r__2);
		}
		if (red001 > rc0069 || red002 < zero || red003 > atot || nup2 
			> 10.f) {
		}
	    }
/* Lsplit9028: */
	}

    upqv[12+int(blockIdx.y)]=nup2;upqv[20+int(blockIdx.y)]=(real)m471_local_top;

    }

    }
    if constexpr(M471_PHASE==9){
    minwidth=upqv[0];
    pgfac=upqv[1];
    dzpmax=upqv[2];
    psig_w__=upqv[3];
    fluxportion=upqv[4];
    cf_thresh__=upqv[5];
    atot=upqv[6];
    cdet=upqv[7];
    csub=upqv[8];

    nup2=8.f;*ktop=0;
    for(int m471_p=0;m471_p<8;++m471_p){if(upqv[12+m471_p]==0.f)nup2=0.f;*ktop=max(*ktop,(integer)(real)upqv[20+m471_p]);}
/* Computing MIN */
    i__1 = *ktop, i__2 = *kte - 1;
    *ktop = min(i__1,i__2);
    if (*ktop == 0) {
	*ztop = zero;
    } else {
	*ztop = zw1[*ktop];
    }
    if (nup2 > 0.f) {
	
#pragma unroll 1
for(k=*kts;k<=*kte-1;++k){
real m471_fold0=s_aw1__[k + 1];
real m471_fold1=s_awthl1__[k + 1];
real m471_fold2=s_awqt1__[k + 1];
real m471_fold3=s_awqc1__[k + 1];
real m471_fold4=s_awqv1__[k + 1];
#pragma unroll 1
for(ip=1;ip<=8;++ip){

		m471_fold0 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[k 
			+ ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * psig_w__;
		m471_fold1 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upthl[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			psig_w__;
		m471_fold2 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[
			k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqt[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * psig_w__;
		qc_plume__ = upqc[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		m471_fold3 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[
			k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * qc_plume__ * psig_w__;
		m471_fold4 = m471_fold2 - m471_fold3;
/* Lsplit9043: */
	    
}
s_aw1__[k + 1]=m471_fold0;
s_awthl1__[k + 1]=m471_fold1;
s_awqt1__[k + 1]=m471_fold2;
s_awqc1__[k + 1]=m471_fold3;
s_awqv1__[k + 1]=m471_fold4;
}
/* Lsplit9041: */
	if (*momentum_opt__ > 0) {
	    
#pragma unroll 1
for(k=*kts;k<=*kte-1;++k){
real m471_fold0=s_awu1__[k + 1];
real m471_fold1=s_awv1__[k + 1];
#pragma unroll 1
for(ip=1;ip<=8;++ip){

		    m471_fold0 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upu[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    psig_w__;
		    m471_fold1 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upv[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    psig_w__;
/* Lsplit9049: */
		
}
s_awu1__[k + 1]=m471_fold0;
s_awv1__[k + 1]=m471_fold1;
}
/* Lsplit9047: */
	}
	if (*tke_opt__ > 0) {
	    
#pragma unroll 1
for(k=*kts;k<=*kte-1;++k){
real m471_fold0=s_awqke1__[k + 1];
#pragma unroll 1
for(ip=1;ip<=8;++ip){

		    m471_fold0 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] *
			     upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqke[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] *
			     psig_w__;
/* Lsplit9055: */
		
}
s_awqke1__[k + 1]=m471_fold0;
}
/* Lsplit9053: */
	}
	if (*mix_chem__) {
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
		    i__2 = *nchem;
		    
#pragma unroll 1
for (ic = 1; ic <= i__2; ++ic) {
			s_awchem1__[k + 1 + ic * s_awchem1_dim1] += rhoz[k - 
				1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[k + ip * (1*(MYNN471_NZ-49)+50) 
				- (1*(MYNN471_NZ-49)+51)] * upchem[k + (ip + (ic << 3)) * (1*(MYNN471_NZ-49)+50) - 
				(9*(MYNN471_NZ-49)+451)] * psig_w__;
/* Lsplit9064: */
		    }
/* Lsplit9065: */
/* Lsplit9061: */
		}
/* Lsplit9062: */
/* Lsplit9058: */
	    }
/* Lsplit9059: */
	}
	if (*scalar_opt__ > 0) {
	    
#pragma unroll 1
for(k=*kts;k<=*kte-1;++k){
real m471_fold0=s_awqnc1__[k + 1];
real m471_fold1=s_awqni1__[k + 1];
real m471_fold2=s_awqnwfa1__[k + 1];
real m471_fold3=s_awqnifa1__[k + 1];
real m471_fold4=s_awqnbca1__[k + 1];
#pragma unroll 1
for(ip=1;ip<=8;++ip){

		    m471_fold0 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] *
			     upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqnc[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] *
			     psig_w__;
		    m471_fold1 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] *
			     upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqni[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] *
			     psig_w__;
		    m471_fold2 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]
			     * upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqnwfa[k + ip * (1*(MYNN471_NZ-49)+50) - 
			    (1*(MYNN471_NZ-49)+51)] * psig_w__;
		    m471_fold3 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]
			     * upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqnifa[k + ip * (1*(MYNN471_NZ-49)+50) - 
			    (1*(MYNN471_NZ-49)+51)] * psig_w__;
		    m471_fold4 += rhoz[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]
			     * upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqnbca[k + ip * (1*(MYNN471_NZ-49)+50) - 
			    (1*(MYNN471_NZ-49)+51)] * psig_w__;
/* Lsplit9070: */
		
}
s_awqnc1__[k + 1]=m471_fold0;
s_awqni1__[k + 1]=m471_fold1;
s_awqnwfa1__[k + 1]=m471_fold2;
s_awqnifa1__[k + 1]=m471_fold3;
s_awqnbca1__[k + 1]=m471_fold4;
}
/* Lsplit9068: */
	}
	if (s_aw1__[*kts + 1] != rc0007) {
	    dzi[*kts - 1] = half * (dz1[*kts] + dz1[*kts + 1]);
/* Computing MAX */
	    r__1 = s_aw1__[*kts + 1] * (thv1[*kts] - thv1[*kts + 1]) / dzi[*
		    kts - 1];
	    flx1 = max(r__1,rc0070);
	} else {
	    flx1 = zero;
	}
	adjustment = one;
	flt2 = max(*fltv,zero);
	if (flx1 > fluxportion * flt2 / dz1[*kts] && flx1 > zero) {
/* Computing MAX */
	    r__1 = rc0025, r__2 = fluxportion * flt2 / dz1[*kts] / flx1;
	    adjustment = max(r__1,r__2);
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_aw1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awthl1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqt1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqc1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqv1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqnc1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqni1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqnwfa1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqnifa1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqnbca1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    if (*momentum_opt__ > 0) {
		i__1 = *kte + 1 - *kts + 1;
		
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		    s_awu1__[*kts + ixarr1 - 1] *= adjustment;
		}
		i__1 = *kte + 1 - *kts + 1;
		
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		    s_awv1__[*kts + ixarr1 - 1] *= adjustment;
		}
	    }
	    if (*tke_opt__ > 0) {
		i__1 = *kte + 1 - *kts + 1;
		
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		    s_awqke1__[*kts + ixarr1 - 1] *= adjustment;
		}
	    }
	    if (*mix_chem__) {
		i__1 = *nchem;
		
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= i__1; ++ixarr2) {
		    i__2 = *kte + 1 - *kts + 1;
		    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
			s_awchem1__[*kts + ixarr1 - 1 + ixarr2 * 
				s_awchem1_dim1] *= adjustment;
		    }
		}
	    }
	    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
		
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
		    upa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] *= adjustment;
		}
	    }
	}
	if (FALSE_) {
	}
	
#pragma unroll 1
for(k=*kts;k<=*kte-1;++k){
real m471_fold0=edmf_a1__[k];
real m471_fold1=edmf_w1__[k];
real m471_fold2=edmf_qt1__[k];
real m471_fold3=edmf_thl1__[k];
real m471_fold4=edmf_ent1__[k];
real m471_fold5=edmf_qc1__[k];
#pragma unroll 1
for(ip=1;ip<=8;++ip){

		m471_fold0 += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		m471_fold1 += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		m471_fold2 += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqt[k + ip * (1*(MYNN471_NZ-49)+50) - 
			(1*(MYNN471_NZ-49)+51)];
		m471_fold3 += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upthl[k + ip * (1*(MYNN471_NZ-49)+50) - 
			(1*(MYNN471_NZ-49)+51)];
		m471_fold4 += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * ent[k + ip * (1*(MYNN471_NZ-49)+49) - 
			(1*(MYNN471_NZ-49)+50)];
		m471_fold5 += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqc[k + ip * (1*(MYNN471_NZ-49)+50) - 
			(1*(MYNN471_NZ-49)+51)];
/* Lsplit9076: */
	    
}
edmf_a1__[k]=m471_fold0;
edmf_w1__[k]=m471_fold1;
edmf_qt1__[k]=m471_fold2;
edmf_thl1__[k]=m471_fold3;
edmf_ent1__[k]=m471_fold4;
edmf_qc1__[k]=m471_fold5;
}
/* Lsplit9074: */
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    if (edmf_a1__[k] > rc0007) {
		edmf_w1__[k] /= edmf_a1__[k];
		edmf_qt1__[k] /= edmf_a1__[k];
		edmf_thl1__[k] /= edmf_a1__[k];
		edmf_ent1__[k] /= edmf_a1__[k];
		edmf_qc1__[k] /= edmf_a1__[k];
		edmf_a1__[k] *= psig_w__;
		if (edmf_a1__[k] * edmf_w1__[k] > *maxmf) {
		    *maxmf = edmf_a1__[k] * edmf_w1__[k];
		}
	    }
/* Computing 3rd power */
	    r__2 = (r__1 = edmf_w1__[k], abs(r__1));
/* Computing MAX */
	    r__3 = el1[k];
	    tkeprod_up__[k] = r__2 * (r__2 * r__2) * edmf_a1__[k] / (b1 * max(
		    r__3,rc0038));
/* Lsplit9079: */
	}
/* Lsplit9080: */
	if (*mix_chem__) {
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
		    i__2 = *nchem;
		    
#pragma unroll 1
for (ic = 1; ic <= i__2; ++ic) {
			edmf_chem__[k + ic * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]
				 * upchem[k + (ip + (ic << 3)) * (1*(MYNN471_NZ-49)+50) - (9*(MYNN471_NZ-49)+451)];
/* Lsplit9088: */
		    }
/* Lsplit9089: */
/* Lsplit9085: */
		}
/* Lsplit9086: */
/* Lsplit9082: */
	    }
/* Lsplit9083: */
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		if (edmf_a1__[k] > rc0007) {
		    i__2 = *nchem;
		    
#pragma unroll 1
for (ic = 1; ic <= i__2; ++ic) {
			edmf_chem__[k + ic * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] /= edmf_a1__[k];
/* Lsplit9094: */
		    }
/* Lsplit9095: */
		}
/* Lsplit9091: */
	    }
/* Lsplit9092: */
	}
	if (FALSE_) {
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
		envi_w__[k - 1] = onethird * (edmf_w1__[k - 1] + edmf_w1__[k] 
			+ edmf_w1__[k + 1]);
		envi_a__[k - 1] = onethird * (edmf_a1__[k - 1] + edmf_a1__[k] 
			+ edmf_a1__[k + 1]) * adjustment;
/* Lsplit9097: */
	    }
/* Lsplit9098: */
	    envi_w__[*kts - 1] = edmf_w1__[*kts];
	    envi_a__[*kts - 1] = edmf_a1__[*kts];
	    envi_w__[*kte - 1] = zero;
	    envi_a__[*kte - 1] = edmf_a1__[*kte];
	    envi_w__[*kte] = zero;
	    envi_a__[*kte] = edmf_a1__[*kte];
	    if (envi_w__[*kts - 1] > rc0024 * dz1[*kts] / *dt) {
		sublim = rc0024 * dz1[*kts] / *dt / envi_w__[*kts - 1];
	    } else {
		sublim = one;
	    }
	    i__1 = *kte;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		temp = envi_a__[k - 1];
		envi_a__[k - 1] = one - temp;
		envi_w__[k - 1] = csub * sublim * envi_w__[k - 1] * temp / (
			rc0011 - temp);
/* Lsplit9100: */
	    }
/* Lsplit9101: */
	    dzi[*kts - 1] = rc0009 * (dz1[*kts] + dz1[*kts + 1]);
	    sub_thl1__[*kts] = rc0009 * envi_w__[*kts - 1] * envi_a__[*kts - 
		    1] * (rho1[*kts + 1] * thl1[*kts + 1] - rho1[*kts] * thl1[
		    *kts]) / dzi[*kts - 1] / rhoz[*kts - 1];
	    sub_sqv1__[*kts] = rc0009 * envi_w__[*kts - 1] * envi_a__[*kts - 
		    1] * (rho1[*kts + 1] * qv1[*kts + 1] - rho1[*kts] * qv1[*
		    kts]) / dzi[*kts - 1] / rhoz[*kts - 1];
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
		dzi[k - 1] = rc0009 * (dz1[k] + dz1[k + 1]);
		sub_thl1__[k] = rc0009 * (envi_w__[k - 1] + envi_w__[k - 2]) *
			 rc0009 * (envi_a__[k - 1] + envi_a__[k - 2]) * (rho1[
			k + 1] * thl1[k + 1] - rho1[k] * thl1[k]) / dzi[k - 1]
			 / rhoz[k - 1];
		sub_sqv1__[k] = rc0009 * (envi_w__[k - 1] + envi_w__[k - 2]) *
			 rc0009 * (envi_a__[k - 1] + envi_a__[k - 2]) * (rho1[
			k + 1] * qv1[k + 1] - rho1[k] * qv1[k]) / dzi[k - 1] /
			 rhoz[k - 1];
/* Lsplit9103: */
	    }
/* Lsplit9104: */
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		det_thl1__[k] = cdet * (envm_thl__[k - 1] - thl1[k]) * 
			envi_a__[k - 1] * psig_w__;
		det_sqv1__[k] = cdet * (envm_sqv__[k - 1] - qv1[k]) * 
			envi_a__[k - 1] * psig_w__;
		det_sqc1__[k] = cdet * (envm_sqc__[k - 1] - qc1[k]) * 
			envi_a__[k - 1] * psig_w__;
/* Lsplit9106: */
	    }
/* Lsplit9107: */
	    if (*momentum_opt__ > 0) {
		sub_u1__[*kts] = rc0009 * envi_w__[*kts - 1] * envi_a__[*kts 
			- 1] * (rho1[*kts + 1] * u1[*kts + 1] - rho1[*kts] * 
			u1[*kts]) / dzi[*kts - 1] / rhoz[*kts - 1];
		sub_v1__[*kts] = rc0009 * envi_w__[*kts - 1] * envi_a__[*kts 
			- 1] * (rho1[*kts + 1] * v1[*kts + 1] - rho1[*kts] * 
			v1[*kts]) / dzi[*kts - 1] / rhoz[*kts - 1];
		i__1 = *kte - 1;
		
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
		    sub_u1__[k] = rc0009 * (envi_w__[k - 1] + envi_w__[k - 2])
			     * rc0009 * (envi_a__[k - 1] + envi_a__[k - 2]) * 
			    (rho1[k + 1] * u1[k + 1] - rho1[k] * u1[k]) / dzi[
			    k - 1] / rhoz[k - 1];
		    sub_v1__[k] = rc0009 * (envi_w__[k - 1] + envi_w__[k - 2])
			     * rc0009 * (envi_a__[k - 1] + envi_a__[k - 2]) * 
			    (rho1[k + 1] * v1[k + 1] - rho1[k] * v1[k]) / dzi[
			    k - 1] / rhoz[k - 1];
/* Lsplit9109: */
		}
/* Lsplit9110: */
		i__1 = *kte - 1;
		
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		    det_u1__[k] = cdet * (envm_u__[k - 1] - u1[k]) * envi_a__[
			    k - 1] * psig_w__;
		    det_v1__[k] = cdet * (envm_v__[k - 1] - v1[k]) * envi_a__[
			    k - 1] * psig_w__;
/* Lsplit9112: */
		}
/* Lsplit9113: */
	    }
	}
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    if (FALSE_) { exneri[k - 1] = (ex1[k] * dz1[k + 1] + ex1[k + 1] * dz1[k]) / (
		    dz1[k + 1] + dz1[k]); }
	    edmf_th1__[k - 1] = edmf_thl1__[k] + xlvcp / ex1[k] * edmf_qc1__[
		    k];
	    dzi[k - 1] = rc0009 * (dz1[k] + dz1[k + 1]);
/* Lsplit9115: */
	}
/* Lsplit9116: */
	i__1 = *kte - 2;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    if (k > *ktop) {
		goto Lsplit9119;
	    }
	    if (rc0009 * (edmf_qc1__[k] + edmf_qc1__[k - 1]) > zero && 
		    cldfra_bl1__[k] < cf_thresh__) {
		aup = (edmf_a1__[k] * dzi[k - 2] + edmf_a1__[k - 1] * dzi[k - 
			1]) / (dzi[k - 2] + dzi[k - 1]);
		thp = (edmf_th1__[k - 1] * dzi[k - 2] + edmf_th1__[k - 2] * 
			dzi[k - 1]) / (dzi[k - 2] + dzi[k - 1]);
		qtp = (edmf_qt1__[k] * dzi[k - 2] + edmf_qt1__[k - 1] * dzi[k 
			- 1]) / (dzi[k - 2] + dzi[k - 1]);
		esat = (real)esat_blend__(m471_work,m471_n, &tk1[k]);
/* Computing MAX */
		r__1 = rc0071, r__2 = p1[k] - ep_3__ * esat;
		qsl = ep_2__ * esat / max(r__1,r__2);
		if (edmf_qc1__[k] > zero && edmf_qc1__[k - 1] > zero) {
		    qcp = (edmf_qc1__[k] * dzi[k - 2] + edmf_qc1__[k - 1] * 
			    dzi[k - 1]) / (dzi[k - 2] + dzi[k - 1]);
		} else {
/* Computing MAX */
		    r__1 = edmf_qc1__[k], r__2 = edmf_qc1__[k - 1];
		    qcp = max(r__1,r__2);
		}
		xl = (real)xl_blend__(m471_work,m471_n, &tk1[k]);
		qsat_tk__ = (real)qsat_blend__(m471_work,m471_n, &tk1[k], &p1[k]);
/* Computing 2nd power */
		r__1 = tk1[k];
		rsl = xl * qsat_tk__ / (r_v__ * (r__1 * r__1));
		cpm = cp + qt1[k] * cpv;
		a = rc0011 / (rc0011 + xl * rsl / cpm);
		b9 = a * rsl;
		q2p = xlvcp / ex1[k];
		pt = thl1[k] + q2p * qcp * aup;
		bb = b9 * tk1[k] / pt;
		qww = rc0011 + rc0072 * qt1[k];
		alpha = rc0072 * pt;
		beta = pt * xl / (tk1[k] * cp) - rc0073 * pt;
		if (a > zero) {
/* Computing MIN */
		    r__1 = rc0074 / a;
		    f = min(r__1,rc0044);
		} else {
		    f = one;
		}
		sigq = rc0036 * aup * (qtp - qt1[k]);
/* Computing MAX */
		r__1 = sigq, r__2 = qsat_tk__ * rc0031;
		sigq = max(r__1,r__2);
/* Computing MIN */
		r__1 = sigq, r__2 = qsat_tk__ * rc0013;
		sigq = min(r__1,r__2);
		qmq = a * (qt1[k] - qsat_tk__);
		q1 = qmq / sigq;
		if (*landsea - rc0003 >= zero) {
/* Computing MIN */
/* Computing MAX */
		    r__2 = rc0009 + rc0075 * (real)atan((real)(rc0076 * q1));
		    r__1 = max(r__2,rc0025);
		    mf_cf__ = min(r__1,rc0045);
/* Computing MAX */
		    r__1 = mf_cf__, r__2 = rc0008 * aup;
		    mf_cf__ = max(r__1,r__2);
		} else {
/* Computing MIN */
/* Computing MAX */
		    r__2 = rc0009 + rc0075 * (real)atan((real)(rc0076 * q1));
		    r__1 = max(r__2,rc0025);
		    mf_cf__ = min(r__1,rc0045);
/* Computing MAX */
		    r__1 = mf_cf__, r__2 = rc0077 * aup;
		    mf_cf__ = max(r__1,r__2);
		}
		if (*landsea - rc0003 >= zero) {
		    if (qcp * aup > rc0078) {
			qc_bl1__[k] = rc0079 * (qcp * aup) - rc0080;
		    } else {
			qc_bl1__[k] = rc0081 * (qcp * aup);
		    }
		    cldfra_bl1__[k] = mf_cf__;
		    ac_mf__ = mf_cf__;
		} else {
		    if (qcp * aup > rc0078) {
			qc_bl1__[k] = rc0079 * (qcp * aup) - rc0080;
		    } else {
			qc_bl1__[k] = rc0081 * (qcp * aup);
		    }
		    cldfra_bl1__[k] = mf_cf__;
		    ac_mf__ = mf_cf__;
		}
/* Computing MAX */
		r__1 = q1, r__2 = -rc0082;
		q1 = max(r__1,r__2);
		if (q1 >= one) {
		    fng = one;
		} else if (q1 >= -rc0083 && q1 < one) {
		    fng = (real)exp((real)(-rc0039 * (q1 - one)));
		} else if (q1 >= -rc0084 && q1 < -rc0083) {
		    fng = rc0061 + (real)exp((real)(-rc0085 * (q1 + rc0083)));
		} else {
/* Computing MIN */
		    r__1 = rc0086 + (real)exp((real)(-rc0087 * (q1 + rc0084)))
			    ;
		    fng = min(r__1,rc0088);
		}
		vt1[k] = qww - rc0003 * aup * beta * bb * fng - one;
		vq1[k] = alpha + rc0003 * aup * beta * a * fng - tv0;
	    }
/* Lsplit9118: */
	}
Lsplit9119:
	;
    }
    if (*ktop > 0) {
	red001 = -rc0068;
	i__1 = *ktop;
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
/* Computing MAX */
	    r__1 = red001, r__2 = edmf_qc1__[ixarr1];
	    red001 = max(r__1,r__2);
	}
	maxqc = red001;
	if (maxqc < rc0089) {
	    *maxmf = -rc0011 * *maxmf;
	}
    }
    if (FALSE_) {
    }
    return 0;

    }
    }
    return 0;
} /* dmp_mf__ */

__device__ M471_CALL /* Subroutine */ int condensation_edmf__(unsigned char *m471_work, int m471_n, M471Ptr<real> qt, M471Ptr<real> thl, M471Ptr<real> p, M471Ptr<real> zagl, M471Ptr<real> thv, M471Ptr<real> qc) {
    /* Initialized data */

    real p1000mb = 1e5f;
    real rcp = .2857142984867095947265625f;
    real rvovrd = 1.6083624362945556640625f;
    real xlvcp = 2488.80029296875f;
    real rc0000 = 1e-6f;
    real rc0001 = .5f;
    real rc0002 = 0.f;
    real rc0003 = 100.f;
    real rc0004 = 1.f;

    /* System generated locals */
    integer i__1;
    real r__1;
    doublereal d__1, d__2;

    /* Builtin functions */
    

    /* Local variables */
    real t;
    integer ni;
    real qs;
    
    real exn, diff, qcold;
    integer niter;

    niter = 50;
    diff = rc0000;
    d__1 =  (*p / p1000mb);
    d__2 =  rcp;
    exn = (real)pow_dd(&d__1, &d__2);
    i__1 = niter;
    
#pragma unroll 1
for (ni = 1; ni <= i__1; ++ni) {
	t = exn * *thl + xlvcp * *qc;
	qs = (real)qsat_blend__(m471_work,m471_n, &t, p);
	qcold = *qc;
/* Computing MAX */
	r__1 = *qt - qs;
	*qc = rc0001 * *qc + rc0001 * max(r__1,rc0002);
	if ((r__1 = *qc - qcold, abs(r__1)) < diff) {
	    goto L9005;
	}
/* L9004: */
    }
L9005:
    t = exn * *thl + xlvcp * *qc;
    qs = (real)qsat_blend__(m471_work,m471_n, &t, p);
/* Computing MAX */
    r__1 = *qt - qs;
    *qc = max(r__1,rc0002);
    if (*zagl < rc0003) {
	*qc = rc0002;
    }
    *thv = (*thl + xlvcp * *qc) * (rc0004 + *qt * (rvovrd - rc0004) - rvovrd *
	     *qc);
    return 0;
} /* condensation_edmf__ */

__device__ M471_CALL /* Subroutine */ int condensation_ddmf__(unsigned char *m471_work, int m471_n, M471Ptr<real> qt, M471Ptr<real> thl, M471Ptr<real> p, M471Ptr<real> zagl, M471Ptr<real> thv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<integer> debug_dd__) {
    /* Initialized data */

    real p1000mb = 1e5f;
    real rcp = .2857142984867095947265625f;
    real p608 = .6083624362945556640625f;
    real zero = 0.f;
    real one = 1.f;
    real xlscp = 2837.232421875f;
    real xlvcp = 2488.80029296875f;
    real rc0000 = 1e-6f;
    real rc0001 = .5f;
    real rc0002 = 0.f;
    real rc0003 = 1.f;

    /* System generated locals */
    integer i__1;
    real r__1;
    doublereal d__1, d__2;

    /* Builtin functions */
    

    /* Local variables */
    real frac_ice__, frac_liq__, t;
    integer ni;
    real qs, qx;
    
    real exn, diff;
    integer niter;
    real qxold, thvin;

    niter = 50;
    diff = rc0000;
    if (*qi + *qc > zero) {
	frac_ice__ = *qi / (*qi + *qc);
	frac_liq__ = one - frac_ice__;
    } else {
	frac_ice__ = zero;
	frac_liq__ = zero;
    }
    if (*debug_dd__ == 1) {
	thvin = *thv;
    }
    d__1 =  (*p / p1000mb);
    d__2 =  rcp;
    exn = (real)pow_dd(&d__1, &d__2);
    i__1 = niter;
    
#pragma unroll 1
for (ni = 1; ni <= i__1; ++ni) {
	t = exn * *thl + xlvcp * *qc + xlscp * *qi;
	qs = (real)qsat_blend__(m471_work,m471_n, &t, p);
	qxold = *qc + *qi;
	qx = *qc + *qi;
/* Computing MAX */
	r__1 = *qt - qs;
	qx = rc0001 * qx + rc0001 * max(r__1,rc0002);
	*qc = frac_liq__ * qx;
	*qi = frac_ice__ * qx;
	if ((r__1 = qx - qxold, abs(r__1)) < diff) {
	    goto L9005;
	}
/* L9004: */
    }
L9005:
    t = exn * *thl + xlvcp * *qc + xlscp * *qi;
    qs = (real)qsat_blend__(m471_work,m471_n, &t, p);
/* Computing MAX */
    r__1 = *qt - qs;
    qx = max(r__1,rc0002);
    *qc = frac_liq__ * qx;
    *qi = frac_ice__ * qx;
    *thv = (*thl + xlvcp * *qc + xlscp * *qi) * (rc0003 + p608 * (*qt - qx));
    if (*debug_dd__ == 1) {
    }
    return 0;
} /* condensation_ddmf__ */

__device__ M471_CALL /* Subroutine */ int condensation_edmf_r__(unsigned char *m471_work, int m471_n, M471Ptr<real> qt, M471Ptr<real> thl, M471Ptr<real> p, M471Ptr<real> zagl, M471Ptr<real> thv, M471Ptr<real> qc) {
    /* Initialized data */

    real p1000mb = 1e5f;
    real cp = 1004.5f;
    real rcp = .2857142984867095947265625f;
    real xlv = 2.5e6f;
    real rvovrd = 1.6083624362945556640625f;
    real rc0000 = 2e-5f;
    real rc0001 = 0.f;
    real rc0002 = 1.f;

    /* System generated locals */
    integer i__1;
    real r__1;
    doublereal d__1, d__2;

    /* Builtin functions */
    

    /* Local variables */
    real t;
    integer ni;
    real qs;
    
    real exn, diff, qcold;
    integer niter;

    niter = 50;
    diff = rc0000;
    d__1 =  (*p / p1000mb);
    d__2 =  rcp;
    exn = (real)pow_dd(&d__1, &d__2);
    t = *thv * exn;
    *qc = rc0001;
    i__1 = niter;
    
#pragma unroll 1
for (ni = 1; ni <= i__1; ++ni) {
	qcold = *qc;
	t = exn * *thv / (rc0002 + *qt * (rvovrd - rc0002) - rvovrd * *qc);
	qs = (real)qsat_blend__(m471_work,m471_n, &t, p);
/* Computing MAX */
	r__1 = *qt - qs;
	*qc = max(r__1,rc0001);
	if ((r__1 = *qc - qcold, abs(r__1)) < diff) {
	    goto L9005;
	}
/* L9004: */
    }
L9005:
    *thl = (t - xlv / cp * *qc) / exn;
    return 0;
} /* condensation_edmf_r__ */

__device__ M471_CALL /* Subroutine */ int ddmp_mf__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dt, M471Ptr<real> dx,
	 M471Ptr<real> zw, M471Ptr<real> dz, M471Ptr<real> p, M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> th, M471Ptr<real> thl, 
	M471Ptr<real> thv, M471Ptr<real> tk, M471Ptr<real> qt, M471Ptr<real> qv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<real> qnc, M471Ptr<real> qni, M471Ptr<real> qnwfa, M471Ptr<real> qnifa, M471Ptr<real> qke, M471Ptr<real> rho, M471Ptr<real> exner, M471Ptr<real> qc_bl1__, M471Ptr<real> qi_bl1__, M471Ptr<real> cldfra_bl1__, M471Ptr<real> ust, 
	M471Ptr<real> flt, M471Ptr<real> flq, M471Ptr<real> fltv, M471Ptr<real> pblh, M471Ptr<integer> kpbl, M471Ptr<real> edmf_a_dd__, M471Ptr<real> edmf_w_dd__, M471Ptr<real> edmf_qt_dd__, M471Ptr<real> edmf_thl_dd__, M471Ptr<real> edmf_ent_dd__, M471Ptr<real> edmf_qc_dd__, M471Ptr<real> sd_aw__,
	 M471Ptr<real> sd_awthl__, M471Ptr<real> sd_awqt__, M471Ptr<real> sd_awqv__, M471Ptr<real> sd_awqc__, 
	M471Ptr<real> sd_awqi__, M471Ptr<real> sd_awqnc__, M471Ptr<real> sd_awqni__, M471Ptr<real> sd_awqnwfa__, M471Ptr<real> sd_awqnifa__, M471Ptr<real> sd_awu__, M471Ptr<real> sd_awv__, 
	M471Ptr<real> sd_awqke__, M471Ptr<real> tkeprod_dn__, M471Ptr<real> el, M471Ptr<real> rthraten) {
    /* Initialized data */

    real cp = 1004.5f;
    real p608 = .6083624362945556640625f;
    real zero = 0.f;
    real one = 1.f;
    real onethird = .3333333432674407958984375f;
    real grav = 9.81000041961669921875f;
    real b1 = 24.f;
    real rc0000 = 400.f;
    real rc0001 = 50.f;
    real rc0002 = 200.f;
    real rc0003 = 1.f;
    real rc0004 = 1.5f;
    real rc0005 = 100.f;
    real rc0006 = 1.2f;
    real rc0007 = 2.5e-5f;
    real rc0008 = 1e-6f;
    real rc0009 = 1.f;
    real rc0010 = .1f;
    real rc0011 = .002f;
    real rc0012 = .3f;
    real rc0013 = .01f;
    real rc0014 = 10.f;
    real rc0015 = 10.f;
    real rc0016 = 15.f;
    real rc0017 = .5f;
    real rc0018 = 3.f;
    real rc0019 = 40.f;
    real rc0020 = 3.f;
    real rc0021 = .2f;
    real rc0022 = .05f;
    real rc0023 = .33f;
    real rc0024 = 5e-4f;
    real rc0025 = .53f;
    real rc0026 = .6f;
    real rc0027 = 3e-4f;
    real rc0028 = .06f;
    real rc0029 = .9f;
    real rc0030 = 2.f;
    real rc0031 = 250.f;
    real rc0032 = 1.25f;
    real rc0033 = 0.f;

    /* System generated locals */
    integer u_offset, v_offset, th_offset, thl_offset, tk_offset, qt_offset, 
	    qv_offset, qc_offset, qi_offset, thv_offset, p_offset, qke_offset,
	     rho_offset, exner_offset, qnc_offset, qni_offset, qnwfa_offset, 
	    qnifa_offset, dz_offset, qc_bl1_offset, qi_bl1_offset, 
	    cldfra_bl1_offset, el_offset, rthraten_offset, zw_offset, 
	    edmf_a_dd_offset, edmf_w_dd_offset, edmf_qt_dd_offset, 
	    edmf_thl_dd_offset, edmf_ent_dd_offset, edmf_qc_dd_offset, 
	    tkeprod_dn_offset, sd_aw_offset, sd_awthl_offset, sd_awu_offset, 
	    sd_awv_offset, sd_awqt_offset, sd_awqc_offset, sd_awqv_offset, 
	    sd_awqi_offset, sd_awqnc_offset, sd_awqni_offset, 
	    sd_awqnwfa_offset, sd_awqnifa_offset, sd_awqke_offset, i__1, i__2,
	     i__3;
    real r__1, r__2, r__3, r__4;
    doublereal d__1, d__2;

    /* Builtin functions */
    

    /* Local variables */
    real frac_ice__, refthvm1, frac_liq__;
    M471Ptr<integer> dd_initk__=m471_array<integer>(m471_work,(848*(MYNN471_NZ-49)+59576),m471_n);
    real refqnifa;
    logical cloudflg;
    real refqnwfa, wspd_pbl__, jump_thv__, mindownw;
    integer i__, k;
    real f0, wstar_rad__;
M471Ptr<real> downqnifa=m471_array<real>(m471_work,(848*(MYNN471_NZ-49)+59600),m471_n);
M471Ptr<real> downqnwfa=m471_array<real>(m471_work,(872*(MYNN471_NZ-49)+60600),m471_n);
    
    real dl;
    integer ki;
    real wa, pk, un, vn, wn, wb, z00;
    
    real qs1, qs2, add, ddd, qtn, qcn, qin, 
	    dz200, jump_thetal__;
M471Ptr<real> ent=m471_array<real>(m471_work,(896*(MYNN471_NZ-49)+61600),m471_n);
    logical singlelayer;
    real refp, qncn, refu, thln, qnin, refv, qx_k__, wmin, wmax, zmin, went, 
	    thvn, thvk, buoy, refqc, refqi, 
	    reftk, cftop, refqt, qstar, pwmin;
M471Ptr<real> downa=m471_array<real>(m471_work,(920*(MYNN471_NZ-49)+62600),m471_n);
M471Ptr<real> downu=m471_array<real>(m471_work,(944*(MYNN471_NZ-49)+63600),m471_n);
M471Ptr<real> downw=m471_array<real>(m471_work,(968*(MYNN471_NZ-49)+64600),m471_n);
M471Ptr<real> downv=m471_array<real>(m471_work,(992*(MYNN471_NZ-49)+65600),m471_n);
    integer qltop;
    real pwmax, maxdd2, qx_km1__, refqt2;
    integer ixarr1, ixarr2;
    real bcoeff, ecoeff, def_th__, minddd, maxddd, def_qt__;
    integer qlbase;
    real qnifan, refqke, refqnc, ac_wsp__, minexc, maxexc, sigmaw, refthl, 
	    refqni, qnwfan, dz_ent__, refthv, 
	    entexm, entexp, thstar, sigmath, rho_int__, sigmaqt, radflux, refthlv, jump_qt__;
M471Ptr<real> downqc=m471_array<real>(m471_work,(1016*(MYNN471_NZ-49)+66600),m471_n);
M471Ptr<real> downqi=m471_array<real>(m471_work,(1040*(MYNN471_NZ-49)+67600),m471_n);
M471Ptr<real> downqt=m471_array<real>(m471_work,(1064*(MYNN471_NZ-49)+68600),m471_n);
M471Ptr<real> downqnc=m471_array<real>(m471_work,(1088*(MYNN471_NZ-49)+69600),m471_n);
M471Ptr<real> downthl=m471_array<real>(m471_work,(1112*(MYNN471_NZ-49)+70600),m471_n);
M471Ptr<real> downqni=m471_array<real>(m471_work,(1136*(MYNN471_NZ-49)+71600),m471_n);
M471Ptr<real> downthv=m471_array<real>(m471_work,(1160*(MYNN471_NZ-49)+72600),m471_n);

    /* Parameter adjustments */
    rthraten_offset = *kts;
    rthraten -= rthraten_offset;
    el_offset = *kts;
    el -= el_offset;
    tkeprod_dn_offset = *kts;
    tkeprod_dn__ -= tkeprod_dn_offset;
    sd_awqke_offset = *kts;
    sd_awqke__ -= sd_awqke_offset;
    sd_awv_offset = *kts;
    sd_awv__ -= sd_awv_offset;
    sd_awu_offset = *kts;
    sd_awu__ -= sd_awu_offset;
    sd_awqnifa_offset = *kts;
    sd_awqnifa__ -= sd_awqnifa_offset;
    sd_awqnwfa_offset = *kts;
    sd_awqnwfa__ -= sd_awqnwfa_offset;
    sd_awqni_offset = *kts;
    sd_awqni__ -= sd_awqni_offset;
    sd_awqnc_offset = *kts;
    sd_awqnc__ -= sd_awqnc_offset;
    sd_awqi_offset = *kts;
    sd_awqi__ -= sd_awqi_offset;
    sd_awqc_offset = *kts;
    sd_awqc__ -= sd_awqc_offset;
    sd_awqv_offset = *kts;
    sd_awqv__ -= sd_awqv_offset;
    sd_awqt_offset = *kts;
    sd_awqt__ -= sd_awqt_offset;
    sd_awthl_offset = *kts;
    sd_awthl__ -= sd_awthl_offset;
    sd_aw_offset = *kts;
    sd_aw__ -= sd_aw_offset;
    edmf_qc_dd_offset = *kts;
    edmf_qc_dd__ -= edmf_qc_dd_offset;
    edmf_ent_dd_offset = *kts;
    edmf_ent_dd__ -= edmf_ent_dd_offset;
    edmf_thl_dd_offset = *kts;
    edmf_thl_dd__ -= edmf_thl_dd_offset;
    edmf_qt_dd_offset = *kts;
    edmf_qt_dd__ -= edmf_qt_dd_offset;
    edmf_w_dd_offset = *kts;
    edmf_w_dd__ -= edmf_w_dd_offset;
    edmf_a_dd_offset = *kts;
    edmf_a_dd__ -= edmf_a_dd_offset;
    cldfra_bl1_offset = *kts;
    cldfra_bl1__ -= cldfra_bl1_offset;
    qi_bl1_offset = *kts;
    qi_bl1__ -= qi_bl1_offset;
    qc_bl1_offset = *kts;
    qc_bl1__ -= qc_bl1_offset;
    exner_offset = *kts;
    exner -= exner_offset;
    rho_offset = *kts;
    rho -= rho_offset;
    qke_offset = *kts;
    qke -= qke_offset;
    qnifa_offset = *kts;
    qnifa -= qnifa_offset;
    qnwfa_offset = *kts;
    qnwfa -= qnwfa_offset;
    qni_offset = *kts;
    qni -= qni_offset;
    qnc_offset = *kts;
    qnc -= qnc_offset;
    qi_offset = *kts;
    qi -= qi_offset;
    qc_offset = *kts;
    qc -= qc_offset;
    qv_offset = *kts;
    qv -= qv_offset;
    qt_offset = *kts;
    qt -= qt_offset;
    tk_offset = *kts;
    tk -= tk_offset;
    thv_offset = *kts;
    thv -= thv_offset;
    thl_offset = *kts;
    thl -= thl_offset;
    th_offset = *kts;
    th -= th_offset;
    v_offset = *kts;
    v -= v_offset;
    u_offset = *kts;
    u -= u_offset;
    p_offset = *kts;
    p -= p_offset;
    dz_offset = *kts;
    dz -= dz_offset;
    zw_offset = *kts;
    zw -= zw_offset;

    /* Function Body */
    minddd = rc0000;
    maxddd = 1e3f;
    zmin = rc0001;
    dz200 = rc0002;
    wa = rc0003;
    wb = rc0004;
    z00 = rc0005;
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downw[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downthl[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downthv[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downqt[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downu[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downv[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downqc[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downqi[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downqnc[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downqni[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downqnwfa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downqnifa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    ent[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	tkeprod_dn__[*kts + ixarr1 - 1] = zero;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= 5; ++ixarr1) {
	dd_initk__[ixarr1 - 1] = 0;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_a_dd__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_w_dd__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_qt_dd__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_thl_dd__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_ent_dd__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_qc_dd__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_aw__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awthl__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqt__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqv__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqc__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqi__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqnc__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqni__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqnwfa__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqnifa__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awu__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awv__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqke__[*kts + ixarr1 - 1] = zero;
    }
/* Computing MIN */
    r__1 = maxddd, r__2 = rc0006 * *dx;
    maxdd2 = min(r__1,r__2);
/* Computing MAX */
    r__1 = maxdd2 - minddd;
    dl = max(r__1,zero) / 4.f;
    cloudflg = FALSE_;
    singlelayer = FALSE_;
    qltop = 1;
    qlbase = 1;
/* Computing MAX */
    i__1 = 4, i__2 = *kpbl - 2;
    i__3 = *kpbl + 10;
    
#pragma unroll 1
for (k = max(i__1,i__2); k <= i__3; ++k) {
/* Computing MAX */
	r__1 = qt[k], r__2 = qc_bl1__[k] + qi_bl1__[k];
	qx_k__ = max(r__1,r__2);
/* Computing MAX */
	r__1 = qt[k - 1], r__2 = qc_bl1__[k - 1] + qi_bl1__[k - 1];
	qx_km1__ = max(r__1,r__2);
	if (cldfra_bl1__[k - 1] > .5f && rthraten[k - 1] < -rc0007 && dl > 
		zero) {
	    cloudflg = TRUE_;
	    qltop = k - 1;
	    cftop = cldfra_bl1__[k - 1];
	    if (cldfra_bl1__[k - 2] < .1f) {
		singlelayer = TRUE_;
	    }
	}
/* L9004: */
    }
/* L9005: */
    if (cloudflg) {
	i__3 = *kts;
	
#pragma unroll 1
for (k = qltop; k >= i__3; --k) {
/* Computing MAX */
	    r__1 = qt[k], r__2 = qc_bl1__[k] + qi_bl1__[k];
	    qx_k__ = max(r__1,r__2);
	    if (qx_k__ > rc0008) {
		qlbase = k;
	    }
/* L9007: */
	}
/* L9008: */
	
#pragma unroll 1
for (i__ = 1; i__ <= 5; ++i__) {
	    dd_initk__[i__ - 1] = qltop;
/* L9010: */
	}
/* L9011: */
	f0 = zero;
	i__3 = qltop;
	
#pragma unroll 1
for (k = 3; k <= i__3; ++k) {
	    radflux = rthraten[k] * exner[k];
	    radflux = radflux * cp / grav * (p[k] - p[k + 1]);
	    if (radflux < zero) {
		f0 = abs(radflux) + f0;
	    }
/* L9013: */
	}
/* L9014: */
/* Computing MIN */
	r__1 = max(f0,rc0009);
	f0 = min(r__1,rc0002);
/* Computing MIN */
	r__1 = rc0010 + f0 * rc0011;
	add = min(r__1,rc0012);
/* Computing MAX */
/* Computing 2nd power */
	r__2 = u[*kts];
/* Computing 2nd power */
	r__3 = v[*kts];
	r__1 = r__2 * r__2 + r__3 * r__3;
	wspd_pbl__ = (real)sqrt((real)(max(r__1,rc0013)));
	if (wspd_pbl__ <= rc0014) {
	    ac_wsp__ = one;
	} else {
/* Computing MIN */
	    r__1 = (wspd_pbl__ - rc0015) / rc0016;
	    ac_wsp__ = one - min(r__1,rc0009);
	}
	add *= ac_wsp__;
	dz_ent__ = rc0017 * (dz[qltop + 1] + dz[qltop]);
	jump_thv__ = (thv[qltop + 1] - thv[qltop]) / dz_ent__ * dz200;
	jump_qt__ = (qt[qltop + 1] - qt[qltop]) / dz_ent__ * dz200;
	jump_thetal__ = (thl[qltop + 1] - thl[qltop]) / dz_ent__ * dz200;
	ki = qltop;
	if (singlelayer) {
	    refthl = thl[ki];
	    refthlv = refthl * (rc0003 + p608 * qv[ki]);
	    refthv = thv[ki];
	    refthvm1 = thv[ki - 1];
	    reftk = tk[ki];
	    refqt = qt[ki];
	    refqc = qc[ki];
	    refqi = qi[ki];
	    refqnc = qnc[ki];
	    refqni = qni[ki];
	    refqnwfa = qnwfa[ki];
	    refqnifa = qnifa[ki];
	    refu = u[ki];
	    refv = v[ki];
	    refqke = qke[ki];
	    refp = p[ki];
	} else {
	    refthl = (thl[ki - 1] * dz[ki] + thl[ki] * dz[ki - 1]) / (dz[ki] 
		    + dz[ki - 1]);
	    refthlv = refthl * (rc0003 + p608 * (qv[ki - 1] * dz[ki] + qv[ki] 
		    * dz[ki - 1]) / (dz[ki] + dz[ki - 1]));
	    refthv = (thv[ki - 1] * dz[ki] + thv[ki] * dz[ki - 1]) / (dz[ki] 
		    + dz[ki - 1]);
	    refthvm1 = (thv[ki - 2] * dz[ki - 1] + thv[ki - 1] * dz[ki - 2]) /
		     (dz[ki - 1] + dz[ki - 2]);
	    reftk = (tk[ki - 1] * dz[ki] + tk[ki] * dz[ki - 1]) / (dz[ki] + 
		    dz[ki - 1]);
	    refqt = (qt[ki - 1] * dz[ki] + qt[ki] * dz[ki - 1]) / (dz[ki] + 
		    dz[ki - 1]);
	    refqc = (qc[ki - 1] * dz[ki] + qc[ki] * dz[ki - 1]) / (dz[ki] + 
		    dz[ki - 1]);
	    refqi = (qi[ki - 1] * dz[ki] + qi[ki] * dz[ki - 1]) / (dz[ki] + 
		    dz[ki - 1]);
	    refqnc = (qnc[ki - 1] * dz[ki] + qnc[ki] * dz[ki - 1]) / (dz[ki] 
		    + dz[ki - 1]);
	    refqni = (qni[ki - 1] * dz[ki] + qni[ki] * dz[ki - 1]) / (dz[ki] 
		    + dz[ki - 1]);
	    refqnwfa = (qnwfa[ki - 1] * dz[ki] + qnwfa[ki] * dz[ki - 1]) / (
		    dz[ki] + dz[ki - 1]);
	    refqnifa = (qnifa[ki - 1] * dz[ki] + qnifa[ki] * dz[ki - 1]) / (
		    dz[ki] + dz[ki - 1]);
	    refu = (u[ki - 1] * dz[ki] + u[ki] * dz[ki - 1]) / (dz[ki] + dz[
		    ki - 1]);
	    refv = (v[ki - 1] * dz[ki] + v[ki] * dz[ki - 1]) / (dz[ki] + dz[
		    ki - 1]);
	    refqke = (qke[ki - 1] * dz[ki] + qke[ki] * dz[ki - 1]) / (dz[ki] 
		    + dz[ki - 1]);
	    refp = (p[ki - 1] * dz[ki] + p[ki] * dz[ki - 1]) / (dz[ki] + dz[
		    ki - 1]);
	}
	d__1 =  (grav * dz200 * f0 / (refthl * rho[ki] * cp));
	d__2 =  onethird;
	wstar_rad__ = rc0014 * (real)pow_dd(&d__1, &d__2);
/* Computing MIN */
	r__1 = max(wstar_rad__,rc0010);
	wstar_rad__ = min(r__1,rc0018);
/* Computing 3rd power */
	r__1 = wstar_rad__;
	went = thv[1] / (grav * jump_thv__ * dz200) * (rc0017 * (r__1 * (r__1 
		* r__1)));
	qstar = (r__1 = went * jump_qt__ / wstar_rad__, abs(r__1));
	thstar = f0 / rho[ki] / cp / wstar_rad__ - went * jump_thv__ / 
		wstar_rad__;
	sigmaw = rc0012 * wstar_rad__;
	sigmaqt = rc0019 * qstar;
	sigmath = rc0009 * thstar;
	pwmin = -rc0003;
	pwmax = -rc0020;
/* Computing MAX */
/* Computing MIN */
	r__3 = sigmaw * pwmin, r__4 = -rc0010;
	r__1 = min(r__3,r__4), r__2 = -rc0021;
	wmin = max(r__1,r__2);
/* Computing MAX */
/* Computing MIN */
	r__3 = sigmaw * pwmax, r__4 = -rc0021;
	r__1 = min(r__3,r__4), r__2 = -rc0017;
	wmax = max(r__1,r__2);
/* Computing MIN */
	r__1 = -rc0022, r__2 = rc0017 * (refthvm1 - refthv);
	minexc = min(r__1,r__2);
/* Computing MIN */
	r__1 = minexc - rc0010, r__2 = -rc0017;
	maxexc = min(r__1,r__2);
/* Computing MAX */
	r__1 = maxexc, r__2 = thv[1] - refthlv;
	maxexc = max(r__1,r__2);
/* Computing MAX */
/* Computing MIN */
	r__2 = rc0022 * (-rc0012) * sigmath / sigmaw;
	r__1 = min(r__2,minexc);
	def_th__ = max(r__1,maxexc);
	qs1 = (real)qsat_blend__(m471_work,m471_n, &reftk, &refp);
	r__1 = reftk + def_th__;
	qs2 = (real)qsat_blend__(m471_work,m471_n, &r__1, &refp);
	refqt2 = refqt * qs2 / qs1;
/* Computing MIN */
	r__1 = qs2 - qs1, r__2 = -rc0023 * (refqc + refqi);
	def_qt__ = min(r__1,r__2);
	if (refqc + refqi > zero) {
	    frac_liq__ = refqc / (refqc + refqi);
	    frac_ice__ = one - frac_liq__;
	} else {
	    frac_liq__ = zero;
	    frac_ice__ = zero;
	}
	if (FALSE_) {
	}
	
#pragma unroll 1
for (i__ = 1; i__ <= 5; ++i__) {
	    ki = dd_initk__[i__ - 1];
	    downw[ki + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = wmin - (abs(wmax) - abs(wmin)) / 4.f *
		     (i__ - 1);
	    downa[ki + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = add / 5.f;
	    downu[ki + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = refu;
	    downv[ki + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = refv;
/* Computing MAX */
	    r__1 = refqc + def_qt__ * frac_liq__;
	    downqc[ki + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = max(r__1,zero);
/* Computing MAX */
	    r__1 = refqi + def_qt__ * frac_ice__;
	    downqi[ki + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = max(r__1,zero);
	    downqnc[ki + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = refqnc;
	    downqni[ki + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = refqni;
	    downqnwfa[ki + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = refqnwfa;
	    downqnifa[ki + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = refqnifa;
	    downqt[ki + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = refqt + def_qt__;
	    downthv[ki + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = refthv + def_th__ * (real) i__ / 
		    5.f;
	    downthl[ki + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = refthl + def_th__ * (real) i__ / 
		    5.f;
/* L9016: */
	}
/* L9017: */
	if (FALSE_) {
	}
	
#pragma unroll 1
for (i__ = 1; i__ <= 5; ++i__) {
	    ddd = minddd + dl * (real) (i__ - 1);
	    thvn = downthv[ki + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    qcn = downqc[ki + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    qin = downqi[ki + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    i__3 = *kts + 1;
	    
#pragma unroll 1
for (k = dd_initk__[i__ - 1] - 1; k >= i__3; --k) {
		wmin = rc0010 + ddd * rc0024;
/* Computing MIN */
/* Computing MAX */
		r__3 = (r__1 = downw[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)], abs(r__1));
		r__2 = max(r__3,wmin);
		ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = rc0025 / (min(r__2,rc0026) * ddd);
/* Computing MAX */
		r__1 = ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = max(r__1,rc0027);
/* Computing MAX */
		r__1 = ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)], r__2 = rc0028 / zw[k];
		ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = max(r__1,r__2);
/* Computing MIN */
		r__1 = ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)], r__2 = rc0029 / dz[k];
		ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = min(r__1,r__2);
		entexp = ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * dz[k];
		entexm = ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * dz[k] * rc0017;
		qtn = downqt[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0003 - entexp) + qt[
			k] * entexp;
		thln = downthl[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0003 - entexp) + 
			thl[k] * entexp;
		qncn = downqnc[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0003 - entexp) + 
			qnc[k] * entexp;
		qnin = downqni[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0003 - entexp) + 
			qni[k] * entexp;
		qnwfan = downqnwfa[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0003 - entexp) 
			+ qnwfa[k] * entexp;
		qnifan = downqnifa[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0003 - entexp) 
			+ qnifa[k] * entexp;
		un = downu[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0003 - entexm) + u[k] *
			 entexm;
		vn = downv[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (rc0003 - entexm) + v[k] *
			 entexm;
		pk = (p[k - 1] * dz[k] + p[k] * dz[k - 1]) / (dz[k] + dz[k - 
			1]);
		condensation_ddmf__(m471_work,m471_n, &qtn, &thln, &pk, &zw[k], &thvn, &qcn, &
			qin, &c__0);
		thvk = (thv[k - 1] * dz[k] + thv[k] * dz[k - 1]) / (dz[k] + 
			dz[k - 1]);
		buoy = grav * (thvn / thvk - one);
		if (buoy > zero) {
		    bcoeff = rc0021;
		    ecoeff = -rc0030;
		} else {
/* Computing MIN */
		    r__1 = zw[k] / *pblh;
		    bcoeff = rc0021 * min(r__1,one);
/* Computing MAX */
		    r__1 = rc0017 * *pblh - zw[k];
		    ecoeff = -rc0030 - max(r__1,zero) / (rc0017 * *pblh);
/* Computing MIN */
		    r__1 = zw[k] / *pblh;
		    buoy *= min(r__1,one);
		}
		if (k == *kts + 2) {
		    buoy = max(buoy,rc0013);
		}
		if (k == *kts + 1) {
		    buoy = max(buoy,rc0022);
		}
/* Computing MIN */
		r__1 = downw[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)], r__2 = -rc0021;
		mindownw = min(r__1,r__2);
/* Computing MIN */
		r__1 = dz[k];
		wn = downw[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + (ecoeff * ent[k + i__ * 
			(1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] - bcoeff * 
			buoy / mindownw) * min(r__1,rc0031);
/* Computing MIN */
		r__1 = rc0032 * dz[k] / rc0002;
		if (wn < downw[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] - min(r__1,rc0030)) {
/* Computing MIN */
		    r__1 = rc0032 * dz[k] / rc0002;
		    wn = downw[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] - min(r__1,rc0030);
		}
/* Computing MIN */
		r__1 = rc0032 * dz[k] / rc0002;
		if (wn > downw[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + min(r__1,rc0030)) {
/* Computing MIN */
		    r__1 = rc0032 * dz[k] / rc0002;
		    wn = downw[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + min(r__1,rc0030);
		}
/* Computing MAX */
		r__1 = min(wn,zero), r__2 = -rc0018;
		wn = max(r__1,r__2);
		if (FALSE_) {
		}
		if (wn < rc0033) {
		    downw[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = wn;
		    downthv[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = thvn;
		    downthl[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = thln;
		    downqt[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qtn;
		    downqc[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qcn;
		    downqi[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qin;
		    downqnc[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qncn;
		    downqni[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qnin;
		    downqnwfa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qnwfan;
		    downqnifa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qnifan;
		    downu[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = un;
		    downv[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = vn;
		    downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = downa[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		} else {
		    goto L9023;
		}
/* L9022: */
	    }
L9023:
/* L9019: */
	    ;
	}
/* L9020: */
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= 5; ++ixarr1) {
	downw[ixarr1 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = rc0033;
    }
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= 5; ++ixarr1) {
	downa[ixarr1 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = rc0033;
    }
    i__3 = *kts;
    
#pragma unroll 1
for (k = qltop; k >= i__3; --k) {
	
#pragma unroll 1
for (i__ = 1; i__ <= 5; ++i__) {
	    edmf_a_dd__[k] += downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    edmf_w_dd__[k] += downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[k + i__ * (1*(MYNN471_NZ-49)+50) - 
		    (1*(MYNN471_NZ-49)+51)];
	    edmf_qt_dd__[k] += downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqt[k + i__ * (1*(MYNN471_NZ-49)+50) 
		    - (1*(MYNN471_NZ-49)+51)];
	    edmf_thl_dd__[k] += downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downthl[k + i__ * 
		    (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    edmf_ent_dd__[k] += downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * ent[k + i__ * (1*(MYNN471_NZ-49)+50) - 
		    (1*(MYNN471_NZ-49)+51)];
	    edmf_qc_dd__[k] += downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqc[k + i__ * (1*(MYNN471_NZ-49)+50) 
		    - (1*(MYNN471_NZ-49)+51)];
/* L9028: */
	}
/* L9029: */
	if (edmf_a_dd__[k] > rc0033) {
	    edmf_w_dd__[k] /= edmf_a_dd__[k];
	    edmf_qt_dd__[k] /= edmf_a_dd__[k];
	    edmf_thl_dd__[k] /= edmf_a_dd__[k];
	    edmf_ent_dd__[k] /= edmf_a_dd__[k];
	    edmf_qc_dd__[k] /= edmf_a_dd__[k];
	}
/* Computing 3rd power */
	r__2 = (r__1 = edmf_w_dd__[k], abs(r__1));
/* Computing MAX */
	r__3 = el[k];
	tkeprod_dn__[k] = r__2 * (r__2 * r__2) * edmf_a_dd__[k] / (b1 * max(
		r__3,rc0021));
/* L9025: */
    }
/* L9026: */
/* Computing MAX */
    r__1 = el[qltop + 1];
    tkeprod_dn__[qltop + 1] = went * edmf_a_dd__[qltop] / (b1 * max(r__1,
	    rc0010));
    i__3 = qltop;
    
#pragma unroll 1
for (k = *kts; k <= i__3; ++k) {
	rho_int__ = (rho[k] * dz[k + 1] + rho[k + 1] * dz[k]) / (dz[k + 1] + 
		dz[k]);
	
#pragma unroll 1
for (i__ = 1; i__ <= 5; ++i__) {
	    sd_aw__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (r__1 = 
		    downw[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)], abs(r__1));
	    sd_awthl__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[k + 
		    i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downthl[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    sd_awqt__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[k + 
		    i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqt[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    sd_awqc__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[k + 
		    i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqc[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    sd_awqi__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[k + 
		    i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqi[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    sd_awqnc__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[k + 
		    i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqnc[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    sd_awqni__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[k + 
		    i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqni[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    sd_awqnwfa__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[k 
		    + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqnwfa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    sd_awqnifa__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[k 
		    + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqnifa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    sd_awu__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[k + 
		    i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downu[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    sd_awv__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[k + 
		    i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downv[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
/* L9034: */
	}
/* L9035: */
	sd_awqv__[k] = sd_awqt__[k] - sd_awqc__[k] - sd_awqi__[k];
/* L9031: */
    }
/* L9032: */
    return 0;
} /* ddmp_mf__ */

__device__ M471_CALL /* Subroutine */ int scale_aware__(unsigned char *m471_work, int m471_n, M471Ptr<real> dx, M471Ptr<real> pblh, M471Ptr<real> psig_bl__, 
	M471Ptr<real> psig_shcu__) {
    /* Initialized data */

    real zero = 0.f;
    real one = 1.f;
    real rc0000 = 2.5f;
    real rc0001 = 10.f;
    real rc0002 = 3e3f;
    real rc0003 = .106f;
    real rc0004 = .667f;
    real rc0005 = .066f;
    real rc0006 = .071f;
    real rc0007 = 500.f;
    real rc0008 = 3500.f;
    real rc0009 = .145f;
    real rc0010 = .172f;
    real rc0011 = .17f;

    /* System generated locals */
    real r__1, r__2;
    doublereal d__1, d__2, d__3, d__4;

    /* Builtin functions */
    

    /* Local variables */
    real dxdh;

    *psig_bl__ = one;
    *psig_shcu__ = one;
/* Computing MAX */
    r__1 = rc0000 * *dx;
    dxdh = max(r__1,rc0001) / min(*pblh,rc0002);
/* Computing 2nd power */
    r__1 = dxdh;
    d__1 =  dxdh;
    d__2 =  rc0004;
/* Computing 2nd power */
    r__2 = dxdh;
    d__3 =  dxdh;
    d__4 =  rc0004;
    *psig_bl__ = (r__1 * r__1 + rc0003 * (real)pow_dd(&d__1, &d__2)) / (r__2 *
	     r__2 + rc0005 * (real)pow_dd(&d__3, &d__4) + rc0006);
/* Computing MAX */
    r__1 = rc0000 * *dx;
/* Computing MIN */
    r__2 = *pblh + rc0007;
    dxdh = max(r__1,rc0001) / min(r__2,rc0008);
/* Computing 2nd power */
    r__1 = dxdh;
    d__1 =  dxdh;
    d__2 =  rc0004;
/* Computing 2nd power */
    r__2 = dxdh;
    d__3 =  dxdh;
    d__4 =  rc0004;
    *psig_shcu__ = (r__1 * r__1 + rc0009 * (real)pow_dd(&d__1, &d__2)) / (
	    r__2 * r__2 + rc0010 * (real)pow_dd(&d__3, &d__4) + rc0011);
    if (*psig_bl__ > one) {
	*psig_bl__ = one;
    }
    if (*psig_bl__ < zero) {
	*psig_bl__ = zero;
    }
    if (*psig_shcu__ > one) {
	*psig_shcu__ = one;
    }
    if (*psig_shcu__ < zero) {
	*psig_shcu__ = zero;
    }
    return 0;
} /* scale_aware__ */

__device__ M471_CALL real esat_blend__(unsigned char *m471_work, int m471_n, M471Ptr<real> t) {
    /* Initialized data */

    real tice = 240.f;
    real t0c = 273.149993896484375f;
    real rc0000 = 611.583699f;
    real rc0001 = 44.4606896f;
    real rc0002 = 1.43177157f;
    real rc0003 = .0264224321f;
    real rc0004 = 2.99291081e-4f;
    real rc0005 = 2.03154182e-6f;
    real rc0006 = 7.02620698e-9f;
    real rc0007 = 3.7953431e-12f;
    real rc0008 = 3.21582393e-14f;
    real rc0009 = 609.868993f;
    real rc0010 = 49.9320233f;
    real rc0011 = 1.84672631f;
    real rc0012 = .0402737184f;
    real rc0013 = 5.65392987e-4f;
    real rc0014 = 5.21693933e-6f;
    real rc0015 = 3.07839583e-8f;
    real rc0016 = 1.0578516e-10f;
    real rc0017 = 1.61444444e-13f;
    real rc0018 = 80.f;
    real rc0019 = 6.f;
    real rc0020 = 1.f;

    /* System generated locals */
    real ret_val, r__1, r__2;

    /* Local variables */
    real j0, j1, j2, j3, j4, j5, j6, j7, j8, k0, k1, k2, k3, k4, k5, k6, k7, 
	    k8, xc, chi, esi, esl;

    j0 = rc0000;
    j1 = rc0001;
    j2 = rc0002;
    j3 = rc0003;
    j4 = rc0004;
    j5 = rc0005;
    j6 = rc0006;
    j7 = rc0007;
    j8 = -rc0008;
    k0 = rc0009;
    k1 = rc0010;
    k2 = rc0011;
    k3 = rc0012;
    k4 = rc0013;
    k5 = rc0014;
    k6 = rc0015;
    k7 = rc0016;
    k8 = rc0017;
/* Computing MAX */
    r__1 = -rc0018, r__2 = *t - t0c;
    xc = max(r__1,r__2);
    if (*t >= t0c - rc0019) {
	ret_val = j0 + xc * (j1 + xc * (j2 + xc * (j3 + xc * (j4 + xc * (j5 + 
		xc * (j6 + xc * (j7 + xc * j8)))))));
    } else if (*t <= tice) {
	ret_val = k0 + xc * (k1 + xc * (k2 + xc * (k3 + xc * (k4 + xc * (k5 + 
		xc * (k6 + xc * (k7 + xc * k8)))))));
    } else {
	esl = j0 + xc * (j1 + xc * (j2 + xc * (j3 + xc * (j4 + xc * (j5 + xc *
		 (j6 + xc * (j7 + xc * j8)))))));
	esi = k0 + xc * (k1 + xc * (k2 + xc * (k3 + xc * (k4 + xc * (k5 + xc *
		 (k6 + xc * (k7 + xc * k8)))))));
	chi = (t0c - rc0019 - *t) / (t0c - rc0019 - tice);
	ret_val = (rc0020 - chi) * esl + chi * esi;
    }
    return ret_val;
} /* esat_blend__ */

__device__ M471_CALL real qsat_blend__(unsigned char *m471_work, int m471_n, M471Ptr<real> t, M471Ptr<real> p) {
    /* Initialized data */

    real tice = 240.f;
    real t0c = 273.149993896484375f;
    real rc0000 = 611.583699f;
    real rc0001 = 44.4606896f;
    real rc0002 = 1.43177157f;
    real rc0003 = .0264224321f;
    real rc0004 = 2.99291081e-4f;
    real rc0005 = 2.03154182e-6f;
    real rc0006 = 7.02620698e-9f;
    real rc0007 = 3.7953431e-12f;
    real rc0008 = 3.21582393e-14f;
    real rc0009 = 609.868993f;
    real rc0010 = 49.9320233f;
    real rc0011 = 1.84672631f;
    real rc0012 = .0402737184f;
    real rc0013 = 5.65392987e-4f;
    real rc0014 = 5.21693933e-6f;
    real rc0015 = 3.07839583e-8f;
    real rc0016 = 1.0578516e-10f;
    real rc0017 = 1.61444444e-13f;
    real rc0018 = 80.f;
    real rc0019 = 6.f;
    real rc0020 = .15f;
    real rc0021 = .622f;
    real rc0022 = 1e-5f;
    real rc0023 = 1.f;

    /* System generated locals */
    real ret_val, r__1, r__2;

    /* Local variables */
    real j0, j1, j2, j3, j4, j5, j6, j7, j8, k0, k1, k2, k3, k4, k5, k6, k7, 
	    k8, xc, chi, esi, esl, rsif, rslf;

    j0 = rc0000;
    j1 = rc0001;
    j2 = rc0002;
    j3 = rc0003;
    j4 = rc0004;
    j5 = rc0005;
    j6 = rc0006;
    j7 = rc0007;
    j8 = -rc0008;
    k0 = rc0009;
    k1 = rc0010;
    k2 = rc0011;
    k3 = rc0012;
    k4 = rc0013;
    k5 = rc0014;
    k6 = rc0015;
    k7 = rc0016;
    k8 = rc0017;
/* Computing MAX */
    r__1 = -rc0018, r__2 = *t - t0c;
    xc = max(r__1,r__2);
    if (*t >= t0c - rc0019) {
	esl = j0 + xc * (j1 + xc * (j2 + xc * (j3 + xc * (j4 + xc * (j5 + xc *
		 (j6 + xc * (j7 + xc * j8)))))));
/* Computing MIN */
	r__1 = esl, r__2 = *p * rc0020;
	esl = min(r__1,r__2);
/* Computing MAX */
	r__1 = *p - esl;
	ret_val = rc0021 * esl / max(r__1,rc0022);
    } else if (*t <= tice) {
	esi = k0 + xc * (k1 + xc * (k2 + xc * (k3 + xc * (k4 + xc * (k5 + xc *
		 (k6 + xc * (k7 + xc * k8)))))));
/* Computing MIN */
	r__1 = esi, r__2 = *p * rc0020;
	esi = min(r__1,r__2);
/* Computing MAX */
	r__1 = *p - esi;
	ret_val = rc0021 * esi / max(r__1,rc0022);
    } else {
	esl = j0 + xc * (j1 + xc * (j2 + xc * (j3 + xc * (j4 + xc * (j5 + xc *
		 (j6 + xc * (j7 + xc * j8)))))));
/* Computing MIN */
	r__1 = esl, r__2 = *p * rc0020;
	esl = min(r__1,r__2);
	esi = k0 + xc * (k1 + xc * (k2 + xc * (k3 + xc * (k4 + xc * (k5 + xc *
		 (k6 + xc * (k7 + xc * k8)))))));
/* Computing MIN */
	r__1 = esi, r__2 = *p * rc0020;
	esi = min(r__1,r__2);
/* Computing MAX */
	r__1 = *p - esl;
	rslf = rc0021 * esl / max(r__1,rc0022);
/* Computing MAX */
	r__1 = *p - esi;
	rsif = rc0021 * esi / max(r__1,rc0022);
	chi = (t0c - rc0019 - *t) / (t0c - rc0019 - tice);
	ret_val = (rc0023 - chi) * rslf + chi * rsif;
    }
    return ret_val;
} /* qsat_blend__ */

__device__ M471_CALL real xl_blend__(unsigned char *m471_work, int m471_n, M471Ptr<real> t) {
    /* Initialized data */

    real xlv = 2.5e6f;
    real xls = 2.85e6f;
    real cpv = 1846.4000244140625f;
    real cliq = 4190.f;
    real cice = 2106.f;
    real tice = 240.f;
    real t0c = 273.149993896484375f;
    real rc0000 = 1.f;

    /* System generated locals */
    real ret_val;

    /* Local variables */
    real chi, xlst, xlvt;

    if (*t >= t0c) {
	ret_val = xlv + (cpv - cliq) * (*t - t0c);
    } else if (*t <= tice) {
	ret_val = xls + (cpv - cice) * (*t - t0c);
    } else {
	xlvt = xlv + (cpv - cliq) * (*t - t0c);
	xlst = xls + (cpv - cice) * (*t - t0c);
	chi = (t0c - *t) / (t0c - tice);
	ret_val = (rc0000 - chi) * xlvt + chi * xlst;
    }
    return ret_val;
} /* xl_blend__ */

__device__ M471_CALL real phim_(unsigned char *m471_work, int m471_n, M471Ptr<real> zet) {
    /* Initialized data */

    real zero = 0.f;
    real one = 1.f;
    real cphm_unst__ = 16.f;
    real rc0000 = 6.1f;
    real rc0001 = 2.5f;
    real rc0002 = 1.f;
    real rc0003 = 5.3f;
    real rc0004 = 1.1f;
    real rc0005 = 10.f;
    real rc0006 = 34.f;
    real rc0007 = .25f;
    real rc0008 = 2.f;
    real rc0009 = .5f;
    real rc0010 = 1.570796f;
    real rc0011 = .333333f;
    real rc0012 = .33333f;
    real rc0013 = .6666667f;
    real rc0014 = .3333f;
    real rc0015 = .57735f;
    real rc0016 = 1.1547f;
    real rc0017 = 1.5f;
    real rc0018 = 1.73205f;
    real rc0019 = 1.813799364f;

    /* System generated locals */
    real ret_val, r__1;
    doublereal d__1, d__2, d__3, d__4;

    /* Builtin functions */
    

    /* Local variables */
    real dummy_psi__, phi_m__, ah_st__, bh_st__, am_st__, bm_st__, rbh_st__, 
	    rbm_st__, dummy_0__, dummy_1__, dummy_2__, dummy_3__, dummy_4__, 
	    dummy_11__, dummy_22__, dummy_33__, ah_unst__, dummy_44__, 
	    am_unst__;

    am_st__ = rc0000;
    bm_st__ = rc0001;
    rbm_st__ = rc0002 / bm_st__;
    ah_st__ = rc0003;
    bh_st__ = rc0004;
    rbh_st__ = rc0002 / bh_st__;
    am_unst__ = rc0005;
    ah_unst__ = rc0006;
    if (*zet >= zero) {
	d__1 =  (*zet);
	d__2 =  bm_st__;
	dummy_0__ = (real)pow_dd(&d__1, &d__2) + 1;
	d__1 =  dummy_0__;
	d__2 =  rbm_st__;
	dummy_1__ = *zet + (real)pow_dd(&d__1, &d__2);
	d__1 =  dummy_0__;
	d__2 =  (rbm_st__ - 1);
	d__3 =  (*zet);
	d__4 =  (bm_st__ - 1);
	dummy_11__ = (real)pow_dd(&d__1, &d__2) * (real)pow_dd(&d__3, &d__4) 
		+ 1;
	dummy_2__ = -am_st__ / dummy_1__ * dummy_11__;
	phi_m__ = 1 - *zet * dummy_2__;
    } else {
	d__1 =  (one - cphm_unst__ * *zet);
	d__2 =  rc0007;
	dummy_0__ = (real)pow_dd(&d__1, &d__2);
	phi_m__ = rc0002 / dummy_0__;
/* Computing 2nd power */
	r__1 = dummy_0__;
	dummy_psi__ = rc0008 * (real)log((real)(rc0009 * (rc0002 + dummy_0__))
		) + (real)log((real)(rc0009 * (rc0002 + r__1 * r__1))) - 
		rc0008 * (real)atan(dummy_0__) + rc0010;
	dummy_0__ = rc0002 - am_unst__ * *zet;
	d__1 =  dummy_0__;
	d__2 =  rc0011;
	dummy_1__ = (real)pow_dd(&d__1, &d__2);
	d__1 =  dummy_0__;
	d__2 =  (-rc0013);
	dummy_11__ = -rc0012 * am_unst__ * (real)pow_dd(&d__1, &d__2);
	d__1 =  dummy_1__;
	d__2 =  rc0008;
	dummy_2__ = rc0012 * ((real)pow_dd(&d__1, &d__2) + dummy_1__ + rc0002)
		;
	dummy_22__ = rc0014 * dummy_11__ * (rc0008 * dummy_1__ + rc0002);
	dummy_3__ = rc0015 * (rc0008 * dummy_1__ + rc0002);
	dummy_33__ = rc0016 * dummy_11__;
	dummy_4__ = rc0017 * (real)log(dummy_2__) - rc0018 * (real)atan(
		dummy_3__) + rc0019;
/* Computing 2nd power */
	r__1 = dummy_3__;
	dummy_44__ = rc0017 / dummy_2__ * dummy_22__ - rc0018 * dummy_33__ / (
		rc0002 + r__1 * r__1);
/* Computing 2nd power */
	r__1 = *zet;
	dummy_0__ = r__1 * r__1;
	dummy_1__ = rc0002 / (rc0002 + dummy_0__);
	dummy_11__ = rc0008 * *zet;
	dummy_2__ = ((1 - phi_m__) / *zet + dummy_11__ * dummy_4__ + 
		dummy_0__ * dummy_44__) * dummy_1__;
/* Computing 2nd power */
	r__1 = dummy_1__;
	dummy_22__ = -dummy_11__ * (dummy_psi__ + dummy_0__ * dummy_4__) * (
		r__1 * r__1);
	phi_m__ = rc0002 - *zet * (dummy_2__ + dummy_22__);
    }
    ret_val = phi_m__;
    return ret_val;
} /* phim_ */

__device__ M471_CALL real phih_(unsigned char *m471_work, int m471_n, M471Ptr<real> zet) {
    /* Initialized data */

    real zero = 0.f;
    real one = 1.f;
    real cphh_unst__ = 16.f;
    real rc0000 = 6.1f;
    real rc0001 = 2.5f;
    real rc0002 = 1.f;
    real rc0003 = 5.3f;
    real rc0004 = 1.1f;
    real rc0005 = 10.f;
    real rc0006 = 34.f;
    real rc0007 = .5f;
    real rc0008 = 2.f;
    real rc0009 = .333333f;
    real rc0010 = .33333f;
    real rc0011 = .6666667f;
    real rc0012 = .3333f;
    real rc0013 = .57735f;
    real rc0014 = 1.1547f;
    real rc0015 = 1.5f;
    real rc0016 = 1.73205f;
    real rc0017 = 1.813799364f;

    /* System generated locals */
    real ret_val, r__1;
    doublereal d__1, d__2, d__3, d__4;

    /* Builtin functions */
    

    /* Local variables */
    real dummy_psi__, phh, ah_st__, bh_st__, am_st__, bm_st__, rbh_st__, 
	    rbm_st__, dummy_0__, dummy_1__, dummy_2__, dummy_3__, dummy_4__, 
	    dummy_11__, dummy_22__, dummy_33__, ah_unst__, dummy_44__, 
	    am_unst__;

    am_st__ = rc0000;
    bm_st__ = rc0001;
    rbm_st__ = rc0002 / bm_st__;
    ah_st__ = rc0003;
    bh_st__ = rc0004;
    rbh_st__ = rc0002 / bh_st__;
    am_unst__ = rc0005;
    ah_unst__ = rc0006;
    if (*zet >= zero) {
	d__1 =  (*zet);
	d__2 =  bh_st__;
	dummy_0__ = (real)pow_dd(&d__1, &d__2) + 1;
	d__1 =  dummy_0__;
	d__2 =  rbh_st__;
	dummy_1__ = *zet + (real)pow_dd(&d__1, &d__2);
	d__1 =  dummy_0__;
	d__2 =  (rbh_st__ - 1);
	d__3 =  (*zet);
	d__4 =  (bh_st__ - 1);
	dummy_11__ = (real)pow_dd(&d__1, &d__2) * (real)pow_dd(&d__3, &d__4) 
		+ 1;
	dummy_2__ = -ah_st__ / dummy_1__ * dummy_11__;
	ret_val = 1 - *zet * dummy_2__;
    } else {
	d__1 =  (one - cphh_unst__ * *zet);
	d__2 =  rc0007;
	dummy_0__ = (real)pow_dd(&d__1, &d__2);
	phh = one / dummy_0__;
	dummy_psi__ = rc0008 * (real)log((real)(rc0007 * (rc0002 + dummy_0__))
		);
	dummy_0__ = one - ah_unst__ * *zet;
	d__1 =  dummy_0__;
	d__2 =  rc0009;
	dummy_1__ = (real)pow_dd(&d__1, &d__2);
	d__1 =  dummy_0__;
	d__2 =  (-rc0011);
	dummy_11__ = -rc0010 * ah_unst__ * (real)pow_dd(&d__1, &d__2);
	d__1 =  dummy_1__;
	d__2 =  rc0008;
	dummy_2__ = rc0010 * ((real)pow_dd(&d__1, &d__2) + dummy_1__ + one);
	dummy_22__ = rc0012 * dummy_11__ * (rc0008 * dummy_1__ + one);
	dummy_3__ = rc0013 * (rc0008 * dummy_1__ + one);
	dummy_33__ = rc0014 * dummy_11__;
	dummy_4__ = rc0015 * (real)log(dummy_2__) - rc0016 * (real)atan(
		dummy_3__) + rc0017;
/* Computing 2nd power */
	r__1 = dummy_3__;
	dummy_44__ = rc0015 / dummy_2__ * dummy_22__ - rc0016 * dummy_33__ / (
		rc0002 + r__1 * r__1);
/* Computing 2nd power */
	r__1 = *zet;
	dummy_0__ = r__1 * r__1;
	dummy_1__ = one / (one + dummy_0__);
	dummy_11__ = rc0008 * *zet;
	dummy_2__ = ((1 - phh) / *zet + dummy_11__ * dummy_4__ + dummy_0__ * 
		dummy_44__) * dummy_1__;
/* Computing 2nd power */
	r__1 = dummy_1__;
	dummy_22__ = -dummy_11__ * (dummy_psi__ + dummy_0__ * dummy_4__) * (
		r__1 * r__1);
	ret_val = rc0002 - *zet * (dummy_2__ + dummy_22__);
    }
    return ret_val;
} /* phih_ */

__device__ M471_CALL /* Subroutine */ int topdown_cloudrad__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dz1,
	 M471Ptr<real> zw, M471Ptr<real> fltv, M471Ptr<real> xland, M471Ptr<integer> kpbl, M471Ptr<real> pblh, M471Ptr<real> sqc, M471Ptr<real> sqi, M471Ptr<real> sqw, M471Ptr<real> thl, M471Ptr<real> th1, M471Ptr<real> ex1, M471Ptr<real> p1, 
	M471Ptr<real> rho1, M471Ptr<real> thv, M471Ptr<real> cldfra_bl1__, M471Ptr<real> rthraten, M471Ptr<real> maxkhtopdown, M471Ptr<real> khtopdown, M471Ptr<real> tkeprodtd) {
    /* Initialized data */

    real karman = .4000000059604644775390625f;
    real cp = 1004.5f;
    real r_d__ = 287.f;
    real xlv = 2.5e6f;
    real p608 = .6083624362945556640625f;
    real ep_2__ = .621750414371490478515625f;
    real zero = 0.f;
    real onethird = .3333333432674407958984375f;
    real twothirds = .666666686534881591796875f;
    real grav = 9.81000041961669921875f;
    real xlvcp = 2488.80029296875f;
    real rc0000 = 2.f;
    real rc0001 = .01f;
    real rc0002 = 8.f;
    real rc0003 = 100.f;
    real rc0004 = 1e-6f;
    real rc0005 = .5f;
    real rc0006 = 6.112f;
    real rc0007 = 17.67f;
    real rc0008 = 273.16f;
    real rc0009 = 29.65f;
    real rc0010 = 273.15f;
    real rc0011 = 0.f;
    real rc0012 = .1f;
    real rc0013 = .2f;
    real rc0014 = 8.f;
    real rc0015 = 1.5f;
    real rc0016 = 90.f;
    real rc0017 = .25f;
    real rc0018 = 30.f;
    real rc0019 = 1500.f;
    real rc0020 = 1.f;
    real rc0021 = 10.f;
    real rc0022 = 2.f;
    real rc0023 = 3.402823466e38f;

    /* System generated locals */
    integer dz1_offset, sqc_offset, sqi_offset, sqw_offset, thl_offset, 
	    th1_offset, ex1_offset, p1_offset, rho1_offset, thv_offset, 
	    cldfra_bl1_offset, rthraten_offset, zw_offset, khtopdown_offset, 
	    tkeprodtd_offset, i__1, i__2, i__3;
    real r__1, r__2;
    doublereal d__1, d__2;

    /* Builtin functions */
    

    /* Local variables */
    logical cloudflg;
    integer k, kk;
    real we, wm2, wm3, zl1, bfx0, tmp1, pfac, rvls, red001, rcldb, 
	    templ, zfmin, temps, dthvx;
M471Ptr<real> zfac=m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+73600),m471_n);
    integer ixarr1;
    real phifac, minrad, bfxpbl, radsum, ent_eff__;
    integer kminrad;
    real  radflux, zminrad;
M471Ptr<real> zfacent=m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+73800),m471_n);
M471Ptr<real> wscalek2=m471_array<real>(m471_work,(1184*(MYNN471_NZ-49)+74000),m471_n);

    /* Parameter adjustments */
    tkeprodtd_offset = *kts;
    tkeprodtd -= tkeprodtd_offset;
    khtopdown_offset = *kts;
    khtopdown -= khtopdown_offset;
    rthraten_offset = *kts;
    rthraten -= rthraten_offset;
    cldfra_bl1_offset = *kts;
    cldfra_bl1__ -= cldfra_bl1_offset;
    thv_offset = *kts;
    thv -= thv_offset;
    rho1_offset = *kts;
    rho1 -= rho1_offset;
    p1_offset = *kts;
    p1 -= p1_offset;
    ex1_offset = *kts;
    ex1 -= ex1_offset;
    th1_offset = *kts;
    th1 -= th1_offset;
    thl_offset = *kts;
    thl -= thl_offset;
    sqw_offset = *kts;
    sqw -= sqw_offset;
    sqi_offset = *kts;
    sqi -= sqi_offset;
    sqc_offset = *kts;
    sqc -= sqc_offset;
    zw_offset = *kts;
    zw -= zw_offset;
    dz1_offset = *kts;
    dz1 -= dz1_offset;

    /* Function Body */
    pfac = rc0000;
    zfmin = rc0001;
    phifac = rc0002;
    cloudflg = FALSE_;
    minrad = rc0003;
    kminrad = *kpbl;
    zminrad = *pblh;
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	khtopdown[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	tkeprodtd[*kts + ixarr1 - 1] = zero;
    }
    *maxkhtopdown = zero;
/* Computing MAX */
    i__1 = 1, i__2 = *kpbl - 2;
    i__3 = *kpbl + 3;
    
#pragma unroll 1
for (kk = max(i__1,i__2); kk <= i__3; ++kk) {
	if (sqc[kk] > rc0004 || sqi[kk] > rc0004 || cldfra_bl1__[kk] > .5f) {
	    cloudflg = TRUE_;
	}
	if (rthraten[kk] < minrad) {
	    minrad = rthraten[kk];
	    kminrad = kk;
	    zminrad = zw[kk] + rc0005 * dz1[kk];
	}
/* L9004: */
    }
/* L9005: */
    if (max(kminrad,*kpbl) < 2) {
	cloudflg = FALSE_;
    }
    if (cloudflg) {
	zl1 = dz1[*kts];
/* Computing MAX */
	i__3 = *kpbl - 1, i__1 = kminrad - 1;
	k = max(i__3,i__1);
	templ = thl[k] * ex1[k];
	rvls = rc0003 * rc0006 * (real)exp((real)(rc0007 * (templ - rc0008) / 
		(templ - rc0009))) * (ep_2__ / p1[k + 1]);
/* Computing 2nd power */
	r__1 = templ;
	temps = templ + (sqw[k] - rvls) / (cp / xlv + ep_2__ * xlv * rvls / (
		r_d__ * (r__1 * r__1)));
	rvls = rc0003 * rc0006 * (real)exp((real)(rc0007 * (temps - rc0010) / 
		(temps - rc0009))) * (ep_2__ / p1[k + 1]);
/* Computing MAX */
	r__1 = sqw[k] - rvls;
	rcldb = max(r__1,rc0011);
	dthvx = thl[k + 2] + th1[k + 2] * p608 * sqw[k + 2] - (thl[k] + th1[k]
		 * p608 * sqw[k]);
	dthvx = max(dthvx,rc0012);
	tmp1 = xlvcp * rcldb / (ex1[k] * dthvx);
	ent_eff__ = rc0013 + rc0013 * rc0014 * tmp1;
	radsum = rc0011;
/* Computing MAX */
	i__3 = 1, i__1 = *kpbl - 3;
	i__2 = *kpbl + 3;
	
#pragma unroll 1
for (kk = max(i__3,i__1); kk <= i__2; ++kk) {
	    radflux = rthraten[kk] * ex1[kk];
	    radflux = radflux * cp / grav * (p1[kk] - p1[kk + 1]);
	    if (radflux < zero) {
		radsum = abs(radflux) + radsum;
	    }
/* L9007: */
	}
/* L9008: */
	if (*xland - rc0015 >= 0.f) {
	    radsum = min(radsum,rc0016);
/* Computing MAX */
	    r__1 = radsum / rho1[k] / cp;
	    bfx0 = max(r__1,zero);
	} else {
/* Computing MIN */
	    r__1 = rc0017 * radsum;
	    radsum = min(r__1,rc0018);
/* Computing MAX */
	    r__1 = radsum / rho1[k] / cp - max(*fltv,zero);
	    bfx0 = max(r__1,zero);
	}
	wm3 = grav / thv[k] * bfx0 * min(*pblh,rc0019);
	d__1 =  wm3;
	d__2 =  twothirds;
	wm2 += (real)pow_dd(&d__1, &d__2);
	bfxpbl = -ent_eff__ * bfx0;
/* Computing MAX */
	r__1 = thv[k + 1] - thv[k];
	dthvx = max(r__1,rc0012);
/* Computing MAX */
	d__1 =  wm3;
	d__2 =  twothirds;
	r__1 = bfxpbl / dthvx, r__2 = -(real)sqrt((real)((real)pow_dd(&d__1, &
		d__2)));
	we = max(r__1,r__2);
	i__2 = *kpbl + 3;
	
#pragma unroll 1
for (kk = *kts; kk <= i__2; ++kk) {
/* Computing MIN */
/* Computing MAX */
	    r__2 = rc0020 - (zw[kk + 1] - zl1) / (zminrad - zl1);
	    r__1 = max(r__2,zfmin);
	    zfac[kk - 1] = min(r__1,rc0020);
/* Computing MAX */
	    r__1 = (zminrad - zw[kk + 1]) / zminrad;
/* Computing 3rd power */
	    r__2 = rc0020 - zfac[kk - 1];
	    zfacent[kk - 1] = rc0021 * max(r__1,zero) * (r__2 * (r__2 * r__2))
		    ;
	    d__1 =  (phifac * karman * wm3 * zfac[kk - 1]);
	    d__2 =  onethird;
	    wscalek2[kk - 1] = (real)pow_dd(&d__1, &d__2);
/* Computing 3rd power */
	    r__1 = rc0020 - zfac[kk - 1];
	    khtopdown[kk] = wscalek2[kk - 1] * karman * (zminrad - zw[kk + 1])
		     * (r__1 * (r__1 * r__1));
/* Computing MAX */
	    r__1 = khtopdown[kk];
	    khtopdown[kk] = max(r__1,zero);
	    tkeprodtd[kk] = rc0022 * ent_eff__ * wm3 / max(*pblh,rc0003) * 
		    zfacent[kk - 1];
/* Computing MAX */
	    r__1 = tkeprodtd[kk];
	    tkeprodtd[kk] = max(r__1,zero);
/* L9010: */
	}
/* L9011: */
    }
    red001 = -rc0023;
    i__2 = *kte - *kts + 1;
    
#pragma unroll 1
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
/* Computing MAX */
	r__1 = red001, r__2 = khtopdown[*kts + ixarr1 - 1];
	red001 = max(r__1,r__2);
    }
    *maxkhtopdown = red001;
    return 0;
} /* topdown_cloudrad__ */

__device__ M471_CALL integer ceiling_(unsigned char *m471_work, int m471_n, M471Ptr<real> x) {
    /* System generated locals */
    integer ret_val;

    ret_val = (integer) (*x);
    if (*x > (real) ret_val) {
	++ret_val;
    }
    return ret_val;
} /* ceiling_ */


template<int M471_PHASE> __device__ __forceinline__ void m471_entry(int n,float*f32,int*i32,unsigned char*arena){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
unsigned char*m471_work=arena;int m471_n=n;
int a_i;
int a_j;
int a_initflag;
int a_restart;
int a_cycling;
float a_delt;
M471Ptr<float> a_dz1={f32+1*n+c,n};
float a_dx;
M471Ptr<float> a_u1={f32+(1*(MYNN471_NZ-49)+51)*n+c,n};
M471Ptr<float> a_v1={f32+(2*(MYNN471_NZ-49)+100)*n+c,n};
M471Ptr<float> a_w1={f32+(3*(MYNN471_NZ-49)+149)*n+c,n};
M471Ptr<float> a_th1={f32+(4*(MYNN471_NZ-49)+198)*n+c,n};
M471Ptr<float> a_sqv1={f32+(5*(MYNN471_NZ-49)+247)*n+c,n};
M471Ptr<float> a_sqc1={f32+(6*(MYNN471_NZ-49)+296)*n+c,n};
M471Ptr<float> a_sqi1={f32+(7*(MYNN471_NZ-49)+345)*n+c,n};
M471Ptr<float> a_sqs1={f32+(8*(MYNN471_NZ-49)+394)*n+c,n};
M471Ptr<float> a_qnc1={f32+(9*(MYNN471_NZ-49)+443)*n+c,n};
M471Ptr<float> a_qni1={f32+(10*(MYNN471_NZ-49)+492)*n+c,n};
M471Ptr<float> a_qnwfa1={f32+(11*(MYNN471_NZ-49)+541)*n+c,n};
M471Ptr<float> a_qnifa1={f32+(12*(MYNN471_NZ-49)+590)*n+c,n};
M471Ptr<float> a_qnbca1={f32+(13*(MYNN471_NZ-49)+639)*n+c,n};
M471Ptr<float> a_ozone1={f32+(14*(MYNN471_NZ-49)+688)*n+c,n};
M471Ptr<float> a_p1={f32+(15*(MYNN471_NZ-49)+737)*n+c,n};
M471Ptr<float> a_ex1={f32+(16*(MYNN471_NZ-49)+786)*n+c,n};
M471Ptr<float> a_rho1={f32+(17*(MYNN471_NZ-49)+835)*n+c,n};
M471Ptr<float> a_tk1={f32+(18*(MYNN471_NZ-49)+884)*n+c,n};
float a_xland;
float a_ts;
float a_qsfc;
float a_ps;
float a_ust;
float a_ch;
float a_hfx;
float a_qfx;
float a_znt;
float a_wspd;
float a_uoce;
float a_voce;
M471Ptr<float> a_qke1={f32+(19*(MYNN471_NZ-49)+945)*n+c,n};
M471Ptr<float> a_qke_adv1={f32+(20*(MYNN471_NZ-49)+994)*n+c,n};
M471Ptr<float> a_el1={f32+(21*(MYNN471_NZ-49)+1043)*n+c,n};
M471Ptr<float> a_sh1={f32+(22*(MYNN471_NZ-49)+1092)*n+c,n};
M471Ptr<float> a_sm1={f32+(23*(MYNN471_NZ-49)+1141)*n+c,n};
M471Ptr<float> a_kh1={f32+(24*(MYNN471_NZ-49)+1190)*n+c,n};
M471Ptr<float> a_km1={f32+(25*(MYNN471_NZ-49)+1239)*n+c,n};
int a_nchem;
int a_kdvel;
int a_ndvel;
M471Ptr<float> a_chem={f32+(26*(MYNN471_NZ-49)+1288)*n+c,n};
M471Ptr<float> a_vdep={f32+(27*(MYNN471_NZ-49)+1337)*n+c,n};
float a_frp;
float a_emis_ant_no;
int a_mix_chem;
int a_enh_mix;
int a_rrfs_sd;
int a_smoke_dbg;
M471Ptr<float> a_tsq1={f32+(27*(MYNN471_NZ-49)+1340)*n+c,n};
M471Ptr<float> a_qsq1={f32+(28*(MYNN471_NZ-49)+1389)*n+c,n};
M471Ptr<float> a_cov1={f32+(29*(MYNN471_NZ-49)+1438)*n+c,n};
M471Ptr<float> a_du1={f32+(30*(MYNN471_NZ-49)+1487)*n+c,n};
M471Ptr<float> a_dv1={f32+(31*(MYNN471_NZ-49)+1536)*n+c,n};
M471Ptr<float> a_dth1={f32+(32*(MYNN471_NZ-49)+1585)*n+c,n};
M471Ptr<float> a_dqv1={f32+(33*(MYNN471_NZ-49)+1634)*n+c,n};
M471Ptr<float> a_dqc1={f32+(34*(MYNN471_NZ-49)+1683)*n+c,n};
M471Ptr<float> a_dqi1={f32+(35*(MYNN471_NZ-49)+1732)*n+c,n};
M471Ptr<float> a_dqnc1={f32+(36*(MYNN471_NZ-49)+1781)*n+c,n};
M471Ptr<float> a_dqni1={f32+(37*(MYNN471_NZ-49)+1830)*n+c,n};
M471Ptr<float> a_dqs1={f32+(38*(MYNN471_NZ-49)+1879)*n+c,n};
M471Ptr<float> a_dqnwfa1={f32+(39*(MYNN471_NZ-49)+1928)*n+c,n};
M471Ptr<float> a_dqnifa1={f32+(40*(MYNN471_NZ-49)+1977)*n+c,n};
M471Ptr<float> a_dqnbca1={f32+(41*(MYNN471_NZ-49)+2026)*n+c,n};
M471Ptr<float> a_dozone1={f32+(42*(MYNN471_NZ-49)+2075)*n+c,n};
M471Ptr<float> a_rthraten1={f32+(43*(MYNN471_NZ-49)+2124)*n+c,n};
float a_pblh;
int a_kpbl;
float a_maxwidth;
float a_maxmf;
float a_ztop_plume;
M471Ptr<float> a_dqke1={f32+(44*(MYNN471_NZ-49)+2177)*n+c,n};
M471Ptr<float> a_qwt1={f32+(45*(MYNN471_NZ-49)+2226)*n+c,n};
M471Ptr<float> a_qshear1={f32+(46*(MYNN471_NZ-49)+2275)*n+c,n};
M471Ptr<float> a_qbuoy1={f32+(47*(MYNN471_NZ-49)+2324)*n+c,n};
M471Ptr<float> a_qdiss1={f32+(48*(MYNN471_NZ-49)+2373)*n+c,n};
M471Ptr<float> a_qc_bl1={f32+(49*(MYNN471_NZ-49)+2422)*n+c,n};
M471Ptr<float> a_qi_bl1={f32+(50*(MYNN471_NZ-49)+2471)*n+c,n};
M471Ptr<float> a_cldfra_bl1={f32+(51*(MYNN471_NZ-49)+2520)*n+c,n};
int a_bl_mynn_tkeadvect;
int a_tke_budget;
int a_bl_mynn_cloudpdf;
int a_bl_mynn_mixlength;
int a_icloud_bl;
float a_closure;
int a_bl_mynn_edmf;
int a_bl_mynn_edmf_mom;
int a_bl_mynn_edmf_tke;
int a_bl_mynn_mixscalars;
int a_bl_mynn_output;
int a_bl_mynn_cloudmix;
int a_bl_mynn_mixqt;
M471Ptr<float> a_edmf_a1={f32+(52*(MYNN471_NZ-49)+2570)*n+c,n};
M471Ptr<float> a_edmf_w1={f32+(53*(MYNN471_NZ-49)+2619)*n+c,n};
M471Ptr<float> a_edmf_qt1={f32+(54*(MYNN471_NZ-49)+2668)*n+c,n};
M471Ptr<float> a_edmf_thl1={f32+(55*(MYNN471_NZ-49)+2717)*n+c,n};
M471Ptr<float> a_edmf_ent1={f32+(56*(MYNN471_NZ-49)+2766)*n+c,n};
M471Ptr<float> a_edmf_qc1={f32+(57*(MYNN471_NZ-49)+2815)*n+c,n};
M471Ptr<float> a_sub_thl1={f32+(58*(MYNN471_NZ-49)+2864)*n+c,n};
M471Ptr<float> a_sub_sqv1={f32+(59*(MYNN471_NZ-49)+2913)*n+c,n};
M471Ptr<float> a_det_thl1={f32+(60*(MYNN471_NZ-49)+2962)*n+c,n};
M471Ptr<float> a_det_sqv1={f32+(61*(MYNN471_NZ-49)+3011)*n+c,n};
int a_spp_pbl;
M471Ptr<float> a_pattern_spp_pbl1={f32+(62*(MYNN471_NZ-49)+3060)*n+c,n};
int a_flag_qc;
int a_flag_qi;
int a_flag_qnc;
int a_flag_qni;
int a_flag_qs;
int a_flag_qnwfa;
int a_flag_qnifa;
int a_flag_qnbca;
int a_flag_ozone;
int a_kts;
int a_kte;
int a_errflg;
a_i=i32[0*n+c];
a_j=i32[1*n+c];
a_initflag=i32[2*n+c];
a_restart=i32[3*n+c];
a_cycling=i32[4*n+c];
a_delt=f32[0*n+c];


a_dx=f32[(1*(MYNN471_NZ-49)+50)*n+c];




































a_xland=f32[(19*(MYNN471_NZ-49)+933)*n+c];
a_ts=f32[(19*(MYNN471_NZ-49)+934)*n+c];
a_qsfc=f32[(19*(MYNN471_NZ-49)+935)*n+c];
a_ps=f32[(19*(MYNN471_NZ-49)+936)*n+c];
a_ust=f32[(19*(MYNN471_NZ-49)+937)*n+c];
a_ch=f32[(19*(MYNN471_NZ-49)+938)*n+c];
a_hfx=f32[(19*(MYNN471_NZ-49)+939)*n+c];
a_qfx=f32[(19*(MYNN471_NZ-49)+940)*n+c];
a_znt=f32[(19*(MYNN471_NZ-49)+941)*n+c];
a_wspd=f32[(19*(MYNN471_NZ-49)+942)*n+c];
a_uoce=f32[(19*(MYNN471_NZ-49)+943)*n+c];
a_voce=f32[(19*(MYNN471_NZ-49)+944)*n+c];














a_nchem=i32[5*n+c];
a_kdvel=i32[6*n+c];
a_ndvel=i32[7*n+c];




a_frp=f32[(27*(MYNN471_NZ-49)+1338)*n+c];
a_emis_ant_no=f32[(27*(MYNN471_NZ-49)+1339)*n+c];
a_mix_chem=i32[8*n+c];
a_enh_mix=i32[9*n+c];
a_rrfs_sd=i32[10*n+c];
a_smoke_dbg=i32[11*n+c];


































a_pblh=f32[(44*(MYNN471_NZ-49)+2173)*n+c];
a_kpbl=i32[12*n+c];
a_maxwidth=f32[(44*(MYNN471_NZ-49)+2174)*n+c];
a_maxmf=f32[(44*(MYNN471_NZ-49)+2175)*n+c];
a_ztop_plume=f32[(44*(MYNN471_NZ-49)+2176)*n+c];
















a_bl_mynn_tkeadvect=i32[13*n+c];
a_tke_budget=i32[14*n+c];
a_bl_mynn_cloudpdf=i32[15*n+c];
a_bl_mynn_mixlength=i32[16*n+c];
a_icloud_bl=i32[17*n+c];
a_closure=f32[(52*(MYNN471_NZ-49)+2569)*n+c];
a_bl_mynn_edmf=i32[18*n+c];
a_bl_mynn_edmf_mom=i32[19*n+c];
a_bl_mynn_edmf_tke=i32[20*n+c];
a_bl_mynn_mixscalars=i32[21*n+c];
a_bl_mynn_output=i32[22*n+c];
a_bl_mynn_cloudmix=i32[23*n+c];
a_bl_mynn_mixqt=i32[24*n+c];




















a_spp_pbl=i32[25*n+c];


a_flag_qc=i32[26*n+c];
a_flag_qi=i32[27*n+c];
a_flag_qnc=i32[28*n+c];
a_flag_qni=i32[29*n+c];
a_flag_qs=i32[30*n+c];
a_flag_qnwfa=i32[31*n+c];
a_flag_qnifa=i32[32*n+c];
a_flag_qnbca=i32[33*n+c];
a_flag_ozone=i32[34*n+c];
a_kts=i32[35*n+c];
a_kte=i32[36*n+c];
a_errflg=i32[37*n+c];
 if constexpr(M471_PHASE==8){if(a_bl_mynn_edmf==0)return;}
 mynnedmf_<M471_PHASE>(m471_work,n,&a_i,&a_j,&a_initflag,&a_restart,&a_cycling,&a_delt,a_dz1,&a_dx,a_u1,a_v1,a_w1,a_th1,a_sqv1,a_sqc1,a_sqi1,a_sqs1,a_qnc1,a_qni1,a_qnwfa1,a_qnifa1,a_qnbca1,a_ozone1,a_p1,a_ex1,a_rho1,a_tk1,&a_xland,&a_ts,&a_qsfc,&a_ps,&a_ust,&a_ch,&a_hfx,&a_qfx,&a_znt,&a_wspd,&a_uoce,&a_voce,a_qke1,a_qke_adv1,a_el1,a_sh1,a_sm1,a_kh1,a_km1,&a_nchem,&a_kdvel,&a_ndvel,a_chem,a_vdep,&a_frp,&a_emis_ant_no,&a_mix_chem,&a_enh_mix,&a_rrfs_sd,&a_smoke_dbg,a_tsq1,a_qsq1,a_cov1,a_du1,a_dv1,a_dth1,a_dqv1,a_dqc1,a_dqi1,a_dqnc1,a_dqni1,a_dqs1,a_dqnwfa1,a_dqnifa1,a_dqnbca1,a_dozone1,a_rthraten1,&a_pblh,&a_kpbl,&a_maxwidth,&a_maxmf,&a_ztop_plume,a_dqke1,a_qwt1,a_qshear1,a_qbuoy1,a_qdiss1,a_qc_bl1,a_qi_bl1,a_cldfra_bl1,&a_bl_mynn_tkeadvect,&a_tke_budget,&a_bl_mynn_cloudpdf,&a_bl_mynn_mixlength,&a_icloud_bl,&a_closure,&a_bl_mynn_edmf,&a_bl_mynn_edmf_mom,&a_bl_mynn_edmf_tke,&a_bl_mynn_mixscalars,&a_bl_mynn_output,&a_bl_mynn_cloudmix,&a_bl_mynn_mixqt,a_edmf_a1,a_edmf_w1,a_edmf_qt1,a_edmf_thl1,a_edmf_ent1,a_edmf_qc1,a_sub_thl1,a_sub_sqv1,a_det_thl1,a_det_sqv1,&a_spp_pbl,a_pattern_spp_pbl1,&a_flag_qc,&a_flag_qi,&a_flag_qnc,&a_flag_qni,&a_flag_qs,&a_flag_qnwfa,&a_flag_qnifa,&a_flag_qnbca,&a_flag_ozone,&a_kts,&a_kte,&a_errflg);




































































if constexpr(M471_PHASE!=8){
f32[(44*(MYNN471_NZ-49)+2173)*n+c]=a_pblh;
i32[12*n+c]=a_kpbl;
f32[(44*(MYNN471_NZ-49)+2174)*n+c]=a_maxwidth;
f32[(44*(MYNN471_NZ-49)+2175)*n+c]=a_maxmf;
f32[(44*(MYNN471_NZ-49)+2176)*n+c]=a_ztop_plume;




































i32[37*n+c]=a_errflg;
}
}
extern "C" __global__ M471_LAUNCH void mynn471_step(int n,float*f32,int*i32,unsigned char*arena){m471_entry<-1>(n,f32,i32,arena);}
extern "C" __global__ M471_LAUNCH void mynn471_phase0(int n,float*f32,int*i32,unsigned char*arena){m471_entry<0>(n,f32,i32,arena);}
extern "C" __global__ M471_LAUNCH void mynn471_phase1(int n,float*f32,int*i32,unsigned char*arena){m471_entry<1>(n,f32,i32,arena);}
extern "C" __global__ M471_LAUNCH void mynn471_phase2(int n,float*f32,int*i32,unsigned char*arena){m471_entry<2>(n,f32,i32,arena);}
extern "C" __global__ M471_LAUNCH void mynn471_phase3(int n,float*f32,int*i32,unsigned char*arena){m471_entry<3>(n,f32,i32,arena);}
extern "C" __global__ M471_LAUNCH void mynn471_phase4(int n,float*f32,int*i32,unsigned char*arena){m471_entry<4>(n,f32,i32,arena);}
extern "C" __global__ M471_LAUNCH void mynn471_phase5(int n,float*f32,int*i32,unsigned char*arena){m471_entry<5>(n,f32,i32,arena);}
extern "C" __global__ M471_LAUNCH void mynn471_phase6(int n,float*f32,int*i32,unsigned char*arena){m471_entry<6>(n,f32,i32,arena);}
extern "C" __global__ M471_LAUNCH void mynn471_phase7(int n,float*f32,int*i32,unsigned char*arena){m471_entry<7>(n,f32,i32,arena);}
extern "C" __global__ M471_LAUNCH void mynn471_phase8(int n,float*f32,int*i32,unsigned char*arena){m471_entry<8>(n,f32,i32,arena);}
extern "C" __global__ M471_LAUNCH void mynn471_phase9(int n,float*f32,int*i32,unsigned char*arena){m471_entry<9>(n,f32,i32,arena);}
