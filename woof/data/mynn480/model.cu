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
// 8f8e8b59's single column kernel: no minimum-blocks bound. With (128,4)
// ptxas capped it at 128 registers and spilled 6960 bytes per thread; with
// (128) it uses 255 and spills 924 (kernel 6.7 -> 5.6 ms, 20k real columns).
#define M471_LAUNCH __launch_bounds__(128)
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
// WRF 4.8.0 MYNN-EDMF 8f8e8b59 (Apache-2.0) numerical source; generated offline.



template<int M471_PHASE>
__device__ M471_CALL /* Subroutine */ int mynnedmf_(unsigned char *m471_work, int m471_n, M471Ptr<integer> i__, M471Ptr<integer> j, M471Ptr<integer> initflag, 
	M471Ptr<logical> restart, M471Ptr<logical> cycling, M471Ptr<real> delt, M471Ptr<real> dz1, M471Ptr<real> dx, 
	M471Ptr<real> u1, M471Ptr<real> v1, M471Ptr<real> w1, M471Ptr<real> th1, M471Ptr<real> sqv1, M471Ptr<real> sqc1, M471Ptr<real> sqi1, M471Ptr<real> sqs1, M471Ptr<real> qnc1, M471Ptr<real> qni1, M471Ptr<real> qnwfa1, M471Ptr<real> qnifa1,
	 M471Ptr<real> qnbca1, M471Ptr<real> ozone1, M471Ptr<real> pres1, M471Ptr<real> ex1, M471Ptr<real> rho1, M471Ptr<real> tk1, M471Ptr<real> xland, M471Ptr<real> ts, M471Ptr<real> qsfc, M471Ptr<real> ps, M471Ptr<real> ust, M471Ptr<real> ch, M471Ptr<real> hfx, M471Ptr<real> qfx, M471Ptr<real> znt, M471Ptr<real> wspd, M471Ptr<real> uoce, M471Ptr<real> voce, M471Ptr<real> qke1, M471Ptr<real> qke_adv1__, M471Ptr<real> el1, M471Ptr<real> sh1, M471Ptr<real> sm1, 
	M471Ptr<real> kh1, M471Ptr<real> km1, M471Ptr<integer> nchem, M471Ptr<integer> ndvel, M471Ptr<real> chem1, 
	M471Ptr<real> vdep, M471Ptr<real> frp, M471Ptr<real> emis_ant_no__, M471Ptr<logical> mix_chem__, 
	M471Ptr<logical> enh_mix__, M471Ptr<real> settle1, M471Ptr<real> scalars, M471Ptr<integer> nscalars, 
	M471Ptr<real> tsq1, M471Ptr<real> qsq1, M471Ptr<real> cov1, M471Ptr<real> du1, M471Ptr<real> dv1, M471Ptr<real> dth1, 
	M471Ptr<real> dqv1, M471Ptr<real> dqc1, M471Ptr<real> dqi1, M471Ptr<real> dqnc1, M471Ptr<real> dqni1, M471Ptr<real> dqs1, M471Ptr<real> dqnwfa1, M471Ptr<real> dqnifa1, M471Ptr<real> dqnbca1, M471Ptr<real> dozone1, 
	M471Ptr<real> rthraten1, M471Ptr<real> pblh, M471Ptr<integer> kpbl, M471Ptr<real> maxwidth_dd__, M471Ptr<real> maxwidth, M471Ptr<real> maxmf, M471Ptr<real> ztop_plume__, M471Ptr<real> excess_h__, M471Ptr<real> excess_q__, M471Ptr<real> maxmf_dd__, M471Ptr<real> maxtkeprod, M471Ptr<real> cldtop_cooling__, M471Ptr<real> ent_eff__, M471Ptr<real> dqke1, M471Ptr<real> qwt1, M471Ptr<real> qshear1, M471Ptr<real> qbuoy1, M471Ptr<real> qdiss1, M471Ptr<real> qc_bl1__, M471Ptr<real> qi_bl1__, 
	M471Ptr<real> cldfra_bl1__, M471Ptr<logical> bl_mynn_tkeadvect__, M471Ptr<integer> tke_budget__, M471Ptr<integer> bl_mynn_cloudpdf__, M471Ptr<integer> bl_mynn_mixlength__, M471Ptr<real> bl_mynn_closure__, M471Ptr<integer> bl_mynn_ess__, 
	M471Ptr<integer> bl_mynn_edmf__, M471Ptr<integer> bl_mynn_edmf_mom__, M471Ptr<integer> bl_mynn_edmf_tke__, M471Ptr<integer> bl_mynn_mixscalars__, M471Ptr<integer> bl_mynn_mixaerosols__, M471Ptr<integer> bl_mynn_mixnumcon__, M471Ptr<integer> bl_mynn_output__, M471Ptr<integer> bl_mynn_cloudmix__, M471Ptr<integer> bl_mynn_mixqt__, M471Ptr<integer> bl_mynn_edmf_dd__, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_qt1__, M471Ptr<real> edmf_thl1__, M471Ptr<real> edmf_ent1__, 
	M471Ptr<real> edmf_qc1__, M471Ptr<real> sub_thl1__, M471Ptr<real> sub_sqv1__, M471Ptr<real> det_thl1__, M471Ptr<real> det_sqv1__, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__, M471Ptr<logical> flag_qc__, M471Ptr<logical> flag_qi__, M471Ptr<logical> flag_qnc__, M471Ptr<logical> flag_qni__, M471Ptr<logical> flag_qs__, M471Ptr<logical> flag_qnwfa__, M471Ptr<logical> flag_qnifa__, M471Ptr<logical> flag_qnbca__, M471Ptr<logical> flag_ozone__, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<integer> errflg);
__device__ M471_CALL /* Subroutine */ int mym_initialize__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> ckmod,
	 M471Ptr<real> taue, M471Ptr<real> xland, M471Ptr<real> pblh, M471Ptr<real> dz, M471Ptr<real> dx, M471Ptr<real> zw, 
	M471Ptr<real> p, M471Ptr<real> exner, M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> thl, M471Ptr<real> qw, M471Ptr<real> theta, M471Ptr<real> thv, M471Ptr<real> thlv, M471Ptr<real> sh, M471Ptr<real> sm, M471Ptr<real> ust, M471Ptr<real> rmol, M471Ptr<real> rmolh, M471Ptr<real> el, M471Ptr<real> qke, M471Ptr<real> tsq, M471Ptr<real> qsq, M471Ptr<real> cov, M471Ptr<real> psig_bl__, M471Ptr<real> cldfra_bl1__, M471Ptr<integer> bl_mynn_mixlength__, M471Ptr<integer> bl_mynn_ess__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w_dd1__, M471Ptr<real> edmf_a_dd1__, M471Ptr<logical> initialize_qke__, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__);
__device__ M471_CALL /* Subroutine */ int mym_level2__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> ckmod, 
	M471Ptr<integer> bl_mynn_ess__, M471Ptr<real> zw, M471Ptr<real> dz, M471Ptr<real> xland, M471Ptr<real> pblh, 
	M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> thl, M471Ptr<real> thv, M471Ptr<real> thlv, M471Ptr<real> th, M471Ptr<real> p,
	 M471Ptr<real> exner, M471Ptr<real> qke, M471Ptr<real> cldfra, M471Ptr<real> edmf_a__, M471Ptr<real> edmf_w__,
	 M471Ptr<real> qw, M471Ptr<real> ql, M471Ptr<real> vt, M471Ptr<real> vq, M471Ptr<real> dtl, M471Ptr<real> dqw, M471Ptr<real> dtv, M471Ptr<real> gm, M471Ptr<real> gh, M471Ptr<real> sm, M471Ptr<real> sh);
__device__ M471_CALL /* Subroutine */ int mym_length__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> xland, 
	M471Ptr<real> dz, M471Ptr<real> dx, M471Ptr<real> zw, M471Ptr<real> rmol, M471Ptr<real> rmolh, M471Ptr<real> flt, 
	M471Ptr<real> fltv, M471Ptr<real> flq, M471Ptr<real> vt, M471Ptr<real> vq, M471Ptr<real> u1, M471Ptr<real> v1, M471Ptr<real> qke, M471Ptr<real> qpe, M471Ptr<real> dtv, M471Ptr<real> el, M471Ptr<real> pblh, M471Ptr<real> theta, M471Ptr<real> qkw, M471Ptr<real> qtw, M471Ptr<real> psig_bl__, M471Ptr<real> cldfra_bl1__, M471Ptr<integer> bl_mynn_mixlength__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w_dd1__, M471Ptr<real> edmf_a_dd1__);
__device__ M471_CALL /* Subroutine */ int boulac_length0__(unsigned char *m471_work, int m471_n, M471Ptr<integer> k, M471Ptr<integer> kts, M471Ptr<integer> kte, 
	M471Ptr<real> zw, M471Ptr<real> dz, M471Ptr<real> qtke, M471Ptr<real> theta, M471Ptr<real> lb1, M471Ptr<real> lb2);
__device__ M471_CALL /* Subroutine */ int boulac_length__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> zw, 
	M471Ptr<real> dz, M471Ptr<real> qtke, M471Ptr<real> theta, M471Ptr<real> lb1, M471Ptr<real> lb2);
__device__ M471_CALL /* Subroutine */ int mym_turbulence__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> xland,
	 M471Ptr<real> closure, M471Ptr<real> ckmod, M471Ptr<real> taue, M471Ptr<real> dz, M471Ptr<real> dx, M471Ptr<real> zw,
	 M471Ptr<real> p, M471Ptr<real> exner, M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> thl, M471Ptr<real> thv, M471Ptr<real> thlv, M471Ptr<real> ql, M471Ptr<real> qw, M471Ptr<real> qke, M471Ptr<real> tsq, M471Ptr<real> qsq, M471Ptr<real> cov, 
	M471Ptr<real> vt, M471Ptr<real> vq, M471Ptr<real> rmol, M471Ptr<real> rmolh, M471Ptr<real> flt, M471Ptr<real> fltv, 
	M471Ptr<real> flq, M471Ptr<real> pblh, M471Ptr<real> theta, M471Ptr<real> sh, M471Ptr<real> sm, M471Ptr<real> el, 
	M471Ptr<real> dfm, M471Ptr<real> dfh, M471Ptr<real> dfq, M471Ptr<real> tcd, M471Ptr<real> qcd, M471Ptr<real> pdk, 
	M471Ptr<real> pdt, M471Ptr<real> pdq, M471Ptr<real> pdc, M471Ptr<real> qwt1, M471Ptr<real> qshear1, M471Ptr<real> qbuoy1, M471Ptr<real> qdiss1, M471Ptr<integer> tke_budget__, M471Ptr<real> psig_bl__, M471Ptr<real> psig_shcu__, M471Ptr<real> cldfra_bl1__, M471Ptr<integer> bl_mynn_mixlength__, 
	M471Ptr<integer> bl_mynn_ess__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w_dd1__, M471Ptr<real> edmf_a_dd1__, M471Ptr<real> tkeprod_dn__, M471Ptr<real> tkeprod_up__, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__);
__device__ M471_CALL /* Subroutine */ int mym_predict__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> closure, 
	M471Ptr<real> delt, M471Ptr<real> dz, M471Ptr<real> ust, M471Ptr<real> flt, M471Ptr<real> flq, M471Ptr<real> pmz, 
	M471Ptr<real> phh, M471Ptr<real> el, M471Ptr<real> dfq, M471Ptr<real> rho, M471Ptr<real> pdk, M471Ptr<real> pdt, M471Ptr<real> pdq, M471Ptr<real> pdc, M471Ptr<real> qke, M471Ptr<real> tsq, M471Ptr<real> qsq, M471Ptr<real> cov, M471Ptr<real> s_aw1__, M471Ptr<real> s_awqke1__, M471Ptr<real> s_awqsq1__, M471Ptr<integer> bl_mynn_edmf__, 
	M471Ptr<integer> bl_mynn_edmf_tke__, M471Ptr<real> qwt1, M471Ptr<real> qdiss1, M471Ptr<integer> tke_budget__, M471Ptr<real> upwind);
__device__ M471_CALL /* Subroutine */ int mym_condensation__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dx, 
	M471Ptr<real> dz, M471Ptr<real> zw, M471Ptr<real> xland, M471Ptr<real> thl, M471Ptr<real> qw, M471Ptr<real> qv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<real> qs, M471Ptr<real> p, M471Ptr<real> exner, M471Ptr<real> tsq, M471Ptr<real> qsq, 
	M471Ptr<real> cov, M471Ptr<real> sh, M471Ptr<real> el, M471Ptr<integer> bl_mynn_cloudpdf__, M471Ptr<real> qc_bl1__, M471Ptr<real> qi_bl1__, M471Ptr<real> cldfra_bl1__, M471Ptr<real> pblh1, M471Ptr<real> hfx1,
	 M471Ptr<real> vt, M471Ptr<real> vq, M471Ptr<real> th, M471Ptr<real> sgm, M471Ptr<real> closure, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__);
__device__ M471_CALL /* Subroutine */ int mynn_tendencies__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<integer> i__, M471Ptr<real> delt, M471Ptr<real> dz, M471Ptr<real> zw, M471Ptr<real> xland, M471Ptr<real> pblh, M471Ptr<real> rho, M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> th, M471Ptr<real> tk, M471Ptr<real> qv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<real> qs, M471Ptr<real> qnc, M471Ptr<real> qni, M471Ptr<real> psfc, M471Ptr<real> p, M471Ptr<real> exner, 
	M471Ptr<real> thl, M471Ptr<real> sqv, M471Ptr<real> sqc, M471Ptr<real> sqi, M471Ptr<real> sqs, M471Ptr<real> sqw, 
	M471Ptr<real> thl_tot1__, M471Ptr<real> qc_tot1__, M471Ptr<real> qi_tot1__, M471Ptr<real> qnwfa, M471Ptr<real> qnifa, M471Ptr<real> qnbca, M471Ptr<real> ozone, M471Ptr<real> ust, M471Ptr<real> flt, M471Ptr<real> flq, 
	M471Ptr<real> flqv, M471Ptr<real> flqc, M471Ptr<real> wspd, M471Ptr<real> uoce, M471Ptr<real> voce, M471Ptr<real> tsq,
	 M471Ptr<real> qsq, M471Ptr<real> cov, M471Ptr<real> tcd, M471Ptr<real> qcd, M471Ptr<real> dfm, M471Ptr<real> dfh, 
	M471Ptr<real> dfq, M471Ptr<real> du, M471Ptr<real> dv, M471Ptr<real> dth, M471Ptr<real> dqv, M471Ptr<real> dqc, M471Ptr<real> dqi, M471Ptr<real> dqs, M471Ptr<real> dqnc, M471Ptr<real> dqni, M471Ptr<real> dqnwfa, M471Ptr<real> dqnifa, 
	M471Ptr<real> dqnbca, M471Ptr<real> dozone, M471Ptr<real> diss_heat__, M471Ptr<real> s_aw1__, M471Ptr<real> s_awthl1__, M471Ptr<real> s_awqt1__, M471Ptr<real> s_awqv1__, M471Ptr<real> s_awqc1__, M471Ptr<real> s_awu1__, M471Ptr<real> s_awv1__, M471Ptr<real> s_awqnc1__, M471Ptr<real> s_awqni1__, M471Ptr<real> s_awqnwfa1__, M471Ptr<real> s_awqnifa1__, M471Ptr<real> s_awqnbca1__, M471Ptr<real> sd_aw1__, 
	M471Ptr<real> sd_awthl1__, M471Ptr<real> sd_awqt1__, M471Ptr<real> sd_awqv1__, M471Ptr<real> sd_awqc1__, M471Ptr<real> sd_awqi1__, M471Ptr<real> sd_awqnc1__, M471Ptr<real> sd_awqni1__, 
	M471Ptr<real> sd_awqnwfa1__, M471Ptr<real> sd_awqnifa1__, M471Ptr<real> sd_awu1__, M471Ptr<real> sd_awv1__, M471Ptr<real> sub_thl__, M471Ptr<real> sub_sqv__, M471Ptr<real> sub_u__, M471Ptr<real> sub_v__, M471Ptr<real> det_thl__, M471Ptr<real> det_sqv__, M471Ptr<real> det_sqc__, M471Ptr<real> det_u__, M471Ptr<real> det_v__, M471Ptr<logical> flag_qc__, M471Ptr<logical> flag_qi__, 
	M471Ptr<logical> flag_qnc__, M471Ptr<logical> flag_qni__, M471Ptr<logical> flag_qs__, M471Ptr<logical> flag_qnwfa__, M471Ptr<logical> flag_qnifa__, M471Ptr<logical> flag_qnbca__, M471Ptr<logical> flag_ozone__, M471Ptr<real> cldfra_bl1__, M471Ptr<integer> bl_mynn_cloudmix__, 
	M471Ptr<integer> bl_mynn_mixqt__, M471Ptr<integer> bl_mynn_edmf__, M471Ptr<integer> bl_mynn_edmf_mom__, M471Ptr<integer> bl_mynn_mixscalars__, M471Ptr<integer> bl_mynn_mixaerosols__, M471Ptr<integer> bl_mynn_mixnumcon__, M471Ptr<real> upwind);
__device__ M471_CALL /* Subroutine */ int moisture_check__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kte, M471Ptr<real> delt, M471Ptr<real> dp, 
	M471Ptr<real> exner, M471Ptr<real> qv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<real> qs, M471Ptr<real> th, M471Ptr<real> dqv, M471Ptr<real> dqc, M471Ptr<real> dqi, M471Ptr<real> dqs, M471Ptr<real> dth);
__device__ M471_CALL /* Subroutine */ int mynn_mix_chem__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<integer> i__,
	 M471Ptr<real> delt, M471Ptr<real> dz, M471Ptr<real> pblh, M471Ptr<integer> nchem, M471Ptr<integer> ndvel, 
	M471Ptr<real> chem1, M471Ptr<real> vd1, M471Ptr<real> settle1, M471Ptr<real> rho, M471Ptr<real> flt, M471Ptr<real> tcd, M471Ptr<real> qcd, M471Ptr<real> dfh, M471Ptr<real> s_aw1__, M471Ptr<real> s_awchem1__, M471Ptr<real> emis_ant_no__, M471Ptr<real> frp, M471Ptr<logical> enh_mix__, M471Ptr<integer> bl_mynn_edmf__,
	 M471Ptr<real> upwind);
__device__ M471_CALL /* Subroutine */ int mynn_mix_scalars__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<integer> i__, M471Ptr<real> delt, M471Ptr<real> dz, M471Ptr<integer> nscalars, M471Ptr<real> scalars, M471Ptr<real> rho, M471Ptr<real> flt, M471Ptr<real> tcd, M471Ptr<real> qcd, M471Ptr<real> dfh, M471Ptr<real> s_aw1__, M471Ptr<real> s_awscalars1__, M471Ptr<integer> bl_mynn_edmf__, M471Ptr<real> upwind);
__device__ M471_CALL /* Subroutine */ int retrieve_exchange_coeffs__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, 
	M471Ptr<real> dfm, M471Ptr<real> dfh, M471Ptr<real> dz, M471Ptr<real> km1, M471Ptr<real> kh1);
__device__ M471_CALL /* Subroutine */ int tridiag_(unsigned char *m471_work, int m471_n, M471Ptr<integer> n, M471Ptr<real> a, M471Ptr<real> b, M471Ptr<real> c__, M471Ptr<real> d__);
__device__ M471_CALL /* Subroutine */ int tridiag2_(unsigned char *m471_work, int m471_n, M471Ptr<integer> kte, M471Ptr<real> a, M471Ptr<real> b, M471Ptr<real> c__, 
	M471Ptr<real> d__, M471Ptr<real> x);
__device__ M471_CALL /* Subroutine */ int tridiag3_(unsigned char *m471_work, int m471_n, M471Ptr<integer> kte, M471Ptr<real> a, M471Ptr<real> b, M471Ptr<real> c__, 
	M471Ptr<real> d__, M471Ptr<real> x);
__device__ M471_CALL /* Subroutine */ int get_pblh__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> pblh, M471Ptr<real> thv1, M471Ptr<real> qke1, M471Ptr<real> ust, M471Ptr<real> zw1, M471Ptr<real> dz1, M471Ptr<real> landsea, 
	M471Ptr<integer> kpbl);
__device__ M471_CALL /* Subroutine */ int dmp_mf__(unsigned char *m471_work, int m471_n, M471Ptr<integer> i__, M471Ptr<integer> j, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dt, M471Ptr<real> zw1, M471Ptr<real> dz1, M471Ptr<real> pres1, M471Ptr<real> rho1, 
	M471Ptr<integer> momentum_opt__, M471Ptr<integer> tke_opt__, M471Ptr<integer> scalar_opt__, 
	M471Ptr<integer> aerosol_opt__, M471Ptr<integer> numcon_opt__, M471Ptr<real> bl_mynn_closure__, M471Ptr<real> u1, M471Ptr<real> v1, M471Ptr<real> w1, M471Ptr<real> th1, M471Ptr<real> thl1, M471Ptr<real> thv1, M471Ptr<real> tk1, M471Ptr<real> qt1, M471Ptr<real> qv1, M471Ptr<real> qc1, M471Ptr<real> qke1, M471Ptr<real> qsq1, M471Ptr<real> qnc1, M471Ptr<real> qni1, M471Ptr<real> qnwfa1, M471Ptr<real> qnifa1, 
	M471Ptr<real> qnbca1, M471Ptr<real> ex1, M471Ptr<real> vt1, M471Ptr<real> vq1, M471Ptr<real> sgm1, M471Ptr<real> ust, 
	M471Ptr<real> flt, M471Ptr<real> fltv, M471Ptr<real> flq, M471Ptr<real> flqv, M471Ptr<real> pblh, M471Ptr<integer> kpbl, M471Ptr<real> dx, M471Ptr<real> landsea, M471Ptr<real> ts, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_qt1__, M471Ptr<real> edmf_thl1__, M471Ptr<real> edmf_ent1__, 
	M471Ptr<real> edmf_qc1__, M471Ptr<real> edmf_qv1__, M471Ptr<real> edmf_u1__, M471Ptr<real> edmf_v1__, 
	M471Ptr<real> s_aw1__, M471Ptr<real> s_awthl1__, M471Ptr<real> s_awqt1__, M471Ptr<real> s_awqv1__, 
	M471Ptr<real> s_awqc1__, M471Ptr<real> s_awu1__, M471Ptr<real> s_awv1__, M471Ptr<real> s_awqke1__, 
	M471Ptr<real> s_awqsq1__, M471Ptr<real> s_awqnc1__, M471Ptr<real> s_awqni1__, M471Ptr<real> s_awqnwfa1__, M471Ptr<real> s_awqnifa1__, M471Ptr<real> s_awqnbca1__, M471Ptr<real> sub_thl1__, M471Ptr<real> sub_sqv1__, M471Ptr<real> sub_u1__, M471Ptr<real> sub_v1__, M471Ptr<real> det_thl1__, M471Ptr<real> det_sqv1__, M471Ptr<real> det_sqc1__, M471Ptr<real> det_u1__, M471Ptr<real> det_v1__, M471Ptr<integer> nchem, M471Ptr<real> chem1, M471Ptr<real> s_awchem1__, M471Ptr<logical> mix_chem__, M471Ptr<integer> nscalars, M471Ptr<real> scalars, M471Ptr<real> s_awscalars1__, 
	M471Ptr<real> qc_bl1__, M471Ptr<real> cldfra_bl1__, M471Ptr<real> qc_bl1_old__, M471Ptr<real> cldfra_bl1_old__, M471Ptr<logical> f_qc__, M471Ptr<logical> f_qi__, M471Ptr<logical> f_qnc__, 
	M471Ptr<logical> f_qni__, M471Ptr<logical> f_qnwfa__, M471Ptr<logical> f_qnifa__, M471Ptr<logical> f_qnbca__, M471Ptr<real> psig_shcu__, M471Ptr<real> maxwidth, M471Ptr<integer> ktop, M471Ptr<real> maxmf, M471Ptr<real> ztop, M471Ptr<real> excess_h__, M471Ptr<real> excess_q__, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__, M471Ptr<real> tkeprod_up__, M471Ptr<real> el1);
__device__ M471_CALL /* Subroutine */ int condensation_edmf__(unsigned char *m471_work, int m471_n, M471Ptr<real> qt, M471Ptr<real> thl, M471Ptr<real> p, M471Ptr<real> zagl, M471Ptr<real> thv, M471Ptr<real> qc);
__device__ M471_CALL /* Subroutine */ int condensation_ddmf__(unsigned char *m471_work, int m471_n, M471Ptr<real> qt, M471Ptr<real> thl, M471Ptr<real> p, M471Ptr<real> zagl, M471Ptr<real> thv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<integer> debug_dd__);
__device__ M471_CALL /* Subroutine */ int condensation_edmf_r__(unsigned char *m471_work, int m471_n, M471Ptr<real> qt, M471Ptr<real> thl, M471Ptr<real> p, M471Ptr<real> zagl, M471Ptr<real> thv, M471Ptr<real> qc);
__device__ M471_CALL /* Subroutine */ int ddmp_mf__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dt, M471Ptr<real> dx,
	 M471Ptr<real> zw, M471Ptr<real> dz, M471Ptr<real> p, M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> th, M471Ptr<real> thl, 
	M471Ptr<real> thv, M471Ptr<real> tk, M471Ptr<real> qt, M471Ptr<real> qv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<real> qnc, M471Ptr<real> qni, M471Ptr<real> qnwfa, M471Ptr<real> qnifa, M471Ptr<real> qke, M471Ptr<real> rho, M471Ptr<real> exner, M471Ptr<real> qc_bl1__, M471Ptr<real> qi_bl1__, M471Ptr<real> cldfra_bl1__, M471Ptr<real> ust, 
	M471Ptr<real> flt, M471Ptr<real> flq, M471Ptr<real> fltv, M471Ptr<real> pblh, M471Ptr<integer> kpbl, M471Ptr<real> edmf_a_dd__, M471Ptr<real> edmf_w_dd__, M471Ptr<real> edmf_qt_dd__, M471Ptr<real> edmf_thl_dd__, M471Ptr<real> edmf_ent_dd__, M471Ptr<real> edmf_qc_dd__, M471Ptr<real> sd_aw__,
	 M471Ptr<real> sd_awthl__, M471Ptr<real> sd_awqt__, M471Ptr<real> sd_awqv__, M471Ptr<real> sd_awqc__, 
	M471Ptr<real> sd_awqi__, M471Ptr<real> sd_awqnc__, M471Ptr<real> sd_awqni__, M471Ptr<real> sd_awqnwfa__, M471Ptr<real> sd_awqnifa__, M471Ptr<real> sd_awu__, M471Ptr<real> sd_awv__, 
	M471Ptr<real> sd_awqke__, M471Ptr<real> tkeprod_dn__, M471Ptr<real> el, M471Ptr<real> rthraten, M471Ptr<real> psig, M471Ptr<real> maxmf_dd__, M471Ptr<real> maxwidth_dd__);
__device__ M471_CALL /* Subroutine */ int scale_aware__(unsigned char *m471_work, int m471_n, M471Ptr<real> dx, M471Ptr<real> pblh, M471Ptr<real> psig_bl__, 
	M471Ptr<real> psig_shcu__);
__device__ M471_CALL real esat_blend__(unsigned char *m471_work, int m471_n, M471Ptr<real> t);
__device__ M471_CALL real qsat_blend__(unsigned char *m471_work, int m471_n, M471Ptr<real> t, M471Ptr<real> p);
__device__ M471_CALL real xl_blend__(unsigned char *m471_work, int m471_n, M471Ptr<real> t);
__device__ M471_CALL real phim_(unsigned char *m471_work, int m471_n, M471Ptr<real> zet);
__device__ M471_CALL real phih_(unsigned char *m471_work, int m471_n, M471Ptr<real> zet);
__device__ M471_CALL /* Subroutine */ int topdown_cloudrad__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dz1,
	 M471Ptr<real> zw, M471Ptr<real> fltv, M471Ptr<real> u1, M471Ptr<real> v1, M471Ptr<real> xland, M471Ptr<integer> kpbl,
	 M471Ptr<real> pblh, M471Ptr<real> sqc, M471Ptr<real> sqi, M471Ptr<real> sqw, M471Ptr<real> thl, M471Ptr<real> th1, 
	M471Ptr<real> ex1, M471Ptr<real> pres1, M471Ptr<real> rho1, M471Ptr<real> thv, M471Ptr<real> cldfra_bl1__, 
	M471Ptr<real> qc_bl__, M471Ptr<real> qi_bl__, M471Ptr<real> rthraten, M471Ptr<real> tkeprod_dn__, 
	M471Ptr<real> psig, M471Ptr<real> maxtkeprod, M471Ptr<real> cldtop_cooling__, M471Ptr<real> ent_eff__);
__device__ M471_CALL integer ceiling_(unsigned char *m471_work, int m471_n, M471Ptr<real> x);
/* Table of constant values */

static __device__ integer c__0 = 0;

template<int M471_PHASE>
__device__ M471_CALL /* Subroutine */ int mynnedmf_(unsigned char *m471_work, int m471_n, M471Ptr<integer> i__, M471Ptr<integer> j, M471Ptr<integer> initflag, 
	M471Ptr<logical> restart, M471Ptr<logical> cycling, M471Ptr<real> delt, M471Ptr<real> dz1, M471Ptr<real> dx, 
	M471Ptr<real> u1, M471Ptr<real> v1, M471Ptr<real> w1, M471Ptr<real> th1, M471Ptr<real> sqv1, M471Ptr<real> sqc1, M471Ptr<real> sqi1, M471Ptr<real> sqs1, M471Ptr<real> qnc1, M471Ptr<real> qni1, M471Ptr<real> qnwfa1, M471Ptr<real> qnifa1,
	 M471Ptr<real> qnbca1, M471Ptr<real> ozone1, M471Ptr<real> pres1, M471Ptr<real> ex1, M471Ptr<real> rho1, M471Ptr<real> tk1, M471Ptr<real> xland, M471Ptr<real> ts, M471Ptr<real> qsfc, M471Ptr<real> ps, M471Ptr<real> ust, M471Ptr<real> ch, M471Ptr<real> hfx, M471Ptr<real> qfx, M471Ptr<real> znt, M471Ptr<real> wspd, M471Ptr<real> uoce, M471Ptr<real> voce, M471Ptr<real> qke1, M471Ptr<real> qke_adv1__, M471Ptr<real> el1, M471Ptr<real> sh1, M471Ptr<real> sm1, 
	M471Ptr<real> kh1, M471Ptr<real> km1, M471Ptr<integer> nchem, M471Ptr<integer> ndvel, M471Ptr<real> chem1, 
	M471Ptr<real> vdep, M471Ptr<real> frp, M471Ptr<real> emis_ant_no__, M471Ptr<logical> mix_chem__, 
	M471Ptr<logical> enh_mix__, M471Ptr<real> settle1, M471Ptr<real> scalars, M471Ptr<integer> nscalars, 
	M471Ptr<real> tsq1, M471Ptr<real> qsq1, M471Ptr<real> cov1, M471Ptr<real> du1, M471Ptr<real> dv1, M471Ptr<real> dth1, 
	M471Ptr<real> dqv1, M471Ptr<real> dqc1, M471Ptr<real> dqi1, M471Ptr<real> dqnc1, M471Ptr<real> dqni1, M471Ptr<real> dqs1, M471Ptr<real> dqnwfa1, M471Ptr<real> dqnifa1, M471Ptr<real> dqnbca1, M471Ptr<real> dozone1, 
	M471Ptr<real> rthraten1, M471Ptr<real> pblh, M471Ptr<integer> kpbl, M471Ptr<real> maxwidth_dd__, M471Ptr<real> maxwidth, M471Ptr<real> maxmf, M471Ptr<real> ztop_plume__, M471Ptr<real> excess_h__, M471Ptr<real> excess_q__, M471Ptr<real> maxmf_dd__, M471Ptr<real> maxtkeprod, M471Ptr<real> cldtop_cooling__, M471Ptr<real> ent_eff__, M471Ptr<real> dqke1, M471Ptr<real> qwt1, M471Ptr<real> qshear1, M471Ptr<real> qbuoy1, M471Ptr<real> qdiss1, M471Ptr<real> qc_bl1__, M471Ptr<real> qi_bl1__, 
	M471Ptr<real> cldfra_bl1__, M471Ptr<logical> bl_mynn_tkeadvect__, M471Ptr<integer> tke_budget__, M471Ptr<integer> bl_mynn_cloudpdf__, M471Ptr<integer> bl_mynn_mixlength__, M471Ptr<real> bl_mynn_closure__, M471Ptr<integer> bl_mynn_ess__, 
	M471Ptr<integer> bl_mynn_edmf__, M471Ptr<integer> bl_mynn_edmf_mom__, M471Ptr<integer> bl_mynn_edmf_tke__, M471Ptr<integer> bl_mynn_mixscalars__, M471Ptr<integer> bl_mynn_mixaerosols__, M471Ptr<integer> bl_mynn_mixnumcon__, M471Ptr<integer> bl_mynn_output__, M471Ptr<integer> bl_mynn_cloudmix__, M471Ptr<integer> bl_mynn_mixqt__, M471Ptr<integer> bl_mynn_edmf_dd__, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_qt1__, M471Ptr<real> edmf_thl1__, M471Ptr<real> edmf_ent1__, 
	M471Ptr<real> edmf_qc1__, M471Ptr<real> sub_thl1__, M471Ptr<real> sub_sqv1__, M471Ptr<real> det_thl1__, M471Ptr<real> det_sqv1__, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__, M471Ptr<logical> flag_qc__, M471Ptr<logical> flag_qi__, M471Ptr<logical> flag_qnc__, M471Ptr<logical> flag_qni__, M471Ptr<logical> flag_qs__, M471Ptr<logical> flag_qnwfa__, M471Ptr<logical> flag_qnifa__, M471Ptr<logical> flag_qnbca__, M471Ptr<logical> flag_ozone__, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<integer> errflg) {
    /* Initialized data */

    real karman = .4000000059604644775390625f;
    real rc0000 = 1.f;
    real rc0001 = 80.f;
    real rc0002 = 1200.f;
    real rc0003 = .001f;
    real rc0004 = 270.f;
    real rc0005 = 380.f;
    real rc0006 = 160.f;
    real rc0007 = 3.402823466e38f;
    real rc0008 = 2e-4f;
    real rc0009 = 700.f;
    real rc0010 = .84f;
    real rc0011 = 1e-8f;
    real rc0012 = 1e-6f;
    real rc0013 = 10.f;
    real rc0014 = 1.5f;
    real rc0015 = 3.f;
    real rc0016 = .25f;
    real rc0017 = .95f;
    real rc0018 = 3.5f;
    real rc0019 = .002f;
    real rc0020 = 1e4f;
    real rc0021 = 1e-12f;
    real rc0022 = 4.f;
    real p608 = .6083624362945556640625f;
    real gtr = .032700002193450927734375f;
    real p1000mb = 1e5f;
    real xlscp = 2837.232421875f;
    real xlvcp = 2488.80029296875f;
    real cp = 1004.5f;
    real cphm_unst__ = 16.f;
    real cphh_st__ = 5.f;
    real cphh_unst__ = 16.f;
    real b1 = 24.f;
    real rcp = .2857142984867095947265625f;
    real zero = 0.f;
    real one = 1.f;
    real five = 5.f;
    real p01 = .00999999977648258209228515625f;
    real p5 = .5f;

    /* System generated locals */
    integer dz1_offset, u1_offset, v1_offset, w1_offset, th1_offset, pres1_offset, ex1_offset, rho1_offset, tk1_offset, rthraten1_offset, sqv1_offset, sqc1_offset, sqi1_offset, sqs1_offset, qni1_offset, qnc1_offset, qnwfa1_offset, qnifa1_offset, qnbca1_offset, ozone1_offset, qke1_offset, tsq1_offset, qsq1_offset, cov1_offset, qke_adv1_offset, sh1_offset, sm1_offset, el1_offset, du1_offset, dv1_offset, dth1_offset, dqv1_offset, dqc1_offset, dqi1_offset, dqs1_offset, dqni1_offset, dqnc1_offset, dqnwfa1_offset, dqnifa1_offset, dqnbca1_offset, dozone1_offset, qc_bl1_offset, qi_bl1_offset, cldfra_bl1_offset, edmf_a1_offset, edmf_w1_offset, edmf_qt1_offset, edmf_thl1_offset, edmf_ent1_offset, edmf_qc1_offset, sub_thl1_offset, sub_sqv1_offset, det_thl1_offset, det_sqv1_offset, qwt1_offset, qshear1_offset, qbuoy1_offset, qdiss1_offset, dqke1_offset, kh1_offset, km1_offset, chem1_dim1, chem1_offset, settle1_dim1, settle1_offset, scalars_dim1, scalars_offset, pattern_spp_pbl1_offset, i__1, i__2;
    real r__1, r__2, r__3;
    doublereal d__1, d__2;

    /* Builtin functions */
    

    /* Local variables */
    M471Ptr<real> s_awthl1__=m471_array<real>(m471_work,0,m471_n);
M471Ptr<real> s_awqni1__=m471_array<real>(m471_work,(4*(MYNN471_NZ-49)+200),m471_n);
M471Ptr<real> sd_awqt1__=m471_array<real>(m471_work,(8*(MYNN471_NZ-49)+400),m471_n);
M471Ptr<real> sd_awqv1__=m471_array<real>(m471_work,(12*(MYNN471_NZ-49)+600),m471_n);
M471Ptr<real> thl_tot1__=m471_array<real>(m471_work,(16*(MYNN471_NZ-49)+800),m471_n);
M471Ptr<real> s_awqsq1__=m471_array<real>(m471_work,(20*(MYNN471_NZ-49)+996),m471_n);
    
    integer kproblem;
    M471Ptr<real> s_awchem1__=m471_array<real>(m471_work,(24*(MYNN471_NZ-49)+1196),m471_n);
M471Ptr<real> sd_awqke1__=m471_array<real>(m471_work,(28*(MYNN471_NZ-49)+1396),m471_n);
M471Ptr<real> sd_awqnc1__=m471_array<real>(m471_work,(32*(MYNN471_NZ-49)+1596),m471_n);
M471Ptr<real> sd_awthl1__=m471_array<real>(m471_work,(36*(MYNN471_NZ-49)+1796),m471_n);
M471Ptr<real> sd_awqni1__=m471_array<real>(m471_work,(40*(MYNN471_NZ-49)+1996),m471_n);
    integer k;
    
    real local_psig_shcu__; real &psig_shcu__=M471_PHASE<0 ? local_psig_shcu__ : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+0),m471_n);
M471Ptr<real> edmf_a_dd1__=m471_array<real>(m471_work,(44*(MYNN471_NZ-49)+2196),m471_n);
M471Ptr<real> qc_bl1_old__=m471_array<real>(m471_work,(48*(MYNN471_NZ-49)+2392),m471_n);
M471Ptr<real> edmf_w_dd1__=m471_array<real>(m471_work,(52*(MYNN471_NZ-49)+2588),m471_n);
M471Ptr<real> qi_bl1_old__=m471_array<real>(m471_work,(56*(MYNN471_NZ-49)+2784),m471_n);
M471Ptr<real> s_awqnbca1__=m471_array<real>(m471_work,(60*(MYNN471_NZ-49)+2980),m471_n);
M471Ptr<real> diss_heat1__=m471_array<real>(m471_work,(64*(MYNN471_NZ-49)+3180),m471_n);
M471Ptr<real> s_awqnifa1__=m471_array<real>(m471_work,(68*(MYNN471_NZ-49)+3376),m471_n);
    integer ic;
    M471Ptr<real> s_awqnwfa1__=m471_array<real>(m471_work,(72*(MYNN471_NZ-49)+3576),m471_n);
M471Ptr<real> qc1=m471_array<real>(m471_work,(76*(MYNN471_NZ-49)+3776),m471_n);
M471Ptr<real> edmf_qc_dd1__=m471_array<real>(m471_work,(80*(MYNN471_NZ-49)+3972),m471_n);
M471Ptr<real> tkeprod_dn__=m471_array<real>(m471_work,(84*(MYNN471_NZ-49)+4168),m471_n);
M471Ptr<real> qi1=m471_array<real>(m471_work,(88*(MYNN471_NZ-49)+4364),m471_n);
M471Ptr<real> qs1=m471_array<real>(m471_work,(92*(MYNN471_NZ-49)+4560),m471_n);
M471Ptr<real> qv1=m471_array<real>(m471_work,(96*(MYNN471_NZ-49)+4756),m471_n);
M471Ptr<real> vq1=m471_array<real>(m471_work,(100*(MYNN471_NZ-49)+4952),m471_n);
M471Ptr<real> vt1=m471_array<real>(m471_work,(104*(MYNN471_NZ-49)+5148),m471_n);
M471Ptr<real> edmf_qt_dd1__=m471_array<real>(m471_work,(108*(MYNN471_NZ-49)+5344),m471_n);
    integer local_ktop_plume__; integer &ktop_plume__=M471_PHASE<0 ? local_ktop_plume__ : *m471_array<integer>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+8),m471_n);
    M471Ptr<real> tkeprod_up__=m471_array<real>(m471_work,(112*(MYNN471_NZ-49)+5540),m471_n);
M471Ptr<real> zw1=m471_array<real>(m471_work,(116*(MYNN471_NZ-49)+5736),m471_n);
M471Ptr<real> sd_awqnifa1__=m471_array<real>(m471_work,(120*(MYNN471_NZ-49)+5936),m471_n);
    integer imd, jmd;
    real cpm;
    real local_flq; real &flq=M471_PHASE<0 ? local_flq : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+16),m471_n);
    real local_flt; real &flt=M471_PHASE<0 ? local_flt : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+24),m471_n);
    real local_phh; real &phh=M471_PHASE<0 ? local_phh : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+32),m471_n);
    real local_zet; real &zet=M471_PHASE<0 ? local_zet : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+40),m471_n);
M471Ptr<real> sd_awqnwfa1__=m471_array<real>(m471_work,(124*(MYNN471_NZ-49)+6136),m471_n);
    
    real wsp;
    real local_pmz; real &pmz=M471_PHASE<0 ? local_pmz : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+48),m471_n);
M471Ptr<real> dfh1=m471_array<real>(m471_work,(128*(MYNN471_NZ-49)+6336),m471_n);
M471Ptr<real> dfm1=m471_array<real>(m471_work,(132*(MYNN471_NZ-49)+6532),m471_n);
M471Ptr<real> dfq1=m471_array<real>(m471_work,(136*(MYNN471_NZ-49)+6728),m471_n);
M471Ptr<real> tcd1=m471_array<real>(m471_work,(140*(MYNN471_NZ-49)+6924),m471_n);
M471Ptr<real> qcd1=m471_array<real>(m471_work,(144*(MYNN471_NZ-49)+7120),m471_n);
M471Ptr<real> pdk1=m471_array<real>(m471_work,(148*(MYNN471_NZ-49)+7316),m471_n);
M471Ptr<real> pdc1=m471_array<real>(m471_work,(152*(MYNN471_NZ-49)+7512),m471_n);
M471Ptr<real> pdq1=m471_array<real>(m471_work,(156*(MYNN471_NZ-49)+7708),m471_n);
M471Ptr<real> thl1=m471_array<real>(m471_work,(160*(MYNN471_NZ-49)+7904),m471_n);
M471Ptr<real> pdt1=m471_array<real>(m471_work,(164*(MYNN471_NZ-49)+8100),m471_n);
M471Ptr<real> sgm1=m471_array<real>(m471_work,(168*(MYNN471_NZ-49)+8296),m471_n);
M471Ptr<real> edmf_thl_dd1__=m471_array<real>(m471_work,(172*(MYNN471_NZ-49)+8492),m471_n);
M471Ptr<real> thv1=m471_array<real>(m471_work,(176*(MYNN471_NZ-49)+8688),m471_n);
M471Ptr<real> edmf_ent_dd1__=m471_array<real>(m471_work,(180*(MYNN471_NZ-49)+8884),m471_n);
    
    real local_flqc; real &flqc=M471_PHASE<0 ? local_flqc : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+56),m471_n);
M471Ptr<real> sqw1=m471_array<real>(m471_work,(184*(MYNN471_NZ-49)+9080),m471_n);
M471Ptr<real> s_awscalars1__=m471_array<real>(m471_work,(188*(MYNN471_NZ-49)+9276),m471_n);
    
    real sqcg, red001;
    real local_taue; real &taue=M471_PHASE<0 ? local_taue : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+64),m471_n);
    real local_flqv; real &flqv=M471_PHASE<0 ? local_flqv : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+72),m471_n);
    real local_rmol; real &rmol=M471_PHASE<0 ? local_rmol : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+80),m471_n);
    real local_fltv; real &fltv=M471_PHASE<0 ? local_fltv : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+88),m471_n);
    real local_delt2; real &delt2=M471_PHASE<0 ? local_delt2 : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+96),m471_n);
    real local_phi_m__; real &phi_m__=M471_PHASE<0 ? local_phi_m__ : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+104),m471_n);
    real local_ckmod; real &ckmod=M471_PHASE<0 ? local_ckmod : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+112),m471_n);
    real local_rmolh; real &rmolh=M471_PHASE<0 ? local_rmolh : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+120),m471_n);
M471Ptr<real> s_aw1__=m471_array<real>(m471_work,(192*(MYNN471_NZ-49)+9476),m471_n);
M471Ptr<real> thlv1=m471_array<real>(m471_work,(196*(MYNN471_NZ-49)+9676),m471_n);
M471Ptr<real> sd_aw1__=m471_array<real>(m471_work,(200*(MYNN471_NZ-49)+9872),m471_n);
M471Ptr<real> det_u1__=m471_array<real>(m471_work,(204*(MYNN471_NZ-49)+10072),m471_n);
M471Ptr<real> det_v1__=m471_array<real>(m471_work,(208*(MYNN471_NZ-49)+10268),m471_n);
M471Ptr<real> cldfra_bl1_old__=m471_array<real>(m471_work,(212*(MYNN471_NZ-49)+10464),m471_n);
    
    M471Ptr<real> sub_u1__=m471_array<real>(m471_work,(216*(MYNN471_NZ-49)+10660),m471_n);
M471Ptr<real> sub_v1__=m471_array<real>(m471_work,(220*(MYNN471_NZ-49)+10856),m471_n);
M471Ptr<real> s_awu1__=m471_array<real>(m471_work,(224*(MYNN471_NZ-49)+11052),m471_n);
M471Ptr<real> s_awv1__=m471_array<real>(m471_work,(228*(MYNN471_NZ-49)+11252),m471_n);
    integer ixarr1, ixarr2;
    M471Ptr<real> dummy1=m471_array<real>(m471_work,(232*(MYNN471_NZ-49)+11452),m471_n);
M471Ptr<real> dummy2=m471_array<real>(m471_work,(236*(MYNN471_NZ-49)+11648),m471_n);
M471Ptr<real> kzero1=m471_array<real>(m471_work,(240*(MYNN471_NZ-49)+11844),m471_n);
    
    real exnerg;
    real local_th_sfc__; real &th_sfc__=M471_PHASE<0 ? local_th_sfc__ : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+128),m471_n);
    real local_upwind; real &upwind=M471_PHASE<0 ? local_upwind : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+136),m471_n);
M471Ptr<real> edmf_u1__=m471_array<real>(m471_work,(244*(MYNN471_NZ-49)+12040),m471_n);
M471Ptr<real> edmf_v1__=m471_array<real>(m471_work,(248*(MYNN471_NZ-49)+12236),m471_n);
M471Ptr<real> s_awqc1__=m471_array<real>(m471_work,(252*(MYNN471_NZ-49)+12432),m471_n);
    logical initialize_qke__;
    M471Ptr<real> sd_awu1__=m471_array<real>(m471_work,(256*(MYNN471_NZ-49)+12632),m471_n);
M471Ptr<real> sd_awv1__=m471_array<real>(m471_work,(260*(MYNN471_NZ-49)+12832),m471_n);
M471Ptr<real> qc_tot1__=m471_array<real>(m471_work,(264*(MYNN471_NZ-49)+13032),m471_n);
M471Ptr<real> qi_tot1__=m471_array<real>(m471_work,(268*(MYNN471_NZ-49)+13228),m471_n);
M471Ptr<real> s_awqt1__=m471_array<real>(m471_work,(272*(MYNN471_NZ-49)+13424),m471_n);
M471Ptr<real> s_awqv1__=m471_array<real>(m471_work,(276*(MYNN471_NZ-49)+13624),m471_n);
    
    real local_psig_bl__; real &psig_bl__=M471_PHASE<0 ? local_psig_bl__ : *m471_array<real>(m471_work,((((964*(MYNN471_NZ-49)+47980)+7)/8*8)+144),m471_n);
    logical problem;
    M471Ptr<real> edmf_qv1__=m471_array<real>(m471_work,(280*(MYNN471_NZ-49)+13824),m471_n);
M471Ptr<real> det_sqc1__=m471_array<real>(m471_work,(284*(MYNN471_NZ-49)+14020),m471_n);
M471Ptr<real> sd_awqc1__=m471_array<real>(m471_work,(288*(MYNN471_NZ-49)+14216),m471_n);
M471Ptr<real> sd_awqi1__=m471_array<real>(m471_work,(292*(MYNN471_NZ-49)+14416),m471_n);
M471Ptr<real> s_awqke1__=m471_array<real>(m471_work,(296*(MYNN471_NZ-49)+14616),m471_n);
M471Ptr<real> s_awqnc1__=m471_array<real>(m471_work,(300*(MYNN471_NZ-49)+14816),m471_n);
    

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
    scalars_dim1 = *kte - *kts + 1;
    scalars_offset = *kts + scalars_dim1;
    scalars -= scalars_offset;
    settle1_dim1 = *kte - *kts + 1;
    settle1_offset = *kts + settle1_dim1;
    settle1 -= settle1_offset;
    chem1_dim1 = *kte - *kts + 1;
    chem1_offset = *kts + chem1_dim1;
    chem1 -= chem1_offset;
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
    pres1_offset = *kts;
    pres1 -= pres1_offset;
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
    ckmod = rc0000;
    taue = rc0001;
    upwind = rc0000;
    *errflg = 0;
    if (FALSE_) {
	problem = FALSE_;
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing 2nd power */
	    r__1 = u1[k];
/* Computing 2nd power */
	    r__2 = v1[k];
	    wsp = (real)sqrt((real)(r__1 * r__1 + r__2 * r__2));
	    if (abs(*hfx) > rc0002 || abs(*qfx) > rc0003 || wsp > rc0004 || 
		    tk1[k] > rc0005 || tk1[k] < rc0006 || sqv1[k] < zero || 
		    sqc1[k] < zero || *ps - pres1[*kts] < zero) {
		kproblem = k;
		problem = TRUE_;
	    }
/* L9004: */
	}
/* L9005: */
	if (problem) {
	}
    }
    /* unread private value */;
    /* unread private value */;
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_a1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_w1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_qt1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_thl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_ent1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_qc1__[*kts + ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	if (*bl_mynn_edmf__ > 0) /* m480 unread unless */ { edmf_u1__[ixarr1 - 1] = zero; }
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	if (*bl_mynn_edmf__ > 0) /* m480 unread unless */ { edmf_v1__[ixarr1 - 1] = zero; }
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	if (*bl_mynn_edmf__ > 0) /* m480 unread unless */ { edmf_qv1__[ixarr1 - 1] = zero; }
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_thl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_sqv1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_thl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_sqv1__[*kts + ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	edmf_a_dd1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	edmf_w_dd1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	edmf_qc_dd1__[ixarr1 - 1] = zero;
    }
    if (*bl_mynn_edmf__ > 0) /* m480 unread unless */ { ktop_plume__ = 0; }
    *ztop_plume__ = zero;
    *maxwidth = zero;
    *maxmf = zero;
    *excess_h__ = zero;
    *excess_q__ = zero;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	kzero1[ixarr1 - 1] = zero;
    }
    if (*initflag > 0 && ! (*restart)) {
	if (*restart || *cycling) {
	    red001 = -rc0007;
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
/* Computing MAX */
		r__1 = red001, r__2 = qke1[*kts + ixarr1 - 1];
		red001 = max(r__1,r__2);
	    }
	    if (red001 < rc0008) {
		initialize_qke__ = TRUE_;
	    } else {
		initialize_qke__ = FALSE_;
	    }
	} else {
	    initialize_qke__ = TRUE_;
	}
	if (! (*restart) || ! (*cycling)) {
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		sh1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		sm1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		el1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		tsq1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qsq1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		cov1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		cldfra_bl1__[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qc_bl1__[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qi_bl1__[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qke1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qke_adv1__[*kts + ixarr1 - 1] = zero;
	    }
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dqc1[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dqi1[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dqni1[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dqnc1[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dqnwfa1[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dqnifa1[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dqnbca1[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dozone1[*kts + ixarr1 - 1] = zero;
	}
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    if (*bl_mynn_edmf__ > 0) /* m480 unread unless */ { qc_bl1_old__[ixarr1 - 1] = zero; }
	}
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    if (*bl_mynn_edmf__ > 0) /* m480 unread unless */ { cldfra_bl1_old__[ixarr1 - 1] = zero; }
	}
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    sgm1[ixarr1 - 1] = zero;
	}
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    vt1[ixarr1 - 1] = zero;
	}
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    vq1[ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    km1[*kts + ixarr1 - 1] = zero;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    kh1[*kts + ixarr1 - 1] = zero;
	}
	if (*tke_budget__ == 1) {
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qwt1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qshear1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qbuoy1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		qdiss1[*kts + ixarr1 - 1] = zero;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		dqke1[*kts + ixarr1 - 1] = zero;
	    }
	}
	zw1[*kts - 1] = zero;
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    sqw1[k - 1] = sqv1[k] + sqc1[k] + sqi1[k];
	    thl1[k - 1] = th1[k] - xlvcp / ex1[k] * sqc1[k] - xlscp / ex1[k] *
		     sqi1[k];
	    thlv1[k - 1] = thl1[k - 1] * (one + p608 * sqv1[k]);
	    thv1[k - 1] = th1[k] * (one + p608 * sqv1[k] - (sqc1[k] + sqi1[k])
		    );
	    zw1[k] = zw1[k - 1] + dz1[k];
/* L9007: */
	}
/* L9008: */
	if (initialize_qke__) {
	    i__1 = *kte;
	    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
		r__1 = (*ust * rc0009 - zw1[k - 1]) / (max(*ust,p01) * rc0009)
			;
		qke1[k] = five * *ust * max(r__1,p01);
/* L9010: */
	    }
/* L9011: */
	}
	get_pblh__(m471_work,m471_n, kts, kte, pblh, thv1, &qke1[qke1_offset], ust, zw1, &dz1[
		dz1_offset], xland, kpbl);
	scale_aware__(m471_work,m471_n, dx, pblh, &psig_bl__, &psig_shcu__);
	rmol = zero;
	rmolh = zero;
	mym_initialize__(m471_work,m471_n, kts, kte, &ckmod, &taue, xland, pblh, &dz1[
		dz1_offset], dx, zw1, &pres1[pres1_offset], &ex1[ex1_offset], 
		&u1[u1_offset], &v1[v1_offset], thl1, &sqv1[sqv1_offset], &
		th1[th1_offset], thv1, thlv1, &sh1[sh1_offset], &sm1[
		sm1_offset], ust, &rmol, &rmolh, &el1[el1_offset], &qke1[
		qke1_offset], &tsq1[tsq1_offset], &qsq1[qsq1_offset], &cov1[
		cov1_offset], &psig_bl__, &cldfra_bl1__[cldfra_bl1_offset], 
		bl_mynn_mixlength__, bl_mynn_ess__, &edmf_w1__[edmf_w1_offset]
		, &edmf_a1__[edmf_a1_offset], edmf_w_dd1__, edmf_a_dd1__, &
		initialize_qke__, spp_pbl__, &pattern_spp_pbl1__[
		pattern_spp_pbl1_offset]);
	if (! (*restart)) {
	    if (*bl_mynn_tkeadvect__) {
		i__1 = *kte;
		
#pragma unroll 4
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
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    qke1[*kts + ixarr1 - 1] = qke_adv1__[*kts + ixarr1 - 1];
	}
    }
    if (*tke_budget__ == 1) {
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    dqke1[*kts + ixarr1 - 1] = qke1[*kts + ixarr1 - 1];
	}
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	if (*bl_mynn_edmf__ > 0) /* m480 unread unless */ { cldfra_bl1_old__[*kts + ixarr1 - 2] = cldfra_bl1__[*kts + ixarr1 - 1]; }
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	if (*bl_mynn_edmf__ > 0) /* m480 unread unless */ { qc_bl1_old__[*kts + ixarr1 - 2] = qc_bl1__[*kts + ixarr1 - 1]; }
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	/* unread private value */;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	cldfra_bl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	qc_bl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	qi_bl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dqc1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dqi1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dqs1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dqni1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dqnc1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dqnwfa1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dqnifa1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dqnbca1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dozone1[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_a1__[*kts + ixarr1 - 1] = zero;
    }
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears edmf_w1__ */
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_w1__[*kts + ixarr1 - 1] = zero;
    }
    }
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears edmf_qc1__ */
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_qc1__[*kts + ixarr1 - 1] = zero;
    }
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	if (*bl_mynn_edmf__ > 0) /* m480 unread unless */ { edmf_qv1__[ixarr1 - 1] = zero; }
    }
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears edmf_thl1__ */
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_thl1__[*kts + ixarr1 - 1] = zero;
    }
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	if (*bl_mynn_edmf__ > 0) /* m480 unread unless */ { edmf_u1__[ixarr1 - 1] = zero; }
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	if (*bl_mynn_edmf__ > 0) /* m480 unread unless */ { edmf_v1__[ixarr1 - 1] = zero; }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears s_aw1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_aw1__[ixarr1 - 1] = zero;
    }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears s_awthl1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awthl1__[ixarr1 - 1] = zero;
    }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears s_awqt1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqt1__[ixarr1 - 1] = zero;
    }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears s_awqv1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqv1__[ixarr1 - 1] = zero;
    }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears s_awqc1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqc1__[ixarr1 - 1] = zero;
    }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears s_awu1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awu1__[ixarr1 - 1] = zero;
    }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears s_awv1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awv1__[ixarr1 - 1] = zero;
    }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears s_awqke1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqke1__[ixarr1 - 1] = zero;
    }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears s_awqsq1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqsq1__[ixarr1 - 1] = zero;
    }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears s_awqnc1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqnc1__[ixarr1 - 1] = zero;
    }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears s_awqni1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqni1__[ixarr1 - 1] = zero;
    }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears s_awqnwfa1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqnwfa1__[ixarr1 - 1] = zero;
    }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears s_awqnifa1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqnifa1__[ixarr1 - 1] = zero;
    }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears s_awqnbca1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	s_awqnbca1__[ixarr1 - 1] = zero;
    }
    }
    i__1 = *nchem;
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= i__1; ++ixarr2) {
	i__2 = *kte + 1 - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
	    s_awchem1__[*kts + ixarr1 - 1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    i__1 = *nscalars;
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= i__1; ++ixarr2) {
	i__2 = *kte + 1 - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
	    s_awscalars1__[*kts + ixarr1 - 1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	edmf_a_dd1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	edmf_w_dd1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	edmf_qc_dd1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_aw1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awthl1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awqt1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awqv1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awqc1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awqi1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awqnc1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awqni1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awqnwfa1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awqnifa1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awu1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	sd_awv1__[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	if (*bl_mynn_edmf_dd__ > 0) /* m480 unread unless */ { sd_awqke1__[ixarr1 - 1] = zero; }
    }
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears sub_thl1__ */
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_thl1__[*kts + ixarr1 - 1] = zero;
    }
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_sqv1__[*kts + ixarr1 - 1] = zero;
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears sub_u1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	sub_u1__[ixarr1 - 1] = zero;
    }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears sub_v1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	sub_v1__[ixarr1 - 1] = zero;
    }
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_thl1__[*kts + ixarr1 - 1] = zero;
    }
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears det_sqv1__ */
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_sqv1__[*kts + ixarr1 - 1] = zero;
    }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears det_sqc1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	det_sqc1__[ixarr1 - 1] = zero;
    }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears det_u1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	det_u1__[ixarr1 - 1] = zero;
    }
    }
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears det_v1__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	det_v1__[ixarr1 - 1] = zero;
    }
    }
    zw1[*kts - 1] = zero;
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	zw1[k] = zw1[k - 1] + dz1[k];
	qv1[k - 1] = sqv1[k] / (one - sqv1[k]);
	qc1[k - 1] = sqc1[k] / (one - sqv1[k]);
	qi1[k - 1] = sqi1[k] / (one - sqv1[k]);
	/* unread private value */;
	sqw1[k - 1] = sqv1[k] + sqc1[k] + sqi1[k];
	thl1[k - 1] = th1[k] - xlvcp / ex1[k] * sqc1[k] - xlscp / ex1[k] * 
		sqi1[k];
	thv1[k - 1] = th1[k] * (one + p608 * sqv1[k] - (sqc1[k] + sqi1[k]));
/* L9016: */
    }
/* L9017: */
    get_pblh__(m471_work,m471_n, kts, kte, pblh, thv1, &qke1[qke1_offset], ust, zw1, &dz1[
	    dz1_offset], xland, kpbl);
    scale_aware__(m471_work,m471_n, dx, pblh, &psig_bl__, &psig_shcu__);
    /* unread private value */;
/* Computing MAX */
    r__1 = sqv1[*kts];
    cpm = cp * (one + rc0010 * max(r__1,rc0011));
    d__1 =  (*ps / p1000mb);
    d__2 =  rcp;
    /* unread private value */;
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
    rmol = -karman * gtr * fltv / max(r__1,rc0012);
/* Computing MAX */
/* Computing 3rd power */
    r__2 = *ust;
    r__1 = r__2 * (r__2 * r__2);
    rmolh = -karman * gtr * flt / max(r__1,rc0012);
    zet = p5 * dz1[*kts] * rmol;
/* Computing MAX */
    r__1 = zet, r__2 = -rc0013;
    zet = max(r__1,r__2);
    zet = min(zet,rc0013);
    if (TRUE_) {
	if (*xland - rc0014 >= zero) {
	    if (zet >= zero) {
		pmz = one + rc0015 * zet;
		phh = one + cphh_st__ * zet;
	    } else {
		d__1 =  (one - cphm_unst__ * zet);
		d__2 =  rc0016;
		pmz = one / (real)pow_dd(&d__1, &d__2) - zet;
		phh = one / (real)sqrt((real)(one - cphh_unst__ * zet));
	    }
	} else {
	    if (zet >= zero) {
		pmz = rc0017 + rc0018 * zet;
		phh = one + cphh_st__ * zet;
	    } else {
		d__1 =  (one - cphm_unst__ * zet);
		d__2 =  rc0016;
		pmz = rc0017 / (real)pow_dd(&d__1, &d__2) - zet;
		phh = one / (real)sqrt((real)(one - cphh_unst__ * zet));
	    }
	}
	phi_m__ = (real)phim_(m471_work,m471_n, &zet);
    } else {
	phi_m__ = (real)phim_(m471_work,m471_n, &zet);
	pmz = phi_m__ - zet;
	phh = (real)phih_(m471_work,m471_n, &zet);
    }

    }
    if constexpr(M471_PHASE<0 || M471_PHASE==1) {
    mym_condensation__(m471_work,m471_n, kts, kte, dx, &dz1[dz1_offset], zw1, xland, thl1, sqw1,
	     &sqv1[sqv1_offset], &sqc1[sqc1_offset], &sqi1[sqi1_offset], &
	    sqs1[sqs1_offset], &pres1[pres1_offset], &ex1[ex1_offset], &tsq1[
	    tsq1_offset], &qsq1[qsq1_offset], &cov1[cov1_offset], &sh1[
	    sh1_offset], &el1[el1_offset], bl_mynn_cloudpdf__, &qc_bl1__[
	    qc_bl1_offset], &qi_bl1__[qi_bl1_offset], &cldfra_bl1__[
	    cldfra_bl1_offset], pblh, hfx, vt1, vq1, &th1[th1_offset], sgm1, 
	    bl_mynn_closure__, spp_pbl__, &pattern_spp_pbl1__[
	    pattern_spp_pbl1_offset]);
    
    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears tkeprod_up__ */
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	tkeprod_up__[ixarr1 - 1] = zero;
    }
    }

    }
    if constexpr(M471_PHASE<0 || M471_PHASE==2) {
    if (*bl_mynn_edmf__ > 0) {
	dmp_mf__(m471_work,m471_n, i__, j, kts, kte, delt, zw1, &dz1[dz1_offset], &pres1[
		pres1_offset], &rho1[rho1_offset], bl_mynn_edmf_mom__, 
		bl_mynn_edmf_tke__, bl_mynn_mixscalars__, 
		bl_mynn_mixaerosols__, bl_mynn_mixnumcon__, bl_mynn_closure__,
		 &u1[u1_offset], &v1[v1_offset], &w1[w1_offset], &th1[
		th1_offset], thl1, thv1, &tk1[tk1_offset], sqw1, &sqv1[
		sqv1_offset], &sqc1[sqc1_offset], &qke1[qke1_offset], &qsq1[
		qsq1_offset], &qnc1[qnc1_offset], &qni1[qni1_offset], &qnwfa1[
		qnwfa1_offset], &qnifa1[qnifa1_offset], &qnbca1[qnbca1_offset]
		, &ex1[ex1_offset], vt1, vq1, sgm1, ust, &flt, &fltv, &flq, &
		flqv, pblh, kpbl, dx, xland, &th_sfc__, &edmf_a1__[
		edmf_a1_offset], &edmf_w1__[edmf_w1_offset], &edmf_qt1__[
		edmf_qt1_offset], &edmf_thl1__[edmf_thl1_offset], &
		edmf_ent1__[edmf_ent1_offset], &edmf_qc1__[edmf_qc1_offset], 
		edmf_qv1__, edmf_u1__, edmf_v1__, s_aw1__, s_awthl1__, 
		s_awqt1__, s_awqv1__, s_awqc1__, s_awu1__, s_awv1__, 
		s_awqke1__, s_awqsq1__, s_awqnc1__, s_awqni1__, s_awqnwfa1__, 
		s_awqnifa1__, s_awqnbca1__, &sub_thl1__[sub_thl1_offset], &
		sub_sqv1__[sub_sqv1_offset], sub_u1__, sub_v1__, &det_thl1__[
		det_thl1_offset], &det_sqv1__[det_sqv1_offset], det_sqc1__, 
		det_u1__, det_v1__, nchem, &chem1[chem1_offset], s_awchem1__, 
		mix_chem__, nscalars, &scalars[scalars_offset], 
		s_awscalars1__, &qc_bl1__[qc_bl1_offset], &cldfra_bl1__[
		cldfra_bl1_offset], qc_bl1_old__, cldfra_bl1_old__, flag_qc__,
		 flag_qi__, flag_qnc__, flag_qni__, flag_qnwfa__, 
		flag_qnifa__, flag_qnbca__, &psig_shcu__, maxwidth, &
		ktop_plume__, maxmf, ztop_plume__, excess_h__, excess_q__, 
		spp_pbl__, &pattern_spp_pbl1__[pattern_spp_pbl1_offset], 
		tkeprod_up__, &el1[el1_offset]);
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	tkeprod_dn__[ixarr1 - 1] = zero;
    }
    *maxtkeprod = zero;
    *cldtop_cooling__ = zero;
    *ent_eff__ = zero;
    *maxmf_dd__ = zero;
    *maxwidth_dd__ = zero;
    if (*bl_mynn_edmf_dd__ > 0) {
	ddmp_mf__(m471_work,m471_n, kts, kte, delt, dx, zw1, &dz1[dz1_offset], &pres1[
		pres1_offset], &u1[u1_offset], &v1[v1_offset], &th1[
		th1_offset], thl1, thv1, &tk1[tk1_offset], sqw1, &sqv1[
		sqv1_offset], &sqc1[sqc1_offset], &sqi1[sqi1_offset], &qnc1[
		qnc1_offset], &qni1[qni1_offset], &qnwfa1[qnwfa1_offset], &
		qnifa1[qnifa1_offset], &qke1[qke1_offset], &rho1[rho1_offset],
		 &ex1[ex1_offset], &qc_bl1__[qc_bl1_offset], &qi_bl1__[
		qi_bl1_offset], &cldfra_bl1__[cldfra_bl1_offset], ust, &flt, &
		flq, &fltv, pblh, kpbl, edmf_a_dd1__, edmf_w_dd1__, 
		edmf_qt_dd1__, edmf_thl_dd1__, edmf_ent_dd1__, edmf_qc_dd1__, 
		sd_aw1__, sd_awthl1__, sd_awqt1__, sd_awqv1__, sd_awqc1__, 
		sd_awqi1__, sd_awqnc1__, sd_awqni1__, sd_awqnwfa1__, 
		sd_awqnifa1__, sd_awu1__, sd_awv1__, sd_awqke1__, 
		tkeprod_dn__, &el1[el1_offset], &rthraten1[rthraten1_offset], 
		&psig_bl__, maxmf_dd__, maxwidth_dd__);
	topdown_cloudrad__(m471_work,m471_n, kts, kte, &dz1[dz1_offset], zw1, &fltv, &u1[*kts], 
		&v1[*kts], xland, kpbl, pblh, &sqc1[sqc1_offset], &sqi1[
		sqi1_offset], sqw1, thl1, &th1[th1_offset], &ex1[ex1_offset], 
		&pres1[pres1_offset], &rho1[rho1_offset], thv1, &cldfra_bl1__[
		cldfra_bl1_offset], &qc_bl1__[qc_bl1_offset], &qi_bl1__[
		qi_bl1_offset], &rthraten1[rthraten1_offset], tkeprod_dn__, &
		psig_bl__, maxtkeprod, cldtop_cooling__, ent_eff__);
    }
    delt2 = *delt;
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	r__1 = qc_bl1__[k], r__2 = sqc1[k];
	qc_tot1__[k - 1] = max(r__1,r__2);
/* Computing MAX */
	r__1 = qi_bl1__[k], r__2 = sqi1[k] + sqs1[k];
	qi_tot1__[k - 1] = max(r__1,r__2);
	thlv1[k - 1] = (th1[k] - xlvcp / ex1[k] * qc_tot1__[k - 1] - xlscp / 
		ex1[k] * qi_tot1__[k - 1]) * (one + p608 * sqv1[k]);
	thl_tot1__[k - 1] = th1[k] - xlvcp / ex1[k] * qc_tot1__[k - 1] - 
		xlscp / ex1[k] * qi_tot1__[k - 1];
	thv1[k - 1] = th1[k] * (one + p608 * sqv1[k] - (qc_tot1__[k - 1] + 
		qi_tot1__[k - 1]));
/* L9019: */
    }
/* L9020: */

    }
    if constexpr(M471_PHASE<0 || M471_PHASE==3) {
    mym_turbulence__(m471_work,m471_n, kts, kte, xland, bl_mynn_closure__, &ckmod, &taue, &dz1[
	    dz1_offset], dx, zw1, &pres1[pres1_offset], &ex1[ex1_offset], &u1[
	    u1_offset], &v1[v1_offset], thl1, thv1, thlv1, &sqc1[sqc1_offset],
	     sqw1, &qke1[qke1_offset], &tsq1[tsq1_offset], &qsq1[qsq1_offset],
	     &cov1[cov1_offset], vt1, vq1, &rmol, &rmolh, &flt, &fltv, &flq, 
	    pblh, &th1[th1_offset], &sh1[sh1_offset], &sm1[sm1_offset], &el1[
	    el1_offset], dfm1, dfh1, dfq1, tcd1, qcd1, pdk1, pdt1, pdq1, pdc1,
	     &qwt1[qwt1_offset], &qshear1[qshear1_offset], &qbuoy1[
	    qbuoy1_offset], &qdiss1[qdiss1_offset], tke_budget__, &psig_bl__, 
	    &psig_shcu__, &cldfra_bl1__[cldfra_bl1_offset], 
	    bl_mynn_mixlength__, bl_mynn_ess__, &edmf_w1__[edmf_w1_offset], &
	    edmf_a1__[edmf_a1_offset], edmf_w_dd1__, edmf_a_dd1__, 
	    tkeprod_dn__, tkeprod_up__, spp_pbl__, &pattern_spp_pbl1__[
	    pattern_spp_pbl1_offset]);

    }
    if constexpr(M471_PHASE<0 || M471_PHASE==4) {
    mym_predict__(m471_work,m471_n, kts, kte, bl_mynn_closure__, &delt2, &dz1[dz1_offset], ust, 
	    &flt, &flq, &pmz, &phh, &el1[el1_offset], dfq1, &rho1[rho1_offset]
	    , pdk1, pdt1, pdq1, pdc1, &qke1[qke1_offset], &tsq1[tsq1_offset], 
	    &qsq1[qsq1_offset], &cov1[cov1_offset], s_aw1__, s_awqke1__, 
	    s_awqsq1__, bl_mynn_edmf__, bl_mynn_edmf_tke__, &qwt1[qwt1_offset]
	    , &qdiss1[qdiss1_offset], tke_budget__, &upwind);
    i__1 = *kte - 1;
    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing MIN */
/* Computing MAX */
	d__1 =  qke1[k];
	d__2 =  rc0014;
/* Computing MAX */
	r__3 = p5 * (el1[k] + el1[k + 1]);
	r__2 = (real)pow_dd(&d__1, &d__2) / (b1 * max(r__3,one)) / cp;
	r__1 = max(r__2,zero);
	diss_heat1__[k - 1] = min(r__1,rc0019);
/* Computing MAX */
	r__1 = pres1[k];
	diss_heat1__[k - 1] *= (real)exp((real)(-rc0020 / max(r__1,one)));
/* L9022: */
    }
/* L9023: */
    diss_heat1__[*kte - 1] = zero;

    }
    if constexpr(M471_PHASE<0 || M471_PHASE==5) {
    mynn_tendencies__(m471_work,m471_n, kts, kte, i__, delt, &dz1[dz1_offset], zw1, xland, pblh,
	     &rho1[rho1_offset], &u1[u1_offset], &v1[v1_offset], &th1[
	    th1_offset], &tk1[tk1_offset], qv1, qc1, qi1, kzero1, &qnc1[
	    qnc1_offset], &qni1[qni1_offset], ps, &pres1[pres1_offset], &ex1[
	    ex1_offset], thl1, &sqv1[sqv1_offset], &sqc1[sqc1_offset], &sqi1[
	    sqi1_offset], kzero1, sqw1, thl_tot1__, qc_tot1__, qi_tot1__, &
	    qnwfa1[qnwfa1_offset], &qnifa1[qnifa1_offset], &qnbca1[
	    qnbca1_offset], &ozone1[ozone1_offset], ust, &flt, &flq, &flqv, &
	    flqc, wspd, uoce, voce, &tsq1[tsq1_offset], &qsq1[qsq1_offset], &
	    cov1[cov1_offset], tcd1, qcd1, dfm1, dfh1, dfq1, &du1[du1_offset],
	     &dv1[dv1_offset], &dth1[dth1_offset], &dqv1[dqv1_offset], &dqc1[
	    dqc1_offset], &dqi1[dqi1_offset], &dqs1[dqs1_offset], &dqnc1[
	    dqnc1_offset], &dqni1[dqni1_offset], &dqnwfa1[dqnwfa1_offset], &
	    dqnifa1[dqnifa1_offset], &dqnbca1[dqnbca1_offset], &dozone1[
	    dozone1_offset], diss_heat1__, s_aw1__, s_awthl1__, s_awqt1__, 
	    s_awqv1__, s_awqc1__, s_awu1__, s_awv1__, s_awqnc1__, s_awqni1__, 
	    s_awqnwfa1__, s_awqnifa1__, s_awqnbca1__, sd_aw1__, sd_awthl1__, 
	    sd_awqt1__, sd_awqv1__, sd_awqc1__, sd_awqi1__, sd_awqnc1__, 
	    sd_awqni1__, sd_awqnwfa1__, sd_awqnifa1__, sd_awu1__, sd_awv1__, &
	    sub_thl1__[sub_thl1_offset], &sub_sqv1__[sub_sqv1_offset], 
	    sub_u1__, sub_v1__, &det_thl1__[det_thl1_offset], &det_sqv1__[
	    det_sqv1_offset], det_sqc1__, det_u1__, det_v1__, flag_qc__, 
	    flag_qi__, flag_qnc__, flag_qni__, flag_qs__, flag_qnwfa__, 
	    flag_qnifa__, flag_qnbca__, flag_ozone__, &cldfra_bl1__[
	    cldfra_bl1_offset], bl_mynn_cloudmix__, bl_mynn_mixqt__, 
	    bl_mynn_edmf__, bl_mynn_edmf_mom__, bl_mynn_mixscalars__, 
	    bl_mynn_mixaerosols__, bl_mynn_mixnumcon__, &upwind);

    }
    if constexpr(M471_PHASE<0 || M471_PHASE==6) {
    if (*mix_chem__) {
	mynn_mix_chem__(m471_work,m471_n, kts, kte, i__, delt, &dz1[dz1_offset], pblh, nchem, 
		ndvel, &chem1[chem1_offset], &vdep[1], &settle1[
		settle1_offset], &rho1[rho1_offset], &flt, tcd1, qcd1, dfh1, 
		s_aw1__, s_awchem1__, emis_ant_no__, frp, enh_mix__, 
		bl_mynn_edmf__, &upwind);
	i__1 = *nchem;
	
#pragma unroll 1
for (ic = 1; ic <= i__1; ++ic) {
	    i__2 = *kte;
	    
#pragma unroll 4
for (k = *kts; k <= i__2; ++k) {
/* Computing MAX */
		r__1 = rc0021, r__2 = chem1[k + ic * chem1_dim1];
		chem1[k + ic * chem1_dim1] = max(r__1,r__2);
/* L9028: */
	    }
/* L9029: */
/* L9025: */
	}
/* L9026: */
    }
    if (*bl_mynn_mixscalars__ == 1) {
	mynn_mix_scalars__(m471_work,m471_n, kts, kte, i__, delt, &dz1[dz1_offset], nscalars, &
		scalars[scalars_offset], &rho1[rho1_offset], &flt, tcd1, qcd1,
		 dfh1, s_aw1__, s_awscalars1__, bl_mynn_edmf__, &upwind);
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
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    dummy1[k - 1] = p5 * (qshear1[k] + qshear1[k + 1]);
	    dummy2[k - 1] = p5 * (qbuoy1[k] + qbuoy1[k + 1]);
	    dqke1[k] = p5 * (qke1[k] - dqke1[k]) / *delt;
/* L9031: */
	}
/* L9032: */
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    qshear1[*kts + ixarr1 - 1] = dummy1[ixarr1 - 1];
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
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
	if (*bl_mynn_output__ == 2 && *bl_mynn_edmf_dd__ == 1) {
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		edmf_a1__[*kts + ixarr1 - 1] = edmf_a_dd1__[*kts + ixarr1 - 2]
			;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		edmf_w1__[*kts + ixarr1 - 1] = edmf_w_dd1__[*kts + ixarr1 - 2]
			;
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		edmf_qt1__[*kts + ixarr1 - 1] = edmf_qt_dd1__[*kts + ixarr1 - 
			2];
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		edmf_thl1__[*kts + ixarr1 - 1] = edmf_thl_dd1__[*kts + ixarr1 
			- 2];
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		edmf_ent1__[*kts + ixarr1 - 1] = edmf_ent_dd1__[*kts + ixarr1 
			- 2];
	    }
	    i__1 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		edmf_qc1__[*kts + ixarr1 - 1] = edmf_qc_dd1__[*kts + ixarr1 - 
			2];
	    }
	}
    }
    if (FALSE_) {
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    if (cldfra_bl1__[k] < zero || cldfra_bl1__[k] > one) {
	    }
/* L9034: */
	}
/* L9035: */
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	qke_adv1__[*kts + ixarr1 - 1] = qke1[*kts + ixarr1 - 1];
    }

    }
    return 0;
} /* mynnedmf_ */

__device__ M471_CALL /* Subroutine */ int mym_initialize__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> ckmod,
	 M471Ptr<real> taue, M471Ptr<real> xland, M471Ptr<real> pblh, M471Ptr<real> dz, M471Ptr<real> dx, M471Ptr<real> zw, 
	M471Ptr<real> p, M471Ptr<real> exner, M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> thl, M471Ptr<real> qw, M471Ptr<real> theta, M471Ptr<real> thv, M471Ptr<real> thlv, M471Ptr<real> sh, M471Ptr<real> sm, M471Ptr<real> ust, M471Ptr<real> rmol, M471Ptr<real> rmolh, M471Ptr<real> el, M471Ptr<real> qke, M471Ptr<real> tsq, M471Ptr<real> qsq, M471Ptr<real> cov, M471Ptr<real> psig_bl__, M471Ptr<real> cldfra_bl1__, M471Ptr<integer> bl_mynn_mixlength__, M471Ptr<integer> bl_mynn_ess__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w_dd1__, M471Ptr<real> edmf_a_dd1__, M471Ptr<logical> initialize_qke__, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__) {
    /* Initialized data */

    real karman = .4000000059604644775390625f;
    real gtr = .032700002193450927734375f;
    real qkemin = 9.999999747378751635551452636719e-6f;
    real b1 = 24.f;
    real b2 = 15.f;
    real zero = 0.f;
    real one = 1.f;
    real two = 2.f;
    real three = 3.f;
    real hundred = 100.f;
    real p01 = .00999999977648258209228515625f;
    real p333 = .3333333432674407958984375f;
    real p5 = .5f;
    real p666 = .666666686534881591796875f;
    real rc0000 = 1.f;
    real rc0001 = 0.f;
    real rc0002 = 1.5f;
    real rc0003 = 700.f;
    real rc0004 = 5e-6f;
    real rc0005 = .25f;
    real rc0006 = 125.f;

    /* System generated locals */
    integer dz_offset, zw_offset, u_offset, v_offset, thl_offset, thlv_offset,
	     qw_offset, cldfra_bl1_offset, edmf_w1_offset, edmf_a1_offset, 
	    edmf_w_dd1_offset, edmf_a_dd1_offset, p_offset, exner_offset, 
	    tsq_offset, qsq_offset, cov_offset, el_offset, qke_offset, 
	    sm_offset, sh_offset, theta_offset, thv_offset, 
	    pattern_spp_pbl1_offset, i__1, i__2;
    real r__1, r__2, r__3, r__4, r__5, r__6;
    doublereal d__1, d__2;

    /* Builtin functions */
    

    /* Local variables */
    integer k, l;
    M471Ptr<real> gh=m471_array<real>(m471_work,(304*(MYNN471_NZ-49)+15016),m471_n);
M471Ptr<real> gm=m471_array<real>(m471_work,(308*(MYNN471_NZ-49)+15212),m471_n);
M471Ptr<real> ql=m471_array<real>(m471_work,(312*(MYNN471_NZ-49)+15408),m471_n);
    
    real  b1l, b2l;
M471Ptr<real> vq=m471_array<real>(m471_work,(316*(MYNN471_NZ-49)+15604),m471_n);
M471Ptr<real> vt=m471_array<real>(m471_work,(320*(MYNN471_NZ-49)+15800),m471_n);
    
    real  phh, elq, phm, vkz, elv, pmz, flt, flq;
M471Ptr<real> pdc=m471_array<real>(m471_work,(324*(MYNN471_NZ-49)+15996),m471_n);
M471Ptr<real> pdk=m471_array<real>(m471_work,(328*(MYNN471_NZ-49)+16192),m471_n);
M471Ptr<real> pdq=m471_array<real>(m471_work,(332*(MYNN471_NZ-49)+16388),m471_n);
M471Ptr<real> pdt=m471_array<real>(m471_work,(336*(MYNN471_NZ-49)+16584),m471_n);
M471Ptr<real> dtl=m471_array<real>(m471_work,(340*(MYNN471_NZ-49)+16780),m471_n);
M471Ptr<real> dqw=m471_array<real>(m471_work,(344*(MYNN471_NZ-49)+16976),m471_n);
M471Ptr<real> dtv=m471_array<real>(m471_work,(348*(MYNN471_NZ-49)+17172),m471_n);
M471Ptr<real> qkw=m471_array<real>(m471_work,(352*(MYNN471_NZ-49)+17368),m471_n);
M471Ptr<real> qtw=m471_array<real>(m471_work,(356*(MYNN471_NZ-49)+17564),m471_n);
M471Ptr<real> qpe=m471_array<real>(m471_work,(360*(MYNN471_NZ-49)+17760),m471_n);
    integer lmax;
    real fltv, tmpq, ustm;

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
    exner_offset = *kts;
    exner -= exner_offset;
    p_offset = *kts;
    p -= p_offset;
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
    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	ql[k - 1] = zero;
	vt[k - 1] = zero;
	vq[k - 1] = zero;
	qkw[k - 1] = zero;
	qtw[k - 1] = zero;
/* L9004: */
    }
/* L9005: */
    ustm = max(p01,*ust);
    mym_level2__(m471_work,m471_n, kts, kte, ckmod, bl_mynn_ess__, &zw[zw_offset], &dz[
	    dz_offset], xland, pblh, &u[u_offset], &v[v_offset], &thl[
	    thl_offset], &thv[thv_offset], &thlv[thlv_offset], &theta[
	    theta_offset], &p[p_offset], &exner[exner_offset], &qke[
	    qke_offset], &cldfra_bl1__[cldfra_bl1_offset], &edmf_a1__[
	    edmf_a1_offset], &edmf_w1__[edmf_w1_offset], &qw[qw_offset], ql, 
	    vt, vq, dtl, dqw, dtv, gm, gh, &sm[sm_offset], &sh[sh_offset]);
    el[*kts] = zero;
    if (*initialize_qke__) {
/* Computing 2nd power */
	r__1 = ustm;
	d__1 =  (b1 * pmz);
	d__2 =  p666;
	qke[*kts] = rc0002 * (r__1 * r__1) * (real)pow_dd(&d__1, &d__2);
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = (*ust * rc0003 - zw[k]) / (ustm * rc0003);
	    qke[k] = qke[*kts] * max(r__1,p01);
/* L9007: */
	}
/* L9008: */
    }
    d__1 =  (b1 * pmz);
    d__2 =  p333;
    phm = phh * b2 / (real)pow_dd(&d__1, &d__2);
/* Computing MIN */
/* Computing 2nd power */
    r__3 = max(zero,flt) / ustm;
    r__1 = three, r__2 = phm * (r__3 * r__3);
    tsq[*kts] = min(r__1,r__2);
/* Computing MIN */
/* Computing 2nd power */
    r__3 = max(zero,flq) / ustm;
    r__1 = rc0004, r__2 = phm * (r__3 * r__3);
    qsq[*kts] = min(r__1,r__2);
    cov[*kts] = phm * (flt / ustm) * (flq / ustm);
/* Computing MIN */
/* Computing MAX */
/* Computing 2nd power */
    r__5 = gtr;
/* Computing 2nd power */
    r__6 = *taue;
    r__3 = zero, r__4 = two * tsq[*kts] * (r__5 * r__5) * (r__6 * r__6);
    r__1 = three, r__2 = max(r__3,r__4);
    qpe[*kts - 1] = min(r__1,r__2);
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	vkz = karman * zw[k];
	el[k] = vkz / (one + vkz / hundred);
	tsq[k] = zero;
	qsq[k] = zero;
	cov[k] = zero;
	qpe[k - 1] = zero;
/* L9010: */
    }
/* L9011: */
    lmax = 5;
    i__1 = lmax;
    
#pragma unroll 1
for (l = 1; l <= i__1; ++l) {
	mym_length__(m471_work,m471_n, kts, kte, xland, &dz[dz_offset], dx, &zw[zw_offset], 
		rmol, rmolh, &flt, &fltv, &flq, vt, vq, &u[u_offset], &v[
		v_offset], &qke[qke_offset], qpe, dtv, &el[el_offset], pblh, &
		theta[theta_offset], qkw, qtw, psig_bl__, &cldfra_bl1__[
		cldfra_bl1_offset], bl_mynn_mixlength__, &edmf_w1__[
		edmf_w1_offset], &edmf_a1__[edmf_a1_offset], &edmf_w_dd1__[
		edmf_w_dd1_offset], &edmf_a_dd1__[edmf_a_dd1_offset]);
	i__2 = *kte;
	
#pragma unroll 4
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
	vkz = karman * p5 * dz[*kts];
	elv = p5 * (el[*kts + 1] + el[*kts]) / vkz;
	if (*initialize_qke__) {
/* Computing 2nd power */
	    r__1 = ustm;
	    d__1 =  (b1 * pmz * elv);
	    d__2 =  p666;
	    qke[*kts] = one * (r__1 * r__1) * (real)pow_dd(&d__1, &d__2);
	}
/* Computing 2nd power */
	r__1 = elv;
	d__1 =  (b1 * pmz / (r__1 * r__1));
	d__2 =  p333;
	phm = phh * b2 / (real)pow_dd(&d__1, &d__2);
/* Computing MIN */
/* Computing 2nd power */
	r__3 = max(zero,flt) / ustm;
	r__1 = three, r__2 = phm * (r__3 * r__3);
	tsq[*kts] = min(r__1,r__2);
/* Computing MIN */
/* Computing 2nd power */
	r__3 = max(zero,flq) / ustm;
	r__1 = rc0004, r__2 = phm * (r__3 * r__3);
	qsq[*kts] = min(r__1,r__2);
	cov[*kts] = phm * (flt / ustm) * (flq / ustm);
	i__2 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__2; ++k) {
	    b1l = b1 * rc0005 * (el[k + 1] + el[k]);
/* Computing MIN */
/* Computing MAX */
	    r__2 = b1l * (pdk[k] + pdk[k - 1]);
	    r__1 = max(r__2,qkemin);
	    if (*initialize_qke__) /* m480 unread unless */ { tmpq = min(r__1,rc0006); }
	    if (*initialize_qke__) {
		d__1 =  tmpq;
		d__2 =  p666;
		qke[k] = (real)pow_dd(&d__1, &d__2);
	    }
	    if (qke[k] <= zero) {
		b2l = zero;
	    } else {
		b2l = b2 * (b1l / b1) / (real)sqrt(qke[k]);
	    }
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = b2l * (pdt[k] + pdt[k - 1]);
	    r__1 = three, r__2 = max(r__3,r__4);
	    tsq[k] = min(r__1,r__2);
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = b2l * (pdq[k] + pdq[k - 1]);
	    r__1 = rc0004, r__2 = max(r__3,r__4);
	    qsq[k] = min(r__1,r__2);
	    cov[k] = b2l * (pdc[k] + pdc[k - 1]);
/* L9019: */
	}
/* L9020: */
/* L9013: */
    }
/* L9014: */
    if (*initialize_qke__) {
	qke[*kts] = p5 * (qke[*kts] + qke[*kts + 1]);
	qke[*kte] = qke[*kte - 1];
    }
    tsq[*kte] = tsq[*kte - 1];
    qsq[*kte] = qsq[*kte - 1];
    cov[*kte] = cov[*kte - 1];
    return 0;
} /* mym_initialize__ */

__device__ M471_CALL /* Subroutine */ int mym_level2__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> ckmod, 
	M471Ptr<integer> bl_mynn_ess__, M471Ptr<real> zw, M471Ptr<real> dz, M471Ptr<real> xland, M471Ptr<real> pblh, 
	M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> thl, M471Ptr<real> thv, M471Ptr<real> thlv, M471Ptr<real> th, M471Ptr<real> p,
	 M471Ptr<real> exner, M471Ptr<real> qke, M471Ptr<real> cldfra, M471Ptr<real> edmf_a__, M471Ptr<real> edmf_w__,
	 M471Ptr<real> qw, M471Ptr<real> ql, M471Ptr<real> vt, M471Ptr<real> vq, M471Ptr<real> dtl, M471Ptr<real> dqw, M471Ptr<real> dtv, M471Ptr<real> gm, M471Ptr<real> gh, M471Ptr<real> sm, M471Ptr<real> sh) {
    /* Initialized data */

    real cp = 1004.5f;
    real gtr = .032700002193450927734375f;
    real tv0 = 182.50872802734375f;
    real qkemin = 9.999999747378751635551452636719e-6f;
    real g1 = .23499999940395355224609375f;
    real b1 = 24.f;
    real c2 = .72899997234344482421875f;
    real c5 = .20000000298023223876953125f;
    real a1 = 1.1800000667572021484375f;
    real c1 = .137067615985870361328125f;
    real a2 = .664521038532257080078125f;
    real g2 = .56412994861602783203125f;
    real zero = 0.f;
    real one = 1.f;
    real two = 2.f;
    real three = 3.f;
    real four = 4.f;
    real ten = 10.f;
    real twenty = 20.f;
    real hundred = 100.f;
    real p1 = .100000001490116119384765625f;
    real p2 = .20000000298023223876953125f;
    real p25 = .25f;
    real p3 = .300000011920928955078125f;
    real p5 = .5f;
    real p95 = .949999988079071044921875f;
    real rc0000 = 0.f;
    real rc0001 = 1e-10f;
    real rc0002 = 15.f;
    real rc0003 = 1.5f;
    real rc0004 = 1.8f;
    real rc0005 = .08f;
    real rc0006 = 400.f;
    real rc0007 = 200.f;
    real rc0008 = .01f;
    real rc0009 = 2.7f;

    /* System generated locals */
    integer dz_offset, u_offset, v_offset, thl_offset, qw_offset, ql_offset, 
	    vt_offset, vq_offset, thv_offset, thlv_offset, th_offset, 
	    p_offset, exner_offset, edmf_a_offset, edmf_w_offset, qke_offset, 
	    cldfra_offset, zw_offset, dtl_offset, dqw_offset, dtv_offset, 
	    gm_offset, gh_offset, sm_offset, sh_offset, i__1;
    real r__1, r__2, r__3, r__4, r__5, r__6;

    /* Builtin functions */
    

    /* Local variables */
    
    integer k;
    real f1, f2, rf, xl, wt;
M471Ptr<real> ri=m471_array<real>(m471_work,(364*(MYNN471_NZ-49)+17956),m471_n);
    
    real rf1, rf2, ri1, ri2, ri3, ri4, abk, afk, rfc, mfi, shc, smc, dqz, dtq,
	     dzk, duz, dtz, vqq, vtt, eth0, eth1, thv0, thv1, cfac, clam, 
	    tabs, qkei, qsat, a2fac, thvp, clam0, ugrid, taper, lambda, 
	    uonset, cldfrai;

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
    edmf_w_offset = *kts;
    edmf_w__ -= edmf_w_offset;
    edmf_a_offset = *kts;
    edmf_a__ -= edmf_a_offset;
    cldfra_offset = *kts;
    cldfra -= cldfra_offset;
    qke_offset = *kts;
    qke -= qke_offset;
    exner_offset = *kts;
    exner -= exner_offset;
    p_offset = *kts;
    p -= p_offset;
    th_offset = *kts;
    th -= th_offset;
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
    zw_offset = *kts;
    zw -= zw_offset;

    /* Function Body */
    thvp = rc0000;
    dtl[*kts] = zero;
    dqw[*kts] = zero;
    dtv[*kts] = zero;
    gm[*kts] = zero;
    gh[*kts] = zero;
    sm[*kts] = zero;
    sh[*kts] = zero;
    ri[*kts - 1] = zero;
    if (*bl_mynn_ess__ == 0) {
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    dzk = p5 * (dz[k] + dz[k - 1]);
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
	    dtq = vtt * dtz + vqq * dqz;
	    dtl[k] = dtz;
	    dqw[k] = dqz;
	    dtv[k] = dtq;
	    gm[k] = duz;
	    gh[k] = -dtq * gtr;
	    ri[k - 1] = -gh[k] / max(duz,rc0001);
/* L9004: */
	}
/* L9005: */
    } else if (*bl_mynn_ess__ == 1) {
/* Computing 2nd power */
	r__1 = u[*kts];
/* Computing 2nd power */
	r__2 = v[*kts];
	ugrid = (real)sqrt((real)(r__1 * r__1 + r__2 * r__2));
	uonset = rc0002;
/* Computing MIN */
/* Computing MAX */
	r__3 = zero, r__4 = ugrid - uonset;
	r__1 = one, r__2 = max(r__3,r__4) / twenty;
	taper = one - p95 * min(r__1,r__2);
	if (*xland - rc0003 >= zero) {
	    clam0 = rc0004 * taper;
	} else {
	    clam0 = rc0005;
	}
	i__1 = *kte;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    dzk = p5 * (dz[k] + dz[k - 1]);
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
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = zw[k] - (*pblh + hundred);
/* Computing MAX */
	    r__5 = rc0006, r__6 = p3 * *pblh;
	    r__1 = one, r__2 = max(r__3,r__4) / max(r__5,r__6);
	    wt = min(r__1,r__2);
	    clam = clam0 * (one - wt) + p2 * wt;
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = zw[k] - twenty;
	    r__1 = one, r__2 = max(r__3,r__4) / rc0007;
	    clam *= min(r__1,r__2);
	    thv1 = thvp * thv[k] + (one - thvp) * thlv[k];
	    thv0 = thvp * thv[k - 1] + (one - thvp) * thlv[k - 1];
	    tabs = th[k] * exner[k];
	    xl = (real)xl_blend__(m471_work,m471_n, &tabs);
	    qsat = (real)qsat_blend__(m471_work,m471_n, &tabs, &p[k]);
	    eth1 = th[k] * (real)exp((real)(xl * qsat / (cp * tabs)));
	    tabs = th[k - 1] * exner[k - 1];
	    xl = (real)xl_blend__(m471_work,m471_n, &tabs);
	    qsat = (real)qsat_blend__(m471_work,m471_n, &tabs, &p[k - 1]);
	    eth0 = th[k - 1] * (real)exp((real)(xl * qsat / (cp * tabs)));
/* Computing MAX */
/* Computing MIN */
	    r__3 = p2, r__4 = p5 * (edmf_a__[k] + edmf_a__[k - 1]) * (p5 * (
		    edmf_w__[k] + edmf_w__[k - 1]));
	    r__1 = zero, r__2 = min(r__3,r__4);
	    mfi = max(r__1,r__2);
/* Computing MAX */
	    r__1 = qkemin, r__2 = p5 * (qke[k] + qke[k - 1]);
	    qkei = max(r__1,r__2);
/* Computing MIN */
	    r__1 = p5, r__2 = (real)sqrt((real)(p25 * qkei));
	    qkei = min(r__1,r__2) * p2;
	    cldfrai = p5 * (cldfra[k] + cldfra[k - 1]);
/* Computing MAX */
/* Computing MIN */
/* Computing 2nd power */
	    r__4 = cldfrai - one;
	    r__3 = rc0009 * (r__4 * r__4);
	    r__1 = rc0008, r__2 = min(r__3,one);
	    cfac = max(r__1,r__2);
/* Computing MIN */
	    r__1 = p1, r__2 = cldfrai * cfac;
	    cldfrai = min(r__1,r__2) * ten;
	    lambda = clam * max(mfi,qkei) * cldfrai;
/* Computing MIN */
	    r__1 = zero, r__2 = (eth1 - eth0) / dzk;
	    dtq = (thv1 - thv0) / dzk + lambda * min(r__1,r__2);
	    dtl[k] = dtz;
	    dqw[k] = dqz;
	    dtv[k] = dtq;
	    gm[k] = duz;
	    gh[k] = -dtq * gtr;
	    ri[k - 1] = -gh[k] / max(duz,rc0001);
/* L9007: */
	}
/* L9008: */
    }
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
/* Computing MAX */
	r__1 = ri[k - 1];
	a2fac = one / (one + *ckmod * max(r__1,zero));
	rfc = g1 / (g1 + g2);
	f1 = b1 * (g1 - c1) + three * a2 * a2fac * (one - c2) * (one - c5) + 
		two * a1 * (three - two * c2);
	f2 = b1 * (g1 + g2) - three * a1 * (one - c2);
	rf1 = b1 * (g1 - c1) / f1;
	rf2 = b1 * g1 / f2;
	smc = a1 / (a2 * a2fac) * f1 / f2;
	shc = three * (a2 * a2fac) * (g1 + g2);
	ri1 = p5 / smc;
	ri2 = rf1 * smc;
	ri3 = four * rf2 * smc - two * ri2;
/* Computing 2nd power */
	r__1 = ri2;
	ri4 = r__1 * r__1;
/* Computing MIN */
/* Computing 2nd power */
	r__2 = ri[k - 1];
	r__1 = ri1 * (ri[k - 1] + ri2 - (real)sqrt((real)(r__2 * r__2 - ri3 * 
		ri[k - 1] + ri4)));
	rf = min(r__1,rfc);
	sh[k] = shc * (rfc - rf) / (one - rf);
	sm[k] = smc * (rf1 - rf) / (rf2 - rf) * sh[k];
/* L9010: */
    }
/* L9011: */
    return 0;
} /* mym_level2__ */

__device__ M471_CALL /* Subroutine */ int mym_length__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> xland, 
	M471Ptr<real> dz, M471Ptr<real> dx, M471Ptr<real> zw, M471Ptr<real> rmol, M471Ptr<real> rmolh, M471Ptr<real> flt, 
	M471Ptr<real> fltv, M471Ptr<real> flq, M471Ptr<real> vt, M471Ptr<real> vq, M471Ptr<real> u1, M471Ptr<real> v1, M471Ptr<real> qke, M471Ptr<real> qpe, M471Ptr<real> dtv, M471Ptr<real> el, M471Ptr<real> pblh, M471Ptr<real> theta, M471Ptr<real> qkw, M471Ptr<real> qtw, M471Ptr<real> psig_bl__, M471Ptr<real> cldfra_bl1__, M471Ptr<integer> bl_mynn_mixlength__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w_dd1__, M471Ptr<real> edmf_a_dd1__) {
    /* Initialized data */

    real karman = .4000000059604644775390625f;
    real grav = 9.81000041961669921875f;
    real gtr = .032700002193450927734375f;
    real tv0 = 182.50872802734375f;
    real qmin = 0.f;
    real zmax = 1.f;
    real qkemin = 9.999999747378751635551452636719e-6f;
    real zero = 0.f;
    real one = 1.f;
    real four = 4.f;
    real five = 5.f;
    real ten = 10.f;
    real twenty = 20.f;
    real thirty = 30.f;
    real forty = 40.f;
    real fifty = 50.f;
    real hundred = 100.f;
    real p1 = .100000001490116119384765625f;
    real p2 = .20000000298023223876953125f;
    real p25 = .25f;
    real p3 = .300000011920928955078125f;
    real p333 = .3333333432674407958984375f;
    real p4 = .4000000059604644775390625f;
    real p5 = .5f;
    real rc0000 = 250.f;
    real rc0001 = 750.f;
    real rc0002 = 100.f;
    real rc0003 = 2.f;
    real rc0004 = .05f;
    real rc0005 = 1e3f;
    real rc0006 = 2.7f;
    real rc0007 = .23f;
    real rc0008 = 1e-5f;
    real rc0009 = .03f;
    real rc0010 = 1e10f;
    real rc0011 = 0.f;
    real rc0012 = 15.f;
    real rc0013 = 4e3f;
    real rc0014 = 1.2e4f;
    real rc0015 = 3.5f;
    real rc0016 = .24f;
    real rc0017 = .4f;
    real rc0018 = 1.5f;
    real rc0019 = 15.f;
    real rc0020 = 200.f;
    real rc0021 = 300.f;
    real rc0022 = 600.f;
    real rc0023 = .005f;
    real rc0024 = .01f;
    real rc0025 = 30.f;
    real rc0026 = 350.f;
    real rc0027 = 400.f;
    real rc0028 = 8.f;
    real rc0029 = .001f;
    real rc0030 = .22f;
    real rc0031 = 2.5f;
    real rc0032 = 1.25f;
    real rc0033 = 1e-4f;
    real rc0034 = 30.f;
    real rc0035 = 150.f;
    real rc0036 = 1.f;
    real rc0037 = 40.f;
    real rc0038 = 800.f;

    /* System generated locals */
    integer dz_offset, zw_offset, u1_offset, v1_offset, qke_offset, 
	    qpe_offset, vt_offset, vq_offset, cldfra_bl1_offset, 
	    edmf_w1_offset, edmf_a1_offset, edmf_w_dd1_offset, 
	    edmf_a_dd1_offset, el_offset, qkw_offset, qtw_offset, dtv_offset, 
	    theta_offset, i__1;
    real r__1, r__2, r__3, r__4, r__5;
    doublereal d__1, d__2;

    /* Builtin functions */
    

    /* Local variables */
    integer k;
    real h1, h2, tau_cloud__, el_unstab__, bv, wt, abk, afk, elb, elf, elt, 
	    vsc, cns, csl, dzk, qdz, els, zwk, z_m__, qkw_elb_min__, alp1, 
	    alp2, alp3, alp4, alp5, alp6, els1, zwk1, ctau, zslh, 
	    vflx, pblh2, alp3z, wt_u1__, wt_u2__, cpblh, ugrid, maxdz, mindz, 
	    wt_dx__, wstar;
M471Ptr<real> qtke=m471_array<real>(m471_work,(364*(MYNN471_NZ-49)+17956),m471_n);
    
    real pblh_plus_ent__;
    integer ixarr1;
    real elb_mf__, cldavg, el_les__, qkw_mf__, uonset, minpblh, elt_max__;
M471Ptr<real> thetaw=m471_array<real>(m471_work,(368*(MYNN471_NZ-49)+18152),m471_n);
M471Ptr<real> elblavg=m471_array<real>(m471_work,(372*(MYNN471_NZ-49)+18348),m471_n);
M471Ptr<real> elblmin=m471_array<real>(m471_work,(376*(MYNN471_NZ-49)+18544),m471_n);

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
    qtw_offset = *kts;
    qtw -= qtw_offset;
    qkw_offset = *kts;
    qkw -= qkw_offset;
    theta_offset = *kts;
    theta -= theta_offset;
    el_offset = *kts;
    el -= el_offset;
    dtv_offset = *kts;
    dtv -= dtv_offset;
    qpe_offset = *kts;
    qpe -= qpe_offset;
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
    /* unread private value */;
    maxdz = rc0001;
    mindz = rc0000;
    /* unread private value */;
    /* unread private value */;
    qkw_elb_min__ = rc0004;
    ctau = rc0005;
    if (*bl_mynn_mixlength__ == 0) {
	cns = rc0006;
	alp1 = rc0007;
	alp2 = one;
	alp3 = five;
	alp4 = hundred;
	alp5 = p3;
	pblh2 = *pblh + dz[*kts];
/* Computing MAX */
	r__1 = p3 * pblh2;
	h1 = max(r__1,mindz);
	h1 = min(h1,maxdz);
	h2 = h1 * p5;
/* Computing MAX */
	r__1 = qke[*kts];
	qkw[*kts] = (real)sqrt((real)(max(r__1,qkemin)));
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    afk = dz[k] / (dz[k] + dz[k - 1]);
	    abk = one - afk;
/* Computing MAX */
	    r__1 = qke[k] * abk + qke[k - 1] * afk;
	    qkw[k] = (real)sqrt((real)(max(r__1,qkemin)));
/* L9004: */
	}
/* L9005: */
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    qtw[*kts + ixarr1 - 1] = qkw[*kts + ixarr1 - 1];
	}
	elt = rc0008;
	vsc = rc0008;
	k = *kts + 1;
	zwk = zw[k];
L9006:
	if (! (zwk <= pblh2 + h1)) {
	    goto L9007;
	}
	dzk = p5 * (dz[k] + dz[k - 1]);
/* Computing MAX */
	r__1 = qkw[k] - qmin;
	qdz = max(r__1,rc0009) * dzk;
	elt += qdz * zwk;
	vsc += qdz;
	++k;
	zwk = zw[k];
	goto L9006;
L9007:
	elt = alp1 * elt / vsc;
	vflx = (vt[*kts] + one) * *flt + (vq[*kts] + tv0) * *flq;
	d__1 =  (gtr * elt * max(vflx,zero));
	d__2 =  p333;
	vsc = (real)pow_dd(&d__1, &d__2);
	el[*kts] = zero;
	/* unread private value */;
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    zwk = zw[k];
	    if (dtv[k] > zero) {
		bv = (real)sqrt((real)(gtr * dtv[k]));
		elb = alp2 * qkw[k] / bv * (one + alp3 / alp2 * (real)sqrt((
			real)(vsc / (bv * elt))));
		elf = alp2 * qkw[k] / bv;
	    } else {
		elb = rc0010;
		elf = elb;
	    }
	    if (*rmol > rc0011) {
/* Computing MIN */
		r__1 = zwk * *rmol;
		els = karman * zwk / (one + cns * min(r__1,zmax));
	    } else {
		d__1 =  (one - alp4 * zwk * *rmol);
		d__2 =  p2;
		els = karman * zwk * (real)pow_dd(&d__1, &d__2);
	    }
	    wt = p5 * (real)tanh((real)((zwk - (pblh2 + h1)) / h2)) + p5;
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
	uonset = rc0012;
/* Computing MIN */
/* Computing MAX */
	r__3 = zero, r__4 = ugrid - uonset;
	r__1 = one, r__2 = max(r__3,r__4) / forty;
	wt_u1__ = one - p2 * min(r__1,r__2);
/* Computing MIN */
/* Computing MAX */
	r__3 = zero, r__4 = ugrid - uonset;
	r__1 = one, r__2 = max(r__3,r__4) / forty;
	wt_u2__ = one - p5 * min(r__1,r__2);
/* Computing MIN */
/* Computing MAX */
	r__3 = zero, r__4 = *dx - rc0013;
	r__1 = one, r__2 = max(r__3,r__4) / rc0014;
	wt_dx__ = one - min(r__1,r__2);
	cns = rc0015;
	alp1 = rc0007 * wt_dx__ + (one - wt_dx__) * rc0016;
	alp2 = p3 * wt_dx__ + (one - wt_dx__) * rc0017;
	alp3 = five * wt_u2__;
	if (*xland - rc0018 >= zero) {
	    alp4 = twenty * wt_u2__;
	} else {
	    alp4 = rc0019;
	}
	alp5 = p3;
	alp6 = fifty;
	pblh2 = max(*pblh,rc0020);
/* Computing MAX */
	r__1 = p3 * pblh2;
	h1 = max(r__1,rc0021);
	h1 = min(h1,rc0022);
	h2 = h1 * p5;
/* Computing MAX */
	r__1 = p5 * qke[*kts], r__2 = p5 * qkemin;
	qtke[*kts - 1] = max(r__1,r__2);
	thetaw[*kts - 1] = theta[*kts];
/* Computing MAX */
	r__1 = qke[*kts];
	qkw[*kts] = (real)sqrt((real)(max(r__1,qkemin)));
/* Computing MAX */
	r__1 = qke[*kts] + qpe[*kts];
	qtw[*kts] = (real)sqrt((real)(max(r__1,qkemin)));
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    afk = dz[k] / (dz[k] + dz[k - 1]);
	    abk = one - afk;
/* Computing MAX */
	    r__1 = qke[k] * abk + qke[k - 1] * afk;
	    qkw[k] = (real)sqrt((real)(max(r__1,qkemin)));
/* Computing MAX */
	    r__1 = (qke[k] + qpe[k]) * abk + (qke[k - 1] + qpe[k - 1]) * afk;
	    qtw[k] = (real)sqrt((real)(max(r__1,qkemin)));
/* Computing MAX */
/* Computing 2nd power */
	    r__2 = qkw[k];
	    r__1 = p5 * (r__2 * r__2);
	    qtke[k - 1] = max(r__1,rc0023);
	    thetaw[k - 1] = theta[k] * abk + theta[k - 1] * afk;
/* L9013: */
	}
/* L9014: */
	elt = rc0008;
	vsc = rc0008;
	k = *kts + 1;
	zwk = zw[k];
L9015:
/* Computing MAX */
/* Computing MIN */
	r__3 = p3 * pblh2;
	r__1 = rc0020, r__2 = min(r__3,rc0022);
	if (! (zwk <= pblh2 + max(r__1,r__2))) {
	    goto L9016;
	}
	dzk = p5 * (dz[k] + dz[k - 1]);
/* Computing MIN */
/* Computing MAX */
	r__2 = qkw[k];
	r__1 = max(r__2,rc0024);
	qdz = min(r__1,rc0025) * dzk;
	elt += qdz * zwk;
	vsc += qdz;
	++k;
	zwk = zw[k];
	goto L9015;
L9016:
	if (*xland - rc0018 >= zero) {
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = ugrid - twenty;
	    r__1 = one, r__2 = max(r__3,r__4) / thirty;
	    elt_max__ = rc0026 + rc0002 * min(r__1,r__2);
	} else {
	    elt_max__ = rc0027;
	}
/* Computing MIN */
/* Computing MAX */
	r__2 = alp1 * elt / vsc;
	r__1 = max(r__2,rc0028);
	elt = min(r__1,elt_max__);
	vflx = *fltv - rc0023;
	d__1 =  (gtr * elt * max(vflx,zero));
	d__2 =  p333;
	vsc = (real)pow_dd(&d__1, &d__2);
	el[*kts] = zero;
	/* unread private value */;
	boulac_length__(m471_work,m471_n, kts, kte, &zw[zw_offset], &dz[dz_offset], qtke, 
		thetaw, elblmin, elblavg);
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    zwk = zw[k];
/* Computing MAX */
	    r__2 = p5 * (edmf_a1__[k] + edmf_a1__[k - 1]) * (p5 * (edmf_w1__[
		    k] + edmf_w1__[k - 1])), r__3 = (r__1 = edmf_a_dd1__[k - 
		    1] * edmf_w_dd1__[k - 1], abs(r__1));
	    qkw_mf__ = max(r__2,r__3);
/* Computing MIN */
	    r__1 = (zwk + p25 * pblh2) / pblh2;
	    cpblh = min(r__1,one);
	    qkw_mf__ = cpblh * qkw_mf__;
	    alp3z = cpblh * alp3;
	    if (dtv[k] > rc0011) {
/* Computing MAX */
		r__1 = (real)sqrt((real)(gtr * dtv[k]));
		bv = max(r__1,rc0029);
		elb_mf__ = alp6 * qkw_mf__ / bv;
		elb_mf__ /= one + elb_mf__ / rc0002;
/* Computing MAX */
		r__1 = qkw[k];
		elb = alp2 * max(r__1,qkw_elb_min__) / bv * (one + alp3z * (
			real)sqrt((real)(vsc / (bv * elt))));
		elb = max(elb,elb_mf__);
		elb = min(elb,zwk);
/* Computing MAX */
		r__1 = qkw[k];
		elf = one * max(r__1,qkw_elb_min__) / bv;
		elf = max(elf,elb_mf__);
/* Computing MAX */
		r__1 = alp5 * elblavg[k - 1];
		elblavg[k - 1] = max(r__1,elb_mf__);
	    } else {
		elb = rc0010;
		elf = elb;
		elblavg[k - 1] = p4 * elblavg[k - 1];
	    }
/* Computing MAX */
	    r__1 = karman, r__2 = zwk - four;
	    z_m__ = max(r__1,r__2);
	    if (*rmol > rc0011) {
/* Computing MIN */
		r__1 = zwk * *rmol;
		els = karman * zwk / (one + cns * min(r__1,zmax));
/* Computing MIN */
		r__1 = zwk * *rmol;
		els1 = karman * z_m__ / (one + cns * min(r__1,zmax));
	    } else {
		d__1 =  (one - alp4 * zwk * *rmol);
		d__2 =  p2;
		els = karman * zwk * (real)pow_dd(&d__1, &d__2);
		d__1 =  (one - alp4 * zwk * *rmol);
		d__2 =  p2;
		els1 = karman * z_m__ * (real)pow_dd(&d__1, &d__2);
	    }
	    wt = p5 * (real)tanh((real)((zwk - (pblh2 + h1)) / h2)) + p5;
	    el_unstab__ = els / (one + els1 / elt);
	    el[k] = min(el_unstab__,elb);
/* Computing MIN */
	    r__1 = el[k];
	    el[k] = min(r__1,elf);
	    if (*xland - rc0018 >= zero) {
		el[k] *= wt_u1__;
	    }
	    el[k] = el[k] * (one - wt) + elblavg[k - 1] * wt;
/* L9019: */
	}
/* L9020: */
    } else if (*bl_mynn_mixlength__ == 2) {
	uonset = rc0015 + dz[*kts] * p1;
/* Computing 2nd power */
	r__1 = u1[*kts];
/* Computing 2nd power */
	r__2 = v1[*kts];
	ugrid = (real)sqrt((real)(r__1 * r__1 + r__2 * r__2));
	cns = rc0015;
	alp1 = rc0030;
	alp2 = p3;
	alp3 = rc0031;
	alp4 = five;
	alp5 = alp2;
	alp6 = fifty;
	pblh2 = max(*pblh,rc0020);
/* Computing MAX */
	r__1 = p3 * pblh2;
	h1 = max(r__1,rc0021);
	h1 = min(h1,rc0022);
	h2 = h1 * p5;
/* Computing MAX */
	r__1 = p5 * qke[*kts], r__2 = p5 * qkemin;
	qtke[*kts - 1] = max(r__1,r__2);
/* Computing MAX */
	r__1 = qke[*kts];
	qkw[*kts] = (real)sqrt((real)(max(r__1,qkemin)));
/* Computing MAX */
	r__1 = qke[*kts] + qpe[*kts];
	qtw[*kts] = (real)sqrt((real)(max(r__1,qkemin)));
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    afk = dz[k] / (dz[k] + dz[k - 1]);
	    abk = one - afk;
/* Computing MAX */
	    r__1 = qke[k] * abk + qke[k - 1] * afk;
	    qkw[k] = (real)sqrt((real)(max(r__1,qkemin)));
/* Computing MAX */
	    r__1 = (qke[k] + qpe[k]) * abk + (qke[k - 1] + qpe[k - 1]) * afk;
	    qtw[k] = (real)sqrt((real)(max(r__1,qkemin)));
/* Computing 2nd power */
	    r__1 = qkw[k];
	    qtke[k - 1] = p5 * (r__1 * r__1);
/* L9022: */
	}
/* L9023: */
	elt = rc0008;
	vsc = rc0008;
/* Computing MAX */
	r__1 = *pblh + h1;
	pblh_plus_ent__ = max(r__1,rc0002);
	k = *kts + 1;
	zwk = zw[k];
L9024:
	if (! (zwk <= pblh_plus_ent__)) {
	    goto L9025;
	}
	dzk = p5 * (dz[k] + dz[k - 1]);
/* Computing MIN */
/* Computing MAX */
	r__2 = qkw[k] - qmin;
	r__1 = max(r__2,rc0009);
	qdz = min(r__1,rc0025) * dzk;
	elt += qdz * zwk;
	vsc += qdz;
	++k;
	zwk = zw[k];
	goto L9024;
L9025:
/* Computing MIN */
/* Computing MAX */
	r__2 = alp1 * elt / vsc;
	r__1 = max(r__2,ten);
	elt = min(r__1,rc0027);
	vflx = *fltv;
	d__1 =  (gtr * elt * max(vflx,zero));
	d__2 =  p333;
	vsc = (real)pow_dd(&d__1, &d__2);
	el[*kts] = zero;
	/* unread private value */;
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    zwk = zw[k];
	    dzk = p5 * (dz[k] + dz[k - 1]);
	    /* unread private value */;
/* Computing MAX */
	    r__2 = p5 * (edmf_a1__[k] + edmf_a1__[k - 1]) * (p5 * (edmf_w1__[
		    k] + edmf_w1__[k - 1])), r__3 = (r__1 = edmf_a_dd1__[k - 
		    1] * edmf_w_dd1__[k - 1], abs(r__1));
	    qkw_mf__ = max(r__2,r__3);
	    if (dtv[k] > rc0011) {
/* Computing MAX */
		r__1 = (real)sqrt((real)(gtr * dtv[k]));
		bv = max(r__1,rc0029);
/* Computing MAX */
		r__1 = alp2 * qkw[k], r__2 = alp6 * qkw_mf__;
		elb_mf__ = max(r__1,r__2) / bv * (one + alp3 * (real)sqrt((
			real)(vsc / (bv * elt))));
/* Computing MIN */
/* Computing MAX */
		r__2 = alp5 * qkw[k], r__3 = alp6 * qkw_mf__;
		r__1 = max(r__2,r__3) / bv;
		elb = min(r__1,zwk);
		d__1 =  (gtr * *pblh * max(vflx,rc0033));
		d__2 =  p333;
		wstar = rc0032 * (real)pow_dd(&d__1, &d__2);
/* Computing MIN */
/* Computing MAX */
		r__2 = ctau * wstar / grav;
		r__1 = max(r__2,rc0034);
		tau_cloud__ = min(r__1,rc0035);
		wt = p5 * (real)tanh((real)((zwk - (pblh2 + h1)) / h2)) + p5;
		tau_cloud__ = tau_cloud__ * (rc0036 - wt) + fifty * wt;
/* Computing MIN */
/* Computing MAX */
/* Computing MIN */
		r__4 = qtke[k - 1];
		r__2 = tau_cloud__ * (real)sqrt((real)(min(r__4,forty))), 
			r__3 = alp6 * qkw_mf__ / bv;
		r__1 = max(r__2,r__3);
		elf = min(r__1,zwk);
	    } else {
		d__1 =  (gtr * *pblh * max(vflx,rc0033));
		d__2 =  p333;
		wstar = rc0032 * (real)pow_dd(&d__1, &d__2);
/* Computing MIN */
/* Computing MAX */
		r__2 = ctau * wstar / grav;
		r__1 = max(r__2,fifty);
		tau_cloud__ = min(r__1,rc0020);
		wt = p5 * (real)tanh((real)((zwk - (pblh2 + h1)) / h2)) + p5;
/* Computing MAX */
		r__1 = hundred, r__2 = dzk * p25;
		tau_cloud__ = tau_cloud__ * (one - wt) + max(r__1,r__2) * wt;
/* Computing MIN */
/* Computing MIN */
		r__2 = qtke[k - 1];
		r__1 = tau_cloud__ * (real)sqrt((real)(min(r__2,rc0037)));
		elb = min(r__1,zwk);
		elf = elb;
		elb_mf__ = elb;
	    }
	    elf /= one + elf / rc0038;
	    elb_mf__ = max(elb_mf__,rc0024);
	    if (*rmol > rc0011) {
/* Computing MIN */
		r__1 = zwk * *rmol;
		els = karman * zwk / (one + cns * min(r__1,zmax));
	    } else {
		d__1 =  (one - alp4 * zwk * *rmol);
		d__2 =  p2;
		els = karman * zwk * (real)pow_dd(&d__1, &d__2);
	    }
	    wt = p5 * (real)tanh((real)((zwk - (pblh2 + h1)) / h2)) + p5;
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
	    el[k] = (real)sqrt((real)(r__1 * r__1 / (one + r__2 * r__2 / (
		    r__3 * r__3) + r__4 * r__4 / (r__5 * r__5))));
	    el[k] = el[k] * (one - wt) + elf * wt;
/* L9028: */
	}
/* L9029: */
    }
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	el_les__ = p25 * p5 * (dz[k] + dz[k - 1]);
/* Computing MIN */
	r__1 = el_les__, r__2 = el[k];
	el[k] = el[k] * *psig_bl__ + (one - *psig_bl__) * min(r__1,r__2);
/* L9031: */
    }
/* L9032: */
    return 0;
} /* mym_length__ */

__device__ M471_CALL /* Subroutine */ int boulac_length0__(unsigned char *m471_work, int m471_n, M471Ptr<integer> k, M471Ptr<integer> kts, M471Ptr<integer> kte, 
	M471Ptr<real> zw, M471Ptr<real> dz, M471Ptr<real> qtke, M471Ptr<real> theta, M471Ptr<real> lb1, M471Ptr<real> lb2) {
    /* Initialized data */

    real gtr = .032700002193450927734375f;
    real zero = 0.f;
    real two = 2.f;
    real p1 = .100000001490116119384765625f;
    real p5 = .5f;
    real rc0000 = 1e3f;

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
    zup = zero;
    dlu = zw[*kte + 1] - zw[*k] - dz[*k] * p5;
    zzz = zero;
    zup_inf__ = zero;
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
	    zup += beta * (theta[izz + 1] + theta[izz]) * dzt * p5;
	    zzz += dzt;
	    if (qtke[*k] < zup && qtke[*k] >= zup_inf__) {
		bbb = (theta[izz + 1] - theta[izz]) / dzt;
		if (bbb != zero) {
/* Computing MAX */
/* Computing 2nd power */
		    r__3 = beta * (theta[izz] - theta[*k]);
		    r__1 = zero, r__2 = r__3 * r__3 + two * bbb * beta * (
			    qtke[*k] - zup_inf__);
		    tl = (-beta * (theta[izz] - theta[*k]) + (real)sqrt((real)
			    (max(r__1,r__2)))) / bbb / beta;
		} else {
		    if (theta[izz] != theta[*k]) {
			tl = (qtke[*k] - zup_inf__) / (beta * (theta[izz] - 
				theta[*k]));
		    } else {
			tl = zero;
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
    zdo = zero;
    zdo_sup__ = zero;
    dld = zw[*k];
    zzz = zero;
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
	    zdo -= beta * (theta[izz - 1] + theta[izz]) * dzt * p5;
	    zzz += dzt;
	    if (qtke[*k] < zdo && qtke[*k] >= zdo_sup__) {
		bbb = (theta[izz] - theta[izz - 1]) / dzt;
		if (bbb != zero) {
/* Computing MAX */
/* Computing 2nd power */
		    r__3 = beta * (theta[izz] - theta[*k]);
		    r__1 = zero, r__2 = r__3 * r__3 + two * bbb * beta * (
			    qtke[*k] - zdo_sup__);
		    tl = (beta * (theta[izz] - theta[*k]) + (real)sqrt((real)(
			    max(r__1,r__2)))) / bbb / beta;
		} else {
		    if (theta[izz] != theta[*k]) {
			tl = (qtke[*k] - zdo_sup__) / (beta * (theta[izz] - 
				theta[*k]));
		    } else {
			tl = zero;
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
    r__1 = p1, r__2 = min(dlu,rc0000);
    dlu = max(r__1,r__2);
/* Computing MAX */
    r__1 = p1, r__2 = min(dld,rc0000);
    dld = max(r__1,r__2);
    *lb2 = (real)sqrt((real)(dlu * dld));
    if (*k == *kte) {
	*lb1 = zero;
	*lb2 = zero;
    }
    return 0;
} /* boulac_length0__ */

__device__ M471_CALL /* Subroutine */ int boulac_length__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> zw, 
	M471Ptr<real> dz, M471Ptr<real> qtke, M471Ptr<real> theta, M471Ptr<real> lb1, M471Ptr<real> lb2) {
    /* Initialized data */

    real gtr = .032700002193450927734375f;
    real zero = 0.f;
    real one = 1.f;
    real two = 2.f;
    real p1 = .100000001490116119384765625f;
    real p5 = .5f;
    real rc0000 = 1500.f;
    real rc0001 = 0.f;

    /* System generated locals */
    integer qtke_offset, dz_offset, theta_offset, lb1_offset, lb2_offset, 
	    zw_offset, i__1;
    real r__1, r__2, r__3;

    /* Builtin functions */
    

    /* Local variables */
    real tl;
    integer iz;
    real bbb, zdo, dzt;
M471Ptr<real> dld=m471_array<real>(m471_work,(380*(MYNN471_NZ-49)+18740),m471_n);
M471Ptr<real> dlu=m471_array<real>(m471_work,(384*(MYNN471_NZ-49)+18936),m471_n);
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
    
#pragma unroll 4
for (iz = *kts; iz <= i__1; ++iz) {
	zup = zero;
	dlu[iz - 1] = zw[*kte + 1] - zw[iz] - dz[iz] * p5;
	zzz = zero;
	zup_inf__ = zero;
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
		zup += beta * (theta[izz + 1] + theta[izz]) * dzt * p5;
		zzz += dzt;
		if (qtke[iz] < zup && qtke[iz] >= zup_inf__) {
		    bbb = (theta[izz + 1] - theta[izz]) / dzt;
		    if (bbb != rc0001) {
/* Computing MAX */
/* Computing 2nd power */
			r__3 = beta * (theta[izz] - theta[iz]);
			r__1 = zero, r__2 = r__3 * r__3 + two * bbb * beta * (
				qtke[iz] - zup_inf__);
			tl = (-beta * (theta[izz] - theta[iz]) + (real)sqrt((
				real)(max(r__1,r__2)))) / bbb / beta;
		    } else {
			if (theta[izz] != theta[iz]) {
			    tl = (qtke[iz] - zup_inf__) / (beta * (theta[izz] 
				    - theta[iz]));
			} else {
			    tl = zero;
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
	zdo = zero;
	zdo_sup__ = zero;
	dld[iz - 1] = zw[iz];
	zzz = zero;
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
		zdo -= beta * (theta[izz - 1] + theta[izz]) * dzt * p5;
		zzz += dzt;
		if (qtke[iz] < zdo && qtke[iz] >= zdo_sup__) {
		    bbb = (theta[izz] - theta[izz - 1]) / dzt;
		    if (bbb != rc0001) {
/* Computing MAX */
/* Computing 2nd power */
			r__3 = beta * (theta[izz] - theta[iz]);
			r__1 = zero, r__2 = r__3 * r__3 + two * bbb * beta * (
				qtke[iz] - zdo_sup__);
			tl = (beta * (theta[izz] - theta[iz]) + (real)sqrt((
				real)(max(r__1,r__2)))) / bbb / beta;
		    } else {
			if (theta[izz] != theta[iz]) {
			    tl = (qtke[iz] - zdo_sup__) / (beta * (theta[izz] 
				    - theta[iz]));
			} else {
			    tl = zero;
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
	r__1 = p1, r__2 = dlu[iz - 1] / (one + dlu[iz - 1] / lmax);
	dlu[iz - 1] = max(r__1,r__2);
/* Computing MAX */
	r__1 = p1, r__2 = dld[iz - 1] / (one + dld[iz - 1] / lmax);
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
	 M471Ptr<real> closure, M471Ptr<real> ckmod, M471Ptr<real> taue, M471Ptr<real> dz, M471Ptr<real> dx, M471Ptr<real> zw,
	 M471Ptr<real> p, M471Ptr<real> exner, M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> thl, M471Ptr<real> thv, M471Ptr<real> thlv, M471Ptr<real> ql, M471Ptr<real> qw, M471Ptr<real> qke, M471Ptr<real> tsq, M471Ptr<real> qsq, M471Ptr<real> cov, 
	M471Ptr<real> vt, M471Ptr<real> vq, M471Ptr<real> rmol, M471Ptr<real> rmolh, M471Ptr<real> flt, M471Ptr<real> fltv, 
	M471Ptr<real> flq, M471Ptr<real> pblh, M471Ptr<real> theta, M471Ptr<real> sh, M471Ptr<real> sm, M471Ptr<real> el, 
	M471Ptr<real> dfm, M471Ptr<real> dfh, M471Ptr<real> dfq, M471Ptr<real> tcd, M471Ptr<real> qcd, M471Ptr<real> pdk, 
	M471Ptr<real> pdt, M471Ptr<real> pdq, M471Ptr<real> pdc, M471Ptr<real> qwt1, M471Ptr<real> qshear1, M471Ptr<real> qbuoy1, M471Ptr<real> qdiss1, M471Ptr<integer> tke_budget__, M471Ptr<real> psig_bl__, M471Ptr<real> psig_shcu__, M471Ptr<real> cldfra_bl1__, M471Ptr<integer> bl_mynn_mixlength__, 
	M471Ptr<integer> bl_mynn_ess__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w_dd1__, M471Ptr<real> edmf_a_dd1__, M471Ptr<real> tkeprod_dn__, M471Ptr<real> tkeprod_up__, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__) {
    /* Initialized data */

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
    real zero = 0.f;
    real one = 1.f;
    real two = 2.f;
    real three = 3.f;
    real four = 4.f;
    real seven = 7.f;
    real nine = 9.f;
    real p1 = .100000001490116119384765625f;
    real p2 = .20000000298023223876953125f;
    real p5 = .5f;
    real rc0000 = 6.f;
    real rc0001 = 4.f;
    real rc0002 = 200.f;
    real rc0003 = 2.7f;
    real rc0004 = .8f;
    real rc0005 = 1e-5f;
    real rc0006 = 1e-10f;
    real rc0007 = .01f;
    real rc0008 = 6.873f;
    real rc0009 = 0.f;
    real rc0010 = 1e-8f;
    real rc0011 = .004f;
    real rc0012 = 3.f;
    real rc0013 = 27.f;
    real rc0014 = 54.f;
    real rc0015 = 12.f;
    real rc0016 = 18.f;
    real rc0017 = .12f;
    real rc0018 = .76f;
    real rc0019 = .3f;
    real rc0020 = .1f;
    real rc0021 = 2.f;
    real rc0022 = .04f;
    real rc0023 = 10.f;
    real rc0024 = 1e-6f;
    real rc0025 = 1.5f;
    real rc0026 = 8e3f;
    real rc0027 = 2e3f;
    real rc0028 = .001f;
    real gtr = .032700002193450927734375f;
    real tv0 = 182.50872802734375f;

    /* System generated locals */
    integer dz_offset, zw_offset, u_offset, v_offset, thl_offset, thv_offset, 
	    thlv_offset, qw_offset, ql_offset, vt_offset, vq_offset, 
	    qke_offset, tsq_offset, qsq_offset, cov_offset, cldfra_bl1_offset,
	     edmf_w1_offset, edmf_a1_offset, edmf_w_dd1_offset, 
	    edmf_a_dd1_offset, tkeprod_dn_offset, tkeprod_up_offset, p_offset,
	     exner_offset, dfm_offset, dfh_offset, dfq_offset, pdk_offset, 
	    pdt_offset, pdq_offset, pdc_offset, tcd_offset, qcd_offset, 
	    el_offset, qwt1_offset, qshear1_offset, qbuoy1_offset, 
	    qdiss1_offset, sm_offset, sh_offset, theta_offset, 
	    pattern_spp_pbl1_offset, i__1;
    real r__1, r__2, r__3, r__4, r__5, r__6;
    doublereal d__1, d__2, d__3, d__4;

    /* Builtin functions */
    

    /* Local variables */
    integer k;
    doublereal e1, e2, e3, e4;
    real  ri;
M471Ptr<real> gh=m471_array<real>(m471_work,(304*(MYNN471_NZ-49)+15016),m471_n);
M471Ptr<real> gm=m471_array<real>(m471_work,(308*(MYNN471_NZ-49)+15212),m471_n);
    
    real wt, e6c;
    
    real adh, abk, aeh, afk, adm, aem, elh, sh20, auh, shb, dzk, vtt, vqq, cw25, smd, elq,
	     duz, aum, req, rsl, sm20;
M471Ptr<real> dtl=m471_array<real>(m471_work,(312*(MYNN471_NZ-49)+15408),m471_n);
M471Ptr<real> dqw=m471_array<real>(m471_work,(316*(MYNN471_NZ-49)+15604),m471_n);
M471Ptr<real> qkw=m471_array<real>(m471_work,(320*(MYNN471_NZ-49)+15800),m471_n);
M471Ptr<real> qtw=m471_array<real>(m471_work,(324*(MYNN471_NZ-49)+15996),m471_n);
M471Ptr<real> dtv=m471_array<real>(m471_work,(328*(MYNN471_NZ-49)+16192),m471_n);
M471Ptr<real> qpe=m471_array<real>(m471_work,(332*(MYNN471_NZ-49)+16388),m471_n);
    doublereal c2sq, c3sq;
    real prlimit_fre__, prlimit_sfc__, rsl2;
    doublereal q2sq, r2sq, q3sq, t2sq, t3sq, r3sq, eden, ghel;
    real gamq;
    doublereal gmel;
    real gamt, gamv;
    doublereal wden, dlsq;
    real clow, cupp, elqt, a2fac;
    doublereal elsq, qdiv, enum__;
    real pblh2, cpblh, gmelq, mfmax, prlim;
    integer ixarr1;
    real cldavg, sh25min, sh25max, sm25min, sm25max;

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
    exner_offset = *kts;
    exner -= exner_offset;
    p_offset = *kts;
    p -= p_offset;
    zw_offset = *kts;
    zw -= zw_offset;
    dz_offset = *kts;
    dz -= dz_offset;

    /* Function Body */
    prlimit_fre__ = rc0000;
    prlimit_sfc__ = rc0001;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	qkw[ixarr1 - 1] = zero;
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	qtw[ixarr1 - 1] = zero;
    }
    pblh2 = max(*pblh,rc0002);
    if (*closure < rc0003) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    qpe[ixarr1 - 1] = zero;
	}
    } else {
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing MIN */
	    r__1 = (zw[k] + p1 * pblh2) / (rc0004 * pblh2);
	    cpblh = min(r__1,one);
/* Computing MIN */
/* Computing MAX */
/* Computing 2nd power */
	    r__5 = gtr;
/* Computing 2nd power */
	    r__6 = *taue;
	    r__3 = zero, r__4 = cpblh * two * tsq[k] * (r__5 * r__5) * (r__6 *
		     r__6);
	    r__1 = three, r__2 = max(r__3,r__4);
	    qpe[k - 1] = min(r__1,r__2);
/* L9004: */
	}
/* L9005: */
    }
    mym_level2__(m471_work,m471_n, kts, kte, ckmod, bl_mynn_ess__, &zw[zw_offset], &dz[
	    dz_offset], xland, pblh, &u[u_offset], &v[v_offset], &thl[
	    thl_offset], &thv[thv_offset], &thlv[thlv_offset], &theta[
	    theta_offset], &p[p_offset], &exner[exner_offset], &qke[
	    qke_offset], &cldfra_bl1__[cldfra_bl1_offset], &edmf_a1__[
	    edmf_a1_offset], &edmf_w1__[edmf_w1_offset], &qw[qw_offset], &ql[
	    ql_offset], &vt[vt_offset], &vq[vq_offset], dtl, dqw, dtv, gm, gh,
	     &sm[sm_offset], &sh[sh_offset]);
    mym_length__(m471_work,m471_n, kts, kte, xland, &dz[dz_offset], dx, &zw[zw_offset], rmol, 
	    rmolh, flt, fltv, flq, &vt[vt_offset], &vq[vq_offset], &u[
	    u_offset], &v[v_offset], &qke[qke_offset], qpe, dtv, &el[
	    el_offset], pblh, &theta[theta_offset], qkw, qtw, psig_bl__, &
	    cldfra_bl1__[cldfra_bl1_offset], bl_mynn_mixlength__, &edmf_w1__[
	    edmf_w1_offset], &edmf_a1__[edmf_a1_offset], &edmf_w_dd1__[
	    edmf_w_dd1_offset], &edmf_a_dd1__[edmf_a_dd1_offset]);
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	dzk = p5 * (dz[k] + dz[k - 1]);
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
	/* unread private value */;
/* Computing MAX */
	r__1 = sm[k];
	/* unread private value */;
/* Computing MAX */
	r__1 = sh[k];
	sh[k] = max(r__1,rc0005);
/* Computing 2nd power */
	r__1 = u[k] - u[k - 1];
/* Computing 2nd power */
	r__2 = v[k] - v[k - 1];
	duz = r__1 * r__1 + r__2 * r__2;
/* Computing 2nd power */
	r__1 = dzk;
	duz /= r__1 * r__1;
	ri = -gh[k - 1] / max(duz,rc0006);
	a2fac = one / (one + *ckmod * max(ri,zero));
	if (ri >= one) {
/* Computing MIN */
	    r__1 = seven * ri;
	    prlim = min(r__1,prlimit_fre__);
	} else if (ri >= rc0007 && ri <= one) {
/* Computing MIN */
	    r__1 = rc0008 * ri + one / (rc0008 * ri);
	    prlim = min(r__1,prlimit_fre__);
	} else {
	    prlim = prlimit_fre__;
	}
/* Computing MIN */
	r__1 = one, r__2 = zw[k] / max(*pblh,rc0002);
	wt = min(r__1,r__2);
	prlim = wt * prlim + (one - wt) * min(prlimit_sfc__,prlim);
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
	    sm[k] = q3sq * a1 * (e3 - three * c1 * e4) / eden;
	    sh[k] = q3sq * (a2 * a2fac) * (e2 + three * c1 * e5c * gmel) / 
		    eden;
	}
/* Computing MAX */
	r__1 = (real) (gmel / q3sq);
	/* unread private value */;
	/* unread private value */;
	sh25max = four;
	/* unread private value */;
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
	shb = max(r__1,rc0011);
/* Computing MIN */
	r__1 = sm[k], r__2 = prlim * shb;
	sm[k] = min(r__1,r__2);
	if (*closure >= rc0012) {
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
	    cw25 = e1 * (e2 + three * c1 * e5c * gmel * (d__1 * d__1)) / (
		    three * eden);
	    dlsq = elsq;
	    if (q3sq / dlsq < -gh[k - 1]) {
		q3sq = -dlsq * gh[k - 1];
	    }
/* Computing 2nd power */
	    r__1 = a2 * a2fac;
/* Computing 2nd power */
	    r__2 = gtr;
	    /* unread private value */;
/* Computing 2nd power */
	    r__1 = a1;
	    /* unread private value */;
/* Computing 2nd power */
	    r__1 = a2 * a2fac;
/* Computing 2nd power */
	    r__2 = gtr;
	    /* unread private value */;
/* Computing 2nd power */
	    r__1 = a1;
	    /* unread private value */;
/* Computing 2nd power */
	    r__1 = a2 * a2fac;
/* Computing 2nd power */
	    r__2 = a2 * a2fac;
	    /* unread private value */;
/* Computing 2nd power */
	    r__1 = a1;
	    /* unread private value */;
	    /* unread private value */;
	    /* unread private value */;
	    /* unread private value */;
	    /* unread private value */;
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
		clow = q3sq * (rc0017 - cw25) * eden / wden;
		cupp = q3sq * (rc0018 - cw25) * eden / wden;
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
	    e6c = rc0012 * (a2 * a2fac) * cc3 * gtr * dlsq / elsq;
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
		if (sh[k] < -rc0019 || sm[k] < -rc0019 || qke[k] < -rc0020 || 
			abs(smd) > rc0021) {
		}
	    }
	} else {
	    gamt = zero;
	    gamq = zero;
	    gamv = zero;
	}
	cldavg = p5 * (cldfra_bl1__[k - 1] + cldfra_bl1__[k]);
/* Computing MAX */
	r__2 = p5 * (edmf_a1__[k - 1] + edmf_a1__[k]) * p5 * (edmf_w1__[k - 1]
		 + edmf_w1__[k]), r__3 = (r__1 = edmf_a_dd1__[k] * 
		edmf_w_dd1__[k], abs(r__1));
	mfmax = max(r__2,r__3);
/* Computing MAX */
/* Computing MIN */
	r__3 = rc0023 * mfmax;
	r__1 = sm[k], r__2 = rc0022 * min(r__3,one);
	sm[k] = max(r__1,r__2);
/* Computing MAX */
/* Computing MIN */
	r__3 = rc0023 * mfmax;
	r__1 = sh[k], r__2 = rc0022 * min(r__3,one);
	sh[k] = max(r__1,r__2);
/* Computing MAX */
	r__1 = sm[k], r__2 = rc0022 * min(cldavg,p5);
	sm[k] = max(r__1,r__2);
/* Computing MAX */
	r__1 = sh[k], r__2 = rc0022 * min(cldavg,p5);
	sh[k] = max(r__1,r__2);
	if (tkeprod_dn__[k] > rc0024) {
/* Computing MAX */
	    r__1 = sm[k];
	    sm[k] = max(r__1,rc0007);
	}
	if (tkeprod_dn__[k] > rc0024) {
/* Computing MAX */
	    r__1 = sh[k];
	    sh[k] = max(r__1,rc0007);
	}
	if (*closure >= rc0003 && *closure < rc0012) {
/* Computing MAX */
/* Computing MIN */
/* Computing MAX */
	    r__5 = zero, r__6 = three * qpe[k - 1];
	    r__3 = p2, r__4 = max(r__5,r__6);
	    r__1 = sm[k], r__2 = min(r__3,r__4);
	    sm[k] = max(r__1,r__2);
	}
	elq = el[k] * qkw[k - 1];
	elqt = el[k] * qtw[k - 1];
	elh = elq * qdiv;
	pdk[k] = elq * (sm[k] * gm[k - 1] + sh[k] * gh[k - 1] + gamv) + p5 * 
		tkeprod_dn__[k] + p5 * tkeprod_up__[k];
	pdt[k] = elh * (sh[k] * dtl[k - 1] + gamt) * dtl[k - 1];
	pdq[k] = elh * (sh[k] * dqw[k - 1] + gamq) * dqw[k - 1];
	pdc[k] = elh * (sh[k] * dtl[k - 1] + gamt) * dqw[k - 1] * p5 + elh * (
		sh[k] * dqw[k - 1] + gamq) * dtl[k - 1] * p5;
	tcd[k] = elq * gamt;
	qcd[k] = elq * gamq;
	dfm[k] = elqt * sm[k] / dzk;
	dfh[k] = elq * sh[k] / dzk;
	dfq[k] = dfm[k];
	if (*tke_budget__ == 1) {
	    qshear1[k] = elq * sm[k] * gm[k - 1];
	    qbuoy1[k] = elq * (sh[k] * gh[k - 1] + gamv) + p5 * tkeprod_dn__[
		    k] + p5 * tkeprod_up__[k];
	}
/* L9007: */
    }
/* L9008: */
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
    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	dzk = dz[k];
	tcd[k] = (tcd[k + 1] - tcd[k]) / dzk;
	qcd[k] = (qcd[k + 1] - qcd[k]) / dzk;
/* L9010: */
    }
/* L9011: */
    if (*spp_pbl__ == 1) {
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
/* Computing MAX */
	    r__2 = zw[k] - rc0026;
	    r__1 = (real)exp((real)(-max(r__2,zero) / rc0027));
	    dfm[k] += dfm[k] * pattern_spp_pbl1__[k] * rc0025 * max(r__1,
		    rc0028);
/* Computing MAX */
/* Computing MAX */
	    r__2 = zw[k] - rc0026;
	    r__1 = (real)exp((real)(-max(r__2,zero) / rc0027));
	    dfh[k] += dfh[k] * pattern_spp_pbl1__[k] * rc0025 * max(r__1,
		    rc0028);
/* L9013: */
	}
/* L9014: */
    }
    return 0;
} /* mym_turbulence__ */

__device__ M471_CALL /* Subroutine */ int mym_predict__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> closure, 
	M471Ptr<real> delt, M471Ptr<real> dz, M471Ptr<real> ust, M471Ptr<real> flt, M471Ptr<real> flq, M471Ptr<real> pmz, 
	M471Ptr<real> phh, M471Ptr<real> el, M471Ptr<real> dfq, M471Ptr<real> rho, M471Ptr<real> pdk, M471Ptr<real> pdt, M471Ptr<real> pdq, M471Ptr<real> pdc, M471Ptr<real> qke, M471Ptr<real> tsq, M471Ptr<real> qsq, M471Ptr<real> cov, M471Ptr<real> s_aw1__, M471Ptr<real> s_awqke1__, M471Ptr<real> s_awqsq1__, M471Ptr<integer> bl_mynn_edmf__, 
	M471Ptr<integer> bl_mynn_edmf_tke__, M471Ptr<real> qwt1, M471Ptr<real> qdiss1, M471Ptr<integer> tke_budget__, M471Ptr<real> upwind) {
    /* Initialized data */

    real karman = .4000000059604644775390625f;
    real sqfac = 3.f;
    real qkemin = 9.999999747378751635551452636719e-6f;
    real b1 = 24.f;
    real b2 = 15.f;
    real zero = 0.f;
    real one = 1.f;
    real two = 2.f;
    real three = 3.f;
    real p5 = .5f;
    real rc0000 = 1e-4f;
    real rc0001 = .5f;
    real rc0002 = 1.f;
    real rc0003 = 0.f;
    real rc0004 = 150.f;
    real rc0005 = 2.5f;
    real rc0006 = 1e-17f;
    real rc0007 = 6e-6f;
    real rc0008 = .25f;
    real rc0009 = 2.7f;
    real rc0010 = 3.f;

    /* System generated locals */
    integer dz_offset, dfq_offset, el_offset, rho_offset, pdk_offset, 
	    pdt_offset, pdq_offset, pdc_offset, qke_offset, tsq_offset, 
	    qsq_offset, cov_offset, s_awqke1_offset, s_awqsq1_offset, 
	    s_aw1_offset, qwt1_offset, qdiss1_offset, i__1;
    real r__1, r__2, r__3, r__4;

    /* Builtin functions */
    

    /* Local variables */
    M471Ptr<real> a=m471_array<real>(m471_work,(304*(MYNN471_NZ-49)+15016),m471_n);
M471Ptr<real> b=m471_array<real>(m471_work,(308*(MYNN471_NZ-49)+15212),m471_n);
M471Ptr<real> c__=m471_array<real>(m471_work,(312*(MYNN471_NZ-49)+15408),m471_n);
M471Ptr<real> d__=m471_array<real>(m471_work,(316*(MYNN471_NZ-49)+15604),m471_n);
    integer k;
    real  b1l, b2l, phm, vkz, pdc1, pdk1, pdq1, pdt1, onoff;
M471Ptr<real> x=m471_array<real>(m471_work,(320*(MYNN471_NZ-49)+15800),m471_n);
M471Ptr<real> bp=m471_array<real>(m471_work,(324*(MYNN471_NZ-49)+15996),m471_n);
M471Ptr<real> rp=m471_array<real>(m471_work,(328*(MYNN471_NZ-49)+16192),m471_n);
M471Ptr<real> qkw=m471_array<real>(m471_work,(332*(MYNN471_NZ-49)+16388),m471_n);
M471Ptr<real> dtz=m471_array<real>(m471_work,(336*(MYNN471_NZ-49)+16584),m471_n);
M471Ptr<real> df3q=m471_array<real>(m471_work,(340*(MYNN471_NZ-49)+16780),m471_n);
M471Ptr<real> kmdz=m471_array<real>(m471_work,(344*(MYNN471_NZ-49)+16976),m471_n);
M471Ptr<real> kqdz=m471_array<real>(m471_work,(348*(MYNN471_NZ-49)+17176),m471_n);
M471Ptr<real> rhoz=m471_array<real>(m471_work,(352*(MYNN471_NZ-49)+17376),m471_n);
M471Ptr<real> dzinv=m471_array<real>(m471_work,(356*(MYNN471_NZ-49)+17576),m471_n);
    integer ixarr1;
    M471Ptr<real> dncont=m471_array<real>(m471_work,(360*(MYNN471_NZ-49)+17772),m471_n);
M471Ptr<real> tke_up__=m471_array<real>(m471_work,(364*(MYNN471_NZ-49)+17968),m471_n);
M471Ptr<real> rhoinv=m471_array<real>(m471_work,(368*(MYNN471_NZ-49)+18164),m471_n);
M471Ptr<real> upcont=m471_array<real>(m471_work,(372*(MYNN471_NZ-49)+18360),m471_n);
    

    /* Parameter adjustments */
    qdiss1_offset = *kts;
    qdiss1 -= qdiss1_offset;
    qwt1_offset = *kts;
    qwt1 -= qwt1_offset;
    s_awqsq1_offset = *kts;
    s_awqsq1__ -= s_awqsq1_offset;
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
    vkz = karman * p5 * dz[*kts];
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	r__1 = qke[k];
	qkw[k - 1] = (real)sqrt((real)(max(r__1,qkemin)));
	df3q[k - 1] = sqfac * dfq[k];
	dtz[k - 1] = *delt / dz[k];
/* L9004: */
    }
/* L9005: */
    rhoz[*kts - 1] = rho[*kts];
    rhoinv[*kts - 1] = one / rho[*kts];
    kqdz[*kts - 1] = rhoz[*kts - 1] * df3q[*kts - 1];
    kmdz[*kts - 1] = rhoz[*kts - 1] * dfq[*kts];
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	rhoz[k - 1] = (rho[k] * dz[k - 1] + rho[k - 1] * dz[k]) / (dz[k - 1] 
		+ dz[k]);
/* Computing MAX */
	r__1 = rhoz[k - 1];
	rhoz[k - 1] = max(r__1,rc0000);
/* Computing MAX */
	r__1 = rho[k];
	rhoinv[k - 1] = one / max(r__1,rc0000);
	kqdz[k - 1] = rhoz[k - 1] * df3q[k - 1];
	kmdz[k - 1] = rhoz[k - 1] * dfq[k];
/* L9007: */
    }
/* L9008: */
    rhoz[*kte] = rhoz[*kte - 1];
    kqdz[*kte] = rhoz[*kte] * df3q[*kte - 1];
    kmdz[*kte] = rhoz[*kte] * dfq[*kte];
    if (*bl_mynn_edmf__ == 1) {
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = kqdz[k - 1], r__2 = p5 * s_aw1__[k];
	    kqdz[k - 1] = max(r__1,r__2);
/* Computing MAX */
	    r__1 = kqdz[k - 1], r__2 = -p5 * (s_aw1__[k] - s_aw1__[k + 1]);
	    kqdz[k - 1] = max(r__1,r__2);
/* Computing MAX */
	    r__1 = kmdz[k - 1], r__2 = p5 * s_aw1__[k];
	    kmdz[k - 1] = max(r__1,r__2);
/* Computing MAX */
	    r__1 = kmdz[k - 1], r__2 = -p5 * (s_aw1__[k] - s_aw1__[k + 1]);
	    kmdz[k - 1] = max(r__1,r__2);
/* L9010: */
	}
/* L9011: */
    }
/* Computing 3rd power */
    r__1 = *ust;
    pdk1 = two * (r__1 * (r__1 * r__1)) * *pmz / vkz;
    /* unread private value */;
/* Computing 2nd power */
    r__1 = *flt;
    /* unread private value */;
/* Computing 2nd power */
    r__1 = *flq;
    /* unread private value */;
    /* unread private value */;
    pdk[*kts] = pdk1 - pdk[*kts + 1];
    pdt[*kts] = pdt[*kts + 1];
    pdq[*kts] = pdq[*kts + 1];
    pdc[*kts] = pdc[*kts + 1];
    i__1 = *kte - 1;
    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	b1l = b1 * p5 * (el[k + 1] + el[k]);
	bp[k - 1] = two * qkw[k - 1] / b1l;
	rp[k - 1] = pdk[k + 1] + pdk[k];
/* L9013: */
    }
/* L9014: */
    bp[*kte - 1] = zero;
    if (*bl_mynn_edmf__ > 1) {
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    upcont[k - 1] = s_awqke1__[k] - s_aw1__[k] * (qke[k] * *upwind + 
		    qke[k - 1] * (one - *upwind));
	    dncont[k - 1] = zero;
/* L9016: */
	}
/* L9017: */
	upcont[*kte - 1] = zero;
	dncont[*kte - 1] = zero;
	k = *kts;
	a[0] = zero;
	b[0] = one + dtz[k - 1] * kqdz[k] * rhoinv[k - 1] + bp[k - 1] * *delt;
	c__[0] = -dtz[k - 1] * kqdz[k] * rhoinv[k - 1];
/* Computing MAX */
	r__1 = qkemin, r__2 = qke[k];
	d__[0] = max(r__1,r__2) + rp[k - 1] * *delt - dtz[k - 1] * (upcont[k] 
		+ dncont[k]);
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * kqdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (kqdz[k - 1] + kqdz[k]) * rhoinv[k 
		    - 1] + bp[k - 1] * *delt;
	    c__[k - 1] = -dtz[k - 1] * kqdz[k] * rhoinv[k - 1];
/* Computing MAX */
	    r__1 = qkemin, r__2 = qke[k];
	    d__[k - 1] = max(r__1,r__2) + rp[k - 1] * *delt - dtz[k - 1] * (
		    upcont[k] - upcont[k - 1] + dncont[k] - dncont[k - 1]);
/* L9019: */
	}
/* L9020: */
    } else {
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * kqdz[k - 1] * rhoinv[k - 1] + rc0001 * 
		    dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] * onoff;
	    b[k - 1] = rc0002 + dtz[k - 1] * (kqdz[k - 1] + kqdz[k]) * rhoinv[
		    k - 1] + rc0001 * dtz[k - 1] * rhoinv[k - 1] * (s_aw1__[k]
		     - s_aw1__[k + 1]) * onoff + bp[k - 1] * *delt;
	    c__[k - 1] = -dtz[k - 1] * kqdz[k] * rhoinv[k - 1] - rc0001 * dtz[
		    k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] * onoff;
	    d__[k - 1] = rp[k - 1] * *delt + qke[k] + dtz[k - 1] * rhoinv[k - 
		    1] * (s_awqke1__[k] - s_awqke1__[k + 1]) * onoff;
/* L9022: */
	}
/* L9023: */
    }
    a[*kte - 1] = rc0003;
    b[*kte - 1] = rc0002;
    c__[*kte - 1] = rc0003;
    d__[*kte - 1] = qke[*kte];
    tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	r__1 = x[k - 1];
	qke[k] = max(r__1,qkemin);
/* Computing MIN */
	r__1 = qke[k];
	qke[k] = min(r__1,rc0004);
/* L9025: */
    }
/* L9026: */
    if (*tke_budget__ == 1) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    tke_up__[ixarr1 - 1] = p5 * qke[*kts + ixarr1 - 1];
	}
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    dzinv[ixarr1 - 1] = one / dz[*kts + ixarr1 - 1];
	}
	k = *kts;
	qwt1[k] = dzinv[k - 1] * (kqdz[k] * (tke_up__[k] - tke_up__[k - 1]) - 
		kqdz[k - 1] * tke_up__[k - 1] + p5 * rhoinv[k - 1] * (s_aw1__[
		k + 1] * tke_up__[k] + (s_aw1__[k + 1] - s_aw1__[k]) * 
		tke_up__[k - 1] + (s_awqke1__[k] - s_awqke1__[k + 1])) * 
		onoff);
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    qwt1[k] = dzinv[k - 1] * (kqdz[k] * (tke_up__[k] - tke_up__[k - 1]
		    ) - kqdz[k - 1] * (tke_up__[k - 1] - tke_up__[k - 2]) + 
		    p5 * rhoinv[k - 1] * (s_aw1__[k + 1] * tke_up__[k] + (
		    s_aw1__[k + 1] - s_aw1__[k]) * tke_up__[k - 1] - s_aw1__[
		    k] * tke_up__[k - 2] + (s_awqke1__[k] - s_awqke1__[k + 1])
		    ) * onoff);
/* L9028: */
	}
/* L9029: */
	k = *kte;
	qwt1[k] = dzinv[k - 1] * (-kqdz[k - 1] * (tke_up__[k - 1] - tke_up__[
		k - 2]) + p5 * rhoinv[k - 1] * (-s_aw1__[k] * tke_up__[k - 1] 
		- s_aw1__[k] * tke_up__[k - 2] + s_awqke1__[k]) * onoff);
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    qdiss1[*kts + ixarr1 - 1] = bp[ixarr1 - 1] * tke_up__[ixarr1 - 1];
	}
    }
    if (*closure > rc0005) {
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    b2l = b2 * p5 * (el[k + 1] + el[k]);
	    bp[k - 1] = two * qkw[k - 1] / b2l;
	    rp[k - 1] = pdq[k + 1] + pdq[k];
/* L9031: */
	}
/* L9032: */
	bp[*kte - 1] = zero;
	if (*bl_mynn_edmf__ > 1) {
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		upcont[k - 1] = s_awqsq1__[k] - s_aw1__[k] * (qsq[k] * *
			upwind + qsq[k - 1] * (one - *upwind));
		dncont[k - 1] = zero;
/* L9034: */
	    }
/* L9035: */
	    upcont[*kte - 1] = zero;
	    dncont[*kte - 1] = zero;
	    k = *kts;
	    a[0] = zero;
	    b[0] = one + dtz[k - 1] * kmdz[k] * rhoinv[k - 1] + bp[k - 1] * *
		    delt;
	    c__[0] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1];
/* Computing MAX */
	    r__1 = zero, r__2 = qsq[k];
	    d__[0] = max(r__1,r__2) + rp[k - 1] * *delt - dtz[k - 1] * (
		    upcont[k] + dncont[k]);
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * kmdz[k - 1] * rhoinv[k - 1];
		b[k - 1] = one + dtz[k - 1] * (kmdz[k - 1] + kmdz[k]) * 
			rhoinv[k - 1] + bp[k - 1] * *delt;
		c__[k - 1] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1];
/* Computing MAX */
		r__1 = zero, r__2 = qsq[k];
		d__[k - 1] = max(r__1,r__2) + rp[k - 1] * *delt - dtz[k - 1] *
			 (upcont[k] - upcont[k - 1] + dncont[k] - dncont[k - 
			1]);
/* L9037: */
	    }
/* L9038: */
	} else {
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * kmdz[k - 1] * rhoinv[k - 1] + rc0001 
			* dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] * onoff;
		b[k - 1] = rc0002 + dtz[k - 1] * (kmdz[k - 1] + kmdz[k]) * 
			rhoinv[k - 1] + rc0001 * dtz[k - 1] * rhoinv[k - 1] * 
			(s_aw1__[k] - s_aw1__[k + 1]) * onoff + bp[k - 1] * *
			delt;
		c__[k - 1] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1] - rc0001 * 
			dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] * onoff;
		d__[k - 1] = rp[k - 1] * *delt + qsq[k] + dtz[k - 1] * rhoinv[
			k - 1] * (s_awqsq1__[k] - s_awqsq1__[k + 1]) * onoff;
/* L9040: */
	    }
/* L9041: */
	}
	a[*kte - 1] = -one;
	b[*kte - 1] = one;
	c__[*kte - 1] = zero;
	d__[*kte - 1] = zero;
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = x[k - 1];
	    qsq[k] = max(r__1,rc0006);
/* Computing MIN */
	    r__1 = qsq[k];
	    qsq[k] = min(r__1,rc0007);
/* L9043: */
	}
/* L9044: */
    } else {
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    if (qkw[k - 1] <= zero) {
		b2l = zero;
	    } else {
		b2l = b2 * rc0008 * (el[k + 1] + el[k]) / qkw[k - 1];
	    }
/* Computing MIN */
/* Computing MAX */
	    r__3 = rc0006, r__4 = b2l * (pdq[k + 1] + pdq[k]);
	    r__1 = rc0007, r__2 = max(r__3,r__4);
	    qsq[k] = min(r__1,r__2);
/* L9046: */
	}
/* L9047: */
	qsq[*kte] = qsq[*kte - 1];
    }
    if (*closure >= rc0009) {
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    b2l = b2 * p5 * (el[k + 1] + el[k]);
	    bp[k - 1] = two * qkw[k - 1] / b2l;
	    rp[k - 1] = pdt[k + 1] + pdt[k];
/* L9049: */
	}
/* L9050: */
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * kmdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (kmdz[k - 1] + kmdz[k]) * rhoinv[k 
		    - 1] + bp[k - 1] * *delt;
	    c__[k - 1] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1];
/* Computing MAX */
	    r__1 = zero, r__2 = tsq[k];
	    d__[k - 1] = rp[k - 1] * *delt + max(r__1,r__2);
/* L9052: */
	}
/* L9053: */
	a[*kte - 1] = -rc0002;
	b[*kte - 1] = one;
	c__[*kte - 1] = zero;
	d__[*kte - 1] = zero;
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = x[k - 1];
	    r__1 = three, r__2 = max(r__3,r__4);
	    tsq[k] = min(r__1,r__2);
/* L9055: */
	}
/* L9056: */
    } else {
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    if (qkw[k - 1] <= zero) {
		b2l = zero;
	    } else {
		b2l = b2 * rc0008 * (el[k + 1] + el[k]) / qkw[k - 1];
	    }
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = b2l * (pdt[k + 1] + pdt[k]);
	    r__1 = three, r__2 = max(r__3,r__4);
	    tsq[k] = min(r__1,r__2);
/* L9058: */
	}
/* L9059: */
	tsq[*kte] = tsq[*kte - 1];
    }
    if (*closure >= rc0010) {
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    b2l = b2 * p5 * (el[k + 1] + el[k]);
	    bp[k - 1] = two * qkw[k - 1] / b2l;
	    rp[k - 1] = pdc[k + 1] + pdc[k];
/* L9061: */
	}
/* L9062: */
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * kmdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (kmdz[k - 1] + kmdz[k]) * rhoinv[k 
		    - 1] + bp[k - 1] * *delt;
	    c__[k - 1] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1];
	    d__[k - 1] = rp[k - 1] * *delt + cov[k];
/* L9064: */
	}
/* L9065: */
	a[*kte - 1] = -rc0002;
	b[*kte - 1] = one;
	c__[*kte - 1] = zero;
	d__[*kte - 1] = zero;
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    cov[k] = x[k - 1];
/* L9067: */
	}
/* L9068: */
    } else {
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    if (qkw[k - 1] <= zero) {
		b2l = zero;
	    } else {
		b2l = b2 * rc0008 * (el[k + 1] + el[k]) / qkw[k - 1];
	    }
	    cov[k] = b2l * (pdc[k + 1] + pdc[k]);
/* L9070: */
	}
/* L9071: */
	cov[*kte] = cov[*kte - 1];
    }
    return 0;
} /* mym_predict__ */

__device__ M471_CALL /* Subroutine */ int mym_condensation__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dx, 
	M471Ptr<real> dz, M471Ptr<real> zw, M471Ptr<real> xland, M471Ptr<real> thl, M471Ptr<real> qw, M471Ptr<real> qv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<real> qs, M471Ptr<real> p, M471Ptr<real> exner, M471Ptr<real> tsq, M471Ptr<real> qsq, 
	M471Ptr<real> cov, M471Ptr<real> sh, M471Ptr<real> el, M471Ptr<integer> bl_mynn_cloudpdf__, M471Ptr<real> qc_bl1__, M471Ptr<real> qi_bl1__, M471Ptr<real> cldfra_bl1__, M471Ptr<real> pblh1, M471Ptr<real> hfx1,
	 M471Ptr<real> vt, M471Ptr<real> vq, M471Ptr<real> th, M471Ptr<real> sgm, M471Ptr<real> closure, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__) {
    /* Initialized data */

    real cp = 1004.5f;
    real r_d__ = 287.f;
    real r_v__ = 461.600006103515625f;
    real xlv = 2.5e6f;
    real p608 = .6083624362945556640625f;
    real ep_2__ = .621750414371490478515625f;
    real cpv = 1846.4000244140625f;
    real tice = 238.f;
    real tliq = 269.f;
    real ep_3__ = .378249585628509521484375f;
    real tv0 = 182.50872802734375f;
    real xlvcp = 2488.80029296875f;
    real rr2 = .7071068286895751953125f;
    real rrp = .398942291736602783203125f;
    real b2 = 15.f;
    real zero = 0.f;
    real one = 1.f;
    real two = 2.f;
    real three = 3.f;
    real four = 4.f;
    real five = 5.f;
    real ten = 10.f;
    real hundred = 100.f;
    real p1 = .100000001490116119384765625f;
    real p2 = .20000000298023223876953125f;
    real p5 = .5f;
    real rc0000 = .008f;
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
    real rc0019 = .03f;
    real rc0020 = 1e-10f;
    real rc0021 = 1e-10f;
    real rc0022 = .001f;
    real rc0023 = 2.7f;
    real rc0024 = 3.f;
    real rc0025 = 1e-13f;
    real rc0026 = 100.f;
    real rc0027 = 500.f;
    real rc0028 = 250.f;
    real rc0029 = 150.f;
    real rc0030 = 50.f;
    real rc0031 = .02f;
    real rc0032 = 300.f;
    real rc0033 = 1e-10f;
    real rc0034 = .037f;
    real rc0035 = 3.f;
    real rc0036 = 1e-5f;
    real rc0037 = .07f;
    real rc0038 = 200.f;
    real rc0039 = .35f;
    real rc0040 = 3.6f;
    real rc0041 = .05f;
    real rc0042 = .39f;
    real rc0043 = 1.6f;
    real rc0044 = .55f;
    real rc0045 = .56f;
    real rc0046 = .976f;
    real rc0047 = .955f;
    real rc0048 = .055f;
    real rc0049 = 1.5f;
    real rc0050 = 2.5f;
    real rc0051 = 1.7f;
    real rc0052 = .4f;
    real rc0053 = 3.8f;
    real rc0054 = 23.9f;
    real rc0055 = 60.f;
    real rc0056 = .6f;
    real rc0057 = 25.f;
    real rc0058 = 1.61f;

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
M471Ptr<real> a=m471_array<real>(m471_work,(304*(MYNN471_NZ-49)+15016),m471_n);
M471Ptr<real> b=m471_array<real>(m471_work,(308*(MYNN471_NZ-49)+15212),m471_n);
    integer k;
    real t, cldfra_rh__, cldfra_rh0__, cldfra_rh1__, bb, pt, qt, xl;
M471Ptr<real> q1=m471_array<real>(m471_work,(312*(MYNN471_NZ-49)+15408),m471_n);
M471Ptr<real> ql=m471_array<real>(m471_work,(316*(MYNN471_NZ-49)+15604),m471_n);
M471Ptr<real> rh=m471_array<real>(m471_work,(320*(MYNN471_NZ-49)+15800),m471_n);
    
    real wt;
    
    real cldfra_qsq__, eq1, q1k, ht1, ht2, q2p, wt2, cldfra_qsq0__, 
	    cldfra_qsq1__, rac, fng;
M471Ptr<real> bet=m471_array<real>(m471_work,(324*(MYNN471_NZ-49)+15996),m471_n);
M471Ptr<real> alp=m471_array<real>(m471_work,(328*(MYNN471_NZ-49)+16192),m471_n);
    
    real cpm, dth, qsl, qll, rsl, qmq, zsl, dqw, dzk, qww;
    doublereal c3sq, r3sq, t3sq;
    real beta, clim, esat, sgmc, zagl, qlim, dqsl, sgmq, pblh2, q1_rh__, 
	    alpha, cfmax, rhmax, theta1, theta2, rh_adj__, ql_ice__, rhcrit, 
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
    clim_pbl__ = rc0001;
    clim_trp__ = rc0004;
    rhcrit = rc0005;
    rhmax = rc0006;
    i__1 = *kts;
    
#pragma unroll 4
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
	    r__1 = r__2 * r__2 * max(r__3,p1) * b2 * max(r__5,rc0019) / four *
		     (r__6 * r__6);
	    sgm[k] = (real)sqrt((real)(max(r__1,rc0020)));
	    qmq = qw[k] - qsl;
	    q1[k - 1] = qmq / sgm[k];
	    r__1 = q1[k - 1] * rr2;
	    cldfra_bl1__[k] = p5 * (one + (real)erf_(&r__1));
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
	pblh2 = max(ten,*pblh1);
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
	    zagl = zw[k] + p5 * dz[k];
	    t = th[k] * exner[k];
	    xl = (real)xl_blend__(m471_work,m471_n, &t);
	    qsat_tk__ = (real)qsat_blend__(m471_work,m471_n, &t, &p[k]);
/* Computing MAX */
/* Computing MIN */
	    r__2 = rhmax, r__3 = qw[k] / max(rc0021,qsat_tk__);
	    r__1 = min(r__2,r__3);
	    rh[k - 1] = max(r__1,rc0022);
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
	    qw_pert__ = qw[k] + qw[k] * p5 * pattern_spp_pbl1__[k] * (real) (*
		    spp_pbl__);
	    qmq = qw_pert__ - qsat_tk__;
/* Computing MAX */
	    r__1 = qsq[k];
	    r3sq = max(r__1,zero);
/* Computing MAX */
	    r__1 = tsq[k];
	    t3sq = max(r__1,zero);
	    c3sq = cov[k];
/* Computing MIN */
	    d__2 = abs(c3sq), d__3 = sqrt(t3sq * r3sq);
	    d__1 = min(d__2,d__3);
	    c3sq = d_sign(&d__1, &c3sq);
	    if (*closure < rc0023) {
		r3sq = r3sq;
	    } else if (*closure >= rc0023 && *closure < rc0024) {
/* Computing 2nd power */
		r__1 = bet[k - 1];
		r3sq += r__1 * r__1 * t3sq;
	    } else {
/* Computing 2nd power */
		r__1 = bet[k - 1];
		r3sq = r3sq + r__1 * r__1 * t3sq - two * bet[k - 1] * c3sq;
	    }
/* Computing MAX */
	    r__1 = rc0025, r__2 = (real)sqrt((real)((real) r3sq));
	    sgm[k] = max(r__1,r__2);
/* Computing MIN */
	    r__1 = sgm[k], r__2 = qw[k] * p5;
	    sgm[k] = min(r__1,r__2);
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = dz[k] - rc0026;
	    r__1 = one, r__2 = max(r__3,r__4) / rc0027;
	    wt = min(r__1,r__2);
	    sgm[k] += sgm[k] * p2 * wt;
	    sgmq = sgm[k];
	    sgmc = sgm[k];
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = zagl - (pblh2 + rc0010);
	    r__1 = one, r__2 = max(r__3,r__4) / rc0028;
	    wt = min(r__1,r__2);
	    clim = clim_pbl__ * (one - wt) + clim_trp__ * wt;
/* Computing MIN */
/* Computing MAX */
	    r__3 = rc0030, r__4 = p1 * pblh2;
	    r__1 = rc0029, r__2 = max(r__3,r__4);
	    zsl = min(r__1,r__2);
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = zagl - zsl;
	    r__1 = one, r__2 = max(r__3,r__4) / rc0029;
	    wt = min(r__1,r__2);
	    clim = clim_sfc__ * (one - wt) + clim * wt;
/* Computing MAX */
	    r__1 = sgmc, r__2 = qw[k] * clim;
	    sgmc = max(r__1,r__2);
	    sgmc = max(rc0025,sgmc);
	    if (qmq >= zero) {
/* Computing MAX */
		r__1 = rc0031 * qw[k];
		sgmc = max(r__1,sgmc);
	    }
	    q1[k - 1] = qmq / sgmc;
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = zagl - pblh2;
	    r__1 = one, r__2 = max(r__3,r__4) / rc0032;
	    wt2 = min(r__1,r__2);
	    if (qi[k] + qs[k] > rc0033 && zagl > pblh2) {
/* Computing MIN */
/* Computing MAX */
		r__5 = qi[k] + qs[k];
		r__3 = zero, r__4 = ten + (real)r_lg10(&r__5);
		r__1 = rhmax, r__2 = rhcrit + wt2 * rc0034 * max(r__3,r__4);
		rh_adj__ = min(r__1,r__2);
/* Computing MAX */
		r__1 = rh[k - 1];
		rh_adj__ = max(r__1,rh_adj__);
		q1_rh__ = -rc0035 + three * (rh_adj__ - rhcrit) / (one - 
			rhcrit);
/* Computing MAX */
		r__1 = q1_rh__, r__2 = q1[k - 1];
		q1[k - 1] = max(r__1,r__2);
	    }
	    if (qc[k] > rc0036 && zagl > pblh2) {
/* Computing MIN */
/* Computing MAX */
		r__3 = zero, r__4 = five + (real)r_lg10(&qc[k]);
		r__1 = rhmax, r__2 = rhcrit + wt2 * rc0037 * max(r__3,r__4);
		rh_adj__ = min(r__1,r__2);
/* Computing MAX */
		r__1 = rh[k - 1];
		rh_adj__ = max(r__1,rh_adj__);
		q1_rh__ = -rc0035 + three * (rh_adj__ - rhcrit) / (one - 
			rhcrit);
/* Computing MAX */
		r__1 = q1_rh__, r__2 = q1[k - 1];
		q1[k - 1] = max(r__1,r__2);
	    }
	    q1k = q1[k - 1];
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = (zagl - (*pblh1 - rc0026)) / rc0038;
	    r__1 = one, r__2 = max(r__3,r__4);
	    wt2 = min(r__1,r__2);
/* Computing MAX */
/* Computing MIN */
	    r__3 = one, r__4 = p5 + rc0039 * (real)atan((real)(rc0040 * (q1k 
		    + rc0041)));
	    r__1 = zero, r__2 = min(r__3,r__4);
	    cldfra_qsq0__ = max(r__1,r__2);
/* Computing MAX */
/* Computing MIN */
	    r__3 = one, r__4 = p5 + rc0042 * (real)atan((real)(rc0043 * (q1k 
		    + rc0044)));
	    r__1 = zero, r__2 = min(r__3,r__4);
	    cldfra_qsq1__ = max(r__1,r__2);
	    cldfra_qsq__ = cldfra_qsq0__ * (one - wt2) + cldfra_qsq1__ * wt2;
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = rc0045 * (real)tanh((real)((rh[k - 1] - 
		    rc0046) / rc0004)) + p5;
	    r__1 = one, r__2 = max(r__3,r__4);
	    /* unread private value */;
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = rc0044 * (real)tanh((real)((rh[k - 1] - 
		    rc0047) / rc0048)) + p5;
	    r__1 = one, r__2 = max(r__3,r__4);
	    /* unread private value */;
	    cldfra_rh__ = zero;
	    cldfra_bl1__[k] = max(cldfra_qsq__,cldfra_rh__);
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = zagl - (pblh2 + ten);
	    r__1 = one, r__2 = max(r__3,r__4) / rc0032;
	    wt = min(r__1,r__2);
	    qlim = qlim_pbl__ * (one - wt) + qlim_trp__ * wt;
/* Computing MIN */
/* Computing MAX */
	    r__3 = rc0030, r__4 = p1 * pblh2;
	    r__1 = rc0029, r__2 = max(r__3,r__4);
	    zsl = min(r__1,r__2);
/* Computing MIN */
/* Computing MAX */
	    r__3 = zero, r__4 = zagl - zsl;
	    r__1 = one, r__2 = max(r__3,r__4) / rc0038;
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
	    if (cldfra_bl1__[k] < rc0022) {
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
	    if (*xland - rc0049 >= zero) {
/* Computing MAX */
		r__1 = q1[k - 1], r__2 = -rc0050;
		q1k = max(r__1,r__2);
	    } else {
/* Computing MAX */
		r__1 = q1[k - 1], r__2 = -rc0014;
		q1k = max(r__1,r__2);
	    }
	    if (q1k >= one) {
		fng = one;
	    } else if (q1k >= -rc0051 && q1k < one) {
		fng = (real)exp((real)(-rc0052 * (q1k - one)));
	    } else if (q1k >= -rc0050 && q1k < -rc0051) {
		fng = three + (real)exp((real)(-rc0053 * (q1k + rc0051)));
	    } else {
/* Computing MIN */
		r__1 = rc0054 + (real)exp((real)(-rc0043 * (q1k + rc0050)));
		fng = min(r__1,rc0055);
	    }
/* Computing MIN */
	    r__1 = cldfra_bl1__[k];
	    cfmax = min(r__1,rc0056);
/* Computing MIN */
/* Computing MAX */
	    r__2 = rc0057, r__3 = p1 * pblh2;
	    r__1 = max(r__2,r__3);
	    zsl = min(r__1,hundred);
/* Computing MIN */
	    r__1 = zagl / zsl;
	    wt = min(r__1,one);
	    cfmax *= wt;
	    bb = b[k - 1] * t / th[k];
	    qww = one + p608 * qw[k];
	    alpha = p608 * th[k];
	    beta = th[k] / t * (xl / cp) - rc0058 * th[k];
	    vt[k] = qww - cfmax * beta * bb * fng - one;
	    vq[k] = alpha + cfmax * beta * a[k - 1] * fng - tv0;
/* L9013: */
	}
/* L9014: */
    }
    if (*bl_mynn_cloudpdf__ < 0) {
	i__1 = *kte - 1;
	
#pragma unroll 4
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

__device__ M471_CALL /* Subroutine */ int mynn_tendencies__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<integer> i__, M471Ptr<real> delt, M471Ptr<real> dz, M471Ptr<real> zw, M471Ptr<real> xland, M471Ptr<real> pblh, M471Ptr<real> rho, M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> th, M471Ptr<real> tk, M471Ptr<real> qv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<real> qs, M471Ptr<real> qnc, M471Ptr<real> qni, M471Ptr<real> psfc, M471Ptr<real> p, M471Ptr<real> exner, 
	M471Ptr<real> thl, M471Ptr<real> sqv, M471Ptr<real> sqc, M471Ptr<real> sqi, M471Ptr<real> sqs, M471Ptr<real> sqw, 
	M471Ptr<real> thl_tot1__, M471Ptr<real> qc_tot1__, M471Ptr<real> qi_tot1__, M471Ptr<real> qnwfa, M471Ptr<real> qnifa, M471Ptr<real> qnbca, M471Ptr<real> ozone, M471Ptr<real> ust, M471Ptr<real> flt, M471Ptr<real> flq, 
	M471Ptr<real> flqv, M471Ptr<real> flqc, M471Ptr<real> wspd, M471Ptr<real> uoce, M471Ptr<real> voce, M471Ptr<real> tsq,
	 M471Ptr<real> qsq, M471Ptr<real> cov, M471Ptr<real> tcd, M471Ptr<real> qcd, M471Ptr<real> dfm, M471Ptr<real> dfh, 
	M471Ptr<real> dfq, M471Ptr<real> du, M471Ptr<real> dv, M471Ptr<real> dth, M471Ptr<real> dqv, M471Ptr<real> dqc, M471Ptr<real> dqi, M471Ptr<real> dqs, M471Ptr<real> dqnc, M471Ptr<real> dqni, M471Ptr<real> dqnwfa, M471Ptr<real> dqnifa, 
	M471Ptr<real> dqnbca, M471Ptr<real> dozone, M471Ptr<real> diss_heat__, M471Ptr<real> s_aw1__, M471Ptr<real> s_awthl1__, M471Ptr<real> s_awqt1__, M471Ptr<real> s_awqv1__, M471Ptr<real> s_awqc1__, M471Ptr<real> s_awu1__, M471Ptr<real> s_awv1__, M471Ptr<real> s_awqnc1__, M471Ptr<real> s_awqni1__, M471Ptr<real> s_awqnwfa1__, M471Ptr<real> s_awqnifa1__, M471Ptr<real> s_awqnbca1__, M471Ptr<real> sd_aw1__, 
	M471Ptr<real> sd_awthl1__, M471Ptr<real> sd_awqt1__, M471Ptr<real> sd_awqv1__, M471Ptr<real> sd_awqc1__, M471Ptr<real> sd_awqi1__, M471Ptr<real> sd_awqnc1__, M471Ptr<real> sd_awqni1__, 
	M471Ptr<real> sd_awqnwfa1__, M471Ptr<real> sd_awqnifa1__, M471Ptr<real> sd_awu1__, M471Ptr<real> sd_awv1__, M471Ptr<real> sub_thl__, M471Ptr<real> sub_sqv__, M471Ptr<real> sub_u__, M471Ptr<real> sub_v__, M471Ptr<real> det_thl__, M471Ptr<real> det_sqv__, M471Ptr<real> det_sqc__, M471Ptr<real> det_u__, M471Ptr<real> det_v__, M471Ptr<logical> flag_qc__, M471Ptr<logical> flag_qi__, 
	M471Ptr<logical> flag_qnc__, M471Ptr<logical> flag_qni__, M471Ptr<logical> flag_qs__, M471Ptr<logical> flag_qnwfa__, M471Ptr<logical> flag_qnifa__, M471Ptr<logical> flag_qnbca__, M471Ptr<logical> flag_ozone__, M471Ptr<real> cldfra_bl1__, M471Ptr<integer> bl_mynn_cloudmix__, 
	M471Ptr<integer> bl_mynn_mixqt__, M471Ptr<integer> bl_mynn_edmf__, M471Ptr<integer> bl_mynn_edmf_mom__, M471Ptr<integer> bl_mynn_mixscalars__, M471Ptr<integer> bl_mynn_mixaerosols__, M471Ptr<integer> bl_mynn_mixnumcon__, M471Ptr<real> upwind) {
    /* Initialized data */

    real r_d__ = 287.f;
    real p608 = .6083624362945556640625f;
    real grav = 9.81000041961669921875f;
    real xlscp = 2837.232421875f;
    real xlvcp = 2488.80029296875f;
    real wfa_max__ = 7999999967232.f;
    real wfa_min__ = 1e6f;
    real ifa_max__ = 5e8f;
    real ifa_min__ = 0.f;
    real wfa_ht__ = 2e3f;
    real ifa_ht__ = 1e4f;
    real zero = 0.f;
    real one = 1.f;
    real p333 = .3333333432674407958984375f;
    real p5 = .5f;
    real rc0000 = 1.f;
    real rc0001 = 100.f;
    real rc0002 = 1e-6f;
    real rc0003 = 1e-4f;
    real rc0004 = 0.f;
    real rc0005 = 1.f;
    real rc0006 = .9f;
    real rc0007 = 1e-8f;
    real rc0008 = 1e-12f;
    real rc0009 = 1.5f;
    real rc0010 = 100.f;
    real rc0011 = 200.f;
    real rc0012 = 6e8f;
    real rc0013 = .99f;
    real rc0014 = 360.f;
    real rc0015 = 160.f;

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
	    thl_tot1_offset, qc_tot1_offset, qi_tot1_offset, thl_offset, 
	    sqw_offset, sqv_offset, sqc_offset, sqi_offset, sqs_offset, 
	    qnwfa_offset, qnifa_offset, qnbca_offset, ozone_offset, 
	    dfm_offset, dfh_offset, du_offset, dv_offset, dth_offset, 
	    dqv_offset, dqc_offset, dqi_offset, dqs_offset, dqni_offset, 
	    dqnc_offset, dqnwfa_offset, dqnifa_offset, dqnbca_offset, 
	    dozone_offset, i__1;
    real r__1, r__2, r__3, r__4;

    /* Builtin functions */
    

    /* Local variables */
    real aero_min__, aero_max__;
    integer kproblem;
    M471Ptr<real> a=m471_array<real>(m471_work,(304*(MYNN471_NZ-49)+15016),m471_n);
M471Ptr<real> b=m471_array<real>(m471_work,(308*(MYNN471_NZ-49)+15212),m471_n);
M471Ptr<real> c__=m471_array<real>(m471_work,(312*(MYNN471_NZ-49)+15408),m471_n);
M471Ptr<real> d__=m471_array<real>(m471_work,(316*(MYNN471_NZ-49)+15604),m471_n);
    integer k;
    real t, ustovwsp, wt;
M471Ptr<real> x=m471_array<real>(m471_work,(320*(MYNN471_NZ-49)+15800),m471_n);
    
    real condensate, th2, tk2, portion_qc__, portion_qi__, wsp, dzk, wsp2, qsat, onoff, 
	    dztop;
M471Ptr<real> dtz=m471_array<real>(m471_work,(324*(MYNN471_NZ-49)+15996),m471_n);
M471Ptr<real> qnc2=m471_array<real>(m471_work,(328*(MYNN471_NZ-49)+16192),m471_n);
M471Ptr<real> sqc2=m471_array<real>(m471_work,(332*(MYNN471_NZ-49)+16388),m471_n);
M471Ptr<real> qni2=m471_array<real>(m471_work,(336*(MYNN471_NZ-49)+16584),m471_n);
M471Ptr<real> sqi2=m471_array<real>(m471_work,(340*(MYNN471_NZ-49)+16780),m471_n);
M471Ptr<real> sqs2=m471_array<real>(m471_work,(344*(MYNN471_NZ-49)+16976),m471_n);
M471Ptr<real> sqv2=m471_array<real>(m471_work,(348*(MYNN471_NZ-49)+17172),m471_n);
M471Ptr<real> sqw2=m471_array<real>(m471_work,(352*(MYNN471_NZ-49)+17368),m471_n);
M471Ptr<real> delp=m471_array<real>(m471_work,(356*(MYNN471_NZ-49)+17564),m471_n);
M471Ptr<real> khdz=m471_array<real>(m471_work,(360*(MYNN471_NZ-49)+17760),m471_n);
M471Ptr<real> kmdz=m471_array<real>(m471_work,(364*(MYNN471_NZ-49)+17960),m471_n);
M471Ptr<real> rhoz=m471_array<real>(m471_work,(368*(MYNN471_NZ-49)+18160),m471_n);
M471Ptr<real> qnbca2=m471_array<real>(m471_work,(372*(MYNN471_NZ-49)+18360),m471_n);
M471Ptr<real> qnifa2=m471_array<real>(m471_work,(376*(MYNN471_NZ-49)+18556),m471_n);
M471Ptr<real> qnwfa2=m471_array<real>(m471_work,(380*(MYNN471_NZ-49)+18752),m471_n);
    integer ixarr1;
    real nc_min__, ni_min__, rhosfc, th_new__, nonloc, qvflux;
M471Ptr<real> dncont=m471_array<real>(m471_work,(384*(MYNN471_NZ-49)+18948),m471_n);
M471Ptr<real> rhoinv=m471_array<real>(m471_work,(388*(MYNN471_NZ-49)+19144),m471_n);
M471Ptr<real> upcont=m471_array<real>(m471_work,(392*(MYNN471_NZ-49)+19340),m471_n);
    
    logical problem;
    real wfa_max2__;
    

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
    qi_tot1_offset = *kts;
    qi_tot1__ -= qi_tot1_offset;
    qc_tot1_offset = *kts;
    qc_tot1__ -= qc_tot1_offset;
    thl_tot1_offset = *kts;
    thl_tot1__ -= thl_tot1_offset;
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
    /* unread private value */;
    if (*bl_mynn_edmf_mom__ == 0) {
	onoff = zero;
    } else {
	onoff = one;
    }
    if (*wspd <= rc0002) {
	ustovwsp = zero;
    } else {
	ustovwsp = *ust / *wspd;
    }
    rhosfc = *psfc / (r_d__ * (tk[*kts] + p608 * qv[*kts]));
    dtz[*kts - 1] = *delt / dz[*kts];
    rhoz[*kts - 1] = rho[*kts];
    rhoinv[*kts - 1] = one / rho[*kts];
    khdz[*kts - 1] = rhoz[*kts - 1] * dfh[*kts];
    kmdz[*kts - 1] = rhoz[*kts - 1] * dfm[*kts];
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	dtz[k - 1] = *delt / dz[k];
	rhoz[k - 1] = (rho[k] * dz[k - 1] + rho[k - 1] * dz[k]) / (dz[k - 1] 
		+ dz[k]);
/* Computing MAX */
	r__1 = rhoz[k - 1];
	rhoz[k - 1] = max(r__1,rc0003);
/* Computing MAX */
	r__1 = rho[k];
	rhoinv[k - 1] = one / max(r__1,rc0003);
	/* unread private value */;
	khdz[k - 1] = rhoz[k - 1] * dfh[k];
	kmdz[k - 1] = rhoz[k - 1] * dfm[k];
/* L9004: */
    }
/* L9005: */
    rhoz[*kte] = rhoz[*kte - 1];
    khdz[*kte] = rhoz[*kte] * dfh[*kte];
    kmdz[*kte] = rhoz[*kte] * dfm[*kte];
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	delp[k - 1] = rho[k] * grav * dz[k];
/* L9007: */
    }
/* L9008: */
    if (delp[*kts - 1] < p333 * delp[*kts]) {
	delp[*kts - 1] = p333 * delp[*kts];
    }
    if (*bl_mynn_edmf__ == 1) {
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = khdz[k - 1], r__2 = p5 * (s_aw1__[k] + sd_aw1__[k]);
	    khdz[k - 1] = max(r__1,r__2);
/* Computing MAX */
	    r__1 = khdz[k - 1], r__2 = -p5 * (s_aw1__[k] - s_aw1__[k + 1]) - 
		    p5 * (sd_aw1__[k] - sd_aw1__[k + 1]);
	    khdz[k - 1] = max(r__1,r__2);
/* Computing MAX */
	    r__1 = kmdz[k - 1], r__2 = p5 * (s_aw1__[k] + sd_aw1__[k]);
	    kmdz[k - 1] = max(r__1,r__2);
/* Computing MAX */
	    r__1 = kmdz[k - 1], r__2 = -p5 * (s_aw1__[k] - s_aw1__[k + 1]) - 
		    p5 * (sd_aw1__[k] - sd_aw1__[k + 1]);
	    kmdz[k - 1] = max(r__1,r__2);
/* L9010: */
	}
/* L9011: */
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	dth[*kts + ixarr1 - 1] = zero;
    }
    if (*bl_mynn_edmf__ > 1) {
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    upcont[k - 1] = onoff * (s_awu1__[k] - s_aw1__[k] * (u[k] * *
		    upwind + u[k - 1] * (one - *upwind)));
	    dncont[k - 1] = onoff * (sd_awu1__[k] - sd_aw1__[k] * (u[k] * *
		    upwind + u[k - 1] * (one - *upwind)));
/* L9013: */
	}
/* L9014: */
	upcont[*kte - 1] = zero;
	dncont[*kte - 1] = zero;
	k = *kts;
	a[0] = zero;
	b[0] = one + dtz[k - 1] * (kmdz[k] + rhosfc * *ust * ustovwsp) * 
		rhoinv[k - 1];
	c__[0] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1];
/* Computing 2nd power */
	r__1 = *ust;
	d__[0] = u[k] + dtz[k - 1] * *uoce * (r__1 * r__1) / *wspd * rho[k] - 
		dtz[k - 1] * (upcont[k] + dncont[k]);
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * kmdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (kmdz[k - 1] + kmdz[k]) * rhoinv[k 
		    - 1];
	    c__[k - 1] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1];
	    d__[k - 1] = u[k] - dtz[k - 1] * (upcont[k] - upcont[k - 1] + 
		    dncont[k] - dncont[k - 1]) + sub_u__[k] * *delt + det_u__[
		    k] * *delt;
/* L9016: */
	}
/* L9017: */
    } else {
	k = *kts;
	a[k - 1] = -dtz[k - 1] * kmdz[k - 1] * rhoinv[k - 1];
	b[k - 1] = one + dtz[k - 1] * (kmdz[k] + rhosfc * *ust * ustovwsp) * 
		rhoinv[k - 1] - p5 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 
		1] * onoff - p5 * dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1]
		 * onoff;
	c__[k - 1] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1] - p5 * dtz[k - 1] *
		 rhoinv[k - 1] * s_aw1__[k + 1] * onoff - p5 * dtz[k - 1] * 
		rhoinv[k - 1] * sd_aw1__[k + 1] * onoff;
/* Computing 2nd power */
	r__1 = *ust;
	d__[k - 1] = u[k] + dtz[k - 1] * *uoce * (r__1 * r__1) / *wspd - dtz[
		k - 1] * rhoinv[k - 1] * s_awu1__[k + 1] * onoff + dtz[k - 1] 
		* rhoinv[k - 1] * sd_awu1__[k + 1] * onoff + sub_u__[k] * *
		delt + det_u__[k] * *delt;
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * kmdz[k - 1] * rhoinv[k - 1] + p5 * dtz[k 
		    - 1] * rhoinv[k - 1] * s_aw1__[k] * onoff + p5 * dtz[k - 
		    1] * rhoinv[k - 1] * sd_aw1__[k] * onoff;
	    b[k - 1] = one + dtz[k - 1] * (kmdz[k - 1] + kmdz[k]) * rhoinv[k 
		    - 1] + p5 * dtz[k - 1] * rhoinv[k - 1] * (s_aw1__[k] - 
		    s_aw1__[k + 1]) * onoff + p5 * dtz[k - 1] * rhoinv[k - 1] 
		    * (sd_aw1__[k] - sd_aw1__[k + 1]) * onoff;
	    c__[k - 1] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1] - p5 * dtz[k - 
		    1] * rhoinv[k - 1] * s_aw1__[k + 1] * onoff - p5 * dtz[k 
		    - 1] * rhoinv[k - 1] * sd_aw1__[k + 1] * onoff;
	    d__[k - 1] = u[k] + dtz[k - 1] * rhoinv[k - 1] * (s_awu1__[k] - 
		    s_awu1__[k + 1]) * onoff - dtz[k - 1] * rhoinv[k - 1] * (
		    sd_awu1__[k] - sd_awu1__[k + 1]) * onoff + sub_u__[k] * *
		    delt + det_u__[k] * *delt;
/* L9019: */
	}
/* L9020: */
    }
    a[*kte - 1] = zero;
    b[*kte - 1] = one;
    c__[*kte - 1] = zero;
    d__[*kte - 1] = u[*kte];
    tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	du[k] = (x[k - 1] - u[k]) / *delt;
/* L9022: */
    }
/* L9023: */
    if (*bl_mynn_edmf__ > 1) {
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    upcont[k - 1] = onoff * (s_awv1__[k] - s_aw1__[k] * (v[k] * *
		    upwind + v[k - 1] * (one - *upwind)));
	    dncont[k - 1] = onoff * (sd_awv1__[k] - sd_aw1__[k] * (v[k] * *
		    upwind + v[k - 1] * (one - *upwind)));
/* L9025: */
	}
/* L9026: */
	upcont[*kte - 1] = rc0004;
	dncont[*kte - 1] = rc0004;
	k = *kts;
	a[0] = zero;
	b[0] = one + dtz[k - 1] * (kmdz[k] + rhosfc * *ust * ustovwsp) * 
		rhoinv[k - 1];
	c__[0] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1];
/* Computing 2nd power */
	r__1 = *ust;
	d__[0] = v[k] + dtz[k - 1] * *voce * (r__1 * r__1) / *wspd * rho[k] - 
		dtz[k - 1] * (upcont[k] + dncont[k]);
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * kmdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (kmdz[k - 1] + kmdz[k]) * rhoinv[k 
		    - 1];
	    c__[k - 1] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1];
	    d__[k - 1] = v[k] - dtz[k - 1] * (upcont[k] - upcont[k - 1] + 
		    dncont[k] - dncont[k - 1]) + sub_v__[k] * *delt + det_v__[
		    k] * *delt;
/* L9028: */
	}
/* L9029: */
    } else {
	k = *kts;
	a[k - 1] = -dtz[k - 1] * kmdz[k - 1] * rhoinv[k - 1];
/* Computing 2nd power */
	r__1 = *ust;
	b[k - 1] = one + dtz[k - 1] * (kmdz[k] + rhosfc * (r__1 * r__1) / *
		wspd) * rhoinv[k - 1] - p5 * dtz[k - 1] * rhoinv[k - 1] * 
		s_aw1__[k + 1] * onoff - p5 * dtz[k - 1] * rhoinv[k - 1] * 
		sd_aw1__[k + 1] * onoff;
	c__[k - 1] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1] - p5 * dtz[k - 1] *
		 rhoinv[k - 1] * s_aw1__[k + 1] * onoff - p5 * dtz[k - 1] * 
		rhoinv[k - 1] * sd_aw1__[k + 1] * onoff;
/* Computing 2nd power */
	r__1 = *ust;
	d__[k - 1] = v[k] + dtz[k - 1] * *voce * (r__1 * r__1) / *wspd - dtz[
		k - 1] * rhoinv[k - 1] * s_awv1__[k + 1] * onoff + dtz[k - 1] 
		* rhoinv[k - 1] * sd_awv1__[k + 1] * onoff + sub_v__[k] * *
		delt + det_v__[k] * *delt;
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * kmdz[k - 1] * rhoinv[k - 1] + p5 * dtz[k 
		    - 1] * rhoinv[k - 1] * s_aw1__[k] * onoff + p5 * dtz[k - 
		    1] * rhoinv[k - 1] * sd_aw1__[k] * onoff;
	    b[k - 1] = one + dtz[k - 1] * (kmdz[k - 1] + kmdz[k]) * rhoinv[k 
		    - 1] + p5 * dtz[k - 1] * rhoinv[k - 1] * (s_aw1__[k] - 
		    s_aw1__[k + 1]) * onoff + p5 * dtz[k - 1] * rhoinv[k - 1] 
		    * (sd_aw1__[k] - sd_aw1__[k + 1]) * onoff;
	    c__[k - 1] = -dtz[k - 1] * kmdz[k] * rhoinv[k - 1] - p5 * dtz[k - 
		    1] * rhoinv[k - 1] * s_aw1__[k + 1] * onoff - p5 * dtz[k 
		    - 1] * rhoinv[k - 1] * sd_aw1__[k + 1] * onoff;
	    d__[k - 1] = v[k] + dtz[k - 1] * rhoinv[k - 1] * (s_awv1__[k] - 
		    s_awv1__[k + 1]) * onoff - dtz[k - 1] * rhoinv[k - 1] * (
		    sd_awv1__[k] - sd_awv1__[k + 1]) * onoff + sub_v__[k] * *
		    delt + det_v__[k] * *delt;
/* L9031: */
	}
/* L9032: */
    }
    a[*kte - 1] = zero;
    b[*kte - 1] = one;
    c__[*kte - 1] = zero;
    d__[*kte - 1] = v[*kte];
    tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	dv[k] = (x[k - 1] - v[k]) / *delt;
/* L9034: */
    }
/* L9035: */
    if (*bl_mynn_edmf__ > 1) {
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    upcont[k - 1] = s_awthl1__[k] - s_aw1__[k] * (thl[k] * *upwind + 
		    thl[k - 1] * (one - *upwind));
	    dncont[k - 1] = sd_awthl1__[k] - sd_aw1__[k] * (thl[k] * *upwind 
		    + thl[k - 1] * (one - *upwind));
/* L9037: */
	}
/* L9038: */
	upcont[*kte - 1] = zero;
	dncont[*kte - 1] = zero;
	k = *kts;
	a[0] = zero;
	b[0] = one + dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	c__[0] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	d__[0] = thl[k] + dtz[k - 1] * rhosfc * *flt * rhoinv[k - 1] + tcd[k] 
		* *delt - dtz[k - 1] * (upcont[k] + dncont[k]) + diss_heat__[
		k] * *delt + sub_thl__[k] * *delt + det_thl__[k] * *delt;
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[k 
		    - 1];
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    d__[k - 1] = thl[k] - dtz[k - 1] * (upcont[k] - upcont[k - 1] + 
		    dncont[k] - dncont[k - 1]) + diss_heat__[k] * *delt + 
		    sub_thl__[k] * *delt + det_thl__[k] * *delt;
/* L9040: */
	}
/* L9041: */
    } else {
	k = *kts;
	a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	b[k - 1] = rc0005 + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k - 
		1] - p5 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - p5 * 
		dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[k - 1] *
		 rhoinv[k - 1] * s_aw1__[k + 1] - p5 * dtz[k - 1] * rhoinv[k 
		- 1] * sd_aw1__[k + 1];
	d__[k - 1] = thl[k] + dtz[k - 1] * rhosfc * *flt * rhoinv[k - 1] + 
		tcd[k] * *delt - dtz[k - 1] * rhoinv[k - 1] * s_awthl1__[k + 
		1] + dtz[k - 1] * rhoinv[k - 1] * sd_awthl1__[k + 1] + 
		diss_heat__[k] * *delt + sub_thl__[k] * *delt + det_thl__[k] *
		 *delt;
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + p5 * dtz[k 
		    - 1] * rhoinv[k - 1] * s_aw1__[k] + p5 * dtz[k - 1] * 
		    rhoinv[k - 1] * sd_aw1__[k];
	    b[k - 1] = rc0005 + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[
		    k - 1] + p5 * dtz[k - 1] * rhoinv[k - 1] * (s_aw1__[k] - 
		    s_aw1__[k + 1]) + p5 * dtz[k - 1] * rhoinv[k - 1] * (
		    sd_aw1__[k] - sd_aw1__[k + 1]);
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[k - 
		    1] * rhoinv[k - 1] * s_aw1__[k + 1] - p5 * dtz[k - 1] * 
		    rhoinv[k - 1] * sd_aw1__[k + 1];
	    d__[k - 1] = thl[k] + tcd[k] * *delt + dtz[k - 1] * rhoinv[k - 1] 
		    * (s_awthl1__[k] - s_awthl1__[k + 1]) - dtz[k - 1] * 
		    rhoinv[k - 1] * (sd_awthl1__[k] - sd_awthl1__[k + 1]) + 
		    diss_heat__[k] * *delt + sub_thl__[k] * *delt + det_thl__[
		    k] * *delt;
/* L9043: */
	}
/* L9044: */
    }
    a[*kte - 1] = zero;
    b[*kte - 1] = one;
    c__[*kte - 1] = zero;
    d__[*kte - 1] = thl[*kte];
    tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	thl[k] = x[k - 1];
/* L9046: */
    }
/* L9047: */
    if (*bl_mynn_mixqt__ > 0) {
	if (*bl_mynn_edmf__ > 1) {
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		upcont[k - 1] = s_awqt1__[k] - s_aw1__[k] * (sqw[k] * *upwind 
			+ sqw[k - 1] * (one - *upwind));
		dncont[k - 1] = sd_awqt1__[k] - sd_aw1__[k] * (sqw[k] * *
			upwind + sqw[k - 1] * (one - *upwind));
/* L9049: */
	    }
/* L9050: */
	    upcont[*kte - 1] = zero;
	    dncont[*kte - 1] = zero;
	    k = *kts;
	    a[0] = zero;
	    b[0] = one + dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    c__[0] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    d__[0] = sqw[k] + dtz[k - 1] * rhosfc * *flq * rhoinv[k - 1] + 
		    qcd[k] * *delt - dtz[k - 1] * (upcont[k] + dncont[k]) + 
		    sub_sqv__[k] * *delt + det_sqv__[k] * *delt;
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1];
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
		d__[k - 1] = sqw[k] - dtz[k - 1] * (upcont[k] - upcont[k - 1] 
			+ dncont[k] - dncont[k - 1]) + sub_sqv__[k] * *delt + 
			det_sqv__[k] * *delt;
/* L9052: */
	    }
/* L9053: */
	} else {
	    k = *kts;
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k 
		    - 1] - p5 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - 
		    p5 * dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[k - 
		    1] * rhoinv[k - 1] * s_aw1__[k + 1] - p5 * dtz[k - 1] * 
		    rhoinv[k - 1] * sd_aw1__[k + 1];
	    d__[k - 1] = sqw[k] + dtz[k - 1] * rhosfc * *flq * rhoinv[k - 1] 
		    + qcd[k] * *delt - dtz[k - 1] * rhoinv[k - 1] * s_awqt1__[
		    k + 1] + dtz[k - 1] * rhoinv[k - 1] * sd_awqt1__[k + 1];
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + p5 * 
			dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] + p5 * dtz[k 
			- 1] * rhoinv[k - 1] * sd_aw1__[k];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1] + p5 * dtz[k - 1] * rhoinv[k - 1] * (
			s_aw1__[k] - s_aw1__[k + 1]) + p5 * dtz[k - 1] * 
			rhoinv[k - 1] * (sd_aw1__[k] - sd_aw1__[k + 1]);
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[
			k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - p5 * dtz[k 
			- 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
		d__[k - 1] = sqw[k] + qcd[k] * *delt + dtz[k - 1] * rhoinv[k 
			- 1] * (s_awqt1__[k] - s_awqt1__[k + 1]) - dtz[k - 1] 
			* rhoinv[k - 1] * (sd_awqt1__[k] - sd_awqt1__[k + 1]);
/* L9055: */
	    }
/* L9056: */
	}
	a[*kte - 1] = zero;
	b[*kte - 1] = one;
	c__[*kte - 1] = zero;
	d__[*kte - 1] = sqw[*kte];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, sqw2);
    } else {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    if (*bl_mynn_mixqt__ > 0) /* m480 unread unless */ { sqw2[ixarr1 - 1] = sqw[*kts + ixarr1 - 1]; }
	}
    }
    if (*bl_mynn_mixqt__ == 0) {
	if (*bl_mynn_cloudmix__ > 0 && *flag_qc__) {
	    if (*bl_mynn_edmf__ > 1) {
		i__1 = *kte - 1;
		
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		    upcont[k - 1] = s_awqc1__[k] - s_aw1__[k] * (sqc[k] * *
			    upwind + sqc[k - 1] * (one - *upwind));
		    dncont[k - 1] = sd_awqc1__[k] - sd_aw1__[k] * (sqc[k] * *
			    upwind + sqc[k - 1] * (one - *upwind));
/* L9058: */
		}
/* L9059: */
		upcont[*kte - 1] = zero;
		dncont[*kte - 1] = zero;
		k = *kts;
		a[0] = zero;
		b[0] = one + dtz[k - 1] * khdz[k] * rhoinv[k - 1];
		c__[0] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
		d__[0] = sqc[k] + dtz[k - 1] * rhosfc * *flqc * rhoinv[k - 1] 
			+ qcd[k] * *delt - dtz[k - 1] * (upcont[k] + dncont[k]
			) + det_sqc__[k] * *delt;
		i__1 = *kte - 1;
		
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
		    b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			    rhoinv[k - 1];
		    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
		    d__[k - 1] = sqc[k] - dtz[k - 1] * (upcont[k] - upcont[k 
			    - 1] + dncont[k] - dncont[k - 1]) + det_sqc__[k] *
			     *delt;
/* L9061: */
		}
/* L9062: */
	    } else {
		k = *kts;
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
		b[k - 1] = one + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * 
			rhoinv[k - 1] - p5 * dtz[k - 1] * rhoinv[k - 1] * 
			s_aw1__[k + 1] - p5 * dtz[k - 1] * rhoinv[k - 1] * 
			sd_aw1__[k + 1];
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[
			k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - p5 * dtz[k 
			- 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
		d__[k - 1] = sqc[k] + dtz[k - 1] * rhosfc * *flqc * rhoinv[k 
			- 1] + qcd[k] * *delt - dtz[k - 1] * rhoinv[k - 1] * 
			s_awqc1__[k + 1] + dtz[k - 1] * rhoinv[k - 1] * 
			sd_awqc1__[k + 1] + det_sqc__[k] * *delt;
		i__1 = *kte - 1;
		
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + p5 
			    * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] + p5 * 
			    dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k];
		    b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			    rhoinv[k - 1] + p5 * dtz[k - 1] * rhoinv[k - 1] * 
			    (s_aw1__[k] - s_aw1__[k + 1]) + p5 * dtz[k - 1] * 
			    rhoinv[k - 1] * (sd_aw1__[k] - sd_aw1__[k + 1]);
		    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * 
			    dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - p5 *
			     dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
		    d__[k - 1] = sqc[k] + qcd[k] * *delt + dtz[k - 1] * 
			    rhoinv[k - 1] * (s_awqc1__[k] - s_awqc1__[k + 1]) 
			    - dtz[k - 1] * rhoinv[k - 1] * (sd_awqc1__[k] - 
			    sd_awqc1__[k + 1]) + det_sqc__[k] * *delt;
/* L9064: */
		}
/* L9065: */
	    }
	    a[*kte - 1] = zero;
	    b[*kte - 1] = one;
	    c__[*kte - 1] = zero;
	    d__[*kte - 1] = sqc[*kte];
	    tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, sqc2);
	} else {
	    
#pragma unroll 4
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
	    r__3 = rc0006 * sqv[*kts] - rc0007;
	    r__1 = qvflux, r__2 = min(r__3,zero) / dtz[*kts - 1];
	    qvflux = max(r__1,r__2);
	}
	if (*bl_mynn_edmf__ > 1) {
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		upcont[k - 1] = s_awqv1__[k] - s_aw1__[k] * (sqv[k] * *upwind 
			+ sqv[k - 1] * (one - *upwind));
		dncont[k - 1] = sd_awqv1__[k] - sd_aw1__[k] * (sqv[k] * *
			upwind + sqv[k - 1] * (one - *upwind));
/* L9067: */
	    }
/* L9068: */
	    upcont[*kte - 1] = zero;
	    dncont[*kte - 1] = zero;
	    k = *kts;
	    a[0] = zero;
	    b[0] = one + dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    c__[0] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    d__[0] = sqv[k] + dtz[k - 1] * rhosfc * qvflux * rhoinv[k - 1] + 
		    qcd[k] * *delt - dtz[k - 1] * (upcont[k] + dncont[k]) + 
		    sub_sqv__[k] * *delt + det_sqv__[k] * *delt;
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1];
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
		d__[k - 1] = sqv[k] - dtz[k - 1] * (upcont[k] - upcont[k - 1] 
			+ dncont[k] - dncont[k - 1]) + sub_sqv__[k] * *delt + 
			det_sqv__[k] * *delt;
/* L9070: */
	    }
/* L9071: */
	} else {
	    k = *kts;
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k 
		    - 1] - p5 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - 
		    p5 * dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[k - 
		    1] * rhoinv[k - 1] * s_aw1__[k + 1] - p5 * dtz[k - 1] * 
		    rhoinv[k - 1] * sd_aw1__[k + 1];
	    d__[k - 1] = sqv[k] + dtz[k - 1] * rhosfc * qvflux * rhoinv[k - 1]
		     + qcd[k] * *delt - dtz[k - 1] * rhoinv[k - 1] * 
		    s_awqv1__[k + 1] + dtz[k - 1] * rhoinv[k - 1] * 
		    sd_awqv1__[k + 1] + sub_sqv__[k] * *delt + det_sqv__[k] * 
		    *delt;
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + p5 * 
			dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] + p5 * dtz[k 
			- 1] * rhoinv[k - 1] * sd_aw1__[k];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1] + p5 * dtz[k - 1] * rhoinv[k - 1] * (
			s_aw1__[k] - s_aw1__[k + 1]) + p5 * dtz[k - 1] * 
			rhoinv[k - 1] * (sd_aw1__[k] - sd_aw1__[k + 1]);
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[
			k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - p5 * dtz[k 
			- 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
		d__[k - 1] = sqv[k] + qcd[k] * *delt + dtz[k - 1] * rhoinv[k 
			- 1] * (s_awqv1__[k] - s_awqv1__[k + 1]) - dtz[k - 1] 
			* rhoinv[k - 1] * (sd_awqv1__[k] - sd_awqv1__[k + 1]) 
			+ sub_sqv__[k] * *delt + det_sqv__[k] * *delt;
/* L9073: */
	    }
/* L9074: */
	}
	a[*kte - 1] = zero;
	b[*kte - 1] = one;
	c__[*kte - 1] = zero;
	d__[*kte - 1] = sqv[*kte];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, sqv2);
    } else {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    sqv2[ixarr1 - 1] = sqv[*kts + ixarr1 - 1];
	}
    }
    if (*bl_mynn_cloudmix__ > 0 && *flag_qi__) {
	if (*bl_mynn_edmf__ > 1) {
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		upcont[k - 1] = zero;
		dncont[k - 1] = sd_awqi1__[k] - sd_aw1__[k] * (sqi[k] * *
			upwind + sqi[k - 1] * (one - *upwind));
/* L9076: */
	    }
/* L9077: */
	    upcont[*kte - 1] = zero;
	    dncont[*kte - 1] = zero;
	    k = *kts;
	    a[0] = zero;
	    b[0] = one + dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    c__[0] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    d__[0] = sqi[k] - dtz[k - 1] * (upcont[k] + dncont[k]);
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1];
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
		d__[k - 1] = sqi[k] - dtz[k - 1] * (upcont[k] - upcont[k - 1] 
			+ dncont[k] - dncont[k - 1]);
/* L9079: */
	    }
/* L9080: */
	} else {
	    k = *kts;
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k 
		    - 1] - p5 * dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[k - 
		    1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	    d__[k - 1] = sqi[k] + dtz[k - 1] * rhoinv[k - 1] * sd_awqi1__[k + 
		    1];
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + p5 * 
			dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1] + p5 * dtz[k - 1] * rhoinv[k - 1] * (
			sd_aw1__[k] - sd_aw1__[k + 1]);
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[
			k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
		d__[k - 1] = sqi[k] - dtz[k - 1] * rhoinv[k - 1] * (
			sd_awqi1__[k] - sd_awqi1__[k + 1]);
/* L9082: */
	    }
/* L9083: */
	}
	a[*kte - 1] = zero;
	b[*kte - 1] = one;
	c__[*kte - 1] = zero;
	d__[*kte - 1] = sqi[*kte];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, sqi2);
    } else {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    sqi2[ixarr1 - 1] = sqi[*kts + ixarr1 - 1];
	}
    }
    if (FALSE_) {
	k = *kts;
	a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	b[k - 1] = one + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k - 1];
	c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	d__[k - 1] = sqs[k];
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[k 
		    - 1];
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    d__[k - 1] = sqs[k];
/* L9085: */
	}
/* L9086: */
	a[*kte - 1] = zero;
	b[*kte - 1] = one;
	c__[*kte - 1] = zero;
	d__[*kte - 1] = sqs[*kte];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, sqs2);
    } else {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    sqs2[ixarr1 - 1] = sqs[*kts + ixarr1 - 1];
	}
    }
    if (*bl_mynn_cloudmix__ > 0 && *flag_qni__ && *bl_mynn_mixnumcon__ > 0) {
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = qni[k];
	    qni2[k - 1] = max(r__1,zero);
	    if (sqi[k] > rc0008) {
/* Computing MAX */
		r__1 = qni2[k - 1];
		qni2[k - 1] = max(r__1,ni_min__);
	    }
/* L9088: */
	}
/* L9089: */
	if (*bl_mynn_edmf__ > 1) {
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		upcont[k - 1] = s_awqni1__[k] - s_aw1__[k] * (qni2[k - 1] * *
			upwind + qni2[k - 2] * (one - *upwind));
		dncont[k - 1] = sd_awqni1__[k] - sd_aw1__[k] * (qni2[k - 1] * 
			*upwind + qni2[k - 2] * (one - *upwind));
/* L9091: */
	    }
/* L9092: */
	    upcont[*kte - 1] = zero;
	    dncont[*kte - 1] = zero;
	    k = *kts;
	    a[0] = zero;
	    b[0] = one + dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    c__[0] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    d__[0] = qni2[k - 1] - dtz[k - 1] * (upcont[k] + dncont[k]);
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1];
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
		d__[k - 1] = qni2[k - 1] - dtz[k - 1] * (upcont[k] - upcont[k 
			- 1] + dncont[k] - dncont[k - 1]);
/* L9094: */
	    }
/* L9095: */
	} else {
	    k = *kts;
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k 
		    - 1] - p5 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - 
		    p5 * dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[k - 
		    1] * rhoinv[k - 1] * s_aw1__[k + 1] - p5 * dtz[k - 1] * 
		    rhoinv[k - 1] * sd_aw1__[k + 1];
	    d__[k - 1] = qni2[k - 1] - dtz[k - 1] * rhoinv[k - 1] * 
		    s_awqni1__[k + 1] + dtz[k - 1] * rhoinv[k - 1] * 
		    sd_awqni1__[k + 1];
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + p5 * 
			dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] + p5 * dtz[k 
			- 1] * rhoinv[k - 1] * sd_aw1__[k];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1] + p5 * dtz[k - 1] * rhoinv[k - 1] * (
			s_aw1__[k] - s_aw1__[k + 1]) + p5 * dtz[k - 1] * 
			rhoinv[k - 1] * (sd_aw1__[k] - sd_aw1__[k + 1]);
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[
			k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - p5 * dtz[k 
			- 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
		d__[k - 1] = qni2[k - 1] + dtz[k - 1] * rhoinv[k - 1] * (
			s_awqni1__[k] - s_awqni1__[k + 1]) - dtz[k - 1] * 
			rhoinv[k - 1] * (sd_awqni1__[k] - sd_awqni1__[k + 1]);
/* L9097: */
	    }
/* L9098: */
	}
	a[*kte - 1] = zero;
	b[*kte - 1] = one;
	c__[*kte - 1] = zero;
	d__[*kte - 1] = qni2[*kte - 1];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    qni2[k - 1] = x[k - 1];
	    if (sqi2[k - 1] > rc0008) {
/* Computing MAX */
		r__1 = qni2[k - 1];
		qni2[k - 1] = max(r__1,ni_min__);
	    }
/* L9100: */
	}
/* L9101: */
    } else {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    qni2[ixarr1 - 1] = qni[*kts + ixarr1 - 1];
	}
    }
    if (*bl_mynn_cloudmix__ > 0 && *flag_qnc__ && *bl_mynn_mixnumcon__ > 0) {
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = qnc[k];
	    qnc2[k - 1] = max(r__1,zero);
	    if (sqc[k] > rc0008) {
/* Computing MAX */
		r__1 = qnc2[k - 1];
		qnc2[k - 1] = max(r__1,nc_min__);
	    }
/* L9103: */
	}
/* L9104: */
	if (*bl_mynn_edmf__ > 1) {
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		upcont[k - 1] = s_awqnc1__[k] - s_aw1__[k] * (qnc2[k - 1] * *
			upwind + qnc2[k - 2] * (one - *upwind));
		dncont[k - 1] = sd_awqnc1__[k] - sd_aw1__[k] * (qnc2[k - 1] * 
			*upwind + qnc2[k - 2] * (one - *upwind));
/* L9106: */
	    }
/* L9107: */
	    upcont[*kte - 1] = zero;
	    dncont[*kte - 1] = zero;
	    k = *kts;
	    a[0] = zero;
	    b[0] = one + dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    c__[0] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    d__[0] = qnc2[k - 1] - dtz[k - 1] * (upcont[k] + dncont[k]);
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1];
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
		d__[k - 1] = qnc2[k - 1] - dtz[k - 1] * (upcont[k] - upcont[k 
			- 1] + dncont[k] - dncont[k - 1]);
/* L9109: */
	    }
/* L9110: */
	} else {
	    k = *kts;
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k 
		    - 1] - p5 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - 
		    p5 * dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[k - 
		    1] * rhoinv[k - 1] * s_aw1__[k + 1] - p5 * dtz[k - 1] * 
		    rhoinv[k - 1] * sd_aw1__[k + 1];
	    d__[k - 1] = qnc2[k - 1] - dtz[k - 1] * rhoinv[k - 1] * 
		    s_awqnc1__[k + 1] + dtz[k - 1] * rhoinv[k - 1] * 
		    sd_awqnc1__[k + 1];
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + p5 * 
			dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] + p5 * dtz[k 
			- 1] * rhoinv[k - 1] * sd_aw1__[k];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1] + p5 * dtz[k - 1] * rhoinv[k - 1] * (
			s_aw1__[k] - s_aw1__[k + 1]) + p5 * dtz[k - 1] * 
			rhoinv[k - 1] * (sd_aw1__[k] - sd_aw1__[k + 1]);
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[
			k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - p5 * dtz[k 
			- 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
		d__[k - 1] = qnc2[k - 1] + dtz[k - 1] * rhoinv[k - 1] * (
			s_awqnc1__[k] - s_awqnc1__[k + 1]) - dtz[k - 1] * 
			rhoinv[k - 1] * (sd_awqnc1__[k] - sd_awqnc1__[k + 1]);
/* L9112: */
	    }
/* L9113: */
	}
	a[*kte - 1] = zero;
	b[*kte - 1] = one;
	c__[*kte - 1] = zero;
	d__[*kte - 1] = qnc2[*kte - 1];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    qnc2[k - 1] = x[k - 1];
	    if (sqc2[k - 1] > rc0008) {
/* Computing MAX */
		r__1 = qnc2[k - 1];
		qnc2[k - 1] = max(r__1,nc_min__);
	    }
/* L9115: */
	}
/* L9116: */
    } else {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    qnc2[ixarr1 - 1] = qnc[*kts + ixarr1 - 1];
	}
    }
    if (*flag_qnwfa__ && *bl_mynn_mixaerosols__ > 0) {
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = qnwfa[k];
	    qnwfa2[k - 1] = max(r__1,zero);
/* L9118: */
	}
/* L9119: */
	if (*bl_mynn_edmf__ > 1) {
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		upcont[k - 1] = s_awqnwfa1__[k] - s_aw1__[k] * (qnwfa2[k - 1] 
			* *upwind + qnwfa2[k - 2] * (one - *upwind));
		dncont[k - 1] = sd_awqnwfa1__[k] - sd_aw1__[k] * (qnwfa2[k - 
			1] * *upwind + qnwfa2[k - 2] * (one - *upwind));
/* L9121: */
	    }
/* L9122: */
	    upcont[*kte - 1] = zero;
	    dncont[*kte - 1] = zero;
	    k = *kts;
	    a[0] = zero;
	    b[0] = one + dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    c__[0] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    d__[0] = qnwfa2[k - 1] - dtz[k - 1] * (upcont[k] + dncont[k]);
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1];
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
		d__[k - 1] = qnwfa2[k - 1] - dtz[k - 1] * (upcont[k] - upcont[
			k - 1] + dncont[k] - dncont[k - 1]);
/* L9124: */
	    }
/* L9125: */
	} else {
	    k = *kts;
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k 
		    - 1] - p5 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - 
		    p5 * dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[k - 
		    1] * rhoinv[k - 1] * s_aw1__[k + 1] - p5 * dtz[k - 1] * 
		    rhoinv[k - 1] * sd_aw1__[k + 1];
	    d__[k - 1] = qnwfa2[k - 1] - dtz[k - 1] * rhoinv[k - 1] * 
		    s_awqnwfa1__[k + 1] + dtz[k - 1] * rhoinv[k - 1] * 
		    sd_awqnwfa1__[k + 1];
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + p5 * 
			dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] + p5 * dtz[k 
			- 1] * rhoinv[k - 1] * sd_aw1__[k];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1] + p5 * dtz[k - 1] * rhoinv[k - 1] * (
			s_aw1__[k] - s_aw1__[k + 1]) + p5 * dtz[k - 1] * 
			rhoinv[k - 1] * (sd_aw1__[k] - sd_aw1__[k + 1]);
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[
			k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - p5 * dtz[k 
			- 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
		d__[k - 1] = qnwfa2[k - 1] + dtz[k - 1] * rhoinv[k - 1] * (
			s_awqnwfa1__[k] - s_awqnwfa1__[k + 1]) - dtz[k - 1] * 
			rhoinv[k - 1] * (sd_awqnwfa1__[k] - sd_awqnwfa1__[k + 
			1]);
/* L9127: */
	    }
/* L9128: */
	}
	a[*kte - 1] = zero;
	b[*kte - 1] = one;
	c__[*kte - 1] = zero;
	d__[*kte - 1] = qnwfa2[*kte - 1];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = x[k - 1];
	    qnwfa2[k - 1] = max(r__1,zero);
/* L9130: */
	}
/* L9131: */
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    if (*xland - rc0009 >= 0.f) {
/* Computing MIN */
/* Computing MAX */
		r__3 = zero, r__4 = zw[k] - rc0010;
		r__1 = one, r__2 = max(r__3,r__4) / max(rc0011,*pblh);
		wt = min(r__1,r__2);
		wfa_max2__ = (one - wt) * rc0012 + wt * wfa_max__;
	    } else {
		wfa_max2__ = wfa_max__;
	    }
	    aero_min__ = wfa_min__ * (real)exp((real)(-zw[k] / wfa_ht__));
	    aero_max__ = wfa_max2__ * (real)exp((real)(-zw[k] / wfa_ht__));
/* Computing MIN */
/* Computing MAX */
	    r__2 = aero_min__, r__3 = qnwfa2[k - 1];
	    r__1 = max(r__2,r__3);
	    qnwfa2[k - 1] = min(r__1,aero_max__);
/* L9133: */
	}
/* L9134: */
    } else {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    qnwfa2[ixarr1 - 1] = qnwfa[*kts + ixarr1 - 1];
	}
    }
    if (*flag_qnifa__ && *bl_mynn_mixaerosols__ > 0) {
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = qnifa[k];
	    qnifa2[k - 1] = max(r__1,zero);
/* L9136: */
	}
/* L9137: */
	if (*bl_mynn_edmf__ > 1) {
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		upcont[k - 1] = s_awqnifa1__[k] - s_aw1__[k] * (qnifa2[k - 1] 
			* *upwind + qnifa2[k - 2] * (one - *upwind));
		dncont[k - 1] = sd_awqnifa1__[k] - sd_aw1__[k] * (qnifa2[k - 
			1] * *upwind + qnifa2[k - 2] * (one - *upwind));
/* L9139: */
	    }
/* L9140: */
	    upcont[*kte - 1] = zero;
	    dncont[*kte - 1] = zero;
	    k = *kts;
	    a[0] = zero;
	    b[0] = one + dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    c__[0] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    d__[0] = qnifa2[k - 1] - dtz[k - 1] * (upcont[k] + dncont[k]);
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1];
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
		d__[k - 1] = qnifa2[k - 1] - dtz[k - 1] * (upcont[k] - upcont[
			k - 1] + dncont[k] - dncont[k - 1]);
/* L9142: */
	    }
/* L9143: */
	} else {
	    k = *kts;
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k 
		    - 1] - p5 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - 
		    p5 * dtz[k - 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[k - 
		    1] * rhoinv[k - 1] * s_aw1__[k + 1] - p5 * dtz[k - 1] * 
		    rhoinv[k - 1] * sd_aw1__[k + 1];
	    d__[k - 1] = qnifa2[k - 1] - dtz[k - 1] * rhoinv[k - 1] * 
		    s_awqnifa1__[k + 1] + dtz[k - 1] * rhoinv[k - 1] * 
		    sd_awqnifa1__[k + 1];
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + p5 * 
			dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] + p5 * dtz[k 
			- 1] * rhoinv[k - 1] * sd_aw1__[k];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1] + p5 * dtz[k - 1] * rhoinv[k - 1] * (
			s_aw1__[k] - s_aw1__[k + 1]) + p5 * dtz[k - 1] * 
			rhoinv[k - 1] * (sd_aw1__[k] - sd_aw1__[k + 1]);
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[
			k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] - p5 * dtz[k 
			- 1] * rhoinv[k - 1] * sd_aw1__[k + 1];
		d__[k - 1] = qnifa2[k - 1] + dtz[k - 1] * rhoinv[k - 1] * (
			s_awqnifa1__[k] - s_awqnifa1__[k + 1]) - dtz[k - 1] * 
			rhoinv[k - 1] * (sd_awqnifa1__[k] - sd_awqnifa1__[k + 
			1]);
/* L9145: */
	    }
/* L9146: */
	}
	a[*kte - 1] = zero;
	b[*kte - 1] = one;
	c__[*kte - 1] = zero;
	d__[*kte - 1] = qnifa2[*kte - 1];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = x[k - 1];
	    qnifa2[k - 1] = max(r__1,zero);
/* L9148: */
	}
/* L9149: */
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    aero_min__ = ifa_min__ * (real)exp((real)(-zw[k] / ifa_ht__));
	    aero_max__ = ifa_max__ * (real)exp((real)(-zw[k] / ifa_ht__));
/* Computing MIN */
/* Computing MAX */
	    r__2 = aero_min__, r__3 = qnifa2[k - 1];
	    r__1 = max(r__2,r__3);
	    qnifa2[k - 1] = min(r__1,aero_max__);
/* L9151: */
	}
/* L9152: */
    } else {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    qnifa2[ixarr1 - 1] = qnifa[*kts + ixarr1 - 1];
	}
    }
    if (*flag_qnbca__ && *bl_mynn_mixaerosols__ > 0) {
	if (*bl_mynn_edmf__ > 1) {
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		upcont[k - 1] = s_awqnbca1__[k] - s_aw1__[k] * (qnbca[k] * *
			upwind + qnbca[k - 1] * (one - *upwind));
		dncont[k - 1] = zero;
/* L9154: */
	    }
/* L9155: */
	    upcont[*kte - 1] = zero;
	    dncont[*kte - 1] = zero;
	    k = *kts;
	    a[0] = zero;
	    b[0] = one + dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    c__[0] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    d__[0] = qnbca[k] - dtz[k - 1] * (upcont[k] + dncont[k]);
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1];
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
		d__[k - 1] = qnbca[k] - dtz[k - 1] * (upcont[k] - upcont[k - 
			1] + dncont[k] - dncont[k - 1]);
/* L9157: */
	    }
/* L9158: */
	} else {
	    k = *kts;
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[k 
		    - 1] - p5 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] * 
		    nonloc;
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[k - 
		    1] * rhoinv[k - 1] * s_aw1__[k + 1] * nonloc;
	    d__[k - 1] = qnbca[k] - dtz[k - 1] * rhoinv[k - 1] * s_awqnbca1__[
		    k + 1] * nonloc;
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + p5 * 
			dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k] * nonloc;
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1] + p5 * dtz[k - 1] * rhoinv[k - 1] * (
			s_aw1__[k] - s_aw1__[k + 1]) * nonloc;
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[
			k - 1] * rhoinv[k - 1] * s_aw1__[k + 1] * nonloc;
		d__[k - 1] = qnbca[k] + dtz[k - 1] * rhoinv[k - 1] * (
			s_awqnbca1__[k] - s_awqnbca1__[k + 1]) * nonloc;
/* L9160: */
	    }
/* L9161: */
	}
	a[*kte - 1] = zero;
	b[*kte - 1] = one;
	c__[*kte - 1] = zero;
	d__[*kte - 1] = qnbca[*kte];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = zero, r__2 = x[k - 1];
	    qnbca2[k - 1] = max(r__1,r__2);
/* L9163: */
	}
/* L9164: */
    } else {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    qnbca2[ixarr1 - 1] = qnbca[*kts + ixarr1 - 1];
	}
    }
    if (*flag_ozone__) {
	k = *kts;
	a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	b[k - 1] = one + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k - 1];
	c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	d__[k - 1] = ozone[k];
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * rhoinv[k 
		    - 1];
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    d__[k - 1] = ozone[k];
/* L9166: */
	}
/* L9167: */
	a[*kte - 1] = zero;
	b[*kte - 1] = one;
	c__[*kte - 1] = zero;
	d__[*kte - 1] = ozone[*kte];
	tridiag2_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = x[k - 1];
	    dozone[k] = (max(r__1,zero) - ozone[k]) / *delt;
/* L9169: */
	}
/* L9170: */
    } else {
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
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
/* L9172: */
	}
/* L9173: */
    }
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	dqv[k] = (sqv2[k - 1] - sqv[k]) / *delt;
/* L9175: */
    }
/* L9176: */
    if (*bl_mynn_cloudmix__ > 0) {
	if (*flag_qc__) {
	    i__1 = *kte;
	    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
		dqc[k] = (sqc2[k - 1] - sqc[k]) / *delt;
/* L9178: */
	    }
/* L9179: */
	} else {
	    i__1 = *kte;
	    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
		dqc[k] = rc0004;
/* L9181: */
	    }
/* L9182: */
	}
	if (*flag_qnc__ && *bl_mynn_mixnumcon__ > 0) {
	    i__1 = *kte;
	    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
		dqnc[k] = (qnc2[k - 1] - qnc[k]) / *delt;
/* L9184: */
	    }
/* L9185: */
	} else {
	    i__1 = *kte;
	    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
		dqnc[k] = rc0004;
/* L9187: */
	    }
/* L9188: */
	}
	if (*flag_qi__) {
	    i__1 = *kte;
	    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
		dqi[k] = (sqi2[k - 1] - sqi[k]) / *delt;
/* L9190: */
	    }
/* L9191: */
	} else {
	    i__1 = *kte;
	    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
		dqi[k] = rc0004;
/* L9193: */
	    }
/* L9194: */
	}
	if (FALSE_) {
	    i__1 = *kte;
	    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
		dqs[k] = (sqs2[k - 1] - sqs[k]) / *delt;
/* L9196: */
	    }
/* L9197: */
	} else {
	    i__1 = *kte;
	    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
		dqs[k] = rc0004;
/* L9199: */
	    }
/* L9200: */
	}
	if (*flag_qni__ && *bl_mynn_mixnumcon__ > 0) {
	    i__1 = *kte;
	    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
		dqni[k] = (qni2[k - 1] - qni[k]) / *delt;
/* L9202: */
	    }
/* L9203: */
	} else {
	    i__1 = *kte;
	    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
		dqni[k] = rc0004;
/* L9205: */
	    }
/* L9206: */
	}
    } else {
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    dqc[k] = rc0004;
	    dqnc[k] = rc0004;
	    dqi[k] = rc0004;
	    dqni[k] = rc0004;
	    dqs[k] = rc0004;
/* L9208: */
	}
/* L9209: */
    }
    moisture_check__(m471_work,m471_n, kte, delt, delp, &exner[exner_offset], sqv2, sqc2, sqi2, 
	    sqs2, &thl[thl_offset], &dqv[dqv_offset], &dqc[dqc_offset], &dqi[
	    dqi_offset], &dqs[dqs_offset], &dth[dth_offset]);
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	if (dozone[k] * *delt + ozone[k] < rc0004) {
	    dozone[k] = -ozone[k] * rc0013 / *delt;
	}
/* L9211: */
    }
/* L9212: */
    if (*flag_qi__) {
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    dth[k] = (thl[k] + xlvcp / exner[k] * sqc[k] + xlscp / exner[k] * 
		    sqi[k] - th[k]) / *delt;
/* L9214: */
	}
/* L9215: */
    } else {
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    dth[k] = (thl[k] + xlvcp / exner[k] * sqc2[k - 1] - th[k]) / *
		    delt;
/* L9217: */
	}
/* L9218: */
    }
    if (*flag_qnwfa__ && *flag_qnifa__ && *bl_mynn_mixaerosols__ > 0) {
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    dqnwfa[k] = (qnwfa2[k - 1] - qnwfa[k]) / *delt;
	    dqnifa[k] = (qnifa2[k - 1] - qnifa[k]) / *delt;
/* L9220: */
	}
/* L9221: */
    } else {
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    dqnwfa[k] = rc0004;
	    dqnifa[k] = rc0004;
/* L9223: */
	}
/* L9224: */
    }
    if (*flag_qnbca__ && *bl_mynn_mixaerosols__ > 0) {
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    dqnbca[k] = (qnbca2[k - 1] - qnbca[k]) / *delt;
/* L9226: */
	}
/* L9227: */
    } else {
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    dqnbca[k] = rc0004;
/* L9229: */
	}
/* L9230: */
    }
    if (FALSE_) {
	problem = FALSE_;
	i__1 = *kte;
	
#pragma unroll 4
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
	    if (wsp2 > rc0011 || tk2 > rc0014 || tk2 < rc0015) {
		problem = TRUE_;
		kproblem = k;
	    }
/* L9232: */
	}
/* L9233: */
	if (problem) {
	}
    }
    return 0;
} /* mynn_tendencies__ */

__device__ M471_CALL /* Subroutine */ int moisture_check__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kte, M471Ptr<real> delt, M471Ptr<real> dp, 
	M471Ptr<real> exner, M471Ptr<real> qv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<real> qs, M471Ptr<real> th, M471Ptr<real> dqv, M471Ptr<real> dqc, M471Ptr<real> dqi, M471Ptr<real> dqs, M471Ptr<real> dth) {
    /* Initialized data */

    real xlscp = 2837.232421875f;
    real xlvcp = 2488.80029296875f;
    real zero = 0.f;
    real two = 2.f;
    real p5 = .5f;
    real rc0000 = 1e-20f;
    real rc0001 = 1e-20f;

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
    
#pragma unroll 4
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
	
#pragma unroll 4
for (k = 1; k <= i__1; ++k) {
	    if (qv[k] > two * qvmin) {
		sum += qv[k] * dp[k];
	    }
/* L9007: */
	}
/* L9008: */
	aa = dqv2 * dp[1] / max(rc0001,sum);
	if (aa < p5) {
	    i__1 = *kte;
	    
#pragma unroll 4
for (k = 1; k <= i__1; ++k) {
		if (qv[k] > two * qvmin) {
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
	 M471Ptr<real> delt, M471Ptr<real> dz, M471Ptr<real> pblh, M471Ptr<integer> nchem, M471Ptr<integer> ndvel, 
	M471Ptr<real> chem1, M471Ptr<real> vd1, M471Ptr<real> settle1, M471Ptr<real> rho, M471Ptr<real> flt, M471Ptr<real> tcd, M471Ptr<real> qcd, M471Ptr<real> dfh, M471Ptr<real> s_aw1__, M471Ptr<real> s_awchem1__, M471Ptr<real> emis_ant_no__, M471Ptr<real> frp, M471Ptr<logical> enh_mix__, M471Ptr<integer> bl_mynn_edmf__,
	 M471Ptr<real> upwind) {
    /* Initialized data */

    real zero = 0.f;
    real one = 1.f;
    real two = 2.f;
    real p5 = .5f;
    real rc0000 = .1f;
    real rc0001 = 10.f;
    real rc0002 = 100.f;
    real rc0003 = .5f;
    real rc0004 = 1e-4f;
    real rc0005 = .15f;
    real rc0006 = 1.1f;

    /* System generated locals */
    integer dfh_offset, dz_offset, tcd_offset, qcd_offset, rho_offset, 
	    s_aw1_offset, chem1_dim1, chem1_offset, settle1_dim1, 
	    settle1_offset, s_awchem1_dim1, s_awchem1_offset, i__1, i__2;
    real r__1, r__2, r__3;

    /* Builtin functions */
    

    /* Local variables */
    real khdz_old__;
    integer kmaxfire;
    M471Ptr<real> a=m471_array<real>(m471_work,(304*(MYNN471_NZ-49)+15016),m471_n);
M471Ptr<real> b=m471_array<real>(m471_work,(308*(MYNN471_NZ-49)+15212),m471_n);
M471Ptr<real> c__=m471_array<real>(m471_work,(312*(MYNN471_NZ-49)+15408),m471_n);
M471Ptr<real> d__=m471_array<real>(m471_work,(316*(MYNN471_NZ-49)+15604),m471_n);
    integer k;
    real  khdz_back__;
M471Ptr<real> x=m471_array<real>(m471_work,(320*(MYNN471_NZ-49)+15800),m471_n);
    integer ic;
    real dzk, no_threshold__, dztop, 
	    frp_threshold__, 
	    pblh_threshold__;
M471Ptr<real> dtz=m471_array<real>(m471_work,(324*(MYNN471_NZ-49)+15996),m471_n);
M471Ptr<real> khdz=m471_array<real>(m471_work,(328*(MYNN471_NZ-49)+16192),m471_n);
M471Ptr<real> rhoz=m471_array<real>(m471_work,(332*(MYNN471_NZ-49)+16392),m471_n);
M471Ptr<real> dncont=m471_array<real>(m471_work,(336*(MYNN471_NZ-49)+16592),m471_n);
M471Ptr<real> rhoinv=m471_array<real>(m471_work,(340*(MYNN471_NZ-49)+16788),m471_n);
M471Ptr<real> upcont=m471_array<real>(m471_work,(344*(MYNN471_NZ-49)+16984),m471_n);
    
    

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
    settle1_dim1 = *kte - *kts + 1;
    settle1_offset = *kts + settle1_dim1;
    settle1 -= settle1_offset;
    chem1_dim1 = *kte - *kts + 1;
    chem1_offset = *kts + chem1_dim1;
    chem1 -= chem1_offset;
    --vd1;

    /* Function Body */
    if (*enh_mix__) /* m480 unread unless */ { no_threshold__ = rc0000; }
    if (*enh_mix__) /* m480 unread unless */ { frp_threshold__ = rc0001; }
    if (*enh_mix__) /* m480 unread unless */ { pblh_threshold__ = rc0002; }
    /* unread private value */;
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	dtz[k - 1] = *delt / dz[k];
/* L9004: */
    }
/* L9005: */
    rhoz[*kts - 1] = rho[*kts];
    rhoinv[*kts - 1] = one / rho[*kts];
    khdz[*kts - 1] = rhoz[*kts - 1] * dfh[*kts];
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	rhoz[k - 1] = (rho[k] * dz[k - 1] + rho[k - 1] * dz[k]) / (dz[k - 1] 
		+ dz[k]);
/* Computing MAX */
	r__1 = rhoz[k - 1];
	rhoz[k - 1] = max(r__1,rc0004);
/* Computing MAX */
	r__1 = rho[k];
	rhoinv[k - 1] = one / max(r__1,rc0004);
	/* unread private value */;
	khdz[k - 1] = rhoz[k - 1] * dfh[k];
/* L9007: */
    }
/* L9008: */
    rhoz[*kte] = rhoz[*kte - 1];
    khdz[*kte] = rhoz[*kte] * dfh[*kte];
    if (*bl_mynn_edmf__ == 1) {
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = khdz[k - 1], r__2 = p5 * s_aw1__[k];
	    khdz[k - 1] = max(r__1,r__2);
/* Computing MAX */
	    r__1 = khdz[k - 1], r__2 = -p5 * (s_aw1__[k] - s_aw1__[k + 1]);
	    khdz[k - 1] = max(r__1,r__2);
/* L9010: */
	}
/* L9011: */
    }
    if (*enh_mix__) {
	i__1 = *kte - 1;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    /* unread private value */;
	    /* unread private value */;
	    if (*pblh < pblh_threshold__) {
		if (*emis_ant_no__ > no_threshold__) {
/* Computing MAX */
		    r__1 = rc0006 * khdz[k - 1], r__2 = (real)sqrt((real)(*
			    emis_ant_no__ / no_threshold__)) / dz[k] * rhoz[k 
			    - 1];
		    khdz[k - 1] = max(r__1,r__2);
		}
		if (*frp > frp_threshold__) {
		    r__1 = (real)log(*frp);
		    kmaxfire = ceiling_(m471_work,m471_n, &r__1);
/* Computing MAX */
/* Computing 2nd power */
		    r__3 = (real)log(*frp);
		    r__1 = rc0006 * khdz[k - 1], r__2 = (one - k / (kmaxfire *
			     two)) * (r__3 * r__3 - two * (real)log(*frp)) / 
			    dz[k] * rhoz[k - 1];
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
	if (*bl_mynn_edmf__ > 1) {
	    i__2 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__2; ++k) {
		upcont[k - 1] = s_awchem1__[k + ic * s_awchem1_dim1] - 
			s_aw1__[k] * (chem1[k + ic * chem1_dim1] * *upwind + 
			chem1[k - 1 + ic * chem1_dim1] * (one - *upwind));
		dncont[k - 1] = zero;
/* L9019: */
	    }
/* L9020: */
	    upcont[*kte - 1] = zero;
	    dncont[*kte - 1] = zero;
	    k = *kts;
	    a[0] = zero;
	    b[0] = one + dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    c__[0] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    d__[0] = chem1[k + ic * chem1_dim1] - dtz[k - 1] * (upcont[k] + 
		    dncont[k]);
	    i__2 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__2; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1];
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
		d__[k - 1] = chem1[k + ic * chem1_dim1] - dtz[k - 1] * (
			upcont[k] - upcont[k - 1] + dncont[k] - dncont[k - 1])
			;
/* L9022: */
	    }
/* L9023: */
	} else {
	    k = *kts;
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k 
		    - 1] - p5 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1];
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[k - 
		    1] * rhoinv[k - 1] * s_aw1__[k + 1];
	    d__[k - 1] = chem1[k + ic * chem1_dim1] - dtz[k - 1] * vd1[ic] * 
		    chem1[k + ic * chem1_dim1] - dtz[k - 1] * rhoinv[k - 1] * 
		    s_awchem1__[k + 1 + ic * s_awchem1_dim1];
	    i__2 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__2; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + p5 * 
			dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1] + p5 * dtz[k - 1] * rhoinv[k - 1] * (
			s_aw1__[k] - s_aw1__[k + 1]);
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[
			k - 1] * rhoinv[k - 1] * s_aw1__[k + 1];
		d__[k - 1] = chem1[k + ic * chem1_dim1] + dtz[k - 1] * rhoinv[
			k - 1] * (s_awchem1__[k + ic * s_awchem1_dim1] - 
			s_awchem1__[k + 1 + ic * s_awchem1_dim1]);
/* L9025: */
	    }
/* L9026: */
	}
	a[*kte - 1] = zero;
	b[*kte - 1] = one;
	c__[*kte - 1] = zero;
	d__[*kte - 1] = chem1[*kte + ic * chem1_dim1];
	tridiag3_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__2 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__2; ++k) {
	    chem1[k + ic * chem1_dim1] = x[k - 1] + settle1[k + ic * 
		    settle1_dim1];
/* L9028: */
	}
/* L9029: */
/* L9016: */
    }
/* L9017: */
    return 0;
} /* mynn_mix_chem__ */

__device__ M471_CALL /* Subroutine */ int mynn_mix_scalars__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<integer> i__, M471Ptr<real> delt, M471Ptr<real> dz, M471Ptr<integer> nscalars, M471Ptr<real> scalars, M471Ptr<real> rho, M471Ptr<real> flt, M471Ptr<real> tcd, M471Ptr<real> qcd, M471Ptr<real> dfh, M471Ptr<real> s_aw1__, M471Ptr<real> s_awscalars1__, M471Ptr<integer> bl_mynn_edmf__, M471Ptr<real> upwind) {
    /* Initialized data */

    real zero = 0.f;
    real one = 1.f;
    real p5 = .5f;
    real rc0000 = 1e-4f;
    real rc0001 = 1e-12f;

    /* System generated locals */
    integer dfh_offset, dz_offset, tcd_offset, qcd_offset, rho_offset, 
	    s_aw1_offset, scalars_dim1, scalars_offset, s_awscalars1_dim1, 
	    s_awscalars1_offset, i__1, i__2;
    real r__1, r__2;

    /* Local variables */
    M471Ptr<real> a=m471_array<real>(m471_work,(304*(MYNN471_NZ-49)+15016),m471_n);
M471Ptr<real> b=m471_array<real>(m471_work,(308*(MYNN471_NZ-49)+15212),m471_n);
M471Ptr<real> c__=m471_array<real>(m471_work,(312*(MYNN471_NZ-49)+15408),m471_n);
M471Ptr<real> d__=m471_array<real>(m471_work,(316*(MYNN471_NZ-49)+15604),m471_n);
    integer k;
    M471Ptr<real> x=m471_array<real>(m471_work,(320*(MYNN471_NZ-49)+15800),m471_n);
    integer ns;
    real dzk, dztop;
M471Ptr<real> dtz=m471_array<real>(m471_work,(324*(MYNN471_NZ-49)+15996),m471_n);
M471Ptr<real> khdz=m471_array<real>(m471_work,(328*(MYNN471_NZ-49)+16192),m471_n);
M471Ptr<real> rhoz=m471_array<real>(m471_work,(332*(MYNN471_NZ-49)+16392),m471_n);
M471Ptr<real> dncont=m471_array<real>(m471_work,(336*(MYNN471_NZ-49)+16592),m471_n);
M471Ptr<real> rhoinv=m471_array<real>(m471_work,(340*(MYNN471_NZ-49)+16788),m471_n);
M471Ptr<real> upcont=m471_array<real>(m471_work,(344*(MYNN471_NZ-49)+16984),m471_n);
    

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
    s_awscalars1_dim1 = *kte + 1 - *kts + 1;
    s_awscalars1_offset = *kts + s_awscalars1_dim1;
    s_awscalars1__ -= s_awscalars1_offset;
    scalars_dim1 = *kte - *kts + 1;
    scalars_offset = *kts + scalars_dim1;
    scalars -= scalars_offset;

    /* Function Body */
    /* unread private value */;
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	dtz[k - 1] = *delt / dz[k];
/* L9004: */
    }
/* L9005: */
    rhoz[*kts - 1] = rho[*kts];
    rhoinv[*kts - 1] = one / rho[*kts];
    khdz[*kts - 1] = rhoz[*kts - 1] * dfh[*kts];
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	rhoz[k - 1] = (rho[k] * dz[k - 1] + rho[k - 1] * dz[k]) / (dz[k - 1] 
		+ dz[k]);
/* Computing MAX */
	r__1 = rhoz[k - 1];
	rhoz[k - 1] = max(r__1,rc0000);
/* Computing MAX */
	r__1 = rho[k];
	rhoinv[k - 1] = one / max(r__1,rc0000);
	/* unread private value */;
	khdz[k - 1] = rhoz[k - 1] * dfh[k];
/* L9007: */
    }
/* L9008: */
    rhoz[*kte] = rhoz[*kte - 1];
    khdz[*kte] = rhoz[*kte] * dfh[*kte];
    if (*bl_mynn_edmf__ == 1) {
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = khdz[k - 1], r__2 = p5 * s_aw1__[k];
	    khdz[k - 1] = max(r__1,r__2);
/* Computing MAX */
	    r__1 = khdz[k - 1], r__2 = -p5 * (s_aw1__[k] - s_aw1__[k + 1]);
	    khdz[k - 1] = max(r__1,r__2);
/* L9010: */
	}
/* L9011: */
    }
    i__1 = *nscalars;
    
#pragma unroll 1
for (ns = 1; ns <= i__1; ++ns) {
	if (*bl_mynn_edmf__ > 1) {
	    i__2 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__2; ++k) {
		upcont[k - 1] = s_awscalars1__[k + ns * s_awscalars1_dim1] - 
			s_aw1__[k] * (scalars[k + ns * scalars_dim1] * *
			upwind + scalars[k - 1 + ns * scalars_dim1] * (one - *
			upwind));
		dncont[k - 1] = zero;
/* L9016: */
	    }
/* L9017: */
	    upcont[*kte - 1] = zero;
	    dncont[*kte - 1] = zero;
	    k = *kts;
	    a[0] = zero;
	    b[0] = one + dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    c__[0] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
	    d__[0] = scalars[k + ns * scalars_dim1] - dtz[k - 1] * (upcont[k] 
		    + dncont[k]);
	    i__2 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__2; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1];
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1];
		d__[k - 1] = scalars[k + ns * scalars_dim1] - dtz[k - 1] * (
			upcont[k] - upcont[k - 1] + dncont[k] - dncont[k - 1])
			;
/* L9019: */
	    }
/* L9020: */
	} else {
	    k = *kts;
	    a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1];
	    b[k - 1] = one + dtz[k - 1] * (khdz[k] + khdz[k - 1]) * rhoinv[k 
		    - 1] - p5 * dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k + 1];
	    c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[k - 
		    1] * rhoinv[k - 1] * s_aw1__[k + 1];
	    d__[k - 1] = scalars[k + ns * scalars_dim1] - dtz[k - 1] * rhoinv[
		    k - 1] * s_awscalars1__[k + 1 + ns * s_awscalars1_dim1];
	    i__2 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__2; ++k) {
		a[k - 1] = -dtz[k - 1] * khdz[k - 1] * rhoinv[k - 1] + p5 * 
			dtz[k - 1] * rhoinv[k - 1] * s_aw1__[k];
		b[k - 1] = one + dtz[k - 1] * (khdz[k - 1] + khdz[k]) * 
			rhoinv[k - 1] + p5 * dtz[k - 1] * rhoinv[k - 1] * (
			s_aw1__[k] - s_aw1__[k + 1]);
		c__[k - 1] = -dtz[k - 1] * khdz[k] * rhoinv[k - 1] - p5 * dtz[
			k - 1] * rhoinv[k - 1] * s_aw1__[k + 1];
		d__[k - 1] = scalars[k + ns * scalars_dim1] + dtz[k - 1] * 
			rhoinv[k - 1] * (s_awscalars1__[k + ns * 
			s_awscalars1_dim1] - s_awscalars1__[k + 1 + ns * 
			s_awscalars1_dim1]);
/* L9022: */
	    }
/* L9023: */
	}
	a[*kte - 1] = zero;
	b[*kte - 1] = one;
	c__[*kte - 1] = zero;
	d__[*kte - 1] = scalars[*kte + ns * scalars_dim1];
	tridiag3_(m471_work,m471_n, kte, a, b, c__, d__, x);
	i__2 = *kte;
	
#pragma unroll 4
for (k = *kts; k <= i__2; ++k) {
/* Computing MAX */
	    r__1 = rc0001, r__2 = x[k - 1];
	    scalars[k + ns * scalars_dim1] = max(r__1,r__2);
/* L9025: */
	}
/* L9026: */
/* L9013: */
    }
/* L9014: */
    return 0;
} /* mynn_mix_scalars__ */

__device__ M471_CALL /* Subroutine */ int retrieve_exchange_coeffs__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, 
	M471Ptr<real> dfm, M471Ptr<real> dfh, M471Ptr<real> dz, M471Ptr<real> km1, M471Ptr<real> kh1) {
    /* Initialized data */

    real zero = 0.f;
    real p5 = .5f;

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
    km1[(0 + (0 + (*kts << 2))) / 4] = zero;
    kh1[*kts] = zero;
    i__1 = *kte;
    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	dzk = p5 * (dz[k] + dz[k - 1]);
	km1[k] = dfm[k] * dzk;
	kh1[k] = dfh[k] * dzk;
/* L9004: */
    }
/* L9005: */
    return 0;
} /* retrieve_exchange_coeffs__ */

__device__ M471_CALL /* Subroutine */ int tridiag_(unsigned char *m471_work, int m471_n, M471Ptr<integer> n, M471Ptr<real> a, M471Ptr<real> b, M471Ptr<real> c__, M471Ptr<real> d__) {
    /* Initialized data */

    real zero = 0.f;
    real one = 1.f;

    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__;
    real p;
M471Ptr<real> q=m471_array<real>(m471_work,(304*(MYNN471_NZ-49)+15016),m471_n);

    /* Parameter adjustments */
    --d__;
    --c__;
    --b;
    --a;

    /* Function Body */
    c__[(0 + (0 + (*n << 2))) / 4] = zero;
    q[0] = -c__[1] / b[1];
    d__[1] /= b[1];
    i__1 = *n;
    
#pragma unroll 4
for (i__ = 2; i__ <= i__1; ++i__) {
	p = one / (b[i__] + a[i__] * q[i__ - 2]);
	q[i__ - 1] = -c__[i__] * p;
	d__[i__] = (d__[i__] - a[i__] * d__[i__ - 1]) * p;
/* L9004: */
    }
/* L9005: */
    
#pragma unroll 4
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
M471Ptr<real> cp1=m471_array<real>(m471_work,(396*(MYNN471_NZ-49)+19536),m471_n);
M471Ptr<real> dp1=m471_array<real>(m471_work,(400*(MYNN471_NZ-49)+19732),m471_n);

    /* Parameter adjustments */
    --x;
    --d__;
    --c__;
    --b;
    --a;

    /* Function Body */
        real m480_cp = c__[1] / b[1];
    real m480_dp = d__[1] / b[1];
    cp1[0] = m480_cp;
    dp1[0] = m480_dp;
    i__1 = *kte;

#pragma unroll 4
for (k = 2; k <= i__1; ++k) {
	m = b[k] - m480_cp * a[k];
	m480_cp = c__[k] / m;
	cp1[k - 1] = m480_cp;
	m480_dp = (d__[k] - m480_dp * a[k]) / m;
	dp1[k - 1] = m480_dp;
/* L9004: */
    }
/* L9005: */
    real m480_x = dp1[*kte - 1];
    x[*kte] = m480_x;

#pragma unroll 4
for (k = *kte - 1; k >= 1; --k) {
	m480_x = dp1[k - 1] - cp1[k - 1] * m480_x;
	x[k] = m480_x;
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
    
#pragma unroll 4
for (k = *kte - 1; k >= 1; --k) {
	d__[k] -= c__[k] * d__[k + 1] / b[k + 1];
	b[k] -= c__[k] * a[k + 1] / b[k + 1];
/* L9004: */
    }
/* L9005: */
    i__1 = *kte;
    
#pragma unroll 4
for (k = 2; k <= i__1; ++k) {
	d__[k] -= a[k] * d__[k - 1] / b[k - 1];
/* L9007: */
    }
/* L9008: */
    i__1 = *kte;
    
#pragma unroll 4
for (k = 1; k <= i__1; ++k) {
	x[k] = d__[k] / b[k];
/* L9010: */
    }
/* L9011: */
    return 0;
} /* tridiag3_ */

__device__ M471_CALL /* Subroutine */ int get_pblh__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> pblh, M471Ptr<real> thv1, M471Ptr<real> qke1, M471Ptr<real> ust, M471Ptr<real> zw1, M471Ptr<real> dz1, M471Ptr<real> landsea, 
	M471Ptr<integer> kpbl) {
    /* Initialized data */

    real zero = 0.f;
    real one = 1.f;
    real ten = 10.f;
    real hundred = 100.f;
    real p5 = .5f;
    real rc0000 = 200.f;
    real rc0001 = 170.f;
    real rc0002 = 9e9f;
    real rc0003 = 1.5f;
    real rc0004 = .75f;
    real rc0005 = 1.25f;
    real rc0006 = 1e-6f;
    real rc0007 = 40.f;
    real rc0008 = .01f;
    real rc0009 = 350.f;
    real rc0010 = 400.f;
    real rc0011 = 700.f;

    /* System generated locals */
    integer thv1_offset, qke1_offset, dz1_offset, zw1_offset, i__1;
    real r__1, r__2, r__3;

    /* Builtin functions */
    

    /* Local variables */
    real sbl_damp__, delt_thv__;
    integer k;
    real wt, pblh_stable__;
    integer ktke;
    real qtke;
    integer kthv;
    real cpblh, qtkem1, maxqke, tkeeps, minthv, sbl_lim__;

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
    *pblh = zero;
    k = kthv + 1;
    if (*landsea - rc0003 >= zero) {
	delt_thv__ = rc0004;
    } else {
	delt_thv__ = rc0005;
    }
    *pblh = zero;
    k = kthv + 1;
    i__1 = *kte - 1;
    
#pragma unroll 4
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
	if (*pblh > zero) {
	    goto L9008;
	}
/* L9007: */
    }
L9008:
    if (FALSE_) {
	ktke = 1;
/* Computing MAX */
	r__1 = qke1[*kts];
	maxqke = max(r__1,zero);
	tkeeps = maxqke / rc0007;
	tkeeps = max(tkeeps,rc0008);
	pblh_stable__ = zero;
	k = ktke + 1;
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = p5 * qke1[k];
	    qtke = max(r__1,zero);
/* Computing MAX */
	    r__1 = p5 * qke1[k - 1];
	    qtkem1 = max(r__1,zero);
	    if (qtke <= tkeeps) {
/* Computing MIN */
/* Computing MAX */
		r__2 = qtkem1 - qtke;
		r__1 = (tkeeps - qtke) / max(r__2,rc0006);
		pblh_stable__ = zw1[k] - dz1[k - 1] * min(r__1,one);
/* Computing MAX */
		r__1 = pblh_stable__, r__2 = zw1[*kts + 1];
		pblh_stable__ = max(r__1,r__2);
	    }
	    if (k == *kte - 1) {
		pblh_stable__ = zw1[*kts + 1];
	    }
	    if (pblh_stable__ != zero) {
		goto L9011;
	    }
/* L9010: */
	}
L9011:
/* Computing MIN */
	r__1 = pblh_stable__, r__2 = *pblh + rc0009;
	pblh_stable__ = min(r__1,r__2);
/* Computing MAX */
/* Computing MAX */
	r__3 = *pblh - rc0009;
	r__1 = pblh_stable__, r__2 = max(r__3,ten);
	pblh_stable__ = max(r__1,r__2);
	if (maxqke <= tkeeps) {
	    pblh_stable__ = *pblh;
	}
    } else {
	wt = p5 * (real)tanh((real)((*pblh - hundred) / hundred)) + p5;
	cpblh = rc0010 * (one - wt) + rc0011 * wt;
/* Computing MAX */
	r__1 = *ust * cpblh;
	pblh_stable__ = max(r__1,ten);
/* Computing MAX */
	r__1 = pblh_stable__, r__2 = *pblh - sbl_lim__;
	pblh_stable__ = max(r__1,r__2);
/* Computing MIN */
	r__1 = pblh_stable__, r__2 = *pblh + sbl_lim__;
	pblh_stable__ = min(r__1,r__2);
    }
    wt = p5 * (real)tanh((real)((*pblh - sbl_lim__) / sbl_damp__)) + p5;
    *pblh = pblh_stable__ * (one - wt) + *pblh * wt;
    i__1 = *kte - 1;
    
#pragma unroll 4
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

__device__ M471_CALL /* Subroutine */ int dmp_mf__(unsigned char *m471_work, int m471_n, M471Ptr<integer> i__, M471Ptr<integer> j, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dt, M471Ptr<real> zw1, M471Ptr<real> dz1, M471Ptr<real> pres1, M471Ptr<real> rho1, 
	M471Ptr<integer> momentum_opt__, M471Ptr<integer> tke_opt__, M471Ptr<integer> scalar_opt__, 
	M471Ptr<integer> aerosol_opt__, M471Ptr<integer> numcon_opt__, M471Ptr<real> bl_mynn_closure__, M471Ptr<real> u1, M471Ptr<real> v1, M471Ptr<real> w1, M471Ptr<real> th1, M471Ptr<real> thl1, M471Ptr<real> thv1, M471Ptr<real> tk1, M471Ptr<real> qt1, M471Ptr<real> qv1, M471Ptr<real> qc1, M471Ptr<real> qke1, M471Ptr<real> qsq1, M471Ptr<real> qnc1, M471Ptr<real> qni1, M471Ptr<real> qnwfa1, M471Ptr<real> qnifa1, 
	M471Ptr<real> qnbca1, M471Ptr<real> ex1, M471Ptr<real> vt1, M471Ptr<real> vq1, M471Ptr<real> sgm1, M471Ptr<real> ust, 
	M471Ptr<real> flt, M471Ptr<real> fltv, M471Ptr<real> flq, M471Ptr<real> flqv, M471Ptr<real> pblh, M471Ptr<integer> kpbl, M471Ptr<real> dx, M471Ptr<real> landsea, M471Ptr<real> ts, M471Ptr<real> edmf_a1__, M471Ptr<real> edmf_w1__, M471Ptr<real> edmf_qt1__, M471Ptr<real> edmf_thl1__, M471Ptr<real> edmf_ent1__, 
	M471Ptr<real> edmf_qc1__, M471Ptr<real> edmf_qv1__, M471Ptr<real> edmf_u1__, M471Ptr<real> edmf_v1__, 
	M471Ptr<real> s_aw1__, M471Ptr<real> s_awthl1__, M471Ptr<real> s_awqt1__, M471Ptr<real> s_awqv1__, 
	M471Ptr<real> s_awqc1__, M471Ptr<real> s_awu1__, M471Ptr<real> s_awv1__, M471Ptr<real> s_awqke1__, 
	M471Ptr<real> s_awqsq1__, M471Ptr<real> s_awqnc1__, M471Ptr<real> s_awqni1__, M471Ptr<real> s_awqnwfa1__, M471Ptr<real> s_awqnifa1__, M471Ptr<real> s_awqnbca1__, M471Ptr<real> sub_thl1__, M471Ptr<real> sub_sqv1__, M471Ptr<real> sub_u1__, M471Ptr<real> sub_v1__, M471Ptr<real> det_thl1__, M471Ptr<real> det_sqv1__, M471Ptr<real> det_sqc1__, M471Ptr<real> det_u1__, M471Ptr<real> det_v1__, M471Ptr<integer> nchem, M471Ptr<real> chem1, M471Ptr<real> s_awchem1__, M471Ptr<logical> mix_chem__, M471Ptr<integer> nscalars, M471Ptr<real> scalars, M471Ptr<real> s_awscalars1__, 
	M471Ptr<real> qc_bl1__, M471Ptr<real> cldfra_bl1__, M471Ptr<real> qc_bl1_old__, M471Ptr<real> cldfra_bl1_old__, M471Ptr<logical> f_qc__, M471Ptr<logical> f_qi__, M471Ptr<logical> f_qnc__, 
	M471Ptr<logical> f_qni__, M471Ptr<logical> f_qnwfa__, M471Ptr<logical> f_qnifa__, M471Ptr<logical> f_qnbca__, M471Ptr<real> psig_shcu__, M471Ptr<real> maxwidth, M471Ptr<integer> ktop, M471Ptr<real> maxmf, M471Ptr<real> ztop, M471Ptr<real> excess_h__, M471Ptr<real> excess_q__, M471Ptr<integer> spp_pbl__, M471Ptr<real> pattern_spp_pbl1__, M471Ptr<real> tkeprod_up__, M471Ptr<real> el1) {
    /* Initialized data */

    real cp = 1004.5f;
    real r_v__ = 461.600006103515625f;
    real p608 = .6083624362945556640625f;
    real ep_2__ = .621750414371490478515625f;
    real cpv = 1846.4000244140625f;
    real grav = 9.81000041961669921875f;
    real ep_3__ = .378249585628509521484375f;
    real gtr = .032700002193450927734375f;
    real tv0 = 182.50872802734375f;
    real xlvcp = 2488.80029296875f;
    real b1 = 24.f;
    real zero = 0.f;
    real one = 1.f;
    real two = 2.f;
    real three = 3.f;
    real four = 4.f;
    real five = 5.f;
    real thirty = 30.f;
    real hundred = 100.f;
    real p2 = .20000000298023223876953125f;
    real p3 = .300000011920928955078125f;
    real p333 = .3333333432674407958984375f;
    real p4 = .4000000059604644775390625f;
    real p5 = .5f;
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
    real rc0020 = .01f;
    real rc0021 = .0024f;
    real rc0022 = .003f;
    real rc0023 = .0018f;
    real rc0024 = 1.1f;
    real rc0025 = .8f;
    real rc0026 = .01f;
    real rc0027 = .6f;
    real rc0028 = .04f;
    real rc0029 = .04f;
    real rc0030 = .007f;
    real rc0031 = .02f;
    real rc0032 = .008f;
    real rc0033 = 1.9f;
    real rc0034 = .05f;
    real rc0035 = .012f;
    real rc0036 = .03f;
    real rc0037 = 13.f;
    real rc0038 = 12.f;
    real rc0039 = .1f;
    real rc0040 = .4f;
    real rc0041 = .01f;
    real rc0042 = 1.34f;
    real rc0043 = 9.28f;
    real rc0044 = .58f;
    real rc0045 = 1.2e4f;
    real rc0046 = 3e3f;
    real rc0047 = 5e-4f;
    real rc0048 = .21f;
    real rc0049 = 1.57f;
    real rc0050 = 1.3f;
    real rc0051 = .33f;
    real rc0052 = 3e-4f;
    real rc0053 = 1500.f;
    real rc0054 = 4e3f;
    real rc0055 = 5e-6f;
    real rc0056 = .9f;
    real rc0057 = .15f;
    real rc0058 = .2f;
    real rc0059 = 250.f;
    real rc0060 = 1.25f;
    real rc0061 = 200.f;
    real rc0062 = 3.f;
    real rc0063 = 1e-8f;
    real rc0064 = 8e-5f;
    real rc0065 = .06f;
    real rc0066 = 2e-4f;
    real rc0067 = 1e-4f;
    real rc0068 = 1e-6f;
    real rc0069 = 3.402823466e38f;
    real rc0070 = 10.f;
    real rc0071 = 2.5f;
    real rc0072 = 1e-6f;
    real rc0073 = 1e-10f;
    real rc0074 = 1e-7f;
    real rc0075 = 1.61f;
    real rc0076 = 1800.f;
    real rc0077 = .004f;
    real rc0078 = .015f;
    real rc0079 = .36f;
    real rc0080 = 1.8f;
    real rc0081 = 1.2f;
    real rc0082 = 1.f;
    real rc0083 = 5e-5f;
    real rc0084 = 1.86f;
    real rc0085 = 2.2e-5f;
    real rc0086 = .025f;
    real rc0087 = 2.25f;
    real rc0088 = 1.7f;
    real rc0089 = 3.8f;
    real rc0090 = 23.9f;
    real rc0091 = 1.6f;
    real rc0092 = 60.f;
    real rc0093 = 1e-8f;

    /* System generated locals */
    integer pattern_spp_pbl1_offset, u1_offset, v1_offset, w1_offset, 
	    th1_offset, thl1_offset, tk1_offset, qt1_offset, qv1_offset, 
	    qc1_offset, ex1_offset, dz1_offset, thv1_offset, pres1_offset, 
	    rho1_offset, qke1_offset, qsq1_offset, qnc1_offset, qni1_offset, 
	    qnwfa1_offset, qnifa1_offset, qnbca1_offset, el1_offset, 
	    zw1_offset, edmf_a1_offset, edmf_w1_offset, edmf_qt1_offset, 
	    edmf_thl1_offset, edmf_ent1_offset, edmf_qc1_offset, 
	    edmf_u1_offset, edmf_v1_offset, edmf_qv1_offset, s_aw1_offset, 
	    s_awthl1_offset, s_awqt1_offset, s_awqv1_offset, s_awqc1_offset, 
	    s_awqnc1_offset, s_awqni1_offset, s_awqnwfa1_offset, 
	    s_awqnifa1_offset, s_awqnbca1_offset, s_awu1_offset, 
	    s_awv1_offset, s_awqke1_offset, s_awqsq1_offset, qc_bl1_offset, 
	    cldfra_bl1_offset, qc_bl1_old_offset, cldfra_bl1_old_offset, 
	    tkeprod_up_offset, chem1_dim1, chem1_offset, s_awchem1_dim1, 
	    s_awchem1_offset, scalars_dim1, scalars_offset, s_awscalars1_dim1,
	     s_awscalars1_offset, vt1_offset, vq1_offset, sgm1_offset, 
	    sub_thl1_offset, sub_sqv1_offset, sub_u1_offset, sub_v1_offset, 
	    det_thl1_offset, det_sqv1_offset, det_sqc1_offset, det_u1_offset, 
	    det_v1_offset, i__1, i__2;
    real r__1, r__2, r__3, r__4;
    doublereal d__1, d__2;

    /* Builtin functions */
    

    /* Local variables */
    integer cb_check__;
    real mf_at_cb__, exc_heat__;
    
    real cf_strat__, qc_plume__, wspd_pbl__, qc_strat__, minwidth, a, b, c__, 
	    d__, f;
M471Ptr<real> scalarsn=m471_array<real>(m471_work,(304*(MYNN471_NZ-49)+15016),m471_n);
M471Ptr<real> envm_thl__=m471_array<real>(m471_work,(304*(MYNN471_NZ-49)+15020),m471_n);
M471Ptr<real> envm_sqc__=m471_array<real>(m471_work,(308*(MYNN471_NZ-49)+15216),m471_n);
M471Ptr<real> envm_sqv__=m471_array<real>(m471_work,(312*(MYNN471_NZ-49)+15412),m471_n);
    integer k;
    real l, n, qc_mf_min__, b9, 
	    cf_thresh__, z0, q1, 
	    detrateuv, exc_moist__;
M471Ptr<real> edmf_chem__=m471_array<real>(m471_work,(316*(MYNN471_NZ-49)+15608),m471_n);
M471Ptr<real> upscalars=m471_array<real>(m471_work,(320*(MYNN471_NZ-49)+15804),m471_n);
    
    real bb;
    integer overshoot, ic, k50;
    real dl, cn;
    integer ip;
    real wa, pk, wb, un, vn, wn, wc, xl, pt, uk, vk;
    
    real cloud_base__;
    
    real an2, q2p;
    M471Ptr<integer> ktop_plume__=m471_array<integer>(m471_work,(352*(MYNN471_NZ-49)+17404),m471_n);
    real wt2, adjustment, fng, bvf, cpm, qcn, qmq, aup, rsl, qtn, hux, wlv, qww, thp, qtp, 
	    qcp, qsl, frz, dzp, oow, tvs, flx1, flt2, ukm1, maxwidth_dx__, 
	    vkm1, nup2, beta, cdet, fluxportion, qken, qncn, upak, lmax, wpbl,
	     thln, qnin, wmin, wmax, maxw, 
	    thvn, qsqn, thvk, atot, lmin, 
	    dcut, red001, red002, red003, 
	    sigq, tauc, maxwidth_cld__, esat, temp, zagl, csub, 
	    dxsa, maxwidth_pbl__, maxwidth_flx__, fltv2, acfac, cffac, 
	    cf_mf__, qcfac, pgfac, alpha, qc_mf__, qkebl, dlmin, 
	    maxqc, qstar, pwmin, pwmax, wstar, dqwdz;
M471Ptr<real> upa=m471_array<real>(m471_work,(352*(MYNN471_NZ-49)+17436),m471_n);
M471Ptr<real> ent=m471_array<real>(m471_work,(384*(MYNN471_NZ-49)+19036),m471_n);
M471Ptr<real> upw=m471_array<real>(m471_work,(416*(MYNN471_NZ-49)+20604),m471_n);
M471Ptr<real> upu=m471_array<real>(m471_work,(448*(MYNN471_NZ-49)+22204),m471_n);
M471Ptr<real> upv=m471_array<real>(m471_work,(480*(MYNN471_NZ-49)+23804),m471_n);
M471Ptr<real> dzi=m471_array<real>(m471_work,(512*(MYNN471_NZ-49)+25404),m471_n);
M471Ptr<real> qti=m471_array<real>(m471_work,(516*(MYNN471_NZ-49)+25600),m471_n);
M471Ptr<real> upqc=m471_array<real>(m471_work,(520*(MYNN471_NZ-49)+25796),m471_n);
M471Ptr<real> upqt=m471_array<real>(m471_work,(552*(MYNN471_NZ-49)+27396),m471_n);
M471Ptr<real> upqv=m471_array<real>(m471_work,(584*(MYNN471_NZ-49)+28996),m471_n);
M471Ptr<real> rhoi=m471_array<real>(m471_work,(616*(MYNN471_NZ-49)+30596),m471_n);
M471Ptr<real> chemn=m471_array<real>(m471_work,(620*(MYNN471_NZ-49)+30792),m471_n);
M471Ptr<real> upqke=m471_array<real>(m471_work,(620*(MYNN471_NZ-49)+30796),m471_n);
M471Ptr<real> upqnc=m471_array<real>(m471_work,(652*(MYNN471_NZ-49)+32396),m471_n);
M471Ptr<real> upthl=m471_array<real>(m471_work,(684*(MYNN471_NZ-49)+33996),m471_n);
M471Ptr<real> upqni=m471_array<real>(m471_work,(716*(MYNN471_NZ-49)+35596),m471_n);
M471Ptr<real> upthv=m471_array<real>(m471_work,(748*(MYNN471_NZ-49)+37196),m471_n);
M471Ptr<real> upqsq=m471_array<real>(m471_work,(780*(MYNN471_NZ-49)+38796),m471_n);
    integer ixarr1, ixarr2, ixarr3;
    real thvkm1, bcoeff, entfac, qnbcan, csigma, pct_mf__, 
	    qc_ent__, qnifan, ac_wsp__, aratio, qc_sgs__, sigmaw, psig_w__, qnwfan, entexm, sublim, qv_ent__, entexp,
	     qt_ent__, thstar, dthvdz, dzpmax;
M471Ptr<real> envi_a__=m471_array<real>(m471_work,(812*(MYNN471_NZ-49)+40396),m471_n);
M471Ptr<real> upchem=m471_array<real>(m471_work,(816*(MYNN471_NZ-49)+40596),m471_n);
M471Ptr<real> envm_u__=m471_array<real>(m471_work,(848*(MYNN471_NZ-49)+42196),m471_n);
M471Ptr<real> exneri=m471_array<real>(m471_work,(852*(MYNN471_NZ-49)+42392),m471_n);
M471Ptr<real> envm_v__=m471_array<real>(m471_work,(856*(MYNN471_NZ-49)+42588),m471_n);
M471Ptr<real> envi_w__=m471_array<real>(m471_work,(860*(MYNN471_NZ-49)+42784),m471_n);
    logical superadiabatic;
    real exc_fac__, qc_grid__, sigmath, 
	    thl_ent__, detrate, sigmaqt, 
	    qsat_tk__, detturb;
M471Ptr<real> upqnbca=m471_array<real>(m471_work,(864*(MYNN471_NZ-49)+42984),m471_n);
M471Ptr<real> upqnifa=m471_array<real>(m471_work,(896*(MYNN471_NZ-49)+44584),m471_n);
M471Ptr<real> upqnwfa=m471_array<real>(m471_work,(928*(MYNN471_NZ-49)+46184),m471_n);
M471Ptr<real> edmf_th1__=m471_array<real>(m471_work,(960*(MYNN471_NZ-49)+47784),m471_n);

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
    s_awqsq1_offset = *kts;
    s_awqsq1__ -= s_awqsq1_offset;
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
    edmf_v1_offset = *kts;
    edmf_v1__ -= edmf_v1_offset;
    edmf_u1_offset = *kts;
    edmf_u1__ -= edmf_u1_offset;
    edmf_qv1_offset = *kts;
    edmf_qv1__ -= edmf_qv1_offset;
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
    qsq1_offset = *kts;
    qsq1 -= qsq1_offset;
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
    pres1_offset = *kts;
    pres1 -= pres1_offset;
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
    s_awscalars1_dim1 = *kte + 1 - *kts + 1;
    s_awscalars1_offset = *kts + s_awscalars1_dim1;
    s_awscalars1__ -= s_awscalars1_offset;
    scalars_dim1 = *kte - *kts + 1;
    scalars_offset = *kts + scalars_dim1;
    scalars -= scalars_offset;

    /* Function Body */
    /* unread private value */;
    /* unread private value */;
    /* unread private value */;
    atot = rc0004;
    lmax = rc0005;
    lmin = rc0006;
    /* unread private value */;
    dcut = rc0008;
    /* unread private value */;
    fluxportion = rc0010;
    /* unread private value */;
    /* unread private value */;
    /* unread private value */;
    if (*momentum_opt__ > 0) /* m480 unread unless */ { pgfac = rc0014; }
    *ktop = 0;
    *ztop = zero;
    *maxmf = zero;
    *maxwidth = zero;
    *excess_h__ = zero;
    *excess_q__ = zero;
    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    if (*mix_chem__) {
	i__1 = *nchem;
	
#pragma unroll 1
for (ixarr3 = 1; ixarr3 <= i__1; ++ixarr3) {
	    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
		i__2 = *kte + 1 - *kts + 1;
		
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
		    upchem[*kts + ixarr1 - 1 + (ixarr2 + (ixarr3 << 3)) * (1*(MYNN471_NZ-49)+50) 
			    - (9*(MYNN471_NZ-49)+451)] = zero;
		}
	    }
	}
    }
    if (*scalar_opt__ > 0) {
	i__1 = *nscalars;
	
#pragma unroll 1
for (ixarr3 = 1; ixarr3 <= i__1; ++ixarr3) {
	    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
		i__2 = *kte + 1 - *kts + 1;
		
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
		    upscalars[*kts + ixarr1 - 1 + (ixarr2 + (ixarr3 << 3)) * 
			    (1*(MYNN471_NZ-49)+50) - (9*(MYNN471_NZ-49)+451)] = zero;
		}
	    }
	}
    }
    

    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_a1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_w1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_qt1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_thl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_ent1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_qc1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_qv1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_u1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_v1__[*kts + ixarr1 - 1] = zero;
    }
    if (*mix_chem__) {
	i__1 = *nchem;
	
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= i__1; ++ixarr2) {
	    i__2 = *kte - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
		/* unread private value */;
	    }
	}
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_aw1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awthl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqt1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqv1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqc1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awu1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awv1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqke1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqsq1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqnc1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqni1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqnwfa1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqnifa1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	s_awqnbca1__[*kts + ixarr1 - 1] = zero;
    }
    if (*mix_chem__) {
	i__1 = *nchem;
	
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= i__1; ++ixarr2) {
	    i__2 = *kte + 1 - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
		s_awchem1__[*kts + ixarr1 - 1 + ixarr2 * s_awchem1_dim1] = 
			zero;
	    }
	}
    }
    if (*scalar_opt__ > 0) {
	i__1 = *nscalars;
	
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= i__1; ++ixarr2) {
	    i__2 = *kte + 1 - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
		s_awscalars1__[*kts + ixarr1 - 1 + ixarr2 * s_awscalars1_dim1]
			 = zero;
	    }
	}
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_thl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_sqv1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_u1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sub_v1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_thl1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_sqv1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_sqc1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_u1__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	det_v1__[*kts + ixarr1 - 1] = zero;
    }
    nup2 = 8.f;
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	tkeprod_up__[*kts + ixarr1 - 1] = zero;
    }
    if (FALSE_) {
    }
    maxw = zero;
    cloud_base__ = rc0016;
    qkebl = zero;
    i__1 = *kte - 1;
    
#pragma unroll 4
for (k = 1; k <= i__1; ++k) {
	zagl = zw1[k] + p5 * dz1[k];
	if (zagl > *pblh + rc0017) {
	    goto L9005;
	}
	wpbl = w1[k];
	if (w1[k] < zero) {
	    wpbl = two * w1[k];
	}
/* Computing MAX */
	r__1 = maxw, r__2 = abs(wpbl);
	maxw = max(r__1,r__2);
	if (zagl <= rc0018) {
	    k50 = k;
	}
	if (k <= *kpbl) {
	    qkebl += qke1[k];
	}
/* Computing MAX */
	r__1 = qc1[k], r__2 = qc_bl1__[k];
	qc_sgs__ = max(r__1,r__2);
	if (qc_sgs__ > rc0019 && cldfra_bl1__[k] >= p5 && cloud_base__ == 
		rc0016) {
	    cloud_base__ = zw1[k];
	}
/* L9004: */
    }
L9005:
    qkebl /= (real) (*kpbl);
/* Computing MAX */
    r__1 = zero, r__2 = maxw - one;
    maxw = max(r__1,r__2);
/* Computing MAX */
    r__1 = zero, r__2 = one - maxw;
    psig_w__ = max(r__1,r__2);
    psig_w__ = min(psig_w__,*psig_shcu__);
    fltv2 = *fltv;
    if (psig_w__ < rc0020 && *fltv > zero) {
	fltv2 = -rc0011 * *fltv;
    }
    if (FALSE_) {
    }
    superadiabatic = FALSE_;
    if (*landsea - rc0003 >= zero) {
	hux = -rc0021;
    } else {
	hux = -rc0022;
    }
    tvs = *ts * (one + p608 * qv1[*kts]);
    if ((thv1[*kts] - tvs) / (p5 * dz1[*kts]) < hux) {
	superadiabatic = TRUE_;
	i__1 = max(2,k50);
	
#pragma unroll 4
for (k = 2; k <= i__1; ++k) {
	    hux = -rc0023;
	    dthvdz = (thv1[k] - thv1[k - 1]) / (p5 * (dz1[k] + dz1[k - 1]));
	    if (dthvdz < hux) {
		superadiabatic = TRUE_;
	    } else {
		superadiabatic = FALSE_;
		goto L9008;
	    }
/* L9007: */
	}
L9008:
	;
    }
    if (FALSE_) {
    }
/* Computing MIN */
    r__1 = *dx * dcut;
    maxwidth_dx__ = min(r__1,lmax);
/* Computing MIN */
    r__1 = rc0024 * *pblh;
    maxwidth_pbl__ = min(r__1,lmax);
    if (*landsea - rc0003 < zero) {
/* Computing MIN */
/* Computing MAX */
	r__3 = rc0009 * cloud_base__;
	r__1 = lmax, r__2 = max(r__3,rc0006);
	maxwidth_cld__ = min(r__1,r__2);
    } else {
/* Computing MIN */
/* Computing MAX */
	r__3 = rc0025 * cloud_base__;
	r__1 = lmax, r__2 = max(r__3,rc0006);
	maxwidth_cld__ = min(r__1,r__2);
    }
/* Computing MAX */
/* Computing 2nd power */
    r__2 = u1[*kts];
/* Computing 2nd power */
    r__3 = v1[*kts];
    r__1 = r__2 * r__2 + r__3 * r__3;
    wspd_pbl__ = (real)sqrt((real)(max(r__1,rc0026)));
    if (*landsea - rc0003 < zero) {
/* Computing MAX */
/* Computing MIN */
	r__2 = rc0005 * (rc0027 * (real)tanh((real)((*fltv - rc0028) / rc0029)
		) + p5);
	r__1 = min(r__2,rc0005);
	maxwidth_flx__ = max(r__1,zero);
    } else {
/* Computing MAX */
/* Computing MIN */
	r__2 = rc0005 * (rc0027 * (real)tanh((real)((*fltv - rc0030) / rc0031)
		) + p5);
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
    if (*flt > zero && fltv2 > rc0032 && *maxwidth > minwidth && 
	    superadiabatic) {
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upw[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upthl[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    /* unread private value */;
	}
    }
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upqt[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*momentum_opt__ > 0) /* m480 unread unless */ { upu[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*momentum_opt__ > 0) /* m480 unread unless */ { upv[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    upqc[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    /* unread private value */;
	}
    }
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*tke_opt__ > 0) /* m480 unread unless */ { upqke[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*tke_opt__ > 0) /* m480 unread unless */ { upqsq[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*numcon_opt__ > 0) /* m480 unread unless */ { upqnc[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*numcon_opt__ > 0) /* m480 unread unless */ { upqni[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*aerosol_opt__ > 0) /* m480 unread unless */ { upqnwfa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*aerosol_opt__ > 0) /* m480 unread unless */ { upqnifa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    if (*aerosol_opt__ > 0) /* m480 unread unless */ { upqnbca[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero; }
	}
    }
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    ent[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] = rc0015;
	}
    }

	cn = zero;
	d__ = -rc0033;
	dl = (*maxwidth - minwidth) / 7.f;
	
#pragma unroll 4
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
	    acfac = p5 * (real)tanh((real)((fltv2 - rc0031) / rc0034)) + p5;
	} else {
	    acfac = p5 * (real)tanh((real)((fltv2 - rc0035) / rc0036)) + p5;
	}
/* Computing MIN */
/* Computing MAX */
	r__2 = wspd_pbl__ - rc0037;
	r__1 = max(r__2,zero) / rc0038;
	ac_wsp__ = one - min(r__1,one);
	acfac = min(acfac,ac_wsp__);
	/* unread private value */;
	
#pragma unroll 4
for (ip = 1; ip <= 8; ++ip) {
	    l = minwidth + dl * (real) (ip - 1);
	    d__1 =  l;
	    d__2 =  d__;
	    n = c__ * (real)pow_dd(&d__1, &d__2);
	    upa[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = n * l * l / (*dx * *dx) * dl;
	    upa[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] *= acfac;
	    /* unread private value */;
/* L9013: */
	}
/* L9014: */
	if (FALSE_) {
	}
	z0 = rc0018;
	pwmin = rc0039;
	pwmax = rc0040;
/* Computing MAX */
	d__1 =  (gtr * fltv2 * *pblh);
	d__2 =  p333;
	r__1 = rc0041, r__2 = (real)pow_dd(&d__1, &d__2);
	wstar = max(r__1,r__2);
	qstar = max(*flq,rc0019) / wstar;
	thstar = *flt / wstar;
	csigma = rc0042;
	if (FALSE_) {
	    exc_fac__ = zero;
	} else {
	    if (*landsea - rc0003 >= zero) {
		exc_fac__ = rc0043;
	    } else {
		exc_fac__ = rc0044;
	    }
	}
	exc_fac__ *= ac_wsp__;
	d__1 =  (z0 / *pblh);
	d__2 =  p333;
	sigmaw = csigma * wstar * (real)pow_dd(&d__1, &d__2) * (one - rc0025 *
		 z0 / *pblh);
	d__1 =  (z0 / *pblh);
	d__2 =  p333;
	sigmaqt = csigma * qstar * (real)pow_dd(&d__1, &d__2);
	d__1 =  (z0 / *pblh);
	d__2 =  p333;
	sigmath = csigma * thstar * (real)pow_dd(&d__1, &d__2);
/* Computing MIN */
	r__1 = sigmaw * pwmin;
	wmin = min(r__1,rc0039);
/* Computing MIN */
	r__1 = sigmaw * pwmax;
	wmax = min(r__1,rc0009);
	if (FALSE_) {
	}
	
#pragma unroll 4
for (ip = 1; ip <= 8; ++ip) {
	    /* unread private value */;
	    upa[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+49)] = upa[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)];
	    upw[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+49)] = wmin + (real) ip / 8.f * (wmax - wmin);
	    if (*momentum_opt__ > 0) /* m480 unread unless */ { upu[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+49)] = (u1[*kts] * dz1[*kts + 1] + u1[*kts + 1] * 
		    dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    if (*momentum_opt__ > 0) /* m480 unread unless */ { upv[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+49)] = (v1[*kts] * dz1[*kts + 1] + v1[*kts + 1] * 
		    dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    upqc[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+49)] = zero;
	    exc_heat__ = exc_fac__ * upw[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+49)] * sigmath / sigmaw;
	    /* unread private value */;
	    upthl[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+49)] = (thl1[*kts] * dz1[*kts + 1] + thl1[*kts + 1]
		     * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]) + exc_heat__;
	    exc_moist__ = exc_fac__ * upw[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+49)] * sigmaqt / sigmaw;
	    upqt[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+49)] = (qt1[*kts] * dz1[*kts + 1] + qt1[*kts + 1] * 
		    dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]) + exc_moist__;
	    if (*tke_opt__ > 0) /* m480 unread unless */ { upqke[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+49)] = (qke1[*kts] * dz1[*kts + 1] + qke1[*kts + 1]
		     * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    if (*tke_opt__ > 0) /* m480 unread unless */ { upqsq[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+49)] = (qsq1[*kts] * dz1[*kts + 1] + qsq1[*kts + 1]
		     * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    if (*numcon_opt__ > 0) /* m480 unread unless */ { upqnc[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+49)] = (qnc1[*kts] * dz1[*kts + 1] + qnc1[*kts + 1]
		     * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    if (*numcon_opt__ > 0) /* m480 unread unless */ { upqni[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+49)] = (qni1[*kts] * dz1[*kts + 1] + qni1[*kts + 1]
		     * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    if (*aerosol_opt__ > 0) /* m480 unread unless */ { upqnwfa[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+49)] = (qnwfa1[*kts] * dz1[*kts + 1] + qnwfa1[*
		    kts + 1] * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    if (*aerosol_opt__ > 0) /* m480 unread unless */ { upqnifa[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+49)] = (qnifa1[*kts] * dz1[*kts + 1] + qnifa1[*
		    kts + 1] * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
	    if (*aerosol_opt__ > 0) /* m480 unread unless */ { upqnbca[ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+49)] = (qnbca1[*kts] * dz1[*kts + 1] + qnbca1[*
		    kts + 1] * dz1[*kts]) / (dz1[*kts] + dz1[*kts + 1]); }
/* L9016: */
	}
/* L9017: */
	if (*mix_chem__) {
	    
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
		i__1 = *nchem;
		
#pragma unroll 4
for (ic = 1; ic <= i__1; ++ic) {
		    upchem[(ip + (ic << 3)) * (1*(MYNN471_NZ-49)+50) - (9*(MYNN471_NZ-49)+449)] = (chem1[*kts + ic * 
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
	if (*scalar_opt__ > 0) {
	    
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
		i__1 = *nscalars;
		
#pragma unroll 4
for (ic = 1; ic <= i__1; ++ic) {
		    upscalars[(ip + (ic << 3)) * (1*(MYNN471_NZ-49)+50) - (9*(MYNN471_NZ-49)+449)] = (scalars[*kts + 
			    ic * scalars_dim1] * dz1[*kts + 1] + scalars[*kts 
			    + 1 + ic * scalars_dim1] * dz1[*kts]) / (dz1[*kts]
			     + dz1[*kts + 1]);
/* L9028: */
		}
/* L9029: */
/* L9025: */
	    }
/* L9026: */
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    /* unread private value */;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    /* unread private value */;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    /* unread private value */;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    /* unread private value */;
	}
	i__1 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	    /* unread private value */;
	}
	rhoi[*kts - 1] = rho1[*kts];
	dzi[*kts - 1] = p5 * dz1[*kts];
	/* unread private value */;
	i__1 = *kte;
	
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
	    rhoi[k - 1] = (rho1[k - 1] * dz1[k] + rho1[k] * dz1[k - 1]) / (
		    dz1[k - 1] + dz1[k]);
	    dzi[k - 1] = (dz1[k - 1] * dz1[k] + dz1[k] * dz1[k - 1]) / (dz1[k 
		    - 1] + dz1[k]);
	    /* unread private value */;
/* L9031: */
	}
/* L9032: */
/* Computing MIN */
/* Computing MAX */
	r__2 = (rc0045 - *dx) / (rc0045 - rc0046);
	r__1 = max(r__2,zero);
	if (*momentum_opt__ > 0) /* m480 unread unless */ { dxsa = one - min(r__1,one); }
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= 8; ++ixarr1) {
	    ktop_plume__[ixarr1 - 1] = 0;
	}
	
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
	    qcn = zero;
	    overshoot = 0;
	    l = minwidth + dl * (real) (ip - 1);
	    i__1 = *kte - 2;
	    
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
		zagl = zw1[k] + p5 * dz1[k];
		wmin = p3 + l * rc0047;
/* Computing MIN */
/* Computing MAX */
		r__3 = rc0050, r__4 = (real)sqrt(qkebl);
		r__1 = rc0049, r__2 = max(r__3,r__4);
		entfac = rc0048 * min(r__1,r__2);
/* Computing MIN */
/* Computing MAX */
		r__3 = zero, r__4 = zagl - *pblh;
		r__1 = one, r__2 = max(r__3,r__4) / rc0017;
		wt2 = min(r__1,r__2);
		entfac = entfac * (one - wt2) + wt2 * rc0051;
/* Computing MIN */
/* Computing MAX */
		r__2 = upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		r__1 = max(r__2,wmin);
		ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] = entfac / (min(r__1,one) * l);
/* Computing MAX */
		r__1 = ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)];
		ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] = max(r__1,rc0052);
/* Computing MIN */
		r__1 = *pblh + rc0053;
		if (zw1[k] >= min(r__1,rc0054)) {
/* Computing MIN */
		    r__1 = *pblh + rc0053;
		    ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] += (zw1[k] - min(r__1,rc0054)) * 
			    rc0055;
		}
		ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] *= one - pattern_spp_pbl1__[k];
/* Computing MIN */
		r__1 = ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)], r__2 = rc0056 / dz1[k];
		ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] = min(r__1,r__2);
		if (*momentum_opt__ > 0) /* m480 unread unless */ { uk = (u1[k] * dz1[k + 1] + u1[k + 1] * dz1[k]) / (dz1[k + 1] 
			+ dz1[k]); }
		if (*momentum_opt__ > 0) /* m480 unread unless */ { ukm1 = (u1[k - 1] * dz1[k] + u1[k] * dz1[k - 1]) / (dz1[k - 1]
			 + dz1[k]); }
		if (*momentum_opt__ > 0) /* m480 unread unless */ { vk = (v1[k] * dz1[k + 1] + v1[k + 1] * dz1[k]) / (dz1[k + 1] 
			+ dz1[k]); }
		if (*momentum_opt__ > 0) /* m480 unread unless */ { vkm1 = (v1[k - 1] * dz1[k] + v1[k] * dz1[k - 1]) / (dz1[k - 1]
			 + dz1[k]); }
		entexp = ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] * dz1[k];
		if (*momentum_opt__ > 0) /* m480 unread unless */ { entexm = entexp * p333; }
		qtn = upqt[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexp) + qt1[k] * 
			entexp;
		thln = upthl[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexp) + thl1[k] * 
			entexp;
		if (*momentum_opt__ > 0) /* m480 unread unless */ { un = upu[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexm) + u1[k] * entexm 
			+ dxsa * pgfac * (uk - ukm1); }
		if (*momentum_opt__ > 0) /* m480 unread unless */ { vn = upv[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexm) + v1[k] * entexm 
			+ dxsa * pgfac * (vk - vkm1); }
		if (*tke_opt__ > 0) /* m480 unread unless */ { qken = upqke[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexp) + qke1[k] * 
			entexp; }
		if (*tke_opt__ > 0) /* m480 unread unless */ { qsqn = upqsq[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexp) + qsq1[k] * 
			entexp; }
		if (*numcon_opt__ > 0) /* m480 unread unless */ { qncn = upqnc[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexp) + qnc1[k] * 
			entexp; }
		if (*numcon_opt__ > 0) /* m480 unread unless */ { qnin = upqni[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexp) + qni1[k] * 
			entexp; }
		if (*aerosol_opt__ > 0) /* m480 unread unless */ { qnwfan = upqnwfa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexp) + qnwfa1[
			k] * entexp; }
		if (*aerosol_opt__ > 0) /* m480 unread unless */ { qnifan = upqnifa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexp) + qnifa1[
			k] * entexp; }
		if (*aerosol_opt__ > 0) /* m480 unread unless */ { qnbcan = upqnbca[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexp) + qnbca1[
			k] * entexp; }
		qc_ent__ = qcn;
		qt_ent__ = qtn;
		/* unread private value */;
		if (*mix_chem__) {
		    i__2 = *nchem;
		    
#pragma unroll 4
for (ic = 1; ic <= i__2; ++ic) {
			chemn[ic - 1] = upchem[k + (ip + (ic << 3)) * (1*(MYNN471_NZ-49)+50) - 
				(9*(MYNN471_NZ-49)+451)] * (one - entexp) + chem1[k + ic * 
				chem1_dim1] * entexp;
/* L9040: */
		    }
/* L9041: */
		}
		if (*scalar_opt__ > 0) {
		    i__2 = *nscalars;
		    
#pragma unroll 4
for (ic = 1; ic <= i__2; ++ic) {
			scalarsn[ic - 1] = upscalars[k + (ip + (ic << 3)) * 
				(1*(MYNN471_NZ-49)+50) - (9*(MYNN471_NZ-49)+451)] * (one - entexp) + scalars[k + ic * 
				scalars_dim1] * entexp;
/* L9043: */
		    }
/* L9044: */
		}
		pk = (pres1[k] * dz1[k + 1] + pres1[k + 1] * dz1[k]) / (dz1[k]
			 + dz1[k + 1]);
		condensation_edmf__(m471_work,m471_n, &qtn, &thln, &pk, &zw1[k + 1], &thvn, &
			qcn);
		thvk = (thv1[k] * dz1[k + 1] + thv1[k + 1] * dz1[k]) / (dz1[k 
			+ 1] + dz1[k]);
		thvkm1 = (thv1[k - 1] * dz1[k] + thv1[k] * dz1[k - 1]) / (dz1[
			k - 1] + dz1[k]);
		b = grav * (thvn / thvk - one);
		if (b > rc0007) {
		    bcoeff = rc0057;
		} else {
		    bcoeff = rc0058;
		}
		if (upw[k - 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] < rc0058) {
/* Computing MAX */
		    r__1 = upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
/* Computing MIN */
		    r__2 = dz1[k];
		    wn = upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + (-rc0000 * ent[k + ip * (1*(MYNN471_NZ-49)+49) - 
			    (1*(MYNN471_NZ-49)+50)] * upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + bcoeff * b / max(
			    r__1,rc0058)) * min(r__2,rc0059);
		} else {
/* Computing MIN */
		    r__1 = dz1[k];
		    wn = upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + (-rc0000 * ent[k + ip * (1*(MYNN471_NZ-49)+49) - 
			    (1*(MYNN471_NZ-49)+50)] * upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + bcoeff * b / upw[k 
			    + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) * min(r__1,rc0059);
		}
/* Computing MIN */
		r__1 = rc0060 * dz1[k] / rc0061;
		if (wn > upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + min(r__1,two)) {
/* Computing MIN */
		    r__1 = rc0060 * dz1[k] / rc0061;
		    wn = upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + min(r__1,two);
		}
/* Computing MIN */
		r__1 = rc0060 * dz1[k] / rc0061;
		if (wn < upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] - min(r__1,two)) {
/* Computing MIN */
		    r__1 = rc0060 * dz1[k] / rc0061;
		    wn = upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] - min(r__1,two);
		}
/* Computing MIN */
		r__1 = max(wn,zero);
		wn = min(r__1,rc0062);
		if (k == *kts + 1 && wn < rc0063) {
		    nup2 = 0.f;
		    goto L9038;
		}
		if (FALSE_) {
		    if (wn >= rc0062) {
		    }
		}
		if (wn <= zero && overshoot == 0) {
		    overshoot = 1;
		    if (thvk - thvkm1 > zero) {
			bvf = (real)sqrt((real)(gtr * (thvk - thvkm1) / dz1[k]
				));
			frz = upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] / (bvf * dz1[k]);
/* Computing MAX */
			r__1 = min(frz,one);
			/* unread private value */;
		    }
		} else {
		    /* unread private value */;
		}
/* Computing MIN */
		r__1 = upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] / (one - upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]);
		/* unread private value */;
		detturb = rc0064;
/* Computing MAX */
		r__1 = one, r__2 = rc0009 * (wn + upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]);
		oow = -rc0065 / max(r__1,r__2);
/* Computing MIN */
/* Computing MAX */
		r__2 = oow * (wn - upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) / dz1[k];
		r__1 = max(r__2,detturb);
		/* unread private value */;
/* Computing MIN */
/* Computing MAX */
		r__2 = oow * (wn - upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) / dz1[k];
		r__1 = max(r__2,detturb);
		/* unread private value */;
		/* unread private value */;
/* Computing MAX */
		r__1 = qt_ent__ - qc_ent__;
/* Computing MAX */
		r__2 = upqt[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] - upqc[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		/* unread private value */;
		/* unread private value */;
		if (upqc[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] > rc0063) {
		    if (qc1[k] > rc0068) {
			qc_grid__ = qc1[k];
		    } else {
			qc_grid__ = qc_bl1__[k];
		    }
/* Computing MAX */
		    r__1 = upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * p5 * (qcn + upqc[k + ip * 
			    (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)]) - qc_grid__;
		    /* unread private value */;
		}
		/* unread private value */;
		/* unread private value */;
		if (wn > rc0007) {
		    upw[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = wn;
		    /* unread private value */;
		    upthl[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = thln;
		    upqt[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qtn;
		    upqc[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qcn;
		    if (*momentum_opt__ > 0) /* m480 unread unless */ { upu[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = un; }
		    if (*momentum_opt__ > 0) /* m480 unread unless */ { upv[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = vn; }
		    if (*tke_opt__ > 0) /* m480 unread unless */ { upqke[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qken; }
		    if (*tke_opt__ > 0) /* m480 unread unless */ { upqsq[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qsqn; }
		    if (*numcon_opt__ > 0) /* m480 unread unless */ { upqnc[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qncn; }
		    if (*numcon_opt__ > 0) /* m480 unread unless */ { upqni[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qnin; }
		    if (*aerosol_opt__ > 0) /* m480 unread unless */ { upqnwfa[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qnwfan; }
		    if (*aerosol_opt__ > 0) /* m480 unread unless */ { upqnifa[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qnifan; }
		    if (*aerosol_opt__ > 0) /* m480 unread unless */ { upqnbca[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = qnbcan; }
		    upa[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		    if (*mix_chem__) {
			i__2 = *nchem;
			
#pragma unroll 4
for (ic = 1; ic <= i__2; ++ic) {
			    upchem[k + 1 + (ip + (ic << 3)) * (1*(MYNN471_NZ-49)+50) - (9*(MYNN471_NZ-49)+451)] = 
				    chemn[ic - 1];
/* L9046: */
			}
/* L9047: */
		    }
		    if (*scalar_opt__ > 0) {
			i__2 = *nscalars;
			
#pragma unroll 4
for (ic = 1; ic <= i__2; ++ic) {
			    upscalars[k + 1 + (ip + (ic << 3)) * (1*(MYNN471_NZ-49)+50) - (9*(MYNN471_NZ-49)+451)] = 
				    scalarsn[ic - 1];
/* L9049: */
			}
/* L9050: */
		    }
		    *ktop = max(*ktop,k);
		} else {
		    if (FALSE_) {
		    }
		    goto L9038;
		}
/* L9037: */
	    }
L9038:
	    if (FALSE_) {
		red001 = -rc0069;
		
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
/* Computing MAX */
		    r__1 = red001, r__2 = upw[ixarr1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		    red001 = max(r__1,r__2);
		}
		red002 = rc0069;
		
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
/* Computing MIN */
		    r__1 = red002, r__2 = upa[ixarr1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		    red002 = min(r__1,r__2);
		}
		red003 = -rc0069;
		
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
/* Computing MAX */
		    r__1 = red003, r__2 = upa[ixarr1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		    red003 = max(r__1,r__2);
		}
		if (red001 > rc0070 || red002 < zero || red003 > atot || nup2 
			> 10.f) {
		}
	    }
	    ktop_plume__[ip - 1] = k;
	    *excess_h__ = max(*excess_h__,exc_heat__);
	    *excess_q__ = max(*excess_q__,exc_moist__);
/* L9034: */
	}
/* L9035: */
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
real m480_fold0=s_aw1__[k];
real m480_fold1=s_awthl1__[k];
real m480_fold2=s_awqt1__[k];
real m480_fold3=s_awqc1__[k];
real m480_fold4=s_awqv1__[k];
#pragma unroll 4
for (ip=1;ip<=8;++ip) {

		m480_fold0 += rhoi[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[k + 
			ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * psig_w__;
		m480_fold1 += rhoi[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[k 
			+ ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upthl[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * psig_w__;
		m480_fold2 += rhoi[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[k + 
			ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqt[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * psig_w__;
		qc_plume__ = upqc[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		m480_fold3 += rhoi[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[k + 
			ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * qc_plume__ * psig_w__;
		m480_fold4 = m480_fold2 - m480_fold3;
/* L9055: */
	    
}
s_aw1__[k]=m480_fold0;
s_awthl1__[k]=m480_fold1;
s_awqt1__[k]=m480_fold2;
s_awqc1__[k]=m480_fold3;
s_awqv1__[k]=m480_fold4;
}
/* L9053: */
	if (*momentum_opt__ > 0) {
	    
#pragma unroll 1
for(k=*kts;k<=*kte-1;++k){
real m480_fold0=s_awu1__[k];
real m480_fold1=s_awv1__[k];
#pragma unroll 4
for (ip=1;ip<=8;++ip) {

		    m480_fold0 += rhoi[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[
			    k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upu[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    psig_w__;
		    m480_fold1 += rhoi[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[
			    k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upv[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    psig_w__;
/* L9061: */
		
}
s_awu1__[k]=m480_fold0;
s_awv1__[k]=m480_fold1;
}
/* L9059: */
	}
	if (*tke_opt__ > 0) {
	    
#pragma unroll 1
for(k=*kts;k<=*kte-1;++k){
real m480_fold0=s_awqke1__[k];
#pragma unroll 4
for (ip=1;ip<=8;++ip) {

		    m480_fold0 += rhoi[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqke[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    psig_w__;
/* L9067: */
		
}
s_awqke1__[k]=m480_fold0;
}
/* L9065: */
	    if (*bl_mynn_closure__ > rc0071) {
		
#pragma unroll 1
for(k=*kts;k<=*kte-1;++k){
real m480_fold0=s_awqsq1__[k];
#pragma unroll 4
for (ip=1;ip<=8;++ip) {

			m480_fold0 += rhoi[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] *
				 upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqsq[k + ip * (1*(MYNN471_NZ-49)+50) - 
				(1*(MYNN471_NZ-49)+51)] * psig_w__;
/* L9073: */
		    
}
s_awqsq1__[k]=m480_fold0;
}
/* L9071: */
	    }
	}
	if (*mix_chem__) {
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
		    i__2 = *nchem;
		    
#pragma unroll 4
for (ic = 1; ic <= i__2; ++ic) {
			s_awchem1__[k + ic * s_awchem1_dim1] += rhoi[k - 1] * 
				upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] 
				* upchem[k + (ip + (ic << 3)) * (1*(MYNN471_NZ-49)+50) - (9*(MYNN471_NZ-49)+451)] * 
				psig_w__;
/* L9082: */
		    }
/* L9083: */
/* L9079: */
		}
/* L9080: */
/* L9076: */
	    }
/* L9077: */
	}
	if (*scalar_opt__ > 0) {
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
		    i__2 = *nscalars;
		    
#pragma unroll 4
for (ic = 1; ic <= i__2; ++ic) {
			s_awscalars1__[k + ic * s_awscalars1_dim1] += rhoi[k 
				- 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[k + ip * 
				(1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upscalars[k + (ip + (ic << 3)) * 
				(1*(MYNN471_NZ-49)+50) - (9*(MYNN471_NZ-49)+451)] * psig_w__;
/* L9091: */
		    }
/* L9092: */
/* L9088: */
		}
/* L9089: */
/* L9085: */
	    }
/* L9086: */
	}
	if (*numcon_opt__ > 0) {
	    
#pragma unroll 1
for(k=*kts;k<=*kte-1;++k){
real m480_fold0=s_awqnc1__[k];
real m480_fold1=s_awqni1__[k];
#pragma unroll 4
for (ip=1;ip<=8;++ip) {

		    m480_fold0 += rhoi[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqnc[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    psig_w__;
		    m480_fold1 += rhoi[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqni[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    psig_w__;
/* L9097: */
		
}
s_awqnc1__[k]=m480_fold0;
s_awqni1__[k]=m480_fold1;
}
/* L9095: */
	}
	if (*aerosol_opt__ > 0) {
	    
#pragma unroll 1
for(k=*kts;k<=*kte-1;++k){
real m480_fold0=s_awqnwfa1__[k];
real m480_fold1=s_awqnifa1__[k];
real m480_fold2=s_awqnbca1__[k];
#pragma unroll 4
for (ip=1;ip<=8;++ip) {

		    m480_fold0 += rhoi[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqnwfa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] 
			    * psig_w__;
		    m480_fold1 += rhoi[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqnifa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] 
			    * psig_w__;
		    m480_fold2 += rhoi[k - 1] * upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			    upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqnbca[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] 
			    * psig_w__;
/* L9103: */
		
}
s_awqnwfa1__[k]=m480_fold0;
s_awqnifa1__[k]=m480_fold1;
s_awqnbca1__[k]=m480_fold2;
}
/* L9101: */
	}
	if ((r__1 = s_aw1__[*kts + 1], abs(r__1)) > rc0063) {
/* Computing MAX */
	    r__1 = s_aw1__[*kts + 1] * (thv1[*kts] - thv1[*kts + 1]) / dzi[*
		    kts];
	    flx1 = max(r__1,rc0072);
	} else {
	    flx1 = zero;
	}
	adjustment = one;
	flt2 = max(*fltv,zero);
	if (flx1 > fluxportion * flt2 / dz1[*kts] && flx1 > zero) {
/* Computing MAX */
	    r__1 = rc0026, r__2 = fluxportion * flt2 / dz1[*kts] / flx1;
	    adjustment = max(r__1,r__2);
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_aw1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awthl1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqt1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqc1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    i__1 = *kte + 1 - *kts + 1;
	    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		s_awqv1__[*kts + ixarr1 - 1] *= adjustment;
	    }
	    if (*numcon_opt__ > 0) {
		i__1 = *kte + 1 - *kts + 1;
		
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		    s_awqnc1__[*kts + ixarr1 - 1] *= adjustment;
		}
		i__1 = *kte + 1 - *kts + 1;
		
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		    s_awqni1__[*kts + ixarr1 - 1] *= adjustment;
		}
	    }
	    if (*aerosol_opt__ > 0) {
		i__1 = *kte + 1 - *kts + 1;
		
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		    s_awqnwfa1__[*kts + ixarr1 - 1] *= adjustment;
		}
		i__1 = *kte + 1 - *kts + 1;
		
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		    s_awqnifa1__[*kts + ixarr1 - 1] *= adjustment;
		}
		i__1 = *kte + 1 - *kts + 1;
		
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		    s_awqnbca1__[*kts + ixarr1 - 1] *= adjustment;
		}
	    }
	    if (*momentum_opt__ > 0) {
		i__1 = *kte + 1 - *kts + 1;
		
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		    s_awu1__[*kts + ixarr1 - 1] *= adjustment;
		}
		i__1 = *kte + 1 - *kts + 1;
		
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		    s_awv1__[*kts + ixarr1 - 1] *= adjustment;
		}
	    }
	    if (*tke_opt__ > 0) {
		i__1 = *kte + 1 - *kts + 1;
		
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
		    s_awqke1__[*kts + ixarr1 - 1] *= adjustment;
		}
		if (*bl_mynn_closure__ > rc0071) {
		    i__1 = *kte + 1 - *kts + 1;
		    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
			s_awqsq1__[*kts + ixarr1 - 1] *= adjustment;
		    }
		}
	    }
	    if (*mix_chem__) {
		i__1 = *nchem;
		
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= i__1; ++ixarr2) {
		    i__2 = *kte + 1 - *kts + 1;
		    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
			s_awchem1__[*kts + ixarr1 - 1 + ixarr2 * 
				s_awchem1_dim1] *= adjustment;
		    }
		}
	    }
	    if (*scalar_opt__ > 0) {
		i__1 = *nscalars;
		
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= i__1; ++ixarr2) {
		    i__2 = *kte + 1 - *kts + 1;
		    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
			s_awscalars1__[*kts + ixarr1 - 1 + ixarr2 * 
				s_awscalars1_dim1] *= adjustment;
		    }
		}
	    }
	    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 8; ++ixarr2) {
		
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
		    upa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] *= adjustment;
		}
	    }
	}
	if (FALSE_) {
	}
    k = *kts;
real m480_fold0=edmf_a1__[k];
real m480_fold1=edmf_w1__[k];
real m480_fold2=edmf_qt1__[k];
real m480_fold3=edmf_thl1__[k];
real m480_fold4=edmf_ent1__[k];
real m480_fold5=edmf_qc1__[k];
#pragma unroll 4
for (ip = 1; ip <= 8; ++ip) {
	    m480_fold0 += upa[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    m480_fold1 += upa[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * p5 * upw[k + 1 + ip * 
		    (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    m480_fold2 += upa[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqt[k + 1 + ip * (1*(MYNN471_NZ-49)+50) 
		    - (1*(MYNN471_NZ-49)+51)];
	    m480_fold3 += upa[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upthl[k + 1 + ip * 
		    (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    m480_fold4 += upa[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * ent[k + 1 + ip * (1*(MYNN471_NZ-49)+49) 
		    - (1*(MYNN471_NZ-49)+50)];
	    m480_fold5 += upa[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqc[k + 1 + ip * (1*(MYNN471_NZ-49)+50) 
		    - (1*(MYNN471_NZ-49)+51)];
/* L9106: */
	}
edmf_a1__[k]=m480_fold0;
edmf_w1__[k]=m480_fold1;
edmf_qt1__[k]=m480_fold2;
edmf_thl1__[k]=m480_fold3;
edmf_ent1__[k]=m480_fold4;
edmf_qc1__[k]=m480_fold5;

/* L9107: */
	
#pragma unroll 1
integer m480_top=0;
#pragma unroll 1
for(ip=1;ip<=8;++ip){if(ktop_plume__[ip-1]-1>m480_top)m480_top=ktop_plume__[ip-1]-1;}
for(k=*kts+1;k<=m480_top;++k){
real m480_fold0=edmf_a1__[k];
real m480_fold1=edmf_w1__[k];
real m480_fold2=edmf_qt1__[k];
real m480_fold3=edmf_thl1__[k];
real m480_fold4=edmf_ent1__[k];
real m480_fold5=edmf_qc1__[k];
#pragma unroll 4
for (ip=1;ip<=8;++ip) {if(k<=ktop_plume__[ip-1]-1){

		upak = (upa[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * dzi[k - 1] + upa[k + ip * 
			(1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * dzi[k]) / (dzi[k] + dzi[k - 1]);
		m480_fold0 += upak;
		m480_fold1 += upak * (upw[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * dzi[k - 1]
			 + upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * dzi[k]) / (dzi[k] + dzi[k 
			- 1]);
		m480_fold2 += upak * (upqt[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * dzi[k - 
			1] + upqt[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * dzi[k]) / (dzi[k] + dzi[
			k - 1]);
		m480_fold3 += upak * (upthl[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * dzi[k 
			- 1] + upthl[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * dzi[k]) / (dzi[k] + 
			dzi[k - 1]);
		m480_fold4 += upak * (ent[k + 1 + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] * dzi[k - 
			1] + ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)] * dzi[k]) / (dzi[k] + dzi[
			k - 1]);
		if ((r__1 = upqc[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)], abs(r__1)) < rc0073 && 
			upqc[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] > zero) {
		    m480_fold5 += upak * upqc[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		} else {
		    m480_fold5 += upak * (upqc[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * dzi[
			    k - 1] + upqc[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * dzi[k]) / (dzi[
			    k] + dzi[k - 1]);
		}
/* L9112: */
	    
}}
edmf_a1__[k]=m480_fold0;
edmf_w1__[k]=m480_fold1;
edmf_qt1__[k]=m480_fold2;
edmf_thl1__[k]=m480_fold3;
edmf_ent1__[k]=m480_fold4;
edmf_qc1__[k]=m480_fold5;
}
/* L9110: */
	
#pragma unroll 4
for (ip = 1; ip <= 8; ++ip) {
	    k = ktop_plume__[ip - 1];
	    edmf_a1__[k] += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    edmf_w1__[k] += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upw[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    edmf_qt1__[k] += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqt[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    edmf_thl1__[k] += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upthl[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
	    edmf_ent1__[k] += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * ent[k + ip * (1*(MYNN471_NZ-49)+49) - (1*(MYNN471_NZ-49)+50)];
	    edmf_qc1__[k] += upa[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * upqc[k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
/* L9115: */
	}
/* L9116: */
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    if (edmf_a1__[k] > rc0007) {
		edmf_w1__[k] /= edmf_a1__[k];
		edmf_qt1__[k] /= edmf_a1__[k];
		edmf_thl1__[k] /= edmf_a1__[k];
		edmf_ent1__[k] /= edmf_a1__[k];
		edmf_qc1__[k] /= edmf_a1__[k];
		edmf_a1__[k] *= psig_w__;
	    }
/* L9118: */
	}
/* L9119: */
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    if (edmf_a1__[k] * edmf_w1__[k] > *maxmf) {
		*maxmf = edmf_a1__[k] * edmf_w1__[k];
	    }
/* Computing 3rd power */
	    r__2 = (r__1 = edmf_w1__[k], abs(r__1));
/* Computing MAX */
	    r__3 = p5 * (el1[k] + el1[k + 1]);
	    tkeprod_up__[k] = r__2 * (r__2 * r__2) * edmf_a1__[k] / (b1 * max(
		    r__3,rc0039));
/* L9121: */
	}
/* L9122: */
	if (*mix_chem__) {
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		
#pragma unroll 1
for (ip = 1; ip <= 8; ++ip) {
		    i__2 = *nchem;
		    
#pragma unroll 4
for (ic = 1; ic <= i__2; ++ic) {
			upak = (upa[k + 1 + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * dzi[k - 1] + upa[
				k + ip * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * dzi[k]) / (dzi[k] + dzi[k 
				- 1]);
			/* unread private value */;
/* L9130: */
		    }
/* L9131: */
/* L9127: */
		}
/* L9128: */
/* L9124: */
	    }
/* L9125: */
	    i__1 = *kte - 1;
	    
#pragma unroll 1
for (k = *kts; k <= i__1; ++k) {
		if (edmf_a1__[k] > rc0007) {
		    i__2 = *nchem;
		    
#pragma unroll 4
for (ic = 1; ic <= i__2; ++ic) {
			/* unread private value */;
/* L9136: */
		    }
/* L9137: */
		}
/* L9133: */
	    }
/* L9134: */
	}
	if (FALSE_) {
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		envi_w__[k - 1] = p333 * (edmf_w1__[k - 1] + edmf_w1__[k] + 
			edmf_w1__[k + 1]);
		envi_a__[k - 1] = p333 * (edmf_a1__[k - 1] + edmf_a1__[k] + 
			edmf_a1__[k + 1]) * adjustment;
/* L9139: */
	    }
/* L9140: */
	    envi_w__[*kts - 1] = edmf_w1__[*kts];
	    envi_a__[*kts - 1] = edmf_a1__[*kts];
	    envi_w__[*kte - 1] = zero;
	    envi_a__[*kte - 1] = edmf_a1__[*kte];
	    envi_w__[*kte] = zero;
	    envi_a__[*kte] = edmf_a1__[*kte];
	    if (envi_w__[*kts - 1] > rc0056 * dz1[*kts] / *dt) {
		sublim = rc0056 * dz1[*kts] / *dt / envi_w__[*kts - 1];
	    } else {
		sublim = one;
	    }
	    i__1 = *kte;
	    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
		temp = envi_a__[k - 1];
		envi_a__[k - 1] = one - temp;
		envi_w__[k - 1] = csub * sublim * envi_w__[k - 1] * temp / (
			rc0011 - temp);
/* L9142: */
	    }
/* L9143: */
	    sub_thl1__[*kts] = rc0009 * envi_w__[*kts - 1] * envi_a__[*kts - 
		    1] * (rho1[*kts + 1] * thl1[*kts + 1] - rho1[*kts] * thl1[
		    *kts]) / dzi[*kts - 1] / rhoi[*kts - 1];
	    sub_sqv1__[*kts] = rc0009 * envi_w__[*kts - 1] * envi_a__[*kts - 
		    1] * (rho1[*kts + 1] * qv1[*kts + 1] - rho1[*kts] * qv1[*
		    kts]) / dzi[*kts - 1] / rhoi[*kts - 1];
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		dzi[k - 1] = rc0009 * (dz1[k] + dz1[k + 1]);
		sub_thl1__[k] = rc0009 * (envi_w__[k - 1] + envi_w__[k - 2]) *
			 rc0009 * (envi_a__[k - 1] + envi_a__[k - 2]) * (rho1[
			k + 1] * thl1[k + 1] - rho1[k] * thl1[k]) / dzi[k - 1]
			 / rhoi[k - 1];
		sub_sqv1__[k] = rc0009 * (envi_w__[k - 1] + envi_w__[k - 2]) *
			 rc0009 * (envi_a__[k - 1] + envi_a__[k - 2]) * (rho1[
			k + 1] * qv1[k + 1] - rho1[k] * qv1[k]) / dzi[k - 1] /
			 rhoi[k - 1];
/* L9145: */
	    }
/* L9146: */
	    i__1 = *kte - 1;
	    
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
		det_thl1__[k] = cdet * (envm_thl__[k - 1] - thl1[k]) * 
			envi_a__[k - 1] * psig_w__;
		det_sqv1__[k] = cdet * (envm_sqv__[k - 1] - qv1[k]) * 
			envi_a__[k - 1] * psig_w__;
		det_sqc1__[k] = cdet * (envm_sqc__[k - 1] - qc1[k]) * 
			envi_a__[k - 1] * psig_w__;
/* L9148: */
	    }
/* L9149: */
	    if (*momentum_opt__ > 0) {
		sub_u1__[*kts] = rc0009 * envi_w__[*kts - 1] * envi_a__[*kts 
			- 1] * (rho1[*kts + 1] * u1[*kts + 1] - rho1[*kts] * 
			u1[*kts]) / dzi[*kts - 1] / rhoi[*kts - 1];
		sub_v1__[*kts] = rc0009 * envi_w__[*kts - 1] * envi_a__[*kts 
			- 1] * (rho1[*kts + 1] * v1[*kts + 1] - rho1[*kts] * 
			v1[*kts]) / dzi[*kts - 1] / rhoi[*kts - 1];
		i__1 = *kte - 1;
		
#pragma unroll 4
for (k = *kts + 1; k <= i__1; ++k) {
		    sub_u1__[k] = rc0009 * (envi_w__[k - 1] + envi_w__[k - 2])
			     * rc0009 * (envi_a__[k - 1] + envi_a__[k - 2]) * 
			    (rho1[k + 1] * u1[k + 1] - rho1[k] * u1[k]) / dzi[
			    k - 1] / rhoi[k - 1];
		    sub_v1__[k] = rc0009 * (envi_w__[k - 1] + envi_w__[k - 2])
			     * rc0009 * (envi_a__[k - 1] + envi_a__[k - 2]) * 
			    (rho1[k + 1] * v1[k + 1] - rho1[k] * v1[k]) / dzi[
			    k - 1] / rhoi[k - 1];
/* L9151: */
		}
/* L9152: */
		i__1 = *kte - 1;
		
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
		    det_u1__[k] = cdet * (envm_u__[k - 1] - u1[k]) * envi_a__[
			    k - 1] * psig_w__;
		    det_v1__[k] = cdet * (envm_v__[k - 1] - v1[k]) * envi_a__[
			    k - 1] * psig_w__;
/* L9154: */
		}
/* L9155: */
	    }
	}
	i__1 = *kte - 1;
	
#pragma unroll 4
for (k = *kts; k <= i__1; ++k) {
	    /* unread private value */;
	    qti[k] = (qt1[k] * dz1[k + 1] + qt1[k + 1] * dz1[k]) / (dz1[k + 1]
		     + dz1[k]);
/* L9157: */
	}
/* L9158: */
	qti[*kts - 1] = qt1[*kts];
	cb_check__ = 0;
	i__1 = *kte - 2;
	
#pragma unroll 1
for (k = *kts + 1; k <= i__1; ++k) {
	    if (k > *ktop) {
		goto L9161;
	    }
	    if (edmf_qc1__[k] > zero) {
		if (cb_check__ == 0) {
		    mf_at_cb__ = edmf_a1__[k] * edmf_w1__[k];
		    cb_check__ = 1;
		}
		aup = edmf_a1__[k];
		/* unread private value */;
		qtp = edmf_qt1__[k];
		qcp = edmf_qc1__[k];
		dqwdz = (qti[k - 1] - qti[k]) / dz1[k];
		esat = (real)esat_blend__(m471_work,m471_n, &tk1[k]);
/* Computing MAX */
		r__1 = rc0074, r__2 = pres1[k] - ep_3__ * esat;
		/* unread private value */;
		xl = (real)xl_blend__(m471_work,m471_n, &tk1[k]);
		qsat_tk__ = (real)qsat_blend__(m471_work,m471_n, &tk1[k], &pres1[k]);
/* Computing 2nd power */
		r__1 = tk1[k];
		rsl = xl * qsat_tk__ / (r_v__ * (r__1 * r__1));
		cpm = cp + qt1[k] * cpv;
		a = one / (one + xl * rsl / cpm);
		b9 = a * rsl;
		q2p = xlvcp / ex1[k];
		pt = thl1[k] + q2p * qcp * aup;
		bb = b9 * tk1[k] / pt;
		qww = one + p608 * qt1[k];
		alpha = p608 * pt;
		beta = pt * xl / (tk1[k] * cp) - rc0075 * pt;
		if (a > zero) {
/* Computing MIN */
		    r__1 = one / a;
		    /* unread private value */;
		} else {
		    /* unread private value */;
		}
		tauc = rc0076;
/* Computing MAX */
		r__1 = rc0077, r__2 = tauc * max(zero,dqwdz);
		sigq = five * aup * edmf_w1__[k] * max(r__1,r__2);
/* Computing MAX */
		r__1 = sigq, r__2 = three * aup * (qtp - qt1[k]);
		sigq = max(r__1,r__2);
		qmq = qt1[k] - qsat_tk__;
		if (qmq > zero) {
/* Computing MIN */
		    r__1 = qsat_tk__ * rc0078;
		    sigq = min(r__1,sigq);
		}
		q1 = qmq / sigq;
		if (*landsea - rc0003 >= zero) {
/* Computing MIN */
/* Computing MAX */
		    r__2 = zero, r__3 = mf_at_cb__ - rc0034;
		    r__1 = max(r__2,r__3) / rc0034;
		    cffac = min(r__1,one);
/* Computing MIN */
/* Computing MAX */
		    r__2 = p5 + rc0079 * (real)atan((real)(rc0080 * (q1 + p2 
			    + cffac)));
		    r__1 = max(r__2,rc0026);
		    cf_mf__ = min(r__1,one);
/* Computing MAX */
		    r__1 = cf_mf__, r__2 = rc0008 * aup;
		    cf_mf__ = max(r__1,r__2);
		} else {
/* Computing MIN */
/* Computing MAX */
		    r__2 = p5 + rc0079 * (real)atan((real)(rc0080 * (q1 + p2))
			    );
		    r__1 = max(r__2,rc0026);
		    cf_mf__ = min(r__1,one);
/* Computing MAX */
		    r__1 = cf_mf__, r__2 = rc0080 * aup;
		    cf_mf__ = max(r__1,r__2);
		}
/* Computing MIN */
/* Computing MAX */
		r__2 = zero, r__3 = mf_at_cb__ - rc0029;
		r__1 = max(r__2,r__3) / rc0034;
		qcfac = rc0081 + rc0082 * min(r__1,one);
		if (*landsea - rc0003 >= zero) {
		    qc_mf__ = qcfac * (qcp * aup);
		} else {
		    if (qcp * aup > rc0083) {
			qc_mf__ = rc0084 * (qcp * aup) - rc0085;
		    } else {
			qc_mf__ = rc0081 * (qcp * aup);
		    }
		}
/* Computing MIN */
/* Computing MAX */
		r__3 = zero, r__4 = cf_mf__ - thirty;
		r__1 = one, r__2 = max(r__3,r__4) / (hundred - thirty);
		wt2 = one - min(r__1,r__2);
		qc_mf_min__ = wt2 * qsat_tk__ * rc0026 * cf_mf__ + (one - wt2)
			 * qsat_tk__ * rc0086 * cf_mf__;
		qc_mf__ = max(qc_mf__,qc_mf_min__);
		cf_strat__ = cldfra_bl1__[k];
		qc_strat__ = qc_bl1__[k];
		cldfra_bl1__[k] = max(cf_mf__,cf_strat__);
		pct_mf__ = cf_mf__ / cldfra_bl1__[k];
		qc_bl1__[k] = qc_mf__ * pct_mf__ + (one - pct_mf__) * 
			qc_strat__;
/* Computing MAX */
		r__1 = q1, r__2 = -rc0087;
		q1 = max(r__1,r__2);
		if (q1 >= one) {
		    fng = one;
		} else if (q1 >= -rc0088 && q1 < one) {
		    fng = (real)exp((real)(-p4 * (q1 - one)));
		} else if (q1 >= -rc0071 && q1 < -rc0088) {
		    fng = three + (real)exp((real)(-rc0089 * (q1 + rc0088)));
		} else {
/* Computing MIN */
		    r__1 = rc0090 + (real)exp((real)(-rc0091 * (q1 + rc0071)))
			    ;
		    fng = min(r__1,rc0092);
		}
		vt1[k] = qww - rc0003 * aup * beta * bb * fng - one;
		vq1[k] = alpha + rc0003 * aup * beta * a * fng - tv0;
	    }
/* L9160: */
	}
L9161:
	;
    }
    if (*ktop > 0) {
	red001 = -rc0069;
	i__1 = *ktop;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
/* Computing MAX */
	    r__1 = red001, r__2 = edmf_qc1__[ixarr1];
	    red001 = max(r__1,r__2);
	}
	maxqc = red001;
	if (maxqc < rc0093) {
	    *maxmf = -rc0011 * *maxmf;
	}
    }
    if (FALSE_) {
    }
    return 0;
} /* dmp_mf__ */

__device__ M471_CALL /* Subroutine */ int condensation_edmf__(unsigned char *m471_work, int m471_n, M471Ptr<real> qt, M471Ptr<real> thl, M471Ptr<real> p, M471Ptr<real> zagl, M471Ptr<real> thv, M471Ptr<real> qc) {
    /* Initialized data */

    real p1000mb = 1e5f;
    real rcp = .2857142984867095947265625f;
    real rvovrd = 1.6083624362945556640625f;
    real xlvcp = 2488.80029296875f;
    real zero = 0.f;
    real one = 1.f;
    real p5 = .5f;
    real rc0000 = 1e-6f;
    real rc0001 = 100.f;

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
    *qc = zero;
    i__1 = niter;
    
#pragma unroll 1
for (ni = 1; ni <= i__1; ++ni) {
	t = exn * *thl + xlvcp / exn * *qc;
	qs = (real)qsat_blend__(m471_work,m471_n, &t, p);
	qcold = *qc;
/* Computing MAX */
	r__1 = *qt - qs;
	*qc = p5 * *qc + p5 * max(r__1,zero);
	if ((r__1 = *qc - qcold, abs(r__1)) < diff) {
	    goto L9005;
	}
/* L9004: */
    }
L9005:
    t = exn * *thl + xlvcp / exn * *qc;
    qs = (real)qsat_blend__(m471_work,m471_n, &t, p);
/* Computing MAX */
    r__1 = *qt - qs;
    *qc = max(r__1,zero);
    if (*zagl < rc0001) {
	*qc = zero;
    }
    *thv = (*thl + xlvcp / exn * *qc) * (one + *qt * (rvovrd - one) - rvovrd *
	     *qc);
    return 0;
} /* condensation_edmf__ */

__device__ M471_CALL /* Subroutine */ int condensation_ddmf__(unsigned char *m471_work, int m471_n, M471Ptr<real> qt, M471Ptr<real> thl, M471Ptr<real> p, M471Ptr<real> zagl, M471Ptr<real> thv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<integer> debug_dd__) {
    /* Initialized data */

    real p1000mb = 1e5f;
    real rcp = .2857142984867095947265625f;
    real p608 = .6083624362945556640625f;
    real xlscp = 2837.232421875f;
    real xlvcp = 2488.80029296875f;
    real zero = 0.f;
    real one = 1.f;
    real p5 = .5f;
    real rc0000 = 1e-6f;

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
	t = exn * *thl + xlvcp / exn * *qc + xlscp / exn * *qi;
	qs = (real)qsat_blend__(m471_work,m471_n, &t, p);
	qxold = *qc + *qi;
	qx = *qc + *qi;
/* Computing MAX */
	r__1 = *qt - qs;
	qx = p5 * qx + p5 * max(r__1,zero);
	*qc = frac_liq__ * qx;
	*qi = frac_ice__ * qx;
	if ((r__1 = qx - qxold, abs(r__1)) < diff) {
	    goto L9005;
	}
/* L9004: */
    }
L9005:
    t = exn * *thl + xlvcp / exn * *qc + xlscp / exn * *qi;
    qs = (real)qsat_blend__(m471_work,m471_n, &t, p);
/* Computing MAX */
    r__1 = *qt - qs;
    qx = max(r__1,zero);
    *qc = frac_liq__ * qx;
    *qi = frac_ice__ * qx;
    *thv = (*thl + xlvcp / exn * *qc + xlscp / exn * *qi) * (one + p608 * (*
	    qt - qx) - qx);
    if (*debug_dd__ == 1) {
    }
    return 0;
} /* condensation_ddmf__ */

__device__ M471_CALL /* Subroutine */ int condensation_edmf_r__(unsigned char *m471_work, int m471_n, M471Ptr<real> qt, M471Ptr<real> thl, M471Ptr<real> p, M471Ptr<real> zagl, M471Ptr<real> thv, M471Ptr<real> qc) {
    /* Initialized data */

    real p1000mb = 1e5f;
    real rcp = .2857142984867095947265625f;
    real rvovrd = 1.6083624362945556640625f;
    real xlvcp = 2488.80029296875f;
    real zero = 0.f;
    real one = 1.f;
    real rc0000 = 2e-5f;

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
    *qc = zero;
    i__1 = niter;
    
#pragma unroll 1
for (ni = 1; ni <= i__1; ++ni) {
	qcold = *qc;
	t = exn * *thv / (one + *qt * (rvovrd - one) - rvovrd * *qc);
	qs = (real)qsat_blend__(m471_work,m471_n, &t, p);
/* Computing MAX */
	r__1 = *qt - qs;
	*qc = max(r__1,zero);
	if ((r__1 = *qc - qcold, abs(r__1)) < diff) {
	    goto L9005;
	}
/* L9004: */
    }
L9005:
    *thl = (t - xlvcp * *qc) / exn;
    return 0;
} /* condensation_edmf_r__ */

__device__ M471_CALL /* Subroutine */ int ddmp_mf__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dt, M471Ptr<real> dx,
	 M471Ptr<real> zw, M471Ptr<real> dz, M471Ptr<real> p, M471Ptr<real> u, M471Ptr<real> v, M471Ptr<real> th, M471Ptr<real> thl, 
	M471Ptr<real> thv, M471Ptr<real> tk, M471Ptr<real> qt, M471Ptr<real> qv, M471Ptr<real> qc, M471Ptr<real> qi, M471Ptr<real> qnc, M471Ptr<real> qni, M471Ptr<real> qnwfa, M471Ptr<real> qnifa, M471Ptr<real> qke, M471Ptr<real> rho, M471Ptr<real> exner, M471Ptr<real> qc_bl1__, M471Ptr<real> qi_bl1__, M471Ptr<real> cldfra_bl1__, M471Ptr<real> ust, 
	M471Ptr<real> flt, M471Ptr<real> flq, M471Ptr<real> fltv, M471Ptr<real> pblh, M471Ptr<integer> kpbl, M471Ptr<real> edmf_a_dd__, M471Ptr<real> edmf_w_dd__, M471Ptr<real> edmf_qt_dd__, M471Ptr<real> edmf_thl_dd__, M471Ptr<real> edmf_ent_dd__, M471Ptr<real> edmf_qc_dd__, M471Ptr<real> sd_aw__,
	 M471Ptr<real> sd_awthl__, M471Ptr<real> sd_awqt__, M471Ptr<real> sd_awqv__, M471Ptr<real> sd_awqc__, 
	M471Ptr<real> sd_awqi__, M471Ptr<real> sd_awqnc__, M471Ptr<real> sd_awqni__, M471Ptr<real> sd_awqnwfa__, M471Ptr<real> sd_awqnifa__, M471Ptr<real> sd_awu__, M471Ptr<real> sd_awv__, 
	M471Ptr<real> sd_awqke__, M471Ptr<real> tkeprod_dn__, M471Ptr<real> el, M471Ptr<real> rthraten, M471Ptr<real> psig, M471Ptr<real> maxmf_dd__, M471Ptr<real> maxwidth_dd__) {
    /* Initialized data */

    real cp = 1004.5f;
    real p608 = .6083624362945556640625f;
    real grav = 9.81000041961669921875f;
    real xlvcp = 2488.80029296875f;
    real b1 = 24.f;
    real zero = 0.f;
    real one = 1.f;
    real two = 2.f;
    real eight = 8.f;
    real p1 = .100000001490116119384765625f;
    real p2 = .20000000298023223876953125f;
    real p333 = .3333333432674407958984375f;
    real p5 = .5f;
    real p666 = .666666686534881591796875f;
    real rc0000 = 100.f;
    real rc0001 = 500.f;
    real rc0002 = 50.f;
    real rc0003 = 200.f;
    real rc0004 = 1e-4f;
    real rc0005 = 1.f;
    real rc0006 = 1.5f;
    real rc0007 = 3.402823466e38f;
    real rc0008 = 1.2f;
    real rc0009 = 3600.f;
    real rc0010 = 80.f;
    real rc0011 = 1e-6f;
    real rc0012 = .003f;
    real rc0013 = .3f;
    real rc0014 = .01f;
    real rc0015 = 10.f;
    real rc0016 = 15.f;
    real rc0017 = 1.25f;
    real rc0018 = .2f;
    real rc0019 = 3.f;
    real rc0020 = .1f;
    real rc0021 = .5f;
    real rc0022 = .06f;
    real rc0023 = .6f;
    real rc0024 = .05f;
    real rc0025 = .33f;
    real rc0026 = 5e-4f;
    real rc0027 = .18f;
    real rc0028 = 3e-4f;
    real rc0029 = .9f;
    real rc0030 = 2.f;
    real rc0031 = .15f;
    real rc0032 = 250.f;
    real rc0033 = 0.f;
    real rc0034 = 4e-4f;

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
	     i__3, i__4, i__5;
    real r__1, r__2, r__3, r__4;
    doublereal d__1, d__2;

    /* Builtin functions */
    

    /* Local variables */
    real frac_ice__, refthvm1, frac_liq__;
    M471Ptr<integer> dd_initk__=m471_array<integer>(m471_work,(304*(MYNN471_NZ-49)+15016),m471_n);
    real refqnifa;
    logical cloudflg;
    real refqnwfa, wspd_pbl__, jump_thv__, mindownw;
M471Ptr<real> massflux=m471_array<real>(m471_work,(304*(MYNN471_NZ-49)+15036),m471_n);
    integer i__, k;
    real f0, wstar_rad__;
M471Ptr<real> downqnifa=m471_array<real>(m471_work,(308*(MYNN471_NZ-49)+15232),m471_n);
M471Ptr<real> downqnwfa=m471_array<real>(m471_work,(328*(MYNN471_NZ-49)+16232),m471_n);
    
    real dl;
    integer ki;
    real dp, pk, un, vn, wn, wa, wb, z00;
    
    real maxcooling, qs1, qs2, add, ddd, 
	    qtn, qcn, qin, dz200, jump_thetal__;
M471Ptr<real> ent=m471_array<real>(m471_work,(348*(MYNN471_NZ-49)+17232),m471_n);
    logical singlelayer;
    real tmp1, qncn, thln, qnin, refu, refv, wmin, zmin, wmax, went, thvn, 
	    thvk, refp, qx_k__, buoy, red001, rcldb, refqc, crate, refqi, reftk, cftop, refqt, qstar, pwmin;
M471Ptr<real> downa=m471_array<real>(m471_work,(368*(MYNN471_NZ-49)+18232),m471_n);
M471Ptr<real> downu=m471_array<real>(m471_work,(388*(MYNN471_NZ-49)+19232),m471_n);
M471Ptr<real> downw=m471_array<real>(m471_work,(408*(MYNN471_NZ-49)+20232),m471_n);
M471Ptr<real> downv=m471_array<real>(m471_work,(428*(MYNN471_NZ-49)+21232),m471_n);
    integer qltop;
    real pwmax, dthvx, maxdd2, refqt2;
    integer ixarr1, ixarr2;
    real bcoeff, ecoeff, def_th__, minddd, maxddd, def_qt__;
    integer qlbase;
    real qnifan, refqke, refqnc, ac_wsp__, minexc, maxexc, sigmaw, refthl, 
	    refqni, qnwfan, dz_ent__, refthv, 
	    entexm, entexp, thstar, cooling_thresh__, ent_eff__, sigmath, 
	    rho_int__, sigmaqt, radflux, 
	    refthlv, jump_qt__;
M471Ptr<real> downqc=m471_array<real>(m471_work,(448*(MYNN471_NZ-49)+22232),m471_n);
M471Ptr<real> downqi=m471_array<real>(m471_work,(468*(MYNN471_NZ-49)+23232),m471_n);
M471Ptr<real> downqt=m471_array<real>(m471_work,(488*(MYNN471_NZ-49)+24232),m471_n);
M471Ptr<real> downqnc=m471_array<real>(m471_work,(508*(MYNN471_NZ-49)+25232),m471_n);
M471Ptr<real> downthl=m471_array<real>(m471_work,(528*(MYNN471_NZ-49)+26232),m471_n);
M471Ptr<real> downqni=m471_array<real>(m471_work,(548*(MYNN471_NZ-49)+27232),m471_n);
M471Ptr<real> downthv=m471_array<real>(m471_work,(568*(MYNN471_NZ-49)+28232),m471_n);

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
    maxddd = rc0001;
    /* unread private value */;
    dz200 = rc0003;
    cooling_thresh__ = -rc0004;
    /* unread private value */;
    /* unread private value */;
    /* unread private value */;
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downw[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downthl[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downthv[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downqt[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downu[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downv[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downqc[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downqi[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downqnc[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downqni[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downqnwfa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    downqnifa[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 1
for (ixarr2 = 1; ixarr2 <= 5; ++ixarr2) {
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+50); ++ixarr1) {
	    ent[ixarr1 + ixarr2 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = zero;
	}
    }
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= 5; ++ixarr1) {
	dd_initk__[ixarr1 - 1] = 0;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_a_dd__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_w_dd__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_qt_dd__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_thl_dd__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_ent_dd__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	edmf_qc_dd__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_aw__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awthl__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqt__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqv__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqc__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqi__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqnc__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqni__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqnwfa__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqnifa__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awu__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awv__[*kts + ixarr1 - 1] = zero;
    }
    i__1 = *kte + 1 - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	sd_awqke__[*kts + ixarr1 - 1] = zero;
    }
    cloudflg = FALSE_;
    singlelayer = FALSE_;
    qltop = 1;
    /* unread private value */;
    crate = zero;
/* Computing MAX */
    i__1 = 3, i__2 = *kpbl - 5;
/* Computing MIN */
    i__4 = *kpbl + 20, i__5 = *kte - 1;
    i__3 = min(i__4,i__5);
    
#pragma unroll 1
for (k = max(i__1,i__2); k <= i__3; ++k) {
	red001 = rc0007;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= 3; ++ixarr1) {
/* Computing MIN */
	    r__1 = red001, r__2 = rthraten[k - 1 + ixarr1 - 1];
	    red001 = min(r__1,r__2);
	}
	maxcooling = red001;
	if (cldfra_bl1__[k] > p5 && cldfra_bl1__[k + 1] < p5 && maxcooling < 
		cooling_thresh__) {
	    cloudflg = TRUE_;
	    if (maxcooling < crate) {
		qltop = k;
		/* unread private value */;
		crate = maxcooling;
		if (cldfra_bl1__[k - 1] < p2) {
		    singlelayer = TRUE_;
		}
	    }
	}
/* L9004: */
    }
/* L9005: */
/* Computing MIN */
    r__1 = maxddd, r__2 = rc0008 * *dx;
    maxdd2 = min(r__1,r__2);
/* Computing MIN */
    r__1 = maxdd2, r__2 = -rc0000 * crate * rc0009 + rc0003;
    maxdd2 = min(r__1,r__2);
/* Computing MIN */
    r__1 = maxdd2, r__2 = rc0010 + p666 * zw[qltop + 1];
    maxdd2 = min(r__1,r__2);
    *maxwidth_dd__ = maxdd2;
/* Computing MAX */
    r__1 = maxdd2 - minddd;
    dl = max(r__1,zero) / 4.f;
    if (cloudflg && dl > zero) {
	i__3 = *kts;
	
#pragma unroll 4
for (k = qltop; k >= i__3; --k) {
/* Computing MAX */
	    r__1 = qt[k], r__2 = qc_bl1__[k] + qi_bl1__[k];
	    qx_k__ = max(r__1,r__2);
	    if (qx_k__ > rc0011) {
		/* unread private value */;
	    }
/* L9007: */
	}
/* L9008: */
	
#pragma unroll 4
for (i__ = 1; i__ <= 5; ++i__) {
	    dd_initk__[i__ - 1] = qltop;
/* L9010: */
	}
/* L9011: */
	f0 = zero;
/* Computing MAX */
	i__3 = 3, i__1 = qltop - 2;
	i__2 = qltop + 1;
	
#pragma unroll 4
for (k = max(i__3,i__1); k <= i__2; ++k) {
	    radflux = rthraten[k] * exner[k];
	    dp = p5 * (p[k] - p[k + 1] + (p[k - 1] - p[k]));
	    radflux = radflux * cp / grav * dp;
	    if (radflux < zero) {
		f0 = abs(radflux) + f0;
	    }
/* L9013: */
	}
/* L9014: */
/* Computing MIN */
	r__1 = max(f0,one);
	f0 = min(r__1,rc0003);
/* Computing MIN */
	r__1 = f0 * rc0012;
	add = min(r__1,rc0013);
/* Computing MAX */
/* Computing 2nd power */
	r__2 = u[*kts];
/* Computing 2nd power */
	r__3 = v[*kts];
	r__1 = r__2 * r__2 + r__3 * r__3;
	wspd_pbl__ = (real)sqrt((real)(max(r__1,rc0014)));
/* Computing MIN */
/* Computing MAX */
	r__2 = zero, r__3 = wspd_pbl__ - rc0015;
	r__1 = max(r__2,r__3) / rc0016;
	ac_wsp__ = one - min(r__1,one);
	add *= min(ac_wsp__,*psig);
	dz_ent__ = p5 * (dz[qltop + 1] + dz[qltop]);
	jump_thv__ = (thv[qltop + 1] - thv[qltop]) / dz_ent__ * dz200;
	/* unread private value */;
	/* unread private value */;
	ki = qltop;
	if (singlelayer) {
	    refthl = thl[ki];
	    refthlv = refthl * (rc0005 + p608 * qv[ki]);
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
	    /* unread private value */;
	    refp = p[ki];
	} else {
	    refthl = (thl[ki - 1] * dz[ki] + thl[ki] * dz[ki - 1]) / (dz[ki] 
		    + dz[ki - 1]);
	    refthlv = refthl * (rc0005 + p608 * (qv[ki - 1] * dz[ki] + qv[ki] 
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
	    /* unread private value */;
	    refp = (p[ki - 1] * dz[ki] + p[ki] * dz[ki - 1]) / (dz[ki] + dz[
		    ki - 1]);
	}
	d__1 =  (grav * dz200 * f0 / (refthl * rho[ki] * cp));
	d__2 =  p333;
	wstar_rad__ = rc0017 * (real)pow_dd(&d__1, &d__2);
/* Computing MIN */
	r__1 = max(wstar_rad__,p1);
	wstar_rad__ = min(r__1,rc0006);
	dthvx = thl[ki + 2] + th[ki + 2] * p608 * qt[ki + 2] - (thl[ki] + th[
		ki] * p608 * qt[ki]);
	dthvx = dthvx * dz200 / dz_ent__;
	dthvx = max(dthvx,p1);
/* Computing MAX */
	r__1 = refqc + refqi, r__2 = qc_bl1__[ki] + qi_bl1__[ki];
	rcldb = max(r__1,r__2);
	tmp1 = xlvcp * rcldb / (exner[ki] * dthvx);
/* Computing MAX */
/* Computing MIN */
	r__3 = one, r__4 = p2 + p2 * eight * tmp1;
	r__1 = zero, r__2 = min(r__3,r__4);
	ent_eff__ = max(r__1,r__2);
	went = p333 * ent_eff__ * wstar_rad__;
	/* unread private value */;
	thstar = -f0 / rho[ki] / cp / wstar_rad__ - went * jump_thv__ / 
		wstar_rad__;
	sigmaw = rc0018 * wstar_rad__;
	/* unread private value */;
	sigmath = rc0006 * abs(thstar);
	pwmin = -rc0005;
	pwmax = -rc0019;
/* Computing MAX */
/* Computing MIN */
	r__3 = sigmaw * pwmin, r__4 = -rc0020;
	r__1 = min(r__3,r__4), r__2 = -rc0018;
	wmin = max(r__1,r__2);
/* Computing MAX */
/* Computing MIN */
	r__3 = sigmaw * pwmax, r__4 = -rc0018;
	r__1 = min(r__3,r__4), r__2 = -rc0021;
	wmax = max(r__1,r__2);
/* Computing MIN */
	r__1 = -rc0022, r__2 = rc0021 * (refthvm1 - refthv);
	minexc = min(r__1,r__2);
/* Computing MIN */
	r__1 = minexc - rc0020, r__2 = -rc0023;
	maxexc = min(r__1,r__2);
/* Computing MAX */
	r__1 = maxexc, r__2 = thv[1] - refthlv;
	maxexc = max(r__1,r__2);
/* Computing MAX */
/* Computing MIN */
	r__2 = rc0024 * (-rc0013) * sigmath / sigmaw;
	r__1 = min(r__2,minexc);
	def_th__ = max(r__1,maxexc);
	qs1 = (real)qsat_blend__(m471_work,m471_n, &reftk, &refp);
	r__1 = reftk + def_th__;
	qs2 = (real)qsat_blend__(m471_work,m471_n, &r__1, &refp);
	/* unread private value */;
/* Computing MIN */
	r__1 = qs2 - qs1, r__2 = -rc0025 * (refqc + refqi);
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
	
#pragma unroll 4
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
	    i__2 = *kts + 1;
	    
#pragma unroll 1
for (k = dd_initk__[i__ - 1] - 1; k >= i__2; --k) {
		wmin = p1 + ddd * rc0026;
/* Computing MIN */
/* Computing MAX */
		r__3 = (r__1 = downw[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)], abs(r__1));
		r__2 = max(r__3,wmin);
		ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = rc0027 / (min(r__2,rc0023) * ddd);
/* Computing MAX */
		r__1 = ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = max(r__1,rc0028);
/* Computing MAX */
		r__1 = ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)], r__2 = rc0022 / zw[k];
		ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = max(r__1,r__2);
/* Computing MIN */
		r__1 = ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)], r__2 = rc0029 / dz[k];
		ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] = min(r__1,r__2);
		entexp = ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * dz[k];
		entexm = ent[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * dz[k] * p5;
		qtn = downqt[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexp) + qt[k] *
			 entexp;
		thln = downthl[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexp) + thl[
			k] * entexp;
		qncn = downqnc[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexp) + qnc[
			k] * entexp;
		qnin = downqni[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexp) + qni[
			k] * entexp;
		qnwfan = downqnwfa[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexp) + 
			qnwfa[k] * entexp;
		qnifan = downqnifa[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexp) + 
			qnifa[k] * entexp;
		un = downu[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexm) + u[k] * 
			entexm;
		vn = downv[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (one - entexm) + v[k] * 
			entexm;
		pk = (p[k - 1] * dz[k] + p[k] * dz[k - 1]) / (dz[k] + dz[k - 
			1]);
		condensation_ddmf__(m471_work,m471_n, &qtn, &thln, &pk, &zw[k], &thvn, &qcn, &
			qin, &c__0);
		thvk = (thv[k - 1] * dz[k] + thv[k] * dz[k - 1]) / (dz[k] + 
			dz[k - 1]);
		buoy = grav * (thvn / thvk - one);
		if (buoy > zero) {
		    bcoeff = rc0018;
		    ecoeff = -rc0030;
		} else {
		    bcoeff = rc0018 * (one - (real)exp((real)(-zw[k] / (
			    rc0031 * *pblh))));
		    ecoeff = -rc0030 - (real)exp((real)(-zw[k] / (rc0031 * *
			    pblh)));
		    buoy *= one - (real)exp((real)(-zw[k] / (rc0031 * *pblh)))
			    ;
		}
		if (k == *kts + 2) {
		    buoy = max(buoy,rc0014);
		}
		if (k == *kts + 1) {
		    buoy = max(buoy,rc0024);
		}
/* Computing MIN */
		r__1 = downw[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)], r__2 = -rc0018;
		mindownw = min(r__1,r__2);
/* Computing MIN */
		r__1 = dz[k];
		wn = downw[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + (ecoeff * ent[k + i__ * 
			(1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] - bcoeff * 
			buoy / mindownw) * min(r__1,rc0032);
/* Computing MIN */
		r__1 = rc0017 * dz[k] / rc0003;
		if (wn < downw[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] - min(r__1,rc0030)) {
/* Computing MIN */
		    r__1 = rc0017 * dz[k] / rc0003;
		    wn = downw[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] - min(r__1,rc0030);
		}
/* Computing MIN */
		r__1 = rc0017 * dz[k] / rc0003;
		if (wn > downw[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + min(r__1,rc0030)) {
/* Computing MIN */
		    r__1 = rc0017 * dz[k] / rc0003;
		    wn = downw[k + 1 + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] + min(r__1,rc0030);
		}
/* Computing MAX */
		r__1 = min(wn,zero), r__2 = -rc0030;
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
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= 5; ++ixarr1) {
	    downw[ixarr1 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = zero;
	}
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= 5; ++ixarr1) {
	    downa[ixarr1 * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+50)] = zero;
	}
	i__2 = *kts;
	
#pragma unroll 1
for (k = qltop; k >= i__2; --k) {
	    
#pragma unroll 4
for (i__ = 1; i__ <= 5; ++i__) {
		edmf_a_dd__[k] += downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		edmf_w_dd__[k] += downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[k + i__ * 
			(1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		edmf_qt_dd__[k] += downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqt[k + i__ *
			 (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		edmf_thl_dd__[k] += downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downthl[k + 
			i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		edmf_ent_dd__[k] += downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * ent[k + i__ * 
			(1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		edmf_qc_dd__[k] += downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqc[k + i__ *
			 (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
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
	    tkeprod_dn__[k] = r__2 * (r__2 * r__2) * edmf_a_dd__[k] / (b1 * 
		    max(r__3,p5));
/* Computing MIN */
	    r__1 = rc0034, r__2 = tkeprod_dn__[k];
	    tkeprod_dn__[k] = min(r__1,r__2);
/* L9025: */
	}
/* L9026: */
/* Computing 3rd power */
	r__1 = went;
/* Computing MAX */
	r__2 = el[qltop + 1];
	tkeprod_dn__[qltop + 1] = r__1 * (r__1 * r__1) * edmf_a_dd__[qltop] / 
		(b1 * max(r__2,p5));
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    massflux[ixarr1 - 1] = edmf_a_dd__[*kts + ixarr1 - 1] * 
		    edmf_w_dd__[*kts + ixarr1 - 1];
	}
	red001 = rc0007;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
/* Computing MIN */
	    r__1 = red001, r__2 = massflux[ixarr1 - 1];
	    red001 = min(r__1,r__2);
	}
	*maxmf_dd__ = abs(red001);
	if (FALSE_) {
	    i__2 = *kte;
	    
#pragma unroll 4
for (k = *kts; k <= i__2; ++k) {
		if (tkeprod_dn__[k] > rc0034 || edmf_w_dd__[k] < -two) {
		}
/* L9031: */
	    }
/* L9032: */
	}
	i__2 = qltop;
	
#pragma unroll 1
for (k = *kts; k <= i__2; ++k) {
	    rho_int__ = (rho[k] * dz[k + 1] + rho[k + 1] * dz[k]) / (dz[k + 1]
		     + dz[k]);
	    
#pragma unroll 4
for (i__ = 1; i__ <= 5; ++i__) {
		sd_aw__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * (r__1 = 
			downw[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)], abs(r__1));
		sd_awthl__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[
			k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downthl[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		sd_awqt__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[
			k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqt[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		sd_awqc__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[
			k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqc[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		sd_awqi__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[
			k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqi[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		sd_awqnc__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[
			k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqnc[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		sd_awqni__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[
			k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqni[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		sd_awqnwfa__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			downw[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqnwfa[k + i__ * (1*(MYNN471_NZ-49)+50) - 
			(1*(MYNN471_NZ-49)+51)];
		sd_awqnifa__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * 
			downw[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downqnifa[k + i__ * (1*(MYNN471_NZ-49)+50) - 
			(1*(MYNN471_NZ-49)+51)];
		sd_awu__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[k 
			+ i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downu[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
		sd_awv__[k] += rho_int__ * downa[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downw[k 
			+ i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)] * downv[k + i__ * (1*(MYNN471_NZ-49)+50) - (1*(MYNN471_NZ-49)+51)];
/* L9037: */
	    }
/* L9038: */
	    sd_awqv__[k] = sd_awqt__[k] - sd_awqc__[k] - sd_awqi__[k];
/* L9034: */
	}
/* L9035: */
    }
    return 0;
} /* ddmp_mf__ */

__device__ M471_CALL /* Subroutine */ int scale_aware__(unsigned char *m471_work, int m471_n, M471Ptr<real> dx, M471Ptr<real> pblh, M471Ptr<real> psig_bl__, 
	M471Ptr<real> psig_shcu__) {
    /* Initialized data */

    real zero = 0.f;
    real one = 1.f;
    real ten = 10.f;
    real rc0000 = 2.5f;
    real rc0001 = 3e3f;
    real rc0002 = .106f;
    real rc0003 = .667f;
    real rc0004 = .066f;
    real rc0005 = .071f;
    real rc0006 = 500.f;
    real rc0007 = 3500.f;
    real rc0008 = .145f;
    real rc0009 = .172f;
    real rc0010 = .17f;

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
    dxdh = max(r__1,ten) / min(*pblh,rc0001);
/* Computing 2nd power */
    r__1 = dxdh;
    d__1 =  dxdh;
    d__2 =  rc0003;
/* Computing 2nd power */
    r__2 = dxdh;
    d__3 =  dxdh;
    d__4 =  rc0003;
    *psig_bl__ = (r__1 * r__1 + rc0002 * (real)pow_dd(&d__1, &d__2)) / (r__2 *
	     r__2 + rc0004 * (real)pow_dd(&d__3, &d__4) + rc0005);
/* Computing MAX */
    r__1 = rc0000 * *dx;
/* Computing MIN */
    r__2 = *pblh + rc0006;
    dxdh = max(r__1,ten) / min(r__2,rc0007);
/* Computing 2nd power */
    r__1 = dxdh;
    d__1 =  dxdh;
    d__2 =  rc0003;
/* Computing 2nd power */
    r__2 = dxdh;
    d__3 =  dxdh;
    d__4 =  rc0003;
    *psig_shcu__ = (r__1 * r__1 + rc0008 * (real)pow_dd(&d__1, &d__2)) / (
	    r__2 * r__2 + rc0009 * (real)pow_dd(&d__3, &d__4) + rc0010);
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

    real t0c = 273.149993896484375f;
    real tice = 238.f;
    real one = 1.f;
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
	ret_val = (one - chi) * esl + chi * esi;
    }
    return ret_val;
} /* esat_blend__ */

__device__ M471_CALL real qsat_blend__(unsigned char *m471_work, int m471_n, M471Ptr<real> t, M471Ptr<real> p) {
    /* Initialized data */

    real t0c = 273.149993896484375f;
    real tice = 238.f;
    real one = 1.f;
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
	ret_val = (one - chi) * rslf + chi * rsif;
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
    real t0c = 273.149993896484375f;
    real tice = 238.f;
    real one = 1.f;

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
	ret_val = (one - chi) * xlvt + chi * xlst;
    }
    return ret_val;
} /* xl_blend__ */

__device__ M471_CALL real phim_(unsigned char *m471_work, int m471_n, M471Ptr<real> zet) {
    /* Initialized data */

    real cphm_unst__ = 16.f;
    real zero = 0.f;
    real one = 1.f;
    real two = 2.f;
    real p333 = .3333333432674407958984375f;
    real p5 = .5f;
    real p666 = .666666686534881591796875f;
    real rc0000 = 6.1f;
    real rc0001 = 2.5f;
    real rc0002 = 1.f;
    real rc0003 = 5.3f;
    real rc0004 = 1.1f;
    real rc0005 = 10.f;
    real rc0006 = 34.f;
    real rc0007 = .25f;
    real rc0008 = 1.570796f;
    real rc0009 = .57735f;
    real rc0010 = 1.1547f;
    real rc0011 = 1.5f;
    real rc0012 = 1.73205f;
    real rc0013 = 1.813799364f;

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
	dummy_0__ = one + (real)pow_dd(&d__1, &d__2);
	d__1 =  dummy_0__;
	d__2 =  rbm_st__;
	dummy_1__ = *zet + (real)pow_dd(&d__1, &d__2);
	d__1 =  dummy_0__;
	d__2 =  (rbm_st__ - 1);
	d__3 =  (*zet);
	d__4 =  (bm_st__ - one);
	dummy_11__ = one + (real)pow_dd(&d__1, &d__2) * (real)pow_dd(&d__3, &
		d__4);
	dummy_2__ = -am_st__ / dummy_1__ * dummy_11__;
	phi_m__ = one - *zet * dummy_2__;
    } else {
	d__1 =  (one - cphm_unst__ * *zet);
	d__2 =  rc0007;
	dummy_0__ = (real)pow_dd(&d__1, &d__2);
	phi_m__ = one / dummy_0__;
/* Computing 2nd power */
	r__1 = dummy_0__;
	dummy_psi__ = two * (real)log((real)(p5 * (one + dummy_0__))) + (real)
		log((real)(p5 * (one + r__1 * r__1))) - two * (real)atan(
		dummy_0__) + rc0008;
	dummy_0__ = one - am_unst__ * *zet;
	d__1 =  dummy_0__;
	d__2 =  p333;
	dummy_1__ = (real)pow_dd(&d__1, &d__2);
	d__1 =  dummy_0__;
	d__2 =  (-p666);
	dummy_11__ = -p333 * am_unst__ * (real)pow_dd(&d__1, &d__2);
/* Computing 2nd power */
	r__1 = dummy_1__;
	dummy_2__ = p333 * (r__1 * r__1 + dummy_1__ + one);
	dummy_22__ = p333 * dummy_11__ * (two * dummy_1__ + one);
	dummy_3__ = rc0009 * (two * dummy_1__ + one);
	dummy_33__ = rc0010 * dummy_11__;
	dummy_4__ = rc0011 * (real)log(dummy_2__) - rc0012 * (real)atan(
		dummy_3__) + rc0013;
/* Computing 2nd power */
	r__1 = dummy_3__;
	dummy_44__ = rc0011 / dummy_2__ * dummy_22__ - rc0012 * dummy_33__ / (
		one + r__1 * r__1);
/* Computing 2nd power */
	r__1 = *zet;
	dummy_0__ = r__1 * r__1;
	dummy_1__ = one / (one + dummy_0__);
	dummy_11__ = two * *zet;
	dummy_2__ = ((one - phi_m__) / *zet + dummy_11__ * dummy_4__ + 
		dummy_0__ * dummy_44__) * dummy_1__;
/* Computing 2nd power */
	r__1 = dummy_1__;
	dummy_22__ = -dummy_11__ * (dummy_psi__ + dummy_0__ * dummy_4__) * (
		r__1 * r__1);
	phi_m__ = one - *zet * (dummy_2__ + dummy_22__);
    }
    ret_val = phi_m__;
    return ret_val;
} /* phim_ */

__device__ M471_CALL real phih_(unsigned char *m471_work, int m471_n, M471Ptr<real> zet) {
    /* Initialized data */

    real cphh_unst__ = 16.f;
    real zero = 0.f;
    real one = 1.f;
    real two = 2.f;
    real p333 = .3333333432674407958984375f;
    real p5 = .5f;
    real p666 = .666666686534881591796875f;
    real rc0000 = 6.1f;
    real rc0001 = 2.5f;
    real rc0002 = 1.f;
    real rc0003 = 5.3f;
    real rc0004 = 1.1f;
    real rc0005 = 10.f;
    real rc0006 = 34.f;
    real rc0007 = .57735f;
    real rc0008 = 1.1547f;
    real rc0009 = 1.5f;
    real rc0010 = 1.73205f;
    real rc0011 = 1.813799364f;

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
	dummy_0__ = one + (real)pow_dd(&d__1, &d__2);
	d__1 =  dummy_0__;
	d__2 =  rbh_st__;
	dummy_1__ = *zet + (real)pow_dd(&d__1, &d__2);
	d__1 =  dummy_0__;
	d__2 =  (rbh_st__ - one);
	d__3 =  (*zet);
	d__4 =  (bh_st__ - one);
	dummy_11__ = one + (real)pow_dd(&d__1, &d__2) * (real)pow_dd(&d__3, &
		d__4);
	dummy_2__ = -ah_st__ / dummy_1__ * dummy_11__;
	ret_val = one - *zet * dummy_2__;
    } else {
	d__1 =  (one - cphh_unst__ * *zet);
	d__2 =  p5;
	dummy_0__ = (real)pow_dd(&d__1, &d__2);
	phh = one / dummy_0__;
	dummy_psi__ = two * (real)log((real)(p5 * (one + dummy_0__)));
	dummy_0__ = one - ah_unst__ * *zet;
	d__1 =  dummy_0__;
	d__2 =  p333;
	dummy_1__ = (real)pow_dd(&d__1, &d__2);
	d__1 =  dummy_0__;
	d__2 =  (-p666);
	dummy_11__ = -p333 * ah_unst__ * (real)pow_dd(&d__1, &d__2);
/* Computing 2nd power */
	r__1 = dummy_1__;
	dummy_2__ = p333 * (r__1 * r__1 + dummy_1__ + one);
	dummy_22__ = p333 * dummy_11__ * (two * dummy_1__ + one);
	dummy_3__ = rc0007 * (two * dummy_1__ + one);
	dummy_33__ = rc0008 * dummy_11__;
	dummy_4__ = rc0009 * (real)log(dummy_2__) - rc0010 * (real)atan(
		dummy_3__) + rc0011;
/* Computing 2nd power */
	r__1 = dummy_3__;
	dummy_44__ = rc0009 / dummy_2__ * dummy_22__ - rc0010 * dummy_33__ / (
		one + r__1 * r__1);
/* Computing 2nd power */
	r__1 = *zet;
	dummy_0__ = r__1 * r__1;
	dummy_1__ = one / (one + dummy_0__);
	dummy_11__ = two * *zet;
	dummy_2__ = ((one - phh) / *zet + dummy_11__ * dummy_4__ + dummy_0__ *
		 dummy_44__) * dummy_1__;
/* Computing 2nd power */
	r__1 = dummy_1__;
	dummy_22__ = -dummy_11__ * (dummy_psi__ + dummy_0__ * dummy_4__) * (
		r__1 * r__1);
	ret_val = one - *zet * (dummy_2__ + dummy_22__);
    }
    return ret_val;
} /* phih_ */

__device__ M471_CALL /* Subroutine */ int topdown_cloudrad__(unsigned char *m471_work, int m471_n, M471Ptr<integer> kts, M471Ptr<integer> kte, M471Ptr<real> dz1,
	 M471Ptr<real> zw, M471Ptr<real> fltv, M471Ptr<real> u1, M471Ptr<real> v1, M471Ptr<real> xland, M471Ptr<integer> kpbl,
	 M471Ptr<real> pblh, M471Ptr<real> sqc, M471Ptr<real> sqi, M471Ptr<real> sqw, M471Ptr<real> thl, M471Ptr<real> th1, 
	M471Ptr<real> ex1, M471Ptr<real> pres1, M471Ptr<real> rho1, M471Ptr<real> thv, M471Ptr<real> cldfra_bl1__, 
	M471Ptr<real> qc_bl__, M471Ptr<real> qi_bl__, M471Ptr<real> rthraten, M471Ptr<real> tkeprod_dn__, 
	M471Ptr<real> psig, M471Ptr<real> maxtkeprod, M471Ptr<real> cldtop_cooling__, M471Ptr<real> ent_eff__) {
    /* Initialized data */

    real cp = 1004.5f;
    real r_d__ = 287.f;
    real xlv = 2.5e6f;
    real p608 = .6083624362945556640625f;
    real ep_2__ = .621750414371490478515625f;
    real grav = 9.81000041961669921875f;
    real xlvcp = 2488.80029296875f;
    real zero = 0.f;
    real one = 1.f;
    real two = 2.f;
    real three = 3.f;
    real eight = 8.f;
    real p1 = .100000001490116119384765625f;
    real p2 = .20000000298023223876953125f;
    real p25 = .25f;
    real p3 = .300000011920928955078125f;
    real p333 = .3333333432674407958984375f;
    real p5 = .5f;
    real rc0000 = 500.f;
    real rc0001 = 200.f;
    real rc0002 = 6.9e-5f;
    real rc0003 = 2.f;
    real rc0004 = .01f;
    real rc0005 = 8.f;
    real rc0006 = 3.402823466e38f;
    real rc0007 = 100.f;
    real rc0008 = 6.112f;
    real rc0009 = 17.67f;
    real rc0010 = 273.16f;
    real rc0011 = 29.65f;
    real rc0012 = 273.15f;
    real rc0013 = 1.5f;
    real rc0014 = 150.f;
    real rc0015 = 30.f;
    real rc0016 = 3600.f;
    real rc0017 = 1.25f;
    real rc0018 = 800.f;
    real rc0019 = .4f;
    real rc0020 = 1e-5f;
    real rc0021 = 4e-4f;
    real rc0022 = 10.f;
    real rc0023 = 15.f;

    /* System generated locals */
    integer dz1_offset, sqc_offset, sqi_offset, sqw_offset, thl_offset, 
	    th1_offset, ex1_offset, pres1_offset, rho1_offset, thv_offset, 
	    cldfra_bl1_offset, qc_bl_offset, qi_bl_offset, rthraten_offset, 
	    zw_offset, tkeprod_dn_offset, i__1, i__2, i__3, i__4;
    real r__1, r__2, r__3, r__4;
    doublereal d__1, d__2;

    /* Builtin functions */
    

    /* Local variables */
    logical cloudflg;
    real wspd_pbl__;
    integer k;
    real f0, wstar_rad__, dp;
    integer kk;
    real zb1, maxcooling, zl1, zfacent_max__, dz200, dz500, tmp1, pfac, zagl, went, rvls, red001, rcldb, templ, 
	    zfmin, temps, dthvx;
M471Ptr<real> tkeprodorig=m471_array<real>(m471_work,(304*(MYNN471_NZ-49)+15016),m471_n);
M471Ptr<real> zfac=m471_array<real>(m471_work,(308*(MYNN471_NZ-49)+15212),m471_n);
    integer ixarr1;
    real phifac, minrad, ac_wsp__, dz_ent__, cooling_thresh__;
    integer kminrad;
    real  zminrad, radflux;
M471Ptr<real> zfacent=m471_array<real>(m471_work,(312*(MYNN471_NZ-49)+15408),m471_n);

    /* Parameter adjustments */
    tkeprod_dn_offset = *kts;
    tkeprod_dn__ -= tkeprod_dn_offset;
    rthraten_offset = *kts;
    rthraten -= rthraten_offset;
    qi_bl_offset = *kts;
    qi_bl__ -= qi_bl_offset;
    qc_bl_offset = *kts;
    qc_bl__ -= qc_bl_offset;
    cldfra_bl1_offset = *kts;
    cldfra_bl1__ -= cldfra_bl1_offset;
    thv_offset = *kts;
    thv -= thv_offset;
    rho1_offset = *kts;
    rho1 -= rho1_offset;
    pres1_offset = *kts;
    pres1 -= pres1_offset;
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
    dz500 = rc0000;
    dz200 = rc0001;
    cooling_thresh__ = -rc0002;
    /* unread private value */;
    zfmin = rc0004;
    /* unread private value */;
    cloudflg = FALSE_;
    minrad = zero;
    kminrad = *kpbl;
    zminrad = *pblh;
    i__1 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__1; ++ixarr1) {
	/* unread private value */;
    }
/* Computing MIN */
    i__2 = *kpbl + 10, i__3 = *kte - 1;
    i__1 = min(i__2,i__3);
    
#pragma unroll 1
for (k = 1; k <= i__1; ++k) {
	red001 = rc0006;
/* Computing MAX */
	i__3 = *kts, i__4 = k - 1;
	i__2 = k + 1 - max(i__3,i__4) + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__2; ++ixarr1) {
/* Computing MIN */
/* Computing MAX */
	    i__3 = *kts, i__4 = k - 1;
	    r__1 = red001, r__2 = rthraten[max(i__3,i__4) + ixarr1 - 1];
	    red001 = min(r__1,r__2);
	}
	maxcooling = red001;
	if (cldfra_bl1__[k] > p5 && cldfra_bl1__[k + 1] < p5 && maxcooling < 
		cooling_thresh__) {
	    cloudflg = TRUE_;
	    if (maxcooling < minrad) {
		minrad = maxcooling;
		kminrad = k;
		zminrad = zw[k] + p5 * dz1[k];
	    }
	}
/* L9004: */
    }
/* L9005: */
    if (cloudflg) {
	zl1 = p25 * dz1[*kts];
	k = kminrad;
	templ = thl[k] * ex1[k];
	rvls = rc0007 * rc0008 * (real)exp((real)(rc0009 * (templ - rc0010) / 
		(templ - rc0011))) * (ep_2__ / pres1[k + 1]);
/* Computing 2nd power */
	r__1 = templ;
	temps = templ + (sqw[k] - rvls) / (cp / xlv + ep_2__ * xlv * rvls / (
		r_d__ * (r__1 * r__1)));
	rvls = rc0007 * rc0008 * (real)exp((real)(rc0009 * (temps - rc0012) / 
		(temps - rc0011))) * (ep_2__ / pres1[k + 1]);
/* Computing MAX */
	r__1 = sqw[k] - rvls;
	rcldb = max(r__1,zero);
/* Computing MAX */
	r__1 = rcldb, r__2 = qc_bl__[k] + qi_bl__[k];
	rcldb = max(r__1,r__2);
	dthvx = thl[k + 2] + th1[k + 2] * p608 * sqw[k + 2] - (thl[k] + th1[k]
		 * p608 * sqw[k]);
	dz_ent__ = (dz1[k] * dz1[k + 1] + dz1[k + 1] * dz1[k]) / (dz1[k + 1] 
		+ dz1[k]) + (dz1[k + 1] * dz1[k + 2] + dz1[k + 2] * dz1[k + 1]
		) / (dz1[k + 1] + dz1[k + 2]);
	dthvx = dthvx * dz200 / dz_ent__;
	dthvx = max(dthvx,p1);
	tmp1 = xlvcp * rcldb / (ex1[k] * dthvx);
/* Computing MIN */
	r__1 = two, r__2 = p2 + p2 * eight * tmp1;
	*ent_eff__ = min(r__1,r__2);
	f0 = zero;
/* Computing MAX */
	i__1 = *kts, i__2 = kminrad - 1;
	i__3 = kminrad + 1;
	
#pragma unroll 4
for (kk = max(i__1,i__2); kk <= i__3; ++kk) {
	    radflux = rthraten[kk] * ex1[kk];
	    dp = rho1[k] * grav * dz1[k];
	    radflux = radflux * cp / grav * dp;
	    if (radflux < zero) {
		f0 = abs(radflux) + f0;
	    }
/* L9007: */
	}
/* L9008: */
	if (*xland - rc0013 >= zero) {
	    f0 = min(f0,rc0014);
	} else {
/* Computing MIN */
	    r__1 = p25 * f0;
	    f0 = min(r__1,rc0015);
	}
	*cldtop_cooling__ = abs(minrad) * rc0016;
	d__1 =  (grav / thv[k] * dz200 * f0 / (rho1[k] * cp));
	d__2 =  p333;
	wstar_rad__ = rc0017 * (real)pow_dd(&d__1, &d__2);
/* Computing MIN */
	r__1 = max(wstar_rad__,p1);
	wstar_rad__ = min(r__1,three);
/* Computing MAX */
	r__1 = thv[k + 1] - thv[k];
	dthvx = max(r__1,p1);
	/* unread private value */;
/* Computing MAX */
	r__1 = zl1, r__2 = zminrad - wstar_rad__ * rc0018;
	zb1 = max(r__1,r__2);
	i__3 = kminrad + 3;
	
#pragma unroll 4
for (k = *kts; k <= i__3; ++k) {
	    zagl = zw[k] + p5 * dz1[k];
	    if (k == *kts && kminrad == *kts) {
		zagl = rc0019 * dz1[*kts];
	    }
/* Computing MIN */
/* Computing MAX */
/* Computing MAX */
	    r__3 = zero, r__4 = zagl - zb1;
	    r__2 = one - max(r__3,r__4) / (zminrad - zl1);
	    r__1 = max(r__2,zfmin);
	    zfac[k - 1] = min(r__1,one);
/* Computing MAX */
	    r__1 = (zminrad - zagl) / zminrad;
/* Computing 3rd power */
	    r__2 = one - zfac[k - 1];
	    zfacent[k - 1] = max(r__1,zero) * (r__2 * (r__2 * r__2));
/* L9010: */
	}
/* L9011: */
	red001 = -rc0006;
	i__3 = kminrad + 3 - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__3; ++ixarr1) {
/* Computing MAX */
	    r__1 = red001, r__2 = zfacent[*kts + ixarr1 - 2];
	    red001 = max(r__1,r__2);
	}
	zfacent_max__ = red001;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= (1*(MYNN471_NZ-49)+49); ++ixarr1) {
	    zfacent[ixarr1 - 1] /= max(zfacent_max__,rc0020);
	}
	i__3 = kminrad + 3;
	
#pragma unroll 4
for (k = *kts; k <= i__3; ++k) {
/* Computing MAX */
/* Computing 3rd power */
	    r__3 = wstar_rad__;
	    r__1 = tkeprod_dn__[k], r__2 = p3 * *ent_eff__ * (r__3 * (r__3 * 
		    r__3)) / dz500 * zfacent[k - 1];
	    tkeprod_dn__[k] = max(r__1,r__2);
/* Computing MAX */
	    r__1 = tkeprod_dn__[k];
	    tkeprod_dn__[k] = max(r__1,zero);
/* Computing MIN */
	    r__1 = rc0021, r__2 = tkeprod_dn__[k];
	    tkeprod_dn__[k] = min(r__1,r__2);
/* L9013: */
	}
/* L9014: */
/* Computing MAX */
/* Computing MAX */
	i__3 = *kts, i__1 = kminrad - 1;
	r__1 = tkeprod_dn__[kminrad], r__2 = p333 * tkeprod_dn__[max(i__3,
		i__1)];
	tkeprod_dn__[kminrad] = max(r__1,r__2);
/* Computing MAX */
	r__1 = tkeprod_dn__[kminrad + 1], r__2 = min(p5,*ent_eff__) * 
		tkeprod_dn__[max(*kts,kminrad)];
	tkeprod_dn__[kminrad + 1] = max(r__1,r__2);
/* Computing MAX */
/* Computing 2nd power */
	r__2 = *u1;
/* Computing 2nd power */
	r__3 = *v1;
	r__1 = r__2 * r__2 + r__3 * r__3;
	wspd_pbl__ = (real)sqrt((real)(max(r__1,rc0004)));
/* Computing MIN */
/* Computing MAX */
	r__2 = zero, r__3 = wspd_pbl__ - rc0022;
	r__1 = max(r__2,r__3) / rc0023;
	ac_wsp__ = one - min(r__1,one);
	i__3 = *kte - *kts + 1;
	
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__3; ++ixarr1) {
	    tkeprod_dn__[*kts + ixarr1 - 1] *= min(ac_wsp__,*psig);
	}
    }
    red001 = -rc0006;
    i__3 = *kte - *kts + 1;
    
#pragma unroll 4
for (ixarr1 = 1; ixarr1 <= i__3; ++ixarr1) {
/* Computing MAX */
	r__1 = red001, r__2 = tkeprod_dn__[*kts + ixarr1 - 1];
	red001 = max(r__1,r__2);
    }
    *maxtkeprod = red001 * rc0016;
    if (FALSE_) {
	if (*xland - rc0013 >= zero) {
	} else {
	}
	i__3 = *kts;
	
#pragma unroll 4
for (k = kminrad + 3; k >= i__3; --k) {
/* L9016: */
	}
/* L9017: */
    }
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
M471Ptr<float> a_pres1={f32+(15*(MYNN471_NZ-49)+737)*n+c,n};
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
int a_ndvel;
M471Ptr<float> a_chem1={f32+(26*(MYNN471_NZ-49)+1288)*n+c,n};
M471Ptr<float> a_vdep={f32+(27*(MYNN471_NZ-49)+1337)*n+c,n};
float a_frp;
float a_emis_ant_no;
int a_mix_chem;
int a_enh_mix;
M471Ptr<float> a_settle1={f32+(27*(MYNN471_NZ-49)+1340)*n+c,n};
M471Ptr<float> a_scalars={f32+(28*(MYNN471_NZ-49)+1389)*n+c,n};
int a_nscalars;
M471Ptr<float> a_tsq1={f32+(29*(MYNN471_NZ-49)+1438)*n+c,n};
M471Ptr<float> a_qsq1={f32+(30*(MYNN471_NZ-49)+1487)*n+c,n};
M471Ptr<float> a_cov1={f32+(31*(MYNN471_NZ-49)+1536)*n+c,n};
M471Ptr<float> a_du1={f32+(32*(MYNN471_NZ-49)+1585)*n+c,n};
M471Ptr<float> a_dv1={f32+(33*(MYNN471_NZ-49)+1634)*n+c,n};
M471Ptr<float> a_dth1={f32+(34*(MYNN471_NZ-49)+1683)*n+c,n};
M471Ptr<float> a_dqv1={f32+(35*(MYNN471_NZ-49)+1732)*n+c,n};
M471Ptr<float> a_dqc1={f32+(36*(MYNN471_NZ-49)+1781)*n+c,n};
M471Ptr<float> a_dqi1={f32+(37*(MYNN471_NZ-49)+1830)*n+c,n};
M471Ptr<float> a_dqnc1={f32+(38*(MYNN471_NZ-49)+1879)*n+c,n};
M471Ptr<float> a_dqni1={f32+(39*(MYNN471_NZ-49)+1928)*n+c,n};
M471Ptr<float> a_dqs1={f32+(40*(MYNN471_NZ-49)+1977)*n+c,n};
M471Ptr<float> a_dqnwfa1={f32+(41*(MYNN471_NZ-49)+2026)*n+c,n};
M471Ptr<float> a_dqnifa1={f32+(42*(MYNN471_NZ-49)+2075)*n+c,n};
M471Ptr<float> a_dqnbca1={f32+(43*(MYNN471_NZ-49)+2124)*n+c,n};
M471Ptr<float> a_dozone1={f32+(44*(MYNN471_NZ-49)+2173)*n+c,n};
M471Ptr<float> a_rthraten1={f32+(45*(MYNN471_NZ-49)+2222)*n+c,n};
float a_pblh;
int a_kpbl;
float a_maxwidth_dd;
float a_maxwidth;
float a_maxmf;
float a_ztop_plume;
float a_excess_h;
float a_excess_q;
float a_maxmf_dd;
float a_maxtkeprod;
float a_cldtop_cooling;
float a_ent_eff;
M471Ptr<float> a_dqke1={f32+(46*(MYNN471_NZ-49)+2282)*n+c,n};
M471Ptr<float> a_qwt1={f32+(47*(MYNN471_NZ-49)+2331)*n+c,n};
M471Ptr<float> a_qshear1={f32+(48*(MYNN471_NZ-49)+2380)*n+c,n};
M471Ptr<float> a_qbuoy1={f32+(49*(MYNN471_NZ-49)+2429)*n+c,n};
M471Ptr<float> a_qdiss1={f32+(50*(MYNN471_NZ-49)+2478)*n+c,n};
M471Ptr<float> a_qc_bl1={f32+(51*(MYNN471_NZ-49)+2527)*n+c,n};
M471Ptr<float> a_qi_bl1={f32+(52*(MYNN471_NZ-49)+2576)*n+c,n};
M471Ptr<float> a_cldfra_bl1={f32+(53*(MYNN471_NZ-49)+2625)*n+c,n};
int a_bl_mynn_tkeadvect;
int a_tke_budget;
int a_bl_mynn_cloudpdf;
int a_bl_mynn_mixlength;
float a_bl_mynn_closure;
int a_bl_mynn_ess;
int a_bl_mynn_edmf;
int a_bl_mynn_edmf_mom;
int a_bl_mynn_edmf_tke;
int a_bl_mynn_mixscalars;
int a_bl_mynn_mixaerosols;
int a_bl_mynn_mixnumcon;
int a_bl_mynn_output;
int a_bl_mynn_cloudmix;
int a_bl_mynn_mixqt;
int a_bl_mynn_edmf_dd;
M471Ptr<float> a_edmf_a1={f32+(54*(MYNN471_NZ-49)+2675)*n+c,n};
M471Ptr<float> a_edmf_w1={f32+(55*(MYNN471_NZ-49)+2724)*n+c,n};
M471Ptr<float> a_edmf_qt1={f32+(56*(MYNN471_NZ-49)+2773)*n+c,n};
M471Ptr<float> a_edmf_thl1={f32+(57*(MYNN471_NZ-49)+2822)*n+c,n};
M471Ptr<float> a_edmf_ent1={f32+(58*(MYNN471_NZ-49)+2871)*n+c,n};
M471Ptr<float> a_edmf_qc1={f32+(59*(MYNN471_NZ-49)+2920)*n+c,n};
M471Ptr<float> a_sub_thl1={f32+(60*(MYNN471_NZ-49)+2969)*n+c,n};
M471Ptr<float> a_sub_sqv1={f32+(61*(MYNN471_NZ-49)+3018)*n+c,n};
M471Ptr<float> a_det_thl1={f32+(62*(MYNN471_NZ-49)+3067)*n+c,n};
M471Ptr<float> a_det_sqv1={f32+(63*(MYNN471_NZ-49)+3116)*n+c,n};
int a_spp_pbl;
M471Ptr<float> a_pattern_spp_pbl1={f32+(64*(MYNN471_NZ-49)+3165)*n+c,n};
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
a_ndvel=i32[6*n+c];




a_frp=f32[(27*(MYNN471_NZ-49)+1338)*n+c];
a_emis_ant_no=f32[(27*(MYNN471_NZ-49)+1339)*n+c];
a_mix_chem=i32[7*n+c];
a_enh_mix=i32[8*n+c];




a_nscalars=i32[9*n+c];


































a_pblh=f32[(46*(MYNN471_NZ-49)+2271)*n+c];
a_kpbl=i32[10*n+c];
a_maxwidth_dd=f32[(46*(MYNN471_NZ-49)+2272)*n+c];
a_maxwidth=f32[(46*(MYNN471_NZ-49)+2273)*n+c];
a_maxmf=f32[(46*(MYNN471_NZ-49)+2274)*n+c];
a_ztop_plume=f32[(46*(MYNN471_NZ-49)+2275)*n+c];
a_excess_h=f32[(46*(MYNN471_NZ-49)+2276)*n+c];
a_excess_q=f32[(46*(MYNN471_NZ-49)+2277)*n+c];
a_maxmf_dd=f32[(46*(MYNN471_NZ-49)+2278)*n+c];
a_maxtkeprod=f32[(46*(MYNN471_NZ-49)+2279)*n+c];
a_cldtop_cooling=f32[(46*(MYNN471_NZ-49)+2280)*n+c];
a_ent_eff=f32[(46*(MYNN471_NZ-49)+2281)*n+c];
















a_bl_mynn_tkeadvect=i32[11*n+c];
a_tke_budget=i32[12*n+c];
a_bl_mynn_cloudpdf=i32[13*n+c];
a_bl_mynn_mixlength=i32[14*n+c];
a_bl_mynn_closure=f32[(54*(MYNN471_NZ-49)+2674)*n+c];
a_bl_mynn_ess=i32[15*n+c];
a_bl_mynn_edmf=i32[16*n+c];
a_bl_mynn_edmf_mom=i32[17*n+c];
a_bl_mynn_edmf_tke=i32[18*n+c];
a_bl_mynn_mixscalars=i32[19*n+c];
a_bl_mynn_mixaerosols=i32[20*n+c];
a_bl_mynn_mixnumcon=i32[21*n+c];
a_bl_mynn_output=i32[22*n+c];
a_bl_mynn_cloudmix=i32[23*n+c];
a_bl_mynn_mixqt=i32[24*n+c];
a_bl_mynn_edmf_dd=i32[25*n+c];




















a_spp_pbl=i32[26*n+c];


a_flag_qc=i32[27*n+c];
a_flag_qi=i32[28*n+c];
a_flag_qnc=i32[29*n+c];
a_flag_qni=i32[30*n+c];
a_flag_qs=i32[31*n+c];
a_flag_qnwfa=i32[32*n+c];
a_flag_qnifa=i32[33*n+c];
a_flag_qnbca=i32[34*n+c];
a_flag_ozone=i32[35*n+c];
a_kts=i32[36*n+c];
a_kte=i32[37*n+c];
a_errflg=i32[38*n+c];
 mynnedmf_<M471_PHASE>(m471_work,n,&a_i,&a_j,&a_initflag,&a_restart,&a_cycling,&a_delt,a_dz1,&a_dx,a_u1,a_v1,a_w1,a_th1,a_sqv1,a_sqc1,a_sqi1,a_sqs1,a_qnc1,a_qni1,a_qnwfa1,a_qnifa1,a_qnbca1,a_ozone1,a_pres1,a_ex1,a_rho1,a_tk1,&a_xland,&a_ts,&a_qsfc,&a_ps,&a_ust,&a_ch,&a_hfx,&a_qfx,&a_znt,&a_wspd,&a_uoce,&a_voce,a_qke1,a_qke_adv1,a_el1,a_sh1,a_sm1,a_kh1,a_km1,&a_nchem,&a_ndvel,a_chem1,a_vdep,&a_frp,&a_emis_ant_no,&a_mix_chem,&a_enh_mix,a_settle1,a_scalars,&a_nscalars,a_tsq1,a_qsq1,a_cov1,a_du1,a_dv1,a_dth1,a_dqv1,a_dqc1,a_dqi1,a_dqnc1,a_dqni1,a_dqs1,a_dqnwfa1,a_dqnifa1,a_dqnbca1,a_dozone1,a_rthraten1,&a_pblh,&a_kpbl,&a_maxwidth_dd,&a_maxwidth,&a_maxmf,&a_ztop_plume,&a_excess_h,&a_excess_q,&a_maxmf_dd,&a_maxtkeprod,&a_cldtop_cooling,&a_ent_eff,a_dqke1,a_qwt1,a_qshear1,a_qbuoy1,a_qdiss1,a_qc_bl1,a_qi_bl1,a_cldfra_bl1,&a_bl_mynn_tkeadvect,&a_tke_budget,&a_bl_mynn_cloudpdf,&a_bl_mynn_mixlength,&a_bl_mynn_closure,&a_bl_mynn_ess,&a_bl_mynn_edmf,&a_bl_mynn_edmf_mom,&a_bl_mynn_edmf_tke,&a_bl_mynn_mixscalars,&a_bl_mynn_mixaerosols,&a_bl_mynn_mixnumcon,&a_bl_mynn_output,&a_bl_mynn_cloudmix,&a_bl_mynn_mixqt,&a_bl_mynn_edmf_dd,a_edmf_a1,a_edmf_w1,a_edmf_qt1,a_edmf_thl1,a_edmf_ent1,a_edmf_qc1,a_sub_thl1,a_sub_sqv1,a_det_thl1,a_det_sqv1,&a_spp_pbl,a_pattern_spp_pbl1,&a_flag_qc,&a_flag_qi,&a_flag_qnc,&a_flag_qni,&a_flag_qs,&a_flag_qnwfa,&a_flag_qnifa,&a_flag_qnbca,&a_flag_ozone,&a_kts,&a_kte,&a_errflg);






































































f32[(46*(MYNN471_NZ-49)+2271)*n+c]=a_pblh;
i32[10*n+c]=a_kpbl;
f32[(46*(MYNN471_NZ-49)+2272)*n+c]=a_maxwidth_dd;
f32[(46*(MYNN471_NZ-49)+2273)*n+c]=a_maxwidth;
f32[(46*(MYNN471_NZ-49)+2274)*n+c]=a_maxmf;
f32[(46*(MYNN471_NZ-49)+2275)*n+c]=a_ztop_plume;
f32[(46*(MYNN471_NZ-49)+2276)*n+c]=a_excess_h;
f32[(46*(MYNN471_NZ-49)+2277)*n+c]=a_excess_q;
f32[(46*(MYNN471_NZ-49)+2278)*n+c]=a_maxmf_dd;
f32[(46*(MYNN471_NZ-49)+2279)*n+c]=a_maxtkeprod;
f32[(46*(MYNN471_NZ-49)+2280)*n+c]=a_cldtop_cooling;
f32[(46*(MYNN471_NZ-49)+2281)*n+c]=a_ent_eff;




































i32[38*n+c]=a_errflg;
}
#if GPUWM_WRF_EXACT
#define M480_PHASE_LAUNCH(blocks)
#else
#define M480_PHASE_LAUNCH(blocks) __launch_bounds__(128,blocks)
#endif
extern "C" __global__ M471_LAUNCH void mynn471_step(int n,float*f32,int*i32,unsigned char*arena){m471_entry<-1>(n,f32,i32,arena);}
extern "C" __global__ M471_LAUNCH void mynn471_phase0(int n,float*f32,int*i32,unsigned char*arena){m471_entry<0>(n,f32,i32,arena);}
extern "C" __global__ M471_LAUNCH void mynn471_phase1(int n,float*f32,int*i32,unsigned char*arena){m471_entry<1>(n,f32,i32,arena);}
extern "C" __global__ M480_PHASE_LAUNCH(3) void mynn471_phase2(int n,float*f32,int*i32,unsigned char*arena){m471_entry<2>(n,f32,i32,arena);}
extern "C" __global__ M480_PHASE_LAUNCH(4) void mynn471_phase3(int n,float*f32,int*i32,unsigned char*arena){m471_entry<3>(n,f32,i32,arena);}
extern "C" __global__ M471_LAUNCH void mynn471_phase4(int n,float*f32,int*i32,unsigned char*arena){m471_entry<4>(n,f32,i32,arena);}
extern "C" __global__ M471_LAUNCH void mynn471_phase5(int n,float*f32,int*i32,unsigned char*arena){m471_entry<5>(n,f32,i32,arena);}
extern "C" __global__ M471_LAUNCH void mynn471_phase6(int n,float*f32,int*i32,unsigned char*arena){m471_entry<6>(n,f32,i32,arena);}
