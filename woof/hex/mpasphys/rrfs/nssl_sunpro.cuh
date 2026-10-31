// SunPro expm1/tanh CUDA lowering, preserved from the earlier NSSL replay.
// Full permissive notices are retained with each function. No IBM sine source.
#if !defined(__CUDACC_RTC__)
#include <stdint.h>
#else
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef int int32_t;
#endif
#ifndef DBL_MIN
#define DBL_MIN 0x1p-1022
#define NSSL_PROPOSAL_DEFINED_DBL_MIN 1
#endif
#define HIGH_HALF 1
#define LOW_HALF 0
#define LITTLE_ENDI 1
#define SECTION __device__ __noinline__
#define attribute_hidden
#define SET_RESTORE_ROUND_53BIT(x) ((void)0)
#define FE_TONEAREST 0
#define math_check_force_underflow(x) ((void)0)
#define math_force_eval(x) ((void)0)
#define __set_errno(x) ((void)0)
#define libm_alias_double(a,b)
#define GET_HIGH_WORD(h,x) do { union {double d; uint64_t u;} v; v.d=(x); (h)=(uint32_t)(v.u>>32); } while(0)
#define GET_LOW_WORD(l,x) do { union {double d; uint64_t u;} v; v.d=(x); (l)=(uint32_t)v.u; } while(0)
#define EXTRACT_WORDS(h,l,x) do {GET_HIGH_WORD(h,x); GET_LOW_WORD(l,x);} while(0)
#define SET_HIGH_WORD(x,h) do { union {double d; uint64_t u;} v; v.d=(x); v.u=(v.u&0xffffffffULL)|((uint64_t)(uint32_t)(h)<<32); (x)=v.d; } while(0)
namespace nssl_sunpro {
__device__ __forceinline__ double native_fma(double a,double b,double c) {
#ifdef __CUDACC__
return __fma_rn(a,b,c);
#else
return __builtin_fma(a,b,c);
#endif
}
typedef int int4;
typedef union { int4 i[2]; double x; double d; } mynumber;
namespace expm1_impl {
/* @(#)s_expm1.c 5.1 93/09/24 */
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
/* Modified by Naohiko Shimizu/Tokai University, Japan 1997/08/25,
   for performance improvement on pipelined processors.
 */

/* expm1(x)
 * Returns exp(x)-1, the exponential of x minus 1.
 *
 * Method
 *   1. Argument reduction:
 *	Given x, find r and integer k such that
 *
 *               x = k*ln2 + r,  |r| <= 0.5*ln2 ~ 0.34658
 *
 *      Here a correction term c will be computed to compensate
 *	the error in r when rounded to a floating-point number.
 *
 *   2. Approximating expm1(r) by a special rational function on
 *	the interval [0,0.34658]:
 *	Since
 *	    r*(exp(r)+1)/(exp(r)-1) = 2+ r^2/6 - r^4/360 + ...
 *	we define R1(r*r) by
 *	    r*(exp(r)+1)/(exp(r)-1) = 2+ r^2/6 * R1(r*r)
 *	That is,
 *	    R1(r**2) = 6/r *((exp(r)+1)/(exp(r)-1) - 2/r)
 *		     = 6/r * ( 1 + 2.0*(1/(exp(r)-1) - 1/r))
 *		     = 1 - r^2/60 + r^4/2520 - r^6/100800 + ...
 *      We use a special Reme algorithm on [0,0.347] to generate
 *	a polynomial of degree 5 in r*r to approximate R1. The
 *	maximum error of this polynomial approximation is bounded
 *	by 2**-61. In other words,
 *	    R1(z) ~ 1.0 + Q1*z + Q2*z**2 + Q3*z**3 + Q4*z**4 + Q5*z**5
 *	where	Q1  =  -1.6666666666666567384E-2,
 *		Q2  =   3.9682539681370365873E-4,
 *		Q3  =  -9.9206344733435987357E-6,
 *		Q4  =   2.5051361420808517002E-7,
 *		Q5  =  -6.2843505682382617102E-9;
 *	(where z=r*r, and the values of Q1 to Q5 are listed below)
 *	with error bounded by
 *	    |                  5           |     -61
 *	    | 1.0+Q1*z+...+Q5*z   -  R1(z) | <= 2
 *	    |                              |
 *
 *	expm1(r) = exp(r)-1 is then computed by the following
 *	specific way which minimize the accumulation rounding error:
 *			       2     3
 *			      r     r    [ 3 - (R1 + R1*r/2)  ]
 *	      expm1(r) = r + --- + --- * [--------------------]
 *			      2     2    [ 6 - r*(3 - R1*r/2) ]
 *
 *	To compensate the error in the argument reduction, we use
 *		expm1(r+c) = expm1(r) + c + expm1(r)*c
 *			   ~ expm1(r) + c + r*c
 *	Thus c+r*c will be added in as the correction terms for
 *	expm1(r+c). Now rearrange the term to avoid optimization
 *	screw up:
 *			(      2                                    2 )
 *			({  ( r    [ R1 -  (3 - R1*r/2) ]  )  }    r  )
 *	 expm1(r+c)~r - ({r*(--- * [--------------------]-c)-c} - --- )
 *			({  ( 2    [ 6 - r*(3 - R1*r/2) ]  )  }    2  )
 *                      (                                             )
 *
 *		   = r - E
 *   3. Scale back to obtain expm1(x):
 *	From step 1, we have
 *	   expm1(x) = either 2^k*[expm1(r)+1] - 1
 *		    = or     2^k*[expm1(r) + (1-2^-k)]
 *   4. Implementation notes:
 *	(A). To save one multiplication, we scale the coefficient Qi
 *	     to Qi*2^i, and replace z by (x^2)/2.
 *	(B). To achieve maximum accuracy, we compute expm1(x) by
 *	  (i)   if x < -56*ln2, return -1.0, (raise inexact if x!=inf)
 *	  (ii)  if k=0, return r-E
 *	  (iii) if k=-1, return 0.5*(r-E)-0.5
 *        (iv)	if k=1 if r < -0.25, return 2*((r+0.5)- E)
 *		       else	     return  1.0+2.0*(r-E);
 *	  (v)   if (k<-2||k>56) return 2^k(1-(E-r)) - 1 (or exp(x)-1)
 *	  (vi)  if k <= 20, return 2^k((1-2^-k)-(E-r)), else
 *	  (vii) return 2^k(1-((E+2^-k)-r))
 *
 * Special cases:
 *	expm1(INF) is INF, expm1(NaN) is NaN;
 *	expm1(-INF) is -1, and
 *	for finite argument, only expm1(0)=0 is exact.
 *
 * Accuracy:
 *	according to an error analysis, the error is always less than
 *	1 ulp (unit in the last place).
 *
 * Misc. info.
 *	For IEEE double
 *	    if x >  7.09782712893383973096e+02 then expm1(x) overflow
 *
 * Constants:
 * The hexadecimal values are the intended ones for the following
 * constants. The decimal values may be used, provided that the
 * compiler will convert from decimal to binary accurately enough
 * to produce the hexadecimal values shown.
 */

#define one Q[0]
static __device__ __constant__ const double
  huge = 1.0e+300,
  tiny = 1.0e-300,
  o_threshold = 7.09782712893383973096e+02,  /* 0x40862E42, 0xFEFA39EF */
  ln2_hi = 6.93147180369123816490e-01,       /* 0x3fe62e42, 0xfee00000 */
  ln2_lo = 1.90821492927058770002e-10,       /* 0x3dea39ef, 0x35793c76 */
  invln2 = 1.44269504088896338700e+00,       /* 0x3ff71547, 0x652b82fe */
/* scaled coefficients related to expm1 */
  Q[] = { 1.0, -3.33333333333331316428e-02, /* BFA11111 111110F4 */
	  1.58730158725481460165e-03, /* 3F5A01A0 19FE5585 */
	  -7.93650757867487942473e-05, /* BF14CE19 9EAADBB7 */
	  4.00821782732936239552e-06, /* 3ED0CFCA 86E65239 */
	  -2.01099218183624371326e-07 }; /* BE8AFDB7 6E09C32D */

#ifndef SECTION
# define SECTION
#endif

SECTION
double
__expm1 (double x)
{
  double y, hi, lo, c, t, e, hxs, hfx, r1, h2, h4, R1, R2, R3;
  int32_t k, xsb;
  uint32_t hx;

  GET_HIGH_WORD (hx, x);
  xsb = hx & 0x80000000;                /* sign bit of x */
  if (xsb == 0)
    y = x;
  else
    y = -x;                             /* y = |x| */
  hx &= 0x7fffffff;                     /* high word of |x| */

  /* filter out huge and non-finite argument */
  if (hx >= 0x4043687A)                         /* if |x|>=56*ln2 */
    {
      if (hx >= 0x40862E42)                     /* if |x|>=709.78... */
	{
	  if (hx >= 0x7ff00000)
	    {
	      uint32_t low;
	      GET_LOW_WORD (low, x);
	      if (((hx & 0xfffff) | low) != 0)
		return x + x;            /* NaN */
	      else
		return (xsb == 0) ? x : -1.0;    /* exp(+-inf)={inf,-1} */
	    }
	  if (x > o_threshold)
	    {
	      __set_errno (ERANGE);
	      return huge * huge;   /* overflow */
	    }
	}
      if (xsb != 0)      /* x < -56*ln2, return -1.0 with inexact */
	{
	  math_force_eval (x + tiny);           /* raise inexact */
	  return tiny - one;            /* return -1 */
	}
    }

  /* argument reduction */
  if (hx > 0x3fd62e42)                  /* if  |x| > 0.5 ln2 */
    {
      if (hx < 0x3FF0A2B2)              /* and |x| < 1.5 ln2 */
	{
	  if (xsb == 0)
	    {
	      hi = x - ln2_hi; lo = ln2_lo;  k = 1;
	    }
	  else
	    {
	      hi = x + ln2_hi; lo = -ln2_lo;  k = -1;
	    }
	}
      else
	{
	  k = invln2 * x + ((xsb == 0) ? 0.5 : -0.5);
	  t = k;
	  hi = native_fma(-t,ln2_hi,x);          /* t*ln2_hi is exact here */
	  lo = t * ln2_lo;
	}
      x = hi - lo;
      c = (hi - x) - lo;
    }
  else if (hx < 0x3c900000)             /* when |x|<2**-54, return x */
    {
      math_check_force_underflow (x);
      t = huge + x;     /* return x with inexact flags when x!=0 */
      return x - (t - (huge + x));
    }
  else
    k = 0;

  /* x is now in primary range */
  hfx = 0.5 * x;
  hxs = x * hfx;
  R1 = native_fma(hxs,Q[1],one); h2 = hxs * hxs;
  R2 = native_fma(hxs,Q[3],Q[2]); h4 = h2 * h2;
  R3 = native_fma(hxs,Q[5],Q[4]);
  r1 = native_fma(h4,R3,native_fma(h2,R2,R1));
  t = native_fma(-r1,hfx,3.0);
  e = hxs * ((r1 - t) / (native_fma(-x,t,6.0)));
  if (k == 0)
    return x - native_fma(x,e,-hxs);                   /* c is 0 */
  else
    {
      e = native_fma(x,e-c,-c);
      e -= hxs;
      if (k == -1)
	return native_fma(0.5,x-e,-0.5);
      if (k == 1)
	{
	  if (x < -0.25)
	    return -2.0 * (e - (x + 0.5));
	  else
	    return native_fma(2.0,x-e,one);
	}
      if (k <= -2 || k > 56)         /* suffice to return exp(x)-1 */
	{
	  uint32_t high;
	  y = one - (e - x);
	  GET_HIGH_WORD (high, y);
	  SET_HIGH_WORD (y, high + (k << 20));  /* add k to y's exponent */
	  return y - one;
	}
      t = one;
      if (k < 20)
	{
	  uint32_t high;
	  SET_HIGH_WORD (t, 0x3ff00000 - (0x200000 >> k));    /* t=1-2^-k */
	  y = t - (e - x);
	  GET_HIGH_WORD (high, y);
	  SET_HIGH_WORD (y, high + (k << 20));  /* add k to y's exponent */
	}
      else
	{
	  uint32_t high;
	  SET_HIGH_WORD (t, ((0x3ff - k) << 20));       /* 2^-k */
	  y = x - (e + t);
	  y += one;
	  GET_HIGH_WORD (high, y);
	  SET_HIGH_WORD (y, high + (k << 20));  /* add k to y's exponent */
	}
    }
  return y;
}
#ifndef __expm1
libm_alias_double (__expm1, expm1)
#endif

}
#undef one

namespace tanh_impl {
/* @(#)s_tanh.c 5.1 93/09/24 */
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

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: s_tanh.c,v 1.7 1995/05/10 20:48:22 jtc Exp $";
#endif

/* Tanh(x)
 * Return the Hyperbolic Tangent of x
 *
 * Method :
 *				       x    -x
 *				      e  - e
 *	0. tanh(x) is defined to be -----------
 *				       x    -x
 *				      e  + e
 *	1. reduce x to non-negative by tanh(-x) = -tanh(x).
 *	2.  0      <= x <= 2**-55 : tanh(x) := x*(one+x)
 *					        -t
 *	    2**-55 <  x <=  1     : tanh(x) := -----; t = expm1(-2x)
 *					       t + 2
 *						     2
 *	    1      <= x <=  22.0  : tanh(x) := 1-  ----- ; t=expm1(2x)
 *						   t + 2
 *	    22.0   <  x <= INF    : tanh(x) := 1.
 *
 * Special cases:
 *	tanh(NaN) is NaN;
 *	only tanh(0)=0 is exact for finite argument.
 */


static __device__ __constant__ const double one = 1.0, two = 2.0, tiny = 1.0e-300;

#ifndef SECTION
# define SECTION
#endif

SECTION
double
__tanh (double x)
{
  double t, z;
  int32_t jx, ix, lx;

  /* High word of |x|. */
  EXTRACT_WORDS (jx, lx, x);
  ix = jx & 0x7fffffff;

  /* x is INF or NaN */
  if (ix >= 0x7ff00000)
    {
      if (jx >= 0)
	return one / x + one;               /* tanh(+-inf)=+-1 */
      else
	return one / x - one;               /* tanh(NaN) = NaN */
    }

  /* |x| < 22 */
  if (ix < 0x40360000)                  /* |x|<22 */
    {
      if ((ix | lx) == 0)
	return x;                       /* x == +-0 */
      if (ix < 0x3c800000)              /* |x|<2**-55 */
	{
	  math_check_force_underflow (x);
	  return x * (one + x);           /* tanh(small) = small */
	}
      if (ix >= 0x3ff00000)             /* |x|>=1  */
	{
	  t = expm1_impl::__expm1 (two * fabs (x));
	  z = one - two / (t + two);
	}
      else
	{
	  t = expm1_impl::__expm1 (-two * fabs (x));
	  z = -t / (t + two);
	}
      /* |x| > 22, return +-1 */
    }
  else
    {
      z = one - tiny;                   /* raised inexact flag */
    }
  return (jx >= 0) ? z : -z;
}
libm_alias_double (__tanh, tanh)

}



}

#ifdef NSSL_PROPOSAL_DEFINED_DBL_MIN
#undef DBL_MIN
#undef NSSL_PROPOSAL_DEFINED_DBL_MIN
#endif
#undef HIGH_HALF
#undef LOW_HALF
#undef LITTLE_ENDI
#undef SECTION
#undef attribute_hidden
#undef SET_RESTORE_ROUND_53BIT
#undef FE_TONEAREST
#undef math_check_force_underflow
#undef math_force_eval
#undef __set_errno
#undef libm_alias_double
#undef GET_HIGH_WORD
#undef GET_LOW_WORD
#undef EXTRACT_WORDS
#undef SET_HIGH_WORD

__device__ double nssl_tanh(double x) { return nssl_sunpro::tanh_impl::__tanh(x); }
