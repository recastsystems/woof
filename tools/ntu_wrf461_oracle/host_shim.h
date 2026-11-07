// Host shim: lets the CUDA transcription and WOOF's own device math words
// compile as plain C++ on the CPU box, for transcription checks against the
// gfortran oracle.  Every shimmed intrinsic is the same IEEE operation.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__
static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __fsqrt_rn(float a) { return std::sqrt(a); }
static inline double __dadd_rn(double a, double b) { return a + b; }
static inline double __dsub_rn(double a, double b) { return a - b; }
static inline double __dmul_rn(double a, double b) { return a * b; }
static inline double __ddiv_rn(double a, double b) { return a / b; }
static inline double __dsqrt_rn(double a) { return std::sqrt(a); }
static inline double __fma_rn(double a, double b, double c) { return std::fma(a, b, c); }
static inline float __double2float_rn(double a) { return (float)a; }
static inline unsigned int __float_as_uint(float f) { unsigned int u; std::memcpy(&u, &f, 4); return u; }
static inline float __uint_as_float(unsigned int u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline float __int_as_float(int u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline long long __double_as_longlong(double d) { long long u; std::memcpy(&u, &d, 8); return u; }
static inline double __longlong_as_double(long long u) { double d; std::memcpy(&d, &u, 8); return d; }
static inline int __clzll(long long x) { return x == 0 ? 64 : __builtin_clzll((unsigned long long)x); }
static inline unsigned long long __umul64hi(unsigned long long a, unsigned long long b) {
    return (unsigned long long)(((unsigned __int128)a * b) >> 64);
}
