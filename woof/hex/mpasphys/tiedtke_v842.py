"""MPAS-A v8.4.2 Tiedtke convection, default REAL32, direct driver boundary.

Source: physics_wrf/module_cu_tiedtke.F:134-3086 and
mpas_atmphys_driver_convection.F:405-621,624-860,863-988.
The generated routines preserve source order and MPAS per-cell sigma levels.
The existing WOOF cu6 adapter and its WRF constants are not used.
"""
from __future__ import annotations

from woof.backend import compilation_target as _compilation_target
from woof.hex import numerics as _hex_numerics

import math
from functools import lru_cache
import numpy as np

SCHEME = "cu_tiedtke"
BACKEND = "hex-cuda"
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                   "--prec-div=true", "--prec-sqrt=true")
MPAS_SOURCE_SHA256 = "5dc29952181cafc12f12a9ed2df95512366f0085ffcf7e876a1d41301da1fd70"
INPUTS = ("u3d", "v3d", "w", "t3d", "qv3d", "qc3d", "qi3d", "pi3d",
          "rho3d", "qvften", "qvpblten", "dz8w", "pcps", "p8w", "xland", "qfx", "znu")
LEVEL_OUTPUTS = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten")
SURFACE_OUTPUTS = ("raincv", "pratec")

SUPPORT_SOURCE = r"""
// MPAS-A v8.4.2 Tiedtke column storage and rounded arithmetic. The transcendental calls
// use WOOF's own routines supplied by the kernel loader.
#pragma once
#ifndef TK_TRACE
#define TK_TRACE(line, value)
#endif

__device__ inline float tk_add(float a, float b) { return __fadd_rn(a, b); }
__device__ inline float tk_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ inline float tk_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ inline float tk_div(float a, float b) { return __fdiv_rn(a, b); }
template<class T> __device__ inline T tk_min(T a, T b) { return a < b ? a : b; }
template<class T> __device__ inline T tk_max(T a, T b) { return a > b ? a : b; }
__device__ inline float tk_abs(float a) { return fabsf(a); }
__device__ inline int tk_abs(int a) { return a < 0 ? -a : a; }
template<int N> __device__ inline float tk_powi(float a) {
    if constexpr (N == 1) return a;
    else if constexpr (N % 2 == 0) {
        float b = tk_powi<N / 2>(a);
        return tk_mul(b, b);
    } else return tk_mul(a, tk_powi<N - 1>(a));
}

// Each thread is one Fortran column. Array coordinates stay one-based.
// The second dimension is vertical and the storage is level-major, so
// adjacent CUDA lanes use adjacent words at every level.
template<class T> struct TkArray {
    T* data;
    int stride;
    int first;
    int count;
    int* status;
    __device__ T& at(int index) const {
        // MPAS module_cu_tiedtke.F:2329/2339-2341 writes level zero when
        // cloud top reaches level three. Preserve the original linear
        // offset, including the preceding slab word; strict sentinel
        // witnesses measure these writes rather than discarding them.
        return data[index * stride];
    }
    __device__ T& operator()(int i) const { return at(i - 1); }
    __device__ T& operator()(int i, int k) const {
        return at((k - 1) * first + i - 1);
    }
    __device__ TkArray<T> slice(int i) const {
        return {data + (i - 1) * stride, stride, first, count - i + 1, status};
    }
};
template<class A, class B> struct TkSame { static constexpr bool value = false; };
template<class A> struct TkSame<A, A> { static constexpr bool value = true; };

// Passed by value to each routine. A child starts after its parent's live
// arrays; sequential calls reuse the same child region. No local column
// arrays or per-thread fixed vertical ceiling are compiled.
struct TkArena {
    float* f;
    int* i;
    bool* b;
    int nf, ni, nb, stride;
    int* status;
    int* activity;
    bool force_continue;
    template<class T> __device__ TkArray<T> take(int n, int first = 1) {
        if constexpr (TkSame<T, float>::value) {
            TkArray<T> a = {f + nf * stride, stride, first, n, status};
            nf += n;
            return a;
        } else if constexpr (TkSame<T, int>::value) {
            TkArray<T> a = {i + ni * stride, stride, first, n, status};
            ni += n;
            return a;
        } else {
            TkArray<T> a = {b + nb * stride, stride, first, n, status};
            nb += n;
            return a;
        }
    }
};

"""

ROUTINES_SOURCE = r"""
// Transcribed from pinned MPAS-A v8.4.2 using the existing strict cu6 generator.

// MPAS physics_wrf/module_cu_tiedtke.F SHA256: 5dc29952181cafc12f12a9ed2df95512366f0085ffcf7e876a1d41301da1fd70

// Source line numbers refer to that exact file.

static constexpr float tk_rd = 0x1.1f00000000000p+8f;

static constexpr float tk_rv = 0x1.cd999a0000000p+8f;

static constexpr float tk_cpd = 0x1.f640000000000p+9f;

static constexpr float tk_alv = 0x1.312d000000000p+21f;

static constexpr float tk_als = 0x1.5be6800000000p+21f;

static constexpr float tk_alf = 0x1.55cc000000000p+18f;

static constexpr float tk_g = 0x1.39cc100000000p+3f;

static constexpr float tk_rcpd = 0x1.04f83a0000000p-10f;

static constexpr float tk_rhoh2o = 0x1.f400000000000p+9f;

static constexpr float tk_tmelt = 0x1.1128f60000000p+8f;

static constexpr float tk_t000 = 0x1.1126660000000p+8f;

static constexpr float tk_hgfr = 0x1.d24ccc0000000p+7f;

static constexpr float tk_zrg = 0x1.a1b2560000000p-4f;

static constexpr float tk_c1es = 0x1.3163d80000000p+9f;

static constexpr float tk_c2es = 0x1.7bc0b40000000p+8f;

static constexpr float tk_c3les = 0x1.144dd20000000p+4f;

static constexpr float tk_c4les = 0x1.1ee1480000000p+5f;

static constexpr float tk_c5les = 0x1.001ef00000000p+12f;

static constexpr float tk_c3ies = 0x1.5e00000000000p+4f;

static constexpr float tk_c4ies = 0x1.ea3d700000000p+2f;

static constexpr float tk_c5ies = 0x1.6afd000000000p+12f;

static constexpr float tk_vtmpc1 = 0x1.377b480000000p-1f;

static constexpr float tk_cevapcu1 = 0x1.aee9840000000p-16f;

static constexpr float tk_cevapcu2 = 0x1.6472260000000p+6f;

static constexpr float tk_entrpen = 0x1.a36e2e0000000p-14f;

static constexpr float tk_entrscv = 0x1.3a92a40000000p-10f;

static constexpr float tk_entrmid = 0x1.a36e2e0000000p-14f;

static constexpr float tk_entrdd = 0x1.a36e2e0000000p-13f;

static constexpr float tk_cmfctop = 0x1.3333340000000p-2f;

static constexpr float tk_cmfcmax = 0x1.0000000000000p+0f;

static constexpr float tk_cmfcmin = 0x1.b7cdfe0000000p-34f;

static constexpr float tk_cmfdeps = 0x1.3333340000000p-2f;

static constexpr float tk_cprcon = 0x1.d67e540000000p-14f;

static constexpr float tk_zdnoprc = 0x1.d4c0000000000p+13f;

static constexpr float tk_rhc = 0x1.99999a0000000p-1f;

static constexpr float tk_rhm = 0x1.0000000000000p+0f;

static constexpr float tk_zbuo0 = 0x1.0000000000000p-1f;

static constexpr float tk_crirh = 0x1.6666660000000p-1f;

static constexpr float tk_fdbk = 0x1.0000000000000p+0f;

static constexpr float tk_ztau = 0x1.2c00000000000p+11f;

static constexpr bool tk_lmfpen = true;

static constexpr bool tk_lmfmid = true;

static constexpr bool tk_lmfscv = true;

static constexpr bool tk_lmfdd = true;

static constexpr bool tk_lmfdudv = true;

__device__ void tk_tiecnv(TkArray<float> pu, TkArray<float> pv, TkArray<float> pt, TkArray<float> pqv, TkArray<float> pqc, TkArray<float> pqi, TkArray<float> pqvf, TkArray<float> pqvbl, TkArray<float> poz, TkArray<float> pomg, TkArray<float> pap, TkArray<float> paph, TkArray<float> evap, TkArray<float> zprecc, TkArray<int> lndj, TkArray<int> ktype, int lq, int km, int km1, TkArray<float> sig1, float dt, TkArena arena);

__device__ void tk_cumastr_new(int klon, int klev, int klevp1, int klevm1, TkArray<float> pten, TkArray<float> pqen, TkArray<float> puen, TkArray<float> pven, TkArray<float> pverv, TkArray<float> pqsen, TkArray<float> pqhfl, float ztmst, TkArray<float> pap, TkArray<float> paph, TkArray<float> pgeo, TkArray<float> ptte, TkArray<float> pqte, TkArray<float> pvom, TkArray<float> pvol, TkArray<float> prsfc, TkArray<float> pssfc, TkArray<float> paprc, TkArray<float> paprsm, TkArray<float> paprs, TkArray<bool> ldcum, TkArray<int> ktype, TkArray<int> kcbot, TkArray<int> kctop, TkArray<float> ptu, TkArray<float> pqu, TkArray<float> plu, TkArray<float> plude, TkArray<float> pmfu, TkArray<float> pmfd, TkArray<float> prain, float& psrain, float& psevap, float& psheat, float& psdiss, float& psmelt, TkArray<float> pcte, TkArray<float> sig1, TkArray<int> lndj, TkArena arena);

__device__ void tk_cuini(int klon, int klev, int klevp1, int klevm1, TkArray<float> pten, TkArray<float> pqen, TkArray<float> pqsen, TkArray<float> puen, TkArray<float> pven, TkArray<float> pverv, TkArray<float> pgeo, TkArray<float> paph, TkArray<float> pgeoh, TkArray<float> ptenh, TkArray<float> pqenh, TkArray<float> pqsenh, TkArray<int> klwmin, TkArray<float> ptu, TkArray<float> pqu, TkArray<float> ptd, TkArray<float> pqd, TkArray<float> puu, TkArray<float> pvu, TkArray<float> pud, TkArray<float> pvd, TkArray<float> pmfu, TkArray<float> pmfd, TkArray<float> pmfus, TkArray<float> pmfds, TkArray<float> pmfuq, TkArray<float> pmfdq, TkArray<float> pdmfup, TkArray<float> pdmfdp, TkArray<float> pdpmel, TkArray<float> plu, TkArray<float> plude, TkArray<int> klab, TkArena arena);

__device__ void tk_cubase(int klon, int klev, int klevp1, int klevm1, TkArray<float> ptenh, TkArray<float> pqenh, TkArray<float> pgeoh, TkArray<float> paph, TkArray<float> ptu, TkArray<float> pqu, TkArray<float> plu, TkArray<float> puen, TkArray<float> pven, TkArray<float> puu, TkArray<float> pvu, TkArray<bool> ldcum, TkArray<int> kcbot, TkArray<int> klab, TkArena arena);

__device__ void tk_cuasc_new(int klon, int klev, int klevp1, int klevm1, TkArray<float> ptenh, TkArray<float> pqenh, TkArray<float> puen, TkArray<float> pven, TkArray<float> pten, TkArray<float> pqen, TkArray<float> pqsen, TkArray<float> pgeo, TkArray<float> pgeoh, TkArray<float> pap, TkArray<float> paph, TkArray<float> pqte, TkArray<float> pverv, TkArray<int> klwmin, TkArray<bool> ldcum, TkArray<float> phcbase, TkArray<int> ktype, TkArray<int> klab, TkArray<float> ptu, TkArray<float> pqu, TkArray<float> plu, TkArray<float> puu, TkArray<float> pvu, TkArray<float> pmfu, TkArray<float> pmfub, TkArray<float> pentr, TkArray<float> pmfus, TkArray<float> pmfuq, TkArray<float> pmful, TkArray<float> plude, TkArray<float> pdmfup, TkArray<int> kcbot, TkArray<int> kctop, TkArray<int> kctop0, int& kcum, float ztmst, TkArray<int> khmin, TkArray<float> phhatt, TkArray<float> pqsenh, TkArena arena);

__device__ void tk_cudlfs(int klon, int klev, int klevp1, TkArray<float> ptenh, TkArray<float> pqenh, TkArray<float> puen, TkArray<float> pven, TkArray<float> pgeoh, TkArray<float> paph, TkArray<float> ptu, TkArray<float> pqu, TkArray<float> puu, TkArray<float> pvu, TkArray<bool> ldcum, TkArray<int> kcbot, TkArray<int> kctop, TkArray<float> pmfub, TkArray<float> prfl, TkArray<float> ptd, TkArray<float> pqd, TkArray<float> pud, TkArray<float> pvd, TkArray<float> pmfd, TkArray<float> pmfds, TkArray<float> pmfdq, TkArray<float> pdmfdp, TkArray<int> kdtop, TkArray<bool> lddraf, TkArena arena);

__device__ void tk_cuddraf(int klon, int klev, int klevp1, TkArray<float> ptenh, TkArray<float> pqenh, TkArray<float> puen, TkArray<float> pven, TkArray<float> pgeoh, TkArray<float> paph, TkArray<float> prfl, TkArray<bool> lddraf, TkArray<float> ptd, TkArray<float> pqd, TkArray<float> pud, TkArray<float> pvd, TkArray<float> pmfd, TkArray<float> pmfds, TkArray<float> pmfdq, TkArray<float> pdmfdp, TkArena arena);

__device__ void tk_cuflx(int klon, int klev, int klevp1, TkArray<float> pqen, TkArray<float> pqsen, TkArray<float> ptenh, TkArray<float> pqenh, TkArray<float> paph, TkArray<float> pgeoh, TkArray<int> kcbot, TkArray<int> kctop, TkArray<int> kdtop, TkArray<int> ktype, TkArray<bool> lddraf, TkArray<bool> ldcum, TkArray<float> pmfu, TkArray<float> pmfd, TkArray<float> pmfus, TkArray<float> pmfds, TkArray<float> pmfuq, TkArray<float> pmfdq, TkArray<float> pmful, TkArray<float> plude, TkArray<float> pdmfup, TkArray<float> pdmfdp, TkArray<float> prfl, TkArray<float> prain, TkArray<float> pten, TkArray<float> psfl, TkArray<float> pdpmel, int& ktopm2, float ztmst, TkArray<float> sig1, TkArena arena);

__device__ void tk_cudtdq(int klon, int klev, int klevp1, int ktopm2, TkArray<float> paph, TkArray<bool> ldcum, TkArray<float> pten, TkArray<float> ptte, TkArray<float> pqte, TkArray<float> pmfus, TkArray<float> pmfds, TkArray<float> pmfuq, TkArray<float> pmfdq, TkArray<float> pmful, TkArray<float> pdmfup, TkArray<float> pdmfdp, float ztmst, TkArray<float> pdpmel, TkArray<float> prain, TkArray<float> prfl, TkArray<float> psfl, float& psrain, float& psevap, float& psheat, float& psmelt, TkArray<float> prsfc, TkArray<float> pssfc, TkArray<float> paprc, TkArray<float> paprsm, TkArray<float> paprs, TkArray<float> pqen, TkArray<float> pqsen, TkArray<float> plude, TkArray<float> pcte, TkArena arena);

__device__ void tk_cududv(int klon, int klev, int klevp1, int ktopm2, TkArray<int> ktype, TkArray<int> kcbot, TkArray<float> paph, TkArray<bool> ldcum, TkArray<float> puen, TkArray<float> pven, TkArray<float> pvom, TkArray<float> pvol, TkArray<float> puu, TkArray<float> pud, TkArray<float> pvu, TkArray<float> pvd, TkArray<float> pmfu, TkArray<float> pmfd, float& psdiss, TkArena arena);

__device__ void tk_cubasmc(int klon, int klev, int klevm1, int kk, TkArray<float> pten, TkArray<float> pqen, TkArray<float> pqsen, TkArray<float> puen, TkArray<float> pven, TkArray<float> pverv, TkArray<float> pgeo, TkArray<float> pgeoh, TkArray<bool> ldcum, TkArray<int> ktype, TkArray<int> klab, TkArray<float> pmfu, TkArray<float> pmfub, TkArray<float> pentr, TkArray<int> kcbot, TkArray<float> ptu, TkArray<float> pqu, TkArray<float> plu, TkArray<float> puu, TkArray<float> pvu, TkArray<float> pmfus, TkArray<float> pmfuq, TkArray<float> pmful, TkArray<float> pdmfup, TkArray<float> pmfuu, TkArray<float> pmfuv, TkArena arena);

__device__ void tk_cuadjtq(int klon, int klev, int kk, TkArray<float> pp, TkArray<float> pt, TkArray<float> pq, TkArray<bool> ldflag, int kcall, TkArena arena);

__device__ void tk_cuentr_new(int klon, int klev, int klevp1, int kk, TkArray<float> ptenh, TkArray<float> paph, TkArray<float> pap, TkArray<float> pgeoh, TkArray<int> klwmin, TkArray<bool> ldcum, TkArray<int> ktype, TkArray<int> kcbot, TkArray<int> kctop0, TkArray<float> zpbase, TkArray<float> pmfu, TkArray<float> pentr, TkArray<float> zdmfen, TkArray<float> zdmfde, TkArray<float> zodetr, TkArray<int> khmin, TkArena arena);

__device__ float tk_ssum(int n, TkArray<float> x, int ix, TkArena arena);

__device__ float tk_tlucua(float tt, TkArena arena);

__device__ float tk_tlucub(float tt, TkArena arena);

__device__ float tk_tlucuc(float tt, TkArena arena);

__device__ void tk_tiecnv(TkArray<float> pu, TkArray<float> pv, TkArray<float> pt, TkArray<float> pqv, TkArray<float> pqc, TkArray<float> pqi, TkArray<float> pqvf, TkArray<float> pqvbl, TkArray<float> poz, TkArray<float> pomg, TkArray<float> pap, TkArray<float> paph, TkArray<float> evap, TkArray<float> zprecc, TkArray<int> lndj, TkArray<int> ktype, int lq, int km, int km1, TkArray<float> sig1, float dt, TkArena arena) {
    // module_cu_tiedtke.F:556
    TkArray<float> pum1 = arena.take<float>((lq) * (km), lq);
    TkArray<float> pvm1 = arena.take<float>((lq) * (km), lq);
    TkArray<float> ptte = arena.take<float>((lq) * (km), lq);
    TkArray<float> pqte = arena.take<float>((lq) * (km), lq);
    TkArray<float> pvom = arena.take<float>((lq) * (km), lq);
    TkArray<float> pvol = arena.take<float>((lq) * (km), lq);
    TkArray<float> pverv = arena.take<float>((lq) * (km), lq);
    TkArray<float> pgeo = arena.take<float>((lq) * (km), lq);
    TkArray<float> pqhfl = arena.take<float>((lq), lq);
    TkArray<float> zqq = arena.take<float>((lq) * (km), lq);
    TkArray<float> paprc = arena.take<float>((lq), lq);
    TkArray<float> paprs = arena.take<float>((lq), lq);
    TkArray<float> prsfc = arena.take<float>((lq), lq);
    TkArray<float> pssfc = arena.take<float>((lq), lq);
    TkArray<float> paprsm = arena.take<float>((lq), lq);
    TkArray<float> pcte = arena.take<float>((lq) * (km), lq);
    TkArray<float> ztp1 = arena.take<float>((lq) * (km), lq);
    TkArray<float> zqp1 = arena.take<float>((lq) * (km), lq);
    TkArray<float> ztu = arena.take<float>((lq) * (km), lq);
    TkArray<float> zqu = arena.take<float>((lq) * (km), lq);
    TkArray<float> zlu = arena.take<float>((lq) * (km), lq);
    TkArray<float> zlude = arena.take<float>((lq) * (km), lq);
    TkArray<float> zmfu = arena.take<float>((lq) * (km), lq);
    TkArray<float> zmfd = arena.take<float>((lq) * (km), lq);
    TkArray<float> zqsat = arena.take<float>((lq) * (km), lq);
    TkArray<float> zrain = arena.take<float>((lq), lq);
    TkArray<float> sig = arena.take<float>((km1), km1);
    TkArray<int> icbot = arena.take<int>((lq), lq);
    TkArray<int> ictop = arena.take<int>((lq), lq);
    TkArray<bool> locum = arena.take<bool>((lq), lq);
    float psheat;
    float psrain;
    float psevap;
    float psmelt;
    float psdiss;
    float tt;
    float ztmst;
    float ztpp1;
    float fliq;
    float fice;
    float ztc;
    float zalf;
    int i;
    int j;
    int k;
    int lp;
    // F:594
    ztmst = dt; TK_TRACE(594, ztmst);
    // F:597
    psheat = 0.0f; TK_TRACE(597, psheat);
    // F:598
    psrain = 0.0f; TK_TRACE(598, psrain);
    // F:599
    psevap = 0.0f; TK_TRACE(599, psevap);
    // F:600
    psmelt = 0.0f; TK_TRACE(600, psmelt);
    // F:601
    psdiss = 0.0f; TK_TRACE(601, psdiss);
    // F:602
    const int tk_limit_602 = lq;
    for (j = 1; j <= tk_limit_602; j += 1) {
        // F:603
        zrain(j) = 0.0f; TK_TRACE(603, zrain(j));
        // F:604
        locum(j) = false;
        // F:605
        prsfc(j) = 0.0f; TK_TRACE(605, prsfc(j));
        // F:606
        pssfc(j) = 0.0f; TK_TRACE(606, pssfc(j));
        // F:607
        paprc(j) = 0.0f; TK_TRACE(607, paprc(j));
        // F:608
        paprs(j) = 0.0f; TK_TRACE(608, paprs(j));
        // F:609
        paprsm(j) = 0.0f; TK_TRACE(609, paprsm(j));
        // F:610
        pqhfl(j) = evap(j); TK_TRACE(610, pqhfl(j));
    // F:611
    }
    // F:615
    const int tk_limit_615 = km;
    for (k = 1; k <= tk_limit_615; k += 1) {
        // F:616
        const int tk_limit_616 = lq;
        for (j = 1; j <= tk_limit_616; j += 1) {
            // F:617
            ptte(j, k) = 0.0f; TK_TRACE(617, ptte(j, k));
            // F:618
            pcte(j, k) = 0.0f; TK_TRACE(618, pcte(j, k));
            // F:619
            pvom(j, k) = 0.0f; TK_TRACE(619, pvom(j, k));
            // F:620
            pvol(j, k) = 0.0f; TK_TRACE(620, pvol(j, k));
            // F:621
            ztp1(j, k) = pt(j, k); TK_TRACE(621, ztp1(j, k));
            // F:622
            zqp1(j, k) = tk_div(pqv(j, k), tk_add(1.0f, pqv(j, k))); TK_TRACE(622, zqp1(j, k));
            // F:623
            pum1(j, k) = pu(j, k); TK_TRACE(623, pum1(j, k));
            // F:624
            pvm1(j, k) = pv(j, k); TK_TRACE(624, pvm1(j, k));
            // F:625
            pverv(j, k) = pomg(j, k); TK_TRACE(625, pverv(j, k));
            // F:626
            pgeo(j, k) = tk_mul(tk_g, poz(j, k)); TK_TRACE(626, pgeo(j, k));
            // F:627
            tt = ztp1(j, k); TK_TRACE(627, tt);
            // F:628
            zqsat(j, k) = tk_div(tk_tlucua(tt, arena), pap(j, k)); TK_TRACE(628, zqsat(j, k));
            // F:629
            zqsat(j, k) = tk_min(0.5f, zqsat(j, k)); TK_TRACE(629, zqsat(j, k));
            // F:630
            zqsat(j, k) = tk_div(zqsat(j, k), tk_sub(1.f, tk_mul(tk_vtmpc1, zqsat(j, k)))); TK_TRACE(630, zqsat(j, k));
            // F:631
            pqte(j, k) = tk_add(pqvf(j, k), pqvbl(j, k)); TK_TRACE(631, pqte(j, k));
            // F:632
            zqq(j, k) = pqte(j, k); TK_TRACE(632, zqq(j, k));
        // F:633
        }
    // F:634
    }
    // F:639
    tk_cumastr_new(lq, km, km1, (km - 1), ztp1, zqp1, pum1, pvm1, pverv, zqsat, pqhfl, ztmst, pap, paph, pgeo, ptte, pqte, pvom, pvol, prsfc, pssfc, paprc, paprsm, paprs, locum, ktype, icbot, ictop, ztu, zqu, zlu, zlude, zmfu, zmfd, zrain, psrain, psevap, psheat, psdiss, psmelt, pcte, sig1, lndj, arena);
    // F:652
    if ((tk_fdbk >= 1.0e-9f)) {
        // F:653
        const int tk_limit_653 = km;
        for (k = 1; k <= tk_limit_653; k += 1) {
            // F:654
            const int tk_limit_654 = lq;
            for (j = 1; j <= tk_limit_654; j += 1) {
                // F:655
                if ((pcte(j, k) > 0.0f)) {
                    // F:656
                    ztpp1 = tk_add(pt(j, k), tk_mul(ptte(j, k), ztmst)); TK_TRACE(656, ztpp1);
                    // F:657
                    if ((ztpp1 >= tk_t000)) {
                        // F:658
                        fliq = 1.0f; TK_TRACE(658, fliq);
                        // F:659
                        zalf = 0.0f; TK_TRACE(659, zalf);
                    // F:660
                    } else if ((ztpp1 <= tk_hgfr)) {
                        // F:661
                        fliq = 0.0f; TK_TRACE(661, fliq);
                        // F:662
                        zalf = tk_alf; TK_TRACE(662, zalf);
                    // F:663
                    } else {
                        // F:664
                        ztc = tk_sub(ztpp1, tk_t000); TK_TRACE(664, ztc);
                        // F:665
                        fliq = tk_add(0.0059f, tk_mul(0.9941f, gfk_exp(tk_mul(tk_mul((-0.003102f), ztc), ztc)))); TK_TRACE(665, fliq);
                        // F:666
                        zalf = tk_alf; TK_TRACE(666, zalf);
                    // F:667
                    }
                    // F:668
                    fice = tk_sub(1.0f, fliq); TK_TRACE(668, fice);
                    // F:669
                    pqc(j, k) = tk_add(pqc(j, k), tk_mul(tk_mul(fliq, pcte(j, k)), ztmst)); TK_TRACE(669, pqc(j, k));
                    // F:670
                    pqi(j, k) = tk_add(pqi(j, k), tk_mul(tk_mul(fice, pcte(j, k)), ztmst)); TK_TRACE(670, pqi(j, k));
                    // F:671
                    ptte(j, k) = tk_sub(ptte(j, k), tk_mul(tk_mul(tk_mul(zalf, tk_rcpd), fliq), pcte(j, k))); TK_TRACE(671, ptte(j, k));
                // F:672
                }
            // F:673
            }
        // F:674
        }
    // F:675
    }
    // F:677
    const int tk_limit_677 = km;
    for (k = 1; k <= tk_limit_677; k += 1) {
        // F:678
        const int tk_limit_678 = lq;
        for (j = 1; j <= tk_limit_678; j += 1) {
            // F:679
            pt(j, k) = tk_add(ztp1(j, k), tk_mul(ptte(j, k), ztmst)); TK_TRACE(679, pt(j, k));
            // F:680
            zqp1(j, k) = tk_add(zqp1(j, k), tk_mul(tk_sub(pqte(j, k), zqq(j, k)), ztmst)); TK_TRACE(680, zqp1(j, k));
            // F:681
            pqv(j, k) = tk_div(zqp1(j, k), tk_sub(1.0f, zqp1(j, k))); TK_TRACE(681, pqv(j, k));
        // F:682
        }
    // F:683
    }
    // F:684
    const int tk_limit_684 = lq;
    for (j = 1; j <= tk_limit_684; j += 1) {
        // F:685
        zprecc(j) = tk_max(0.0f, tk_mul(tk_add(prsfc(j), pssfc(j)), ztmst)); TK_TRACE(685, zprecc(j));
    // F:686
    }
    // F:687
    if (tk_lmfdudv) {
        // F:688
        const int tk_limit_688 = km;
        for (k = 1; k <= tk_limit_688; k += 1) {
            // F:689
            const int tk_limit_689 = lq;
            for (j = 1; j <= tk_limit_689; j += 1) {
                // F:690
                pu(j, k) = tk_add(pu(j, k), tk_mul(pvom(j, k), ztmst)); TK_TRACE(690, pu(j, k));
                // F:691
                pv(j, k) = tk_add(pv(j, k), tk_mul(pvol(j, k), ztmst)); TK_TRACE(691, pv(j, k));
            // F:692
            }
        // F:693
        }
    // F:694
    }
    // F:696
    return;
}

__device__ void tk_cumastr_new(int klon, int klev, int klevp1, int klevm1, TkArray<float> pten, TkArray<float> pqen, TkArray<float> puen, TkArray<float> pven, TkArray<float> pverv, TkArray<float> pqsen, TkArray<float> pqhfl, float ztmst, TkArray<float> pap, TkArray<float> paph, TkArray<float> pgeo, TkArray<float> ptte, TkArray<float> pqte, TkArray<float> pvom, TkArray<float> pvol, TkArray<float> prsfc, TkArray<float> pssfc, TkArray<float> paprc, TkArray<float> paprsm, TkArray<float> paprs, TkArray<bool> ldcum, TkArray<int> ktype, TkArray<int> kcbot, TkArray<int> kctop, TkArray<float> ptu, TkArray<float> pqu, TkArray<float> plu, TkArray<float> plude, TkArray<float> pmfu, TkArray<float> pmfd, TkArray<float> prain, float& psrain, float& psevap, float& psheat, float& psdiss, float& psmelt, TkArray<float> pcte, TkArray<float> sig1, TkArray<int> lndj, TkArena arena) {
    // module_cu_tiedtke.F:707
    float zcons2;
    int jk;
    int jl;
    int ikb;
    float zqumqe;
    float zdqmin;
    float zmfmax;
    float zalvdcp;
    float zqalv;
    float zhsat;
    float zgam;
    float zzz;
    float zhhat;
    float zbi;
    float zro;
    float zdz;
    float zdhdz;
    float zdepth;
    float zfac;
    float zrh;
    float zpbmpt;
    float dept;
    float zht;
    float zeps;
    int icum;
    int itopm2;
    TkArray<float> ztenh = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zqenh = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zgeoh = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zqsenh = arena.take<float>((klon) * (klev), klon);
    TkArray<float> ztd = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zqd = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zmfus = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zmfds = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zmfuq = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zmfdq = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zdmfup = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zdmfdp = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zmful = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zrfl = arena.take<float>((klon), klon);
    TkArray<float> zuu = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zvu = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zud = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zvd = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zentr = arena.take<float>((klon), klon);
    TkArray<float> zhcbase = arena.take<float>((klon), klon);
    TkArray<float> zmfub = arena.take<float>((klon), klon);
    TkArray<float> zmfub1 = arena.take<float>((klon), klon);
    TkArray<float> zdqpbl = arena.take<float>((klon), klon);
    TkArray<float> zdqcv = arena.take<float>((klon), klon);
    TkArray<float> zsfl = arena.take<float>((klon), klon);
    TkArray<float> zdpmel = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zcape = arena.take<float>((klon), klon);
    TkArray<float> zheat = arena.take<float>((klon), klon);
    TkArray<float> zhhatt = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zhmin = arena.take<float>((klon), klon);
    TkArray<float> zrelh = arena.take<float>((klon), klon);
    TkArray<int> ilab = arena.take<int>((klon) * (klev), klon);
    TkArray<int> idtop = arena.take<int>((klon), klon);
    TkArray<int> ictop0 = arena.take<int>((klon), klon);
    TkArray<int> ilwmin = arena.take<int>((klon), klon);
    TkArray<int> ihmin = arena.take<int>((klon), klon);
    int ktop0;
    TkArray<bool> loddraf = arena.take<bool>((klon), klon);
    bool llo1;
    // F:848
    zcons2 = tk_div(1.f, tk_mul(tk_g, ztmst)); TK_TRACE(848, zcons2);
    // F:852
    tk_cuini(klon, klev, klevp1, klevm1, pten, pqen, pqsen, puen, pven, pverv, pgeo, paph, zgeoh, ztenh, zqenh, zqsenh, ilwmin, ptu, pqu, ztd, zqd, zuu, zvu, zud, zvd, pmfu, pmfd, zmfus, zmfds, zmfuq, zmfdq, zdmfup, zdmfdp, zdpmel, plu, plude, ilab, arena);
    // F:866
    tk_cubase(klon, klev, klevp1, klevm1, ztenh, zqenh, zgeoh, paph, ptu, pqu, plu, puen, pven, zuu, zvu, ldcum, kcbot, ilab, arena);
    // F:874
    jk = 1;
    // F:875
    const int tk_limit_875 = klon;
    for (jl = 1; jl <= tk_limit_875; jl += 1) {
        // F:876
        zdqcv(jl) = tk_mul(pqte(jl, jk), tk_sub(paph(jl, (jk + 1)), paph(jl, jk))); TK_TRACE(876, zdqcv(jl));
        // F:877
        zdqpbl(jl) = 0.0f; TK_TRACE(877, zdqpbl(jl));
        // F:878
        idtop(jl) = 0;
    // F:879
    }
    // F:881
    const int tk_limit_881 = klev;
    for (jk = 2; jk <= tk_limit_881; jk += 1) {
        // F:882
        const int tk_limit_882 = klon;
        for (jl = 1; jl <= tk_limit_882; jl += 1) {
            // F:883
            zdqcv(jl) = tk_add(zdqcv(jl), tk_mul(pqte(jl, jk), tk_sub(paph(jl, (jk + 1)), paph(jl, jk)))); TK_TRACE(883, zdqcv(jl));
            // F:884
            if ((jk >= kcbot(jl))) { zdqpbl(jl) = tk_add(zdqpbl(jl), tk_mul(pqte(jl, jk), tk_sub(paph(jl, (jk + 1)), paph(jl, jk)))); TK_TRACE(884, zdqpbl(jl)); }
        // F:886
        }
    // F:887
    }
    // F:889
    const int tk_limit_889 = klon;
    for (jl = 1; jl <= tk_limit_889; jl += 1) {
        // F:890
        ktype(jl) = 0;
        // F:891
        if ((zdqcv(jl) > tk_max(0.f, tk_mul(tk_mul(1.1f, pqhfl(jl)), tk_g)))) {
            // F:892
            ktype(jl) = 1;
        // F:893
        } else {
            // F:894
            ktype(jl) = 2;
        // F:895
        }
        // F:901
        ikb = kcbot(jl);
        // F:902
        zqumqe = tk_sub(tk_add(pqu(jl, ikb), plu(jl, ikb)), zqenh(jl, ikb)); TK_TRACE(902, zqumqe);
        // F:903
        zdqmin = tk_max(tk_mul(0.01f, zqenh(jl, ikb)), 1.e-10f); TK_TRACE(903, zdqmin);
        // F:904
        if ((((zdqpbl(jl) > 0.f) & (zqumqe > zdqmin)) & ldcum(jl))) {
            // F:905
            zmfub(jl) = tk_div(zdqpbl(jl), tk_mul(tk_g, tk_max(zqumqe, zdqmin))); TK_TRACE(905, zmfub(jl));
        // F:906
        } else {
            // F:907
            zmfub(jl) = 0.01f; TK_TRACE(907, zmfub(jl));
            // F:908
            ldcum(jl) = false;
        // F:909
        }
        // F:910
        zmfmax = tk_mul(tk_sub(paph(jl, ikb), paph(jl, (ikb - 1))), zcons2); TK_TRACE(910, zmfmax);
        // F:911
        zmfub(jl) = tk_min(zmfub(jl), zmfmax); TK_TRACE(911, zmfub(jl));
        // F:919
        ikb = kcbot(jl);
        // F:920
        zhcbase(jl) = tk_add(tk_add(tk_mul(tk_cpd, ptu(jl, ikb)), zgeoh(jl, ikb)), tk_mul(tk_alv, pqu(jl, ikb))); TK_TRACE(920, zhcbase(jl));
        // F:921
        ictop0(jl) = (kcbot(jl) - 1);
    // F:922
    }
    // F:924
    zalvdcp = tk_div(tk_alv, tk_cpd); TK_TRACE(924, zalvdcp);
    // F:925
    zqalv = tk_div(1.f, tk_alv); TK_TRACE(925, zqalv);
    // F:926
    const int tk_limit_926 = 3;
    for (jk = klevm1; jk >= tk_limit_926; jk += (-1)) {
        // F:927
        const int tk_limit_927 = klon;
        for (jl = 1; jl <= tk_limit_927; jl += 1) {
            // F:928
            zhsat = tk_add(tk_add(tk_mul(tk_cpd, ztenh(jl, jk)), zgeoh(jl, jk)), tk_mul(tk_alv, zqsenh(jl, jk))); TK_TRACE(928, zhsat);
            // F:929
            zgam = tk_div(tk_mul(tk_mul(tk_c5les, zalvdcp), zqsenh(jl, jk)), tk_mul(tk_sub(1.f, tk_mul(tk_vtmpc1, zqsenh(jl, jk))), tk_mul(tk_sub(ztenh(jl, jk), tk_c4les), tk_sub(ztenh(jl, jk), tk_c4les)))); TK_TRACE(929, zgam);
            // F:931
            zzz = tk_mul(tk_mul(tk_cpd, ztenh(jl, jk)), 0.608f); TK_TRACE(931, zzz);
            // F:932
            zhhat = tk_sub(zhsat, tk_mul(tk_div(tk_add(zzz, tk_mul(zgam, zzz)), tk_add(1.f, tk_mul(tk_mul(zgam, zzz), zqalv))), tk_max(tk_sub(zqsenh(jl, jk), zqenh(jl, jk)), 0.f))); TK_TRACE(932, zhhat);
            // F:934
            zhhatt(jl, jk) = zhhat; TK_TRACE(934, zhhatt(jl, jk));
            // F:935
            if (((jk < ictop0(jl)) & (zhcbase(jl) > zhhat))) { ictop0(jl) = jk; }
        // F:936
        }
    // F:937
    }
    // F:939
    const int tk_limit_939 = klon;
    for (jl = 1; jl <= tk_limit_939; jl += 1) {
        // F:940
        jk = kcbot(jl);
        // F:941
        zhsat = tk_add(tk_add(tk_mul(tk_cpd, ztenh(jl, jk)), zgeoh(jl, jk)), tk_mul(tk_alv, zqsenh(jl, jk))); TK_TRACE(941, zhsat);
        // F:942
        zgam = tk_div(tk_mul(tk_mul(tk_c5les, zalvdcp), zqsenh(jl, jk)), tk_mul(tk_sub(1.f, tk_mul(tk_vtmpc1, zqsenh(jl, jk))), tk_mul(tk_sub(ztenh(jl, jk), tk_c4les), tk_sub(ztenh(jl, jk), tk_c4les)))); TK_TRACE(942, zgam);
        // F:944
        zzz = tk_mul(tk_mul(tk_cpd, ztenh(jl, jk)), 0.608f); TK_TRACE(944, zzz);
        // F:945
        zhhat = tk_sub(zhsat, tk_mul(tk_div(tk_add(zzz, tk_mul(zgam, zzz)), tk_add(1.f, tk_mul(tk_mul(zgam, zzz), zqalv))), tk_max(tk_sub(zqsenh(jl, jk), zqenh(jl, jk)), 0.f))); TK_TRACE(945, zhhat);
        // F:947
        zhhatt(jl, jk) = zhhat; TK_TRACE(947, zhhatt(jl, jk));
    // F:948
    }
    // F:952
    const int tk_limit_952 = klon;
    for (jl = 1; jl <= tk_limit_952; jl += 1) {
        // F:953
        zhmin(jl) = 0.f; TK_TRACE(953, zhmin(jl));
        // F:954
        if ((ldcum(jl) & (ktype(jl) == 1))) {
            // F:955
            ihmin(jl) = kcbot(jl);
        // F:956
        } else {
            // F:957
            ihmin(jl) = (-1);
        // F:958
        }
    // F:959
    }
    // F:961
    zbi = tk_div(1.f, tk_mul(25.f, tk_g)); TK_TRACE(961, zbi);
    // F:962
    const int tk_limit_962 = 1;
    for (jk = klev; jk >= tk_limit_962; jk += (-1)) {
        // F:963
        const int tk_limit_963 = klon;
        for (jl = 1; jl <= tk_limit_963; jl += 1) {
            // F:964
            llo1 = ((ldcum(jl) & (ktype(jl) == 1)) & (ihmin(jl) == kcbot(jl)));
            // F:965
            if (((llo1 & (jk < kcbot(jl))) & (jk >= ictop0(jl)))) {
                // F:966
                ikb = kcbot(jl);
                // F:967
                zro = tk_div(tk_mul(tk_rd, ztenh(jl, jk)), tk_mul(tk_g, paph(jl, jk))); TK_TRACE(967, zro);
                // F:968
                zdz = tk_mul(tk_sub(paph(jl, jk), paph(jl, (jk - 1))), zro); TK_TRACE(968, zdz);
                // F:969
                zdhdz = tk_div(tk_mul(tk_add(tk_add(tk_mul(tk_cpd, tk_sub(pten(jl, (jk - 1)), pten(jl, jk))), tk_mul(tk_alv, tk_sub(pqen(jl, (jk - 1)), pqen(jl, jk)))), tk_sub(pgeo(jl, (jk - 1)), pgeo(jl, jk))), tk_g), tk_sub(pgeo(jl, (jk - 1)), pgeo(jl, jk))); TK_TRACE(969, zdhdz);
                // F:972
                zdepth = tk_sub(zgeoh(jl, jk), zgeoh(jl, ikb)); TK_TRACE(972, zdepth);
                // F:973
                zfac = sqrtf(tk_add(1.f, tk_mul(zdepth, zbi))); TK_TRACE(973, zfac);
                // F:974
                zhmin(jl) = tk_add(zhmin(jl), tk_mul(tk_mul(zdhdz, zfac), zdz)); TK_TRACE(974, zhmin(jl));
                // F:975
                zrh = tk_mul(tk_mul((-tk_alv), tk_sub(zqsenh(jl, jk), zqenh(jl, jk))), zfac); TK_TRACE(975, zrh);
                // F:976
                if ((zhmin(jl) > zrh)) { ihmin(jl) = jk; }
            // F:977
            }
        // F:978
        }
    // F:979
    }
    // F:981
    const int tk_limit_981 = klon;
    for (jl = 1; jl <= tk_limit_981; jl += 1) {
        // F:982
        if ((ldcum(jl) & (ktype(jl) == 1))) {
            // F:983
            if ((ihmin(jl) < ictop0(jl))) { ihmin(jl) = ictop0(jl); }
        // F:984
        }
        // F:985
        if ((ktype(jl) == 1)) {
            // F:986
            zentr(jl) = tk_entrpen; TK_TRACE(986, zentr(jl));
        // F:987
        } else {
            // F:988
            zentr(jl) = tk_entrscv; TK_TRACE(988, zentr(jl));
        // F:989
        }
        // F:990
        if ((lndj(jl) == 1)) { zentr(jl) = tk_mul(zentr(jl), 1.1f); TK_TRACE(990, zentr(jl)); }
    // F:991
    }
    // F:994
    tk_cuasc_new(klon, klev, klevp1, klevm1, ztenh, zqenh, puen, pven, pten, pqen, pqsen, pgeo, zgeoh, pap, paph, pqte, pverv, ilwmin, ldcum, zhcbase, ktype, ilab, ptu, pqu, plu, zuu, zvu, pmfu, zmfub, zentr, zmfus, zmfuq, zmful, plude, zdmfup, kcbot, kctop, ictop0, icum, ztmst, ihmin, zhhatt, zqsenh, arena);
    // F:1004
    *arena.activity = icum; if (icum == 0 && !arena.force_continue) { return; }
    // F:1008
    const int tk_limit_1008 = klon;
    for (jl = 1; jl <= tk_limit_1008; jl += 1) {
        // F:1009
        zpbmpt = tk_sub(paph(jl, kcbot(jl)), paph(jl, kctop(jl))); TK_TRACE(1009, zpbmpt);
        // F:1010
        if (ldcum(jl)) { ictop0(jl) = kctop(jl); }
        // F:1011
        if (((ldcum(jl) & (ktype(jl) == 1)) & (zpbmpt < tk_zdnoprc))) { ktype(jl) = 2; }
        // F:1012
        if ((ktype(jl) == 2)) {
            // F:1013
            zentr(jl) = tk_entrscv; TK_TRACE(1013, zentr(jl));
            // F:1014
            if ((lndj(jl) == 1)) { zentr(jl) = tk_mul(zentr(jl), 1.1f); TK_TRACE(1014, zentr(jl)); }
        // F:1015
        }
        // F:1016
        zrfl(jl) = zdmfup(jl, 1); TK_TRACE(1016, zrfl(jl));
    // F:1017
    }
    // F:1019
    const int tk_limit_1019 = klev;
    for (jk = 2; jk <= tk_limit_1019; jk += 1) {
        // F:1020
        const int tk_limit_1020 = klon;
        for (jl = 1; jl <= tk_limit_1020; jl += 1) {
            // F:1021
            zrfl(jl) = tk_add(zrfl(jl), zdmfup(jl, jk)); TK_TRACE(1021, zrfl(jl));
        // F:1022
        }
    // F:1023
    }
    // F:1027
    if (tk_lmfdd) {
        // F:1030
        tk_cudlfs(klon, klev, klevp1, ztenh, zqenh, puen, pven, zgeoh, paph, ptu, pqu, zuu, zvu, ldcum, kcbot, kctop, zmfub, zrfl, ztd, zqd, zud, zvd, pmfd, zmfds, zmfdq, zdmfdp, idtop, loddraf, arena);
        // F:1039
        tk_cuddraf(klon, klev, klevp1, ztenh, zqenh, puen, pven, zgeoh, paph, zrfl, loddraf, ztd, zqd, zud, zvd, pmfd, zmfds, zmfdq, zdmfdp, arena);
    // F:1047
    }
    // F:1055
    const int tk_limit_1055 = klon;
    for (jl = 1; jl <= tk_limit_1055; jl += 1) {
        // F:1056
        zheat(jl) = 0.0f; TK_TRACE(1056, zheat(jl));
        // F:1057
        zcape(jl) = 0.0f; TK_TRACE(1057, zcape(jl));
        // F:1058
        zrelh(jl) = 0.0f; TK_TRACE(1058, zrelh(jl));
        // F:1059
        zmfub1(jl) = zmfub(jl); TK_TRACE(1059, zmfub1(jl));
    // F:1060
    }
    // F:1062
    const int tk_limit_1062 = klon;
    for (jl = 1; jl <= tk_limit_1062; jl += 1) {
        // F:1063
        if ((ldcum(jl) & (ktype(jl) == 1))) {
            // F:1064
            ktop0 = tk_max(12, kctop(jl));
            // F:1065
            ikb = kcbot(jl);
            // F:1066
            const int tk_limit_1066 = klev;
            for (jk = 2; jk <= tk_limit_1066; jk += 1) {
                // F:1067
                if (((jk <= kcbot(jl)) & (jk > kctop(jl)))) {
                    // F:1068
                    zro = tk_div(paph(jl, jk), tk_mul(tk_rd, ztenh(jl, jk))); TK_TRACE(1068, zro);
                    // F:1069
                    zdz = tk_div(tk_sub(paph(jl, jk), paph(jl, (jk - 1))), tk_mul(tk_g, zro)); TK_TRACE(1069, zdz);
                    // F:1070
                    zheat(jl) = tk_add(zheat(jl), tk_div(tk_mul(tk_mul(tk_add(tk_div(tk_add(tk_sub(pten(jl, (jk - 1)), pten(jl, jk)), tk_div(tk_mul(tk_g, zdz), tk_cpd)), ztenh(jl, jk)), tk_mul(0.608f, tk_sub(pqen(jl, (jk - 1)), pqen(jl, jk)))), tk_add(pmfu(jl, jk), pmfd(jl, jk))), tk_g), zro)); TK_TRACE(1070, zheat(jl));
                    // F:1073
                    zcape(jl) = tk_add(zcape(jl), tk_mul(tk_mul(tk_g, tk_sub(tk_div(tk_mul(ptu(jl, jk), tk_sub(tk_add(1.f, tk_mul(.608f, pqu(jl, jk))), plu(jl, jk))), tk_mul(ztenh(jl, jk), tk_add(1.f, tk_mul(.608f, zqenh(jl, jk))))), 1.0f)), zdz)); TK_TRACE(1073, zcape(jl));
                // F:1076
                }
                // F:1077
                if (((jk <= kcbot(jl)) & (jk > ktop0))) {
                    // F:1078
                    dept = tk_div(tk_sub(paph(jl, (jk + 1)), paph(jl, jk)), tk_sub(paph(jl, (ikb + 1)), paph(jl, (ktop0 + 1)))); TK_TRACE(1078, dept);
                    // F:1080
                    zrelh(jl) = tk_add(zrelh(jl), tk_div(tk_mul(dept, pqen(jl, jk)), pqsen(jl, jk))); TK_TRACE(1080, zrelh(jl));
                // F:1081
                }
            // F:1082
            }
            // F:1084
            if ((zrelh(jl) >= tk_crirh)) {
                // F:1085
                zht = tk_div(tk_max(0.0f, tk_sub(zcape(jl), 0.0f)), tk_mul(tk_ztau, zheat(jl))); TK_TRACE(1085, zht);
                // F:1086
                zmfub1(jl) = tk_max(tk_mul(zmfub(jl), zht), 0.01f); TK_TRACE(1086, zmfub1(jl));
                // F:1087
                zmfmax = tk_mul(tk_sub(paph(jl, ikb), paph(jl, (ikb - 1))), zcons2); TK_TRACE(1087, zmfmax);
                // F:1088
                zmfub1(jl) = tk_min(zmfub1(jl), zmfmax); TK_TRACE(1088, zmfub1(jl));
            // F:1089
            } else {
                // F:1090
                zmfub1(jl) = 0.01f; TK_TRACE(1090, zmfub1(jl));
                // F:1091
                zmfub(jl) = 0.01f; TK_TRACE(1091, zmfub(jl));
                // F:1092
                ldcum(jl) = false;
            // F:1093
            }
        // F:1094
        }
    // F:1095
    }
    // F:1100
    const int tk_limit_1100 = klon;
    for (jl = 1; jl <= tk_limit_1100; jl += 1) {
        // F:1101
        if ((ktype(jl) != 1)) {
            // F:1102
            ikb = kcbot(jl);
            // F:1103
            if (((pmfd(jl, ikb) < 0.0f) & loddraf(jl))) {
                // F:1104
                zeps = tk_cmfdeps; TK_TRACE(1104, zeps);
            // F:1105
            } else {
                // F:1106
                zeps = 0.f; TK_TRACE(1106, zeps);
            // F:1107
            }
            // F:1108
            zqumqe = tk_sub(tk_sub(tk_add(pqu(jl, ikb), plu(jl, ikb)), tk_mul(zeps, zqd(jl, ikb))), tk_mul(tk_sub(1.f, zeps), zqenh(jl, ikb))); TK_TRACE(1108, zqumqe);
            // F:1110
            zdqmin = tk_max(tk_mul(0.01f, zqenh(jl, ikb)), 1.e-10f); TK_TRACE(1110, zdqmin);
            // F:1111
            zmfmax = tk_mul(tk_sub(paph(jl, ikb), paph(jl, (ikb - 1))), zcons2); TK_TRACE(1111, zmfmax);
            // F:1112
            if (((((zdqpbl(jl) > 0.f) & (zqumqe > zdqmin)) & ldcum(jl)) & (zmfub(jl) < zmfmax))) {
                // F:1114
                zmfub1(jl) = tk_div(zdqpbl(jl), tk_mul(tk_g, tk_max(zqumqe, zdqmin))); TK_TRACE(1114, zmfub1(jl));
            // F:1115
            } else {
                // F:1116
                zmfub1(jl) = zmfub(jl); TK_TRACE(1116, zmfub1(jl));
            // F:1117
            }
            // F:1118
            llo1 = ((ktype(jl) == 2) & (tk_abs(tk_sub(zmfub1(jl), zmfub(jl))) < tk_mul(0.2f, zmfub(jl))));
            // F:1120
            if ((!llo1)) { zmfub1(jl) = zmfub(jl); TK_TRACE(1120, zmfub1(jl)); }
            // F:1121
            zmfub1(jl) = tk_min(zmfub1(jl), zmfmax); TK_TRACE(1121, zmfub1(jl));
        // F:1122
        }
    // F:1123
    }
    // F:1125
    const int tk_limit_1125 = klev;
    for (jk = 1; jk <= tk_limit_1125; jk += 1) {
        // F:1126
        const int tk_limit_1126 = klon;
        for (jl = 1; jl <= tk_limit_1126; jl += 1) {
            // F:1127
            if (ldcum(jl)) {
                // F:1128
                zfac = tk_div(zmfub1(jl), tk_max(zmfub(jl), 1.e-10f)); TK_TRACE(1128, zfac);
                // F:1129
                pmfd(jl, jk) = tk_mul(pmfd(jl, jk), zfac); TK_TRACE(1129, pmfd(jl, jk));
                // F:1130
                zmfds(jl, jk) = tk_mul(zmfds(jl, jk), zfac); TK_TRACE(1130, zmfds(jl, jk));
                // F:1131
                zmfdq(jl, jk) = tk_mul(zmfdq(jl, jk), zfac); TK_TRACE(1131, zmfdq(jl, jk));
                // F:1132
                zdmfdp(jl, jk) = tk_mul(zdmfdp(jl, jk), zfac); TK_TRACE(1132, zdmfdp(jl, jk));
            // F:1133
            } else {
                // F:1134
                pmfd(jl, jk) = 0.0f; TK_TRACE(1134, pmfd(jl, jk));
                // F:1135
                zmfds(jl, jk) = 0.0f; TK_TRACE(1135, zmfds(jl, jk));
                // F:1136
                zmfdq(jl, jk) = 0.0f; TK_TRACE(1136, zmfdq(jl, jk));
                // F:1137
                zdmfdp(jl, jk) = 0.0f; TK_TRACE(1137, zdmfdp(jl, jk));
            // F:1138
            }
        // F:1139
        }
    // F:1140
    }
    // F:1142
    const int tk_limit_1142 = klon;
    for (jl = 1; jl <= tk_limit_1142; jl += 1) {
        // F:1143
        if (ldcum(jl)) {
            // F:1144
            zmfub(jl) = zmfub1(jl); TK_TRACE(1144, zmfub(jl));
        // F:1145
        } else {
            // F:1146
            zmfub(jl) = 0.0f; TK_TRACE(1146, zmfub(jl));
        // F:1147
        }
    // F:1148
    }
    // F:1156
    tk_cuasc_new(klon, klev, klevp1, klevm1, ztenh, zqenh, puen, pven, pten, pqen, pqsen, pgeo, zgeoh, pap, paph, pqte, pverv, ilwmin, ldcum, zhcbase, ktype, ilab, ptu, pqu, plu, zuu, zvu, pmfu, zmfub, zentr, zmfus, zmfuq, zmful, plude, zdmfup, kcbot, kctop, ictop0, icum, ztmst, ihmin, zhhatt, zqsenh, arena);
    // F:1169
    tk_cuflx(klon, klev, klevp1, pqen, pqsen, ztenh, zqenh, paph, zgeoh, kcbot, kctop, idtop, ktype, loddraf, ldcum, pmfu, pmfd, zmfus, zmfds, zmfuq, zmfdq, zmful, plude, zdmfup, zdmfdp, zrfl, prain, pten, zsfl, zdpmel, itopm2, ztmst, sig1, arena);
    // F:1180
    tk_cudtdq(klon, klev, klevp1, itopm2, paph, ldcum, pten, ptte, pqte, zmfus, zmfds, zmfuq, zmfdq, zmful, zdmfup, zdmfdp, ztmst, zdpmel, prain, zrfl, zsfl, psrain, psevap, psheat, psmelt, prsfc, pssfc, paprc, paprsm, paprs, pqen, pqsen, plude, pcte, arena);
    // F:1191
    if (tk_lmfdudv) {
        // F:1192
        tk_cududv(klon, klev, klevp1, itopm2, ktype, kcbot, paph, ldcum, puen, pven, pvom, pvol, zuu, zud, zvu, zvd, pmfu, pmfd, psdiss, arena);
    // F:1197
    }
    // F:1198
    return;
}

__device__ void tk_cuini(int klon, int klev, int klevp1, int klevm1, TkArray<float> pten, TkArray<float> pqen, TkArray<float> pqsen, TkArray<float> puen, TkArray<float> pven, TkArray<float> pverv, TkArray<float> pgeo, TkArray<float> paph, TkArray<float> pgeoh, TkArray<float> ptenh, TkArray<float> pqenh, TkArray<float> pqsenh, TkArray<int> klwmin, TkArray<float> ptu, TkArray<float> pqu, TkArray<float> ptd, TkArray<float> pqd, TkArray<float> puu, TkArray<float> pvu, TkArray<float> pud, TkArray<float> pvd, TkArray<float> pmfu, TkArray<float> pmfd, TkArray<float> pmfus, TkArray<float> pmfds, TkArray<float> pmfuq, TkArray<float> pmfdq, TkArray<float> pdmfup, TkArray<float> pdmfdp, TkArray<float> pdpmel, TkArray<float> plu, TkArray<float> plude, TkArray<int> klab, TkArena arena) {
    // module_cu_tiedtke.F:1211
    int jk;
    int jl;
    int ik;
    int icall;
    float zdp;
    float zzs;
    TkArray<float> zwmax = arena.take<float>((klon), klon);
    TkArray<float> zph = arena.take<float>((klon), klon);
    TkArray<bool> loflag = arena.take<bool>((klon), klon);
    // F:1267
    zdp = 0.5f; TK_TRACE(1267, zdp);
    // F:1268
    const int tk_limit_1268 = klev;
    for (jk = 2; jk <= tk_limit_1268; jk += 1) {
        // F:1269
        const int tk_limit_1269 = klon;
        for (jl = 1; jl <= tk_limit_1269; jl += 1) {
            // F:1270
            pgeoh(jl, jk) = tk_add(pgeo(jl, jk), tk_mul(tk_sub(pgeo(jl, (jk - 1)), pgeo(jl, jk)), zdp)); TK_TRACE(1270, pgeoh(jl, jk));
            // F:1271
            ptenh(jl, jk) = tk_mul(tk_sub(tk_max(tk_add(tk_mul(tk_cpd, pten(jl, (jk - 1))), pgeo(jl, (jk - 1))), tk_add(tk_mul(tk_cpd, pten(jl, jk)), pgeo(jl, jk))), pgeoh(jl, jk)), tk_rcpd); TK_TRACE(1271, ptenh(jl, jk));
            // F:1273
            pqsenh(jl, jk) = pqsen(jl, (jk - 1)); TK_TRACE(1273, pqsenh(jl, jk));
            // F:1274
            zph(jl) = paph(jl, jk); TK_TRACE(1274, zph(jl));
            // F:1275
            loflag(jl) = true;
        // F:1276
        }
        // F:1278
        ik = jk;
        // F:1279
        icall = 0;
        // F:1280
        tk_cuadjtq(klon, klev, ik, zph, ptenh, pqsenh, loflag, icall, arena);
        // F:1281
        const int tk_limit_1281 = klon;
        for (jl = 1; jl <= tk_limit_1281; jl += 1) {
            // F:1282
            pqenh(jl, jk) = tk_add(tk_min(pqen(jl, (jk - 1)), pqsen(jl, (jk - 1))), tk_sub(pqsenh(jl, jk), pqsen(jl, (jk - 1)))); TK_TRACE(1282, pqenh(jl, jk));
            // F:1284
            pqenh(jl, jk) = tk_max(pqenh(jl, jk), 0.f); TK_TRACE(1284, pqenh(jl, jk));
        // F:1285
        }
    // F:1286
    }
    // F:1288
    const int tk_limit_1288 = klon;
    for (jl = 1; jl <= tk_limit_1288; jl += 1) {
        // F:1289
        ptenh(jl, klev) = tk_mul(tk_sub(tk_add(tk_mul(tk_cpd, pten(jl, klev)), pgeo(jl, klev)), pgeoh(jl, klev)), tk_rcpd); TK_TRACE(1289, ptenh(jl, klev));
        // F:1291
        pqenh(jl, klev) = pqen(jl, klev); TK_TRACE(1291, pqenh(jl, klev));
        // F:1292
        ptenh(jl, 1) = pten(jl, 1); TK_TRACE(1292, ptenh(jl, 1));
        // F:1293
        pqenh(jl, 1) = pqen(jl, 1); TK_TRACE(1293, pqenh(jl, 1));
        // F:1294
        pgeoh(jl, 1) = pgeo(jl, 1); TK_TRACE(1294, pgeoh(jl, 1));
        // F:1295
        klwmin(jl) = klev;
        // F:1296
        zwmax(jl) = 0.f; TK_TRACE(1296, zwmax(jl));
    // F:1297
    }
    // F:1299
    const int tk_limit_1299 = 2;
    for (jk = klevm1; jk >= tk_limit_1299; jk += (-1)) {
        // F:1300
        const int tk_limit_1300 = klon;
        for (jl = 1; jl <= tk_limit_1300; jl += 1) {
            // F:1301
            zzs = tk_max(tk_add(tk_mul(tk_cpd, ptenh(jl, jk)), pgeoh(jl, jk)), tk_add(tk_mul(tk_cpd, ptenh(jl, (jk + 1))), pgeoh(jl, (jk + 1)))); TK_TRACE(1301, zzs);
            // F:1303
            ptenh(jl, jk) = tk_mul(tk_sub(zzs, pgeoh(jl, jk)), tk_rcpd); TK_TRACE(1303, ptenh(jl, jk));
        // F:1304
        }
    // F:1305
    }
    // F:1307
    const int tk_limit_1307 = 3;
    for (jk = klev; jk >= tk_limit_1307; jk += (-1)) {
        // F:1308
        const int tk_limit_1308 = klon;
        for (jl = 1; jl <= tk_limit_1308; jl += 1) {
            // F:1309
            if ((pverv(jl, jk) < zwmax(jl))) {
                // F:1310
                zwmax(jl) = pverv(jl, jk); TK_TRACE(1310, zwmax(jl));
                // F:1311
                klwmin(jl) = jk;
            // F:1312
            }
        // F:1313
        }
    // F:1314
    }
    // F:1318
    const int tk_limit_1318 = klev;
    for (jk = 1; jk <= tk_limit_1318; jk += 1) {
        // F:1319
        ik = (jk - 1);
        // F:1320
        if ((jk == 1)) { ik = 1; }
        // F:1321
        const int tk_limit_1321 = klon;
        for (jl = 1; jl <= tk_limit_1321; jl += 1) {
            // F:1322
            ptu(jl, jk) = ptenh(jl, jk); TK_TRACE(1322, ptu(jl, jk));
            // F:1323
            ptd(jl, jk) = ptenh(jl, jk); TK_TRACE(1323, ptd(jl, jk));
            // F:1324
            pqu(jl, jk) = pqenh(jl, jk); TK_TRACE(1324, pqu(jl, jk));
            // F:1325
            pqd(jl, jk) = pqenh(jl, jk); TK_TRACE(1325, pqd(jl, jk));
            // F:1326
            plu(jl, jk) = 0.f; TK_TRACE(1326, plu(jl, jk));
            // F:1327
            puu(jl, jk) = puen(jl, ik); TK_TRACE(1327, puu(jl, jk));
            // F:1328
            pud(jl, jk) = puen(jl, ik); TK_TRACE(1328, pud(jl, jk));
            // F:1329
            pvu(jl, jk) = pven(jl, ik); TK_TRACE(1329, pvu(jl, jk));
            // F:1330
            pvd(jl, jk) = pven(jl, ik); TK_TRACE(1330, pvd(jl, jk));
            // F:1331
            pmfu(jl, jk) = 0.f; TK_TRACE(1331, pmfu(jl, jk));
            // F:1332
            pmfd(jl, jk) = 0.f; TK_TRACE(1332, pmfd(jl, jk));
            // F:1333
            pmfus(jl, jk) = 0.f; TK_TRACE(1333, pmfus(jl, jk));
            // F:1334
            pmfds(jl, jk) = 0.f; TK_TRACE(1334, pmfds(jl, jk));
            // F:1335
            pmfuq(jl, jk) = 0.f; TK_TRACE(1335, pmfuq(jl, jk));
            // F:1336
            pmfdq(jl, jk) = 0.f; TK_TRACE(1336, pmfdq(jl, jk));
            // F:1337
            pdmfup(jl, jk) = 0.f; TK_TRACE(1337, pdmfup(jl, jk));
            // F:1338
            pdmfdp(jl, jk) = 0.f; TK_TRACE(1338, pdmfdp(jl, jk));
            // F:1339
            pdpmel(jl, jk) = 0.f; TK_TRACE(1339, pdpmel(jl, jk));
            // F:1340
            plude(jl, jk) = 0.f; TK_TRACE(1340, plude(jl, jk));
            // F:1341
            klab(jl, jk) = 0;
        // F:1342
        }
    // F:1343
    }
    // F:1344
    return;
}

__device__ void tk_cubase(int klon, int klev, int klevp1, int klevm1, TkArray<float> ptenh, TkArray<float> pqenh, TkArray<float> pgeoh, TkArray<float> paph, TkArray<float> ptu, TkArray<float> pqu, TkArray<float> plu, TkArray<float> puen, TkArray<float> pven, TkArray<float> puu, TkArray<float> pvu, TkArray<bool> ldcum, TkArray<int> kcbot, TkArray<int> klab, TkArena arena) {
    // module_cu_tiedtke.F:1350
    int jl;
    int jk;
    int is;
    int ik;
    int icall;
    int ikb;
    float zbuo;
    float zz;
    TkArray<float> zqold = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zph = arena.take<float>((klon), klon);
    TkArray<bool> loflag = arena.take<bool>((klon), klon);
    TkArray<bool> ldbase = arena.take<bool>((klon), klon);
    bool llo1;
    // F:1412
    const int tk_limit_1412 = klon;
    for (jl = 1; jl <= tk_limit_1412; jl += 1) {
        // F:1413
        klab(jl, klev) = 1;
        // F:1414
        kcbot(jl) = klevm1;
        // F:1415
        ldcum(jl) = false;
        // F:1416
        ldbase(jl) = false;
        // F:1417
        puu(jl, klev) = tk_mul(puen(jl, klev), tk_sub(paph(jl, klevp1), paph(jl, klev))); TK_TRACE(1417, puu(jl, klev));
        // F:1418
        pvu(jl, klev) = tk_mul(pven(jl, klev), tk_sub(paph(jl, klevp1), paph(jl, klev))); TK_TRACE(1418, pvu(jl, klev));
    // F:1419
    }
    // F:1426
    const int tk_limit_1426 = klev;
    for (jk = 1; jk <= tk_limit_1426; jk += 1) {
        // F:1427
        const int tk_limit_1427 = klon;
        for (jl = 1; jl <= tk_limit_1427; jl += 1) {
            // F:1428
            zqold(jl, jk) = 0.0f; TK_TRACE(1428, zqold(jl, jk));
        // F:1429
        }
    // F:1430
    }
    // F:1432
    const int tk_limit_1432 = 2;
    for (jk = klevm1; jk >= tk_limit_1432; jk += (-1)) {
        // F:1433
        is = 0;
        // F:1434
        const int tk_limit_1434 = klon;
        for (jl = 1; jl <= tk_limit_1434; jl += 1) {
            // F:1435
            if (((klab(jl, (jk + 1)) == 1) | (ldcum(jl) & (kcbot(jl) == (jk + 1))))) {
                // F:1436
                is = (is + 1);
                // F:1437
                loflag(jl) = true;
            // F:1438
            } else {
                // F:1439
                loflag(jl) = false;
            // F:1440
            }
            // F:1441
            zph(jl) = paph(jl, jk); TK_TRACE(1441, zph(jl));
        // F:1442
        }
        // F:1443
        if ((is == 0)) { continue; }
        // F:1448
        if (tk_lmfdudv) {
            // F:1449
            const int tk_limit_1449 = klon;
            for (jl = 1; jl <= tk_limit_1449; jl += 1) {
                // F:1450
                if ((!ldbase(jl))) {
                    // F:1451
                    puu(jl, klev) = tk_add(puu(jl, klev), tk_mul(puen(jl, jk), tk_sub(paph(jl, (jk + 1)), paph(jl, jk)))); TK_TRACE(1451, puu(jl, klev));
                    // F:1453
                    pvu(jl, klev) = tk_add(pvu(jl, klev), tk_mul(pven(jl, jk), tk_sub(paph(jl, (jk + 1)), paph(jl, jk)))); TK_TRACE(1453, pvu(jl, klev));
                // F:1455
                }
            // F:1456
            }
        // F:1457
        }
        // F:1459
        const int tk_limit_1459 = klon;
        for (jl = 1; jl <= tk_limit_1459; jl += 1) {
            // F:1460
            if (loflag(jl)) {
                // F:1461
                pqu(jl, jk) = pqu(jl, (jk + 1)); TK_TRACE(1461, pqu(jl, jk));
                // F:1462
                ptu(jl, jk) = tk_mul(tk_sub(tk_add(tk_mul(tk_cpd, ptu(jl, (jk + 1))), pgeoh(jl, (jk + 1))), pgeoh(jl, jk)), tk_rcpd); TK_TRACE(1462, ptu(jl, jk));
                // F:1464
                zqold(jl, jk) = pqu(jl, jk); TK_TRACE(1464, zqold(jl, jk));
            // F:1465
            }
        // F:1466
        }
        // F:1468
        ik = jk;
        // F:1469
        icall = 1;
        // F:1470
        tk_cuadjtq(klon, klev, ik, zph, ptu, pqu, loflag, icall, arena);
        // F:1472
        const int tk_limit_1472 = klon;
        for (jl = 1; jl <= tk_limit_1472; jl += 1) {
            // F:1473
            if (loflag(jl)) {
                // F:1474
                if ((pqu(jl, jk) == zqold(jl, jk))) {
                    // F:1475
                    zbuo = tk_add(tk_sub(tk_mul(ptu(jl, jk), tk_add(1.f, tk_mul(tk_vtmpc1, pqu(jl, jk)))), tk_mul(ptenh(jl, jk), tk_add(1.f, tk_mul(tk_vtmpc1, pqenh(jl, jk))))), tk_zbuo0); TK_TRACE(1475, zbuo);
                    // F:1477
                    if ((zbuo > 0.f)) { klab(jl, jk) = 1; }
                // F:1478
                } else {
                    // F:1479
                    klab(jl, jk) = 2;
                    // F:1480
                    plu(jl, jk) = tk_sub(tk_add(plu(jl, jk), zqold(jl, jk)), pqu(jl, jk)); TK_TRACE(1480, plu(jl, jk));
                    // F:1481
                    zbuo = tk_add(tk_sub(tk_mul(ptu(jl, jk), tk_add(1.f, tk_mul(tk_vtmpc1, pqu(jl, jk)))), tk_mul(ptenh(jl, jk), tk_add(1.f, tk_mul(tk_vtmpc1, pqenh(jl, jk))))), tk_zbuo0); TK_TRACE(1481, zbuo);
                    // F:1483
                    llo1 = ((zbuo > 0.f) & (klab(jl, (jk + 1)) == 1));
                    // F:1484
                    if (llo1) {
                        // F:1485
                        kcbot(jl) = jk;
                        // F:1486
                        ldcum(jl) = true;
                        // F:1487
                        ldbase(jl) = true;
                    // F:1488
                    }
                // F:1489
                }
            // F:1490
            }
        // F:1491
        }
    // F:1492
    }
    // F:1494
    if (tk_lmfdudv) {
        // F:1495
        const int tk_limit_1495 = klon;
        for (jl = 1; jl <= tk_limit_1495; jl += 1) {
            // F:1496
            if (ldcum(jl)) {
                // F:1497
                ikb = kcbot(jl);
                // F:1498
                zz = tk_div(1.f, tk_sub(paph(jl, klevp1), paph(jl, ikb))); TK_TRACE(1498, zz);
                // F:1499
                puu(jl, klev) = tk_mul(puu(jl, klev), zz); TK_TRACE(1499, puu(jl, klev));
                // F:1500
                pvu(jl, klev) = tk_mul(pvu(jl, klev), zz); TK_TRACE(1500, pvu(jl, klev));
            // F:1501
            } else {
                // F:1502
                puu(jl, klev) = puen(jl, klevm1); TK_TRACE(1502, puu(jl, klev));
                // F:1503
                pvu(jl, klev) = pven(jl, klevm1); TK_TRACE(1503, pvu(jl, klev));
            // F:1504
            }
        // F:1505
        }
    // F:1506
    }
    // F:1507
    return;
}

__device__ void tk_cuasc_new(int klon, int klev, int klevp1, int klevm1, TkArray<float> ptenh, TkArray<float> pqenh, TkArray<float> puen, TkArray<float> pven, TkArray<float> pten, TkArray<float> pqen, TkArray<float> pqsen, TkArray<float> pgeo, TkArray<float> pgeoh, TkArray<float> pap, TkArray<float> paph, TkArray<float> pqte, TkArray<float> pverv, TkArray<int> klwmin, TkArray<bool> ldcum, TkArray<float> phcbase, TkArray<int> ktype, TkArray<int> klab, TkArray<float> ptu, TkArray<float> pqu, TkArray<float> plu, TkArray<float> puu, TkArray<float> pvu, TkArray<float> pmfu, TkArray<float> pmfub, TkArray<float> pentr, TkArray<float> pmfus, TkArray<float> pmfuq, TkArray<float> pmful, TkArray<float> plude, TkArray<float> pdmfup, TkArray<int> kcbot, TkArray<int> kctop, TkArray<int> kctop0, int& kcum, float ztmst, TkArray<int> khmin, TkArray<float> phhatt, TkArray<float> pqsenh, TkArena arena) {
    // module_cu_tiedtke.F:1513
    float zcons2;
    float zdz;
    float zdrodz;
    int jl;
    int jk;
    int ikb;
    int ik;
    int is;
    int ikt;
    int icall;
    float zmfmax;
    float zfac;
    float zmftest;
    float zdprho;
    float zmse;
    float znevn;
    float zodmax;
    float zqeen;
    float zseen;
    float zscde;
    float zga;
    float zdt;
    float zscod;
    float zqude;
    float zqcod;
    float zmfusk;
    float zmfuqk;
    float zmfulk;
    float zbuo;
    float zprcon;
    float zlnew;
    float zz;
    float zdmfeu;
    float zdmfdu;
    float zbuoyz;
    float zzdmf;
    TkArray<float> zph = arena.take<float>((klon), klon);
    TkArray<float> zdmfen = arena.take<float>((klon), klon);
    TkArray<float> zdmfde = arena.take<float>((klon), klon);
    TkArray<float> zmfuu = arena.take<float>((klon), klon);
    TkArray<float> zmfuv = arena.take<float>((klon), klon);
    TkArray<float> zpbase = arena.take<float>((klon), klon);
    TkArray<float> zqold = arena.take<float>((klon), klon);
    TkArray<float> zodetr = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zoentr = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zbuoy = arena.take<float>((klon), klon);
    TkArray<bool> loflag = arena.take<bool>((klon), klon);
    // F:1632
    zcons2 = tk_div(1.f, tk_mul(tk_g, ztmst)); TK_TRACE(1632, zcons2);
    // F:1636
    const int tk_limit_1636 = klon;
    for (jl = 1; jl <= tk_limit_1636; jl += 1) {
        // F:1637
        zmfuu(jl) = 0.f; TK_TRACE(1637, zmfuu(jl));
        // F:1638
        zmfuv(jl) = 0.f; TK_TRACE(1638, zmfuv(jl));
        // F:1639
        zbuoy(jl) = 0.f; TK_TRACE(1639, zbuoy(jl));
        // F:1640
        if ((!ldcum(jl))) { ktype(jl) = 0; }
    // F:1641
    }
    // F:1643
    const int tk_limit_1643 = klev;
    for (jk = 1; jk <= tk_limit_1643; jk += 1) {
        // F:1644
        const int tk_limit_1644 = klon;
        for (jl = 1; jl <= tk_limit_1644; jl += 1) {
            // F:1645
            plu(jl, jk) = 0.f; TK_TRACE(1645, plu(jl, jk));
            // F:1646
            pmfu(jl, jk) = 0.f; TK_TRACE(1646, pmfu(jl, jk));
            // F:1647
            pmfus(jl, jk) = 0.f; TK_TRACE(1647, pmfus(jl, jk));
            // F:1648
            pmfuq(jl, jk) = 0.f; TK_TRACE(1648, pmfuq(jl, jk));
            // F:1649
            pmful(jl, jk) = 0.f; TK_TRACE(1649, pmful(jl, jk));
            // F:1650
            plude(jl, jk) = 0.f; TK_TRACE(1650, plude(jl, jk));
            // F:1651
            pdmfup(jl, jk) = 0.f; TK_TRACE(1651, pdmfup(jl, jk));
            // F:1652
            zoentr(jl, jk) = 0.f; TK_TRACE(1652, zoentr(jl, jk));
            // F:1653
            zodetr(jl, jk) = 0.f; TK_TRACE(1653, zodetr(jl, jk));
            // F:1654
            if (((!ldcum(jl)) | (ktype(jl) == 3))) { klab(jl, jk) = 0; }
            // F:1655
            if (((!ldcum(jl)) & (paph(jl, jk) < 4.e4f))) { kctop0(jl) = jk; }
        // F:1656
        }
    // F:1657
    }
    // F:1661
    const int tk_limit_1661 = klon;
    for (jl = 1; jl <= tk_limit_1661; jl += 1) {
        // F:1662
        kctop(jl) = klevm1;
        // F:1663
        if ((!ldcum(jl))) {
            // F:1664
            kcbot(jl) = klevm1;
            // F:1665
            pmfub(jl) = 0.f; TK_TRACE(1665, pmfub(jl));
            // F:1666
            pqu(jl, klev) = 0.f; TK_TRACE(1666, pqu(jl, klev));
        // F:1667
        }
        // F:1668
        pmfu(jl, klev) = pmfub(jl); TK_TRACE(1668, pmfu(jl, klev));
        // F:1669
        pmfus(jl, klev) = tk_mul(pmfub(jl), tk_add(tk_mul(tk_cpd, ptu(jl, klev)), pgeoh(jl, klev))); TK_TRACE(1669, pmfus(jl, klev));
        // F:1670
        pmfuq(jl, klev) = tk_mul(pmfub(jl), pqu(jl, klev)); TK_TRACE(1670, pmfuq(jl, klev));
        // F:1671
        if (tk_lmfdudv) {
            // F:1672
            zmfuu(jl) = tk_mul(pmfub(jl), puu(jl, klev)); TK_TRACE(1672, zmfuu(jl));
            // F:1673
            zmfuv(jl) = tk_mul(pmfub(jl), pvu(jl, klev)); TK_TRACE(1673, zmfuv(jl));
        // F:1674
        }
    // F:1675
    }
    // F:1679
    const int tk_limit_1679 = klon;
    for (jl = 1; jl <= tk_limit_1679; jl += 1) {
        // F:1680
        ldcum(jl) = false;
        // F:1681
        if ((ktype(jl) == 1)) {
            // F:1682
            ikb = kcbot(jl);
            // F:1683
            zbuoy(jl) = tk_mul(tk_g, tk_add(tk_div(tk_sub(ptu(jl, ikb), ptenh(jl, ikb)), ptenh(jl, ikb)), tk_mul(0.608f, tk_sub(pqu(jl, ikb), pqenh(jl, ikb))))); TK_TRACE(1683, zbuoy(jl));
            // F:1685
            if ((zbuoy(jl) > 0.f)) {
                // F:1686
                zdz = tk_mul(tk_sub(pgeo(jl, (ikb - 1)), pgeo(jl, ikb)), tk_zrg); TK_TRACE(1686, zdz);
                // F:1687
                zdrodz = tk_sub(tk_div((-gfk_log(tk_div(pten(jl, (ikb - 1)), pten(jl, ikb)))), zdz), tk_div(tk_g, tk_mul(tk_rd, ptenh(jl, ikb)))); TK_TRACE(1687, zdrodz);
                // F:1689
                zoentr(jl, (ikb - 1)) = tk_add(tk_div(tk_mul(zbuoy(jl), 0.5f), tk_add(1.f, tk_mul(zbuoy(jl), zdz))), zdrodz); TK_TRACE(1689, zoentr(jl, (ikb - 1)));
                // F:1691
                zoentr(jl, (ikb - 1)) = tk_min(zoentr(jl, (ikb - 1)), 1.e-3f); TK_TRACE(1691, zoentr(jl, (ikb - 1)));
                // F:1692
                zoentr(jl, (ikb - 1)) = tk_max(zoentr(jl, (ikb - 1)), 0.f); TK_TRACE(1692, zoentr(jl, (ikb - 1)));
            // F:1693
            }
        // F:1694
        }
    // F:1695
    }
    // F:1703
    const int tk_limit_1703 = 2;
    for (jk = klevm1; jk >= tk_limit_1703; jk += (-1)) {
        // F:1707
        ik = jk;
        // F:1708
        if (((tk_lmfmid & (ik < klevm1)) & (ik > (klev - 13)))) {
            // F:1709
            tk_cubasmc(klon, klev, klevm1, ik, pten, pqen, pqsen, puen, pven, pverv, pgeo, pgeoh, ldcum, ktype, klab, pmfu, pmfub, pentr, kcbot, ptu, pqu, plu, puu, pvu, pmfus, pmfuq, pmful, pdmfup, zmfuu, zmfuv, arena);
        // F:1716
        }
        // F:1717
        is = 0;
        // F:1718
        const int tk_limit_1718 = klon;
        for (jl = 1; jl <= tk_limit_1718; jl += 1) {
            // F:1719
            zqold(jl) = 0.0f; TK_TRACE(1719, zqold(jl));
            // F:1720
            is = (is + klab(jl, (jk + 1)));
            // F:1721
            if ((klab(jl, (jk + 1)) == 0)) { klab(jl, jk) = 0; }
            // F:1722
            loflag(jl) = (klab(jl, (jk + 1)) > 0);
            // F:1723
            zph(jl) = paph(jl, jk); TK_TRACE(1723, zph(jl));
            // F:1724
            if (((ktype(jl) == 3) & (jk == kcbot(jl)))) {
                // F:1725
                zmfmax = tk_mul(tk_sub(paph(jl, jk), paph(jl, (jk - 1))), zcons2); TK_TRACE(1725, zmfmax);
                // F:1726
                if ((pmfub(jl) > zmfmax)) {
                    // F:1727
                    zfac = tk_div(zmfmax, pmfub(jl)); TK_TRACE(1727, zfac);
                    // F:1728
                    pmfu(jl, (jk + 1)) = tk_mul(pmfu(jl, (jk + 1)), zfac); TK_TRACE(1728, pmfu(jl, (jk + 1)));
                    // F:1729
                    pmfus(jl, (jk + 1)) = tk_mul(pmfus(jl, (jk + 1)), zfac); TK_TRACE(1729, pmfus(jl, (jk + 1)));
                    // F:1730
                    pmfuq(jl, (jk + 1)) = tk_mul(pmfuq(jl, (jk + 1)), zfac); TK_TRACE(1730, pmfuq(jl, (jk + 1)));
                    // F:1731
                    zmfuu(jl) = tk_mul(zmfuu(jl), zfac); TK_TRACE(1731, zmfuu(jl));
                    // F:1732
                    zmfuv(jl) = tk_mul(zmfuv(jl), zfac); TK_TRACE(1732, zmfuv(jl));
                    // F:1733
                    pmfub(jl) = zmfmax; TK_TRACE(1733, pmfub(jl));
                // F:1734
                }
            // F:1735
            }
        // F:1736
        }
        // F:1738
        if ((is == 0)) { continue; }
        // F:1742
        ik = jk;
        // F:1743
        tk_cuentr_new(klon, klev, klevp1, ik, ptenh, paph, pap, pgeoh, klwmin, ldcum, ktype, kcbot, kctop0, zpbase, pmfu, pentr, zdmfen, zdmfde, zodetr, khmin, arena);
        // F:1758
        const int tk_limit_1758 = klon;
        for (jl = 1; jl <= tk_limit_1758; jl += 1) {
            // F:1759
            if (loflag(jl)) {
                // F:1760
                if ((jk < kcbot(jl))) {
                    // F:1761
                    zmftest = tk_sub(tk_add(pmfu(jl, (jk + 1)), zdmfen(jl)), zdmfde(jl)); TK_TRACE(1761, zmftest);
                    // F:1762
                    zmfmax = tk_min(zmftest, tk_mul(tk_sub(paph(jl, jk), paph(jl, (jk - 1))), zcons2)); TK_TRACE(1762, zmfmax);
                    // F:1763
                    zdmfen(jl) = tk_max(tk_sub(zdmfen(jl), tk_max(tk_sub(zmftest, zmfmax), 0.f)), 0.f); TK_TRACE(1763, zdmfen(jl));
                // F:1764
                }
                // F:1765
                zdmfde(jl) = tk_min(zdmfde(jl), tk_mul(0.75f, pmfu(jl, (jk + 1)))); TK_TRACE(1765, zdmfde(jl));
                // F:1766
                pmfu(jl, jk) = tk_sub(tk_add(pmfu(jl, (jk + 1)), zdmfen(jl)), zdmfde(jl)); TK_TRACE(1766, pmfu(jl, jk));
                // F:1767
                if ((jk < kcbot(jl))) {
                    // F:1768
                    zdprho = tk_mul(tk_sub(pgeoh(jl, jk), pgeoh(jl, (jk + 1))), tk_zrg); TK_TRACE(1768, zdprho);
                    // F:1769
                    zoentr(jl, jk) = tk_mul(tk_mul(zoentr(jl, jk), zdprho), pmfu(jl, (jk + 1))); TK_TRACE(1769, zoentr(jl, jk));
                    // F:1770
                    zmftest = tk_sub(tk_add(pmfu(jl, jk), zoentr(jl, jk)), zodetr(jl, jk)); TK_TRACE(1770, zmftest);
                    // F:1771
                    zmfmax = tk_min(zmftest, tk_mul(tk_sub(paph(jl, jk), paph(jl, (jk - 1))), zcons2)); TK_TRACE(1771, zmfmax);
                    // F:1772
                    zoentr(jl, jk) = tk_max(tk_sub(zoentr(jl, jk), tk_max(tk_sub(zmftest, zmfmax), 0.f)), 0.f); TK_TRACE(1772, zoentr(jl, jk));
                // F:1773
                }
                // F:1777
                if ((((ktype(jl) == 1) & (jk < kcbot(jl))) & (jk <= khmin(jl)))) {
                    // F:1778
                    zmse = tk_add(tk_add(tk_mul(tk_cpd, ptu(jl, (jk + 1))), tk_mul(tk_alv, pqu(jl, (jk + 1)))), pgeoh(jl, (jk + 1))); TK_TRACE(1778, zmse);
                    // F:1779
                    ikt = kctop0(jl);
                    // F:1780
                    znevn = tk_mul(tk_mul(tk_sub(pgeoh(jl, ikt), pgeoh(jl, (jk + 1))), tk_sub(zmse, phhatt(jl, (jk + 1)))), tk_zrg); TK_TRACE(1780, znevn);
                    // F:1782
                    if ((znevn <= 0.f)) { znevn = 1.f; TK_TRACE(1782, znevn); }
                    // F:1783
                    zdprho = tk_mul(tk_sub(pgeoh(jl, jk), pgeoh(jl, (jk + 1))), tk_zrg); TK_TRACE(1783, zdprho);
                    // F:1784
                    zodmax = tk_mul(tk_mul(tk_div(tk_sub(phcbase(jl), zmse), znevn), zdprho), pmfu(jl, (jk + 1))); TK_TRACE(1784, zodmax);
                    // F:1785
                    zodmax = tk_max(zodmax, 0.f); TK_TRACE(1785, zodmax);
                    // F:1786
                    zodetr(jl, jk) = tk_min(zodetr(jl, jk), zodmax); TK_TRACE(1786, zodetr(jl, jk));
                // F:1787
                }
                // F:1788
                zodetr(jl, jk) = tk_min(zodetr(jl, jk), tk_mul(0.75f, pmfu(jl, jk))); TK_TRACE(1788, zodetr(jl, jk));
                // F:1789
                pmfu(jl, jk) = tk_sub(tk_add(pmfu(jl, jk), zoentr(jl, jk)), zodetr(jl, jk)); TK_TRACE(1789, pmfu(jl, jk));
                // F:1790
                zqeen = tk_mul(pqenh(jl, (jk + 1)), zdmfen(jl)); TK_TRACE(1790, zqeen);
                // F:1791
                zqeen = tk_add(zqeen, tk_mul(pqenh(jl, (jk + 1)), zoentr(jl, jk))); TK_TRACE(1791, zqeen);
                // F:1792
                zseen = tk_mul(tk_add(tk_mul(tk_cpd, ptenh(jl, (jk + 1))), pgeoh(jl, (jk + 1))), zdmfen(jl)); TK_TRACE(1792, zseen);
                // F:1793
                zseen = tk_add(zseen, tk_mul(tk_add(tk_mul(tk_cpd, ptenh(jl, (jk + 1))), pgeoh(jl, (jk + 1))), zoentr(jl, jk))); TK_TRACE(1793, zseen);
                // F:1795
                zscde = tk_mul(tk_add(tk_mul(tk_cpd, ptu(jl, (jk + 1))), pgeoh(jl, (jk + 1))), zdmfde(jl)); TK_TRACE(1795, zscde);
                // F:1797
                zga = tk_div(tk_mul(tk_alv, pqsenh(jl, (jk + 1))), tk_mul(tk_rv, tk_mul(ptenh(jl, (jk + 1)), ptenh(jl, (jk + 1))))); TK_TRACE(1797, zga);
                // F:1798
                zdt = tk_div(tk_sub(plu(jl, (jk + 1)), tk_mul(0.608f, tk_sub(pqsenh(jl, (jk + 1)), pqenh(jl, (jk + 1))))), tk_add(tk_div(1.f, ptenh(jl, (jk + 1))), tk_mul(0.608f, zga))); TK_TRACE(1798, zdt);
                // F:1800
                zscod = tk_add(tk_add(tk_mul(tk_cpd, ptenh(jl, (jk + 1))), pgeoh(jl, (jk + 1))), tk_mul(tk_cpd, zdt)); TK_TRACE(1800, zscod);
                // F:1801
                zscde = tk_add(zscde, tk_mul(zodetr(jl, jk), zscod)); TK_TRACE(1801, zscde);
                // F:1802
                zqude = tk_mul(pqu(jl, (jk + 1)), zdmfde(jl)); TK_TRACE(1802, zqude);
                // F:1803
                zqcod = tk_add(pqsenh(jl, (jk + 1)), tk_mul(zga, zdt)); TK_TRACE(1803, zqcod);
                // F:1804
                zqude = tk_add(zqude, tk_mul(zodetr(jl, jk), zqcod)); TK_TRACE(1804, zqude);
                // F:1805
                plude(jl, jk) = tk_mul(plu(jl, (jk + 1)), zdmfde(jl)); TK_TRACE(1805, plude(jl, jk));
                // F:1806
                plude(jl, jk) = tk_add(plude(jl, jk), tk_mul(plu(jl, (jk + 1)), zodetr(jl, jk))); TK_TRACE(1806, plude(jl, jk));
                // F:1807
                zmfusk = tk_sub(tk_add(pmfus(jl, (jk + 1)), zseen), zscde); TK_TRACE(1807, zmfusk);
                // F:1808
                zmfuqk = tk_sub(tk_add(pmfuq(jl, (jk + 1)), zqeen), zqude); TK_TRACE(1808, zmfuqk);
                // F:1809
                zmfulk = tk_sub(pmful(jl, (jk + 1)), plude(jl, jk)); TK_TRACE(1809, zmfulk);
                // F:1810
                plu(jl, jk) = tk_mul(zmfulk, tk_div(1.f, tk_max(tk_cmfcmin, pmfu(jl, jk)))); TK_TRACE(1810, plu(jl, jk));
                // F:1811
                pqu(jl, jk) = tk_mul(zmfuqk, tk_div(1.f, tk_max(tk_cmfcmin, pmfu(jl, jk)))); TK_TRACE(1811, pqu(jl, jk));
                // F:1812
                ptu(jl, jk) = tk_mul(tk_sub(tk_mul(zmfusk, tk_div(1.f, tk_max(tk_cmfcmin, pmfu(jl, jk)))), pgeoh(jl, jk)), tk_rcpd); TK_TRACE(1812, ptu(jl, jk));
                // F:1814
                ptu(jl, jk) = tk_max(100.f, ptu(jl, jk)); TK_TRACE(1814, ptu(jl, jk));
                // F:1815
                ptu(jl, jk) = tk_min(400.f, ptu(jl, jk)); TK_TRACE(1815, ptu(jl, jk));
                // F:1816
                zqold(jl) = pqu(jl, jk); TK_TRACE(1816, zqold(jl));
            // F:1817
            }
        // F:1818
        }
        // F:1822
        ik = jk;
        // F:1823
        icall = 1;
        // F:1825
        tk_cuadjtq(klon, klev, ik, zph, ptu, pqu, loflag, icall, arena);
        // F:1827
        const int tk_limit_1827 = klon;
        for (jl = 1; jl <= tk_limit_1827; jl += 1) {
            // F:1828
            if ((loflag(jl) & (pqu(jl, jk) != zqold(jl)))) {
                // F:1829
                klab(jl, jk) = 2;
                // F:1830
                plu(jl, jk) = tk_sub(tk_add(plu(jl, jk), zqold(jl)), pqu(jl, jk)); TK_TRACE(1830, plu(jl, jk));
                // F:1831
                zbuo = tk_sub(tk_mul(ptu(jl, jk), tk_sub(tk_add(1.f, tk_mul(tk_vtmpc1, pqu(jl, jk))), plu(jl, jk))), tk_mul(ptenh(jl, jk), tk_add(1.f, tk_mul(tk_vtmpc1, pqenh(jl, jk))))); TK_TRACE(1831, zbuo);
                // F:1833
                if ((klab(jl, (jk + 1)) == 1)) { zbuo = tk_add(zbuo, tk_zbuo0); TK_TRACE(1833, zbuo); }
                // F:1834
                if ((((zbuo > 0.f) & (pmfu(jl, jk) > tk_mul(0.01f, pmfub(jl)))) & (jk >= kctop0(jl)))) {
                    // F:1836
                    kctop(jl) = jk;
                    // F:1837
                    ldcum(jl) = true;
                    // F:1838
                    if ((tk_sub(zpbase(jl), paph(jl, jk)) >= tk_zdnoprc)) {
                        // F:1839
                        zprcon = tk_cprcon; TK_TRACE(1839, zprcon);
                    // F:1840
                    } else {
                        // F:1841
                        zprcon = 0.f; TK_TRACE(1841, zprcon);
                    // F:1842
                    }
                    // F:1843
                    zlnew = tk_div(plu(jl, jk), tk_add(1.f, tk_mul(zprcon, tk_sub(pgeoh(jl, jk), pgeoh(jl, (jk + 1)))))); TK_TRACE(1843, zlnew);
                    // F:1844
                    pdmfup(jl, jk) = tk_max(0.f, tk_mul(tk_sub(plu(jl, jk), zlnew), pmfu(jl, jk))); TK_TRACE(1844, pdmfup(jl, jk));
                    // F:1845
                    plu(jl, jk) = zlnew; TK_TRACE(1845, plu(jl, jk));
                // F:1846
                } else {
                    // F:1847
                    klab(jl, jk) = 0;
                    // F:1848
                    pmfu(jl, jk) = 0.f; TK_TRACE(1848, pmfu(jl, jk));
                // F:1849
                }
            // F:1850
            }
            // F:1851
            if (loflag(jl)) {
                // F:1852
                pmful(jl, jk) = tk_mul(plu(jl, jk), pmfu(jl, jk)); TK_TRACE(1852, pmful(jl, jk));
                // F:1853
                pmfus(jl, jk) = tk_mul(tk_add(tk_mul(tk_cpd, ptu(jl, jk)), pgeoh(jl, jk)), pmfu(jl, jk)); TK_TRACE(1853, pmfus(jl, jk));
                // F:1854
                pmfuq(jl, jk) = tk_mul(pqu(jl, jk), pmfu(jl, jk)); TK_TRACE(1854, pmfuq(jl, jk));
            // F:1855
            }
        // F:1856
        }
        // F:1858
        if (tk_lmfdudv) {
            // F:1860
            const int tk_limit_1860 = klon;
            for (jl = 1; jl <= tk_limit_1860; jl += 1) {
                // F:1861
                zdmfen(jl) = tk_add(zdmfen(jl), zoentr(jl, jk)); TK_TRACE(1861, zdmfen(jl));
                // F:1862
                zdmfde(jl) = tk_add(zdmfde(jl), zodetr(jl, jk)); TK_TRACE(1862, zdmfde(jl));
                // F:1863
                if (loflag(jl)) {
                    // F:1864
                    if (((ktype(jl) == 1) | (ktype(jl) == 3))) {
                        // F:1865
                        if ((zdmfen(jl) <= 1.e-20f)) {
                            // F:1866
                            zz = 3.f; TK_TRACE(1866, zz);
                        // F:1867
                        } else {
                            // F:1868
                            zz = 2.f; TK_TRACE(1868, zz);
                        // F:1869
                        }
                    // F:1870
                    } else {
                        // F:1871
                        if ((zdmfen(jl) <= 1.0e-20f)) {
                            // F:1872
                            zz = 1.f; TK_TRACE(1872, zz);
                        // F:1873
                        } else {
                            // F:1874
                            zz = 0.f; TK_TRACE(1874, zz);
                        // F:1875
                        }
                    // F:1876
                    }
                    // F:1877
                    zdmfeu = tk_add(zdmfen(jl), tk_mul(zz, zdmfde(jl))); TK_TRACE(1877, zdmfeu);
                    // F:1878
                    zdmfdu = tk_add(zdmfde(jl), tk_mul(zz, zdmfde(jl))); TK_TRACE(1878, zdmfdu);
                    // F:1879
                    zdmfdu = tk_min(zdmfdu, tk_mul(0.75f, pmfu(jl, (jk + 1)))); TK_TRACE(1879, zdmfdu);
                    // F:1880
                    zmfuu(jl) = tk_sub(tk_add(zmfuu(jl), tk_mul(zdmfeu, puen(jl, jk))), tk_mul(zdmfdu, puu(jl, (jk + 1)))); TK_TRACE(1880, zmfuu(jl));
                    // F:1882
                    zmfuv(jl) = tk_sub(tk_add(zmfuv(jl), tk_mul(zdmfeu, pven(jl, jk))), tk_mul(zdmfdu, pvu(jl, (jk + 1)))); TK_TRACE(1882, zmfuv(jl));
                    // F:1884
                    if ((pmfu(jl, jk) > 0.f)) {
                        // F:1885
                        puu(jl, jk) = tk_mul(zmfuu(jl), tk_div(1.f, pmfu(jl, jk))); TK_TRACE(1885, puu(jl, jk));
                        // F:1886
                        pvu(jl, jk) = tk_mul(zmfuv(jl), tk_div(1.f, pmfu(jl, jk))); TK_TRACE(1886, pvu(jl, jk));
                    // F:1887
                    }
                // F:1888
                }
            // F:1889
            }
        // F:1891
        }
        // F:1896
        const int tk_limit_1896 = klon;
        for (jl = 1; jl <= tk_limit_1896; jl += 1) {
            // F:1897
            if ((loflag(jl) & (ktype(jl) == 1))) {
                // F:1898
                zbuoyz = tk_mul(tk_g, tk_sub(tk_add(tk_div(tk_sub(ptu(jl, jk), ptenh(jl, jk)), ptenh(jl, jk)), tk_mul(0.608f, tk_sub(pqu(jl, jk), pqenh(jl, jk)))), plu(jl, jk))); TK_TRACE(1898, zbuoyz);
                // F:1900
                zbuoyz = tk_max(zbuoyz, 0.0f); TK_TRACE(1900, zbuoyz);
                // F:1901
                zdz = tk_mul(tk_sub(pgeo(jl, (jk - 1)), pgeo(jl, jk)), tk_zrg); TK_TRACE(1901, zdz);
                // F:1902
                zdrodz = tk_sub(tk_div((-gfk_log(tk_div(pten(jl, (jk - 1)), pten(jl, jk)))), zdz), tk_div(tk_g, tk_mul(tk_rd, ptenh(jl, jk)))); TK_TRACE(1902, zdrodz);
                // F:1904
                zbuoy(jl) = tk_add(zbuoy(jl), tk_mul(zbuoyz, zdz)); TK_TRACE(1904, zbuoy(jl));
                // F:1905
                zoentr(jl, (jk - 1)) = tk_add(tk_div(tk_mul(zbuoyz, 0.5f), tk_add(1.f, zbuoy(jl))), zdrodz); TK_TRACE(1905, zoentr(jl, (jk - 1)));
                // F:1906
                zoentr(jl, (jk - 1)) = tk_min(zoentr(jl, (jk - 1)), 1.e-3f); TK_TRACE(1906, zoentr(jl, (jk - 1)));
                // F:1907
                zoentr(jl, (jk - 1)) = tk_max(zoentr(jl, (jk - 1)), 0.f); TK_TRACE(1907, zoentr(jl, (jk - 1)));
            // F:1908
            }
        // F:1909
        }
    // F:1911
    }
    // F:1918
    const int tk_limit_1918 = klon;
    for (jl = 1; jl <= tk_limit_1918; jl += 1) {
        // F:1919
        if ((kctop(jl) == klevm1)) { ldcum(jl) = false; }
        // F:1920
        kcbot(jl) = tk_max(kcbot(jl), kctop(jl));
    // F:1921
    }
    // F:1923
    is = 0;
    // F:1924
    const int tk_limit_1924 = klon;
    for (jl = 1; jl <= tk_limit_1924; jl += 1) {
        // F:1925
        if (ldcum(jl)) {
            // F:1926
            is = (is + 1);
        // F:1927
        }
    // F:1928
    }
    // F:1929
    kcum = is;
    // F:1930
    if ((is == 0)) { return; }
    // F:1931
    const int tk_limit_1931 = klon;
    for (jl = 1; jl <= tk_limit_1931; jl += 1) {
        // F:1932
        if (ldcum(jl)) {
            // F:1933
            jk = (kctop(jl) - 1);
            // F:1934
            zzdmf = tk_cmfctop; TK_TRACE(1934, zzdmf);
            // F:1935
            zdmfde(jl) = tk_mul(tk_sub(1.f, zzdmf), pmfu(jl, (jk + 1))); TK_TRACE(1935, zdmfde(jl));
            // F:1936
            plude(jl, jk) = tk_mul(zdmfde(jl), plu(jl, (jk + 1))); TK_TRACE(1936, plude(jl, jk));
            // F:1937
            pmfu(jl, jk) = tk_sub(pmfu(jl, (jk + 1)), zdmfde(jl)); TK_TRACE(1937, pmfu(jl, jk));
            // F:1938
            pmfus(jl, jk) = tk_mul(tk_add(tk_mul(tk_cpd, ptu(jl, jk)), pgeoh(jl, jk)), pmfu(jl, jk)); TK_TRACE(1938, pmfus(jl, jk));
            // F:1939
            pmfuq(jl, jk) = tk_mul(pqu(jl, jk), pmfu(jl, jk)); TK_TRACE(1939, pmfuq(jl, jk));
            // F:1940
            pmful(jl, jk) = tk_mul(plu(jl, jk), pmfu(jl, jk)); TK_TRACE(1940, pmful(jl, jk));
            // F:1941
            plude(jl, (jk - 1)) = pmful(jl, jk); TK_TRACE(1941, plude(jl, (jk - 1)));
            // F:1942
            pdmfup(jl, jk) = 0.f; TK_TRACE(1942, pdmfup(jl, jk));
        // F:1943
        }
    // F:1944
    }
    // F:1946
    if (tk_lmfdudv) {
        // F:1947
        const int tk_limit_1947 = klon;
        for (jl = 1; jl <= tk_limit_1947; jl += 1) {
            // F:1948
            if (ldcum(jl)) {
                // F:1949
                jk = (kctop(jl) - 1);
                // F:1950
                puu(jl, jk) = puu(jl, (jk + 1)); TK_TRACE(1950, puu(jl, jk));
                // F:1951
                pvu(jl, jk) = pvu(jl, (jk + 1)); TK_TRACE(1951, pvu(jl, jk));
            // F:1952
            }
        // F:1953
        }
    // F:1954
    }
    // F:1955
    return;
}

__device__ void tk_cudlfs(int klon, int klev, int klevp1, TkArray<float> ptenh, TkArray<float> pqenh, TkArray<float> puen, TkArray<float> pven, TkArray<float> pgeoh, TkArray<float> paph, TkArray<float> ptu, TkArray<float> pqu, TkArray<float> puu, TkArray<float> pvu, TkArray<bool> ldcum, TkArray<int> kcbot, TkArray<int> kctop, TkArray<float> pmfub, TkArray<float> prfl, TkArray<float> ptd, TkArray<float> pqd, TkArray<float> pud, TkArray<float> pvd, TkArray<float> pmfd, TkArray<float> pmfds, TkArray<float> pmfdq, TkArray<float> pdmfdp, TkArray<int> kdtop, TkArray<bool> lddraf, TkArena arena) {
    // module_cu_tiedtke.F:1962
    int jl;
    int ke;
    int jk;
    int is;
    int ik;
    int icall;
    float zttest;
    float zqtest;
    float zbuo;
    float zmftop;
    TkArray<float> ztenwb = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zqenwb = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zcond = arena.take<float>((klon), klon);
    TkArray<float> zph = arena.take<float>((klon), klon);
    TkArray<bool> llo2 = arena.take<bool>((klon), klon);
    // F:2016
    const int tk_limit_2016 = klon;
    for (jl = 1; jl <= tk_limit_2016; jl += 1) {
        // F:2017
        lddraf(jl) = false;
        // F:2018
        kdtop(jl) = klevp1;
    // F:2019
    }
    // F:2020
    if ((!tk_lmfdd)) { return; }
    // F:2033
    ke = (klev - 3);
    // F:2034
    const int tk_limit_2034 = ke;
    for (jk = 3; jk <= tk_limit_2034; jk += 1) {
        // F:2038
        is = 0;
        // F:2039
        const int tk_limit_2039 = klon;
        for (jl = 1; jl <= tk_limit_2039; jl += 1) {
            // F:2040
            ztenwb(jl, jk) = ptenh(jl, jk); TK_TRACE(2040, ztenwb(jl, jk));
            // F:2041
            zqenwb(jl, jk) = pqenh(jl, jk); TK_TRACE(2041, zqenwb(jl, jk));
            // F:2042
            zph(jl) = paph(jl, jk); TK_TRACE(2042, zph(jl));
            // F:2043
            llo2(jl) = (((ldcum(jl) & (prfl(jl) > 0.f)) & (!lddraf(jl))) & ((jk < kcbot(jl)) & (jk > kctop(jl))));
            // F:2045
            if (llo2(jl)) {
                // F:2046
                is = (is + 1);
            // F:2047
            }
        // F:2048
        }
        // F:2050
        if ((is == 0)) { continue; }
        // F:2051
        ik = jk;
        // F:2052
        icall = 2;
        // F:2053
        tk_cuadjtq(klon, klev, ik, zph, ztenwb, zqenwb, llo2, icall, arena);
        // F:2058
        const int tk_limit_2058 = klon;
        for (jl = 1; jl <= tk_limit_2058; jl += 1) {
            // F:2059
            if (llo2(jl)) {
                // F:2060
                zttest = tk_mul(0.5f, tk_add(ptu(jl, jk), ztenwb(jl, jk))); TK_TRACE(2060, zttest);
                // F:2061
                zqtest = tk_mul(0.5f, tk_add(pqu(jl, jk), zqenwb(jl, jk))); TK_TRACE(2061, zqtest);
                // F:2062
                zbuo = tk_sub(tk_mul(zttest, tk_add(1.f, tk_mul(tk_vtmpc1, zqtest))), tk_mul(ptenh(jl, jk), tk_add(1.f, tk_mul(tk_vtmpc1, pqenh(jl, jk))))); TK_TRACE(2062, zbuo);
                // F:2064
                zcond(jl) = tk_sub(pqenh(jl, jk), zqenwb(jl, jk)); TK_TRACE(2064, zcond(jl));
                // F:2065
                zmftop = tk_mul((-tk_cmfdeps), pmfub(jl)); TK_TRACE(2065, zmftop);
                // F:2066
                if (((zbuo < 0.f) & (prfl(jl) > tk_mul(tk_mul(10.f, zmftop), zcond(jl))))) {
                    // F:2067
                    kdtop(jl) = jk;
                    // F:2068
                    lddraf(jl) = true;
                    // F:2069
                    ptd(jl, jk) = zttest; TK_TRACE(2069, ptd(jl, jk));
                    // F:2070
                    pqd(jl, jk) = zqtest; TK_TRACE(2070, pqd(jl, jk));
                    // F:2071
                    pmfd(jl, jk) = zmftop; TK_TRACE(2071, pmfd(jl, jk));
                    // F:2072
                    pmfds(jl, jk) = tk_mul(pmfd(jl, jk), tk_add(tk_mul(tk_cpd, ptd(jl, jk)), pgeoh(jl, jk))); TK_TRACE(2072, pmfds(jl, jk));
                    // F:2073
                    pmfdq(jl, jk) = tk_mul(pmfd(jl, jk), pqd(jl, jk)); TK_TRACE(2073, pmfdq(jl, jk));
                    // F:2074
                    pdmfdp(jl, (jk - 1)) = tk_mul(tk_mul((-0.5f), pmfd(jl, jk)), zcond(jl)); TK_TRACE(2074, pdmfdp(jl, (jk - 1)));
                    // F:2075
                    prfl(jl) = tk_add(prfl(jl), pdmfdp(jl, (jk - 1))); TK_TRACE(2075, prfl(jl));
                // F:2076
                }
            // F:2077
            }
        // F:2078
        }
        // F:2080
        if (tk_lmfdudv) {
            // F:2081
            const int tk_limit_2081 = klon;
            for (jl = 1; jl <= tk_limit_2081; jl += 1) {
                // F:2082
                if ((pmfd(jl, jk) < 0.f)) {
                    // F:2083
                    pud(jl, jk) = tk_mul(0.5f, tk_add(puu(jl, jk), puen(jl, (jk - 1)))); TK_TRACE(2083, pud(jl, jk));
                    // F:2084
                    pvd(jl, jk) = tk_mul(0.5f, tk_add(pvu(jl, jk), pven(jl, (jk - 1)))); TK_TRACE(2084, pvd(jl, jk));
                // F:2085
                }
            // F:2086
            }
        // F:2087
        }
    // F:2089
    }
    // F:2090
    return;
}

__device__ void tk_cuddraf(int klon, int klev, int klevp1, TkArray<float> ptenh, TkArray<float> pqenh, TkArray<float> puen, TkArray<float> pven, TkArray<float> pgeoh, TkArray<float> paph, TkArray<float> prfl, TkArray<bool> lddraf, TkArray<float> ptd, TkArray<float> pqd, TkArray<float> pud, TkArray<float> pvd, TkArray<float> pmfd, TkArray<float> pmfds, TkArray<float> pmfdq, TkArray<float> pdmfdp, TkArena arena) {
    // module_cu_tiedtke.F:2097
    int jk;
    int is;
    int jl;
    int itopde;
    int ik;
    int icall;
    float zentr;
    float zseen;
    float zqeen;
    float zsdde;
    float zqdde;
    float zmfdsk;
    float zmfdqk;
    float zbuo;
    float zdmfdp;
    float zmfduk;
    float zmfdvk;
    TkArray<float> zdmfen = arena.take<float>((klon), klon);
    TkArray<float> zdmfde = arena.take<float>((klon), klon);
    TkArray<float> zcond = arena.take<float>((klon), klon);
    TkArray<float> zph = arena.take<float>((klon), klon);
    TkArray<bool> llo2 = arena.take<bool>((klon), klon);
    // F:2154
    const int tk_limit_2154 = klev;
    for (jk = 3; jk <= tk_limit_2154; jk += 1) {
        // F:2155
        is = 0;
        // F:2156
        const int tk_limit_2156 = klon;
        for (jl = 1; jl <= tk_limit_2156; jl += 1) {
            // F:2157
            zph(jl) = paph(jl, jk); TK_TRACE(2157, zph(jl));
            // F:2158
            llo2(jl) = (lddraf(jl) & (pmfd(jl, (jk - 1)) < 0.f));
            // F:2159
            if (llo2(jl)) {
                // F:2160
                is = (is + 1);
            // F:2161
            }
        // F:2162
        }
        // F:2164
        if ((is == 0)) { continue; }
        // F:2165
        const int tk_limit_2165 = klon;
        for (jl = 1; jl <= tk_limit_2165; jl += 1) {
            // F:2166
            if (llo2(jl)) {
                // F:2167
                zentr = tk_mul(tk_div(tk_mul(tk_mul(tk_mul(tk_entrdd, pmfd(jl, (jk - 1))), tk_rd), ptenh(jl, (jk - 1))), tk_mul(tk_g, paph(jl, (jk - 1)))), tk_sub(paph(jl, jk), paph(jl, (jk - 1)))); TK_TRACE(2167, zentr);
                // F:2169
                zdmfen(jl) = zentr; TK_TRACE(2169, zdmfen(jl));
                // F:2170
                zdmfde(jl) = zentr; TK_TRACE(2170, zdmfde(jl));
            // F:2171
            }
        // F:2172
        }
        // F:2174
        itopde = (klev - 2);
        // F:2175
        if ((jk > itopde)) {
            // F:2176
            const int tk_limit_2176 = klon;
            for (jl = 1; jl <= tk_limit_2176; jl += 1) {
                // F:2177
                if (llo2(jl)) {
                    // F:2178
                    zdmfen(jl) = 0.f; TK_TRACE(2178, zdmfen(jl));
                    // F:2179
                    zdmfde(jl) = tk_div(tk_mul(pmfd(jl, itopde), tk_sub(paph(jl, jk), paph(jl, (jk - 1)))), tk_sub(paph(jl, klevp1), paph(jl, itopde))); TK_TRACE(2179, zdmfde(jl));
                // F:2182
                }
            // F:2183
            }
        // F:2184
        }
        // F:2186
        const int tk_limit_2186 = klon;
        for (jl = 1; jl <= tk_limit_2186; jl += 1) {
            // F:2187
            if (llo2(jl)) {
                // F:2188
                pmfd(jl, jk) = tk_sub(tk_add(pmfd(jl, (jk - 1)), zdmfen(jl)), zdmfde(jl)); TK_TRACE(2188, pmfd(jl, jk));
                // F:2189
                zseen = tk_mul(tk_add(tk_mul(tk_cpd, ptenh(jl, (jk - 1))), pgeoh(jl, (jk - 1))), zdmfen(jl)); TK_TRACE(2189, zseen);
                // F:2190
                zqeen = tk_mul(pqenh(jl, (jk - 1)), zdmfen(jl)); TK_TRACE(2190, zqeen);
                // F:2191
                zsdde = tk_mul(tk_add(tk_mul(tk_cpd, ptd(jl, (jk - 1))), pgeoh(jl, (jk - 1))), zdmfde(jl)); TK_TRACE(2191, zsdde);
                // F:2192
                zqdde = tk_mul(pqd(jl, (jk - 1)), zdmfde(jl)); TK_TRACE(2192, zqdde);
                // F:2193
                zmfdsk = tk_sub(tk_add(pmfds(jl, (jk - 1)), zseen), zsdde); TK_TRACE(2193, zmfdsk);
                // F:2194
                zmfdqk = tk_sub(tk_add(pmfdq(jl, (jk - 1)), zqeen), zqdde); TK_TRACE(2194, zmfdqk);
                // F:2195
                pqd(jl, jk) = tk_mul(zmfdqk, tk_div(1.f, tk_min((-tk_cmfcmin), pmfd(jl, jk)))); TK_TRACE(2195, pqd(jl, jk));
                // F:2196
                ptd(jl, jk) = tk_mul(tk_sub(tk_mul(zmfdsk, tk_div(1.f, tk_min((-tk_cmfcmin), pmfd(jl, jk)))), pgeoh(jl, jk)), tk_rcpd); TK_TRACE(2196, ptd(jl, jk));
                // F:2198
                ptd(jl, jk) = tk_min(400.f, ptd(jl, jk)); TK_TRACE(2198, ptd(jl, jk));
                // F:2199
                ptd(jl, jk) = tk_max(100.f, ptd(jl, jk)); TK_TRACE(2199, ptd(jl, jk));
                // F:2200
                zcond(jl) = pqd(jl, jk); TK_TRACE(2200, zcond(jl));
            // F:2201
            }
        // F:2202
        }
        // F:2204
        ik = jk;
        // F:2205
        icall = 2;
        // F:2206
        tk_cuadjtq(klon, klev, ik, zph, ptd, pqd, llo2, icall, arena);
        // F:2207
        const int tk_limit_2207 = klon;
        for (jl = 1; jl <= tk_limit_2207; jl += 1) {
            // F:2208
            if (llo2(jl)) {
                // F:2209
                zcond(jl) = tk_sub(zcond(jl), pqd(jl, jk)); TK_TRACE(2209, zcond(jl));
                // F:2210
                zbuo = tk_sub(tk_mul(ptd(jl, jk), tk_add(1.f, tk_mul(tk_vtmpc1, pqd(jl, jk)))), tk_mul(ptenh(jl, jk), tk_add(1.f, tk_mul(tk_vtmpc1, pqenh(jl, jk))))); TK_TRACE(2210, zbuo);
                // F:2212
                if (((zbuo >= 0.f) | (prfl(jl) <= tk_mul(pmfd(jl, jk), zcond(jl))))) {
                    // F:2213
                    pmfd(jl, jk) = 0.f; TK_TRACE(2213, pmfd(jl, jk));
                // F:2214
                }
                // F:2215
                pmfds(jl, jk) = tk_mul(tk_add(tk_mul(tk_cpd, ptd(jl, jk)), pgeoh(jl, jk)), pmfd(jl, jk)); TK_TRACE(2215, pmfds(jl, jk));
                // F:2216
                pmfdq(jl, jk) = tk_mul(pqd(jl, jk), pmfd(jl, jk)); TK_TRACE(2216, pmfdq(jl, jk));
                // F:2217
                zdmfdp = tk_mul((-pmfd(jl, jk)), zcond(jl)); TK_TRACE(2217, zdmfdp);
                // F:2218
                pdmfdp(jl, (jk - 1)) = zdmfdp; TK_TRACE(2218, pdmfdp(jl, (jk - 1)));
                // F:2219
                prfl(jl) = tk_add(prfl(jl), zdmfdp); TK_TRACE(2219, prfl(jl));
            // F:2220
            }
        // F:2221
        }
        // F:2223
        if (tk_lmfdudv) {
            // F:2224
            const int tk_limit_2224 = klon;
            for (jl = 1; jl <= tk_limit_2224; jl += 1) {
                // F:2225
                if ((llo2(jl) & (pmfd(jl, jk) < 0.f))) {
                    // F:2226
                    zmfduk = tk_sub(tk_add(tk_mul(pmfd(jl, (jk - 1)), pud(jl, (jk - 1))), tk_mul(zdmfen(jl), puen(jl, (jk - 1)))), tk_mul(zdmfde(jl), pud(jl, (jk - 1)))); TK_TRACE(2226, zmfduk);
                    // F:2228
                    zmfdvk = tk_sub(tk_add(tk_mul(pmfd(jl, (jk - 1)), pvd(jl, (jk - 1))), tk_mul(zdmfen(jl), pven(jl, (jk - 1)))), tk_mul(zdmfde(jl), pvd(jl, (jk - 1)))); TK_TRACE(2228, zmfdvk);
                    // F:2230
                    pud(jl, jk) = tk_mul(zmfduk, tk_div(1.f, tk_min((-tk_cmfcmin), pmfd(jl, jk)))); TK_TRACE(2230, pud(jl, jk));
                    // F:2231
                    pvd(jl, jk) = tk_mul(zmfdvk, tk_div(1.f, tk_min((-tk_cmfcmin), pmfd(jl, jk)))); TK_TRACE(2231, pvd(jl, jk));
                // F:2232
                }
            // F:2233
            }
        // F:2234
        }
    // F:2236
    }
    // F:2237
    return;
}

__device__ void tk_cuflx(int klon, int klev, int klevp1, TkArray<float> pqen, TkArray<float> pqsen, TkArray<float> ptenh, TkArray<float> pqenh, TkArray<float> paph, TkArray<float> pgeoh, TkArray<int> kcbot, TkArray<int> kctop, TkArray<int> kdtop, TkArray<int> ktype, TkArray<bool> lddraf, TkArray<bool> ldcum, TkArray<float> pmfu, TkArray<float> pmfd, TkArray<float> pmfus, TkArray<float> pmfds, TkArray<float> pmfuq, TkArray<float> pmfdq, TkArray<float> pmful, TkArray<float> plude, TkArray<float> pdmfup, TkArray<float> pdmfdp, TkArray<float> prfl, TkArray<float> prain, TkArray<float> pten, TkArray<float> psfl, TkArray<float> pdpmel, int& ktopm2, float ztmst, TkArray<float> sig1, TkArena arena) {
    // module_cu_tiedtke.F:2244
    int itop;
    int jl;
    int jk;
    int ikb;
    float zcons1;
    float zcons2;
    float zcucov;
    float ztmelp2;
    float zzp;
    float zfac;
    float zsnmlt;
    float zrfl;
    float cevapcu;
    float zrnew;
    float zrmin;
    float zrfln;
    float zdrfl;
    float zdpevap;
    TkArray<float> zpsubcl = arena.take<float>((klon), klon);
    // F:2293
    zcons1 = tk_div(tk_cpd, tk_mul(tk_mul(tk_alf, tk_g), ztmst)); TK_TRACE(2293, zcons1);
    // F:2294
    zcons2 = tk_div(1.f, tk_mul(tk_g, ztmst)); TK_TRACE(2294, zcons2);
    // F:2295
    zcucov = 0.05f; TK_TRACE(2295, zcucov);
    // F:2296
    ztmelp2 = tk_add(tk_tmelt, 2.f); TK_TRACE(2296, ztmelp2);
    // F:2299
    itop = klev;
    // F:2300
    const int tk_limit_2300 = klon;
    for (jl = 1; jl <= tk_limit_2300; jl += 1) {
        // F:2301
        prfl(jl) = 0.f; TK_TRACE(2301, prfl(jl));
        // F:2302
        psfl(jl) = 0.f; TK_TRACE(2302, psfl(jl));
        // F:2303
        prain(jl) = 0.f; TK_TRACE(2303, prain(jl));
        // F:2305
        if (((!tk_lmfscv) & (ktype(jl) == 2))) {
            // F:2306
            ldcum(jl) = false;
            // F:2307
            lddraf(jl) = false;
        // F:2308
        }
        // F:2309
        itop = tk_min(itop, kctop(jl));
        // F:2310
        if (((!ldcum(jl)) | (kdtop(jl) < kctop(jl)))) { lddraf(jl) = false; }
        // F:2311
        if ((!ldcum(jl))) { ktype(jl) = 0; }
    // F:2312
    }
    // F:2314
    ktopm2 = (itop - 2);
    // F:2315
    const int tk_limit_2315 = klev;
    for (jk = ktopm2; jk <= tk_limit_2315; jk += 1) {
        // F:2316
        const int tk_limit_2316 = klon;
        for (jl = 1; jl <= tk_limit_2316; jl += 1) {
            // F:2317
            if ((ldcum(jl) & (jk >= (kctop(jl) - 1)))) {
                // F:2318
                pmfus(jl, jk) = tk_sub(pmfus(jl, jk), tk_mul(pmfu(jl, jk), tk_add(tk_mul(tk_cpd, ptenh(jl, jk)), pgeoh(jl, jk)))); TK_TRACE(2318, pmfus(jl, jk));
                // F:2320
                pmfuq(jl, jk) = tk_sub(pmfuq(jl, jk), tk_mul(pmfu(jl, jk), pqenh(jl, jk))); TK_TRACE(2320, pmfuq(jl, jk));
                // F:2321
                if ((lddraf(jl) & (jk >= kdtop(jl)))) {
                    // F:2322
                    pmfds(jl, jk) = tk_sub(pmfds(jl, jk), tk_mul(pmfd(jl, jk), tk_add(tk_mul(tk_cpd, ptenh(jl, jk)), pgeoh(jl, jk)))); TK_TRACE(2322, pmfds(jl, jk));
                    // F:2324
                    pmfdq(jl, jk) = tk_sub(pmfdq(jl, jk), tk_mul(pmfd(jl, jk), pqenh(jl, jk))); TK_TRACE(2324, pmfdq(jl, jk));
                // F:2325
                } else {
                    // F:2326
                    pmfd(jl, jk) = 0.f; TK_TRACE(2326, pmfd(jl, jk));
                    // F:2327
                    pmfds(jl, jk) = 0.f; TK_TRACE(2327, pmfds(jl, jk));
                    // F:2328
                    pmfdq(jl, jk) = 0.f; TK_TRACE(2328, pmfdq(jl, jk));
                    // F:2329
                    pdmfdp(jl, (jk - 1)) = 0.f; TK_TRACE(2329, pdmfdp(jl, (jk - 1)));
                // F:2330
                }
            // F:2331
            } else {
                // F:2332
                pmfu(jl, jk) = 0.f; TK_TRACE(2332, pmfu(jl, jk));
                // F:2333
                pmfd(jl, jk) = 0.f; TK_TRACE(2333, pmfd(jl, jk));
                // F:2334
                pmfus(jl, jk) = 0.f; TK_TRACE(2334, pmfus(jl, jk));
                // F:2335
                pmfds(jl, jk) = 0.f; TK_TRACE(2335, pmfds(jl, jk));
                // F:2336
                pmfuq(jl, jk) = 0.f; TK_TRACE(2336, pmfuq(jl, jk));
                // F:2337
                pmfdq(jl, jk) = 0.f; TK_TRACE(2337, pmfdq(jl, jk));
                // F:2338
                pmful(jl, jk) = 0.f; TK_TRACE(2338, pmful(jl, jk));
                // F:2339
                pdmfup(jl, (jk - 1)) = 0.f; TK_TRACE(2339, pdmfup(jl, (jk - 1)));
                // F:2340
                pdmfdp(jl, (jk - 1)) = 0.f; TK_TRACE(2340, pdmfdp(jl, (jk - 1)));
                // F:2341
                plude(jl, (jk - 1)) = 0.f; TK_TRACE(2341, plude(jl, (jk - 1)));
            // F:2342
            }
        // F:2343
        }
    // F:2344
    }
    // F:2346
    const int tk_limit_2346 = klev;
    for (jk = ktopm2; jk <= tk_limit_2346; jk += 1) {
        // F:2347
        const int tk_limit_2347 = klon;
        for (jl = 1; jl <= tk_limit_2347; jl += 1) {
            // F:2348
            if ((ldcum(jl) & (jk > kcbot(jl)))) {
                // F:2349
                ikb = kcbot(jl);
                // F:2350
                zzp = tk_div(tk_sub(paph(jl, klevp1), paph(jl, jk)), tk_sub(paph(jl, klevp1), paph(jl, ikb))); TK_TRACE(2350, zzp);
                // F:2352
                if ((ktype(jl) == 3)) {
                    // F:2353
                    zzp = tk_mul(zzp, zzp); TK_TRACE(2353, zzp);
                // F:2354
                }
                // F:2355
                pmfu(jl, jk) = tk_mul(pmfu(jl, ikb), zzp); TK_TRACE(2355, pmfu(jl, jk));
                // F:2356
                pmfus(jl, jk) = tk_mul(pmfus(jl, ikb), zzp); TK_TRACE(2356, pmfus(jl, jk));
                // F:2357
                pmfuq(jl, jk) = tk_mul(pmfuq(jl, ikb), zzp); TK_TRACE(2357, pmfuq(jl, jk));
                // F:2358
                pmful(jl, jk) = tk_mul(pmful(jl, ikb), zzp); TK_TRACE(2358, pmful(jl, jk));
            // F:2359
            }
            // F:2364
            if (ldcum(jl)) {
                // F:2365
                prain(jl) = tk_add(prain(jl), pdmfup(jl, jk)); TK_TRACE(2365, prain(jl));
                // F:2366
                if ((pten(jl, jk) > tk_tmelt)) {
                    // F:2367
                    prfl(jl) = tk_add(tk_add(prfl(jl), pdmfup(jl, jk)), pdmfdp(jl, jk)); TK_TRACE(2367, prfl(jl));
                    // F:2368
                    if (((psfl(jl) > 0.f) & (pten(jl, jk) > ztmelp2))) {
                        // F:2369
                        zfac = tk_mul(zcons1, tk_sub(paph(jl, (jk + 1)), paph(jl, jk))); TK_TRACE(2369, zfac);
                        // F:2370
                        zsnmlt = tk_min(psfl(jl), tk_mul(zfac, tk_sub(pten(jl, jk), ztmelp2))); TK_TRACE(2370, zsnmlt);
                        // F:2371
                        pdpmel(jl, jk) = zsnmlt; TK_TRACE(2371, pdpmel(jl, jk));
                        // F:2372
                        psfl(jl) = tk_sub(psfl(jl), zsnmlt); TK_TRACE(2372, psfl(jl));
                        // F:2373
                        prfl(jl) = tk_add(prfl(jl), zsnmlt); TK_TRACE(2373, prfl(jl));
                    // F:2374
                    }
                // F:2375
                } else {
                    // F:2376
                    psfl(jl) = tk_add(tk_add(psfl(jl), pdmfup(jl, jk)), pdmfdp(jl, jk)); TK_TRACE(2376, psfl(jl));
                // F:2377
                }
            // F:2378
            }
        // F:2379
        }
    // F:2380
    }
    // F:2382
    const int tk_limit_2382 = klon;
    for (jl = 1; jl <= tk_limit_2382; jl += 1) {
        // F:2383
        prfl(jl) = tk_max(prfl(jl), 0.f); TK_TRACE(2383, prfl(jl));
        // F:2384
        psfl(jl) = tk_max(psfl(jl), 0.f); TK_TRACE(2384, psfl(jl));
        // F:2385
        zpsubcl(jl) = tk_add(prfl(jl), psfl(jl)); TK_TRACE(2385, zpsubcl(jl));
    // F:2386
    }
    // F:2388
    const int tk_limit_2388 = klev;
    for (jk = ktopm2; jk <= tk_limit_2388; jk += 1) {
        // F:2389
        const int tk_limit_2389 = klon;
        for (jl = 1; jl <= tk_limit_2389; jl += 1) {
            // F:2390
            if (((ldcum(jl) & (jk >= kcbot(jl))) & (zpsubcl(jl) > 1.e-20f))) {
                // F:2392
                zrfl = zpsubcl(jl); TK_TRACE(2392, zrfl);
                // F:2395
                cevapcu = tk_mul(tk_cevapcu1, sqrtf(tk_mul(tk_cevapcu2, sqrtf(sig1(jl, jk))))); TK_TRACE(2395, cevapcu);
                // F:2399
                zrnew = tk_mul(tk_mul(tk_max(0.f, tk_sub(sqrtf(tk_div(zrfl, zcucov)), tk_mul(tk_mul(cevapcu, tk_sub(paph(jl, (jk + 1)), paph(jl, jk))), tk_max(0.f, tk_sub(pqsen(jl, jk), pqen(jl, jk)))))), tk_max(0.f, tk_sub(sqrtf(tk_div(zrfl, zcucov)), tk_mul(tk_mul(cevapcu, tk_sub(paph(jl, (jk + 1)), paph(jl, jk))), tk_max(0.f, tk_sub(pqsen(jl, jk), pqen(jl, jk))))))), zcucov); TK_TRACE(2399, zrnew);
                // F:2402
                zrmin = tk_sub(zrfl, tk_mul(tk_mul(tk_mul(zcucov, tk_max(0.f, tk_sub(tk_mul(0.8f, pqsen(jl, jk)), pqen(jl, jk)))), zcons2), tk_sub(paph(jl, (jk + 1)), paph(jl, jk)))); TK_TRACE(2402, zrmin);
                // F:2404
                zrnew = tk_max(zrnew, zrmin); TK_TRACE(2404, zrnew);
                // F:2405
                zrfln = tk_max(zrnew, 0.f); TK_TRACE(2405, zrfln);
                // F:2406
                zdrfl = tk_min(0.f, tk_sub(zrfln, zrfl)); TK_TRACE(2406, zdrfl);
                // F:2407
                pdmfup(jl, jk) = tk_add(pdmfup(jl, jk), zdrfl); TK_TRACE(2407, pdmfup(jl, jk));
                // F:2408
                zpsubcl(jl) = zrfln; TK_TRACE(2408, zpsubcl(jl));
            // F:2409
            }
        // F:2410
        }
    // F:2411
    }
    // F:2413
    const int tk_limit_2413 = klon;
    for (jl = 1; jl <= tk_limit_2413; jl += 1) {
        // F:2414
        zdpevap = tk_sub(zpsubcl(jl), tk_add(prfl(jl), psfl(jl))); TK_TRACE(2414, zdpevap);
        // F:2415
        prfl(jl) = tk_add(prfl(jl), tk_mul(tk_mul(zdpevap, prfl(jl)), tk_div(1.f, tk_max(1.e-20f, tk_add(prfl(jl), psfl(jl)))))); TK_TRACE(2415, prfl(jl));
        // F:2417
        psfl(jl) = tk_add(psfl(jl), tk_mul(tk_mul(zdpevap, psfl(jl)), tk_div(1.f, tk_max(1.e-20f, tk_add(prfl(jl), psfl(jl)))))); TK_TRACE(2417, psfl(jl));
    // F:2419
    }
    // F:2421
    return;
}

__device__ void tk_cudtdq(int klon, int klev, int klevp1, int ktopm2, TkArray<float> paph, TkArray<bool> ldcum, TkArray<float> pten, TkArray<float> ptte, TkArray<float> pqte, TkArray<float> pmfus, TkArray<float> pmfds, TkArray<float> pmfuq, TkArray<float> pmfdq, TkArray<float> pmful, TkArray<float> pdmfup, TkArray<float> pdmfdp, float ztmst, TkArray<float> pdpmel, TkArray<float> prain, TkArray<float> prfl, TkArray<float> psfl, float& psrain, float& psevap, float& psheat, float& psmelt, TkArray<float> prsfc, TkArray<float> pssfc, TkArray<float> paprc, TkArray<float> paprsm, TkArray<float> paprs, TkArray<float> pqen, TkArray<float> pqsen, TkArray<float> plude, TkArray<float> pcte, TkArena arena) {
    // module_cu_tiedtke.F:2428
    int jl;
    int jk;
    float zdiagt;
    float zdiagw;
    float zalv;
    float rhk;
    float rhcoe;
    float pldfd;
    float zdtdt;
    float zdqdt;
    TkArray<float> pgeo = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zsheat = arena.take<float>((klon), klon);
    TkArray<float> zmelt = arena.take<float>((klon), klon);
    // F:2468
    zdiagt = ztmst; TK_TRACE(2468, zdiagt);
    // F:2469
    zdiagw = tk_div(zdiagt, tk_rhoh2o); TK_TRACE(2469, zdiagw);
    // F:2473
    const int tk_limit_2473 = klon;
    for (jl = 1; jl <= tk_limit_2473; jl += 1) {
        // F:2474
        zmelt(jl) = 0.f; TK_TRACE(2474, zmelt(jl));
        // F:2475
        zsheat(jl) = 0.f; TK_TRACE(2475, zsheat(jl));
    // F:2476
    }
    // F:2478
    const int tk_limit_2478 = klev;
    for (jk = ktopm2; jk <= tk_limit_2478; jk += 1) {
        // F:2479
        if ((jk < klev)) {
            // F:2480
            const int tk_limit_2480 = klon;
            for (jl = 1; jl <= tk_limit_2480; jl += 1) {
                // F:2481
                if (ldcum(jl)) {
                    // F:2482
                    if ((pten(jl, jk) > tk_tmelt)) {
                        // F:2483
                        zalv = tk_alv; TK_TRACE(2483, zalv);
                    // F:2484
                    } else {
                        // F:2485
                        zalv = tk_als; TK_TRACE(2485, zalv);
                    // F:2486
                    }
                    // F:2487
                    rhk = tk_min(1.0f, tk_div(pqen(jl, jk), pqsen(jl, jk))); TK_TRACE(2487, rhk);
                    // F:2488
                    rhcoe = tk_max(0.0f, tk_div(tk_sub(rhk, tk_rhc), tk_sub(tk_rhm, tk_rhc))); TK_TRACE(2488, rhcoe);
                    // F:2489
                    pldfd = tk_max(0.0f, tk_mul(tk_mul(rhcoe, tk_fdbk), plude(jl, jk))); TK_TRACE(2489, pldfd);
                    // F:2490
                    zdtdt = tk_mul(tk_mul(tk_div(tk_g, tk_sub(paph(jl, (jk + 1)), paph(jl, jk))), tk_rcpd), tk_sub(tk_sub(tk_sub(tk_add(tk_sub(pmfus(jl, (jk + 1)), pmfus(jl, jk)), pmfds(jl, (jk + 1))), pmfds(jl, jk)), tk_mul(tk_alf, pdpmel(jl, jk))), tk_mul(zalv, tk_sub(tk_sub(tk_sub(pmful(jl, (jk + 1)), pmful(jl, jk)), pldfd), tk_add(pdmfup(jl, jk), pdmfdp(jl, jk)))))); TK_TRACE(2490, zdtdt);
                    // F:2495
                    ptte(jl, jk) = tk_add(ptte(jl, jk), zdtdt); TK_TRACE(2495, ptte(jl, jk));
                    // F:2496
                    zdqdt = tk_mul(tk_div(tk_g, tk_sub(paph(jl, (jk + 1)), paph(jl, jk))), tk_sub(tk_sub(tk_sub(tk_add(tk_sub(tk_add(tk_sub(pmfuq(jl, (jk + 1)), pmfuq(jl, jk)), pmfdq(jl, (jk + 1))), pmfdq(jl, jk)), pmful(jl, (jk + 1))), pmful(jl, jk)), pldfd), tk_add(pdmfup(jl, jk), pdmfdp(jl, jk)))); TK_TRACE(2496, zdqdt);
                    // F:2501
                    pqte(jl, jk) = tk_add(pqte(jl, jk), zdqdt); TK_TRACE(2501, pqte(jl, jk));
                    // F:2502
                    pcte(jl, jk) = tk_mul(tk_div(tk_g, tk_sub(paph(jl, (jk + 1)), paph(jl, jk))), pldfd); TK_TRACE(2502, pcte(jl, jk));
                    // F:2503
                    zsheat(jl) = tk_add(zsheat(jl), tk_mul(zalv, tk_add(pdmfup(jl, jk), pdmfdp(jl, jk)))); TK_TRACE(2503, zsheat(jl));
                    // F:2504
                    zmelt(jl) = tk_add(zmelt(jl), pdpmel(jl, jk)); TK_TRACE(2504, zmelt(jl));
                // F:2505
                }
            // F:2506
            }
        // F:2507
        } else {
            // F:2508
            const int tk_limit_2508 = klon;
            for (jl = 1; jl <= tk_limit_2508; jl += 1) {
                // F:2509
                if (ldcum(jl)) {
                    // F:2510
                    if ((pten(jl, jk) > tk_tmelt)) {
                        // F:2511
                        zalv = tk_alv; TK_TRACE(2511, zalv);
                    // F:2512
                    } else {
                        // F:2513
                        zalv = tk_als; TK_TRACE(2513, zalv);
                    // F:2514
                    }
                    // F:2515
                    rhk = tk_min(1.0f, tk_div(pqen(jl, jk), pqsen(jl, jk))); TK_TRACE(2515, rhk);
                    // F:2516
                    rhcoe = tk_max(0.0f, tk_div(tk_sub(rhk, tk_rhc), tk_sub(tk_rhm, tk_rhc))); TK_TRACE(2516, rhcoe);
                    // F:2517
                    pldfd = tk_max(0.0f, tk_mul(tk_mul(rhcoe, tk_fdbk), plude(jl, jk))); TK_TRACE(2517, pldfd);
                    // F:2518
                    zdtdt = tk_mul(tk_mul((-tk_div(tk_g, tk_sub(paph(jl, (jk + 1)), paph(jl, jk)))), tk_rcpd), tk_sub(tk_add(tk_add(pmfus(jl, jk), pmfds(jl, jk)), tk_mul(tk_alf, pdpmel(jl, jk))), tk_mul(zalv, tk_add(tk_add(tk_add(pmful(jl, jk), pdmfup(jl, jk)), pdmfdp(jl, jk)), pldfd)))); TK_TRACE(2518, zdtdt);
                    // F:2521
                    ptte(jl, jk) = tk_add(ptte(jl, jk), zdtdt); TK_TRACE(2521, ptte(jl, jk));
                    // F:2522
                    zdqdt = tk_mul((-tk_div(tk_g, tk_sub(paph(jl, (jk + 1)), paph(jl, jk)))), tk_add(tk_add(tk_add(pmfuq(jl, jk), pmfdq(jl, jk)), pldfd), tk_add(tk_add(pmful(jl, jk), pdmfup(jl, jk)), pdmfdp(jl, jk)))); TK_TRACE(2522, zdqdt);
                    // F:2525
                    pqte(jl, jk) = tk_add(pqte(jl, jk), zdqdt); TK_TRACE(2525, pqte(jl, jk));
                    // F:2526
                    pcte(jl, jk) = tk_mul(tk_div(tk_g, tk_sub(paph(jl, (jk + 1)), paph(jl, jk))), pldfd); TK_TRACE(2526, pcte(jl, jk));
                    // F:2527
                    zsheat(jl) = tk_add(zsheat(jl), tk_mul(zalv, tk_add(pdmfup(jl, jk), pdmfdp(jl, jk)))); TK_TRACE(2527, zsheat(jl));
                    // F:2528
                    zmelt(jl) = tk_add(zmelt(jl), pdpmel(jl, jk)); TK_TRACE(2528, zmelt(jl));
                // F:2529
                }
            // F:2530
            }
        // F:2531
        }
    // F:2532
    }
    // F:2536
    const int tk_limit_2536 = klon;
    for (jl = 1; jl <= tk_limit_2536; jl += 1) {
        // F:2537
        prsfc(jl) = prfl(jl); TK_TRACE(2537, prsfc(jl));
        // F:2538
        pssfc(jl) = psfl(jl); TK_TRACE(2538, pssfc(jl));
        // F:2539
        paprc(jl) = tk_add(paprc(jl), tk_mul(zdiagw, tk_add(prfl(jl), psfl(jl)))); TK_TRACE(2539, paprc(jl));
        // F:2540
        paprs(jl) = tk_add(paprsm(jl), tk_mul(zdiagw, psfl(jl))); TK_TRACE(2540, paprs(jl));
        // F:2541
        psheat = tk_add(psheat, zsheat(jl)); TK_TRACE(2541, psheat);
        // F:2542
        psrain = tk_add(psrain, prain(jl)); TK_TRACE(2542, psrain);
        // F:2543
        psevap = tk_sub(psevap, tk_add(prfl(jl), psfl(jl))); TK_TRACE(2543, psevap);
        // F:2544
        psmelt = tk_add(psmelt, zmelt(jl)); TK_TRACE(2544, psmelt);
    // F:2545
    }
    // F:2546
    psevap = tk_add(psevap, psrain); TK_TRACE(2546, psevap);
    // F:2547
    return;
}

__device__ void tk_cududv(int klon, int klev, int klevp1, int ktopm2, TkArray<int> ktype, TkArray<int> kcbot, TkArray<float> paph, TkArray<bool> ldcum, TkArray<float> puen, TkArray<float> pven, TkArray<float> pvom, TkArray<float> pvol, TkArray<float> puu, TkArray<float> pud, TkArray<float> pvu, TkArray<float> pvd, TkArray<float> pmfu, TkArray<float> pmfd, float& psdiss, TkArena arena) {
    // module_cu_tiedtke.F:2554
    int jk;
    int ik;
    int jl;
    int ikb;
    float zzp;
    float zdudt;
    float zdvdt;
    float zsum;
    TkArray<float> zmfuu = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zmfdu = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zmfuv = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zmfdv = arena.take<float>((klon) * (klev), klon);
    TkArray<float> zdiss = arena.take<float>((klon), klon);
    // F:2586
    const int tk_limit_2586 = klev;
    for (jk = ktopm2; jk <= tk_limit_2586; jk += 1) {
        // F:2587
        ik = (jk - 1);
        // F:2588
        const int tk_limit_2588 = klon;
        for (jl = 1; jl <= tk_limit_2588; jl += 1) {
            // F:2589
            if (ldcum(jl)) {
                // F:2590
                zmfuu(jl, jk) = tk_mul(pmfu(jl, jk), tk_sub(puu(jl, jk), puen(jl, ik))); TK_TRACE(2590, zmfuu(jl, jk));
                // F:2591
                zmfuv(jl, jk) = tk_mul(pmfu(jl, jk), tk_sub(pvu(jl, jk), pven(jl, ik))); TK_TRACE(2591, zmfuv(jl, jk));
                // F:2592
                zmfdu(jl, jk) = tk_mul(pmfd(jl, jk), tk_sub(pud(jl, jk), puen(jl, ik))); TK_TRACE(2592, zmfdu(jl, jk));
                // F:2593
                zmfdv(jl, jk) = tk_mul(pmfd(jl, jk), tk_sub(pvd(jl, jk), pven(jl, ik))); TK_TRACE(2593, zmfdv(jl, jk));
            // F:2594
            }
        // F:2595
        }
    // F:2596
    }
    // F:2598
    const int tk_limit_2598 = klev;
    for (jk = ktopm2; jk <= tk_limit_2598; jk += 1) {
        // F:2599
        const int tk_limit_2599 = klon;
        for (jl = 1; jl <= tk_limit_2599; jl += 1) {
            // F:2600
            if ((ldcum(jl) & (jk > kcbot(jl)))) {
                // F:2601
                ikb = kcbot(jl);
                // F:2602
                zzp = tk_div(tk_sub(paph(jl, klevp1), paph(jl, jk)), tk_sub(paph(jl, klevp1), paph(jl, ikb))); TK_TRACE(2602, zzp);
                // F:2604
                if ((ktype(jl) == 3)) {
                    // F:2605
                    zzp = tk_mul(zzp, zzp); TK_TRACE(2605, zzp);
                // F:2606
                }
                // F:2607
                zmfuu(jl, jk) = tk_mul(zmfuu(jl, ikb), zzp); TK_TRACE(2607, zmfuu(jl, jk));
                // F:2608
                zmfuv(jl, jk) = tk_mul(zmfuv(jl, ikb), zzp); TK_TRACE(2608, zmfuv(jl, jk));
                // F:2609
                zmfdu(jl, jk) = tk_mul(zmfdu(jl, ikb), zzp); TK_TRACE(2609, zmfdu(jl, jk));
                // F:2610
                zmfdv(jl, jk) = tk_mul(zmfdv(jl, ikb), zzp); TK_TRACE(2610, zmfdv(jl, jk));
            // F:2611
            }
        // F:2612
        }
    // F:2613
    }
    // F:2615
    const int tk_limit_2615 = klon;
    for (jl = 1; jl <= tk_limit_2615; jl += 1) {
        // F:2616
        zdiss(jl) = 0.f; TK_TRACE(2616, zdiss(jl));
    // F:2617
    }
    // F:2619
    const int tk_limit_2619 = klev;
    for (jk = ktopm2; jk <= tk_limit_2619; jk += 1) {
        // F:2620
        if ((jk < klev)) {
            // F:2621
            const int tk_limit_2621 = klon;
            for (jl = 1; jl <= tk_limit_2621; jl += 1) {
                // F:2622
                if (ldcum(jl)) {
                    // F:2623
                    zdudt = tk_mul(tk_div(tk_g, tk_sub(paph(jl, (jk + 1)), paph(jl, jk))), tk_sub(tk_add(tk_sub(zmfuu(jl, (jk + 1)), zmfuu(jl, jk)), zmfdu(jl, (jk + 1))), zmfdu(jl, jk))); TK_TRACE(2623, zdudt);
                    // F:2626
                    zdvdt = tk_mul(tk_div(tk_g, tk_sub(paph(jl, (jk + 1)), paph(jl, jk))), tk_sub(tk_add(tk_sub(zmfuv(jl, (jk + 1)), zmfuv(jl, jk)), zmfdv(jl, (jk + 1))), zmfdv(jl, jk))); TK_TRACE(2626, zdvdt);
                    // F:2629
                    zdiss(jl) = tk_add(tk_add(zdiss(jl), tk_mul(puen(jl, jk), tk_sub(tk_add(tk_sub(zmfuu(jl, (jk + 1)), zmfuu(jl, jk)), zmfdu(jl, (jk + 1))), zmfdu(jl, jk)))), tk_mul(pven(jl, jk), tk_sub(tk_add(tk_sub(zmfuv(jl, (jk + 1)), zmfuv(jl, jk)), zmfdv(jl, (jk + 1))), zmfdv(jl, jk)))); TK_TRACE(2629, zdiss(jl));
                    // F:2634
                    pvom(jl, jk) = tk_add(pvom(jl, jk), zdudt); TK_TRACE(2634, pvom(jl, jk));
                    // F:2635
                    pvol(jl, jk) = tk_add(pvol(jl, jk), zdvdt); TK_TRACE(2635, pvol(jl, jk));
                // F:2636
                }
            // F:2637
            }
        // F:2638
        } else {
            // F:2639
            const int tk_limit_2639 = klon;
            for (jl = 1; jl <= tk_limit_2639; jl += 1) {
                // F:2640
                if (ldcum(jl)) {
                    // F:2641
                    zdudt = tk_mul((-tk_div(tk_g, tk_sub(paph(jl, (jk + 1)), paph(jl, jk)))), tk_add(zmfuu(jl, jk), zmfdu(jl, jk))); TK_TRACE(2641, zdudt);
                    // F:2643
                    zdvdt = tk_mul((-tk_div(tk_g, tk_sub(paph(jl, (jk + 1)), paph(jl, jk)))), tk_add(zmfuv(jl, jk), zmfdv(jl, jk))); TK_TRACE(2643, zdvdt);
                    // F:2645
                    zdiss(jl) = tk_sub(zdiss(jl), tk_add(tk_mul(puen(jl, jk), tk_add(zmfuu(jl, jk), zmfdu(jl, jk))), tk_mul(pven(jl, jk), tk_add(zmfuv(jl, jk), zmfdv(jl, jk))))); TK_TRACE(2645, zdiss(jl));
                    // F:2648
                    pvom(jl, jk) = tk_add(pvom(jl, jk), zdudt); TK_TRACE(2648, pvom(jl, jk));
                    // F:2649
                    pvol(jl, jk) = tk_add(pvol(jl, jk), zdvdt); TK_TRACE(2649, pvol(jl, jk));
                // F:2650
                }
            // F:2651
            }
        // F:2652
        }
    // F:2653
    }
    // F:2654
    zsum = tk_ssum(klon, zdiss.slice(1), 1, arena); TK_TRACE(2654, zsum);
    // F:2655
    psdiss = tk_add(psdiss, zsum); TK_TRACE(2655, psdiss);
    // F:2656
    return;
}

__device__ void tk_cubasmc(int klon, int klev, int klevm1, int kk, TkArray<float> pten, TkArray<float> pqen, TkArray<float> pqsen, TkArray<float> puen, TkArray<float> pven, TkArray<float> pverv, TkArray<float> pgeo, TkArray<float> pgeoh, TkArray<bool> ldcum, TkArray<int> ktype, TkArray<int> klab, TkArray<float> pmfu, TkArray<float> pmfub, TkArray<float> pentr, TkArray<int> kcbot, TkArray<float> ptu, TkArray<float> pqu, TkArray<float> plu, TkArray<float> puu, TkArray<float> pvu, TkArray<float> pmfus, TkArray<float> pmfuq, TkArray<float> pmful, TkArray<float> pdmfup, TkArray<float> pmfuu, TkArray<float> pmfuv, TkArena arena) {
    // module_cu_tiedtke.F:2668
    int klevp1;
    int jl;
    float zzzmb;
    // F:2715
    const int tk_limit_2715 = klon;
    for (jl = 1; jl <= tk_limit_2715; jl += 1) {
        // F:2716
        if ((((!ldcum(jl)) & (klab(jl, (kk + 1)) == 0.0f)) & (pqen(jl, kk) > tk_mul(0.80f, pqsen(jl, kk))))) {
            // F:2718
            ptu(jl, (kk + 1)) = tk_mul(tk_sub(tk_add(tk_mul(tk_cpd, pten(jl, kk)), pgeo(jl, kk)), pgeoh(jl, (kk + 1))), tk_rcpd); TK_TRACE(2718, ptu(jl, (kk + 1)));
            // F:2720
            pqu(jl, (kk + 1)) = pqen(jl, kk); TK_TRACE(2720, pqu(jl, (kk + 1)));
            // F:2721
            plu(jl, (kk + 1)) = 0.f; TK_TRACE(2721, plu(jl, (kk + 1)));
            // F:2722
            zzzmb = tk_max(tk_cmfcmin, tk_div((-pverv(jl, kk)), tk_g)); TK_TRACE(2722, zzzmb);
            // F:2723
            zzzmb = tk_min(zzzmb, tk_cmfcmax); TK_TRACE(2723, zzzmb);
            // F:2724
            pmfub(jl) = zzzmb; TK_TRACE(2724, pmfub(jl));
            // F:2725
            pmfu(jl, (kk + 1)) = pmfub(jl); TK_TRACE(2725, pmfu(jl, (kk + 1)));
            // F:2726
            pmfus(jl, (kk + 1)) = tk_mul(pmfub(jl), tk_add(tk_mul(tk_cpd, ptu(jl, (kk + 1))), pgeoh(jl, (kk + 1)))); TK_TRACE(2726, pmfus(jl, (kk + 1)));
            // F:2727
            pmfuq(jl, (kk + 1)) = tk_mul(pmfub(jl), pqu(jl, (kk + 1))); TK_TRACE(2727, pmfuq(jl, (kk + 1)));
            // F:2728
            pmful(jl, (kk + 1)) = 0.f; TK_TRACE(2728, pmful(jl, (kk + 1)));
            // F:2729
            pdmfup(jl, (kk + 1)) = 0.f; TK_TRACE(2729, pdmfup(jl, (kk + 1)));
            // F:2730
            kcbot(jl) = kk;
            // F:2731
            klab(jl, (kk + 1)) = 1;
            // F:2732
            ktype(jl) = 3;
            // F:2733
            pentr(jl) = tk_entrmid; TK_TRACE(2733, pentr(jl));
            // F:2734
            if (tk_lmfdudv) {
                // F:2735
                puu(jl, (kk + 1)) = puen(jl, kk); TK_TRACE(2735, puu(jl, (kk + 1)));
                // F:2736
                pvu(jl, (kk + 1)) = pven(jl, kk); TK_TRACE(2736, pvu(jl, (kk + 1)));
                // F:2737
                pmfuu(jl) = tk_mul(pmfub(jl), puu(jl, (kk + 1))); TK_TRACE(2737, pmfuu(jl));
                // F:2738
                pmfuv(jl) = tk_mul(pmfub(jl), pvu(jl, (kk + 1))); TK_TRACE(2738, pmfuv(jl));
            // F:2739
            }
        // F:2740
        }
    // F:2741
    }
    // F:2742
    return;
}

__device__ void tk_cuadjtq(int klon, int klev, int kk, TkArray<float> pp, TkArray<float> pt, TkArray<float> pq, TkArray<bool> ldflag, int kcall, TkArena arena) {
    // module_cu_tiedtke.F:2749
    int isum;
    int jl;
    float zqsat;
    float zcor;
    float zcond1;
    float tt;
    TkArray<float> zcond = arena.take<float>((klon), klon);
    TkArray<float> zqp = arena.take<float>((klon), klon);
    // F:2786
    if ((kcall == 1)) {
        // F:2787
        isum = 0;
        // F:2788
        const int tk_limit_2788 = klon;
        for (jl = 1; jl <= tk_limit_2788; jl += 1) {
            // F:2789
            zcond(jl) = 0.f; TK_TRACE(2789, zcond(jl));
            // F:2790
            if (ldflag(jl)) {
                // F:2791
                zqp(jl) = tk_div(1.f, pp(jl)); TK_TRACE(2791, zqp(jl));
                // F:2792
                tt = pt(jl, kk); TK_TRACE(2792, tt);
                // F:2793
                zqsat = tk_mul(tk_tlucua(tt, arena), zqp(jl)); TK_TRACE(2793, zqsat);
                // F:2794
                zqsat = tk_min(0.5f, zqsat); TK_TRACE(2794, zqsat);
                // F:2795
                zcor = tk_div(1.f, tk_sub(1.f, tk_mul(tk_vtmpc1, zqsat))); TK_TRACE(2795, zcor);
                // F:2796
                zqsat = tk_mul(zqsat, zcor); TK_TRACE(2796, zqsat);
                // F:2797
                zcond(jl) = tk_div(tk_sub(pq(jl, kk), zqsat), tk_add(1.f, tk_mul(tk_mul(zqsat, zcor), tk_tlucub(tt, arena)))); TK_TRACE(2797, zcond(jl));
                // F:2798
                zcond(jl) = tk_max(zcond(jl), 0.f); TK_TRACE(2798, zcond(jl));
                // F:2799
                pt(jl, kk) = tk_add(pt(jl, kk), tk_mul(tk_tlucuc(tt, arena), zcond(jl))); TK_TRACE(2799, pt(jl, kk));
                // F:2800
                pq(jl, kk) = tk_sub(pq(jl, kk), zcond(jl)); TK_TRACE(2800, pq(jl, kk));
                // F:2801
                if ((zcond(jl) != 0.0f)) { isum = (isum + 1); }
            // F:2802
            }
        // F:2803
        }
        // F:2805
        if ((isum == 0)) { return; }
        // F:2806
        const int tk_limit_2806 = klon;
        for (jl = 1; jl <= tk_limit_2806; jl += 1) {
            // F:2807
            if ((ldflag(jl) & (zcond(jl) != 0.f))) {
                // F:2808
                tt = pt(jl, kk); TK_TRACE(2808, tt);
                // F:2809
                zqsat = tk_mul(tk_tlucua(tt, arena), zqp(jl)); TK_TRACE(2809, zqsat);
                // F:2810
                zqsat = tk_min(0.5f, zqsat); TK_TRACE(2810, zqsat);
                // F:2811
                zcor = tk_div(1.f, tk_sub(1.f, tk_mul(tk_vtmpc1, zqsat))); TK_TRACE(2811, zcor);
                // F:2812
                zqsat = tk_mul(zqsat, zcor); TK_TRACE(2812, zqsat);
                // F:2813
                zcond1 = tk_div(tk_sub(pq(jl, kk), zqsat), tk_add(1.f, tk_mul(tk_mul(zqsat, zcor), tk_tlucub(tt, arena)))); TK_TRACE(2813, zcond1);
                // F:2814
                pt(jl, kk) = tk_add(pt(jl, kk), tk_mul(tk_tlucuc(tt, arena), zcond1)); TK_TRACE(2814, pt(jl, kk));
                // F:2815
                pq(jl, kk) = tk_sub(pq(jl, kk), zcond1); TK_TRACE(2815, pq(jl, kk));
            // F:2816
            }
        // F:2817
        }
    // F:2818
    }
    // F:2819
    if ((kcall == 2)) {
        // F:2820
        isum = 0;
        // F:2821
        const int tk_limit_2821 = klon;
        for (jl = 1; jl <= tk_limit_2821; jl += 1) {
            // F:2822
            zcond(jl) = 0.f; TK_TRACE(2822, zcond(jl));
            // F:2823
            if (ldflag(jl)) {
                // F:2824
                tt = pt(jl, kk); TK_TRACE(2824, tt);
                // F:2825
                zqp(jl) = tk_div(1.f, pp(jl)); TK_TRACE(2825, zqp(jl));
                // F:2826
                zqsat = tk_mul(tk_tlucua(tt, arena), zqp(jl)); TK_TRACE(2826, zqsat);
                // F:2827
                zqsat = tk_min(0.5f, zqsat); TK_TRACE(2827, zqsat);
                // F:2828
                zcor = tk_div(1.f, tk_sub(1.f, tk_mul(tk_vtmpc1, zqsat))); TK_TRACE(2828, zcor);
                // F:2829
                zqsat = tk_mul(zqsat, zcor); TK_TRACE(2829, zqsat);
                // F:2830
                zcond(jl) = tk_div(tk_sub(pq(jl, kk), zqsat), tk_add(1.f, tk_mul(tk_mul(zqsat, zcor), tk_tlucub(tt, arena)))); TK_TRACE(2830, zcond(jl));
                // F:2831
                zcond(jl) = tk_min(zcond(jl), 0.f); TK_TRACE(2831, zcond(jl));
                // F:2832
                pt(jl, kk) = tk_add(pt(jl, kk), tk_mul(tk_tlucuc(tt, arena), zcond(jl))); TK_TRACE(2832, pt(jl, kk));
                // F:2833
                pq(jl, kk) = tk_sub(pq(jl, kk), zcond(jl)); TK_TRACE(2833, pq(jl, kk));
                // F:2834
                if ((zcond(jl) != 0.0f)) { isum = (isum + 1); }
            // F:2835
            }
        // F:2836
        }
        // F:2838
        if ((isum == 0)) { return; }
        // F:2839
        const int tk_limit_2839 = klon;
        for (jl = 1; jl <= tk_limit_2839; jl += 1) {
            // F:2840
            if ((ldflag(jl) & (zcond(jl) != 0.f))) {
                // F:2841
                tt = pt(jl, kk); TK_TRACE(2841, tt);
                // F:2842
                zqsat = tk_mul(tk_tlucua(tt, arena), zqp(jl)); TK_TRACE(2842, zqsat);
                // F:2843
                zqsat = tk_min(0.5f, zqsat); TK_TRACE(2843, zqsat);
                // F:2844
                zcor = tk_div(1.f, tk_sub(1.f, tk_mul(tk_vtmpc1, zqsat))); TK_TRACE(2844, zcor);
                // F:2845
                zqsat = tk_mul(zqsat, zcor); TK_TRACE(2845, zqsat);
                // F:2846
                zcond1 = tk_div(tk_sub(pq(jl, kk), zqsat), tk_add(1.f, tk_mul(tk_mul(zqsat, zcor), tk_tlucub(tt, arena)))); TK_TRACE(2846, zcond1);
                // F:2847
                pt(jl, kk) = tk_add(pt(jl, kk), tk_mul(tk_tlucuc(tt, arena), zcond1)); TK_TRACE(2847, pt(jl, kk));
                // F:2848
                pq(jl, kk) = tk_sub(pq(jl, kk), zcond1); TK_TRACE(2848, pq(jl, kk));
            // F:2849
            }
        // F:2850
        }
    // F:2851
    }
    // F:2852
    if ((kcall == 0)) {
        // F:2853
        isum = 0;
        // F:2854
        const int tk_limit_2854 = klon;
        for (jl = 1; jl <= tk_limit_2854; jl += 1) {
            // F:2855
            tt = pt(jl, kk); TK_TRACE(2855, tt);
            // F:2856
            zqp(jl) = tk_div(1.f, pp(jl)); TK_TRACE(2856, zqp(jl));
            // F:2857
            zqsat = tk_mul(tk_tlucua(tt, arena), zqp(jl)); TK_TRACE(2857, zqsat);
            // F:2858
            zqsat = tk_min(0.5f, zqsat); TK_TRACE(2858, zqsat);
            // F:2859
            zcor = tk_div(1.f, tk_sub(1.f, tk_mul(tk_vtmpc1, zqsat))); TK_TRACE(2859, zcor);
            // F:2860
            zqsat = tk_mul(zqsat, zcor); TK_TRACE(2860, zqsat);
            // F:2861
            zcond(jl) = tk_div(tk_sub(pq(jl, kk), zqsat), tk_add(1.f, tk_mul(tk_mul(zqsat, zcor), tk_tlucub(tt, arena)))); TK_TRACE(2861, zcond(jl));
            // F:2862
            pt(jl, kk) = tk_add(pt(jl, kk), tk_mul(tk_tlucuc(tt, arena), zcond(jl))); TK_TRACE(2862, pt(jl, kk));
            // F:2863
            pq(jl, kk) = tk_sub(pq(jl, kk), zcond(jl)); TK_TRACE(2863, pq(jl, kk));
            // F:2864
            if ((zcond(jl) != 0.0f)) { isum = (isum + 1); }
        // F:2865
        }
        // F:2867
        if ((isum == 0)) { return; }
        // F:2868
        const int tk_limit_2868 = klon;
        for (jl = 1; jl <= tk_limit_2868; jl += 1) {
            // F:2869
            tt = pt(jl, kk); TK_TRACE(2869, tt);
            // F:2870
            zqsat = tk_mul(tk_tlucua(tt, arena), zqp(jl)); TK_TRACE(2870, zqsat);
            // F:2871
            zqsat = tk_min(0.5f, zqsat); TK_TRACE(2871, zqsat);
            // F:2872
            zcor = tk_div(1.f, tk_sub(1.f, tk_mul(tk_vtmpc1, zqsat))); TK_TRACE(2872, zcor);
            // F:2873
            zqsat = tk_mul(zqsat, zcor); TK_TRACE(2873, zqsat);
            // F:2874
            zcond1 = tk_div(tk_sub(pq(jl, kk), zqsat), tk_add(1.f, tk_mul(tk_mul(zqsat, zcor), tk_tlucub(tt, arena)))); TK_TRACE(2874, zcond1);
            // F:2875
            pt(jl, kk) = tk_add(pt(jl, kk), tk_mul(tk_tlucuc(tt, arena), zcond1)); TK_TRACE(2875, pt(jl, kk));
            // F:2876
            pq(jl, kk) = tk_sub(pq(jl, kk), zcond1); TK_TRACE(2876, pq(jl, kk));
        // F:2877
        }
    // F:2878
    }
    // F:2879
    if ((kcall == 4)) {
        // F:2880
        const int tk_limit_2880 = klon;
        for (jl = 1; jl <= tk_limit_2880; jl += 1) {
            // F:2881
            tt = pt(jl, kk); TK_TRACE(2881, tt);
            // F:2882
            zqp(jl) = tk_div(1.f, pp(jl)); TK_TRACE(2882, zqp(jl));
            // F:2883
            zqsat = tk_mul(tk_tlucua(tt, arena), zqp(jl)); TK_TRACE(2883, zqsat);
            // F:2884
            zqsat = tk_min(0.5f, zqsat); TK_TRACE(2884, zqsat);
            // F:2885
            zcor = tk_div(1.f, tk_sub(1.f, tk_mul(tk_vtmpc1, zqsat))); TK_TRACE(2885, zcor);
            // F:2886
            zqsat = tk_mul(zqsat, zcor); TK_TRACE(2886, zqsat);
            // F:2887
            zcond(jl) = tk_div(tk_sub(pq(jl, kk), zqsat), tk_add(1.f, tk_mul(tk_mul(zqsat, zcor), tk_tlucub(tt, arena)))); TK_TRACE(2887, zcond(jl));
            // F:2888
            pt(jl, kk) = tk_add(pt(jl, kk), tk_mul(tk_tlucuc(tt, arena), zcond(jl))); TK_TRACE(2888, pt(jl, kk));
            // F:2889
            pq(jl, kk) = tk_sub(pq(jl, kk), zcond(jl)); TK_TRACE(2889, pq(jl, kk));
        // F:2890
        }
        // F:2892
        const int tk_limit_2892 = klon;
        for (jl = 1; jl <= tk_limit_2892; jl += 1) {
            // F:2893
            tt = pt(jl, kk); TK_TRACE(2893, tt);
            // F:2894
            zqsat = tk_mul(tk_tlucua(tt, arena), zqp(jl)); TK_TRACE(2894, zqsat);
            // F:2895
            zqsat = tk_min(0.5f, zqsat); TK_TRACE(2895, zqsat);
            // F:2896
            zcor = tk_div(1.f, tk_sub(1.f, tk_mul(tk_vtmpc1, zqsat))); TK_TRACE(2896, zcor);
            // F:2897
            zqsat = tk_mul(zqsat, zcor); TK_TRACE(2897, zqsat);
            // F:2898
            zcond1 = tk_div(tk_sub(pq(jl, kk), zqsat), tk_add(1.f, tk_mul(tk_mul(zqsat, zcor), tk_tlucub(tt, arena)))); TK_TRACE(2898, zcond1);
            // F:2899
            pt(jl, kk) = tk_add(pt(jl, kk), tk_mul(tk_tlucuc(tt, arena), zcond1)); TK_TRACE(2899, pt(jl, kk));
            // F:2900
            pq(jl, kk) = tk_sub(pq(jl, kk), zcond1); TK_TRACE(2900, pq(jl, kk));
        // F:2901
        }
    // F:2902
    }
    // F:2903
    return;
}

__device__ void tk_cuentr_new(int klon, int klev, int klevp1, int kk, TkArray<float> ptenh, TkArray<float> paph, TkArray<float> pap, TkArray<float> pgeoh, TkArray<int> klwmin, TkArray<bool> ldcum, TkArray<int> ktype, TkArray<int> kcbot, TkArray<int> kctop0, TkArray<float> zpbase, TkArray<float> pmfu, TkArray<float> pentr, TkArray<float> zdmfen, TkArray<float> zdmfde, TkArray<float> zodetr, TkArray<int> khmin, TkArena arena) {
    // module_cu_tiedtke.F:2910
    int jl;
    int iklwmin;
    int ikb;
    int ikt;
    int ikh;
    float zrrho;
    float zdprho;
    float zpmid;
    float zentr;
    float zzmzk;
    float ztmzk;
    float arg;
    float zorgde;
    bool llo1;
    bool llo2;
    // F:2957
    const int tk_limit_2957 = klon;
    for (jl = 1; jl <= tk_limit_2957; jl += 1) {
        // F:2958
        zpbase(jl) = paph(jl, kcbot(jl)); TK_TRACE(2958, zpbase(jl));
        // F:2959
        zrrho = tk_div(tk_mul(tk_rd, ptenh(jl, (kk + 1))), paph(jl, (kk + 1))); TK_TRACE(2959, zrrho);
        // F:2960
        zdprho = tk_mul(tk_sub(paph(jl, (kk + 1)), paph(jl, kk)), tk_zrg); TK_TRACE(2960, zdprho);
        // F:2961
        zpmid = tk_mul(0.5f, tk_add(zpbase(jl), paph(jl, kctop0(jl)))); TK_TRACE(2961, zpmid);
        // F:2962
        zentr = tk_mul(tk_mul(tk_mul(pentr(jl), pmfu(jl, (kk + 1))), zdprho), zrrho); TK_TRACE(2962, zentr);
        // F:2963
        llo1 = ((kk < kcbot(jl)) & ldcum(jl));
        // F:2964
        if (llo1) {
            // F:2965
            zdmfde(jl) = zentr; TK_TRACE(2965, zdmfde(jl));
        // F:2966
        } else {
            // F:2967
            zdmfde(jl) = 0.0f; TK_TRACE(2967, zdmfde(jl));
        // F:2968
        }
        // F:2969
        llo2 = ((llo1 & (ktype(jl) == 2)) & ((tk_sub(zpbase(jl), paph(jl, kk)) < tk_zdnoprc) | (paph(jl, kk) > zpmid)));
        // F:2971
        if (llo2) {
            // F:2972
            zdmfen(jl) = zentr; TK_TRACE(2972, zdmfen(jl));
        // F:2973
        } else {
            // F:2974
            zdmfen(jl) = 0.0f; TK_TRACE(2974, zdmfen(jl));
        // F:2975
        }
        // F:2976
        iklwmin = tk_max(klwmin(jl), (kctop0(jl) + 2));
        // F:2977
        llo2 = ((llo1 & (ktype(jl) == 3)) & ((kk >= iklwmin) | (pap(jl, kk) > zpmid)));
        // F:2979
        if (llo2) { zdmfen(jl) = zentr; TK_TRACE(2979, zdmfen(jl)); }
        // F:2980
        llo2 = (llo1 & (ktype(jl) == 1));
        // F:2982
        if (llo2) { zdmfen(jl) = zentr; TK_TRACE(2982, zdmfen(jl)); }
        // F:2984
        ikb = kcbot(jl);
        // F:2985
        zodetr(jl, kk) = 0.f; TK_TRACE(2985, zodetr(jl, kk));
        // F:2986
        if (((llo2 & (kk <= khmin(jl))) & (kk >= kctop0(jl)))) {
            // F:2987
            ikt = kctop0(jl);
            // F:2988
            ikh = khmin(jl);
            // F:2989
            if ((ikh > ikt)) {
                // F:2990
                zzmzk = tk_mul((-tk_sub(pgeoh(jl, ikh), pgeoh(jl, kk))), tk_zrg); TK_TRACE(2990, zzmzk);
                // F:2991
                ztmzk = tk_mul((-tk_sub(pgeoh(jl, ikh), pgeoh(jl, ikt))), tk_zrg); TK_TRACE(2991, ztmzk);
                // F:2992
                arg = tk_mul(tk_mul(3.1415f, tk_div(zzmzk, ztmzk)), 0.5f); TK_TRACE(2992, arg);
                // F:2993
                zorgde = tk_div(tk_mul(tk_mul(glibc_tanf(arg), 3.1415f), 0.5f), ztmzk); TK_TRACE(2993, zorgde);
                // F:2994
                zdprho = tk_mul(tk_sub(paph(jl, (kk + 1)), paph(jl, kk)), tk_mul(tk_zrg, zrrho)); TK_TRACE(2994, zdprho);
                // F:2995
                zodetr(jl, kk) = tk_mul(tk_mul(tk_min(zorgde, 1.e-3f), pmfu(jl, (kk + 1))), zdprho); TK_TRACE(2995, zodetr(jl, kk));
            // F:2996
            }
        // F:2997
        }
    // F:2998
    }
    // F:3000
    return;
}

__device__ float tk_ssum(int n, TkArray<float> x, int ix, TkArena arena) {
    // module_cu_tiedtke.F:3007
    float result;
    float zsum;
    int jx;
    int jl;
    // F:3017
    jx = 1;
    // F:3018
    zsum = 0.0f; TK_TRACE(3018, zsum);
    // F:3019
    const int tk_limit_3019 = n;
    for (jl = 1; jl <= tk_limit_3019; jl += 1) {
        // F:3020
        zsum = tk_add(zsum, x(jx)); TK_TRACE(3020, zsum);
        // F:3021
        jx = (jx + ix);
    // F:3022
    }
    // F:3024
    result = zsum; TK_TRACE(3024, result);
    // F:3026
    return result;
}

__device__ float tk_tlucua(float tt, TkArena arena) {
    // module_cu_tiedtke.F:3029
    float result;
    float zcvm3;
    float zcvm4;
    // F:3036
    if ((tk_sub(tt, tk_tmelt) > 0.f)) {
        // F:3037
        zcvm3 = tk_c3les; TK_TRACE(3037, zcvm3);
        // F:3038
        zcvm4 = tk_c4les; TK_TRACE(3038, zcvm4);
    // F:3039
    } else {
        // F:3040
        zcvm3 = tk_c3ies; TK_TRACE(3040, zcvm3);
        // F:3041
        zcvm4 = tk_c4ies; TK_TRACE(3041, zcvm4);
    // F:3042
    }
    // F:3043
    result = tk_mul(tk_c2es, gfk_exp(tk_mul(tk_mul(zcvm3, tk_sub(tt, tk_tmelt)), tk_div(1.f, tk_sub(tt, zcvm4))))); TK_TRACE(3043, result);
    // F:3045
    return result;
}

__device__ float tk_tlucub(float tt, TkArena arena) {
    // module_cu_tiedtke.F:3048
    float result;
    float z5alvcp;
    float z5alscp;
    float zcvm4;
    float zcvm5;
    // F:3055
    z5alvcp = tk_div(tk_mul(tk_c5les, tk_alv), tk_cpd); TK_TRACE(3055, z5alvcp);
    // F:3056
    z5alscp = tk_div(tk_mul(tk_c5ies, tk_als), tk_cpd); TK_TRACE(3056, z5alscp);
    // F:3057
    if ((tk_sub(tt, tk_tmelt) > 0.f)) {
        // F:3058
        zcvm4 = tk_c4les; TK_TRACE(3058, zcvm4);
        // F:3059
        zcvm5 = z5alvcp; TK_TRACE(3059, zcvm5);
    // F:3060
    } else {
        // F:3061
        zcvm4 = tk_c4ies; TK_TRACE(3061, zcvm4);
        // F:3062
        zcvm5 = z5alscp; TK_TRACE(3062, zcvm5);
    // F:3063
    }
    // F:3064
    result = tk_mul(zcvm5, tk_mul(tk_div(1.f, tk_sub(tt, zcvm4)), tk_div(1.f, tk_sub(tt, zcvm4)))); TK_TRACE(3064, result);
    // F:3066
    return result;
}

__device__ float tk_tlucuc(float tt, TkArena arena) {
    // module_cu_tiedtke.F:3069
    float result;
    float zalvdcp;
    float zalsdcp;
    float zldcp;
    // F:3076
    zalvdcp = tk_div(tk_alv, tk_cpd); TK_TRACE(3076, zalvdcp);
    // F:3077
    zalsdcp = tk_div(tk_als, tk_cpd); TK_TRACE(3077, zalsdcp);
    // F:3078
    if ((tk_sub(tt, tk_tmelt) > 0.f)) {
        // F:3079
        zldcp = zalvdcp; TK_TRACE(3079, zldcp);
    // F:3080
    } else {
        // F:3081
        zldcp = zalsdcp; TK_TRACE(3081, zldcp);
    // F:3082
    }
    // F:3083
    result = zldcp; TK_TRACE(3083, result);
    // F:3085
    return result;
}

"""

DRIVER_SOURCE = r"""
// MPAS-A v8.4.2 cu_tiedtke. Each thread runs one column through
// tiecnv and all its original routines. This does not call the cu16 code.
// Common MPAS atmosphere preparation is supplied by the physics driver.
extern "C" __global__ void tiedtke_column(
    const float* u3d, const float* v3d, const float* w,
    const float* t3d, const float* qv3d, const float* qc3d,
    const float* qi3d, const float* pi3d, const float* rho3d,
    const float* qvften, const float* qvpblten, const float* dz8w,
    const float* pcps, const float* p8w, const float* xland,
    const float* qfx, const float* znu,
    float* rthcuten, float* rqvcuten, float* rqccuten, float* rqicuten,
    float* rucuten, float* rvcuten, float* raincv, float* pratec,
    int* cu_act_flag, float* workspace_f, int* workspace_i,
    bool* workspace_b, const float dt, const int itimestep,
    const int stepcu, const int nz, const int ncol,
    const int f_qc, const int f_qi, const int force_continue)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    // force_continue: 0 takes MPAS module_cu_tiedtke.F:1004's early return, 1 continues past it,
    // 2 reads the column's path from its activity word, which the adapter
    // wrote: 0 early return, 1 continue, any other value keeps the words
    // this column already holds and computes nothing.
    int* activity = workspace_i + (nz + 8) * ncol + c;
    bool continue_past = force_continue == 1;
    if (force_continue == 2) {
        const int path = *activity;
        if (path != 0 && path != 1) return;
        continue_past = path == 1;
    }
    TkArena arena = {workspace_f + c, workspace_i + c,
                     workspace_b + c, 0, 0, 0, ncol, cu_act_flag + c,
                     activity, continue_past};
    TkArray<float> zi = arena.take<float>(nz);
    TkArray<float> zl = arena.take<float>(nz);
    TkArray<float> u1 = arena.take<float>(nz);
    TkArray<float> v1 = arena.take<float>(nz);
    TkArray<float> t1 = arena.take<float>(nz);
    TkArray<float> q1 = arena.take<float>(nz);
    TkArray<float> q2 = arena.take<float>(nz);
    TkArray<float> q3 = arena.take<float>(nz);
    TkArray<float> q1b = arena.take<float>(nz);
    TkArray<float> q1bl = arena.take<float>(nz);
    TkArray<float> omg = arena.take<float>(nz);
    TkArray<float> ght = arena.take<float>(nz);
    TkArray<float> prsl = arena.take<float>(nz);
    TkArray<float> prsi = arena.take<float>(nz + 1);
    TkArray<float> sig1 = arena.take<float>(nz);
    TkArray<float> evap = arena.take<float>(1);
    TkArray<float> rn = arena.take<float>(1);
    TkArray<int> slimsk = arena.take<int>(1);
    TkArray<int> ktype = arena.take<int>(1);

    // module_cu_tiedtke.F:331-376. Surface height starts at zero.
    cu_act_flag[c] = 1;
    const float delt = tk_mul(dt, float(stepcu));
    const float rdelt = tk_div(1.0f, delt);
    zi(1) = 0.0f;
    for (int k = 2; k <= nz; ++k)
        zi(k) = tk_add(zi(k - 1), dz8w[(k - 2) * ncol + c]);
    for (int k = 2; k <= nz; ++k)
        zl(k - 1) = tk_mul(tk_add(zi(k), zi(k - 1)), 0.5f);
    zl(nz) = tk_sub(tk_mul(2.0f, zi(nz)), zl(nz - 1));
    slimsk(1) = int(tk_abs(tk_sub(xland[c], 2.0f)));

    // module_cu_tiedtke.F:378-422. MPAS reverses the vertical before tiecnv.
    for (int k = 1; k <= nz; ++k) {
        const int a = (k - 1) * ncol + c;
        const int zz = nz + 1 - k;
        u1(zz) = u3d[a];
        v1(zz) = v3d[a];
        t1(zz) = t3d[a];
        q1(zz) = qv3d[a];
        q1b(zz) = itimestep == 1 ? 0.0f : qvften[a];
        q1bl(zz) = itimestep == 1 ? 0.0f : qvpblten[a];
        q2(zz) = qc3d[a];
        q3(zz) = qi3d[a];
        omg(zz) = tk_mul(tk_mul(tk_mul(-0.5f, tk_g), rho3d[a]),
                         tk_add(w[a], w[a + ncol]));
        ght(zz) = zl(k);
        prsl(zz) = pcps[a];
        sig1(zz) = znu[a];
    }
    for (int k = 1; k <= nz + 1; ++k)
        prsi(nz + 2 - k) = p8w[(k - 1) * ncol + c];
    evap(1) = qfx[c];
    // module_cu_tiedtke.F:432-433. klon=1 removes no per-column arithmetic.
    tk_tiecnv(u1, v1, t1, q1, q2, q3, q1b, q1bl, ght, omg,
               prsl, prsi, evap, rn, slimsk, ktype, 1, nz, nz + 1,
               sig1, delt, arena);
    // module_cu_tiedtke.F:435-469. Both precipitation outputs are kept at the oracle boundary.
    raincv[c] = tk_div(rn(1), float(stepcu));
    pratec[c] = tk_div(rn(1), tk_mul(float(stepcu), dt));
    for (int k = 1; k <= nz; ++k) {
        const int a = (k - 1) * ncol + c;
        const int zz = nz + 1 - k;
        rthcuten[a] = tk_mul(tk_div(tk_sub(t1(zz), t3d[a]), pi3d[a]), rdelt);
        rqvcuten[a] = tk_mul(tk_sub(q1(zz), qv3d[a]), rdelt);
        rucuten[a] = tk_mul(tk_sub(u1(zz), u3d[a]), rdelt);
        rvcuten[a] = tk_mul(tk_sub(v1(zz), v3d[a]), rdelt);
        if (f_qc) rqccuten[a] = tk_mul(tk_sub(q2(zz), qc3d[a]), rdelt);
        if (f_qi) rqicuten[a] = tk_mul(tk_sub(q3(zz), qi3d[a]), rdelt);
    }
}

"""


@lru_cache(maxsize=1)
def cuda_source():
    """Shared glibc device maths; no private elementary-function implementation."""
    from importlib.resources import files
    headers = files("woof.core.kernels")
    return "\n".join(headers.joinpath(name).read_text() for name in
                     ("glibc_flt32.cuh", "glibc_trig_flt32.cuh")) + "\n" + \
        SUPPORT_SOURCE + ROUTINES_SOURCE + DRIVER_SOURCE


def compile_module(source):
    """Load CUDA sources through direct NVRTC, preserving binary32 subnormals.

    CuPy 14.2 RawModule appends -ftz=true after caller options. Direct
    compile_using_nvrtc avoids that override (pool NATIVE-DEFECT-POLICY.md).
    """
    import cupy as cp
    from cupy.cuda import compiler, function
    binary, _ = compiler.compile_using_nvrtc(
        source, options=_hex_numerics.options(COMPILE_OPTIONS), arch=_compilation_target(xp=cp),
        cache_in_memory=True)
    module = function.Module()
    module.load(binary)
    return module


@lru_cache(maxsize=None)
def _module(device):
    import cupy as cp
    with cp.cuda.Device(device):
        return compile_module(cuda_source())


def _kernel():
    import cupy as cp
    return _module(cp.cuda.Device().id).get_function("tiedtke_column")


def workspace_shapes(nz, ncol):
    """Call-tree arena bounds for module_cu_tiedtke.F:134-3001."""
    if nz < 4 or ncol < 1:
        raise ValueError("Tiedtke needs nz>=4 and ncol>=1: interior cloud-base levels are read")
    return ((58 * nz + 25, ncol), (nz + 9, ncol), (4, ncol))


def _device_array(a, shape, dtype, name):
    import cupy as cp
    if (not isinstance(a, cp.ndarray) or a.shape != shape or a.dtype != dtype
            or not a.flags.c_contiguous):
        raise ValueError(f"{name} requires contiguous device {dtype} {shape}: kernel offsets use this layout")


def column_step(inputs, *, dt, itimestep=0, stepcu=1, f_qc=True, f_qi=True,
                workspace=None, outputs=None, row_width=None):
    """cu_tiedtke; module_cu_tiedtke.F:134-475.

    Bottom-up float32 levels are (nz,ncol), interfaces (nz+1,ncol), and
    surfaces (ncol,). MPAS znu is (nz,ncol), never shared across cells.
    itimestep is MPAS initflag: 1 suppresses forcing, 0 enables it.
    Optional inactive tendencies retain their existing output words.
    row_width preserves each native horizontal tile's F:1004 predicate.
    """
    import cupy as cp
    if set(inputs) != set(INPUTS):
        raise ValueError(f"Tiedtke driver requires {INPUTS}")
    if inputs["t3d"].ndim != 2:
        raise ValueError("Tiedtke temperature requires (nz,ncol)")
    nz, ncol = map(int, inputs["t3d"].shape)
    shapes = workspace_shapes(nz, ncol)
    if (not math.isfinite(dt) or dt <= 0 or type(stepcu) is not int or stepcu < 1
            or type(itimestep) is not int or itimestep < 0):
        raise ValueError("positive finite dt/stepcu and nonnegative initflag required: delt divides tendencies")
    width = ncol if row_width is None else row_width
    if type(width) is not int or width < 1 or ncol % width:
        raise ValueError("row_width must divide ncol: native first-ascent return is per complete horizontal tile")
    for name in INPUTS:
        shape = ((nz + 1, ncol) if name in ("w", "p8w") else
                 (ncol,) if name in ("xland", "qfx") else (nz, ncol))
        _device_array(inputs[name], shape, np.float32, name)
    if workspace is None:
        workspace = tuple(cp.empty(s, dtype=d) for s, d in zip(
            shapes, (cp.float32, cp.int32, cp.bool_), strict=True))
    if len(workspace) != 3:
        raise ValueError("Tiedtke requires separate float, integer and logical workspaces")
    for a, shape, dtype in zip(workspace, shapes, (np.float32, np.int32, np.bool_), strict=True):
        _device_array(a, shape, dtype, "workspace")
    if outputs is None:
        outputs = {n: cp.zeros((nz, ncol), dtype=cp.float32) for n in LEVEL_OUTPUTS}
        outputs.update({n: cp.empty(ncol, dtype=cp.float32) for n in SURFACE_OUTPUTS})
        outputs["cu_act_flag"] = cp.empty(ncol, dtype=cp.int32)
    if set(outputs) != set(LEVEL_OUTPUTS + SURFACE_OUTPUTS + ("cu_act_flag",)):
        raise ValueError("Tiedtke outputs must include all six tendencies, both precipitation fields and activity")
    for name, a in outputs.items():
        _device_array(a, (nz, ncol) if name in LEVEL_OUTPUTS else (ncol,),
                      np.int32 if name == "cu_act_flag" else np.float32, name)
    args = tuple(inputs[n] for n in INPUTS) + tuple(outputs[n] for n in
        LEVEL_OUTPUTS + SURFACE_OUTPUTS + ("cu_act_flag",)) + tuple(workspace) + (
        np.float32(dt), np.int32(itimestep), np.int32(stepcu), np.int32(nz), np.int32(ncol),
        np.int32(f_qc), np.int32(f_qi), np.int32(0))
    launch = _kernel()
    grid, block = ((ncol + 31) // 32,), (32,)
    if width == 1:
        launch(grid, block, args)
    else:
        # A column's first ascent is independent; only its early return is
        # row-wide. Recompute inactive columns whose tile has an active one.
        activity = workspace[1][-1].reshape(-1, width)
        activity.fill(0)
        launch(grid, block, args[:-1] + (np.int32(2),))
        active = activity != 0
        continuation = active.any(axis=1, keepdims=True)
        activity[...] = cp.where(~active & continuation, cp.int32(1), cp.int32(-1))
        launch(grid, block, args[:-1] + (np.int32(2),))
    return outputs


def tiedtkeinit(outputs, *, restart, p_qc, p_qi, p_first_scalar):
    """tiedtkeinit; module_cu_tiedtke.F:478-540, complete active tile.

    mpas_atmphys_driver_convection.F:361-402 does not call this routine; expose the module's
    initializer for direct replay. The caller supplies only its active tile.
    """
    if not restart:
        for name in ("rthcuten", "rqvcuten", "rucuten", "rvcuten"):
            outputs[name].fill(0)
        if p_qc >= p_first_scalar:
            outputs["rqccuten"].fill(0)
        if p_qi >= p_first_scalar:
            outputs["rqicuten"].fill(0)


_PREPARED_FIELDS = {
    "u3d": "u_p", "v3d": "v_p", "w": "w_p", "t3d": "t_p",
    "qv3d": "qv_p", "qc3d": "qc_p", "qi3d": "qi_p", "pi3d": "pi_p",
    "rho3d": "rho_p", "dz8w": "dz_p", "pcps": "pres_hyd_p",
    "p8w": "pres2_hyd_p", "znu": "znu_hyd_p",
}


def convection_from_MPAS(pools):
    """mpas_atmphys_driver_convection.F:624-676,790-825.

    MPAS_to_physics owns common thermodynamic/hydrostatic preparation.
    Prepared fields and MPAS pools use contiguous (level,cell) device arrays.
    No transpose arithmetic is needed for the singleton j dimension.
    """
    import cupy as cp
    prepared, surface = pools["prepared"], pools["sfc_input"]
    tend, diag = pools["tend_physics"], pools["diag_physics"]
    inputs = {name: prepared[source] for name, source in _PREPARED_FIELDS.items()}
    inputs.update(xland=surface["xland"], qfx=diag["qfx"],
                  qvften=tend["rqvdynten"], qvpblten=tend["rqvblten"])
    outputs = {name: cp.ascontiguousarray(tend[name]).copy() for name in LEVEL_OUTPUTS}
    outputs.update(raincv=diag["raincv"].copy(), pratec=diag["cuprec"].copy(),
                   cu_act_flag=cp.empty(inputs["xland"].shape, dtype=cp.int32))
    return inputs, outputs


def convection_to_MPAS(outputs, pools):
    """mpas_atmphys_driver_convection.F:863-918,971-983."""
    for name in LEVEL_OUTPUTS:
        pools["tend_physics"][name][...] = outputs[name]
    pools["diag_physics"]["raincv"][...] = outputs["raincv"]
    pools["diag_physics"]["cuprec"][...] = outputs["pratec"]
    pools["cu_act_flag"][...] = outputs["cu_act_flag"]


def bind_native_pools(provider, call):
    """Bind f4 solve-cell views; mpas_atmphys_driver_convection.F:555-578.

    The provider owns native preparation, scalar words and halo preservation.
    Only the integer launch control is converted to Python int; field and
    floating scalar words stay in their native storage types.
    """
    scalars = provider["scalar_state"]
    levels = provider.session.layout.dimensions["nVertLevels"]
    prepared = {source: provider.prepared(source)[:levels + (name in ("w", "p8w"))]
                for name, source in _PREPARED_FIELDS.items()}
    return {name: provider[name] for name in
            ("sfc_input", "tend_physics", "diag_physics")} | {
                "prepared": prepared,
                "cu_act_flag": provider.prepared("cu_act_flag"),
                "dt_dyn": scalars["dt_dyn"], "n_cu": int(scalars["n_cu"]),
                "f_qc": scalars["f_qc"], "f_qi": scalars["f_qi"],
                "config_do_restart": provider["configs"]["config_do_restart"]}


def run_slot(call):
    """driver_convection; mpas_atmphys_driver_convection.F:405-478,555-578."""
    inputs, outputs = convection_from_MPAS(call.pools)
    p = call.pools
    outputs = column_step(inputs, dt=p["dt_dyn"],
                          itimestep=0 if p["config_do_restart"] or call.itimestep > 1 else 1,
                          stepcu=p["n_cu"], f_qc=p["f_qc"], f_qi=p["f_qi"],
                          outputs=outputs, row_width=p.get("row_width"))
    convection_to_MPAS(outputs, p)
    return outputs


def register(registry):
    """Register exactly the MPAS convection/selector/backend triple."""
    registry.register("convection", SCHEME, BACKEND, run_slot)
