"""MPAS-A v8.4.2 Kain-Fritsch CUDA column backend.

Authority is the complete MPAS module, not the WRF KF kernel. Each translated
routine cites its pinned source range; L comments preserve individual statement
locations. Fields use single precision, surface-to-top levels, cells fastest.
No host numerical implementation is used by this backend.

The packed CPS ABI includes one extra level in every vertical buffer; W reads
that upper interface and other arrays retain its unused padding. The driver
owns persistent state and calls step1 before convection and step2 afterwards.
"""
from __future__ import annotations

from woof.backend import compilation_target as _compilation_target
from woof.hex import numerics as _hex_numerics

from functools import lru_cache
from importlib.resources import files
import hashlib
import json
import os
from pathlib import Path
import numpy as np

SCHEME = "cu_kain_fritsch"
BACKEND = "hex-cuda"
SOURCE_SHA256 = "984f4f41c1a21207c19c8f0f4525072a94240da957be6724821d3c910341666d"
FIELDS = ("u", "v", "th", "t", "w", "dz8w", "pcps", "pi", "rho", "qv")
STATE = ("w0avg", "rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten", "rqscuten")
SURFACE = ("dxcell", "areacell", "raincv", "pratec", "nca", "cutop", "cubot")
PARAMETERS = ("dt", "cudt", "curr_secs", "xlv0", "xlv1", "xls0", "xls1", "cp", "r", "g", "ep1", "ep2", "svp1", "svp2", "svp3", "svpt0")
FLAGS = ("stepcu", "ktau", "adapt_step_flag", "warm_rain", "f_qr", "f_qi", "f_qs")
TABLE_WORDS = 110423
WORKSPACE_ARRAYS = 112
CUDA_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")

CUDA_SOURCE = r'''

#ifndef KFDEV
#define KFDEV __device__
#endif
const int kfnt = 250, kfnp = 220;
template<class T> KFDEV T kf_max(T a, T b) { return a > b ? a : b; }
template<class T> KFDEV T kf_min(T a, T b) { return a < b ? a : b; }
template<class T> struct FArray {
    T *p; int lo1, lo2, lo3, s1, s2, s3;
    KFDEV T &operator()(int i) const { return p[(i-lo1)*s1]; }
    KFDEV T &operator()(int i,int j) const { return p[(i-lo1)*s1+(j-lo2)*s2]; }
    KFDEV T &operator()(int i,int j,int k) const { return p[(i-lo1)*s1+(j-lo2)*s2+(k-lo3)*s3]; }
};
struct Arena {
    float *p; int used, stride;
    template<class T> KFDEV FArray<T> alloc(int lo, int hi) {
        T *v = reinterpret_cast<T*>(p+used*stride); used += hi-lo+1;
        return {v,lo,1,1,stride,0,0};
    }
};
struct ArenaScope { Arena &a; int mark; KFDEV ArenaScope(Arena &v,int m):a(v),mark(m) {} KFDEV ~ArenaScope() { a.used=mark; } };
struct Tables {
    FArray<float> ttab, qstab, the0k, alu;
    float &rdpr, &rdthk, &plutop;
    KFDEV Tables(float *p):
      ttab{p,1,1,1,1,250,0},qstab{p+55000,1,1,1,1,250,0},
      the0k{p+110000,1,1,1,1,0,0},alu{p+110220,1,1,1,1,0,0},
      rdpr(p[110420]),rdthk(p[110421]),plutop(p[110422]) {}
};

KFDEV void kf_eta_cps(Tables &tb, Arena &arena, int  ids, int  ide, int  jds, int  jde, int  kds, int  kde, int  ims, int  ime, int  jms, int  jme, int  kms, int  kme, int  its, int  ite, int  jts, int  jte, int  kts, int  kte, float  dt, int  ktau, FArray<float> dxcell, FArray<float> areacell, float  cudt, float  curr_secs, bool  adapt_step_flag, FArray<float> rho, FArray<float> raincv, FArray<float> pratec, FArray<float> nca, FArray<float> u, FArray<float> v, FArray<float> th, FArray<float> t, FArray<float> w, FArray<float> dz8w, FArray<float> pcps, FArray<float> pi, FArray<float> w0avg, float  xlv0, float  xlv1, float  xls0, float  xls1, float  cp, float  r, float  g, float  ep1, float  ep2, float  svp1, float  svp2, float  svp3, float  svpt0, int  stepcu, FArray<bool> cu_act_flag, bool  warm_rain, FArray<float> cutop, FArray<float> cubot, FArray<float> qv, bool & f_qr, bool & f_qi, bool & f_qs, FArray<float> rthcuten, FArray<float> rqvcuten, FArray<float> rqccuten, FArray<float> rqrcuten, FArray<float> rqicuten, FArray<float> rqscuten, bool has_f_qr, bool has_f_qi, bool has_f_qs, bool has_rthcuten, bool has_rqvcuten, bool has_rqccuten, bool has_rqrcuten, bool has_rqicuten, bool has_rqscuten);
KFDEV void kf_eta_para(Tables &tb, Arena &arena, int  i, int  j, FArray<float> u0, FArray<float> v0, FArray<float> t0, FArray<float> qv0, FArray<float> p0, FArray<float> dzq, FArray<float> w0avg1d, float  dt, float  dx, float  dxsq, FArray<float> rhoe, float  xlv0, float  xlv1, float  xls0, float  xls1, float  cp, float  r, float  g, float  ep2, float  svp1, float  svp2, float  svp3, float  svpt0, FArray<float> dqdt, FArray<float> dqidt, FArray<float> dqcdt, FArray<float> dqrdt, FArray<float> dqsdt, FArray<float> dtdt, FArray<float> raincv, FArray<float> pratec, FArray<float> nca, bool  f_qi, bool  f_qs, bool  warm_rain, FArray<float> cutop, FArray<float> cubot, float  cudt, int  ids, int  ide, int  jds, int  jde, int  kds, int  kde, int  ims, int  ime, int  jms, int  jme, int  kms, int  kme, int  its, int  ite, int  jts, int  jte, int  kts, int  kte);
KFDEV void tpmix2(Tables &tb, Arena &arena, float  p, float  thes, float & tu, float & qu, float & qliq, float & qice, float & qnewlq, float & qnewic, float  xlv1, float  xlv0);
KFDEV void dtfrznew(Tables &tb, Arena &arena, float & tu, float  p, float & thteu, float & qu, float  qfrz, float & qice, float  aliq, float  bliq, float  cliq, float  dliq);
KFDEV void condload(Tables &tb, Arena &arena, float & qliq, float & qice, float & wtw, float  dz, float  boterm, float  enterm, float  rate, float & qnewlq, float & qnewic, float & qlqout, float & qicout, float  g);
KFDEV void prof5(Tables &tb, Arena &arena, float  eq, float & ee, float & ud);
KFDEV void tpmix2dd(Tables &tb, Arena &arena, float  p, float  thes, float & ts, float & qs, int  i, int  j);
KFDEV void envirtht(Tables &tb, Arena &arena, float  p1, float  t1, float  q1, float & tht1, float  aliq, float  bliq, float  cliq, float  dliq);
KFDEV void kf_eta_init(Tables &tb, Arena &arena, FArray<float> rthcuten, FArray<float> rqvcuten, FArray<float> rqccuten, FArray<float> rqrcuten, FArray<float> rqicuten, FArray<float> rqscuten, FArray<float> nca, FArray<float> w0avg, int  p_qi, int  p_qs, float  svp1, float  svp2, float  svp3, float  svpt0, int  p_first_scalar, bool  restart, bool  allowed_to_read, int  ids, int  ide, int  jds, int  jde, int  kds, int  kde, int  ims, int  ime, int  jms, int  jme, int  kms, int  kme, int  its, int  ite, int  jts, int  jte, int  kts, int  kte);
KFDEV void kf_lutab(Tables &tb, Arena &arena, float  svp1, float  svp2, float  svp3, float  svpt0);

// MPAS-A v8.4.2 module_cu_kfeta.F:2373-2493
KFDEV void tpmix2(Tables &tb, Arena &arena, float  p, float  thes, float & tu, float & qu, float & qliq, float & qice, float & qnewlq, float & qnewic, float  xlv1, float  xlv0) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float tp;
    float qq;
    float bth;
    float tth;
    float pp;
    float t00;
    float t10;
    float t01;
    float t11;
    float q00;
    float q10;
    float q01;
    float q11;
    float temp;
    float qs;
    float qnew;
    float dq;
    float qtot;
    float rll;
    float cpp;
    int iptb;
    int ithtb;
    tp = __fmul_rn(__fsub_rn(p, tb.plutop), tb.rdpr); // L2404
    qq = __fsub_rn(tp, truncf(tp)); // L2405
    iptb = (int(tp) + 1); // L2406
    bth = __fadd_rn(__fmul_rn(__fsub_rn(tb.the0k((iptb + 1)), tb.the0k(iptb)), qq), tb.the0k(iptb)); // L2414
    tth = __fmul_rn(__fsub_rn(thes, bth), tb.rdthk); // L2415
    pp = __fsub_rn(tth, truncf(tth)); // L2416
    ithtb = (int(tth) + 1); // L2417
    if (((((iptb >= 220) || (iptb <= 1)) || (ithtb >= 250)) || (ithtb <= 1))) { // L2418
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2419
    } // L2421
    t00 = tb.ttab(ithtb, iptb); // L2423
    t10 = tb.ttab((ithtb + 1), iptb); // L2424
    t01 = tb.ttab(ithtb, (iptb + 1)); // L2425
    t11 = tb.ttab((ithtb + 1), (iptb + 1)); // L2426
    q00 = tb.qstab(ithtb, iptb); // L2428
    q10 = tb.qstab((ithtb + 1), iptb); // L2429
    q01 = tb.qstab(ithtb, (iptb + 1)); // L2430
    q11 = tb.qstab((ithtb + 1), (iptb + 1)); // L2431
    temp = __fadd_rn(__fadd_rn(__fadd_rn(t00, __fmul_rn(__fsub_rn(t10, t00), pp)), __fmul_rn(__fsub_rn(t01, t00), qq)), __fmul_rn(__fmul_rn(__fadd_rn(__fsub_rn(__fsub_rn(t00, t10), t01), t11), pp), qq)); // L2437
    qs = __fadd_rn(__fadd_rn(__fadd_rn(q00, __fmul_rn(__fsub_rn(q10, q00), pp)), __fmul_rn(__fsub_rn(q01, q00), qq)), __fmul_rn(__fmul_rn(__fadd_rn(__fsub_rn(__fsub_rn(q00, q10), q01), q11), pp), qq)); // L2439
    dq = __fsub_rn(qs, qu); // L2441
    if ((dq <= 0.f)) { // L2442
    qnew = __fsub_rn(qu, qs); // L2443
    qu = qs; // L2444
    } else { // L2445
    qnew = 0.f; // L2450
    qtot = __fadd_rn(qliq, qice); // L2451
    if ((qtot >= dq)) { // L2465
    qliq = __fsub_rn(qliq, __fdiv_rn(__fmul_rn(dq, qliq), __fadd_rn(qtot, 1.e-10f))); // L2466
    qice = __fsub_rn(qice, __fdiv_rn(__fmul_rn(dq, qice), __fadd_rn(qtot, 1.e-10f))); // L2467
    qu = qs; // L2468
    } else { // L2469
    rll = __fsub_rn(xlv0, __fmul_rn(xlv1, temp)); // L2470
    cpp = __fmul_rn(1004.5f, __fadd_rn(1.f, __fmul_rn(0.89f, qu))); // L2471
    if ((qtot < 1.e-10f)) { // L2472
    temp = __fadd_rn(temp, __fdiv_rn(__fmul_rn(rll, __fdiv_rn(dq, __fadd_rn(1.f, dq))), cpp)); // L2475
    } else { // L2476
    temp = __fadd_rn(temp, __fdiv_rn(__fmul_rn(rll, __fdiv_rn(__fsub_rn(dq, qtot), __fsub_rn(__fadd_rn(1, dq), qtot))), cpp)); // L2481
    qu = __fadd_rn(qu, qtot); // L2482
    qtot = 0.f; // L2483
    qliq = 0.f; // L2484
    qice = 0.f; // L2485
    } // L2486
    } // L2487
    } // L2488
    tu = temp; // L2489
    qnewlq = qnew; // L2490
    qnewic = 0.f; // L2491
}

// MPAS-A v8.4.2 module_cu_kfeta.F:2495-2538
KFDEV void dtfrznew(Tables &tb, Arena &arena, float & tu, float  p, float & thteu, float & qu, float  qfrz, float & qice, float  aliq, float  bliq, float  cliq, float  dliq) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float rlc;
    float rls;
    float rlf;
    float cpp;
    float a;
    float dtfrz;
    float es;
    float qs;
    float dqevap;
    float pii;
    rlc = __fsub_rn(2.5e6f, __fmul_rn(2369.276f, __fsub_rn(tu, 273.16f))); // L2511
    rls = __fsub_rn(2833922.f, __fmul_rn(259.532f, __fsub_rn(tu, 273.16f))); // L2512
    rlf = __fsub_rn(rls, rlc); // L2513
    cpp = __fmul_rn(1004.5f, __fadd_rn(1.f, __fmul_rn(0.89f, qu))); // L2514
    a = __fdiv_rn(__fsub_rn(cliq, __fmul_rn(bliq, dliq)), __fmul_rn(__fsub_rn(tu, dliq), __fsub_rn(tu, dliq))); // L2519
    dtfrz = __fdiv_rn(__fmul_rn(rlf, qfrz), __fadd_rn(cpp, __fmul_rn(__fmul_rn(rls, qu), a))); // L2520
    tu = __fadd_rn(tu, dtfrz); // L2521
    es = __fmul_rn(aliq, gfk_exp_fma(__fdiv_rn(__fsub_rn(__fmul_rn(bliq, tu), cliq), __fsub_rn(tu, dliq)))); // L2523
    qs = __fdiv_rn(__fmul_rn(es, 0.622f), __fsub_rn(p, es)); // L2524
    dqevap = __fsub_rn(qs, qu); // L2532
    qice = __fsub_rn(qice, dqevap); // L2533
    qu = __fadd_rn(qu, dqevap); // L2534
    pii = gfk_pow(__fdiv_rn(1.e5f, p), __fmul_rn(0.2854f, __fsub_rn(1.f, __fmul_rn(0.28f, qu)))); // L2535
    thteu = __fmul_rn(__fmul_rn(tu, pii), gfk_exp_fma(__fmul_rn(__fmul_rn(__fsub_rn(__fdiv_rn(3374.6525f, tu), 2.5403f), qu), __fadd_rn(1.f, __fmul_rn(0.81f, qu))))); // L2536
}

// MPAS-A v8.4.2 module_cu_kfeta.F:2541-2611
KFDEV void condload(Tables &tb, Arena &arena, float & qliq, float & qice, float & wtw, float  dz, float  boterm, float  enterm, float  rate, float & qnewlq, float & qnewic, float & qlqout, float & qicout, float  g) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float qtot;
    float qnew;
    float qest;
    float g1;
    float wavg;
    float conv;
    float ratio3;
    float oldq;
    float ratio4;
    float dq;
    float pptdrg;
    qtot = __fadd_rn(qliq, qice); // L2564
    qnew = __fadd_rn(qnewlq, qnewic); // L2565
    qest = __fmul_rn(0.5f, __fadd_rn(qtot, qnew)); // L2571
    g1 = __fsub_rn(__fsub_rn(__fadd_rn(wtw, boterm), enterm), __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(2.f, g), dz), qest), 1.5f)); // L2572
    if ((g1 < 0.0f)) { g1 = 0.f; } // L2573
    wavg = __fmul_rn(0.5f, __fadd_rn(__fsqrt_rn(wtw), __fsqrt_rn(g1))); // L2574
    conv = __fdiv_rn(__fmul_rn(rate, dz), wavg); // L2575
    ratio3 = __fdiv_rn(qnewlq, __fadd_rn(qnew, 1.e-8f)); // L2582
    qtot = __fadd_rn(qtot, __fmul_rn(0.6f, qnew)); // L2584
    oldq = qtot; // L2585
    ratio4 = __fdiv_rn(__fadd_rn(__fmul_rn(0.6f, qnewlq), qliq), __fadd_rn(qtot, 1.e-8f)); // L2586
    qtot = __fmul_rn(qtot, gfk_exp_fma((-conv))); // L2587
    dq = __fsub_rn(oldq, qtot); // L2592
    qlqout = __fmul_rn(ratio4, dq); // L2593
    qicout = __fmul_rn(__fsub_rn(1.f, ratio4), dq); // L2594
    pptdrg = __fmul_rn(0.5f, __fsub_rn(__fadd_rn(oldq, qtot), __fmul_rn(0.2f, qnew))); // L2599
    wtw = __fsub_rn(__fsub_rn(__fadd_rn(wtw, boterm), enterm), __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(2.f, g), dz), pptdrg), 1.5f)); // L2600
    if ((fabsf(wtw) < 1.e-4f)) { wtw = 1.e-4f; } // L2601
    qliq = __fadd_rn(__fmul_rn(ratio4, qtot), __fmul_rn(__fmul_rn(ratio3, 0.4f), qnew)); // L2606
    qice = __fadd_rn(__fmul_rn(__fsub_rn(1.f, ratio4), qtot), __fmul_rn(__fmul_rn(__fsub_rn(1.f, ratio3), 0.4f), qnew)); // L2607
    qnewlq = 0.f; // L2608
    qnewic = 0.f; // L2609
}

// MPAS-A v8.4.2 module_cu_kfeta.F:2614-2656
KFDEV void prof5(Tables &tb, Arena &arena, float  eq, float & ee, float & ud) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float sqrt2p = 2.506628f;
    float a1 = 0.4361836f;
    float a2 = (-0.1201676f);
    float a3 = 0.9372980f;
    float p = 0.33267f;
    float sigma = 0.166666667f;
    float fe = 0.202765151f;
    float x;
    float y;
    float ey;
    float e45;
    float t1;
    float t2;
    float c1;
    float c2;
    x = __fdiv_rn(__fsub_rn(eq, 0.5f), sigma); // L2636
    y = __fsub_rn(__fmul_rn(6.f, eq), 3.f); // L2637
    ey = gfk_exp_fma(__fdiv_rn(__fmul_rn(y, y), (-2))); // L2638
    e45 = gfk_exp_fma((-4.5f)); // L2639
    t2 = __fdiv_rn(1.f, __fadd_rn(1.f, __fmul_rn(p, fabsf(y)))); // L2640
    t1 = 0.500498f; // L2641
    c1 = __fadd_rn(__fadd_rn(__fmul_rn(a1, t1), __fmul_rn(__fmul_rn(a2, t1), t1)), __fmul_rn(__fmul_rn(__fmul_rn(a3, t1), t1), t1)); // L2642
    c2 = __fadd_rn(__fadd_rn(__fmul_rn(a1, t2), __fmul_rn(__fmul_rn(a2, t2), t2)), __fmul_rn(__fmul_rn(__fmul_rn(a3, t2), t2), t2)); // L2643
    if ((y >= 0.f)) { // L2644
    ee = __fsub_rn(__fmul_rn(sigma, __fadd_rn(__fmul_rn(0.5f, __fsub_rn(__fsub_rn(sqrt2p, __fmul_rn(e45, c1)), __fmul_rn(ey, c2))), __fmul_rn(sigma, __fsub_rn(e45, ey)))), __fdiv_rn(__fmul_rn(__fmul_rn(e45, eq), eq), 2.f)); // L2645
    ud = __fsub_rn(__fmul_rn(sigma, __fadd_rn(__fmul_rn(0.5f, __fsub_rn(__fmul_rn(ey, c2), __fmul_rn(e45, c1))), __fmul_rn(sigma, __fsub_rn(e45, ey)))), __fmul_rn(e45, __fsub_rn(__fadd_rn(0.5f, __fdiv_rn(__fmul_rn(eq, eq), 2.f)), eq))); // L2646
    } else { // L2648
    ee = __fsub_rn(__fmul_rn(sigma, __fadd_rn(__fmul_rn(0.5f, __fsub_rn(__fmul_rn(ey, c2), __fmul_rn(e45, c1))), __fmul_rn(sigma, __fsub_rn(e45, ey)))), __fdiv_rn(__fmul_rn(__fmul_rn(e45, eq), eq), 2.f)); // L2649
    ud = __fsub_rn(__fmul_rn(sigma, __fadd_rn(__fmul_rn(0.5f, __fsub_rn(__fsub_rn(sqrt2p, __fmul_rn(e45, c1)), __fmul_rn(ey, c2))), __fmul_rn(sigma, __fsub_rn(e45, ey)))), __fmul_rn(e45, __fsub_rn(__fadd_rn(0.5f, __fdiv_rn(__fmul_rn(eq, eq), 2.f)), eq))); // L2650
    } // L2652
    ee = __fdiv_rn(ee, fe); // L2653
    ud = __fdiv_rn(ud, fe); // L2654
}

// MPAS-A v8.4.2 module_cu_kfeta.F:2659-2723
KFDEV void tpmix2dd(Tables &tb, Arena &arena, float  p, float  thes, float & ts, float & qs, int  i, int  j) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float tp;
    float qq;
    float bth;
    float tth;
    float pp;
    float t00;
    float t10;
    float t01;
    float t11;
    float q00;
    float q10;
    float q01;
    float q11;
    int iptb;
    int ithtb;
    tp = __fmul_rn(__fsub_rn(p, tb.plutop), tb.rdpr); // L2691
    qq = __fsub_rn(tp, truncf(tp)); // L2692
    iptb = (int(tp) + 1); // L2693
    bth = __fadd_rn(__fmul_rn(__fsub_rn(tb.the0k((iptb + 1)), tb.the0k(iptb)), qq), tb.the0k(iptb)); // L2700
    tth = __fmul_rn(__fsub_rn(thes, bth), tb.rdthk); // L2701
    pp = __fsub_rn(tth, truncf(tth)); // L2702
    ithtb = (int(tth) + 1); // L2703
    t00 = tb.ttab(ithtb, iptb); // L2705
    t10 = tb.ttab((ithtb + 1), iptb); // L2706
    t01 = tb.ttab(ithtb, (iptb + 1)); // L2707
    t11 = tb.ttab((ithtb + 1), (iptb + 1)); // L2708
    q00 = tb.qstab(ithtb, iptb); // L2710
    q10 = tb.qstab((ithtb + 1), iptb); // L2711
    q01 = tb.qstab(ithtb, (iptb + 1)); // L2712
    q11 = tb.qstab((ithtb + 1), (iptb + 1)); // L2713
    ts = __fadd_rn(__fadd_rn(__fadd_rn(t00, __fmul_rn(__fsub_rn(t10, t00), pp)), __fmul_rn(__fsub_rn(t01, t00), qq)), __fmul_rn(__fmul_rn(__fadd_rn(__fsub_rn(__fsub_rn(t00, t10), t01), t11), pp), qq)); // L2719
    qs = __fadd_rn(__fadd_rn(__fadd_rn(q00, __fmul_rn(__fsub_rn(q10, q00), pp)), __fmul_rn(__fsub_rn(q01, q00), qq)), __fmul_rn(__fmul_rn(__fadd_rn(__fsub_rn(__fsub_rn(q00, q10), q01), q11), pp), qq)); // L2721
}

// MPAS-A v8.4.2 module_cu_kfeta.F:2726-2762
KFDEV void envirtht(Tables &tb, Arena &arena, float  p1, float  t1, float  q1, float & tht1, float  aliq, float  bliq, float  cliq, float  dliq) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float ee;
    float tlog;
    float astrt;
    float ainc;
    float a1;
    float tp;
    float value;
    float aintrp;
    float tdpt;
    float tsat;
    float tht;
    float t00 = 273.16f;
    float p00 = 1.e5f;
    float c1 = 3374.6525f;
    float c2 = 2.5403f;
    float c3 = 3114.834f;
    float c4 = 0.278296f;
    float c5 = 1.0723e-3f;
    int indlu;
    ee = __fdiv_rn(__fmul_rn(q1, p1), __fadd_rn(0.622f, q1)); // L2744
    astrt = 1.e-3f; // L2748
    ainc = 0.075f; // L2749
    a1 = __fdiv_rn(ee, aliq); // L2750
    tp = __fdiv_rn(__fsub_rn(a1, astrt), ainc); // L2751
    indlu = (int(tp) + 1); // L2752
    value = __fadd_rn(__fmul_rn((indlu - 1), ainc), astrt); // L2753
    aintrp = __fdiv_rn(__fsub_rn(a1, value), ainc); // L2754
    tlog = __fadd_rn(__fmul_rn(aintrp, tb.alu((indlu + 1))), __fmul_rn(__fsub_rn(1, aintrp), tb.alu(indlu))); // L2755
    tdpt = __fdiv_rn(__fsub_rn(cliq, __fmul_rn(dliq, tlog)), __fsub_rn(bliq, tlog)); // L2757
    tsat = __fsub_rn(tdpt, __fmul_rn(__fsub_rn(__fadd_rn(.212f, __fmul_rn(1.571e-3f, __fsub_rn(tdpt, t00))), __fmul_rn(4.36e-4f, __fsub_rn(t1, t00))), __fsub_rn(t1, tdpt))); // L2758
    tht = __fmul_rn(t1, gfk_pow(__fdiv_rn(p00, p1), __fmul_rn(0.2854f, __fsub_rn(1.f, __fmul_rn(0.28f, q1))))); // L2759
    tht1 = __fmul_rn(tht, gfk_exp_fma(__fmul_rn(__fmul_rn(__fsub_rn(__fdiv_rn(c1, tsat), c2), q1), __fadd_rn(1.f, __fmul_rn(0.81f, q1))))); // L2760
}

// MPAS-A v8.4.2 module_cu_kfeta.F:2855-2982
KFDEV void kf_lutab(Tables &tb, Arena &arena, float  svp1, float  svp2, float  svp3, float  svpt0) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    int kp;
    int it;
    int itcnt;
    int i;
    float dth = 1.f;
    float tmin = 150.f;
    float toler = 0.001f;
    float pbot;
    float dpr;
    float temp;
    float p;
    float es;
    float qs;
    float pi;
    float thes;
    float tgues;
    float thgues;
    float f0;
    float t1;
    float t0;
    float thgs;
    float f1;
    float dt;
    float astrt;
    float ainc;
    float a1;
    float thtgs;
    float aliq;
    float bliq;
    float cliq;
    float dliq;
    tb.plutop = 5000.0f; // L2887
    pbot = 110000.0f; // L2889
    aliq = __fmul_rn(svp1, 1000.f); // L2891
    bliq = svp2; // L2892
    cliq = __fmul_rn(svp2, svpt0); // L2893
    dliq = svp3; // L2894
    tb.rdthk = __fdiv_rn(1.f, dth); // L2900
    dpr = __fdiv_rn(__fsub_rn(pbot, tb.plutop), float((kfnp - 1))); // L2903
    tb.rdpr = __fdiv_rn(1.f, dpr); // L2906
    temp = tmin; // L2912
    p = __fsub_rn(tb.plutop, dpr); // L2913
    for (kp = 1; kp <= kfnp; kp += 1) { loop1_cycle: ; // L2914
    p = __fadd_rn(p, dpr); // L2915
    es = __fmul_rn(aliq, gfk_exp_fma(__fdiv_rn(__fsub_rn(__fmul_rn(bliq, temp), cliq), __fsub_rn(temp, dliq)))); // L2916
    qs = __fdiv_rn(__fmul_rn(0.622f, es), __fsub_rn(p, es)); // L2917
    pi = gfk_pow(__fdiv_rn(1.e5f, p), __fmul_rn(0.2854f, __fsub_rn(1.f, __fmul_rn(0.28f, qs)))); // L2918
    tb.the0k(kp) = __fmul_rn(__fmul_rn(temp, pi), gfk_exp_fma(__fmul_rn(__fmul_rn(__fsub_rn(__fdiv_rn(3374.6525f, temp), 2.5403f), qs), __fadd_rn(1.f, __fmul_rn(0.81f, qs))))); // L2919
    loop1_continue: ;
    } loop1_exit: ; // L2921
    p = __fsub_rn(tb.plutop, dpr); // L2925
    for (kp = 1; kp <= kfnp; kp += 1) { loop2_cycle: ; // L2926
    thes = __fsub_rn(tb.the0k(kp), dth); // L2927
    p = __fadd_rn(p, dpr); // L2928
    for (it = 1; it <= kfnt; it += 1) { loop3_cycle: ; // L2929
    thes = __fadd_rn(thes, dth); // L2931
    if ((it == 1)) { // L2934
    tgues = tmin; // L2935
    } else { // L2936
    tgues = tb.ttab((it - 1), kp); // L2937
    } // L2938
    es = __fmul_rn(aliq, gfk_exp_fma(__fdiv_rn(__fsub_rn(__fmul_rn(bliq, tgues), cliq), __fsub_rn(tgues, dliq)))); // L2939
    qs = __fdiv_rn(__fmul_rn(0.622f, es), __fsub_rn(p, es)); // L2940
    pi = gfk_pow(__fdiv_rn(1.e5f, p), __fmul_rn(0.2854f, __fsub_rn(1.f, __fmul_rn(0.28f, qs)))); // L2941
    thgues = __fmul_rn(__fmul_rn(tgues, pi), gfk_exp_fma(__fmul_rn(__fmul_rn(__fsub_rn(__fdiv_rn(3374.6525f, tgues), 2.5403f), qs), __fadd_rn(1.f, __fmul_rn(0.81f, qs))))); // L2942
    f0 = __fsub_rn(thgues, thes); // L2944
    t1 = __fsub_rn(tgues, __fmul_rn(0.5f, f0)); // L2945
    t0 = tgues; // L2946
    itcnt = 0; // L2947
    for (itcnt = 1; itcnt <= 11; itcnt += 1) { loop4_cycle: ; // L2949
    es = __fmul_rn(aliq, gfk_exp_fma(__fdiv_rn(__fsub_rn(__fmul_rn(bliq, t1), cliq), __fsub_rn(t1, dliq)))); // L2950
    qs = __fdiv_rn(__fmul_rn(0.622f, es), __fsub_rn(p, es)); // L2951
    pi = gfk_pow(__fdiv_rn(1.e5f, p), __fmul_rn(0.2854f, __fsub_rn(1.f, __fmul_rn(0.28f, qs)))); // L2952
    thtgs = __fmul_rn(__fmul_rn(t1, pi), gfk_exp_fma(__fmul_rn(__fmul_rn(__fsub_rn(__fdiv_rn(3374.6525f, t1), 2.5403f), qs), __fadd_rn(1.f, __fmul_rn(0.81f, qs))))); // L2953
    f1 = __fsub_rn(thtgs, thes); // L2954
    if ((fabsf(f1) < toler)) { // L2955
    goto loop4_exit; // L2956
    } // L2957
    dt = __fdiv_rn(__fmul_rn(f1, __fsub_rn(t1, t0)), __fsub_rn(f1, f0)); // L2959
    t0 = t1; // L2960
    f0 = f1; // L2961
    t1 = __fsub_rn(t1, dt); // L2962
    loop4_continue: ;
    } loop4_exit: ; // L2963
    tb.ttab(it, kp) = t1; // L2964
    tb.qstab(it, kp) = qs; // L2965
    loop3_continue: ;
    } loop3_exit: ; // L2966
    loop2_continue: ;
    } loop2_exit: ; // L2967
    astrt = 1.e-3f; // L2973
    ainc = 0.075f; // L2974
    a1 = __fsub_rn(astrt, ainc); // L2976
    for (i = 1; i <= 200; i += 1) { loop5_cycle: ; // L2977
    a1 = __fadd_rn(a1, ainc); // L2978
    tb.alu(i) = gfk_log(a1); // L2979
    loop5_continue: ;
    } loop5_exit: ; // L2980
}

// MPAS-A v8.4.2 module_cu_kfeta.F:364-2370
KFDEV void kf_eta_para(Tables &tb, Arena &arena, int  i, int  j, FArray<float> u0, FArray<float> v0, FArray<float> t0, FArray<float> qv0, FArray<float> p0, FArray<float> dzq, FArray<float> w0avg1d, float  dt, float  dx, float  dxsq, FArray<float> rhoe, float  xlv0, float  xlv1, float  xls0, float  xls1, float  cp, float  r, float  g, float  ep2, float  svp1, float  svp2, float  svp3, float  svpt0, FArray<float> dqdt, FArray<float> dqidt, FArray<float> dqcdt, FArray<float> dqrdt, FArray<float> dqsdt, FArray<float> dtdt, FArray<float> raincv, FArray<float> pratec, FArray<float> nca, bool  f_qi, bool  f_qs, bool  warm_rain, FArray<float> cutop, FArray<float> cubot, float  cudt, int  ids, int  ide, int  jds, int  jde, int  kds, int  kde, int  ims, int  ime, int  jms, int  jme, int  kms, int  kme, int  its, int  ite, int  jts, int  jte, int  kts, int  kte) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    FArray<float> q0 = arena.alloc<float>(kts, kte);
    FArray<float> z0 = arena.alloc<float>(kts, kte);
    FArray<float> tv0 = arena.alloc<float>(kts, kte);
    FArray<float> tu = arena.alloc<float>(kts, kte);
    FArray<float> tvu = arena.alloc<float>(kts, kte);
    FArray<float> qu = arena.alloc<float>(kts, kte);
    FArray<float> tz = arena.alloc<float>(kts, kte);
    FArray<float> tvd = arena.alloc<float>(kts, kte);
    FArray<float> qd = arena.alloc<float>(kts, kte);
    FArray<float> qes = arena.alloc<float>(kts, kte);
    FArray<float> thtes = arena.alloc<float>(kts, kte);
    FArray<float> tg = arena.alloc<float>(kts, kte);
    FArray<float> tvg = arena.alloc<float>(kts, kte);
    FArray<float> qg = arena.alloc<float>(kts, kte);
    FArray<float> wu = arena.alloc<float>(kts, kte);
    FArray<float> wd = arena.alloc<float>(kts, kte);
    FArray<float> w0 = arena.alloc<float>(kts, kte);
    FArray<float> ems = arena.alloc<float>(kts, kte);
    FArray<float> emsd = arena.alloc<float>(kts, kte);
    FArray<float> umf = arena.alloc<float>(kts, kte);
    FArray<float> uer = arena.alloc<float>(kts, kte);
    FArray<float> udr = arena.alloc<float>(kts, kte);
    FArray<float> dmf = arena.alloc<float>(kts, kte);
    FArray<float> der = arena.alloc<float>(kts, kte);
    FArray<float> ddr = arena.alloc<float>(kts, kte);
    FArray<float> umf2 = arena.alloc<float>(kts, kte);
    FArray<float> uer2 = arena.alloc<float>(kts, kte);
    FArray<float> udr2 = arena.alloc<float>(kts, kte);
    FArray<float> dmf2 = arena.alloc<float>(kts, kte);
    FArray<float> der2 = arena.alloc<float>(kts, kte);
    FArray<float> ddr2 = arena.alloc<float>(kts, kte);
    FArray<float> dza = arena.alloc<float>(kts, kte);
    FArray<float> thta0 = arena.alloc<float>(kts, kte);
    FArray<float> thetee = arena.alloc<float>(kts, kte);
    FArray<float> thtau = arena.alloc<float>(kts, kte);
    FArray<float> theteu = arena.alloc<float>(kts, kte);
    FArray<float> thtad = arena.alloc<float>(kts, kte);
    FArray<float> theted = arena.alloc<float>(kts, kte);
    FArray<float> qliq = arena.alloc<float>(kts, kte);
    FArray<float> qice = arena.alloc<float>(kts, kte);
    FArray<float> qlqout = arena.alloc<float>(kts, kte);
    FArray<float> qicout = arena.alloc<float>(kts, kte);
    FArray<float> pptliq = arena.alloc<float>(kts, kte);
    FArray<float> pptice = arena.alloc<float>(kts, kte);
    FArray<float> detlq = arena.alloc<float>(kts, kte);
    FArray<float> detic = arena.alloc<float>(kts, kte);
    FArray<float> detlq2 = arena.alloc<float>(kts, kte);
    FArray<float> detic2 = arena.alloc<float>(kts, kte);
    FArray<float> ratio = arena.alloc<float>(kts, kte);
    FArray<float> ratio2 = arena.alloc<float>(kts, kte);
    FArray<float> domgdp = arena.alloc<float>(kts, kte);
    FArray<float> exn = arena.alloc<float>(kts, kte);
    FArray<float> tvqu = arena.alloc<float>(kts, kte);
    FArray<float> dp = arena.alloc<float>(kts, kte);
    FArray<float> rh = arena.alloc<float>(kts, kte);
    FArray<float> eqfrc = arena.alloc<float>(kts, kte);
    FArray<float> wspd = arena.alloc<float>(kts, kte);
    FArray<float> qdt = arena.alloc<float>(kts, kte);
    FArray<float> fxm = arena.alloc<float>(kts, kte);
    FArray<float> thtag = arena.alloc<float>(kts, kte);
    FArray<float> thpa = arena.alloc<float>(kts, kte);
    FArray<float> thfxout = arena.alloc<float>(kts, kte);
    FArray<float> thfxin = arena.alloc<float>(kts, kte);
    FArray<float> qpa = arena.alloc<float>(kts, kte);
    FArray<float> qfxout = arena.alloc<float>(kts, kte);
    FArray<float> qfxin = arena.alloc<float>(kts, kte);
    FArray<float> qlpa = arena.alloc<float>(kts, kte);
    FArray<float> qlfxin = arena.alloc<float>(kts, kte);
    FArray<float> qlfxout = arena.alloc<float>(kts, kte);
    FArray<float> qipa = arena.alloc<float>(kts, kte);
    FArray<float> qifxin = arena.alloc<float>(kts, kte);
    FArray<float> qifxout = arena.alloc<float>(kts, kte);
    FArray<float> qrpa = arena.alloc<float>(kts, kte);
    FArray<float> qrfxin = arena.alloc<float>(kts, kte);
    FArray<float> qrfxout = arena.alloc<float>(kts, kte);
    FArray<float> qspa = arena.alloc<float>(kts, kte);
    FArray<float> qsfxin = arena.alloc<float>(kts, kte);
    FArray<float> qsfxout = arena.alloc<float>(kts, kte);
    FArray<float> ql0 = arena.alloc<float>(kts, kte);
    FArray<float> qlg = arena.alloc<float>(kts, kte);
    FArray<float> qi0 = arena.alloc<float>(kts, kte);
    FArray<float> qig = arena.alloc<float>(kts, kte);
    FArray<float> qr0 = arena.alloc<float>(kts, kte);
    FArray<float> qrg = arena.alloc<float>(kts, kte);
    FArray<float> qs0 = arena.alloc<float>(kts, kte);
    FArray<float> qsg = arena.alloc<float>(kts, kte);
    FArray<float> omg = arena.alloc<float>(kts, (kte + 1));
    FArray<float> rainfb = arena.alloc<float>(kts, kte);
    FArray<float> snowfb = arena.alloc<float>(kts, kte);
    FArray<float> cldhgt = arena.alloc<float>(kts, kte);
    FArray<float> qsd = arena.alloc<float>(kts, kte);
    FArray<float> dilfrc = arena.alloc<float>(kts, kte);
    FArray<float> ddilfrc = arena.alloc<float>(kts, kte);
    FArray<float> tke = arena.alloc<float>(kts, kte);
    FArray<float> tgu = arena.alloc<float>(kts, kte);
    FArray<float> qgu = arena.alloc<float>(kts, kte);
    FArray<float> thteeg = arena.alloc<float>(kts, kte);
    float p00 = 1.e5f;
    float t00 = 273.16f;
    float rlf = 3.339e5f;
    float rhic = 1.f;
    float rhbc = 0.90f;
    float pie = 3.141592654f;
    float ttfrz = 268.16f;
    float tbfrz = 248.16f;
    float c5 = 1.0723e-3f;
    float rate = 0.03f;
    float gdry;
    float rocp;
    float aliq;
    float bliq;
    float cliq;
    float dliq;
    float fbfrc;
    float p300;
    float dpthmx;
    float thmix;
    float qmix;
    float zmix;
    float pmix;
    float rocpq;
    float tmix;
    float emix;
    float tlog;
    float tdpt;
    float tlcl;
    float tvlcl;
    float cporq;
    float plcl;
    float es;
    float dlp;
    float tenv;
    float qenv;
    float tven;
    float tvbar;
    float zlcl;
    float wkl;
    float wabs;
    float trppt;
    float wsigne;
    float dtlcl;
    float gdt;
    float wlcl;
    float tvavg;
    float qese;
    float wtw;
    float rholcl;
    float au0;
    float vmflcl;
    float upold;
    float upnew;
    float abe;
    float wklcl;
    float ttemp;
    float frc1;
    float qnewic;
    float rl;
    float r1;
    float qnwfrz;
    float effq;
    float be;
    float boterm;
    float enterm;
    float dzz;
    float udlbe;
    float rei;
    float ee2;
    float ud2;
    float ttmp;
    float f1;
    float f2;
    float thttmp;
    float qtmp;
    float tmpliq;
    float tmpice;
    float tu95;
    float tu10;
    float ee1;
    float ud1;
    float dptt;
    float qnewlq;
    float dumfdp;
    float ee;
    float tsat;
    float thta;
    float vconv;
    float timec;
    float shsign;
    float vws;
    float pef;
    float cbh;
    float rcbh;
    float pefcbh;
    float peff;
    float peff2;
    float tder;
    float thtmin;
    float dtmltd;
    float qs;
    float tadvec;
    float dpdd;
    float frc;
    float dpt;
    float rdd;
    float a1;
    float dssdt;
    float dtmp;
    float t1rh;
    float qsrh;
    float pptflx;
    float cpr;
    float cndtnf;
    float updinc;
    float aincm2;
    float devdmf;
    float ppr;
    float rced;
    float dpptdf;
    float dmflfs;
    float dmflfs2;
    float rced2;
    float ddinc;
    float aincmx;
    float aincm1;
    float ainc;
    float tder2;
    float pptfl2;
    float fabe;
    float stab;
    float dtt;
    float dtt1;
    float dtime;
    float tma;
    float tmb;
    float tmm;
    float bcoeff;
    float acoeff;
    float qvdiff;
    float topomg;
    float cpm;
    float dq;
    float abeg;
    float dabe;
    float dfda;
    float frc2;
    float dr;
    float udfrc;
    float tuc;
    float qgs;
    float rh0;
    float rhg;
    float qinit;
    float qfnl;
    float err2;
    float relerr;
    float rlc;
    float rls;
    float rnc;
    float fabeold;
    float aincold;
    float uefrc;
    float ddfrc;
    float tdc;
    float defrc;
    float rhbar;
    float dmffrc;
    float dpmin;
    float dilbe;
    float astrt;
    float tp;
    float value;
    float aintrp;
    float tkemax;
    float qfrz;
    float qss;
    float pptmlt;
    float dtmelt;
    float rhh;
    float evac;
    float binc;
    int indlu;
    int nu;
    int nuchm;
    int nnn;
    int klfs;
    float chmin;
    float pm15;
    float chmax;
    float dtrh;
    float rad;
    float dppp;
    float tvdiff;
    float dttot;
    float absomg;
    float absomgtc;
    float frdp;
    int kx;
    int k;
    int kl;
    int ncheck;
    FArray<int> kcheck = arena.alloc<int>(kts, kte);
    int istop;
    int ml;
    int l5;
    int kmix;
    int low;
    int lc;
    int mxlayr;
    int llfc;
    int nlayrs;
    int nk;
    int kpbl;
    int klcl;
    int lcl;
    int let;
    int iflag;
    int nk1;
    int ltop;
    int nj;
    int ltop1;
    int ltopm1;
    int lvf;
    int kstart;
    int kmin;
    int lfs;
    int nd;
    int nic;
    int ldb;
    int ldt;
    int nd1;
    int ndk;
    int nm;
    int lmax;
    int ncount;
    int noitr;
    int nstep;
    int ntc;
    int nchm;
    int ishall;
    int nshall;
    bool iprnt;
    iprnt = false; // L515
    gdry = __fdiv_rn((-g), cp); // L516
    rocp = __fdiv_rn(r, cp); // L517
    nshall = 0; // L518
    kl = kte; // L519
    kx = kte; // L520
    aliq = __fmul_rn(svp1, 1000.f); // L526
    bliq = svp2; // L527
    cliq = __fmul_rn(svp2, svpt0); // L528
    dliq = svp3; // L529
    fbfrc = 0.0f; // L538
    nchm = 0; // L540
    ishall = 0; // L541
    dpmin = 5.e3f; // L542
    p300 = __fsub_rn(p0(1), 30000.f); // L544
    ml = 0; // L553
    for (k = 1; k <= kx; k += 1) { loop1_cycle: ; // L557
    es = __fmul_rn(aliq, gfk_exp_fma(__fdiv_rn(__fsub_rn(__fmul_rn(bliq, t0(k)), cliq), __fsub_rn(t0(k), dliq)))); // L561
    qes(k) = __fdiv_rn(__fmul_rn(0.622f, es), __fsub_rn(p0(k), es)); // L562
    q0(k) = kf_min<float>(qes(k), qv0(k)); // L563
    q0(k) = kf_max<float>(0.000001f, q0(k)); // L564
    ql0(k) = 0.f; // L565
    qi0(k) = 0.f; // L566
    qr0(k) = 0.f; // L567
    qs0(k) = 0.f; // L568
    rh(k) = __fdiv_rn(q0(k), qes(k)); // L569
    dilfrc(k) = 1.f; // L570
    tv0(k) = __fmul_rn(t0(k), __fadd_rn(1.f, __fmul_rn(0.608f, q0(k)))); // L571
    dp(k) = __fmul_rn(__fmul_rn(rhoe(k), g), dzq(k)); // L574
    tke(k) = 0.f; // L578
    cldhgt(k) = 0.f; // L579
    if ((p0(k) >= __fmul_rn(0.5f, p0(1)))) { l5 = k; } // L581
    if ((p0(k) >= p300)) { llfc = k; } // L582
    loop1_continue: ;
    } loop1_exit: ; // L583
    z0(1) = __fmul_rn(.5f, dzq(1)); // L586
    for (k = 2; k <= kl; k += 1) { loop2_cycle: ; // L588
    z0(k) = __fadd_rn(z0((k - 1)), __fmul_rn(.5f, __fadd_rn(dzq(k), dzq((k - 1))))); // L589
    dza((k - 1)) = __fsub_rn(z0(k), z0((k - 1))); // L590
    loop2_continue: ;
    } loop2_exit: ; // L591
    dza(kl) = 0.f; // L592
    ncheck = 1; // L602
    kcheck(ncheck) = 1; // L603
    pm15 = __fsub_rn(p0(1), 15.e2f); // L604
    for (k = 2; k <= llfc; k += 1) { loop3_cycle: ; // L605
    if ((p0(k) < pm15)) { // L606
    ncheck = (ncheck + 1); // L607
    kcheck(ncheck) = k; // L608
    pm15 = __fsub_rn(pm15, 15.e2f); // L609
    } // L610
    loop3_continue: ;
    } loop3_exit: ; // L611
    nu = 0; // L613
    nuchm = 0; // L614
    for (;;) { usl_cycle: ; // L615
    nu = (nu + 1); // L616
    if ((nu > ncheck)) { // L617
    if ((ishall == 1)) { // L618
    chmax = 0.f; // L619
    nchm = 0; // L620
    for (nk = 1; nk <= ncheck; nk += 1) { loop5_cycle: ; // L621
    nnn = kcheck(nk); // L622
    if ((cldhgt(nnn) > chmax)) { // L623
    nchm = nnn; // L624
    nuchm = nk; // L625
    chmax = cldhgt(nnn); // L626
    } // L627
    loop5_continue: ;
    } loop5_exit: ; // L628
    nu = (nuchm - 1); // L629
    fbfrc = 1.f; // L630
    goto usl_continue; // L631
    } else { // L632
    return; // L633
    } // L634
    } // L635
    kmix = kcheck(nu); // L636
    low = kmix; // L637
    lc = low; // L639
    nlayrs = 0; // L646
    dpthmx = 0.f; // L647
    nk = (lc - 1); // L648
    if (((nk + 1) < kts)) { // L649
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L650
    } else { // L652
    for (;;) { loop6_cycle: ; // L653
    nk = (nk + 1); // L654
    if ((nk > kte)) { // L655
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L656
    goto loop6_exit; // L658
    } // L659
    dpthmx = __fadd_rn(dpthmx, dp(nk)); // L660
    nlayrs = (nlayrs + 1); // L661
    if ((dpthmx > dpmin)) { // L662
    goto loop6_exit; // L663
    } // L664
    loop6_continue: ;
    } loop6_exit: ; // L665
    } // L666
    if ((dpthmx < dpmin)) { // L667
    return; // L668
    } // L669
    kpbl = ((lc + nlayrs) - 1); // L670
    tmix = 0.f; // L677
    qmix = 0.f; // L678
    zmix = 0.f; // L679
    pmix = 0.f; // L680
    for (nk = lc; nk <= kpbl; nk += 1) { loop7_cycle: ; // L687
    tmix = __fadd_rn(tmix, __fmul_rn(dp(nk), t0(nk))); // L688
    qmix = __fadd_rn(qmix, __fmul_rn(dp(nk), q0(nk))); // L689
    zmix = __fadd_rn(zmix, __fmul_rn(dp(nk), z0(nk))); // L690
    pmix = __fadd_rn(pmix, __fmul_rn(dp(nk), p0(nk))); // L691
    loop7_continue: ;
    } loop7_exit: ; // L692
    tmix = __fdiv_rn(tmix, dpthmx); // L694
    qmix = __fdiv_rn(qmix, dpthmx); // L695
    zmix = __fdiv_rn(zmix, dpthmx); // L696
    pmix = __fdiv_rn(pmix, dpthmx); // L697
    emix = __fdiv_rn(__fmul_rn(qmix, pmix), __fadd_rn(0.622f, qmix)); // L698
    astrt = 1.e-3f; // L705
    ainc = 0.075f; // L706
    a1 = __fdiv_rn(emix, aliq); // L707
    tp = __fdiv_rn(__fsub_rn(a1, astrt), ainc); // L708
    indlu = (int(tp) + 1); // L709
    value = __fadd_rn(__fmul_rn((indlu - 1), ainc), astrt); // L710
    aintrp = __fdiv_rn(__fsub_rn(a1, value), ainc); // L711
    tlog = __fadd_rn(__fmul_rn(aintrp, tb.alu((indlu + 1))), __fmul_rn(__fsub_rn(1, aintrp), tb.alu(indlu))); // L712
    tdpt = __fdiv_rn(__fsub_rn(cliq, __fmul_rn(dliq, tlog)), __fsub_rn(bliq, tlog)); // L713
    tlcl = __fsub_rn(tdpt, __fmul_rn(__fsub_rn(__fadd_rn(.212f, __fmul_rn(1.571e-3f, __fsub_rn(tdpt, t00))), __fmul_rn(4.36e-4f, __fsub_rn(tmix, t00))), __fsub_rn(tmix, tdpt))); // L714
    tlcl = kf_min<float>(tlcl, tmix); // L715
    tvlcl = __fmul_rn(tlcl, __fadd_rn(1.f, __fmul_rn(0.608f, qmix))); // L716
    zlcl = __fadd_rn(zmix, __fdiv_rn(__fsub_rn(tlcl, tmix), gdry)); // L717
    nk = (lc - 1); // L718
    for (;;) { loop8_cycle: ; // L719
    nk = (nk + 1); // L720
    klcl = nk; // L721
    if (((zlcl <= z0(nk)) || (nk > kl))) { // L722
    goto loop8_exit; // L723
    } // L724
    loop8_continue: ;
    } loop8_exit: ; // L725
    if ((nk > kl)) { // L726
    return; // L727
    } // L728
    k = (klcl - 1); // L729
    dlp = __fdiv_rn(__fsub_rn(zlcl, z0(k)), __fsub_rn(z0(klcl), z0(k))); // L730
    tenv = __fadd_rn(t0(k), __fmul_rn(__fsub_rn(t0(klcl), t0(k)), dlp)); // L734
    qenv = __fadd_rn(q0(k), __fmul_rn(__fsub_rn(q0(klcl), q0(k)), dlp)); // L735
    tven = __fmul_rn(tenv, __fadd_rn(1.f, __fmul_rn(0.608f, qenv))); // L736
    if ((zlcl < 2.e3f)) { // L748
    wklcl = __fdiv_rn(__fmul_rn(0.02f, zlcl), 2.e3f); // L749
    } else { // L750
    wklcl = 0.02f; // L751
    } // L752
    wkl = __fsub_rn(__fdiv_rn(__fmul_rn(__fadd_rn(w0avg1d(k), __fmul_rn(__fsub_rn(w0avg1d(klcl), w0avg1d(k)), dlp)), dx), 25.e3f), wklcl); // L753
    if ((wkl < 0.0001f)) { // L754
    dtlcl = 0.f; // L755
    } else { // L756
    dtlcl = __fmul_rn(4.64f, gfk_pow(wkl, 0.33f)); // L757
    } // L758
    dtrh = 0.f; // L775
    if ((__fadd_rn(__fadd_rn(tlcl, dtlcl), dtrh) < tenv)) { // L782
    goto usl_continue; // L786
    } else { // L788
    envirtht(tb, arena, pmix, tmix, qmix, theteu(k), aliq, bliq, cliq, dliq); // L794
    dttot = __fadd_rn(dtlcl, dtrh); // L798
    if ((dttot > 1.e-4f)) { // L799
    gdt = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(2.f, g), dttot), 500.f), tven); // L800
    wlcl = __fadd_rn(1.f, __fmul_rn(0.5f, __fsqrt_rn(gdt))); // L801
    wlcl = kf_min<float>(wlcl, 3.f); // L802
    } else { // L803
    wlcl = 1.f; // L804
    } // L805
    plcl = __fadd_rn(p0(k), __fmul_rn(__fsub_rn(p0(klcl), p0(k)), dlp)); // L806
    wtw = __fmul_rn(wlcl, wlcl); // L807
    tvlcl = __fmul_rn(tlcl, __fadd_rn(1.f, __fmul_rn(0.608f, qmix))); // L809
    rholcl = __fdiv_rn(plcl, __fmul_rn(r, tvlcl)); // L810
    lcl = klcl; // L812
    let = lcl; // L813
    if ((wkl < 0.f)) { // L815
    rad = 1000.f; // L816
    } else if ((wkl > 0.1f)) { // L817
    rad = 2000.f; // L818
    } else { // L819
    rad = __fadd_rn(1000.f, __fdiv_rn(__fmul_rn(1000, wkl), 0.1f)); // L820
    } // L821
    wu(k) = wlcl; // L833
    au0 = __fmul_rn(0.01f, dxsq); // L834
    umf(k) = __fmul_rn(rholcl, au0); // L835
    vmflcl = umf(k); // L836
    upold = vmflcl; // L837
    upnew = upold; // L838
    ratio2(k) = 0.f; // L845
    uer(k) = 0.f; // L846
    abe = 0.f; // L847
    trppt = 0.f; // L848
    tu(k) = tlcl; // L849
    tvu(k) = tvlcl; // L850
    qu(k) = qmix; // L851
    eqfrc(k) = 1.f; // L852
    qliq(k) = 0.f; // L853
    qice(k) = 0.f; // L854
    qlqout(k) = 0.f; // L855
    qicout(k) = 0.f; // L856
    detlq(k) = 0.f; // L857
    detic(k) = 0.f; // L858
    pptliq(k) = 0.f; // L859
    pptice(k) = 0.f; // L860
    iflag = 0; // L861
    ttemp = ttfrz; // L869
    ee1 = 1.f; // L876
    ud1 = 0.f; // L877
    rei = 0.f; // L878
    dilbe = 0.f; // L879
    for (nk = k; nk <= (kl - 1); nk += 1) { updraft_cycle: ; // L880
    nk1 = (nk + 1); // L881
    ratio2(nk1) = ratio2(nk); // L882
    frc1 = 0.f; // L883
    tu(nk1) = t0(nk1); // L884
    theteu(nk1) = theteu(nk); // L885
    qu(nk1) = qu(nk); // L886
    qliq(nk1) = qliq(nk); // L887
    qice(nk1) = qice(nk); // L888
    tpmix2(tb, arena, p0(nk1), theteu(nk1), tu(nk1), qu(nk1), qliq(nk1), qice(nk1), qnewlq, qnewic, xlv1, xlv0); // L889
    if ((tu(nk1) <= ttfrz)) { // L899
    if ((tu(nk1) > tbfrz)) { // L900
    if ((ttemp > ttfrz)) { ttemp = ttfrz; } // L901
    frc1 = __fdiv_rn(__fsub_rn(ttemp, tu(nk1)), __fsub_rn(ttemp, tbfrz)); // L902
    } else { // L903
    frc1 = 1.f; // L904
    iflag = 1; // L905
    } // L906
    ttemp = tu(nk1); // L907
    qfrz = __fmul_rn(__fadd_rn(qliq(nk1), qnewlq), frc1); // L912
    qnewic = __fadd_rn(qnewic, __fmul_rn(qnewlq, frc1)); // L913
    qnewlq = __fsub_rn(qnewlq, __fmul_rn(qnewlq, frc1)); // L914
    qice(nk1) = __fadd_rn(qice(nk1), __fmul_rn(qliq(nk1), frc1)); // L915
    qliq(nk1) = __fsub_rn(qliq(nk1), __fmul_rn(qliq(nk1), frc1)); // L916
    dtfrznew(tb, arena, tu(nk1), p0(nk1), theteu(nk1), qu(nk1), qfrz, qice(nk1), aliq, bliq, cliq, dliq); // L917
    } // L919
    tvu(nk1) = __fmul_rn(tu(nk1), __fadd_rn(1.f, __fmul_rn(0.608f, qu(nk1)))); // L920
    if ((nk == k)) { // L924
    be = __fsub_rn(__fdiv_rn(__fadd_rn(tvlcl, tvu(nk1)), __fadd_rn(tven, tv0(nk1))), 1.f); // L925
    boterm = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(2.f, __fsub_rn(z0(nk1), zlcl)), g), be), 1.5f); // L926
    dzz = __fsub_rn(z0(nk1), zlcl); // L927
    } else { // L928
    be = __fsub_rn(__fdiv_rn(__fadd_rn(tvu(nk), tvu(nk1)), __fadd_rn(tv0(nk), tv0(nk1))), 1.f); // L929
    boterm = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(2.f, dza(nk)), g), be), 1.5f); // L930
    dzz = dza(nk); // L931
    } // L932
    enterm = __fdiv_rn(__fmul_rn(__fmul_rn(2.f, rei), wtw), upold); // L933
    condload(tb, arena, qliq(nk1), qice(nk1), wtw, dzz, boterm, enterm, rate, qnewlq, qnewic, qlqout(nk1), qicout(nk1), g); // L935
    if ((wtw < 1.e-3f)) { // L941
    goto updraft_exit; // L942
    } else { // L943
    wu(nk1) = __fsqrt_rn(wtw); // L944
    } // L945
    envirtht(tb, arena, p0(nk1), t0(nk1), q0(nk1), thetee(nk1), aliq, bliq, cliq, dliq); // L948
    rei = __fdiv_rn(__fmul_rn(__fmul_rn(vmflcl, dp(nk1)), 0.03f), rad); // L952
    tvqu(nk1) = __fmul_rn(tu(nk1), __fsub_rn(__fsub_rn(__fadd_rn(1.f, __fmul_rn(0.608f, qu(nk1))), qliq(nk1)), qice(nk1))); // L953
    if ((nk == k)) { // L954
    dilbe = __fmul_rn(__fsub_rn(__fdiv_rn(__fadd_rn(tvlcl, tvqu(nk1)), __fadd_rn(tven, tv0(nk1))), 1.f), dzz); // L955
    } else { // L956
    dilbe = __fmul_rn(__fsub_rn(__fdiv_rn(__fadd_rn(tvqu(nk), tvqu(nk1)), __fadd_rn(tv0(nk), tv0(nk1))), 1.f), dzz); // L957
    } // L958
    if ((dilbe > 0.f)) { abe = __fadd_rn(abe, __fmul_rn(dilbe, g)); } // L959
    if ((tvqu(nk1) <= tv0(nk1))) { // L964
    ee2 = 0.5f; // L965
    ud2 = 1.f; // L966
    eqfrc(nk1) = 0.f; // L967
    } else { // L968
    let = nk1; // L969
    ttmp = tvqu(nk1); // L970
    f1 = 0.95f; // L974
    f2 = __fsub_rn(1.f, f1); // L975
    thttmp = __fadd_rn(__fmul_rn(f1, thetee(nk1)), __fmul_rn(f2, theteu(nk1))); // L976
    qtmp = __fadd_rn(__fmul_rn(f1, q0(nk1)), __fmul_rn(f2, qu(nk1))); // L977
    tmpliq = __fmul_rn(f2, qliq(nk1)); // L978
    tmpice = __fmul_rn(f2, qice(nk1)); // L979
    tpmix2(tb, arena, p0(nk1), thttmp, ttmp, qtmp, tmpliq, tmpice, qnewlq, qnewic, xlv1, xlv0); // L980
    tu95 = __fmul_rn(ttmp, __fsub_rn(__fsub_rn(__fadd_rn(1.f, __fmul_rn(0.608f, qtmp)), tmpliq), tmpice)); // L982
    if ((tu95 > tv0(nk1))) { // L983
    ee2 = 1.f; // L984
    ud2 = 0.f; // L985
    eqfrc(nk1) = 1.0f; // L986
    } else { // L987
    f1 = 0.10f; // L988
    f2 = __fsub_rn(1.f, f1); // L989
    thttmp = __fadd_rn(__fmul_rn(f1, thetee(nk1)), __fmul_rn(f2, theteu(nk1))); // L990
    qtmp = __fadd_rn(__fmul_rn(f1, q0(nk1)), __fmul_rn(f2, qu(nk1))); // L991
    tmpliq = __fmul_rn(f2, qliq(nk1)); // L992
    tmpice = __fmul_rn(f2, qice(nk1)); // L993
    tpmix2(tb, arena, p0(nk1), thttmp, ttmp, qtmp, tmpliq, tmpice, qnewlq, qnewic, xlv1, xlv0); // L994
    tu10 = __fmul_rn(ttmp, __fsub_rn(__fsub_rn(__fadd_rn(1.f, __fmul_rn(0.608f, qtmp)), tmpliq), tmpice)); // L996
    tvdiff = fabsf(__fsub_rn(tu10, tvqu(nk1))); // L997
    if ((tvdiff < 1.e-3f)) { // L998
    ee2 = 1.f; // L999
    ud2 = 0.f; // L1000
    eqfrc(nk1) = 1.0f; // L1001
    } else { // L1002
    eqfrc(nk1) = __fdiv_rn(__fmul_rn(__fsub_rn(tv0(nk1), tvqu(nk1)), f1), __fsub_rn(tu10, tvqu(nk1))); // L1003
    eqfrc(nk1) = kf_max<float>(0.0f, eqfrc(nk1)); // L1004
    eqfrc(nk1) = kf_min<float>(1.0f, eqfrc(nk1)); // L1005
    if ((eqfrc(nk1) == 1)) { // L1006
    ee2 = 1.f; // L1007
    ud2 = 0.f; // L1008
    } else if ((eqfrc(nk1) == 0.f)) { // L1009
    ee2 = 0.f; // L1010
    ud2 = 1.f; // L1011
    } else { // L1012
    prof5(tb, arena, eqfrc(nk1), ee2, ud2); // L1017
    } // L1018
    } // L1019
    } // L1020
    } // L1021
    ee2 = kf_max<float>(ee2, 0.5f); // L1027
    ud2 = __fmul_rn(1.5f, ud2); // L1028
    uer(nk1) = __fmul_rn(__fmul_rn(0.5f, rei), __fadd_rn(ee1, ee2)); // L1029
    udr(nk1) = __fmul_rn(__fmul_rn(0.5f, rei), __fadd_rn(ud1, ud2)); // L1030
    if ((__fsub_rn(umf(nk), udr(nk1)) < 10.f)) { // L1035
    if ((dilbe > 0.f)) { // L1041
    abe = __fsub_rn(abe, __fmul_rn(dilbe, g)); // L1042
    } // L1043
    let = nk; // L1044
    goto updraft_exit; // L1046
    } else { // L1047
    ee1 = ee2; // L1048
    ud1 = ud2; // L1049
    upold = __fsub_rn(umf(nk), udr(nk1)); // L1050
    upnew = __fadd_rn(upold, uer(nk1)); // L1051
    umf(nk1) = upnew; // L1052
    dilfrc(nk1) = __fdiv_rn(upnew, upold); // L1053
    detlq(nk1) = __fmul_rn(qliq(nk1), udr(nk1)); // L1058
    detic(nk1) = __fmul_rn(qice(nk1), udr(nk1)); // L1059
    qdt(nk1) = qu(nk1); // L1060
    qu(nk1) = __fdiv_rn(__fadd_rn(__fmul_rn(upold, qu(nk1)), __fmul_rn(uer(nk1), q0(nk1))), upnew); // L1061
    theteu(nk1) = __fdiv_rn(__fadd_rn(__fmul_rn(theteu(nk1), upold), __fmul_rn(thetee(nk1), uer(nk1))), upnew); // L1062
    qliq(nk1) = __fdiv_rn(__fmul_rn(qliq(nk1), upold), upnew); // L1063
    qice(nk1) = __fdiv_rn(__fmul_rn(qice(nk1), upold), upnew); // L1064
    pptliq(nk1) = __fmul_rn(qlqout(nk1), umf(nk)); // L1071
    pptice(nk1) = __fmul_rn(qicout(nk1), umf(nk)); // L1072
    trppt = __fadd_rn(__fadd_rn(trppt, pptliq(nk1)), pptice(nk1)); // L1074
    if ((nk1 <= kpbl)) { uer(nk1) = __fadd_rn(uer(nk1), __fdiv_rn(__fmul_rn(vmflcl, dp(nk1)), dpthmx)); } // L1075
    } // L1076
    updraft_continue: ;
    } updraft_exit: ; // L1078
    ltop = nk; // L1088
    cldhgt(lc) = __fsub_rn(z0(ltop), zlcl); // L1089
    if ((tlcl > 293.f)) { // L1096
    chmin = 4.e3f; // L1097
    } else if (((tlcl <= 293.f) && (tlcl >= 273))) { // L1098
    chmin = __fadd_rn(2.e3f, __fmul_rn(100.f, __fsub_rn(tlcl, 273.f))); // L1099
    } else if ((tlcl < 273.f)) { // L1100
    chmin = 2.e3f; // L1101
    } // L1102
    if ((((ltop <= klcl) || (ltop <= kpbl)) || ((let + 1) <= kpbl))) { // L1120
    cldhgt(lc) = 0.f; // L1121
    for (nk = k; nk <= ltop; nk += 1) { loop10_cycle: ; // L1122
    umf(nk) = 0.f; // L1123
    udr(nk) = 0.f; // L1124
    uer(nk) = 0.f; // L1125
    detlq(nk) = 0.f; // L1126
    detic(nk) = 0.f; // L1127
    pptliq(nk) = 0.f; // L1128
    pptice(nk) = 0.f; // L1129
    loop10_continue: ;
    } loop10_exit: ; // L1130
    } else if (((cldhgt(lc) > chmin) && (abe > 1))) { // L1132
    ishall = 0; // L1133
    goto usl_exit; // L1134
    } else { // L1135
    ishall = 1; // L1138
    if ((nu == nuchm)) { // L1139
    goto usl_exit; // L1140
    } else { // L1141
    for (nk = k; nk <= ltop; nk += 1) { loop11_cycle: ; // L1143
    umf(nk) = 0.f; // L1144
    udr(nk) = 0.f; // L1145
    uer(nk) = 0.f; // L1146
    detlq(nk) = 0.f; // L1147
    detic(nk) = 0.f; // L1148
    pptliq(nk) = 0.f; // L1149
    pptice(nk) = 0.f; // L1150
    loop11_continue: ;
    } loop11_exit: ; // L1151
    } // L1152
    } // L1153
    } // L1154
    usl_continue: ;
    } usl_exit: ; // L1155
    if ((ishall == 1)) { // L1156
    kstart = kf_max<int>(kpbl, klcl); // L1157
    let = kstart; // L1158
    } // L1159
    if ((let == ltop)) { // L1164
    udr(ltop) = __fsub_rn(__fadd_rn(umf(ltop), udr(ltop)), uer(ltop)); // L1165
    detlq(ltop) = __fdiv_rn(__fmul_rn(__fmul_rn(qliq(ltop), udr(ltop)), upnew), upold); // L1166
    detic(ltop) = __fdiv_rn(__fmul_rn(__fmul_rn(qice(ltop), udr(ltop)), upnew), upold); // L1167
    uer(ltop) = 0.f; // L1168
    umf(ltop) = 0.f; // L1169
    } else { // L1170
    dptt = 0.f; // L1174
    for (nj = (let + 1); nj <= ltop; nj += 1) { loop12_cycle: ; // L1175
    dptt = __fadd_rn(dptt, dp(nj)); // L1176
    loop12_continue: ;
    } loop12_exit: ; // L1177
    dumfdp = __fdiv_rn(umf(let), dptt); // L1178
    for (nk = (let + 1); nk <= ltop; nk += 1) { loop13_cycle: ; // L1183
    if ((nk == ltop)) { // L1191
    udr(nk) = umf((nk - 1)); // L1192
    uer(nk) = 0.f; // L1193
    detlq(nk) = __fmul_rn(__fmul_rn(udr(nk), qliq(nk)), dilfrc(nk)); // L1194
    detic(nk) = __fmul_rn(__fmul_rn(udr(nk), qice(nk)), dilfrc(nk)); // L1195
    } else { // L1196
    umf(nk) = __fsub_rn(umf((nk - 1)), __fmul_rn(dp(nk), dumfdp)); // L1197
    uer(nk) = __fmul_rn(umf(nk), __fsub_rn(1.f, __fdiv_rn(1.f, dilfrc(nk)))); // L1198
    udr(nk) = __fadd_rn(__fsub_rn(umf((nk - 1)), umf(nk)), uer(nk)); // L1199
    detlq(nk) = __fmul_rn(__fmul_rn(udr(nk), qliq(nk)), dilfrc(nk)); // L1200
    detic(nk) = __fmul_rn(__fmul_rn(udr(nk), qice(nk)), dilfrc(nk)); // L1201
    } // L1202
    if ((nk >= (let + 2))) { // L1203
    trppt = __fsub_rn(__fsub_rn(trppt, pptliq(nk)), pptice(nk)); // L1204
    pptliq(nk) = __fmul_rn(umf((nk - 1)), qlqout(nk)); // L1205
    pptice(nk) = __fmul_rn(umf((nk - 1)), qicout(nk)); // L1206
    trppt = __fadd_rn(__fadd_rn(trppt, pptliq(nk)), pptice(nk)); // L1207
    } // L1208
    loop13_continue: ;
    } loop13_exit: ; // L1209
    } // L1210
    for (nk = 1; nk <= ltop; nk += 1) { loop14_cycle: ; // L1214
    if ((t0(nk) > t00)) { ml = nk; } // L1215
    loop14_continue: ;
    } loop14_exit: ; // L1216
    for (nk = 1; nk <= k; nk += 1) { loop15_cycle: ; // L1217
    if ((nk >= lc)) { // L1218
    if ((nk == lc)) { // L1219
    umf(nk) = __fdiv_rn(__fmul_rn(vmflcl, dp(nk)), dpthmx); // L1220
    uer(nk) = __fdiv_rn(__fmul_rn(vmflcl, dp(nk)), dpthmx); // L1221
    } else if ((nk <= kpbl)) { // L1222
    uer(nk) = __fdiv_rn(__fmul_rn(vmflcl, dp(nk)), dpthmx); // L1223
    umf(nk) = __fadd_rn(umf((nk - 1)), uer(nk)); // L1224
    } else { // L1225
    umf(nk) = vmflcl; // L1226
    uer(nk) = 0.f; // L1227
    } // L1228
    tu(nk) = __fadd_rn(tmix, __fmul_rn(__fsub_rn(z0(nk), zmix), gdry)); // L1229
    qu(nk) = qmix; // L1230
    wu(nk) = wlcl; // L1231
    } else { // L1232
    tu(nk) = 0.f; // L1233
    qu(nk) = 0.f; // L1234
    umf(nk) = 0.f; // L1235
    wu(nk) = 0.f; // L1236
    uer(nk) = 0.f; // L1237
    } // L1238
    udr(nk) = 0.f; // L1239
    qdt(nk) = 0.f; // L1240
    qliq(nk) = 0.f; // L1241
    qice(nk) = 0.f; // L1242
    qlqout(nk) = 0.f; // L1243
    qicout(nk) = 0.f; // L1244
    pptliq(nk) = 0.f; // L1245
    pptice(nk) = 0.f; // L1246
    detlq(nk) = 0.f; // L1247
    detic(nk) = 0.f; // L1248
    ratio2(nk) = 0.f; // L1249
    envirtht(tb, arena, p0(nk), t0(nk), q0(nk), thetee(nk), aliq, bliq, cliq, dliq); // L1250
    eqfrc(nk) = 1.0f; // L1251
    loop15_continue: ;
    } loop15_exit: ; // L1252
    ltop1 = (ltop + 1); // L1254
    ltopm1 = (ltop - 1); // L1255
    for (nk = ltop1; nk <= kx; nk += 1) { loop16_cycle: ; // L1259
    umf(nk) = 0.f; // L1260
    udr(nk) = 0.f; // L1261
    uer(nk) = 0.f; // L1262
    qdt(nk) = 0.f; // L1263
    qliq(nk) = 0.f; // L1264
    qice(nk) = 0.f; // L1265
    qlqout(nk) = 0.f; // L1266
    qicout(nk) = 0.f; // L1267
    detlq(nk) = 0.f; // L1268
    detic(nk) = 0.f; // L1269
    pptliq(nk) = 0.f; // L1270
    pptice(nk) = 0.f; // L1271
    if ((nk > ltop1)) { // L1272
    tu(nk) = 0.f; // L1273
    qu(nk) = 0.f; // L1274
    wu(nk) = 0.f; // L1275
    } // L1276
    thta0(nk) = 0.f; // L1277
    thtau(nk) = 0.f; // L1278
    ems(nk) = 0.f; // L1279
    emsd(nk) = 0.f; // L1280
    tg(nk) = t0(nk); // L1281
    qg(nk) = q0(nk); // L1282
    qlg(nk) = 0.f; // L1283
    qig(nk) = 0.f; // L1284
    qrg(nk) = 0.f; // L1285
    qsg(nk) = 0.f; // L1286
    omg(nk) = 0.f; // L1287
    loop16_continue: ;
    } loop16_exit: ; // L1288
    omg((kx + 1)) = 0.f; // L1289
    for (nk = 1; nk <= ltop; nk += 1) { loop17_cycle: ; // L1290
    ems(nk) = __fdiv_rn(__fmul_rn(dp(nk), dxsq), g); // L1291
    emsd(nk) = __fdiv_rn(1.f, ems(nk)); // L1292
    exn(nk) = gfk_pow(__fdiv_rn(p00, p0(nk)), __fmul_rn(0.2854f, __fsub_rn(1.f, __fmul_rn(0.28f, qdt(nk))))); // L1296
    thtau(nk) = __fmul_rn(tu(nk), exn(nk)); // L1297
    exn(nk) = gfk_pow(__fdiv_rn(p00, p0(nk)), __fmul_rn(0.2854f, __fsub_rn(1.f, __fmul_rn(0.28f, q0(nk))))); // L1298
    thta0(nk) = __fmul_rn(t0(nk), exn(nk)); // L1299
    ddilfrc(nk) = __fdiv_rn(1.f, dilfrc(nk)); // L1300
    omg(nk) = 0.f; // L1301
    loop17_continue: ;
    } loop17_exit: ; // L1302
    wspd(klcl) = __fsqrt_rn(__fadd_rn(__fmul_rn(u0(klcl), u0(klcl)), __fmul_rn(v0(klcl), v0(klcl)))); // L1313
    wspd(l5) = __fsqrt_rn(__fadd_rn(__fmul_rn(u0(l5), u0(l5)), __fmul_rn(v0(l5), v0(l5)))); // L1314
    wspd(ltop) = __fsqrt_rn(__fadd_rn(__fmul_rn(u0(ltop), u0(ltop)), __fmul_rn(v0(ltop), v0(ltop)))); // L1315
    vconv = __fmul_rn(.5f, __fadd_rn(wspd(klcl), wspd(l5))); // L1316
    timec = __fdiv_rn(dx, vconv); // L1319
    tadvec = timec; // L1320
    timec = kf_max<float>(1800.f, timec); // L1321
    timec = kf_min<float>(3600.f, timec); // L1322
    if ((ishall == 1)) { timec = 2400.f; } // L1323
    nic = int(roundf(__fdiv_rn(timec, dt))); // L1324
    timec = __fmul_rn(float(nic), dt); // L1325
    if ((wspd(ltop) > wspd(klcl))) { // L1329
    shsign = 1.f; // L1330
    } else { // L1331
    shsign = (-1.f); // L1332
    } // L1333
    vws = __fadd_rn(__fmul_rn(__fsub_rn(u0(ltop), u0(klcl)), __fsub_rn(u0(ltop), u0(klcl))), __fmul_rn(__fsub_rn(v0(ltop), v0(klcl)), __fsub_rn(v0(ltop), v0(klcl)))); // L1334
    vws = __fdiv_rn(__fmul_rn(__fmul_rn(1.e3f, shsign), __fsqrt_rn(vws)), __fsub_rn(z0(ltop), z0(lcl))); // L1336
    pef = __fadd_rn(1.591f, __fmul_rn(vws, __fadd_rn((-.639f), __fmul_rn(vws, __fsub_rn(9.53e-2f, __fmul_rn(vws, 4.96e-3f)))))); // L1337
    pef = kf_max<float>(pef, .2f); // L1338
    pef = kf_min<float>(pef, .9f); // L1339
    cbh = __fmul_rn(__fsub_rn(zlcl, z0(1)), 3.281e-3f); // L1343
    if ((cbh < 3.f)) { // L1344
    rcbh = .02f; // L1345
    } else { // L1346
    rcbh = __fadd_rn(.96729352f, __fmul_rn(cbh, __fadd_rn((-.70034167f), __fmul_rn(cbh, __fadd_rn(.162179896f, __fmul_rn(cbh, __fadd_rn((-1.2569798e-2f), __fmul_rn(cbh, __fsub_rn(4.2772e-4f, __fmul_rn(cbh, 5.44e-6f)))))))))); // L1347
    } // L1349
    if ((cbh > 25)) { rcbh = 2.4f; } // L1350
    pefcbh = __fdiv_rn(1.f, __fadd_rn(1.f, rcbh)); // L1351
    pefcbh = kf_min<float>(pefcbh, .9f); // L1352
    peff = __fmul_rn(.5f, __fadd_rn(pef, pefcbh)); // L1356
    peff2 = peff; // L1357
    if (iprnt) { // L1358
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1360
    } // L1363
    tder = 0.f; // L1372
    if ((ishall == 1)) { // L1373
    lfs = 1; // L1374
    } else { // L1375
    kstart = (kpbl + 1); // L1381
    klfs = (let - 1); // L1382
    for (nk = (kstart + 1); nk <= kl; nk += 1) { loop18_cycle: ; // L1383
    dppp = __fsub_rn(p0(kstart), p0(nk)); // L1384
    if ((dppp > 150.e2f)) { // L1386
    klfs = nk; // L1387
    goto loop18_exit; // L1388
    } // L1389
    loop18_continue: ;
    } loop18_exit: ; // L1390
    klfs = kf_min<int>(klfs, (let - 1)); // L1391
    lfs = klfs; // L1392
    if ((__fsub_rn(p0(kstart), p0(lfs)) > 50.e2f)) { // L1398
    theted(lfs) = thetee(lfs); // L1399
    qd(lfs) = q0(lfs); // L1400
    tpmix2dd(tb, arena, p0(lfs), theted(lfs), tz(lfs), qss, i, j); // L1404
    thtad(lfs) = __fmul_rn(tz(lfs), gfk_pow(__fdiv_rn(p00, p0(lfs)), __fmul_rn(0.2854f, __fsub_rn(1.f, __fmul_rn(0.28f, qss))))); // L1405
    tvd(lfs) = __fmul_rn(tz(lfs), __fadd_rn(1.f, __fmul_rn(0.608f, qss))); // L1409
    rdd = __fdiv_rn(p0(lfs), __fmul_rn(r, tvd(lfs))); // L1410
    a1 = __fmul_rn(__fsub_rn(1.f, peff), au0); // L1411
    dmf(lfs) = __fmul_rn((-a1), rdd); // L1412
    der(lfs) = dmf(lfs); // L1413
    ddr(lfs) = 0.f; // L1414
    rhbar = __fmul_rn(rh(lfs), dp(lfs)); // L1415
    dptt = dp(lfs); // L1416
    for (nd = (lfs - 1); nd >= kstart; nd += (-1)) { loop19_cycle: ; // L1417
    nd1 = (nd + 1); // L1418
    der(nd) = __fdiv_rn(__fmul_rn(der(lfs), ems(nd)), ems(lfs)); // L1419
    ddr(nd) = 0.f; // L1420
    dmf(nd) = __fadd_rn(dmf(nd1), der(nd)); // L1421
    theted(nd) = __fdiv_rn(__fadd_rn(__fmul_rn(theted(nd1), dmf(nd1)), __fmul_rn(thetee(nd), der(nd))), dmf(nd)); // L1422
    qd(nd) = __fdiv_rn(__fadd_rn(__fmul_rn(qd(nd1), dmf(nd1)), __fmul_rn(q0(nd), der(nd))), dmf(nd)); // L1423
    dptt = __fadd_rn(dptt, dp(nd)); // L1424
    rhbar = __fadd_rn(rhbar, __fmul_rn(rh(nd), dp(nd))); // L1425
    loop19_continue: ;
    } loop19_exit: ; // L1426
    rhbar = __fdiv_rn(rhbar, dptt); // L1427
    dmffrc = __fmul_rn(2.f, __fsub_rn(1.f, rhbar)); // L1428
    dpdd = 0.f; // L1429
    pptmlt = 0.f; // L1433
    for (nk = klcl; nk <= ltop; nk += 1) { loop20_cycle: ; // L1434
    pptmlt = __fadd_rn(pptmlt, pptice(nk)); // L1435
    loop20_continue: ;
    } loop20_exit: ; // L1436
    if ((lc < ml)) { // L1437
    dtmelt = __fdiv_rn(__fmul_rn(rlf, pptmlt), __fmul_rn(cp, umf(klcl))); // L1441
    } else { // L1442
    dtmelt = 0.f; // L1443
    } // L1444
    ldt = kf_min<int>((lfs - 1), (kstart - 1)); // L1445
    tpmix2dd(tb, arena, p0(kstart), theted(kstart), tz(kstart), qss, i, j); // L1447
    tz(kstart) = __fsub_rn(tz(kstart), dtmelt); // L1449
    es = __fmul_rn(aliq, gfk_exp_fma(__fdiv_rn(__fsub_rn(__fmul_rn(bliq, tz(kstart)), cliq), __fsub_rn(tz(kstart), dliq)))); // L1450
    qss = __fdiv_rn(__fmul_rn(0.622f, es), __fsub_rn(p0(kstart), es)); // L1451
    theted(kstart) = __fmul_rn(__fmul_rn(tz(kstart), gfk_pow(__fdiv_rn(1.e5f, p0(kstart)), __fmul_rn(0.2854f, __fsub_rn(1.f, __fmul_rn(0.28f, qss))))), gfk_exp_fma(__fmul_rn(__fmul_rn(__fsub_rn(__fdiv_rn(3374.6525f, tz(kstart)), 2.5403f), qss), __fadd_rn(1.f, __fmul_rn(0.81f, qss))))); // L1452
    ldt = kf_min<int>((lfs - 1), (kstart - 1)); // L1455
    for (nd = ldt; nd >= 1; nd += (-1)) { loop21_cycle: ; // L1456
    dpdd = __fadd_rn(dpdd, dp(nd)); // L1457
    theted(nd) = theted(kstart); // L1458
    qd(nd) = qd(kstart); // L1459
    tpmix2dd(tb, arena, p0(nd), theted(nd), tz(nd), qss, i, j); // L1463
    qsd(nd) = qss; // L1464
    rhh = __fsub_rn(1.f, __fmul_rn(__fdiv_rn(0.2f, 1000.f), __fsub_rn(z0(kstart), z0(nd)))); // L1468
    if ((rhh < 1.f)) { // L1472
    dssdt = __fdiv_rn(__fsub_rn(cliq, __fmul_rn(bliq, dliq)), __fmul_rn(__fsub_rn(tz(nd), dliq), __fsub_rn(tz(nd), dliq))); // L1473
    rl = __fsub_rn(xlv0, __fmul_rn(xlv1, tz(nd))); // L1474
    dtmp = __fdiv_rn(__fmul_rn(__fmul_rn(rl, qss), __fsub_rn(1.f, rhh)), __fadd_rn(cp, __fmul_rn(__fmul_rn(__fmul_rn(rl, rhh), qss), dssdt))); // L1475
    t1rh = __fadd_rn(tz(nd), dtmp); // L1476
    es = __fmul_rn(__fmul_rn(rhh, aliq), gfk_exp_fma(__fdiv_rn(__fsub_rn(__fmul_rn(bliq, t1rh), cliq), __fsub_rn(t1rh, dliq)))); // L1477
    qsrh = __fdiv_rn(__fmul_rn(0.622f, es), __fsub_rn(p0(nd), es)); // L1478
    if ((qsrh < qd(nd))) { // L1483
    qsrh = qd(nd); // L1484
    t1rh = __fadd_rn(tz(nd), __fdiv_rn(__fmul_rn(__fsub_rn(qss, qsrh), rl), cp)); // L1485
    } // L1486
    tz(nd) = t1rh; // L1487
    qss = qsrh; // L1488
    qsd(nd) = qss; // L1489
    } // L1490
    tvd(nd) = __fmul_rn(tz(nd), __fadd_rn(1.f, __fmul_rn(0.608f, qsd(nd)))); // L1491
    if (((tvd(nd) > tv0(nd)) || (nd == 1))) { // L1492
    ldb = nd; // L1493
    goto loop21_exit; // L1494
    } // L1495
    loop21_continue: ;
    } loop21_exit: ; // L1496
    if ((__fsub_rn(p0(ldb), p0(lfs)) > 50.e2f)) { // L1497
    for (nd = ldt; nd >= ldb; nd += (-1)) { loop22_cycle: ; // L1498
    nd1 = (nd + 1); // L1499
    ddr(nd) = __fdiv_rn(__fmul_rn((-dmf(kstart)), dp(nd)), dpdd); // L1500
    der(nd) = 0.f; // L1501
    dmf(nd) = __fadd_rn(dmf(nd1), ddr(nd)); // L1502
    tder = __fadd_rn(tder, __fmul_rn(__fsub_rn(qsd(nd), qd(nd)), ddr(nd))); // L1503
    qd(nd) = qsd(nd); // L1504
    thtad(nd) = __fmul_rn(tz(nd), gfk_pow(__fdiv_rn(p00, p0(nd)), __fmul_rn(0.2854f, __fsub_rn(1.f, __fmul_rn(0.28f, qd(nd)))))); // L1505
    loop22_continue: ;
    } loop22_exit: ; // L1506
    } // L1507
    } // L1508
    } // L1509
    if ((tder < 1.f)) { // L1514
    pptflx = trppt; // L1517
    cpr = trppt; // L1518
    tder = 0.f; // L1519
    cndtnf = 0.f; // L1520
    updinc = 1.f; // L1521
    ldb = lfs; // L1522
    for (ndk = 1; ndk <= ltop; ndk += 1) { loop23_cycle: ; // L1523
    dmf(ndk) = 0.f; // L1524
    der(ndk) = 0.f; // L1525
    ddr(ndk) = 0.f; // L1526
    thtad(ndk) = 0.f; // L1527
    wd(ndk) = 0.f; // L1528
    tz(ndk) = 0.f; // L1529
    qd(ndk) = 0.f; // L1530
    loop23_continue: ;
    } loop23_exit: ; // L1531
    aincm2 = 100.f; // L1532
    } else { // L1533
    ddinc = __fdiv_rn(__fmul_rn((-dmffrc), umf(klcl)), dmf(kstart)); // L1534
    updinc = 1.f; // L1535
    if ((__fmul_rn(tder, ddinc) > trppt)) { // L1536
    ddinc = __fdiv_rn(trppt, tder); // L1537
    } // L1538
    tder = __fmul_rn(tder, ddinc); // L1539
    for (nk = ldb; nk <= lfs; nk += 1) { loop24_cycle: ; // L1540
    dmf(nk) = __fmul_rn(dmf(nk), ddinc); // L1541
    der(nk) = __fmul_rn(der(nk), ddinc); // L1542
    ddr(nk) = __fmul_rn(ddr(nk), ddinc); // L1543
    loop24_continue: ;
    } loop24_exit: ; // L1544
    cpr = trppt; // L1545
    pptflx = __fsub_rn(trppt, tder); // L1546
    peff = __fdiv_rn(pptflx, trppt); // L1547
    if (iprnt) { // L1548
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1550
    } // L1553
    if ((ldb > 1)) { // L1573
    for (nk = 1; nk <= (ldb - 1); nk += 1) { loop25_cycle: ; // L1574
    dmf(nk) = 0.f; // L1575
    der(nk) = 0.f; // L1576
    ddr(nk) = 0.f; // L1577
    wd(nk) = 0.f; // L1578
    tz(nk) = 0.f; // L1579
    qd(nk) = 0.f; // L1580
    thtad(nk) = 0.f; // L1581
    loop25_continue: ;
    } loop25_exit: ; // L1582
    } // L1583
    for (nk = (lfs + 1); nk <= kx; nk += 1) { loop26_cycle: ; // L1584
    dmf(nk) = 0.f; // L1585
    der(nk) = 0.f; // L1586
    ddr(nk) = 0.f; // L1587
    wd(nk) = 0.f; // L1588
    tz(nk) = 0.f; // L1589
    qd(nk) = 0.f; // L1590
    thtad(nk) = 0.f; // L1591
    loop26_continue: ;
    } loop26_exit: ; // L1592
    for (nk = (ldt + 1); nk <= (lfs - 1); nk += 1) { loop27_cycle: ; // L1593
    tz(nk) = 0.f; // L1594
    qd(nk) = 0.f; // L1595
    thtad(nk) = 0.f; // L1596
    loop27_continue: ;
    } loop27_exit: ; // L1597
    } // L1598
    aincmx = 1000.f; // L1604
    lmax = kf_max<int>(klcl, lfs); // L1605
    for (nk = lc; nk <= lmax; nk += 1) { loop28_cycle: ; // L1606
    if ((__fsub_rn(uer(nk), der(nk)) > 1.e-3f)) { // L1607
    aincm1 = __fdiv_rn(ems(nk), __fmul_rn(__fsub_rn(uer(nk), der(nk)), timec)); // L1608
    aincmx = kf_min<float>(aincmx, aincm1); // L1609
    } // L1610
    loop28_continue: ;
    } loop28_exit: ; // L1611
    ainc = 1.f; // L1612
    if ((aincmx < ainc)) { ainc = aincmx; } // L1613
    tder2 = tder; // L1619
    pptfl2 = pptflx; // L1620
    for (nk = 1; nk <= ltop; nk += 1) { loop29_cycle: ; // L1621
    detlq2(nk) = detlq(nk); // L1622
    detic2(nk) = detic(nk); // L1623
    udr2(nk) = udr(nk); // L1624
    uer2(nk) = uer(nk); // L1625
    ddr2(nk) = ddr(nk); // L1626
    der2(nk) = der(nk); // L1627
    umf2(nk) = umf(nk); // L1628
    dmf2(nk) = dmf(nk); // L1629
    loop29_continue: ;
    } loop29_exit: ; // L1630
    fabe = 1.f; // L1631
    stab = 0.95f; // L1632
    noitr = 0; // L1633
    istop = 0; // L1634
    if ((ishall == 1)) { // L1636
    tkemax = 5.f; // L1644
    evac = __fmul_rn(__fmul_rn(0.5f, tkemax), 0.1f); // L1657
    ainc = __fdiv_rn(__fmul_rn(__fmul_rn(evac, dpthmx), dxsq), __fmul_rn(__fmul_rn(vmflcl, g), timec)); // L1660
    tder = __fmul_rn(tder2, ainc); // L1661
    pptflx = __fmul_rn(pptfl2, ainc); // L1662
    for (nk = 1; nk <= ltop; nk += 1) { loop30_cycle: ; // L1663
    umf(nk) = __fmul_rn(umf2(nk), ainc); // L1664
    dmf(nk) = __fmul_rn(dmf2(nk), ainc); // L1665
    detlq(nk) = __fmul_rn(detlq2(nk), ainc); // L1666
    detic(nk) = __fmul_rn(detic2(nk), ainc); // L1667
    udr(nk) = __fmul_rn(udr2(nk), ainc); // L1668
    uer(nk) = __fmul_rn(uer2(nk), ainc); // L1669
    der(nk) = __fmul_rn(der2(nk), ainc); // L1670
    ddr(nk) = __fmul_rn(ddr2(nk), ainc); // L1671
    loop30_continue: ;
    } loop30_exit: ; // L1672
    } // L1673
    for (ncount = 1; ncount <= 10; ncount += 1) { iter_cycle: ; // L1675
    dtt = timec; // L1686
    for (nk = 1; nk <= ltop; nk += 1) { loop32_cycle: ; // L1687
    domgdp(nk) = __fmul_rn((-__fsub_rn(__fsub_rn(__fsub_rn(uer(nk), der(nk)), udr(nk)), ddr(nk))), emsd(nk)); // L1688
    if ((nk > 1)) { // L1689
    omg(nk) = __fsub_rn(omg((nk - 1)), __fmul_rn(dp((nk - 1)), domgdp((nk - 1)))); // L1690
    absomg = fabsf(omg(nk)); // L1691
    absomgtc = __fmul_rn(absomg, timec); // L1692
    frdp = __fmul_rn(0.75f, dp((nk - 1))); // L1693
    if ((absomgtc > frdp)) { // L1694
    dtt1 = __fdiv_rn(frdp, absomg); // L1695
    dtt = kf_min<float>(dtt, dtt1); // L1696
    } // L1697
    } // L1698
    loop32_continue: ;
    } loop32_exit: ; // L1699
    for (nk = 1; nk <= ltop; nk += 1) { loop33_cycle: ; // L1700
    thpa(nk) = thta0(nk); // L1701
    qpa(nk) = q0(nk); // L1702
    nstep = int(roundf(__fadd_rn(__fdiv_rn(timec, dtt), 1))); // L1703
    dtime = __fdiv_rn(timec, float(nstep)); // L1704
    fxm(nk) = __fdiv_rn(__fmul_rn(omg(nk), dxsq), g); // L1705
    loop33_continue: ;
    } loop33_exit: ; // L1706
    for (ntc = 1; ntc <= nstep; ntc += 1) { loop34_cycle: ; // L1710
    for (nk = 1; nk <= ltop; nk += 1) { loop35_cycle: ; // L1715
    thfxin(nk) = 0.f; // L1716
    thfxout(nk) = 0.f; // L1717
    qfxin(nk) = 0.f; // L1718
    qfxout(nk) = 0.f; // L1719
    loop35_continue: ;
    } loop35_exit: ; // L1720
    for (nk = 2; nk <= ltop; nk += 1) { loop36_cycle: ; // L1721
    if ((omg(nk) <= 0.f)) { // L1722
    thfxin(nk) = __fmul_rn((-fxm(nk)), thpa((nk - 1))); // L1723
    qfxin(nk) = __fmul_rn((-fxm(nk)), qpa((nk - 1))); // L1724
    thfxout((nk - 1)) = __fadd_rn(thfxout((nk - 1)), thfxin(nk)); // L1725
    qfxout((nk - 1)) = __fadd_rn(qfxout((nk - 1)), qfxin(nk)); // L1726
    } else { // L1727
    thfxout(nk) = __fmul_rn(fxm(nk), thpa(nk)); // L1728
    qfxout(nk) = __fmul_rn(fxm(nk), qpa(nk)); // L1729
    thfxin((nk - 1)) = __fadd_rn(thfxin((nk - 1)), thfxout(nk)); // L1730
    qfxin((nk - 1)) = __fadd_rn(qfxin((nk - 1)), qfxout(nk)); // L1731
    } // L1732
    loop36_continue: ;
    } loop36_exit: ; // L1733
    for (nk = 1; nk <= ltop; nk += 1) { loop37_cycle: ; // L1737
    thpa(nk) = __fadd_rn(thpa(nk), __fmul_rn(__fmul_rn(__fsub_rn(__fsub_rn(__fadd_rn(__fadd_rn(thfxin(nk), __fmul_rn(udr(nk), thtau(nk))), __fmul_rn(ddr(nk), thtad(nk))), thfxout(nk)), __fmul_rn(__fsub_rn(uer(nk), der(nk)), thta0(nk))), dtime), emsd(nk))); // L1738
    qpa(nk) = __fadd_rn(qpa(nk), __fmul_rn(__fmul_rn(__fsub_rn(__fsub_rn(__fadd_rn(__fadd_rn(qfxin(nk), __fmul_rn(udr(nk), qdt(nk))), __fmul_rn(ddr(nk), qd(nk))), qfxout(nk)), __fmul_rn(__fsub_rn(uer(nk), der(nk)), q0(nk))), dtime), emsd(nk))); // L1741
    loop37_continue: ;
    } loop37_exit: ; // L1743
    loop34_continue: ;
    } loop34_exit: ; // L1744
    for (nk = 1; nk <= ltop; nk += 1) { loop38_cycle: ; // L1745
    thtag(nk) = thpa(nk); // L1746
    qg(nk) = qpa(nk); // L1747
    loop38_continue: ;
    } loop38_exit: ; // L1748
    for (nk = 1; nk <= ltop; nk += 1) { loop39_cycle: ; // L1753
    if ((qg(nk) < 0.f)) { // L1754
    if ((nk == 1)) { // L1755
    } // L1759
    nk1 = (nk + 1); // L1760
    if ((nk == ltop)) { // L1761
    nk1 = klcl; // L1762
    } // L1763
    tma = __fmul_rn(qg(nk1), ems(nk1)); // L1764
    tmb = __fmul_rn(qg((nk - 1)), ems((nk - 1))); // L1765
    tmm = __fmul_rn(__fsub_rn(qg(nk), 1.e-9f), ems(nk)); // L1766
    bcoeff = __fdiv_rn((-tmm), __fadd_rn(__fdiv_rn(__fmul_rn(tma, tma), tmb), tmb)); // L1767
    acoeff = __fdiv_rn(__fmul_rn(bcoeff, tma), tmb); // L1768
    tmb = __fmul_rn(tmb, __fsub_rn(1.f, bcoeff)); // L1769
    tma = __fmul_rn(tma, __fsub_rn(1.f, acoeff)); // L1770
    if ((nk == ltop)) { // L1771
    qvdiff = __fdiv_rn(__fmul_rn(__fsub_rn(qg(nk1), __fmul_rn(tma, emsd(nk1))), 100.f), qg(nk1)); // L1772
    } // L1779
    qg(nk) = 1.e-9f; // L1780
    qg(nk1) = __fmul_rn(tma, emsd(nk1)); // L1781
    qg((nk - 1)) = __fmul_rn(tmb, emsd((nk - 1))); // L1782
    } // L1783
    loop39_continue: ;
    } loop39_exit: ; // L1784
    topomg = __fmul_rn(__fmul_rn(__fsub_rn(udr(ltop), uer(ltop)), dp(ltop)), emsd(ltop)); // L1785
    if ((fabsf(__fsub_rn(topomg, omg(ltop))) > 1.e-3f)) { // L1786
    istop = 1; // L1790
    iprnt = true; // L1791
    goto iter_exit; // L1792
    } // L1793
    for (nk = 1; nk <= ltop; nk += 1) { loop40_cycle: ; // L1797
    exn(nk) = gfk_pow(__fdiv_rn(p00, p0(nk)), __fmul_rn(0.2854f, __fsub_rn(1.f, __fmul_rn(0.28f, qg(nk))))); // L1798
    tg(nk) = __fdiv_rn(thtag(nk), exn(nk)); // L1799
    tvg(nk) = __fmul_rn(tg(nk), __fadd_rn(1.f, __fmul_rn(0.608f, qg(nk)))); // L1800
    loop40_continue: ;
    } loop40_exit: ; // L1801
    if ((ishall == 1)) { // L1802
    goto iter_exit; // L1803
    } // L1804
    tmix = 0.f; // L1815
    qmix = 0.f; // L1816
    for (nk = lc; nk <= kpbl; nk += 1) { loop41_cycle: ; // L1822
    tmix = __fadd_rn(tmix, __fmul_rn(dp(nk), tg(nk))); // L1823
    qmix = __fadd_rn(qmix, __fmul_rn(dp(nk), qg(nk))); // L1824
    loop41_continue: ;
    } loop41_exit: ; // L1825
    tmix = __fdiv_rn(tmix, dpthmx); // L1826
    qmix = __fdiv_rn(qmix, dpthmx); // L1827
    es = __fmul_rn(aliq, gfk_exp_fma(__fdiv_rn(__fsub_rn(__fmul_rn(tmix, bliq), cliq), __fsub_rn(tmix, dliq)))); // L1828
    qss = __fdiv_rn(__fmul_rn(0.622f, es), __fsub_rn(pmix, es)); // L1829
    if ((qmix > qss)) { // L1833
    rl = __fsub_rn(xlv0, __fmul_rn(xlv1, tmix)); // L1834
    cpm = __fmul_rn(cp, __fadd_rn(1.f, __fmul_rn(0.887f, qmix))); // L1835
    dssdt = __fdiv_rn(__fmul_rn(qss, __fsub_rn(cliq, __fmul_rn(bliq, dliq))), __fmul_rn(__fsub_rn(tmix, dliq), __fsub_rn(tmix, dliq))); // L1836
    dq = __fdiv_rn(__fsub_rn(qmix, qss), __fadd_rn(1.f, __fdiv_rn(__fmul_rn(rl, dssdt), cpm))); // L1837
    tmix = __fadd_rn(tmix, __fmul_rn(__fdiv_rn(rl, cp), dq)); // L1838
    qmix = __fsub_rn(qmix, dq); // L1839
    tlcl = tmix; // L1840
    } else { // L1841
    qmix = kf_max<float>(qmix, 0.f); // L1842
    emix = __fdiv_rn(__fmul_rn(qmix, pmix), __fadd_rn(0.622f, qmix)); // L1843
    astrt = 1.e-3f; // L1844
    binc = 0.075f; // L1845
    a1 = __fdiv_rn(emix, aliq); // L1846
    tp = __fdiv_rn(__fsub_rn(a1, astrt), binc); // L1847
    indlu = (int(tp) + 1); // L1848
    value = __fadd_rn(__fmul_rn((indlu - 1), binc), astrt); // L1849
    aintrp = __fdiv_rn(__fsub_rn(a1, value), binc); // L1850
    tlog = __fadd_rn(__fmul_rn(aintrp, tb.alu((indlu + 1))), __fmul_rn(__fsub_rn(1, aintrp), tb.alu(indlu))); // L1851
    tdpt = __fdiv_rn(__fsub_rn(cliq, __fmul_rn(dliq, tlog)), __fsub_rn(bliq, tlog)); // L1852
    tlcl = __fsub_rn(tdpt, __fmul_rn(__fsub_rn(__fadd_rn(.212f, __fmul_rn(1.571e-3f, __fsub_rn(tdpt, t00))), __fmul_rn(4.36e-4f, __fsub_rn(tmix, t00))), __fsub_rn(tmix, tdpt))); // L1853
    tlcl = kf_min<float>(tlcl, tmix); // L1854
    } // L1855
    tvlcl = __fmul_rn(tlcl, __fadd_rn(1.f, __fmul_rn(0.608f, qmix))); // L1856
    zlcl = __fadd_rn(zmix, __fdiv_rn(__fsub_rn(tlcl, tmix), gdry)); // L1857
    for (nk = lc; nk <= kl; nk += 1) { loop42_cycle: ; // L1858
    klcl = nk; // L1859
    if ((zlcl <= z0(nk))) { // L1860
    goto loop42_exit; // L1861
    } // L1862
    loop42_continue: ;
    } loop42_exit: ; // L1863
    k = (klcl - 1); // L1864
    dlp = __fdiv_rn(__fsub_rn(zlcl, z0(k)), __fsub_rn(z0(klcl), z0(k))); // L1865
    tenv = __fadd_rn(tg(k), __fmul_rn(__fsub_rn(tg(klcl), tg(k)), dlp)); // L1869
    qenv = __fadd_rn(qg(k), __fmul_rn(__fsub_rn(qg(klcl), qg(k)), dlp)); // L1870
    tven = __fmul_rn(tenv, __fadd_rn(1.f, __fmul_rn(0.608f, qenv))); // L1871
    plcl = __fadd_rn(p0(k), __fmul_rn(__fsub_rn(p0(klcl), p0(k)), dlp)); // L1872
    theteu(k) = __fmul_rn(__fmul_rn(tmix, gfk_pow(__fdiv_rn(1.e5f, pmix), __fmul_rn(0.2854f, __fsub_rn(1.f, __fmul_rn(0.28f, qmix))))), gfk_exp_fma(__fmul_rn(__fmul_rn(__fsub_rn(__fdiv_rn(3374.6525f, tlcl), 2.5403f), qmix), __fadd_rn(1.f, __fmul_rn(0.81f, qmix))))); // L1873
    abeg = 0.f; // L1878
    for (nk = k; nk <= ltopm1; nk += 1) { loop43_cycle: ; // L1879
    nk1 = (nk + 1); // L1880
    theteu(nk1) = theteu(nk); // L1881
    tpmix2dd(tb, arena, p0(nk1), theteu(nk1), tgu(nk1), qgu(nk1), i, j); // L1883
    tvqu(nk1) = __fmul_rn(tgu(nk1), __fsub_rn(__fsub_rn(__fadd_rn(1.f, __fmul_rn(0.608f, qgu(nk1))), qliq(nk1)), qice(nk1))); // L1885
    if ((nk == k)) { // L1886
    dzz = __fsub_rn(z0(klcl), zlcl); // L1887
    dilbe = __fmul_rn(__fsub_rn(__fdiv_rn(__fadd_rn(tvlcl, tvqu(nk1)), __fadd_rn(tven, tvg(nk1))), 1.f), dzz); // L1888
    } else { // L1889
    dzz = dza(nk); // L1890
    dilbe = __fmul_rn(__fsub_rn(__fdiv_rn(__fadd_rn(tvqu(nk), tvqu(nk1)), __fadd_rn(tvg(nk), tvg(nk1))), 1.f), dzz); // L1891
    } // L1892
    if ((dilbe > 0.f)) { abeg = __fadd_rn(abeg, __fmul_rn(dilbe, g)); } // L1893
    envirtht(tb, arena, p0(nk1), tg(nk1), qg(nk1), thteeg(nk1), aliq, bliq, cliq, dliq); // L1897
    theteu(nk1) = __fadd_rn(__fmul_rn(theteu(nk1), ddilfrc(nk1)), __fmul_rn(thteeg(nk1), __fsub_rn(1.f, ddilfrc(nk1)))); // L1898
    loop43_continue: ;
    } loop43_exit: ; // L1899
    if ((noitr == 1)) { // L1904
    goto iter_exit; // L1909
    } // L1910
    dabe = kf_max<float>(__fsub_rn(abe, abeg), __fmul_rn(0.1f, abe)); // L1911
    fabe = __fdiv_rn(abeg, abe); // L1912
    if (((fabe > 1.f) && (ishall == 0))) { // L1913
    return; // L1916
    } // L1917
    if ((ncount != 1)) { // L1918
    if ((fabsf(__fsub_rn(ainc, aincold)) < 0.0001f)) { // L1919
    noitr = 1; // L1920
    ainc = aincold; // L1921
    goto iter_continue; // L1922
    } // L1923
    dfda = __fdiv_rn(__fsub_rn(fabe, fabeold), __fsub_rn(ainc, aincold)); // L1924
    if ((dfda > 0.f)) { // L1925
    noitr = 1; // L1926
    ainc = aincold; // L1927
    goto iter_continue; // L1928
    } // L1929
    } // L1930
    aincold = ainc; // L1931
    fabeold = fabe; // L1932
    if (((__fdiv_rn(ainc, aincmx) > 0.999f) && (fabe > __fsub_rn(1.05f, stab)))) { // L1933
    goto iter_exit; // L1938
    } // L1939
    if ((((fabe <= __fsub_rn(1.05f, stab)) && (fabe >= __fsub_rn(0.95f, stab))) || (ncount == 10))) { // L1940
    goto iter_exit; // L1941
    } else { // L1942
    if ((ncount > 10)) { // L1943
    goto iter_exit; // L1948
    } // L1949
    if ((fabe == 0.f)) { // L1954
    ainc = __fmul_rn(ainc, 0.5f); // L1955
    } else { // L1956
    if ((dabe < 1.e-4f)) { // L1957
    noitr = 1; // L1958
    ainc = aincold; // L1959
    goto iter_continue; // L1960
    } else { // L1961
    ainc = __fdiv_rn(__fmul_rn(__fmul_rn(ainc, stab), abe), dabe); // L1962
    } // L1963
    } // L1964
    ainc = kf_min<float>(aincmx, ainc); // L1966
    if ((ainc < 0.05f)) { // L1969
    return; // L1970
    } // L1971
    tder = __fmul_rn(tder2, ainc); // L1973
    pptflx = __fmul_rn(pptfl2, ainc); // L1974
    for (nk = 1; nk <= ltop; nk += 1) { loop44_cycle: ; // L1979
    umf(nk) = __fmul_rn(umf2(nk), ainc); // L1980
    dmf(nk) = __fmul_rn(dmf2(nk), ainc); // L1981
    detlq(nk) = __fmul_rn(detlq2(nk), ainc); // L1982
    detic(nk) = __fmul_rn(detic2(nk), ainc); // L1983
    udr(nk) = __fmul_rn(udr2(nk), ainc); // L1984
    uer(nk) = __fmul_rn(uer2(nk), ainc); // L1985
    der(nk) = __fmul_rn(der2(nk), ainc); // L1986
    ddr(nk) = __fmul_rn(ddr2(nk), ainc); // L1987
    loop44_continue: ;
    } loop44_exit: ; // L1988
    } // L1992
    iter_continue: ;
    } iter_exit: ; // L1993
    if ((cpr > 0.f)) { // L2002
    frc2 = __fdiv_rn(pptflx, __fmul_rn(cpr, ainc)); // L2003
    } else { // L2004
    frc2 = 0.f; // L2005
    } // L2006
    for (nk = 1; nk <= ltop; nk += 1) { loop45_cycle: ; // L2007
    qlpa(nk) = ql0(nk); // L2008
    qipa(nk) = qi0(nk); // L2009
    qrpa(nk) = qr0(nk); // L2010
    qspa(nk) = qs0(nk); // L2011
    rainfb(nk) = __fmul_rn(__fmul_rn(__fmul_rn(pptliq(nk), ainc), fbfrc), frc2); // L2012
    snowfb(nk) = __fmul_rn(__fmul_rn(__fmul_rn(pptice(nk), ainc), fbfrc), frc2); // L2013
    loop45_continue: ;
    } loop45_exit: ; // L2014
    for (ntc = 1; ntc <= nstep; ntc += 1) { loop46_cycle: ; // L2015
    for (nk = 1; nk <= ltop; nk += 1) { loop47_cycle: ; // L2020
    qlfxin(nk) = 0.f; // L2021
    qlfxout(nk) = 0.f; // L2022
    qifxin(nk) = 0.f; // L2023
    qifxout(nk) = 0.f; // L2024
    qrfxin(nk) = 0.f; // L2025
    qrfxout(nk) = 0.f; // L2026
    qsfxin(nk) = 0.f; // L2027
    qsfxout(nk) = 0.f; // L2028
    loop47_continue: ;
    } loop47_exit: ; // L2029
    for (nk = 2; nk <= ltop; nk += 1) { loop48_cycle: ; // L2030
    if ((omg(nk) <= 0.f)) { // L2031
    qlfxin(nk) = __fmul_rn((-fxm(nk)), qlpa((nk - 1))); // L2032
    qifxin(nk) = __fmul_rn((-fxm(nk)), qipa((nk - 1))); // L2033
    qrfxin(nk) = __fmul_rn((-fxm(nk)), qrpa((nk - 1))); // L2034
    qsfxin(nk) = __fmul_rn((-fxm(nk)), qspa((nk - 1))); // L2035
    qlfxout((nk - 1)) = __fadd_rn(qlfxout((nk - 1)), qlfxin(nk)); // L2036
    qifxout((nk - 1)) = __fadd_rn(qifxout((nk - 1)), qifxin(nk)); // L2037
    qrfxout((nk - 1)) = __fadd_rn(qrfxout((nk - 1)), qrfxin(nk)); // L2038
    qsfxout((nk - 1)) = __fadd_rn(qsfxout((nk - 1)), qsfxin(nk)); // L2039
    } else { // L2040
    qlfxout(nk) = __fmul_rn(fxm(nk), qlpa(nk)); // L2041
    qifxout(nk) = __fmul_rn(fxm(nk), qipa(nk)); // L2042
    qrfxout(nk) = __fmul_rn(fxm(nk), qrpa(nk)); // L2043
    qsfxout(nk) = __fmul_rn(fxm(nk), qspa(nk)); // L2044
    qlfxin((nk - 1)) = __fadd_rn(qlfxin((nk - 1)), qlfxout(nk)); // L2045
    qifxin((nk - 1)) = __fadd_rn(qifxin((nk - 1)), qifxout(nk)); // L2046
    qrfxin((nk - 1)) = __fadd_rn(qrfxin((nk - 1)), qrfxout(nk)); // L2047
    qsfxin((nk - 1)) = __fadd_rn(qsfxin((nk - 1)), qsfxout(nk)); // L2048
    } // L2049
    loop48_continue: ;
    } loop48_exit: ; // L2050
    for (nk = 1; nk <= ltop; nk += 1) { loop49_cycle: ; // L2054
    qlpa(nk) = __fadd_rn(qlpa(nk), __fmul_rn(__fmul_rn(__fsub_rn(__fadd_rn(qlfxin(nk), detlq(nk)), qlfxout(nk)), dtime), emsd(nk))); // L2055
    qipa(nk) = __fadd_rn(qipa(nk), __fmul_rn(__fmul_rn(__fsub_rn(__fadd_rn(qifxin(nk), detic(nk)), qifxout(nk)), dtime), emsd(nk))); // L2056
    qrpa(nk) = __fadd_rn(qrpa(nk), __fmul_rn(__fmul_rn(__fadd_rn(__fsub_rn(qrfxin(nk), qrfxout(nk)), rainfb(nk)), dtime), emsd(nk))); // L2057
    qspa(nk) = __fadd_rn(qspa(nk), __fmul_rn(__fmul_rn(__fadd_rn(__fsub_rn(qsfxin(nk), qsfxout(nk)), snowfb(nk)), dtime), emsd(nk))); // L2058
    loop49_continue: ;
    } loop49_exit: ; // L2059
    loop46_continue: ;
    } loop46_exit: ; // L2060
    for (nk = 1; nk <= ltop; nk += 1) { loop50_cycle: ; // L2061
    qlg(nk) = qlpa(nk); // L2062
    qig(nk) = qipa(nk); // L2063
    qrg(nk) = qrpa(nk); // L2064
    qsg(nk) = qspa(nk); // L2065
    loop50_continue: ;
    } loop50_exit: ; // L2066
    if (iprnt) { // L2074
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2076
    } // L2079
    if (iprnt) { // L2084
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2087
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2089
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2092
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2095
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2098
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2101
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2103
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2105
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2109
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2113
    for (nk = 1; nk <= ltop; nk += 1) { loop51_cycle: ; // L2116
    k = ((ltop - nk) + 1); // L2117
    dtt = __fdiv_rn(__fmul_rn(__fsub_rn(tg(k), t0(k)), 86400.f), timec); // L2118
    rl = __fsub_rn(xlv0, __fmul_rn(xlv1, tg(k))); // L2119
    dr = __fdiv_rn(__fmul_rn(__fmul_rn((-__fsub_rn(qg(k), q0(k))), rl), 86400.f), __fmul_rn(timec, cp)); // L2120
    udfrc = __fmul_rn(__fmul_rn(udr(k), timec), emsd(k)); // L2121
    uefrc = __fmul_rn(__fmul_rn(uer(k), timec), emsd(k)); // L2122
    ddfrc = __fmul_rn(__fmul_rn(ddr(k), timec), emsd(k)); // L2123
    defrc = __fmul_rn(__fmul_rn((-der(k)), timec), emsd(k)); // L2124
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2125
    loop51_continue: ;
    } loop51_exit: ; // L2130
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2131
    for (nk = 1; nk <= kl; nk += 1) { loop52_cycle: ; // L2134
    k = ((kx - nk) + 1); // L2135
    dtt = __fsub_rn(tg(k), t0(k)); // L2136
    tuc = __fsub_rn(tu(k), t00); // L2137
    if (((k < lc) || (k > ltop))) { tuc = 0.f; } // L2138
    tdc = __fsub_rn(tz(k), t00); // L2139
    if ((((k < ldb) || (k > ldt)) && (k != lfs))) { tdc = 0.f; } // L2140
    if ((t0(k) < t00)) { // L2141
    es = __fmul_rn(aliq, gfk_exp_fma(__fdiv_rn(__fsub_rn(__fmul_rn(bliq, tg(k)), cliq), __fsub_rn(tg(k), dliq)))); // L2142
    } else { // L2143
    es = __fmul_rn(aliq, gfk_exp_fma(__fdiv_rn(__fsub_rn(__fmul_rn(bliq, tg(k)), cliq), __fsub_rn(tg(k), dliq)))); // L2144
    } // L2145
    qgs = __fdiv_rn(__fmul_rn(es, 0.622f), __fsub_rn(p0(k), es)); // L2146
    rh0 = __fdiv_rn(q0(k), qes(k)); // L2147
    rhg = __fdiv_rn(qg(k), qgs); // L2148
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2149
    loop52_continue: ;
    } loop52_exit: ; // L2154
    for (nk = 1; nk <= kl; nk += 1) { loop53_cycle: ; // L2167
    k = ((kl - nk) + 1); // L2168
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2169
    loop53_continue: ;
    } loop53_exit: ; // L2174
    if ((istop == 1)) { // L2175
    } // L2177
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2179
    } // L2180
    cndtnf = __fmul_rn(__fmul_rn(__fsub_rn(1.f, eqfrc(lfs)), __fadd_rn(qliq(lfs), qice(lfs))), dmf(lfs)); // L2181
    pratec(i, j) = __fdiv_rn(__fmul_rn(pptflx, __fsub_rn(1.f, fbfrc)), dxsq); // L2182
    raincv(i, j) = __fmul_rn(dt, pratec(i, j)); // L2183
    rnc = __fmul_rn(raincv(i, j), nic); // L2186
    if (((ishall == 0) && iprnt)) { ; /* Native diagnostic I/O omitted; no physics mutation. */ } // L2187
    qinit = 0.f; // L2194
    qfnl = 0.f; // L2195
    dpt = 0.f; // L2196
    for (nk = 1; nk <= ltop; nk += 1) { loop54_cycle: ; // L2197
    dpt = __fadd_rn(dpt, dp(nk)); // L2198
    qinit = __fadd_rn(qinit, __fmul_rn(q0(nk), ems(nk))); // L2199
    qfnl = __fadd_rn(qfnl, __fmul_rn(qg(nk), ems(nk))); // L2200
    qfnl = __fadd_rn(qfnl, __fmul_rn(__fadd_rn(__fadd_rn(__fadd_rn(qlg(nk), qig(nk)), qrg(nk)), qsg(nk)), ems(nk))); // L2201
    loop54_continue: ;
    } loop54_exit: ; // L2202
    qfnl = __fadd_rn(qfnl, __fmul_rn(__fmul_rn(pptflx, timec), __fsub_rn(1.f, fbfrc))); // L2203
    err2 = __fdiv_rn(__fmul_rn(__fsub_rn(qfnl, qinit), 100.f), qinit); // L2205
    if (iprnt) { ; /* Native diagnostic I/O omitted; no physics mutation. */ } // L2206
    if (((fabsf(err2) > 0.05f) && (istop == 0))) { // L2207
    iprnt = true; // L2210
    istop = 1; // L2211
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2212
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2213
    for (nk = 1; nk <= kl; nk += 1) { loop55_cycle: ; // L2214
    k = ((kl - nk) + 1); // L2215
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2221
    loop55_continue: ;
    } loop55_exit: ; // L2223
    } // L2228
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2229
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2230
    if ((pptflx > 0.f)) { // L2231
    relerr = __fdiv_rn(__fmul_rn(err2, qinit), __fmul_rn(pptflx, timec)); // L2232
    } else { // L2233
    relerr = 0.f; // L2234
    } // L2235
    if (iprnt) { // L2236
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2237
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2238
    } // L2240
    if ((tadvec < timec)) { nic = int(roundf(__fdiv_rn(tadvec, dt))); } // L2247
    nca(i, j) = __fmul_rn(float(nic), dt); // L2248
    if ((ishall == 1)) { // L2249
    timec = 2400.f; // L2250
    nca(i, j) = __fmul_rn(cudt, 60.f); // L2251
    nshall = (nshall + 1); // L2252
    } // L2253
    for (k = 1; k <= kx; k += 1) { loop56_cycle: ; // L2255
    if (((!f_qi) && warm_rain)) { // L2277
    cpm = __fmul_rn(cp, __fadd_rn(1.f, __fmul_rn(0.887f, qg(k)))); // L2279
    tg(k) = __fsub_rn(tg(k), __fdiv_rn(__fmul_rn(__fadd_rn(qig(k), qsg(k)), rlf), cpm)); // L2280
    dqcdt(k) = __fdiv_rn(__fsub_rn(__fsub_rn(__fadd_rn(qlg(k), qig(k)), ql0(k)), qi0(k)), timec); // L2281
    dqidt(k) = 0.f; // L2282
    dqrdt(k) = __fdiv_rn(__fsub_rn(__fsub_rn(__fadd_rn(qrg(k), qsg(k)), qr0(k)), qs0(k)), timec); // L2283
    dqsdt(k) = 0.f; // L2284
    } else if (((!f_qi) && (!warm_rain))) { // L2285
    cpm = __fmul_rn(cp, __fadd_rn(1.f, __fmul_rn(0.887f, qg(k)))); // L2290
    if ((k <= ml)) { // L2291
    tg(k) = __fsub_rn(tg(k), __fdiv_rn(__fmul_rn(__fadd_rn(qig(k), qsg(k)), rlf), cpm)); // L2292
    } else if ((k > ml)) { // L2293
    tg(k) = __fadd_rn(tg(k), __fdiv_rn(__fmul_rn(__fadd_rn(qlg(k), qrg(k)), rlf), cpm)); // L2294
    } // L2295
    dqcdt(k) = __fdiv_rn(__fsub_rn(__fsub_rn(__fadd_rn(qlg(k), qig(k)), ql0(k)), qi0(k)), timec); // L2296
    dqidt(k) = 0.f; // L2297
    dqrdt(k) = __fdiv_rn(__fsub_rn(__fsub_rn(__fadd_rn(qrg(k), qsg(k)), qr0(k)), qs0(k)), timec); // L2298
    dqsdt(k) = 0.f; // L2299
    } else if (f_qi) { // L2300
    dqcdt(k) = __fdiv_rn(__fsub_rn(qlg(k), ql0(k)), timec); // L2305
    dqidt(k) = __fdiv_rn(__fsub_rn(qig(k), qi0(k)), timec); // L2306
    dqrdt(k) = __fdiv_rn(__fsub_rn(qrg(k), qr0(k)), timec); // L2307
    if (f_qs) { // L2308
    dqsdt(k) = __fdiv_rn(__fsub_rn(qsg(k), qs0(k)), timec); // L2309
    } else { // L2310
    dqidt(k) = __fadd_rn(dqidt(k), __fdiv_rn(__fsub_rn(qsg(k), qs0(k)), timec)); // L2311
    } // L2312
    } else { // L2313
    } // L2316
    dtdt(k) = __fdiv_rn(__fsub_rn(tg(k), t0(k)), timec); // L2317
    dqdt(k) = __fdiv_rn(__fsub_rn(qg(k), q0(k)), timec); // L2318
    loop56_continue: ;
    } loop56_exit: ; // L2319
    pratec(i, j) = __fdiv_rn(__fmul_rn(pptflx, __fsub_rn(1.f, fbfrc)), dxsq); // L2320
    raincv(i, j) = __fmul_rn(dt, pratec(i, j)); // L2321
    rnc = __fmul_rn(raincv(i, j), nic); // L2324
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2325
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2331
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2332
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2333
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2334
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2335
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2339
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2342
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2348
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2349
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2350
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2352
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2353
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2354
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2355
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2357
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2360
    cutop(i, j) = float(ltop); // L2366
    cubot(i, j) = float(lcl); // L2367
}

// MPAS-A v8.4.2 module_cu_kfeta.F:19-361
KFDEV void kf_eta_cps(Tables &tb, Arena &arena, int  ids, int  ide, int  jds, int  jde, int  kds, int  kde, int  ims, int  ime, int  jms, int  jme, int  kms, int  kme, int  its, int  ite, int  jts, int  jte, int  kts, int  kte, float  dt, int  ktau, FArray<float> dxcell, FArray<float> areacell, float  cudt, float  curr_secs, bool  adapt_step_flag, FArray<float> rho, FArray<float> raincv, FArray<float> pratec, FArray<float> nca, FArray<float> u, FArray<float> v, FArray<float> th, FArray<float> t, FArray<float> w, FArray<float> dz8w, FArray<float> pcps, FArray<float> pi, FArray<float> w0avg, float  xlv0, float  xlv1, float  xls0, float  xls1, float  cp, float  r, float  g, float  ep1, float  ep2, float  svp1, float  svp2, float  svp3, float  svpt0, int  stepcu, FArray<bool> cu_act_flag, bool  warm_rain, FArray<float> cutop, FArray<float> cubot, FArray<float> qv, bool & f_qr, bool & f_qi, bool & f_qs, FArray<float> rthcuten, FArray<float> rqvcuten, FArray<float> rqccuten, FArray<float> rqrcuten, FArray<float> rqicuten, FArray<float> rqscuten, bool has_f_qr, bool has_f_qi, bool has_f_qs, bool has_rthcuten, bool has_rqvcuten, bool has_rqccuten, bool has_rqrcuten, bool has_rqicuten, bool has_rqscuten) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    bool flag_qr;
    bool flag_qi;
    bool flag_qs;
    FArray<float> u1d = arena.alloc<float>(kts, kte);
    FArray<float> v1d = arena.alloc<float>(kts, kte);
    FArray<float> t1d = arena.alloc<float>(kts, kte);
    FArray<float> dz1d = arena.alloc<float>(kts, kte);
    FArray<float> qv1d = arena.alloc<float>(kts, kte);
    FArray<float> p1d = arena.alloc<float>(kts, kte);
    FArray<float> rho1d = arena.alloc<float>(kts, kte);
    FArray<float> w0avg1d = arena.alloc<float>(kts, kte);
    FArray<float> dqdt = arena.alloc<float>(kts, kte);
    FArray<float> dqidt = arena.alloc<float>(kts, kte);
    FArray<float> dqcdt = arena.alloc<float>(kts, kte);
    FArray<float> dqrdt = arena.alloc<float>(kts, kte);
    FArray<float> dqsdt = arena.alloc<float>(kts, kte);
    FArray<float> dtdt = arena.alloc<float>(kts, kte);
    float tst;
    float tv;
    float prs;
    float rhoe;
    float w0;
    float scr1;
    float dx;
    float dxsq;
    float tmp;
    int i;
    int j;
    int k;
    int ntst;
    float lastdt = (-1.0f);
    float w0avgfctr;
    float w0fctr;
    float w0den;
    bool run_param;
    ntst = stepcu; // L186
    tst = float((ntst * 2)); // L187
    flag_qr = false; // L188
    flag_qi = false; // L189
    flag_qs = false; // L190
    if (has_f_qr) { flag_qr = f_qr; } // L191
    if (has_f_qi) { flag_qi = f_qi; } // L192
    if (has_f_qs) { flag_qs = f_qs; } // L193
    if ((lastdt < 0)) { // L195
    lastdt = dt; // L196
    } // L197
    if (adapt_step_flag) { // L199
    w0avgfctr = __fsub_rn(__fmul_rn(2, kf_max<float>(__fmul_rn(cudt, 60), dt)), dt); // L200
    w0fctr = dt; // L201
    w0den = __fmul_rn(2, kf_max<float>(__fmul_rn(cudt, 60), dt)); // L202
    } else { // L203
    w0avgfctr = __fsub_rn(tst, 1.f); // L204
    w0fctr = 1.f; // L205
    w0den = tst; // L206
    } // L207
    for (j = jts; j <= jte; j += 1) { loop1_cycle: ; // L209
    for (k = kts; k <= kte; k += 1) { loop2_cycle: ; // L210
    for (i = its; i <= ite; i += 1) { loop3_cycle: ; // L211
    w0 = __fmul_rn(0.5f, __fadd_rn(w(i, k, j), w(i, (k + 1), j))); // L216
    w0avg(i, k, j) = __fdiv_rn(__fadd_rn(__fmul_rn(w0avg(i, k, j), w0avgfctr), __fmul_rn(w0, w0fctr)), w0den); // L224
    loop3_continue: ;
    } loop3_exit: ; // L225
    loop2_continue: ;
    } loop2_exit: ; // L226
    loop1_continue: ;
    } loop1_exit: ; // L227
    lastdt = dt; // L228
    if (adapt_step_flag) { // L239
    if ((((ktau == 1) || (cudt == 0)) || (__fadd_rn(curr_secs, dt) >= __fmul_rn(__fmul_rn((int(__fdiv_rn(curr_secs, __fmul_rn(cudt, 60))) + 1), cudt), 60)))) { // L240
    run_param = true; // L243
    } else { // L244
    run_param = false; // L245
    } // L246
    } else { // L248
    if ((((ktau % ntst) == 0) || (ktau == 1))) { // L249
    run_param = true; // L250
    } else { // L251
    run_param = false; // L252
    } // L253
    } // L254
    if (run_param) { // L256
    for (j = jts; j <= jte; j += 1) { loop4_cycle: ; // L258
    for (i = its; i <= ite; i += 1) { loop5_cycle: ; // L259
    cu_act_flag(i, j) = true; // L260
    loop5_continue: ;
    } loop5_exit: ; // L261
    loop4_continue: ;
    } loop4_exit: ; // L262
    for (j = jts; j <= jte; j += 1) { loop6_cycle: ; // L264
    for (i = its; i <= ite; i += 1) { loop7_cycle: ; // L265
    if ((nca(i, j) >= __fmul_rn(0.5f, dt))) { // L268
    cu_act_flag(i, j) = false; // L269
    } else { // L270
    dxsq = areacell(i, j); // L273
    dx = dxcell(i, j); // L274
    for (k = kts; k <= kte; k += 1) { loop8_cycle: ; // L278
    dqdt(k) = 0.f; // L279
    dqidt(k) = 0.f; // L280
    dqcdt(k) = 0.f; // L281
    dqrdt(k) = 0.f; // L282
    dqsdt(k) = 0.f; // L283
    dtdt(k) = 0.f; // L284
    loop8_continue: ;
    } loop8_exit: ; // L285
    raincv(i, j) = 0.f; // L286
    cutop(i, j) = kts; // L287
    cubot(i, j) = (kte + 1); // L288
    pratec(i, j) = 0.f; // L289
    for (k = kts; k <= kte; k += 1) { loop9_cycle: ; // L293
    u1d(k) = u(i, k, j); // L294
    v1d(k) = v(i, k, j); // L295
    t1d(k) = t(i, k, j); // L296
    rho1d(k) = rho(i, k, j); // L297
    qv1d(k) = qv(i, k, j); // L298
    p1d(k) = pcps(i, k, j); // L299
    w0avg1d(k) = w0avg(i, k, j); // L300
    dz1d(k) = dz8w(i, k, j); // L301
    loop9_continue: ;
    } loop9_exit: ; // L302
    kf_eta_para(tb, arena, i, j, u1d, v1d, t1d, qv1d, p1d, dz1d, w0avg1d, dt, dx, dxsq, rho1d, xlv0, xlv1, xls0, xls1, cp, r, g, ep2, svp1, svp2, svp3, svpt0, dqdt, dqidt, dqcdt, dqrdt, dqsdt, dtdt, raincv, pratec, nca, flag_qi, flag_qs, warm_rain, cutop, cubot, cudt, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte); // L303
    if ((has_rthcuten && has_rqvcuten)) { // L315
    for (k = kts; k <= kte; k += 1) { loop10_cycle: ; // L316
    rthcuten(i, k, j) = __fdiv_rn(dtdt(k), pi(i, k, j)); // L317
    rqvcuten(i, k, j) = dqdt(k); // L318
    loop10_continue: ;
    } loop10_exit: ; // L319
    } // L320
    if ((has_rqrcuten && has_rqccuten)) { // L322
    if (f_qr) { // L323
    for (k = kts; k <= kte; k += 1) { loop11_cycle: ; // L324
    rqrcuten(i, k, j) = dqrdt(k); // L325
    rqccuten(i, k, j) = dqcdt(k); // L326
    loop11_continue: ;
    } loop11_exit: ; // L327
    } else { // L328
    for (k = kts; k <= kte; k += 1) { loop12_cycle: ; // L330
    rqrcuten(i, k, j) = 0.f; // L331
    rqccuten(i, k, j) = __fadd_rn(dqrdt(k), dqcdt(k)); // L332
    loop12_continue: ;
    } loop12_exit: ; // L333
    } // L334
    } // L335
    if (has_rqicuten) { // L340
    if (f_qi) { // L341
    for (k = kts; k <= kte; k += 1) { loop13_cycle: ; // L342
    rqicuten(i, k, j) = dqidt(k); // L343
    loop13_continue: ;
    } loop13_exit: ; // L344
    } // L345
    } // L346
    if (has_rqscuten) { // L348
    if (f_qs) { // L349
    for (k = kts; k <= kte; k += 1) { loop14_cycle: ; // L350
    rqscuten(i, k, j) = dqsdt(k); // L351
    loop14_continue: ;
    } loop14_exit: ; // L352
    } // L353
    } // L354
    } // L356
    loop7_continue: ;
    } loop7_exit: ; // L357
    loop6_continue: ;
    } loop6_exit: ; // L358
    } // L359
}

// MPAS-A v8.4.2 module_cu_kfeta.F:2765-2851
KFDEV void kf_eta_init(Tables &tb, Arena &arena, FArray<float> rthcuten, FArray<float> rqvcuten, FArray<float> rqccuten, FArray<float> rqrcuten, FArray<float> rqicuten, FArray<float> rqscuten, FArray<float> nca, FArray<float> w0avg, int  p_qi, int  p_qs, float  svp1, float  svp2, float  svp3, float  svpt0, int  p_first_scalar, bool  restart, bool  allowed_to_read, int  ids, int  ide, int  jds, int  jde, int  kds, int  kde, int  ims, int  ime, int  jms, int  jme, int  kms, int  kme, int  its, int  ite, int  jts, int  jte, int  kts, int  kte) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    int i;
    int j;
    int k;
    int itf;
    int jtf;
    int ktf;
    jtf = kf_min<int>(jte, (jde - 1)); // L2796
    ktf = kf_min<int>(kte, (kde - 1)); // L2797
    itf = kf_min<int>(ite, (ide - 1)); // L2798
    if ((!restart)) { // L2800
    for (j = jts; j <= jtf; j += 1) { loop1_cycle: ; // L2802
    for (k = kts; k <= ktf; k += 1) { loop2_cycle: ; // L2803
    for (i = its; i <= itf; i += 1) { loop3_cycle: ; // L2804
    rthcuten(i, k, j) = 0.f; // L2805
    rqvcuten(i, k, j) = 0.f; // L2806
    rqccuten(i, k, j) = 0.f; // L2807
    rqrcuten(i, k, j) = 0.f; // L2808
    loop3_continue: ;
    } loop3_exit: ; // L2809
    loop2_continue: ;
    } loop2_exit: ; // L2810
    loop1_continue: ;
    } loop1_exit: ; // L2811
    if ((p_qi >= p_first_scalar)) { // L2813
    for (j = jts; j <= jtf; j += 1) { loop4_cycle: ; // L2814
    for (k = kts; k <= ktf; k += 1) { loop5_cycle: ; // L2815
    for (i = its; i <= itf; i += 1) { loop6_cycle: ; // L2816
    rqicuten(i, k, j) = 0.f; // L2817
    loop6_continue: ;
    } loop6_exit: ; // L2818
    loop5_continue: ;
    } loop5_exit: ; // L2819
    loop4_continue: ;
    } loop4_exit: ; // L2820
    } // L2821
    if ((p_qs >= p_first_scalar)) { // L2823
    for (j = jts; j <= jtf; j += 1) { loop7_cycle: ; // L2824
    for (k = kts; k <= ktf; k += 1) { loop8_cycle: ; // L2825
    for (i = its; i <= itf; i += 1) { loop9_cycle: ; // L2826
    rqscuten(i, k, j) = 0.f; // L2827
    loop9_continue: ;
    } loop9_exit: ; // L2828
    loop8_continue: ;
    } loop8_exit: ; // L2829
    loop7_continue: ;
    } loop7_exit: ; // L2830
    } // L2831
    for (j = jts; j <= jtf; j += 1) { loop10_cycle: ; // L2833
    for (i = its; i <= itf; i += 1) { loop11_cycle: ; // L2834
    nca(i, j) = (-100.f); // L2835
    loop11_continue: ;
    } loop11_exit: ; // L2836
    loop10_continue: ;
    } loop10_exit: ; // L2837
    for (j = jts; j <= jtf; j += 1) { loop12_cycle: ; // L2839
    for (k = kts; k <= ktf; k += 1) { loop13_cycle: ; // L2840
    for (i = its; i <= itf; i += 1) { loop14_cycle: ; // L2841
    w0avg(i, k, j) = 0.f; // L2842
    loop14_continue: ;
    } loop14_exit: ; // L2843
    loop13_continue: ;
    } loop13_exit: ; // L2844
    loop12_continue: ;
    } loop12_exit: ; // L2845
    } // L2847
    kf_lutab(tb, arena, svp1, svp2, svp3, svpt0); // L2849
}
KFDEV void kf_batch_column(int nz, int nc, int c, float *fields, float *state, float *surface,
        unsigned char *active, float *params, int *flags, float *table, float *work) {
    Tables tb(table); Arena arena{work+c,0,nc};
    bool adapt_step_flag=flags[2], warm_rain=flags[3], f_qr=flags[4], f_qi=flags[5], f_qs=flags[6];
    FArray<bool> cu_act_flag{reinterpret_cast<bool*>(active+c),1,1,1,0,0,0};

    FArray<float> u{fields+0*(nz+1)*nc+c,1,1,1,0,nc,0};
    FArray<float> v{fields+1*(nz+1)*nc+c,1,1,1,0,nc,0};
    FArray<float> th{fields+2*(nz+1)*nc+c,1,1,1,0,nc,0};
    FArray<float> t{fields+3*(nz+1)*nc+c,1,1,1,0,nc,0};
    FArray<float> w{fields+4*(nz+1)*nc+c,1,1,1,0,nc,0};
    FArray<float> dz8w{fields+5*(nz+1)*nc+c,1,1,1,0,nc,0};
    FArray<float> pcps{fields+6*(nz+1)*nc+c,1,1,1,0,nc,0};
    FArray<float> pi{fields+7*(nz+1)*nc+c,1,1,1,0,nc,0};
    FArray<float> rho{fields+8*(nz+1)*nc+c,1,1,1,0,nc,0};
    FArray<float> qv{fields+9*(nz+1)*nc+c,1,1,1,0,nc,0};
    FArray<float> w0avg{state+0*(nz+1)*nc+c,1,1,1,0,nc,0};
    FArray<float> rthcuten{state+1*(nz+1)*nc+c,1,1,1,0,nc,0};
    FArray<float> rqvcuten{state+2*(nz+1)*nc+c,1,1,1,0,nc,0};
    FArray<float> rqccuten{state+3*(nz+1)*nc+c,1,1,1,0,nc,0};
    FArray<float> rqrcuten{state+4*(nz+1)*nc+c,1,1,1,0,nc,0};
    FArray<float> rqicuten{state+5*(nz+1)*nc+c,1,1,1,0,nc,0};
    FArray<float> rqscuten{state+6*(nz+1)*nc+c,1,1,1,0,nc,0};
    FArray<float> dxcell{surface+0*nc+c,1,1,1,0,0,0};
    FArray<float> areacell{surface+1*nc+c,1,1,1,0,0,0};
    FArray<float> raincv{surface+2*nc+c,1,1,1,0,0,0};
    FArray<float> pratec{surface+3*nc+c,1,1,1,0,0,0};
    FArray<float> nca{surface+4*nc+c,1,1,1,0,0,0};
    FArray<float> cutop{surface+5*nc+c,1,1,1,0,0,0};
    FArray<float> cubot{surface+6*nc+c,1,1,1,0,0,0};
    kf_eta_cps(tb, arena, 1, 1, 1, 1, 1, nz+1, 1, 1, 1, 1, 1, nz+1, 1, 1, 1, 1, 1, nz, params[0], flags[1], dxcell, areacell, params[1], params[2], adapt_step_flag, rho, raincv, pratec, nca, u, v, th, t, w, dz8w, pcps, pi, w0avg, params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], flags[0], cu_act_flag, warm_rain, cutop, cubot, qv, f_qr, f_qi, f_qs, rthcuten, rqvcuten, rqccuten, rqrcuten, rqicuten, rqscuten, true, true, true, true, true, true, true, true, true);
}

#ifndef KF_HOST_TEST
// MPAS-A v8.4.2 module_cu_kfeta.F:2855-2982; device initialization entry.
extern "C" __global__ void kf_tables_kernel(float *table, float *params) {
    if (blockIdx.x || threadIdx.x) return;
    Tables tb(table); Arena a{nullptr,0,1};
    kf_lutab(tb,a,params[12],params[13],params[14],params[15]);
}
// MPAS-A v8.4.2 module_cu_kfeta.F:19-361; device CPS entry.
extern "C" __global__ void kf_cps_kernel(int nz, int nc, float *fields, float *state,
        float *surface, unsigned char *active, float *params, int *flags, float *table, float *work) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<nc) kf_batch_column(nz,nc,c,fields,state,surface,active,params,flags,table,work);
}
#else
extern "C" void kf_host_tables(float *table,float *params) {
    Tables tb(table); Arena a{nullptr,0,1};
    kf_lutab(tb,a,params[12],params[13],params[14],params[15]);
}
extern "C" void kf_host_column(int nz,int nc,float *fields,float *state,float *surface,
        unsigned char *active,float *params,int *flags,float *table,float *work) {
    for(int c=0;c<nc;++c) kf_batch_column(nz,nc,c,fields,state,surface,active,params,flags,table,work);
}
#endif


#ifndef KF_HOST_TEST
// MPAS-A v8.4.2 mpas_atmphys_driver_convection.F:666-687,752-790.
// Public MPAS pool arrays are cells-fast [level,cell]; packed CPS uses padding.
extern "C" __global__ void kf_pack_kernel(int nz,int nc,const unsigned long long *fp,
    const unsigned long long *sp,const unsigned long long *vp,const float *density,
    float len_disp,float *fields,float *state,float *surface) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int c=idx%nc,k=idx/nc;
    if(k>nz)return;
    for(int f=0;f<10;++f) {
        const float *src=reinterpret_cast<const float*>(fp[f]);
        fields[(f*(nz+1)+k)*nc+c]=(k<nz || f==4)?src[idx]:0.0f;
    }
    for(int f=0;f<7;++f) {
        const float *src=reinterpret_cast<const float*>(sp[f]);
        state[(f*(nz+1)+k)*nc+c]=k<nz?src[idx]:0.0f;
    }
    if(k==0) {
        surface[c]=__fdiv_rn(len_disp,gfk_pow(density[c],0.25f));
        for(int f=1;f<7;++f) surface[f*nc+c]=reinterpret_cast<const float*>(vp[f])[c];
    }
}
// MPAS-A v8.4.2 mpas_atmphys_driver_convection.F:892-912,949-969.
extern "C" __global__ void kf_unpack_kernel(int nz,int nc,const unsigned long long *sp,
    const unsigned long long *vp,const float *state,const float *surface) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int c=idx%nc,k=idx/nc;
    if(k>=nz)return;
    for(int f=0;f<7;++f) reinterpret_cast<float*>(sp[f])[idx]=state[(f*(nz+1)+k)*nc+c];
    if(k==0) for(int f=2;f<7;++f) reinterpret_cast<float*>(vp[f])[c]=surface[f*nc+c];
}
// MPAS-A v8.4.2 module_cu_kfeta.F:2795-2847; LUTAB runs separately once.
extern "C" __global__ void kf_initialize_kernel(int nz,int nc,float *state,float *surface,
    bool restart,bool f_qi,bool f_qs) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int c=idx%nc,k=idx/nc;
    if(k>=nz || restart)return;
    for(int f=0;f<5;++f) state[(f*(nz+1)+k)*nc+c]=0.0f;
    if(f_qi)state[(5*(nz+1)+k)*nc+c]=0.0f;
    if(f_qs)state[(6*(nz+1)+k)*nc+c]=0.0f;
    if(k==0)surface[4*nc+c]=-100.0f;
}
// MPAS-A v8.4.2 mpas_atmphys_driver_convection.F:1030-1054.
extern "C" __global__ void kf_step1_kernel(int nz,int nc,float dt,float *state,float *surface) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=nc)return;
    float &nca=surface[4*nc+c];
    if(nca>0.0f) {
        nca=__fsub_rn(nca,dt);
        if(nca<__fmul_rn(0.5f,dt)) {
            for(int k=0;k<nz;++k)for(int f=1;f<7;++f)state[(f*(nz+1)+k)*nc+c]=0.0f;
            surface[2*nc+c]=0.0f;surface[3*nc+c]=0.0f;
            surface[6*nc+c]=float(nz+1);surface[5*nc+c]=1.0f;
        }
    }
}
// MPAS-A v8.4.2 mpas_atmphys_driver_convection.F:1088-1099.
extern "C" __global__ void kf_step2_kernel(int nc,float dt,float bucket,bool alarm,
    const float *cuprec,float *rainc,int *i_rainc) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=nc)return;
    rainc[c]=__fadd_rn(rainc[c],__fmul_rn(cuprec[c],dt));
    if(alarm && bucket>0.0f && rainc[c]>bucket) {
        i_rainc[c]+=1;rainc[c]=__fsub_rn(rainc[c],bucket);
    }
}
// MPAS-A v8.4.2 mpas_atmphys_driver_convection.F:464-473,516-522.
extern "C" __global__ void kf_driver_flags_kernel(float dt_dyn,float dt_cu,int step,int n_cu,
    float *params,int *flags,unsigned char *active,int nc) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<nc)active[c]=0;
    if(c==0) {
        params[0]=dt_dyn;params[1]=__fdiv_rn(dt_cu,60.0f);params[2]=-1.0f;
        flags[0]=n_cu;flags[1]=step==1?step:step+1;flags[2]=0;
    }
}
#endif

'''

def shared_math_source() -> str:
    """Math binding for module_cu_kfeta.F:364-2370,2855-2982.

    The contracted expf matches the host's FMA-dispatched glibc implementation.
    These are shared headers; no elementary function is copied or rederived here.
    """
    root = files("woof.core.kernels")
    return "\n".join(root.joinpath(name).read_text(encoding="utf-8") for name in
                     ("glibc_flt32.cuh", "flt32_expf_fma.cuh"))


def source_digest() -> str:
    """Identify the translation of module_cu_kfeta.F:19-2982 and shared math."""
    return hashlib.sha256((shared_math_source() + CUDA_SOURCE).encode()).hexdigest()


def cuda_image(path, source, architecture):
    """Authenticate a compiled image of module_cu_kfeta.F:19-2982.

    A CPU-only host can compile the same complete source through its build
    queue without occupying the GPU. The image retains every strict compiler option.
    """
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    expected = hashlib.sha256(source.encode()).hexdigest()
    if (manifest.get("schema") != "mpas-kf-cuda-image/v1"
            or manifest.get("source_sha256") != expected
            or manifest.get("options") != list(_hex_numerics.options(CUDA_OPTIONS))
            or manifest.get("architecture") != str(architecture)
            or manifest.get("format") != "cubin"):
        raise ValueError("KF CUDA image source, strict options or architecture differs from the selected port")
    binary = (path.parent / manifest["image"]).resolve()
    if not binary.is_relative_to(path.parent):
        raise ValueError("KF CUDA image path escapes its manifest directory")
    image = binary.read_bytes()
    if hashlib.sha256(image).hexdigest() != manifest["image_sha256"]:
        raise ValueError("KF CUDA image bytes differ from the compiled artifact")
    return image


@lru_cache(maxsize=None)
def _module(device_id: int):
    """Compile the device backend for module_cu_kfeta.F:19-2982."""
    import cupy as cp
    from cupy.cuda import compiler
    with cp.cuda.Device(device_id):
        # CuPy's RawModule cache compiler appends -ftz=true after caller
        # options. That loses native subnormal ice tendencies (the strict
        # standalone witness has 7 differing words). Invoke the same NVRTC
        # compiler directly so --ftz=false reaches it without that override.
        source = shared_math_source() + CUDA_SOURCE
        architecture = _compilation_target(device_id, xp=cp)
        image_path = os.environ.get("MPAS_KF_CUDA_IMAGE")
        if image_path:
            binary = cuda_image(image_path, source, architecture)
        else:
            binary, _ = compiler.compile_using_nvrtc(
                source, options=_hex_numerics.options(CUDA_OPTIONS), arch=architecture)
        module = cp.cuda.function.Module()
        module.load(binary)
        return module


def _function(name):
    """Resolve a translated entry from module_cu_kfeta.F:19-2982."""
    import cupy as cp
    return _module(cp.cuda.Device().id).get_function(name)


def _device_buffer(array, shape, dtype, name):
    """Check storage for the explicit-shape CPS ABI, module_cu_kfeta.F:19-110."""
    import cupy as cp
    if not isinstance(array, cp.ndarray) or array.dtype != dtype or array.shape != shape or not array.flags.c_contiguous:
        raise ValueError(f"{name} requires a contiguous CUDA {dtype} array of shape {shape}")


def _packed_state(state, surface):
    """Validate CPS state extents; module_cu_kfeta.F:19-110."""
    if state.ndim != 3 or state.shape[1] < 2:
        raise ValueError("state requires (field, nz+1, cell) with at least one mass level")
    _, padded, nc = state.shape
    _device_buffer(state, (len(STATE), padded, nc), np.dtype("float32"), "state")
    _device_buffer(surface, (len(SURFACE), nc), np.dtype("float32"), "surface")
    return padded - 1, nc


def build_tables(params):
    """MPAS-A v8.4.2 module_cu_kfeta.F:2855-2982; no host table path."""
    import cupy as cp
    _device_buffer(params, (len(PARAMETERS),), np.dtype("float32"), "params")
    tables = cp.empty(TABLE_WORDS, dtype=cp.float32)
    _function("kf_tables_kernel")((1,), (1,), (tables, params))
    return tables


def run_packed(fields, state, surface, active, params, flags, *, tables=None, work=None):
    """MPAS-A v8.4.2 module_cu_kfeta.F:19-361,364-2370.

    Execute CPS in place. All buffers, including configuration scalars, are
    CUDA arrays; fields/state use (field, nz+1, cell). Tables are reusable only
    while the four SVP constants remain unchanged. Caller owns stream ordering.
    """
    import cupy as cp
    if fields.ndim != 3 or fields.shape[1] < 2:
        raise ValueError("fields requires (field, level, cell) dimensions")
    _, padded, nc = fields.shape
    nz = padded - 1
    _device_buffer(fields, (len(FIELDS), padded, nc), np.dtype("float32"), "fields")
    _device_buffer(state, (len(STATE), padded, nc), np.dtype("float32"), "state")
    _device_buffer(surface, (len(SURFACE), nc), np.dtype("float32"), "surface")
    _device_buffer(active, (nc,), np.dtype("uint8"), "active")
    _device_buffer(params, (len(PARAMETERS),), np.dtype("float32"), "params")
    _device_buffer(flags, (len(FLAGS),), np.dtype("int32"), "flags")
    if nc == 0:
        return tables
    if tables is None:
        tables = build_tables(params)
    _device_buffer(tables, (TABLE_WORDS,), np.dtype("float32"), "tables")
    shape = (WORKSPACE_ARRAYS * nz + 1, nc)
    if work is None:
        work = cp.empty(shape, dtype=cp.float32)
    _device_buffer(work, shape, np.dtype("float32"), "work")
    _function("kf_cps_kernel")(((nc + 63) // 64,), (64,),
        (np.int32(nz), np.int32(nc), fields, state, surface, active, params, flags, tables, work))
    return tables


_PREPARED_FIELDS = ("u_p", "v_p", "th_p", "t_p", "w_p", "dz_p", "pres_hyd_p", "pi_p", "rho_p", "qv_p")
_SURFACE_POOL = (None, "areaCell", "raincv", "cuprec", "nca", "cutop", "cubot")


def convection_from_MPAS(prepared, mesh, diag_physics, tend_physics, len_disp):
    """MPAS-A v8.4.2 mpas_atmphys_driver_convection.F:666-687,752-790.

    Only the convection slice packs here. Generalized preparation must already
    have produced the named *_p fields; no preparation formula is duplicated.
    Caller passes solve-cell views, excluding halos/dummy cells.
    """
    import cupy as cp
    # Native globals include the extra memory level: mpas_atmphys_interface.F:103-113;
    # mpas_atmphys_manager.F:655-663 sets kte=kme-1. The pool holds the solve-level extent.
    # Slice device views only; leave the unused native top storage untouched.
    nz, nc = diag_physics["w0avg"].shape
    fa = [prepared[name][:nz + (name == "w_p")] for name in _PREPARED_FIELDS]
    sa = [diag_physics["w0avg"]] + [tend_physics[name] for name in STATE[1:]]
    va = [mesh["meshDensity"], mesh["areaCell"]] + [diag_physics[name] for name in _SURFACE_POOL[2:]]
    for name, array in zip(_PREPARED_FIELDS, fa):
        _device_buffer(array, (nz + (name == "w_p"), nc), np.dtype("float32"), name)
    for name, array in zip(STATE, sa):
        _device_buffer(array, (nz, nc), np.dtype("float32"), name)
    for name, array in zip(("meshDensity",) + _SURFACE_POOL[1:], va):
        _device_buffer(array, (nc,), np.dtype("float32"), name)
    # Pointer metadata is orchestration; all field copies run in the RawKernel.
    fp, sp, vp = [cp.asarray([a.data.ptr for a in arrays], dtype=cp.uint64) for arrays in (fa, sa, va)]
    fields = cp.empty((len(FIELDS), nz + 1, nc), cp.float32)
    state = cp.empty((len(STATE), nz + 1, nc), cp.float32)
    surface = cp.empty((len(SURFACE), nc), cp.float32)
    if nc:
        count = (nz + 1) * nc
        _function("kf_pack_kernel")(((count + 255) // 256,), (256,),
            (np.int32(nz), np.int32(nc), fp, sp, vp, va[0], np.float32(len_disp), fields, state, surface))
    return fields, state, surface, (fp, sp, vp, fa, sa, va)


def convection_to_MPAS(state, surface, packed_pointers):
    """MPAS-A v8.4.2 mpas_atmphys_driver_convection.F:892-912,949-969."""
    nz, nc = _packed_state(state, surface)
    _, sp, vp, _, _, _ = packed_pointers
    if nc:
        _function("kf_unpack_kernel")(((nz * nc + 255) // 256,), (256,),
            (np.int32(nz), np.int32(nc), sp, vp, state, surface))


def initialize_packed(state, surface, *, restart, f_qi, f_qs):
    """MPAS-A v8.4.2 module_cu_kfeta.F:2765-2851; LUTAB is build_tables."""
    nz, nc = _packed_state(state, surface)
    if nc:
        _function("kf_initialize_kernel")(((nz * nc + 255) // 256,), (256,),
            (np.int32(nz), np.int32(nc), state, surface, np.bool_(restart), np.bool_(f_qi), np.bool_(f_qs)))


def update_convection_step1(state, surface, dt_dyn):
    """MPAS-A v8.4.2 mpas_atmphys_driver_convection.F:1016-1054.

    f4 owns invocation before the slot; the slot callback does not repeat it.
    """
    nz, nc = _packed_state(state, surface)
    if nc:
        _function("kf_step1_kernel")(((nc + 63) // 64,), (64,),
            (np.int32(nz), np.int32(nc), np.float32(dt_dyn), state, surface))


def update_convection_step2(cuprec, rainc, i_rainc, dt_dyn, bucket_rainc, l_acrain):
    """MPAS-A v8.4.2 mpas_atmphys_driver_convection.F:1061-1102.

    Native subtracts one bucket only and uses a strict greater-than threshold.
    f4 owns invocation every dynamics step after the slot.
    """
    nc = cuprec.size
    _device_buffer(cuprec, (nc,), np.dtype("float32"), "cuprec")
    _device_buffer(rainc, (nc,), np.dtype("float32"), "rainc")
    _device_buffer(i_rainc, (nc,), np.dtype("int32"), "i_rainc")
    if nc:
        _function("kf_step2_kernel")(((nc + 63) // 64,), (64,),
            (np.int32(nc), np.float32(dt_dyn), np.float32(bucket_rainc), np.bool_(l_acrain), cuprec, rainc, i_rainc))


_NATIVE_CONSTANTS = ("xlv0", "xlv1", "xls0", "xls1", "cp", "r_d", "gravity",
                     "ep_1", "ep_2", "svp1", "svp2", "svp3", "svpt0")


#: An install-built packet (``woof hex physics-resources build``) marks its warm_rain word with this.
#: Native KF reads the driver-local warm_rain at two statements only, and both test ``.NOT. F_QI``
#: first (module_cu_kfeta.F:2277,2285; L2277/L2285 above), so with microphysics that carries cloud ice
#: no word changes any result and nothing needs measuring.  The word is 0, LOGICAL .false., the value
#: of the mpas_atmphys_vars.F:247 parameter the unassigned local shadows.
INSTALL_WARM_RAIN = "unread-with-ice"
INSTALL_WARM_RAIN_STATEMENT = (
    "cu_kain_fritsch hex-cuda argument packet built by an install: warm_rain is read only at "
    "module_cu_kfeta.F:2277 and :2285, each after .NOT. F_QI; with f_qi true no word changes any "
    "result; the word is 0 (.false., mpas_atmphys_vars.F:247); a run with f_qi false is refused")


def install_packet():
    """The cu_kain_fritsch packet entry an install builds; mpas_atmphys_driver_convection.F:435,542."""
    return {"warm_rain": INSTALL_WARM_RAIN, "warm_rain_word": 0,
            "witness_sha256": hashlib.sha256(INSTALL_WARM_RAIN_STATEMENT.encode()).hexdigest()}


def load_native_arguments(packet, device):
    """Load measured argument metadata; mpas_atmphys_driver_convection.F:435,542.

    The driver-local warm_rain shadows mpas_atmphys_vars.F:247's false parameter and is
    unassigned. A session packet carries its measured raw LOGICAL word and witness
    digest, rather than substituting that unrelated global constant. An install-built
    packet (install_packet) carries word 0 and is bound only where no word is read.
    No weather arrays or numerical formulas are processed here.
    """
    kind = packet.get("warm_rain", "measured")
    word = packet["warm_rain_word"]
    witness = packet["witness_sha256"]
    if kind not in ("measured", INSTALL_WARM_RAIN):
        raise ValueError(f"cu_kain_fritsch packet: unknown warm_rain kind {kind!r}")
    if isinstance(word, bool) or not isinstance(word, int) or not -(1 << 31) <= word < (1 << 31):
        raise ValueError("native warm_rain argument requires its 32-bit LOGICAL storage word")
    if kind == INSTALL_WARM_RAIN and packet != install_packet():
        raise ValueError("cu_kain_fritsch packet: an install-built entry must equal install_packet()")
    if not isinstance(witness, str) or len(witness) != 64 or any(c not in "0123456789abcdef" for c in witness):
        raise ValueError("native argument packet requires the SHA256 of its measurement witness")
    return {"warm_rain_word": np.int32(word), "witness_sha256": witness, "warm_rain": kind}


def check_warm_rain(arguments, f_qi):
    """Refuse the one case an install-built packet cannot cover; module_cu_kfeta.F:2277,2285.

    Breakage it prevents: with no cloud-ice species (f_qi false, e.g. mp_kessler or microphysics
    off) native KF reads a LOGICAL MPAS never sets, so a GPU run on an unmeasured word would
    silently differ from native by whatever that memory held.
    """
    if arguments.get("warm_rain", "measured") == INSTALL_WARM_RAIN and not bool(f_qi):
        raise RuntimeError(
            "cu_kain_fritsch on hex-cuda with microphysics that carries no cloud ice (f_qi false): native "
            "MPAS then reads its driver-local warm_rain, which it never sets (mpas_atmphys_driver_convection.F"
            ":435,542), so no install-built word matches it; choose microphysics with ice (mp_wsm6, "
            "mp_thompson), run convection on native-cpu (config_convection_backend='native-cpu'), or pass a "
            "--physics-resources packet whose cu_kain_fritsch entry carries the measured word")


def bind_native_pools(provider, call):
    """Bind f4 solve-cell storage; mpas_atmphys_driver_convection.F:464-473,516-559,666-790.

    Constants are copied from the compiled native module, species flags from
    native manager scalars (mpas_atmphys_vars.F:161,250-254,303). Driver-local warm_rain
    comes from provider.kain_fritsch_native_arguments, loaded by the selected
    callback resource descriptor. All conversions and timestep arithmetic
    stay in CUDA. config_len_disp is read from the live native config pointer
    (mpas_atmphys_driver_convection.F:455,684,753), which core initialization can update
    (mpas_atm_core.F:156-200). NativeDevicePoolViews.commit retains halos and dummy cells.
    """
    cp = provider.cp
    scalars, constants = provider["scalar_state"], provider["constants"]
    arguments = provider.kain_fritsch_native_arguments
    check_warm_rain(arguments, scalars["f_qi"])
    params = cp.asarray(np.asarray([0, 0, 0, *(constants[name] for name in _NATIVE_CONSTANTS)],
                                   dtype=np.float32))
    flags = cp.asarray(np.asarray([0, 0, 0, arguments["warm_rain_word"],
                                  *(scalars[name] for name in FLAGS[4:])], dtype=np.int32))
    # LUTAB is saved native module state. Its four compiled constants are
    # immutable within this native session; retain the initialized device copy.
    tables = getattr(provider, "kain_fritsch_tables", None)
    if tables is None:
        tables = build_tables(params)
        provider.kain_fritsch_tables = tables
    return {name: provider[name] for name in
            ("prepared", "mesh", "diag_physics", "tend_physics")} | {
                "dt_dyn": scalars["dt_dyn"], "n_cu": scalars["n_cu"],
                "len_disp": provider.native.get_real_config("config_len_disp"),
                "kain_fritsch": {"params": params, "flags": flags, "tables": tables}}


def launch(call):
    """f4 SlotCall callback; mpas_atmphys_driver_convection.F:464-473,516-559.

    pools carries prepared, mesh, diag_physics, tend_physics, dt_dyn, n_cu, len_disp and
    kain_fritsch {params, flags, tables?}. Initialization can supply persistent
    CUDA LUTAB tables, matching native module storage. params/flags are CUDA configuration buffers
    containing native constants and species flags. State lives in MPAS pools.
    The flags' warm_rain word must be supplied by the native bridge: native
    driver declares but does not initialize it (F:435,542); no value is invented.
    """
    import cupy as cp
    pools = call.pools
    config = pools["kain_fritsch"]
    params, flags = config["params"], config["flags"]
    _device_buffer(params, (len(PARAMETERS),), np.dtype("float32"), "params")
    _device_buffer(flags, (len(FLAGS),), np.dtype("int32"), "flags")
    fields, state, surface, pointers = convection_from_MPAS(
        pools["prepared"], pools["mesh"], pools["diag_physics"], pools["tend_physics"],
        pools["len_disp"])
    nc = fields.shape[2]
    active = cp.empty(nc, cp.uint8)
    if nc:
        _function("kf_driver_flags_kernel")(((nc + 63) // 64,), (64,),
            (np.float32(pools["dt_dyn"]), np.float32(call.dt), np.int32(call.itimestep),
             np.int32(pools["n_cu"]), params, flags, active, np.int32(nc)))
        run_packed(fields, state, surface, active, params, flags, tables=config.get("tables"))
        convection_to_MPAS(state, surface, pointers)
    # Activation is scratch in native; held rates/NCA/W0AVG are MPAS state.
    return active


def register(registry):
    """Install the selected entry; mpas_atmphys_driver_convection.F:516-559."""
    registry.register("convection", SCHEME, BACKEND, launch)


BACKEND_ENTRY = ("convection", SCHEME, BACKEND, launch)
