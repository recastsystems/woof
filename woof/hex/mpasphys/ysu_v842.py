"""MPAS-A v8.4.2 YSU boundary layer, independent of the pinned WOOF backend.

The complete MMM scheme is transcribed with one CUDA thread per column.
Every executable statement carries its native source line. The MPAS WRF
wrapper and PBL driver conversions run on device; Python only launches them.
Shared device mathematics is injected by the MPAS physics seam, never replaced
by a scheme-local implementation. No forecast acceptance is claimed here.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics

from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEME = "bl_ysu"
SLOT = "pbl"
BACKEND = "hex-cuda"
# Campaign precision is RKIND=4 and kind_phys=4.
WORKSPACE_ARRAYS = 61  # sixty scheme vectors plus tridin_ysu's private aul
COMPILER_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                    "--prec-div=true", "--prec-sqrt=true")
SOURCE_SHA256 = 'b2c9f11f92f967b053948b42a7c5dbaf1b7f72ddb9d35948e96c32a5d91092d2'
WRAPPER_SHA256 = 'a1f21428c48598afef912ece6901db1a15890e72a31944fe735a19841a45071c'
DRIVER_SHA256 = '31610fde6a68c13101acf845ff728251336b6507d80cb297487cc4dd0337a468'
ARRAY_ARGUMENTS = ('dz8w2d', 'pi2d', 'tx', 'qvx', 'qcx', 'qix', 'qmix', 'utnp', 'vtnp', 'ttnp', 'qvtnp', 'qctnp', 'qitnp', 'qmixtnp', 'p2di', 'p2d', 'hpbl', 'dusfc', 'dvsfc', 'dtsfc', 'dqsfc', 'ust', 'znt', 'xland', 'hfx', 'qfx', 'wspd', 'br', 'psim', 'psih', 'psfcpa', 'kpbl1d', 'ux', 'vx', 'rthraten', 'ctopo', 'ctopo2', 'a_u', 'a_v', 'a_t', 'a_q', 'a_e', 'b_u', 'b_v', 'b_t', 'b_q', 'b_e', 'sfk', 'vlk', 'dlu', 'dlg', 'frcurb', 'exch_hx', 'exch_mx', 'u10', 'v10', 'uox', 'vox', 'wstar', 'delta')
FLOAT_ARGUMENTS = ('dt', 'cp', 'g', 'rovcp', 'rovg', 'rd', 'xlv', 'rv', 'ep1', 'ep2', 'karman')
INTEGER_ARGUMENTS = ('its', 'ite', 'kte', 'kme', 'ysu_topdown_pblmix', 'nmix', 'f_qc', 'f_qi', 'flag_bep')

# MPAS v8.4.2 physics_mmm/bl_ysu.F90:56-1693
_CUDA_BODY = r'''
#ifdef YSU_HOST_WITNESS
#include <cmath>
#define YSU_DEVICE inline
inline float ysu_add(float a, float b) { return a+b; }
inline float ysu_sub(float a, float b) { return a-b; }
inline float ysu_mul(float a, float b) { return a*b; }
inline float ysu_div(float a, float b) { return a/b; }
#define mpas_ysu_sqrtf ::sqrtf
#define mpas_ysu_powf ::powf
#define mpas_ysu_expf ::expf
#define mpas_ysu_tanhf ::tanhf
#define mpas_ysu_absf ::fabsf
#else
// MPAS v8.4.2 physics_mmm/bl_ysu.F90:552-561 (elementary arithmetic order).
#define YSU_DEVICE __device__ __forceinline__
YSU_DEVICE float ysu_add(float a, float b) { return __fadd_rn(a,b); }
YSU_DEVICE float ysu_sub(float a, float b) { return __fsub_rn(a,b); }
YSU_DEVICE float ysu_mul(float a, float b) { return __fmul_rn(a,b); }
YSU_DEVICE float ysu_div(float a, float b) { return __fdiv_rn(a,b); }
// Elementary functions are supplied by the shared device math header.
#endif
// MPAS v8.4.2 physics_mmm/bl_ysu.F90:931-962: GNU folds constant REAL**2 to one RKIND multiply.
YSU_DEVICE float ysu_square(float a) { return ysu_mul(a, a); }
YSU_DEVICE float ysu_minf(float a, float b) { return a < b ? a : b; }
YSU_DEVICE float ysu_maxf(float a, float b) { return a > b ? a : b; }
YSU_DEVICE int ysu_mini(int a, int b) { return a < b ? a : b; }
YSU_DEVICE int ysu_maxi(int a, int b) { return a > b ? a : b; }
struct YsuView {
    float *p;
    int col, ncol, nz;
    YSU_DEVICE float &operator()() const { return p[col]; }
    YSU_DEVICE float &operator()(int k) const { return p[(k-1)*ncol+col]; }
    YSU_DEVICE float &operator()(int k, int n) const { return p[((n-1)*nz+k-1)*ncol+col]; }
};
struct YsuLocal {
    float *p;
    int stride;
    YSU_DEVICE float &operator[](int k) const { return p[k*stride]; }
    YSU_DEVICE float &operator()(int k) const { return p[k*stride]; }
};

struct YsuIntView {
    int *p;
    int col, ncol, nz;
    YSU_DEVICE int &operator()() const { return p[col]; }
};
struct YsuArgs {
    YsuView ux;
    YsuView vx;
    YsuView tx;
    YsuView qvx;
    YsuView qcx;
    YsuView qix;
    int nmix;
    YsuView qmix;
    YsuView p2d;
    YsuView p2di;
    YsuView pi2d;
    bool f_qc;
    bool f_qi;
    YsuView utnp;
    YsuView vtnp;
    YsuView ttnp;
    YsuView qvtnp;
    YsuView qctnp;
    YsuView qitnp;
    YsuView qmixtnp;
    float cp;
    float g;
    float rovcp;
    float rd;
    float rovg;
    float ep1;
    float ep2;
    float karman;
    float xlv;
    float rv;
    YsuView dz8w2d;
    YsuView psfcpa;
    YsuView znt;
    YsuView ust;
    YsuView hpbl;
    YsuView dusfc;
    YsuView dvsfc;
    YsuView dtsfc;
    YsuView dqsfc;
    YsuView psim;
    YsuView psih;
    YsuView xland;
    YsuView hfx;
    YsuView qfx;
    YsuView wspd;
    YsuView br;
    float dt;
    YsuIntView kpbl1d;
    YsuView exch_hx;
    YsuView exch_mx;
    YsuView wstar;
    YsuView delta;
    YsuView u10;
    YsuView v10;
    YsuView uox;
    YsuView vox;
    YsuView rthraten;
    bool ysu_topdown_pblmix;
    YsuView ctopo;
    YsuView ctopo2;
    YsuView a_u;
    YsuView a_v;
    YsuView a_t;
    YsuView a_q;
    YsuView a_e;
    YsuView b_u;
    YsuView b_v;
    YsuView b_t;
    YsuView b_q;
    YsuView b_e;
    YsuView sfk;
    YsuView vlk;
    YsuView dlu;
    YsuView dlg;
    YsuView frcurb;
    bool flag_bep;
    int its;
    int ite;
    int kte;
    int kme;
};
// MPAS v8.4.2 physics_mmm/bl_ysu.F90:1423-1511
YSU_DEVICE void tridi2n(YsuLocal cl, YsuLocal cm, YsuLocal cm1, YsuLocal cu, YsuLocal r1, YsuLocal r2, YsuLocal au, YsuLocal f1, YsuLocal f2, int its, int ite, int kts, int kte, int nt, float *workspace, int ws_stride, int *status) {
    int local_slot = 0;
    float fk;
    int i;
    int k;
    int l;
    int n;
    int it;
    // MPAS line 1452: l = ite
    l = ite;
    // MPAS line 1453: n = kte
    n = kte;
    // MPAS line 1455: do i = its,l
    { // independent column: i = its = ite
        // MPAS line 1456: fk = 1./cm(i,1)
        fk = ysu_div(0x1.0000000000000p+0f, cm(1));
        // MPAS line 1457: au(i,1) = fk*cu(i,1)
        au(1) = ysu_mul(fk, cu(1));
        // MPAS line 1458: f1(i,1) = fk*r1(i,1)
        f1(1) = ysu_mul(fk, r1(1));
        // MPAS line 1459: enddo
    }
    // MPAS line 1461: do it = 1,nt
    for (it = 1; it <= nt; it += 1) {
        // MPAS line 1462: do i = its,l
        { // independent column: i = its = ite
            // MPAS line 1463: fk = 1./cm1(i,1)
            fk = ysu_div(0x1.0000000000000p+0f, cm1(1));
            // MPAS line 1464: f2(i,1,it) = fk*r2(i,1,it)
            f2(1) = ysu_mul(fk, r2(1));
            // MPAS line 1465: enddo
        }
        // MPAS line 1466: enddo
    }
    // MPAS line 1468: do k = kts+1,n-1
    for (k = (kts + 1); k <= (n - 1); k += 1) {
        // MPAS line 1469: do i = its,l
        { // independent column: i = its = ite
            // MPAS line 1470: fk = 1./(cm(i,k)-cl(i,k)*au(i,k-1))
            fk = ysu_div(0x1.0000000000000p+0f, ysu_sub(cm(k), ysu_mul(cl((k - 1)), au((k - 1)))));
            // MPAS line 1471: au(i,k) = fk*cu(i,k)
            au(k) = ysu_mul(fk, cu(k));
            // MPAS line 1472: f1(i,k) = fk*(r1(i,k)-cl(i,k)*f1(i,k-1))
            f1(k) = ysu_mul(fk, ysu_sub(r1(k), ysu_mul(cl((k - 1)), f1((k - 1)))));
            // MPAS line 1473: enddo
        }
        // MPAS line 1474: enddo
    }
    // MPAS line 1476: do it = 1,nt
    for (it = 1; it <= nt; it += 1) {
        // MPAS line 1477: do k = kts+1,n-1
        for (k = (kts + 1); k <= (n - 1); k += 1) {
            // MPAS line 1478: do i = its,l
            { // independent column: i = its = ite
                // MPAS line 1479: fk = 1./(cm1(i,k)-cl(i,k)*au(i,k-1))
                fk = ysu_div(0x1.0000000000000p+0f, ysu_sub(cm1(k), ysu_mul(cl((k - 1)), au((k - 1)))));
                // MPAS line 1480: f2(i,k,it) = fk*(r2(i,k,it)-cl(i,k)*f2(i,k-1,it))
                f2(k) = ysu_mul(fk, ysu_sub(r2(k), ysu_mul(cl((k - 1)), f2((k - 1)))));
                // MPAS line 1481: enddo
            }
            // MPAS line 1482: enddo
        }
        // MPAS line 1483: enddo
    }
    // MPAS line 1485: do i = its,l
    { // independent column: i = its = ite
        // MPAS line 1486: fk = 1./(cm(i,n)-cl(i,n)*au(i,n-1))
        fk = ysu_div(0x1.0000000000000p+0f, ysu_sub(cm(n), ysu_mul(cl((n - 1)), au((n - 1)))));
        // MPAS line 1487: f1(i,n) = fk*(r1(i,n)-cl(i,n)*f1(i,n-1))
        f1(n) = ysu_mul(fk, ysu_sub(r1(n), ysu_mul(cl((n - 1)), f1((n - 1)))));
        // MPAS line 1488: enddo
    }
    // MPAS line 1490: do it = 1,nt
    for (it = 1; it <= nt; it += 1) {
        // MPAS line 1491: do i = its,l
        { // independent column: i = its = ite
            // MPAS line 1492: fk = 1./(cm1(i,n)-cl(i,n)*au(i,n-1))
            fk = ysu_div(0x1.0000000000000p+0f, ysu_sub(cm1(n), ysu_mul(cl((n - 1)), au((n - 1)))));
            // MPAS line 1493: f2(i,n,it) = fk*(r2(i,n,it)-cl(i,n)*f2(i,n-1,it))
            f2(n) = ysu_mul(fk, ysu_sub(r2(n), ysu_mul(cl((n - 1)), f2((n - 1)))));
            // MPAS line 1494: enddo
        }
        // MPAS line 1495: enddo
    }
    // MPAS line 1497: do k = n-1,kts,-1
    for (k = (n - 1); k >= kts; k += -1) {
        // MPAS line 1498: do i = its,l
        { // independent column: i = its = ite
            // MPAS line 1499: f1(i,k) = f1(i,k)-au(i,k)*f1(i,k+1)
            f1(k) = ysu_sub(f1(k), ysu_mul(au(k), f1((k + 1))));
            // MPAS line 1500: enddo
        }
        // MPAS line 1501: enddo
    }
    // MPAS line 1503: do it = 1,nt
    for (it = 1; it <= nt; it += 1) {
        // MPAS line 1504: do k = n-1,kts,-1
        for (k = (n - 1); k >= kts; k += -1) {
            // MPAS line 1505: do i = its,l
            { // independent column: i = its = ite
                // MPAS line 1506: f2(i,k,it) = f2(i,k,it)-au(i,k)*f2(i,k+1,it)
                f2(k) = ysu_sub(f2(k), ysu_mul(au(k), f2((k + 1))));
                // MPAS line 1507: enddo
            }
            // MPAS line 1508: enddo
        }
        // MPAS line 1509: enddo
    }
}

// MPAS v8.4.2 physics_mmm/bl_ysu.F90:1515-1583
YSU_DEVICE void tridin_ysu(YsuLocal cl, YsuLocal cm, YsuLocal cu, YsuLocal r2, YsuLocal au, YsuLocal f2, int its, int ite, int kts, int kte, int nt, float *workspace, int ws_stride, int *status) {
    int local_slot = 0;
    float fk;
    YsuLocal aul{workspace, ws_stride};
    int i;
    int k;
    int l;
    int n;
    int it;
    // MPAS line 1541: l = ite
    l = ite;
    // MPAS line 1542: n = kte
    n = kte;
    // MPAS line 1544: do  i = its,ite
    { // independent column: i = its = ite
        // MPAS line 1545: do k = kts,kte
        for (k = kts; k <= kte; k += 1) {
            // MPAS line 1546: aul(i,k) = 0.
            aul[k] = 0x0.0p+0f;
            // MPAS line 1547: enddo
        }
        // MPAS line 1548: enddo
    }
    // MPAS line 1550: do it = 1,nt
    for (it = 1; it <= nt; it += 1) {
        // MPAS line 1551: do i = its,l
        { // independent column: i = its = ite
            // MPAS line 1552: fk = 1./cm(i,1)
            fk = ysu_div(0x1.0000000000000p+0f, cm(1));
            // MPAS line 1553: aul(i,1) = fk*cu(i,1)
            aul[1] = ysu_mul(fk, cu(1));
            // MPAS line 1554: f2(i,1,it) = fk*r2(i,1,it)
            f2(1) = ysu_mul(fk, r2(1));
            // MPAS line 1555: enddo
        }
        // MPAS line 1556: enddo
    }
    // MPAS line 1558: do it = 1,nt
    for (it = 1; it <= nt; it += 1) {
        // MPAS line 1559: do k = kts+1,n-1
        for (k = (kts + 1); k <= (n - 1); k += 1) {
            // MPAS line 1560: do i = its,l
            { // independent column: i = its = ite
                // MPAS line 1561: fk = 1./(cm(i,k)-cl(i,k)*aul(i,k-1))
                fk = ysu_div(0x1.0000000000000p+0f, ysu_sub(cm(k), ysu_mul(cl((k - 1)), aul[(k - 1)])));
                // MPAS line 1562: aul(i,k) = fk*cu(i,k)
                aul[k] = ysu_mul(fk, cu(k));
                // MPAS line 1563: f2(i,k,it) = fk*(r2(i,k,it)-cl(i,k)*f2(i,k-1,it))
                f2(k) = ysu_mul(fk, ysu_sub(r2(k), ysu_mul(cl((k - 1)), f2((k - 1)))));
                // MPAS line 1564: enddo
            }
            // MPAS line 1565: enddo
        }
        // MPAS line 1566: enddo
    }
    // MPAS line 1568: do it = 1,nt
    for (it = 1; it <= nt; it += 1) {
        // MPAS line 1569: do i = its,l
        { // independent column: i = its = ite
            // MPAS line 1570: fk = 1./(cm(i,n)-cl(i,n)*aul(i,n-1))
            fk = ysu_div(0x1.0000000000000p+0f, ysu_sub(cm(n), ysu_mul(cl((n - 1)), aul[(n - 1)])));
            // MPAS line 1571: f2(i,n,it) = fk*(r2(i,n,it)-cl(i,n)*f2(i,n-1,it))
            f2(n) = ysu_mul(fk, ysu_sub(r2(n), ysu_mul(cl((n - 1)), f2((n - 1)))));
            // MPAS line 1572: enddo
        }
        // MPAS line 1573: enddo
    }
    // MPAS line 1575: do it = 1,nt
    for (it = 1; it <= nt; it += 1) {
        // MPAS line 1576: do k = n-1,kts,-1
        for (k = (n - 1); k >= kts; k += -1) {
            // MPAS line 1577: do i = its,l
            { // independent column: i = its = ite
                // MPAS line 1578: f2(i,k,it) = f2(i,k,it)-aul(i,k)*f2(i,k+1,it)
                f2(k) = ysu_sub(f2(k), ysu_mul(aul[k], f2((k + 1))));
                // MPAS line 1579: enddo
            }
            // MPAS line 1580: enddo
        }
        // MPAS line 1581: enddo
    }
}

// MPAS v8.4.2 physics_mmm/bl_ysu.F90:1586-1693
YSU_DEVICE void get_pblh(int kts, int kte, float & zi, YsuLocal thetav1d, YsuLocal qke1d, YsuLocal zw1d, YsuLocal dz1d, float landsea, float *workspace, int ws_stride, int *status) {
    int local_slot = 0;
    float pblh_tke;
    float qtke;
    float qtkem1;
    float wt;
    float maxqke;
    float tkeeps;
    float minthv;
    float delt_thv;
    const float sbl_lim = 0x1.9000000000000p+7f;
    const float sbl_damp = 0x1.9000000000000p+8f;
    int i;
    int j;
    int k;
    int kthv;
    int ktke;
    // MPAS line 1619: k = kts+1
    k = (kts + 1);
    // MPAS line 1620: kthv = 1
    kthv = 1;
    // MPAS line 1621: ktke = 1
    ktke = 1;
    // MPAS line 1622: maxqke = 0.
    maxqke = 0x0.0p+0f;
    // MPAS line 1623: minthv = 9.e9
    minthv = 0x1.0c388c0000000p+33f;
    // MPAS line 1625: do while (zw1d(k) .le. 500.)
    while (true) {
        // Retain the native loop and reads, including its unbounded extent.
        // MPAS bl_ysu.F90:1625-1636; native-defect policy forbids an abort.
        if (!(zw1d(k) <= 0x1.f400000000000p+8f)) break;
        // MPAS line 1626: qtke  =max(qke1d(k),0.)
        qtke = ysu_maxf(qke1d(k), 0x0.0p+0f);
        // MPAS line 1627: if (maxqke < qtke) then
        if ((maxqke < qtke)) {
            // MPAS line 1628: maxqke = qtke
            maxqke = qtke;
            // MPAS line 1629: ktke = k
            ktke = k;
            // MPAS line 1630: endif
        }
        // MPAS line 1631: if (minthv > thetav1d(k)) then
        if ((minthv > thetav1d(k))) {
            // MPAS line 1632: minthv = thetav1d(k)
            minthv = thetav1d(k);
            // MPAS line 1633: kthv = k
            kthv = k;
            // MPAS line 1634: endif
        }
        // MPAS line 1635: k = k+1
        k = (k + 1);
        // MPAS line 1636: enddo
    }
    // MPAS line 1638: tkeeps = maxqke/40.
    tkeeps = ysu_div(maxqke, 0x1.4000000000000p+5f);
    // MPAS line 1639: tkeeps = max(tkeeps,0.025)
    tkeeps = ysu_maxf(tkeeps, 0x1.99999a0000000p-6f);
    // MPAS line 1640: tkeeps = min(tkeeps,0.25)
    tkeeps = ysu_minf(tkeeps, 0x1.0000000000000p-2f);
    // MPAS line 1643: zi=0.
    zi = 0x0.0p+0f;
    // MPAS line 1644: k = kthv+1
    k = (kthv + 1);
    // MPAS line 1645: if((landsea-1.5).ge.0)then
    if ((ysu_sub(landsea, 0x1.8000000000000p+0f) >= 0)) {
        // MPAS line 1647: delt_thv = 0.75
        delt_thv = 0x1.8000000000000p-1f;
        // MPAS line 1648: else
    } else {
        // MPAS line 1650: delt_thv = 1.5
        delt_thv = 0x1.8000000000000p+0f;
        // MPAS line 1651: endif
    }
    // MPAS line 1653: zi=0.
    zi = 0x0.0p+0f;
    // MPAS line 1654: k = kthv+1
    k = (kthv + 1);
    // MPAS line 1655: do while (zi .eq. 0.)
    while ((zi == 0x0.0p+0f)) {
        // MPAS line 1656: if (thetav1d(k) .ge. (minthv + delt_thv))then
        if ((thetav1d(k) >= ysu_add(minthv, delt_thv))) {
            // MPAS line 1657: zi = zw1d(k) - dz1d(k-1)*  min((thetav1d(k)-(minthv + delt_thv))/max(thetav1d(k)-thetav1d(k-1),1e-6),1.0)
            zi = ysu_sub(zw1d(k), ysu_mul(dz1d((k - 1)), ysu_minf(ysu_div(ysu_sub(thetav1d(k), ysu_add(minthv, delt_thv)), ysu_maxf(ysu_sub(thetav1d(k), thetav1d((k - 1))), 0x1.0c6f7a0000000p-20f)), 0x1.0000000000000p+0f)));
            // MPAS line 1659: endif
        }
        // MPAS line 1660: k = k+1
        k = (k + 1);
        // MPAS line 1661: if (k .eq. kte-1) zi = zw1d(kts+1)
        if ((k == (kte - 1))) zi = zw1d((kts + 1));
        // MPAS line 1662: enddo
    }
    // MPAS line 1671: pblh_tke=0.
    pblh_tke = 0x0.0p+0f;
    // MPAS line 1672: k = ktke+1
    k = (ktke + 1);
    // MPAS line 1673: do while (pblh_tke .eq. 0.)
    while ((pblh_tke == 0x0.0p+0f)) {
        // MPAS line 1675: qtke  =max(qke1d(k)/2.,0.)
        qtke = ysu_maxf(ysu_div(qke1d(k), 0x1.0000000000000p+1f), 0x0.0p+0f);
        // MPAS line 1676: qtkem1=max(qke1d(k-1)/2.,0.)
        qtkem1 = ysu_maxf(ysu_div(qke1d((k - 1)), 0x1.0000000000000p+1f), 0x0.0p+0f);
        // MPAS line 1677: if (qtke .le. tkeeps) then
        if ((qtke <= tkeeps)) {
            // MPAS line 1678: pblh_tke = zw1d(k) - dz1d(k-1)*  min((tkeeps-qtke)/max(qtkem1-qtke, 1e-6), 1.0)
            pblh_tke = ysu_sub(zw1d(k), ysu_mul(dz1d((k - 1)), ysu_minf(ysu_div(ysu_sub(tkeeps, qtke), ysu_maxf(ysu_sub(qtkem1, qtke), 0x1.0c6f7a0000000p-20f)), 0x1.0000000000000p+0f)));
            // MPAS line 1681: pblh_tke = max(pblh_tke,zw1d(kts+1))
            pblh_tke = ysu_maxf(pblh_tke, zw1d((kts + 1)));
            // MPAS line 1683: endif
        }
        // MPAS line 1684: k = k+1
        k = (k + 1);
        // MPAS line 1685: if (k .eq. kte-1) pblh_tke = zw1d(kts+1)
        if ((k == (kte - 1))) pblh_tke = zw1d((kts + 1));
        // MPAS line 1686: enddo
    }
    // MPAS line 1690: wt=.5*tanh((zi - sbl_lim)/sbl_damp) + .5
    wt = ysu_add(ysu_mul(0x1.0000000000000p-1f, mpas_ysu_tanhf(ysu_div(ysu_sub(zi, sbl_lim), sbl_damp))), 0x1.0000000000000p-1f);
    // MPAS line 1691: zi=pblh_tke*(1.-wt) + zi*wt
    zi = ysu_add(ysu_mul(pblh_tke, ysu_sub(0x1.0000000000000p+0f, wt)), ysu_mul(zi, wt));
}

// MPAS v8.4.2 physics_mmm/bl_ysu.F90:56-1420
YSU_DEVICE void bl_ysu_run(YsuArgs p, float *workspace, int ws_stride, int *status) {
    auto ux = p.ux;
    auto vx = p.vx;
    auto tx = p.tx;
    auto qvx = p.qvx;
    auto qcx = p.qcx;
    auto qix = p.qix;
    auto nmix = p.nmix;
    auto qmix = p.qmix;
    auto p2d = p.p2d;
    auto p2di = p.p2di;
    auto pi2d = p.pi2d;
    auto f_qc = p.f_qc;
    auto f_qi = p.f_qi;
    auto utnp = p.utnp;
    auto vtnp = p.vtnp;
    auto ttnp = p.ttnp;
    auto qvtnp = p.qvtnp;
    auto qctnp = p.qctnp;
    auto qitnp = p.qitnp;
    auto qmixtnp = p.qmixtnp;
    auto cp = p.cp;
    auto g = p.g;
    auto rovcp = p.rovcp;
    auto rd = p.rd;
    auto rovg = p.rovg;
    auto ep1 = p.ep1;
    auto ep2 = p.ep2;
    auto karman = p.karman;
    auto xlv = p.xlv;
    auto rv = p.rv;
    auto dz8w2d = p.dz8w2d;
    auto psfcpa = p.psfcpa;
    auto znt = p.znt;
    auto ust = p.ust;
    auto hpbl = p.hpbl;
    auto dusfc = p.dusfc;
    auto dvsfc = p.dvsfc;
    auto dtsfc = p.dtsfc;
    auto dqsfc = p.dqsfc;
    auto psim = p.psim;
    auto psih = p.psih;
    auto xland = p.xland;
    auto hfx = p.hfx;
    auto qfx = p.qfx;
    auto wspd = p.wspd;
    auto br = p.br;
    auto dt = p.dt;
    auto kpbl1d = p.kpbl1d;
    auto exch_hx = p.exch_hx;
    auto exch_mx = p.exch_mx;
    auto wstar = p.wstar;
    auto delta = p.delta;
    auto u10 = p.u10;
    auto v10 = p.v10;
    auto uox = p.uox;
    auto vox = p.vox;
    auto rthraten = p.rthraten;
    auto ysu_topdown_pblmix = p.ysu_topdown_pblmix;
    auto ctopo = p.ctopo;
    auto ctopo2 = p.ctopo2;
    auto a_u = p.a_u;
    auto a_v = p.a_v;
    auto a_t = p.a_t;
    auto a_q = p.a_q;
    auto a_e = p.a_e;
    auto b_u = p.b_u;
    auto b_v = p.b_v;
    auto b_t = p.b_t;
    auto b_q = p.b_q;
    auto b_e = p.b_e;
    auto sfk = p.sfk;
    auto vlk = p.vlk;
    auto dlu = p.dlu;
    auto dlg = p.dlg;
    auto frcurb = p.frcurb;
    auto flag_bep = p.flag_bep;
    auto its = p.its;
    auto ite = p.ite;
    auto kte = p.kte;
    auto kme = p.kme;
    int local_slot = 0;
    const float xkzminm = 0x1.99999a0000000p-4f;
    const float xkzminh = 0x1.47ae140000000p-7f;
    const float xkzmin = 0x1.47ae140000000p-7f;
    const float xkzmax = 0x1.f400000000000p+9f;
    const float rimin = (-0x1.9000000000000p+6f);
    const float rlam = 0x1.e000000000000p+4f;
    const float prmin = 0x1.0000000000000p-2f;
    const float prmax = 0x1.0000000000000p+2f;
    const float brcr_ub = 0x0.0p+0f;
    const float brcr_sb = 0x1.0000000000000p-2f;
    const float cori = 0x1.a36e2e0000000p-14f;
    const float afac = 0x1.b333340000000p+2f;
    const float bfac = 0x1.b333340000000p+2f;
    const float pfac = 0x1.0000000000000p+1f;
    const float pfac_q = 0x1.0000000000000p+1f;
    const float phifac = 0x1.0000000000000p+3f;
    const float sfcfrac = 0x1.99999a0000000p-4f;
    const float d1 = 0x1.47ae140000000p-6f;
    const float d2 = 0x1.99999a0000000p-5f;
    const float d3 = 0x1.0624de0000000p-10f;
    const float h1 = 0x1.5555560000000p-2f;
    const float h2 = 0x1.5555560000000p-1f;
    const float zfmin = 0x1.5798ee0000000p-27f;
    const float aphi5 = 0x1.4000000000000p+2f;
    const float aphi16 = 0x1.0000000000000p+4f;
    const float tmin = 0x1.47ae140000000p-7f;
    const float gamcrt = 0x1.8000000000000p+1f;
    const float gamcrq = 0x1.0624de0000000p-9f;
    const float xka = 0x1.92a7380000000p-16f;
    const int imvdif = 1;
    const float rcl = 0x1.0000000000000p+0f;
    const int kts = 1;
    const int kms = 1;
    float hol;
    YsuLocal zq{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal thx{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal thvx{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal thlix{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal del{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal dza{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal dzq{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal xkzom{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal xkzoh{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal za{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    float rhox;
    float govrth;
    float zl1;
    float thermal;
    float wscale;
    float hgamt;
    float hgamq;
    float brdn;
    float brup;
    float phim;
    float phih;
    float prpbl;
    float wspd1;
    float thermalli;
    YsuLocal xkzh{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal xkzm{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal xkzq{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal f1{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal f2{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal r1{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal r2{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal ad{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal au{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal cu{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal al{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal zfac{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal rhox2{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal hgamt2{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal ad1{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal adm{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal adv{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    float uoxl;
    float voxl;
    float brcr;
    float sflux;
    float zol1;
    float brcr_sbro;
    YsuLocal r3{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal f3{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    int kpbl;
    int kpblold;
    bool pblflg;
    bool sfcflg;
    bool stable;
    bool cloudflg;
    bool definebrup;
    int n;
    int i;
    int k;
    int l;
    int ic;
    int is;
    int kk;
    int klpbl;
    float dt2;
    float rdt;
    float spdk2;
    float fm;
    float fh;
    float hol1;
    float gamfac;
    float vpert;
    float prnum;
    float prnum0;
    float ss;
    float ri;
    float qmean;
    float tmean;
    float alph;
    float chi;
    float zk;
    float rl2;
    float dk;
    float sri;
    float brint;
    float dtodsd;
    float dtodsu;
    float rdz;
    float dsdzt;
    float dsdzq;
    float dsdz2;
    float rlamdz;
    float utend;
    float vtend;
    float ttend;
    float qtend;
    float dtstep;
    float govrthv;
    float cont;
    float conq;
    float conw;
    float conwrc;
    YsuLocal wscalek{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal wscalek2{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal xkzml{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal xkzhl{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal zfacent{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal entfac{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal qcxl{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal qixl{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    float ust3;
    float wstar3;
    float wstar3_2;
    float hgamu;
    float hgamv;
    float wm2;
    float we;
    float bfxpbl;
    float hfxpbl;
    float qfxpbl;
    float ufxpbl;
    float vfxpbl;
    float dthvx;
    float prnumfac;
    float bfx0;
    float hfx0;
    float qfx0;
    float delb;
    float dux;
    float dvx;
    float dsdzu;
    float dsdzv;
    float wm3;
    float dthx;
    float dqx;
    float wspd10;
    float ross;
    float tem1;
    float dsig;
    float tvcon;
    float conpr;
    float prfac;
    float prfac2;
    float phim8z;
    float radsum;
    float tmp1;
    float templ;
    float rvls;
    float temps;
    float ent_eff;
    float rcldb;
    float bruptmp;
    float radflux;
    float vconvlim;
    float vconvnew;
    float fluxc;
    float vconvc;
    float vconv;
    YsuLocal fric{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal tke_ysu{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal el_ysu{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal shear_ysu{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal buoy_ysu{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    float pblh_ysu;
    float vconvfx;
    float bepswitch;
    YsuLocal a_u2d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal a_v2d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal a_t2d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal a_q2d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal a_e2d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal b_u2d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal b_v2d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal b_t2d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal b_q2d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal b_e2d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal sfk2d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal vlk2d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal dlu2d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal dlg2d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    float frc_urb1d;
    YsuLocal thvx_1d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal tke_1d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal dzq_1d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    YsuLocal zq_1d{workspace + (local_slot++) * (kte + 2) * ws_stride, ws_stride};
    // MPAS line 380: klpbl = kte
    klpbl = kte;
    // MPAS line 382: cont=cp/g
    cont = ysu_div(cp, g);
    // MPAS line 383: conq=xlv/g
    conq = ysu_div(xlv, g);
    // MPAS line 384: conw=1./g
    conw = ysu_div(0x1.0000000000000p+0f, g);
    // MPAS line 385: conwrc = conw*sqrt(rcl)
    conwrc = ysu_mul(conw, mpas_ysu_sqrtf(rcl));
    // MPAS line 386: conpr = bfac*karman*sfcfrac
    conpr = ysu_mul(ysu_mul(bfac, karman), sfcfrac);
    // MPAS line 390: if(f_qc) then
    if (f_qc) {
        // MPAS line 391: do k = kts,kte
        for (k = kts; k <= kte; k += 1) {
            // MPAS line 392: do i = its,ite
            { // independent column: i = its = ite
                // MPAS line 393: qcxl(i,k)  = qcx(i,k)
                qcxl[k] = qcx(k);
                // MPAS line 394: enddo
            }
            // MPAS line 395: enddo
        }
        // MPAS line 396: else
    } else {
        // MPAS line 397: do k = kts,kte
        for (k = kts; k <= kte; k += 1) {
            // MPAS line 398: do i = its,ite
            { // independent column: i = its = ite
                // MPAS line 399: qcxl(i,k) = 0.
                qcxl[k] = 0x0.0p+0f;
                // MPAS line 400: enddo
            }
            // MPAS line 401: enddo
        }
        // MPAS line 402: endif
    }
    // MPAS line 404: if(f_qi) then
    if (f_qi) {
        // MPAS line 405: do k = kts,kte
        for (k = kts; k <= kte; k += 1) {
            // MPAS line 406: do i = its,ite
            { // independent column: i = its = ite
                // MPAS line 407: qixl(i,k)  = qix(i,k)
                qixl[k] = qix(k);
                // MPAS line 408: enddo
            }
            // MPAS line 409: enddo
        }
        // MPAS line 410: else
    } else {
        // MPAS line 411: do k = kts,kte
        for (k = kts; k <= kte; k += 1) {
            // MPAS line 412: do i = its,ite
            { // independent column: i = its = ite
                // MPAS line 413: qixl(i,k) = 0.
                qixl[k] = 0x0.0p+0f;
                // MPAS line 414: enddo
            }
            // MPAS line 415: enddo
        }
        // MPAS line 416: endif
    }
    // MPAS line 418: do k = kts,kte
    for (k = kts; k <= kte; k += 1) {
        // MPAS line 419: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 420: thx(i,k) = tx(i,k)/pi2d(i,k)
            thx[k] = ysu_div(tx(k), pi2d(k));
            // MPAS line 421: thlix(i,k) = (tx(i,k)-xlv*qcxl(i,k)/cp-2.834e6*qixl(i,k)/cp)/pi2d(i,k)
            thlix[k] = ysu_div(ysu_sub(ysu_sub(tx(k), ysu_div(ysu_mul(xlv, qcxl[k]), cp)), ysu_div(ysu_mul(0x1.59f2800000000p+21f, qixl[k]), cp)), pi2d(k));
            // MPAS line 422: enddo
        }
        // MPAS line 423: enddo
    }
    // MPAS line 425: do k = kts,kte
    for (k = kts; k <= kte; k += 1) {
        // MPAS line 426: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 427: tvcon = (1.+ep1*qvx(i,k))
            tvcon = ysu_add(0x1.0000000000000p+0f, ysu_mul(ep1, qvx(k)));
            // MPAS line 428: thvx(i,k) = thx(i,k)*tvcon
            thvx[k] = ysu_mul(thx[k], tvcon);
            // MPAS line 429: enddo
        }
        // MPAS line 430: enddo
    }
    // MPAS line 432: if ( present(uox) .and. present(vox) ) then
    if (((uox.p != nullptr) && (vox.p != nullptr))) {
        // MPAS line 433: do i =its,ite
        { // independent column: i = its = ite
            // MPAS line 434: uoxl(i) = uox(i)
            uoxl = uox();
            // MPAS line 435: voxl(i) = vox(i)
            voxl = vox();
            // MPAS line 436: enddo
        }
        // MPAS line 437: else
    } else {
        // MPAS line 438: do i =its,ite
        { // independent column: i = its = ite
            // MPAS line 439: uoxl(i) = 0
            uoxl = 0;
            // MPAS line 440: voxl(i) = 0
            voxl = 0;
            // MPAS line 441: enddo
        }
        // MPAS line 442: endif
    }
    // MPAS line 444: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 445: tvcon = (1.+ep1*qvx(i,1))
        tvcon = ysu_add(0x1.0000000000000p+0f, ysu_mul(ep1, qvx(1)));
        // MPAS line 446: rhox(i) = psfcpa(i)/(rd*tx(i,1)*tvcon)
        rhox = ysu_div(psfcpa(), ysu_mul(ysu_mul(rd, tx(1)), tvcon));
        // MPAS line 447: govrth(i) = g/thx(i,1)
        govrth = ysu_div(g, thx[1]);
        // MPAS line 448: enddo
    }
    // MPAS line 450: if(present(a_u) .and. present(a_v) .and. present(a_t) .and.  present(a_q) .and. present(a_t) .and. present(a_e) .and.  present(b_u) .and. present(b_v) .and. present(b_t) .and.  present(b_q) .and. present(b_e) .and. present(dlg) .and.  present(dlu) .and. present(sfk) .and. present(vlk) .and.  present(frcurb) .and. flag_bep) then
    if ((((((((((((((((((a_u.p != nullptr) && (a_v.p != nullptr)) && (a_t.p != nullptr)) && (a_q.p != nullptr)) && (a_t.p != nullptr)) && (a_e.p != nullptr)) && (b_u.p != nullptr)) && (b_v.p != nullptr)) && (b_t.p != nullptr)) && (b_q.p != nullptr)) && (b_e.p != nullptr)) && (dlg.p != nullptr)) && (dlu.p != nullptr)) && (sfk.p != nullptr)) && (vlk.p != nullptr)) && (frcurb.p != nullptr)) && flag_bep)) {
        // MPAS line 457: bepswitch=1.0
        bepswitch = 0x1.0000000000000p+0f;
        // MPAS line 458: do k = kts, kte
        for (k = kts; k <= kte; k += 1) {
            // MPAS line 459: do i = its,ite
            { // independent column: i = its = ite
                // MPAS line 460: a_u2d(i,k) = a_u(i,k)
                a_u2d[k] = a_u(k);
                // MPAS line 461: a_v2d(i,k) = a_v(i,k)
                a_v2d[k] = a_v(k);
                // MPAS line 462: a_t2d(i,k) = a_t(i,k)
                a_t2d[k] = a_t(k);
                // MPAS line 463: a_q2d(i,k) = a_q(i,k)
                a_q2d[k] = a_q(k);
                // MPAS line 464: a_e2d(i,k) = a_e(i,k)
                a_e2d[k] = a_e(k);
                // MPAS line 465: b_u2d(i,k) = b_u(i,k)
                b_u2d[k] = b_u(k);
                // MPAS line 466: b_v2d(i,k) = b_v(i,k)
                b_v2d[k] = b_v(k);
                // MPAS line 467: b_t2d(i,k) = b_t(i,k)
                b_t2d[k] = b_t(k);
                // MPAS line 468: b_q2d(i,k) = b_q(i,k)
                b_q2d[k] = b_q(k);
                // MPAS line 469: b_e2d(i,k) = b_e(i,k)
                b_e2d[k] = b_e(k);
                // MPAS line 470: dlg2d(i,k) = dlg(i,k)
                dlg2d[k] = dlg(k);
                // MPAS line 471: dlu2d(i,k) = dlu(i,k)
                dlu2d[k] = dlu(k);
                // MPAS line 472: vlk2d(i,k) = vlk(i,k)
                vlk2d[k] = vlk(k);
                // MPAS line 473: sfk2d(i,k) = sfk(i,k)
                sfk2d[k] = sfk(k);
                // MPAS line 474: enddo
            }
            // MPAS line 475: enddo
        }
        // MPAS line 476: do i = its, ite
        { // independent column: i = its = ite
            // MPAS line 477: frc_urb1d(i) = frcurb(i)
            frc_urb1d = frcurb();
            // MPAS line 478: enddo
        }
        // MPAS line 479: else
    } else {
        // MPAS line 480: bepswitch=0.0
        bepswitch = 0x0.0p+0f;
        // MPAS line 481: do k = kts, kte
        for (k = kts; k <= kte; k += 1) {
            // MPAS line 482: do i = its,ite
            { // independent column: i = its = ite
                // MPAS line 483: a_u2d(i,k) = 0.0
                a_u2d[k] = 0x0.0p+0f;
                // MPAS line 484: a_v2d(i,k) = 0.0
                a_v2d[k] = 0x0.0p+0f;
                // MPAS line 485: a_t2d(i,k) = 0.0
                a_t2d[k] = 0x0.0p+0f;
                // MPAS line 486: a_q2d(i,k) = 0.0
                a_q2d[k] = 0x0.0p+0f;
                // MPAS line 487: a_e2d(i,k) = 0.0
                a_e2d[k] = 0x0.0p+0f;
                // MPAS line 488: b_u2d(i,k) = 0.0
                b_u2d[k] = 0x0.0p+0f;
                // MPAS line 489: b_v2d(i,k) = 0.0
                b_v2d[k] = 0x0.0p+0f;
                // MPAS line 490: b_t2d(i,k) = 0.0
                b_t2d[k] = 0x0.0p+0f;
                // MPAS line 491: b_q2d(i,k) = 0.0
                b_q2d[k] = 0x0.0p+0f;
                // MPAS line 492: b_e2d(i,k) = 0.0
                b_e2d[k] = 0x0.0p+0f;
                // MPAS line 493: dlg2d(i,k) = 0.0
                dlg2d[k] = 0x0.0p+0f;
                // MPAS line 494: dlu2d(i,k) = 0.0
                dlu2d[k] = 0x0.0p+0f;
                // MPAS line 495: vlk2d(i,k) = 1.0
                vlk2d[k] = 0x1.0000000000000p+0f;
                // MPAS line 496: sfk2d(i,k) = 1.0
                sfk2d[k] = 0x1.0000000000000p+0f;
                // MPAS line 497: enddo
            }
            // MPAS line 498: enddo
        }
        // MPAS line 499: do i = its, ite
        { // independent column: i = its = ite
            // MPAS line 500: frc_urb1d(i) = 0.0
            frc_urb1d = 0x0.0p+0f;
            // MPAS line 501: enddo
        }
        // MPAS line 502: endif
    }
    // MPAS line 507: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 508: zq(i,1) = 0.
        zq[1] = 0x0.0p+0f;
        // MPAS line 509: enddo
    }
    // MPAS line 511: do k = kts,kte
    for (k = kts; k <= kte; k += 1) {
        // MPAS line 512: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 513: zq(i,k+1) = dz8w2d(i,k)+zq(i,k)
            zq[(k + 1)] = ysu_add(dz8w2d(k), zq[k]);
            // MPAS line 514: tvcon = (1.+ep1*qvx(i,k))
            tvcon = ysu_add(0x1.0000000000000p+0f, ysu_mul(ep1, qvx(k)));
            // MPAS line 515: rhox2(i,k) = p2d(i,k)/(rd*tx(i,k)*tvcon)
            rhox2[k] = ysu_div(p2d(k), ysu_mul(ysu_mul(rd, tx(k)), tvcon));
            // MPAS line 516: enddo
        }
        // MPAS line 517: enddo
    }
    // MPAS line 519: do k = kts,kte
    for (k = kts; k <= kte; k += 1) {
        // MPAS line 520: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 521: za(i,k) = 0.5*(zq(i,k)+zq(i,k+1))
            za[k] = ysu_mul(0x1.0000000000000p-1f, ysu_add(zq[k], zq[(k + 1)]));
            // MPAS line 522: dzq(i,k) = zq(i,k+1)-zq(i,k)
            dzq[k] = ysu_sub(zq[(k + 1)], zq[k]);
            // MPAS line 523: del(i,k) = p2di(i,k)-p2di(i,k+1)
            del[k] = ysu_sub(p2di(k), p2di((k + 1)));
            // MPAS line 524: enddo
        }
        // MPAS line 525: enddo
    }
    // MPAS line 527: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 528: dza(i,1) = za(i,1)
        dza[1] = za[1];
        // MPAS line 529: enddo
    }
    // MPAS line 531: do k = kts+1,kte
    for (k = 2; k <= kte; k += 1) {
        // MPAS line 532: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 533: dza(i,k) = za(i,k)-za(i,k-1)
            dza[k] = ysu_sub(za[k], za[(k - 1)]);
            // MPAS line 534: enddo
        }
        // MPAS line 535: enddo
    }
    // MPAS line 538: do k = kts,kte
    for (k = kts; k <= kte; k += 1) {
        // MPAS line 539: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 540: exch_hx(i,k) = 0.
            exch_hx(k) = 0x0.0p+0f;
            // MPAS line 541: exch_mx(i,k) = 0.
            exch_mx(k) = 0x0.0p+0f;
            // MPAS line 542: xkzh(i,k)    = 0.
            xkzh[k] = 0x0.0p+0f;
            // MPAS line 543: xkzhl(i,k)   = 0.
            xkzhl[k] = 0x0.0p+0f;
            // MPAS line 544: xkzm(i,k)    = 0.
            xkzm[k] = 0x0.0p+0f;
            // MPAS line 545: xkzml(i,k)   = 0.
            xkzml[k] = 0x0.0p+0f;
            // MPAS line 546: xkzq(i,k)    = 0.
            xkzq[k] = 0x0.0p+0f;
            // MPAS line 547: enddo
        }
        // MPAS line 548: enddo
    }
    // MPAS line 550: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 551: wspd1(i) = sqrt( (ux(i,1)-uoxl(i))*(ux(i,1)-uoxl(i)) + (vx(i,1)-voxl(i))*(vx(i,1)-voxl(i)) )+1.e-9
        wspd1 = ysu_add(mpas_ysu_sqrtf(ysu_add(ysu_mul(ysu_sub(ux(1), uoxl), ysu_sub(ux(1), uoxl)), ysu_mul(ysu_sub(vx(1), voxl), ysu_sub(vx(1), voxl)))), 0x1.12e0be0000000p-30f);
        // MPAS line 552: enddo
    }
    // MPAS line 559: dtstep = dt
    dtstep = dt;
    // MPAS line 560: dt2 = 2.*dtstep
    dt2 = ysu_mul(0x1.0000000000000p+1f, dtstep);
    // MPAS line 561: rdt = 1./dt2
    rdt = ysu_div(0x1.0000000000000p+0f, dt2);
    // MPAS line 563: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 564: bfxpbl(i) = 0.0
        bfxpbl = 0x0.0p+0f;
        // MPAS line 565: hfxpbl(i) = 0.0
        hfxpbl = 0x0.0p+0f;
        // MPAS line 566: qfxpbl(i) = 0.0
        qfxpbl = 0x0.0p+0f;
        // MPAS line 567: ufxpbl(i) = 0.0
        ufxpbl = 0x0.0p+0f;
        // MPAS line 568: vfxpbl(i) = 0.0
        vfxpbl = 0x0.0p+0f;
        // MPAS line 569: hgamu(i)  = 0.0
        hgamu = 0x0.0p+0f;
        // MPAS line 570: hgamv(i)  = 0.0
        hgamv = 0x0.0p+0f;
        // MPAS line 571: delta(i)  = 0.0
        delta() = 0x0.0p+0f;
        // MPAS line 572: wstar3_2(i) =  0.0
        wstar3_2 = 0x0.0p+0f;
        // MPAS line 573: enddo
    }
    // MPAS line 575: do k = kts,klpbl
    for (k = kts; k <= klpbl; k += 1) {
        // MPAS line 576: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 577: wscalek(i,k) = 0.0
            wscalek[k] = 0x0.0p+0f;
            // MPAS line 578: wscalek2(i,k) = 0.0
            wscalek2[k] = 0x0.0p+0f;
            // MPAS line 579: enddo
        }
        // MPAS line 580: enddo
    }
    // MPAS line 582: do k = kts,klpbl
    for (k = kts; k <= klpbl; k += 1) {
        // MPAS line 583: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 584: zfac(i,k) = 0.0
            zfac[k] = 0x0.0p+0f;
            // MPAS line 585: enddo
        }
        // MPAS line 586: enddo
    }
    // MPAS line 587: do k = kts,klpbl-1
    for (k = kts; k <= (klpbl - 1); k += 1) {
        // MPAS line 588: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 589: xkzom(i,k) = xkzminm
            xkzom[k] = xkzminm;
            // MPAS line 590: xkzoh(i,k) = xkzminh
            xkzoh[k] = xkzminh;
            // MPAS line 591: enddo
        }
        // MPAS line 592: enddo
    }
    // MPAS line 594: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 595: if(present(dusfc)) dusfc(i) = 0.
        if ((dusfc.p != nullptr)) dusfc() = 0x0.0p+0f;
        // MPAS line 596: if(present(dvsfc)) dvsfc(i) = 0.
        if ((dvsfc.p != nullptr)) dvsfc() = 0x0.0p+0f;
        // MPAS line 597: if(present(dtsfc)) dtsfc(i) = 0.
        if ((dtsfc.p != nullptr)) dtsfc() = 0x0.0p+0f;
        // MPAS line 598: if(present(dqsfc)) dqsfc(i) = 0.
        if ((dqsfc.p != nullptr)) dqsfc() = 0x0.0p+0f;
        // MPAS line 599: enddo
    }
    // MPAS line 601: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 602: hgamt(i)  = 0.
        hgamt = 0x0.0p+0f;
        // MPAS line 603: hgamq(i)  = 0.
        hgamq = 0x0.0p+0f;
        // MPAS line 604: wscale(i) = 0.
        wscale = 0x0.0p+0f;
        // MPAS line 605: we(i)     = 0.
        we = 0x0.0p+0f;
        // MPAS line 606: kpbl(i)   = 1
        kpbl = 1;
        // MPAS line 607: hpbl(i)   = zq(i,1)
        hpbl() = zq[1];
        // MPAS line 608: zl1(i)    = za(i,1)
        zl1 = za[1];
        // MPAS line 609: thermal(i)= thvx(i,1)
        thermal = thvx[1];
        // MPAS line 610: thermalli(i) = thlix(i,1)
        thermalli = thlix[1];
        // MPAS line 611: pblflg(i) = .true.
        pblflg = true;
        // MPAS line 612: sfcflg(i) = .true.
        sfcflg = true;
        // MPAS line 613: sflux(i) = hfx(i)/rhox(i)/cp + qfx(i)/rhox(i)*ep1*thx(i,1)
        sflux = ysu_add(ysu_div(ysu_div(hfx(), rhox), cp), ysu_mul(ysu_mul(ysu_div(qfx(), rhox), ep1), thx[1]));
        // MPAS line 614: if(br(i).gt.0.0) sfcflg(i) = .false.
        if ((br() > 0x0.0p+0f)) sfcflg = false;
        // MPAS line 615: enddo
    }
    // MPAS line 619: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 620: stable(i) = .false.
        stable = false;
        // MPAS line 621: brup(i) = br(i)
        brup = br();
        // MPAS line 622: brcr(i) = brcr_ub
        brcr = brcr_ub;
        // MPAS line 623: enddo
    }
    // MPAS line 625: do k = 2,klpbl
    for (k = 2; k <= klpbl; k += 1) {
        // MPAS line 626: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 627: if(.not.stable(i))then
            if ((!stable)) {
                // MPAS line 628: brdn(i) = brup(i)
                brdn = brup;
                // MPAS line 629: spdk2   = max(ux(i,k)**2+vx(i,k)**2,1.)
                spdk2 = ysu_maxf(ysu_add(ysu_mul(ux(k), ux(k)), ysu_mul(vx(k), vx(k))), 0x1.0000000000000p+0f);
                // MPAS line 630: brup(i) = (thvx(i,k)-thermal(i))*(g*za(i,k)/thvx(i,1))/spdk2
                brup = ysu_div(ysu_mul(ysu_sub(thvx[k], thermal), ysu_div(ysu_mul(g, za[k]), thvx[1])), spdk2);
                // MPAS line 631: kpbl(i) = k
                kpbl = k;
                // MPAS line 632: stable(i) = brup(i).gt.brcr(i)
                stable = (brup > brcr);
                // MPAS line 633: endif
            }
            // MPAS line 634: enddo
        }
        // MPAS line 635: enddo
    }
    // MPAS line 637: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 638: k = kpbl(i)
        k = kpbl;
        // MPAS line 639: if(brdn(i).ge.brcr(i))then
        if ((brdn >= brcr)) {
            // MPAS line 640: brint = 0.
            brint = 0x0.0p+0f;
            // MPAS line 641: elseif(brup(i).le.brcr(i))then
        } else if ((brup <= brcr)) {
            // MPAS line 642: brint = 1.
            brint = 0x1.0000000000000p+0f;
            // MPAS line 643: else
        } else {
            // MPAS line 644: brint = (brcr(i)-brdn(i))/(brup(i)-brdn(i))
            brint = ysu_div(ysu_sub(brcr, brdn), ysu_sub(brup, brdn));
            // MPAS line 645: endif
        }
        // MPAS line 646: hpbl(i) = za(i,k-1)+brint*(za(i,k)-za(i,k-1))
        hpbl() = ysu_add(za[(k - 1)], ysu_mul(brint, ysu_sub(za[k], za[(k - 1)])));
        // MPAS line 647: if(hpbl(i).lt.zq(i,2)) kpbl(i) = 1
        if ((hpbl() < zq[2])) kpbl = 1;
        // MPAS line 648: if(kpbl(i).le.1) pblflg(i) = .false.
        if ((kpbl <= 1)) pblflg = false;
        // MPAS line 649: enddo
    }
    // MPAS line 651: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 652: fm = psim(i)
        fm = psim();
        // MPAS line 653: fh = psih(i)
        fh = psih();
        // MPAS line 654: zol1(i) = max(br(i)*fm*fm/fh,rimin)
        zol1 = ysu_maxf(ysu_div(ysu_mul(ysu_mul(br(), fm), fm), fh), rimin);
        // MPAS line 655: if(sfcflg(i))then
        if (sfcflg) {
            // MPAS line 656: zol1(i) = min(zol1(i),-zfmin)
            zol1 = ysu_minf(zol1, (-zfmin));
            // MPAS line 657: else
        } else {
            // MPAS line 658: zol1(i) = max(zol1(i),zfmin)
            zol1 = ysu_maxf(zol1, zfmin);
            // MPAS line 659: endif
        }
        // MPAS line 660: hol1 = zol1(i)*hpbl(i)/zl1(i)*sfcfrac
        hol1 = ysu_mul(ysu_div(ysu_mul(zol1, hpbl()), zl1), sfcfrac);
        // MPAS line 661: if(sfcflg(i))then
        if (sfcflg) {
            // MPAS line 662: phim(i) = (1.-aphi16*hol1)**(-1./4.)
            phim = mpas_ysu_powf(ysu_sub(0x1.0000000000000p+0f, ysu_mul(aphi16, hol1)), -0x1.0000000000000p-2f);
            // MPAS line 663: phih(i) = (1.-aphi16*hol1)**(-1./2.)
            phih = mpas_ysu_powf(ysu_sub(0x1.0000000000000p+0f, ysu_mul(aphi16, hol1)), -0x1.0000000000000p-1f);
            // MPAS line 664: bfx0 = max(sflux(i),0.)
            bfx0 = ysu_maxf(sflux, 0x0.0p+0f);
            // MPAS line 665: hfx0 = max(hfx(i)/rhox(i)/cp,0.)
            hfx0 = ysu_maxf(ysu_div(ysu_div(hfx(), rhox), cp), 0x0.0p+0f);
            // MPAS line 666: qfx0 = max(ep1*thx(i,1)*qfx(i)/rhox(i),0.)
            qfx0 = ysu_maxf(ysu_div(ysu_mul(ysu_mul(ep1, thx[1]), qfx()), rhox), 0x0.0p+0f);
            // MPAS line 667: wstar3(i) = (govrth(i)*bfx0*hpbl(i))
            wstar3 = ysu_mul(ysu_mul(govrth, bfx0), hpbl());
            // MPAS line 668: wstar(i) = (wstar3(i))**h1
            wstar() = mpas_ysu_powf(wstar3, h1);
            // MPAS line 669: else
        } else {
            // MPAS line 670: phim(i) = (1.+aphi5*hol1)
            phim = ysu_add(0x1.0000000000000p+0f, ysu_mul(aphi5, hol1));
            // MPAS line 671: phih(i) = phim(i)
            phih = phim;
            // MPAS line 672: wstar(i)  = 0.
            wstar() = 0x0.0p+0f;
            // MPAS line 673: wstar3(i) = 0.
            wstar3 = 0x0.0p+0f;
            // MPAS line 674: endif
        }
        // MPAS line 675: ust3(i)   = ust(i)**3.
        ust3 = mpas_ysu_powf(ust(), 0x1.8000000000000p+1f);
        // MPAS line 676: wscale(i) = (ust3(i)+phifac*karman*wstar3(i)*0.5)**h1
        wscale = mpas_ysu_powf(ysu_add(ust3, ysu_mul(ysu_mul(ysu_mul(phifac, karman), wstar3), 0x1.0000000000000p-1f)), h1);
        // MPAS line 677: wscale(i) = min(wscale(i),ust(i)*aphi16)
        wscale = ysu_minf(wscale, ysu_mul(ust(), aphi16));
        // MPAS line 678: wscale(i) = max(wscale(i),ust(i)/aphi5)
        wscale = ysu_maxf(wscale, ysu_div(ust(), aphi5));
        // MPAS line 679: enddo
    }
    // MPAS line 684: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 685: if(sfcflg(i).and.sflux(i).gt.0.0)then
        if ((sfcflg && (sflux > 0x0.0p+0f))) {
            // MPAS line 686: gamfac   = bfac/rhox(i)/wscale(i)
            gamfac = ysu_div(ysu_div(bfac, rhox), wscale);
            // MPAS line 687: hgamt(i) = min(gamfac*hfx(i)/cp,gamcrt)
            hgamt = ysu_minf(ysu_div(ysu_mul(gamfac, hfx()), cp), gamcrt);
            // MPAS line 688: hgamq(i) = min(gamfac*qfx(i),gamcrq)
            hgamq = ysu_minf(ysu_mul(gamfac, qfx()), gamcrq);
            // MPAS line 689: vpert = (hgamt(i)+ep1*thx(i,1)*hgamq(i))/bfac*afac
            vpert = ysu_mul(ysu_div(ysu_add(hgamt, ysu_mul(ysu_mul(ep1, thx[1]), hgamq)), bfac), afac);
            // MPAS line 690: thermal(i) = thermal(i)+max(vpert,0.)*min(za(i,1)/(sfcfrac*hpbl(i)),1.0)
            thermal = ysu_add(thermal, ysu_mul(ysu_maxf(vpert, 0x0.0p+0f), ysu_minf(ysu_div(za[1], ysu_mul(sfcfrac, hpbl())), 0x1.0000000000000p+0f)));
            // MPAS line 691: thermalli(i)= thermalli(i)+max(vpert,0.)*min(za(i,1)/(sfcfrac*hpbl(i)),1.0)
            thermalli = ysu_add(thermalli, ysu_mul(ysu_maxf(vpert, 0x0.0p+0f), ysu_minf(ysu_div(za[1], ysu_mul(sfcfrac, hpbl())), 0x1.0000000000000p+0f)));
            // MPAS line 692: hgamt(i) = max(hgamt(i),0.0)
            hgamt = ysu_maxf(hgamt, 0x0.0p+0f);
            // MPAS line 693: hgamq(i) = max(hgamq(i),0.0)
            hgamq = ysu_maxf(hgamq, 0x0.0p+0f);
            // MPAS line 694: brint    = -15.9*ust(i)*ust(i)/wspd(i)*wstar3(i)/(wscale(i)**4.)
            brint = ysu_div(ysu_mul(ysu_div(ysu_mul(ysu_mul((-0x1.fccccc0000000p+3f), ust()), ust()), wspd()), wstar3), mpas_ysu_powf(wscale, 0x1.0000000000000p+2f));
            // MPAS line 695: hgamu(i) = brint*ux(i,1)
            hgamu = ysu_mul(brint, ux(1));
            // MPAS line 696: hgamv(i) = brint*vx(i,1)
            hgamv = ysu_mul(brint, vx(1));
            // MPAS line 697: else
        } else {
            // MPAS line 698: pblflg(i) = .false.
            pblflg = false;
            // MPAS line 699: endif
        }
        // MPAS line 700: enddo
    }
    // MPAS line 704: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 705: if(pblflg(i))then
        if (pblflg) {
            // MPAS line 706: kpbl(i) = 1
            kpbl = 1;
            // MPAS line 707: hpbl(i) = zq(i,1)
            hpbl() = zq[1];
            // MPAS line 708: endif
        }
        // MPAS line 709: enddo
    }
    // MPAS line 711: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 712: if(pblflg(i))then
        if (pblflg) {
            // MPAS line 713: stable(i) = .false.
            stable = false;
            // MPAS line 714: brup(i) = br(i)
            brup = br();
            // MPAS line 715: brcr(i) = brcr_ub
            brcr = brcr_ub;
            // MPAS line 716: endif
        }
        // MPAS line 717: enddo
    }
    // MPAS line 719: do k = 2,klpbl
    for (k = 2; k <= klpbl; k += 1) {
        // MPAS line 720: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 721: if(.not.stable(i).and.pblflg(i))then
            if (((!stable) && pblflg)) {
                // MPAS line 722: brdn(i) = brup(i)
                brdn = brup;
                // MPAS line 723: spdk2   = max(ux(i,k)**2+vx(i,k)**2,1.)
                spdk2 = ysu_maxf(ysu_add(ysu_mul(ux(k), ux(k)), ysu_mul(vx(k), vx(k))), 0x1.0000000000000p+0f);
                // MPAS line 724: brup(i) = (thvx(i,k)-thermal(i))*(g*za(i,k)/thvx(i,1))/spdk2
                brup = ysu_div(ysu_mul(ysu_sub(thvx[k], thermal), ysu_div(ysu_mul(g, za[k]), thvx[1])), spdk2);
                // MPAS line 725: kpbl(i) = k
                kpbl = k;
                // MPAS line 726: stable(i) = brup(i).gt.brcr(i)
                stable = (brup > brcr);
                // MPAS line 727: endif
            }
            // MPAS line 728: enddo
        }
        // MPAS line 729: enddo
    }
    // MPAS line 733: if (ysu_topdown_pblmix)then
    if (ysu_topdown_pblmix) {
        // MPAS line 734: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 735: kpblold(i) = kpbl(i)
            kpblold = kpbl;
            // MPAS line 736: definebrup=.false.
            definebrup = false;
            // MPAS line 737: do k = kpblold(i), kte-1
            for (k = kpblold; k <= (kte - 1); k += 1) {
                // MPAS line 738: spdk2   = max(ux(i,k)**2+vx(i,k)**2,1.)
                spdk2 = ysu_maxf(ysu_add(ysu_mul(ux(k), ux(k)), ysu_mul(vx(k), vx(k))), 0x1.0000000000000p+0f);
                // MPAS line 739: bruptmp = (thlix(i,k)-thermalli(i))*(g*za(i,k)/thlix(i,1))/spdk2
                bruptmp = ysu_div(ysu_mul(ysu_sub(thlix[k], thermalli), ysu_div(ysu_mul(g, za[k]), thlix[1])), spdk2);
                // MPAS line 740: stable(i) = bruptmp.ge.brcr(i)
                stable = (bruptmp >= brcr);
                // MPAS line 741: if (definebrup) then
                if (definebrup) {
                    // MPAS line 742: kpbl(i) = k
                    kpbl = k;
                    // MPAS line 743: brup(i) = bruptmp
                    brup = bruptmp;
                    // MPAS line 744: definebrup=.false.
                    definebrup = false;
                    // MPAS line 745: endif
                }
                // MPAS line 746: if (.not.stable(i)) then
                if ((!stable)) {
                    // MPAS line 747: brdn(i)=bruptmp
                    brdn = bruptmp;
                    // MPAS line 748: definebrup=.true.
                    definebrup = true;
                    // MPAS line 749: pblflg(i)=.true.
                    pblflg = true;
                    // MPAS line 750: endif
                }
                // MPAS line 751: enddo
            }
            // MPAS line 752: enddo
        }
        // MPAS line 753: endif
    }
    // MPAS line 755: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 756: if(pblflg(i)) then
        if (pblflg) {
            // MPAS line 757: k = kpbl(i)
            k = kpbl;
            // MPAS line 758: if(brdn(i).ge.brcr(i))then
            if ((brdn >= brcr)) {
                // MPAS line 759: brint = 0.
                brint = 0x0.0p+0f;
                // MPAS line 760: elseif(brup(i).le.brcr(i))then
            } else if ((brup <= brcr)) {
                // MPAS line 761: brint = 1.
                brint = 0x1.0000000000000p+0f;
                // MPAS line 762: else
            } else {
                // MPAS line 763: brint = (brcr(i)-brdn(i))/(brup(i)-brdn(i))
                brint = ysu_div(ysu_sub(brcr, brdn), ysu_sub(brup, brdn));
                // MPAS line 764: endif
            }
            // MPAS line 765: hpbl(i) = za(i,k-1)+brint*(za(i,k)-za(i,k-1))
            hpbl() = ysu_add(za[(k - 1)], ysu_mul(brint, ysu_sub(za[k], za[(k - 1)])));
            // MPAS line 766: if(hpbl(i).lt.zq(i,2)) kpbl(i) = 1
            if ((hpbl() < zq[2])) kpbl = 1;
            // MPAS line 767: if(kpbl(i).le.1) pblflg(i) = .false.
            if ((kpbl <= 1)) pblflg = false;
            // MPAS line 768: endif
        }
        // MPAS line 769: enddo
    }
    // MPAS line 773: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 774: if((.not.sfcflg(i)).and.hpbl(i).lt.zq(i,2)) then
        if (((!sfcflg) && (hpbl() < zq[2]))) {
            // MPAS line 775: brup(i) = br(i)
            brup = br();
            // MPAS line 776: stable(i) = .false.
            stable = false;
            // MPAS line 777: else
        } else {
            // MPAS line 778: stable(i) = .true.
            stable = true;
            // MPAS line 779: endif
        }
        // MPAS line 780: enddo
    }
    // MPAS line 782: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 783: if((.not.stable(i)).and.((xland(i)-1.5).ge.0))then
        if (((!stable) && (ysu_sub(xland(), 0x1.8000000000000p+0f) >= 0))) {
            // MPAS line 784: wspd10 = u10(i)*u10(i) + v10(i)*v10(i)
            wspd10 = ysu_add(ysu_mul(u10(), u10()), ysu_mul(v10(), v10()));
            // MPAS line 785: wspd10 = sqrt(wspd10)
            wspd10 = mpas_ysu_sqrtf(wspd10);
            // MPAS line 786: ross = wspd10 / (cori*znt(i))
            ross = ysu_div(wspd10, ysu_mul(cori, znt()));
            // MPAS line 787: brcr_sbro(i) = min(0.16*(1.e-7*ross)**(-0.18),.3)
            brcr_sbro = ysu_minf(ysu_mul(0x1.47ae140000000p-3f, mpas_ysu_powf(ysu_mul(0x1.ad7f2a0000000p-24f, ross), (-0x1.70a3d80000000p-3f))), 0x1.3333340000000p-2f);
            // MPAS line 788: endif
        }
        // MPAS line 789: enddo
    }
    // MPAS line 791: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 792: if(.not.stable(i))then
        if ((!stable)) {
            // MPAS line 793: if((xland(i)-1.5).ge.0)then
            if ((ysu_sub(xland(), 0x1.8000000000000p+0f) >= 0)) {
                // MPAS line 794: brcr(i) = brcr_sbro(i)
                brcr = brcr_sbro;
                // MPAS line 795: else
            } else {
                // MPAS line 796: brcr(i) = brcr_sb
                brcr = brcr_sb;
                // MPAS line 797: endif
            }
            // MPAS line 798: endif
        }
        // MPAS line 799: enddo
    }
    // MPAS line 801: do k = 2,klpbl
    for (k = 2; k <= klpbl; k += 1) {
        // MPAS line 802: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 803: if(.not.stable(i))then
            if ((!stable)) {
                // MPAS line 804: brdn(i) = brup(i)
                brdn = brup;
                // MPAS line 805: spdk2   = max(ux(i,k)**2+vx(i,k)**2,1.)
                spdk2 = ysu_maxf(ysu_add(ysu_mul(ux(k), ux(k)), ysu_mul(vx(k), vx(k))), 0x1.0000000000000p+0f);
                // MPAS line 806: brup(i) = (thvx(i,k)-thermal(i))*(g*za(i,k)/thvx(i,1))/spdk2
                brup = ysu_div(ysu_mul(ysu_sub(thvx[k], thermal), ysu_div(ysu_mul(g, za[k]), thvx[1])), spdk2);
                // MPAS line 807: kpbl(i) = k
                kpbl = k;
                // MPAS line 808: stable(i) = brup(i).gt.brcr(i)
                stable = (brup > brcr);
                // MPAS line 809: endif
            }
            // MPAS line 810: enddo
        }
        // MPAS line 811: enddo
    }
    // MPAS line 813: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 814: if((.not.sfcflg(i)).and.hpbl(i).lt.zq(i,2)) then
        if (((!sfcflg) && (hpbl() < zq[2]))) {
            // MPAS line 815: k = kpbl(i)
            k = kpbl;
            // MPAS line 816: if(brdn(i).ge.brcr(i))then
            if ((brdn >= brcr)) {
                // MPAS line 817: brint = 0.
                brint = 0x0.0p+0f;
                // MPAS line 818: elseif(brup(i).le.brcr(i))then
            } else if ((brup <= brcr)) {
                // MPAS line 819: brint = 1.
                brint = 0x1.0000000000000p+0f;
                // MPAS line 820: else
            } else {
                // MPAS line 821: brint = (brcr(i)-brdn(i))/(brup(i)-brdn(i))
                brint = ysu_div(ysu_sub(brcr, brdn), ysu_sub(brup, brdn));
                // MPAS line 822: endif
            }
            // MPAS line 823: hpbl(i) = za(i,k-1)+brint*(za(i,k)-za(i,k-1))
            hpbl() = ysu_add(za[(k - 1)], ysu_mul(brint, ysu_sub(za[k], za[(k - 1)])));
            // MPAS line 824: if(hpbl(i).lt.zq(i,2)) kpbl(i) = 1
            if ((hpbl() < zq[2])) kpbl = 1;
            // MPAS line 825: if(kpbl(i).le.1) pblflg(i) = .false.
            if ((kpbl <= 1)) pblflg = false;
            // MPAS line 826: endif
        }
        // MPAS line 827: enddo
    }
    // MPAS line 831: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 832: cloudflg(i)=.false.
        cloudflg = false;
        // MPAS line 833: if(pblflg(i)) then
        if (pblflg) {
            // MPAS line 834: k = kpbl(i) - 1
            k = (kpbl - 1);
            // MPAS line 835: wm3       = wstar3(i) + 5. * ust3(i)
            wm3 = ysu_add(wstar3, ysu_mul(0x1.4000000000000p+2f, ust3));
            // MPAS line 836: wm2(i)    = wm3**h2
            wm2 = mpas_ysu_powf(wm3, h2);
            // MPAS line 837: bfxpbl(i) = -0.15*thvx(i,1)/g*wm3/hpbl(i)
            bfxpbl = ysu_div(ysu_mul(ysu_div(ysu_mul((-0x1.3333340000000p-3f), thvx[1]), g), wm3), hpbl());
            // MPAS line 838: dthvx(i)  = max(thvx(i,k+1)-thvx(i,k),tmin)
            dthvx = ysu_maxf(ysu_sub(thvx[(k + 1)], thvx[k]), tmin);
            // MPAS line 839: we(i) = max(bfxpbl(i)/dthvx(i),-sqrt(wm2(i)))
            we = ysu_maxf(ysu_div(bfxpbl, dthvx), (-mpas_ysu_sqrtf(wm2)));
            // MPAS line 840: if((qcxl(i,k)+qixl(i,k)).gt.0.01e-3.and.ysu_topdown_pblmix)then
            if (((ysu_add(qcxl[k], qixl[k]) > 0x1.4f8b580000000p-17f) && ysu_topdown_pblmix)) {
                // MPAS line 841: if ( kpbl(i) .ge. 2) then
                if ((kpbl >= 2)) {
                    // MPAS line 842: cloudflg(i)=.true.
                    cloudflg = true;
                    // MPAS line 843: templ=thlix(i,k)*(p2di(i,k+1)/100000)**rovcp
                    templ = ysu_mul(thlix[k], mpas_ysu_powf(ysu_div(p2di((k + 1)), 100000), rovcp));
                    // MPAS line 845: rvls=100.*6.112*exp(17.67*(templ-273.16)/(templ-29.65))*(ep2/p2di(i,k+1))
                    rvls = ysu_mul(ysu_mul(0x1.31999a0000000p+9f, mpas_ysu_expf(ysu_div(ysu_mul(0x1.1ab8520000000p+4f, ysu_sub(templ, 0x1.1128f60000000p+8f)), ysu_sub(templ, 0x1.da66660000000p+4f)))), ysu_div(ep2, p2di((k + 1))));
                    // MPAS line 846: temps=templ + ((qvx(i,k)+qcxl(i,k))-rvls)/(cp/xlv  +  ep2*xlv*rvls/(rd*templ**2))
                    temps = ysu_add(templ, ysu_div(ysu_sub(ysu_add(qvx(k), qcxl[k]), rvls), ysu_add(ysu_div(cp, xlv), ysu_div(ysu_mul(ysu_mul(ep2, xlv), rvls), ysu_mul(rd, ysu_mul(templ, templ))))));
                    // MPAS line 848: rvls=100.*6.112*exp(17.67*(temps-273.15)/(temps-29.65))*(ep2/p2di(i,k+1))
                    rvls = ysu_mul(ysu_mul(0x1.31999a0000000p+9f, mpas_ysu_expf(ysu_div(ysu_mul(0x1.1ab8520000000p+4f, ysu_sub(temps, 0x1.1126660000000p+8f)), ysu_sub(temps, 0x1.da66660000000p+4f)))), ysu_div(ep2, p2di((k + 1))));
                    // MPAS line 849: rcldb=max((qvx(i,k)+qcxl(i,k))-rvls,0.)
                    rcldb = ysu_maxf(ysu_sub(ysu_add(qvx(k), qcxl[k]), rvls), 0x0.0p+0f);
                    // MPAS line 851: dthvx(i)  = (thlix(i,k+2)+thx(i,k+2)*ep1*(qvx(i,k+2)+qcxl(i,k+2)))  - (thlix(i,k) + thx(i,k)  *ep1*(qvx(i,k)  +qcxl(i,k)))
                    dthvx = ysu_sub(ysu_add(thlix[(k + 2)], ysu_mul(ysu_mul(thx[(k + 2)], ep1), ysu_add(qvx((k + 2)), qcxl[(k + 2)]))), ysu_add(thlix[k], ysu_mul(ysu_mul(thx[k], ep1), ysu_add(qvx(k), qcxl[k]))));
                    // MPAS line 853: dthvx(i)  = max(dthvx(i),0.1)
                    dthvx = ysu_maxf(dthvx, 0x1.99999a0000000p-4f);
                    // MPAS line 854: tmp1      = xlv/cp * rcldb/(pi2d(i,k)*dthvx(i))
                    tmp1 = ysu_div(ysu_mul(ysu_div(xlv, cp), rcldb), ysu_mul(pi2d(k), dthvx));
                    // MPAS line 855: ent_eff   = 0.2 * 8. * tmp1 +0.2
                    ent_eff = ysu_add(ysu_mul(0x1.99999a0000000p+0f, tmp1), 0x1.99999a0000000p-3f);
                    // MPAS line 857: radsum=0.
                    radsum = 0x0.0p+0f;
                    // MPAS line 858: do kk = 1,kpbl(i)-1
                    for (kk = 1; kk <= (kpbl - 1); kk += 1) {
                        // MPAS line 859: radflux=rthraten(i,kk)*pi2d(i,kk)
                        radflux = ysu_mul(rthraten(kk), pi2d(kk));
                        // MPAS line 860: radflux=radflux*cp/g*(p2di(i,kk)-p2di(i,kk+1))
                        radflux = ysu_mul(ysu_div(ysu_mul(radflux, cp), g), ysu_sub(p2di(kk), p2di((kk + 1))));
                        // MPAS line 861: if (radflux < 0.0 ) radsum=abs(radflux)+radsum
                        if ((radflux < 0x0.0p+0f)) radsum = ysu_add(mpas_ysu_absf(radflux), radsum);
                        // MPAS line 862: enddo
                    }
                    // MPAS line 863: radsum=max(radsum,0.0)
                    radsum = ysu_maxf(radsum, 0x0.0p+0f);
                    // MPAS line 866: bfx0 = max(max(sflux(i),0.0)-radsum/rhox2(i,k)/cp,0.)
                    bfx0 = ysu_maxf(ysu_sub(ysu_maxf(sflux, 0x0.0p+0f), ysu_div(ysu_div(radsum, rhox2[k]), cp)), 0x0.0p+0f);
                    // MPAS line 867: bfx0 = max(sflux(i),0.0)
                    bfx0 = ysu_maxf(sflux, 0x0.0p+0f);
                    // MPAS line 868: wm3 = (govrth(i)*bfx0*hpbl(i))+5. * ust3(i)
                    wm3 = ysu_add(ysu_mul(ysu_mul(govrth, bfx0), hpbl()), ysu_mul(0x1.4000000000000p+2f, ust3));
                    // MPAS line 869: wm2(i)    = wm3**h2
                    wm2 = mpas_ysu_powf(wm3, h2);
                    // MPAS line 870: bfxpbl(i) = -0.15*thvx(i,1)/g*wm3/hpbl(i)
                    bfxpbl = ysu_div(ysu_mul(ysu_div(ysu_mul((-0x1.3333340000000p-3f), thvx[1]), g), wm3), hpbl());
                    // MPAS line 871: dthvx(i)  = max(thvx(i,k+1)-thvx(i,k),tmin)
                    dthvx = ysu_maxf(ysu_sub(thvx[(k + 1)], thvx[k]), tmin);
                    // MPAS line 872: we(i) = max(bfxpbl(i)/dthvx(i),-sqrt(wm2(i)))
                    we = ysu_maxf(ysu_div(bfxpbl, dthvx), (-mpas_ysu_sqrtf(wm2)));
                    // MPAS line 875: bfx0 = max(radsum/rhox2(i,k)/cp-max(sflux(i),0.0),0.)
                    bfx0 = ysu_maxf(ysu_sub(ysu_div(ysu_div(radsum, rhox2[k]), cp), ysu_maxf(sflux, 0x0.0p+0f)), 0x0.0p+0f);
                    // MPAS line 876: bfx0 = max(radsum/rhox2(i,k)/cp,0.)
                    bfx0 = ysu_maxf(ysu_div(ysu_div(radsum, rhox2[k]), cp), 0x0.0p+0f);
                    // MPAS line 877: wm3       = (g/thvx(i,k)*bfx0*hpbl(i))
                    wm3 = ysu_mul(ysu_mul(ysu_div(g, thvx[k]), bfx0), hpbl());
                    // MPAS line 878: wm2(i)    = wm2(i)+wm3**h2
                    wm2 = ysu_add(wm2, mpas_ysu_powf(wm3, h2));
                    // MPAS line 879: bfxpbl(i) = - ent_eff * bfx0
                    bfxpbl = ysu_mul((-ent_eff), bfx0);
                    // MPAS line 880: dthvx(i)  = max(thvx(i,k+1)-thvx(i,k),0.1)
                    dthvx = ysu_maxf(ysu_sub(thvx[(k + 1)], thvx[k]), 0x1.99999a0000000p-4f);
                    // MPAS line 881: we(i) = we(i) + max(bfxpbl(i)/dthvx(i),-sqrt(wm3**h2))
                    we = ysu_add(we, ysu_maxf(ysu_div(bfxpbl, dthvx), (-mpas_ysu_sqrtf(mpas_ysu_powf(wm3, h2)))));
                    // MPAS line 884: bfx0 = max(radsum/rhox2(i,k)/cp,0.)
                    bfx0 = ysu_maxf(ysu_div(ysu_div(radsum, rhox2[k]), cp), 0x0.0p+0f);
                    // MPAS line 885: wstar3_2(i) =  (g/thvx(i,k)*bfx0*hpbl(i))
                    wstar3_2 = ysu_mul(ysu_mul(ysu_div(g, thvx[k]), bfx0), hpbl());
                    // MPAS line 887: wscale(i) = (ust3(i)+phifac*karman*(wstar3(i)+wstar3_2(i))*0.5)**h1
                    wscale = mpas_ysu_powf(ysu_add(ust3, ysu_mul(ysu_mul(ysu_mul(phifac, karman), ysu_add(wstar3, wstar3_2)), 0x1.0000000000000p-1f)), h1);
                    // MPAS line 888: wscale(i) = min(wscale(i),ust(i)*aphi16)
                    wscale = ysu_minf(wscale, ysu_mul(ust(), aphi16));
                    // MPAS line 889: wscale(i) = max(wscale(i),ust(i)/aphi5)
                    wscale = ysu_maxf(wscale, ysu_div(ust(), aphi5));
                    // MPAS line 890: gamfac   = bfac/rhox(i)/wscale(i)
                    gamfac = ysu_div(ysu_div(bfac, rhox), wscale);
                    // MPAS line 891: hgamt(i) = min(gamfac*hfx(i)/cp,gamcrt)
                    hgamt = ysu_minf(ysu_div(ysu_mul(gamfac, hfx()), cp), gamcrt);
                    // MPAS line 892: hgamq(i) = min(gamfac*qfx(i),gamcrq)
                    hgamq = ysu_minf(ysu_mul(gamfac, qfx()), gamcrq);
                    // MPAS line 893: gamfac   = bfac/rhox2(i,k)/wscale(i)
                    gamfac = ysu_div(ysu_div(bfac, rhox2[k]), wscale);
                    // MPAS line 894: hgamt2(i,k) = min(gamfac*radsum/cp,gamcrt)
                    hgamt2[k] = ysu_minf(ysu_div(ysu_mul(gamfac, radsum), cp), gamcrt);
                    // MPAS line 895: hgamt(i) = max(hgamt(i),0.0) + max(hgamt2(i,k),0.0)
                    hgamt = ysu_add(ysu_maxf(hgamt, 0x0.0p+0f), ysu_maxf(hgamt2[k], 0x0.0p+0f));
                    // MPAS line 896: brint    = -15.9*ust(i)*ust(i)/wspd(i)*(wstar3(i)+wstar3_2(i))/(wscale(i)**4.)
                    brint = ysu_div(ysu_mul(ysu_div(ysu_mul(ysu_mul((-0x1.fccccc0000000p+3f), ust()), ust()), wspd()), ysu_add(wstar3, wstar3_2)), mpas_ysu_powf(wscale, 0x1.0000000000000p+2f));
                    // MPAS line 897: hgamu(i) = brint*ux(i,1)
                    hgamu = ysu_mul(brint, ux(1));
                    // MPAS line 898: hgamv(i) = brint*vx(i,1)
                    hgamv = ysu_mul(brint, vx(1));
                    // MPAS line 899: endif
                }
                // MPAS line 900: endif
            }
            // MPAS line 901: prpbl(i) = 1.0
            prpbl = 0x1.0000000000000p+0f;
            // MPAS line 902: dthx  = max(thx(i,k+1)-thx(i,k),tmin)
            dthx = ysu_maxf(ysu_sub(thx[(k + 1)], thx[k]), tmin);
            // MPAS line 903: dqx   = min(qvx(i,k+1)-qvx(i,k),0.0)
            dqx = ysu_minf(ysu_sub(qvx((k + 1)), qvx(k)), 0x0.0p+0f);
            // MPAS line 904: hfxpbl(i) = we(i)*dthx
            hfxpbl = ysu_mul(we, dthx);
            // MPAS line 905: qfxpbl(i) = we(i)*dqx
            qfxpbl = ysu_mul(we, dqx);
            // MPAS line 907: dux = ux(i,k+1)-ux(i,k)
            dux = ysu_sub(ux((k + 1)), ux(k));
            // MPAS line 908: dvx = vx(i,k+1)-vx(i,k)
            dvx = ysu_sub(vx((k + 1)), vx(k));
            // MPAS line 909: if(dux.gt.tmin) then
            if ((dux > tmin)) {
                // MPAS line 910: ufxpbl(i) = max(prpbl(i)*we(i)*dux,-ust(i)*ust(i))
                ufxpbl = ysu_maxf(ysu_mul(ysu_mul(prpbl, we), dux), ysu_mul((-ust()), ust()));
                // MPAS line 911: elseif(dux.lt.-tmin) then
            } else if ((dux < (-tmin))) {
                // MPAS line 912: ufxpbl(i) = min(prpbl(i)*we(i)*dux,ust(i)*ust(i))
                ufxpbl = ysu_minf(ysu_mul(ysu_mul(prpbl, we), dux), ysu_mul(ust(), ust()));
                // MPAS line 913: else
            } else {
                // MPAS line 914: ufxpbl(i) = 0.0
                ufxpbl = 0x0.0p+0f;
                // MPAS line 915: endif
            }
            // MPAS line 916: if(dvx.gt.tmin) then
            if ((dvx > tmin)) {
                // MPAS line 917: vfxpbl(i) = max(prpbl(i)*we(i)*dvx,-ust(i)*ust(i))
                vfxpbl = ysu_maxf(ysu_mul(ysu_mul(prpbl, we), dvx), ysu_mul((-ust()), ust()));
                // MPAS line 918: elseif(dvx.lt.-tmin) then
            } else if ((dvx < (-tmin))) {
                // MPAS line 919: vfxpbl(i) = min(prpbl(i)*we(i)*dvx,ust(i)*ust(i))
                vfxpbl = ysu_minf(ysu_mul(ysu_mul(prpbl, we), dvx), ysu_mul(ust(), ust()));
                // MPAS line 920: else
            } else {
                // MPAS line 921: vfxpbl(i) = 0.0
                vfxpbl = 0x0.0p+0f;
                // MPAS line 922: endif
            }
            // MPAS line 923: delb  = govrth(i)*d3*hpbl(i)
            delb = ysu_mul(ysu_mul(govrth, d3), hpbl());
            // MPAS line 924: delta(i) = min(d1*hpbl(i) + d2*wm2(i)/delb,100.)
            delta() = ysu_minf(ysu_add(ysu_mul(d1, hpbl()), ysu_div(ysu_mul(d2, wm2), delb)), 0x1.9000000000000p+6f);
            // MPAS line 925: endif
        }
        // MPAS line 926: enddo
    }
    // MPAS line 928: do k = kts,klpbl
    for (k = kts; k <= klpbl; k += 1) {
        // MPAS line 929: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 930: if(pblflg(i).and.k.ge.kpbl(i))then
            if ((pblflg && (k >= kpbl))) {
                // MPAS line 931: entfac(i,k) = ((zq(i,k+1)-hpbl(i))/delta(i))**2.
                entfac[k] = ysu_square(ysu_div(ysu_sub(zq[(k + 1)], hpbl()), delta()));
                // MPAS line 932: else
            } else {
                // MPAS line 933: entfac(i,k) = 1.e30
                entfac[k] = 0x1.93e5940000000p+99f;
                // MPAS line 934: endif
            }
            // MPAS line 935: enddo
        }
        // MPAS line 936: enddo
    }
    // MPAS line 940: do k = kts,klpbl
    for (k = kts; k <= klpbl; k += 1) {
        // MPAS line 941: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 942: if(k.lt.kpbl(i)) then
            if ((k < kpbl)) {
                // MPAS line 943: zfac(i,k) = min(max((1.-(zq(i,k+1)-zl1(i))/(hpbl(i)-zl1(i))),zfmin),1.)
                zfac[k] = ysu_minf(ysu_maxf(ysu_sub(0x1.0000000000000p+0f, ysu_div(ysu_sub(zq[(k + 1)], zl1), ysu_sub(hpbl(), zl1))), zfmin), 0x1.0000000000000p+0f);
                // MPAS line 944: zfacent(i,k) = (1.-zfac(i,k))**3.
                zfacent[k] = mpas_ysu_powf(ysu_sub(0x1.0000000000000p+0f, zfac[k]), 0x1.8000000000000p+1f);
                // MPAS line 945: wscalek(i,k) = (ust3(i)+phifac*karman*wstar3(i)*(1.-zfac(i,k)))**h1
                wscalek[k] = mpas_ysu_powf(ysu_add(ust3, ysu_mul(ysu_mul(ysu_mul(phifac, karman), wstar3), ysu_sub(0x1.0000000000000p+0f, zfac[k]))), h1);
                // MPAS line 946: wscalek2(i,k) = (phifac*karman*wstar3_2(i)*(zfac(i,k)))**h1
                wscalek2[k] = mpas_ysu_powf(ysu_mul(ysu_mul(ysu_mul(phifac, karman), wstar3_2), zfac[k]), h1);
                // MPAS line 947: if(sfcflg(i)) then
                if (sfcflg) {
                    // MPAS line 948: prfac = conpr
                    prfac = conpr;
                    // MPAS line 949: prfac2 = 15.9*(wstar3(i)+wstar3_2(i))/ust3(i)/(1.+4.*karman*(wstar3(i)+wstar3_2(i))/ust3(i))
                    prfac2 = ysu_div(ysu_div(ysu_mul(0x1.fccccc0000000p+3f, ysu_add(wstar3, wstar3_2)), ust3), ysu_add(0x1.0000000000000p+0f, ysu_div(ysu_mul(ysu_mul(0x1.0000000000000p+2f, karman), ysu_add(wstar3, wstar3_2)), ust3)));
                    // MPAS line 950: prnumfac = -3.*(max(zq(i,k+1)-sfcfrac*hpbl(i),0.))**2./hpbl(i)**2.
                    prnumfac = ysu_div(ysu_mul((-0x1.8000000000000p+1f), ysu_square(ysu_maxf(ysu_sub(zq[(k + 1)], ysu_mul(sfcfrac, hpbl())), 0x0.0p+0f))), ysu_square(hpbl()));
                    // MPAS line 951: else
                } else {
                    // MPAS line 952: prfac = 0.
                    prfac = 0x0.0p+0f;
                    // MPAS line 953: prfac2 = 0.
                    prfac2 = 0x0.0p+0f;
                    // MPAS line 954: prnumfac = 0.
                    prnumfac = 0x0.0p+0f;
                    // MPAS line 955: phim8z = 1.+aphi5*zol1(i)*zq(i,k+1)/zl1(i)
                    phim8z = ysu_add(0x1.0000000000000p+0f, ysu_div(ysu_mul(ysu_mul(aphi5, zol1), zq[(k + 1)]), zl1));
                    // MPAS line 956: wscalek(i,k) = ust(i)/phim8z
                    wscalek[k] = ysu_div(ust(), phim8z);
                    // MPAS line 957: wscalek(i,k) = max(wscalek(i,k),0.001)
                    wscalek[k] = ysu_maxf(wscalek[k], 0x1.0624de0000000p-10f);
                    // MPAS line 958: endif
                }
                // MPAS line 959: prnum0 = (phih(i)/phim(i)+prfac)
                prnum0 = ysu_add(ysu_div(phih, phim), prfac);
                // MPAS line 960: prnum0 = max(min(prnum0,prmax),prmin)
                prnum0 = ysu_maxf(ysu_minf(prnum0, prmax), prmin);
                // MPAS line 961: xkzm(i,k) = wscalek(i,k) *karman*    zq(i,k+1)      *    zfac(i,k)**pfac+  wscalek2(i,k)*karman*(hpbl(i)-zq(i,k+1))*(1-zfac(i,k))**pfac
                xkzm[k] = ysu_add(ysu_mul(ysu_mul(ysu_mul(wscalek[k], karman), zq[(k + 1)]), ysu_square(zfac[k])), ysu_mul(ysu_mul(ysu_mul(wscalek2[k], karman), ysu_sub(hpbl(), zq[(k + 1)])), ysu_square(ysu_sub(1, zfac[k]))));
                // MPAS line 964: if (k.eq.kpbl(i)-1.and.cloudflg(i).and.we(i).lt.0.0) then
                if ((((k == (kpbl - 1)) && cloudflg) && (we < 0x0.0p+0f))) {
                    // MPAS line 965: xkzm(i,k) = 0.0
                    xkzm[k] = 0x0.0p+0f;
                    // MPAS line 966: endif
                }
                // MPAS line 967: prnum =  1. + (prnum0-1.)*exp(prnumfac)
                prnum = ysu_add(0x1.0000000000000p+0f, ysu_mul(ysu_sub(prnum0, 0x1.0000000000000p+0f), mpas_ysu_expf(prnumfac)));
                // MPAS line 968: xkzq(i,k) = xkzm(i,k)/prnum*zfac(i,k)**(pfac_q-pfac)
                xkzq[k] = ysu_mul(ysu_div(xkzm[k], prnum), mpas_ysu_powf(zfac[k], 0x0.0p+0f));
                // MPAS line 969: prnum0 = prnum0/(1.+prfac2*karman*sfcfrac)
                prnum0 = ysu_div(prnum0, ysu_add(0x1.0000000000000p+0f, ysu_mul(ysu_mul(prfac2, karman), sfcfrac)));
                // MPAS line 970: prnum =  1. + (prnum0-1.)*exp(prnumfac)
                prnum = ysu_add(0x1.0000000000000p+0f, ysu_mul(ysu_sub(prnum0, 0x1.0000000000000p+0f), mpas_ysu_expf(prnumfac)));
                // MPAS line 971: xkzh(i,k) = xkzm(i,k)/prnum
                xkzh[k] = ysu_div(xkzm[k], prnum);
                // MPAS line 972: xkzm(i,k) = xkzm(i,k)+xkzom(i,k)
                xkzm[k] = ysu_add(xkzm[k], xkzom[k]);
                // MPAS line 973: xkzh(i,k) = xkzh(i,k)+xkzoh(i,k)
                xkzh[k] = ysu_add(xkzh[k], xkzoh[k]);
                // MPAS line 974: xkzq(i,k) = xkzq(i,k)+xkzoh(i,k)
                xkzq[k] = ysu_add(xkzq[k], xkzoh[k]);
                // MPAS line 975: xkzm(i,k) = min(xkzm(i,k),xkzmax)
                xkzm[k] = ysu_minf(xkzm[k], xkzmax);
                // MPAS line 976: xkzh(i,k) = min(xkzh(i,k),xkzmax)
                xkzh[k] = ysu_minf(xkzh[k], xkzmax);
                // MPAS line 977: xkzq(i,k) = min(xkzq(i,k),xkzmax)
                xkzq[k] = ysu_minf(xkzq[k], xkzmax);
                // MPAS line 978: endif
            }
            // MPAS line 979: enddo
        }
        // MPAS line 980: enddo
    }
    // MPAS line 984: do k = kts,kte-1
    for (k = kts; k <= (kte - 1); k += 1) {
        // MPAS line 985: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 986: if(k.ge.kpbl(i)) then
            if ((k >= kpbl)) {
                // MPAS line 987: ss = ((ux(i,k+1)-ux(i,k))*(ux(i,k+1)-ux(i,k))                          +(vx(i,k+1)-vx(i,k))*(vx(i,k+1)-vx(i,k)))                         /(dza(i,k+1)*dza(i,k+1))+1.e-9
                ss = ysu_add(ysu_div(ysu_add(ysu_mul(ysu_sub(ux((k + 1)), ux(k)), ysu_sub(ux((k + 1)), ux(k))), ysu_mul(ysu_sub(vx((k + 1)), vx(k)), ysu_sub(vx((k + 1)), vx(k)))), ysu_mul(dza[(k + 1)], dza[(k + 1)])), 0x1.12e0be0000000p-30f);
                // MPAS line 990: govrthv = g/(0.5*(thvx(i,k+1)+thvx(i,k)))
                govrthv = ysu_div(g, ysu_mul(0x1.0000000000000p-1f, ysu_add(thvx[(k + 1)], thvx[k])));
                // MPAS line 991: ri = govrthv*(thvx(i,k+1)-thvx(i,k))/(ss*dza(i,k+1))
                ri = ysu_div(ysu_mul(govrthv, ysu_sub(thvx[(k + 1)], thvx[k])), ysu_mul(ss, dza[(k + 1)]));
                // MPAS line 992: if(imvdif.eq.1)then
                if ((imvdif == 1)) {
                    // MPAS line 993: if((qcxl(i,k)+qixl(i,k)).gt.0.01e-3.and.  (qcxl(i,k+1)+qixl(i,k+1)).gt.0.01e-3)then
                    if (((ysu_add(qcxl[k], qixl[k]) > 0x1.4f8b580000000p-17f) && (ysu_add(qcxl[(k + 1)], qixl[(k + 1)]) > 0x1.4f8b580000000p-17f))) {
                        // MPAS line 996: qmean = 0.5*(qvx(i,k)+qvx(i,k+1))
                        qmean = ysu_mul(0x1.0000000000000p-1f, ysu_add(qvx(k), qvx((k + 1))));
                        // MPAS line 997: tmean = 0.5*(tx(i,k)+tx(i,k+1))
                        tmean = ysu_mul(0x1.0000000000000p-1f, ysu_add(tx(k), tx((k + 1))));
                        // MPAS line 998: alph  = xlv*qmean/rd/tmean
                        alph = ysu_div(ysu_div(ysu_mul(xlv, qmean), rd), tmean);
                        // MPAS line 999: chi   = xlv*xlv*qmean/cp/rv/tmean/tmean
                        chi = ysu_div(ysu_div(ysu_div(ysu_div(ysu_mul(ysu_mul(xlv, xlv), qmean), cp), rv), tmean), tmean);
                        // MPAS line 1000: ri    = (1.+alph)*(ri-g*g/ss/tmean/cp*((chi-alph)/(1.+chi)))
                        ri = ysu_mul(ysu_add(0x1.0000000000000p+0f, alph), ysu_sub(ri, ysu_mul(ysu_div(ysu_div(ysu_div(ysu_mul(g, g), ss), tmean), cp), ysu_div(ysu_sub(chi, alph), ysu_add(0x1.0000000000000p+0f, chi)))));
                        // MPAS line 1001: endif
                    }
                    // MPAS line 1002: endif
                }
                // MPAS line 1003: zk = karman*zq(i,k+1)
                zk = ysu_mul(karman, zq[(k + 1)]);
                // MPAS line 1004: rlamdz = min(max(0.1*dza(i,k+1),rlam),300.)
                rlamdz = ysu_minf(ysu_maxf(ysu_mul(0x1.99999a0000000p-4f, dza[(k + 1)]), rlam), 0x1.2c00000000000p+8f);
                // MPAS line 1005: rlamdz = min(dza(i,k+1),rlamdz)
                rlamdz = ysu_minf(dza[(k + 1)], rlamdz);
                // MPAS line 1006: rl2 = (zk*rlamdz/(rlamdz+zk))**2
                rl2 = ysu_mul(ysu_div(ysu_mul(zk, rlamdz), ysu_add(rlamdz, zk)), ysu_div(ysu_mul(zk, rlamdz), ysu_add(rlamdz, zk)));
                // MPAS line 1007: dk = rl2*sqrt(ss)
                dk = ysu_mul(rl2, mpas_ysu_sqrtf(ss));
                // MPAS line 1008: if(ri.lt.0.)then
                if ((ri < 0x0.0p+0f)) {
                    // MPAS line 1010: ri = max(ri, rimin)
                    ri = ysu_maxf(ri, rimin);
                    // MPAS line 1011: sri = sqrt(-ri)
                    sri = mpas_ysu_sqrtf((-ri));
                    // MPAS line 1012: xkzm(i,k) = dk*(1+8.*(-ri)/(1+1.746*sri))
                    xkzm[k] = ysu_mul(dk, ysu_add(1, ysu_div(ysu_mul(0x1.0000000000000p+3f, (-ri)), ysu_add(1, ysu_mul(0x1.bef9dc0000000p+0f, sri)))));
                    // MPAS line 1013: xkzh(i,k) = dk*(1+8.*(-ri)/(1+1.286*sri))
                    xkzh[k] = ysu_mul(dk, ysu_add(1, ysu_div(ysu_mul(0x1.0000000000000p+3f, (-ri)), ysu_add(1, ysu_mul(0x1.49374c0000000p+0f, sri)))));
                    // MPAS line 1014: else
                } else {
                    // MPAS line 1016: xkzh(i,k) = dk/(1+5.*ri)**2
                    xkzh[k] = ysu_div(dk, ysu_mul(ysu_add(1, ysu_mul(0x1.4000000000000p+2f, ri)), ysu_add(1, ysu_mul(0x1.4000000000000p+2f, ri))));
                    // MPAS line 1017: prnum = 1.0+2.1*ri
                    prnum = ysu_add(0x1.0000000000000p+0f, ysu_mul(0x1.0ccccc0000000p+1f, ri));
                    // MPAS line 1018: prnum = min(prnum,prmax)
                    prnum = ysu_minf(prnum, prmax);
                    // MPAS line 1019: xkzm(i,k) = xkzh(i,k)*prnum
                    xkzm[k] = ysu_mul(xkzh[k], prnum);
                    // MPAS line 1020: endif
                }
                // MPAS line 1022: xkzm(i,k) = xkzm(i,k)+xkzom(i,k)
                xkzm[k] = ysu_add(xkzm[k], xkzom[k]);
                // MPAS line 1023: xkzh(i,k) = xkzh(i,k)+xkzoh(i,k)
                xkzh[k] = ysu_add(xkzh[k], xkzoh[k]);
                // MPAS line 1024: xkzm(i,k) = min(xkzm(i,k),xkzmax)
                xkzm[k] = ysu_minf(xkzm[k], xkzmax);
                // MPAS line 1025: xkzh(i,k) = min(xkzh(i,k),xkzmax)
                xkzh[k] = ysu_minf(xkzh[k], xkzmax);
                // MPAS line 1026: xkzml(i,k) = xkzm(i,k)
                xkzml[k] = xkzm[k];
                // MPAS line 1027: xkzhl(i,k) = xkzh(i,k)
                xkzhl[k] = xkzh[k];
                // MPAS line 1028: endif
            }
            // MPAS line 1029: enddo
        }
        // MPAS line 1030: enddo
    }
    // MPAS line 1034: do k = kts,kte
    for (k = kts; k <= kte; k += 1) {
        // MPAS line 1035: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 1036: au(i,k) = 0.
            au[k] = 0x0.0p+0f;
            // MPAS line 1037: al(i,k) = 0.
            al[k] = 0x0.0p+0f;
            // MPAS line 1038: ad(i,k) = 0.
            ad[k] = 0x0.0p+0f;
            // MPAS line 1039: f1(i,k) = 0.
            f1[k] = 0x0.0p+0f;
            // MPAS line 1040: enddo
        }
        // MPAS line 1041: enddo
    }
    // MPAS line 1043: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 1044: ad(i,1) = 1.
        ad[1] = 0x1.0000000000000p+0f;
        // MPAS line 1045: f1(i,1) = thx(i,1)-300.+(1.0-bepswitch)*hfx(i)/cont/del(i,1)*dt2
        f1[1] = ysu_add(ysu_sub(thx[1], 0x1.2c00000000000p+8f), ysu_mul(ysu_div(ysu_div(ysu_mul(ysu_sub(0x1.0000000000000p+0f, bepswitch), hfx()), cont), del[1]), dt2));
        // MPAS line 1046: enddo
    }
    // MPAS line 1048: do k = kts,kte-1
    for (k = kts; k <= (kte - 1); k += 1) {
        // MPAS line 1049: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 1050: dtodsd = sfk2d(i,k)*dt2/del(i,k)
            dtodsd = ysu_div(ysu_mul(sfk2d[k], dt2), del[k]);
            // MPAS line 1051: dtodsu = sfk2d(i,k)*dt2/del(i,k+1)
            dtodsu = ysu_div(ysu_mul(sfk2d[k], dt2), del[(k + 1)]);
            // MPAS line 1052: dsig   = p2d(i,k)-p2d(i,k+1)
            dsig = ysu_sub(p2d(k), p2d((k + 1)));
            // MPAS line 1053: rdz    = 1./dza(i,k+1)
            rdz = ysu_div(0x1.0000000000000p+0f, dza[(k + 1)]);
            // MPAS line 1054: tem1   = dsig*xkzh(i,k)*rdz
            tem1 = ysu_mul(ysu_mul(dsig, xkzh[k]), rdz);
            // MPAS line 1055: if(pblflg(i).and.k.lt.kpbl(i)) then
            if ((pblflg && (k < kpbl))) {
                // MPAS line 1056: dsdzt = tem1*(-hgamt(i)/hpbl(i)-hfxpbl(i)*zfacent(i,k)/xkzh(i,k))
                dsdzt = ysu_mul(tem1, ysu_sub(ysu_div((-hgamt), hpbl()), ysu_div(ysu_mul(hfxpbl, zfacent[k]), xkzh[k])));
                // MPAS line 1057: f1(i,k)   = f1(i,k)+dtodsd*dsdzt
                f1[k] = ysu_add(f1[k], ysu_mul(dtodsd, dsdzt));
                // MPAS line 1058: f1(i,k+1) = thx(i,k+1)-300.-dtodsu*dsdzt
                f1[(k + 1)] = ysu_sub(ysu_sub(thx[(k + 1)], 0x1.2c00000000000p+8f), ysu_mul(dtodsu, dsdzt));
                // MPAS line 1059: elseif(pblflg(i).and.k.ge.kpbl(i).and.entfac(i,k).lt.4.6) then
            } else if (((pblflg && (k >= kpbl)) && (entfac[k] < 0x1.2666660000000p+2f))) {
                // MPAS line 1060: xkzh(i,k) = -we(i)*dza(i,kpbl(i))*exp(-entfac(i,k))
                xkzh[k] = ysu_mul(ysu_mul((-we), dza[kpbl]), mpas_ysu_expf((-entfac[k])));
                // MPAS line 1061: xkzh(i,k) = sqrt(xkzh(i,k)*xkzhl(i,k))
                xkzh[k] = mpas_ysu_sqrtf(ysu_mul(xkzh[k], xkzhl[k]));
                // MPAS line 1062: xkzh(i,k) = max(xkzh(i,k),xkzoh(i,k))
                xkzh[k] = ysu_maxf(xkzh[k], xkzoh[k]);
                // MPAS line 1063: xkzh(i,k) = min(xkzh(i,k),xkzmax)
                xkzh[k] = ysu_minf(xkzh[k], xkzmax);
                // MPAS line 1064: f1(i,k+1) = thx(i,k+1)-300.
                f1[(k + 1)] = ysu_sub(thx[(k + 1)], 0x1.2c00000000000p+8f);
                // MPAS line 1065: else
            } else {
                // MPAS line 1066: f1(i,k+1) = thx(i,k+1)-300.
                f1[(k + 1)] = ysu_sub(thx[(k + 1)], 0x1.2c00000000000p+8f);
                // MPAS line 1067: endif
            }
            // MPAS line 1068: tem1   = dsig*xkzh(i,k)*rdz
            tem1 = ysu_mul(ysu_mul(dsig, xkzh[k]), rdz);
            // MPAS line 1069: dsdz2     = tem1*rdz
            dsdz2 = ysu_mul(tem1, rdz);
            // MPAS line 1070: au(i,k)   = -dtodsd*dsdz2/vlk2d(i,k)
            au[k] = ysu_div(ysu_mul((-dtodsd), dsdz2), vlk2d[k]);
            // MPAS line 1071: al(i,k)   = -dtodsu*dsdz2/vlk2d(i,k)
            al[k] = ysu_div(ysu_mul((-dtodsu), dsdz2), vlk2d[k]);
            // MPAS line 1072: ad(i,k)   = ad(i,k)-au(i,k)
            ad[k] = ysu_sub(ad[k], au[k]);
            // MPAS line 1073: ad(i,k+1) = 1.-al(i,k)
            ad[(k + 1)] = ysu_sub(0x1.0000000000000p+0f, al[k]);
            // MPAS line 1074: exch_hx(i,k+1) = xkzh(i,k)
            exch_hx((k + 1)) = xkzh[k];
            // MPAS line 1075: enddo
        }
        // MPAS line 1076: enddo
    }
    // MPAS line 1080: do k = kts,kte
    for (k = kts; k <= kte; k += 1) {
        // MPAS line 1081: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 1082: ad(i,k) = ad(i,k) - a_t2d(i,k)*dt2
            ad[k] = ysu_sub(ad[k], ysu_mul(a_t2d[k], dt2));
            // MPAS line 1083: f1(i,k) = f1(i,k) + b_t2d(i,k)*dt2
            f1[k] = ysu_add(f1[k], ysu_mul(b_t2d[k], dt2));
            // MPAS line 1084: enddo
        }
        // MPAS line 1085: enddo
    }
    // MPAS line 1089: do k = kts,kte
    for (k = kts; k <= kte; k += 1) {
        // MPAS line 1090: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 1091: cu(i,k) = au(i,k)
            cu[k] = au[k];
            // MPAS line 1092: r1(i,k) = f1(i,k)
            r1[k] = f1[k];
            // MPAS line 1093: enddo
        }
        // MPAS line 1094: enddo
    }
    // MPAS line 1096: call tridin_ysu(al,ad,cu,r1,au,f1,its,ite,kts,kte,1)
    tridin_ysu(al, ad, cu, r1, au, f1, its, ite, kts, kte, 1, workspace + local_slot * (kte + 2) * ws_stride, ws_stride, status);
    // MPAS line 1100: do k = kte,kts,-1
    for (k = kte; k >= kts; k += -1) {
        // MPAS line 1101: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 1103: ttend = (f1(i,k)-thx(i,k)+300.)*rdt*pi2d(i,k)
            ttend = ysu_mul(ysu_mul(ysu_add(ysu_sub(f1[k], thx[k]), 0x1.2c00000000000p+8f), rdt), pi2d(k));
            // MPAS line 1104: ttnp(i,k) = ttend
            ttnp(k) = ttend;
            // MPAS line 1105: if(present(dtsfc)) dtsfc(i) = dtsfc(i)+ttend*cont*del(i,k)/pi2d(i,k)
            if ((dtsfc.p != nullptr)) dtsfc() = ysu_add(dtsfc(), ysu_div(ysu_mul(ysu_mul(ttend, cont), del[k]), pi2d(k)));
            // MPAS line 1111: enddo
        }
        // MPAS line 1112: enddo
    }
    // MPAS line 1117: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 1118: do k = kts,kte-1
        for (k = kts; k <= (kte - 1); k += 1) {
            // MPAS line 1119: if(k .ge. kpbl(i)) xkzq(i,k) = xkzh(i,k)
            if ((k >= kpbl)) xkzq[k] = xkzh[k];
            // MPAS line 1120: enddo
        }
        // MPAS line 1121: enddo
    }
    // MPAS line 1124: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 1125: do k = kts,kte
        for (k = kts; k <= kte; k += 1) {
            // MPAS line 1126: au(i,k) = 0.
            au[k] = 0x0.0p+0f;
            // MPAS line 1127: al(i,k) = 0.
            al[k] = 0x0.0p+0f;
            // MPAS line 1128: ad(i,k) = 0.
            ad[k] = 0x0.0p+0f;
            // MPAS line 1129: f1(i,k) = 0.
            f1[k] = 0x0.0p+0f;
            // MPAS line 1130: r1(i,k) = 0.
            r1[k] = 0x0.0p+0f;
            // MPAS line 1131: enddo
        }
        // MPAS line 1133: k = 1
        k = 1;
        // MPAS line 1134: ad(i,1) = 1.
        ad[1] = 0x1.0000000000000p+0f;
        // MPAS line 1135: f1(i,1) = qvx(i,1)+(1.0-bepswitch)*qfx(i)*g/del(i,1)*dt2
        f1[1] = ysu_add(qvx(1), ysu_mul(ysu_div(ysu_mul(ysu_mul(ysu_sub(0x1.0000000000000p+0f, bepswitch), qfx()), g), del[1]), dt2));
        // MPAS line 1137: do k = kts,kte-1
        for (k = kts; k <= (kte - 1); k += 1) {
            // MPAS line 1138: dtodsd = sfk2d(i,k)*dt2/del(i,k)
            dtodsd = ysu_div(ysu_mul(sfk2d[k], dt2), del[k]);
            // MPAS line 1139: dtodsu = sfk2d(i,k)*dt2/del(i,k+1)
            dtodsu = ysu_div(ysu_mul(sfk2d[k], dt2), del[(k + 1)]);
            // MPAS line 1140: dsig   = p2d(i,k)-p2d(i,k+1)
            dsig = ysu_sub(p2d(k), p2d((k + 1)));
            // MPAS line 1141: rdz    = 1./dza(i,k+1)
            rdz = ysu_div(0x1.0000000000000p+0f, dza[(k + 1)]);
            // MPAS line 1142: tem1   = dsig*xkzq(i,k)*rdz
            tem1 = ysu_mul(ysu_mul(dsig, xkzq[k]), rdz);
            // MPAS line 1143: if(pblflg(i).and.k.lt.kpbl(i)) then
            if ((pblflg && (k < kpbl))) {
                // MPAS line 1144: dsdzq = tem1*(-qfxpbl(i)*zfacent(i,k)/xkzq(i,k))
                dsdzq = ysu_mul(tem1, ysu_div(ysu_mul((-qfxpbl), zfacent[k]), xkzq[k]));
                // MPAS line 1145: f1(i,k) = f1(i,k)+dtodsd*dsdzq
                f1[k] = ysu_add(f1[k], ysu_mul(dtodsd, dsdzq));
                // MPAS line 1146: f1(i,k+1) = qvx(i,k+1)-dtodsu*dsdzq
                f1[(k + 1)] = ysu_sub(qvx((k + 1)), ysu_mul(dtodsu, dsdzq));
                // MPAS line 1147: elseif(pblflg(i).and.k.ge.kpbl(i).and.entfac(i,k).lt.4.6) then
            } else if (((pblflg && (k >= kpbl)) && (entfac[k] < 0x1.2666660000000p+2f))) {
                // MPAS line 1148: xkzq(i,k) = -we(i)*dza(i,kpbl(i))*exp(-entfac(i,k))
                xkzq[k] = ysu_mul(ysu_mul((-we), dza[kpbl]), mpas_ysu_expf((-entfac[k])));
                // MPAS line 1149: xkzq(i,k) = sqrt(xkzq(i,k)*xkzhl(i,k))
                xkzq[k] = mpas_ysu_sqrtf(ysu_mul(xkzq[k], xkzhl[k]));
                // MPAS line 1150: xkzq(i,k) = max(xkzq(i,k),xkzoh(i,k))
                xkzq[k] = ysu_maxf(xkzq[k], xkzoh[k]);
                // MPAS line 1151: xkzq(i,k) = min(xkzq(i,k),xkzmax)
                xkzq[k] = ysu_minf(xkzq[k], xkzmax);
                // MPAS line 1152: f1(i,k+1) = qvx(i,k+1)
                f1[(k + 1)] = qvx((k + 1));
                // MPAS line 1153: else
            } else {
                // MPAS line 1154: f1(i,k+1) = qvx(i,k+1)
                f1[(k + 1)] = qvx((k + 1));
                // MPAS line 1155: endif
            }
            // MPAS line 1156: tem1      = dsig*xkzq(i,k)*rdz
            tem1 = ysu_mul(ysu_mul(dsig, xkzq[k]), rdz);
            // MPAS line 1157: dsdz2     = tem1*rdz
            dsdz2 = ysu_mul(tem1, rdz);
            // MPAS line 1158: au(i,k)   = -dtodsd*dsdz2/vlk2d(i,k)
            au[k] = ysu_div(ysu_mul((-dtodsd), dsdz2), vlk2d[k]);
            // MPAS line 1159: al(i,k)   = -dtodsu*dsdz2/vlk2d(i,k)
            al[k] = ysu_div(ysu_mul((-dtodsu), dsdz2), vlk2d[k]);
            // MPAS line 1160: ad(i,k)   = ad(i,k)-au(i,k)
            ad[k] = ysu_sub(ad[k], au[k]);
            // MPAS line 1161: ad(i,k+1) = 1.-al(i,k)
            ad[(k + 1)] = ysu_sub(0x1.0000000000000p+0f, al[k]);
            // MPAS line 1162: enddo
        }
        // MPAS line 1166: do k = kts,kte
        for (k = kts; k <= kte; k += 1) {
            // MPAS line 1167: adv(i,k) = ad(i,k) - a_q2d(i,k)*dt2
            adv[k] = ysu_sub(ad[k], ysu_mul(a_q2d[k], dt2));
            // MPAS line 1168: f1(i,k)  = f1(i,k) + b_q2d(i,k)*dt2
            f1[k] = ysu_add(f1[k], ysu_mul(b_q2d[k], dt2));
            // MPAS line 1169: enddo
        }
        // MPAS line 1171: do k = kts,kte
        for (k = kts; k <= kte; k += 1) {
            // MPAS line 1172: cu(i,k) = au(i,k)
            cu[k] = au[k];
            // MPAS line 1173: r1(i,k) = f1(i,k)
            r1[k] = f1[k];
            // MPAS line 1174: enddo
        }
        // MPAS line 1175: enddo
    }
    // MPAS line 1176: call tridin_ysu(al,adv,cu,r1,au,f1,its,ite,kts,kte,1)
    tridin_ysu(al, adv, cu, r1, au, f1, its, ite, kts, kte, 1, workspace + local_slot * (kte + 2) * ws_stride, ws_stride, status);
    // MPAS line 1178: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 1179: do k = kte,kts,-1
        for (k = kte; k >= kts; k += -1) {
            // MPAS line 1180: qtend = (f1(i,k)-qvx(i,k))*rdt
            qtend = ysu_mul(ysu_sub(f1[k], qvx(k)), rdt);
            // MPAS line 1181: qvtnp(i,k) = qtend
            qvtnp(k) = qtend;
            // MPAS line 1182: if(present(dqsfc)) dqsfc(i)   = dqsfc(i)+qtend*conq*del(i,k)
            if ((dqsfc.p != nullptr)) dqsfc() = ysu_add(dqsfc(), ysu_mul(ysu_mul(qtend, conq), del[k]));
            // MPAS line 1183: enddo
        }
        // MPAS line 1184: enddo
    }
    // MPAS line 1187: if(f_qc) then
    if (f_qc) {
        // MPAS line 1188: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 1189: do k = kts,kte
            for (k = kts; k <= kte; k += 1) {
                // MPAS line 1190: f1(i,k) = qcxl(i,k)
                f1[k] = qcxl[k];
                // MPAS line 1191: r1(i,k) = f1(i,k)
                r1[k] = f1[k];
                // MPAS line 1192: enddo
            }
            // MPAS line 1193: enddo
        }
        // MPAS line 1194: call tridin_ysu(al,ad,cu,r1,au,f1,its,ite,kts,kte,1)
        tridin_ysu(al, ad, cu, r1, au, f1, its, ite, kts, kte, 1, workspace + local_slot * (kte + 2) * ws_stride, ws_stride, status);
        // MPAS line 1196: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 1197: do k = kte,kts,-1
            for (k = kte; k >= kts; k += -1) {
                // MPAS line 1198: qtend = (f1(i,k)-qcxl(i,k))*rdt
                qtend = ysu_mul(ysu_sub(f1[k], qcxl[k]), rdt);
                // MPAS line 1199: qctnp(i,k) = qtend
                qctnp(k) = qtend;
                // MPAS line 1200: enddo
            }
            // MPAS line 1201: enddo
        }
        // MPAS line 1202: endif
    }
    // MPAS line 1205: if(f_qi) then
    if (f_qi) {
        // MPAS line 1206: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 1207: do k = kts,kte
            for (k = kts; k <= kte; k += 1) {
                // MPAS line 1208: f1(i,k) = qixl(i,k)
                f1[k] = qixl[k];
                // MPAS line 1209: r1(i,k) = f1(i,k)
                r1[k] = f1[k];
                // MPAS line 1210: enddo
            }
            // MPAS line 1211: enddo
        }
        // MPAS line 1212: call tridin_ysu(al,ad,cu,r1,au,f1,its,ite,kts,kte,1)
        tridin_ysu(al, ad, cu, r1, au, f1, its, ite, kts, kte, 1, workspace + local_slot * (kte + 2) * ws_stride, ws_stride, status);
        // MPAS line 1214: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 1215: do k = kte,kts,-1
            for (k = kte; k >= kts; k += -1) {
                // MPAS line 1216: qtend = (f1(i,k)-qixl(i,k))*rdt
                qtend = ysu_mul(ysu_sub(f1[k], qixl[k]), rdt);
                // MPAS line 1217: qitnp(i,k) = qtend
                qitnp(k) = qtend;
                // MPAS line 1218: enddo
            }
            // MPAS line 1219: enddo
        }
        // MPAS line 1220: endif
    }
    // MPAS line 1224: do n = 1, nmix
    for (n = 1; n <= nmix; n += 1) {
        // MPAS line 1225: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 1226: do k = kts,kte
            for (k = kts; k <= kte; k += 1) {
                // MPAS line 1227: f1(i,k) = qmix(i,k,n)
                f1[k] = qmix(k, n);
                // MPAS line 1228: r1(i,k) = f1(i,k)
                r1[k] = f1[k];
                // MPAS line 1229: enddo
            }
            // MPAS line 1230: enddo
        }
        // MPAS line 1231: call tridin_ysu(al,ad,cu,r1,au,f1,its,ite,kts,kte,1)
        tridin_ysu(al, ad, cu, r1, au, f1, its, ite, kts, kte, 1, workspace + local_slot * (kte + 2) * ws_stride, ws_stride, status);
        // MPAS line 1233: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 1234: do k = kte,kts,-1
            for (k = kte; k >= kts; k += -1) {
                // MPAS line 1235: qtend = (f1(i,k)-qmix(i,k,n))*rdt
                qtend = ysu_mul(ysu_sub(f1[k], qmix(k, n)), rdt);
                // MPAS line 1236: qmixtnp(i,k,n) = qtend
                qmixtnp(k, n) = qtend;
                // MPAS line 1237: enddo
            }
            // MPAS line 1238: enddo
        }
        // MPAS line 1239: enddo
    }
    // MPAS line 1244: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 1245: do k = kts,kte
        for (k = kts; k <= kte; k += 1) {
            // MPAS line 1246: au(i,k) = 0.
            au[k] = 0x0.0p+0f;
            // MPAS line 1247: al(i,k) = 0.
            al[k] = 0x0.0p+0f;
            // MPAS line 1248: ad(i,k) = 0.
            ad[k] = 0x0.0p+0f;
            // MPAS line 1249: f1(i,k) = 0.
            f1[k] = 0x0.0p+0f;
            // MPAS line 1250: f2(i,k) = 0.
            f2[k] = 0x0.0p+0f;
            // MPAS line 1251: enddo
        }
        // MPAS line 1252: enddo
    }
    // MPAS line 1256: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 1257: do k= kts, kte-1
        for (k = kts; k <= (kte - 1); k += 1) {
            // MPAS line 1258: shear_ysu(i,k)=xkzm(i,k)*((-hgamu(i)/hpbl(i)+(ux(i,k+1)-ux(i,k))/dza(i,k+1))*(ux(i,k+1)-ux(i,k))/dza(i,k+1)  + (-hgamv(i)/hpbl(i)+(vx(i,k+1)-vx(i,k))/dza(i,k+1))*(vx(i,k+1)-vx(i,k))/dza(i,k+1))
            shear_ysu[k] = ysu_mul(xkzm[k], ysu_add(ysu_div(ysu_mul(ysu_add(ysu_div((-hgamu), hpbl()), ysu_div(ysu_sub(ux((k + 1)), ux(k)), dza[(k + 1)])), ysu_sub(ux((k + 1)), ux(k))), dza[(k + 1)]), ysu_div(ysu_mul(ysu_add(ysu_div((-hgamv), hpbl()), ysu_div(ysu_sub(vx((k + 1)), vx(k)), dza[(k + 1)])), ysu_sub(vx((k + 1)), vx(k))), dza[(k + 1)])));
            // MPAS line 1260: buoy_ysu(i,k)=xkzh(i,k)*g*(1.0/thx(i,k))*(-hgamt(i)/hpbl(i)+(thx(i,k+1)-thx(i,k))/dza(i,k+1))
            buoy_ysu[k] = ysu_mul(ysu_mul(ysu_mul(xkzh[k], g), ysu_div(0x1.0000000000000p+0f, thx[k])), ysu_add(ysu_div((-hgamt), hpbl()), ysu_div(ysu_sub(thx[(k + 1)], thx[k]), dza[(k + 1)])));
            // MPAS line 1262: zk = karman*zq(i,k+1)
            zk = ysu_mul(karman, zq[(k + 1)]);
            // MPAS line 1264: if (k.ge.kpbl(i)) then
            if ((k >= kpbl)) {
                // MPAS line 1265: rlamdz = min(max(0.1*dza(i,k+1),rlam),300.)
                rlamdz = ysu_minf(ysu_maxf(ysu_mul(0x1.99999a0000000p-4f, dza[(k + 1)]), rlam), 0x1.2c00000000000p+8f);
                // MPAS line 1266: rlamdz = min(dza(i,k+1),rlamdz)
                rlamdz = ysu_minf(dza[(k + 1)], rlamdz);
                // MPAS line 1267: else
            } else {
                // MPAS line 1269: rlamdz = 150.0
                rlamdz = 0x1.2c00000000000p+7f;
                // MPAS line 1270: endif
            }
            // MPAS line 1271: el_ysu(i,k) = zk*rlamdz/(rlamdz+zk)
            el_ysu[k] = ysu_div(ysu_mul(zk, rlamdz), ysu_add(rlamdz, zk));
            // MPAS line 1272: tke_ysu(i,k)=16.6*el_ysu(i,k)*(shear_ysu(i,k)-buoy_ysu(i,k))
            tke_ysu[k] = ysu_mul(ysu_mul(0x1.09999a0000000p+4f, el_ysu[k]), ysu_sub(shear_ysu[k], buoy_ysu[k]));
            // MPAS line 1274: if(tke_ysu(i,k).le.0) then
            if ((tke_ysu[k] <= 0)) {
                // MPAS line 1275: tke_ysu(i,k)=0.0
                tke_ysu[k] = 0x0.0p+0f;
                // MPAS line 1276: else
            } else {
                // MPAS line 1277: tke_ysu(i,k)=(tke_ysu(i,k))**0.66
                tke_ysu[k] = mpas_ysu_powf(tke_ysu[k], 0x1.51eb860000000p-1f);
                // MPAS line 1278: endif
            }
            // MPAS line 1279: enddo
        }
        // MPAS line 1284: do k = kts,kte
        for (k = kts; k <= kte; k += 1) {
            // MPAS line 1285: thvx_1d(k) = thvx(i,k)
            thvx_1d[k] = thvx[k];
            // MPAS line 1286: tke_1d(k)  = tke_ysu(i,k)
            tke_1d[k] = tke_ysu[k];
            // MPAS line 1287: zq_1d(k)   = zq(i,k)
            zq_1d[k] = zq[k];
            // MPAS line 1288: dzq_1d(k)  = dzq(i,k)
            dzq_1d[k] = dzq[k];
            // MPAS line 1289: enddo
        }
        // MPAS line 1290: zq_1d(kte+1) = zq(i,kte+1)
        zq_1d[(kte + 1)] = zq[(kte + 1)];
        // MPAS line 1291: call get_pblh(kts,kte,pblh_ysu(i),thvx_1d,tke_1d,zq_1d,dzq_1d,xland(i))
        get_pblh(kts, kte, pblh_ysu, thvx_1d, tke_1d, zq_1d, dzq_1d, xland(), workspace + local_slot * (kte + 2) * ws_stride, ws_stride, status);
        // MPAS line 1296: if (xland(i).lt.1.5) then
        if ((xland() < 0x1.8000000000000p+0f)) {
            // MPAS line 1297: fluxc = max(sflux(i),0.0)
            fluxc = ysu_maxf(sflux, 0x0.0p+0f);
            // MPAS line 1298: vconvc=1.
            vconvc = 0x1.0000000000000p+0f;
            // MPAS line 1299: vconv = vconvc*(g/thvx(i,1)*pblh_ysu(i)*fluxc)**.33
            vconv = ysu_mul(vconvc, mpas_ysu_powf(ysu_mul(ysu_mul(ysu_div(g, thvx[1]), pblh_ysu), fluxc), 0x1.51eb860000000p-2f));
            // MPAS line 1300: else
        } else {
            // MPAS line 1302: vconv = 0.
            vconv = 0x0.0p+0f;
            // MPAS line 1303: endif
        }
        // MPAS line 1304: vconvfx(i) = vconv
        vconvfx = vconv;
        // MPAS line 1307: fric(i,1)=ust(i)**2/wspd1(i)*rhox(i)*g/del(i,1)*dt2          *(wspd1(i)/wspd(i))**2
        fric[1] = ysu_mul(ysu_mul(ysu_div(ysu_mul(ysu_mul(ysu_div(ysu_mul(ust(), ust()), wspd1), rhox), g), del[1]), dt2), ysu_mul(ysu_div(wspd1, wspd()), ysu_div(wspd1, wspd())));
        // MPAS line 1309: if(present(ctopo)) then
        if ((ctopo.p != nullptr)) {
            // MPAS line 1310: vconvnew=0.9*vconvfx(i)+1.5*(max((pblh_ysu(i)-500)/1000.0,0.0))
            vconvnew = ysu_add(ysu_mul(0x1.cccccc0000000p-1f, vconvfx), ysu_mul(0x1.8000000000000p+0f, ysu_maxf(ysu_div(ysu_sub(pblh_ysu, 500), 0x1.f400000000000p+9f), 0x0.0p+0f)));
            // MPAS line 1311: vconvlim = min(vconvnew,1.0)
            vconvlim = ysu_minf(vconvnew, 0x1.0000000000000p+0f);
            // MPAS line 1312: ad(i,1) = 1.+fric(i,1)*vconvlim+ctopo(i)*fric(i,1)*(1-vconvlim)
            ad[1] = ysu_add(ysu_add(0x1.0000000000000p+0f, ysu_mul(fric[1], vconvlim)), ysu_mul(ysu_mul(ctopo(), fric[1]), ysu_sub(1, vconvlim)));
            // MPAS line 1313: ad(i,1) = ad(i,1) - bepswitch*frc_urb1d(i)*  (fric(i,1)*vconvlim+ctopo(i)*fric(i,1)*(1-vconvlim))
            ad[1] = ysu_sub(ad[1], ysu_mul(ysu_mul(bepswitch, frc_urb1d), ysu_add(ysu_mul(fric[1], vconvlim), ysu_mul(ysu_mul(ctopo(), fric[1]), ysu_sub(1, vconvlim)))));
            // MPAS line 1317: else
        } else {
            // MPAS line 1318: ad(i,1) = 1.+fric(i,1)
            ad[1] = ysu_add(0x1.0000000000000p+0f, fric[1]);
            // MPAS line 1319: endif
        }
        // MPAS line 1320: f1(i,1) = ux(i,1)+uoxl(i)*ust(i)**2*rhox(i)*g/del(i,1)*dt2/wspd1(i)*(wspd1(i)/wspd(i))**2
        f1[1] = ysu_add(ux(1), ysu_mul(ysu_div(ysu_mul(ysu_div(ysu_mul(ysu_mul(ysu_mul(uoxl, ysu_mul(ust(), ust())), rhox), g), del[1]), dt2), wspd1), ysu_mul(ysu_div(wspd1, wspd()), ysu_div(wspd1, wspd()))));
        // MPAS line 1321: f2(i,1) = vx(i,1)+voxl(i)*ust(i)**2*rhox(i)*g/del(i,1)*dt2/wspd1(i)*(wspd1(i)/wspd(i))**2
        f2[1] = ysu_add(vx(1), ysu_mul(ysu_div(ysu_mul(ysu_div(ysu_mul(ysu_mul(ysu_mul(voxl, ysu_mul(ust(), ust())), rhox), g), del[1]), dt2), wspd1), ysu_mul(ysu_div(wspd1, wspd()), ysu_div(wspd1, wspd()))));
        // MPAS line 1322: enddo
    }
    // MPAS line 1324: do k = kts,kte-1
    for (k = kts; k <= (kte - 1); k += 1) {
        // MPAS line 1325: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 1326: dtodsd = sfk2d(i,k)*dt2/del(i,k)
            dtodsd = ysu_div(ysu_mul(sfk2d[k], dt2), del[k]);
            // MPAS line 1327: dtodsu = sfk2d(i,k)*dt2/del(i,k+1)
            dtodsu = ysu_div(ysu_mul(sfk2d[k], dt2), del[(k + 1)]);
            // MPAS line 1328: dsig   = p2d(i,k)-p2d(i,k+1)
            dsig = ysu_sub(p2d(k), p2d((k + 1)));
            // MPAS line 1329: rdz    = 1./dza(i,k+1)
            rdz = ysu_div(0x1.0000000000000p+0f, dza[(k + 1)]);
            // MPAS line 1330: tem1   = dsig*xkzm(i,k)*rdz
            tem1 = ysu_mul(ysu_mul(dsig, xkzm[k]), rdz);
            // MPAS line 1331: if(pblflg(i).and.k.lt.kpbl(i))then
            if ((pblflg && (k < kpbl))) {
                // MPAS line 1332: dsdzu     = tem1*(-hgamu(i)/hpbl(i)-ufxpbl(i)*zfacent(i,k)/xkzm(i,k))
                dsdzu = ysu_mul(tem1, ysu_sub(ysu_div((-hgamu), hpbl()), ysu_div(ysu_mul(ufxpbl, zfacent[k]), xkzm[k])));
                // MPAS line 1333: dsdzv     = tem1*(-hgamv(i)/hpbl(i)-vfxpbl(i)*zfacent(i,k)/xkzm(i,k))
                dsdzv = ysu_mul(tem1, ysu_sub(ysu_div((-hgamv), hpbl()), ysu_div(ysu_mul(vfxpbl, zfacent[k]), xkzm[k])));
                // MPAS line 1334: f1(i,k)   = f1(i,k)+dtodsd*dsdzu
                f1[k] = ysu_add(f1[k], ysu_mul(dtodsd, dsdzu));
                // MPAS line 1335: f1(i,k+1) = ux(i,k+1)-dtodsu*dsdzu
                f1[(k + 1)] = ysu_sub(ux((k + 1)), ysu_mul(dtodsu, dsdzu));
                // MPAS line 1336: f2(i,k)   = f2(i,k)+dtodsd*dsdzv
                f2[k] = ysu_add(f2[k], ysu_mul(dtodsd, dsdzv));
                // MPAS line 1337: f2(i,k+1) = vx(i,k+1)-dtodsu*dsdzv
                f2[(k + 1)] = ysu_sub(vx((k + 1)), ysu_mul(dtodsu, dsdzv));
                // MPAS line 1338: elseif(pblflg(i).and.k.ge.kpbl(i).and.entfac(i,k).lt.4.6) then
            } else if (((pblflg && (k >= kpbl)) && (entfac[k] < 0x1.2666660000000p+2f))) {
                // MPAS line 1339: xkzm(i,k) = prpbl(i)*xkzh(i,k)
                xkzm[k] = ysu_mul(prpbl, xkzh[k]);
                // MPAS line 1340: xkzm(i,k) = sqrt(xkzm(i,k)*xkzml(i,k))
                xkzm[k] = mpas_ysu_sqrtf(ysu_mul(xkzm[k], xkzml[k]));
                // MPAS line 1341: xkzm(i,k) = max(xkzm(i,k),xkzom(i,k))
                xkzm[k] = ysu_maxf(xkzm[k], xkzom[k]);
                // MPAS line 1342: xkzm(i,k) = min(xkzm(i,k),xkzmax)
                xkzm[k] = ysu_minf(xkzm[k], xkzmax);
                // MPAS line 1343: f1(i,k+1) = ux(i,k+1)
                f1[(k + 1)] = ux((k + 1));
                // MPAS line 1344: f2(i,k+1) = vx(i,k+1)
                f2[(k + 1)] = vx((k + 1));
                // MPAS line 1345: else
            } else {
                // MPAS line 1346: f1(i,k+1) = ux(i,k+1)
                f1[(k + 1)] = ux((k + 1));
                // MPAS line 1347: f2(i,k+1) = vx(i,k+1)
                f2[(k + 1)] = vx((k + 1));
                // MPAS line 1348: endif
            }
            // MPAS line 1349: tem1   = dsig*xkzm(i,k)*rdz
            tem1 = ysu_mul(ysu_mul(dsig, xkzm[k]), rdz);
            // MPAS line 1350: dsdz2     = tem1*rdz
            dsdz2 = ysu_mul(tem1, rdz);
            // MPAS line 1351: au(i,k)   = -dtodsd*dsdz2/vlk2d(i,k)
            au[k] = ysu_div(ysu_mul((-dtodsd), dsdz2), vlk2d[k]);
            // MPAS line 1352: al(i,k)   = -dtodsu*dsdz2/vlk2d(i,k)
            al[k] = ysu_div(ysu_mul((-dtodsu), dsdz2), vlk2d[k]);
            // MPAS line 1353: ad(i,k)   = ad(i,k)-au(i,k)
            ad[k] = ysu_sub(ad[k], au[k]);
            // MPAS line 1354: ad(i,k+1) = 1.-al(i,k)
            ad[(k + 1)] = ysu_sub(0x1.0000000000000p+0f, al[k]);
            // MPAS line 1355: exch_mx(i,k+1) = xkzm(i,k)
            exch_mx((k + 1)) = xkzm[k];
            // MPAS line 1356: enddo
        }
        // MPAS line 1357: enddo
    }
    // MPAS line 1361: do k = kts,kte
    for (k = kts; k <= kte; k += 1) {
        // MPAS line 1362: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 1363: ad1(i,k) = ad(i,k)
            ad1[k] = ad[k];
            // MPAS line 1364: end do
        }
        // MPAS line 1365: end do
    }
    // MPAS line 1366: do k = kts,kte
    for (k = kts; k <= kte; k += 1) {
        // MPAS line 1367: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 1368: ad(i,k) = ad(i,k) - a_u2d(i,k)*dt2
            ad[k] = ysu_sub(ad[k], ysu_mul(a_u2d[k], dt2));
            // MPAS line 1369: ad1(i,k) = ad1(i,k) - a_v2d(i,k)*dt2
            ad1[k] = ysu_sub(ad1[k], ysu_mul(a_v2d[k], dt2));
            // MPAS line 1370: f1(i,k) = f1(i,k) + b_u2d(i,k)*dt2
            f1[k] = ysu_add(f1[k], ysu_mul(b_u2d[k], dt2));
            // MPAS line 1371: f2(i,k) = f2(i,k) + b_v2d(i,k)*dt2
            f2[k] = ysu_add(f2[k], ysu_mul(b_v2d[k], dt2));
            // MPAS line 1372: enddo
        }
        // MPAS line 1373: enddo
    }
    // MPAS line 1377: do k = kts,kte
    for (k = kts; k <= kte; k += 1) {
        // MPAS line 1378: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 1379: cu(i,k) = au(i,k)
            cu[k] = au[k];
            // MPAS line 1380: r1(i,k) = f1(i,k)
            r1[k] = f1[k];
            // MPAS line 1381: r2(i,k) = f2(i,k)
            r2[k] = f2[k];
            // MPAS line 1382: enddo
        }
        // MPAS line 1383: enddo
    }
    // MPAS line 1387: call tridi2n(al,ad,ad1,cu,r1,r2,au,f1,f2,its,ite,kts,kte,1)
    tridi2n(al, ad, ad1, cu, r1, r2, au, f1, f2, its, ite, kts, kte, 1, workspace + local_slot * (kte + 2) * ws_stride, ws_stride, status);
    // MPAS line 1391: do k = kte,kts,-1
    for (k = kte; k >= kts; k += -1) {
        // MPAS line 1392: do i = its,ite
        { // independent column: i = its = ite
            // MPAS line 1393: utend = (f1(i,k)-ux(i,k))*rdt
            utend = ysu_mul(ysu_sub(f1[k], ux(k)), rdt);
            // MPAS line 1394: vtend = (f2(i,k)-vx(i,k))*rdt
            vtend = ysu_mul(ysu_sub(f2[k], vx(k)), rdt);
            // MPAS line 1395: utnp(i,k) = utend
            utnp(k) = utend;
            // MPAS line 1396: vtnp(i,k) = vtend
            vtnp(k) = vtend;
            // MPAS line 1397: if(present(dusfc)) dusfc(i) = dusfc(i) + utend*conwrc*del(i,k)
            if ((dusfc.p != nullptr)) dusfc() = ysu_add(dusfc(), ysu_mul(ysu_mul(utend, conwrc), del[k]));
            // MPAS line 1398: if(present(dvsfc)) dvsfc(i) = dvsfc(i) + vtend*conwrc*del(i,k)
            if ((dvsfc.p != nullptr)) dvsfc() = ysu_add(dvsfc(), ysu_mul(ysu_mul(vtend, conwrc), del[k]));
            // MPAS line 1399: enddo
        }
        // MPAS line 1400: enddo
    }
    // MPAS line 1404: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 1405: if(present(ctopo).and.present(ctopo2)) then
        if (((ctopo.p != nullptr) && (ctopo2.p != nullptr))) {
            // MPAS line 1406: u10(i) = ctopo2(i)*u10(i)+(1-ctopo2(i))*ux(i,1)
            u10() = ysu_add(ysu_mul(ctopo2(), u10()), ysu_mul(ysu_sub(1, ctopo2()), ux(1)));
            // MPAS line 1407: v10(i) = ctopo2(i)*v10(i)+(1-ctopo2(i))*vx(i,1)
            v10() = ysu_add(ysu_mul(ctopo2(), v10()), ysu_mul(ysu_sub(1, ctopo2()), vx(1)));
            // MPAS line 1408: endif
        }
        // MPAS line 1409: enddo
    }
    // MPAS line 1413: do i = its,ite
    { // independent column: i = its = ite
        // MPAS line 1414: kpbl1d(i) = kpbl(i)
        kpbl1d() = kpbl;
        // MPAS line 1415: enddo
    }
    // MPAS line 1417: errmsg = 'bl_ysu_run ok'
    // MPAS line 1418: errflg = 0
}
'''

_CUDA_LAUNCH = r'''

// MPAS v8.4.2 physics_mmm/bl_ysu.F90:56-1420
extern "C" __global__ void ysu_v842_columns(
    const unsigned long long *arrays, const float *scalars, const int *flags,
    float *workspace, int *status, int ncol, int offset, int count) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count) return;
    int col = offset + lane;
    YsuArgs p{};
    p.dz8w2d = YsuView{(float*)arrays[0], col, ncol, flags[2]};
    p.pi2d = YsuView{(float*)arrays[1], col, ncol, flags[2]};
    p.tx = YsuView{(float*)arrays[2], col, ncol, flags[2]};
    p.qvx = YsuView{(float*)arrays[3], col, ncol, flags[2]};
    p.qcx = YsuView{(float*)arrays[4], col, ncol, flags[2]};
    p.qix = YsuView{(float*)arrays[5], col, ncol, flags[2]};
    p.qmix = YsuView{(float*)arrays[6], col, ncol, flags[2]};
    p.utnp = YsuView{(float*)arrays[7], col, ncol, flags[2]};
    p.vtnp = YsuView{(float*)arrays[8], col, ncol, flags[2]};
    p.ttnp = YsuView{(float*)arrays[9], col, ncol, flags[2]};
    p.qvtnp = YsuView{(float*)arrays[10], col, ncol, flags[2]};
    p.qctnp = YsuView{(float*)arrays[11], col, ncol, flags[2]};
    p.qitnp = YsuView{(float*)arrays[12], col, ncol, flags[2]};
    p.qmixtnp = YsuView{(float*)arrays[13], col, ncol, flags[2]};
    p.p2di = YsuView{(float*)arrays[14], col, ncol, flags[2]};
    p.p2d = YsuView{(float*)arrays[15], col, ncol, flags[2]};
    p.hpbl = YsuView{(float*)arrays[16], col, ncol, flags[2]};
    p.dusfc = YsuView{(float*)arrays[17], col, ncol, flags[2]};
    p.dvsfc = YsuView{(float*)arrays[18], col, ncol, flags[2]};
    p.dtsfc = YsuView{(float*)arrays[19], col, ncol, flags[2]};
    p.dqsfc = YsuView{(float*)arrays[20], col, ncol, flags[2]};
    p.ust = YsuView{(float*)arrays[21], col, ncol, flags[2]};
    p.znt = YsuView{(float*)arrays[22], col, ncol, flags[2]};
    p.xland = YsuView{(float*)arrays[23], col, ncol, flags[2]};
    p.hfx = YsuView{(float*)arrays[24], col, ncol, flags[2]};
    p.qfx = YsuView{(float*)arrays[25], col, ncol, flags[2]};
    p.wspd = YsuView{(float*)arrays[26], col, ncol, flags[2]};
    p.br = YsuView{(float*)arrays[27], col, ncol, flags[2]};
    p.psim = YsuView{(float*)arrays[28], col, ncol, flags[2]};
    p.psih = YsuView{(float*)arrays[29], col, ncol, flags[2]};
    p.psfcpa = YsuView{(float*)arrays[30], col, ncol, flags[2]};
    p.kpbl1d = YsuIntView{(int*)arrays[31], col, ncol, flags[2]};
    p.ux = YsuView{(float*)arrays[32], col, ncol, flags[2]};
    p.vx = YsuView{(float*)arrays[33], col, ncol, flags[2]};
    p.rthraten = YsuView{(float*)arrays[34], col, ncol, flags[2]};
    p.ctopo = YsuView{(float*)arrays[35], col, ncol, flags[2]};
    p.ctopo2 = YsuView{(float*)arrays[36], col, ncol, flags[2]};
    p.a_u = YsuView{(float*)arrays[37], col, ncol, flags[2]};
    p.a_v = YsuView{(float*)arrays[38], col, ncol, flags[2]};
    p.a_t = YsuView{(float*)arrays[39], col, ncol, flags[2]};
    p.a_q = YsuView{(float*)arrays[40], col, ncol, flags[2]};
    p.a_e = YsuView{(float*)arrays[41], col, ncol, flags[2]};
    p.b_u = YsuView{(float*)arrays[42], col, ncol, flags[2]};
    p.b_v = YsuView{(float*)arrays[43], col, ncol, flags[2]};
    p.b_t = YsuView{(float*)arrays[44], col, ncol, flags[2]};
    p.b_q = YsuView{(float*)arrays[45], col, ncol, flags[2]};
    p.b_e = YsuView{(float*)arrays[46], col, ncol, flags[2]};
    p.sfk = YsuView{(float*)arrays[47], col, ncol, flags[2]};
    p.vlk = YsuView{(float*)arrays[48], col, ncol, flags[2]};
    p.dlu = YsuView{(float*)arrays[49], col, ncol, flags[2]};
    p.dlg = YsuView{(float*)arrays[50], col, ncol, flags[2]};
    p.frcurb = YsuView{(float*)arrays[51], col, ncol, flags[2]};
    p.exch_hx = YsuView{(float*)arrays[52], col, ncol, flags[2]};
    p.exch_mx = YsuView{(float*)arrays[53], col, ncol, flags[2]};
    p.u10 = YsuView{(float*)arrays[54], col, ncol, flags[2]};
    p.v10 = YsuView{(float*)arrays[55], col, ncol, flags[2]};
    p.uox = YsuView{(float*)arrays[56], col, ncol, flags[2]};
    p.vox = YsuView{(float*)arrays[57], col, ncol, flags[2]};
    p.wstar = YsuView{(float*)arrays[58], col, ncol, flags[2]};
    p.delta = YsuView{(float*)arrays[59], col, ncol, flags[2]};
    p.dt = scalars[0];
    p.cp = scalars[1];
    p.g = scalars[2];
    p.rovcp = scalars[3];
    p.rovg = scalars[4];
    p.rd = scalars[5];
    p.xlv = scalars[6];
    p.rv = scalars[7];
    p.ep1 = scalars[8];
    p.ep2 = scalars[9];
    p.karman = scalars[10];
    p.its = flags[0];
    p.ite = flags[1];
    p.kte = flags[2];
    p.kme = flags[3];
    p.ysu_topdown_pblmix = flags[4];
    p.nmix = flags[5];
    p.f_qc = flags[6];
    p.f_qi = flags[7];
    p.flag_bep = flags[8];
    bl_ysu_run(p, workspace + lane, count, status + col);
}
'''

_CONVERSIONS = r'''
// MPAS v8.4.2 mpas_atmphys_driver_pbl.F:354-375
extern "C" __global__ void ysu_pbl_from_mpas(
    const float *lw, const float *sw, float *rthraten, int size) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < size) rthraten[q] = __fadd_rn(lw[q], sw[q]);
}
// MPAS v8.4.2 physics_wrf/module_bl_ysu.F:441-449
extern "C" __global__ void ysu_temperature_to_theta(
    const float *ttnp, const float *pi, float *rthblten, int size) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < size) rthblten[q] = __fdiv_rn(ttnp[q], pi[q]);
}
'''

_REQUIRED_LEVEL = ("ux", "vx", "tx", "qvx", "qcx", "qix", "p2d", "pi2d",
                   "dz8w2d", "rthraten")
_REQUIRED_SURFACE = ("psfcpa", "znt", "ust", "psim", "psih", "xland", "hfx",
                     "qfx", "wspd", "br", "u10", "v10")
_OUTPUT_LEVEL = ("utnp", "vtnp", "ttnp", "qvtnp", "qctnp", "qitnp",
                 "exch_hx", "exch_mx")
_OUTPUT_SURFACE = ("hpbl", "wstar", "delta", "kpbl1d")
_OPTIONAL_SURFACE = ("dusfc", "dvsfc", "dtsfc", "dqsfc", "ctopo", "ctopo2",
                     "uox", "vox", "frcurb")
_OPTIONAL_LEVEL = ("a_u", "a_v", "a_t", "a_q", "a_e", "b_u", "b_v", "b_t",
                   "b_q", "b_e", "sfk", "vlk", "dlu", "dlg")



def bl_ysu_init() -> tuple[str, int]:
    """MPAS v8.4.2 physics_mmm/bl_ysu.F90:20-32."""
    return "bl_ysu_init OK", 0


def bl_ysu_finalize() -> tuple[str, int]:
    """MPAS v8.4.2 physics_mmm/bl_ysu.F90:38-50."""
    return "bl_ysu_finalize OK", 0


@lru_cache(maxsize=1)
def shared_device_math_header() -> str:
    """Math calls in physics_mmm/bl_ysu.F90:56-1420 use shared float32 math.

    The pinned MYNN header supplies expf/powf and the shared SFIRE header
    supplies glibc 2.43 CORE-MATH tanhf. These pins identify the tested math
    implementation independently of the untouched WOOF YSU backend.
    """
    pins = {
        "mynn_libm.cuh": "bf5831efbdf1ddc1111e8f0bafc2e272b5597d7d2b12d8320675b8a938618fbb",
        "sfire_libm.cuh": "22f44144a81d60fbbd767cfcb33d260e4d1f08fdc759d962165bae70bb603ee3",
    }
    headers = []
    root = files("woof.core.kernels")
    for name, expected in pins.items():
        data = root.joinpath(name).read_bytes()
        if sha256(data).hexdigest() != expected:
            raise ValueError(f"Shared {name} changed; its YSU word-identity "
                             "receipt no longer identifies the device math")
        headers.append(data.decode("utf-8"))
    return "typedef float real;\n" + "\n".join(headers) + r'''
#define mpas_ysu_powf mynn_glibc_powf
#define mpas_ysu_expf mynn_glibc_expf
#define mpas_ysu_tanhf sfire_tanhf
#define mpas_ysu_sqrtf __fsqrt_rn
#define mpas_ysu_absf fabsf
'''


def cuda_source(device_math_header: str) -> str:
    """Compose physics_mmm/bl_ysu.F90:56-1420 with shared glibc routines.

    The header supplies mpas_ysu_powf/expf/tanhf/sqrtf/absf aliases to its
    shared implementation. There is no CUDA-libm fallback: that would change
    the reference's powf/exponential rounding without a residue witness.
    """
    if not device_math_header.strip():
        raise ValueError("YSU requires the shared MPAS glibc 2.43 device math "
                         "header; CUDA libm would invalidate word identity")
    return device_math_header + "\n" + _CUDA_BODY + _CUDA_LAUNCH + _CONVERSIONS


def validated_cuda_binary(path: Path, device_math_header: str, architecture: str) -> bytes:
    """Load compiled physics_mmm/bl_ysu.F90:56-1420 on the lane's run host.

    Source and option checks prevent a stale artifact or forced FTZ build
    from entering the numerical claim. The normal launcher still compiles
    directly through NVRTC when no artifact is configured.
    """
    metadata = json.loads(path.with_suffix('.json').read_text())
    expected = {
        'schema': 'mpas-ysu-v842/compiled-kernel-v1',
        'cuda_sha256': CUDA_SHA256,
        'source_sha256': sha256(cuda_source(device_math_header).encode()).hexdigest(),
        'device_math_sha256': sha256(device_math_header.encode()).hexdigest(),
        'compiler_options': list(_hex_numerics.options(COMPILER_OPTIONS)) + ['--gpu-architecture=sm_' + architecture],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'YSU compiled CUDA artifact differs from current strict source/options: {key}')
    binary = path.read_bytes()
    if sha256(binary).hexdigest() != metadata.get('binary_sha256'):
        raise ValueError('YSU compiled CUDA artifact binary SHA256 differs')
    return binary


# Unbounded on purpose (one entry per card and variant): a bounded per-card cache thrashes once ranks x
# variants exceed it, and every rank then rebuilds its module each step (thompson_v842._module, 8 cards,
# 2026-10-28: 1.28 s/step against 0.25 s with the cache fixed).
@lru_cache(maxsize=None)
def _kernels(device_math_header: str, device_id: int):
    """Load scheme and conversions; physics_mmm/bl_ysu.F90:56-1420;
    mpas_atmphys_driver_pbl.F:309-420; physics_wrf/module_bl_ysu.F:450-461.
    """
    import cupy as cp

    # CuPy's RawModule cache appends -ftz=true after caller options. The
    # direct NVRTC entry preserves --ftz=false, including BR=1 subnormal.
    from cupy.cuda import compiler, function

    with cp.cuda.Device(device_id):
        artifact = os.environ.get('HEX_MPAS_YSU_CUDA_ARTIFACT')
        if artifact:
            binary = validated_cuda_binary(Path(artifact), device_math_header,
                                           cp.cuda.Device(device_id).compute_capability)
        else:
            binary, _ = compiler.compile_using_nvrtc(
                cuda_source(device_math_header), options=_hex_numerics.options(COMPILER_OPTIONS),
                cache_in_memory=True)
        module = function.Module()
        module.load(binary)
        return tuple(module.get_function(name) for name in (
            "ysu_v842_columns", "ysu_pbl_from_mpas", "ysu_temperature_to_theta"))


def _check_array(name: str, array: Any, shape: tuple[int, ...], dtype: Any, cp):
    """Check declared storage; physics_wrf/module_bl_ysu.F:16-237."""
    if (not isinstance(array, cp.ndarray) or array.shape != shape
            or array.dtype != dtype or not array.flags.c_contiguous):
        raise ValueError(f"{name} must be a C-contiguous CuPy {dtype} array "
                         f"with shape {shape}")
    if array.device.id != cp.cuda.Device().id:
        raise ValueError(f"{name} is on a different CUDA device")


#: Byte budget of one YSU launch's column workspace (the WSM6 launcher's budget).
WORKSPACE_BUDGET_MIB = 1536


def workspace_tile(nz: int, ncol: int) -> int:
    """Columns per YSU launch: every column at once, capped by the workspace byte budget, whole warps.

    THE BREAKAGE THIS ENDS: a fixed 128-column tile launched YSU as 4 blocks of 32 threads per launch, so a
    149,322-cell 96 -> 3 km zoom globe ran 1,167 launches per step one after another (about 1.8 ms each)
    with the card nearly idle, and mesoscale_reference stepped 4.1x slower than convection_permitting there
    (V8X6000 nsys, 2026-10-29). Each column's arithmetic is the same at any tile: the tile only sets the
    workspace stride (bl_ysu_run(p, workspace + lane, count, ...)). WOOF_HEX_YSU_WORKSPACE_MIB overrides.
    """
    import os
    value = os.environ.get("WOOF_HEX_YSU_WORKSPACE_MIB", "").strip().lower()
    mib = WORKSPACE_BUDGET_MIB if value in ("", "auto") else int(value)
    if mib < 1:
        raise ValueError("WOOF_HEX_YSU_WORKSPACE_MIB must be at least 1")
    column_bytes = WORKSPACE_ARRAYS * (nz + 2) * 4
    budget = max(32, ((mib << 20) // column_bytes) // 32 * 32)
    return max(1, min(-(-ncol // 32) * 32, budget))


def bl_ysu_run(arrays: Mapping[str, Any], constants: Mapping[str, float], *,
               dt: float, f_qc: bool, f_qi: bool,
               ysu_topdown_pblmix: bool, device_math_header: str,
               flag_bep: bool = False, workspace_columns: int | None = None
               ) -> dict[str, Any]:
    """MPAS v8.4.2 physics_mmm/bl_ysu.F90:56-1420.

    Level arrays are (nz, ncol), interfaces (nz+1, ncol), and tracer arrays
    (nmix, nz, ncol), all C-contiguous float32. Surface arrays are (ncol,).
    These layouts retain MPAS's contiguous horizontal direction on device.
    Output arrays can be supplied to preserve inactive-species carriers;
    otherwise they are zero-initialized. u10/v10 are native INOUT arguments.
    An optional argument is absent only when it is missing or None.
    """
    import cupy as cp

    if not device_math_header.strip():
        cuda_source(device_math_header)  # fail before creating a CUDA context
    tx = arrays["tx"]
    if len(tx.shape) != 2:
        raise ValueError("tx must have shape (nz, ncol)")
    nz, ncol = tx.shape
    if workspace_columns is None:
        workspace_columns = workspace_tile(nz, ncol)
    if type(workspace_columns) is not int or workspace_columns <= 0:
        raise ValueError("workspace_columns must be a positive integer")
    state = {name: arrays.get(name) for name in ARRAY_ARGUMENTS}
    for name in _REQUIRED_LEVEL:
        _check_array(name, state[name], (nz, ncol), cp.float32, cp)
    _check_array("p2di", state["p2di"], (nz+1, ncol), cp.float32, cp)
    for name in _REQUIRED_SURFACE:
        _check_array(name, state[name], (ncol,), cp.float32, cp)
    for name in _OPTIONAL_LEVEL:
        if state[name] is not None:
            _check_array(name, state[name], (nz, ncol), cp.float32, cp)
    for name in _OPTIONAL_SURFACE:
        if state[name] is not None:
            _check_array(name, state[name], (ncol,), cp.float32, cp)
    for name in _OUTPUT_LEVEL + _OUTPUT_SURFACE:
        shape = (nz, ncol) if name in _OUTPUT_LEVEL else (ncol,)
        dtype = cp.int32 if name == "kpbl1d" else cp.float32
        if state[name] is None:
            state[name] = cp.zeros(shape, dtype=dtype)
        else:
            _check_array(name, state[name], shape, dtype, cp)
    if state["qmix"] is None:
        nmix = 0
        state["qmix"] = cp.empty((0, nz, ncol), cp.float32)
    else:
        if len(state["qmix"].shape) != 3:
            raise ValueError("qmix must have shape (nmix, nz, ncol)")
        nmix = state["qmix"].shape[0]
        _check_array("qmix", state["qmix"], (nmix, nz, ncol), cp.float32, cp)
    if state["qmixtnp"] is None:
        state["qmixtnp"] = cp.zeros_like(state["qmix"])
    else:
        _check_array("qmixtnp", state["qmixtnp"], (nmix, nz, ncol), cp.float32, cp)
    scalar_values = {**constants, "dt": dt}
    scalars = cp.asarray([scalar_values[name] for name in FLOAT_ARGUMENTS], cp.float32)
    flags = dict(its=1, ite=1, kte=nz, kme=nz+1, nmix=nmix, f_qc=f_qc,
                 f_qi=f_qi, ysu_topdown_pblmix=ysu_topdown_pblmix,
                 flag_bep=flag_bep)
    integers = cp.asarray([flags[name] for name in INTEGER_ARGUMENTS], cp.int32)
    pointers = cp.asarray([0 if state[name] is None else state[name].data.ptr
                           for name in ARRAY_ARGUMENTS], cp.uint64)
    count = min(workspace_columns, ncol)
    status = cp.zeros(ncol, cp.int32)
    if count:
        workspace = cp.empty(WORKSPACE_ARRAYS * (nz+2) * count, cp.float32)
        kernel = _kernels(device_math_header, cp.cuda.Device().id)[0]
        for offset in range(0, ncol, count):
            tile = min(count, ncol-offset)
            kernel(((tile+31)//32,), (32,),
                   (pointers, scalars, integers, workspace, status,
                    np.int32(ncol), np.int32(offset), np.int32(tile)))
    result = {name: state[name] for name in _OUTPUT_LEVEL + _OUTPUT_SURFACE}
    result["qmixtnp"] = state["qmixtnp"]
    result.update({name: state[name] for name in ("u10", "v10")})
    result.update({name: state[name] for name in ("dusfc", "dvsfc", "dtsfc", "dqsfc")
                   if state[name] is not None})
    return result


_WRAPPER_FIELDS = {
    "u3d": "ux", "v3d": "vx", "t3d": "tx", "qv3d": "qvx",
    "qc3d": "qcx", "qi3d": "qix", "p3d": "p2d", "p3di": "p2di",
    "pi3d": "pi2d", "dz8w": "dz8w2d", "psfc": "psfcpa", "uoce": "uox",
    "voce": "vox", "a_u_bep": "a_u", "a_v_bep": "a_v", "a_t_bep": "a_t",
    "a_q_bep": "a_q", "a_e_bep": "a_e", "b_u_bep": "b_u", "b_v_bep": "b_v",
    "b_t_bep": "b_t", "b_q_bep": "b_q", "b_e_bep": "b_e", "sf_bep": "sfk",
    "vl_bep": "vlk", "dl_u_bep": "dlu", "dlg_bep": "dlg", "frc_urb2d": "frcurb",
}
_WRAPPER_OUTPUTS = {
    "rublten": "utnp", "rvblten": "vtnp", "rqvblten": "qvtnp",
    "rqcblten": "qctnp", "rqiblten": "qitnp", "exch_h": "exch_hx",
    "exch_m": "exch_mx", "hpbl": "hpbl", "kpbl2d": "kpbl1d",
    "wstar": "wstar", "delta": "delta", "u10": "u10", "v10": "v10",
}


def ysu(arrays: Mapping[str, Any], constants: Mapping[str, float], *, dt: float,
        flag_qc: bool, flag_qi: bool, ysu_topdown_pblmix: int,
        device_math_header: str, flag_bep: bool = False) -> dict[str, Any]:
    """MPAS v8.4.2 physics_wrf/module_bl_ysu.F:16-475.

    Horizontal/vertical copies are views in the device column layout. The
    wrapper has nmix=0, maps integer top-down flag by equality to one, and
    divides ttnp by Exner after the B4B temperature-tendency multiplication.
    """
    state = {name: value for name, value in arrays.items() if name in ARRAY_ARGUMENTS}
    state.update({_WRAPPER_FIELDS[name]: value for name, value in arrays.items()
                  if name in _WRAPPER_FIELDS})
    import cupy as cp

    # Native wrapper local arrays are not initialized when the species is
    # inactive (module_bl_ysu.F:304-310,450-461). Allocate corresponding local
    # storage without assigning a numerical substitute. MPAS's driver selects
    # both species for bl_ysu (Registry.xml:1705-1715; init.F:501-515).
    if not flag_qc:
        state["qctnp"] = cp.empty_like(state["tx"])
    if not flag_qi:
        state["qitnp"] = cp.empty_like(state["tx"])
    state.pop("qmix", None)  # wrapper's nmix=0 (module_bl_ysu.F:238)
    state.pop("qmixtnp", None)
    result = bl_ysu_run(state, constants, dt=dt, f_qc=flag_qc, f_qi=flag_qi,
                        ysu_topdown_pblmix=(ysu_topdown_pblmix == 1),
                        flag_bep=flag_bep, device_math_header=device_math_header)
    out = {name: result[field] for name, field in _WRAPPER_OUTPUTS.items()}
    out["rthblten"] = cp.empty_like(result["ttnp"])
    size = result["ttnp"].size
    if size:
        _kernels(device_math_header, cp.cuda.Device().id)[2](
            ((size+255)//256,), (256,),
            (result["ttnp"], state["pi2d"], out["rthblten"], np.int32(size)))
    return out


def pbl_from_MPAS(carriers: Mapping[str, Any], *,
                  diag_physics: Mapping[str, Any], sfc_input: Mapping[str, Any],
                  tend_physics: Mapping[str, Any], device_math_header: str
                  ) -> dict[str, Any]:
    """MPAS v8.4.2 mpas_atmphys_driver_pbl.F:309-420.

    General physics carriers (pres_hyd_p, t_p, etc.) are already prepared by
    the common driver. This slice reads surface-layer fields and adds the
    separate LW and SW potential-temperature rates in native order.
    """
    import cupy as cp

    fields = {"p3d": "pres_hyd_p", "p3di": "pres2_hyd_p", "t3d": "t_p",
              "dz8w": "dz_p", "pi3d": "pi_p", "u3d": "u_p", "v3d": "v_p",
              "qv3d": "qv_p", "qc3d": "qc_p", "qi3d": "qi_p", "psfc": "psfc_p"}
    # MPAS's prepared profiles have kme storage, while this driver copies
    # kts:kte (mpas_atmphys_driver_pbl.F:372-378). The pool tendencies retain
    # exactly those active levels; only p3di includes the upper interface.
    nz, ncol = tend_physics["rthratenlw"].shape
    state = {name: carriers[field] if name == "psfc" else
             carriers[field][:nz + int(name == "p3di")]
             for name, field in fields.items()}
    for name in ("znt", "ust", "hfx", "qfx", "wspd", "br", "u10", "v10"):
        state[name] = diag_physics[name].copy() if name in ("u10", "v10") else diag_physics[name]
    state["psim"] = diag_physics["fm"]
    state["psih"] = diag_physics["fh"]
    state["xland"] = sfc_input["xland"]
    for name in ("rthratenlw", "rthratensw"):
        _check_array(name, tend_physics[name], (nz, ncol), cp.float32, cp)
    state["rthraten"] = cp.empty((nz, ncol), cp.float32)
    state["ctopo"] = cp.ones(ncol, cp.float32)
    state["ctopo2"] = cp.ones(ncol, cp.float32)
    state["uoce"] = cp.zeros(ncol, cp.float32)
    state["voce"] = cp.zeros(ncol, cp.float32)
    size = nz*ncol
    if size:
        _kernels(device_math_header, cp.cuda.Device().id)[1](
            ((size+255)//256,), (256,),
            (tend_physics["rthratenlw"], tend_physics["rthratensw"],
             state["rthraten"], np.int32(size)))
    return state


def pbl_to_MPAS(result: Mapping[str, Any], *, diag_physics: Mapping[str, Any],
                tend_physics: Mapping[str, Any]) -> None:
    """MPAS v8.4.2 mpas_atmphys_driver_pbl.F:555-633."""
    import cupy as cp

    for name in ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten", "rqiblten"):
        _check_array(name, tend_physics[name], result[name].shape, result[name].dtype, cp)
        cp.copyto(tend_physics[name], result[name])
    for name, field in (("hpbl", "hpbl"), ("kpbl", "kpbl2d"), ("kzh", "exch_h"),
                        ("kzm", "exch_m"), ("kzq", "exch_h"), ("delta", "delta"),
                        ("wstar", "wstar")):
        _check_array(name, diag_physics[name], result[field].shape, result[field].dtype, cp)
        cp.copyto(diag_physics[name], result[field])
    # The native driver writes exch_p, initialized to zero at lines 412-416;
    # ysu's exch_h argument is kzh_p, so exch_p is never modified by YSU.
    _check_array("exch_h", diag_physics["exch_h"], result["exch_h"].shape,
                 cp.float32, cp)
    diag_physics["exch_h"].fill(0)


def driver_pbl(carriers: Mapping[str, Any], *, diag_physics: Mapping[str, Any],
               sfc_input: Mapping[str, Any], tend_physics: Mapping[str, Any],
               constants: Mapping[str, float], dt: float,
               config_ysu_pblmix: bool, f_qc: bool, f_qi: bool,
               device_math_header: str) -> dict[str, Any]:
    """MPAS v8.4.2 mpas_atmphys_driver_pbl.F:805-852."""
    state = pbl_from_MPAS(carriers, diag_physics=diag_physics,
                          sfc_input=sfc_input, tend_physics=tend_physics,
                          device_math_header=device_math_header)
    result = ysu(state, constants, dt=dt, flag_qc=f_qc, flag_qi=f_qi,
                 ysu_topdown_pblmix=int(config_ysu_pblmix), flag_bep=False,
                 device_math_header=device_math_header)
    pbl_to_MPAS(result, diag_physics=diag_physics, tend_physics=tend_physics)
    return result


def launch_slot(call: Any) -> dict[str, Any]:
    """Prepared f4 slot boundary; mpas_atmphys_driver_pbl.F:309-420,555-633,805-852.

    The common adapter owns the solve-cell views and commits changed pool
    storage after this callback. Constants come from the compiled native
    provider, including expressions such as rdg and rcp.
    """
    pools = call.pools
    prepared = pools["prepared"]
    carrier_names = ("pres_hyd_p", "pres2_hyd_p", "t_p", "dz_p", "pi_p",
                     "u_p", "v_p", "qv_p", "qc_p", "qi_p", "psfc_p")
    aliases = {"g": "gravity", "rovcp": "rcp", "rovg": "rdg",
               "rd": "R_d", "rv": "R_v", "ep1": "ep_1", "ep2": "ep_2"}
    constant_names = ("cp", "g", "rovcp", "rovg", "rd", "xlv", "rv",
                      "ep1", "ep2", "karman")
    return driver_pbl(
        {name: prepared[name] for name in carrier_names},
        diag_physics=pools["diag_physics"], sfc_input=pools["sfc_input"],
        tend_physics=pools["tend_physics"],
        constants={name: pools["constants"][aliases.get(name, name)]
                   for name in constant_names},
        dt=call.dt, config_ysu_pblmix=call.options["config_ysu_pblmix"],
        f_qc=bool(pools["scalar_state"]["f_qc"]),
        f_qi=bool(pools["scalar_state"]["f_qi"]),
        device_math_header=shared_device_math_header())


def register(registry: Any, *, get_carriers: Any, device_math_header: str) -> None:
    """Register mpas_atmphys_driver_pbl.F:805-852 for config_pbl_scheme=bl_ysu.

    get_carriers(call) is the common driver's adapter returning carriers,
    diag_physics, sfc_input, tend_physics, constants, f_qc and f_qi keyword
    arguments for driver_pbl. The slot retains config_ysu_pblmix and dt_pbl.
    """
    def invoke(call):
        """Native YSU dispatch; mpas_atmphys_driver_pbl.F:823-852."""
        return driver_pbl(**get_carriers(call), dt=call.dt,
                          config_ysu_pblmix=call.options["config_ysu_pblmix"],
                          device_math_header=device_math_header)

    registry.register(SLOT, SCHEME, BACKEND, invoke)


CUDA_SHA256 = sha256((_CUDA_BODY + _CUDA_LAUNCH + _CONVERSIONS).encode()).hexdigest()
