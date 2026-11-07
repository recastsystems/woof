"""Fast GBM scheduling; the canonical strict translation remains untouched.

WRF module_bl_gbmpbl.F:843-863 solves two right hand sides with identical
coefficients at :622-623 and :696-697. Factor once and keep the previous
forward/backward word in a register. Each right hand side retains its own
original operation order, including the correctly rounded reciprocal.
The equations derive only from WRF's public-domain module (see its pin in
tools/gbmpbl_wrf461_oracle/source-pin.json).
"""

_SOLVERS = r"""
// WRF tridag, with the loop-carried value retained instead of reloaded.
__device__ void gb_tridag(const GbCol &a, const GbCol &b, const GbCol &c,
                         const GbCol &r, const GbCol &u, const GbCol &gam,
                         int nn) {
    real rbet = GB_LITERAL_7 / b[1];
    real previous = r[1] * rbet;
    u[1] = previous;
    for (int j = 2; j <= nn; ++j) {
        const real gamma = c[j - 1] * rbet;
        gam[j] = gamma;
        const real aj = a[j];
        rbet = GB_LITERAL_7 / (b[j] - aj * gamma);
        previous = (r[j] - aj * previous) * rbet;
        u[j] = previous;
    }
    for (int j = nn - 1; j >= 1; --j) {
        previous = u[j] - gam[j + 1] * previous;
        u[j] = previous;
    }
}

// The same factorization, applied to both independent right hand sides.
__device__ void gb_tridag_pair(const GbWork &w, int nn, bool ice = false) {
    real rbet = GB_LITERAL_7 / w.bimp[1];
    real previous1 = w.rimp1[1] * rbet;
    real previous2 = w.rimp2[1] * rbet;
    real previous3 = ice ? w.qix[1] * rbet : GB_LITERAL_2;
    if (ice) w.qix[1] = previous3;
    w.uimp1[1] = previous1;
    w.uimp2[1] = previous2;
    for (int j = 2; j <= nn; ++j) {
        const real gamma = w.cimp[j - 1] * rbet;
        w.gam[j] = gamma;
        const real aj = w.aimp[j];
        rbet = GB_LITERAL_7 / (w.bimp[j] - aj * gamma);
        previous1 = (w.rimp1[j] - aj * previous1) * rbet;
        previous2 = (w.rimp2[j] - aj * previous2) * rbet;
        if (ice) {
            previous3 = (w.qix[j] - aj * previous3) * rbet;
            w.qix[j] = previous3;
        }
        w.uimp1[j] = previous1;
        w.uimp2[j] = previous2;
    }
    for (int j = nn - 1; j >= 1; --j) {
        const real gamma = w.gam[j + 1];
        previous1 = w.uimp1[j] - gamma * previous1;
        previous2 = w.uimp2[j] - gamma * previous2;
        if (ice) {
            previous3 = w.qix[j] - gamma * previous3;
            w.qix[j] = previous3;
        }
        w.uimp1[j] = previous1;
        w.uimp2[j] = previous2;
    }
}

"""

_WARP_SOLVERS = r"""
// Serial Thomas arithmetic with rows gathered from warp registers. The
// recurrence stays in WRF order; coalesced shared loads happen once per
// 32-row tile instead of between dependent elimination instructions.
__device__ void gb_tridag(const GbCol &a, const GbCol &b, const GbCol &c,
                         const GbCol &r, const GbCol &u, const GbCol &gam,
                         int nn) {
    const int lane = threadIdx.x;
    real rbet = GB_LITERAL_2, previous = GB_LITERAL_2;
    for (int base = 1; base <= nn; base += 32) {
        const int k = base + lane;
        const real av = k <= nn ? a[k] : GB_LITERAL_2;
        const real bv = k <= nn ? b[k] : GB_LITERAL_7;
        const real cv = k > 1 && k <= nn ? c[k-1] : GB_LITERAL_2;
        const real rv = k <= nn ? r[k] : GB_LITERAL_2;
        real uv = GB_LITERAL_2, gv = GB_LITERAL_2;
        for (int j = 0; j < 32 && base+j <= nn; ++j) {
            const real aj = __shfl_sync(0xffffffff, av, j);
            const real bj = __shfl_sync(0xffffffff, bv, j);
            const real cj = __shfl_sync(0xffffffff, cv, j);
            const real rj = __shfl_sync(0xffffffff, rv, j);
            const real gamma = cj * rbet;
            rbet = GB_LITERAL_7 / (base+j == 1 ? bj : bj - aj * gamma);
            previous = (base+j == 1 ? rj : rj - aj * previous) * rbet;
            if (lane == j) {uv = previous; gv = gamma;}
        }
        if (k <= nn) {u[k] = uv; if (k > 1) gam[k] = gv;}
        __syncwarp();
    }
    for (int base = ((nn-1)/32)*32+1; base >= 1; base -= 32) {
        const int k = base+lane;
        const real uv = k <= nn ? u[k] : GB_LITERAL_2;
        const real gv = k < nn ? gam[k+1] : GB_LITERAL_2;
        real output = uv;
        for (int j = min(31,nn-base); j >= 0; --j) {
            const real uj = __shfl_sync(0xffffffff, uv, j);
            const real gj = __shfl_sync(0xffffffff, gv, j);
            if (base+j < nn) previous = uj - gj * previous;
            if (lane == j) output = previous;
        }
        if (k <= nn) u[k] = output;
        __syncwarp();
    }
}

__device__ void gb_tridag_pair(const GbWork &w, int nn, bool ice = false) {
    const int lane = threadIdx.x;
    real rbet = GB_LITERAL_2, previous1 = GB_LITERAL_2;
    real previous2 = GB_LITERAL_2, previous3 = GB_LITERAL_2;
    for (int base = 1; base <= nn; base += 32) {
        const int k = base+lane;
        const real av = k <= nn ? w.aimp[k] : GB_LITERAL_2;
        const real bv = k <= nn ? w.bimp[k] : GB_LITERAL_7;
        const real cv = k > 1 && k <= nn ? w.cimp[k-1] : GB_LITERAL_2;
        const real rv1 = k <= nn ? w.rimp1[k] : GB_LITERAL_2;
        const real rv2 = k <= nn ? w.rimp2[k] : GB_LITERAL_2;
        const real rv3 = ice && k <= nn ? w.qix[k] : GB_LITERAL_2;
        real uv1 = GB_LITERAL_2, uv2 = GB_LITERAL_2;
        real uv3 = GB_LITERAL_2, gv = GB_LITERAL_2;
        for (int j = 0; j < 32 && base+j <= nn; ++j) {
            const real aj = __shfl_sync(0xffffffff, av, j);
            const real bj = __shfl_sync(0xffffffff, bv, j);
            const real cj = __shfl_sync(0xffffffff, cv, j);
            const real rj1 = __shfl_sync(0xffffffff, rv1, j);
            const real rj2 = __shfl_sync(0xffffffff, rv2, j);
            const real rj3 = __shfl_sync(0xffffffff, rv3, j);
            const real gamma = cj * rbet;
            rbet = GB_LITERAL_7 / (base+j == 1 ? bj : bj - aj * gamma);
            previous1 = (base+j == 1 ? rj1 : rj1 - aj * previous1) * rbet;
            previous2 = (base+j == 1 ? rj2 : rj2 - aj * previous2) * rbet;
            if (ice) previous3 = (base+j == 1 ? rj3 : rj3 - aj * previous3) * rbet;
            if (lane == j) {uv1=previous1;uv2=previous2;uv3=previous3;gv=gamma;}
        }
        if (k <= nn) {
            w.uimp1[k]=uv1;w.uimp2[k]=uv2;
            if (ice) w.qix[k]=uv3;
            if (k > 1) w.gam[k]=gv;
        }
        __syncwarp();
    }
    for (int base = ((nn-1)/32)*32+1; base >= 1; base -= 32) {
        const int k = base+lane;
        const real uv1 = k <= nn ? w.uimp1[k] : GB_LITERAL_2;
        const real uv2 = k <= nn ? w.uimp2[k] : GB_LITERAL_2;
        const real uv3 = ice && k <= nn ? w.qix[k] : GB_LITERAL_2;
        const real gv = k < nn ? w.gam[k+1] : GB_LITERAL_2;
        real output1=uv1,output2=uv2,output3=uv3;
        for (int j=min(31,nn-base); j>=0; --j) {
            const real uj1=__shfl_sync(0xffffffff,uv1,j);
            const real uj2=__shfl_sync(0xffffffff,uv2,j);
            const real uj3=__shfl_sync(0xffffffff,uv3,j);
            const real gj=__shfl_sync(0xffffffff,gv,j);
            if (base+j < nn) {
                previous1=uj1-gj*previous1;previous2=uj2-gj*previous2;
                if (ice) previous3=uj3-gj*previous3;
            }
            if (lane==j) {output1=previous1;output2=previous2;output3=previous3;}
        }
        if (k <= nn) {
            w.uimp1[k]=output1;w.uimp2[k]=output2;
            if (ice) w.qix[k]=output3;
        }
        __syncwarp();
    }
}

"""

COOPERATIVE_SLOTS = 38


def _compact_workspace(source: str) -> str:
    """Reuse disjoint live ranges in the cooperative column's shared storage.

    The solver loads a complete row tile before storing its forward values,
    so its RHS/solution and diagonal/gamma may share storage. MY stability
    functions and initial virtual temperature die before RHS assembly. The
    first pass's ice is unpublished, and the second pass has no TKE solve.
    """
    aliases = dict(sm="rimp1", sh="rimp2", tvx="rimp2", shear="rimp2",
                   bouyan="qix", rcldb="bimp", gam="bimp",
                   uimp1="rimp1", uimp2="rimp2",
                   ktop_save="rimp1", kbot="rimp2")
    start = source.index('__device__ __forceinline__ GbWork gb_work(')
    end = source.index('__device__ void gb_tridag(', start)
    # List the original layout explicitly; a changed canonical roster must
    # fail here rather than silently omit a workspace member.
    import re
    names = re.findall(r'&w\.(\w+)', source[start:end])
    if len(names) != 46 or len(set(names)) != 46:
        raise RuntimeError('GBM shared workspace roster changed')
    names += ['ktop', 'kbot', 'ktop_save']
    slots = {name: i for i, name in enumerate(n for n in names if n not in aliases)}
    if len(slots) != COOPERATIVE_SLOTS:
        raise RuntimeError('GBM compact shared workspace count changed')
    body = '__device__ __forceinline__ GbWork gb_work(real *base, int kp) {\n    GbWork w;\n'
    for name in names:
        body += f'    w.{name}.p = GBWS_AT(base, {slots[aliases.get(name, name)]}, kp);\n'
    body += '    return w;\n}\n\n'
    return source[:start] + body + source[end:]



def paired_source(source: str) -> str:
    """Apply the checked, GBM-only fast scheduling transformation."""
    start = source.index('__device__ void gb_tridag(')
    end = source.index('// n2 (', start)
    source = source[:start] + _SOLVERS + source[end:]
    first = 'gb_tridag(w.aimp, w.bimp, w.cimp, w.rimp1, w.uimp1, w.gam, kte);'
    second = 'gb_tridag(w.aimp, w.bimp, w.cimp, w.rimp2, w.uimp2, w.gam, kte);'
    for indent in ('        ', '    '):
        pair = first + '\n' + indent + second
        if source.count(pair) != 1:
            raise RuntimeError('GBM paired-solver source boundary changed')
        source = source.replace(pair, 'gb_tridag_pair(w, kte);', 1)
    return source


def _replace(source, old, new, count=1):
    if source.count(old) != count:
        raise RuntimeError('GBM cooperative source boundary changed: ' + old[:70])
    return source.replace(old, new)


def _loop(source, marker, old, new):
    """Distribute one explicitly reviewed independent loop, then reconverge."""
    start = source.index(marker)
    at = source.index(old, start)
    body = source.index('{', at)
    depth = 1
    end = body + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    source = source[:end] + '\n    __syncwarp();' + source[end:]
    return source[:at] + source[at:].replace(old, new, 1)


def warp_source(source: str) -> str:
    """One warp per column; independent levels execute in parallel.

    Shared arrays keep all serial recurrences on chip. PBL layer merging,
    height accumulation and Thomas elimination retain their serial order.
    Every cross-level producer completes before its consumers start.
    """
    source = paired_source(source)
    source = _replace(source, '#define GBWS_LANES 32', '#define GBWS_LANES 1')
    # Every shuffle participant must supply an initialized input, even though
    # only lane zero's PBL-height result is selected by the broadcast.
    source = _replace(source, '    real pblx;', '    real pblx = GB_LITERAL_2;')
    source = _loop(source, '__device__ void gb_n2(',
                   'for (int k = 2; k <= kte; ++k)',
                   'for (int k = 2 + threadIdx.x; k <= kte; k += 32)')
    source = _replace(source, '    w.nsquar[1] = w.nsquar[2];',
                      '    if (threadIdx.x == 0) w.nsquar[1] = w.nsquar[2];\n    __syncwarp();')
    # MY's kethl reads the next level's new kzm: separate that producer loop.
    source = _replace(source,
        '        w.kethl[k] = GB_NUK * sqrtf(w.kzm[k] * w.kzm[k + 1]);\n'
        '        w.kethl[k] = fminf(w.kethl[k], GB_KMAX);', '')
    source = _loop(source, '__device__ void gb_my(',
                   'for (int k = kte; k >= 2; --k)',
                   'for (int k = kte - threadIdx.x; k >= 2; k -= 32)')
    at = source.index('    for (int ilay = 1;', source.index('__device__ void gb_my('))
    end = source.index('\n}\n', at)
    source = (source[:at] + '''    for (int k = 2 + threadIdx.x; k <= kte; k += 32) {
        w.kethl[k] = GB_NUK * sqrtf(w.kzm[k] * w.kzm[k + 1]);
        w.kethl[k] = fminf(w.kethl[k], GB_KMAX);
    }
    __syncwarp();
    if (threadIdx.x == 0) {
''' + source[at:end] + '\n    }\n    __syncwarp();' + source[end:])
    # Only lane zero initializes endpoint words in MY.
    at = source.index('    w.kzm[kte + 1]', source.index('__device__ void gb_my('))
    end = source.index('    for (int k = kte -', at)
    source = source[:at] + '    if (threadIdx.x == 0) {\n' + source[at:end] + '    }\n    __syncwarp();\n' + source[end:]
    source = _loop(source, '__device__ __forceinline__ void gb_halfcoef(',
                   'for (int k = 1; k <= kte; ++k)',
                   'for (int k = 1 + threadIdx.x; k <= kte; k += 32)')
    # The cumulative heights retain WRF's summation order in lane zero.
    geom = '''        if (pass == 1) {
            w.zqx[k] = w.zqx[k + 1] + GG(dz8w, kr);
            w.zax[k] = GB_LITERAL_5 * (w.zqx[k] + w.zqx[k + 1]);
        }
'''
    source = _replace(source, geom, '')
    start = source.index('    w.zqx[kte + 1]', source.index('__device__ void gb_gbm2d('))
    end = source.index('    const real *tke_src', start)
    source = source[:start] + '''    if (threadIdx.x == 0) {
        w.zqx[kte + 1] = GB_LITERAL_2;
        w.zqx[kte + 2] = GB_LITERAL_2;
        w.tke[kte + 1] = GB_LITERAL_2;
        if (pass == 1) for (int k = kte; k >= 1; --k) {
            const int kr = kte + 1 - k;
            w.zqx[k] = w.zqx[k + 1] + GG(dz8w, kr);
            w.zax[k] = GB_LITERAL_5 * (w.zqx[k] + w.zqx[k + 1]);
        }
    }
    __syncwarp();
''' + source[end:]
    source = _loop(source, '    const real *tke_src',
                   'for (int k = kte; k >= 1; --k)',
                   'for (int k = kte - threadIdx.x; k >= 1; k -= 32)')
    source = _loop(source, '    real tvcon =',
                   'for (int k = 1; k <= kte; ++k)',
                   'for (int k = 1 + threadIdx.x; k <= kte; k += 32)')
    source = _replace(source, '    w.presfl[1] = GB_LITERAL_2;',
                      '    tvcon = (GB_LITERAL_7 + GB_EP_1 * w.qx[kte]) - w.qcx[kte];\n'
                      '    if (threadIdx.x == 0) w.presfl[1] = GB_LITERAL_2;')
    source = _loop(source, '    tvcon = (GB_LITERAL_7',
                   'for (int k = 2; k <= kte; ++k)',
                   'for (int k = 2 + threadIdx.x; k <= kte; k += 32)')
    start = source.index('    w.presfl[kte + 1] = psfc;')
    end = source.index('    const real q0s', start)
    source = source[:start] + '    if (threadIdx.x == 0) {\n' + source[start:end] + '    }\n    __syncwarp();\n' + source[end:]
    source = _replace(source,
        '    w.nsquar[kte + 1] = ((GB_G / w.thvx[kte]) * dthv) / w.zax[kte];',
        '    if (threadIdx.x == 0) w.nsquar[kte + 1] = ((GB_G / w.thvx[kte]) * dthv) / w.zax[kte];\n    __syncwarp();')
    source = _replace(source,
        '    const int iconv = gb_pblhgt(w, xlvocp, kte, &pblx, &kpbl2dx);',
        '''    // Stable length scales are independent of the layer merge.
    for (int k = 2 + threadIdx.x; k <= kte + 1; k += 32)
        if (w.nsquar[k] > GB_LITERAL_2)
            w.bbls[k] = fminf(GB_RSTBL * sqrtf(w.tke[k] / w.nsquar[k]),
                              GB_KARMAN * w.zqx[k]);
    __syncwarp();
    int iconv = 0;
    if (threadIdx.x == 0) iconv = gb_pblhgt(w, xlvocp, kte, &pblx, &kpbl2dx);
    __syncwarp();
    iconv = __shfl_sync(0xffffffff, iconv, 0);
    pblx = __shfl_sync(0xffffffff, pblx, 0);
    kpbl2dx = __shfl_sync(0xffffffff, kpbl2dx, 0);''')
    source = _loop(source, '    for (int ip = 1;',
                   'for (int k = 1; k <= kte; ++k)',
                   'for (int k = 1 + threadIdx.x; k <= kte; k += 32)')
    source = _replace(source,
        '        w.nsquar[kte + 1] = ((GB_G / thvx_t) * dthv_t) / w.zax[kte];',
        '        if (threadIdx.x == 0) w.nsquar[kte + 1] = ((GB_G / thvx_t) * dthv_t) / w.zax[kte];\n        __syncwarp();')
    source = _loop(source, '// Update the scalars',
                   'for (int k = 1; k <= kte; ++k)',
                   'for (int k = 1 + threadIdx.x; k <= kte; k += 32)')
    source = _loop(source, '// ---8---',
                   'for (int k = 1; k <= kte; ++k)',
                   'for (int k = 1 + threadIdx.x; k <= kte; k += 32)')
    source = _replace(source, '''        w.ux[k] = w.uimp1[k];
        w.vx[k] = w.uimp2[k];
        if (k >= 2) {''', '''        w.ux[k] = w.uimp1[k];
        w.vx[k] = w.uimp2[k];
    }
    __syncwarp();
    for (int k = 1 + threadIdx.x; k <= kte; k += 32) {
        if (k >= 2) {''')
    source = _loop(source, '    gb_tridag_pair(w, kte);\n    for',
                   'for (int k = 1; k <= kte; ++k)',
                   'for (int k = 1 + threadIdx.x; k <= kte; k += 32)')
    source = _replace(source, 'for (int k = 1; k <= kte; ++k) w.rimp1[k] = w.qix[k];',
        'for (int k = 1 + threadIdx.x; k <= kte; k += 32) w.rimp1[k] = w.qix[k];\n    __syncwarp();')
    source = _loop(source, '// ---10---',
                   'for (int k = 2; k <= kte; ++k)',
                   'for (int k = 2 + threadIdx.x; k <= kte; k += 32)')
    start = source.index('    for (int ilay = 1;', source.index('// ---10---'))
    end = source.index('    for (int k = 2;', start)
    source = source[:start] + '    if (threadIdx.x == 0) {\n' + source[start:end] + '    }\n    __syncwarp();\n' + source[end:]
    source = _loop(source, '    w.tke[kte + 1] = (GB_LITERAL_16',
                   'for (int k = 2; k <= kte; ++k)',
                   'for (int k = 2 + threadIdx.x; k <= kte; k += 32)')
    source = _replace(source, 'for (int k = 2; k <= kte; ++k)\n        w.tke[k] = fmaxf(w.uimp1[k - 1], GB_LITERAL_0);',
        'for (int k = 2 + threadIdx.x; k <= kte; k += 32)\n        w.tke[k] = fmaxf(w.uimp1[k - 1], GB_LITERAL_0);\n    __syncwarp();')
    source = _loop(source, '// ---11---',
                   'for (int k = 1; k <= kte; ++k)',
                   'for (int k = 1 + threadIdx.x; k <= kte; k += 32)')
    source = _replace(source, '    if (pass == 2) {\n        pblh[col]',
                      '    if (pass == 2 && threadIdx.x == 0) {\n        pblh[col]')
    start = source.index('    const long long local =', source.index('void gbmpbl_column('))
    end = source.index('    const GbWork w =', start)
    source = source[:start] + '''    const long long colll = col0 + blockIdx.x;
    if (colll >= ncol) return;
    const size_t col = (size_t)colll;
    const size_t n = (size_t)ncol;
    extern __shared__ real shared_ws[];
    real *base = shared_ws;
''' + source[end:]
    start = source.index('__device__ void gb_tridag(')
    end = source.index('// n2 (', start)
    source = source[:start] + _WARP_SOLVERS + source[end:]
    # Ice does not enter GBM's other equations. Its first-pass solution is
    # discarded; the second-pass equation has exactly the coefficients of
    # the second scalar solve, so use a third RHS in that factorization.
    source = _replace(source, '        gb_tridag_pair(w, kte);',
                      '        gb_tridag_pair(w, kte, pass == 2 && ip == 2);')
    start = source.index('    // ---9--- implicit diffusion of cloud ice')
    end = source.index('    // ---10--- prediction of TKE', start)
    source = source[:start] + source[end:]
    # Only first-pass TKE is published by the WRF GBMPBL wrapper. The second
    # pass's TKE solve has no consumer; its bbls[1] endpoint is still output.
    start = source.index('    // ---10--- prediction of TKE')
    end = source.index('    // ---11--- tendencies for output', start)
    source = (source[:start] + '    if (pass == 1) {\n' + source[start:end]
              + '    }\n    if (threadIdx.x == 0) w.bbls[1] = GB_LITERAL_2;\n'
                '    __syncwarp();\n' + source[end:])
    source = _replace(source, '''            istabl = 1;
            w.bbls[k] = fminf(GB_RSTBL * sqrtf(w.tke[k] / w.nsquar[k]),
                              GB_KARMAN * w.zqx[k]);''', '            istabl = 1;')
    # Pass one's final shared TKE is exactly the state just written to the
    # carried global array; pass two can consume those words in place.
    source = _replace(source, '        w.tke[k] = GG(tke_src, kr);',
                      '        if (pass == 1) w.tke[k] = GG(tke_src, kr);')
    # Copy-in and thermodynamic preparation have the same independent row
    # domain. Fuse them so the copied moisture stays live in registers.
    first = source.index('    const real *tke_src =')
    second = source.index('    real tvcon = GB_LITERAL_2;', first)
    third = source.index('\n    tvcon = (GB_LITERAL_7', second) + 1
    old = source[second:third]
    opening = '    for (int k = 1 + threadIdx.x; k <= kte; k += 32) {\n        const int kr = kte + 1 - k;'
    if old.count(opening) != 1 or not old.endswith('    __syncwarp();\n'):
        raise RuntimeError('GBM thermodynamic fusion boundary changed')
    body = old.replace('    real tvcon = GB_LITERAL_2;\n' + opening,
                       '        real tvcon = GB_LITERAL_2;', 1)
    closing = source.rfind('    }\n    __syncwarp();\n', first, second)
    source = (source[:closing] + body + '    real tvcon;\n' + source[third:])
    # The last buoyancy update feeds only the discarded second-pass TKE
    # prediction. The published diffusivities were computed before it.
    start = source.index('        gb_n2(w.uimp1, w.uimp2,')
    end = source.index('\n    }', start)
    source = (source[:start] + '        if (pass == 1 || ip == 1) {\n'
              + source[start:end] + '\n        }' + source[end:])
    # Gather thicknesses before the dependency chain. Addition order and the
    # half-level sum are unchanged, but height accumulation no longer issues
    # one dependent global load per level from a single lane.
    source = _replace(source, '''        if (pass == 1) for (int k = kte; k >= 1; --k) {
            const int kr = kte + 1 - k;
            w.zqx[k] = w.zqx[k + 1] + GG(dz8w, kr);
            w.zax[k] = GB_LITERAL_5 * (w.zqx[k] + w.zqx[k + 1]);
        }
    }
    __syncwarp();''', '''    }
    __syncwarp();
    if (pass == 1) {
        real height = GB_LITERAL_2;
        for (int base = kte; base >= 1; base -= 32) {
            const int k = base - threadIdx.x;
            const int kr = kte + 1 - k;
            const real dz = k >= 1 ? GG(dz8w, kr) : GB_LITERAL_2;
            real full = GB_LITERAL_2, half = GB_LITERAL_2;
            for (int j = 0; j < min(32, base); ++j) {
                const real previous = height;
                height = previous + __shfl_sync(0xffffffff, dz, j);
                if (threadIdx.x == j) {
                    full = height;
                    half = GB_LITERAL_5 * (height + previous);
                }
            }
            if (k >= 1) {w.zqx[k] = full; w.zax[k] = half;}
        }
        __syncwarp();
    }''')
    # Pass one publishes only momentum and TKE. Saturation adjustment feeds
    # its TKE radiation term only at convective layer tops; its other scalar
    # results have no consumer. Pass two still adjusts every level.
    at = source.index('    for (int k = 1 + threadIdx.x;', source.index('// Update the scalars'))
    body = source.index('{', at) + 1
    source = source[:body] + '''
        if (pass == 1) {
            bool needed = false;
            for (int ilay = 1; ilay <= iconv; ++ilay)
                needed |= k == w.ktop.get(ilay);
            if (!needed) continue;
        }
''' + source[body:]
    return _compact_workspace(source)
