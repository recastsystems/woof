"""Selectable RRFS-MPAS MYNN-SFC v6.1.0 (NCAR/MYNN-SFC a038a9e), complete MPAS-linked CUDA scheme.

The ``mynn-v6.1`` suite version (``config_mynn_version`` in the hex backend namelist group) selects this module in
place of the pinned v6.0.0 port (sf_mynnsfc_v600, RRFS 44c0849 / MYNN-SFC b66187f), which stays the default and is
not modified. The CUDA source is the v6.0.0 source plus the asserted v6.0.0 -> v6.1.0 edits below, each citing the
v6.1.0 line it implements; with the edits removed the text is the v6.0.0 source exactly. Upstream changes covered:
- f0c1839: cold-start reset of pblh/hfx/qfx/mol only when neither restart nor cycling (was: unless both).
- fca16e4: the step-1 land qsfc blend tabs=0.99*tskin+0.01*t_1 is removed (and the equivalent ice branch).
- 1142f95: psi-table index guard nzol>=0 in the six unstable lookups.
- 3185e36, 8ff4597, c322acc: water winds relative to the ocean current (uoce=voce=0 on the MPAS host, which passes
  none), the stress vector taux/tauy, earth-relative u10/v10 (0+x makes a -0.0 word +0.0, as native does).
- c047ae6: vegtype is read when vegtype is present (was: when ztpert is present).
- flag_iter is no longer an argument (the former .true. path); cqs is intent(out), its input read is removed.
OPTIONS and FIELDS keep the v6.0.0 order, so every host glue path is shared; the flag_iter option slot is ignored.

MYNN-SFC v6.1.0: Copyright 2026 NOAA, CU/CIRES, UCAR/NCAR; Apache License 2.0 (NCAR/MYNN-SFC LICENSE @a038a9e).
"""
from __future__ import annotations

import re
import sys
from types import FunctionType

from . import sf_mynnsfc_v600 as _pinned
from .sf_mynnsfc_v600 import (SCHEME, BACKEND, FIELDS, OPTIONS, OUTPUTS, OPTIONAL, MPAS_PRESENT,
                              _DIAG_INPUT, _SURFACE, _WRITE, _DIAG_NAMES, _BLEND, _SEA_HANDOFF,
                              DRIVER_SOURCE, _driver_source)
from woof.backend import compilation_target as _compilation_target
from woof.hex import numerics as _hex_numerics
from importlib.resources import files
import numpy as np

VERSION = 'mynn-v6.1'
MYNN_SFC = 'a038a9ea35a12f0bf879fac8b64c99a77ffdae1c'

_EDITS = (
    ('m_land_mynnsfc_land', 1,
     'if (flag_iter) {\nif ((itimestep==1)) {\nif (((lsm==lsm_ruc)&&(qsfc>0x0.0p+0f))) {\nqsfcmr = __fdiv_rn(qsfc,__fsub_rn(0x1.0000000000000p+0f,qsfc));\n} else {\ntabs = __fadd_rn(__fmul_rn(0x1.fae1480000000p-1f,tskin),__fmul_rn(0x1.47ae140000000p-7f,t_1));\ne1 = __fmul_rn(0x1.38ef340000000p-1f,SL_EXP(__fdiv_rn(__fmul_rn(0x1.1ab8520000000p+4f,__fsub_rn(tabs,0x1.1126660000000p+8f)),__fsub_rn(tabs,0x1.da66660000000p+4f))));\nqsfc = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(psfc,__fmul_rn(0x1.8353dc0000000p-2f,e1)));\nqsfcmr = __fdiv_rn(qsfc,__fsub_rn(0x1.0000000000000p+0f,qsfc));\n}\n} else {\nif (((lsm==lsm_ruc)&&(qsfc>0x0.0p+0f))) {\nqsfcmr = __fdiv_rn(qsfc,__fsub_rn(0x1.0000000000000p+0f,qsfc));\n} else {\ntabs = tskin;\ne1 = __fmul_rn(0x1.38ef340000000p-1f,SL_EXP(__fdiv_rn(__fmul_rn(0x1.1ab8520000000p+4f,__fsub_rn(tabs,0x1.1126660000000p+8f)),__fsub_rn(tabs,0x1.da66660000000p+4f))));\nqsfc = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(psfc,__fmul_rn(0x1.8353dc0000000p-2f,e1)));\nqsfcmr = __fdiv_rn(qsfc,__fsub_rn(0x1.0000000000000p+0f,qsfc));\n}\n}\n}\n',
     'if (flag_iter) {\n// MYNN-SFC v6.1.0 src/module_sf_mynnsfc_land.F90:289-302 @a038a9e (fca16e4: step-1 tabs blend removed)\nif (((lsm==lsm_ruc)&&(qsfc>0x0.0p+0f))) {\nqsfcmr = __fdiv_rn(qsfc,__fsub_rn(0x1.0000000000000p+0f,qsfc));\n} else {\ntabs = tskin;\ne1 = __fmul_rn(0x1.38ef340000000p-1f,SL_EXP(__fdiv_rn(__fmul_rn(0x1.1ab8520000000p+4f,__fsub_rn(tabs,0x1.1126660000000p+8f)),__fsub_rn(tabs,0x1.da66660000000p+4f))));\nqsfc = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(psfc,__fmul_rn(0x1.8353dc0000000p-2f,e1)));\nqsfcmr = __fdiv_rn(qsfc,__fsub_rn(0x1.0000000000000p+0f,qsfc));\n}\n}\n'),
    ('m_ice_mynnsfc_ice', 1,
     'if (flag_iter) {\nif ((itimestep==1)) {\ntsk = tskin;\nif (((lsm==lsm_ruc)&&(qsfc>0x0.0p+0f))) {\nqsfcmr = __fdiv_rn(qsfc,__fsub_rn(0x1.0000000000000p+0f,qsfc));\n} else {\ne1 = __fmul_rn(0x1.38ef340000000p-1f,SL_EXP(__fadd_rn(__fsub_rn(__fmul_rn(0x1.2280000000000p+12f,__fsub_rn(0x1.dfda840000000p-9f,__fdiv_rn(0x1.0000000000000p+0f,tsk))),__fmul_rn(0x1.747ae20000000p+3f,SL_LOG(__fdiv_rn(0x1.1126660000000p+8f,tsk)))),__fmul_rn(0x1.7318fc0000000p-6f,__fsub_rn(0x1.1126660000000p+8f,tsk)))));\nqsfc = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(psfc,__fmul_rn(0x1.8353dc0000000p-2f,e1)));\nqsfcmr = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(psfc,e1));\n}\n} else {\nif (((lsm==lsm_ruc)&&(qsfc>0x0.0p+0f))) {\nqsfcmr = __fdiv_rn(qsfc,__fsub_rn(0x1.0000000000000p+0f,qsfc));\n} else {\ne1 = __fmul_rn(0x1.38ef340000000p-1f,SL_EXP(__fadd_rn(__fsub_rn(__fmul_rn(0x1.2280000000000p+12f,__fsub_rn(0x1.dfda840000000p-9f,__fdiv_rn(0x1.0000000000000p+0f,tskin))),__fmul_rn(0x1.747ae20000000p+3f,SL_LOG(__fdiv_rn(0x1.1126660000000p+8f,tskin)))),__fmul_rn(0x1.7318fc0000000p-6f,__fsub_rn(0x1.1126660000000p+8f,tskin)))));\nqsfc = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(psfc,__fmul_rn(0x1.8353dc0000000p-2f,e1)));\nqsfcmr = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(psfc,e1));\n}\n}\n}\n',
     'if (flag_iter) {\n// MYNN-SFC v6.1.0 src/module_sf_mynnsfc_ice.F90:276-286 @a038a9e (fca16e4: step-1 branch removed)\nif (((lsm==lsm_ruc)&&(qsfc>0x0.0p+0f))) {\nqsfcmr = __fdiv_rn(qsfc,__fsub_rn(0x1.0000000000000p+0f,qsfc));\n} else {\ne1 = __fmul_rn(0x1.38ef340000000p-1f,SL_EXP(__fadd_rn(__fsub_rn(__fmul_rn(0x1.2280000000000p+12f,__fsub_rn(0x1.dfda840000000p-9f,__fdiv_rn(0x1.0000000000000p+0f,tskin))),__fmul_rn(0x1.747ae20000000p+3f,SL_LOG(__fdiv_rn(0x1.1126660000000p+8f,tskin)))),__fmul_rn(0x1.7318fc0000000p-6f,__fsub_rn(0x1.1126660000000p+8f,tskin)))));\nqsfc = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(psfc,__fmul_rn(0x1.8353dc0000000p-2f,e1)));\nqsfcmr = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(psfc,e1));\n}\n}\n'),
    (None, 6,
     'nzol = (int)(__fmul_rn((-zolf),0x1.9000000000000p+6f));\nrzol = __fsub_rn(__fmul_rn((-zolf),0x1.9000000000000p+6f),nzol);\nif (((nzol+1)<1000)) {\n',
     'nzol = (int)(__fmul_rn((-zolf),0x1.9000000000000p+6f));\nrzol = __fsub_rn(__fmul_rn((-zolf),0x1.9000000000000p+6f),nzol);\n// MYNN-SFC v6.1.0 src/module_sf_mynnsfc_{land,water,ice}.F90:1680/1700,2081/2101,1482/1502 @a038a9e (1142f95)\nif (((nzol>=0)&&((nzol+1)<1000))) {\n'),
    (None, 2,
     'float lakemask, float wat_depth, ',
     'float lakemask, float wat_depth, float uoce, float voce, float &taux, float &tauy, '),
    ('m_water_mynnsfc_water', 1,
     'wspd = SL_SQRT(__fadd_rn(__fmul_rn(u_1,u_1),__fmul_rn(v_1,v_1)));\n',
     '// MYNN-SFC v6.1.0 src/module_sf_mynnsfc_water.F90:351 @a038a9e\nwspd = SL_SQRT(__fadd_rn(__fmul_rn(__fsub_rn(u_1,uoce),__fsub_rn(u_1,uoce)),__fmul_rn(__fsub_rn(v_1,voce),__fsub_rn(v_1,voce))));\n'),
    ('m_water_mynnsfc_water', 1,
     'wspdi = fmaxf(SL_SQRT(__fadd_rn(__fmul_rn(u_1,u_1),__fmul_rn(v_1,v_1))),0x1.99999a0000000p-4f);\n',
     '// MYNN-SFC v6.1.0 src/module_sf_mynnsfc_water.F90:682 @a038a9e\nwspdi = fmaxf(SL_SQRT(__fadd_rn(__fmul_rn(__fsub_rn(u_1,uoce),__fsub_rn(u_1,uoce)),__fmul_rn(__fsub_rn(v_1,voce),__fsub_rn(v_1,voce)))),0x1.99999a0000000p-4f);\n'),
    ('m_water_mynnsfc_water', 1,
     'cm = 0x0.0p+0f;\n',
     'cm = 0x0.0p+0f;\n// MYNN-SFC v6.1.0 src/module_sf_mynnsfc_water.F90:727-728 @a038a9e\ntaux = 0x0.0p+0f;\ntauy = 0x0.0p+0f;\n'),
    ('m_water_mynnsfc_water', 1,
     'cm = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix),__fdiv_rn(0x1.99999a0000000p-2f,psix));\n}\n}\nif (compute_diag) {\n',
     'cm = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix),__fdiv_rn(0x1.99999a0000000p-2f,psix));\n// MYNN-SFC v6.1.0 src/module_sf_mynnsfc_water.F90:810-811 @a038a9e\ntaux = (-__fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(rho_1,ust),ust),__fsub_rn(u_1,uoce)),wspdi));\ntauy = (-__fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(rho_1,ust),ust),__fsub_rn(v_1,voce)),wspdi));\n}\n}\nif (compute_diag) {\n'),
    ('m_water_mynnsfc_water', 2,
     'u10 = __fdiv_rn(__fmul_rn(u_1,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));\n',
     'u10 = __fadd_rn(uoce,__fdiv_rn(__fmul_rn(__fsub_rn(u_1,uoce),SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch))));\n'),
    ('m_water_mynnsfc_water', 1,
     'u10 = __fdiv_rn(__fmul_rn(u_1,psix10),psix);\n',
     'u10 = __fadd_rn(uoce,__fdiv_rn(__fmul_rn(__fsub_rn(u_1,uoce),psix10),psix));\n'),
    ('m_water_mynnsfc_water', 2,
     'v10 = __fdiv_rn(__fmul_rn(v_1,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));\n',
     'v10 = __fadd_rn(voce,__fdiv_rn(__fmul_rn(__fsub_rn(v_1,voce),SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch))));\n'),
    ('m_water_mynnsfc_water', 1,
     'v10 = __fdiv_rn(__fmul_rn(v_1,psix10),psix);\n',
     'v10 = __fadd_rn(voce,__fdiv_rn(__fmul_rn(__fsub_rn(v_1,voce),psix10),psix));\n'),
    ('m_drv_mynnsfc_driver', 1,
     'float ztpert_1;\n',
     'float ztpert_1;\n// MYNN-SFC v6.1.0 src/module_sf_mynnsfc_driver.F90:379 @a038a9e\nfloat uoce_1;\nfloat voce_1;\nfloat taux_1;\nfloat tauy_1;\n'),
    ('m_drv_mynnsfc_driver', 1,
     'if (((itimestep==1)&&((!restart)||(!loc_cycle)))) {\n',
     '// MYNN-SFC v6.1.0 src/module_sf_mynnsfc_driver.F90:419-420,430 @a038a9e (f0c1839: cold start only when neither restart nor cycling; loc_iter=1, iter absent)\nif ((((itimestep==1)&&(iter==1))&&(!(restart||loc_cycle)))) {\n'),
    ('m_drv_mynnsfc_driver', 1,
     'cqs_1 = cqs;\n',
     '// MYNN-SFC v6.1.0 src/module_sf_mynnsfc_driver.F90:526 @a038a9e (cqs is intent(out); the input read is removed)\n'),
    ('m_drv_mynnsfc_driver', 1,
     'wstar_1 = 0x0.0p+0f;\n',
     'wstar_1 = 0x0.0p+0f;\n// MYNN-SFC v6.1.0 src/module_sf_mynnsfc_driver.F90:511-514,544-545 @a038a9e (uoce/voce absent on the MPAS host)\nuoce_1 = 0x0.0p+0f;\nvoce_1 = 0x0.0p+0f;\ntaux_1 = 0x0.0p+0f;\ntauy_1 = 0x0.0p+0f;\n'),
    ('m_drv_mynnsfc_driver', 1,
     'loc_iter = true;\nif (has_flag_iter) loc_iter = flag_iter;\n',
     '// MYNN-SFC v6.1.0 src/module_sf_mynnsfc_driver.F90:257-259 @a038a9e (flag_iter removed; the schemes run the former flag_iter=.true. path)\nloc_iter = true;\n'),
    ('m_drv_mynnsfc_driver', 1,
     'if (has_ztpert) {\nvegtype_1 = vegtype;\n',
     '// MYNN-SFC v6.1.0 src/module_sf_mynnsfc_driver.F90:573 @a038a9e (c047ae6)\nif (has_vegtype) {\nvegtype_1 = vegtype;\n'),
    ('m_drv_mynnsfc_driver', 1,
     'lakemask_1,wat_depth_1,u_1,',
     'lakemask_1,wat_depth_1,uoce_1,voce_1,taux_1,tauy_1,u_1,'),
)


def _span(text, name):
    match = re.search(r'__device__[^\n]*\b' + re.escape(name) + r'\([^\n]*\)\s*\{', text)
    if match is None:
        raise ValueError('MYNN-SFC v6.1.0 lost function ' + name)
    following = text.find('\n__device__', match.end())
    return match.end(), len(text) if following < 0 else following


def _apply(text):
    """The v6.0.0 source with each v6.1.0 edit applied; every anchor must occur exactly as counted."""
    for scope, count, old, new in _EDITS:
        start, end = _span(text, scope) if scope else (0, len(text))
        found = text[start:end].count(old)
        if found != count:
            raise ValueError(f'MYNN-SFC v6.1.0 edit anchor {old[:60]!r} found {found} times, expected {count}')
        text = text[:start] + text[start:end].replace(old, new) + text[end:]
    return text


SOURCE = _apply(_pinned.SOURCE)
_MODULES = {}


def cuda_source(waves=False):
    """NCAR/MYNN-SFC v6.1.0 src/module_sf_mynnsfc_driver.F90:73-87 @a038a9e."""
    kernels=files('woof').joinpath('core/kernels')
    source = ("typedef float real;\n#define SL_LOG mynn_glibc_logf\n"
            "#define SL_EXP mynn_glibc_expf\n#define SL_POW mynn_glibc_powf\n"
            "#define SL_ATAN glibc_atanf\n#define SL_SQRT __fsqrt_rn\n"
            + kernels.joinpath('mynn_libm.cuh').read_text()
            + kernels.joinpath('glibc_trig_flt32.cuh').read_text() + SOURCE + _driver_source())
    if waves:
        from ...wave_surface import rrfs_mynn_source
        source = rrfs_mynn_source(source)
    return source


def module(waves=False, corrections=()):
    """Own image and key, the v6.0.0 NVRTC options (``corrections`` is the rebound v6.0.0 glue's keyword: v6.1.0
    already carries SF1, upstream f0c1839, so no fork id changes this source); NCAR/MYNN-SFC v6.1.0 src/module_sf_mynnsfc_driver.F90:73-87,838-881."""
    import cupy as cp
    from cupy.cuda import compiler, function
    dev=cp.cuda.runtime.getDevice()
    key = (dev, waves, 'v610')
    if key not in _MODULES:
        binary,_=compiler.compile_using_nvrtc(cuda_source(waves=waves),
            options=_hex_numerics.options(('--std=c++17','--fmad=false','--ftz=false',
                     '--prec-div=true','--prec-sqrt=true')),
            arch=_compilation_target(dev, xp=cp),cache_in_memory=True)
        obj=function.Module(); obj.load(binary)
        error=cp.empty(1,dtype=cp.int32)
        obj.get_function('mynnsfc_tables')((1,),(1,),(error,))
        _MODULES[key]=obj
    return _MODULES[key]


def _rebind(function):
    """The v6.0.0 host glue, unchanged, resolving module() and its helpers in this module."""
    bound = FunctionType(function.__code__, globals(), function.__name__, function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    bound.__doc__ = function.__doc__
    return bound


run_packed = _rebind(_pinned.run_packed)
from_mpas = _rebind(_pinned.from_mpas)
to_mpas = _rebind(_pinned.to_mpas)
driver = _rebind(_pinned.driver)


def launch_slot(call):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:1148-1463 @44c0849 host."""
    if (call.slot,call.scheme,str(call.backend))!=('surface_layer',SCHEME,BACKEND):
        raise ValueError('MYNN-SFC launcher requires surface_layer/sf_mynnsfclay/hex-cuda')
    from .resident_sfclay_rrfs import resident_slot
    slot = resident_slot(call.pools, call, scheme=sys.modules[__name__])
    if slot is not None:
        return slot.run(call)
    pools=call.pools
    mesh,diag,sfc,atm,configs,state=(pools[k] for k in
        ('mesh','diag_physics','sfc_input','atm_input','configs','scalar_state'))
    from ...wave_surface import feedback
    return driver(mesh,diag,sfc,atm,itimestep=call.itimestep,
        len_disp=configs['config_len_disp'],xice_threshold=state['xice_threshold'],
        config_do_restart=configs['config_do_restart'],config_do_DAcycling=configs['config_do_DAcycling'],
        config_frac_seaice=call.options['config_frac_seaice'],
        config_lsm_scheme=call.options['config_lsm_scheme'],
        config_mynn_sfcflux_land=call.options['config_mynn_sfcflux_land'],
        config_mynn_sfcflux_water=call.options['config_mynn_sfcflux_water'],
        sea_state=pools['prepared'],ocean_z0=feedback(pools))


def register_backend(registry):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:1390-1463 @44c0849 host."""
    registry.register('surface_layer',SCHEME,BACKEND,launch_slot)


# WOOF_HEX_AD_PARAMS: the MYNN-SFC constants are carried for v6.0.0 only; this version refuses them (unset: nothing happens).
from .. import ad_params_override as _ad_params_override
_ad_params_override.refuse_family("mynnsfc", "MYNN-SFC v6.1.0")
