"""Pure setup controls for WRF v4.6.1 digital-filter initialization."""
from __future__ import annotations

from fractions import Fraction

RHO_LABEL = ('native WRF 4.7.1 fails here; this fixes it, not bitwise to WRF')

DFI_KEYS = frozenset(('dfi_opt','dfi_savehydmeteors','dfi_nfilter',
    'dfi_cutoff_seconds','dfi_bckstop_seconds','dfi_fwdstop_seconds',
    'dfi_write_filtered_input','dfi_write_dfi_history','time_step_dfi',
    'moist_adv_dfi_opt','dfi_constant_bc','dfi_time_dim'))


def validate_dfi(cfg):
    if type(cfg.dfi_opt) is not int or cfg.dfi_opt not in (0,1,2,3):
        raise ValueError('dfi_opt must be 0 (off), 1 (DFL), 2 (DDFI) or 3 (TDFI)')
    if not cfg.dfi_opt:
        return
    if cfg.bl_pbl_physics == 10:
        raise ValueError('DFI not available for bl_pbl_physics=10 (WRF module_check_a_mundo:2324)')
    if type(cfg.dfi_savehydmeteors) is not int or cfg.dfi_savehydmeteors not in (0,1):
        raise ValueError('dfi_savehydmeteors must be 0 or 1')
    if type(cfg.dfi_nfilter) is not int or cfg.dfi_nfilter not in range(9):
        raise ValueError('dfi_nfilter must be 0..8; filter 8: '+RHO_LABEL)
    if type(cfg.dfi_cutoff_seconds) is not int or cfg.dfi_cutoff_seconds <= 0:
        raise ValueError('dfi_cutoff_seconds must be a positive integer')
    for key in ('dfi_bckstop_seconds','dfi_fwdstop_seconds','time_step_dfi'):
        if type(getattr(cfg,key)) is not int:
            raise ValueError(key+' must be integer seconds')
    if cfg.dfi_fwdstop_seconds <= 0 or (cfg.dfi_opt != 1 and cfg.dfi_bckstop_seconds >= 0):
        raise ValueError('DFI forward stop must follow the analysis and backward stop must precede it')
    if cfg.dfi_fwdstop_seconds > cfg.run_seconds:
        raise ValueError('DFI forward window extends beyond the configured model end time')
    dt = Fraction(cfg.time_step_dfi) if cfg.time_step_dfi > 0 else Fraction(str(cfg.clock_dt or cfg.dt))
    if dt <= 0:
        raise ValueError('DFI requires a positive configured filter timestep')
    if cfg.dfi_nfilter == 8 and cfg.dfi_cutoff_seconds <= 2*dt:
        raise ValueError('RHO cutoff must exceed two timesteps (Nyquist period); '+RHO_LABEL)
    start = (0 if cfg.dfi_opt == 1 else cfg.dfi_bckstop_seconds if cfg.dfi_opt == 2
             else cfg.dfi_bckstop_seconds-int(cfg.dfi_bckstop_seconds/2))
    durations = [cfg.dfi_fwdstop_seconds-start]
    if cfg.dfi_opt != 1:
        durations.append(-cfg.dfi_bckstop_seconds)
    if not cfg.nested and any((Fraction(duration)/dt).denominator != 1 for duration in durations):
        raise ValueError('DFI phase duration must contain a whole number of configured model steps')
    if type(cfg.dfi_time_dim) is not int or cfg.dfi_time_dim < 1:
        raise ValueError('dfi_time_dim must be positive')
    if any(int(Fraction(duration)/dt)+1 > cfg.dfi_time_dim for duration in durations):
        raise ValueError('dfi_time_dim is smaller than the required native filter coefficient vector')
    if cfg.moist_adv_dfi_opt not in (0,1,2):
        raise ValueError('moist_adv_dfi_opt must select a supported WRF moisture advection option')
    for key in ('dfi_write_filtered_input','dfi_write_dfi_history','dfi_constant_bc'):
        if type(getattr(cfg,key)) is not bool:
            raise ValueError(key+' must be boolean')


def import_dfi(inp, *, error):
    """Translate the stock Registry controls without changing the UTC origin."""
    from datetime import datetime
    entries = inp.get('dfi_control',{})
    selector = (entries.get('dfi_opt') or [0])[0]
    if not selector or (inp.get('time_control',{}).get('restart') or [False])[0]:
        return {}
    if type(selector) is not int or selector not in (1,2,3):
        raise error('dfi_control','dfi_opt',selector,'must be 0, 1, 2 or 3')
    defaults = dict(dfi_savehydmeteors=0,dfi_nfilter=7,dfi_cutoff_seconds=3600,
        dfi_write_filtered_input=True,dfi_write_dfi_history=False,dfi_time_dim=1000)
    dates = {}
    for prefix,day in (('dfi_bckstop',14),('dfi_fwdstop',13)):
        for unit,value in zip(('year','month','day','hour','minute','second'),(2004,3,day,12,0,0)):
            defaults[prefix+'_'+unit] = value
    unknown = entries.keys() - (defaults.keys() | {'dfi_opt'})
    if unknown:
        raise error('dfi_control',sorted(unknown)[0],entries[sorted(unknown)[0]],'unknown native DFI control')
    values = {key:(entries.get(key) or [default])[0] for key,default in defaults.items()}
    time = inp.get('time_control',{})
    try:
        origin = datetime(*[(time.get('start_'+unit) or [default])[0]
            for unit,default in zip(('year','month','day','hour','minute','second'),(2004,1,1,0,0,0))])
        for prefix in ('dfi_bckstop','dfi_fwdstop'):
            date = datetime(*[values[prefix+'_'+unit] for unit in ('year','month','day','hour','minute','second')])
            dates[prefix+'_seconds'] = int((date-origin).total_seconds())
    except (ValueError,TypeError) as exc:
        raise error('dfi_control','dfi_opt',selector,'invalid DFI stop date: '+str(exc)) from None
    # The filter window must bracket the analysis.  A namelist that turns
    # DFI on without its stop dates reads the Registry's 2004 defaults,
    # years before any real start; refuse that here, by name, instead of
    # letting the run validator raise a bare ValueError mid-import.
    if dates['dfi_fwdstop_seconds'] <= 0 or (selector != 1 and dates['dfi_bckstop_seconds'] >= 0):
        raise error('dfi_control','dfi_opt',selector,
            'the DFI window does not bracket the start: forward stop '
            f"{dates['dfi_fwdstop_seconds']} s and backward stop "
            f"{dates['dfi_bckstop_seconds']} s from start_* (an omitted "
            'dfi_bckstop_*/dfi_fwdstop_* date is the Registry default, '
            '2004-03-14/13 12Z).  Next: give dfi_bckstop_* before and '
            'dfi_fwdstop_* after the start, or set dfi_opt = 0.')
    result = {key:values[key] for key in defaults if key in DFI_KEYS}
    result.update(dfi_opt=selector,**dates)
    domains, dynamics, boundary = (inp.get(key,{}) for key in ('domains','dynamics','bdy_control'))
    result['time_step_dfi'] = (domains.get('time_step_dfi') or [-1])[0]
    result['moist_adv_dfi_opt'] = (dynamics.get('moist_adv_dfi_opt') or [0])[0]
    result['dfi_constant_bc'] = (boundary.get('constant_bc') or [False])[0]
    return result


def registry_parameters():
    """DFI declarations consumed by the registry's existing generator."""
    from dataclasses import fields
    from woof.config import RunConfig
    defaults = {f.name:f.default for f in fields(RunConfig) if f.name in DFI_KEYS}
    rows = {key:dict(type='boolean' if type(default) is bool else
                         'integer' if type(default) is int else 'number',
                    default=default,per_domain=key=='moist_adv_dfi_opt',
                    consuming_read='woof/core/dfi_initialization.py')
            for key,default in defaults.items()}
    rows['dfi_opt']['enum'] = [0,1,2,3]
    rows['dfi_savehydmeteors']['enum'] = [0,1]
    rows['dfi_nfilter']['enum'] = list(range(9))
    rows['dfi_nfilter']['warnings'] = ['Opt-in filter 8: '+RHO_LABEL+
        '. Bounded RHOFIL row-prefix repair retains native midpoint reflection; filters 0..7 are unchanged.']
    rows['moist_adv_dfi_opt']['enum'] = [0,1,2]
    for key in ('dfi_time_dim','dfi_write_dfi_history','dfi_write_filtered_input','dfi_constant_bc'):
        rows[key]['consuming_read'] = ('woof/dfi_control.py' if key=='dfi_time_dim' else
            'woof/ingest/lateral_bc.py' if key=='dfi_constant_bc' else 'woof/core/dfi_output.py')
    return rows
