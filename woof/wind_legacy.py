"""Preparation for the public WRF V3.5 analytic turbine scheme."""
import hashlib
import json
import math
import subprocess
from pathlib import Path

DEFAULTS = dict(td_turbgridid=-1, td_hubheight=100., td_diameter=60., td_stdthrcoef=.158,
    td_cutinspeed=4., td_cutoutspeed=27., td_power=2., td_turbpercell=1., td_ewfx=0,
    td_ewfy=0, td_pwfx=1, td_pwfy=1)


def read_spec(path):
    from woof.wind_inputs import table_binary
    result = subprocess.run([str(table_binary()), '--windspec', str(Path(path).resolve())],
                            capture_output=True, timeout=90)
    if result.returncode:
        raise ValueError('WRF windspec: ' + result.stderr.decode(errors='replace').strip())
    data = json.loads(result.stdout)
    if data.get('schema') != 'gpuwm-wind-legacy.v1':
        raise ValueError('WRF windspec decoder returned the wrong schema')
    return data, hashlib.sha256(result.stdout).hexdigest()


def validate_config(cfg):
    if cfg.bl_pbl_physics != 5 or cfg.bldt != 0:
        raise ValueError('windturbines_spec requires every-step MYNN (bl_pbl_physics=5, bldt=0)')
    if cfg.windturbines_spec == 'none':
        raise ValueError('fitch_legacy requires windturbines_spec=ideal or a native windspec filename')
    if cfg.windfarm_induction:
        raise ValueError('windfarm_induction belongs to modern Fitch; WRF V3.5 defines no induction selector')
    if cfg.windturbines_spec != 'ideal':
        read_spec(cfg.windturbines_spec)
        return
    for key, default in DEFAULTS.items():
        value = getattr(cfg, key)
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite')
        if type(default) is int and type(value) is not int:
            raise ValueError(f'{key} must be an integer')
    if (cfg.td_diameter <= 0 or cfg.td_hubheight <= cfg.td_diameter/2 or cfg.td_power <= 0
        or not 0 <= cfg.td_stdthrcoef <= 1 or cfg.td_cutinspeed < 0
        or cfg.td_cutoutspeed <= cfg.td_cutinspeed or cfg.td_turbpercell < 0
        or min(cfg.td_ewfx,cfg.td_ewfy) < 0):
        raise ValueError('invalid WRF ideal turbine geometry, density or coefficients')
