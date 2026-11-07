"""WRF monthly aerosol forcing for the shared cumulus and microphysics state."""
from __future__ import annotations
import datetime
import numpy as np

MONTHLY_FIELDS = ('AEROMCU','AEROPCU')


def load_native_monthly_aerosol(raw, source_directory):
    """Interpolate prepared fields or the checked public NASA replacement."""
    if all(name in raw for name in MONTHLY_FIELDS):
        return raw
    from pathlib import Path
    import subprocess,tempfile
    from woof.public_cumulus_aerosol import binary, public_asset
    asset,receipt=public_asset(source_directory)
    provider=binary('aerosol-grid','WOOF_AEROSOL_GRID')
    latitude=np.asarray(raw['XLAT'],dtype=np.float32)
    longitude=np.asarray(raw['XLONG'],dtype=np.float32)
    if latitude.shape!=longitude.shape:
        raise ValueError('aerosol interpolation needs latitude and longitude on the same native grid')
    with tempfile.TemporaryDirectory(prefix='aerosol-grid-',dir=Path.cwd()) as work:
        work=Path(work)
        np.concatenate([latitude.ravel(),longitude.ravel()]).astype('<f4').tofile(work/'grid.bin')
        subprocess.run([str(provider),str(asset),str(work/'grid.bin'),str(work/'monthly.bin')],check=True)
        values=np.fromfile(work/'monthly.bin','<f4').reshape((11,12,30)+latitude.shape)
    return {**raw,'AEROPCU':values[0],'AEROMCU':values[1:],'AEROSOL_FORCING_RECEIPT':receipt}


def restore_monthly_aerosol(state, raw, cfg, start_time):
    if cfg.aercu_opt<=0:
        return
    import cupy as cp
    missing=[name for name in MONTHLY_FIELDS if name not in raw]
    if missing:
        raise ValueError('aercu_opt > 0 needs supplied monthly aerosol '+', '.join(missing)+
                         ': without these fields the convection and mp=40 activation would consume zero or stale aerosol. Supply native-grid AEROMCU (10,12,30,ny,nx), AEROPCU (12,30,ny,nx) in Pa through the prepared input.')
    shape=tuple(state.p.shape)
    expected={'AEROMCU':(10,12,30)+shape[1:],'AEROPCU':(12,30)+shape[1:]}
    for name in MONTHLY_FIELDS:
        if tuple(raw[name].shape)!=expected[name]:
            raise ValueError(f'{name} must have shape {expected[name]}; the forcing would otherwise interpolate a different grid or species inventory')
        value=cp.asarray(raw[name],dtype=cp.float32)
        if not bool(cp.isfinite(value).all()) or bool((value<0).any()):
            raise ValueError(f'{name} contains nonfinite or negative aerosol forcing')
        setattr(state,name.lower(),cp.ascontiguousarray(value))
    state.aerocu=cp.empty((10,)+shape,cp.float32)
    if start_time is None:
        raise ValueError('monthly aerosol forcing needs the declared forecast start time to interpolate its month')
    if isinstance(start_time,str):
        start_time=datetime.datetime.fromisoformat(start_time.replace('Z','+00:00').replace('_','T'))
    state.aerosol_start_time=float(start_time.replace(tzinfo=datetime.timezone.utc).timestamp())
    if 'AEROSOL_FORCING_RECEIPT' in raw:
        state.aerosol_forcing=dict(raw['AEROSOL_FORCING_RECEIPT'])
    update_monthly_aerosol(state,cfg,state.p)


def forcing_julian(start_epoch,elapsed_seconds):
    """WRF zero-based Julian day, matching the shared CAM calendar route."""
    time=datetime.datetime.fromtimestamp(float(start_epoch)+float(elapsed_seconds),datetime.timezone.utc)
    hour=time.hour+time.minute/60.0+time.second/3600.0+time.microsecond/3.6e9
    return np.float32(time.timetuple().tm_yday-1+hour/24.0)


def update_monthly_aerosol(state, cfg, pressure):
    if cfg.aercu_opt<=0:
        return
    import cupy as cp
    from woof.core.kernels.no_ftz import get_kernel_no_ftz as get_kernel
    if getattr(state,'aeromcu',None) is None or getattr(state,'aeropcu',None) is None:
        raise ValueError('monthly aerosol forcing is missing AEROMCU/AEROPCU; activation cannot use an absent aerosol state')
    julian=forcing_julian(state.aerosol_start_time,state.elapsed_seconds)
    nz,ny,nx=pressure.shape
    ncol=ny*nx
    status=cp.zeros(1,cp.int32)
    get_kernel('cumulus_aerosol','cumulus_aerosol')(((ncol+127)//128,),(128,),
        (state.aeromcu,state.aeropcu,cp.ascontiguousarray(pressure),state.aerocu,status,
         julian,np.int32(30),np.int32(nz),np.int32(ncol)))
    from woof.core.health_ledger import read_status
    failure = read_status(status, site="mskf-aerosol-pressure",
                          describe=_raise_deferred_pressure)
    if failure:
        _raise_pressure()


def _raise_pressure(note=""):
    raise ValueError('AER_P_INT: aerosol pressure data is not finite, positive, and strictly increasing; WRF module_cumulus_driver.F:1920 refuses suspected non-monotonicity before activation' + note)


def _raise_deferred_pressure(status):
    from woof.core.health_ledger import deferred_note
    if status:
        _raise_pressure(deferred_note())
