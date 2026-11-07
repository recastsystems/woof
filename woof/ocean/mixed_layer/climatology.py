"""Named NOAA WOA18 monthly MLD fallback: Rust decode, CUDA remapping.

Only mixed-layer depth comes from climatology. SST remains the atmospheric
analysis SST; salinity is 35 PSU and currents start at rest. This is not a
contemporaneous ocean analysis and cannot initialize ocean anomalies.
"""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import logging
import os
import shutil
import time
import numpy as np

SOURCE = {
    'name': 'NOAA WOA18 2005-2017 monthly MLD, density increment 0.125 kg m-3 from 10 m',
    'url': 'https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/woa/mld/A5B7/0.25/woa18_A5B7_M02{month:02d}_04.nc',
    'variable': 'M_an', 'latitude': 'lat', 'longitude': 'lon',
    'terms': 'https://www.ncei.noaa.gov/sites/default/files/2023-12/NCEI%20PD-10-2-02%20-%20Open%20Data%20Policy%20Signed.pdf',
    'citation': 'Locarnini et al. 2018, NOAA Atlas NESDIS 81; Zweng et al. 2018, NOAA Atlas NESDIS 82',
}

#: Every route NOAA serves the same WOA18 MLD bytes on, tried in order on a
#: cache miss.  Named breakage (2.9.4 WOOF readiness, 2026-10-22): the
#: THREDDS route was the only one, read once at FORECAST START with no
#: retry, and NCEI THREDDS answered 503 / timed out from 08:12Z to 09:25Z;
#: a Mexico City 750 m run died after 60 min of preparation.
#: The plain HTTPS data path goes first: on 2026-10-22 it served all 12
#: months while THREDDS timed out on every one for hours.
URL_TEMPLATES = (
    'https://www.ncei.noaa.gov/data/oceans/woa/WOA18/DATA/mld/netcdf/A5B7/0.25/woa18_A5B7_M02{month:02d}_04.nc',
    SOURCE['url'],
)

#: An operator mirror (a bucket, an intranet server): the file name is
#: appended.  When set it is the ONLY network route, so an air-gapped site
#: never reaches out to NCEI behind its back.
URL_BASE_ENV = 'WOOF_OCEAN_CLIMATOLOGY_URL_BASE'
CACHE_ENV = 'WOOF_OCEAN_CACHE'
STAGE_COMMAND = 'woof fetch-tables --ocean'

#: (bytes, SHA-256) of each month as NCEI publishes it (all 12 read from the
#: HTTPS data path on 2026-10-22; months 04, 06 and 10 also from THREDDS, same
#: bytes).  Pinned like the
#: Thompson and WIF table assets: a download is verified before the atomic
#: install, and a cached copy with other bytes is moved aside and refetched.
MONTH_PINS = {
    1: (7059931, 'f0824a1a5aa63617bc8fe138296be389e712dae2bd23e6fdbff8a6feafee2a69'),
    2: (7001993, '88045cf16487c2a133156cddf86bac53726f50e55d98d0c0211897121cad6102'),
    3: (7077588, '6d96f72a226cfbc72cdce47582dd70815670734685c3e4204e1152181c074271'),
    4: (7080799, '22308a2ecc9f5ac70effe6039eeac8c8fa8da79f67ca6d6215811ac1da3659ec'),
    5: (7135206, '0e6a0418f810b1e3efee39a0b11d9ff1916626a3c80f04b599654c73449c943f'),
    6: (7080737, '85a85810357b2d49d0004527be3b9aa21dd9689c391abf03bf4a3d92f01db8ce'),
    7: (7098972, '2f722b09065698f5b226533e99822f79600caa6fc83e79464b0fac72298c3d46'),
    8: (7125573, 'c17f52a8578702fd6bd0f81b64b2a8cd9799ecfaac49c93dba773b61032e1c52'),
    9: (7122581, 'a622e0a31b9120625ccd456e6aa0c9f172f115d965c19138c2801da8fafc4256'),
    10: (7151890, 'b4495bb60c48674d192f740fcc574a8625554b11e21008404a9379809d9ae177'),
    11: (7098125, '7799924bc67549979096d23a3acae3ff6ab4fc12d8f879bb5e227fa71d63c5e2'),
    12: (7093713, '7866f62dce3fb1d8c7567964b3408b62204c4e582e96b33e018146c381b3be31'),
}

#: Rounds over every source, and the pause before each later round.  A
#: transient 503 gets three chances per route; a long outage is what
#: ``woof fetch-tables --ocean`` (staging ahead of time) is for.
RETRY_BACKOFF_S = (0.0, 5.0, 20.0)
REQUEST_TIMEOUT_S = 60.0


class OceanClimatologyUnavailable(RuntimeError):
    """No cached copy and no route delivered the pinned month."""


def filename(month):
    return f'woa18_A5B7_M02{month:02d}_04.nc'


def cache_root(cache=None):
    """The one root acquire() reads and ``fetch-tables --ocean`` stages into."""
    if cache:
        return Path(cache)
    raw = os.environ.get(CACHE_ENV, '').strip()
    if raw:
        return Path(raw)
    from woof.case_data import case_data_root
    return case_data_root()/'ocean'


def source_urls(month):
    base = os.environ.get(URL_BASE_ENV, '').strip()
    if base:
        return (base.rstrip('/')+'/'+filename(month),)
    return tuple(template.format(month=month) for template in URL_TEMPLATES)


def _digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_pin(path, month):
    """None when ``path`` holds the pinned bytes for ``month``, else why not."""
    pin = MONTH_PINS.get(month)
    if pin is None:
        return None
    size = Path(path).stat().st_size
    if size != pin[0]:
        return f'{size} bytes, expected {pin[0]}'
    digest = _digest(path)
    if digest != pin[1]:
        return f'SHA-256 {digest}, expected {pin[1]}'
    return None


def _validate(candidate, month):
    """Pinned bytes, and a NetCDF file carrying MLD and its coordinates."""
    from woof.netcdf_bridge import open_dataset
    problem = check_pin(candidate, month)
    if problem:
        raise ValueError(f'NOAA WOA18 month {month:02d}: {problem}')
    with open_dataset(candidate) as ds:
        if not all(SOURCE[key] in ds.variables for key in ('variable','latitude','longitude')):
            raise ValueError('NOAA WOA18 response is missing MLD or coordinates')
        return str(getattr(ds,'title',''))


def _permanent(error):
    """A 4xx other than timeout or rate limiting will not change on a retry."""
    from urllib.error import HTTPError
    code = getattr(error, 'code', None)
    return isinstance(error, HTTPError) and code is not None and 400 <= code < 500 and code not in (408, 425, 429)


def acquire(month, *, cache=None, source_dir=None, backoff=RETRY_BACKOFF_S, sleeper=time.sleep):
    """The month's WOA18 MLD file: the cache first, the network only on a miss.

    On a miss every route in :func:`source_urls` is tried, in rounds with
    backoff, and only pinned, complete NetCDF bytes are committed.
    ``source_dir`` stages from a local directory instead (offline).  When
    nothing delivers, :class:`OceanClimatologyUnavailable` names
    ``woof fetch-tables --ocean``; the ocean is never switched off here.
    """
    from woof.fetch_guard import OutputLock, atomic_write_text
    from woof.nomads_governor import paced_urlopen
    from woof.netcdf_bridge import NetcdfFileError
    from http.client import HTTPException
    if type(month) is not int or not 1 <= month <= 12:
        raise ValueError('WOA18 needs a calendar month in 1..12')
    root=cache_root(cache)
    path=root/filename(month)
    root.mkdir(parents=True,exist_ok=True)
    log=logging.getLogger(__name__)
    with OutputLock('ocean-climatology',path):
        if path.is_file():
            try:
                _validate(path, month)
                return path
            except (NetcdfFileError, ValueError, OSError) as error:
                quarantine=path.with_suffix(f'.invalid-{time.time_ns()}.nc')
                path.rename(quarantine)
                log.warning('Unusable ocean climatology moved to %s; fetching a complete copy: %s',quarantine,error)
        temporary=path.with_suffix('.nc.part')
        start=time.monotonic()
        errors=[]
        try:
            if source_dir is not None:
                candidates=[Path(source_dir)/filename(month), Path(source_dir)/'ocean'/filename(month)]
                found=next((c for c in candidates if c.is_file()), None)
                if found is None:
                    raise OceanClimatologyUnavailable(
                        f'NOAA WOA18 month {month:02d}: {filename(month)} is not in {source_dir} '
                        f'(nor its ocean/ subfolder)')
                shutil.copyfile(found, temporary)
                title=_validate(temporary, month)
                url=found.resolve().as_uri()
            else:
                urls=source_urls(month)
                dead=set()
                url=title=None
                for attempt, pause in enumerate(backoff):
                    if pause:
                        sleeper(pause)
                    for candidate in urls:
                        if candidate in dead:
                            continue
                        try:
                            with paced_urlopen(candidate,timeout=REQUEST_TIMEOUT_S) as response,temporary.open('wb') as stream:
                                shutil.copyfileobj(response,stream,1024*1024)
                            title=_validate(temporary, month)
                            url=candidate
                            break
                        except (OSError, HTTPException, ValueError, NetcdfFileError) as error:
                            # Wrong bytes or a 404 will not change on a retry; a
                            # 503 or a timeout may.  Wrong bytes are never installed.
                            if _permanent(error) or isinstance(error, (ValueError, NetcdfFileError)):
                                dead.add(candidate)
                            errors.append(f'{candidate} (try {attempt+1}): {type(error).__name__}: {error}')
                            log.warning('WOA18 month %02d: %s failed (try %d of %d): %s',
                                        month, candidate, attempt+1, len(backoff), error)
                            temporary.unlink(missing_ok=True)
                    if url is not None or len(dead) == len(urls):
                        break
                if url is None:
                    raise OceanClimatologyUnavailable(
                        f'NOAA WOA18 monthly mixed-layer depth for month {month:02d} is not staged at '
                        f'{path} and no source delivered it after {len(backoff)} round(s): '
                        + '; '.join(errors[-4:])
                        + f'. Stage all 12 months ahead of time with `{STAGE_COMMAND}` '
                        f'(`{STAGE_COMMAND} --from DIR` offline; {CACHE_ENV} names another cache, '
                        f'{URL_BASE_ENV} a mirror).')
            digest=_digest(temporary)
            temporary.replace(path)
            atomic_write_text(path.with_suffix('.json'),json.dumps({
                'schema':'gpuwm-ocean-climatology-acquisition-v1',
                **SOURCE,'url':url,'month':month,'title':title,
                'sha256':digest,'bytes':path.stat().st_size,
                'pinned':month in MONTH_PINS,
                'wall_seconds':time.monotonic()-start},indent=2)+'\n')
        finally:
            temporary.unlink(missing_ok=True)
    return path


def stage_all(*, cache=None, source_dir=None, progress=print):
    """``woof fetch-tables --ocean``: all 12 months into the acquire() cache."""
    root=cache_root(cache)
    staged, failed = [], []
    outage=None
    for month in range(1, 13):
        present=(root/filename(month)).is_file()
        if outage is not None and not present:
            # One month already spent every retry on every route; eleven more
            # rounds of 503s would only make the refusal take 40 minutes.
            failed.append(month)
            progress(f'woof fetch-tables --ocean: REFUSED month {month:02d}: not attempted, '
                     f'every route failed for month {outage:02d} just now; re-run when NCEI answers')
            continue
        try:
            path=acquire(month, cache=root, source_dir=source_dir)
        except (OceanClimatologyUnavailable, OSError, ValueError) as error:
            failed.append(month)
            if source_dir is None and isinstance(error, OceanClimatologyUnavailable):
                outage=month
            progress(f'woof fetch-tables --ocean: REFUSED month {month:02d}: {error}')
            continue
        staged.append(month)
        verb='already staged and valid' if present else 'verified and installed'
        progress(f'woof fetch-tables --ocean: {path.name} {verb} '
                 f'({path.stat().st_size:,} B, SHA-256 {_digest(path)[:12]}...)')
    return staged, failed


def staged_state(cache=None):
    """Doctor's view: (root, valid months, damaged months), no network."""
    root=cache_root(cache)
    valid, damaged = [], []
    for month in range(1, 13):
        path=root/filename(month)
        if not path.is_file():
            continue
        try:
            problem=check_pin(path, month)
        except OSError as error:
            problem=str(error)
        (damaged if problem else valid).append(month)
    return root, valid, damaged


def _borrow_beyond_one_degree(indices, latitude, longitude, source_latitude, source_longitude, valid):
    """Give each water column the one-degree search missed its nearest valid WOA18 cell.

    Only mixed-layer depth is read from WOA18 (SST is the analysis'), so a
    water column farther than one degree from the WOA18 ocean -- a reservoir
    or river reach outside the lake mask -- borrows the nearest valid MLD at
    any distance, by the same great-circle measure and the same lowest-index
    tie rule as the one-degree kernel.  Returns ``(farthest km, count)``, or
    ``None`` when every column was covered.

    Named breakage: PWP is the default for forecasts of 48 h and more
    (a382c235f2), and the fallback refused every continental domain with
    such water ("WOA18 monthly MLD has no valid value within one degree of
    forecast ocean"): a full-CONUS HRRR 48 h forecast on the fold died
    before its first step (test host, 2026-10-21).
    """
    import cupy as cp
    missing = cp.flatnonzero(indices.ravel() == -1)
    if missing.size == 0:
        return None
    to_rad = cp.float32(0.01745329252)
    candidates = cp.flatnonzero(valid.ravel())
    if candidates.size == 0:
        raise ValueError('WOA18 monthly MLD has no valid value anywhere')
    slat = source_latitude.ravel()[candidates].astype(cp.float32) * to_rad
    slon = source_longitude.ravel()[candidates].astype(cp.float32) * to_rad
    tlat = cp.asarray(latitude, dtype=cp.float32).ravel()[missing] * to_rad
    tlon = cp.asarray(longitude, dtype=cp.float32).ravel()[missing] * to_rad
    best = cp.empty(missing.size, dtype=cp.int64)
    score = cp.empty(missing.size, dtype=cp.float32)
    chunk = max(1, int(2**24 // max(1, candidates.size)))
    for start in range(0, missing.size, chunk):
        a, b = tlat[start:start+chunk, None], tlon[start:start+chunk, None]
        dy, dx = cp.sin((a - slat[None]) * 0.5), cp.sin((b - slon[None]) * 0.5)
        dist = dy * dy + cp.cos(a) * cp.cos(slat[None]) * dx * dx
        # argmin returns the first minimum: the lowest source index, as the kernel ties.
        best[start:start+chunk] = cp.argmin(dist, axis=1)
        score[start:start+chunk] = cp.min(dist, axis=1)
    flat = indices.ravel()
    flat[missing] = candidates[best].astype(cp.int32)
    farthest = float(2.0 * 6371.0 * cp.arcsin(cp.sqrt(cp.clip(score.max(), 0, 1))))
    return farthest, int(missing.size)


def load(*, latitude, longitude, land, sst, levels, gamma, expected_time, path=None):
    import cupy as cp
    from woof.netcdf_bridge import open_dataset
    from .analysis import nearest_indices
    from . import kernels
    if latitude is None or longitude is None or expected_time is None:
        raise ValueError('WOA18 monthly MLD fallback needs forecast coordinates and UTC start')
    if isinstance(expected_time,str):
        expected_time=datetime.fromisoformat(expected_time.replace('Z','+00:00'))
    path=Path(path) if path else acquire(expected_time.month)
    with open_dataset(path) as ds:
        ds.prefetch([SOURCE[key] for key in ('variable','latitude','longitude')])
        mld=cp.asarray(ds.variables[SOURCE['variable']][...],dtype=cp.float32).squeeze()
        unit=str(getattr(ds.variables[SOURCE['variable']],'units','')).lower()
        lat=cp.asarray(ds.variables[SOURCE['latitude']][...],dtype=cp.float32)
        lon=cp.asarray(ds.variables[SOURCE['longitude']][...],dtype=cp.float32)
    if unit not in ('m','meters','metres') or lat.ndim!=1 or lon.ndim!=1 or mld.shape!=(lat.size,lon.size):
        raise ValueError('WOA18 requires a metre-valued MLD plane over latitude/longitude axes')
    lon,lat=cp.meshgrid(lon,lat)
    valid=cp.isfinite(mld)&(mld>0)&(mld<11000)
    indices=nearest_indices(latitude,longitude,land,lat,lon,valid,levels)
    farthest_km=_borrow_beyond_one_degree(indices,latitude,longitude,lat,lon,valid)
    if farthest_km is not None:
        logging.getLogger(__name__).warning(
            'WOA18 MLD: %d forecast water column(s) farther than one degree from WOA18 ocean '
            '(inland water outside the lake mask) took the nearest WOA18 mixed-layer depth, '
            'up to %.0f km away; SST stays the atmospheric analysis there',
            int(farthest_km[1]),farthest_km[0])
    result={name:cp.empty((levels,*land.shape),dtype=cp.float32)
            for name in ('om_tmp','om_s','om_u','om_v','om_depth')}
    n=result['om_tmp'].size
    kernels(levels).get_function('ocean_mld_profiles')(((n+127)//128,),(128,),
        (cp.ascontiguousarray(mld),indices,cp.ascontiguousarray(sst),
         *(result[name] for name in ('om_tmp','om_s','om_u','om_v','om_depth')),
         np.float32(gamma),np.int32(land.size)))
    logging.getLogger(__name__).warning('OCEAN INITIALIZATION FALLBACK: %s; month=%02d file=%s; atmospheric SST, 35 PSU, resting currents, configured lapse rate=%g K/m down to sea-water freezing (271.25 K). No contemporaneous ocean analysis.',
        SOURCE['name'],expected_time.month,path,gamma)
    return result
