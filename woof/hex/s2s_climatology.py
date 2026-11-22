"""ERA5 historical window distributions: Rust decode/NetCDF and CUDA numerics.

Each baseline year contributes one exact matching calendar window. Quantiles
are taken over those window means, never over instantaneous or smoothed normals.
Only compressed bytes, metadata and the Rust bridge's buffer transport cross
Python. Raw chunks live in bounded memory and are released after each source/year.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import lru_cache
import ctypes
import ctypes.util
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import urllib.request

from .s2s_door import S2SError, cores, write_json

STORE = ('https://storage.googleapis.com/weatherbench2/datasets/era5/'
         '1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr')
# Rows describe fields, units, scale and pressure level. New fields share the
# same reader, reducer, remap and calendar path.
FIELDS = {'temperature2': ('2m_temperature', 'K', 1., None),
          'precipitation_rate': ('total_precipitation_6hr', 'm', 1000., None),
          'z500': ('geopotential', 'm**2 s**-2', 1 / 9.80665, 500),
          'mslp': ('mean_sea_level_pressure', 'Pa', .01, None)}
FIELDS.update({f'{component}{level}': (source, 'm s**-1', 1., level)
               for component, source in (('u', 'u_component_of_wind'), ('v', 'v_component_of_wind'))
               for level in (925, 850, 200)})


def calendar_samples(start, low, high):
    """Gregorian day-of-year and UTC hour, retained for archive consumers."""
    return [(stamp.hour, stamp.timetuple().tm_yday) for stamp in
            (start + timedelta(hours=6 * n) for n in range(low * 4, high * 4 + 1))]


def historical_start(start, year):
    """Match month/day and hour. Feb 29 uses Feb 28 in non-leap years."""
    try:
        return start.replace(year=year)
    except ValueError:
        if (start.month, start.day) != (2, 29):
            raise
        return start.replace(year=year, day=28)


def reduce_window(xp, samples, *, precipitation, days):
    if days <= 0 or len(samples) != days * 4 + 1:
        raise S2SError('ERA5 window needs every six-hour sample and both endpoints')
    values = xp.stack(samples)
    if precipitation:
        # Six-hour precipitation ends at each timestamp; exclude initial rain.
        return values[1:].sum(axis=0) / days
    return (values[0] * .5 + values[-1] * .5 + values[1:-1].sum(axis=0)) / (days * 4)


def window_distribution(xp, yearly):
    """Complete-year support and linear empirical quantiles over window means."""
    values = xp.stack(yearly)
    if values.shape[0] < 3:
        raise S2SError('ERA5 terciles need at least three historical window means')
    complete = xp.isfinite(values).all(axis=0)
    quantiles = xp.quantile(values, xp.asarray([1 / 3, 2 / 3]), axis=0, method='linear')
    return (xp.where(complete, values.mean(axis=0), xp.nan),
            xp.where(complete[None], quantiles, xp.nan))


def _fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise


@lru_cache(maxsize=1)
def _decoder():
    from .helpers import path as helper_path
    library = Path(os.environ.get('WOOF_HEX_S2S_ZARR_LIBRARY') or helper_path('libhex_s2s_zarr.so'))
    if not library.is_file():
        raise S2SError(f'ERA5 Rust chunk decoder missing: {library}; build rust/s2s-zarr and install '
                       'libhex_s2s_zarr.so in woof/hex/libexec or set WOOF_HEX_S2S_ZARR_LIBRARY')
    blosc = os.environ.get('WOOF_HEX_S2S_BLOSC_LIBRARY') or ctypes.util.find_library('blosc')
    if not blosc:
        spec = importlib.util.find_spec('netCDF4')
        candidates = sorted((Path(spec.origin).parent.parent / 'netcdf4.libs').glob('libblosc-*.so.*')) if spec else []
        blosc = str(candidates[0]) if candidates else None
    if not blosc:
        raise S2SError('ERA5 Rust chunk decoder needs BSD-3-Clause c-blosc >=1.21 '
                       '(libblosc or the netCDF4 wheel library); set WOOF_HEX_S2S_BLOSC_LIBRARY')
    native = ctypes.CDLL(str(library))
    decode = native.hex_s2s_decode
    decode.argtypes = (ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t)
    decode.restype = ctypes.c_int64
    return native, decode, os.fsencode(blosc)


def decode_chunk(payload, info):
    """Rust validates and decodes; NumPy wraps the unchanged transport buffer."""
    import numpy as np
    if (info['zarr_format'] != 2 or info['order'] != 'C' or info.get('filters')
            or (info.get('compressor') or {}).get('id') != 'blosc'
            or info['dtype'] not in ('<f4', '<f8', '<i4', '<i8')):
        raise S2SError('ERA5 requires C-order unfiltered little-endian numeric Blosc Zarr v2')
    _, decode, codec = _decoder()
    size = math.prod(info['chunks']) * np.dtype(info['dtype']).itemsize
    if not 0 < size <= 512 * 1024 * 1024:
        raise S2SError('ERA5 chunk exceeds the 512 MiB per-worker memory limit')
    buffer = ctypes.create_string_buffer(size)
    actual = decode(codec, payload, len(payload), buffer, size)
    if actual != size:
        raise S2SError(f'Rust ERA5 Blosc decode failed ({actual}); expected {size} bytes')
    return np.frombuffer(buffer, dtype=info['dtype']).reshape(info['chunks'])


class Era5Store:
    def __init__(self, url=STORE):
        self.url = url.rstrip('/')
        raw = _fetch(self.url + '/.zmetadata')
        self.metadata_hash = hashlib.sha256(raw).hexdigest()
        self.metadata = json.loads(raw)['metadata']
        self.receipts = []

    def read(self, name, indices):
        key = name + '/' + '.'.join(map(str, indices))
        payload = _fetch(self.url + '/' + key)
        array = decode_chunk(payload, self.metadata[name + '/.zarray'])
        return array, {'url': self.url + '/' + key, 'bytes': len(payload),
                       'sha256': hashlib.sha256(payload).hexdigest(), 'decoded_bytes': array.nbytes}

    def coordinate(self, cp, name):
        info = self.metadata[name + '/.zarray']
        if len(info['shape']) != 1 or info['chunks'][0] < info['shape'][0]:
            raise S2SError('ERA5 coordinate needs one chunk: ' + name)
        array, receipt = self.read(name, [0])
        self.receipts.append(receipt)
        return cp.asarray(array[:info['shape'][0]])

    def time_origin(self, cp):
        """Validate stored CF time words, including a nonzero first offset.

        The archive's CF epoch is a unit origin, not necessarily its first
        sample. In particular some daily-accumulation archives begin later.
        """
        if hasattr(self, '_time_origin'):
            return self._time_origin
        attrs = self.metadata['time/.zattrs']
        if attrs != {'_ARRAY_DIMENSIONS':['time'], 'calendar':'proleptic_gregorian',
                     'units':'hours since 1959-01-01'}:
            raise S2SError('ERA5 time coordinate convention changed')
        info = self.metadata['time/.zarray']
        first_hour = None
        for index in range(math.ceil(info['shape'][0] / info['chunks'][0])):
            times, receipt = self.read('time', [index])
            self.receipts.append(receipt)
            first = index * info['chunks'][0]
            size = min(info['chunks'][0], info['shape'][0] - first)
            device = cp.asarray(times[:size])
            if first_hour is None:
                first_hour = float(device[0])
                if not math.isfinite(first_hour) or first_hour % 6:
                    raise S2SError('ERA5 first time is not a six-hour offset')
            if not bool(cp.array_equal(device, first_hour + cp.arange(first, first+size) * 6)):
                raise S2SError('ERA5 archive time coordinate is not a complete six-hour clock')
        self._time_origin = datetime(1959,1,1) + timedelta(hours=first_hour)
        return self._time_origin

    def source_windows(self, cp, source, years, start, product_windows, pool):
        info = self.metadata[source + '/.zarray']
        attrs = self.metadata[source + '/.zattrs']
        pressure = 'level' in attrs['_ARRAY_DIMENSIONS']
        dims = ['time'] + (['level'] if pressure else []) + ['longitude', 'latitude']
        if (attrs['_ARRAY_DIMENSIONS'] != dims or info['chunks'][1:] != info['shape'][1:]):
            raise S2SError('ERA5 field must have complete spatial/level chunks: ' + source)
        epoch = self.time_origin(cp)
        chunk_size = info['chunks'][0]
        for year in years:
            anchor = historical_start(start, year)
            hours = (anchor - epoch).total_seconds() / 3600
            if hours % 6:
                raise S2SError('ERA5 window initialization must be on a six-hour UTC boundary')
            base = int(hours // 6)
            last = base + max(high for _, _, high in product_windows) * 4
            if base < 0 or last >= info['shape'][0]:
                raise S2SError(f'ERA5 archive lacks complete window for baseline year {year}')
            indices = list(range(base // chunk_size, last // chunk_size + 1))
            # Iterate bounded batches. A pressure chunk can be >100 MiB.
            chunks = {}
            for at in range(0, len(indices), 4):
                batch = indices[at:at + 4]
                jobs = [(source, [index] + [0] * (len(dims) - 1)) for index in batch]
                for index, (array, receipt) in zip(batch, pool.map(lambda job: self.read(*job), jobs)):
                    chunks[index] = cp.asarray(array)
                    self.receipts.append(receipt)
            reduced = {}
            for name, low, high in product_windows:
                selected = [chunks[index // chunk_size][index % chunk_size]
                            for index in range(base + low * 4, base + high * 4 + 1)]
                reduced[name] = reduce_window(cp, selected,
                    precipitation=source == 'total_precipitation_6hr', days=high - low)
            yield year, reduced
            del chunks


def fetch_normals(value, args, latitude, longitude):
    # The ERA5 archive's Licence to Use Copernicus Products permits commercial use with
    # attribution: acquisition is default, the attribution rides the receipt.
    import cupy as cp
    from woof.io.nc_writer_bridge import ClassicSchema
    from .s2s_products import TIME_FORMAT, windows, FIELDS as UNITS
    started = time.monotonic()
    root = Path(value['paths']['work'])
    target, receipt_path = root / 'era5-window-climatology.nc', root / 'era5-climatology-receipt.json'
    start = datetime.strptime(value['start_time'], TIME_FORMAT)
    # Exclude the event's year, including for hindcasts, so the target cannot
    # set its own tercile boundaries. The receipt names the actual sample.
    years = [year for year in range(1990, 2020) if year != start.year]
    store = Era5Store(os.environ.get('WOOF_HEX_S2S_ERA5_STORE', STORE))
    identity = {'store': store.url, 'start': value['start_time'], 'days': args.days,
                'shape': list(latitude.shape), 'schema': 's2s-era5-window-distribution/v3', 'years': years,
                'metadata_sha256': store.metadata_hash, 'fields': {key: list(row) for key, row in FIELDS.items()},
                'grid_sha256': hashlib.sha256(cp.asnumpy(cp.stack((latitude, longitude))).tobytes()).hexdigest()}
    if target.is_file() and receipt_path.is_file():
        prior = json.loads(receipt_path.read_text())
        if prior.get('identity') == identity and prior.get('sha256') == hashlib.sha256(target.read_bytes()).hexdigest():
            return target, prior['baseline']
    workers = min(cores(), 16)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        source_lat = store.coordinate(cp, 'latitude')
        source_lon = store.coordinate(cp, 'longitude')
        levels = store.coordinate(cp, 'level')
        store.time_origin(cp)
        iy = cp.abs(latitude[..., None] - source_lat).argmin(axis=-1)
        ix = cp.abs((longitude[..., None] - source_lon + 180) % 360 - 180).argmin(axis=-1)
        terrain_info = store.metadata['geopotential_at_surface/.zattrs']
        if terrain_info.get('units') != 'm**2 s**-2' or terrain_info['_ARRAY_DIMENSIONS'] != ['longitude', 'latitude']:
            raise S2SError('ERA5 orography units/dimensions changed')
        terrain, receipt = store.read('geopotential_at_surface', [0, 0])
        store.receipts.append(receipt)
        terrain = cp.asarray(terrain)[ix, iy] / 9.80665
        if not bool(cp.isfinite(terrain).all()):
            raise S2SError('nonfinite ERA5 source orography')
        schema = ClassicSchema('cdf5')
        y = schema.def_dim('south_north', latitude.shape[0])
        x = schema.def_dim('west_east', latitude.shape[1])
        payloads = []
        def variable(name, units, array):
            number = schema.def_var(name, 'f4', (y, x))
            schema.put_var_attr(number, 'units', units)
            payloads.append((number, cp.asnumpy(array.astype(cp.float32))))
        baseline = ('ERA5 1990-2019 matching calendar window means; target year excluded; '
                    f'{len(years)} years; {source_lon.size}x{source_lat.size} conservative source; '
                    'Rust Blosc/NetCDF; GPU means, terciles and nearest-grid remap')
        schema.put_global_attr('s2s_climatology_source', 'ERA5')
        schema.put_global_attr('s2s_data_license', 'Licence to Use Copernicus Products; research-only opt-in')
        schema.put_global_attr('attribution', 'Contains modified Copernicus Climate Change Service information; '
                               'neither the European Commission nor ECMWF is responsible for use of this data')
        schema.put_global_attr('s2s_climatology_start_mmdd', start.strftime('%m-%d'))
        schema.put_global_attr('s2s_climatology_start_hour', start.hour, dtype='i4')
        schema.put_global_attr('s2s_climatology_baseline', baseline)
        schema.put_global_attr('s2s_climatology_years', json.dumps(years))
        schema.put_global_attr('s2s_climatology_calendar', 'matching month/day/hour; Feb29 maps to Feb28 in non-leap years')
        schema.put_global_attr('s2s_climatology_terciles', 'linear quantile of historical window means at 1/3 and 2/3')
        variable('XLAT', 'degrees_north', latitude)
        variable('XLONG', 'degrees_east', longitude)
        variable('orography', 'm', terrain)
        product_windows = windows(args.days)
        source_seconds = {}
        for source in dict.fromkeys(row[0] for row in FIELDS.values()):
            source_started = time.monotonic()
            rows = {field: row for field, row in FIELDS.items() if row[0] == source}
            expected_units = next(iter(rows.values()))[1]
            if store.metadata[source + '/.zattrs']['units'] != expected_units:
                raise S2SError('ERA5 source units changed: ' + source)
            samples = {(name, field): [] for name, _, _ in product_windows for field in rows}
            for _, reduced in store.source_windows(cp, source, years, start, product_windows, pool):
                for field, (_, _, scale, level) in rows.items():
                    level_index = None
                    if level is not None:
                        matches = cp.where(levels == level)[0]
                        if matches.size != 1:
                            raise S2SError(f'ERA5 archive needs exactly one {level} hPa level')
                        level_index = int(matches[0])
                    for name, _, _ in product_windows:
                        plane = reduced[name] if level is None else reduced[name][level_index]
                        # Keep historical samples on the smaller source grid.
                        samples[name, field].append(plane * scale)
            for (name, field), yearly in samples.items():
                mean, quantiles = window_distribution(cp, yearly)
                key = name.replace('-', '_') + '_' + field
                for suffix, plane in (('', mean), ('_tercile_lower', quantiles[0]), ('_tercile_upper', quantiles[1])):
                    remapped = plane[ix, iy]
                    if not bool(cp.isfinite(remapped).all()):
                        raise S2SError('incomplete historical ERA5 support: ' + key + suffix)
                    variable(key + suffix, UNITS[field], remapped)
            cp.cuda.Stream.null.synchronize()
            source_seconds[source] = time.monotonic() - source_started
    temporary = target.with_suffix('.part')
    with schema.create(temporary) as writer:
        for number, payload in payloads:
            writer.write_var(number, payload)
    temporary.replace(target)
    write_json(receipt_path, {'identity': identity, 'baseline': baseline,
        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size,
        'wall_seconds': time.monotonic() - started, 'source_stage_seconds': source_seconds,
        'source_chunks': store.receipts,
        'parallelism': {'fetch_decode_threads': workers, 'concurrent_data_chunks': 4,
                        'blosc_threads_per_chunk': 1, 'reductions_quantiles_remap': 'GPU'},
        'calendar': 'matching Gregorian month/day/hour; Feb29 maps to Feb28 in non-leap baseline years',
        'raw_disposition': 'compressed and decoded chunks held in memory only',
        'terciles': 'linear empirical quantiles of historical window means, not daily quantiles'})
    return target, baseline
