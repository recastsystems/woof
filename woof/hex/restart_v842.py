"""Native MPAS-A v8.4.2 stream I/O and restart orchestration.

Membership: src/core_atmosphere/Registry.xml:650-925. Initialization:
src/core_atmosphere/mpas_atm_core.F:109-150,522-530,663-676,894-902.
Primitive encoding/decoding and dimension permutation live in rw_mpas_stream;
this module only hands arrays and metadata to that Rust artifact.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field as dataclass_field, replace
from datetime import datetime, timedelta
import errno
import json
import mmap
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterator, Mapping

import numpy as np

from .streams_v842 import Stream, StreamAlarms, field_definition, registry_stream, parse_streams
from .engines import EngineSpec, resolve
from . import history_compression


_STREAM_ENGINE = EngineSpec(
    name='rw_mpas_stream', flag='stream bridge',
    env_names=('WOOF_HEX_RW_MPAS_STREAM',),
    gpuwm_env='WOOF_RW_MPAS_STREAM', subject='the native stream reader/writer',
    what_breaks='history publication and native restart I/O cannot encode or decode field words',
    cargo_package='rw-mpas',
)


def _bridge() -> str:
    """Use the engine's staged bridge ladder, without implicit builds.

    The old override/PATH-only lookup missed installed libexec and user
    bridges in clean G0 processes, breaking every swath history fixture.
    Explicit missing paths still fail rather than selecting another binary.
    """
    return str(resolve(_STREAM_ENGINE))


def _json_attrs(attrs: Mapping[str, Any]) -> dict:
    """Serialize driver metadata for the bridge without changing field words."""
    def plain(value):
        if isinstance(value, Mapping): return {str(k):plain(v) for k,v in value.items()}
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, (tuple,list)): return [plain(v) for v in value]
        return value
    return {str(k):plain(v) for k,v in attrs.items() if k != '_NCProperties'}


def _call(request: dict, directory: Path) -> dict:
    """Run the real Rust stream reader/writer and propagate its error."""
    request_path = directory / 'request.json'
    request_path.write_text(json.dumps(request, allow_nan=False))
    result = subprocess.run([_bridge(), str(request_path)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    response = json.loads(result.stdout)
    if response.get('schema') != 'mpas-native-stream/v1':
        raise ValueError('stream bridge ABI mismatch')
    if 'dimension_order' in request and not response.get('ordered_metadata'):
        raise ValueError('rw_mpas_stream must be rebuilt with ordered metadata support')
    return response


def _date(value: datetime | str) -> datetime:
    from .output import _as_datetime
    return _as_datetime(value)


def _publish_stream(staged: Path, path: Path, mode: str) -> None:
    """Publish complete durable bytes atomically, copying only across devices."""
    try:
        if mode == 'never_modify':
            os.link(staged, path)
            staged.unlink()
        else:
            os.replace(staged, path)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        # Cross-filesystem scratch needs a durable temporary on the target.
        publish_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix=f'.{path.name}.', dir=path.parent,
                                             delete=False) as publish:
                publish_path = Path(publish.name)
                with staged.open('rb') as handle:
                    shutil.copyfileobj(handle, publish)
                publish.flush()
                os.fsync(publish.fileno())
            if mode == 'never_modify':
                os.link(publish_path, path)
                publish_path.unlink()
            else:
                os.replace(publish_path, path)
        finally:
            if publish_path is not None:
                publish_path.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _handoff_root(fields: Mapping[str, Any], fallback: Path) -> Path:
    """Where raw field words wait for rw_mpas_stream: memory, not the disk.

    The handoff was written to the output disk and read back by the bridge
    before it wrote the same bytes again as NetCDF: a 1.5 GB history frame
    cost about 3 GB of disk traffic per frame, longer than the hour between
    frames at 60 km, so the forecast waited on the previous frame. A tmpfs
    (WOOF_HEX_STREAM_HANDOFF, default /dev/shm) holds them when it has room
    for the frame twice over; otherwise the output scratch is used as before.
    """
    from .output import HistoryField
    choice = os.environ.get('WOOF_HEX_STREAM_HANDOFF', '/dev/shm')
    if not choice or choice == 'scratch':
        return fallback
    root = Path(choice)
    try:
        need = sum(np.asarray(v.values).nbytes if isinstance(v, HistoryField) else
                   getattr(v, 'nbytes', 0) for v in fields.values())
        if root.is_dir() and os.access(root, os.W_OK) and shutil.disk_usage(root).free > 2*need:
            return root
    except OSError:
        pass
    return fallback


def write_stream(path: str | Path, stream: Stream | str, mesh: object,
                 fields: Mapping[str, Any], valid_time: datetime | str, *,
                 initial_time: datetime | str,
                 active_packages=(), physics: bool = True,
                 streams: Mapping[str, Stream] | None = None,
                 global_attrs: Mapping[str, Any] | None = None,
                 field_attrs: Mapping[str, Mapping[str, Any]] | None = None,
                 scratch: str | Path | None = None,
                 record_index: int | None = None,
                 registry_attributes: bool = True,
                 extra_fields: Mapping[str, Any] | None = None) -> Path:
    """MPAS src/core_atmosphere/Registry.xml:498-1349, strict stream snapshot.

    ``extra_fields`` (name -> (float32 array, its axes without Time, units)) appends variables the Registry
    does not declare after the stream's own: the suite route's chemistry species and products
    (woof.hex.mpasphys.chemistry_v842).  None (every other caller) writes exactly the Registry stream.

    Every active field is required. Arrays are handed over in their named
    in-memory axes; Rust reorders to reversed Registry axes. Connectivity is
    offset only for variables marked converted by the mesh authority.
    Existing files follow the native clobber policy; writes publish atomically.
    """
    from .output import HistoryField
    if isinstance(stream, str): stream = registry_stream(stream, physics=physics)
    if 'output' not in stream.options['type'].split(';'):
        raise ValueError(f'{stream.name}: not an output stream')
    names = stream.fields(active_packages, physics=physics, streams=streams)
    if not names:
        raise ValueError(f'{stream.name}: stream is inactive or has no fields')
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stream.options.get('clobber_mode','never_modify')
    exists = path.exists()
    if exists and mode == 'never_modify': raise FileExistsError(path)
    if exists and mode in ('append','overwrite') and record_index is None:
        raise ValueError(f'{stream.name}: append/overwrite requires explicit record_index to protect existing records')
    if record_index is not None and (type(record_index) is not int or record_index < 0):
        raise ValueError('record_index must be a nonnegative integer')
    if not exists and record_index not in (None,0):
        raise ValueError('new stream file must start at record zero')
    arrays = getattr(mesh,'arrays',{})
    mesh_dims = getattr(mesh,'variable_dimensions',{})
    sizes = {k:int(v) for k,v in getattr(mesh,'dimensions',{}).items() if k != 'Time'}
    sizes.update(Time=1,StrLen=64)
    initial, valid = _date(initial_time), _date(valid_time)
    converted = set(getattr(mesh,'converted_connectivity',()))
    precision = stream.options.get('precision','native')
    missing = [n for n in names if n not in ('Time','xtime','initial_time') and n not in fields and n not in arrays]
    if missing: raise ValueError(f'{stream.name}: missing active native fields: {", ".join(missing)}')
    scratch_root = Path(scratch) if scratch else path.parent
    scratch_root.mkdir(parents=True,exist_ok=True)
    handoff_root = _handoff_root(fields, scratch_root)
    with tempfile.TemporaryDirectory(prefix='.mpas-stream-',dir=scratch_root) as temporary, \
         tempfile.TemporaryDirectory(prefix='.mpas-handoff-',dir=handoff_root) as raw:
        stage=Path(temporary); handoff_dir=Path(raw); variables=[]; dimension_order=[]
        for index,name in enumerate(names):
            definition=field_definition(name,physics=physics)
            logical_dims=tuple(definition['dimensions'])
            for dim in logical_dims:
                if dim not in dimension_order:
                    dimension_order.append(dim)
            disk_dims=tuple(reversed(logical_dims))
            attrs=dict(definition['attrs']) if registry_attributes else {}
            if field_attrs and name in field_attrs:
                attrs.update(field_attrs[name])
            kind=definition['type']
            dtype='S1' if kind=='text' else 'i4' if kind in ('integer','logical') else 'f8' if precision=='double' else 'f4'
            variable={'name':name,'dtype':dtype,'dimensions':disk_dims,'attrs':_json_attrs(attrs)}
            if name in ('xtime','initial_time'):
                value=valid if name=='xtime' else initial
                variable['text']=[value.strftime('%Y-%m-%d_%H:%M:%S')]
            else:
                if name=='Time':
                    # Native accumulates this RKIND word per step. A registered
                    # pool value must survive fractional-step rounding exactly;
                    # elapsed calendar seconds are only the standalone default.
                    raw=fields.get(name,HistoryField(np.asarray([(valid-initial).total_seconds()],dtype=np.float32),('Time',)))
                    variable['attrs']['units']=f'seconds since {initial:%Y-%m-%d %H:%M:%S}'
                else: raw=fields[name] if name in fields else arrays[name]
                if isinstance(raw,HistoryField):
                    array=np.asarray(raw.values); dims=raw.dimensions
                    # Registry is authoritative; callers may supply runtime Time units.
                    variable['attrs'].update(_json_attrs(raw.attrs))
                elif isinstance(raw,str):
                    array=None; dims=None; variable['text']=[raw]
                else:
                    array=np.asarray(raw); dims=mesh_dims.get(name) if name in arrays and name not in fields else None
                if array is not None:
                    if not array.dtype.isnative:
                        raise TypeError(f'{name}: handoff array must have native byte order')
                    if array.dtype.kind=='S' and array.dtype.itemsize==1:
                        input_dtype='S1'
                    elif array.dtype.kind=='b':
                        input_dtype='u1'
                    else:
                        input_dtype=f'{array.dtype.kind}{array.dtype.itemsize}'
                    if kind=='text' and input_dtype!='S1':
                        raise TypeError(f'{name}: text requires a string or raw S1 array')
                    if kind=='real' and input_dtype not in ('f4','f8'):
                        raise TypeError(f'{name}: real field requires float32 or float64')
                    if kind in ('integer','logical') and input_dtype not in ('i4','i8','u1'):
                        raise TypeError(f'{name}: native integer/logical field requires int32, int64, or bool')
                    if dims is None:
                        dims=tuple(d for d in logical_dims if d!='Time')
                    dims=tuple(dims)
                    if len(dims)!=array.ndim or len(set(dims))!=len(dims):
                        raise ValueError(f'{name}: supply explicit HistoryField dimensions for shape {array.shape}')
                    if set(dims)-set(logical_dims) or set(logical_dims)-{'Time'}-set(dims):
                        raise ValueError(f'{name}: input axes {dims} disagree with Registry {logical_dims}')
                    for dim,size in zip(dims,array.shape):
                        if dim in sizes and sizes[dim]!=size:
                            raise ValueError(f'{name}: {dim}={size} differs from authority {sizes[dim]}')
                        sizes[dim]=int(size)
                    # Raw contiguous host handoff; native file encoding is Rust.
                    handoff=handoff_dir/f'field-{index}.raw'
                    array.tofile(handoff)
                    variable.update(filename=str(handoff),shape=array.shape,input_dimensions=dims,input_dtype=input_dtype,
                                    connectivity_offset=1 if name in converted else 0)
            variable['attribute_order']=list(variable['attrs'])
            variables.append(variable)
        for offset,(name,(array,dims,units)) in enumerate((extra_fields or {}).items()):
            if name in names:
                raise ValueError(f'{stream.name}: extra field {name} is a Registry field of this stream')
            array=np.ascontiguousarray(array,dtype=np.float32); dims=tuple(dims)
            if len(dims)!=array.ndim:
                raise ValueError(f'{name}: extra field axes {dims} do not match shape {array.shape}')
            for dim,size in zip(dims,array.shape):
                if dim in sizes and sizes[dim]!=size:
                    raise ValueError(f'{name}: {dim}={size} differs from authority {sizes[dim]}')
                sizes[dim]=int(size)
            logical=tuple(sorted(dims,key=lambda d:('nVertLevels','nVertLevelsP1','nSoilLevels','nCells').index(d)
                                 if d in ('nVertLevels','nVertLevelsP1','nSoilLevels','nCells') else 9))+('Time',)
            for dim in logical:
                if dim not in dimension_order:
                    dimension_order.append(dim)
            handoff=stage/f'extra-{offset}.raw'
            array.tofile(handoff)
            attrs={'units':str(units),'long_name':name}
            variables.append({'name':name,'dtype':'f8' if precision=='double' else 'f4',
                              'dimensions':tuple(reversed(logical)),'attrs':_json_attrs(attrs),
                              'filename':str(handoff),'shape':array.shape,'input_dimensions':dims,
                              'input_dtype':'f4','connectivity_offset':0,'attribute_order':list(attrs)})
        used={d for v in variables for d in v['dimensions']}
        absent=used-set(sizes)
        if absent: raise ValueError(f'{stream.name}: missing dimension sizes {sorted(absent)}')
        staged=stage/'native.nc'
        append=exists and mode in ('append','overwrite')
        if append:
            # Validate timestamps before selecting a write position.
            with read_stream(path, scratch=stage, select=('xtime',)) as current:
                stamps=current.variables['xtime'].text
                if record_index > len(stamps): raise ValueError('record_index would leave a gap in the native file')
                if mode=='append' and record_index!=len(stamps): raise ValueError('append refuses an existing record')
                if mode=='overwrite' and record_index<len(stamps) and stamps[record_index]!=valid.strftime('%Y-%m-%d_%H:%M:%S'):
                    raise ValueError('overwrite record timestamp differs from the native file')
            shutil.copyfile(path,staged)
        attrs=_json_attrs(getattr(mesh,'attrs',{}))
        if global_attrs: attrs.update(_json_attrs(global_attrs))
        request={'operation':'write','path':str(staged),'format':stream.options.get('io_type','pnetcdf'),
               'dimensions':{d:sizes[d] for d in dimension_order},'dimension_order':dimension_order,
               'variables':variables,'global_attrs':attrs,'global_attribute_order':list(attrs),
               # MPAS v8.4.2 SMIOL smiol.c:15,341-344 reserves a 128 KiB
               # header. Matching field words alone misses this byte contract.
               'header_alignment':131072 if stream.options.get('io_type','pnetcdf').startswith('pnetcdf') else 0,
               'append':append,'record_index':record_index or 0}
        # History and diagnostics files: lossless compressed NetCDF-4 by default (woof.hex.history_compression).
        compression=history_compression.for_stream(stream.name,stream.options,append=append,record_index=record_index)
        if compression is not None:
            request['compression']=compression
        _call(request,stage)
        with staged.open('rb') as handle: os.fsync(handle.fileno())
        _publish_stream(staged, path, mode)
    return path


@dataclass(frozen=True)
class NativeVariable:
    # Context exit closes the mmap. Exception formatting may still represent
    # this descriptor afterward; it must never dereference unmapped storage.
    values: np.ndarray = dataclass_field(repr=False)
    dimensions: tuple[str,...]
    attrs: Mapping[str,Any]
    text: tuple[str,...] = ()


@dataclass(frozen=True)
class NativeFile:
    dimensions: Mapping[str,int]
    variables: Mapping[str,NativeVariable]
    global_attrs: Mapping[str,Any]
    unlimited: tuple[str,...]


def inspect_stream(path: str | Path, *, scratch: str | Path) -> dict:
    """MPAS src/framework/mpas_io_streams.F:4315-4363: typed file header.

    Rust inspects every field without decoding weather arrays or creating raw
    dumps. Preserve stored primitive attribute types and ordered dimensions.
    """
    root=Path(scratch);root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.mpas-header-',dir=root) as temporary:
        return _call({'operation':'inspect','path':str(Path(path).resolve())},Path(temporary))


def _seek_record(stamps, valid_time, whence):
    """MPAS src/framework/mpas_io_streams.F:153-339: all seek modes.

    This selects timestamp metadata only; Rust owns all NetCDF decoding and
    field conversion. Ties retain the first record. Exact fallback retains the
    last matching record; the native >20-record optimization can return first.
    """
    from fractions import Fraction
    from .streams_v842 import _clock
    modes = ('exact_time','nearest','latest_before','earliest_after',
             'latest_strictly_before','earliest_strictly_after')
    whence = whence.lower().removeprefix('mpas_stream_')
    if whence not in modes:
        raise ValueError(f'unrecognized native stream seek position: {whence}')

    def timestamp(text):
        # mpas_set_time truncates timestamp metadata to32characters; rational
        # fractional seconds retain the eight digits of the native time parser.
        text = text[:32].strip().rstrip('\0')
        base, separator, decimal = text.partition('.')
        fraction = decimal[:8] if separator else ''
        return _clock(_date(base)) + (Fraction(int(fraction),10**len(fraction))
                                     if fraction else Fraction(0))

    target = timestamp(valid_time.strftime('%Y-%m-%d_%H:%M:%S.%f')
                       if isinstance(valid_time,datetime) else str(valid_time))
    times = [timestamp(stamp) for stamp in stamps]
    if whence=='exact_time' and len(times)>20:
        period = times[1]-times[0]
        guessed = times[0]
        for index, actual in enumerate(times):
            if guessed==target and actual==guessed:
                return index
            guessed += period
    best = Fraction(10000*86400)
    selected = None
    for index, time in enumerate(times):
        if whence=='exact_time':
            if time==target:
                selected=index
            continue
        eligible = (whence=='nearest' or
            whence=='latest_before' and time<=target or
            whence=='earliest_after' and time>=target or
            whence=='latest_strictly_before' and time<target or
            whence=='earliest_strictly_after' and time>target)
        distance = abs(time-target)
        if eligible and distance<best:
            selected, best = index, distance
    if selected is None:
        raise ValueError(f'native stream has no {whence} record for {valid_time}')
    return selected


def seek_stream(path, *, valid_time, whence='exact_time', scratch):
    """MPAS mpas_io_streams.F:153-339: select native input frame.

    IAU uses nearest at initialization (mpas_atm_core.F:222-233). The record
    index is zero based for the Rust bridge; native Fortran returns one based.
    """
    with read_stream(path,scratch=scratch,select=('xtime',)) as source:
        if source.dimensions.get('Time',0)<=0:
            raise ValueError('native stream has no time records')
        stamps = source.variables['xtime'].text
        if len(stamps)!=source.dimensions['Time']:
            raise ValueError('native xtime record count differs from Time')
        index = _seek_record(stamps,valid_time,whence)
        return index, stamps[index][:32].strip()


@contextmanager
def read_stream(path: str | Path, *, scratch: str | Path, select=(),
                record: int | None = None, layouts: Mapping[str,Mapping] | None = None,
                valid_time=None, whence='exact_time') -> Iterator[NativeFile]:
    """MPAS src/framework/mpas_io_streams.F:2430-3251, native records.

    Arrays are views on Rust-decoded raw storage. Their lifetime is the context;
    the caller copies them into its authority buffers before the context exits.
    Attributes retain primitive types as {dtype, values} bridge descriptors.
    Optional named authority layouts are reordered/cast only inside Rust.
    """
    root=Path(scratch); root.mkdir(parents=True,exist_ok=True)
    if valid_time is not None:
        if record is not None:
            raise ValueError('native stream record and timestamp selection are mutually exclusive')
        record, _ = seek_stream(path,valid_time=valid_time,whence=whence,scratch=scratch)
    elif whence!='exact_time':
        raise ValueError('native stream seek mode requires a timestamp')
    if record is not None and (type(record) is not int or record<0): raise ValueError('record must be a nonnegative integer')
    maps=[]
    with tempfile.TemporaryDirectory(prefix='.mpas-read-',dir=root) as temporary:
        stage=Path(temporary)
        info=_call({'operation':'read','path':str(Path(path).resolve()),'output_directory':str(stage),
                    'select':list(select),'record':record,'layouts':layouts or {}},stage)
        variables={}
        try:
            for raw in info['variables']:
                dtype=np.dtype('S1' if raw['dtype']=='S1' else '='+raw['dtype'])
                shape=tuple(raw['shape'])
                if 0 in shape: array=np.empty(shape,dtype=dtype)
                else:
                    # NumPy memmap retains a descriptor per field. Nested native
                    # restart/reference reads exceed a 1024-descriptor limit.
                    # Rust still owns decoding; these are read-only raw views.
                    with open(raw['filename'],'rb') as handle:
                        try:
                            mapping=mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ,
                                **({'trackfd':False} if os.name=='posix' else {}))
                        except TypeError:
                            # Python before 3.13 lacks trackfd. Preserve raw
                            # words in an immutable buffer without retaining FDs.
                            array=np.ndarray(shape,dtype=dtype,buffer=handle.read())
                        else:
                            maps.append(mapping)
                            array=np.ndarray(shape,dtype=dtype,buffer=mapping)
                variables[raw['name']]=NativeVariable(array,tuple(raw['dimensions']),raw['attrs'],tuple(raw['text']))
            yield NativeFile(info['dimensions'],variables,info['global_attrs'],tuple(info['unlimited']))
        finally:
            for mapping in maps: mapping.close()


@dataclass(frozen=True)
class RestartControl:
    """MPAS src/core_atmosphere/mpas_atm_core.F:129-133,522-530,663-676."""
    config_do_restart: bool = False
    config_do_DAcycling: bool = False

    def __post_init__(self):
        if type(self.config_do_restart) is not bool or type(self.config_do_DAcycling) is not bool:
            raise TypeError('restart and DAcycling controls must be Fortran logical values')

    @property
    def initial_stream(self) -> str:
        return 'restart' if self.config_do_restart else 'input'

    @property
    def initialize_coupled_diagnostics(self) -> bool:
        return not self.config_do_restart or self.config_do_DAcycling

    @property
    def suppressed_initial_outputs(self) -> tuple[str,...]:
        return ('restart','output','diagnostics') if self.config_do_restart else ('restart',)

    @classmethod
    def from_namelist(cls,text: str) -> RestartControl:
        from .config import parse_fortran_namelist
        parsed=parse_fortran_namelist(text)
        values={key.lower():value for group in parsed.values() for key,value in group.items()}
        return cls(values.get('config_do_restart',False),values.get('config_do_dacycling',False))


class NativePoolStreams:
    """Whole native-pool stream boundary; mpas_atm_core.F:663-705,894-902.

    The native-cpu stand-in owns field values. This adapter hands its registered
    storage to the Rust stream bridge, including constituent names and persistent
    physics fields. It performs no diagnostic or physics calculation. The driver
    must synchronize its actual CUDA authority before publishing this boundary.
    """

    def __init__(self, native, layout, streams_path, output_directory, *,
                 list_directory=None, global_attrs=None, field_attrs=None,
                 final_time=None, unavailable_fields=None, stream_definitions=None,
                 alarms=None, contract=None):
        """MPAS mpas_stream_manager.F:4577-4666: selected Registry storage.

        ``contract`` (mpasphys.route_table.StreamContract, the route row's) supplies the Registry, the
        alarms, the restart controls and the writer; None is MPAS-A v8.4.2's.
        """
        from types import SimpleNamespace
        from .mpasphys.route_table import stream_contract
        self.contract = contract = contract if contract is not None else stream_contract()
        self.native, self.layout = native, layout
        directory = {} if list_directory is None else {'list_directory': list_directory}
        self.streams = (dict(stream_definitions) if stream_definitions is not None else
                        contract.registry.parse_streams(streams_path, **directory))
        self.output = Path(output_directory).resolve()
        self.start = _date(layout.configs['config_start_time'])
        self.initial = self.start
        self.control = contract.writer.RestartControl(layout.configs['config_do_restart'],
                                                      layout.configs['config_do_DAcycling'])
        self.mesh = SimpleNamespace(dimensions=dict(layout.dimensions), arrays={},
                                    converted_connectivity=(), attrs={})
        self.attrs = dict(global_attrs or {})
        self.field_attrs = {name:dict(attrs) for name,attrs in (field_attrs or {}).items()}
        self.rows = {row['name']:(row,None) for row in layout.fields}
        self.rows.update({name:(row,index) for row in layout.fields
                          for index,name in enumerate(row['components'])})
        # The foundation records diagnostic storage which does not yet have a
        # computed authority. An explicit driver inventory may retain those
        # omissions; allocated placeholder buffers are never evidence of values.
        # The default still requires the complete native contract. Restart input
        # always requires every active field, independently of this output mask.
        self.unavailable_fields = {name:frozenset(fields)
                                   for name,fields in (unavailable_fields or {}).items()}
        for name,names in self.unavailable_fields.items():
            if name not in self.streams:
                raise ValueError(f'unknown stream in unavailable authority inventory: {name}')
            selected=set(self.streams[name].fields(layout.packages,streams=self.streams))
            if names-selected:
                raise ValueError(f'{name}: unavailable inventory contains unselected fields: {sorted(names-selected)}')
            if names & {'Time','xtime','initial_time'}:
                raise ValueError(f'{name}: native output clock authority cannot be omitted')
        self.records = {}
        configured_stop = final_time or layout.configs.get('config_stop_time','none')
        if configured_stop == 'none':
            registry = contract.registry
            self.final = registry._clock_datetime(registry._advance(registry._clock(self.start),
                registry._interval(layout.configs['config_run_duration'])))
        else:
            self.final = _date(configured_stop)
        self.alarms = alarms if alarms is not None else contract.alarms(
            self.streams, self.start, self.initial, self.final, self.control, layout.packages)

    @property
    def contract(self):
        """The writer's stream contract (MPAS-A v8.4.2's for a writer built without __init__)."""
        contract = self.__dict__.get('_contract')
        if contract is None:
            from .mpasphys.route_table import stream_contract
            contract = self.__dict__['_contract'] = stream_contract()
        return contract

    @contract.setter
    def contract(self, value):
        self.__dict__['_contract'] = value

    def read_input_attributes(self, path, *, scratch):
        """MPAS mpas_io_streams.F:4315-4363: input updates known attributes.

        Runtime input attributes can differ from their Registry defaults. Rust
        reads the file; Python retains only attributes declared by the Registry,
        exactly as native traverses its allocated attribute list. Global source
        metadata remains available unless explicitly overridden by the caller;
        MPAS src/framework/mpas_bootstrapping.F:287-334. Dropping sphere_radius
        makes a native restart silently use radius 1.0.
        """
        with read_stream(path,scratch=scratch) as source:
            self.attrs = {**source.global_attrs, **self.attrs}
            for name,var in source.variables.items():
                if name not in self.rows:
                    continue
                known=self.contract.registry.field_definition(name)['attrs']
                self.field_attrs.setdefault(name,{}).update(
                    {key:value for key,value in var.attrs.items() if key in known})

    def publication_stream(self, stream):
        """MPAS mpas_io_streams.F:1659-1857: driver-owned available storage.

        Keep the original native membership contract intact. A driver-supplied
        missing inventory narrows publication only; the verifier must check it
        against the sealed ratchet. No omission is inferred from buffer values.
        """
        missing=self.unavailable_fields.get(stream.name,frozenset())
        if not missing:
            return stream
        names=stream.fields(self.layout.packages,streams=self.streams)
        return replace(stream,members=tuple(('var',name,'') for name in names if name not in missing))

    def fields(self, stream):
        """MPAS mpas_io_streams.F:1659-1857: field/component handoff.

        Native decomposition includes one dummy entity. Only its declared owned
        extent is published, preserving native one-based connectivity. Missing
        authority is an error; present diagnostic storage is never synthesized.
        """
        from .output import HistoryField
        result, storage = {}, {}
        names = self.publication_stream(stream).fields(self.layout.packages, streams=self.streams)
        for name in names:
            if name in ('xtime','initial_time'):
                continue
            if name not in self.rows:
                raise ValueError(f'{stream.name}: missing native pool authority: {name}')
            row, component = self.rows[name]
            pool, field = row['pool'], row['pool_name']
            if row['kind'] == 'character':
                result[name] = self.native.get_text_field(pool,field,level=1)
                continue
            key = (pool,field)
            if key not in storage:
                storage[key] = self.native.get_field(pool,field,level=1)
            value = storage[key]
            dims = tuple(row['dimensions'])
            if component is not None:
                value, dims = value[component], dims[1:]
            expected = tuple(self.layout.dimensions[d] for d in dims)
            actual = tuple(size+int(d in ('nCells','nEdges','nVertices'))
                           for d,size in zip(dims,expected))
            if value.shape != actual:
                raise ValueError(f'{name}: native pool shape {value.shape} differs from {actual}')
            attrs = dict(self.field_attrs.get(name,{}))
            if name=='Time':
                # mpas_atm_core.F:264-277 initializes the runtime CF epoch,
                # while update_state_time retains the actual accumulated word.
                attrs['units']=f'seconds since {self.initial:%Y-%m-%d %H:%M:%S}'
            result[name] = HistoryField(value[tuple(slice(0,size) for size in expected)],dims,attrs)
        return result

    def write(self, *, step, dt, receipt):
        """MPAS mpas_atm_core.F:675-705: publish, then reset each alarm."""
        when = self.start + timedelta(seconds=step*dt)
        receipt.setdefault('missing_fields',{}).update(
            {name:sorted(fields) for name,fields in self.unavailable_fields.items()})
        for name,stream in self.streams.items():
            if 'output' not in stream.options['type'].split(';'):
                continue
            if not stream.fields(self.layout.packages,streams=self.streams):
                continue
            if not self.alarms.due(name,when):
                continue
            target = (self.output/stream.filename(when,self.initial)).resolve()
            if not target.is_relative_to(self.output):
                raise ValueError(f'{name}: native stream filename escapes output directory')
            record = self.records.get(target,0)
            publication = self.publication_stream(stream)
            kwargs = dict(initial_time=self.initial,active_packages=self.layout.packages,
                          streams=self.streams,global_attrs=self.attrs,
                          field_attrs=self.field_attrs,
                          scratch=self.output/'stream-scratch',record_index=record)
            writer = self.contract.writer
            if name == 'restart':
                writer.write_restart(target,self.mesh,self.fields(stream),when,stream=publication,
                                     timestamp_path=self.output/'restart_timestamp',**kwargs)
            else:
                writer.write_stream(target,publication,self.mesh,self.fields(stream),when,**kwargs)
            self.records[target] = record+1
            self.alarms.reset(name,when)
            receipt.setdefault('streams',[]).append(dict(name=name,step=step,
                    writer='rw_mpas_stream',path=str(target),record_index=record))

    def restore(self, path, *, scratch):
        """MPAS mpas_atm_core.F:129-150: complete native restart ingress.

        Copy into already allocated native buffers before initialization or the
        first step. Rust owns decode, precision and axis conversion. The caller
        still owns diagnostic reinitialization and CUDA upload at this boundary.
        """
        stream = self.streams['restart']
        restored = []
        with self.contract.writer.read_restart(path,valid_time=self.start,scratch=scratch,
                                               active_packages=self.layout.packages,stream=stream,
                                               authority_layout=True) as source:
            # MPAS src/framework/mpas_bootstrapping.F:287-334: retain the
            # geometry and source metadata even without an attribute pre-read.
            initial = _date(source.variables['initial_time'].text[0])
            pending, texts, attributes = {}, [], {}
            for name,var in source.variables.items():
                if name not in self.rows:
                    raise ValueError(f'restart: missing native pool authority: {name}')
                row,component = self.rows[name]
                attributes[name]=dict(var.attrs)
                pool,field = row['pool'],row['pool_name']
                if row['kind'] == 'character':
                    texts.append((row,var.text[0]))
                else:
                    key = (pool,field)
                    if key not in pending:
                        pending[key] = (row,self.native.get_field(pool,field,level=1))
                    buffer = pending[key][1]
                    if component is not None:
                        buffer = buffer[component]
                    target = (buffer[tuple(slice(0,self.layout.dimensions[d])
                                          for d in var.dimensions)] if var.dimensions else buffer)
                    if target.shape != var.values.shape or target.dtype != var.values.dtype:
                        raise ValueError(f'{name}: restart authority shape or kind differs')
                    target[...] = var.values
                restored.append(name)
            for row,text in texts:
                for level in range(1,row['nlevels']+1):
                    self.native.put_text_field(row['pool'],row['pool_name'],text,
                                               level=level,nlevels=row['nlevels'])
            for (pool,field),(row,buffer) in pending.items():
                for level in range(1,row['nlevels']+1):
                    self.native.put_field(pool,field,buffer,level=level,nlevels=row['nlevels'])
            self.initial = initial
            self.attrs = {**source.global_attrs, **self.attrs}
            self.field_attrs.update(attributes)
        self.alarms = self.contract.alarms(self.streams,self.start,self.initial,self.final,
                                           self.control,self.layout.packages)
        return tuple(restored)


def write_restart(path,mesh,fields,valid_time,*,initial_time,active_packages=(),physics=True,
                  stream: Stream | None = None,timestamp_path: str | Path | None = None,**kwargs):
    """MPAS src/core_atmosphere/mpas_atm_core.F:894-902, timestamp after write."""
    result=write_stream(path,stream or registry_stream('restart',physics=physics),mesh,fields,
                        valid_time,initial_time=initial_time,active_packages=active_packages,physics=physics,**kwargs)
    if timestamp_path is not None:
        target=Path(timestamp_path);target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w',prefix='.restart-time-',dir=target.parent,delete=False) as handle:
            temporary=Path(handle.name)
            handle.write(' '+_date(valid_time).strftime('%Y-%m-%d_%H:%M:%S')+'\n')
            handle.flush();os.fsync(handle.fileno())
        try: os.replace(temporary,target)
        finally: temporary.unlink(missing_ok=True)
    return result


@contextmanager
def read_restart(path,*,valid_time,scratch,active_packages=(),physics=True,
                 stream: Stream | None = None, authority_layout: bool = False,
                 zero_based_connectivity=()) -> Iterator[NativeFile]:
    """MPAS src/core_atmosphere/mpas_atm_core.F:129-150: restart input.

    Choose the native exact frame, then require every active persistent Registry field.
    Coupled diagnostic reinitialization is the driver's RestartControl decision.
    authority_layout returns numeric fields in Registry axes without Time and
    restores RKIND float32 through Rust, ready for the caller's device upload.
    """
    record,_=seek_stream(path,valid_time=valid_time,whence='exact_time',scratch=scratch)
    selected=(stream or registry_stream('restart',physics=physics)).fields(active_packages,physics=physics)
    if not selected: raise ValueError('restart stream is inactive or has no fields')
    layouts={}
    if zero_based_connectivity and not authority_layout:
        raise ValueError('connectivity conversion requires authority_layout')
    if authority_layout:
        for name in selected:
            definition=field_definition(name,physics=physics)
            if definition['type']=='text': continue
            layouts[name]={'dtype':'f4' if definition['type']=='real' else 'i4',
                          'dimensions':[d for d in definition['dimensions'] if d!='Time'],
                          'expected_dimensions':list(reversed(definition['dimensions'])),
                          'connectivity_offset':-1 if name in zero_based_connectivity else 0}
    with read_stream(path,scratch=scratch,select=selected,record=record,layouts=layouts) as native:
        if native.unlimited!=('Time',): raise ValueError('native restart requires an unlimited Time dimension')
        for name,var in native.variables.items():
            definition=field_definition(name,physics=physics)
            expected_dims=tuple(layouts[name]['dimensions']) if name in layouts else tuple(reversed(definition['dimensions']))
            if var.dimensions!=expected_dims:
                raise ValueError(f'{name}: native restart dimension order differs from Registry')
            kind=definition['type']
            expected={'text':('|S1',),'integer':('int32',),'logical':('int32',),'real':('float32','float64')}[kind]
            if str(var.values.dtype) not in expected:
                raise ValueError(f'{name}: native restart primitive differs from Registry')
            if kind=='text' and native.dimensions.get('StrLen')!=64:
                raise ValueError(f'{name}: native restart requires StrLen=64')
        yield native
