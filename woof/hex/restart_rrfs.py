"""Native RRFS fork stream I/O and restart orchestration.

Membership: MPAS-Model-rrfs/src/core_atmosphere/Registry.xml:498-1332.
Initialization: MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:120-166,536-541,662-678,898-909.
Primitive encoding/decoding and dimension permutation live in rw_mpas_stream;
this module only hands arrays and metadata to that Rust artifact.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterator, Mapping

import numpy as np

from .streams_rrfs import Stream, field_definition, registry_stream
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
    """MPAS-Model-rrfs/src/framework/mpas_io_streams.F:38-4361.

    Use the shared installed-artifact resolver without an implicit build.
    Override/PATH-only lookup missed libexec and user bridges in clean runs.
    An explicit missing setting still fails rather than choosing another file.
    """
    return str(resolve(_STREAM_ENGINE))


def _json_attrs(attrs: Mapping[str, Any], *, real_precision: str = 'f4') -> dict:
    """MPAS-Model-rrfs/src/framework/mpas_io_streams.F:4140-4207,4315-4361.

    Numeric attribute encoding stays in Rust. Untyped real attributes use the
    native RKIND, or the stream precision for global attributes. An explicit
    bridge dtype preserves an attribute read from a native file.
    """
    def plain(value):
        """MPAS-Model-rrfs/src/framework/mpas_io_streams.F:4140-4207,4315-4361."""
        if isinstance(value, Mapping): return {str(k):plain(v) for k,v in value.items()}
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, (tuple,list)): return [plain(v) for v in value]
        return value
    result = {}
    for name,value in attrs.items():
        if name == '_NCProperties': continue
        value = plain(value)
        # The bridge encodes _FillValue in the variable primitive, as NetCDF
        # requires. Explicit typed attributes already carry their authority.
        if name != '_FillValue' and not isinstance(value, Mapping):
            values = value if isinstance(value, list) else [value]
            if values and all(isinstance(v, (int,float)) for v in values):
                dtype = real_precision if any(isinstance(v,float) for v in values) else 'i4'
                value = {'dtype':dtype,'values':value}
        result[str(name)] = value
    return result


def _call(request: dict, directory: Path) -> dict:
    """MPAS-Model-rrfs/src/framework/mpas_io_streams.F:38-4361. Run the real Rust stream reader/writer and propagate its error."""
    request_path = directory / 'request.json'
    request_path.write_text(json.dumps(request, allow_nan=False))
    result = subprocess.run([_bridge(), str(request_path)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    response = json.loads(result.stdout)
    if response.get('schema') != 'mpas-native-stream/v1':
        raise ValueError('stream bridge ABI mismatch')
    return response


def _date(value: datetime | str) -> datetime:
    """MPAS-Model-rrfs/src/framework/mpas_io_streams.F:38-4361."""
    from .output import _as_datetime
    return _as_datetime(value)


def write_stream(path: str | Path, stream: Stream | str, mesh: object,
                 fields: Mapping[str, Any], valid_time: datetime | str, *,
                 initial_time: datetime | str,
                 active_packages=(), physics: bool = True,
                 streams: Mapping[str, Stream] | None = None,
                 global_attrs: Mapping[str, Any] | None = None,
                 scratch: str | Path | None = None,
                 record_index: int | None = None,
                 update_global_attrs: bool = True,
                 field_attrs: Mapping[str, Mapping[str, Any]] | None = None,
                 registry_attributes: bool = True,
                 extra_fields: Mapping[str, Any] | None = None) -> Path:
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/Registry.xml:498-1332, strict stream snapshot.

    Every active field is required. Arrays are handed over in their named
    in-memory axes; Rust reorders to reversed Registry axes. Connectivity is
    offset only for variables marked converted by the mesh authority.
    Existing files follow the native clobber policy; writes publish atomically.

    ``field_attrs``, ``registry_attributes`` and ``extra_fields`` are the v8.4.2 writer's
    (restart_v842.write_stream), so one stream writer (mpasphys.streams_output_v842.PhysicsStreams)
    drives either contract: input-file attributes after the Registry's, the Registry attributes
    themselves, and variables the Registry does not declare (name -> (float32 array, axes without
    Time, units)) after the stream's own.
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
    with tempfile.TemporaryDirectory(prefix='.mpas-stream-',dir=scratch_root) as temporary:
        stage=Path(temporary); variables=[]
        for index,name in enumerate(names):
            definition=stream.field_definition(name,physics=physics)
            logical_dims=tuple(definition['dimensions'])
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
                    raw=fields.get(name, HistoryField(np.asarray([(valid-initial).total_seconds()],dtype=np.float32),('Time',)))
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
                    handoff=stage/f'field-{index}.raw'
                    array.tofile(handoff)
                    variable.update(filename=str(handoff),shape=array.shape,input_dimensions=dims,input_dtype=input_dtype,
                                    connectivity_offset=1 if name in converted else 0)
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
            order=('nVertLevels','nVertLevelsP1','nSoilLevels','nCells')
            logical=tuple(sorted(dims,key=lambda d:order.index(d) if d in order else 9))+('Time',)
            handoff=stage/f'extra-{offset}.raw'
            array.tofile(handoff)
            variables.append({'name':name,'dtype':'f8' if precision=='double' else 'f4',
                              'dimensions':tuple(reversed(logical)),
                              'attrs':_json_attrs({'units':str(units),'long_name':name}),
                              'filename':str(handoff),'shape':array.shape,'input_dimensions':dims,
                              'input_dtype':'f4','connectivity_offset':0})
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
        attribute_precision = 'f8' if precision == 'double' else 'f4'
        attrs=_json_attrs(getattr(mesh,'attrs',{}),real_precision=attribute_precision)
        if global_attrs: attrs.update(_json_attrs(global_attrs,real_precision=attribute_precision))
        if not update_global_attrs or not global_attributes_enabled(stream): attrs = {}
        request={'operation':'write','path':str(staged),'format':stream.options.get('io_type','pnetcdf'),
               'dimensions':{d:sizes[d] for d in used},'variables':variables,'global_attrs':attrs,
               'append':append,'record_index':record_index or 0}
        # History and diagnostics files: lossless compressed NetCDF-4 by default (woof.hex.history_compression).
        compression=history_compression.for_stream(stream.name,stream.options,append=append,record_index=record_index)
        if compression is not None:
            request['compression']=compression
        _call(request,stage)
        with staged.open('rb') as handle: os.fsync(handle.fileno())
        # Publish on the destination filesystem even when scratch is elsewhere.
        with tempfile.NamedTemporaryFile(prefix=f'.{path.name}.',dir=path.parent,delete=False) as publish:
            publish_path=Path(publish.name)
            with staged.open('rb') as handle: shutil.copyfileobj(handle,publish)
            publish.flush(); os.fsync(publish.fileno())
        try:
            if mode=='never_modify':
                os.link(publish_path,path)
                publish_path.unlink()
            else: os.replace(publish_path,path)
            directory_fd=os.open(path.parent,os.O_RDONLY)
            try: os.fsync(directory_fd)
            finally: os.close(directory_fd)
        finally:
            publish_path.unlink(missing_ok=True)
    if done_marker_enabled(stream): write_done_marker(path)
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


@contextmanager
def read_stream(path: str | Path, *, scratch: str | Path, select=(),
                record: int | None = None, layouts: Mapping[str,Mapping] | None = None) -> Iterator[NativeFile]:
    """RRFS MPAS-Model-rrfs/src/framework/mpas_io_streams.F:2430-3251, native records.

    Arrays are views on Rust-decoded raw storage. Their lifetime is the context;
    the caller copies them into its authority buffers before the context exits.
    Attributes retain primitive types as {dtype, values} bridge descriptors.
    Optional named authority layouts are reordered/cast only inside Rust.
    """
    root=Path(scratch); root.mkdir(parents=True,exist_ok=True)
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
                    array=np.memmap(raw['filename'],mode='r',dtype=dtype,shape=shape)
                    maps.append(array)
                variables[raw['name']]=NativeVariable(array,tuple(raw['dimensions']),raw['attrs'],tuple(raw['text']))
            yield NativeFile(info['dimensions'],variables,info['global_attrs'],tuple(info['unlimited']))
        finally:
            for array in maps: array._mmap.close()


@dataclass(frozen=True)
class RestartControl:
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:137-166,536-541,662-678."""
    config_do_restart: bool = False
    config_do_DAcycling: bool = False

    def __post_init__(self):
        """MPAS-Model-rrfs/src/framework/mpas_io_streams.F:38-4361."""
        if type(self.config_do_restart) is not bool or type(self.config_do_DAcycling) is not bool:
            raise TypeError('restart and DAcycling controls must be Fortran logical values')

    @property
    def initial_stream(self) -> str:
        """MPAS-Model-rrfs/src/framework/mpas_io_streams.F:38-4361."""
        return 'restart' if self.config_do_restart else 'da_state' if self.config_do_DAcycling else 'input'

    @property
    def initialize_coupled_diagnostics(self) -> bool:
        """MPAS-Model-rrfs/src/framework/mpas_io_streams.F:38-4361."""
        return not self.config_do_restart or self.config_do_DAcycling

    @property
    def suppressed_initial_outputs(self) -> tuple[str,...]:
        """MPAS-Model-rrfs/src/framework/mpas_io_streams.F:38-4361."""
        return ('restart','output','diagnostics') if self.config_do_restart else ('restart','da_state') if self.config_do_DAcycling else ('restart',)

    @classmethod
    def from_namelist(cls,text: str) -> RestartControl:
        """MPAS-Model-rrfs/src/framework/mpas_io_streams.F:38-4361."""
        from .config import parse_fortran_namelist
        parsed=parse_fortran_namelist(text)
        values={key.lower():value for group in parsed.values() for key,value in group.items()}
        return cls(values.get('config_do_restart',False),values.get('config_do_dacycling',False))


def write_restart(path,mesh,fields,valid_time,*,initial_time,active_packages=(),physics=True,
                  stream: Stream | None = None,timestamp_path: str | Path | None = None,**kwargs):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:898-909, timestamp after write."""
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
                 zero_based_connectivity=(), streams=None) -> Iterator[NativeFile]:
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:137-166: restart input.

    Choose an exact xtime, then require every active persistent Registry field.
    Coupled diagnostic reinitialization is the driver's RestartControl decision.
    authority_layout returns numeric fields in Registry axes without Time and
    restores RKIND float32 through Rust, ready for the caller's device upload.
    """
    stamp=_date(valid_time).strftime('%Y-%m-%d_%H:%M:%S')
    with read_stream(path,scratch=scratch,select=('xtime',)) as time_file:
        matches=[i for i,t in enumerate(time_file.variables['xtime'].text) if t==stamp]
    if len(matches)!=1: raise ValueError(f'restart must contain exactly one record at {stamp}')
    stream = stream or registry_stream('restart',physics=physics)
    selected=stream.fields(active_packages,physics=physics,streams=streams)
    if not selected: raise ValueError('restart stream is inactive or has no fields')
    layouts={}
    if zero_based_connectivity and not authority_layout:
        raise ValueError('connectivity conversion requires authority_layout')
    if authority_layout:
        for name in selected:
            definition=stream.field_definition(name,physics=physics)
            if definition['type']=='text': continue
            layouts[name]={'dtype':'f4' if definition['type']=='real' else 'i4',
                          'dimensions':[d for d in definition['dimensions'] if d!='Time'],
                          'expected_dimensions':list(reversed(definition['dimensions'])),
                          'connectivity_offset':-1 if name in zero_based_connectivity else 0}
    with read_stream(path,scratch=scratch,select=selected,record=matches[0],layouts=layouts) as native:
        if native.unlimited!=('Time',): raise ValueError('native restart requires an unlimited Time dimension')
        for name,var in native.variables.items():
            definition=stream.field_definition(name,physics=physics)
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


def global_attributes_enabled(stream: Stream) -> bool:
    """RRFS MPAS-Model-rrfs/src/framework/xml_stream_parser.c:1603-1623."""
    setting = stream.options.get('gattr_update', 'yes')
    return 'yes' in setting or 'no' not in setting


def done_marker_enabled(stream: Stream) -> bool:
    """RRFS MPAS-Model-rrfs/src/framework/xml_stream_parser.c:1741-1752."""
    return 'yes' in stream.options.get('output_done_marker', '')


def write_done_marker(path: str | Path, *, now: datetime | None = None, rank: int = 0,
                      io_rank: int = 0, block_write: bool = False) -> Path | None:
    """RRFS MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:7196-7225.

    Publish the native local wall-clock timestamp after the stream is closed.
    Rank arguments describe the actual writer, including block-write ownership.
    """
    if rank != io_rank and not block_write: return None
    marker = Path(str(path) + '.done')
    with marker.open('w') as handle:
        handle.write((now or datetime.now()).strftime('%Y-%m-%d %H:%M:%S') + '\n')
    return marker
