"""Serial history snapshots and native MPAS-A v8.4.2 stream output.

Membership and metadata follow src/core_atmosphere/Registry.xml:927-1029.
The snapshot API also accepts explicitly named extra diagnostic fields for
existing consumers. Use write_native_stream for exact Registry membership,
package activation and native attributes. File I/O is rw_mpas_stream Rust.
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .errors import ConfigurationRefusal
from .streams_v842 import _REGISTRY, _all_packages, field_definition, registry_stream


_FROZEN_OUTPUT_SOURCE = "MPAS-Model-v8.4.2/src/core_atmosphere/Registry.xml:927-1029"
HISTORY_OUTPUT_VARIABLES = registry_stream('output').fields(_all_packages())

_STR_LEN = 64

# The invariant fields named by the frozen atmosphere ``output`` stream.  A
# field is only emitted when it is present on the supplied mesh.
HISTORY_MESH_VARIABLES: tuple[str, ...] = (
    "latCell",
    "lonCell",
    "xCell",
    "yCell",
    "zCell",
    "indexToCellID",
    "latEdge",
    "lonEdge",
    "xEdge",
    "yEdge",
    "zEdge",
    "indexToEdgeID",
    "latVertex",
    "lonVertex",
    "xVertex",
    "yVertex",
    "zVertex",
    "indexToVertexID",
    "cellsOnEdge",
    "nEdgesOnCell",
    "nEdgesOnEdge",
    "edgesOnCell",
    "edgesOnEdge",
    "weightsOnEdge",
    "dvEdge",
    "dcEdge",
    "angleEdge",
    "areaCell",
    "areaTriangle",
    "cellsOnCell",
    "verticesOnCell",
    "verticesOnEdge",
    "edgesOnVertex",
    "cellsOnVertex",
    "kiteAreasOnVertex",
    "meshDensity",
    "zgrid",
    "fzm",
    "fzp",
    "zz",
)

_CONNECTIVITY_VARIABLES = {
    "cellsOnEdge",
    "edgesOnCell",
    "edgesOnEdge",
    "cellsOnCell",
    "verticesOnCell",
    "verticesOnEdge",
    "edgesOnVertex",
    "cellsOnVertex",
    "advCellsForEdge",
    "edgesOnCellOne",
    "edgesOnCellTwo",
}

_ENTITY_DIMENSIONS = ("nCells", "nEdges", "nVertices")
_VERTICAL_DIMENSIONS = ("nVertLevels", "nVertLevelsP1", "nVertLevelsP2")

# v8.4.2 metadata and logical dimensions for all registered stream fields.
_FIELD_METADATA = {name:dict(value['attrs']) for name,value in _REGISTRY['fields'].items()}
_KNOWN_FIELD_DIMENSIONS = {
    name:tuple(d for d in value['dimensions'] if d != 'Time')
    for name,value in _REGISTRY['fields'].items()
}


@dataclass(frozen=True, slots=True)
class HistoryField:
    """A history field and the dimensions of its in-memory array.

    Dimension names describe the axes of ``values``.  The writer transposes
    these axes into MPAS NetCDF order: ``Time``, horizontal entity, vertical,
    then any remaining dimensions.  Thus both the port's readable logical
    layout ``(nVertLevels, nCells)`` and an already on-disk-shaped
    ``(nCells, nVertLevels)`` input are accepted without changing values.
    """

    values: NDArray[Any] | Sequence[Any]
    dimensions: tuple[str, ...] | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)


# Friendly short alias for callers constructing many variables.
Field = HistoryField


@dataclass(frozen=True, slots=True)
class HistoryStreamOptions:
    """The honestly supported slice of MPAS output-stream options."""

    stream_name: str = "output"
    io_type: str = "netcdf4"
    precision: str = "native"
    clobber_mode: str = "never_modify"
    runtime_format: str = "separate_file"
    output_interval: str | None = None
    filename_template: str | None = None
    filename_interval: str | None = None
    reference_time: str = "initial_time"
    record_interval: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "HistoryStreamOptions":
        if raw is None:
            return cls()
        aliases = {"name": "stream_name", "type": "stream_name"}
        allowed = set(cls.__dataclass_fields__)
        normalized: dict[str, Any] = {}
        for original, value in raw.items():
            name = aliases.get(original, original)
            if name not in allowed:
                raise ConfigurationRefusal(
                    original,
                    value,
                    "this snapshot option has no defined effect; use parse_streams for native deck scheduling",
                    f"omit {original}",
                )
            normalized[name] = value
        return cls(**normalized)

    def validate(self) -> "HistoryStreamOptions":
        if self.stream_name not in {"output", "history"}:
            raise ConfigurationRefusal(
                "stream_name",
                self.stream_name,
                "this compatibility snapshot API handles output/history; use write_native_stream for other v8.4.2 streams",
                "stream_name='output'",
            )
        if self.io_type not in {"netcdf", "serial_netcdf", "netcdf4", "pnetcdf", "pnetcdf,cdf5"}:
            raise ConfigurationRefusal(
                "io_type",
                self.io_type,
                "the serial Rust bridge supports NetCDF, CDF-2, CDF-5 and NetCDF4 file formats",
                "io_type='netcdf4', 'netcdf', 'pnetcdf', or 'pnetcdf,cdf5'",
            )
        if self.precision not in {"native", "single", "double"}:
            raise ConfigurationRefusal(
                "precision",
                self.precision,
                "the frozen stream parser only defines native, single, and double real precision",
                "precision='native'",
            )
        if self.clobber_mode not in {"never_modify", "truncate", "replace_files"}:
            raise ConfigurationRefusal(
                "clobber_mode",
                self.clobber_mode,
                "record append/overwrite semantics are not implemented by the snapshot writer",
                "clobber_mode='never_modify' or clobber_mode='truncate'",
            )
        if self.runtime_format != "separate_file":
            raise ConfigurationRefusal(
                "runtime_format",
                self.runtime_format,
                "only one complete history file per call is implemented",
                "runtime_format='separate_file'",
            )
        for knob in (
            "output_interval",
            "filename_template",
            "filename_interval",
            "record_interval",
        ):
            value = getattr(self, knob)
            if value not in (None, "none"):
                raise ConfigurationRefusal(
                    knob,
                    value,
                    "snapshot calls do not consume stream alarms; use parse_streams and StreamAlarms for native decks",
                    f"{knob}=None and call write_history at the desired time",
                )
        if self.reference_time != "initial_time":
            raise ConfigurationRefusal(
                "reference_time",
                self.reference_time,
                "only initial_time-relative CF time is emitted",
                "reference_time='initial_time'",
            )
        return self

    @property
    def netcdf_format(self) -> str:
        if os.environ.get("WOOF_HEX_NETCDF_64BIT", "0").strip().lower() in ("1", "true", "yes", "on"):
            return "NETCDF4" if self.io_type == "netcdf4" else "NETCDF3_64BIT_DATA"
        return {"netcdf4":"NETCDF4", "pnetcdf,cdf5":"NETCDF3_64BIT_DATA"}.get(self.io_type,"NETCDF3_64BIT_OFFSET")

    def netcdf_format_for_max_var_bytes(self, max_var_bytes: int) -> str:
        fmt = self.netcdf_format
        if fmt == "NETCDF3_64BIT_OFFSET" and int(max_var_bytes) >= (1 << 31):
            return "NETCDF3_64BIT_DATA"
        return fmt


def format_mpas_filename(template: str, valid_time: datetime | str) -> str:
    """Expand the time tokens accepted by frozen MPAS filename templates."""

    stamp = _as_datetime(valid_time)
    replacements = {
        "$Y": f"{stamp.year:04d}",
        "$M": f"{stamp.month:02d}",
        "$D": f"{stamp.day:02d}",
        "$h": f"{stamp.hour:02d}",
        "$m": f"{stamp.minute:02d}",
        "$s": f"{stamp.second:02d}",
    }
    result = str(template)
    for token, replacement in replacements.items():
        result = result.replace(token, replacement)
    if "$" in result:
        token = result[result.index("$") : result.index("$") + 2]
        raise ConfigurationRefusal(
            "filename_template",
            template,
            f"unrecognized MPAS filename token {token!r}",
            "use only $Y, $M, $D, $h, $m, and $s",
        )
    return result


def _as_datetime(value: datetime | str | np.datetime64) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("valid_time must not be NaT")
        microseconds = value.astype("datetime64[us]").astype(np.int64)
        result = datetime.fromtimestamp(float(microseconds) / 1_000_000.0, timezone.utc)
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        result = datetime.fromisoformat(text.replace("_", "T", 1))
    if result.tzinfo is not None:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result.replace(microsecond=0)


def _time_values(
    valid_time: datetime | str | np.datetime64 | Sequence[datetime | str | np.datetime64],
) -> tuple[datetime, ...]:
    if isinstance(valid_time, (datetime, str, np.datetime64)):
        return (_as_datetime(valid_time),)
    values = tuple(_as_datetime(value) for value in valid_time)
    if not values:
        raise ValueError("valid_time must contain at least one record")
    return values


def _mpas_time_string(value: datetime) -> str:
    return value.strftime("%Y-%m-%d_%H:%M:%S")


def _mesh_arrays(mesh: object) -> Mapping[str, Any]:
    arrays = getattr(mesh, "arrays", None)
    if arrays is not None:
        return arrays
    return {
        name: getattr(mesh, name)
        for name in HISTORY_MESH_VARIABLES
        if hasattr(mesh, name)
    }


def _mesh_dimensions(mesh: object) -> dict[str, int]:
    dimensions = getattr(mesh, "dimensions", None)
    if dimensions is not None:
        return {name: int(value) for name, value in dimensions.items()}
    result: dict[str, int] = {}
    arrays = _mesh_arrays(mesh)
    for name, variable in (
        ("nCells", "latCell"),
        ("nEdges", "latEdge"),
        ("nVertices", "latVertex"),
    ):
        if variable in arrays:
            result[name] = int(np.asarray(arrays[variable]).size)
    return result


def _mesh_variable_dimensions(mesh: object) -> Mapping[str, tuple[str, ...]]:
    return getattr(mesh, "variable_dimensions", {})


def _mesh_variable_attrs(mesh: object) -> Mapping[str, Mapping[str, Any]]:
    return getattr(mesh, "variable_attrs", {})


def _plain_attr(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _field_parts(
    name: str,
    raw: Any,
    field_dimensions: Mapping[str, Sequence[str]] | None,
    field_attrs: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[NDArray[Any], tuple[str, ...] | None, dict[str, Any]]:
    if isinstance(raw, HistoryField):
        array = np.asarray(raw.values)
        dimensions = raw.dimensions
        attrs = dict(raw.attrs)
    else:
        array = np.asarray(raw)
        dimensions = None
        attrs = {}
    if field_dimensions is not None and name in field_dimensions:
        supplied = tuple(str(dim) for dim in field_dimensions[name])
        if dimensions is not None and tuple(dimensions) != supplied:
            raise ValueError(f"{name}: HistoryField dimensions disagree with field_dimensions")
        dimensions = supplied
    if field_attrs is not None and name in field_attrs:
        attrs.update(field_attrs[name])
    return array, dimensions, attrs


def _infer_vertical_size(
    fields: Mapping[str, Any],
    field_dimensions: Mapping[str, Sequence[str]] | None,
    explicit: int | None,
) -> int | None:
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("n_vert_levels must be positive")
        return int(explicit)
    for name, raw in fields.items():
        array, dimensions, _ = _field_parts(name, raw, field_dimensions, None)
        if dimensions is None:
            continue
        for axis, dim in enumerate(dimensions):
            if dim == "nVertLevels":
                return int(array.shape[axis])
            if dim == "nVertLevelsP1":
                return int(array.shape[axis]) - 1
            if dim == "nVertLevelsP2":
                return int(array.shape[axis]) - 2
    return None


def _dimension_candidates(
    size: int,
    sizes: Mapping[str, int],
    *,
    exclude: set[str],
) -> list[str]:
    preference = [*_ENTITY_DIMENSIONS, *_VERTICAL_DIMENSIONS]
    preference.extend(name for name in sizes if name not in preference)
    return [name for name in preference if name not in exclude and sizes.get(name) == size]


def _infer_field_dimensions(
    name: str,
    array: NDArray[Any],
    sizes: dict[str, int],
    n_times: int,
) -> tuple[str, ...]:
    shape = array.shape

    known = _KNOWN_FIELD_DIMENSIONS.get(name)
    if known is not None:
        candidates = [known]
        if len(known) > 1:
            candidates.append(tuple(reversed(known)))
        for candidate in candidates:
            if len(candidate) != array.ndim:
                continue
            compatible = True
            inferred_vertical: dict[str, int] = {}
            for axis, dim in enumerate(candidate):
                size = int(shape[axis])
                if dim in _ENTITY_DIMENSIONS and sizes.get(dim) != size:
                    compatible = False
                    break
                if dim == "nVertLevels":
                    inferred_vertical[dim] = size
                elif dim == "nVertLevelsP1":
                    inferred_vertical["nVertLevels"] = size - 1
                elif dim == "nVertLevelsP2":
                    inferred_vertical["nVertLevels"] = size - 2
            if not compatible:
                continue
            if "nVertLevels" in inferred_vertical:
                nlev = inferred_vertical["nVertLevels"]
                expected = sizes.get("nVertLevels")
                if nlev <= 0 or (expected is not None and expected != nlev):
                    continue
                sizes.setdefault("nVertLevels", nlev)
                sizes.setdefault("nVertLevelsP1", nlev + 1)
                sizes.setdefault("nVertLevelsP2", nlev + 2)
            return candidate

    dimensions: list[str | None] = [None] * array.ndim
    used: set[str] = set()

    # Multiple records are unambiguous when the first axis equals Time.  A
    # single record is intentionally not guessed from an arbitrary size-one
    # axis; callers can name it explicitly.
    if n_times > 1 and shape and shape[0] == n_times:
        dimensions[0] = "Time"
        used.add("Time")

    for axis, size in enumerate(shape):
        if dimensions[axis] is not None:
            continue
        entity = [dim for dim in _ENTITY_DIMENSIONS if sizes.get(dim) == size]
        if len(entity) == 1 and entity[0] not in used:
            dimensions[axis] = entity[0]
            used.add(entity[0])

    unresolved = [axis for axis, dim in enumerate(dimensions) if dim is None]
    # A single non-entity dimension on a horizontally located field is the
    # vertical axis.  This is the common MPAS (level, entity) authority layout.
    has_entity = any(dim in _ENTITY_DIMENSIONS for dim in dimensions)
    if has_entity and len(unresolved) == 1:
        axis = unresolved.pop()
        size = shape[axis]
        vertical = [dim for dim in _VERTICAL_DIMENSIONS if sizes.get(dim) == size]
        if len(vertical) == 1:
            dimensions[axis] = vertical[0]
            used.add(vertical[0])
        elif "nVertLevels" not in sizes:
            sizes["nVertLevels"] = size
            sizes["nVertLevelsP1"] = size + 1
            sizes["nVertLevelsP2"] = size + 2
            dimensions[axis] = "nVertLevels"
            used.add("nVertLevels")

    for axis, dim in enumerate(dimensions):
        if dim is not None:
            continue
        candidates = _dimension_candidates(shape[axis], sizes, exclude=used)
        if len(candidates) == 1:
            dimensions[axis] = candidates[0]
            used.add(candidates[0])
            continue
        generated = f"{name}_dim_{axis}"
        sizes.setdefault(generated, int(shape[axis]))
        dimensions[axis] = generated
        used.add(generated)
    return tuple(str(dim) for dim in dimensions)


def _canonical_dimension_order(dimensions: Sequence[str]) -> tuple[str, ...]:
    dims = tuple(dimensions)
    result: list[str] = []
    if "Time" in dims:
        result.append("Time")
    result.extend(dim for dim in _ENTITY_DIMENSIONS if dim in dims)
    result.extend(dim for dim in _VERTICAL_DIMENSIONS if dim in dims)
    result.extend(dim for dim in dims if dim not in result)
    return tuple(result)


def _mesh_dims_for_array(
    name: str,
    array: NDArray[Any],
    supplied: Mapping[str, tuple[str, ...]],
    sizes: dict[str, int],
) -> tuple[str, ...]:
    if name in supplied:
        dims = tuple(supplied[name])
        if len(dims) == array.ndim and all(sizes.get(d, array.shape[i]) == array.shape[i] for i, d in enumerate(dims)):
            return dims
    inferred: list[str] = []
    used: set[str] = set()
    for axis, size in enumerate(array.shape):
        candidates = _dimension_candidates(int(size), sizes, exclude=used)
        if not candidates:
            dim = f"{name}_dim_{axis}"
            sizes[dim] = int(size)
        else:
            dim = candidates[0]
        inferred.append(dim)
        used.add(dim)
    return tuple(inferred)


def write_history(
    path: str | Path,
    mesh: object,
    fields: Mapping[str, Any],
    valid_time: datetime | str | np.datetime64 | Sequence[datetime | str | np.datetime64],
    *,
    initial_time: datetime | str | np.datetime64 | None = None,
    time_seconds: float | Sequence[float] | None = None,
    n_vert_levels: int | None = None,
    field_dimensions: Mapping[str, Sequence[str]] | None = None,
    field_attrs: Mapping[str, Mapping[str, Any]] | None = None,
    global_attrs: Mapping[str, Any] | None = None,
    include_mesh: bool = True,
    stream_options: HistoryStreamOptions | Mapping[str, Any] | None = None,
) -> Path:
    """Write legacy history snapshots through the real Rust stream artifact.

    This API preserves snapshot consumers' float dtypes and CF time metadata.
    Exact native membership is the write_native_stream contract below.
    MPAS src/core_atmosphere/Registry.xml:927-1029,1674-1682.
    """
    import tempfile
    from .restart_v842 import _call, _json_attrs
    from .history_writer import atomic_write
    options = (stream_options if isinstance(stream_options, HistoryStreamOptions)
               else HistoryStreamOptions.from_mapping(stream_options)).validate()
    destination = Path(path).expanduser().resolve()
    if destination.exists() and options.clobber_mode == 'never_modify':
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    times = _time_values(valid_time)
    reference = _as_datetime(initial_time) if initial_time is not None else times[0]
    if time_seconds is None:
        seconds = [(t-reference).total_seconds() for t in times]
    elif np.isscalar(time_seconds):
        if len(times)!=1: raise ValueError('scalar time_seconds requires one record')
        seconds = [float(time_seconds)]
    else: seconds = list(time_seconds)
    if len(seconds)!=len(times) or not np.all(np.isfinite(seconds)):
        raise ValueError('time_seconds must be finite with one value per record')
    arrays=_mesh_arrays(mesh)
    sizes={k:v for k,v in _mesh_dimensions(mesh).items() if k not in ('Time','StrLen')}
    sizes.update(Time=len(times),StrLen=_STR_LEN)
    nlev=_infer_vertical_size(fields,field_dimensions,n_vert_levels)
    if nlev is not None:
        for d,n in (('nVertLevels',nlev),('nVertLevelsP1',nlev+1),('nVertLevelsP2',nlev+2)):
            if d in sizes and sizes[d]!=n:raise ValueError(f'mesh {d} disagrees with {n}')
            sizes[d]=n
    with tempfile.TemporaryDirectory(prefix='.history-stream-',dir=destination.parent) as temporary:
        stage=Path(temporary);variables=[]
        def handoff(name,array,dims,attrs,offset=0):
            if not array.dtype.isnative:raise TypeError(f'{name}: handoff array must have native byte order')
            if len(dims)!=array.ndim or len(set(dims))!=len(dims):raise ValueError(f'{name}: dimensions do not match shape')
            for d,n in zip(dims,array.shape):
                if d in sizes and sizes[d]!=n:raise ValueError(f'{name}: {d} has {n}, expected {sizes[d]}')
                sizes[d]=int(n)
            input_dtype=f'{array.dtype.kind}{array.dtype.itemsize}'
            dtype=('f4' if options.precision=='single' else 'f8') if array.dtype.kind=='f' and options.precision!='native' else input_dtype
            if offset:dtype='i4'
            raw=stage/f'field-{len(variables)}.raw';array.tofile(raw)
            disk_dims=_canonical_dimension_order(dims if name in HISTORY_MESH_VARIABLES and name not in fields else (dims if 'Time' in dims else ('Time',*dims)))
            variables.append(dict(name=name,dtype=dtype,input_dtype=input_dtype,dimensions=disk_dims,
                input_dimensions=dims,shape=array.shape,filename=str(raw),attrs=_json_attrs(attrs),connectivity_offset=offset))
        for name,raw in fields.items():
            if name in ('Time','xtime','initial_time') or (include_mesh and name in HISTORY_MESH_VARIABLES):
                raise ValueError(f'history field name {name!r} is reserved')
            array,dims,attrs=_field_parts(name,raw,field_dimensions,field_attrs)
            if dims is None:dims=_infer_field_dimensions(name,array,sizes,len(times))
            metadata=dict(_FIELD_METADATA.get(name,{}));metadata.update(attrs)
            handoff(name,array,tuple(dims),metadata)
        if include_mesh:
            converted=set(getattr(mesh,'converted_connectivity',())) | _CONNECTIVITY_VARIABLES
            for name in HISTORY_MESH_VARIABLES:
                if name not in arrays:continue
                array=np.asarray(arrays[name])
                dims=_mesh_dims_for_array(name,array,_mesh_variable_dimensions(mesh),sizes)
                attrs=dict(_FIELD_METADATA.get(name,{}));attrs.update(_mesh_variable_attrs(mesh).get(name,{}))
                handoff(name,array,dims,attrs,1 if name in converted else 0)
        variables.extend([
            dict(name='initial_time',dtype='S1',dimensions=('StrLen',),text=[_mpas_time_string(reference)],attrs=_FIELD_METADATA['initial_time']),
            dict(name='xtime',dtype='S1',dimensions=('Time','StrLen'),text=[_mpas_time_string(t) for t in times],attrs=_FIELD_METADATA['xtime'])])
        handoff('Time',np.asarray(seconds,dtype=np.float64),('Time',),dict(units=f'seconds since {reference:%Y-%m-%d %H:%M:%S}',long_name='CF-compliant valid time',standard_name='time',calendar='gregorian'))
        inherited=dict(getattr(mesh,'attrs',{}))
        inherited.update(Conventions='CF-1.10',source='mpas-port serial atmosphere history writer',
                         hexcore_evidence='implemented-unverified',hexcore_frozen_source=_FROZEN_OUTPUT_SOURCE,stream_name='output')
        if global_attrs:inherited.update(global_attrs)
        used={d for v in variables for d in v['dimensions']}
        used.update(d for d in (*_ENTITY_DIMENSIONS,*_VERTICAL_DIMENSIONS) if d in sizes)
        def emit(target):
            _call(dict(operation='write',path=str(target),format='netcdf' if options.io_type=='serial_netcdf' else options.io_type,
                       dimensions={d:sizes[d] for d in used},variables=variables,global_attrs=_json_attrs(inherited)),stage)
            return destination
        # Replacing a frame also refreshes its matching durability marker.
        atomic_write(emit,destination,replace=options.clobber_mode!='never_modify')
    return destination


def write_native_stream(*args: Any, **kwargs: Any) -> Path:
    """Exact native stream membership (Registry.xml:498-1349), via Rust."""
    from .restart_v842 import write_stream
    return write_stream(*args, **kwargs)


def write_mpas_history(*args: Any, **kwargs: Any) -> Path:
    """Compatibility spelling for :func:`write_history`."""

    return write_history(*args, **kwargs)


write_history_file = write_mpas_history


__all__ = [
    "Field",
    "HISTORY_MESH_VARIABLES",
    "HISTORY_OUTPUT_VARIABLES",
    "write_native_stream",
    "HistoryField",
    "HistoryStreamOptions",
    "format_mpas_filename",
    "write_history",
    "write_history_file",
    "write_mpas_history",
]
