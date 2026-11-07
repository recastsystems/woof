"""Experimental whole-step submission boundary, not attached to a forecast.

Construct one immutable sequence per shape/cadence during setup. Kernel handles,
typed argument storage, modules and device allocations must remain valid until
the caller drains the stream. Owners pin their existing buffers here; this
executor neither stages nor allocates weather fields. Clock/counter arguments
come from a caller-owned ctypes Structure passed to submit once per step.

The native library is built explicitly before setup from native/step_submit.c.
There is no automatic build or eager fallback. The caller selects the current
CUDA context and passes a plain stream handle. Cross-stream dependencies, halo
exchange, allocation reuse, buffer rotation and physics cadence must be resolved
when binding the whole step. This module does not capture Python or CUDA graphs.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from pathlib import Path


class _Argument(C.Structure):
    _fields_ = [("value", C.c_void_p), ("offset", C.c_size_t),
                ("width", C.c_size_t), ("from_frame", C.c_uint32)]


class _Command(C.Structure):
    _fields_ = [("kind", C.c_uint32), ("grid", C.c_uint32 * 3),
                ("block", C.c_uint32 * 3), ("shared_bytes", C.c_uint32),
                ("argc", C.c_uint32), ("function", C.c_void_p),
                ("args", C.POINTER(_Argument)), ("destination", C.c_uint64),
                ("source", C.c_uint64), ("bytes", C.c_size_t), ("fill", C.c_ubyte)]


class _Driver(C.Structure):
    _fields_ = [("launch", C.c_void_p), ("copy", C.c_void_p), ("clear", C.c_void_p)]


@dataclass(frozen=True)
class FrameArgument:
    """Read this typed field from the current submission's input structure."""
    name: str


@dataclass(frozen=True)
class Kernel:
    function: int
    grid: tuple[int, ...]
    block: tuple[int, ...]
    arguments: tuple
    shared_bytes: int = 0


@dataclass(frozen=True)
class DeviceCopy:
    destination: int
    source: int
    bytes: int


@dataclass(frozen=True)
class DeviceClear:
    destination: int
    value: int
    bytes: int


class NativeSubmissionError(RuntimeError):
    def __init__(self, status, issued):
        self.status, self.issued = status, issued
        super().__init__(f"GPU driver status {status} at command {issued}; "
                         f"{issued} commands issued, step poisoned; no automatic retry")


def _unsigned(value, bits):
    if not isinstance(value, int) or not 0 <= value < (1 << bits):
        raise ValueError(f"expected unsigned {bits}-bit integer, got {value!r}")
    return value


def _dimensions(values):
    values = tuple(values)
    if not 1 <= len(values) <= 3 or any(value <= 0 for value in values):
        raise ValueError("kernel dimensions need one to three positive integers")
    return (C.c_uint32 * 3)(*(_unsigned(v, 32) for v in values),
                           *((1,) * (3 - len(values))))


class NativeStep:
    """Sealed ordinary CUDA commands. No Python runs between native launches.

    Driver errors poison this object because earlier commands may be queued.
    Success means enqueued, never synchronized or numerically validated.
    Concurrent submissions sharing mutable device buffers are caller errors.
    """
    def __init__(self, library: str | Path, commands, *, frame_type,
                 owners, cuda_library=None):
        if not issubclass(frame_type, C.Structure):
            raise TypeError("frame_type must be a ctypes Structure")
        # Resolve once, during setup. CDLL releases the GIL during submission.
        self._library = C.CDLL(str(library))
        self._library.hex_step_abi_version.argtypes = []
        self._library.hex_step_abi_version.restype = C.c_uint32
        if self._library.hex_step_abi_version() != 1:
            raise ValueError("unsupported native step ABI")
        prefix = ''
        if cuda_library is None:
            from woof.hip_backend import active
            if active():
                from woof.hip_rust import driver
                cuda_library, prefix = driver(), 'gpuwm_'
            else:
                cuda_library = 'libcuda.so.1'
        self._cuda = C.CDLL(str(cuda_library))
        self._driver = _Driver(*(C.cast(getattr(self._cuda, prefix + name), C.c_void_p).value
                                 for name in ("cuLaunchKernel", "cuMemcpyDtoDAsync_v2",
                                              "cuMemsetD8Async")))
        self._library.hex_step_validate.argtypes = [C.POINTER(_Command), C.c_size_t, C.c_size_t]
        self._library.hex_step_validate.restype = C.c_int
        self._submit = self._library.hex_step_submit
        self._submit.argtypes = [C.POINTER(_Driver), C.POINTER(_Command), C.c_size_t,
                                 C.c_void_p, C.c_void_p, C.POINTER(C.c_size_t)]
        self._submit.restype = C.c_int
        self._frame_type = frame_type
        self._owners = tuple(owners)
        self._storage = []
        fields = {field[0]: field for field in frame_type._fields_}
        encoded = []
        self._kernel_count = 0
        for command in commands:
            item = _Command()
            if isinstance(command, Kernel):
                item.kind = 1
                item.function = _unsigned(command.function, 64)
                item.grid, item.block = _dimensions(command.grid), _dimensions(command.block)
                item.shared_bytes = _unsigned(command.shared_bytes, 32)
                arguments = []
                for value in command.arguments:
                    if isinstance(value, FrameArgument):
                        field = fields[value.name]
                        if len(field) != 2:
                            raise ValueError("bitfields cannot be CUDA kernel arguments")
                        arguments.append(_Argument(None, getattr(frame_type, value.name).offset,
                                                   C.sizeof(field[1]), 1))
                    else:
                        # A private byte copy prevents later scalar mutation
                        # from silently changing a supposedly sealed plan.
                        size = C.sizeof(value)
                        saved = C.create_string_buffer(C.string_at(C.addressof(value), size), size)
                        self._storage.append(saved)
                        arguments.append(_Argument(C.addressof(saved), 0, size, 0))
                if len(arguments) > 256:
                    raise ValueError("native kernel supports at most 256 arguments")
                args = (_Argument * len(arguments))(*arguments)
                self._storage.append(args)
                item.argc, item.args = len(arguments), args
                self._kernel_count += 1
            elif isinstance(command, (DeviceCopy, DeviceClear)):
                item.kind = 2 if isinstance(command, DeviceCopy) else 3
                item.destination = _unsigned(command.destination, 64)
                item.bytes = _unsigned(command.bytes, C.sizeof(C.c_size_t) * 8)
                if isinstance(command, DeviceCopy):
                    item.source = _unsigned(command.source, 64)
                else:
                    item.fill = _unsigned(command.value, 8)
            else:
                raise TypeError(f"unsupported native command {type(command).__name__}")
            encoded.append(item)
        self._commands = (_Command * len(encoded))(*encoded)
        if self._library.hex_step_validate(self._commands, len(encoded), C.sizeof(frame_type)):
            raise ValueError("invalid native step descriptor; no commands submitted")
        self._command_count = len(encoded)
        self._poisoned = False

    @property
    def command_count(self):
        return self._command_count

    @property
    def kernel_count(self):
        return self._kernel_count

    def submit(self, frame, *, stream: int):
        """Issue the sealed sequence with ONE native call, without a GPU wait."""
        if self._poisoned:
            raise RuntimeError("native step is poisoned after a partial submission")
        if type(frame) is not self._frame_type:
            raise TypeError("submission frame differs from the sealed frame type")
        _unsigned(stream, 64)
        issued = C.c_size_t()
        status = self._submit(C.byref(self._driver), self._commands, self.command_count,
                              C.byref(frame), stream, C.byref(issued))
        if status:
            self._poisoned = True
            raise NativeSubmissionError(status, issued.value)
