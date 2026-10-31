"""Untimed CUDA API and Python host-read census, with call-site stacks.

CUPTI callbacks observe Cython-internal calls too, unlike monkeypatching CuPy.
Runtime and driver rows are separate layers and must never be summed together.
ABI: https://docs.nvidia.com/cupti/13.2.0/api/structCUpti__CallbackData.html
"""
from __future__ import annotations

from collections import Counter
import ctypes as C
import json
import linecache
import os
import re
from pathlib import Path
import sys
import threading
import time


class CallbackData(C.Structure):
    # Only the stable prefix is used; no dependence on trailing ABI fields.
    _fields_ = [('site', C.c_int), ('name', C.c_char_p), ('params', C.c_void_p)]


class RuntimeCopy(C.Structure):
    _fields_ = [('dst', C.c_void_p), ('src', C.c_void_p),
                ('size', C.c_size_t), ('kind', C.c_int)]


class DriverCopy(C.Structure):
    _fields_ = [('dst', C.c_void_p), ('src', C.c_void_p), ('size', C.c_size_t)]


class Audit:
    def __init__(self, library):
        self.library = str(library)
        self.lib = C.CDLL(self.library)
        self.rows = Counter()
        self.python = Counter()
        self.errors = []
        self.local = threading.local()
        self.subscriber = C.c_void_p()
        self.callback = C.CFUNCTYPE(None, C.c_void_p, C.c_int, C.c_uint32, C.c_void_p)(self._callback)
        self.lib.cuptiSubscribe.argtypes = [C.POINTER(C.c_void_p), type(self.callback), C.c_void_p]
        self.lib.cuptiEnableDomain.argtypes = [C.c_uint32, C.c_void_p, C.c_int]
        self.lib.cuptiUnsubscribe.argtypes = [C.c_void_p]

    @staticmethod
    def context(frame):
        step, op, stack = None, 'framework', []
        while frame:
            code, loc = frame.f_code, frame.f_locals
            path = code.co_filename
            if path.rsplit('/', 1)[-1] not in ('audit.py', 'launch.py') or 'rrfs_sync_audit' not in path:
                if '/hexcore/' in path:
                    path = 'woof/hex/' + path.split('/hexcore/', 1)[1]
                    if code.co_filename.endswith('/runtime_v842.py') and 'step' in loc:
                        step = loc['step']
                    if code.co_filename.endswith('/streams_output_v842.py') and 'step' in loc:
                        step, op = loc['step'], 'output-worker'
                    if code.co_name == '_call' and 'slot' in loc:
                        op = str(loc['slot'] or loc.get('operation', 'framework'))
                    elif code.co_name == '_step_device_v841' and op == 'framework':
                        op = 'dycore'
                    elif code.co_name == 'write_native_output' and op == 'framework':
                        op = 'output'
                stack.append(f'{path}:{frame.f_lineno}:{code.co_name}')
            frame = frame.f_back
        return step, op, tuple(stack)

    def _callback(self, userdata, domain, cbid, ptr):
        try:
            data = C.cast(ptr, C.POINTER(CallbackData)).contents
            if data.site != 0:  # CUPTI_API_ENTER
                return
            name = data.name.decode()
            category, size, blocking = None, 0, False
            if 'Synchronize' in name:
                category, blocking = 'synchronize', True
            elif 'Memset' in name or 'memset' in name:
                category = 'fill'
            elif 'Memcpy' in name or 'memcpy' in name:
                if domain == 2 and name.startswith(('cudaMemcpy_v', 'cudaMemcpyAsync')):
                    p = C.cast(data.params, C.POINTER(RuntimeCopy)).contents
                    category, size = ('D2H' if p.kind == 2 else 'copy-other'), p.size
                elif domain == 1 and name.startswith(('cuMemcpyDtoH', 'cuMemcpyHtoD', 'cuMemcpyDtoD')):
                    p = C.cast(data.params, C.POINTER(DriverCopy)).contents
                    category, size = ('D2H' if name.startswith('cuMemcpyDtoH') else 'copy-other'), p.size
                else:
                    category = 'copy-unclassified'
                blocking = category == 'D2H' and 'Async' not in name
            elif 'LaunchKernel' in name or name.startswith('cuLaunchCooperativeKernel'):
                category = 'launch'
            if category is None:
                return
            step, op, stack = self.context(sys._getframe(1))
            self.rows[(step, op, domain, category, name, blocking, size, stack)] += 1
            if domain == 1 and category == 'D2H':
                # Python's numeric slots (int/float/bool) are not c_call events.
                # The CUDA callback still carries their Python call-site frame.
                frame = sys._getframe(1)
                source = linecache.getline(frame.f_code.co_filename, frame.f_lineno)
                for conversion in re.findall(r'\b(int|float|bool)\s*\(', source):
                    self.python[(step, op, 'implicit.' + conversion, stack)] += 1
        except Exception as exc:
            self.errors.append(repr(exc))

    def profile(self, frame, event, arg):
        if event == 'c_call':
            owner = getattr(arg, '__self__', None)
            module = type(owner).__module__
            name = getattr(arg, '__name__', '')
            if module.startswith('cupy') and name in ('get', 'item', '__int__', '__float__', '__bool__', 'synchronize'):
                step, op, stack = self.context(frame)
                self.python[(step, op, module + '.' + name, stack)] += 1
        elif event == 'call':
            code = frame.f_code
            # Count Python builders/launch dispatch as well as asnumpy. All rows
            # have stacks so callers can separate initialization from steps.
            if (code.co_name == 'asnumpy' and '/cupy/' in code.co_filename or
                    code.co_name in ('_launch', 'from_mpas') and '/hexcore/' in code.co_filename):
                step, op, stack = self.context(frame)
                self.python[(step, op, code.co_name, stack)] += 1

    def start(self):
        if sys.getprofile() is not None:
            raise RuntimeError('audit requires an unused Python profile hook')
        for result in (self.lib.cuptiSubscribe(C.byref(self.subscriber), self.callback, None),):
            if result:
                raise RuntimeError(f'cuptiSubscribe failed: {result}')
        for domain in (1, 2):
            result = self.lib.cuptiEnableDomain(1, self.subscriber, domain)
            if result:
                raise RuntimeError(f'cuptiEnableDomain({domain}) failed: {result}')
        sys.setprofile(self.profile)
        threading.setprofile(self.profile)
        return self

    def stop(self, output):
        sys.setprofile(None)
        threading.setprofile(None)
        self.lib.cuptiUnsubscribe(self.subscriber)
        rows = []
        for (step, op, domain, category, name, blocking, size, stack), count in self.rows.items():
            rows.append(dict(step=step, op=op, layer='driver' if domain == 1 else 'runtime',
                             category=category, api=name, blocking=blocking,
                             bytes_each=size, count=count, stack=stack))
        python = [dict(step=s, op=o, api=n, stack=t, count=c)
                  for (s, o, n, t), c in self.python.items()]
        Path(output).write_text(json.dumps(dict(schema='rrfs-sync-audit-v1', untimed=True,
            library=self.library, errors=self.errors, cuda=rows, python=python), indent=1) + '\n')
        if self.errors:
            raise RuntimeError(f'audit callback errors: {self.errors[:3]}')
