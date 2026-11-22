#!/usr/bin/env python3
"""Observe array ownership candidates without traversing held stochastic code.

Only built-in containers and instance dictionaries of hex/gpuwm objects are
traversed. This deliberately bounded walk neither reads field data nor proves
the carried set. Unknown owners, hidden references, and truncated subtrees are
reported so that a producer cannot mistake this for a complete census.
"""
# SPDX-License-Identifier: Apache-2.0
from collections import Counter, deque
import hashlib
import json
import struct
import time
import types


def _key(key):
    if isinstance(key, (str, int)) and not isinstance(key, bool):
        return json.dumps(key, ensure_ascii=True)
    if isinstance(key, tuple) and all(isinstance(item, (str, int)) and not isinstance(item, bool) for item in key):
        return json.dumps(key, ensure_ascii=True, separators=(',', ':'))
    return None


def discover(roots, cp, metadata, *, max_nodes=30000, max_depth=12, include_scalar_hashes=False,
             include_slots=False, include_numpy_scalars=False, include_slices=False):
    import numpy as np
    if max_nodes < 1 or max_depth < 0:
        raise ValueError('invalid object graph limit')
    started = time.monotonic_ns()
    queue = deque((name, value, 0) for name, value in sorted(roots.items()))
    seen, arrays, aliases, scalars = {}, [], [], []
    pruned = Counter()
    unexpanded_types = Counter()
    traversal = Counter()
    visited = 0
    primitive_leaves = 0

    def enqueue(path, value, depth):
        # Large scalar tables used to fill the bounded queue before it reached
        # unrelated array owners. Scalars have no children, so skip them on
        # insertion; separate frame rules still have to census their values.
        nonlocal primitive_leaves
        if any(word in path.lower() for word in ('password', 'credential', 'secret', 'token')):
            pruned['sensitive_name'] += 1
            return
        # `isinstance` can consult an object's overridden __class__, so use
        # exact built-in types before the later held-module exclusion.
        if value is None or type(value) in (bool, int, float, complex, str, bytes, bytearray):
            primitive_leaves += 1
            if include_scalar_hashes and (value is None or type(value) in (bool, int, float, complex)):
                if value is None:
                    encoded = b'none'
                elif type(value) is bool:
                    encoded = b'bool:' + (b'1' if value else b'0')
                elif type(value) is int:
                    encoded = b'int:' + str(value).encode('ascii')
                elif type(value) is float:
                    encoded = b'float:' + struct.pack('<d', value)
                else:
                    encoded = b'complex:' + struct.pack('<dd', value.real, value.imag)
                scalars.append({'path': path, 'type': type(value).__name__,
                                'sha256': hashlib.sha256(encoded).hexdigest()})
            return
        # Only numeric NumPy scalars, and only after the sensitive-name check.
        # A held object is never interrogated through isinstance or its bytes.
        if (include_numpy_scalars and type(value).__module__.startswith('numpy')
                and isinstance(value, np.generic) and value.dtype.kind in 'biufc'):
            primitive_leaves += 1
            if include_scalar_hashes:
                encoded = b'numpy:' + value.dtype.str.encode('ascii') + b':' + value.tobytes()
                scalars.append({'path': path, 'type': 'numpy.' + type(value).__name__,
                                'sha256': hashlib.sha256(encoded).hexdigest()})
            return
        queue.append((path, value, depth))
    while queue:
        if visited >= max_nodes:
            pruned['node_limit'] += len(queue)
            break
        path, value, depth = queue.popleft()
        visited += 1
        if value is None:
            continue
        module = type(value).__module__
        if module.startswith(('woof.hex.stochastic', 'gpuwm.stochastic')):
            pruned['held_stochastic'] += 1
            continue
        if isinstance(value, (bool, int, float, complex, str, bytes, bytearray)):
            continue
        ident = id(value)
        if ident in seen:
            aliases.append({'path': path, 'first': seen[ident]})
            continue
        seen[ident] = path
        if isinstance(value, cp.ndarray):
            arrays.append({'path': path, 'memory': 'device', **metadata(value)})
            continue
        if isinstance(value, np.ndarray):
            arrays.append({'path': path, 'memory': 'host', 'address': int(value.__array_interface__['data'][0]),
                           'bytes': int(value.nbytes), 'dtype': value.dtype.str, 'shape': list(value.shape),
                           'strides': list(value.strides)})
            continue
        if depth >= max_depth:
            pruned['depth_limit'] += 1
            continue
        if include_slices and type(value) is slice:
            for name in ('start', 'stop', 'step'):
                enqueue(path + '.' + name, getattr(value, name), depth + 1)
        elif isinstance(value, dict):
            children = []
            # Bypass custom container methods: a framework container can lazily
            # materialize entries when its ordinary iterator is called.
            for key, child in dict.items(value):
                token = _key(key)
                if token is None:
                    pruned['unsupported_key'] += 1
                elif isinstance(key, str) and 'stochastic' in key.lower():
                    pruned['held_stochastic'] += 1
                else:
                    children.append((token, child))
            for token, child in sorted(children, key=lambda pair: pair[0]):
                enqueue(path + '[' + token + ']', child, depth + 1)
        elif isinstance(value, (list, tuple)):
            iterator = list.__iter__(value) if isinstance(value, list) else tuple.__iter__(value)
            for index, child in enumerate(iterator):
                enqueue(path + '[' + str(index) + ']', child, depth + 1)
        elif isinstance(value, (set, frozenset)):
            pruned['unordered_collection'] += 1
        elif isinstance(value, types.SimpleNamespace) or module.startswith(('woof.hex.', 'woof.')):
            try:
                attrs = object.__getattribute__(value, '__dict__')
            except AttributeError:
                attrs = None
            if attrs is not None and not isinstance(attrs, dict):
                pruned['non_dict_attributes'] += 1
                continue
            children = {} if attrs is None else dict(dict.items(attrs))
            if include_slots:
                # Member descriptors access actual Python slot storage. Do not
                # call properties, custom __getattribute__, or foreign owners.
                for cls in type.__getattribute__(type(value), '__mro__'):
                    class_dict = type.__getattribute__(cls, '__dict__')
                    for name, descriptor in class_dict.items():
                        if name in children or type(descriptor) is not types.MemberDescriptorType:
                            continue
                        if ('stochastic' in name.lower() or any(word in name.lower() for word in
                                ('password', 'credential', 'secret', 'token'))):
                            pruned['held_stochastic' if 'stochastic' in name.lower() else 'sensitive_name'] += 1
                            continue
                        try:
                            children[name] = descriptor.__get__(value, type(value))
                            traversal['slot_values'] += 1
                        except AttributeError:
                            pass
            if attrs is None:
                if children:
                    traversal['slotted_objects_without_dict'] += 1
                else:
                    pruned['no_instance_dict'] += 1
            for name, child in sorted(children.items()):
                if 'stochastic' in name.lower():
                    pruned['held_stochastic'] += 1
                else:
                    enqueue(path + '.' + name, child, depth + 1)
        else:
            pruned['outside_modules_or_callable'] += 1
            if include_scalar_hashes:
                unexpanded_types[module + '.' + type(value).__qualname__] += 1
    result = {'schema': 'ad.binder.phys-object-graph/v1', 'complete': False,
            'clock_ns': started, 'end_clock_ns': time.monotonic_ns(), 'visited': visited,
            'primitive_leaves_skipped': primitive_leaves,
            'arrays': arrays, 'aliases': aliases, 'pruned': dict(sorted(pruned.items())),
            'limitations': ['Array owners and alias paths are candidates only.',
                            'Native, slotted, hidden, unsupported and pruned objects are not covered.',
                            'Array data are not retained or classified as carried.',
                            'The held stochastic implementation is never traversed.']}
    if include_slots or include_numpy_scalars or include_slices:
        result['traversal'] = dict(sorted(traversal.items()))
    if include_scalar_hashes:
        result['scalar_hashes'] = scalars
        result['unexpanded_types'] = dict(sorted(unexpanded_types.items()))
        result['limitations'].append('Scalar hashes are checks, not replay inputs; strings and bytes are not read.')
    if include_slots:
        result['limitations'].append('Only Python member slots are visited; native hidden state is not covered.')
    return result
