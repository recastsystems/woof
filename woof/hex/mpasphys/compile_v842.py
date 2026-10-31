"""Strict compiler for MPAS physics carriers; mpas_atmphys_interface.F:284-425.

This cache belongs to physics only. Dynamics and the default WOOF suite retain
their existing compiler and cache. Direct NVRTC preserves native subnormals.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics

import inspect
import os

from ..cuda_backend.runtime import KernelCache
from ..cuda_backend.compile_contract import validate_module_key


class MpasPhysicsKernelCache(KernelCache):
    """Preserve native RKIND carrier words; mpas_atmphys_interface.F:284-425."""

    def raw_kernel(self, name, source, *, module_key, options=()):
        """Compile carrier/conversion kernels; mpas_atmphys_todynamics.F:98-498."""
        # Same arguments, same numerics mode: the kernel the first resolution produced. Breakage it
        # prevents: U-MPASPHYS-1, each per-step launch re-read three maths headers, rewrote and twice
        # hashed its whole translation unit on the forecast's single host thread before launching.
        memo_key = (name, module_key, source, tuple(options), _hex_numerics.is_strict())
        hit = self._resolved.get(memo_key)
        if hit is not None:
            return hit[1] if hit[0] is None else self._observe(hit[0], hit[1], name, hit[2])
        kernel = self._raw_kernel_v842(name, source, module_key=module_key, options=options)
        self._resolved[memo_key] = self._last_resolution
        return kernel

    def _raw_kernel_v842(self, name, source, *, module_key, options=()):
        if module_key == 'hexcore.cuda_physics_prep_v841':
            from .math_v842 import bind_common_carrier_math
            source = bind_common_carrier_math(source)
        # The additive shared API is an integrator reconciliation. Until it
        # arrives, the same direct compiler is owned entirely by this seam.
        if _hex_numerics.is_strict() and 'mpas_strict' in inspect.signature(KernelCache.raw_kernel).parameters:
            parent_raw_kernel = super().raw_kernel
            kernel = parent_raw_kernel(name, source, module_key=module_key,
                                       options=options, mpas_strict=True)
            self._last_resolution = (None, kernel, module_key)  # already observed by the parent
            return kernel
        from cupy.cuda import function
        from .strict_cache_v842 import compile_image
        module_key = validate_module_key(module_key)
        extra = (*options, '--fmad=false', '--ftz=false',
                 '--prec-div=true', '--prec-sqrt=true')
        requested = _hex_numerics.options((*self.base_options, *extra))
        compiled_key = self._module_key(source, requested)
        kernel_key = f'{compiled_key}:{name}'
        manifest = self._compile_manifest
        manifest.declare_module(module_key, source=source,
                                requested_options=requested,
                                module_cache_key=compiled_key)
        kernel = self._kernels.get(kernel_key)
        if kernel is None:
            module = self._modules.get(compiled_key)
            if module is None:
                bank = shared = key = None
                if (os.environ.get('WOOF_HEX_MOVING_RESIDENT') == '1'
                        and os.environ.get('WOOF_HEX_MOVING_KERNEL_MODULES', '1') != '0'):
                    from .. import moving_kernel_modules as bank
                    key = ('mpas-direct', self.capability.device_id, compiled_key, module_key,
                           getattr(self._aot_pack, 'resident_identity', None))
                    shared = bank.get(key)
                if shared is None:
                    image, evidence = compile_image(source, options=requested,
                                                    cache_dir=self.cache_directory)
                    module = function.Module()
                    module.load(image)
                else:
                    module, evidence, _ = shared
                manifest.record_effective_compile(module_key, evidence)
                if bank is not None and shared is None:
                    bank.put(key, module, evidence, 'mpas-direct')
                self._modules[compiled_key] = module
            kernel = module.get_function(name)
            self._kernels[kernel_key] = kernel
        manifest.record_kernel(module_key, name)
        self._last_resolution = (kernel_key, kernel, module_key)
        return self._observe(kernel_key, kernel, name, module_key)
