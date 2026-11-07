"""Engine land-kernel modules held per card, for same-host ranks on several cards.

THE BREAKAGE THIS PREVENTS (measured 2026-10-15 on a two-card host, gpuwm 2.9.0): same-host ranks drive
several cards from one process, and gpuwm 2.9.0 holds its Noah-MP kernel modules once per process: the
driver, energy, thermal and glacier factories cache on their compile options alone, and the slab-libm
module is one global.  A CUDA module belongs to the card it was loaded on, so the second card was handed
the first card's kernel.  A two-card active local-time-stepping forecast stopped at step 4 with
``CUDA_ERROR_INVALID_HANDLE`` in ``noahmp_glacier_column``; the same forecast on one card passed.  The
engine already holds its VEGE_FLUX module per card (``noahmp_vegeflux_gpu._module_on``); these were missed.

Every one of those modules is built by ``woof.core.ieee_module.compile_module`` and is a
``_LoadedModule``.  This records the card each module is compiled on.  On any other card the module loads
its own compiled image (the same bytes) once, and every function resolves on the card that calls it, so a
function cached on one card and called on another still launches that card's copy.  On the card it was
compiled on a module is used exactly as before.  It is installed only when ranks sit on distinct cards;
a one-card run never loads it.

Retire this when the pinned woof keys those modules per card itself.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_INSTALLED = False

REFUSAL = ("same-host ranks on several cards need the engine's land-kernel modules held per card, and this "
           "woof's ieee_module does not have the shape woof.hex.engine_per_card keys ({missing}); the second "
           "card would launch the first card's Noah-MP kernels and stop with CUDA_ERROR_INVALID_HANDLE. "
           "Run on one card")


def installed() -> bool:
    return _INSTALLED


def bind_engine_modules_per_card(cp, ieee_module=None) -> bool:
    """Key the engine's land-kernel modules per card (idempotent).  True when this call installed it."""

    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return False
        if ieee_module is None:
            try:
                from woof.core import ieee_module
            except ImportError:
                # An engine without ieee_module builds none of the modules this keys.
                return False

        loaded_type = getattr(ieee_module, "_LoadedModule", None)
        function_type = getattr(ieee_module, "_Function", None)
        compile_module = getattr(ieee_module, "compile_module", None)
        missing = [name for name, value in (("_LoadedModule", loaded_type), ("_Function", function_type),
                                            ("compile_module", compile_module)) if value is None]
        if missing:
            raise RuntimeError(REFUSAL.format(missing=", ".join(missing)))
        load_lock = threading.Lock()

        def current_card() -> int:
            return int(cp.cuda.Device().id)

        def module_here(loaded):
            card = loaded.__dict__.get("_hex_card")
            device = current_card()
            if card is None or card == device:
                return loaded.module
            with load_lock:
                copies = loaded.__dict__.setdefault("_hex_copies", {})
                module = copies.get(device)
                if module is None:
                    image = loaded.__dict__.get("cubin")
                    if image is None:
                        image = loaded.__dict__.get("ptx")
                    if image is None:
                        raise RuntimeError(REFUSAL.format(missing="the loaded module's image"))
                    module = cp.cuda.function.Module()
                    module.load(image.encode() if isinstance(image, str) else image)
                    copies[device] = module
            return module

        class PerCardFunction(function_type):
            """A function that launches on the card that calls it."""

            def __init__(self, loaded, name):  # noqa: D107 - the base binds one function; this binds a name
                self._loaded = loaded
                self._name = name
                self._by_card = {}

            @property
            def function(self):
                device = current_card()
                function = self._by_card.get(device)
                if function is None:
                    function = module_here(self._loaded).get_function(self._name)
                    self._by_card[device] = function
                return function

        def get_function(self, name):
            return PerCardFunction(self, name)

        def get_global(self, name):
            module = module_here(self)
            pointer = module.get_global_var(name)
            memory = cp.cuda.UnownedMemory(pointer, 0, module)
            return cp.cuda.MemoryPointer(memory, 0)

        def compile_on_card(*args, _compile=compile_module, **kwargs):
            module = _compile(*args, **kwargs)
            module._hex_card = current_card()
            return module

        loaded_type.get_function = get_function
        loaded_type.get_global = get_global
        ieee_module.compile_module = compile_on_card
        _INSTALLED = True
        return True
