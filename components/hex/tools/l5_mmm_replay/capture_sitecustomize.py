"""L5 capture hook (test tool, never shipped): saves the inputs of hex's YSU wrapper and MM5 column calls.

Active only when L5_CAPTURE_DIR is set. Inputs are copied to the host BEFORE the call, so the forecast itself is
unchanged (the hook only reads). L5_CAPTURE_EVERY=K keeps every K-th call, up to L5_CAPTURE_MAX per scheme.
"""
import os

_DIR = os.environ.get("L5_CAPTURE_DIR")
if _DIR:
    import json
    import numpy as np

    _EVERY = int(os.environ.get("L5_CAPTURE_EVERY", "1"))
    _MAX = int(os.environ.get("L5_CAPTURE_MAX", "4"))
    _count = {"ysu": 0, "mm5": 0}
    _kept = {"ysu": 0, "mm5": 0}
    _held = {}

    def _host(value):
        import cupy as cp
        if isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
        if isinstance(value, np.ndarray):
            return value.copy()
        return None

    def _due(kind):
        _count[kind] += 1
        n = _count[kind]
        return (n == 1 or n % _EVERY == 0) and _kept[kind] < _MAX

    def _install():
        from woof.hex.mpasphys import ysu_v842 as ysu, sfclay_mm5_v842 as mm5
        if getattr(ysu, "_l5_hooked", False):
            return
        original_ysu = ysu.ysu
        original_launch = mm5.launch_columns
        original_surface = mm5.run_surface_layer

        def ysu_hook(arrays, constants, **kwargs):
            if _due("ysu"):
                tag = f"ysu-{os.getpid()}-{_count['ysu']:06d}"
                saved = {k: _host(v) for k, v in arrays.items() if _host(v) is not None}
                np.savez(os.path.join(_DIR, tag + ".npz"), **saved)
                meta = {k: (v if isinstance(v, (int, float, bool, str)) else None) for k, v in kwargs.items()
                        if k != "device_math_header"}
                with open(os.path.join(_DIR, tag + ".json"), "w") as f:
                    json.dump({"call": _count["ysu"], "constants": {k: float(v) for k, v in constants.items()},
                               "kwargs": meta}, f)
                _kept["ysu"] += 1
            return original_ysu(arrays, constants, **kwargs)

        def surface_hook(atmosphere, diag_physics, sfc_input, *args, **kwargs):
            _held["var2d"] = sfc_input.get("var2d") if hasattr(sfc_input, "get") else None
            return original_surface(atmosphere, diag_physics, sfc_input, *args, **kwargs)

        def launch_hook(packed, constants, **kwargs):
            if _due("mm5"):
                tag = f"mm5-{os.getpid()}-{_count['mm5']:06d}"
                saved = dict(packed=_host(packed), constants=_host(constants))
                var2d = _held.get("var2d")
                if var2d is not None:
                    saved["var2d"] = _host(var2d)
                np.savez(os.path.join(_DIR, tag + ".npz"), **saved)
                meta = {k: (v if isinstance(v, (int, float, bool, str)) else repr(type(v))) for k, v in kwargs.items()}
                with open(os.path.join(_DIR, tag + ".json"), "w") as f:
                    json.dump({"call": _count["mm5"], "kwargs": meta}, f)
                _kept["mm5"] += 1
            return original_launch(packed, constants, **kwargs)

        ysu.ysu = ysu_hook
        mm5.launch_columns = launch_hook
        mm5.run_surface_layer = surface_hook
        ysu._l5_hooked = True

    try:
        os.makedirs(_DIR, exist_ok=True)
        _install()
    except Exception as error:  # a process that never runs the physics (fetch, mesh) has nothing to hook
        with open(os.path.join(_DIR, f"hook-skip-{os.getpid()}.txt"), "w") as f:
            f.write(repr(error))
