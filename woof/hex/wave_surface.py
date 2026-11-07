"""Optional wave-roughness arguments for the existing surface-layer kernels.

The wave solver owns the roughness law and high-wind treatment. These variants
replace the water momentum roughness before the surface layer uses it. The
original source and launch arguments remain the feedback-off path.
"""
from __future__ import annotations

import re


def feedback(pools):
    binding = getattr(getattr(pools, "session", None), "wave_binding", None)
    return None if binding is None else binding.roughness()


def validate(array, n):
    import cupy as cp
    if (not isinstance(array, cp.ndarray) or array.dtype != cp.float32 or
            array.shape != (n,) or not array.flags.c_contiguous):
        raise ValueError("wave roughness must be contiguous device float32 [nCells]")


def _scalar(source, function, expression):
    lines = source.splitlines(keepends=True)
    declarations = calls = 0
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if function + "(" not in line or stripped.startswith("//"):
            continue
        if stripped.startswith("__device__"):
            value = "float ocean_z0"
            declarations += 1
        elif stripped.startswith(function + "("):
            value = expression
            calls += 1
        else:
            continue
        end = line.rfind(")")
        lines[i] = line[:end] + ", " + value + line[end:]
    if declarations != 2 or calls != 1:
        raise ValueError(f"wave variant lost {function} signature/call anchors: {declarations}/{calls}")
    return "".join(lines)


def _kernel(source, name):
    pattern = r'(extern "C" __global__ void ' + re.escape(name) + r'\([^)]*)(\)\s*\{)'
    source, count = re.subn(pattern, r'\1, const float *ocean_z0=nullptr\2', source)
    if count != 1:
        raise ValueError(f"wave variant lost {name} kernel signature")
    return source


def _insert_water(source, name, marker):
    pattern = r'__device__[^\n]*\b' + re.escape(name) + r'\([^\n]*\)\s*\{'
    match = re.search(pattern, source)
    if match is None:
        raise ValueError("wave variant lost water function")
    next_function = source.find("\n__device__", match.end())
    end = len(source) if next_function < 0 else next_function
    offset = source.find(marker, match.end(), end)
    if offset < 0:
        raise ValueError("wave variant lost water roughness use")
    return (source[:offset] + "// Wave solver z0, before thermal roughness and flux iteration.\n"
            "if (ocean_z0>0.f) znt=ocean_z0;\n" + source[offset:])


def mpas_mynn_source(source):
    source = _scalar(source, "m_sf_mynn_run", "ocean_z0[i]")
    source = _kernel(source, "sf_mynn_columns")
    return _insert_water(source, "m_sf_mynn_run", "if (spp_pbl) {\nzntstoch =")


def rrfs_mynn_source(source):
    source = _scalar(source, "m_drv_mynnsfc_driver", "ocean_z0[c]")
    source = _scalar(source, "m_water_mynnsfc_water", "ocean_z0")
    source = _kernel(source, "mynnsfc_columns")
    return _insert_water(source, "m_water_mynnsfc_water", "if ((spp_sfc==1)) {\nzntstoch =")
