"""Authoritative Noah-MP runtime translation units and compilation identities.

Five source files are fragments rather than standalone translation units:
``noahmp_driver``, ``noahmp_energy``, ``noahmp_thermal``, ``noahmp_glacier``
and ``noahmp_libm_slab``. They borrow the r_pow/r_exp/r_log helpers from
``noahmp_leaves``; the slab also borrows helpers from ``noahmp_energy``.
Some standalone units carry independent WOOF float32 math routines. In particular,
prepending leaves to fluxprep would introduce duplicate definitions. Nothing
here changes CUDA math or consolidates those routines.

Runtime factories and memory measurement share :func:`runtime_unit`, which
binds ordered parts, the common preamble, exact compiler options and all global
exports (including inherited helpers). VEGE_FLUX deliberately uses C++14 and no
common preamble; the other production units use C++17 with the preamble.

:func:`translation_unit_source` remains the generic compilation-check helper
for existing tools, not a memory-measurement identity. Its optional preamble
switch must not erase the distinct runtime VEGE_FLUX contract.

The CPU provenance tests check factory inputs and source drift. The existing
GPU compile tests check that the five composed fragments fail alone.  The
per-thread local frames these units compile to are recorded per compile
platform in :mod:`woof.core.kernel_frame_recordings`
(``NOAHMP_COMPOSED_FRAME_RECORDINGS``) and priced by
:mod:`woof.core.noahmp_frame_provenance`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"

#: WOOF's shared float32 helpers for composed units. Its ``r_pow``,
#: ``r_exp`` and ``r_log`` are what the composed units below are missing.
LIBM_UNIT = "noahmp_leaves"

#: Translation unit name -> the ordered ``.cu`` stems it is compiled from.
#: The key is what a caller asks for; the value is the concatenation order.
NOAHMP_TRANSLATION_UNITS: dict[str, tuple[str, ...]] = {
    "noahmp_bareflux": ("noahmp_bareflux",),
    "noahmp_driver": (LIBM_UNIT, "noahmp_driver"),
    "noahmp_energy": (LIBM_UNIT, "noahmp_energy"),
    "noahmp_fluxprep": ("noahmp_fluxprep",),
    # NOAHMP_GLACIER borrows r_pow/r_exp/r_log (and the AD/SU/MU/DV
    # macros plus NMP_* constants) from the libm unit and carries only its
    # own float32 atan routine, like noahmp_bareflux does.
    "noahmp_glacier": (LIBM_UNIT, "noahmp_glacier"),
    "noahmp_leaves": ("noahmp_leaves",),
    # The slab composition's elementwise transcendentals borrow r_pow/r_exp/
    # r_log from LIBM_UNIT and nmpe_tanhf from noahmp_energy, so it is the one
    # unit built from three parts.
    "noahmp_libm_slab": (LIBM_UNIT, "noahmp_energy", "noahmp_libm_slab"),
    "noahmp_radiation": ("noahmp_radiation",),
    "noahmp_sflx": ("noahmp_sflx",),
    "noahmp_snow": ("noahmp_snow",),
    "noahmp_soilwater": ("noahmp_soilwater",),
    "noahmp_thermal": (LIBM_UNIT, "noahmp_thermal"),
    "noahmp_vegeflux": ("noahmp_vegeflux",),
    "noahmp_vegprecip": ("noahmp_vegprecip",),
    "noahmp_water": ("noahmp_water",),
    "noahmp_output": ("noahmp_output",),
}

#: The units that are *fragments*: they need :data:`LIBM_UNIT` in front and
#: fail NVRTC without it.  Named separately because "this file cannot be
#: compiled alone" is the fact a sweep trips over, and a test asserts each of
#: these really does still fail alone rather than having quietly become
#: self-contained.
COMPOSED_UNITS = tuple(
    name for name, parts in NOAHMP_TRANSLATION_UNITS.items() if len(parts) > 1)


def translation_unit_source(name: str, *, preamble: bool = True) -> str:
    """The exact source text NVRTC should be handed for ``name``.

    ``preamble`` prepends :func:`woof.core.kernels._preamble`, which is what
    ``load_module`` does; pass ``False`` only when composing this into a larger
    unit that already carries it.
    """
    try:
        parts = NOAHMP_TRANSLATION_UNITS[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a Noah-MP translation unit; the units are "
            f"{sorted(NOAHMP_TRANSLATION_UNITS)}") from None
    prefix = ""
    if preamble:
        from woof.core.kernels import _preamble
        prefix = _preamble()
    return _assemble(_component_texts(parts), prefix)


def kernel_files() -> tuple[str, ...]:
    """Every ``noahmp*.cu`` stem present on disk, sorted."""
    return tuple(sorted(path.stem for path in KERNEL_DIR.glob("noahmp*.cu")))


__all__ = [
    "COMPOSED_UNITS",
    "KERNEL_DIR",
    "LIBM_UNIT",
    "NOAHMP_TRANSLATION_UNITS",
    "kernel_files",
    "translation_unit_source",
]


# Runtime identity is deliberately separate from a standalone .cu stem.  In
# particular, load_module("noahmp_driver") must STILL fail: that is a fragment,
# not the unit the driver factory compiles.  The old standalone census and its
# measurements retain their meanings.
DEFAULT_OPTIONS = ("-std=c++17", "--ftz=false")


def pricing_key(name: str) -> str:
    """The memory-inventory key for the runtime unit (never a raw fragment)."""
    if name == "noahmp_gecros":
        return name
    parts = NOAHMP_TRANSLATION_UNITS[name]
    if len(parts) > 1:
        return name + "_composed"
    if name == "noahmp_vegeflux":
        # Its runtime uses C++14 WITHOUT the common preamble.  A generic
        # load_module(name) C++17 measurement does not describe that image.
        return name + "_runtime"
    return name


NOAHMP_PRICING_MODULES = tuple(pricing_key(name)
                              for name in NOAHMP_TRANSLATION_UNITS)


@dataclass(frozen=True)
class RuntimeUnit:
    """A snapshot of the *actual* RawModule inputs, constructed without CuPy.

    The source SHA hashes the exact UTF-8 string passed to RawModule, including
    the common header/defines when present.  The identity additionally binds
    ordered components, compiler options and every exported kernel, including
    helper exports inherited by a composed unit.  No byte measurements live
    here: ``identity()["identity_sha256"]`` is what a frame recording row
    binds itself to, so a row read from a different source or option tuple
    stops matching and the platform is refused until re-read.
    """

    name: str
    source: str
    parts: tuple[tuple[str, str], ...]
    preamble_sha256: str
    options: tuple[str, ...]
    exports: tuple[str, ...]

    @property
    def key(self) -> str:
        return pricing_key(self.name)

    def identity(self) -> dict:
        payload = {
            "name": self.name,
            "pricing_key": self.key,
            "parts": [{"file": name if name.endswith(".cuh") else name + ".cu",
                       "sha256": digest}
                      for name, digest in self.parts],
            "preamble_sha256": self.preamble_sha256,
            "source_sha256": _sha(self.source),
            "backend": "nvrtc",
            "options": list(self.options),
            "name_expressions": None,
            "exports": list(self.exports),
        }
        payload["identity_sha256"] = _sha(json.dumps(
            payload, sort_keys=True, separators=(",", ":")))
        return payload


_COMMENT = re.compile(r'/\*.*?\*/|//[^\n]*', re.S)
_EXPORT = re.compile(
    r'extern\s+"C"\s+__global__\s+void\s+([A-Za-z_]\w*)\s*\(')


def exported_kernels(source: str) -> tuple[str, ...]:
    """Enumerate all exports; refuse unfamiliar declarations instead of omitting.

    The shipped Noah-MP sources use explicit extern-C global functions.  This
    is a checked source contract, not a general CUDA parser: a new macro,
    template or linkage-block spelling must extend this parser and its tests.
    """
    text = _COMMENT.sub("", source)
    names = _EXPORT.findall(text)
    if (not names or len(names) != len(set(names))
            or len(names) != len(re.findall(r"\b__global__\b", text))):
        raise ValueError("Noah-MP exports are empty, duplicated or use an "
                         "unrecognised __global__ declaration")
    return tuple(sorted(names))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _component_texts(parts, kernel_dir=KERNEL_DIR):
    return tuple((part, (Path(kernel_dir) / f"{part}.cu").read_text(encoding="ascii"))
                 for part in parts)


def _assemble(texts, prefix):
    return prefix + "".join(text for _, text in texts)


def validate_vege_atan_profile(profile: str) -> str:
    """Validate capture arithmetic metadata without importing CUDA."""
    if profile not in ("glibc_2_39", "glibc_2_43"):
        raise ValueError("unknown Noah-MP atan profile")
    return profile


def runtime_unit(name: str, *, kernel_dir: Path = KERNEL_DIR,
                 atan_profile: str = "glibc_2_39",
                 log10_profile: str = "glibc_2_39",
                 selected_options: bool = False) -> RuntimeUnit:
    """Snapshot the production composition and options, including VEGE_FLUX.

    All custom runtime factories and the compile-only measurement harness use
    this function.  Ordinary standalone units have byte-identical inputs to
    kernels.load_module; the CPU factory-interception tests enforce that too.
    ``kernel_dir`` composes the unit from another tree's kernel files (the
    A146 census gates the tree it scans, A193).

    ``atan_profile`` selects VEGE_FLUX's native libm generation explicitly;
    it never follows the execution host. The default retains existing 2.39
    oracle arithmetic. The opt-in 2.43 profile uses the retained CORE-MATH
    header and records it as a source component. No other libm is switched.
    """
    if name == "noahmp_gecros":
        base = runtime_unit("noahmp_vegeflux", kernel_dir=kernel_dir,
                            atan_profile=atan_profile, selected_options=True)
        headers = ([] if atan_profile == "glibc_2_43" else ["glibc_trig_flt32.cuh"]) + [
            "noahmp_gecros.cuh", "noahmp_gecros_lifecycle.cuh", "noahmp_gecros_flux.cuh"]
        extra = tuple((h, (Path(kernel_dir)/h).read_text(encoding="ascii")) for h in headers)
        extra=tuple((h,("#define glibc_atanf nmp_gecros_atanf\n"+t+"\n#undef glibc_atanf\n") if h=="glibc_trig_flt32.cuh" else t) for h,t in extra)
        source = base.source + "\n" + _assemble(extra, "")
        return RuntimeUnit(name=name, source=source,
            parts=base.parts+tuple((h,_sha(t)) for h,t in extra),
            preamble_sha256=base.preamble_sha256,
            options=("-std=c++17","--ftz=false"), exports=exported_kernels(source))
    validate_vege_atan_profile(atan_profile)
    if atan_profile != "glibc_2_39" and name != "noahmp_vegeflux":
        raise ValueError("the selected atan profile is implemented only for VEGE_FLUX")
    parts = NOAHMP_TRANSLATION_UNITS[name]
    # Read each component ONCE so its recorded digest and compiled text cannot
    # describe different reads of a concurrently edited file.
    texts = _component_texts(parts, kernel_dir)
    if log10_profile not in ("glibc_2_39","glibc_2_43"):
        raise ValueError("unknown Noah-MP native LOG10 profile")
    if log10_profile != "glibc_2_39":
        if name != "noahmp_thermal":
            raise ValueError("native LOG10 profile is implemented only for THERMOPROP")
        header="noahmp_log10_profile.cuh"
        texts=((header,(Path(kernel_dir)/header).read_text(encoding="ascii")),)+tuple(
            (part,text.replace("r_log10(satratio)","noahmp_native_log10(satratio)"))
            for part,text in texts)

    if selected_options:
        if name not in {"noahmp_vegprecip", "noahmp_thermal", "noahmp_leaves", "noahmp_water", "noahmp_bareflux", "noahmp_glacier", "noahmp_radiation",
                        "noahmp_vegeflux"}:
            raise ValueError(f"no selected option unit for {name}")
        # Default images retain their source, exports and measured identity.
        # Extensions reuse its helpers and add separately sealed entry points.
        if name == "noahmp_radiation":
            trig = "glibc_trig_flt32.cuh"
            texts = texts + ((trig, (Path(kernel_dir) / trig).read_text(encoding="ascii")),)
        if name=="noahmp_leaves":
            header="noahmp_carbon_options.cuh"
            texts=texts+((header,(Path(kernel_dir)/header).read_text(encoding="ascii")),)
            header="noahmp_crop_options.cuh"
            texts=texts+((header,(Path(kernel_dir)/header).read_text(encoding="ascii")),)
            header="noahmp_soil_options.cuh"
            texts=texts+((header,(Path(kernel_dir)/header).read_text(encoding="ascii")),)
        header = name + "_options.cuh"
        texts = texts + ((header, (Path(kernel_dir) / header).read_text(encoding="ascii")),)
        if name=="noahmp_water":
            header="noahmp_groundwater_options.cuh"
            texts=texts+((header,(Path(kernel_dir)/header).read_text(encoding="ascii")),)
    if name == "noahmp_vegeflux":
        prefix = ""
        options = ("-std=c++14", "--ftz=false")
        if atan_profile == "glibc_2_43":
            # The constructed parent referee uses glibc 2.43. Preserve the
            # glibc 2.39 default used by the existing WRF oracle fixtures.
            header = (Path(kernel_dir) / "glibc_trig_flt32.cuh").read_text(
                encoding="ascii")
            prefix = ("#define glibc_atanf noahmp_glibc243_atanf\n" + header
                      + "\n#undef glibc_atanf\n"
                      + "#define GPUWM_NOAHMP_ATAN_GLIBC243 1\n")
            texts = (("glibc_trig_flt32.cuh", header),) + texts
            options = ("-std=c++17", "--ftz=false")
    else:
        from woof.core.kernels import _preamble
        prefix = _preamble(kernel_dir)
        options = DEFAULT_OPTIONS
    source = _assemble(texts[1:] if atan_profile == "glibc_2_43" else texts,
                       prefix)
    return RuntimeUnit(
        name=name, source=source,
        parts=tuple((part, _sha(text)) for part, text in texts),
        preamble_sha256=_sha(prefix), options=options,
        exports=exported_kernels(source))


def compile_runtime_unit(name: str, *, module_key: str,
                         options: tuple[str, ...] | None = None,
                         source: str | None = None, log_stream=None,
                         atan_profile: str = "glibc_2_39",
                         log10_profile: str = "glibc_2_39",
                         selected_options: bool = False):
    """Compile and load one Noah-MP runtime unit; no launch, no constant upload.

    The one direct-NVRTC route for Noah-MP. Every runtime factory
    (driver, energy, thermal, glacier, the libm slab, the generic loader's
    standalone units) and the frame measurement in
    :func:`woof.core.noahmp_frame_provenance.measure_live` compile through
    here, so the source string and option tuple a forecast hands NVRTC are
    the ones the recorded frames were read from -- by construction rather
    than by a second assembler agreeing with the first.

    ``source`` / ``options`` exist for the negative controls the parity
    suites compile (a perturbed copy, ``-fmad=false``); they are recorded
    under the caller's own ``module_key`` and are not the production unit.
    """
    unit = runtime_unit(name, atan_profile=atan_profile, log10_profile=log10_profile,
                        selected_options=selected_options)
    code = unit.source if source is None else source
    opts = unit.options if options is None else tuple(options)
    return _compile_loaded_module(code, opts, module_key, log_stream)


def _compile_loaded_module(code, opts, module_key, log_stream=None):
    """Keep generated transfers and scientific units on the same compile route."""
    from woof.core.ieee_module import compile_module
    # Negative controls may supply their own options; they still need the
    # production subnormal arithmetic to isolate the expression they change.
    opts = tuple(o for o in opts if o.lstrip('-').split('=')[0] != 'ftz') + ('--ftz=false',)
    return compile_module(code, opts, module_key, log_stream)


def compile_generated_slab_kernel(source, name):
    """Record each generated layout so the runtime manifest includes its code."""
    key = f"woof.core.noahmp.generated:{name}:{_sha(source)}"
    module = _compile_loaded_module(source, DEFAULT_OPTIONS, key)
    return module.get_function(name)


@lru_cache(maxsize=None)
def _selected_module_on(device, name):
    return compile_runtime_unit(
        name, module_key=f"woof.core.noahmp.selected:{name}",
        selected_options=True)


def selected_kernel(name, symbol):
    """Selected leaf image, isolated from the measured default image."""
    import cupy as cp
    return _selected_module_on(cp.cuda.runtime.getDevice(), name).get_function(symbol)



__all__ += ["DEFAULT_OPTIONS", "NOAHMP_PRICING_MODULES", "RuntimeUnit",
            "compile_runtime_unit", "compile_generated_slab_kernel", "exported_kernels", "pricing_key",
            "runtime_unit", "selected_kernel"]
