"""Physics constants on the existing, identity-bound runtime parameter path.

These knobs change forecast coefficients, so their prepared-state edits are
metadata only. They never alter a temperature, moisture or soil analysis.
The returned runtime table must be passed to the forecast. A caller that
cannot retain it is refused instead of claiming an inert field edit.
"""
from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping
from dataclasses import replace
import hashlib
from pathlib import Path
from types import MappingProxyType
import tomllib


def rows():
    return tomllib.loads(Path(__file__).with_name("families").joinpath(
        "physics.toml").read_text(encoding="utf-8"))["knobs"]


def _row(name):
    for row in rows():
        if row["name"] == name:
            return row
    raise ValueError(f"unknown physics knob {name!r}")


_BASE_REGISTRY = None
_BASE_SHA = None
_BASE_CHECK = None
_BASE_EDIT = None
_INSTALLED = False


def constant_registry():
    """Combine family rows with the existing runtime objects, without binding."""
    from woof import physics_params as pp
    base = _BASE_REGISTRY or pp.registry
    result = dict(base())
    # Short, disjoint anchors permit simultaneous closure changes. The base
    # registry's long Prandtl anchor also contains the other closure literals.
    result["mynn.prandtl"] = replace(result["mynn.prandtl"], sites=tuple(
        pp.KernelSite(module, "pr = 0.74f", "0.74f", 3)
        for module in ("mynn_pbl", "mynn_dmp_sibling")))
    for row in rows():
        if "physical_default" not in row:
            continue
        default = row["physical_default"]
        lower = default * (1 + row["min"])
        upper = default * (1 + row["max"])
        if row.get("transform") == "absolute":
            lower, upper = row["min"], row["max"]
        result[row["parameter"]] = pp.TunableConstant(
            name=row["parameter"], scheme=row["scheme"],
            requires=MappingProxyType({key: tuple(value) for key, value in
                                       row["requires"].items()}),
            kind=row.get("kind", "kernel-literal"), default=default,
            lower=lower, upper=upper, units=row.get("physical_units", "1"),
            description=row["description"], sites=tuple(pp.KernelSite(
                site["module"], site["anchor"], site["literal"], site["count"])
                for site in row.get("sites", [])))
    return MappingProxyType(result)


def install_runtime_registry():
    """Opt-in bootstrap of the existing process-bound parameter mechanism.

    Family ownership prohibits edits to the common registry. This bootstrap
    supplies family rows through its existing registry accessor before any
    affected physics kernel is built. Neutral setup kernels may already
    have restored the prepared state. The identity hashes both tables. It never
    runs for an all-zero request and never unbinds a forecast process.
    """
    global _BASE_REGISTRY, _BASE_SHA, _BASE_CHECK, _BASE_EDIT, _INSTALLED
    from woof import physics_params as pp
    if _INSTALLED:
        return
    merged = constant_registry()
    affected = {site.module for row in merged.values() for site in row.sites}
    affected.update(("rrtmg_legacy_prep", "rrtmg_mcica_wrf"))
    compiled_physics = affected.intersection(pp._STATE["compiled"])
    if pp.active() is not None or compiled_physics or pp._STATE["tables_issued"]:
        raise pp.PhysicsParamsError(
            "physics knob registry must load before any parameter set, affected physics kernel "
            "or table; otherwise cached physics could use different coefficients")
    _BASE_REGISTRY, _BASE_SHA = pp.registry, pp.registry_sha256
    identity = hashlib.sha256(b"gpuwm.physics-runtime-scalars/v1\n" + pp.registry_bytes().replace(b"\r\n", b"\n") +
        Path(__file__).with_name("families").joinpath("physics.toml").read_bytes()
        .replace(b"\r\n", b"\n")).hexdigest()
    pp.registry = lambda: merged
    pp.registry_sha256 = lambda: identity
    _BASE_CHECK = pp.check_schemes

    class EffectiveRun:
        def __init__(self, run):
            self.run = run

        def __getattr__(self, key):
            # RunConfig's -1 inherits the unified radiation selector. Check
            # the actual consumer, retaining an explicit zero as disabled.
            if key in ("ra_lw_physics", "ra_sw_physics"):
                from woof.config import radiation_scheme_ids
                pair = radiation_scheme_ids(self.run)
                return pair[0 if key == "ra_lw_physics" else 1]
            return getattr(self.run, key)

    def check_schemes(pset, run_configs, *, source):
        configs = [EffectiveRun(cfg) for cfg in run_configs]
        _BASE_CHECK(pset, configs, source=source)
        for row in rows():
            if row["parameter"] not in pset.changed():
                continue
            conditions = row.get("conditions", {})
            requirements = merged[row["parameter"]].requires
            if not any(all(getattr(cfg, key, None) in allowed for key, allowed in
                           (*requirements.items(), *conditions.items())) for cfg in configs):
                raise pp.PhysicsParamsError(
                    f"{source}: {row['name']} needs {dict(requirements)} and "
                    f"{conditions}; otherwise the parameter would be inactive "
                    "while the forecast was labeled as tuned")

    pp.check_schemes = check_schemes
    _BASE_EDIT = pp.edit_kernel_source

    def edit_kernel_source(module, text):
        active = pp.active()
        names = _runtime_names(module)
        expressions = {name: f"gpuwm_physics_scalars_v1[{i}]"
                       for i, name in enumerate(names)}
        text = _BASE_EDIT(module, text, value_expressions=expressions)
        if (module in ("mynn_pbl", "mynn_dmp_sibling") and active is not None
                and "mynn.b1" in active.changed()):
            anchor = "2.88449914061481660f"
            if text.count(anchor) != 3:
                raise pp.PhysicsParamsError(
                    "B1-dependent C1 cube-root sites moved; refusing an inconsistent closure")
            text = text.replace(anchor, expressions["__mynn_b1_root"])
        text = _edit_family_source(module, text, expressions)
        if names:
            text = ('extern "C" { __device__ __constant__ float '
                    f'gpuwm_physics_scalars_v1[{len(names)}]; }}\n' + text)
        return text

    edit_kernel_source.bind_module = bind_runtime_constants
    pp.edit_kernel_source = edit_kernel_source
    _INSTALLED = True


def parameter_values(values):
    """Translate fractional offsets to physical constants; zero omits a row."""
    from woof import physics_params as pp
    if not isinstance(values, Mapping):
        raise ValueError("physics knobs must be a mapping of names to finite numbers")
    result = {}
    registry = constant_registry()
    for name, value in values.items():
        row = _row(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        if not math.isfinite(value) or not row["min"] <= value <= row["max"]:
            raise ValueError(f"{name} outside [{row['min']}, {row['max']}]")
        if value != 0:
            key = row["parameter"]
            result[key] = (value if row.get("transform") == "absolute" else
                           registry[key].default * (1.0 + value))
    return result


def make_set(values):
    from woof import physics_params as pp
    constants = parameter_values(values)
    if not constants:
        return None
    install_runtime_registry()
    # A stable name lets different JSON key orders produce one identity.
    result = pp.make_set("physics-knobs", constants)
    _validate_closure(result)
    return result


def _validate_closure(pset):
    """Admit a jointly valid level-2 stability quadratic before card work.

    a2fac is in (0,1]. Its upper bound maximizes f1 and minimizes rf1.
    rf1 >= rf2 makes the quadratic discriminant nonnegative for every Ri.
    This checks parameter scalars only, not meteorological data paths.
    """
    names = ("mynn.b1", "mynn.b2", "mynn.g1", "mynn.c2", "mynn.c3",
             "mynn.c5", "mynn.prandtl")
    if not set(names).intersection(pset.changed()):
        return
    b1, b2, g1, c2, c3, c5, pr = (pset.value(name) for name in names)
    a1 = b1*(1-3*g1)/6
    root = b1**(1/3) if "mynn.b1" in pset.changed() else 2.88449914061481660
    c1 = g1 - 1/(3*a1*root)
    a2 = a1*(g1-c1)/(g1*pr)
    g2 = b2/b1*(1-c3)+2*a1/b1*(3-2*c2)
    f1 = b1*(g1-c1)+3*a2*(1-c2)*(1-c5)+2*a1*(3-2*c2)
    f2 = b1*(g1+g2)-3*a1*(1-c2)
    rf1 = b1*(g1-c1)/f1
    rf2 = b1*g1/f2
    if min(a1, a2, f1, f2, rf1, rf2) <= 0 or rf1-rf2 <= 1.e-5:
        from woof.physics_params import PhysicsParamsError
        raise PhysicsParamsError(
            "MYNN closure gives rf1 <= rf2, which can make the level-2 "
            "stability square-root discriminant negative and produce a "
            "non-finite wind tendency; choose jointly compatible closure constants")


def apply(state, value, ctx):
    """Store one coefficient override in the harness's runtime metadata sink."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("physics knob must be a finite number, not a boolean")
    if value == 0:
        return {"touched_fields": [], "balanced": True, "noop": True}
    name = ctx["name"]
    values = dict(ctx.get("physics_values", {}))
    values[name] = value
    pset = make_set(values)
    from woof import physics_params as pp
    pp.check_schemes(pset, [ctx["cfg"]], source="physics knobs")
    sink = ctx.get("runtime_physics")
    if not isinstance(sink, dict):
        raise ValueError(
            "physics knobs require ctx.runtime_physics metadata retained by "
            "the prepared writer and passed to WOOF_PHYSICS_PARAMS; without "
            "that sink the forecast would silently use default coefficients")
    sink.clear()
    sink.update({"name": pset.name, "values": dict(pset.values),
                 "knobs": values, "bootstrap": "woof.knobs.physics.activate_runtime",
                 "registry_sha256": pp.registry_sha256()})
    ctx.setdefault("physics_values", {})[name] = value
    return {"touched_fields": [], "balanced": True,
            "touched_parameters": list(pset.changed()),
            "physics_params": pp.document(pset),
            "scope": "all domains running the selected physics scheme"}


APPLIERS = {row["name"]: apply for row in rows()}


def registry_row():
    """Publish the family without upgrading any scheme's proof status."""
    constants = constant_registry()
    controls = {}
    replay_groups = {
        "mynn": (7152, {"prandtl", "cns", "alp1", "edmf_entrainment",
                        "b1", "b2", "g1", "c2", "c3", "c5"}),
        "thompson": (27048, {"auto_amplitude", "auto_timescale", "auto_diameter",
                            "ice_fall", "rain_fall", "graupel_fall"}),
        "ruc": (528, {"z0.tall", "z0.short", "rs"}),
    }
    for row in rows():
        constant = constants[row["parameter"]]
        controls[row["name"]] = {
            "parameter": row["parameter"], "default": 0,
            "minimum": row["min"], "maximum": row["max"],
            "units": row["units"],
            "status": "implemented-unverified",
            "requires": {key: list(value) for key, value in constant.requires.items()},
            "conditions": row.get("conditions", {}),
        }
        family, _, coefficient = row["parameter"].partition(".")
        group = replay_groups.get(family)
        words = (280 if row["parameter"] == "mynn.czil" else
                 group[0] if group and coefficient in group[1] else None)
        if row["parameter"].startswith("mynn.length2."):
            words = 7912
        elif row["parameter"] == "convective.entrainment_deep":
            words = 12832
        elif row["parameter"] == "convective.entrainment_shallow":
            words = 25664
        if words is not None:
            controls[row["name"]]["verification"] = {
                "native_source": (
                    "NOAA HRRR v4.1.21 GSD MYNN; module SHA256 "
                    "42178a22693ad14b1a3adcf3e44be3047f051abf80f3a64c0ce9ed9ac2ec6d8a"
                    if row["parameter"].startswith("mynn.length2.") else
                    "WRF v4.6.1 d66e442fccc04111067e29274c9f9eaccc3cef28"),
                "real_input_replay_words": words, "differing_words": 0,
                "scope": "joint 1% coefficient offsets in the documented case; "
                         "shared group word count, not an independent run per control",
                "forecast": "1974-04-03 12-15 UTC, 250x200x49, fast numerics; "
                            "Rust maps beside native WRF and station observations",
            }
        if family == "rrtmg":
            controls[row["name"]]["licence"] = "AER BSD-3-Clause; WRF distribution notices retained"
            controls[row["name"]]["verification"] = {
                "reference": "WRF-gated preparation wrapper and McICA endpoints; "
                             "research controls have no stock native namelist key",
                "radius_real_input_words_per_control": 507936 if coefficient != "overlap_break_fraction" else 0,
                "radius_real_input_differing_words": 0,
                "overlap_words_per_fraction": 1999296 if coefficient == "overlap_break_fraction" else 0,
                "overlap_fractions": [0, 0.25, 1] if coefficient == "overlap_break_fraction" else [],
                "scope": "16 cloudy columns per side for each 1% radius offset; "
                         "overlap synthetic native-gated endpoints in three device layouts; "
                         "joint 1% radii / 0.25 overlap whole forecast",
                "forecast": "2000-01-24 12-15 UTC, 73x60x34; Rust native/off/tuned maps and ASOS",
                "receipt": "docs/rrtmg-controls-proof.md",
            }
    return {
        "source": "woof/knobs/families/physics.toml",
        "admission": "experiment [physics_params.knobs] or WOOF_PHYSICS_PARAMS set file",
        "scope": "one coefficient set per process, shared by all domains using the scheme",
        "native_namelist_keys": False,
        "compilation": "fixed names and float32 dtypes compile; immutable values bind once to device constant memory",
        "proof": "33 case-qualified controls: 29 native coefficient replays plus four BSD-3-Clause RRTMG research controls with preparation and overlap checks. Whole three-hour forecasts, Rust maps, timing, both nest feedback modes, native WRF restart import, byte-identical checkpoint continuation and distinct ensemble members. Research controls do not confer native namelist equivalence or upgrade scheme maturity; see docs/public/PHYSICS.md and docs/rrtmg-controls-proof.md",
        "controls": controls,
    }


def kernel_source(module, text):
    """The source edit used by ordinary and direct NVRTC loaders."""
    from woof import physics_params as pp
    return pp.edit_kernel_source(module, text)


def _runtime_names(module):
    """Fixed names determine an executable; values remain device data."""
    from woof import physics_params as pp
    active = pp.active()
    if active is None:
        return ()
    names = {name for name in active.changed()
             if any(site.module == module for site in pp.registry()[name].sites)}
    if module in ("mynn_pbl", "mynn_dmp_sibling") and "mynn.b1" in names:
        names.add("__mynn_b1_root")
    if module == "thompson_classic" and "thompson.rain_fall" in names:
        names.update(("__rain_collect", "__rain_ice", "__rain_evap"))
    if module == "rrtmg_mcica_wrf" and "rrtmg.overlap_break_fraction" in active.changed():
        names.add("rrtmg.overlap_break_fraction")
    return tuple(sorted(names))


def runtime_scalar_values(module):
    """Resolve the versioned float32 payload once, before integration."""
    from woof import physics_params as pp
    active = pp.active()
    f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]
    def value(name):
        if name == "__mynn_b1_root":
            # The dependent parameter expression documented in WRF's
            # module_bl_mynn.F:286, with its real32 parameter operands.
            return f32(math.pow(f32(active.value("mynn.b1")), f32(1.0 / 3.0)))
        if name.startswith("__rain_"):
            # WRF v4.6.1 module_mp_thompson.F:786-801. Parameter
            # expressions and every runtime operation round to real32.
            av = f32(active.value("thompson.rain_fall"))
            pi = f32(3.1415926536)
            pi4 = f32(pi * f32(.25))
            if name == "__rain_collect":
                return f32(f32(pi4 * av) * f32(6.0))
            if name == "__rain_ice":
                am = f32(f32(pi * f32(1000.0)) / f32(6.0))
                crg8 = float.fromhex("0x1.680002p+9")
                return f32(f32(f32(pi4 * am) * av) * crg8)
            sc3 = float.fromhex("0x1.b761cep-1")
            return f32(f32(f32(f32(.308) * sc3) * f32(math.sqrt(av))) * f32(2.0))
        return f32(active.value(name))
    return tuple((name, value(name)) for name in _runtime_names(module))


def bind_runtime_constants(module, compiled_module):
    """Bind immutable per-process scalars on the initialization stream.

    The parameter identity already prevents a process switching sets. Each
    CUDA module owns its constant-memory payload; ensemble workers and
    devices therefore cannot overwrite another forecast's coefficients.
    No host copy, lookup or synchronization is added to a timestep.
    """
    values = runtime_scalar_values(module)
    if not values:
        return
    import ctypes
    import numpy as np
    import cupy as cp
    # Both direct NVRTC modules and RawModule reach this same binding site.
    getter = getattr(compiled_module, "get_global", None)
    pointer = (getter("gpuwm_physics_scalars_v1") if getter is not None else
               compiled_module.get_global_var("gpuwm_physics_scalars_v1"))
    payload = np.asarray([value for _, value in values], dtype=np.float32)
    if isinstance(pointer, int):
        pointer = cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(
            pointer, payload.nbytes, compiled_module), 0)
    # Constant memory is initialized with a CUDA copy, never a kernel store.
    # A setup-only synchronous copy also publishes the immutable payload to
    # every stream before a forecast or CUDA graph can use the module.
    pointer.copy_from_host(ctypes.c_void_p(payload.ctypes.data), payload.nbytes)


def _edit_family_source(module, text, expressions):
    from woof import physics_params as pp
    active = pp.active()
    if active is not None and module == "rrtmg_legacy_prep":
        for row in rows():
            name = row["parameter"]
            if not name.startswith("rrtmg.radius_") or name not in active.changed():
                continue
            for site in row["sites"]:
                if text.count(site["anchor"]) != site["count"]:
                    raise pp.PhysicsParamsError(
                        f"{name}: radius handoff moved; refusing an incomplete multiplier")
                value = expressions[name]
                replacement = site["anchor"].replace(site["literal"],
                    "RP_MU(" + site["literal"] + ", " + value + ")")
                text = text.replace(site["anchor"], replacement)
    if active is not None and module == "thompson_classic" and (
            "thompson.rain_fall" in active.changed()):
        # Replace uses, retaining the generated native default header.
        # Rain collision-table velocities are the fixed polynomial at native
        # :4188 and :4405; those tables do not depend on av_r.
        for token, name, count in (("TC_t1_qr_qc", "__rain_collect", 2),
                                   ("TC_t1_qr_qi", "__rain_collect", 2),
                                   ("TC_t2_qr_qi", "__rain_ice", 1),
                                   ("TC_t2_qr_ev", "__rain_evap", 1)):
            import re
            # Header definitions may be inlined by the production loader.
            pattern = r"(?m)^(?!#define).*\b" + token + r"\b.*$"
            lines = re.findall(pattern, text)
            occurrences = sum(len(re.findall(r"\b" + token + r"\b", line)) for line in lines)
            if occurrences != count:
                raise pp.PhysicsParamsError(f"native Thompson rain coefficient uses moved: {token}")
            text = re.sub(pattern, lambda m: re.sub(r"\b" + token + r"\b", expressions[name], m[0]), text)
    active = pp.active()
    if module == "rrtmg_legacy_prep" and active is not None and (
            "rrtmg.radius_ice" in active.changed()):
        anchor = "if (iceflg>=4) rice=ric;"
        if text.count(anchor) != 1:
            raise pp.PhysicsParamsError(
                "ice radius optics handoff moved; refusing a multiplier "
                "that could exceed the 140 micrometre optical table bound")
        # Both legacy LW and SW generalized ice optics require 5..140 um.
        # The registered multiplier already precedes the existing 5 um floor.
        text = text.replace(anchor,
            "if (iceflg>=4) rice=rp_min(140.0f, ric);")
    if module == "rrtmg_mcica_wrf" and active is not None and (
            "rrtmg.overlap_break_fraction" in active.changed()):
        anchor = "walked = (prev > clear) ? prev : RMCW_MU(walked, clear);"
        if text.count(anchor) != 2:
            raise pp.PhysicsParamsError(
                "cloud overlap walk moved; refusing an incomplete overlap mixture")
        value = expressions["rrtmg.overlap_break_fraction"]
        # Spread the selection over spectral g-points and columns, rather
        # than assigning the first bands to a different overlap treatment.
        # Each selected subcolumn keeps fresh draws at every height, so the
        # marginal layer cloud fraction is preserved. No RNG draw is added.
        replacement = ("unsigned int selector = (unsigned int)(col0 + cc) * "
            "0x9E3779B9u ^ (unsigned int)isub * 0x85EBCA6Bu; "
            "selector ^= selector >> 16; selector *= 0x7FEB352Du; "
            "selector ^= selector >> 15; selector *= 0x846CA68Bu; "
            "selector ^= selector >> 16; "
            "float select_fraction = __fmul_rn((float)(selector >> 8), 0x1p-24f); "
            "if (select_fraction >= " + value + ") { " + anchor + " }")
        text = text.replace(anchor, replacement)
    return text


def activate(values):
    """Bind a forecast process before parsing or initializing any physics."""
    pset = make_set(values)
    from woof import physics_params as pp
    if pset is None:
        if pp.active() is not None:
            raise pp.PhysicsParamsError(
                "zero physics knobs cannot reuse a process already bound to "
                "nondefault coefficients; run the default forecast in a fresh process")
        return None
    pp.declare(pset, source="physics knobs")
    return pset


def parse_parameter_table(table, *, source):
    """Admit family values through the ordinary experiment/environment door.

    The fractional request is retained so every worker installs the same
    registry before resolving its parameter identity. Optional resolved values
    are checked, never used as a second source of coefficients.
    """
    from woof import physics_params as pp
    unknown = set(table) - {"name", "values", "knobs"}
    if unknown:
        raise pp.PhysicsParamsError(f"{source}: unknown physics knob keys {sorted(unknown)}")
    pset = make_set(table["knobs"])
    if pset is None:
        if "name" in table or "values" in table:
            raise pp.PhysicsParamsError(
                f"{source}: all-zero knobs cannot label a parameter set")
        return None
    if ("name" in table and table["name"] != pset.name) or (
            "values" in table and table["values"] != dict(pset.values)):
        raise pp.PhysicsParamsError(
            f"{source}: resolved physics parameters differ from the knob request")
    return pset


def activate_runtime(runtime):
    """Forecast hook for the harness's retained knob-runtime.json document."""
    values = runtime["knobs"]
    pset = make_set(values)
    if pset is None:
        raise ValueError("runtime physics metadata names no changed parameter")
    from woof import physics_params as pp
    if (runtime["name"] != pset.name or runtime["values"] != dict(pset.values)
            or runtime["registry_sha256"] != pp.registry_sha256()):
        raise ValueError("runtime physics metadata differs from its knob values or "
                         "registry; refusing a forecast with mismatched coefficients")
    return activate(values)


def write_runtime(values, destination):
    """Write the existing environment-set format, with no file for all-zero."""
    pset = make_set(values)
    if pset is None:
        return {}
    path = Path(destination).resolve()
    text = "[physics_params]\nname = " + json.dumps(pset.name) + "\n"
    text += "[physics_params.values]\n"
    for name, value in pset.values:
        text += json.dumps(name) + " = " + repr(value) + "\n"
    text += "[physics_params.knobs]\n"
    for name, value in sorted(values.items()):
        text += json.dumps(name) + " = " + repr(float(value)) + "\n"
    path.write_text(text, encoding="utf-8")
    from woof.physics_params import ENV_VAR
    return {ENV_VAR: str(path)}


def main(argv=None):
    """Run the ordinary prepared-tree CLI in the bound worker process."""
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knobs", type=Path, required=True)
    parser.add_argument("--parameter-file", type=Path, required=True)
    parser.add_argument("forecast_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate physics knob {key!r}")
            result[key] = value
        return result
    values = json.loads(args.knobs.read_text(encoding="utf-8-sig"), object_pairs_hook=unique)
    from woof.physics_params import ENV_VAR
    if os.environ.get(ENV_VAR):
        raise ValueError("the physics knob worker requires its own parameter set; "
                         "an inherited WOOF_PHYSICS_PARAMS could run different coefficients")
    env = write_runtime(values, args.parameter_file)
    activate(values)
    os.environ.update(env)
    from woof.prepared_domain_tree_forecast import main as forecast
    rest = args.forecast_args
    return forecast(rest[1:] if rest[:1] == ["--"] else rest)


if __name__ == "__main__":
    raise SystemExit(main())
