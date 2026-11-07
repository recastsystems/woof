"""Prepared dry EOS diagnostics over explicitly inventoried member state.

The existing calc_p_alpha source performs all arithmetic. This module binds
original scalar conversions and base layouts to one member-wide launch. It
allocates no device fields and does not implement a complete forecast driver.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from woof.core.device_inventory import state_array_shapes
from woof.ensemble.batch_kernel import KernelSpec, PointerSpec, prepare_batch_kernel_launch
from woof.ensemble.batch_state import BatchedDomainState, BatchStateUnsupported
from woof.ensemble.batch_storage import BatchArraySpec


def _strict_enabled():
    from woof.wrf_exact import DIAGNOSTICS_ENABLED
    return DIAGNOSTICS_ENABLED


def diagnostics_specs(cfg, *, strict=None):
    """Exact optional state extension required by the installed EOS branch.

    Ordinary diagnostics write existing p/al/alt fields. The strict branch
    also carries perturbation pressure in a separate mutable member field.
    Pass this declaration and complete prepared host buffers to the state's
    extra_specs seam before admission/allocation; the launcher never invents
    an empty_like allocation. This is not a whole-forecast admission model.
    """
    strict = _strict_enabled() if strict is None else strict
    if not isinstance(strict, (bool, np.bool_)):
        raise TypeError("strict diagnostic inventory selection must be boolean")
    if not strict:
        return ()
    if "p_perturbation" in state_array_shapes(cfg):
        return ()  # The strict constructor already owns this member field.
    return (BatchArraySpec("p_perturbation", state_array_shapes(cfg)["p"], "member"),)


def diagnostic_pointer_fields(*, strict=None, moist=False):
    """CUDA parameter names and their existing allocation identities."""
    strict = _strict_enabled() if strict is None else strict
    pointers = [
        ("thp", "thp"), ("php", "php"), ("mup", "mup"), ("thb", "thb"),
        ("phb", "phb"), ("dphbr", "dphb_resid"), ("alb", "alb"),
    ]
    if strict:
        pointers.append(("pb", "pb"))
    pointers.extend((name, name) for name in (
        "rdnw", "c1h", "c2h", "c3h", "c4h", "c3f", "c4f", "dc3f", "dc4f"))
    # The inactive qv placeholder is thp, exactly as the original helper.
    pointers.extend((("mub", "mub2d"), ("qv", "qv" if moist else "thp"),
                     ("p", "p"), ("al", "al"), ("alt", "alt")))
    if strict:
        pointers.append(("p_perturbation", "p_perturbation"))
    return tuple(pointers)


def diagnostics_kernel_spec(state, *, strict=None):
    """Resolve every pointer's role from audited state storage, CPU only."""
    if not isinstance(state, BatchedDomainState):
        raise TypeError("a diagnostics batch specification needs inventoried state storage")
    strict = _strict_enabled() if strict is None else strict
    original_shapes = state_array_shapes(state.cfg)
    expected_extra = {spec.name: spec for spec in diagnostics_specs(state.cfg, strict=strict)}
    output_names = {"p", "al", "alt", "p_perturbation"}
    pointers = []
    for parameter, name in diagnostic_pointer_fields(strict=strict, moist=bool(state.cfg.moist and not state.cfg.use_theta_m)):
        if name not in state.storage.specs:
            raise BatchStateUnsupported(
                f"diagnostics requires planned member allocation {name!r}; "
                "implicit pressure-carrier allocation would bypass admission")
        spec = state.storage.specs[name]
        expected_shape = (expected_extra[name].shape if name in expected_extra else original_shapes[name])
        if spec.shape != expected_shape or np.dtype(spec.dtype) != np.dtype(np.float32):
            raise ValueError(f"diagnostic allocation {name!r} differs from its original shape/dtype")
        if parameter in output_names and spec.ownership != "member":
            raise BatchStateUnsupported(f"diagnostic output {name!r} cannot share mutable member backing")
        pointers.append(PointerSpec(parameter, spec.ownership, spec.dtype))
    return KernelSpec("diagnostics", "calc_p_alpha", tuple(pointers))


def _single_view(state):
    if not isinstance(state, BatchedDomainState):
        return state
    values = dict(state.scalars)
    values.update({name: state.member_view(name, 0) for name in state.storage.specs
                   if not name.startswith("scratch:")})
    values.setdefault("qv", None)
    return SimpleNamespace(**values)


def _require_strict_scalar_output(state):
    output = getattr(state, "p_perturbation", None)
    if output is None:
        raise BatchStateUnsupported(
            "strict scalar diagnostics need an explicitly inventoried "
            "p_perturbation backing before binding; the scalar helper's lazy "
            "allocation would otherwise bypass the declared memory plan")
    if output.shape != state.p.shape or output.dtype != np.dtype(np.float32):
        raise ValueError("strict scalar perturbation pressure differs from the pressure shape/dtype")


def prepare_update_diagnostics(state, hypsometric_opt=1, window=None):
    """Bind one dry EOS operation using the original helper's argument tree.

    Both hypsometric options, flat/per-column bases, hybrid coefficients and
    exact column windows retain the original CUDA source. Map factors are not
    EOS inputs. The strict source retains its canonical theta-minus-300
    convention; this adapter does not replace that carrier with full theta.
    N=1 calls update_diagnostics on original-shaped fields with all optional
    outputs already present. N>1 binds one prepared raw launch and performs no
    field allocation, member loop, reduction, decode, regrid or file operation.
    """
    from woof.core import diagnostics
    if hypsometric_opt not in (1, 2):
        raise ValueError(f"hypsometric_opt must be 1 or 2, got {hypsometric_opt}")
    if hypsometric_opt == 2 and state.p_top is None:
        raise RuntimeError("hypsometric_opt=2 needs state.p_top: load_base must run before the log-pressure EOS diagnostic")
    batch = isinstance(state, BatchedDomainState)
    moist = bool(state.cfg.moist) if batch else getattr(state, "qv", None) is not None
    if moist and getattr(state, "qv", None) is None:
        raise BatchStateUnsupported("moist diagnostics need the admitted member vapor field; an inactive dummy would diagnose dry pressure")
    moist = moist and not int(getattr(state, "use_theta_m", getattr(getattr(state, "cfg", None), "use_theta_m", 0)))
    strict = _strict_enabled()
    if strict != bool(diagnostics.DIAGNOSTICS_ENABLED):
        raise BatchStateUnsupported("diagnostic compiler mode changed after the original helper was imported; a fresh process is needed for one consistent pressure ABI")
    members = state.members if batch else 1
    if batch:
        spec = diagnostics_kernel_spec(state, strict=strict)
        nz, ny, nx = state.storage.specs["p"].shape
    else:
        nz, ny, nx = state.p.shape
    normalized_window = diagnostics._validated_window(window, ny, nx)
    if members == 1:
        single = _single_view(state)
        if strict:
            _require_strict_scalar_output(single)

        def launch():
            diagnostics.update_diagnostics(single, hypsometric_opt, window)

        launch.numerical_entries = ("calc_p_alpha",)
        launch.window = normalized_window
        return launch
    a = lambda name: state.storage.arrays[name]
    args = (a("thp"), a("php"), a("mup"), a("thb"), a("phb"),
            a("dphb_resid"), a("alb"))
    if strict:
        args += (a("pb"),)
    args += tuple(a(name) for name in (
        "rdnw", "c1h", "c2h", "c3h", "c4h", "c3f", "c4f", "dc3f", "dc4f", "mub2d"))
    args += (a("qv" if moist else "thp"), np.float32(0.0 if state.p_top is None else state.p_top),
             np.int32(hypsometric_opt), np.int32(moist),
             np.int32(len(state.storage.specs["thb"].shape) == 3),
             np.int32(nz), np.int32(ny), np.int32(nx))
    args += tuple(np.int32(value) for value in normalized_window)
    args += (a("p"), a("al"), a("alt"))
    if strict:
        args += (a("p_perturbation"),)
    strides = {parameter: state.storage.pointer_stride_bytes(name)
               for parameter, name in diagnostic_pointer_fields(strict=strict, moist=moist)}
    _, _, nyw, nxw = normalized_window
    from woof.core.kernel_shape import shape_defines
    launch = prepare_batch_kernel_launch(
        spec, members, ((nxw * nyw + 255) // 256,), (256,), args,
        pointer_strides=strides,
        defines=shape_defines("diagnostics", "calc_p_alpha", (nz, ny, nx)))
    launch.numerical_entries = ("calc_p_alpha",)
    launch.window = normalized_window
    return launch
