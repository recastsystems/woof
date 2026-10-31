"""Direct sm_120 FTZ/DAZ measurements for additive v8.4.1 CUDA kernels.

This module launches every release-specific production entrypoint with a
source-derived bit anchor.  It is intentionally separate from the shared
inherited-kernel audit implementation: the combined 95-entrypoint receipt is
assembled by :mod:`woof.hex.cuda_ftz` only after two enabled and two
disabled-fallback passes are byte-identical.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _run_v841_new_kernel_audit_once(
    *,
    fallback_disabled: bool,
    cache_dir: str | Path,
) -> dict[str, Any]:
    """Launch each of the 31 release-specific compiled entrypoints once."""

    from . import (
        cuda_acoustic_v841,
        cuda_dynamics_v841,
        cuda_horizontal_v841,
        cuda_transport,
        cuda_transport_v841,
    )
    from .cuda_backend import KernelCache, require_cuda
    from .cuda_ftz import _run_transport_kernel_localization

    capability = require_cuda(
        min_compute=(12, 0),
        required_compute=(12, 0),
        cache_dir=cache_dir,
    )
    import cupy as cp

    sources = {
        "hexcore.cuda_acoustic_v841": cuda_acoustic_v841._CUDA_SOURCE,
        "hexcore.cuda_dynamics_v841": cuda_dynamics_v841._CUDA_SOURCE,
        "hexcore.cuda_horizontal_v841": cuda_horizontal_v841._CUDA_SOURCE,
        "hexcore.cuda_transport_v841": cuda_transport_v841._CUDA_SOURCE,
    }
    caches = {
        module_key: KernelCache(capability=capability, cache_dir=cache_dir)
        for module_key in sources
    }
    prefix = "#define MPAS_FTZ_FALLBACK_ENABLED 0\n" if fallback_disabled else ""
    records: dict[str, Any] = {}
    sub = np.uint32(0x000116C2).view(np.float32)
    negative_sub = np.uint32(
        np.uint32(sub.view(np.uint32)) | np.uint32(0x80000000)
    ).view(np.float32)

    def device(value: Any, dtype: Any = np.float32) -> Any:
        return cp.asarray(np.asarray(value, dtype=dtype))

    def zeros(shape: Any, dtype: Any = np.float32) -> Any:
        return cp.zeros(shape, dtype=dtype)

    def kernel(module_key: str, name: str) -> Any:
        return caches[module_key].raw_kernel(
            name,
            prefix + sources[module_key],
            module_key=module_key,
        )

    def launch(
        module_key: str,
        name: str,
        count: int,
        args: tuple[Any, ...],
    ) -> None:
        threads = 128
        kernel(module_key, name)(
            ((count + threads - 1) // threads,),
            (threads,),
            args,
        )
        cp.cuda.runtime.deviceSynchronize()

    def record(
        module_key: str,
        name: str,
        *,
        lane: str,
        classification: str,
        expected: Mapping[str, np.ndarray],
        actual: Mapping[str, Any],
    ) -> None:
        expected_hashes = {
            field: _array_sha256(np.asarray(value)) for field, value in expected.items()
        }
        actual_hashes = {
            field: _array_sha256(cp.asnumpy(value)) for field, value in actual.items()
        }
        records[f"{module_key}::{name}"] = {
            "translation_unit": module_key,
            "kernel": name,
            "classification": classification,
            "lane": lane,
            "expected_bits": expected_hashes,
            "observed_bits": actual_hashes,
            "matches_expected": actual_hashes == expected_hashes,
        }

    acoustic = "hexcore.cuda_acoustic_v841"
    dynamics = "hexcore.cuda_dynamics_v841"
    horizontal = "hexcore.cuda_horizontal_v841"
    transport = "hexcore.cuda_transport_v841"

    # Acoustic: copy, coefficient construction, each explicit update, and the
    # complete column solve are separately observed.
    cofrz = zeros(1)
    launch(acoustic, "acoustic_cofrz_v841", 1, (np.int32(1), device([sub]), cofrz))
    record(
        acoustic,
        "acoustic_cofrz_v841",
        lane="bitwise rdzw copy",
        classification="fallback_invariant",
        expected={"cofrz": np.asarray([sub], dtype=np.float32)},
        actual={"cofrz": cofrz},
    )

    nlev = 1
    cell = (nlev, 1)
    cofwr = zeros(cell)
    cofwz = zeros(cell)
    coftz = zeros((nlev + 1, 1))
    cofwt = zeros(cell)
    a_tri = zeros(cell)
    b_tri = zeros(cell)
    c_tri = zeros(cell)
    alpha = zeros(cell)
    gamma = zeros(cell)
    singular = zeros(1, np.int32)
    launch(
        acoustic,
        "acoustic_coefficients_v841",
        1,
        (
            np.int32(nlev),
            np.int32(1),
            np.float32(1.0),
            np.float32(1.0),
            np.float32(1.0),
            np.float32(2.0),
            device([[1.0]]),
            device([[1.0]]),
            device([[1.0]]),
            device([[1.0]]),
            device([[sub]]),
            device([[1.0]]),
            device([[1.0]]),
            device([[0.0]]),
            device([[0.0]]),
            device([1.0]),
            device([0.0]),
            device([0.0]),
            device([0.0]),
            device([1.0]),
            device([1.0, 1.0]),
            cofwr,
            cofwz,
            coftz,
            cofwt,
            device([1.0]),
            a_tri,
            b_tri,
            c_tri,
            alpha,
            gamma,
            singular,
        ),
    )
    expected_half_sub = np.float32(np.float64(sub) * 0.5)
    record(
        acoustic,
        "acoustic_coefficients_v841",
        lane="subnormal rho_base in released cofwt product chain",
        classification="guarded_fallback_required",
        expected={"cofwt": np.asarray([[expected_half_sub]], dtype=np.float32)},
        actual={"cofwt": cofwt},
    )

    ru_p = zeros((1, 1))
    ru_avg = zeros((1, 1))
    launch(
        acoustic,
        "acoustic_ru_v841",
        1,
        (
            np.int32(1),
            np.int32(1),
            np.int32(2),
            np.int32(1),
            np.float32(1.0),
            np.float32(1.0),
            np.float32(1.0),
            np.float32(2.0),
            device([0, 1], np.int32),
            device([1.0]),
            device([[1.0, 1.0]]),
            device([[1.0, 1.0]]),
            device([[1.0]]),
            device([[0.0]]),
            device([[sub]]),
            zeros((1, 2)),
            zeros((1, 2)),
            ru_p,
            ru_avg,
        ),
    )
    record(
        acoustic,
        "acoustic_ru_v841",
        lane="first small step dts times subnormal edge tendency",
        classification="guarded_fallback_required",
        expected={
            "ru_p": np.asarray([[sub]], dtype=np.float32),
            "ru_avg": np.asarray([[sub]], dtype=np.float32),
        },
        actual={"ru_p": ru_p, "ru_avg": ru_avg},
    )

    rw_p = zeros((3, 1))
    rtheta = device([[sub], [sub]])
    rtheta_old = zeros((2, 1))
    rho_pp = zeros((2, 1))
    ww_avg = zeros((3, 1))
    launch(
        acoustic,
        "acoustic_prepare_v841",
        1,
        (
            np.int32(2),
            np.int32(1),
            np.int32(2),
            rw_p,
            rtheta,
            rtheta_old,
            rho_pp,
            ww_avg,
        ),
    )
    record(
        acoustic,
        "acoustic_prepare_v841",
        lane="small-step rtheta snapshot copy",
        classification="fallback_invariant",
        expected={"rtheta_old": np.asarray([[sub], [sub]], dtype=np.float32)},
        actual={"rtheta_old": rtheta_old},
    )

    rs = zeros((1, 1))
    ts = zeros((1, 1))
    launch(
        acoustic,
        "acoustic_rs_ts_v841",
        1,
        (
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.float32(1.0),
            device([0], np.int32),
            device([0], np.int32),
            device([0, 0], np.int32),
            device([1.0]),
            device([1.0]),
            device([1.0]),
            device([[300.0]]),
            device([1.0]),
            device([0.0]),
            zeros((2, 1)),
            zeros((2, 1)),
            zeros((1, 1)),
            zeros((2, 1)),
            zeros((1, 1)),
            zeros((1, 1)),
            device([[sub]]),
            device([[sub]]),
            rs,
            ts,
        ),
    )
    record(
        acoustic,
        "acoustic_rs_ts_v841",
        lane="subnormal cell tendencies with zero flux",
        classification="guarded_fallback_required",
        expected={
            "rs": np.asarray([[sub]], dtype=np.float32),
            "ts": np.asarray([[sub]], dtype=np.float32),
        },
        actual={"rs": rs, "ts": ts},
    )

    nlev = 2
    rw_p = zeros((3, 1))
    rho_pp = zeros((2, 1))
    rtheta_pp = zeros((2, 1))
    ww_avg = zeros((3, 1))
    zero_cell = zeros((2, 1))
    zero_interface = zeros((3, 1))
    launch(
        acoustic,
        "acoustic_column_solve_v841",
        1,
        (
            np.int32(nlev),
            np.int32(1),
            np.float32(1.0),
            device([[1.0], [1.0]]),
            device([[1.0], [1.0]]),
            device([0.0, 0.5]),
            device([0.0, 0.5]),
            device([1.0, 1.0]),
            zero_cell,
            zero_interface,
            zero_interface,
            zero_interface,
            device([[0.0], [sub], [0.0]]),
            device([[sub], [sub]]),
            device([[sub], [sub]]),
            zero_cell,
            zero_cell,
            zeros((3, 1)),
            zero_cell,
            device([0.0, 0.0]),
            zero_cell,
            device([[1.0], [1.0]]),
            zero_cell,
            device([0.0, 0.0]),
            device([0.0, 0.0]),
            device([1.0, 1.0, 1.0]),
            device([1.0, 1.0, 1.0]),
            rw_p,
            rho_pp,
            rtheta_pp,
            ww_avg,
        ),
    )
    record(
        acoustic,
        "acoustic_column_solve_v841",
        lane="subnormal inner vertical tendency through dts and identity Thomas solve",
        classification="guarded_fallback_required",
        expected={
            "rw_p": np.asarray([[0.0], [sub], [0.0]], dtype=np.float32),
            "rho_pp": np.asarray([[sub], [sub]], dtype=np.float32),
            "rtheta_pp": np.asarray([[sub], [sub]], dtype=np.float32),
        },
        actual={"rw_p": rw_p, "rho_pp": rho_pp, "rtheta_pp": rtheta_pp},
    )

    # Dynamics and reduction kernels.
    result = zeros((1, 1))
    launch(
        dynamics,
        "vector_momentum_v841_f32",
        1,
        (
            np.int32(1),
            np.int32(2),
            np.int32(1),
            np.int32(1),
            zeros((1, 1)),
            device([[1.0]]),
            zeros((1, 1)),
            device([[0.0, sub]]),
            zeros((1, 2)),
            device([0, 1], np.int32),
            device([0], np.int32),
            device([0], np.int32),
            device([0.0]),
            device([1.0]),
            device([0.0]),
            device([0.0]),
            device([0.0]),
            device([0.0]),
            result,
        ),
    )
    record(
        dynamics,
        "vector_momentum_v841_f32",
        lane="stored-inverse subnormal kinetic-energy gradient",
        classification="guarded_fallback_required",
        expected={"result": np.asarray([[negative_sub]], dtype=np.float32)},
        actual={"result": result},
    )

    theta_result = zeros((1, 1))
    launch(
        dynamics,
        "theta_finish_v841_f32",
        1,
        (
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            device([1], np.int32),
            device([0], np.int32),
            device([-1.0]),
            device([1.0]),
            device([0.0]),
            device([[sub]]),
            zeros((2, 1)),
            theta_result,
        ),
    )
    record(
        dynamics,
        "theta_finish_v841_f32",
        lane="subnormal horizontal theta flux with stored inverse area",
        classification="guarded_fallback_required",
        expected={"result": np.asarray([[sub]], dtype=np.float32)},
        actual={"result": theta_result},
    )

    w_result = zeros((3, 1))
    launch(
        dynamics,
        "w_finish_v841_f32",
        1,
        (
            np.int32(2),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            device([1], np.int32),
            device([0], np.int32),
            device([-1.0]),
            device([1.0]),
            device([0.0, 0.0]),
            device([[0.0], [sub], [0.0]]),
            zeros((3, 1)),
            w_result,
        ),
    )
    record(
        dynamics,
        "w_finish_v841_f32",
        lane="subnormal horizontal w flux with explicit endpoints",
        classification="guarded_fallback_required",
        expected={"result": np.asarray([[0.0], [sub], [0.0]], dtype=np.float32)},
        actual={"result": w_result},
    )

    endpoints = device([[sub], [1.0], [sub]])
    launch(
        dynamics,
        "enforce_rw_endpoints_v841_f32",
        1,
        (np.int32(2), np.int32(1), endpoints),
    )
    record(
        dynamics,
        "enforce_rw_endpoints_v841_f32",
        lane="explicit positive-zero endpoint assignment",
        classification="fallback_invariant",
        expected={"rw": np.asarray([[0.0], [1.0], [0.0]], dtype=np.float32)},
        actual={"rw": endpoints},
    )

    flag = zeros(1, np.int32)
    launch(
        dynamics,
        "validate_positive_state_v841_f32",
        1,
        (np.int32(1), device([sub]), device([sub]), flag),
    )
    record(
        dynamics,
        "validate_positive_state_v841_f32",
        lane="DAZ-refused positive subnormal state validation",
        classification="fallback_invariant",
        expected={"invalid": np.asarray([1], dtype=np.int32)},
        actual={"invalid": flag},
    )

    flag = zeros(1, np.int32)
    launch(
        dynamics,
        "validate_recovered_v841_f32",
        2,
        (
            np.int32(1),
            np.int32(1),
            np.int32(1),
            device([[1.0]]),
            device([[1.0]]),
            device([[sub]]),
            device([[0.0], [0.0]]),
            device([[1.0]]),
            device([[1.0]]),
            device([[sub]]),
            device([[sub]]),
            device([[sub]]),
            device([[sub]]),
            device([[0.0], [0.0]]),
            flag,
        ),
    )
    record(
        dynamics,
        "validate_recovered_v841_f32",
        lane="finite subnormal perturbation and momentum validation",
        classification="fallback_invariant",
        expected={"invalid": np.asarray([0], dtype=np.int32)},
        actual={"invalid": flag},
    )

    flag = zeros(1, np.int32)
    launch(
        dynamics,
        "validate_finite_array_v841_f32",
        1,
        (np.int32(1), device([sub]), flag),
    )
    record(
        dynamics,
        "validate_finite_array_v841_f32",
        lane="finite subnormal scalar validation",
        classification="fallback_invariant",
        expected={"invalid": np.asarray([0], dtype=np.int32)},
        actual={"invalid": flag},
    )

    accumulator = zeros((1, 1))
    launch(
        dynamics,
        "split_flux_first_v841_f32",
        1,
        (np.int32(1), np.int32(1), device([[sub]]), accumulator),
    )
    record(
        dynamics,
        "split_flux_first_v841_f32",
        lane="first split-flux bitwise copy",
        classification="fallback_invariant",
        expected={"accumulator": np.asarray([[sub]], dtype=np.float32)},
        actual={"accumulator": accumulator},
    )

    accumulator = zeros((1, 1))
    launch(
        dynamics,
        "split_flux_add_v841_f32",
        1,
        (np.int32(1), np.int32(1), device([[sub]]), accumulator),
    )
    record(
        dynamics,
        "split_flux_add_v841_f32",
        lane="source-order current plus zero accumulator",
        classification="guarded_fallback_required",
        expected={"accumulator": np.asarray([[sub]], dtype=np.float32)},
        actual={"accumulator": accumulator},
    )

    average = zeros((1, 1))
    launch(
        dynamics,
        "split_flux_finish_v841_f32",
        1,
        (np.int32(1), np.int32(1), np.float32(1.0), device([[sub]]), average),
    )
    record(
        dynamics,
        "split_flux_finish_v841_f32",
        lane="typed reciprocal times subnormal split sum",
        classification="guarded_fallback_required",
        expected={"average": np.asarray([[sub]], dtype=np.float32)},
        actual={"average": average},
    )

    # Released stored-inverse horizontal operators.
    vorticity = zeros((1, 1))
    ke_vertex = zeros((1, 1))
    pv_vertex = zeros((1, 1))
    launch(
        horizontal,
        "vertex_diagnostics_v841_f32",
        1,
        (
            device([[sub]]),
            zeros((1, 1)),
            device([0], np.int32),
            device([0, 0], np.int32),
            device([1.0]),
            device([1.0]),
            device([0.0]),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            vorticity,
            ke_vertex,
            pv_vertex,
        ),
    )
    record(
        horizontal,
        "vertex_diagnostics_v841_f32",
        lane="subnormal circulation times stored inverse triangle area",
        classification="guarded_fallback_required",
        expected={"vorticity": np.asarray([[negative_sub]], dtype=np.float32)},
        actual={"vorticity": vorticity},
    )

    divergence = zeros((1, 1))
    kinetic = zeros((1, 1))
    launch(
        horizontal,
        "cell_diagnostics_v841_f32",
        1,
        (
            device([[sub]]),
            zeros((1, 1)),
            zeros((1, 1)),
            device([0], np.int32),
            device([1], np.int32),
            device([0, 0], np.int32),
            device([0], np.int32),
            device([0], np.int32),
            device([1.0]),
            device([1.0]),
            device([0.0]),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            divergence,
            kinetic,
        ),
    )
    record(
        horizontal,
        "cell_diagnostics_v841_f32",
        lane="subnormal cell divergence times stored inverse area",
        classification="guarded_fallback_required",
        expected={"divergence": np.asarray([[sub]], dtype=np.float32)},
        actual={"divergence": divergence},
    )

    pv_cell = zeros((1, 1))
    launch(
        horizontal,
        "pv_cell_v841_f32",
        1,
        (
            device([[sub]]),
            device([0], np.int32),
            device([1], np.int32),
            device([0], np.int32),
            device([1.0]),
            device([1.0]),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            pv_cell,
        ),
    )
    record(
        horizontal,
        "pv_cell_v841_f32",
        lane="subnormal kite PV times stored inverse area",
        classification="guarded_fallback_required",
        expected={"pv_cell": np.asarray([[sub]], dtype=np.float32)},
        actual={"pv_cell": pv_cell},
    )

    pv_edge = zeros((1, 1))
    grad_n = zeros((1, 1))
    grad_t = zeros((1, 1))
    launch(
        horizontal,
        "pv_apvm_v841_f32",
        1,
        (
            zeros((1, 1)),
            zeros((1, 1)),
            device([[0.0, sub]]),
            device([[0.0, sub]]),
            device([0, 1], np.int32),
            device([0, 1], np.int32),
            device([1.0]),
            device([1.0]),
            np.float32(1.0),
            np.int32(1),
            np.int32(2),
            np.int32(1),
            np.int32(2),
            pv_edge,
            grad_n,
            grad_t,
        ),
    )
    record(
        horizontal,
        "pv_apvm_v841_f32",
        lane="subnormal normal and tangential PV gradients",
        classification="guarded_fallback_required",
        expected={
            "grad_n": np.asarray([[sub]], dtype=np.float32),
            "grad_t": np.asarray([[sub]], dtype=np.float32),
        },
        actual={"grad_n": grad_n, "grad_t": grad_t},
    )

    mass_divergence = zeros((1, 1))
    launch(
        horizontal,
        "mass_flux_divergence_v841_f32",
        1,
        (
            device([[sub]]),
            device([0], np.int32),
            device([1], np.int32),
            device([0, 0], np.int32),
            device([1.0]),
            device([1.0]),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            mass_divergence,
        ),
    )
    record(
        horizontal,
        "mass_flux_divergence_v841_f32",
        lane="subnormal mass flux times stored inverse area",
        classification="guarded_fallback_required",
        expected={"mass_divergence": np.asarray([[sub]], dtype=np.float32)},
        actual={"mass_divergence": mass_divergence},
    )

    tendency = zeros((1, 1))
    launch(
        horizontal,
        "pressure_gradient_v841_f32",
        1,
        (
            device([[0.0, sub]]),
            zeros((1, 2)),
            device([[1.0]]),
            device([[1.0, 1.0]]),
            zeros((1, 1)),
            device([0, 1], np.int32),
            device([1.0]),
            np.int32(1),
            np.int32(2),
            np.int32(1),
            tendency,
        ),
    )
    record(
        horizontal,
        "pressure_gradient_v841_f32",
        lane="subnormal pressure difference times stored inverse dc",
        classification="guarded_fallback_required",
        expected={"tendency": np.asarray([[negative_sub]], dtype=np.float32)},
        actual={"tendency": tendency},
    )

    # New transport entrypoints, including the currently refused FCT surface.
    target = zeros(1)
    launch(
        transport,
        "transport_interpolate_target_v841",
        1,
        (np.int32(1), np.float32(0.0), device([sub]), device([0.0]), target),
    )
    record(
        transport,
        "transport_interpolate_target_v841",
        lane="zero-weight target interpolation preserving subnormal old density",
        classification="guarded_fallback_required",
        expected={"target": np.asarray([sub], dtype=np.float32)},
        actual={"target": target},
    )

    flag = zeros(1, np.int32)
    launch(
        transport,
        "validate_density_v841",
        1,
        (np.int32(1), device([sub]), flag),
    )
    record(
        transport,
        "validate_density_v841",
        lane="DAZ-refused positive subnormal density validation",
        classification="fallback_invariant",
        expected={"invalid": np.asarray([1], dtype=np.int32)},
        actual={"invalid": flag},
    )

    flag = zeros(1, np.int32)
    launch(
        transport,
        "validate_transport_indices_v841",
        1,
        (
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            device([1], np.int32),
            device([0], np.int32),
            device([0], np.int32),
            device([0, 0], np.int32),
            device([1], np.int32),
            device([0], np.int32),
            flag,
        ),
    )
    record(
        transport,
        "validate_transport_indices_v841",
        lane="valid one-cell one-edge transport topology",
        classification="fallback_invariant",
        expected={"invalid": np.asarray([0], dtype=np.int32)},
        actual={"invalid": flag},
    )

    output = zeros((1, 1, 1))
    launch(
        transport,
        "transport_standard_finish_v841",
        1,
        (
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.float32(0.0),
            device([0], np.int32),
            device([0], np.int32),
            device([1.0]),
            device([1.0]),
            zeros((1, 1)),
            zeros((1, 1, 1)),
            zeros((1, 2, 1)),
            device([1.0]),
            device([[[sub]]]),
            device([[1.0]]),
            device([[1.0]]),
            zeros((1, 1, 1)),
            output,
        ),
    )
    record(
        transport,
        "transport_standard_finish_v841",
        lane="subnormal old scalar mass at zero dt times reciprocal density",
        classification="guarded_fallback_required",
        expected={"output": np.asarray([[[sub]]], dtype=np.float32)},
        actual={"output": output},
    )

    target_density = zeros((1, 1))
    launch(
        transport,
        "transport_target_density_v841",
        1,
        (
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.float32(1.0),
            device([0], np.int32),
            device([0], np.int32),
            device([1.0]),
            device([1.0]),
            device([1.0]),
            zeros((1, 1)),
            zeros((2, 1)),
            device([1.0]),
            device([[sub]]),
            device([[1.0]]),
            target_density,
        ),
    )
    record(
        transport,
        "transport_target_density_v841",
        lane="subnormal old density plus zero advanced flux",
        classification="guarded_fallback_required",
        expected={"target_density": np.asarray([[sub]], dtype=np.float32)},
        actual={"target_density": target_density},
    )

    edge_values = zeros((1, 1, 1))
    launch(
        transport,
        "transport_edge_values_mono_v841",
        1,
        (
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.float32(0.25),
            device([[[sub]]]),
            device([[1.0]]),
            device([[1.0]]),
            device([[0.0]]),
            device([1], np.int32),
            device([[0]], np.int32),
            edge_values,
        ),
    )
    record(
        transport,
        "transport_edge_values_mono_v841",
        lane="generic monotonic stencil velocity times subnormal scalar",
        classification="guarded_fallback_required",
        expected={"edge_values": np.asarray([[[sub]]], dtype=np.float32)},
        actual={"edge_values": edge_values},
    )

    upwind = zeros((1, 1, 1))
    residual = zeros((1, 1, 1))
    launch(
        transport,
        "fct_edge_residual_v841",
        1,
        (
            np.int32(1),
            np.int32(1),
            np.int32(2),
            np.int32(1),
            np.float32(1.0),
            device([[[sub, sub]]]),
            device([[1.0]]),
            device([1.0]),
            device([0, 1], np.int32),
            zeros((1, 1, 1)),
            upwind,
            residual,
        ),
    )
    record(
        transport,
        "fct_edge_residual_v841",
        lane="subnormal upwind scalar flux and signed residual",
        classification="guarded_fallback_required",
        expected={
            "upwind": np.asarray([[[sub]]], dtype=np.float32),
            "residual": np.asarray([[[negative_sub]]], dtype=np.float32),
        },
        actual={"upwind": upwind, "residual": residual},
    )

    mass = zeros((1, 1, 2))
    scale_in = zeros((1, 1, 2))
    scale_out = zeros((1, 1, 2))
    launch(
        transport,
        "fct_horizontal_low_order_v841",
        2,
        (
            np.int32(1),
            np.int32(1),
            np.int32(2),
            np.int32(1),
            np.int32(1),
            device([1, 1], np.int32),
            device([[0], [0]], np.int32),
            device([[-1.0], [1.0]]),
            device([1.0, 1.0]),
            device([[[sub]]]),
            zeros((1, 1, 1)),
            mass,
            scale_in,
            scale_out,
        ),
    )
    record(
        transport,
        "fct_horizontal_low_order_v841",
        lane="signed subnormal upwind flux times stored inverse area",
        classification="guarded_fallback_required",
        expected={"mass": np.asarray([[[sub, negative_sub]]], dtype=np.float32)},
        actual={"mass": mass},
    )

    output = zeros((1, 1, 1))
    launch(
        transport,
        "fct_finish_v841",
        1,
        (
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            np.int32(1),
            device([0], np.int32),
            device([0], np.int32),
            device([1.0]),
            device([1.0]),
            device([1.0]),
            zeros((1, 1, 1)),
            zeros((1, 2, 1)),
            device([[1.0]]),
            device([[[sub]]]),
            output,
        ),
    )
    record(
        transport,
        "fct_finish_v841",
        lane="subnormal limited mass divided by unit target density",
        classification="guarded_fallback_required",
        expected={"output": np.asarray([[[sub]]], dtype=np.float32)},
        actual={"output": output},
    )

    # The v8.4.1 transport TU embeds the twelve inherited production
    # entrypoints as well as the nine release-specific ones above.  Reuse the
    # frozen direct-launch deck, but force every launch through this exact
    # v8.4.1 source and manifest owner rather than cuda_transport's source.
    class _ExactTransportSourceCache:
        def raw_kernel(
            self,
            name: str,
            ignored_source: str,
            *,
            module_key: str,
            options: tuple[str, ...] = (),
        ) -> Any:
            del ignored_source, module_key
            return caches[transport].raw_kernel(
                name,
                prefix + sources[transport],
                module_key=transport,
                options=options,
            )

    inherited = _run_transport_kernel_localization(
        cp,
        cuda_transport,
        _ExactTransportSourceCache(),
    )
    for row in inherited:
        name = str(row["kernel"])
        fields = row["fields"]
        records[f"{transport}::{name}"] = {
            "translation_unit": transport,
            "kernel": name,
            "classification": "guarded_fallback_required",
            "lane": str(row["measured_site"]),
            "expected_bits": {
                field: str(evidence["cpu_bits_sha256"])
                for field, evidence in fields.items()
            },
            "observed_bits": {
                field: str(evidence["gpu_bits_sha256"])
                for field, evidence in fields.items()
            },
            "matches_expected": bool(row["bitwise_equal"]),
        }

    manifests = {
        module_key: cache.compile_manifest() for module_key, cache in caches.items()
    }
    return {
        "records": records,
        "compile_platforms": {
            module_key: manifest["compile_platform"]
            for module_key, manifest in manifests.items()
        },
        "compile_modules": {
            module_key: manifest["modules"][module_key]
            for module_key, manifest in manifests.items()
        },
        "device": {
            key: value
            for key, value in capability.as_dict().items()
            if key != "cache_directory"
        },
    }


def _encoded(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


V841_FTZ_PASS_SCHEMA = "mpas-port.cuda-ftz-v841-device-pass/v2"
V841_FTZ_TRANSCRIPT_SCHEMA = "mpas-port.cuda-ftz-v841-transcript/v2"
_FALLBACK_DISABLED_PREFIX = "#define MPAS_FTZ_FALLBACK_ENABLED 0\n"
_SHARED_MODULE_KEYS = {
    "recovery": "hexcore.cuda_backend.recovery",
    "acoustic": "hexcore.cuda_acoustic",
    "driver": "hexcore.cuda_driver",
    "horizontal": "hexcore.cuda_horizontal",
}


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def _normalized_shared_records(payload: Mapping[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for row in payload["records"].values():
        label = str(row["translation_unit"])
        module_key = _SHARED_MODULE_KEYS[label]
        kernel = str(row["kernel"])
        classification = str(row["classification"])
        records[f"{module_key}::{kernel}"] = {
            "translation_unit": module_key,
            "kernel": kernel,
            "classification": (
                "guarded_fallback_required"
                if classification == "guarded_fallback_required"
                else "fallback_invariant"
            ),
            "lane": str(row["lane"]),
            "expected_bits": row["expected_bits"],
            "observed_bits": row["observed_bits"],
            "matches_expected": bool(row["matches_expected"]),
        }
    return records


def _measurement_pass(
    *,
    fallback_disabled: bool,
    ordinal: int,
    compile_manifest: Mapping[str, Any],
    relation: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    runner_source_sha256: str,
    cache_dir: str | Path,
) -> dict[str, Any]:
    from .cuda_ftz import (
        _run_guarded_kernel_audit_once,
        _validate_v841_measurement_translation_units,
        canonical_sha256,
        v841_compiled_translation_units,
    )

    shared = _run_guarded_kernel_audit_once(
        fallback_disabled=fallback_disabled,
        transcript_module_keys=_SHARED_MODULE_KEYS,
        transcript_cache_dir=cache_dir,
    )
    release_specific = _run_v841_new_kernel_audit_once(
        fallback_disabled=fallback_disabled,
        cache_dir=cache_dir,
    )
    records = _normalized_shared_records(shared)
    overlap = set(records) & set(release_specific["records"])
    if overlap:
        raise RuntimeError(f"v8.4.1 FTZ probe inventory overlaps: {sorted(overlap)}")
    records.update(release_specific["records"])

    compiled = relation["translation_units"]
    expected = {
        f"{module_key}::{kernel}"
        for module_key, row in compiled.items()
        for kernel in row["compiled_kernel_surface"]
    }
    if set(records) != expected or len(records) != 95:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise RuntimeError(
            "v8.4.1 FTZ direct-production probe inventory differs from the "
            f"compiled surface: missing={missing}, extra={extra}"
        )

    device = shared["device"]
    if release_specific["device"] != device:
        raise RuntimeError("v8.4.1 FTZ probe device changed within one pass")
    platform_sha256 = relation["compile_platform"]["sha256"]
    compile_platforms = {
        **shared["compile_platforms"],
        **release_specific["compile_platforms"],
    }
    if set(compile_platforms) != set(compiled) or any(
        platform.get("sha256") != platform_sha256
        or canonical_sha256(platform.get("fingerprint", {})) != platform_sha256
        for platform in compile_platforms.values()
    ):
        raise RuntimeError(
            "v8.4.1 FTZ probe platform differs from the compiled executable"
        )
    translation_units = {
        **shared["compile_modules"],
        **release_specific["compile_modules"],
    }
    mode = "fallback-disabled" if fallback_disabled else "fallback-enabled"
    _validate_v841_measurement_translation_units(
        translation_units,
        mode=mode,
        compile_manifest=compile_manifest,
        relation=relation,
        compiled=v841_compiled_translation_units(),
    )

    source_binding_sha256 = canonical_sha256(source_bindings)
    record_sha256 = canonical_sha256(records)
    core = {
        "schema": V841_FTZ_PASS_SCHEMA,
        "mode": mode,
        "ordinal": ordinal,
        "compile_manifest_sha256": canonical_sha256(compile_manifest),
        "compile_platform_fingerprint_sha256": platform_sha256,
        "source_binding_sha256": source_binding_sha256,
        "runner_source_sha256": runner_source_sha256,
        "runtime": device,
        "runtime_sha256": canonical_sha256(device),
        "translation_units_sha256": canonical_sha256(translation_units),
        "translation_units": translation_units,
        "kernel_count": len(records),
        "records_sha256": record_sha256,
        "records": records,
    }
    return {**core, "pass_sha256": _sha256_json(core)}


def run_v841_guarded_kernel_subnormal_audit(
    *,
    compile_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure all 95 compiled v8.4.1 entrypoints on the live sm_120 device.

    Each arm constructs fresh device inputs and launches the production source
    directly.  The fallback-disabled arms compile the same eight sources with
    only the feature macro prefix changed.  No timing enters this transcript.
    """

    from .cuda_ftz import (
        V841_DISABLED_RECORDS_SHA256,
        V841_ENABLED_RECORDS_SHA256,
        V841_KERNEL_AUDIT_SCHEMA,
        V841_PROBE_SPEC_SHA256,
        canonical_sha256,
        v841_kernel_audit_measurement,
        validate_v841_compile_manifest_relation,
        v841_compiled_translation_units,
        v841_reached_translation_units,
    )

    relation = validate_v841_compile_manifest_relation(compile_manifest)
    compiled = v841_compiled_translation_units()
    reached = v841_reached_translation_units()
    runner_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_bindings = {
        module_key: {
            "production_source_sha256": hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest(),
            "fallback_enabled_source_sha256": hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest(),
            "fallback_disabled_source_sha256": hashlib.sha256(
                (_FALLBACK_DISABLED_PREFIX + source).encode("utf-8")
            ).hexdigest(),
            "reached_kernels": list(reached[module_key][1]),
            "compiled_kernels": list(kernels),
        }
        for module_key, (source, kernels) in compiled.items()
    }

    def fresh_measurement_pass(*, fallback_disabled: bool, ordinal: int) -> dict:
        import cupy as cp

        label = "disabled" if fallback_disabled else "enabled"
        with tempfile.TemporaryDirectory(
            prefix=f"mpas-v841-ftz-{label}-{ordinal}-"
        ) as raw_cache_dir:
            cache_dir = Path(raw_cache_dir)
            if any(cache_dir.iterdir()):
                raise RuntimeError("v8.4.1 FTZ pass cache was not born empty")
            # The outer eight-TU compile runs in this same process. CuPy's
            # public memo cache can otherwise satisfy a RawModule resolution
            # without entering NVRTC even though this pass's disk cache is
            # genuinely fresh. Clear only that process-local memo immediately
            # before the pass; the validator below still requires one exact
            # NVRTC-entry observation for every one of the eight TUs.
            cp.clear_memo()
            measured = _measurement_pass(
                fallback_disabled=fallback_disabled,
                ordinal=ordinal,
                compile_manifest=compile_manifest,
                relation=relation,
                source_bindings=source_bindings,
                runner_source_sha256=runner_source_sha256,
                cache_dir=cache_dir,
            )
            if not any(cache_dir.iterdir()):
                raise RuntimeError("v8.4.1 FTZ pass produced no fresh cache image")
            return measured

    enabled = [
        fresh_measurement_pass(
            fallback_disabled=False,
            ordinal=ordinal,
        )
        for ordinal in (1, 2)
    ]
    disabled = [
        fresh_measurement_pass(
            fallback_disabled=True,
            ordinal=ordinal,
        )
        for ordinal in (1, 2)
    ]
    if enabled[0]["records_sha256"] != enabled[1]["records_sha256"]:
        raise RuntimeError("v8.4.1 enabled FTZ device passes are not deterministic")
    if disabled[0]["records_sha256"] != disabled[1]["records_sha256"]:
        raise RuntimeError("v8.4.1 disabled FTZ device passes are not deterministic")
    if enabled[0]["translation_units"] != enabled[1]["translation_units"]:
        raise RuntimeError("v8.4.1 enabled FTZ compiled images are not deterministic")
    if disabled[0]["translation_units"] != disabled[1]["translation_units"]:
        raise RuntimeError("v8.4.1 disabled FTZ compiled images are not deterministic")
    if enabled[0]["records_sha256"] != V841_ENABLED_RECORDS_SHA256:
        raise RuntimeError("v8.4.1 enabled FTZ outcome differs from reviewed bytes")
    if disabled[0]["records_sha256"] != V841_DISABLED_RECORDS_SHA256:
        raise RuntimeError("v8.4.1 disabled FTZ outcome differs from reviewed bytes")

    production = enabled[0]["records"]
    mutation = disabled[0]["records"]
    probe_spec = {
        key: {
            "translation_unit": row["translation_unit"],
            "kernel": row["kernel"],
            "classification": row["classification"],
            "lane": row["lane"],
            "expected_bits": row["expected_bits"],
        }
        for key, row in production.items()
    }
    if canonical_sha256(probe_spec) != V841_PROBE_SPEC_SHA256:
        raise RuntimeError("v8.4.1 FTZ probe deck differs from reviewed bytes")
    kernels: dict[str, Any] = {}
    for key, candidate in production.items():
        killed = mutation[key]
        requires_red = candidate["classification"] == "guarded_fallback_required"
        mutation_red = not bool(killed["matches_expected"])
        if candidate["matches_expected"] is not True:
            raise RuntimeError(f"v8.4.1 production FTZ probe failed at {key}")
        if mutation_red is not requires_red:
            raise RuntimeError(
                f"v8.4.1 fallback mutation has the wrong disposition at {key}"
            )
        module_key, kernel = key.split("::", 1)
        source = compiled[module_key][0]
        kernels[key] = {
            "translation_unit": module_key,
            "kernel": kernel,
            "compiled_source_sha256": hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest(),
            "reached_by_admitted_step": kernel in reached[module_key][1],
            "classification": candidate["classification"],
            "lane": candidate["lane"],
            "expected_bits": candidate["expected_bits"],
            "enabled_observed_bits": candidate["observed_bits"],
            "disabled_fallback_observed_bits": killed["observed_bits"],
            "enabled_matches_expected": True,
            "disabled_fallback_matches_expected": not mutation_red,
            "mutation_red": mutation_red,
        }

    transcript_core = {
        "schema": V841_FTZ_TRANSCRIPT_SCHEMA,
        "runner_source_sha256": runner_source_sha256,
        "compile_manifest_sha256": canonical_sha256(compile_manifest),
        "compile_platform_fingerprint_sha256": relation["compile_platform"]["sha256"],
        "source_bindings": source_bindings,
        "source_binding_sha256": canonical_sha256(source_bindings),
        "probe_spec_sha256": canonical_sha256(probe_spec),
        "runtime": enabled[0]["runtime"],
        "runtime_sha256": enabled[0]["runtime_sha256"],
        "enabled_passes": enabled,
        "disabled_fallback_passes": disabled,
        "enabled_records_sha256": enabled[0]["records_sha256"],
        "disabled_fallback_records_sha256": disabled[0]["records_sha256"],
    }
    transcript = {
        **transcript_core,
        "transcript_sha256": _sha256_json(transcript_core),
    }
    audit_capability = str(
        relation["compile_platform"]["fingerprint"].get(
            "device_compute_capability"
        )
    )
    return {
        "schema": V841_KERNEL_AUDIT_SCHEMA,
        "source_release": "v8.4.1",
        "measurement": v841_kernel_audit_measurement(audit_capability),
        "device_compute_capability": audit_capability,
        "compile_manifest_sha256": canonical_sha256(compile_manifest),
        "compile_platform_fingerprint_sha256": relation["compile_platform"]["sha256"],
        "fallback_verified": True,
        "dual_run_byte_identical": True,
        "kernel_count": len(kernels),
        "kernels": kernels,
        "measurement_transcript": transcript,
        "authority_claim": False,
    }


__all__ = [
    "V841_FTZ_PASS_SCHEMA",
    "V841_FTZ_TRANSCRIPT_SCHEMA",
    "_run_v841_new_kernel_audit_once",
    "run_v841_guarded_kernel_subnormal_audit",
]
