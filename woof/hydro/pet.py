"""Prepared GPU port of NOAA-OWP's Apache-2.0 PET module.

Python binds resident arrays and compiles once. All per-catchment arithmetic and
native mutable parameter defaults live in CUDA. Native BMI remains a referee.
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

PET_PIN = "6c410f2a8bc843043ad25e13239928031b8c5e64"
PARAMETER_NAMES = (
    "wind_speed_measurement_height_m", "humidity_measurement_height_m",
    "vegetation_height_m", "zero_plane_displacement_height_m",
    "momentum_transfer_roughness_length_m", "heat_transfer_roughness_length_m",
    "surface_longwave_emissivity", "surface_shortwave_albedo",
    "canopy_resistance_sec_per_m", "water_temperature_C",
    "ground_heat_flux_W_per_sq_m", "surface_skin_temperature_C",
)
_INPUTS = ("T2D", "Q2D", "U2D", "V2D", "PSFC", "LWDOWN", "SWDOWN")


class PreparedPET:
    """One native-precision PET update per catchment on the parent stream.

    ``parameters`` is a resident float64 (catchment, 12) array in
    PARAMETER_NAMES order. It is mutated in place exactly as native run_pet.
    ``forcing`` accepts canonical float32 fields, or native float64 BMI fields
    for replay without narrowing the referee inputs. All fields share dtype.
    An identity must identify catchment order, parameters and forcing origin.
    Sticky status bits: 1 means invalid forcing; 2 a nonfinite numerical result.
    A later healthy update cannot erase a fault before publication checks.
    Inspect status only at output/checkpoint boundaries, never per step.
    """

    def __init__(self, forcing, parameters, *, method=5, identity, device_id=None):
        import cupy as cp

        if not identity:
            raise ValueError("PET requires catchment/parameter/forcing identity")
        selected = tuple(forcing[key] for key in _INPUTS)
        if not all(isinstance(a, cp.ndarray) for a in (*selected, parameters)):
            raise TypeError("PET inputs must already be resident CUDA arrays")
        dtype = selected[0].dtype
        n = selected[0].size
        device = selected[0].device.id
        if n == 0 or dtype not in (cp.dtype("float32"), cp.dtype("float64")):
            raise ValueError("PET needs nonempty float32 or float64 forcing")
        if device_id is not None and device != device_id:
            raise ValueError("PET must use the parent device")
        for a in selected:
            if a.shape != (n,) or a.dtype != dtype or not a.flags.c_contiguous or a.device.id != device:
                raise ValueError("PET fields must be contiguous same-card vectors with one dtype")
        if parameters.shape != (n, 12) or parameters.dtype != cp.float64 or not parameters.flags.c_contiguous or parameters.device.id != device:
            raise ValueError("PET parameters must be contiguous same-card float64 (n,12)")
        with cp.cuda.Device(device):
            if isinstance(method, int):
                if method not in (1, 2, 3, 4, 5):
                    raise ValueError("PET method must be 1..5")
                methods = cp.full(n, method, dtype=cp.int32)
                method_identity = method
            else:
                methods = method
                if not isinstance(methods, cp.ndarray) or methods.shape != (n,) or methods.dtype != cp.int32 or not methods.flags.c_contiguous or methods.device.id != device:
                    raise ValueError("PET methods must be same-card contiguous int32 (n,)")
                if not bool(((methods >= 1) & (methods <= 5)).all().item()):
                    raise ValueError("PET method must be 1..5")
                method_identity = "resident per-catchment table; values sealed in identity"
            valid = cp.isfinite(parameters).all()
            valid &= (parameters[:, 0] > 0).all()
            valid &= (parameters[:, 1] == 2).all()
            valid &= (parameters[:, 3] > 0).all()
            if not bool(valid.item()):
                raise ValueError("PET requires finite parameters, positive wind/displacement heights and 2m humidity")
            from woof import numerics
            here = Path(__file__).parent
            source = (here / "pet.cu").read_text().replace(
                '#include "pet_numeric.cuh"', (here / "pet_numeric.cuh").read_text())
            # The saturation-vapour-pressure exponential is cancellation
            # sensitive in PET; use the existing MIT Arm implementation in
            # both modes. Strict proof additionally selects native log/pow.
            math_header = (here.parent / "core/kernels/glibc_flt64.cuh").read_text()
            source = math_header + ("\n#define PET_STRICT 1\n" if numerics.is_strict() else "\n") + source
            scalar_type = "float" if dtype == cp.float32 else "double"
            from woof.certify.kernel_manifest import record_module
            options = numerics.effective_options(("--std=c++17", "--fmad=false", f"-DFORCING_TYPE={scalar_type}"))
            self._module = cp.RawModule(code=source, options=options)
            record_module("nextgen_pet", source=source, options=options, module=self._module)
            self._kernel = self._module.get_function("catchment_pet")
            output = cp.empty(n, dtype=cp.float64)
            diagnostics = cp.empty((n, 6), dtype=cp.float64)
            status = cp.zeros(n, dtype=cp.int32)
        self.fields = MappingProxyType({"potential_et_m_s": output})
        self.arrays = MappingProxyType({
            **{f"forcing/{key}": value for key, value in zip(_INPUTS, selected)},
            "parameters": parameters, "methods": methods, "potential_et_m_s": output,
            "diagnostics": diagnostics, "status": status,
        })
        self.status_arrays = MappingProxyType({"status": status})
        self.restart_identity = {
            "algorithm": "noaa-owp-pet-cuda-v2", "source_pin": PET_PIN,
            "wind_norm": "Borges-2019-4.2-balanced-unfused-separated-fused-v1",
            "method": method_identity, "arithmetic": "float64", "numerics": numerics.mode(),
            "libm": "WOOF-ArWen MIT Arm glibc-word adaptation" if numerics.is_strict() else "MIT Arm exp; CUDA libdevice log/pow",
            "forcing_dtype": str(dtype), "identity": identity,
        }
        # Python ints are explicitly sized for the CUDA ABI without NumPy.
        import numpy as np
        self._arguments = (np.int64(n), methods, *selected,
                           parameters, output, diagnostics, status)
        self._grid = ((n + 127) // 128,)

    def advance(self, dt_seconds, *, stream=None):
        """Enqueue the bound native PET update; output units are m/s."""
        self._kernel(self._grid, (128,), self._arguments, stream=stream)


def prepare(*, settings, options, state, physics, device_id, mode, components):
    """Realization factory binding a prior forcing stage and explicit PET parameters.

    Supply twelve named scalars and one method for a homogeneous population,
    or catchment_parameters rows with catchment_id/method/parameters in exact
    forcing order. Method selection stays resident. No defaults are guessed.
    """
    import cupy as cp
    source=components[options["forcing_component"]]
    forcing_identity=source.restart_identity
    binding=forcing_identity.get("identity", {})
    ids=forcing_identity.get("catchment_ids", binding.get("catchment_ids"))
    origin=forcing_identity.get("time_origin", binding.get("time_origin"))
    n=source.fields["T2D"].size
    rows=options.get("catchment_parameters")
    if rows is not None:
        if ids is None or len(rows)!=n or [row["catchment_id"] for row in rows]!=list(ids):
            raise ValueError("PET parameter rows must follow the forcing catchment order")
        if "parameters" in options or "method" in options:
            raise ValueError("select catchment parameter rows or a single broadcast row")
        if any(type(row["method"]) is not int or row["method"] not in (1,2,3,4,5) for row in rows):
            raise ValueError("PET methods must be integer values 1..5")
        configured=[row["parameters"] for row in rows]
    else:
        configured=[options["parameters"]]
    if any(set(row)!=set(PARAMETER_NAMES) for row in configured):
        raise ValueError("PET parameters must name all twelve PARAMETER_NAMES")
    with cp.cuda.Device(device_id):
        table=cp.asarray([[row[key] for key in PARAMETER_NAMES] for row in configured],dtype=cp.float64)
        parameters=cp.broadcast_to(table,(n,12)).copy()
        method=cp.asarray([row["method"] for row in rows],dtype=cp.int32) if rows is not None else options.get("method",5)
    # CFE's setup admission checks this direct order/epoch binding. Keep the
    # complete forcing provenance too; nesting it alone hid wrong-row/wrong-
    # epoch PET from those checks even when the resident extents matched.
    identity={"forcing":forcing_identity,
              "configuration":rows if rows is not None else options}
    if ids is not None:
        identity["catchment_ids"]=list(ids)
    if origin is not None:
        identity["time_origin"]=origin
    return PreparedPET(source.fields,parameters,method=method,
        device_id=device_id,identity=identity)


def prepare_memory(*, settings, options, context, components):
    from .runoff_memory import pet_memory
    return pet_memory(settings=settings, options=options, context=context, components=components)
