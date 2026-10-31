"""Split the pinned NSSL driver at its native routine boundaries.

The column wrapper has one horizontal row. Preserve its local arrays in the
existing device arena and its scalars between launches. The original driver
remains available to the native replay witness; no field expression changes.
"""

import re


def _block(source, start):
    """Return the braces of a generated C++ block, whose comments have no braces."""
    opening = source.index("{", start)
    depth = 1
    closing = opening + 1
    while depth:
        depth += (source[closing] == "{") - (source[closing] == "}")
        closing += 1
    return opening, closing - 1


def split_driver(module, entry):
    """Lower native F:1995-3031 into setup, sediment, collect, condense, finish."""
    start = module.index("__device__ __noinline__ void f_nssl_2mom_driver(")
    opening, closing = _block(module, start)
    signature = module[start:opening]
    body = module[opening + 1:closing]
    executable = body.index("// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_mp_nssl_2mom.F:2208")
    declarations = body[:executable]
    scalar_pattern = re.compile(r"^(bool|float|double|int) (\w+)(?: = ([^;]+))?;$", re.M)
    scalars = list(scalar_pattern.finditer(declarations))
    assert len(scalars) > 50
    fields = "\n".join(f"{m[1]} {m[2]};" for m in scalars)
    declarations = scalar_pattern.sub(
        lambda m: f"{m[1]} {m[2]} = Phase == 0 ? {m[3] or m[1] + '{}' } : saved.{m[2]};",
        declarations,
    )
    save = "\n".join(f"saved.{m[2]} = {m[2]};" for m in scalars)
    # These two loops alone enclose the native routine boundaries. Their
    # generated labels identify the pinned loops, not a case or mesh size.
    row = body.index("for (int nssl_first_15=")
    ro, rc = _block(body, row)
    loop = body.index("for (int nssl_first_21=")
    lo, lc = _block(body, loop)
    assert row < loop < lc < rc
    collection = body.index("// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_mp_nssl_2mom.F:2757")
    condensation = body.index("// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_mp_nssl_2mom.F:2790")
    assert lo < collection < condensation < lc
    phases = (
        body[executable:row] + "jy = jts;\n" + body[ro + 1:loop],
        "loopcnt = substep;\n" + body[lo + 1:collection],
        body[collection:condensation],
        body[condensation:lc],
        "loopcnt = loopmax + 1;\n" + body[lc + 1:rc] + "jy = jte + 1;\n" + body[rc + 1:],
    )
    assert phases[4].count("return;") == 1
    phases = (*phases[:4], phases[4].replace("return;", ""))
    signature = signature.replace("__device__ __noinline__ void f_nssl_2mom_driver(",
        "template<int Phase> __device__ __forceinline__ void f_nssl_2mom_phase(NsslDriverState& saved, int substep, ")
    phased = signature + "{\n" + declarations + "\n".join(
        f"if constexpr (Phase == {i}) {{\n{part}\n}}" for i, part in enumerate(phases)
    ) + "\n" + save + "\n}\n"
    structs = "struct NsslDriverState {\n" + fields + "\n};\n"
    structs += "struct NsslPhaseState { NsslState globals; NsslDriverState driver; };\n"
    module = module.replace("struct NsslColumn : NsslState {", structs + "struct NsslColumn : NsslState {", 1)
    # Insert alongside the original driver, which is also the CPU witness.
    start = module.index("__device__ __noinline__ void f_nssl_2mom_driver(")
    module = module[:start] + phased + module[start:]

    start = entry.index('extern "C" __device__ void nssl_step_body(')
    opening, closing = _block(entry, start)
    step = entry[start:closing + 1]
    step = step.replace('extern "C" __device__ void nssl_step_body(',
        'template<int Phase> __device__ __forceinline__ void nssl_phase_body(int substep,NsslPhaseState* contexts,')
    old = "column.copy_globals(*reinterpret_cast<const NsslState*>(parameters));"
    new = "NsslPhaseState& context = contexts[workspace_column]; column.copy_globals(Phase == 0 ? *reinterpret_cast<const NsslState*>(parameters) : context.globals); if constexpr(Phase != 0) column.native_error=context.globals.native_error;"
    assert step.count(old) == 1
    step = step.replace(old, new)
    step = step.replace("column.f_nssl_2mom_driver(", "column.f_nssl_2mom_phase<Phase>(context.driver,substep,")
    step = step.replace("errors[active]=column.native_error;", "context.globals=column; errors[active]=column.native_error;")
    kernel_start = entry.index('extern "C" __global__ void nssl_step(')
    _, kernel_end = _block(entry, kernel_start)
    kernel = entry[kernel_start:kernel_end + 1]
    kernels = []
    for phase, name in enumerate(("setup", "sediment", "collect", "condense", "finish")):
        current = kernel.replace("void nssl_step(", f"void nssl_{name}(int substep,NsslPhaseState* contexts,")
        current = current.replace("nssl_step_body(", f"nssl_phase_body<{phase}>(substep,contexts,")
        kernels.append(current)
    entry += "\n" + step + "\n" + "\n".join(kernels)
    entry = entry.replace("result[0]=sizeof(NsslState);", "result[0]=sizeof(NsslState);result[1]=sizeof(NsslPhaseState);")
    return module, entry
