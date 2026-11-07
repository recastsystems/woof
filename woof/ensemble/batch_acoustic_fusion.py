"""Isolated default acoustic Mu/W column fusion qualification candidate.

UV remains a separate launch and its visibility boundary remains intact.
The candidate concatenates existing column bodies without arithmetic edits.
It retains all global outputs and introduces no per-column array workspace.
Exact words and actual cache/DRAM behavior require independent device proof.
"""
from __future__ import annotations

from hashlib import sha256
from functools import lru_cache
import re

import numpy as np

from woof.ensemble.batch_kernel import (
    KernelSpec, PointerSpec, _active_source, _entry_parts, _runtime_audit_options,
    prepare_batch_source_launch,
)
from woof.ensemble.batch_state import BatchedDomainState, BatchStateUnsupported

ENTRY = "ensemble_acoustic_mu_w"
F = np.float32
I = np.int32


def _shared_column_phase(body, *, writes):
    """Mirror original stores or replace column-local reads, with closed indices."""
    from woof.ensemble.batch_kernel import _close, _masked
    masked = _masked(body)
    caches = {'th_pp_old': 'trial_old_theta', 'th_pp': 'trial_new_theta',
              'ww_pp': 'trial_eta_flux'}
    # Each closed form is a level variable times the column stride plus the
    # own column, so its cache row is that same variable. kr is the forcing
    # row advance_w_phi_msf names in its default arm (2.8.4, a11786408).
    levels = {'c': '0', 'ci': 'k', 'h': 'k', '(size_t)st+c': '1',
              '(size_t)k*st+c': 'k', '(size_t)(k+1)*st+c': '(k + 1)',
              '(size_t)kk*st+c': 'kk', '(size_t)nz*st+c': 'nz',
              '(size_t)kr*st+c': 'kr', '(size_t)(kr+1)*st+c': '(kr + 1)'}
    edits = []
    stores = {name: 0 for name in caches}
    reads = {name: 0 for name in caches}
    for match in re.finditer(r'\b(th_pp_old|th_pp|ww_pp)\s*\[', masked):
        close = _close(masked, match.end() - 1, '[', ']')
        index = ''.join(masked[match.end():close].split())
        if index not in levels:
            raise BatchStateUnsupported('fused shared column index changed; audit its level and ownership')
        access = caches[match[1]] + '[' + levels[index] + ' * blockDim.x + threadIdx.x]'
        tail = close + 1
        while tail < len(masked) and masked[tail].isspace():
            tail += 1
        is_store = masked[tail:tail + 2] == '+=' or (
            masked[tail:tail + 1] == '=' and masked[tail:tail + 2] != '==')
        if writes and is_store:
            end = masked.index(';', tail)
            edits.append((match.start(), end + 1,
                          access + ' = (' + body[match.start():end] + ');'))
            stores[match[1]] += 1
        elif not is_store and (not writes or match[1] == 'ww_pp'):
            edits.append((match.start(), close + 1, access))
            reads[match[1]] += 1
        elif writes and not is_store:
            # The snapshot reads the previous theta before the new-theta
            # cache exists. Keep this original input load and every update
            # expression's original old-value read.
            continue
        else:
            raise BatchStateUnsupported('the W phase writes an intermediate shared only with Mu; audit the changed lifetime')
    if writes and stores != {'th_pp_old': 1, 'th_pp': 2, 'ww_pp': 3}:
        raise BatchStateUnsupported('Mu intermediate store inventory changed; every observed output must remain materialized')
    for start, end, replacement in reversed(edits):
        body = body[:start] + replacement + body[end:]
    return body, {'mirrored_stores': stores, 'shared_reads': reads}


@lru_cache(maxsize=None)
def fusion_source(source, mu_spec, w_spec, audit_options, *, shared_intermediates=False):
    """Keep two audited original bodies byte-for-byte inside separate scopes.

    Each body is the preprocessor arm the compiler selects under
    ``audit_options``: inactive arms and the conditional lines themselves
    are blanked at unchanged offsets. The store inventory therefore counts
    what is compiled. Read from the raw text, a kernel that keeps a strict
    and a default body side by side in one conditional block (2.8.4's
    advance_mu_th_msf) counted every store twice and was refused although
    nothing about its compiled stores had changed.

    A local column function return is rewritten as leaving its own scope,
    so the following column body retains the original separate-launch guard.
    The supported periodic path has no specified/open frame return arm.
    """
    from woof.ensemble.batch_kernel import _close, _masked
    entries = [_entry_parts(source, spec, audit_options) for spec in (mu_spec, w_spec)]
    active = _active_source(source, audit_options)
    parameters = []
    declarations = {}
    bodies = []
    cache_receipts = []
    for ordinal, (spec, parts) in enumerate(zip((mu_spec, w_spec), entries)):
        signature, names = parts[-1], parts[-2]
        for declaration, name in zip(_masked(signature).split(","), names, strict=True):
            declaration = declaration.strip()
            if name not in declarations:
                parameters.append(name)
                declarations[name] = declaration
            elif "*" in declaration and "const" not in declaration.split():
                declarations[name] = declaration
        # The option-resolved view blanks every preprocessor line. A define,
        # undef or pragma inside a column body would vanish from the fused
        # phase while the separate launch still compiles it.
        if re.search(r"^[ \t]*#[ \t]*(?!(?:if|ifdef|ifndef|elif|else|endif)\b)",
                     _masked(source[parts[3] + 1:parts[4]]), re.M):
            raise BatchStateUnsupported("a fused column body carries a preprocessor directive other than a conditional; the option-resolved phase would drop it")
        body = active[parts[3] + 1:parts[4]]
        # A return in Mu must not suppress the separately guarded W phase.
        # do/while(false) supplies that same phase-local exit without a call.
        masked = _masked(body)
        returns = list(re.finditer(r"\breturn\s*;", masked))
        for loop in re.finditer(r"\b(?:for|while)\s*\(", masked):
            end_condition = _close(masked, loop.end() - 1, "(", ")")
            statement = end_condition + 1
            while masked[statement].isspace():
                statement += 1
            end_statement = (_close(masked, statement, "{", "}")
                             if masked[statement] == "{" else masked.index(";", statement))
            if any(statement <= item.start() <= end_statement for item in returns):
                raise BatchStateUnsupported("a column return inside a loop would exit that loop rather than the fused column phase")
        for item in reversed(returns):
            body = body[:item.start()] + "break;" + body[item.end():]
        if shared_intermediates:
            body, cache_receipt = _shared_column_phase(body, writes=ordinal == 0)
            cache_receipts.append(cache_receipt)
        bodies.append("\n    do {\n" + body + "\n    } while (false);\n")
    prefix = ''
    if shared_intermediates:
        prefix = ('\n    extern __shared__ real trial_column_cache[];\n'
                  '    real* trial_old_theta = trial_column_cache;\n'
                  '    real* trial_new_theta = trial_column_cache + nz * blockDim.x;\n'
                  '    real* trial_eta_flux = trial_column_cache + 2 * nz * blockDim.x;\n')
    appended = ('\nextern "C" __global__\nvoid ' + ENTRY + "(" +
                ", ".join(declarations[name] for name in parameters) + ")\n{\n" +
                prefix + "".join(bodies) + "}\n")
    return source + appended, tuple(parameters), {
        "original_source_sha256": sha256(source.encode()).hexdigest(),
        "column_entries": (mu_spec.entry, w_spec.entry),
        "arithmetic": ("original expressions; own-column caches mirror stores and replace loads"
                       if shared_intermediates else
                       "original column body text; phase-local return becomes break"),
        "body_view": "preprocessor arm selected by the audit options; inactive arms blanked",
        "additional_device_workspace_bytes": 0,
        "cut_behavior": "global stores retained; compiler forwarding and DRAM traffic unmeasured",
        "original_body_sha256": tuple(sha256(source[part[3] + 1:part[4]].encode()).hexdigest()
                                       for part in entries),
        "intermediate_storage": "shared" if shared_intermediates else "original_global",
        "shared_phase_inventory": cache_receipts,
    }


def prepare_acoustic_substep_launch(state, cfg, dtau, coefficients, *, mudf=None,
                                    shared_intermediates=False):
    """Prepare original UV then one candidate Mu/W kernel, default periodic dry."""
    from woof.ensemble import batch_acoustic as separate
    from woof.core import acoustic as original
    from woof import wrf_exact
    if not isinstance(state, BatchedDomainState):
        raise TypeError("isolated acoustic fusion requires an admitted BatchedDomainState")
    if state.members == 1:
        launch = separate.prepare_acoustic_substep_launch(state, cfg, dtau, coefficients, mudf=mudf)
        launch.fusion_receipt = {"effective": "original_n1", "additional_device_workspace_bytes": 0}
        return launch
    if wrf_exact.ENABLED:
        raise BatchStateUnsupported("strict Mu/W helper coordinates and floating DAG require their own fused qualification")
    if (cfg.moist or state.physics is not None or cfg.open_x or cfg.open_y
            or cfg.specified or original._boundary_forced(cfg) or mudf is not None
            or cfg.emdiv or cfg.zadvect_implicit):
        raise BatchStateUnsupported("this isolated fusion closes only periodic dry Mu/W without forcing, moisture, emdiv or implicit transport")
    if len(coefficients) != 4:
        raise BatchStateUnsupported("dry fused Mu/W requires the original four stage coefficient arrays")
    bindings = separate._Bindings(state, cfg)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    mu_old = state.scratch((ny, nx), "acoustic_mu_pp_old")
    th_old = state.scratch((nz, ny, nx), "acoustic_th_pp_old")
    c2a, a, alpha, gam = coefficients
    values = dict(state.storage.arrays)
    values.update(mu_pp_old=mu_old, th_pp_old=th_old, c2a=c2a, a=a, alpha=alpha,
                  gam=gam, w_ref=state.w, mudf=state.mup, cqu=state.p, cqv=state.p, cqw=state.p,
                  write_mudf=I(0), moist_cq=I(0), cf1=state.cf1, cf2=state.cf2, cf3=state.cf3,
                  top_lid=I(cfg.top_lid), rdx=F(1.0 / cfg.dx), rdy=F(1.0 / cfg.dy),
                  dtau=F(dtau), epssm=F(cfg.epssm),
                  dampmag=original.damp_magnitude(cfg, dtau), zdamp=F(cfg.zdamp),
                  boundary_x=I(0), boundary_y=I(0), open_x=I(0), open_y=I(0),
                  spec_zone=I(0), base3d=bindings.base3d, nz=I(nz), ny=I(ny), nx=I(nx))
    # Signature authority is the installed original CUDA source, not a copied
    # list of argument positions. Pointer ownership still comes from admission.
    from woof.core.kernels import module_source, module_source_int_defines
    defines = original.wphi_module_defines(nz)
    source = module_source_int_defines("acoustic", defines) if defines else module_source("acoustic")
    entries = ("advance_mu_th_msf", "advance_w_phi_msf") if state.has_msf else ("advance_mu_th", "advance_w_phi")

    def source_spec(entry):
        from woof.ensemble.batch_kernel import _close, _masked
        # Read the signature from the compiled view (inactive preprocessor
        # branches blanked, e.g. the strict ww_ref parameter), not the raw
        # text: a signature whose strict and default arms each close their
        # own parenthesis (advance_uv) would otherwise end inside the strict
        # arm and declare its pointers.
        active = _active_source(source, _runtime_audit_options(
            KernelSpec("acoustic", entry, ())))
        masked = _masked(active)
        declaration = re.search(r"\bvoid\s+" + entry + r"\s*\(", masked)
        end = _close(masked, declaration.end() - 1, "(", ")")
        signature = active[declaration.end():end]
        pointers = []
        for part in signature.split(","):
            if "*" not in part:
                continue
            name = part.split()[-1].lstrip("*")
            _, spec = bindings.owner(values[name])
            pointers.append(PointerSpec(name, spec.ownership, spec.dtype))
        return KernelSpec("acoustic", entry, tuple(pointers))

    mu_spec, w_spec = (source_spec(entry) for entry in entries)
    options = _runtime_audit_options(mu_spec)
    fused, names, receipt = fusion_source(source, mu_spec, w_spec, options,
                                         shared_intermediates=shared_intermediates)
    # The member-batched fused entry compiles generic (runtime nz/ny/nx), as
    # qualified in 3e2d23212e. 248b9f57e9 specialized it with the shape
    # constants; under default (fast) numerics that compile contracts the
    # mapped Mu/W column differently from the scalar route, so members
    # differed from single runs by 1-4 ULP in w_pp/u_pp (40 reds in
    # test_ensemble_batch_acoustic_fusion*_gpu). The scalar route keeps its
    # own specialization; only word identity with it decides this text.
    pointer_names = []
    for spec in (mu_spec, w_spec):
        for pointer in spec.pointers:
            if pointer.name not in pointer_names:
                pointer_names.append(pointer.name)
    pointers = []
    strides = {}
    for name in pointer_names:
        owner, spec = bindings.owner(values[name])
        pointers.append(PointerSpec(name, spec.ownership, spec.dtype))
        strides[name] = state.storage.pointer_stride_bytes(owner)
    fused_spec = KernelSpec("acoustic", ENTRY, tuple(pointers))
    threads = 64 if shared_intermediates else 256
    shared_bytes = (3 * nz + 1) * threads * np.dtype('float32').itemsize if shared_intermediates else 0
    if shared_bytes:
        import cupy as cp
        limit = int(cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['sharedMemPerBlock'])
        if shared_bytes > limit:
            raise BatchStateUnsupported('the joined column cache exceeds the device per-block shared-memory capacity')
    columns = ((ny * nx + threads - 1) // threads,)
    column = prepare_batch_source_launch(fused, fused_spec, state.members, columns, (threads,),
                                         tuple(values[name] for name in names), pointer_strides=strides,
                                         shared_mem=shared_bytes)
    uv_spec = source_spec("advance_uv")
    uv_names = _entry_parts(source, uv_spec, _runtime_audit_options(uv_spec))[-2]
    uv_arrays = [(name, values[name]) for name in uv_names if name in {p.name for p in uv_spec.pointers}]
    uv_grid = ((nz * (ny + 1) * (nx + 1) + 255) // 256,)
    values["smdiv"] = F(0.0)
    first = bindings.bind("advance_uv", tuple(values[name] for name in uv_names), uv_arrays, uv_grid)
    values["smdiv"] = F(cfg.smdiv)
    later = bindings.bind("advance_uv", tuple(values[name] for name in uv_names), uv_arrays, uv_grid)

    def launch(*, first):
        (uv_first if first else uv_later)()
        column()

    uv_first, uv_later = first, later
    launch.fusion_receipt = {**receipt, "effective": "default_mu_w_concatenation",
                             "members": state.members, "architecture_options": options,
                             "column_threads": threads, "dynamic_shared_bytes_per_block": shared_bytes,
                             "binding": dict(column.binding_receipt)}
    launch.numerical_entries = ("advance_uv", ENTRY)
    launch.column_launch = column
    from inspect import getclosurevars
    raw = getclosurevars(column).nonlocals.get("kernel")
    launch.compiled_attributes = dict(getattr(raw, "attributes", {}))
    return launch
