"""Run the production NSSL coordinator on the oracle's identical inputs.

Admission is checked before importing CuPy. Each unported mode produces an
explicit refusal receipt and causes a nonzero exit. No neighboring mode runs.
The per-step tendency is the FP32 net state change divided by the step, not an
instantaneous process rate. All prognostics and diagnostics are saved each step.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases
from run_oracle import OUTPUT_FIELDS, SURFACE_FIELDS


def admit(mode):
    from woof.config import RunConfig, validate_run_config
    from woof.core.nssl2_contract import (
        WRF_NAMELIST_DEFAULTS, require_ported_nssl2_mode,
        resolve_nssl2_mode_for_config,
    )
    names = {f.name for f in fields(RunConfig)}
    supplied = {k: v for k, v in mode.selectors.items()
                if k in names or v != WRF_NAMELIST_DEFAULTS[k]}
    cfg = RunConfig(nx=6, ny=1, nz=cases.NZ, dx=1000.0, dy=1000.0,
                    ztop=cases.NZ * cases.DZ, dt=6.0, run_seconds=18.0,
                    moist=True, mp_physics=18, **supplied)
    validate_run_config(cfg)
    return require_ported_nssl2_mode(resolve_nssl2_mode_for_config(cfg))


def run_mode(mode, resolved, nsteps, dt, it0, *, input_case=None, surface_initial=None,
             diagnostic_initial=None):
    if not resolved.two_moment:
        from run_one_moment import run_mode as run_one
        return run_one(mode,resolved,nsteps,dt,it0,input_case=input_case,
                       surface_initial=surface_initial,diagnostic_initial=diagnostic_initial)
    import cupy as cp
    from woof.core.nssl2_fused_gs import launch_fused_gs
    from woof.core.nssl2_nucond import launch_nucond
    from woof.core.nssl2_driver_support import NSSL2DriverWorkspace
    from woof.core.nssl2_production_coordinator import (
        NSSL2PrecipitationFields, NSSL2ProductionHooks, NSSL2RegistryFields,
        run_nssl2_production_step,
    )
    from woof.core.nssl2_shape import resolved_shapes
    coefficients = resolved_shapes(mode.alphah, mode.alphahl, mode.icdx, mode.icdxhl)
    state, env = cases.build_case(mode) if input_case is None else input_case
    nx, nz = state['theta'].shape
    shape = (nz, 1, nx)

    def cell(a):
        return cp.asarray(np.ascontiguousarray(a.T[:, None, :]))

    device = {name: cell(a) for name, a in state.items()}
    rho, pressure, exner, dz = (cell(env[k]) for k in ('rho', 'pressure', 'exner', 'dz'))
    # WRF uses w(nz) at the top mass level. The extra interface is unused.
    w = cell(np.concatenate((env['w'], env['w'][:, -1:]), axis=1))
    theta = device['theta']
    temperature, primary_ice, supersaturation = (cp.empty(shape, cp.float32) for _ in range(3))
    diagnostic = {n: (cp.full(shape, -999.0, cp.float32) if diagnostic_initial is None
                      else cell(diagnostic_initial[n]))
                  for n in ('refl_10cm', 'effc_m', 'effi_m', 'effs_m')}
    surface = {n: (cp.zeros((1, nx), cp.float32) if surface_initial is None
                   else cp.asarray(surface_initial[n][None, :])) for n in SURFACE_FIELDS}
    registry = NSSL2RegistryFields(**{f.name: device[f.name] for f in fields(NSSL2RegistryFields)})
    workspace = NSSL2DriverWorkspace(
        state=cp.empty((16, *shape), cp.float32),
        category_surface_export=cp.empty((5, 1, nx), cp.float32),
        shape=shape, ignored_accumulator=cp.empty((1, nx), cp.float32),
        drag_cd=cp.empty((7,nz+1,1,nx),cp.float32) if len(coefficients)>2 and (coefficients[2]>6 or coefficients[3]==2 or coefficients[3]>6) else None)

    def fused(workspace):
        launch_fused_gs(workspace, theta, rho, pressure, exner, w,
                        temperature, primary_ice, dz, dt, hail_on=resolved.hail,
                        loaded_vapor=registry.qv, shapes=coefficients)

    def condensation(workspace):
        launch_nucond(theta, rho, pressure, exner, w,
                      *(workspace.field(n) for n in (
                          'qv', 'qc', 'qr', 'qi', 'qs', 'qndrop', 'qnr',
                          'qni', 'qns', 'qnn')),
                      dt, supersaturation_scratch=supersaturation,
                      concentration_space=True, predicted_ccn=resolved.predicted_ccn,
                      nssl_cccn=mode.cccn, shapes=coefficients, validate_values=False,
                      frozen=tuple(workspace.field(n) for n in
                                   ('qg', 'qh', 'qng', 'qnh', 'qvolg', 'qvolh')))
        cp.multiply(theta, exner, out=temperature)

    hooks = NSSL2ProductionHooks(fused_gs=fused, nucond_qvexcess=condensation,
                                moist_physics_finish=lambda workspace: None)
    trajectory = {n: [] for n in (*OUTPUT_FIELDS, *SURFACE_FIELDS)}
    for step in range(nsteps):
        cp.multiply(theta, exner, out=temperature)
        run_nssl2_production_step(
            rho, dz, registry, NSSL2PrecipitationFields(**surface), hooks, dt,
            first_step=(it0 + step == 1), output_due=True,
            temperature_k=temperature, refl_10cm=diagnostic['refl_10cm'],
            radiation_due=bool(resolved.radiation_fields),
            re_cloud_m=diagnostic['effc_m'], re_ice_m=diagnostic['effi_m'],
            re_snow_m=diagnostic['effs_m'], validate_values=False, mode=resolved, nssl_cccn=mode.cccn,
            workspace=workspace, shapes=coefficients,
        )
        cp.cuda.Stream.null.synchronize()
        for name in OUTPUT_FIELDS:
            value = diagnostic[name] if name in diagnostic else device[name]
            trajectory[name].append(cp.asnumpy(value)[:, 0, :].T.copy())
        for name in SURFACE_FIELDS:
            trajectory[name].append(cp.asnumpy(surface[name])[0].copy())
    out = {n: np.stack(v) for n, v in trajectory.items()}
    for name in cases.STATE_FIELDS:
        previous = np.concatenate((state[name][None, ...], out[name][:-1]), axis=0)
        out[f'tendency_{name}'] = (out[name] - previous) / np.float32(dt)
    input_hashes = {n: hashlib.sha256(a.tobytes()).hexdigest()
                    for n, a in {**state, **env}.items()}
    return out, input_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out_dir')
    parser.add_argument('--modes', default='')
    parser.add_argument('--nsteps', type=int, default=3)
    parser.add_argument('--dt', type=float, default=6.0)
    parser.add_argument('--itimestep0', type=int, default=2)
    parser.add_argument('--admission-only', action='store_true')
    args = parser.parse_args()
    labels = args.modes.split(',') if args.modes else list(cases.MODE_BY_LABEL)
    if args.nsteps < 1 or not np.isfinite(args.dt) or args.dt <= 0:
        parser.error('nsteps and finite dt must be positive')
    target = Path(args.out_dir)
    target.mkdir(parents=True, exist_ok=True)
    receipt = {'arm': 'strict' if os.environ.get('GPUWM_WRF_EXACT') == '1' else 'default',
               'settings': {'nsteps': args.nsteps, 'dt': args.dt, 'itimestep0': args.itimestep0},
               'admission_only': args.admission_only, 'modes': {}}
    status = 0
    for label in labels:
        mode = cases.MODE_BY_LABEL[label]
        try:
            resolved = admit(mode)
        except (TypeError, ValueError) as exc:
            reason = f'{type(exc).__name__}: {exc}'
            receipt['modes'][label] = {'refused': reason, 'selectors': mode.selectors}
            if not args.admission_only:
                np.savez(target / f'{label}.npz', refused=np.array(reason))
            print(label, 'REFUSED', reason, flush=True)
            status = 1
            continue
        receipt['modes'][label] = {'admitted': True, 'selectors': mode.selectors}
        if args.admission_only:
            continue
        outputs, hashes = run_mode(mode, resolved, args.nsteps, args.dt, args.itimestep0)
        np.savez(target / f'{label}.npz', settings=np.array([args.nsteps, args.itimestep0], np.int64),
                 dt=np.float32(args.dt), mode_label=np.array(label),
                 input_sha256=np.array(json.dumps(hashes, sort_keys=True)),
                 **{f'gpu_{n}': a for n, a in outputs.items()})
        receipt['modes'][label]['input_sha256'] = hashes
        receipt['modes'][label]['finite'] = all(np.isfinite(a).all() for a in outputs.values())
        receipt['modes'][label]['output_sha256'] = {
            n: hashlib.sha256(a.tobytes()).hexdigest() for n, a in outputs.items()}
        print(label, 'RAN', flush=True)
    from woof.certify.kernel_manifest import kernel_manifest
    receipt['kernel_manifest'] = kernel_manifest()
    (target / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
