"""One independent dry regional forecast for a runtime class mint.

Uses the same assembler and dry namelist as tools/run_cuda_regional_forecast.
Only field digests are retained. This is reproducibility evidence for the dry
boundary machinery, not a full-physics or CPU-equivalence claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

FIELDS = ("u", "w", "theta", "rho", "qv", "pressure")
STEPS = 15


def mint_steps(start_time, stop_time, dt_seconds):
    """Bound the reproducibility sample by the admitted forecast window.

    Fifteen is the sample cap, not a physical spin-up requirement. Short
    forecasts still compare every field at every step in two strict runs.
    The parent route has already required boundaries through stop_time.
    """
    start = datetime.strptime(start_time, '%Y-%m-%d_%H:%M:%S')
    stop = datetime.strptime(stop_time, '%Y-%m-%d_%H:%M:%S')
    if not math.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError('regional mint needs a finite positive timestep')
    count = (stop - start).total_seconds() / dt_seconds
    if not math.isfinite(count) or count < 1 or abs(count - round(count)) > 1e-9:
        raise ValueError(
            'regional mint window must contain a positive whole number of forecast steps; '
            'a partial step would integrate past the admitted boundary window')
    return min(STEPS, round(count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("grid", "static", "init", "lbc-dir", "start-time", "stop-time", "out"):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--dt', type=float, required=True)
    parser.add_argument('--len-disp', type=float, required=True)
    args = parser.parse_args()
    try:
        steps = mint_steps(args.start_time, args.stop_time, args.dt)
    except ValueError as error:
        parser.error(str(error))
    from woof.hex import numerics
    numerics.set_mode('strict', source='regional class mint')
    from woof.hex.config_v841 import V841DryDycoreConfig
    from woof.hex.cuda_backend.regional_admission import RegionalClassKey, kernel_set_sha256
    from woof.hex.cuda_regional_forecast_v841 import assemble_regional_driver_v841
    from woof.hex.driver import load_mpas_initial_state, load_mpas_vertical_grid
    from woof.hex.dynamics_v841 import load_v841_reference_wind_profiles
    from woof.hex.mesh import Mesh, regional_boundary_mask_digest
    from woof.hex.regional_v841 import RVORD_F32

    started = time.monotonic()
    mesh = Mesh.from_netcdf(Path(args.grid), Path(args.static), validate=False)
    vertical = load_mpas_vertical_grid(Path(args.init), mesh, allow_regional_sentinels=True)
    state, reference, saved = load_mpas_initial_state(
        Path(args.init), mesh, vertical.vertical_grid, scalar_names=('qv',),
        terrain_metrics=vertical.terrain_metrics, allow_regional_sentinels=True,
        return_saved_diagnostics=True, cold_start=True)
    config = V841DryDycoreConfig(
        config_dt=args.dt, config_split_dynamics_transport=True,
        config_dynamics_split_steps=3, config_number_of_sub_steps=6,
        config_scalar_advection=True, config_monotonic=False,
        config_positive_definite=False, config_scalar_adv_order=3,
        config_scalar_vadv_order=3, config_coef_3rd_order=0.25,
        config_apvm_upwinding=0.5, config_horiz_mixing='2d_smagorinsky',
        config_len_disp=args.len_disp, config_visc4_2dsmag=0.05,
        config_smagorinsky_coef=0.125, config_del4u_div_factor=10.0,
        config_h_ScaleWithMesh=True, config_smdiv=0.1,
        config_divergence_damping=True, config_xnutr=0.2, config_zd=22000.0,
        config_apply_lbcs=True, config_moist_physics=True)
    key = RegionalClassKey.build(
        boundary_zone_width=int(np.max(mesh.bdyMaskCell)),
        n_vert_levels=int(state.rho.shape[0]), finest_edge_m=float(np.min(mesh.dcEdge)),
        dt_seconds=args.dt, kernel_set=kernel_set_sha256())
    masks = {name: getattr(mesh, name) for name in ('bdyMaskCell', 'bdyMaskEdge', 'bdyMaskVertex')}
    driver = assemble_regional_driver_v841(
        mesh, state, vertical.vertical_grid, reference, saved, vertical.terrain_metrics, config,
        reference_wind_profiles=load_v841_reference_wind_profiles(Path(args.init)),
        lbc_paths=[str(path) for path in sorted(Path(args.lbc_dir).glob('lbc.*.nc'))],
        start_time=datetime.strptime(args.start_time, '%Y-%m-%d_%H:%M:%S'))
    runtime = driver.regional_v841
    zz = np.asarray(vertical.vertical_grid.zz, dtype=np.float32)
    frames = []

    def record(step):
        def host(array):
            return np.ascontiguousarray(driver.cp.asnumpy(runtime.history_slice(array)), dtype=np.float32)
        current = driver.atmosphere
        qv = host(current.state.scalars[0])
        fields = dict(u=host(current.saved.normal_velocity), w=host(current.saved.vertical_velocity),
                      qv=qv, theta=host(current.saved.theta_m) / (np.float32(1) + RVORD_F32 * qv),
                      rho=host(current.state.rho) * zz,
                      pressure=np.asarray(reference.pressure_base, np.float32)
                      + host(current.saved.pressure_perturbation))
        finite = {name: bool(np.isfinite(fields[name]).all()) for name in FIELDS}
        if not all(finite.values()):
            raise RuntimeError(f'non-finite regional mint at step {step}: {finite}')
        frames.append(dict(step=step, finite=finite, field_sha256={
            name: hashlib.sha256(np.ascontiguousarray(fields[name]).tobytes()).hexdigest()
            for name in FIELDS}))
        print(f'MINT step={step} finite=True', flush=True)

    record(0)
    for step in range(1, steps + 1):
        driver.atmosphere = driver.step_device().atmosphere
        record(step)
    card = driver.cp.cuda.runtime.getDeviceProperties(0)['name']
    if isinstance(card, bytes):
        card = card.decode()
    receipt = dict(schema='gpuwm-hex.regional-mint-run/v1', instrument='run_cuda_regional_mint',
                   pid=os.getpid(), numerics='strict', minted_without_anchor_gate=True,
                   key=key.as_dict(), bdy_mask_sha256=regional_boundary_mask_digest(masks),
                   n_cells=int(runtime.n_cells_solve), config_sha256=driver.configuration_sha256,
                   driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   start_time=args.start_time, stop_time=args.stop_time,
                   steps=steps, frames=frames, seconds=round(time.monotonic() - started, 3),
                   card=card)
    Path(args.out).write_text(json.dumps(receipt, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
