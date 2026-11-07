"""TC receipt, assembled at the forecast's existing completion boundary."""


def event_cost(meter, *, steps):
    """Never turn incomplete sampling into a zero-cost claim."""
    complete = meter.sampled + meter.unsampled == meter.calls
    valid = complete and (meter.calls == 0 or (meter.sampled > 0 and steps > 0))
    factor = meter.calls / meter.sampled if meter.sampled else 0.
    return dict(cost_measurement_valid=valid,
        cost_measurement_error=None if valid else 'unclosed or entirely unsampled exchange calls',
        exchange_calls=meter.calls,
        sampled_calls=meter.sampled, unsampled_calls=meter.unsampled,
        exchange_and_waves_gpu_ms=meter.before_ms*factor if valid else None,
        spray_gpu_ms=meter.advance_ms*factor if valid else None,
        added_surface_gpu_ms_per_step=meter.total_ms*factor/max(steps, 1) if valid else None)


def restore_cost(physics, meter):
    """Carry pending events and cumulative timings across a nest rebuild.

    The meter contains no spatial fields. Keep its identity without harvesting
    or synchronizing; newly allocated surface diagnostics retain the new grid.
    """
    surface = getattr(physics, "tc_surface", None)
    if surface is not None and meter is not None:
        surface.cost = meter


def receipt(physics, *, steps):
    surface = getattr(physics, 'tc_surface', None)
    if surface is None:
        return None
    import cupy as cp
    meter = surface.cost.finish()
    flags = surface.spray_diag['flags']
    return dict(schema='gpuwm-tc.v1',
        curve=surface.cfg.tc_flux_curve, spray=surface.cfg.tc_spray,
        wave_source=surface.cfg.tc_wave_source,
        wave_roughness=surface.cfg.tc_wave_roughness,
        fetch_m=surface.cfg.tc_fetch_m,
        **event_cost(meter, steps=steps),
        measurement='Nonblocking CUDA event ring; final harvest at forecast completion. '
                    'Unsampled calls estimated from sampled mean. Intervals bracket launches '
                    'and include stream gaps between enqueues; not isolated kernel times. '
                    'CPU dispatch is not separately timed; horizontal mixing is excluded. '
                    'Not a counterfactual wall-time delta.',
        final_frame_spray_increments={name: {
            'minimum_w_m2': float(cp.min(surface.spray_diag[name]).get()),
            'maximum_w_m2': float(cp.max(surface.spray_diag[name]).get()),
            'mean_w_m2': float(cp.mean(surface.spray_diag[name]).get()),
            'nonzero_cells': int(cp.count_nonzero(surface.spray_diag[name]).get())}
            for name in ('sensible_increment', 'latent_increment')},
        final_frame_spray_flags={str(flag): int(cp.count_nonzero(flags==flag).get())
                                 for flag in (0,1,2,4,8,16)},
        spray_flag_meanings={'1':'outside evaporation/positive enthalpy domain',
            '2':'invalid transfer geometry or negative spray enthalpy',
            '4':'condensation branch unsupported by positive-relaxation equations',
            '8':'nonfinite feedback iteration', '16':'feedback failed to converge'},
        scope='Flags describe the final surface call only, including dry cells in flag zero.')
