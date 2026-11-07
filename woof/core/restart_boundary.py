"""Stop at an existing restart alarm without changing trajectory identity."""
from fractions import Fraction


def dispatch_at_restart(seconds):
    """Use the ordinary callbacks and calendars through an interior boundary.

    The full schedule remains authoritative, including its interior feedback
    order. This neither shortens the experiment nor seals different forcing.
    """
    def dispatch(*, model, schedule, callbacks, adaptive_driver, clocks,
                 step_bindings, skip_feedback_path, start_period,
                 started_grid_ids, committed_initial_history_grid_ids):
        from woof.core.clock import execute_schedule
        ticks = Fraction(str(seconds)) * schedule.clock.tick_den
        if ticks.denominator != 1:
            raise ValueError("cycle stop must lie on the engine tick lattice")
        return execute_schedule(schedule, clocks=clocks,
            on_period_steps=adaptive_driver, start_period=start_period,
            started_grid_ids=started_grid_ids,
            committed_initial_history_grid_ids=committed_initial_history_grid_ids,
            skip_feedback_path=skip_feedback_path, stop_at_restart_ticks=int(ticks),
            **{'on_'+name: callback for name, callback in callbacks.items()})
    # The leg walk (dormant nests) runs the schedule as several legs and must
    # end at this boundary instead of starting the next leg past it.
    dispatch.stop_seconds = float(seconds)
    return dispatch
