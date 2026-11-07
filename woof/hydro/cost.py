"""Per-step hydro cost in every forecast receipt, measured without slowing it.

Owner, 2026-10-18: hydro is default on because it is cheap; the per-step cost is
measured and reported with parallelism named in every forecast receipt, and it
stays under 5 percent of the atmosphere step or it is a defect.

Measurement: CUDA events recorded on the hydro launch stream around the
driver's two hooks (``before_lsm`` and ``advance``), so the LSM between them is
excluded. Events are created once at attach, recorded in a fixed ring and
harvested with a non-blocking query; nothing here synchronises the device in
the step loop (profiling never slows the run). The final
harvest runs after the forecast's own end-of-run device synchronisation.
"""
from __future__ import annotations

from collections import deque

SCHEMA = "gpuwm-hydro-cost.v1"
DEFECT = "hydro-cost-over-budget"


class HydroCostMeter:
    def __init__(self, *, ring=256, events=None, elapsed=None):
        if events is None or elapsed is None:
            import cupy as cp
            events = events or cp.cuda.Event
            elapsed = elapsed or cp.cuda.get_elapsed_time
        self._elapsed = elapsed
        # One slot = four events: before_lsm start/end, advance start/end.
        self._slots = [tuple(events() for _ in range(4)) for _ in range(int(ring))]
        self._free = list(range(len(self._slots)))
        self._pending = deque()
        self._current = None
        self.calls = 0
        self.boundary_flush_calls = 0
        self.sampled = 0
        self.unsampled = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.before_ms = 0.0
        self.advance_ms = 0.0

    def _record(self, index, stream):
        if stream is None:
            import cupy as cp
            stream = cp.cuda.get_current_stream()
        self._slots[self._current][index].record(stream)

    def before_lsm_start(self, stream=None):
        self.calls += 1
        if not self._free:
            self._harvest(final=False)
        if not self._free:
            self._current = None
            self.unsampled += 1
            return
        self._current = self._free.pop()
        self._record(0, stream)

    def before_lsm_end(self, stream=None):
        if self._current is not None:
            self._record(1, stream)

    def advance_start(self, stream=None):
        if self._current is not None:
            self._record(2, stream)

    def advance_end(self, stream=None):
        if self._current is not None:
            self._record(3, stream)
            self._pending.append(self._current)
            self._current = None

    def _harvest(self, *, final):
        while self._pending:
            index = self._pending[0]
            e0, e1, e2, e3 = self._slots[index]
            if final:
                e3.synchronize()
            elif not e3.done:
                break
            before = self._elapsed(e0, e1)
            advance = self._elapsed(e2, e3)
            self.before_ms += before
            self.advance_ms += advance
            self.total_ms += before + advance
            self.max_ms = max(self.max_ms, before + advance)
            self.sampled += 1
            self._pending.popleft()
            self._free.append(index)

    def finish(self):
        self._harvest(final=True)
        return self


def attach(physics):
    """Bind a meter to an attached resident driver at preparation."""
    if getattr(physics, "hydro", None) is None:
        return None
    physics.hydro_cost = HydroCostMeter()
    return physics.hydro_cost


def receipt(physics, *, forecast_seconds, atmosphere_steps, budget=0.05, settings=None):
    """The hydro cost block for a forecast receipt (call after device sync)."""
    driver = getattr(physics, "hydro", None)
    meter = getattr(physics, "hydro_cost", None)
    if driver is None or meter is None:
        return {"schema": SCHEMA, "status": "not-run",
                "reason": "hydro was not attached to this forecast (see hydro_activation)"}
    meter.finish()
    calls, sampled = meter.calls, meter.sampled
    mean_call_ms = meter.total_ms / sampled if sampled else 0.0
    hydro_ms = meter.total_ms + mean_call_ms * meter.unsampled
    forecast_ms = float(forecast_seconds) * 1000.0
    steps = max(int(atmosphere_steps), 1)
    atmosphere_ms = max(forecast_ms - hydro_ms, 0.0)
    atmosphere_step_ms = atmosphere_ms / steps
    hydro_per_atmosphere_step_ms = hydro_ms / steps
    fraction = (hydro_per_atmosphere_step_ms / atmosphere_step_ms
                if atmosphere_step_ms > 0 else float("inf") if hydro_ms > 0 else 0.0)
    status = "PASS" if fraction <= budget else "DEFECT"
    block = {
        "schema": SCHEMA, "status": status,
        "memory_admission": getattr(driver, "memory_admission", None),
        "budget_fraction_of_atmosphere_step": budget,
        "fraction_of_atmosphere_step": fraction,
        "hydro_ms_per_atmosphere_step": hydro_per_atmosphere_step_ms,
        "atmosphere_ms_per_step_excluding_hydro": atmosphere_step_ms,
        "hydro_ms_per_lsm_call_mean": mean_call_ms,
        "hydro_ms_per_lsm_call_max": meter.max_ms,
        "hydro_ms_before_lsm_total": meter.before_ms,
        "hydro_ms_advance_total": meter.advance_ms,
        "hydro_ms_total": hydro_ms,
        "lsm_calls": calls - meter.boundary_flush_calls, "measured_hook_pairs": calls,
        "boundary_flush_calls": meter.boundary_flush_calls,
        "sampled_calls": sampled, "unsampled_calls": meter.unsampled,
        "routing_clock": getattr(driver, 'routing_clock', None),
        "parent_elapsed_seconds": driver.elapsed_seconds,
        "atmosphere_steps": int(atmosphere_steps),
        "forecast_wall_ms": forecast_ms,
        "interval_seconds": float(driver.interval_seconds),
        "initialization": {name: component.initialization_receipt
                           for name, component in getattr(driver, "components", {}).items()
                           if hasattr(component, "initialization_receipt")},
        "parallelism": [{"stage": stage.name, "parallelism": stage.parallelism,
                         "cadence_lsm_calls": getattr(stage, "cadence_steps", 1),
                         "interval_seconds": getattr(stage, "interval_seconds", None) or driver.interval_seconds,
                         "launches": (driver.stage_launches[stage.name] if getattr(driver, 'routing_clock', None)
                                      else calls // getattr(stage, "cadence_steps", 1))}
                        for stage in getattr(driver, "prepared_stages", ())],
        "host_per_lsm_call": "prebound launch sequence plus four CUDA event records; no device sync",
        "measurement": ("CUDA events around before_lsm and advance on the hydro stream "
                        "(GPU time, LSM excluded); atmosphere = forecast wall minus hydro, "
                        "both from this run, no profiler attached"),
    }
    if status == "DEFECT":
        block["defect"] = {
            "name": DEFECT,
            "breakage": "hydro makes every default forecast slower than its budget",
            "detail": f"hydro {hydro_per_atmosphere_step_ms:.4f} ms per atmosphere step is "
                      f"{fraction:.2%} of the {atmosphere_step_ms:.3f} ms atmosphere step "
                      f"(budget {budget:.0%})",
            "owner": "hydro-speed"}
    return block


def refuse_if_over_budget(block):
    """Report the named performance defect without failing a valid forecast.

    Coordinator 2026-10-19 00:30Z: cost is a receipt defect, not a physical
    breakage that warrants refusing the forecast. Retain this compatibility
    entry point for all forecast publishers.
    """
    if block and block.get("status") == "DEFECT":
        import sys
        print(f"{DEFECT}: {block['defect']['detail']}; DEFECT recorded; "
              "forecast continues; owner hydro-speed", file=sys.stderr)
