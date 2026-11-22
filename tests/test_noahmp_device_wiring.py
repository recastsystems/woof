"""Does a Noah-MP forecast actually launch the CUDA leaves?

This file exists because the answer has been "no" twice in this repository,
in the neighbouring lane, within a week.  ``woof/core/ruc_gpu.py`` carried
four bitwise device leaves and a bitwise snow-preparation stage, and
``ruc_lsm_step`` called its driver without them, so **every RUC forecast this
project had ever run took the host Python path** at 613 s per d04 land-surface
call while the per-leaf parity gates stayed green and a published 16.8 s figure
described code nothing launched.  The RUC soil ingest had the same shape:
finished, verified, and wired to nothing.

**A timing gate cannot catch it.**  The host path is also correct, merely
slow, so nothing fails -- there is just a number nobody can reproduce.  What
catches it is making the host path *impossible* and requiring the forecast to
survive anyway.

Noah-MP is wired: ``LEAF_BATCH_EVALUATOR`` defaults to the device evaluator
and ``noahmp_runtime`` imports all eight batches.  That is a property of the
tree today, not a law, and it is exactly the kind of property that rots in a
refactor without anything going red.  So:

* :func:`test_a_forecast_runs_with_every_host_leaf_booby_trapped` replaces
  every CPython leaf answer with a raising tripwire and steps a forecast.  If
  any leaf were being answered on the host the step cannot finish.
* :func:`test_the_booby_trap_fires_when_the_device_set_is_withheld` withholds
  the device evaluator from the same call and requires the tripwire to fire,
  so the pass above is evidence rather than a gate that has never been able
  to fail.
* :func:`test_every_device_batch_is_imported_from_the_runtime` reads the seam
  directly: each entry of ``DEVICE_LEAF_BATCH`` must resolve to a function
  defined in a ``woof.core.noahmp_*_gpu`` module.  A batch quietly rebound to
  a host routine would keep every other test in this suite green.
"""

from __future__ import annotations

import pytest

from conftest import requires_gpu

import woof.core.noahmp_energy as energy_module
import woof.core.noahmp_runtime as runtime_module

from test_noahmp_runtime import _build


class HostLeafTripwire(RuntimeError):
    """Raised if any Noah-MP leaf is answered by the CPython routine."""


def _booby_trap(monkeypatch):
    """Replace every registered host leaf answer with a tripwire."""
    trapped = {}
    for name in energy_module._HOST_LEAF:
        def trip(*args, _name=name, **kwargs):
            raise HostLeafTripwire(
                f"the Noah-MP {_name!r} leaf was answered on the host; the "
                "device batch did not run")
        trapped[name] = trip
    monkeypatch.setattr(energy_module, "_HOST_LEAF", trapped)
    return trapped


#: Every leaf a land column pauses at.  Asserted rather than assumed so that
#: trapping "every host leaf" cannot quietly become trapping three of them.
_EXPECTED_LEAVES = frozenset({
    "sflx_pre", "thermoprop", "radiation", "vege_flux", "bare_flux",
    "tsnosoi", "phasechange", "water",
})


def test_the_trap_covers_every_batched_leaf():
    """The trap is only worth anything if it covers the whole seam."""
    assert set(runtime_module.DEVICE_LEAF_BATCH) == _EXPECTED_LEAVES
    # RADIATION and SFLX_PRE register their host answers on import of the
    # modules that own them; importing the runtime is what pulls them in.
    missing = _EXPECTED_LEAVES - set(energy_module._HOST_LEAF)
    assert missing == set(), (
        f"{sorted(missing)} have no registered host answer, so a tripwire "
        "placed over the registry would not cover them")


@requires_gpu
def test_a_forecast_runs_with_every_host_leaf_booby_trapped(monkeypatch):
    """A forecast, with the CPython leaves made impossible to call."""
    from woof.core.dycore import step

    state, cfg, driver = _build(nx=6, ny=4, water_columns=1,
                                snow_mm=45.0, snow_depth_m=0.16)
    _booby_trap(monkeypatch)
    monkeypatch.setattr(runtime_module, "LEAF_BATCH_EVALUATOR",
                        runtime_module.evaluate_leaf_batch_on_device)
    monkeypatch.delenv(runtime_module.HOST_LEAVES_ENV, raising=False)
    step(state, cfg)
    assert driver.last_noahmp_census["land"] > 0, (
        "no land column ran, so the trap was never in a position to fire")


@requires_gpu
def test_the_booby_trap_fires_when_the_device_set_is_withheld(monkeypatch):
    """The same call with the device evaluator withheld MUST trip.

    Without this, the test above would pass just as well against a trap that
    had been installed in the wrong place.
    """
    from woof.core.dycore import step

    # The host leaves are WRF's arithmetic, so correct-mode SN1 is named off
    # (a host run with it on is refused before any leaf is called).
    state, cfg, driver = _build(nx=6, ny=4, water_columns=1,
                                snow_mm=45.0, snow_depth_m=0.16,
                                noahmp_melt_reset_esat=False)
    _booby_trap(monkeypatch)
    monkeypatch.setattr(runtime_module, "LEAF_BATCH_EVALUATOR",
                        runtime_module.evaluate_leaf_batch_on_host)
    with pytest.raises(HostLeafTripwire):
        step(state, cfg)


@requires_gpu
def test_the_env_override_also_reaches_the_host_path(monkeypatch):
    """``WOOF_NOAHMP_HOST_LEAVES=1`` is the other way to select the host.

    It is the switch every timing run uses, so it has to be shown selecting
    the path it claims to.
    """
    from woof.core.dycore import step

    state, cfg, driver = _build(nx=6, ny=4, water_columns=1,
                                noahmp_melt_reset_esat=False)
    _booby_trap(monkeypatch)
    monkeypatch.setattr(runtime_module, "LEAF_BATCH_EVALUATOR",
                        runtime_module.evaluate_leaf_batch_on_device)
    monkeypatch.setenv(runtime_module.HOST_LEAVES_ENV, "1")
    with pytest.raises(HostLeafTripwire):
        step(state, cfg)


class StagedPathTripwire(RuntimeError):
    """Raised if the per-column staged driver runs on the default path."""


class SlabPathTripwire(RuntimeError):
    """Raised inside the slab orchestration, to prove a forecast reaches it."""


@requires_gpu
def test_the_forecast_takes_the_slab_path_not_the_staged_one(monkeypatch):
    """Dead code has happened twice on RUC; this is the trap for round three.

    The slab orchestration exists and is gated bitwise -- which is exactly
    the state the RUC device leaves were in while every forecast took the
    other path.  So the staged driver is made impossible to call and a
    forecast is required to survive: if any land column were driven through
    the per-column seam, the step could not finish.
    """
    from woof.core.dycore import step

    state, cfg, driver = _build(nx=6, ny=4, water_columns=1,
                                snow_mm=45.0, snow_depth_m=0.16)

    def trip(*args, **kwargs):
        raise StagedPathTripwire(
            "the per-column staged driver ran on a default forecast; the "
            "slab orchestration is not wired")

    monkeypatch.setattr(runtime_module, "_drive_staged_columns", trip)
    monkeypatch.delenv(runtime_module.STAGED_COLUMNS_ENV, raising=False)
    monkeypatch.delenv(runtime_module.HOST_LEAVES_ENV, raising=False)
    step(state, cfg)
    assert driver.last_noahmp_census["land"] > 0, (
        "no land column ran, so the trap was never in a position to fire")


@requires_gpu
def test_the_staged_trap_fires_when_the_staged_path_is_selected(monkeypatch):
    """The same trap must fire when the staged path is asked for by name.

    Without this, the pass above would be consistent with a trap installed
    somewhere nothing ever calls.
    """
    from woof.core.dycore import step

    state, cfg, _driver = _build(nx=6, ny=4, water_columns=1)

    def trip(*args, **kwargs):
        raise StagedPathTripwire("staged path ran")

    monkeypatch.setattr(runtime_module, "_drive_staged_columns", trip)
    monkeypatch.setenv(runtime_module.STAGED_COLUMNS_ENV, "1")
    with pytest.raises(StagedPathTripwire):
        step(state, cfg)


@requires_gpu
def test_the_slab_orchestration_is_what_answers_the_columns(monkeypatch):
    """The positive half: the orchestration itself must be on the path.

    Surviving the staged trap proves the staged path did not run; this
    proves the slab path did, by making its evaluator raise and requiring
    the forecast to fail.  A dispatch that silently answered land columns a
    third way would pass the trap above and fail here.
    """
    import woof.core.noahmp_column_slab as column_slab
    from woof.core.dycore import step

    state, cfg, _driver = _build(nx=6, ny=4, water_columns=1)

    def trip(fields, n, *, noahmp_snow_age):
        raise SlabPathTripwire(f"evaluate_sflx_slab reached with n={n}")

    monkeypatch.setattr(column_slab, "evaluate_sflx_slab", trip)
    monkeypatch.delenv(runtime_module.STAGED_COLUMNS_ENV, raising=False)
    monkeypatch.delenv(runtime_module.HOST_LEAVES_ENV, raising=False)
    with pytest.raises(SlabPathTripwire, match="n="):
        step(state, cfg)


def test_every_device_batch_is_imported_from_the_runtime():
    """The import edge itself, read rather than trusted.

    ``ruc_runtime`` gates this for RUC precisely because the edge is what
    rotted.  Every Noah-MP batch must come from a ``*_gpu`` module; a batch
    rebound to a CPython routine would leave every other test green.
    """
    offenders = {}
    for leaf, batch in runtime_module.DEVICE_LEAF_BATCH.items():
        module = getattr(batch, "__module__", "")
        if not (module.startswith("woof.core.noahmp")
                and module.endswith("_gpu")):
            offenders[leaf] = module
    assert offenders == {}, (
        "these Noah-MP leaves are not answered by a device batch module: "
        f"{offenders}")


@requires_gpu
def test_a_host_leaf_run_with_sn1_on_is_refused(monkeypatch):
    """The host leaves carry WRF's stale-ESTG snowmelt reset.  Breakage
    prevented: a host-authority run under the correct-mode default quietly
    differing from the forecast route at every sunlit melting snow column."""
    from woof.core.dycore import step

    state, cfg, driver = _build(nx=6, ny=4, water_columns=1,
                                noahmp_melt_reset_esat=True)
    monkeypatch.setattr(runtime_module, "LEAF_BATCH_EVALUATOR",
                        runtime_module.evaluate_leaf_batch_on_host)
    with pytest.raises(ValueError, match="noahmp_melt_reset_esat"):
        step(state, cfg)
