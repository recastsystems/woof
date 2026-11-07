"""The GOCART processes' host-side decisions: WRF step cadences and the rows
a process may write.

CPU-only on purpose: conftest marks a whole module ``gpu`` when a helper in it
imports cupy, and these are pure integer decisions that every routine run
should check.
"""
from types import SimpleNamespace


def test_optics_cadence_is_wrf_photstep_on_the_radiation_interval():
    """do_photstep (chem_driver.F:349-365) with stepphot =
    max(nint(photdt*60/dt), 1) (chemics_init.F:643-648), photdt = radt."""
    from woof.core.chem_optics import photolysis_step

    cfg = SimpleNamespace(radt=0.0, radt_minutes=10.0)
    assert [k for k in range(1, 41) if photolysis_step(cfg, k, 30.0)] == [1, 20, 40]
    # nint rounds half away from zero: 1.25 min at 30 s is 2.5 -> 3 steps.
    cfg = SimpleNamespace(radt=1.25, radt_minutes=10.0)
    assert [k for k in range(1, 10) if photolysis_step(cfg, k, 30.0)] == [1, 3, 6, 9]
    cfg = SimpleNamespace(radt=0.0, radt_minutes=0.0)
    assert all(photolysis_step(cfg, k, 30.0) for k in range(1, 5))


def test_chem_step_is_wrf_stepchem_with_fortran_nint():
    """chem_driver.F:393-416: ktau 1 over dt, then every stepchem-th step over
    the time since the last one; stepchem = max(nint(chemdt*60/dt), 1)."""
    from woof.core.chem_sulfur import chem_step_dt

    cfg = SimpleNamespace(chemdt=1.25, dt=30.0)   # 2.5 steps: nint -> 3
    got = [(k, chem_step_dt(cfg, k, 30.0)) for k in range(1, 8)]
    assert [k for k, (run, _) in got if run] == [1, 3, 6]
    assert dict((k, d) for k, (run, d) in got if run) == {1: 30.0, 3: 60.0, 6: 90.0}
    cfg = SimpleNamespace(chemdt=0.0, dt=30.0)
    assert all(chem_step_dt(cfg, k, 30.0) == (True, 30.0) for k in range(1, 4))


def test_ageing_refuses_a_target_row_that_does_not_name_it():
    """The driver books a process's change only on rows that name it; a
    partner row outside that set had its aged mass filed under another
    process (seen in a 4 h forecast: oc2's under coupling, bc2's under
    transport).  The shipped rows name it; a table that does not is refused."""
    from dataclasses import replace

    import pytest

    from woof.chem_table import chem_names, load_sets
    from woof.core.chem_ageing import KEY, participants

    table = load_sets(chem_names("gocart_simple"), ())
    ordered, _ = participants(table)
    assert [r.name for r in ordered] == ["bc1", "bc2", "oc1", "oc2"]
    stripped = tuple(
        replace(r, processes=tuple(p for p in r.processes if p != KEY))
        if r.name == "bc2" else r for r in table.rows)
    broken = replace(table, rows=stripped, _by_name=None)
    with pytest.raises(ValueError, match=r"\['bc2'\] receive aged mass"):
        participants(broken)



def test_every_gocart_process_names_its_kernels_and_each_has_a_ceiling():
    """The preflight prices a chem domain's kernel frames from each
    process's KERNEL_MODULES and refuses a process without one; every
    module a GOCART set launches must also carry a shipped frame ceiling,
    or pricing fails closed.  (wetdep.ls, whose module lands with
    development branch, is dropped from the rows here.)"""
    import dataclasses

    from woof.chem_table import chem_names, load_sets
    from woof.core import preflight as pf
    from woof.core.chem_context import chem_kernel_modules

    for name in ("gocart_simple", "gocart_lite", "dust", "gocart_primary"):
        table = load_sets(chem_names(name), ())
        rows = tuple(dataclasses.replace(r, processes=tuple(
            p for p in r.processes if p != "wetdep.ls")) for r in table.rows)
        table = dataclasses.replace(table, rows=rows, _by_name=None)
        modules = chem_kernel_modules(table, 1)
        assert {"chem_dust", "chem_settling", "chem_drydep_gocart",
                "chem_optics"} <= modules, name
        missing = sorted(m for m in modules
                         if m not in pf.KERNEL_MAX_LOCAL_SIZE_BYTES)
        assert not missing, (name, missing)
