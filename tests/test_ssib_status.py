"""Fatal SSiB status survives until the existing model health boundary."""
import sys

import numpy as np

from woof.core import ssib
from woof.core.health import HealthField, gpu_integer_policy, rule_for_field, validate_fields_cpu


def test_production_launch_accumulates_fatal_bits_without_readback_or_table_io(monkeypatch):
    # A device-array API stand-in deliberately has no .get method. Passing
    # a bound function must also keep the table loader out of the step.
    monkeypatch.setitem(sys.modules, "cupy", np)
    def unexpected_table_load(*args):
        raise AssertionError("production step reloaded tables")
    monkeypatch.setattr(ssib, "kernel", unexpected_table_load)
    ncol, nz = 2, 3
    static = np.zeros((len(ssib.STATIC), ncol), np.float32)
    state = np.zeros((len(ssib.STATE), ncol), np.float32)
    forcing = np.zeros((len(ssib.FORCING) + nz, ncol), np.float32)
    status = np.zeros(1, np.int32)
    for step, bit in enumerate((8, 1, 8, 2, 4), 1):
        def launch(grid, block, args):
            args[-1][0] |= bit
        result = ssib.step(static, state, forcing, nz=nz, itimestep=step,
                           fixes=0, function=launch, status=status)
        assert result is None
        assert int(status[0]) == (0, 1, 1, 3, 7)[step - 1]


def test_health_boundary_rejects_every_fatal_status_word():
    name = "surface._ssib_status"
    assert gpu_integer_policy(name, np.int32) is not None
    rule = rule_for_field(name)
    for bits in range(8):
        result = validate_fields_cpu([
            HealthField(name, np.array([bits], np.int32), rule)])
        assert result.ok == (bits == 0)
