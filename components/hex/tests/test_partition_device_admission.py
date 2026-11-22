"""The 2-GPU scheduler answers from the ONE admission surface.

The partition scheduler carried a second admission surface: a 22 GiB
whole-mesh constant scaled linearly through the origin with no fixed term
(the exact shape the 2026-08-25 L6 re-derivation retired), plus a separate
20 GiB ``min_free_bytes`` default.  The linear shape is UNDER-protective
for a partition: the fixed term of the measured row (4,339.1 MiB on the
170 SM card, CUDA context, local-memory backing store, module images) is
paid in full by every rank's process regardless of how few cells its
partition owns, so a floor with no fixed term admits a small partition on
a device that cannot even hold the process.  These tests pin the floors to
``woof.hex.device_admission`` so the scheduler cannot drift from the gate
every other free-memory decision applies.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from woof.hex import device_admission as surface
from woof.hex import partition_device_scheduler_v841 as scheduler

GIB = 1024**3
MIB = 1024**2
X4_CELLS = 163_842

#: The retired second surface, restated here as the before arm so the
#: breakage stays a checked fact: 22 GiB scaled linearly, no fixed term,
#: times a 1.25 headroom multiplier.
RETIRED_LINEAR = lambda local, total: int(  # noqa: E731
    22.0 * GIB * (float(local) / float(total)) * 1.25
)


# ---------------------------------------------------------------------------
# the partition floor is the one admission sum
# ---------------------------------------------------------------------------
def test_partition_floor_is_the_admission_surface_sum() -> None:
    for local in (20_000, 54_614, 90_000, X4_CELLS):
        assert scheduler.partition_min_free_bytes(local, X4_CELLS) == (
            surface.required_free_bytes(local)
        )


def test_whole_mesh_partition_agrees_with_the_native_floor() -> None:
    """A 1-partition 'split' is the whole mesh: the scheduler and the door
    must demand the same bytes for it, to the byte."""

    assert scheduler.partition_min_free_bytes(X4_CELLS, X4_CELLS) == (
        surface.native_device_floor_bytes()
    )


def test_small_partitions_are_floored_above_the_retired_linear_shape() -> None:
    """The breakage the reroute fixes: with no fixed term the retired shape
    under-charges small partitions by the whole process fixed cost."""

    # Small partitions: below about 30,000 local columns the fixed cost is
    # what the retired shape left out.  (At 40,962 today's build, the census
    # cuts, costs less than the retired line charged.)
    for local in (10_000, 20_000):
        retired = RETIRED_LINEAR(local, X4_CELLS)
        rerouted = scheduler.partition_min_free_bytes(local, X4_CELLS)
        assert retired < surface.FOOTPRINT_MODEL.predict_bytes(local), (
            "the retired shape admitted below the shaped prediction itself"
        )
        assert rerouted > retired


def test_partition_floor_still_validates_its_inputs() -> None:
    with pytest.raises(ValueError):
        scheduler.partition_min_free_bytes(0, X4_CELLS)
    with pytest.raises(ValueError):
        scheduler.partition_min_free_bytes(X4_CELLS + 1, X4_CELLS)


def test_partition_floor_derivation_is_labeled_derived_not_measured() -> None:
    """No 2-GPU per-allocation ledger row exists at the converged pin, so the
    per-partition application of the measured row is a DECLARED DERIVATION
    and must say so, and must name the measurement that replaces it."""

    record = scheduler.PARTITION_FLOOR_DERIVATION
    assert "DERIVED, NOT MEASURED" in record["status"]
    assert "2-GPU" in record["replaced_by"] or "2gpu" in record["replaced_by"]
    assert "26daaab7e" in record["replaced_by"], (
        "the replacing measurement must name the pin it is due at"
    )
    # The derivation names its source row rather than restating numbers.
    assert record["model"] == "hexcore.device_admission.FOOTPRINT_MODEL"


# ---------------------------------------------------------------------------
# require_devices defaults from the surface, and refuses by name
# ---------------------------------------------------------------------------
class _FakeCupy:
    """Just enough of the cupy surface for require_devices, CPU-only."""

    def __init__(self, free_bytes: int, total_bytes: int = 32 * GIB) -> None:
        outer = self

        class _Device:
            def __init__(self, _index: int) -> None:
                pass

            def __enter__(self) -> "_Device":
                return self

            def __exit__(self, *exc: object) -> None:
                return None

        self.cuda = SimpleNamespace(
            Device=_Device,
            runtime=SimpleNamespace(
                getDeviceCount=lambda: 1,
                getDeviceProperties=lambda _d: {
                    "major": 12,
                    "minor": 0,
                    "name": b"fake device",
                },
                memGetInfo=lambda: (outer._free, outer._total),
            ),
        )
        self._free = int(free_bytes)
        self._total = int(total_bytes)


def test_require_devices_defaults_to_the_native_floor() -> None:
    floor = surface.native_device_floor_bytes()
    receipts = scheduler.require_devices([0], cupy_module=_FakeCupy(floor))
    assert receipts[0]["free_bytes"] == floor
    with pytest.raises(scheduler.SchedulerError):
        scheduler.require_devices([0], cupy_module=_FakeCupy(floor - 1))


def test_require_devices_honours_an_explicit_partition_floor() -> None:
    local = 54_614  # the small side of a 2:1 x4 split, before halo
    floor = scheduler.partition_min_free_bytes(local, X4_CELLS)
    receipts = scheduler.require_devices(
        [0], min_free_bytes=floor, cupy_module=_FakeCupy(floor)
    )
    assert receipts[0]["free_bytes"] == floor


# ---------------------------------------------------------------------------
# the retired constants are gone
# ---------------------------------------------------------------------------
def test_the_retired_whole_mesh_constant_is_retired() -> None:
    """Fixing a defect retires its guards: the 22 GiB
    linear constant died with the L6 re-derivation and must not survive as
    an importable second surface."""

    assert not hasattr(scheduler, "WHOLE_MESH_RESIDENT_BYTES_X4")
