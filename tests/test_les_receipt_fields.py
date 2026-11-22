"""The LES receipt's determinism-screen field partition.

CPU-only.  The dual-run screen for the CBL receipts is documented as
"compare every numeric receipt field, excluding wall-clock timings"
(docs/internal/receipts/les/ARWEN-REALISATION-SPREAD.md §1).  The
sm_89 stress lane falsified that phrasing on a shared card: two
same-seed reps were byte-identical in every other field and differed
ONLY in ``vram_gib`` (0.72 vs 1.60 GiB), because that figure was
device-wide occupancy ((total - free) from ``cudaMemGetInfo``) and a
forecast co-tenanted the card.  A screen whose field set contains an
environment measurement false-positives on healthy runs.

The receipt therefore says, in code, which of its fields are properties
of the run (screened -- the corruption-detection surface) and which are
properties of the machine at the moment of measurement (environmental
-- excluded).  These tests pin that partition: it must be exhaustive
and disjoint over real committed receipts, the environmental side must
own every timing and every device-wide memory name past and present,
and the per-process pool figure must stay on the screened side.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from woof.verify.cases import convective_boundary_layer as cbl

_LES_RECEIPTS = (Path(__file__).resolve().parents[1]
                 / "docs" / "internal" / "receipts" / "les")
_COMMITTED = sorted(
    _LES_RECEIPTS.glob("cbl-2026-08-02/*/cbl_*_receipt.json"))

#: The committed cbl-2026-08-02 receipts live under a release-excluded
#: path, so the three tests that read them skip in a published clone --
#: the same convention test_resolved_scale_receipt.py uses for its
#: internal half.  The partition-logic tests below need no files and
#: are checked in both trees.
_INTERNAL_RECEIPTS_PRESENT = bool(_COMMITTED)
internal_receipts = pytest.mark.skipif(
    not _INTERNAL_RECEIPTS_PRESENT,
    reason=(
        "the committed cbl-2026-08-02 receipts live under a "
        "release-excluded path; a published clone does not have them, "
        "and the partition they pin is asserted against real receipts "
        "in the development tree"
    ),
)


def _committed_receipt(name: str) -> dict:
    path = _LES_RECEIPTS / "cbl-2026-08-02" / name
    return json.loads(path.read_text(encoding="utf-8"))


@internal_receipts
def test_partition_is_exhaustive_and_disjoint_on_committed_receipts():
    """Every field of every committed receipt lands in exactly one
    bucket -- a field the partition cannot place would silently fall out
    of the screen, which is the defect this fixes wearing a new name."""
    assert _COMMITTED, "committed cbl-2026-08-02 receipts are missing"
    for path in _COMMITTED:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        screened, environmental = cbl.partition_receipt_fields(receipt)
        assert set(screened) | set(environmental) == set(receipt), path.name
        assert not set(screened) & set(environmental), path.name


@internal_receipts
def test_environment_owns_timings_and_the_legacy_device_wide_field():
    """The committed receipts carry the pre-split ``vram_gib`` -- the
    device-wide figure under its old name.  It partitions environmental,
    beside the wall-clock timings, so the screen definition applies to
    the historical receipts without editing them."""
    screened, environmental = cbl.partition_receipt_fields(
        _committed_receipt("km2/cbl_km2_receipt.json"))
    for name in ("vram_gib", "wall_seconds", "steps_per_second"):
        assert name in environmental, name
        assert name not in screened, name


def test_the_screened_set_excludes_every_device_wide_memory_name():
    """Past name and present name both classify environmental; the
    per-process pool figure is deliberately NOT environmental, because
    this process's live allocation is a deterministic property of the
    run and belongs in the comparison."""
    assert "vram_gib" in cbl.ENVIRONMENTAL_FIELDS
    assert "vram_device_gib" in cbl.ENVIRONMENTAL_FIELDS
    assert "vram_pool_gib" not in cbl.ENVIRONMENTAL_FIELDS


@internal_receipts
def test_physics_and_step_count_fields_are_screened():
    """The partition removes environment measurements, never physics --
    and ``steps`` stays screened even though it smells like a timing,
    because two same-seed runs that stepped different counts ARE
    different runs."""
    screened, _ = cbl.partition_receipt_fields(
        _committed_receipt("km3/cbl_km3_receipt.json"))
    for name in ("zi_m", "w_max", "wth_res_max_over_qs",
                 "tke_res_max", "updraft_fraction_low", "steps", "nan"):
        assert name in screened, name


def test_a_post_split_receipt_partitions_the_two_vram_fields_apart():
    """The whole point of the split: the two VRAM figures land on
    opposite sides of the screen."""
    receipt = {"zi_m": 1500.0, "vram_pool_gib": 0.7,
               "vram_device_gib": 14.4, "wall_seconds": 100.0}
    screened, environmental = cbl.partition_receipt_fields(receipt)
    assert "vram_pool_gib" in screened
    assert "vram_device_gib" in environmental


def test_vram_fields_reads_pool_and_device_from_their_own_sources():
    """The pool figure comes from the injected pool reader alone and the
    device figure from the injected memGetInfo-shaped reader alone --
    the accounting is testable CPU-side, the gpu_mem_watch pattern."""
    fields = cbl._vram_fields(
        pool_used_bytes=lambda: 3 * 2**30,
        device_free_total=lambda: (1 * 2**30, 24 * 2**30))
    assert fields == {"vram_pool_gib": 3.0, "vram_device_gib": 23.0}


def test_a_failing_memory_reader_yields_null_never_a_dead_run():
    """A missing memory figure must not kill a finished integration, and
    one reader failing must not take the other's figure with it."""
    def boom():
        raise RuntimeError("no CUDA context")

    fields = cbl._vram_fields(
        pool_used_bytes=boom,
        device_free_total=lambda: (2**30, 2 * 2**30))
    assert fields["vram_pool_gib"] is None
    assert fields["vram_device_gib"] == 1.0

    fields = cbl._vram_fields(
        pool_used_bytes=lambda: 2**29, device_free_total=boom)
    assert fields["vram_pool_gib"] == 0.5
    assert fields["vram_device_gib"] is None


def test_vram_fields_names_agree_with_the_partition():
    """The helper and the field partition cannot drift apart: every name
    the helper emits is classified, and the device-wide one is the only
    environmental one among them."""
    names = set(cbl._vram_fields(
        pool_used_bytes=lambda: 0, device_free_total=lambda: (0, 0)))
    assert names == {"vram_pool_gib", "vram_device_gib"}
    assert names & cbl.ENVIRONMENTAL_FIELDS == {"vram_device_gib"}
