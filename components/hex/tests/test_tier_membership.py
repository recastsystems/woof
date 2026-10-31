"""Every test file on disk is named by a tier list, and every tier row is real.

THE BREAKAGE THIS PREVENTS
--------------------------
``tools/battery/cpu_files.txt`` opens by saying why it is a list of files and
not a bare ``tests/``::

    The list is files, not a bare "tests/", so that adding a test file is a
    deliberate act of deciding which tier it belongs to.  A new file that
    nobody added here is not silently absorbed into the tier that runs
    everywhere.

Nothing enforced that, and the opposite happened.  Measured in this tree at
``dd6183d``, with ``WOOF_HEX_NO_LOCAL_GPU=1`` and the tier-1 selector
``-m "not gpu and not bigcard and not assets"``::

    56 test files on disk
    16 named by a tier list          -> 238 tests, run by CI on every push
    40 named by no tier list at all  -> 629 tests (73%), run by NOTHING
    the 40, run together:  618 passed, 11 skipped, 5 deselected in 73.91 s

Not one of the forty was device-only, asset-only or slow.  They were simply
never added.  The evidence that this is a repeating fault rather than a single
oversight is already inside ``cpu_files.txt``: the comment above
``tests/test_obs_referee.py`` records that the obs referee "was in no tier list
at all until the referee's first real run, which meant the machinery that
decides whether the physics is right was not gated on any push".  That was one
file, found by hand, when somebody happened to run it.  There were forty more.

A test file on no tier list does not fail, does not skip and does not report.
It is not slow, not flaky and not deferred -- it simply never runs, and the
green check on the pull request is telling the truth about a smaller estate
than the reader thinks they are looking at.

WHAT THIS FILE ASSERTS
----------------------
1. Every ``tests/test_*.py`` on disk is named by at least one tier list, so
   the forty-first cannot arrive quietly.
2. A file named by more than one tier is DECLARED, with the reason each tier
   takes it.  Two files earn that at dd6183d and are pinned; a third arriving
   silently doubles a leg's cost and is argued for by neither list.
3. Every path a tier list names exists, so a rename cannot leave a list
   pointing at nothing -- the same fault ``tests/test_source_manifests.py``
   already guards for the port's own source manifests, applied to the lists
   that decide what runs.
4. The tier lists are non-empty and parse, because a census over an empty
   list passes and protects nothing.

Cost: three file reads and one directory listing.  It imports nothing from
the port, needs no device, no assets and no network.
"""

from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BATTERY = ROOT / "tools" / "battery"

#: The tier lists, in tier order.  Adding a tier is a new row here and a new
#: file beside the others -- metadata, not a code path.
TIER_LISTS: dict[str, str] = {
    "tier 1 (CPU, every push)": "cpu_files.txt",
    "tier 2 (assets)": "asset_gates.txt",
    "tier 3 (gpu / bigcard)": "gpu_gates.txt",
}

#: A tier list that exists only in the maintainers' own checkout: the tests of
#: the home pool's queue, host and archive tooling, which is not distributed.
#: The public export leaves out the list and every file on it together, so a
#: public checkout has neither the list nor its files and the census below is
#: unchanged there; a home checkout counts them like any other tier.
OPTIONAL_TIER_LISTS: dict[str, str] = {
    "home pool tooling (not distributed)": "home_files.txt",
}
TIER_LISTS.update({label: name for label, name in OPTIONAL_TIER_LISTS.items()
                   if (BATTERY / name).is_file()})


def _entries(name: str) -> list[str]:
    """The pytest members of one tier list.

    The asset and gpu lists also carry harness gates that are commands rather
    than pytest files; those are prose in the file and are not entries, so
    only lines naming a ``.py`` are membership.
    """

    path = BATTERY / name
    lines = [line.strip() for line in
             path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines
            if line and not line.startswith("#") and line.endswith(".py")]


MEMBERSHIP: dict[str, list[str]] = {
    label: _entries(name) for label, name in TIER_LISTS.items()}
ON_DISK = sorted(f"tests/{path.name}"
                 for path in (ROOT / "tests").glob("test_*.py"))
ALL_ENTRIES = [(label, entry) for label, entries in sorted(MEMBERSHIP.items())
               for entry in entries]


def test_every_tier_list_is_here_and_parses() -> None:
    """A census over lists that do not exist passes and protects nothing."""

    missing = sorted(name for name in TIER_LISTS.values()
                     if not (BATTERY / name).is_file())
    assert not missing, (
        f"tier list(s) {missing} are not in tools/battery/; the census below "
        "would then report every test file as unlisted, or as listed, "
        "depending on which list vanished.  Restore the list or amend "
        "TIER_LISTS here in the same commit.")


def test_the_disk_scan_found_the_test_files_at_all() -> None:
    """The other end of the same tripwire."""

    assert len(ON_DISK) >= 56, (
        f"the scan found {len(ON_DISK)} test files under {ROOT / 'tests'}; "
        "there were 56 at dd6183d.  A scan that finds nothing passes "
        "everything, so this gate refuses to run on a tree it cannot see.")


@pytest.mark.parametrize("test_file", ON_DISK)
def test_every_test_file_on_disk_is_named_by_a_tier(test_file: str) -> None:
    """THE MEASURED FAULT.  Forty files, 629 tests, run by nothing."""

    tiers = sorted(label for label, entries in MEMBERSHIP.items()
                   if test_file in entries)
    assert tiers, (
        f"{test_file} is on no tier list, so no leg runs it: not CI (which "
        "runs tools/battery/cpu_files.txt and nothing else), not the assets "
        "tier, not the card tier.  It does not fail and it does not skip -- "
        "it reports nothing at all, and the green check claims an estate that "
        "does not include it.  Decide its tier and add it with its reason: "
        "cpu_files.txt if it runs with no card, no assets and no network, "
        "asset_gates.txt if it reads the byte-pinned asset set, "
        "gpu_gates.txt if it needs a device.")


#: The files deliberately named by more than one tier, measured at dd6183d.
#: Multi-tier membership is legitimate and both of these earn it: the tiers
#: are separated by MARKERS, so the same file can contribute its CPU members
#: to tier 1 and its device or asset members to another tier.  It is pinned
#: rather than forbidden so that a THIRD one is a decision somebody makes on
#: purpose -- an entry silently gaining a second tier doubles a leg's cost and
#: makes "which tier owns this file" unanswerable.
DUAL_TIER_ALLOWED: dict[str, str] = {
    "tests/test_fct_horizontal_scale_fusion.py":
        "tier 1 takes host hook and admission checks; tier 3 takes fused device equality",
    "tests/test_stochastic.py":
        "tier 1 takes controls, member seeds and admission; tier 3 takes "
        "mesh harmonics, GPU statistics, exact words and forcing carriers",
    "tests/test_stochastic_verification.py":
        "tier 1 takes paired daily receipt validation; tier 3 takes "
        "real-device weighted spread/RMSE and common support reductions",
    "tests/test_late_health_pipeline.py":
        "tier 1 takes host rollback and receipt failures; tier 3 takes "
        "real-device two-slot stream ordering and carrier restoration",
    "tests/test_step_gate_batch.py":
        "tier 1 takes refusal output policy; tier 3 takes pinned CUDA "
        "word transport and endpoint envelope refusals",
    "tests/test_upper_damping_v841.py": "CPU tier holds Rust/native bits and real configuration doors; card tier holds CUDA/native upper damping bits",
    "tests/test_fct_v841.py": "CPU tier replays real native nonzero transport; device tier holds bitwise replay, production dispatch and regional pool masks",
    "tests/test_rk2_v841.py": "CPU tier holds native schedule bits and coefficient reuse; device tier holds scalar density bits and repeated device steps",
    "tests/test_ntiedtke_column_row.py": "CPU tier holds the New Tiedtke row's constructor admission and engine source pin; device tier runs the row's seam on synthetic columns",
    "tests/test_cuda_v841_full_physics_x4.py":
        "tier 1 takes its constructor audits, contract pins and surface "
        "classification; tier 2 takes its four asset-touching members",
    "tests/test_device_capacity.py":
        "tier 1 proves only that the big-card preflight SKIPS with a measured "
        "number rather than passing vacuously; tier 3 runs it on the card",
    "tests/test_cuda_seam_device_snapshot.py":
        "tier 1 takes the fake-engine schema, detached-copy and allocation "
        "fallback checks; tier 3 takes the real-device alias and restore checks",
    "tests/test_partition_samehost_transport_v841.py":
        "tier 1 takes synthetic packets, route selection and fake completion "
        "fences; tier 3 takes explicitly assigned same-card and cross-card copies",
    "tests/test_cuda_graph_tables.py":
        "tier 1 takes fake allocator and descriptor residency checks; tier 3 "
        "takes retained descriptor tables during actual CUDA capture",
    "tests/test_ocean_mixed_layer.py":
        "tier 1 takes the ocean configuration, restart and budget checks; tier 3 "
        "steps the native CUDA mixed-layer ocean and sea-ice kernels",
    "tests/test_aerosol_workspace.py":
        "tier 1 takes the tiled launch's full-field, bounded-workspace plan; tier 3 takes the aerosol "
        "workspace tiles bitwise on the device",
    "tests/test_cloud_merge_oracle.py":
        "tier 1 takes the CPU cloud-merge oracle; tier 3 takes the device merge against it",
    "tests/test_transport_species_batch.py":
        "tier 1 takes batch budgets, halo round schedules, signed-zero plans and host launches; tier 3 "
        "takes the device kernel launches bitwise",
    "tests/test_rrfs_convective_inplace.py":
        "tier 1 takes the will-be-written event set; tier 3 takes the in-place rows and split launches "
        "word for word on the device",
    "tests/test_gfs_hydrometeors_f000.py":
        "tier 1 takes the GFS hydrometeor vtable rows, the .idx select and the scheme pick; tier 3 takes "
        "the start-state f000 echo and its column slabs word for word on the device",
    "tests/test_moving_kernel_modules.py":
        "tier 1 takes the moving kernel module inventory; tier 3 compiles and launches the modules on the device",
    "tests/test_moving_rates_batch.py":
        "tier 1 takes the batched rate plans on the host; tier 3 takes the batched device rates against them",
    "tests/test_moving_upload_cache.py":
        "tier 1 takes the upload cache's host bookkeeping; tier 3 takes the exact incremental device uploads",
}


@pytest.mark.parametrize("test_file", ON_DISK)
def test_multi_tier_membership_is_declared(test_file: str) -> None:
    """A second tier is legitimate; a second tier nobody wrote down is not.

    The tiers are separated by markers, not by lists, so a file with both CPU
    and device members belongs on both and its markers do the splitting --
    which is exactly how the two files above work.  What this refuses is a
    file quietly acquiring a second tier: the leg's cost doubles, the file
    runs twice, and neither list's comment block argues for the other's copy.
    """

    tiers = sorted(label for label, entries in MEMBERSHIP.items()
                   if test_file in entries)
    if len(tiers) <= 1:
        assert test_file not in DUAL_TIER_ALLOWED, (
            f"{test_file} is declared in DUAL_TIER_ALLOWED and is now on "
            f"{tiers}; the declaration outlived its reason.  Remove the row "
            "so the list keeps saying something true.")
        return
    assert test_file in DUAL_TIER_ALLOWED, (
        f"{test_file} is named by {tiers}.  If both are intended, declare it "
        "in DUAL_TIER_ALLOWED here with the reason each tier takes it, in "
        "the same commit; if not, put it on the list for the tier that owns "
        "it and let the markers separate its members.")


@pytest.mark.parametrize("tier,entry", ALL_ENTRIES, ids=lambda value: value)
def test_every_tier_entry_names_a_file_that_is_here(
    tier: str, entry: str,
) -> None:
    """A list pointing at a renamed file runs one file fewer, silently.

    This is the fault ``tests/test_source_manifests.py`` records for the
    port's own manifests -- a row naming ``run_cuda_v841_partitioned_x4.py``,
    a spelling no commit ever carried, hashed as ``null`` and carried on --
    applied to the lists that decide what runs at all.
    """

    assert (ROOT / entry).is_file(), (
        f"{tier} names {entry} and that file is not in the tree.  pytest is "
        "given a path that does not exist, or the leg quietly runs one file "
        "fewer than its list claims.  Fix the spelling, or remove the row in "
        "the same commit as the deletion and say where the coverage went.")


@pytest.mark.parametrize("tier", sorted(MEMBERSHIP))
def test_no_tier_list_names_the_same_file_twice(tier: str) -> None:
    entries = MEMBERSHIP[tier]
    repeated = sorted({entry for entry in entries
                       if entries.count(entry) > 1})
    assert not repeated, (
        f"{tier} names {repeated} more than once; the leg runs them twice and "
        "the amendment history reads as two decisions where there was one.")


def test_collection_classifies_each_source_once_and_refreshes_between_calls(tmp_path, monkeypatch):
    import conftest as tiers

    class Item:
        def __init__(self, path, name):
            self.fspath, self.name, self.markers = path, name, []

        def add_marker(self, marker):
            self.markers.append(marker)

        def get_closest_marker(self, name):
            return next((m for m in self.markers if m.name == name), None)

    source = tmp_path / "module.py"
    source.write_text("def device():\n import cupy\ndef cpu():\n pass\ndef authority():\n default_authority_paths()\n")
    monkeypatch.setattr(tiers, "NO_LOCAL_GPU", True)
    monkeypatch.setattr(tiers, "_missing_authorities", lambda: ["absent"])
    monkeypatch.setattr(tiers, "_authority_relative_paths", lambda: ["absent"])
    monkeypatch.setattr(tiers, "_bigcard_reason", lambda: None)
    calls = []
    original = tiers._cupy_scope

    def classify(text):
        calls.append(text)
        return original(text)

    monkeypatch.setattr(tiers, "_cupy_scope", classify)
    items = [Item(source, name) for name in ["device[a]", "device[b]", "cpu", "authority"]]
    tiers.pytest_collection_modifyitems(None, items)
    assert [[m.name for m in item.markers] for item in items] == [
        ["gpu", "skip"], ["gpu", "skip"], [], ["assets", "skip"]]
    assert len(calls) == 1
    source.write_text("import cupy\ndef cpu():\n pass\n")
    item = Item(source, "cpu")
    tiers.pytest_collection_modifyitems(None, [item])
    assert [m.name for m in item.markers] == ["gpu", "skip"]
    assert len(calls) == 2
