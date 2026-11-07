"""The solve-device A/B harness, tested against known answers both ways.

CPU only.  Nothing here needs a card: the point of these tests is that the
harness reports a real difference as a difference and reports NO difference
as a fault, which is checkable without ever running the device arm.

The trap this suite exists to close is the one an A/B harness fails
silently at.  A comparison that prints "max difference 0.0" reads as
"the two arms agree" and is far more often "the treatment never ran" --
a flag that was never forwarded, a device arm that fell back to the host
path.  So the negative control is tested as hard as the positive one:
two host arms MUST come back bitwise identical, and a judge handed that
pair MUST refuse to call it agreement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import da_solve_ab as ab

#: Small enough to run in a couple of seconds, large enough that the
#: localisation lens reaches several columns and the interior is genuinely
#: observed while the rim is not.
SHAPE = dict(nz=4, ny=16, nx=16, members=4, radars=1, margin=4,
             dx_m=3000.0, top_m=10000.0, horizontal_loc_m=12000.0,
             vertical_loc_m=3000.0, obs_err_ms=1.0, thin_cells=1,
             rtps_alpha=0.9, memory_budget_mib=512.0, seed=99,
             ref_lat=35.0, ref_lon=-97.0)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("solve-ab")
    ab.make_bundle(root / "bundle", **SHAPE)
    return root / "bundle"


@pytest.fixture(scope="module")
def host_arm(bundle, tmp_path_factory):
    out = tmp_path_factory.mktemp("arm-a")
    record = ab.run_arm(bundle, "host", out_json=out / "arm.json",
                        out_npz=out / "incr.npz")
    return record, out / "incr.npz"


# ---------------------------------------------------------------------------
# the bundle is a real, self-describing, digest-checked input set
# ---------------------------------------------------------------------------

def test_bundle_is_labelled_synthetic_and_verifies(bundle):
    manifest = ab.read_manifest(bundle)
    assert manifest["kind"] == "synthetic"
    # The word has to be in the manifest, because it is copied into every
    # receipt and it is the only thing standing between a shape study and
    # a claim about a case.
    assert "not a real leg" in manifest["note"]
    ab.verify_bundle(bundle, manifest)


def test_config_survives_the_round_trip_and_the_device_is_the_treatment(
        bundle):
    """The bundle carries the whole config EXCEPT the one field it varies.

    ``dump_real_bundle`` serialises a live config and ``build_config``
    rebuilds it; if that round trip dropped a knob, the two arms would
    still agree with each other and both would be solving a different
    analysis than the leg they came from.
    """

    manifest = ab.read_manifest(bundle)
    assert "solve_device" not in manifest["config"]

    host = ab.build_config(manifest, "host")
    cuda = ab.build_config(manifest, "cuda")
    assert host.solve_device == "host"
    assert cuda.solve_device == "cuda"

    import dataclasses

    differing = [spec.name for spec in dataclasses.fields(host)
                 if getattr(host, spec.name) != getattr(cuda, spec.name)]
    assert differing == ["solve_device"]

    # And a config serialised out of a built one rebuilds identically,
    # which is what the driver's dump depends on.
    again = ab.build_config({"config": ab.serialize_config(host)}, "host")
    assert again == host


def test_a_changed_input_file_refuses_rather_than_compares(bundle, tmp_path):
    """Both arms must read the same bytes, and that is checked, not hoped.

    A bundle whose observation file was regenerated between the two arms
    would produce a difference that looks numerical and is not.
    """

    import shutil

    copy = tmp_path / "tampered"
    shutil.copytree(bundle, copy)
    obs = copy / "obs-radar-grid.nc"
    payload = bytearray(obs.read_bytes())
    payload[-1] ^= 0xFF
    obs.write_bytes(bytes(payload))
    manifest = ab.read_manifest(copy)
    with pytest.raises(SystemExit, match="digests"):
        ab.verify_bundle(copy, manifest)


# ---------------------------------------------------------------------------
# the arm records what an A/B needs to be judged
# ---------------------------------------------------------------------------

def test_arm_record_carries_the_three_way_wall_split(host_arm):
    record, _ = host_arm
    seconds = record["seconds"]
    for key in ("import", "load_and_verify", "assimilate", "letkf_setup",
                "letkf_solve", "letkf_finish", "letkf_weights",
                "letkf_transform", "stage_to_device",
                "unstage_from_device"):
        assert key in seconds, key
    # The split has to be real: before this lane there was no timing
    # anywhere in woof.da.letkf, and a zero here would mean the receipt
    # is decorating rather than measuring.
    assert seconds["letkf_solve"] > 0.0
    assert seconds["assimilate"] >= seconds["letkf_solve"]
    # The two phases account for the chunk loop, and NEVER for more than
    # it: a split that overran would mean a timer was stopped in the wrong
    # place, and the phase attribution -- which is what decides whether a
    # faster eigensolver could help this analysis at all -- would be
    # pointing at the wrong code.  The residual is per-chunk bookkeeping
    # outside both phases, chiefly the deallocation of a chunk's scratch
    # when _solve_chunk returns, and it is small.
    assert seconds["letkf_weights"] > 0.0
    assert seconds["letkf_transform"] > 0.0
    phases = seconds["letkf_weights"] + seconds["letkf_transform"]
    assert phases <= seconds["letkf_solve"] + 1e-3
    assert phases >= 0.8 * seconds["letkf_solve"]
    # A host arm stages nothing, which is what makes the device arm's
    # staging cost comparable against it.
    assert seconds["stage_to_device"] == 0.0
    assert seconds["unstage_from_device"] == 0.0


def test_arm_record_carries_the_counts_that_prove_the_inputs(host_arm):
    record, _ = host_arm
    assert record["device"] == "host"
    assert record["bundle_kind"] == "synthetic"
    filt = record["filter"]
    assert filt["total_points"] == SHAPE["nz"] * SHAPE["ny"] * SHAPE["nx"]
    # The rim is unobserved by construction, so a filter that called every
    # point active would be one that lost its localisation.
    assert 0 < filt["active_points"] < filt["total_points"]
    assert filt["max_local_obs"] > 0


# ---------------------------------------------------------------------------
# the comparator, tested in both directions
# ---------------------------------------------------------------------------

def test_two_host_arms_are_bitwise_identical(bundle, tmp_path, host_arm):
    """The negative control: same device, same inputs, same bytes.

    If this ever fails, every nonzero delta the harness reports is noise
    of its own making and no verdict off it means anything.
    """

    _, first = host_arm
    second = tmp_path / "incr-b.npz"
    ab.run_arm(bundle, "host", out_json=tmp_path / "arm-b.json",
               out_npz=second)
    fields = ab.compare_increments(first, second)
    assert fields, "the comparison found no fields to compare"
    for name, entry in fields.items():
        assert entry["bitwise_identical"], name
        assert entry["max_abs"] == 0.0, name


def test_the_comparator_sees_a_planted_difference(host_arm, tmp_path):
    """The positive control, at a magnitude the tolerance must reject.

    A comparator is only worth its zeroes if it has been shown to produce
    a non-zero on demand.
    """

    _, reference = host_arm
    with np.load(reference, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    key = sorted(payload)[0]
    nudged = payload[key].copy()
    scale = float(np.max(np.abs(nudged))) or 1.0
    nudged.flat[0] += 0.25 * scale
    payload[key] = nudged
    planted = tmp_path / "planted.npz"
    np.savez(planted, **payload)

    fields = ab.compare_increments(reference, planted)
    disagreeing = [name for name, entry in fields.items()
                   if not entry["bitwise_identical"]]
    assert disagreeing, "the planted difference was not seen at all"
    worst = max(entry["max_rel_to_field"] for entry in fields.values())
    assert worst > ab.AGREEMENT_TOLERANCE


def test_shape_mismatch_between_arms_refuses(host_arm, tmp_path):
    _, reference = host_arm
    with np.load(reference, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key])[..., :-1] for key in data.files}
    truncated = tmp_path / "truncated.npz"
    np.savez(truncated, **payload)
    with pytest.raises(SystemExit, match="shapes"):
        ab.compare_increments(reference, truncated)


# ---------------------------------------------------------------------------
# the judge: what it must refuse to call a win
# ---------------------------------------------------------------------------

def _cell(record, npz, device, seconds):
    cell = dict(record)
    cell["device"] = device
    cell["seconds"] = dict(record["seconds"])
    cell["seconds"]["assimilate"] = seconds
    cell["seconds"]["letkf_solve"] = seconds
    return {"device": device, "interleaving": "alone", "trial": 0,
            "returncode": 0, "load_returncode": None,
            "end_to_end_seconds": seconds, "load_seconds": None,
            "arm": cell, "increments_npz": str(npz),
            "arm_json": "", "log": ""}


def test_bitwise_agreement_across_devices_is_voided(bundle, host_arm,
                                                    tmp_path):
    """A cuda arm whose bytes match the host arm's did not run cuda.

    This is the A/B-arms law wired into the judge: an exact-zero delta
    across a treatment boundary is evidence the treatment never applied,
    and the harness must say so instead of reporting the speedup that
    sits beside it.
    """

    record, npz = host_arm
    manifest = ab.read_manifest(bundle)
    cells = [_cell(record, npz, "host", 10.0),
             _cell(record, npz, "cuda", 1.0)]
    receipt = ab._judge(bundle, manifest, cells, ["host", "cuda"],
                        ["alone"], None, None, 1)
    assert receipt["verdict"] == "VOID"
    assert any("BITWISE identical" in f for f in receipt["findings"])
    # And the tempting number is still present but explicitly not blessed.
    assert receipt["speedup_host_over_cuda"]["alone"][
        "assimilate_seconds"] == 10.0


def test_a_real_difference_is_called_agreement(bundle, host_arm, tmp_path):
    """The other direction: a rounding-level delta reads as AGREE."""

    record, npz = host_arm
    with np.load(npz, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    scale = max(float(np.max(np.abs(v))) for v in payload.values()) or 1.0
    rng = np.random.default_rng(7)
    for key in payload:
        payload[key] = payload[key] + rng.normal(
            0.0, 1e-13 * scale, payload[key].shape)
    nudged = tmp_path / "rounding.npz"
    np.savez(nudged, **payload)

    manifest = ab.read_manifest(bundle)
    cells = [_cell(record, npz, "host", 10.0),
             _cell(record, nudged, "cuda", 2.0)]
    receipt = ab._judge(bundle, manifest, cells, ["host", "cuda"],
                        ["alone"], None, None, 1)
    assert receipt["verdict"] == "AGREE"
    assert receipt["speedup_host_over_cuda"]["alone"][
        "assimilate_seconds"] == 5.0
    # A synthetic bundle must never let an AGREE read as settled.
    assert any("SYNTHETIC" in f for f in receipt["findings"])


def test_differing_observation_counts_void_the_comparison(host_arm):
    record, _ = host_arm
    other = json.loads(json.dumps(record))
    other["filter"]["active_points"] += 1
    counts = ab.compare_counts(record, other)
    assert not counts["same_inputs"]
    assert "active_points" in counts["mismatches"]


# ---------------------------------------------------------------------------
# the library instrumentation the harness reads
# ---------------------------------------------------------------------------

def test_letkf_diagnostics_time_the_phases_it_claims():
    """``analyze`` must fill its own three-way split, on numpy.

    Anchors the claim the receipt makes: the numbers come out of the
    filter, not out of a wrapper's stopwatch around it.
    """

    from woof.da.letkf import (GridGeometry, GriddedObs, LetkfConfig,
                                LetkfDiagnostics, Localization, analyze)

    nz, ny, nx, members = 4, 12, 12, 5
    rng = np.random.default_rng(3)
    prior = {"u": rng.normal(0.0, 1.0, (members, nz, ny, nx)),
             "v": rng.normal(0.0, 1.0, (members, nz, ny, nx))}
    mask = np.zeros((nz, ny, nx), bool)
    mask[:, 4:8, 4:8] = True
    obs = [GriddedObs(name="u", values=rng.normal(0.0, 1.0, (nz, ny, nx)),
                      errors=np.ones((nz, ny, nx)),
                      simulated=prior["u"].copy(), mask=mask)]
    geometry = GridGeometry(dx_m=3000.0, dy_m=3000.0,
                            heights_m=np.linspace(100.0, 8000.0, nz))
    config = LetkfConfig(
        localization=Localization(horizontal_m=9000.0, vertical_m=3000.0),
        analysis_fields=("u", "v"), rtps_alpha=0.9)
    # The phases are read as the minimum over repeats after one warm-up
    # call.  The first analyze in a process bills its one-time costs
    # (lazy imports and caches) to setup: on test host 4, 7.4-9.0 ms of setup
    # against 5.2-6.3 ms of solve on the first call and 1.6-1.8 ms against
    # 4.0 ms on every later one, at 3ad7ac74d and after.  And one
    # millisecond-scale sample under a loaded parallel run read 3.6 ms of
    # setup against 3.0 ms of solve.  The claim is about the phases of an
    # analysis, not about process start-up or a neighbour's load, and the
    # minimum is the least-disturbed reading of each.
    analyze(prior, obs, geometry, config, LetkfDiagnostics())
    runs = []
    for _ in range(7):
        diagnostics = LetkfDiagnostics()
        analyze(prior, obs, geometry, config, diagnostics)
        runs.append(diagnostics)

    assert diagnostics.setup_seconds > 0.0
    assert diagnostics.solve_seconds > 0.0
    assert diagnostics.finish_seconds >= 0.0
    # The chunk loop is the analysis.  A split that put the majority in
    # setup would mean the timer was reading the wrong span.
    assert (min(d.solve_seconds for d in runs)
            > min(d.setup_seconds for d in runs))
