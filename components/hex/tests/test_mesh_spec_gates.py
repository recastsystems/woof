"""A spec the BUILD refuses must be refused by SIZING first.

THE DEFECT THESE PIN, MEASURED 2026-08-28 on the current engine.
``rw_mpas_mesh --dry-run`` sizes a resolution spec, prints a receipt and
exits 0 without applying the gates its own build applies.  Two designs were
reported lost to that, and both reproduce here:

* a 0.75/6/75 km design sized clean and the build refused it in 87 ms on the
  transition-band gate -- pure arithmetic on the same gradient the dry-run
  receipt had just printed;
* a 0.75/3/15/75 km design sized clean, cleared that gate, spent 1,251 s
  relaxing and deriving 217,621 cells, and was refused on the 200 m
  shortest-dual-edge floor at edge 562175 (36.8 m over an 886 m dcEdge) with
  no grid written.  The report that opened this lane read 711 s and 39.2 m
  for the same shape on another box.

The first is decidable from the spec and is now refused by sizing.  The
second is not decidable from a spec, and the fix for it is that sizing SAYS
so, with the quantities that decide it, instead of being silent.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import mesh_spec_gates as gates  # noqa: E402
from woof.hex.cli import build_parser, main  # noqa: E402
from woof.hex.swath import sizing  # noqa: E402
from woof.hex.swath.errors import SwathRefusal  # noqa: E402


#: A spec whose ramp is too steep for the surgery locality.  Reconstructed
#: from the reported 0.75/6/75 km design; the exact authored spec is not
#: available, so what is pinned here is the FAMILY and the refusal, not the
#: reporter's own number.
STEEP = {
    "name": "spec-gate-steep",
    "background_km": 75.0,
    "regions": [
        {"shape": {"kind": "cap", "center_deg": [33.0, -104.0], "radius_km": 400.0},
         "spacing_km": 6.0, "transition_cells": 40.0},
        {"shape": {"kind": "cap", "center_deg": [33.0, -104.0], "radius_km": 100.0},
         "spacing_km": 0.75, "transition_cells": 40.0},
    ],
}

#: The revised design: it CLEARS the transition-band gate, which is what
#: makes it the case the second gate has to speak about.
DEEP = {
    "name": "spec-gate-deep",
    "background_km": 75.0,
    "regions": [
        {"shape": {"kind": "cap", "center_deg": [33.0, -104.0], "radius_km": 800.0},
         "spacing_km": 15.0, "transition_cells": 108.0},
        {"shape": {"kind": "cap", "center_deg": [33.0, -104.0], "radius_km": 400.0},
         "spacing_km": 3.0, "transition_cells": 108.0},
        {"shape": {"kind": "cap", "center_deg": [33.0, -104.0], "radius_km": 120.0},
         "spacing_km": 0.75, "transition_cells": 108.0},
    ],
}

#: A spec no length floor can bind: 15 km finest is above every threshold in
#: :func:`woof.hex.mesh_spec_gates.short_dual_edge_exposure`.
COARSE = {
    "name": "spec-gate-coarse",
    "background_km": 120.0,
    "regions": [
        {"shape": {"kind": "cap", "center_deg": [39.0, -98.0], "radius_km": 1200.0},
         "spacing_km": 15.0, "transition_cells": 81.0},
    ],
}


def _engine_or_skip() -> Path:
    from woof.hex import engine_pin as _engine_pin

    try:
        engine = sizing.resolve_engine(None)
    except SwathRefusal as refusal:
        pytest.skip(f"rw_mpas_mesh is not staged on this box: {refusal}")
    if not _engine_measures_the_gradient_where_the_regions_are(str(engine)):
        pytest.fail(
            f"REFUSED: the staged {engine.name} predates the pinned engine's "
            f"own bridge bundle, so this box would judge nothing and read "
            f"green. This distribution pins {_engine_pin.gpuwm_requirement()} "
            f"and every {_engine_pin.wanted_version()} bundle probes the "
            f"gradient at the spec's own regions "
            f"(measured 2026-09-01: the published bundle's receipt reads "
            f"gradient_probe_coverage 'complete'), so a staged binary whose "
            f"--dry-run receipt carries no gradient_probe_coverage is a bridge "
            f"left over from an earlier release -- measured on the proving "
            f"desktop the same day as a v2.5.3-era bridge under a 2.6.1 "
            f"engine, with these ten gates skipping quietly past it. "
            f"THE BREAKAGE THIS REFUSAL PREVENTS: the transition-band gate "
            f"correctly declines to judge a spec whose steepest-gradient "
            f"number was measured on a 101 km global lattice, so every "
            f"mesh-spec gate and every swath sizing call stops being judged "
            f"while the battery still reports success. A bridge that is not "
            f"staged at all skips above, because a box with no bridge cannot "
            f"prove or disprove anything; a bridge that IS here and is older "
            f"than the pin is this refusal. Re-stage with `woof "
            f"fetch-bridges`, which verifies every artifact against the "
            f"packaged pins, or point $WOOF_HEX_RW_MPAS_MESH at the binary "
            f"from a {_engine_pin.wanted_version()} bundle.  staged: {engine}"
        )
    return engine


@functools.lru_cache(maxsize=None)
def _engine_measures_the_gradient_where_the_regions_are(engine: str) -> bool:
    """Does the staged engine probe the gradient at the spec's own regions?

    A STAGING PRECONDITION, not a softened gate. Before 2026-08-29 the
    generator sampled its steepest-gradient number on a Fibonacci lattice
    uniform over the whole sphere, whose 50,000 points sit 101 km apart, so it
    could not see a refinement transition narrower than that and reported the
    flat background it landed on. A receipt from such an engine carries no
    ``gradient_probe_coverage`` word, and
    :func:`woof.hex.mesh_spec_gates.gates_from_receipt` refuses to judge a spec
    on a number that is not a measurement -- which is the correct behaviour and
    is what these tests would otherwise be asserting against. The tests that
    need a real reading skip until the engine is rebuilt from the woof
    ``tools/rustwx`` workspace; they do not lower the bar to meet it.
    """

    with tempfile.TemporaryDirectory() as work:
        spec = Path(work) / "probe.mesh-spec.json"
        spec.write_text(
            json.dumps({"background_km": 240.0, "regions": []}),
            encoding="utf-8",
            newline="\n",
        )
        done = subprocess.run(
            [engine, "--spec", str(spec), "--out", str(Path(work) / "x.nc"),
             "--dry-run"],
            capture_output=True,
            text=True,
            check=False,
        )
    if done.returncode != 0:
        return False
    try:
        return "gradient_probe_coverage" in json.loads(done.stdout)
    except json.JSONDecodeError:
        return False


def _spec_file(tmp_path: Path, spec: dict) -> Path:
    path = tmp_path / f"{spec['name']}.json"
    path.write_text(
        json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    return path


def _engine(engine: Path, arguments: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(engine), *arguments], capture_output=True, text=True, check=False
    )


# ---------------------------------------------------------------------------
# the defect, on the artifact
# ---------------------------------------------------------------------------
def test_the_generator_still_sizes_a_spec_its_own_build_refuses(tmp_path) -> None:
    """The reproduction, against the exe rather than a story about it.

    IF THIS GOES RED because ``--dry-run`` began refusing, the engine has
    grown the gate and the Python one beside it is a guard whose defect is
    fixed: retire it rather than patch this test.
    """

    engine = _engine_or_skip()
    path = _spec_file(tmp_path, STEEP)

    sized = _engine(engine, ["--spec", str(path), "--dry-run"])
    assert sized.returncode == 0, sized.stderr
    receipt = json.loads(sized.stdout)
    gradient = receipt["steepest_requested_gradient_percent_per_cell"]
    assert gradient > gates.MAX_GRADIENT_PER_CELL * 100.0

    built = _engine(
        engine, ["--spec", str(path), "--out", str(tmp_path / "g.nc"), "--clobber"]
    )
    assert built.returncode != 0
    assert "surgery locality" in built.stderr
    assert not (tmp_path / "g.nc").exists()


def test_a_spec_generation_refuses_is_refused_by_sizing_first(tmp_path) -> None:
    """The fix, stated as the thing that was missing.

    Sizing had the number in its hand -- the dry-run receipt prints it -- and
    returned it as a row instead of applying it as a gate.
    """

    _engine_or_skip()
    with pytest.raises(gates.MeshSpecRefusal) as refusal:
        sizing.dry_run(STEEP)
    assert "surgery locality" not in str(refusal.value)  # our words, not a relay
    assert "before anything is built" in str(refusal.value)


def test_the_refusal_carries_the_measurement_the_limit_and_a_numeric_remedy(
    tmp_path,
) -> None:
    _engine_or_skip()
    with pytest.raises(gates.MeshSpecRefusal) as refusal:
        sizing.dry_run(STEEP)
    text = str(refusal.value)
    assert "measured" in text and "% per cell" in text
    assert f"{gates.MAX_GRADIENT_PER_CELL * 100.0:.4f} % per cell" in text
    assert "mesh/hierarchy.rs" in text
    # The remedy is a spec edit with numbers in it, not "widen the transition".
    assert "transition_cells 40 ->" in text


def test_the_front_door_refuses_before_building_and_exits_two(
    tmp_path, capsys
) -> None:
    _engine_or_skip()
    path = _spec_file(tmp_path, STEEP)
    # The ladder's gate (geometry 1); geometry 2, the default since 0.6.2, grows its own ramp.
    assert main(["mesh-plan", "--spec", str(path), "--geometry", "1"]) == 2
    assert "refused before anything is built" in capsys.readouterr().err


def test_mesh_plan_is_a_subcommand_of_the_console_script() -> None:
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # noqa: SLF001
    assert "mesh-plan" in commands


# ---------------------------------------------------------------------------
# the gate a spec cannot decide is NAMED, not skipped
# ---------------------------------------------------------------------------
def test_a_spec_that_clears_the_band_gate_still_reports_the_length_floor() -> None:
    _engine_or_skip()
    receipt = sizing.dry_run(DEEP)
    applied = receipt["gates_applied_by_hexcore"]
    assert applied["transition_band"]["band_cells"] >= gates.MIN_TRANSITION_BAND_CELLS
    edge = applied["short_dual_edge_floor"]
    assert applied["gates_this_sizing_cannot_apply"] == ["short_dual_edge_floor"]
    assert edge["decidable_before_the_build"] is False
    # THE FLOOR MOVED WITH THE STORAGE (2026-08-29).  It is 400 coordinate
    # quanta, and a mesh rw_mpas_mesh generates has no native MPAS-A
    # counterpart, so its static stores coordinates at binary64 and the same
    # budget reads 3.725e-7 m instead of 200.0 m.  This 0.75 km spec was the
    # measured 1,251 s refusal at binary32; the reading it produces now is the
    # unlock, and the 200.0 m figure survives beside it as the floor a mesh
    # WITH a native counterpart is still judged by.
    assert edge["limit_m"] == pytest.approx(3.725290298461914e-07, rel=1e-12)
    assert edge["limit_quanta"] == 400.0
    assert edge["coordinate_representation"] == "binary64_earth_centred"
    assert edge["limit_m_for_a_mesh_with_a_native_counterpart"] == 200.0
    assert edge["verdict"] == "cannot_bind"
    # The 0.75 km request's own predicted band -- unchanged, and now decades
    # above the floor rather than decades below it.  The reported binary32
    # build measured 39.2 m.
    low, high = edge["predicted_shortest_dual_edge_m"]
    assert low < high < 200.0
    assert low > edge["limit_m"] * 1e6
    assert edge["first_ladder_level_that_can_carry_a_refusal"] is None
    assert "711" not in edge["earliest_this_can_be_known"]  # a level, not a duration
    assert "not reachable" in edge["earliest_this_can_be_known"]


def test_a_coarse_spec_reports_that_the_length_floor_cannot_bind() -> None:
    exposure = gates.short_dual_edge_exposure(COARSE)
    assert exposure["verdict"] == "cannot_bind"
    assert exposure["finest_requested_spacing_km"] == 15.0
    assert exposure["first_ladder_level_that_can_carry_a_refusal"] is None
    assert min(exposure["predicted_shortest_dual_edge_m"]) > 200.0
    # Coarse or fine, the generated-mesh floor cannot bind either one now.
    assert min(exposure["predicted_shortest_dual_edge_m"]) > exposure["limit_m"]


def test_the_prediction_agrees_with_every_mesh_that_actually_built() -> None:
    """The instrument, tested in the direction that would embarrass it.

    A first draft predicted ``surgery flag floor x finest spacing`` and
    called every shipped 4 km swath a refusal while five real meshes had
    built.  A predictor is only worth printing if it reproduces the meshes
    that exist, so every row of the measured table is replayed as a spec.
    """

    for row in gates.MEASURED_GRADED_MESHES:
        label, finest, background, measured_m, ratio, emitted = row
        spec = {
            "name": label,
            "background_km": background,
            "regions": [
                {"shape": {"kind": "cap", "center_deg": [0.0, 0.0],
                           "radius_km": 400.0},
                 "spacing_km": finest, "transition_cells": 81.0},
            ],
        }
        exposure = gates.short_dual_edge_exposure(spec)
        low, high = exposure["predicted_shortest_dual_edge_m"]
        # The band's own endpoints are two of these rows, so the comparison
        # is loosened by a relative epsilon and by nothing else.
        assert low * (1 - 1e-9) <= measured_m <= high * (1 + 1e-9), (
            f"{label}: {measured_m} outside {low}-{high}"
        )
        assert ratio == pytest.approx(gates.SURGERY_FLAG_FLOOR_DV_OVER_DC, abs=0.003)
        # The table is a record of what the BINARY32 floor did, and it still
        # reproduces it exactly: `emitted` is `measured_m > 200.0 m` on every
        # row.  That is the instrument check, and it is what makes the row
        # that follows meaningful.
        assert (measured_m > gates.PUBLISHED_MIN_DV_EDGE_M) == emitted, label
        # Against the floor a GENERATED mesh is judged by today, every row
        # clears -- including the two that were refused.  The predictor must
        # never call one of them a refusal now.
        assert measured_m > gates.MIN_DV_EDGE_M, label
        assert exposure["verdict"] != "predicted_refusal", (
            f"{label} measured {measured_m:.1f} m and the predictor calls it a "
            "refusal against a 3.7e-7 m floor"
        )


def test_the_length_floor_is_never_a_refusal() -> None:
    """It is a prediction, and a prediction may not refuse.

    Refusing a spec on a prediction would refuse meshes the generator would
    have emitted -- the 7,500 m episode, in the other direction.
    """

    _engine_or_skip()
    receipt = sizing.dry_run(DEEP)
    assert receipt["predicted_cells"] > 0.0
    # It is a prediction whatever it says.  At binary32 storage this spec read
    # "predicted_refusal" and sizing still returned a receipt; at the
    # generated-mesh representation it reads "cannot_bind" and sizing still
    # returns one.  The contract under test is that neither verdict raises.
    assert (
        receipt["gates_applied_by_hexcore"]["short_dual_edge_floor"]["verdict"]
        == "cannot_bind"
    )


# ---------------------------------------------------------------------------
# the arithmetic, with no engine in the room
# ---------------------------------------------------------------------------
def test_the_band_arithmetic_crosses_six_cells_at_the_stated_ceiling() -> None:
    ceiling = gates.MAX_GRADIENT_PER_CELL
    assert gates.transition_band_cells(ceiling * 0.999) > 6.0
    assert gates.transition_band_cells(ceiling * 1.001) < 6.0
    # The published variable-resolution mesh, 1.53 % per cell, spends 45.6
    # cells crossing a doubling -- 7.6x the floor.
    assert gates.transition_band_cells(0.0153) == pytest.approx(45.649, abs=0.01)
    assert gates.transition_band_cells(0.0) == float("inf")


def test_the_ladder_is_the_generators_own() -> None:
    assert gates.ladder_km(DEEP) == [
        75.0, 37.5, 18.75, 9.375, 4.6875, 2.34375, 1.171875, 0.75
    ]
    assert gates.ladder_km({"background_km": 120.0, "regions": []}) == [120.0]


def test_widening_the_ramp_scales_whichever_spelling_a_region_uses() -> None:
    widened = gates.scaled_transitions(
        {
            "background_km": 75.0,
            "regions": [
                {"spacing_km": 3.0, "transition_cells": 40.0},
                {"spacing_km": 6.0, "transition_km": 200.0},
            ],
        },
        2.5,
    )
    assert widened["regions"][0]["transition_cells"] == 100.0
    assert widened["regions"][1]["transition_km"] == 500.0


def test_a_receipt_without_the_gradient_refuses_rather_than_skipping() -> None:
    with pytest.raises(gates.MeshSpecRefusal) as refusal:
        gates.gates_from_receipt(DEEP, {"predicted_cells": 1000.0})
    assert "cannot be applied here" in str(refusal.value)


# ---------------------------------------------------------------------------
# the constants are transcribed, so a drift must be findable
# ---------------------------------------------------------------------------
def _rust_mesh_source(engine: Path) -> Path | None:
    """``crates/rw-mpas/src/mesh`` beside a checkout-built engine, or None."""

    for parent in engine.resolve().parents:
        candidate = parent / "crates" / "rw-mpas" / "src" / "mesh"
        if candidate.is_dir():
            return candidate
    return None


def test_every_constant_still_reads_the_same_in_the_rust_that_enforces_it() -> None:
    """Skips on a box with only a staged binary, and says what went unchecked."""

    engine = _engine_or_skip()
    source = _rust_mesh_source(engine)
    if source is None:
        pytest.skip(
            f"{engine} is not a checkout build, so the rw-mpas mesh source is "
            "not beside it and the transcribed constants "
            "(SURGERY_LOCALITY_CELLS, DV_EDGE_FLOOR_QUANTA, MIN_DV_OVER_DC, "
            "LEVEL_SHIP_FLOOR_DV_OVER_DC, SURGERY_FLAG_FLOOR_DV_OVER_DC) "
            "go unchecked on this box"
        )
    hierarchy = (source / "hierarchy.rs").read_text(encoding="utf-8")
    validate = (source / "validate.rs").read_text(encoding="utf-8")
    surgery = (source / "surgery.rs").read_text(encoding="utf-8")
    coordframe = (
        source.parent / "staticfile" / "coordframe.rs"
    ).read_text(encoding="utf-8")
    assert (
        f"const SURGERY_LOCALITY_CELLS: f64 = {gates.SURGERY_LOCALITY_CELLS};"
        in hierarchy
    )
    assert "band_cells < 2.0 * SURGERY_LOCALITY_CELLS" in hierarchy
    # The floor is DERIVED there and derived here, from the same two numbers,
    # so a storage change moves both without an edit and a change to either
    # number is caught by this grep.
    assert (
        f"pub const DV_EDGE_FLOOR_QUANTA: f64 = {gates.DV_EDGE_FLOOR_QUANTA};"
        in validate
    )
    assert "min_dv_edge_m: DV_EDGE_FLOOR_QUANTA" in validate
    assert (
        "CoordinateRepresentation::Binary64EarthCentred" in coordframe
        and "pub fn for_generated_mesh() -> Self" in coordframe
    ), "the representation a generated mesh is stored at is no longer named there"
    assert f"min_dv_over_dc: {gates.MIN_DV_OVER_DC}," in validate
    assert f"flag_floor: {gates.SURGERY_FLAG_FLOOR_DV_OVER_DC}," in surgery
    assert f"ship_floor: {gates.LEVEL_SHIP_FLOOR_DV_OVER_DC}," in surgery


# ---------------------------------------------------------------------------
# the swath probe's gradient-gate verdict fails closed
# ---------------------------------------------------------------------------
def _swath_probe_module():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_swath_probe", root / "tools" / "probe_swath_spec_generable.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_swath_probe_does_not_call_an_unmeasured_spec_cleared() -> None:
    """A refusal with no printed gradient is not a cleared gate.

    The repaired meter refuses partial coverage and never-visited
    refinements with NO number in the text; the probe's old verdict
    (returncode == 0 or gradient is None) reported exactly those specs as
    having cleared the gradient gate.
    """
    probe = _swath_probe_module()

    numbered = probe._classify(1, "the requested gradient (65.20% per cell) "
                                  "makes each level's transition band ...")
    assert numbered["refused_on_gradient"] is True
    assert numbered["cleared_gradient_gate"] is False
    assert numbered["gradient_percent_per_cell"] == 65.20

    for fragment in probe._UNMEASURED_FRAGMENTS:
        unmeasured = probe._classify(1, f"refusal: {fragment} for cap-1")
        assert unmeasured["refused_unmeasured"] is True
        assert unmeasured["cleared_gradient_gate"] is False

    elsewhere = probe._classify(1, "refused: dv edge floor, another gate")
    assert elsewhere["cleared_gradient_gate"] is True
    assert elsewhere["refused_unmeasured"] is False

    built = probe._classify(0, "")
    assert built["cleared_gradient_gate"] is True


def test_the_swath_probe_fragments_still_read_the_same_in_the_rust() -> None:
    """Skips on a box with only a staged binary, like the constants test."""

    engine = _engine_or_skip()
    source = _rust_mesh_source(engine)
    if source is None:
        pytest.skip(
            f"{engine} is not a checkout build; the coverage-refusal "
            "fragments the swath probe keys on go unchecked on this box")
    probe = _swath_probe_module()
    hierarchy = (source / "hierarchy.rs").read_text(encoding="utf-8")
    for fragment in probe._UNMEASURED_FRAGMENTS:
        assert fragment in hierarchy, fragment

