"""The MPAS mesh generator has to be reachable, not merely built.

``rw_mpas_mesh`` reproduced the published NCAR meshes to 1e-11 and had
no row in :data:`woof.bridge_assets.BUNDLED_ARTIFACTS`, no contract
marker, no environment variable, no resolver and no Python that called
it.  ``rw_mpas_init`` and ``rw_mpas_convert`` were in the same state.
That is the ``rw_mrms`` wave again and the ``rw_mpas_convert`` failure
by name: committed is not shipped, and a capability with no front door
is not a feature.

Every test here is a property of the REACHABLE path -- the bundle
table, the contract marker, the resolution ladder, the sizing refusal,
the smoothness refusal, the command a user types -- rather than of the
crate, which has its own goldens.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from woof import bridge_assets, bridges, mpas_mesh

#: The artifacts this crate builds and nothing shipped.  ``rw_mpas_lbc``
#: is the fourth and it arrived in the same state as the other three:
#: a declared ``[[bin]]`` that every release cut BUILT, that no bundle
#: carried, and that no check reported -- so the only way to get it was
#: a Rust toolchain and a source checkout.  It is the other half of the
#: limited-area route (``rw_mpas_mesh --cull-parent`` cuts the mesh;
#: this produces the boundary series that mesh is driven by), so its
#: absence made the cull's output a file nothing could run.
MPAS_ARTIFACTS = ("rw_mpas_mesh", "rw_mpas_init", "rw_mpas_convert",
                  "rw_mpas_lbc", "rw_mpas_geometry", "rw_mpas_hostprep")


# ---------------------------------------------------------------------------
# 1. the bundle table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", MPAS_ARTIFACTS)
def test_every_mpas_binary_is_a_bundled_artifact(name):
    """FAILS before the fix: no bundle carries it, so no remedy can."""

    names = {a.name for a in bridge_assets.BUNDLED_ARTIFACTS}
    assert name in names, (
        f"{name} is built by tools/rustwx and resolved by "
        "woof.mpas_mesh, but woof fetch-bridges does not stage it -- "
        "so the command its refusal names cannot supply it.  That is "
        "rw_mpas_convert again, which is the failure this file exists "
        "to have stopped.")


@pytest.mark.parametrize('name', ['rw_mpas_convert'])
def test_the_bundled_row_uses_the_resolvers_own_env_var(name):
    """A staged bundle and a hand-built tree, found by one code path."""

    artifact, = [a for a in bridge_assets.BUNDLED_ARTIFACTS
                 if a.name == name]
    bridge = mpas_mesh.BRIDGES[name]
    assert artifact.env_var == bridge.env_var
    assert artifact.kind == "executable"
    assert artifact.crate == bridges.RUSTWX_CRATE_RELATIVE
    assert not artifact.vendored, (
        f"{name} is woof's own crate, so its staleness is proved by the "
        "source-revision stamp, not by a vendoring note")


def test_the_mesh_env_var_is_the_one_the_brief_names():
    """The name is part of the contract; a synonym is a second ladder."""

    assert mpas_mesh.BRIDGES["rw_mpas_mesh"].env_var == "WOOF_RW_MPAS_MESH"


# ---------------------------------------------------------------------------
# 2. the contract marker
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['rw_mpas_convert'])
def test_every_mpas_binary_declares_a_contract_marker(name):
    """A stale build must fail statically, not at the first mesh."""

    assert name in bridges.BRIDGE_ABI_MARKERS, (
        f"{name} has no entry in woof.bridges.BRIDGE_ABI_MARKERS, so a "
        "binary built before this release's argument vector passes "
        "`woof doctor` and then mis-parses the request")


def test_the_marker_and_the_drivers_abi_line_are_the_same_string():
    """Two spellings of one contract is how they drift apart."""

    marker = bridges.BRIDGE_ABI_MARKERS["rw_mpas_mesh"]
    assert marker == mpas_mesh.MESH_ABI_MARKER.encode("utf-8")


def test_the_converter_marker_names_the_flags_its_callers_pass(tmp_path):
    """A converter that predates ``--remap``/``--jobs`` is stale, not current.

    THE BREAKAGE: the marker stayed ``...FINISHED`` when 2.9.0 added
    ``--remap`` (REMAP/BUDGET lines) and ``--jobs``, so a pre-2.9.0
    ``~/.woof/bridges/rw_mpas_convert`` passed ``woof doctor`` and the
    woof hex route-A product jobs died on ``unknown flag --remap``.
    """

    marker = bridges.BRIDGE_ABI_MARKERS["rw_mpas_convert"]
    assert marker == mpas_mesh.CONVERT_ABI_MARKER.encode("utf-8")
    for flag in (b"--remap nearest|product", b"--jobs N", b"\tREMAP\t", b"\tBUDGET\t"):
        assert flag in marker, flag
    source = (REPO_ROOT / "tools" / "rustwx" / "crates" / "rw-mpas" / "src"
              / "bin" / "rw_mpas_convert.rs")
    if source.is_file():
        literal = re.search(r'^const ABI: &str = "([^"]*)";$',
                            source.read_text(encoding="utf-8"), re.M)
        assert literal is not None
        assert literal.group(1).replace("\\t", "\t").encode("utf-8") == marker
    # A pre-2.9.0 build carries only the old literal: it must read as stale.
    stale = tmp_path / "rw_mpas_convert"
    stale.write_bytes(b"\0gpuwm-rw-mpas-convert-v1\tCONVERTED\tWINDOW\tWEIGHTS"
                      b"\tFINISHED--history is required\0")
    ok, evidence = bridges.bridge_abi_matches("rw_mpas_convert", stale)
    assert not ok and "predates" in evidence


# ---------------------------------------------------------------------------
# 3. the source-revision stamp
# ---------------------------------------------------------------------------

def test_the_crate_carries_a_build_script_that_stamps_the_revision():
    """The cut refuses a bundle whose binaries cannot prove their source."""

    crate = REPO_ROOT / "tools" / "rustwx" / "crates" / "rw-mpas"
    if not crate.is_dir():                       # pragma: no cover - wheel
        pytest.skip("no checkout crate in this install")
    script = crate / "build.rs"
    assert script.is_file(), (
        "tools/rustwx/crates/rw-mpas has no build.rs, so nothing injects "
        "GPUWM_BRIDGE_SOURCE_REV and the release cut cannot prove the "
        "staged binary was built from the released commit")
    text = script.read_text(encoding="utf-8")
    assert "GPUWM_BRIDGE_SOURCE_REV" in text
    manifest = (crate / "Cargo.toml").read_text(encoding="utf-8")
    assert 'build = "build.rs"' in manifest, (
        "Cargo will not run a build script the manifest does not declare "
        "when the crate also has an explicit [lib]/[[bin]] layout")


@pytest.mark.parametrize('name', ['rw_mpas_convert'])
def test_each_entry_point_embeds_the_stamp(name):
    """A build script that no entry point reads stamps nothing."""

    crate = REPO_ROOT / "tools" / "rustwx" / "crates" / "rw-mpas"
    source = crate / "src" / "bin" / f"{name}.rs"
    if not source.is_file():                     # pragma: no cover - wheel
        pytest.skip(f"{source} is not in this install")
    assert 'env!("GPUWM_BRIDGE_SOURCE_REV")' in source.read_text(
        encoding="utf-8"), (
        f"{name}.rs never reads GPUWM_BRIDGE_SOURCE_REV, so the compiler "
        "embeds no literal and the stamp cannot be read out of the bytes")


# ---------------------------------------------------------------------------
# 4. the resolution ladder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['rw_mpas_mesh'])
def test_the_ladder_is_the_one_every_other_rustwx_door_uses(name):
    """One ladder, or a staged bundle and a checkout disagree."""

    from woof.obs.frontdoor import FrontDoor

    reference = FrontDoor(name=name, env_var=mpas_mesh.BRIDGES[name].env_var,
                          subject="reference", abi_marker="")
    assert mpas_mesh.BRIDGES[name].candidates() == reference.candidates()


def test_an_env_override_naming_a_missing_file_is_a_hard_error(tmp_path,
                                                               monkeypatch):
    """Explicit configuration fails loudly, never falls through."""

    monkeypatch.setenv("WOOF_RW_MPAS_MESH", str(tmp_path / "absent.exe"))
    with pytest.raises(FileNotFoundError):
        mpas_mesh.BRIDGES["rw_mpas_mesh"].find()


def test_a_missing_binary_refuses_with_a_remedy_naming_this_install():
    """A refusal with no route out is the wave this file records."""

    remedy = mpas_mesh.BRIDGES["rw_mpas_mesh"].remedy()
    assert "rw_mpas_mesh" in remedy or "rustwx" in remedy
    assert remedy.strip()


# ---------------------------------------------------------------------------
# 5. the capacity model is DATA, and card-specific
# ---------------------------------------------------------------------------

def test_the_capacity_model_lives_in_a_data_file_not_in_code():
    """The fixed term is a property of the CARD; a literal freezes one."""

    path = mpas_mesh.SIZING_DATA_PATH
    assert path.is_file(), f"{path} is not shipped"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema"] == mpas_mesh.SIZING_SCHEMA
    source = (REPO_ROOT / "woof" / "mpas_mesh.py").read_text(
        encoding="utf-8")
    for card in mpas_mesh.load_sizing().cards.values():
        if card.fixed_mib is None:
            continue
        assert str(int(card.fixed_mib)) not in source, (
            f"{card.key}'s measured fixed term is a literal in "
            "woof/mpas_mesh.py; it has to move by editing the table")


def test_at_least_one_card_row_is_measured_and_names_its_anchors():
    """A model with no anchors is an assumption wearing a number.

    Two terms need two anchors to separate them BY SUBTRACTION.  A row
    with one anchor is admissible only when it says where the other term
    came from and that source is a row which did separate its own -- the
    5070 Ti's slope is the 5090's, and the measurement that says the two
    are the same (5,404.5 MiB of pool on both, to the decimal) is in
    ``bytes_per_cell_provenance``.  Silently inheriting a term is the
    thing this test exists to stop.
    """

    sizing = mpas_mesh.load_sizing()
    measured = [c for c in sizing.cards.values() if c.measured]
    assert measured, "no card row carries a measured footprint model"
    separated = [c for c in measured if len(c.anchors) >= 2]
    assert separated, (
        "no card row separated the fixed and per-cell terms from two or "
        "more of its own anchors, so every number in the table is "
        "inherited from something that is not here")
    for card in measured:
        assert card.fixed_mib and card.bytes_per_cell
        if len(card.anchors) >= 2:
            continue
        assert card.bytes_per_cell_provenance.strip(), (
            f"{card.key} fixes a two-term model from {len(card.anchors)} "
            "anchor(s) without saying what pinned the other term")
        assert any(abs(c.bytes_per_cell - card.bytes_per_cell) < 1.0
                   for c in separated), (
            f"{card.key}'s per-cell term matches no row that separated "
            "one of its own, so it came from nowhere checkable")


def test_the_measured_model_reproduces_its_own_anchors():
    """The instrument is checked against the answers it was fitted to."""

    checked = 0
    for card in mpas_mesh.load_sizing().cards.values():
        if not card.measured:
            continue
        for anchor in card.anchors:
            predicted = card.footprint_mib(anchor["cells"])
            # Each anchor states how close the model has to come.  The
            # readings taken off nvidia-smi against an idle baseline are
            # wider than the ones off the per-allocation ledger, and
            # holding both to the ledger's tolerance would be claiming a
            # precision the instrument did not have.
            tolerance = float(anchor.get("tolerance_mib", 1.0))
            assert abs(predicted - anchor["process_mib"]) <= tolerance, (
                f"{card.key}: the model reads {predicted:.1f} MiB at "
                f"{anchor['cells']} cells against a measured "
                f"{anchor['process_mib']:.1f} MiB")
            checked += 1
    assert checked >= 5, f"only {checked} anchors are pinned"


def test_the_fixed_term_is_derived_from_the_card_not_stored():
    """The defect: one card's fixed term quoted for every card.

    A 16 GiB owner was told 79,717 cells from a 170 SM part's 9,798 MiB
    fixed term.  The 70 SM part people actually own pays 5,384 MiB and
    holds 133,144 at the same budget.  The fixed term therefore has to
    come out of the card's own SM count and resident-thread capacity,
    not out of a stored number.
    """

    sizing = mpas_mesh.load_sizing()
    small = sizing.card("rtx-5070-ti")
    big = sizing.card("rtx-5090")

    # the derived store reproduces the measured null-launch readings
    assert abs(small.local_store_mib - 2896.0) < 5.0, small.local_store_mib
    assert abs(big.local_store_mib - 7034.0) < 5.0, big.local_store_mib

    # and it is where the difference between the two fixed terms lives
    gap = big.fixed_mib - small.fixed_mib
    store_gap = big.local_store_mib - small.local_store_mib
    assert store_gap > 0.9 * gap, (
        f"the derived store explains only {100 * store_gap / gap:.0f}% of "
        f"the {gap:.0f} MiB between the two measured fixed terms; below "
        "that the model has stopped being a derivation")

    # the numbers a user is handed, both directions
    assert sizing.cells_that_fit("rtx-5090", 16 * 1024.0) == 79_717
    assert sizing.cells_that_fit("rtx-5070-ti", 16 * 1024.0) == 133_144
    assert sizing.cells_that_fit("rtx-5090", 10 * 1024.0) == 5_349
    assert sizing.cells_that_fit("rtx-5070-ti", 10 * 1024.0) == 58_777


def test_the_rtx_pro_6000_is_a_measured_row():
    """`woof mesh --card rtx-pro-6000` was refused as an unknown card.

    The breakage: the 96 GiB part had no row, so the door refused it by
    name and its owner could only guess a --cells number.  Its row is a
    measured one (x1.40962 full physics on the card itself, 13,988 MiB,
    pinned build 629ddb6f0), and its fixed term is the 188 SM store the
    formula derives plus the residue that run left, not a borrowed one.
    """

    sizing = mpas_mesh.load_sizing()
    pro = sizing.card("rtx-pro-6000")
    assert pro.measured
    assert pro.streaming_multiprocessors == 188
    assert abs(pro.local_store_mib - 7777.0) < 5.0, pro.local_store_mib
    assert abs(pro.fixed_mib - 10603.8) < 1.0, pro.fixed_mib
    assert abs(pro.footprint_mib(40962) - 13988.0) < 1.0
    assert sizing.cells_that_fit("rtx-pro-6000") == 1_048_792
    assert sizing.cells_that_fit("rtx-pro-6000", 16 * 1024.0) == 69_963


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(("git", "-C", str(REPO_ROOT)) + args,
                          capture_output=True, text=True)


#: Where the MPAS port declares the ArWen build it is pinned to.  The
#: pin is LIVE state in the port checkout, not a record in this tree.
_PORT_PIN_SOURCE = Path("src") / "mpas_port" / "cuda_arwen_physics_v841.py"
_PORT_CHECKOUT_ENV = "WOOF_HEX_TREE"


def _port_checkout() -> Path | None:
    """Find the MPAS port checkout whose live pin sizes this door.

    ``WOOF_HEX_TREE`` names it outright; otherwise the port checkout
    is looked for as ``woof hex/tree`` beside any ancestor of this
    repository (which also finds it from a nested worktree).
    """

    override = os.environ.get(_PORT_CHECKOUT_ENV)
    if override:
        checkout = Path(override)
        return checkout if (checkout / _PORT_PIN_SOURCE).is_file() else None
    for ancestor in Path(REPO_ROOT).resolve().parents:
        checkout = ancestor / "woof hex" / "tree"
        if (checkout / _PORT_PIN_SOURCE).is_file():
            return checkout
    return None


def _port_live_arwen_pin(checkout: Path) -> str:
    source = (checkout / _PORT_PIN_SOURCE).read_text(encoding="utf-8")
    match = re.search(
        r'^ARWEN_BUILD_COMMIT\s*=\s*"([0-9a-f]{7,40})"', source, flags=re.M)
    assert match, (
        f"{checkout / _PORT_PIN_SOURCE} no longer declares "
        "ARWEN_BUILD_COMMIT, so the port's live pin cannot be read and "
        "the frame-staleness gate below has nothing to compare against")
    return match.group(1)


def test_the_frame_the_fixed_term_is_derived_from_cannot_go_stale():
    """The frame names a build, and the port's LIVE pin has not passed its cut.

    The whole fixed term is derived from ONE kernel frame, and that
    frame is a property of the BUILD.  The build in question is the
    WOOF checkout the MPAS port is pinned to -- NOT the tree this test
    runs in, which has already cut that frame from 22,416 B to under the
    default stack.  So the number is right and the label is what makes
    it checkable.

    The staleness comparison MUST use the port's live
    ``ARWEN_BUILD_COMMIT``.  Until 2026-08-25 this test compared the cut
    against the table's own frozen ``measured_against_arwen_commit``,
    which by construction predates the cut -- so the watchdog stayed
    green through two pin moves while the exact condition it names had
    occurred (stale-guard audit 2026-08-25, finding 2, ledger #347).

    The breakage, both directions, because the frame and the residues
    are one measurement set and each residue was obtained by
    SUBTRACTING the store this frame derives:

    * the pin advances to or past the cut (the #294 frame cut) and the
      frame does not: the store stays priced at 2,895.7 MiB on a 70 SM
      part when the real one is near zero, the fixed term comes out
      thousands of MiB high, and the door under-sizes -- the card is
      wasted quietly.
    * the frame is updated to a post-cut build and the residues are not:
      the derived store falls to 841.6 MiB, and 841.6 plus an unchanged
      2,488.3 MiB residue gives 3,330 MiB against the pinned build's
      real 5,384 -- 2,054 MiB LOW, so the door over-sizes and the run
      dies at step 1 on an out-of-memory that blames the model.

    Neither is visible in any output, which is why this is a gate.
    """

    store = mpas_mesh.load_sizing().cards["rtx-5090"].local_store
    recorded = store.measured_against_arwen_commit
    cut = store.frame_cut_commit
    assert recorded and cut, (
        "the frame block names no build, so the frame silently means "
        "whichever checkout the reader assumes")
    assert recorded != cut

    if _git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("not a git checkout, so no commit can be resolved")

    checkout = _port_checkout()
    if checkout is None:
        pytest.skip(
            "no MPAS port checkout (woof hex/tree beside an ancestor, "
            f"or ${_PORT_CHECKOUT_ENV}), so the port's live WOOF pin "
            "cannot be read; the frame-staleness condition is a property "
            "of that live pin and is UNCHECKED on this machine")
    pin = _port_live_arwen_pin(checkout)

    for name, rev in (("recorded build", recorded), ("frame cut", cut),
                      ("port's live pin", pin)):
        assert _git("cat-file", "-e", f"{rev}^{{commit}}").returncode == 0, (
            f"the {name} commit {rev} does not resolve in this "
            "repository, so nothing can check what build the footprint "
            "model describes")

    # The LIVE pin has to PREDATE the cut.  `--is-ancestor cut pin`
    # succeeds exactly when the pin is at or past the cut, which is the
    # stale case.
    assert _git("merge-base", "--is-ancestor", cut, pin).returncode != 0, (
        f"the MPAS port's live WOOF pin {pin} is at or past {cut}, the "
        "commit that cut the frame the footprint model derives its fixed "
        "term from.  Every residue in the sizing table was measured by "
        "subtracting a store computed from the PRE-cut frame, so the "
        "fixed terms are now over-priced by thousands of MiB and every "
        "mesh is under-sized.  Re-measure the frame AND every residue on "
        "the new pin, in one session, and move both here together")

    # And the frame is a pre-cut frame BECAUSE the cut is downstream of
    # the pin: the tree running this test descends from it.
    head = _git("rev-parse", "HEAD").stdout.strip()
    if _git("merge-base", "--is-ancestor", cut, head).returncode == 0:
        assert store.widest_kernel_frame_bytes > store.frame_credit_bytes * 8, (
            "this tree contains the frame cut, so a frame this small "
            "here would mean the table has quietly adopted the tree's "
            "frame while keeping residues measured against the pinned "
            "build's")


def test_every_residue_names_the_same_build_as_the_frame():
    """One measurement set, one build.

    A row re-measured on a newer checkout, sitting beside a frame from
    an older one, re-prices that card by thousands of MiB with nothing
    on screen to say so.
    """

    sizing = mpas_mesh.load_sizing()
    for card in sizing.cards.values():
        if not card.measured:
            assert card.residue_measured_against_arwen_commit is None, (
                f"{card.key} has no residue but names a build for one")
            continue
        assert (card.residue_measured_against_arwen_commit
                == card.local_store.measured_against_arwen_commit), (
            f"{card.key}'s residue was measured against "
            f"{card.residue_measured_against_arwen_commit} while the "
            "frame it was subtracted from belongs to "
            f"{card.local_store.measured_against_arwen_commit}; one of "
            "the two is stale and the fixed term derived from the pair "
            "is wrong by thousands of MiB")


def test_the_sizing_table_says_the_frame_is_not_this_trees_frame():
    """A number that means another checkout's build has to say so."""

    store = mpas_mesh.load_sizing().cards["rtx-5090"].local_store
    assert store.why_it_cannot_move_alone.strip(), (
        "the table does not say that the frame and the residues have to "
        "move together, which is the one thing a reader has to know "
        "before editing either")
    for expected in (store.measured_against_arwen_commit,
                     store.frame_cut_commit):
        assert expected in store.why_it_cannot_move_alone or expected in (
            mpas_mesh.SIZING_DATA_PATH.read_text(encoding="utf-8")), expected


def test_a_ten_gib_budget_is_told_what_the_card_actually_ran():
    """MEASURED: 40,962 cells completed inside 9,656.7 MiB free, peaking
    at 8,768.0 MiB, on the 70 SM part.  A sizing model that tells that
    owner 5,350 cells is 7.7x low against a run that happened."""

    card = mpas_mesh.load_sizing().card("rtx-5070-ti")
    peak = card.footprint_mib(40_962)
    assert abs(peak - 8768.0) <= 20.0, peak
    assert peak < 9656.7, (
        "the model does not reproduce the run completing inside the "
        "budget it was measured completing inside")
    assert card.cells_that_fit(10 * 1024.0) > 40_962


def test_an_unmeasured_card_is_refused_by_name():
    """Borrowing another card's fixed term is the silent wrong answer."""

    sizing = mpas_mesh.load_sizing()
    unmeasured = [c for c in sizing.cards.values() if not c.measured]
    assert unmeasured, (
        "the table declares no unmeasured card, so this refusal is "
        "unreachable and the door would quietly answer for cards nobody "
        "measured")
    with pytest.raises(mpas_mesh.MeshRequestError) as raised:
        sizing.cells_that_fit(unmeasured[0].key)
    message = str(raised.value)
    assert unmeasured[0].key in message
    assert "local-memory" in message or "NOT MEASURED" in message
    for card in sizing.cards.values():
        if card.measured:
            assert card.key in message, (
                "a refusal that does not say what WOULD work is a dead end")


def test_an_unknown_card_is_refused_and_lists_the_known_ones():
    sizing = mpas_mesh.load_sizing()
    with pytest.raises(mpas_mesh.MeshRequestError) as raised:
        sizing.cells_that_fit("not-a-card")
    assert "not-a-card" in str(raised.value)


def test_a_request_that_cannot_fit_the_named_card_is_refused_by_name():
    """The refusal states the card, the shortfall and what does fit."""

    sizing = mpas_mesh.load_sizing()
    card = next(c for c in sizing.cards.values() if c.measured)
    fits = sizing.cells_that_fit(card.key)
    with pytest.raises(mpas_mesh.MeshRequestError) as raised:
        sizing.check_fits(card.key, cells=fits * 4)
    message = str(raised.value)
    assert card.key in message
    assert f"{fits:,}" in message, (
        "the refusal has to say what WOULD fit, in cells")
    # And the one that does fit is not refused.
    sizing.check_fits(card.key, cells=fits)


# ---------------------------------------------------------------------------
# 6. the smoothness bound is DATA, and it moves
# ---------------------------------------------------------------------------

def test_the_smoothness_threshold_is_data_not_a_literal():
    document = json.loads(
        mpas_mesh.SIZING_DATA_PATH.read_text(encoding="utf-8"))
    bound = document["smoothness"]["refuse_above_percent_per_cell"]
    assert isinstance(bound, (int, float)) and bound > 0
    source = (REPO_ROOT / "woof" / "mpas_mesh.py").read_text(
        encoding="utf-8")
    assert f"= {bound}" not in source, (
        "the smoothness bound is a literal in the module; the dycore "
        "evidence that moves it would then be a code change")


def test_the_bound_declares_that_no_dycore_run_has_set_it_yet():
    """A provisional number that reads as measured is the worse lie."""

    smoothness = mpas_mesh.load_sizing().smoothness
    assert smoothness.status == "provisional"
    assert "NOT MEASURED" in smoothness.evidence
    assert smoothness.moves_when.strip()


def test_a_mesh_rougher_than_the_bound_is_refused_and_names_the_breakage():
    smoothness = mpas_mesh.load_sizing().smoothness
    rough = smoothness.refuse_above_percent_per_cell * 1.5
    with pytest.raises(mpas_mesh.MeshRoughnessError) as raised:
        smoothness.check(rough)
    message = str(raised.value)
    assert f"{rough:.2f}" in message
    assert f"{smoothness.refuse_above_percent_per_cell:.2f}" in message
    assert "%/cell" in message


def test_a_mesh_inside_the_warn_band_is_emitted_and_says_so():
    smoothness = mpas_mesh.load_sizing().smoothness
    middle = (smoothness.warn_above_percent_per_cell
              + smoothness.refuse_above_percent_per_cell) / 2.0
    note = smoothness.check(middle)
    assert note is not None and "NOT MEASURED" in note


def test_a_mesh_at_the_published_reference_is_silent():
    smoothness = mpas_mesh.load_sizing().smoothness
    assert smoothness.check(
        smoothness.published_reference_percent_per_cell) is None


def test_the_threshold_moves_when_the_table_moves(tmp_path):
    """Proof it is data: a different table gives a different verdict."""

    document = json.loads(
        mpas_mesh.SIZING_DATA_PATH.read_text(encoding="utf-8"))
    rough = document["smoothness"]["refuse_above_percent_per_cell"] * 1.5
    document["smoothness"]["refuse_above_percent_per_cell"] = rough * 2
    moved = tmp_path / "moved.json"
    moved.write_text(json.dumps(document), encoding="utf-8")
    assert mpas_mesh.load_sizing(moved).smoothness.check(rough) is not None


def test_the_roughness_override_is_reported_as_a_workaround():
    """Fixed means default; an opt-out is named as what it is."""

    smoothness = mpas_mesh.load_sizing().smoothness
    rough = smoothness.refuse_above_percent_per_cell * 1.5
    note = smoothness.check(rough, allow_rough=True)
    assert note is not None
    assert "WORKAROUND" in note


def test_the_delivered_metric_is_not_judged_against_the_requested_bound():
    """MEASURED 2026-08-20, and it is why this test exists.

    The first cut of this door compared the delivered
    ``max_adjacent_spacing_ratio`` against the same %/cell bound the
    REQUEST is gated on, and the two are not the same quantity.  The
    requested gradient is the slope of a smooth sampled density field;
    the delivered ratio is a max over every edge in the mesh and is
    dominated by the SCVT's own hexagon/pentagon texture.  A control run
    settles it: a UNIFORM 12,000-cell mesh, requested gradient
    0.00 %/cell, delivers ``max_adjacent_spacing_ratio`` 1.1407 --
    14.07 %/cell on the requested scale.  So the check fired on every
    mesh ever emitted, including one with no refinement in it at all.

    The table therefore carries the measured control and declares NO
    bound on the delivered metric, and the door reports it rather than
    judging it.
    """

    delivered = mpas_mesh.load_sizing().delivered
    assert delivered.bound is None, (
        "the delivered adjacent-spacing metric has a bound in the table, "
        "but no run has established one -- and the uniform-mesh control "
        "shows the metric reads 14.07 %/cell with zero refinement, so any "
        "bound borrowed from the requested-gradient scale refuses every "
        "mesh")
    assert "NOT MEASURED" in delivered.evidence
    assert delivered.uniform_control_ratio > 1.0, (
        "the control that proves the metric is not the requested gradient "
        "has to be in the table, or the next reader re-derives it")


def test_the_delivered_reading_is_reported_with_its_control():
    """A number with no scale beside it is the wrong verdict waiting."""

    delivered = mpas_mesh.load_sizing().delivered
    note = delivered.report(1.1590)
    assert "15.90" in note
    assert "14.07" in note or f"{delivered.uniform_control_percent:.2f}" in note
    assert "NOT MEASURED" in note


# ---------------------------------------------------------------------------
# 7. the door a user types
# ---------------------------------------------------------------------------

def test_gpuwm_mesh_is_a_registered_subcommand():
    """Engine-proven is not shipped: this is the front door."""

    from woof.cli import build_parser

    actions = [a for a in build_parser()._actions
               if hasattr(a, "choices") and isinstance(a.choices, dict)]
    names = set()
    for action in actions:
        names.update(action.choices)
    assert "mesh" in names, (
        "`woof mesh` is not registered, so the mesh generator has no "
        "command a user can type")


def test_the_door_takes_where_how_fine_how_coarse_and_which_card():
    parser = mpas_mesh.build_parser()
    flags = {option for action in parser._actions
             for option in action.option_strings}
    for flag in ("--refine", "--background-km", "--card", "--out",
                 "--dry-run", "--spec", "--cells", "--list-cards",
                 "--allow-rough-mesh"):
        assert flag in flags, f"the mesh door has no {flag}"


def test_the_refine_grammar_is_parsed_into_a_spec_row():
    spec = mpas_mesh.build_spec(background_km=120.0,
                                refine=["39.0,-98.0,1200,15"])
    region, = spec["regions"]
    assert region["shape"] == {"kind": "cap",
                               "center_deg": [39.0, -98.0],
                               "radius_km": 1200.0}
    assert region["spacing_km"] == 15.0
    assert "transition_cells" in region or "transition_km" in region


def test_a_second_region_is_a_second_row_and_not_a_second_code_path():
    """The arbitrary acceptance test, at this door."""

    spec = mpas_mesh.build_spec(
        background_km=120.0,
        refine=["39.0,-98.0,1200,15", "-33.9,151.2,600,8"])
    assert len(spec["regions"]) == 2


def test_a_malformed_refine_row_is_refused_by_name():
    with pytest.raises(mpas_mesh.MeshRequestError) as raised:
        mpas_mesh.build_spec(background_km=120.0, refine=["39.0,-98.0"])
    assert "--refine" in str(raised.value)


def test_a_finer_background_than_the_refinement_is_refused():
    """"How coarse elsewhere" that is finer than "how fine there"."""

    with pytest.raises(mpas_mesh.MeshRequestError) as raised:
        mpas_mesh.build_spec(background_km=10.0,
                             refine=["39.0,-98.0,1200,15"])
    assert "background" in str(raised.value)


# ---------------------------------------------------------------------------
# 8. doctor reports it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['rw_mpas_convert'])
def test_doctor_names_the_door_each_binary_unlocks(name):
    """A bundled artifact no check reports is a silent-green estate."""

    from woof import doctor

    assert name in doctor._CHECKED_ARTIFACTS, (
        f"{name} is in BUNDLED_ARTIFACTS and no doctor check names it, "
        "so the estate reads green on a box where the mesh route is dead")


def test_doctor_emits_a_line_per_mpas_binary():
    from woof import doctor

    checks = doctor._mpas_bridge_checks()
    reported = {check.name for check in checks}
    for name in MPAS_ARTIFACTS:
        assert any(name in line for line in reported), (
            f"woof doctor prints no line for {name}")


# ---------------------------------------------------------------------------
# 9. the generated CLI reference
# ---------------------------------------------------------------------------

def test_the_generated_cli_reference_carries_the_mesh_door():
    page = REPO_ROOT / "docs" / "public" / "CLI-OPTIONS.md"
    if not page.is_file():                       # pragma: no cover - wheel
        pytest.skip("the generated reference is not in this install")
    text = page.read_text(encoding="utf-8")
    assert "woof mesh" in text, (
        "the generated CLI reference does not name `woof mesh`; "
        "regenerate it with python -m tools.build_cli_options_doc")
    assert "--refine" in text and "--card" in text


# ---------------------------------------------------------------------------
# 10. against the artifact
# ---------------------------------------------------------------------------

def _mesh_binary() -> Path | None:
    try:
        return mpas_mesh.BRIDGES["rw_mpas_mesh"].find()
    except FileNotFoundError:
        return None


@pytest.mark.skipif(_mesh_binary() is None,
                    reason="rw_mpas_mesh is not built or staged here")
def test_the_real_binary_answers_the_pinned_contract():
    """Run the exe.  The marker is a claim about bytes on this disk."""

    binary = _mesh_binary()
    result = subprocess.run([str(binary), "--abi"], capture_output=True,
                            text=True, errors="replace", timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == mpas_mesh.MESH_ABI_MARKER


@pytest.mark.skipif(_mesh_binary() is None,
                    reason="rw_mpas_mesh is not built or staged here")
def test_the_door_plans_a_real_request_through_the_real_binary(tmp_path):
    plan = mpas_mesh.plan(
        spec=mpas_mesh.build_spec(background_km=480.0,
                                  refine=["39.0,-98.0,1200,120"]),
        cells=2000, workdir=tmp_path)
    assert plan["dry_run"] is True
    assert plan["target_cells"] == 2000
    assert plan["steepest_requested_gradient_percent_per_cell"] >= 0.0
    # The repaired meter's contract: a plan must SAY its probe coverage,
    # and the door refuses anything but "complete" (see
    # test_the_door_refuses_a_plan_without_complete_gradient_coverage).
    assert plan["gradient_probe_coverage"] == "complete"


@pytest.mark.skipif(_mesh_binary() is None,
                    reason="rw_mpas_mesh is not built or staged here")
def test_the_static_gap_is_stated_by_the_binary_itself(tmp_path):
    """The deliverable boundary is not a docstring claim."""

    plan = mpas_mesh.plan(
        spec=mpas_mesh.build_spec(background_km=480.0, refine=[]),
        cells=2000, workdir=tmp_path)
    assert "static" in plan["deliverable_boundary"]


def test_the_door_refuses_a_plan_without_complete_gradient_coverage():
    """A missing measurement must never become the friendliest verdict.

    The engine's receipt contract (rw-mpas mesh/mod.rs) says a consumer
    refuses a plan whose gradient_probe_coverage is missing or not
    "complete".  This door used to read the gradient with .get(..., 0.0)
    and never read the coverage word at all -- the exact shape of the
    lattice blindness the meter repair closed: a spec truly at 65.2 %/cell
    read 8.3 and was admitted at 7.3x the build ceiling.
    """

    complete = {"steepest_requested_gradient_percent_per_cell": 4.25,
                "gradient_probe_coverage": "complete"}
    assert mpas_mesh.gradient_gate_reading(complete) == 4.25

    with pytest.raises(mpas_mesh.MeshRequestError) as caught:
        mpas_mesh.gradient_gate_reading(
            {"gradient_probe_coverage": "complete"})
    assert "will not invent a 0.0" in str(caught.value)

    for coverage in ("partial", None):
        plan_doc = {"steepest_requested_gradient_percent_per_cell": 0.0}
        if coverage is not None:
            plan_doc["gradient_probe_coverage"] = coverage
        with pytest.raises(mpas_mesh.MeshRequestError) as caught:
            mpas_mesh.gradient_gate_reading(plan_doc)
        assert "not a measurement of this spec's field" in str(caught.value)
        assert repr(coverage) in str(caught.value)


# ---------------------------------------------------------------------------
# 11. the triangulation arm: a capability with a front door, and a class
#     boundary a caller cannot cross by accident
# ---------------------------------------------------------------------------

@pytest.fixture
def generate_argv(tmp_path, monkeypatch):
    calls = []

    def fake_run(arguments, progress=None):
        calls.append(list(arguments))
        return {"receipt": {}}

    monkeypatch.setattr(mpas_mesh, "_run_mesh", fake_run)

    def generate(**options):
        mpas_mesh.generate(spec=mpas_mesh.build_spec(background_km=120.0, refine=[]),
                           cells=2562, out=tmp_path / "x.nc", workdir=tmp_path,
                           **options)
        return calls

    return generate


def test_the_default_request_names_no_triangulation_arm(generate_argv):
    """THE BREAKAGE: the default arm is what every registered mesh digest
    was minted on.  If the door started sending `--triangulation
    incremental` by default -- or sent `rebuild` explicitly to a binary
    that does not know the flag -- a bare `woof mesh` would stop
    reproducing the published x1 and x4 grids and every registry pin that
    names them would lapse with nothing in the run saying why."""

    calls = generate_argv()
    assert "--triangulation" not in calls[0], (
        "the door sent a triangulation arm on a request that did not ask "
        f"for one: {calls[0]}")


def test_asking_for_the_fast_arm_passes_it_through_by_name(generate_argv):
    """A capability the door cannot reach is not shipped."""

    calls = generate_argv(triangulation="incremental")
    argv = calls[0]
    assert argv[argv.index("--triangulation") + 1] == "incremental"


def test_an_unknown_triangulation_arm_is_refused_by_name(tmp_path):
    """A misspelt arm must not fall back to the other one silently: one
    direction wastes the hours the caller was avoiding, the other pins a
    digest against an arm that never ran."""

    with pytest.raises(mpas_mesh.MeshRequestError) as caught:
        mpas_mesh.generate(spec=mpas_mesh.build_spec(background_km=120.0,
                                                     refine=[]),
                           cells=2562, out=tmp_path / "x.nc",
                           workdir=tmp_path, triangulation="fast")
    text = str(caught.value)
    assert "rebuild" in text and "incremental" in text and "registered" in text


def test_the_cli_offers_the_arm_and_defaults_to_the_registered_one():
    """`woof mesh` is the front door; the flag has to exist on it."""

    parser = mpas_mesh.build_parser()
    args = parser.parse_args(["--out", "g.nc", "--background-km", "120",
                              "--cells", "2562"])
    assert args.triangulation is None
    args = parser.parse_args(["--out", "g.nc", "--background-km", "120",
                              "--cells", "2562",
                              "--triangulation", "incremental"])
    assert args.triangulation == "incremental"
    with pytest.raises(SystemExit):
        parser.parse_args(["--out", "g.nc", "--background-km", "120",
                           "--cells", "2562", "--triangulation", "fast"])


@pytest.mark.skipif(_mesh_binary() is None,
                    reason="rw_mpas_mesh is not built or staged here")
def test_the_real_binary_knows_both_arms_and_refuses_a_third():
    """Run the exe.  The arm list is a claim about bytes on this disk, and
    a build predating the flag would take `incremental` as an unknown
    option and generate on the rebuild arm anyway."""

    binary = _mesh_binary()
    for arm in mpas_mesh.TRIANGULATION_ARMS:
        result = subprocess.run(
            [str(binary), "--background-km", "480", "--cells", "162",
             "--triangulation", arm, "--dry-run"],
            capture_output=True, text=True, errors="replace", timeout=120)
        assert result.returncode == 0, f"{arm}: {result.stderr}"
    result = subprocess.run(
        [str(binary), "--background-km", "480", "--cells", "162",
         "--triangulation", "fast", "--dry-run"],
        capture_output=True, text=True, errors="replace", timeout=120)
    assert result.returncode != 0
    assert "rebuild" in result.stderr and "incremental" in result.stderr


@pytest.mark.skipif(_mesh_binary() is None,
                    reason="rw_mpas_mesh is not built or staged here")
def test_the_dry_run_still_prints_the_record_alone_on_either_arm():
    """THE BREAKAGE THIS RE-ARMS: `woof.mpas_mesh` and `woof.hex.swath` both
    parse the WHOLE of a dry run's stdout as JSON, so any progress line above
    the record is a JSONDecodeError at line 1 column 1 rather than a message
    anybody reads -- a defect this repo has already fixed once.  The
    incremental arm prints a `TRIANGULATION` progress line naming its class,
    and that line must not reach the dry run.  Asserted through
    `json.loads` on raw stdout, which is exactly what the consumers do."""

    binary = _mesh_binary()
    for extra in ([], ["--triangulation", "incremental"],
                  ["--triangulation", "rebuild"]):
        result = subprocess.run(
            [str(binary), "--background-km", "480", "--cells", "2000",
             "--dry-run", *extra],
            capture_output=True, text=True, errors="replace", timeout=120)
        assert result.returncode == 0, f"{extra}: {result.stderr}"
        plan = json.loads(result.stdout)
        assert plan["dry_run"] is True, extra
