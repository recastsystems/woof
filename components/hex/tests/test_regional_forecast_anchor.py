"""The earned regional anchor, and the memory model the forecast runs in.

CPU-testable by design.  The numbers are proved on hardware by
``tools/run_cuda_regional_forecast.py``; what these tests hold is the shape
of the claim: that the anchor names evidence which EXISTS, that the two runs
it names actually agree frame by frame, that an unregistered regional
configuration still refuses by name, and that the padded memory model the
device forecast runs in reproduces the CPU authority's conventions element
for element.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from woof.hex import regional_v841
from woof.hex.cuda_backend import regional_admission
from woof.hex.cuda_regional_forecast_v841 import (
    REGIONAL_GARBAGE_POOL,
    PaddedRegionalHostMesh,
)
from _layout import package_path

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_ROW = "conus-x1.2971"


# ---------------------------------------------------------------------------
# the anchor names evidence that exists, and the evidence says what it claims
# ---------------------------------------------------------------------------


def test_the_registry_holds_the_earned_row():
    anchor = regional_admission.admitted_region(ANCHOR_ROW)
    assert anchor is not None
    assert anchor.n_cells == 2_971
    assert anchor.boundary_zone_width == regional_v841.N_BDY_ZONE
    assert len(anchor.bdy_mask_sha256) == 64


def test_both_halves_of_the_anchor_are_files_in_this_repository(receipts):
    """An anchor that names evidence which is not here proves nothing."""

    anchor = regional_admission.admitted_region(ANCHOR_ROW)
    assert anchor is not None
    for relative in (anchor.contract_receipt, anchor.forecast_anchor):
        assert (ROOT / relative).is_file(), relative


def test_the_named_forecast_pair_is_masked_digest_identical(receipts):
    """The claim the row makes, re-checked against the receipts it names."""

    anchor = regional_admission.admitted_region(ANCHOR_ROW)
    assert anchor is not None
    base = ROOT / anchor.forecast_anchor
    runs = [
        json.loads((base.parent.parent / name / "forecast.json").read_text())
        for name in ("mint-run1", "mint-run2", "run1", "run2")
    ]
    # Two runs predate the row (gate bypassed -- that is what earned it) and
    # two went through the gate afterwards.  A row whose evidence was ONLY
    # produced through its own gate would be circular.
    assert [run["minted_without_anchor_gate"] for run in runs] == [
        True, True, False, False
    ]
    first = runs[0]
    assert first["frames"] and first["n_cells_solve"] == anchor.n_cells
    whole_file_collisions = 0
    for other in runs[1:]:
        assert len(other["frames"]) == len(first["frames"])
        assert other["config_sha256"] == first["config_sha256"]
        for left, right in zip(first["frames"], other["frames"]):
            assert left["step"] == right["step"]
            assert left["masked_sha256"] == right["masked_sha256"]
            assert left["field_sha256"] == right["field_sha256"]
            assert all(left["finite"].values())
            whole_file_collisions += int(left["sha256"] == right["sha256"])
    # Every container differs: each run stamps its own file_id, so a
    # whole-file digest that AGREED would mean a run never actually ran.
    assert whole_file_collisions == 0


#: Sources that have MOVED since the regional campaign ran, with the reason
#: and the argument that the move cannot change what those receipts measured.
#: An archived receipt is evidence and is never rewritten, so a legitimate
#: later edit to a recorded source has to be declared here rather than
#: papered over -- and a drift with no entry is refused by name below, which
#: is the breakage this gate prevents: an anchor quietly resting on bytes
#: nobody can account for.
SOURCE_DRIFT_SINCE_CAMPAIGN: dict[str, dict[str, str]] = {
    "src/hexcore/cuda_driver.py": {
        "recorded": "cb6477ef2fd02ad551e9b6c8af0724c94fc4d90bdc712fff1378d218662f5d04",
        "current": "e8d3c8484928c85f126dfce6bacb35ba215cf59dc7005ded780b96aa639f0f42",
        "why": (
            "convection ruling, 2026-08-26: _v841_physics_cadences reports "
            "convection: None when no cumulus scheme is selected instead of "
            "calling float() on it.  Every configuration these four receipts "
            "recorded selects Grell-Freitas, and on that branch the function "
            "returns a byte-identical table -- checked below rather than "
            "asserted -- so the receipts' numbers are unaffected and only the "
            "file digest moved.  Moved again by the limited-area physics "
            "lane, 2026-08-26: regional_bdy_mask_digest now delegates to the "
            "mesh contract's one definition instead of carrying a second "
            "convention.  These four receipts resolved their anchor by "
            "DIGEST, and the two rows in regional_admission were re-minted "
            "onto the surviving spelling in the same commit, so the same "
            "arrays still resolve the same row.  Moved a third time by the "
            "provenance scrub, 2026-08-27: three comment and "
            "docstring lines naming a person and a machine were rewritten "
            "so that the public assembly stops having to move bytes that "
            "sit under a digest pin.  Measured inert rather than asserted: "
            "the pre- and post-scrub bytes parse to the identical AST once "
            "docstrings are stripped and every string constant is "
            "normalised, and the string-constant count is unchanged, so no "
            "number, branch or name in this file moved"
            "  Moved again 2026-08-28 by the 0.2.0 package rename: this "
            "file is byte-identical to its pre-rename self once the token "
            "mpas_port is substituted with hexcore, so nothing an archived "
            "receipt measured can differ.  Re-derived by "
            "tools/repin_source_tables.py"
            "  Moved again 2026-10-13 by the forecast batch: the host mesh "
            "validation's row loops became whole-table checks with the same "
            "verdict and message (tests/test_mesh_rows_vectorized.py keeps "
            "the loops as the reference); it runs on the host before any CUDA "
            "interaction and no CUDA source, launch or arithmetic line moved"
            "  Moved again 2026-10-13 by landing wave 2 (SG2): the lagged "
            "microphysics heating carry (e7be214b) adds one kernel and three "
            "recover_cells_f32 arguments, both inactive unless physics hands "
            "the dycore a diabatic_theta; nothing these receipts ran supplies "
            "one, so the arithmetic they measured is unchanged"
            "  Moved again 2026-10-14 by the 0.4.0 release scrub: one comment "
            "line names an owner ruling in place of a person; comment only, no "
            "code line moved"
            "  Moved again 2026-10-15 by rel/hex040-fix-les: the RK step calls "
            "MPAS-A v8.4.2's LES dissipation only when a driver is handed an "
            "LES launcher (config_les_model or config_mix_scalars on the MPAS "
            "route); the moist w Euler kernel gains a has_mixing flag that is "
            "0 otherwise and subtracts from 0.0f as before. These receipts ran "
            "with LES off, so the arithmetic they measured is unchanged"
        ),
    },
    "src/hexcore/cuda_regional_forecast_v841.py": {
        "recorded": "3aa30ec3c3a11c1152a67d1fcbc8b2d470fd8ac2446ec8f7ec34906873421e88",
        "current": "9a87cd6b3456c2edf2e290de70b9eee8b4b0235d1102c11b80fab154f912a9c2",
        "why": (
            "Integration 2026-10-04 adds explicit rank constructor dispatch "
            "and preserves the ordinary constructor. The twelve arithmetic "
            "source files remain exact e6 bytes. The current source set passed "
            "eight native point decks and all twenty-two kernels in "
            "verification/hex-next/native-point.json. Archived source hashes "
            "and class measurements remain unchanged. "
            "ANCHOR RE-KEYING, 2026-08-27: open_regional_forecast_v841 now "
            "measures the configuration CLASS off the run it is about to "
            "admit -- boundary-zone width, column count, min(dcEdge) and the "
            "timestep -- and hands it to require_regional_anchor, and two "
            "pure readers were added for the first two of those.  It cannot "
            "change what these receipts measured, by construction rather "
            "than by argument: every one of them was produced by "
            "--mint-anchor, which calls assemble_regional_driver_v841 and "
            "never enters open_regional_forecast_v841 at all.  The assembler "
            "is byte-unchanged.  Before that, limited-area physics lane, "
            "2026-08-26: PaddedRegionalHostMesh "
            "grew what the WOOF physics seam reads and the dycore does not "
            "-- coeffs_reconstruct and edgeNormalVectors, which DeviceMesh "
            "has no field for and never uploads -- and it restored the "
            "loader's canonical -1 in the INACTIVE edgesOnCell slots, the "
            "ones past nEdgesOnCell.  Every ACTIVE connectivity entry, every "
            "geometry pad and every zone mask is bit-identical, which is "
            "checked below rather than asserted, and no kernel on either "
            "side reads an inactive slot: both loop slot < nEdgesOnCell.  "
            "The module also gained pad_regional_physics_host and a driven-"
            "species count for the boundary scalar nudge.  Neither touches "
            "what these four dry-dycore receipts ran: they carry ONE passive "
            "qv and their stream carries one, so the driven count and the "
            "model count agree exactly as they always did, and the physics "
            "pad is never reached without a physics stack"
            "  Moved again 2026-08-28 by the 0.2.0 package rename: this "
            "file is byte-identical to its pre-rename self once the token "
            "mpas_port is substituted with hexcore, so nothing an archived "
            "receipt measured can differ.  Re-derived by "
            "tools/repin_source_tables.py"
            "  Moved again 2026-10-14 by the 0.4.0 release scrub: one docstring "
            "line of open_regional_forecast_v841 names an owner ruling in place "
            "of a person; docstring only, no code line moved"
        ),
    },
    "src/hexcore/regional_v841.py": {
        "recorded": "e74d345e3a2e02fe6887840e7ad3a8166ea302082e6a505d138c5293fef69a43",
        "current": "84b3b62a8c7d6ad629f99ed89890384668aee1636be0cdc53831f7a634a4b44d",
        "why": (
            "provenance scrub, 2026-08-27: ONE line of the module "
            "docstring named the machine the native v8.4.1 sources were "
            "transcribed against.  The line now names the rig by its role "
            "rather than by its host, and every source path, file and line "
            "range it cites is unchanged.  Nothing else in the file moved: "
            "the pre- and post-scrub bytes parse to the identical AST once "
            "docstrings are stripped and every string constant is "
            "normalised, and the string-constant count is unchanged, so no "
            "number, branch or message in this module differs.  This file "
            "is also one of the fourteen the regional kernel-set digest "
            "hashes, which is why the two shipped classes are RE-MINTED in "
            "the same commit rather than re-pointed"
            "  Moved again 2026-08-28 by the 0.2.0 package rename: this "
            "file is byte-identical to its pre-rename self once the token "
            "mpas_port is substituted with hexcore, so nothing an archived "
            "receipt measured can differ.  Re-derived by "
            "tools/repin_source_tables.py"
        ),
    },
    "src/hexcore/cuda_backend/runtime.py": {
        "recorded": "4deab4b3a37467491a531750e8618705c6884c19c0395891bc426fce2ea6ba5a",
        "current": "0a25dbf274d4b502542c40c7dc30d223324e9d12f7b87b2caa7bc161e7bdb81b",
        "why": (
            "2026-10-15: require_cuda's default runtime floor 13000 -> 12000 "
            "(CUDA 12 support) and one docstring paragraph; host Python "
            "only, no CUDA source string or arithmetic moved. "
            "Integration 2026-10-04 adds explicit current-device selection "
            "for rank ownership. No CUDA arithmetic changes. The current "
            "source set passed the native point contract recorded in "
            "verification/hex-next/native-point.json; other class records "
            "are carried and are not newly measured. "
            "limited-area physics lane, 2026-08-26: the per-launch observer "
            "now carries the MODULE KEY that launched, as well as the kernel "
            "name.  One KernelCache serves the dycore AND the WOOF physics "
            "seam, and the regional garbage discipline was rewriting the "
            "padded column of every float32 argument of every physics launch "
            "-- including the source state arrays it was handed.  It skips "
            "the four self-managing modules now.  For these four receipts "
            "nothing changed at all: they are DRY-DYCORE runs with no "
            "physics stack attached, so no launch they made carries a "
            "self-managing module key and every one of them is scrubbed "
            "exactly as before"
            "  The 0.2.0 package rename, 2026-08-28, did NOT move this "
            "file: it carries the token nowhere at all, so its digest is "
            "the same string it was before the rename and this row is "
            "unchanged by it"
        ),
    },
    "tools/run_cuda_regional_forecast.py": {
        "recorded": "0829a0d11501388c974f267d85f42adea39e762884948b36304d6d4f8ed64167",
        "current": "f0d139d867a24c8b65d2e30cf1dec1a8d82b0705b321a457dda712a960465f80",
        "why": (
            "swath-as-lam lane, 2026-08-27: the instrument grew named "
            "--grid/--init/--lbc-dir/--start-time/--dt/--len-disp paths and a "
            "--global-control arm, so a cull of a mesh this program placed "
            "for itself can be run without mirroring the 2026-08-25 record "
            "set's directory shape.  Every one of the four archived receipts "
            "was produced by `--arm x1`, which takes none of the new "
            "arguments: the reference-dir branch resolves the same four "
            "paths it always did, and dry_reference_config() called with no "
            "arguments returns the CANDIDATE-REGIONAL-DRY namelist "
            "field-for-field -- checked below rather than asserted"
            "  Moved again 2026-08-28 by the 0.2.0 package rename: this "
            "file is byte-identical to its pre-rename self once the token "
            "mpas_port is substituted with hexcore, so nothing an archived "
            "receipt measured can differ.  Re-derived by "
            "tools/repin_source_tables.py"
            "  Moved again 2026-10-05: the "
            "instrument grew --invariant (the native invariant stream, read "
            "for the vertical grid as MPAS-A does) and now loads its init as "
            "a cold start, recomputing pressure_p/pressure_base/exner as "
            "atm_init_coupled_diagnostics does instead of consuming the init "
            "file's copies.  The four archived x1 receipts were minted on the "
            "file copies; --file-coupled-diagnostics reproduces that load "
            "exactly (no other argument changed), and each new receipt "
            "records which load it used as coupled_diagnostics"
        ),
    },
    "src/hexcore/cuda_acoustic_v841.py": {
        "recorded": "fc5059cfc3fc894c60371caf0d8b89f309605152360a18f983b31e4d32cb6ed2",
        "current": "fd80600d3f1c069d4283dd526ce172eb932242e5211a446a1e0addf3575521ed",
        "why": (
            "2026-10-19 LTS hook repair: unspecialized lookups retain their "
            "two-part key; the acoustic tap uses the launched shape identity. "
            "CUDA arithmetic and archived recorded hashes are unchanged. "
            "2026-10-19 current-source refresh after f65ab11c: immutable "
            "kernel dimensions are specialized and cache keys include shape. "
            "The archived recorded hash and receipt scope stay unchanged; "
            "this is not a new forecast mint. Earlier migrations follow. "
            "0.2.0 package rename, 2026-08-28: the token mpas_port became "
            "hexcore. This file is byte-identical to the copy these receipts "
            "ran once that one identifier is substituted -- measured file "
            "against file, not argued from a diff -- and the 1 line carrying "
            "it is 1 NVRTC module_key or module-name label. No line of CUDA "
            "source text differs, so every kernel these receipts launched "
            "compiles from the same bytes; a module_key names a cache slot "
            "and an observer label, never anything the arithmetic reads. "
            "Re-derived by tools/repin_source_tables.py."
        ),
    },
    "src/hexcore/cuda_backend/containers.py": {
        "recorded": "02b7b2bee1779a5072df1c3f79fa82ff0d71ccc047e9edd05c09ffc6f0967ec1",
        "current": "3a071300d7d7bbd439220fc905659e2eb259e312896500ebb7680ca2eac93d71",
        "why": (
            "Package rename and compact host copies were followed by the "
            "2026-10-17 startup fix: finite checks use bounded host blocks; "
            "edge signs are constructed on device from uploaded indices. "
            "The old scalar loop and the new kernel have identical words "
            "for float32/int32 and float64/int64, including inactive negative "
            "sentinels (tests/test_mesh_signs_cuda.py). Later startup changes "
            "pack host fields in bounded blocks, check finite words on CUDA "
            "and overlap independent atmosphere components on private streams "
            "(tests/test_device_field_upload.py). This source refresh "
            "does not re-run or replace the archived regional forecast proof."
        ),
    },
    "src/hexcore/cuda_dynamics_v841.py": {
        "recorded": "d050cabd1a1d199f014de0306f2987a68fb3184bf45291c57dafdb528569e9e7",
        "current": "d3908f72339816d83cb9f03cf633b1f06fffa264758bdc7a1ef451c15a952beb",
        "why": (
            "2026-10-19 current-source refresh after f65ab11c: immutable "
            "kernel dimensions are specialized and cache keys include shape. "
            "The archived recorded hash and receipt scope stay unchanged; "
            "this is not a new forecast mint. Earlier migrations follow. "
            "The recorded digest retains the archived campaign's source. "
            "Current tracks subsequent source edits, including the "
            "2026-10-16 corrected-theta stencil reuse (hex/speed-1). "
            "Its evidence is the direct old/new uint32 test and the new "
            "G1/G2 landing replay, not the archived campaign. "
            "Re-derived by tools/repin_source_tables.py."
        ),
    },
    "src/hexcore/cuda_horizontal.py": {
        "recorded": "97faf0869a0a5ea9ebbc4c67b3c2d6c68cefdfa10dece73cd204d818962efde4",
        "current": "f94137176db65d24df0d9ff1d491c278d6c835bb08cd76dc43c5cfcbf323a345",
        "why": (
            "0.2.0 package rename, 2026-08-28: the token mpas_port became "
            "hexcore. This file is byte-identical to the copy these receipts "
            "ran once that one identifier is substituted -- measured file "
            "against file, not argued from a diff -- and the 3 lines carrying "
            "it are 2 comment or docstring lines, 1 NVRTC module_key or "
            "module-name label. No line of CUDA source text differs, so every "
            "kernel these receipts launched compiles from the same bytes; a "
            "module_key names a cache slot and an observer label, never "
            "anything the arithmetic reads. Re-derived by "
            "tools/repin_source_tables.py."
        ),
    },
    "src/hexcore/cuda_horizontal_v841.py": {
        "recorded": "3fc0b860ebd67dfed453617c348810964ea1110e782fe85db10283afb406e2fe",
        "current": "37714ab89935af5b02d79eeea4c06cfad038de457502116c9dad27a8bdd55465",
        "why": (
            "2026-10-19 current-source refresh after f65ab11c: immutable "
            "kernel dimensions are specialized and cache keys include shape. "
            "The archived recorded hash and receipt scope stay unchanged; "
            "this is not a new forecast mint. Earlier migrations follow. "
            "0.2.0 package rename, 2026-08-28: the token mpas_port became "
            "hexcore. This file is byte-identical to the copy these receipts "
            "ran once that one identifier is substituted -- measured file "
            "against file, not argued from a diff -- and the 3 lines carrying "
            "it are 2 comment or docstring lines, 1 NVRTC module_key or "
            "module-name label. No line of CUDA source text differs, so every "
            "kernel these receipts launched compiles from the same bytes; a "
            "module_key names a cache slot and an observer label, never "
            "anything the arithmetic reads. Re-derived by "
            "tools/repin_source_tables.py."
        ),
    },
    "src/hexcore/cuda_regional_v841.py": {
        "recorded": "b9f1dbdbbd123066f1fec2ebebbb664da7b4d5b62bce06a15f42dc96193a4ab5",
        "current": "b7692e79ddc1297b8f7ca1b0d05e2bf545a103b8e5ffbfe067e315074c7674da",
        "why": (
            "0.2.0 package rename, 2026-08-28: the token mpas_port became "
            "hexcore. This file is byte-identical to the copy these receipts "
            "ran once that one identifier is substituted -- measured file "
            "against file, not argued from a diff -- and the 7 lines carrying "
            "it are 6 comment or docstring lines, 1 NVRTC module_key or "
            "module-name label. No line of CUDA source text differs, so every "
            "kernel these receipts launched compiles from the same bytes; a "
            "module_key names a cache slot and an observer label, never "
            "anything the arithmetic reads. Re-derived by "
            "tools/repin_source_tables.py."
        ),
    },
    "src/hexcore/cuda_transport.py": {
        "recorded": "44c1147fe06fdcd2eacb6198260918c02670297108e9533e850a239180ced850",
        "current": "013ed46f8ef3bcc1a445b2e81135e262c23c49950e52e5b62b22e4274644b15e",
        "why": (
            "0.2.0 package rename, 2026-08-28: the token mpas_port became "
            "hexcore. This file is byte-identical to the copy these receipts "
            "ran once that one identifier is substituted -- measured file "
            "against file, not argued from a diff -- and the 1 line carrying "
            "it is 1 NVRTC module_key or module-name label. No line of CUDA "
            "source text differs, so every kernel these receipts launched "
            "compiles from the same bytes; a module_key names a cache slot "
            "and an observer label, never anything the arithmetic reads. "
            "Re-derived by tools/repin_source_tables.py."
        ),
    },
    "src/hexcore/cuda_transport_v841.py": {
        "recorded": "55c66759d9c81f65ed71ce77570897c102fd64661da6ad6c37b438b27771ab23",
        "current": "178246f5032b512717edf764edaa6d1ddec6f04241f329e6cb6acffe9112acb6",
        "why": (
            "2026-10-28 level-parallel integer zero classification: cells then "
            "levels preserve uint32 OR exactly; fast and strict device "
            "transport checks pass. Floating-point CUDA text is unchanged. "
            "2026-10-28 bounded batching and repeated-zero result reuse: "
            "CUDA arithmetic strings are unchanged; host launch word checks "
            "pass. Device checks and same-card speed remain release gates. "
            "This refresh describes current sources, not a new forecast mint. "
            "2026-10-19 LTS hook repair: unspecialized lookups retain their "
            "two-part key; the acoustic tap uses the launched shape identity. "
            "CUDA arithmetic and archived recorded hashes are unchanged. "
            "2026-10-19 current-source refresh after f65ab11c: immutable "
            "kernel dimensions are specialized and cache keys include shape. "
            "The archived recorded hash and receipt scope stay unchanged; "
            "this is not a new forecast mint. Earlier migrations follow. "
            "0.2.0 package rename, 2026-08-28: the token mpas_port became "
            "hexcore. This file is byte-identical to the copy these receipts "
            "ran once that one identifier is substituted -- measured file "
            "against file, not argued from a diff -- and the 1 line carrying "
            "it is 1 NVRTC module_key or module-name label. No line of CUDA "
            "source text differs, so every kernel these receipts launched "
            "compiles from the same bytes; a module_key names a cache slot "
            "and an observer label, never anything the arithmetic reads. "
            "Re-derived by tools/repin_source_tables.py."
            "  Moved again 2026-10-14 by the 0.4.0 release scrub: one comment "
            "line names the card host by its hardware; comment only, no launch, "
            "argument or arithmetic line moved"
        ),
    },
}


#: Directory names this repository has retired, newest spelling last.
#:
#: AN ARCHIVED RECEIPT IS NEVER REWRITTEN, so a receipt names its sources in
#: the spelling the tree used on the day it ran, and a rename strands every one
#: of those names.  THE BREAKAGE THIS MAP PREVENTS, and it is not theoretical:
#: the four L5b regional receipts each name fourteen sources under
#: ``src/mpas_port/``, the 0.2.0 package rename retired that directory, and the
#: gate below stopped being a gate -- it died inside ``read_bytes`` with a
#: FileNotFoundError before it could compare a single digest, which reads as an
#: infrastructure fault rather than as "this anchor is resting on bytes nobody
#: can account for".
#:
#: This is a TRANSLATION of a retired name, never a relaxation.  A translated
#: name still has to resolve to a real file, and still has to match the
#: recorded digest or carry a declared drift entry; a name that lands nowhere
#: fails BY NAME, saying which spellings it tried, instead of raising an OS
#: error from three frames down.
RETIRED_PATH_PREFIXES: tuple[tuple[str, str], ...] = (
    ("src/mpas_port/", "src/hexcore/"),
)


def resolve_recorded_source(name: str) -> tuple[str, Path]:
    """Today's spelling of a name an archived receipt recorded, and its file."""

    candidates = [name]
    for retired, current in RETIRED_PATH_PREFIXES:
        if name.startswith(retired):
            candidates.append(current + name[len(retired):])
    for candidate in candidates:
        path = ROOT / candidate
        if path.is_file():
            return candidate, path
    raise AssertionError(
        f"an archived receipt names {name!r} and no file of this tree answers "
        f"to it under any spelling this repository has used "
        f"({', '.join(candidates)}).  Either the source was deleted, or a "
        f"rename happened without a row being added to RETIRED_PATH_PREFIXES"
    )


def test_every_receipt_names_the_source_bytes_that_produced_it(receipts):
    """A receipt that cannot say which source produced it cannot be re-run.

    Each forecast receipt records the SHA-256 of the sources the run
    executed.  Those must be the bytes committed here, or an entry in
    :data:`SOURCE_DRIFT_SINCE_CAMPAIGN` saying what moved them and why the
    move cannot change what the receipt measured.  An unexplained drift
    fails: the anchor would be resting on a tree nobody can account for.

    A recorded name is read through :func:`resolve_recorded_source` first,
    because the spelling a receipt recorded is the spelling of the day it ran
    and this repository has renamed a package since.
    """

    import hashlib

    base = ROOT / "evidence/regional-forecast-l5b-20260826/forecast"
    receipts = sorted(base.glob("*/forecast.json"))
    assert len(receipts) == 4
    for receipt in receipts:
        recorded = json.loads(receipt.read_text())["source_sha256"]
        assert recorded, receipt
        for recorded_name, digest in recorded.items():
            name, path = resolve_recorded_source(recorded_name)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual == digest:
                continue
            drift = SOURCE_DRIFT_SINCE_CAMPAIGN.get(name)
            assert drift is not None, (
                f"{receipt.parent.name} names {name} at {digest}, the tree "
                f"carries {actual}, and nothing says what moved it"
            )
            assert drift["recorded"] == digest, (receipt.parent.name, name)
            assert drift["current"] == actual, (receipt.parent.name, name)
            assert drift["why"].strip(), (receipt.parent.name, name)


def test_the_declared_drift_leaves_the_recorded_configuration_untouched():
    """The drift entry's claim is checked, not taken.

    ``SOURCE_DRIFT_SINCE_CAMPAIGN`` says the physics cadence table is
    byte-identical for any configuration that selects a cumulus scheme.
    That is the whole of why the four archived receipts still stand, so it
    is measured here rather than asserted in a comment.
    """

    from woof.hex.cuda_driver import _v841_physics_cadences
    from woof.hex.config_v841 import V841MpasColumnPhysicsConfig

    # The archived receipts were produced at the native x4 reference's
    # schedule, which since 2026-10-10 is passed explicitly rather than
    # being the dataclass default (woof.hex.mpas_defaults).
    from woof.hex.mpas_defaults import X4_RECORDED_SCHEDULE

    selected = V841MpasColumnPhysicsConfig(**X4_RECORDED_SCHEDULE)
    assert selected.config_convection_scheme == "cu_grell_freitas"
    assert _v841_physics_cadences(selected) == {
        "radiation_lw": 600.0,
        "radiation_sw": 600.0,
        "surface_pbl": 120.0,
        "convection": 120.0,
        "microphysics": 120.0,
    }

    off = V841MpasColumnPhysicsConfig(
        config_convection_scheme="off", config_cudt_seconds=None
    )
    assert _v841_physics_cadences(off)["convection"] is None


def test_the_specified_zone_is_bitwise_against_the_cpu_authority(receipts):
    """Rings 6-7 are the regional stages' own output; they must be exact."""

    report = json.loads(
        (
            ROOT
            / "evidence/regional-forecast-l5b-20260826"
            / "probes/cpu-authority-step1.json"
        ).read_text()
    )
    fields = {row["field"]: row for row in report["cpu_authority"][0]["fields"]}
    for name in ("u", "w", "theta", "rho", "qv"):
        rings = fields[name]["by_ring"]
        for ring in ("6", "7"):
            assert rings[ring]["bitwise_equal"], (name, ring)
            assert rings[ring]["mismatches"] == 0


def test_the_attribution_control_has_a_shared_kernel_diverging_on_its_own(receipts):
    """The residue is claimed to be the shared dycore's; that is measured.

    ``solve_diagnostics`` runs no regional stage at all, so a divergence it
    shows is the shared CUDA lane against the numpy authority on this mesh.
    The control has teeth in both directions: some of its fields are bitwise,
    so a probe that simply reported "everything differs" would be caught.
    """

    probe = json.loads(
        (
            ROOT
            / "evidence/regional-forecast-l5b-20260826"
            / "probes/shared-dycore-attribution.json"
        ).read_text()
    )
    rows = {row["field"]: row for row in probe["fields"]}
    assert rows["divergence"]["bitwise_equal"] is True
    assert rows["tangential_velocity"]["bitwise_equal"] is True
    diverging = rows["kinetic_energy"]
    assert diverging["bitwise_equal"] is False
    assert diverging["mismatch_count"] > 0
    # Flat with ring: a boundary-shaped defect would concentrate outward.
    fractions = [
        block["mismatches"] / block["values"]
        for block in diverging["by_ring"].values()
    ]
    assert max(fractions) - min(fractions) < 0.05


# ---------------------------------------------------------------------------
# the NVRTC reciprocal-rewrite A/B, measured on a live sm_120 forecast
# ---------------------------------------------------------------------------


def test_the_nvrtc_fix_moved_the_interior_and_left_the_boundary_alone(receipts):
    """A flux-kernel change must not reach the driven rings.

    The specified zone is written from the lateral boundary every step, so a
    change to interior arithmetic has no route into it.  If rings 6-7 had
    moved, the change reached somewhere it has no business reaching, and the
    regional plumbing would be the suspect.
    """

    delta = json.loads(
        (
            ROOT
            / "evidence/regional-forecast-l5b-20260826"
            / "probes/nvrtc-reciprocal-forecast-delta.json"
        ).read_text()
    )
    frames = delta["frames"]
    assert len(frames) >= 2
    # Frame 0 is the initial state: nothing has been computed, so nothing can
    # have moved.  That is the control that says the comparison can see zero.
    assert all(row["bitwise_equal"] for row in frames[0]["fields"])
    moved_interior = 0
    for frame in frames[1:]:
        for row in frame["fields"]:
            for ring in ("6", "7"):
                block = row["by_ring"].get(ring)
                if block is None:
                    continue
                assert block["changed"] == 0, (frame["frame"], row["field"], ring)
            moved_interior += sum(
                row["by_ring"][str(ring)]["changed"]
                for ring in range(6)
                if str(ring) in row["by_ring"]
            )
    # And it did move the interior: an A/B where nothing moved would mean the
    # two arms were the same build.
    assert moved_interior > 0


def test_the_attribution_count_did_not_move_across_the_nvrtc_fix(receipts):
    """A count that did not move is a cause that is still open.

    The kinetic-energy divergence was a candidate for the NVRTC reciprocal
    defect.  It is not: the count is identical before and after, and the NVRTC reciprocal-rewrite defect's
    own census reports no rewritten site left anywhere in the port.  Pinning
    the number here stops a later lane from quietly adopting the fix as its
    explanation.
    """

    probe = json.loads(
        (
            ROOT
            / "evidence/regional-forecast-l5b-20260826"
            / "probes/shared-dycore-attribution.json"
        ).read_text()
    )
    rows = {row["field"]: row for row in probe["fields"]}
    assert rows["kinetic_energy"]["mismatch_count"] == 36_750
    assert rows["kinetic_energy"]["values"] == 163_405
    assert "explains NONE of it" in probe["finding"]

    census = json.loads(
        (
            ROOT
            / "evidence/nvrtc-reciprocal-20260826"
            / "census/census-after-the-fix.json"
        ).read_text()
    )
    assert census["total_rewritten_sites"] == 0
    assert census["cuda_bearing_modules_not_censused"] == []


# ---------------------------------------------------------------------------
# the gate still refuses everything it has no evidence for
# ---------------------------------------------------------------------------


def test_an_unregistered_digest_is_refused_by_name():
    """RE-POINTED 2026-08-27 by the anchor re-keying, and it says more now.

    This guard read ``no regional anchor is registered`` -- one sentence for
    two different missing things.  An anchor has two halves earned two
    different ways, so a refusal that does not say WHICH half is missing
    sends the reader to mint a forecast pair they may already have.  A cull
    with an unknown digest is missing its DECK.
    """

    with pytest.raises(RuntimeError) as error:
        regional_admission.require_regional_anchor(
            None, bdy_mask_sha256="00" * 32, n_cells=2_971
        )
    message = str(error.value)
    assert "no contract deck has been run" in message
    assert "tools/run_cuda_regional_contract.py" in message
    assert "indexed by ring" in message
    assert ANCHOR_ROW in message

    # And the other half, which is a different refusal naming a different
    # remedy: a configuration nobody has ever run twice and compared.
    with pytest.raises(RuntimeError) as error:
        regional_admission.require_regional_anchor(
            None, bdy_mask_sha256="00" * 32, n_cells=2_971,
            boundary_zone_width=7, n_vert_levels=55,
            finest_edge_m=1234.5, dt_seconds=7.0,
        )
    message = str(error.value)
    assert "holds no forecast mint" in message
    assert "tools/run_cuda_regional_forecast.py --mint-anchor" in message


def test_an_unregistered_row_name_is_refused_even_with_a_registered_digest():
    """Naming a row is a claim; the bytes are the proof, and both must hold."""

    anchor = regional_admission.admitted_region(ANCHOR_ROW)
    assert anchor is not None
    with pytest.raises(RuntimeError):
        regional_admission.require_regional_anchor(
            "a-region-nobody-anchored",
            bdy_mask_sha256=anchor.bdy_mask_sha256,
            n_cells=anchor.n_cells,
        )


def test_a_registered_row_whose_masks_moved_is_refused_by_name():
    anchor = regional_admission.admitted_region(ANCHOR_ROW)
    assert anchor is not None
    with pytest.raises(RuntimeError) as error:
        regional_admission.require_regional_anchor(
            ANCHOR_ROW, bdy_mask_sha256="11" * 32, n_cells=anchor.n_cells
        )
    assert "is not the zone geometry this run would execute" in str(error.value)


def test_a_different_cull_of_the_same_region_is_refused():
    anchor = regional_admission.admitted_region(ANCHOR_ROW)
    assert anchor is not None
    with pytest.raises(RuntimeError) as error:
        regional_admission.require_regional_anchor(
            ANCHOR_ROW,
            bdy_mask_sha256=anchor.bdy_mask_sha256,
            n_cells=anchor.n_cells + 1,
        )
    assert "a different cull is a different configuration" in str(error.value)


def test_digest_resolution_cannot_admit_something_unregistered():
    assert regional_admission.admitted_region_by_digest(None) is None
    assert regional_admission.admitted_region_by_digest("") is None
    assert regional_admission.admitted_region_by_digest("ab" * 32) is None


# ---------------------------------------------------------------------------
# the padded memory model
# ---------------------------------------------------------------------------


class _SyntheticCull:
    """A three-cell strip with one sentinel-bearing outer row.

    Small enough to assert on element by element, and shaped like the thing
    that matters: one edge with a single present cell, which is the row a
    regional cull's outermost ring carries.
    """

    def __init__(self) -> None:
        n_cells, n_edges, n_vertices, max_edges = 3, 4, 2, 3
        self.dimensions = {
            "nCells": n_cells,
            "nEdges": n_edges,
            "nVertices": n_vertices,
            "maxEdges": max_edges,
            "maxEdges2": 4,
            "vertexDegree": 3,
        }
        self.arrays = {
            "cellsOnEdge": np.array(
                [[0, 1], [1, 2], [2, -1], [0, 2]], dtype=np.int64
            ),
            "cellsOnCell": np.array(
                [[1, 2, -1], [0, 2, -1], [0, 1, -1]], dtype=np.int64
            ),
            "edgesOnCell": np.array(
                [[0, 3, 0], [0, 1, 0], [1, 3, 2]], dtype=np.int64
            ),
            "nEdgesOnCell": np.array([2, 2, 3], dtype=np.int64),
            "edgesOnEdge": np.array(
                [[1, 3, -1, -1], [0, 3, -1, -1], [-1, -1, -1, -1],
                 [0, 1, -1, -1]],
                dtype=np.int64,
            ),
            "nEdgesOnEdge": np.array([2, 2, 0, 2], dtype=np.int64),
            "verticesOnEdge": np.array(
                [[0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.int64
            ),
            "verticesOnCell": np.array(
                [[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.int64
            ),
            "edgesOnVertex": np.array([[0, 1, 3], [0, 1, -1]], dtype=np.int64),
            "cellsOnVertex": np.array([[0, 1, 2], [0, 1, -1]], dtype=np.int64),
            "dcEdge": np.full(n_edges, 2.0, dtype=np.float64),
            "dvEdge": np.full(n_edges, 3.0, dtype=np.float64),
            "areaCell": np.full(n_cells, 5.0, dtype=np.float64),
            "areaTriangle": np.full(n_vertices, 7.0, dtype=np.float64),
            "weightsOnEdge": np.zeros((n_edges, 4), dtype=np.float64),
            "kiteAreasOnVertex": np.ones((n_vertices, 3), dtype=np.float64),
            "latCell": np.zeros(n_cells, dtype=np.float64),
            "lonCell": np.zeros(n_cells, dtype=np.float64),
            "latEdge": np.zeros(n_edges, dtype=np.float64),
            "lonEdge": np.zeros(n_edges, dtype=np.float64),
            "angleEdge": np.zeros(n_edges, dtype=np.float64),
            "meshDensity": np.ones(n_cells, dtype=np.float64),
            "fVertex": np.zeros(n_vertices, dtype=np.float64),
            "fEdge": np.zeros(n_edges, dtype=np.float64),
            "nominalMinDc": np.float64(2.0),
        }
        self.attrs: dict[str, object] = {}

    def __getattr__(self, name: str):
        arrays = self.__dict__.get("arrays")
        if arrays is not None and name in arrays:
            return arrays[name]
        raise AttributeError(name)


def _masks(mesh: _SyntheticCull) -> regional_v841.RegionalMasks:
    n_cells = 3
    n_edges = 4
    n_vertices = 2
    dtype = np.dtype(np.float32)
    return regional_v841.RegionalMasks(
        bdy_mask_cell=np.zeros(n_cells, dtype=np.int64),
        bdy_mask_edge=np.zeros(n_edges, dtype=np.int64),
        bdy_mask_vertex=np.zeros(n_vertices, dtype=np.int64),
        spec_zone_mask_cell=np.zeros(n_cells, dtype=dtype),
        spec_zone_mask_edge=np.zeros(n_edges, dtype=dtype),
        spec_zone_mask_vertex=np.zeros(n_vertices, dtype=dtype),
        nearest_relaxation_cell=np.full(n_cells, n_cells, dtype=np.int64),
        spec_cells=np.zeros(0, dtype=np.int64),
        spec_edges=np.zeros(0, dtype=np.int64),
        relax_cells=np.zeros(0, dtype=np.int64),
        relax_edges=np.zeros(0, dtype=np.int64),
        nudged_cells=np.zeros(0, dtype=np.int64),
    )


def test_the_padded_view_declares_one_extra_element_per_dimension():
    mesh = _SyntheticCull()
    padded = PaddedRegionalHostMesh(mesh, _masks(mesh))
    assert padded.dimensions["nCells"] == 4
    assert padded.dimensions["nEdges"] == 5
    assert padded.dimensions["nVertices"] == 3
    assert (padded.garbage_cell, padded.garbage_edge, padded.garbage_vertex) == (
        3, 4, 2
    )


def test_every_sentinel_is_remapped_and_the_garbage_row_points_at_itself():
    mesh = _SyntheticCull()
    padded = PaddedRegionalHostMesh(mesh, _masks(mesh))
    for name, garbage in (
        ("cellsOnEdge", padded.garbage_cell),
        ("cellsOnCell", padded.garbage_cell),
        ("cellsOnVertex", padded.garbage_cell),
        ("edgesOnEdge", padded.garbage_edge),
        ("edgesOnVertex", padded.garbage_edge),
    ):
        rows = padded.arrays[name]
        assert not np.any(rows < 0), name
        assert np.all(rows[-1] == garbage), name
    assert int(padded.arrays["nEdgesOnCell"][-1]) == 0
    assert int(padded.arrays["nEdgesOnEdge"][-1]) == 0


def test_the_acoustic_lane_keeps_its_sentinel_connectivity():
    """``acoustic_ru_regional_v841`` finds a one-cell edge by a NEGATIVE entry.

    It mirrors the CPU authority's ``acoustic_v841.py:385-405``, which tests
    ``cellsOnEdge < 0`` because native multiplies that edge's
    garbage-gathered pressure gradient by exactly zero (F:3909).  Handing it
    the remapped array would silently retire that branch, so the padded view
    carries a second, sentinel-preserving copy -- whose GARBAGE row is
    remapped, so the garbage edge does not trip the kernel's refusal flag.
    """

    mesh = _SyntheticCull()
    padded = PaddedRegionalHostMesh(mesh, _masks(mesh))
    sentinel = padded.arrays["cellsOnEdgeSentinel"]
    assert sentinel.shape == (5, 2)
    assert int(sentinel[2, 1]) == -1
    assert np.all(sentinel[-1] == padded.garbage_cell)


def test_the_geometry_pads_are_the_cpu_authority_s_own():
    mesh = _SyntheticCull()
    padded = PaddedRegionalHostMesh(mesh, _masks(mesh))
    assert float(padded.arrays["dcEdge"][-1]) == 0.0
    assert float(padded.arrays["dvEdge"][-1]) == 0.0
    assert float(padded.arrays["weightsOnEdge"][-1][0]) == 0.0
    assert float(padded.arrays["kiteAreasOnVertex"][-1][0]) == 0.0
    # areaCell/areaTriangle pad 1.0 so a non-inverse caller cannot trap on a
    # dead division (regional_v841.py:1100-1104); the v8.4.1 lane consumes
    # inverses, which pad with the native allocation 0.0.
    assert float(padded.arrays["areaCell"][-1]) == 1.0
    assert float(padded.arrays["areaTriangle"][-1]) == 1.0


def test_the_zone_masks_make_the_garbage_element_a_specified_element():
    """That is what makes its computed values the native pool zeros."""

    mesh = _SyntheticCull()
    padded = PaddedRegionalHostMesh(mesh, _masks(mesh))
    assert (
        float(padded.arrays["spec_zone_mask_edge"][-1])
        == REGIONAL_GARBAGE_POOL["spec_zone_mask"]
    )
    assert (
        float(padded.arrays["spec_zone_mask_cell"][-1])
        == REGIONAL_GARBAGE_POOL["spec_zone_mask"]
    )


def test_inactive_deformation_placeholders_are_named_not_silent():
    """A cull carries no defc_a/defc_b; the v8.4.1 lane never reads them."""

    mesh = _SyntheticCull()
    padded = PaddedRegionalHostMesh(mesh, _masks(mesh))
    assert padded.inactive_deformation_placeholders == ["defc_a", "defc_b"]
    assert padded.arrays["defc_a"].shape == (4, 3)
    assert not np.any(padded.arrays["defc_a"])


# ---------------------------------------------------------------------------
# the global lane is untouched
# ---------------------------------------------------------------------------


def test_the_driver_hook_is_absent_unless_a_regional_runtime_sets_it():
    """Every regional site in cuda_driver is guarded by this attribute."""

    source = package_path("src/hexcore/cuda_driver.py").read_text()
    lines = source.splitlines()
    assert "        self.regional_v841: Any | None = None" in lines
    guards = [
        index
        for index, line in enumerate(lines)
        if "if self.regional_v841 is None" in line
        or "if self.regional_v841 is not None" in line
    ]
    assert len(guards) >= 8
    # Every call THROUGH the runtime must sit inside a guard: walk back up
    # from each call site and require a guard at a strictly smaller indent
    # before the enclosing def.  A regional call reachable on the global lane
    # would dereference None on a whole-mesh run.
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("self.regional_v841."):
            continue
        indent = len(line) - len(line.lstrip())
        guarded = False
        for back in range(index - 1, -1, -1):
            candidate = lines[back]
            if not candidate.strip():
                continue
            back_indent = len(candidate) - len(candidate.lstrip())
            if back_indent >= indent:
                continue
            stripped_back = candidate.strip()
            if "self.regional_v841 is not None" in stripped_back:
                guarded = True
                break
            if stripped_back == "else:":
                partner = next(
                    (
                        lines[up]
                        for up in range(back - 1, -1, -1)
                        if lines[up].strip()
                        and len(lines[up]) - len(lines[up].lstrip())
                        == back_indent
                    ),
                    "",
                )
                if "self.regional_v841 is None" in partner:
                    guarded = True
                    break
            assert not stripped_back.startswith("def "), (
                f"line {index + 1} calls the regional runtime unguarded"
            )
            indent = back_indent
        assert guarded, (
            f"line {index + 1} calls the regional runtime unguarded"
        )


def test_the_kernel_cache_observer_is_inert_by_default():
    from woof.hex.cuda_backend import runtime

    source = package_path("src/hexcore/cuda_backend/runtime.py").read_text()
    assert "self.post_launch: Any | None = None" in source
    assert "if self.post_launch is None:\n            return kernel" in source
    assert hasattr(runtime, "_PostLaunchKernel")


def test_the_default_namelist_is_still_the_campaign_namelist():
    """The drift entry's claim about the instrument is checked, not taken.

    ``SOURCE_DRIFT_SINCE_CAMPAIGN`` says the swath-as-lam lane's new
    arguments cannot change what the four archived receipts measured,
    because ``dry_reference_config()`` called with no arguments still
    returns the CANDIDATE-REGIONAL-DRY namelist.  A default that drifted
    would silently re-label every one of those receipts, so it is measured
    here.
    """

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_swath_lam_instrument", ROOT / "tools" / "run_cuda_regional_forecast.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.dry_reference_config()
    assert config.config_dt == 120.0
    assert config.config_len_disp == 25000.0
    assert config.config_apply_lbcs is True
    assert config.config_moist_physics is True
    assert config.config_dynamics_split_steps == 3
    assert config.config_number_of_sub_steps == 6
    assert config.config_scalar_adv_order == 3
    assert config.config_horiz_mixing == "2d_smagorinsky"
    assert config.config_monotonic is False
    assert config.config_divergence_damping is True
    assert config.config_zd == 22_000.0


def test_the_swath_cull_anchor_names_evidence_that_exists(receipts):
    """The second regional row, and the same standard as the first.

    A row that names a contract receipt or a forecast anchor which is not
    in this repository is a row nobody can check -- which is the exact
    breakage the anchor gate exists to prevent, re-introduced one level up.
    """

    from woof.hex.cuda_backend.regional_admission import ADMITTED_REGIONS

    anchor = ADMITTED_REGIONS["r4.75.11020"]
    assert anchor.n_cells == 11_020
    assert anchor.boundary_zone_width == 7
    for named in (anchor.contract_receipt, anchor.forecast_anchor):
        assert (ROOT / named).is_file(), named
    contract = json.loads((ROOT / anchor.contract_receipt).read_text())
    assert contract["summary"]["decks_passed"] == contract["summary"]["decks"]
    assert contract["summary"]["all_dual_run_identical"] is True
    assert contract["summary"]["all_controls_have_teeth"] is True
    # THE GEOMETRY HALF IS THIS CULL'S OWN, and that did not move.
    assert contract["mesh"]["n_cells"] == anchor.n_cells
    forecast = json.loads((ROOT / anchor.forecast_anchor).read_text())
    # THE FORECAST HALF IS THE CLASS'S, and since 2026-08-27 it is DELIBERATELY
    # not this cull's.  A mint measures that a kernel set reproduces itself at
    # a timestep on a domain of a shape; five concentric culls of this parent
    # were minted independently and no input the instrument reads distinguished
    # them, so charging every re-placed cull for a second pair was 5.5 to 8.7
    # minutes of card naming no breakage.  Asserting the mint ran on a
    # DIFFERENT cull of the same class is the stronger check: it is the
    # re-keying, exercised.
    assert forecast["n_cells_solve"] != anchor.n_cells
    klass = regional_admission.ADMITTED_CLASSES[anchor.class_id]
    assert anchor.forecast_anchor in klass.mint_receipts
    assert forecast["config_dt"] == klass.key.dt_ms / 1000.0
    assert forecast["config_apply_lbcs"] is True
    assert forecast["minted_without_anchor_gate"] is True
    # The forecast half is a PAIR; one run agreeing with itself is nothing.
    sibling = (ROOT / anchor.forecast_anchor).parent.parent / "mint-run2" / "forecast.json"
    assert sibling.is_file(), sibling
    other = json.loads(sibling.read_text())
    assert [f["masked_sha256"] for f in forecast["frames"]] == [
        f["masked_sha256"] for f in other["frames"]
    ]
    assert [f["sha256"] for f in forecast["frames"]] != [
        f["sha256"] for f in other["frames"]
    ]
