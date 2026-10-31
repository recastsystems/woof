"""``woof mesh``: make an MPAS mesh at the resolution you ask for.

The generator itself is ``rw_mpas_mesh``, a binary in
``tools/rustwx/crates/rw-mpas`` that reproduces the published NCAR
meshes to 1e-11 or better.  Until this module existed it was
engine-proven and unreachable: no row in
:data:`woof.bridge_assets.BUNDLED_ARTIFACTS`, no contract marker, no
environment variable, no resolver, and no Python that called it.  A
capability with no front door is not a feature, so this file is the
door and the three legs behind it -- resolution, sizing, and the
quality gate.

What the door is for
--------------------
A user says four things: WHERE to refine, HOW FINE there, HOW COARSE
everywhere else, and WHICH CARD it has to run on.  ``woof mesh``
turns that into a resolution spec, sizes it against the measured
footprint of the named card, checks the result is not rougher than
anything that has been integrated, and drives the binary::

    woof mesh --out kansas.grid.nc \\
        --refine 39.0,-98.0,1200,15 --background-km 120 --card rtx-5090

Regions are DATA.  A second place to refine is a second ``--refine``
row and a second element of ``regions`` in the spec the binary reads;
adding one touches no code here and no code in the crate.  That is the
arbitrary acceptance test at this door.

Sizing is CARD-SPECIFIC, and that is not a detail
-------------------------------------------------
The device footprint model is ``fixed_mib + cells * bytes_per_cell``,
and the fixed term is DERIVED per card rather than stored.  Most of it
is one allocation: the CUDA per-context local-memory backing store,
which the driver sizes for the widest launched kernel frame at full
residency -- ``(frame_bytes - 1024) * SMs * maxThreadsPerSM`` -- and
never gives back.  Two of those three factors are properties of the
PART, so identical code pays 7,032.4 MiB on a 170 SM card and
2,895.7 MiB on a 70 SM one, and the measured fixed terms come out
9,798 MiB and 5,384 MiB.  A fixed term quoted without naming the card
is meaningless, and using one card's number for another mis-sizes the
request by thousands of MiB.

What is NOT derived is the remainder -- context, module images, driver
scratch.  That is measured per card as ``local_store_residue_mib``, it
is not the same on the two parts measured (2,765.6 against 2,488.3 MiB)
and it does not scale with the SM count, so it cannot be predicted for
a part nobody has run.

So the model is a TABLE (:data:`SIZING_DATA_PATH`) with one row per
card, each row carrying the card's SM count, its resident-thread
capacity, its measured residue and its anchors, and a card with no
measured row is REFUSED by name rather than answered with a
neighbour's number.  Measuring a new card is a row.

The binary carries the same model, per card, behind ``--card``, and
refuses ``--vram-gib`` without one.  This door still computes the cell
count itself and passes ``--cells``, so the two never disagree
silently; :func:`woof.mpas_mesh.SIZING_DATA_PATH`'s numbers and the
crate's ``mesh::footprint::CARDS`` are the same measurements.

The quality gate
----------------
A mesh can be generated far rougher than anything NCAR publishes: a
15 km regional patch inside a 10 GiB budget needs a spacing ratio of 12
and reaches 4.88 %/cell at its steepest, against 1.53 %/cell in the
production ``x4.163842``.  Whether the dycore is stable there is NOT
MEASURED.  The bound is therefore a PROVISIONAL row in the same table,
with the evidence field saying so in those words, and it moves when a
dycore run sets it -- by editing the table, not the code.

The deliverable boundary
------------------------
This produces a PAIR: the grid file, and beside it the matching STATIC
(terrain, land use, soil, ``nVertLevels=55``, ``nSoilLevels=4``, an
FP32-bit-exact ``nominalMinDc``, and ``deriv_two``).  Both, because the
mesh registry pins the two together by byte count and SHA-256 and a
grid alone reaches no dycore.

This paragraph used to say the door produced a grid and left the static
to somebody else, which was true when it was written and stopped being
true twice: once when ``rw_mpas_static`` was built, and again when that
engine started writing ``deriv_two``.  The verdict a run gets is
therefore COMPUTED from the engine's own absent-field list
(:func:`runnability_verdict`) rather than asserted in prose, so a
sentence like this one cannot outlive what it describes a second time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from woof.bridges import (RUSTWX_CRATE_RELATIVE, artifact_remedy,
                           default_bridge_dir,
                           legacy_bridge_candidates, lazy_build_hints,
                           rustwx_build_hint,
                           accept_resolved, executable_name,
                           packaged_bridge_dir)
from woof.cli_numbers import positive_float, positive_int

#: Schema of the sizing table this module reads.
SIZING_SCHEMA = "gpuwm-mpas-mesh-sizing-v1"

#: The sizing table, relative to the ``woof`` package.
SIZING_DATA_PATH = Path(__file__).resolve().parent / "data" / "mpas" / \
    "mesh-sizing.json"

#: The exact ``--abi`` line ``rw_mpas_mesh`` prints, which is also its
#: contract marker in :data:`woof.bridges.BRIDGE_ABI_MARKERS`.  It
#: spells the argument vector out rather than carrying a version, so a
#: rebuilt-but-unchanged binary still matches and a binary whose flags
#: moved does not.  A stale build would accept ``--cells`` and mean
#: something else by it, which is the 1.1.0 GFS series-file failure with
#: a mesh on the other end.
MESH_ABI_MARKER = (
    "rw_mpas_mesh --out GRID.nc [--spec SPEC.json | --background-km KM | "
    "--from-centres GRID.nc] [--cells N | --card KEY [--vram-gib X]] "
    "[--fit-spacing yes|no] [--sweeps N] [--tolerance X] [--omega X] "
    "[--receipt JSON] [--triangulation rebuild|incremental] [--clobber] "
    "[--dry-run] [--list-cards]")

#: ``rw_mpas_static``'s contract, at the head of its argument vector.
#: The crate's own literal continues past this into the optional flags
#: and the progress grammar (:data:`woof.rustwx_static.STATIC_ABI_MARKER`
#: carries the whole thing); this prefix is what
#: :data:`woof.bridges.BRIDGE_ABI_MARKERS` pins, and a test binds the
#: two so they cannot drift apart.
STATIC_ABI_MARKER = "rw_mpas_static --grid GRID.nc --out STATIC.nc"

#: ``rw_mpas_init``'s contract, at the head of its argument vector.  The
#: crate's own literal embeds a newline inside the vector, so this is a
#: prefix rather than the whole line -- stated, because a marker that
#: silently checks less than it looks like it checks is worse than a
#: shorter one that says so.  Spelled to match
#: :data:`woof.bridges.BRIDGE_ABI_MARKERS`; a test binds the two.
INIT_ABI_MARKER = (
    "rw_mpas_init --met MET --static STATIC.nc --capsule CAPSULE.nc")

#: ``rw_mpas_convert``'s contract: the output schema and the progress
#: tokens, which is the literal that changes when the tape contract does,
#: plus the flags callers pass that an older build refuses (``--remap`` and
#: its REMAP/BUDGET lines, ``--jobs``).  Without them a pre-2.9.0 converter
#: matched, passed ``woof doctor``, and the woof hex product stage died
#: on ``unknown flag --remap``.
CONVERT_ABI_MARKER = (
    "gpuwm-rw-mpas-convert-v1\tCONVERTED\tWINDOW\tWEIGHTS\tFINISHED"
    "\tREMAP\tBUDGET\t--remap nearest|product\t--jobs N")

#: ``rw_mpas_lbc``'s contract, its first usage line through the
#: physics-mode switch.  The crate's literal is four usage lines with
#: newlines between them, so this is a prefix rather than the whole thing,
#: for the reason :data:`INIT_ABI_MARKER` states: a marker that has to
#: reproduce an embedded newline exactly breaks on a reflow that changed
#: nothing.  It reaches ``[--physics-mode correct|exact]`` (2.9.0: correct
#: by default, spelled default first) because the producer reads unknown
#: flags without complaint: a build from before the switch would take
#: ``--physics-mode correct`` and write native case-9 boundary winds.
#: Spelled to match :data:`woof.bridges.BRIDGE_ABI_MARKERS`; a test
#: binds the two, and both to the crate's literal.
LBC_ABI_MARKER = (
    "rw_mpas_lbc --grid INIT.nc --out-dir DIR "
    "--start-time YYYY-MM-DD_HH:MM:SS --stop-time YYYY-MM-DD_HH:MM:SS "
    "--interval YYYY-MM-DD_HH:MM:SS=MET_PATH [--interval ...] "
    "--nfglevels N --fg-interval-seconds N --extrap-airtemp MODE "
    "--use-spechumd yes|no --theta-adv-order N --coef-3rd-order X "
    "--oned-underflow preserve|reproduce-ifx-ftz "
    "[--physics-mode correct|exact]")

#: ``rw_hex_points``'s contract: its ``--abi`` line (schema name and the
#: stdout progress tokens).  Spelled to match
#: :data:`woof.bridges.BRIDGE_ABI_MARKERS`.
POINTS_ABI_MARKER = "gpuwm-rw-hex-points-v1\tPOINT\tSOUNDING\tMETEOGRAM\tFINISHED"

#: The RRFS-MPAS initializer's --abi line (rw-mpas rrfs_init::job::ABI_MARKER).
INIT_RRFS_ABI_MARKER = (
    "rw_mpas_init_v831 stdin: hex-init-job/v1 JSON; stage=case7|case9|"
    "static|terrain|vertical|artifact|prepare|gwd|surface")

#: The GSL orographic statistics generator's usage head.
GSL_ORO_ABI_MARKER = "rw_mpas_gsl_oro --grid MESH.nc --geog WPS_GEOG --out ORO.nc"

#: ``CARGO_BUILD_HINT``: the one-liner that builds the mesh tools, from a
#: checkout root, spelled for the shell rule when it is read.
__getattr__ = lazy_build_hints(
    __name__, CARGO_BUILD_HINT=RUSTWX_CRATE_RELATIVE)

_PROBE_TIMEOUT_S = 60


class MeshRequestError(RuntimeError):
    """A refusal about the REQUEST: the card, the size, the grammar."""


class MeshRoughnessError(RuntimeError):
    """A refusal about the mesh QUALITY: rougher than the stated bound."""


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """The directory containing the ``woof`` package."""

    return Path(__file__).resolve().parent.parent


def crate_dir() -> Path:
    """The vendored renderer workspace of a checkout (may not exist)."""

    return _repo_root() / "tools" / "rustwx"


@dataclass(frozen=True)
class MpasBridge:
    """One MPAS binary: its names, its contract, its remedy.

    The ladder is the one every other ``tools/rustwx`` artifact uses --
    environment override, checkout release, checkout debug, ``libexec``
    beside the package, the wheel-bundled directory, ``~/.woof/bridges``
    -- and ``tests/test_mpas_mesh_door.py`` binds it to
    :class:`woof.obs.frontdoor.FrontDoor`'s so the two cannot drift.
    It is spelled here rather than imported from there because reaching
    that module executes ``woof/obs/__init__.py``, which imports the
    gridding stack and therefore numpy; this module is reached at
    parser-build time by every ``woof`` invocation.
    """

    name: str
    env_var: str
    subject: str
    abi_marker: str

    def candidates(self) -> tuple[Path, ...]:
        """Deterministic candidate paths, best first."""

        filename = executable_name(self.name)
        candidates: list[Path] = []
        override = os.environ.get(self.env_var)
        if override:
            candidates.append(Path(override))
        candidates.extend((
            crate_dir() / "target" / "release" / filename,
            crate_dir() / "target" / "debug" / filename,
            _repo_root() / "libexec" / "bridges" / filename,
            packaged_bridge_dir() / filename,
            default_bridge_dir() / filename,
            *legacy_bridge_candidates(filename),
        ))
        return tuple(candidates)

    def find(self) -> Path | None:
        """First existing candidate, or None.

        An environment override naming a missing file is a hard error:
        explicit configuration must fail loudly rather than fall through
        to a different executable.
        """

        override = os.environ.get(self.env_var)
        for candidate in self.candidates():
            if candidate.is_file():
                return accept_resolved(candidate.resolve())
            if override and candidate == Path(override):
                raise FileNotFoundError(
                    f"{self.env_var} names a missing file: {candidate}")
        return None

    def require(self) -> Path:
        """The binary, or a refusal carrying this install's remedy."""

        found = self.find()
        if found is None and not os.environ.get(self.env_var):
            # An artifact the platform wheel leaves to the bridge bundle
            # (woof.bridge_assets.on_demand_artifacts) is staged on first use.
            from woof.bridges import fetch_on_demand
            if fetch_on_demand(self.name):
                found = self.find()
        if found is None:
            raise MeshRequestError(
                f"{self.subject} ({self.name}) is not built or not found.\n"
                f"{self.remedy()}")
        return found

    def remedy(self) -> str:
        """The remedy for a missing binary, true for THIS install."""

        return artifact_remedy(
            env_var=self.env_var, filename=executable_name(self.name),
            subject=self.subject, crate_relative=RUSTWX_CRATE_RELATIVE,
            one_liner=rustwx_build_hint(), artifact=self.name)

    def probe(self, path: Path) -> tuple[bool, str]:
        """``--abi``: is this the binary this wrapper was written against?

        Read by EXECUTING the binary, unlike the static marker check
        :func:`woof.bridges.bridge_abi_matches` applies to the same
        string.  Both exist for the same reason and answer at different
        moments: the static one runs inside ``woof fetch-bridges``
        before a byte is installed and inside the release cut, which
        inspects the other platform's bundle and cannot run it; this one
        runs where a door is about to be opened.
        """

        try:
            result = subprocess.run(
                [str(path), "--abi"], capture_output=True, text=True,
                errors="replace", timeout=_PROBE_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as error:
            return False, f"{path} did not run: {error}"
        if result.returncode != 0:
            return False, f"{path} --abi exited {result.returncode}"
        if self.abi_marker not in (result.stdout or ""):
            return False, (
                "--abi does not carry the argument vector this woof "
                f"drives; rebuild it: {rustwx_build_hint()}")
        return True, "--abi matches this release's argument vector"


#: The mesh generator.
MESH = MpasBridge(
    name="rw_mpas_mesh",
    env_var="WOOF_RW_MPAS_MESH",
    subject="the MPAS mesh generator",
    abi_marker=MESH_ABI_MARKER,
)

#: The static builder, the other half of what ``woof mesh`` delivers.
#: Not an optional extra: the mesh registry pins the GRID and the STATIC
#: together, by byte count and sha256, so a grid file on its own reaches
#: no dycore.
STATIC = MpasBridge(
    name="rw_mpas_static",
    env_var="WOOF_RW_MPAS_STATIC",
    subject="the MPAS static builder",
    abi_marker=STATIC_ABI_MARKER,
)

#: The initial-condition builder that takes a grid and a static file.
INIT = MpasBridge(
    name="rw_mpas_init",
    env_var="WOOF_RW_MPAS_INIT",
    subject="the MPAS initial-condition builder",
    abi_marker=INIT_ABI_MARKER,
)

#: Reconstruction vectors and coefficients used by the hex init door.
GEOMETRY = MpasBridge(
    name="rw_mpas_geometry",
    env_var="WOOF_RW_MPAS_GEOMETRY",
    subject="the MPAS reconstruction geometry builder",
    abi_marker="rw_mpas_geometry --protocol hex-geometry-v1",
)


#: Exact connectivity, terrain and initialized-momentum host preparation.
HOSTPREP = MpasBridge(
    name="rw_mpas_hostprep",
    env_var="WOOF_RW_MPAS_HOSTPREP",
    subject="the MPAS forecast host preparation builder",
    abi_marker="rw_mpas_hostprep --protocol hex-hostprep-v1",
)


#: The history converter onto the renderer's tape.
CONVERT = MpasBridge(
    name="rw_mpas_convert",
    env_var="WOOF_RW_MPAS_CONVERT",
    subject="the MPAS history converter",
    abi_marker=CONVERT_ABI_MARKER,
)

#: The lateral-boundary producer.  It is the fifth binary this comment
#: anticipated, and it arrived in the state ``rw_mpas_convert`` and
#: ``rw_mpas_mesh`` each arrived in: written, committed, built by every
#: release cut, and carried by no bundle -- so the only way to obtain it
#: was a Rust toolchain and a source checkout, which by this project's
#: rule means it was not shipped.  A limited-area mesh reaches no
#: forecast without it: the cull produces the mesh, and this produces
#: the boundary series that mesh is driven by.
LBC = MpasBridge(
    name="rw_mpas_lbc",
    env_var="WOOF_RW_MPAS_LBC",
    subject="the MPAS lateral-boundary producer",
    abi_marker=LBC_ABI_MARKER,
)

#: Soundings and meteograms at named points straight off MPAS native
#: history: the woof hex product door's third binary (2.9.0).  It is a bin
#: of rw-wrfbatch, not rw-mpas, but it sits on this ladder because this is
#: the table woof hex resolves its engines through and ``woof doctor``
#: reports from; left off it, a pip install of both packages had no point
#: pictures and ``woof hex doctor`` exited 1 naming it.
POINTS = MpasBridge(
    name="rw_hex_points",
    env_var="WOOF_RW_HEX_POINTS",
    subject="the MPAS sounding and meteogram renderer",
    abi_marker=POINTS_ABI_MARKER,
)

#: Every MPAS binary the bundle carries, by artifact name.  Read by
#: ``woof doctor`` and by the bundle-coverage test, so a sixth binary
#: is a row here and is reported without a second edit.
#: The RRFS-MPAS init door's two binaries (woof hex init --physics-suite
#: hrrrv5): the fork's init_atmosphere in Rust, and the GSL drag statistics.
INIT_RRFS = MpasBridge(
    name="rw_mpas_init_v831",
    env_var="WOOF_RW_MPAS_INIT_V831",
    subject="the RRFS-MPAS initializer",
    abi_marker=INIT_RRFS_ABI_MARKER,
)
GSL_ORO = MpasBridge(
    name="rw_mpas_gsl_oro",
    env_var="WOOF_RW_MPAS_GSL_ORO",
    subject="the GSL orographic statistics generator",
    abi_marker=GSL_ORO_ABI_MARKER,
)

BRIDGES: dict[str, MpasBridge] = {
    bridge.name: bridge for bridge in (MESH, STATIC, INIT, GEOMETRY, HOSTPREP, CONVERT, LBC, POINTS,
                                       INIT_RRFS, GSL_ORO)}


# ---------------------------------------------------------------------------
# The sizing table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LocalStore:
    """How CUDA sizes the per-context local-memory backing store.

    The DERIVED half of the fixed term, and the reason the fixed term is
    per card: two of the three factors are properties of the part.
    """

    formula: str
    widest_kernel: str
    widest_kernel_frame_bytes: float
    frame_credit_bytes: float
    #: The ArWen checkout the frame -- and so every residue subtracted
    #: against it -- was measured on.  The MPAS port runs ArWen's physics
    #: from a pinned checkout, and this is NOT the tree this table ships
    #: in: that tree has already cut the frame.  A bare frame that
    #: silently means some other checkout's build is the defect.
    measured_against_arwen_commit: str
    #: The commit that ENDS the validity of the frame above, if the
    #: port's pin ever reaches it.
    frame_cut_commit: str
    validation: str
    what_is_not_derived: str
    why_it_cannot_move_alone: str

    def mib(self, streaming_multiprocessors: int,
            max_threads_per_sm: int) -> float:
        """The store a part is forced to reserve, in MiB."""

        return ((self.widest_kernel_frame_bytes - self.frame_credit_bytes)
                * streaming_multiprocessors * max_threads_per_sm
                / 1_048_576.0)


@dataclass(frozen=True)
class CardCapacity:
    """One card's measured device footprint model, or its absence."""

    key: str
    display_name: str
    device_total_mib: float
    device_addressable_mib: float | None
    streaming_multiprocessors: int | None
    max_threads_per_sm: int | None
    compute_capability: str | None
    measured: bool
    local_store: LocalStore
    local_store_residue_mib: float | None
    #: The ArWen checkout this residue was measured against.  Carried per
    #: row, not assumed, so a card re-measured on a newer build cannot sit
    #: silently beside a frame from an older one.
    residue_measured_against_arwen_commit: str | None
    bytes_per_cell: float | None
    bytes_per_cell_provenance: str
    measured_on: str | None
    anchors: tuple[dict, ...]
    provenance: str
    notes: str

    @property
    def local_store_mib(self) -> float:
        """This part's CUDA local-memory backing store, DERIVED."""

        return self.local_store.mib(self.streaming_multiprocessors or 0,
                                    self.max_threads_per_sm or 0)

    @property
    def fixed_mib(self) -> float | None:
        """The fixed term, DERIVED from the part plus its measured residue.

        ``None`` when nobody has run this part.  It is deliberately not a
        stored number: the store half comes out of the card's own SM count
        and resident-thread capacity, and the residue is the only part a
        run has to tell us.
        """

        if self.local_store_residue_mib is None:
            return None
        return self.local_store_mib + self.local_store_residue_mib

    @property
    def sizing_budget_mib(self) -> float:
        """What to size against with no explicit budget.

        What CUDA can address, never what ``nvidia-smi`` prints.  On the
        70 SM part those differ by 422 MiB and the larger one is not
        addressable at all.
        """

        if self.device_addressable_mib is not None:
            return self.device_addressable_mib
        return self.device_total_mib

    def footprint_mib(self, cells: int) -> float:
        """Device footprint of ``cells`` cells on this card, in MiB."""

        if not self.measured:
            raise MeshRequestError(self.unmeasured_refusal())
        fixed = self.fixed_mib
        assert fixed is not None and self.bytes_per_cell is not None
        return fixed + cells * self.bytes_per_cell / 1_048_576.0

    def cells_that_fit(self, budget_mib: float) -> int:
        """Largest whole cell count inside ``budget_mib`` on this card."""

        if not self.measured:
            raise MeshRequestError(self.unmeasured_refusal())
        fixed = self.fixed_mib
        assert fixed is not None and self.bytes_per_cell is not None
        spare = budget_mib - fixed
        if spare <= 0:
            raise MeshRequestError(
                f"{self.key} ({self.display_name}) pays a {fixed:,.0f} MiB "
                "fixed footprint before the first cell exists, and the "
                f"budget given is {budget_mib:,.0f} MiB -- no mesh of ANY "
                "size fits.\n"
                f"  where it goes: {self.local_store_mib:,.0f} MiB is the "
                "CUDA per-context local-memory backing store this part's "
                f"{self.streaming_multiprocessors} SMs x "
                f"{self.max_threads_per_sm} resident threads force for the "
                f"{self.local_store.widest_kernel_frame_bytes:,.0f} B frame "
                f"of {self.local_store.widest_kernel}, and "
                f"{self.local_store_residue_mib:,.0f} MiB is the measured "
                "residue.  Both are paid whether the mesh is one cell or a "
                "million.\n"
                "  what changes it: a card with more memory, a budget that "
                "is not already spent on something else, or a build that "
                "cuts that frame.")
        return int(spare * 1_048_576.0 / self.bytes_per_cell)

    def measured_on_text(self) -> str:
        """The provenance sentence a refusal quotes."""

        return self.provenance

    def unmeasured_refusal(self) -> str:
        """Why this part cannot be sized, what IS known, and what works.

        The breakage is named in full because a refusal that does not
        name it is not a refusal: a mesh sized against a card whose
        footprint nobody measured comes out at a cell count that is
        confidently wrong in a direction nobody can name.  Both
        directions have happened.
        """

        sm = (f" ({self.streaming_multiprocessors} SM"
              + (f", {self.compute_capability}" if self.compute_capability
                 else "") + ")") if self.streaming_multiprocessors else ""
        return (
            f"{self.key}{sm} has NO MEASURED device footprint, so no mesh "
            "will be sized against it.\n"
            "  what IS known about this part: "
            f"{self.streaming_multiprocessors} SMs x "
            f"{self.max_threads_per_sm} resident threads derive a "
            f"{self.local_store_mib:,.0f} MiB CUDA per-context "
            "local-memory backing store from the "
            f"{self.local_store.widest_kernel_frame_bytes:,.0f} B frame of "
            f"{self.local_store.widest_kernel}.  That is the dominant "
            "term and it comes out of the card's own numbers.\n"
            "  what is NOT known: the rest of the fixed term.  "
            f"{self.local_store.what_is_not_derived}\n"
            "  the breakage this refuses: a mesh sized against a card "
            "nobody measured comes out at a cell count that is "
            "confidently wrong in a direction nobody can name.  Too high "
            "and the run dies at step 1 with an out-of-memory that blames "
            "the model; too low and the card is wasted -- borrowing the "
            "170 SM part's fixed term under-sizes a 70 SM part by about "
            "1.6x.\n"
            f"  {self.provenance}")


@dataclass(frozen=True)
class Smoothness:
    """The mesh-roughness bound, and how accurate it is about itself."""

    metric: str
    published_reference_percent_per_cell: float
    published_reference_mesh: str
    published_ramp_cells: float
    warn_above_percent_per_cell: float
    refuse_above_percent_per_cell: float
    status: str
    rule: str
    evidence: str
    moves_when: str
    breakage: str

    def check(self, gradient_percent_per_cell: float, *,
              allow_rough: bool = False) -> str | None:
        """``None`` when the mesh is as smooth as the published family.

        A note when it is rougher than that but inside the bound, and a
        :class:`MeshRoughnessError` above it.  ``allow_rough`` is an
        OPT-OUT of the refusal and is reported as a workaround in those
        words, because a correctness gate that a flag silently disables
        is not a gate.
        """

        steep = gradient_percent_per_cell
        if steep > self.refuse_above_percent_per_cell:
            if not allow_rough:
                raise MeshRoughnessError(self._refusal(steep))
            return (
                "WORKAROUND: --allow-rough-mesh was passed, so the "
                f"{steep:.2f} %/cell roughness bound of "
                f"{self.refuse_above_percent_per_cell:.2f} %/cell was NOT "
                "enforced.  This is not a supported configuration and is "
                "reported as a workaround rather than a setting.\n"
                f"  {self.breakage}\n"
                f"  {self.evidence}")
        if steep > self.warn_above_percent_per_cell:
            return (
                f"this mesh is {steep:.2f} %/cell at its steepest against "
                f"{self.published_reference_percent_per_cell:.2f} %/cell in "
                f"the published {self.published_reference_mesh}, so it is "
                "rougher than anything with dycore evidence behind it.  "
                f"NOT MEASURED whether that matters: {self.evidence}")
        return None

    def _refusal(self, steep: float) -> str:
        return (
            f"the requested mesh is {steep:.2f} %/cell at its steepest "
            f"spacing gradient, past the "
            f"{self.refuse_above_percent_per_cell:.2f} %/cell this "
            "release will emit.\n"
            f"  what breaks: {self.breakage}\n"
            f"  the reference: {self.published_reference_mesh} is "
            f"{self.published_reference_percent_per_cell:.2f} %/cell, and "
            f"the bound is {self.rule}\n"
            f"  status: {self.status} -- {self.evidence}\n"
            "  what to change: widen the transition (a fifth field on "
            "--refine, in km), coarsen the refinement, or refine a "
            "smaller area.  --allow-rough-mesh emits it anyway and is "
            "reported as a workaround.\n"
            f"  when this bound moves: {self.moves_when}")


@dataclass(frozen=True)
class DeliveredSmoothness:
    """What the emitted mesh measures, and why it is not the request.

    ``max_adjacent_spacing_ratio`` is a maximum over every edge in the
    delivered mesh.  The requested gradient is the slope of a smooth
    sampled density field.  The first cut of this door judged the first
    by the second's bound, and a control settled it: a UNIFORM
    12,000-cell mesh -- requested gradient 0.00 %/cell, no refinement
    region at all -- delivers 1.1407, which is 14.07 % on the requested
    scale.  The check therefore fired on every mesh ever emitted.

    So there is no bound here, and the absence is declared rather than
    filled with a plausible number.  The reading is REPORTED beside the
    control that gives it a scale.
    """

    metric: str
    uniform_control_ratio: float
    uniform_control_provenance: str
    bound: float | None
    evidence: str
    moves_when: str

    @property
    def uniform_control_percent(self) -> float:
        return (self.uniform_control_ratio - 1.0) * 100.0

    def report(self, ratio: float) -> str:
        """The delivered reading, with the scale that makes it readable."""

        percent = (float(ratio) - 1.0) * 100.0
        return (
            f"delivered max adjacent spacing ratio {ratio:.4f} "
            f"({percent:.2f} % between the two most different neighbours).  "
            f"For scale, a UNIFORM mesh with no refinement at all measures "
            f"{self.uniform_control_ratio:.4f} "
            f"({self.uniform_control_percent:.2f} %) on this metric, so it "
            "is mostly the tessellation's own texture and is NOT the "
            f"requested-gradient number above.  {self.evidence}")


@dataclass(frozen=True)
class Sizing:
    """The whole table: the cards, the quality bound, the defaults."""

    cards: dict[str, CardCapacity]
    smoothness: Smoothness
    delivered: DeliveredSmoothness
    transition_cells: float
    source: Path
    capacity_note: str = ""

    def card(self, key: str) -> CardCapacity:
        """One card row, or a refusal listing the ones that exist."""

        try:
            return self.cards[key]
        except KeyError:
            raise MeshRequestError(
                f"no card named {key!r} is in the sizing table.  Known "
                f"cards: {', '.join(sorted(self.cards))}.  A card is a row "
                f"in {self.source.name}, not a code path -- measuring a "
                "new one and filling in its fixed_mib, bytes_per_cell and "
                "anchors is the whole cost of adding it."
            ) from None

    def measured_cards(self) -> list[CardCapacity]:
        return [c for c in self.cards.values() if c.measured]

    def cells_that_fit(self, key: str,
                       budget_mib: float | None = None) -> int:
        """Largest cell count ``key`` can hold, or a refusal by name."""

        card = self.card(key)
        if not card.measured:
            measured = ", ".join(c.key for c in self.measured_cards()) \
                or "none"
            raise MeshRequestError(
                card.unmeasured_refusal()
                + f"\n  what does work: --card with a measured row "
                  f"({measured}), or --cells N to state the size directly "
                  "and skip the device model altogether.")
        return card.cells_that_fit(
            card.sizing_budget_mib if budget_mib is None else budget_mib)

    def check_fits(self, key: str, *, cells: int,
                   budget_mib: float | None = None) -> float:
        """The footprint, or a refusal naming the card and what fits."""

        card = self.card(key)
        budget = (card.sizing_budget_mib if budget_mib is None
                  else budget_mib)
        footprint = card.footprint_mib(cells)
        if footprint <= budget:
            return footprint
        fits = card.cells_that_fit(budget)
        raise MeshRequestError(
            f"{cells:,} cells needs {footprint:,.0f} MiB on {card.key} "
            f"({card.display_name}), which has {budget:,.0f} MiB.\n"
            f"  what fits on this card: {fits:,} cells "
            f"({card.footprint_mib(fits):,.0f} MiB).\n"
            f"  the model: {card.fixed_mib:,.0f} MiB fixed + "
            f"{card.bytes_per_cell:,.0f} B/cell, measured "
            f"{card.measured_on_text()}.\n"
            "  what to change: refine a smaller area, coarsen the "
            "refinement or the background, pass --cells with a number "
            "that fits, or name a bigger card.")


_SIZING_CACHE: dict[str, Sizing] = {}


def load_sizing(path: Path | None = None) -> Sizing:
    """Read and validate the sizing table.

    Cached by path: this is read at every ``woof mesh`` invocation and
    by ``woof doctor``, and it never changes inside one process.
    """

    resolved = SIZING_DATA_PATH if path is None else Path(path)
    key = str(resolved)
    if key in _SIZING_CACHE:
        return _SIZING_CACHE[key]
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as error:
        raise MeshRequestError(
            f"the MPAS mesh sizing table {resolved} is unreadable "
            f"({error}), so no request can be sized against any card and "
            "no quality bound can be applied.  It is package data; a "
            "reinstall replaces it.") from None
    except ValueError as error:
        raise MeshRequestError(
            f"the MPAS mesh sizing table {resolved} is not JSON: {error}"
        ) from None
    if document.get("schema") != SIZING_SCHEMA:
        raise MeshRequestError(
            f"{resolved} declares schema {document.get('schema')!r}, "
            f"expected {SIZING_SCHEMA!r}")
    capacity = document.get("capacity") or {}
    store_row = capacity.get("local_memory_backing_store")
    if not store_row:
        raise MeshRequestError(
            f"{resolved} carries no local_memory_backing_store block, so "
            "no card's fixed term can be derived.  Without it the table "
            "would have to carry a fixed term per card as a bare number, "
            "which is how one part's number came to be quoted for every "
            "part.")
    local_store = LocalStore(
        formula=store_row["formula"],
        widest_kernel=store_row["widest_kernel"],
        widest_kernel_frame_bytes=float(
            store_row["widest_kernel_frame_bytes"]),
        frame_credit_bytes=float(store_row["frame_credit_bytes"]),
        measured_against_arwen_commit=store_row[
            "measured_against_arwen_commit"],
        frame_cut_commit=store_row["frame_cut_commit"],
        validation=store_row.get("validation", ""),
        what_is_not_derived=store_row.get("what_is_not_derived", ""),
        why_it_cannot_move_alone=store_row.get(
            "why_it_cannot_move_alone", ""))
    cards: dict[str, CardCapacity] = {}
    for card_key, row in sorted((capacity.get("cards") or {}).items()):
        cards[card_key] = CardCapacity(
            key=card_key,
            display_name=row.get("display_name", card_key),
            device_total_mib=float(row["device_total_mib"]),
            device_addressable_mib=(
                None if row.get("device_addressable_mib") is None
                else float(row["device_addressable_mib"])),
            streaming_multiprocessors=row.get("streaming_multiprocessors"),
            max_threads_per_sm=row.get("max_threads_per_sm"),
            compute_capability=row.get("compute_capability"),
            measured=bool(row.get("measured")),
            local_store=local_store,
            local_store_residue_mib=(
                None if row.get("local_store_residue_mib") is None
                else float(row["local_store_residue_mib"])),
            residue_measured_against_arwen_commit=row.get(
                "residue_measured_against_arwen_commit"),
            bytes_per_cell=(None if row.get("bytes_per_cell") is None
                            else float(row["bytes_per_cell"])),
            bytes_per_cell_provenance=row.get(
                "bytes_per_cell_provenance", ""),
            measured_on=row.get("measured_on"),
            anchors=tuple(row.get("anchors") or ()),
            provenance=row.get("provenance", ""),
            notes=row.get("notes", ""))
    raw = document.get("smoothness") or {}
    smoothness = Smoothness(
        metric=raw["metric"],
        published_reference_percent_per_cell=float(
            raw["published_reference_percent_per_cell"]),
        published_reference_mesh=raw["published_reference_mesh"],
        published_ramp_cells=float(raw["published_ramp_cells"]),
        warn_above_percent_per_cell=float(raw["warn_above_percent_per_cell"]),
        refuse_above_percent_per_cell=float(
            raw["refuse_above_percent_per_cell"]),
        status=raw["status"], rule=raw["rule"], evidence=raw["evidence"],
        moves_when=raw["moves_when"], breakage=raw["breakage"])
    raw_delivered = document.get("delivered_smoothness") or {}
    delivered = DeliveredSmoothness(
        metric=raw_delivered["metric"],
        uniform_control_ratio=float(raw_delivered["uniform_control_ratio"]),
        uniform_control_provenance=raw_delivered[
            "uniform_control_provenance"],
        bound=(None if raw_delivered.get("bound") is None
               else float(raw_delivered["bound"])),
        evidence=raw_delivered["evidence"],
        moves_when=raw_delivered["moves_when"])
    sizing = Sizing(
        cards=cards, smoothness=smoothness, delivered=delivered,
        transition_cells=float(
            (document.get("defaults") or {}).get("transition_cells", 81.0)),
        source=resolved,
        capacity_note=capacity.get("why_the_fixed_term_is_per_card", ""))
    _SIZING_CACHE[key] = sizing
    return sizing


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------

def _number(text: str, what: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise MeshRequestError(
            f"--refine: {what} is {text!r}, which is not a number") from None


def build_spec(*, background_km: float, refine: list[str] | None = None,
               refine_box: list[str] | None = None,
               name: str | None = None,
               transition_cells: float | None = None) -> dict:
    """Turn the command line into the resolution spec the binary reads.

    ``--refine LAT,LON,RADIUS_KM,SPACING_KM[,TRANSITION_KM]`` is one
    row.  ``--refine-box LAT0,LAT1,LON0,LON1,SPACING_KM[,TRANSITION_KM]``
    is the same thing over a latitude/longitude box.  Both append to
    ``regions``, which is why a new place to refine costs a row here, a
    row in the spec, and nothing else anywhere.

    The transition width defaults to the published family's ramp
    (81 cells of the region's own spacing), because that is the ramp the
    meshes with dycore evidence behind them are smooth at.  A fifth (or
    seventh) field states it in kilometres instead.
    """

    # `> 0` alone let infinity through, and an infinite spacing has no
    # cell count either.
    if not (background_km > 0 and background_km != float("inf")):
        raise MeshRequestError(
            f"--background-km is {background_km}; the background spacing "
            "sets the cell size over most of the sphere, and a "
            "non-positive or infinite one has no cell count")
    ramp = (load_sizing().transition_cells if transition_cells is None
            else transition_cells)
    regions: list[dict] = []
    for row in refine or []:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) not in (4, 5):
            raise MeshRequestError(
                f"--refine {row!r} has {len(parts)} field(s); it takes "
                "LAT,LON,RADIUS_KM,SPACING_KM and optionally a fifth "
                "TRANSITION_KM (example: --refine 39.0,-98.0,1200,15)")
        lat = _number(parts[0], "latitude")
        lon = _number(parts[1], "longitude")
        radius = _number(parts[2], "radius_km")
        spacing = _number(parts[3], "spacing_km")
        region: dict = {
            "shape": {"kind": "cap", "center_deg": [lat, lon],
                      "radius_km": radius},
            "spacing_km": spacing,
        }
        if len(parts) == 5:
            region["transition_km"] = _number(parts[4], "transition_km")
        else:
            region["transition_cells"] = ramp
        regions.append(region)
    for row in refine_box or []:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) not in (5, 6):
            raise MeshRequestError(
                f"--refine-box {row!r} has {len(parts)} field(s); it takes "
                "LAT0,LAT1,LON0,LON1,SPACING_KM and optionally a sixth "
                "TRANSITION_KM")
        region = {
            "shape": {"kind": "lat_lon_box",
                      "lat_deg": [_number(parts[0], "lat0"),
                                  _number(parts[1], "lat1")],
                      "lon_deg": [_number(parts[2], "lon0"),
                                  _number(parts[3], "lon1")]},
            "spacing_km": _number(parts[4], "spacing_km"),
        }
        if len(parts) == 6:
            region["transition_km"] = _number(parts[5], "transition_km")
        else:
            region["transition_cells"] = ramp
        regions.append(region)
    for index, region in enumerate(regions):
        spacing = region["spacing_km"]
        if spacing <= 0:
            raise MeshRequestError(
                f"refinement {index} asks for {spacing} km spacing; a "
                "non-positive spacing has no cell count")
        if spacing >= background_km:
            raise MeshRequestError(
                f"refinement {index} asks for {spacing:g} km inside a "
                f"{background_km:g} km background, so the 'refinement' is "
                "the same size or coarser than everywhere else.  A "
                "background finer than the region it surrounds spends the "
                "whole cell budget on the part of the sphere nobody asked "
                "to resolve; swap the two numbers, or drop the region and "
                "ask for a uniform mesh with --background-km alone.")
    spec: dict = {"background_km": float(background_km), "regions": regions}
    if name:
        spec["name"] = name
    return spec


# ---------------------------------------------------------------------------
# Driving the binary
# ---------------------------------------------------------------------------

def _json_tail(stdout: str, *, what: str) -> dict:
    """The JSON record at the end of the binary's output.

    ``rw_mpas_mesh`` prints one tab-separated progress line per stage
    and then the receipt, so the record is the tail rather than the
    whole stream.  A ``--dry-run`` prints the record alone and lands on
    the same code path.
    """

    lines = stdout.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("{"):
            try:
                return json.loads("\n".join(lines[index:]))
            except ValueError as error:
                raise MeshRequestError(
                    f"{what} printed something that starts like a JSON "
                    f"record and is not one: {error}") from None
    raise MeshRequestError(
        f"{what} printed no JSON record.  This is not the binary this "
        f"release drives; rebuild it: {rustwx_build_hint()}")


def _run_mesh(arguments: list[str], *, progress=None) -> dict:
    """Run ``rw_mpas_mesh`` once and return its JSON record."""

    binary = MESH.require()
    ok, evidence = MESH.probe(binary)
    if not ok:
        raise MeshRequestError(f"{binary}: {evidence}")
    command = [str(binary), *arguments]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                errors="replace")
    except OSError as error:
        raise MeshRequestError(
            f"rw_mpas_mesh failed to launch: {error}") from error
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        head = detail[0] if detail else f"exit {result.returncode}"
        raise MeshRequestError(f"rw_mpas_mesh refused: {head}")
    if progress is not None:
        for line in (result.stdout or "").splitlines():
            if line and not line.startswith("{") and "\t" in line:
                progress(line)
    return _json_tail(result.stdout or "", what="rw_mpas_mesh")


def _spec_file(spec: dict, workdir: Path) -> Path:
    path = Path(workdir) / "mesh-spec.json"
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return path


def plan(*, spec: dict, cells: int, workdir: Path) -> dict:
    """Size and cost the request without writing a mesh.

    The plan is what the quality and capacity gates are applied to, and
    it comes from the BINARY rather than from a Python re-derivation of
    the density field: the number that matters -- the steepest requested
    gradient -- is a property of the same sampled density function the
    generator relaxes against, and a second implementation of it here
    would be a second answer nobody reconciles.
    """

    return _run_mesh(["--spec", str(_spec_file(spec, workdir)),
                      "--cells", str(int(cells)), "--dry-run"])


#: The two arms of the generator.
#:
#: ``incremental`` is the binary's DEFAULT (2026-10-16: no
#: single-threaded stages): the facet list is kept and repaired by Lawson
#: flips.  ``rebuild`` rebuilds the spherical Delaunay from scratch after
#: every relaxation sweep, which is slower.  Neighbour rings start at their
#: smallest neighbour index, so both arms write the same bytes; meshes
#: registered before 2026-10-16 used the old ring rotation and are not
#: reproduced byte for byte by either.  The door passes an arm only when one
#: is asked for.
TRIANGULATION_ARMS = ("rebuild", "incremental")


def generate(*, spec: dict, cells: int, out: Path, workdir: Path,
             clobber: bool = False, receipt: Path | None = None,
             sweeps: int | None = None, tolerance: float | None = None,
             triangulation: str | None = None,
             progress=None) -> dict:
    """Generate the mesh and return the binary's receipt."""

    arguments = ["--spec", str(_spec_file(spec, workdir)),
                 "--cells", str(int(cells)), "--out", str(out)]
    if clobber:
        arguments.append("--clobber")
    if receipt is not None:
        arguments += ["--receipt", str(receipt)]
    if sweeps is not None:
        arguments += ["--sweeps", str(int(sweeps))]
    if tolerance is not None:
        arguments += ["--tolerance", repr(float(tolerance))]
    if triangulation is not None:
        if triangulation not in TRIANGULATION_ARMS:
            raise MeshRequestError(
                f"--triangulation {triangulation} is not an arm; it is "
                "`incremental` (the default: the triangulation kept and "
                "repaired by Lawson flips) or `rebuild` (rebuilt every sweep, "
                "slower); both write the same bytes, and meshes registered "
                "before 2026-10-16 used the old ring rotation and are not "
                "reproduced by either")
        arguments += ["--triangulation", triangulation]
    return _run_mesh(arguments, progress=progress)


#: Static fields whose absence stops the MPAS GPU port.
#:
#: EXACTLY ONE, and the list is now measured rather than guessed at from
#: which fields the published static happens to carry.  Searching the
#: port's whole tree finds one read of ``deriv_two``
#: (``transport.py:469``, into ``build_advection_coefficients``) and NO
#: read of ``cell_gradient_coef_x`` or ``cell_gradient_coef_y`` anywhere --
#: those two were on this list because a published static carries them,
#: which is not the same question.  ``defc_a`` and ``defc_b`` are read, but
#: they are absent from the published ``x4.163842.static.nc`` as well and
#: the mesh binding that admits it drops them by name.
#:
#: The engine WRITES ``deriv_two`` now, so this tuple is the check that
#: keeps a regression from being reported as a runnable pair, not a
#: standing gap.
PORT_BLOCKING_STATIC_FIELDS = ("deriv_two",)


def unwritten_operator_fields_from_receipt(receipt) -> list[str] | None:
    """The engine's absent list, read from what the engine actually wrote.

    THE BREAKAGE THIS FIXES (found 2026-09-02 while proving the soil-archive
    fix on a real pair): the door read ``unwritten_operator_fields`` and
    ``unwritten_soil_composition_fields`` out of the static receipt, and the
    unified static writer (receipt schema ``rw-mpas-static.build/v3-unified``)
    has never written either key.  It writes ``variables``, every variable it
    put in the file.  So every pair the door built since that writer landed
    was announced "NOT MEASURED: the static engine's absent-field list did
    not come back", with both the published binary and a fresh build, and
    the runnability verdict the door exists to give was never given.

    The absent list is therefore DERIVED: the fields the port reads that are
    not in the written list.  A receipt that still carries the old keys is
    read as before, so an archived receipt keeps its meaning.  A receipt with
    neither is unreadable and stays ``None``, which the verdict reports as
    not measured rather than as runnable.
    """

    if receipt is None:
        return None
    legacy = receipt.get("unwritten_operator_fields")
    if legacy is not None:
        absent = list(legacy)
        absent += list(receipt.get("unwritten_soil_composition_fields") or ())
        return absent
    variables = receipt.get("variables")
    if variables is None:
        return None
    from woof.rustwx_static import FORMERLY_UNWRITTEN_FIELDS

    written = set(variables)
    wanted = list(PORT_BLOCKING_STATIC_FIELDS) + [
        name for name in FORMERLY_UNWRITTEN_FIELDS
        if name not in PORT_BLOCKING_STATIC_FIELDS]
    return [name for name in wanted if name not in written]


def _short_edge_refusal(agreement) -> str | None:
    """The pair is complete but the port's own gates would still stop it.

    THE LIVE CONTRACT (the port changed it on 2026-08-23; this door kept
    quoting the retired one until the stale-guard audit of 2026-08-25,
    finding 3, caught it):

    * STORAGE: the port recomputes ``dvEdge`` from the stored vertices
      and compares with an ABSOLUTE floor of ``2*sqrt(3)`` coordinate
      ULPs -- 1.73 m for FP32 at earth radius -- plus a ~9.5e-7 rtol
      (``mpas_port/mesh.py``, ``spherical_arc_tolerance``).  The retired
      ``rtol 2e-5 / atol 0.0`` comparison no longer exists.  MEASURED
      2026-08-25: a generated 654,432-cell pair carrying 12,732 edges
      past the retired 2e-5 bound (worst 4.64e-5 relative, 0.634 m
      absolute) is ACCEPTED whole by the live
      ``Mesh.from_netcdf(validate=True)``.
    * RATIO: short dual edges are gated by ``DualEdgePolicy``:
      ``dvEdge/dcEdge >= 0.02``, refused before any CUDA allocation with
      ``DualEdgeAdmissionError``.  The TRiSK tangential terms divide by
      ``dvEdge``, so a ratio of 6.83e-5 amplifies a tangential gradient
      ~15,000x, and the measured runaway in the first outer step sits on
      the worst such edge.

    Reported rather than refused at the door: the files are written and a
    consumer with a looser gate can use them.  What must not happen is
    the door quoting a comparison the port stopped performing.
    """

    if not isinstance(agreement, dict):
        agreement = {}
    worst_abs = agreement.get("max_dv_edge_absolute_m")
    atol = agreement.get("port_arc_atol_m")
    ratio = agreement.get("min_dv_over_dc")
    floor = agreement.get("port_min_dv_over_dc")
    if worst_abs is None or atol is None or ratio is None or floor is None:
        # Silence is not evidence of health, and neither is a reading
        # taken against the RETIRED contract: an engine that judged
        # storage at rtol 2e-5 / atol 0.0 answered a question the port
        # stopped asking on 2026-08-23.
        return ("NOT MEASURED whether this pair reaches the MPAS port: the "
                "static engine reported no FP32 metric agreement against "
                "the port's live contract (the 1.73 m storage atol and the "
                "0.02 dvEdge/dcEdge floor)."
                + ("  This receipt's reading was judged against the retired "
                   "rtol 2e-5 comparison the port no longer performs, so it "
                   "cannot be quoted either way."
                   if agreement.get("consumer_metric_rtol") is not None
                   else "")
                + "  Rebuild the static with a current engine, or re-run "
                  "with --receipt and read fp32_metric_agreement there.")
    past_storage = int(agreement.get("edges_past_port_storage_tolerance", 0))
    if past_storage:
        return (
            "NOT RUNNABLE through the MPAS port: "
            f"{past_storage:,} edge(s) disagree with their own stored "
            f"vertices by more than the port's {float(atol):.2f} m storage "
            f"tolerance (worst {float(worst_abs):.2f} m).  The port "
            "recomputes dvEdge from the stored vertices with an absolute "
            "floor of 2*sqrt(3) coordinate ULPs and refuses the whole "
            "pair.  A generated pair sits under one coordinate rounding "
            "(0.87 m) by construction, so a reading past 1.73 m means the "
            "file was rescaled or edited after generation, not that the "
            "mesh is too fine.")
    if float(ratio) < float(floor):
        below = int(agreement.get("edges_below_admission_floor", 0))
        return (
            "NOT RUNNABLE through the MPAS port: this mesh's worst dual "
            f"edge is {float(agreement.get('min_dv_edge_m', 0.0)):,.1f} m, "
            f"dvEdge/dcEdge = {float(ratio):.2e} against the port's "
            f"admitted floor of {float(floor):g} ({below:,} edge(s) below "
            "it).  The TRiSK tangential terms divide by dvEdge, so a "
            f"gradient across that edge is amplified {1.0 / max(float(ratio), 1e-300):,.0f}x "
            "over a normal one, and the port refuses the whole pair with "
            "DualEdgeAdmissionError before any CUDA allocation.\n"
            "what this is: a pentagon-heptagon dislocation -- two dual "
            "vertices that nearly coincide.  A uniform request does not "
            "produce one (the icosahedral seed has no dislocations); "
            "MEASURED 2026-08-25, re-relaxing the same refinement request "
            "re-rolls the dislocation rather than draining it.\n"
            "what does work, measured: a different refinement layout, or "
            "fewer cells in the refined region.")
    return None


def runnability_verdict(unwritten_operator_fields,
                        fp32_metric_agreement=None) -> str:
    """What this pair can and cannot do, said before anyone tries it.

    The line this replaced read "this pair is what a run needs: grid and
    static together".  A real forecast disproved it.  A sentence that
    survives its own disproof is how engine-proven gets reported as
    shipped, so the verdict is computed from the engine's own absent list
    rather than asserted.
    """

    if unwritten_operator_fields is None:
        # The optimistic branch is not the safe default.  A door that
        # answers "runnable" whenever it failed to read the engine's list
        # is a door that says the encouraging thing when it knows least,
        # which is how the disproved sentence survived the first time.
        return ("NOT MEASURED: the static engine's absent-field list did "
                "not come back, so whether this pair reaches the MPAS "
                "port is unknown.  Read rw_static_provenance_json out of "
                "the static, or re-run with --receipt.")
    blocking = [name for name in unwritten_operator_fields
                if name in PORT_BLOCKING_STATIC_FIELDS]
    if not blocking:
        short = _short_edge_refusal(fp32_metric_agreement)
        if short:
            return short
        return ("this pair is what a run needs: grid and static together, "
                "deriv_two included")
    return (
        "NOT YET RUNNABLE through the MPAS port: "
        f"{', '.join(blocking)} carry real values in a published static "
        "and this engine writes neither the values nor the slot.  A run "
        "stops in transport.build_advection_coefficients before any "
        "device memory is allocated.  Build the static with MPAS "
        "init_atmosphere (config_static_interp = true) on this grid until "
        "that gap closes.")


def default_static_path(grid: Path) -> Path:
    """Where the static goes when ``--static-out`` was not given.

    Beside the grid, sharing its stem, because the mesh registry admits
    the two as a pair and a user who moves one and not the other has a
    grid that no longer runs.
    """

    grid = Path(grid)
    name = grid.name
    for suffix in (".grid.nc", ".grid", ".nc"):
        if name.endswith(suffix):
            return grid.with_name(name[: -len(suffix)] + ".static.nc")
    return grid.with_name(name + ".static.nc")


def build_static(*, grid: Path, out: Path, geog: Path | None = None,
                 nominal_dx_m: float | None = None, clobber: bool = False,
                 receipt: Path | None = None, progress=None,
                 legacy_static: bool = False) -> dict:
    """Build the static half of the pair and return the engine's document.

    Separated from :func:`generate` because the two binaries are separate
    artifacts with separate contracts, and joined at the door because a
    grid without its static is not a deliverable: the mesh registry pins
    both by byte count and SHA-256 and refuses a mesh carrying one.

    The static is MPAS-A v8.4.2's native one by default (2.9.0).
    ``legacy_static`` asks for the pre-2.9.0 stream (82 variables, the
    with-lakes land use with its lake class stored as water), for a
    consumer that still needs that schema.  It is not the bytes
    of a pre-2.9.0 build: 2.9.0's shared static arithmetic applies to it.
    """

    from woof import rustwx_static

    binary = STATIC.require()
    ok, evidence = rustwx_static.probe_static_bin(binary)
    if not ok:
        raise MeshRequestError(f"{binary}: {evidence}")
    argv = ["--grid", str(grid), "--out", str(out)]
    if geog is not None:
        argv += ["--geog", str(geog)]
    if nominal_dx_m is not None:
        argv += ["--nominal-dx-m", repr(float(nominal_dx_m))]
    if receipt is not None:
        argv += ["--receipt", str(receipt)]
    if clobber:
        argv.append("--clobber")
    if legacy_static:
        argv.append("--legacy-static")
    try:
        return rustwx_static.run_static(binary, argv, on_progress=progress)
    except rustwx_static.StaticEngineError as error:
        raise MeshRequestError(f"rw_mpas_static refused: {error}") from None


def static_geog_refusal(geog: Path | None) -> tuple[Path | None, str | None]:
    """The archive the static will read, and the refusal if it cannot.

    Answered BEFORE the relaxation runs.  The whole point: a mesh is
    minutes of relaxation, and discovering afterwards that no geography
    archive is reachable would spend all of it to deliver half a pair.
    """

    from woof import rustwx_static

    root = Path(geog) if geog is not None \
        else rustwx_static.default_geog_root()
    if geog is not None and not root.is_dir():
        return root, (
            f"--geog names {root}, which is not a directory.  The static "
            "half of the pair reads terrain, land use, soil, green-ness "
            "and albedo from a WPS_GEOG archive, and the mesh registry "
            "refuses a grid that has no matching static.")
    return root, rustwx_static.geog_refusal(root)


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Make an MPAS mesh at the resolution you ask for.

Say WHERE to refine, HOW FINE there, HOW COARSE everywhere else, and
WHICH CARD it has to run on:

  woof mesh --out kansas.grid.nc --refine 39.0,-98.0,1200,15 \\
      --background-km 120 --card rtx-5090

Refinement regions are data: a second place to refine is a second
--refine row.  --dry-run sizes and costs the request and writes
nothing.  --list-cards prints the cards with a measured footprint model
and the largest mesh each one holds.

This writes a RUNNABLE PAIR: the grid file and, beside it, the matching
STATIC file (terrain, land use, soil, nVertLevels=55, nSoilLevels=4, an
FP32-bit-exact nominalMinDc, and deriv_two, the transport stencil input
a forecast stops without).  Both, because the mesh registry pins the two
together by byte count and SHA-256, so a grid alone reaches no dycore.
The static reads a WPS_GEOG archive; --geog names it, and the default is
wherever `woof fetch-geog` stages -- $GPUWM_WPS_GEOG first, then
$GPUWM_CASE_DATA_ROOT/WPS_GEOG.  The refusal prints every path it looked
at.
"""


def build_parser() -> argparse.ArgumentParser:
    """The mesh door's own parser, standalone for the docs generator."""

    parser = argparse.ArgumentParser(
        prog="woof mesh", description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_arguments(parser)
    return parser


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out", type=Path, default=None, metavar="GRID.nc",
        help="grid file to write (not needed with --dry-run or "
             "--list-cards)")
    parser.add_argument(
        "--refine", action="append", default=[], metavar="LAT,LON,KM,KM",
        help="refine a circle: LAT,LON,RADIUS_KM,SPACING_KM and "
             "optionally a fifth TRANSITION_KM.  Repeatable -- each row "
             "is one region and adding one is data, not a code path")
    parser.add_argument(
        "--refine-box", action="append", default=[],
        metavar="LAT0,LAT1,LON0,LON1,KM",
        help="refine a latitude/longitude box: "
             "LAT0,LAT1,LON0,LON1,SPACING_KM and optionally a sixth "
             "TRANSITION_KM.  Repeatable")
    parser.add_argument(
        "--background-km", type=float, default=None, metavar="KM",
        help="cell spacing far from every refinement region; with no "
             "--refine this is a uniform mesh at that spacing")
    parser.add_argument(
        "--spec", type=Path, default=None, metavar="SPEC.json",
        help="read the whole resolution spec from a JSON document "
             "instead of building it from --background-km/--refine")
    parser.add_argument(
        "--card", default=None, metavar="NAME",
        help="size the mesh to this card's measured device footprint; "
             "--list-cards prints the ones that have been measured")
    parser.add_argument(
        "--vram-gib", type=positive_float, default=None, metavar="X",
        help="device budget in GiB, instead of the named card's total "
             "memory (for a card that is shared with something else); "
             "needs --card, because the fixed term is per card")
    parser.add_argument(
        "--cells", type=positive_int, default=None, metavar="N",
        help="exact cell count, skipping the device model entirely")
    parser.add_argument(
        "--name", default=None, metavar="TEXT",
        help="label carried into the grid file's provenance attributes")
    parser.add_argument(
        "--receipt", type=Path, default=None, metavar="JSON",
        help="write the measured receipt here as well as to stdout")
    parser.add_argument(
        "--sweeps", type=positive_int, default=None, metavar="N",
        help="relaxation budget passed to the generator")
    parser.add_argument(
        "--tolerance", type=positive_float, default=None, metavar="X",
        help="relaxation convergence tolerance passed to the generator")
    parser.add_argument(
        "--triangulation", choices=TRIANGULATION_ARMS, default=None,
        help="how the Delaunay is kept between relaxation sweeps.  "
             "incremental (the default) keeps the facets and repairs them "
             "by Lawson flips; rebuild rebuilds it every sweep (slower).  "
             "Both write the same bytes")
    parser.add_argument(
        "--clobber", action="store_true",
        help="replace an existing --out")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="size and cost the request, apply both gates, write nothing")
    parser.add_argument(
        "--list-cards", action="store_true",
        help="print the cards in the sizing table and the largest mesh "
             "each measured one holds, then exit")
    parser.add_argument(
        "--allow-rough-mesh", action="store_true",
        help="WORKAROUND: emit a mesh rougher than the stated smoothness "
             "bound.  Reported as a workaround in the receipt, never as "
             "a setting")
    parser.add_argument(
        "--static-out", type=Path, default=None, metavar="STATIC.nc",
        help="where the matching static goes.  Defaults to --out with "
             "a .static.nc suffix beside the grid, because the mesh "
             "registry admits the two as a pair")
    parser.add_argument(
        "--geog", type=Path, default=None, metavar="DIR",
        help="WPS_GEOG archive the static's terrain, land use, soil, "
             "green-ness and albedo come from.  Defaults to "
             "$GPUWM_WPS_GEOG, then ~/.local/share/woof/WPS_GEOG")
    parser.add_argument(
        "--nominal-dx-m", type=positive_float, default=None, metavar="M",
        help="the nominal spacing the static DECLARES, in metres.  "
             "Defaults to the grid's own implied value.  This scalar is "
             "compared FP32-bit-exactly by the mesh registry, so a "
             "declaration the grid disagrees with is refused")
    parser.add_argument(
        "--no-static", dest="static", action="store_false", default=True,
        help="WORKAROUND: write only the grid.  The result is NOT "
             "runnable -- the mesh registry pins grid AND static, so a "
             "lone grid file is refused before any dycore sees it")
    parser.set_defaults(func=mesh_main)


def register_cli(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "mesh",
        help="make an MPAS mesh at the resolution you ask for -- where to "
             "refine, how fine there, how coarse elsewhere, and which "
             "card it has to fit",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_arguments(parser)
    return parser


def _print_cards(sizing: Sizing) -> int:
    print(f"cards in {sizing.source}")
    for card in sizing.cards.values():
        if not card.measured:
            print(f"  {card.key:<14} {card.display_name}")
            print(f"                 NOT MEASURED -- {card.provenance}")
            continue
        fits = card.cells_that_fit(card.sizing_budget_mib)
        print(f"  {card.key:<14} {card.display_name}")
        addressable = ("addressable" if card.device_addressable_mib
                       is not None else "nvidia-smi total, NOT MEASURED "
                       "through CUDA")
        print(f"                 {card.sizing_budget_mib:,.0f} MiB "
              f"({addressable}); {card.streaming_multiprocessors} SM x "
              f"{card.max_threads_per_sm} threads")
        print(f"                 model {card.fixed_mib:,.0f} MiB fixed "
              f"({card.local_store_mib:,.0f} derived store + "
              f"{card.local_store_residue_mib:,.0f} measured residue) + "
              f"{card.bytes_per_cell:,.0f} B/cell; holds {fits:,} cells")
        for anchor in card.anchors:
            print(f"                 anchor {anchor['mesh']}: "
                  f"{anchor['cells']:,} cells measured at "
                  f"{anchor['process_mib']:,.1f} MiB")
    print(f"\nsmoothness bound ({sizing.smoothness.status}): refuse above "
          f"{sizing.smoothness.refuse_above_percent_per_cell:.2f} %/cell, "
          f"published reference "
          f"{sizing.smoothness.published_reference_percent_per_cell:.2f} "
          f"%/cell")
    return 0


def _resolve_cells(args, sizing: Sizing) -> tuple[int, str]:
    """The cell count and the sentence that says where it came from."""

    # Before the --cells branch, not after it: with an explicit count
    # and no card, a stated budget used to be dropped without a word,
    # although it is exactly as unusable there as it is here.
    if args.vram_gib is not None and not args.card:
        raise MeshRequestError(
            "--vram-gib is a budget, not a model: the fixed part of the "
            "footprint is a property of the CARD (it is the CUDA "
            "local-memory backing store, sized from resident-thread "
            "capacity), so a budget alone cannot say how many cells fit.  "
            "Name the card as well: --card NAME --vram-gib X.")
    if args.cells is not None:
        if args.card:
            budget = (None if args.vram_gib is None
                      else args.vram_gib * 1024.0)
            footprint = sizing.check_fits(args.card, cells=args.cells,
                                          budget_mib=budget)
            return args.cells, (
                f"{args.cells:,} cells, stated with --cells; "
                f"{footprint:,.0f} MiB on {args.card}")
        return args.cells, f"{args.cells:,} cells, stated with --cells"
    if args.card:
        budget = None if args.vram_gib is None else args.vram_gib * 1024.0
        cells = sizing.cells_that_fit(args.card, budget)
        card = sizing.card(args.card)
        where = ("--vram-gib" if args.vram_gib is not None
                 else f"{card.sizing_budget_mib:,.0f} MiB addressable")
        return cells, (
            f"{cells:,} cells, the most {args.card} holds ({where}); "
            f"{card.footprint_mib(cells):,.0f} MiB")
    raise MeshRequestError(
        "the mesh has no size: pass --card NAME to fill a card, or "
        "--cells N to state the count.  `woof mesh --list-cards` prints "
        "the cards with a measured footprint model.")


def gradient_gate_reading(proposed) -> float:
    """The plan's steepest-gradient number, or a named refusal.

    The receipt contract (rw-mpas ``mesh/mod.rs``) says a consumer must
    refuse a plan whose ``gradient_probe_coverage`` is missing or not
    ``"complete"`` -- and this door used to read the gradient with
    ``.get(..., 0.0)``, which hands a MISSING measurement the friendliest
    possible verdict.  THE BREAKAGE THIS PREVENTS: an engine that sampled
    the gradient on a sphere-uniform lattice could not see a refinement
    transition narrower than its own ~101 km point spacing and reported
    the flat background it landed on; a spec truly at 65.2 %/cell read
    8.3 and was admitted at 7.3x the build ceiling.  A missing coverage
    word means an engine from before the repaired meter; ``"partial"``
    means the probe budget could not cover this spec's own transition
    shells, and an unmeasured ramp is not a gentle one.
    """

    gradient = proposed.get("steepest_requested_gradient_percent_per_cell")
    if gradient is None:
        raise MeshRequestError(
            "the plan carries no "
            "steepest_requested_gradient_percent_per_cell, so the "
            "smoothness gate cannot be applied and this door will not "
            "invent a 0.0 for it.  Stage an rw_mpas_mesh binary that "
            "prints it (woof doctor names the staged binary).")
    coverage = proposed.get("gradient_probe_coverage")
    if coverage != "complete":
        raise MeshRequestError(
            "the plan reports steepest-gradient probe coverage "
            f"{coverage!r}, so the number beside it is not a measurement "
            "of this spec's field: a lattice-only engine stepped over "
            "every transition narrower than its ~101 km point spacing "
            "and admitted a 65.2 %/cell spec as 8.3.  Stage an "
            "rw_mpas_mesh binary that measures the gradient where the "
            "regions are, or widen the probe budget if the engine "
            "reported 'partial' for this spec.")
    return float(gradient)


def mesh_main(args) -> int:
    """``woof mesh``: plan, gate, generate."""

    try:
        sizing = load_sizing()
        if args.list_cards:
            return _print_cards(sizing)
        if args.spec is not None:
            if args.background_km is not None or args.refine \
                    or args.refine_box:
                raise MeshRequestError(
                    "--spec and --background-km/--refine both name the "
                    "resolution; give one.  A spec file that quietly lost "
                    "to a flag would write a mesh nobody asked for.")
            try:
                spec = json.loads(args.spec.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise MeshRequestError(
                    f"cannot read the resolution spec {args.spec}: "
                    f"{error}") from None
        else:
            if args.background_km is None:
                raise MeshRequestError(
                    "no resolution was given: pass --background-km for "
                    "the spacing away from every refinement (add --refine "
                    "rows for the places to refine), or --spec for a "
                    "whole spec document.")
            spec = build_spec(background_km=args.background_km,
                              refine=args.refine,
                              refine_box=args.refine_box, name=args.name)
        cells, sizing_note = _resolve_cells(args, sizing)
        if not args.dry_run and args.out is None:
            raise MeshRequestError(
                "--out was not given, and it has no default.  Add "
                "--dry-run to size and cost the request without writing a "
                "mesh.")
        # BEFORE the relaxation, not after it.  A mesh is minutes of
        # work, and a geography archive that cannot build a static is
        # knowable in milliseconds; discovering it afterwards spends the
        # whole run to deliver a grid the registry refuses.
        geog_root = None
        static_out = None
        if args.static and not args.dry_run:
            geog_root, refusal = static_geog_refusal(args.geog)
            if refusal:
                raise MeshRequestError(refusal)
            static_out = (args.static_out if args.static_out is not None
                          else default_static_path(args.out))

        with tempfile.TemporaryDirectory(prefix="gpuwm-mesh-") as scratch:
            work = Path(scratch)
            proposed = plan(spec=spec, cells=cells, workdir=work)
            steep = gradient_gate_reading(proposed)
            note = sizing.smoothness.check(
                steep, allow_rough=args.allow_rough_mesh)
            if args.card:
                sizing.check_fits(
                    args.card,
                    cells=int(proposed.get("predicted_cells", cells)),
                    budget_mib=(None if args.vram_gib is None
                                else args.vram_gib * 1024.0))
            reference = \
                sizing.smoothness.published_reference_percent_per_cell
            print(f"woof mesh: {sizing_note}")
            print(f"woof mesh: steepest requested gradient "
                  f"{steep:.2f} %/cell (published reference "
                  f"{reference:.2f} %/cell)")
            if note:
                print(f"woof mesh: {note}")
            if args.dry_run:
                proposed["gpuwm_sizing"] = sizing_note
                proposed["gpuwm_smoothness_note"] = note
                proposed["gpuwm_smoothness_status"] = \
                    sizing.smoothness.status
                print(json.dumps(proposed, indent=2))
                return 0
            receipt = generate(
                spec=spec, cells=cells, out=args.out, workdir=work,
                clobber=args.clobber, receipt=args.receipt,
                sweeps=args.sweeps, tolerance=args.tolerance,
                triangulation=args.triangulation,
                progress=lambda line: print(f"woof mesh: {line}"))
        # REPORTED, not judged.  See DeliveredSmoothness: this metric is
        # a max over every edge and reads 14.07 % on a mesh with no
        # refinement in it at all, so the requested-gradient bound is the
        # wrong ruler for it and no bound of its own has been measured.
        delivered = ((receipt.get("receipt") or {}).get("mesh") or {}).get(
            "max_adjacent_spacing_ratio")
        if delivered is not None:
            print(f"woof mesh: {sizing.delivered.report(delivered)}")
        emitted = receipt.get("emitted") or {}
        if emitted:
            print(f"woof mesh: wrote {emitted.get('path')} "
                  f"({emitted.get('bytes', 0):,} B, SHA-256 "
                  f"{str(emitted.get('sha256', ''))[:12]}...)")
        if not args.static:
            boundary = (receipt.get("receipt") or {}).get(
                "deliverable_boundary")
            if boundary:
                print(f"woof mesh: {boundary}")
            print("woof mesh: WORKAROUND --no-static: this is a grid file "
                  "alone, and the mesh registry pins grid AND static.  It "
                  "is not runnable until a matching static is built beside "
                  "it.")
            return 0
        print(f"woof mesh: building the matching static from {geog_root}")
        # The receipt is asked for whether or not the caller wanted one:
        # the runnability verdict below is computed from the engine's own
        # absent-field list, and without the receipt that list is silently
        # empty and the verdict reads as the encouraging answer.
        with tempfile.TemporaryDirectory(prefix="gpuwm-static-") as scratch:
            document = build_static(
                grid=args.out, out=static_out, geog=geog_root,
                nominal_dx_m=args.nominal_dx_m, clobber=args.clobber,
                receipt=Path(scratch) / "static-receipt.json",
                progress=lambda line: print(f"woof mesh: {line}"))
        written = document.get("static") or {}
        if written:
            print(f"woof mesh: wrote {written.get('path')} "
                  f"({written.get('bytes', 0):,} B, "
                  f"{written.get('variables', 0)} variables, SHA-256 "
                  f"{str(written.get('sha256', ''))[:12]}...)")
        declared = document.get("nominal_dx") or {}
        if declared:
            print(f"woof mesh: nominalMinDc "
                  f"{declared.get('metres', 0.0):,.6f} m declared as FP32 "
                  f"{declared.get('f32_bits', '')} -- the mesh registry "
                  "compares this scalar bit for bit")
        static_receipt = document.get("receipt") or {}
        unwritten = unwritten_operator_fields_from_receipt(static_receipt)
        absent = list(unwritten or ())
        if absent:
            print("woof mesh: absent from the static (not zero-filled): "
                  + ", ".join(absent))
        print(f"woof mesh: the pair is {args.out} + {static_out}")
        print(f"woof mesh: "
              f"{runnability_verdict(unwritten, static_receipt.get('fp32_metric_agreement'))}")
        return 0
    except MeshRoughnessError as error:
        print(f"woof mesh: REFUSED: {error}", file=sys.stderr)
        return 2
    except MeshRequestError as error:
        print(f"woof mesh: REFUSED: {error}", file=sys.stderr)
        return 2


__all__ = [
    "BRIDGES", "CONVERT", "CONVERT_ABI_MARKER", "CardCapacity", "GEOMETRY", "INIT",
    "INIT_ABI_MARKER", "LBC", "LBC_ABI_MARKER",
    "MESH", "MESH_ABI_MARKER", "MeshRequestError", "TRIANGULATION_ARMS",
    "MeshRoughnessError", "MpasBridge", "SIZING_DATA_PATH", "SIZING_SCHEMA",
    "PORT_BLOCKING_STATIC_FIELDS", "STATIC", "STATIC_ABI_MARKER", "Sizing",
    "Smoothness", "build_parser", "build_spec", "build_static", "crate_dir",
    "default_static_path", "runnability_verdict", "static_geog_refusal",
    "generate", "load_sizing", "mesh_main", "plan", "register_cli",
]
