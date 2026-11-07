"""Strict contracts for native offline parent-to-child downscaling.

This module is the file-facing half of woof's CUDA-native ``ndown``
replacement.  It deliberately separates cheap metadata validation from the
later GPU interpolation/build transaction: an archived parent series must be
proved complete, geometrically identical, regularly ordered, and sufficiently
frequent before any child state is allocated.

Both woof and stock-WRF history files are accepted.  The reader is closed
world for trajectory fields but tolerant of additional diagnostic variables.
Physics conversion is explicit.  In particular, active condensate may never
be paired with a fabricated zero number moment merely because the parent used
a different microphysics scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence
import time

from woof import netcdf_bridge
from woof.io.netcdf_serialization import netCDF4  # first use only (D-10)
import numpy as np

from woof.explain import warn
from woof.core import microphysics_transition as _mt
from woof.core import portable_math as pm
from woof.core.grid import (BaseState, compute_hybrid_coeffs,
                             finalize_vertical_coord, make_vertical_coord)
from woof.core.nest_interp import register_nest, sint
from woof.core.noahmp_libm import log1pf_array
from woof.core.state import mu_at_u_faces, mu_at_v_faces
from woof.core import constants as c
from woof.vertical_remap import (
    dry_mass_edges,
    geopotential_thickness_per_mass,
    rebuild_geopotential,
    remap_interface_values,
    remap_layer_means,
    remap_receipt,
)
from woof.ingest.lateral_bc import (
    BoundaryInterval,
    FieldBoundary,
    LateralBoundaries,
    SideBoundary,
    build_lateral_interval_from_sides,
    extract_lateral_side,
    record_built_end_frame,
)
# The SINT positive-definite fix-up, imported rather than re-spelled: the
# online nest lane owns the tolerance policy and the moment membership, and
# a second copy here would be free to drift from it.  Membership matters
# per scheme: P3's number moments ni/nr ARE members while its rime pair
# qir/qib -- small-magnitude mixing ratios -- is DELIBERATELY not
# (woof/ingest/nest_init.py::POSITIVE_DEFINITE_MOMENTS and the comment
# above it), and both lanes get that answer from the one tuple.  The RK
# time-t seeding table is shared for the same reason: nest_init's pair
# list is the census the online lane's own gates iterate, and a re-spelled
# copy here is how NSSL's moment seeds and P3's qir0/qib0 would go
# missing on exactly one of the two birth paths.
from woof.ingest.nest_init import (clamp_sint_undershoot_mapping,
                                    seed_rk_time_t_copies)


_TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
_REQUIRED_DYNAMICS = frozenset({
    "T", "U", "V", "W", "PH", "PHB", "MU", "MUB", "HGT",
    "P", "PB", "P_TOP", "ZNU", "ZNW", "QVAPOR",
})
_GEOMETRY_DIMS = (
    "west_east", "south_north", "bottom_top", "west_east_stag",
    "south_north_stag", "bottom_top_stag",
)
_GEOMETRY_ATTRS = (
    "DX", "DY", "MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON",
    "MOAD_CEN_LAT", "CEN_LAT", "CEN_LON", "POLE_LAT", "POLE_LON",
    "HYBRID_OPT", "ETAC",
)
_STATIC_GEOMETRY_FIELDS = ("HGT", "XLAT", "XLONG", "MAPFAC_M", "ZNU", "ZNW")

_MASS_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg")
_WRF_TO_STATE = MappingProxyType({
    "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
    "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg",
    "QNCLOUD": "nc", "QNRAIN": "nr", "QNICE": "ni",
    "QNSNOW": "ns", "QNGRAUPEL": "ng",
    # mp_physics=28 (Thompson aerosol-aware).  Transported scalars in WRF's
    # own Registry (Registry.EM_COMMON:3036,
    # ``scalar:qni,qnr,qnc,qnwfa,qnifa,qnbca``); QNCLOUD is already above
    # because Morrison declares the same name.  Only the names
    # ``_transported_source_fields`` asks for are ever read, so adding rows
    # here cannot change what any other scheme reads.
    "QNWFA": "nwfa", "QNIFA": "nifa",
    # mp_physics=50 (P3, one-category ice with prognostic riming).  The
    # rime MASS / rime VOLUME pair rides in the same 4-D ``scalar`` array
    # as the two number moments beside it (Registry.EM_COMMON:555/:557,
    # package ``moist:qv,qc,qr,qi;scalar:qni,qnr,qir,qib`` at :3038); its
    # qi/ni/nr rows are already above under the names Morrison and
    # Thompson declared first.  No radius rows: this map carries no
    # re_cloud/re_ice for ANY scheme, because effc/effi are per-call
    # diagnostics every scheme (P3 included, module_mp_p3.F:2280-2282)
    # rebuilds before reading, never inherited state.
    "QIR": "qir", "QIB": "qib",
    "QICE2": "qi2", "QNICE2": "ni2", "QIR2": "qir2", "QIB2": "qib2",
    # mp_physics=53 (P3 three-moment ice, :3041): its droplet number rides
    # QNCLOUD above; the advected ice moment (N*Z)**0.5 is QZI (:563).
    "QZI": "qzi",
    "QRIMEF": "qrimef",  # MP15: mass-weighted rime, transported with ice.
})

#: mp_physics=28 surface aerosol emission tendencies (# kg-1 s-1).  NOT
#: transported scalars: they are per-domain cross-step CONSTANTS, and the
#: offline child MUST inherit them from the parent rather than re-derive
#: them.  WRF's ``thompson_init`` fills ``nwfa2d`` at
#: module_mp_thompson.F:510 ONLY inside the "no initial CCN" branch
#: (:493); a child that inherits a parent's nonzero ``nwfa`` takes the
#: ``has_CCN = .TRUE.`` branch at :516-522 instead, which fills nothing.
#: So an offline mp=28 child built without these would run its entire
#: forecast with zero surface aerosol emission and nothing would raise.
#: This is the same argument, and the same resolution, that
#: ``woof/ingest/nest_init.py`` already applies to the ONLINE nest lane,
#: which SINTs both fields on the mass stagger.
_AEROSOL_SURFACE_EMISSION_WRF = ("QNWFA2D", "QNIFA2D")
_AEROSOL_SURFACE_EMISSION_STATE = MappingProxyType({
    "QNWFA2D": "nwfa2d", "QNIFA2D": "nifa2d",
})
_NSSL_WRF_TO_STATE = MappingProxyType({
    "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
    "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg", "QHAIL": "qh",
    "QNDROP": "qndrop", "QNRAIN": "qnr", "QNICE": "qni",
    "QNSNOW": "qns", "QNGRAUPEL": "qng", "QNHAIL": "qnh",
    "QNCCN": "qnn", "QVGRAUPEL": "qvolg", "QVHAIL": "qvolh",
})
#: mp_physics=9 (Milbrandt-Yau).  A THIRD scheme-qualified map, for the
#: same reason the NSSL one above is a second: QHAIL and QNHAIL are
#: declared by both milbrandt2mom (Registry.EM_COMMON:3025,
#: ``scalar:qh,qnc,qnr,qni,qns,qng,qnh``) and nssl_2mom, and the two
#: schemes bind them to DIFFERENT state fields -- MY2's hail number is
#: ``nh``, NSSL's is ``qnh`` -- so one shared map could only be wrong for
#: one of them.  woof/ingest/wrfinput.py already carries exactly these
#: rows for the same reason; this is the offline lane learning what the
#: root door knew (audit R-017).  The six number moments and the five
#: shared masses reuse the names Morrison and Thompson declared first.
_MY2_WRF_TO_STATE = MappingProxyType({
    "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
    "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg", "QHAIL": "qh",
    "QNCLOUD": "nc", "QNRAIN": "nr", "QNICE": "ni",
    "QNSNOW": "ns", "QNGRAUPEL": "ng", "QNHAIL": "nh",
})
#: mp_physics=16 (WDM6, Registry.EM_COMMON:3031, ``scalar:qnn,qnc,qnr``).
#: The FOURTH scheme-qualified map, for the reason the NSSL and MY2 ones
#: exist: QNCCN is published by WDM6 (its CCN reservoir, state ``nn``) and
#: by NSSL (``qnn``), and the two bind to different state fields.  The six
#: masses and the warm-rain number pair reuse the generic names.  This
#: row is what woof/ingest/wrfinput.py::MOISTURE_MAP already carries for
#: the root door; until it landed here the offline lane refused every
#: WDM6 parent for want of it.
_WDM6_WRF_TO_STATE = MappingProxyType({
    "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
    "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg",
    "QNCLOUD": "nc", "QNRAIN": "nr", "QNCCN": "nn",
})



def _scheme_wrf_to_state(source_mp_physics: int) -> Mapping[str, str]:
    """The wrfout-name -> state-field map for one parent scheme.

    Four schemes need their own: QHAIL/QNHAIL/QNCCN are declared by more
    than one WRF package and bind to different state fields in each, so a
    single shared map could only be right for one of them.  Everything
    else reads the generic map, whose rows are the names Morrison,
    Thompson and P3 declared.  Dispatching here rather than at each call
    site is what kept mp=9 out of the lane after the NSSL map landed
    (audit R-017).
    """

    source_mp = int(source_mp_physics)
    if source_mp == 18:
        return _NSSL_WRF_TO_STATE
    if source_mp == 9:
        return _MY2_WRF_TO_STATE
    if source_mp == 16:
        return _WDM6_WRF_TO_STATE
    if source_mp == 56:
        from woof.io.wrf_output_schema import NTU_OUTPUT_FIELDS
        return MappingProxyType({**_MY2_WRF_TO_STATE,
                                 **{row.netcdf_name: name for name,row in NTU_OUTPUT_FIELDS.items()}})
    return {**_WRF_TO_STATE, "RIMI": "rimi"} if source_mp == 13 else _WRF_TO_STATE


#: Parent microphysics schemes this offline-child route can carry.
#:
#: NOT a profile whitelist: the child's hydrometeor reading is written
#: against the transported species of each scheme, and a parent outside
#: the set has no field map to read it with.  DERIVED from the physics
#: registry's per-option ``consumers.offline_child`` rows (``same_scheme``),
#: so admitting a scheme is one row in tools/build_registry.py and never a
#: literal here; a refused row names its defect.  mp=16 (WDM6) joined the
#: set when the field map learned its scheme-qualified QNCCN row
#: (:data:`_WDM6_WRF_TO_STATE`); every other ported scheme was already in.
def _offline_child_mp_physics() -> frozenset[int]:
    from woof.physics_registry import consumer_rows_by_selector

    return frozenset(
        int(mp) for mp, row in
        consumer_rows_by_selector("microphysics", "offline_child").items()
        if row.get("same_scheme") is True)


def offline_child_refusal(mp_physics: int) -> str | None:
    """Why a same-scheme parent of ``mp_physics`` is refused, or ``None``."""

    from woof.physics_registry import consumer_rows_by_selector

    row = consumer_rows_by_selector("microphysics", "offline_child").get(
        int(mp_physics))
    if row is None:
        return (f"mp_physics={mp_physics} is not an implemented microphysics "
                "option in woof/physics_registry_v2.json")
    return None if row.get("same_scheme") is True else row.get("refusal")


OFFLINE_CHILD_MP_PHYSICS = _offline_child_mp_physics()

#: The schemes a CROSS-scheme offline conversion has a contract for, at
#: either end of the edge: every parent this lane can read whose mixed
#: nest edge the online transition kernel ports
#: (woof/core/microphysics_transition.PORTED_MP_PHYSICS).  The offline
#: conversion IS the online one -- the same ``resolve_microphysics_transition``
#: contract and the same ``microphysics_edge_field`` kernel, run on the
#: archived parent's own grid before the horizontal interpolation, in the
#: online lane's order (``TRANSITION_ORDER``) -- so the set is derived from
#: that tuple rather than re-spelled, and a scheme ratified online is
#: admitted offline in the same change.  What is NOT in it is mp=0: a
#: microphysics-off domain carries no hydrometeor inventory to close an
#: edge over, and the online code has no contract for a 0->X or X->0 edge
#: either (its import-time registry check cites exactly that absence).
PARENT_SCHEME_CONTRACT = (
    OFFLINE_CHILD_MP_PHYSICS & frozenset(_mt.PORTED_MP_PHYSICS))


def offline_cross_scheme_refusal(
        source_mp_physics: int, target_mp_physics: int) -> str | None:
    """The one refusal a mixed offline edge can still meet, or ``None``.

    A same-scheme edge is never refused here.  A mixed edge is refused
    only when one end has no hydrometeor contract at all: mp=0 transports
    vapour (and, on a woof tape, the warm-rain pair) and no scheme-owned
    species, so there is no mass to diagnose a target's moments from and
    nothing for a target of 0 to receive -- the online nest edge has no
    closure for it and this lane runs the online closure, so it refuses
    the same edge for the same reason rather than inventing one.
    """

    source_mp, target_mp = int(source_mp_physics), int(target_mp_physics)
    if source_mp == target_mp:
        return None
    outside = sorted({source_mp, target_mp} - PARENT_SCHEME_CONTRACT)
    if not outside:
        return None
    mp = outside[0]
    return (
        f"offline cross-physics conversion across the mp_physics="
        f"{source_mp} -> {target_mp} edge is REFUSED: mp_physics={mp} "
        "has no microphysics-transition contract at either end of a mixed "
        "edge.  A microphysics-off domain transports no scheme-owned "
        "hydrometeor species, so a target's number moments and rimed "
        "categories cannot be diagnosed from it and a target of 0 has no "
        "state to receive them; the online nest edge "
        "(woof/core/microphysics_transition.PORTED_MP_PHYSICS) ports no "
        "such closure and this lane runs that closure rather than a second "
        f"one.  Same-scheme {mp} -> {mp} downscaling IS supported; the "
        "schemes a cross-scheme edge converts between are "
        + ", ".join(str(value) for value in sorted(PARENT_SCHEME_CONTRACT))
        + ".")


def _require_cross_scheme_contract(source_mp_physics: int,
                                   target_mp_physics: int) -> None:
    refusal = offline_cross_scheme_refusal(source_mp_physics, target_mp_physics)
    if refusal is not None:
        raise OfflineChildContractError(refusal)


class OfflineChildContractError(ValueError):
    """The archived parent cannot safely force the requested child."""


#: The owner file a downscale holds inside the output folder it claimed.
#: It names the process writing the folder; a second downscale aimed at
#: the same folder finds it and refuses instead of writing beside it.
OUTPUT_OWNER_NAME = ".gpuwm-output.owner"


def output_owner_path(path) -> Path:
    """Where the owner file of output folder ``path`` lives."""

    return Path(path) / OUTPUT_OWNER_NAME


def _output_entries(path: Path) -> list[str]:
    return sorted(child.name for child in path.iterdir()
                  if child.name != OUTPUT_OWNER_NAME
                  and not child.name.startswith(OUTPUT_OWNER_NAME + "."))


def reserve_output_root(path, *, flag: str = "--out") -> Path:
    """Claim one child-run output directory, in words when it cannot.

    A downscale never merges into a directory that already holds a run:
    the ``report.json`` it publishes has to describe ONE run, and the
    frame series beside it has to be that run's.  ``mkdir(exist_ok=
    False)`` enforces that, but its ``FileExistsError`` reached the
    reader as a traceback whose last line was a Windows error number --
    at exit 1, from a command that had already printed a refusal moments
    earlier.  The person re-running with the same output directory gets
    one sentence naming the directory, what it holds, and the two ways
    out instead.

    An EMPTY directory is adopted, not refused: it carries no frames to
    merge with and no receipt to overwrite, so refusing it prevents
    nothing.  ``flag`` is the flag the caller actually typed -- the two
    doors onto this route spell it ``--out`` and ``--outdir``.
    """

    from woof import ownership

    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        try:
            held = _output_entries(path)
        except OSError as probe_error:
            # Not the empty-directory case at all: the entry exists and
            # cannot be read as a directory (a plain file under that
            # name, a symlink with no destination).  This function's
            # whole subject is turning an OS-level exception into a
            # sentence, so the probe must not become the one that
            # escapes.
            detail = (getattr(probe_error, "strerror", None)
                      or str(probe_error))
            raise OfflineChildContractError(
                f"{flag} {path} exists but is not a directory this run can "
                f"claim: {detail}.  Pass a {flag} that names a directory "
                f"path, or remove {path} first.") from probe_error
        if held:
            raise OfflineChildContractError(
                f"{flag} {path} already holds a child run's output "
                f"({', '.join(held)[:120]}), and a downscale never writes "
                f"into a directory it did not create -- the report.json it "
                f"publishes has to describe one run, and the frames beside "
                f"it have to be that run's.  Pass a new {flag}, or remove "
                f"{path} first.") from error
    # ONE owner, whether this call created the folder or adopted an empty
    # one.  Two downscales started together used to both pass the checks
    # above (one created the folder, the other found it empty and adopted
    # it), then wrote the same outputs, and one's cleanup deleted the
    # other's child.toml.  The owner file is created exclusively, so
    # exactly one of them holds it; the other is told who does.
    try:
        claim = ownership.claim(output_owner_path(path),
                                purpose="downscale output")
    except ownership.OwnershipError as error:
        raise OfflineChildContractError(
            f"{flag} {path} is in use by "
            f"{ownership.describe_holder(error.holder)}, which is writing "
            f"a child run into it.  Wait for that run to finish, or pass "
            f"a new {flag}." + ownership.recovery_words(error)) from None
    except OSError as error:
        detail = getattr(error, "strerror", None) or str(error)
        raise OfflineChildContractError(
            f"{flag} {path} cannot be claimed for this run: {detail}.  "
            f"Pass a {flag} in a folder you can write to.") from None
    try:
        held = _output_entries(path)
    except OSError:
        held = []
    if held:
        # Written between the check above and the claim (or left by a
        # run whose owner has since died): still not ours to merge into.
        claim.release()
        raise OfflineChildContractError(
            f"{flag} {path} already holds a child run's output "
            f"({', '.join(held)[:120]}), and a downscale never writes "
            f"into a directory it did not create -- the report.json it "
            f"publishes has to describe one run, and the frames beside "
            f"it have to be that run's.  Pass a new {flag}, or remove "
            f"{path} first.")
    return path.resolve()


#: The one sentence that says why an offline downscale child's ROOT has to
#: take specified boundaries.  Spelled ONCE, here, and raised by
#: :func:`require_offline_child_root_forcing`, which every door onto this
#: route calls: plan review (``woof downscale``), admission
#: (``woof.offline_child_run.run``) and the state builder below.  It used
#: to be open-coded twice, in two different wordings, in two of those three
#: places and in NEITHER plan-review door, so the same configuration was
#: turned away with one sentence after a run had started and with another
#: after the whole parent archive had been interpolated.
OFFLINE_CHILD_ROOT_FORCING_REFUSAL = (
    "the offline route forces this domain from an external LBC mirror, so "
    "the run's root must take specified boundaries (specified = true, "
    "nested = false); specified = false or nested = true would leave the "
    "root with no Davies forcing and it would drift freely off the "
    "archive.  Sub-nests of this child are declared as domains in an "
    "experiment TOML, not by flipping nested on the root."
)


def require_offline_child_root_forcing(cfg):
    """Refuse a root the archived parent cannot force.  Returns ``cfg``.

    The rule is about the RUN'S ROOT, which is the only domain this route
    integrates: it is forced entirely from the archived parent, so it needs
    Davies relaxation on its own lateral boundaries.  A child tree is
    declared as domains in an experiment TOML; flipping ``nested`` on this
    config would only remove the forcing the route depends on.
    """

    if not cfg.specified or cfg.nested:
        raise OfflineChildContractError(OFFLINE_CHILD_ROOT_FORCING_REFUSAL)
    return cfg


#: The override sentences this process has already said.  Keyed by the
#: sentence itself, which carries the config path and both of the values
#: that disagreed, so a different file or a different disagreement is a
#: different key and still speaks.
_RESOLUTION_OVERRIDES_SAID: set = set()


def _warn_resolution_once(action: str, *, why: str) -> None:
    """Say one flag-over-file override once, however many doors reach it.

    Rule of this route: one resolution function, both doors call it, so
    plan review and admission cannot answer differently.  The cost is that
    on a real run the SAME sentence is reached twice in one process -- once
    while the plan is reviewed and once while the run is admitted -- and
    the published contract is that a disagreement earns ONE warning naming
    both values.  Printing it twice would make the record disagree with the
    documentation and read, to someone watching stderr, like two separate
    overrides.  So the sentence is said the first time it is reached and is
    silent after; the plan document already holds it (the observer that
    fills ``downscale-plan.json``'s ``warnings`` is attached around the
    whole command).

    Deduplicated on the normalized sentence, which is what
    :func:`woof.explain.warn` prints, so the key and the line cannot drift,
    and the set spans exactly one command because
    :func:`reset_resolution_notices` empties it as that command opens.
    """

    action = " ".join(str(action).split())
    if action in _RESOLUTION_OVERRIDES_SAID:
        return
    _RESOLUTION_OVERRIDES_SAID.add(action)
    warn(action, why=why)


def reset_resolution_notices() -> None:
    """Forget the override sentences already said, as a command opens.

    The published contract is one warning per INVOCATION however many
    doors resolve the same configuration, not one per process.  The set
    above is module state, so without this a second ``woof downscale``
    in one process, over the same file and the same flag, would override
    the same written statement in silence: the same defect this route was
    repaired for, moved one level up.  ``downscale_main`` calls this
    first, so the set covers exactly one command, plan review and the run
    admission it dispatches to share it, and the next command starts from
    silence.
    """

    _RESOLUTION_OVERRIDES_SAID.clear()


def resolve_child_streaming_options(child_config_path, flag_mode):
    """The ``[tiles]`` options the child actually integrates under.

    ONE resolution rule for the two doors onto this route, so plan review
    and the runner cannot answer differently about one configuration:

    * no flag -> whatever the child config declares;
    * the config has NO ``[tiles]`` table (the shared OFF object) -> the
      flag, written into the run and printed at plan review;
    * both, and they agree -> a no-op;
    * both, and they disagree -> the flag wins as the later and more
      specific statement, and one warning names both modes.  A file that
      spells ``[tiles]`` mode = "off" out loud has declared a mode, so
      ``--tiles on|auto`` beside it is a disagreement and earns that
      warning; only the absence of the table is silence.
    * they disagree AND the file pins knobs the flag's mode cannot carry
      -> those knobs leave with the mode that could carry them, and the
      SAME one warning names them and both ways of keeping them.

    ``[tiles]`` binds no restart identity (``woof.core.streaming
    .identity_payload_entry`` returns nothing for it), so a mode
    disagreement is not a breakage to refuse over; it is a choice to
    resolve and say out loud.  The sibling front door
    (``woof.prepared_single_domain_forecast``) resolves the same
    disagreement in the same direction but NOT by the same test: it asks
    ``declared is not None and declared.enabled and declared != tiles``,
    so an experiment that spells ``[tiles] mode = "off"`` out loud is
    still replaced there without a word.  The identity test below is what
    keeps silence and an explicit off apart; that door needs the same one.
    """

    from woof.config import load_streaming_options
    from woof.core.streaming import OFF, StreamingOptions

    supplied = load_streaming_options(child_config_path)
    if flag_mode is None:
        return supplied
    mode = str(flag_mode)
    # IDENTITY, not equality.  ``load_streaming_options`` returns the
    # shared OFF object for a file with NO ``[tiles]`` table at all, and
    # builds a fresh object for a file that HAS one -- including a file
    # that spells ``[tiles]`` mode = "off" out loud, which is a legal,
    # documented spelling and compares EQUAL to OFF.  Under ``==`` an
    # explicit mode = "off" read as "the config declares nothing" and
    # ``--tiles on|auto`` replaced it in silence: no warning on stderr and
    # no trace in the plan document that the user's own statement had been
    # overridden.  ``is`` keeps the two apart, so silence means silence and
    # an explicit off is a declaration the override has to name.
    if supplied is OFF:
        return StreamingOptions.from_mapping({"mode": mode}, source="--tiles")
    if supplied.mode == mode:
        return supplied
    # THE WHOLE TABLE, REBUILT UNDER THE NEW MODE AND VALIDATED BY THE
    # CLASS THAT OWNS THE RULES.  This used to be ``replace(supplied,
    # mode=mode)``, a field swap whose only reader of the result was
    # ``StreamingOptions.__post_init__`` -- and that reader's sentences are
    # addressed to whoever WROTE the block.  A child config that legally
    # pins a tiling under its own declared mode ("on" with tile_nx,
    # tile_ny, or nbuffers) therefore became unconstructible the moment
    # ``--tiles auto`` changed the mode: the command announced that the
    # flag had won and then, on the next line, exited on "[tiles] sets
    # tile_nx, tile_ny while mode = 'auto' ... say which you meant: mode =
    # 'on' to pin the tiling", which is exactly what the file already said.
    # The flag imposed that mode, so the file was not the statement that
    # could be edited to clear it, and there was no way out of the
    # invocation at all.  Resolved here instead, on the merged mapping:
    # the flag still wins, the knobs its mode cannot carry leave WITH the
    # mode that could carry them, and the one warning a disagreement earns
    # says which they were and how to keep them.
    merged = supplied.to_mapping()
    merged["mode"] = mode
    source = f"--tiles {mode} over {child_config_path}"
    dropped: tuple[str, ...] = ()
    try:
        resolved = StreamingOptions.from_mapping(merged, source=source)
    except ValueError:
        # WHICH keys a mode cannot carry stays the class's ruling and is
        # not copied here: these four are the only keys any mode rejects,
        # and one is cleared only because the class has just refused the
        # table that carried it under this mode.
        dropped = tuple(
            name for name in ("tile_nx", "tile_ny", "nbuffers", "halo")
            if merged.get(name) is not None)
        for name in dropped:
            merged[name] = None
        try:
            resolved = StreamingOptions.from_mapping(merged, source=source)
        except ValueError as error:
            # Not reachable from the two flag values argparse admits, and
            # kept so that a widened flag can never land back on a bare
            # ValueError with no way out printed beside it.
            raise OfflineChildContractError(
                f"--tiles {mode} cannot be resolved against the [tiles] "
                f"block the child config {child_config_path} declares for "
                f"mode = '{supplied.mode}': {error}  Drop --tiles to run "
                "the mode and the block the file declares, or edit "
                f"[tiles] in that file to one mode = '{mode}' accepts."
            ) from error
    said = (
        f"--tiles mode = '{mode}' replaces the [tiles] mode = "
        f"'{supplied.mode}' the child config {child_config_path} declares; "
        "the flag is the later and more specific statement, and [tiles] "
        "binds no identity either way")
    why = ("Streaming is a promise that a domain integrated as one "
           "resident block and the same domain streamed from host RAM "
           "produce the same bytes, so the mode changes how the child "
           "runs and nothing it computes.  The rest of the block (tile "
           "size, buffers, store, write mode) is kept exactly as the "
           "file wrote it.")
    if dropped:
        keys = ", ".join(dropped)
        said = (
            f"--tiles mode = '{mode}' replaces the [tiles] mode = "
            f"'{supplied.mode}' the child config {child_config_path} "
            f"declares, and with it the {keys} that block pins for mode = "
            f"'{supplied.mode}', which mode = '{mode}' cannot carry and "
            "would ignore in silence; the flag is the later and more "
            "specific statement, and [tiles] binds no identity either "
            "way.  Drop --tiles to keep the pinned tiling, or delete "
            f"{keys} from that file to let mode = '{mode}' plan one.")
        why = ("Streaming is a promise that a domain integrated as one "
               "resident block and the same domain streamed from host RAM "
               "produce the same bytes, so the mode changes how the child "
               f"runs and nothing it computes.  {keys} belong to mode = "
               "'on', which pins a tiling; 'auto' plans its own and its "
               "answer IS the planner's, so a pinned tile makes it stream "
               "a domain that fits and a pinned nbuffers reads back as a "
               "count nobody chose.  The rest of the block (store, write "
               "mode, budgets) is kept exactly as the file wrote it.")
    _warn_resolution_once(said, why=why)
    return resolved


def resolve_child_run_config(child_config_path, *, child_levels=None):
    """The child ``RunConfig`` the run is actually built on.

    ONE resolution rule for the two doors, as above.  ``child_levels`` is
    the ``--child-levels N[,STRETCH]`` spec: absent, the file decides;
    present, the ladder it names replaces ``eta_levels``/``nz`` and
    ``validate_run_config`` is re-run so the config authority's own length
    and monotonicity sentences are the ones that speak.  A file that
    already named a DIFFERENT ladder is warned about, not refused: the
    flag is the later and more specific statement, and the warning says
    which ladder won.

    ``p_top``, ``hybrid_opt`` and ``etac`` are left inherited, which is
    what keeps ``woof.vertical_remap.require_shared_column_basis``
    satisfied by construction.
    """

    from dataclasses import replace

    from woof.config import load_config, validate_run_config

    from types import SimpleNamespace
    from woof.config import load_device_options
    from woof.core.devices import refuse_unrouted_devices
    refuse_unrouted_devices(
        SimpleNamespace(devices=load_device_options(child_config_path)),
        "offline child and downscale")
    # "auto" is admitted: this route carries the label to the floor
    # (adapt_child_acoustics reads it with child_epssm_is_auto).
    cfg = load_config(child_config_path, accept_epssm_auto=True,
                      child_static=True)
    if child_levels is None:
        return cfg
    # Imported here and not at module scope: ``woof.downscale`` imports
    # this module, and the ladder generator plus its spec parser live
    # there beside the flag that spells them.
    from woof.downscale import _parse_child_levels

    # Never None here: _parse_child_levels answers None only for a None
    # spec, which the guard above already returned on, and every other
    # unreadable spec is a refusal it raises itself.
    ladder = _parse_child_levels(child_levels)
    if cfg.eta_levels is not None:
        declared = tuple(float(value) for value in cfg.eta_levels)
        if declared != tuple(ladder):
            _warn_resolution_once(
                f"--child-levels {child_levels} replaces the "
                f"{len(declared) - 1}-level eta_levels ladder the child "
                f"config {child_config_path} declares with a "
                f"{len(ladder) - 1}-level one; the flag is the later and "
                "more specific statement, and the run and its restarts "
                "record the ladder that won",
                why="eta_levels binds the restart identity, so the "
                    "ladder written into the run is the ladder the "
                    "checkpoints are bound to.  p_top, hybrid_opt and "
                    "etac stay inherited from the archived parent, which "
                    "is what gives the two ladders coincident endpoints.")
    cfg = replace(cfg, eta_levels=tuple(ladder), nz=len(ladder) - 1)
    try:
        validate_run_config(cfg)
    except ValueError as error:
        # The config authority's own sentence, raised as this route's
        # refusal so one reader answers for a hand-written ladder and a
        # flag-built one alike.
        raise OfflineChildContractError(str(error)) from error
    return cfg


def child_epssm_is_auto(child_config_path) -> bool:
    """Whether the child config leaves ``epssm`` to the model.

    True where no ``[grid]``/``[dynamics]``/``[run]`` table names it or
    one names ``"auto"`` or ``{ auto = VALUE }``, which retains a derived
    parent's numeric setting and its model-choice label. The same rule
    the experiment loader applies to a
    ``[[domain]]`` (``ExperimentConfig.auto_epssm``), read off the same
    bytes :func:`resolve_child_run_config` loads.
    """

    import io
    import tomllib

    from woof.config import _RUN_CONFIG_TABLES, EPSSM_AUTO
    from woof.config_authority import read_config_authority

    raw = tomllib.load(io.BytesIO(
        read_config_authority(child_config_path).payload))
    for table in _RUN_CONFIG_TABLES:
        entries = raw.get(table, {})
        if isinstance(entries, dict) and "epssm" in entries:
            choice = entries["epssm"]
            return choice == EPSSM_AUTO or (
                isinstance(choice, dict) and set(choice) == {EPSSM_AUTO})
    return True


def child_terrain_reading(frame_path, placement: "OfflineChildPlacement", *,
                          child_cfg):
    """The steepest slope of the terrain an offline child integrates.

    The child runs its parent's ``HGT`` carried onto its own grid by the
    SINT its initial state is built with (``terrain_policy``
    ``sint-parent-inherited``, :func:`interpolate_parent_initial_state`),
    on the map factors the same SINT carries down, so that is the ground
    read: the prepared doors read each domain's static ``HGT_M`` the same
    way (:func:`woof.acoustic_adaptation.readings_from_static`).  Three
    two-dimensional fields of one frame, on the host, so the answer is
    known before the archive is interpolated.
    """

    from woof.acoustic_adaptation import steepest_slope

    with _ParentHistory(frame_path) as dataset:
        fields = {name: _read_record(dataset, name)
                  for name in ("HGT", "MAPFAC_U", "MAPFAC_V")}
    child = {
        name: sint(np.ascontiguousarray(value, dtype=np.float32),
                   placement.registration(
                       _INITIAL_FIELD_STAGGER.get(name, ""),
                       wrapper="interp"))
        for name, value in fields.items()}
    return steepest_slope(
        child["HGT"], float(child_cfg.dx), float(child_cfg.dy),
        msfu=child["MAPFAC_U"], msfv=child["MAPFAC_V"],
        label=f"d{int(child_cfg.grid_id):02d}")


def adapt_child_acoustics(cfg, *, child_config_path, frame_path,
                          placement: "OfflineChildPlacement"):
    """The child config with the acoustic rule its own ground needs.

    ``(cfg, adaptation)``: the off-centering floor and the substep count
    (:func:`woof.acoustic_adaptation.adapt_run_to_terrain`), read off
    the terrain the child integrates (:func:`child_terrain_reading`), the
    rule every prepared door applies to each of its domains.  An
    ``epssm`` the child config leaves to the model (unset or ``"auto"``,
    :func:`child_epssm_is_auto`) is raised to the floor; one it writes is
    the user's and is refused below the floor, remedy first.  Plan review
    and the runner both call this, so each line is said once per command
    (:func:`_warn_resolution_once`).
    """

    from woof.acoustic_adaptation import adapt_run_to_terrain

    reading = child_terrain_reading(frame_path, placement, child_cfg=cfg)
    try:
        return adapt_run_to_terrain(
            cfg, reading, grid_id=int(cfg.grid_id),
            auto_epssm=child_epssm_is_auto(child_config_path),
            say=lambda sentence, why: _warn_resolution_once(
                sentence, why=why))
    except ValueError as error:
        raise OfflineChildContractError(str(error)) from error


#: What a downscaled child's own ``report.json`` calls the pipeline that
#: wrote it, on BOTH of its outcomes -- the run that reached its last
#: frame and the one whose fields stopped being finite.  One spelling,
#: used by the writer (:mod:`woof.offline_child_run`) and by the reader
#: that decides whether a run directory holds a child at all
#: (:func:`woof.resume.offline_child_run_at`), so the document can say
#: which route made it without either side keeping its own copy of the
#: string.
CHILD_REPORT_PIPELINE = "archived-parent-to-native-standalone-cuda-child"


#: The horizontal spacing, in metres, at or below which this tree calls a
#: child an LES rather than a very fine mesoscale run.  It is the tree's
#: own number and not a new one: ``docs/public/LES.md`` ships its nested
#: LES child at 250 m and calls that child "COARSE LES at the gray-zone
#: edge", and ``docs/public/GRAYZONE-NEST.md`` puts the gray zone between
#: roughly 2 km and that child.  At or below this spacing the grid
#: resolves the eddies a boundary-layer scheme exists to stand in for.
LES_CHILD_SPACING_M = 250.0

#: Where the numbers above are written down, quoted in the statement so a
#: reader can check the threshold rather than take it.
LES_CHILD_SPACING_SOURCE = "docs/public/LES.md"

#: The closures that mix in three dimensions, WRF's own ``km_opt``
#: spelling: 2 is the 1.5-order prognostic TKE scheme and 3 is 3-D
#: Smagorinsky.  1 and 4 are the two-dimensional operators, which under a
#: boundary-layer scheme leave the vertical entirely to that scheme.
#:
#: THIS TREE admits either three-dimensional one with
#: ``bl_pbl_physics = 0`` and no other way (``woof.config``'s
#: ``validate_km_opt``): the vertical exchange pair of both is applied by
#: ``vertical_diffusion_2``, which is PBL-off gated, so with a scheme on
#: only the horizontal half of the selected closure would run.  WRF's
#: ``diff_opt = 2`` selects that metric-aware vertical exchange path;
#: ``diff_opt = 1`` selects model-coordinate horizontal diffusion.
LES_CHILD_THREE_DIMENSIONAL_CLOSURES = (2, 3)

#: The two-dimensional operators, which compute no vertical exchange pair
#: of their own.  Under a boundary-layer scheme they leave the vertical to
#: that scheme, which is a division of labour; with the scheme OFF they
#: leave nothing doing it, so a child carrying one of these and
#: ``bl_pbl_physics = 0`` mixes heat and moisture vertically by no route
#: at all.  ``km_opt = 0`` is deliberately absent: this tree admits it
#: only behind an acknowledgement the user writes out in full
#: (``woof.config``'s ``KM_OPT_ZERO_ACK``), so that child was told.
LES_CHILD_NO_VERTICAL_MIXING_CLOSURES = (1, 4)


def child_inherits_parent_levels(cfg, *, child_levels_spec,
                                parent_levels) -> bool:
    """Did anyone CHOOSE this child's vertical ladder, or is it the parent's?

    Three readings, in the order they settle the question, and the same
    three at the door and inside the runner so the two cannot disagree:

    * ``--child-levels`` was given, so the ladder was chosen -- whatever
      it came out as, someone asked for it;
    * the configuration declares no ``eta_levels`` at all, so the child
      is built on the parent's ladder (``docs/public/DOWNSCALE.md``:
      "without ``--child-levels`` the child keeps the parent's levels");
    * it declares one, so it is the parent's only if it is the same DEPTH
      as the parent tape's, which is the reading the parent archive can
      actually answer.  This is the arm a derived config lands on:
      ``_derive_child_run_config`` copies ``eta_levels`` from the parent
      verbatim along with everything else.
    """

    if child_levels_spec is not None:
        return False
    if getattr(cfg, "eta_levels", None) is None:
        return True
    return parent_levels is not None and int(parent_levels) == int(cfg.nz)


def les_child_regime(cfg, *, inherits_parent_levels: bool,
                     parent_levels: int | None = None) -> dict | None:
    """The LES-regime statement a sub-250 m child earns, or ``None``.

    ONE rule, read by the two places that need it: the downscale door
    says it before the run starts, and the non-finite refusal says it
    again when a run of that shape ends the way this shape ends.  A
    second copy of the arithmetic is how the door and the refusal come to
    disagree about what regime a child was in.

    It is a STATEMENT and not a refusal.  A child at this spacing on an
    inherited ladder is a run somebody may very well want -- the shipped
    nested LES child is exactly one -- and nothing here changes what the
    run does.  What it changes is that the reader is told which regime
    they asked for before they wait for it, instead of afterwards.

    ``inherits_parent_levels`` is the caller's answer to "did anyone
    choose this child's vertical ladder?", because only the caller knows:
    the door knows whether ``--child-levels`` was given and whether the
    resolved ladder is still the parent's, and the runner knows what the
    door resolved.  ``parent_levels`` is the parent archive's own level
    count when the caller has it, and only sharpens the sentence.

    Returns ``None`` when the child is coarser than
    :data:`LES_CHILD_SPACING_M`, and otherwise when ALL THREE of these
    are true of it -- there is nothing to say to a child that was
    configured for the regime it is running in:

    * it carries a vertical ladder of its own, so nobody handed it a
      parent's;
    * it does not run a 1-D boundary-layer scheme without a 3-D closure
      beside it (:data:`LES_CHILD_THREE_DIMENSIONAL_CLOSURES`); and
    * it is not left with no vertical mixing of heat or moisture at all,
      which is the scheme off and a two-dimensional closure
      (:data:`LES_CHILD_NO_VERTICAL_MIXING_CLOSURES`).

    The third condition is the one this function's own ``why`` text has
    always named and the rule once left out: an 83 m child on its own
    ladder with ``bl_pbl_physics = 0`` and ``km_opt = 4`` heard nothing,
    although by that text it has no vertical mixing by any route.
    """

    spacing = min(float(cfg.dx), float(cfg.dy))
    if spacing > LES_CHILD_SPACING_M:
        return None
    km_opt = int(getattr(cfg, "km_opt", 0) or 0)
    pbl = int(getattr(cfg, "bl_pbl_physics", 0) or 0)
    three_dimensional = km_opt in LES_CHILD_THREE_DIMENSIONAL_CLOSURES
    pbl_without_closure = bool(pbl) and not three_dimensional
    no_vertical_mixing = (not pbl
                          and km_opt in LES_CHILD_NO_VERTICAL_MIXING_CLOSURES)
    if not (inherits_parent_levels or pbl_without_closure
            or no_vertical_mixing):
        return None
    reasons = []
    if inherits_parent_levels:
        reasons.append(
            f"inherits the parent's {int(cfg.nz)}-level vertical ladder"
            if parent_levels is None or int(parent_levels) == int(cfg.nz)
            else f"runs {int(cfg.nz)} levels carried down from the "
                 f"parent's {int(parent_levels)}")
    if pbl_without_closure:
        reasons.append(
            f"runs a 1-D boundary-layer scheme (bl_pbl_physics = {pbl}) "
            f"with no 3-D closure (km_opt = {km_opt})")
    if no_vertical_mixing:
        reasons.append(
            f"mixes heat and moisture vertically by no route at all "
            f"(km_opt = {km_opt} computes no vertical exchange pair and "
            f"bl_pbl_physics = {pbl})")
    ways_out = []
    if inherits_parent_levels:
        ways_out.append(
            "--child-levels N,STRETCH gives the child its own vertical "
            "ladder instead of the parent's")
    if pbl_without_closure or no_vertical_mixing:
        ways_out.append(
            "km_opt = 3 (3-D Smagorinsky) or km_opt = 2 (prognostic TKE) "
            "with bl_pbl_physics = 0 in the --child-config TOML selects a "
            "3-D closure, which this tree admits with the boundary-layer "
            "scheme off and no other way")
    ways_out.append(
        "--child-surface-from gives the child its own geography, which a "
        "grid this fine can resolve and the parent's cannot")
    # THE SHAPE the reader is being warned about, and only the one that
    # is true of this child: the walked failure belongs to a child with a
    # boundary-layer scheme running at LES spacing, and putting that
    # sentence on a child that has no such scheme would be a warning
    # about a mechanism that is not there.
    if pbl_without_closure:
        shape = ("A boundary-layer-scheme child at LES spacing tends to "
                 "grow vertical velocity check after check until the "
                 "field goes non-finite.")
    elif no_vertical_mixing:
        shape = ("Nothing carries heat or moisture between this child's "
                 "levels except the motion it resolves, so a column's "
                 "stratification is held by that motion alone.")
    else:
        shape = (f"A child at this spacing on a ladder chosen for a "
                 f"coarser grid leaves more of its turbulence to the "
                 f"subgrid model than a resolved column does, 12.7 "
                 f"percent against 7.9 ({LES_CHILD_SPACING_SOURCE}).")
    statement = (
        f"this child runs at {spacing:g} m spacing, at or below the "
        f"{LES_CHILD_SPACING_M:g} m this tree calls coarse LES at the "
        f"gray-zone edge ({LES_CHILD_SPACING_SOURCE}), and it "
        + " and ".join(reasons)
        + f".  {shape}  Ways out: " + "; ".join(ways_out))
    why = (
        "At this spacing the grid resolves the eddies a 1-D "
        "boundary-layer scheme exists to stand in for, so the scheme's "
        "vertical transport and the resolved motion do the same work "
        "twice; and km_opt 1 and 4 compute no separate vertical exchange "
        "pair at all, so with the scheme off there is no vertical mixing "
        "of heat or moisture by any route "
        f"({LES_CHILD_SPACING_SOURCE}, the two closures and the "
        "vertical-scalar-mixing table).  The vertical ladder is the "
        "binding constraint on the shipped nested child for the same "
        "reason it is here: 250 m columns on a 3 km grandparent's 49 "
        "shared levels, measured at 18 levels inside a 1741 m boundary "
        "layer.")
    return {
        "spacing_m": spacing,
        "threshold_m": LES_CHILD_SPACING_M,
        "nz": int(cfg.nz),
        "parent_nz": None if parent_levels is None else int(parent_levels),
        "km_opt": km_opt,
        "bl_pbl_physics": pbl,
        "inherits_parent_levels": bool(inherits_parent_levels),
        "pbl_without_three_dimensional_closure": pbl_without_closure,
        "no_vertical_mixing_of_heat_or_moisture": no_vertical_mixing,
        "statement": statement,
        "why": why,
        "source": LES_CHILD_SPACING_SOURCE,
    }


def _unsupported_parent_clause(mp_physics: int, *, what: str) -> str:
    """The refusal text for a parent scheme this lane cannot read.

    Names the scheme, the breakage that keeps it out and the way out, from
    the registry's own ``consumers.offline_child`` row -- so the message a
    user reads is the row an editor would change, and a bare
    "unsupported ... mp_physics=N" can never come back (audit R-017).
    """

    admitted = ", ".join(str(value) for value in sorted(
        OFFLINE_CHILD_MP_PHYSICS))
    reason = offline_child_refusal(int(mp_physics))
    if reason is None:
        reason = ("no reason is recorded for it in the physics registry's "
                  "consumers.offline_child row")
    return (
        f"the offline downscale lane cannot read a {what} parent of "
        f"mp_physics={mp_physics}: {reason}. Parent schemes this lane "
        f"carries same-scheme: {admitted}. The admission is one row in "
        "tools/build_registry.py "
        "(components.microphysics.options.<option>.consumers.offline_child), "
        "not a literal here; run the parent's own scheme as the child, or "
        "prepare the child from a parent archive written by an admitted "
        "scheme.")


@dataclass(frozen=True)
class ParentPhysicsBinding:
    """Authoritative parent-scheme identity from a companion setup record."""

    mp_physics: int
    morr_rimed_ice: int | None
    domain_id: int
    evidence_kind: str
    evidence_path: Path
    evidence_sha256: str
    #: WSM6's ``hail_opt`` (mp=6, RunConfig ``wsm6_hail_opt``) or WDM6's
    #: (mp=16, ``wdm6_hail_opt``): whether the parent's single rimed
    #: category means graupel (0) or hail (1).  A cross-scheme conversion
    #: maps that category by its MEANING (``microphysics_transition.
    #: _rimed_category``), so the switch is parent evidence exactly as
    #: ``morr_rimed_ice`` is for Morrison.  ``None`` for every other
    #: scheme, and for a companion that does not record it, where the
    #: contract takes WRF's default of graupel.
    hail_opt: int | None = None

    def __post_init__(self) -> None:
        if int(self.mp_physics) not in OFFLINE_CHILD_MP_PHYSICS:
            raise OfflineChildContractError(
                _unsupported_parent_clause(self.mp_physics, what="bound"))
        if int(self.domain_id) < 1:
            raise OfflineChildContractError("bound parent domain_id must be >= 1")
        if int(self.mp_physics) in (10, 40) and self.morr_rimed_ice not in {0, 1}:
            raise OfflineChildContractError(
                "bound Morrison parent requires morr_rimed_ice=0/1")
        if (int(self.mp_physics) not in (10, 40)
                and self.morr_rimed_ice is not None):
            raise OfflineChildContractError(
                "morr_rimed_ice evidence is only valid for mp_physics=10 or 40")
        if self.hail_opt is not None:
            if int(self.mp_physics) not in {6, 16}:
                raise OfflineChildContractError(
                    "hail_opt evidence is only valid for mp_physics=6 or 16")
            if int(self.hail_opt) not in {0, 1}:
                raise OfflineChildContractError(
                    "bound WSM6/WDM6 parent hail_opt must be 0 or 1")
        if self.evidence_kind not in {"gpuwm-restart", "wrf-namelist"}:
            raise OfflineChildContractError(
                f"unsupported parent physics evidence {self.evidence_kind!r}")
        if not Path(self.evidence_path).is_file():
            raise OfflineChildContractError(
                f"parent physics evidence does not exist: {self.evidence_path}")
        digest = str(self.evidence_sha256).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise OfflineChildContractError(
                "parent physics evidence_sha256 must be one SHA-256 hex digest")

    def receipt(self) -> Mapping[str, object]:
        return MappingProxyType({
            "mp_physics": int(self.mp_physics),
            "morr_rimed_ice": self.morr_rimed_ice,
            "domain_id": int(self.domain_id),
            "evidence_kind": self.evidence_kind,
            "evidence_path": str(Path(self.evidence_path).resolve()),
            "evidence_sha256": str(self.evidence_sha256).lower(),
            "hail_opt": (None if self.hail_opt is None else int(self.hail_opt)),
        })


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_parent_physics_from_gpuwm_restart(
        path: str | Path) -> ParentPhysicsBinding:
    """Bind parent MP identity to a validated woof restart header."""

    from woof.io.restart import read_restart_header
    path = Path(path).resolve()
    header = read_restart_header(path)
    config = header.get("config")
    setup = header.get("physics_setup")
    fingerprint = header.get("physics_setup_fingerprint")
    if not isinstance(config, dict) or not isinstance(setup, dict):
        raise OfflineChildContractError(
            f"{path} lacks complete woof config/physics setup evidence")
    if fingerprint != _canonical_sha256(setup):
        raise OfflineChildContractError(
            f"{path} physics setup fingerprint is invalid")
    microphysics = setup.get("microphysics")
    if not isinstance(microphysics, dict):
        raise OfflineChildContractError(
            f"{path} lacks resolved microphysics setup evidence")
    mp_physics = int(config.get("mp_physics", -1))
    if int(microphysics.get("scheme_id", -2)) != mp_physics:
        raise OfflineChildContractError(
            f"{path} config and resolved microphysics identities disagree")
    if mp_physics == 40 and int(config.get("aercu_opt", 0)) != 0:
        raise OfflineChildContractError(
            "MP40 offline parent requires aercu_opt=0; positive coupling "
            "requires the CU11/aerosol coupling identity absent from the archived-parent binding; use an online coupled nest")
    domain_id = int(header.get("grid_id", config.get("grid_id", 0)))
    morr = None
    if mp_physics in (10, 40):
        morr = int(config.get("morr_rimed_ice", -1))
        resolved = microphysics.get("morrison_rimed_ice" if mp_physics == 10
                                    else "morrison_aero")
        selection = "selection" if mp_physics == 10 else "morr_rimed_ice"
        if (not isinstance(resolved, dict)
                or int(resolved.get(selection, -2)) != morr):
            raise OfflineChildContractError(
                f"{path} Morrison rimed-ice config/setup identities disagree")
    evidence = {
        "format_version": header.get("format_version"),
        "grid_id": domain_id,
        "config": config,
        "physics_setup": setup,
        "physics_setup_fingerprint": fingerprint,
    }
    hail_opt = None
    if mp_physics in (6, 16):
        raw_hail = config.get("wsm6_hail_opt" if mp_physics == 6
                              else "wdm6_hail_opt")
        hail_opt = None if raw_hail is None else int(raw_hail)
    return ParentPhysicsBinding(
        mp_physics=mp_physics, morr_rimed_ice=morr, domain_id=domain_id,
        evidence_kind="gpuwm-restart", evidence_path=path,
        evidence_sha256=_canonical_sha256(evidence), hail_opt=hail_opt)


def bind_parent_physics_from_wrf_namelist(
        path: str | Path, *, domain_id: int = 1) -> ParentPhysicsBinding:
    """Bind one WRF domain's MP identity to exact namelist bytes."""

    from woof.namelist_import import parse_namelist
    path = Path(path).resolve()
    domain_id = int(domain_id)
    if domain_id < 1:
        raise OfflineChildContractError("WRF binding domain_id must be >= 1")
    parsed = parse_namelist(path)
    physics = parsed.get("physics", {})

    def domain_value(name: str, *, required: bool, default=None):
        values = physics.get(name)
        if not values:
            if required:
                raise OfflineChildContractError(
                    f"{path} &physics lacks authoritative {name}")
            return default
        return values[min(domain_id - 1, len(values) - 1)]

    mp_physics = int(domain_value("mp_physics", required=True))
    if mp_physics == 40 and int(domain_value("aercu_opt", required=False, default=0)) != 0:
        raise OfflineChildContractError(
            "MP40 offline parent requires aercu_opt=0; positive coupling "
            "requires the CU11/aerosol coupling identity absent from the archived-parent binding; use an online coupled nest")
    morr = None
    if mp_physics in (10, 40):
        morr = int(domain_value(
            "morr_rimed_ice", required=False, default=1))
    hail_opt = None
    if mp_physics in (6, 16):
        raw_hail = domain_value("hail_opt", required=False, default=None)
        hail_opt = None if raw_hail is None else int(raw_hail)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ParentPhysicsBinding(
        mp_physics=mp_physics, morr_rimed_ice=morr, domain_id=domain_id,
        evidence_kind="wrf-namelist", evidence_path=path,
        evidence_sha256=digest, hail_opt=hail_opt)


def _decode_time(variable) -> datetime:
    raw = np.asarray(variable[:])
    if raw.shape[0] != 1:
        raise OfflineChildContractError(
            f"one history file must contain exactly one Time record, got {raw.shape}")
    row = raw[0]
    if row.dtype.kind == "S":
        value = b"".join(row.tolist()).decode("ascii")
    else:
        value = "".join(str(item) for item in row.tolist())
    try:
        return datetime.strptime(value, _TIME_FORMAT)
    except ValueError as exc:
        raise OfflineChildContractError(
            f"invalid WRF Times value {value!r}") from exc


def _hash_array(digest, label: str, value) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(label.encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _infer_mp_physics(
        inventory: frozenset[str], *, source_kind: str) -> int | None:
    """Advisory scheme id from the WRF names one history frame carries.

    The ladder is ordered MOST DISCRIMINATING FIRST, because several
    packages' inventories are strict supersets of others'.  Every arm
    below is a name declared by exactly one WRF package, or a pair whose
    presence-and-absence only one package satisfies; an arm that merely
    matched a subset reported the wrong scheme with no way for a reader to
    tell, which is what this function was doing for mp 9, 16 and 18 -- all
    three were reported as somebody else's scheme in receipts and in the
    cross-frame agreement check (audit R-018).

    ``source_kind`` is the producer, ``"wrf"`` or ``"woof"``, because the
    warm-rain packages are separated by the WRITER and not by the scheme:
    stock WRF's passiveqv (mp=0) transports qv alone, but woof's own mp=0
    allocates and advects the warm-rain pair beside it
    (:func:`_transported_source_fields`) and woof/io/wrfout.py writes all
    three whenever the state is moist.  A woof frame carrying exactly
    QVAPOR/QCLOUD/QRAIN is therefore mp=0 OR mp=1 with nothing in the
    inventory to separate them, and this function says so by returning
    ``None`` rather than naming the one that happens to be second.
    """
    if source_kind not in ("wrf", "woof"):
        raise ValueError(
            f"parent history producer must be 'wrf' or 'woof', got "
            f"{source_kind!r}")
    # NSSL is first: its volume moments are declared by no other package
    # (Registry.EM_COMMON:3033), so QVGRAUPEL/QVHAIL identify it outright.
    # Without this arm an NSSL stream matched the Morrison arm below on
    # {QNSNOW, QNGRAUPEL} -- and NSSL is an ADMITTED offline parent, so
    # the mislabel reached a receipt for a supported configuration.
    # NTU shares all six number names with MY2; its aerosol/shape moments
    # identify it before the less specific mass-and-number inventories.
    if {"QDCN", "QTCN", "QCCN", "QRCN", "QNIN", "I3M"} <= inventory:
        return 56
    if {"QVGRAUPEL", "QVHAIL"} <= inventory:
        return 18
    # Milbrandt-Yau (Registry.EM_COMMON:3025) is the other package that
    # declares QHAIL and QNHAIL; NSSL is already claimed above, so the
    # pair without the volume moments is MY2's discriminant.  It must
    # precede the Morrison arm for the same superset reason: MY2 carries
    # QNSNOW and QNGRAUPEL too.
    if {"QHAIL", "QNHAIL"} <= inventory:
        return 9
    # mp=28 before mp=8 because its inventory is a strict superset of
    # mp=8's: it carries QNRAIN/QNICE too, so the classic-Thompson arm
    # below would claim an aerosol-aware stream.
    if {"QNWFA", "QNIFA"} <= inventory:
        return 28
    # mp=50 next, and before mp=8, for the same superset reason: a P3
    # stream carries QNRAIN/QNICE beside its rime pair, so the classic-
    # Thompson arm below would claim it.  QIR/QIB are declared by no other
    # scheme (Registry.EM_COMMON:555/:557), which makes the pair the
    # discriminant.
    if {"QIR", "QIB"} <= inventory:
        return 50
    # WDM6 (Registry.EM_COMMON:3031, ``scalar:qnn,qnc,qnr``) before the
    # single-moment WSM6 arm it would otherwise fall into: it carries the
    # same six masses as WSM6 and adds a warm-rain number pair plus the
    # CCN reservoir, and it declares NO ice number, which is what
    # separates it from Morrison and Thompson.  QNCCN alone is not the
    # discriminant -- NSSL publishes ``qnn`` under the same name -- but
    # NSSL is claimed two arms above.
    if ({"QNCCN", "QNCLOUD", "QNRAIN"} <= inventory
            and "QNICE" not in inventory):
        return 16
    if {"QNSNOW", "QNGRAUPEL"} <= inventory:
        return 10
    if {"QNRAIN", "QNICE"} <= inventory:
        return 8
    if {"QICE", "QSNOW", "QGRAUP"} <= inventory:
        return 6
    # Kessler (Registry.EM_COMMON:3015) declares moist:qv,qc,qr and no
    # frozen species; passiveqv (:3014) declares qv alone.  Both were
    # reported as "unknown" -- a receipt row that reads as "this stream is
    # unreadable" for two packages this lane now carries.  Both arms are
    # ABSENCE tests as well as presence tests, because every remaining
    # package is a superset of Kessler's and Kessler's is a superset of
    # passiveqv's: a frame carrying any frozen species or any number
    # moment is not one of these two, and stays unknown rather than being
    # labelled with the smallest package that fits.
    _frozen = {"QICE", "QSNOW", "QGRAUP", "QHAIL"}
    _moments = {"QNCLOUD", "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL",
                "QNHAIL", "QNCCN", "QNDROP", "QNWFA", "QNIFA",
                "QVGRAUPEL", "QVHAIL", "QIR", "QIB"}
    if not (inventory & (_frozen | _moments)):
        if {"QVAPOR", "QCLOUD", "QRAIN"} <= inventory:
            # Ambiguous on a woof tape and only there: see the docstring.
            # Claiming Kessler would be the same class of mislabel this
            # ladder was rewritten to end, one package later.
            return None if source_kind == "woof" else 1
        if inventory & {"QVAPOR"} and not (inventory & {"QCLOUD", "QRAIN"}):
            return 0
    return None


def _unreadable_history(path: Path, error: BaseException) -> OfflineChildContractError:
    """A parent history file that could not be opened or decoded, in words.

    A truncated or corrupt wrfout used to end the command in the NetCDF
    library's own traceback ("NetCDF: HDF error"), naming neither the
    remedy nor, on some routes, the file.  Only the FILE's failures come
    here (see :class:`_ParentHistory`), so the remedy is always the file's.
    """

    if isinstance(error, FileNotFoundError) and not path.exists():
        return OfflineChildContractError(
            f"{path} is not there any more: the parent history series "
            "names it, but the file is gone.  Restore it from the parent "
            "run, or run the parent again to regenerate it, then retry")
    if isinstance(error, PermissionError):
        return OfflineChildContractError(
            f"{path} cannot be read by this account (permission denied).  "
            "Give this account read access to the parent's history files, then retry")
    # The reader's own words about the file, not the command that ran
    # (which would name the file a second time).
    text = str(getattr(error, "reason", None) or error).strip()
    detail = text.splitlines()[0] if text else type(error).__name__
    return OfflineChildContractError(
        f"{path} cannot be read as a parent history file ({detail}). "
        "It is incomplete or damaged: restore it from the parent run, or "
        "run the parent again to regenerate it, then retry")


class _ParentHistory:
    """Open one parent history file; a failure OF THAT FILE becomes a refusal naming it.

    ``opener`` ``None`` is the Rust reader.  It is resolved in
    :meth:`__enter__` BEFORE the file is opened and outside the
    translation: a missing, stale or incompatible ``rw_netcdf`` (or a
    ``WOOF_RW_NETCDF`` naming nothing) is the reader's failure, and it
    keeps its own message and the reader's remedy.  Translating it too is
    how a healthy parent was once called damaged, with advice to re-run a
    forecast that can cost hours of GPU, while the build instructions that
    would have fixed the reader were cut off.

    What IS translated: the reader running and refusing this file
    (:class:`netcdf_bridge.NetcdfFileError` naming this path), and this
    path not being there.  With a library ``opener`` (``netCDF4.Dataset``)
    there is no separate reader, and its ``OSError`` is the file's.
    """

    def __init__(self, path, opener=None):
        self._path = Path(path)
        self._opener = opener
        self._dataset = None

    def _this_files(self, error) -> bool:
        if self._opener is not None:
            return isinstance(error, OSError)
        if isinstance(error, netcdf_bridge.NetcdfFileError):
            return error.path == self._path
        return isinstance(error, FileNotFoundError) and not self._path.is_file()

    def __enter__(self):
        if self._opener is None:
            reader = netcdf_bridge.resolve_netcdf_bin()

            def opener(path):
                return netcdf_bridge.open_dataset(path, executable=reader)
        else:
            opener = self._opener
        try:
            self._dataset = opener(self._path)
        except (OSError, netcdf_bridge.NetcdfDecodeError) as error:
            if self._this_files(error):
                raise _unreadable_history(self._path, error) from None
            raise
        return self._dataset.__enter__()

    def __exit__(self, kind, error, trace):
        self._dataset.__exit__(kind, error, trace)
        if error is not None and self._this_files(error):
            raise _unreadable_history(self._path, error) from None
        return False


@dataclass(frozen=True)
class ParentHistoryFrame:
    """Metadata-only proof for one archived parent state."""

    path: Path
    valid_time: datetime
    source_kind: str
    source_mp_physics: int | None
    inferred_mp_physics: int | None
    dimensions: Mapping[str, int]
    variables: frozenset[str]
    geometry_sha256: str


@dataclass(frozen=True)
class ParentHistoryContract:
    """Validated ordered parent series ready for native child preparation."""

    frames: tuple[ParentHistoryFrame, ...]
    interval_seconds: float
    geometry_sha256: str
    source_kind: str
    source_mp_physics: int | None
    physics_binding: ParentPhysicsBinding | None
    max_boundary_interval_seconds: float

    @property
    def start_time(self) -> datetime:
        return self.frames[0].valid_time

    @property
    def end_time(self) -> datetime:
        return self.frames[-1].valid_time


#: The first four bytes of a classic NetCDF file: CDF-1, CDF-2, CDF-5.
_CLASSIC_SIGNATURES = frozenset({b"CDF\x01", b"CDF\x02", b"CDF\x05"})

#: Classic history files already proven whole in this process, keyed by
#: resolved path, size and modification time.  One command inspects a
#: frame more than once (the cadence, the contract, then each read), and
#: a file restored or rewritten since is a new key.
_WHOLE_HISTORY: set[tuple[str, int, int]] = set()


def _last_stored_variable(dataset) -> str | None:
    """The variable whose bytes end a classic file, or None when none has any.

    A classic file stores its fixed variables in the order they were
    defined and then its records, each holding the record variables in
    that same order, so the last record variable (else the last fixed
    one) ends the file.
    """

    records = {name for name, dimension in dataset.dimensions.items()
               if dimension.isunlimited() and len(dimension) > 0}
    # A scalar holds one value; a variable with an empty axis holds none.
    stored = [name for name, variable in dataset.variables.items()
              if all(variable.shape)]
    on_records = [name for name in stored
                  if dataset.variables[name].dimensions[:1]
                  and dataset.variables[name].dimensions[0] in records]
    chosen = on_records or stored
    return chosen[-1] if chosen else None


def _require_whole_history(path: Path) -> None:
    """Refuse a classic history file that ends before its data does.

    woof's own wrfout writer and stock WRF both write CDF-2.  Such a file
    cut off partway through its data keeps its header: netCDF4 opens it and
    reads every missing value as zero without an error, and the header's
    GPUWM_WRITE_COMPLETE stamp still says it was finished.  A truncated
    first frame then reached the NetCDF decoder as a traceback, and a
    truncated later frame was refused as a parent whose geometry changed
    between frames, which is not what is wrong with it.

    The Rust reader's inventory proves every byte the header describes is
    in the file.  A reader older than that proof is asked instead for the
    variable stored last, whose decode checks the same final bytes.  A
    NetCDF-4 file is its own library's to check, and it refuses a short
    one at open.  Either refusal names the file and the way back.
    """

    try:
        with open(path, "rb") as handle:
            signature = handle.read(4)
        stat = path.stat()
    except OSError as error:
        raise _unreadable_history(path, error) from None
    if signature not in _CLASSIC_SIGNATURES:
        return
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key in _WHOLE_HISTORY:
        return
    with _ParentHistory(path) as dataset:
        if dataset.extent_checked is None:
            last = _last_stored_variable(dataset)
            if last is not None:
                dataset.variables[last][:]
    _WHOLE_HISTORY.add(key)


def open_parent_history(path: str | Path, opener=None) -> _ParentHistory:
    """One parent history file, proven whole and opened; failures are sentences.

    For every reader of a parent frame outside this module.  ``opener``
    ``None`` is the Rust reader; ``netCDF4.Dataset`` is for the readers
    that take identity attributes and dimensions off the header.  A file
    that is not there, cannot be opened or decoded, or is a classic file
    cut off partway through its data is refused naming the file and the
    way back, never a traceback, and a stale or missing reader keeps its
    own message (see :class:`_ParentHistory`).
    """

    path = Path(path)
    _require_whole_history(path)
    return _ParentHistory(path, opener)


def inspect_parent_history_frame(
        path: str | Path, *, source_mp_physics: int | None = None,
) -> ParentHistoryFrame:
    """Inspect one woof/WRF history file without loading 3-D trajectory data.

    A classic-format file is first proven whole
    (:func:`_require_whole_history`): netCDF4 reads a truncated one's
    missing data as zeros, which every check below would take as data.
    """

    path = Path(path)
    _require_whole_history(path)
    with _ParentHistory(path, netCDF4.Dataset) as dataset:
        feedback = (
            dataset.getncattr("GPUWM_FEEDBACK")
            if "GPUWM_FEEDBACK" in dataset.ncattrs() else None)
        if str(feedback).strip().lower() == "experimental":
            raise OfflineChildContractError(
                f"{path} carries experimental two-way feedback provenance; "
                "woof downscale assumes a one-way parent archive and "
                "refuses feedback-modified parent history")
        if "Times" not in dataset.variables:
            raise OfflineChildContractError(f"{path} has no WRF Times variable")
        valid_time = _decode_time(dataset.variables["Times"])
        variables = frozenset(dataset.variables)
        missing = sorted(_REQUIRED_DYNAMICS - variables)
        if missing:
            raise OfflineChildContractError(
                f"{path} is missing offline-child trajectory fields {missing}")
        dimensions = {}
        for name in _GEOMETRY_DIMS:
            if name not in dataset.dimensions:
                raise OfflineChildContractError(
                    f"{path} is missing WRF geometry dimension {name!r}")
            dimensions[name] = len(dataset.dimensions[name])

        title = str(getattr(dataset, "TITLE", ""))
        source_kind = "woof" if (
            "woof" in title.lower() or "GPUWM_WRITE_COMPLETE" in dataset.ncattrs()
        ) else "wrf"
        inferred_mp = _infer_mp_physics(variables, source_kind=source_kind)
        bound_mp = None
        if source_mp_physics is None and inferred_mp is not None and (
                inferred_mp not in OFFLINE_CHILD_MP_PHYSICS):
            # Nothing was declared, so this inference is the only scheme
            # evidence there is -- and the arms are single-package
            # discriminants, not subset matches, so a positive one is not a
            # guess.  Without this the reader fell through to the blind
            # six-species contract, which a WDM6 archive SATISFIES: the run
            # would have been prepared with the parent's warm-rain numbers
            # and its CCN reservoir silently dropped, which is the exact
            # cross-scheme entry-closure breakage the mixed nest edge
            # refuses by name (audit R-018).
            raise OfflineChildContractError(
                f"{path} carries the transported inventory of "
                f"mp_physics={inferred_mp} and no parent scheme was "
                "declared, so that inventory is the only evidence: "
                + _unsupported_parent_clause(inferred_mp, what="inferred"))
        if source_mp_physics is not None:
            requested = int(source_mp_physics)
            if requested not in OFFLINE_CHILD_MP_PHYSICS:
                raise OfflineChildContractError(
                    _unsupported_parent_clause(requested, what="declared"))
            # Inventory-only inference is advisory. WRF streams may retain
            # dormant number variables, and unified NSSL can expose an
            # inventory that looks like another multi-moment scheme. The
            # companion namelist/setup/manifest is authoritative.
            bound_mp = requested

        digest = hashlib.sha256()
        for name in _GEOMETRY_DIMS:
            digest.update(f"dim:{name}={dimensions[name]};".encode("ascii"))
        for name in _GEOMETRY_ATTRS:
            if name in dataset.ncattrs():
                digest.update(f"attr:{name}={dataset.getncattr(name)!r};".encode())
        for name in _STATIC_GEOMETRY_FIELDS:
            if name not in dataset.variables:
                if name in {"XLAT", "XLONG", "MAPFAC_M"}:
                    # Some minimal stock-WRF history streams omit these; the
                    # projection/dimension identity remains enforceable and a
                    # companion native setup archive supplies the arrays.
                    continue
                raise OfflineChildContractError(
                    f"{path} is missing vertical/static geometry field {name}")
            value = dataset.variables[name][:]
            if value.ndim and value.shape[0] == 1:
                value = value[0]
            _hash_array(digest, name, value)

    return ParentHistoryFrame(
        path=path.resolve(), valid_time=valid_time, source_kind=source_kind,
        source_mp_physics=bound_mp, inferred_mp_physics=inferred_mp,
        dimensions=MappingProxyType(dimensions), variables=variables,
        geometry_sha256=digest.hexdigest(),
    )


def validate_parent_history(
        paths: Sequence[str | Path], *, max_boundary_interval_seconds: float,
        source_mp_physics: int | None = None,
        physics_binding: ParentPhysicsBinding | None = None,
) -> ParentHistoryContract:
    """Prove an archived parent series before constructing a child.

    ``max_boundary_interval_seconds`` is intentionally mandatory.  Cadence is
    a scientific choice tied to child resolution and expected advection; the
    tool will not silently bless hourly parent history for a 500-m child.

    This proves the series' shape (times, cadence, geometry, scheme
    identity) from metadata and does not read the 3-D fields.  Their
    values are proven where they are read: every required field of every
    frame goes through :func:`_read_record`, which refuses a missing or
    non-finite value naming the file, the variable and the cell.  The
    runner reads the initial frame and then every boundary frame
    (:func:`build_offline_lateral_boundaries`) before it builds the
    stepper, so a damaged later frame stops the run before one step is
    integrated, without this check reading the whole archive twice.
    """

    if physics_binding is not None:
        if source_mp_physics is not None:
            raise OfflineChildContractError(
                "pass physics_binding or source_mp_physics, not both")
        source_mp_physics = int(physics_binding.mp_physics)
    maximum = float(max_boundary_interval_seconds)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise OfflineChildContractError(
            "max_boundary_interval_seconds must be finite and positive")
    frames = tuple(inspect_parent_history_frame(
        path, source_mp_physics=source_mp_physics) for path in paths)
    if len(frames) < 2:
        raise OfflineChildContractError(
            "offline child forcing requires at least two parent history frames")
    if len({frame.geometry_sha256 for frame in frames}) != 1:
        raise OfflineChildContractError(
            "parent history geometry/static state changes between frames "
            "(a moving parent, such as a storm run's following nest: a child "
            "placed once on one frame's grid would read the others at the "
            "wrong ground); woof downscale --follow storm|parent|fixed "
            "reads each frame on its own footprint")
    if len({frame.source_kind for frame in frames}) != 1:
        raise OfflineChildContractError(
            "parent history mixes woof and stock-WRF producers")
    if len({frame.source_mp_physics for frame in frames}) != 1:
        raise OfflineChildContractError(
            "parent history bound microphysics identity changes between frames")
    if len({frame.inferred_mp_physics for frame in frames}) != 1:
        raise OfflineChildContractError(
            "parent history advisory moisture inventory changes between frames")
    seconds = np.asarray([
        (frame.valid_time - frames[0].valid_time).total_seconds()
        for frame in frames
    ], dtype=np.float64)
    differences = np.diff(seconds)
    if not np.all(np.isfinite(differences)) or np.any(differences <= 0.0):
        raise OfflineChildContractError(
            "parent history times must be unique and strictly increasing")
    interval = float(differences[0])
    if not np.all(differences == interval):
        raise OfflineChildContractError(
            f"parent history cadence is irregular: {differences.tolist()} seconds")
    if interval > maximum:
        raise OfflineChildContractError(
            f"parent history cadence {interval:g} s exceeds the declared child "
            f"forcing limit {maximum:g} s; regenerate a denser parent archive")
    return ParentHistoryContract(
        frames=frames, interval_seconds=interval,
        geometry_sha256=frames[0].geometry_sha256,
        source_kind=frames[0].source_kind,
        source_mp_physics=frames[0].source_mp_physics,
        physics_binding=physics_binding,
        max_boundary_interval_seconds=maximum,
    )


def read_parent_microphysics(
        path: str | Path, *, source_mp_physics: int | None = None,
) -> Mapping[str, np.ndarray]:
    """Read only transported moisture fields from one WRF history record.

    With ``source_mp_physics`` bound, the completeness check is the
    scheme's own transported inventory (``_transported_source_fields``),
    read through the same WRF-name mapping ``_raw_parent_state`` uses.
    Without it the historical closed-world contract stands: the six-species
    mass set is required, because with no scheme evidence there is no
    smaller inventory this reader could accurately call complete -- a P3
    archive (no QSNOW/QGRAUP by Registry.EM_COMMON:3038) is readable
    through the evidence-bearing form, not by weakening the blind one.
    """

    fields: dict[str, np.ndarray] = {}
    wrf_mapping = _WRF_TO_STATE
    required = set(_MASS_FIELDS)
    label = "transported parent mass fields"
    if source_mp_physics is not None:
        source_mp = int(source_mp_physics)
        if source_mp not in OFFLINE_CHILD_MP_PHYSICS:
            raise OfflineChildContractError(
                _unsupported_parent_clause(source_mp, what="declared"))
        wrf_mapping = _scheme_wrf_to_state(source_mp)
        required = set(_transported_source_fields(source_mp))
        label = f"mp_physics={source_mp} transported parent fields"
    # Decoded by the Rust bridge: transported moisture is meteorological
    # field data, whoever wrote the tape.
    with _ParentHistory(path) as dataset:
        for wrf_name, state_name in wrf_mapping.items():
            if wrf_name not in dataset.variables:
                continue
            value = np.asarray(dataset.variables[wrf_name][:])
            if value.shape[0] != 1:
                raise OfflineChildContractError(
                    f"{path}/{wrf_name} must have exactly one Time record")
            fields[state_name] = np.ascontiguousarray(value[0], dtype=np.float32)
    missing = sorted(required - set(fields))
    if missing:
        raise OfflineChildContractError(
            f"{path} lacks {label} {missing}")
    return MappingProxyType(fields)


#: Cells per device round on the host-chunked conversion route.  The edge
#: kernel is column-local (every output cell reads its own cell's planes
#: and its own column's 2-D mass), so a parent is converted a band of
#: rows at a time and the band is sized to this many cells: the CPU
#: preprocess route keeps working on a card too small to hold the whole
#: parent, and the interpolation after it stays on the host as before.
CONVERSION_CHUNK_CELLS = 262144


@dataclass(frozen=True)
class OfflineSchemeTransition:
    """One resolved cross-scheme conversion, and how it was executed."""

    contract: _mt.MicrophysicsTransitionContract
    fields: Mapping[str, np.ndarray]
    device: str
    host_chunked: bool
    chunk_rows: int | None
    chunks: int
    parent_hypsometric_opt: int

    def receipt(self) -> Mapping[str, object]:
        """The online contract's own receipt, plus how this lane ran it.

        Carries the source and target identity twice on purpose: once as
        the contract spells it (``source_mp_physics``/``target_mp_physics``)
        and once under this lane's own ``source``/``target`` rows, so a
        restart or run record that syncs on the conversion seam finds both
        ends named whichever key its reader walks.
        """
        receipt = dict(self.contract.receipt())
        receipt.update({
            "source": {"mp_physics": int(self.contract.source_mp_physics),
                       "rimed_category": self.contract.source_rimed_category},
            "target": {"mp_physics": int(self.contract.target_mp_physics),
                       "rimed_category": self.contract.target_rimed_category},
            "converted_fields": tuple(self.fields),
            "conversion_site": "archived-parent-grid",
            "translation_order": _mt.TRANSITION_ORDER,
            "executor": "woof.core.microphysics_transition."
                        "launch_microphysics_edge_parent_field",
            "device": self.device,
            "host_chunked": bool(self.host_chunked),
            "chunk_rows": self.chunk_rows,
            "chunks": int(self.chunks),
            "parent_density": (
                "1/alt from the archived total geopotential and dry mass, "
                f"hypsometric_opt={int(self.parent_hypsometric_opt)}, the "
                "form woof.core.diagnostics.update_diagnostics evaluates"),
        })
        return MappingProxyType(receipt)


def _offline_transition_contract(
        source_mp_physics: int, target_mp_physics: int, *,
        morr_rimed_ice: int | None, hail_opt: int | None,
        child_cfg=None) -> _mt.MicrophysicsTransitionContract:
    """Resolve the ONLINE edge contract for an archived parent and a child.

    The parent is described by its bound evidence (scheme, Morrison's
    rimed-ice switch, WSM6/WDM6's hail switch); the child by its own
    RunConfig when the caller has one, and otherwise by the target scheme
    with the config defaults, which is what a direct library caller gets.
    The child's ``nest_microphysics_transition`` is always the unset
    default here -- a standalone child is not a nested domain and
    validate_run_config refuses any other spelling on it -- so the pair
    resolves to the one closure the matrix defines for it, exactly as an
    online nest with the key left out does.
    """
    from types import SimpleNamespace

    source_mp, target_mp = int(source_mp_physics), int(target_mp_physics)
    _require_cross_scheme_contract(source_mp, target_mp)
    # HN1 (correct-mode key hydrometeor_handover_numbers) rides to the
    # resolver, which reads it off the child to decide whether the edge
    # carries the parent's numbers; exact mode turns the carry off.
    from woof.ingest.handover_numbers import closure_active
    handover_numbers = closure_active(child_cfg)
    parent = SimpleNamespace(
        mp_physics=source_mp, moist=True, moist_cq=True,
        morr_rimed_ice=(1 if morr_rimed_ice is None else int(morr_rimed_ice)),
        wsm6_hail_opt=(0 if hail_opt is None else int(hail_opt)),
        wdm6_hail_opt=(0 if hail_opt is None else int(hail_opt)),
        nest_microphysics_transition=_mt.SAME_SCHEME_POLICY)
    if child_cfg is None:
        child = SimpleNamespace(
            mp_physics=target_mp, moist=True, moist_cq=True,
            morr_rimed_ice=1, wsm6_hail_opt=0, wdm6_hail_opt=0,
            wdm6_ccn_conc=1.0e8,
            hydrometeor_handover_numbers=handover_numbers,
            nest_microphysics_transition=_mt.SAME_SCHEME_POLICY)
    else:
        if int(getattr(child_cfg, "mp_physics")) != target_mp:
            raise OfflineChildContractError(
                f"child cfg mp_physics={child_cfg.mp_physics} != prepared "
                f"target {target_mp}")
        child = SimpleNamespace(
            mp_physics=target_mp,
            moist=bool(getattr(child_cfg, "moist", True)),
            moist_cq=bool(getattr(child_cfg, "moist_cq", True)),
            morr_rimed_ice=int(getattr(child_cfg, "morr_rimed_ice", 1)),
            wsm6_hail_opt=int(getattr(child_cfg, "wsm6_hail_opt", 0)),
            wdm6_hail_opt=int(getattr(child_cfg, "wdm6_hail_opt", 0)),
            wdm6_ccn_conc=float(getattr(child_cfg, "wdm6_ccn_conc", 1.0e8)),
            hydrometeor_handover_numbers=handover_numbers,
            nest_microphysics_transition=_mt.SAME_SCHEME_POLICY)
    try:
        return _mt.resolve_microphysics_transition(parent, child)
    except ValueError as error:
        raise OfflineChildContractError(
            "offline cross-physics conversion across the mp_physics="
            f"{source_mp} -> {target_mp} edge is refused by the shared "
            f"nest-transition contract: {error}") from error


def _parent_alt(phi_total, mu, coeffs, znw, *, p_top: float,
                hypsometric_opt: int) -> np.ndarray:
    """``alt`` (inverse dry density) on the parent grid, from the archive.

    The same float32 expression ``woof.core.diagnostics.update_diagnostics``
    evaluates on a live parent for each ``hypsometric_opt``, so the density
    the edge kernel diagnoses a target's moments against offline is the
    density it would have read from the live parent's state.
    """
    f32 = np.float32
    phi = np.asarray(phi_total, dtype=np.float32)
    dphi = np.asarray(phi[1:] - phi[:-1], dtype=np.float32)
    mu = np.asarray(mu, dtype=np.float32)[None]
    if int(hypsometric_opt) == 2:
        c3f = np.asarray(coeffs["c3f"], dtype=np.float32)[:, None, None]
        c4f = np.asarray(coeffs["c4f"], dtype=np.float32)[:, None, None]
        c3h = np.asarray(coeffs["c3h"], dtype=np.float32)[:, None, None]
        c4h = np.asarray(coeffs["c4h"], dtype=np.float32)[:, None, None]
        top = f32(p_top)
        pfu = np.asarray(c3f[1:] * mu + c4f[1:] + top, dtype=np.float32)
        dpf = np.asarray((c3f[:-1] - c3f[1:]) * mu + (c4f[:-1] - c4f[1:]),
                         dtype=np.float32)
        phm = np.asarray(c3h * mu + c4h + top, dtype=np.float32)
        alt = np.asarray(
            dphi / phm / log1pf_array(np.asarray(dpf / pfu,
                                                 dtype=np.float32)),
            dtype=np.float32)
    else:
        znw = np.asarray(znw, dtype=np.float32).reshape(-1)
        rdnw = np.asarray(f32(1.0) / (znw[1:] - znw[:-1]),
                          dtype=np.float32)[:, None, None]
        c1h = np.asarray(coeffs["c1h"], dtype=np.float32)[:, None, None]
        c2h = np.asarray(coeffs["c2h"], dtype=np.float32)[:, None, None]
        alt = np.asarray(-dphi * rdnw / (c1h * mu + c2h), dtype=np.float32)
    if not np.isfinite(alt).all() or np.any(alt <= 0.0):
        raise OfflineChildContractError(
            "cross-scheme conversion requires a positive finite parent "
            "layer density, and the archived geopotential and dry mass "
            "give none")
    return np.ascontiguousarray(alt)


def _parent_transition_donor(raw, moisture, coeffs, *, p_top: float,
                             hypsometric_opt: int, target_mp_physics: int):
    """The parent-grid planes the edge kernel reads, as host float32 arrays.

    The same set the live lane hands the kernel off a resident
    ``DomainState`` (``microphysics_transition._WINDOWED_EDGE_PLANES``),
    rebuilt from the archive: ``alt`` from the total geopotential and dry
    mass, the species under their state names, the 2-D mass pair and the
    hybrid coefficients.  Entering Milbrandt-Yau the kernel forms the
    absolute temperature from thb/thp/p per cell, so for that target the
    archived total theta is handed over as ``thb`` with a zero ``thp``
    and the full pressure as ``p``; every other target ignores the three.
    """
    ny, nx = np.asarray(raw["MUB"]).shape[-2:]
    phb = np.asarray(raw["PHB"], dtype=np.float64)
    ph = np.asarray(raw["PH"], dtype=np.float64)
    mub = np.ascontiguousarray(raw["MUB"], dtype=np.float32)
    mup = np.ascontiguousarray(raw["MU"], dtype=np.float32)
    total_mu = np.asarray(mub, dtype=np.float32) + np.asarray(mup, dtype=np.float32)
    planes = {
        "alt": _parent_alt(phb + ph, total_mu, coeffs, raw["ZNW"],
                           p_top=p_top, hypsometric_opt=hypsometric_opt),
        "mub2d": mub, "mup": mup,
        "c1h": np.ascontiguousarray(coeffs["c1h"], dtype=np.float32),
        "c2h": np.ascontiguousarray(coeffs["c2h"], dtype=np.float32),
    }
    # nr/ni: the numbers a mixed edge carries rather than re-diagnoses
    # (microphysics_transition._CARRIED_NUMBERS); None when the parent
    # scheme has none, and the kernel's closure then runs.
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh", "qir", "qib",
                 "nr", "ni"):
        value = moisture.get(name)
        planes[name] = (None if value is None
                        else np.ascontiguousarray(value, dtype=np.float32))
    if int(target_mp_physics) == 9:
        theta = np.asarray(raw["T"], dtype=np.float32) + np.float32(300.0)
        planes["thb"] = np.ascontiguousarray(theta, dtype=np.float32)
        planes["thp"] = np.zeros(theta.shape, dtype=np.float32)
        planes["p"] = np.ascontiguousarray(
            np.asarray(raw["P"], dtype=np.float32)
            + np.asarray(raw["PB"], dtype=np.float32), dtype=np.float32)
    else:
        planes["thb"] = planes["thp"] = planes["p"] = None
    return planes


def _launch_conversion(contract, planes, names, *, coupled: bool, xp):
    """Run the edge kernel once per target field on device-resident planes."""
    from types import SimpleNamespace

    donor = SimpleNamespace(**planes)
    shape = tuple(int(v) for v in donor.qv.shape)
    out = {}
    for name in names:
        target = xp.empty(shape, dtype=xp.float32)
        _mt.launch_microphysics_edge_parent_field(
            contract, donor, name, out=target, coupled=bool(coupled))
        out[name] = target
    return out


def _convert_parent_microphysics(
        contract, planes, *, coupled: bool, backend: str,
        hypsometric_opt: int) -> OfflineSchemeTransition:
    """Diagnose the target scheme's fields on the PARENT grid.

    ``backend == "cuda"``: every plane goes to the device once and the
    kernel runs on the whole parent, the converted fields staying on the
    device for the interpolation that follows.  ``backend == "cpu"``: the
    kernel has no host implementation, so the parent is converted a band
    of rows at a time through the device (host to device to host), the
    way the retired NSSL initializer did, and the fields come back to the
    host for the CPU interpolation.  Either way the conversion runs before
    any horizontal interpolation, in the online lane's order.
    """
    names = _mt.transition_target_fields(contract)
    try:
        import cupy as cp
    except Exception as error:   # pragma: no cover - box without a GPU
        raise OfflineChildContractError(
            "offline cross-physics conversion runs the nest-transition "
            "kernel (woof/core/kernels/nest_microphysics.cu), which has "
            "no host implementation, and CuPy/CUDA is not usable on this "
            f"machine: {error}.  Prepare the child on a machine with a "
            "working CUDA device, or keep the parent's own scheme") from error
    if backend == "cuda":
        device_planes = {
            name: (None if value is None
                   else cp.ascontiguousarray(cp.asarray(value, dtype=cp.float32)))
            for name, value in planes.items()}
        fields = _launch_conversion(contract, device_planes, names,
                                    coupled=coupled, xp=cp)
        return OfflineSchemeTransition(
            contract=contract, fields=MappingProxyType(fields), device="cuda",
            host_chunked=False, chunk_rows=None, chunks=1,
            parent_hypsometric_opt=int(hypsometric_opt))
    if backend != "cpu":
        raise OfflineChildContractError(
            f"offline-child backend must be cpu or cuda, got {backend!r}")
    nz, ny, nx = (int(v) for v in np.asarray(planes["qv"]).shape)
    rows = max(1, min(ny, CONVERSION_CHUNK_CELLS // max(1, nz * nx)))
    host = {name: np.empty((nz, ny, nx), dtype=np.float32) for name in names}
    columnar = {"c1h", "c2h"}
    chunks = 0
    for j0 in range(0, ny, rows):
        j1 = min(ny, j0 + rows)
        band = {}
        for name, value in planes.items():
            if value is None:
                band[name] = None
            elif name in columnar:
                band[name] = cp.ascontiguousarray(
                    cp.asarray(value, dtype=cp.float32))
            elif np.ndim(value) == 2:
                band[name] = cp.ascontiguousarray(
                    cp.asarray(value[j0:j1], dtype=cp.float32))
            else:
                band[name] = cp.ascontiguousarray(
                    cp.asarray(value[:, j0:j1], dtype=cp.float32))
        converted = _launch_conversion(contract, band, names,
                                       coupled=coupled, xp=cp)
        for name, value in converted.items():
            host[name][:, j0:j1] = cp.asnumpy(value)
        del band, converted
        chunks += 1
    return OfflineSchemeTransition(
        contract=contract, fields=MappingProxyType(host), device="cuda",
        host_chunked=True, chunk_rows=int(rows), chunks=int(chunks),
        parent_hypsometric_opt=int(hypsometric_opt))


#: Land/soil identity attributes a child surface source must declare.
#: Defaults here would silently rebind category semantics (water index,
#: ice index) across landuse tables, so they are required evidence.
_SURFACE_IDENTITY_ATTRS = ("MMINLU", "ISWATER", "ISLAKE", "ISICE")

#: Child-grid surface fields.  ``required`` is the minimum accurate warm
#: start for a land-surface + surface-layer child; ``optional`` fields
#: are carried when present and receipted either way.  All arrays are
#: read on the EXACT child grid -- like WRF's ``ndown``, woof's offline
#: child takes its static/soil identity from a child-grid initialization
#: file and replaces only the meteorology from the parent archive.
_SURFACE_REQUIRED_FIELDS = (
    "LU_INDEX", "LANDMASK", "ISLTYP", "TSK", "TMN", "VEGFRA",
    "TSLB", "SMOIS", "SNOW",
)
_SURFACE_OPTIONAL_FIELDS = (
    "SH2O", "SNOWH", "SNOWC", "SEAICE", "XICE", "PBLH", "UST",
    "PSFC", "T2", "Q2", "TH2", "U10", "V10", "XLAT", "XLONG",
)
_SURFACE_CATEGORY_FIELDS = frozenset({"LU_INDEX", "ISLTYP"})
_SURFACE_SOIL_FIELDS = frozenset({"TSLB", "SMOIS", "SH2O"})

#: Noah's four layer midpoints in whole centimetres (0-10, 10-40, 40-100 and
#: 100-200 cm), the ladder a parent history without a ZS record is read on.
NOAH_LAYER_MIDPOINTS_CM = (5, 25, 70, 150)


def parent_soil_onto_child_levels(parent_layers: int,
                                  num_soil_layers: int) -> bool:
    """Whether a parent's soil column is rebuilt on the child LSM's levels.

    One edge: a four-layer Noah parent under a RUC child (six or nine
    levels), the child a refined nest-down takes when it runs the HRRR-shape
    suite under a Noah parent (:func:`woof.physics_menu.
    convection_permitting_child_profile`).  Every other mismatch stays the
    refusal it was.
    """

    from woof.ingest.ruc_soil import RUC_LEVEL_DEPTHS_M

    return (int(parent_layers) == len(NOAH_LAYER_MIDPOINTS_CM)
            and int(num_soil_layers) in RUC_LEVEL_DEPTHS_M
            and int(num_soil_layers) != int(parent_layers))


def noah_soil_onto_ruc_levels(fields: dict, *, parent_zs_m,
                              num_soil_layers: int) -> dict:
    """``fields`` with TSLB/SMOIS rebuilt on RUC's levels, SH2O dropped.

    The parent's four Noah layers are a LAYER source in WRF's sense, so the
    column is real.exe's ``init_soil_3_real`` for a layer source
    (:func:`woof.ingest.ruc_soil.remap_soil_to_ruc_levels`): anchored at
    0 m by the skin temperature and at 3 m by the child's deep temperature
    (TMN), moisture extrapolated to the surface off the top two layers.
    SH2O is left to the land model's own initialization, as a RUC cold start
    has it.  Water columns take the skin temperature and saturated moisture,
    as that routine writes them.
    """

    from woof.ingest.ruc_soil import remap_soil_to_ruc_levels

    if parent_zs_m is None:
        levels_cm = np.asarray(NOAH_LAYER_MIDPOINTS_CM, dtype=np.int64)
    else:
        levels_cm = np.rint(np.asarray(parent_zs_m, dtype=np.float64)
                            .reshape(-1) * 100.0).astype(np.int64)
    columns = remap_soil_to_ruc_levels(
        source_temperature=np.asarray(fields["TSLB"], dtype=np.float32),
        source_moisture=np.asarray(fields["SMOIS"], dtype=np.float32),
        source_levels_cm=levels_cm, source_geometry="layers",
        skin_temperature=np.asarray(fields["TSK"], dtype=np.float32),
        deep_temperature=np.asarray(fields["TMN"], dtype=np.float32),
        landmask=np.asarray(fields["LANDMASK"], dtype=np.float32),
        num_soil_layers=int(num_soil_layers))
    out = {name: value for name, value in fields.items() if name != "SH2O"}
    out["TSLB"] = columns.soil_temperature
    out["SMOIS"] = columns.soil_moisture
    out["TSK"] = columns.skin_temperature
    return out


def noah_soil_onto_ruc_receipt(parent_zs_m, num_soil_layers: int) -> dict:
    return {
        "law": "init_soil_3_real, layer source (module_soil_pre.F)",
        "source_layer_midpoints_cm": (
            list(NOAH_LAYER_MIDPOINTS_CM) if parent_zs_m is None else
            [int(v) for v in np.rint(np.asarray(parent_zs_m).reshape(-1)
                                     * 100.0)]),
        "child_levels": int(num_soil_layers),
        "sh2o": "dropped; the land model initializes it",
    }


#: The remedy half of the child-surface refusal, shared by the front
#: door's early check (`woof downscale`) and the runner's late guard so
#: the two cannot drift.  It names the flag, the contract, AND an
#: in-product way to SATISFY it: the walked 2.4.1 refusal named only the
#: first two, and the walked user's own preparation already held a valid
#: child-grid file at ``wrf-native-input/wrfinput_d0N`` -- rw-wps emits
#: one per nest -- with no sentence anywhere pointing at it.
CHILD_SURFACE_SOURCE_REMEDY = (
    "pass --child-surface-from with a wrfinput or history file on the "
    "EXACT child grid (ndown-equivalent contract: downscaling replaces "
    "the meteorology, never the land identity).  woof's own "
    "preprocessor builds one per nest: prepare a hierarchy whose nest "
    "IS this child grid (`woof domain --ladder ...`, then rw-wps) and "
    "point the flag at <prepared>/wrf-native-input/wrfinput_d0N; an "
    "archived woof or WRF history frame on the exact child grid works "
    "too")


def child_surface_requirement(cfg) -> str | None:
    """Why this child config needs ``--child-surface-from``, or ``None``.

    The predicate is :func:`woof.offline_child_run.
    _initialize_child_physics`'s own: any of the land-surface, surface-
    layer or PBL selections requires child-grid soil state and land
    identity, which are never fabricated on a real-data child.
    """
    if not (getattr(cfg, "sf_surface_physics", 0)
            or getattr(cfg, "sf_sfclay_physics", 0)
            or getattr(cfg, "bl_pbl_physics", 0)):
        return None
    return (
        "child config enables surface physics (sf_surface_physics="
        f"{cfg.sf_surface_physics}, sf_sfclay_physics="
        f"{cfg.sf_sfclay_physics}, bl_pbl_physics={cfg.bl_pbl_physics}) "
        "but no child-grid surface source was given; "
        + CHILD_SURFACE_SOURCE_REMEDY)


def child_mosaic_refusal(cfg) -> str | None:
    """Why this child config cannot run here, for Noah mosaic, or ``None``.

    Noah mosaic's tiles are built from LANDUSEF by an initialisation door
    (woof/core/noah_mosaic_door.py), and this route builds its child's
    physics driver (woof.offline_child_run._initialize_child_physics)
    from a child-grid surface file or from the parent's own history, with
    no such door: a mosaic child was accepted, paid for the parent
    interpolation, and then stopped on its first land-surface step ("this
    initialisation door built no tile state").  A derived child copies
    the parent's physics, so a mosaic parent's child arrives here too.
    """
    if int(getattr(cfg, "sf_surface_mosaic", 0) or 0) != 1:
        return None
    return (
        "child config sets sf_surface_mosaic = 1 (Noah mosaic), which the "
        "offline child cannot run: it builds the child's land surface from "
        "a child-grid surface file or from the parent's history, and "
        "neither path builds the land-use tiles mosaic integrates, so its "
        "first land-surface step would stop.  Pass --child-config with "
        "sf_surface_mosaic = 0 (the parent keeps its mosaic), or run the "
        "nest in the same tree as its parent, where the nest door builds "
        "the tiles.")


@dataclass(frozen=True)
class ChildSurfaceState:
    """Surface/soil warm-start state read from one child-grid file."""

    path: Path
    fields: Mapping[str, np.ndarray]
    identity: Mapping[str, object]
    receipt: Mapping[str, object]


def read_child_surface_state(
        path: str | Path, *, child_ny: int, child_nx: int,
        num_soil_layers: int) -> ChildSurfaceState:
    """Read surface/soil warm-start fields from an exact-child-grid file.

    Accepts a stock-WRF ``wrfinput``/history file or a woof history
    file whose mass grid is exactly the child's.  This is the offline
    analogue of WRF ``ndown``'s requirement that the child's own
    ``wrfinput`` (from ``real.exe``) supplies static and soil state --
    downscaling replaces the meteorology, never the land identity.
    """

    path = Path(path)
    fields: dict[str, np.ndarray] = {}
    # Reads the child surface FIELDS (_SURFACE_REQUIRED_FIELDS), not just
    # the dimensions above them, so it decodes and goes through Rust.
    # The f32 cast below is unaffected: the bridge promotes f32 storage to
    # f64 exactly, and casting back reproduces the stored bits.
    with netcdf_bridge.open_dataset(path) as dataset:
        for name, expected in (("south_north", int(child_ny)),
                               ("west_east", int(child_nx))):
            if name not in dataset.dimensions:
                raise OfflineChildContractError(
                    f"{path} is missing surface-grid dimension {name!r}")
            actual = len(dataset.dimensions[name])
            if actual != expected:
                raise OfflineChildContractError(
                    f"{path} {name}={actual} does not match the child "
                    f"grid {name}={expected}; the surface source must be "
                    "on the EXACT child grid")
        if "soil_layers_stag" in dataset.dimensions:
            soil = len(dataset.dimensions["soil_layers_stag"])
            if soil != int(num_soil_layers):
                raise OfflineChildContractError(
                    f"{path} soil_layers_stag={soil} does not match the "
                    f"child LSM's {num_soil_layers} soil layers")
        elif int(num_soil_layers) > 0:
            raise OfflineChildContractError(
                f"{path} has no soil_layers_stag dimension; the child's "
                "land-surface scheme requires child-grid soil state")
        missing_attrs = [name for name in _SURFACE_IDENTITY_ATTRS
                         if name not in dataset.ncattrs()]
        if missing_attrs:
            raise OfflineChildContractError(
                f"{path} lacks landuse identity attributes "
                f"{missing_attrs}; category semantics cannot be assumed")
        identity = {
            "MMINLU": str(dataset.getncattr("MMINLU")).strip(),
            "ISWATER": int(dataset.getncattr("ISWATER")),
            "ISLAKE": int(dataset.getncattr("ISLAKE")),
            "ISICE": int(dataset.getncattr("ISICE")),
            "ISOILWATER": int(dataset.getncattr("ISOILWATER"))
            if "ISOILWATER" in dataset.ncattrs() else 14,
        }
        missing = [name for name in _SURFACE_REQUIRED_FIELDS
                   if name not in dataset.variables]
        if missing:
            raise OfflineChildContractError(
                f"{path} lacks required child surface fields {missing}")
        for name in _SURFACE_REQUIRED_FIELDS + _SURFACE_OPTIONAL_FIELDS:
            if name not in dataset.variables:
                continue
            value = np.asarray(dataset.variables[name][:])
            dimensions = list(dataset.variables[name].dimensions)
            if value.ndim and value.shape[0] == 1 and (
                    dimensions[:1] == ["Time"]):
                value = value[0]
                dimensions = dimensions[1:]
            value = np.ascontiguousarray(value, dtype=np.float32)
            _require_finite(
                path, name, value, dimensions,
                remedy="this surface file is damaged or incomplete; make "
                       "it again on the child grid before the child uses it")
            expected_shape = (
                (int(num_soil_layers), int(child_ny), int(child_nx))
                if name in _SURFACE_SOIL_FIELDS
                else (int(child_ny), int(child_nx)))
            if tuple(value.shape) != expected_shape:
                raise OfflineChildContractError(
                    f"{path}/{name} shape {tuple(value.shape)} != "
                    f"{expected_shape}")
            if name in _SURFACE_CATEGORY_FIELDS and not np.array_equal(
                    value, np.rint(value)):
                raise OfflineChildContractError(
                    f"{path}/{name} carries non-integer categories; a "
                    "smoothed/interpolated category field is not a valid "
                    "surface identity")
            fields[name] = value
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = MappingProxyType({
        "path": str(path.resolve()),
        "sha256": digest,
        "identity": dict(identity),
        "carried_fields": tuple(sorted(fields)),
        "policy": "child-grid-surface-source; downscaling replaces "
                  "meteorology, land/soil identity comes from the child's "
                  "own initialization (ndown-equivalent contract)",
    })
    return ChildSurfaceState(
        path=path.resolve(), fields=MappingProxyType(fields),
        identity=MappingProxyType(identity), receipt=receipt)


#: Child-grid surface fields whose value IS a category or a mask, so they
#: are copied from the donor parent cell rather than interpolated: a
#: bilinear land-use index is not a land-use index, and
#: :func:`read_child_surface_state` refuses a non-integer category for
#: exactly that reason.  ``LANDMASK`` rides with them so the child's mask
#: and its land-use come from the SAME parent cell and cannot disagree.
_SURFACE_DONOR_COPY_FIELDS = frozenset(
    {"LU_INDEX", "ISLTYP", "LANDMASK", "SNOWC"})

#: Fields the Registry masks against ``ISICE`` rather than ``ISWATER``
#: (``Registry.EM_COMMON:1417``: ``XICE ... interp_mask_field:lu_index,isice``).
_SURFACE_SEAICE_FIELDS = frozenset({"SEAICE", "XICE"})

#: Not derived from the parent: the child's own latitude/longitude are
#: projection geometry, and the runner already has them exactly -- it
#: SINTs the parent's XLAT/XLONG whenever the surface source carries
#: none (``offline_child_run._initialize_child_physics``).  Interpolating
#: them here would substitute a coarser answer for one already in hand.
_SURFACE_NOT_DERIVED_FIELDS = frozenset({"XLAT", "XLONG"})

#: What the derived child surface IS, in one sentence, for the receipt and
#: for the warning the front door prints.  It is WRF's own answer for a
#: nest that has no ``wrfinput`` of its own (``input_from_file = .false.``,
#: ``med_nest_initial``'s unconditional ``med_interp_domain``,
#: share/mediation_integrate.F:670), run through the Registry-named masked
#: interpolator for the surface/soil family.
DERIVED_CHILD_SURFACE_POLICY = (
    "parent-history-interpolated child surface: land identity and soil "
    "warm start come from the parent's own history frame through WRF's "
    "nest-birth operators -- categories and the landmask by donor-cell "
    "copy, the continuous surface/soil family by interp_mask_field "
    "(Registry.EM_COMMON's masked land interpolator, lu_index/iswater), "
    "which is what WRF does for a nest with input_from_file = .false.  "
    "The child's land identity is therefore its PARENT's, resolved at "
    "the parent's spacing")

#: The fidelity sentence, printed once at the front door.  Named as a
#: cost rather than buried: a child whose coastline is its parent's
#: coastline is a real difference from one built by geogrid at the
#: child's own dx, and a reader of the child's charts must know which
#: they are looking at.
DERIVED_CHILD_SURFACE_CAVEAT = (
    "the child's land-use, soil category and landmask are the PARENT's, "
    "carried down from the parent cell each child column sits in -- "
    "coastlines, lakes and islands the child's spacing could resolve are "
    "not resolved.  Pass --child-surface-from with a child-grid "
    "wrfinput/history file to give the child its own geography instead")


def _donor_copy(field, *, ci, cj):
    """Nearest-donor copy on WRF's masked-interpolator donor cell.

    ``cfld`` is ``(..., ny_parent, nx_parent)``; the result takes the
    value of the coarse cell the child column falls in, so a category or
    a 0/1 mask survives the mapping exactly.
    """
    return np.asarray(field)[..., cj[:, None], ci[None, :]]


def derive_child_surface_from_parent(
        path, *, placement, num_soil_layers: int) -> ChildSurfaceState:
    """Build the child-grid surface state out of the parent's own history.

    THE CLOSED LOOP THIS OPENS (defect #275).  A full-physics child needs
    child-grid land identity and soil state.  Until now the only way to
    supply it was ``--child-surface-from`` pointing at a file on the
    exact child grid, and for a config-driven parent -- the route the
    product steers ERA5 users onto -- no command in the product produced
    one: ``woof run`` writes no ``wrf-native-input/``, ``woof go``
    refuses ``[case_data]`` configs by name, and the prepared-tree route
    needs a front-door manifest the fetch door would only author for one
    source.  So the refusal demanded a file the product could not make,
    and said so only after the parent forecast had been paid for.

    The data was never missing.  The parent's history carries all nine of
    :data:`_SURFACE_REQUIRED_FIELDS` and the landuse identity attributes
    (:data:`woof.io.wrf_output_schema.SURFACE_IDENTITY_OUTPUT_FIELDS`,
    plus the LSM-gated soil family) whenever a land-surface scheme is
    routed -- which a full-physics parent by definition has.  Only the
    GRID was wrong, and putting a parent's surface state on a child grid
    is WRF's own operator, not new science: ``input_from_file = .false.``
    interpolates every field the nest needs from the coarse domain
    (Users' Guide chapter 5; ``med_nest_initial``'s unconditional
    ``med_interp_domain``, share/mediation_integrate.F:670), with the
    surface/soil family going through the Registry's landmask-aware
    ``interp_mask_field`` rather than a plain interpolator.

    WHAT IT COSTS, and why it is a default anyway.  The child inherits
    the parent's land identity at the parent's spacing, which is strictly
    less than a geogrid-built child-grid ``wrfinput`` gives.  It is also
    exactly what a live WRF nest with no input file of its own gets, and
    the product already takes this route for trigger-spawned nests
    (:func:`woof.ingest.nest_spawn_init.spawn_land_state_from_parent`).
    A refusal that names no reachable remedy is not a safeguard; this is
    the reachable remedy, ``--child-surface-from`` stays the
    higher-fidelity one, and the difference is warned about at the front
    door and receipted in the child's report.
    """

    from woof.core.nest_interp import interp_mask_field, mask_donor_index

    path = Path(path)
    child_ny = int(placement.child_ny)
    child_nx = int(placement.child_nx)
    ratio = int(placement.parent_grid_ratio)
    with _ParentHistory(path) as dataset:
        for name, expected in (("south_north", int(placement.parent_ny)),
                               ("west_east", int(placement.parent_nx))):
            actual = len(dataset.dimensions[name]) \
                if name in dataset.dimensions else None
            if actual != expected:
                raise OfflineChildContractError(
                    f"{path} {name}={actual} is not the parent grid "
                    f"{name}={expected} this placement was built against")
        missing_attrs = [name for name in _SURFACE_IDENTITY_ATTRS
                         if name not in dataset.ncattrs()]
        if missing_attrs:
            raise OfflineChildContractError(
                f"{path} lacks landuse identity attributes "
                f"{missing_attrs}, so the child's category semantics "
                "cannot be read off it; " + CHILD_SURFACE_SOURCE_REMEDY)
        identity = {
            "MMINLU": str(dataset.getncattr("MMINLU")).strip(),
            "ISWATER": int(dataset.getncattr("ISWATER")),
            "ISLAKE": int(dataset.getncattr("ISLAKE")),
            "ISICE": int(dataset.getncattr("ISICE")),
            "ISOILWATER": int(dataset.getncattr("ISOILWATER"))
            if "ISOILWATER" in dataset.ncattrs() else 14,
        }
        missing = [name for name in _SURFACE_REQUIRED_FIELDS
                   if name not in dataset.variables]
        if missing:
            # NAMES THE BREAKAGE: without these the child has no land
            # identity or soil state at all, and the remedy is either a
            # parent history that publishes them (the default inventory
            # of any run with a land-surface scheme) or an explicit
            # child-grid file.
            raise OfflineChildContractError(
                f"{path} does not carry the child surface fields "
                f"{missing}, so a child-grid surface state cannot be "
                "derived from it; re-run the parent with a history "
                "selection that keeps the land-surface inventory, or "
                + CHILD_SURFACE_SOURCE_REMEDY)
        if int(num_soil_layers) > 0:
            soil_dim = ("soil_layers_stag" in dataset.dimensions
                        and len(dataset.dimensions["soil_layers_stag"]))
            remap_soil = bool(soil_dim) and parent_soil_onto_child_levels(
                soil_dim, num_soil_layers)
            if soil_dim != int(num_soil_layers) and not remap_soil:
                raise OfflineChildContractError(
                    f"{path} soil_layers_stag={soil_dim} does not match "
                    f"the child LSM's {num_soil_layers} soil layers")
        else:
            remap_soil = False
        parent_zs = (np.asarray(dataset.variables["ZS"][:]).reshape(-1)
                     if remap_soil and "ZS" in dataset.variables else None)
        parent_fields: dict[str, np.ndarray] = {}
        for name in (_SURFACE_REQUIRED_FIELDS + _SURFACE_OPTIONAL_FIELDS):
            if name in _SURFACE_NOT_DERIVED_FIELDS:
                continue
            if name not in dataset.variables:
                continue
            value = np.asarray(dataset.variables[name][:])
            dimensions = list(dataset.variables[name].dimensions)
            if value.ndim and value.shape[0] == 1 and (
                    dimensions[:1] == ["Time"]):
                value = value[0]
                dimensions = dimensions[1:]
            value = np.ascontiguousarray(value, dtype=np.float32)
            _require_finite(path, name, value, dimensions,
                            remedy=_DAMAGED_PARENT_HISTORY)
            parent_fields[name] = value

    ci, _ = mask_donor_index(child_nx, ratio, int(placement.i_parent_start))
    cj, _ = mask_donor_index(child_ny, ratio, int(placement.j_parent_start))
    parent_lu = parent_fields["LU_INDEX"]
    child_lu = _donor_copy(parent_lu, ci=ci, cj=cj)

    fields: dict[str, np.ndarray] = {}
    branch_counts: dict[str, dict[str, int]] = {}
    for name, value in parent_fields.items():
        if name in _SURFACE_DONOR_COPY_FIELDS:
            result = _donor_copy(value, ci=ci, cj=cj)
        else:
            flag = (identity["ISICE"] if name in _SURFACE_SEAICE_FIELDS
                    else identity["ISWATER"])
            result, counts = interp_mask_field(
                value, nri=ratio, nrj=ratio,
                i_parent_start=int(placement.i_parent_start),
                j_parent_start=int(placement.j_parent_start),
                child_landuse=child_lu, parent_landuse=parent_lu,
                flag_category=flag)
            branch_counts[name] = dict(counts)
        result = np.ascontiguousarray(result, dtype=np.float32)
        soil_count = (result.shape[0] if remap_soil
                      else int(num_soil_layers))
        expected_shape = ((soil_count, child_ny, child_nx)
                          if name in _SURFACE_SOIL_FIELDS
                          else (child_ny, child_nx))
        if tuple(result.shape) != expected_shape:
            raise OfflineChildContractError(
                f"{path}/{name} derived shape {tuple(result.shape)} != "
                f"{expected_shape}")
        if name in _SURFACE_CATEGORY_FIELDS and not np.array_equal(
                result, np.rint(result)):
            raise OfflineChildContractError(
                f"{path}/{name} derived non-integer categories; a "
                "smoothed category field is not a valid surface identity")
        if not np.isfinite(result).all():
            raise OfflineChildContractError(
                f"{path}/{name} derived a non-finite child value")
        fields[name] = result
    if remap_soil:
        fields = noah_soil_onto_ruc_levels(
            fields, parent_zs_m=parent_zs, num_soil_layers=num_soil_layers)

    receipt = MappingProxyType({
        **({"soil_levels_rebuilt": noah_soil_onto_ruc_receipt(
            parent_zs, num_soil_layers)} if remap_soil else {}),
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source": "parent-history-interpolated",
        "identity": dict(identity),
        "carried_fields": tuple(sorted(fields)),
        "donor_copied_fields": tuple(
            sorted(set(fields) & _SURFACE_DONOR_COPY_FIELDS)),
        "placement": {
            "parent_grid_ratio": ratio,
            "i_parent_start": int(placement.i_parent_start),
            "j_parent_start": int(placement.j_parent_start),
        },
        "mask_interpolation_branches": {
            name: counts for name, counts in sorted(branch_counts.items())},
        "policy": DERIVED_CHILD_SURFACE_POLICY,
        "caveat": DERIVED_CHILD_SURFACE_CAVEAT,
    })
    return ChildSurfaceState(
        path=path.resolve(), fields=MappingProxyType(fields),
        identity=MappingProxyType(identity), receipt=receipt)


@dataclass(frozen=True)
class OfflineChildPlacement:
    """Fixed child placement inside one archived parent grid.

    Starts retain WRF's one-based namelist semantics.  Construction runs the
    exact SINT stencil coverage gate for mass, x-staggered, and y-staggered
    fields, so an invalid footprint fails before any parent data are read.
    """

    parent_nx: int
    parent_ny: int
    child_nx: int
    child_ny: int
    parent_grid_ratio: int
    i_parent_start: int
    j_parent_start: int

    def __post_init__(self) -> None:
        values = (
            self.parent_nx, self.parent_ny, self.child_nx, self.child_ny,
            self.parent_grid_ratio, self.i_parent_start, self.j_parent_start,
        )
        if any(isinstance(value, bool) or int(value) != value for value in values):
            raise OfflineChildContractError(
                "offline-child placement values must be integers")
        if min(self.parent_nx, self.parent_ny, self.child_nx, self.child_ny) < 1:
            raise OfflineChildContractError(
                "offline-child parent/child extents must be positive")
        if self.parent_grid_ratio < 1:
            raise OfflineChildContractError("parent_grid_ratio must be >= 1")
        # Coverage checks include SINT's +-2 donor stencil.
        for stagger in ("", "x", "y"):
            self.registration(stagger, wrapper="bdy")

    def registration(self, stagger: str, *, wrapper: str):
        return register_nest(
            nri=self.parent_grid_ratio, nrj=self.parent_grid_ratio,
            i_parent_start=self.i_parent_start,
            j_parent_start=self.j_parent_start,
            child_nx=self.child_nx, child_ny=self.child_ny,
            parent_nx=self.parent_nx, parent_ny=self.parent_ny,
            stagger=stagger, wrapper=wrapper,
        )


@dataclass(frozen=True)
class InterpolatedBoundarySnapshot:
    valid_time: datetime
    fields: Mapping[str, np.ndarray]
    receipt: Mapping[str, object]


@dataclass(frozen=True)
class OfflineBoundaryResult:
    boundaries: LateralBoundaries
    frame_receipts: tuple[Mapping[str, object], ...]
    preparation_seconds: float


@dataclass(frozen=True)
class InterpolatedInitialState:
    valid_time: datetime
    fields: Mapping[str, np.ndarray]
    microphysics: Mapping[str, np.ndarray]
    receipt: Mapping[str, object]


def _read_record(dataset, name: str, *, required: bool = True):
    if name not in dataset.variables:
        if required:
            raise OfflineChildContractError(
                f"{Path(dataset.filepath())} is missing required field {name}")
        return None
    value = np.asarray(dataset.variables[name][:])
    if value.ndim and dataset.variables[name].dimensions[:1] == ("Time",):
        if value.shape[0] != 1:
            raise OfflineChildContractError(
                f"{Path(dataset.filepath())}/{name} needs exactly one Time record")
        value = value[0]
    if value.dtype.kind not in "fiu":
        raise OfflineChildContractError(
            f"{Path(dataset.filepath())}/{name} is not numeric")
    value = np.ascontiguousarray(value, dtype=np.float32)
    _require_finite(Path(dataset.filepath()), name, value,
                    [d for d in dataset.variables[name].dimensions
                     if d != "Time"],
                    remedy=_DAMAGED_PARENT_HISTORY)
    return value


#: What a reader does about a parent history field with missing values.
_DAMAGED_PARENT_HISTORY = (
    "this parent history file is damaged and must be restored or "
    "regenerated before a child can be made from it")


def _require_finite(path, name: str, value, dimensions, *, remedy: str) -> None:
    """Refuse a field with a missing or non-finite value, naming the cell.

    A missing value (a declared fill, or the NetCDF default fill a writer
    leaves where it set nothing) arrives from the reader as NaN.  The
    refusal names the file, the variable, how many values and the first
    cell, and ``remedy`` says what to do about the file.
    """

    finite = np.isfinite(value)
    if finite.all():
        return
    bad = int(value.size - np.count_nonzero(finite))
    first = tuple(int(k) for k in np.argwhere(~finite)[0])
    where = ", ".join(f"{d}={k}" for d, k in zip(dimensions, first)) or "the value"
    raise OfflineChildContractError(
        f"{path}/{name} has {bad} missing or "
        f"non-finite value{'s' if bad != 1 else ''} (first at {where}); "
        f"{remedy}")


def _backend_array(value, backend: str):
    if backend == "cpu":
        return np.ascontiguousarray(value, dtype=np.float32)
    if backend != "cuda":
        raise OfflineChildContractError(
            f"offline-child backend must be cpu or cuda, got {backend!r}")
    import cupy as cp
    return cp.asarray(value, dtype=cp.float32)


def _to_host(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value, dtype=np.float32)
    import cupy as cp
    return np.ascontiguousarray(cp.asnumpy(value), dtype=np.float32)


def _transported_source_fields(source_mp_physics: int) -> tuple[str, ...]:
    """Read the admitted parent's online transported scalar inventory."""
    from types import SimpleNamespace
    from woof.core.preflight import nest_field_kinds

    source_mp = int(source_mp_physics)
    if source_mp not in OFFLINE_CHILD_MP_PHYSICS:
        raise OfflineChildContractError(
            _unsupported_parent_clause(source_mp, what="resolved"))
    dynamics = {"u", "v", "w", "t", "ph", "mu"}
    return tuple(name for name in nest_field_kinds(
        SimpleNamespace(moist=True, mp_physics=source_mp)) if name not in dynamics)


def _resolve_source_physics(
        source_mp_physics: int | None,
        physics_binding: ParentPhysicsBinding | None,
        morr_rimed_ice: int | None,
) -> tuple[int, int | None]:
    if physics_binding is not None:
        if (source_mp_physics is not None
                and int(source_mp_physics) != int(physics_binding.mp_physics)):
            raise OfflineChildContractError(
                "declared source mp_physics conflicts with companion binding")
        source_mp_physics = int(physics_binding.mp_physics)
        if physics_binding.morr_rimed_ice is not None:
            if (morr_rimed_ice is not None
                    and int(morr_rimed_ice)
                    != int(physics_binding.morr_rimed_ice)):
                raise OfflineChildContractError(
                    "declared morr_rimed_ice conflicts with companion binding")
            morr_rimed_ice = int(physics_binding.morr_rimed_ice)
    if source_mp_physics is None:
        raise OfflineChildContractError(
            "source physics must be bound from a companion setup record")
    source_mp_physics = int(source_mp_physics)
    if source_mp_physics not in OFFLINE_CHILD_MP_PHYSICS:
        raise OfflineChildContractError(
            _unsupported_parent_clause(source_mp_physics, what="resolved"))
    return source_mp_physics, morr_rimed_ice


# Condensate and vapor mass have the same physical lower bound, but never
# the number-moment absolute floor. Keep original corrupt source data visible.
_POSITIVE_MASS_FIELDS = _MASS_FIELDS + ("qh",)


def _validate_parent_mass_fields(moisture):
    for name in _POSITIVE_MASS_FIELDS:
        if name not in moisture:
            continue
        value = np.asarray(moisture[name])
        if not np.isfinite(value).all() or np.any(value < 0):
            raise OfflineChildContractError(
                f"parent history mass field {name} is non-finite or negative before interpolation")


def _raw_parent_state(dataset, source_mp_physics: int, *, use_theta_m=0):
    raw = {
        name: _read_record(dataset, name)
        for name in (
            "T", "U", "V", "W", "PH", "MU", "MUB", "MAPFAC_M",
            "MAPFAC_U", "MAPFAC_V", "ZNU", "ZNW", "P_TOP",
        )
    }
    if use_theta_m:
        if int(getattr(dataset, "USE_THETA_M", 0)) != 1:
            raise OfflineChildContractError(
                "use_theta_m=1 offline child requires parent archive USE_THETA_M=1 "
                "and native THM; dry history T cannot supply selected thermal interpolation")
        raw["THM"] = _read_record(dataset, "THM")
    moisture = {}
    wrf_mapping = _scheme_wrf_to_state(int(source_mp_physics))
    inverse = {state_name: wrf_name for wrf_name, state_name in wrf_mapping.items()}
    missing = []
    for name in _transported_source_fields(source_mp_physics):
        wrf_name = inverse.get(name)
        if wrf_name is None:
            missing.append(name)
            continue
        moisture[name] = _read_record(dataset, wrf_name)
    if missing:
        raise OfflineChildContractError(
            "history reader has no bound WRF variable mapping for parent "
            f"mp_physics={source_mp_physics} fields {missing}")
    _validate_parent_mass_fields(moisture)
    return raw, moisture


def _vertical_coefficients(raw, dataset):
    znw = np.asarray(raw["ZNW"], dtype=np.float64).reshape(-1)
    znu = np.asarray(raw["ZNU"], dtype=np.float64).reshape(-1)
    if znw.size != znu.size + 1 or not np.all(np.diff(znw) < 0.0):
        raise OfflineChildContractError("parent ZNU/ZNW are not one valid eta grid")
    p_top_values = np.asarray(raw["P_TOP"], dtype=np.float64).reshape(-1)
    if p_top_values.size != 1:
        raise OfflineChildContractError("parent P_TOP must be scalar")
    hybrid_opt = int(getattr(dataset, "HYBRID_OPT", 2))
    etac = float(getattr(dataset, "ETAC", 0.2))
    from woof.core.constants import P0
    coeffs = compute_hybrid_coeffs(
        znw, hybrid_opt, etac, float(P0), float(p_top_values[0]))
    return coeffs, hybrid_opt, etac, float(p_top_values[0])



#: Fields on the child's own ladder are rebuilt, not interpolated: ``PB`` is
#: an exact function of the ladder and ``MUB``, and both geopotentials come
#: from the discrete hydrostatic recurrence.  Everything else is rebinned.
_LADDER_INDEPENDENT_INITIAL_FIELDS = frozenset({
    "MU", "MUB", "HGT", "PSFC", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E",
    "SINALPHA", "COSALPHA", "XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V",
    "XLONG_V",
})


def resolve_child_ladder(child_eta_levels, *, nz: int | None = None):
    """Validate one declared child ladder, or ``None`` for 'inherit'.

    ``None`` is the shipped trajectory: the child takes its parent's ladder
    verbatim and nothing in this module remaps anything.
    """

    if child_eta_levels is None:
        return None
    znw = np.asarray(child_eta_levels, dtype=np.float64).reshape(-1)
    if nz is not None and znw.size != int(nz) + 1:
        raise OfflineChildContractError(
            f"child ladder has {znw.size} interfaces but the child config "
            f"declares nz={nz}, which needs {int(nz) + 1}: the ladder and "
            "the level count have to describe one grid")
    if znw.size < 2 or znw[0] != 1.0 or znw[-1] != 0.0 or not np.all(
            np.diff(znw) < 0.0):
        raise OfflineChildContractError(
            "child eta_levels must decrease strictly from 1.0 at the surface "
            f"to 0.0 at the model top, got {znw[0]!r} .. {znw[-1]!r}: a "
            "non-monotone ladder folds the coordinate and the reference dry "
            "pressure stops decreasing with height")
    return znw


def _mass_edges(znw, mu, hybrid_opt, etac, p_top):
    return dry_mass_edges(np.asarray(znw, dtype=np.float64),
                          hybrid_opt=int(hybrid_opt), etac=float(etac),
                          p_top=float(p_top),
                          mu=np.asarray(mu, dtype=np.float64))


def _remap_geopotential(phi, src_edges, dst_edges):
    """Remap ``alt = dphi/dm`` and rebuild, rather than interpolating PHI.

    PHI is a coordinate quantity, not an extensive one.  Rebuilding it from
    the remapped ``alt`` through the same recurrence
    ``woof/core/grid.py`` :: ``make_base_state`` uses means the child's own
    ``update_diagnostics`` recovers exactly the ``alt`` that was remapped, so
    the state satisfies the dycore's discrete hydrostatic relation instead of
    merely coming close to it.  It also conserves the column's geopotential
    DEPTH exactly, so the child's model top sits where the parent's did.
    """

    phi = np.asarray(phi, dtype=np.float64)
    alt = geopotential_thickness_per_mass(phi, src_edges)
    return rebuild_geopotential(
        phi[0], remap_layer_means(src_edges, alt, dst_edges), dst_edges)


def _remap_initial_state_to_child_ladder(
        fields, source_mixing, *, parent_znw, child_znw, hybrid_opt, etac,
        p_top):
    """Move one horizontally-SINTed parent state onto the child's ladder.

    Runs once, on the host, in float64, after the horizontal interpolation
    and before anything is uploaded.  The integration loop never sees it.

    Total fields are remapped and the perturbations re-derived against the
    NEW base state: a perturbation is bookkeeping relative to a base that is
    itself changing here, so remapping one directly would carry the parent's
    base into the child's.
    """

    receipts = []
    host = {name: np.asarray(value, dtype=np.float64)
            for name, value in fields.items()}
    mub = host["MUB"]
    mu_total = mub + host["MU"]
    hyc = compute_hybrid_coeffs(np.asarray(child_znw, dtype=np.float64),
                               int(hybrid_opt), float(etac), float(c.P0),
                               float(p_top))

    base_src = _mass_edges(parent_znw, mub, hybrid_opt, etac, p_top)
    base_dst = _mass_edges(child_znw, mub, hybrid_opt, etac, p_top)
    tot_src = _mass_edges(parent_znw, mu_total, hybrid_opt, etac, p_top)
    tot_dst = _mass_edges(child_znw, mu_total, hybrid_opt, etac, p_top)

    out = {name: value for name, value in fields.items()
           if name in _LADDER_INDEPENDENT_INITIAL_FIELDS}

    # PB is an exact function of the ladder and MUB -- the same expression
    # make_base_state evaluates -- so it is RECOMPUTED, never rebinned.
    out["PB"] = (hyc["c3h"][:, None, None] * mub[None]
                 + hyc["c4h"][:, None, None] + float(p_top))
    out["PHB"] = _remap_geopotential(host["PHB"], base_src, base_dst)

    # Total geopotential, then the perturbation against the NEW base.
    phi_total = _remap_geopotential(host["PHB"] + host["PH"], tot_src, tot_dst)
    out["PH"] = phi_total - out["PHB"]

    # Total potential temperature, then the perturbation against 300 K (the
    # child's own base theta is derived downstream from the new PB/PHB).
    thermal = "THM" if "THM" in host else "T"
    theta = remap_layer_means(tot_src, host[thermal] + 300.0, tot_dst)
    receipts.append(remap_receipt("theta", tot_src, host[thermal] + 300.0,
                                  tot_dst, theta))
    out[thermal] = theta - 300.0

    out["W"] = remap_interface_values(tot_src, host["W"], tot_dst)
    if "P" in host:
        out["P"] = remap_layer_means(tot_src, host["P"], tot_dst)

    # Momentum carries the mass at its own faces, the same convention
    # _couple_parent uses for the boundary route.
    for name, faces in (("U", mu_at_u_faces), ("V", mu_at_v_faces)):
        if name not in host:
            continue
        mu_face = _edge_pinned(np.asarray(faces(mu_total), dtype=np.float64),
                               mu_total, axis=1 if name == "U" else 0)
        src = _mass_edges(parent_znw, mu_face, hybrid_opt, etac, p_top)
        dst = _mass_edges(child_znw, mu_face, hybrid_opt, etac, p_top)
        out[name] = remap_layer_means(src, host[name], dst)

    child_mixing = {}
    for name, value in source_mixing.items():
        array = np.asarray(value, dtype=np.float64)
        remapped = remap_layer_means(tot_src, array, tot_dst)
        receipts.append(remap_receipt(name, tot_src, array, tot_dst, remapped))
        child_mixing[name] = remapped

    if thermal == "THM":
        out["T"] = theta / (1.0 + float(c.RVOVRD) * child_mixing["qv"]) - 300.0
    znu = 0.5 * (np.asarray(child_znw)[:-1] + np.asarray(child_znw)[1:])
    out["ZNW"] = np.asarray(child_znw, dtype=np.float64)
    out["ZNU"] = znu
    out["P_TOP"] = np.asarray([float(p_top)], dtype=np.float64)

    # Anything not named above rides through UNCHANGED, which is correct only
    # for a field that does not live on the ladder.  A field that does would
    # otherwise reach the child at the PARENT's level count inside a dict
    # whose other members are on the child's -- a mixed-nz state that the
    # shape checks downstream would not all catch.  Refused by name instead:
    # a 3-D field added to _INITIAL_CORE_FIELDS later has to be given a
    # weight here, and the refusal says so.
    parent_levels = {int(np.asarray(parent_znw).size),
                     int(np.asarray(parent_znw).size) - 1}
    for name, value in fields.items():
        if name in out:
            continue
        array = np.asarray(value)
        if array.ndim >= 3 and int(array.shape[0]) in parent_levels:
            raise OfflineChildContractError(
                f"{name} has {array.shape[0]} levels on the parent's ladder "
                "and no remap weight in "
                "_remap_initial_state_to_child_ladder, so a child on its own "
                "ladder would receive it at the parent's level count while "
                "every other field arrived at the child's.  Give it a weight "
                "(mass for a layer field, interface for a staggered one) or "
                "add it to _LADDER_INDEPENDENT_INITIAL_FIELDS if it does not "
                "live on the ladder.")
        out[name] = value
    return out, child_mixing, receipts

_INITIAL_FIELD_STAGGER = MappingProxyType({
    "U": "x", "V": "y", "MAPFAC_U": "x", "MAPFAC_V": "y",
    "XLAT_U": "x", "XLONG_U": "x", "XLAT_V": "y", "XLONG_V": "y",
})
_INITIAL_CORE_FIELDS = (
    "T", "U", "V", "W", "PH", "MU", "PHB", "MUB", "HGT", "P", "PB",
    "PSFC", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E",
    "SINALPHA", "COSALPHA", "XLAT", "XLONG",
)


def interpolate_parent_initial_state(
        path: str | Path, placement: OfflineChildPlacement, *,
        source_mp_physics: int | None = None,
        physics_binding: ParentPhysicsBinding | None = None,
        target_mp_physics: int | None = None,
        morr_rimed_ice: int | None = None, backend: str = "cpu",
        child_eta_levels=None, child_cfg=None, geography=None,
) -> InterpolatedInitialState:
    """SINT one archived parent state into a standalone child cold start.

    With ``geography`` ``None`` the child inherits the SINT parent terrain
    and base state, as every child did before engine item E3, and the
    receipt says ``sint-parent-inherited``.  With a
    :class:`woof.offline_child_geography.ChildGeography`, the interpolated
    state is moved onto the child's OWN terrain the way WRF's ``ndown``
    moves it (:func:`_onto_child_terrain`): the two terrains are blended
    across the child's boundary zone (``blend_terrain``) and the state is
    rebalanced onto the blend (ndown's ``rebalance``), after the vertical
    remap onto the child's own ladder when it declares one, which is
    ndown's order (``vertical_interp`` before ``rebalance_driver``).

    A child of a DIFFERENT microphysics scheme is converted on the parent's
    own grid first, by the online nest edge's contract and kernel
    (:func:`_convert_parent_microphysics`), and the target scheme's fields
    are what gets interpolated -- the live lane's order.  ``child_cfg`` is
    the child's RunConfig when the caller has one; it supplies the
    switches the contract reads (rimed-category options, WDM6's CCN seed).
    """

    source_mp_physics, morr_rimed_ice = _resolve_source_physics(
        source_mp_physics, physics_binding, morr_rimed_ice)
    backend = str(backend).strip().lower()
    target_mp = int(source_mp_physics if target_mp_physics is None
                    else target_mp_physics)
    transition = None
    if target_mp != int(source_mp_physics):
        transition = _offline_transition_contract(
            int(source_mp_physics), target_mp,
            morr_rimed_ice=morr_rimed_ice,
            hail_opt=(None if physics_binding is None
                      else physics_binding.hail_opt),
            child_cfg=child_cfg)
    hypsometric_opt = int(getattr(child_cfg, "hypsometric_opt", 2))
    info = inspect_parent_history_frame(path, source_mp_physics=source_mp_physics)
    expected = (placement.parent_ny, placement.parent_nx)
    actual = (info.dimensions["south_north"], info.dimensions["west_east"])
    if actual != expected:
        raise OfflineChildContractError(
            f"parent history mass grid {actual} != placement parent grid {expected}")
    started = time.perf_counter()
    use_theta_m = int(getattr(child_cfg, "use_theta_m", 0))
    initial_fields = _INITIAL_CORE_FIELDS + (("THM",) if use_theta_m else ())
    if int(source_mp_physics) == 28:
        # Read them as REQUIRED.  A missing QNWFA2D is not a stream a child
        # may be silently built from: nothing downstream re-derives it (see
        # _AEROSOL_SURFACE_EMISSION_WRF), so tolerating the absence would
        # produce a finite, bounded, aerosol-emission-free forecast.
        initial_fields = initial_fields + _AEROSOL_SURFACE_EMISSION_WRF
    with _ParentHistory(path) as dataset:
        raw, moisture = _raw_parent_state(dataset, int(source_mp_physics),
                                          use_theta_m=getattr(child_cfg, "use_theta_m", 0))
        for name in initial_fields:
            if name not in raw:
                raw[name] = _read_record(dataset, name)
        coeffs, hybrid_opt, etac, p_top = _vertical_coefficients(raw, dataset)
    registrations = {
        "": placement.registration("", wrapper="interp"),
        "x": placement.registration("x", wrapper="interp"),
        "y": placement.registration("y", wrapper="interp"),
    }
    constants = {"ZNU", "ZNW", "P_TOP"}
    interpolated = {}
    for name in initial_fields:
        value = _backend_array(raw[name], backend)
        interpolated[name] = sint(
            value, registrations[_INITIAL_FIELD_STAGGER.get(name, "")])
    conversion_receipt = None
    donor = moisture
    if transition is not None:
        # Diagnose the target scheme on the PARENT, then interpolate: the
        # converted fields replace the parent's own species as the donor
        # of everything below, so the clamp's reference peaks are the
        # converted fields' own.
        converted = _convert_parent_microphysics(
            transition,
            _parent_transition_donor(
                raw, moisture, coeffs, p_top=p_top,
                hypsometric_opt=hypsometric_opt, target_mp_physics=target_mp),
            coupled=False, backend=backend, hypsometric_opt=hypsometric_opt)
        conversion_receipt = converted.receipt()
        donor = converted.fields
        from woof.ingest.handover_numbers import (
            MORRISON_HANDOVER_NUMBERS, closure_active)
        if target_mp == 10 and closure_active(child_cfg):
            # The edge kernel enters Morrison at its slope limit (the N = 0
            # Morrison would size by itself: morrison_edge_number), which is
            # the very defect woof/ingest/handover_numbers.py closes.  On
            # this offline route the numbers are diagnosed from the
            # converted masses instead, on the parent's own grid.
            donor = {name: np.array(_to_host(value), dtype=np.float32)
                     for name, value in donor.items()}
            for _, number in MORRISON_HANDOVER_NUMBERS:
                if number in donor:
                    donor[number][...] = 0.0
            close_child_numbers(10, donor, raw, morr_rimed_ice=morr_rimed_ice,
                                cfg=child_cfg)
    source_mixing = {
        name: sint(_backend_array(value, backend), registrations[""])
        for name, value in donor.items()
    }
    # The number moments carry magnitudes of 1e3..1e9 per kilogram, so a
    # float32 SINT can round one across zero.  Applied BEFORE any consumer:
    # the engine's radiation gate refuses a negative nr at the first
    # radiative call.  That is correct; the artefact is what has to go.
    initial_clamp = clamp_sint_undershoot_mapping(source_mixing)
    # Actual nonnegative archived cloud water can land at -6e-22 kg/kg
    # after SINT. Reuse the bounded eight-ULP policy with the DONOR scale,
    # and no absolute floor; larger negatives still reach the strict gate.
    initial_clamp.update(clamp_sint_undershoot_mapping(
        source_mixing, names=_POSITIVE_MASS_FIELDS, floor_scale=0.0,
        reference_fields={name: _to_host(value)
                          for name, value in donor.items()}))
    fields = {name: _to_host(value) for name, value in interpolated.items()}
    fields.update({name: np.array(raw[name], copy=True, dtype=np.float32)
                   for name in constants})
    if use_theta_m:
        fields["T"] = ((fields["THM"] + np.float32(300.0))
                       / (np.float32(1.0) + np.float32(c.RVOVRD)
                          * _to_host(source_mixing["qv"])) - np.float32(300.0))
    # The child's own ladder, when it declares one.  A child that declares
    # nothing never reaches this branch and its state is bitwise what it was
    # before per-domain ladders existed.
    child_znw = resolve_child_ladder(child_eta_levels)
    remap_receipts = ()
    if child_znw is not None:
        parent_znw = np.asarray(raw["ZNW"], dtype=np.float64).reshape(-1)
        fields, source_mixing, remap_receipts = (
            _remap_initial_state_to_child_ladder(
                fields, {name: _to_host(value)
                         for name, value in source_mixing.items()},
                parent_znw=parent_znw, child_znw=child_znw,
                hybrid_opt=hybrid_opt, etac=etac, p_top=p_top))
        # PHB stays float64.  The remap above produced the child's base
        # geopotential in float64 on a ladder the parent never had, and
        # DomainState.set_base_geopotential subtracts the float32 store
        # from it to build dphb_resid -- the correction that stops the
        # diagnosed pressure degrading as 1/dz.  Rounding it here made
        # that subtraction identically zero, so the FP32 EOS remedy was
        # OFF on precisely the deep-column route it exists for, while
        # being on everywhere else.  Every other consumer casts to
        # float32 explicitly at its own use (``assign`` below), so this
        # widens nothing downstream.  On the NO-ladder route this block
        # does not run at all and PHB arrives float32 from the archive,
        # where the information genuinely does not exist.
        fields = {name: value if name == "PHB"
                  else np.asarray(value, dtype=np.float32)
                  for name, value in fields.items()}
        fields["PHB"] = np.ascontiguousarray(fields["PHB"], dtype=np.float64)
    terrain_receipt = None
    if geography is not None:
        fields, terrain_receipt = _onto_child_terrain(
            fields, source_mixing, geography=geography, child_cfg=child_cfg,
            hybrid_opt=hybrid_opt, etac=etac, p_top=p_top)
    # Every child number beside a mass, last, on the child's own state: the
    # parent's numbers ride the interpolation, and what is still at zero
    # beside a positive mass (a moment the parent's scheme does not carry,
    # or a SINT clamp) is diagnosed (woof/ingest/handover_numbers.py).
    # This state is also every own-terrain boundary frame
    # (_own_geography_boundary_snapshot), so the ring gets the same numbers.
    # Exact mode (correct-mode key hydrometeor_handover_numbers off) hands
    # back the interpolated species untouched, as before the closure.
    from woof.ingest.handover_numbers import closure_active
    number_closure = None
    if closure_active(child_cfg):
        source_mixing = {name: np.array(_to_host(value), dtype=np.float32)
                         for name, value in source_mixing.items()}
        number_closure = close_child_numbers(
            target_mp, source_mixing, fields,
            morr_rimed_ice=(morr_rimed_ice if morr_rimed_ice is not None
                            else getattr(child_cfg, "morr_rimed_ice", 1)),
            cfg=child_cfg)
    else:
        source_mixing = {name: _to_host(value)
                         for name, value in source_mixing.items()}
    if use_theta_m:
        # Derive history/closure theta from the exact FP32 fields the child
        # stores, after any host FP64 vertical remap and terrain rebalance.
        vapour = np.asarray(source_mixing["qv"], dtype=np.float32)
        fields["T"] = ((np.asarray(fields["THM"], dtype=np.float32) + np.float32(300.0))
                       / (np.float32(1.0) + np.float32(c.RVOVRD) * vapour)
                       - np.float32(300.0))
    receipt = MappingProxyType({
        "path": str(Path(path).resolve()),
        "valid_time": info.valid_time.isoformat(),
        "geometry_sha256": info.geometry_sha256,
        "source_mp_physics": int(source_mp_physics),
        "advisory_inferred_mp_physics": info.inferred_mp_physics,
        "source_physics_binding": (
            None if physics_binding is None else dict(physics_binding.receipt())),
        "target_mp_physics": target_mp,
        "backend": backend,
        **({"use_theta_m": 1} if use_theta_m else {}),
        "positive_definite_clamp": initial_clamp,
        "terrain_policy": ("sint-parent-inherited" if terrain_receipt is None
                           else terrain_receipt["policy"]),
        "spinup_policy": (
            "new-standalone-child; source held physics tendencies and "
            "scheduler state are not inherited"),
        "hybrid_opt": hybrid_opt,
        "etac": etac,
        "p_top": p_top,
        # None when the child inherited its parent's ladder.  Otherwise the
        # measured conservation of every field that was rebinned: a remap
        # that quietly failed to conserve must not look like one that did.
        "vertical_remap": None if child_znw is None else {
            "source_levels": int(np.asarray(raw["ZNW"]).size - 1),
            "target_levels": int(child_znw.size - 1),
            "child_eta_levels": tuple(float(v) for v in child_znw),
            "fields": tuple(item.summary() for item in remap_receipts),
            "max_relative_drift": max(
                (item.max_relative_drift for item in remap_receipts),
                default=0.0),
        },
        "conversion": None if conversion_receipt is None else dict(conversion_receipt),
        "number_closure": number_closure,
        "seconds": time.perf_counter() - started,
        **({} if terrain_receipt is None else {"terrain": terrain_receipt}),
    })
    return InterpolatedInitialState(
        info.valid_time, MappingProxyType(fields),
        MappingProxyType(source_mixing), receipt)


def close_child_numbers(target_mp, mixing, fields, *, morr_rimed_ice=1, cfg):
    """Diagnose the child's numbers that sit at zero beside a positive mass.

    ``mixing`` holds the child's uncoupled species (per kg), changed in
    place; ``fields`` the child's T (perturbation theta), P and PB on the
    same grid; ``cfg`` the child's RunConfig.  Returns the closure receipt,
    or None when the scheme closes nothing (single-moment, or the closure off:
    exact mode or a strict WRF process).
    """

    from woof.ingest import handover_numbers as hn

    if not hn.closure_active(cfg):
        return None
    pairs = hn.handover_number_pairs(int(target_mp), cfg)
    if not pairs:
        return None
    masses = {m: mixing[m] for m, _ in pairs if m in mixing}
    numbers = {n: mixing[n] for _, n in pairs if n in mixing}
    shape = next(iter(masses.values())).shape if masses else None

    def temperature(idx):
        theta = hn.take(fields["T"], idx, shape).astype(np.float64) + 300.0
        pressure = (hn.take(fields["P"], idx, shape).astype(np.float64)
                    + hn.take(fields["PB"], idx, shape).astype(np.float64))
        return hn.temperature_from_theta(theta, pressure)

    def density(idx):
        pressure = (hn.take(fields["P"], idx, shape).astype(np.float64)
                    + hn.take(fields["PB"], idx, shape).astype(np.float64))
        return hn.density_from_state(pressure, temperature(idx),
                                     hn.take(mixing["qv"], idx, shape))

    xland = fields.get("XLAND")
    return hn.close_numbers(
        int(target_mp), masses, numbers, rho=density, temperature=temperature,
        rimed=int(morr_rimed_ice if morr_rimed_ice is not None else 1),
        nwfa=mixing.get("nwfa"),
        xland=None if xland is None else np.asarray(xland)[None],
        bound_thompson=True)


def _onto_child_terrain(fields, source_mixing, *, geography, child_cfg,
                        hybrid_opt, etac, p_top):
    """Move one interpolated child state onto the child's own terrain.

    WRF ndown's two steps, in its order (main/ndown_em.F:586-710): the
    parent's interpolated terrain and the child's own are blended across the
    child's boundary zone (``blend_terrain``), and the state is rebalanced
    onto the blend (``rebalance``,
    :func:`woof.offline_child_geography.rebalance_to_child_terrain`).
    Every boundary frame takes the same two steps
    (:func:`_own_geography_boundary_snapshot`), so the boundary strips are
    made on exactly the terrain under the rows they are applied to.

    The base state comes back in float64 (``MUB``, ``PB``, ``PHB``): the
    child's base on its own terrain is analytic, and
    :func:`_base_from_interpolated_initial` re-derives the inverse density
    and base potential temperature from those three, so rounding them here
    would put a float32 step between the base the state was balanced on
    and the base the child integrates on.
    """

    from woof.offline_child_geography import (
        OWN_TERRAIN_POLICY, WRF_BLEND_WIDTH, blend_child_terrain,
        rebalance_to_child_terrain)

    if child_cfg is None:
        raise OfflineChildContractError(
            "a child on its own terrain is rebalanced with its own config "
            "(spec_bdy_width, base_temp, hypsometric_opt); pass child_cfg")
    if "qv" not in source_mixing:
        raise OfflineChildContractError(
            "a child on its own terrain is rebalanced with its vapour, and "
            "the interpolated state carries no qv")
    znw = np.asarray(fields["ZNW"], dtype=np.float64).reshape(-1)
    coord = make_vertical_coord(znw.size - 1, hybrid_opt=int(hybrid_opt),
                                etac=float(etac), eta_levels=znw)
    finalize_vertical_coord(coord, float(p_top))
    terrain_interpolated = np.asarray(fields["HGT"], dtype=np.float64)
    spec_bdy_width = int(child_cfg.spec_bdy_width)
    terrain_child = blend_child_terrain(
        terrain_interpolated, geography.terrain,
        spec_bdy_width=spec_bdy_width, blend_width=WRF_BLEND_WIDTH)
    columns = rebalance_to_child_terrain(
        theta=fields["T"], mu=fields["MU"],
        qv=_to_host(source_mixing["qv"]), coord=coord, p_top=float(p_top),
        terrain_interpolated=terrain_interpolated,
        terrain_child=terrain_child, base_temp=float(child_cfg.base_temp),
        hypsometric_opt=int(child_cfg.hypsometric_opt))
    out = dict(fields)
    base = columns.base
    out.update({
        "HGT": np.asarray(terrain_child, dtype=np.float32),
        "MUB": np.ascontiguousarray(base.mub, dtype=np.float64),
        "PB": np.ascontiguousarray(base.pb, dtype=np.float64),
        "PHB": np.ascontiguousarray(base.phb, dtype=np.float64),
        "T": np.asarray(columns.theta, dtype=np.float32),
        "PH": np.asarray(columns.ph, dtype=np.float32),
        "P": np.asarray(columns.p, dtype=np.float32),
        "PSFC": np.asarray(columns.psfc, dtype=np.float32),
    })
    if getattr(child_cfg, "use_theta_m", 0):
        out["THM"] = ((out["T"] + np.float32(300.0))
                      * (np.float32(1.0) + np.float32(c.RVOVRD)
                         * np.asarray(_to_host(source_mixing["qv"]), dtype=np.float32))
                      - np.float32(300.0))
    change = terrain_child - terrain_interpolated
    receipt = {
        "policy": OWN_TERRAIN_POLICY,
        "method": "WRF v4.7.1 ndown: blend_terrain (dyn_em/"
                  "nest_init_utils.F:712-785) then rebalance (dyn_em/"
                  "module_initialize_real.F:4982-5266), every frame",
        "blend": {"spec_bdy_width": spec_bdy_width,
                  "blend_width": WRF_BLEND_WIDTH},
        "terrain_change_m": {
            "max_rise": float(change.max()), "max_drop": float(-change.min()),
            "mean_abs": float(np.mean(np.abs(change)))},
        "terrain_m": {"interpolated_max": float(terrain_interpolated.max()),
                      "child_max": float(terrain_child.max())},
        "theta_shift_k": {
            "lowest_level_min": float(columns.theta_shift[0].min()),
            "lowest_level_max": float(columns.theta_shift[0].max())},
        "geography_sha256": str(geography.receipt.get("terrain_sha256")),
    }
    return out, receipt


def _child_base_geopotential(phb_parent, mub, *, parent_znw, child_znw,
                             hybrid_opt, etac, p_top):
    """The child ladder's base geopotential, from the parent's own PHB.

    Shared by the initial-state and lateral-boundary routes so the boundary
    strips are relative to exactly the base state the domain was built on; a
    second, subtly different reconstruction here would put a step in the
    geopotential at the edge of the relaxation zone.
    """

    return _remap_geopotential(
        np.asarray(phb_parent, dtype=np.float64),
        _mass_edges(parent_znw, mub, hybrid_opt, etac, p_top),
        _mass_edges(child_znw, mub, hybrid_opt, etac, p_top))


def _remap_boundary_snapshot_to_child_ladder(
        interpolated, *, child_mub, child_phb, parent_znw, child_znw,
        hybrid_opt, etac, p_top, moisture_names):
    """Move one SINTed, COUPLED boundary strip onto the child's ladder.

    The strips arrive coupled by the parent ladder's ``chm``/``chf``
    (:func:`_couple_parent`).  Coupling is a per-layer mass weight, so a
    coupled field is not rebinnable as it stands: it is uncoupled on the
    child with the parent ladder's weight, remapped, and recoupled with the
    child ladder's.  Uncoupling against a weight built from the child's own
    ``mu`` is the convention this module already uses for the cross-physics
    boundary edge, not a second one invented here.
    """

    receipts = []
    # ``interpolated`` arrives on the PREPROCESS BACKEND: on the default
    # cuda route every value in it is a device array, which is why the
    # field loop below reads each one through ``_to_host``.  ``mu`` is
    # read here instead of there, and reading it with ``np.asarray``
    # made `woof downscale --child-levels` -- the whole reason this
    # function exists -- die on its first boundary frame with CuPy's
    # "Implicit conversion to a NumPy array is not allowed", at exit 1,
    # on the default backend.
    child_mu = np.asarray(child_mub, dtype=np.float64) + np.asarray(
        _to_host(interpolated["mu"])[0], dtype=np.float64)

    def coeffs_for(znw):
        return compute_hybrid_coeffs(np.asarray(znw, dtype=np.float64),
                                     int(hybrid_opt), float(etac),
                                     float(c.P0), float(p_top))

    src_c, dst_c = coeffs_for(parent_znw), coeffs_for(child_znw)
    faces = {
        "": child_mu,
        "x": _edge_pinned(np.asarray(mu_at_u_faces(child_mu),
                                     dtype=np.float64), child_mu, axis=1),
        "y": _edge_pinned(np.asarray(mu_at_v_faces(child_mu),
                                     dtype=np.float64), child_mu, axis=0),
    }
    stagger = {"u": "x", "v": "y"}
    edges = {key: (_mass_edges(parent_znw, value, hybrid_opt, etac, p_top),
                   _mass_edges(child_znw, value, hybrid_opt, etac, p_top))
             for key, value in faces.items()}

    def half_weight(co, mu_face):
        return (co["c1h"][:, None, None] * mu_face[None]
                + co["c2h"][:, None, None])

    def full_weight(co, mu_face):
        return (co["c1f"][:, None, None] * mu_face[None]
                + co["c2f"][:, None, None])

    out = {"mu": interpolated["mu"]}
    for name, value in interpolated.items():
        if name == "mu":
            continue
        key = stagger.get(name, "")
        mu_face = faces[key]
        src_edges, dst_edges = edges[key]
        array = np.asarray(_to_host(value), dtype=np.float64)
        if name in ("w", "phi"):
            plain = array / full_weight(src_c, mu_face)
            if name == "phi":
                # Total geopotential, remapped through alt = dphi/dm, then
                # made a perturbation against the CHILD's base again.
                total = _remap_geopotential(
                    np.asarray(child_phb, dtype=np.float64) + plain,
                    src_edges, dst_edges)
                child_base = _child_base_geopotential(
                    child_phb, np.asarray(child_mub, dtype=np.float64),
                    parent_znw=parent_znw, child_znw=child_znw,
                    hybrid_opt=hybrid_opt, etac=etac, p_top=p_top)
                remapped = total - child_base
            else:
                remapped = remap_interface_values(src_edges, plain, dst_edges)
            out[name] = remapped * full_weight(dst_c, mu_face)
            continue
        plain = array / half_weight(src_c, mu_face)
        remapped = remap_layer_means(src_edges, plain, dst_edges)
        if name in moisture_names or name == "theta":
            receipts.append(
                remap_receipt(name, src_edges, plain, dst_edges, remapped))
        out[name] = remapped * half_weight(dst_c, mu_face)
    return out, receipts


def _edge_pinned(face, centre, *, axis):
    """``mu`` at a staggered face, with the outer faces pinned to the centre.

    The same convention :func:`_couple_parent` uses; kept in one place so the
    two routes cannot drift apart.
    """

    face = np.array(face, dtype=np.float64, copy=True)
    if axis == 1:
        face[:, 0] = centre[:, 0]
        face[:, -1] = centre[:, -1]
    else:
        face[0, :] = centre[0, :]
        face[-1, :] = centre[-1, :]
    return face

def _base_from_interpolated_initial(initial: InterpolatedInitialState,
                                    cfg):
    fields = initial.fields
    znw = np.asarray(fields["ZNW"], dtype=np.float64).reshape(-1)
    coord = make_vertical_coord(
        cfg.nz, hybrid_opt=int(initial.receipt["hybrid_opt"]),
        etac=float(initial.receipt["etac"]), eta_levels=znw)
    p_top = float(np.asarray(fields["P_TOP"]).reshape(-1)[0])
    finalize_vertical_coord(coord, p_top)
    mub = np.asarray(fields["MUB"], dtype=np.float64)
    pb = np.asarray(fields["PB"], dtype=np.float64)
    phb = np.asarray(fields["PHB"], dtype=np.float64)
    if pb.shape != (cfg.nz, cfg.ny, cfg.nx):
        raise OfflineChildContractError(
            f"interpolated PB shape {pb.shape} does not match child")
    if phb.shape != (cfg.nz + 1, cfg.ny, cfg.nx):
        raise OfflineChildContractError(
            f"interpolated PHB shape {phb.shape} does not match child")
    delta_phi = phb[1:] - phb[:-1]
    if cfg.hypsometric_opt == 1:
        denominator = (-coord.dnw[:, None, None]
                       * (coord.c1h[:, None, None] * mub[None]
                          + coord.c2h[:, None, None]))
    elif cfg.hypsometric_opt == 2:
        pfu = (coord.c3f[1:, None, None] * mub[None]
               + coord.c4f[1:, None, None] + p_top)
        pfd = (coord.c3f[:-1, None, None] * mub[None]
               + coord.c4f[:-1, None, None] + p_top)
        phm = (coord.c3h[:, None, None] * mub[None]
               + coord.c4h[:, None, None] + p_top)
        denominator = phm * pm.log(pfd / pfu)
    else:
        raise OfflineChildContractError(
            f"unsupported hypsometric_opt={cfg.hypsometric_opt}")
    if np.any(denominator <= 0.0):
        raise OfflineChildContractError(
            "interpolated parent base has non-positive hydrostatic increments")
    alb = delta_phi / denominator
    if not np.isfinite(alb).all() or np.any(alb <= 0.0):
        raise OfflineChildContractError(
            "interpolated parent PHB/PB imply invalid base inverse density")
    from woof.core import constants as c
    thb = alb * pb / (c.RD * pm.power(pb / c.P0, c.RCP))
    if not np.isfinite(thb).all() or np.any(thb <= 0.0):
        raise OfflineChildContractError(
            "interpolated parent base implies invalid base potential temperature")
    return coord, BaseState(
        mub=mub, p_top=p_top, pb=pb, alb=alb, thb=thb, phb=phb,
        terrain_z=np.asarray(fields["HGT"], dtype=np.float64))


def _require_prepared_child_ladder(initial, cfg) -> None:
    """The prepared state and the child config must name ONE ladder.

    A prepared state carries the ladder it was remapped onto.  If the config
    handed to this function names a different one, the arrays would be built
    against a coordinate the state was never interpolated to -- the level
    counts might even agree while the interfaces sit elsewhere, so the shape
    checks downstream would pass and the child would integrate a state whose
    layers are not where its coordinate says they are.
    """

    prepared = initial.receipt.get("vertical_remap")
    declared = None if cfg.eta_levels is None else tuple(
        float(value) for value in cfg.eta_levels)
    if prepared is None:
        if declared is not None:
            raise OfflineChildContractError(
                f"child config declares its own {len(declared) - 1}-level "
                "eta ladder but the prepared state was built on the parent's "
                "ladder: pass child_eta_levels to "
                "interpolate_parent_initial_state so the state is remapped "
                "onto the ladder the child will integrate on")
        return
    if declared is None:
        raise OfflineChildContractError(
            f"prepared state was remapped onto a "
            f"{prepared['target_levels']}-level child ladder but the child "
            "config declares no eta_levels: the config has to carry the "
            "ladder the state was built for")
    if declared != tuple(prepared["child_eta_levels"]):
        raise OfflineChildContractError(
            f"child config eta_levels ({len(declared) - 1} levels) is not "
            f"the ladder the state was prepared on "
            f"({prepared['target_levels']} levels): the state would be "
            "loaded against a coordinate it was never remapped to")


def require_runnable_child_radiation(cfg, p_top: float):
    """Refuse a child ladder this domain's own radiation cannot run.  Returns ``cfg``.

    ``RunConfig`` carries no model-top pressure, so
    ``validate_run_config`` reaches
    ``validate_resolved_physics_vertical_levels`` with ``p_top=None`` and the
    radiation cap-layer arithmetic is skipped (woof/physics_compat.py, the
    "RunConfig-only checks" branch).  That was unreachable while a child's nz
    was pinned to its parent's and real parents run ~50 levels; a child that
    may now name its own deeper ladder can walk straight into it, and the run
    would die at the FIRST radiative call -- after the fetch, the SINT, the
    remap and the whole preparation had been paid for.  The parent archive
    knows the model top, so the check runs with it.

    ONE function, three doors, like the root-forcing rule beside it:
    ``woof downscale``'s plan review and the runner's admission both reach
    it through :func:`require_runnable_child_radiation_from_archive`, which
    reads the model top off the parent tape before anything is fetched or
    interpolated, and the state builder below calls it again with the
    interpolation's own receipt.  Until plan review asked, a ladder this
    deep planned clean on ``--dry-run`` and the run died at the first
    radiative call, after the fetch, the SINT, the remap and the whole
    preparation had been paid for.
    """

    from woof.physics_compat import (
        PhysicsVerticalPreflightError,
        validate_resolved_physics_vertical_levels,
    )

    try:
        validate_resolved_physics_vertical_levels(cfg, p_top=float(p_top))
    except PhysicsVerticalPreflightError as exc:
        raise OfflineChildContractError(
            f"child nz={cfg.nz} at the parent's p_top={float(p_top):g} Pa "
            f"exceeds a radiation adapter's layer ceiling: {exc}") from exc
    return cfg


def parent_archive_p_top(parent_frame):
    """The archived parent's model-top pressure, or ``None``.

    Read straight off the tape, from the same ``P_TOP`` variable the
    interpolation's vertical coefficients are built from
    (:func:`_vertical_coefficients`), so plan review and the state builder
    are asking about one number.  ``None`` when the frame does not carry
    it or carries more than one value: the later door still holds the rule
    with the receipt's own figure, and a review that cannot read the model
    top must not invent one.
    """

    if parent_frame is None:
        return None
    try:
        # Through the history wrapper, so a frame the reader refuses is a
        # sentence naming it rather than the decoder's traceback.
        with _ParentHistory(parent_frame) as dataset:
            variable = dataset.variables.get("P_TOP")
            if variable is None:
                return None
            values = np.asarray(variable[:], dtype=np.float64).reshape(-1)
    except OSError:
        return None
    if values.size != 1:
        return None
    return float(values[0])


def require_runnable_child_radiation_from_archive(cfg, parent_frame):
    """The radiation ladder rule, asked BEFORE anything is interpolated.

    The plan-review and admission doors hold parent frames and no
    interpolation receipt, so they read the model top off the tape and put
    the same question to the same function.  A frame that does not carry
    ``P_TOP`` leaves the rule to the state builder, which always has the
    receipt.  Returns ``cfg``.
    """

    p_top = parent_archive_p_top(parent_frame)
    if p_top is None:
        return cfg
    return require_runnable_child_radiation(cfg, p_top)


def _climatology_surface_emission(initial, cfg) -> dict:
    """QNWFA2D/QNIFA2D for an mp=28 child of a parent of another scheme.

    WRF's climatology branch (module_initialize_real.F:4530-4536, through
    :func:`woof.ingest.wif_climatology.wif_surface_emission`): the monthly
    lowest-level water-friendly number on the child's mass points, dated to
    the child's start, over the child's first-layer thickness; QNIFA2D is
    zero, as WRF leaves it.  The dataset is staged by the pinned,
    SHA-verified routine when it is not on the machine; with no dataset at
    all the child is refused by name rather than run with zero emission.
    """

    from woof.ingest.wif_climatology import (
        load_wif_climatology, resolve_wif_climatology,
        wif_lowest_number_for_grid, wif_surface_emission)

    named = getattr(cfg, "wif_climatology_path", "") or None
    resolution = resolve_wif_climatology(named)
    if not resolution.resolved and named is None:
        from woof.table_assets import stage_wif_dataset

        stage_wif_dataset()
        resolution = resolve_wif_climatology(None)
    if not resolution.resolved:
        raise OfflineChildContractError(
            "an mp_physics=28 child of a parent of another scheme takes its "
            "surface aerosol emission (QNWFA2D) from WRF's monthly WIF "
            "climatology, and none resolved: "
            f"{resolution.fallback_reason}  Stage it with `woof "
            "fetch-tables --wif`, or name another suite with "
            "--physics-profile")
    number = wif_lowest_number_for_grid(
        load_wif_climatology(resolution.path),
        np.asarray(initial.fields["XLAT"], dtype=np.float32),
        np.asarray(initial.fields["XLONG"], dtype=np.float32),
        initial.valid_time.isoformat())
    nwfa2d, nifa2d = wif_surface_emission(
        number, np.asarray(initial.fields["PHB"], dtype=np.float32))
    return {"QNWFA2D": nwfa2d, "QNIFA2D": nifa2d}


def build_offline_child_domain_state(
        initial: InterpolatedInitialState, cfg, *, array_module=None):
    """Upload an interpolated parent-only cold start into ``DomainState``.

    The returned state has complete dynamics, moisture, map factors, and RK
    time-t copies. Physics and lateral tables are attached by their existing
    production APIs so callers can select the new child physics explicitly.
    """

    # The rule, not a second wording of it: one function, every door.
    require_offline_child_root_forcing(cfg)
    if initial.receipt.get("source_physics_binding") is None:
        raise OfflineChildContractError(
            "standalone offline child requires authoritative parent physics "
            "evidence from a setup/namelist/restart companion")
    if not cfg.moist:
        raise OfflineChildContractError(
            "offline child with microphysics requires moist=True")
    if not cfg.terrain_opt:
        raise OfflineChildContractError(
            "parent-inherited offline child base requires terrain_opt != 0")
    if int(cfg.hybrid_opt) != int(initial.receipt["hybrid_opt"]):
        raise OfflineChildContractError(
            f"child hybrid_opt={cfg.hybrid_opt} differs from archived parent "
            f"{initial.receipt['hybrid_opt']}")
    # History files persist ETAC as FP32, while RunConfig retains the user's
    # decimal as a Python float.  Compare at the precision of the archived
    # evidence so an exact FP32 round-trip (for example 0.2) is accepted.
    if not np.isclose(
            float(cfg.etac), float(initial.receipt["etac"]),
            rtol=0.0,
            atol=abs(float(np.spacing(np.float32(initial.receipt["etac"]))))):
        raise OfflineChildContractError(
            f"child etac={cfg.etac} differs from archived parent "
            f"{initial.receipt['etac']}")
    target_mp = int(initial.receipt["target_mp_physics"])
    if int(cfg.mp_physics) != target_mp:
        raise OfflineChildContractError(
            f"child cfg mp_physics={cfg.mp_physics} != prepared target {target_mp}")
    _require_prepared_child_ladder(initial, cfg)
    require_runnable_child_radiation(cfg, float(initial.receipt["p_top"]))
    from woof.core.diagnostics import update_diagnostics
    from woof.core.state import DomainState
    coord, base = _base_from_interpolated_initial(initial, cfg)
    state = DomainState(cfg, array_module=array_module)
    state.load_base(coord, base)
    xp = np if array_module is np else __import__("cupy")

    def assign(name, value):
        target = getattr(state, name)
        host = np.asarray(value, dtype=np.float32)
        if tuple(target.shape) != tuple(host.shape):
            raise OfflineChildContractError(
                f"initial {name} shape {host.shape} != state {target.shape}")
        target[...] = xp.asarray(host, dtype=xp.float32)

    for state_name, wrf_name in (
            ("u", "U"), ("v", "V"), ("w", "W"),
            ("php", "PH"), ("mup", "MU")):
        assign(state_name, initial.fields[wrf_name])
    thermal = "THM" if cfg.use_theta_m else "T"
    if int(initial.receipt.get("use_theta_m", 0)) != cfg.use_theta_m:
        raise OfflineChildContractError(
            "offline child thermal selector differs from its prepared archive interpolation; "
            "prepare initial and boundary fields with the child use_theta_m value")
    total_theta = np.asarray(initial.fields[thermal], dtype=np.float32) + np.float32(300.0)
    state.thp[...] = xp.asarray(
        total_theta - np.asarray(base.thb, dtype=np.float32), dtype=xp.float32)
    state.set_map_coriolis(
        initial.fields["MAPFAC_M"], initial.fields["MAPFAC_U"],
        initial.fields["MAPFAC_V"], initial.fields["F"], initial.fields["E"],
        sina=initial.fields["SINALPHA"], cosa=initial.fields["COSALPHA"])
    for name, value in initial.microphysics.items():
        target = getattr(state, name, None)
        if target is None:
            raise OfflineChildContractError(
                f"child DomainState has no target microphysics field {name!r}")
        assign(name, value)
    if target_mp == 28:
        # The two per-domain surface aerosol emission constants.  They are
        # NOT in ``initial.microphysics`` because they are not transported
        # scalars; they arrive through ``initial.fields`` alongside the
        # geometry.  See _AEROSOL_SURFACE_EMISSION_WRF for why the child
        # cannot re-derive them from its own state.  A parent of another
        # scheme has none to hand on, so a mixed edge into mp=28 takes them
        # from WRF's monthly WIF climatology on the child's grid, as
        # real.exe builds them for a cold mp=28 domain
        # (:func:`_climatology_surface_emission`).
        conversion = initial.receipt.get("conversion") or {}
        mixed_into_28 = (conversion.get("source_mp_physics") is not None
                         and int(conversion["source_mp_physics"]) != 28)
        emission = (_climatology_surface_emission(initial, cfg)
                    if mixed_into_28 and any(
                        name not in initial.fields
                        for name in _AEROSOL_SURFACE_EMISSION_WRF) else {})
        for wrf_name, state_name in _AEROSOL_SURFACE_EMISSION_STATE.items():
            if getattr(state, state_name, None) is None:
                raise OfflineChildContractError(
                    "mp_physics=28 child DomainState has no surface aerosol "
                    f"emission field {state_name!r}")
            value = initial.fields.get(wrf_name, emission.get(wrf_name))
            if value is None:
                raise OfflineChildContractError(
                    "mp_physics=28 offline child requires the parent's "
                    f"{wrf_name}; without it the child would integrate with "
                    "zero surface aerosol emission and nothing would raise")
            assign(state_name, value)
    # The online nest lane's own seeding table
    # (woof/ingest/nest_init.py::RK_TIME_T_SEED_PAIRS), not a local copy.
    # The local copy this replaced had already drifted: it stopped at the
    # Morrison moments, so an offline NSSL child's ten moment seeds
    # (qh0/qndrop0/...) and an offline P3 child's rime seeds (qir0/qib0)
    # would have started the first substep at zero while the current
    # fields carried the parent -- the exact twelve-of-fourteen defect the
    # shared table's comment records for the online lane.  Every pair is
    # None-guarded, so schemes without a field skip it, as before.
    seed_rk_time_t_copies(state)
    update_diagnostics(state, cfg.hypsometric_opt)
    if target_mp == 10 and getattr(state, "nc", None) is not None:
        # Morrison's droplet number is not transported, so no parent value
        # reaches it: it starts as the scheme writes it on every call
        # (NDCNST per cm3 over the density), not at the zero the first
        # frame used to publish beside the parent's cloud water.
        from woof.ingest import handover_numbers as hn
        if hn.closure_active(cfg):
            state.nc[...] = xp.asarray(
                hn.MORRISON_NDCNST_PER_CM3 * 1.0e6 * state.alt, dtype=xp.float32)
    return state


def _couple_parent(raw, moisture, coeffs, backend: str, *, use_theta_m=0):
    arrays = {name: _backend_array(value, backend) for name, value in raw.items()}
    scalars = {name: _backend_array(value, backend)
               for name, value in moisture.items()}
    total_mu = arrays["MUB"] + arrays["MU"]
    mux = mu_at_u_faces(total_mu)
    muy = mu_at_v_faces(total_mu)
    mux[:, 0] = total_mu[:, 0]
    mux[:, -1] = total_mu[:, -1]
    muy[0, :] = total_mu[0, :]
    muy[-1, :] = total_mu[-1, :]
    xp = np if backend == "cpu" else __import__("cupy")
    c1h = xp.asarray(coeffs["c1h"], dtype=xp.float32)[:, None, None]
    c2h = xp.asarray(coeffs["c2h"], dtype=xp.float32)[:, None, None]
    c1f = xp.asarray(coeffs["c1f"], dtype=xp.float32)[:, None, None]
    c2f = xp.asarray(coeffs["c2f"], dtype=xp.float32)[:, None, None]
    chm = c1h * total_mu[None] + c2h
    chf = c1f * total_mu[None] + c2f
    result = {
        "u": (c1h * mux[None] + c2h) * arrays["U"] / arrays["MAPFAC_U"][None],
        "v": (c1h * muy[None] + c2h) * arrays["V"] / arrays["MAPFAC_V"][None],
        "w": chf * arrays["W"] / arrays["MAPFAC_M"][None],
        "theta": chm * arrays["THM" if use_theta_m else "T"],
        "phi": chf * arrays["PH"],
        "mu": arrays["MU"][None],
    }
    result.update({name: chm * value for name, value in scalars.items()})
    return result, arrays, chm


def interpolate_parent_boundary_snapshot(
        path: str | Path, placement: OfflineChildPlacement, *,
        source_mp_physics: int | None = None,
        physics_binding: ParentPhysicsBinding | None = None,
        target_mp_physics: int | None = None,
        morr_rimed_ice: int | None = None, backend: str = "cpu",
        child_eta_levels=None, child_cfg=None, geography=None,
) -> InterpolatedBoundarySnapshot:
    """Conservatively SINT one archived parent state onto a child frame.

    Dynamics and source scalars are coupled on the parent before SINT, matching
    online ``bdy_interp1`` spatial semantics.  When target physics differs,
    the target scheme's fields are diagnosed on the parent by the online
    edge kernel, COUPLED by the kernel itself with the parent's hybrid mass
    (the ``coupled=True`` form the live nest coupler uses), and interpolated
    beside the dynamics exactly as the parent's own species would have been.

    A child on its own ``geography`` takes ndown's route instead
    (:func:`_own_geography_boundary_snapshot`).
    """

    if geography is not None:
        return _own_geography_boundary_snapshot(
            path, placement, source_mp_physics=source_mp_physics,
            physics_binding=physics_binding,
            target_mp_physics=target_mp_physics,
            morr_rimed_ice=morr_rimed_ice, backend=backend,
            child_eta_levels=child_eta_levels, child_cfg=child_cfg,
            geography=geography)
    source_mp_physics, morr_rimed_ice = _resolve_source_physics(
        source_mp_physics, physics_binding, morr_rimed_ice)
    backend = str(backend).strip().lower()
    info = inspect_parent_history_frame(path, source_mp_physics=source_mp_physics)
    expected = (placement.parent_ny, placement.parent_nx)
    actual = (info.dimensions["south_north"], info.dimensions["west_east"])
    if actual != expected:
        raise OfflineChildContractError(
            f"parent history mass grid {actual} != placement parent grid {expected}")
    target_mp = int(source_mp_physics if target_mp_physics is None
                    else target_mp_physics)
    transition = None
    if target_mp != int(source_mp_physics):
        transition = _offline_transition_contract(
            int(source_mp_physics), target_mp,
            morr_rimed_ice=morr_rimed_ice,
            hail_opt=(None if physics_binding is None
                      else physics_binding.hail_opt),
            child_cfg=child_cfg)
    hypsometric_opt = int(getattr(child_cfg, "hypsometric_opt", 2))
    started = time.perf_counter()
    child_znw = resolve_child_ladder(child_eta_levels)
    with _ParentHistory(path) as dataset:
        raw, moisture = _raw_parent_state(dataset, int(source_mp_physics),
                                          use_theta_m=getattr(child_cfg, "use_theta_m", 0))
        coeffs, hybrid_opt, etac, p_top = _vertical_coefficients(raw, dataset)
        if child_znw is not None or transition is not None:
            # Needed to make the child's geopotential a perturbation
            # against its OWN base, and for the parent density a
            # cross-scheme conversion diagnoses moments against; read here
            # rather than in _raw_parent_state so a child that inherits its
            # parent's ladder and scheme still requires exactly the
            # variables it always did.
            raw["PHB"] = _read_record(dataset, "PHB")
        if transition is not None and target_mp == 9:
            # Entering Milbrandt-Yau the kernel forms the absolute
            # temperature per cell from theta and the full pressure.
            raw["P"] = _read_record(dataset, "P")
            raw["PB"] = _read_record(dataset, "PB")
        from woof.ingest.handover_numbers import (closure_active,
                                                   handover_number_pairs)
        close_ring = (transition is None and closure_active(child_cfg)
                      and bool(handover_number_pairs(target_mp, child_cfg)))
        if close_ring:
            for name in ("P", "PB"):
                if name not in raw:
                    raw[name] = _read_record(dataset, name)
    if close_ring:
        # A same-scheme ring: numbers the parent left at zero beside a mass
        # are closed on the parent before coupling, so the ring never brings
        # a mass in without its number (woof/ingest/handover_numbers.py).
        moisture = {name: np.array(value, dtype=np.float32)
                    for name, value in moisture.items()}
        close_child_numbers(target_mp, moisture, raw,
                            morr_rimed_ice=(morr_rimed_ice if morr_rimed_ice is not None
                                            else getattr(child_cfg, "morr_rimed_ice", 1)),
                            cfg=child_cfg)
    conversion_receipt = None
    converted = None
    if transition is not None:
        converted = _convert_parent_microphysics(
            transition,
            _parent_transition_donor(
                raw, moisture, coeffs, p_top=p_top,
                hypsometric_opt=hypsometric_opt, target_mp_physics=target_mp),
            coupled=True, backend=backend, hypsometric_opt=hypsometric_opt)
        conversion_receipt = converted.receipt()
    coupled, raw_device, parent_chm = _couple_parent(
        raw, {} if converted is not None else moisture, coeffs, backend,
        use_theta_m=getattr(child_cfg, "use_theta_m", 0))
    if converted is not None:
        # Already coupled by the kernel with the parent's own hybrid mass.
        coupled.update({name: _backend_array(value, backend)
                        for name, value in converted.fields.items()})
    moisture_names = frozenset(
        moisture if converted is None else converted.fields)
    registrations = {
        "": placement.registration("", wrapper="bdy"),
        "x": placement.registration("x", wrapper="bdy"),
        "y": placement.registration("y", wrapper="bdy"),
    }
    stagger = {"u": "x", "v": "y"}
    interpolated = {
        name: sint(value, registrations[stagger.get(name, "")])
        for name, value in coupled.items()
    }
    # Same fix-up as the initial state, on the COUPLED moments.  Coupling
    # scales the field and its own peak by the same chm, so the relative
    # tolerance carries over untouched, but the ABSOLUTE floor is in the
    # field's units and has to be carried across with it -- hence
    # floor_scale.  The parent's chm is the right scale for it: it differs
    # from the child's by interpolation, and this is an order-of-magnitude
    # floor, not a precision instrument.  The boundary strip feeds the
    # relaxation zone, so an artefact left here walks into the child's
    # interior on the first blend.
    boundary_clamp = clamp_sint_undershoot_mapping(
        interpolated, floor_scale=float(abs(parent_chm).max()))
    boundary_clamp.update(clamp_sint_undershoot_mapping(
        interpolated, names=_POSITIVE_MASS_FIELDS, floor_scale=0.0,
        reference_fields=coupled))
    remap_receipts = ()
    if child_znw is not None:
        parent_znw = np.asarray(raw["ZNW"], dtype=np.float64).reshape(-1)
        interpolated, remap_receipts = (
            _remap_boundary_snapshot_to_child_ladder(
                interpolated,
                child_mub=_to_host(sint(raw_device["MUB"], registrations[""])),
                child_phb=_to_host(sint(
                    _backend_array(raw["PHB"], backend), registrations[""])),
                parent_znw=parent_znw, child_znw=child_znw,
                hybrid_opt=hybrid_opt, etac=etac, p_top=p_top,
                moisture_names=moisture_names))
    fields = MappingProxyType({name: _to_host(value)
                               for name, value in interpolated.items()})
    receipt = MappingProxyType({
        "path": str(Path(path).resolve()),
        "valid_time": info.valid_time.isoformat(),
        "geometry_sha256": info.geometry_sha256,
        "source_kind": info.source_kind,
        "source_mp_physics": int(source_mp_physics),
        "advisory_inferred_mp_physics": info.inferred_mp_physics,
        "source_physics_binding": (
            None if physics_binding is None else dict(physics_binding.receipt())),
        "target_mp_physics": target_mp,
        "backend": backend,
        **({"use_theta_m": 1} if getattr(child_cfg, "use_theta_m", 0) else {}),
        "positive_definite_clamp": boundary_clamp,
        "hybrid_opt": hybrid_opt,
        "etac": etac,
        "p_top": p_top,
        "field_inventory": tuple(sorted(fields)),
        "vertical_remap": None if child_znw is None else {
            "source_levels": int(np.asarray(raw["ZNW"]).size - 1),
            "target_levels": int(child_znw.size - 1),
            "fields": tuple(item.summary() for item in remap_receipts),
            "max_relative_drift": max(
                (item.max_relative_drift for item in remap_receipts),
                default=0.0),
        },
        "conversion": None if conversion_receipt is None else dict(conversion_receipt),
        "seconds": time.perf_counter() - started,
    })
    return InterpolatedBoundarySnapshot(info.valid_time, fields, receipt)


def _own_geography_boundary_snapshot(
        path, placement, *, source_mp_physics, physics_binding,
        target_mp_physics, morr_rimed_ice, backend, child_eta_levels,
        child_cfg, geography) -> InterpolatedBoundarySnapshot:
    """One boundary frame of a child on its own terrain, as ndown makes it.

    ndown builds its boundary file from the fine grid's own rebalanced
    state, frame by frame (main/ndown_em.F:806-860): every parent frame is
    interpolated, blended and rebalanced exactly as the initial state is,
    then coupled with the CHILD's own total dry mass and map factors
    (``couple``: u with ``msfuy``, v with ``msfvx``, theta, geopotential
    and scalars with the mass-point mass).  The interpolation is therefore
    the initial state's own (:func:`interpolate_parent_initial_state`, the
    uncoupled SINT and its positive-definite fix-up), and the coupling is
    :func:`_couple_parent`'s arithmetic on the child's arrays, the same
    convention the parent-terrain route couples with.
    """

    started = time.perf_counter()
    state = interpolate_parent_initial_state(
        path, placement, source_mp_physics=source_mp_physics,
        physics_binding=physics_binding,
        target_mp_physics=target_mp_physics, morr_rimed_ice=morr_rimed_ice,
        backend=backend, child_eta_levels=child_eta_levels,
        child_cfg=child_cfg, geography=geography)
    receipt = state.receipt
    info = inspect_parent_history_frame(
        path, source_mp_physics=int(receipt["source_mp_physics"]))
    fields = state.fields
    znw = np.asarray(fields["ZNW"], dtype=np.float64).reshape(-1)
    coeffs = compute_hybrid_coeffs(znw, int(receipt["hybrid_opt"]),
                                   float(receipt["etac"]), float(c.P0),
                                   float(receipt["p_top"]))

    def column(name):
        return np.asarray(coeffs[name], dtype=np.float64)[:, None, None]

    c1h, c2h, c1f, c2f = (column(name) for name in ("c1h", "c2h",
                                                    "c1f", "c2f"))
    mu = np.asarray(fields["MU"], dtype=np.float64)
    total = np.asarray(fields["MUB"], dtype=np.float64) + mu
    mux = _edge_pinned(np.asarray(mu_at_u_faces(total), dtype=np.float64),
                       total, axis=1)
    muy = _edge_pinned(np.asarray(mu_at_v_faces(total), dtype=np.float64),
                       total, axis=0)
    chm = c1h * total[None] + c2h
    chf = c1f * total[None] + c2f

    def host(name):
        return np.asarray(fields[name], dtype=np.float64)

    coupled = {
        "u": (c1h * mux[None] + c2h) * host("U") / host("MAPFAC_U")[None],
        "v": (c1h * muy[None] + c2h) * host("V") / host("MAPFAC_V")[None],
        "w": chf * host("W") / host("MAPFAC_M")[None],
        "theta": chm * host("THM" if getattr(child_cfg, "use_theta_m", 0) else "T"),
        "phi": chf * host("PH"),
        "mu": mu[None],
    }
    coupled.update({name: chm * np.asarray(value, dtype=np.float64)
                    for name, value in state.microphysics.items()})
    out = MappingProxyType({name: np.ascontiguousarray(value, dtype=np.float32)
                            for name, value in coupled.items()})
    snapshot_receipt = MappingProxyType({
        "path": receipt["path"],
        "valid_time": receipt["valid_time"],
        "geometry_sha256": receipt["geometry_sha256"],
        "source_kind": info.source_kind,
        "source_mp_physics": receipt["source_mp_physics"],
        "advisory_inferred_mp_physics": receipt["advisory_inferred_mp_physics"],
        "source_physics_binding": receipt["source_physics_binding"],
        "target_mp_physics": receipt["target_mp_physics"],
        "backend": receipt["backend"],
        **({"use_theta_m": 1} if receipt.get("use_theta_m", 0) else {}),
        "positive_definite_clamp": receipt["positive_definite_clamp"],
        "hybrid_opt": receipt["hybrid_opt"],
        "etac": receipt["etac"],
        "p_top": receipt["p_top"],
        "field_inventory": tuple(sorted(out)),
        "vertical_remap": receipt["vertical_remap"],
        "conversion": receipt["conversion"],
        "terrain_policy": receipt["terrain_policy"],
        "terrain": receipt.get("terrain"),
        "seconds": time.perf_counter() - started,
    })
    return InterpolatedBoundarySnapshot(state.valid_time, out,
                                        snapshot_receipt)


def _single_precision_interval(interval: BoundaryInterval) -> BoundaryInterval:
    """The same interval held at the precision the device reads it at.

    The device mirror is FP32 and every upload rounds the host tables to
    FP32 (``lateral_bc._reload_streaming_external_interval``), so rounding
    them once here, after the tendency is formed in float64, hands the card
    the same bits and halves what the archive holds on the host.  That
    matters because a derived child's relaxation zone is sized in parent
    cells (``woof.downscale.child_lateral_zone``): 41 rows at ratio 20,
    where WRF's zone is 5, and every row is held for every parent frame.
    No offline side carries a time law, so nothing is evaluated on the
    host from these tables.
    """

    fields = {}
    for name, boundary in interval.fields.items():
        sides = {}
        for side_name in ("west", "east", "south", "north"):
            side = getattr(boundary, side_name)
            if side.time_law is not None:
                sides[side_name] = side
                continue
            sides[side_name] = SideBoundary(
                np.asarray(side.value, dtype=np.float32),
                np.asarray(side.tendency, dtype=np.float32))
        fields[name] = FieldBoundary(**sides)
    # Rounding the tables does not move the frame the tendency was built
    # toward, and the next interval's FP32 start frame is that frame's bytes.
    return record_built_end_frame(
        BoundaryInterval(interval.start_seconds, interval.end_seconds,
                         fields), interval.end_frame_sha256)


def build_offline_lateral_boundaries(
        contract: ParentHistoryContract, placement: OfflineChildPlacement, *,
        target_mp_physics: int | None = None,
        morr_rimed_ice: int | None = None, backend: str = "cpu",
        child_eta_levels=None, child_cfg=None,
        spec_bdy_width: int = 5, spec_zone: int = 1, relax_zone: int = 4,
        geography=None,
) -> OfflineBoundaryResult:
    """Stream parent frames into compact child lateral value/tendency strips.

    ``geography`` puts every frame on the child's own terrain first
    (:func:`_own_geography_boundary_snapshot`); ``None`` keeps the parent's.
    """

    if contract.source_mp_physics is None:
        raise OfflineChildContractError(
            "offline boundary generation requires source mp_physics bound "
            "from a companion setup/namelist/manifest; inventory inference is "
            "advisory only")
    if contract.physics_binding is None:
        raise OfflineChildContractError(
            "offline boundary generation requires authoritative companion "
            "setup/namelist/restart evidence, not a bare scheme integer")
    started = time.perf_counter()
    side_names = ("west", "east", "south", "north")
    previous = None
    intervals = []
    receipts = []
    origin = contract.start_time
    for frame in contract.frames:
        snapshot = interpolate_parent_boundary_snapshot(
            frame.path, placement,
            source_mp_physics=contract.source_mp_physics,
            physics_binding=contract.physics_binding,
            target_mp_physics=target_mp_physics,
            morr_rimed_ice=morr_rimed_ice, backend=backend,
            child_eta_levels=child_eta_levels, child_cfg=child_cfg,
            geography=geography,
        )
        sides = {
            side: extract_lateral_side(snapshot.fields, side, spec_bdy_width)
            for side in side_names
        }
        if previous is not None:
            previous_time, previous_sides = previous
            intervals.append(_single_precision_interval(
                build_lateral_interval_from_sides(
                    previous_sides, sides,
                    start_seconds=(previous_time - origin).total_seconds(),
                    end_seconds=(
                        snapshot.valid_time - origin).total_seconds(),
                )))
        previous = (snapshot.valid_time, sides)
        receipts.append(snapshot.receipt)
    boundaries = LateralBoundaries(
        tuple(intervals), int(spec_bdy_width), int(spec_zone), int(relax_zone))
    return OfflineBoundaryResult(
        boundaries=boundaries, frame_receipts=tuple(receipts),
        preparation_seconds=time.perf_counter() - started,
    )


__all__ = [
    "CHILD_REPORT_PIPELINE",
    "ChildSurfaceState",
    "LES_CHILD_SPACING_M", "LES_CHILD_SPACING_SOURCE",
    "LES_CHILD_THREE_DIMENSIONAL_CLOSURES", "les_child_regime",
    "child_inherits_parent_levels",
    "OFFLINE_CHILD_MP_PHYSICS",
    "InterpolatedBoundarySnapshot", "InterpolatedInitialState",
    "OfflineBoundaryResult",
    "OfflineChildPlacement",
    "DERIVED_CHILD_SURFACE_CAVEAT", "DERIVED_CHILD_SURFACE_POLICY",
    "derive_child_surface_from_parent",
    "read_child_surface_state",
    "OfflineChildContractError", "OfflineSchemeTransition",
    "PARENT_SCHEME_CONTRACT",
    "ParentHistoryContract", "ParentHistoryFrame", "ParentPhysicsBinding",
    "bind_parent_physics_from_gpuwm_restart",
    "bind_parent_physics_from_wrf_namelist",
    "build_offline_child_domain_state", "build_offline_lateral_boundaries",
    "inspect_parent_history_frame", "interpolate_parent_boundary_snapshot",
    "interpolate_parent_initial_state", "offline_cross_scheme_refusal",
    "open_parent_history",
    "read_parent_microphysics", "reserve_output_root",
    "validate_parent_history",
]
