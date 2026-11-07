"""Production canonical frame-state hashing shared by run controllers.

The hash is model evidence, not a verification-only implementation detail.
Keeping it in the production package prevents run/benchmark entry points from
depending on the developer-only ``woof.verify`` tree.

TWO PLACES A DOMAIN CAN LIVE, ONE DIGEST
----------------------------------------
:func:`canonical_state_digest` walks a resident ``DomainState``.  A streamed
forecast has one; a STORE-DIRECT forecast does not, and that is the whole
point of it -- ``woof.ingest.prepared_store`` builds the domain slab by slab
into pinned host RAM precisely so no domain-shaped device state is ever
allocated, and above the card's ceiling none can be.  Pointing the resident
walker at whatever state such a run happens to be holding is not an error
that announces itself: with ``store = "host"`` the attached state is the
snapshot that FILLED the store and never moves again, so the final digest of
a six-hour forecast is the digest of its analysis, complete, self-consistent
and wrong.

:func:`canonical_store_digest` is the same digest taken off the store, and
the two must agree BIT FOR BIT for the same weather.  Three things make that
a claim rather than a hope:

*One hasher.*  Both routes assemble a ``{member name: array}`` manifest and
hand it to :func:`_canonical_digest_document`, which is the only code in this
module that touches ``hashlib``.  There is no second serialization to drift.

*One member-name mapping.*  The store is already keyed by the restart
archive's own member names (``state/thp``, ``scratch/mp_rainnc``,
``driver/pbl_tendencies/ru``, ``fields/ust``) because
``tilestream.physics_inventory.carrier_manifest`` IS
``restart.state_manifest`` + ``_scratch_manifest`` +
``carried_scratch_manifest`` + ``_driver_manifest`` merged, in that order.
So :func:`_canonical_store_manifest` reads the
store through ``carrier_inventory`` -- the same call
``tilestream.restart_stream.write_streamed_restart`` writes a byte-identical
archive out of -- and re-applies the digest's own membership rules to the
KEYS.  It does not maintain a table of names.

*One normalisation.*  The resident road hashes device arrays pulled through
``restart._host`` (a ``.get()``); the store road hashes pinned host arrays
that were never on the card.  :func:`_canonical_host` is the single funnel,
and it says out loud what a host array can carry that a ``.get()`` never
produces -- a non-native byte order, which would move both the member
descriptor and every payload byte.

WHAT THE STORE CANNOT CARRY, STATED RATHER THAN PAPERED OVER
------------------------------------------------------------
Two member classes are in the resident digest and are absent from a
store-direct domain's, and both absences are properties of the run rather
than of this module.  (A third, ``restart.RESTART_ONLY_DRIVER_SLOTS``, is
absent from BOTH: the resident walk reaches it through
``restart._driver_manifest`` and :func:`canonical_state_digest` subtracts it
again by the same name the store side does, because a member one road can
never produce is not evidence -- it is a difference between two roads
reporting the same weather.)  ``scratch/lbc_weights_0`` is created on FIRST
FORCE by ``lateral_bc._resident_weights`` on whichever state stepped; a
streamed domain steps tile buffers, so it does not exist on the domain
object either way, and a store-direct run has no domain object at all.  The
``nest_*`` rolling tables belong to a coupled child, and streaming refuses a
nest.  A store that nonetheless turns up carrying an unrecognised member
under one of :data:`_CANONICAL_EXTRA_PREFIXES` is refused here exactly as
the resident walk refuses one, because that is the shape a silently-dropped
member class arrives in.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping

import numpy as np

from woof.ingest.lateral_bc import COUPLED_SCALAR_STATE_FIELDS


CANONICAL_STATE_SCHEMA = "gpuwm-canonical-state-v2"
CANONICAL_INVENTORY_SCHEMA = "gpuwm-canonical-state-inventory-v2"
CANONICAL_LAZY_MEMBER_CLASSES = (
    "nest rolling value/tendency, donor, and SINT tables",
    "lateral-boundary relaxation weights",
    "REFL_10CM frame stash",
    "microphysics carrying accumulators",
    "cumulus carrying accumulators",
    "KF W0AVG trigger history",
    "nest precipitation placement archive",
)
# The digest's nest-slot membership is derived from the SAME shared
# inventory the coupling machinery reads (COUPLED_SCALAR_STATE_FIELDS),
# never spelled inline: this was the SIXTH hand-copied table on 1.9.1
# D1's route.  A hand-maintained duplicate here lacked mp=9's nc/nh,
# WDM6's nn, P3's rime pair AND mp=28's nc/nwfa/nifa, so a nested run of
# any of those schemes integrated perfectly and then died in the
# end-of-run canonical digest ("scratch member 'nest_nc_btxe' ... not a
# concrete registered member") -- the same false-failure class as D3.
_CANONICAL_NEST_FIELD_KINDS = (
    "u", "v", "w", "t", "ph", "mu",
    *sorted(COUPLED_SCALAR_STATE_FIELDS),
)
_CANONICAL_NEST_SLOTS = frozenset({
    *(f"nest_{kind}_{prefix}{side}"
      for kind in _CANONICAL_NEST_FIELD_KINDS
      for prefix in ("b", "bt")
      for side in ("xs", "xe", "ys", "ye")),
    "nest_parent_field",
    "nest_child_field",
    *(f"nest_sint_{component}_{stagger}"
      for component in ("ci", "ip", "cj", "jp", "xig", "xjg")
      for stagger in ("m", "x", "y")),
})
#: A nest's rolling value/tendency frames of a chem row
#: (woof/core/nest_fields.py appends every transported ``chem_<row>``):
#: the same eight sides as the fixed kinds above, but the kinds are the
#: chem table's rows, so they are matched by shape rather than listed.
_NEST_FRAME_SUFFIXES = tuple(f"_{prefix}{side}" for prefix in ("b", "bt")
                             for side in ("xs", "xe", "ys", "ye"))


def _is_chem_nest_slot(slot: str | None) -> bool:
    from woof.state_serialization_contract import CHEM_STATE_PREFIX

    head = "nest_" + CHEM_STATE_PREFIX
    if slot is None or not slot.startswith(head):
        return False
    return any(slot.endswith(suffix) and len(slot) > len(head) + len(suffix)
               for suffix in _NEST_FRAME_SUFFIXES)


_CANONICAL_LBC_SLOTS = frozenset({"lbc_weights_0"})
_CANONICAL_REFL_SLOTS = frozenset({"refl_10cm"})
_CANONICAL_EXTRA_PREFIXES = ("nest_", "lbc_weights_", "refl_")
# Reflectivity phase arrays are overwritten on each WSM3 diagnostic call.
_CANONICAL_EXCLUDED_FRAME_SCRATCH = frozenset({
    "refl_t", "refl_sbu_snow", "refl_sbu_graupel", "refl_wsm3_rain", "refl_wsm3_snow", "refl_wsm3_graupel",
    # The fork's gsd_diagnostics reflectivity at 1 km and 4 km AGL
    # (woof/core/gsd_diagnostics.py), recomputed from the state at every
    # history time and read by nothing that steps: output, not trajectory.
    # Breakage this prevents: a whole HRRR-recipe forecast finished 3 h and
    # then failed its final digest, "canonical frame-state scratch member
    # 'refl_10cm_1km' is inside an audited lazy prefix but is not a concrete
    # registered member" (a rented box, 2026-10-21 19:23Z).
    "refl_10cm_1km", "refl_10cm_4km",
})
CHILD_DUTY_SCRATCH_MEMBERS = (
    "scratch/nest_parent_field",
    "scratch/nest_child_field",
)


def _inventory_sha256(members: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256(b"gpuwm-canonical-state-inventory-v2\0")
    for member in members:
        descriptor = json.dumps(
            [str(member["name"]), str(member["dtype"]),
             [int(size) for size in member["shape"]]],
            separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
    return digest.hexdigest()


def lazy_inventory_member_class(name: str) -> str | None:
    """Return the documented lazy class for a canonical member path."""
    from woof.io import restart as restart_io
    from woof.core.nest_accumulation import is_archive_slot

    scratch_slot = (
        name.removeprefix("scratch/") if name.startswith("scratch/") else None
    )
    if (scratch_slot in _CANONICAL_NEST_SLOTS or _is_chem_nest_slot(scratch_slot)
            or (scratch_slot is not None and is_archive_slot(scratch_slot))):
        return CANONICAL_LAZY_MEMBER_CLASSES[0]
    if scratch_slot in _CANONICAL_LBC_SLOTS:
        return CANONICAL_LAZY_MEMBER_CLASSES[1]
    if scratch_slot in _CANONICAL_REFL_SLOTS:
        return CANONICAL_LAZY_MEMBER_CLASSES[2]
    if name.startswith("scratch/"):
        slot = name.removeprefix("scratch/")
        if slot in restart_io.SERIALIZED_SCRATCH_SLOTS:
            if slot.startswith("mp_"):
                return CANONICAL_LAZY_MEMBER_CLASSES[3]
            if slot.startswith("cu_"):
                return CANONICAL_LAZY_MEMBER_CLASSES[4]
    if is_archive_slot(scratch_slot):
        return CANONICAL_LAZY_MEMBER_CLASSES[6]
    if (name.startswith("cumulus/")
            and name.removeprefix("cumulus/") in restart_io.CUMULUS_CALLABLE_ARRAYS):
        return CANONICAL_LAZY_MEMBER_CLASSES[5]
    return None


def _canonical_extra_manifest(state) -> dict[str, object]:
    from woof.io import restart as restart_io

    manifest = {}
    for slot, value in sorted(getattr(state, "_scratch", {}).items()):
        # Already included by _scratch_manifest; do not reject it again
        # merely because its registered name starts with an audited prefix.
        if restart_io.classify_scratch_slot(slot) == "serialize":
            continue
        name = f"scratch/{slot}"
        # GSD's held refl_10cm_1km/4km planes are already registered in
        # restart's persistent output inventory and hashed by _scratch_manifest.
        # The extra lazy-prefix audit must not reject those same members.
        if slot in restart_io.SERIALIZED_SCRATCH_SLOTS:
            continue
        if slot in _CANONICAL_EXCLUDED_FRAME_SCRATCH:
            continue
        if lazy_inventory_member_class(name) in CANONICAL_LAZY_MEMBER_CLASSES[:3]:
            manifest[name] = value
        elif slot.startswith(_CANONICAL_EXTRA_PREFIXES):
            raise RuntimeError(
                f"canonical frame-state scratch member {slot!r} is inside an "
                "audited lazy prefix but is not a concrete registered member"
            )
    return manifest


def _canonical_scalar_bytes(scalars: Mapping[str, object]) -> bytes:
    return json.dumps(
        scalars, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")


def _canonical_host(name: str, value) -> np.ndarray:
    """One manifest member as the exact bytes the digest hashes.

    ``restart._host`` is a device-to-host ``.get()`` for a CuPy array and a
    bare ``np.asarray`` for anything already on the host, so a resident
    domain's device carriers and a store-direct domain's pinned host carriers
    arrive here as the same numbers in the same dtype;
    ``np.ascontiguousarray`` then puts them in the same order.

    ONE difference survives that funnel and only on the host side: BYTE ORDER.
    A ``.get()`` always lands native-endian, while a host array may carry a
    swapped dtype -- which changes the member descriptor (``str(dtype)`` is
    ``'>f4'`` rather than ``'float32'``) AND every byte of the payload, so a
    store built that way would hash as a different domain while comparing
    equal to the resident one element by element.  Nothing in this project
    builds one; this refuses rather than silently minting a digest no other
    road can reproduce.
    """
    from woof.io import restart as restart_io

    host = np.ascontiguousarray(restart_io._host(value))
    # isnative, not byteorder in ("=", "|"): an explicitly little-endian dtype ("<f4", as a NetCDF read or
    # np.frombuffer gives) is this machine order on x86 and was refused at the end of every hex global
    # forecast ("atmosphere/saved/vertical_velocity has dtype <f4", WOOF 1.1.0 dry bake, 2026-10-19).
    if not host.dtype.isnative:
        raise ValueError(
            f"canonical digest member {name!r} has dtype {host.dtype!r}, "
            "whose byte order is not this machine's.  The resident road "
            "hashes device arrays pulled with .get(), which are always "
            "native-endian, so a byte-swapped host array would hash as a "
            "different domain while holding identical numbers.  Store the "
            "carrier in native order.")
    return host


def _canonical_digest_document(manifest: Mapping[str, object],
                               scalars: Mapping[str, object],
                               scope: str, *,
                               before_hash=None) -> dict[str, object]:
    """The digest document, from an already-assembled member manifest.

    The ONLY place this module hashes anything.  Both roads -- the resident
    walk and the store walk -- reach it with a ``{member name: array}`` dict
    and the same three scalars, so "the store-direct digest equals the
    resident one" reduces to "the two manifests hold the same arrays under
    the same names", which is a statement about inventories that
    :func:`_canonical_store_manifest` can be read against.  A second copy of
    this serialization would put the schema, the descriptor framing and the
    member order back in play, and every one of those is a way for two
    correct manifests to hash differently.

    ``before_hash``, when given, is called once with the manifest's total
    bytes before the first member is read, so a caller can say how much
    it is about to hash (the run's finalization heartbeat does).
    """
    if before_hash is not None:
        before_hash(sum(int(getattr(value, "nbytes", 0) or 0)
                        for value in manifest.values()))
    scalar_bytes = _canonical_scalar_bytes(scalars)
    digest = hashlib.sha256(
        b"gpuwm-canonical-state-v2:" + scope.encode("ascii") + b"\0"
        + scalar_bytes
    )
    members = []
    for key in sorted(manifest):
        host = _canonical_host(key, manifest[key])
        member = {"name": key, "dtype": str(host.dtype),
                  "shape": list(host.shape)}
        descriptor = json.dumps(
            [member["name"], member["dtype"], member["shape"]],
            separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        # The array's C-order bytes, hashed in place (the bytes tobytes()
        # returns, without its copy of every member).
        digest.update(memoryview(
            np.ascontiguousarray(host).reshape(-1).view(np.uint8)))
        members.append(member)
    inventory = {
        "schema": CANONICAL_INVENTORY_SCHEMA,
        "sha256": _inventory_sha256(members),
        "array_count": len(members),
        "members": members,
    }
    return {
        "schema": CANONICAL_STATE_SCHEMA,
        "sha256": digest.hexdigest(),
        "inventory_sha256": inventory["sha256"],
        "scalar_sha256": hashlib.sha256(scalar_bytes).hexdigest(),
        "array_count": len(members),
        "field_order": [item["name"] for item in members],
        "inventory": inventory,
        "scalars": dict(scalars),
    }


def _dtbc_fp32_bits(clock) -> int:
    """``clock.dtbc_fp32``'s exact FP32 bit pattern.

    Hashed as BITS rather than as a float so the digest cannot be moved by
    JSON's decimal round-tripping of a number the model uses in single
    precision.
    """
    return int(np.asarray(np.float32(clock.dtbc_fp32)).view(np.uint32).item())


def canonical_state_digest(state, clock, *,
                           scope: str = "trajectory",
                           before_hash=None) -> dict[str, object]:
    """Hash complete frame-time trajectory state in restart order."""
    if scope not in ("trajectory", "full"):
        raise ValueError(f"unknown canonical digest scope {scope!r}")
    from woof.io import restart as restart_io

    manifest: dict[str, object] = {}
    manifest.update(restart_io.state_manifest(state))
    manifest.update(restart_io._scratch_manifest(state))
    driver = getattr(state, "physics", None)
    if driver is not None:
        manifest.update(restart_io._driver_manifest(driver))
        # Match the actual streaming producer's distinction: standalone
        # legacy-RRTMG host ozone is recomputed, while an explicit CAM
        # owner supplies a retained field to later radiation/nest calls.
        # Both use the historical checkpoint name radiation/o33d_grid;
        # retain the carried owner exactly once, on both digest roads.
        for slot in restart_io.RESTART_ONLY_DRIVER_SLOTS:
            # The explicit CAM owner is a swept device carrier under this
            # same key. Only the standalone adapter's host copy is excluded.
            if slot == "radiation/o33d_grid" and getattr(driver, "o3rad", None) is not None:
                continue
            manifest.pop(slot, None)
    manifest.update(_canonical_extra_manifest(state))
    if scope == "trajectory":
        for member in CHILD_DUTY_SCRATCH_MEMBERS:
            manifest.pop(member, None)
    scalars = {
        "elapsed_seconds": float(state.elapsed_seconds),
        "dtbc_fp32_bits": _dtbc_fp32_bits(clock),
        "driver": (None if driver is None else {
            "call_counts": {
                key: int(value)
                for key, value in sorted(driver.call_counts.items())
            },
            "ysu_nan_guard_fires": int(driver.ysu_nan_guard_fires),
            "microphysics_updates": int(driver.microphysics_updates),
        }),
    }
    if hasattr(state, '_dfi_physics_epoch'):
        scalars['dfi_physics_epoch'] = float(state._dfi_physics_epoch)
    return _canonical_digest_document(manifest, scalars, scope,
                                      before_hash=before_hash)


def _canonical_store_manifest(store) -> dict[str, object]:
    """The digest's member set, read out of a streamed domain's STORE.

    The store is keyed by the restart archive's own member names, so nothing
    is renamed or collected here; the whole job is deciding which of those
    keys the digest hashes, and it is decided by re-applying the resident
    walk's own three rules to the key rather than to a live attribute:

    * anything that is not ``scratch/*`` is a ``state/``, ``fields/``,
      ``driver/``, ``cumulus/`` or ``radiation/`` member and belongs, exactly
      as ``state_manifest`` + ``_driver_manifest`` take all of theirs;
    * a ``scratch/`` slot classified ``serialize`` belongs, which is
      ``_scratch_manifest``'s filter verbatim -- asked of
      ``restart.classify_scratch_slot``, not of a copy of its table;
    * a ``scratch/`` slot in one of the first three audited lazy classes --
      the nest rolling tables, the Davies weights, the REFL_10CM frame stash
      -- belongs, which is :func:`_canonical_extra_manifest`'s filter.

    Everything else drops, and that is not an oversight in either direction.
    The ``carry`` class (the two ``nwp_diagnostics`` tracker windows) is IN
    the store because a sweep must not lose it between steps and is NOT in the
    resident digest, because ``_scratch_manifest`` keeps only ``serialize``;
    dropping it here is what keeps the two digests equal.  ``refl_t`` drops
    for the same reason the resident walk drops it, by the same name.

    Reached through ``physics_inventory.carrier_inventory`` -- the identical
    call ``tilestream.restart_stream.write_streamed_restart`` builds a
    byte-identical archive from -- so a store and its checkpoint can never
    disagree about what the domain is.
    """
    from woof.io import restart as restart_io

    from tilestream import physics_inventory as physinv

    manifest: dict[str, object] = {}
    for key, value in physinv.carrier_inventory(store).items():
        head, _, slot = key.partition("/")
        if head != "scratch":
            manifest[key] = value
        elif (restart_io.classify_scratch_slot(slot) == "serialize"
              or slot in restart_io.CHECKPOINTED_CARRIED_SCRATCH_SLOTS):
            manifest[key] = value
        elif slot in _CANONICAL_EXCLUDED_FRAME_SCRATCH:
            continue
        elif lazy_inventory_member_class(key) in \
                CANONICAL_LAZY_MEMBER_CLASSES[:3]:
            manifest[key] = value
        elif slot.startswith(_CANONICAL_EXTRA_PREFIXES):
            raise RuntimeError(
                f"canonical frame-state store member {slot!r} is inside an "
                "audited lazy prefix but is not a concrete registered member"
            )
    return manifest


def canonical_store_digest(store, scalars, clock, *,
                           scope: str = "trajectory",
                           before_hash=None) -> dict[str, object]:
    """:func:`canonical_state_digest` for a domain that has no resident state.

    ``store`` is the streamed domain's ``{member name: host array}`` mapping
    and ``scalars`` is its ``tilestream.physics_inventory.carrier_scalars`` --
    the clock and the driver counters the sweep advances once per model step,
    which are the same three numbers a restart header carries and the same
    three the resident digest reads off ``state`` and ``state.physics``.  A
    store-direct run has nowhere else to read them: the driver that owns the
    counters on the resident road is a TILE BUFFER's here, and it has stepped
    once per TILE rather than once per domain step.

    The result is the identical document ``canonical_state_digest`` returns
    for the same domain integrated resident, including ``field_order`` --
    which is why the manifest rules above are stated against restart.py's own
    classifier rather than against a list of names.
    """
    if scope not in ("trajectory", "full"):
        raise ValueError(f"unknown canonical digest scope {scope!r}")
    manifest = _canonical_store_manifest(store)
    if scope == "trajectory":
        for member in CHILD_DUTY_SCRATCH_MEMBERS:
            manifest.pop(member, None)
    scalars = dict(scalars or {})
    if "elapsed_seconds" not in scalars:
        raise ValueError(
            "a store-direct canonical digest needs the DOMAIN's carrier "
            "scalars; 'elapsed_seconds' is missing, and defaulting it to "
            "zero would stamp every checkpoint of a run with its analysis "
            "time")
    # ``call_counts`` present iff the domain has a physics driver at all --
    # ``carrier_scalars`` adds the three counters together or not at all --
    # which is the same test ``write_streamed_restart`` makes for the header's
    # ``driver`` block, kept identical so a digest and a checkpoint of the
    # same instant never disagree about whether physics is running.
    driver = (None if "call_counts" not in scalars else {
        "call_counts": {
            key: int(value)
            for key, value in sorted(scalars["call_counts"].items())
        },
        "ysu_nan_guard_fires": int(scalars["ysu_nan_guard_fires"]),
        "microphysics_updates": int(scalars["microphysics_updates"]),
    })
    return _canonical_digest_document(manifest, {
        "elapsed_seconds": float(scalars["elapsed_seconds"]),
        "dtbc_fp32_bits": _dtbc_fp32_bits(clock),
        "driver": driver,
    }, scope, before_hash=before_hash)


__all__ = [
    "CANONICAL_INVENTORY_SCHEMA",
    "CANONICAL_LAZY_MEMBER_CLASSES",
    "CANONICAL_STATE_SCHEMA",
    "CHILD_DUTY_SCRATCH_MEMBERS",
    "canonical_state_digest",
    "canonical_store_digest",
    "lazy_inventory_member_class",
]
