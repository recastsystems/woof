"""Fail-closed status for WRF physics suites that are being ported.

This module is intentionally small and declarative.  A WRF scheme number is
not an implementation: the namelist importer may only admit a scheme after
its state, setup, driver, CUDA implementation, restart/output contract, and
validation gates have all landed.  Until then, a request fails once with a
complete list of the missing coupled components instead of failing on the
first number or, worse, silently choosing a nearby scheme.

The source of truth for the requested bindings is WRF v4.6.1 commit
``d66e442fccc04111067e29274c9f9eaccc3cef28``.  Source anchors and the staged
acceptance gates live in ``docs/wrf_thompson_mynn_ruc_port.md`` and
``docs/wrf_nssl2_mp18_port.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    # The "Path" return annotations below; pathlib itself is imported
    # where a path is built.
    from pathlib import Path

from woof.explain import layered, warn
#: Re-exported, not redefined.  The class lives in
#: ``woof.physics_vertical_contract`` because the physics LAUNCHERS raise it
#: too and they must not import this module; a second class here would mean a
#: caller's ``except`` caught the preparation-time refusal and missed the
#: identical first-call one.
from woof.physics_vertical_contract import PhysicsVerticalPreflightError
from woof.wrf461_compatibility import (
    PBL_OPTIONS,
    SURFACE_LAYER_OPTIONS,
    WRFVerdict,
    pbl_surface_layer_verdict,
)


# Exact token emitted into imported woof configurations when WRF's legacy
# RRTMG 4/4 pair is intentionally routed to woof's modern RTE+RRTMGP
# implementation.  It is trajectory-bound through RunConfig/restart identity.
#
# TOKEN FAMILIES: the substitution family (wrf-rrtmg-4-4-to-rte-rrtmgp-*)
# versions the RTE+RRTMGP adapter's WRF-matching behavior; the legacy
# family (wrf-rrtmg-4-4-legacy-*) is a DIFFERENT algorithm (the exact
# port) and carries WRF's snow discount verbatim from birth, so it has
# no -v2.  Assembly resolution (2026-07-28, per the radiation branch's
# own merge note): WRF_RRTMG_TO_RTE_RRTMGP rebinds to -v2, -v1 stays
# accepted for historical receipts, and the substitution tuple carries
# both -- the pairing rule in woof.config (every substitution-family
# token requires the rte-rrtmgp variant; the legacy token requires
# rrtmg_legacy) generalizes unchanged.
#
# Substitution-family version history (the token is a receipt: bumping
# it relabels NO old run):
#   -v1: snow entered the radiative ice path at full mass with its native
#        effective radius (adapter-native coupling).
#   -v2: WRF-matching option-4 explicit-snow-radius coupling -- ice path from
#        cloud ice only, snow mass discounted by MIN(0.99, (130/re_s)^2) with
#        re_s capped at 130 microns (module_ra_rrtmg_lw.F:12500-12532,
#        module_ra_rrtmg_sw.F:11040-11067).  Current importer default.
# Configurations carrying the -v1 token keep the -v1 behavior; unknown token
# values fail closed in config validation.
WRF_RRTMG_TO_RTE_RRTMGP = "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"
WRF_RRTMG_TO_RTE_RRTMGP_V1 = "wrf-rrtmg-4-4-to-rte-rrtmgp-v1"

# Peer token for the exact port of WRF v4.6.1's bundled legacy RRTMG
# (ra_lw_physics = ra_sw_physics = 4 running WRF's own algorithm rather
# than the RTE+RRTMGP substitution above).  Selected through
# RunConfig.ra_rrtmg_variant = "rrtmg_legacy"; the pairing rules live in
# woof.config.validate_run_config.
WRF_RRTMG_LEGACY = "wrf-rrtmg-4-4-legacy-v1"

#: Every wrf_rrtmg_compatibility value this lineage's code can honor,
#: beside "none".  Substitution-family tokens pair with the rte-rrtmgp
#: variant; the legacy token pairs with rrtmg_legacy (enforced in
#: woof.config.validate_run_config).
WRF_RRTMG_SUBSTITUTION_TOKENS = (
    WRF_RRTMG_TO_RTE_RRTMGP_V1,
    WRF_RRTMG_TO_RTE_RRTMGP,
)
WRF_RRTMG_COMPATIBILITY_TOKENS = (
    *WRF_RRTMG_SUBSTITUTION_TOKENS,
    WRF_RRTMG_LEGACY,
)

#: RunConfig.ra_rrtmg_variant values: which implementation serves a
#: resolved 4/4 RRTMG request.
RRTMG_VARIANT_RTE_RRTMGP = "rte-rrtmgp"
RRTMG_VARIANT_LEGACY = "rrtmg_legacy"


def rrtmg_variant(cfg) -> str:
    """The 4/4 implementation selector, tolerant of pre-field configs."""

    if isinstance(cfg, Mapping):
        return str(cfg.get(
            "ra_rrtmg_variant", RRTMG_VARIANT_RTE_RRTMGP))
    return str(getattr(cfg, "ra_rrtmg_variant", RRTMG_VARIANT_RTE_RRTMGP))


#: Import surface of the legacy RRTMG port: every compute/prep/ingest
#: module a legacy-selected forecast executes.
_RRTMG_LEGACY_MODULES = (
    "woof.ingest.rrtmg_coeffs",
    "woof.ingest.wrf_ozone",
    "woof.core.rrtmg_mcica",
    "woof.core.rrtmg_lw",
    "woof.core.rrtmg_sw",
    "woof.core.rrtmg_legacy_prep",
    "woof.core.rrtmg_legacy_device",
    "woof.core.rrtmg_legacy",
)

#: Packaged data assets and CUDA kernel sources the legacy port reads,
#: relative to the woof package directory.
_RRTMG_LEGACY_ASSETS = (
    "data/wrf_radiation/RRTMG_LW_DATA",
    "data/wrf_radiation/RRTMG_SW_DATA",
    "data/wrf_radiation/ozone.formatted",
    "data/wrf_radiation/ozone_lat.formatted",
    "data/wrf_radiation/ozone_plev.formatted",
    # SHA-pinned and REQUIRED (woof/core/rrtmg_legacy.py builds the
    # longwave coefficient set from it and refuses on a digest mismatch),
    # and absent from this list until audit R-030: a stripped install
    # passed the readiness check and then died inside the coefficient
    # build, which is the failure this check exists to move earlier.
    "data/wrf_radiation/rrtmg_lw_statics.npz",
    "core/kernels/rrtmg_lw.cu",
    "core/kernels/rrtmg_lw_chain.cu",
    "core/kernels/rrtmg_lw_zbatched.cu",
    "core/kernels/rrtmg_lw_taugb02_10_11_12.cu",
    "core/kernels/rrtmg_lw_taugb03_05.cu",
    "core/kernels/rrtmg_lw_taugb06_09.cu",
    "core/kernels/rrtmg_lw_taugb13_16.cu",
    "core/kernels/rrtmg_mcica_wrf.cu",
    "core/kernels/rrtmg_sw.cu",
    "core/kernels/rrtmg_legacy_adapter.cu",
    "core/kernels/rrtmg_legacy_prep.cu",
)


def require_rrtmg_legacy_ready() -> None:
    """Fail closed unless the legacy RRTMG port is genuinely present.

    Import-checks every compute module of the port and verifies the
    packaged coefficient/ozone data files and CUDA kernel sources exist,
    raising ONE receipt listing everything missing.  This replaced the
    pre-integration ``require_rrtmg_legacy_executable`` stub (which
    raised unconditionally); construction of
    ``woof.core.rrtmg_legacy.RRTMGLegacyRadiation`` performs the
    deeper readiness proof (SHA-pinned loads, kernel compilation,
    live-device preflights) and fails closed itself.  Selecting the
    legacy port must never silently fall back to RTE+RRTMGP.
    """

    import importlib
    from pathlib import Path

    missing: list[str] = []
    for name in _RRTMG_LEGACY_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - receipt, then raise
            missing.append(f"module {name}: {exc}")
    package_dir = Path(__file__).resolve().parent
    from woof.tables import _manifest, table_path
    pins = {"data/" + row["relative"]: row["family"] for row in _manifest()}
    for relative in _RRTMG_LEGACY_ASSETS:
        if relative in pins:
            table_path(pins[relative], package_dir / relative)
        elif not (package_dir / relative).is_file():
            missing.append(f"asset woof/{relative}: file not found")
    if missing:
        raise NotImplementedError(
            "ra_rrtmg_variant='rrtmg_legacy' (exact port of WRF v4.6.1's "
            "bundled RRTMG, ra_lw/sw_physics = 4/4) is selected but not "
            "executable on this installation; no silent fallback to "
            "RTE+RRTMGP is applied.  Missing:\n  - "
            + "\n  - ".join(missing))


#: Backwards-compatible name: earlier construction sites and external
#: callers guarded legacy selection through this symbol while the compute
#: lanes were still landing.  It now performs the readiness check.
require_rrtmg_legacy_executable = require_rrtmg_legacy_ready

# Thompson 8 completed the audited test host C CUDA/WRF-oracle forecast lane and
# the matched four-domain verification rerun of 2026-07-28
# (docs/thompson-rematch-20260728.md), and the canonical WRF v4.6.1 classic
# tables now ship as package data (woof_data/data/thompson/tables, in the
# recast-woof-data companion distribution since 2.5.0).  mp_physics=8
# is therefore selectable first-class: the process-environment enable guard
# is retired for selection (product decision, product/v1 packaging lane
# 2026-07-28), and the table root defaults to the packaged directory.
# WOOF_THOMPSON_TABLE_ROOT remains honored as an override naming a
# directory with the same byte-validated table set; every load still
# re-validates exact sizes and SHA-256 before GPU setup.  The retired
# enable-guard name is kept because the guarded evidence/benchmark runners
# under tools/ still enforce it for their own launch contracts.
EXPERIMENTAL_THOMPSON_ENV = "WOOF_EXPERIMENTAL_THOMPSON_MP8"
THOMPSON_TABLE_ROOT_ENV = "WOOF_THOMPSON_TABLE_ROOT"

#: The registry option id aerosol-aware Thompson (``mp_physics=28``) resolves
#: to.  Named once, here, because three different questions have to agree on
#: it: the vertical-bounds dispatch below, the registry document's
#: ``components.microphysics.options`` key, and any test that asserts what a
#: selector resolves to.  A literal repeated in those three places is how a
#: renamed option silently stops being bounds-checked.
MP28_REGISTRY_OPTION_ID = "thompson-aerosol-mp28"


def packaged_thompson_table_root() -> "Path":
    """The packaged canonical classic-table directory.

    The four assets and their MANIFEST.sha256 are committed under
    ``woof_data/data/thompson/tables`` -- same relative path, same
    bytes, in the ``recast-woof-data`` companion distribution since 2.5.0
    (see :mod:`woof.data_assets` for the measurement that split it out).
    Byte identity against
    ``woof.core.thompson_contract.CLASSIC_TABLE_ASSETS`` is enforced at
    load time, not assumed here.

    Raises the companion's named refusal when ``recast-woof-data`` is missing
    or version-skewed: this function answers "where does the canonical
    set live", and a wrong answer there is a load of the wrong bytes.
    The resolver in :func:`thompson_table_root` is the one that may keep
    walking, because a *staged* root is an equally canonical answer.
    """

    from woof import data_assets

    return data_assets.thompson_table_dir()


#: ``~/.woof/tables/thompson`` as path SEGMENTS under the home directory.
#:
#: Declared so a caller that must not resolve ``~`` can still read the one
#: spelling of this root.  The registry builder is that caller: it writes a
#: ``~``-relative string into a document required to be byte-identical on
#: every machine, and it had re-typed the three segments -- a second
#: spelling of a path, which is the drift class audit R-045 exists to
#: retire.
USER_THOMPSON_TABLE_ROOT_PARTS = (".woof", "tables", "thompson")


def user_thompson_table_root() -> "Path":
    """User-level staging directory: ``~/.woof/tables/thompson``.

    The same place ``~/.woof/bridges`` occupies for the built bridges,
    and for the same reason: ``woof fetch-tables`` used to stage its
    two downloads *inside site-packages*, where the next wheel upgrade
    or venv rebuild deletes them without saying so.  A user who had
    already paid for a 315 MiB download then met a bare
    ``FileNotFoundError`` in the middle of a forecast.  Under the home
    directory the staged set outlives the install that fetched it, and
    every checkout and venv on the machine reads the same bytes.

    Named per table set rather than a flat ``tables/`` because a root is
    validated as a *complete set*: mixing two schemes' assets in one
    directory would make "complete" unanswerable.
    """

    from pathlib import Path

    return Path.home().joinpath(*USER_THOMPSON_TABLE_ROOT_PARTS)


def _table_root_is_complete(root: "Path") -> bool:
    """Every classic asset filename present as a file in ``root``.

    Presence only.  The SHA-256 pins are re-checked at load by
    ``validate_table_assets``; resolution asks the cheap question
    (which root can this run read?) and answers it with stats, not with
    362 MiB of hashing on every import.
    """

    from woof.core.thompson_contract import CLASSIC_TABLE_ASSETS

    try:
        return all((root / asset.filename).is_file()
                   for asset in CLASSIC_TABLE_ASSETS)
    except OSError:  # pragma: no cover - unreadable home/site-packages
        return False


def thompson_table_root() -> str:
    """Resolved mp8 table root: env override, then staged, then packaged.

    The override exists for byte-identical mirrors (fast local disks,
    cluster scratch); a root with different bytes fails closed in
    ``validate_table_assets`` exactly as the packaged one would.

    Below it, a *complete* packaged directory still answers first --
    that is the read fallback that keeps every install which already
    staged into site-packages working, and it makes a git clone (whose
    packaged root ships the whole set) resolve exactly where it always
    did, the same way a checkout's own bridge build outranks a staged
    one in :func:`woof.bridges.artifact_candidates`.  When the packaged
    directory is short an asset -- every fresh wheel, and every wheel an
    upgrade has just emptied -- the complete staged set under
    :func:`user_thompson_table_root` answers instead.  Both roots are
    pinned to the same bytes, so this order chooses a location, never a
    numerical setup.

    Since 2.5.0 the packaged root lives in the ``recast-woof-data`` companion
    distribution, so asking for it can itself refuse.  That refusal is
    caught HERE and only here: a complete staged root under
    ``~/.woof/tables/thompson`` is an equally canonical answer, pinned
    to the same bytes, and refusing a run that can read every table it
    needs would name no breakage.  When nothing answers, the companion's
    refusal is re-raised rather than swallowed, because "no tables
    anywhere" and "no companion" want different fixes and the second is
    one pip line.
    """

    override = os.environ.get(THOMPSON_TABLE_ROOT_ENV)
    if override:
        return override
    packaged: "Path | None"
    try:
        packaged = packaged_thompson_table_root()
    except (ImportError, OSError) as error:
        packaged, companion_refusal = None, error
    else:
        companion_refusal = None
        if _table_root_is_complete(packaged):
            return str(packaged)
    try:
        staged = user_thompson_table_root()
    except (RuntimeError, OSError):  # pragma: no cover - no home directory
        staged = None
    if staged is not None and _table_root_is_complete(staged):
        return str(staged)
    if packaged is None:
        raise companion_refusal
    return str(packaged)


#: Environment override for the operational WRF 3.9 fork's Thompson tables
#: (RunConfig.thompson_version = "wrf_39_noaa").
THOMPSON_FORK_TABLE_ROOT_ENV = "WOOF_THOMPSON_FORK_TABLE_ROOT"


def thompson_fork_table_root() -> str:
    """Where the WRF 3.9 fork's Thompson tables live.

    The override first, then ``~/.woof/tables/thompson-wrf39-noaa``, the
    root ``woof fetch-tables --thompson-fork`` stages and a run that
    selects the fork fills on first use from the release assets
    (:mod:`woof.thompson_fork_assets`).  The set is generated from the
    fork's own ``thompson_init`` by ``tools/thompson_fork_oracle/build.sh``
    (gfortran on any Linux CPU, about two minutes) and is pinned by size and
    SHA-256 in :data:`woof.core.thompson_contract.FORK_TABLE_ASSETS`, so a
    root with other bytes fails closed exactly as the v4.6.1 root does.  The fork's
    CCN activation table is byte-identical to the v4.6.1 one and is read
    from :func:`thompson_table_root`.
    """
    override = os.environ.get(THOMPSON_FORK_TABLE_ROOT_ENV)
    if override:
        return override
    return str(user_thompson_table_root().parent / "thompson-wrf39-noaa")


def thompson_guard_exports() -> tuple[str, str]:
    """The two exports the guarded mp8 runners demand, ready to paste.

    Selection through the library is first-class and needs neither
    variable (see the block above).  The evidence and benchmark runners
    under ``tools/`` kept the launch contract on purpose, and a field run
    of the shipped 1.5.0 wheel met it as two consecutive one-line
    RuntimeErrors with no value in either: the reader had to find the
    table root themselves, having already downloaded it.

    So the pair is composed ONCE, here, beside the names and the
    resolver -- the generated command chain prints it before the stage
    that needs it, and the refusal that fires when it is missing prints
    the same two lines.  The root is this install's own resolution, so
    the printed line is the root the loader would have used, not a
    guess; ``validate_table_assets`` still re-checks every byte, so a
    root that has been pasted from somewhere else fails closed exactly
    as it did before.

    The shell form follows the platform, because a POSIX ``export`` line
    pasted into PowerShell is a syntax error and the reader is then
    debugging the instructions instead of the run.
    """

    import shlex

    from woof.bridges import WINDOWS_SHELL

    root = thompson_table_root()
    if WINDOWS_SHELL:
        return (f'$env:{EXPERIMENTAL_THOMPSON_ENV} = "1"',
                f'$env:{THOMPSON_TABLE_ROOT_ENV} = "{root}"')
    return (f"export {EXPERIMENTAL_THOMPSON_ENV}=1",
            f"export {THOMPSON_TABLE_ROOT_ENV}={shlex.quote(str(root))}")

# Noah-MP's whole column runs on the DEVICE: the slab orchestration in
# woof/core/noahmp_column_slab.py answers every land column with no Python
# per column, in chunks of woof.core.noahmp_runtime.SLAB_COLUMN_CHUNK
# (65,536), bitwise (max ULP 0) against the scalar column authority.
# Measured 2026-07-27, twice, on one RTX 5090, end to end through
# noahmp_lsm_step:
#
#     one 360,000-land-column call      0.202 - 0.227 s   (slab path)
#     the same call, per-column staged  166   - 206   s   (paired 2nd impl)
#
# At dt=1.667 s and bldt=0 that is 7.3 - 8.2 wall seconds per simulated
# minute of land surface, against 4,982 - 5,977 through the retired
# per-column CPython solver whose flat 7.18 ms/column this block used to
# tabulate.  Absolute seconds are a property of the machine; this box varies
# up to 30% between harnesses and hours.
#
# The ceiling below is still the LARGEST MEASURED configuration and nothing
# wider.  It is not a performance target and raising it makes nothing
# faster: above it a Noah-MP request WARNS -- it names the measured cost
# and the linear projection to the requested width, and continues.  It is
# not a blocker and has not been one since the warn-not-block ruling: the
# width is measurement coverage, not a correctness or memory limit, and a
# projection is not a defect.  The environment budget below silences the
# warning; it consents to a cost, it does not lift a refusal.  Until
# 2026-07-27 the ceiling was
# 352, the widest the host-era solver was ever measured at; the slab
# measurement at d04's full 360,000 columns is what moved it.
NOAHMP_EXPERT_COLUMN_BUDGET_ENV = "WOOF_NOAHMP_EXPERT_COLUMN_BUDGET"
#: Largest column count at which Noah-MP throughput has been measured:
#: d04 of the four-domain production tree, slab path, 2026-07-27.
NOAHMP_MEASURED_COLUMN_CEILING = 360_000
#: Seconds per land-surface call at exactly that ceiling, the (low, high)
#: of two paired timing runs on one RTX 5090, 2026-07-27.  Replaces the
#: retired host-era ``NOAHMP_MEASURED_MS_PER_COLUMN`` (a flat 7.18
#: ms/column), which described the per-column CPython solver.
NOAHMP_MEASURED_SLAB_CALL_SECONDS = (0.202, 0.227)

# Public forecast-runner profile identifiers.  These strings are shared with
# the Rust Studio runtime manifest and are deliberately more specific than a
# bare WRF selector: selecting ``mp_physics=8`` must not silently inherit the
# fixed WSM6 runner configuration.
WSM6_PROFILE_ID = "wsm6-ysu-mm5-noah-no-radiation-v1"
#: The native-HRRR warm-rain profile.  WRF v4.6.1
#: ``Registry/Registry.EM_COMMON:3015`` declares Kessler's qv/qc/qr package.
#: The profile is admitted only with the end-to-end HRRR probe and its
#: source-frozen-species discard receipt.
KESSLER_PROFILE_ID = "kessler-mp1-ysu-mm5-noah-dudhia-v1"
THOMPSON_PROFILE_ID = "thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1"
#: The observation battery's registered composition (lead ruling,
#: obs-battery integration wave 2026-08-04): the Thompson validation
#: suite with the exact WRF v4.6.1 legacy RRTMG in place of no-radiation,
#: cumulus off, transcribed switch for switch from the battery's
#: registered configuration.  wrf-matched-run-candidate in the registry:
#: the composition's first stock-WRF-paired t0/case receipt is the named
#: upgrade payer.
THOMPSON_LEGACY_RRTMG_PROFILE_ID = (
    "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1"
)
#: The same composition on the other 4/4 radiation engine, and the HRRR
#: route's default since the owner ruling of 2026-09-19 that RTE+RRTMGP is
#: the default radiation arm on every route.  It differs from the row
#: above in ``ra_rrtmg_variant`` and the compatibility token that records
#: it, and in nothing else: same microphysics, same PBL, same surface
#: layer, same land surface, cumulus off, same per-domain row.  The
#: registry ranks it at its composition ceiling,
#: implemented-unverified, because no receipt covers the composed suite
#: on this engine; the payer that moves it is this composition's first
#: stock-WRF-paired t0/case receipt.
THOMPSON_RTE_RRTMGP_PROFILE_ID = (
    "thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1"
)
#: The Shin-Hong sibling of the row above: the SAME composition with the
#: gray-zone PBL in place of YSU (``bl_pbl_physics`` 1 -> 11), which is
#: the one edge the divergence ledger's L3 entry moves
#: (:mod:`woof.physics_mode`).  It is registered because a fidelity-axis
#: arm that selects L3 resolves to exactly this suite, and an arm whose
#: physics no profile names cannot have a root prepared for it at all.
#: Every other switch is the row above's, transcribed rather than
#: re-derived, so the two rows differ in the PBL and nothing else and a
#: paired run of them isolates the closure.  wrf-matched-run-candidate in
#: the registry on the same terms as its sibling: Shin-Hong's own port is
#: measured bitwise against WRF v4.6.1 on both halves, and the payer that
#: moves the label is the composition's first stock-WRF-paired t0/case
#: receipt.
THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID = (
    "thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1"
)
MORRISON_PROFILE_ID = (
    "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"
)
NSSL2_PROFILE_ID = (
    "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-wrf-comparison-candidate-v1"
)
NSSL2_LEGACY_RRTMG_PROFILE_ID = (
    "nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-wrf-comparison-candidate-v1"
)
#: The P3 one-category composition: the Thompson legacy-RRTMG suite with
#: exactly ONE selector moved (``mp_physics`` 8 -> 50), transcribed switch
#: for switch so a paired run of the two isolates the microphysics.
#: Legacy RRTMG is this profile's composition of record.  It was chosen
#: when it was the one 4/4 engine P3 could couple to; the RTE+RRTMGP
#: cloud-optics row for mp=50 exists now (``50: "p3"`` in
#: ``woof.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME``, WRF's own has_reqs=0
#: remap coupling), so that pairing is selectable by config -- but this
#: shipped id keeps the composition it was issued under rather than
#: relabeling an already-published row.  ``cu_physics`` stays 0 (its
#: sibling's value), which also keeps the suite admissible on the nested
#: HRRR route's own physics gate.  Registered HRRR-only on the Kessler
#: rule.
P3_LEGACY_RRTMG_PROFILE_ID = (
    "p3-mp50-ysu-mm5-noah-rrtmg-legacy-v1"
)
#: The MYNN 5/5 suite, which differs from :data:`WSM6_PROFILE_ID` in exactly
#: two selectors (``bl_pbl_physics`` 1 -> 5 and ``sf_sfclay_physics`` 91 -> 5).
#: It is a peer profile rather than an expert one because it RUNS at production
#: width: ``woof/core/physics.py`` ``initialize_physics`` allocates every MYNN
#: array itself, so no runner needs extra wiring, and the standing runtime gate
#: forecasts 300 coupled steps with a bitwise restart
#: (``tests/test_mynn_pbl_runtime.py``).
MYNN_PROFILE_ID = "wsm6-mynn-mynn-noah-no-radiation-implemented-unverified-v1"
#: The RUC land-surface suite.  It differs from :data:`WSM6_PROFILE_ID` in
#: exactly one selector, ``sf_surface_physics`` 2 -> 3, with
#: ``num_soil_layers`` 4 -> 9 following from it, so a side-by-side run
#: isolates the land-surface change from everything else.  It is a peer
#: profile rather than an expert one because its column runs on the CARD:
#: 16.8 wall seconds per simulated minute at d04's 360,000 columns snow-free
#: and 65.7 fully snow-covered, measured at that width, not extrapolated.
RUC_PROFILE_ID = "wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1"
#: HRRR's operational suite class: MYNN surface/PBL with the RUC LSM.
MYNN_RUC_PROFILE_ID = (
    "wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1"
)
#: The Noah-MP fixed template remains expert-only because its component is
#: implemented but has no GPUWM/WRF forecast-trajectory comparison.  The
#: registry owns the acknowledgement id and warnings; callers must not
#: duplicate them as a profile-name special case.
NOAHMP_PROFILE_ID = (
    "wsm6-ysu-mm5-noahmp-no-radiation-expert-only-v1"
)
#: The MYNN 5/5 counterpart of the expert Noah-MP template.
MYNN_NOAHMP_PROFILE_ID = (
    "wsm6-mynn-mynn-noahmp-no-radiation-expert-only-v1"
)

# ---------------------------------------------------------------------------
# The radiation-bearing MYNN family.
#
# Until 1.8.7 every shipped MYNN suite was one of the three rows above, and
# all three run shortwave with longwave OFF.  That pairing is a DAYTIME
# validation configuration -- ``nocturnal_radiation_refusal`` below refuses
# it at config load for any window containing local night, and
# docs/public/PHYSICS.md lists all three under "nocturnally valid: no" --
# so a user who chose MYNN from a menu landed in the nocturnally-invalid
# class with no MYNN alternative to move to.  Composing MYNN with radiation
# through a hand-written config was always accepted (there is no engine
# lock: bl_pbl_physics 5 and ra_lw/sw_physics 4/4 resolve and are preserved
# switch for switch); what did not exist was a NAMED suite, and a named
# suite is what every menu, every ``--physics-profile`` choice list and
# every route declaration is keyed by.
#
# WHICH RADIATION FAMILY.  woof serves the resolved 4/4 pair two ways
# (:data:`RRTMG_VARIANT_RTE_RRTMGP` and :data:`RRTMG_VARIANT_LEGACY`); these
# rows take RTE+RRTMGP, the family the comparable YSU peers take --
# :data:`MORRISON_PROFILE_ID` (the wizard's gfs/era5 default, and the
# profile the nocturnal refusal itself names as the remedy) and
# :data:`NSSL2_PROFILE_ID`.  Its registry option is 'supported', its
# coefficient tables ship as package data, and it is the ``ra_rrtmg_variant``
# default, so these rows sit beside those two without an asset-staging or
# ``require_rrtmg_legacy_ready`` step of their own.  The legacy port stays
# available to any config that names it; it is not what a menu default
# should hand someone.
#
# Each row below is its no-radiation sibling with the RADIATION BLOCK
# replaced and nothing else touched, so a paired run isolates radiation.
# ``radt`` is part of that block and moves 1.0 -> 12.0: every
# radiation-bearing suite this file ships runs 12.0, and calling RRTMG every
# simulated minute costs twelve times what the sibling's Dudhia call did for
# no forecast benefit.  The MYNN option identity (bl_mynn_*, icloud_bl,
# iz0tlnd) is NOT restated here, exactly as it is not restated by the
# no-radiation rows: those are RunConfig defaults validate_run_config pins
# unconditionally, and a profile that repeated them could drift from them.
#: MYNN 5/5 + Noah + RTE+RRTMGP longwave AND shortwave: the nocturnally
#: valid member of the MYNN family, and the one a menu should offer first.
MYNN_RTE_RRTMGP_PROFILE_ID = (
    "wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1"
)
#: The HRRR operational pairing class (MYNN surface/PBL + RUC LSM) with
#: both radiation streams on.
MYNN_RUC_RTE_RRTMGP_PROFILE_ID = (
    "wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1"
)
#: The expert Noah-MP member of the family with both streams on.  Noah-MP
#: stays expert-only for the reason its no-radiation sibling does (no
#: woof/WRF forecast-trajectory comparison), not for a radiation reason.
MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID = (
    "wsm6-mynn-mynn-noahmp-rte-rrtmgp-expert-only-v1"
)

# ---------------------------------------------------------------------------
# The Thompson members of the MYNN + RUC family.
#
# The HRRR operational class is Thompson microphysics with MYNN surface
# layer and PBL over the RUC land surface.  The two WSM6 rows above carry
# the surface/PBL/land half of it, and until these rows there was no named
# suite with Thompson in it, so a user who wanted that class had to write
# the tuple by hand and no menu, no ``--physics-profile`` list and no
# preset could offer it.  Each row is its WSM6 sibling with ONE component
# moved (microphysics wsm6-mp6 -> thompson-mp8) and nothing else, so a
# paired run of the two isolates the microphysics; the maturity is the
# composition ceiling the registry derives, which is the siblings' own
# implemented-unverified, and the routes and sources are exactly the
# sibling's, because the RUC nine-layer soil ingest is what limits where
# either can run.
#
# The Dudhia member is named ``-dudhia-``, not ``-no-radiation-``: the
# WSM6 ids that say "no-radiation" run Dudhia shortwave with longwave off,
# and the registry carries a standing warning that those ids are wrong and
# frozen.  A new id does not repeat a name the registry calls wrong.
#: Thompson + MYNN 5/5 + RUC + RTE+RRTMGP on both streams: the nocturnally
#: valid member, and the one a menu offers.
THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID = (
    "thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1"
)
#: The same composition with Dudhia shortwave and longwave off, the
#: sibling of :data:`MYNN_RUC_PROFILE_ID`.  A daytime-only suite, as
#: that sibling is.
THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID = (
    "thompson-mp8-mynn-mynn-ruc-dudhia-implemented-unverified-v1"
)

# ---------------------------------------------------------------------------
# The composition suites (audit R-067).
#
# Each of these is an implemented option's FIRST named suite: before them
# the option was selectable only by hand-writing a component tuple, which
# is the ship-only-what-users-can-reach rule failing quietly.  Each is its
# base suite with the FEWEST components moved that reach the option, so a
# paired run isolates the change, and each is implemented-unverified --
# the registry owns that maturity and the warnings that go with it.
#
# None of them reads anything source-specific, which is why the route
# declarations offer them on every source that declares any suite at all.
#: Milbrandt-Yau two-moment (mp_physics=9) with New Tiedtke cumulus, on
#: legacy RRTMG: the composition the mp9 radiation constraint admits.
MILBRANDT2MOM_NTIEDTKE_PROFILE_ID = (
    "milbrandt2mom-mp9-ysu-mm5-noah-ntiedtke-rrtmg-legacy-v1"
)
#: WDM6 (mp_physics=16) with Grell-Freitas cumulus on RTE+RRTMGP.
WDM6_GRELL_FREITAS_PROFILE_ID = (
    "wdm6-mp16-ysu-mm5-noah-grell-freitas-rte-rrtmgp-v1"
)
#: Aerosol-aware Thompson (mp_physics=28) with the MYJ / Eta similarity
#: pair, cumulus off, on RTE+RRTMGP.
THOMPSON_AEROSOL_MYJ_PROFILE_ID = (
    "thompson-aerosol-mp28-myj-eta-noah-rte-rrtmgp-v1"
)
#: The SASE PBL on the revised MM5 surface layer, with the turbulence
#: closure supplied by the PBL scheme (km_opt 0, bldt/khdif/kvdif 0).
SASE_CLOSURE_SUPPLIED_PROFILE_ID = (
    "wsm6-sase-revised-mm5-noah-closure-supplied-v1"
)
#: PBL off with the 1.5-order TKE closure (km_opt 2) -- the large-eddy row.
TKE_1_5_ORDER_PROFILE_ID = (
    "wsm6-pbl-off-mm5-noah-tke-1-5-order-v1"
)
#: PBL off with the 3D Smagorinsky closure (km_opt 4), which differs from
#: the TKE row in exactly one component.
SMAGORINSKY_3D_PROFILE_ID = (
    "wsm6-pbl-off-mm5-noah-smagorinsky-3d-v1"
)
#: PBL off with the constant-K closure (km_opt 1), the fourth distinct
#: turbulence closure and the one left without a front door until R-067.
CONSTANT_K_PROFILE_ID = (
    "wsm6-pbl-off-mm5-noah-constant-k-v1"
)
#: The seven composition suites in the order every route declaration
#: appends them (tools/build_registry.py _SUITELESS_TEMPLATES), so a
#: per-source list and the registry route it is checked against cannot
#: disagree about their order.  The aerosol-aware suite joined the other
#: six when its cold-start arm landed (audit R-044): the prepared
#: single-domain runner builds its initialization like any other suite's,
#: with nwfa/nifa from the WIF monthly climatology.
NSAS_PROFILE_ID = "wsm6-ysu-mm5-noah-nsas-rrtmg-legacy-v1"
CCPP_TKE_EDMF_PROFILE_ID = "ccpp-tke-edmf-thompson-mm5-noah-gf-rte-v1"

COMPOSITION_SUITE_PROFILE_IDS = (
    NSAS_PROFILE_ID,
    CCPP_TKE_EDMF_PROFILE_ID,
    MILBRANDT2MOM_NTIEDTKE_PROFILE_ID,
    WDM6_GRELL_FREITAS_PROFILE_ID,
    THOMPSON_AEROSOL_MYJ_PROFILE_ID,
    SASE_CLOSURE_SUPPLIED_PROFILE_ID,
    TKE_1_5_ORDER_PROFILE_ID,
    SMAGORINSKY_3D_PROFILE_ID,
    CONSTANT_K_PROFILE_ID,
)
#: Option-identity knobs that are NOT part of a shipped profile's runtime
#: product (audit R-067, deriving what used to be nineteen hand-typed
#: rows).  Every name here is a value ``woof.config.RunConfig`` pins
#: unconditionally for the option that owns it, so a profile restating it
#: could only drift from it -- which is the reason the hand-written rows
#: gave for leaving the MYNN block, the RUC mosaic block, the Noah-MP
#: option block, the Smagorinsky coefficient and the radiation cloud
#: fraction out, one comment at a time.  ``nest_microphysics_transition``
#: is here for a different reason: it is a nest-EDGE policy, and a
#: single-domain product has no edge.
#:
#: The exclusion applies to what an OPTION declares -- its parameter block
#: and the identity values its ``required_settings`` pin.  A value the
#: TEMPLATE states is a choice about this composition rather than an
#: option's identity, so it is always carried.
_SWITCHES_OUTSIDE_THE_SINGLE_DOMAIN_PRODUCT = frozenset({
    "c_s", "icloud", "icloud_bl", "iz0tlnd",
    "bl_mynn_closure", "bl_mynn_cloudmix", "bl_mynn_cloudpdf",
    "bl_mynn_edmf", "bl_mynn_edmf_mom", "bl_mynn_edmf_tke",
    "bl_mynn_mixlength", "bl_mynn_mixqt", "bl_mynn_mixscalars",
    "bl_mynn_output", "bl_mynn_tkeadvect",
    "flag_sm_adj", "spp_lsm",
    "dveg", "noahmp_acc_dt", "noahmp_output", "soiltstep",
    "opt_alb", "opt_btr", "opt_crop", "opt_crs", "opt_frz", "opt_gla",
    "opt_inf", "opt_infdv", "opt_irr", "opt_irrm", "opt_pedo", "opt_rad",
    "opt_rsf", "opt_run", "opt_sfc", "opt_snf", "opt_soil", "opt_stc",
    "opt_tbot", "opt_tdrn",
    "nest_microphysics_transition",
})

#: Switches every prepared single-domain product pins, whether or not the
#: composition names them.  A profile that leaves one unstated resolves it
#: from the registry's own declared default for that parameter, which is
#: what an unstated switch has always meant -- so this is a list of NAMES,
#: never of values, and the values live in exactly one place.
_SINGLE_DOMAIN_SWITCH_FLOOR = (
    "moist", "moist_cq", "mp_physics", "top_lid", "epssm", "morr_rimed_ice",
    "wsm6_hail_opt", "ra_physics", "ra_lw_physics", "ra_sw_physics", "radt",
    "wrf_rrtmg_compatibility", "sf_sfclay_physics", "sf_surface_physics",
    "bl_pbl_physics", "cu_physics", "cudt_minutes", "num_soil_layers",
    "terrain_opt", "km_opt", "diff_6th_opt", "diff_6th_factor",
    "diff_6th_slopeopt",
)


def _route_declared_templates(route: Mapping[str, object]) -> tuple[str, ...]:
    """Every template one route declares, in declaration order.

    Sources are read from the route's OWN declaration rather than named
    here: a route that serves one source and a route that serves
    eighteen are the same walk, and adding a source is a registry row.
    Normal templates come before expert ones for every source, which is
    the order a route publishes and the order the doors report, so the
    two remain comparable element for element.
    """

    declared: tuple[str, ...] = ()
    for group in ("source_template_ids", "expert_template_ids"):
        table = route.get(group) or {}
        if not isinstance(table, Mapping):
            continue
        for source_id in sorted(table):
            for template_id in table[source_id] or ():
                if template_id not in declared:
                    declared += (template_id,)
    return declared


def route_physics_profiles(route_id: str) -> tuple[str, ...]:
    """The fixed templates ONE route offers, in its own declared order.

    A runner asks this about itself.  Its per-profile tables -- a native
    namelist contract, an initialization contract, a switch-forwarding
    map -- are keyed by the profiles IT offers, and keying them off the
    shared single-domain menu instead is what let one route's declaration
    grow past another route's replay tables and refuse at the door what
    plan review had accepted.
    """

    from woof.physics_registry import physics_registry

    route = physics_registry()["runner_routes"].get(route_id)
    if not isinstance(route, Mapping):
        raise ValueError(f"no registered runner route {route_id!r}")
    return _route_declared_templates(route)


def _derive_single_domain_profiles():
    """The single-domain profile menu and its runtime products, DERIVED.

    AUDIT R-067.  Eleven implemented options had no shipped template at
    all, and the reason adding one was expensive is here: a template
    needed a hand-typed row of two dozen switches in this module, and the
    row had to be transcribed correctly from the registry that already
    stated every one of them.  Nineteen rows, four hundred values, each
    one a chance to disagree with the composition it claims to be.

    The menu is EVERY template a fixed-template route declares, in route
    order and then in declaration order.  It used to be one route's list
    -- the native benchmark route's -- and that made one route's replay
    table the menu authority for a different runner: a suite the other
    fixed-template route offers was resolvable only if the benchmark
    route happened to declare it too, and a suite the benchmark route
    declares that its own per-profile tables cannot replay reached the
    door as an offer and refused there.  The routes are selected by
    ``mode``, not by id, so a third fixed-template route is a row in the
    registry rather than an edit here.

    Each product is resolved from the composition: the component's
    selectors, its option's parameter block (less the identity knobs
    above), whatever its ``required_settings`` demand, and the template's
    own parameters, with the floor filled from the registry's declared
    defaults.

    MEASURED before the hand-written table was deleted: this derivation
    reproduces all nineteen shipped rows exactly, key for key and value
    for value, on the registry as built.
    """

    from woof.physics_registry import physics_registry

    registry = physics_registry()
    components = registry["components"]
    parameters = registry["parameters"]
    menu: tuple[str, ...] = ()
    for route_id in sorted(registry["runner_routes"]):
        route = registry["runner_routes"][route_id]
        if route.get("mode") != "fixed-template":
            continue
        menu += tuple(
            template_id for template_id in _route_declared_templates(route)
            if template_id not in menu)

    products = {}
    for template_id in menu:
        template = registry["templates"][template_id]
        switches: dict[str, object] = {}
        for component_id, option_id in sorted(
                template["components"].items()):
            option = components[component_id]["options"][option_id]
            switches.update(option.get("selectors") or {})
            switches.update({
                name: value
                for name, value in (option.get("parameters") or {}).items()
                if name not in _SWITCHES_OUTSIDE_THE_SINGLE_DOMAIN_PRODUCT
            })
            switches.update({
                name: value
                for name, value in (
                    (option.get("constraints") or {}).get(
                        "required_settings") or {}).items()
                if name not in _SWITCHES_OUTSIDE_THE_SINGLE_DOMAIN_PRODUCT
            })
        switches.update({
            name: value
            for name, value in (template.get("parameters") or {}).items()
            # Explicit template choices survive option-identity exclusions.
            # A single-domain product still has no nest-edge policy.
            if name != "nest_microphysics_transition"
        })
        floor = list(_SINGLE_DOMAIN_SWITCH_FLOOR)
        if (switches.get("ra_lw_physics"), switches.get("ra_sw_physics")) == (
                4, 4):
            # The 4/4 option names the resolved spectral PAIR; which engine
            # computes it is ra_rrtmg_variant, and the choice exists only
            # for that pair.  A suite pinning the legacy port states it in
            # its template parameters; one that does not runs the
            # parameter's declared default, and the product says so
            # instead of leaving the reader to infer it from silence.
            floor.append("ra_rrtmg_variant")
        for name in floor:
            if name in switches:
                continue
            spec = parameters.get(name)
            if isinstance(spec, Mapping) and "default" in spec:
                switches[name] = spec["default"]
        products[template_id] = MappingProxyType(
            dict(sorted(switches.items())))
    return menu, MappingProxyType(products)


#: Every fixed single-domain template the front door can validate, in the
#: order the benchmark route declares them.  Expert templates remain in
#: this discovery tuple: selection is accepted by the parser, then the
#: registry-owned acknowledgement/capability checks below fail closed
#: before preparation if consent or an implementation is absent.
#:
#: Complete runtime products, not bare microphysics selectors: each
#: profile fixes the surrounding surface/PBL/cumulus/radiation/diffusion
#: switches of the composition it names.  Source adapters may materialize
#: these switches into a case-specific experiment descriptor, but they
#: must not reinterpret or partially apply them.
SINGLE_DOMAIN_PHYSICS_PROFILES, _SINGLE_DOMAIN_RUNTIME_SWITCHES = (
    _derive_single_domain_profiles())


def single_domain_runtime_switches(profile: str) -> dict[str, object]:
    """Return one complete canonical single-domain runtime product."""

    from woof.physics_registry import canonical_template_id
    profile = canonical_template_id(profile)
    try:
        return dict(_SINGLE_DOMAIN_RUNTIME_SWITCHES[profile])
    except KeyError:
        raise ValueError(
            f"unsupported single-domain physics profile {profile!r}"
        ) from None


def identify_single_domain_profile(run_config) -> str | None:
    """Which shipped profile a run config's switches ARE, or ``None``.

    The inverse of :func:`single_domain_runtime_switches`, and the reason
    it lives beside it: a printed next-command has to name the
    ``--physics-profile`` the prepared-forecast runner will accept, and
    the only accurate way to know is to ask the same table the runner's
    guard asks.  A second copy of this comparison somewhere else is how
    a printed command drifts into being wrong.

    Returns ``None`` when the config matches no profile exactly -- a
    hand-authored suite, or a profile this WOOF does not ship -- which
    the caller must report rather than guess around.  Ambiguity cannot
    arise (no two profiles share a switch set), but if it ever did,
    ``None`` is the answer: two names for one config is not an answer.
    """

    matched = [
        profile for profile in SINGLE_DOMAIN_PHYSICS_PROFILES
        if all(getattr(run_config, name, _MISSING) == value
               for name, value in
               single_domain_runtime_switches(profile).items())
    ]
    return matched[0] if len(matched) == 1 else None


#: Sentinel for :func:`identify_single_domain_profile`: a config that
#: lacks a switch entirely never matches a profile that pins it.
_MISSING = object()


#: Runtime switches a WRF namelist cannot state accurately, so somebody has
#: to decide them for it.  ``moist_cq`` has no WRF namelist key at all
#: (WRF derives calc_cq internally from the moist species that exist),
#: and woof's ``top_lid`` default is deliberately NOT WRF's Registry
#: default (woof/config.py records the falsification bar for flipping
#: it).  Both are part of every prepared-cache domain identity.
#:
#: They had three answers.  The shipped profiles above state them; the
#: domain wizard reads those same profiles; the WRF importer invented its
#: own -- ``moist_cq = mp_physics > 0`` and WRF's open-top default -- and
#: so a public root prepared from a profile and a public hierarchy
#: imported from the SAME namelist could not produce matching d01
#: identities for WSM6, Kessler, MYNN, RUC or Noah-MP.  This function is
#: the single answer all three now ask for.
IMPLICIT_RUNTIME_SWITCHES = ("moist_cq", "top_lid")

#: The switches that pick a shipped profile's row.  Every profile above
#: is unique on this tuple except the NSSL pair, which differs only in
#: its RRTMG variant and agrees on both implicit switches -- so a tie is
#: an answer here, not an ambiguity.
_PROFILE_SELECTOR_KEYS = (
    "mp_physics", "sf_sfclay_physics", "sf_surface_physics",
    "bl_pbl_physics", "cu_physics", "num_soil_layers",
    "ra_lw_physics", "ra_sw_physics",
)


def implicit_runtime_switches(**selection) -> dict[str, object]:
    """Certified ``moist_cq``/``top_lid`` for one WRF physics selection.

    ``selection`` is the WRF switch set an importer already resolved
    (:data:`_PROFILE_SELECTOR_KEYS`; unknown keys are ignored so callers
    may pass their whole suite).  When every shipped profile matching it
    agrees, those are the values -- byte-for-byte the ones
    :func:`single_domain_runtime_switches` hands the root preparer and
    the domain wizard. All selections enable WRF's moisture pressure
    correction, including microphysics off with passive vapor. The
    acoustic driver bypasses states without vapor. Other implicit switches,
    and ambiguous profile
    matches, retain ``RunConfig`` defaults.

    Returns ``{"moist_cq": ..., "top_lid": ..., "source": ...,
    "profiles": (...)}``: the source string is a receipt line, because a
    value decided FOR the user has to be able to say who decided it.
    """

    from woof.config import RunConfig

    requested = {key: selection[key] for key in _PROFILE_SELECTOR_KEYS
                 if key in selection}
    matched = []
    for profile in SINGLE_DOMAIN_PHYSICS_PROFILES:
        row = _SINGLE_DOMAIN_RUNTIME_SWITCHES[profile]
        if requested and all(row.get(key) == value
                             for key, value in requested.items()):
            matched.append(profile)
    answers = {
        tuple(_SINGLE_DOMAIN_RUNTIME_SWITCHES[profile][name]
              for name in IMPLICIT_RUNTIME_SWITCHES)
        for profile in matched
    }
    if len(answers) == 1:
        row = _SINGLE_DOMAIN_RUNTIME_SWITCHES[matched[0]]
        return {
            **{name: row[name] for name in IMPLICIT_RUNTIME_SWITCHES},
            "source": (
                "the shipped single-domain physics profile this suite IS "
                if len(matched) == 1 else
                "the shipped single-domain physics profiles this suite IS, "
                "which agree here: ")
            + ", ".join(matched),
            "profiles": tuple(matched),
        }
    defaults = RunConfig(nx=1, ny=1, nz=1, dx=1.0, dy=1.0, ztop=1.0,
                         dt=1.0, run_seconds=1.0)
    if not matched:
        return {
            "moist_cq": True,
            "top_lid": defaults.top_lid,
            "source": (
                "WRF moisture pressure correction whenever water vapor "
                "exists, including passive vapor; this suite is not one "
                "of the shipped single-domain physics profiles; other "
                "switches use RunConfig defaults"),
            "profiles": (),
        }
    return {
        **{name: getattr(defaults, name)
           for name in IMPLICIT_RUNTIME_SWITCHES},
        "source": (
            "woof's RunConfig defaults: this suite matches shipped "
            "profiles that disagree about these switches "
            f"({', '.join(sorted(matched))})"),
        "profiles": tuple(matched),
    }


class PhysicsCapabilityError(ValueError):
    """A selector combination has no executable registry capability."""


ACK_FLAG_SOURCE = "--ack"
ACK_TOML_SOURCE = "[experiment].acknowledgements"


def acknowledgement_delivery(
        *,
        flag: tuple[str, ...] = (),
        toml: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], dict[str, list[str]]]:
    """Merge acknowledgement delivery and retain its exact provenance."""

    delivered: dict[str, set[str]] = {}
    for source, values in (
        (ACK_FLAG_SOURCE, flag),
        (ACK_TOML_SOURCE, toml),
    ):
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{source} acknowledgement IDs must be non-empty strings")
            delivered.setdefault(value, set()).add(source)
    return (
        tuple(sorted(delivered)),
        {
            value: sorted(sources)
            for value, sources in sorted(delivered.items())
        },
    )


def _acknowledgement_receipt(
        acknowledged: set[str],
        required: set[str],
        provenance: Mapping[str, object] | None,
) -> tuple[list[str], dict[str, list[str]]]:
    used = sorted(acknowledged & required)
    sources: dict[str, list[str]] = {}
    for acknowledgement in used:
        raw = None if provenance is None else provenance.get(acknowledgement)
        if isinstance(raw, (list, tuple)) and all(
                isinstance(value, str) for value in raw):
            sources[acknowledgement] = sorted(set(raw))
        else:
            sources[acknowledgement] = ["api"]
    return used, sources


def _ack_instruction(acknowledgement: str) -> str:
    return (
        f"--ack {acknowledgement} or "
        f'acknowledgements = ["{acknowledgement}"]'
    )


#: Declared-experiment acknowledgement for running an asymmetric radiation
#: pairing (shortwave ON, longwave OFF) through a window that includes
#: local night.  Same governance idiom as the registry's expert-template
#: acknowledgements, delivered in the config itself
#: (``acknowledgements = [...]`` under ``[experiment]``) because the
#: refusal happens at config LOAD, before any front door's ``--ack``
#: flags are merged.
#:
#: Provenance (2026-08-06): a wizard-emitted 48 h real case bound
#: ``thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1`` (ra_lw_physics 0,
#: ra_sw_physics 1).  Shortwave heated the surface by day; at night the
#: surface radiated with no downward longwave, skin temperature
#: cratered, the surface saturation humidity collapsed with it, and 2 m
#: dewpoints read in the 50s F inside a 70s airmass.  The pairing is a
#: legitimate DAYTIME-only configuration and stays selectable --
#: loudly, never silently.
ASYMMETRIC_RADIATION_NOCTURNAL_ACK = (
    "asymmetric-radiation-nocturnal-window-v1"
)


#: THE DECLARED CONSTANT downward longwave, in W m-2.  It is a number a
#: caller may TYPE.  It is not a measurement, it is not a scheme, and no
#: default hands it out.
#:
#: WHY IT IS DEFINED HERE and re-exported by :mod:`woof.core.physics`
#: rather than the other way round.  The two config-load refusals that
#: quote the number -- :func:`constant_longwave_refusal` and
#: :func:`radiation_off_land_surface_refusal` -- sit in this module beside
#: :data:`CONSTANT_DOWNWARD_LONGWAVE_ACK`, the token that declares it, and
#: a refusal must be able to state the number without importing a CUDA
#: engine to read it.  The standalone RW-WPS preprocessing wheel is where
#: that stopped being a preference: it stages ``woof/physics_compat.py``
#: and forbids ``woof/core/physics.py``, so the refusal that reached
#: upward for this constant raised ImportError instead of refusing, for
#: every user who hit it.  One number, one definition, owned by the layer
#: that can ship on its own.
#:
#: Provenance.  Through 1.8.7 this value was the ``glw=300.0`` DEFAULT of
#: :func:`woof.core.physics.initialize_physics`, and
#: ``woof/core/dudhia.py`` -- shortwave only -- returns
#: ``glw=fields["glw"]``, the array it was handed, echoed back untouched.
#: No production call site ever passed ``glw=``.  So every run with
#: ``ra_lw_physics = 0`` had a downward longwave of exactly 300.0 W m-2,
#: everywhere, for the whole forecast: a plausible-looking number that
#: never responded to temperature, humidity or cloud.  It produced a real
#: user report -- 2 m dewpoints collapsing tens of degrees below the
#: airmass over the Gulf warm sector overnight -- because radiative
#: equilibrium at 300 W m-2 is 269.7 K (25.8 F) while a Gulf-coast October
#: night runs near 410 W m-2, or 291.6 K (65.2 F).  A ~105 W m-2 nightly
#: deficit craters skin temperature, and surface saturation humidity
#: follows it down.
DECLARED_CONSTANT_GLW_WM2 = 300.0


#: Declared-experiment acknowledgement for integrating a real case whose
#: downward longwave is a CONSTANT rather than a computed flux.
#:
#: Distinct from :data:`ASYMMETRIC_RADIATION_NOCTURNAL_ACK` on purpose,
#: and the distinction is the point.  That token declares "I know this
#: window contains night"; it says nothing about where GLW comes from,
#: and because it is checked before any physics is inspected it lifted
#: the whole question.  Ten shipped configs carried it, including the two
#: files a new ERA5 or GFS user copies first, so copying one walked
#: straight past the guard into a frozen 300 W m-2 forecast.  A token
#: that lifts a nocturnal guard must not also lift a "this flux is
#: fabricated" guard: they are different claims, so they are different
#: tokens, and a config that means both says both.
#:
#: What it declares: every domain of this experiment that runs a
#: land-surface scheme with ``ra_lw_physics = 0`` will consume
#: :data:`DECLARED_CONSTANT_GLW_WM2` as its downward longwave for the
#: whole forecast, and the run receipt will say so.
CONSTANT_DOWNWARD_LONGWAVE_ACK = "constant-downward-longwave-v1"


#: The land-surface schemes that READ GLW every surface step, and are
#: therefore what turns an absent longwave scheme into a wrong forecast
#: rather than an unused buffer.  ``sf_surface_physics`` numbering.
#:
#: MEMBERSHIP IS NOT DECIDED HERE any more, only the display names are.
#: The fact "this scheme reads GLW" belongs to the scheme's own carrier
#: contract (``woof.core.radiation_carriers.CONSUMER_CARRIERS``, which
#: refuses a scheme it has no row for rather than defaulting it to "reads
#: nothing"); the registry builder derives every land-surface option's
#: ``consumers.reads_glw`` row from that contract, and
#: :func:`_require_agreement_with_the_registry` below holds this mapping's
#: keys equal to those rows at import.  So a future GLW-consuming LSM --
#: the Pleim-Xiu case -- cannot be registered, run and silently pass this
#: guard: it fails the import agreement until it appears here, and what it
#: needs here is a NAME, not a judgement.  The keys are not imported from
#: the contract directly because ``woof.config`` imports this module and
#: the contract imports ``woof.config``; the registry sits between them.
_GLW_CONSUMING_SURFACE_SCHEMES = MappingProxyType({
    2: "Noah LSM", 3: "RUC LSM", 4: "Noah-MP",
})


def downward_longwave_disposition(
        *, ra_lw_physics: int, ra_sw_physics: int,
        sf_surface_physics: int) -> tuple[str, str | None]:
    """What becomes of the GLW buffer under these selectors.

    THE single classification every GLW decision reads --
    :func:`constant_longwave_refusal` (the config-load guard),
    ``woof.core.physics._resolve_initial_glw`` (the initialize-time
    guard), and ``woof.runtime.downward_longwave_source`` (the receipt
    line) all call this, so the door, the engine and the receipt can
    never disagree about which configurations have a downward-longwave
    question to answer.

    Returns ``(kind, consumer)`` where ``consumer`` is the land-surface
    scheme's name for ``"consumed"`` and ``None`` otherwise, and
    ``kind`` is one of:

    * ``"scheme"`` -- ``ra_lw_physics > 0``: a longwave scheme writes
      GLW every radiation call; the initial buffer is scratch.
    * ``"consumed"`` -- no longwave scheme, and a land-surface scheme
      (:data:`_GLW_CONSUMING_SURFACE_SCHEMES`) reads GLW every surface
      step.  Whatever is in the buffer becomes surface physics.
    * ``"published"`` -- no longwave scheme and no land-surface
      consumer, but shortwave is on, so the radiation slot is active and
      the GLW row is written to every wrfout frame.  Nothing integrates
      it, but a fabricated field in a wrfout file is indistinguishable
      from a measured one downstream.
    * ``"unused"`` -- radiation entirely off and no consumer: the buffer
      reaches no scheme and no file.

    ``"consumed"`` and ``"published"`` are the two kinds that must be
    DECLARED (:data:`CONSTANT_DOWNWARD_LONGWAVE_ACK`, or an explicit
    ``glw=`` at the initialize call) or refused.
    """

    if int(ra_lw_physics) > 0:
        return "scheme", None
    consumer = _GLW_CONSUMING_SURFACE_SCHEMES.get(int(sf_surface_physics))
    if consumer is not None:
        return "consumed", consumer
    if int(ra_sw_physics) > 0:
        return "published", None
    return "unused", None


def settings_declared_acknowledgements(settings) -> tuple[str, ...]:
    """The governance tokens these resolved selectors declare of themselves.

    A suite's own switches can make a claim that is true of every window
    and every place it is run in.  Exactly one does today: shortwave with
    ``ra_lw_physics = 0`` means nothing computes the downward longwave,
    so :data:`DECLARED_CONSTANT_GLW_WM2` is what the land surface
    integrates -- :func:`downward_longwave_disposition` is asked, so this
    can never require a token the load guard does not want or omit one it
    does.

    IT IS NOT THE WINDOW'S CLAIM.  :data:`ASYMMETRIC_RADIATION_NOCTURNAL_ACK`
    says "I know this run contains local night", which depends on the
    start time and the reference point rather than on the selectors, so
    it is never derived here: it stays the operator's, stated at the door
    that asked for the window.

    ``settings`` is anything :func:`woof.config.radiation_scheme_ids_from_settings`
    reads -- a runtime switch product or a RunConfig -- so the selection a
    profile resolves and the selection a configuration carries are
    classified by one reader.
    """

    from woof.config import radiation_scheme_ids_from_settings

    lw, sw = radiation_scheme_ids_from_settings(settings)
    if isinstance(settings, Mapping):
        surface = int(settings.get("sf_surface_physics", 0) or 0)
    else:
        surface = int(getattr(settings, "sf_surface_physics", 0) or 0)
    kind, _consumer = downward_longwave_disposition(
        ra_lw_physics=lw, ra_sw_physics=sw, sf_surface_physics=surface)
    if kind in ("consumed", "published"):
        return (CONSTANT_DOWNWARD_LONGWAVE_ACK,)
    return ()


def profile_declared_acknowledgements(profile: str | None) -> tuple[str, ...]:
    """What naming ``profile`` declares, for a route with no TOML to read.

    THE DEFECT THIS CLOSES.  The constant-longwave declaration is written
    into the case TOML by the configuration door, and a WRF namelist has
    no field for it -- so a shortwave-only suite ran on the route that
    reads the TOML and was refused on the route that reads the namelist,
    although the two routes run the same physics from the same profile.
    Naming the suite IS the declaration, exactly as it already is when a
    materialized experiment inherits it, so a namelist-routed run carries
    it through the profile it names and the same load guard reads it from
    the same ``[experiment].acknowledgements`` array on both routes.

    An unknown or unnamed profile declares nothing.
    """

    if profile is None:
        return ()
    try:
        switches = single_domain_runtime_switches(profile)
    except (KeyError, ValueError):
        return ()
    return settings_declared_acknowledgements(switches)


def constant_longwave_refusal(
        domains, *, acknowledgements: tuple[str, ...] = ()) -> str | None:
    """Why this real case may not fabricate its downward longwave, or None.

    THE constant-GLW guard, and the companion to
    :func:`nocturnal_radiation_refusal` -- same front door, same load,
    different question.  The nocturnal guard asks whether the WINDOW is
    survivable; this one asks whether the downward longwave EXISTS.

    Refuses when any domain's selectors classify as ``"consumed"`` or
    ``"published"`` under :func:`downward_longwave_disposition` -- that
    is EXACTLY the set ``woof.core.physics.initialize_physics`` refuses
    at initialize time, so a config either fails here, at load, or runs;
    it can never pass the door and die mid-preparation.  ``"consumed"``:
    a land-surface scheme reads GLW every surface step and nothing
    computes it -- ``woof/core/dudhia.py`` is shortwave-only and
    returns the array it was handed -- so the land surface integrates
    one frozen number for the whole forecast.  Through 1.8.7 that number
    was :func:`woof.core.physics.initialize_physics`'s ``glw=300.0``
    default -- 269.7 K of radiative equilibrium under a warm-sector night
    that wants about 410 W m-2 and 291.6 K.  ``"published"``: no land
    surface reads it, but shortwave keeps the radiation slot active, so
    the fabricated constant is written to every wrfout frame as if it
    were a flux somebody computed.

    WRF v4.6.1 refuses the shortwave-on half of this outright: with
    ``ra_sw_physics > 0`` its ``radiation_driver`` reaches ``lwrad_select``
    (``phys/module_radiation_driver.F:1839``), which has no ``CASE (0)``,
    and calls ``wrf_error_fatal`` at ``:2245``.  With both streams off it
    leaves GLW at 0.0 W m-2 and lets the land surface consume that.
    woof offers a third answer -- a DECLARED constant -- and this is
    where the declaration is required.

    :data:`CONSTANT_DOWNWARD_LONGWAVE_ACK` in the config's
    ``[experiment].acknowledgements`` declares it and lifts the refusal;
    the nocturnal token does not, and silence does not.

    ``domains`` is an iterable of per-domain RunConfig-like objects.
    """

    from woof.config import radiation_scheme_ids

    if CONSTANT_DOWNWARD_LONGWAVE_ACK in acknowledgements:
        return None
    affected = []
    for run in domains:
        lw, sw = radiation_scheme_ids(run)
        scheme = int(getattr(run, "sf_surface_physics", 0))
        kind, consumer = downward_longwave_disposition(
            ra_lw_physics=lw, ra_sw_physics=sw, sf_surface_physics=scheme)
        if kind in ("consumed", "published"):
            affected.append((int(getattr(run, "grid_id", 0)), sw, scheme,
                             kind, consumer))
    if not affected:
        return None
    grid_ids = ", ".join(str(grid_id) for grid_id, *_ in affected)
    _, sw, scheme, kind, consumer = affected[0]
    shortwave = ("with shortwave still ON (ra_sw_physics "
                 f"{sw})" if sw else "with radiation entirely OFF")
    if kind == "consumed":
        # The classifier already named the consumer, so take its word for
        # it rather than subscripting the table a second time.  A second
        # lookup is a second definition of "consuming", and the moment
        # they disagree -- a scheme the classifier learns to call consuming
        # before this table lists it -- the door raises KeyError instead of
        # refusing.  The engine (woof.core.physics._resolve_initial_glw)
        # and the receipt (woof.runtime.downward_longwave_source) read the
        # returned name for exactly this reason.
        surface = consumer or "a GLW-consuming land-surface scheme"
        exposure = (
            f"domain(s) {grid_ids} run {surface} "
            f"(sf_surface_physics {scheme}) with ra_lw_physics 0 -- no "
            f"longwave scheme, {shortwave} -- so their downward longwave "
            "would be a fixed 300 W m-2 for the whole forecast rather "
            "than a computed flux.")
    else:
        exposure = (
            f"domain(s) {grid_ids} run ra_lw_physics 0 -- no longwave "
            f"scheme, {shortwave} -- with no land-surface scheme "
            f"(sf_surface_physics {scheme}), so nothing integrates GLW, "
            "but the active radiation slot publishes the fabricated "
            "fixed 300 W m-2 as the GLW row of every wrfout frame.")
    return layered(
        exposure + "  Give the run real longwave (ra_lw_physics = 4 "
        "with ra_sw_physics = 4, which is what every nocturnally valid "
        f"shipped profile does -- e.g. {MORRISON_PROFILE_ID}), or, if "
        "the fixed longwave is the experiment, declare it by adding "
        f'acknowledgements = ["{CONSTANT_DOWNWARD_LONGWAVE_ACK}"] to '
        "[experiment]",
        "Nothing computes GLW when ra_lw_physics is 0: woof's Dudhia "
        "adapter is shortwave-only and returns the GLW array it was "
        "given, unchanged, on every radiation call.  A land-surface "
        "scheme then integrates that one number for the entire run.  "
        "Radiative equilibrium at 300 W m-2 is 269.7 K (25.8 F) while a "
        "Gulf-coast October night runs near 410 W m-2, or 291.6 K "
        "(65.2 F); the deficit craters skin temperature, collapses the "
        "surface saturation humidity with it, and drives 2 m dewpoints "
        "far below the airmass.  That is a shipped user report, not a "
        "hypothetical.  WRF v4.6.1 does not offer this pairing at all "
        "with shortwave on -- its lwrad_select has no lw=0 case and "
        "calls wrf_error_fatal (phys/module_radiation_driver.F:2245) -- "
        "and with both streams off it hands the land surface 0.0 W m-2 "
        "instead.  The acknowledgement is config-side (not --ack) "
        "because the refusal happens at config load, before any runner "
        "flag is read; it is a SEPARATE token from the nocturnal one "
        "because 'this window has night in it' and 'this flux is "
        "fabricated' are different claims.")


def solar_elevation_deg(when, lat_deg: float, lon_deg: float) -> float:
    """Approximate solar elevation (degrees) at a naive-UTC instant.

    Deliberately stdlib-only and coarse: subsolar declination by
    Cooper's formula (error under ~0.5 degrees) and local MEAN solar
    time with the equation of time omitted (under ~4 minutes of clock,
    ~1 degree of elevation).  Near the terminator that bounds the total
    error to roughly ten minutes of clock time -- this is a "does the
    window include night" instrument, not an ephemeris, and callers must
    not read sunrise/sunset times out of it.
    """

    import math

    day_of_year = when.timetuple().tm_yday
    declination = math.radians(
        -23.44 * math.cos(math.radians(360.0 / 365.0 * (day_of_year + 10))))
    utc_hours = when.hour + when.minute / 60.0 + when.second / 3600.0
    local_solar_hours = (utc_hours + lon_deg / 15.0) % 24.0
    hour_angle = math.radians(15.0 * (local_solar_hours - 12.0))
    latitude = math.radians(lat_deg)
    sin_elevation = (
        math.sin(latitude) * math.sin(declination)
        + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle))
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_elevation))))


#: Sampling cadence for the night-window scan.  Fifteen minutes cannot
#: step over a night at any latitude that has one, and 48 h of samples
#: is 192 sine evaluations -- cheap enough for the wizard's sizing loop,
#: which loads every candidate through the same guard.
_NIGHT_SCAN_SECONDS = 900.0


def first_local_night_time(start_time, run_seconds: float, *,
                           ref_lat: float, ref_lon: float):
    """First sampled instant of the window with the sun below the horizon.

    ``None`` when every sample has the sun up (a daytime window, or
    polar day).  Sampled every :data:`_NIGHT_SCAN_SECONDS` from start to
    end inclusive, at the experiment's reference point -- the wizard
    centers every domain of a tree on it, so it stands for the tree.
    Geometric horizon (elevation < 0), consistent with the instrument's
    stated ~ten-minute resolution at the terminator.
    """

    from datetime import timedelta

    elapsed = 0.0
    total = float(run_seconds)
    while True:
        when = start_time + timedelta(seconds=min(elapsed, total))
        if solar_elevation_deg(when, ref_lat, ref_lon) < 0.0:
            return when
        if elapsed >= total:
            return None
        elapsed += _NIGHT_SCAN_SECONDS


def nocturnal_radiation_refusal(
        domains, *, start_time, run_seconds: float,
        ref_lat: float, ref_lon: float,
        acknowledgements: tuple[str, ...] = ()) -> str | None:
    """Why this real-case window may not run its radiation pairing, or None.

    THE nocturnal-radiation guard, one spelling for every front door:
    :func:`woof.experiment.build_experiment` calls it while loading any
    real experiment (a config with a ``[projection]`` table), so ``woof
    run``, ``woof go``, ``woof check``, both prepared runners, the DA
    drivers and the wizard's own candidate loop all refuse the same way.

    Refuses when any domain resolves shortwave ON with longwave OFF
    (``ra_sw_physics > 0`` and ``ra_lw_physics == 0``) and the window
    includes local night at the reference point: the shortwave scheme
    heats the surface by day, and at night the surface radiates with no
    downward longwave, so skin temperature and 2 m moisture collapse.
    Delivering :data:`ASYMMETRIC_RADIATION_NOCTURNAL_ACK` in the
    config's ``[experiment].acknowledgements`` declares the validation
    experiment and lifts the refusal; silence does not.

    ``domains`` is an iterable of per-domain RunConfig-like objects.
    """

    from woof.config import radiation_scheme_ids

    if ASYMMETRIC_RADIATION_NOCTURNAL_ACK in acknowledgements:
        return None
    asymmetric = []
    for run in domains:
        lw, sw = radiation_scheme_ids(run)
        if sw > 0 and lw == 0:
            asymmetric.append((int(getattr(run, "grid_id", 0)), lw, sw))
    if not asymmetric:
        return None
    night = first_local_night_time(
        start_time, run_seconds, ref_lat=ref_lat, ref_lon=ref_lon)
    if night is None:
        return None
    sw_names = {1: "Dudhia", 4: "RRTMG-class"}
    grid_ids = ", ".join(str(grid_id) for grid_id, _, _ in asymmetric)
    _, _, sw = asymmetric[0]
    profile = identify_single_domain_profile(domains[0])
    caused_by = (
        f"profile {profile}" if profile is not None
        else "a suite matching no shipped profile")
    # ROUTE-SAFE, BECAUSE ROUTE-AWARE IS NOT AVAILABLE HERE (2026-08-20).
    #
    # A loaded experiment carries no forcing source -- ExperimentConfig
    # has no such field -- so this refusal cannot pick the remedy the
    # active source's route admits, the way the wizard's own nocturnal
    # refusal now does.  What it CAN do is name a suite no registered
    # source's route refuses, computed rather than assumed.  It used to
    # name the gfs/era5 default, which the native HRRR route refuses for
    # cu_physics=1, so a user who took the example met a second refusal.
    from woof.physics_menu import universally_admissible_profile

    example = universally_admissible_profile() or MORRISON_PROFILE_ID
    return layered(
        f"this run's window includes local night (first at "
        f"{night:%Y-%m-%dT%H:%M}Z at {ref_lat:.4g}, {ref_lon:.4g}) while "
        f"domain(s) {grid_ids} run shortwave radiation with longwave OFF "
        f"(ra_sw_physics {sw} = {sw_names.get(sw, 'scheme %d' % sw)}, "
        f"ra_lw_physics 0; {caused_by}).  Choose a nocturnally valid "
        f"profile (both radiation streams on -- e.g. {example}, which "
        f"every registered source's route admits; `woof run-plan "
        f"--physics-profiles` lists what each source admits), or "
        f"declare the "
        f"validation experiment by adding acknowledgements = "
        f'["{ASYMMETRIC_RADIATION_NOCTURNAL_ACK}"] to [experiment].  '
        f"With ra_lw_physics 0 this configuration also FABRICATES its "
        f"downward longwave, which is a second and separate claim, so "
        f"the declaration it needs is both tokens together: "
        f'acknowledgements = ["{ASYMMETRIC_RADIATION_NOCTURNAL_ACK}", '
        f'"{CONSTANT_DOWNWARD_LONGWAVE_ACK}"].  Two claims, two tokens',
        "Shortwave heats the surface by day while no longwave scheme "
        "runs, so after sunset the surface radiates to space with no "
        "downward longwave to balance it: skin temperature craters, the "
        "surface saturation humidity collapses with it, and 2 m "
        "dewpoints read far below the airmass.  This pairing is a "
        "daytime-only configuration; a shipped 48 h case emitted "
        "with it verified exactly this failure.  The acknowledgement is "
        "config-side (not --ack) because the refusal happens at config "
        "load, before any runner flag is read.")


#: Declared-experiment acknowledgement for running a LAND-SURFACE MODEL
#: with radiation switched entirely off (``ra_lw_physics = 0`` AND
#: ``ra_sw_physics = 0``), which means nothing will ever compute the
#: downward longwave that model's energy budget reads.
#:
#: Provenance (2026-08-09).  The nocturnal guard above tests ``sw > 0 and
#: lw == 0``, so a suite with BOTH streams off slid past it, and
#: :func:`woof.core.physics.initialize_physics` attaches no radiation
#: adapter at all in that case (``radiation_active = bool(ra_lw_physics or
#: ra_sw_physics)``).  Noah (``woof/core/noah.py``), Noah-MP
#: (``woof/core/noahmp_runtime.py``) and RUC (``woof/core/ruc.py``) each
#: read ``fields["glw"]`` every surface step regardless, so the whole run's
#: downward longwave was the constructor's seed -- a plausible-looking
#: 300.0 that no scheme produced.  There is no seed any more:
#: :func:`woof.core.physics.initialize_physics` takes ``glw`` with NO
#: DEFAULT and refuses to invent one, so the number a land surface
#: integrates is always one somebody typed.  A surface budget with a
#: declared sky is still not a forecast, so a real case says so out loud
#: or does not run.
#:
#: THIS IS A DIFFERENT CLAIM FROM :data:`CONSTANT_DOWNWARD_LONGWAVE_ACK`,
#: and both are required of the configuration they overlap on -- both
#: radiation streams off, under a land-surface scheme.  This token says
#: "nothing computes my sky at all"; that one says "the downward longwave
#: my land surface integrates is a constant I declared".  Neither implies
#: the other: a shortwave-on run makes only the second claim, and the
#: second is what tells :func:`woof.runtime.declared_constant_glw` which
#: number to hand the engine.
#:
#: Same governance idiom and the same config-side delivery as
#: :data:`ASYMMETRIC_RADIATION_NOCTURNAL_ACK`, for the same reason: the
#: refusal happens at config LOAD, before any front door merges ``--ack``.
RADIATION_OFF_LAND_SURFACE_ACK = "radiation-off-land-surface-v1"


#: Every key a configuration can use to select radiation.  All three are
#: ``[shared]``-only in the experiment schema (a ``[[domain]]`` that
#: names one is refused by name a gate earlier), so "did this file
#: choose radiation at all?" is answerable from one table.
RADIATION_SELECTOR_KEYS = ("ra_physics", "ra_lw_physics", "ra_sw_physics")


def declared_radiation_selectors(shared: Mapping[str, object]) -> tuple[str, ...]:
    """Which radiation selectors a configuration's ``[shared]`` SPELLS.

    Presence, not value: the point is whether the author made a choice,
    not what they chose.  ``RunConfig`` defaults ``ra_physics`` to 0 and
    ``ra_lw_physics``/``ra_sw_physics`` to -1, so by the time a
    ``RunConfig`` exists a file that wrote ``ra_physics = 0`` and a file
    that wrote nothing at all are the same object -- and they are not the
    same mistake.  This is read off the raw table for the same reason
    ``build_experiment`` reads ``declared_map_proj`` there.
    """

    return tuple(key for key in RADIATION_SELECTOR_KEYS if key in shared)


def radiation_off_land_surface_refusal(
        domains, *,
        acknowledgements: tuple[str, ...] = (),
        declared_selectors: tuple[str, ...] | None = None) -> str | None:
    """Why this real case may not run an LSM with no radiation, or None.

    Refuses when any domain resolves BOTH radiation streams off
    (``ra_lw_physics == 0`` and ``ra_sw_physics == 0``) while a
    land-surface model is selected (``sf_surface_physics != 0``).  No
    scheme will ever write ``GLW``, so the land surface integrates
    against a sky that is not a computed quantity for the entire run.

    THE OMISSION CASE IS IN SCOPE AND SAYS SO.  Radiation resolves to off
    by DEFAULT, so a real configuration that simply never named a
    radiation selector is refused on exactly the same physics as one that
    spelled two zeros -- it is the larger class of the two, and it is the
    class that breaks previously-loading files.  ``declared_selectors``
    (from :func:`declared_radiation_selectors`, which reads the raw
    ``[shared]`` table) is how the message tells them apart: EMPTY means
    the file chose nothing, and the refusal reports the absent line
    instead of quoting back two zeros the author never wrote.  ``None``
    means the caller could not say, and takes the spelled wording,
    because claiming someone wrote nothing is a claim that needs
    evidence.  A file spelling only the legacy ``ra_physics = 0`` DID
    choose, and is told so -- which is what the three shipped declaring
    configs do.

    Unlike the nocturnal guard this asks nothing about the clock: a zero
    sky is wrong at noon as well as at midnight, so there is no window to
    scan and no reference point to need.  Delivering
    :data:`RADIATION_OFF_LAND_SURFACE_ACK` in the config's
    ``[experiment].acknowledgements`` declares the experiment and lifts
    the refusal; silence does not.

    ``domains`` is an iterable of per-domain RunConfig-like objects.
    """

    from woof.config import radiation_scheme_ids

    if RADIATION_OFF_LAND_SURFACE_ACK in acknowledgements:
        return None
    domains = list(domains)
    offenders = []
    for run in domains:
        lw, sw = radiation_scheme_ids(run)
        surface = int(getattr(run, "sf_surface_physics", 0) or 0)
        if lw == 0 and sw == 0 and surface != 0:
            offenders.append((int(getattr(run, "grid_id", 0)), surface))
    if not offenders:
        return None
    grid_ids = ", ".join(str(grid_id) for grid_id, _ in offenders)
    _, surface = offenders[0]
    component = land_surface_component_for_selector(surface)
    surface_name = (
        f"sf_surface_physics {surface} = {component}" if component
        else f"sf_surface_physics {surface}")
    profile = identify_single_domain_profile(domains[0])
    caused_by = (
        f"profile {profile}" if profile is not None
        else "a suite matching no shipped profile")
    if declared_selectors is not None and not declared_selectors:
        selectors = (
            "with NO RADIATION SELECTOR SET AT ALL -- this configuration "
            "names none of "
            + ", ".join(RADIATION_SELECTOR_KEYS)
            + ", and radiation defaults to OFF, which resolves to "
            "ra_lw_physics 0 and ra_sw_physics 0 exactly as writing them "
            "would")
        remedy = (
            "Most likely the radiation line is simply missing: name the "
            "suite you meant, with both streams on (e.g. "
            f"{MORRISON_PROFILE_ID}, the wizard's default)")
    else:
        # The RESOLVED pair, which is what the physics reads.  A file may
        # have spelled it as the legacy aggregate (ra_physics = 0); the
        # pair is still the accurate statement of what was selected, and
        # `written` names the key the reader should go looking for.
        written = (
            "" if not declared_selectors
            else " (written here as " + ", ".join(declared_selectors) + ")")
        selectors = ("with radiation switched entirely OFF (ra_lw_physics 0 "
                     f"and ra_sw_physics 0{written})")
        remedy = ("Choose a suite with both radiation streams on (e.g. "
                  f"{MORRISON_PROFILE_ID}, the wizard's default)")
    # The number the DECLARED branch actually runs, single-sourced from
    # the module constant rather than restated: a refusal that steers by
    # a number no reachable run uses is worse than one that gives none.
    declared_constant = f"{DECLARED_CONSTANT_GLW_WM2:g}"
    return layered(
        f"domain(s) {grid_ids} run a land-surface model ({surface_name}) "
        f"{selectors}; {caused_by}.  Nothing computes the downward "
        f"longwave the land surface reads every step.  {remedy}, or "
        f"switch the land surface off too (sf_surface_physics = 0, a "
        f"prescribed skin temperature, which reads no GLW at all), or "
        f"declare the experiment by adding acknowledgements = "
        f'["{RADIATION_OFF_LAND_SURFACE_ACK}"] to [experiment] -- a '
        f"declared run integrates the DECLARED constant "
        f"{declared_constant} W m-2, not zero, which is a SECOND claim "
        f"about this configuration and needs its own token beside the "
        f'first: acknowledgements = ["{RADIATION_OFF_LAND_SURFACE_ACK}", '
        f'"{CONSTANT_DOWNWARD_LONGWAVE_ACK}"].  Two claims, two tokens',
        "A land-surface model closes a surface energy budget: downward "
        "shortwave and downward longwave in, sensible/latent/ground heat "
        "and sigma*T^4 out.  With no radiation scheme attached the two "
        "incoming terms are never computed.  WHAT THE SURFACE THEN "
        "INTEGRATES depends on which way out you take, and the two are "
        "different physics.  Switch the land surface off and GLW reaches "
        "no scheme at all.  Declare the experiment and the land surface "
        f"integrates {declared_constant} W m-2 every step for the whole "
        "run -- woof's declared constant "
        "(woof.physics_compat.DECLARED_CONSTANT_GLW_WM2), not WRF's "
        "answer: WRF sets GLW = 0 for a longwave-free run "
        "(phys/module_physics_init.F:1168-1170 and "
        "phys/module_radiation_driver.F:1719-1722) and woof deliberately "
        "diverges, because a number somebody declared is auditable where "
        "a zero that looks like a measurement is not.  Radiative "
        f"equilibrium at {declared_constant} W m-2 is 269.7 K (25.8 F), "
        "so a declared run is a surface-budget experiment and not a "
        "forecast; what the declaration buys is that its sky is on the "
        "record.  This refusal is config-side (not --ack) because it "
        "happens at config load, before any runner flag is read.")


#: Sentinel for "this caller did not mention the selector at all", which is
#: a different statement from "it named None".
_ABSENT = object()


def _selection_value(settings: Mapping[str, object] | object, name: str):
    value = _selection_value_or_absent(settings, name)
    return None if value is _ABSENT else value


def _selection_value_or_absent(settings: Mapping[str, object] | object,
                               name: str):
    if isinstance(settings, Mapping):
        return settings.get(name, _ABSENT)
    return getattr(settings, name, _ABSENT)


#: The three keys that spell ONE radiation choice.  ``ra_physics = N``
#: means "engine N on both streams" and leaves ``ra_lw_physics`` and
#: ``ra_sw_physics`` at -1, the sentinel for "not stated here"; the split
#: pair states the two streams and leaves ``ra_physics`` at 0.
#: woof.config.radiation_scheme_ids is the rule, and every run-path
#: consumer reaches the resolved pair through it.
_RADIATION_SPELLING_KEYS = ("ra_physics", "ra_lw_physics", "ra_sw_physics")


class _OneSpelling:
    """``settings``, read with some keys answered from a resolved form.

    A read-through view, never a copy: every other selector, parameter
    and array on the wrapped object or mapping answers exactly as it
    did, and only the keys passed in ``resolved`` answer from it.  Used
    for the radiation pair's two spellings and for the cumulus cadence
    a cumulus-off configuration carries dead.
    """

    __slots__ = ("_settings", "_resolved")

    def __init__(self, settings, resolved: Mapping[str, int]):
        self._settings = settings
        self._resolved = dict(resolved)

    def __getattr__(self, name: str):
        resolved = self._resolved
        if name in resolved:
            return resolved[name]
        value = _selection_value_or_absent(self._settings, name)
        if value is _ABSENT:
            raise AttributeError(name)
        return value

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._settings!r})"


def _one_radiation_spelling(settings: Mapping[str, object] | object):
    """``settings`` with the two radiation spellings collapsed into one.

    THE DEFECT THIS CLOSES.  A component option is matched on its
    component's selector keys, and radiation's are ``ra_lw_physics`` and
    ``ra_sw_physics``.  Read raw, those keys carry -1 whenever the
    configuration spelled its choice through the aggregate
    ``ra_physics`` -- which is what the WRF namelist importer emits for
    a coupled pair (woof/namelist_import.py, ``coupled_legacy``) -- so
    the registry resolved the SPELLING and not the engine: an imported
    4/4 configuration matched no option a shipped template declares and
    could never equal its own profile, while an imported 0/0
    configuration was refused for not setting ``ra_physics = 4``.  Both
    spellings run identical radiation, because the engine reaches the
    pair through woof.config.radiation_scheme_ids.

    So the door reads the pair the same way the engine does, once, here.
    woof/ingest/prepared_cache.py canonicalizes the same two spellings
    for cache identity and says so in the same words; this is that rule
    at the capability door.

    Absence is preserved, because absence is a different statement from
    a value: a caller that mentions none of the three keys is not
    selecting radiation at all and is returned unchanged, and a caller
    that names one half of the split pair is returned unchanged so the
    "these keys are chosen together" refusal below keeps its own
    sentence.  A CONTRADICTION -- the aggregate naming one engine while
    the split pair names another -- raises out of
    :func:`woof.config.radiation_scheme_ids` with its own message,
    which is the refusal this door wants for it.
    """

    from types import SimpleNamespace

    from woof.config import radiation_scheme_ids

    raw = {key: _selection_value_or_absent(settings, key)
           for key in _RADIATION_SPELLING_KEYS}
    if all(value is _ABSENT for value in raw.values()):
        return settings
    split = (raw["ra_lw_physics"], raw["ra_sw_physics"])
    if (split[0] is _ABSENT) != (split[1] is _ABSENT):
        return settings
    probe = SimpleNamespace(**{key: value for key, value in raw.items()
                               if value is not _ABSENT})
    lw, sw = radiation_scheme_ids(probe)
    if split == (lw, sw) and raw["ra_physics"] in (0, _ABSENT):
        return settings
    return _OneSpelling(
        settings,
        {"ra_physics": 0, "ra_lw_physics": lw, "ra_sw_physics": sw})


def _one_dead_cumulus_cadence(settings: Mapping[str, object] | object):
    """``settings`` with a cumulus interval no cumulus scheme reads pinned to 0.

    THE DEFECT THIS CLOSES.  ``cudt_minutes`` is the interval between
    cumulus calls, and at ``cu_physics = 0`` there are none:
    woof/core/clock.py builds a cumulus calendar only for cu_physics in
    (1, 3, 16), and woof/core/physics.py takes no cumulus step without
    one, so the value is dead namelist state that reaches no kernel and
    changes no number.  Its two producers spell the dead value
    differently -- the WRF namelist importer omits the key for a
    cumulus-off suite, deliberately (woof/namelist_import.py), so the
    configuration inherits RunConfig's live 5.0, while every shipped
    cumulus-off profile states 0.0 -- and the capability door compared
    them raw.  So a namelist-routed run of a cumulus-off profile was
    refused with ``settings={'cudt_minutes': {'selected': 5.0,
    'expected': 0.0}}``, which is the whole of the difference between
    the two configurations, on a switch neither of them runs.  That was
    every HRRR preparation of the nowcast door's own default profile.

    woof/ingest/prepared_cache.py pins the same key for the same reason
    when it compares two prepared identities, and says so in the same
    words; this is that rule at the capability door, beside
    :func:`_one_radiation_spelling`.

    Absence is preserved: a caller that states no ``cu_physics`` is not
    selecting cumulus at all, and one that states no ``cudt_minutes``
    has nothing to pin.  A configuration that runs a cumulus scheme is
    returned untouched, cadence and all, because there the interval is
    read every step it fires.
    """

    cu_physics = _selection_value_or_absent(settings, "cu_physics")
    cadence = _selection_value_or_absent(settings, "cudt_minutes")
    if cu_physics is _ABSENT or cadence is _ABSENT:
        return settings
    if cu_physics != 0 or cadence == 0.0:
        return settings
    return _OneSpelling(settings, {"cudt_minutes": 0.0})


def selection_values_one_spelling(settings, names):
    """``names`` read off ``settings`` with the dead spellings resolved.

    THE ONE PLACE the two spellings are reconciled for a comparison
    against a profile, so a second door cannot reconcile them
    differently or forget to.  Two rules, each documented on the
    function that carries it:

    * :func:`_one_radiation_spelling` -- the aggregate ``ra_physics``
      and the split ``ra_lw_physics``/``ra_sw_physics`` pair are two
      spellings of one selection, and the engine reaches the pair
      through :func:`woof.config.radiation_scheme_ids` either way;
    * :func:`_one_dead_cumulus_cadence` -- ``cudt_minutes`` at
      ``cu_physics = 0`` is a cumulus interval no cumulus call reads.

    Both were reported as the same defect from the same door: a WRF
    namelist importer and a shipped profile write the same run in
    different words, and a key-by-key comparison read the words.
    """

    view = _one_dead_cumulus_cadence(_one_radiation_spelling(settings))
    return {name: _selection_value(view, name) for name in names}


def _registry_pointer(component_id: str, option_id: str | None = None) -> str:
    pointer = (
        "woof/physics_registry_v2.json#/components/"
        f"{component_id}"
    )
    if option_id is not None:
        pointer += f"/options/{option_id}"
    return pointer


def _resolve_physics_component_options(
        settings: Mapping[str, object] | object,
) -> tuple[dict[str, str], dict[str, Mapping[str, object]]]:
    """Resolve implemented component options from selectors only."""
    from woof.physics_registry import physics_registry

    # ONE spelling of the radiation choice before anything is matched
    # against a selector tuple; see :func:`_one_radiation_spelling`.
    settings = _one_radiation_spelling(settings)
    registry = physics_registry()
    resolved: dict[str, str] = {}
    options_by_component: dict[str, Mapping[str, object]] = {}
    for component_id, raw_component in registry["components"].items():
        component = raw_component
        selector_keys = tuple(component.get("selector_keys", ()))
        if not selector_keys:
            continue
        raw_selected = {
            key: _selection_value_or_absent(settings, key)
            for key in selector_keys
        }
        if all(value is _ABSENT for value in raw_selected.values()):
            # A caller that never mentions a component's selectors is not
            # selecting that component, and resolving one it did not ask
            # about cannot be right.  This matters because a selector can
            # MOVE here: km_opt was a registry parameter with a default
            # until it became components/turbulence's selector key, and
            # every caller that passes an explicit settings mapping --
            # woof.physics_compat.validate_resolved_physics_vertical_levels
            # is the public one -- was written before the component
            # existed and omits it.  Refusing all of them because one key
            # is unmentioned would be a silent contract change on a
            # public API, and it reported itself as "no implemented
            # option for selectors {'km_opt': None}", which names a
            # value nobody wrote.  An object (a RunConfig) always carries
            # its own default, so nothing on the run path is skipped.
            continue
        selected = {
            key: (None if value is _ABSENT else value)
            for key, value in raw_selected.items()
        }
        candidates = []
        for option_id, raw_option in component["options"].items():
            option = raw_option
            selectors = option.get("selectors", {})
            if (isinstance(selectors, Mapping)
                    and set(selectors) == set(selector_keys)
                    and all(selected[key] == selectors[key]
                            for key in selector_keys)):
                candidates.append((option_id, option))
        if not candidates and component_id == "radiation":
            # The runtime composes independently implemented spectra. A
            # paired preset is one spelling of that capability, not a gate.
            spectra = {}
            requirements = {}
            for key, value in selected.items():
                matches = [(name, item) for name, item in component["options"].items()
                           if item.get("implemented") is True
                           and item.get("selectors", {}).get(key) == value
                           and isinstance(value, int) and value >= 0]
                if not matches:
                    break
                name, item = matches[0]
                spectra[key] = {"selector": value, "component_option": name}
                requirements.update(item.get("constraints", {}).get("required_settings", {}))
            if len(spectra) == len(selector_keys):
                option_id = "independent-" + "-".join(str(selected[key]) for key in selector_keys)
                candidates.append((option_id, {
                    "implemented": True, "selectors": selected,
                    "spectra": spectra, "constraints": {"required_settings": requirements},
                    "execution": "woof.core.radiation_composition.make_radiation",
                }))
        if len(candidates) != 1:
            # TWO DIFFERENT FAILURES, two messages.  The comment above
            # fixed the ALL-absent case; a mapping that names SOME of a
            # component's selector keys still landed here and was told it
            # had no option for `{'ra_lw_physics': 4, 'ra_sw_physics':
            # None}` -- a value nobody wrote, printed back as if they
            # had.  Selector keys are deliberately absent from the
            # registry's parameter declarations, so there is no default
            # to fill in and substituting one would be the silent
            # substitution this refusal exists to prevent: the answer is
            # to say which keys are missing and that they are chosen
            # together.
            absent = sorted(
                key for key in selector_keys if raw_selected[key] is _ABSENT)
            if absent:
                named = {key: value for key, value in selected.items()
                         if key not in absent}
                raise PhysicsCapabilityError(
                    f"{_registry_pointer(component_id)} is selected by "
                    f"{list(selector_keys)} TOGETHER, and this request "
                    f"names only {named}: {absent} carry no value here. "
                    "These keys have no registry default on purpose -- an "
                    "option is a whole selector tuple, and filling one in "
                    "would substitute a scheme nobody asked for. Give "
                    "every key a value, or omit the component entirely "
                    "and let the run configuration supply its own")
            implemented = sorted(
                str(option.get("selectors", {}))
                for option in component["options"].values()
                if option.get("implemented") is True)
            raise PhysicsCapabilityError(
                f"{_registry_pointer(component_id)} has no implemented "
                f"option for selectors {selected}; no source/profile "
                "substitution is allowed. Implemented selector tuples for "
                f"this component: {'; '.join(implemented)}")
        option_id, option = candidates[0]
        if option.get("implemented") is not True:
            declaration = option.get("reachability", {})
            blocker = (
                declaration.get("blocker")
                if isinstance(declaration, Mapping) else None
            )
            raise PhysicsCapabilityError(
                f"{_registry_pointer(component_id, option_id)} blocks "
                f"selectors {selected}: "
                f"{blocker or 'component is declared unimplemented'}")
        resolved[component_id] = option_id
        options_by_component[component_id] = option
    return resolved, options_by_component


def validate_physics_capabilities(
        settings: Mapping[str, object] | object,
) -> dict[str, str]:
    """Resolve selectors to implemented registry components, fail closed.

    This check deliberately knows no source id and no profile id.  It answers
    only whether the selected component implementations and their couplings
    exist.  The returned mapping is component id -> option id and is suitable
    for comparing a named template after capability has been established.
    """

    from woof.physics_registry import (
        _conditional_refusal_fires, _conditional_refusals,
        conditional_refusal_sentence, physics_registry)

    settings = _one_radiation_spelling(settings)
    resolved, options_by_component = _resolve_physics_component_options(
        settings)
    parameter_specs = physics_registry().get("parameters", {})

    for component_id, option in options_by_component.items():
        constraints = option.get("constraints", {})
        if not isinstance(constraints, Mapping):
            continue
        required_settings = constraints.get("required_settings", {})
        if isinstance(required_settings, Mapping):
            drift = {
                name: {
                    "selected": _selection_value(settings, name),
                    "required": required,
                }
                for name, required in required_settings.items()
                if _selection_value(settings, name) is not None
                and _selection_value(settings, name) != required
            }
            if drift:
                option_id = resolved[component_id]
                raise PhysicsCapabilityError(
                    f"{_registry_pointer(component_id, option_id)} requires "
                    f"settings {drift}")
        # The multi-valued twin of required_settings, read here for the
        # reason the conditional kind is read here: this resolver and
        # woof.physics_registry.validate_physics_plan are two readers of
        # ONE constraint table, and a kind only one door understands is a
        # constraint that fires or not depending on which door the user
        # came through.
        admitted = constraints.get("admitted_setting_values", {})
        admitted_reasons = constraints.get(
            "admitted_setting_values_reasons", {})
        if not isinstance(admitted_reasons, Mapping):
            admitted_reasons = {}
        if isinstance(admitted, Mapping):
            for name, values in admitted.items():
                if not isinstance(values, list):
                    continue
                observed = _selection_value(settings, name)
                if observed is None:
                    # The registry's own declared default, so this door
                    # and validate_physics_plan answer alike when the
                    # caller never named the setting.
                    spec = parameter_specs.get(name)
                    observed = (spec.get("default")
                                if isinstance(spec, Mapping) else None)
                if observed is None or observed in values:
                    continue
                option_id = resolved[component_id]
                reason = admitted_reasons.get(name)
                detail = (f": {reason}"
                          if isinstance(reason, str) and reason else "")
                raise PhysicsCapabilityError(
                    f"{_registry_pointer(component_id, option_id)} admits "
                    f"{name} in {values!r}, got {observed!r}{detail}")
        requirements = constraints.get("requires_components", {})
        if isinstance(requirements, Mapping):
            for required_component, allowed in requirements.items():
                selected_option = resolved.get(required_component)
                if (not isinstance(allowed, list)
                        or selected_option not in allowed):
                    option_id = resolved[component_id]
                    raise PhysicsCapabilityError(
                        f"{_registry_pointer(component_id, option_id)} "
                        f"requires {required_component} in {allowed}, got "
                        f"{selected_option!r}")
        # The conditional kind, evaluated here for the same reason the
        # other three are: this resolver and
        # woof.physics_registry.validate_physics_plan are two readers of
        # ONE constraint table, and a kind only one of them understands is
        # a constraint that fires or not depending on which door the user
        # came through.  The SEMANTICS live in physics_registry so there is
        # one implementation of them; what is local here is reading the
        # values off a RunConfig-shaped object instead of a resolved
        # settings dict.
        rules = _conditional_refusals(constraints)
        if rules:
            named = {
                name
                for rule in rules
                for name in (rule.get("settings") or {})
            }
            observed = {
                name: _selection_value(settings, name)
                for name in named
                if _selection_value_or_absent(settings, name) is not _ABSENT
            }
            # A rule carrying a ``sources`` clause (audit R-005) cannot
            # fire here and must not: a RunConfig carries no source
            # identity, and inventing one would make this door refuse or
            # admit by guess.  Those rules are evaluated at plan review,
            # which has ``context.source_id``, and by the preparation door
            # that reads the source itself.
            for rule in rules:
                if _conditional_refusal_fires(
                        rule, resolved, observed, parameter_specs):
                    option_id = resolved[component_id]
                    raise PhysicsCapabilityError(
                        f"{_registry_pointer(component_id, option_id)} is "
                        f"refused here: "
                        f"{conditional_refusal_sentence(rule)}")

    # Runtime-only coupled restrictions and measured-width rails remain the
    # executable authority.  They cite the exact WRF/CUDA reason in their
    # blocker receipts and are intentionally applied after registry
    # implementation resolution.
    runtime_selection = {
        name: int(_selection_value(settings, name))
        for name in (
            "mp_physics", "sf_sfclay_physics", "bl_pbl_physics",
            "sf_surface_physics", "num_soil_layers",
        )
    }
    nx = _selection_value(settings, "nx")
    ny = _selection_value(settings, "ny")
    if (isinstance(nx, int) and not isinstance(nx, bool)
            and isinstance(ny, int) and not isinstance(ny, bool)):
        runtime_selection["columns"] = nx * ny
    require_ready_wrf_physics(**runtime_selection)
    return resolved


def conditional_refusals_for(
        settings: Mapping[str, object] | object,
) -> list[dict[str, object]]:
    """Every ``refused_when`` rule the registry fires for these settings.

    The same table and the same evaluator
    :func:`validate_physics_capabilities` raises on, asked to REPORT
    instead.  A front end that has already met the parser's refusal needs
    two things the sentence does not carry -- which option the rule
    belongs to, and the machine-applicable way out -- and re-deriving
    either one by matching substrings of the prose is what
    woof/companion_physics.py did for one scheme: the branch fired on
    "mp_physics=9" and "cloud-optics" appearing in the error, so every
    other scheme with the same refusal got no tailored repair and a
    reworded sentence would have silently dropped the one that existed.

    Each returned row is ``{"component", "option", "reason", "remedy_label",
    "remedy_settings"}``: ``reason`` is the whole sentence a reader is
    shown, remedy included, and ``remedy_settings`` is that remedy as an
    edit, or ``None`` where the rule declares none.  Nothing is raised
    for a draft whose selectors resolve to no implemented option -- there
    are no rules to report about a combination the resolver cannot place,
    and the door that called here already holds its own refusal.
    """

    from woof.physics_registry import (
        _conditional_refusal_fires, _conditional_refusals,
        conditional_refusal_remedy, conditional_refusal_sentence,
        physics_registry)

    try:
        resolved, options_by_component = _resolve_physics_component_options(
            settings)
    except PhysicsCapabilityError:
        return []
    parameter_specs = physics_registry().get("parameters", {})
    fired: list[dict[str, object]] = []
    for component_id, option in options_by_component.items():
        constraints = option.get("constraints", {})
        if not isinstance(constraints, Mapping):
            continue
        for rule in _conditional_refusals(constraints):
            named = {name for name in (rule.get("settings") or {})}
            observed = {
                name: _selection_value(settings, name)
                for name in named
                if _selection_value_or_absent(settings, name) is not _ABSENT
            }
            if not _conditional_refusal_fires(
                    rule, resolved, observed, parameter_specs):
                continue
            fired.append({
                "component": component_id,
                "option": resolved[component_id],
                "reason": conditional_refusal_sentence(rule),
                "remedy_label": rule.get("remedy_label"),
                "remedy_settings": conditional_refusal_remedy(rule),
            })
    return fired


def validate_resolved_physics_vertical_levels(
        settings: Mapping[str, object] | object, *,
        p_top: float | None = None,
) -> dict[str, object]:
    """Aggregate component-owned first-call vertical bounds.

    ``p_top=None`` performs the RunConfig-only checks.  Experiment
    preparation calls again with its authoritative model-top pressure so the
    radiation adapters can include their WRF above-model cap layers.
    """

    resolved, options_by_component = _resolve_physics_component_options(
        settings)
    raw_nz = _selection_value(settings, "nz")
    if isinstance(raw_nz, bool):
        raise PhysicsVerticalPreflightError("nz must be an integer")
    try:
        nz = int(raw_nz)
    except (TypeError, ValueError):
        raise PhysicsVerticalPreflightError(
            f"nz must be an integer, got {raw_nz!r}") from None

    checks: list[dict[str, object]] = []
    violations: list[str] = []

    def bounded(
            label: str,
            bounds: tuple[int | None, int | None],
    ) -> None:
        minimum, maximum = bounds
        checks.append({
            "component": label,
            "model_levels": nz,
            "minimum": minimum,
            "maximum": maximum,
        })
        # One spelling, shared with the launcher-side refusals so a user who
        # somehow reaches the first-call rejection reads the same sentence.
        from woof.physics_vertical_contract import (
            outside_vertical_bounds, vertical_bounds_wording)
        if outside_vertical_bounds(nz, bounds):
            violations.append(f"{label} requires "
                              f"{vertical_bounds_wording(bounds)}, got nz={nz}")

    from woof.physics_registry import CONSUMER_ROWS_KEY
    from woof.physics_vertical_contract import (
        MAX_LEGACY_LONGWAVE_LAYERS,
        MAX_LEGACY_SHORTWAVE_LAYERS,
        MAX_RRTMGP_LAYERS,
        legacy_radiation_layer_counts,
        rrtmgp_above_model_layer_counts,
    )

    # EVERY implemented component option, from the REGISTRY's own row for
    # it (``consumers.vertical_level_bounds``, generated from the contract
    # constants by tools/build_registry.py) rather than from an if/elif
    # chain here.  Two defects this closes: the chain once covered cumulus,
    # radiation and microphysics but not the PBL, so a 130-level YSU
    # configuration passed `woof check` and both preparation stages and
    # died on the first physics call; and it later covered six of nine
    # microphysics schemes and two of three cumulus schemes, so a WDM6,
    # Milbrandt-Yau, P3 or New Tiedtke configuration was never asked at
    # all and met its bound at the first call the same way.  A scheme with
    # no row is now a plan-review refusal (woof.physics_registry.
    # consumer_row_gaps), not a silent skip.
    for component_id in ("pbl", "cumulus", "microphysics"):
        option = options_by_component.get(component_id)
        if not isinstance(option, Mapping):
            continue
        rows = option.get(CONSUMER_ROWS_KEY)
        window = rows.get("vertical_level_bounds") if isinstance(rows, Mapping) else None
        if isinstance(window, Mapping):
            bounded(str(window["label"]),
                    (window.get("minimum"), window.get("maximum")))

    from types import SimpleNamespace
    from woof.config import radiation_scheme_ids
    radiation_settings = (SimpleNamespace(**settings) if isinstance(settings, Mapping) else settings)
    pair = radiation_scheme_ids(radiation_settings)
    if 4 in pair:
        top_pressure = 0.0 if p_top is None else float(p_top)
        if rrtmg_variant(settings) == RRTMG_VARIANT_LEGACY:
            lw_layers, sw_layers = legacy_radiation_layer_counts(nz, top_pressure)
            for index, kind, total, maximum in (
                    (0, "longwave", lw_layers, MAX_LEGACY_LONGWAVE_LAYERS),
                    (1, "shortwave", sw_layers, MAX_LEGACY_SHORTWAVE_LAYERS)):
                if pair[index] != 4:
                    continue
                checks.append({"component": f"legacy RRTMG {kind}",
                    "model_levels": nz, "above_model_layers": total - nz,
                    "total_layers": total, "maximum": maximum})
                if maximum is not None and total > maximum:
                    violations.append(f"legacy RRTMG {kind} requires model plus cap layers "
                        f"<= {maximum}, got {nz}+{total - nz}={total}")
        else:
            lw_upper, sw_upper = rrtmgp_above_model_layer_counts(top_pressure)
            for index, kind, upper in ((0, "longwave", lw_upper), (1, "shortwave", sw_upper)):
                if pair[index] != 4:
                    continue
                total = nz + upper
                checks.append({"component": f"RTE+RRTMGP {kind}",
                    "model_levels": nz, "above_model_layers": upper,
                    "total_layers": total, "maximum": MAX_RRTMGP_LAYERS})
                if total > MAX_RRTMGP_LAYERS:
                    violations.append(f"RTE+RRTMGP {kind} requires model plus cap layers "
                        f"<= {MAX_RRTMGP_LAYERS}, got {nz}+{upper}={total}")

    if violations:
        raise PhysicsVerticalPreflightError(
            "resolved physics vertical preflight failed:\n  - "
            + "\n  - ".join(violations))
    return {
        "schema": "gpuwm-resolved-physics-vertical-preflight-v1",
        "model_levels": nz,
        "p_top_pa": p_top,
        "resolved_components": dict(sorted(resolved.items())),
        "checks": checks,
    }


def validate_single_domain_physics_profile(
        profile: str,
        *,
        config: Mapping[str, object] | object | None = None,
        expert_acknowledgements: tuple[str, ...] = (),
        acknowledgement_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate one named template through registry/runtime capabilities.

    A named profile supplies the intended immutable product.  The executable
    decision remains selector-based: when ``config`` is present its selectors
    are resolved first, so an unfinished piece reports its cited capability
    blocker rather than being hidden behind a profile-name mismatch.
    """

    from woof.physics_registry import (
        canonical_template_id, physics_registry, registry_physics_receipt,
        registry_sha256)

    registry = physics_registry()
    profile = canonical_template_id(profile, registry)
    template = registry["templates"].get(profile)
    if not isinstance(template, Mapping):
        raise ValueError(
            f"physics profile {profile!r} is not a registered fixed template")
    expected = single_domain_runtime_switches(profile)
    selected_settings = expected if config is None else config
    resolved_components = validate_physics_capabilities(selected_settings)
    expected_components = dict(template.get("components", {}))
    if resolved_components != expected_components:
        raise ValueError(
            f"selected physics differs from profile {profile!r}: "
            f"components={resolved_components}, expected={expected_components}")
    if config is not None:
        # Compared through the one spelling, for the reason the
        # capability resolution above uses it: a configuration that
        # reached this profile's two radiation streams through the
        # aggregate selector has not asked for different physics, and a
        # key-by-key comparison refused a run that is bit for bit the
        # run the profile names.
        # ... and through the one cumulus cadence, for the same reason:
        # a cumulus-off configuration that carries RunConfig's live
        # interval has not asked for different physics, because nothing
        # reads that interval without a cumulus scheme to call.
        compared = selection_values_one_spelling(config, expected)
        drift = {
            name: {"selected": compared[name], "expected": value}
            for name, value in expected.items()
            if compared[name] != value
        }
        if drift:
            raise ValueError(
                f"selected physics differs from profile {profile!r}: "
                f"settings={drift}")

    required_acknowledgements: dict[str, list[str]] = {}
    for route_id, route in registry["runner_routes"].items():
        expert = route.get("expert_template_ids", {})
        if not isinstance(expert, Mapping):
            continue
        if any(
                isinstance(template_ids, list) and profile in template_ids
                for template_ids in expert.values()):
            acknowledgement = route.get("expert_acknowledgement_id")
            if isinstance(acknowledgement, str):
                required_acknowledgements.setdefault(
                    acknowledgement, []).append(route_id)
    acknowledged = set(expert_acknowledgements)
    missing = sorted(set(required_acknowledgements) - acknowledged)
    if missing:
        selectors = {
            key: _selection_value(selected_settings, key)
            for component in registry["components"].values()
            for key in component.get("selector_keys", ())
        }
        tuple_text = ", ".join(
            f"{key}={selectors[key]!r}" for key in sorted(selectors))
        for acknowledgement in missing:
            warn(
                f"d01 resolved physics tuple ({tuple_text}) selects expert "
                f"profile {profile!r}; running with its evidence advisory "
                f"unacknowledged; add {_ack_instruction(acknowledgement)} "
                "to silence this warning",
                why="The selected physics passes the runtime capability "
                    "checks. Profile evidence and throughput advisories "
                    "do not change which physics can execute.")

    # The selectors as the engine reads them (the one spelling): a run
    # that chose its radiation through ra_physics = 4 reported its two
    # streams as -1/-1 here, beside a resolved profile saying 4/4 legacy
    # RRTMG (rolling tests, HRRR recipe preparation report).
    selector_names = [key for component in registry["components"].values()
                      for key in component.get("selector_keys", ())]
    selectors = selection_values_one_spelling(selected_settings, selector_names)
    used, provenance = _acknowledgement_receipt(
        acknowledged, set(required_acknowledgements),
        acknowledgement_provenance)
    return {
        "schema": "gpuwm-front-door-physics-selection-v1",
        "profile": profile,
        "registry_sha256": registry_sha256(registry),
        "registry_physics": registry_physics_receipt(
            registry, options=resolved_components, profile=profile),
        "components": resolved_components,
        "selectors": selectors,
        "resolved": expected,
        "acknowledgements": used,
        "acknowledgement_provenance": provenance,
        "governance": {
            "state": ("registry-expert-template" if required_acknowledgements
                      else "registry-template"),
            "required_acknowledgements": sorted(required_acknowledgements),
            "acknowledged": not missing,
        },
        "maturity": template.get("maturity"),
    }


#: Verification-status vocabulary for the REPORTED physics metadata.
#:
#: Owner ruling (2026-07-31): the physics-suite choice is the user's.
#: The single-domain profile whitelist is gone as a gate; what remains
#: of it is this status, computed from the same profile data that used
#: to feed the whitelist and STATED in receipts and ``--explain`` --
#: never used to refuse a suite the engine implements.
VERIFICATION_WRF_VERIFIED = "wrf-verified"
VERIFICATION_SUPPORTED = "supported-not-wrf-verified"
#: A suite selecting at least one EXPERIMENTAL option.  Distinct from
#: "supported": that word promises a WRF comparison that is outstanding,
#: and an ArWen-only scheme has none to be outstanding.
VERIFICATION_EXPERIMENTAL = "experimental-not-wrf-verified"

#: The maturity rung whose options are experimental.  Read off the
#: registry ladder rather than spelled twice.
_EXPERIMENTAL_MATURITY = "experimental-runtime"

#: The SECOND, independent trigger for the experimental warning: an
#: option the registry declares has no WRF counterpart at all.
#:
#: The conformance ladder measures distance from WRF, so an
#: ArWen-original scheme cannot climb it and sits at
#: 'implemented-unverified' permanently -- the same rung as a
#: WRF-transcribed port whose forecast comparison is merely outstanding.
#: Keying the warning on the rung alone would therefore make the two
#: indistinguishable and silence the warning for exactly the option that
#: needs it most.  The declaration is what separates them.
_NO_WRF_COUNTERPART = "wrf_counterpart"
VERIFICATION_STATUS_SCHEMA = "gpuwm-physics-verification-status-v1"

#: A historical label; current exact-suite evidence also needs a scope.
_WRF_VERIFIED_MATURITY = "wrf-matched-run"


def _experimental_reason(option) -> str | None:
    """Why this option warns, or ``None`` when it does not.

    Two independent triggers, and the reason differs because what the
    reader must not conclude differs.  A table-bound runtime IS a WRF
    scheme whose comparison is outstanding; an ArWen-original closure
    has no comparison to be outstanding, and telling a user it is
    merely "not WRF-verified" invites them to wait for a verification
    that will never arrive.
    """
    counterpart = option.get(_NO_WRF_COUNTERPART)
    if isinstance(counterpart, Mapping) and counterpart.get("exists") is False:
        if counterpart.get("implementation_origin") == "upstream-port":
            return "experimental, upstream port with no WRF counterpart"
        return "experimental, WOOF-original with no WRF counterpart"
    if option.get("maturity") == _EXPERIMENTAL_MATURITY:
        return "experimental, not WRF-verified"
    return None


def experimental_component_labels(run_config, registry=None):
    """Labels of every EXPERIMENTAL component option this config selects.

    Returns a sorted tuple, empty when the suite is entirely
    non-experimental.  Reads the registry rather than naming any scheme:
    a second experimental option added later is surfaced by registering
    it, not by editing this function.
    """
    return tuple(sorted(
        label for label, _ in _experimental_components(run_config, registry)))


def _experimental_components(run_config, registry=None):
    """(label, reason) for every experimental option this config selects."""
    from woof.physics_registry import physics_registry

    if registry is None:
        registry = physics_registry()
    # Matched PER COMPONENT off its own selectors, deliberately, rather
    # than through a whole-suite capability resolve: that resolve fails
    # as a unit, so an unrelated mismatch elsewhere in the suite would
    # silently swallow this warning -- and a warning that disappears
    # when something else is wrong is worse than no warning.
    found = []
    for component in registry.get("components", {}).values():
        if not isinstance(component, Mapping):
            continue
        for option_id, option in component.get("options", {}).items():
            if not isinstance(option, Mapping):
                continue
            reason = _experimental_reason(option)
            if reason is None:
                continue
            selectors = option.get("selectors") or {}
            if selectors and all(
                    getattr(run_config, key, None) == value
                    for key, value in selectors.items()):
                found.append((str(option.get("label") or option_id), reason))
    return found


def experimental_selection_sentence(run_configs, registry=None) -> str | None:
    """The one warn-not-block sentence for a run's experimental options.

    ``None`` when nothing experimental is selected.  Takes an ITERABLE
    of run configs rather than one, because a domain tree has several
    and the reader is owed the sentence ONCE for the run rather than
    once per nest -- and because the two runners must not drift into
    saying different things about the same closure.  Every product
    surface that prints this sentence gets it from here; the string is
    defined once.
    """

    clauses: dict[str, str] = {}
    for run_config in run_configs:
        for label, reason in _experimental_components(run_config, registry):
            clauses.setdefault(label, reason)
    if not clauses:
        return None
    # An experimental option is not "supported, not yet WRF-verified"
    # -- that phrasing promises a WRF comparison that is merely
    # outstanding.  For a scheme WRF does not have, no such comparison
    # exists or can, and saying otherwise would be the most misleading
    # sentence the product prints.  Warn-not-block: this is the
    # wording, never a refusal.
    return ("physics: "
            + "; ".join(f"{label}: {clauses[label]}"
                        for label in sorted(clauses))
            + " -- the run continues.")


def single_domain_verification_status(run_config) -> dict[str, object]:
    """WRF-verification evidence for one run config, as reported metadata.

    Never a gate.  A switch-exact match against a shipped profile
    carries that profile's registry-template maturity; a suite matching
    no profile may still match a registered template at COMPONENT level
    (the wizard's default suite does), which is named without being
    claimed as switch-level evidence.  The ``sentence`` is the one line
    product surfaces print -- detail stays in this receipt.
    """

    from woof.physics_registry import canonical_maturity, physics_registry

    registry = physics_registry()
    matched = identify_single_domain_profile(run_config)
    maturity = None
    verification_scope = None
    if matched is not None:
        template = registry["templates"].get(matched)
        if isinstance(template, Mapping):
            maturity = canonical_maturity(template.get("maturity"), registry)
            verification_scope = template.get("verification_scope")
    component_match: dict[str, object] | None = None
    if matched is None:
        try:
            resolved = validate_physics_capabilities(run_config)
        except (PhysicsCapabilityError, TypeError, ValueError):
            resolved = None
        if resolved is not None:
            for template_id, template in registry["templates"].items():
                if (isinstance(template, Mapping)
                        and dict(template.get("components", {}))
                        == resolved):
                    component_match = {
                        "template": template_id,
                        "maturity": canonical_maturity(template.get("maturity"), registry),
                        "scope": "components-only-not-switch-level",
                    }
                    break
    verified = (matched is not None and maturity == _WRF_VERIFIED_MATURITY
                and verification_scope == "current-matched-run")
    experimental = experimental_component_labels(run_config, registry)
    if experimental:
        # One definition, in experimental_selection_sentence, so this
        # surface and the domain-tree runner cannot drift apart.
        sentence = experimental_selection_sentence([run_config], registry)
    elif verified:
        sentence = (
            f"physics: {matched} carries WRF-verification evidence "
            f"(registry maturity {maturity!r}).")
    elif matched is not None:
        sentence = (
            f"physics: {matched} is supported, not yet WRF-verified "
            f"(registry maturity {maturity!r}); the run continues.")
    else:
        sentence = (
            "physics: this suite is supported, not yet WRF-verified; "
            "the run continues.")
    return {
        "schema": VERIFICATION_STATUS_SCHEMA,
        "status": (
            VERIFICATION_EXPERIMENTAL if experimental
            else VERIFICATION_WRF_VERIFIED if verified
            else VERIFICATION_SUPPORTED),
        "matched_profile": matched,
        "matched_profile_maturity": maturity,
        "matched_profile_verification_scope": verification_scope,
        "component_matched_template": component_match,
        "experimental_components": list(experimental),
        "sentence": sentence,
    }


def single_domain_physics_selection(
        config,
        *,
        profile: str | None = None,
        expert_acknowledgements: tuple[str, ...] = (),
        acknowledgement_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The single-domain front-door selection receipt, profile optional.

    Owner ruling (2026-07-31): any physics suite the engine implements
    runs on the prepared single-domain route.  A profile the caller
    NAMED still gates -- a gate asked for is not a gate to drop, and it
    is how "this config IS that shipped suite" stays assertable -- but
    an unnamed config is governed exactly the way the domain-tree route
    has always governed a one-node tree: engine-valid selectors, the
    registry's source-neutral tuple governance, and a recorded blocker
    (not a refusal) where the registry has no spelling for the tuple.

    This is THE one spelling of the single-domain selection decision,
    and its callers are the whole point: the GFS front door
    (:func:`woof.gfs_direct.front_door_physics_selection`), the direct
    exporter's profileless contract
    (:func:`woof.wrf_direct.export_prepared_wrf` with
    ``experiment_config_suite=True``), and the prepared-forecast
    runner's own tuple governance and proof recompute
    (:mod:`woof.prepared_single_domain_forecast`).  All of them compute
    THIS receipt from the hash-bound experiment config, which is what
    keeps "the physics executed is the physics prepared"
    byte-comparable across every seam.
    """

    if profile is not None:
        return validate_single_domain_physics_profile(
            profile, config=config,
            expert_acknowledgements=expert_acknowledgements,
            acknowledgement_provenance=acknowledgement_provenance)
    # The reachability union here spans EVERY implemented route's
    # declared templates, not only the domain-tree route's: the
    # registry declares RUC among this route's own shipped products
    # (ERA5), and a shipped product must not be governed as an outside
    # tuple at its own runner.  Expert templates keep their published
    # acknowledgement from whichever route declares them.
    return multi_domain_physics_selection(
        {1: config},
        expert_acknowledgements=expert_acknowledgements,
        acknowledgement_provenance=acknowledgement_provenance,
        governance_route_modes=("experiment-per-domain", "fixed-template"))


# RETIRED: ``_ROUTE_FOR_SOURCE`` and ``offered_land_surfaces`` stood here
# and were deleted with nothing to replace them.  They were the residue of
# a preparation-time gate that refused a land-surface scheme for not
# appearing in a source's declared TEMPLATE list -- template membership is
# a maturity catalogue, never a runtime requirement, so the refusal never
# named a breakage a user could act on.  The physical failure it proxied
# (a GFS-initialised RUC run dying on its first surface call, `mavail must
# be finite`) was fixed at source instead: every door now routes soil
# categories through woof/ingest/soil.py door_reconciled_soil_category,
# the way real.exe reconciles that shoreline column, and the gate, its
# front-door caller and its menu rule went with the fix.  The table and
# this function were what survived it -- no caller anywhere in the tree,
# and a docstring still claiming enforcement that had already been
# removed.
#
# NO REPLACEMENT GATE.  What replaced it is default-on and already here:
# scheme/soil-geometry coherence at config load, each door's own runtime
# field and category checks (which fail closed on an unrouted selector),
# and the template-route-evidence WARNING at plan review.  A future
# per-source land-surface refusal must key on a runtime REQUIREMENT -- a
# field the scheme reads that this source does not carry, named in the
# message -- and never on template membership.  The one that exists does:
# woof.physics_menu.source_soil_blocker asks the source's declared soil
# geometry the question RUC's soil ingest asks at preparation, so a RUC
# suite on a source that publishes one soil layer is refused before the
# download, in that ingest's own words.
#
# The registry is the one authority on which land surfaces a route
# offers: ``runner_routes.<route>.allowed_component_options.land_surface``,
# read at plan review by ``validate_physics_plan`` (AUDIT R-022).


def land_surface_component_for_selector(value) -> str | None:
    """The registry's name for an ``sf_surface_physics`` value.

    Resolved from the one selector rather than from a whole-suite
    capability resolution, because the two answer different questions.
    Resolving the suite can fail for a reason that has nothing to do
    with the land surface -- the committed two-domain descriptor selects
    a radiation spelling the registry has no option for -- and a route
    gate that quietly skips those configurations is a gate with a hole
    in exactly the shape of the configurations nobody has classified.
    """

    from woof.physics_registry import physics_registry

    if value is None or isinstance(value, bool):
        return None
    try:
        selector = int(value)
    except (TypeError, ValueError):
        return None
    options = physics_registry()["components"]["land_surface"]["options"]
    for option_id, option in options.items():
        if not isinstance(option, Mapping):
            continue
        selectors = option.get("selectors", {})
        if (isinstance(selectors, Mapping)
                and selectors.get("sf_surface_physics") == selector):
            return option_id
    return None


#: The domain-tree route's front-door physics receipt.  A DISTINCT
#: schema from the single-domain one because it records a different
#: decision: a tree has no single profile, it has one resolved selector
#: set per domain, and giving both shapes one schema id would make a
#: consumer guess which it was handed.
MULTI_DOMAIN_SELECTION_SCHEMA = (
    "gpuwm-front-door-physics-selection-multi-domain-v1")


def multi_domain_physics_selection(
        domain_settings: Mapping[int, Mapping[str, object] | object],
        *,
        profile: str | None = None,
        expert_acknowledgements: tuple[str, ...] = (),
        acknowledgement_provenance: Mapping[str, object] | None = None,
        governance_route_modes: tuple[str, ...] = ("experiment-per-domain",),
) -> dict[str, object]:
    """Record and govern a domain tree's resolved physics tuple.

    The domain-tree runner never had a profile whitelist and does not
    grow one here, in either of its two possible spellings -- and since
    the 2026-07-31 owner ruling this same governance also admits an
    UNNAMED single-domain configuration (a one-node tree) on the
    prepared single-domain route, via
    :func:`single_domain_physics_selection`.

    Every domain has already passed the executable selector checks.  This
    second gate asks a different, registry-owned governance question about
    the complete resolved component tuple.  The tuple is compared with the
    union of registry-declared tree templates and component overrides across
    every source, so source identity can neither admit nor refuse it.
    Registry-normal tuples are unchanged, expert-template tuples retain their
    published acknowledgement, and a tuple outside the declared union needs
    the authority-level acknowledgement named in the refusal.

    ``profile`` is honoured when the caller named one explicitly, and
    then it does gate -- a gate asked for is not a gate to drop.  It
    binds the ROOT only: the wizard's own nested emission of a shipped
    profile turns cumulus off on the inner domains, so demanding the
    profile of every domain would refuse the very configuration the
    product tells a user to write.
    """

    from woof.physics_registry import (
        physics_registry, registry_physics_receipt, registry_sha256)

    registry = physics_registry()
    selector_keys = tuple(
        key
        for component in registry["components"].values()
        for key in component.get("selector_keys", ())
    )
    grid_ids = sorted(int(grid_id) for grid_id in domain_settings)
    if not grid_ids:
        raise ValueError("a domain tree has no domains to record")
    domains: dict[str, object] = {}
    acknowledged = set(expert_acknowledgements)
    for grid_id in grid_ids:
        settings = domain_settings[grid_id]
        try:
            components = validate_physics_capabilities(settings)
            blocker = None
        except PhysicsCapabilityError as error:
            components = None
            blocker = str(error)
        governance = {
            "state": "unresolved",
            "required_acknowledgement": None,
            "acknowledged": False,
        }
        if components is not None:
            state, required = _tree_tuple_registry_governance(
                registry, components, route_modes=governance_route_modes)
            governance = {
                "state": state,
                "required_acknowledgement": required,
                "acknowledged": (
                    required is None or required in acknowledged),
            }
            if required is not None and required not in acknowledged:
                # Warn-not-block: an implemented tuple outside the
                # registry's blessed reachability RUNS, with one line
                # saying so.  The governance record above already
                # carries acknowledged=False, so every receipt states
                # the truth; the acknowledgement remains the way to
                # silence the warning.
                tuple_text = ", ".join(
                    f"{key}={_selection_value(settings, key)!r}"
                    for key in sorted(selector_keys))
                warn(f"d{grid_id:02d} physics tuple ({tuple_text}) has "
                     f"registry reachability {state!r} -- running it "
                     f"unblessed; add {_ack_instruction(required)} to "
                     "silence this warning",
                     why="Every component in the tuple is individually "
                         "implemented and verified; the registry has "
                         "simply not blessed this combination end to "
                         "end.  The run record carries "
                         "acknowledged=false either way.")
        domains[str(grid_id)] = {
            "components": components,
            "registry_blocker": blocker,
            "governance": governance,
            # Read as the engine reads them (the one spelling), so a
            # ra_physics = 4 run reports its streams as 4/4, not -1/-1.
            "selectors": selection_values_one_spelling(
                settings, sorted(selector_keys)),
        }
    if profile is not None:
        # The root carries the named profile exactly as the single-domain
        # door would check it, so an explicitly bound tree is refused for
        # the same reason and in the same words.
        validate_single_domain_physics_profile(
            profile, config=domain_settings[grid_ids[0]],
            expert_acknowledgements=expert_acknowledgements,
            acknowledgement_provenance=acknowledgement_provenance)
    used, provenance = _acknowledgement_receipt(
        acknowledged, acknowledged, acknowledgement_provenance)
    selected_options: dict[str, list[str]] = {}
    for domain in domains.values():
        for component_id, option_id in (domain["components"] or {}).items():
            if option_id not in selected_options.setdefault(
                    component_id, []):
                selected_options[component_id].append(option_id)
    return {
        "schema": MULTI_DOMAIN_SELECTION_SCHEMA,
        "profile": profile,
        "registry_sha256": registry_sha256(registry),
        "registry_physics": registry_physics_receipt(
            registry, options=selected_options, profile=profile),
        "domains": domains,
        "acknowledgements": used,
        "acknowledgement_provenance": provenance,
    }


#: Selection-receipt fields a preparation's receipt is NOT compared on
#: when this build checks it (A153), each with why it cannot describe a
#: physics difference.  Every other field is compared as written.
SELECTION_RECEIPT_RECORD_ONLY_FIELDS: Mapping[str, str] = MappingProxyType({
    "registry_sha256": (
        "the registry DOCUMENT digest, which moves with any citation, "
        "warning or label; the physics it bound is registry_physics"),
    "registry_physics": (
        "compared part by part instead, a receipt written before it existed "
        "resolving through woof/physics_registry_history.json"),
    "maturity": (
        "the named profile's conformance evidence label; a promotion changes "
        "what is claimed about a run, not what runs"),
})


def _receipt_value_text(value: object) -> str:
    if value is _MISSING:
        return "absent"
    text = repr(value)
    # A string is a sentence (a registry blocker names the rule that no
    # longer admits the configuration after its registry pointer), so it
    # is shown whole up to a bound; a structure is shown in outline.
    limit = 400 if isinstance(value, str) else 60
    return text if len(text) <= limit else text[:limit - 3] + "..."


#: Selection-receipt maps keyed by a SETTING (each value is the setting's
#: value) and by a COMPONENT (each value is its option id), by field name,
#: at the top level of a named receipt and under each domain of a
#: per-domain one.  A key a preparation's receipt lacks is compared with
#: the setting's or component's off value (A153).
SELECTION_RECEIPT_SETTING_MAPS = frozenset({"selectors", "resolved"})
SELECTION_RECEIPT_COMPONENT_MAPS = frozenset({"components"})


def physics_selection_differences(
        recorded: object, expected: Mapping[str, object], *,
        settings: Mapping[str, object] | object | None = None,
) -> list[str]:
    """What differs between a recorded selection receipt and this build's.

    ``expected`` is the receipt this build computes for the same config;
    ``recorded`` is the one a preparation wrote, possibly under an earlier
    build; ``settings`` is the configuration ``expected`` was computed
    from.  Returns one named difference per differing field or registry
    physics part; empty means the prepared physics selection is this
    build's.  The fields in :data:`SELECTION_RECEIPT_RECORD_ONLY_FIELDS`
    are not compared as written; the registry physics is compared part by
    part, so a citation, warning, label or maturity edit resolves and a
    changed option, parameter, table or kernel identity is named.

    Physics this build ADDED since the preparation counts as equal where
    the configuration resolves it to its off value, from the registry and
    ``RunConfig`` defaults (:func:`woof.physics_registry.setting_off_value`
    and :func:`woof.physics_registry.component_off_option`): a receipt
    field the preparation lacks in a setting or component map
    (:data:`SELECTION_RECEIPT_SETTING_MAPS`,
    :data:`SELECTION_RECEIPT_COMPONENT_MAPS`), a component or option part
    its registry lacked where the selection resolves the component to its
    off option, and a knob part its registry lacked (or did not implement)
    where ``settings`` holds the knob at its off value or omits it, or does
    not read the knob at all because a setting its declaration's
    ``read_when`` names does not hold
    (:func:`woof.physics_registry.registry_knob_is_read`: a tile count
    with the tiles off).  None of that ran in the preparation and none of
    it runs here.  Any other value of an added setting, component or knob
    is named; without ``settings`` a knob's value is unknown, so an added
    knob is named.
    """

    from woof.physics_registry import (
        NO_OFF_VALUE, REGISTRY_PHYSICS_IDENTITY_SCHEMA, component_off_option,
        canonical_template_id, canonical_sha256,
        physics_registry, recorded_registry_physics_parts,
        registry_knob_is_read, same_setting_value, setting_off_value,
        strip_registry_documentation)

    if not isinstance(recorded, Mapping):
        return ["the recorded physics receipt is missing"]
    registry = physics_registry()
    differences: list[str] = []

    def added_at_off_value(container: object, key: object,
                           current: object) -> bool:
        if not isinstance(key, str):
            return False
        if container in SELECTION_RECEIPT_SETTING_MAPS:
            return same_setting_value(
                current, setting_off_value(key, registry))
        if container in SELECTION_RECEIPT_COMPONENT_MAPS:
            off = component_off_option(key, registry)
            return off is not None and current == off
        return False

    def walk(prepared: object, current: object, path: str,
             key: object = None, container: object = None) -> None:
        if key == "profile":
            prepared = canonical_template_id(prepared, registry)
            current = canonical_template_id(current, registry)
        if isinstance(prepared, Mapping) and isinstance(current, Mapping):
            for child in sorted(set(prepared) | set(current), key=str):
                if not path and child in SELECTION_RECEIPT_RECORD_ONLY_FIELDS:
                    continue
                walk(prepared.get(child, _MISSING),
                     current.get(child, _MISSING),
                     f"{path}.{child}" if path else str(child), child, key)
            return
        if prepared is _MISSING and added_at_off_value(
                container, key, current):
            return
        if prepared != current:
            differences.append(
                f"{path} (prepared {_receipt_value_text(prepared)}, this "
                f"build {_receipt_value_text(current)})")

    walk(recorded, expected, "")
    physics = expected.get("registry_physics")
    expected_parts = (physics.get("parts") if isinstance(physics, Mapping)
                      else None)
    if not isinstance(expected_parts, Mapping):
        differences.append("this build's receipt carries no registry physics")
        return differences
    recorded_parts = recorded_registry_physics_parts(recorded)
    if recorded_parts is None:
        differences.append(
            "registry physics: the preparation names registry document "
            f"{str(recorded.get('registry_sha256'))[:12]}, which is not in "
            "this build's registry history, so its physics cannot be "
            "established; prepare again")
        return differences

    selected_options: dict[str, set[str]] = {}
    for name in expected_parts:
        kind, _, rest = str(name).partition(".")
        component, separator, option = rest.partition(".options.")
        if kind == "components" and separator:
            selected_options.setdefault(component, set()).add(option)

    def knob_at_off_value(knob: str) -> bool:
        if settings is None:
            return False
        # A knob this configuration does not read runs nothing, whatever
        # value it holds (mosaic_cat with sf_surface_mosaic = 0).
        if not registry_knob_is_read(knob, settings, registry):
            return True
        off = setting_off_value(knob, registry)
        if off is NO_OFF_VALUE:
            return False
        value = _selection_value_or_absent(settings, knob)
        # A configuration that omits the knob runs its off value.
        return value is _ABSENT or same_setting_value(value, off)

    def part_added_at_off_value(name: str) -> bool:
        kind, _, rest = name.partition(".")
        if kind == "components":
            component, separator, option = rest.partition(".options.")
            off = component_off_option(component, registry)
            if off is None:
                return False
            if separator:
                return option == off
            return selected_options.get(component) == {off}
        if kind == "parameters" and rest:
            return knob_at_off_value(rest)
        return False

    def parameter_enum_was_extended(name: str) -> bool:
        # A newly admitted value does not change an older prepared value.
        # Match the complete prior declaration, changing only its enum;
        # altered defaults, types, bounds or physics therefore still refuse.
        kind, _, knob = name.partition(".")
        if kind != "parameters" or settings is None:
            return False
        spec = (registry.get("parameters") or {}).get(knob)
        if not isinstance(spec, Mapping) or not isinstance(spec.get("enum"), list):
            return False
        if expected_parts.get(name) != canonical_sha256(strip_registry_documentation(spec)):
            return False
        value = _selection_value_or_absent(settings, knob)
        if value is _ABSENT:
            value = setting_off_value(knob, registry)
        for previous in spec.get("compatible_previous_enums", ()):
            if not isinstance(previous, list) or not previous:
                continue
            if not all(any(same_setting_value(old, current) for current in spec["enum"])
                       for old in previous):
                continue
            if not any(same_setting_value(value, old) for old in previous):
                continue
            prior_spec = {**spec, "enum": previous}
            if recorded_parts.get(name) == canonical_sha256(
                    strip_registry_documentation(prior_spec)):
                return True
        return False

    def forbidden_values_were_extended(name: str) -> bool:
        # New forbidden values for new controls do not change an admitted
        # older selection. The complete old option must still hash exactly.
        kind, _, rest = name.partition(".")
        component, separator, option_id = rest.partition(".options.")
        if kind != "components" or not separator or settings is None:
            return False
        option = ((registry.get("components") or {}).get(component, {}).get("options") or {}).get(option_id)
        if not isinstance(option, Mapping):
            return False
        if expected_parts.get(name) != canonical_sha256(strip_registry_documentation(option)):
            return False
        constraints = option.get("constraints") or {}
        current = constraints.get("forbidden_setting_values") or {}
        for knob, values in current.items():
            value = _selection_value_or_absent(settings, knob)
            if value is _ABSENT:
                value = setting_off_value(knob, registry)
            if value is NO_OFF_VALUE or any(same_setting_value(value, v) for v in values):
                return False
        for previous in option.get("compatible_previous_forbidden_settings", ()):
            if not isinstance(previous, Mapping):
                continue
            if not all(knob in current and isinstance(values, list)
                       and all(any(same_setting_value(old, new) for new in current[knob])
                               for old in values) for knob, values in previous.items()):
                continue
            prior_option = {**option, "constraints": {
                **constraints, "forbidden_setting_values": previous}}
            if recorded_parts.get(name) == canonical_sha256(strip_registry_documentation(prior_option)):
                return True
        return False

    # A receipt written before registry_physics existed (or in another
    # identity schema) resolves to every part its document had; the
    # selection's own scope is then this build's.  A receipt's own parts
    # are compared both ways.
    names = set(expected_parts)
    own = recorded.get("registry_physics")
    if (isinstance(own, Mapping)
            and own.get("schema") == REGISTRY_PHYSICS_IDENTITY_SCHEMA):
        names |= set(recorded_parts)
    for name in sorted(names):
        if recorded_parts.get(name) == expected_parts.get(name):
            continue
        if (settings is not None and name.startswith("parameters.")
                and not registry_knob_is_read(
                    name.removeprefix("parameters."), settings, registry)):
            # A newly scoped knob can leave an older receipt's parts.
            # Its metadata cannot change physics this configuration never
            # reads, just as an added unread knob cannot change it.
            continue
        if parameter_enum_was_extended(name):
            continue
        if forbidden_values_were_extended(name):
            continue
        if (name not in recorded_parts and name in expected_parts
                and part_added_at_off_value(name)):
            continue
        where = ("absent from the prepared registry"
                 if name not in recorded_parts else
                 "absent from this build's registry"
                 if name not in expected_parts else "changed")
        differences.append(f"registry physics of {name} ({where})")
    return differences


def _tree_tuple_registry_governance(
        registry: Mapping[str, object],
        selected: Mapping[str, str],
        *,
        route_modes: tuple[str, ...] = ("experiment-per-domain",),
) -> tuple[str, str | None]:
    """Resolve a complete tuple without consulting source identity.

    ``route_modes`` names which implemented routes' declared templates
    make up the reachability union.  The domain-tree question keeps its
    historical union (the per-domain routes and their component
    overrides); the single-domain selection widens it to the
    fixed-template routes too, because a template the registry declares
    as one of THIS route's shipped products (RUC on ERA5 is the
    exhibit) must not read as outside the registry's reachability at
    its own runner.
    """

    components = registry.get("components", {})
    templates = registry.get("templates", {})
    routes = registry.get("runner_routes", {})
    if (
        not isinstance(components, Mapping)
        or not isinstance(templates, Mapping)
        or not isinstance(routes, Mapping)
    ):
        raise PhysicsCapabilityError(
            "physics registry lacks tuple reachability declarations")

    normal = False
    expert: set[str] = set()

    for route in routes.values():
        if (
            not isinstance(route, Mapping)
            or route.get("implemented") is not True
            or route.get("mode") not in route_modes
        ):
            continue
        override_components = [
            value for value in route.get(
                "allowed_component_overrides", ())
            if isinstance(value, str)
        ]
        option_sets: dict[str, tuple[str, ...]] = {}
        for component_id in override_components:
            component = components.get(component_id)
            options = (
                component.get("options", {})
                if isinstance(component, Mapping) else {}
            )
            option_sets[component_id] = tuple(sorted(
                option_id
                for option_id, option in options.items()
                if isinstance(option_id, str)
                and isinstance(option, Mapping)
                and option.get("implemented") is True
            ))
        allowed_component_options = route.get(
            "allowed_component_options", {})
        if isinstance(allowed_component_options, Mapping):
            for component_id, option_ids in allowed_component_options.items():
                if not (
                    isinstance(component_id, str)
                    and isinstance(option_ids, (list, tuple))
                ):
                    continue
                component = components.get(component_id)
                options = (
                    component.get("options", {})
                    if isinstance(component, Mapping) else {}
                )
                admitted = {
                    option_id for option_id in option_ids
                    if isinstance(option_id, str)
                    and isinstance(options.get(option_id), Mapping)
                    and options[option_id].get("implemented") is True
                }
                admitted.update(option_sets.get(component_id, ()))
                option_sets[component_id] = tuple(sorted(admitted))

        def admits(template_id: str) -> bool:
            template = templates.get(template_id)
            base = (
                template.get("components", {})
                if isinstance(template, Mapping) else {}
            )
            if not isinstance(base, Mapping):
                return False
            # Preparation used to materialize millions of Cartesian tuples
            # to ask whether this one tuple belonged. Each override dimension
            # is independent, so exact membership needs only its own choices.
            if set(selected) != set(base) | set(option_sets):
                return False
            if any(selected[name] != value for name, value in base.items()
                   if name not in option_sets):
                return False
            for component_id, option_ids in option_sets.items():
                own = base.get(component_id)
                value = selected[component_id]
                # Keep the template's own choice even when it is absent
                # from the route override list (the R-022 expert demotion).
                if value not in option_ids and not (isinstance(own, str) and value == own):
                    return False
            return True

        source_template_ids = route.get("source_template_ids", {})
        if isinstance(source_template_ids, Mapping):
            normal_ids = {
                template_id
                for declared in source_template_ids.values()
                if isinstance(declared, list)
                for template_id in declared
                if isinstance(template_id, str)
            }
            for template_id in normal_ids:
                normal = normal or admits(template_id)

        expert_template_ids = route.get("expert_template_ids", {})
        acknowledgement = route.get("expert_acknowledgement_id")
        if (
            isinstance(expert_template_ids, Mapping)
            and isinstance(acknowledgement, str)
        ):
            expert_ids = {
                template_id
                for declared in expert_template_ids.values()
                if isinstance(declared, list)
                for template_id in declared
                if isinstance(template_id, str)
            }
            for template_id in expert_ids:
                if admits(template_id):
                    expert.add(acknowledgement)

    if normal:
        return "registry-reachable", None
    if expert:
        acknowledgements = sorted(expert)
        if len(acknowledgements) != 1:
            raise PhysicsCapabilityError(
                "registry expert tuple publishes ambiguous acknowledgements "
                f"{acknowledgements}")
        return "registry-expert-template", acknowledgements[0]
    authority = registry.get("authority", {})
    acknowledgement = (
        authority.get(
            "unnamed_tree_outside_reachability_acknowledgement_id")
        if isinstance(authority, Mapping) else None
    )
    if not isinstance(acknowledgement, str) or not acknowledgement:
        raise PhysicsCapabilityError(
            "registry does not publish the unnamed-tree outside-"
            "reachability acknowledgement setting")
    return "outside-registry-declared-reachability", acknowledgement


def thompson_runtime_requirements() -> dict[str, object]:
    """Describe the guarded evidence-runner MP8 contract (env untouched).

    This receipt is consumed by the guarded benchmark/evidence runners under
    ``tools/`` (hrrr_single_domain_benchmark, prepared_single_domain_forecast),
    which keep their own explicit environment gates.  The PRODUCT selection
    path (config validation, namelist import, forecast dispatch, restart
    identity) admits mp_physics=8 first-class with the packaged table root
    (:func:`thompson_table_root`); it does not read this dict.
    """

    from woof.core.thompson_contract import (
        CLASSIC_TABLE_ASSETS,
        TABLE_SET_ID,
        WRF_REFERENCE_COMMIT,
        WRF_REFERENCE_VERSION,
    )

    return {
        "readiness": "WRF_MATCHED_RUN_EXPERIMENTAL_RUNTIME",
        "explicit_expert_consent_required": False,
        "runtime_guard": {
            "environment": EXPERIMENTAL_THOMPSON_ENV,
            "required_value": "1",
        },
        "table_root": {
            "environment": THOMPSON_TABLE_ROOT_ENV,
            "must_name_directory": True,
            "validation": "exact-size-and-sha256-before-GPU-setup",
        },
        "table_authority": {
            "table_set": TABLE_SET_ID,
            "wrf_version": WRF_REFERENCE_VERSION,
            "wrf_commit": WRF_REFERENCE_COMMIT,
            "payload_bytes": sum(asset.bytes for asset in CLASSIC_TABLE_ASSETS),
            "assets": [
                {
                    "filename": asset.filename,
                    "bytes": asset.bytes,
                    "sha256": asset.sha256,
                }
                for asset in CLASSIC_TABLE_ASSETS
            ],
        },
        "capability_probe_validates_environment_or_table_bytes": False,
    }


def noahmp_expert_column_budget() -> int:
    """Columns the caller has explicitly accepted for a Noah-MP run.

    Zero when unset or unparseable, so a malformed budget is worth exactly as
    much as no budget at all.  A budget is never read as a ceiling on its own:
    :func:`pending_wrf_physics_components` takes the larger of it and the
    measured ceiling, so the environment can only widen what the caller has
    said they accept, never narrow the measured evidence.
    """

    raw = os.environ.get(NOAHMP_EXPERT_COLUMN_BUDGET_ENV)
    if raw is None:
        return 0
    try:
        budget = int(raw.strip())
    except ValueError:
        return 0
    return budget if budget > 0 else 0


def noahmp_projected_call_seconds(columns: int) -> float:
    """Projected wall clock of one land-surface call at ``columns`` columns.

    Linear from the top of the measured range at the 360,000-column ceiling
    (slab path, 2026-07-27), and stated as a PROJECTION: widths beyond the
    ceiling are exactly the ones nothing has measured, which is why the
    budget rail quotes this number instead of admitting them silently.
    """

    return (columns * NOAHMP_MEASURED_SLAB_CALL_SECONDS[1]
            / NOAHMP_MEASURED_COLUMN_CEILING)


@dataclass(frozen=True)
class PhysicsPortBlocker:
    """One coupled physics component requested before it is executable.

    ``missing`` has been an ordered tuple since it was written: element
    zero states the RULE -- the admitted configuration, which is also
    the thing to change -- and the elements after it explain the
    mechanism that makes the rule necessary.  ``action_count`` names
    where that boundary falls instead of leaving it to a reader (or a
    formatter) to infer: a blocker whose second element is a REMEDY
    rather than a mechanism raises its count, because burying a remedy
    behind a flag is the one thing this layering must never do.

    The example that used to be given here -- the Noah-MP column budget --
    is no longer a blocker at all: the width advisory warns and continues
    (see the ceiling comment above), so no ``PhysicsPortBlocker`` named
    "Noah-MP column budget" is constructed anywhere in this tree.  Every
    blocker that IS constructed here refuses at plan review, and the
    default ``action_count`` of one holds for all of them.
    """

    component: str
    selectors: tuple[tuple[str, int], ...]
    missing: tuple[str, ...]
    #: How many leading ``missing`` elements are the action half.
    action_count: int = 1

    def _selected(self) -> str:
        return ", ".join(f"{key}={value}" for key, value in self.selectors)

    def format(self) -> str:
        """The whole blocker on one line -- rule and mechanism together.

        Unchanged, and still the text the namelist importer's port
        receipt embeds: this is the full statement, and every consumer
        that wants all of it keeps getting all of it.
        """

        return f"{self.component} ({self._selected()}): " + "; ".join(
            self.missing)

    def action(self) -> str:
        """The rule and any remedy: what a reader has to change."""

        return f"{self.component} ({self._selected()}): " + "; ".join(
            self.missing[:self.action_count])

    def why(self) -> str:
        """The mechanism behind the rule; ``""`` when none was written."""

        return "; ".join(self.missing[self.action_count:])


class UnsupportedPhysicsSuiteError(ValueError):
    """A requested WRF suite contains one or more unfinished components.

    The message is layered (:mod:`woof.explain`): every blocker's rule
    prints by default, and the mechanism paragraphs -- which coupling
    writes over which diagnostic, which oracle fixtures exist -- follow
    ``--explain``.  Both halves live in this one string, so anything
    reading ``str(error)`` still sees the complete receipt; only the CLI
    print boundary chooses.
    """

    def __init__(self, blockers: tuple[PhysicsPortBlocker, ...]):
        if not blockers:
            raise ValueError("UnsupportedPhysicsSuiteError needs blockers")
        self.blockers = blockers
        actions = "\n".join(f"  - {item.action()}" for item in blockers)
        reasons = "\n".join(f"  - {item.component}: {item.why()}"
                            for item in blockers if item.why())
        super().__init__(layered(
            "requested WRF physics suite is not executable in woof yet; "
            "no substitutions were applied:\n" + actions,
            ("why these pairings are refused:\n" + reasons) if reasons
            else ""))


def pending_wrf_physics_components(
        *, mp_physics: int, sf_sfclay_physics: int,
        bl_pbl_physics: int, sf_surface_physics: int,
        num_soil_layers: int,
        columns: int | None = None,
        ) -> tuple[PhysicsPortBlocker, ...]:
    """Return unfinished components selected by a WRF physics request.

    The PBL/surface-layer verdict comes from the complete declarative WRF
    v4.6.1 table in :mod:`woof.wrf461_compatibility`.  RUC's layer count is
    included in its blocker even though WRF also supports a six-layer RUC
    configuration; the target suite explicitly requests nine.

    This function is the readiness authority for the ported selector set
    it represents, and it is NOT the only one.  Three selector values are
    deliberately outside :mod:`woof.wrf461_compatibility`'s axes and are
    refused, or admitted, elsewhere -- and the guard below skips the
    matrix lookup for them rather than crashing on a table that has no row
    to have:

    * ``bl_pbl_physics=2`` and ``sf_sfclay_physics=2`` (MYJ and its Eta
      surface layer) carry WRF's own law implemented directly and in both
      directions, in ``woof.config.validate_myj_pairing``, which
      ``validate_run_config`` reaches BEFORE this function;
    * ``bl_pbl_physics=900`` (SASE) has no WRF counterpart to transcribe
      and is admitted by ``woof.config.validate_sase_config``.

    Those exclusions are declared, with their reasons, in
    ``woof.wrf461_compatibility.AXIS_EXCLUSIONS``.  For everything else
    ``woof/config.py`` accepts every selector value in its schema tables
    and lets this receipt do the refusing, so admitting a scheme is an
    edit here and a dispatch row in ``woof/core/physics.py`` -- never a
    silent widening of a numeric range check.
    """
    blockers: list[PhysicsPortBlocker] = []
    # mp_physics == 8 no longer appends a blocker: Thompson was promoted to
    # a first-class selection when the canonical classic tables became
    # package data (see the EXPERIMENTAL_THOMPSON_ENV comment block above).
    # Byte validation of the resolved table root still fails closed at
    # setup, which is the guard that was ever essential at run time.
    # Namelist preview invokes this readiness layer before its documented
    # WRF-to-ArWen selector mappings (ISHMAEL 55 -> Morrison 10 was one
    # until it became a named refusal, woof.config._MP_UNPORTED_SCHEMES).
    # Shin-Hong 11 is no longer such a mapping either: it imports natively
    # since the Shin-Hong port, so bl_pbl_physics=11 is inside PBL_OPTIONS
    # and its four (11, sfclay) cells get their verdict from the WRF matrix
    # below exactly like 0/1/5.  The matrix governs the ported selector set
    # only; an outside raw WRF value continues to the importer's
    # mapping/refusal authority.
    #
    # mp_physics == 28 (aerosol-aware Thompson) appends no blocker either,
    # and that is a decision recorded here rather than an omission.  It has
    # a dispatch row (woof/core/microphysics.py), a complete adapter
    # (woof/core/microphysics_aerosol.py) over eight aerosol CUDA
    # translation units, prognostic nc/nwfa/nifa state with transport,
    # restart and nesting, a reflectivity route, and a wrfout inventory.
    # What is NOT admitted is refused by NAME somewhere a user can see it,
    # never by a silent numeric gate here:
    #   * the two aerosol-source selectors fail closed in
    #     woof.config.validate_aerosol_source_options -- aer_init_opt and
    #     wif_input_opt select climatology or analyzed fields; the analyzed
    #     route requires both QNWFA/QNIFA. Black carbon has no nbca species;
    #   * WRF's real.exe FATALs mp_physics=28 at wif_input_opt=0
    #     (dyn_em/module_initialize_real.F:2735-2736) while ArWen runs
    #     thompson_init's synthetic CCN/IN profile.  Same physics, an
    #     initialization WRF's initializer refuses to produce.  That is
    #     published as woof.config.MP28_AEROSOL_SOURCE_DEFAULT (the
    #     WIF climatology, default since lane/wif-default) with
    #     MP28_AEROSOL_SYNTHETIC_FALLBACK as its named fallback, and is
    #     carried in the namelist importer's printed receipt;
    #   * the registry decides REACHABILITY.  Audit R-067 gave mp=28 its
    #     first named suite -- ONE, thompson-aerosol-mp28-myj-eta-noah-
    #     rte-rrtmgp-v1, declared on the experiment-per-domain route and
    #     on the prepared single-domain route (the latter since its
    #     cold-start arm landed, audit R-044); the native benchmark keeps
    #     it off because no native run of it exists to replay.  It is
    #     not the default template's microphysics and no route makes it a
    #     default, so it is still never the scheme a user gets by
    #     accident: it is reached by naming that suite or as a per-domain
    #     component override.  A second named suite, the operational HRRR
    #     fork composition thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1
    #     (4193eb0da), is the recommended profile of the HRRR doors (hrrr,
    #     hrrr-prs, hrrr-native) and nowhere else, and the native benchmark
    #     declares it on the HRRR source because the operational namelist
    #     it replays is that composition.  Neither is the global default.
    #     (Verified against the shipped registry by
    #     tests/test_mp28_runnable.py.)
    # Adding a blocker here instead would be the wrong shape twice over: it
    # would refuse the whole scheme for a limitation that is really about
    # the aerosol SOURCE, and it would hide the WRF citation that makes the
    # limitation checkable.
    #
    # mp_physics == 16 (WDM6) appends no blocker either, and like 28 that is
    # a decision recorded here rather than an omission.  It has a dispatch
    # row (woof/core/microphysics.py), an adapter (woof/core/wdm6.py) over
    # its own CUDA translation unit, prognostic nn/nc/nr with transport and
    # restart, a reflectivity route on its own rain number, a wrfout
    # inventory and a registry option.  What is NOT admitted is refused by
    # NAME somewhere a user can see it, never by a silent numeric gate here:
    #   * WDM5 (14) and WDM7 (26) fail closed in woof.config with the
    #     scheme spelled out -- different hydrometeor sets, and WDM6 may not
    #     stand in for either;
    #   * a MIXED nest edge touching 16 RESOLVES (audit R-004): the entry
    #     closure seeds nc=0, nr=0 and nn at the domain's own ccn_conc,
    #     which is what this tree's cold start writes and what WRF's
    #     flow_dep_bdy_qnn pushes through an inflow face.  An OFFLINE
    #     downscale from or into a WDM6 domain runs that same edge on the
    #     parent archive (woof.offline_child reads QNCCN through its own
    #     scheme-qualified WDM6 map);
    #   * a run with no XLAND is refused by the adapter rather than given a
    #     fabricated land mask, because the mask picks the autoconversion
    #     threshold (module_mp_wdm6.F:607-614);
    #   * the registry decides REACHABILITY.  Audit R-067 gave mp=16 its
    #     first named suite (wdm6-mp16-ysu-mm5-noah-grell-freitas-rte-
    #     rrtmgp-v1), which is where a user reaches it besides a per-domain
    #     component override; it is no route's default.
    # The scheme's real limitation is EVIDENCE, not capability: no oracle
    # comparison against WRF's own module_mp_wdm6.F has been run.  That
    # belongs on the registry option's maturity and warning, where a user
    # reads it, not in a blocker that would refuse the scheme outright.
    # THE ONE DECISION IN THIS FUNCTION THAT USED TO CARRY NO COMMENT.
    # It is not a filter, it is a DELEGATION: a pair outside these axes
    # has no transcribed WRF cell -- pbl_surface_layer_verdict raises for
    # it rather than returning a verdict -- and it is answered by the
    # authority named in this function's docstring, which runs earlier.
    # Read as a filter it looks like silent admission, which is how a
    # reader concludes that MYJ or SASE reaches a run unchecked.
    if (
        bl_pbl_physics in PBL_OPTIONS
        and sf_sfclay_physics in SURFACE_LAYER_OPTIONS
    ):
        pair_verdict, pair_citation = pbl_surface_layer_verdict(
            bl_pbl_physics, sf_sfclay_physics)
        if pair_verdict is WRFVerdict.FATAL:
            blockers.append(PhysicsPortBlocker(
                component="WRF v4.6.1 PBL/surface-layer compatibility",
                selectors=(("sf_sfclay_physics", sf_sfclay_physics),
                           ("bl_pbl_physics", bl_pbl_physics)),
                missing=(
                    "WRF v4.6.1 refuses this pairing, and so does woof: "
                    "select the revised MM5 (1) or classic MM5 (91) "
                    "surface layer, which publish the full similarity "
                    "functions every ported PBL scheme reads, or pair "
                    "each PBL scheme with the surface layer its own "
                    "registry option declares",
                    f"{pair_citation.anchor}: {pair_citation.law}",
                    # ARWEN'S OWN BREAKAGE, beside WRF's citation and not
                    # instead of it.  A refusal that says only "WRF
                    # refuses this" tells a user which authority to argue
                    # with, not what would go wrong here -- and under the
                    # own-way ruling WRF's verdict is evidence, not the
                    # reason.  The mechanism is one missing pair of
                    # fields: woof/core/physics_inventory.py's
                    # SFCLAY_OUTPUTS carries fm/fh and the MYNN surface
                    # layer's output set does not, while _run_ysu and
                    # _run_shinhong bind psim=f["fm"], psih=f["fh"] and
                    # reconstruct zol = br*fm^2/fh from them.  Every
                    # surface-layer output is allocated as zeros for all
                    # schemes, so the pairing does not raise: it
                    # integrates on fm=fh=0 for the whole forecast.
                    "in woof the mechanism is the fm/fh pair: only the "
                    "MM5 surface layers publish the full similarity "
                    "denominators ln(z/z0)-psi, and YSU and Shin-Hong "
                    "bind them directly and divide by them, so a pairing "
                    "that leaves them unwritten runs on the allocated "
                    "zeros -- finite, plausible and wrong -- rather than "
                    "failing",
                )))
    if sf_surface_physics == 3:
        # Deferred exactly as woof.config defers it: ruc_contract is
        # forecast-side and the RW-WPS preprocessing wheel does not stage it,
        # so reading RUC's geometry at import time would make this module
        # unimportable there.  Reading it only when scheme 3 is the question
        # keeps both properties, and keeps the counts out of this file.
        from woof.core.ruc_contract import (
            NUM_SOIL_LAYERS as RUC_NUM_SOIL_LAYERS,
            WRF_SUPPORTED_NUM_SOIL_LAYERS as RUC_WRF_SUPPORTED_NUM_SOIL_LAYERS,
        )
    if (sf_surface_physics == 3
            and num_soil_layers not in RUC_WRF_SUPPORTED_NUM_SOIL_LAYERS):
        # WRF'S OWN refusal, quoted.  share/module_check_a_mundo.F:3574-3581
        # resolves num_soil_layers for sf_surface_physics=3 to 6 or to 9 and
        # silently coerces every other request to 6; share/module_soil_pre.F
        # init_soil_depth_3 (:1161-1167) has a zs table for exactly those two
        # lengths and leaves zs UNINITIALISED for any other, then calls
        # wrf_error_fatal at 4 and 5 (:1189-1192).  So the set is closed
        # because WRF has no level table outside it, not because woof has
        # not got round to a number.
        blockers.append(PhysicsPortBlocker(
            component="RUC soil geometry",
            selectors=(("sf_surface_physics", 3),
                       ("num_soil_layers", num_soil_layers)),
            missing=(
                "WRF's RUC defines "
                + " and ".join(
                    str(count)
                    for count in RUC_WRF_SUPPORTED_NUM_SOIL_LAYERS)
                + " soil levels and no other geometry",
                "share/module_soil_pre.F:init_soil_depth_3:1161-1167 "
                "tabulates zs for those lengths only and leaves zs "
                "uninitialised otherwise; :1189-1192 is fatal at 4 and 5, "
                "and share/module_check_a_mundo.F:3574-3581 coerces every "
                "other request to 6. In woof the same fact is one table: "
                "woof.ingest.ruc_soil.RUC_LEVEL_DEPTHS_M holds the zs "
                "column for each admitted count and dzs is derived from "
                "it, so a count with no row there has no soil column at "
                "all. Adding a geometry is adding that row -- an evidenced "
                "zs table, not a code path -- and the admitted set, the "
                "ingest and this refusal all widen with it",
            )))
    elif sf_surface_physics == 3 and num_soil_layers != RUC_NUM_SOIL_LAYERS:
        # EVIDENCE, not a missing branch.  The forecast column compiles and
        # runs at six levels: woof/core/kernels/ruc.cu sizes every soil
        # scratch from RUC_NZS and selects the level table with it, and
        # gpuwm.core.ruc/.ruc_gpu resolve the count from the profile.  What
        # six levels does NOT have is a WRF forecast oracle -- every fixture
        # in woof/data/ruc is nine-level and regenerating them needs the
        # pinned WRF tree.  So this is a statement about what has been
        # MEASURED, and the warn-not-block ruling applies: slower evidence,
        # not a wrong answer.
        warn(
            f"RUC at {num_soil_layers} soil levels carries no WRF forecast "
            "oracle; the column is host/device bit-identical at this "
            "geometry and completes a real forecast, but has never been "
            "compared to WRF's Fortran at it -- unverified, not wrong; "
            "continuing",
            why="woof.ingest.ruc_soil.ruc_soil_depths(6) and "
                "woof.core.ruc.ruc_soil_geometry(6) reproduce WRF 4.7.1 "
                "real.exe wrfinput_d01 ZS/DZS exactly (ZS 0 0.05 0.2 0.4 1.6 "
                "3; DZS 0.025 0.125 0.175 0.7 1.3 0.7), so the GEOMETRY is "
                "oracle-matched.  The forecast COLUMN at six levels is not: "
                "every lsmruc/sfctmp/soilmoist/snowtemp fixture is "
                "nine-level.  What IS measured at six: the 43-field "
                "host/device driver comparison at max_ulp 0 "
                "(tests/test_ruc_nzs_device.py) and a completed HRRR-"
                "initialised forecast.  See "
                "docs/wrf_ruc_runtime_admission.md.")
    if sf_surface_physics == 4 and columns is not None:
        # GRID WIDTH, not a missing branch.  Noah-MP is fully ported and
        # bitwise, and since the slab orchestration it also FINISHES at
        # production width; a wider grid than anything measured is a
        # PERFORMANCE projection, not a correctness gap -- so it is a
        # one-line warning now, never a blocker (warn-not-block ruling).
        # The env override is kept as the way to silence the warning.
        budget = max(NOAHMP_MEASURED_COLUMN_CEILING,
                     noahmp_expert_column_budget())
        if columns > budget:
            warn(
                f"Noah-MP at {columns} columns is beyond the "
                f"{NOAHMP_MEASURED_COLUMN_CEILING}-column measured width; "
                f"the land-surface call projects to about "
                f"{noahmp_projected_call_seconds(columns):.2f} s -- "
                "slow, not wrong; continuing",
                why=f"Measured {NOAHMP_MEASURED_SLAB_CALL_SECONDS[0]}-"
                    f"{NOAHMP_MEASURED_SLAB_CALL_SECONDS[1]} s per call "
                    f"at {NOAHMP_MEASURED_COLUMN_CEILING} columns on one "
                    "RTX 5090 (2026-07-27); the projection is linear in "
                    f"columns.  Set {NOAHMP_EXPERT_COLUMN_BUDGET_ENV} at "
                    "or above your column count to silence this warning.")
    if sf_surface_physics == 4 and num_soil_layers != 4:
        # Noah-MP itself is no longer refused: it has a dispatch row, a
        # driver, a cold start, restart identity and output.  What is still
        # refused is a soil geometry no Noah-MP fixture covers.  This is a
        # narrower blocker than the old one, not a widened gate: the old row
        # refused every Noah-MP request.
        blockers.append(PhysicsPortBlocker(
            component="Noah-MP soil geometry",
            selectors=(("sf_surface_physics", 4),
                       ("num_soil_layers", num_soil_layers)),
            missing=(
                "Noah-MP is admitted at num_soil_layers=4 only",
                # THE GEOMETRY, not the fixtures.  Arguing from missing
                # oracle coverage invites the answer the tree already
                # gives RUC at six levels -- warn, do not block -- and
                # that answer is wrong here: RUC's six-level geometry is
                # reproduced exactly and only its forecast column lacks an
                # oracle, whereas Noah-MP at another count has no layer
                # depths in existence to run on.
                "Noah-MP takes its layer depths from init_soil_depth_2 "
                "(share/module_soil_pre.F:1128-1151), the same generator "
                "as Noah, which is fatal at any count but 4; "
                "woof.core.noah.SOIL_LAYER_THICKNESS_M is that "
                "generator's only output and nothing in the tree produces "
                "zs/dzs for another count, so a different count has no "
                "layer depths to run on -- a row here would be invented "
                "soil, not a table entry",
            )))
    return tuple(blockers)


def require_ready_wrf_physics(**selection: int) -> None:
    """Raise a complete fail-closed receipt for any pending component."""
    blockers = pending_wrf_physics_components(**selection)
    if blockers:
        raise UnsupportedPhysicsSuiteError(blockers)


__all__ = [
    "ASYMMETRIC_RADIATION_NOCTURNAL_ACK",
    "conditional_refusals_for",
    "CONSTANT_DOWNWARD_LONGWAVE_ACK",
    "EXPERIMENTAL_THOMPSON_ENV",
    "KESSLER_PROFILE_ID",
    "MYNN_NOAHMP_PROFILE_ID",
    "MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID",
    "MYNN_PROFILE_ID",
    "MYNN_RTE_RRTMGP_PROFILE_ID",
    "MYNN_RUC_PROFILE_ID",
    "MYNN_RUC_RTE_RRTMGP_PROFILE_ID",
    "NOAHMP_PROFILE_ID",
    "NOAHMP_EXPERT_COLUMN_BUDGET_ENV",
    "NOAHMP_MEASURED_COLUMN_CEILING",
    "NOAHMP_MEASURED_SLAB_CALL_SECONDS",
    "COMPOSITION_SUITE_PROFILE_IDS",
    "SINGLE_DOMAIN_PHYSICS_PROFILES",
    "route_physics_profiles",
    "MORRISON_PROFILE_ID",
    "MP28_REGISTRY_OPTION_ID",
    "MULTI_DOMAIN_SELECTION_SCHEMA",
    "land_surface_component_for_selector",
    "multi_domain_physics_selection",
    "noahmp_expert_column_budget",
    "noahmp_projected_call_seconds",
    "NSSL2_PROFILE_ID",
    "P3_LEGACY_RRTMG_PROFILE_ID",
    "RUC_PROFILE_ID",
    "PhysicsPortBlocker",
    "PhysicsCapabilityError",
    "PhysicsVerticalPreflightError",
    "RADIATION_OFF_LAND_SURFACE_ACK",
    "RRTMG_VARIANT_LEGACY",
    "RRTMG_VARIANT_RTE_RRTMGP",
    "THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID",
    "THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID",
    "THOMPSON_PROFILE_ID",
    "THOMPSON_TABLE_ROOT_ENV",
    "UnsupportedPhysicsSuiteError",
    "WRF_RRTMG_COMPATIBILITY_TOKENS",
    "WRF_RRTMG_LEGACY",
    "WSM6_PROFILE_ID",
    "WRF_RRTMG_TO_RTE_RRTMGP",
    "WRF_RRTMG_TO_RTE_RRTMGP_V1",
    "IMPLICIT_RUNTIME_SWITCHES",
    "first_local_night_time",
    "constant_longwave_refusal",
    "downward_longwave_disposition",
    "identify_single_domain_profile",
    "implicit_runtime_switches",
    "nocturnal_radiation_refusal",
    "radiation_off_land_surface_refusal",
    "solar_elevation_deg",
    "packaged_thompson_table_root",
    "pending_wrf_physics_components",
    "profile_declared_acknowledgements",
    "settings_declared_acknowledgements",
    "thompson_guard_exports",
    "thompson_fork_table_root",
    "thompson_table_root",
    "require_ready_wrf_physics",
    "require_rrtmg_legacy_executable",
    "require_rrtmg_legacy_ready",
    "rrtmg_variant",
    "single_domain_physics_selection",
    "single_domain_runtime_switches",
    "thompson_runtime_requirements",
    "user_thompson_table_root",
    "validate_physics_capabilities",
    "validate_resolved_physics_vertical_levels",
    "validate_single_domain_physics_profile",
]


# ---------------------------------------------------------------------------
# AGREEMENT WITH THE REGISTRY, AT IMPORT.
#
# ``_GLW_CONSUMING_SURFACE_SCHEMES`` is the source of every land-surface
# option's ``consumers.reads_glw`` row; the two profile menus are hand-kept
# lists of template ids, and an implemented composition with no menu row
# has no front door.  Each deliberate omission is cited, so the retirement
# sweep is a grep (R-068 for the ONE template the runtime-switch
# derivation still reaches no route for; the sibling citations are
# retired below).
def _templates_outside_the_single_domain_menu() -> dict[str, str]:
    """The omissions that remain, named through the registry's own records.

    No id is spelled as a literal here: one template's id carries the
    forcing source it was registered on, and this module is a protected
    zone for source and case tokens.  Each omission is a fact about the
    template's COMPOSITION or about the registry's default template, so
    both are resolved from the registry.

    RETIRED here, with the menu that replaced it: the R-068 citation for
    the one WSM6 + KF template on the aggregate RTE+RRTMGP option.  The
    menu is derived from the fixed-template routes' own declarations now,
    that template is declared on one of them, and its runtime product is
    resolved from the composition like every other row -- so there is no
    omission left to cite.

    RETIRED here, with the fix it waited on: the R-067 citation for the
    aerosol-aware Thompson suite, whose blocker (R-044) was a missing
    mp_physics=28 arm in woof/ingest/microphysics_cold_start.py.  The arm
    exists, the prepared single-domain route declares the suite, and the
    menu therefore carries it by derivation.
    """

    from woof.physics_registry import (
        DEFAULT_TEMPLATE_ID, REGISTRY_REBUILD_ENV)

    if os.environ.get(REGISTRY_REBUILD_ENV) == "1":
        # tools/build_registry.py reaches this module while it REGENERATES
        # the registry, so the templates these citations name may not exist
        # on disk yet.  The agreement check these feed is skipped for the
        # same reason and in the same window, so an empty map here refuses
        # nothing that the rebuilt registry will not be held to.
        return {}
    return {
        DEFAULT_TEMPLATE_ID: (
            "audit R-068: the registry's DEFAULT_TEMPLATE_ID, registered for "
            "plan review, never given a _SINGLE_DOMAIN_RUNTIME_SWITCHES row"),
    }


_TEMPLATES_OUTSIDE_THE_SINGLE_DOMAIN_MENU = (
    _templates_outside_the_single_domain_menu())


def _require_agreement_with_the_registry() -> None:
    from woof.physics_registry import (
        require_consumer_rows_agreement, require_template_menu_agreement)

    require_consumer_rows_agreement(
        "woof.physics_compat._GLW_CONSUMING_SURFACE_SCHEMES",
        "land_surface", "reads_glw",
        {value: True for value in _GLW_CONSUMING_SURFACE_SCHEMES},
        cited_absences={0: "no land surface reads GLW"})
    require_template_menu_agreement(
        "woof.physics_compat.SINGLE_DOMAIN_PHYSICS_PROFILES",
        SINGLE_DOMAIN_PHYSICS_PROFILES,
        cited_absences=_TEMPLATES_OUTSIDE_THE_SINGLE_DOMAIN_MENU)
    require_template_menu_agreement(
        "woof.physics_compat._SINGLE_DOMAIN_RUNTIME_SWITCHES",
        tuple(_SINGLE_DOMAIN_RUNTIME_SWITCHES),
        cited_absences=_TEMPLATES_OUTSIDE_THE_SINGLE_DOMAIN_MENU)


_require_agreement_with_the_registry()
