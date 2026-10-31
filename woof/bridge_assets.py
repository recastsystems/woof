"""``woof fetch-bridges``: stage the prebuilt Rust artifacts.

A PLATFORM wheel now ships the compiled Rust itself: the artifacts below
are staged into ``woof/libexec/bridges`` by
``tools/stage_wheel_bridges.py`` before the wheel is built, so on a
platform a bundle exists for, ``pip install recast-woof`` carries every declared
artifact but the on-demand observation-stream doors (:func:`on_demand_artifacts`),
which the first door that needs them stages from the bundle by itself.  This
command remains the route for everyone else: the ``py3-none-any`` fallback
wheel, which pip resolves wherever no platform bundle is published,
carries no binaries at all.  Before either existed, the only way to get
the GRIB decoders, the CPU preprocessing library, the fetch backbone,
the batch renderer, the two radar front doors, the MRMS, Stage-IV,
surface, GOES and European-composite front doors, the NetCDF decoder,
the mapped decode engine, the Zarr reader, the observation remap, the
isobaric-height reader, the radar superob, the native GRIB2 and ML dataset
exporters, the nowcast frames converter, the global globe packer, the
observation scorecard, the forecast verification scorer, the simulated
radar, the regular-grid intermediate producer, the MPAS point renderer, the
TEMPO initialization library, the projected MET intermediate library, the
seven observation-stream doors, the radar feed listings and the terminal
workspace onto a
wheel install was to clone the repository and run ``cargo build`` -- a
Rust toolchain, a 2.5 GB checkout and a few minutes of compiling, for
the declared native files.
``woof fetch-bridges`` is the same trade :mod:`woof.table_assets`
already makes for the externalized physics tables: the artifacts are
published as versioned GitHub release assets, their exact size and
SHA-256 pins are packaged inside the wheel at release time, and every
byte is verified against those pins *before* anything is installed.

What is staged, and where
-------------------------
One bundle per platform, holding the declared native artifacts of
:data:`BUNDLED_ARTIFACTS`, staged into :func:`woof.bridges
.default_bridge_dir` (``~/.woof/bridges/<release>-<bundle digest>``)
-- the staged rung of the resolution ladder every consumer already
searches, so nothing else in woof needs to know this command exists.
The directory is this release's alone: until 2.8.6 every release
staged flat into the shared ``~/.woof/bridges``, so two installs of
different versions on one machine re-fetched over each other and each
broke the other's runs.  The flat directory is still read, by the
rung after the versioned one, but only for a file whose bytes are this
release's pin, and it is never written by a pinned install.  ``--dest DIR`` stages
somewhere else; the per-artifact environment variables keep overriding
everything, and a set override is reported rather than silently
shadowed.

A bundle also carries the renderer's map assets -- the Natural Earth
and US Census shapefiles ``rw_wrfbatch`` draws coastlines, borders,
state lines and counties from -- under :data:`ASSET_ROOT`, staged
beside the binaries as ``<dest>/assets/basemap/...``.  That is not a
new lookup: ``rw_wrfbatch`` already resolves ``assets/basemap`` under
its own ancestors, so a renderer staged at ``~/.woof/bridges`` finds
the shapefiles itself, with no environment variable set and no Python
in the loop.  They are not in the ``woof`` wheel because the wheel
has no room, and that is measured rather than estimated: as of
2026-08-17 the platform wheels were 108.70 MB (win_amd64) and 111.70 MB
(manylinux_2_28_x86_64) against PyPI's 100 MB per-file cap, already
over it before a byte of basemap, which deflates to 21.1 MB; see
``tools/stage_wheel_bridges.py``'s docstring for the full table.  So a
wheel install gets them from the ``recast-woof-data`` companion instead, which
every install pulls, and :func:`woof.rustwx.renderer_env` hands that
copy to the wheel's renderer.  Until 2.8.0 this command was their only
route onto a wheel install, no install text ran it, and every picture
of such an install was drawn with no coastlines, borders or state
lines.  The bundle still carries them for the renderer it stages here.

Platform support is a capability check on the OS and machine
architecture -- can this box run the bytes in that bundle -- and never
an identity gate.  A platform with no published bundle is told so by
name and handed the build-from-source route, which remains the
universal one.

The staging contract
--------------------
Every artifact is verified three ways before it is installed: the exact
byte count, the SHA-256 pin packaged with this release, and (for the
decoders that declare one) the contract marker
:data:`woof.bridges.BRIDGE_ABI_MARKERS` -- the same static check
``woof doctor`` applies to a binary already on disk.  A staged file
that fails any of the three is deleted and refused, never installed.
Extraction reads members by their exact pinned filename, so an archive
cannot place a byte anywhere the pins did not name.

Where this deliberately differs from ``fetch-tables``: an existing file
whose bytes do not match the pin is *replaced* here rather than
refused.  A physics table is a fixed asset, so a wrong one on disk is
the operator's to explain; a bridge executable is a versioned build,
and yesterday's copy sitting in ``~/.woof/bridges`` after a Python-half
upgrade is exactly the skew that made 1.1.0 preparations fail with a
message blaming the file woof had just written correctly.  Replacing
it is the point of the command.  The replacement is still gated on the
new bytes passing all three checks first.

Offline and mirrors
-------------------
``--from DIR`` stages from a local directory under identical
verification: either the bundle archive itself, or the declared
artifacts loose in that directory (what an air-gapped operator has
after building them on a machine that does have a toolchain).
``WOOF_BRIDGE_ASSET_URL_BASE`` overrides the download base URL; the
bundle filename is appended to it either way.

Pins are generated at release time
----------------------------------
The bundles are built by the release workflow from the same commit as
the wheel, and ``tools/build_bridge_bundle.py`` computes the pins from
those exact bytes and writes them into :data:`PINS_RESOURCE` before the
wheel is built.  A tree that has not been through that step carries a
pins document declaring no platforms, and this command says so and
refuses rather than inventing a hash to check against.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import platform as platform_module
import shutil
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

from woof import bridges, fetch_guard
from woof.explain import warn

#: Schema of the packaged pins document.
PINS_SCHEMA = "gpuwm-bridge-pins-v1"

#: Schema of the manifest published beside the bundles as a release
#: asset (the human/machine-readable index of what a release shipped).
BUNDLE_MANIFEST_SCHEMA = "gpuwm-bridge-bundle-manifest-v1"

#: The packaged pins document, relative to the ``woof`` package.
PINS_RESOURCE = "data/bridges/bridge-pins.json"

#: Platform keys a release publishes bundles for.  A key names an OS and
#: a machine architecture and nothing else: it answers "can this box run
#: those bytes", never "who is asking".
SUPPORTED_PLATFORMS = ("win-x86_64", "linux-x86_64")

#: Override the download base URL (private mirrors, tests).  The bundle
#: filename is appended to it.
ASSET_URL_BASE_ENV = "WOOF_BRIDGE_ASSET_URL_BASE"

#: Subdirectory of the staging destination holding in-flight and
#: verified bundle archives (kept only under ``--keep-bundle``).
ARCHIVE_SUBDIR = ".fetch-bridges"

#: Prefix, inside a bundle and inside the staging destination, of every
#: data file the renderer reads.  ``rw_wrfbatch`` searches
#: ``assets/basemap`` under the first eight ancestors of its own
#: directory, so staging under this prefix puts the shapefiles where the
#: binary already looks -- the reason the asset half needs no new
#: resolution logic, no environment variable, and no cooperation from
#: the Python half at all.
ASSET_ROOT = "assets"

#: The asset subtrees a bundle is required to carry, relative to
#: :data:`ASSET_ROOT`.  Directories, never filenames: the file list is
#: generated from the tree at pack time and pinned by hash, so it cannot
#: drift the way a hand-maintained enumeration does.  This tuple is the
#: declaration a release is checked against.
REQUIRED_ASSET_SUBDIRS = ("basemap",)

#: Byte marker every bundled artifact's build embeds, followed by the
#: 40-hex git commit the binary was built from.  The 1.4.1 cut nearly
#: shipped bundles predating the source tip -- the two platform zips
#: were not even the same revision -- and every check on the cut path
#: hashed whatever bytes it was handed, so a stale binary pinned and
#: verified perfectly.  The stamp is what makes "built from the release
#: commit" a property of the bytes themselves: the cut extracts it and
#: refuses a mismatch without executing anything (the same static
#: string-in-the-binary convention as :data:`woof.bridges
#: .BRIDGE_ABI_MARKERS` and the renderer's ``GPUWM_INITIAL_*``
#: attribute literals).  The Rust half lives in the workspace build
#: scripts (``tools/grib1_bridge/build.rs`` and the ``build.rs`` of the
#: six bundled ``tools/rustwx`` crates), which inject the checkout's
#: HEAD as ``GPUWM_BRIDGE_SOURCE_REV`` for each entry point to embed.
#:
#: Asked of every artifact whose source moves with this checkout, which
#: is every artifact woof wrote.  A ``vendored`` artifact is proved
#: instead by its contract marker -- see :class:`BundledArtifact`.
SOURCE_REV_MARKER = b"GPUWM_BRIDGE_SOURCE_REV="

#: Character length of a full git commit hash, which is what the stamp
#: carries -- an abbreviation can collide, a branch name can move.
_SOURCE_REV_LENGTH = 40

_BLOCK_BYTES = 8 * 1024 * 1024
_USER_AGENT = "gpuwm-fetch-bridges/1"
_TIMEOUT_S = 120


class BridgeAssetError(RuntimeError):
    """A refusal: wrong bytes, unfetchable bundle, or unusable source."""


# ---------------------------------------------------------------------------
# What a bundle carries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BundledArtifact:
    """One built file a bundle carries, and what it is for.

    ``kind`` is ``"executable"`` or ``"library"``: the two spellings a
    platform gives a built artifact, which is the whole reason the
    filename cannot be derived from the logical name alone.

    ``vendored`` says the crate is a verbatim copy of an upstream tree
    frozen at a recorded commit, and it changes what proves the bytes are
    not stale.  A gpuwm-authored artifact is proved by
    :data:`SOURCE_REV_MARKER`: its source moves with this checkout, so the
    release commit stamped into the binary is the statement that the two
    agree.  A vendored crate does not move with this checkout at all --
    re-vendoring is a commit that edits ``UPSTREAM_COMMIT`` and the
    directory together -- so a HEAD stamp would say nothing about it, and
    injecting one would mean editing a tree whose whole claim is that no
    file in it differs from upstream by a byte.  What is asked of a
    vendored artifact instead is its declared contract marker
    (:data:`woof.bridges.BRIDGE_ABI_MARKERS`), which is a property of the
    bytes and names the ABI this release speaks, on top of the size and
    SHA-256 every artifact is pinned by.

    ``on_demand`` says the platform wheel does not carry the artifact.  It
    stays in this roster, in each platform bundle and in its pins, so every
    proof of the bundle still covers it; the first door that needs it stages
    it by itself (:func:`woof.bridges.fetch_on_demand`, the pinned ZIP
    member alone, verified against the packaged size and SHA-256), and
    ``woof fetch-bridges --partial --artifact NAME`` or a plain
    ``woof fetch-bridges`` stages it ahead of time.  Breakage this
    prevents: the 2.9.0 Linux wheel was 96,049,884 bytes against PyPI's
    100,000,000-byte per-file limit, and the six observation-stream doors
    marked below compress to about 9.7 MB together, so carried in the wheel
    they would make it unpublishable; left out of every package, as they
    were in 2.9.0, buoys, radiosondes, motion vectors, occultations, WIS2
    and prepbufr could not be fetched from an install at all.  Only an
    artifact no forecast reaches may be marked: a decoder, a writer or the
    renderer marked here would make a bare install refuse its own forecast.
    :func:`wheel_artifacts` is the one place the two sets are told apart.
    """

    name: str
    kind: str
    crate: str
    env_var: str
    consumer: str
    vendored: bool = False
    on_demand: bool = False


#: The artifacts a bundle carries (every BundledArtifact below), in build order: the five
#: GRIB decoders and the CPU preprocessing library from the decoder
#: workspace, then the fetch backbone, the batch renderer, the two radar
#: front doors, the five observation front doors, the NetCDF decoder,
#: the NetCDF writer, the static-field builder and the observation remap
#: from the renderer workspace, then the region-global dealiasing library
#: from its own vendored crate and the mapped decode engine from the
#: engine workspace.  The environment variables are
#: the ones the resolution ladder already honours, so a staged bundle
#: and a hand-built tree are found by exactly the same code.
#:
#: ``rw_odim`` joined for the reason ``rw_nexrad`` did, one continent
#: over.  European polar-volume ingest had a complete library path, a
#: ``woof obs radar`` front door, and no way to obtain the binary that
#: path drives without a Rust toolchain and a source checkout -- which by
#: this project's rule means it was not shipped: a capability the engine
#: has and the user cannot reach does not exist.  It builds ``--locked
#: --offline`` from the same vendored closure on the same two platforms
#: as its siblings and carries the same source-revision stamp, so
#: bundling it costs an entry here and nothing else.
#:
#: ``rw_opera`` was the one door still left out when these two lanes were
#: written apart, and for a reason that stopped being true when they were
#: put together: ``crates/rw-obs`` carried no ``build.rs`` and none of its
#: entry points embedded the source-revision stamp, so a cut could not
#: prove the binary it staged.  The observation-front-door lane added that
#: ``build.rs`` and stamped three of the crate's four entry points; the
#: fourth, ``opera.rs``, did not exist on that lane.  Stamping it here is
#: the whole remaining cost, so the composite ships with its siblings
#: rather than waiting for a lane of its own.
#:
#: ``rw_nexrad`` joined this list once it was clear that leaving it out
#: made ``woof doctor`` pass on boxes that could not ingest a single
#: radar observation: it is the only route into the radar-DA nowcast,
#: it has no fallback the way the fetch backbone and the renderer do,
#: and it builds ``--locked --offline`` out of the same vendored
#: closure on the same two platforms as its two workspace siblings.
#: The alternative -- telling every user to install a Rust toolchain --
#: is the one prerequisite this project otherwise never imposes.
#:
#: ``rw_mrms``, ``rw_stage4``, ``rw_asos`` and ``rw_goes`` joined for
#: the same reason, one wave later and one wave too late.  All four were
#: written, tested and committed; all four are resolved out of
#: ``~/.woof/bridges`` by :mod:`woof.obs.frontdoor`; and none of them
#: was in this tuple, so ``woof fetch-bridges`` -- the command their
#: own refusals named -- staged a complete bundle and the refusal
#: repeated verbatim.  That is the ``rw_mpas_convert`` failure exactly:
#: committed is not shipped, and a refusal naming a command that cannot
#: help is worse than no message.  A front door that a resolver looks
#: for and a refusal names belongs in the bundle that resolver reads.
#: The release workspace already BUILT them -- ``cargo build --release
#: --locked`` at ``tools/rustwx`` builds every workspace member -- so
#: the four binaries were produced by the cut, probed by nothing, and
#: thrown away at the end of the job.
BUNDLED_ARTIFACTS: tuple[BundledArtifact, ...] = (
    BundledArtifact("rw-wind-table", "executable", "tools/wind-table",
                    "WOOF_RW_WIND_TABLE", "wind turbine inventories and WRF Fitch tables",
                    # 2.9.11 (PyPI per-file limit): see rw_simradar. Opt-in
                    # (--wind-farms); woof.wind_inputs stages it on first use.
                    on_demand=True),
    # 2.9.3: the Linux wheel measured 110,716,883 bytes against PyPI's 100,000,000-byte per-file limit
    # (rw_soundings and the NextGen tools are new), so the five observation front doors served by
    # woof.obs.frontdoor (rw_mrms, rw_stage4, rw_asos, rw_goes, rw_opera), the sounding window renderer
    # and the NextGen gauge verifier are on demand: pinned in the bundle, staged on first use.
    # 2.9.4: the Linux wheel measured 100,878,471 bytes (the cut's wheel-size pre-gate; rw_hail and
    # rw_fire_indices are new, rw_wrfbatch grew), so five verification and post-run doors no forecast
    # reaches are on demand too: rw_hail and rw_airnow (woof.obs.frontdoor), rw_scoreboard
    # (woof scoreboard), rw_tc_fields (woof tc-tracks) and rw_fire_indices (woof.fire_danger_state).
    # WOOF 1.1.0 r16: the r15 Linux wheel measured 138,837,702 bytes (the hex helpers joined it), so with
    # the hex helpers moved to their own release archive (woof.hex.helpers) four more are on demand:
    # rw_odim (woof.obs.frontdoor, European radar only) and rw_mpas_geometry, rw_mpas_convert and
    # rw_mpas_lbc, which every hex door reads from the hex helpers first (woof.hex.engines rung 3);
    # woof.mpas_mesh and the hex rung 4 stage them on first use.
    BundledArtifact(
        "nextgen_hydrofabric", "library", "tools/nextgen_hydrofabric",
        "WOOF_HYDROFABRIC_LIBRARY",
        "NextGen prepared numeric payload transport into caller-owned memory",
        # 2.9.11 (PyPI per-file limit): see rw_simradar. Hydro is opt-in
        # (--hydro on) and its other tools are already on demand;
        # woof.hydrofabric stages this library on its first load. The
        # terminal workspace stays in the wheel: the publish workflow's
        # platform smoke draws `woof tui --snapshot` from the installed
        # wheel before the release's bundle is public.
        on_demand=True),
    BundledArtifact(
        "rw_nextgen_output", "library", "tools/nextgen_output",
        "WOOF_NEXTGEN_OUTPUT_LIBRARY", "coupled hydro history and restart publication",
        on_demand=True),
    BundledArtifact(
        "arwen_nextgen_forcing", "library", "tools/nextgen_forcing",
        "WOOF_NEXTGEN_FORCING_BRIDGE",
        "NextGen external forcing preparation and catchment area regridding"),
    BundledArtifact(
        "rw_nextgen_verify", "executable", "tools/nextgen_verify",
        "WOOF_NEXTGEN_VERIFY", "NextGen streamflow observation verification",
        on_demand=True),
    # 2.9.3 (improve-1, default-on hydro never attached): the default hydro door builds
    # the covering fabric, its parameters, its reservoirs and its initial water with
    # these five provider tools, which no install could find (woof.hydro.door probe:
    # "builder executable rw_hydro_world not found"). On demand, staged by the door's
    # first preparation, so the wheel stays under PyPI's per-file limit.
    BundledArtifact(
        "rw_hydro_world", "executable", "tools/hydro_world",
        "WOOF_RW_HYDRO_WORLD", "hydro fabric builder (default-on hydro door)",
        on_demand=True),
    BundledArtifact(
        "rw_hydro_params", "executable", "tools/hydro_world_params",
        "GPUWM_RW_HYDRO_PARAMS", "hydro catchment parameters (default-on hydro door)",
        on_demand=True),
    BundledArtifact(
        "rw_hydrofabric", "executable", "tools/nextgen_hydrofabric",
        "WOOF_RW_HYDROFABRIC", "hydro routing and reservoir preparation (default-on hydro door)",
        on_demand=True),
    BundledArtifact(
        "nextgen-reservoir-prepare", "executable", "tools/nextgen_reservoir_prepare",
        "GPUWM_NEXTGEN_RESERVOIR_PREPARE", "hydro reservoir preparation (default-on hydro door)",
        on_demand=True),
    BundledArtifact(
        "rw_hydro_init", "executable", "tools/hydro_world_init",
        "WOOF_RW_HYDRO_INIT", "hydro initial water state (default-on hydro door)",
        on_demand=True),
    BundledArtifact(
        "rw_ocean", "executable", "tools/ocean_prepare",
        "GPUWM_RW_OCEAN", "Ocean bathymetry and atmospheric forcing preparation",
        on_demand=True),
    BundledArtifact(
        "rw_bathy", "executable", "tools/rw-bathy",
        "WOOF_RW_BATHY", "storm-surge bathymetry ingest and products (woof.ocean.surge)",
        on_demand=True),
    BundledArtifact(
        "grib1_bridge", "executable", bridges.CRATE_RELATIVE,
        bridges.BRIDGE_ENV["grib1_bridge"],
        "ERA5 route (woof check/run, rw-wps --source era5)"),
    BundledArtifact(
        "gdt101_remap", "executable", bridges.CRATE_RELATIVE,
        bridges.BRIDGE_ENV["gdt101_remap"],
        "unstructured GDT-101 source normalization "
        "(woof go --source icon-global or icon-d2)"),
    # 2.9.0 (box/rrfs-shape) defect: public initial-condition preparation needs this producer,
    # but fetch-bridges previously omitted the already-built executable.
    BundledArtifact(
        "met_intermediate", "executable", bridges.CRATE_RELATIVE,
        bridges.BRIDGE_ENV["met_intermediate"],
        "regular-grid meteorological intermediates for native initialization"),
    BundledArtifact(
        "gfs_grib2_bridge", "executable", bridges.CRATE_RELATIVE,
        bridges.BRIDGE_ENV["gfs_grib2_bridge"],
        "GFS front door (rw-wps --source gfs)"),
    BundledArtifact(
        "hrrr_grib2_bridge", "executable", bridges.CRATE_RELATIVE,
        bridges.BRIDGE_ENV["hrrr_grib2_bridge"],
        "HRRR front door (rw-wps --source hrrr)"),
    BundledArtifact(
        "grib2_inventory", "executable", bridges.CRATE_RELATIVE,
        bridges.BRIDGE_ENV["grib2_inventory"],
        "20CRv3/mapped GRIB2 routes"),
    BundledArtifact(
        "grib2_dump", "executable", bridges.CRATE_RELATIVE,
        bridges.BRIDGE_ENV["grib2_dump"],
        "20CRv3/mapped GRIB2 routes"),
    BundledArtifact(
        "gpuwm_preprocess_cpu", "library", bridges.CRATE_RELATIVE,
        "WOOF_CPU_PREPROCESS_BRIDGE",
        "--preprocess-backend cpu, and the masked surface fields "
        "(soil, snow, skin, sea ice) under every backend"),
    BundledArtifact(
        "rw_fetch", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_FETCH", "woof fetch --engine rust"),
    BundledArtifact(
        "rw_globe_pack", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_GLOBE_PACK", "global globe frames and verified statics publication"),
    BundledArtifact(
        "rw_wrfbatch", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_WRFBATCH", "woof render --engine rust"),
    BundledArtifact(
        "rw_ensbatch", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_ENSEMBLE_RENDERER",
        "ensemble diagnostic reduction and aggregate product rendering"),
    BundledArtifact(
        "rw_probgrid", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_PROBGRID", "post-processed probability maps on a fixed ladder (WOOF severe ML)",
        # 2.9.9 (PyPI per-file limit): no forecast reaches it, and carried in the
        # wheel it took the Linux platform wheel past 100,000,000 bytes.
        on_demand=True),
    BundledArtifact(
        "rw_compare", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_COMPARE", "native reference panels and observation overlays"),
    BundledArtifact(
        "rw_verify", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_VERIFY", "woof verify-visuals and completed-run verification"),
    BundledArtifact(
        "rw_scorecard", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_SCORECARD", "woof obs scorecard and paired case bootstrap"),
    # The simulated radar.  Beam tracing, field sampling, the Level II,
    # CfRadial and ODIM writers and the PPI images all run in it, so a
    # bundle without it is an install whose `[simulated_radar]` table and
    # `woof simulated-radar` door can only refuse.  Its own build.rs
    # stamps the source revision like every gpuwm-authored row here.
    # Environment variable spelled to match woof.rustwx's resolver.
    BundledArtifact(
        "rw_simradar", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_SIMRADAR", "WOOF simulated radar volumes and PPI images",
        # 2.9.11 (PyPI per-file limit): the C- and X-band scattering tables
        # compiled into it (14.5 MB of float tables) took it from 6.2 to
        # 16.6 MB compressed, and with the CUDA 12/13 kernel packs (8.3 MB)
        # the Linux platform wheel measured 118,951,005 bytes against
        # 100,000,000.  Simulated radar is opt-in ([simulated_radar], woof
        # simulated-radar, woof.simulated_radar.simulate); every door
        # reaches it through rustwx.require_simulated_radar_binary, which
        # stages it from this release's bundle on its first use. With it,
        # rw_mlexport, rw-wind-table and nextgen_hydrofabric on demand, about 96.1 MB.
        on_demand=True),
    BundledArtifact(
        "rw_nexrad", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_NEXRAD", "radar observation ingest (the DA nowcast)"),
    BundledArtifact(
        "rw_odim", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_ODIM",
        "European polar volumes (woof obs radar)",
        on_demand=True),
    BundledArtifact(
        "rw_mrms", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_MRMS", "MRMS composite reflectivity (woof obs mrms)",
        on_demand=True),
    BundledArtifact(
        "rw_stage4", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_STAGE4",
        "Stage-IV precipitation (woof obs stage4)",
        on_demand=True),
    BundledArtifact(
        "rw_asos", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_ASOS",
        "ASOS/METAR surface observations (woof obs asos)",
        on_demand=True),
    BundledArtifact(
        "rw_ice_verify", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_ICE_VERIFY", "freezing-rain and ice observation binding",
        on_demand=True),
    BundledArtifact(
        "rw_hail", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_HAIL", "MRMS MESH and SPC hail size verification",
        on_demand=True),
    BundledArtifact(
        "rw_airnow", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_AIRNOW",
        "AirNow hourly air-quality observations",
        on_demand=True),
    BundledArtifact(
        "rw_goes", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_GOES",
        "GOES ABI cloud-water-path packs (woof obs goes)",
        on_demand=True),
    BundledArtifact(
        "rw_opera", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_OPERA",
        "European composite reflectivity (woof obs opera)",
        on_demand=True),
    # The NetCDF decoder.  It joined this list the moment NetCDF decode
    # moved off `netCDF4.Dataset` and onto `netcrust`: from that commit
    # `rw-wps --source netcdf` and every mapped NetCDF route REFUSE
    # without it, by design and by name.  A bundle that omits it is a
    # bundle that cannot read the one source format a user is most
    # likely to bring of their own -- and the alternative, telling them
    # to install a Rust toolchain, is the prerequisite this project
    # otherwise never imposes.
    BundledArtifact(
        "rw_netcdf", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_NETCDF", "NetCDF sources (rw-wps --source netcdf, "
        "woof adapt, the mapped routes)"),
    # The dealiasing engine every radar ingest gets by default.
    # It joined this list the day it became the default and for the same
    # reason `rw_nexrad` did: a prerequisite of the shipped configuration
    # that a wheel install cannot satisfy is a wheel install that cannot
    # run the shipped configuration, and the alternative is telling every
    # user to install a Rust toolchain.  It is the first VENDORED entry
    # here -- a verbatim copy of an upstream crate, built the same way on
    # the same two platforms, whose staleness is proved by its contract
    # marker rather than by a stamp of this checkout's HEAD.
    # The crate path and the environment variable are spelled out rather
    # than imported from `woof.obs.dealias_region`, exactly as the other
    # nine are: this module is reached by `woof fetch-bridges` on a fresh
    # wheel and has no business importing the observation stack to read
    # two strings.  `tests/test_bridge_fetch.py` binds both to that
    # module's own constants.
    BundledArtifact(
        "region_global_dealias", "library", "tools/region_global_dealias",
        "WOOF_DEALIAS_REGION_BRIDGE",
        "velocity dealiasing (on by default; --no-dealias opts out)",
        vendored=True),
    # The NetCDF WRITER.  It joined this list the moment the product tape
    # flipped onto it BY DEFAULT: from that commit `woof sim` and every
    # history write REFUSE without it, by design and by name
    # (`woof.io.wrfout.WrfoutWriter` names the breakage and the
    # WOOF_WRFOUT_WRITER=python workaround), so a wheel that omits it is
    # a wheel that cannot write its own forecast.  The wrfinput/wrfbdy
    # pair then flipped onto the same library, so such a wheel now cannot
    # write its own WRF INPUT either -- `woof prep` refuses the same way,
    # naming WOOF_WRFINPUT_WRITER=python.  Environment variable spelled
    # to match woof.io.nc_writer_bridge.NCWRITE_BRIDGE_ENV, and a test
    # binds the two.
    BundledArtifact(
        "netcdf_writer", "library", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_NCWRITE_BRIDGE",
        "wrfout history writes and the wrfinput/wrfbdy export "
        "(both default engines)"),
    # The STATIC-FIELD builder.  It joined this list the moment
    # `build_static` and the ProjectedGrid array methods flipped onto the
    # Rust crate BY DEFAULT (the static-rust-port lanes): from that
    # commit every bare static build on a wheel without it is a reported
    # WORKAROUND onto the numpy fallback -- a degradation, never the
    # shipped configuration.  Environment variable spelled to match
    # woof.static.rust_bridge.STATIC_BRIDGE_ENV; a test binds the two
    # (tests/test_static_rust_parity.py::TestEstateAndDoctor).
    BundledArtifact(
        "static_fields", "library", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_STATIC_BRIDGE",
        "static-field builds (the default geogrid-equivalent builder)"),
    # The MAPPED DECODE ENGINE, and the plainest case on this list.
    # `woof.mapped_engine_bridge.DEFAULT_ENGINE` is `rust`, so every
    # mapped source -- the route almost every registered source reaches
    # canonical frames through -- decodes in this binary on a bare run.
    # It was left out of this tuple when the route flipped, and the
    # measurement of what that costs is not a hypothetical: a fresh
    # py3.14 wheel install reports `MISSING mapped decode engine ...
    # blocks every mapped source, which is the default decode path`, and
    # no woof command could supply it -- `woof fetch-bridges` staged a
    # complete bundle and the gap survived, because the roster it reads
    # is this one.  A wheel user's only route was a clone and a cargo
    # build, for the DEFAULT decode path.  That is the rw_mpas_convert
    # failure exactly, one layer more expensive: committed is not
    # shipped, and the alternative -- telling every user to install a
    # Rust toolchain -- is the prerequisite this project otherwise never
    # imposes.
    #
    # It is the first entry from the engine workspace (`tools/rw_wps`),
    # which builds `--locked --offline` from its own vendored closure
    # exactly as the other two workspaces do.  The crate path and the
    # environment variable are spelled out rather than imported from
    # `woof.mapped_engine_bridge` for the reason the dealiasing entry
    # gives: this module is reached by `woof fetch-bridges` on a fresh
    # wheel and has no business importing the decode stack to read two
    # strings.  `tests/test_bridge_fetch.py` binds both to that module's
    # own constants.  NOT vendored: `crates/mapped-engine` is woof's
    # own crate beside the donor snapshot (VENDOR.md), so it is proved
    # by the source-revision stamp its own `build.rs` injects, like
    # every other gpuwm-authored artifact here.
    BundledArtifact(
        "gpuwm_mapped_engine", "executable", "tools/rw_wps",
        "WOOF_MAPPED_ENGINE_BIN",
        "mapped-source decode (the default decode path)"),
    # The OBSERVATION REMAP.  It joined this list the moment
    # `woof.verify.obs.regrid`'s plan build and apply flipped onto the
    # Rust crate BY DEFAULT: from that commit every observation score on
    # a wheel without it is a reported WORKAROUND onto the scipy
    # fallback, whose exact-tie answers are cKDTree traversal order
    # rather than a rule.  Environment variable spelled to match
    # woof.obs_regrid_bridge.OBSREGRID_BRIDGE_ENV; a test binds
    # the two (tests/test_obs_regrid_rust_parity.py).
    BundledArtifact(
        "obs_regrid", "library", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_OBSREGRID_BRIDGE",
        "observation remap plans (the default battery remap engine)"),
    BundledArtifact(
        "obs_score", "library", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_OBSSCORE_BRIDGE",
        "observation fractions, contingency and station scoring"),
    # The ISOBARIC-HEIGHT READER.  It joined this list the moment the
    # Python consumers of a height on a pressure surface moved onto the
    # Rust crate every chart already reads with: the vortex tracker on a
    # host state, the GNSS-RO refractivity operator, the verification maps
    # and the flagship products.  There is no Python implementation behind
    # it, so a wheel without it refuses those reads by name.  Environment
    # variable spelled to match woof.isobaric_bridge.ISOBARIC_BRIDGE_ENV;
    # a test binds the two (tests/test_bridge_fetch.py).
    BundledArtifact(
        "rw_isobaric", "library", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_ISOBARIC_BRIDGE",
        "isobaric heights read between layer interfaces (vortex tracker, "
        "GNSS-RO operator, verification maps)"),
    # The RADAR SUPEROB.  It joined this list the moment the superob
    # stage of the radar observation build flipped onto the Rust crate by
    # default.  The stage refuses rather than falls back when the library
    # is absent (the numpy module is the parity reference and an explicit
    # opt-out, WOOF_SUPEROB_PYTHON=1), so a wheel without it could not
    # build one radar observation: every radar DA cycle would refuse an
    # hour in.  Environment variable spelled to match
    # woof.obs.superob_bridge.SUPEROB_BRIDGE_ENV; a test binds the two
    # (tests/test_bridge_fetch.py).
    BundledArtifact(
        "rw_superob", "library", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_SUPEROB_BRIDGE",
        "radar superob, merge and region-global dealias (the default "
        "radar observation build)"),
    # The four MPAS binaries of the first wave (the fifth, the
    # lateral-boundary producer, is the last entry in this tuple and
    # carries its own note), and `rw_mpas_convert` is the artifact this
    # whole tuple's comment block keeps naming as the precedent: it was
    # written, committed, resolved by name and carried by no bundle, so
    # the refusals that named `woof fetch-bridges` sent readers to a
    # command that staged a complete bundle without it.  It stayed in
    # that state one wave after the observation front doors were fixed
    # for the identical reason, and `rw_mpas_mesh` -- the generator that
    # reproduces the published NCAR meshes to 1e-11 -- joined it there
    # the day it was built: no bundle row, no marker, no environment
    # variable, no resolver and no Python that called it.
    #
    # All four build `--locked --offline` out of the same vendored
    # closure on the same two platforms as their eleven `tools/rustwx`
    # siblings, and the release workspace already BUILT them -- `cargo
    # build --release --locked` at `tools/rustwx` builds every workspace
    # member -- so the binaries were produced by the cut, probed by
    # nothing, and thrown away at the end of the job.  Bundling them
    # costs four entries here and the `build.rs` that stamps the crate.
    #
    # Environment variables spelled to match woof.mpas_mesh.BRIDGES; a
    # test binds each row to that module's own resolver.
    BundledArtifact(
        "rw_mpas_mesh", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_MPAS_MESH",
        "MPAS mesh generation (woof mesh)"),
    # The other half of what `woof mesh` delivers, and it is not
    # optional: a grid file reaches no dycore on its own, because the
    # mesh registry pins BOTH the grid and a matching static by byte
    # count and sha256.  Shipping the generator without the static
    # builder ships a command whose output nothing accepts.
    BundledArtifact(
        "rw_mpas_static", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_MPAS_STATIC",
        "the matching MPAS static (woof mesh)"),
    BundledArtifact(
        "rw_mpas_init", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_MPAS_INIT",
        "MPAS initial conditions from a grid and a static file"),
    BundledArtifact(
        "rw_mpas_geometry", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_MPAS_GEOMETRY",
        "MPAS reconstruction geometry",
        on_demand=True),
    BundledArtifact(
        "rw_mpas_hostprep", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_MPAS_HOSTPREP",
        "MPAS forecast host preparation"),
    BundledArtifact(
        "rw_mpas_convert", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_MPAS_CONVERT",
        "MPAS history onto the renderer's tape",
        on_demand=True),
    # The lateral-boundary producer, and the plainest repeat of the
    # failure every paragraph above describes.  Its SOURCE has been in
    # this tree since the limited-area lane landed
    # (`tools/rustwx/crates/rw-mpas/src/bin/rw_mpas_lbc.rs`); the
    # release workspace has BUILT it at every cut since, because `cargo
    # build --release --locked` at `tools/rustwx` builds every workspace
    # member and this is a declared `[[bin]]`; and no published bundle
    # has ever carried it, because this tuple is the roster `pack`
    # walks and it was not in it.  Source presence is not shipping: the
    # binary was produced by the cut, probed by nothing, and thrown
    # away at the end of the job, four times over.
    #
    # What its absence costs is not an optional extra.  A limited-area
    # mesh is two halves: `rw_mpas_mesh --cull-parent` cuts the mesh,
    # and this produces the boundary series that mesh is driven by.
    # Shipping the cull without the producer ships a mesh nothing can
    # run -- the same shape as shipping the generator without
    # `rw_mpas_static`, which is why that one is in this tuple too.
    BundledArtifact(
        "rw_mpas_lbc", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_MPAS_LBC",
        "MPAS lateral boundaries for a limited-area mesh",
        on_demand=True),
    # The RRFS-MPAS init door's two engines (woof hex init --physics-suite
    # hrrrv5): the fork's init_atmosphere in Rust, and the GSL drag
    # statistics generator.  Source-only bins reach no pip install, so
    # without these rows the door's producer is unreachable.
    BundledArtifact(
        "rw_mpas_init_v831", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_MPAS_INIT_V831",
        "RRFS-MPAS initial conditions and boundaries (the fork's init_atmosphere)",
        # 2.9.10 (PyPI per-file limit): carried in the wheel, it and
        # rw_mpas_gsl_oro (~3.0 MB compressed together) took the Linux platform
        # wheel (2.9.9: 98,751,849 bytes) past 100,000,000 bytes. Only the
        # woof hex RRFS init door reaches them, and it stages them on first use.
        on_demand=True),
    BundledArtifact(
        "rw_mpas_gsl_oro", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_MPAS_GSL_ORO",
        "GSL orographic drag statistics (ugwp_oro_data.nc) for the RRFS suites",
        # 2.9.10 (PyPI per-file limit): see rw_mpas_init_v831 above.
        on_demand=True),
    # 2.9.0.  The woof hex product door draws soundings and meteograms
    # with this binary and nothing else.  As a source-only cargo bin it
    # reached no pip install: `woof hex doctor` exited 1 naming it and
    # `woof fetch-bridges` could not supply it.
    BundledArtifact(
        "rw_hex_points", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_HEX_POINTS",
        "soundings and meteograms at named points from MPAS native history "
        "(the woof hex product door)"),
    # 2.9.1.  The projected MET intermediate library (crate
    # projected-intermediate in the engine workspace), loaded through
    # ctypes by `woof hex intermediate` for every projected source (HRRR
    # and any other Lambert regional model).  It regrids the decoder's
    # window onto the hex point target with the engine's own exact donors,
    # rotates the winds to earth and writes and verifies the MET file.  It
    # was built by nothing that shipped, so on 2.9.0 a pip install could
    # start hex only from lat-lon sources: `woof hex intermediate --source
    # hrrr` exited 2 with "librw_projected_intermediate.so was not found".
    # Environment variable spelled to match the one woof hex resolves.
    BundledArtifact(
        "rw_projected_intermediate", "library", "tools/rw_wps",
        "WOOF_PROJECTED_INTERMEDIATE_BRIDGE",
        "projected-source MET intermediates for hex point runs "
        "(woof hex intermediate --source hrrr)"),
    # 2.9.0.  TEMPO's tempo_init port (crate rw-tempo), loaded through
    # ctypes by woof.core.tempo_rust whenever a run selects TEMPO.  It was
    # built by every cut and carried by no bundle, so TEMPO -- and the
    # rrfs-v2-hex-tempo preset with it -- could not initialize from a pip
    # install: the loader's last word was "build it from a checkout".
    # Environment variable spelled to match
    # woof.core.tempo_rust.TEMPO_BRIDGE_ENV.
    BundledArtifact(
        "rw_tempo", "library", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_TEMPO_BRIDGE",
        "TEMPO microphysics initialization tables (mp TEMPO, the "
        "rrfs-v2-hex-tempo preset)"),
    # The native GRIB2 exporter (2.8.7, crate rw-grib2export): reads
    # history, computes every product with the clean-room post-processor
    # rw-post (GPU by default, CPU reference path) and packs GRIB2 in Rust.
    # An install without it cannot run `export-grib2`, `go --grib2` or
    # `render --grib2-out`.
    BundledArtifact(
        "rw_grib2export", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_GRIB2EXPORT", "native GRIB2 history export (woof export-grib2)"),
    # The nowcast frames converter (2.8.7, crate rw-wrfbatch): a 2D
    # reflectivity NetCDF nowcast written out as the
    # `gpuwm-obs.nowcast-frames.v1` contract, the only form the
    # comparison sheets (`rw_compare --panel LABEL=frames:ROOT@VALID`) and
    # the radar-heating adapter read.  Without this row the cut built it,
    # probed nothing and threw it away, so a wheel install had no way to
    # put a NetCDF nowcast on a StormScope | WOOF | HRRR | MRMS sheet.
    # Environment variable spelled to match
    # woof.rustwx_lanes.NOWCAST_FRAMES_ENV; a test binds the two.
    BundledArtifact(
        "rw_nowcast_frames", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_NOWCAST_FRAMES",
        "NetCDF nowcasts as frames roots for the comparison sheets "
        "(rw_nowcast_frames from-netcdf)"),
    # The ML dataset exporter.  Every array operation `woof ml-export`
    # performs happens in it (the history read, the vertical interpolation,
    # the regrid, the Zarr and ZIP writing), so a bundle without it is an
    # install whose `ml-export` can only refuse.  Environment variable
    # spelled to match woof.ml_export.BINARY_ENV; a test binds the two.
    BundledArtifact(
        "rw_icechunk", "executable", "tools/rw_icechunk",
        "WOOF_RW_ICECHUNK", "atomic native history output (Icechunk)",
        # 2.9.10 (PyPI per-file limit): ~10.8 MB compressed per platform wheel,
        # for an opt-in output (--output-format icechunk); its first use stages it.
        on_demand=True),
    BundledArtifact(
        "rw_mlexport", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_MLEXPORT",
        "machine-learning dataset export (woof ml-export)",
        # 2.9.11 (PyPI per-file limit): see rw_simradar. A post-run door no
        # forecast reaches; woof ml-export stages it on its first use.
        on_demand=True),
    BundledArtifact(
        "rw_tc_fields", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_TC_FIELDS", "completed-output cyclone field mapping (woof tc-tracks)",
        on_demand=True),
    # The sounding window renderer (the website sounding window drawn from
    # plain run output).  Environment variable spelled to match
    # woof.soundings.ENV; the doctor line resolves through that module.
    BundledArtifact(
        "rw_soundings", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_SOUNDINGS", "website-style sounding windows (woof soundings)",
        on_demand=True),
    BundledArtifact(
        "rw_scoreboard", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_SCOREBOARD", "incremental forecast verification (woof scoreboard)",
        on_demand=True),
    BundledArtifact(
        "rw_zarr", "executable", "tools/zarr_bridge",
        "WOOF_RW_ZARR", "regular-grid Zarr acquisition and forcing record decode"),
    # 2.9.1.  The observation-stream doors of `woof obs streams` and the
    # local data assimilation (woof.obs.streams, woof.da.obs_conventional).
    # Until 2.9.1 they were in no wheel and no bundle, so `woof obs streams
    # fetch --stream ndbc` (and every stream but METAR) refused on any install
    # with a cargo build from a clone.  On demand: see BundledArtifact.
    # Environment variables spelled to match woof.obs.frontdoor; a test
    # binds the two.
    BundledArtifact(
        "rw_igra2", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_IGRA2", "IGRA2 radiosondes (woof obs streams fetch --stream igra2)",
        on_demand=True),
    BundledArtifact(
        "rw_amv", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_AMV", "GOES derived motion winds (woof obs streams fetch --stream goes-dmw)",
        on_demand=True),
    BundledArtifact(
        "rw_ndbc", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_NDBC", "NDBC buoys and ships (woof obs streams fetch --stream ndbc)",
        on_demand=True),
    BundledArtifact(
        "rw_ocean_verify", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_OCEAN_VERIFY", "ocean SST verification (woof.ocean.mixed_layer.verify)",
        on_demand=True),
    BundledArtifact(
        "rw_gnssro", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_GNSSRO",
        "GNSS radio occultation (woof obs streams fetch --stream gnss-ro / cdaac-ro)",
        on_demand=True),
    BundledArtifact(
        "rw_wis2", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_WIS2", "the WIS2 global feed (woof obs streams fetch --stream wis2)",
        on_demand=True),
    BundledArtifact(
        "rw_prepbufr", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_PREPBUFR", "NCEP prepbufr conventional observations (local DA)",
        on_demand=True),
    # 2.9.2.  MADIS aircraft, radiosondes and mesonets from the public
    # real-time and archive files (woof obs streams fetch --stream
    # madis-acars / madis-raob / madis-mesonet).  The door was built in the
    # tree and reached by nothing: the stream table called MADIS
    # account-gated, which the public files never were.
    BundledArtifact(
        "rw_madis", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_MADIS",
        "MADIS aircraft, radiosondes and mesonets (woof obs streams fetch --stream madis-*)",
        on_demand=True),
    # 2.9.2.  The listings that feed rw_odim: without them a European volume
    # was reachable only by a URL the user already knew, so the polar-volume
    # door had no live or archive route.  BowEcho's catalog adapters plus
    # FMI/DMI archive lookup; its build.rs stamps the source revision.  On
    # demand: no forecast reaches it, and it carries an HTTP client.
    BundledArtifact(
        "rw_radar_feeds", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_RW_RADAR_FEEDS",
        "national radar-volume feed listings (woof obs radar-feeds)",
        on_demand=True),
    BundledArtifact(
        "rw_fire_indices", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "WOOF_FIRE_INDICES", "stateful Canadian FWI and KBDI (woof.fire_danger_state)",
        on_demand=True),
)


#: The ABI handshake of every ``kind == "library"`` artifact, by artifact
#: name: the symbol a loader resolves and the version that symbol must
#: answer.  Per artifact, and declared exactly once.
#:
#: Every library used to be probed with `gpuwm_preprocess_cpu`'s symbol,
#: which was true while there was one library and became false the day
#: the vendored dealiasing cdylib joined the bundle exporting
#: `bw_abi_version` instead.  The workflow's per-runner probe was taught
#: the difference; `tools/verify_release_artifacts.py` was not, so the
#: 2.1.0 prepare job refused a correct bundle after the tag was public.
#: The lesson is not "fix the second copy" but "have one": this table is
#: the single source both the release verifier and the workflow probe
#: read, so a third library cannot be added to `BUNDLED_ARTIFACTS` and
#: quietly inherit some other library's handshake.
LIBRARY_ABI: dict[str, tuple[str, int]] = {
    "rw_nextgen_output": ("rw_nextgen_output_abi_version", 1),
    "nextgen_hydrofabric": ("arwen_hydrofabric_abi_version", 1),
    "arwen_nextgen_forcing": ("arwen_forcing_abi_version", 1),
    "gpuwm_preprocess_cpu": ("gpuwm_preprocess_cpu_abi_version", 1),
    "region_global_dealias": ("bw_abi_version", 1),
    # Matches woof.io.nc_writer_bridge.NCWRITE_ABI; a test binds them.
    "netcdf_writer": ("gpuwm_ncwrite_abi_version", 1),
    # Matches woof.static.rust_bridge.STATIC_ABI; a test binds them.
    "static_fields": ("gpuwm_static_abi_version", 1),
    # Matches woof.obs_regrid_bridge.OBSREGRID_ABI; a test
    # binds them.
    "obs_regrid": ("gpuwm_obsregrid_abi_version", 1),
    "obs_score": ("gpuwm_obsscore_abi_version", 1),
    # Matches woof.isobaric_bridge.ISOBARIC_ABI; a test binds them.
    "rw_isobaric": ("gpuwm_isobaric_abi_version", 1),
    # Matches woof.core.tempo_rust.ABI_VERSION.
    "rw_tempo": ("gpuwm_tempo_abi_version", 1),
    # Matches woof.obs.superob_bridge.SUPEROB_ABI; a test binds them.
    "rw_superob": ("gpuwm_superob_abi_version", 1),
    # Exported by crates/projected-intermediate/src/lib.rs.
    "rw_projected_intermediate": ("gpuwm_projected_intermediate_abi_version", 1),
}


def wheel_artifacts() -> tuple[BundledArtifact, ...]:
    """The artifacts a platform wheel carries, in roster order.

    Every declared artifact except the on-demand ones.  Staging into the
    package, the packaged ``BUNDLE.json`` and every check of an installed
    or built wheel's native inventory read this, so the set is decided
    once.  The bundle and the pins are not narrowed by it: they carry
    :data:`BUNDLED_ARTIFACTS` whole.
    """
    return tuple(artifact for artifact in BUNDLED_ARTIFACTS if not artifact.on_demand)


def on_demand_artifacts() -> tuple[BundledArtifact, ...]:
    """The artifacts the bundle carries and the platform wheel does not."""
    return tuple(artifact for artifact in BUNDLED_ARTIFACTS if artifact.on_demand)


def library_abi_for(name: str) -> tuple[str, int]:
    """``(symbol, expected_version)`` for the library artifact ``name``.

    Fail closed.  A library artifact with no entry here is refused
    rather than probed with a neighbour's symbol: the wrong symbol
    either fails to resolve (a confusing refusal blamed on the bytes) or
    resolves against a library that happens to export the same name and
    reports a handshake nobody checked.  Adding a library therefore
    means adding its entry, and the refusal says so by name.
    """

    try:
        return LIBRARY_ABI[name]
    except KeyError:
        raise BridgeAssetError(
            f"{name}: library artifact with no declared ABI handshake in "
            "woof.bridge_assets.LIBRARY_ABI, so nothing says which "
            "exported symbol proves these bytes speak this release's "
            "contract; declare it there beside BUNDLED_ARTIFACTS"
        ) from None


def artifact_filename(artifact: BundledArtifact, platform: str) -> str:
    """The filename ``artifact`` is built under on ``platform``.

    Parametric in the platform rather than in the host, because the
    release tool inspects both bundles from one machine.  On the host it
    agrees with the resolvers woof already uses
    (:func:`woof.bridges.executable_name` and the CPU backend's own
    library naming); a test binds the two.
    """

    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"unknown platform {platform!r}; known: "
                         f"{SUPPORTED_PLATFORMS}")
    windows = platform.startswith("win-")
    if artifact.kind == "library":
        return f"{artifact.name}.dll" if windows else f"lib{artifact.name}.so"
    return f"{artifact.name}.exe" if windows else artifact.name


def host_platform() -> str | None:
    """This box's platform key, or None when no bundle is published.

    A capability check: the operating system and the machine
    architecture, which together decide whether the bytes in a bundle
    can execute here.  None is not a refusal of service -- it routes to
    the build-from-source remedy, which works everywhere.
    """

    machine = platform_module.machine().lower()
    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        return "win-x86_64"
    if sys.platform.startswith("linux") and machine == "x86_64":
        return "linux-x86_64"
    return None


def host_platform_description() -> str:
    """What :func:`host_platform` looked at, for an accurate refusal."""

    return f"{sys.platform}/{platform_module.machine() or 'unknown'}"


# ---------------------------------------------------------------------------
# The pins document
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BinaryPin:
    """One artifact inside a bundle, pinned by exact size and SHA-256."""

    artifact: str
    filename: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class AssetPin:
    """One data file inside a bundle, pinned by exact size and SHA-256.

    ``path`` is the archive member name and, unchanged, the path the
    file is staged to under the destination -- a relative POSIX path
    beginning with :data:`ASSET_ROOT`.  Unlike a :class:`BinaryPin` it
    names no artifact, no environment variable and no ABI marker: a
    shapefile is data, identical on every platform, and the only
    questions worth asking of it are how many bytes it is and whether
    they hash to what the release published.
    """

    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class BundlePin:
    """One platform's bundle: the archive, pinned, and its contents.

    ``assets`` defaults to empty so a pins document written before the
    asset half existed still parses; a *release* with no assets is
    refused by ``tools/build_bridge_bundle.py pin``, which is where that
    staleness would otherwise be introduced.
    """

    platform: str
    filename: str
    bytes: int
    sha256: str
    binaries: tuple[BinaryPin, ...]
    assets: tuple[AssetPin, ...] = ()


@dataclass(frozen=True)
class BridgePins:
    """The packaged pins: which release, and one bundle per platform."""

    release: str | None
    platforms: dict[str, BundlePin]

    def bundle_for(self, platform: str | None) -> BundlePin | None:
        if platform is None:
            return None
        return self.platforms.get(platform)


def packaged_pins_path() -> Path:
    """Path of the pins document inside this installation."""

    return Path(__file__).resolve().parent / Path(PINS_RESOURCE)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BridgeAssetError(f"bridge pins document is malformed: {message}")


def parse_pins(payload: object, *, origin: str = "pins") -> BridgePins:
    """Validate a pins document and return it, or refuse.

    A malformed pins document must fail loudly here rather than produce
    a bundle nobody can verify: every field this staging path relies on
    is required, and a hash is required to look like one.
    """

    _require(isinstance(payload, dict), f"{origin} is not an object")
    assert isinstance(payload, dict)  # narrowed by _require
    _require(payload.get("schema") == PINS_SCHEMA,
             f"{origin} schema is {payload.get('schema')!r}, expected "
             f"{PINS_SCHEMA!r}")
    release = payload.get("release")
    _require(release is None or (isinstance(release, str) and release),
             f"{origin} release must be a non-empty string or null")
    raw_platforms = payload.get("platforms")
    _require(isinstance(raw_platforms, dict),
             f"{origin} platforms must be an object")
    assert isinstance(raw_platforms, dict)
    platforms: dict[str, BundlePin] = {}
    for key, record in sorted(raw_platforms.items()):
        if key not in SUPPORTED_PLATFORMS:
            # A pins document from a newer release may name a platform
            # this build does not know; that entry is irrelevant to
            # this host and skipping it keeps the document usable.
            warn(f"{origin} declares platform {key!r} this build does "
                 "not know; skipping that entry")
            continue
        _require(isinstance(record, dict), f"{key}: record is not an object")
        bundle = record.get("bundle")
        _require(isinstance(bundle, dict), f"{key}: bundle is not an object")
        binaries_raw = record.get("binaries")
        _require(isinstance(binaries_raw, list) and binaries_raw,
                 f"{key}: binaries must be a non-empty list")
        binaries: list[BinaryPin] = []
        for entry in binaries_raw:
            _require(isinstance(entry, dict),
                     f"{key}: a binaries entry is not an object")
            binaries.append(BinaryPin(
                artifact=_string(entry, "artifact", key),
                filename=_string(entry, "filename", key),
                bytes=_size(entry, key),
                sha256=_digest(entry, key)))
        assets_raw = record.get("assets", [])
        _require(isinstance(assets_raw, list),
                 f"{key}: assets must be a list")
        assets: list[AssetPin] = []
        for entry in assets_raw:
            _require(isinstance(entry, dict),
                     f"{key}: an assets entry is not an object")
            assets.append(AssetPin(
                path=_asset_path(entry, key),
                bytes=_size(entry, key),
                sha256=_digest(entry, key)))
        _require(release is not None,
                 f"{key}: a platform is pinned but the release is null")
        platforms[key] = BundlePin(
            platform=key, filename=_string(bundle, "filename", key),
            bytes=_size(bundle, key), sha256=_digest(bundle, key),
            binaries=tuple(binaries), assets=tuple(assets))
    return BridgePins(release=release, platforms=platforms)


def _string(record: dict, field: str, where: str) -> str:
    value = record.get(field)
    _require(isinstance(value, str) and value,
             f"{where}: {field} must be a non-empty string")
    assert isinstance(value, str)
    return value


def _asset_path(record: dict, where: str) -> str:
    """A relative asset path under :data:`ASSET_ROOT`, or a refusal.

    The pins document decides where staging writes, so it is the place
    to refuse a path that could escape the destination.  Absolute paths,
    drive letters, backslashes and ``..`` components are all rejected
    here rather than left for the filesystem to interpret; staging then
    joins a path it already knows is contained.
    """

    value = _string(record, "path", where)
    parts = value.split("/")
    _require(
        not value.startswith("/") and "\\" not in value and ":" not in value
        and all(part not in ("", ".", "..") for part in parts)
        and parts[0] == ASSET_ROOT and len(parts) > 1,
        f"{where}: asset path {value!r} must be a relative path under "
        f"{ASSET_ROOT!r}/ with no '..' component")
    return value


def _size(record: dict, where: str) -> int:
    value = record.get("bytes")
    _require(isinstance(value, int) and not isinstance(value, bool)
             and value > 0, f"{where}: bytes must be a positive integer")
    assert isinstance(value, int)
    return value


def _digest(record: dict, where: str) -> str:
    value = record.get("sha256")
    _require(isinstance(value, str) and len(value) == 64
             and all(c in "0123456789abcdef" for c in value),
             f"{where}: sha256 must be 64 lowercase hex characters")
    assert isinstance(value, str)
    return value


def load_pins(path: Path | None = None) -> BridgePins:
    """Read and validate the packaged pins document."""

    resolved = packaged_pins_path() if path is None else Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as error:
        raise BridgeAssetError(
            f"bridge pins document {resolved} is unreadable: {error}")
    except ValueError as error:
        raise BridgeAssetError(
            f"bridge pins document {resolved} is not JSON: {error}")
    return parse_pins(payload, origin=str(resolved))


def staging_available(pins: BridgePins | None = None) -> bool:
    """Can THIS install stage a bundle for THIS platform?

    Both halves are capabilities of artifacts rather than opinions: a
    platform key for this OS/arch, and a bundle pinned for it in the
    document this wheel actually carries.  ``woof doctor`` offers
    ``woof fetch-bridges`` exactly when this is true, so the report
    never advertises a command that would refuse.
    """

    platform = host_platform()
    if platform is None:
        return False
    try:
        resolved = load_pins() if pins is None else pins
    except BridgeAssetError:
        return False
    return resolved.bundle_for(platform) is not None


def asset_url_base(pins: BridgePins) -> str:
    """Base URL the bundle is downloaded from (env override wins)."""

    override = os.environ.get(ASSET_URL_BASE_ENV)
    if override and override.strip():
        return override.strip().rstrip("/")
    if not pins.release:
        raise BridgeAssetError(
            "the packaged pins declare no release, so there is no URL to "
            f"download from; set {ASSET_URL_BASE_ENV} or use --from DIR")
    return f"{bridges.REPOSITORY_URL}/releases/download/{pins.release}"


def bundle_url(pins: BridgePins, bundle: BundlePin) -> str:
    return f"{asset_url_base(pins)}/{bundle.filename}"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def verify_pinned_file(path: Path, *, expected_bytes: int,
                       expected_sha256: str, label: str) -> None:
    """Exact size and SHA-256, or a refusal naming both observations."""

    actual = path.stat().st_size
    if actual != expected_bytes:
        raise BridgeAssetError(
            f"{label}: {actual:,} bytes, expected {expected_bytes:,}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise BridgeAssetError(
            f"{label}: SHA-256 {observed}, expected {expected_sha256}")


def _is_source_revision(text: str) -> bool:
    return (len(text) == _SOURCE_REV_LENGTH
            and all(c in "0123456789abcdef" for c in text))


def embedded_source_revisions(payload: bytes) -> tuple[str, ...]:
    """Every distinct well-formed source-revision stamp in ``payload``.

    A stamp is :data:`SOURCE_REV_MARKER` immediately followed by 40
    lowercase hex characters.  Marker occurrences followed by anything
    else -- ``unknown`` from a build outside a git checkout, or
    truncated bytes -- are not revisions and are not returned; the
    verifier tells those cases apart from a missing marker.  Order of
    first appearance is preserved so a refusal can name what it saw.
    """

    found: list[str] = []
    start = 0
    while (index := payload.find(SOURCE_REV_MARKER, start)) != -1:
        begin = index + len(SOURCE_REV_MARKER)
        token = payload[begin:begin + _SOURCE_REV_LENGTH]
        text = token.decode("ascii", errors="replace")
        if _is_source_revision(text) and text not in found:
            found.append(text)
        start = begin
    return tuple(found)


def verify_source_revision(payload: bytes, *, expected: str,
                           label: str, equivalent=None) -> None:
    """``payload`` was built from commit ``expected``, or a refusal.

    Read from the bytes, never by executing the artifact: the cut
    inspects both platforms' bundles from one machine, and a staleness
    check that needs to run the binary cannot look at the other
    platform's.  Refusals name what was found so the remedy (rebuild
    from the release checkout and repack) is unambiguous.
    """

    expected = expected.strip().lower()
    if not _is_source_revision(expected):
        raise BridgeAssetError(
            f"{label}: expected source revision {expected!r} is not a "
            "full 40-hex git commit; pass the exact commit being "
            "released, never a ref name that can move")
    marker = SOURCE_REV_MARKER.decode("ascii").rstrip("=")
    revisions = embedded_source_revisions(payload)
    if not revisions:
        if SOURCE_REV_MARKER in payload:
            raise BridgeAssetError(
                f"{label}: carries a {marker} stamp with no 40-hex "
                "revision after it -- it was built outside a git "
                "checkout, so nothing proves which source produced it; "
                "rebuild from the release checkout")
        raise BridgeAssetError(
            f"{label}: carries no {marker} stamp, so nothing proves it "
            f"was built from the source revision being released "
            f"({expected}); it is a stale build predating the stamped "
            "bridges, or not a woof bridge artifact at all -- rebuild "
            "from the release checkout and repack")
    if len(revisions) > 1:
        raise BridgeAssetError(
            f"{label}: carries {len(revisions)} distinct source-revision "
            f"stamps ({', '.join(revisions)}); one binary cannot be from "
            "two commits, so this build is not trustworthy -- rebuild "
            "from the release checkout")
    if revisions[0] != expected and equivalent is not None:
        # A binary built at an earlier commit whose every build input is
        # byte-identical at the commit being released is that release's
        # binary: the release cut reuses it rather than recompiling the
        # same sources.  ``equivalent`` answers None for that case and a
        # reason otherwise; see :func:`native_input_difference`.
        reason = equivalent(revisions[0])
        if reason is None:
            return
        raise BridgeAssetError(
            f"{label}: was built from source revision {revisions[0]}, "
            f"not the {expected} being released, and it is not "
            f"interchangeable with a build at {expected}: {reason}; "
            "rebuild from the release checkout and repack")
    if revisions[0] != expected:
        raise BridgeAssetError(
            f"{label}: was built from source revision {revisions[0]}, "
            f"not the {expected} being released -- a stale build; "
            "rebuild from the release checkout and repack")


#: Every tracked path a native crate's build reads, keyed by the crate
#: root (``BundledArtifact.crate``).  The crate's own directory first,
#: then whatever it reaches outside it: Cargo ``path`` dependencies,
#: ``include_str!``/``include_bytes!`` of repository files, and the
#: shared vendored trees.  Transitive: a crate that depends on another
#: root carries that root's own outside inputs too.
#:
#: What it is for.  A release cut reuses a binary built at an earlier
#: commit when every path listed for its crate is byte-identical (the
#: same git object) at the commit being released, so a release that
#: changed one Python file does not recompile forty-five unchanged
#: binaries.  A path missing from this table is a binary that could be
#: reused while carrying a stale copy of that file, so
#: ``tests/test_native_build_inputs.py`` re-derives the outside inputs
#: from the Cargo manifests and the Rust sources and refuses any it does
#: not find covered here.
NATIVE_BUILD_INPUTS: dict[str, tuple[str, ...]] = {
    # Its crates.io closure resolves to the rustwx vendor tree (tools/wind-table/.cargo/config.toml).
    "tools/wind-table": ("tools/wind-table", "tools/rustwx/vendor/crates-io"),
    "tools/rw_icechunk": ("tools/rw_icechunk",),  # completed below, after tools/rustwx
    "tools/nextgen_hydrofabric": ("tools/nextgen_hydrofabric",),
    "tools/nextgen_verify": ("tools/nextgen_verify",),
    "tools/grib1_bridge": ("tools/grib1_bridge", "tools/preparation_resources.rs",
                           "tools/rustwx/vendor/crates-io"),
    "tools/rustwx": ("tools/rustwx", "tools/grib1_bridge/vendor/grib-core",
                     "tools/preparation_resources.rs",
                     # static-fields' SFIRE debug controls include the
                     # native oracle fixtures (sfire_debug.rs).
                     "tools/sfire_wrf471_oracle/fixtures/debug",
                     # rw-superob links the region-global solver by path;
                     # rw_wps and the Zarr bridge reach it through this
                     # workspace, so they list it too (the inputs test
                     # walks the closure).
                     "tools/region_global_dealias",
                     # rw-mpas init reads its RUC soil and aerosol oracle columns by
                     # include_str! (golden/hex-rrfs, 2.9.0).
                     "woof/data/ruc/oracle/soil_ingest.csv",
                     "woof/data/thompson/oracle-aero/aero-init-profile-column.csv",
                     # rw-wrfbatch's build.rs embeds the engine version it
                     # reads from the root pyproject.toml (0c1fc01f9), so a
                     # version bump must rebuild every crate that reaches it.
                     "pyproject.toml",
                     # 2.9.4 capabilities campaign include_str! tables:
                     # lane/cap-hubwind's hub-height products, observations,
                     # TKE sources and turbine power curves (rw-wrfbatch,
                     # rustwx-calc), the WPS archive layouts (rustwx-models
                     # archives.rs) and the SFIRE input-field and Albers CRS
                     # authorities (static-fields landfire.rs, raster).
                     "woof/data/hubwind_observations.json",
                     "woof/data/hubwind_products.json",
                     "woof/data/hubwind_tke_sources.json",
                     "woof/data/turbine_power_curves.json",
                     "woof/authorities/rw-wps-archive-layouts.v1.json",
                     "woof/authorities/sfire-albers-crs.v1.json",
                     "woof/authorities/sfire-input-fields.v1.json"),
    # The shared NetCDF reader (netcrust and its HDF5 reader) lives under
    # tools/rustwx/vendor since the array-ceiling work; rw_wps and the Zarr
    # bridge build against it by path.
    "tools/rw_wps": ("tools/rw_wps", "tools/rustwx/vendor/crates-io",
                     "tools/grib1_bridge/vendor/grib-core",
                     "tools/preparation_resources.rs",
                     # projected-intermediate compiles grib1_bridge's WPS
                     # intermediate writer in by #[path] (2.9.1).
                     "tools/grib1_bridge/src/wps_intermediate.rs",
                     "tools/rustwx/vendor/netcrust",
                     "tools/sfire_wrf471_oracle/fixtures/debug",
                     "tools/region_global_dealias",
                     # mapped-engine's grib.rs embeds the two packaged
                     # HRRR mappings (2ed5afdc7's HRRRv1 corner test).
                     "woof/authorities/rw-wps-hrrr-prs-grib2.mapping.json",
                     "woof/authorities/rw-wps-hrrr-native-grib2.mapping.json",
                     "woof/data/ruc/oracle/soil_ingest.csv",
                     "woof/data/thompson/oracle-aero/aero-init-profile-column.csv",
                     "woof/data/hubwind_observations.json",
                     "woof/data/hubwind_products.json",
                     "woof/data/hubwind_tke_sources.json",
                     "woof/data/turbine_power_curves.json",
                     "woof/authorities/rw-wps-archive-layouts.v1.json",
                     "woof/authorities/sfire-albers-crs.v1.json",
                     "woof/authorities/sfire-input-fields.v1.json",
                     "pyproject.toml"),
    "tools/region_global_dealias": ("tools/region_global_dealias",),
    "tools/zarr_bridge": (
        "tools/zarr_bridge", "tools/rustwx/vendor/crates-io",
        "tools/rustwx/crates/netcdf-writer",
        "tools/rw_wps", "tools/grib1_bridge/vendor/grib-core",
        "tools/grib1_bridge/src/wps_intermediate.rs",
        "tools/preparation_resources.rs",
        "tools/rustwx/vendor/netcrust",
        "tools/sfire_wrf471_oracle/fixtures/debug",
        "tools/region_global_dealias",
        "woof/authorities/rw-wps-hrrr-prs-grib2.mapping.json",
        "woof/authorities/rw-wps-hrrr-native-grib2.mapping.json",
        "woof/data/ruc/oracle/soil_ingest.csv",
        "woof/data/thompson/oracle-aero/aero-init-profile-column.csv",
        "woof/data/hubwind_observations.json",
        "woof/data/hubwind_products.json",
        "woof/data/hubwind_tke_sources.json",
        "woof/data/turbine_power_curves.json",
        "woof/authorities/rw-wps-archive-layouts.v1.json",
        "woof/authorities/sfire-albers-crs.v1.json",
        "woof/authorities/sfire-input-fields.v1.json",
        "pyproject.toml"),
}

# The forcing and gauge tools reach the shared reader, and optional forcing
# subset tools also reach the writer. Keep their workspace's full conservative
# dependency closure, including optional features, rather than pricing reuse
# only for a default build. The closure test treats a workspace as one unit.
for _hydro_crate in ("tools/nextgen_forcing", "tools/nextgen_verify"):
    NATIVE_BUILD_INPUTS[_hydro_crate] = (
        _hydro_crate, *NATIVE_BUILD_INPUTS["tools/rustwx"])

# Gauge discovery and matching also compile the worldwide fabric reader.
NATIVE_BUILD_INPUTS["tools/nextgen_verify"] += ("tools/hydro_world",)

# The default-on hydro door's provider tools (2.9.3 improve-1).
NATIVE_BUILD_INPUTS["tools/hydro_world"] = ("tools/hydro_world",)
NATIVE_BUILD_INPUTS["tools/nextgen_output"] = (
    "tools/nextgen_output", *NATIVE_BUILD_INPUTS["tools/rustwx"])
NATIVE_BUILD_INPUTS["tools/hydro_world_params"] = ("tools/hydro_world_params", "tools/hydro_world")
NATIVE_BUILD_INPUTS["tools/nextgen_reservoir_prepare"] = ("tools/nextgen_reservoir_prepare",)
# The initial-state tool reads NetCDF through the shared reader, so it keeps the
# renderer workspace's closure, as the forcing and gauge tools do.
NATIVE_BUILD_INPUTS["tools/hydro_world_init"] = (
    "tools/hydro_world_init", "tools/hydro_world", *NATIVE_BUILD_INPUTS["tools/rustwx"])

# These bridges compile the pinned static projection library, including its
# embedded database. A change there must invalidate every dependent artifact.
for _hydro_crate in ("tools/hydro_world", "tools/hydro_world_params",
                     "tools/hydro_world_init", "tools/nextgen_hydrofabric",
                     "tools/nextgen_verify"):
    NATIVE_BUILD_INPUTS[_hydro_crate] += ("tools/hydro_proj",)
    if "tools/rustwx/vendor/crates-io" not in NATIVE_BUILD_INPUTS[_hydro_crate]:
        NATIVE_BUILD_INPUTS[_hydro_crate] += ("tools/rustwx/vendor/crates-io",)

# The 2.9.4 ocean preparation (lane/cap-slab-ocean, rw_ocean) and bathymetry
# (rw_bathy) tools build against the renderer workspace's shared NetCDF
# reader, static-fields and the NetCDF writer by path, so they keep that
# workspace's closure, as the hydro tools do.
NATIVE_BUILD_INPUTS["tools/ocean_prepare"] = (
    "tools/ocean_prepare", *NATIVE_BUILD_INPUTS["tools/rustwx"])
NATIVE_BUILD_INPUTS["tools/rw-bathy"] = (
    "tools/rw-bathy", *NATIVE_BUILD_INPUTS["tools/rustwx"])
# rw_icechunk builds against the shared NetCDF reader (tools/rustwx/vendor/netcrust) and resolves its registry
# crates from tools/rustwx/vendor/crates-io (its .cargo/config.toml); reaching into tools/rustwx reaches its
# inputs too (tests/test_native_build_inputs.py closure rule, red for rw_icechunk at the EB4 tip 00c11b5330).
NATIVE_BUILD_INPUTS["tools/rw_icechunk"] = (
    "tools/rw_icechunk", *NATIVE_BUILD_INPUTS["tools/rustwx"])


#: Paths inside a crate root that its build never reads: packed into the
#: bundle as data members, hashed by the pins, never compiled.  The map
#: asset tree carries the binary licence notice, which moves whenever the
#: terminal's lock does; listing it as a build input would recompile the
#: whole renderer workspace for a text refresh.  The inputs test refuses
#: an exclusion that any Rust source embeds from.
NATIVE_BUILD_INPUT_EXCLUSIONS: dict[str, tuple[str, ...]] = {
    "tools/rustwx": ("tools/rustwx/assets",),
}


@functools.lru_cache(maxsize=256)
def native_input_difference(repo: Path, crate: str, built: str,
                            released: str) -> str | None:
    """None when a binary of ``crate`` built at ``built`` is the build at ``released``.

    Two conditions, both read from git and nothing else, so the public
    publication workflow proves them from its own full clone:

    * ``built`` is an ancestor of ``released``: the reused binary came
      from this repository's published history, not from a side branch
      a reader cannot fetch; and
    * no path :data:`NATIVE_BUILD_INPUTS` lists for ``crate`` (less its
      :data:`NATIVE_BUILD_INPUT_EXCLUSIONS`) differs between the two
      commits.

    Anything else, including a commit the clone does not have, is a
    reason string and the binary is refused as before.

    Answers are kept per (repository, crate, commit pair): a bundle holds many
    binaries of one crate, and on Windows each git process costs a large part
    of a second, which made pinning two bundles take a minute and a half.
    """

    import subprocess

    inputs = NATIVE_BUILD_INPUTS.get(crate)
    if inputs is None:
        return f"no build inputs are declared for {crate}"
    repo = Path(repo)
    ancestry = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", built,
         released], capture_output=True, text=True)
    if ancestry.returncode != 0:
        return (f"{built} is not an ancestor of {released} in this "
                "checkout's history, so its sources cannot be compared")
    excluded = [":(exclude)" + path
                for path in NATIVE_BUILD_INPUT_EXCLUSIONS.get(crate, ())]
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--no-renames", "--name-only",
         built, released, "--", *inputs, *excluded],
        capture_output=True, text=True)
    if diff.returncode != 0:
        return "git could not compare the two commits: " + diff.stderr.strip()
    changed = diff.stdout.split()
    if changed:
        shown = ", ".join(changed[:8]) + (" ..." if len(changed) > 8 else "")
        return f"its build inputs changed ({len(changed)} files: {shown})"
    return None


def verify_contract_marker(artifact: str, path: Path) -> None:
    """The static contract check ``woof doctor`` applies, before install.

    A bundle is built from the same commit as the wheel that carries its
    pins, so a marker mismatch here means the release process assembled
    the two from different trees.  Catching it at staging is the whole
    difference between a refusal and a preparation that dies hours later
    blaming a file woof wrote correctly.
    """

    if artifact not in bridges.BRIDGE_ABI_MARKERS:
        return
    ok, evidence = bridges.bridge_abi_matches(artifact, path)
    if not ok:
        raise BridgeAssetError(f"{path.name}: {evidence}")


def matches_pin(path: Path, pin: BinaryPin) -> bool:
    """Is the file already exactly these bytes?  No exceptions raised."""

    try:
        if not path.is_file() or path.stat().st_size != pin.bytes:
            return False
        return sha256_file(path) == pin.sha256
    except OSError:
        return False


def classify_destination(dest: Path, bundle: BundlePin
                         ) -> tuple[list[BinaryPin], list[BinaryPin],
                                    list[BinaryPin]]:
    """(already pinned, present but different, absent) at ``dest``."""

    staged: list[BinaryPin] = []
    stale: list[BinaryPin] = []
    absent: list[BinaryPin] = []
    for pin in bundle.binaries:
        path = dest / pin.filename
        if not path.is_file():
            absent.append(pin)
        elif matches_pin(path, pin):
            staged.append(pin)
        else:
            stale.append(pin)
    return staged, stale, absent


def classify_assets(dest: Path, bundle: BundlePin
                    ) -> tuple[list[AssetPin], list[AssetPin],
                               list[AssetPin]]:
    """(already pinned, present but different, absent) for the assets.

    Separate from :func:`classify_destination` because the two answer
    different questions for the caller: the binaries are listed
    individually in a report, and several dozen shapefiles are counted.
    """

    staged: list[AssetPin] = []
    stale: list[AssetPin] = []
    absent: list[AssetPin] = []
    for pin in bundle.assets:
        path = dest / pin.path
        if not path.is_file():
            absent.append(pin)
        elif matches_pin(path, pin):
            staged.append(pin)
        else:
            stale.append(pin)
    return staged, stale, absent


# ---------------------------------------------------------------------------
# What a RESOLUTION is allowed to hand back
# ---------------------------------------------------------------------------

#: Policy for a staged artifact that is not the one this release pinned,
#: applied where a resolution ladder is about to hand the file to a door.
#:
#: ``refresh`` (the default) re-fetches this release's bundle and uses
#: the verified bytes; ``refuse`` never touches the network and names
#: the mismatch instead; ``allow`` runs the staged file anyway and is a
#: reported WORKAROUND, not a configuration.
STALE_POLICY_ENV = "WOOF_BRIDGE_STALE_POLICY"

#: The accepted values of :data:`STALE_POLICY_ENV`, default first.
STALE_POLICIES = ("refresh", "refuse", "allow")

#: Socket timeout for an automatic refresh.  Shorter than
#: :data:`_TIMEOUT_S`, which serves a command the operator typed and is
#: watching: an auto-refresh happens inside somebody else's door, so an
#: unreachable network has to become the offline refusal quickly rather
#: than hold a run open for two minutes per read.
_AUTO_REFRESH_TIMEOUT_S = 30

#: ``{(path, size, mtime_ns): matches}`` for pin comparisons already
#: made in this process.  A door can resolve the same artifact many
#: times in one run and the answer cannot change under it without the
#: size or the mtime moving, so the SHA-256 is computed once.
_PIN_MEMO: dict[tuple[str, int, int], bool] = {}


def _under(path: Path, directory: Path) -> bool:
    """Is ``path`` inside ``directory``?  Never raises."""

    try:
        return path.resolve().is_relative_to(directory.resolve())
    except (OSError, ValueError):
        return False


@dataclass(frozen=True)
class StagedPinStatus:
    """One staged artifact, judged against the pins this wheel carries.

    Produced only for a file under :func:`woof.bridges
    .default_bridge_dir` that this release publishes a pin for, which is
    the one rung whose contents no version of woof controls: a wheel
    upgrade replaces the Python half and leaves that directory exactly
    as the previous release wrote it.
    """

    path: Path
    pin: BinaryPin
    release: str
    matches: bool
    observed_bytes: int
    observed_revisions: tuple[str, ...]

    @property
    def observed_revision(self) -> str | None:
        """The single stamp the file carries, or None if it is not one."""

        return (self.observed_revisions[0]
                if len(self.observed_revisions) == 1 else None)

    def provenance(self) -> str:
        """What the bytes on disk say about where they came from."""

        if self.observed_revision is not None:
            return f"built from source revision {self.observed_revision}"
        if not self.observed_revisions:
            marker = SOURCE_REV_MARKER.decode("ascii").rstrip("=")
            return f"carrying no {marker} stamp"
        return (f"carrying {len(self.observed_revisions)} distinct "
                "source-revision stamps ("
                + ", ".join(self.observed_revisions) + ")")

    def describe(self) -> str:
        """The mismatch in one sentence, naming both sides by number."""

        return (f"{self.path} is {self.observed_bytes:,} B "
                f"{self.provenance()}; {self.release} pins "
                f"{self.pin.bytes:,} B (SHA-256 {self.pin.sha256[:12]}...) "
                f"for {self.pin.artifact}")


def staged_pin_status(path: Path | str, *,
                      pins: BridgePins | None = None
                      ) -> StagedPinStatus | None:
    """Judge a resolved artifact against this release's pins, or None.

    ``None`` -- the question does not arise -- for every case where the
    pins have nothing to say, and each of those is a deliberate
    exemption rather than an oversight:

    * the file is not under :func:`woof.bridges.default_bridge_dir`
      (this install's own staging directory) or the shared
      :func:`woof.bridges.legacy_bridge_dir` root it sits in.
      An environment override is an explicit declaration, a checkout's
      ``target/release`` is a build the developer just made, and
      ``libexec`` beside the package or inside it arrived with this
      version.  None of the three is the skew this judgement exists
      for, and a cargo build never matches a release pin by
      construction, so judging them would refuse the developer path on
      every run.
    * this install carries no pins for this platform -- a source
      checkout (whose packaged document declares no release until the
      cut stamps it), or a platform no bundle is published for.
    * this release publishes no pin for a file by that name.

    Otherwise the answer is the one ``woof fetch-bridges`` and ``woof
    doctor`` already ask of the same file: are these the exact bytes,
    size and SHA-256, that this release published?  A mismatch reads the
    embedded ``GPUWM_BRIDGE_SOURCE_REV`` stamps as well, so a refusal
    can name which release the file on disk came from rather than only
    that it is wrong.
    """

    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError):
        return None
    if not (_under(resolved, bridges.default_bridge_dir())
            or _under(resolved, bridges.legacy_bridge_dir())):
        return None
    if pins is None:
        try:
            pins = load_pins()
        except BridgeAssetError:
            # An unreadable pins document is doctor's line to report; it
            # is not a reason to block every door on this box.
            return None
    bundle = pins.bundle_for(host_platform())
    if bundle is None or not pins.release:
        return None
    pin = next((entry for entry in bundle.binaries
                if entry.filename == resolved.name), None)
    if pin is None:
        return None
    try:
        stat = resolved.stat()
    except OSError:
        return None
    key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    matches = _PIN_MEMO.get(key)
    if matches is None:
        matches = matches_pin(resolved, pin)
        _PIN_MEMO[key] = matches
    revisions: tuple[str, ...] = ()
    if not matches:
        try:
            revisions = embedded_source_revisions(resolved.read_bytes())
        except OSError:
            revisions = ()
    return StagedPinStatus(
        path=resolved, pin=pin, release=pins.release, matches=matches,
        observed_bytes=stat.st_size, observed_revisions=revisions)


def forget_pin_memo() -> None:
    """Drop the in-process pin cache (a refresh has moved the bytes)."""

    _PIN_MEMO.clear()


def stale_policy() -> str:
    """The configured policy for a stale staged artifact, validated.

    An unrecognised value is not silently treated as the default: a
    misspelled policy would then look like it was honoured.
    """

    raw = (os.environ.get(STALE_POLICY_ENV) or "").strip().lower()
    if not raw:
        return STALE_POLICIES[0]
    if raw not in STALE_POLICIES:
        warn(f"{STALE_POLICY_ENV}={raw!r} is not one of "
             f"{', '.join(STALE_POLICIES)}; using {STALE_POLICIES[0]!r}")
        return STALE_POLICIES[0]
    return raw


def refresh_staged_bundle(*, pins: BridgePins | None = None,
                          dest: Path | None = None, progress=None,
                          urlopen_fn=None) -> list[Path]:
    """Stage this release's bundle into this release's own directory.

    ``dest`` defaults to :func:`woof.bridges.default_bridge_dir`, which
    for a pinned install is versioned by release and bundle digest, so
    the refresh can only ever write files no other install resolves.

    The same verified path ``woof fetch-bridges`` walks -- download,
    size and SHA-256 against the packaged pins, contract marker, atomic
    replace -- reached from a resolution rather than from a command, so
    that a bare run stops using an artifact this release did not
    publish.  Every refusal on that path is a :class:`BridgeAssetError`
    and stays one here; the caller turns it into the offline arm.
    """

    resolved_pins = load_pins() if pins is None else pins
    bundle = resolved_pins.bundle_for(host_platform())
    if bundle is None:
        raise BridgeAssetError(
            "this install carries no bundle pin for "
            f"{host_platform_description()}, so there is nothing to "
            "refresh from")
    destination = bridges.default_bridge_dir() if dest is None else Path(dest)
    opener = urlopen_fn
    if opener is None:
        def opener(request):
            return urlopen(request, timeout=_AUTO_REFRESH_TIMEOUT_S)
    installed = fetch_bundle(resolved_pins, bundle, destination,
                             progress=progress or (lambda message: None),
                             urlopen_fn=opener)
    forget_pin_memo()
    return installed


# ---------------------------------------------------------------------------
# Download (resumable, restartable)
# ---------------------------------------------------------------------------

def _default_urlopen(request: Request):
    return urlopen(request, timeout=_TIMEOUT_S)


def _partial_note(dest: Path) -> str:
    """What a failed transfer leaves for the next run.

    :func:`download_bundle` resumes any partial no longer than the pin,
    so a partial is always worth naming; the old text promised a resume
    even when the failure left nothing on disk.
    """

    try:
        size = dest.stat().st_size
    except OSError:
        size = 0
    if size <= 0:
        return "nothing was downloaded, so a re-run starts from the beginning"
    return (f"the partial file ({size:,} B) is kept and a re-run resumes "
            "from it")


def download_bundle(url: str, dest: Path, *, expected_bytes: int,
                    progress, urlopen_fn=_default_urlopen) -> None:
    """Download ``url`` to ``dest``, resuming or restarting as needed.

    An interrupted run leaves a partial file; the next run asks the
    server to continue from its byte count.  A server that answers
    anything but ``206 Partial Content`` -- including one that has no
    idea what a range is -- restarts the transfer from zero instead of
    appending to bytes it did not extend, which is the failure mode a
    naive resume turns into a corrupt archive.  A partial longer than
    the pin is discarded outright: it cannot be a prefix of the file we
    are pinned to.
    """

    offset = dest.stat().st_size if dest.exists() else 0
    if offset > expected_bytes:
        dest.unlink()
        progress(f"woof fetch-bridges: the partial download was larger "
                 f"than the pinned {expected_bytes:,} B; discarded it and "
                 "restarted")
        offset = 0
    if offset == expected_bytes:
        progress("woof fetch-bridges: the partial download is already "
                 "complete; verifying it instead of re-downloading")
        return
    headers = {"User-Agent": _USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        progress(f"woof fetch-bridges: resuming at {offset:,} B of "
                 f"{expected_bytes:,} B")
    request = Request(url, headers=headers)
    try:
        with urlopen_fn(request) as response:
            status = getattr(response, "status", None)
            mode = "ab" if (offset and status == 206) else "wb"
            if offset and status != 206:
                progress("woof fetch-bridges: the server ignored the "
                         "resume range; restarting the download")
            fetch_guard.receive(response, dest, mode,
                                block_bytes=_BLOCK_BYTES)
    except fetch_guard.LocalWriteFailed as failure:
        # Before the network handlers: a file this computer would not
        # write used to read "download failed ...; a re-run resumes".
        raise BridgeAssetError(fetch_guard.local_write_refusal(
            "woof fetch-bridges", failure.path, failure.error,
            _partial_note(dest))) from failure.error
    except HTTPError as error:
        raise BridgeAssetError(
            f"HTTP {error.code} from {url}; {_partial_note(dest)}")
    except (URLError, OSError, TimeoutError) as error:
        raise BridgeAssetError(
            f"download failed from {url}: {error}; {_partial_note(dest)}")
    size = dest.stat().st_size
    if size != expected_bytes:
        dest.unlink(missing_ok=True)
        raise BridgeAssetError(
            f"downloaded {size:,} B from {url}, expected "
            f"{expected_bytes:,} B; the incomplete file was removed -- "
            "re-run to fetch it again")


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def _install(temp: Path, final: Path, pin: BinaryPin) -> None:
    """Verify ``temp`` three ways, then atomically install it."""

    try:
        verify_pinned_file(temp, expected_bytes=pin.bytes,
                           expected_sha256=pin.sha256, label=pin.filename)
        verify_contract_marker(pin.artifact, temp)
    except BridgeAssetError:
        temp.unlink(missing_ok=True)
        raise
    if os.name != "nt":
        # A zip carries no reliable execute bit, and an executable that
        # cannot be executed is not staged, it is merely present.
        os.chmod(temp, 0o755)
    os.replace(temp, final)


def _install_asset(temp: Path, final: Path, pin: AssetPin) -> None:
    """Verify ``temp`` by size and hash, then atomically install it.

    Two checks, not three, and no execute bit: a shapefile has no ABI
    marker to match and nothing should ever run it.  Everything else --
    verify before install, delete on failure, atomic replace -- is what
    :func:`_install` does for a binary.
    """

    try:
        verify_pinned_file(temp, expected_bytes=pin.bytes,
                           expected_sha256=pin.sha256, label=pin.path)
    except BridgeAssetError:
        temp.unlink(missing_ok=True)
        raise
    if os.name != "nt":
        os.chmod(temp, 0o644)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp, final)


def _bundle_members(archive: Path) -> list[str]:
    try:
        with zipfile.ZipFile(archive) as zf:
            return zf.namelist()
    except (zipfile.BadZipFile, OSError) as error:
        raise BridgeAssetError(f"{archive.name} is not a readable bundle "
                               f"archive: {error}")


def stage_from_bundle(archive: Path, bundle: BundlePin, dest: Path,
                      *, progress) -> list[Path]:
    """Extract, verify and install every pinned artifact from ``archive``.

    Members are read by their exact pinned filename, never by whatever
    path the archive proposes, so nothing can be written outside
    ``dest``.  A pinned name the archive does not carry is a refusal
    that names what the archive holds instead -- which is how the other
    platform's bundle identifies itself.
    """

    present = set(_bundle_members(archive))
    missing = [pin.filename for pin in bundle.binaries
               if pin.filename not in present]
    if missing:
        listed = ", ".join(sorted(present)[:8]) or "nothing"
        raise BridgeAssetError(
            f"{archive.name} does not carry {', '.join(missing)} -- it "
            f"holds {listed}.  This is not the {bundle.platform} bundle")
    absent_assets = [pin.path for pin in bundle.assets
                     if pin.path not in present]
    if absent_assets:
        raise BridgeAssetError(
            f"{archive.name} is missing {len(absent_assets)} pinned map "
            f"asset(s), starting with {absent_assets[0]}.  A bundle whose "
            "assets do not match its pins would render geography-less "
            "plots; refusing it")
    dest.mkdir(parents=True, exist_ok=True)
    # Per PROCESS, because staging is no longer only something an
    # operator types: a resolution that finds a stale artifact refreshes
    # the estate itself, so two runs on one box can be extracting into
    # this directory at the same moment.  A shared scratch name means
    # the ``finally`` below deletes the other run's half-written files.
    work = dest / f"{ARCHIVE_SUBDIR}-stage-{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    try:
        # Phase 1: extract and verify EVERYTHING before installing
        # anything.  Installing file by file used to leave a directory
        # holding the first artifacts of this release and the old (or
        # no) copies of the rest whenever one member failed its pin, so
        # the routes resolving there ran a mix of two releases.  A
        # bundle that does not verify whole installs nothing.
        verified: list[tuple[Path, Path, str]] = []
        with zipfile.ZipFile(archive) as zf:
            for pin in bundle.binaries:
                temp = work / pin.filename
                with zf.open(pin.filename) as source,                         temp.open("wb") as sink:
                    shutil.copyfileobj(source, sink, _BLOCK_BYTES)
                _verify_partial_refusal(
                    archive, bundle, pin.filename,
                    lambda temp=temp, pin=pin: _verify_binary(temp, pin))
                verified.append((temp, dest / pin.filename,
                                 f"({pin.bytes:,} B, SHA-256 "
                                 f"{pin.sha256[:12]}...)"))
            for index, pin in enumerate(bundle.assets):
                # The pinned path is validated relative and contained by
                # parse_pins, so the join cannot leave dest.
                temp = work / f"asset-{index}.part"
                with zf.open(pin.path) as source, temp.open("wb") as sink:
                    shutil.copyfileobj(source, sink, _BLOCK_BYTES)
                _verify_partial_refusal(
                    archive, bundle, pin.path,
                    lambda temp=temp, pin=pin: _verify_asset(temp, pin))
                verified.append((temp, dest / pin.path, ""))
        # Phase 2: every byte verified; place each with an atomic
        # replace.  Two processes staging the same release land the same
        # verified bytes, so the order they finish in does not matter.
        binary_count = len(bundle.binaries)
        asset_bytes = 0
        for index, (temp, final, note) in enumerate(verified):
            replaced = final.is_file()
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp, final)
            installed.append(final)
            if index < binary_count:
                progress(
                    f"woof fetch-bridges: {'replaced' if replaced else 'staged'}"
                    f" {final} {note}")
            else:
                asset_bytes += bundle.assets[index - binary_count].bytes
        staged_assets = len(verified) - binary_count
        if staged_assets:
            progress(
                f"woof fetch-bridges: staged {staged_assets} map asset "
                f"file(s) ({asset_bytes / (1024 * 1024):.1f} MiB) under "
                f"{dest / ASSET_ROOT}; the renderer finds them there "
                "without any environment variable")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return installed


def _verify_binary(temp: Path, pin: BinaryPin) -> None:
    """Size, SHA-256 and contract marker; the execute bit on success."""

    verify_pinned_file(temp, expected_bytes=pin.bytes,
                       expected_sha256=pin.sha256, label=pin.filename)
    verify_contract_marker(pin.artifact, temp)
    if os.name != "nt":
        os.chmod(temp, 0o755)


def _verify_asset(temp: Path, pin: AssetPin) -> None:
    """Size and SHA-256; read-only data mode on success."""

    verify_pinned_file(temp, expected_bytes=pin.bytes,
                       expected_sha256=pin.sha256, label=pin.path)
    if os.name != "nt":
        os.chmod(temp, 0o644)


def _verify_partial_refusal(archive: Path, bundle: BundlePin, member: str,
                            check) -> None:
    """Run ``check``; a failure refuses the WHOLE bundle, by name."""

    try:
        check()
    except BridgeAssetError as error:
        raise BridgeAssetError(
            f"{archive.name}: {error}.  Refused the whole bundle and "
            "installed none of it: staging the members that did verify "
            "would leave a bridge directory holding part of "
            f"{bundle.platform}'s set beside missing or older copies of "
            "the rest, so some routes would run this release's binaries "
            f"and others another release's, or none ({member} failed)"
        ) from None


def stage_from_loose_files(source_dir: Path, bundle: BundlePin, dest: Path,
                           *, progress) -> list[Path]:
    """Install the pinned artifacts sitting loose in ``source_dir``.

    What an air-gapped operator has after building on a machine that
    does have a toolchain: the declared artifact files, no archive.  Same three
    checks, same atomic install.
    """

    absent = [pin.filename for pin in bundle.binaries
              if not (source_dir / pin.filename).is_file()]
    present = [pin for pin in bundle.binaries
               if (source_dir / pin.filename).is_file()]
    if absent and not present:
        raise BridgeAssetError(
            f"{source_dir} carries neither {bundle.filename} nor the loose "
            f"artifacts; missing {', '.join(absent)}")
    if absent:
        # The declared artifacts are independent; an air-gapped operator
        # with the decoders but not the renderer gets the decoders,
        # verified, and doctor names what is still missing.
        warn(f"{source_dir} is missing {len(absent)} of "
             f"{len(bundle.binaries)} pinned artifact(s) "
             f"({', '.join(absent)}); staging the {len(present)} that "
             "are present -- woof doctor reports the rest")
    dest.mkdir(parents=True, exist_ok=True)
    # Per PROCESS, because staging is no longer only something an
    # operator types: a resolution that finds a stale artifact refreshes
    # the estate itself, so two runs on one box can be extracting into
    # this directory at the same moment.  A shared scratch name means
    # the ``finally`` below deletes the other run's half-written files.
    # The install itself is already safe -- verify, then ``os.replace``.
    work = dest / f"{ARCHIVE_SUBDIR}-stage-{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    try:
        for pin in present:
            temp = work / pin.filename
            shutil.copyfile(source_dir / pin.filename, temp)
            final = dest / pin.filename
            replaced = final.is_file()
            _install(temp, final, pin)
            installed.append(final)
            progress(
                f"woof fetch-bridges: {'replaced' if replaced else 'staged'}"
                f" {final} ({pin.bytes:,} B, SHA-256 {pin.sha256[:12]}...)")
        # The map assets keep their layout in a loose directory too, so
        # an operator who unzipped a bundle by hand and an operator who
        # copied a checkout's assets/ both land here.
        staged_assets = 0
        missing_assets: list[str] = []
        for pin in bundle.assets:
            origin = source_dir / pin.path
            if not origin.is_file():
                missing_assets.append(pin.path)
                continue
            temp = work / "asset.part"
            shutil.copyfile(origin, temp)
            final = dest / pin.path
            _install_asset(temp, final, pin)
            installed.append(final)
            staged_assets += 1
        if staged_assets:
            progress(f"woof fetch-bridges: staged {staged_assets} map "
                     f"asset file(s) under {dest / ASSET_ROOT}")
        if missing_assets:
            warn(f"{source_dir} carries no {ASSET_ROOT}/ tree for "
                 f"{len(missing_assets)} pinned map asset file(s); the "
                 "renderer will draw plots without coastlines or borders "
                 "unless it finds them elsewhere")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return installed


def stage_from_dir(source_dir: Path, bundle: BundlePin, dest: Path,
                   *, progress) -> list[Path]:
    """``--from DIR``: the bundle archive if it is there, else the files."""

    archive = source_dir / bundle.filename
    if archive.is_file():
        verify_pinned_file(archive, expected_bytes=bundle.bytes,
                           expected_sha256=bundle.sha256,
                           label=bundle.filename)
        progress(f"woof fetch-bridges: {archive} matches the packaged "
                 "bundle pin; staging from it")
        return stage_from_bundle(archive, bundle, dest, progress=progress)
    return stage_from_loose_files(source_dir, bundle, dest, progress=progress)


#: The owner file of a bridge folder while one setup stages into it.
BRIDGE_OWNER_NAME = ".fetch-bridges.owner"


class BridgeFolderUnwritable(BridgeAssetError):
    """The bridge folder cannot be written, so no setup can own it.

    Raised in place of the bare OSError from creating the folder or its
    owner file, which used to reach the screen as a traceback.  Nothing
    can be staged there, but an estate that is already complete is still
    usable, and ``woof fetch-bridges`` says so.
    """

    def __init__(self, dest: Path, error: OSError):
        dest = Path(dest)
        if dest.exists() and not dest.is_dir():
            reason = "it is a file, not a folder"
        else:
            words = (error.strerror or "").strip().rstrip(".")
            reason = words.lower() if words else "the system refused it"
        super().__init__(f"{dest} cannot be written ({reason})")
        self.dest = dest
        self.reason = reason


#: How long a second setup waits for the first to finish staging before
#: it refuses.  A whole bundle over a slow line fits inside it.
_BRIDGE_OWNER_WAIT_S = 3600.0


@contextmanager
def bridge_owner(dest: Path, *, progress=print, waited: list | None = None):
    """Hold ``dest`` for one setup across classify, download and cleanup.

    Two setups (two terminals, or a run's automatic refresh beside a
    manual ``woof fetch-bridges``) used to share the one partial archive
    under ``dest``: the second resumed it, installed it and deleted it
    while the first was still writing, and the first then failed on a
    file that was gone.  The second now waits for the first.  Re-entrant
    within one thread, so the command can hold it around a
    :func:`fetch_bundle` that takes it too.  ``waited`` gets one entry
    when this call had to wait for another owner.
    """

    from woof import ownership

    dest = Path(dest)

    def told(holder):
        if waited is not None:
            waited.append(holder)
        progress(f"woof fetch-bridges: waiting for "
                 f"{ownership.describe_holder(holder)} to finish staging "
                 f"bridges at {dest}")

    # Only taking the claim is translated here.  An error raised by the
    # caller's block passes through unchanged, so a failure while staging
    # is never reported as a folder that cannot be written.
    with ExitStack() as stack:
        try:
            dest.mkdir(parents=True, exist_ok=True)
            held = stack.enter_context(ownership.owned(
                dest / BRIDGE_OWNER_NAME, purpose="bridge setup",
                wait=_BRIDGE_OWNER_WAIT_S, poll=0.5, on_wait=told))
        except ownership.OwnershipError as error:
            raise BridgeAssetError(
                f"{dest} is still being staged by "
                f"{ownership.describe_holder(error.holder)} after "
                f"{_BRIDGE_OWNER_WAIT_S / 60:.0f} minutes; run this again "
                "once it finishes." + ownership.recovery_words(error)
            ) from None
        except OSError as error:
            raise BridgeFolderUnwritable(dest, error) from None
        yield held


def _estate_complete(dest: Path, bundle: BundlePin) -> bool:
    _staged, stale, absent = classify_destination(dest, bundle)
    _held, stale_assets, absent_assets = classify_assets(dest, bundle)
    return not (stale or absent or stale_assets or absent_assets)


def fetch_bundle(pins: BridgePins, bundle: BundlePin, dest: Path, *,
                 keep_bundle: bool = False, progress=print,
                 urlopen_fn=_default_urlopen) -> list[Path]:
    """Download, verify and stage the platform bundle, as ``dest``'s owner.

    When another setup held ``dest`` and this call waited for it, the
    estate is checked again first: the other setup has usually staged
    exactly this bundle, and downloading it a second time would only
    replace verified bytes with the same bytes.
    """

    waited: list = []
    with bridge_owner(dest, progress=progress, waited=waited):
        if waited and _estate_complete(dest, bundle):
            progress(f"woof fetch-bridges: the setup that held {dest} "
                     "staged this bundle; nothing left to fetch")
            return []
        return _fetch_bundle_owned(pins, bundle, dest,
                                   keep_bundle=keep_bundle,
                                   progress=progress, urlopen_fn=urlopen_fn)


def _fetch_bundle_owned(pins: BridgePins, bundle: BundlePin, dest: Path, *,
                        keep_bundle: bool, progress, urlopen_fn) -> list[Path]:
    archive_dir = dest / ARCHIVE_SUBDIR
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / bundle.filename
    url = bundle_url(pins, bundle)
    if archive.is_file() and archive.stat().st_size == bundle.bytes \
            and sha256_file(archive) == bundle.sha256:
        progress(f"woof fetch-bridges: {bundle.filename} is already "
                 "downloaded and pin-verified; reusing it")
    else:
        progress(f"woof fetch-bridges: downloading {bundle.filename} "
                 f"({bundle.bytes / (1024 * 1024):.1f} MiB) from {url}")
        download_bundle(url, archive, expected_bytes=bundle.bytes,
                        progress=progress, urlopen_fn=urlopen_fn)
        try:
            verify_pinned_file(archive, expected_bytes=bundle.bytes,
                               expected_sha256=bundle.sha256,
                               label=bundle.filename)
        except BridgeAssetError:
            archive.unlink(missing_ok=True)
            raise
    installed = stage_from_bundle(archive, bundle, dest, progress=progress)
    if keep_bundle:
        progress(f"woof fetch-bridges: kept the verified bundle {archive}")
    else:
        archive.unlink(missing_ok=True)
        try:
            archive_dir.rmdir()
        except OSError:
            pass
    return installed


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

#: What stays broken while the artifacts are absent.  Named once, in
#: routes rather than in filenames, because "nine artifacts are missing"
#: is not a consequence a reader can act on and "prep refuses" is.
_MISSING_ARTIFACT_BREAKAGE = (
    "woof fetch-bridges: until the artifacts exist, the routes that run "
    "them stay unresolved -- GRIB decode, NetCDF read/write and the "
    "renderer -- so `woof prep` refuses to decode source bytes and "
    "`woof render` has no renderer to call.")


def _source_build_lines() -> list[str]:
    """The build route as commands, never as a pointer at a command.

    Two installs, two true answers, and the difference is on disk: a
    checkout has the crates, so the one-liner runs as printed; a wheel
    has none, so the answer is the whole bootstrap down to the clone.
    Taken straight from :mod:`woof.bridges` rather than through its
    ``_bundle_first`` wrapper, so no branch of this refusal can grow a
    ``woof fetch-bridges`` line at its head -- that is the loop this
    text exists to be out of.
    """

    if bridges.sources_present():
        return [f"  {bridges.cargo_build_one_liner()}",
                "  # builds every bridge this install's routes look for,"
                " offline."]
    return list(bridges.build_from_clone_hint())


def _unsupported_platform_note(platform: str | None) -> str:
    """Why no bundle can be staged here, and the commands that fix it.

    Circularity is the defect this text exists to avoid.  Both branches
    used to end "`woof doctor` prints the exact steps for this
    install", and doctor's own next command for a missing bridge is
    ``woof fetch-bridges`` -- so is ``woof setup``, which runs it --
    so a reader who did as they were told arrived back at this same
    refusal having learned nothing and cost themselves two commands.  A
    refusal that hands over another command of ours has not named a
    remedy; the steps are printed here instead.

    The no-pins branch also has a state to name that nothing on the
    reader's disk shows.  The pins are computed from the published
    bytes by ``tools/build_bridge_bundle.py`` and written into
    :data:`PINS_RESOURCE` before the wheel is built, so a wheel
    installed from PyPI is the only artifact that carries them: a
    source checkout, and a wheel built out of one, both carry the empty
    document this branch is reading, and always will.  Saying so is the
    difference between "my install is broken" and "this install was
    never going to have them", and only the second one tells the reader
    which of the two routes below is theirs.

    A checkout is never told to ``pip install`` over itself: its
    sources are present, so the build is both shorter and true, and the
    wheel appears there as the explanation rather than as the step.
    """

    if platform is None:
        return "\n".join([
            f"woof fetch-bridges: no bundle is published for "
            f"{host_platform_description()} (bundles exist for "
            f"{', '.join(SUPPORTED_PLATFORMS)}).",
            "woof fetch-bridges: bundles are built per OS and "
            "architecture, so this is not something a newer install "
            "carries: no released wheel has bytes for this box, and this "
            "command has nothing it could stage here.",
            _MISSING_ARTIFACT_BREAKAGE,
            "woof fetch-bridges: build the artifacts from a clone "
            "instead -- the route that works on every platform:",
            *_source_build_lines(),
        ])

    lines = [
        f"woof fetch-bridges: this install carries no bundle pins for "
        f"{platform}.",
        "woof fetch-bridges: the pins are computed from the published "
        f"bytes and written into {packaged_pins_path()} when the wheel is "
        "built, so a wheel installed from PyPI is the install that carries "
        "them; this one declares no platforms at all, which is what a "
        "source checkout -- or a wheel built out of one -- looks like.  "
        "This command stages only bytes it holds a SHA-256 for, so with no "
        "pins there is nothing here for it to fetch.",
        _MISSING_ARTIFACT_BREAKAGE,
    ]
    if bridges.sources_present():
        lines.append(
            "woof fetch-bridges: this install has the Rust sources, so "
            "build the artifacts here:")
        lines += _source_build_lines()
        return "\n".join(lines)
    lines.append(
        "woof fetch-bridges: install the published wheel, which carries "
        f"the pins for {platform} and the bundle they verify:")
    lines += [
        "  pip install --upgrade recast-woof",
        "  woof fetch-bridges",
        "  # the second line stages the bundle; it refuses again only if",
        "  # the first one did not replace this install.",
        "woof fetch-bridges: or build the artifacts from a clone "
        "instead, which needs no published bundle at all:",
    ]
    lines += _source_build_lines()
    return "\n".join(lines)


def _override_warnings(bundle: BundlePin) -> list[str]:
    """Environment overrides that would shadow what we just staged."""

    by_name = {artifact.name: artifact for artifact in BUNDLED_ARTIFACTS}
    notes: list[str] = []
    for pin in bundle.binaries:
        artifact = by_name.get(pin.artifact)
        if artifact is None:
            continue
        override = os.environ.get(artifact.env_var)
        if override:
            notes.append(
                f"woof fetch-bridges: NOTE: {artifact.env_var} is set to "
                f"{override}, which wins over the staged copy; unset it to "
                "use what was just staged")
    return notes


def _print_listing(pins: BridgePins, bundle: BundlePin, dest: Path) -> None:
    print(f"woof fetch-bridges: platform {bundle.platform} "
          f"({host_platform_description()})")
    print(f"woof fetch-bridges: release {pins.release}")
    try:
        print(f"woof fetch-bridges: bundle {bundle_url(pins, bundle)} "
              f"({bundle.bytes / (1024 * 1024):.1f} MiB)")
    except BridgeAssetError as error:
        print(f"woof fetch-bridges: bundle URL unavailable: {error}")
    print(f"woof fetch-bridges: destination {dest}")
    staged, stale, absent = classify_destination(dest, bundle)
    states: dict[str, str] = {}
    for state, group in (("staged", staged), ("differs", stale),
                         ("needed", absent)):
        for pin in group:
            states[pin.filename] = state
    for pin in bundle.binaries:
        print(f"  {states[pin.filename]:<8} {pin.filename} "
              f"({pin.bytes:,} B)")
    print(f"woof fetch-bridges: {len(staged)} of "
          f"{len(bundle.binaries)} artifacts already staged and pin-valid")
    if bundle.assets:
        held, differs, needed = classify_assets(dest, bundle)
        total = sum(pin.bytes for pin in bundle.assets)
        print(f"woof fetch-bridges: {len(held)} of {len(bundle.assets)} map "
              f"asset files staged and pin-valid under {dest / ASSET_ROOT} "
              f"({total / (1024 * 1024):.1f} MiB packed)"
              + (f"; {len(differs)} differ, {len(needed)} needed"
                 if (differs or needed) else ""))


def fetch_bridges_main(args) -> int:
    with bridges.inspection_only():
        return _fetch_bridges_main(args)


def _fetch_bridges_main(args) -> int:
    # Inside :func:`woof.bridges.inspection_only` because this command
    # IS the remedy the stale-artifact refusal names.  A door that
    # resolves a stale binary refuses and points here; if getting here
    # could itself hit that refusal, the remedy would be unreachable
    # from the exact state it exists to repair.
    dest = (Path(args.dest) if getattr(args, "dest", None)
            else bridges.default_bridge_dir())
    platform = host_platform()
    try:
        pins = load_pins()
    except BridgeAssetError as error:
        print(f"woof fetch-bridges: REFUSED: {error}")
        return 2
    bundle = pins.bundle_for(platform)
    if bundle is None:
        print(_unsupported_platform_note(platform))
        return 2

    if getattr(args, "list", False):
        _print_listing(pins, bundle, dest)
        return 0

    # Classify, stage and verify as the folder's one owner: a second
    # setup that classified the estate while this one was mid-download
    # would act on a picture that is about to change.
    try:
        with bridge_owner(dest, progress=print):
            if getattr(args, "partial", False):
                if getattr(args, "from_dir", None) or getattr(args, "keep_bundle", False):
                    raise BridgeAssetError("--partial cannot be combined with --from or --keep-bundle; partial fetch installs pinned members through HTTP ranges")
                from .bridge_partial import fetch_partial
                fetch_partial(pins, bundle, dest, getattr(args, "artifact", None))
                return 0
            if getattr(args, "artifact", None):
                raise BridgeAssetError("--artifact requires --partial")
            code = _stage_estate(args, pins, bundle, dest)
            if code == 0 and getattr(args, "dest", None) is None:
                code = _stage_hex_helpers(args)
            return code
    except BridgeFolderUnwritable as error:
        return _unwritable_estate(bundle, dest, error)
    except BridgeAssetError as error:
        print(f"woof fetch-bridges: REFUSED: {error}")
        return 2


def _stage_hex_helpers(args) -> int:
    """Stage the hex model's helpers too, when this wheel leaves them to the release.

    A PyPI platform wheel carries their pins, not the helpers
    (:mod:`woof.hex.helpers`; WOOF 1.1.0 r16, PyPI's per-file limit), so the
    command that stages this release's natives ahead of time stages them as
    well: from ``--from DIR`` when it holds the helper archive, else from the
    release.  An ops wheel that carries them, and an install with no pins,
    have nothing to do here.
    """

    try:
        from woof.hex import helpers
    except Exception:                                # noqa: BLE001
        return 0
    pins = helpers.load_pins()
    if pins is None or helpers.packaged():
        return 0
    try:
        helpers.stage(pins, from_dir=getattr(args, "from_dir", None), progress=print)
    except Exception as error:                       # noqa: BLE001
        print(f"woof fetch-bridges: REFUSED: the hex helpers ({pins['bundle']['filename']}): {error}")
        return 2
    return 0


def _print_verified(bundle: BundlePin, dest: Path, staged: list,
                    held_assets: list) -> None:
    print(f"woof fetch-bridges: all {len(staged)} artifacts and "
          f"{len(held_assets)} map asset file(s) at {dest} verified "
          "(exact size + SHA-256); nothing to fetch")
    for note in _override_warnings(bundle):
        print(note)


def _unwritable_estate(bundle: BundlePin, dest: Path,
                       error: BridgeFolderUnwritable) -> int:
    """A folder this user cannot write: check it, and change nothing.

    Nothing can be staged there, so there is nothing to own either, and
    an estate that is already complete is as usable as it was.  Only an
    incomplete one is refused, in words that say what to do instead.
    """

    staged, stale, absent = classify_destination(dest, bundle)
    held_assets, stale_assets, absent_assets = classify_assets(dest, bundle)
    if not stale and not absent and not stale_assets and not absent_assets:
        _print_verified(bundle, dest, staged, held_assets)
        return 0
    needed = len(stale) + len(absent) + len(stale_assets) + len(absent_assets)
    total = len(bundle.binaries) + len(bundle.assets)
    print(f"woof fetch-bridges: REFUSED: {error}, and {needed} of the "
          f"{total} bridge file(s) this release pins are missing or out of "
          "date there.  Make that folder writable and run this again, or "
          "pass --dest with a folder you can write.")
    return 2


def _stage_estate(args, pins: BridgePins, bundle: BundlePin,
                  dest: Path) -> int:
    staged, stale, absent = classify_destination(dest, bundle)
    held_assets, stale_assets, absent_assets = classify_assets(dest, bundle)
    if not stale and not absent and not stale_assets and not absent_assets:
        _print_verified(bundle, dest, staged, held_assets)
        return 0
    if (stale_assets or absent_assets) and not stale and not absent:
        # The exact shape of the bug this asset half exists to close: a
        # complete set of binaries staged by an older release, and no
        # map assets beside them.  Say so, rather than reporting a
        # generic re-fetch.
        print(f"woof fetch-bridges: the {len(staged)} artifacts at {dest} "
              f"are current, but {len(stale_assets) + len(absent_assets)} of "
              f"{len(bundle.assets)} map asset file(s) are missing or stale "
              "-- the renderer falls back to them when the recast-woof-data "
              "package carries none, and without either it draws plots "
              "with no coastlines or borders")
    if stale:
        print(f"woof fetch-bridges: {len(stale)} artifact(s) at {dest} do "
              "not match this release's pins and will be replaced once the "
              f"new bytes verify: {', '.join(p.filename for p in stale)}")

    source_dir = getattr(args, "from_dir", None)
    try:
        if source_dir is not None:
            installed = stage_from_dir(Path(source_dir), bundle, dest,
                                       progress=print)
        else:
            installed = fetch_bundle(
                pins, bundle, dest,
                keep_bundle=getattr(args, "keep_bundle", False),
                progress=print)
    except BridgeAssetError as error:
        print(f"woof fetch-bridges: REFUSED: {error}")
        return 2

    remaining = [pin.filename for pin in bundle.binaries
                 if not matches_pin(dest / pin.filename, pin)]
    installed_names = {path.name for path in installed}
    broken = [name for name in remaining if name in installed_names]
    if broken:
        # Something this invocation just wrote fails its pin: that is
        # an integrity failure, and it stays a refusal.
        print("woof fetch-bridges: REFUSED: staging finished but "
              f"{', '.join(broken)} still do not match their pins")
        return 2
    if remaining:
        # A partial --from DIR staging: everything written verified;
        # what the directory did not carry is reported, not fatal.
        warn(f"{len(remaining)} pinned artifact(s) are still not "
             f"staged ({', '.join(remaining)}); woof doctor names "
             "the remedy for each")
    verified = len(bundle.binaries) - len(remaining)
    print(f"woof fetch-bridges: {len(installed)} file(s) staged at "
          f"{dest}; {verified} of {len(bundle.binaries)} artifacts verified "
          "against the packaged pins.  `woof doctor` re-checks the "
          "full estate")
    unstaged_assets = [pin.path for pin in bundle.assets
                       if not matches_pin(dest / pin.path, pin)]
    if unstaged_assets:
        warn(f"{len(unstaged_assets)} of {len(bundle.assets)} pinned map "
             "asset file(s) are still not staged; the renderer will draw "
             "plots without coastlines or borders unless it finds them "
             "elsewhere")
    elif bundle.assets:
        print(f"woof fetch-bridges: {len(bundle.assets)} map asset file(s) "
              f"verified under {dest / ASSET_ROOT}")
    for note in _override_warnings(bundle):
        print(note)
    return 0


def staged_artifact_summary() -> str:
    """What a bundle carries, in one line, DERIVED from the table.

    ``--help`` used to carry a hand-written parenthesis -- "GRIB
    decoders, CPU preprocessing library, fetch backbone, batch renderer"
    -- written when those were all there was.  ``rw_nexrad`` joined the
    bundle and the line did not move, so the command that stages the
    radar front door did not say it stages the radar front door, and a
    reader looking for it had no reason to run this.  A hand-written
    inventory of a machine-readable table drifts on the first addition
    and stays wrong until someone notices; deriving it means the two
    cannot disagree.

    One phrase per artifact would be thirteen clauses, so the names are
    grouped by the workspace that builds them, in the table's own order,
    and each group names its members.  Adding an artifact adds a name
    here on the same commit, with no second edit to remember.
    """

    decoders = [a.name for a in BUNDLED_ARTIFACTS
                if a.crate == bridges.CRATE_RELATIVE]
    renderer = [a.name for a in BUNDLED_ARTIFACTS
                if a.crate != bridges.CRATE_RELATIVE]
    return (f"{', '.join(decoders)}; {', '.join(renderer)}")


def register_cli(subparsers) -> None:
    summary = staged_artifact_summary()
    parser = subparsers.add_parser(
        "fetch-bridges",
        help="download this platform's prebuilt Rust artifacts into "
             "this release's own ~/.woof/bridges/<release>-<digest> -- "
             "the GRIB decoders and CPU preprocessing "
             "library, the mapped decode engine, the fetch backbone, the "
             "batch renderer, the dealiasing engine and the radar, MRMS, "
             "Stage-IV, surface and GOES observation front doors -- each "
             "one verified "
             "against the packaged SHA-256 pins before it is installed; "
             "idempotent when everything is already staged",
        # `description` and not `help` alone, because `woof
        # fetch-bridges --help` prints the description and the audit
        # measured exactly that surface.  The artifact names are
        # DERIVED, so the day a bridge joins the bundle is the day this
        # text names it -- which is the property the hand-written
        # parenthesis did not have: it still said "GRIB decoders, CPU
        # preprocessing library, fetch backbone, batch renderer" long
        # after rw_nexrad and the dealiasing engine had joined.
        description=(
            "Download this platform's prebuilt Rust artifacts into "
            "this release's own directory, ~/.woof/bridges/<release>-"
            "<bundle digest>, each verified against the packaged "
            "SHA-256 pins before it is installed.  Another woof "
            "version's directory, and the flat ~/.woof/bridges that "
            "releases before 2.8.6 staged into, are never written.\n\n"
            f"This release's bundle carries: {summary}.\n\n"
            "Idempotent: everything already staged and pin-valid is "
            "left alone."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--from", dest="from_dir", metavar="DIR", default=None,
        help="stage from a local directory instead of downloading "
             "(offline installs): either the bundle archive or the "
             "artifacts loose in it; verification is identical")
    parser.add_argument(
        "--dest", metavar="DIR", default=None,
        help="stage into DIR instead of this release's own "
             "~/.woof/bridges/<release>-<digest> (woof finds "
             "the default on its own; anywhere else needs the "
             "per-artifact environment variables)")
    parser.add_argument(
        "--partial", action="store_true",
        help="fetch only missing or stale pinned binaries with HTTP byte ranges; "
             "map assets require a full fetch and servers must support ranges")
    parser.add_argument(
        "--artifact", action="append", metavar="NAME",
        help="with --partial, select this artifact (repeatable); renderer "
             "launchers also fetch their shared rw_wrfbatch executable")
    parser.add_argument(
        "--keep-bundle", action="store_true",
        help=f"keep the verified archive under <dest>/{ARCHIVE_SUBDIR} "
             "after staging (default: remove it)")
    parser.add_argument(
        "--list", action="store_true",
        help="print the platform, bundle and per-artifact staged state, "
             "then exit without touching the network")
    parser.set_defaults(func=fetch_bridges_main)
    return parser


__all__ = [
    "ARCHIVE_SUBDIR", "ASSET_ROOT", "ASSET_URL_BASE_ENV",
    "BUNDLED_ARTIFACTS", "BUNDLE_MANIFEST_SCHEMA", "REQUIRED_ASSET_SUBDIRS",
    "AssetPin", "BinaryPin", "BridgeAssetError", "BridgeFolderUnwritable",
    "BridgePins",
    "BundlePin", "BundledArtifact", "LIBRARY_ABI", "library_abi_for",
    "PINS_RESOURCE", "PINS_SCHEMA",
    "SOURCE_REV_MARKER", "STALE_POLICIES", "STALE_POLICY_ENV",
    "StagedPinStatus", "forget_pin_memo", "refresh_staged_bundle",
    "staged_pin_status", "stale_policy",
    "SUPPORTED_PLATFORMS", "artifact_filename", "asset_url_base",
    "bundle_url", "classify_assets", "classify_destination",
    "download_bundle", "embedded_source_revisions", "fetch_bundle",
    "NATIVE_BUILD_INPUTS", "NATIVE_BUILD_INPUT_EXCLUSIONS",
    "native_input_difference",
    "fetch_bridges_main", "host_platform", "host_platform_description",
    "load_pins", "matches_pin", "packaged_pins_path", "parse_pins",
    "register_cli", "sha256_file", "staged_artifact_summary",
    "stage_from_bundle", "stage_from_dir",
    "stage_from_loose_files", "staging_available", "verify_contract_marker",
    "verify_pinned_file", "verify_source_revision",
    "on_demand_artifacts", "wheel_artifacts",
]
