"""The mapped-engine parity battery: two engines, one answer, or FAIL.

The gate the decode-vendor design puts on flipping the default mapped
decode engine from Python to Rust.  It runs on REAL staged agency bytes
-- the model-gauntlet staging tree -- never on fixtures written to
flatter either side, and it runs the real ``gpuwm_mapped_engine``
executable rather than a stand-in.

Three things are proven here, and they answer three different questions.

1. **The seam format is lossless** (``frames.json`` + ``frames.f64``).
   Every staged source is decoded by the Python engine -- the behaviour
   of record -- written through the contract codec, and read back.
   Field arrays must be byte-identical, axes and missing counts equal,
   headers equal.  If the format cannot carry the Python engine's own
   answer it cannot carry the Rust engine's either, and every later
   comparison would be measuring the codec instead of the engines.

2. **The goldens are fixed** (``tests/data/mapped_engine_goldens``).
   The same decode is reduced to a parity digest -- per field: units,
   axes, location, staggering, shape, missing count and the SHA-256 of
   the ``<f8`` bytes; per frame: the times, the member, the grid
   fingerprint, the axis hashes and the portable header hash (rule
   gpuwm-portable-frame-header-v1) -- and compared to a
   committed golden.  A source whose Python decode REFUSES has its
   refusal class and message as its golden instead, because a refusal is
   an answer and the two engines have to agree about it too.  These are
   the numbers the Rust engine is measured against, extracted from the
   real Python engine on real bytes by
   ``tools/extract_mapped_engine_goldens.py``.

   Every committed golden DECLARES THE PLATFORM THAT MEASURED IT, and
   the declaration is what decides which reference a row clears; see
   "Platform scope" below.

3. **The engines agree.**  Where the engine implements a subcommand, the
   same source is decoded through it and its digest must equal the
   reference exactly.  An unexplained diff is FAIL: there is no
   tolerance and no "close enough" here, because both engines share
   grib-core lineage and a differing bit is a differing decode, not
   noise.

**Platform scope.**  A committed golden is a measurement of a platform,
not only of a decode.  On the platform that measured it the battery
compares against it byte for byte, exactly as it always did.  On any
other platform the same rows run the LIVE DUAL-ENGINE comparison at the
same strictness instead -- the same source, the same digest function,
the same fields, the Python engine against the Rust engine ON THIS BOX,
byte-identical or FAIL -- because a cross-platform golden diff there
reads a C-runtime difference as an engine defect.  What it never becomes
is a skip: every row still runs both engines and still has a bar to
clear, and the row output names which reference it cleared and why.

Where the vendored engine still refuses a subcommand with class
``not_implemented``, part 3 reports SKIPPED FOR THAT REASON, by name.
It is not silent and it is not green: the battery states which
subcommand is missing, so "the parity battery passes" can never be read
as "the Rust engine was compared" before there is an engine to compare.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import re
from typing import Sequence

import numpy as np
import pytest

from woof import bridges
from woof import mapped_engine_bridge as engine_bridge
from woof.mapped_source import (_array_sha256, _decode_mapped_source_python,
                                 _FORMATS as _MAPPED_SOURCE_FORMATS,
                                 _inspect_mapped_source_python, _sha256,
                                 libm_dependent_fields)
from woof.source_frame import (PORTABLE_HEADER_RULE,
                                portable_frame_header_sha256)
from _reserved_drives import gauntlet_staging_root


ROOT = Path(__file__).parents[1]
AUTHORITIES = ROOT / "woof" / "authorities"
CONFIGS = ROOT / "configs"
FIXTURES = ROOT / "tests" / "fixtures"
GOLDENS = ROOT / "tests" / "data" / "mapped_engine_goldens"
#: The staging root, never one on or linked onto a reserved drive
#: (tests/_reserved_drives.py names the breakage).
STAGING = gauntlet_staging_root()
#: The 1974 reference bundle (ERA5 GRIB1 + the CDO NetCDF oracle), the
#: same environment contract the rest of the suite uses for it.
BUNDLE = Path(os.environ.get(
    "WOOF_TEST_WRF74_BUNDLE", "gpuwm-fixture-unset/wrf74-bundle"))
#: The 20CRv3 private member sample (``memNNN_YYYYMMDDHH_{pl,sfc}.grb2``
#: pairs under ``PRESSURE/`` and ``SURFACE/``); the default is where the
#: recon staged member 072.
#:
#: THE DEFAULT IS WHAT RUNS THIS ROW; DO NOT TRADE IT FOR A SENTINEL.
#: These bytes are private, which is an argument for not writing a
#: holding's name into committed source -- and that argument was acted on
#: here once and reverted, because on the box that has the archive the
#: default is what makes this row RUN.  Swapping it for
#: ``gpuwm-fixture-unset/...`` turned a passing real-bytes row into a
#: silent skip, which is a coverage loss dressed as tidiness.  If the
#: path is to leave this file, the row has to keep running on the box
#: that owns the bytes by some other means first.
#:
#: The environment override is the portable half and is what a second box
#: uses.  Point it at a copy and the row runs there too: the golden's
#: portable header digest compares the decode independently of its root.
MEMBER_SAMPLE = Path(os.environ.get(
    "GPUWM_20CRV3_MEMBER_SAMPLE",
    str(Path.home() / "Downloads" / "1932-03-21 20CR Member 72 Files"),
))

DIGEST_SCHEMA = "gpuwm-mapped-parity-digest-v2"


# --------------------------------------------------------------------
# Platform scope: which box's C runtime measured a committed golden.
# --------------------------------------------------------------------
#
# MEASURED on a development machine (Linux) 2026-08-18 at 7f0db8ab2, field by field on
# `rap-awip32` -- 22 leaf diffs against the committed golden, and every
# one of them in the DERIVED layer: `eastward_wind`, `northward_wind`,
# `eastward_wind_10m`, `northward_wind_10m` (the grid-relative wind
# rotation) and `specific_humidity` (the relative-humidity derivation),
# plus the v1 raw header hashes that carry those fields. Every DIRECTLY
# DECODED field hashed identical.  And the PYTHON reference engine
# missed exactly the goldens the Rust engine missed, on exactly the same
# rows: 26 failures, 4 python decode + 4 rust decode + 3 rust inspection
# + 7 python compose + 7 rust compose.  Two engines cannot both be wrong
# in the same direction on the same five fields; what differs is the
# platform's libm under the derivation layer.
#
# So a committed golden is a measurement OF A PLATFORM, and it now says
# which.  On that platform the battery compares against it exactly as
# before: no tolerance, no masked member, nothing widened.  Anywhere
# else that comparison reads a C-runtime difference as an engine defect,
# so the row runs the LIVE DUAL-ENGINE comparison instead -- same
# source, same digest function, same fields, Python engine against Rust
# engine ON THIS BOX, byte-identical or FAIL.  That is the parity claim
# this battery exists to make, and it is the half that is portable; the
# committed numbers are not.
#
# The named breakage this prevents: without the scope, a Linux release
# verify reports 26 RED rows that name the Rust engine, when the Rust
# engine reproduced the Python engine's answer on that box exactly --
# and a cut is blocked, or worse, unblocked by someone widening the
# comparison until the derived fields stop being compared at all.
#
# What it must never become is a skip.  Every row still runs both
# engines and still has a bar to clear; only WHICH reference it clears
# against moves, and the row says which and why. Making the derived
# field arrays reproducible across platforms remains GOLD-PORTABLE
# (#157). The v2 digest uses portable headers while retaining the exact
# array comparison, including every libm-dependent field.

#: The member every committed golden carries naming its measuring
#: platform.  Top level of the golden FILE and never inside a digest, so
#: the compared content is byte-for-byte what it always was.
GOLDEN_PLATFORM_KEY = "measured_on"

#: Who owns making the committed numbers portable, named in every row
#: that falls back so a reader lands on the open item and not on a
#: rediscovery.
PORTABILITY_OWNER = (
    "cross-platform portability of the committed numbers is open item "
    "GOLD-PORTABLE (task #157), whose description carries the "
    "field-by-field libm measurement")


def measuring_platform() -> dict[str, str]:
    """The identity of the platform measuring, or being measured on.

    Deliberately COARSE -- the operating system and the machine
    architecture, nothing finer.  It names the C runtime family whose
    libm produces the derived fields, which is the difference that was
    measured; pinning a libc version instead would make a golden stop
    matching its own box after a system update, and pinning the hostname
    would make every box foreign.

    Two boxes of the same family with genuinely different libm therefore
    compare against each other's numbers.  That direction is the safe
    one: it produces a RED row naming a real numeric difference, never a
    green one that skipped the comparison.
    """

    return {"system": platform.system(), "machine": platform.machine()}


def load_golden(path: Path) -> tuple[dict, dict[str, str]]:
    """Read a committed golden as ``(document, measuring platform)``.

    The stamp is split OFF the document, so what the battery compares is
    exactly the content it compared before this member existed.

    A golden with no stamp FAILS rather than defaulting either way.
    Named breakage: defaulting to "this box measured it" compares one
    platform's numbers against another's and blames the engine;
    defaulting to "some other box measured it" silently stops comparing
    against the committed golden on the box that measured it, which is
    the gate itself going quiet.  Both are silent, so neither is a
    default.
    """

    document = json.loads(path.read_text(encoding="utf-8"))
    stamp = document.pop(GOLDEN_PLATFORM_KEY, None)
    if not isinstance(stamp, dict) or not stamp.get("system"):
        raise AssertionError(
            f"{path.name} does not declare the platform that measured it "
            f"(no {GOLDEN_PLATFORM_KEY!r} member).  A golden is a "
            "measurement of a platform: derived field arrays (wind "
            "rotation, specific humidity) differ by C runtime, so an "
            "unstamped golden would be "
            "compared across platforms and read as an engine defect.  "
            "Re-measure it on the box it belongs to with "
            "`python tools/extract_mapped_engine_goldens.py --source "
            f"{path.stem}` (add `--kind compose` for a compose golden)")
    return document, {"system": str(stamp.get("system")),
                      "machine": str(stamp.get("machine", ""))}


def golden_is_native(stamp) -> bool:
    """Was this golden measured on the platform running the battery?"""

    return dict(stamp) == measuring_platform()


def foreign_golden_note(source: str, stamp, what: str) -> str:
    """The sentence a fallen-back row carries, in its own output.

    Every assertion in the live path repeats it, so a failure explains
    the posture without a reader having to find this module first.
    """

    here = measuring_platform()
    return (
        f"{source}: the committed {what} golden is PLATFORM-SCOPED -- it "
        f"was measured on {stamp['system']}/{stamp['machine']} and this "
        f"is {here['system']}/{here['machine']}.  Comparing it here "
        "would read a libm difference in the derived fields as an engine "
        "defect, so this row runs the LIVE dual-engine comparison "
        "instead, at full strictness: the Python engine against the Rust "
        "engine on THIS box, same fields, same digests, byte-identical "
        f"or FAIL.  {PORTABILITY_OWNER}.")


def announce(note: str) -> None:
    """State a row's posture in its own output.

    Printed rather than warned: pytest captures it with the row, shows
    it on failure and with `-rA`, and it cannot be turned into an error
    by a warning filter and take a green battery red for saying what it
    did.
    """

    print(note)


# --------------------------------------------------------------------
# The staged gauntlet: one row per source, files named exactly.
# --------------------------------------------------------------------
#
# Named rather than globbed.  A glob would quietly change what the
# battery decodes when the staging tree gains a file, and a golden that
# silently starts describing different inputs is worse than no golden.
# The rows carry the SMALLEST input set that exercises the source's
# grammar: enough valid times for its declared boundary cadence where
# the staging tree has them, and a single time where it does not -- in
# which case the source's answer is a refusal, and the refusal is the
# golden.

def _files(directory: str, *names: str) -> tuple[Path, ...]:
    return tuple(STAGING / directory / name for name in names)


STAGED_SOURCES: dict[str, dict[str, object]] = {
    # 0.25-degree analysis-cycle GRIB2 on a regular lat/lon grid: the
    # broadest field set in the tree (33 pressure levels, 4 soil layers)
    # and the one both engines must get right first.
    "gdas-pgrb2-0p25": {
        "mapping": "rw-wps-gdas-pgrb2-0p25-grib2.mapping.json",
        "files": _files("gdas", "gdas.t06z.pgrb2.0p25.f000",
                        "gdas.t06z.pgrb2.0p25.f001"),
    },
    # Lambert conformal, grid-relative winds: the projected-grid and
    # wind-rotation path, which no lat/lon source exercises.
    "rap-awip32": {
        "mapping": "rw-wps-rap-awip32-grib2.mapping.json",
        "files": _files("rap", "rap.t00z.awip32f00.grib2",
                        "rap.t00z.awip32f01.grib2"),
    },
    # Two products per valid time (prslev + 2dfld) on one Lambert grid:
    # the multi-file-per-time assembly path.
    "rrfs-prslev-2dfld": {
        "mapping": "rw-wps-rrfs-prslev-2dfld-grib2.mapping.json",
        "files": _files(
            "rrfs",
            "OPS-rrfs.t00z.prslev.3km.f000.conus.grib2",
            "OPS-rrfs.t00z.2dfld.3km.f000.conus.grib2",
            "OPS-rrfs.t00z.prslev.3km.f001.conus.grib2",
            "OPS-rrfs.t00z.2dfld.3km.f001.conus.grib2",
        ),
    },
    # ECMWF open data: one file per valid time, many messages, and the
    # dewpoint/geopotential derivations.
    "ecmwf-open-data-oper": {
        "mapping": "rw-wps-ecmwf-open-data-oper-grib2.mapping.json",
        "files": _files("ifs", "20260816000000-0h-oper-fc.grib2",
                        "20260816000000-3h-oper-fc.grib2"),
    },
    # An AI atmosphere with no land surface: the source whose accurate
    # answer is a REFUSAL naming the state it does not publish.
    "aifs-single": {
        "mapping": "rw-wps-aifs-single-grib2.mapping.json",
        "files": _files("aifs", "20260817000000-0h-oper-fc.grib2",
                        "20260817000000-6h-oper-fc.grib2"),
    },
    # The AIFS step-0 donor decoded on its own, over the one object it
    # is ever handed: the cycle's f000, whose land mask and surface
    # geopotential every lead of the composition borrows.
    "aifs-single-step0-donor": {
        "mapping": "rw-wps-aifs-single-step0-donor.mapping.json",
        "files": _files("aifs", "20260817000000-0h-oper-fc.grib2"),
    },
    # The same shape from a different producer, split pressure/surface.
    # The NOMADS files, because the NOMADS mapping's selectors resolve
    # against NOMADS bytes: the same cycle re-published on S3 carries a
    # different product layout, and decoding it through this table
    # resolves two surface fields out of the whole atmosphere.
    "aigfs-nomads": {
        "mapping": "rw-wps-aigfs-nomads-grib2.mapping.json",
        "files": _files("aigfs", "NOMADS.aigfs.t00z.pres.f000.grib2",
                        "NOMADS.aigfs.t00z.sfc.f000.grib2",
                        "NOMADS.aigfs.t00z.sfc.f006.grib2"),
    },
    # Member-bearing bytes: the embedded ensemble identity path.
    "aigefs-member-hybrid": {
        "mapping": "rw-wps-aigefs-member-hybrid-grib2.mapping.json",
        "files": _files("aigefs", "mem001.aigefs.t00z.pres.f000.grib2",
                        "mem001.aigefs.t00z.sfc.f000.grib2",
                        "mem001.aigefs.t00z.pres.f006.grib2",
                        "mem001.aigefs.t00z.sfc.f006.grib2"),
    },
    # A 0.15-degree global grid published one field per file, at a
    # single valid time: the many-small-files case, refusing on cadence.
    "gem-gdps": {
        "mapping": "rw-wps-gem-gdps-grib2.mapping.json",
        "files": tuple(sorted(
            (STAGING / "gem-gdps").glob(
                "20260817T00Z_MSC_GDPS_*_PT000H.grib2"))),
    },
    # bzip2-compressed field-per-file inputs, 251 of them: the argv
    # transport and the acquisition codec in one row.
    "icon-eu-regular": {
        "mapping": "rw-wps-icon-eu-regular-grib2.mapping.json",
        "files": tuple(sorted(
            (STAGING / "icon" / "eu-regular-latlon").glob("*.grib2.bz2"))),
    },
    # The cross-source donor table, decoded on its own: the mapping a
    # composition pins by hash when it borrows a surface.
    "gdas-pgrb2-donor": {
        "mapping": "rw-wps-gdas-pgrb2-donor.mapping.json",
        "files": _files("crosssource", "gdas.t00z.pgrb2.0p25.f000"),
    },
    # Ensemble member bytes through the ensemble mapping: control member
    # pgrb2a+pgrb2b pairs at two valid times.  The member-identity octets
    # (PDT 1/11) ride every record, so this is where the two engines must
    # agree about an ensemble's embedded identity, not just its fields.
    "gefs-ensemble-control": {
        "mapping": "rw-wps-gefs-ensemble-grib2.mapping.json",
        "files": _files("gefs", "gec00.t00z.pgrb2a.0p50.f000",
                        "gec00.t00z.pgrb2b.0p50.f000",
                        "gec00.t00z.pgrb2a.0p50.f003",
                        "gec00.t00z.pgrb2b.0p50.f003"),
    },
    # The hybrid-init mapping decoded against its pressure product alone:
    # the accurate answer is a refusal naming the surface state the file
    # does not carry, and both engines must give it.
    "aigfs-gdas-hybrid-pres": {
        "mapping": "rw-wps-aigfs-gdas-hybrid-grib2.mapping.json",
        "files": _files("aigfs-hybrid",
                        "NOMADS.aigfs.t00z.pres.f006.grib2"),
    },
    # HRRR's public pressure product, whole files: the 1799x1059 Lambert
    # grid under complex/spatial-differencing packing -- the exact octet
    # family the grib-core convergence was hardened for, at full scale.
    "hrrr-prs": {
        "mapping": "rw-wps-hrrr-prs-grib2.mapping.json",
        "files": _files("hrrr", "hrrr.t00z.wrfprsf00.grib2",
                        "hrrr.t00z.wrfprsf01.grib2"),
    },
    # The 20CRv3 private member sample: paired pressure/surface files
    # whose PDT carries NO ensemble identity -- membership lives in the
    # filename and the sealed manifest, and decode without that manifest
    # must answer identically in both engines.
    "twentycrv3-member-pl-sfc": {
        "mapping": "rw-wps-20crv3-member-grib2.mapping.json",
        "files": (
            MEMBER_SAMPLE / "PRESSURE" / "mem072_1932032100_pl.grb2",
            MEMBER_SAMPLE / "SURFACE" / "mem072_1932032100_sfc.grb2",
            MEMBER_SAMPLE / "PRESSURE" / "mem072_1932032103_pl.grb2",
            MEMBER_SAMPLE / "SURFACE" / "mem072_1932032103_sfc.grb2",
        ),
    },
    # The repository's own real GFS fixture bytes through the compiled
    # pressure mapping: the row that runs on every checkout with no
    # staging tree at all, so the battery is never entirely optional.
    "gfs-pressure-fixture": {
        "mapping_path": CONFIGS / "rw-wps-gfs-pressure-grib2.mapping.json",
        "files": (
            FIXTURES / "gfs-scan-order"
            / "nomads-crop-20260729t18z-f000.grib2",
            FIXTURES / "gfs-scan-order"
            / "nomads-crop-soilw-20260729t18z-f000.grib2",
        ),
    },
    # ERA5 GRIB1 from the 1974 reference bundle.  This row existed to
    # measure a ROUTE while GRIB1 was an engine gap; since the port it
    # measures the DECODE, and it is the only real-bytes GRIB1
    # comparison there is: forty-two array digests across three frames
    # and fourteen fields, the grid fingerprint, and the
    # materialization refusal.
    "era5-1974-grib1": {
        "mapping_path": CONFIGS / "rw-wps-era5-1974-probe.mapping.json",
        "files": (BUNDLE / "era5_grib" / "era5_19740403.grb",),
        "format": "grib1",
    },
    # The CDO NetCDF oracle beside it.  Read the verdict narrowly: the
    # Python engine refuses this file in SELECTOR RESOLUTION, before any
    # array digest is recorded, because it carries no surface
    # geopotential and `terrain_height` is unresolvable.  So this row
    # compares one refusal sentence and ZERO field arrays -- refusal
    # parity, not byte parity.  The byte-level NetCDF evidence is the
    # `netcdf-pressure-level` crate golden and the Python NetCDF suites
    # under GPUWM_MAPPED_ENGINE=rust, not this row.
    "era5-netcdf": {
        "mapping_path": CONFIGS / "rw-wps-era5-netcdf.mapping.json",
        "files": (BUNDLE / "era5_grib" / "nc" / "era5_19740403_12.nc",),
        "format": "netcdf",
    },
}


def _mapping_of(entry) -> Path:
    """A row names its mapping in the authorities dir or by path."""

    if "mapping_path" in entry:
        return Path(entry["mapping_path"])
    return AUTHORITIES / str(entry["mapping"])


def _format_of(entry) -> str:
    return str(entry.get("format", "grib2"))


def _staged(entry) -> bool:
    files = entry["files"]
    return bool(files) and all(Path(path).is_file() for path in files)


# --------------------------------------------------------------------
# The parity digest, and the mask that makes two engines comparable.
# --------------------------------------------------------------------
#
# ENGINE-IDENTITY FIELDS ARE MASKED, and this is the whole mask:
#
#   * absolute input paths -- the manifest keys frames carry are
#     machine paths, so the digest keys inputs by BASENAME plus content
#     hash.  Two engines on one machine would agree on the path; a
#     golden that carried it would only be comparable on the machine
#     that made it.
#     The rule reaches every member that carries one, not only the
#     `inputs` table: `machine_independent` below rewrites the staged
#     paths wherever they appear, which is how the per-field
#     `source_references` and the refusal MESSAGES joined a rule they
#     had been walking past.
#   * decoder identity -- which binary decoded (path, size, hash) and
#     how long it took.  That is exactly what differs BY DESIGN between
#     the two engines, and it is the only thing that may.
#
# Nothing else is masked.  Every number a frame carries is in the digest
# and compared exactly.


def _staged_path_strings(source: str) -> tuple[str, ...]:
    """Every absolute path this battery hands the engines for ``source``.

    Include both handed and resolved spellings for linked staging roots.
    Longest first, so a directory that is a prefix of a file path can
    never eat half of that path before the file's own rule is tried.
    """

    entry = STAGED_SOURCES[source]
    handed = [str(path) for path in entry["files"]]
    handed.append(str(_mapping_of(entry)))
    paths = set(handed)
    paths.update(os.path.realpath(path) for path in handed)
    return tuple(sorted(paths, key=len, reverse=True))


def machine_independent(source: str, value, paths: Sequence[str] | None = None):
    """``value`` with every staged input path reduced to its basename.

    The mask above states the rule -- absolute input paths are keyed by
    BASENAME, because "a golden that carried it would only be comparable
    on the machine that made it" -- and two members walked past it: the
    per-field ``source_references`` an inspection report carries
    (``<input path>:<record index>``), and the refusal MESSAGE an engine
    writes when it names the file it could not satisfy.  Measured on the
    2.5.0 tip: fourteen goldens carried 3,966 absolute paths into one
    developer's staging root, a private reference bundle and a
    ``Downloads`` folder.  Two consequences, both real -- the release
    snapshot's machine-path gate refuses to build a tree carrying them,
    and no second box could ever reproduce those goldens, so the
    battery's parity claim was pinned to one machine while reading as a
    property of the code.

    The replacement is EXACT rather than pattern-matched: the battery
    knows the paths it handed the engines, so it substitutes those
    strings and nothing else.  A path that is not one of this source's
    inputs is left alone, where a regular expression would have to guess
    where a path with a space in it ends -- and one staged tree is
    ``1932-03-21 20CR Member 72 Files``.

    What survives is everything an engine comparison is about: the
    record index after the colon (which message a field came from is a
    number the two engines must agree on), the basename, and every
    hash.  Both sides of every comparison pass through here, so the
    engines are still compared on the same terms.

    ``paths`` names the strings to reduce when the caller knows a set this
    module's decode registry does not -- a COMPOSED source binds a
    supplement, a provenance document and a contributing mapping beside
    its primary inputs, and every one of them is an absolute path on the
    machine that measured.  Omitted, it is this source's decode-row
    inputs, which is what every existing golden was written with.
    """

    replacements = tuple(
        (path, re.split(r"[\\/]", path)[-1])
        for path in (_staged_path_strings(source) if paths is None
                     else sorted(set(paths), key=len, reverse=True)))

    def scrub(item):
        if isinstance(item, str):
            for path, name in replacements:
                if path in item:
                    item = item.replace(path, name)
            return item
        if isinstance(item, dict):
            return {key: scrub(inner) for key, inner in item.items()}
        if isinstance(item, (list, tuple)):
            return [scrub(inner) for inner in item]
        return item

    return scrub(value)


def parity_digest(source: str, mapping: Path, frames) -> dict[str, object]:
    """Reduce a decode to the numbers parity is judged on."""

    libm_fields = libm_dependent_fields(json.loads(
        mapping.read_text(encoding="utf-8")))
    return machine_independent(source, {
        "schema": DIGEST_SCHEMA,
        "header_rule": PORTABLE_HEADER_RULE,
        "source": source,
        "verdict": "DECODED",
        "mapping": {"name": mapping.name, "sha256": _sha256(mapping)},
        "frame_count": len(frames),
        "frames": [
            {
                "valid_time": frame.valid_time.isoformat(),
                "member": frame.member,
                "source_cycle": frame.source_cycle.isoformat(),
                "vertical_kind": frame.vertical_kind,
                "vertical_units": frame.vertical_units,
                "grid_fingerprint": frame.grid_fingerprint,
                "latitude": _axis_digest(frame.latitude),
                "longitude": _axis_digest(frame.longitude),
                "vertical": _axis_digest(frame.vertical_values),
                "inputs": sorted(
                    ({"name": Path(path).name, "sha256": digest}
                     for path, digest in frame.input_sha256.items()),
                    key=lambda row: (row["name"], row["sha256"]),
                ),
                # The arrays, including libm-derived fields, remain exact
                # in the fields table. Only the header uses the portable
                # rule shared with compose, before any path is hashed.
                "portable_header_sha256": portable_frame_header_sha256(
                    frame.header, inputs=_staged_path_strings(source),
                    libm_dependent=libm_fields),
                "fields": {
                    name: {
                        "units": field.units,
                        "axes": list(field.axes),
                        "location": field.location,
                        "staggering": field.staggering,
                        "shape": [int(size) for size in field.values.shape],
                        "missing_count": int(field.missing_count),
                        "sha256": _array_sha256(field.values),
                    }
                    for name, field in sorted(frame.fields.items())
                },
            }
            for frame in frames
        ],
    })


def refusal_digest(source: str, mapping: Path, error: BaseException):
    """A refusal is an answer, so it gets a digest of the same shape."""

    return machine_independent(source, {
        "schema": DIGEST_SCHEMA,
        "source": source,
        "verdict": "REFUSED",
        "mapping": {"name": mapping.name, "sha256": _sha256(mapping)},
        "refusal": {"type": type(error).__name__, "message": str(error)},
    })


#: The inspection document's decoder identity is masked. Materialized
#: frame headers use their portable digest below, as decode and compose
#: do. Every per-field ``sha256`` stays in the comparison exactly.
INSPECTION_MASK = ("decoders",)


def inspection_digest(source: str, document, error=None) -> dict[str, object]:
    """Reduce an inspection report to its machine-independent content.

    Inspection earns its place in the battery because it records the
    SHA-256 of every directly decoded array even when materialization
    then refuses -- which is the state most staged sources are in, since
    they reach a complete canonical frame only through a composition.
    Without it, most rows here would compare one refusal sentence and no
    bytes at all, and a Rust engine that decoded every array wrong would
    pass by refusing with the right words.
    """

    if document is None:
        return machine_independent(source, {
            "schema": DIGEST_SCHEMA,
            "source": source,
            "verdict": "REFUSED",
            "refusal": {"type": type(error).__name__, "message": str(error)},
        })
    document = dict(document)
    for key in INSPECTION_MASK:
        document.pop(key, None)
    document["mapping"] = {
        "name": Path(str(dict(document["mapping"])["path"])).name,
        "sha256": dict(document["mapping"])["sha256"],
    }
    document["inputs"] = sorted(
        ({"name": Path(str(row["path"])).name,
          "bytes": row["bytes"], "sha256": row["sha256"]}
         for row in document["inputs"]),
        key=lambda row: (row["name"], row["sha256"]),
    )
    materialization = dict(document.get("materialization") or {})
    if materialization.get("verdict") == "PASS":
        headers = materialization.get("frame_header_sha256_portable")
        assert (materialization.get("portable_rule") == PORTABLE_HEADER_RULE
                and isinstance(headers, list)
                and len(headers) == materialization["frame_count"]
                and all(isinstance(value, str)
                        and re.fullmatch(r"[0-9a-f]{64}", value)
                        for value in headers)), (
            "materialized inspection lacks complete portable header digests; "
            "dropping its raw headers would leave frame metadata uncompared")
        materialization.pop("frame_header_sha256", None)
    document["materialization"] = materialization
    return machine_independent(source, {
        "schema": DIGEST_SCHEMA,
        "source": source,
        "verdict": "INSPECTED",
        "inspection": document,
    })


def _axis_digest(values) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return {"count": int(array.size), "sha256": _array_sha256(array)}


def _canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        default=str,
    ).encode("utf-8")).hexdigest()


def canonical(value) -> str:
    """Canonical JSON, for the one comparison the battery makes."""

    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False,
                      default=str)


#: Which subprocess decoder tools the Python engine needs per format.
#: NetCDF needs none here: the battery calls the underscore Python-engine
#: functions directly, so the pin the tools also provide is not needed to
#: keep the reference off the Rust route.
_FORMAT_TOOLS = {
    "grib2": ("grib2_inventory", "grib2_dump"),
    "grib1": ("grib1_bridge",),
    "netcdf": (),
}


def _python_tools(
        source_format: str | tuple[str, ...] = "grib2") -> dict[str, Path]:
    """The subprocess decoder tools, named EXPLICITLY.

    Naming them is also how a caller pins the Python engine, so this
    side of the battery is independent of whichever engine happens to be
    the default -- the reference cannot quietly become the thing being
    measured.  Accepts one format or several, because a composition's
    decoder contract is the UNION of its primary and donor formats
    (see :func:`_composed_formats`).
    """

    formats = ((source_format,) if isinstance(source_format, str)
               else tuple(source_format))
    tools = {
        name: bridges.find_bridge(name)
        for fmt in formats
        for name in _FORMAT_TOOLS[fmt]
    }
    if any(path is None for path in tools.values()):
        raise FileNotFoundError("staged decoder executables absent")
    return tools


def decode_with_python_engine(source: str):
    """Decode one staged source on the Python engine of record."""

    entry = STAGED_SOURCES[source]
    mapping = _mapping_of(entry)
    tools = _python_tools(_format_of(entry))
    try:
        frames = _decode_mapped_source_python(
            mapping, entry["files"], **tools)
    except (KeyError, TypeError, ValueError) as error:
        return mapping, None, error
    return mapping, frames, None


def inspect_with_python_engine(source: str):
    """Inspect one staged source on the Python engine of record.

    Inspection can itself REFUSE: it reports on materialization, but a
    fault below that -- a selector that matched no record, a vertical
    ladder the bytes do not cover -- stops the decode before there is
    anything to report on.  That refusal is an answer like any other and
    is digested as one, so a Rust engine that decoded such a source into
    a cheerful report would fail here rather than pass.
    """

    entry = STAGED_SOURCES[source]
    tools = _python_tools(_format_of(entry))
    try:
        return _inspect_mapped_source_python(
            _mapping_of(entry), entry["files"], **tools,
        ), None
    except (KeyError, TypeError, ValueError) as error:
        return None, error


# --------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------

_DECODES: dict[str, tuple] = {}


def python_decode(source: str):
    """One Python decode per source per session; these are not cheap."""

    if source not in _DECODES:
        _DECODES[source] = decode_with_python_engine(source)
    return _DECODES[source]


def release_python_answers(source: str) -> None:
    """Drop a source's cached frames once its last consumer has run.

    Named breakage: seventeen cached decodes are ~50 GB of frames on
    this battery's staged set, and a cache that outlives its consumers
    starves the DECODER CHILDREN spawned late in the session -- the
    engine exe, `grib2_dump` on a 400 MB file -- so a full run failed on
    a source whose isolated run passes, and the failure landed on
    whichever comparison happened to need memory next.  Each test that
    consumes the Python answer releases it; the next consumer re-decodes,
    which costs minutes and is deterministic, where the cache was fast
    and was not.
    """

    _DECODES.pop(source, None)


def measure_decode_digest(source: str) -> dict[str, object]:
    """The Python engine's DECODE answer for one source, measured HERE.

    The frames are dropped before returning for the reason
    :func:`release_python_answers` records: whatever runs next spawns a
    decoder child on the same bytes and has to fit beside them.
    """

    mapping, frames, refusal = decode_with_python_engine(source)
    digest = (refusal_digest(source, mapping, refusal) if frames is None
              else parity_digest(source, mapping, frames))
    del frames
    return digest


def measure_inspection_digest(source: str) -> dict[str, object]:
    """The Python engine's INSPECTION answer for one source, measured HERE."""

    return inspection_digest(source, *inspect_with_python_engine(source))


def measure_decode_document(source: str) -> dict[str, object]:
    """The complete Python-engine answer for one source: decode AND inspect.

    One function for two callers -- the extractor, which commits it as a
    golden, and the row that falls back to it as a live reference off the
    measuring platform -- so the fallback cannot drift into measuring
    something the golden never did.
    """

    return {
        "schema": DIGEST_SCHEMA,
        "source": source,
        "decode": measure_decode_digest(source),
        "inspect": measure_inspection_digest(source),
    }


def golden_document(source: str) -> dict[str, object]:
    """A committed decode golden: the measurement, and whose it is."""

    return {**measure_decode_document(source),
            GOLDEN_PLATFORM_KEY: measuring_platform()}


def requires(source: str) -> None:
    entry = STAGED_SOURCES[source]
    if not _staged(entry):
        pytest.skip(f"{source}: staged real-byte samples absent")
    for name in _FORMAT_TOOLS[_format_of(entry)]:
        if bridges.find_bridge(name) is None:
            pytest.skip(f"staged {name} decoder executable absent")


def golden_path(source: str) -> Path:
    return GOLDENS / f"{source}.json"


def native_golden_note(source: str, stamp, what: str) -> str:
    """The sentence a row carries when the committed golden is this box's."""

    return (f"{source}: the committed {what} golden was measured on "
            f"{stamp['system']}/{stamp['machine']}, which is this platform, "
            "so it is compared byte for byte against the committed numbers")


def engine_available_for(source: str, subcommand: str,
                         monkeypatch) -> tuple[bool, str]:
    """Is there a second engine to compare this source's ``subcommand``?

    The two conditions the battery has always conceded, in one place and
    both of them measured rather than assumed:

    * the format is a declared engine gap, in which case the ROUTE is
      asserted first (a bare run goes to Python, an explicit rust
      request still reaches the engine), so the answer states a measured
      fact;
    * the built engine refuses the subcommand as ``not_implemented``,
      which it is asked by RUNNING it.

    Neither is a platform question and neither is silent; the caller
    reports which one, by name.
    """

    entry = STAGED_SOURCES[source]
    source_format = _format_of(entry)
    if not engine_bridge.engine_supports(subcommand, source_format):
        _unported_format_route_is_pinned(source, monkeypatch)
        return False, (
            f"format {source_format!r} is a declared engine gap "
            "(ENGINE_GAPS); a bare run routes to the Python engine -- the "
            "route was just asserted, and there is no second engine to "
            "compare until the gap closes")
    implemented, reason = engine_implements(subcommand)
    if not implemented:
        return False, (
            f"gpuwm_mapped_engine does not implement `{subcommand}` yet, so "
            "no Rust answer exists to compare -- this is NOT a passing "
            f"comparison: {reason}")
    return True, ""


def assert_engine_decode_matches(source: str, entry, mapping: Path,
                                 expected, note: str) -> None:
    """Run the engine's ``decode`` and hold it to ``expected``, exactly.

    ``expected`` is the committed golden's decode digest on the platform
    that measured it, and the Python engine's live decode digest on this
    box anywhere else.  The comparison is identical either way -- same
    digest function, same fields, no tolerance -- so the fallback is a
    change of reference and not a change of strictness.
    """

    import tempfile

    if expected["verdict"] == "REFUSED":
        with pytest.raises(Exception) as caught:      # noqa: PT011 - typed below
            with tempfile.TemporaryDirectory() as work:
                engine_bridge.run_engine(
                    "decode", mapping=mapping, files=entry["files"],
                    output=Path(work) / "frameset")
        assert type(caught.value).__name__ == expected["refusal"]["type"], (
            "the engines disagree about the TYPE of refusal this source "
            "earns, so a caller catching one would not catch the other.  "
            + note)
        assert machine_independent(source, str(caught.value)).startswith(
                expected["refusal"]["message"]), (
            "the engines refuse this source for the same reason and say "
            "it differently; the message is what a reader acts on, so it "
            "is compared too (the engine appends its remedy sentence).  "
            + note)
        return

    with tempfile.TemporaryDirectory(prefix="gpuwm-parity-") as work:
        directory = Path(work) / "frameset"
        engine_bridge.run_engine(
            "decode", mapping=mapping, files=entry["files"],
            output=directory)
        frames = engine_bridge.read_frameset(directory)
    observed = parity_digest(source, mapping, frames)
    del frames
    assert canonical(observed) == canonical(expected), (
        "the Rust engine's decode differs from the Python engine's on "
        f"{source}; an unexplained diff is FAIL -- name the defective "
        "side and its evidence in a commit before calling this expected.  "
        + note)


def assert_engine_inspection_matches(source: str, entry, expected,
                                     note: str) -> None:
    """Run the engine's ``inspect`` and hold it to ``expected``, exactly."""

    import tempfile

    if expected.get("verdict") == "REFUSED":
        # A source both engines refuse still belongs here: the comparison
        # is the refusal itself.  Letting the exception escape would
        # report an ERROR and read as "the Rust engine broke", when what
        # happened is the answer both engines give.
        with pytest.raises(Exception) as caught:      # noqa: PT011
            with tempfile.TemporaryDirectory() as work:
                engine_bridge.run_engine(
                    "inspect", mapping=_mapping_of(entry),
                    files=entry["files"], output=Path(work) / "inspection")
        assert type(caught.value).__name__ == expected["refusal"]["type"], note
        assert machine_independent(source, str(caught.value)).startswith(
                expected["refusal"]["message"]), (
            "the engines refuse this source for the same reason and say "
            "it differently.  " + note)
        return

    with tempfile.TemporaryDirectory(prefix="gpuwm-parity-inspect-") as work:
        result = engine_bridge.run_engine(
            "inspect", mapping=_mapping_of(entry),
            files=entry["files"], output=Path(work) / "inspection")
    from woof.mapped_source import _inspection_from_stdout

    document = _inspection_from_stdout(str(result["stdout"]))
    observed = inspection_digest(source, document)
    assert canonical(observed) == canonical(expected), (
        "the Rust engine's inspection differs from the Python engine's "
        f"on {source}; an unexplained diff is FAIL.  " + note)


# --------------------------------------------------------------------
# The COMPOSE half: one row per registered composition source.
# --------------------------------------------------------------------
#
# `decode` is not the subcommand a mapped preparation runs.  Every
# `mapped_composition_v1` source reaches a complete canonical frame
# through `compose` -- terrain composition, bound-field borrow, the
# coordinate-subset solve -- and `mapped_engine_bridge`'s
# MAPPED_ROUTE_SUBCOMMAND says so.  So the byte work that a bare
# `woof prep` actually performs had no golden at all: the seventeen
# rows above measure the decode UNDER it.
#
# These rows measure the composed answer, from the same real staged
# bytes, through the same Python engine of record.  What a row names is
# the PRIMARY input files, exactly, the way the decode rows do -- a glob
# would quietly change what a golden describes.  Everything else is
# derived from the REGISTRY: the mapping, the composition, the
# provenance document, the contributing donor mappings and the two
# composition roles all come off the packaged profile, so a model added
# as table data inherits a compose golden instead of needing a recipe of
# its own (the arbitrary acceptance test).

COMPOSE_GOLDENS = ROOT / "tests" / "data" / "mapped_engine_compose_goldens"

#: v2: each frame's header is digested in its portable form
#: (``portable_header_sha256``, rule ``gpuwm-portable-frame-header-v1``)
#: where v1 digested the raw header.  The raw header quotes every field's
#: ``<absolute input path>:<record index>``, and a hash cannot be masked
#: after the fact, so a v1 compose golden reproduced only with the bytes
#: staged at the very path that measured it: two goldens stamped from a
#: staging folder that was later deleted failed on every other box and at
#: every other path, while every array, input hash and receipt matched.
COMPOSE_DIGEST_SCHEMA = "gpuwm-mapped-compose-parity-digest-v2"

#: How a row's supplement inventory is drawn from its primary inputs.
#:
#: These are the three shapes the shipped route table declares
#: (``prep.supplement.from`` in ``rw-wps-fetch-routes.v1.json``): terrain
#: in every input, terrain in a separate invariant object, and no terrain
#: supplement at all because a cross-source binding supplies it (a
#: same-cycle donor, or the cycle's own step-0 object).
_SUPPLEMENT_EVERY = "every_primary"
_SUPPLEMENT_NAMED = "named"
_SUPPLEMENT_DONOR = "donor"

COMPOSED_SOURCES: dict[str, dict[str, object]] = {
    # In-band terrain, exact valid-time alignment, 1799x1059 Lambert:
    # the composition shape most registered sources have, at full scale.
    "hrrr-prs": {
        "primary": _files("hrrr", "hrrr.t00z.wrfprsf00.grib2",
                          "hrrr.t00z.wrfprsf01.grib2"),
        "supplement": _SUPPLEMENT_EVERY,
    },
    # The same shape on a 32 km Lambert whose winds are grid-relative and
    # whose packing is JPEG2000 (DRT 5.40).
    "rap": {
        "primary": _files("rap", "rap.t00z.awip32f00.grib2",
                          "rap.t00z.awip32f01.grib2"),
        "supplement": _SUPPLEMENT_EVERY,
    },
    # RAP's native hybrid-level product (awp130bgrb, 13 km Lambert): fifty
    # model levels with explicit pressure, analyzed aerosol and hydrometeor
    # numbers that keep their masks, in-band terrain and the nine-node soil
    # column, all in one file per valid time.
    "rap-native": {
        "primary": _files("rap", "rap.t00z.awp130bgrbf00.grib2",
                          "rap.t00z.awp130bgrbf01.grib2"),
        "supplement": _SUPPLEMENT_EVERY,
    },
    # CROSS-PRODUCT: HRRR's native hybrid analysis (wrfnat) borrowing
    # terrain and the nine-node soil column from the same cycle's pressure
    # product (wrfprs) under the analysis broadcast.  One frame: the native
    # source supplies an initial state and no lateral boundary.
    "hrrr-native": {
        "primary": _files("hrrr", "hrrr.t00z.wrfnatf00.grib2"),
        "supplement": _SUPPLEMENT_DONOR,
        "donor_files": _files("hrrr", "hrrr.t00z.wrfprsf00.grib2"),
    },
    # Two products per valid time on one Lambert grid.
    "rrfs": {
        "primary": _files(
            "rrfs",
            "OPS-rrfs.t00z.prslev.3km.f000.conus.grib2",
            "OPS-rrfs.t00z.2dfld.3km.f000.conus.grib2",
            "OPS-rrfs.t00z.prslev.3km.f001.conus.grib2",
            "OPS-rrfs.t00z.2dfld.3km.f001.conus.grib2",
        ),
        "supplement": _SUPPLEMENT_EVERY,
    },
    # Regular lat/lon, the broadest field set in the tree.
    "gdas": {
        "primary": _files("gdas", "gdas.t06z.pgrb2.0p25.f000",
                          "gdas.t06z.pgrb2.0p25.f001"),
        "supplement": _SUPPLEMENT_EVERY,
    },
    # Ensemble member bytes: the embedded-identity path, composed.
    "gefs": {
        "primary": _files("gefs", "gec00.t00z.pgrb2a.0p50.f000",
                          "gec00.t00z.pgrb2b.0p50.f000",
                          "gec00.t00z.pgrb2a.0p50.f003",
                          "gec00.t00z.pgrb2b.0p50.f003"),
        "supplement": _SUPPLEMENT_EVERY,
    },
    # ECMWF open data: the cycle-invariant terrain broadcast, where the
    # producer publishes its statics into the analysis frame only.
    "ecmwf-open-data": {
        "primary": _files("ifs", "20260816000000-0h-oper-fc.grib2",
                          "20260816000000-3h-oper-fc.grib2"),
        "supplement": _SUPPLEMENT_EVERY,
        # The sea-ice binding's donor: the same cycle's GFS analysis (ECMWF
        # open data publishes sea-ice thickness only), from the AWS archive.
        "field_source_files": {
            "seaice_data": _files("crosssource", "gfs.20260816.t00z.pgrb2.0p25.f000"),
        },
    },
    # An AI model whose DECODE refuses: its land mask and terrain ride
    # the step-0 object alone, so compose binds both from the cycle's
    # own f000 (the donor) and carries them to every lead.  This window
    # starts at f000, so the donor is also the first primary: f000 is
    # matched and f006 carried.
    "aifs": {
        "primary": _files("aifs", "20260817000000-0h-oper-fc.grib2",
                          "20260817000000-6h-oper-fc.grib2"),
        "supplement": _SUPPLEMENT_DONOR,
        "donor_files": _files("aifs", "20260817000000-0h-oper-fc.grib2"),
        # The sea-ice binding's donor: the same cycle's GFS analysis (AIFS
        # publishes no sea-ice field), from the AWS archive.
        "field_source_files": {
            "seaice_data": _files("crosssource", "gfs.20260817.t00z.pgrb2.0p25.f000"),
        },
    },
    # A 0.15-degree global grid published one field per file, terrain in
    # its own invariant object.
    "gem-gdps": {
        "primary": _files(
            "gem-gdps",
            *(f"20260817T00Z_MSC_GDPS_{name}_LatLon0.15_PT000H.grib2"
              for name in (
                  "AirTemp_AGL-2m",
                  "AirTemp_IsbL-0850",
                  "GeopotentialHeight_IsbL-0850",
                  "GeopotentialHeight_Sfc",
                  "LandWaterProportion_Sfc",
                  "Pressure_Sfc",
                  "RadiativeTemp_Sfc",
                  "RelativeHumidity_IsbL-0850",
                  "SeaIceFraction_Sfc",
                  "SnowDensity_Sfc",
                  "SnowDepth_Sfc",
                  "SoilTemp_DBS-0to10cm",
                  "SoilTemp_Sfc",
                  "SoilVolumetricIceContent_Sfc",
                  "SoilVolumetricWaterContent_DBS-0to10cm",
                  "SoilVolumetricWaterContent_DBS-0to1cm",
                  "SpecificHumidity_IsbL-0850",
                  "WindU_AGL-10m",
                  "WindU_IsbL-0850",
                  "WindV_AGL-10m",
                  "WindV_IsbL-0850",
              )),
        ),
        "supplement": _SUPPLEMENT_NAMED,
        "supplement_files": _files(
            "gem-gdps",
            "20260817T00Z_MSC_GDPS_GeopotentialHeight_Sfc_LatLon0.15_"
            "PT000H.grib2"),
    },
    # DWD ICON-EU: 251 bz2-wrapped field-per-file objects plus a separate
    # invariant terrain object -- the acquisition codec inside compose.
    # The ONE row whose inputs are not literals; see `_icon_eu_inputs`
    # for why (252 literals, and the grammar is the fetch door's).
    "icon-eu": {"primary": (), "supplement": _SUPPLEMENT_NAMED},
    # CROSS-SOURCE: an atmosphere-only AI product borrowing seven
    # land-surface fields from the same cycle's GDAS analysis, on an
    # exact coordinate subset, under the analysis broadcast.  The DECODE
    # golden for these bytes is a REFUSAL naming the missing surface
    # state; compose is what supplies it.
    #
    # The input set is the one the hybrid lane's end-to-end proof used
    # and the staging tree's own README names: the recon-staged NOMADS
    # pres/sfc pair at f000, the f006 pres object the hybrid lane added
    # (recon staged pres at f000 only), the f006 sfc object, and the
    # same-cycle GDAS donor.  Operational NOMADS bytes, not the S3
    # re-publication, because the packaged mapping's selectors resolve
    # against the operational product layout.
    "aigfs": {
        "primary": (
            *_files("aigfs", "NOMADS.aigfs.t00z.pres.f000.grib2",
                    "NOMADS.aigfs.t00z.sfc.f000.grib2"),
            *_files("aigfs-hybrid", "NOMADS.aigfs.t00z.pres.f006.grib2"),
            *_files("aigfs", "NOMADS.aigfs.t00z.sfc.f006.grib2"),
        ),
        "supplement": _SUPPLEMENT_DONOR,
        "donor_files": _files("crosssource", "gdas.t00z.pgrb2.0p25.f000"),
    },
    # The same borrow with an ENSEMBLE MEMBER primary: member identity
    # has to survive a cross-source join, and the donor is single-member.
    "aigefs": {
        "primary": _files("aigefs", "mem001.aigefs.t00z.pres.f000.grib2",
                          "mem001.aigefs.t00z.sfc.f000.grib2",
                          "mem001.aigefs.t00z.pres.f006.grib2",
                          "mem001.aigefs.t00z.sfc.f006.grib2"),
        "supplement": _SUPPLEMENT_DONOR,
        "donor_files": _files("crosssource", "gdas.t00z.pgrb2.0p25.f000"),
    },
    # ERA5 native model levels: a 137-hybrid-level GRIB2 primary (t/u/v/q;
    # pressure and height are DERIVED via the in-band Section-4 pv ladder)
    # borrowing its ten surface/soil fields and its water state (sea
    # surface temperature, sea ice, lake water and ice) from the same
    # hours' pressure-level/single-level combined file -- the exact donor
    # bytes the certified `era5` front door produces.  One hour-block,
    # 2026-05-05 12-15Z, 0.25-degree subset; alignment is
    # exact_coordinate_subset + valid_time_exact by construction of the
    # fetch.  The model-level file was staged 2026-08-31; the donor was
    # fetched again 2026-09-28 because the one staged beside it predates
    # the lake state the era5 fetch has requested since 2.7.0.
    "era5-l137": {
        "primary": _files("era5-l137",
                          "era5-ml-l137-20260505-12z-15z.grib2"),
        "supplement": _SUPPLEMENT_DONOR,
        "donor_files": _files("era5-l137",
                              "era5-combined-lakes-20260505-12z-15z.grib"),
    },
}


def _icon_eu_inputs() -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """ICON-EU's 251 primary objects and its one invariant terrain object.

    Named by DWD's own file grammar rather than listed: the row would
    otherwise be 251 literals, and the grammar (``<cycle>_<lead>_<grid>_
    <PARAM>.grib2.bz2``) is the acquisition door's, not this battery's.
    The terrain object is the HSURF one, which DWD publishes once per
    cycle as a time-invariant object; everything else is primary.
    """

    directory = STAGING / "icon" / "eu-regular-latlon"
    objects = tuple(sorted(directory.glob("*.grib2.bz2")))
    terrain = tuple(path for path in objects if "_HSURF" in path.name.upper())
    return objects, terrain


def _composed_files(source: str) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """``(primary, supplement)`` for one composed row."""

    entry = COMPOSED_SOURCES[source]
    if source == "icon-eu":
        return _icon_eu_inputs()
    primary = tuple(entry["primary"])
    kind = str(entry["supplement"])
    if kind == _SUPPLEMENT_EVERY:
        return primary, primary
    if kind == _SUPPLEMENT_NAMED:
        return primary, tuple(entry["supplement_files"])
    if kind == _SUPPLEMENT_DONOR:
        return primary, tuple(entry["donor_files"])
    raise ValueError(f"{source}: unknown supplement shape {kind!r}")


def composed_recipe(source: str) -> dict[str, object]:
    """The complete compose invocation for one registered source.

    Read off the REGISTRY: the packaged profile decides the mapping, the
    composition, the provenance document, the contributing donor
    mappings and the two role names, exactly as ``woof prep`` decides
    them, so this cannot describe a route no user can reach.
    """

    from woof.source_adapters import get_source_adapter
    from woof.source_authorities import (packaged_authorities,
                                          packaged_contributing_mappings,
                                          packaged_profile,
                                          packaged_provenance_files)

    adapter = get_source_adapter(source)
    assert adapter.runner == "mapped_composition_v1", source
    name = str(adapter.packaged_profile)
    profile = packaged_profile(name)
    authorities = packaged_authorities(name)
    primary, supplement = _composed_files(source)
    # A binding whose donor is not the profile's own supplement role (the
    # ecmwf-open-data sea-ice donor beside its in-band terrain) names its
    # files by data role; every provenance role takes its packaged document.
    supplements = {str(profile["data_role"]): supplement}
    supplements.update({str(role): tuple(files) for role, files in
                        COMPOSED_SOURCES[source].get("field_source_files", {}).items()})
    return {
        "source": source,
        "format": str(profile["source_format"]),
        "mapping": Path(authorities["mapping"]),
        "composition": Path(authorities["composition"]),
        "primary": primary,
        "supplements": supplements,
        "provenance": {str(role): Path(path) for role, path
                       in packaged_provenance_files(name).items()},
        "contributing": {str(role): Path(path) for role, path
                         in packaged_contributing_mappings(name).items()},
    }


def _composed_path_strings(recipe) -> tuple[str, ...]:
    """Every absolute path a composed row hands the engines.

    Each one is listed as handed AND as resolved: the composition resolves
    its inputs before it records them, so a staging root reached through a
    junction or a symlink is quoted by its target's spelling, which the
    handed spelling does not match.  Measured with a default staging root
    that is a junction: the frame provenance read the physical location
    and escaped the mask.
    """

    handed = [str(path) for path in recipe["primary"]]
    handed.extend(str(path) for paths_ in recipe["supplements"].values()
                  for path in paths_)
    handed.extend(str(path) for path in recipe["provenance"].values())
    handed.extend(str(path) for path in recipe["contributing"].values())
    handed.append(str(recipe["mapping"]))
    handed.append(str(recipe["composition"]))
    paths = set(handed)
    paths.update(os.path.realpath(path) for path in handed)
    return tuple(sorted(paths, key=len, reverse=True))


def _composed_libm_fields(recipe) -> frozenset[str]:
    """The fields a composed frame may carry from a transcendental.

    Read from the declarations of every mapping the composition binds, the
    primary and each contributing donor, so a new source is table work.
    """

    names: set[str] = set()
    for path in (recipe["mapping"], *recipe["contributing"].values()):
        with open(path, encoding="utf-8") as handle:
            names.update(libm_dependent_fields(json.load(handle)))
    return frozenset(names)


def composed_header_digest(recipe, header) -> str:
    """A composed frame's header, digested where no box can move it.

    The raw header quotes every field's ``<absolute input path>:<record
    index>``, and a digest cannot be masked afterwards, so the raw digest is
    an identity of where the bytes were staged.  The portable form
    (:func:`woof.source_frame.portable_frame_header`) reduces each input
    this row hands the engines to its file name and names each
    libm-dependent field's array instead of hashing it.  Nothing leaves the
    comparison: every field's array digest, the libm-dependent ones
    included, is compared exactly in the frame's ``fields`` table beside
    this one.
    """

    return portable_frame_header_sha256(
        header, inputs=_composed_path_strings(recipe),
        libm_dependent=_composed_libm_fields(recipe))


def _composed_staged(source: str) -> bool:
    try:
        recipe = composed_recipe(source)
    except (KeyError, FileNotFoundError, RuntimeError):
        return False
    files = (*recipe["primary"],
             *(path for paths in recipe["supplements"].values()
               for path in paths),
             *recipe["provenance"].values(), *recipe["contributing"].values())
    return bool(recipe["primary"]) and all(Path(p).is_file() for p in files)


@contextlib.contextmanager
def _python_engine_pinned():
    """Both halves of a compose measurement on the Python engine.

    Naming the subprocess tools pins Python for a GRIB source, and that
    is the rule :func:`_python_tools` states for the decode half.  It is
    not enough here, for two reasons the measurement has to survive:

    * a NetCDF mapping launches NO subprocess tool, so there is nothing
      to name and the tool spelling cannot pin anything;
    * the manifest and the composition ask the capability table about
      DIFFERENT subcommands -- ``mapped_authoring`` about ``decode``,
      ``decode_composed_source`` about ``compose``.  With no pin, a
      NetCDF row would seal its manifest against the in-process engine
      (decode is declared) and then verify it against an empty decoder
      inventory (compose is not), and ``_verify_manifest`` would refuse
      a preparation that is in fact correct -- recording a REFUSAL
      golden for a source that composes.

    So the reference is pinned by the documented environment spelling
    for the whole measurement, restored afterwards.
    """

    previous = os.environ.get(engine_bridge.ENGINE_ENV)
    os.environ[engine_bridge.ENGINE_ENV] = engine_bridge.ENGINE_PYTHON
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(engine_bridge.ENGINE_ENV, None)
        else:
            os.environ[engine_bridge.ENGINE_ENV] = previous


def _composed_formats(recipe) -> tuple[str, ...]:
    """Every format the composition decodes, primary and donors alike.

    ``decode_composed_source`` computes its decoder-inventory contract
    from the primary mapping's format UNIONED with each contributing
    donor mapping's (woof/mapped_composition.py), so the reference must
    hand it the same tool set.  Reading only the profile's primary
    format under-supplies any cross-format donor: the first GRIB1 donor
    to arrive (era5-l137's pressure-level/single-level combined file
    under a GRIB2 model-level primary) refused with
    ``missing=['grib1_bridge']`` on a fully staged tree.
    """

    formats = {str(recipe["format"])}
    for path in recipe["contributing"].values():
        with open(path, encoding="utf-8") as handle:
            formats.add(str(json.load(handle)["format"]))
    return tuple(sorted(formats))


def compose_with_python_engine(source: str, work: Path):
    """Compose one registered source on the Python engine of record.

    The subprocess decoder tools are named EXPLICITLY and the engine is
    pinned by environment, so the reference cannot quietly become the
    thing being measured.
    """

    from woof.mapped_authoring import author_input_manifest
    from woof.mapped_composition import decode_composed_source

    recipe = composed_recipe(source)
    tools = _python_tools(_composed_formats(recipe))
    manifest = Path(work) / "input-manifest.json"
    with _python_engine_pinned():
        author_input_manifest(
            manifest,
            mapping_path=recipe["mapping"],
            composition_path=recipe["composition"],
            primary_files=recipe["primary"],
            supplement_files=recipe["supplements"],
            provenance_files=recipe["provenance"],
            # The donor mappings, exactly as `decode_composed_source`
            # receives them below: authoring computes its decoder
            # contract from the format UNION they declare, so leaving
            # them out under-seals any cross-format donor's decoder
            # (grib2-only donors produce byte-identical manifests with
            # or without this argument).
            contributing_mappings=recipe["contributing"] or None,
            **tools)
        digest = _sha256(manifest)
        try:
            bundle = decode_composed_source(
                recipe["composition"], recipe["mapping"], recipe["primary"],
                recipe["supplements"], recipe["provenance"],
                input_manifest=manifest, input_manifest_sha256=digest,
                contributing_mappings=recipe["contributing"] or None,
                **tools)
        except (KeyError, TypeError, ValueError, FileNotFoundError) as error:
            return recipe, None, error
    return recipe, bundle, None


def compose_digest(recipe, bundle) -> dict[str, object]:
    """Reduce a composed decode to the numbers parity is judged on.

    Everything the decode digest carries per frame, PLUS the two evidence
    products only the composition byte work can produce: the alignment
    receipt (which coordinate subset, which clock rule, which times were
    matched and which broadcast) and the per-binding contributing-source
    records.  Those two are the whole point of `compose`; a golden that
    compared frames alone would pass an engine that borrowed the right
    numbers by the wrong rule and recorded nothing about it.

    Each frame's header is digested in its portable form
    (:func:`composed_header_digest`), so a golden reproduces wherever the
    bytes are staged.
    """

    source = str(recipe["source"])
    paths = _composed_path_strings(recipe)
    frames = bundle.frames
    return machine_independent(source, {
        "schema": COMPOSE_DIGEST_SCHEMA,
        "source": source,
        "verdict": "COMPOSED",
        "header_rule": PORTABLE_HEADER_RULE,
        "mapping": {"name": recipe["mapping"].name,
                    "sha256": bundle.mapping_sha256},
        "composition": {"name": recipe["composition"].name,
                        "sha256": bundle.composition_sha256},
        "frame_count": len(frames),
        "frames": [
            {
                "valid_time": frame.valid_time.isoformat(),
                "member": frame.member,
                "source_cycle": frame.source_cycle.isoformat(),
                "vertical_kind": frame.vertical_kind,
                "vertical_units": frame.vertical_units,
                "grid_fingerprint": frame.grid_fingerprint,
                "latitude": _axis_digest(frame.latitude),
                "longitude": _axis_digest(frame.longitude),
                "vertical": _axis_digest(frame.vertical_values),
                "inputs": sorted(
                    ({"name": Path(path).name, "sha256": value}
                     for path, value in frame.input_sha256.items()),
                    key=lambda row: (row["name"], row["sha256"]),
                ),
                "portable_header_sha256": composed_header_digest(
                    recipe, frame.header),
                "fields": {
                    name: {
                        "units": field.units,
                        "axes": list(field.axes),
                        "location": field.location,
                        "staggering": field.staggering,
                        "shape": [int(size) for size in field.values.shape],
                        "missing_count": int(field.missing_count),
                        "sha256": _array_sha256(field.values),
                    }
                    for name, field in sorted(frame.fields.items())
                },
            }
            for frame in frames
        ],
        "alignment_receipt": dict(bundle.alignment_receipt),
        "contributing_sources": [
            dict(record) for record in bundle.contributing_sources],
        "soil_layers": dict(bundle.soil_layer_contract),
        "terrain_data": sorted(
            ({"name": Path(path).name, "sha256": value}
             for path, value in zip(bundle.terrain_data_paths,
                                    bundle.terrain_data_sha256)),
            key=lambda row: (row["name"], row["sha256"]),
        ),
        "terrain_provenance": {
            "name": Path(bundle.terrain_provenance_path).name,
            "sha256": bundle.terrain_provenance_sha256,
        },
    }, paths=paths)


def compose_refusal_digest(recipe, error) -> dict[str, object]:
    """A composed refusal is an answer, so it gets a digest too."""

    source = str(recipe["source"])
    return machine_independent(source, {
        "schema": COMPOSE_DIGEST_SCHEMA,
        "source": source,
        "verdict": "REFUSED",
        "mapping": {"name": recipe["mapping"].name,
                    "sha256": _sha256(recipe["mapping"])},
        "composition": {"name": recipe["composition"].name,
                        "sha256": _sha256(recipe["composition"])},
        "refusal": {"type": type(error).__name__, "message": str(error)},
    }, paths=_composed_path_strings(recipe))


def measure_compose_digest(source: str, work: Path) -> dict[str, object]:
    """The Python engine's COMPOSED answer for one source, measured HERE.

    One function for two callers -- the extractor, which commits it as a
    golden, and the row that falls back to it as a live reference off the
    measuring platform.  ``work`` receives the sealed input manifest, and
    the two engines get separate directories when both are measured, so
    neither can read the other's.
    """

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    recipe, bundle, refusal = compose_with_python_engine(source, Path(work))
    digest = (compose_refusal_digest(recipe, refusal) if bundle is None
              else compose_digest(recipe, bundle))
    del bundle
    return digest


def compose_golden_document(source: str) -> dict[str, object]:
    """A committed compose golden: the measurement, and whose it is."""

    import tempfile

    with tempfile.TemporaryDirectory(prefix="gpuwm-compose-golden-") as work:
        return {**measure_compose_digest(source, Path(work)),
                GOLDEN_PLATFORM_KEY: measuring_platform()}


def compose_golden_path(source: str) -> Path:
    return COMPOSE_GOLDENS / f"{source}.json"


def compose_engine_available_for(source: str, monkeypatch) -> tuple[bool, str]:
    """Is there a second engine to compare this composed source?

    The compose twin of :func:`engine_available_for`: same two measured
    conditions, asked of the compose route's own format and with the
    compose route asserted before the gap is conceded.
    """

    source_format = str(composed_recipe(source)["format"])
    if not engine_bridge.engine_supports("compose", source_format):
        _unported_compose_route_is_pinned(source, monkeypatch)
        return False, (
            f"compose for format {source_format!r} is a declared engine gap "
            "(ENGINE_GAPS); a bare prep routes to the Python engine -- the "
            "route was just asserted, and there is no second engine to "
            "compare until the gap closes")
    implemented, reason = engine_implements("compose")
    if not implemented:
        return False, (
            "gpuwm_mapped_engine does not implement `compose` yet, so no "
            "Rust answer exists to compare -- this is NOT a passing "
            f"comparison: {reason}")
    return True, ""


def assert_engine_compose_matches(source: str, recipe, expected, work: Path,
                                  monkeypatch, note: str) -> None:
    """Compose through the engine and hold it to ``expected``, exactly.

    Run through ``decode_composed_source`` with NO tool flags and the
    engine requested, which is the REAL front-door route
    (``_compose_through_engine``), rather than by driving the exe and
    rebuilding a bundle by hand -- the bundle assembly, the frames'
    mapping-hash check and the "did it compose the declared bindings"
    check are part of what has to work.
    """

    from woof.mapped_authoring import author_input_manifest
    from woof.mapped_composition import decode_composed_source

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(engine_bridge.ENGINE_ENV, engine_bridge.ENGINE_RUST)
    manifest = work / "input-manifest.json"
    author_input_manifest(
        manifest,
        mapping_path=recipe["mapping"], composition_path=recipe["composition"],
        primary_files=recipe["primary"],
        supplement_files=recipe["supplements"],
        provenance_files=recipe["provenance"])
    digest = _sha256(manifest)

    if expected["verdict"] == "REFUSED":
        with pytest.raises(Exception) as caught:     # noqa: PT011 - typed below
            decode_composed_source(
                recipe["composition"], recipe["mapping"], recipe["primary"],
                recipe["supplements"], recipe["provenance"],
                input_manifest=manifest, input_manifest_sha256=digest,
                contributing_mappings=recipe["contributing"] or None)
        assert type(caught.value).__name__ == expected["refusal"]["type"], (
            "the engines disagree about the TYPE of refusal this composed "
            "source earns, so a caller catching one would not catch the "
            "other.  " + note)
        assert machine_independent(
            source, str(caught.value),
            paths=_composed_path_strings(recipe),
        ).startswith(expected["refusal"]["message"]), (
            "the engines refuse this composition for the same reason and say "
            "it differently; the message is what a reader acts on.  " + note)
        return

    bundle = decode_composed_source(
        recipe["composition"], recipe["mapping"], recipe["primary"],
        recipe["supplements"], recipe["provenance"],
        input_manifest=manifest, input_manifest_sha256=digest,
        contributing_mappings=recipe["contributing"] or None)
    observed = compose_digest(recipe, bundle)
    del bundle
    assert canonical(observed) == canonical(expected), (
        "the Rust engine's composition differs from the Python engine's on "
        f"{source}; an unexplained diff is FAIL -- name the defective side "
        "and its evidence in a commit before calling this expected.  " + note)


def compose_requires(source: str) -> None:
    if not _composed_staged(source):
        pytest.skip(f"{source}: staged composed real-byte samples absent")
    recipe = composed_recipe(source)
    for name in _FORMAT_TOOLS[str(recipe["format"])]:
        if bridges.find_bridge(name) is None:
            pytest.skip(f"staged {name} decoder executable absent")


# --------------------------------------------------------------------
# 1. The seam format carries the Python engine's own answer, exactly.
# --------------------------------------------------------------------

@pytest.mark.parametrize("source", sorted(STAGED_SOURCES))
def test_the_frameset_codec_round_trips_a_real_decode(source, tmp_path):
    """gpuwm-mapped-frameset-v1 loses nothing on real staged bytes.

    Named breakage: if the codec perturbed one array, one axis or one
    header member, every engine comparison built on it would blame the
    Rust engine for the format's error -- and a preparation reading a
    frameset would silently initialize from numbers the decoder never
    produced.
    """

    requires(source)
    _, frames, refusal = python_decode(source)
    release_python_answers(source)
    if frames is None:
        pytest.skip(
            f"{source}: the Python engine refuses this input "
            f"({type(refusal).__name__}), so there is no frameset to "
            "round trip; the refusal itself is covered by the golden test")

    directory = engine_bridge.write_frameset(tmp_path / source, frames)
    restored = engine_bridge.read_frameset(directory)

    assert len(restored) == len(frames)
    for original, copy in zip(frames, restored):
        assert copy.valid_time == original.valid_time
        assert copy.member == original.member
        assert copy.source_cycle == original.source_cycle
        assert copy.vertical_kind == original.vertical_kind
        assert copy.vertical_units == original.vertical_units
        assert copy.grid_fingerprint == original.grid_fingerprint
        assert copy.mapping_sha256 == original.mapping_sha256
        assert dict(copy.input_sha256) == dict(original.input_sha256)
        assert copy.header == original.header
        for axis in ("latitude", "longitude", "vertical_values"):
            assert np.array_equal(getattr(copy, axis), getattr(original, axis))
        assert set(copy.fields) == set(original.fields)
        for name, field in original.fields.items():
            other = copy.fields[name]
            assert other.units == field.units
            assert other.axes == field.axes
            assert other.location == field.location
            assert other.staggering == field.staggering
            assert other.missing_count == field.missing_count
            assert np.array_equal(other.values, field.values, equal_nan=True)
            assert _array_sha256(other.values) == _array_sha256(field.values)


# --------------------------------------------------------------------
# 1b. Every byte of a frameset stream is still checked.
# --------------------------------------------------------------------
#
# The reader used to hash the whole stream AND then hash every field --
# two full passes for one answer, and on a 7.5 GiB frameset the first
# pass alone cost 4.7 s of a 12.3 s read.  It now hashes the whole
# stream only when the fields do NOT tile it, because when they do the
# per-field digests cover every byte.  These two tests are that
# argument, made executable: one for each half.
#
# They hand-write the frameset document rather than decoding staged
# bytes.  Both refusals fire before a single frame is rebuilt, so the
# rest of the document is not needed -- and this way the coverage
# argument is checked on every checkout, not only where the staging
# tree exists.

def _hand_written_frameset(directory: Path, arrays, *, trailing=b"") -> Path:
    """A frameset carrying ``arrays``, plus optional unclaimed bytes."""

    directory.mkdir(parents=True, exist_ok=True)
    payload = b""
    fields = []
    for index, array in enumerate(arrays):
        array = np.ascontiguousarray(array, dtype=np.float64)
        blob = array.tobytes()
        fields.append({
            "name": f"field{index}",
            "units": "1",
            "axes": ["y", "x"],
            "location": "mass",
            "staggering": "none",
            "shape": list(array.shape),
            "dtype": engine_bridge.STREAM_DTYPE,
            "offset": len(payload),
            "length": len(blob),
            "sha256": _array_sha256(array),
            "missing_count": 0,
            "source_references": [],
        })
        payload += blob
    payload += trailing
    (directory / engine_bridge.FRAMES_STREAM).write_bytes(payload)
    (directory / engine_bridge.FRAMES_DOCUMENT).write_text(json.dumps({
        "schema": engine_bridge.FRAMESET_SCHEMA,
        "stream": {
            "path": engine_bridge.FRAMES_STREAM,
            "dtype": engine_bridge.STREAM_DTYPE,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "frames": [{"fields": fields}],
    }), encoding="utf-8")
    return directory


def test_a_perturbed_field_array_is_refused_by_name(tmp_path):
    """One flipped bit inside a field is caught, and the field is named.

    Named breakage: a frameset whose arrays are not the numbers its
    manifest describes would initialize a run from values no decoder
    produced.  This is the check that covers the stream's bytes now that
    the whole-stream digest is not re-computed for a tiled frameset, so
    it is proven on a stream the whole-stream digest can no longer save.
    """

    directory = _hand_written_frameset(
        tmp_path / "perturbed",
        [np.arange(12.0).reshape(3, 4), np.full((3, 4), 2.5)])
    stream = directory / engine_bridge.FRAMES_STREAM
    raw = bytearray(stream.read_bytes())
    raw[-1] ^= 0xFF
    stream.write_bytes(bytes(raw))

    document = json.loads(
        (directory / engine_bridge.FRAMES_DOCUMENT).read_text("utf-8"))
    assert engine_bridge._fields_tile_stream(
        document, int(document["stream"]["bytes"])), \
        "the case is only meaningful on a stream the whole-stream " \
        "digest is skipped for"
    with pytest.raises(ValueError, match=r"field 'field1' hashes to"):
        engine_bridge.read_frameset(directory)


def test_stream_bytes_no_field_claims_are_refused(tmp_path):
    """Bytes outside every field extent are still hashed.

    Named breakage: per-field digests only look at bytes some field
    claims, so a stream with padding, a gap, or a truncated last field
    could carry unchecked content between the arrays.  A frameset whose
    fields do not tile its stream therefore keeps the whole-stream
    digest, and this is the case that proves the fallback runs.
    """

    directory = _hand_written_frameset(
        tmp_path / "padded",
        [np.arange(12.0).reshape(3, 4)],
        trailing=b"\x00" * 8)
    stream = directory / engine_bridge.FRAMES_STREAM
    raw = bytearray(stream.read_bytes())
    raw[-1] = 0x7F
    stream.write_bytes(bytes(raw))

    document = json.loads(
        (directory / engine_bridge.FRAMES_DOCUMENT).read_text("utf-8"))
    assert not engine_bridge._fields_tile_stream(
        document, int(document["stream"]["bytes"]))
    # The path, not just "hashes to": the per-field refusal reads the
    # same way, and this test must fail if the fallback stopped running.
    with pytest.raises(ValueError, match=r"frames\.f64 hashes to"):
        engine_bridge.read_frameset(directory)


# --------------------------------------------------------------------
# 1c. Every committed golden declares whose measurement it is.
# --------------------------------------------------------------------
#
# These three run on every checkout, staged bytes or not: they read the
# committed files and the scope rule, never the agency bytes.  They are
# what stops the platform scope from becoming the quiet way out.

def _committed_goldens() -> list[Path]:
    return sorted([*GOLDENS.glob("*.json"), *COMPOSE_GOLDENS.glob("*.json")])


@pytest.mark.parametrize(
    "path", _committed_goldens(), ids=lambda path: f"{path.parent.name}/{path.stem}")
def test_every_committed_golden_declares_its_measuring_platform(path):
    """A golden that does not say whose numbers it carries is unusable.

    Named breakage: the derived field arrays (grid-relative wind rotation,
    the relative-humidity derivation) depend on the platform's libm,
    measured field by field on Linux against these Windows-measured files.
    An unstamped golden cannot be scoped, so it is either compared across
    platforms and reads as an engine defect, or excused everywhere and
    stops being a gate. Neither may be a default, so the stamp is required.
    """

    _, stamp = load_golden(path)
    assert stamp["system"] and stamp["machine"], (
        f"{path.name}: {GOLDEN_PLATFORM_KEY} must name both the system and "
        "the machine architecture that measured it")


def test_a_golden_with_no_platform_stamp_is_refused_by_name(tmp_path):
    """The requirement above is enforced by :func:`load_golden`, not by
    the extractor alone -- a hand-edited or hand-copied golden reaches
    the battery without ever passing through the tool."""

    path = tmp_path / "unstamped.json"
    path.write_text(json.dumps({"schema": DIGEST_SCHEMA, "source": "x"}),
                    encoding="utf-8")
    with pytest.raises(AssertionError, match="does not declare the platform"):
        load_golden(path)
    # A stamp that is present but empty is the same defect, not a pass.
    path.write_text(json.dumps({"schema": DIGEST_SCHEMA, "source": "x",
                                GOLDEN_PLATFORM_KEY: {}}), encoding="utf-8")
    with pytest.raises(AssertionError, match="does not declare the platform"):
        load_golden(path)


def test_the_battery_carries_committed_goldens_to_scope():
    """The stamp check above must not be vacuous.

    Named breakage: `_committed_goldens` is read at collection time, so
    an empty golden directory would parametrize zero rows and the whole
    contract would report green having checked nothing.
    """

    decode = sorted(GOLDENS.glob("*.json"))
    compose = sorted(COMPOSE_GOLDENS.glob("*.json"))
    assert decode, f"no committed decode goldens under {GOLDENS}"
    assert compose, f"no committed compose goldens under {COMPOSE_GOLDENS}"
    assert len(_committed_goldens()) == len(decode) + len(compose)


def test_the_platform_scope_is_decided_by_the_stamp_and_nothing_else():
    """Native means the stamp equals this box; everything else is foreign.

    Named breakage: a scope rule that read an environment variable, a
    hostname or "is this Windows" would let a box declare itself the
    measuring platform and compare against numbers it never produced.
    The only input is the golden's own stamp against this interpreter's
    own platform.
    """

    here = measuring_platform()
    assert set(here) == {"system", "machine"}
    assert golden_is_native(here)
    assert not golden_is_native({**here, "system": here["system"] + "-other"})
    assert not golden_is_native({**here, "machine": here["machine"] + "64"})
    # And the fallback sentence names the open item, so a reader lands on
    # the owner of the portability work rather than on a rediscovery.
    note = foreign_golden_note("some-source", {"system": "Other",
                                               "machine": "arch"}, "decode")
    assert "GOLD-PORTABLE" in note and "#157" in note
    assert "LIVE dual-engine comparison" in note


# --------------------------------------------------------------------
# 2. The goldens are fixed.
# --------------------------------------------------------------------

@pytest.mark.parametrize("source", sorted(STAGED_SOURCES))
def test_the_python_engine_still_answers_its_golden(source, monkeypatch):
    """The reference has not moved under the port.

    This is the half of parity that can regress without any Rust at all:
    a change to the Python engine that shifts one array shifts the
    reference the Rust engine is being built against, and every later
    comparison would then be measuring the wrong thing.

    That question is only answerable against the committed numbers on
    the platform that measured them.  Off it, the golden's derived
    fields carry the measuring box's libm and this row would report
    drift that is not drift -- so it runs the LIVE dual-engine
    comparison instead, at the same strictness, and says so.  It never
    skips: a committed golden that quietly stopped being compared is the
    gate going silent.
    """

    requires(source)
    path = golden_path(source)
    if not path.is_file():
        pytest.fail(
            f"no golden for {source}; extract it from the real Python "
            "engine on the real staged bytes with "
            "`python tools/extract_mapped_engine_goldens.py "
            f"--source {source}`")
    expected, stamp = load_golden(path)

    if not golden_is_native(stamp):
        note = foreign_golden_note(source, stamp, "decode")
        announce(note)
        entry = STAGED_SOURCES[source]
        for subcommand in ("decode", "inspect"):
            available, reason = engine_available_for(
                source, subcommand, monkeypatch)
            if not available:
                pytest.skip(f"{note}  That comparison cannot run here: "
                            f"{reason}")
        reference = measure_decode_document(source)
        assert_engine_decode_matches(
            source, entry, _mapping_of(entry), reference["decode"], note)
        assert_engine_inspection_matches(
            source, entry, reference["inspect"], note)
        return

    announce(native_golden_note(source, stamp, "decode"))
    observed = measure_decode_document(source)
    assert canonical(observed) == canonical(expected)


# --------------------------------------------------------------------
# 3. The engines agree -- or the battery says which subcommand is absent.
# --------------------------------------------------------------------

def engine_implements(subcommand: str) -> tuple[bool, str]:
    """Does the resolved engine implement this subcommand yet?

    Probed by RUNNING it (verify against the artifact), on arguments
    that are complete enough to reach the subcommand and be refused for
    a reason of the subcommand's own choosing.  A ``not_implemented``
    class is the vendored skeleton saying so in the contract's own
    words; anything else means the subcommand exists and is refusing
    these particular arguments.
    """

    try:
        binary = engine_bridge.require_engine()
    except FileNotFoundError as error:
        return False, str(error)
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="gpuwm-engine-probe-") as work:
        listing = Path(work) / "inputs.txt"
        listing.write_text("", encoding="utf-8")
        completed = subprocess.run(
            [str(binary), subcommand, "--mapping",
             str(Path(work) / "absent.json"), "--input-list", str(listing),
             "--output", str(Path(work) / "out")],
            capture_output=True, text=True, check=False,
        )
    refusal = engine_bridge.parse_refusal(completed.stderr or "")
    if refusal is not None and refusal["class"] == "not_implemented":
        return False, refusal["message"]
    return True, ""


def _unported_format_route_is_pinned(source: str, monkeypatch) -> None:
    """Before an unported-format row is allowed to skip the engine
    comparison, the ROUTE for that format is asserted: a bare run goes to
    the Python engine, an explicit rust request still reaches the engine.
    The skip then states a measured fact, not an assumption."""

    from woof.mapped_source import _mapped_engine_choice

    entry = STAGED_SOURCES[source]
    source_format = _format_of(entry)
    monkeypatch.delenv(engine_bridge.ENGINE_ENV, raising=False)
    assert _mapped_engine_choice(
        grib1_bridge=None, grib2_inventory=None, grib2_dump=None,
        subcommand="decode", source_format=source_format,
    ) == engine_bridge.ENGINE_PYTHON, (
        f"{source_format} is not in ENGINE_CAPABILITIES['decode'] and a "
        "bare run did NOT route to the Python engine; a default run of "
        "this source would refuse where it used to work")
    monkeypatch.setenv(engine_bridge.ENGINE_ENV, engine_bridge.ENGINE_RUST)
    assert _mapped_engine_choice(
        grib1_bridge=None, grib2_inventory=None, grib2_dump=None,
        subcommand="decode", source_format=source_format,
    ) == engine_bridge.ENGINE_RUST, (
        "an explicit rust request must reach the engine and earn its own "
        "not_implemented refusal, or a caller could believe a Rust decode "
        "happened when it never did")
    monkeypatch.delenv(engine_bridge.ENGINE_ENV, raising=False)


@pytest.mark.parametrize("source", sorted(STAGED_SOURCES))
def test_the_rust_engine_reproduces_the_golden(source, monkeypatch):
    """The gate itself: the engine's answer equals the reference, bit for bit.

    The reference is the committed golden on the platform that measured
    it, and the Python engine's LIVE answer on this box anywhere else --
    same digest, same fields, same "an unexplained diff is FAIL".  Only
    the reference moves, and the row says which one it used.
    """

    requires(source)
    entry = STAGED_SOURCES[source]
    # `engine_available_for` asks `engine_supports` rather than testing
    # membership; see the compose gate for the empty-frozenset trap the
    # membership idiom carried.
    available, reason = engine_available_for(source, "decode", monkeypatch)
    if not available:
        pytest.skip(f"{source}: {reason}")

    path = golden_path(source)
    assert path.is_file(), f"no golden for {source}"
    golden, stamp = load_golden(path)
    if golden_is_native(stamp):
        note = native_golden_note(source, stamp, "decode")
        expected = golden["decode"]
    else:
        note = foreign_golden_note(source, stamp, "decode")
        expected = measure_decode_digest(source)
    announce(note)

    assert_engine_decode_matches(
        source, entry, _mapping_of(entry), expected, note)


@pytest.mark.parametrize("source", sorted(STAGED_SOURCES))
def test_the_rust_engine_reproduces_the_inspection_golden(source, monkeypatch):
    """Every decoded array, on every source, including the refusing ones.

    Inspection reports the SHA-256 of each directly decoded array before
    materialization has a verdict, so this is where the seven staged
    sources that cannot materialize alone still contribute byte-level
    coverage.  Without it a Rust engine could decode every array wrong
    and pass by refusing with the right sentence.
    """

    requires(source)
    entry = STAGED_SOURCES[source]
    available, reason = engine_available_for(source, "inspect", monkeypatch)
    if not available:
        pytest.skip(f"{source}: {reason}")

    path = golden_path(source)
    assert path.is_file(), f"no golden for {source}"
    golden, stamp = load_golden(path)
    if golden_is_native(stamp):
        note = native_golden_note(source, stamp, "inspection")
        expected = golden["inspect"]
    else:
        note = foreign_golden_note(source, stamp, "inspection")
        expected = measure_inspection_digest(source)
    announce(note)

    assert_engine_inspection_matches(source, entry, expected, note)


# --------------------------------------------------------------------
# The refusal battery.
# --------------------------------------------------------------------

def test_every_refusal_class_maps_to_a_python_exception_type():
    """The class table is the contract, and it is complete.

    Named breakage: an engine refusal whose class is absent from this
    table reaches a caller as a bare ``RuntimeError`` where the Python
    engine raised, say, ``FileNotFoundError`` -- so a caller's ``except``
    stops matching and a recoverable condition becomes a crash.
    """

    expected = {
        "usage", "not_implemented", "missing_input", "mapping_invalid",
        "manifest_mismatch", "selector_unmatched", "grid_mismatch",
        "decode_failed", "frame_invalid", "forcing_series",
        "authority_moved", "disk_full", "write_failed", "requester_closed",
        "host_memory",
    }
    assert set(engine_bridge.REFUSAL_CLASSES) == expected
    for name, exception in engine_bridge.REFUSAL_CLASSES.items():
        assert isinstance(exception, type) and issubclass(exception, Exception)


@pytest.mark.parametrize("name", sorted(engine_bridge.REFUSAL_CLASSES))
def test_a_refusal_object_becomes_its_declared_exception(name):
    """class -> exception type, message and remedy both carried through."""

    error = engine_bridge.refusal_error(
        {"class": name, "message": "the named breakage",
         "remedy": "the named remedy"},
        ["gpuwm_mapped_engine", "decode"],
    )
    assert type(error) is engine_bridge.REFUSAL_CLASSES[name]
    assert "the named breakage" in str(error)
    assert "the named remedy" in str(error)


def test_a_refusal_maps_without_a_command_line():
    """The capability query maps its refusal with no command in hand."""

    error = engine_bridge.refusal_error(
        {"class": "usage", "message": "the named breakage",
         "remedy": "the named remedy"})
    assert type(error) is ValueError
    assert "the named breakage" in str(error)


def test_an_unknown_refusal_class_is_itself_a_defect():
    """Never widened into the nearest-looking exception."""

    error = engine_bridge.refusal_error(
        {"class": "invented", "message": "m", "remedy": "r"}, [])
    assert type(error) is RuntimeError
    assert "invented" in str(error)
    assert "different contracts" in str(error)


def test_a_nonzero_exit_without_a_refusal_object_is_named_as_such():
    """A crash is not a refusal, and must not be reported as one."""

    assert engine_bridge.parse_refusal("boom\n") is None
    assert engine_bridge.parse_refusal("") is None
    assert engine_bridge.parse_refusal('{"schema":"other"}') is None


def test_the_real_engine_refusal_reaches_python_as_its_declared_type():
    """End to end against the ARTIFACT: run the exe, catch what it earns.

    The one test here that proves the whole refusal path -- ladder
    resolution, ABI handshake, subprocess launch, last-stderr-line
    parse, class lookup -- on the binary a user would run, rather than
    on a hand-written refusal string.
    """

    try:
        engine_bridge.require_engine()
    except FileNotFoundError as error:
        pytest.skip(f"engine not built in this checkout: {error}")

    import tempfile

    with tempfile.TemporaryDirectory() as work:
        with pytest.raises(Exception) as caught:      # noqa: PT011 - typed below
            engine_bridge.run_engine(
                "decode", mapping=Path(work) / "m.json",
                files=[Path(work) / "a.grb2"],
                output=Path(work) / "out")
    observed = type(caught.value)
    assert observed in set(engine_bridge.REFUSAL_CLASSES.values()), (
        f"the engine refused with something outside the class table: "
        f"{observed.__name__}: {caught.value}")


def _public_decode_both_routes(monkeypatch, mapping, files):
    """The PUBLIC entry under each engine request; refusals compared."""

    from woof.mapped_source import decode_mapped_source

    answers = {}
    for engine in (engine_bridge.ENGINE_PYTHON, engine_bridge.ENGINE_RUST):
        monkeypatch.setenv(engine_bridge.ENGINE_ENV, engine)
        try:
            decode_mapped_source(mapping, files)
        except Exception as error:  # noqa: BLE001 - the refusal IS the answer
            answers[engine] = (type(error), str(error))
        else:
            answers[engine] = (None, "")
    monkeypatch.delenv(engine_bridge.ENGINE_ENV, raising=False)
    return answers


def test_a_broken_mapping_grammar_refuses_identically_on_both_routes(
        monkeypatch, tmp_path):
    """Same document defect, same sentence, whichever engine decodes.

    Named breakage: the Rust route used to hand grammar validation to
    the engine alone, so one broken mapping earned two different
    explanations -- `mapping is missing required key(s): ['fields']` on
    the Python engine, `mapping.fields must be an object` on the Rust
    one -- and a reader chasing the second sentence would fix a type
    problem the document does not have.  Both routes now run the ONE
    Python validator first; the engine's own validator remains behind it
    for hand-run invocations of the exe.
    """

    try:
        engine_bridge.require_engine()
    except FileNotFoundError as error:
        pytest.skip(f"engine not built in this checkout: {error}")
    entry = STAGED_SOURCES["rap-awip32"]
    if not _staged(entry):
        pytest.skip("staged real bytes absent")
    document = json.loads(_mapping_of(entry).read_text(encoding="utf-8"))
    document.pop("fields", None)
    broken = tmp_path / "broken-grammar.mapping.json"
    broken.write_text(json.dumps(document), encoding="utf-8")

    answers = _public_decode_both_routes(
        monkeypatch, broken, entry["files"])
    python_answer = answers[engine_bridge.ENGINE_PYTHON]
    rust_answer = answers[engine_bridge.ENGINE_RUST]
    assert python_answer[0] is not None, "the broken mapping decoded"
    assert rust_answer[0] is python_answer[0]
    assert rust_answer[1] == python_answer[1], (
        "the two routes explain the same grammar defect differently")


def test_corrupt_grib_bytes_refuse_identically_on_both_routes(
        monkeypatch, tmp_path):
    """Truncated real bytes earn the same exception class either way.

    Named breakage: the Python engine's subprocess wrapper used to raise
    RuntimeError for ANY nonzero decoder-tool exit, so undecodable BYTES
    surfaced as a tool failure -- while the Rust engine's `decode_failed`
    class maps to ValueError.  A caller's `except ValueError` around a
    fetch-then-prepare loop would retry the fetch on one engine and
    crash on the other.  Bytes that produce a decoder diagnostic are a
    decode refusal (ValueError) on both routes; a tool that dies without
    saying anything is still a RuntimeError, because that one is about
    the installation, not the bytes.
    """

    try:
        engine_bridge.require_engine()
    except FileNotFoundError as error:
        pytest.skip(f"engine not built in this checkout: {error}")
    entry = STAGED_SOURCES["rap-awip32"]
    if not _staged(entry):
        pytest.skip("staged real bytes absent")
    payload = Path(entry["files"][0]).read_bytes()
    truncated = tmp_path / "truncated.grib2"
    truncated.write_bytes(payload[: max(64, len(payload) // 7)])

    answers = _public_decode_both_routes(
        monkeypatch, _mapping_of(entry),
        (truncated, *entry["files"][1:]))
    python_answer = answers[engine_bridge.ENGINE_PYTHON]
    rust_answer = answers[engine_bridge.ENGINE_RUST]
    assert python_answer[0] is ValueError, (
        f"undecodable bytes must refuse as a decode refusal, got "
        f"{python_answer[0]}: {python_answer[1][:200]}")
    assert rust_answer[0] is ValueError


def test_the_engine_command_line_is_the_contract():
    """The argv a route runs, readable without running it."""

    command = engine_bridge.engine_command(
        "compose",
        engine=Path("engine.exe"), mapping=Path("m.json"),
        input_list=Path("inputs.txt"), output=Path("out"),
        composition=Path("c.json"),
        supplements={"role_b": (Path("b1"), Path("b2")),
                     "role_a": (Path("a1"),)},
        provenance={"prov": Path("p.md")},
        contributing_mappings={"donor": Path("d.json")},
        input_manifest=Path("mf.json"), input_manifest_sha256="deadbeef",
    )
    assert command[:2] == ["engine.exe", "compose"]
    # Roles are emitted in sorted order so one composition always
    # produces one command line; an unordered argv would make the
    # engine's own receipts unstable between runs.
    assert command.index("--supplement") < command.index("--provenance")
    assert "role_a=a1" in command and "role_b=b1" in command
    assert "role_b=b2" in command
    assert command.index("role_a=a1") < command.index("role_b=b1")
    assert "donor=d.json" in command
    assert command[-2:] == ["--input-manifest-sha256", "deadbeef"]


def test_an_unknown_subcommand_never_reaches_the_engine():
    with pytest.raises(ValueError, match="decode, compose and inspect"):
        engine_bridge.engine_command(
            "render", engine=Path("e"), mapping=Path("m"),
            input_list=Path("i"), output=Path("o"))


# --------------------------------------------------------------------
# Engine selection, and the one constant the default flip moves.
# --------------------------------------------------------------------

def test_the_default_engine_and_its_blocker_agree(monkeypatch):
    """The §5 flip is one edit, and it cannot be a half edit.

    Named breakage: a default flipped to ``rust`` while the blocker text
    still stands would ship a default that refuses every mapped
    preparation, and a blocker left behind a ``rust`` default would tell
    a reader the opposite of what the code does.
    """

    if engine_bridge.DEFAULT_ENGINE == engine_bridge.ENGINE_RUST:
        assert engine_bridge.DEFAULT_ENGINE_BLOCKER is None
    else:
        assert engine_bridge.DEFAULT_ENGINE == engine_bridge.ENGINE_PYTHON
        assert engine_bridge.DEFAULT_ENGINE_BLOCKER, (
            "the Python engine is the default and nothing says why; a "
            "workaround posture must name the breakage that holds it")


def test_the_capability_table_matches_the_built_engine():
    """What the bridge claims the engine can do, the engine must do.

    ``ENGINE_CAPABILITIES`` is what routes a bare run: a subcommand
    listed there goes to Rust, one that is not goes to Python.  If the
    table said the engine implements something it refuses, a default run
    would refuse where it used to work; if it said the engine cannot do
    something it can, the migration would be quietly incomplete and
    nothing would say so.  Both are silent, so the table is checked
    against the ARTIFACT -- the built binary, probed by running it.
    """

    try:
        declared = engine_bridge.declared_capabilities()
    except FileNotFoundError as error:
        pytest.skip(f"the engine is not resolvable here: {error}")
    table = {
        name: sorted(formats or ())
        for name, formats in engine_bridge.ENGINE_CAPABILITIES.items()
    }
    assert {name: sorted(formats) for name, formats in declared.items()} == table


def test_every_capability_gap_is_written_down_for_a_reader():
    """A gap the table knows about and the prose does not is a surprise."""

    gaps = " ".join(engine_bridge.ENGINE_GAPS).lower()
    for subcommand, formats in engine_bridge.ENGINE_CAPABILITIES.items():
        if formats is not None and not formats:
            assert subcommand in gaps, (
                f"{subcommand!r} routes to the Python engine and "
                "ENGINE_GAPS does not say so")
    # Every mapped source format the router knows, not a written-down
    # subset: a subset silently stops covering a format the product gains.
    for source_format in _MAPPED_SOURCE_FORMATS:
        if source_format not in (
                engine_bridge.ENGINE_CAPABILITIES["decode"] or ()):
            assert source_format in gaps, (
                f"{source_format} mappings route to the Python engine and "
                "ENGINE_GAPS does not say so")


def test_a_bare_run_of_an_unported_format_routes_without_pretending(
        monkeypatch, tmp_path):
    """Each format routes where the table says, and asking for Rust reaches it.

    Named breakage: routing an unported format to Python is only
    defensible while an EXPLICIT ``--mapped-engine rust`` still reaches
    the engine and earns its own ``not_implemented`` refusal.  If the
    route swallowed the explicit request too, a caller could believe a
    Rust decode happened when it never did.

    Driven by ``woof.mapped_source``'s own format list crossed with
    :data:`ENGINE_CAPABILITIES`, rather than by a written-down list of
    formats, because a written-down list goes stale the moment a lane
    closes a gap: this test named ``grib1`` and ``netcdf`` as unported
    and both landed on the same night, and each edit could just as
    easily have been "delete the row" instead of "move the row".
    Reading the table asserts the routing property for EVERY format in
    BOTH directions -- an unported format must route to Python, and a
    format the table declares must be reached by a BARE run, or the port
    is engine-proven and not shipped -- so it cannot be quietly narrowed
    and does not go vacuous now that the last decode gap has closed.
    """

    from woof.mapped_source import _mapped_engine_choice

    ported = engine_bridge.ENGINE_CAPABILITIES["decode"] or frozenset()
    bare = dict(grib1_bridge=None, grib2_inventory=None, grib2_dump=None)
    assert _MAPPED_SOURCE_FORMATS, "no mapped source formats to route"

    monkeypatch.delenv(engine_bridge.ENGINE_ENV, raising=False)
    for source_format in _MAPPED_SOURCE_FORMATS:
        expected = (
            engine_bridge.DEFAULT_ENGINE if source_format in ported
            else engine_bridge.ENGINE_PYTHON
        )
        assert _mapped_engine_choice(
            subcommand="decode", source_format=source_format, **bare,
        ) == expected, (
            f"a bare run of {source_format!r} must route to {expected!r}: "
            f"it is {'declared in' if source_format in ported else 'absent from'} "
            "ENGINE_CAPABILITIES['decode']")

    # The other half, and the one that makes the routing defensible: an
    # explicit request reaches the engine for EVERY format, so an
    # unported one returns its own `not_implemented` instead of being
    # silently answered by Python.  Asserted for all formats rather than
    # only the unported ones, so it cannot go vacuous when the last gap
    # closes.
    monkeypatch.setenv(engine_bridge.ENGINE_ENV, "rust")
    for source_format in _MAPPED_SOURCE_FORMATS:
        assert _mapped_engine_choice(
            subcommand="decode", source_format=source_format, **bare,
        ) == engine_bridge.ENGINE_RUST, (
            f"an explicit rust request for {source_format!r} was re-routed; "
            "a caller could believe a Rust decode happened when it did not")


def test_engine_selection_is_explicit_then_environment_then_default(
        monkeypatch):
    monkeypatch.delenv(engine_bridge.ENGINE_ENV, raising=False)
    assert engine_bridge.resolve_engine() == engine_bridge.DEFAULT_ENGINE
    monkeypatch.setenv(engine_bridge.ENGINE_ENV, "python")
    assert engine_bridge.resolve_engine() == "python"
    assert engine_bridge.resolve_engine("rust") == "rust"
    monkeypatch.setenv(engine_bridge.ENGINE_ENV, "RUST")
    assert engine_bridge.resolve_engine() == "rust"


def test_an_unknown_engine_name_refuses_rather_than_falling_back(monkeypatch):
    """A silent fallback would report a Rust run that was a Python one."""

    monkeypatch.setenv(engine_bridge.ENGINE_ENV, "rustic")
    with pytest.raises(ValueError, match="rust, python"):
        engine_bridge.resolve_engine()
    monkeypatch.delenv(engine_bridge.ENGINE_ENV, raising=False)
    with pytest.raises(ValueError, match="--mapped-engine"):
        engine_bridge.resolve_engine("go")


def test_a_pinned_decoder_tool_routes_to_the_python_engine(monkeypatch):
    """Explicit configuration beats the default, in both directions."""

    from woof.mapped_source import _mapped_engine_choice

    monkeypatch.setenv(engine_bridge.ENGINE_ENV, "rust")
    assert _mapped_engine_choice(
        grib1_bridge=None, grib2_inventory=None, grib2_dump=None) == "rust"
    assert _mapped_engine_choice(
        grib1_bridge=None, grib2_inventory="tool", grib2_dump="tool"
    ) == "python", (
        "a pinned tool must win: the Rust engine decodes in process and "
        "would silently ignore the pin")
    assert _mapped_engine_choice(
        grib1_bridge="tool", grib2_inventory=None, grib2_dump=None
    ) == "python"


def test_the_abi_marker_is_registered_and_spelled_once():
    """One literal, in the bridge estate and in the seam module, and it
    carries BOTH contracts a stale engine can break: the frameset shape it
    writes and the Section-5 template set it can read."""

    marker = engine_bridge.ABI_MARKER
    assert bridges.BRIDGE_ABI_MARKERS["gpuwm_mapped_engine"] == marker
    assert engine_bridge.FRAMESET_SCHEMA.encode("ascii") in marker
    assert b"grib2-drt=" in marker
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools/rw_wps/crates/mapped-engine/src/lib.rs").read_text()
    assert marker.decode("ascii") in source, (
        "the Rust half must compile in the same literal the Python half "
        "searches the binary for")
    core = (root / "tools/grib1_bridge/vendor/grib-core/src/grib2/parser.rs"
            ).read_text()
    declared = re.search(
        'pub const DECODE_TEMPLATES: &str = "([^"]+)"', core).group(1)
    assert marker.decode("ascii").endswith(declared), (
        "the marker's template list must be the decoder's own declaration")


def test_the_doctor_estate_reports_the_engine():
    from woof import doctor

    assert "gpuwm_mapped_engine" in doctor._CHECKED_ARTIFACTS
    check = doctor._mapped_engine_check()
    assert check.status in {"verified", "present", "info", "missing"}
    assert engine_bridge.DEFAULT_ENGINE in check.brief


def test_the_front_doors_carry_the_workaround_flag():
    """A workaround with no door is not reachable, and a door that does
    not SAY it is a workaround is a supported mode by accident."""

    from woof import mapped_direct, source_cli

    for build in (source_cli._parser, mapped_direct._parser):
        text = build().format_help()
        assert "--mapped-engine" in text
    prep = source_cli._parser().format_help()
    assert "WORKAROUND" in prep


# --------------------------------------------------------------------
# 7. COMPOSE: the subcommand a mapped preparation actually runs.
# --------------------------------------------------------------------
#
# ``MAPPED_ROUTE_SUBCOMMAND`` is ``compose``, so every registered
# ``mapped_composition_v1`` source reaches its canonical frames through
# the composition byte work, not through ``decode``.  The rows above
# measure the decode UNDER that work; these measure the work itself.

@pytest.mark.parametrize("source", sorted(COMPOSED_SOURCES))
def test_every_composed_source_has_a_committed_golden(source):
    """A staged composed source without a golden is uncovered, loudly.

    Named breakage: the compose path had no golden at all while seventeen
    decode goldens read as thorough coverage, so a composition that
    borrowed the wrong cells, broadcast the wrong record or wrote the
    wrong receipt would have passed the whole battery.
    """

    compose_requires(source)
    path = compose_golden_path(source)
    assert path.is_file(), (
        f"no compose golden for {source}; measure it from the real Python "
        "engine on the real staged bytes with "
        "`python tools/extract_mapped_engine_goldens.py --kind compose "
        f"--source {source}`")


def test_the_compose_registry_covers_every_registered_composition_source():
    """A model added as table data inherits compose coverage.

    The arbitrary acceptance test, applied to this battery: adding ICON,
    AIFS or a Canadian model must not mean writing a new recipe here.
    Every registry row whose runner is ``mapped_composition_v1`` and
    which ships a packaged profile is either covered by a row above or
    named in the exemption below WITH ITS REASON -- never silently
    missing.
    """

    from woof.source_adapters import source_adapters

    #: Registered composition sources with no compose row, and why.
    exempt = {
        # NetCDF-CF primitives: the 20CRv3 NetCDF corpus is not in the
        # staging tree on any box this battery has run on, so a row would
        # be permanently skipped rather than covered.  It is reported by
        # `--kind compose --list` the moment the bytes appear.
        "20crv3-cf": "no staged NetCDF-CF corpus",
        # Historical GEFS products have distinct member/surface archives.
        # This battery's current GEFS corpus contains operational products,
        # not these retained historical pairs. Route tests cover their
        # metadata; no dual-engine historical-array comparison is claimed.
        "gefs-reforecast": "no staged GEFSv12 reforecast member/surface pair",
        "gefs-reforecast-weekly": "no staged extended-lead GEFSv12 reforecast pair",
        "gefs-v11": "no staged GEFSv11 historical member/surface pair",
        "gefs-v11-flat": "no staged flattened GEFSv11 member/surface pair",
        # One perturbed member, cut out of ECMWF's enfo-ef object by its
        # JSON index ranges.  The staging tree holds the oper product only
        # (ifs/*-oper-fc.grib2): no member has been cut and staged on any
        # box this battery has run on, so a row would be permanently
        # skipped rather than covered.  A row belongs here the moment a
        # member's two steps are staged.
        "ecmwf-ens": "no staged ENS member bytes",
        # The 0.1-degree pl+sfc pair decodes through the ecmwf-open-data
        # row's own packaged profile (composed above on the 0.25-degree
        # bytes); no 0.1-degree pair is staged on any box this battery has
        # run on, so a row of its own would be permanently skipped.
        "ecmwf-open-data-0p1": "same packaged profile as ecmwf-open-data; no staged 0.1-degree pair",
        # The one source whose compose inputs are not published bytes: a
        # GDT-101 mesh reaches compose only through the normalization
        # stage, so this row's primary files would be the 350 regional
        # intermediates that stage writes, not anything a staging tree can
        # hold.  Producing them means running the native remapper over 352
        # raw objects first, which is a golden that regenerates its own
        # inputs -- a different shape from every row above, and it needs
        # its own staging contract rather than a literal file list.
        "icon-global": "compose inputs are produced by the normalization "
                       "stage, not published",
        # The same GDT-101 normalization, on DWD's regional mesh.
        "icon-d2": "compose inputs are produced by the normalization "
                   "stage, not published",
        # One RRFS ensemble member composed with its declared companion
        # (the same cycle's deterministic prslev+2dfld pair).  No member or
        # companion bytes are retained in the gauntlet corpus, and
        # the row recipe above hands ONE supplement role, while this
        # composition binds two (the member's terrain and the companion's
        # data), so a row would be permanently skipped rather than covered.
        # Its archived dual-engine evidence is the 2026-10-05 12Z m001 prep on
        # both engines (lane pi-c11b): every composed array identical except
        # the declared libm-dependent 2 m humidity.
        "rrfs-ens": "no staged member+companion bytes; the row recipe "
                    "hands one supplement role and this composition binds "
                    "two",
    }
    # Inference products are generated with a same-cycle donor, rather than
    # published objects in this battery's agency staging corpus. Whole-job
    # receipts cover their composition; no agency golden is claimed here.
    exempt.update({adapter.source_id: "no staged generated forecast and donor corpus"
                   for adapter in source_adapters() if adapter.acquisition_owner})
    exempt["hex-global"] = "no staged exported global forecast history corpus"
    registered = {
        adapter.source_id for adapter in source_adapters()
        if adapter.runner == "mapped_composition_v1" and adapter.runnable
        and adapter.packaged_profile is not None
    }
    uncovered = registered - set(COMPOSED_SOURCES) - set(exempt)
    assert not uncovered, (
        f"registered composition source(s) {sorted(uncovered)} have no "
        "compose golden row and no written-down exemption; a source that "
        "reaches its frames through `compose` and is measured only through "
        "`decode` is measured on work a bare prep never runs")
    stale = set(exempt) & set(COMPOSED_SOURCES)
    assert not stale, f"exempt but covered: {sorted(stale)}"


@pytest.mark.parametrize("source", sorted(COMPOSED_SOURCES))
def test_the_python_engine_still_answers_its_compose_golden(
        source, tmp_path, monkeypatch):
    """The composed reference has not moved under the port.

    The same half of parity the decode battery guards, on the work a
    preparation actually performs: a change to ``mapped_composition``
    that shifts one borrowed cell, one broadcast time or one receipt key
    shifts the reference the Rust engine is being built against, and
    every later comparison would then measure the wrong thing.

    Platform-scoped like its decode twin: off the box that measured the
    golden this row runs the LIVE dual-engine comparison at the same
    strictness instead of reporting a libm difference as drift, and it
    says so rather than skipping.
    """

    compose_requires(source)
    path = compose_golden_path(source)
    if not path.is_file():
        pytest.fail(
            f"no compose golden for {source}; extract it with "
            "`python tools/extract_mapped_engine_goldens.py --kind compose "
            f"--source {source}`")
    expected, stamp = load_golden(path)

    if not golden_is_native(stamp):
        note = foreign_golden_note(source, stamp, "compose")
        announce(note)
        available, reason = compose_engine_available_for(source, monkeypatch)
        if not available:
            pytest.skip(f"{note}  That comparison cannot run here: {reason}")
        recipe = composed_recipe(source)
        reference = measure_compose_digest(source, tmp_path / "python")
        assert_engine_compose_matches(
            source, recipe, reference, tmp_path / "rust", monkeypatch, note)
        return

    announce(native_golden_note(source, stamp, "compose"))
    observed = measure_compose_digest(source, tmp_path)
    assert canonical(observed) == canonical(expected)


def _unported_compose_route_is_pinned(source: str, monkeypatch) -> None:
    """Assert the ROUTE before letting an unported row skip.

    The skip then states a measured fact: a bare prep of this source
    composes on the Python engine, and an explicit rust request still
    reaches the engine and earns its own refusal.  Without both, "the
    compose battery passed" could mean nothing ran either way.
    """

    from woof.mapped_source import _mapped_engine_choice

    source_format = str(composed_recipe(source)["format"])
    monkeypatch.delenv(engine_bridge.ENGINE_ENV, raising=False)
    assert _mapped_engine_choice(
        grib1_bridge=None, grib2_inventory=None, grib2_dump=None,
        subcommand="compose", source_format=source_format,
    ) == engine_bridge.ENGINE_PYTHON, (
        f"{source_format} is not in ENGINE_CAPABILITIES['compose'] and a "
        "bare run did NOT route to the Python engine; a default prep of "
        "this source would refuse where it used to work")
    monkeypatch.setenv(engine_bridge.ENGINE_ENV, engine_bridge.ENGINE_RUST)
    assert _mapped_engine_choice(
        grib1_bridge=None, grib2_inventory=None, grib2_dump=None,
        subcommand="compose", source_format=source_format,
    ) == engine_bridge.ENGINE_RUST, (
        "an explicit rust request must reach the engine and earn its own "
        "not_implemented refusal, or a caller could believe a Rust "
        "composition happened when it never did")
    monkeypatch.delenv(engine_bridge.ENGINE_ENV, raising=False)


@pytest.mark.parametrize("source", sorted(COMPOSED_SOURCES))
def test_the_rust_engine_reproduces_the_compose_golden(
        source, tmp_path, monkeypatch):
    """The compose gate: the engine's composed answer equals the reference.

    Run through ``decode_composed_source`` with NO tool flags and the
    engine requested, which is the REAL front-door route
    (``_compose_through_engine``), rather than by driving the exe and
    rebuilding a bundle by hand -- the bundle assembly, the frames'
    mapping-hash check and the "did it compose the declared bindings"
    check are part of what has to work.

    The reference is the committed compose golden on the platform that
    measured it and the Python engine's LIVE composed answer on this box
    anywhere else, at identical strictness either way.
    """

    compose_requires(source)
    recipe = composed_recipe(source)
    # `compose_engine_available_for` asks through `engine_supports`, the
    # ONE reader of the capability table, rather than testing membership.
    # An EMPTY frozenset means "this subcommand implements no format",
    # and the `x or {...}` idiom reads that as "every format" because an
    # empty set is falsy -- so a hand-rolled check would have compared
    # against an engine that refuses everything.
    available, reason = compose_engine_available_for(source, monkeypatch)
    if not available:
        pytest.skip(f"{source}: {reason}")

    path = compose_golden_path(source)
    assert path.is_file(), f"no compose golden for {source}"
    golden, stamp = load_golden(path)
    if golden_is_native(stamp):
        note = native_golden_note(source, stamp, "compose")
        expected = golden
    else:
        note = foreign_golden_note(source, stamp, "compose")
        expected = measure_compose_digest(source, tmp_path / "python")
    announce(note)

    assert_engine_compose_matches(
        source, recipe, expected, tmp_path, monkeypatch, note)


def _restaged(recipe, root: Path):
    """``recipe`` with every staged input moved under ``root``."""

    def move(path) -> Path:
        path = Path(path)
        return Path(root) / path.parent.name / path.name

    return {
        **recipe,
        "primary": tuple(move(path) for path in recipe["primary"]),
        "supplements": {role: tuple(move(path) for path in paths)
                        for role, paths in recipe["supplements"].items()},
    }


def _composed_header(recipe, spell, *, temperature: str, pressure: str,
                     record: int = 12):
    """A composed header shaped like the real one, quoting ``recipe``'s inputs.

    ``spell`` turns a handed path into the spelling the header records, and
    ``record`` is the primary record the temperature was read from.
    """

    primary = spell(recipe["primary"][0])
    donor = spell(next(iter(recipe["supplements"].values()))[0])
    return {
        "schema": "gpuwm.source_frame.v1",
        "source_id": str(recipe["source"]),
        "source_cycle": "2026-08-17T00:00:00+00:00",
        "grid": {"projection": "regular_latitude_longitude",
                 "nx": 1440, "ny": 721},
        "vertical_coordinates": {},
        "initialization_policies": {},
        "fields": [
            {"canonical_name": "air_temperature",
             "data_reference": f"sha256:{temperature}",
             "source_field": f"{primary}:{record}"},
            {"canonical_name": "soil_temperature",
             "data_reference": "sha256:bbbb",
             "source_field": f"{donor}:40"},
            {"canonical_name": "surface_pressure",
             "data_reference": f"sha256:{pressure}",
             "source_field": f"{primary}:3;{primary}:7"},
        ],
    }


def _header_digest_as_handed(row, **values) -> str:
    return composed_header_digest(
        row, _composed_header(row, str, **values))


def test_a_composed_header_digest_does_not_move_with_the_staging_root(
        tmp_path):
    """A compose golden reproduces wherever its bytes are staged.

    Named breakage: the ``aigfs`` and ``aigefs`` compose goldens were
    stamped with the staging override pointing at a folder that was
    later deleted.  The raw frame header quotes each field's absolute input
    path, so both goldens failed on every other box and at every other
    path while every array, input hash and receipt matched.  The digest
    must not move with the root, and must still move when a directly
    decoded array does.
    """

    recipe = composed_recipe("aigefs")
    here = _restaged(recipe, tmp_path / "one" / "staging")
    there = _restaged(recipe, tmp_path / "two" / "elsewhere")
    digest = _header_digest_as_handed

    assert (digest(here, temperature="aaaa", pressure="cccc")
            == digest(there, temperature="aaaa", pressure="cccc"))
    # The raw header is what moved, which is the defect this retires.
    assert _canonical_sha256(_composed_header(
        here, str, temperature="aaaa", pressure="cccc")) !=         _canonical_sha256(_composed_header(
            there, str, temperature="aaaa", pressure="cccc"))
    # A directly decoded array that changed still moves the digest.
    assert (digest(here, temperature="aaaa", pressure="cccc")
            != digest(here, temperature="zzzz", pressure="cccc"))
    # So does the record a field was read from: reducing each path to its
    # file name must not also mask which record the engine took.
    assert (digest(here, temperature="aaaa", pressure="cccc")
            != digest(here, temperature="aaaa", pressure="cccc", record=13))
    # The derived surface pressure is named, not hashed, in the header;
    # its array digest is compared exactly in the frame's fields table.
    assert "surface_pressure" in _composed_libm_fields(recipe)
    assert (digest(here, temperature="aaaa", pressure="cccc")
            == digest(here, temperature="aaaa", pressure="dddd"))


def test_a_composed_header_digest_does_not_move_through_a_linked_root(
        tmp_path):
    """A staging root reached through a directory link digests the same.

    Named breakage: the composition resolves its inputs before it records
    them, so a root reached through a junction or a symlink is quoted by
    its target's spelling while the row hands the linked one; masking the
    handed spelling alone left the resolved path in the digest.
    """

    recipe = composed_recipe("aigefs")
    here = _restaged(recipe, tmp_path / "one" / "staging")
    target = tmp_path / "physical"
    linked = tmp_path / "linked"
    target.mkdir()
    try:
        os.symlink(target, linked, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"this box cannot create a directory link: {error}")
    through_link = _restaged(recipe, linked)
    for path in (*through_link["primary"],
                 *next(iter(through_link["supplements"].values()))):
        Path(os.path.realpath(path)).parent.mkdir(parents=True,
                                                   exist_ok=True)
        Path(os.path.realpath(path)).touch()
    resolved = composed_header_digest(through_link, _composed_header(
        through_link, os.path.realpath, temperature="aaaa",
        pressure="cccc"))
    assert resolved == _header_digest_as_handed(
        here, temperature="aaaa", pressure="cccc")


def test_a_declared_compose_format_is_a_declared_decode_format():
    """Compose cannot outrun decode, per format.

    Named breakage, measured on this tip: ``mapped_authoring`` seals a
    preparation's decoder rows by asking the capability table about
    ``decode``, while ``decode_composed_source`` resolves the decoders it
    will actually run by asking about ``compose``.  A format declared for
    one and not the other makes those two answers name DIFFERENT decoder
    role sets -- one in-process engine row versus the subprocess pair --
    so the manifest seals one binary and the composition then verifies
    against another, and ``_verify_manifest`` refuses a preparation that
    is in fact correct.

    The rule that keeps them in step: a format may be declared for
    ``compose`` only when it is already declared for ``decode``.
    """

    decode = engine_bridge.ENGINE_CAPABILITIES["decode"]
    compose = engine_bridge.ENGINE_CAPABILITIES["compose"]
    if decode is None or compose is None:
        return
    assert set(compose) <= set(decode), (
        f"format(s) {sorted(set(compose) - set(decode))} are declared for "
        "`compose` but not for `decode`; the input manifest would be sealed "
        "against a decoder inventory the composition then refuses")


@pytest.mark.parametrize("source_format", sorted(_MAPPED_SOURCE_FORMATS))
def test_the_two_questions_name_one_decoder_inventory(
        source_format, monkeypatch):
    """The other half of the rule above, and it is not symmetric.

    The subset rule keeps ``compose`` from outrunning ``decode``.  This
    one measures what happens in the opposite direction -- a format
    declared for ``decode`` and NOT for ``compose`` -- because that is
    the state every format sits in before its compose port is measured,
    and it is only survivable for some of them.

    Both production sites are driven here, with the same helpers they
    call: ``mapped_authoring`` asks the capability table about ``decode``
    and seals the manifest against the answer, while
    ``decode_composed_source`` asks about ``compose`` and verifies the
    manifest against ITS answer.  ``_verify_manifest`` compares the role
    SETS, so a preparation is only correct when the two agree.

    Measured, not deduced: a GRIB format survives the split because the
    front door forwards its subprocess tools, and an explicit tool pins
    the Python engine at BOTH sites.  NetCDF has no subprocess tool to
    forward, so nothing can bring the two answers into step -- the
    manifest seals the in-process engine row and the composition then
    verifies against an empty inventory, and a correct NetCDF
    preparation refuses.  That is why ``compose`` must be declared for
    NetCDF exactly while ``decode`` is.
    """

    from woof.mapped_composition import _decoder_inventory
    from woof.mapped_source import _mapped_engine_choice

    monkeypatch.delenv(engine_bridge.ENGINE_ENV, raising=False)
    # The tools the front door would forward for this format on a route
    # that needs them: none for NetCDF, by contract.
    forwarded = {
        name: Path(f"/staged/{name}")
        for name in _FORMAT_TOOLS[source_format]
    }

    def inventory(subcommand):
        engine = None
        if _mapped_engine_choice(
                grib1_bridge=forwarded.get("grib1_bridge"),
                grib2_inventory=forwarded.get("grib2_inventory"),
                grib2_dump=forwarded.get("grib2_dump"),
                subcommand=subcommand,
                source_format=source_format,
        ) == engine_bridge.ENGINE_RUST:
            engine = Path("/staged/gpuwm_mapped_engine")
        return set(_decoder_inventory(
            source_format,
            grib1_bridge=forwarded.get("grib1_bridge"),
            grib2_inventory=forwarded.get("grib2_inventory"),
            grib2_dump=forwarded.get("grib2_dump"),
            engine=engine,
        ))

    # As shipped: both entries declare this format, both answers agree.
    assert inventory("decode") == inventory("compose")

    # With `compose` un-declared for it, the answers agree only when the
    # format HAS a subprocess tool for the door to forward.
    monkeypatch.setattr(
        engine_bridge, "ENGINE_CAPABILITIES",
        dict(engine_bridge.ENGINE_CAPABILITIES,
             compose=frozenset(engine_bridge.ENGINE_CAPABILITIES["compose"]
                               or ()) - {source_format}))
    agrees = inventory("decode") == inventory("compose")
    if _FORMAT_TOOLS[source_format]:
        assert agrees, (
            f"{source_format} has subprocess tools the door forwards, and an "
            "explicit tool pins the Python engine at both sites, so the two "
            "answers must still name one inventory")
    else:
        assert not agrees, (
            f"{source_format} has no subprocess tool to pin, so an undeclared "
            "`compose` CANNOT agree with a declared `decode` -- if this ever "
            "passes, the asymmetry this gate exists for has changed and the "
            "capability table's rule has to be re-derived rather than assumed")
