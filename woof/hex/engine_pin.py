"""
Current public pin: woof 2.9.7 at public engine commit
4726b689c6db142577eca3968a1f74256e3c057f, tag v2.9.7. The measurement
repin-297-20261025.json verifies all published platform wheels, the complete
sixteen-path seam, composed glacier digest, contract surface and helper set.
The dependency is woof>=2.9.11,<2.10 (release installation minimum: 2.9.11 carries the wind-farm
forcing the hex wind-farm door drives and the rw_mpas_convert --grid-label the view titles use; 2.9.10 was the
first engine with woof.output_formats, which every forecast door's parser imports for --output-format). The sealed 2.9.3 public packet has
its own complete measured PACKET_ENGINES row, recorded separately in
verification/engine-verdicts/prepared-engines.json. Historical pin records
below remain measurements of their original inputs.
Which published woof this port can run on: measured once, derived after.

THE BREAKAGE THIS MODULE PREVENTS, measured on a real install on 2026-08-27
(``evidence/userwalk-20260827/``): ``pip install woof hex`` resolved
``woof 2.5.7``, whose bytes the sixteen-file seam manifest this port pins
refused.  The forecast lane stopped at launch with two SHA-256 digests and
no version number, ``woof hex doctor`` reported the estate healthy and
exited 0, and no document in the distribution named a version that works.
A green install and a dead run, with no route out by reading.

The dependency floor alone cannot prevent that, because a floor with no
ceiling always resolves to the newest engine, and the newest engine is by
definition the one this port has never measured.

So the specifier is DERIVED rather than transcribed:

* :data:`PUBLISHED_ENGINES` is the OUTPUT of a network measurement --
  ``tools/measure_engine_verdicts.py``, which downloads every published
  wheel of every engine from 2.5.0 up, hashes the sixteen pinned paths
  inside each one, and reads the build road off the engine's git tags --
  not a restatement of anybody's prose.  Every hand-written statement of
  these numbers in this tree was wrong: the declaration said 5 of 16 files
  differ at 2.5.0 (measured: 9) and 3 at 2.5.4 (measured: 2), and the test
  guarding it said 13 at 2.5.0 and 6 at 2.5.1 (measured: 9 and 8).  Three
  transcriptions, three different wrong answers.  The table is SPLICED into
  the generated block below by the instrument itself (``--splice``), its
  JSON ships beside the contracts in ``verification/engine-verdicts/``, and
  ``tests/test_engine_pin.py`` re-renders that JSON and demands equality, so
  there is no fourth transcription to get wrong.  The instrument used to
  live under ``evidence/``, which every published surface holds out; it
  moved into ``tools/`` on 2026-09-13 so a reader of the public tree can run
  the thing the table cites.
* :func:`gpuwm_floor`, :func:`gpuwm_ceiling` and :func:`gpuwm_requirement`
  compute the specifier from that table.  ``pyproject.toml`` states the
  result as a literal because a dependency specifier has to be static
  metadata, and ``tests/test_packaging_declaration.py`` fails if the literal
  and the derivation ever disagree.

**The ceiling is EXCLUSIVE and sits at the first engine that is not measured
usable, which is what stops the next engine cut from silently re-opening
this.**  An engine nobody has measured against the manifest is never
resolved onto by default; admitting one is a deliberate act that re-runs the
instrument and moves this table.

WHERE THIS STANDS NOW, re-measured 2026-09-29
(``verification/engine-verdicts/repin-280-20260929.json``, after the
pre-move measurement ``premeasure-280-20260929.json`` the same day).  The
engine published 2.7.5, 2.7.6, 2.7.7 and 2.8.0 while the port pinned
2.7.4; measured against that manifest all four fail, 2.8.0 by eight paths
(``woof/config.py``, ``woof/core/gf.py``, ``woof/core/kernels/gf.cu``,
``woof/core/kernels/__init__.py``, ``woof/core/microphysics.py``,
``woof/core/physics.py``, ``woof/core/rrtmg_legacy.py``,
``woof/io/restart.py``).  The other eight are byte-identical between the
published 2.7.4 and 2.8.0 wheels: the column batch, the seam document and
the Noah-MP files, the composed glacier unit included, so
``ARWEN_GLACIER_COMPOSED_TU_SHA256`` did not move.  The manifest was
re-pinned to the published 2.8.0 wheel's bytes (tag ``0164ae0f``, whose
tree agrees with all three platform wheels on every pinned path) and the
instrument re-run over every published engine.  The derivation now returns
``woof>=2.8.0,<2.8.1``: 2.8.0 is the one engine measured usable and 2.7.4
fails the re-pinned manifest by those eight paths.  What the moved paths
do to the fixed-step WSM6, YSU, Noah-MP, legacy-RRTMG and Grell-Freitas
seam was measured with the seam-level A/B
(``tools/measure_engine_seam_ab.py``) and a point-cull forecast arm, cited
from ``docs/declared-divergences.md``.

THE 2.7.4 MOVE, re-measured 2026-09-15
(``verification/engine-verdicts/repin-274-20260915.json``, after the
pre-move measurement ``premeasure-274-20260915.json`` the same day).  The
engine published 2.7.4 (2026-09-13); measured against the manifest the
0.3.0 candidate carried, it fails by two paths -- ``woof/config.py`` (one
added reader, ``radiation_scheme_ids_from_settings``, a settings-map form
of the radiation rule that the port never calls) and
``woof/core/microphysics.py`` (two keyword arguments on the classic
Thompson path's cloud adjustment and rain evaporation, a scheme the port's
WSM6 backend never runs) -- and the other fourteen are byte-identical
between the two published wheels, the composed Noah-MP glacier unit
included, so ``ARWEN_GLACIER_COMPOSED_TU_SHA256`` did not move.  The
manifest was re-pinned to the published 2.7.4 wheel's bytes (tag
``ed957997``, whose tree agrees with all three platform wheels on every
pinned path) and the instrument re-run over every published engine.  The
derivation then returned ``woof>=2.7.4,<2.7.5``: 2.7.4 was the one engine
measured usable, 2.7.3 fails the re-pinned manifest by those two paths,
and every earlier engine fails it by the same counts it failed the 2.7.3
manifest by, because the two paths that moved were already in every one
of their ``moved`` lists.  Neither moved path reaches a number the port's
fixed-step WSM6, YSU, Noah-MP, legacy-RRTMG and Grell-Freitas seam
computes; the seam-level A/B (``tools/measure_engine_seam_ab.py``) and the
one-hour point-cull arm that measured that are cited from
``docs/declared-divergences.md``.

THE PREVIOUS MOVE, re-measured 2026-09-13
(``repin-273-20260913.json``, after ``premeasure-273-20260913.json`` the
same day), kept here because it is the one that moved the whole ``moved``
column.  The engine published 2.7.0, 2.7.1, 2.7.2 and then 2.7.3 (2026-09-12); measured
against the manifest 0.2.3 shipped, every one of the four fails it -- the
three earlier 2.7 wheels by seven paths and 2.7.3 by eleven -- so the
``<2.6.6`` ceiling did exactly what it exists to do for ten days: a user who
installed woof hex 0.2.3 after 2.7.0 was published got 2.6.5, not the
newest engine, and no run died on a digest.  This port then re-pinned its
manifest to the published 2.7.3 wheel's bytes and the instrument was re-run
over every published engine plus the two tags that have no release.  The
derivation then returned ``woof>=2.7.3,<2.7.4``: 2.7.3 was the one
engine measured usable.  Letting it in cost what it is meant to cost: re-run the
instrument, move the table, let the literal follow.  The 2026-09-02 re-pins
to 2.6.4 and 2.6.5, the same day's re-pin to 2.6.3, the 2026-09-01 re-pin
to 2.6.1, the 2026-08-31 re-pin to 2.6.0 and the 2026-08-28 re-pin to 2.5.8
paid the same toll the same way (their receipts are under the private
tree's ``evidence/repin-*``; this tree carries the current measurement).

**EXACTLY ONE PUBLISHED ENGINE IS USABLE, and that is a measured finding
rather than a cautious choice.**  It is 2.7.4 now; at the previous re-pin
it was 2.7.3, and that move is the one to read for what a re-pin does to
this table.  Re-pinning the manifest to 2.7.3 moved the
whole ``moved`` column, because that column is relative to the manifest:
every engine below 2.7.3 fails it, INCLUDING the 2.6.4 and 2.6.5 that were
the whole admitted range the day before.  Eleven manifest paths moved
between the published 2.6.5 wheel and the published 2.7.3 wheel, measured
wheel against wheel -- ``woof/core/kernels/gf.cu`` (the Grell-Freitas
kernel: the glibc ``tgammaf`` transcription is gone and the scheme's
``fzu`` normalisation now uses the engine's own correctly rounded gamma;
the cloud-work integral's level loop now includes the ``kbcon`` layer;
three per-column ``fzu`` override slots the shipped path leaves at zero),
``woof/core/gf.py`` (the adapter's scalar block grew from six slots to
nine for those overrides), ``woof/core/physics.py`` (the phase-one driver:
adaptive-clock overrides for the radiation and cumulus due predicates that
stay ``None`` on every fixed-step run, the cumulus advective-forcing
carriers owned from construction and copied into rather than rebound, the
held PBL raw rates, the SASE vertical-momentum component),
``woof/core/microphysics.py`` (the ring-state field list is derived from
the scheme registry), ``woof/core/rrtmg_legacy.py`` (selectable spectra
and trace-gas overrides with unchanged defaults, a registry-agreement check
at import, the third-party notice header), ``woof/core/noahmp_kernel_sources.py``,
``woof/core/noahmp_glacier_gpu.py`` and ``woof/core/kernels/__init__.py``
(one Noah-MP compile site with a recorded unit identity; the composed
glacier unit's text moves with it, so ``ARWEN_GLACIER_COMPOSED_TU_SHA256``
moved too), ``woof/core/kernels/noahmp_leaves.cu`` (the ``log10(+-0)``
spelling, which the engine compared equal on every float32 bit pattern),
``woof/config.py`` (adaptive time-step fields and run-preparation
preconditions; ``MP_PHYSICS_ACCEPTED`` and ``CU_SCHEMES`` did not move) and
``woof/io/restart.py`` (restart format 6, the algorithm-identity tables
moved into the registry) -- while ``woof/core/mpas_column_batch.py``,
``woof/core/noahmp_runtime.py``, ``woof/core/noahmp_glacier.py``,
``woof/core/kernels/noahmp_glacier.cu`` and ``docs/mpas-seam.md`` are
byte-identical between the two wheels.  The seam's own driver did not move
and its constructor takes the same keywords, which is why the port's
adapter needed no change to admit the new engine.  2.7.0, 2.7.1 and 2.7.2
carry the same sixteen bytes as each other and fail the re-pinned manifest
by eight paths.  Of the eleven moved files, ONE moves a physics number
through the seam on a fixed-step run: the Grell-Freitas kernel, measured
seam against seam by ``tools/measure_engine_seam_ab.py``; the rest are off
the fixed-step path, bit-equal by the engine's own measurement, or identity
plumbing.  The same instrument found a SECOND move that sits outside the
sixteen -- ``woof/core/kernels/ysu.cu``, the boundary-layer kernel, which
this manifest has never pinned -- and attributed it exactly, by swapping
that one file.  Both are stated with their magnitudes in
``docs/declared-divergences.md``; the manifest's reach is a named
follow-up.  If the one admitted engine were yanked from PyPI, the correct
answer is :class:`EnginePinError`, not a loosened bound.

ADMITTED ENGINES.  A build that is not on PyPI -- a private engine carrying a PEP 440 local
version label on a published base, ``2.7.4+x.1`` -- has no row in the published table, because that table is a
measurement of what PyPI serves, and it satisfies the derived specifier
anyway, because PEP 440 orders a local version immediately after its base.
So the specifier is not the gate for it and was never meant to be; the
bytes are.  :data:`ADMITTED_ENGINES` is the second table this module
carries, written by the same instrument in its ``--admit`` mode from the
engine's own artifacts (its wheels and its source tree, which must agree
with each other and with the git tree's HEAD on all sixteen paths), keyed
by the exact version string and the sixteen digests together, and carrying
the build commit and glacier digest a run's receipts name instead of the
public pin's.  :func:`inspect_seam` tries an admitted row only when the
public pin is not matched, matches it only in full, and refuses the row's
bytes under any other version string.  An engine with no row -- a private
build nobody measured, a re-versioned copy -- is refused exactly as an
unmeasured public engine is.  The published distribution ships the table
EMPTY: a row names a build that is not public, so it belongs to whoever
builds and installs that engine, who writes it with ``--admit`` into their
own copy of this tree.  The mechanism ships and is tested with a
synthetic row, so an admitted engine is gated and provenanced by exactly
the same code whether or not a public row exists.  A row whose base the
specifier refuses admits nothing (``tests/test_engine_pin.py`` refuses
such a row by name).

Two facts the 2026-08-27 measurement turned up that no document in the tree
carried.  Both are still true, and both are now moot for the FLOOR, because
2.5.5 and 2.5.6 fail the re-pinned manifest on bytes before either one
applies.  They stay in the table and stay written down here, because a row
removed is a reason the derivation can no longer show:

* **woof 2.5.5 is not on PyPI.**  It is a git tag with no release, so the
  floor the declaration once named -- ``woof>=2.5.5`` -- named a version
  pip cannot install.
* **2.5.5's published tree cannot follow its own printed build road.**
  ``tools/rustwx/vendor/crates-io/cc/src/target/generated.rs`` is absent at
  that tag, so the manual's ``cargo build --release --locked --offline``
  stops there; the same command finished in 56.6 s at 2.5.6.  A version that
  satisfies the pin and cannot be built from is not a remedy, so the
  derivation requires both -- which is why :attr:`PublishedEngine.usable`
  keeps all three clauses even now that the byte clause alone decides the
  answer.

This module imports nothing heavy at module scope on purpose: ``doctor``
imports it, and a report about a broken estate must not need the estate to
work.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Any

#: The engine distribution, spelled once.
DISTRIBUTION = "recast-woof"

#: The public repository a checkout comes from, named in remedies.
REPOSITORY = "https://github.com/recastsystems/woof"


@dataclass(frozen=True)
class PublishedEngine:
    """One published woof, as measured against this port's seam manifest."""

    version: str
    #: Whether pip can resolve it at all.  A tag is not a release.
    on_pypi: bool
    #: Manifest paths whose bytes differ at this version.  Empty means the
    #: whole sixteen-file manifest matches.
    moved: tuple[str, ...]
    #: Whether the vendored crate tree at this version is complete enough for
    #: the ``--offline`` build the manual prints as its remedy.
    offline_build_road: bool

    @property
    def satisfies_manifest(self) -> bool:
        return not self.moved

    @property
    def usable(self) -> bool:
        """Installable, byte-compatible, and buildable from.  All three."""

        return self.on_pypi and self.satisfies_manifest and self.offline_build_road


#: THE OUTPUT OF AN INSTRUMENT, NOT A TRANSCRIPTION.  Re-measured 2026-10-14
#: against the published bytes of every 2.5.x, 2.6.x, 2.7.x and 2.8.x woof and
#: the 2.9.0 wheels built from its release commit, plus the v2.5.5 and v2.6.2
#: tags that have no PyPI release.  The instrument is
#: ``tools/measure_engine_verdicts.py`` (network: it downloads every
#: published wheel of every version and refuses a version whose platform
#: wheels disagree); the JSON behind THIS table is
#: ``verification/engine-verdicts/repin-290-20261014.json``, and the block
#: below is that JSON rendered by the same instrument's ``--splice``.
#:
#: DO NOT HAND-PATCH A ROW.  ``moved`` is measured against THIS port's
#: sixteen-file seam manifest, so re-pinning the manifest moves the whole
#: column at once -- including rows for engines cut long before it.  The
#: 2026-08-28 re-pin to 2.5.8 changed every one of the nine rows: 2.5.7 both
#: GAINED entries (config.py, physics.py) and LOST one (microphysics.py,
#: whose bytes 2.5.7 and 2.5.8 share).  A patched row is a guess wearing a
#: measurement's clothes, and this module's docstring counts what that has
#: cost so far.
# --- BEGIN GENERATED: PUBLISHED_ENGINES ---
PUBLISHED_ENGINES: tuple[PublishedEngine, ...] = (
    PublishedEngine(
        version="2.5.0",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/kernels/noahmp_leaves.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.5.1",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/kernels/noahmp_leaves.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.5.2",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/kernels/noahmp_leaves.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.5.3",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/kernels/noahmp_leaves.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.5.4",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/kernels/noahmp_leaves.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.5.5",
        on_pypi=False,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/kernels/noahmp_leaves.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=False,
    ),
    PublishedEngine(
        version="2.5.6",
        on_pypi=False,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/kernels/noahmp_leaves.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.5.7",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/kernels/noahmp_leaves.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.5.8",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/kernels/noahmp_leaves.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.6.0",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/kernels/noahmp_leaves.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.6.1",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/kernels/noahmp_leaves.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.6.2",
        on_pypi=False,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/kernels/noahmp_leaves.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.6.3",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/kernels/noahmp_leaves.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.6.4",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/kernels/noahmp_leaves.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.6.5",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/kernels/noahmp_leaves.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.7.0",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.7.1",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.7.2",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.7.3",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.7.4",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.7.5",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.7.6",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.7.7",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.8.0",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.8.1",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.8.2",
        on_pypi=False,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.8.3",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.8.4",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.8.5",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.8.6",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.8.7",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.8.9",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/kernels/noahmp_glacier.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.9.0",
        on_pypi=True,
        moved=(
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/kernels/gf.cu",
            "woof/core/microphysics.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.9.2",
        on_pypi=True,
        moved=(
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_kernel_sources.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.9.3",
        on_pypi=True,
        moved=(
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/noahmp_glacier.py",
            "woof/core/noahmp_glacier_gpu.py",
            "woof/core/noahmp_runtime.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.9.4",
        on_pypi=True,
        moved=(
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.9.5",
        on_pypi=True,
        moved=(
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.9.6",
        on_pypi=True,
        moved=(
            "woof/config.py",
            "woof/core/gf.py",
            "woof/core/kernels/__init__.py",
            "woof/core/mpas_column_batch.py",
            "woof/core/physics.py",
            "woof/core/rrtmg_legacy.py",
            "woof/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.9.7",
        on_pypi=True,
        moved=(),
        offline_build_road=True,
    ),
)
# --- END GENERATED: PUBLISHED_ENGINES ---


@dataclass(frozen=True)
class AdmittedEngine:
    """One engine admitted by a measured row of its own.

    An engine that is not on PyPI -- a private build carrying a PEP 440
    local version label such as ``2.7.4+x.1`` -- cannot appear in
    :data:`PUBLISHED_ENGINES`, because that table is a measurement of what
    PyPI serves.  It can still be the engine a user has installed, and the
    dependency specifier already lets it through: PEP 440 orders a local
    version immediately after its public base, so ``woof>=2.7.4,<2.7.5``
    resolves ``2.7.4+x.1``.  What decides whether this port RUNS on it is
    the same thing that decides for a published engine -- its sixteen seam
    bytes -- so an admitted engine is a row of the same kind: measured by
    the instrument from the engine's own artifacts, keyed by its exact
    version string AND its bytes, and refused when either differs.

    A tree whose bytes match this row under this version string is accepted
    by the seam check, the forecast door and ``doctor``, and its receipts
    name THIS row's build commit and digests rather than the public pin's.
    A tree that carries these bytes under another version string, or this
    version string over other bytes, is refused by name: the row admits one
    engine, not a family.
    """

    #: The exact version string, local label included.
    version: str
    #: The public version whose pip window this engine rides.
    base: str
    #: HEAD of the tree the measured artifacts were built from.
    build_commit: str
    #: All sixteen manifest paths and their digests in this engine, sorted.
    digests: tuple[tuple[str, str], ...]
    #: ``sha256(translation_unit_source('noahmp_glacier'))`` by this
    #: engine's own hand, from its extracted wheel.
    glacier_composed_tu_sha256: str
    #: The manifest paths whose bytes differ from the public pin.
    moved_from_base: tuple[str, ...]
    #: The artifacts the row was hashed from, as (filename, sha256).  Every
    #: one of them agreed on all sixteen paths, or the row was not written.
    artifacts: tuple[tuple[str, str], ...]

    @property
    def manifest(self) -> dict[str, str]:
        return dict(self.digests)


#: THE OUTPUT OF THE SAME INSTRUMENT, in its ``--admit`` mode: it hashes the
#: sixteen paths inside every artifact named (the engine's wheels and its
#: source tree), refuses if they disagree with each other or with the git
#: tree's HEAD, computes the glacier digest from the extracted wheel, and
#: writes ``verification/engine-verdicts/admitted-engines.json``, which the
#: block below is rendered from by ``--splice-admitted``.
#: ``tests/test_engine_pin.py`` re-renders that JSON and demands equality.
#:
#: DO NOT HAND-PATCH A ROW, for the reason the published table gives.
# --- BEGIN GENERATED: ADMITTED_ENGINES ---
ADMITTED_ENGINES: tuple[AdmittedEngine, ...] = ()
# --- END GENERATED: ADMITTED_ENGINES ---

#: Sealed public packets are measured separately from PyPI and private builds.
#: Each row is generated by tools/measure_engine_packet.py from the sealed
#: wheel and its public source commit. All sixteen paths and the exact version
#: must match; these rows never change the published table or dependency floor.
# --- BEGIN GENERATED: PACKET_ENGINES ---
PACKET_ENGINES: tuple[AdmittedEngine, ...] = ()
# --- END GENERATED: PACKET_ENGINES ---


def admitted_engine(version: str) -> AdmittedEngine | None:
    """The admitted row for ``version`` exactly, or ``None``."""

    for row in (*ADMITTED_ENGINES, *PACKET_ENGINES):
        if row.version == version:
            return row
    return None


@dataclass(frozen=True)
class EngineIdentity:
    """The engine a tree's bytes were measured to be: what receipts name.

    One shape for both kinds of admission.  For the public pin the fields
    are the adapter module's own constants; for an admitted row they are
    that row's.  The adapter records ``build_commit``, ``manifest`` and the
    glacier digest into every receipt and restart identity, so a run on an
    admitted engine is provenanced by the commit its bytes came from rather
    than by the public pin it does not carry.
    """

    version: str
    build_commit: str
    manifest: dict[str, str]
    glacier_composed_tu_sha256: str
    #: ``False`` for the public pin, ``True`` for an admitted row.
    admitted: bool
    #: ``True`` when the tree's bytes are the public pin's or an admitted
    #: row's in full; ``False`` for a tree that is recorded as it was found
    #: (:func:`identity_of`).  Evidence for the receipt, never a permission.
    matches_measured_row: bool = True
    #: Manifest paths whose bytes differ from the public pin, and paths the
    #: tree does not carry.  Both empty when ``matches_measured_row``.
    moved_from_pin: tuple[str, ...] = ()
    absent_from_pin: tuple[str, ...] = ()


def pinned_engine() -> EngineIdentity:
    """The public pin as an identity, read from the adapter's constants."""

    from .cuda_arwen_physics_v841 import (
        ARWEN_BUILD_COMMIT,
        ARWEN_GLACIER_COMPOSED_TU_SHA256,
        ARWEN_SOURCE_MANIFEST,
    )

    return EngineIdentity(
        version=wanted_version(),
        build_commit=ARWEN_BUILD_COMMIT,
        manifest=dict(ARWEN_SOURCE_MANIFEST),
        glacier_composed_tu_sha256=ARWEN_GLACIER_COMPOSED_TU_SHA256,
        admitted=False,
    )


def identity_for(version: str) -> EngineIdentity:
    """The identity a matched engine version names; the pin or an admitted row."""

    row = admitted_engine(version)
    if row is None:
        pinned = pinned_engine()
        if version != pinned.version:
            raise EnginePinError(f"woof {version} is neither the pin nor an admitted engine")
        return pinned
    return EngineIdentity(
        version=row.version,
        build_commit=row.build_commit,
        manifest=row.manifest,
        glacier_composed_tu_sha256=row.glacier_composed_tu_sha256,
        admitted=True,
    )


#: ``build_commit`` of an engine tree that matches no measured row.  The
#: tree's own commit (or an install's RECORD digest) is recorded beside it by
#: the forecast driver's ``arwen_git`` block; this string only says that no
#: measured row named one.
UNMEASURED_BUILD_COMMIT = "no-measured-row"


def identity_of(inspection: "SeamInspection") -> EngineIdentity:
    """The engine a tree IS, whether or not a measured row names it.

    RETIRED 2026-10-10 as a gate (owner ruling; Gate law, Refusals law; woof hex goes its own way, 2026-08-20):
    a tree whose sixteen seam bytes matched neither the public pin nor an
    admitted row used to refuse every forecast with "running against moved
    bytes would be running physics nobody proved" / "pinned Arwen source
    hash mismatch".  That names missing evidence, not a breakage, and it
    refused every run on any engine newer than the pin (measured on the 8-card RTX PRO 6000 host:
    woof 2.8.9 against the 2.8.7 pin, 7 of 16 files moved, the door closed).
    A byte match is now EVIDENCE: a matched tree is named by its measured
    row exactly as before, and an unmatched tree is named by the version it
    declares and the digests it was found with, with the moved and absent
    paths listed.  Nothing refuses on the difference.  The x4 proof harness
    keeps its own byte checks, because byte identity is what that harness
    measures.
    """

    if inspection.engine is not None and inspection.accepted:
        return identity_for(inspection.engine)
    pinned = seam_manifest()
    found = dict(inspection.digests)
    return EngineIdentity(
        version=inspection.declared or "undeclared",
        build_commit=UNMEASURED_BUILD_COMMIT,
        manifest=found,
        # Filled by the adapter from the engine's own composed source once
        # the engine is imported; unknown from bytes on disk alone.
        glacier_composed_tu_sha256="",
        admitted=False,
        matches_measured_row=False,
        moved_from_pin=tuple(sorted(
            {relative for relative, digest in found.items()
             if digest != pinned.get(relative)} | set(inspection.moved))),
        absent_from_pin=tuple(sorted(inspection.absent)),
    )


def measured_row_record(inspection: "SeamInspection") -> dict[str, Any]:
    """Whether a tree's seam bytes are a measured row's, for a receipt.

    Evidence only: nothing refuses on it.
    """

    identity = identity_of(inspection)
    return {
        "pinned_engine_version": wanted_version(),
        "declared_engine_version": inspection.declared,
        "matches_measured_row": identity.matches_measured_row,
        "measured_row": identity.version if identity.matches_measured_row else None,
        "admitted_row": identity.admitted,
        "files_checked": inspection.checked,
        "moved_from_pin": list(identity.moved_from_pin),
        "absent_from_pin": list(identity.absent_from_pin),
    }


class EnginePinError(RuntimeError):
    """No published engine satisfies the pin, so no specifier can be derived."""


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def _next_after(version: str) -> str:
    parts = list(_key(version))
    parts[-1] += 1
    return ".".join(str(part) for part in parts)


def engine(version: str) -> PublishedEngine | None:
    """The measured row for ``version``, or ``None`` if it was never measured."""

    for row in PUBLISHED_ENGINES:
        if row.version == version:
            return row
    return None


GPUWM_INSTALL_MINIMUM = "2.9.11"


def measured_gpuwm_floor() -> str:
    """The lowest usable row in the measured published-engine table."""

    usable = [row for row in PUBLISHED_ENGINES if row.usable]
    if not usable:
        raise EnginePinError(
            "no published woof satisfies this port's seam manifest, is on "
            "PyPI and carries a complete vendored crate tree, so no "
            "dependency specifier can be derived.  Re-run "
            "tools/measure_engine_verdicts.py; if it agrees, the engine has "
            "to publish before this distribution can"
        )
    return min(usable, key=lambda row: _key(row.version)).version


def gpuwm_floor() -> str:
    """The installation floor, including the release policy minimum.

    A higher installation minimum is not a new seam measurement. Use
    ``measured_gpuwm_floor`` for table provenance and pinned source identity.
    """

    return max((measured_gpuwm_floor(), GPUWM_INSTALL_MINIMUM), key=_key)


def gpuwm_ceiling() -> str:
    """The EXCLUSIVE upper bound: the next MINOR after the floor.

    THE BREAKAGE THIS PREVENTS (open defect B2, plan call 14, 2026-10-14): a
    range one patch wide made ``pip install woof hex`` quietly DOWNGRADE the
    engine a user already had.  0.3.3 pinned ``<2.8.1``, so installing it
    beside woof 2.8.7 replaced the engine with 2.8.0, and an exact-patch pin
    on 2.9.0 would do the same to every user the day 2.9.1 ships.  So the
    range is the floor's whole minor (``woof>=2.9.0,<2.10``).

    What the narrow range used to do is now done at the door, by name: an
    engine inside the minor whose radiation helper set no row of this
    distribution admits is refused by
    :func:`woof.hex.cuda_arwen_helper_scope_v2.bind_arwen_helpers_v2`, naming
    the installed version, the engine this port was measured on and the two
    remedies; a 2.9.x that moves the seam ships with a woof hex 0.4.x that
    measures it.  A measured row inside the minor that is not usable does not
    lower the ceiling: it is evidence the refusal and ``doctor`` report.

    The next minor is still never resolved onto, because nobody has measured
    it and a minor is where the engine moves its seams.
    """

    major, minor = _key(gpuwm_floor())[:2]
    return f"{major}.{minor + 1}"


def gpuwm_requirement() -> str:
    """The whole dependency specifier ``pyproject.toml`` must declare."""

    return f"{DISTRIBUTION}>={gpuwm_floor()},<{gpuwm_ceiling()}"


def wanted_version() -> str:
    """The measured seam identity named in source provenance and clone remedies.

    Installation commands use ``gpuwm_requirement`` so the release policy
    minimum also applies. A policy change cannot relabel measured bytes.
    """

    return measured_gpuwm_floor()


# ---------------------------------------------------------------------------
# what is actually on this machine
# ---------------------------------------------------------------------------
def seam_manifest() -> dict[str, str]:
    """The sixteen pinned paths and their digests.

    Imported here rather than at module scope because the manifest lives in a
    frozen module that pulls numpy, and ``doctor`` has to be able to report a
    numpy-less install rather than fail to import while doing so.
    """

    from .cuda_arwen_physics_v841 import ARWEN_SOURCE_MANIFEST

    return dict(ARWEN_SOURCE_MANIFEST)


@dataclass(frozen=True)
class SeamInspection:
    """What one tree's bytes say about the seam pin."""

    root: Path
    #: Manifest paths that are present and match.
    matched: tuple[str, ...]
    #: Manifest paths that are present and whose bytes have moved.
    moved: tuple[str, ...]
    #: Manifest paths this tree does not carry at all.
    absent: tuple[str, ...]
    #: The engine these bytes were measured to be: the pinned version when
    #: every path matches the manifest, an admitted row's version when every
    #: path matches that row, ``None`` when they match neither.
    engine: str | None = None
    #: The version the tree declares for itself (``pyproject.toml`` for a
    #: checkout, installed metadata for an install root), or ``None``.
    declared: str | None = None
    #: The digest of every manifest path the tree carries, so a refusal can
    #: print the byte it found beside the byte it wanted.
    digests: tuple[tuple[str, str], ...] = ()

    @property
    def checked(self) -> int:
        return len(self.matched) + len(self.moved)

    @property
    def drifted(self) -> bool:
        return bool(self.moved)

    @property
    def admitted(self) -> bool:
        """Matched an ADMITTED row rather than the public pin."""

        return self.engine is not None and admitted_engine(self.engine) is not None

    @property
    def misdeclared(self) -> bool:
        """The bytes are an admitted engine's and the version string is not.

        An admitted row is keyed by both.  Bytes alone would admit a
        re-versioned copy of a private build and let its receipts name a
        version the build never had; the version alone is the thing this
        whole module exists to distrust.  The public pin keeps its old
        contract -- bytes decide, whatever the tree calls itself -- because
        its rows are measured from PyPI, where the version IS the bytes.
        """

        return self.admitted and self.declared != self.engine

    @property
    def accepted(self) -> bool:
        """Nothing moved, nothing absent, and the version string agrees."""

        return not self.moved and not self.absent and not self.misdeclared


def declared_version(root: Path) -> str | None:
    """What a tree says it is: its own ``pyproject.toml``, else pip's metadata
    when ``root`` is the installed package's root, else ``None``."""

    found = checkout_version(root)
    if found is not None:
        return found
    installed = installed_root()
    if installed is not None and installed == Path(root).resolve():
        return installed_version()
    return None


def inspect_seam(root: Path) -> SeamInspection:
    """Compare every manifest path under ``root`` against the pinned bytes.

    ``absent`` is reported separately from ``moved`` because the two are
    different findings with different remedies: a path this tree does not
    carry at all is a packaging question, while a byte that moved is the
    engine being the wrong version.

    ``absent`` used to be the NORMAL answer for an install.  Engines up to
    2.5.7 shipped no ``docs/mpas-seam.md`` in the wheel, so fifteen of the
    sixteen resolved and the sixteenth could only be reached from a source
    checkout.  2.5.8 places that document in site-packages at the manifest's
    own key: measured 2026-08-28 against the wheel PyPI serves, this returns
    ``checked=16, matched=16, moved=(), absent=()``.  The branch stays,
    because an older engine still answers the old way and the two remedies
    are still different.

    ADMITTED ROWS are tried only when the public pin is not matched in full,
    and a row is matched only in full: all sixteen paths present and equal
    to the row's digests.  Then ``moved`` and ``absent`` are empty and
    ``engine`` names the row.  A tree that matches neither is reported
    against the public pin, as before, so the refusal names the engine this
    port actually pins.
    """

    manifest = seam_manifest()
    found: dict[str, str] = {}
    absent: list[str] = []
    for relative in manifest:
        path = root / relative
        try:
            payload = path.read_bytes()
        except OSError:
            absent.append(relative)
            continue
        found[relative] = sha256(payload).hexdigest()

    engine: str | None = None
    against = manifest
    if not absent and found == manifest:
        engine = wanted_version()
    elif not absent:
        for row in (*ADMITTED_ENGINES, *PACKET_ENGINES):
            if row.manifest == found:
                engine = row.version
                against = row.manifest
                break
    matched = [relative for relative, digest in found.items() if digest == against[relative]]
    moved = [relative for relative, digest in found.items() if digest != against[relative]]
    return SeamInspection(
        root=root,
        matched=tuple(sorted(matched)),
        moved=tuple(sorted(moved)),
        absent=tuple(sorted(absent)),
        engine=engine,
        declared=declared_version(root),
        digests=tuple(sorted(found.items())),
    )


def installed_version() -> str | None:
    """The woof distribution version pip has installed, or ``None``."""

    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as distribution_version

    try:
        return distribution_version(DISTRIBUTION)
    except PackageNotFoundError:
        return None


def installed_root() -> Path | None:
    """The directory the manifest's ``gpuwm/...`` paths hang off, installed.

    ``find_spec`` rather than an import: importing woof pulls its whole
    physics estate, and a report about that estate must not depend on it
    loading.
    """

    import importlib.util

    try:
        spec = importlib.util.find_spec("woof")
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent.parent


def checkout_version(root: Path) -> str | None:
    """The version a woof SOURCE CHECKOUT declares, read without importing.

    ``pyproject.toml`` states it statically, which is the only place a
    checkout carries its own version: ``woof/__init__.py`` reads it back out
    of installed metadata, so importing it from a checkout answers with
    whatever is installed instead of what is on disk -- the exact confusion a
    refusal naming a version has to avoid.

    An INSTALL root answers ``None`` here, and correctly: site-packages
    carries no ``pyproject.toml``.  ``installed_version()`` is the reader for
    that case, and ``version_from_moved`` the last resort for a tree that
    states nothing.
    """

    declaration = root / "pyproject.toml"
    try:
        text = declaration.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        import tomllib

        parsed = tomllib.loads(text)
    except Exception:
        match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
        return match.group(1) if match else None
    project = parsed.get("project")
    if isinstance(project, dict):
        version = project.get("version")
        if isinstance(version, str):
            return version
    return None


def version_from_moved(moved: tuple[str, ...]) -> str | None:
    """Name the published version whose measured drift is exactly ``moved``.

    The last resort when a tree states no version of its own: the table
    records which paths moved at each cut, so an observed drift set that
    matches one row's exactly identifies it.  Ambiguity answers ``None``
    rather than guessing.
    """

    hits = [row.version for row in PUBLISHED_ENGINES if row.moved == tuple(sorted(moved))]
    return hits[0] if len(hits) == 1 else None


def remedy(found: str | None = None) -> str:
    """The command that puts a usable engine on this machine.

    The install is the whole remedy: all sixteen pinned paths resolve from
    site-packages (since 2.5.8) and, since 0.3.2, the forecast names an
    install by version and pip's RECORD digest, so the clone this text used
    to owe for receipt provenance is optional.  It is still printed, as the
    one alternative, because a clone passed as ``--gpuwm-checkout`` is
    recorded by commit.
    """

    wanted = wanted_version()
    lines = [
        f'  pip install "{gpuwm_requirement()}"',
        "      # optional: a clone at the pinned tag, passed as",
        "      # --gpuwm-checkout, is recorded by commit instead of by the",
        "      # install's version and RECORD digest",
        f"  git clone --depth 1 --branch v{wanted} {REPOSITORY}",
    ]
    if found is not None:
        lines.insert(0, f"  # found woof {found}; this port pins {wanted}.")
    return "\n".join(lines)


__all__ = [
    "ADMITTED_ENGINES",
    "AdmittedEngine",
    "DISTRIBUTION",
    "EngineIdentity",
    "EnginePinError",
    "GPUWM_INSTALL_MINIMUM",
    "PUBLISHED_ENGINES",
    "REPOSITORY",
    "PublishedEngine",
    "SeamInspection",
    "admitted_engine",
    "checkout_version",
    "declared_version",
    "engine",
    "gpuwm_ceiling",
    "gpuwm_floor",
    "gpuwm_requirement",
    "UNMEASURED_BUILD_COMMIT",
    "identity_for",
    "identity_of",
    "inspect_seam",
    "measured_gpuwm_floor",
    "measured_row_record",
    "installed_root",
    "installed_version",
    "pinned_engine",
    "remedy",
    "seam_manifest",
    "version_from_moved",
    "wanted_version",
]
