"""Results and compatibility that move between releases, said once.

A release that corrects a transcription against WRF changes the numbers
an existing configuration produces, and it is default-on because a fix
ships default-on.  The reader who upgrades needs to be told WHICH of
their configurations move and by how much, on the two surfaces they
actually read: the CHANGELOG's release section and ``woof doctor``'s
upgrade note.  Those two surfaces used to be written independently, and
the 2.7.0 candidate showed the failure mode: seventeen default-on
changes were in neither.

So the table lives here, once, and both surfaces are built from it:

* ``CHANGELOG.md`` carries every entry verbatim under the release's
  ``### Results that move from <previous>`` and ``### Restart and import
  compatibility`` headings.  ``tests/test_upgrade_notes.py`` asserts
  that the file contains every entry, so a bullet cannot be edited in
  one place and not the other.
* :mod:`woof.whats_changed` appends the same entries to the release's
  upgrade note, which :func:`woof.doctor.upgrade_note` prints once
  after an upgrade and ``woof doctor --since VERSION`` prints on
  demand.

Every entry names the configuration it touches (``km_opt``,
``mp_physics``, ``cu_physics``, ``sf_sfclay_physics``, a ``[tiles]``
key) so a reader can tell in one pass whether their run is affected.
Plain sentences, no dashes as punctuation: the same lines are printed
on a terminal and pasted into release posts.
"""

from __future__ import annotations

#: Release -> the release it is compared against in the heading.
PREVIOUS_RELEASE: dict[str, str] = {
    "2.7.0": "2.6.5",
}

#: Default-on numerical changes: a bare configuration that produced one
#: answer on the previous release produces a different one on this one.
RESULTS_THAT_MOVE: dict[str, tuple[str, ...]] = {
    "2.7.0": (
        "Dry mixing tendencies carry WRF's rk_addtend_dry map-factor "
        "division (1/msf) on mapped grids with km_opt = 2, 3 or 4. Every "
        "shipped real-data configuration selects km_opt = 4, so the u, v, w "
        "and theta mixing rows move by the local map factor and 2.6.5 "
        "mapped-grid bitwise baselines no longer reproduce.",

        "Real-data initialization loads the hydrostatic pressure recurrence "
        "with total water (vapour plus every analyzed condensate species), as "
        "module_initialize_real.F does. The initial pressure moves wherever "
        "the forcing carries hydrometeors: native HRRR, wrfinput, met_em and "
        "any analyzed_species route. Measured worst move 4.34 Pa on the "
        "native HRRR case.",

        "The MM5 surface layer (sf_sfclay_physics = 1 or 91) publishes USTM "
        "unconditionally, so the km_opt = 2 TKE surface drag with a PBL "
        "scheme on reads the real u* instead of zero. That configuration's "
        "TKE budget and mixing coefficients change at first order.",

        "Kain-Fritsch (cu_physics = 1) shallow feedback tendencies divide by "
        "the unrounded 2400 s TIMEC that module_cu_kfeta.F re-sets before "
        "its feedback loop. Whenever dt does not divide 2400 s all six "
        "shallow tendencies move; at dt = 90 s they were 1.23 percent low.",

        "NSSL two-moment (mp_physics = 18): the Bigg rain-freezing graupel "
        "gate is 1e-12 kg/kg (WRF's live override of 1e-7), the fused "
        "ice-nucleation vertical velocity is WRF's single interface average "
        "0.5 (w[k] + w[k+1]), and the low-temperature cnuc term reads the "
        "raw staggered w. Transfers move in the 1e-12 to 1e-7 window and "
        "primary ice nucleation moves wherever w has vertical shear.",

        "Morrison (mp_physics = 10) rebuilds the PGAM reference density from "
        "the current temperature, pres / (287.15 T), instead of the density "
        "frozen at entry. Cloud effective radius moves and RRTMG radiation "
        "moves with it.",

        "Grell-Freitas (cu_physics = 3) keeps the k = kbcon layer in the "
        "undilute CAPE integral, as WRF does. On the measured column aa0 "
        "moved from 0 to 0.0349, which changes where convection triggers.",

        "Grell-Freitas (cu_physics = 3) evaluates gamma with WOOF-ArWen's "
        "correctly rounded float32 implementation; the glibc-derived one is "
        "removed. On the 216-column capture RAINCV and PRATEC move by at "
        "most 7.3 percent, median 1.6 percent.",

        "P3 (mp_physics = 50) spells three integer exponents as "
        "multiplications, as gfortran compiles them, instead of routing "
        "them through powf; results move at the last-bit level.",

        "MYNN surface layer (sf_sfclay_physics = 5): the stable psih table's "
        "1/1.1 exponent is folded in float32 as gfortran folds it, which "
        "moves 268 of the 1001 table words by up to 3 ULP; the table's "
        "pinned SHA-256 changed.",

        "Every non-root domain owns its acoustic carrier ww_pp instead of "
        "sharing a view of the root's, so a child no longer overwrites its "
        "parent's held boundary flux. Multi-domain results change, and "
        "resident GPU memory grows by one (nz+1, ny, nx) float32 field per "
        "non-root domain; a tree that just fitted a card in 2.6.5 can now be "
        "refused.",

        "ERA5 preparation fetches the lake-model mixed-layer temperature and "
        "uses it for lake cells with ice-free lake state, replacing the skin "
        "temperature fallback, so lake surface temperatures move on every "
        "fresh ERA5 case with lakes. A lake cell whose source lake state is "
        "frozen, partially frozen or missing retains the existing "
        "analysed-water or skin-temperature fallback. The preparation "
        "receipt reports the affected cells, including wholly frozen lakes. "
        "A 2.6.5 ERA5 GRIB cache without the lake messages keeps its fallback.",

        "Prepared HRRR initialization plans a soil-texture mesh on the "
        "projected source grid, so soil moisture and deep soil temperature "
        "are downscaled onto domains finer than the 3 km source. A 3 km "
        "domain from HRRR is unchanged; a 1 km domain moves.",

        "A nested child's soil column is built on the configured layer count "
        "(num_soil_layers, resolved per land-surface scheme) instead of the "
        "nine-layer default; six-layer RUC children used to fail forecast "
        "loading.",

        "wrfout files carry sixteen more global attributes (NUM_LAND_CAT, "
        "GMT, JULYR, JULDAY and the twelve *_PATCH_* extents), and "
        "SIMULATION_START_DATE on a delayed-start nest is the run's origin "
        "rather than the domain's own start, so lead times read from that "
        "attribute agree across domains.",

        "[tiles] mode = \"auto\" with any of tile_nx, tile_ny, nbuffers or "
        "halo pinned is refused when the configuration loads. 2.6.5 "
        "accepted and silently ignored the pinned keys (nbuffers = 2 planned "
        "3; a pinned tiling streamed a domain that fits). Pin them under "
        "mode = \"on\", or cap auto with vram_budget_bytes.",

        "`woof go` applies explicit native planner refusals and the "
        "admission budget (free GPU memory minus a 0.5 GiB margin for "
        "other processes). Memory refusals require a measured card. An "
        "unexpected failure in the tile-planning report is announced and "
        "admission uses the resident price; that report error alone does "
        "not refuse a runnable configuration.",

        "Opt-in, diff_6th_opt = 1 or 2: the moisture, number, volume and "
        "TKE rows are normalised on dt/3 and damp three times harder than "
        "the four dry rows, as WRF's rk_scalar_tend does.",

        "Opt-in, feedback = 1: child-to-parent feedback restricts the raw "
        "prognostic fields with theta rebased, without mass weighting, as "
        "WRF does; two-way runs move.",
    ),
}

#: What a reader's existing files can no longer do on this release.
RESTART_AND_IMPORT: dict[str, tuple[str, ...]] = {
    "2.7.0": (
        "A 2.6.5 checkpoint cannot be resumed by 2.7.0. The restart identity "
        "compares the run configuration key by key and 2.7.0 added thirteen "
        "fields (eta_levels and the twelve adaptive-time-step controls), "
        "and the member inventory now requires fields/ustm on every MM5 "
        "surface-layer run and held/gf_rthblten plus held/gf_rqvblten on "
        "every Grell-Freitas and New Tiedtke run. 2.7.0 writes restart "
        "format 6. Format-5 checkpoints, including earlier 2.7.0 previews, "
        "are refused with a compatibility explanation. Start a fresh run "
        "from the original configuration and inputs; checkpoint migration "
        "is not included in this release.",

        "Preparation rebuilds a prepared cache written by 2.6.5 for a "
        "native-HRRR or other condensate-carrying source, because this "
        "release prepares those sources differently; the superseded output "
        "is preserved.",

        "A namelist that omits input_from_file, or gives it a short "
        "per-domain column, keeps WRF's Registry default of .true. for "
        "the omitted entries on import-namelist, run --wrfinput and "
        "run --met-em. Explicit values retain their normal validation.",

        "use_theta_m = 1 in an imported namelist is no longer refused on the "
        "met_em and wrfinput doors. It is emitted as 0 because those "
        "doors construct the dry-theta state themselves. The import "
        "report explicitly identifies this as a reasoned substitution, "
        "without claiming equivalence to WRF's moist-theta integration.",
    ),
}


def results_that_move(release: str) -> tuple[str, ...]:
    """The default-on result changes recorded for ``release`` (may be empty)."""

    return RESULTS_THAT_MOVE.get(release, ())


def restart_and_import(release: str) -> tuple[str, ...]:
    """The compatibility lines recorded for ``release`` (may be empty)."""

    return RESTART_AND_IMPORT.get(release, ())


def results_heading(release: str) -> str:
    """The CHANGELOG heading the results list sits under."""

    return f"Results that move from {PREVIOUS_RELEASE[release]}"


COMPATIBILITY_HEADING = "Restart and import compatibility"


def changelog_sections(release: str) -> str:
    """The two sections exactly as ``CHANGELOG.md`` carries them.

    Used by the test that pins the file to this table and by whoever
    regenerates the section; the CHANGELOG is still edited by hand, this
    just says what the hand must write.
    """

    lines = [f"### {results_heading(release)}", ""]
    lines += [f"- {entry}" for entry in results_that_move(release)]
    lines += ["", f"### {COMPATIBILITY_HEADING}", ""]
    lines += [f"- {entry}" for entry in restart_and_import(release)]
    return "\n".join(lines) + "\n"


def upgrade_note_lines(release: str) -> tuple[str, ...]:
    """The same table, shaped for the doctor's upgrade note.

    One introductory line per block so the terminal reader knows why the
    list is there, then the entries verbatim: the note and the CHANGELOG
    must say the same thing in the same words.
    """

    moved = results_that_move(release)
    compat = restart_and_import(release)
    lines: list[str] = []
    if moved:
        lines.append(f"{results_heading(release)} (a bare configuration "
                     "produces different numbers on this release):")
        lines.extend(moved)
    if compat:
        lines.append(f"{COMPATIBILITY_HEADING}:")
        lines.extend(compat)
    return tuple(lines)


__all__ = ["COMPATIBILITY_HEADING", "PREVIOUS_RELEASE", "RESTART_AND_IMPORT",
           "RESULTS_THAT_MOVE", "changelog_sections", "restart_and_import",
           "results_heading", "results_that_move", "upgrade_note_lines"]
